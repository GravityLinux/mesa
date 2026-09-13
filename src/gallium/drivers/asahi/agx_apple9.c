/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9.h"
#include "asahi/compiler/agx_apple9_ir.h"
#include "asahi/lib/pool.h"
#include "pipe/p_defines.h"
#include "agx_apple9_launch.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/libagx/libagx_dgc.h"
#include "util/compress.h"
#include "util/list.h"
#include "util/u_call_once.h"
#include "util/u_math.h"
#include "agx_device.h"
#include "asahi/layout/layout.h"

static_assert(AGX_APPLE9_RENDER_ENTRY_REGION_SIZE <=
                 AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET,
              "Render entries must not overlap compiler state");
static_assert(AGX_APPLE9_RENDER_COLOR_BUFFER_OFFSET + 0x20 <=
                 AGX_APPLE9_RENDER_COMPILER_STATE_END,
              "Attachment descriptors must fit the relocated state region");

static_assert(AGX_APPLE9_COMPUTE_STATE_LITERAL_STORAGE_CAPACITY *
                    sizeof(uint32_t) <=
                 0x20,
              "Apple9 state literals must fit after state +0x20");
static_assert(AGX_APPLE9_COMPUTE_CODE_SIZE == AGX_APPLE9_COMPUTE_ARCHIVE_SIZE,
              "Gallium and libagx must agree on the compute archive size");

/* Immutable source-generated compute helpers. */
static uint8_t apple9_compute_constant[0x40];
static void apple9_build_sentinel_constant_program(uint8_t *out, unsigned slots);
static util_once_flag apple9_compute_helpers_once = UTIL_ONCE_FLAG_INIT;

static void
apple9_initialize_compute_helpers(void)
{
   apple9_build_sentinel_constant_program(apple9_compute_constant, 30);
}

struct apple9_compute_abi_desc {
   uint8_t helper_slots;
   uint8_t resource_count;
   uint16_t resource_record_size;
   bool has_dynamic_state;
   uint8_t state_uniform_base;
   uint8_t state_literal_capacity;
   uint32_t cdm_config;
   uint32_t cdm_constant;
   uint32_t cdm_tail;
   bool supports_indirect_dispatch;
};

static const uint8_t *
apple9_compute_constant_program(const struct apple9_compute_abi_desc *abi)
{
   return apple9_compute_constant;
}

static size_t
apple9_compute_constant_size(const struct apple9_compute_abi_desc *abi)
{
   return sizeof(apple9_compute_constant);
}

static const struct apple9_compute_abi_desc *
apple9_compute_abi(const struct agx_apple9_compute_profile *profile)
{
   static const struct apple9_compute_abi_desc ssbo8_superset = {
      .helper_slots = 10,
      .resource_count = AGX_APPLE9_COMPUTE_MAX_RESOURCES,
      .resource_record_size = AGX_APPLE9_COMPUTE_SUPERSET_RESOURCE_STRIDE,
      .has_dynamic_state = true,
      .cdm_config = 0x00880000,
      .cdm_constant = 0x01000040,
      .cdm_tail = 0x60000160,
      .supports_indirect_dispatch = true,
   };

   if (!profile)
      return NULL;

   switch (profile->abi) {
   case AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS:
      util_call_once(&apple9_compute_helpers_once,
                     apple9_initialize_compute_helpers);
      return &ssbo8_superset;
   default:
      return NULL;
   }
}

static bool
apple9_compute_profile_valid(const struct agx_apple9_compute_profile *profile,
                             const struct apple9_compute_abi_desc *abi)
{
   if (!profile || !abi ||
       profile->resource_binding_count == 0 ||
       profile->resource_binding_count > abi->resource_count ||
       !agx_apple9_launch_threadgroup_memory_supported(
          profile->required_threadgroup_memory_bytes) ||
       (profile->atomic_frame_size != 0 && profile->atomic_frame_size != 4) ||
       profile->preamble_size > AGX_APPLE9_MAX_PREAMBLE_BYTES ||
       profile->scratch_size > AGX_APPLE9_MAX_SCRATCH_BYTES ||
       (profile->scratch_size & 15))
      return false;

   const unsigned active = profile->resource_binding_count;
   const uint32_t resource_mask = BITFIELD_MASK(active);
   const uint32_t read_mask = profile->resource_read_mask;
   const uint32_t write_mask = profile->resource_write_mask;
   if (!write_mask || ((read_mask | write_mask) & ~resource_mask))
      return false;

   uint64_t local_threads = 1;
   for (unsigned d = 0; d < 3; ++d) {
      if (profile->variable_local_size) {
         if (profile->local_size[d] != 0)
            return false;
      } else {
         if (!profile->local_size[d] ||
             profile->local_size[d] > 1024 / local_threads)
            return false;
         local_threads *= profile->local_size[d];
      }
   }

   for (unsigned i = 0; i < active; ++i) {
      if (profile->resource_kind[i] > AGX_APPLE9_COMPUTE_RESOURCE_UBO ||
          ((write_mask & BITFIELD_BIT(i)) &&
           profile->resource_kind[i] != AGX_APPLE9_COMPUTE_RESOURCE_SSBO))
         return false;
   }

   if (!abi->has_dynamic_state)
      return profile->state_literal_count == 0;

   if (profile->state_literal_count > abi->state_literal_capacity)
      return false;

   for (unsigned i = profile->state_literal_count;
        i < ARRAY_SIZE(profile->state_literals); ++i) {
      if (profile->state_literals[i] != 0)
         return false;
   }

   return true;
}

static inline void
apple9_put_u16(void *ptr, uint16_t value)
{
   memcpy(ptr, &value, sizeof(value));
}

static inline void
apple9_put_u24(void *ptr, uint32_t value)
{
   assert(value <= 0xffffff);
   uint8_t *bytes = ptr;
   bytes[0] = value;
   bytes[1] = value >> 8;
   bytes[2] = value >> 16;
}

static inline void
apple9_put_u32(void *ptr, uint32_t value)
{
   memcpy(ptr, &value, sizeof(value));
}

static inline void
apple9_put_u64(void *ptr, uint64_t value)
{
   memcpy(ptr, &value, sizeof(value));
}

static inline uint32_t
apple9_get_u24(const void *ptr)
{
   const uint8_t *bytes = ptr;
   return bytes[0] | (bytes[1] << 8) | (bytes[2] << 16);
}

static inline uint32_t
apple9_get_u32(const void *ptr)
{
   uint32_t value;
   memcpy(&value, ptr, sizeof(value));
   return value;
}

static inline uint64_t
apple9_get_u64(const void *ptr)
{
   uint64_t value;
   memcpy(&value, ptr, sizeof(value));
   return value;
}

static inline void
apple9_put_f32(void *ptr, float value)
{
   memcpy(ptr, &value, sizeof(value));
}


static bool
apple9_range_fits(size_t mapping_size, uint32_t offset, size_t size)
{
   return offset <= mapping_size && size <= mapping_size - offset;
}

static bool
apple9_ranges_overlap(uint32_t a_offset, size_t a_size, uint32_t b_offset,
                      size_t b_size)
{
   uint64_t a_end = (uint64_t)a_offset + a_size;
   uint64_t b_end = (uint64_t)b_offset + b_size;
   return a_offset < b_end && b_offset < a_end;
}

static void
apple9_fill_helper_table(uint8_t *image, unsigned base, unsigned slots)
{
   assert(slots > 0 && slots <= 10);
   for (unsigned index = 0; index < slots; ++index) {
      uint8_t *record = image + base + (index * 0x10);
      memset(record, 0, 0x10);
      record[0] = 0x0f;
      record[2] = 0x54;
      record[3] = (slots - index) * 0x10;
      record[10] = record[12] = record[14] = 0x06;
   }

   const uint8_t terminal[0x10] = {
      0xf7, 0x03, 0xaa, 0x00, 0x8f, 0x02, 0x54, 0x01,
      0x06, 0x00, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
   };
   memcpy(image + base + slots * 0x10, terminal, sizeof(terminal));
   memcpy(image + base + (slots + 1) * 0x10, terminal, sizeof(terminal));
}

static void
apple9_build_sentinel_constant_program(uint8_t *out, unsigned slots)
{
   assert(slots <= 30);
   memset(out, 0, 0x40);
   apple9_put_u32(out, 0x0e);
   for (unsigned index = 0; index < slots; ++index)
      apple9_put_u16(out + 4 + index * 2, 0x0006);
}

static bool
apple9_compact_pointer_supported(uint64_t usc_exec_base, uint64_t address)
{
   if (address < usc_exec_base)
      return false;

   return ((address - usc_exec_base) >> 13) <= UINT16_MAX;
}

static bool
apple9_patch_compact_pointer(uint8_t *out, unsigned low_byte,
                             unsigned middle_byte, unsigned high_byte,
                             unsigned chunk_byte, uint64_t usc_exec_base,
                             uint64_t address)
{
   if (!apple9_compact_pointer_supported(usc_exec_base, address))
      return false;

   uint64_t relative = address - usc_exec_base;
   uint64_t chunk = relative >> 13;
   uint32_t selector = relative & 0x1fff;
   out[low_byte] = 0x80 | (selector & 0x7f);
   out[middle_byte] = (out[middle_byte] & ~0x1f) | ((selector >> 6) & 0x1e);
   out[high_byte] = (out[high_byte] & ~0x0c) | ((selector >> 9) & 0x0c);
   apple9_put_u16(out + chunk_byte, chunk);
   return true;
}

bool
agx_apple9_compute_state_address_supported(uint64_t usc_exec_base,
                                           uint64_t state_address)
{
   if (!apple9_compact_pointer_supported(usc_exec_base, state_address))
      return false;

   /* Dynamic Caching selects the +0x20 payload half of a 0x40-byte record. */
   return ((state_address - usc_exec_base) &
           (AGX_APPLE9_COMPUTE_STATE_STRIDE - 1)) == 0x20;
}

static bool
apple9_build_compute_launch(uint8_t *out, uint64_t usc_exec_base,
                            uint64_t package_base, uint32_t main_offset,
                            uint64_t state_address,
                            uint32_t resource_table_offset, uint32_t launch_offset,
                            const struct agx_apple9_compute_profile *profile,
                            uint64_t preamble_address)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || !apple9_compute_profile_valid(profile, abi) ||
       main_offset < AGX_APPLE9_RENDER_FIRST_MAIN_OFFSET)
      return false;

   if ((abi->has_dynamic_state && !agx_apple9_compute_state_address_supported(
                                     usc_exec_base, state_address)) ||
       (!abi->has_dynamic_state && state_address != 0))
      return false;

   if (package_base > UINT64_MAX - resource_table_offset)
      return false;

   struct agx_apple9_launch_parameters params = {
      .entry_offset = main_offset,
      .preamble_address = preamble_address,
      .launch_address = package_base + launch_offset,
   };
   params.shader_base = usc_exec_base;
   params.resource_table = package_base + resource_table_offset;
   params.state = state_address;
   params.resource_count = profile->resource_binding_count;
   params.threadgroup_memory_bytes = profile->required_threadgroup_memory_bytes;
   if (profile->scratch_size) {
      params.frame_extent_a = profile->scratch_size;
      params.frame_extent_b = profile->scratch_size;
   }
   params.frame_extent_a = MAX2(params.frame_extent_a, profile->atomic_frame_size);
   return agx_apple9_launch_build(
      out, agx_apple9_compute_launch_size(profile), AGX_APPLE9_LAUNCH_COMPUTE, &params);
}

size_t
agx_apple9_compute_launch_size(const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi ? AGX_APPLE9_COMPUTE_LAUNCH_SIZE : 0;
}

unsigned
agx_apple9_compute_resource_count(
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi && apple9_compute_profile_valid(profile, abi)
             ? profile->resource_binding_count
             : 0;
}

static size_t
apple9_compute_resource_record_size_for_abi(
   const struct apple9_compute_abi_desc *abi)
{
   return abi && abi->resource_record_size
             ? abi->resource_record_size
             : AGX_APPLE9_COMPUTE_RESOURCE_STRIDE;
}

size_t
agx_apple9_compute_resource_record_size(
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi && apple9_compute_profile_valid(profile, abi)
             ? apple9_compute_resource_record_size_for_abi(abi)
             : 0;
}

uint32_t
agx_apple9_compute_required_threadgroup_memory_bytes(
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi && apple9_compute_profile_valid(profile, abi)
             ? profile->required_threadgroup_memory_bytes
             : 0;
}

unsigned
agx_apple9_compute_resource_binding(
   const struct agx_apple9_compute_profile *profile, unsigned argument)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || argument >= profile->resource_binding_count)
      return UINT8_MAX;

   return profile->resource_binding[argument];
}

enum agx_apple9_compute_resource_kind
agx_apple9_compute_resource_kind(
   const struct agx_apple9_compute_profile *profile, unsigned argument)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || argument >= profile->resource_binding_count)
      return AGX_APPLE9_COMPUTE_RESOURCE_SSBO;

   return profile->resource_kind[argument];
}

uint32_t
agx_apple9_compute_read_mask(const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi ? profile->resource_read_mask : 0;
}

uint32_t
agx_apple9_compute_write_mask(const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi ? profile->resource_write_mask : 0;
}

uint32_t
agx_apple9_compute_archive_call_offset(
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi ? agx_apple9_launch_call_offset(AGX_APPLE9_LAUNCH_COMPUTE,
                                              profile->resource_binding_count)
              : 0;
}

bool
agx_apple9_compute_has_dynamic_state(
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi && abi->has_dynamic_state;
}

unsigned
agx_apple9_compute_state_uniform_base(
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi && abi->has_dynamic_state ? abi->state_uniform_base : UINT8_MAX;
}

unsigned
agx_apple9_compute_state_literal_capacity(
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi && abi->has_dynamic_state ? abi->state_literal_capacity : 0;
}

bool
agx_apple9_compute_grid_supported(
   const struct agx_apple9_compute_profile *profile, const uint32_t global[3],
   const uint32_t local[3])
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || !global || !local)
      return false;

   /* CMD-8 and EXP-0092 establish that direct CDM geometry is dispatch state,
    * not part of either launch/archive ABI.  Keep the hardware's public M4
    * threadgroup limit explicit and reject zero axes before encoding. */
   uint64_t threads_per_group = 1;
   for (unsigned d = 0; d < 3; ++d) {
      if (!global[d] || !local[d] ||
          (!profile->variable_local_size &&
           local[d] != profile->local_size[d]))
         return false;
      if (local[d] > 1024 / threads_per_group)
         return false;
      /* Enforce the advertised per-axis group limit. */
      if ((uint64_t)global[d] > (uint64_t)local[d] * 65535)
         return false;
      threads_per_group *= local[d];
   }

   return true;
}

bool
agx_apple9_compute_indirect_dispatch_supported(
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi && abi->supports_indirect_dispatch;
}

static void
apple9_init_compute_archive(uint8_t *code, unsigned helper_slots)
{
   assert(helper_slots > 0 && helper_slots <= 10);
   memset(code, 0, AGX_APPLE9_COMPUTE_CODE_SIZE);
   apple9_put_u32(code, AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE);
   for (unsigned offset = 0x40; offset < AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE;
        offset += 2)
      apple9_put_u16(code + offset, 0x0006);

   /* The archive header is shared, so retain the complete helper directory. */
   apple9_fill_helper_table(code, 0x100, helper_slots);
   apple9_fill_helper_table(code, 0x200, helper_slots);
}


static bool apple9_build_body_entry(uint8_t *out, uint64_t entry,
                                    uint64_t body);

bool
agx_apple9_build_compute_entry(void *mapping, unsigned dispatch,
                               uint64_t shader_base, uint64_t body,
                               const struct agx_apple9_compute_profile *profile,
                               uint32_t *entry_offset)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!mapping || !entry_offset || !abi ||
       dispatch >= AGX_APPLE9_COMPUTE_MAX_ENTRIES ||
       apple9_compute_constant_size(abi) != 0x40)
      return false;
   uint8_t *image = mapping;
   if (!dispatch)
      apple9_init_compute_archive(image, abi->helper_slots);
   uint32_t block = AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE + dispatch * 0xc0;
   uint32_t entry = block + 0x80;
   memset(image + block, 0, 0xc0);
   apple9_put_u32(image + block, 0xc0);
   memcpy(image + block + 0x40, apple9_compute_constant_program(abi), 0x40);
   if (!apple9_build_body_entry(image + entry, shader_base + entry, body))
      return false;
   *entry_offset = entry;
   return true;
}

static bool
apple9_compute_transient_dispatch_fits(
   size_t mapping_size, uint32_t launch_offset, uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   size_t launch_size = agx_apple9_compute_launch_size(profile);
   size_t resource_record_size =
      apple9_compute_resource_record_size_for_abi(abi);
   if (!abi || !launch_size || !apple9_compute_profile_valid(profile, abi) ||
       (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + abi->resource_count) * sizeof(uint64_t) >
          resource_record_size)
      return false;

   const uint32_t resource_start = AGX_APPLE9_COMPUTE_RESOURCE_OFFSET +
                                   AGX_APPLE9_COMPUTE_RESOURCE_TABLE_OFFSET;

   return resource_record_size >= AGX_APPLE9_COMPUTE_RESOURCE_STRIDE &&
          !(resource_record_size & (resource_record_size - 1)) &&
          !(launch_offset & (AGX_APPLE9_COMPUTE_LAUNCH_ALIGN - 1)) &&
          launch_offset >= AGX_APPLE9_COMPUTE_LAUNCH_OFFSET &&
          launch_offset < AGX_APPLE9_COMPUTE_LAUNCH_REGION_END &&
          launch_size <= AGX_APPLE9_COMPUTE_LAUNCH_REGION_END - launch_offset &&
          apple9_range_fits(mapping_size, launch_offset, launch_size) &&
          resource_table_offset >= resource_start &&
          !((resource_table_offset - resource_start) &
            (resource_record_size - 1)) &&
          apple9_range_fits(mapping_size, resource_table_offset,
                            resource_record_size) &&
          !apple9_ranges_overlap(launch_offset, launch_size,
                                 resource_table_offset,
                                 resource_record_size);
}

bool
agx_apple9_compute_dispatch_fits_persistent(
   size_t mapping_size, uint32_t launch_offset, uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile)
{
   return apple9_compute_transient_dispatch_fits(
      mapping_size, launch_offset, resource_table_offset, profile);
}

bool
agx_apple9_compute_dispatch_fits(
   size_t mapping_size, uint32_t launch_offset, uint32_t state_offset,
   uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   size_t resource_record_size =
      apple9_compute_resource_record_size_for_abi(abi);
   if (!abi || !apple9_compute_transient_dispatch_fits(
                  mapping_size, launch_offset, resource_table_offset, profile))
      return false;

   /* Stateless launch ABIs have no state allocation at all.  Requiring the
    * sentinel offset makes this a property of the selected ABI rather than an
    * inference from an otherwise-valid zero-filled state record. */
   if (!abi->has_dynamic_state)
      return state_offset == 0;

   return !(state_offset & (AGX_APPLE9_COMPUTE_STATE_STRIDE - 1)) &&
          state_offset >= AGX_APPLE9_COMPUTE_STATE_OFFSET &&
          state_offset < AGX_APPLE9_COMPUTE_LAUNCH_OFFSET &&
          apple9_range_fits(mapping_size, state_offset,
                            AGX_APPLE9_COMPUTE_STATE_STRIDE) &&
          !apple9_ranges_overlap(
             launch_offset, agx_apple9_compute_launch_size(profile),
             state_offset, AGX_APPLE9_COMPUTE_STATE_STRIDE) &&
          !apple9_ranges_overlap(state_offset, AGX_APPLE9_COMPUTE_STATE_STRIDE,
                                 resource_table_offset,
                                 resource_record_size);
}

bool
agx_apple9_build_compute_state(void *mapping, size_t mapping_size,
                               const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!mapping || !abi || !abi->has_dynamic_state ||
       mapping_size < AGX_APPLE9_COMPUTE_STATE_STRIDE ||
       !apple9_compute_profile_valid(profile, abi))
      return false;

   uint8_t state[AGX_APPLE9_COMPUTE_STATE_STRIDE] = {
      AGX_APPLE9_COMPUTE_STATE_STRIDE,
   };
   memcpy(state + 0x20, profile->state_literals,
          profile->state_literal_count * sizeof(profile->state_literals[0]));
   memcpy(mapping, state, sizeof(state));
   return true;
}

static bool
apple9_build_superset_resource_record(
   uint8_t *package, size_t mapping_size, uint64_t package_base,
   uint32_t resource_table_offset, const struct apple9_compute_abi_desc *abi,
   const uint64_t *resources, unsigned resource_count,
   const struct agx_apple9_compute_geometry *geometry)
{
   const size_t record_size = apple9_compute_resource_record_size_for_abi(abi);
   if (!package || !abi || !resources ||
       !geometry || resource_count == 0 ||
       resource_count > abi->resource_count ||
       record_size < AGX_APPLE9_COMPUTE_SUPERSET_RESOURCE_STRIDE ||
       package_base > UINT64_MAX - resource_table_offset -
                         AGX_APPLE9_COMPUTE_GEOMETRY_GROUPS_OFFSET)
      return false;

   uint8_t *record = package + resource_table_offset;
   uint64_t record_address = package_base + resource_table_offset;

   memset(record, 0, record_size);
   if (!agx_apple9_build_compute_geometry_fields(
          record, record_size, record_address, geometry))
      return false;
   for (unsigned i = 0; i < abi->resource_count; ++i)
      apple9_put_u64(record + (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + i) * sizeof(uint64_t),
                     i < resource_count ? resources[i] : 0);
   /* Remaining record bytes stay zero; geometry lives beyond all pointers. */

   return true;
}

bool
agx_apple9_build_compute_dispatch(
   void *mapping, size_t mapping_size, uint64_t usc_exec_base,
   uint64_t package_base, uint32_t main_offset, uint32_t launch_offset,
   uint32_t state_offset, uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!mapping || !abi || !resources || !geometry ||
       resource_count != profile->resource_binding_count ||
       !!preamble_address != !!profile->preamble_size)
      return false;

   size_t launch_size = agx_apple9_compute_launch_size(profile);
   if (!agx_apple9_compute_dispatch_fits(mapping_size, launch_offset,
                                         state_offset, resource_table_offset,
                                         profile) ||
       (abi->has_dynamic_state &&
        package_base > UINT64_MAX - state_offset - 0x20))
      return false;

   uint8_t state_image[AGX_APPLE9_COMPUTE_STATE_STRIDE];
   if (abi->has_dynamic_state && !agx_apple9_build_compute_state(
                                    state_image, sizeof(state_image), profile))
      return false;

   uint8_t *temporary = malloc(launch_size);
   if (!temporary)
      return false;
   if (!apple9_build_compute_launch(
          temporary, usc_exec_base, package_base, main_offset,
          abi->has_dynamic_state ? package_base + state_offset + 0x20 : 0,
          resource_table_offset, launch_offset, profile, preamble_address)) {
      free(temporary);
      return false;
   }

   uint8_t *package = mapping;
   uint8_t *launch = package + launch_offset;
   if (abi->has_dynamic_state)
      memcpy(package + state_offset, state_image, sizeof(state_image));
   memcpy(launch, temporary, launch_size);
   free(temporary);
   return apple9_build_superset_resource_record(
      package, mapping_size, package_base, resource_table_offset, abi,
      resources, resource_count, geometry);
}

bool
agx_apple9_build_compute_dispatch_persistent(
   void *mapping, size_t mapping_size, uint64_t usc_exec_base,
   uint64_t package_base, uint32_t main_offset, uint32_t launch_offset,
   uint64_t state_address, uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!mapping || !abi || !resources || !geometry ||
       resource_count != profile->resource_binding_count ||
       !!preamble_address != !!profile->preamble_size ||
       (abi->has_dynamic_state && !agx_apple9_compute_state_address_supported(
                                     usc_exec_base, state_address)) ||
       (!abi->has_dynamic_state && state_address != 0) ||
       package_base > UINT64_MAX - resource_table_offset ||
       !apple9_compact_pointer_supported(
          usc_exec_base, package_base + resource_table_offset) ||
       !agx_apple9_compute_dispatch_fits_persistent(
          mapping_size, launch_offset, resource_table_offset, profile))
      return false;

   size_t launch_size = agx_apple9_compute_launch_size(profile);
   uint8_t *temporary = malloc(launch_size);
   if (!temporary)
      return false;
   if (!apple9_build_compute_launch(
          temporary, usc_exec_base, package_base, main_offset, state_address,
          resource_table_offset, launch_offset, profile, preamble_address)) {
      free(temporary);
      return false;
   }

   uint8_t *package = mapping;
   uint8_t *launch = package + launch_offset;
   memcpy(launch, temporary, launch_size);
   free(temporary);
   return apple9_build_superset_resource_record(
      package, mapping_size, package_base, resource_table_offset, abi,
      resources, resource_count, geometry);
}

bool
agx_apple9_compute_enabled(const struct agx_device *dev)
{
   /* The current launch wrappers, helper directory, resource ordering, and
    * CDM constants are all exact T8132 captures.  Apple9 ISA support may be
    * shared with G17P, but that does not prove its userspace package ABI. */
   return dev->chip == AGX_CHIP_G16G;
}

bool
agx_apple9_emit_direct_dispatch(
   void *out, uint64_t launch, const uint32_t global[3],
   const uint32_t local[3], const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!out || !abi || (launch & 0x3f) ||
       !apple9_compute_profile_valid(profile, abi) ||
       !agx_apple9_compute_grid_supported(profile, global, local))
      return false;

   uint8_t *record = out;
   uint64_t shader = ((launch >> 6) & 0xffffffffull) |
                     ((0x40000000ull | (launch >> 40)) << 32);
   apple9_put_u32(record + 0x00, abi->cdm_config);
   apple9_put_u32(record + 0x04, abi->cdm_constant);
   apple9_put_u64(record + 0x08, shader);
   for (unsigned i = 0; i < 3; ++i) {
      apple9_put_u32(record + 0x10 + (i * 4), global[i]);
      apple9_put_u32(record + 0x1c + (i * 4), local[i]);
   }
   apple9_put_u32(record + 0x28, abi->cdm_tail);
   return true;
}

bool
agx_apple9_emit_indirect_dispatch(
   void *out, uint64_t launch, uint64_t indirect, const uint32_t local[3],
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!out || !abi || !abi->supports_indirect_dispatch || !indirect ||
       (launch & 0x3f) || (indirect & 3) ||
       !apple9_compute_profile_valid(profile, abi))
      return false;

   uint64_t threads = 1;
   for (unsigned d = 0; d < 3; ++d) {
      if (!local[d] ||
          (!profile->variable_local_size &&
           local[d] != profile->local_size[d]) ||
          local[d] > 1024 / threads)
         return false;
      threads *= local[d];
   }

   uint8_t *record = out;
   uint64_t shader = ((launch >> 6) & 0xffffffffull) |
                     ((0x40000000ull | (launch >> 40)) << 32);
   apple9_put_u32(record + 0x00, abi->cdm_config | 0x08000000);
   apple9_put_u32(record + 0x04, abi->cdm_constant);
   apple9_put_u64(record + 0x08, shader);
   /* Native Apple9 indirect CDM stores the pointer halves high then low. */
   apple9_put_u32(record + 0x10, indirect >> 32);
   apple9_put_u32(record + 0x14, indirect);
   for (unsigned d = 0; d < 3; ++d)
      apple9_put_u32(record + 0x18 + d * 4, local[d]);
   apple9_put_u32(record + 0x24, abi->cdm_tail);
   return true;
}

void
agx_apple9_pack_r32f_texture(void *out, uint64_t address, uint32_t width,
                             uint32_t height, uint32_t stride_B)
{
   assert((address & 0xf) == 0);
   assert(width > 0 && width <= 0x4000 && height > 0 && height <= 0x4000);
   /* The linear R32F profile has an implicit tightly-packed row stride. */
   assert(stride_B == width * 4);
   uint32_t w = width - 1, h = height - 1;
   uint64_t units = address >> 4;
   uint32_t words[8] = {
      0x09688862 | ((w & 0xf) << 28),
      ((w >> 4) & 0x3ff) | (h << 10),
      units,
      (units >> 32) & 0xfff,
   };
   memcpy(out, words, sizeof(words));
}

bool
agx_apple9_texture_format_supported(enum pipe_format format)
{
   if (agx_apple9_color_is_wide(format) || agx_apple9_color_is_packed(format))
      return true;
   switch (format) {
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R8G8B8X8_SNORM:
   case PIPE_FORMAT_R9G9B9E5_FLOAT:
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_R8G8B8X8_SRGB:
   case PIPE_FORMAT_B8G8R8A8_SRGB:
   case PIPE_FORMAT_B8G8R8X8_SRGB:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_Z32_FLOAT:
   /* The transfer helper stores these as depth32f and separate stencil;
    * sampling the logical combined format reads its depth component. */
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_Z16_UNORM:
      return true;
   default:
      return false;
   }
}

bool
agx_apple9_sampler_wrap_supported(unsigned wrap)
{
   return wrap == PIPE_TEX_WRAP_CLAMP_TO_EDGE || wrap == PIPE_TEX_WRAP_REPEAT ||
          wrap == PIPE_TEX_WRAP_MIRROR_REPEAT || wrap == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
          wrap == PIPE_TEX_WRAP_CLAMP;
}

static unsigned
apple9_sampler_address(unsigned wrap)
{
   /* EXP-M4-08: native descriptor address codes, independently captured for
    * each axis. Wrapping in hardware preserves filtering across seams. */
   switch (wrap) {
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE: return 0;
   case PIPE_TEX_WRAP_REPEAT: return 1;
   case PIPE_TEX_WRAP_MIRROR_REPEAT: return 2;
   case PIPE_TEX_WRAP_CLAMP:
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER: return 3;
   default: UNREACHABLE("unsupported Apple9 sampler address mode");
   }
}

void
agx_apple9_pack_sampler(void *out, bool min_linear, bool mag_linear,
                         unsigned mip_filter, float min_lod, float max_lod,
                         unsigned wrap_s, unsigned wrap_t, unsigned max_anisotropy)
{
   assert(mip_filter <= 2);
   unsigned anisotropy = util_logbase2(util_next_power_of_two(
      CLAMP(max_anisotropy, 1, 16)));
   max_lod = CLAMP(max_lod, 0.0f, 14.0f);
   min_lod = CLAMP(min_lod, 0.0f, max_lod);
   const uint32_t words[2] = {
      (uint32_t)roundf(min_lod * 64) | ((uint32_t)roundf(max_lod * 8) << 13) |
      (anisotropy << 20) | (mag_linear << 23) | (min_linear << 25) | (mip_filter << 27) |
      (apple9_sampler_address(wrap_s) << 29),
      apple9_sampler_address(wrap_t) | (1u << 7) | (7u << 8),
   };
   memcpy(out, words, sizeof(words));
}

void
agx_apple9_pack_nearest_sampler(void *out)
{
   agx_apple9_pack_sampler(out, false, false, 0, 0, 14,
                            PIPE_TEX_WRAP_CLAMP_TO_EDGE, PIPE_TEX_WRAP_CLAMP_TO_EDGE, 1);
}

/* Native Apple9 render-context aperture used by the source-built graph. */
#define AGX_APPLE9_DRAW_STATE         0x1000048000ull
#define AGX_APPLE9_DIRECT_STREAM_SIZE 0x78

#define AGX_APPLE9_BIND0_OFFSET      0x00000u
#define AGX_APPLE9_DRAW_STATE_OFFSET 0x44000u
#define AGX_APPLE9_BIND_GROUP_OFFSET 0x54000u
#define AGX_APPLE9_VIEWPORT_OFFSET   0x64000u

static void
apple9_build_direct_bind0(uint8_t *page, unsigned varying_components)
{
   memset(page, 0, 0x4000);
   for (unsigned index = 0; index < 7; ++index) {
      unsigned base = index * 0x80;
      apple9_put_u32(page + base, 0x00000080);
      apple9_put_u32(page + base + 0x40, 0x10040000);
   }

   /* Public G17 direct-state serializer. */
   apple9_put_u32(page + 0x300, 0x0000fcc0);
   /* EXP-M4-59: user scalar count, excluding the four position words. */
   apple9_put_u32(page + 0x44, varying_components);
}

/* The default coefficient table occupies the end of the graphics resource
 * page, after per-draw argument records. Draw-specific tables live in their
 * own slots on the following pages so mixed interpolation can use one
 * descriptor per scalar without overwriting the next draw's roots. */
#define APPLE9_CF_BINDINGS 0x203d00u
#define APPLE9_CF_BINDINGS_SIZE 0x220u
static_assert(4 + 4 * (AGX_APPLE9_MAX_VARYING_COMPONENTS + 3) <=
                 APPLE9_CF_BINDINGS_SIZE,
              "coefficient table must fit one binding per scalar plus 1/W and Z");

static unsigned
apple9_cf_binding_count(unsigned components)
{
   return 1 + DIV_ROUND_UP(components, 4);
}

static unsigned
apple9_build_cf_bindings(uint8_t *table, unsigned components,
                         struct agx_apple9_interp_mask linear_mask,
                         struct agx_apple9_interp_mask flat_mask,
                         bool flatshade_first, bool reads_z, bool reads_point_coord)
{
   assert(components <= AGX_APPLE9_MAX_VARYING_COMPONENTS);
   memset(table, 0, APPLE9_CF_BINDINGS_SIZE);
   unsigned slots = components + 1 + reads_z + 2 * reads_point_coord;
   apple9_put_u32(table, slots | (slots << 8));
   /* Coefficient zero is 1/W. Keep stable scalar indices across all modes. */
   apple9_put_u32(table + 4, 0x0c);
   unsigned binding = 1;
   for (unsigned start = 0; start < components; ++binding) {
      unsigned shade = agx_apple9_interp_mask_test(flat_mask, start)
                          ? (flatshade_first ? 0 : 2)
                       : agx_apple9_interp_mask_test(linear_mask, start) ? 3
                                                             : 7;
      unsigned count = 1;
      while (count < 4 && start + count < components) {
         unsigned next = start + count;
         unsigned next_shade = agx_apple9_interp_mask_test(flat_mask, next)
                                  ? (flatshade_first ? 0 : 2)
                               : agx_apple9_interp_mask_test(linear_mask, next) ? 3
                                                                    : 7;
         if (next_shade != shade)
            break;
         ++count;
      }
      unsigned base = start + 1;
      apple9_put_u32(table + 4 + binding * 4,
                     (count - 1) | (shade << 2) | ((base + reads_z) << 8) | (base << 16));
      start += count;
   }
   if (reads_z) {
      /* Public CF source=FRAGCOORD_Z, linear, source slot one. Keep user
       * coefficient indices stable and append the depth coefficient. */
      apple9_put_u32(table + 4 + binding++ * 4,
                     0x12c | ((components + 1) << 16));
   }
   if (reads_point_coord) {
      /* Two linear coefficients generated by the point rasterizer. */
      apple9_put_u32(table + 4 + binding++ * 4,
                     0x4d | ((components + 1 + reads_z) << 16));
   }
   return binding;
}

static void
apple9_build_direct_bind_group(uint8_t *page, unsigned varying_components)
{
   static const struct {
      uint16_t offset;
      uint32_t value;
   } common[] = {
      {0x00, 0x00800000}, {0x04, 0x00010100}, {0x08, 0x0000c9c0},
      {0x10, 0x01000000}, {0x14, 0x00066420}, {0x1c, 0x0c0a0000},
      {0x20, 0x00010000}, {0x2c, 0x00000006}, {0x30, 0x010000b4},
      {0x34, 0x00040200}, {0x38, 0x07200f00}, {0x3c, 0x0e000000},
      {0x40, 0x07200f00}, {0x44, 0x0e000000}, {0x4c, 0x02000048},
      {0x50, 0x00000200}, {0x54, 0x07e00000}, {0x58, 0x07e00000},
      {0x5c, 0x0000000f}, {0x60, 0x00410000}, {0x68, 0x00000080},
      {0x6c, 0x00200000}, {0x70, 0x00000480},
   };

   memset(page, 0, 0x80);
   for (unsigned i = 0; i < ARRAY_SIZE(common); ++i)
      apple9_put_u32(page + common[i].offset, common[i].value);

   /* T8140 direct-render deltas reused by Apple9. */
   apple9_put_u32(page + 0x04, 0);
   apple9_put_u32(page + 0x08, 0);
   apple9_put_u32(page + 0x14, 0x00004e19);
   apple9_put_u32(page + 0x20, 0);
   apple9_put_u32(page + 0x2c, 4);
   apple9_put_u32(page + 0x5c, 0x0001ffff);

   /* Small G16 packing deltas established independently on T8132. */
   page[0x05] = 0x01;
   page[0x06] = 0x02;
   page[0x08] = 0x80;
   page[0x09] = 0x04;
   page[0x15] = 0x8c;
   page[0x22] = 0x01;
   assert(varying_components <= AGX_APPLE9_MAX_VARYING_COMPONENTS);
   page[0x2c] = 4 + varying_components;
   /* G16's fragment-state layout moves the coefficient pointer ahead of
    * the pipeline words. EXP-M4-59 confirms the public binding descriptors. */
   apple9_put_u32(page + 0x04,
                  0x100 | (apple9_cf_binding_count(varying_components) << 16));
   apple9_put_u32(page + 0x08, APPLE9_CF_BINDINGS);
   apple9_put_u32(page + 0x18, varying_components / 8);
}

static void
apple9_build_viewport(uint8_t *page, unsigned width, unsigned height)
{
   unsigned tiles_x = DIV_ROUND_UP(width, 32);
   unsigned tiles_y = DIV_ROUND_UP(height, 32);
   memset(page, 0, 0x4000);
   apple9_put_u32(page + 0x900, 0x00000c00);
   apple9_put_u32(page + 0x904, 0x80000000 | (tiles_x - 1));
   apple9_put_u32(page + 0x908, tiles_y - 1);
   apple9_put_f32(page + 0x910, width / 2.0f);
   apple9_put_f32(page + 0x914, width / 2.0f);
   apple9_put_f32(page + 0x918, height / 2.0f);
   apple9_put_f32(page + 0x91c, -(height / 2.0f));
   apple9_put_f32(page + 0x924, 1.0f);
}

static bool
agx_apple9_build_render_state_image_for_varyings(void *mapping,
                                                 size_t mapping_size,
                                                 unsigned width,
                                                 unsigned height,
                                                 unsigned varying_components)
{
   if (!mapping || mapping_size < AGX_APPLE9_RENDER_STATE_SIZE || !width ||
       width > 0x4000 || !height || height > 0x4000 ||
       varying_components > AGX_APPLE9_MAX_VARYING_COMPONENTS)
      return false;

   uint8_t *state = mapping;
   memset(state, 0, AGX_APPLE9_RENDER_STATE_SIZE);
   apple9_build_direct_bind0(state + AGX_APPLE9_BIND0_OFFSET, varying_components);
   apple9_put_u32(state + AGX_APPLE9_DRAW_STATE_OFFSET, 0x00000100);
   apple9_build_direct_bind_group(state + AGX_APPLE9_BIND_GROUP_OFFSET,
                                  varying_components);
   apple9_build_viewport(state + AGX_APPLE9_VIEWPORT_OFFSET, width, height);
   return true;
}

struct apple9_render_self_relocation {
   uint32_t offset;
   uint32_t relative;
};

/* Absolute self-pointers in the render-target state graph.
 * Unlike archive calls and compact instruction operands, these name their
 * containing records directly. Attachment records are published into one
 * device-owned graph, independent of compiled shader objects. */
static const struct apple9_render_self_relocation
   apple9_render_absolute_self_relocations[] = {
      {0x210000, 0x210020}, {0x210008, 0x210120}, {0x210160, 0x210168},
      {0x210300, 0x210320}, {0x210308, 0x210420}, {0x210460, 0x210468},
      {0x210600, 0x210620}, {0x210820, 0x210828},
};

/* A batch shares entries between draws targeting the same shader and stage.
 * The caller waits for prior fixed-USC users before publishing another batch's
 * table. Each slot has a header, reserved constant area and generated transfer. */
static_assert(AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE +
                    AGX_APPLE9_RENDER_MAX_ENTRIES *
                       AGX_APPLE9_RENDER_ENTRY_BLOCK_SIZE <=
                 AGX_APPLE9_RENDER_ENTRY_REGION_SIZE,
              "Shader entries must fit the compact entry region");

static void
apple9_init_render_entries(uint8_t *image)
{
   apple9_put_u32(image, AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE);
   for (unsigned at = 0x40; at < AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE; at += 2)
      apple9_put_u16(image + at, 6);
   apple9_fill_helper_table(image, 0x100, 10);
   apple9_fill_helper_table(image, 0x200, 10);
   for (unsigned slot = 0; slot < AGX_APPLE9_RENDER_MAX_ENTRIES; slot++) {
      unsigned entry = agx_apple9_entry_offset(slot);
      apple9_put_u32(image + entry - AGX_APPLE9_RENDER_ENTRY_CODE_OFFSET,
                     AGX_APPLE9_RENDER_ENTRY_BLOCK_SIZE);
      apple9_put_u32(image + entry, 0x0e);
   }
}

static bool
apple9_build_body_entry(uint8_t *out, uint64_t entry, uint64_t body)
{
   int64_t displacement = (int64_t)body - (int64_t)entry;
   /* Both allocations live within the same 4-GiB USC heap. Wider virtual
    * placement is outside the current entry ABI's validated domain. */
   if ((entry | body) & 1 || displacement < -(int64_t)UINT32_MAX ||
       displacement > UINT32_MAX)
      return false;
   struct agx_apple9_packed_instruction packed;
   if (!agx_apple9_pack_branch(true, displacement, &packed))
      return false;
   memcpy(out, packed.bytes, packed.length);
   apple9_put_u32(out + packed.length, 0x0e); /* No active lanes: terminate. */
   return true;
}

static void
apple9_patch_render_target(uint8_t *package, unsigned offset, bool texture,
                           uint64_t target, unsigned width, unsigned height)
{
   uint32_t words[8];
   memcpy(words, package + offset, sizeof(words));
   unsigned width_minus_one = width - 1;
   unsigned height_minus_one = height - 1;

   /* Reuse the public G17 raw-twiddled descriptor field split.  The G16
    * package differs in its fixed format/class bits, which are preserved. */
   if (texture) {
      words[0] = (words[0] & 0x0fffffff) | ((width_minus_one & 0xf) << 28);
      words[1] = (words[1] & ~0x00ffffff) | ((width_minus_one >> 4) & 0x3ff) |
                 ((height_minus_one & 0x3fff) << 10);
   } else {
      words[0] = (words[0] & 0x00ffffff) | ((width_minus_one & 0xff) << 24);
      words[1] = (words[1] & ~0x000fffff) | ((width_minus_one >> 8) & 0x3f) |
                 ((height_minus_one & 0x3fff) << 6);
   }

   uint64_t encoded = target >> 4;
   words[1] &= ~(1u << 27);
   words[2] = encoded;
   words[3] = (words[3] & ~((1u << 31) | 0xfff)) | ((encoded >> 32) & 0xfff);
   words[4] = 0;
   words[5] &= ~0xfff;
   memcpy(package + offset, words, sizeof(words));
}

/* The fixed header/short entry aperture is device-owned. Everything pointed
 * to by a draw is immutable, batch-owned storage. This object never caches or
 * diffs complete render images. */
struct agx_apple9_graphics {
   struct agx_device *dev;
   struct agx_va *logical_va;
   struct agx_bo *resident;
   bool initialized;
   struct agx_apple9_framebuffer framebuffer;
   unsigned varying_components;
   uint64_t entries[AGX_APPLE9_RENDER_MAX_ENTRIES];
};

struct agx_apple9_graphics *
agx_apple9_graphics_create(struct agx_device *dev)
{
   if (!dev || dev->shader_base != UINT64_C(0x10000000000))
      return NULL;
   struct agx_apple9_graphics *g = calloc(1, sizeof(*g));
   if (!g)
      return NULL;
   g->dev = dev;
   uint64_t logical = dev->shader_base + AGX_APPLE9_RENDER_HEADER_OFFSET;
   g->logical_va = agx_va_alloc(dev, AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE,
                                AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE,
                                AGX_VA_USC | AGX_VA_FIXED, logical);
   if (!g->logical_va) {
      free(g);
      return NULL;
   }
   g->resident = agx_bo_create(dev, AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE,
                               AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE,
                               AGX_BO_WRITEBACK, "Apple9 graphics header");
   if (!g->resident ||
       agx_bo_bind(dev, g->resident, logical,
                   AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE, 0,
                   DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE)) {
      agx_bo_unreference(dev, g->resident);
      agx_va_free(dev, g->logical_va, false);
      free(g);
      return NULL;
   }
   return g;
}

void
agx_apple9_graphics_destroy(struct agx_apple9_graphics *g)
{
   if (!g)
      return;
   agx_bo_bind(g->dev, NULL, g->logical_va->addr,
               AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE, 0,
               DRM_ASAHI_BIND_UNBIND);
   agx_bo_unreference(g->dev, g->resident);
   agx_va_free(g->dev, g->logical_va, false);
   free(g);
}

struct agx_bo *
agx_apple9_graphics_bo(struct agx_apple9_graphics *g)
{
   return g->resident;
}

void
agx_apple9_graphics_invalidate(struct agx_apple9_graphics *g)
{
   if (g)
      g->initialized = false;
}

/* Initialize invariant header/context records once. The old package builder
 * repeated this for every framebuffer/shader combination. No draw allocation
 * or incremental publication scans these apertures. */
static void
apple9_graphics_initialize(struct agx_apple9_graphics *g)
{
   struct agx_device *dev = g->dev;
   uint8_t *resident = agx_bo_map(g->resident);
   uint8_t *fixed = agx_bo_map(dev->apple9_render_fixed_usc);
   memset(resident, 0, AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE);
   apple9_init_render_entries(resident);
   apple9_build_cf_bindings(
      resident + APPLE9_CF_BINDINGS, 0, (struct agx_apple9_interp_mask){0},
      (struct agx_apple9_interp_mask){0}, false, false, false);
   memcpy(fixed, resident, AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE);
   uint8_t *context = agx_bo_map(dev->apple9_render_context);
   memset(context, 0, AGX_APPLE9_RENDER_STATE_SIZE + 0x4000);
   apple9_build_direct_bind0(context, 0);
   memset(context + 0x340, 0, 8);
   agx_apple9_build_render_state_image_for_varyings(
      context + 0x4000, AGX_APPLE9_RENDER_STATE_SIZE, 1, 1, 0);
   agx_bo_note_cpu_write(g->resident, 0,
                         AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE);
   agx_bo_note_cpu_write(dev->apple9_render_fixed_usc, 0,
                         AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE);
   agx_bo_note_cpu_write(dev->apple9_render_context, 0,
                         AGX_APPLE9_RENDER_STATE_SIZE + 0x4000);
   memset(g->entries, 0, sizeof(g->entries));
   g->framebuffer.count = 0;
   g->varying_components = 0;
   g->initialized = true;
}

static bool
apple9_framebuffer_equal(const struct agx_apple9_framebuffer *a,
                         const struct agx_apple9_framebuffer *b)
{
   if (a->width != b->width || a->height != b->height || a->count != b->count ||
       a->samples != b->samples)
      return false;
   for (unsigned rt = 0; rt < a->count; rt++) {
      if (a->targets[rt] != b->targets[rt] || a->formats[rt] != b->formats[rt])
         return false;
   }
   return true;
}

/* Complete attachment records, including inactive slots. Defaults/disabled
 * fields are explicit so shrinking an MRT set cannot retain old state. */
struct apple9_attachment_records {
   uint8_t texture[0x100];
   uint8_t buffer[0x100];
   uint8_t graph[0xa00];
};

static bool
apple9_build_attachments(struct apple9_attachment_records *records,
                         const struct agx_apple9_framebuffer *fb)
{
   unsigned count = fb->count;
   if (count > 8 || !fb->width || !fb->height || fb->width > 16384 ||
       fb->height > 16384 ||
       (fb->samples != 1 && fb->samples != 2 && fb->samples != 4))
      return false;
   memset(records, 0, sizeof(*records));
   unsigned offsets[8], tile_bytes = 0, mask = 0;
   for (unsigned rt = 0; rt < count; rt++) {
      if (fb->targets[rt] & 15)
         return false;
      offsets[rt] = tile_bytes;
      tile_bytes += 4 * agx_apple9_color_words(fb->formats[rt]);
      if (fb->targets[rt])
         mask |= BITFIELD_BIT(rt);
   }
   for (unsigned kind = 0; kind < 2; kind++) {
      bool texture = kind == 0;
      uint8_t *image = texture ? records->texture : records->buffer;
      for (unsigned rt = 0; rt < count; rt++) {
         if (!(mask & BITFIELD_BIT(rt)))
            continue;
         unsigned offset = 32 * rt;
         uint32_t format;
         switch (fb->formats[rt]) {
         case PIPE_FORMAT_R8_UNORM:
            format = texture ? 0x09680000 : 0x000000;
            break;
         case PIPE_FORMAT_R8G8_UNORM:
            format = texture ? 0x09480280 : 0x040280;
            break;
         case PIPE_FORMAT_R8G8B8A8_UNORM:
         case PIPE_FORMAT_R8G8B8X8_UNORM:
            format = texture ? 0x06880a00 : 0xe40a00;
            break;
         case PIPE_FORMAT_R16_FLOAT:
            format = texture ? 0x09688240 : 0x008240;
            break;
         case PIPE_FORMAT_R16G16_FLOAT:
            format = texture ? 0x094888c0 : 0x0488c0;
            break;
         case PIPE_FORMAT_R16G16B16A16_FLOAT:
            format = texture ? 0x06888c80 : 0xe48c80;
            break;
         default:
            format = texture ? 0x060a0a00 : 0xc60a00;
            break;
         }
         if (agx_apple9_color_is_wide(fb->formats[rt]) ||
             agx_apple9_color_is_packed(fb->formats[rt])) {
            const struct ail_pixel_format_entry *entry =
               &ail_pixel_format[fb->formats[rt]];
            unsigned components = agx_apple9_color_components(fb->formats[rt]);
            unsigned swizzle = texture ? (components == 1 ? 0x09680000 :
                                          components == 2 ? 0x09480000 : 0x06880000)
                                       : (components == 1 ? 0 : components == 2 ? 0x040000 : 0xe40000);
            const struct util_format_description *desc =
               util_format_description(fb->formats[rt]);
            if (desc->swizzle[0] == PIPE_SWIZZLE_Z)
               swizzle = texture ? ((swizzle & ~0xffc0000u) | 0x060a0000u) : 0xc60000;
            format = swizzle | (entry->channels << 6) | (entry->type << 13);
         }
         apple9_put_u32(image + offset, format);
         apple9_patch_render_target(image, offset, texture, fb->targets[rt],
                                    fb->width, fb->height);
         if (fb->samples > 1) {
            apple9_put_u32(image + offset,
               (apple9_get_u32(image + offset) & ~0xfu) | 4);
            apple9_put_u32(image + offset + 4,
                           (apple9_get_u32(image + offset + 4) & ~0x09000000u) |
                              (fb->samples == 4 ? 0x01000000u : 0));
         }
      }
   }
   uint8_t *graph = records->graph;
   const uint64_t address = UINT64_C(0x10000000000) +
                            AGX_APPLE9_RENDER_HEADER_OFFSET +
                            AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET;
   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_absolute_self_relocations);
        i++) {
      const struct apple9_render_self_relocation *r =
         &apple9_render_absolute_self_relocations[i];
      apple9_put_u64(
         graph + r->offset - AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET,
         address + r->relative - AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET);
   }
   memcpy(graph + 0x20, records->texture, sizeof(records->texture));
   memcpy(graph + 0x320, records->texture, sizeof(records->texture));
   memcpy(graph + 0x620, records->buffer, sizeof(records->buffer));
   unsigned actions = 0;
   for (unsigned rt = 0; rt < count; rt++) {
      if (mask & BITFIELD_BIT(rt))
         actions |= 2u << (2 * rt);
   }
   for (unsigned copy = 0; copy < 2; copy++) {
      uint8_t *view = graph + 0x300 * copy;
      apple9_put_u32(view + 0x168, actions | (copy ? mask << 24 : 0));
      apple9_put_u32(view + 0x16c, copy ? 0 : mask);
      for (unsigned rt = 0; rt < count; rt++) {
         unsigned format =
            (agx_apple9_color_components(fb->formats[rt]) << 5) |
            (agx_apple9_color_is_half(fb->formats[rt]) ? 0xc : 0x3);
         apple9_put_u32(view + 0x2d0 + 4 * rt,
                        0x100000 | (offsets[rt] << 12) | format);
      }
      apple9_put_u32(view + 0x2f0, 0x400000 | (ALIGN_POT(tile_bytes, 8) << 12) |
                                      (util_logbase2(fb->samples) * 0x500) |
                                      mask);
   }
   for (unsigned rt = 0; rt < count; rt++) {
      unsigned type = agx_apple9_color_is_half(fb->formats[rt]) ? 0xc : 0x3;
      apple9_put_u32(graph + 0x82c + 4 * rt,
                     (agx_apple9_color_components(fb->formats[rt]) << 26) |
                        (fb->samples > 1 ? 0x40000 : 0x20000) | (type << 8));
   }
   apple9_put_u32(graph + 0x874, mask);
   apple9_put_u32(graph + 0x878,
                  0x400000 | fb->samples | (ALIGN_POT(tile_bytes, 8) << 4));
   for (unsigned rt = 0; rt < count; rt++)
      apple9_put_u32(graph + 0x87c + 4 * rt, offsets[rt]);
   if (fb->samples > 1) {
      apple9_put_u64(graph + 0x8c0, address + 0x8e0);
      memset(graph + 0x8c8, 0xff, 24);
      memcpy(graph + 0x8e0, records->buffer, 32 * count);
   }
   return true;
}

bool
agx_apple9_graphics_publish(struct agx_apple9_graphics *g,
                            const struct agx_apple9_framebuffer *fb,
                            const struct agx_apple9_entry_table *entries,
                            unsigned varying_components,
                            const float clear_color[8][4])
{
   if (!g || !entries || !entries->count ||
       entries->count > AGX_APPLE9_RENDER_MAX_ENTRIES)
      return false;
   if (!g->initialized)
      apple9_graphics_initialize(g);
   struct agx_bo *views[] = {g->resident, g->dev->apple9_render_fixed_usc};
   if (!apple9_framebuffer_equal(&g->framebuffer, fb)) {
      struct apple9_attachment_records records;
      if (!apple9_build_attachments(&records, fb))
         return false;
      for (unsigned v = 0; v < ARRAY_SIZE(views); v++) {
         uint8_t *dst = agx_bo_map(views[v]);
         memcpy(dst + AGX_APPLE9_RENDER_COLOR_TEXTURE_OFFSET, records.texture,
                sizeof(records.texture));
         memcpy(dst + AGX_APPLE9_RENDER_COLOR_BUFFER_OFFSET, records.buffer,
                sizeof(records.buffer));
         memcpy(dst + AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET,
                records.graph, sizeof(records.graph));
         agx_bo_note_cpu_write(views[v], AGX_APPLE9_RENDER_COLOR_TEXTURE_OFFSET,
                               sizeof(records.texture));
         agx_bo_note_cpu_write(views[v], AGX_APPLE9_RENDER_COLOR_BUFFER_OFFSET,
                               sizeof(records.buffer));
         agx_bo_note_cpu_write(views[v],
                               AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET,
                               sizeof(records.graph));
      }
      g->framebuffer = *fb;
   }
   if (varying_components != g->varying_components) {
      uint8_t *context = agx_bo_map(g->dev->apple9_render_context);
      apple9_put_u32(context + 0x44, varying_components);
      apple9_put_u32(context + 0x4044, varying_components);
      agx_bo_note_cpu_write(g->dev->apple9_render_context, 0x44, 4);
      agx_bo_note_cpu_write(g->dev->apple9_render_context, 0x4044, 4);
      g->varying_components = varying_components;
   }
   for (unsigned i = 0; i < entries->count; i++) {
      uint64_t code = entries->entries[i].code;
      if (g->entries[i] == code)
         continue;
      unsigned entry = agx_apple9_entry_offset(i);
      uint8_t body_entry[32] = {0};
      if (!apple9_build_body_entry(body_entry, g->dev->shader_base + entry, code))
         return false;
      for (unsigned v = 0; v < ARRAY_SIZE(views); v++) {
         memcpy((uint8_t *)agx_bo_map(views[v]) + entry, body_entry,
                sizeof(body_entry));
         agx_bo_note_cpu_write(views[v], entry, sizeof(body_entry));
      }
      g->entries[i] = code;
   }
   if (clear_color) {
      for (unsigned copy = 0; copy < 2; copy++) {
         unsigned off = 0x170 + copy * 0x300;
         memcpy((uint8_t *)agx_bo_map(g->resident) +
                   AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET + off,
                clear_color, 16 * fb->count);
         memcpy((uint8_t *)agx_bo_map(g->dev->apple9_render_fixed_usc) +
                   AGX_APPLE9_RENDER_FIXED_TARGET_GRAPH_OFFSET + off,
                clear_color, 16 * fb->count);
      }
   }
   return agx_apple9_install_render_archive(g->dev);
}

/* USC records are compact, immutable allocations. A short entry number still
 * selects the body thunk, but roots/launches/coefficient tables do not live in
 * fixed slots and are never overwritten by subsequent batches. */
bool
agx_apple9_prepare_draw(struct agx_device *dev, struct agx_pool *usc_pool,
                        struct agx_pool *context_pool,
                        struct agx_apple9_entry_table *entries,
                        const struct agx_apple9_render_pipeline *pipeline,
                        struct agx_apple9_uniform_draw *draw,
                        const struct agx_apple9_uniform_draw *previous)
{
   const struct agx_apple9_render_stage *stages[] = {&pipeline->vertex,
                                                     &pipeline->fragment};
   unsigned tile_bytes = 0;
   for (unsigned rt = 0; rt < pipeline->fragment.render_targets; rt++)
      tile_bytes += 4 * agx_apple9_color_words(pipeline->color_formats[rt]);
   unsigned samples = MAX2(pipeline->samples, 1);
   if (tile_bytes > (samples == 1 ? 128 : 64))
      return false;
   draw->samples = samples;
   draw->tile_bytes = tile_bytes;
   for (unsigned stage = 0; stage < 2; stage++) {
      const struct agx_apple9_render_stage *shader = stages[stage];
      if (!shader->bo || !shader->publication_count_valid)
         return false;
      draw->program_id[stage] = shader->program_id;
      draw->code[stage] = shader->bo->va->addr;
   }
   unsigned offsets[2];
   if (!agx_apple9_entry_table_add(entries, draw->code, offsets))
      return false;
   for (unsigned stage = 0; stage < 2; stage++) {
      const struct agx_apple9_render_stage *shader = stages[stage];
      uint64_t table = stage ? draw->fragment_table : draw->vertex_table;
      bool reuse =
         previous && shader->program_id &&
         previous->program_id[stage] == shader->program_id &&
         previous->code[stage] == draw->code[stage] &&
         previous->texture_table[stage] == draw->texture_table[stage] &&
         previous->sampler_table[stage] == draw->sampler_table[stage] &&
         (stage ? previous->fragment_table : previous->vertex_table) == table &&
         (!stage ||
          (previous->samples == samples && previous->tile_bytes == tile_bytes));
      if (reuse) {
         draw->launch[stage] = previous->launch[stage];
         continue;
      }
      if (shader->preamble_size > AGX_APPLE9_MAX_PREAMBLE_BYTES ||
          shader->preamble_offset > shader->binary_size ||
          shader->preamble_size > shader->binary_size - shader->preamble_offset)
         return false;
      unsigned launch_size = AGX_APPLE9_GRAPHICS_LAUNCH_SIZE;
      struct agx_ptr state = agx_pool_alloc_aligned(usc_pool, 0x100 + launch_size, 64);
      uint8_t *roots = state.cpu;
      memset(roots, 0, 0x100);
      apple9_put_u64(roots, draw->texture_table[stage]);
      apple9_put_u64(roots + 8, draw->sampler_table[stage]);
      apple9_put_u64(roots + 0x10, table);
      struct agx_apple9_launch_parameters params = {
         .preamble_address = shader->preamble_size
            ? shader->bo->va->addr + shader->preamble_offset : 0,
         .launch_address = state.gpu + 0x100,
         .shader_base = dev->shader_base,
         .resource_table = state.gpu,
         .entry_offset = offsets[stage],
         .publication_count = shader->publication_count,
         .frame_extent_a = shader->scratch_size,
         .frame_extent_b = shader->scratch_size,
         .tile_bytes = stage ? tile_bytes : 0,
         .samples = stage ? samples : 0,
      };
      if (!agx_apple9_launch_build(
             roots + 0x100, launch_size,
             stage ? AGX_APPLE9_LAUNCH_FRAGMENT : AGX_APPLE9_LAUNCH_VERTEX,
             &params))
         return false;
      draw->launch[stage] = agx_usc_addr(dev, state.gpu + 0x100);
   }
   bool reuse_coefficients = previous && pipeline->vertex.program_id &&
                             pipeline->fragment.program_id &&
                             previous->program_id[0] == draw->program_id[0] &&
                             previous->program_id[1] == draw->program_id[1] &&
                             previous->flatshade_first == draw->flatshade_first;
   unsigned cf_count;
   if (reuse_coefficients) {
      draw->coefficients = previous->coefficients;
      cf_count = previous->cf_count;
   } else {
      struct agx_ptr cf =
         agx_pool_alloc_aligned(usc_pool, APPLE9_CF_BINDINGS_SIZE, 64);
      cf_count = apple9_build_cf_bindings(
         cf.cpu, pipeline->vertex.varying_components,
         pipeline->fragment.apple9_linear_mask,
         pipeline->fragment.apple9_flat_mask, draw->flatshade_first,
         pipeline->fragment.apple9_reads_z,
         pipeline->fragment.reads_point_coord);
      draw->coefficients = agx_usc_addr(dev, cf.gpu);
   }
   draw->cf_count = cf_count;
   uint8_t *ppp = draw->ppp_record;
   memset(ppp, 0, sizeof(draw->ppp_record));
   apple9_put_u32(ppp, 0x10040000);
   apple9_put_u32(ppp + 4, pipeline->vertex.varying_components);
   uint8_t *group = ppp + 0x40;
   apple9_build_direct_bind_group(group, pipeline->vertex.varying_components);
   apple9_put_u32(group + 8, draw->coefficients);
   if (pipeline->fragment.apple9_reads_z)
      apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (1u << 21));
   if (pipeline->vertex.writes_point_size) {
      apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (1u << 18));
      apple9_put_u32(group + 0x2c, 5 + pipeline->vertex.varying_components);
   }
   apple9_put_u32(group + 4,
                  (apple9_get_u32(group + 4) & 0xffff) | (cf_count << 16));
   /* Match the textured setup's native state, including with two user
    * varyings. This field's full resource-count formula is unresolved;
    * the varying-only estimate is insufficient to describe the captures. */
   apple9_put_u32(group + 0x18, MAX2(apple9_get_u32(group + 0x18), 1));
   apple9_put_u32(group + 0x14, draw->launch[1] / 0x40);
   apple9_put_u32(group + 0x5c, tile_bytes ? 0x1ffff : 0);
   /* Explicit per-sample stores preserve the omitted samples and require
    * the same ordered tile access as blending. Opaque tag visibility can
    * lose mixed stencil coverage after an intervening render submission. */
   if (draw->reads_tile || samples > 1)
      apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | (1u << 29));
   /* Authored Metal discard traces: select punch-through and disable
    * triangle merging so shader coverage controls late tests. */
   if (draw->uses_discard)
      apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | 0x44000000u);
   /* Keep derivative helper quads within one primitive. Merging fragments
    * with different interpolation planes corrupts implicit texture LOD. */
   if (draw->disable_tri_merging ||
       draw->object_type != AGX_OBJECT_TYPE_TRIANGLE)
      apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | (1u << 26));
   /* Native raster packet: cull front/back bits 0/1, provoking vertex
    * bits 7/8, front winding bit 16. Preserve the clipping fields. */
   apple9_put_u32(group + 0x70, (apple9_get_u32(group + 0x70) & ~0x30183u) |
                                   ((draw->flatshade_first ? 1u : 3u) << 7) |
                                   draw->raster_control);
   apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) |
                                   (draw->visibility_mode << 14) |
                                   (draw->depth_control & (1u << 21)));
   apple9_put_u32(group + 0x48, (uint32_t)draw->occlusion_index << 17);
   apple9_put_u32(group + 0x34, draw->depth_control);
   apple9_put_u32(group + 0x38, draw->depth_face[0]);
   apple9_put_u32(group + 0x3c, draw->stencil[0]);
   apple9_put_u32(group + 0x40, draw->depth_face[1]);
   apple9_put_u32(group + 0x44, draw->stencil[1]);
   for (unsigned face = 0; face < 2; ++face) {
      uint8_t *face2 = group + 0x54 + face * 4;
      apple9_put_u32(face2, (apple9_get_u32(face2) & 0x0f3fffffu) |
                               ((draw->writes_depth ? 0u : 3u) << 22) |
                               ((uint32_t)draw->object_type << 28));
   }
   apple9_put_u32(ppp + 0xc0, 0xc00);
   /* Region clip is tile-granular. The scissor array supplies exact pixel
    * bounds, including empty rectangles and partial edge tiles. */
   for (unsigned axis = 0; axis < 2; ++axis) {
      uint32_t lo = draw->scissor_min[axis] / 32;
      uint32_t hi = DIV_ROUND_UP(MAX2(draw->scissor_max[axis], 1), 32) - 1;
      apple9_put_u32(ppp + 0xc4 + axis * 4,
                     (axis ? 0 : 0x80000000u) | (lo << 16) | hi);
   }
   for (unsigned axis = 0; axis < 3; ++axis) {
      apple9_put_f32(ppp + 0xd0 + axis * 8, draw->viewport_translate[axis]);
      apple9_put_f32(ppp + 0xd4 + axis * 8, draw->viewport_scale[axis]);
   }
   /* PPP depth-bias/scissor record: header followed by two 16-bit indices. */
   apple9_put_u32(ppp + 0xf0, 0x100);
   apple9_put_u32(ppp + 0xf4, draw->scissor_index |
                                 ((uint32_t)draw->depth_bias_index << 16));
   /* Compare only this canonical 256-byte record, in CPU storage. This
    * bounded check reuses an immutable allocation; it neither scans an arena
    * nor derives byte-range patches. A new batch has no predecessor. */
   if (previous &&
       !memcmp(previous->ppp_record, ppp, sizeof(draw->ppp_record))) {
      draw->ppp = previous->ppp;
   } else {
      uint64_t address = agx_pool_upload_aligned(context_pool, ppp,
                                                 sizeof(draw->ppp_record), 64);
      assert(address >= AGX_APPLE9_RENDER_CONTEXT_BASE &&
             address - AGX_APPLE9_RENDER_CONTEXT_BASE <= UINT32_MAX);
      draw->ppp = address - AGX_APPLE9_RENDER_CONTEXT_BASE;
   }
   return true;
}

struct agx_apple9_ppp_update {
   uint32_t relative;
   uint32_t control;
};

/*
 * These payload objects describe fixed-function state, not shaders or
 * scheduler state.  Their contents live in the client context and are built
 * by the same source serializers as the T8140 path.  Packetizing their
 * addresses here makes the command stream itself Mesa-owned.
 */
static const struct agx_apple9_ppp_update direct_ppp[] = {
   {0x00, 0x700}, {0x40, 0x500}, {0x5c, 0x700}, {0x70, 0x500},
   {0x8c, 0xa00}, {0xc0, 0x300}, {0xa0, 0x200}, {0xac, 0x200},
};

bool
agx_apple9_direct_render_enabled(const struct agx_device *dev)
{
   if (dev->chip != AGX_CHIP_G16G && dev->chip != AGX_CHIP_G17P)
      return false;

   const char *value = getenv("AGX_APPLE9_DIRECT_RENDER");
   return value && !strcmp(value, "1");
}

bool
agx_apple9_link_render_pipeline(struct agx_apple9_render_pipeline *pipeline,
                                struct agx_apple9_render_stage vertex,
                                struct agx_apple9_render_stage fragment)
{
   return agx_apple9_link_render_pipeline_with_prolog(
      pipeline, (struct agx_apple9_render_stage){0}, vertex, fragment);
}

bool
agx_apple9_link_render_pipeline_with_prolog(
   struct agx_apple9_render_pipeline *pipeline,
   struct agx_apple9_render_stage vertex_prolog,
   struct agx_apple9_render_stage vertex,
   struct agx_apple9_render_stage fragment)
{
   if (!pipeline || !vertex.binary || !vertex.binary_size || !fragment.binary ||
       !fragment.binary_size || vertex.binary_size > UINT32_MAX ||
       fragment.binary_size > UINT32_MAX || vertex_prolog.binary ||
       vertex_prolog.binary_size)
      return false;

   /*
    * Apple9 compacts the cross-stage interface to scalar UVS slots.  Position
    * occupies slots 0..3; compacted user scalars follow in semantic order.
    */
   if (vertex.position_components != 4 ||
       vertex.varying_components > AGX_APPLE9_MAX_VARYING_COMPONENTS ||
       fragment.position_components != 0 ||
       fragment.varying_components != vertex.varying_components ||
       memcmp(&vertex.varyings, &fragment.varyings, sizeof(vertex.varyings)) ||
       fragment.render_targets < 1 || fragment.render_targets > 8)
      return false;

   const unsigned scalar_outputs = 4 + vertex.varying_components + vertex.writes_point_size;
   *pipeline = (struct agx_apple9_render_pipeline){
      .vertex = vertex,
      .fragment = fragment,
      /* Filled from the installed entry BO before VDM emission. */
      .pipeline_word = 0,
      /* Launch and PPP offsets are filled from batch-owned allocations. */
      .vertex_state_class = scalar_outputs | (scalar_outputs << 8),
      .primitive = AGX_PRIMITIVE_TRIANGLES,
   };
   return true;
}

size_t
agx_apple9_direct_draw_size(const struct agx_apple9_render_pipeline *pipeline)
{
   assert(pipeline && pipeline->vertex.binary && pipeline->fragment.binary);
   return AGX_APPLE9_DIRECT_STREAM_SIZE + (pipeline->index_size ? 20 : 0);
}

uint8_t *
agx_apple9_emit_direct_draw(uint8_t *out,
                            const struct agx_apple9_render_pipeline *pipeline,
                            unsigned vertex_count, unsigned instance_count,
                            unsigned vertex_start)
{
   assert(pipeline && pipeline->vertex.binary && pipeline->fragment.binary);
   assert(vertex_count > 0 && instance_count > 0);
   assert(pipeline->ppp && !(pipeline->ppp & 0x3f));

   uint32_t header[] = {
      0x4000002e, /* direct vertex state, Apple9 envelope */
      0x00000000, /* vertex word 0 */
      pipeline->pipeline_word,
      pipeline->vertex_launch,
      pipeline->vertex_state_class,
      pipeline->flatshade_first ? 0u : 2u, /* VDM provoking vertex */
      0x00000000, /* VDM padding */
      0x00000500, /* first PPP state update header */
   };
   memcpy(out, header, sizeof(header));
   out += sizeof(header);

   struct agx_apple9_ppp_update ppp[ARRAY_SIZE(direct_ppp)];
   for (unsigned i = 0; i < ARRAY_SIZE(ppp); i++) {
      ppp[i] = direct_ppp[i];
      ppp[i].relative += pipeline->ppp;
   }
   memcpy(out, ppp, sizeof(ppp));
   out += sizeof(ppp);

   apple9_put_u32(out, pipeline->ppp + 0xf0);
   out += 4;
   if (pipeline->index_size) {
      assert(pipeline->index_size == 1 || pipeline->index_size == 2 ||
             pipeline->index_size == 4);
      assert(pipeline->index_buffer >= (1ull << 40) &&
             pipeline->index_buffer < (1ull << 40) + (1ull << 32));
      assert(pipeline->index_extent);
      /* Indices use a byte address relative to the USC aperture. The range
       * is measured from its containing dword, so include the leading bytes
       * before rounding. This preserves byte/halfword starts without copies.
       * Bit 16 enables restart independently of the primitive topology. */
      uint32_t draw[] = {
         0x40000001, /* publish the draw's restart comparand */
         pipeline->restart_index,
         (0x61f00000 | (util_logbase2(pipeline->index_size) << 17)) |
            (pipeline->primitive << 8) |
            (pipeline->primitive_restart ? (1u << 16) : 0),
         (uint32_t)pipeline->index_buffer,
         vertex_count,
         instance_count,
         vertex_start, /* signed baseVertex, represented in two's complement */
         DIV_ROUND_UP(pipeline->index_extent +
                      (pipeline->index_buffer & 3), 4) - 1,
         1, /* indexed tail word, shared by the measured u8/u16/u32 forms */
         0xc0000000,
      };
      memcpy(out, draw, sizeof(draw));
      out += sizeof(draw);
   } else {
      uint32_t draw[] = {
         0x61c40000 | (pipeline->primitive << 8),
         vertex_count,
         instance_count,
         vertex_start,
         0xc0000000,
      };
      memcpy(out, draw, sizeof(draw));
      out += sizeof(draw);
   }
   return out;
}
