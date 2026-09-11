/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9.h"
#include "asahi/compiler/agx_apple9_ir.h"
#include "pipe/p_defines.h"
#include "agx_apple9_launch.h"
#include "agx_immutable_state.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/libagx/libagx_dgc.h"
#include "util/compress.h"
#include "util/list.h"
#include "util/os_file.h"
#include "util/u_call_once.h"
#include "util/u_math.h"
#include "agx_device.h"

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

/* External development launch fragments and compute division support.
 * Render entries and state are generated without a seed image. */
#define APPLE9_EXTERNAL_BLOB_DIR "/home/nsheth/Projects/asahi/tmp/agx-apple9"

struct apple9_external_blob {
   uint8_t *data;
   size_t size;
};

static uint8_t apple9_compute_constant[0x40];
static void apple9_build_sentinel_constant_program(uint8_t *out, unsigned slots);
static struct apple9_external_blob apple9_launcher_blob;
static struct agx_apple9_launch_library apple9_launchers;
static struct apple9_external_blob apple9_division_ssbo8_superset;

static util_once_flag apple9_superset_blobs_once = UTIL_ONCE_FLAG_INIT;
static bool apple9_superset_blobs_loaded;

static bool
apple9_load_external_blob(const char *name, struct apple9_external_blob *out)
{
   char path[256];
   int written = snprintf(path, sizeof(path), "%s/%s", APPLE9_EXTERNAL_BLOB_DIR,
                          name);
   if (written < 0 || (size_t)written >= sizeof(path))
      return false;

   size_t size = 0;
   char *data = os_read_file(path, &size);
   if (!data) {
      fprintf(stderr, "asahi: failed to load Apple9 development blob %s\n",
              path);
      return false;
   }

   out->data = (uint8_t *)data;
   out->size = size;
   return true;
}

static util_once_flag apple9_launcher_once = UTIL_ONCE_FLAG_INIT;
static bool apple9_launchers_loaded;

static void
apple9_load_launchers_once(void)
{
   apple9_launchers_loaded =
      apple9_load_external_blob("launcher-fragments-v5.bin", &apple9_launcher_blob) &&
      agx_apple9_launch_library_open(&apple9_launchers, apple9_launcher_blob.data,
                                    apple9_launcher_blob.size);
}

static bool
apple9_launchers_available(void)
{
   util_call_once(&apple9_launcher_once, apple9_load_launchers_once);
   return apple9_launchers_loaded;
}

static void
apple9_load_superset_blobs_once(void)
{
   apple9_build_sentinel_constant_program(apple9_compute_constant, 30);
   apple9_superset_blobs_loaded =
      apple9_launchers_available() &&
      apple9_load_external_blob("carrier8/division.bin",
                                &apple9_division_ssbo8_superset);
}

static bool
apple9_superset_blobs_available(void)
{
   util_call_once(&apple9_superset_blobs_once, apple9_load_superset_blobs_once);
   return apple9_superset_blobs_loaded;
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

static size_t
apple9_compute_launch_program_size(const struct apple9_compute_abi_desc *abi)
{
   struct agx_apple9_launch_recipe recipe;
   struct agx_apple9_launch_parameters params;
   return agx_apple9_launch_select(&apple9_launchers, AGX_APPLE9_LAUNCH_COMPUTE,
                                   false, &recipe, &params) ? recipe.size : 0;
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
      return apple9_superset_blobs_available() ? &ssbo8_superset : NULL;
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

static bool apple9_archive_call(uint32_t main_offset, uint32_t *call);

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

bool
agx_apple9_patch_scratch_frame(uint8_t *launch, size_t size,
                               unsigned call_offset, unsigned bytes)
{
   if (bytes > AGX_APPLE9_MAX_SCRATCH_BYTES || (bytes & 15))
      return false;
   /* Scratch-free shaders preserve their established compatibility setup. */
   if (!bytes)
      return true;
   if (!launch || call_offset > size || size - call_offset < 18)
      return false;
   uint8_t *frame = launch + call_offset + 8;
   if (memcmp(frame, "\xf7\x00\x2a", 3))
      return false;
   /* EXP-M4-60: all three stages use these two little-endian extents;
    * other setup fields differ and must remain intact. */
   apple9_put_u16(frame + 5, bytes);
   apple9_put_u16(frame + 7, bytes);
   return true;
}

static bool
apple9_build_compute_launch(uint8_t *out, uint64_t usc_exec_base,
                            uint64_t package_base, uint32_t main_offset,
                            uint64_t state_address,
                            uint32_t resource_table_offset,
                            const struct agx_apple9_compute_profile *profile,
                            uint32_t launch_offset)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || !apple9_compute_profile_valid(profile, abi))
      return false;

   if ((abi->has_dynamic_state && !agx_apple9_compute_state_address_supported(
                                     usc_exec_base, state_address)) ||
       (!abi->has_dynamic_state && state_address != 0))
      return false;

   if (package_base > UINT64_MAX - resource_table_offset)
      return false;

   struct agx_apple9_launch_recipe recipe;
   struct agx_apple9_launch_parameters params;
   size_t launch_size = apple9_compute_launch_program_size(abi);
   if (!agx_apple9_launch_select(&apple9_launchers, AGX_APPLE9_LAUNCH_COMPUTE,
                                false, &recipe, &params) ||
       !apple9_archive_call(main_offset, &params.main_call))
      return false;

   params.shader_base = usc_exec_base;
   params.resource_table = package_base + resource_table_offset;
   params.state = state_address;
   params.resource_count = profile->resource_binding_count;
   if (!recipe.generated_resources && params.resource_count > 8)
      return false;
   params.threadgroup_memory_bytes = profile->required_threadgroup_memory_bytes;
   if (profile->scratch_size) {
      params.frame_extent_a = profile->scratch_size;
      params.frame_extent_b = profile->scratch_size;
   }
   params.frame_extent_a = MAX2(params.frame_extent_a, profile->atomic_frame_size);
   return agx_apple9_launch_build(
      out, ALIGN_POT(launch_size, AGX_APPLE9_COMPUTE_LAUNCH_ALIGN),
      &recipe, &params);
}

size_t
agx_apple9_compute_launch_size(const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   return abi ? ALIGN_POT(apple9_compute_launch_program_size(abi),
                          AGX_APPLE9_COMPUTE_LAUNCH_ALIGN)
              : 0;
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
   struct agx_apple9_launch_recipe recipe;
   struct agx_apple9_launch_parameters params;
   return abi && agx_apple9_launch_select(&apple9_launchers,
                    AGX_APPLE9_LAUNCH_COMPUTE, false, &recipe, &params)
             ? agx_apple9_launch_call_offset(&recipe, profile->resource_binding_count) : 0;
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
      /* The advertised per-axis group limit is also the numerator bound used
       * by the exact reciprocal ceiling-division proof. */
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

static bool
apple9_archive_call(uint32_t main_offset, uint32_t *call)
{
   if (main_offset < AGX_APPLE9_RENDER_FIRST_MAIN_OFFSET)
      return false;

   uint64_t value = UINT64_C(0x07aa) +
                    2 * (main_offset - AGX_APPLE9_RENDER_FIRST_MAIN_OFFSET);
   /* The third byte carries address bits beyond the original 64 KiB arena.
    * Graphics pipeline tests validate bit 17 with mains above +0x10000.
    * Keep higher, untested bits out of the encoding. */
   if (value > 0x3ffff)
      return false;

   *call = value;
   return true;
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
       package_base > UINT64_MAX - resource_table_offset - AGX_APPLE9_COMPUTE_GEOMETRY_LOCAL_OFFSET ||
       apple9_division_ssbo8_superset.size > AGX_APPLE9_COMPUTE_DIVISION_TABLE_SIZE ||
       !apple9_range_fits(mapping_size, AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET,
                          AGX_APPLE9_COMPUTE_DIVISION_TABLE_SIZE) ||
       package_base > UINT64_MAX - AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET)
      return false;

   uint8_t *record = package + resource_table_offset;
   uint64_t record_address = package_base + resource_table_offset;

   memset(record, 0, record_size);
   uint64_t division_address =
      package_base + AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET;
   memset(package + AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET, 0,
          AGX_APPLE9_COMPUTE_DIVISION_TABLE_SIZE);
   memcpy(package + AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET,
          apple9_division_ssbo8_superset.data,
          apple9_division_ssbo8_superset.size);
   if (!agx_apple9_build_compute_geometry_fields(
          record, record_size, record_address, geometry))
      return false;
   apple9_put_u64(record + 0x10, division_address);
   for (unsigned i = 0; i < abi->resource_count; ++i)
      apple9_put_u64(record + 0x18 + i * sizeof(uint64_t),
                     i < resource_count ? resources[i] : division_address);
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
   const struct agx_apple9_compute_geometry *geometry)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!mapping || !abi || !resources || !geometry ||
       resource_count != profile->resource_binding_count)
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
          resource_table_offset, profile, launch_offset)) {
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
   const struct agx_apple9_compute_geometry *geometry)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!mapping || !abi || !resources || !geometry ||
       resource_count != profile->resource_binding_count ||
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
          resource_table_offset, profile, launch_offset)) {
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
   switch (format) {
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

   memset(page, 0, 0x4000);
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

bool
agx_apple9_build_render_state_image(void *mapping, size_t mapping_size,
                                    unsigned width, unsigned height)
{
   return agx_apple9_build_render_state_image_for_varyings(
      mapping, mapping_size, width, height, 3);
}

bool
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

static const struct agx_apple9_render_region apple9_render_regions[] = {
   {AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE, AGX_APPLE9_RENDER_COLOR_TEXTURE_OFFSET, 0x20},
   {AGX_APPLE9_RENDER_REGION_COLOR_BUFFER, AGX_APPLE9_RENDER_COLOR_BUFFER_OFFSET, 0x20},
   {AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE, 0x210020, 0x20},
   {AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE, 0x210320, 0x20},
   {AGX_APPLE9_RENDER_REGION_COLOR_BUFFER, 0x210620, 0x20},
};

/* Immutable stage programs are interned once when a package is created.
 * Draw planning compares object identities, never executable byte arrays. */
struct apple9_render_program {
   struct list_head link;
   unsigned references;
   unsigned stage;
   uint32_t size;
   struct agx_bo *bo;
};

struct apple9_state_transition {
   struct list_head link;
   uint64_t previous_id;
   struct agx_immutable_state images[3];
};

struct agx_apple9_render_package {
   struct list_head link;
   struct agx_bo *bo;
   struct agx_bo *state_bo;
   struct agx_bo
      *code[2]; /* Vertex, fragment; owned through programs when interned. */
   uint32_t code_size[2];
   bool sealed;
   uint64_t color_target;
   uint64_t color_targets[8];
   enum pipe_format color_formats[8];
   unsigned nr_targets;
   unsigned samples;
   uint16_t width;
   uint16_t height;
   unsigned active_batches;
   uint64_t last_used;
   struct apple9_render_program *programs[2];
   uint64_t shader_ids[2];
   struct agx_immutable_state images[3];
   uint8_t *fixed_image;
   uint8_t *context_image;
   uint64_t cache_id;
   struct list_head transitions;
   unsigned transition_count;
   uint32_t scratch_size[2]; /* Vertex, fragment, per invocation. */
   uint16_t publication_count[2];
   bool publication_count_valid[2];
   unsigned varying_components;
   struct agx_apple9_interp_mask linear_mask, flat_mask;
   bool reads_z;
   bool writes_point_size;
   bool reads_point_coord;
};

struct agx_apple9_render_cache {
   struct agx_device *dev;
   struct list_head packages;
   struct list_head programs;
   struct agx_va *logical_va;
   struct agx_bo *resident_bo;
   struct agx_bo *resident_state_bo;
   struct agx_apple9_render_package *current;
   uint64_t package_serial;
   bool state_initialized;
   uint64_t generation;
   uint64_t use_serial;
   unsigned package_count;
   /* A failed publication invalidates the selected state. Compute switches
    * the USC mapping to separate storage, preserving these render objects. */
   bool fixed_usc_dirty;
};

#define AGX_APPLE9_RENDER_CACHE_MAX_PACKAGES 16

struct apple9_render_self_relocation {
   uint32_t offset;
   uint32_t relative;
};

/* Absolute self-pointers in the package-owned render-target state graph.
 * Unlike archive calls and compact instruction operands, these name their
 * containing records directly and therefore move with the logical package
 * generation.  Inline-vertex draws did not consume this graph; vertex-fetch
 * made the previously missing relocation observable. */
static const struct apple9_render_self_relocation
   apple9_render_absolute_self_relocations[] = {
      {0x210000, 0x210020}, {0x210008, 0x210120}, {0x210160, 0x210168},
      {0x210300, 0x210320}, {0x210308, 0x210420}, {0x210460, 0x210468},
      {0x210600, 0x210620}, {0x210820, 0x210828},
};

/* Entry slots are per submission, not a cache of shader bodies. The caller
 * waits for prior fixed-USC users before replacing them. Each stage has its
 * own block header, reserved constant area and generated transfer. */
#define APPLE9_ENTRY_BLOCK_SIZE 0xc0u
#define APPLE9_ENTRY_PREFIX     0x80u
static_assert(AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE +
                    2 * AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS *
                       APPLE9_ENTRY_BLOCK_SIZE <=
                 AGX_APPLE9_RENDER_ENTRY_REGION_SIZE,
              "Per-draw entries must fit the fixed entry table");
static uint32_t
apple9_render_entry(unsigned draw, unsigned stage)
{
   return AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE +
          (draw * 2 + stage) * APPLE9_ENTRY_BLOCK_SIZE + APPLE9_ENTRY_PREFIX;
}

static void
apple9_init_render_entries(uint8_t *image)
{
   apple9_put_u32(image, AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE);
   for (unsigned at = 0x40; at < AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE; at += 2)
      apple9_put_u16(image + at, 6);
   apple9_fill_helper_table(image, 0x100, 10);
   apple9_fill_helper_table(image, 0x200, 10);
   for (unsigned draw = 0; draw < AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS; draw++) {
      for (unsigned stage = 0; stage < 2; stage++) {
         unsigned entry = apple9_render_entry(draw, stage);
         apple9_put_u32(image + entry - APPLE9_ENTRY_PREFIX,
                        APPLE9_ENTRY_BLOCK_SIZE);
         apple9_put_u32(image + entry, 0x0e);
      }
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

const struct agx_apple9_render_region *
agx_apple9_render_package_regions(size_t *count)
{
   if (count)
      *count = ARRAY_SIZE(apple9_render_regions);

   return apple9_render_regions;
}

bool
agx_apple9_build_render_package_image(
   void *mapping, size_t mapping_size,
   const struct agx_apple9_render_pipeline *pipeline)
{
   if (!mapping || mapping_size != AGX_APPLE9_RENDER_PACKAGE_SIZE ||
       !pipeline || !pipeline->vertex.binary || !pipeline->fragment.binary)
      return false;
   memset(mapping, 0, mapping_size);
   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_absolute_self_relocations);
        ++i) {
      const struct apple9_render_self_relocation *reloc =
         &apple9_render_absolute_self_relocations[i];
      apple9_put_u64((uint8_t *)mapping + reloc->offset,
                     UINT64_C(0x10000000000) + reloc->relative);
   }
   apple9_init_render_entries(mapping);
   apple9_build_cf_bindings((uint8_t *)mapping + APPLE9_CF_BINDINGS,
                            pipeline->vertex.varying_components,
                            (struct agx_apple9_interp_mask){0},
                            (struct agx_apple9_interp_mask){0}, false,
                            pipeline->fragment.apple9_reads_z,
                            pipeline->fragment.reads_point_coord);
   return true;
}

bool
agx_apple9_relocate_render_package_base(void *mapping, size_t mapping_size,
                                        uint32_t package_offset)
{
   if (!mapping || mapping_size != AGX_APPLE9_RENDER_PACKAGE_SIZE ||
       (package_offset & 0x3f))
      return false;

   uint8_t *package = mapping;

   const uint64_t usc_exec_base = UINT64_C(0x10000000000);
   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_absolute_self_relocations);
        ++i) {
      const struct apple9_render_self_relocation *reloc =
         &apple9_render_absolute_self_relocations[i];
      uint64_t original = usc_exec_base + reloc->relative;
      uint64_t relocated = usc_exec_base + package_offset + reloc->relative;
      uint64_t current = apple9_get_u64(package + reloc->offset);
      if (current != original && current != relocated)
         return false;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_absolute_self_relocations);
        ++i) {
      const struct apple9_render_self_relocation *reloc =
         &apple9_render_absolute_self_relocations[i];
      apple9_put_u64(package + reloc->offset,
                     usc_exec_base + package_offset + reloc->relative);
   }
   return true;
}

struct agx_apple9_render_package *
agx_apple9_render_package_create(
   struct agx_device *dev, const struct agx_apple9_render_pipeline *pipeline)
{
   if (!dev || !pipeline)
      return NULL;

   struct agx_apple9_render_package *result = calloc(1, sizeof(*result));
   if (!result)
      return NULL;
   result->scratch_size[0] = pipeline->vertex.scratch_size;
   result->scratch_size[1] = pipeline->fragment.scratch_size;
   result->publication_count[0] = pipeline->vertex.publication_count;
   result->publication_count[1] = pipeline->fragment.publication_count;
   result->publication_count_valid[0] =
      pipeline->vertex.publication_count_valid;
   result->publication_count_valid[1] = pipeline->fragment.publication_count_valid;
   result->nr_targets = MAX2(pipeline->fragment.render_targets, 1);
   result->samples = MAX2(pipeline->samples, 1);
   memcpy(result->color_targets, pipeline->color_targets, sizeof(result->color_targets));
   memcpy(result->color_formats, pipeline->color_formats, sizeof(result->color_formats));
   result->shader_ids[0] = pipeline->fragment.program_id;
   result->shader_ids[1] = pipeline->vertex.program_id;
   result->varying_components = pipeline->vertex.varying_components;
   result->linear_mask = pipeline->fragment.apple9_linear_mask;
   result->reads_z = pipeline->fragment.apple9_reads_z;
   result->writes_point_size = pipeline->vertex.writes_point_size;
   result->reads_point_coord = pipeline->fragment.reads_point_coord;
   result->flat_mask = pipeline->fragment.apple9_flat_mask;

   struct agx_bo *package = agx_bo_create(
      dev, AGX_APPLE9_RENDER_PACKAGE_SIZE, AGX_APPLE9_RENDER_PACKAGE_SIZE,
      AGX_BO_EXEC | AGX_BO_WRITEBACK, "Apple9 render compiler package");
   if (!package) {
      free(result);
      return NULL;
   }

   if (!agx_apple9_build_render_package_image(agx_bo_map(package),
                                              package->size, pipeline) ||
       !agx_apple9_relocate_render_package_base(
          agx_bo_map(package), package->size,
          AGX_APPLE9_RENDER_PACKAGE_OFFSET)) {
      agx_bo_unreference(dev, package);
      free(result);
      return NULL;
   }

   const char *dump_path = getenv("AGX_APPLE9_PACKAGE_DUMP");
   if (dump_path && dump_path[0]) {
      FILE *dump = fopen(dump_path, "wb");
      if (dump) {
         fwrite(agx_bo_map(package), 1, package->size, dump);
         fclose(dump);
      }
   }

   if (getenv("AGX_APPLE9_PACKAGE_TRACE")) {
      fprintf(stderr,
              "APPLE9_RENDER_PACKAGE storage=%#llx shader_base=%#llx "
              "pipeline=%#x\n",
              (unsigned long long)package->va->addr,
              (unsigned long long)dev->shader_base,
              AGX_APPLE9_RENDER_PACKAGE_OFFSET);
   }

   struct agx_bo *state =
      agx_bo_create(dev, AGX_APPLE9_RENDER_STATE_SIZE, 0x4000, AGX_BO_WRITEBACK,
                    "Apple9 render fixed-function state");
   if (!state || !agx_apple9_build_render_state_image_for_varyings(
                    agx_bo_map(state), state->size, 512, 512,
                    pipeline->vertex.varying_components)) {
      agx_bo_unreference(dev, state);
      agx_bo_unreference(dev, package);
      free(result);
      return NULL;
   }

   result->bo = package;
   result->state_bo = state;
   const struct agx_apple9_render_stage *stages[] = {&pipeline->vertex,
                                                     &pipeline->fragment};
   for (unsigned i = 0; i < 2; i++) {
      result->code_size[i] = stages[i]->binary_size;
      result->code[i] = agx_bo_create(
         dev, stages[i]->binary_size, 0,
         AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK, "Apple9 shader body");
      if (!result->code[i]) {
         for (unsigned j = 0; j < i; j++)
            agx_bo_unreference(dev, result->code[j]);
         agx_bo_unreference(dev, result->bo);
         agx_bo_unreference(dev, result->state_bo);
         free(result);
         return NULL;
      }
      memcpy(agx_bo_map(result->code[i]), stages[i]->binary,
             stages[i]->binary_size);
      agx_bo_note_cpu_write(result->code[i], 0, stages[i]->binary_size);
   }
   list_inithead(&result->link);
   list_inithead(&result->transitions);
   return result;
}

void
agx_apple9_render_package_destroy(struct agx_device *dev,
                                  struct agx_apple9_render_package *package)
{
   if (!package)
      return;

   assert(!package->active_batches);
   for (unsigned i = 0; i < ARRAY_SIZE(package->programs); i++) {
      struct apple9_render_program *program = package->programs[i];
      if (!program)
         agx_bo_unreference(dev, package->code[i]);
      if (program && --program->references == 0) {
         list_del(&program->link);
         agx_bo_unreference(dev, program->bo);
         free(program);
      }
   }
   for (unsigned i = 0; i < ARRAY_SIZE(package->images); i++)
      free(package->images[i].ranges);
   list_for_each_entry_safe(struct apple9_state_transition, transition,
                            &package->transitions, link) {
      for (unsigned i = 0; i < ARRAY_SIZE(transition->images); i++)
         free(transition->images[i].ranges);
      list_del(&transition->link);
      free(transition);
   }
   free(package->fixed_image);
   free(package->context_image);
   agx_bo_unreference(dev, package->bo);
   agx_bo_unreference(dev, package->state_bo);
   free(package);
}

void
agx_apple9_render_package_acquire(struct agx_apple9_render_package *package)
{
   assert(package);
   package->active_batches++;
}

void
agx_apple9_render_package_release(struct agx_apple9_render_package *package)
{
   assert(package && package->active_batches);
   package->active_batches--;
}

bool
agx_apple9_render_package_matches(
   const struct agx_apple9_render_package *package,
   const struct agx_apple9_render_pipeline *pipeline)
{
   if (!package || !package->bo || !pipeline || !pipeline->vertex.binary ||
       !pipeline->fragment.binary)
      return false;
   if (package->varying_components != pipeline->vertex.varying_components ||
       package->reads_z != pipeline->fragment.apple9_reads_z ||
       package->writes_point_size != pipeline->vertex.writes_point_size ||
       package->reads_point_coord != pipeline->fragment.reads_point_coord ||
       !agx_apple9_interp_mask_equal(package->linear_mask,
                                     pipeline->fragment.apple9_linear_mask) ||
       !agx_apple9_interp_mask_equal(package->flat_mask,
                                     pipeline->fragment.apple9_flat_mask) ||
       package->scratch_size[0] != pipeline->vertex.scratch_size ||
       package->scratch_size[1] != pipeline->fragment.scratch_size ||
       package->publication_count[0] != pipeline->vertex.publication_count ||
       package->publication_count[1] != pipeline->fragment.publication_count ||
       package->publication_count_valid[0] !=
          pipeline->vertex.publication_count_valid ||
       package->publication_count_valid[1] !=
          pipeline->fragment.publication_count_valid)
      return false;
   if (pipeline->fragment.program_id && pipeline->vertex.program_id)
      return package->shader_ids[0] == pipeline->fragment.program_id &&
             package->shader_ids[1] == pipeline->vertex.program_id;

   const struct agx_apple9_render_stage *stages[] = {&pipeline->vertex,
                                                     &pipeline->fragment};
   for (unsigned i = 0; i < 2; i++)
      if (package->code_size[i] != stages[i]->binary_size ||
          memcmp(agx_bo_map(package->code[i]), stages[i]->binary,
                 stages[i]->binary_size))
         return false;
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

bool
agx_apple9_relocate_render_package_image(void *mapping, size_t mapping_size,
                                         uint64_t color_target, unsigned width,
                                         unsigned height)
{
   if (!mapping || mapping_size != AGX_APPLE9_RENDER_PACKAGE_SIZE ||
       (color_target & 0xf) || !width || width > 0x4000 || !height ||
       height > 0x4000)
      return false;

   uint8_t *package = mapping;
   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_regions); ++i) {
      const struct agx_apple9_render_region *region = &apple9_render_regions[i];
      if (region->kind != AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE &&
          region->kind != AGX_APPLE9_RENDER_REGION_COLOR_BUFFER)
         continue;

      apple9_patch_render_target(
         package, region->offset,
         region->kind == AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE, color_target,
         width, height);
   }

   return true;
}

/* The attachment graph contains independent load/PBE records and per-target
 * tile layouts. The M4 public-Metal count sweep confirms all eight slots.
 * RGBA8 uses four bytes per sample; the tile allocation rounds to eight. */
static bool
apple9_build_multiple_targets(struct agx_apple9_render_package *package,
                              uint64_t first_target)
{
   unsigned count = package->nr_targets;
   if (!count || count > 8)
      return false;
   unsigned offsets[8], tile_bytes = 0;
   for (unsigned rt = 0; rt < count; ++rt) {
      offsets[rt] = tile_bytes;
      tile_bytes += 4 * agx_apple9_color_words(package->color_formats[rt]);
   }
   unsigned mask = 0;
   for (unsigned rt = 0; rt < count; ++rt) {
      if (package->color_targets[rt] & 15)
         return false;
      if (package->color_targets[rt])
         mask |= BITFIELD_BIT(rt);
   }
   /* Older callers provide the sole target through the explicit argument. */
   if (!mask) {
      package->color_targets[0] = first_target;
      mask = 1;
   }
   uint8_t *image = agx_bo_map(package->bo);
   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_regions); ++i) {
      const struct agx_apple9_render_region *region = &apple9_render_regions[i];
      if (region->kind != AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE &&
          region->kind != AGX_APPLE9_RENDER_REGION_COLOR_BUFFER)
         continue;
      for (unsigned rt = 0; rt < count; ++rt) {
         if (!(mask & BITFIELD_BIT(rt)))
            continue;
         unsigned offset = region->offset + 32 * rt;
         if (rt)
            memcpy(image + offset, image + region->offset, 32);
         /* M4 attachment-format sweep: LOAD uses sampled-format swizzles;
          * PBE has its own component encoding. Keep layout and dimensions. */
         bool texture = region->kind == AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE;
         uint32_t format;
         switch (package->color_formats[rt]) {
         case PIPE_FORMAT_R16_FLOAT:
            format = texture ? 0x688240 : 0x008240;
            break;
         case PIPE_FORMAT_R16G16_FLOAT:
            format = texture ? 0x4888c0 : 0x0488c0;
            break;
         case PIPE_FORMAT_R16G16B16A16_FLOAT:
            format = texture ? 0x888c80 : 0xe48c80;
            break;
         default:
            format = texture ? 0x0a0a00 : 0xc60a00;
            break;
         }
         apple9_put_u32(image + offset,
            (apple9_get_u32(image + offset) & 0xff00003f) | format);
         apple9_patch_render_target(
            image, offset, texture,
            package->color_targets[rt], package->width, package->height);
         if (package->samples > 1) {
            apple9_put_u32(image + offset,
               (apple9_get_u32(image + offset) & ~0xfu) | 4);
            apple9_put_u32(image + offset + 4,
               (apple9_get_u32(image + offset + 4) & ~0x09000000u) |
               (package->samples == 4 ? 0x01000000u : 0));
         }
      }
   }
   unsigned actions = 0;
   for (unsigned rt = 0; rt < count; ++rt) {
      if (mask & BITFIELD_BIT(rt))
         actions |= 2u << (2 * rt);
   }
   uint8_t *graph = image + AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET;
   for (unsigned copy = 0; copy < 2; ++copy) {
      uint8_t *view = graph + 0x300 * copy;
      apple9_put_u32(view + 0x168, actions | (copy ? mask << 24 : 0));
      apple9_put_u32(view + 0x16c, copy ? 0 : mask);
      for (unsigned rt = 0; rt < count; ++rt) {
         if (rt) {
            memcpy(view + 0x170 + 16 * rt, view + 0x170, 16);
            memcpy(view + 0x1f0 + 24 * rt, view + 0x1f0, 24);
         }
         enum pipe_format pipe_format = package->color_formats[rt];
         unsigned format = (agx_apple9_color_components(pipe_format) << 5) |
                           (agx_apple9_color_is_half(pipe_format) ? 0xc : 0x3);
         apple9_put_u32(view + 0x2d0 + 4 * rt,
                        0x100000 | (offsets[rt] << 12) | format);
      }
      apple9_put_u32(view + 0x2f0,
                     0x400000 | (ALIGN_POT(tile_bytes, 8) << 12) |
                     (util_logbase2(package->samples) * 0x500) | mask);
   }
   for (unsigned rt = 0; rt < count; ++rt) {
      enum pipe_format format = package->color_formats[rt];
      unsigned type = agx_apple9_color_is_half(format) ? 0xc : 0x3;
      apple9_put_u32(graph + 0x82c + 4 * rt,
         (agx_apple9_color_components(format) << 26) |
         (package->samples > 1 ? 0x40000 : 0x20000) | (type << 8));
   }
   /* The store graph has its own active mask, sample stride and byte offsets.
    * They describe the EOT export independently of the fragment tile layout. */
   apple9_put_u32(graph + 0x874, mask);
   apple9_put_u32(graph + 0x878, 0x400000 | package->samples | (ALIGN_POT(tile_bytes, 8) << 4));
   for (unsigned rt = 0; rt < count; ++rt)
      apple9_put_u32(graph + 0x87c + 4 * rt, offsets[rt]);
   if (package->samples > 1) {
      uint64_t graph_address = apple9_get_u64(graph) - 0x20;
      apple9_put_u64(graph + 0x8c0, graph_address + 0x8e0);
      memset(graph + 0x8c8, 0xff, 24);
      memcpy(graph + 0x8e0, graph + 0x620, 32 * count);
   }
   return true;
}

bool
agx_apple9_render_package_prepare(struct agx_apple9_render_package *package,
                                  uint64_t color_target, unsigned width,
                                  unsigned height)
{
   if (!package || !package->bo || !package->state_bo || (color_target & 0xf) ||
       !width || width > 0x4000 || !height || height > 0x4000)
      return false;

   /* The compatibility transport fixes this package at one USC VA.  Never
    * retarget it while an earlier batch may still reference it. */
   if (package->sealed) {
      return package->color_target == color_target && package->width == width &&
             package->height == height;
   }

   if (!agx_apple9_relocate_render_package_image(agx_bo_map(package->bo),
                                                 package->bo->size,
                                                 color_target, width, height))
      return false;

   apple9_build_viewport(
      (uint8_t *)agx_bo_map(package->state_bo) + AGX_APPLE9_VIEWPORT_OFFSET,
      width, height);

   package->color_target = color_target;
   package->width = width;
   package->height = height;
   if (!apple9_build_multiple_targets(package, color_target))
      return false;
   package->sealed = true;
   return true;
}

struct agx_apple9_render_cache *
agx_apple9_render_cache_create(struct agx_device *dev)
{
   if (!dev || dev->shader_base != UINT64_C(0x10000000000))
      return NULL;

   struct agx_apple9_render_cache *cache = calloc(1, sizeof(*cache));
   if (!cache)
      return NULL;

   const uint64_t logical = dev->shader_base + AGX_APPLE9_RENDER_PACKAGE_OFFSET;
   cache->logical_va = agx_va_alloc(dev, AGX_APPLE9_RENDER_PACKAGE_SIZE,
                                    AGX_APPLE9_RENDER_PACKAGE_SIZE,
                                    AGX_VA_USC | AGX_VA_FIXED, logical);
   if (!cache->logical_va) {
      free(cache);
      return NULL;
   }

   cache->resident_bo = agx_bo_create(
      dev, AGX_APPLE9_RENDER_PACKAGE_SIZE, AGX_APPLE9_RENDER_PACKAGE_SIZE,
      AGX_BO_EXEC | AGX_BO_WRITEBACK, "Apple9 fixed render entries");
   cache->resident_state_bo = dev->apple9_render_context;
   agx_bo_reference(cache->resident_state_bo);
   uint32_t bind = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE;
   bool package_bound = cache->resident_bo &&
                        !agx_bo_bind(dev, cache->resident_bo, logical,
                                     AGX_APPLE9_RENDER_PACKAGE_SIZE, 0, bind);
   bool state_bound = package_bound && cache->resident_state_bo;
   if (!state_bound) {
      if (package_bound)
         agx_bo_bind(dev, NULL, logical, AGX_APPLE9_RENDER_PACKAGE_SIZE, 0,
                     DRM_ASAHI_BIND_UNBIND);
      agx_bo_unreference(dev, cache->resident_bo);
      agx_bo_unreference(dev, cache->resident_state_bo);
      agx_va_free(dev, cache->logical_va, false);
      free(cache);
      return NULL;
   }

   cache->dev = dev;
   list_inithead(&cache->packages);
   list_inithead(&cache->programs);
   return cache;
}

void
agx_apple9_render_cache_destroy(struct agx_device *dev,
                                struct agx_apple9_render_cache *cache)
{
   if (!cache)
      return;

   agx_bo_bind(dev, NULL, cache->logical_va->addr,
               AGX_APPLE9_RENDER_PACKAGE_SIZE, 0, DRM_ASAHI_BIND_UNBIND);
   list_for_each_entry_safe(struct agx_apple9_render_package, package,
                            &cache->packages, link) {
      assert(!package->active_batches);
      list_del(&package->link);
      agx_apple9_render_package_destroy(dev, package);
   }

   agx_bo_unreference(dev, cache->resident_bo);
   agx_bo_unreference(dev, cache->resident_state_bo);
   agx_va_free(dev, cache->logical_va, false);
   free(cache);
}

static bool
apple9_intern_package_programs(struct agx_apple9_render_cache *cache,
                               struct agx_apple9_render_package *package)
{
   for (unsigned i = 0; i < 2; i++) {
      if (package->programs[i])
         continue;
      struct apple9_render_program *found = NULL;
      list_for_each_entry(struct apple9_render_program, program, &cache->programs, link) {
         if (program->stage == i && program->size == package->code_size[i] &&
             !memcmp(agx_bo_map(program->bo), agx_bo_map(package->code[i]),
                     program->size)) {
            found = program;
            break;
         }
      }
      if (found) {
         agx_bo_unreference(cache->dev, package->code[i]);
         package->code[i] = found->bo;
      } else {
         found = calloc(1, sizeof(*found));
         if (!found)
            return false;
         found->stage = i;
         found->size = package->code_size[i];
         found->bo = package->code[i];
         list_addtail(&found->link, &cache->programs);
      }
      found->references++;
      package->programs[i] = found;
   }
   return true;
}

static bool
apple9_build_package_state(struct agx_apple9_render_cache *cache,
                           struct agx_apple9_render_package *package)
{
   const uint8_t *source = agx_bo_map(package->bo);
   const uint32_t context_size = AGX_APPLE9_RENDER_STATE_SIZE +
      (AGX_APPLE9_RENDER_STATE_ADDRESS - AGX_APPLE9_RENDER_CONTEXT_BASE);
   package->fixed_image = malloc(AGX_APPLE9_RENDER_PACKAGE_SIZE);
   package->context_image = calloc(1, context_size);
   if (!package->fixed_image || !package->context_image)
      return false;
   uint8_t *fixed_usc = package->fixed_image;
   memcpy(fixed_usc, source, AGX_APPLE9_RENDER_PACKAGE_SIZE);
   uint8_t *resident_state = package->context_image;
   const uint8_t *package_state = agx_bo_map(package->state_bo);
   /* The former template-overlay shim supplied this context page. The
    * caller now owns it, including the null link at the end of bind0. */
   apple9_build_direct_bind0(resident_state, package->varying_components);
   memset(resident_state + 0x340, 0, 8);
   memcpy(resident_state +
             (AGX_APPLE9_RENDER_STATE_ADDRESS - AGX_APPLE9_RENDER_CONTEXT_BASE),
          package_state, AGX_APPLE9_RENDER_STATE_SIZE);

   const uint8_t *data[] = {source + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET,
                            fixed_usc + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET,
                            package->context_image};
   const uint32_t sizes[] = {
      AGX_APPLE9_RENDER_PACKAGE_SIZE - AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET,
      AGX_APPLE9_RENDER_PACKAGE_SIZE - AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET,
      context_size};
   for (unsigned i = 0; i < ARRAY_SIZE(data); i++)
      package->images[i] = (struct agx_immutable_state){.data = data[i], .size = sizes[i]};
   package->cache_id = ++cache->package_serial;
   return true;
}

static struct apple9_state_transition *
apple9_package_transition(struct agx_apple9_render_package *previous,
                          struct agx_apple9_render_package *next)
{
   list_for_each_entry(struct apple9_state_transition, t, &next->transitions, link) {
      if (t->previous_id == previous->cache_id)
         return t;
   }
   struct apple9_state_transition *t = calloc(1, sizeof(*t));
   if (!t)
      return NULL;
   for (unsigned i = 0; i < ARRAY_SIZE(t->images); i++) {
      if (!agx_immutable_state_init(&t->images[i], next->images[i].data,
                                    previous->images[i].data, next->images[i].size)) {
         for (unsigned j = 0; j < i; j++) free(t->images[j].ranges);
         free(t);
         return NULL;
      }
   }
   t->previous_id = previous->cache_id;
   if (next->transition_count == AGX_APPLE9_RENDER_CACHE_MAX_PACKAGES) {
      struct apple9_state_transition *old = list_first_entry(
         &next->transitions, struct apple9_state_transition, link);
      list_del(&old->link);
      for (unsigned i = 0; i < ARRAY_SIZE(old->images); i++) free(old->images[i].ranges);
      free(old);
      next->transition_count--;
   }
   list_addtail(&t->link, &next->transitions);
   next->transition_count++;
   return t;
}

static void
apple9_publish_package_state(struct agx_apple9_render_cache *cache,
                             struct agx_apple9_render_package *package)
{
   struct agx_bo *bos[] = {cache->resident_bo,
      cache->dev->apple9_render_fixed_usc, cache->resident_state_bo};
   const uint32_t offsets[] = {AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET,
                               AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET, 0};
   /* Cache the exact transition: two states can share fields that both
    * differ from an earlier baseline. Those shared fields need no upload. */
   struct apple9_state_transition *transition = cache->state_initialized &&
      !cache->fixed_usc_dirty ? apple9_package_transition(cache->current, package) : NULL;
   const struct agx_immutable_state empty = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(bos); i++) {
      const struct agx_immutable_state *old = transition ? &empty : NULL;
      const struct agx_immutable_state *next = transition ? &transition->images[i] : &package->images[i];
      uint8_t *dst = (uint8_t *)agx_bo_map(bos[i]) + offsets[i];
      uint32_t oi = 0, ni = 0;
      struct agx_state_range range;
      while ((range = agx_immutable_state_next(old, next, &oi, &ni)).size) {
         memcpy(dst + range.offset, next->data + range.offset, range.size);
         agx_bo_note_cpu_write(bos[i], offsets[i] + range.offset, range.size);
      }
   }
   cache->state_initialized = true;
}

struct agx_apple9_render_package *
agx_apple9_render_cache_get(struct agx_apple9_render_cache *cache,
                            const struct agx_apple9_render_pipeline *pipeline,
                            uint64_t color_target, unsigned width,
                            unsigned height)
{
   if (!cache || !pipeline)
      return NULL;

   uint64_t color_targets[8];
   memcpy(color_targets, pipeline->color_targets, sizeof(color_targets));
   bool any_target = false;
   for (unsigned rt = 0; rt < 8; ++rt)
      any_target |= color_targets[rt] != 0;
   if (!any_target)
      color_targets[0] = color_target;

   list_for_each_entry(struct agx_apple9_render_package, package,
                       &cache->packages, link) {
      if (package->sealed && package->color_target == color_target &&
          package->nr_targets == MAX2(pipeline->fragment.render_targets, 1) &&
          package->samples == MAX2(pipeline->samples, 1) &&
          !memcmp(package->color_targets, color_targets,
                  sizeof(color_targets)) &&
          !memcmp(package->color_formats, pipeline->color_formats,
                  sizeof(package->color_formats)) &&
          package->width == width && package->height == height &&
          agx_apple9_render_package_matches(package, pipeline)) {
         package->last_used = ++cache->use_serial;
         return package;
      }
   }

   while (cache->package_count >= AGX_APPLE9_RENDER_CACHE_MAX_PACKAGES) {
      struct agx_apple9_render_package *victim = NULL;
      list_for_each_entry(struct agx_apple9_render_package, candidate,
                          &cache->packages, link) {
         if (candidate != cache->current && !candidate->active_batches &&
             (!victim || candidate->last_used < victim->last_used))
            victim = candidate;
      }

      /* This is a soft memory bound, never a correctness bound. Active
       * batches pin their source packages and may temporarily exceed it. */
      if (!victim)
         break;
      list_del(&victim->link);
      cache->package_count--;
      agx_apple9_render_package_destroy(cache->dev, victim);
   }

   struct agx_apple9_render_package *package =
      agx_apple9_render_package_create(cache->dev, pipeline);
   if (!package || !agx_apple9_render_package_prepare(package, color_target,
                                                      width, height) ||
       !apple9_intern_package_programs(cache, package) ||
       !apple9_build_package_state(cache, package)) {
      agx_apple9_render_package_destroy(cache->dev, package);
      return NULL;
   }

   package->last_used = ++cache->use_serial;
   list_addtail(&package->link, &cache->packages);
   cache->package_count++;
   return package;
}

/* Publish only attachment/state data. Executable bodies are immutable
 * allocations retained by packages and by every batch that references them. */
bool
agx_apple9_render_cache_bind(struct agx_apple9_render_cache *cache,
                             struct agx_apple9_render_package *package)
{
   if (!cache || !package || !package->sealed)
      return false;
   if (!package->images[0].data &&
       (!apple9_intern_package_programs(cache, package) ||
        !apple9_build_package_state(cache, package)))
      return false;
   if (cache->current != package || cache->fixed_usc_dirty)
      apple9_publish_package_state(cache, package);
   if (!agx_apple9_install_render_archive(cache->dev))
      return false;
   cache->current = package;
   cache->fixed_usc_dirty = false;
   cache->generation++;
   return true;
}

bool
agx_apple9_render_cache_bind_draws(struct agx_apple9_render_cache *cache,
                                   const struct agx_apple9_uniform_draw *draws,
                                   unsigned count)
{
   return cache && draws && count &&
          count <= AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS &&
          agx_apple9_render_cache_bind(cache, draws[0].package);
}

void
agx_apple9_render_cache_invalidate_fixed_usc(
   struct agx_apple9_render_cache *cache)
{
   if (cache)
      cache->fixed_usc_dirty = true;
}

/* Per-draw launch records use the shared builder and a common graphics root
 * ABI. Stage recipes supply opaque fragments; live shader calls, frame sizes,
 * publication budgets, roots and tile layout are independent parameters. */
#define APPLE9_UNIFORM_VS_LAUNCH 0x220400u
#define APPLE9_UNIFORM_FS_LAUNCH 0x230800u
#define APPLE9_UNIFORM_RECORDS   0x200100u
#define APPLE9_UNIFORM_CF_TABLES 0x204000u
#define APPLE9_UNIFORM_CF_STRIDE 0x240u
#define APPLE9_DRAW_PIPELINE_PPP 0x5a000u
#define APPLE9_DRAW_PIPELINE_STRIDE 0x100u
#define APPLE9_UNIFORM_STRIDE    0x100u
static_assert(APPLE9_DRAW_PIPELINE_PPP + AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS *
                 APPLE9_DRAW_PIPELINE_STRIDE <= 0x64000,
              "draw state must end before the viewport page");
static_assert(AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS * APPLE9_UNIFORM_STRIDE <=
                 0x3800,
              "uniform launches must fit in their reserved pages");
static_assert(APPLE9_UNIFORM_RECORDS +
                 AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS * 0x100 <= APPLE9_CF_BINDINGS,
              "draw pointer records must end before the default coefficient table");
static_assert(APPLE9_UNIFORM_CF_TABLES + AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS *
                 APPLE9_UNIFORM_CF_STRIDE <= 0x210000,
              "coefficient tables must end before attachment descriptors");
static_assert(APPLE9_CF_BINDINGS_SIZE <= APPLE9_UNIFORM_CF_STRIDE,
              "each coefficient table must fit its per-draw slot");

uint32_t
agx_apple9_render_draw_fragment_word(unsigned draw)
{
   return draw < AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS
             ? APPLE9_UNIFORM_FS_LAUNCH + draw * APPLE9_UNIFORM_STRIDE : 0;
}

bool
agx_apple9_render_cache_upload_uniforms(
   struct agx_apple9_render_cache *cache,
   const struct agx_apple9_uniform_draw *draws, unsigned count)
{
   if (!cache || !cache->current || !draws ||
       count > AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS)
      return false;
   if (!apple9_launchers_available())
      return false;
   uint8_t *views[] = {agx_bo_map(cache->resident_bo),
                       agx_bo_map(cache->dev->apple9_render_fixed_usc)};
   for (unsigned v = 0; v < ARRAY_SIZE(views); v++) {
      apple9_init_render_entries(views[v]);
      agx_bo_note_cpu_write(
         v ? cache->dev->apple9_render_fixed_usc : cache->resident_bo, 0,
         AGX_APPLE9_RENDER_ENTRY_REGION_SIZE);
   }
   uint8_t *context = agx_bo_map(cache->resident_state_bo);
   for (unsigned i = 0; i < count; ++i) {
      const struct agx_apple9_render_package *package = draws[i].package;
      if (!package || !package->code[0] || !package->code[1])
         return false;
      unsigned record = APPLE9_UNIFORM_RECORDS + i * 0x100;
      /* All graphics stages reserve texture, sampler and buffer roots. This
       * ABI is independent of whether a shader actually samples a texture. */
      const unsigned fs_record = 0x30;
      const unsigned cf_table = APPLE9_UNIFORM_CF_TABLES +
                                i * APPLE9_UNIFORM_CF_STRIDE;
      unsigned cf_count = 0;
      for (unsigned v = 0; v < ARRAY_SIZE(views); ++v) {
         uint8_t *view = views[v];
         memset(view + record, 0, 0x100);
         /* Only one buffer root is consumed by each generated main. Remaining
          * launcher argument slots stay zero in the cleared record. */
         apple9_put_u64(view + record + 0x10,
                        draws[i].vertex_table);
         apple9_put_u64(view + record, draws[i].texture_table[0]);
         apple9_put_u64(view + record + 8, draws[i].sampler_table[0]);
         apple9_put_u64(view + record + fs_record + 0x10,
                        draws[i].fragment_table);
         apple9_put_u64(view + record + fs_record, draws[i].texture_table[1]);
         apple9_put_u64(view + record + fs_record + 8, draws[i].sampler_table[1]);
         cf_count = apple9_build_cf_bindings(
            view + cf_table, package->varying_components,
            package->linear_mask, package->flat_mask, draws[i].flatshade_first,
            package->reads_z, package->reads_point_coord);
         for (unsigned stage = 0; stage < 2; ++stage) {
            unsigned location =
               (stage ? APPLE9_UNIFORM_FS_LAUNCH : APPLE9_UNIFORM_VS_LAUNCH) +
               i * APPLE9_UNIFORM_STRIDE;
            uint8_t *launch = view + location;
            struct agx_apple9_launch_recipe recipe;
            struct agx_apple9_launch_parameters params;
            if (!agx_apple9_launch_select(
                   &apple9_launchers, stage ? AGX_APPLE9_LAUNCH_FRAGMENT
                                            : AGX_APPLE9_LAUNCH_VERTEX,
                   stage && draws[i].uses_discard, &recipe, &params))
               return false;
            params.shader_base = cache->dev->shader_base;
            params.resource_table = params.shader_base + record +
                                    (stage ? fs_record : 0);
            uint32_t entry = apple9_render_entry(i, stage);
            if (!apple9_archive_call(entry, &params.main_call) ||
                !apple9_build_body_entry(view + entry,
                                         params.shader_base + entry,
                                         package->code[stage]->va->addr))
               return false;
            params.publication_count = package->publication_count[stage];
            params.publication_count_valid = package->publication_count_valid[stage];
            if (package->scratch_size[stage]) {
               params.frame_extent_a = package->scratch_size[stage];
               params.frame_extent_b = package->scratch_size[stage];
            }
            if (stage) {
               unsigned tile_bytes = 0;
               for (unsigned rt = 0; rt < package->nr_targets; ++rt)
                  tile_bytes += 4 * agx_apple9_color_words(package->color_formats[rt]);
               if (tile_bytes > 64)
                  return false;
               params.tile_bytes = tile_bytes;
               params.samples = package->samples;
            }
            if (!agx_apple9_launch_build(launch, APPLE9_UNIFORM_STRIDE, &recipe, &params))
               return false;
         }
      }
      /* Retain the complete shader-dependent PPP records, including the
       * UVS scalar count and coefficient count. Viewport state is also a
       * per-draw snapshot: utility clears and API draws can use different
       * transforms within the same submission. */
      const uint8_t *state = agx_bo_map(package->state_bo);
      uint8_t *ppp = context + APPLE9_DRAW_PIPELINE_PPP +
                     i * APPLE9_DRAW_PIPELINE_STRIDE;
      memcpy(ppp, state + 0x40, 0x40);
      memcpy(ppp + 0x40, state + AGX_APPLE9_BIND_GROUP_OFFSET, 0x80);
      uint8_t *group = ppp + 0x40;
      apple9_put_u32(group + 8, cf_table);
      if (package->reads_z)
         apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (1u << 21));
      if (package->writes_point_size) {
         apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (1u << 18));
         apple9_put_u32(group + 0x2c, 5 + package->varying_components);
      }
      apple9_put_u32(group + 4,
                     (apple9_get_u32(group + 4) & 0xffff) | (cf_count << 16));
      /* Match the textured setup's native state, including with two user
       * varyings. This field's full resource-count formula is unresolved;
       * the varying-only estimate is insufficient to describe the captures. */
      apple9_put_u32(group + 0x18, MAX2(apple9_get_u32(group + 0x18), 1));
      apple9_put_u32(group + 0x14,
         (APPLE9_UNIFORM_FS_LAUNCH + i * APPLE9_UNIFORM_STRIDE) / 0x40);
      /* EXP-M4-09 authored blend/write-mask state: enable tile read/modify/write.
       * The shader also waits for its allocated tile-load result slot. */
      if (draws[i].reads_tile)
         apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | (1u << 29));
      /* Authored Metal discard traces: select punch-through and disable
       * triangle merging so shader coverage controls late tests. */
      if (draws[i].uses_discard)
         apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | 0x44000000u);
      /* Keep derivative helper quads within one primitive. Merging fragments
       * with different interpolation planes corrupts implicit texture LOD. */
      if (draws[i].disable_tri_merging || draws[i].object_type != AGX_OBJECT_TYPE_TRIANGLE)
         apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | (1u << 26));
      /* Native raster packet: cull front/back bits0/1, front winding bit16.
       * Preserve the independently configured clipping/provoking fields. */
      apple9_put_u32(group + 0x70,
         (apple9_get_u32(group + 0x70) & ~0x10003u) | draws[i].raster_control);
      apple9_put_u32(group + 0x34, draws[i].depth_control);
      apple9_put_u32(group + 0x38, draws[i].depth_face[0]);
      apple9_put_u32(group + 0x3c, draws[i].stencil[0]);
      apple9_put_u32(group + 0x40, draws[i].depth_face[1]);
      apple9_put_u32(group + 0x44, draws[i].stencil[1]);
      for (unsigned face = 0; face < 2; ++face) {
         uint8_t *face2 = group + 0x54 + face * 4;
         apple9_put_u32(face2, (apple9_get_u32(face2) & 0x0fffffffu) |
                                 ((uint32_t)draws[i].object_type << 28));
      }
      memcpy(ppp + 0xc0, state + AGX_APPLE9_VIEWPORT_OFFSET + 0x900, 0x30);
      /* Region clip is tile-granular. The scissor array supplies exact pixel
       * bounds, including empty rectangles and partial edge tiles. */
      for (unsigned axis = 0; axis < 2; ++axis) {
         uint32_t lo = draws[i].scissor_min[axis] / 32;
         uint32_t hi = DIV_ROUND_UP(MAX2(draws[i].scissor_max[axis], 1), 32) - 1;
         apple9_put_u32(ppp + 0xc4 + axis * 4,
                        (axis ? 0 : 0x80000000u) | (lo << 16) | hi);
      }
      for (unsigned axis = 0; axis < 3; ++axis) {
         apple9_put_f32(ppp + 0xd0 + axis * 8, draws[i].viewport_translate[axis]);
         apple9_put_f32(ppp + 0xd4 + axis * 8, draws[i].viewport_scale[axis]);
      }
      /* PPP depth-bias/scissor record: header followed by two 16-bit indices. */
      apple9_put_u32(ppp + 0xf0, 0x100);
      apple9_put_u32(ppp + 0xf4, draws[i].scissor_index |
                                    ((uint32_t)draws[i].depth_bias_index << 16));
   }
   return true;
}

bool
agx_apple9_render_cache_upload_encoder(struct agx_apple9_render_cache *cache,
                                       const void *data, size_t size)
{
   if (!cache || !cache->current || !cache->resident_state_bo || !data ||
       !size || size > AGX_APPLE9_RENDER_FIXED_ENCODER_SIZE)
      return false;

   const size_t offset =
      AGX_APPLE9_RENDER_FIXED_ENCODER - AGX_APPLE9_RENDER_CONTEXT_BASE;
   if (offset + AGX_APPLE9_RENDER_FIXED_ENCODER_SIZE >
       cache->resident_state_bo->size)
      return false;

   uint8_t *encoder = (uint8_t *)agx_bo_map(cache->resident_state_bo) + offset;

   /* Metal places the direct VDM stream in the fixed context-state BO rather
    * than binding a second BO over that range.  Publish the complete bounded
    * stream only after cache_bind() has installed the selected generation;
    * zeroing the remainder also prevents stale commands from a longer prior
    * draw from surviving behind the native terminator. */
   memset(encoder, 0, AGX_APPLE9_RENDER_FIXED_ENCODER_SIZE);
   memcpy(encoder, data, size);

   if (getenv("AGX_APPLE9_PACKAGE_TRACE")) {
      fprintf(stderr, "APPLE9_ENCODER_FIXED bytes=%zu head", size);
      for (unsigned i = 0; i < MIN2(size, 64); ++i)
         fprintf(stderr, "%s%02x", i ? "" : " ", encoder[i]);
      fprintf(stderr, "\n");
   }

   return true;
}

bool
agx_apple9_render_cache_is_current(
   const struct agx_apple9_render_cache *cache,
   const struct agx_apple9_render_package *package)
{
   return cache && cache->current == package && !cache->fixed_usc_dirty;
}

struct agx_bo *
agx_apple9_render_cache_bo(const struct agx_apple9_render_cache *cache)
{
   return cache ? cache->resident_bo : NULL;
}

struct agx_bo *
agx_apple9_render_cache_state_bo(const struct agx_apple9_render_cache *cache)
{
   return cache ? cache->resident_state_bo : NULL;
}

struct agx_bo *
agx_apple9_render_package_bo(const struct agx_apple9_render_package *package)
{
   return package ? package->bo : NULL;
}

struct agx_bo *
agx_apple9_render_state_bo(const struct agx_apple9_render_package *package)
{
   return package ? package->state_bo : NULL;
}

uint32_t
agx_apple9_render_package_pipeline_word(
   const struct agx_device *dev,
   const struct agx_apple9_render_package *package)
{
   if (!dev || !package || !package->bo || !package->bo->va)
      return 0;

   return AGX_APPLE9_RENDER_PACKAGE_OFFSET;
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
   {0x00004040, 0x00000700}, {0x00058000, 0x00000500}, {0x0005801c, 0x00000700},
   {0x00058030, 0x00000500}, {0x0005804c, 0x00000a00}, {0x00068900, 0x00000300},
   {0x00058060, 0x00000200}, {0x0005806c, 0x00000200},
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
      /* EXP-M4-57: this is a launch address in 64-byte units. The native
       * second draw advances it by three for another 0xc0-byte wrapper,
       * while the scalar output interface remains identical. */
      .vertex_launch = AGX_APPLE9_RENDER_VERTEX_LAUNCH_OFFSET / 0x40,
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
   assert(pipeline->uniform_draw <= AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS);
   assert(!pipeline->package ||
          agx_apple9_render_package_matches(pipeline->package, pipeline));

   uint32_t header[] = {
      0x4000002e, /* direct vertex state, Apple9 envelope */
      0x00000000, /* vertex word 0 */
      pipeline->pipeline_word,
      pipeline->vertex_launch,
      pipeline->vertex_state_class,
      0x00000000,
      0x00000000,
      0x00000500,
   };
   if (pipeline->uniform_draw)
      header[3] = (APPLE9_UNIFORM_VS_LAUNCH +
                   (pipeline->uniform_draw - 1) * APPLE9_UNIFORM_STRIDE) / 0x40;
   memcpy(out, header, sizeof(header));
   out += sizeof(header);

   struct agx_apple9_ppp_update ppp[ARRAY_SIZE(direct_ppp)];
   memcpy(ppp, direct_ppp, sizeof(ppp));

   if (pipeline->uniform_draw) {
      uint32_t state = APPLE9_DRAW_PIPELINE_PPP +
                       (pipeline->uniform_draw - 1) * APPLE9_DRAW_PIPELINE_STRIDE;
      ppp[0].relative = state;
      ppp[1].relative = state + 0x40;
      ppp[2].relative = state + 0x5c;
      ppp[3].relative = state + 0x70;
      ppp[4].relative = state + 0x8c;
      ppp[5].relative = state + 0xc0;
      ppp[6].relative = state + 0xa0;
      ppp[7].relative = state + 0xac;
   }
   memcpy(out, ppp, sizeof(ppp));
   out += sizeof(ppp);

   apple9_put_u32(out, pipeline->uniform_draw
      ? APPLE9_DRAW_PIPELINE_PPP +
           (pipeline->uniform_draw - 1) * APPLE9_DRAW_PIPELINE_STRIDE + 0xf0
      : AGX_APPLE9_DRAW_STATE - AGX_APPLE9_RENDER_CONTEXT_BASE);
   out += 4;
   if (pipeline->index_size) {
      assert(pipeline->index_size == 2 || pipeline->index_size == 4);
      assert(pipeline->index_buffer >= (1ull << 40) &&
             pipeline->index_buffer < (1ull << 40) + (1ull << 32));
      assert(pipeline->index_extent);
      /* M4 CMD-4 and native EXP-M4-58: indices use a 32-bit offset
       * into the fixed USC aperture. Extent is a dword count minus one. */
      uint32_t draw[] = {
         0x40000001, /* publish restart comparand (restart disabled below) */
         pipeline->index_size == 2 ? 0xffff : 0xffffffff,
         (pipeline->index_size == 2 ? 0x61f20000 : 0x61f40000) |
            (pipeline->primitive << 8),
         (uint32_t)pipeline->index_buffer,
         vertex_count,
         instance_count,
         vertex_start, /* signed baseVertex, represented in two's complement */
         DIV_ROUND_UP(pipeline->index_extent, 4) - 1,
         1, /* indexed tail word, shared by the measured u16/u32 forms */
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

/* The authored EXP-M4-09 MRT captures identify float4 clear colors in the
 * attachment state graph. The compatibility image retains the same data
 * records at +0x170 and +0x470. Update both published views per submission;
 * neither program code nor the immutable source package is modified. */
void
agx_apple9_render_cache_set_clear_color(struct agx_apple9_render_cache *cache,
                                       const float color[8][4])
{
   uint8_t *resident = agx_bo_map(cache->resident_bo);
   uint8_t *fixed = agx_bo_map(cache->dev->apple9_render_fixed_usc);
   for (unsigned i = 0; i < 2; ++i) {
      unsigned offset = 0x170 + i * 0x300;
      memcpy(resident + AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET + offset,
             color, 16 * cache->current->nr_targets);
      memcpy(fixed + AGX_APPLE9_RENDER_FIXED_TARGET_GRAPH_OFFSET + offset,
             color, 16 * cache->current->nr_targets);
   }
}

struct agx_bo *
agx_apple9_render_code_bo(const struct agx_apple9_render_package *package,
                          unsigned stage)
{
   return package && stage < 2 ? package->code[stage] : NULL;
}
