/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9.h"

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

static_assert(AGX_APPLE9_COMPUTE_STATE_LITERAL_STORAGE_CAPACITY *
                    sizeof(uint32_t) <=
                 0x20,
              "Apple9 state literals must fit after state +0x20");
static_assert(AGX_APPLE9_COMPUTE_CODE_SIZE == AGX_APPLE9_COMPUTE_ARCHIVE_SIZE,
              "Gallium and libagx must agree on the compute archive size");

/* Temporary development inputs.  These caller/compiler programs and render
 * templates are recaptured separately and deliberately live outside Mesa. */
#define APPLE9_EXTERNAL_BLOB_DIR "/home/nsheth/Projects/asahi/tmp/agx-apple9"

struct apple9_external_blob {
   uint8_t *data;
   size_t size;
};

static struct apple9_external_blob apple9_constant_ssbo8_superset;
static struct apple9_external_blob apple9_launch_ssbo8_superset;
static struct apple9_external_blob apple9_division_ssbo8_superset;
static struct apple9_external_blob apple9_constant_native_atomic8;
static struct apple9_external_blob apple9_launch_native_atomic8;
static struct apple9_external_blob apple9_g16_render_package_zst;
static struct apple9_external_blob apple9_render_interleaved_vbo_launch;

static util_once_flag apple9_superset_blobs_once = UTIL_ONCE_FLAG_INIT;
static util_once_flag apple9_atomic_blobs_once = UTIL_ONCE_FLAG_INIT;
static bool apple9_superset_blobs_loaded;
static bool apple9_atomic_blobs_loaded;

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

static void
apple9_load_superset_blobs_once(void)
{
   apple9_superset_blobs_loaded =
      apple9_load_external_blob("carrier8/constant.bin",
                                &apple9_constant_ssbo8_superset) &&
      apple9_load_external_blob("carrier8/launch.bin",
                                &apple9_launch_ssbo8_superset) &&
      apple9_load_external_blob("carrier8/division.bin",
                                &apple9_division_ssbo8_superset);
}

static void
apple9_load_atomic_blobs_once(void)
{
   apple9_atomic_blobs_loaded =
      apple9_load_external_blob("carrier8-atomic/constant.bin",
                                &apple9_constant_native_atomic8) &&
      apple9_load_external_blob("carrier8-atomic/launch.bin",
                                &apple9_launch_native_atomic8);
}

static bool
apple9_superset_blobs_available(void)
{
   util_call_once(&apple9_superset_blobs_once, apple9_load_superset_blobs_once);
   return apple9_superset_blobs_loaded;
}

static bool
apple9_atomic_blobs_available(void)
{
   util_call_once(&apple9_atomic_blobs_once, apple9_load_atomic_blobs_once);
   return apple9_atomic_blobs_loaded;
}

static util_once_flag apple9_render_blobs_once = UTIL_ONCE_FLAG_INIT;
static bool apple9_render_blobs_loaded;

static void
apple9_load_render_blobs_once(void)
{
   apple9_render_blobs_loaded =
      apple9_load_external_blob("g16_render_package.bin.zst",
                                &apple9_g16_render_package_zst) &&
      apple9_load_external_blob("render_interleaved_vbo_launch.bin",
                                &apple9_render_interleaved_vbo_launch);
}

static bool
apple9_render_blobs_available(void)
{
   util_call_once(&apple9_render_blobs_once, apple9_load_render_blobs_once);
   return apple9_render_blobs_loaded;
}

struct apple9_compute_abi_desc {
   const struct apple9_external_blob *constant_blob;
   const struct apple9_external_blob *launch_blob;
   uint16_t archive_call_offset;
   uint8_t helper_slots;
   uint8_t resource_count;
   uint8_t hidden_resource_count;
   uint16_t resource_record_size;
   bool has_dynamic_state;
   uint8_t state_uniform_base;
   uint8_t state_literal_capacity;
   uint32_t required_threadgroup_memory_bytes;
   uint32_t cdm_config;
   uint32_t cdm_constant;
   uint32_t cdm_tail;
   bool supports_indirect_dispatch;
};

static const uint8_t *
apple9_compute_constant_program(const struct apple9_compute_abi_desc *abi)
{
   return abi->constant_blob->data;
}

static size_t
apple9_compute_constant_size(const struct apple9_compute_abi_desc *abi)
{
   return abi->constant_blob->size;
}

static const uint8_t *
apple9_compute_launch_program(const struct apple9_compute_abi_desc *abi)
{
   return abi->launch_blob->data;
}

static size_t
apple9_compute_launch_program_size(const struct apple9_compute_abi_desc *abi)
{
   return abi->launch_blob->size;
}

static const struct apple9_compute_abi_desc *
apple9_compute_abi(const struct agx_apple9_compute_profile *profile)
{
   static const struct apple9_compute_abi_desc ssbo8_superset = {
      .constant_blob = &apple9_constant_ssbo8_superset,
      .launch_blob = &apple9_launch_ssbo8_superset,
      .archive_call_offset = 0x8c,
      .helper_slots = 10,
      .resource_count = AGX_APPLE9_COMPUTE_MAX_RESOURCES,
      .hidden_resource_count = AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE,
      .resource_record_size = 0x80,
      .has_dynamic_state = true,
      .cdm_config = 0x00880000,
      .cdm_constant = 0x01000040,
      .cdm_tail = 0x60000160,
      .supports_indirect_dispatch = true,
   };
   static const struct apple9_compute_abi_desc native_atomic8 = {
      .constant_blob = &apple9_constant_native_atomic8,
      .launch_blob = &apple9_launch_native_atomic8,
      .archive_call_offset = 0x70,
      .helper_slots = 10,
      .resource_count = AGX_APPLE9_COMPUTE_MAX_RESOURCES,
      .hidden_resource_count = 0,
      .resource_record_size = 0x80,
      .has_dynamic_state = true,
      .cdm_config = 0x00080000,
      .cdm_constant = 0x01000000,
      .cdm_tail = 0x60010060,
   };

   if (!profile)
      return NULL;

   switch (profile->abi) {
   case AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET:
      return apple9_superset_blobs_available() ? &ssbo8_superset : NULL;
   case AGX_APPLE9_COMPUTE_ABI_SSBO8_ATOMIC:
      return apple9_atomic_blobs_available() ? &native_atomic8 : NULL;
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
       profile->required_threadgroup_memory_bytes != 0)
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

   size_t launch_size = apple9_compute_launch_program_size(abi);
   if (abi->archive_call_offset > launch_size ||
       launch_size - abi->archive_call_offset < 3 || launch_size < 8 ||
       (abi->has_dynamic_state && launch_size < 0x18))
      return false;

   size_t allocation =
      ALIGN_POT(launch_size, AGX_APPLE9_COMPUTE_LAUNCH_ALIGN);
   memset(out, 0, allocation);
   memcpy(out, apple9_compute_launch_program(abi), launch_size);

   /*
    * The archive call is a three-byte relative field. Caller-owned M4 captures
    * place the first main at +0x3c0 with field 0x0007aa and advance the field
    * by two for every byte appended to the archive. At main +0x8040, the old
    * 16-bit portion wraps to 0x00aa and launch byte 0x48 becomes 1. Native
    * archives occupy the full 64-KiB executable BO, so preserve that carry.
    */
   uint32_t call;
   if (!apple9_archive_call(main_offset, &call))
      return false;
   apple9_put_u24(out + abi->archive_call_offset, call);

   /* Dynamic Caching state is an immutable, pipeline-owned object.  Its launch
    * pointer names the record's +0x20 half and must remain stable across every
    * dispatch using that compiled shader. */
   if (abi->has_dynamic_state &&
       !apple9_patch_compact_pointer(out, 0x11, 0x14, 0x15, 0x16, usc_exec_base,
                                     state_address))
      return false;

   /*
    * Dispatches sharing a resource BO use ABI-sized Tier-2 records (0x20 for
    * the established carriers, 0x40 for EXP-M4-28 shared memory).  The launch
    * always names the record start: bytes 1/4/5 carry the 13-bit in-chunk
    * selector and bytes 6/7 carry the 8-KiB chunk.
    */
   if (package_base > UINT64_MAX - resource_table_offset)
      return false;
   return apple9_patch_compact_pointer(out, 0x01, 0x04, 0x05, 0x06,
                                       usc_exec_base,
                                       package_base + resource_table_offset);
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
             ? abi->required_threadgroup_memory_bytes
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
   return abi ? abi->archive_call_offset : 0;
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
   if (value > 0x1ffff)
      return false;

   *call = value;
   return true;
}

static bool
apple9_compute_archive_block_layout(size_t main_size,
                                    const struct apple9_compute_abi_desc *abi,
                                    uint32_t *main_relative,
                                    uint32_t *block_size)
{
   if (!abi || !main_relative || !block_size || main_size > UINT32_MAX)
      return false;

   const uint8_t *constant = apple9_compute_constant_program(abi);
   size_t constant_size = apple9_compute_constant_size(abi);
   if (!constant || !constant_size || (constant_size & 0x3f))
      return false;

   size_t main_at = AGX_APPLE9_COMPUTE_BLOCK_HEADER_SIZE + constant_size;
   if (main_at > UINT32_MAX || main_size > SIZE_MAX - main_at ||
       main_at + main_size > SIZE_MAX - 0x3f)
      return false;

   size_t total = ALIGN_POT(main_at + main_size, 0x40);
   if (total > UINT32_MAX)
      return false;

   *main_relative = main_at;
   *block_size = total;
   return true;
}

static bool
apple9_build_compute_archive_block(uint8_t *out, size_t out_size,
                                   const void *main, size_t main_size,
                                   const struct apple9_compute_abi_desc *abi,
                                   uint32_t *main_relative,
                                   uint32_t *block_size)
{
   static const uint8_t stop[] = {0x0e, 0x00, 0x00, 0x00};
   uint32_t main_at, total;
   if (!out || !main || main_size < sizeof(stop) ||
       memcmp((const uint8_t *)main + main_size - sizeof(stop), stop,
              sizeof(stop)) != 0 ||
       !apple9_compute_archive_block_layout(main_size, abi, &main_at, &total) ||
       total > out_size)
      return false;

   memset(out, 0, total);
   apple9_put_u32(out, total);
   memcpy(out + AGX_APPLE9_COMPUTE_BLOCK_HEADER_SIZE,
          apple9_compute_constant_program(abi),
          apple9_compute_constant_size(abi));
   memcpy(out + main_at, main, main_size);
   *main_relative = main_at;
   *block_size = total;
   return true;
}

bool
agx_apple9_build_compute_archive_image(
   void *mapping, size_t mapping_size, const void *main, size_t main_size,
   const struct agx_apple9_compute_profile *profile, uint32_t *main_offset)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!mapping || mapping_size < AGX_APPLE9_COMPUTE_CODE_SIZE || !abi ||
       !main_offset || !apple9_compute_profile_valid(profile, abi))
      return false;

   uint8_t *archive = malloc(AGX_APPLE9_COMPUTE_CODE_SIZE);
   if (!archive)
      return false;
   apple9_init_compute_archive(archive, abi->helper_slots);
   uint32_t main_relative, block_size;
   if (!apple9_build_compute_archive_block(
          archive + AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE,
          AGX_APPLE9_COMPUTE_CODE_SIZE - AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE,
          main, main_size, abi, &main_relative, &block_size)) {
      free(archive);
      return false;
   }

   uint32_t absolute = AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE + main_relative;
   uint32_t call;
   if (!apple9_archive_call(absolute, &call)) {
      free(archive);
      return false;
   }

   memcpy(mapping, archive, AGX_APPLE9_COMPUTE_CODE_SIZE);
   free(archive);
   *main_offset = absolute;
   return true;
}

bool
agx_apple9_upload_compute_shader(
   struct agx_device *dev, const void *main, size_t main_size,
   const struct agx_apple9_compute_profile *profile, uint32_t *main_offset)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!dev || !dev->apple9_compute_archive ||
       !dev->apple9_compute_archive_shadow || !main || !main_size || !abi ||
       !main_offset || !apple9_compute_profile_valid(profile, abi))
      return false;

   uint32_t main_relative, block_size;
   if (!apple9_compute_archive_block_layout(main_size, abi, &main_relative,
                                            &block_size))
      return false;
   uint8_t *candidate = malloc(block_size);
   if (!candidate)
      return false;
   if (!apple9_build_compute_archive_block(candidate, block_size, main,
                                           main_size, abi, &main_relative,
                                           &block_size)) {
      free(candidate);
      return false;
   }

   simple_mtx_lock(&dev->apple9_archive_lock);
   uint8_t *archive = dev->apple9_compute_archive_shadow;
   if (dev->apple9_archive_next == 0) {
      apple9_init_compute_archive(archive, abi->helper_slots);
      dev->apple9_archive_next = AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE;
      dev->apple9_archive_helper_slots = abi->helper_slots;
   } else if (dev->apple9_archive_helper_slots != abi->helper_slots) {
      fprintf(stderr,
              "Apple9 archive helper ABI mismatch: resident=%u requested=%u\n",
              dev->apple9_archive_helper_slots, abi->helper_slots);
      simple_mtx_unlock(&dev->apple9_archive_lock);
      free(candidate);
      return false;
   }

   /*
    * The caller's native archive interns byte-identical executable blocks, so
    * do not spend the compact call window on duplicate mains.
    */
   for (unsigned i = 0; i < dev->apple9_archive_entry_count; ++i) {
      const struct agx_apple9_archive_entry *entry =
         &dev->apple9_archive_entries[i];
      if (entry->block_size == block_size &&
          memcmp(archive + entry->block_offset, candidate, block_size) == 0) {
         *main_offset = entry->main_offset;
         simple_mtx_unlock(&dev->apple9_archive_lock);
         free(candidate);
         return true;
      }
   }

   uint32_t block = dev->apple9_archive_next;
   uint32_t main_at = block + main_relative;
   uint32_t call;
   bool fits =
      !(block & 0x3f) && block <= AGX_APPLE9_COMPUTE_CODE_SIZE &&
      block_size <= AGX_APPLE9_COMPUTE_CODE_SIZE - block &&
      dev->apple9_archive_entry_count < AGX_APPLE9_ARCHIVE_MAX_ENTRIES &&
      apple9_archive_call(main_at, &call);
   if (fits) {
      memcpy(archive + block, candidate, block_size);
      dev->apple9_archive_next = block + block_size;
      ++dev->apple9_compute_archive_generation;
      *main_offset = main_at;
      unsigned index = dev->apple9_archive_entry_count++;
      dev->apple9_archive_entries[index] = (struct agx_apple9_archive_entry){
         .block_offset = block,
         .block_size = block_size,
         .main_offset = main_at,
      };
      if (getenv("AGX_APPLE9_PACKAGE_TRACE") != NULL) {
         fprintf(stderr,
                 "APPLE9_ARCHIVE index=%u block=%#x main=%#x size=%#zx "
                 "next=%#x\n",
                 index, block, main_at, main_size, dev->apple9_archive_next);
      }
   } else {
      fprintf(stderr,
              "Apple9 archive allocation failed: next=%#x main=%#zx "
              "limit=%#x\n",
              block, main_size, AGX_APPLE9_COMPUTE_CODE_SIZE);
   }
   simple_mtx_unlock(&dev->apple9_archive_lock);
   free(candidate);
   return fits;
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
       (abi->hidden_resource_count + abi->resource_count) * sizeof(uint64_t) >
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
       package_base > UINT64_MAX - resource_table_offset - 0x6c ||
       (abi->hidden_resource_count != 0 &&
        (apple9_division_ssbo8_superset.size >
            AGX_APPLE9_COMPUTE_DIVISION_TABLE_SIZE ||
         !apple9_range_fits(mapping_size,
                            AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET,
                            AGX_APPLE9_COMPUTE_DIVISION_TABLE_SIZE) ||
         package_base >
            UINT64_MAX - AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET)))
      return false;

   uint8_t *record = package + resource_table_offset;
   uint64_t record_address = package_base + resource_table_offset;

   memset(record, 0, record_size);
   if (abi->hidden_resource_count == 0) {
      for (unsigned i = 0; i < abi->resource_count; ++i)
         apple9_put_u64(record + i * sizeof(uint64_t),
                        i < resource_count ? resources[i] : 0);
   } else {
      uint64_t division_address =
         package_base + AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET;
      memset(package + AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET, 0,
             AGX_APPLE9_COMPUTE_DIVISION_TABLE_SIZE);
      memcpy(package + AGX_APPLE9_COMPUTE_DIVISION_TABLE_OFFSET,
             apple9_division_ssbo8_superset.data,
             apple9_division_ssbo8_superset.size);
      if (abi->hidden_resource_count != 3 ||
          !agx_apple9_build_compute_geometry_fields(
             record, record_size, record_address, geometry))
         return false;
      apple9_put_u64(record + 0x10, division_address);
      for (unsigned i = 0; i < abi->resource_count; ++i)
         apple9_put_u64(record + 0x18 + i * sizeof(uint64_t),
                        i < resource_count ? resources[i] : division_address);
      /* qword 11 is the native zero sentinel. */
      apple9_put_u64(record + 0x58, 0);
   }
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

void
agx_apple9_build_compute_package(
   void *mapping, uint64_t base, const void *main, size_t main_size,
   const uint64_t *resources, unsigned resource_count,
   const struct agx_apple9_compute_profile *profile)
{
   /*
    * The archive block is header(0x40) + the carrier constant program + main
    * rounded to 0x40 bytes.  EXP-M4-39 proves that the first 0x40 bytes of the
    * captured constant region are sufficient with the largest eight-resource
    * main.  Smaller alignment remains invalid: a 0x10-byte-rounded block can
    * retire while a terminal compare returns garbage.
    */
   uint8_t *package = mapping;
   uint8_t *code = package + AGX_APPLE9_COMPUTE_CODE_OFFSET;
   memset(package, 0, AGX_APPLE9_COMPUTE_PACKAGE_SIZE);
   uint32_t main_offset;
   bool archived = agx_apple9_build_compute_archive_image(
      code, AGX_APPLE9_COMPUTE_CODE_SIZE, main, main_size, profile,
      &main_offset);
   assert(archived && main_offset == AGX_APPLE9_COMPUTE_MAIN_OFFSET);
   bool built = agx_apple9_build_compute_dispatch(
      package, AGX_APPLE9_COMPUTE_PACKAGE_SIZE, base, base, main_offset,
      AGX_APPLE9_COMPUTE_LAUNCH_OFFSET,
      agx_apple9_compute_has_dynamic_state(profile)
         ? AGX_APPLE9_COMPUTE_STATE_OFFSET
         : 0,
      AGX_APPLE9_COMPUTE_RESOURCE_OFFSET +
         AGX_APPLE9_COMPUTE_RESOURCE_TABLE_OFFSET,
      profile, resources, resource_count,
      &(const struct agx_apple9_compute_geometry){
         .mode = AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT,
         .threads = {1, 1, 1},
         .local = {1, 1, 1},
      });
   assert(built);
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

void
agx_apple9_pack_nearest_sampler(void *out)
{
   const uint32_t words[2] = {
      112u << 13,            /* lod_min=0.0, lod_max=14.0 */
      (1u << 7) | (7u << 8), /* clamp-to-edge, nearest, compare always */
   };
   memcpy(out, words, sizeof(words));
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

/* Caller-authored coefficient table in the graphics data page, between the
 * compatibility resource pointers and per-draw buffer argument records. */
#define APPLE9_CF_BINDINGS 0x200040u

static unsigned
apple9_cf_binding_count(unsigned components)
{
   return 1 + DIV_ROUND_UP(components, 4);
}

static void
apple9_build_cf_bindings(uint8_t *table, unsigned components)
{
   assert(components <= AGX_APPLE9_MAX_VARYING_COMPONENTS);
   memset(table, 0, 0x40);
   /* Public CF_BINDING format, independently observed on T8132. Coefficient
    * zero is linearly interpolated 1/W. User varyings stay unprojected in
    * the VS for clipping, then use perspective coefficients paired with
    * the compiler's coefficient-aware projective multiply. */
   apple9_put_u32(table, (components + 1) | ((components + 1) << 8));
   apple9_put_u32(table + 4, 0x0c);
   for (unsigned start = 0, binding = 1; start < components;
        start += 4, ++binding) {
      unsigned count = MIN2(components - start, 4);
      unsigned base = start + 1;
      apple9_put_u32(table + 4 + binding * 4,
                     (count - 1) | (7 << 2) | (base << 8) | (base << 16));
   }
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
   {AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE, 0x018220, 0x20},
   {AGX_APPLE9_RENDER_REGION_COLOR_BUFFER, 0x018420, 0x20},
   {AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE, 0x210020, 0x20},
   {AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE, 0x210320, 0x20},
   {AGX_APPLE9_RENDER_REGION_COLOR_BUFFER, 0x210620, 0x20},
};

struct agx_apple9_render_package {
   struct list_head link;
   struct agx_bo *bo;
   struct agx_bo *state_bo;
   bool sealed;
   uint64_t color_target;
   uint16_t width;
   uint16_t height;
   unsigned active_batches;
   uint64_t last_used;
   struct agx_apple9_render_archive_layout archive;
   uint32_t fragment_call;
   uint32_t vertex_prolog_call;
   uint32_t vertex_prolog_call_offset;
   uint32_t vertex_call;
   uint64_t vertex_buffer;
   uint32_t vertex_buffer_size;
   unsigned varying_components;
};

struct agx_apple9_render_cache {
   struct agx_device *dev;
   struct list_head packages;
   struct agx_va *logical_va;
   struct agx_bo *resident_bo;
   struct agx_bo *resident_state_bo;
   struct agx_apple9_render_package *current;
   uint64_t generation;
   uint64_t use_serial;
   unsigned package_count;
   uint32_t resident_archive_next;
   uint32_t resident_archive_limit;
   /* Compute and VBO render share fixed-USC +0.  A compute installation does
    * not change the package archive, but it invalidates every fixed-USC view
    * published from the current render package. */
   bool fixed_usc_dirty;
   bool installing_draws;
};

#define AGX_APPLE9_RENDER_CACHE_MAX_PACKAGES 16

/*
 * A render pipeline is an ordered compiler-package closure in the process
 * archive.  Three entries are selected by explicit archive calls (fragment,
 * vertex fetch/prolog, and API vertex main); the empty/metadata entry, stage
 * adapters,
 * runtime-library programs, and format-specific fragment color epilog are
 * nevertheless required by hardware.  Removing any nonzero suffix entry
 * made the G16 render queue stop after StartTA/Start3D, while dropping the
 * final all-zero reservation succeeds.
 *
 * Independent caller-owned Metal captures retain the vertex/adapter/library
 * suffix byte-for-byte even when the shader mains change.  The second program
 * changes with vertex-fetch/resource lowering while the third consumes its
 * exported attributes.  Keep both as separately owned archive programs.
 */
#define AGX_APPLE9_RENDER_FRAGMENT_CALL_OFFSET      0x23064au
#define AGX_APPLE9_RENDER_VERTEX_PROLOG_CALL_OFFSET 0x220036u
#define AGX_APPLE9_RENDER_VBO_PROLOG_CALL_OFFSET    0x220046u
#define AGX_APPLE9_RENDER_VERTEX_CALL_OFFSET        0x18ca37u
#define AGX_APPLE9_RENDER_INTERLEAVED_VBO_LAUNCH_SIZE 0x000c0u

#define AGX_APPLE9_RENDER_TEMPLATE_FRAGMENT_CALL      0x007aau
#define AGX_APPLE9_RENDER_TEMPLATE_VERTEX_PROLOG_CALL 0x00aaau
#define AGX_APPLE9_RENDER_TEMPLATE_VERTEX_CALL        0x00daau

#define AGX_APPLE9_RENDER_TEMPLATE_VERTEX_PROLOG_BLOCK  0x004c0u
#define AGX_APPLE9_RENDER_TEMPLATE_VERTEX_PROLOG_MAIN   0x00540u
#define AGX_APPLE9_RENDER_TEMPLATE_VERTEX_PROLOG_SIZE   0x00180u
#define AGX_APPLE9_RENDER_TEMPLATE_VERTEX_ADAPTER_BLOCK 0x00640u
#define AGX_APPLE9_RENDER_TEMPLATE_VERTEX_ADAPTER_SIZE  0x00140u
#define AGX_APPLE9_RENDER_TEMPLATE_EMPTY_BLOCK          0x00440u
#define AGX_APPLE9_RENDER_TEMPLATE_EMPTY_SIZE           0x00080u
#define AGX_APPLE9_RENDER_TEMPLATE_ADAPTER_A            0x00780u
#define AGX_APPLE9_RENDER_TEMPLATE_ADAPTER_B            0x00840u
#define AGX_APPLE9_RENDER_TEMPLATE_ADAPTER_SIZE         0x000c0u
#define AGX_APPLE9_RENDER_TEMPLATE_LIBRARY_A            0x00900u
#define AGX_APPLE9_RENDER_TEMPLATE_LIBRARY_A_SIZE       0x00840u
#define AGX_APPLE9_RENDER_TEMPLATE_LIBRARY_B            0x01140u
#define AGX_APPLE9_RENDER_TEMPLATE_LIBRARY_B_SIZE       0x01400u
#define AGX_APPLE9_RENDER_TEMPLATE_COLOR_EPILOG         0x02540u
#define AGX_APPLE9_RENDER_TEMPLATE_COLOR_EPILOG_SIZE    0x000c0u
#define AGX_APPLE9_RENDER_VBO_RUNTIME_A_SIZE            0x00100u
#define AGX_APPLE9_RENDER_VBO_RUNTIME_B_SIZE            0x000c0u
#define AGX_APPLE9_RENDER_TEMPLATE_TAIL_A               0x02600u
#define AGX_APPLE9_RENDER_TEMPLATE_TAIL_A_SIZE          0x08680u
#define AGX_APPLE9_RENDER_TEMPLATE_TAIL_B               0x0ac80u
#define AGX_APPLE9_RENDER_TEMPLATE_TAIL_B_SIZE          0x01ac0u
#define AGX_APPLE9_RENDER_TEMPLATE_TAIL_C               0x0c740u
#define AGX_APPLE9_RENDER_TEMPLATE_TAIL_C_SIZE          0x00180u

/*
 * The caller-owned compatibility image contains five compact references back
 * into the same logical USC package. They occupy matching instruction fields
 * in the background, load, and store programs. The normal G16 path keeps
 * these values unchanged and aliases each selected storage BO at the native
 * logical package address because the VDM entry token itself is not freely
 * relocatable. This helper remains an offline/research primitive for testing
 * the independently relocatable child references.
 *
 * Keep this deliberately narrow and fail closed.  A broad integer scan is
 * unsafe because Apple9 instruction bytes frequently resemble addresses.
 */
struct apple9_render_self_relocation {
   uint32_t offset;
   uint32_t relative;
};

static const struct apple9_render_self_relocation
   apple9_render_self_relocations[] = {
      {0x230004, 0x080000}, {0x230210, 0x230401}, {0x230244, 0x08000c},
      {0x230450, 0x230401}, {0x230484, 0x080018},
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

/* Sparse compiler table consumed by the fixed VBO launch path.  This is
 * caller-owned output from the public interleaved-VBO Metal probe, expressed
 * as sized records rather than retaining a mutable submission page.  The
 * table is invariant across the independently captured VBO layouts. */
struct apple9_render_sparse_qword {
   uint16_t offset;
   uint64_t value;
};

static const struct apple9_render_sparse_qword apple9_render_vbo_fixed_table[] =
   {
      {0x0000, UINT64_C(0x0000000000000040)},
      {0x0020, UINT64_C(0x0000035b60000000)},
      {0x0040, UINT64_C(0x0000000000000040)},
      {0x0080, UINT64_C(0x0000000000000040)},
      {0x00c0, UINT64_C(0x0000000000000040)},
      {0x00e0, UINT64_C(0x0000035b60000000)},
      {0x0100, UINT64_C(0x0000000000000040)},
      {0x0140, UINT64_C(0x0000000000000040)},
      {0x0160, UINT64_C(0x0000aaabfffeffff)},
      {0x0180, UINT64_C(0x0000000000000040)},
      {0x01a0, UINT64_C(0x4b00000050001c00)},
      {0x01a8, UINT64_C(0x14000a0505030201)},
      {0x01b0, UINT64_C(0xffff3fff4f800000)},
      {0x01b8, UINT64_C(0xfe00fe0001ff0000)},
      {0x01c0, UINT64_C(0x0000000000000080)},
      {0x0200, UINT64_C(0x0000035b60000000)},
      {0x0240, UINT64_C(0x0000000000000080)},
      {0x0280, UINT64_C(0x0000035b60000000)},
      {0x02c0, UINT64_C(0x0000000000000100)},
      {0x02e0, UINT64_C(0x00007fff0000ffff)},
      {0x02e8, UINT64_C(0xffffff80ffff8000)},
      {0x02f0, UINT64_C(0x40000000040003ff)},
      {0x02f8, UINT64_C(0x0000010200000400)},
      {0x0300, UINT64_C(0x0000010600000104)},
      {0x0308, UINT64_C(0x0000010c0000010a)},
      {0x0310, UINT64_C(0x000001100000010e)},
      {0x0318, UINT64_C(0x0000011400000112)},
      {0x0320, UINT64_C(0x0000011a00000118)},
      {0x0328, UINT64_C(0x0000011e0000011c)},
      {0x0330, UINT64_C(0x0000012200000120)},
      {0x0338, UINT64_C(0x0000012800000126)},
      {0x0340, UINT64_C(0x0000012c0000012a)},
      {0x0348, UINT64_C(0x000001300000012e)},
      {0x0350, UINT64_C(0x0000013600000134)},
      {0x0358, UINT64_C(0x0000013a00000138)},
      {0x0380, UINT64_C(0x0000035b60000000)},
      {0x03c0, UINT64_C(0x00000000000000c0)},
      {0x03e0, UINT64_C(0x0110010c01080104)},
      {0x03e8, UINT64_C(0x012401220120011c)},
      {0x03f0, UINT64_C(0x01340130012c0128)},
      {0x03f8, UINT64_C(0x01440140013c0138)},
      {0x0400, UINT64_C(0x01540150014c0148)},
      {0x0408, UINT64_C(0x01640160015c0158)},
      {0x0410, UINT64_C(0x017c0178016c0168)},
      {0x0418, UINT64_C(0x018801840180017e)},
      {0x0420, UINT64_C(0x019801940190018c)},
      {0x0428, UINT64_C(0x01a801a401a0019c)},
      {0x0430, UINT64_C(0x01b801b401b001ac)},
      {0x0438, UINT64_C(0x01c801c401c001bc)},
      {0x0440, UINT64_C(0x01dc01da01d801d4)},
      {0x0448, UINT64_C(0x01ec01e801e401e0)},
      {0x0450, UINT64_C(0x01fc01f801f401f0)},
      {0x0458, UINT64_C(0x020c020802040200)},
      {0x0480, UINT64_C(0x0000000000000080)},
      {0x04a0, UINT64_C(0x0001ffffffffffff)},
      {0x04a8, UINT64_C(0xffff07ff04000000)},
      {0x04b0, UINT64_C(0x0fffffff00010000)},
      {0x04b8, UINT64_C(0xfffff0ffff3fffff)},
      {0x04c0, UINT64_C(0x200000001bffffff)},
      {0x04c8, UINT64_C(0x7000000028000000)},
      {0x04d0, UINT64_C(0xfffe0000f0000000)},
      {0x0500, UINT64_C(0x0000000000000040)},
      {0x0520, UINT64_C(0x0000000080000000)},
      {0x0540, UINT64_C(0x000000000000fac0)},
};

/* Fields in the fixed compiler-pipeline records that describe the
 * interleaved vertex-fetch envelope.  The compatibility image was authored
 * for inline vertices and retains the smaller 0x08/0x0c/0x4c forms at these
 * positions.  Two independent T8132 VBO captures show the corresponding
 * fetch records use 0x0a/0x0e/0x4f.  The repeated 0x1b selectors are the
 * fixed-runtime compact target: the package-local 0x31 value moves down by
 * 0xb0000 / 0x8000 = 0x16 chunks when the runtime is published at +0xd8000.
 */
struct apple9_render_sparse_byte {
   uint16_t offset;
   uint8_t value;
};

static const struct apple9_render_sparse_byte
   apple9_render_interleaved_vbo_fixed_pipeline[] = {
      {0x0014, 0x0a}, {0x00c6, 0x1b}, {0x0254, 0x0a},
      {0x0306, 0x1b}, {0x0494, 0x0e}, {0x0537, 0x4f},
};

static void
apple9_build_render_vbo_fixed_table(uint8_t *page)
{
   memset(page, 0, 0x4000);
   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_vbo_fixed_table); ++i)
      apple9_put_u64(page + apple9_render_vbo_fixed_table[i].offset,
                     apple9_render_vbo_fixed_table[i].value);
}

/*
 * Apple9 keeps the compiler archive and its small resource-layout graph in
 * one fixed USC aperture. Metal does not select another usc_exec_base for a
 * vertex-fetch pipeline. Instead, adding a VBO inserts one resource slot,
 * shifts the two descriptor/top-record pairs by one 0x100-byte slot, and
 * retargets the package resource record to the shifted graph. The actual VBO
 * is named independently by the resource qword at +0xa0.
 *
 * Keep one graph in every immutable package generation. At the synchronized
 * submit boundary, render_cache_bind() installs the selected graph into the
 * fixed resident USC archive together with its package-local command state.
 */
static void
apple9_build_render_compiler_state(
   uint8_t *package, const struct agx_apple9_render_pipeline *pipeline)
{
   if (!pipeline->vertex_prolog.binary)
      return;

   uint8_t *state = package + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET;
   const uint32_t allocation_size =
      ALIGN_POT(pipeline->vertex_buffer_size, 0x100);
   assert(allocation_size && allocation_size <= 0x2000);

   /* Metal suballocates caller vertex buffers beginning at +0x200.  This is
    * one compiler-owned heap, not four independently movable records.  The
    * dense lookup table at +0x2500 and every byte after it move by the same
    * rounded allocation extent.  In particular, a one-slot allocation spills
    * the last 0x100 bytes of the original first page into the next page; only
    * copying 0x4000 bytes leaves the hardware with a truncated table. */
   memmove(state + 0x2500 + allocation_size, state + 0x2500,
           AGX_APPLE9_RENDER_COMPILER_STATE_SIZE - 0x2500 - allocation_size);
   memset(state + 0x2500, 0, allocation_size);

   /* The two top records are adjacent, while the two attachment records have
    * an empty 0x100-byte slot between them.  Clear both vacated attachment
    * slots after copying backwards; retaining the old second record is not a
    * valid heap graph. */
   memmove(state + 0x1a00 + allocation_size, state + 0x1a00, 0x100);
   memmove(state + 0x1900 + allocation_size, state + 0x1900, 0x100);
   memset(state + 0x1900, 0, allocation_size);

   memmove(state + 0x0400 + allocation_size, state + 0x0400, 0x100);
   memmove(state + 0x0200 + allocation_size, state + 0x0200, 0x100);
   memset(state + 0x0200, 0, allocation_size);
   memset(state + 0x0400, 0, allocation_size);

   /* Enabling vertex fetch changes the two attachment layout records from
    * ordinary-store descriptors to the compressed form consumed by Metal's
    * vertex compiler envelope.  The target and metadata addresses remain the
    * source-built G16 attachment graph.  These are complete record values;
    * the mode is not a single flag that can be ORed into the inline records. */
   const uint64_t attachment = UINT64_C(0x10000080000) >> 4;
   const uint64_t metadata =
      (UINT64_C(0x10000000000) + AGX_APPLE9_RENDER_FIXED_VBO_METADATA_OFFSET) >>
      4;
   const uint32_t descriptor_offsets[] = {
      0x0200 + allocation_size,
      0x0400 + allocation_size,
   };
   const uint64_t descriptors[] = {
      UINT64_C(0x08030010060a0a22),
      UINT64_C(0x0800300100c60a22),
   };
   for (unsigned i = 0; i < ARRAY_SIZE(descriptor_offsets); ++i) {
      uint8_t *record = state + descriptor_offsets[i] + 0x20;
      apple9_put_u64(record + 0x00, descriptors[i]);
      apple9_put_u64(record + 0x08, attachment | (UINT64_C(1) << 63));
      apple9_put_u64(record + 0x10, metadata);
   }

   const uint64_t usc_exec_base = UINT64_C(0x10000000000);
   apple9_put_u64(package + AGX_APPLE9_RENDER_RESOURCE_OFFSET + 0x30,
                  usc_exec_base + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET +
                     0x1a00 + allocation_size);
   apple9_put_u64(package + AGX_APPLE9_RENDER_RESOURCE_OFFSET + 0x38,
                  usc_exec_base + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET +
                     0x1900 + allocation_size);
   apple9_put_u64(
      package + AGX_APPLE9_RENDER_VERTEX_RESOURCE_OFFSET,
      usc_exec_base + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET + 0x200);
}

bool
agx_apple9_layout_render_archive(
   const struct agx_apple9_render_pipeline *pipeline,
   struct agx_apple9_render_archive_layout *layout)
{
   if (!pipeline || !layout || !pipeline->fragment.binary ||
       !pipeline->fragment.binary_size || !pipeline->vertex.binary ||
       !pipeline->vertex.binary_size || pipeline->vertex_prolog.binary ||
       pipeline->fragment.binary_size > AGX_APPLE9_RENDER_ARCHIVE_SIZE ||
       pipeline->vertex.binary_size > AGX_APPLE9_RENDER_ARCHIVE_SIZE)
      return false;

   /* Helper-internal references have no complete relocation model. Keep the
    * opaque closure at its original offsets and place our mains in the empty
    * archive reservation. Only the three measured entry calls are patched. */
   const uint32_t prefix =
      AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE + AGX_APPLE9_RENDER_CONSTANT_SIZE;
   uint64_t cursor = AGX_APPLE9_RENDER_TEMPLATE_TAIL_C +
                     AGX_APPLE9_RENDER_TEMPLATE_TAIL_C_SIZE;
   uint64_t fs_size = ALIGN_POT(pipeline->fragment.binary_size, 0x40) + prefix;
   uint64_t vs_size = ALIGN_POT(pipeline->vertex.binary_size, 0x40) + prefix;
   if (fs_size + vs_size > AGX_APPLE9_RENDER_ARCHIVE_SIZE - cursor - 0x80)
      return false;
   memset(layout, 0, sizeof(*layout));
   layout->fragment_block = cursor;
   layout->fragment_main = cursor + prefix;
   layout->fragment_block_size = fs_size;
   cursor += fs_size;
   layout->vertex_prolog_block = cursor;
   layout->vertex_prolog_main = cursor + prefix;
   layout->vertex_prolog_block_size = vs_size;
   cursor += vs_size;
   layout->vertex_main_block = AGX_APPLE9_RENDER_TEMPLATE_VERTEX_ADAPTER_BLOCK;
   layout->vertex_main = layout->vertex_main_block + prefix;
   layout->vertex_main_block_size =
      AGX_APPLE9_RENDER_TEMPLATE_VERTEX_ADAPTER_SIZE;
   if (!apple9_archive_call(layout->fragment_main, &layout->fragment_call) ||
       !apple9_archive_call(layout->vertex_prolog_main,
                            &layout->vertex_prolog_call) ||
       !apple9_archive_call(layout->vertex_main, &layout->vertex_call))
      return false;
   layout->end = cursor;
   if (getenv("AGX_APPLE9_PACKAGE_TRACE") != NULL) {
      fprintf(stderr,
              "APPLE9_RENDER_ARCHIVE fs_bytes=%#zx fs_block=%#x/%#x "
              "fs_main=%#x fs_call=%#x prolog_bytes=%#zx "
              "prolog=%#x/%#x call=%#x vs_bytes=%#zx "
              "vs_block=%#x/%#x vs_main=%#x vs_call=%#x "
              "end=%#x\n",
              pipeline->fragment.binary_size, layout->fragment_block,
              layout->fragment_block_size, layout->fragment_main,
              layout->fragment_call, pipeline->vertex_prolog.binary_size,
              layout->vertex_prolog_block, layout->vertex_prolog_block_size,
              layout->vertex_prolog_call, pipeline->vertex.binary_size,
              layout->vertex_main_block, layout->vertex_main_block_size,
              layout->vertex_main, layout->vertex_call, layout->end);
   }
   return true;
}

static void
apple9_build_render_vertex_constant_program(uint8_t *out)
{
   static const uint8_t header[0x0e] = {
      0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00,
      0x00, 0x60, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   memset(out, 0, AGX_APPLE9_RENDER_CONSTANT_SIZE);
   memcpy(out, header, sizeof(header));
   for (unsigned offset = sizeof(header);
        offset < AGX_APPLE9_RENDER_CONSTANT_SIZE; offset += 2)
      apple9_put_u16(out + offset, 0x0006);
}

static void
apple9_build_render_program_block(uint8_t *archive, uint32_t block,
                                  uint32_t block_size,
                                  const uint8_t *constant_program,
                                  const void *main, size_t main_size)
{
   memset(archive + block, 0, block_size);
   apple9_put_u32(archive + block, block_size);
   memcpy(archive + block + AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE,
          constant_program, AGX_APPLE9_RENDER_CONSTANT_SIZE);
   memcpy(archive + block + AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE +
             AGX_APPLE9_RENDER_CONSTANT_SIZE,
          main, main_size);
}

static bool
apple9_build_render_archive(uint8_t *package,
                            const struct agx_apple9_render_pipeline *pipeline,
                            struct agx_apple9_render_archive_layout *layout)
{
   if (!agx_apple9_layout_render_archive(pipeline, layout))
      return false;

   /* Fail closed if the embedded compatibility fixture changes format. */
   if (apple9_get_u32(package) != AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE ||
       apple9_get_u32(package +
                      AGX_APPLE9_RENDER_TEMPLATE_VERTEX_PROLOG_BLOCK) !=
          AGX_APPLE9_RENDER_TEMPLATE_VERTEX_PROLOG_SIZE ||
       apple9_get_u32(package +
                      AGX_APPLE9_RENDER_TEMPLATE_VERTEX_ADAPTER_BLOCK) !=
          AGX_APPLE9_RENDER_TEMPLATE_VERTEX_ADAPTER_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_EMPTY_BLOCK) !=
          AGX_APPLE9_RENDER_TEMPLATE_EMPTY_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_ADAPTER_A) !=
          AGX_APPLE9_RENDER_TEMPLATE_ADAPTER_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_ADAPTER_B) !=
          AGX_APPLE9_RENDER_TEMPLATE_ADAPTER_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_LIBRARY_A) !=
          AGX_APPLE9_RENDER_TEMPLATE_LIBRARY_A_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_LIBRARY_B) !=
          AGX_APPLE9_RENDER_TEMPLATE_LIBRARY_B_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_COLOR_EPILOG) !=
          AGX_APPLE9_RENDER_TEMPLATE_COLOR_EPILOG_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_TAIL_A) !=
          AGX_APPLE9_RENDER_TEMPLATE_TAIL_A_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_TAIL_B) !=
          AGX_APPLE9_RENDER_TEMPLATE_TAIL_B_SIZE ||
       apple9_get_u32(package + AGX_APPLE9_RENDER_TEMPLATE_TAIL_C) !=
          AGX_APPLE9_RENDER_TEMPLATE_TAIL_C_SIZE ||
       apple9_get_u24(package + AGX_APPLE9_RENDER_FRAGMENT_CALL_OFFSET) !=
          AGX_APPLE9_RENDER_TEMPLATE_FRAGMENT_CALL ||
       apple9_get_u24(package + AGX_APPLE9_RENDER_VERTEX_PROLOG_CALL_OFFSET) !=
          AGX_APPLE9_RENDER_TEMPLATE_VERTEX_PROLOG_CALL ||
       apple9_get_u24(package + AGX_APPLE9_RENDER_VERTEX_CALL_OFFSET) !=
          AGX_APPLE9_RENDER_TEMPLATE_VERTEX_CALL)
      return false;

   uint8_t fragment_constant[AGX_APPLE9_RENDER_CONSTANT_SIZE];
   uint8_t vertex_constant[AGX_APPLE9_RENDER_CONSTANT_SIZE];
   apple9_build_sentinel_constant_program(fragment_constant, 30);
   apple9_build_render_vertex_constant_program(vertex_constant);
   /* Retire both original API bodies. An accidentally stale call terminates
    * instead of silently executing a Metal-produced shader main. */
   memset(package + 0x3c0, 0, 0x80);
   apple9_put_u32(package + 0x3c0, 0x0e);
   memset(package + 0x540, 0, 0x100);
   apple9_put_u32(package + 0x540, 0x0e);
   memset(package + layout->fragment_block, 0,
          AGX_APPLE9_RENDER_ARCHIVE_SIZE - layout->fragment_block);
   apple9_build_render_program_block(
      package, layout->fragment_block, layout->fragment_block_size,
      fragment_constant, pipeline->fragment.binary,
      pipeline->fragment.binary_size);
   apple9_build_render_program_block(
      package, layout->vertex_prolog_block, layout->vertex_prolog_block_size,
      vertex_constant, pipeline->vertex.binary, pipeline->vertex.binary_size);
   apple9_put_u24(package + AGX_APPLE9_RENDER_FRAGMENT_CALL_OFFSET,
                  layout->fragment_call);
   apple9_put_u24(package + AGX_APPLE9_RENDER_VERTEX_PROLOG_CALL_OFFSET,
                  layout->vertex_prolog_call);
   apple9_put_u24(package + AGX_APPLE9_RENDER_VERTEX_CALL_OFFSET,
                  layout->vertex_call);

   /* Metal represents the remaining archive capacity as one empty program
    * block ending at the fixed zero header at +0xffc0.  Its size therefore
    * changes whenever dynamic stage or vertex-fetch blocks change.  A fixed
    * 0x80 record happened to be tolerated by the inline-vertex path but is
    * not the compiler archive grammar consumed by vertex fetch. */
   const uint32_t zero_header = AGX_APPLE9_RENDER_ARCHIVE_SIZE - 0x40;
   if (layout->end > zero_header - 0x40) {
      return false;
   }
   apple9_put_u32(package + layout->end, zero_header - layout->end);
   apple9_build_render_compiler_state(package, pipeline);
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
#ifndef HAVE_COMPRESSION
   (void)mapping;
   (void)mapping_size;
   (void)pipeline;
   return false;
#else
   if (!mapping || mapping_size != AGX_APPLE9_RENDER_PACKAGE_SIZE ||
       !pipeline || !pipeline->vertex.binary || !pipeline->fragment.binary)
      return false;

   if (!apple9_render_blobs_available())
      return false;

   struct agx_apple9_render_archive_layout layout;
   if (!agx_apple9_layout_render_archive(pipeline, &layout))
      return false;

   if (!util_compress_inflate(apple9_g16_render_package_zst.data,
                              apple9_g16_render_package_zst.size, mapping,
                              mapping_size))
      return false;

   if (!apple9_build_render_archive(mapping, pipeline, &layout))
      return false;
   apple9_build_cf_bindings((uint8_t *)mapping + APPLE9_CF_BINDINGS,
                            pipeline->vertex.varying_components);
   return true;
#endif
}

bool
agx_apple9_relocate_render_package_base(void *mapping, size_t mapping_size,
                                        uint32_t package_offset)
{
   if (!mapping || mapping_size != AGX_APPLE9_RENDER_PACKAGE_SIZE ||
       (package_offset & 0x3f))
      return false;

   uint8_t *package = mapping;
   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_self_relocations); ++i) {
      const struct apple9_render_self_relocation *reloc =
         &apple9_render_self_relocations[i];
      if (reloc->relative > UINT32_MAX - package_offset)
         return false;

      uint32_t original = AGX_APPLE9_RENDER_PACKAGE_OFFSET + reloc->relative;
      uint32_t relocated = package_offset + reloc->relative;
      uint32_t current = apple9_get_u32(package + reloc->offset);
      if (current != original && current != relocated)
         return false;
   }

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

   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_self_relocations); ++i) {
      const struct apple9_render_self_relocation *reloc =
         &apple9_render_self_relocations[i];
      apple9_put_u32(package + reloc->offset, package_offset + reloc->relative);
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
   result->varying_components = pipeline->vertex.varying_components;

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
   if (pipeline->vertex_prolog.binary) {
      /* The vertex-fetch compiler envelope selects the same bind-group
       * record as the inline path but changes its stage routing byte.  This
       * is the sole bind-group difference in the two independent T8132 VBO
       * captures with the three-scalar interface. */
      ((uint8_t *)agx_bo_map(state))[AGX_APPLE9_BIND_GROUP_OFFSET + 0x15] =
         0x60;
   }

   result->bo = package;
   result->state_bo = state;
   result->vertex_buffer = pipeline->vertex_buffer;
   result->vertex_buffer_size = pipeline->vertex_buffer_size;
   result->vertex_prolog_call_offset =
      pipeline->vertex_prolog.binary
         ? AGX_APPLE9_RENDER_VBO_PROLOG_CALL_OFFSET
         : AGX_APPLE9_RENDER_VERTEX_PROLOG_CALL_OFFSET;
   bool layout_ok =
      agx_apple9_layout_render_archive(pipeline, &result->archive);
   assert(layout_ok);
   list_inithead(&result->link);
   return result;
}

void
agx_apple9_render_package_destroy(struct agx_device *dev,
                                  struct agx_apple9_render_package *package)
{
   if (!package)
      return;

   assert(!package->active_batches);
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

   if (package->varying_components != pipeline->vertex.varying_components)
      return false;
   if (package->vertex_buffer != pipeline->vertex_buffer)
      return false;
   if (package->vertex_buffer_size != pipeline->vertex_buffer_size)
      return false;

   struct agx_apple9_render_archive_layout layout;
   if (!agx_apple9_layout_render_archive(pipeline, &layout))
      return false;

   const uint8_t *mapping = agx_bo_map(package->bo);
   if (apple9_get_u32(mapping + layout.fragment_block) !=
          layout.fragment_block_size ||
       apple9_get_u32(mapping + layout.vertex_prolog_block) !=
          layout.vertex_prolog_block_size ||
       apple9_get_u32(mapping + layout.vertex_main_block) !=
          layout.vertex_main_block_size ||
       apple9_get_u24(mapping + AGX_APPLE9_RENDER_FRAGMENT_CALL_OFFSET) !=
          layout.fragment_call ||
       apple9_get_u24(mapping + package->vertex_prolog_call_offset) !=
          layout.vertex_prolog_call ||
       apple9_get_u24(mapping + AGX_APPLE9_RENDER_VERTEX_CALL_OFFSET) !=
          layout.vertex_call ||
       memcmp(mapping + layout.fragment_main, pipeline->fragment.binary,
              pipeline->fragment.binary_size))
      return false;

   if (pipeline->vertex_prolog.binary) {
      if (memcmp(mapping + layout.vertex_prolog_main,
                 pipeline->vertex_prolog.binary,
                 pipeline->vertex_prolog.binary_size) ||
          memcmp(mapping + layout.vertex_main, pipeline->vertex.binary,
                 pipeline->vertex.binary_size))
         return false;
   } else if (memcmp(mapping + layout.vertex_prolog_main,
                     pipeline->vertex.binary, pipeline->vertex.binary_size)) {
      return false;
   }

   if (pipeline->vertex_prolog.binary &&
       apple9_get_u64(mapping + AGX_APPLE9_RENDER_VERTEX_RESOURCE_OFFSET) !=
          UINT64_C(0x10000018200))
      return false;

   for (size_t i = pipeline->fragment.binary_size;
        i < layout.fragment_block_size - AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE -
               AGX_APPLE9_RENDER_CONSTANT_SIZE;
        ++i) {
      if (mapping[layout.fragment_main + i] != 0)
         return false;
   }
   if (pipeline->vertex_prolog.binary) {
      for (size_t i = pipeline->vertex.binary_size;
           i < layout.vertex_main_block_size -
                  AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE -
                  AGX_APPLE9_RENDER_CONSTANT_SIZE;
           ++i) {
         if (mapping[layout.vertex_main + i] != 0)
            return false;
      }
      for (size_t i = pipeline->vertex_prolog.binary_size;
           i < layout.vertex_prolog_block_size -
                  AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE -
                  AGX_APPLE9_RENDER_CONSTANT_SIZE;
           ++i) {
         if (mapping[layout.vertex_prolog_main + i] != 0)
            return false;
      }
   } else {
      for (size_t i = pipeline->vertex.binary_size;
           i < layout.vertex_prolog_block_size -
                  AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE -
                  AGX_APPLE9_RENDER_CONSTANT_SIZE;
           ++i) {
         if (mapping[layout.vertex_prolog_main + i] != 0)
            return false;
      }
   }
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
   bool vertex_fetch =
      apple9_get_u64(package + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET +
                     0x0320) == UINT64_C(0x08030010060a0a22) &&
      apple9_get_u64(package + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET +
                     0x0520) == UINT64_C(0x0800300100c60a22);

   /* The two compiler-heap attachment records move when vertex storage is
    * inserted.  They also use a distinct compressed descriptor grammar; the
    * raw-twiddled field patcher below must not rewrite their format word.  We
    * have a byte-exact 257x193 oracle today, so fail closed for other VBO
    * dimensions until their compressed dimension fields are measured. */
   if (vertex_fetch && (width != 257 || height != 193))
      return false;

   for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_regions); ++i) {
      const struct agx_apple9_render_region *region = &apple9_render_regions[i];
      if (region->kind != AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE &&
          region->kind != AGX_APPLE9_RENDER_REGION_COLOR_BUFFER)
         continue;

      if (vertex_fetch && i < 2) {
         unsigned offset = region->offset + 0x100;
         apple9_put_u64(package + offset + 0x08,
                        (color_target >> 4) | (UINT64_C(1) << 63));
         continue;
      }

      apple9_patch_render_target(
         package, region->offset,
         region->kind == AGX_APPLE9_RENDER_REGION_COLOR_TEXTURE, color_target,
         width, height);
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

   package->sealed = true;
   package->color_target = color_target;
   package->width = width;
   package->height = height;
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
      AGX_BO_EXEC | AGX_BO_WRITEBACK, "Apple9 fixed render archive");
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
   cache->resident_archive_limit = AGX_APPLE9_RENDER_ARCHIVE_SIZE;
   const char *test_limit = getenv("AGX_APPLE9_RENDER_ARCHIVE_TEST_LIMIT");
   if (test_limit && test_limit[0]) {
      char *end = NULL;
      unsigned long parsed = strtoul(test_limit, &end, 0);
      if (end && !end[0] && !(parsed & 0x3f) &&
          parsed >= AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE &&
          parsed <= AGX_APPLE9_RENDER_ARCHIVE_SIZE)
         cache->resident_archive_limit = parsed;
   }
   list_inithead(&cache->packages);
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

struct agx_apple9_render_package *
agx_apple9_render_cache_get(struct agx_apple9_render_cache *cache,
                            const struct agx_apple9_render_pipeline *pipeline,
                            uint64_t color_target, unsigned width,
                            unsigned height)
{
   if (!cache || !pipeline)
      return NULL;

   list_for_each_entry(struct agx_apple9_render_package, package,
                       &cache->packages, link) {
      if (package->sealed && package->color_target == color_target &&
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
                                                      width, height)) {
      agx_apple9_render_package_destroy(cache->dev, package);
      return NULL;
   }

   package->last_used = ++cache->use_serial;
   list_addtail(&package->link, &cache->packages);
   cache->package_count++;
   return package;
}

bool
agx_apple9_render_cache_bind(struct agx_apple9_render_cache *cache,
                             struct agx_apple9_render_package *package)
{
   if (!cache || !package || !package->bo || !package->state_bo ||
       !package->sealed || !cache->dev->apple9_render_fixed_usc)
      return false;

   if (cache->current == package && !cache->fixed_usc_dirty)
      return true;

   uint8_t *resident = agx_bo_map(cache->resident_bo);
   uint8_t *fixed_usc = agx_bo_map(cache->dev->apple9_render_fixed_usc);
   const uint8_t *source = agx_bo_map(package->bo);
   const bool force_authored_generation =
      !cache->installing_draws &&
      getenv("AGX_APPLE9_RENDER_FORCE_AUTHORED_GENERATION") != NULL;

   /*
    * Metal keeps one physical archive at the queue's fixed USC base and
    * interns stage programs into it.  The compatibility package has the same
    * split: the first 64 KiB is executable archive, while caller launch/state
    * in the remaining range selects three archive entries.  Preserve the
    * resident archive and install only the selected command state.
    */
   if (!cache->current || force_authored_generation) {
      if (force_authored_generation && cache->current) {
         /* Diagnostic isolation for compact archive calls and Dynamic
          * Caching program identity.  Metal-authored standalone pipelines
          * place their fragment main at the first archive slot.  Reinstall
          * the selected immutable source generation verbatim so a test can
          * distinguish that contract from Mesa's ordinary cross-pipeline
          * block interning without changing any shader, Work, or fixed state
          * bytes. */
         list_for_each_entry(struct agx_apple9_render_package, candidate,
                             &cache->packages, link) {
            candidate->fragment_call = 0;
            candidate->vertex_prolog_call = 0;
            candidate->vertex_call = 0;
         }
      }
      memcpy(resident, source, AGX_APPLE9_RENDER_PACKAGE_SIZE);
      cache->resident_archive_next = package->archive.end;
      package->fragment_call = package->archive.fragment_call;
      package->vertex_prolog_call = package->archive.vertex_prolog_call;
      package->vertex_call = package->archive.vertex_call;
   } else {
      if (!package->fragment_call) {
         list_for_each_entry(struct agx_apple9_render_package, candidate,
                             &cache->packages, link) {
            if (candidate->fragment_call &&
                candidate->archive.fragment_block_size ==
                   package->archive.fragment_block_size &&
                !memcmp((const uint8_t *)agx_bo_map(candidate->bo) +
                           candidate->archive.fragment_block,
                        source + package->archive.fragment_block,
                        package->archive.fragment_block_size)) {
               package->fragment_call = candidate->fragment_call;
               break;
            }
         }
      }
      if (!package->vertex_call) {
         list_for_each_entry(struct agx_apple9_render_package, candidate,
                             &cache->packages, link) {
            if (candidate->vertex_call &&
                candidate->archive.vertex_main_block_size ==
                   package->archive.vertex_main_block_size &&
                !memcmp((const uint8_t *)agx_bo_map(candidate->bo) +
                           candidate->archive.vertex_main_block,
                        source + package->archive.vertex_main_block,
                        package->archive.vertex_main_block_size)) {
               package->vertex_call = candidate->vertex_call;
               break;
            }
         }
      }
      if (!package->vertex_prolog_call) {
         list_for_each_entry(struct agx_apple9_render_package, candidate,
                             &cache->packages, link) {
            if (candidate->vertex_prolog_call &&
                candidate->archive.vertex_prolog_block_size ==
                   package->archive.vertex_prolog_block_size &&
                !memcmp((const uint8_t *)agx_bo_map(candidate->bo) +
                           candidate->archive.vertex_prolog_block,
                        source + package->archive.vertex_prolog_block,
                        package->archive.vertex_prolog_block_size)) {
               package->vertex_prolog_call = candidate->vertex_prolog_call;
               break;
            }
         }
      }

      const struct {
         uint32_t source_block;
         uint32_t block_size;
         uint32_t *call;
      } stages[] = {
         {package->archive.fragment_block, package->archive.fragment_block_size,
          &package->fragment_call},
         {package->archive.vertex_prolog_block,
          package->archive.vertex_prolog_block_size,
          &package->vertex_prolog_call},
         {package->archive.vertex_main_block,
          package->archive.vertex_main_block_size, &package->vertex_call},
      };
      bool appended = true;
      for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
         if (*stages[i].call)
            continue;
         uint32_t block = cache->resident_archive_next;
         uint32_t main = block + AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE +
                         AGX_APPLE9_RENDER_CONSTANT_SIZE;
         if (block > cache->resident_archive_limit ||
             stages[i].block_size > cache->resident_archive_limit - block ||
             !apple9_archive_call(main, stages[i].call)) {
            appended = false;
            break;
         }
         memcpy(resident + block, source + stages[i].source_block,
                stages[i].block_size);
         cache->resident_archive_next = block + stages[i].block_size;
      }

      if (!appended) {
         /* A batch must never evict programs named by its earlier draws. */
         if (cache->installing_draws)
            return false;
         /* Compact archive generation rollover.  The previous user has
          * already retired before cache_bind, so retain the physical BO while
          * rebuilding its contents around the selected pipeline. */
         if (getenv("AGX_APPLE9_PACKAGE_TRACE")) {
            fprintf(stderr,
                    "APPLE9_RENDER_CACHE_ROLLOVER next=%#x need=%#x "
                    "limit=%#x\n",
                    cache->resident_archive_next,
                    package->archive.fragment_block_size +
                       package->archive.vertex_prolog_block_size +
                       package->archive.vertex_main_block_size,
                    cache->resident_archive_limit);
         }
         list_for_each_entry(struct agx_apple9_render_package, candidate,
                             &cache->packages, link) {
            candidate->fragment_call = 0;
            candidate->vertex_prolog_call = 0;
            candidate->vertex_call = 0;
         }
         memcpy(resident, source, AGX_APPLE9_RENDER_PACKAGE_SIZE);
         cache->resident_archive_next = package->archive.end;
         package->fragment_call = package->archive.fragment_call;
         package->vertex_prolog_call = package->archive.vertex_prolog_call;
         package->vertex_call = package->archive.vertex_call;
      } else {
         memcpy(
            resident + AGX_APPLE9_RENDER_ARCHIVE_SIZE,
            source + AGX_APPLE9_RENDER_ARCHIVE_SIZE,
            AGX_APPLE9_RENDER_PACKAGE_SIZE - AGX_APPLE9_RENDER_ARCHIVE_SIZE);
         apple9_put_u24(resident + AGX_APPLE9_RENDER_FRAGMENT_CALL_OFFSET,
                        package->fragment_call);
         apple9_put_u24(resident + package->vertex_prolog_call_offset,
                        package->vertex_prolog_call);
         apple9_put_u24(resident + AGX_APPLE9_RENDER_VERTEX_CALL_OFFSET,
                        package->vertex_call);
      }
   }
   uint8_t *resident_state = agx_bo_map(cache->resident_state_bo);
   const uint8_t *package_state = agx_bo_map(package->state_bo);
   /* The former template-overlay shim supplied this context page. The
    * caller now owns it, including the null link at the end of bind0. */
   apple9_build_direct_bind0(resident_state, package->varying_components);
   memset(resident_state + 0x340, 0, 8);
   memcpy(resident_state +
             (AGX_APPLE9_RENDER_STATE_ADDRESS - AGX_APPLE9_RENDER_CONTEXT_BASE),
          package_state, AGX_APPLE9_RENDER_STATE_SIZE);
   if (package->vertex_buffer_size) {
      /* VBO VDM selects +0x0040 instead of the inline path's +0x4040.
       * Publish the first-page bind record at the context base, clear its
       * inline-path link at +0x340, and leave the original +0x4000 slot
       * empty.  Native keeps the remainder of the fixed-function image at
       * the ordinary addresses starting at +0x8000.  Duplicating the generic
       * first page at +0x4000 made the stale 0x1004 link live and selected a
       * non-VBO state path before TA could retire. */
      memcpy(resident_state, package_state, 0x4000);
      memset(resident_state + 0x340, 0, sizeof(uint64_t));
      memset(resident_state + 0x4000, 0, 0x4000);
   }

   /* The VBO launch wrapper is entered through the fixed USC arena and its
    * compact archive call is relative to that arena, not to the compatibility
    * VDM pipeline offset.  Native T8132 therefore carries the same selected
    * archive at USC_EXEC_BASE+0 as well as in the independently owned package
    * generation.  Install the complete executable archive atomically with
    * the command state; leaving these four pages zero makes TA/3D stall before
    * either queue retires even though the package-offset archive is valid. */
   if (package->vertex_buffer_size) {
      assert(cache->dev->apple9_render_fixed_usc->size >=
             AGX_APPLE9_RENDER_ARCHIVE_SIZE);
      memcpy(fixed_usc, resident, AGX_APPLE9_RENDER_ARCHIVE_SIZE);
   }

   /* Opaque helper/resource references still reach the complete fixed USC
    * window even when VDM selects the separately owned package. Publish the
    * entire selected view; the first 2 MiB alone leaves live callees absent. */
   assert(cache->dev->apple9_render_fixed_usc->size ==
          AGX_APPLE9_RENDER_PACKAGE_SIZE);
   memcpy(fixed_usc, resident, AGX_APPLE9_RENDER_PACKAGE_SIZE);

   /* The queue USC base is fixed. Install the selected compiler resource
    * graph into the base archive rather than changing usc_exec_base or
    * falling back to the shim's immutable client template. */
   memcpy(fixed_usc + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET,
          source + AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET,
          AGX_APPLE9_RENDER_COMPILER_STATE_SIZE);

   /* The vertex-fetch launch wrapper addresses this record with the compact
    * fixed selector 0x00a8.  Keep the independently owned package as the
    * immutable source generation, but publish the selected record into the
    * queue's fixed USC arena at the synchronized cache-bind boundary. */
   assert(cache->dev->apple9_render_fixed_usc->size >=
          AGX_APPLE9_RENDER_FIXED_RESOURCE_OFFSET +
             AGX_APPLE9_RENDER_FIXED_RESOURCE_SIZE);
   memcpy(fixed_usc + AGX_APPLE9_RENDER_FIXED_RESOURCE_OFFSET,
          source + AGX_APPLE9_RENDER_RESOURCE_OFFSET,
          AGX_APPLE9_RENDER_FIXED_RESOURCE_SIZE);

   /* Vertex fetch also enters a dense compiler-runtime image through compact
    * fixed-USC references.  The compatibility package retains this image at
    * +0x188000, displaced by the caller-owned attachment aperture, whereas
    * the live T8132 context selects +0xd8000.  Publish it as one coherent
    * immutable object; copying only the launch/resource leaves its callees
    * unmapped/zero and TA cannot retire. */
   if (package->vertex_buffer_size) {
      assert(cache->dev->apple9_render_fixed_usc->size >=
             AGX_APPLE9_RENDER_FIXED_RUNTIME_OFFSET +
                AGX_APPLE9_RENDER_FIXED_RUNTIME_SIZE);
      memcpy(fixed_usc + AGX_APPLE9_RENDER_FIXED_RUNTIME_OFFSET,
             source + AGX_APPLE9_RENDER_RUNTIME_SOURCE_OFFSET,
             AGX_APPLE9_RENDER_FIXED_RUNTIME_SIZE);

      /* The source range overlaps the package-owned vertex-call selector at
       * +0x18ca37.  That selector is patched to the selected archive main,
       * but Metal's fixed runtime view at +0xdca37 is an independent,
       * immutable compiler-library image and retains the template call
       * 0x0daa.  Copying the patched source byte-for-byte changed its first
       * opcode byte from 0xaa to 0x2a and produced a repeatable UL1C0 read
       * fault at a poison-looking address.  Restore the fixed view's own
       * selector after publishing the shared source range. */
      const uint32_t fixed_vertex_call_offset =
         AGX_APPLE9_RENDER_FIXED_RUNTIME_OFFSET +
         (AGX_APPLE9_RENDER_VERTEX_CALL_OFFSET -
          AGX_APPLE9_RENDER_RUNTIME_SOURCE_OFFSET);
      assert(fixed_vertex_call_offset + 3 <=
             AGX_APPLE9_RENDER_FIXED_RUNTIME_OFFSET +
                AGX_APPLE9_RENDER_FIXED_RUNTIME_SIZE);
      apple9_put_u24(fixed_usc + fixed_vertex_call_offset,
                     AGX_APPLE9_RENDER_TEMPLATE_VERTEX_CALL);
   }

   /* The VBO path has a second fixed-USC dependency: Metal keeps the launch
    * wrapper at +0x170000 even though our independently owned package stores
    * its mutable source image at +0x220000.  Inline vertices never consume
    * this location, which is why relocating the archive mains alone appeared
    * sufficient until the first real vertex fetch.  Publish the complete
    * page at the same generation boundary as its resource record. */
   if (package->vertex_buffer_size) {
      assert(cache->dev->apple9_render_fixed_usc->size >=
             AGX_APPLE9_RENDER_FIXED_VERTEX_LAUNCH_OFFSET +
                AGX_APPLE9_RENDER_FIXED_VERTEX_LAUNCH_SIZE);
      memcpy(fixed_usc + AGX_APPLE9_RENDER_FIXED_VERTEX_LAUNCH_OFFSET,
             source + AGX_APPLE9_RENDER_VERTEX_LAUNCH_OFFSET,
             AGX_APPLE9_RENDER_FIXED_VERTEX_LAUNCH_SIZE);

      /* The launch wrapper's resource, pipeline, and dimension records are
       * independently owned in the compatibility package but consumed from
       * Metal's fixed USC layout.  Publish all three views together. */
      apple9_build_render_vbo_fixed_table(
         fixed_usc + AGX_APPLE9_RENDER_FIXED_VBO_TABLE_OFFSET);
      memset(fixed_usc + AGX_APPLE9_RENDER_FIXED_VBO_METADATA_OFFSET, 0,
             0x4000);
      memset(fixed_usc + AGX_APPLE9_RENDER_FIXED_VBO_METADATA_OFFSET, 0x7f,
             0x1000);
      memcpy(fixed_usc + AGX_APPLE9_RENDER_FIXED_TARGET_GRAPH_OFFSET,
             source + AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET, 0x4000);
      memcpy(fixed_usc + AGX_APPLE9_RENDER_FIXED_PIPELINE_OFFSET,
             source + AGX_APPLE9_RENDER_PIPELINE_SOURCE_OFFSET, 0x8000);
      memcpy(fixed_usc + AGX_APPLE9_RENDER_FIXED_DIMENSIONS_OFFSET,
             source + AGX_APPLE9_RENDER_DIMENSIONS_SOURCE_OFFSET, 0x4000);

      /* Moving the target graph down by 0xb0000 moves its eight absolute
       * record pointers by the same amount. */
      const uint32_t fixed_shift =
         AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET -
         AGX_APPLE9_RENDER_FIXED_TARGET_GRAPH_OFFSET;
      for (unsigned i = 0;
           i < ARRAY_SIZE(apple9_render_absolute_self_relocations); ++i) {
         const struct apple9_render_self_relocation *reloc =
            &apple9_render_absolute_self_relocations[i];
         assert(reloc->offset >= fixed_shift);
         apple9_put_u64(
            fixed_usc + reloc->offset - fixed_shift,
            UINT64_C(0x10000000000) + reloc->relative - fixed_shift);
      }

      /* Three compact references in the fixed pipeline page name the dense
       * runtime copied from +0x188000 to +0xd8000.  This encoding advances by
       * eight for each byte of USC displacement, so the 0xb0000 move is a
       * 0x580000 subtraction.  The two +0x230401 fields are package-local and
       * remain unchanged, exactly as in the native 257x193 capture. */
      for (unsigned i = 0; i < ARRAY_SIZE(apple9_render_self_relocations);
           ++i) {
         const struct apple9_render_self_relocation *reloc =
            &apple9_render_self_relocations[i];
         if (reloc->relative >= 0x100000)
            continue;
         assert(reloc->offset >= fixed_shift);
         uint32_t at = reloc->offset - fixed_shift;
         apple9_put_u32(fixed_usc + at,
                        apple9_get_u32(fixed_usc + at) - (fixed_shift << 3));
      }

      for (unsigned i = 0;
           i < ARRAY_SIZE(apple9_render_interleaved_vbo_fixed_pipeline); ++i) {
         const struct apple9_render_sparse_byte *field =
            &apple9_render_interleaved_vbo_fixed_pipeline[i];
         assert(field->offset < 0x4000);
         fixed_usc[AGX_APPLE9_RENDER_FIXED_PIPELINE_OFFSET + field->offset] =
            field->value;
      }

      /* The fixed VBO pipeline is copied from the package's immutable source
       * image above, but the selected fragment archive call is installed in
       * the resident command image at cache-bind time.  Native switches
       * fragment pipelines by changing this call while retaining the same
       * archive and compiler state.  Publish the selected call into the
       * fixed-USC view as part of the same generation; otherwise VBO draws
       * continue to execute the template fragment entry (which happened to
       * make the pass-through shader work, while arbitrary fragment DAGs
       * produced retired but corrupt output).
       */
      const uint32_t fixed_fragment_call_offset =
         AGX_APPLE9_RENDER_FIXED_PIPELINE_OFFSET +
         (AGX_APPLE9_RENDER_FRAGMENT_CALL_OFFSET -
          AGX_APPLE9_RENDER_PIPELINE_SOURCE_OFFSET);
      assert(fixed_fragment_call_offset + 3 <=
             AGX_APPLE9_RENDER_FIXED_PIPELINE_OFFSET + 0x8000);
      apple9_put_u24(fixed_usc + fixed_fragment_call_offset,
                     package->fragment_call);

      apple9_put_u32(fixed_usc + AGX_APPLE9_RENDER_FIXED_DIMENSIONS_OFFSET,
                     package->width);
      apple9_put_u32(fixed_usc + AGX_APPLE9_RENDER_FIXED_DIMENSIONS_OFFSET + 4,
                     package->height);
   }

   if (!agx_apple9_install_render_archive(cache->dev))
      return false;

   cache->current = package;
   cache->fixed_usc_dirty = false;
   cache->generation++;
   if (getenv("AGX_APPLE9_PACKAGE_TRACE")) {
      fprintf(stderr,
              "APPLE9_RENDER_GENERATION id=%llu storage=%#llx logical=%#llx "
              "resident=%#llx target=%#llx size=%ux%u\n",
              (unsigned long long)cache->generation,
              (unsigned long long)package->bo->va->addr,
              (unsigned long long)cache->logical_va->addr,
              (unsigned long long)cache->resident_bo->va->addr,
              (unsigned long long)package->color_target, package->width,
              package->height);
      fprintf(stderr,
              "APPLE9_RENDER_CACHE fs_call=%#x vs_call=%#x "
              "prolog_call=%#x archive_next=%#x\n",
              package->fragment_call, package->vertex_call,
              package->vertex_prolog_call, cache->resident_archive_next);
   }
   return true;
}

struct apple9_render_archive_plan {
   bool append_fits;
   bool fresh_fits;
   uint64_t end;
};

static bool
apple9_plan_render_archive(const struct agx_apple9_render_cache *cache,
                          const struct agx_apple9_uniform_draw *draws,
                          unsigned count,
                          const struct agx_apple9_render_package *extra,
                          struct apple9_render_archive_plan *plan)
{
   if (!cache || (count && !draws) ||
       count + (extra != NULL) > AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS)
      return false;

   /* Plan the same stage interning performed by cache_bind. Count shared
    * shaders once, including a program used by several linked pipelines. */
   struct {
      const uint8_t *data;
      uint32_t size;
      unsigned stage;
      bool resident, in_first;
   } programs[AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS * 3];
   unsigned program_count = 0;
   const struct agx_apple9_render_package *first = count ? draws[0].package : extra;
   if (!first)
      return false;
   for (unsigned i = 0; i < count + (extra != NULL); ++i) {
      const struct agx_apple9_render_package *p =
         i < count ? draws[i].package : extra;
      if (!p || !p->sealed || p->color_target != first->color_target ||
          p->width != first->width || p->height != first->height)
         return false;
      const uint8_t *source = agx_bo_map(p->bo);
      const uint32_t blocks[] = {
         p->archive.fragment_block, p->archive.vertex_prolog_block,
         p->archive.vertex_main_block};
      const uint32_t sizes[] = {
         p->archive.fragment_block_size, p->archive.vertex_prolog_block_size,
         p->archive.vertex_main_block_size};
      const uint32_t calls[] = {
         p->fragment_call, p->vertex_prolog_call, p->vertex_call};
      for (unsigned stage = 0; stage < 3; ++stage) {
         unsigned j;
         for (j = 0; j < program_count; ++j) {
            if (programs[j].stage == stage && programs[j].size == sizes[stage] &&
                !memcmp(programs[j].data, source + blocks[stage], sizes[stage]))
               break;
         }
         if (j == program_count) {
            programs[j].data = source + blocks[stage];
            programs[j].size = sizes[stage];
            programs[j].stage = stage;
            programs[j].resident = false;
            programs[j].in_first = i == 0;
            program_count++;
         }
         programs[j].resident |= calls[stage] != 0;
      }
   }
   uint64_t append_end = cache->resident_archive_next;
   uint64_t end = first->archive.end;
   bool append_fits = cache->current != NULL;
   bool fresh_fits = end <= cache->resident_archive_limit;
   const uint32_t prefix = AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE +
                           AGX_APPLE9_RENDER_CONSTANT_SIZE;
   for (unsigned i = 0; i < program_count; ++i) {
      uint32_t call;
      if (!programs[i].resident) {
         append_fits &= append_end + prefix <= UINT32_MAX &&
                        apple9_archive_call(append_end + prefix, &call);
         append_end += programs[i].size;
      }
      if (!programs[i].in_first) {
         fresh_fits &= end + prefix <= UINT32_MAX &&
                       apple9_archive_call(end + prefix, &call);
         end += programs[i].size;
      }
   }
   plan->append_fits = append_fits && append_end <= cache->resident_archive_limit;
   plan->fresh_fits = fresh_fits && end <= cache->resident_archive_limit;
   plan->end = end;
   return true;
}

bool
agx_apple9_render_cache_can_add_draw(
   const struct agx_apple9_render_cache *cache,
   const struct agx_apple9_uniform_draw *draws, unsigned count,
   const struct agx_apple9_render_package *package)
{
   struct apple9_render_archive_plan plan;
   return package && apple9_plan_render_archive(cache, draws, count, package,
                                                &plan) && plan.fresh_fits;
}

/* Install the complete draw set before publishing any per-draw calls. The
 * submission lock and previous-owner wait also protect archive rollover. */
bool
agx_apple9_render_cache_bind_draws(
   struct agx_apple9_render_cache *cache,
   const struct agx_apple9_uniform_draw *draws, unsigned count)
{
   if (!cache || !draws || !count || count > AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS)
      return false;

   bool single_package = true;
   for (unsigned i = 1; i < count; ++i)
      single_package &= draws[i].package == draws[0].package;
   if (single_package)
      return agx_apple9_render_cache_bind(cache, draws[0].package);

   struct apple9_render_archive_plan plan;
   if (!apple9_plan_render_archive(cache, draws, count, NULL, &plan))
      return false;
   uint64_t end = plan.end;

   /* Check storage and entry encoding before installing any draw's programs. */
   if (!plan.append_fits) {
      if (!plan.fresh_fits) {
         fprintf(stderr,
                 "Apple9 batch shader archive exceeds storage or entry range: "
                 "end=%#llx storage=%#x\n",
                 (unsigned long long)end, cache->resident_archive_limit);
         return false;
      }
      if (getenv("AGX_APPLE9_PACKAGE_TRACE"))
         fprintf(stderr, "APPLE9_RENDER_BATCH_ROLLOVER end=%#llx limit=%#x\n",
                 (unsigned long long)end, cache->resident_archive_limit);
      list_for_each_entry(struct agx_apple9_render_package, p,
                          &cache->packages, link) {
         p->fragment_call = p->vertex_prolog_call = p->vertex_call = 0;
      }
      cache->current = NULL;
   }

   cache->installing_draws = true;
   bool ok = true;
   for (unsigned i = 0; ok && i < count; ++i) {
      bool seen = false;
      for (unsigned j = 0; j < i; ++j)
         seen |= draws[j].package == draws[i].package;
      if (!seen)
         ok = agx_apple9_render_cache_bind(cache, draws[i].package);
   }
   /* Attachment helpers and viewport are shared; retain the first package
    * as the batch's attachment owner. Shader-dependent PPP follows below. */
   if (ok)
      ok = agx_apple9_render_cache_bind(cache, draws[0].package);
   cache->installing_draws = false;
   return ok;
}

void
agx_apple9_render_cache_invalidate_fixed_usc(
   struct agx_apple9_render_cache *cache)
{
   if (cache)
      cache->fixed_usc_dirty = true;
}

bool
agx_apple9_render_cache_upload_vertex_buffer(
   struct agx_apple9_render_cache *cache, const void *data, size_t size)
{
   if (!cache || !cache->current || !data || !size ||
       size != cache->current->vertex_buffer_size)
      return false;

   const size_t allocation_size = ALIGN_POT(size, 0x100);
   if (allocation_size > 0x2000)
      return false;

   uint8_t *fixed_state =
      (uint8_t *)agx_bo_map(cache->dev->apple9_render_fixed_usc) +
      AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET;
   uint8_t *resident_state = (uint8_t *)agx_bo_map(cache->resident_bo) +
                             AGX_APPLE9_RENDER_COMPILER_STATE_OFFSET;

   /* The queue's fixed USC base resolves to apple9_render_fixed_usc, while
    * the package-relative compiler view resolves through resident_bo. Keep
    * the per-draw vertex payload coherent in both views. */
   for (unsigned i = 0; i < 2; ++i) {
      uint8_t *state = i ? resident_state : fixed_state;
      memset(state + 0x200, 0, allocation_size);
      memcpy(state + 0x200, data, size);
   }
   if (getenv("AGX_APPLE9_PACKAGE_TRACE")) {
      fprintf(stderr, "APPLE9_VERTEX_RESIDENT_BYTES");
      for (unsigned i = 0; i < MIN2(size, 64); ++i)
         fprintf(stderr, "%s%02x", i ? "" : " ", resident_state[0x200 + i]);
      fprintf(stderr, "\n");
   }
   return true;
}

/* EXP-M4-57/58: per-draw opaque launch records keep shared shader calls.
 * The four-buffer preload has its archive selector at +0x36 and needs no
 * separate constant-state reference. These externally captured preloads stay
 * opaque; only known table references and archive selectors are patched. */
#define APPLE9_UNIFORM_VS_LAUNCH 0x220400u
#define APPLE9_UNIFORM_FS_LAUNCH 0x230800u
#define APPLE9_UNIFORM_RECORDS   0x200100u
#define APPLE9_DRAW_PIPELINE_PPP 0x5a000u
#define APPLE9_DRAW_PIPELINE_STRIDE 0x100u
#define APPLE9_UNIFORM_STRIDE    0x100u
static_assert(APPLE9_DRAW_PIPELINE_PPP + AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS *
                 APPLE9_DRAW_PIPELINE_STRIDE <= 0x64000,
              "draw state must end before the viewport page");
static_assert(AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS * APPLE9_UNIFORM_STRIDE <=
                 0x3800,
              "uniform launches must fit in their reserved pages");
static_assert(AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS * 0x100 <= 0x3f00,
              "draw pointer records must fit in the resource page");

static struct apple9_external_blob apple9_uniform_launch;
static util_once_flag apple9_uniform_once = UTIL_ONCE_FLAG_INIT;
static bool apple9_uniform_loaded;

static void
apple9_load_uniform_launch(void)
{
   /* EXP-M4-59: the complete external VS launcher supports all sixteen
    * retained export publications; the old four-buffer VS launcher clobbered
    * one output at that capacity. The FS launcher remains unchanged. */
   apple9_uniform_loaded =
      apple9_load_external_blob("render_buffers_varyings12_launch.bin",
                                &apple9_uniform_launch) &&
      apple9_uniform_launch.size == 0x180;
}

bool
agx_apple9_render_cache_upload_uniforms(
   struct agx_apple9_render_cache *cache,
   const struct agx_apple9_uniform_draw *draws, unsigned count)
{
   if (!cache || !cache->current || !draws ||
       count > AGX_APPLE9_RENDER_MAX_UNIFORM_DRAWS)
      return false;
   util_call_once(&apple9_uniform_once, apple9_load_uniform_launch);
   if (!apple9_uniform_loaded)
      return false;
   uint8_t *views[] = {agx_bo_map(cache->resident_bo),
                       agx_bo_map(cache->dev->apple9_render_fixed_usc)};
   uint8_t *context = agx_bo_map(cache->resident_state_bo);
   for (unsigned i = 0; i < count; ++i) {
      const struct agx_apple9_render_package *package = draws[i].package;
      if (!package || !package->fragment_call || !package->vertex_prolog_call)
         return false;
      unsigned record = APPLE9_UNIFORM_RECORDS + i * 0x100;
      unsigned cf_offset = 0x40;
      for (unsigned v = 0; v < ARRAY_SIZE(views); ++v) {
         uint8_t *view = views[v];
         memset(view + record, 0, 0x100);
         for (unsigned slot = 0; slot < 4; ++slot) {
            apple9_put_u64(view + record + slot * 8, draws[i].vertex[slot]);
            apple9_put_u64(view + record + 0x20 + slot * 8,
                           draws[i].fragment[slot]);
         }
         apple9_build_cf_bindings(view + record + cf_offset,
                                  package->varying_components);
         for (unsigned stage = 0; stage < 2; ++stage) {
            unsigned location =
               (stage ? APPLE9_UNIFORM_FS_LAUNCH : APPLE9_UNIFORM_VS_LAUNCH) +
               i * APPLE9_UNIFORM_STRIDE;
            uint8_t *launch = view + location;
            memcpy(launch, apple9_uniform_launch.data + stage * 0xc0, 0xc0);
            if (!apple9_patch_compact_pointer(
                   launch, 1, 4, 5, 6, cache->dev->shader_base,
                   cache->dev->shader_base + record + stage * 0x20))
               return false;
            apple9_put_u24(launch + 0x36,
                           stage ? package->fragment_call
                                 : package->vertex_prolog_call);
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
      apple9_put_u32(group + 8, record + cf_offset);
      apple9_put_u32(group + 0x14,
         (APPLE9_UNIFORM_FS_LAUNCH + i * APPLE9_UNIFORM_STRIDE) / 0x40);
      /* EXP-M4-09 authored blend/write-mask state: enable tile read/modify/write.
       * The shader also waits for its allocated tile-load result slot. */
      if (draws[i].reads_tile)
         apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | (1u << 29));
      apple9_put_u32(group + 0x34, draws[i].depth_control);
      apple9_put_u32(group + 0x38, draws[i].depth_face);
      apple9_put_u32(group + 0x40, draws[i].depth_face);
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
      /* PPP depth-bias/scissor record: header followed by two 16-bit indices.
       * Depth bias remains disabled; its index is zero. */
      apple9_put_u32(ppp + 0xf0, 0x100);
      apple9_put_u32(ppp + 0xf4, draws[i].scissor_index);
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

uint32_t
agx_apple9_render_package_program_word(
   const struct agx_device *dev,
   const struct agx_apple9_render_package *package, uint32_t offset)
{
   if (!dev || !package || !package->bo || !package->bo->va)
      return 0;

   uint64_t value = AGX_APPLE9_RENDER_PACKAGE_OFFSET + offset;
   return value <= UINT32_MAX ? value : 0;
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
       fragment.render_targets != 1)
      return false;

   const unsigned scalar_outputs = 4 + vertex.varying_components;
   *pipeline = (struct agx_apple9_render_pipeline){
      .vertex_prolog = vertex_prolog,
      .vertex = vertex,
      .fragment = fragment,
      /* Filled from the installed archive BO before VDM emission. */
      .pipeline_word = 0,
      /* EXP-M4-57: this is a launch address in 64-byte units. The native
       * second draw advances it by three for another 0xc0-byte wrapper,
       * while the scalar output interface remains identical. */
      .vertex_launch = (vertex_prolog.binary
                           ? AGX_APPLE9_RENDER_FIXED_VERTEX_LAUNCH_OFFSET
                           : AGX_APPLE9_RENDER_VERTEX_LAUNCH_OFFSET) / 0x40,
      .vertex_state_class = scalar_outputs | (scalar_outputs << 8),
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

   /*
    * The first PPP reference carries a vertex-fetch envelope bit in its
    * relative word.  Caller-owned T8132 streams use 0x4040 for the inline
    * gl_VertexID path and 0x0040 for both measured VBO layouts.  All other
    * VDM bytes, including the referenced state itself, are identical after
    * address normalization.
    */
   if (pipeline->vertex_prolog.binary)
      ppp[0].relative = 0x0040;

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
         pipeline->index_size == 2 ? 0x61f20600 : 0x61f40600,
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
         0x61c40600, /* direct triangle, count/instances/start present */
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
                                       const float color[4])
{
   uint8_t *resident = agx_bo_map(cache->resident_bo);
   uint8_t *fixed = agx_bo_map(cache->dev->apple9_render_fixed_usc);
   for (unsigned i = 0; i < 2; ++i) {
      unsigned offset = 0x170 + i * 0x300;
      memcpy(resident + AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET + offset,
             color, 16);
      memcpy(fixed + AGX_APPLE9_RENDER_FIXED_TARGET_GRAPH_OFFSET + offset,
             color, 16);
   }
}
