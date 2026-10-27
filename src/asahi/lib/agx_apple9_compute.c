/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_compute.h"
#include "agx_apple9_launch.h"
#include "agx_device.h"
#include "pool.h"
#include "gallium/include/pipe/p_defines.h"
#include "asahi/compiler/agx_apple9_ir.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/u_call_once.h"
#include "util/u_math.h"

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
   uint8_t resource_count;
   uint16_t resource_record_size;
   uint32_t cdm_config;
   uint32_t cdm_constant;
   uint32_t cdm_tail;
   bool supports_indirect_dispatch;
};

static const struct apple9_compute_abi_desc *
apple9_compute_abi(const struct agx_apple9_compute_profile *profile)
{
   static const struct apple9_compute_abi_desc ssbo8_superset = {
      .resource_count = AGX_APPLE9_COMPUTE_MAX_RESOURCES,
      .resource_record_size = AGX_APPLE9_COMPUTE_SUPERSET_RESOURCE_STRIDE,
      .cdm_config = 0x00880000,
      .cdm_constant = 0x01000040,
      /* Match the full post-dispatch CDM barrier used by the Apple8 path.
       * GPU cache ordering is independent of asynchronous CPU submission. */
      .cdm_tail = 0x600fffff,
      .supports_indirect_dispatch = true,
   };

   if (!profile)
      return NULL;

   switch (profile->abi) {
   case AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS:
   case AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES:
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
       (profile->abi == AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS &&
        profile->resource_binding_count > AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES) ||
       !agx_apple9_launch_threadgroup_memory_supported(
          profile->required_threadgroup_memory_bytes) ||
       (profile->atomic_frame_size != 0 && profile->atomic_frame_size != 4) ||
       profile->preamble_size > AGX_APPLE9_MAX_PREAMBLE_BYTES ||
       profile->scratch_size > AGX_APPLE9_MAX_SCRATCH_BYTES ||
       profile->publication_count > 1022 ||
       (profile->scratch_size & 15))
      return false;

   const unsigned active = profile->resource_binding_count;
   const uint32_t resource_mask = BITFIELD_MASK(active);
   const uint32_t read_mask = profile->resource_read_mask;
   const uint32_t write_mask = profile->resource_write_mask;
   if ((!write_mask && !profile->writes_global) || ((read_mask | write_mask) & ~resource_mask))
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
      if (profile->resource_kind[i] > AGX_APPLE9_COMPUTE_RESOURCE_SHARED ||
          (profile->resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_SHARED &&
           (!profile->required_threadgroup_memory_bytes ||
            ((read_mask | write_mask) & BITFIELD_BIT(i)))) ||
          ((write_mask & BITFIELD_BIT(i)) &&
           profile->resource_kind[i] != AGX_APPLE9_COMPUTE_RESOURCE_SSBO))
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
apple9_put_u32(void *ptr, uint32_t value)
{
   memcpy(ptr, &value, sizeof(value));
}

static inline void
apple9_put_u64(void *ptr, uint64_t value)
{
   memcpy(ptr, &value, sizeof(value));
}

static inline uint64_t
apple9_get_u64(const void *ptr)
{
   uint64_t value;
   memcpy(&value, ptr, sizeof(value));
   return value;
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
apple9_build_compute_launch(uint8_t *out, uint64_t usc_exec_base,
                            uint64_t launch_address, uint64_t resource_address,
                            uint32_t main_offset,
                            const struct agx_apple9_compute_profile *profile,
                            uint64_t preamble_address)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || !apple9_compute_profile_valid(profile, abi) ||
       main_offset < AGX_APPLE9_ENTRY_HEADER_SIZE + AGX_APPLE9_ENTRY_CODE_OFFSET)
      return false;

   struct agx_apple9_launch_parameters params = {
      .entry_offset = main_offset,
      .publication_count = profile->publication_count,
      .preamble_address = preamble_address,
      .launch_address = launch_address,
   };
   params.shader_base = usc_exec_base;
   params.resource_table = resource_address;
   params.resource_count =
      profile->abi == AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES
         ? AGX_APPLE9_GRAPHICS_ROOT_WORDS / 2 - 1
         : profile->resource_binding_count;
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

void
agx_apple9_initialize_entries(struct agx_device *dev)
{
   uint8_t *header = dev->apple9_entry_map;
   memset(header, 0, AGX_APPLE9_ENTRY_HEADER_SIZE);
   apple9_put_u32(header, AGX_APPLE9_ENTRY_HEADER_SIZE);
   for (unsigned offset = 0x40; offset < AGX_APPLE9_ENTRY_HEADER_SIZE; offset += 2)
      apple9_put_u16(header + offset, 0x0006);
   apple9_fill_helper_table(header, 0x100, 10);
   apple9_fill_helper_table(header, 0x200, 10);
   agx_bo_note_cpu_write(dev->apple9_entries, 0, AGX_APPLE9_ENTRY_HEADER_SIZE);
}

/* Compiled shader objects and pending batches retain the body BO. Tying the
 * entry allocation to that BO prevents deletion or cache reuse from changing
 * the target of an already encoded launch. */
uint32_t
agx_apple9_shader_entry(struct agx_device *dev, struct agx_bo *body, bool compute)
{
   simple_mtx_lock(&dev->apple9_entry_lock);
   uint32_t entry = body->apple9_entry_offset;
   if (!entry) {
      uint32_t block = util_vma_heap_alloc(&dev->apple9_entry_heap,
                                          AGX_APPLE9_ENTRY_BLOCK_SIZE,
                                          AGX_APPLE9_ENTRY_ALIGNMENT);
      if (block) {
         entry = block + AGX_APPLE9_ENTRY_CODE_OFFSET;
         uint8_t *bytes = (uint8_t *)dev->apple9_entry_map + block;
         memset(bytes, 0, AGX_APPLE9_ENTRY_BLOCK_SIZE);
         apple9_put_u32(bytes, AGX_APPLE9_ENTRY_BLOCK_SIZE);
         if (compute) {
            util_call_once(&apple9_compute_helpers_once,
                           apple9_initialize_compute_helpers);
            memcpy(bytes + 0x40, apple9_compute_constant, 0x40);
         }
         if (apple9_build_body_entry(bytes + AGX_APPLE9_ENTRY_CODE_OFFSET,
                                     dev->shader_base + entry, body->va->addr)) {
            agx_bo_note_cpu_write(dev->apple9_entries, block,
                                  AGX_APPLE9_ENTRY_BLOCK_SIZE);
            body->apple9_entry_offset = entry;
         } else {
            util_vma_heap_free(&dev->apple9_entry_heap, block,
                               AGX_APPLE9_ENTRY_BLOCK_SIZE);
            entry = 0;
         }
      }
   }
   simple_mtx_unlock(&dev->apple9_entry_lock);
   return entry;
}

bool
agx_apple9_prepare_compute_dispatch(
   struct agx_device *dev, struct agx_pool *usc_pool, struct agx_bo *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, uint64_t *launch_address)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || !resources || !geometry || !launch_address ||
       !apple9_compute_profile_valid(profile, abi) ||
       resource_count != profile->resource_binding_count ||
       !!preamble_address != !!profile->preamble_size) {
      if (getenv("AGX_APPLE9_TRACE_REJECT"))
         fprintf(stderr, "APPLE9_PREPARE_REJECT reason=profile-or-resources count=%u expected=%u\n",
                 resource_count, profile->resource_binding_count);
      return false;
   }

   uint32_t entry = agx_apple9_shader_entry(dev, body, true);
   if (!entry) {
      if (getenv("AGX_APPLE9_TRACE_REJECT"))
         fprintf(stderr, "APPLE9_PREPARE_REJECT reason=shader-entry\n");
      return false;
   }

   size_t record_size = apple9_compute_resource_record_size_for_abi(abi);
   struct agx_ptr record = agx_pool_alloc_aligned(usc_pool, record_size, 64);
   memset(record.cpu, 0, record_size);
   if (!agx_apple9_build_compute_geometry_fields(record.cpu, record_size,
                                                record.gpu, geometry)) {
      if (getenv("AGX_APPLE9_TRACE_REJECT"))
         fprintf(stderr, "APPLE9_PREPARE_REJECT reason=geometry mode=%u threads=%u,%u,%u local=%u,%u,%u\n",
                 geometry->mode, geometry->threads[0], geometry->threads[1],
                 geometry->threads[2], geometry->local[0], geometry->local[1],
                 geometry->local[2]);
      return false;
   }
   if (profile->abi == AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES) {
      uint64_t roots[AGX_APPLE9_GRAPHICS_ROOT_WORDS / 2] = {
         textures, samplers,
         agx_pool_upload_aligned(usc_pool, resources,
                                 resource_count * sizeof(uint64_t), 64),
         apple9_get_u64(record.cpu), AGX_APPLE9_COMPUTE_SHARED_ROOT, 0,
      };
      memcpy(record.cpu, roots, sizeof(roots));
   } else {
      for (unsigned i = 0; i < resource_count; i++)
         apple9_put_u64((uint8_t *)record.cpu +
                          (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + i) * 8,
                       resources[i]);
   }

   struct agx_ptr launch = agx_pool_alloc_aligned(
      usc_pool, agx_apple9_compute_launch_size(profile), 64);
   if (!apple9_build_compute_launch(launch.cpu, dev->shader_base, launch.gpu,
                                    record.gpu, entry, profile, preamble_address)) {
      if (getenv("AGX_APPLE9_TRACE_REJECT"))
         fprintf(stderr, "APPLE9_PREPARE_REJECT reason=launch abi=%u\n", profile->abi);
      return false;
   }
   *launch_address = launch.gpu;
   return true;
}

bool
agx_apple9_compute_enabled(const struct agx_device *dev)
{
   /* These generated launch and resource encodings have been exercised on
    * T8132. Shared Apple9 ISA support alone does not validate G17P launch ABI. */
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
   bool indirect_local, const struct agx_apple9_compute_profile *profile)
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
   apple9_put_u32(record + 0x00, abi->cdm_config |
                  (indirect_local ? 0x10000000 : 0x08000000));
   apple9_put_u32(record + 0x04, abi->cdm_constant);
   apple9_put_u64(record + 0x08, shader);
   /* Native Apple9 indirect CDM stores the pointer halves high then low. */
   apple9_put_u32(record + 0x10, indirect >> 32);
   apple9_put_u32(record + 0x14, indirect);
   /* Mode 2 reads six words: thread counts followed by local sizes. Mode 1
    * reads three workgroup counts and takes local sizes from this command. */
   if (!indirect_local) {
      for (unsigned d = 0; d < 3; ++d)
         apple9_put_u32(record + 0x18 + d * 4, local[d]);
   }
   apple9_put_u32(record + (indirect_local ? 0x18 : 0x24), abi->cdm_tail);
   return true;
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
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER: return 3;
   case PIPE_TEX_WRAP_CLAMP: return 4;
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
