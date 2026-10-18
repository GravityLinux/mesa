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
#include "util/u_debug.h"
#include "util/u_math.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "asahi/layout/layout.h"

bool
agx_apple9_translate_pbe(void *out, const struct agx_pbe_packed *state,
                        enum pipe_format format, const struct ail_layout *layout,
                        uint64_t layer_base, unsigned level,
                        unsigned layer_count, bool layered,
                        unsigned dimension)
{
   if (dimension > 15 || !layout->levels || layout->levels > 16 ||
       level >= layout->levels || !layout->width_px || !layout->height_px ||
       layout->width_px > 16384 || layout->height_px > 16384 ||
       !layer_count || layer_count > 65536)
      return false;

   bool linear = layout->tiling == AIL_TILING_LINEAR;
   if (linear && (level || layer_count > 1 || layout->compressed))
      return false;
   unsigned descriptor_level = linear ? level : 0;
   uint64_t base = layer_base + layout->level_offsets_B[descriptor_level] -
                   layout->level_offsets_B[0];
   if ((base & 15) || (base >> 44))
      return false;
   uint64_t address = base >> 4;
   uint32_t stride = linear ? ail_get_linear_stride_B(layout, level) : 0;
   if (linear && (!stride || (stride & 15) ||
                  stride > AGX_APPLE9_MAX_LINEAR_STRIDE))
      return false;

   uint8_t descriptor[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
   if (layout->compressed) {
      agx_unpack(stderr, state, PBE, pbe);
      if (!pbe.compressed || !pbe.extended ||
          !pbe.acceleration_buffer || (pbe.acceleration_buffer & 15) ||
          (pbe.acceleration_buffer >> 48))
         return false;
      uint64_t metadata = pbe.acceleration_buffer >> 4;
      memcpy(descriptor + 16, &metadata, sizeof(metadata));
      address |= UINT64_C(1) << 63;
   }
   uint32_t header[2];
   memcpy(header, state, sizeof(header));
   unsigned width = u_minify(layout->width_px, descriptor_level);
   unsigned height = u_minify(layout->height_px, descriptor_level);
   header[0] = (header[0] & 0x00fffff0) | dimension | ((width - 1) << 24);
   header[1] = ((width - 1) >> 8) | ((height - 1) << 6) |
               (header[1] & 0x01000000) |
               ((uint32_t)layout->compressed << 27);
   /* Packed tile words already have memory channel order. */
   if (agx_apple9_color_is_packed(agx_apple9_color_tile_format(format)))
      header[0] = (header[0] & ~0x00ff0000u) | 0x00e40000;
   if (!linear && layout->levels > 1) {
      header[1] |= 1u << 26;
      address |= UINT64_C(1) << 63;
      descriptor[21] |= level << 4;
      descriptor[22] = layout->levels - 1;
   }
   if (linear) {
      address |= ((uint64_t)(stride >> 4) - 1) << 44;
   } else if (layered) {
      address |= (uint64_t)(layer_count - 1) << 44;
      if (layout->page_aligned_layers)
         address |= UINT64_C(1) << 60;
   }
   memcpy(descriptor, header, sizeof(header));
   memcpy(descriptor + 8, &address, sizeof(address));
   memcpy(out, descriptor, sizeof(descriptor));
   return true;
}

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
       profile->preamble_size > AGX_APPLE9_MAX_COMPUTE_PREAMBLE_BYTES ||
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

/* Launch parameters of a compute profile; the caller sets the per-launch
 * entry, resource table and program address. */
static bool
apple9_compute_launch_params(struct agx_apple9_launch_parameters *out,
                             uint64_t usc_exec_base,
                             const struct agx_apple9_compute_profile *profile,
                             uint64_t preamble_address)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || !apple9_compute_profile_valid(profile, abi))
      return false;

   struct agx_apple9_launch_parameters params = {
      .publication_count = profile->publication_count,
      .preamble_address = preamble_address,
   };
   params.shader_base = usc_exec_base;
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
   *out = params;
   return true;
}

static bool
apple9_build_compute_launch(uint8_t *out, uint64_t usc_exec_base,
                            uint64_t launch_address, uint64_t resource_address,
                            uint32_t main_offset,
                            const struct agx_apple9_compute_profile *profile,
                            uint64_t preamble_address)
{
   struct agx_apple9_launch_parameters params;
   if (main_offset < AGX_APPLE9_ENTRY_HEADER_SIZE + AGX_APPLE9_ENTRY_CODE_OFFSET ||
       !apple9_compute_launch_params(&params, usc_exec_base, profile,
                                     preamble_address))
      return false;
   params.entry_offset = main_offset;
   params.resource_table = resource_address;
   params.launch_address = launch_address;
   return agx_apple9_launch_build(
      out, agx_apple9_compute_launch_size(profile), AGX_APPLE9_LAUNCH_COMPUTE, &params);
}

bool
agx_apple9_compute_launch_template(struct agx_apple9_launch_template *tmpl,
                                   const struct agx_apple9_compute_profile *profile,
                                   uint64_t usc_exec_base,
                                   uint64_t preamble_address)
{
   struct agx_apple9_launch_parameters params;
   memset(tmpl, 0, sizeof(*tmpl));
   return apple9_compute_launch_params(&params, usc_exec_base, profile,
                                       preamble_address) &&
          agx_apple9_launch_template_init(
             tmpl, agx_apple9_compute_launch_size(profile),
             AGX_APPLE9_LAUNCH_COMPUTE, &params);
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
      /* Direct geometry counts threads in 32 bits. The API's 65535-group
       * limit does not bind driver-internal dispatches. */
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

static bool apple9_build_body_entry(uint8_t *out, uint64_t entry,
                                    uint64_t body);

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

/* The launch target addresses entries in a few megabytes, room for about
 * 43k shader bodies. Draw-time variants can exceed that over a session, so
 * entries are only fixed while a recording that may encode them is alive:
 * the device's pin hook ties the body to the recording's lifetime. When the
 * arena is full, free the least recently pinned unpinned entries. No live
 * recording, and hence no pending batch, encodes them; their bodies get a new
 * entry on their next use. Launch records cached by content cannot alias a
 * recycled entry, since they also encode the body's preamble address.
 */
static void
apple9_evict_entries(struct agx_device *dev)
{
   const unsigned target =
      AGX_APPLE9_ENTRY_ARENA_SIZE / AGX_APPLE9_ENTRY_BLOCK_SIZE / 4;
   unsigned freed = 0;
   list_for_each_entry_safe(struct agx_bo, bo, &dev->apple9_entry_lru,
                            apple9_entry_link) {
      if (freed == target)
         break;
      if (bo->apple9_entry_pins || bo->apple9_entry_fixed)
         continue;
      util_vma_heap_free(&dev->apple9_entry_heap,
                         bo->apple9_entry_offset - AGX_APPLE9_ENTRY_CODE_OFFSET,
                         AGX_APPLE9_ENTRY_BLOCK_SIZE);
      bo->apple9_entry_offset = 0;
      list_del(&bo->apple9_entry_link);
      ++freed;
   }
}

static uint32_t
apple9_shader_entry(struct agx_device *dev, void *allocator,
                    struct agx_bo *body, bool compute)
{
   simple_mtx_lock(&dev->apple9_entry_lock);
   if (!allocator || !dev->apple9_pin_entry ||
       !dev->apple9_pin_entry(allocator, body))
      body->apple9_entry_fixed = true;
   uint32_t entry = body->apple9_entry_offset;
   if (entry) {
      list_move_to(&body->apple9_entry_link, &dev->apple9_entry_lru);
   } else {
      uint32_t block = util_vma_heap_alloc(&dev->apple9_entry_heap,
                                          AGX_APPLE9_ENTRY_BLOCK_SIZE,
                                          AGX_APPLE9_ENTRY_ALIGNMENT);
      if (!block) {
         apple9_evict_entries(dev);
         block = util_vma_heap_alloc(&dev->apple9_entry_heap,
                                     AGX_APPLE9_ENTRY_BLOCK_SIZE,
                                     AGX_APPLE9_ENTRY_ALIGNMENT);
      }
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
            list_addtail(&body->apple9_entry_link, &dev->apple9_entry_lru);
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
agx_apple9_prepare_compute_dispatch_alloc(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t), struct agx_bo *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, uint64_t *launch_address)
{
   return agx_apple9_prepare_compute_dispatch_alloc_tmpl(
      dev, allocator, alloc, body, profile, resources, resource_count, textures,
      samplers, geometry, preamble_address, NULL, launch_address);
}

DEBUG_GET_ONCE_BOOL_OPTION(apple9_launch_check, "HK_APPLE9_LAUNCH_CHECK", false)

bool
agx_apple9_prepare_compute_dispatch_alloc_tmpl(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t), struct agx_bo *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, const struct agx_apple9_launch_template *tmpl,
   uint64_t *launch_address)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi || !resources || !geometry || !launch_address ||
       !apple9_compute_profile_valid(profile, abi) ||
       resource_count != profile->resource_binding_count ||
       !!preamble_address != !!profile->preamble_size)
      return false;

   uint32_t entry = apple9_shader_entry(dev, allocator, body, true);
   if (!entry)
      return false;

   size_t record_size = apple9_compute_resource_record_size_for_abi(abi);
   struct agx_ptr record = alloc(allocator, record_size, 64);
   if (!record.cpu)
      return false;
   memset(record.cpu, 0, record_size);
   if (!agx_apple9_build_compute_geometry_fields(record.cpu, record_size,
                                                record.gpu, geometry))
      return false;
   if (profile->abi == AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES) {
      struct agx_ptr table = alloc(allocator, resource_count * 8, 64);
      if (!table.cpu)
         return false;
      memcpy(table.cpu, resources, resource_count * 8);
      uint64_t roots[AGX_APPLE9_GRAPHICS_ROOT_WORDS / 2] = {
         textures, samplers,
         table.gpu,
         apple9_get_u64(record.cpu), AGX_APPLE9_COMPUTE_SHARED_ROOT, 0,
      };
      memcpy(record.cpu, roots, sizeof(roots));
   } else {
      for (unsigned i = 0; i < resource_count; i++)
         apple9_put_u64((uint8_t *)record.cpu +
                          (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + i) * 8,
                       resources[i]);
   }

   struct agx_ptr launch = alloc(allocator, agx_apple9_compute_launch_size(profile), 64);
   if (!launch.cpu)
      return false;
   if (tmpl && tmpl->valid) {
      /* The template was built from this profile and preamble; only the
       * record, entry and program address differ per dispatch. */
      assert(tmpl->preamble_address == preamble_address &&
             tmpl->shader_base == dev->shader_base);
      if (entry < AGX_APPLE9_ENTRY_HEADER_SIZE + AGX_APPLE9_ENTRY_CODE_OFFSET ||
          !agx_apple9_launch_template_emit(launch.cpu, tmpl, record.gpu, entry,
                                           launch.gpu))
         return false;
      if (debug_get_option_apple9_launch_check()) {
         uint8_t ref[AGX_APPLE9_COMPUTE_LAUNCH_SIZE];
         size_t size = agx_apple9_compute_launch_size(profile);
         if (!apple9_build_compute_launch(ref, dev->shader_base, launch.gpu,
                                          record.gpu, entry, profile,
                                          preamble_address) ||
             memcmp(ref, launch.cpu, size)) {
            fprintf(stderr, "HK_APPLE9_LAUNCH_CHECK: template launch differs\n");
            abort();
         }
      }
   } else if (!apple9_build_compute_launch(launch.cpu, dev->shader_base,
                                           launch.gpu, record.gpu, entry,
                                           profile, preamble_address)) {
      return false;
   }
   *launch_address = launch.gpu;
   return true;
}

static struct agx_ptr
apple9_pool_alloc(void *pool, size_t size, size_t alignment)
{
   return agx_pool_alloc_aligned(pool, size, alignment);
}

bool
agx_apple9_prepare_compute_dispatch(
   struct agx_device *dev, struct agx_pool *usc_pool, struct agx_bo *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, uint64_t *launch_address)
{
   return agx_apple9_prepare_compute_dispatch_alloc(
      dev, usc_pool, apple9_pool_alloc, body, profile, resources, resource_count,
      textures, samplers, geometry, preamble_address, launch_address);
}

bool
agx_apple9_compute_enabled(const struct agx_device *dev)
{
   /* These generated launch and resource encodings have been exercised on
    * T8132. Shared Apple9 ISA support alone does not validate G17P launch ABI. */
   return dev->chip == AGX_CHIP_G16G;
}

void
agx_apple9_set_compute_barrier(void *record, unsigned record_size)
{
   uint8_t *tail = (uint8_t *)record + record_size - 4;
   apple9_put_u32(tail, 0x60000160);
}

const char *
agx_apple9_dispatch_reject_reason(
   uint64_t launch, bool indirect, uint64_t indirect_address,
   const uint32_t global[3], const uint32_t local[3],
   const struct agx_apple9_compute_profile *profile)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!abi)
      return "no compute ABI for the profile";
   if (launch & 0x3f)
      return "launch address not 64-byte aligned";
   if (!apple9_compute_profile_valid(profile, abi))
      return "invalid compute profile";
   if (indirect) {
      if (!abi->supports_indirect_dispatch)
         return "ABI has no indirect dispatch";
      if (!indirect_address)
         return "null indirect address";
      if (indirect_address & 3)
         return "indirect address not 4-byte aligned";
   } else if (!global) {
      return "no grid";
   }
   if (!local)
      return "no local size";
   uint64_t threads = 1;
   for (unsigned d = 0; d < 3; ++d) {
      if (!indirect && !global[d])
         return "zero grid axis";
      if (!local[d])
         return "zero local axis";
      if (!profile->variable_local_size && local[d] != profile->local_size[d])
         return "local size differs from the compiled one";
      if (local[d] > 1024 / threads)
         return "more than 1024 threads per workgroup";
      threads *= local[d];
   }
   return NULL;
}

bool
agx_apple9_emit_direct_dispatch(
   void *out, uint64_t launch, const uint32_t global[3],
   const uint32_t local[3], const struct agx_apple9_compute_profile *profile,
   bool barrier)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!out ||
       agx_apple9_dispatch_reject_reason(launch, false, 0, global, local, profile))
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
   if (barrier)
      apple9_put_u32(record + 0x28, abi->cdm_tail);
   return true;
}

bool
agx_apple9_emit_indirect_dispatch(
   void *out, uint64_t launch, uint64_t indirect, const uint32_t local[3],
   bool indirect_local, const struct agx_apple9_compute_profile *profile,
   bool barrier)
{
   const struct apple9_compute_abi_desc *abi = apple9_compute_abi(profile);
   if (!out ||
       agx_apple9_dispatch_reject_reason(launch, true, indirect, NULL, local,
                                         profile))
      return false;

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
   if (barrier)
      apple9_put_u32(record + (indirect_local ? 0x18 : 0x24), abi->cdm_tail);
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
   if (agx_apple9_color_is_wide(format) || agx_apple9_color_is_packed(format) ||
       agx_apple9_color_is_normalized(format))
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
          wrap == PIPE_TEX_WRAP_CLAMP ||
          wrap == PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE;
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
   /* Mirrored clamp-to-edge descriptor code, from change-one-parameter
    * descriptor captures (agx-re docs/descriptors). */
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE: return 5;
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
   /* Minimum and maximum LOD are unsigned Q6 fields at bits 0-9 and 10-19.
    * Fractional maximum LODs blend into the next level, as a biased
    * trilinear sample clamped to 2.1 shows. */
   const uint32_t words[2] = {
      (uint32_t)roundf(min_lod * 64) | ((uint32_t)roundf(max_lod * 64) << 10) |
      (anisotropy << 20) | (mag_linear << 23) | (min_linear << 25) | (mip_filter << 27) |
      (apple9_sampler_address(wrap_s) << 29),
      apple9_sampler_address(wrap_t) | (1u << 7) | (7u << 8),
   };
   memcpy(out, words, sizeof(words));
}

bool
agx_apple9_translate_texture(void *out, const struct agx_texture_packed *state)
{
   agx_unpack(stderr, state, TEXTURE, tex);
   return agx_apple9_pack_texture(out, &tex);
}

bool
agx_apple9_pack_texture(void *out, const struct AGX_TEXTURE *state)
{
   struct AGX_TEXTURE tex = *state;
   if ((tex.address & 15) || (tex.address >> 48))
      return false;

   if (tex.dimension == AGX_TEXTURE_DIMENSION_3D &&
       tex.depth > AGX_APPLE9_MAX_3D_TEXTURE_DIMENSION)
      return false;

   bool single_layer_linear = tex.layout == AGX_LAYOUT_LINEAR && tex.extended &&
      (tex.dimension == AGX_TEXTURE_DIMENSION_1D ||
       tex.dimension == AGX_TEXTURE_DIMENSION_2D ||
       tex.dimension == AGX_TEXTURE_DIMENSION_1D_ARRAY ||
       tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY) && tex.depth_linear == 1;
   if ((tex.extended && !tex.compressed && !single_layer_linear) ||
       (tex.compressed &&
        (!tex.extended || tex.layout == AGX_LAYOUT_LINEAR ||
         !tex.acceleration_buffer || (tex.acceleration_buffer & 15) ||
         (tex.acceleration_buffer >> 48))) ||
       tex.mode != AGX_IMAGE_MODE_NORMAL ||
       (tex.layout != AGX_LAYOUT_LINEAR && tex.layout != AGX_LAYOUT_TWIDDLED &&
        tex.layout != AGX_LAYOUT_GPU))
      return false;

   bool array = tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
                tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED ||
                tex.dimension == AGX_TEXTURE_DIMENSION_1D_ARRAY;
   bool ms = tex.dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
             tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;

   /* Cube arrays keep their native dimension, with the layer count field
    * holding the number of cubes. */
   array |= tex.dimension == AGX_TEXTURE_DIMENSION_CUBE_ARRAY;
   /* A single linear layer needs no extended array stride. Preserve the
    * dimension expected by the shader, including array coordinates. */
   if (single_layer_linear) {
      tex.extended = false;
      /* The legacy extended flag overlaps the unpacked stride's top bit. */
      tex.stride &= BITFIELD_MASK(21);
      tex.depth = 1;
   }
   uint8_t *descriptor = out;
   memset(descriptor, 0, AGX_APPLE9_TEXTURE_TABLE_STRIDE);
   /* Only the format and dimensions use the legacy layout. In particular,
    * never route the native 48-bit byte address through its 40-bit field.
    */
   struct agx_texture_packed prefix;
   agx_pack(&prefix, TEXTURE, cfg) {
      cfg.dimension = tex.dimension;
      cfg.layout = tex.layout;
      cfg.channels = tex.channels;
      cfg.type = tex.type;
      cfg.swizzle_r = tex.swizzle_r;
      cfg.swizzle_g = tex.swizzle_g;
      cfg.swizzle_b = tex.swizzle_b;
      cfg.swizzle_a = tex.swizzle_a;
      cfg.width = tex.width;
      cfg.height = tex.height;
   }
   memcpy(descriptor, &prefix, 8);
   uint32_t dimensions;
   memcpy(&dimensions, descriptor + 4, 4);
   dimensions = (dimensions & 0x00ffffff) |
                ((ms && tex.samples == AGX_SAMPLE_COUNT_4) << 24) |
                (tex.unk_mipmapped << 26) | (tex.compressed << 27);
   memcpy(descriptor + 4, &dimensions, 4);

   /* The address starts two bits earlier than Apple8. Depth/linear stride,
    * sRGB and tiled layer alignment retain their positions. */
   uint64_t address = tex.address >> 4;
   if (tex.layout == AGX_LAYOUT_LINEAR)
      address |= (uint64_t)(tex.stride >> 4) << 46;
   else {
      if (array || tex.dimension == AGX_TEXTURE_DIMENSION_3D)
         address |= (uint64_t)(tex.depth - 1) << 46;
      address |= (uint64_t)tex.page_aligned_layers << 62;
   }
   address |= (uint64_t)(tex.unk_mipmapped || tex.compressed) << 63;
   address |= (uint64_t)tex.srgb << 44;
   address |= (uint64_t)tex.srgb_2_channel << 45;
   memcpy(descriptor + 8, &address, sizeof(address));
   if (tex.compressed) {
      uint64_t metadata = tex.acceleration_buffer >> 4;
      memcpy(descriptor + 16, &metadata, sizeof(metadata));
   }
   descriptor[21] |= tex.first_level << 4;
   descriptor[22] = tex.last_level;
   return true;
}

bool
agx_apple9_pack_sampler_state(void *out, const struct AGX_SAMPLER *state)
{
   const struct AGX_SAMPLER sampler = *state;
   if (sampler.magnify > AGX_FILTER_LINEAR ||
       sampler.minify > AGX_FILTER_LINEAR || sampler.mip_filter > AGX_MIP_FILTER_LINEAR ||
       sampler.wrap_s > AGX_WRAP_MIRRORED_CLAMP_TO_EDGE ||
       sampler.wrap_t > AGX_WRAP_MIRRORED_CLAMP_TO_EDGE ||
       sampler.wrap_r > AGX_WRAP_MIRRORED_CLAMP_TO_EDGE ||
       sampler.border_colour == AGX_BORDER_COLOUR_CUSTOM ||
       sampler.pixel_coordinates)
      return false;

   const unsigned wrap[] = {
      PIPE_TEX_WRAP_CLAMP_TO_EDGE, PIPE_TEX_WRAP_REPEAT,
      PIPE_TEX_WRAP_MIRROR_REPEAT, PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      PIPE_TEX_WRAP_CLAMP, PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE,
   };
   memset(out, 0, AGX_APPLE9_SAMPLER_TABLE_STRIDE);
   agx_apple9_pack_sampler(out, sampler.minify == AGX_FILTER_LINEAR,
      sampler.magnify == AGX_FILTER_LINEAR, sampler.mip_filter,
      sampler.minimum_lod, sampler.maximum_lod, wrap[sampler.wrap_s],
      wrap[sampler.wrap_t], sampler.maximum_anisotropy);
   uint32_t modes;
   memcpy(&modes, (uint8_t *)out + 4, 4);
   modes |= sampler.wrap_r << 3;
   modes |= sampler.border_colour << 29;
   modes |= (uint32_t)sampler.seamful_cube_maps << 31;
   if (sampler.compare_enable) {
      static const uint16_t comparisons[] = {
         [AGX_COMPARE_FUNC_LEQUAL] = 0x400,
         [AGX_COMPARE_FUNC_GEQUAL] = 0x480,
         [AGX_COMPARE_FUNC_LESS] = 0x500,
         [AGX_COMPARE_FUNC_GREATER] = 0x580,
         [AGX_COMPARE_FUNC_EQUAL] = 0x600,
         [AGX_COMPARE_FUNC_NOT_EQUAL] = 0x680,
         [AGX_COMPARE_FUNC_ALWAYS] = 0x700,
         [AGX_COMPARE_FUNC_NEVER] = 0x780,
      };
      modes = (modes & ~0x780u) | comparisons[sampler.compare_func];
   }
   memcpy((uint8_t *)out + 4, &modes, 4);
   return true;
}

bool
agx_apple9_translate_sampler(void *out, const struct agx_sampler_packed *state)
{
   agx_unpack(stderr, state, SAMPLER, sampler);
   return agx_apple9_pack_sampler_state(out, &sampler);
}

/* Texel fetches return zero outside the image, as image robustness
 * requires: clamp every axis to the transparent black border.
 */
void
agx_apple9_pack_fetch_sampler(void *out)
{
   agx_apple9_pack_sampler(out, false, false, 1, 0, 14,
                           PIPE_TEX_WRAP_CLAMP_TO_BORDER,
                           PIPE_TEX_WRAP_CLAMP_TO_BORDER, 1);
   uint32_t modes;
   memcpy(&modes, (uint8_t *)out + 4, 4);
   modes |= apple9_sampler_address(PIPE_TEX_WRAP_CLAMP_TO_BORDER) << 3;
   memcpy((uint8_t *)out + 4, &modes, 4);
}

void
agx_apple9_pack_nearest_sampler(void *out)
{
   agx_apple9_pack_sampler(out, false, false, 0, 0, 14,
                            PIPE_TEX_WRAP_CLAMP_TO_EDGE, PIPE_TEX_WRAP_CLAMP_TO_EDGE, 1);
}

/* The default coefficient table occupies the end of the graphics resource
 * page, after per-draw argument records. Draw-specific tables live in their
 * own slots on the following pages so mixed interpolation can use one
 * descriptor per scalar without overwriting the next draw's roots. */
#define APPLE9_CF_BINDINGS 0x203d00u
#define APPLE9_CF_BINDINGS_SIZE 0x2a0u
/* One binding per user scalar in the worst case, then 1/W, primitive ID,
 * Z, point coordinates, two barycentric pairs and the per-vertex inputs. */
static_assert(4 + 4 * (AGX_APPLE9_MAX_VARYING_COMPONENTS + 1 + 1 + 1 + 1 + 2 +
                       AGX_APPLE9_MAX_PER_VERTEX) <= APPLE9_CF_BINDINGS_SIZE,
              "coefficient table must fit its worst-case bindings");

static unsigned
apple9_cf_binding_count(unsigned components)
{
   return 1 + DIV_ROUND_UP(components, 4);
}

static unsigned
apple9_provoking_vertex(unsigned primitive, bool first)
{
   return first ? (primitive == AGX_PRIMITIVE_TRIANGLE_FAN ? 1 : 0) : 2;
}

static unsigned
apple9_build_cf_bindings(uint8_t *table, unsigned components,
                         struct agx_apple9_interp_mask linear_mask,
                         struct agx_apple9_interp_mask flat_mask,
                         unsigned provoking_vertex, bool reads_z, bool reads_point_coord,
                         bool reads_primitive_id,
                         const struct agx_apple9_render_stage *fragment)
{
   const enum agx_shade_model flat_shade =
      provoking_vertex == 1 ? AGX_SHADE_MODEL_FLAT_VERTEX_1
      : provoking_vertex == 0 ? AGX_SHADE_MODEL_FLAT_VERTEX_0
                              : AGX_SHADE_MODEL_FLAT_VERTEX_2;
   assert(components <= AGX_APPLE9_MAX_VARYING_COMPONENTS);
   memset(table, 0, APPLE9_CF_BINDINGS_SIZE);
   unsigned extra = 2 * util_bitcount(fragment->barycentric_mask) +
                    fragment->per_vertex_count;
   unsigned slots = components + 1 + reads_z + 2 * reads_point_coord +
                    reads_primitive_id + extra;
   apple9_put_u32(table, slots | (slots << 8));
   /* Coefficient zero is 1/W. Keep stable scalar indices across all modes. */
   apple9_put_u32(table + 4, 0x0c);
   unsigned binding = 1;
   for (unsigned start = 0; start < components; ++binding) {
      unsigned shade = agx_apple9_interp_mask_test(flat_mask, start)
                          ? flat_shade
                       : agx_apple9_interp_mask_test(linear_mask, start) ? 3
                                                             : 7;
      unsigned count = 1;
      while (count < 4 && start + count < components) {
         unsigned next = start + count;
         unsigned next_shade = agx_apple9_interp_mask_test(flat_mask, next)
                                  ? flat_shade
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
   if (reads_primitive_id) {
      /* Rasterizer-generated scalar; independent of the provoking vertex. */
      apple9_put_u32(table + 4 + binding++ * 4,
                     0x60 | ((components + 1) << 16));
   }
   if (reads_z) {
      /* Public CF source=FRAGCOORD_Z, linear, source slot one. Keep user
       * coefficient indices stable and append the depth coefficient. */
      apple9_put_u32(table + 4 + binding++ * 4,
                     0x12c | ((components + 1 + reads_primitive_id) << 16));
   }
   if (reads_point_coord) {
      /* Two linear coefficients generated by the point rasterizer. */
      apple9_put_u32(table + 4 + binding++ * 4,
                     0x4d | ((components + 1 + reads_primitive_id + reads_z) << 16));
   }
   unsigned coefficient = components + 1 + reads_primitive_id + reads_z +
                          2 * reads_point_coord;
   /* Barycentric weights of vertices 0 and 1: source 5 with the perspective
    * (7) or linear (3) shade model. */
   for (unsigned linear = 0; linear < 2; ++linear) {
      if (!(fragment->barycentric_mask & BITFIELD_BIT(linear)))
         continue;
      apple9_put_u32(table + 4 + binding++ * 4,
                     1 | ((linear ? 3 : 7) << 2) | (5 << 5) | (coefficient << 16));
      coefficient += 2;
   }
   /* Shade model 6 sets a scalar's plane terms to the raw values of vertices
    * 0, 1 and 2. */
   for (unsigned i = 0; i < fragment->per_vertex_count; ++i) {
      unsigned source = fragment->per_vertex[i] + 1 + reads_z;
      apple9_put_u32(table + 4 + binding++ * 4,
                     (6 << 2) | (source << 8) | (coefficient++ << 16));
   }
   assert(4 + 4 * binding <= APPLE9_CF_BINDINGS_SIZE);
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
agx_apple9_stage_launch_cache_init(struct agx_apple9_stage_launch_cache *c)
{
   memset(c, 0, sizeof(*c));
   simple_mtx_init(&c->lock, mtx_plain);
}

void
agx_apple9_stage_launch_cache_finish(struct agx_apple9_stage_launch_cache *c)
{
   simple_mtx_destroy(&c->lock);
}

static void
apple9_graphics_launch_params(struct agx_apple9_launch_parameters *params,
                              const struct agx_device *dev,
                              const struct agx_apple9_render_stage *shader,
                              bool fragment, unsigned tile_bytes,
                              unsigned samples)
{
   *params = (struct agx_apple9_launch_parameters){
      .preamble_address = shader->preamble_size
         ? shader->bo->va->addr + shader->preamble_offset : 0,
      .shader_base = dev->shader_base,
      .publication_count = shader->publication_count,
      .frame_extent_a = MAX2(shader->scratch_size, shader->atomic_frame_size),
      .frame_extent_b = shader->scratch_size,
      .tile_bytes = fragment ? tile_bytes : 0,
      .samples = fragment ? samples : 0,
   };
}

/* Linear search of the first count published templates. */
static const struct agx_apple9_launch_template *
apple9_stage_launch_find(struct agx_apple9_stage_launch_cache *c,
                         unsigned count, unsigned tile_bytes, unsigned samples)
{
   for (unsigned i = 0; i < count; ++i) {
      if (c->entry[i].tile_bytes == tile_bytes &&
          c->entry[i].samples == samples)
         return &c->entry[i].tmpl;
   }
   return NULL;
}

/* The stage's template for this tile allocation and sample count, building
 * it on first use; NULL if the cache is full or the parameters are rejected.
 * Published entries are immutable: lookups read the count with acquire
 * ordering and take no lock; only building locks. */
static const struct agx_apple9_launch_template *
apple9_stage_launch_template(const struct agx_device *dev,
                             const struct agx_apple9_render_stage *shader,
                             bool fragment, unsigned tile_bytes,
                             unsigned samples)
{
   struct agx_apple9_stage_launch_cache *c = shader->launch_cache;
   if (!fragment)
      tile_bytes = samples = 0;
   const struct agx_apple9_launch_template *found = apple9_stage_launch_find(
      c, __atomic_load_n(&c->count, __ATOMIC_ACQUIRE), tile_bytes, samples);
   if (found)
      return found;
   simple_mtx_lock(&c->lock);
   unsigned i = c->count;
   found = apple9_stage_launch_find(c, i, tile_bytes, samples);
   if (!found && i < AGX_APPLE9_STAGE_LAUNCH_TEMPLATES) {
      struct agx_apple9_launch_parameters params;
      apple9_graphics_launch_params(&params, dev, shader, fragment,
                                    tile_bytes, samples);
      if (agx_apple9_launch_template_init(
             &c->entry[i].tmpl, AGX_APPLE9_GRAPHICS_LAUNCH_SIZE,
             fragment ? AGX_APPLE9_LAUNCH_FRAGMENT : AGX_APPLE9_LAUNCH_VERTEX,
             &params)) {
         c->entry[i].tile_bytes = tile_bytes;
         c->entry[i].samples = samples;
         found = &c->entry[i].tmpl;
         __atomic_store_n(&c->count, i + 1, __ATOMIC_RELEASE);
      }
   }
   simple_mtx_unlock(&c->lock);
   return found;
}

bool
agx_apple9_prepare_graphics_stage(
   struct agx_device *dev, void *allocator,
   struct agx_ptr (*alloc)(void *, size_t, size_t),
   const struct agx_apple9_render_stage *shader, bool fragment,
   uint64_t table, uint64_t textures, uint64_t samplers,
   unsigned tile_bytes, unsigned samples, uint32_t *launch,
   agx_apple9_cached_alloc cached_alloc)
{
   if (!shader->bo || !shader->publication_count_valid)
      return false;
   unsigned entry = apple9_shader_entry(dev, allocator, shader->bo, false);
   if (!entry)
      return false;
   if (shader->preamble_size > AGX_APPLE9_MAX_GRAPHICS_PREAMBLE_BYTES ||
       shader->preamble_offset > shader->binary_size ||
       shader->preamble_size > shader->binary_size - shader->preamble_offset) {
      fprintf(stderr, "Apple9 draw: stage %u invalid preamble offset %u size %u in binary %zu\n",
              fragment, shader->preamble_offset, shader->preamble_size,
              (size_t)shader->binary_size);
      return false;
   }
   const unsigned launch_size = AGX_APPLE9_GRAPHICS_LAUNCH_SIZE;
   struct agx_apple9_launch_parameters params;
   apple9_graphics_launch_params(&params, dev, shader, fragment, tile_bytes,
                                 samples);
   params.entry_offset = entry;

   /* With a template, only the roots address, entry and program address are
    * per launch: emit the program next to fresh roots, without encoding it
    * or hashing it into the upload cache. */
   const struct agx_apple9_launch_template *tmpl =
      shader->launch_cache
         ? apple9_stage_launch_template(dev, shader, fragment, tile_bytes,
                                        samples)
         : NULL;
   if (tmpl) {
      struct agx_ptr state = alloc(allocator, 0x100 + launch_size, 64);
      if (!state.cpu)
         return false;
      uint8_t *roots = state.cpu;
      memset(roots, 0, 0x100);
      apple9_put_u64(roots, textures);
      apple9_put_u64(roots + 8, samplers);
      apple9_put_u64(roots + 0x10, table);
      if (!agx_apple9_launch_template_emit(roots + 0x100, tmpl, state.gpu,
                                           entry, state.gpu + 0x100))
         return false;
      if (debug_get_option_apple9_launch_check()) {
         uint8_t ref[AGX_APPLE9_GRAPHICS_LAUNCH_SIZE];
         params.resource_table = state.gpu;
         params.launch_address = state.gpu + 0x100;
         if (!agx_apple9_launch_build(
                ref, launch_size,
                fragment ? AGX_APPLE9_LAUNCH_FRAGMENT : AGX_APPLE9_LAUNCH_VERTEX,
                &params) ||
             memcmp(ref, roots + 0x100, launch_size)) {
            fprintf(stderr, "HK_APPLE9_LAUNCH_CHECK: graphics template launch differs\n");
            abort();
         }
      }
      *launch = agx_usc_addr(dev, state.gpu + 0x100);
      return true;
   }

   uint8_t canonical[0x100 + AGX_APPLE9_GRAPHICS_LAUNCH_SIZE] = {0};
   uint8_t *roots = canonical;
   apple9_put_u64(roots, textures);
   apple9_put_u64(roots + 8, samplers);
   apple9_put_u64(roots + 0x10, table);
   params.launch_address = dev->shader_base + 0x100;
   params.resource_table = dev->shader_base;
   bool is_new = true;
   struct agx_ptr state;
   if (cached_alloc) {
      if (!agx_apple9_launch_build(
             canonical + 0x100, launch_size,
             fragment ? AGX_APPLE9_LAUNCH_FRAGMENT : AGX_APPLE9_LAUNCH_VERTEX,
             &params))
         return false;
      state = cached_alloc(allocator, AGX_APPLE9_STATE_LAUNCH, canonical,
                           sizeof(canonical), 64, &is_new);
   } else {
      state = alloc(allocator, sizeof(canonical), 64);
   }
   if (!state.cpu)
      return false;
   if (!is_new) {
      *launch = agx_usc_addr(dev, state.gpu + 0x100);
      return true;
   }
   roots = state.cpu;
   memcpy(roots, canonical, 0x100);
   params.launch_address = state.gpu + 0x100;
   params.resource_table = state.gpu;
   if (!agx_apple9_launch_build(
          roots + 0x100, launch_size,
          fragment ? AGX_APPLE9_LAUNCH_FRAGMENT : AGX_APPLE9_LAUNCH_VERTEX,
          &params)) {
      fprintf(stderr,
              "Apple9 draw: stage %u launch rejected: entry=0x%x "
              "base=0x%llx roots=0x%llx launch=0x%llx preamble=0x%llx "
              "publications=%u scratch=%u tile=%u samples=%u\n",
              fragment, params.entry_offset,
              (unsigned long long)params.shader_base,
              (unsigned long long)params.resource_table,
              (unsigned long long)params.launch_address,
              (unsigned long long)params.preamble_address,
              params.publication_count, params.frame_extent_a,
              params.tile_bytes, params.samples);
      return false;
   }
   *launch = agx_usc_addr(dev, state.gpu + 0x100);
   return true;
}

/* The PPP record bytes that depend only on the linked stages: header,
 * varying group counts and the direct bind group with its stage-derived bits.
 * Draw-dependent words (coefficients, coefficient count, fragment launch,
 * tile allocation, raster, depth, stencil and viewports) are applied per draw
 * on top. */
void
agx_apple9_build_ppp_static(const struct agx_apple9_render_pipeline *pipeline,
                            uint8_t out[AGX_APPLE9_PPP_STATIC_SIZE])
{
   uint8_t *ppp = out;
   memset(ppp, 0, AGX_APPLE9_PPP_STATIC_SIZE);
   apple9_put_u32(ppp, 0x10040000);
   unsigned counts[3] = {0};
   const struct agx_apple9_varying_layout *varyings = &pipeline->vertex->varyings;
   for (unsigned i = 0; i < ARRAY_SIZE(varyings->mask); ++i) {
      assert(varyings->group[i] < ARRAY_SIZE(counts));
      counts[varyings->group[i]] += util_bitcount(varyings->mask[i]);
   }
   apple9_put_u32(ppp + 4, counts[0] | (counts[1] << 8) | (counts[2] << 16));
   uint8_t *group = ppp + 0x40;
   apple9_build_direct_bind_group(group, pipeline->vertex->varying_components);
   if (pipeline->fragment->apple9_reads_z)
      apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (1u << 21));
   if (pipeline->fragment->reads_primitive_id)
      apple9_put_u32(group + 0x68, apple9_get_u32(group + 0x68) | (1u << 12));
   if (pipeline->vertex->writes_point_size) {
      apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (1u << 18));
   }
   if (pipeline->vertex->writes_layer_viewport)
      apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (3u << 19));
   apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) |
                  BITFIELD_MASK(pipeline->vertex->clip_distance_count));
   apple9_put_u32(group + 0x2c, 4 + pipeline->vertex->varying_components +
                  pipeline->vertex->writes_point_size + pipeline->vertex->clip_distance_count +
                  pipeline->vertex->writes_layer_viewport);
   /* Match the textured setup's native state, including with two user
    * varyings. This field's full resource-count formula is unresolved;
    * the varying-only estimate is insufficient to describe the captures. */
   apple9_put_u32(group + 0x18, MAX2(apple9_get_u32(group + 0x18), 1));
}

/* Launches, roots, coefficients and PPP records are immutable batch storage.
 * The selected entry has the lifetime of its independently owned shader BO. */
bool
agx_apple9_prepare_draw_alloc(
   struct agx_device *dev, void *usc_allocator,
   struct agx_ptr (*usc_alloc)(void *, size_t, size_t), void *context_allocator,
   struct agx_ptr (*context_alloc)(void *, size_t, size_t),
                        const struct agx_apple9_render_pipeline *pipeline,
                        struct agx_apple9_uniform_draw *draw,
                        const struct agx_apple9_uniform_draw *previous,
   agx_apple9_cached_alloc cached_alloc)
{
   const struct agx_apple9_render_stage *stages[] = {pipeline->vertex,
                                                     pipeline->fragment};
   unsigned tile_bytes = 0;
   for (unsigned rt = 0; rt < pipeline->fragment->render_targets; rt++)
      tile_bytes += 4 * agx_apple9_color_words(pipeline->color_formats[rt]);
   unsigned samples = MAX2(pipeline->samples, 1);
   /* Each sample holds up to 128 bytes at every sample count; the default
    * micro-tiles fit such allocations in tile memory. */
   if (tile_bytes > 128) {
      fprintf(stderr, "Apple9 draw: unsupported tile allocation %u bytes, %u samples\n",
              tile_bytes, samples);
      return false;
   }
   draw->samples = samples;
   draw->tile_bytes = tile_bytes;
   for (unsigned stage = 0; stage < 2; stage++) {
      const struct agx_apple9_render_stage *shader = stages[stage];
      if (!shader->bo || !shader->publication_count_valid) {
         fprintf(stderr, "Apple9 draw: stage %u missing shader BO or publication count\n",
                 stage);
         return false;
      }
      draw->program_id[stage] = shader->program_id;
      draw->code[stage] = shader->bo->va->addr;
   }
   for (unsigned stage = 0; stage < 2; stage++) {
      const struct agx_apple9_render_stage *shader = stages[stage];
      uint64_t table = stage ? draw->fragment_table : draw->vertex_table;
      if (agx_apple9_draw_reuses_launch(previous, draw, stage)) {
         draw->launch[stage] = previous->launch[stage];
         continue;
      }
      if (!agx_apple9_prepare_graphics_stage(
             dev, usc_allocator, usc_alloc, shader, stage != 0, table,
             draw->texture_table[stage], draw->sampler_table[stage],
             agx_apple9_tile_sample_bytes(tile_bytes, samples), samples,
             &draw->launch[stage], cached_alloc))
         return false;
   }
   draw->provoking_vertex = apple9_provoking_vertex(pipeline->primitive,
                                                   draw->flatshade_first);
   bool reuse_coefficients = agx_apple9_draw_reuses_coefficients(previous, draw);
   unsigned cf_count;
   if (reuse_coefficients) {
      draw->coefficients = previous->coefficients;
      cf_count = previous->cf_count;
   } else {
      uint8_t contents[APPLE9_CF_BINDINGS_SIZE];
      cf_count = apple9_build_cf_bindings(
         contents, pipeline->vertex->varying_components,
         pipeline->fragment->apple9_linear_mask,
         pipeline->fragment->apple9_flat_mask, draw->provoking_vertex,
         pipeline->fragment->apple9_reads_z,
         pipeline->fragment->reads_point_coord, pipeline->fragment->reads_primitive_id,
         pipeline->fragment);
      bool is_new = true;
      struct agx_ptr cf = cached_alloc
         ? cached_alloc(usc_allocator, AGX_APPLE9_STATE_COEFFICIENTS,
                        contents, sizeof(contents), 64, &is_new)
         : usc_alloc(usc_allocator, sizeof(contents), 64);
      if (!cf.cpu)
         return false;
      if (is_new)
         memcpy(cf.cpu, contents, sizeof(contents));
      draw->coefficients = agx_usc_addr(dev, cf.gpu);
   }
   draw->cf_count = cf_count;
   uint8_t *ppp = draw->ppp_record;
   /* Stage-derived bytes come from the link (or are built here); the rest of
    * the record starts zeroed and takes the draw's state. */
   if (pipeline->ppp_static) {
      memcpy(ppp, pipeline->ppp_static, AGX_APPLE9_PPP_STATIC_SIZE);
#ifndef NDEBUG
      uint8_t check[AGX_APPLE9_PPP_STATIC_SIZE];
      agx_apple9_build_ppp_static(pipeline, check);
      assert(!memcmp(check, ppp, sizeof(check)) && "stale linked PPP template");
#endif
   } else {
      agx_apple9_build_ppp_static(pipeline, ppp);
   }
   memset(ppp + AGX_APPLE9_PPP_STATIC_SIZE, 0,
          sizeof(draw->ppp_record) - AGX_APPLE9_PPP_STATIC_SIZE);
   uint8_t *group = ppp + 0x40;
   apple9_put_u32(group + 8, draw->coefficients);
   apple9_put_u32(group + 4,
                  (apple9_get_u32(group + 4) & 0xffff) | (cf_count << 16));
   apple9_put_u32(group + 0x14, draw->launch[1] / 0x40);
   apple9_put_u32(group + 0x5c, tile_bytes ? 0x1ffff : 0);
   /* Discards and explicit per-sample stores preserve omitted samples and
    * require ordered tile access, just like blending. Otherwise opaque tag
    * visibility can lose the background clear when alpha-to-coverage kills
    * a single-sample fragment, or lose mixed stencil coverage after a flush. */
   if (draw->reads_tile || draw->uses_discard || samples > 1)
      apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | (1u << 29));
   /* Authored Metal discard traces: select punch-through and disable
    * triangle merging so shader coverage controls late tests. */
   if (draw->uses_discard)
      apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | 0x44000000u);
   /* Keep derivative helper quads within one primitive. Merging fragments
    * with different interpolation planes corrupts implicit texture LOD. */
   bool unfilled = draw->object_type == AGX_OBJECT_TYPE_TRIANGLE &&
                   ((draw->depth_face[0] | draw->depth_face[1]) & (3u << 18));
   if (draw->disable_tri_merging || unfilled ||
       draw->object_type != AGX_OBJECT_TYPE_TRIANGLE)
      apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | (1u << 26));
   /* Native raster packet: cull front/back bits 0/1, provoking vertex
    * bits 7/8, clipping/clamping bits 10/11, primitive MSAA bit 15, front
    * winding bit 16. */
   apple9_put_u32(group + 0x70, (apple9_get_u32(group + 0x70) & ~0x38d83u) |
                                   ((draw->provoking_vertex + 1u) << 7) |
                                   draw->raster_control);
   apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) |
                                   (draw->visibility_mode << 14) |
                                   (draw->depth_control & (1u << 21)));
   apple9_put_u32(group + 0x48, (uint32_t)draw->occlusion_index << 17);
   apple9_put_u32(group + 0x34, draw->depth_control |
                                   (unfilled ? 1u << 26 : 0));
   apple9_put_u32(group + 0x38, draw->depth_face[0]);
   apple9_put_u32(group + 0x3c, draw->stencil[0]);
   apple9_put_u32(group + 0x40, draw->depth_face[1]);
   apple9_put_u32(group + 0x44, draw->stencil[1]);
   for (unsigned face = 0; face < 2; ++face) {
      uint8_t *face2 = group + 0x54 + face * 4;
      unsigned polygon_mode = (draw->depth_face[face] >> 18) & 3;
      unsigned object_type = draw->object_type;
      if (object_type == AGX_OBJECT_TYPE_TRIANGLE) {
         if (polygon_mode == AGX_POLYGON_MODE_LINE)
            object_type = AGX_OBJECT_TYPE_LINE_FILLED_TRIANGLE;
         else if (polygon_mode == AGX_POLYGON_MODE_POINT)
            object_type = AGX_OBJECT_TYPE_POINT_FILLED_TRIANGLE;
      }
      apple9_put_u32(face2, (apple9_get_u32(face2) & 0x0f33ffffu) |
                               (polygon_mode << 18) |
                               ((draw->writes_depth ? 0u : 3u) << 22) |
                               (object_type << 28));
   }
   /* Viewport block: a count word, one region clip pair per viewport, a
    * word set when there are several viewports, then {translate, scale} for
    * each axis per viewport. A single viewport keeps its block at 0xc0;
    * several are appended after the fixed state. The preceding PPP update
    * carries the block length in words.
    */
   unsigned viewports = MAX2(draw->viewport_count, 1);
   assert(viewports <= AGX_APPLE9_MAX_VIEWPORTS);
   uint8_t *vp = ppp + (viewports > 1 ? 0x100 : 0xc0);
   draw->ppp_viewport_offset = vp - ppp;
   draw->ppp_viewport_words = 8 * viewports + 2;
   draw->ppp_size = viewports > 1
      ? ALIGN_POT(0x100 + draw->ppp_viewport_words * 4, 64) : 0x100;
   apple9_put_u32(vp, ((viewports - 1) << 12) | 0xc00);
   /* Region clip is tile-granular. The scissor array supplies exact pixel
    * bounds, including empty rectangles and partial edge tiles. */
   for (unsigned v = 0; v < viewports; ++v) {
      for (unsigned axis = 0; axis < 2; ++axis) {
         uint32_t lo = draw->scissor_min[axis] / 32;
         uint32_t hi = DIV_ROUND_UP(MAX2(draw->scissor_max[axis], 1), 32) - 1;
         apple9_put_u32(vp + 4 + v * 8 + axis * 4,
                        (axis ? 0 : 0x80000000u) | (lo << 16) | hi);
      }
   }
   apple9_put_u32(vp + 4 + viewports * 8, viewports > 1);
   uint8_t *transforms = vp + 8 + viewports * 8;
   for (unsigned axis = 0; axis < 3; ++axis) {
      apple9_put_f32(transforms + axis * 8, draw->viewport_translate[axis]);
      apple9_put_f32(transforms + axis * 8 + 4, draw->viewport_scale[axis]);
   }
   for (unsigned v = 1; v < viewports; ++v) {
      for (unsigned i = 0; i < 6; ++i)
         apple9_put_f32(transforms + v * 0x18 + i * 4,
                        draw->viewports[v - 1][i]);
   }
   /* PPP depth-bias/scissor record: header followed by two 16-bit indices. */
   apple9_put_u32(ppp + 0xf0, 0x100);
   apple9_put_u32(ppp + 0xf4, draw->scissor_index |
                                 ((uint32_t)draw->depth_bias_index << 16));
   /* Compare only this canonical 256-byte record, in CPU storage. This
    * bounded check reuses an immutable allocation; it neither scans an arena
    * nor derives byte-range patches. A new batch has no predecessor. */
   if (previous && previous->ppp_size == draw->ppp_size &&
       !memcmp(previous->ppp_record, ppp, draw->ppp_size)) {
      draw->ppp = previous->ppp;
   } else {
      bool is_new = true;
      struct agx_ptr record = cached_alloc
         ? cached_alloc(context_allocator, AGX_APPLE9_STATE_PPP,
                        ppp, draw->ppp_size, 64, &is_new)
         : context_alloc(context_allocator, draw->ppp_size, 64);
      if (!record.cpu || record.gpu < AGX_APPLE9_RENDER_CONTEXT_BASE ||
          record.gpu - AGX_APPLE9_RENDER_CONTEXT_BASE >
             UINT32_MAX - draw->ppp_size + 1)
         return false;
      if (is_new)
         memcpy(record.cpu, ppp, draw->ppp_size);
      draw->ppp = record.gpu - AGX_APPLE9_RENDER_CONTEXT_BASE;
   }
   return true;
}

bool
agx_apple9_prepare_draw(struct agx_device *dev, struct agx_pool *usc_pool,
                        struct agx_pool *context_pool,
                        const struct agx_apple9_render_pipeline *pipeline,
                        struct agx_apple9_uniform_draw *draw,
                        const struct agx_apple9_uniform_draw *previous)
{
   return agx_apple9_prepare_draw_alloc(
      dev, usc_pool, apple9_pool_alloc, context_pool, apple9_pool_alloc,
      pipeline, draw, previous, NULL);
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
static_assert(AGX_APPLE9_PPP_RECORD_MAX >=
                 ALIGN_POT(0x100 + (8 * AGX_APPLE9_MAX_VIEWPORTS + 2) * 4, 64),
              "PPP record holds the largest viewport block");

static const struct agx_apple9_ppp_update direct_ppp[] = {
   {0x00, 0x700}, {0x40, 0x500}, {0x5c, 0x700}, {0x70, 0x500},
   {0x8c, 0xa00}, {0xc0, 0x300}, {0xa0, 0x200}, {0xac, 0x200},
};

bool
agx_apple9_link_render_pipeline(
   struct agx_apple9_render_pipeline *pipeline,
   const struct agx_apple9_render_stage *vertex,
   const struct agx_apple9_render_stage *fragment)
{
   return agx_apple9_link_render_pipeline_with_prolog(pipeline, NULL, vertex,
                                                      fragment);
}

bool
agx_apple9_link_render_pipeline_with_prolog(
   struct agx_apple9_render_pipeline *pipeline,
   const struct agx_apple9_render_stage *vertex_prolog,
   const struct agx_apple9_render_stage *vertex,
   const struct agx_apple9_render_stage *fragment)
{
   if (!pipeline || !vertex || !fragment || !vertex->binary ||
       !vertex->binary_size || !fragment->binary || !fragment->binary_size ||
       vertex->binary_size > UINT32_MAX || fragment->binary_size > UINT32_MAX ||
       (vertex_prolog && (vertex_prolog->binary || vertex_prolog->binary_size)))
      return false;

   /*
    * Apple9 compacts the cross-stage interface to scalar UVS slots.  Position
    * occupies slots 0..3; compacted user scalars follow in semantic order.
    */
   if (vertex->position_components != 4 ||
       vertex->varying_components > AGX_APPLE9_MAX_VARYING_COMPONENTS ||
       vertex->clip_distance_count > 8 ||
       fragment->position_components != 0 ||
       fragment->varying_components != vertex->varying_components ||
       memcmp(&vertex->varyings, &fragment->varyings, sizeof(vertex->varyings)) ||
       fragment->render_targets > 8)
      return false;

   const unsigned scalar_outputs = 4 + vertex->varying_components +
                                   vertex->writes_point_size + vertex->clip_distance_count +
                                   vertex->writes_layer_viewport;
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
   assert(pipeline && pipeline->vertex->binary && pipeline->fragment->binary);
   return 0x78 + (pipeline->index_size ? 20 : 0);
}

static uint8_t *
agx_apple9_emit_draw_state(uint8_t *out,
                           const struct agx_apple9_render_pipeline *pipeline)
{
   assert(pipeline && pipeline->vertex->binary && pipeline->fragment->binary);
   assert(pipeline->ppp && !(pipeline->ppp & 0x3f));

   uint32_t header[] = {
      0x4000002e, /* direct vertex state, Apple9 envelope */
      0x00000000, /* vertex word 0 */
      pipeline->pipeline_word,
      pipeline->vertex_launch,
      pipeline->vertex_state_class,
      apple9_provoking_vertex(pipeline->primitive, pipeline->flatshade_first) |
         (pipeline->fragment->reads_primitive_id ? (1u << 6) : 0),
      0x00000000, /* VDM padding */
      0x00000500, /* first PPP state update header */
   };
   memcpy(out, header, sizeof(header));
   out += sizeof(header);

   /* Each update names a block and the length in words of the next one;
    * the viewport block follows the entry at 0x8c.
    */
   struct agx_apple9_ppp_update ppp[ARRAY_SIZE(direct_ppp)];
   for (unsigned i = 0; i < ARRAY_SIZE(ppp); i++) {
      ppp[i] = direct_ppp[i];
      if (pipeline->ppp_viewport_words && ppp[i].relative == 0x8c)
         ppp[i].control = pipeline->ppp_viewport_words << 8;
      if (pipeline->ppp_viewport_offset && ppp[i].relative == 0xc0)
         ppp[i].relative = pipeline->ppp_viewport_offset;
      ppp[i].relative += pipeline->ppp;
   }
   memcpy(out, ppp, sizeof(ppp));
   out += sizeof(ppp);

   apple9_put_u32(out, pipeline->ppp + 0xf0);
   return out + 4;
}

uint8_t *
agx_apple9_emit_direct_draw(uint8_t *out,
                            const struct agx_apple9_render_pipeline *pipeline,
                            unsigned vertex_count, unsigned instance_count,
                            unsigned vertex_start)
{
   assert(vertex_count > 0 && instance_count > 0);
   out = agx_apple9_emit_draw_state(out, pipeline);
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

uint8_t *
agx_apple9_emit_indirect_draw(uint8_t *out,
                              const struct agx_apple9_render_pipeline *pipeline,
                              uint64_t indirect)
{
   assert(indirect && !(indirect & 3));
   out = agx_apple9_emit_draw_state(out, pipeline);
   if (pipeline->index_size) {
      assert(pipeline->index_size == 1 || pipeline->index_size == 2 ||
             pipeline->index_size == 4);
      assert(pipeline->index_buffer >= (1ull << 40) &&
             pipeline->index_buffer < (1ull << 40) + (1ull << 32));
      assert(pipeline->index_extent);
      uint32_t draw[] = {
         0x40000001,
         pipeline->restart_index,
         0x64300000 | (util_logbase2(pipeline->index_size) << 17) |
            (pipeline->primitive << 8) |
            (pipeline->primitive_restart ? (1u << 16) : 0),
         (uint32_t)pipeline->index_buffer,
         indirect >> 32,
         indirect,
         DIV_ROUND_UP(pipeline->index_extent +
                      (pipeline->index_buffer & 3), 4) - 1,
         1,
         0xc0000000,
      };
      memcpy(out, draw, sizeof(draw));
      out += sizeof(draw);
   } else {
      /* As in CDM, the indirect pointer is high word then low word. The
       * VDM consumes count, instances and first vertex at execution time. */
      uint32_t draw[] = {
         0x64040000 | (pipeline->primitive << 8),
         indirect >> 32,
         indirect,
         0xc0000000,
      };
      memcpy(out, draw, sizeof(draw));
      out += sizeof(draw);
   }
   return out;
}
