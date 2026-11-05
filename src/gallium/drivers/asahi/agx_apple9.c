/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9.h"
#include "asahi/lib/pool.h"
#include "pipe/p_defines.h"
#include "asahi/lib/agx_apple9_launch.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/libagx/libagx_dgc.h"
#include "util/compress.h"
#include "util/list.h"
#include "util/u_math.h"
#include "agx_device.h"
#include "asahi/layout/layout.h"

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

static inline void
apple9_put_f32(void *ptr, float value)
{
   memcpy(ptr, &value, sizeof(value));
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

/* The default coefficient table occupies the end of the graphics resource
 * page, after per-draw argument records. Draw-specific tables live in their
 * own slots on the following pages so mixed interpolation can use one
 * descriptor per scalar without overwriting the next draw's roots. */
#define APPLE9_CF_BINDINGS 0x203d00u
#define APPLE9_CF_BINDINGS_SIZE 0x220u
static_assert(4 + 4 * (AGX_APPLE9_MAX_VARYING_COMPONENTS + 4) <=
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
                         bool flatshade_first, bool reads_z, bool reads_point_coord,
                         bool reads_primitive_id)
{
   assert(components <= AGX_APPLE9_MAX_VARYING_COMPONENTS);
   memset(table, 0, APPLE9_CF_BINDINGS_SIZE);
   unsigned slots = components + 1 + reads_z + 2 * reads_point_coord + reads_primitive_id;
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

/* Launches, roots, coefficients and PPP records are immutable batch storage.
 * The selected entry has the lifetime of its independently owned shader BO. */
bool
agx_apple9_prepare_draw(struct agx_device *dev, struct agx_pool *usc_pool,
                        struct agx_pool *context_pool,
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
   if (tile_bytes > (samples == 1 ? 128 : 64)) {
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
   unsigned offsets[2];
   for (unsigned stage = 0; stage < 2; stage++) {
      offsets[stage] = agx_apple9_shader_entry(dev, stages[stage]->bo, false);
      if (!offsets[stage]) {
         fprintf(stderr, "Apple9 draw: stage %u entry allocation failed for code 0x%llx\n",
                 stage, (unsigned long long)draw->code[stage]);
         return false;
      }
   }
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
          shader->preamble_size > shader->binary_size - shader->preamble_offset) {
         fprintf(stderr, "Apple9 draw: stage %u invalid preamble offset %u size %u in binary %zu\n",
                 stage, shader->preamble_offset, shader->preamble_size,
                 (size_t)shader->binary_size);
         return false;
      }
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
             &params)) {
         fprintf(stderr,
                 "Apple9 draw: stage %u launch rejected: entry=0x%x "
                 "base=0x%llx roots=0x%llx launch=0x%llx preamble=0x%llx "
                 "publications=%u scratch=%u tile=%u samples=%u\n",
                 stage, params.entry_offset,
                 (unsigned long long)params.shader_base,
                 (unsigned long long)params.resource_table,
                 (unsigned long long)params.launch_address,
                 (unsigned long long)params.preamble_address,
                 params.publication_count, params.frame_extent_a,
                 params.tile_bytes, params.samples);
         return false;
      }
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
         pipeline->fragment.reads_point_coord, pipeline->fragment.reads_primitive_id);
      draw->coefficients = agx_usc_addr(dev, cf.gpu);
   }
   draw->cf_count = cf_count;
   uint8_t *ppp = draw->ppp_record;
   memset(ppp, 0, sizeof(draw->ppp_record));
   apple9_put_u32(ppp, 0x10040000);
   unsigned counts[3] = {0};
   const struct agx_apple9_varying_layout *varyings = &pipeline->vertex.varyings;
   for (unsigned i = 0; i < ARRAY_SIZE(varyings->mask); ++i) {
      assert(varyings->group[i] < ARRAY_SIZE(counts));
      counts[varyings->group[i]] += util_bitcount(varyings->mask[i]);
   }
   apple9_put_u32(ppp + 4, counts[0] | (counts[1] << 8) | (counts[2] << 16));
   uint8_t *group = ppp + 0x40;
   apple9_build_direct_bind_group(group, pipeline->vertex.varying_components);
   apple9_put_u32(group + 8, draw->coefficients);
   if (pipeline->fragment.apple9_reads_z)
      apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (1u << 21));
   if (pipeline->fragment.reads_primitive_id)
      apple9_put_u32(group + 0x68, apple9_get_u32(group + 0x68) | (1u << 12));
   if (pipeline->vertex.writes_point_size) {
      apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (1u << 18));
   }
   if (pipeline->vertex.writes_layer_viewport)
      apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) | (3u << 19));
   apple9_put_u32(group + 0x20, apple9_get_u32(group + 0x20) |
                  BITFIELD_MASK(pipeline->vertex.clip_distance_count));
   apple9_put_u32(group + 0x2c, 4 + pipeline->vertex.varying_components +
                  pipeline->vertex.writes_point_size + pipeline->vertex.clip_distance_count +
                  pipeline->vertex.writes_layer_viewport);
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
   bool unfilled = draw->object_type == AGX_OBJECT_TYPE_TRIANGLE &&
                   ((draw->depth_face[0] | draw->depth_face[1]) & (3u << 18));
   if (draw->disable_tri_merging || unfilled ||
       draw->object_type != AGX_OBJECT_TYPE_TRIANGLE)
      apple9_put_u32(group + 0x50, apple9_get_u32(group + 0x50) | (1u << 26));
   /* Native raster packet: cull front/back bits 0/1, provoking vertex
    * bits 7/8, clipping/clamping bits 10/11, front winding bit 16. */
   apple9_put_u32(group + 0x70, (apple9_get_u32(group + 0x70) & ~0x30d83u) |
                                   ((draw->flatshade_first ? 1u : 3u) << 7) |
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
       vertex.clip_distance_count > 8 ||
       fragment.position_components != 0 ||
       fragment.varying_components != vertex.varying_components ||
       memcmp(&vertex.varyings, &fragment.varyings, sizeof(vertex.varyings)) ||
       fragment.render_targets < 1 || fragment.render_targets > 8)
      return false;

   const unsigned scalar_outputs = 4 + vertex.varying_components +
                                   vertex.writes_point_size + vertex.clip_distance_count +
                                   vertex.writes_layer_viewport;
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
   return 0x78 + (pipeline->index_size ? 20 : 0);
}

static uint8_t *
agx_apple9_emit_draw_state(uint8_t *out,
                           const struct agx_apple9_render_pipeline *pipeline)
{
   assert(pipeline && pipeline->vertex.binary && pipeline->fragment.binary);
   assert(pipeline->ppp && !(pipeline->ppp & 0x3f));

   uint32_t header[] = {
      0x4000002e, /* direct vertex state, Apple9 envelope */
      0x00000000, /* vertex word 0 */
      pipeline->pipeline_word,
      pipeline->vertex_launch,
      pipeline->vertex_state_class,
      (pipeline->flatshade_first ? 0u : 2u) |
         (pipeline->fragment.reads_primitive_id ? (1u << 6) : 0),
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
