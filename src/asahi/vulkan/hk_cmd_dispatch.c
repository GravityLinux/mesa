/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_feedback.h"
#include "libagx/query.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"
#include "agx_helpers.h"
#include "agx_apple9.h"
#include "hk_descriptor_set.h"
#include "pipe/p_defines.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "agx_tilebuffer.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "libagx/tessellator_fold.h"
#include "pool.h"

void
hk_cmd_buffer_begin_compute(struct hk_cmd_buffer *cmd,
                            const VkCommandBufferBeginInfo *pBeginInfo)
{
}

void
hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd)
{
   memset(&cmd->state.cs, 0, sizeof(cmd->state.cs));
}

void
hk_cmd_bind_compute_shader(struct hk_cmd_buffer *cmd,
                           struct hk_api_shader *shader)
{
   cmd->state.cs.shader = shader;
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   desc->root.constant_data[MESA_SHADER_COMPUTE] =
      hk_shader_constant_data(shader);
   desc->root_constant_stages = desc->root.constant_data[MESA_SHADER_COMPUTE]
      ? BITFIELD_BIT(MESA_SHADER_COMPUTE) : 0;
   desc->root_dirty = true;
}

/* The compute-only barrier: Metal's 0x160 mask orders later dispatches after
 * earlier ones and makes their shader writes visible to them. Consumers
 * outside the shader core (indirect grid fetches, queries, the stream end)
 * still need the full flush, so cdm_dirty stays set.
 */
void
hk_cdm_compute_barrier(struct hk_device *dev, struct hk_cs *cs)
{
   assert(cs->type == HK_CS_CDM);
   assert(cs->current + AGX_CDM_BARRIER_LENGTH < cs->end &&
          "caller must ensure space");

   uint32_t *barrier = (uint32_t *)cs->current;
   cs->current = agx_cdm_barrier(cs->current, dev->dev.chip);
   *barrier = 0x60000160;
   cs->stats.flushes++;
   cs->cdm_dirty = true;
   cs->cdm_compute_barrier = true;
}

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   assert(cs->type == HK_CS_CDM);
   assert(cs->current + AGX_CDM_BARRIER_LENGTH < cs->end &&
          "caller must ensure space");

   cs->current = agx_cdm_barrier(cs->current, dev->dev.chip);
   cs->stats.flushes++;
   cs->cdm_dirty = false;
   cs->cdm_compute_barrier = false;
   cs->head_flushed = true;
}

void
hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                            struct agx_cdm_launch_word_0_packed launch,
                            uint32_t usc, struct agx_grid grid,
                            struct agx_workgroup wg)
{
   hk_ensure_cs_has_space(cs->cmd, cs, 0x2000 /* TODO */);
   if (!cs->cmd->api_dispatch && cs->cdm_dirty)
      hk_cdm_cache_flush(dev, cs);
   cs->stats.cmds++;
   if (agx_is_indirect(grid))
      hk_cs_note_indirect(cs);

   cs->current =
      agx_cdm_launch(cs->current, dev->dev.chip, grid, wg, launch, usc);

   cs->cdm_dirty = true;
   if (!cs->cmd->api_dispatch)
      hk_cdm_cache_flush(dev, cs);
}

void
hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                     struct agx_shader_info *info, uint32_t usc,
                     struct agx_grid grid, struct agx_workgroup local_size)
{
   struct agx_cdm_launch_word_0_packed launch;
   agx_pack(&launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = info->texture_state_count;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(info->sampler_state_count, false);
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }

   hk_dispatch_with_usc_launch(dev, cs, launch, usc, grid, local_size);
}

static struct agx_ptr
hk_apple9_alloc(void *data, size_t size, size_t alignment)
{
   return hk_pool_usc_alloc(data, size, alignment);
}

/* Per-view texture metadata read by shader queries and bounds checks. */
void
hk_apple9_texture_info(uint32_t *data, const struct agx_texture_packed *state)
{
   memset(data, 0, AGX_APPLE9_TEXTURE_INFO_STRIDE);
   agx_unpack(stderr, state, TEXTURE, image);
   data[0] = u_minify(image.width, image.first_level);
   data[1] = u_minify(image.height, image.first_level);
   data[2] = image.dimension == AGX_TEXTURE_DIMENSION_3D
      ? u_minify(image.depth, image.first_level)
      : image.layout == AGX_LAYOUT_LINEAR && image.extended
         ? image.depth_linear : image.depth;
   /* Null images use a dummy zero-page texture for safe hardware sampling.
    * Preserve their logical emptiness for shader-side LOD queries. */
   data[3] = image.address == AGX_ZERO_PAGE_ADDRESS
                ? 0 : image.last_level - image.first_level + 1;
   data[4] = (image.dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
              image.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED)
      ? (2 << image.samples) : 1;
   const unsigned channels[] = {image.swizzle_r, image.swizzle_g,
                                image.swizzle_b, image.swizzle_a};
   for (unsigned c = 0; c < 4; ++c) {
      unsigned swizzle = channels[c] == AGX_CHANNEL_0 ? PIPE_SWIZZLE_0 :
                         channels[c] == AGX_CHANNEL_1 ? PIPE_SWIZZLE_1 : channels[c];
      data[7] |= swizzle << (3 * c);
   }
}

/* Sampler state for the software sampling paths. */
bool
hk_apple9_runtime_sampler(uint32_t *data, const struct agx_sampler_packed *state,
                          bool clamp_to_edge)
{
   static const uint32_t wraps[] = {
      PIPE_TEX_WRAP_CLAMP_TO_EDGE, PIPE_TEX_WRAP_REPEAT,
      PIPE_TEX_WRAP_MIRROR_REPEAT, PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      PIPE_TEX_WRAP_CLAMP, PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE,
   };
   static const uint32_t comparisons[] = {
      PIPE_FUNC_LEQUAL, PIPE_FUNC_GEQUAL, PIPE_FUNC_LESS, PIPE_FUNC_GREATER,
      PIPE_FUNC_EQUAL, PIPE_FUNC_NOTEQUAL, PIPE_FUNC_ALWAYS, PIPE_FUNC_NEVER,
   };

   memset(data, 0, AGX_APPLE9_RUNTIME_SAMPLER_STRIDE);
   agx_unpack(stderr, state, SAMPLER, sampler);
   if (clamp_to_edge) {
      sampler.wrap_s = AGX_WRAP_CLAMP_TO_EDGE;
      sampler.wrap_t = AGX_WRAP_CLAMP_TO_EDGE;
      sampler.wrap_r = AGX_WRAP_CLAMP_TO_EDGE;
   }
   if (sampler.wrap_s >= ARRAY_SIZE(wraps) ||
       sampler.wrap_t >= ARRAY_SIZE(wraps) || sampler.wrap_r >= ARRAY_SIZE(wraps))
      return false;
   data[AGX_APPLE9_SAMP_WRAP_S] = wraps[sampler.wrap_s];
   data[AGX_APPLE9_SAMP_WRAP_T] = wraps[sampler.wrap_t];
   data[AGX_APPLE9_SAMP_WRAP_R] = wraps[sampler.wrap_r];
   data[AGX_APPLE9_SAMP_MIN] = sampler.minify == AGX_FILTER_LINEAR
      ? PIPE_TEX_FILTER_LINEAR : PIPE_TEX_FILTER_NEAREST;
   data[AGX_APPLE9_SAMP_MAG] = sampler.magnify == AGX_FILTER_LINEAR
      ? PIPE_TEX_FILTER_LINEAR : PIPE_TEX_FILTER_NEAREST;
   data[AGX_APPLE9_SAMP_MIP] = sampler.mip_filter == AGX_MIP_FILTER_NONE
      ? PIPE_TEX_MIPFILTER_NONE : sampler.mip_filter == AGX_MIP_FILTER_LINEAR
      ? PIPE_TEX_MIPFILTER_LINEAR : PIPE_TEX_MIPFILTER_NEAREST;
   data[AGX_APPLE9_SAMP_COMPARE] = comparisons[sampler.compare_func];
   data[AGX_APPLE9_SAMP_MIN_LOD] = fui(sampler.minimum_lod);
   data[AGX_APPLE9_SAMP_MAX_LOD] = fui(sampler.maximum_lod);
   /* Honeykrisp already applies sampler bias when lowering descriptors. */
   data[AGX_APPLE9_SAMP_BIAS] = 0;
   data[AGX_APPLE9_SAMP_BORDER] = sampler.border_colour;
   data[AGX_APPLE9_SAMP_PIXEL_COORDS] = sampler.pixel_coordinates;
   data[AGX_APPLE9_SAMP_MAX_ANISOTROPY] = fui(sampler.maximum_anisotropy);
   return true;
}

bool
hk_apple9_native_sampler(void *out, const struct agx_sampler_packed *state,
                         bool clamp_to_edge, const uint32_t *border)
{
   agx_unpack(stderr, state, SAMPLER, sampler);
   /* Shader lowering normalizes coordinates using per-view dimensions. */
   sampler.pixel_coordinates = false;
   if (clamp_to_edge) {
      sampler.wrap_s = AGX_WRAP_CLAMP_TO_EDGE;
      sampler.wrap_t = AGX_WRAP_CLAMP_TO_EDGE;
      sampler.wrap_r = AGX_WRAP_CLAMP_TO_EDGE;
   }
   bool custom = sampler.border_colour == AGX_BORDER_COLOUR_CUSTOM;
   if (custom)
      sampler.border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
   memset(out, 0, AGX_APPLE9_SAMPLER_TABLE_STRIDE);
   if (!agx_apple9_pack_sampler_state(out, &sampler))
      return false;
   if (custom) {
      /* Custom border mode, with the packed colour after the 8-byte record
       * as in G13 extended samplers. */
      uint32_t modes;
      memcpy(&modes, (uint8_t *)out + 4, 4);
      modes |= (uint32_t)AGX_BORDER_COLOUR_CUSTOM << 29;
      memcpy((uint8_t *)out + 4, &modes, 4);
      if (border)
         memcpy((uint8_t *)out + 8, border, 4 * sizeof(uint32_t));
   }
   return true;
}

void
hk_apple9_compute_sysvals(struct hk_device *dev, void *out)
{
   memset(out, 0, AGX_APPLE9_RUNTIME_SYSVAL_SIZE);
   uint32_t fetch = (dev->apple9_fetch_sampler->va->addr -
                     dev->dev.shader_base) / AGX_APPLE9_SAMPLER_TABLE_STRIDE;
   memcpy((uint8_t *)out + AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET,
          &fetch, sizeof(fetch));
}

/* The line stipple pattern and factor of the draw, packed as the fragment
 * stipple test reads them. Draws that rasterize no lines after any
 * geometry, tessellation or mesh stage, or have stipple disabled, keep every
 * fragment: factor 1 with every pattern bit set.
 */
/* Rasterization state the software polygon-point stage applies itself, see
 * hk_apple9_polygon_point_gs. */
static void
hk_apple9_polygon_point_params(const struct hk_cmd_buffer *cmd,
                               struct agx_apple9_polygon_point_params *p)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   memset(p, 0, sizeof(*p));
   p->flags =
      ((dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT) ? AGX_APPLE9_PP_CULL_FRONT : 0) |
      ((dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT) ? AGX_APPLE9_PP_CULL_BACK : 0) |
      (dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE
          ? AGX_APPLE9_PP_FRONT_CCW : 0) |
      (vk_rasterization_state_depth_clip_enable(&dyn->rs)
          ? AGX_APPLE9_PP_DEPTH_CLIP : 0);
   if (dyn->rs.depth_bias.enable) {
      VkFormat format = cmd->state.gfx.render.depth_att.vk_format;
      bool force_unorm =
         dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
      bool exact = dyn->rs.depth_bias.exact;
      p->flags |= AGX_APPLE9_PP_DEPTH_BIAS;
      p->bias_constant = dyn->rs.depth_bias.constant_factor;
      p->bias_slope = dyn->rs.depth_bias.slope_factor;
      p->bias_clamp = dyn->rs.depth_bias.clamp;
      if (format == VK_FORMAT_D16_UNORM)
         p->bias_unit = 1.0f / (exact ? 65536.0f : 32768.0f);
      else if (force_unorm || format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
               format == VK_FORMAT_D24_UNORM_S8_UINT)
         p->bias_unit = 1.0f / (exact ? 16777216.0f : 8388608.0f);
      else
         p->flags |= AGX_APPLE9_PP_DEPTH_BIAS_FLOAT;
   }
   if (dyn->vp.viewport_count) {
      p->depth_range =
         dyn->vp.viewports[0].maxDepth - dyn->vp.viewports[0].minDepth;
   }
}

static_assert(HK_MAX_VIEWPORTS == AGX_APPLE9_VIEWPORT_TABLE_COUNT,
              "one window transform per viewport");

static uint32_t
hk_apple9_line_stipple_word(const struct vk_dynamic_graphics_state *dyn,
                            bool lines)
{
   if (!lines || !dyn->rs.line.stipple.enable)
      return 0x1ffff;
   return dyn->rs.line.stipple.pattern |
          ((uint32_t)MAX2(dyn->rs.line.stipple.factor, 1) << 16);
}

/* Window transforms of the bound viewports, for the vertex-stage stipple
 * start: x_w = x + w/2 + ndc.x * w/2, and likewise for y with the height,
 * which may be negative. */
static void
hk_apple9_viewport_table(const struct vk_dynamic_graphics_state *dyn,
                         float table[AGX_APPLE9_VIEWPORT_TABLE_COUNT][4])
{
   memset(table, 0, AGX_APPLE9_VIEWPORT_TABLE_COUNT * 4 * sizeof(float));
   unsigned count = MIN2(dyn->vp.viewport_count, AGX_APPLE9_VIEWPORT_TABLE_COUNT);
   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];
      table[i][0] = vp->width * 0.5f;
      table[i][1] = vp->height * 0.5f;
      table[i][2] = vp->x + vp->width * 0.5f;
      table[i][3] = vp->y + vp->height * 0.5f;
   }
}

bool
hk_apple9_upload_textures(struct hk_cmd_buffer *cmd,
                          const struct hk_shader *shader, bool graphics,
                          bool first_fan, bool lines, uint64_t *textures,
                          uint64_t *samplers, uint64_t *sysvals)
{
   /* Textures and samplers are addressed through their descriptors, see
    * hk_apple9_texture_bindings: the stage's table pointers name the USC
    * aperture, and only per-draw state goes in the sysval block. */
   if (shader->b.info.apple9_texture_mask || shader->b.info.apple9_sampler_mask)
      return false;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   *textures = *samplers = dev->dev.shader_base;
   /* A stage without sampler handles names the compiler-private fetch
    * sampler as immediate slot 0 of its sampler table. */
   if (shader->b.info.apple9_uses_texel_fetch)
      *samplers = dev->apple9_fetch_sampler->va->addr;
   if (!graphics) {
      /* Kernels read only the fixed part, built at device creation. */
      *sysvals = dev->apple9_compute_sysvals->va->addr;
      return true;
   }

   /* The graphics block holds command buffer state only. Compare its
    * inputs with the last upload: binding a pipeline or restoring meta
    * state may dirty blend constants and write enables without changing
    * either value. */
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t line_stipple = hk_apple9_line_stipple_word(dyn, lines);
   float viewports[AGX_APPLE9_VIEWPORT_TABLE_COUNT][4];
   hk_apple9_viewport_table(dyn, viewports);
   struct agx_apple9_polygon_point_params polygon_points;
   hk_apple9_polygon_point_params(cmd, &polygon_points);
   if (gfx->apple9_sysvals &&
       !memcmp(&gfx->apple9_sysvals_polygon_points, &polygon_points,
               sizeof(polygon_points)) &&
       !memcmp(gfx->apple9_sysvals_blend_constants, dyn->cb.blend_constants,
               sizeof(gfx->apple9_sysvals_blend_constants)) &&
       gfx->apple9_sysvals_color_write_enables == dyn->cb.color_write_enables &&
       gfx->apple9_sysvals_positions == gfx->apple9_sample_positions &&
       gfx->apple9_sysvals_first_fan == first_fan &&
       gfx->apple9_sysvals_line_stipple == line_stipple &&
       !memcmp(gfx->apple9_sysvals_viewports, viewports, sizeof(viewports))) {
      *sysvals = gfx->apple9_sysvals;
      goto clean;
   }
   _Alignas(64) uint8_t info[AGX_APPLE9_RUNTIME_SYSVAL_SIZE];
   hk_apple9_compute_sysvals(dev, info);
   memcpy(info, dyn->cb.blend_constants, 4 * sizeof(float));
   uint32_t enables = dyn->cb.color_write_enables;
   memcpy((uint8_t *)info + AGX_APPLE9_COLOR_WRITE_ENABLE_OFFSET,
          &enables, sizeof(enables));
   memcpy((uint8_t *)info + AGX_APPLE9_SAMPLE_POSITIONS_OFFSET,
          &gfx->apple9_sample_positions, sizeof(uint32_t));
   uint32_t fan = first_fan;
   memcpy((uint8_t *)info + AGX_APPLE9_FIRST_FAN_OFFSET, &fan, sizeof(fan));
   memcpy((uint8_t *)info + AGX_APPLE9_LINE_STIPPLE_OFFSET, &line_stipple,
          sizeof(line_stipple));
   memcpy((uint8_t *)info + AGX_APPLE9_VIEWPORT_TABLE_OFFSET, viewports,
          sizeof(viewports));
   memcpy((uint8_t *)info + AGX_APPLE9_POLYGON_POINT_OFFSET, &polygon_points,
          sizeof(polygon_points));
   *sysvals = hk_pool_upload(cmd, info, sizeof(info), 64);
   if (!*sysvals)
      return false;

   /* Publish the cache and consume its inputs only after the upload succeeds. */
   gfx->apple9_sysvals = *sysvals;
   memcpy(gfx->apple9_sysvals_blend_constants, dyn->cb.blend_constants,
          sizeof(gfx->apple9_sysvals_blend_constants));
   gfx->apple9_sysvals_color_write_enables = dyn->cb.color_write_enables;
   gfx->apple9_sysvals_positions = gfx->apple9_sample_positions;
   gfx->apple9_sysvals_first_fan = first_fan;
   gfx->apple9_sysvals_line_stipple = line_stipple;
   memcpy(gfx->apple9_sysvals_viewports, viewports, sizeof(viewports));
   gfx->apple9_sysvals_polygon_points = polygon_points;
clean:
   /* Draw-state reuse is invalidated before these bits are consumed. */
   BITSET_CLEAR(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS);
   BITSET_CLEAR(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   return true;
}

/* Report why the CDM record emitter rejected a dispatch, with the values it
 * checked, so an intermittent failure in an application log is diagnosable.
 */
static void
hk_apple9_report_dispatch_reject(const char *where, struct agx_grid grid,
                                 bool indirect_local, uint64_t launch,
                                 const uint32_t threads[3],
                                 const uint32_t local[3],
                                 const struct agx_apple9_compute_profile *p)
{
   bool indirect = agx_is_indirect(grid);
   const char *reason = agx_apple9_dispatch_reject_reason(
      launch, indirect, indirect ? grid.ptr : 0, indirect ? NULL : threads,
      local, p);
   fprintf(stderr,
           "Apple9 dispatch failed (6): %s %s record rejected: %s; "
           "grid %u,%u,%u ptr 0x%" PRIx64 " local %u,%u,%u launch 0x%" PRIx64
           " profile local %u,%u,%u variable %u resources %u\n",
           where,
           indirect_local ? "indirect-local" : indirect ? "indirect" : "direct",
           reason ? reason : "(no check failed)",
           threads[0], threads[1], threads[2], indirect ? grid.ptr : 0,
           local[0], local[1], local[2], launch,
           p->local_size[0], p->local_size[1], p->local_size[2],
           p->variable_local_size, p->resource_binding_count);
}

void
hk_apple9_dispatch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                   struct agx_shader_body *body,
                   const struct agx_apple9_compute_profile *p,
                   struct agx_grid grid, uint64_t root,
                   const struct hk_shader *shader,
                   const struct hk_apple9_prerast_bindings *prerast)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   /* A command buffer in error is never submitted; emitting more records
    * only adds misleading failures after the first one. */
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   if (!body) {
      fprintf(stderr, "Apple9 dispatch failed (1)\n");
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   /* Software pre-rasterization stages read draw state from the graphics
    * block, which extends the compute one: the vertex-stage line stipple
    * start reads the viewport table. */
   uint64_t textures = 0, samplers = 0, sysvals = 0;
   if (shader &&
       !hk_apple9_upload_textures(cmd, shader, prerast != NULL, false, false,
                                  &textures, &samplers, &sysvals)) {
      fprintf(stderr, "Apple9 dispatch failed (2)\n");
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   uint64_t resources[AGX_APPLE9_COMPUTE_MAX_RESOURCES] = {0};
   for (unsigned i = 0; i < p->resource_binding_count; ++i) {
      if (p->resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_SHARED) {
         resources[i] = AGX_APPLE9_COMPUTE_SHARED_ROOT;
         continue;
      }
      unsigned binding = p->resource_binding[i];
      if (p->resource_kind[i] != AGX_APPLE9_COMPUTE_RESOURCE_UBO) {
         fprintf(stderr, "Apple9 dispatch failed (3)\n");
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      if (binding == 0) {
         resources[i] = root;
      } else if (binding == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING) {
         resources[i] = sysvals;
      } else if (prerast && binding == HK_APPLE9_DRAW_PARAMS_BINDING) {
         resources[i] = prerast->draw_params;
      } else if (prerast && binding == HK_APPLE9_VERTEX_BUFFERS_BINDING) {
         resources[i] = prerast->vertex_buffers;
      } else {
         fprintf(stderr, "Apple9 dispatch failed (4)\n");
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   }

   struct agx_apple9_compute_geometry geometry = {
      .mode = agx_is_indirect(grid) ? AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT
                                   : AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT,
   };
   for (unsigned d = 0; d < 3; ++d) {
      geometry.local[d] = p->local_size[d];
      if (!agx_is_indirect(grid)) {
         if (!grid.count[d])
            return;
         geometry.threads[d] = grid.count[d];
      }
   }
   if (agx_is_indirect(grid))
      geometry.group_counts = grid.ptr;

   uint64_t launch;
   if (!root || !agx_apple9_prepare_compute_dispatch_alloc_tmpl(
          &dev->dev, cmd, hk_apple9_alloc, body, p, resources,
          p->resource_binding_count, textures, samplers, &geometry,
          p->preamble_size ? body->addr + p->preamble_offset : 0,
          shader && shader->apple9_body == body ? shader->apple9_launch : NULL, &launch)) {
      fprintf(stderr, "Apple9 dispatch failed (5)\n");
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   /* Indirect-local grids hold thread counts and local sizes, and launch
    * partial workgroups like a direct dispatch.
    */
   bool indirect_local = grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL;
   unsigned record_size =
      indirect_local ? AGX_APPLE9_COMPUTE_INDIRECT_LOCAL_CDM_RECORD_SIZE
      : agx_is_indirect(grid) ? AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE
                             : AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE;
   /* Driver-internal dispatches (helpers and pre-rasterization stages) end
    * with a barrier for whatever runs next in the stream; API dispatches
    * leave it to the next dependency barrier. The compute-only barrier
    * suffices for compute after compute. After it, an indirect dispatch
    * still flushes first, since its grid is fetched by the CDM itself.
    */
   /* Batched tessellated draws record into a lane instead, without a
    * trailing barrier: hk_tess_batch_flush adds one per lane. Anything else
    * entering the pre-gfx stream first emits the pending lanes, keeping
    * stream order for everything outside the batch.
    */
   if (cmd->tess_batch.capture) {
      uint8_t rec[AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE];
      unsigned size = record_size - AGX_CDM_BARRIER_LENGTH;
      bool ok = agx_is_indirect(grid)
         ? agx_apple9_emit_indirect_dispatch(rec, launch, grid.ptr,
                                            geometry.local, indirect_local, p,
                                            false)
         : agx_apple9_emit_direct_dispatch(rec, launch, geometry.threads,
                                          geometry.local, p, false);
      if (!ok) {
         hk_apple9_report_dispatch_reject("batched", grid, indirect_local,
                                          launch, geometry.threads,
                                          geometry.local, p);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      util_dynarray_append(cmd->tess_batch.capture, (uint8_t)size);
      util_dynarray_append_array(cmd->tess_batch.capture, uint8_t, rec, size);
      return;
   }
   if (cs == cmd->current_cs.pre_gfx && cmd->tess_batch.draws)
      hk_tess_batch_flush(cmd);

   bool barrier = !cmd->api_dispatch;
   bool flush_before =
      cs->cdm_dirty &&
      (cs->cdm_compute_barrier ? agx_is_indirect(grid) : barrier);
   if (!barrier)
      record_size -= AGX_CDM_BARRIER_LENGTH;

   /* Reserve the emitted record and any preceding helper flush together.
    * The space check also adds stream-link and overread padding.
    */
   hk_ensure_cs_has_space(cmd, cs, record_size +
                         (flush_before ? AGX_CDM_BARRIER_LENGTH : 0));
   if (flush_before)
      hk_cdm_cache_flush(dev, cs);
   if (agx_is_indirect(grid))
      hk_cs_note_indirect(cs);
   bool ok = agx_is_indirect(grid)
      ? agx_apple9_emit_indirect_dispatch(cs->current, launch, grid.ptr,
                                         geometry.local, indirect_local, p, barrier)
      : agx_apple9_emit_direct_dispatch(cs->current, launch, geometry.threads,
                                       geometry.local, p, barrier);
   if (!ok) {
      hk_apple9_report_dispatch_reject("stream", grid, indirect_local, launch,
                                       geometry.threads, geometry.local, p);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (barrier)
      agx_apple9_set_compute_barrier(cs->current, record_size);
   cs->current += record_size;
   cs->cdm_dirty = true;
   cs->cdm_compute_barrier = barrier;
   cs->stats.flushes += barrier;
   cs->stats.cmds++;
   cs->stats.calls++;
   if (!cs->pass_count++) {
      cs->pass_code = body->addr;
      for (unsigned i = 0; i < 3; ++i)
         cs->pass_grid[i] = agx_is_indirect(grid) ? 0 : grid.count[i];
   }
}

static void
dispatch(struct hk_cmd_buffer *cmd, struct agx_grid grid)
{
   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   if (agx_is_shader_empty(&s->b))
      return;
   if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G) {
      if (!agx_is_indirect(grid) &&
          (!grid.count[0] || !grid.count[1] || !grid.count[2]))
         return;
      /* An API dispatch may run ahead of a render kept open for merging
       * when nothing orders it after the render. Indirect grids,
       * statistics and predication stay in place. */
      const VkPipelineStageFlags2 saved_move = cmd->merge_move_stage;
      if (!cmd->in_meta)
         cmd->merge_move_stage =
            agx_is_indirect(grid) || hk_cond_render_applies(cmd) ||
                  hk_stat_enabled(hk_pipeline_stat_addr(
                     cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT))
               ? 0 : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true);
      cmd->merge_move_stage = saved_move;
      if (!cs)
         return;
      uint64_t stat = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT);
      if (hk_stat_enabled(stat)) {
         libagx_increment_cs_invocations(
            cmd, agx_1d(1), AGX_BARRIER_ALL,
            cmd->state.cs.descriptors.root.cs.group_count_addr, stat,
            s->info.apple9.local_size[0] * s->info.apple9.local_size[1] *
               s->info.apple9.local_size[2]);
      }
      if (hk_cond_render_applies(cmd)) {
         /* Predicated dispatches read zero group counts when skipped. */
         struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
         desc->root.cs.group_count_addr =
            hk_predicate_indirect(cmd, desc->root.cs.group_count_addr, 3,
                                  false);
         if (!desc->root.cs.group_count_addr)
            return;
         grid = agx_grid_indirect(desc->root.cs.group_count_addr);
         cs = hk_cmd_buffer_get_cs(cmd, true);
         if (!cs)
            return;
      }
      if (!agx_is_indirect(grid)) {
         for (unsigned d = 0; d < 3; ++d)
            grid.count[d] *= s->info.apple9.local_size[d];
      }
      uint64_t root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
      cmd->api_dispatch = !cmd->in_meta;
      hk_apple9_dispatch(cmd, cs, s->apple9_body, &s->info.apple9, grid, root, s, NULL);
      cmd->api_dispatch = false;
      static int split = -1;
      if (split < 0)
         split = getenv("HK_GPU_PASS_SPLIT") != NULL;
      if (split && !cmd->in_meta) {
         hk_cs_end_hint(cmd, HK_CS_END_PASS_SPLIT);
         hk_cmd_buffer_end_compute(cmd);
      }
      return;
   }
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   struct agx_workgroup local_size =
      agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                    s->b.info.workgroup_size[2]);

   uint64_t stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT);

   if (hk_stat_enabled(stat)) {
      perf_debug(cmd, "CS invocation statistic");
      uint64_t grid = cmd->state.cs.descriptors.root.cs.group_count_addr;

      libagx_increment_cs_invocations(cmd, agx_1d(1), AGX_BARRIER_ALL, grid,
                                      stat, agx_workgroup_threads(local_size));
   }

   if (!agx_is_indirect(grid)) {
      grid.count[0] *= local_size.x;
      grid.count[1] *= local_size.y;
      grid.count[2] *= local_size.z;
   }

   cmd->api_dispatch = !cmd->in_meta;
   hk_dispatch_with_local_size(cmd, cs, s, grid, local_size);
   cmd->api_dispatch = false;
   cs->stats.calls++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                   uint32_t baseGroupY, uint32_t baseGroupZ,
                   uint32_t groupCountX, uint32_t groupCountY,
                   uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_cmd_flush_empty_clears(cmd);
   if (!cmd->in_meta)
      hk_feedback_invalidate(cmd);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = baseGroupX;
   desc->root.cs.base_group[1] = baseGroupY;
   desc->root.cs.base_group[2] = baseGroupZ;

   /* We don't want to key the shader to whether we're indirectly dispatching,
    * so treat everything as indirect.
    */
   VkDispatchIndirectCommand group_count = {
      .x = groupCountX,
      .y = groupCountY,
      .z = groupCountZ,
   };

   if (!desc->group_count_last_valid ||
       memcmp(desc->group_count_last, &group_count, sizeof(group_count))) {
      desc->group_count_last_addr =
         hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);
      memcpy(desc->group_count_last, &group_count, sizeof(group_count));
      desc->group_count_last_valid = desc->group_count_last_addr != 0;
   }
   desc->root.cs.group_count_addr = desc->group_count_last_addr;

   dispatch(cmd, agx_3d(groupCountX, groupCountY, groupCountZ));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_cmd_flush_empty_clears(cmd);
   if (!cmd->in_meta)
      hk_feedback_invalidate(cmd);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;

   uint64_t dispatch_addr = hk_buffer_address(buffer, offset, true);
   assert(dispatch_addr != 0);

   desc->root.cs.group_count_addr = dispatch_addr;
   dispatch(cmd, agx_grid_indirect(dispatch_addr));
}

DEBUG_GET_ONCE_BOOL_OPTION(tess_fold, "HK_APPLE9_TESS_FOLD", true)

/* Fold the libagx passes of batched tessellated draws (see tess_batch in
 * struct hk_cmd_buffer). HK_APPLE9_TESS_FOLD=0 opts out. Air, TES D64 per
 * render: 3.85 -> 3.51 us/draw emulated, 11.91 -> 11.66 native.
 */
bool
hk_tess_fold_enabled(void)
{
   return debug_get_option_tess_fold();
}

/* Record a libagx program's dispatch into a batch lane. Called while
 * flushing, so it bypasses hk_dispatch_precomp's stream selection.
 */
static void
hk_tess_fold_dispatch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                      struct util_dynarray *lane, enum libagx_program idx,
                      struct agx_grid grid, const void *data, size_t size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_precompiled_shader *prog = agx_get_precompiled(&dev->bg_eot, idx);
   uint64_t args = hk_pool_upload(cmd, data, size, 8);
   if (!prog || !prog->apple9 || !args) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   cmd->tess_batch.capture = lane;
   hk_apple9_dispatch(cmd, cs, &prog->bo->apple9_body, &prog->apple9_profile,
                      grid, args, NULL, NULL);
   cmd->tess_batch.capture = NULL;
}

/* Turn the folded draws into lane records: per primitive one count dispatch
 * (lane 2) and one emit dispatch (lane 4), and one prefix sum of every draw
 * (lane 3). The draws share one table, grouped by primitive.
 */
static void
hk_tess_fold_flush(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   unsigned n = util_dynarray_num_elements(&cmd->tess_batch.fold,
                                           struct hk_tess_fold);
   if (!n)
      return;

   struct agx_ptr table =
      hk_pool_alloc(cmd, n * sizeof(struct libagx_tess_fold_draw), 16);
   if (!table.cpu) {
      /* The folded draws' indirect draws read counts no pass will write, so
       * the command buffer must not be submitted.
       */
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      util_dynarray_clear(&cmd->tess_batch.fold);
      return;
   }

   static const enum tess_primitive_mode prims[] = {
      TESS_PRIMITIVE_TRIANGLES,
      TESS_PRIMITIVE_QUADS,
      TESS_PRIMITIVE_ISOLINES,
   };
   struct libagx_tess_fold_draw *draws = table.cpu;
   unsigned start[ARRAY_SIZE(prims)], count[ARRAY_SIZE(prims)];
   unsigned groups[ARRAY_SIZE(prims)];
   unsigned d = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(prims); ++i) {
      start[i] = d;
      groups[i] = 0;
      util_dynarray_foreach(&cmd->tess_batch.fold, struct hk_tess_fold, f) {
         if (f->prim != prims[i])
            continue;
         draws[d++] = (struct libagx_tess_fold_draw){
            .p = f->params,
            .first_group = groups[i],
            .nr_patches = f->nr_patches,
         };
         groups[i] += DIV_ROUND_UP(f->nr_patches, LIBAGX_TESS_FOLD_GROUP);
      }
      count[i] = d - start[i];
   }
   assert(d == n);
   util_dynarray_clear(&cmd->tess_batch.fold);

   for (unsigned stage = 0; stage < 2; ++stage) {
      enum poly_tess_mode mode =
         stage ? POLY_TESS_MODE_WITH_COUNTS : POLY_TESS_MODE_COUNT;
      struct util_dynarray *lane = &cmd->tess_batch.lane[stage ? 4 : 2];

      for (unsigned i = 0; i < ARRAY_SIZE(prims); ++i) {
         if (!groups[i])
            continue;

         uint64_t addr =
            table.gpu + start[i] * sizeof(struct libagx_tess_fold_draw);
         struct agx_grid grid = agx_1d(groups[i] * LIBAGX_TESS_FOLD_GROUP);
         /* The three kernels take the same arguments. */
         struct libagx_tess_tri_fold_args args = {addr, count[i]};
         enum libagx_program idx =
            prims[i] == TESS_PRIMITIVE_TRIANGLES ? LIBAGX_TESS_TRI_FOLD(mode)
            : prims[i] == TESS_PRIMITIVE_QUADS   ? LIBAGX_TESS_QUAD_FOLD(mode)
                                                 : LIBAGX_TESS_ISOLINE_FOLD(mode);
         hk_tess_fold_dispatch(cmd, cs, lane, idx, grid, &args, sizeof(args));
      }
   }

   struct libagx_prefix_sum_tess_fold_args args = {table.gpu};
   hk_tess_fold_dispatch(cmd, cs, &cmd->tess_batch.lane[3],
                         LIBAGX_PREFIX_SUM_TESS_FOLD, agx_1d(1024 * n), &args,
                         sizeof(args));
}

/* Emit the batched tessellation lanes into the pre-gfx stream: each lane's
 * records back to back, then the compute-only barrier, so stage N of every
 * batched draw finishes before stage N+1 of any starts. The stream stays
 * dirty so its end still gets the full flush before the render Work.
 */
void
hk_tess_batch_flush(struct hk_cmd_buffer *cmd)
{
   if (!cmd->tess_batch.draws)
      return;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_cs *cs = cmd->current_cs.pre_gfx;
   cmd->tess_batch.draws = 0;
   if (!cs) {
      for (unsigned i = 0; i < HK_TESS_BATCH_LANES; ++i)
         util_dynarray_clear(&cmd->tess_batch.lane[i]);
      util_dynarray_clear(&cmd->tess_batch.fold);
      return;
   }

   hk_tess_fold_flush(cmd, cs);

   for (unsigned l = 0; l < HK_TESS_BATCH_LANES; ++l) {
      struct util_dynarray *lane = &cmd->tess_batch.lane[l];
      if (!lane->size)
         continue;

      uint8_t *data = lane->data, *end = data + lane->size;
      bool first = true;
      while (data < end) {
         unsigned size = *data++;
         hk_ensure_cs_has_space(cmd, cs, size + 2 * AGX_CDM_BARRIER_LENGTH);
         /* Whatever came before the batch must be complete. */
         if (first && cs->cdm_dirty && !cs->cdm_compute_barrier)
            hk_cdm_cache_flush(dev, cs);
         /* Batched records may fetch indirect grids. */
         hk_cs_note_indirect(cs);
         first = false;
         memcpy(cs->current, data, size);
         cs->current += size;
         cs->stats.cmds++;
         cs->stats.calls++;
         data += size;
      }

      hk_cdm_compute_barrier(dev, cs);
      util_dynarray_clear(lane);
   }
}
