/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "libagx/query.h"
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

bool
hk_apple9_upload_textures(struct hk_cmd_buffer *cmd,
                          const struct hk_shader *shader, const float *blend,
                          bool first_fan, uint64_t *textures,
                          uint64_t *samplers, uint64_t *sysvals)
{
   /* Textures and samplers are addressed through their descriptors, see
    * hk_apple9_texture_bindings: the stage's table pointers name the USC
    * aperture, and only per-draw state goes in the sysval block. */
   if (shader->b.info.apple9_texture_mask || shader->b.info.apple9_sampler_mask)
      return false;
   _Alignas(64) uint8_t info[AGX_APPLE9_RUNTIME_SYSVAL_SIZE] = {0};
   if (blend) {
      memcpy(info, blend, 4 * sizeof(float));
      uint32_t enables = cmd->vk.dynamic_graphics_state.cb.color_write_enables;
      memcpy((uint8_t *)info + AGX_APPLE9_COLOR_WRITE_ENABLE_OFFSET,
             &enables, sizeof(enables));
      memcpy((uint8_t *)info + AGX_APPLE9_SAMPLE_POSITIONS_OFFSET,
             &cmd->state.gfx.apple9_sample_positions, sizeof(uint32_t));
   }
   uint32_t fan = first_fan;
   memcpy((uint8_t *)info + AGX_APPLE9_FIRST_FAN_OFFSET, &fan, sizeof(fan));
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint32_t fetch = (dev->apple9_fetch_sampler->va->addr -
                     dev->dev.shader_base) / AGX_APPLE9_SAMPLER_TABLE_STRIDE;
   memcpy((uint8_t *)info + AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET,
          &fetch, sizeof(fetch));
   *sysvals = hk_pool_upload_cached(cmd, info, sizeof(info), 64, HK_UPLOAD_RUNTIME);
   *textures = *samplers = dev->dev.shader_base;
   /* A stage without sampler handles names the compiler-private fetch
    * sampler as immediate slot 0 of its sampler table. */
   if (shader->b.info.apple9_uses_texel_fetch)
      *samplers = dev->apple9_fetch_sampler->va->addr;
   return *sysvals != 0;
}

void
hk_apple9_dispatch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                   struct agx_bo *body,
                   const struct agx_apple9_compute_profile *p,
                   struct agx_grid grid, uint64_t root,
                   const struct hk_shader *shader,
                   const struct hk_apple9_prerast_bindings *prerast)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   if (!body) {
      fprintf(stderr, "Apple9 dispatch failed (1)\n");
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   uint64_t textures = 0, samplers = 0, sysvals = 0;
   if (shader &&
       !hk_apple9_upload_textures(cmd, shader, NULL, false, &textures,
                                  &samplers, &sysvals)) {
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
          p->preamble_size ? body->va->addr + p->preamble_offset : 0,
          shader && shader->bo == body ? shader->apple9_launch : NULL, &launch)) {
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
         fprintf(stderr, "Apple9 dispatch failed (6)\n");
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
   bool ok = agx_is_indirect(grid)
      ? agx_apple9_emit_indirect_dispatch(cs->current, launch, grid.ptr,
                                         geometry.local, indirect_local, p, barrier)
      : agx_apple9_emit_direct_dispatch(cs->current, launch, geometry.threads,
                                       geometry.local, p, barrier);
   if (!ok) {
      fprintf(stderr, "Apple9 dispatch failed (6)\n");
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
      struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true);
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
      hk_apple9_dispatch(cmd, cs, s->bo, &s->info.apple9, grid, root, s, NULL);
      cmd->api_dispatch = false;
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
      return;
   }

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
         first = false;
         memcpy(cs->current, data, size);
         cs->current += size;
         cs->stats.cmds++;
         cs->stats.calls++;
         data += size;
      }

      uint32_t *barrier = (uint32_t *)cs->current;
      cs->current = agx_cdm_barrier(cs->current, dev->dev.chip);
      *barrier = 0x60000160;
      cs->stats.flushes++;
      cs->cdm_dirty = true;
      cs->cdm_compute_barrier = true;
      util_dynarray_clear(lane);
   }
}
