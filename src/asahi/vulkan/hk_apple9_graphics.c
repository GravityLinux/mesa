/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "hk_apple9_graphics.h"
#include "hk_apple9_msaa.h"
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
#include "agx_helpers.h"
#include "asahi/layout/layout.h"
#include "hk_cmd_buffer.h"
#include "hk_apple9.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "agx_compiler.h"
#include "nir_builder.h"
#include "nir_format_convert.h"
#include "libagx_shaders.h"
#include "vk_blend.h"
#include "vk_util.h"

/* Tilebuffer representation of a supported color attachment. All 8-bit
 * RGBA variants keep the encoded RGBA8 layout; sRGB is encoded by the
 * shaders. Float and integer attachments keep their native channels. Both
 * 10-bit orders use the RGB10A2 tile; the view's PBE swizzle stores the
 * memory order (a BGR10A2 tile came out R/B-swapped on M4).
 */
static enum pipe_format
hk_apple9_tile_color_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return PIPE_FORMAT_R16G16B16A16_FLOAT;
   case VK_FORMAT_R8_UNORM:
      return PIPE_FORMAT_R8_UNORM;
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return PIPE_FORMAT_R32G32B32A32_FLOAT;
   case VK_FORMAT_R32G32B32A32_UINT:
      return PIPE_FORMAT_R32G32B32A32_UINT;
   case VK_FORMAT_R32G32B32A32_SINT:
      return PIPE_FORMAT_R32G32B32A32_SINT;
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      return PIPE_FORMAT_R10G10B10A2_UNORM;
   default:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   }
}

bool
hk_apple9_blend_state(const struct vk_color_blend_state *cb,
                      VkFormat format, unsigned samples,
                      struct agx_apple9_blend *out)
{
   if (!cb || cb->attachment_count != 1 || cb->logic_op_enable)
      return false;
   const struct vk_color_blend_attachment_state *rt = &cb->attachments[0];
   memset(out, 0, sizeof(*out));
   out->format = format == VK_FORMAT_R8G8B8A8_SRGB ||
                 format == VK_FORMAT_B8G8R8A8_SRGB ||
                 format == VK_FORMAT_A8B8G8R8_SRGB_PACK32
      ? PIPE_FORMAT_R8G8B8A8_SRGB : hk_apple9_tile_color_format(format);
   if (samples != 1 && samples != 2 && samples != 4)
      return false;
   out->samples = samples;
   out->colormask = (cb->color_write_enables & 1) ? rt->write_mask : 0;
   out->rgb_src = out->alpha_src = PIPE_BLENDFACTOR_ONE;
   out->rgb_dst = out->alpha_dst = PIPE_BLENDFACTOR_ZERO;
   out->rgb_func = out->alpha_func = PIPE_BLEND_ADD;
   if (!rt->blend_enable)
      return true;

   /* Constant factors read the fragment sysval words 0..3, uploaded per draw.
    * Dual source requires an additional fragment output and stays rejected.
    */
   const VkBlendFactor factors[] = {
      rt->src_color_blend_factor, rt->dst_color_blend_factor,
      rt->src_alpha_blend_factor, rt->dst_alpha_blend_factor,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(factors); ++i) {
      if ((unsigned)factors[i] > VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA &&
          factors[i] != VK_BLEND_FACTOR_SRC_ALPHA_SATURATE)
         return false;
   }
   if ((unsigned)rt->color_blend_op > VK_BLEND_OP_MAX ||
       (unsigned)rt->alpha_blend_op > VK_BLEND_OP_MAX)
      return false;
   out->rgb_src = vk_blend_factor_to_pipe(factors[0]);
   out->rgb_dst = vk_blend_factor_to_pipe(factors[1]);
   out->alpha_src = vk_blend_factor_to_pipe(factors[2]);
   out->alpha_dst = vk_blend_factor_to_pipe(factors[3]);
   out->rgb_func = vk_blend_op_to_pipe(rt->color_blend_op);
   out->alpha_func = vk_blend_op_to_pipe(rt->alpha_blend_op);
   return true;
}

VkResult
hk_apple9_graphics_reject(struct hk_cmd_buffer *cmd, const char *reason)
{
   if (getenv("AGX_APPLE9_TRACE_REJECT"))
      fprintf(stderr, "APPLE9_RENDER_REJECT reason=%s\n", reason);
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   /* Keep allocations on control_streams for reset, but never run the stock
    * empty-render optimizer on a rejected native job.
    */
   cmd->current_cs.gfx = NULL;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

static struct hk_apple9_render_stage
hk_apple9_stage(struct hk_shader *shader, bool vertex)
{
   const struct agx_shader_info *info = &shader->b.info;
   struct hk_apple9_render_stage stage = {
      .bo = shader->bo,
      .binary = shader->code_ptr,
      .binary_size = shader->code_size,
      .preamble_offset = info->apple9_preamble_offset,
      .preamble_size = info->apple9_preamble_size,
      .scratch_size = info->scratch_size,
      .publication_count = info->apple9_publication_count,
      .publication_count_valid = true,
      .resource_count = info->apple9_resource_count,
      .resource_ssbo_mask = info->apple9_resource_ssbo_mask,
      .resource_write_mask = info->apple9_resource_write_mask,
      .texture_mask = info->apple9_texture_mask,
      .sampler_mask = info->apple9_sampler_mask,
      .uses_texel_fetch = info->apple9_uses_texel_fetch,
      .reads_tile = info->apple9_reads_tile,
      .uses_discard = !vertex && info->apple9_uses_discard,
      .disable_tri_merging = info->disable_tri_merging,
      .position_components = vertex ? 4 : 0,
      .varying_components = info->apple9_varyings.count,
      .varyings = info->apple9_varyings,
      .apple9_linear_mask = info->apple9_linear_mask,
      .apple9_flat_mask = info->apple9_flat_mask,
      .writes_layer_viewport = vertex && info->writes_layer_viewport,
      .render_targets = vertex ? 0 : 1,
   };
   memcpy(stage.resource_binding, info->apple9_resource_binding,
          sizeof(stage.resource_binding));
   return stage;
}

/* Gallium's asahi_reload_fs: texture fetch -> raw tile store, or native
 * block export -> PBE. The initial background instead packs the API clear.
 * Each shader owns an immutable body BO until command-buffer reset.
 */
static bool
hk_apple9_tile_shader(struct hk_cmd_buffer *cmd, unsigned operation, bool srgb,
                      enum pipe_format format, unsigned samples,
                      const VkClearColorValue *clear, const VkRect2D *clear_area,
                      uint64_t textures, uint64_t samplers,
                      struct hk_bg_eot *out)
{
   nir_shader *nir = hk_apple9_tile_nir(operation, srgb, format, samples,
                                      clear, clear_area);
   struct hk_shader shader = {0};
   const char *reason = NULL;
   const struct agx_apple9_blend blend = {.samples = samples};
   bool compiled = agx_compile_apple9_fragment_blend(nir, NULL, &blend,
                                                    &shader.b, &reason);
   ralloc_free(nir);
   if (!compiled || shader.b.info.apple9_resource_count) {
      if (getenv("AGX_APPLE9_TRACE_REJECT"))
         fprintf(stderr, "APPLE9_TILE_COMPILE_FAIL op=%u reason=%s\n",
                 operation, reason ?: "buffer-resource");
      free(shader.b.binary);
      hk_apple9_graphics_reject(cmd, "tile-compiler");
      return false;
   }
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   shader.code_ptr = shader.b.binary;
   shader.code_size = shader.b.info.binary_size;
   shader.bo = agx_bo_create(&dev->dev, shader.code_size, 0,
      AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK, "Apple9 tile body");
   if (!shader.bo) {
      free(shader.b.binary);
      hk_apple9_graphics_reject(cmd, "tile-body-allocation");
      return false;
   }
   memcpy(agx_bo_map(shader.bo), shader.code_ptr, shader.code_size);
   agx_bo_note_cpu_write(shader.bo, 0, shader.code_size);
   util_dynarray_append(&cmd->large_bos, shader.bo);
   uint32_t entry = agx_apple9_shader_entry(&dev->dev, shader.bo, false);
   struct agx_ptr state = agx_pool_alloc_aligned(
      &cmd->apple9_compute, 0x100 + AGX_APPLE9_GRAPHICS_LAUNCH_SIZE, 64);
   memset(state.cpu, 0, 0x100);
   uint64_t roots[6] = {(operation || clear_area) ? textures : 0,
                       (operation == 1 || clear_area) ? samplers : 0};
   memcpy(state.cpu, roots, sizeof(roots));
   struct agx_apple9_launch_parameters params = {
      .shader_base = dev->dev.shader_base,
      .entry_offset = entry,
      .resource_table = state.gpu,
      .publication_count = shader.b.info.apple9_publication_count,
      .frame_extent_a = shader.b.info.scratch_size,
      .frame_extent_b = shader.b.info.scratch_size,
      .tile_bytes = 4 * agx_apple9_color_words(format),
      .samples = samples,
      .preamble_address = shader.b.info.apple9_preamble_size
         ? shader.bo->va->addr + shader.b.info.apple9_preamble_offset : 0,
      .launch_address = state.gpu + 0x100,
   };
   bool launched = entry && agx_apple9_launch_build(
      (uint8_t *)state.cpu + 0x100, AGX_APPLE9_GRAPHICS_LAUNCH_SIZE,
      AGX_APPLE9_LAUNCH_FRAGMENT, &params);
   free(shader.b.binary);
   if (!launched) {
      hk_apple9_graphics_reject(cmd, "tile-launch");
      return false;
   }
   *out = (struct hk_bg_eot){.usc = agx_usc_addr(&dev->dev, state.gpu + 0x100)};
   return true;
}

/* Validate one single-level, single-layer, tiled Z or S plane covering the
 * whole render area, as addressed by the ZLS base/stride registers.
 */
static bool
hk_apple9_zs_attachment(const struct hk_image_view *view,
                        const struct hk_image *image, unsigned plane,
                        const VkRenderingAttachmentInfo *att, unsigned width,
                        unsigned height, unsigned samples, uint64_t *base)
{
   const struct ail_layout *l = &image->planes[plane].layout;
   if ((view->vk.view_type != VK_IMAGE_VIEW_TYPE_2D &&
        view->vk.view_type != VK_IMAGE_VIEW_TYPE_2D_ARRAY) ||
       view->vk.base_mip_level || view->vk.base_array_layer ||
       view->vk.level_count != 1 || view->vk.layer_count != 1 ||
       att->resolveMode != VK_RESOLVE_MODE_NONE ||
       (att->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
        att->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD &&
        att->loadOp != VK_ATTACHMENT_LOAD_OP_DONT_CARE) ||
       (att->storeOp != VK_ATTACHMENT_STORE_OP_STORE &&
        att->storeOp != VK_ATTACHMENT_STORE_OP_DONT_CARE) ||
       image->vk.samples != samples ||
       image->vk.mip_levels != 1 || image->vk.array_layers != 1 ||
       image->vk.image_type != VK_IMAGE_TYPE_2D ||
       image->vk.extent.width != width || image->vk.extent.height != height ||
       image->planes[plane].sparse_map ||
       (image->vk.create_flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                  VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT)) ||
       (l->tiling != AIL_TILING_GPU && l->tiling != AIL_TILING_TWIDDLED) ||
       l->layer_stride_B < AIL_PAGESIZE ||
       !util_is_aligned(l->layer_stride_B, AIL_PAGESIZE) ||
       l->layer_stride_B / AIL_PAGESIZE > (1u << 18) ||
       l->level_offsets_B[0] ||
       (l->compressed &&
        (!l->compression_layer_stride_B ||
         !util_is_aligned(l->compression_layer_stride_B, AIL_CACHELINE) ||
         l->compression_layer_stride_B / AIL_CACHELINE > (1u << 18))))
      return false;
   *base = hk_image_base_address(image, plane);
   return *base && util_is_aligned(*base, AIL_PAGESIZE);
}

void
hk_apple9_begin_rendering(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   if (vk_command_buffer_get_record_result(&cmd->vk) != VK_SUCCESS)
      return;
   if (cmd->current_cs.gfx) {
      hk_apple9_graphics_reject(cmd, "nested-render");
      return;
   }
   /* Render-pass emulation sets local-read access control even for passes
    * without input attachments. Input-attachment descriptors still reject.
    */
   VkRenderingFlags allowed_flags =
      VK_RENDERING_LOCAL_READ_CONCURRENT_ACCESS_CONTROL_BIT_KHR |
      VK_RENDERING_CONTENTS_INLINE_BIT_KHR;
   if ((info->flags & ~allowed_flags) || info->viewMask || info->layerCount != 1 ||
       info->colorAttachmentCount != 1) {
      if (getenv("AGX_APPLE9_TRACE_REJECT"))
         fprintf(stderr, "APPLE9_RENDER_INFO flags=0x%x viewMask=%u layers=%u colors=%u depth=%u stencil=%u\n",
                 info->flags, info->viewMask, info->layerCount,
                 info->colorAttachmentCount,
                 !!(info->pDepthAttachment && info->pDepthAttachment->imageView),
                 !!(info->pStencilAttachment && info->pStencilAttachment->imageView));
      hk_apple9_graphics_reject(cmd, "attachment-count-layers-or-depth-stencil");
      return;
   }
   const VkRenderingAttachmentInfo *att = info->pColorAttachments;
   VK_FROM_HANDLE(hk_image_view, view, att->imageView);
   enum pipe_format view_format =
      view ? vk_format_to_pipe_format(view->vk.format) : PIPE_FORMAT_NONE;
   if (!view ||
       (view_format != PIPE_FORMAT_R8G8B8A8_UNORM &&
        view_format != PIPE_FORMAT_B8G8R8A8_UNORM &&
        view_format != PIPE_FORMAT_R8G8B8A8_SRGB &&
        view_format != PIPE_FORMAT_B8G8R8A8_SRGB &&
        view_format != PIPE_FORMAT_R10G10B10A2_UNORM &&
        view_format != PIPE_FORMAT_B10G10R10A2_UNORM &&
        view_format != PIPE_FORMAT_R16G16B16A16_FLOAT &&
        view_format != PIPE_FORMAT_R8_UNORM &&
        view_format != PIPE_FORMAT_R32G32B32A32_FLOAT &&
        view_format != PIPE_FORMAT_R32G32B32A32_UINT &&
        view_format != PIPE_FORMAT_R32G32B32A32_SINT) ||
       (view->vk.view_type != VK_IMAGE_VIEW_TYPE_2D &&
        view->vk.view_type != VK_IMAGE_VIEW_TYPE_2D_ARRAY) ||
       view->vk.base_mip_level || view->vk.base_array_layer ||
       view->vk.level_count != 1 || view->vk.layer_count != 1 ||
       (att->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
        att->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD &&
        att->loadOp != VK_ATTACHMENT_LOAD_OP_DONT_CARE) ||
       (att->storeOp != VK_ATTACHMENT_STORE_OP_STORE &&
        att->storeOp != VK_ATTACHMENT_STORE_OP_DONT_CARE)) {
      if (getenv("AGX_APPLE9_TRACE_REJECT") && view)
         fprintf(stderr, "APPLE9_COLOR_INFO format=%u view_type=%u level=%u+%u layer=%u+%u resolve=%u load=%u store=%u\n",
                 view->vk.format, view->vk.view_type, view->vk.base_mip_level,
                 view->vk.level_count, view->vk.base_array_layer,
                 view->vk.layer_count, att->resolveMode, att->loadOp,
                 att->storeOp);
      hk_apple9_graphics_reject(cmd, "color-view-or-load-store");
      return;
   }
   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   struct ail_layout *layout = &image->planes[0].layout;
   unsigned samples = image->vk.samples;
   if (image->plane_count != 1 ||
       (samples != 1 && samples != 2 && samples != 4) ||
       (samples > 1 && layout->tiling == AIL_TILING_LINEAR) ||
       image->vk.mip_levels != 1 || image->vk.array_layers != 1 ||
       image->vk.image_type != VK_IMAGE_TYPE_2D || layout->compressed ||
       image->planes[0].sparse_map ||
       (image->vk.create_flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                  VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT))) {
      hk_apple9_graphics_reject(cmd, "image-layout-or-msaa");
      return;
   }
   VK_FROM_HANDLE(hk_image_view, resolve, att->resolveImageView);
   enum pipe_format format = hk_apple9_tile_color_format(view->vk.format);
   struct agx_tilebuffer_layout tilebuffer;
   if (!hk_apple9_tilebuffer(format, samples, &tilebuffer)) {
      hk_apple9_graphics_reject(cmd, "msaa-tile-budget");
      return;
   }
   if (att->resolveMode != VK_RESOLVE_MODE_NONE) {
      bool integer = util_format_is_pure_integer(view_format);
      if (samples == 1 || !resolve || resolve->vk.format != view->vk.format ||
          resolve->vk.image->samples != VK_SAMPLE_COUNT_1_BIT ||
          att->resolveMode != (integer ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT
                                      : VK_RESOLVE_MODE_AVERAGE_BIT)) {
         hk_apple9_graphics_reject(cmd, "color-resolve");
         return;
      }
   }
   unsigned width = image->vk.extent.width, height = image->vk.extent.height;
   const VkRect2D *area = &info->renderArea;
   if (!width || !height || width > 16384 || height > 16384 ||
       area->offset.x < 0 || area->offset.y < 0 ||
       (unsigned)area->offset.x >= width || (unsigned)area->offset.y >= height ||
       !area->extent.width || !area->extent.height ||
       area->extent.width > width - area->offset.x ||
       area->extent.height > height - area->offset.y) {
      hk_apple9_graphics_reject(cmd, "render-area-bounds");
      return;
   }
   bool partial = area->offset.x || area->offset.y ||
                  area->extent.width != width || area->extent.height != height;
   /* As for Z/S below, a partial DONT_CARE color pass loads: pixels outside
    * the render area must survive (meta resolves and blits rely on it).
    */
   const VkAttachmentLoadOp color_load =
      partial && att->loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE
         ? VK_ATTACHMENT_LOAD_OP_LOAD : att->loadOp;
   /* ZLS fast clears cover the whole attachment. Until rectangular Z/S
    * clears are available, only reload Z/S for a partial color render.
    * Loading DONT_CARE contents is legal and preserves pixels outside it.
    */
   if (partial &&
       ((info->pDepthAttachment && info->pDepthAttachment->imageView &&
         info->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) ||
        (info->pStencilAttachment && info->pStencilAttachment->imageView &&
         info->pStencilAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR))) {
      hk_apple9_graphics_reject(cmd, "partial-depth-stencil-clear");
      return;
   }
   uint64_t base = hk_image_base_address(image, 0) + layout->level_offsets_B[0];
   if (!base || (base & 15) ||
       (layout->tiling != AIL_TILING_GPU &&
        layout->tiling != AIL_TILING_TWIDDLED &&
        layout->tiling != AIL_TILING_LINEAR)) {
      hk_apple9_graphics_reject(cmd, "attachment-address-or-tiling");
      return;
   }
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);
   hk_cmd_buffer_end_compute(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   memset(render, 0, sizeof(*render));
   render->area = info->renderArea;
   render->layer_count = 1;
   render->color_att_count = 1;
   render->color_att[0].iview = view;
   render->color_att[0].vk_format = view->vk.format;
   render->color_att[0].resolve_mode = att->resolveMode;
   render->color_att[0].resolve_iview =
      att->resolveMode != VK_RESOLVE_MODE_NONE ? resolve : NULL;
   const VkClearColorValue clear =
      att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR
         ? att->clearValue.color : (VkClearColorValue){0};
   render->color_att[0].clear = color_load != VK_ATTACHMENT_LOAD_OP_LOAD;
   memcpy(render->color_att[0].clear_colour, &clear, sizeof(clear));
   render->tilebuffer = tilebuffer;
   /* Native shaders use dword-packed color and eight-byte sample strides. */
   assert(render->tilebuffer.sample_size_B ==
          ALIGN_POT(4 * agx_apple9_color_words(format), 8));
   assert(render->tilebuffer.tile_size == (samples == 4 ? 32 * 16 : 32 * 32));
   render->cr.width = width;
   render->cr.height = height;
   render->cr.layers = 1;
   render->cr.isp_bgobjvals = 0x300;
   render->cr.process_empty_tiles = true;

   const VkRenderingAttachmentInfo *depth = info->pDepthAttachment;
   const VkRenderingAttachmentInfo *stencil = info->pStencilAttachment;
   if (depth && !depth->imageView)
      depth = NULL;
   if (stencil && !stencil->imageView)
      stencil = NULL;
   struct ail_layout *zl = NULL, *sl = NULL;
   if (depth) {
      VK_FROM_HANDLE(hk_image_view, zv, depth->imageView);
      struct hk_image *zi = container_of(zv->vk.image, struct hk_image, vk);
      uint64_t zbase;
      zl = &zi->planes[0].layout;
      if ((zv->vk.format != VK_FORMAT_D32_SFLOAT &&
           zv->vk.format != VK_FORMAT_D32_SFLOAT_S8_UINT) ||
          zl->format != PIPE_FORMAT_Z32_FLOAT ||
          !(depth->clearValue.depthStencil.depth >= 0.0f &&
            depth->clearValue.depthStencil.depth <= 1.0f) ||
          !hk_apple9_zs_attachment(zv, zi, 0, depth, width, height, samples, &zbase)) {
         hk_apple9_graphics_reject(cmd, "depth-view-layout-or-load-store");
         return;
      }
      render->depth_att.iview = zv;
      render->depth_att.vk_format = zv->vk.format;
      render->depth_att.clear = depth->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD;
      render->cr.depth.buffer = zbase;
      render->cr.depth.stride =
         ((zl->layer_stride_B / AIL_PAGESIZE - 1) << 14) | 1;
      if (zl->compressed) {
         render->cr.depth.meta = zbase + zl->metadata_offset_B +
                                zl->level_offsets_compressed_B[0];
         render->cr.depth.meta_stride =
            (zl->compression_layer_stride_B / AIL_CACHELINE - 1) << 14;
      }
      render->cr.isp_bgobjdepth = fui(depth->clearValue.depthStencil.depth);
      render->depth_att.clear_colour[0] = fui(depth->clearValue.depthStencil.depth);
   }
   if (stencil) {
      VK_FROM_HANDLE(hk_image_view, sv, stencil->imageView);
      struct hk_image *si = container_of(sv->vk.image, struct hk_image, vk);
      /* Stencil is always the last plane, possibly the only one. */
      unsigned plane = si->plane_count - 1;
      uint64_t sbase;
      sl = &si->planes[plane].layout;
      if ((sv->vk.format != VK_FORMAT_S8_UINT &&
           sv->vk.format != VK_FORMAT_D32_SFLOAT_S8_UINT) ||
          sl->format != PIPE_FORMAT_S8_UINT ||
          !hk_apple9_zs_attachment(sv, si, plane, stencil, width, height, samples, &sbase)) {
         hk_apple9_graphics_reject(cmd, "stencil-view-layout-or-load-store");
         return;
      }
      render->stencil_att.iview = sv;
      render->stencil_att.vk_format = sv->vk.format;
      render->stencil_att.clear = stencil->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD;
      render->cr.stencil.buffer = sbase;
      render->cr.stencil.stride =
         ((sl->layer_stride_B / AIL_PAGESIZE - 1) << 14) | 1;
      if (sl->compressed) {
         render->cr.stencil.meta = sbase + sl->metadata_offset_B +
                                  sl->level_offsets_compressed_B[0];
         render->cr.stencil.meta_stride =
            (sl->compression_layer_stride_B / AIL_CACHELINE - 1) << 14;
      }
      render->cr.isp_bgobjvals |= stencil->clearValue.depthStencil.stencil & 0xff;
      render->stencil_att.clear_colour[1] = stencil->clearValue.depthStencil.stencil;
   }
   if (zl || sl) {
      /* Like the Apple8 path, always store Z/S: a pass may be split into
       * several control streams, and every later stream reloads both.
       */
      render->cr.zls_width = width;
      render->cr.zls_height = height;
      struct agx_zls zls = {
         .z_store = zl != NULL,
         .s_store = sl != NULL,
         .z_load = zl && (partial || depth->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD),
         .s_load = sl && (partial || stencil->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD),
      };
      agx_pack_zls_control(&render->cr.zls_control, zl, sl, &zls);
      zls.z_load = zl != NULL;
      zls.s_load = sl != NULL;
      agx_pack_zls_control(&render->cr.zls_control_partial, zl, sl, &zls);
   }

   /* First 64 bits stay Apple8-compatible, but the address moves from bit 66
    * to bit 64. Reload raw encoded bytes: fragment blending performs any
    * sRGB decode, and the tilebuffer must retain the encoded representation.
    */
   uint8_t descriptors[64] = {0};
   uint64_t texture_address;
   /* The reload fetches with 2D coordinates. Use the background descriptor,
    * which is 2D (or 2D MS) whatever the view type: vk_meta blits and
    * resolves render through 2D_ARRAY views, whose sampled descriptor made
    * every reload read texel (0,0).
    */
   if (samples == 1) {
      memcpy(descriptors, &view->planes[0].background, 8);
      uint32_t tex_header;
      memcpy(&tex_header, descriptors + 4, 4);
      tex_header &= 0x00ffffff;
      memcpy(descriptors + 4, &tex_header, 4);
      texture_address = base >> 4;
   } else {
      if (!hk_apple9_pack_texture(descriptors, &view->planes[0].background)) {
         hk_apple9_graphics_reject(cmd, "background-descriptor");
         return;
      }
      memcpy(&texture_address, descriptors + 8, 8);
      /* Native bit 56 selects 4x (2x is zero with an MS dimension).
       * The translator sets address-word bit 44 to decode sRGB; reload
       * must instead preserve the encoded tile bytes, so clear that bit. */
      texture_address &= ~(UINT64_C(1) << 44);
   }
   uint64_t pbe_address = base >> 4;
   if (layout->tiling == AIL_TILING_LINEAR) {
      unsigned stride = ail_get_linear_stride_B(layout, 0);
      if ((stride & 15) || stride < 16 || stride > AGX_APPLE9_MAX_LINEAR_STRIDE) {
         hk_apple9_graphics_reject(cmd, "linear-stride");
         return;
      }
      if (samples == 1)
         texture_address |= ((uint64_t)(stride >> 4) - 1) << 46;
      pbe_address |= ((uint64_t)(stride >> 4) - 1) << 44;
   } else if (samples == 1 && layout->page_aligned_layers) {
      texture_address |= UINT64_C(1) << 62;
   }
   memcpy(descriptors + 8, &texture_address, 8);
   memcpy(descriptors + 32, &view->planes[0].eot, 8);
   uint32_t pbe_header[2];
   memcpy(pbe_header, descriptors + 32, 8);
   pbe_header[0] = (pbe_header[0] & 0x00ffffff) | ((width - 1) << 24);
   pbe_header[1] = ((width - 1) >> 8) | ((height - 1) << 6) |
                   (pbe_header[1] & 0x01000000);
   memcpy(descriptors + 32, pbe_header, 8);
   memcpy(descriptors + 40, &pbe_address, 8);
   uint64_t desc = hk_pool_upload(cmd, descriptors, sizeof(descriptors), 64);
   uint8_t sampler[32] = {0};
   agx_apple9_pack_sampler(sampler, false, false, 1, 0, 14,
      PIPE_TEX_WRAP_CLAMP_TO_EDGE, PIPE_TEX_WRAP_CLAMP_TO_EDGE, 1);
   uint64_t samp = hk_pool_upload(cmd, sampler, sizeof(sampler), 64);
   if (!desc || !samp ||
       !hk_apple9_tile_shader(cmd,
                              color_load == VK_ATTACHMENT_LOAD_OP_LOAD ? 1 : 0,
                              util_format_is_srgb(view_format), format, samples,
                              &clear,
                              partial && color_load != VK_ATTACHMENT_LOAD_OP_LOAD
                                 ? area : NULL, desc, samp,
                              &render->cr.bg.main) ||
       !hk_apple9_tile_shader(cmd, 1, false, format, samples, NULL, NULL, desc, samp,
                              &render->cr.bg.partial) ||
       !hk_apple9_tile_shader(cmd, 2, false, format, samples, NULL, NULL, desc + 32, 0,
                              &render->cr.eot.main)) {
      hk_apple9_graphics_reject(cmd, "tile-setup");
      return;
   }
   render->cr.eot.partial = render->cr.eot.main;
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   if (!cs) {
      hk_apple9_graphics_reject(cmd, "render-stream-allocation");
      return;
   }
   cs->apple9_attachment = (struct drm_asahi_attachment){
      .pointer = base, .size = layout->size_B - layout->level_offsets_B[0],
   };
   /* The initial stream clears Z; subsequent partial renders must reload it. */
   cs->cr.zls_control = render->cr.zls_control;
   /* Retain a native terminator even if recording later fails. */
   const uint32_t stop = 0xc0000000;
   memcpy(cs->current, &stop, sizeof(stop));
   /* The background/EOT is a command even without draws. In particular,
    * hk_optimize_empty_vdm clears whole images, not render->area.
    */
   cs->stats.cmds++;
   /* Queue submission requires valid scissor/bias storage even for a
    * background-only job. Later draws append their own scissor records.
    */
   struct agx_scissor_packed scissor;
   agx_pack(&scissor, SCISSOR, cfg) {
      cfg.min_x = area->offset.x;
      cfg.min_y = area->offset.y;
      cfg.max_x = area->offset.x + area->extent.width;
      cfg.max_y = area->offset.y + area->extent.height;
      cfg.min_z = 0.0f;
      cfg.max_z = 1.0f;
   }
   util_dynarray_append(&cs->scissor, scissor);
   struct agx_depth_bias_packed bias = {0};
   util_dynarray_append(&cs->depth_bias, bias);
   if (getenv("AGX_APPLE9_TRACE_REJECT"))
      fprintf(stderr, "APPLE9_RENDER_BEGIN_OK\n");
}

void
hk_apple9_draw(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   if (vk_command_buffer_get_record_result(&cmd->vk) != VK_SUCCESS)
      return;
   if (!agx_is_indirect(draw.b) && (!draw.b.count[0] || !draw.b.count[1]))
      return;
   struct hk_cs *cs = cmd->current_cs.gfx;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   enum mesa_prim primitive = vk_topology_to_mesa(dyn->ia.primitive_topology);
   if (!cs || !cs->apple9_render ||
       agx_is_indirect(draw.b) ||
       (primitive != MESA_PRIM_TRIANGLES && primitive != MESA_PRIM_TRIANGLE_FAN &&
        primitive != MESA_PRIM_TRIANGLE_STRIP) ||
       dyn->ia.primitive_restart_enable) {
      if (getenv("AGX_APPLE9_TRACE_REJECT"))
         fprintf(stderr, "APPLE9_DRAW_INFO indirect=%u first_instance=%u topology=%u restart=%u indexed=%u\n",
                 agx_is_indirect(draw.b), draw.start_instance,
                 dyn->ia.primitive_topology, dyn->ia.primitive_restart_enable,
                 draw.indexed);
      hk_apple9_graphics_reject(cmd, "draw-shape-or-count");
      return;
   }
   if (!gfx->shaders[MESA_SHADER_VERTEX] || !gfx->shaders[MESA_SHADER_FRAGMENT] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] || gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->xfb_enabled || gfx->descriptors.root.draw.pipeline_stats_flags ||
       gfx->occlusion.mode != AGX_VISIBILITY_MODE_NONE) {
      hk_apple9_graphics_reject(cmd, "stages-transform-feedback-or-queries");
      return;
   }
   unsigned samples = gfx->render.tilebuffer.nr_samples;
   if (dyn->ms.rasterization_samples != samples ||
       dyn->ms.alpha_to_coverage_enable ||
       dyn->ms.alpha_to_one_enable ||
       (dyn->ms.sample_mask & BITFIELD_MASK(samples)) != BITFIELD_MASK(samples) ||
       dyn->ms.sample_locations_enable ||
       dyn->ds.depth.bounds_test.enable) {
      hk_apple9_graphics_reject(cmd, "vertex-input-msaa-or-depth-stencil");
      return;
   }
   if (dyn->vp.viewport_count != 1 || dyn->vp.scissor_count != 1 ||
       dyn->vp.depth_clip_negative_one_to_one ||
       dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
       dyn->rs.rasterizer_discard_enable ||
       (dyn->rs.depth_bias.enable &&
        (dyn->rs.depth_bias.constant_factor != 0.0f ||
         dyn->rs.depth_bias.slope_factor != 0.0f ||
         dyn->rs.depth_bias.clamp != 0.0f)) ||
       dyn->rs.depth_clamp_enable ||
       dyn->rs.conservative_mode != VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT ||
       dyn->rs.rasterization_stream || dyn->feedback_loops) {
      hk_apple9_graphics_reject(cmd, "viewport-or-raster-state");
      return;
   }
   struct hk_shader *vs = &gfx->shaders[MESA_SHADER_VERTEX]->variants[0];
   struct hk_shader *fs = &gfx->shaders[MESA_SHADER_FRAGMENT]->variants[0];
   struct agx_apple9_blend blend;
   if (!hk_apple9_blend_state(&dyn->cb, gfx->render.color_att[0].vk_format,
                              samples, &blend) ||
       memcmp(&blend, &fs->apple9_blend, sizeof(blend))) {
      hk_apple9_graphics_reject(cmd, "unsupported-or-dynamic-blend-state");
      return;
   }
   /* Vertex fetch is part of the body, not a dynamically replaceable prolog.
    * Compare only inputs that the compiled body actually uses.
    */
   u_foreach_bit(a, vs->apple9_mapping.attributes) {
      const struct agx_apple9_vertex_layout *layout = &vs->apple9_mapping.vertex;
      const struct vk_vertex_attribute_state *attr = &dyn->vi->attributes[a];
      if (!(dyn->vi->attributes_valid & BITFIELD_BIT(a)) ||
          attr->binding != layout->buffer[a] ||
          attr->offset != layout->offset[a] ||
          vk_format_to_pipe_format(attr->format) != layout->format[a] ||
          dyn->vi_binding_strides[attr->binding] != layout->stride[a] ||
          (dyn->vi->bindings[attr->binding].input_rate == VK_VERTEX_INPUT_RATE_INSTANCE
             ? dyn->vi->bindings[attr->binding].divisor : 0) != layout->divisor[a]) {
         hk_apple9_graphics_reject(cmd, "dynamic-vertex-input");
         return;
      }
   }
   struct hk_apple9_render_pipeline pipeline;
   if (!vs->apple9 || !fs->apple9 ||
       !hk_apple9_link_render_pipeline(&pipeline, hk_apple9_stage(vs, true),
                                      hk_apple9_stage(fs, false))) {
      hk_apple9_graphics_reject(cmd, "shader-link");
      return;
   }
   pipeline.samples = samples;
   pipeline.primitive = primitive == MESA_PRIM_TRIANGLE_FAN
      ? AGX_PRIMITIVE_TRIANGLE_FAN :
      primitive == MESA_PRIM_TRIANGLE_STRIP
      ? AGX_PRIMITIVE_TRIANGLE_STRIP : AGX_PRIMITIVE_TRIANGLES;
   pipeline.color_formats[0] =
      hk_apple9_tile_color_format(gfx->render.color_att[0].vk_format);
   pipeline.flatshade_first =
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT;
   if (draw.indexed) {
      unsigned size = agx_index_size_to_B(gfx->index.size);
      bool needs_copy;
      if (!hk_apple9_index_range(draw.index_buffer, draw.index_buffer_range_B,
                                 draw.start, draw.b.count[0], size,
                                 &pipeline.index_buffer, &pipeline.index_extent,
                                 &needs_copy)) {
         hk_apple9_graphics_reject(cmd, "index-buffer-range");
         return;
      }
      uint64_t bytes = pipeline.index_extent;
      pipeline.index_size = size;
      /* Gallium allocates all buffer resources in LOW_VA. Vulkan allocations
       * need not be in that 4-GiB aperture. Copy indices on the GPU into
       * command-buffer-owned LOW_VA storage, ordered before this render job.
       * Never read application indices on the CPU at record time.
       */
      if (needs_copy) {
         struct agx_ptr indices =
            agx_pool_alloc_aligned(&cmd->apple9_compute, ALIGN_POT(bytes, 4), 64);
         if (!indices.cpu) {
            hk_apple9_graphics_reject(cmd, "index-buffer-allocation");
            return;
         }
         /* The fetcher's last dword may include up to three padding bytes. */
         memset((uint8_t *)indices.cpu + (bytes & ~3ull), 0,
                ALIGN_POT(bytes, 4) - (bytes & ~3ull));
         for (uint64_t copied = 0; copied < bytes;) {
            unsigned count = MIN2(bytes - copied, 16u * 65535u);
            libagx_copy_uchar(cmd, agx_1d(count), AGX_BARRIER_ALL | AGX_PREGFX,
                              indices.gpu + copied,
                              pipeline.index_buffer + copied);
            copied += count;
         }
         if (vk_command_buffer_get_record_result(&cmd->vk) != VK_SUCCESS)
            return;
         pipeline.index_buffer = indices.gpu;
      }
   }
   /* Native stream links have not been ported. Keep the measured fetch tail
    * mapped, and fail recording instead of emitting an Apple8 stream link.
    */
   if (cs->end - cs->current < hk_apple9_direct_draw_size(&pipeline) + 0x800) {
      hk_apple9_graphics_reject(cmd, "render-stream-capacity");
      return;
   }
   const VkViewport *vp = &dyn->vp.viewports[0];
   const VkRect2D *sc = &dyn->vp.scissors[0];
   const VkRect2D *area = &gfx->render.area;
   int64_t minx = MAX2(sc->offset.x, area->offset.x);
   int64_t miny = MAX2(sc->offset.y, area->offset.y);
   int64_t maxx = MIN2((int64_t)sc->offset.x + sc->extent.width,
                       (int64_t)area->offset.x + area->extent.width);
   int64_t maxy = MIN2((int64_t)sc->offset.y + sc->extent.height,
                       (int64_t)area->offset.y + area->extent.height);
   if (minx >= maxx || miny >= maxy)
      minx = miny = maxx = maxy = 0;
   struct hk_apple9_uniform_draw record = {
      .viewport_translate = {vp->x + vp->width * 0.5f,
                             vp->y + vp->height * 0.5f, vp->minDepth},
      .viewport_scale = {vp->width * 0.5f, vp->height * 0.5f,
                         vp->maxDepth - vp->minDepth},
      .scissor_min = {minx, miny},
      .scissor_max = {maxx, maxy},
      .depth_control = 0x00010200,
      .depth_face = {0x07200f00, 0x07200f00},
      .stencil = {0x0e000000, 0x0e000000},
      /* As on Apple8, clamp raster depth when clipping is disabled. */
      .raster_control =
         (vk_rasterization_state_depth_clip_enable(&dyn->rs) ? 1u << 10 :
                                                              1u << 11) |
         ((dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT) ? 1 : 0) |
         ((dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT) ? 2 : 0) |
         (dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE ? 1u << 16 : 0),
      .object_type = AGX_OBJECT_TYPE_TRIANGLE,
      .flatshade_first = pipeline.flatshade_first,
      .scissor_index = cs->scissor.size / AGX_SCISSOR_LENGTH,
      .disable_tri_merging = pipeline.fragment.disable_tri_merging,
      .uses_discard = pipeline.fragment.uses_discard,
      /* Even a zero-mask shader with no tile instructions must preserve
       * earlier color through ordered visibility, as in Gallium.
       */
      .reads_tile = pipeline.fragment.reads_tile || blend.colormask != 15,
   };
   /* Vulkan compare ops and Gallium PIPE_FUNC use the same eight values.
    * Like Gallium, disabling the test also disables writes. An absent depth
    * attachment behaves as ALWAYS with writes disabled.
    */
   bool depth_enabled = cs->cr.depth.buffer && dyn->ds.depth.test_enable;
   if (depth_enabled && (unsigned)dyn->ds.depth.compare_op > VK_COMPARE_OP_ALWAYS) {
      hk_apple9_graphics_reject(cmd, "depth-compare-op");
      return;
   }
   uint32_t depth_face = 0xf00 |
      ((depth_enabled ? dyn->ds.depth.compare_op : VK_COMPARE_OP_ALWAYS) << 24) |
      ((depth_enabled && dyn->ds.depth.write_enable) ? 0 : (1u << 21));
   record.depth_face[0] = record.depth_face[1] = depth_face;
   /* Vulkan stencil ops and compare ops match the native encodings. An
    * absent stencil attachment behaves as a disabled test.
    */
   if (cs->cr.stencil.buffer && dyn->ds.stencil.test_enable) {
      const struct vk_stencil_test_face_state *faces[2] = {
         &dyn->ds.stencil.front, &dyn->ds.stencil.back,
      };
      record.depth_control |= (1u << 18) | (1u << 19);
      for (unsigned f = 0; f < 2; ++f) {
         const struct vk_stencil_test_face_state *s = faces[f];
         if ((unsigned)s->op.compare > VK_COMPARE_OP_ALWAYS ||
             (unsigned)s->op.fail > VK_STENCIL_OP_DECREMENT_AND_WRAP ||
             (unsigned)s->op.pass > VK_STENCIL_OP_DECREMENT_AND_WRAP ||
             (unsigned)s->op.depth_fail > VK_STENCIL_OP_DECREMENT_AND_WRAP) {
            hk_apple9_graphics_reject(cmd, "stencil-op");
            return;
         }
         record.depth_face[f] |= s->reference & 0xff;
         record.stencil[f] = (s->write_mask & 0xff) |
                             ((s->compare_mask & 0xff) << 8) |
                             ((uint32_t)s->op.pass << 16) |
                             ((uint32_t)s->op.depth_fail << 19) |
                             ((uint32_t)s->op.fail << 22) |
                             ((uint32_t)s->op.compare << 25);
      }
   }
   struct agx_scissor_packed scissor;
   agx_pack(&scissor, SCISSOR, cfg) {
      cfg.min_x = minx;
      cfg.min_y = miny;
      cfg.max_x = maxx;
      cfg.max_y = maxy;
      cfg.min_z = MIN2(vp->minDepth, vp->maxDepth);
      cfg.max_z = MAX2(vp->minDepth, vp->maxDepth);
   }
   util_dynarray_append(&cs->scissor, scissor);
   struct agx_depth_bias_packed bias = {0};
   util_dynarray_append(&cs->depth_bias, bias);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   /* Instance-rate fetches and gl_BaseInstance read these sysvals; the
    * hardware instance ID itself starts at zero.
    */
   gfx->apple9_draw_params[0] = draw.indexed ? draw.index_bias : draw.start;
   gfx->apple9_draw_params[1] = draw.start_instance;
   if (!hk_apple9_graphics_resources(cmd, vs, 0, &record) ||
       !hk_apple9_graphics_resources(cmd, fs, 1, &record)) {
      hk_apple9_graphics_reject(cmd, "draw-resources");
      return;
   }
   if (!hk_apple9_prepare_draw(&dev->dev, &cmd->apple9_compute,
                              &cmd->apple9_context, &pipeline, &record, NULL)) {
      hk_apple9_graphics_reject(cmd, "draw-launch-or-state");
      return;
   }
   pipeline.ppp = record.ppp;
   pipeline.vertex_launch = record.launch[0] / 0x40;
   pipeline.pipeline_word = 0x01000000;
   cs->current = hk_apple9_emit_direct_draw(cs->current, &pipeline,
                                           draw.b.count[0], draw.b.count[1],
                                           draw.indexed ? draw.index_bias : draw.start) - 4;
   cs->stats.calls++;
   cs->stats.cmds++;
   agx_bo_reference(vs->bo);
   util_dynarray_append(&cmd->large_bos, vs->bo);
   agx_bo_reference(fs->bo);
   util_dynarray_append(&cmd->large_bos, fs->bo);
   if (getenv("AGX_APPLE9_TRACE_REJECT"))
      fprintf(stderr, "APPLE9_RENDER_DRAW_OK\n");
}

void
hk_apple9_end_rendering(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (vk_command_buffer_get_record_result(&cmd->vk) != VK_SUCCESS)
      return;
   if (!cs || !cs->apple9_render) {
      hk_apple9_graphics_reject(cmd, "render-without-stream");
      return;
   }
   /* Keep the background/EOT for clear-only passes, including partial areas. */
   hk_cmd_buffer_end_graphics(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   const struct hk_attachment *color = &render->color_att[0];
   if (color->resolve_mode != VK_RESOLVE_MODE_NONE) {
      const VkRenderingAttachmentInfo att = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(color->iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = color->resolve_mode,
         .resolveImageView = hk_image_view_to_handle(color->resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
      const VkRenderingInfo info = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = render->area,
         .layerCount = 1,
         .colorAttachmentCount = 1,
         .pColorAttachments = &att,
      };
      /* EOT stores every sample even for API DONT_CARE: the separate pass
       * samples the intermediate. Drop state before meta begins rendering. */
      memset(render, 0, sizeof(*render));
      hk_meta_resolve_rendering(cmd, &info);
   }
   if (getenv("AGX_APPLE9_TRACE_REJECT"))
      fprintf(stderr, "APPLE9_RENDER_END_OK\n");
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

static inline void
apple9_put_f32(void *ptr, float value)
{
   memcpy(ptr, &value, sizeof(value));
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
hk_apple9_prepare_draw(struct agx_device *dev, struct agx_pool *usc_pool,
                        struct agx_pool *context_pool,
                        const struct hk_apple9_render_pipeline *pipeline,
                        struct hk_apple9_uniform_draw *draw,
                        const struct hk_apple9_uniform_draw *previous)
{
   const struct hk_apple9_render_stage *stages[] = {&pipeline->vertex,
                                                     &pipeline->fragment};
   unsigned tile_bytes = 0;
   for (unsigned rt = 0; rt < pipeline->fragment.render_targets; rt++)
      tile_bytes += 4 * agx_apple9_color_words(pipeline->color_formats[rt]);
   unsigned samples = MAX2(pipeline->samples, 1);
   if (tile_bytes > (samples == 1 ? 128 : 64)) {
      if (getenv("AGX_APPLE9_TRACE_REJECT"))
         fprintf(stderr, "APPLE9_DRAW_STATE_FAIL reason=tile-allocation bytes=%u samples=%u\n",
              tile_bytes, samples);
      return false;
   }
   draw->samples = samples;
   draw->tile_bytes = tile_bytes;
   for (unsigned stage = 0; stage < 2; stage++) {
      const struct hk_apple9_render_stage *shader = stages[stage];
      if (!shader->bo || !shader->publication_count_valid) {
         if (getenv("AGX_APPLE9_TRACE_REJECT"))
            fprintf(stderr, "APPLE9_DRAW_STATE_FAIL reason=missing-stage stage=%u\n",
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
         if (getenv("AGX_APPLE9_TRACE_REJECT"))
            fprintf(stderr, "APPLE9_DRAW_STATE_FAIL reason=entry-allocation stage=%u code=0x%llx\n",
                 stage, (unsigned long long)draw->code[stage]);
         return false;
      }
   }
   for (unsigned stage = 0; stage < 2; stage++) {
      const struct hk_apple9_render_stage *shader = stages[stage];
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
         if (getenv("AGX_APPLE9_TRACE_REJECT"))
            fprintf(stderr, "APPLE9_DRAW_STATE_FAIL reason=preamble stage=%u offset=%u size=%u binary=%zu\n",
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
         if (getenv("AGX_APPLE9_TRACE_REJECT"))
            fprintf(stderr,
                 "APPLE9_DRAW_STATE_FAIL reason=launch stage=%u entry=0x%x "
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
hk_apple9_link_render_pipeline(struct hk_apple9_render_pipeline *pipeline,
                                struct hk_apple9_render_stage vertex,
                                struct hk_apple9_render_stage fragment)
{
   return hk_apple9_link_render_pipeline_with_prolog(
      pipeline, (struct hk_apple9_render_stage){0}, vertex, fragment);
}

bool
hk_apple9_link_render_pipeline_with_prolog(
   struct hk_apple9_render_pipeline *pipeline,
   struct hk_apple9_render_stage vertex_prolog,
   struct hk_apple9_render_stage vertex,
   struct hk_apple9_render_stage fragment)
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
   *pipeline = (struct hk_apple9_render_pipeline){
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
hk_apple9_direct_draw_size(const struct hk_apple9_render_pipeline *pipeline)
{
   assert(pipeline && pipeline->vertex.binary && pipeline->fragment.binary);
   return 0x78 + (pipeline->index_size ? 20 : 0);
}

static uint8_t *
hk_apple9_emit_draw_state(uint8_t *out,
                           const struct hk_apple9_render_pipeline *pipeline)
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
hk_apple9_emit_direct_draw(uint8_t *out,
                            const struct hk_apple9_render_pipeline *pipeline,
                            unsigned vertex_count, unsigned instance_count,
                            unsigned vertex_start)
{
   assert(vertex_count > 0 && instance_count > 0);
   out = hk_apple9_emit_draw_state(out, pipeline);
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
