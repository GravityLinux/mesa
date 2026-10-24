/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include <assert.h>
#include "agx_abi.h"
#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_ppp.h"
#include "asahi/lib/agx_apple9.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "agx_uvs.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_apple9_polygon_points.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_feedback.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_private.h"
#include "hk_shader.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/libagx/compression.h"
#include "asahi/libagx/libagx.h"
#include "asahi/libagx/mesh.h"
#include "asahi/libagx/query.h"
#include "asahi/libagx/tessellator.h"
#include "poly/geometry.h"
#include "util/u_debug.h"
#include "util/blend.h"
#include "util/format/format_utils.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_prim.h"
#include "vulkan/vulkan_core.h"
#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_intrinsics.h"
#include "nir_lower_blend.h"
#include "nir_xfb_info.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"
#include "vk_render_pass.h"
#include "vk_shader.h"
#include "vk_standard_sample_locations.h"
#include "vk_util.h"

DEBUG_GET_ONCE_OPTION(cached_hk_apple9_tess_batch, "HK_APPLE9_TESS_BATCH", NULL)
DEBUG_GET_ONCE_BOOL_OPTION(mesh_route_log, "HK_APPLE9_MESH_ROUTE_LOG", false)
DEBUG_GET_ONCE_OPTION(cached_hk_apple9_tess_scan_block,
                      "HK_APPLE9_TESS_SCAN_BLOCK", NULL)

#define IS_DIRTY(bit) BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_##bit)

#define IS_SHADER_DIRTY(bit)                                                   \
   (cmd->state.gfx.shaders_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

#define IS_LINKED_DIRTY(bit)                                                   \
   (cmd->state.gfx.linked_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

/* CTS coverage of indirect draws is pretty bad, so it's helpful to be able to
 * get some extra smoke testing.
 */
/* Read once: the environment is scanned linearly, which on every draw cost
 * 50-100 ns in a typical Proton environment. */
DEBUG_GET_ONCE_BOOL_OPTION(hk_test_indirects, "HK_TEST_INDIRECTS", false)
#define HK_TEST_INDIRECTS debug_get_option_hk_test_indirects()

UNUSED static inline void
print_draw(struct agx_draw d, FILE *fp)
{
   if (agx_is_indirect(d.b))
      fprintf(fp, "indirect (buffer %" PRIx64 "):", d.b.ptr);
   else
      fprintf(fp, "direct (%ux%u):", d.b.count[0], d.b.count[1]);

   if (d.index_size)
      fprintf(fp, " index_size=%u", agx_index_size_to_B(d.index_size));
   else
      fprintf(fp, " non-indexed");

   if (d.restart)
      fprintf(fp, " restart");

   if (d.index_bias)
      fprintf(fp, " index_bias=%u", d.index_bias);

   if (d.start)
      fprintf(fp, " start=%u", d.start);

   if (d.start_instance)
      fprintf(fp, " start_instance=%u", d.start_instance);

   fprintf(fp, "\n");
}

static bool
hk_rast_discard(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* A non-zero rasterization stream acts as a rasterizer discard unless
    * there's a multistream geometry shader bound.
    */
   if (dyn->rs.rasterization_stream != 0) {
      struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      return !gs || !gs->variants[HK_GS_VARIANT_COUNT].info.gs.multistream;
   }

   return dyn->rs.rasterizer_discard_enable;
}

static void
hk_cmd_buffer_dirty_render_pass(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* These depend on color attachment count */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_WRITE_MASKS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED);

   /* These depend on the depth/stencil format */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_FACTORS);

   /* This may depend on render targets for ESO */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);

   /* This may depend on render targets */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP);
}

void
hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
       (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
      char gcbiar_data[VK_GCBIARR_DATA_SIZE(HK_MAX_RTS)];
      const VkRenderingInfo *resume_info =
         vk_get_command_buffer_inheritance_as_rendering_resume(
            cmd->vk.level, pBeginInfo, gcbiar_data);
      if (resume_info) {
         hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), resume_info);
      } else {
         const VkCommandBufferInheritanceRenderingInfo *inheritance_info =
            vk_get_command_buffer_inheritance_rendering_info(cmd->vk.level,
                                                             pBeginInfo);
         assert(inheritance_info);

         struct hk_rendering_state *render = &cmd->state.gfx.render;
         render->flags = inheritance_info->flags;
         render->area = (VkRect2D){};
         render->layer_count = 0;
         render->view_mask = inheritance_info->viewMask;
         render->tilebuffer.nr_samples = inheritance_info->rasterizationSamples;

         render->color_att_count = inheritance_info->colorAttachmentCount;
         for (uint32_t i = 0; i < render->color_att_count; i++) {
            render->color_att[i].vk_format =
               inheritance_info->pColorAttachmentFormats[i];
         }
         render->depth_att.vk_format = inheritance_info->depthAttachmentFormat;
         render->stencil_att.vk_format =
            inheritance_info->stencilAttachmentFormat;

         const VkRenderingAttachmentLocationInfoKHR att_loc_info_default = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
            .colorAttachmentCount = inheritance_info->colorAttachmentCount,
         };
         const VkRenderingAttachmentLocationInfoKHR *att_loc_info =
            vk_get_command_buffer_rendering_attachment_location_info(
               cmd->vk.level, pBeginInfo);
         if (att_loc_info == NULL)
            att_loc_info = &att_loc_info_default;

         vk_cmd_set_rendering_attachment_locations(&cmd->vk, att_loc_info);

         hk_cmd_buffer_dirty_render_pass(cmd);
      }
   }

   hk_cmd_buffer_dirty_all(cmd);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;
}

void
hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd)
{
   hk_cmd_buffer_dirty_all(cmd);

   /* From the Vulkan 1.3.275 spec:
    *
    *    "...There is one exception to this rule - if the primary command
    *    buffer is inside a render pass instance, then the render pass and
    *    subpass state is not disturbed by executing secondary command
    *    buffers."
    *
    * We need to reset everything EXCEPT the render pass state.
    */
   struct hk_rendering_state render_save = cmd->state.gfx.render;
   memset(&cmd->state.gfx, 0, sizeof(cmd->state.gfx));
   cmd->state.gfx.render = render_save;
}

static void
hk_attachment_use_implicit_sync(struct hk_cmd_buffer *cmd,
                                const struct hk_attachment *att)
{
   if (att->iview) {
      hk_cmd_use_implicit_sync(
         cmd, container_of(att->iview->vk.image, struct hk_image, vk)
                 ->implicit_sync_res);
   }
   if (att->resolve_iview) {
      hk_cmd_use_implicit_sync(
         cmd, container_of(att->resolve_iview->vk.image, struct hk_image, vk)
                 ->implicit_sync_res);
   }
}

static void
hk_attachment_init(struct hk_attachment *att,
                   const VkRenderingAttachmentInfo *info)
{
   if (info == NULL || info->imageView == VK_NULL_HANDLE) {
      *att = (struct hk_attachment){
         .iview = NULL,
      };
      return;
   }

   VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
   *att = (struct hk_attachment){
      .vk_format = iview->vk.format,
      .iview = iview,
   };

   if (info->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(hk_image_view, res_iview, info->resolveImageView);
      att->resolve_mode = info->resolveMode;
      att->resolve_iview = res_iview;
   }

   const VkRenderingAttachmentFlagsInfoKHR *flags =
      vk_find_struct_const(info->pNext, RENDERING_ATTACHMENT_FLAGS_INFO_KHR);
   if (flags)
      att->flags = flags->flags;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetRenderingAreaGranularityKHR(
   VkDevice device, const VkRenderingAreaInfoKHR *pRenderingAreaInfo,
   VkExtent2D *pGranularity)
{
   *pGranularity = (VkExtent2D){.width = 1, .height = 1};
}

static bool
is_attachment_stored(const VkRenderingAttachmentInfo *att)
{
   /* When resolving, we store the intermediate multisampled image as the
    * resolve is a separate control stream. This could be optimized.
    */
   return att->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
          att->resolveMode != VK_RESOLVE_MODE_NONE;
}

static struct hk_bg_eot
hk_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                bool store, bool partial_render, bool incomplete_render_area)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   if (dev->dev.chip == AGX_CHIP_G16G)
      return hk_apple9_build_bg_eot(cmd, info, store, partial_render,
                                   incomplete_render_area);

   /* Construct the key */
   struct agx_bg_eot_key key = {.tib = render->tilebuffer};
   static_assert(AGX_BG_EOT_NONE == 0, "default initializer");

   key.tib.layered = (render->cr.layers > 1);

   bool needs_textures_for_spilled_rts =
      agx_tilebuffer_spills(&render->tilebuffer) && !partial_render && !store;

   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE)
         continue;

      /* Partial render programs exist only to store/load the tilebuffer to
       * main memory. When render targets are already spilled to main memory,
       * there's nothing to do.
       */
      if (key.tib.spilled[i] && (partial_render || store))
         continue;

      if (store) {
         bool should_store = is_attachment_stored(att_info);

         /* Partial renders always need to flush to memory. */
         should_store |= partial_render;

         if (should_store)
            key.op[i] = AGX_EOT_STORE;
      } else {
         bool load = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
         bool clear = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;

         /* The background program used for partial renders must always load
          * whatever was stored in the mid-frame end-of-tile program.
          */
         load |= partial_render;

         /* With an incomplete render area, we're forced to load back tiles and
          * then use the 3D pipe for the clear.
          */
         load |= incomplete_render_area;

         /* Don't read back spilled render targets, they're already in memory */
         load &= !key.tib.spilled[i];

         /* This is a very frustrating corner case. From the spec:
          *
          *     VK_ATTACHMENT_STORE_OP_NONE specifies the contents within the
          *     render area are not accessed by the store operation as long as
          *     no values are written to the attachment during the render pass.
          *
          * With VK_ATTACHMENT_STORE_OP_NONE, we suppress stores on the main
          * end-of-tile program. Unfortunately, that's not enough: we also need
          * to preserve the contents throughout partial renders. The easiest way
          * to do that is forcing a load in the background program, so that
          * partial stores for unused attachments will be no-op'd by writing
          * existing contents.
          *
          * Optimizing this would require nontrivial tracking. Fortunately,
          * this is all Android gunk and we don't have to care too much for
          * dekstop games. So do the simple thing.
          */
         bool no_store = (att_info->storeOp == VK_ATTACHMENT_STORE_OP_NONE);
         bool no_store_wa = no_store && !load && !clear;
         if (no_store_wa) {
            perf_debug(cmd, "STORE_OP_NONE workaround");
         }

         load |= no_store_wa;

         /* Don't apply clears for spilled render targets when we clear the
          * render area explicitly after.
          */
         if (key.tib.spilled[i] && incomplete_render_area)
            continue;

         if (load)
            key.op[i] = AGX_BG_LOAD;
         else if (clear)
            key.op[i] = AGX_BG_CLEAR;
      }
   }

   /* Begin building the pipeline */
   size_t usc_size = agx_usc_size(3 + HK_MAX_RTS);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return (struct hk_bg_eot){.usc = t.gpu};

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   bool uses_txf = false;
   unsigned uniforms = 0;
   unsigned nr_tex = 0;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[rt];
      struct hk_image_view *iview = render->color_att[rt].iview;

      if (key.op[rt] == AGX_BG_LOAD) {
         uses_txf = true;

         const struct agx_texture_packed *desc =
            key.tib.layered ? &iview->planes[0].layered_background
                            : &iview->planes[0].background;

         agx_usc_pack(&b, TEXTURE, cfg) {
            /* Shifted to match eMRT indexing, could be optimized */
            cfg.start = rt * 2;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(cmd, desc, sizeof(*desc), 8);
         }

         nr_tex = (rt * 2) + 1;
      } else if (key.op[rt] == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         uint64_t colour =
            hk_pool_upload(cmd, &att_info->clearValue.color, 16, 16);

         agx_usc_uniform(&b, 4 + (8 * rt), 8, colour);
         uniforms = MAX2(uniforms, 4 + (8 * rt) + 8);
      } else if (key.op[rt] == AGX_EOT_STORE) {
         const struct agx_pbe_packed *desc = key.tib.layered
                                                ? &iview->planes[0].layered_eot
                                                : &iview->planes[0].eot;

         agx_usc_pack(&b, TEXTURE, cfg) {
            cfg.start = rt;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(cmd, desc, sizeof(*desc), 8);
         }

         nr_tex = rt + 1;
      }
   }

   if (needs_textures_for_spilled_rts) {
      hk_usc_upload_spilled_rt_descs(&b, cmd);
      uniforms = MAX2(uniforms, 4);
   }

   if (uses_txf) {
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);
   }

   /* For attachmentless rendering, we don't know the sample count until
    * draw-time. But we have trivial bg/eot programs in that case too.
    */
   if (key.tib.nr_samples >= 1) {
      agx_usc_push_packed(&b, SHARED, &key.tib.usc);
   } else {
      assert(key.tib.sample_size_B == 0);
      agx_usc_shared_none(&b);

      key.tib.nr_samples = 1;
   }

   /* Get the shader */
   key.reserved_preamble = uniforms;
   /* XXX: locking? */
   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, &key);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(&dev->dev, shader->ptr);
      cfg.unk_2 = 0;
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = shader->info.nr_gprs;

   if (shader->info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code =
            agx_usc_addr(&dev->dev, shader->ptr + shader->info.preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   struct hk_bg_eot ret = {.usc = t.gpu};

   agx_pack(&ret.counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->info.push_count;
      cfg.preshader_register_count = shader->info.nr_preamble_gprs;
      cfg.texture_state_register_count = nr_tex;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(uses_txf ? 1 : 0, false);
   }

   return ret;
}

static void
hk_merge_render_iview(struct hk_rendering_state *render,
                      struct hk_image_view *iview, bool zls)
{
   if (iview) {
      unsigned samples = iview->vk.image->samples;
      /* TODO: is this right for ycbcr? */
      unsigned level = iview->vk.base_mip_level;
      unsigned width = u_minify(iview->vk.image->extent.width, level);
      unsigned height = u_minify(iview->vk.image->extent.height, level);

      assert(render->tilebuffer.nr_samples == 0 ||
             render->tilebuffer.nr_samples == samples);
      render->tilebuffer.nr_samples = samples;

      /* TODO: Is this merging logic sound? Not sure how this is supposed to
       * work conceptually.
       */
      render->cr.width = MAX2(render->cr.width, width);
      render->cr.height = MAX2(render->cr.height, height);

      if (zls) {
         render->cr.zls_width = width;
         render->cr.zls_height = height;
      }
   }
}

/* A continued control stream loads what the previous one stored. Only the
 * last control stream of a render pass may drop stores, and only Apple9
 * honours that: its firmware preserves depth/stencil across partial renders
 * independently of these stores.
 */
static void
hk_pack_zls_control(struct agx_zls_control_packed *packed,
                    struct ail_layout *z_layout, struct ail_layout *s_layout,
                    const VkRenderingAttachmentInfo *z,
                    const VkRenderingAttachmentInfo *s,
                    bool incomplete_render_area, bool partial_render, bool last,
                    const bool *chain_load)
{
   struct agx_zls zls = {0};

   if (z) {
      zls.z_store = z->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    z->resolveMode != VK_RESOLVE_MODE_NONE || !last;
      if (last && z->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
          vk_image_layout_is_read_only(z->imageLayout,
                                               VK_IMAGE_ASPECT_DEPTH_BIT))
         zls.z_store = false;

      zls.z_load = z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   if (s) {
      zls.s_store = s->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    s->resolveMode != VK_RESOLVE_MODE_NONE || !last;
      if (last && s->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
          vk_image_layout_is_read_only(s->imageLayout,
                                               VK_IMAGE_ASPECT_STENCIL_BIT))
         zls.s_store = false;

      zls.s_load = s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   /* A merged or suspend/resume chain loads as its first render pass did. */
   if (chain_load && !partial_render) {
      zls.z_load = z && chain_load[0];
      zls.s_load = s && chain_load[1];
   }

   agx_pack_zls_control(packed, z_layout, s_layout, &zls);
}

/* HK_APPLE9_MERGE_RENDERS (default on; =0 disables): continue the hardware render of a
 * render pass in the next render pass when nothing but attachment barriers
 * separates them and both use the same colour attachments. The render covers
 * the whole attachment whatever the render area, and tiles keep their
 * contents between the passes, which satisfies the second pass's load.
 * Depth/stencil views and render areas must also match. As with suspend/resume,
 * keep the first pass's ZLS loads and the final pass's stores.
 */
/* Whether the render pass's draws may read this attachment other than as an
 * attachment (render->feedback). A written attachment may be read so only
 * when feedback loop is enabled for it: the feedback loop layout, or
 * VkAttachmentFeedbackLoopInfoEXT under unified image layouts (as RADV
 * reads it). An attachment nothing has written yet, with LOAD_OP_NONE, may
 * also be sampled in GENERAL. GENERAL alone is not a feedback loop, so
 * renders in GENERAL keep barrier hoisting.
 */
static bool
hk_attachment_may_be_sampled(const VkRenderingAttachmentInfo *att)
{
   if (!att || !att->imageView)
      return false;
   if (att->imageLayout == VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)
      return true;
   const VkAttachmentFeedbackLoopInfoEXT *loop =
      vk_find_struct_const(att->pNext, ATTACHMENT_FEEDBACK_LOOP_INFO_EXT);
   if (loop && loop->feedbackLoopEnable)
      return true;
   return att->imageLayout == VK_IMAGE_LAYOUT_GENERAL &&
          att->loadOp == VK_ATTACHMENT_LOAD_OP_NONE;
}

static bool
hk_render_merge_enabled(struct hk_cmd_buffer *cmd)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_APPLE9_MERGE_RENDERS", true);

   return enabled && hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G &&
          cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY && !cmd->in_meta;
}

/* Key a render pass that may merge, from its attachment views, or return false.
 * The caller checks load and store operations.
 */
bool
hk_render_merge_key_init(struct hk_render_merge_key *key, VkRenderingFlags flags,
                    uint32_t layer_count, uint32_t view_mask,
                    uint32_t color_count, struct hk_image_view *const *views,
                    struct hk_image_view *depth, struct hk_image_view *stencil,
                    uint8_t ds_read_only, VkRect2D area)
{
   if ((flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT |
                 VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)) ||
       color_count > HK_MAX_RTS)
      return false;
   /* A depth/stencil-only pass has the same fixed framebuffer and ZLS state
    * as one with colour attachments. The depth/stencil key, loads, stores
    * and barriers below already constrain its continuation. Attachmentless
    * passes still have no sample count until draw time, so keep them out.
    */
   if (!color_count && !depth && !stencil)
      return false;
   /* Merged multiview renders fail multiview.readback_implicit_clear 8_1_1_8
    * (cause not understood yet); keep multiview renders separate. */
   if (view_mask)
      return false;

   memset(key, 0, sizeof(*key));
   key->color_count = color_count;
   key->layer_count = layer_count;
   key->view_mask = view_mask;
   key->ds_read_only = ds_read_only;
   if (depth || stencil)
      key->ds_area = area;
   for (unsigned i = 0; i < HK_MAX_RTS + 2; ++i) {
      const struct hk_image_view *view = i == HK_MAX_RTS ? depth :
         i == HK_MAX_RTS + 1 ? stencil : i < color_count ? views[i] : NULL;
      if (view) {
         key->image[i] = view->vk.image;
         key->level[i] = view->vk.base_mip_level;
         key->layer[i] = view->vk.base_array_layer;
         key->format[i] = view->vk.format;
      }
   }
   return true;
}

static bool
hk_render_merge_begin(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_image_view *views[HK_MAX_RTS] = {NULL};
   struct hk_render_merge_key key;
   bool empty = cmd->empty_barriers.size != 0;
   if (empty && memcmp(&info->renderArea, &cmd->merge_empty_render->area,
                       sizeof(info->renderArea)))
      return false;
   const VkRenderingAttachmentInfo *ds[2] = {
      info->pDepthAttachment, info->pStencilAttachment};
   struct hk_image_view *ds_views[2] = {NULL};
   uint8_t ds_store_mask = 0, ds_read_only = 0;
   for (unsigned i = 0; i < 2; ++i) {
      if (!ds[i] || !ds[i]->imageView)
         continue;
      if (ds[i]->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ||
          ds[i]->resolveMode != VK_RESOLVE_MODE_NONE ||
          hk_attachment_may_be_sampled(ds[i]) ||
          (cmd->vk.dynamic_graphics_state.feedback_loops &
             (i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT)))
         return false;
      if (empty && (ds[i]->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD ||
                    (ds[i]->imageLayout != VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL &&
                     ds[i]->imageLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
                     ds[i]->imageLayout != (i ? VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL :
                                               VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)) ||
                    hk_attachment_may_be_sampled(ds[i])))
         return false;
      ds_views[i] = hk_image_view_from_handle(ds[i]->imageView);
      if (vk_image_layout_is_read_only(ds[i]->imageLayout,
            i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT))
         ds_read_only |= BITFIELD_BIT(i);
      if (is_attachment_stored(ds[i]))
         ds_store_mask |= BITFIELD_BIT(i);
   }
   if (cmd->merge_ds_store_mask & ~ds_store_mask)
      return false;

   uint32_t store_mask = 0;
   for (unsigned i = 0; i < MIN2(info->colorAttachmentCount, HK_MAX_RTS); ++i) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      views[i] = hk_image_view_from_handle(att->imageView);
      if (empty && att->imageView &&
          (att->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD ||
           (att->imageLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
            att->imageLayout != VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL) ||
           hk_attachment_may_be_sampled(att)))
         return false;
      /* Clears would need an in-pass clear over the render area. */
      if (att->imageView &&
          (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ||
           hk_attachment_may_be_sampled(att) ||
           (cmd->vk.dynamic_graphics_state.feedback_loops & VK_IMAGE_ASPECT_COLOR_BIT)))
         return false;
      if (att->imageView && is_attachment_stored(att))
         store_mask |= BITFIELD_BIT(i);
   }
   /* The merged render ends with this pass's store operations. An attachment
    * an earlier pass stored but this one does not would lose the earlier
    * pass's results outside this pass's render area. */
   if (cmd->merge_store_mask & ~store_mask) {
      hk_render_merge_trace("begin: store", cmd->merge_store_mask, store_mask);
      return false;
   }

   return !cmd->merge_feedback && hk_render_merge_enabled(cmd) &&
          hk_render_merge_key_init(&key,
                              info->flags & ~(empty ? VK_RENDERING_SUSPENDING_BIT : 0),
                              info->layerCount,
                              info->viewMask, info->colorAttachmentCount, views,
                              ds_views[0], ds_views[1], ds_read_only, info->renderArea) &&
          !memcmp(&key, &cmd->merge_key, sizeof(key));
}

/* HK_APPLE9_MERGE_RESUME (default on): a suspended render pass keeps its
 * render open for the resuming one. Only state commands may come between
 * them, so the resuming pass continues the same render pass instance in one
 * hardware render: loads from the first pass, stores from the last. */
static bool
hk_render_suspend_keeps_open(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_APPLE9_MERGE_RESUME", true);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   return enabled && cs && !render->feedback && !render->raw_color_mask &&
          (render->flags & VK_RENDERING_SUSPENDING_BIT) &&
          hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G &&
          cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY && !cmd->in_meta &&
          !cmd->sync_fallback && !cmd->current_cs.pre_gfx &&
          !cmd->current_cs.post_gfx && !cmd->tess_batch.draws;
}

/* At the end of a render pass, whether its render can stay open. */
static bool
hk_render_merge_end(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_image_view *views[HK_MAX_RTS] = {NULL};

   if (!cs || render->feedback || render->raw_color_mask ||
       !hk_render_merge_enabled(cmd) ||
       cmd->sync_fallback || cmd->current_cs.pre_gfx ||
       cmd->current_cs.post_gfx || cmd->tess_batch.draws ||
       cmd->state.gfx.occlusion.active_count)
      return false;

   for (unsigned i = 0; i < MIN2(render->color_att_count, HK_MAX_RTS); ++i) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         return false;
      views[i] = render->color_att[i].iview;
   }

   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE ||
       render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      return false;

   /* A render pass that resumed in its render ends it like any other. */
   return hk_render_merge_key_init(
      &cmd->merge_key,
      render->flags & ~(render->resumed ? VK_RENDERING_RESUMING_BIT : 0),
      render->layer_count, render->view_mask,
      render->color_att_count, views,
      render->depth_att.iview, render->stencil_att.iview,
      render->ds_read_only, render->area);
}

/* A merged empty chain can still end without any draws. Retain the first
 * pass's clear values so empty-render conversion does not lose them when a
 * later LOAD pass, including suspend/resume, supplies the final render state.
 */
static void
hk_render_inherit_empty_clears(struct hk_cmd_buffer *cmd)
{
   if (!cmd->merge_empty)
      return;
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   const struct hk_rendering_state *first = cmd->merge_empty_render;
   for (unsigned i = 0; i < HK_MAX_RTS + 2; ++i) {
      struct hk_attachment *att = i == HK_MAX_RTS ? &render->depth_att :
         i == HK_MAX_RTS + 1 ? &render->stencil_att : &render->color_att[i];
      const struct hk_attachment *old = i == HK_MAX_RTS ? &first->depth_att :
         i == HK_MAX_RTS + 1 ? &first->stencil_att : &first->color_att[i];
      att->clear = old->clear;
      memcpy(att->clear_colour, old->clear_colour, sizeof(att->clear_colour));
   }
}

/* A compressed attachment must use raw stores as soon as feedback is possible.
 * GENERAL alone does not imply sampling, but the layout/enable bit or explicit
 * dynamic feedback state does. Keep raw stores sticky within the render.
 */
static uint8_t
hk_feedback_color_mask(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint8_t mask = 0;
   for (unsigned i = 0; i < info->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *a = &info->pColorAttachments[i];
      if (!a->imageView)
         continue;
      const struct hk_image_view *view = hk_image_view_from_handle(a->imageView);
      const struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
      if (!hk_feedback_can_compress(&dev->dev, image->vk.image_type,
                                    image->vk.format, image->vk.samples) ||
          !ail_is_level_logically_compressed(&image->planes[0].layout,
                                             view->vk.base_mip_level))
         continue;
      if (cmd->feedback_decompress || hk_attachment_may_be_sampled(a) ||
          (cmd->vk.dynamic_graphics_state.feedback_loops & VK_IMAGE_ASPECT_COLOR_BIT))
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

static bool
hk_feedback_decompress(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                         uint8_t mask)
{
   if (cmd->feedback_decompress)
      return true;
   u_foreach_bit(i, mask) {
      const struct hk_image_view *view =
         hk_image_view_from_handle(info->pColorAttachments[i].imageView);
      if (info->viewMask) {
         u_foreach_bit(v, info->viewMask) {
            if (!hk_meta_decompress_color(cmd, view, view->vk.base_array_layer + v))
               return false;
         }
      } else {
         for (unsigned v = 0; v < info->layerCount; v++) {
            if (!hk_meta_decompress_color(cmd, view, view->vk.base_array_layer + v))
               return false;
         }
      }
   }
   return true;
}

/* Splitting a live render keeps the partial EOT's stores and the partial BG's
 * loads. Rebuild only its final/partial colour stores with the new raw mask.
 * This also handles feedback enabled by binding a static-state pipeline.
 */
static bool
hk_feedback_prepare_draw(struct hk_cmd_buffer *cmd)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return false;

   struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (!cmd->rendering || cmd->in_meta ||
       !(cmd->vk.dynamic_graphics_state.feedback_loops & VK_IMAGE_ASPECT_COLOR_BIT))
      return true;
   VkRenderingAttachmentInfo attachments[HK_MAX_RTS];
   for (unsigned i = 0; i < render->color_att_count; i++) {
      attachments[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = render->store_mask & BITFIELD_BIT(i)
                       ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
      };
   }
   const VkRenderingInfo info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = attachments,
   };
   uint8_t mask = hk_feedback_color_mask(cmd, &info) & ~render->raw_color_mask;
   if (!mask)
      return true;

   hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
   hk_cmd_buffer_end_graphics(cmd);
   struct hk_rendering_state saved = *render;
   if (!hk_feedback_decompress(cmd, &info, mask))
      return false;
   *render = saved;
   cmd->rendering = true;
   render->feedback = true;
   render->raw_color_mask |= mask;
   render->cr.bg.main = render->cr.bg.partial;
   for (unsigned i = 0; i < render->color_att_count; i++)
      render->color_att[i].clear = false;
   render->cr.eot.main = hk_build_bg_eot(cmd, &info, true, false, true);
   render->cr.eot.partial = hk_build_bg_eot(cmd, &info, true, true, true);
   cmd->sync_render_continuation = false;
   cmd->state.gfx.render_gen++;
   hk_cmd_buffer_dirty_all(cmd);
   return !vk_command_buffer_has_error(&cmd->vk);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetAttachmentFeedbackLoopEnableEXT(VkCommandBuffer commandBuffer,
                                          VkImageAspectFlags aspectMask)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE) ||
       dyn->feedback_loops != aspectMask) {
      dyn->feedback_loops = aspectMask;
      BITSET_SET(dyn->set, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE);
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE);
   }
   if (cmd->rendering && aspectMask)
      cmd->state.gfx.render.feedback = true;
   if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G)
      hk_feedback_prepare_draw(cmd);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                     const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   uint8_t raw_mask = hk_feedback_color_mask(cmd, pRenderingInfo);
   if (raw_mask) {
      hk_cmd_render_merge_flush(cmd);
      if (!hk_feedback_decompress(cmd, pRenderingInfo, raw_mask))
         return;
   }
   if (!cmd->feedback_decompress && raw_mask)
      cmd->sync_render_continuation = false;
   bool merge = false;
   /* The resuming render pass of a suspended one continues its render. */
   bool resume = false;
   if (cmd->merge_cs && cmd->merge_suspended) {
      resume = (pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT) &&
               !cmd->in_meta;
      if (!resume)
         hk_cmd_render_merge_flush(cmd);
   }
   if (cmd->merge_cs && !resume) {
      merge = hk_render_merge_begin(cmd, pRenderingInfo);
      if (merge)
         hk_cmd_empty_barriers_flush(cmd, true);
      if (!merge) {
         hk_render_merge_trace(cmd->in_meta ? "begin: meta" : "begin: mismatch",
                               pRenderingInfo->colorAttachmentCount,
                               pRenderingInfo->colorAttachmentCount ?
                                  pRenderingInfo->pColorAttachments[0].loadOp : ~0ull);
         hk_cmd_render_merge_flush_why(cmd, NULL);
      }
   }

   /* A continued render needs no dependency on itself. */
   cmd->sync_render_continuation =
      (pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT) != 0 && !resume && !raw_mask;
   cmd->sync_render_stages = cmd->sync_scope ?: HK_SYNC_GRAPHICS;
   cmd->state.gfx.render_gen++;
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   memset(render, 0, sizeof(*render));
   cmd->rendering = true;
   render->raw_color_mask = raw_mask;
   render->feedback = cmd->vk.dynamic_graphics_state.feedback_loops != 0;

   render->flags = pRenderingInfo->flags;
   render->area = pRenderingInfo->renderArea;
   for (uint32_t i = 0; i < MIN2(pRenderingInfo->colorAttachmentCount, 32); i++) {
      if (pRenderingInfo->pColorAttachments[i].imageView &&
          is_attachment_stored(&pRenderingInfo->pColorAttachments[i]))
         render->store_mask |= BITFIELD_BIT(i);
   }
   const VkRenderingAttachmentInfo *ds[2] = {
      pRenderingInfo->pDepthAttachment, pRenderingInfo->pStencilAttachment};
   for (unsigned i = 0; i < 2; ++i) {
      if (ds[i] && ds[i]->imageView) {
         if (is_attachment_stored(ds[i]))
            render->ds_store_mask |= BITFIELD_BIT(i);
         if (vk_image_layout_is_read_only(ds[i]->imageLayout,
               i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT))
            render->ds_read_only |= BITFIELD_BIT(i);
      }
   }
   const VkRenderingAttachmentInfo *feedback_att[HK_MAX_RTS + 2] = {
      pRenderingInfo->pDepthAttachment, pRenderingInfo->pStencilAttachment};
   for (uint32_t i = 0; i < MIN2(pRenderingInfo->colorAttachmentCount, HK_MAX_RTS); i++)
      feedback_att[2 + i] = &pRenderingInfo->pColorAttachments[i];
   for (unsigned i = 0; i < ARRAY_SIZE(feedback_att); ++i) {
      if (hk_attachment_may_be_sampled(feedback_att[i]))
         render->feedback = true;
   }
   for (unsigned i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      if (!(raw_mask & BITFIELD_BIT(i)))
         hk_feedback_invalidate_view(cmd,
            hk_image_view_from_handle(pRenderingInfo->pColorAttachments[i].imageView));
   }
   render->view_mask = pRenderingInfo->viewMask;
   render->layer_count = pRenderingInfo->layerCount;
   render->tilebuffer.nr_samples = 0;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   render->color_att_count = pRenderingInfo->colorAttachmentCount;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_attachment_init(&render->color_att[i],
                         &pRenderingInfo->pColorAttachments[i]);
      hk_attachment_use_implicit_sync(cmd, &render->color_att[i]);
   }

   hk_attachment_init(&render->depth_att, pRenderingInfo->pDepthAttachment);
   hk_attachment_init(&render->stencil_att, pRenderingInfo->pStencilAttachment);
   hk_attachment_use_implicit_sync(cmd, &render->depth_att);
   hk_attachment_use_implicit_sync(cmd, &render->stencil_att);

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_merge_render_iview(render, render->color_att[i].iview, false);
   }

   u_foreach_bit(i, render->store_mask) {
      struct hk_image_view *view = render->color_att[i].iview;
      struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
      if (image->planes[view->planes[0].image_plane].layout.tiling ==
          AIL_TILING_LINEAR)
         render->linear_store = true;
   }

   hk_merge_render_iview(
      render, render->depth_att.iview ?: render->stencil_att.iview, true);

   /* Infer for attachmentless. samples is inferred at draw-time. */
   render->cr.width =
      MAX2(render->cr.width, render->area.offset.x + render->area.extent.width);

   render->cr.height = MAX2(render->cr.height,
                            render->area.offset.y + render->area.extent.height);

   if (!render->cr.zls_width) {
      render->cr.zls_width = render->cr.width;
      render->cr.zls_height = render->cr.height;
   }

   render->cr.layers = layer_count;

   /* Choose a tilebuffer layout given the framebuffer key */
   enum pipe_format formats[HK_MAX_RTS] = {0};
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      formats[i] = hk_format_to_pipe_format(render->color_att[i].vk_format);
   }

   /* For now, we force layered=true since it makes compatibility problems way
    * easier.
    */
   render->tilebuffer = agx_build_tilebuffer_layout(
      formats, render->color_att_count, render->tilebuffer.nr_samples, true);

   if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G) {
      unsigned bytes = 0;
      for (unsigned rt = 0; rt < render->color_att_count; ++rt) {
         render->tilebuffer._offset_B[rt] = bytes;
         bytes += 4 * agx_apple9_color_words(formats[rt]);
         render->tilebuffer.spilled[rt] = false;
      }
      render->tilebuffer.sample_size_B = ALIGN_POT(
         agx_apple9_tile_sample_bytes(bytes, render->tilebuffer.nr_samples), 8);
      render->tilebuffer.tile_size = render->tilebuffer.nr_samples == 4
                                       ? 32 * 16 : 32 * 32;
   }

   if (cmd->feedback_decompress) {
      /* A self-blit must retire each whole compression block before another
       * tile reads it. The metadata covers 16x16 texels for every colour bpp. */
      struct drm_asahi_cmd_render raw = {0};
      agx_tilebuffer_set_drm_cmd(&raw, &render->tilebuffer);
      assert(raw.utile_width_px % 16 == 0);
      assert(raw.utile_height_px % 16 == 0);
   }

   const VkRenderingAttachmentLocationInfoKHR ral_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = pRenderingInfo->colorAttachmentCount,
   };
   vk_cmd_set_rendering_attachment_locations(&cmd->vk, &ral_info);

   hk_cmd_buffer_dirty_render_pass(cmd);

   /* Determine whether the render area is complete, enabling us to use a
    * fast-clear.
    *
    * TODO: If it is incomplete but tile aligned, it should be possibly to fast
    * clear with the appropriate settings. This is critical for performance.
    */
   bool incomplete_render_area =
      render->area.offset.x > 0 || render->area.offset.y > 0 ||
      render->area.extent.width < render->cr.width ||
      render->area.extent.height < render->cr.height ||
      (render->view_mask &&
       render->view_mask != BITFIELD64_MASK(render->cr.layers));

   perf_debug(cmd, "Rendering %ux%ux%u@%u %s%s", render->cr.width,
              render->cr.height, render->cr.layers,
              render->tilebuffer.nr_samples,
              render->view_mask ? " multiview" : "",
              incomplete_render_area ? " incomplete" : "");

   render->cr.bg.main = hk_build_bg_eot(cmd, pRenderingInfo, false, false,
                                        incomplete_render_area);
   render->cr.bg.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, false, true, incomplete_render_area);

   render->cr.eot.main =
      hk_build_bg_eot(cmd, pRenderingInfo, true, false, incomplete_render_area);
   render->cr.eot.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, true, true, incomplete_render_area);

   render->cr.isp_bgobjvals = 0x300;

   const VkRenderingAttachmentInfo *attach_z = pRenderingInfo->pDepthAttachment;
   const VkRenderingAttachmentInfo *attach_s =
      pRenderingInfo->pStencilAttachment;

   struct ail_layout *z_layout = NULL, *s_layout = NULL;

   if (attach_z != NULL && attach_z != VK_NULL_HANDLE && attach_z->imageView) {
      struct hk_image_view *view = render->depth_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      z_layout = &image->planes[0].layout;

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      const struct util_format_description *desc =
         util_format_description(hk_format_to_pipe_format(view->vk.format));

      assert(desc->format == PIPE_FORMAT_Z32_FLOAT ||
             desc->format == PIPE_FORMAT_Z16_UNORM ||
             desc->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT);

      render->cr.depth.buffer =
         hk_image_base_address(image, 0) +
         ail_get_layer_level_B(z_layout, first_layer, level);

      /* Main stride in pages */
      assert((z_layout->depth_px == 1 ||
              util_is_aligned(z_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned Z layers");

      unsigned stride_pages = z_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.depth.stride = ((stride_pages - 1) << 14) | 1;

      assert(z_layout->tiling != AIL_TILING_LINEAR && "must tile");

      if (z_layout->compressed) {
         render->cr.depth.meta =
            hk_image_base_address(image, 0) + z_layout->metadata_offset_B +
            (first_layer * z_layout->compression_layer_stride_B) +
            z_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(z_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned Z meta layers");

         unsigned stride_lines =
            z_layout->compression_layer_stride_B / AIL_CACHELINE;
         render->cr.depth.meta_stride = (stride_lines - 1) << 14;
      }

      float clear_depth = attach_z->clearValue.depthStencil.depth;

      if (z_layout->format == PIPE_FORMAT_Z16_UNORM) {
         render->cr.isp_bgobjdepth = _mesa_float_to_unorm(clear_depth, 16);
      } else {
         render->cr.isp_bgobjdepth = fui(clear_depth);
      }
   }

   if (attach_s != NULL && attach_s != VK_NULL_HANDLE && attach_s->imageView) {
      struct hk_image_view *view = render->stencil_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      /* Stencil is always the last plane (possibly the only plane) */
      unsigned plane = image->plane_count - 1;
      s_layout = &image->planes[plane].layout;
      assert(s_layout->format == PIPE_FORMAT_S8_UINT);

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      render->cr.stencil.buffer =
         hk_image_base_address(image, plane) +
         ail_get_layer_level_B(s_layout, first_layer, level);

      /* Main stride in pages */
      assert((s_layout->depth_px == 1 ||
              util_is_aligned(s_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned S layers");
      unsigned stride_pages = s_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.stencil.stride = ((stride_pages - 1) << 14) | 1;

      if (s_layout->compressed) {
         render->cr.stencil.meta =
            hk_image_base_address(image, plane) + s_layout->metadata_offset_B +
            (first_layer * s_layout->compression_layer_stride_B) +
            s_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(s_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned S meta layers");

         unsigned stride_lines =
            s_layout->compression_layer_stride_B / AIL_CACHELINE;

         render->cr.stencil.meta_stride = (stride_lines - 1) << 14;
      }

      render->cr.isp_bgobjvals |= attach_s->clearValue.depthStencil.stencil;
   }

   hk_pack_zls_control(&render->cr.zls_control, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false, false, NULL);

   hk_pack_zls_control(&render->cr.zls_control_partial, z_layout, s_layout,
                       attach_z, attach_s, incomplete_render_area, true, false,
                       NULL);

   const bool drop_stores =
      hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G &&
      !(render->flags & VK_RENDERING_SUSPENDING_BIT);
   /* The first render pass of a chain decides its loads. */
   if (!resume && !merge) {
      render->chain_load[0] =
         attach_z && attach_z->imageView &&
         (attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || incomplete_render_area);
      render->chain_load[1] =
         attach_s && attach_s->imageView &&
         (attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || incomplete_render_area);
   } else {
      memcpy(render->chain_load, cmd->merge_chain_load, sizeof(render->chain_load));
   }
   hk_pack_zls_control(&render->zls_last, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false, drop_stores,
                       (resume || merge) ? render->chain_load : NULL);
   hk_pack_zls_control(&render->zls_partial_last, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, true, drop_stores,
                       (resume || merge) ? render->chain_load : NULL);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;

   if (resume) {
      /* Same render pass instance: keep the open render's background and
       * load configuration; this pass's store operations apply at its end. */
      perf_debug(cmd, "Resuming the suspended render pass in its render");
      render->first_cs = cmd->merge_first_cs ? cmd->merge_cs : NULL;
      render->feedback |= cmd->merge_feedback;
      render->resumed = true;
      hk_render_inherit_empty_clears(cmd);
      cmd->merge_cs = NULL;
      cmd->merge_empty = false;
      cmd->merge_suspended = false;
      hk_cmd_buffer_dirty_all(cmd);
      return;
   }

   if (render->flags & VK_RENDERING_RESUMING_BIT)
      return;

   if (merge) {
      /* Continue the open render. Its background program, made for the
       * first pass, loads or clears every tile it processes. */
      perf_debug(cmd, "Merging render pass into the previous render");
      render->first_cs = cmd->merge_first_cs ? cmd->merge_cs : NULL;
      /* The render's earlier passes may have read their attachments too. */
      render->feedback |= cmd->merge_feedback;
      render->merge_reach = cmd->merge_reach;
      hk_render_inherit_empty_clears(cmd);
      cmd->merge_cs = NULL;
      cmd->merge_empty = false;
      hk_cmd_buffer_dirty_all(cmd);
      return;
   }

   /* The first control stream of the render pass is special since it gets
    * the clears. Create it and swap in the clear.
    */
   assert(!cmd->current_cs.gfx && "not already in a render pass");
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   if (!cs)
      return;

   cs->cr.bg.main = render->cr.bg.main;
   cs->cr.zls_control = render->cr.zls_control;
   cs->linear_store = render->linear_store;
   render->first_cs = cs;

   /* Reordering barrier for post-gfx, in case we had any. */
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   hk_cs_end_hint(cmd, HK_CS_END_BEGIN_RENDER);
   /* Don't reorder compute across render passes.
    *
    * TODO: Check if this is necessary if the proper PipelineBarriers are
    * handled... there may be CTS bugs...
    */
   hk_cmd_buffer_end_compute(cmd);

   /* If we spill colour attachments, we need to decompress them. This happens
    * at the start of the render; it is not re-emitted when resuming
    * secondaries. It could be hoisted to the start of the command buffer but
    * we're not that clever yet.
    */
   if (agx_tilebuffer_spills(&render->tilebuffer)) {
      perf_debug(cmd, "eMRT render pass");

      for (unsigned i = 0; i < render->color_att_count; ++i) {
         struct hk_image_view *view = render->color_att[i].iview;
         if (view) {
            struct hk_image *image =
               container_of(view->vk.image, struct hk_image, vk);

            /* TODO: YCbCr interaction? */
            uint8_t plane = 0;
            uint8_t image_plane = view->planes[plane].image_plane;
            struct ail_layout *layout = &image->planes[image_plane].layout;

            if (ail_is_level_logically_compressed(layout,
                                                  view->vk.base_mip_level)) {

               perf_debug(cmd, "Decompressing in-place");

               unsigned level = view->vk.base_mip_level;
               unsigned layer = view->vk.base_array_layer;
               uint64_t base = hk_image_base_address(image, image_plane);

               struct libagx_decompress_images imgs = {
                  .compressed = view->planes[plane].emrt_texture,
                  .uncompressed = view->planes[plane].emrt_pbe,
               };

               struct agx_grid grid =
                  agx_3d(ail_metadata_width_tl(layout, level) * 32,
                         ail_metadata_height_tl(layout, level), layer_count);

               libagx_decompress(cmd, grid, AGX_BARRIER_ALL | AGX_PREGFX,
                                 layout, layer, level, base,
                                 hk_pool_upload(cmd, &imgs, sizeof(imgs), 64));
            }
         }
      }
   }

   uint32_t clear_count = 0;
   VkClearAttachment clear_att[HK_MAX_RTS + 1];
   bool resolved_clear = false;

   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att_info =
         &pRenderingInfo->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE ||
          att_info->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;

      clear_att[clear_count++] = (VkClearAttachment){
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = i,
         .clearValue = att_info->clearValue,
      };

      render->color_att[i].clear = true;

      static_assert(sizeof(render->color_att[i].clear_colour) ==
                    sizeof(att_info->clearValue));
      memcpy(render->color_att[i].clear_colour, &att_info->clearValue,
             sizeof(att_info->clearValue));

      resolved_clear |= is_attachment_stored(att_info);
   }

   clear_att[clear_count] = (VkClearAttachment){
      .aspectMask = 0,
   };

   if (attach_z && attach_z->imageView != VK_NULL_HANDLE &&
       attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      clear_att[clear_count].clearValue.depthStencil.depth =
         attach_z->clearValue.depthStencil.depth;

      render->depth_att.clear = true;
      render->depth_att.clear_colour[0] =
         fui(attach_z->clearValue.depthStencil.depth);

      resolved_clear |= is_attachment_stored(attach_z);
   }

   if (attach_s != NULL && attach_s->imageView != VK_NULL_HANDLE &&
       attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      clear_att[clear_count].clearValue.depthStencil.stencil =
         attach_s->clearValue.depthStencil.stencil;

      render->stencil_att.clear = true;
      render->stencil_att.clear_colour[1] =
         attach_s->clearValue.depthStencil.stencil;

      resolved_clear |= is_attachment_stored(attach_s);
   }

   if (clear_att[clear_count].aspectMask != 0)
      clear_count++;

   if (clear_count > 0 && incomplete_render_area) {
      const VkClearRect clear_rect = {
         .rect = render->area,
         .baseArrayLayer = 0,
         .layerCount = render->view_mask ? 1 : render->layer_count,
      };

      hk_CmdClearAttachments(hk_cmd_buffer_to_handle(cmd), clear_count,
                             clear_att, 1, &clear_rect);
   } else {
      /* If a tile is empty, we do not want to process it, as the redundant
       * roundtrip of memory-->tilebuffer-->memory wastes a tremendous amount of
       * memory bandwidth. Any draw marks a tile as non-empty, so we only need
       * to process empty tiles if the background+EOT programs have a side
       * effect. This is the case exactly when there is an attachment we are
       * fast clearing and then storing.
       */
      cs->cr.process_empty_tiles = resolved_clear;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndRendering2KHR(VkCommandBuffer commandBuffer,
                       const VkRenderingEndInfoKHR *pRenderingEndInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* The last control stream of the render pass is special since it gets its
    * stores dropped. Swap it in.
    */
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (cs) {
      cs->cr.eot.main = (render->flags & VK_RENDERING_SUSPENDING_BIT)
                           ? render->cr.eot.partial : render->cr.eot.main;
      /* zls_last already preserves suspended stores. Its first-stream
       * loads must retain CLEAR; the partial control would force a load. */
      cs->cr.zls_control = cs == render->first_cs ? render->zls_last
                                                  : render->zls_partial_last;
   }

   perf_debug(cmd, "End rendering");
   const bool suspend = hk_render_suspend_keeps_open(cmd, cs);
   if (suspend || hk_render_merge_end(cmd, cs)) {
      cmd->merge_suspended = suspend;
      memcpy(cmd->merge_chain_load, render->chain_load,
             sizeof(cmd->merge_chain_load));
      /* Keep the render open for a following render pass to continue. */
      /* A continuing pass stores at least what the passes before it did, so
       * its mask covers the whole open render. */
      cmd->merge_store_mask = render->store_mask;
      cmd->merge_ds_store_mask = render->ds_store_mask;
      cmd->merge_feedback = render->feedback;
      cmd->merge_reach = hk_merge_graphics_stages() | render->merge_reach;
      cmd->merge_cs = cs;
      cmd->merge_first_cs = cs == render->first_cs;
      /* A render without draws yet: keep its state for the empty-render
       * optimization in case nothing continues it. */
      cmd->merge_empty = !cs->stats.cmds;
      if (cmd->merge_empty) {
         if (!cmd->merge_empty_render)
            cmd->merge_empty_render = malloc(sizeof(*render));
         if (cmd->merge_empty_render) {
            memcpy(cmd->merge_empty_render, render, sizeof(*render));
         } else {
            cmd->merge_cs = NULL;
            cmd->merge_empty = false;
            hk_cs_end_hint(cmd, HK_CS_END_RENDER_PASS);
            hk_cmd_buffer_end_graphics(cmd);
         }
      }
      /* Barriers before the next pass go ahead of the stream's first event. */
      cmd->merge_event = *util_dynarray_element(&cmd->sync_first_event, uint32_t,
                                                cs->sync_id);
      assert(cmd->merge_event != HK_SYNC_NONE &&
             "a stream with draws has a sync event");
   } else {
      if (cs) {
         /* Why this render is not kept open (diagnostic). */
         bool resolve = false;
         for (unsigned i = 0; i < render->color_att_count; ++i)
            resolve |= render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE;
         hk_render_merge_trace(
            "end: ineligible",
            (render->depth_att.iview || render->stencil_att.iview) |
               resolve << 1 | cmd->sync_fallback << 2 |
               (cmd->current_cs.pre_gfx || cmd->current_cs.post_gfx) << 3 |
               !!cmd->tess_batch.draws << 4 |
               !!cmd->state.gfx.occlusion.active_count << 5 | cmd->in_meta << 6 |
               (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) << 7 |
               !cs->stats.cmds << 8 | !render->color_att_count << 9,
            render->flags);
      }
      hk_cs_end_hint(cmd, HK_CS_END_RENDER_PASS);
      hk_cmd_buffer_end_graphics(cmd);
   }

   bool need_resolve = false;

   /* Translate render state back to VK for meta */
   VkRenderingAttachmentInfo vk_color_att[HK_MAX_RTS];
   VkRenderingAttachmentFlagsInfoKHR vk_color_flags[HK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         need_resolve = true;

      vk_color_flags[i] = (VkRenderingAttachmentFlagsInfoKHR){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
         .flags = render->color_att[i].flags,
      };
      vk_color_att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .pNext = &vk_color_flags[i],
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = render->color_att[i].resolve_mode,
         .resolveImageView =
            hk_image_view_to_handle(render->color_att[i].resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
   }

   const VkRenderingAttachmentFlagsInfoKHR vk_depth_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->depth_att.flags,
   };
   const VkRenderingAttachmentInfo vk_depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_depth_flags,
      .imageView = hk_image_view_to_handle(render->depth_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->depth_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->depth_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingAttachmentFlagsInfoKHR vk_stencil_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->stencil_att.flags,
   };
   const VkRenderingAttachmentInfo vk_stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_stencil_flags,
      .imageView = hk_image_view_to_handle(render->stencil_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->stencil_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->stencil_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingInfo vk_render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = vk_color_att,
      .pDepthAttachment = &vk_depth_att,
      .pStencilAttachment = &vk_stencil_att,
   };

   if (render->flags & VK_RENDERING_SUSPENDING_BIT)
      need_resolve = false;

   memset(render, 0, sizeof(*render));
   cmd->rendering = false;

   if (need_resolve) {
      perf_debug(cmd, "Resolving render pass, colour store op %u",
                 vk_color_att[0].storeOp);

      hk_meta_resolve_rendering(cmd, &vk_render);
   }
}

static void
hk_init_heap(const void *data)
{
   struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)data;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   perf_debug(cmd, "Allocating heap");

   size_t size = 128 * 1024 * 1024;
   /* Unrolled and geometry-generated index buffers are allocated from this
    * heap. Apple9 draws address index buffers through the USC aperture.
    */
   enum agx_bo_flags flags =
      dev->dev.chip == AGX_CHIP_G16G ? AGX_BO_LOW_VA : 0;
   dev->heap = agx_bo_create(&dev->dev, size, 0, flags, "Geometry heap");

   if (!dev->heap)
      return;

   /* The geometry state buffer is initialized here and then is treated by
    * the CPU as rodata, even though the GPU uses it for scratch internally.
    */
   off_t off = dev->rodata.heap - dev->rodata.bo->va->addr;
   struct poly_heap *map = agx_bo_map(dev->rodata.bo) + off;

   *map = (struct poly_heap){
      .base = dev->heap->va->addr,
      .size = size,
   };
}

uint64_t
hk_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   util_call_once_data(&dev->heap_init_once, hk_init_heap, cmd);
   if (!dev->heap) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return 0;
   }

   /* Acquire the heap before this render's first allocation. The pre-graphics
    * stream waits for earlier work, including the previous heap consumer.
    * Resetting here also covers command-buffer resubmission without a separate
    * post-graphics Work just to release the allocation cursor.
    */
   if (!cmd->uses_heap) {
      perf_debug(cmd, "Resetting heap");
      uint64_t addr = dev->rodata.heap;

      cmd->heap_renders++;
      cmd->uses_heap = true;

      libagx_write_u32(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                       addr + offsetof(struct poly_heap, bottom), 0);
   }

   return dev->rodata.heap;
}

/* GPU-generated index/coordinate offsets use the single geometry heap. */
static struct agx_bo *
hk_geometry_heap_bo(struct hk_cmd_buffer *cmd)
{
   /* Recording can continue after OOM, but the command buffer cannot be
    * submitted. Keep address construction safe on those error paths. */
   if (!hk_heap(cmd))
      return NULL;
   return hk_cmd_buffer_device(cmd)->heap;
}

static struct agx_draw
hk_heap_indexed_indirect(struct hk_cmd_buffer *cmd, uint64_t ptr,
                         enum agx_index_size index_size, bool restart)
{
   struct agx_bo *bo = hk_geometry_heap_bo(cmd);
   return agx_draw_indexed_indirect(ptr, bo ? bo->va->addr : 0,
                                   bo ? bo->size : 0, index_size, restart);
}

static uint64_t
hk_index_buffer(uint64_t index_buffer, uint size_el, uint offset_el,
                uint elsize_B)
{
   if (offset_el < size_el)
      return index_buffer + (offset_el * elsize_B);
   else
      return AGX_ZERO_PAGE_ADDRESS;
}

static uint64_t
hk_upload_vertex_params_for_vs(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                                const struct hk_shader *vs)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_vertex_params params;
   poly_vertex_params_init(&params, 0, wg_size);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) || draw.restart;

   if (!indirect)
      poly_vertex_params_set_draw(&params, draw.b.count[0], draw.b.count[1]);

   if (draw.indexed) {
      unsigned index_size_B = agx_index_size_to_B(draw.index_size);
      unsigned range_el = agx_draw_index_range_el(draw);

      params.index_buffer = hk_index_buffer(agx_draw_index_buffer(draw),
                                            range_el, 0, index_size_B);

      params.index_buffer_range_el = range_el;

      /* Apple9 software vertex shaders read the index size from here rather
       * than from a specialized prolog.
       */
      params.index_size_B = index_size_B;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY]) {

      params.outputs = vs->b.info.outputs;

      if (!indirect) {
         uint32_t verts = draw.b.count[0], instances = draw.b.count[1];
         unsigned vb_size =
            poly_tcs_in_size(verts * instances, vs->b.info.outputs);

         /* Allocate if there are any outputs, or use the null sink to trap
          * reads if there aren't. Those reads are undefined but should not
          * fault. Affects:
          *
          *    dEQP-VK.pipeline.monolithic.no_position.explicit_declarations.basic.single_view.v0_g1
          */
         params.output_buffer = vb_size ? hk_pool_alloc(cmd, vb_size, 4).gpu
                                        : AGX_SCRATCH_PAGE_ADDRESS;
      }
   }

   desc->root.draw.vertex_outputs = params.outputs;

   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static uint64_t
hk_upload_vertex_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   return hk_upload_vertex_params_for_vs(cmd, draw,
                                         hk_bound_sw_vs(&cmd->state.gfx));
}

static enum mesa_prim
hk_gs_in_prim(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];

   if (tes != NULL)
      return gfx->tess.prim;
   else
      return vk_topology_to_mesa(dyn->ia.primitive_topology);
}

static enum mesa_prim
hk_rast_prim(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   if (gs != NULL) {
      return gs->variants[HK_GS_VARIANT_RAST].info.gs.mode;
   } else {
      switch (dyn->ia.primitive_topology) {
      case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_LINES;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_TRIANGLES;
      default:
         return hk_gs_in_prim(cmd);
      }
   }
}

/* Apple9 draws address index buffers through the 32-bit USC aperture. */
static struct agx_ptr
hk_geom_index_alloc(struct hk_cmd_buffer *cmd, uint32_t size)
{
   if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G)
      return hk_pool_usc_alloc(cmd, size, 4);
   else
      return hk_pool_alloc(cmd, size, 4);
}

static uint64_t
hk_upload_geometry_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);
   struct hk_shader *count = hk_count_gs_variant(gs);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) ||
                   gfx->shaders[MESA_SHADER_TESS_EVAL] || draw.restart;
   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      mode = u_decomposed_prim(mode);
   }

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, mode, wg_size);

   params.flat_outputs = fs ? fs->info.fs.interp.flat : 0;
   /* Tessellation precedes the geometry shader and has set up this draw's
    * parameters; the geometry shader recovers patch IDs from them. */
   if (gfx->shaders[MESA_SHADER_TESS_EVAL])
      params.tess_params = gfx->descriptors.root.draw.tess_params;

   if (gfx->xfb_enabled) {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_base_original[i] = gfx->xfb[i].addr;
         params.xfb_size[i] = gfx->xfb[i].range;
         params.xfb_offs_ptrs[i] = gfx->xfb_offsets + i * sizeof(uint32_t);
      }
   } else {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_offs_ptrs[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_base_original[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_size[i] = 0;
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb_query); ++i) {
      uint64_t q = gfx->xfb_query[i];

      if (q) {
         params.xfb_prims_generated_counter[i] = q;
         params.prims_generated_counter[i] = q + sizeof(uint64_t);
      } else {
         params.xfb_prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
         params.prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
      }
      params.pg_query_counter[i] = gfx->pg_query[i];

      /* TODO: Optimize out? */
      params.xfb_overflow[i] = AGX_SCRATCH_PAGE_ADDRESS;
   }

   /* TODO: Optimize out? */
   params.xfb_any_overflow = AGX_SCRATCH_PAGE_ADDRESS;

   /* Calculate input primitive count for direct draws, and allocate the vertex
    * & count buffers. GPU calculates and allocates for indirect draws.
    */
   params.count_buffer_stride = count->info.gs.count_words * 4;

   if (!count->info.gs.prefix_sum && params.count_buffer_stride) {
      struct agx_ptr T = hk_pool_alloc(cmd, 16, 4);
      /* On failure the command buffer is in error and never submitted. */
      if (T.cpu)
         memset(T.cpu, 0, 16);
      params.count_buffer = T.gpu;
   }

   struct poly_gs_info *gsi = &count->info.gs;

   if (indirect) {
      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         struct agx_bo *bo = hk_geometry_heap_bo(cmd);
         cmd->geom_index_buffer = bo ? bo->va->addr : 0;
         cmd->geom_index_count = bo ? bo->size : 0;
      } else {
         cmd->geom_index_count =
            poly_gs_rast_vertices(gsi->shape, gsi->max_indices, 1, 0);
      }
   } else {
      poly_geometry_params_set_draw(&params, mode, gsi->shape, gsi->max_indices,
                                    draw.b.count[0], draw.b.count[1]);

      unsigned size = params.input_primitives * params.count_buffer_stride;
      if (count->info.gs.prefix_sum && size) {
         params.count_buffer = hk_pool_alloc(cmd, size, 4).gpu;
      }

      cmd->geom_index_count = params.draw.index_count;
      cmd->geom_instance_count = params.draw.instance_count;

      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         params.output_index_buffer =
            hk_geom_index_alloc(cmd, cmd->geom_index_count * 4).gpu;

         cmd->geom_index_buffer = params.output_index_buffer;
      }
   }

   if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      struct agx_ptr topology = hk_geom_index_alloc(cmd, gsi->max_indices * 4);
      if (topology.cpu)
         memcpy(topology.cpu, gsi->topology,
                MIN2(gsi->max_indices * 4, sizeof(gsi->topology)));
      cmd->geom_index_buffer = topology.gpu;
   }

   desc->root_dirty = true;
   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static void
hk_upload_tess_params_for_tcs(struct hk_cmd_buffer *cmd,
                              struct poly_tess_params *out,
                              struct agx_draw draw, struct hk_shader *tcs)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   enum poly_tess_partitioning partitioning =
      gfx->tess.info.spacing == TESS_SPACING_EQUAL
         ? POLY_TESS_PARTITIONING_INTEGER
      : gfx->tess.info.spacing == TESS_SPACING_FRACTIONAL_ODD
         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;

   struct poly_tess_params args = {
      .heap = hk_heap(cmd),
      .tcs_stride_el = tcs->info.tess.tcs_output_stride / 4,
      .statistic = hk_pipeline_stat_addr(
         cmd,
         VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT),

      .input_patch_size = dyn->ts.patch_control_points,
      .output_patch_size = tcs->info.tess.tcs_output_patch_size,
      .tcs_patch_constants = tcs->info.tess.tcs_nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->info.tess.tcs_per_vertex_outputs,
      .partitioning = partitioning,
      .points_mode = gfx->tess.info.points,
      .isolines = gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES,
   };

   if (!args.points_mode && gfx->tess.info.mode != TESS_PRIMITIVE_ISOLINES) {
      args.ccw = gfx->tess.info.ccw;
      args.ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }

   uint32_t draw_stride_el = 5;
   size_t draw_stride_B = draw_stride_el * sizeof(uint32_t);

   /* heap is allocated by hk_heap */
   struct agx_bo *bo = hk_geometry_heap_bo(cmd);
   args.patch_coord_buffer = bo ? bo->va->addr : 0;

   if (!agx_is_indirect(draw.b)) {
      unsigned in_patches = draw.b.count[0] / args.input_patch_size;
      unsigned unrolled_patches = in_patches * draw.b.count[1];

      uint32_t alloc = 0;
      uint32_t tcs_out_offs = alloc;
      alloc += unrolled_patches * args.tcs_stride_el * sizeof(uint32_t);

      uint32_t patch_coord_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      uint32_t count_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      /* Single API draw */
      uint32_t draw_offs = alloc;
      alloc += draw_stride_B;

      struct agx_ptr blob = hk_pool_alloc(cmd, alloc, 4);
      args.tcs_buffer = blob.gpu + tcs_out_offs;
      args.patches_per_instance = in_patches;
      args.coord_allocs = blob.gpu + patch_coord_offs;
      args.nr_patches = unrolled_patches;
      args.out_draws = blob.gpu + draw_offs;
      args.counts = blob.gpu + count_offs;
   } else {
      /* Allocate 3x indirect global+local grids for VS/TCS/tess */
      uint32_t grid_stride = sizeof(uint32_t) * 6;
      gfx->tess.grids = hk_pool_alloc(cmd, grid_stride * 3, 4).gpu;

      args.out_draws = hk_pool_alloc(cmd, draw_stride_B, 4).gpu;
   }

   gfx->tess.out_draws = args.out_draws;
   memcpy(out, &args, sizeof(args));
}

static void
hk_upload_tess_params(struct hk_cmd_buffer *cmd, struct poly_tess_params *out,
                      struct agx_draw draw)
{
   hk_upload_tess_params_for_tcs(
      cmd, out, draw,
      hk_only_variant(cmd->state.gfx.shaders[MESA_SHADER_TESS_CTRL]));
}

/* The Apple9 passthrough geometry shader key is the common key followed by
 * the fragment shader's {flat, linear} input masks, which shape the
 * rasterized varying layout, and the per-slot component masks of the
 * fragment shader's compacted input layout.
 */
struct hk_apple9_passthrough_suffix {
   uint64_t fs_interp[2];
   uint8_t components[64];
   /* Flat copies of per-vertex inputs, as in the fragment shader's info. */
   uint8_t per_vertex_copies;
   uint8_t per_vertex_location[HK_MAX_PER_VERTEX_COPIES];
   uint8_t per_vertex_copy_base[HK_MAX_PER_VERTEX_COPIES];
   uint8_t per_vertex_bary[2];
   /* Line stipple start slot to continue along strips, and whether the
    * input is a strip. */
   uint8_t line_stipple_start;
   uint8_t line_stipple_strip;
   /* Software polygon points (hk_apple9_polygon_point_gs), and the varying
    * slot of the polygon's facing when the fragment shader reads it. */
   uint8_t polygon_points;
   uint8_t polygon_face_slot;
   /* polygonModePointSize: forward the shader's PointSize, else size 1. */
   uint8_t polygon_point_size;
   /* The vertex stage publishes a flat cull mask, which the fragment shader
    * reads: combine it over each primitive. Otherwise cull planes stay
    * interpolated values. See hk_apple9_compile_vertex_culls. */
   uint8_t cull_mask;
};

/* Before every emitted vertex, write the per-vertex values of the input
 * primitive's API vertices 0, 1 and 2 to their flat copies, and the emitted
 * API vertex's one-hot barycentric weights. Points repeat their vertex and
 * lines their second vertex. The passthrough GS emits API vertex i as its
 * i-th vertex.
 */

static void
hk_apple9_write_per_vertex_copies(nir_shader *s,
                                  const struct hk_apple9_passthrough_suffix *suffix)
{
   enum mesa_prim prim = s->info.gs.input_primitive;
   unsigned start = prim == MESA_PRIM_LINES_ADJACENCY ? 1 : 0;
   unsigned step = prim == MESA_PRIM_TRIANGLES_ADJACENCY ? 2 : 1;
   unsigned vertices = mesa_vertices_per_prim(u_decomposed_prim(u_reduced_prim(prim)));
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_create(impl);
   unsigned emitted = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_emit_vertex)
            continue;
         b.cursor = nir_before_instr(instr);
         for (unsigned m = 0; m < 2; ++m) {
            if (!suffix->per_vertex_bary[m])
               continue;
            unsigned slot = VARYING_SLOT_VAR0 + suffix->per_vertex_bary[m] - 1;
            for (unsigned c = 0; c < 3; ++c) {
               nir_store_output(&b, nir_imm_float(&b, c == emitted ? 1.0 : 0.0),
                                nir_imm_int(&b, 0),
                                .io_semantics = {.location = slot, .num_slots = 1},
                                .component = c, .src_type = nir_type_float32,
                                .write_mask = 1);
            }
            s->info.outputs_written |= BITFIELD64_BIT(slot);
         }
         ++emitted;
         for (unsigned j = 0; j < suffix->per_vertex_copies; ++j) {
            for (unsigned k = 0; k < 3; ++k) {
               unsigned api_vertex = MIN2(k, vertices - 1);
               nir_def *value = nir_load_per_vertex_input(
                  &b, 4, 32, nir_imm_int(&b, start + api_vertex * step),
                  nir_imm_int(&b, 0),
                  .io_semantics = {
                     .location = VARYING_SLOT_VAR0 + suffix->per_vertex_location[j],
                     .num_slots = 1,
                  },
                  .dest_type = nir_type_uint32);
               unsigned copy = VARYING_SLOT_VAR0 + suffix->per_vertex_copy_base[j] + k;
               for (unsigned c = 0; c < 4; ++c) {
                  nir_store_output(&b, nir_channel(&b, value, c), nir_imm_int(&b, 0),
                                   .io_semantics = {.location = copy, .num_slots = 1},
                                   .component = c, .src_type = nir_type_uint32,
                                   .write_mask = 1);
               }
               s->info.outputs_written |= BITFIELD64_BIT(copy);
            }
         }
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
}

/*
 * Line stipple the vertex shader's start cannot express alone. The vertex
 * shader exports each vertex's snapped window position in the start slot,
 * with the counter there, which the fragment shader measures its counter
 * from, see hk_apple9_lower_line_stipple_fs. Every emitted vertex of the
 * segment gets the segment's first vertex, so either vertex may provoke. On
 * a strip the counter continues: it starts at the major-axis lengths of the
 * strip's earlier segments. The lengths approximate the earlier segments'
 * fragment counts to within one per segment, as the CTS allows; counting
 * rounded endpoints instead failed its strip cases. Strips longer than the
 * bound restart at each segment. With adjacency, the line is input vertices 1 and 2, and strip
 * vertex j + 1 begins segment j.
 */
static void
hk_apple9_line_stipple_gs(nir_shader *s,
                          const struct hk_apple9_passthrough_suffix *suffix)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_io_semantics sem = {.location = suffix->line_stipple_start,
                           .num_slots = 1};
   unsigned first = s->info.gs.input_primitive == MESA_PRIM_LINES_ADJACENCY;
   nir_def *start = nir_load_per_vertex_input(
      &b, 2, 32, nir_imm_int(&b, first), nir_imm_int(&b, 0),
      .io_semantics = sem, .dest_type = nir_type_float32);
   nir_def *base = nir_imm_float(&b, 0.0);

   if (suffix->line_stipple_strip) {
      nir_variable *i = nir_local_variable_create(impl, glsl_uint_type(), NULL);
      nir_variable *sum = nir_local_variable_create(impl, glsl_float_type(), NULL);
      nir_store_var(&b, i, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sum, nir_imm_float(&b, 0.0), 1);
      nir_def *prim = nir_load_primitive_id(&b);
      nir_def *limit =
         nir_bcsel(&b, nir_ult_imm(&b, prim, AGX_LINE_STIPPLE_MAX_STRIP_SEGMENTS),
                   prim, nir_imm_int(&b, 0));
      nir_push_loop(&b);
      {
         nir_def *j = nir_load_var(&b, i);
         nir_break_if(&b, nir_uge(&b, j, limit));
         nir_def *a = nir_load_strip_output_poly(
            &b, 2, nir_iadd_imm(&b, j, first), .io_semantics = sem);
         nir_def *c = nir_load_strip_output_poly(
            &b, 2, nir_iadd_imm(&b, j, first + 1), .io_semantics = sem);
         nir_def *d = nir_fabs(&b, nir_fsub(&b, c, a));
         nir_store_var(&b, sum,
                       nir_fadd(&b, nir_load_var(&b, sum),
                                nir_fmax(&b, nir_channel(&b, d, 0),
                                         nir_channel(&b, d, 1))),
                       1);
         nir_store_var(&b, i, nir_iadd_imm(&b, j, 1), 1);
      }
      nir_pop_loop(&b, NULL);
      base = nir_load_var(&b, sum);
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_emit_vertex)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_store_output(&b, nir_vec3(&b, nir_channel(&b, start, 0),
                                       nir_channel(&b, start, 1), base),
                          nir_imm_int(&b, 0), .io_semantics = sem,
                          .write_mask = 0x7, .src_type = nir_type_float32);
      }
   }
   nir_progress(true, impl, nir_metadata_none);
}

static bool
hk_apple9_trim_passthrough_output(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data)
{
   const uint8_t *components = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned slot = nir_intrinsic_io_semantics(intr).location;
   if (slot < VARYING_SLOT_VAR0 || slot >= 64 ||
       (components[slot] & BITFIELD_BIT(nir_intrinsic_component(intr))))
      return false;
   /* The fragment shader was linked against a producer that did not write
    * this component, so the rasterized layout must not contain it. */
   nir_instr_remove(&intr->instr);
   return true;
}

struct hk_pp_gs_load {
   unsigned step;
};

static nir_def *
hk_pp_gs_load(nir_builder *b, void *data, unsigned vertex, unsigned location,
              unsigned offset, unsigned component, unsigned comps,
              nir_alu_type type)
{
   const struct hk_pp_gs_load *l = data;
   unsigned num_slots = 1;
   if (location == VARYING_SLOT_CLIP_DIST0)
      num_slots = b->shader->info.clip_distance_array_size +
                  b->shader->info.cull_distance_array_size;
   return nir_load_per_vertex_input(
      b, comps, 32, nir_imm_int(b, vertex * l->step), nir_imm_int(b, offset),
      .component = component, .dest_type = type,
      .io_semantics = {.location = location, .num_slots = num_slots});
}

/*
 * Polygon mode POINT for vertex and evaluation shader pipelines: a generated
 * geometry shader draws each input triangle's clipped vertices as points,
 * see hk_apple9_pp_emit_triangle.
 */
static void
hk_apple9_polygon_point_gs(nir_builder *b,
                           const struct poly_passthrough_gs_key *key,
                           const struct hk_apple9_passthrough_suffix *suffix)
{
   nir_shader *s = b->shader;
   assert(u_reduced_prim(key->prim) == MESA_PRIM_TRIANGLES);
   unsigned clip = key->clip_distance_array_size;
   unsigned cull = key->cull_distance_array_size;
   /* The points keep the vertex stage's varying layout, which the fragment
    * shader is linked against: clip and cull distances stay published, as
    * inside values, and the primitive ID is not published. */
   uint64_t outputs = key->outputs &
      ~(VARYING_BIT_BOUNDING_BOX0 | VARYING_BIT_BOUNDING_BOX1 |
        VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 |
        VARYING_BIT_CULL_DIST0 | VARYING_BIT_CULL_DIST1 |
        VARYING_BIT_PRIMITIVE_ID);
   /* Without polygonModePointSize, polygon points have size 1: the point
    * path's default. */
   if (!suffix->polygon_point_size)
      outputs &= ~VARYING_BIT_PSIZ;

   s->info.stage = MESA_SHADER_GEOMETRY;
   s->info.inputs_read = key->outputs;
   s->info.outputs_written = key->outputs &
      ~(VARYING_BIT_BOUNDING_BOX0 | VARYING_BIT_BOUNDING_BOX1 |
        VARYING_BIT_PRIMITIVE_ID);
   if (!suffix->polygon_point_size)
      s->info.outputs_written &= ~VARYING_BIT_PSIZ;
   if (suffix->polygon_face_slot)
      s->info.outputs_written |= BITFIELD64_BIT(suffix->polygon_face_slot);
   s->info.clip_distance_array_size = clip;
   s->info.cull_distance_array_size = cull;
   s->info.gs.input_primitive = key->prim;
   s->info.gs.output_primitive = MESA_PRIM_POINTS;
   s->info.gs.vertices_in = mesa_vertices_per_prim(key->prim);
   s->info.gs.vertices_out = HK_PP_MAX_VERTICES;
   s->info.gs.invocations = 1;
   s->info.gs.active_stream_mask = 1;

   /* Adjacency triangles use input vertices 0, 2 and 4. */
   struct hk_pp_gs_load l = {
      .step = key->prim == MESA_PRIM_TRIANGLES_ADJACENCY ? 2 : 1,
   };
   struct hk_pp_info info = {
      .clip = clip,
      .cull = cull,
      .outputs = outputs,
      .output_components = key->output_components,
      .flat = suffix->fs_interp[0],
      .face_slot = suffix->polygon_face_slot,
   };
   nir_def *last = nir_ine_imm(b, nir_load_provoking_last(b), 0);
   nir_def *provoking = nir_bcsel(b, last, nir_imm_int(b, 2), nir_imm_int(b, 0));
   hk_apple9_pp_emit_triangle(b, &info, hk_pp_gs_load, &l, provoking,
                              nir_load_primitive_id(b), NULL);
}

static void
hk_apple9_passthrough_gs(struct nir_builder *b, const void *key)
{
   {
      const struct poly_passthrough_gs_key *common = key;
      struct hk_apple9_passthrough_suffix suffix;
      memcpy(&suffix, (const uint8_t *)key +
                         poly_passthrough_gs_key_size(common->xfb_info.output_count),
             sizeof(suffix));
      if (suffix.polygon_points) {
         hk_apple9_polygon_point_gs(b, common, &suffix);
         nir_shader_intrinsics_pass(b->shader, hk_apple9_trim_passthrough_output,
                                    nir_metadata_control_flow, suffix.components);
         return;
      }
   }
   poly_nir_passthrough_gs(b, key);
   const struct poly_passthrough_gs_key *common = key;
   size_t key_size =
      poly_passthrough_gs_key_size(common->xfb_info.output_count);
   struct hk_apple9_passthrough_suffix suffix;
   memcpy(&suffix, (const uint8_t *)key + key_size, sizeof(suffix));
   if (suffix.per_vertex_copies || suffix.per_vertex_bary[0] ||
       suffix.per_vertex_bary[1])
      hk_apple9_write_per_vertex_copies(b->shader, &suffix);
   if (suffix.line_stipple_start)
      hk_apple9_line_stipple_gs(b->shader, &suffix);
   nir_shader_intrinsics_pass(b->shader, hk_apple9_trim_passthrough_output,
                              nir_metadata_control_flow, suffix.components);
}

static struct hk_api_shader *
hk_build_meta_shader_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->kernels.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, NULL);
   builder(&b, key->key);

   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };

   struct vk_shader_compile_info info = {
      .stage = b.shader->info.stage,
      .nir = b.shader,
      .robustness = &rs,
   };

   hk_preprocess_nir_internal(dev->vk.physical, b.shader);

   struct hk_api_shader *s;
   if (builder == hk_apple9_passthrough_gs) {
      struct hk_apple9_passthrough_suffix suffix;
      memcpy(&suffix, (uint8_t *)key->key + key->key_size - sizeof(suffix),
             sizeof(suffix));
      if (hk_apple9_compile_internal_gs(dev, b.shader, suffix.fs_interp,
                                        suffix.cull_mask, &s) != VK_SUCCESS)
         return NULL;
   } else if (hk_compile_shader(dev, &info, NULL, NULL, NULL, &s) !=
              VK_SUCCESS) {
      return NULL;
   }

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->kernels.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->kernels.ht, cloned_key, s);
   return s;
}

struct hk_api_shader *
hk_meta_shader(struct hk_device *dev, hk_internal_builder_t builder, void *data,
               size_t data_size)
{
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->kernels.lock);
   struct hk_api_shader *s = hk_build_meta_shader_locked(dev, key, builder);
   simple_mtx_unlock(&dev->kernels.lock);

   return s;
}

static struct agx_draw
hk_draw_as_indexed_indirect(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   assert(draw.indexed);

   if (agx_is_indirect(draw.b))
      return draw;

   VkDrawIndexedIndirectCommand desc = {
      .indexCount = draw.b.count[0],
      .instanceCount = draw.b.count[1],
      .firstIndex = draw.start,
      .vertexOffset = draw.index_bias,
      .firstInstance = draw.start_instance,
   };

   return agx_draw_indexed_indirect(
      hk_pool_upload(cmd, &desc, sizeof(desc), 4), draw.index_buffer,
      draw.index_buffer_range_B, draw.index_size, draw.restart);
}

static struct agx_draw
hk_draw_without_restart(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                        uint32_t draw_count)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   perf_debug(cmd, "Unrolling primitive restart due to GS/XFB");

   /* The unroll kernel assumes an indirect draw. Synthesize one if needed */
   draw = hk_draw_as_indexed_indirect(cmd, draw);

   /* Next, we unroll the index buffer used by the indirect draw */
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   assert(draw_count == 1 && "TODO: multidraw");

   struct libagx_unroll_restart_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.index_buffer,
      .in_draw = draw.b.ptr,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t) * draw_count, 4).gpu,
      .max_draws = 1 /* TODO: MDI */,
      .restart_index = gfx->index.restart,
      .index_buffer_size_el = agx_draw_index_range_el(draw),
      .index_size_log2 = draw.index_size,
      .flatshade_first =
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
   };

   libagx_unroll_restart_struct(cmd, agx_1d(1024 * draw_count),
                                AGX_BARRIER_ALL | AGX_PREGFX, ia,
                                poly_compact_prim(prim));

   return hk_heap_indexed_indirect(cmd, ia.out_draw, draw.index_size,
                                   false /* restart */);
}

/* Native Apple9 fans flat shade consistently only from their first vertex.
 * Draw fans with the last provoking vertex as an unrolled triangle list. */
static struct agx_draw
hk_apple9_unroll_fan(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   perf_debug(cmd, "Unrolling a last-provoking triangle fan");

   uint64_t in_draw;
   if (draw.indexed) {
      draw = hk_draw_as_indexed_indirect(cmd, draw);
      in_draw = draw.b.ptr;
   } else if (agx_is_indirect(draw.b)) {
      in_draw = draw.b.ptr;
   } else {
      VkDrawIndirectCommand desc = {
         .vertexCount = draw.b.count[0],
         .instanceCount = draw.b.count[1],
         .firstVertex = draw.start,
         .firstInstance = draw.start_instance,
      };
      in_draw = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   struct libagx_unroll_geometry_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.indexed ? draw.index_buffer : 0,
      .in_draw = in_draw,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu,
      .restart_index = draw.restart ? gfx->index.restart : UINT32_MAX,
      .index_buffer_size_el = draw.indexed ? agx_draw_index_range_el(draw) : 0,
      .in_index_size_B = draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
      .flatshade_first = false,
   };

   libagx_unroll_geometry_struct(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                                 ia, poly_compact_prim(MESA_PRIM_TRIANGLE_FAN));

   return hk_heap_indexed_indirect(cmd, ia.out_draw, AGX_INDEX_SIZE_U32,
                                   false /* restart */);
}

static struct agx_draw
hk_launch_gs_prerast(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct agx_grid grid_vs, grid_gs;

   struct hk_shader *vs = hk_bound_sw_vs_before_gs(gfx);
   struct hk_shader *main = hk_main_gs_variant(gs);
   struct hk_shader *count = hk_count_gs_variant(gs);
   struct hk_shader *pre_gs = hk_pre_gs_variant(gs);

   uint64_t vertex_params = desc->root.draw.vertex_params;
   uint64_t geometry_params = desc->root.draw.geometry_params;
   unsigned count_words = count->info.gs.count_words;
   struct agx_workgroup wg = agx_workgroup(64, 1, 1);

   if (false /* TODO */)
      perf_debug(cmd, "Transform feedbck");
   else if (count_words)
      perf_debug(cmd, "Geometry shader with counts");
   else
      perf_debug(cmd, "Geometry shader without counts");

   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      mode = u_decomposed_prim(mode);
   }

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      struct libagx_gs_setup_indirect_args gsi = {
         .heap = hk_heap(cmd),
         .index_buffer = draw.index_buffer,
         .draw = draw.b.ptr,
         .vp = desc->root.draw.vertex_params,
         .p = desc->root.draw.geometry_params,
         .vs_outputs = vs->b.info.outputs,
         .prim = mode,
         .is_prefix_summing = count->info.gs.prefix_sum,
         .max_indices = count->info.gs.max_indices,
         .shape = count->info.gs.shape,
      };

      if (draw.indexed) {
         gsi.index_size_B = agx_index_size_to_B(draw.index_size);
         gsi.index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_gs_setup_indirect_struct(cmd, agx_1d(1),
                                      AGX_BARRIER_ALL | AGX_PREGFX, gsi);

      grid_vs = agx_grid_indirect_local(
         vertex_params + offsetof(struct poly_vertex_params, grid));

      grid_gs = agx_grid_indirect_local(
         geometry_params + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = grid_gs = draw.b;
      grid_gs.count[0] = u_decomposed_prims_for_vertices(mode, draw.b.count[0]);
   }

   /* Launch the vertex shader first */
   hk_reserve_scratch(cmd, cs, vs);
   hk_dispatch_with_usc(dev, cs, &vs->b.info,
                        hk_upload_usc_words(cmd, vs,
                                            vs->info.stage == MESA_SHADER_VERTEX
                                               ? gfx->linked[MESA_SHADER_VERTEX]
                                               : vs->only_linked),
                        grid_vs, wg);

   /* Transform feedback and various queries require extra dispatching,
    * determine if we need that here.
    */
   VkQueryPipelineStatisticFlagBits gs_queries =
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;

   struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   bool xfb_or_queries =
      main->info.gs.xfb || (root->draw.pipeline_stats_flags & gs_queries);

   if (xfb_or_queries) {
      /* If we need counts, launch the count shader and prefix sum the results. */
      if (count_words) {
         perf_debug(dev, "Geometry shader count");
         hk_dispatch_with_local_size(cmd, cs, count, grid_gs, wg);
      }

      if (count->info.gs.prefix_sum) {
         perf_debug(dev, "Geometry shader transform feedback prefix sum");
         libagx_prefix_sum_geom(cmd, agx_1d(1024 * count_words),
                                AGX_BARRIER_ALL | AGX_PREGFX, geometry_params);
      }

      /* Transform feedback / query program */
      perf_debug(dev, "Transform feedback / geometry query");
      hk_dispatch_with_local_size(cmd, cs, pre_gs, agx_1d(1),
                                  agx_workgroup(1, 1, 1));
   }

   /* Pre-rast geometry shader */
   hk_dispatch_with_local_size(cmd, cs, main, grid_gs, wg);

   if (poly_gs_indexed(count->info.gs.shape)) {
      enum agx_index_size index_size =
         agx_translate_index_size(poly_gs_index_size(count->info.gs.shape));

      if (agx_is_indirect(draw.b)) {
         return agx_draw_indexed_indirect(
            geometry_params + offsetof(struct poly_geometry_params, draw),
            cmd->geom_index_buffer, cmd->geom_index_count, index_size, true);
      } else {
         return agx_draw_indexed(cmd->geom_index_count,
                                 cmd->geom_instance_count, 0, 0, 0,
                                 cmd->geom_index_buffer,
                                 cmd->geom_index_count * 4, index_size, true);
      }
   } else {
      if (agx_is_indirect(draw.b)) {
         return agx_draw_indirect(geometry_params +
                                  offsetof(struct poly_geometry_params, draw));
      } else {
         return (struct agx_draw){
            .b = agx_3d(cmd->geom_index_count, cmd->geom_instance_count, 1),
         };
      }
   }
}

static void
hk_apple9_dispatch_prerast(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                           struct agx_grid grid,
                           const struct hk_apple9_prerast_bindings *bindings)
{
   if (agx_is_shader_empty(&s->b))
      return;
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   struct hk_cs *cs =
      hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true, HK_SYNC_ALL);
   if (!cs)
      return;

   hk_reserve_scratch(cmd, cs, s);
   hk_apple9_dispatch(cmd, cs, s->apple9_body, &s->info.apple9, grid,
                      cmd->state.gfx.root, s, bindings);
}

/* Only geometry shaders observe adjacent vertices. Without one, draw the
 * primitive list of the remaining vertices through a generated index buffer.
 */
static struct agx_draw
hk_apple9_draw_without_adj(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      prim = u_decomposed_prim(prim);
   }

   uint64_t in_draw = draw.b.ptr;
   if (!agx_is_indirect(draw.b)) {
      uint32_t desc[5] = {draw.b.count[0], draw.b.count[1]};
      if (draw.indexed) {
         desc[2] = draw.start;
         desc[3] = draw.index_bias;
         desc[4] = draw.start_instance;
      } else {
         desc[2] = draw.start;
         desc[3] = draw.start_instance;
      }
      in_draw = hk_pool_upload(cmd, desc, sizeof(desc), 4);
   }

   uint64_t out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu;
   uint64_t heap = hk_heap(cmd);
   libagx_unroll_adjacency(
      cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw, heap, in_draw,
      draw.indexed ? draw.index_buffer : 0,
      draw.indexed ? draw.index_buffer_range_B >> draw.index_size : 0,
      draw.indexed ? agx_index_size_to_B(draw.index_size) : 0, prim,
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT);

   return hk_heap_indexed_indirect(cmd, out_draw, AGX_INDEX_SIZE_U32, false);
}

/* The Apple9 index fetch reads only from the USC aperture and is bounded in
 * whole words. Fetch from a copy for index buffers placed outside the
 * aperture, and when robustness requires zeroes past a binding that ends
 * inside a word.
 */
static bool
hk_apple9_needs_index_copy(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!draw.indexed || draw.index_buffer_range_B == 0)
      return false;

   /* Check the whole binding, not just its base: a direct draw's firstIndex
    * (or an indirect draw's GPU-provided firstIndex) may otherwise move the
    * hardware pointer past the end of the USC aperture. Subtraction avoids
    * wrapping the end of a high-VA binding.
    */
   if (draw.index_buffer < dev->dev.shader_base ||
       draw.index_buffer - dev->dev.shader_base >= (1ull << 32) ||
       draw.index_buffer_range_B >
          (1ull << 32) - (draw.index_buffer - dev->dev.shader_base))
      return true;

   if (!((draw.index_buffer + draw.index_buffer_range_B) & 3))
      return false;

   if (!(dev->vk.enabled_features.robustBufferAccess ||
         dev->vk.enabled_features.robustBufferAccess2 ||
         dev->vk.enabled_features.pipelineRobustness))
      return false;

   return agx_is_indirect(draw.b) || agx_direct_draw_overreads_indices(draw);
}

static struct agx_draw
hk_apple9_robust_index(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   uint64_t in_draw = draw.b.ptr;
   if (!agx_is_indirect(draw.b)) {
      uint32_t desc[5] = {draw.b.count[0], draw.b.count[1], draw.start,
                          draw.index_bias, draw.start_instance};
      in_draw = hk_pool_upload(cmd, desc, sizeof(desc), 4);
   }

   perf_debug(cmd, "copying robust index buffer");
   uint64_t out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4).gpu;
   libagx_robust_index_apple9(
      cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw, hk_heap(cmd),
      in_draw, draw.index_buffer,
      draw.index_buffer_range_B >> draw.index_size,
      agx_index_size_to_B(draw.index_size));

   return hk_heap_indexed_indirect(cmd, out_draw, draw.index_size,
                                   draw.restart);
}

static void
hk_apple9_merge_tess_info(struct hk_cmd_buffer *cmd, bool native)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tcs_api = gfx->shaders[MESA_SHADER_TESS_CTRL];
   struct hk_api_shader *tes_api = gfx->shaders[MESA_SHADER_TESS_EVAL];
   if (native) {
      tcs_api = hk_api_shader_native(tcs_api);
      tes_api = hk_api_shader_native(tes_api);
   }
   struct hk_shader *tcs = hk_only_variant(tcs_api);
   struct hk_tess_info tes_info = native
                                     ? tes_api->apple9_native_tess
                                     : hk_any_variant(tes_api)->info.tess.info;
   gfx->tess.info = hk_tess_info_merge(tes_info, tcs->info.tess.info);
   if (gfx->tess.info.points)
      gfx->tess.prim = MESA_PRIM_POINTS;
   else if (gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES)
      gfx->tess.prim = MESA_PRIM_LINES;
   else
      gfx->tess.prim = MESA_PRIM_TRIANGLES;
}

/* Run the vertex and tessellation control shaders and the tessellator as
 * compute before the render pass, returning the draw of the tessellated
 * domain.
 */
static struct agx_draw
hk_apple9_launch_tess(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                      bool native)
{
   /* Largest direct draw, in patches x instances, that is batched. */
   const unsigned max_batch_patches = 2048;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *tcs_api = gfx->shaders[MESA_SHADER_TESS_CTRL];
   struct hk_api_shader *vs_api = gfx->shaders[MESA_SHADER_VERTEX];
   if (native) {
      tcs_api = hk_api_shader_native(tcs_api);
      vs_api = hk_api_shader_native(vs_api);
   }
   struct hk_shader *vs = &vs_api->variants[HK_VS_VARIANT_SW];
   struct hk_shader *tcs = hk_only_variant(tcs_api);

   struct hk_apple9_prerast_bindings bindings =
      hk_apple9_draw_bindings(cmd, draw);

   desc->root.draw.vertex_params =
      native ? hk_upload_vertex_params_for_vs(cmd, draw, vs)
             : hk_upload_vertex_params(cmd, draw);
   struct agx_ptr tess_args =
      hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
   if (!tess_args.cpu)
      return draw;
   desc->root.draw.tess_params = tess_args.gpu;
   hk_upload_tess_params_for_tcs(cmd, tess_args.cpu, draw, tcs);
   desc->root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

   uint32_t input_patch_size = dyn->ts.patch_control_points;
   uint64_t state = desc->root.draw.tess_params;
   struct agx_grid grid_vs, grid_tcs, grid_tess;

   if (agx_is_indirect(draw.b)) {
      struct libagx_tess_setup_indirect_args args = {
         .p = state,
         .grids = gfx->tess.grids,
         .indirect = draw.b.ptr,
         .vp = desc->root.draw.vertex_params,
         .vertex_outputs = vs->b.info.outputs,
         .tcs_statistic = hk_pipeline_stat_addr(
            cmd,
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT),
      };

      if (draw.indexed) {
         args.in_index_buffer = draw.index_buffer;
         args.in_index_size_B = agx_index_size_to_B(draw.index_size);
         args.in_index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_tess_setup_indirect_struct(cmd, agx_1d(1),
                                        AGX_BARRIER_ALL | AGX_PREGFX, args);

      uint32_t grid_stride = sizeof(uint32_t) * 6;
      grid_vs = agx_grid_indirect_local(gfx->tess.grids + 0 * grid_stride);
      grid_tcs = agx_grid_indirect_local(gfx->tess.grids + 1 * grid_stride);
      grid_tess = agx_grid_indirect_local(gfx->tess.grids + 2 * grid_stride);
   } else {
      uint32_t patches = draw.b.count[0] / input_patch_size;
      grid_vs = grid_tcs = draw.b;
      grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
      grid_tess = agx_1d(patches * draw.b.count[1]);

      /* TCS invocation counter increments once per-patch */
      uint64_t tcs_stat = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT);
      if (hk_stat_enabled(tcs_stat)) {
         libagx_increment_statistic(cmd, agx_1d(1),
                                    AGX_BARRIER_ALL | AGX_PREGFX, tcs_stat,
                                    patches);
      }
   }

   /* Without a geometry shader, the tessellated primitives are the clipper
    * input.
    */
   uint64_t c_prims = 0, c_inv = 0;
   if (!gfx->shaders[MESA_SHADER_GEOMETRY]) {
      c_prims = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);
      c_inv = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);
   }

   /* A direct draw without statistics or transform feedback touches only
    * its own allocations here, so its passes can run alongside those of the
    * neighbouring tessellated draws: record them into the stage lanes (see
    * hk_tess_batch_flush). The stage order within each draw is kept by the
    * barrier after every lane.
    */
   /* HK_APPLE9_TESS_BATCH=0 turns batching off. */
   static int batch_env = -1;
   if (batch_env < 0) {
      const char *env = debug_get_option_cached_hk_apple9_tess_batch();
      batch_env = !env || strcmp(env, "0") != 0;
   }
   /* Only small draws are latency-bound enough to gain from overlapping;
    * a large one fills the GPU on its own and is kept inline, which also
    * keeps its passes out of the stream-end lanes.
    */
   unsigned batch_patches =
      agx_is_indirect(draw.b) ? 0
                              : (draw.b.count[0] / input_patch_size) *
                                   draw.b.count[1];
   /* The native object reads this draw's count. Keep all five producer passes
    * inline in pre_gfx, after flushing older lanes, so no later batch flush can
    * move the count write behind its consumer. Each pass has HK_SYNC_ALL.
    * Batched native renders record them into the lanes too:
    * hk_cmd_buffer_end_graphics flushes the lanes into the ending render's
    * own pre_gfx stream, so they still complete before its tiling. */
   bool batch =
      batch_env && (!native || hk_apple9_native_lanes(hk_cmd_buffer_device(cmd))) &&
      !agx_is_indirect(draw.b) &&
      batch_patches <= max_batch_patches &&
      !hk_stat_enabled(hk_pipeline_stat_addr(
         cmd,
         VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT)) &&
      !hk_stat_enabled(c_prims) && !hk_stat_enabled(c_inv) && !gfx->xfb_enabled;
   if (!batch)
      hk_tess_batch_flush(cmd);

#define HK_TESS_LANE(l) cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[l] : NULL

   HK_TESS_LANE(0);
   hk_apple9_dispatch_prerast(cmd, vs, grid_vs, &bindings);
   HK_TESS_LANE(1);
   hk_apple9_dispatch_prerast(cmd, tcs, grid_tcs, &bindings);

   /* First generate counts, then prefix sum them, and then tessellate. */
   HK_TESS_LANE(2);
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX,
                     gfx->tess.info.mode, POLY_TESS_MODE_COUNT, state);
   HK_TESS_LANE(3);
   /* One workgroup walking every patch takes milliseconds for large draws,
    * so direct draws past a few blocks scan blocks in parallel instead.
    */
   /* HK_APPLE9_TESS_SCAN_BLOCK=<n> sets the block size, so that tests with
    * small draws reach the blocked path too.
    */
   static unsigned block_patches = 0;
   if (!block_patches) {
      const char *env = debug_get_option_cached_hk_apple9_tess_scan_block();
      block_patches = env ? MAX2(strtoul(env, NULL, 0), 1) : 8192;
   }
   unsigned nr_patches =
      agx_is_indirect(draw.b) ? 0
                              : ((struct poly_tess_params *)tess_args.cpu)->nr_patches;
   /* The three passes depend on each other, so they cannot share a stage
    * lane with other draws' passes: batched draws keep the single scan.
    */
   if (!batch && nr_patches > 2 * block_patches) {
      uint64_t counts = ((struct poly_tess_params *)tess_args.cpu)->counts;
      unsigned nr_blocks = DIV_ROUND_UP(nr_patches, block_patches);
      uint64_t block_sums = hk_pool_alloc(cmd, nr_blocks * 4, 4).gpu;
      libagx_prefix_sum_tess_blocks(cmd, agx_1d(1024 * nr_blocks),
                                    AGX_BARRIER_ALL | AGX_PREGFX, counts,
                                    block_sums, nr_patches, block_patches);
      libagx_prefix_sum_tess_offsets(cmd, agx_1d(1024),
                                     AGX_BARRIER_ALL | AGX_PREGFX, state,
                                     block_sums, nr_blocks, c_prims, c_inv,
                                     c_prims || c_inv);
      libagx_prefix_sum_tess_add(cmd, agx_1d(nr_patches - block_patches),
                                 AGX_BARRIER_ALL | AGX_PREGFX, counts,
                                 block_sums, nr_patches, block_patches);
   } else {
      libagx_prefix_sum_tess(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                             state, c_prims, c_inv, c_prims || c_inv);
   }
   HK_TESS_LANE(4);
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX,
                     gfx->tess.info.mode, POLY_TESS_MODE_WITH_COUNTS, state);
   cmd->tess_batch.capture = NULL;
#undef HK_TESS_LANE
   if (batch)
      cmd->tess_batch.draws++;

   return hk_heap_indexed_indirect(cmd, gfx->tess.out_draws,
                                   AGX_INDEX_SIZE_U32, false);
}

/* Run the vertex and geometry shaders as compute before the render pass,
 * returning the draw that rasterizes the geometry shader output.
 */
static struct agx_draw
hk_apple9_launch_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct hk_shader *vs = hk_bound_sw_vs_before_gs(gfx);
   struct hk_shader *main = hk_main_gs_variant(gs);
   struct hk_shader *count = hk_count_gs_variant(gs);
   struct hk_shader *pre_gs = hk_pre_gs_variant(gs);

   /* The software vertex shader fetches attributes relative to the API draw,
    * so bind its parameters before any restart unrolling rewrites the draw.
    */
   struct hk_apple9_prerast_bindings bindings =
      hk_apple9_draw_bindings(cmd, draw);

   desc->root.draw.vertex_params = hk_upload_vertex_params(cmd, draw);
   desc->root.draw.geometry_params = hk_upload_geometry_params(cmd, draw);
   desc->root.draw.api_gs = !gs->is_passthrough;
   desc->root.draw.rasterization_stream = dyn->rs.rasterization_stream;
   desc->root.draw.provoking =
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? 2
                                                                           : 0;
   desc->root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

   uint64_t vertex_params = desc->root.draw.vertex_params;
   uint64_t geometry_params = desc->root.draw.geometry_params;
   enum mesa_prim mode = hk_gs_in_prim(cmd);
   struct agx_grid grid_vs, grid_gs;

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      mode = u_decomposed_prim(mode);
   }

   if (agx_is_indirect(draw.b)) {
      struct libagx_gs_setup_indirect_args gsi = {
         .heap = hk_heap(cmd),
         .index_buffer = draw.index_buffer,
         .draw = draw.b.ptr,
         .vp = vertex_params,
         .p = geometry_params,
         .vs_outputs = vs->b.info.outputs,
         .prim = mode,
         .is_prefix_summing = count->info.gs.prefix_sum,
         .max_indices = count->info.gs.max_indices,
         .shape = count->info.gs.shape,
      };

      if (draw.indexed) {
         gsi.index_size_B = agx_index_size_to_B(draw.index_size);
         gsi.index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_gs_setup_indirect_struct(cmd, agx_1d(1),
                                      AGX_BARRIER_ALL | AGX_PREGFX, gsi);

      grid_vs = agx_grid_indirect_local(
         vertex_params + offsetof(struct poly_vertex_params, grid));
      grid_gs = agx_grid_indirect_local(
         geometry_params + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = grid_gs = draw.b;
      grid_gs.count[0] = u_decomposed_prims_for_vertices(mode, draw.b.count[0]);
   }

   hk_cpu_timing_note_gs(count->info.gs.shape, agx_is_indirect(draw.b),
                         (uint64_t)grid_gs.count[0] * grid_gs.count[1],
                         count->info.gs.max_indices);

   /* Transform feedback and geometry statistics need the count and pre-GS
    * programs.
    */
   VkQueryPipelineStatisticFlagBits gs_queries =
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
   bool xfb_or_queries =
      main->info.gs.xfb || (desc->root.draw.pipeline_stats_flags & gs_queries);
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i)
      xfb_or_queries |= gfx->pg_query[i] || gfx->xfb_query[i];

   /* Without transform feedback or queries, a small direct draw is just the
    * VS and GS passes over its own allocations: record them into the first
    * two batch lanes (as for tessellation) so consecutive draws overlap.
    */
   const unsigned max_batch_prims = 2048;
   bool batch = !agx_is_indirect(draw.b) && !xfb_or_queries &&
                grid_gs.count[0] * grid_gs.count[1] <= max_batch_prims;
   if (!batch)
      hk_tess_batch_flush(cmd);

   cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[0] : NULL;
   hk_apple9_dispatch_prerast(cmd, vs, grid_vs, &bindings);
   cmd->tess_batch.capture = NULL;

   if (xfb_or_queries) {
      if (count->info.gs.count_words)
         hk_apple9_dispatch_prerast(cmd, count, grid_gs, &bindings);

      if (count->info.gs.prefix_sum) {
         libagx_prefix_sum_geom(cmd, agx_1d(1024 * count->info.gs.count_words),
                                AGX_BARRIER_ALL | AGX_PREGFX, geometry_params);
      }

      hk_apple9_dispatch_prerast(cmd, pre_gs, agx_1d(1), &bindings);
   }

   cmd->tess_batch.capture = batch ? &cmd->tess_batch.lane[1] : NULL;
   hk_apple9_dispatch_prerast(cmd, main, grid_gs, &bindings);
   cmd->tess_batch.capture = NULL;
   if (batch)
      cmd->tess_batch.draws++;

   if (poly_gs_indexed(count->info.gs.shape)) {
      enum agx_index_size index_size =
         agx_translate_index_size(poly_gs_index_size(count->info.gs.shape));

      if (agx_is_indirect(draw.b)) {
         return agx_draw_indexed_indirect(
            geometry_params + offsetof(struct poly_geometry_params, draw),
            cmd->geom_index_buffer, cmd->geom_index_count, index_size, true);
      } else {
         return agx_draw_indexed(cmd->geom_index_count,
                                 cmd->geom_instance_count, 0, 0, 0,
                                 cmd->geom_index_buffer,
                                 cmd->geom_index_count * 4, index_size, true);
      }
   } else if (agx_is_indirect(draw.b)) {
      return agx_draw_indirect(geometry_params +
                               offsetof(struct poly_geometry_params, draw));
   } else {
      return (struct agx_draw){
         .b = agx_3d(cmd->geom_index_count, cmd->geom_instance_count, 1),
      };
   }
}

static struct agx_draw
hk_launch_tess(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
               struct agx_draw draw, uint64_t c_prims, uint64_t c_inv)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct agx_grid grid_vs, grid_tcs, grid_tess;

   struct hk_shader *vs = hk_bound_sw_vs(gfx);
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t input_patch_size = dyn->ts.patch_control_points;
   uint64_t state = gfx->descriptors.root.draw.tess_params;
   struct hk_tess_info info = gfx->tess.info;

   perf_debug(cmd, "Tessellation");

   uint64_t tcs_stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT);

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      perf_debug(cmd, "Indirect tessellation");

      struct libagx_tess_setup_indirect_args args = {
         .p = state,
         .grids = gfx->tess.grids,
         .indirect = draw.b.ptr,
         .vp = gfx->descriptors.root.draw.vertex_params,
         .vertex_outputs = vs->b.info.outputs,
         .tcs_statistic = hk_pipeline_stat_addr(
            cmd,
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT),
      };

      if (draw.indexed) {
         args.in_index_buffer = draw.index_buffer;
         args.in_index_size_B = agx_index_size_to_B(draw.index_size);
         args.in_index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_tess_setup_indirect_struct(cmd, agx_1d(1),
                                        AGX_BARRIER_ALL | AGX_PREGFX, args);

      uint32_t grid_stride = sizeof(uint32_t) * 6;
      grid_vs = agx_grid_indirect_local(gfx->tess.grids + 0 * grid_stride);
      grid_tcs = agx_grid_indirect_local(gfx->tess.grids + 1 * grid_stride);
      grid_tess = agx_grid_indirect_local(gfx->tess.grids + 2 * grid_stride);
   } else {
      uint32_t patches = draw.b.count[0] / input_patch_size;
      grid_vs = grid_tcs = draw.b;

      grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
      grid_tess = agx_1d(patches * draw.b.count[1]);

      /* TCS invocation counter increments once per-patch */
      if (hk_stat_enabled(tcs_stat)) {
         perf_debug(cmd, "Direct TCS statistic");
         libagx_increment_statistic(
            cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, tcs_stat, patches);
      }
   }

   /* First launch the VS and TCS */
   hk_reserve_scratch(cmd, cs, vs);
   hk_reserve_scratch(cmd, cs, tcs);

   hk_dispatch_with_usc(
      dev, cs, &vs->b.info,
      hk_upload_usc_words(cmd, vs, gfx->linked[MESA_SHADER_VERTEX]), grid_vs,
      agx_workgroup(64, 1, 1));

   hk_dispatch_with_usc(
      dev, cs, &tcs->b.info, hk_upload_usc_words(cmd, tcs, tcs->only_linked),
      grid_tcs, agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));

   /* First generate counts, then prefix sum them, and then tessellate. */
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_COUNT, state);

   libagx_prefix_sum_tess(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                          state, c_prims, c_inv, c_prims || c_inv);

   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_WITH_COUNTS, state);

   return hk_heap_indexed_indirect(cmd, gfx->tess.out_draws,
                                   AGX_INDEX_SIZE_U32, false);
}

void
hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                            const mesa_shader_stage stage,
                            struct hk_api_shader *shader)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* Null fragment shaders are annoying to handle, because we may still need an
    * actual fragment shader to attach a prolog to. Rather than adding special
    * cases, we just bind an empty fragment shader instead of NULL to make
    * everything work correctly.
    */
   if (stage == MESA_SHADER_FRAGMENT && shader == NULL) {
      shader = dev->null_fs;
   }

   assert(stage < ARRAY_SIZE(cmd->state.gfx.shaders));
   if (cmd->state.gfx.shaders[stage] == shader)
      return;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   uint64_t data = hk_shader_constant_data(shader);
   if (data || (desc->root_constant_stages & BITFIELD_BIT(stage))) {
      desc->root.constant_data[stage] = data;
      desc->root_constant_stages &= ~BITFIELD_BIT(stage);
      if (data)
         desc->root_constant_stages |= BITFIELD_BIT(stage);
      desc->root_dirty = true;
   }
   cmd->state.gfx.shaders[stage] = shader;
   cmd->state.gfx.shaders_dirty |= BITFIELD_BIT(stage);

   if (stage == MESA_SHADER_FRAGMENT) {
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
   }
}

static void
hk_flush_shaders(struct hk_cmd_buffer *cmd)
{
   if (cmd->state.gfx.shaders_dirty == 0)
      return;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   desc->root_dirty = true;

   /* Geometry shading overrides the restart index, reemit on rebind */
   if (IS_SHADER_DIRTY(GEOMETRY)) {
      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

      desc->root.draw.api_gs = gs && !gs->is_passthrough;
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];

   /* If we have a new VS/FS pair, UVS locations may have changed so need to
    * relink. We do this here because there's no dependence on the fast linked
    * shaders.
    */
   agx_assign_uvs(&gfx->linked_varyings, &hw_vs->info.uvs,
                  hk_only_variant(fs)->info.fs.interp.flat,
                  hk_only_variant(fs)->info.fs.interp.linear);

   for (unsigned i = 0; i < VARYING_SLOT_MAX; ++i) {
      desc->root.draw.uvs_index[i] = gfx->linked_varyings.slots[i];
   }
}

static struct agx_shader_part *
hk_get_prolog_epilog_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder, bool preprocess_nir,
                            bool stop, unsigned cf_base)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->prolog_epilog.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   builder(&b, key->key);

   if (preprocess_nir)
      agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev = agx_gather_device_key(&dev->dev),
      .secondary = true,
      .no_stop = !stop,
   };

   /* We always use dynamic sample shading in the GL driver. Indicate that. */
   if (b.shader->info.stage == MESA_SHADER_FRAGMENT) {
      backend_key.fs.cf_base = cf_base;

      if (b.shader->info.fs.uses_sample_shading)
         backend_key.fs.inside_sample_loop = true;
   }

   struct agx_shader_part *part =
      rzalloc(dev->prolog_epilog.ht, struct agx_shader_part);

   agx_compile_shader_nir(b.shader, &backend_key, part);

   ralloc_free(b.shader);

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->prolog_epilog.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->prolog_epilog.ht, cloned_key, part);
   return part;
}

static struct agx_shader_part *
hk_get_prolog_epilog(struct hk_device *dev, void *data, size_t data_size,
                     hk_internal_builder_t builder, bool preprocess_nir,
                     bool stop, unsigned cf_base)
{
   /* Build the meta shader key */
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->prolog_epilog.lock);

   struct agx_shader_part *part = hk_get_prolog_epilog_locked(
      dev, key, builder, preprocess_nir, stop, cf_base);

   simple_mtx_unlock(&dev->prolog_epilog.lock);
   return part;
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_vs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_vs *key)
{
   struct agx_shader_part *prolog =
      hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                           agx_nir_vs_prolog, false, false, 0);

   struct hk_linked_shader *linked =
      hk_fast_link(dev, false, shader, prolog, NULL, 0);

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));

   /* XXX: Fix this higher up the stack */
   linked->sw_indexing = !key->prolog.hw || key->prolog.adjacency;
   linked->b.uses_base_param |= linked->sw_indexing;

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

static bool
lower_fs_root(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_root_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64, .base = AGX_ABI_FUNI_ROOT));
   return true;
}

static void
build_fs_prolog(nir_builder *b, const void *key)
{
   agx_nir_fs_prolog(b, key);

   /* Lower load_stat_query_address_agx, needed for FS statistics */
   NIR_PASS(_, b->shader, hk_lower_uvs_index, MESA_SHADER_FRAGMENT, 0);
   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lower_fs_root,
            nir_metadata_control_flow, NULL);
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_fs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_fs *key)
{
   /* TODO: prolog without fs needs to work too... */
   bool needs_prolog = key->prolog.statistics ||
                       key->prolog.cull_distance_size ||
                       key->prolog.api_sample_mask != 0xff;

   struct agx_shader_part *prolog = NULL;
   if (needs_prolog) {
      prolog = hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                                    build_fs_prolog, false, false,
                                    key->prolog.cf_base);
   }

   /* If sample shading is used, don't stop at the epilog, there's a
    * footer that the fast linker will insert to stop.
    */
   bool epilog_stop = (key->nr_samples_shaded == 0);

   struct agx_shader_part *epilog =
      hk_get_prolog_epilog(dev, &key->epilog, sizeof(key->epilog),
                           agx_nir_fs_epilog, true, epilog_stop, 0);

   struct hk_linked_shader *linked =
      hk_fast_link(dev, true, shader, prolog, epilog, key->nr_samples_shaded);

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

/*
 * First, look for a fully linked variant. Else, build the required shader
 * parts and link.
 */
static struct hk_linked_shader *
hk_get_fast_linked(struct hk_device *dev, struct hk_shader *shader, void *key)
{
   struct hk_linked_shader *linked;
   simple_mtx_lock(&shader->linked.lock);

   struct hash_entry *ent = _mesa_hash_table_search(shader->linked.ht, key);

   if (ent)
      linked = ent->data;
   else if (shader->info.stage == MESA_SHADER_VERTEX)
      linked = hk_get_fast_linked_locked_vs(dev, shader, key);
   else if (shader->info.stage == MESA_SHADER_FRAGMENT)
      linked = hk_get_fast_linked_locked_fs(dev, shader, key);
   else
      UNREACHABLE("invalid stage");

   simple_mtx_unlock(&shader->linked.lock);
   return linked;
}

static void
hk_update_fast_linked(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                      void *key)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *new = hk_get_fast_linked(dev, shader, key);
   mesa_shader_stage stage = shader->info.stage;

   if (cmd->state.gfx.linked[stage] != new) {
      cmd->state.gfx.linked[stage] = new;
      cmd->state.gfx.linked_dirty |= BITFIELD_BIT(stage);
   }
}

static enum agx_polygon_mode
translate_polygon_mode(VkPolygonMode vk_mode)
{
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_FILL ==
                 AGX_POLYGON_MODE_FILL);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_LINE ==
                 AGX_POLYGON_MODE_LINE);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_POINT ==
                 AGX_POLYGON_MODE_POINT);

   assert(vk_mode <= VK_POLYGON_MODE_POINT);
   return (enum agx_polygon_mode)vk_mode;
}

static enum agx_zs_func
translate_compare_op(VkCompareOp vk_mode)
{
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NEVER == AGX_ZS_FUNC_NEVER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS == AGX_ZS_FUNC_LESS);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_EQUAL == AGX_ZS_FUNC_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS_OR_EQUAL ==
                 AGX_ZS_FUNC_LEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER ==
                 AGX_ZS_FUNC_GREATER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NOT_EQUAL ==
                 AGX_ZS_FUNC_NOT_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 AGX_ZS_FUNC_GEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_ALWAYS == AGX_ZS_FUNC_ALWAYS);

   assert(vk_mode <= VK_COMPARE_OP_ALWAYS);
   return (enum agx_zs_func)vk_mode;
}

static enum agx_stencil_op
translate_stencil_op(VkStencilOp vk_op)
{
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_KEEP ==
                 AGX_STENCIL_OP_KEEP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_ZERO ==
                 AGX_STENCIL_OP_ZERO);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_REPLACE ==
                 AGX_STENCIL_OP_REPLACE);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_INCR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_DECR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INVERT ==
                 AGX_STENCIL_OP_INVERT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_INCR_WRAP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_DECR_WRAP);

   return (enum agx_stencil_op)vk_op;
}

static void
hk_ppp_push_stencil_face(struct agx_ppp_update *ppp,
                         struct vk_stencil_test_face_state s, bool enabled)
{
   if (enabled) {
      agx_ppp_push(ppp, FRAGMENT_STENCIL, cfg) {
         cfg.compare = translate_compare_op(s.op.compare);
         cfg.write_mask = s.write_mask;
         cfg.read_mask = s.compare_mask;

         cfg.depth_pass = translate_stencil_op(s.op.pass);
         cfg.depth_fail = translate_stencil_op(s.op.depth_fail);
         cfg.stencil_fail = translate_stencil_op(s.op.fail);
      }
   } else {
      agx_ppp_push(ppp, FRAGMENT_STENCIL, cfg) {
         cfg.compare = AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = 0xFF;
         cfg.read_mask = 0xFF;

         cfg.depth_pass = AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = AGX_STENCIL_OP_KEEP;
      }
   }
}

static bool
hk_stencil_test_enabled(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   return dyn->ds.stencil.test_enable &&
          render->stencil_att.vk_format != VK_FORMAT_UNDEFINED;
}

static void
hk_flush_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* We always need at least 1 viewport for the hardware. With rasterizer
    * discard the app may not supply any, but we can just program garbage.
    */
   unsigned count = MAX2(dyn->vp.viewport_count, 1);

   unsigned minx[HK_MAX_VIEWPORTS] = {0}, miny[HK_MAX_VIEWPORTS] = {0};
   unsigned maxx[HK_MAX_VIEWPORTS] = {0}, maxy[HK_MAX_VIEWPORTS] = {0};

   /* We implicitly scissor to the viewport. We need to do a min/max dance to
    * handle inverted viewports.
    */
   for (uint32_t i = 0; i < dyn->vp.viewport_count; i++) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      minx[i] = MIN2(vp->x, vp->x + vp->width);
      miny[i] = MIN2(vp->y, vp->y + vp->height);
      maxx[i] = MAX2(vp->x, vp->x + vp->width);
      maxy[i] = MAX2(vp->y, vp->y + vp->height);
   }

   /* Additionally clamp to the framebuffer so we don't rasterize
    * off-screen pixels. TODO: Is this necessary? the GL driver does this but
    * it might be cargoculted at this point.
    */
   for (unsigned i = 0; i < count; ++i) {
      minx[i] = MIN2(minx[i], cmd->state.gfx.render.cr.width);
      maxx[i] = MIN2(maxx[i], cmd->state.gfx.render.cr.width);
      miny[i] = MIN2(miny[i], cmd->state.gfx.render.cr.height);
      maxy[i] = MIN2(maxy[i], cmd->state.gfx.render.cr.height);
   }

   /* We additionally apply any API scissors */
   for (unsigned i = 0; i < dyn->vp.scissor_count; ++i) {
      const VkRect2D *s = &dyn->vp.scissors[i];

      minx[i] = MAX2(minx[i], s->offset.x);
      miny[i] = MAX2(miny[i], s->offset.y);
      maxx[i] = MIN2(maxx[i], s->offset.x + s->extent.width);
      maxy[i] = MIN2(maxy[i], s->offset.y + s->extent.height);
   }

   /* Upload a hardware scissor for each viewport, whether there's a
    * corresponding API scissor or not.
    */
   unsigned index = cs->scissor.size / AGX_SCISSOR_LENGTH;
   struct agx_scissor_packed *scissors =
      util_dynarray_grow_bytes(&cs->scissor, count, AGX_SCISSOR_LENGTH);

   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_pack(scissors + i, SCISSOR, cfg) {
         cfg.min_x = minx[i];
         cfg.min_y = miny[i];
         cfg.max_x = maxx[i];
         cfg.max_y = maxy[i];

         /* These settings in conjunction with the PPP control depth clip/clamp
          * settings implement depth clip/clamping. Properly setting them
          * together is required for conformant depth clip enable.
          *
          * TODO: Reverse-engineer the finer interactions here.
          */
         if (dyn->rs.depth_clamp_enable) {
            cfg.min_z = MIN2(vp->minDepth, vp->maxDepth);
            cfg.max_z = MAX2(vp->minDepth, vp->maxDepth);
         } else {
            cfg.min_z = 0.0;
            cfg.max_z = 1.0;
         }
      }
   }

   /* Upload state */
   struct AGX_PPP_HEADER present = {
      .depth_bias_scissor = true,
      .region_clip = true,
      .viewport = true,
      .viewport_count = count,
   };

   size_t size = agx_ppp_update_size(&present);
   struct agx_ptr T = hk_pool_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

   agx_ppp_push(&ppp, DEPTH_BIAS_SCISSOR, cfg) {
      cfg.scissor = index;

      /* Use the current depth bias, we allocate linearly */
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      cfg.depth_bias = count ? count - 1 : 0;
   };

   for (unsigned i = 0; i < count; ++i) {
      agx_ppp_push(&ppp, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
   }

   agx_ppp_push(&ppp, VIEWPORT_CONTROL, cfg)
      ;

   /* Upload viewports */
   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_ppp_push(&ppp, VIEWPORT, cfg) {
         cfg.translate_x = vp->x + 0.5f * vp->width;
         cfg.translate_y = vp->y + 0.5f * vp->height;
         cfg.translate_z = vp->minDepth;

         cfg.scale_x = vp->width * 0.5f;
         cfg.scale_y = vp->height * 0.5f;
         cfg.scale_z = vp->maxDepth - vp->minDepth;
      }
   }

   agx_ppp_fini(out, &ppp);
}

static enum agx_object_type
translate_object_type(enum mesa_prim topology)
{
   static_assert(MESA_PRIM_LINES < MESA_PRIM_LINE_STRIP);
   static_assert(MESA_PRIM_TRIANGLES >= MESA_PRIM_LINE_STRIP);

   if (topology == MESA_PRIM_POINTS)
      return AGX_OBJECT_TYPE_POINT_SPRITE_UV01;
   else if (topology <= MESA_PRIM_LINE_STRIP)
      return AGX_OBJECT_TYPE_LINE;
   else
      return AGX_OBJECT_TYPE_TRIANGLE;
}

static enum agx_primitive
translate_hw_primitive_topology(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return AGX_PRIMITIVE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return AGX_PRIMITIVE_TRIANGLE_FAN;
   default:
      UNREACHABLE("Invalid hardware primitive topology");
   }
}

static inline enum agx_vdm_vertex
translate_vdm_vertex(unsigned vtx)
{
   static_assert(AGX_VDM_VERTEX_0 == 0);
   static_assert(AGX_VDM_VERTEX_1 == 1);
   static_assert(AGX_VDM_VERTEX_2 == 2);

   assert(vtx <= 2);
   return vtx;
}

static inline enum agx_ppp_vertex
translate_ppp_vertex(unsigned vtx)
{
   static_assert(AGX_PPP_VERTEX_0 == 0 + 1);
   static_assert(AGX_PPP_VERTEX_1 == 1 + 1);
   static_assert(AGX_PPP_VERTEX_2 == 2 + 1);

   assert(vtx <= 2);
   return vtx + 1;
}

static void
hk_flush_index(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   uint32_t index = cmd->state.gfx.index.restart;

   if (gs) {
      enum poly_gs_shape shape =
         gs->variants[HK_GS_VARIANT_COUNT].info.gs.shape;
      index = BITFIELD_MASK(8 * poly_gs_index_size(shape));
   }

   /* VDM State updates are relatively expensive, so only emit them when the
    * restart index changes. This is simpler than accurate dirty tracking.
    */
   if (cs->restart_index != index) {
      uint8_t *out = cs->current;
      agx_push(out, VDM_STATE, cfg) {
         cfg.restart_index_present = true;
      }

      agx_push(out, VDM_STATE_RESTART_INDEX, cfg) {
         cfg.value = index;
      }

      cs->current = out;
      cs->restart_index = index;
   }
}

/*
 * Return the given sample positions, packed into a 32-bit word with fixed
 * point nibbles for each x/y component of the (at most 4) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
uint32_t
hk_pack_ppp_multisamplectrl(const struct vk_sample_locations_state *sl)
{
   uint32_t ctrl = 0;

   for (int32_t i = sl->per_pixel - 1; i >= 0; i--) {
      VkSampleLocationEXT loc = sl->locations[i];

      uint32_t x = CLAMP(loc.x, 0.0f, 0.9375f) * 16.0;
      uint32_t y = CLAMP(loc.y, 0.0f, 0.9375f) * 16.0;

      assert(x <= 15);
      assert(y <= 15);

      /* Push bytes in reverse order so we can use constant shifts. */
      ctrl = (ctrl << 8) | (y << 4) | x;
   }

   return ctrl;
}

/*
 * Return the standard sample positions, prepacked as above for efficiency.
 */
uint32_t
hk_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 0:
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   default:
      UNREACHABLE("Invalid sample count");
   }
}

static void
hk_flush_ppp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);

   bool hw_vs_dirty = IS_SHADER_DIRTY(VERTEX) || IS_SHADER_DIRTY(TESS_EVAL) ||
                      IS_SHADER_DIRTY(GEOMETRY);
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   bool varyings_dirty = gfx->dirty & HK_DIRTY_VARYINGS;

   bool face_dirty =
      IS_DIRTY(DS_DEPTH_TEST_ENABLE) || IS_DIRTY(DS_DEPTH_WRITE_ENABLE) ||
      IS_DIRTY(DS_DEPTH_COMPARE_OP) || IS_DIRTY(DS_STENCIL_REFERENCE) ||
      IS_DIRTY(RS_LINE_WIDTH) || IS_DIRTY(RS_POLYGON_MODE) || fs_dirty;

   bool stencil_face_dirty =
      IS_DIRTY(DS_STENCIL_OP) || IS_DIRTY(DS_STENCIL_COMPARE_MASK) ||
      IS_DIRTY(DS_STENCIL_WRITE_MASK) || IS_DIRTY(DS_STENCIL_TEST_ENABLE);

   struct AGX_PPP_HEADER dirty = {
      .fragment_control =
         IS_DIRTY(DS_STENCIL_TEST_ENABLE) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
         IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || gfx->dirty & HK_DIRTY_OCCLUSION,

      .fragment_control_2 =
         IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_fs_dirty,

      .fragment_front_face = face_dirty,
      .fragment_front_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_front_stencil = stencil_face_dirty,
      .fragment_back_face = face_dirty,
      .fragment_back_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_back_stencil = stencil_face_dirty,
      .output_select = hw_vs_dirty || linked_fs_dirty || varyings_dirty,
      .varying_counts_32 = varyings_dirty,
      .varying_counts_16 = varyings_dirty,
      .cull = IS_DIRTY(RS_CULL_MODE) ||
              IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) ||
              IS_DIRTY(RS_FRONT_FACE) || IS_DIRTY(RS_DEPTH_CLIP_ENABLE) ||
              IS_DIRTY(RS_DEPTH_CLAMP_ENABLE) || IS_DIRTY(RS_LINE_MODE) ||
              IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
              (gfx->dirty & HK_DIRTY_PROVOKING) || IS_SHADER_DIRTY(TESS_CTRL) ||
              IS_SHADER_DIRTY(TESS_EVAL) || IS_DIRTY(TS_DOMAIN_ORIGIN),
      .cull_2 = varyings_dirty,

      .fragment_shader = fs_dirty || linked_fs_dirty || varyings_dirty ||
                         gfx->descriptors.root_dirty,

      .occlusion_query = gfx->dirty & HK_DIRTY_OCCLUSION,
      .output_size = hw_vs_dirty,
      .viewport_count = 1, /* irrelevant */
   };

   dirty.fragment_shader &= !linked_fs->b.no_op;

   /* Calculate the update size. If it equals the header, there is nothing to
    * update so early-exit.
    */
   size_t size = agx_ppp_update_size(&dirty);
   if (size == AGX_PPP_HEADER_LENGTH)
      return;

   /* Otherwise, allocate enough space for the update and push it. */
   assert(size > AGX_PPP_HEADER_LENGTH);

   struct agx_ptr T = hk_pool_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &dirty);

   if (dirty.fragment_control) {
      agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
         cfg.visibility_mode = gfx->occlusion.mode;
         cfg.stencil_test_enable = hk_stencil_test_enabled(cmd);

         /* TODO: Consider optimizing this? */
         cfg.two_sided_stencil = cfg.stencil_test_enable;

         cfg.depth_bias_enable = dyn->rs.depth_bias.enable &&
                                 gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;

         /* Always enable scissoring so we may scissor to the viewport (TODO:
          * optimize this out if the viewport is the default and the app does
          * not use the scissor test)
          */
         cfg.scissor_enable = true;

         /* This avoids broken derivatives along primitive edges */
         cfg.disable_tri_merging = gfx->object_type != AGX_OBJECT_TYPE_TRIANGLE;
      }
   }

   if (dirty.fragment_control_2) {
      /* Annoying, rasterizer_discard seems to be ignored (sometimes?) in the
       * main fragment control word and has to be combined into the secondary
       * word for reliable behaviour.
       */
      agx_ppp_push_merged(&ppp, FRAGMENT_CONTROL, cfg,
                          linked_fs->b.fragment_control) {

         cfg.tag_write_disable = hk_rast_discard(cmd);
      }
   }

   struct agx_fragment_face_packed fragment_face = {};
   struct agx_fragment_face_2_packed fragment_face_2 = {};

   if (dirty.fragment_front_face) {
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      agx_pack(&fragment_face, FRAGMENT_FACE, cfg) {
         cfg.line_width = agx_pack_line_width(dyn->rs.line.width);
         cfg.polygon_mode = translate_polygon_mode(dyn->rs.polygon_mode);
         cfg.disable_depth_write = !(z_test && dyn->ds.depth.write_enable);

         if (z_test && !gfx->descriptors.root.draw.force_never_in_shader)
            cfg.depth_function = translate_compare_op(dyn->ds.depth.compare_op);
         else
            cfg.depth_function = AGX_ZS_FUNC_ALWAYS;
      };

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.front.reference;
      }
   }

   if (dirty.fragment_front_face_2) {
      agx_pack(&fragment_face_2, FRAGMENT_FACE_2, cfg) {
         cfg.object_type = gfx->object_type;
      }

      agx_merge(fragment_face_2, fs->frag_face, FRAGMENT_FACE_2);
      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_front_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.front,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.fragment_back_face) {
      assert(dirty.fragment_front_face);

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.back.reference;
      }
   }

   if (dirty.fragment_back_face_2) {
      assert(dirty.fragment_front_face_2);

      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_back_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.back,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.output_select) {
      struct agx_output_select_packed osel = hw_vs->info.uvs.osel;

      agx_ppp_push_merged_blobs(&ppp, AGX_OUTPUT_SELECT_LENGTH, &osel,
                                &linked_fs->b.osel);
   }

   assert(dirty.varying_counts_32 == dirty.varying_counts_16);

   if (dirty.varying_counts_32) {
      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_32,
                          VARYING_COUNTS);

      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_16,
                          VARYING_COUNTS);
   }

   if (dirty.cull) {
      agx_ppp_push(&ppp, CULL, cfg) {
         cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
         cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
         cfg.front_face_ccw = dyn->rs.front_face != VK_FRONT_FACE_CLOCKWISE;

         if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
             !gfx->shaders[MESA_SHADER_GEOMETRY]) {
            cfg.front_face_ccw ^= gfx->tess.info.ccw;
            cfg.front_face_ccw ^= dyn->ts.domain_origin ==
                                  VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
         }

         cfg.flat_shading_vertex = translate_ppp_vertex(gfx->provoking);
         cfg.rasterizer_discard = hk_rast_discard(cmd);

         /* We do not support unrestricted depth, so clamping is inverted from
          * clipping. This implementation seems to pass CTS without unrestricted
          * depth support.
          *
          * TODO: Make sure this is right with gl_FragDepth.
          */
         cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
         cfg.depth_clamp = !cfg.depth_clip;

         cfg.primitive_msaa =
            gfx->object_type == AGX_OBJECT_TYPE_LINE &&
            dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
      }
   }

   if (dirty.cull_2) {
      agx_ppp_push(&ppp, CULL_2, cfg) {
         cfg.needs_primitive_id = gfx->generate_primitive_id;
         cfg.clamp_w = true;
      }
   }

   if (dirty.fragment_shader) {
      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, fs);

      agx_ppp_push_packed(&ppp, &linked_fs->fs_counts, FRAGMENT_SHADER_WORD_0);

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_1, cfg) {
         cfg.pipeline = hk_upload_usc_words(cmd, fs, linked_fs);
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_2, cfg) {
         cfg.cf_bindings = gfx->varyings;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_3, cfg)
         ;
   }

   if (dirty.occlusion_query) {
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY, cfg) {
         cfg.index = gfx->occlusion.index;
      }
   }

   if (dirty.output_size) {
      agx_ppp_push(&ppp, OUTPUT_SIZE, cfg) {
         cfg.count = hw_vs->info.uvs.size;
      }
   }

   agx_ppp_fini(out, &ppp);
}

/*
 * Based somewhat on the calculation in the PowerVR driver, and mostly trial &
 * error to pass CTS. This is a mess.
 */
static float
hk_depth_bias_factor(VkFormat format, bool exact, bool force_unorm)
{
   if (format == VK_FORMAT_D16_UNORM) {
      return exact ? (1 << 16) : (1 << 15);
   } else if (force_unorm) {
      return exact ? (1ull << 24) : (1ull << 23);
   } else {
      return 1.0;
   }
}

static bool
uses_blend_constant(const struct vk_color_blend_state *cb)
{
   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 1 ==
                 VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR);

   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 2 ==
                 VK_BLEND_FACTOR_CONSTANT_ALPHA);

   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 3 ==
                 VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA);

   for (unsigned i = 0; i < cb->attachment_count; ++i) {
      unsigned factors[] = {
         cb->attachments[i].src_color_blend_factor,
         cb->attachments[i].src_alpha_blend_factor,
         cb->attachments[i].dst_color_blend_factor,
         cb->attachments[i].dst_alpha_blend_factor,
      };

      for (unsigned j = 0; j < ARRAY_SIZE(factors); ++j) {
         if (factors[j] >= VK_BLEND_FACTOR_CONSTANT_COLOR &&
             factors[j] <= VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
            return true;
      }
   }

   return false;
}

void
agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                    uint64_t attribs_read, struct agx_velem_key *keys)
{
   u_foreach_bit(a, vi->attributes_valid) {
      struct vk_vertex_attribute_state attr = vi->attributes[a];

      assert(vi->bindings_valid & BITFIELD_BIT(attr.binding));
      struct vk_vertex_binding_state binding = vi->bindings[attr.binding];

      /* nir_assign_io_var_locations compacts vertex inputs, eliminating
       * unused inputs. We need to do the same here to match the locations.
       */
      unsigned slot = util_bitcount64(attribs_read & BITFIELD_MASK(a));

      keys[slot] = (struct agx_velem_key){
         .format = hk_format_to_pipe_format(attr.format),
         .stride = binding.stride,
         .divisor = binding.divisor,
         .instanced = binding.input_rate == VK_VERTEX_INPUT_RATE_INSTANCE,
      };
   }
}

static void
hk_flush_dynamic_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       uint32_t draw_id, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);

   if (!vk_dynamic_graphics_state_any_dirty(dyn) && !gfx->dirty &&
       !gfx->descriptors.root_dirty && !gfx->shaders_dirty &&
       !sw_vs->b.info.uses_draw_id && !sw_vs->b.info.uses_base_param &&
       !(gfx->linked[MESA_SHADER_VERTEX] &&
         gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param))
      return;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   assert(cs->current + 0x1000 < cs->end && "already ensured space");
   uint8_t *out = cs->current;

   struct hk_shader *fs = hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT]);

   bool gt_dirty = IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL) ||
                   IS_SHADER_DIRTY(GEOMETRY);
   bool vgt_dirty = IS_SHADER_DIRTY(VERTEX) || gt_dirty;
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   if (IS_DIRTY(CB_BLEND_CONSTANTS)) {
      static_assert(sizeof(desc->root.draw.blend_constant) ==
                       sizeof(dyn->cb.blend_constants),
                    "common size");

      memcpy(desc->root.draw.blend_constant, dyn->cb.blend_constants,
             sizeof(dyn->cb.blend_constants));
      desc->root_dirty = true;
   }

   if (IS_DIRTY(CB_BLEND_EQUATIONS) || IS_DIRTY(CB_BLEND_ENABLES)) {
      gfx->uses_blend_constant = uses_blend_constant(&dyn->cb);
   }

   if (IS_DIRTY(MS_SAMPLE_MASK)) {
      desc->root.draw.api_sample_mask = dyn->ms.sample_mask;
      desc->root_dirty = true;
   }

   if (IS_DIRTY(RS_RASTERIZATION_STREAM)) {
      desc->root.draw.rasterization_stream = dyn->rs.rasterization_stream;
      desc->root_dirty = true;

      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   if (fs_dirty || IS_DIRTY(DS_DEPTH_TEST_ENABLE) ||
       IS_DIRTY(DS_DEPTH_COMPARE_OP)) {

      const struct hk_rendering_state *render = &cmd->state.gfx.render;
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      desc->root.draw.force_never_in_shader =
         z_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER && fs &&
         fs->info.fs.writes_memory;

      desc->root_dirty = true;
   }

   /* The main shader must not run tests if the epilog will. */
   bool nontrivial_force_early =
      fs && (fs->b.info.early_fragment_tests &&
             (fs->b.info.writes_sample_mask || fs->info.fs.writes_memory));

   bool epilog_discards = dyn->ms.alpha_to_coverage_enable ||
                          (fs && (fs->info.fs.epilog_key.write_z ||
                                  fs->info.fs.epilog_key.write_s));
   epilog_discards &= !nontrivial_force_early;

   if (fs_dirty || IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE)) {
      desc->root.draw.no_epilog_discard = !epilog_discards ? ~0 : 0;
      desc->root_dirty = true;
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || vgt_dirty || true /* TODO */) {

      struct hk_fast_link_key_vs key = {
         .prolog.hw = (sw_vs == hw_vs),

         /* FIXME: handle pipeline robustness "properly" */
         .prolog.robustness.level =
            (dev->vk.enabled_features.robustBufferAccess2 ||
             dev->vk.enabled_features.pipelineRobustness)
               ? AGX_ROBUSTNESS_D3D
            : dev->vk.enabled_features.robustBufferAccess
               ? AGX_ROBUSTNESS_GL
               : AGX_ROBUSTNESS_DISABLED,

         .prolog.robustness.soft_fault = agx_has_soft_fault(&dev->dev),
         .prolog.static_vi = !sw_vs->info.vs.use_prolog,
      };

      enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

      if (mesa_prim_has_adjacency(prim)) {
         if (draw.restart) {
            prim = u_decomposed_prim(prim);
         }

         key.prolog.adjacency = prim;
      }

      if (key.prolog.adjacency || !key.prolog.hw) {
         key.prolog.sw_index_size_B =
            draw.indexed ? agx_index_size_to_B(draw.index_size) : 0;
      }

      static_assert(sizeof(key.prolog.component_mask) ==
                    sizeof(sw_vs->info.vs.attrib_components_read));
      BITSET_COPY(key.prolog.component_mask,
                  sw_vs->info.vs.attrib_components_read);

      if (sw_vs->info.vs.use_prolog) {
         agx_fill_velem_keys(dyn->vi, sw_vs->info.vs.attribs_read,
                             key.prolog.attribs);

         u_foreach_bit(a, dyn->vi->attributes_valid) {
            unsigned slot =
               util_bitcount64(sw_vs->info.vs.attribs_read & BITFIELD_MASK(a));

            key.prolog.attribs[slot].stride =
               dyn->vi_binding_strides[dyn->vi->attributes[a].binding];
         }
      }

      hk_update_fast_linked(cmd, sw_vs, &key);
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || vgt_dirty ||
       (gfx->dirty & HK_DIRTY_VB)) {

      unsigned slot = 0;
      u_foreach_bit(a, sw_vs->info.vs.attribs_read) {
         if (dyn->vi->attributes_valid & BITFIELD_BIT(a)) {
            struct vk_vertex_attribute_state attr = dyn->vi->attributes[a];
            struct hk_addr_range vb = gfx->vb[attr.binding];
            enum pipe_format fmt = hk_format_to_pipe_format(attr.format);
            enum pipe_format interchange_format = agx_vbo_internal_format(fmt);
            unsigned interchange_align =
               util_format_get_blocksize(interchange_format);

            desc->root.draw.attrib_clamps[slot] = agx_calculate_vbo_clamp(
               vb.addr, fmt, vb.range, dyn->vi_binding_strides[attr.binding],
               attr.offset, &desc->root.draw.attrib_base[slot]);

            desc->root.draw.attrib_strides[slot] =
               dyn->vi_binding_strides[attr.binding] / interchange_align;
         } else {
            desc->root.draw.attrib_base[slot] = AGX_ZERO_PAGE_ADDRESS;
            desc->root.draw.attrib_clamps[slot] = 0;
            desc->root.draw.attrib_strides[slot] = 0;
         }

         ++slot;
      }

      desc->root_dirty = true;
   }

   if (vgt_dirty || IS_SHADER_DIRTY(FRAGMENT) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES) || IS_DIRTY(MS_SAMPLE_MASK) ||
       IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE) ||
       IS_DIRTY(MS_ALPHA_TO_ONE_ENABLE) || IS_DIRTY(CB_LOGIC_OP) ||
       IS_DIRTY(CB_LOGIC_OP_ENABLE) || IS_DIRTY(CB_WRITE_MASKS) ||
       IS_DIRTY(CB_COLOR_WRITE_ENABLES) || IS_DIRTY(CB_ATTACHMENT_COUNT) ||
       IS_DIRTY(CB_BLEND_ENABLES) || IS_DIRTY(CB_BLEND_EQUATIONS) ||
       IS_DIRTY(CB_BLEND_CONSTANTS) || IS_DIRTY(CB_BLEND_ADVANCED) ||
       desc->root_dirty /* for pipeline stats */ || true) {

      unsigned tib_sample_mask = BITFIELD_MASK(dyn->ms.rasterization_samples);
      unsigned api_sample_mask = dyn->ms.sample_mask & tib_sample_mask;
      bool has_sample_mask = api_sample_mask != tib_sample_mask;

      if (hw_vs->info.cull_distance_array_size) {
         perf_debug(cmd, "Emulating cull distance (size %u)",
                    hw_vs->info.cull_distance_array_size);
      }

      if (has_sample_mask) {
         perf_debug(cmd, "Emulating sample mask");
      }

      unsigned samples_shaded = 0;
      if (fs->info.fs.epilog_key.sample_shading)
         samples_shaded = dyn->ms.rasterization_samples;

      struct hk_fast_link_key_fs key = {
         .prolog.statistics = hk_stat_enabled(hk_pipeline_stat_addr(
            cmd, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)),

         .prolog.cull_distance_size = hw_vs->info.cull_distance_array_size,
         .prolog.api_sample_mask = has_sample_mask ? api_sample_mask : 0xff,
         .nr_samples_shaded = samples_shaded,
      };

      bool prolog_discards = has_sample_mask || key.prolog.cull_distance_size;

      bool needs_prolog = key.prolog.statistics || prolog_discards;

      if (needs_prolog) {
         /* With late main shader tests, the prolog runs tests if neither the
          * main shader nor epilog will.
          *
          * With (nontrivial) early main shader tests, the prolog does not
          * run tests, the tests will run at the start of the main shader.
          * This ensures tests are after API sample mask and cull distance
          * discards.
          */
         key.prolog.run_zs_tests = !nontrivial_force_early &&
                                   !fs->b.info.writes_sample_mask &&
                                   !epilog_discards && prolog_discards;

         if (key.prolog.cull_distance_size) {
            key.prolog.cf_base = fs->b.info.varyings.fs.nr_cf;
         }
      }

      key.epilog = (struct agx_fs_epilog_key){
         .link = fs->info.fs.epilog_key,
         .nr_samples = MAX2(dyn->ms.rasterization_samples, 1),
         .blend.alpha_to_coverage = dyn->ms.alpha_to_coverage_enable,
         .blend.alpha_to_one = dyn->ms.alpha_to_one_enable,
         .blend.logicop_enable = dyn->cb.logic_op_enable,
         .blend.logicop_func = vk_logic_op_to_pipe(dyn->cb.logic_op),
      };

      for (unsigned rt = 0; rt < ARRAY_SIZE(dyn->cal.color_map); ++rt) {
         int map = dyn->cal.color_map[rt];
         key.epilog.remap[rt] = map == MESA_VK_ATTACHMENT_UNUSED ? -1 : map;
      }

      if (dyn->ms.alpha_to_one_enable || dyn->ms.alpha_to_coverage_enable ||
          dyn->cb.logic_op_enable) {

         perf_debug(
            cmd, "Epilog with%s%s%s",
            dyn->ms.alpha_to_one_enable ? " alpha-to-one" : "",
            dyn->ms.alpha_to_coverage_enable ? " alpha-to-coverage" : "",
            dyn->cb.logic_op_enable ? " logic-op" : "");
      }

      key.epilog.link.already_ran_zs |= nontrivial_force_early;

      struct hk_rendering_state *render = &cmd->state.gfx.render;
      for (uint32_t i = 0; i < render->color_att_count; i++) {
         key.epilog.rt_formats[i] =
            hk_format_to_pipe_format(render->color_att[i].vk_format);

         const struct vk_color_blend_attachment_state *cb =
            &dyn->cb.attachments[i];

         bool write_enable = dyn->cb.color_write_enables & BITFIELD_BIT(i);
         unsigned write_mask = write_enable ? cb->write_mask : 0;

         /* nir_lower_blend always blends, so use a default blend state when
          * blending is disabled at an API level.
          */
         if (!dyn->cb.attachments[i].blend_enable) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .mode = agx_pack_blend_standard(
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
            };
         } else if (cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .advanced_blend = 1,
               .mode = agx_pack_blend_advanced(
                  vk_advanced_blend_op_to_pipe(cb->color_blend_op),
                  vk_blend_overlap_to_pipe(cb->blend_overlap),
                  cb->src_premultiplied, cb->dst_premultiplied,
                  cb->clamp_results),
            };

            assert(cb->clamp_results == false);
         } else {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .mode = agx_pack_blend_standard(
                  vk_blend_op_to_pipe(cb->color_blend_op),
                  vk_blend_factor_to_pipe(cb->src_color_blend_factor),
                  vk_blend_factor_to_pipe(cb->dst_color_blend_factor),
                  vk_blend_op_to_pipe(cb->alpha_blend_op),
                  vk_blend_factor_to_pipe(cb->src_alpha_blend_factor),
                  vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor)),
            };
         }
      }

      hk_update_fast_linked(cmd, fs, &key);
   }

   /* If the vertex shader uses draw parameters, vertex uniforms are dirty every
    * draw. Fragment uniforms are unaffected.
    *
    * For a direct draw, we upload the draw parameters as-if indirect to
    * avoid keying to indirectness.
    */
   if (gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param) {
      if (agx_is_indirect(draw.b)) {
         gfx->draw_params = draw.b.ptr;

         if (draw.indexed) {
            gfx->draw_params +=
               offsetof(VkDrawIndexedIndirectCommand, vertexOffset);
         } else {
            gfx->draw_params += offsetof(VkDrawIndirectCommand, firstVertex);
         }
      } else {
         uint32_t params[] = {
            draw.indexed ? draw.index_bias : draw.start,
            draw.start_instance,
         };

         gfx->draw_params = hk_pool_upload(cmd, params, sizeof(params), 4);
      }
   } else {
      gfx->draw_params = 0;
   }

   if (sw_vs->b.info.uses_draw_id) {
      /* TODO: rodata? */
      gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id, 2, 4);
   } else {
      gfx->draw_id_ptr = 0;
   }

   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || gt_dirty) {
      enum mesa_prim prim = hk_rast_prim(cmd);

      gfx->topology = translate_hw_primitive_topology(prim);
      gfx->object_type = translate_object_type(prim);
   }

   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || IS_DIRTY(RS_PROVOKING_VERTEX)) {
      unsigned provoking;
      if (dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT)
         provoking = 2;
      else if (gfx->topology == AGX_PRIMITIVE_TRIANGLE_FAN)
         provoking = 1;
      else
         provoking = 0;

      if (provoking != gfx->provoking) {
         gfx->provoking = provoking;
         gfx->dirty |= HK_DIRTY_PROVOKING;

         gfx->descriptors.root.draw.provoking = provoking;
         gfx->descriptors.root_dirty = true;
      }
   }

   /* With attachmentless rendering, we don't know the sample count until draw
    * time, so we do a late tilebuffer fix up. But with rasterizer discard,
    * rasterization_samples might be 0.
    *
    * Note that we ignore dyn->ms.rasterization_samples when we do have a sample
    * count from an attachment. In Vulkan, these have to match anyway, but DX12
    * drivers are robust against this scenarios and vkd3d-proton will go out of
    * spec here. No reason we can't be robust here too.
    */
   if (dyn->ms.rasterization_samples && !gfx->render.tilebuffer.nr_samples) {
      agx_tilebuffer_set_samples(&gfx->render.tilebuffer,
                                 dyn->ms.rasterization_samples);

      cs->tib = gfx->render.tilebuffer;
   }

   if (IS_DIRTY(MS_SAMPLE_LOCATIONS) || IS_DIRTY(MS_SAMPLE_LOCATIONS_ENABLE) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES)) {

      uint32_t ctrl;
      if (dyn->ms.sample_locations_enable) {
         ctrl = hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations);
      } else {
         ctrl = hk_default_sample_positions(dyn->ms.rasterization_samples);
      }

      bool dont_commit = cmd->in_meta || dyn->ms.rasterization_samples == 0;

      if (!cs->has_sample_locations) {
         cs->ppp_multisamplectl = ctrl;

         /* If we're in vk_meta, do not commit to the sample locations yet.
          * vk_meta doesn't care, but the app will!
          */
         cs->has_sample_locations |= !dont_commit;
      } else {
         assert(dont_commit || cs->ppp_multisamplectl == ctrl);
      }

      gfx->descriptors.root.draw.ppp_multisamplectl = ctrl;
      gfx->descriptors.root_dirty = true;
   }

   /* Link varyings before uploading tessellation state, becuase the
    * gfx->generate_primitive_id boolean needs to be plumbed.
    */
   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_vs_dirty = IS_LINKED_DIRTY(VERTEX);
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   if ((gfx->dirty & HK_DIRTY_PROVOKING) || vgt_dirty || linked_fs_dirty) {
      unsigned bindings = linked_fs->b.cf.nr_bindings;
      if (bindings) {
         size_t linkage_size =
            AGX_CF_BINDING_HEADER_LENGTH + (bindings * AGX_CF_BINDING_LENGTH);

         struct agx_ptr t = hk_pool_usc_alloc(cmd, linkage_size, 16);
         if (!t.cpu)
            return;

         agx_link_varyings_vs_fs(
            t.cpu, &gfx->linked_varyings, hw_vs->info.uvs.user_size,
            &linked_fs->b.cf, gfx->provoking, 0, &gfx->generate_primitive_id);

         gfx->varyings = agx_usc_addr(&dev->dev, t.gpu);
      } else {
         gfx->varyings = 0;
      }

      gfx->dirty |= HK_DIRTY_VARYINGS;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] || linked_vs->sw_indexing) {
      desc->root.draw.vertex_params = hk_upload_vertex_params(cmd, draw);
      desc->root_dirty = true;
   }

   struct agx_ptr tess_args = {0};
   if (gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      tess_args = hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
      gfx->descriptors.root.draw.tess_params = tess_args.gpu;
      gfx->descriptors.root_dirty = true;
   }

   if (gfx->shaders[MESA_SHADER_GEOMETRY]) {
      gfx->descriptors.root.draw.geometry_params =
         hk_upload_geometry_params(cmd, draw);

      gfx->descriptors.root_dirty = true;
   }

   /* Root must be uploaded after the above, which touch the root */
   if (gfx->descriptors.root_dirty) {
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

      /* Tess parameters depend on the root address, so we defer the upload
       * until after uploading root. But the root depends on the tess address,
       * so we allocate tess parameters before uploading root.
       *
       * This whole mechanism is a mess ported over from the GL driver. I'm
       * planning to do a massive rework of indirect geom/tess so I'm trying not
       * to perfectionism it in the mean time.
       */
      if (tess_args.cpu) {
         hk_upload_tess_params(cmd, tess_args.cpu, draw);
      }
   }

   /* Hardware dynamic state must be deferred until after the root and fast
    * linking, since it will use the root address and the linked shaders.
    */
   if ((gfx->dirty & (HK_DIRTY_PROVOKING | HK_DIRTY_VARYINGS)) ||
       IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_vs_dirty || vgt_dirty ||
       gfx->descriptors.root_dirty || gfx->draw_id_ptr || gfx->draw_params) {

      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, hw_vs);

      agx_push(out, VDM_STATE, cfg) {
         cfg.vertex_shader_word_0_present = true;
         cfg.vertex_shader_word_1_present = true;
         cfg.vertex_outputs_present = true;
         cfg.vertex_unknown_present = true;
      }

      agx_push_packed(out, hw_vs->counts, VDM_STATE_VERTEX_SHADER_WORD_0);

      struct hk_linked_shader *linked_hw_vs =
         (hw_vs == sw_vs) ? linked_vs : hw_vs->only_linked;

      agx_push(out, VDM_STATE_VERTEX_SHADER_WORD_1, cfg) {
         cfg.pipeline = hk_upload_usc_words(cmd, hw_vs, linked_hw_vs);
      }

      agx_push_packed(out, hw_vs->info.uvs.vdm, VDM_STATE_VERTEX_OUTPUTS);

      agx_push(out, VDM_STATE_VERTEX_UNKNOWN, cfg) {
         cfg.flat_shading_control = translate_vdm_vertex(gfx->provoking);
         cfg.unknown_4 = cfg.unknown_5 = hk_rast_discard(cmd);
         cfg.generate_primitive_id = gfx->generate_primitive_id;
      }

      /* Pad up to a multiple of 8 bytes */
      memset(out, 0, 4);
      out += 4;
   }

   if (IS_DIRTY(RS_DEPTH_BIAS_FACTORS)) {
      void *ptr =
         util_dynarray_grow_bytes(&cs->depth_bias, 1, AGX_DEPTH_BIAS_LENGTH);

      bool exact = dyn->rs.depth_bias.exact;
      bool force_unorm =
         dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;

      agx_pack(ptr, DEPTH_BIAS, cfg) {
         cfg.slope_scale = dyn->rs.depth_bias.slope_factor;
         cfg.clamp = dyn->rs.depth_bias.clamp;
         cfg.depth_bias = dyn->rs.depth_bias.constant_factor;
         cfg.depth_bias /= hk_depth_bias_factor(render->depth_att.vk_format,
                                                exact, force_unorm);
      }
   }

   /* Hardware viewport/scissor state is entangled with depth bias. */
   if (IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(VP_SCISSORS) ||
       IS_DIRTY(VP_SCISSOR_COUNT) || IS_DIRTY(VP_VIEWPORTS) ||
       IS_DIRTY(VP_VIEWPORT_COUNT) ||
       IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE)) {

      hk_flush_vp_state(cmd, cs, &out);
   }

   hk_flush_ppp_state(cmd, cs, &out);
   cs->current = out;

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gfx->shaders_dirty = 0;
   gfx->linked_dirty = 0;
   gfx->dirty = 0;
   gfx->descriptors.root_dirty = false;
}

static bool
hk_needs_index_robustness(struct hk_cmd_buffer *cmd, struct agx_draw *draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!draw->indexed)
      return false;

   /* Geometry or tessellation use robust software index buffer fetch anyway */
   if (gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return false;

   /* Soft fault does not cover the hardware index buffer fetch, use the zero
    * page instead.
    */
   if (draw->index_buffer_range_B == 0) {
      draw->index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw->index_buffer_range_B = 4;
      draw->start = 0;
      return false;
   }

   if (!(dev->vk.enabled_features.robustBufferAccess ||
         dev->vk.enabled_features.robustBufferAccess2 ||
         dev->vk.enabled_features.pipelineRobustness))
      return false;

   if (agx_is_indirect(draw->b))
      return true;

   return agx_direct_draw_overreads_indices(*draw);
}

/* Direct draws through software vertex shading size the vertex output buffer
 * on the CPU (hk_upload_vertex_params). Its 32-bit size wraps for very large
 * draws, e.g. a vertex count of UINT32_MAX, and the vertex shader would then
 * write past a small allocation. Such a buffer cannot be allocated anyway, so
 * record the out-of-memory error and skip the draw.
 */
static bool
hk_sw_vs_outputs_fit(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   if (agx_is_indirect(draw.b) || (!gfx->shaders[MESA_SHADER_GEOMETRY] &&
                                   !gfx->shaders[MESA_SHADER_TESS_EVAL]))
      return true;

   uint64_t vertex_B = util_bitcount64(hk_bound_sw_vs(gfx)->b.info.outputs) * 16;
   uint64_t vertices = (uint64_t)draw.b.count[0] * draw.b.count[1];

   if (!vertex_B || vertices <= UINT32_MAX / vertex_B)
      return true;

   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return false;
}

static void
hk_handle_passthrough_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   bool weights_by_vs =
      !draw.indexed &&
      dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

   /* Stippled strips continue their counter across segments, and the stipple
    * start must come from each segment's first vertex: both need the
    * passthrough geometry shader unless the vertex shader's start suffices,
    * see hk_apple9_line_stipple_gs. */
   const struct hk_shader *fs_stipple =
      gfx->shaders[MESA_SHADER_FRAGMENT]
         ? hk_only_variant(gfx->shaders[MESA_SHADER_FRAGMENT])
         : NULL;
   bool stipple_strip =
      dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ||
      dyn->ia.primitive_topology ==
         VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
   bool stipple_gs =
      hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G && fs_stipple &&
      !gfx->shaders[MESA_SHADER_TESS_EVAL] &&
      fs_stipple->info.fs.line_stipple_start && dyn->rs.line.stipple.enable &&
      (stipple_strip ||
       (dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT &&
        !fs_stipple->info.apple9_line_stipple_first));

   /* Polygon mode POINT on triangles draws real points from a software
    * polygon stage, see hk_apple9_polygon_point_gs. */
   bool polygon_points =
      hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G && fs_stipple &&
      fs_stipple->info.fs.polygon_points &&
      dyn->rs.polygon_mode == VK_POLYGON_MODE_POINT &&
      u_reduced_prim(hk_gs_in_prim(cmd)) == MESA_PRIM_TRIANGLES;

   if (!IS_SHADER_DIRTY(VERTEX) && !IS_SHADER_DIRTY(TESS_EVAL) &&
       !IS_SHADER_DIRTY(GEOMETRY) && !IS_SHADER_DIRTY(FRAGMENT) &&
       !IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) &&
       weights_by_vs == gfx->apple9_weights_by_vs &&
       stipple_gs == gfx->apple9_stipple_gs &&
       polygon_points == gfx->apple9_polygon_points)
      return;
   gfx->apple9_weights_by_vs = weights_by_vs;
   gfx->apple9_stipple_gs = stipple_gs;
   gfx->apple9_polygon_points = polygon_points;

   /* If there's an application geometry shader, there's nothing to un/bind */
   if (gs && !gs->is_passthrough)
      return;

   /* Determine if we need a geometry shader to emulate XFB or adjacency */
   struct hk_shader *last_sw = hk_bound_sw_vs_before_gs(gfx);
   uint32_t xfb_outputs = last_sw->info.xfb_info.output_count;
   /* Captured vertices are the polygon's, not points: transform feedback
    * keeps the hardware polygon mode. */
   polygon_points &= !xfb_outputs;
   /* A per-vertex cull mask is the primitive's only for points. */
   bool cull_mask = last_sw->info.cull_mask;
   bool needs_gs = xfb_outputs || stipple_gs || polygon_points ||
                   (cull_mask &&
                    u_reduced_prim(hk_gs_in_prim(cmd)) != MESA_PRIM_POINTS);
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i)
      needs_gs |= gfx->pg_query[i] != 0;

   /* ...or to give the fragment shader flat copies of per-vertex inputs */
   struct hk_api_shader *fs_api_copies = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G && fs_api_copies) {
      const struct hk_shader *fs_copies = hk_only_variant(fs_api_copies);
      /* The vertex shader writes the barycentric weights of non-indexed
       * triangle lists when there are no per-vertex inputs. */
      needs_gs |= fs_copies->info.fs.per_vertex_copies ||
                  ((fs_copies->info.fs.per_vertex_bary[0] ||
                    fs_copies->info.fs.per_vertex_bary[1]) &&
                   !weights_by_vs);
   }

   /* If we don't need a GS but we do have a passthrough, unbind it */
   if (!needs_gs) {
      if (gs != NULL) {
         assert(gs->is_passthrough);
         hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, NULL);
      }
      return;
   }

   /* Else, we need to bind a passthrough GS */
   size_t key_size = poly_passthrough_gs_key_size(xfb_outputs);
   struct poly_passthrough_gs_key *key = alloca(key_size);

   *key = (struct poly_passthrough_gs_key){
      .prim = u_decomposed_prim(hk_gs_in_prim(cmd)),
      .outputs = last_sw->b.info.outputs,
      .clip_distance_array_size = last_sw->info.clip_distance_array_size,
      .cull_distance_array_size = last_sw->info.cull_distance_array_size,
   };

   for (uint32_t i = 0; i < NUM_TOTAL_VARYING_SLOTS; i++) {
      unsigned num_comps = 0;
      switch (i) {
         case VARYING_SLOT_TESS_LEVEL_OUTER:
         case VARYING_SLOT_TESS_LEVEL_INNER:
         case VARYING_SLOT_BOUNDING_BOX1:
            num_comps = 4;
            break;
         default:
            num_comps = nir_slot_num_components(i, MESA_SHADER_VERTEX);
            break;
      }
      key->output_components[i] = num_comps ?: 4;

      /* We don't care about VS varying types in AGX, just set everything to
       * uint32 to improve cache hits. */
      key->output_types[i] = nir_type_uint32;
   }

   if (xfb_outputs) {
      typed_memcpy(key->xfb_stride, last_sw->info.xfb_stride,
                   ARRAY_SIZE(key->xfb_stride));

      memcpy(&key->xfb_info, &last_sw->info.xfb_info,
             nir_xfb_info_size(xfb_outputs));
   }

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Binding passthrough GS for%s\n", xfb_outputs ? " XFB" : "");

   if (dev->dev.chip == AGX_CHIP_G16G) {
      struct hk_api_shader *fs_api = gfx->shaders[MESA_SHADER_FRAGMENT];
      struct hk_shader *fs = fs_api ? hk_only_variant(fs_api) : NULL;
      struct hk_apple9_passthrough_suffix suffix = {
         .fs_interp = {
            fs ? fs->info.fs.interp.flat : 0,
            fs ? fs->info.fs.interp.linear : 0,
         },
         .per_vertex_copies = fs ? fs->info.fs.per_vertex_copies : 0,
         .line_stipple_start = stipple_gs ? fs->info.fs.line_stipple_start : 0,
         .line_stipple_strip = stipple_gs && stipple_strip,
         .polygon_points = polygon_points,
         .polygon_face_slot = polygon_points ? fs->info.fs.polygon_face_slot : 0,
         .polygon_point_size = polygon_points &&
            hk_cmd_buffer_device(cmd)->vk.physical->properties.polygonModePointSize,
         .cull_mask = cull_mask,
      };
      if (fs) {
         memcpy(suffix.per_vertex_location, fs->info.fs.per_vertex_location,
                sizeof(suffix.per_vertex_location));
         memcpy(suffix.per_vertex_copy_base, fs->info.fs.per_vertex_copy_base,
                sizeof(suffix.per_vertex_copy_base));
         memcpy(suffix.per_vertex_bary, fs->info.fs.per_vertex_bary,
                sizeof(suffix.per_vertex_bary));
      }
      /* Without a fragment shader, the rasterized layout is unconstrained. */
      if (fs && fs != hk_only_variant(dev->null_fs))
         memcpy(suffix.components, fs->b.info.apple9_varyings.mask,
                sizeof(suffix.components));
      else
         memset(suffix.components, 0xff, sizeof(suffix.components));
      uint8_t *full = alloca(key_size + sizeof(suffix));
      memcpy(full, key, key_size);
      memcpy(full + key_size, &suffix, sizeof(suffix));
      gs = hk_meta_shader(dev, hk_apple9_passthrough_gs, full,
                          key_size + sizeof(suffix));
   } else {
      gs = hk_meta_shader(dev, poly_nir_passthrough_gs, key, key_size);
   }
   gs->is_passthrough = true;
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, gs);
}

static struct hk_cs *
hk_flush_gfx_state(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                   struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   if (!cs)
      return NULL;

   /* Annoyingly,
    * VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT is
    * render pass state on Imaginapple but draw state in Vulkan. In practice,
    * Proton never changes it within a render pass, but we technically need to
    * handle the switch regardless. Do so early since `cs` will be invalidated
    * if we need to split the render pass to switch representation mid-frame.
    *
    * Note we only do this dance with depth bias is actually enabled to avoid
    * senseless control stream splits with DXVK.
    */
   if ((IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(RS_DEPTH_BIAS_ENABLE)) &&
       dyn->rs.depth_bias.enable) {

      bool dbias_is_int =
         (dyn->rs.depth_bias.representation ==
          VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT) ||
         (gfx->render.depth_att.vk_format == VK_FORMAT_D16_UNORM);

      /* Attempt to set dbias_is_int per the draw requirement. If this fails,
       * flush the control stream and set it on the new control stream.
       */
      bool succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
      if (!succ) {
         perf_debug(cmd, "Splitting control stream due to depth bias");

         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);

         succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(succ && "can always set tri-state on a new control stream");
      }
   }

   cs = hk_occlusion_prepare_draw(cmd, cs);
   if (!cs)
      return NULL;

   hk_ensure_cs_has_space(cmd, cs, 0x2000 /* TODO */);

#ifndef NDEBUG
   if (unlikely(dev->dev.debug & AGX_DBG_DIRTY)) {
      hk_cmd_buffer_dirty_all(cmd);
   }
#endif

   /* Merge tess info before GS construction since that depends on
    * gfx->tess.prim
    */
   if ((IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL)) &&
       gfx->shaders[MESA_SHADER_TESS_CTRL]) {
      struct hk_api_shader *tcs = gfx->shaders[MESA_SHADER_TESS_CTRL];
      struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
      struct hk_shader *tese = hk_any_variant(tes);
      struct hk_shader *tesc = hk_only_variant(tcs);

      gfx->tess.info =
         hk_tess_info_merge(tese->info.tess.info, tesc->info.tess.info);

      /* Determine primitive based on the merged state */
      if (gfx->tess.info.points) {
         gfx->tess.prim = MESA_PRIM_POINTS;
      } else if (gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES) {
         gfx->tess.prim = MESA_PRIM_LINES;
      } else {
         gfx->tess.prim = MESA_PRIM_TRIANGLES;
      }
   }

   /* TODO: Try to reduce draw overhead of this */
   hk_handle_passthrough_gs(cmd, draw);

   hk_flush_shaders(cmd);

   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   if (draw.restart || gfx->shaders[MESA_SHADER_GEOMETRY])
      hk_flush_index(cmd, cs);

   hk_flush_dynamic_state(cmd, cs, draw_id, draw);
   return cs;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, VkDeviceSize size,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   cmd->state.gfx.index = (struct hk_index_buffer_state){
      .buffer = hk_buffer_addr_range(buffer, offset, size, true),
      .size = agx_translate_index_size(vk_index_type_to_bytes(indexType)),
      .restart = vk_index_to_restart(indexType),
   };

   /* TODO: check if necessary, blob does this. G16G bounds the fetch by the
    * exact extent; rounding would expose an out-of-range narrow index.
    */
   if (hk_cmd_buffer_device(cmd)->dev.chip != AGX_CHIP_G16G) {
      cmd->state.gfx.index.buffer.range =
         align(cmd->state.gfx.index.buffer.range, 4);
   }
}

void
hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                          struct hk_addr_range addr_range)
{
   cmd->state.gfx.vb[vb_idx] = addr_range;
   cmd->state.gfx.dirty |= HK_DIRTY_VB;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                         uint32_t bindingCount, const VkBuffer *pBuffers,
                         const VkDeviceSize *pOffsets,
                         const VkDeviceSize *pSizes,
                         const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pStrides) {
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount,
                                        pStrides);
   }

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;

      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
      const struct hk_addr_range addr_range =
         hk_buffer_addr_range(buffer, pOffsets[i], size, true);

      hk_cmd_bind_vertex_buffer(cmd, idx, addr_range);
   }
}

static bool
hk_set_view_index(struct hk_cmd_buffer *cmd, uint32_t view_idx)
{
   if (cmd->state.gfx.render.view_mask) {
      cmd->state.gfx.descriptors.root.draw.view_index = view_idx;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   return true;
}

/*
 * Iterator macro to duplicate a draw for each enabled view (when multiview is
 * enabled, else always view 0). Along with hk_lower_multiview, this forms the
 * world's worst multiview lowering.
 */
#define hk_foreach_view(cmd)                                                   \
   u_foreach_bit(view_idx, cmd->state.gfx.render.view_mask ?: 1)               \
      if (hk_set_view_index(cmd, view_idx))

/* Queue the input assembly counts of direct draws recorded since the last
 * flush. A dispatch per draw serializes the GPU on statistics queries that
 * span many draws, so they are summed when recorded instead, and queued when
 * the counters change and when the query ends.
 */
void
hk_flush_ia_stats(struct hk_cmd_buffer *cmd)
{
   struct hk_ia_stats *s = &cmd->state.gfx.ia_stats;
   const uint32_t counts[5] = {s->vertices, s->prims, s->vertices, s->clipped,
                               s->clipped};
   for (unsigned i = 0; i < ARRAY_SIZE(counts); ++i) {
      if (counts[i] && s->addr[i] && hk_stat_enabled(s->addr[i])) {
         struct libagx_imm_write add = {.address = s->addr[i],
                                        .value = counts[i]};
         util_dynarray_append(&cmd->query_adds, add);
      }
   }
   memset(s, 0, sizeof(*s));
}

static void
hk_ia_update(struct hk_cmd_buffer *cmd, struct agx_draw draw,
             uint64_t ia_vertices, uint64_t ia_prims, uint64_t vs_invocations,
             uint64_t c_prims, uint64_t c_inv)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
   uint32_t patch_size = dyn->ts.patch_control_points;

   if (!agx_is_indirect(draw.b) && !draw.restart) {
      struct hk_ia_stats *s = &cmd->state.gfx.ia_stats;
      const uint64_t addr[5] = {ia_vertices, ia_prims, vs_invocations,
                                c_prims, c_inv};
      if (memcmp(s->addr, addr, sizeof(addr))) {
         hk_flush_ia_stats(cmd);
         memcpy(s->addr, addr, sizeof(addr));
      }
      uint32_t prims = poly_decomposed_prims_for_vertices_with_tess(
                          prim, draw.b.count[0], patch_size) *
                       draw.b.count[1];
      s->vertices += draw.b.count[0] * draw.b.count[1];
      s->prims += prims;
      /* Clipper counters are written with the geometry or tessellation
       * output when those stages are used. */
      if (!geom && !tess)
         s->clipped += prims;
      return;
   }

   perf_debug(cmd, "Input assembly counters");

   uint64_t draw_ptr;
   if (agx_is_indirect(draw.b)) {
      draw_ptr = draw.b.ptr;
   } else {
      /* Direct draws pass their own indices (agx_draw_index_buffer) with a
       * range relative to the first index, so the descriptor's first index
       * (VkDrawIndexedIndirectCommand word 2) is 0. */
      uint32_t desc[] = {draw.b.count[0], draw.b.count[1], 0};
      draw_ptr = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   /* Clipper counters depend on geom/tess outputs and must be written with the
    * geom/tess output. They are updated as IA counters only when geom/tess is
    * not used.
    */
   if (geom || tess) {
      c_prims = 0;
      c_inv = 0;
   }

   if (draw.restart) {
      uint32_t index_size_B = agx_index_size_to_B(draw.index_size);

      libagx_increment_ia_restart(
         cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, ia_vertices, ia_prims,
         vs_invocations, c_prims, c_inv, draw_ptr, agx_draw_index_buffer(draw),
         agx_draw_index_range_el(draw), cmd->state.gfx.index.restart,
         index_size_B, prim, patch_size);
   } else {
      libagx_increment_ia(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                          ia_vertices, ia_prims, vs_invocations, c_prims, c_inv,
                          draw_ptr, prim, patch_size);
   }
}

/* Replace deferred Apple9 shaders with the variant for the current dynamic
 * state and attachments. The synthesized pipeline state carries exactly the
 * inputs of the Apple9 graphics key.
 */
/* Late depth/stencil tests for a feedback loop, enabled dynamically or by
 * the fragment shader's pipeline, only when the draw can write depth or
 * stencil (hk_apple9_zs_writes). Zink reports a loop whenever a depth texture
 * stays bound while it is the attachment; late export costs the early
 * rejection of hidden fragments. */
static bool
hk_apple9_draw_late_zs(const struct vk_dynamic_graphics_state *dyn,
                       const struct hk_rendering_state *render,
                       const struct hk_api_shader *fs)
{
   const bool dynamic = dyn->feedback_loops &
      (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   const bool created = hk_apple9_deferred_zs_self_dep(fs);
   if (!dynamic && !created)
      return false;
   return hk_apple9_zs_writes(&dyn->ds,
      (render->depth_att.iview ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
      (render->stencil_att.iview ? VK_IMAGE_ASPECT_STENCIL_BIT : 0));
}

static bool
hk_apple9_resolve_shaders(struct hk_cmd_buffer *cmd, bool prefer_native)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   /* Shader resolution consumes some dirty bits which also affect native
    * draw state. Invalidate that cache before resolution clears them. */
   if (vk_dynamic_graphics_state_any_dirty(&cmd->vk.dynamic_graphics_state))
      cmd->apple9_draw_state.valid = false;
   bool deferred = false;
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      deferred |= gfx->apple9_api[s] && gfx->apple9_api[s]->apple9_deferred;
   }
   if (!deferred)
      return true;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &gfx->render;

   /* Dynamic state read by the graphics key. Draw-state reuse was already
    * invalidated above, so resolution can consume these bits. */
   static const enum mesa_vk_dynamic_graphics_state inputs[] = {
      MESA_VK_DYNAMIC_VI,
      MESA_VK_DYNAMIC_VI_BINDINGS_VALID,
      MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY,
      MESA_VK_DYNAMIC_VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE,
      MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP,
      MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES,
      MESA_VK_DYNAMIC_MS_SAMPLE_MASK,
      MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE,
      MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE,
      MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE,
      MESA_VK_DYNAMIC_CB_LOGIC_OP,
      MESA_VK_DYNAMIC_CB_ATTACHMENT_COUNT,
      MESA_VK_DYNAMIC_CB_BLEND_ENABLES,
      MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS,
      MESA_VK_DYNAMIC_CB_WRITE_MASKS,
      MESA_VK_DYNAMIC_CB_BLEND_ADVANCED,
      MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE,
      MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE,
      MESA_VK_DYNAMIC_RS_POLYGON_MODE,
   };
   const bool late_zs = hk_apple9_draw_late_zs(dyn, render,
      gfx->apple9_api[MESA_SHADER_FRAGMENT]);
   /* Fragment shaders count invocations only while such a query is active;
    * other draws use variants without the counter. */
   const bool fs_statistics = gfx->descriptors.root.draw.pipeline_stats_flags &
      VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
   bool changed = !gfx->apple9_resolved.valid ||
                  gfx->apple9_resolved.prefer_native != prefer_native ||
                  gfx->apple9_resolved.fs_statistics != fs_statistics ||
                  gfx->apple9_resolved.late_zs != late_zs ||
                  gfx->apple9_resolved.render_gen != gfx->render_gen ||
                  memcmp(gfx->apple9_resolved.api, gfx->apple9_api,
                         sizeof(gfx->apple9_api));
   for (unsigned i = 0; i < ARRAY_SIZE(inputs); ++i)
      changed |= BITSET_TEST(dyn->dirty, inputs[i]);
   if (!changed) {
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
         if (gfx->apple9_api[s] || s == MESA_SHADER_FRAGMENT)
            hk_cmd_bind_graphics_shader(cmd, s, gfx->apple9_resolved.out[s]);
      }
      return true;
   }

   struct vk_multisample_state ms = {
      .rasterization_samples =
         dyn->ms.rasterization_samples
            ? dyn->ms.rasterization_samples
            : MAX2(render->tilebuffer.nr_samples, 1),
      .sample_mask = dyn->ms.sample_mask,
      .alpha_to_coverage_enable = dyn->ms.alpha_to_coverage_enable,
      .alpha_to_one_enable = dyn->ms.alpha_to_one_enable,
      .sample_shading_enable = hk_apple9_deferred_sample_shading(
         gfx->apple9_api[MESA_SHADER_FRAGMENT]),
   };
   struct vk_render_pass_state rp = {
      .color_attachment_count = render->color_att_count,
   };
   for (unsigned i = 0; i < render->color_att_count; ++i)
      rp.color_attachment_formats[i] = render->color_att[i].vk_format;
   /* Shader objects set blend state per attachment without a count. */
   struct vk_color_blend_state cb = dyn->cb;
   cb.attachment_count = MAX2(cb.attachment_count, render->color_att_count);

   struct vk_graphics_pipeline_state state = {
      .vi = dyn->vi,
      /* Mesh pipelines have no input assembly; their topology is unset. */
      .ia = gfx->apple9_api[MESA_SHADER_MESH] ? NULL : &dyn->ia,
      .vp = &dyn->vp,
      .rs = &dyn->rs,
      .ms = &ms,
      .cb = &cb,
      .rp = &rp,
      .cal = &dyn->cal,
      .pipeline_flags =
         late_zs ? VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT
                 : 0,
   };
   /* Color write enables are read at draw time from the sysval. */
   BITSET_SET(state.dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);

   struct hk_api_shader *out[HK_APPLE9_GFX_STAGES];
   uint64_t t_slow = hk_cpu_sub_begin();
   VkResult result = hk_apple9_resolve_graphics(dev, gfx->apple9_api, &state,
                                                out, prefer_native,
                                                fs_statistics);
   hk_cpu_sub_end(t_slow, HK_CPU_SUB_RESOLVE_SLOW);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return false;
   }
   /* Without an API fragment shader, a variant may supply the empty program
    * which applies cull distances; otherwise the device's is used. */
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      if (gfx->apple9_api[s] || s == MESA_SHADER_FRAGMENT)
         hk_cmd_bind_graphics_shader(cmd, s, out[s]);
   }
   gfx->apple9_resolved.valid = true;
   gfx->apple9_resolved.prefer_native = prefer_native;
   gfx->apple9_resolved.fs_statistics = fs_statistics;
   gfx->apple9_resolved.late_zs = late_zs;
   gfx->apple9_resolved.render_gen = gfx->render_gen;
   memcpy(gfx->apple9_resolved.api, gfx->apple9_api, sizeof(gfx->apple9_api));
   memcpy(gfx->apple9_resolved.out, out, sizeof(out));
   for (unsigned i = 0; i < ARRAY_SIZE(inputs); ++i)
      BITSET_CLEAR(dyn->dirty, inputs[i]);
   return true;
}

static struct agx_draw hk_apple9_predicate_draw(struct hk_cmd_buffer *cmd,
                                                struct agx_draw draw);

/* Start with one input triangle and one instance. Multiple GS invocations,
 * input primitives and instances need separate ordering qualification. */
static bool
hk_apple9_native_gs_state_eligible(struct hk_cmd_buffer *cmd,
                                  struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!hk_apple9_native_stage_available_device(hk_cmd_buffer_device(cmd),
                                                MESA_SHADER_GEOMETRY) || cmd->in_meta || agx_is_indirect(draw.b) ||
       draw.indexed || draw.restart || draw.start || draw.start_instance ||
       draw.b.count[0] != 3 || draw.b.count[1] != 1 || draw.b.count[2] != 1 ||
       dyn->ia.primitive_topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ||
       dyn->ia.primitive_restart_enable ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] || gfx->shaders[MESA_SHADER_TASK] ||
       gfx->shaders[MESA_SHADER_MESH] || gfx->render.view_mask ||
       gfx->render.layer_count != 1 || hk_cond_render_applies(cmd) ||
       gfx->descriptors.root.draw.pipeline_stats_flags ||
       gfx->occlusion.active_count || gfx->mesh_pg_query || gfx->xfb_enabled ||
       dyn->rs.rasterization_stream || dyn->rs.rasterizer_discard_enable ||
       dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
       dyn->rs.conservative_mode !=
          VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT ||
       MAX2(gfx->render.tilebuffer.nr_samples ?: dyn->ms.rasterization_samples,
            1) != 1)
      return false;
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i) {
      if (gfx->pg_query[i] || gfx->xfb_query[i])
         return false;
   }
   return true;
}

/* Stage 2 admits one direct patch and one instance only. */
static bool
hk_apple9_native_tess_state_eligible(struct hk_cmd_buffer *cmd,
                                     struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   if (!hk_apple9_native_stage_available_device(hk_cmd_buffer_device(cmd),
                                                MESA_SHADER_TESS_EVAL) || cmd->in_meta ||
       agx_is_indirect(draw.b) || draw.indexed || draw.restart || draw.start ||
       draw.start_instance || !dyn->ts.patch_control_points ||
       dyn->ts.patch_control_points > 32 ||
       draw.b.count[0] != dyn->ts.patch_control_points ||
       draw.b.count[1] != 1 || draw.b.count[2] != 1 ||
       dyn->ia.primitive_topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST ||
       dyn->ia.primitive_restart_enable ||
       !gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       !gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] || gfx->shaders[MESA_SHADER_TASK] ||
       gfx->shaders[MESA_SHADER_MESH] || gfx->render.view_mask ||
       gfx->render.layer_count != 1 || hk_cond_render_applies(cmd) ||
       gfx->descriptors.root.draw.pipeline_stats_flags ||
       gfx->occlusion.active_count || gfx->mesh_pg_query || gfx->xfb_enabled ||
       dyn->rs.rasterization_stream || dyn->rs.rasterizer_discard_enable ||
       dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
       dyn->rs.conservative_mode !=
          VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT ||
       MAX2(gfx->render.tilebuffer.nr_samples ?: dyn->ms.rasterization_samples,
            1) != 1)
      return false;
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->pg_query); ++i) {
      if (gfx->pg_query[i] || gfx->xfb_query[i])
         return false;
   }
   return true;
}

bool
hk_apple9_use_native_tess(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (!hk_apple9_native_tess_state_eligible(cmd, draw))
      return false;
   struct hk_api_shader *tes =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_TESS_EVAL]);
   if (!tes || !tes->apple9_native_leaf ||
       (!hk_apple9_native_tess_enabled() &&
        tes->apple9_native_stage_mask != HK_APPLE9_NATIVE_TES_BIT))
      return false;
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX,
      MESA_SHADER_TESS_CTRL,
      MESA_SHADER_FRAGMENT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      struct hk_api_shader *api = gfx->shaders[stages[i]];
      if (stages[i] == MESA_SHADER_FRAGMENT &&
          (!api || api == hk_cmd_buffer_device(cmd)->null_fs))
         continue;
      api = hk_api_shader_native(api);
      if (!api || !api->apple9_native_leaf ||
          (stages[i] != MESA_SHADER_FRAGMENT && api->writes_memory) ||
          api->apple9_native_stage_mask != tes->apple9_native_stage_mask ||
          memcmp(tes->apple9_native_pair_id, api->apple9_native_pair_id,
                 sizeof(tes->apple9_native_pair_id)))
         return false;
   }
   const struct hk_shader *main = hk_native_producer_main(tes);
   const struct agx_apple9_mesh_body_info *body = &main->info.apple9_native;
   return main->info.apple9_native_stage == HK_APPLE9_NATIVE_MESH &&
          hk_native_producer_object(tes)->info.apple9_native_stage ==
             HK_APPLE9_NATIVE_OBJECT &&
          hk_apple9_native_stage_class(
             MESA_SHADER_TESS_EVAL, body->key.physical_threads,
             body->layout.max_vertices, body->layout.max_primitives) &&
          hk_apple9_packed_mesh_class_selected(
             hk_apple9_native_mesh_packed_shapes(), body->key.physical_threads,
             body->layout.max_vertices, body->layout.max_primitives);
}

DEBUG_GET_ONCE_BOOL_OPTION(tess_route_log, "HK_APPLE9_TESS_ROUTE_LOG", false)

static void
hk_apple9_launch_native_tess(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                             struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   if (!hk_apple9_native_lanes(hk_cmd_buffer_device(cmd)))
      hk_tess_batch_flush(cmd);
   hk_apple9_merge_tess_info(cmd, true);
   desc->root.draw.geometry_params = 0;
   desc->root.draw.api_gs = false;
   desc->root.draw.rasterization_stream = 0;
   desc->root.draw.draw_id = draw_id;
   desc->root_dirty = true;
   hk_apple9_launch_tess(cmd, draw, true);
   /* The render consumes both the count written by prefix sum and the
    * indices/coordinates written by emit. HK_READS_INTERNAL in prepare_draw
    * gives it the pre_gfx tiling dependency, including merge continuations. */
   if (!vk_command_buffer_has_error(&cmd->vk))
      hk_apple9_draw_native_mesh(cmd, draw_id, gfx->tess.out_draws);
}

bool
hk_apple9_use_native_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]);
   struct hk_api_shader *vs =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_VERTEX]);
   if (!hk_apple9_native_gs_state_eligible(cmd, draw) || !gs || !vs ||
       gfx->shaders[MESA_SHADER_GEOMETRY]->is_passthrough ||
       gs->is_passthrough || !gs->apple9_native_leaf ||
       !vs->apple9_native_leaf || vs->writes_memory ||
       (!hk_apple9_native_gs_enabled() &&
        gs->apple9_native_stage_mask != HK_APPLE9_NATIVE_GS_BIT) ||
       vs->apple9_native_stage_mask != gs->apple9_native_stage_mask ||
       memcmp(gs->apple9_native_pair_id, vs->apple9_native_pair_id,
              sizeof(gs->apple9_native_pair_id)))
      return false;
   const struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (fs && fs != hk_cmd_buffer_device(cmd)->null_fs) {
      fs = hk_api_shader_native(fs);
      if (!fs || !fs->apple9_native_leaf ||
          fs->apple9_native_stage_mask != gs->apple9_native_stage_mask ||
          memcmp(gs->apple9_native_pair_id, fs->apple9_native_pair_id,
                 sizeof(gs->apple9_native_pair_id)))
         return false;
   }
   const struct hk_shader *main = hk_native_producer_main(gs);
   const struct agx_apple9_mesh_body_info *body = &main->info.apple9_native;
   return main->info.apple9_native_stage == HK_APPLE9_NATIVE_MESH &&
          hk_native_producer_object(gs)->info.apple9_native_stage ==
             HK_APPLE9_NATIVE_OBJECT &&
          (hk_apple9_native_gs_enabled() || hk_apple9_native_stage_class(
             MESA_SHADER_GEOMETRY, body->key.physical_threads,
             body->layout.max_vertices, body->layout.max_primitives)) &&
          hk_apple9_packed_mesh_class_selected(
             hk_apple9_native_mesh_packed_shapes(), body->key.physical_threads,
             body->layout.max_vertices, body->layout.max_primitives);
}

DEBUG_GET_ONCE_BOOL_OPTION(gs_route_log, "HK_APPLE9_GS_ROUTE_LOG", false)

static void
hk_apple9_launch_native_gs(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                           struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct hk_api_shader *vs_api =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_VERTEX]);
   struct hk_shader *vs = &vs_api->variants[HK_VS_VARIANT_SW];
   /* Complete any previously captured pre-raster lanes before this VS.
    * The existing pre_gfx stream is a dependency of the isolated render;
    * HK_READS_INTERNAL keeps the wait at tiling, never fragment-only.
    * In a batched native render the VS joins lane 0 instead (see
    * hk_apple9_native_lanes). */
   bool lanes = hk_apple9_native_lanes(hk_cmd_buffer_device(cmd)) &&
                !agx_is_indirect(draw.b) &&
                !desc->root.draw.pipeline_stats_flags;
   if (!lanes)
      hk_tess_batch_flush(cmd);
   struct hk_apple9_prerast_bindings bindings = hk_apple9_draw_bindings(cmd, draw);
   desc->root.draw.vertex_params = hk_upload_vertex_params_for_vs(cmd, draw, vs);
   desc->root.draw.geometry_params = 0;
   desc->root.draw.api_gs = true;
   desc->root.draw.rasterization_stream = 0;
   desc->root.draw.draw_id = draw_id;
   desc->root_dirty = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   cmd->tess_batch.capture = lanes ? &cmd->tess_batch.lane[0] : NULL;
   hk_apple9_dispatch_prerast(cmd, vs, draw.b, &bindings);
   cmd->tess_batch.capture = NULL;
   if (lanes)
      cmd->tess_batch.draws++;
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   const uint32_t groups[] = {1, 1, 1};
   uint64_t grid = hk_pool_upload(cmd, groups, sizeof(groups), 4);
   if (grid)
      hk_apple9_draw_native_mesh(cmd, draw_id, grid);
}

static void
hk_draw(struct hk_cmd_buffer *cmd, uint16_t draw_id, struct agx_draw draw_)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* Filter trivial draws so we don't need to worry about null index buffers */
   if (!agx_is_indirect(draw_.b) &&
       (draw_.b.count[0] == 0 || draw_.b.count[1] == 0))
      return;

   if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G) {
      if (!hk_feedback_prepare_draw(cmd))
         return;
      /* Batched native renders defer the ordinary split until stage
       * selection: a native GS/TES draw may continue the render. */
      bool defer_prepare =
         hk_apple9_native_blocks_enabled(hk_cmd_buffer_device(cmd)) &&
         cmd->current_cs.gfx && cmd->current_cs.gfx->apple9_native_mesh;
      /* Index fetch and indirect arguments are buffer reads of tiling. */
      if (!defer_prepare &&
          !hk_apple9_prepare_draw(cmd, agx_is_indirect(draw_.b) || draw_.indexed, false))
         return;
      uint64_t t_resolve = hk_cpu_sub_begin();
      const struct hk_graphics_state *gfx = &cmd->state.gfx;
      mesa_shader_stage producer = gfx->shaders[MESA_SHADER_GEOMETRY]
                                      ? MESA_SHADER_GEOMETRY
                                      : MESA_SHADER_TESS_EVAL;
      const struct hk_api_shader *source =
         gfx->apple9_api[producer] ?: gfx->shaders[producer];
      bool prefer_native =
         source && !source->is_passthrough &&
         p_atomic_read(&source->apple9_native_candidate) &&
         (producer == MESA_SHADER_GEOMETRY
             ? hk_apple9_native_gs_state_eligible(cmd, draw_)
             : hk_apple9_native_tess_state_eligible(cmd, draw_));
      bool resolved = hk_apple9_resolve_shaders(cmd, prefer_native);
      hk_cpu_sub_end(t_resolve, HK_CPU_SUB_RESOLVE);
      if (!resolved)
         return;
      if (cmd->feedback_raw.size) {
         for (unsigned stage = 0; stage <= MESA_SHADER_MESH; stage++) {
            if (cmd->state.gfx.shaders[stage] &&
                cmd->state.gfx.shaders[stage]->writes_memory) {
               hk_feedback_invalidate(cmd);
               break;
            }
         }
      }
      bool native_gs = hk_apple9_use_native_gs(cmd, draw_);
      bool native_tess = hk_apple9_use_native_tess(cmd, draw_);
      if (!native_gs && !native_tess && gfx->shaders[producer] &&
          gfx->shaders[producer]->apple9_native_leaf) {
         if (!hk_apple9_resolve_shaders(cmd, false))
            return;
      }
      if (producer == MESA_SHADER_GEOMETRY && source &&
          debug_get_option_gs_route_log()) {
         fprintf(stderr, "HK_GS_SELECT candidate=%u prefer=%u state=%u leaf=%u vs_leaf=%u vs_writes=%u\n",
                 p_atomic_read(&source->apple9_native_candidate), prefer_native,
                 hk_apple9_native_gs_state_eligible(cmd, draw_),
                 !!hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]),
                 !!hk_api_shader_native(gfx->shaders[MESA_SHADER_VERTEX]),
                 gfx->shaders[MESA_SHADER_VERTEX]->writes_memory);
         struct hk_api_shader *gs = native_gs
            ? hk_api_shader_native(gfx->shaders[MESA_SHADER_GEOMETRY]) : NULL;
         const struct agx_apple9_mesh_body_info *body =
            gs ? &hk_native_producer_main(gs)->info.apple9_native : NULL;
         fprintf(stderr, "HK_GS_ROUTE native=%u T=%u V=%u P=%u\n", native_gs,
                 body ? body->key.physical_threads : 0,
                 body ? body->layout.max_vertices : 0,
                 body ? body->layout.max_primitives : 0);
      }
      if (native_gs) {
         if (hk_apple9_prepare_draw(cmd, true, true))
            hk_apple9_launch_native_gs(cmd, draw_id, draw_);
         return;
      }
      if (producer == MESA_SHADER_TESS_EVAL && source &&
          debug_get_option_tess_route_log())
         fprintf(stderr, "HK_TESS_ROUTE native=%u candidate=%u prefer=%u\n",
                 native_tess, p_atomic_read(&source->apple9_native_candidate),
                 prefer_native);
      if (native_tess) {
         if (hk_apple9_prepare_draw(cmd, true, true))
            hk_apple9_launch_native_tess(cmd, draw_id, draw_);
         return;
      }
      if (defer_prepare &&
          !hk_apple9_prepare_draw(cmd, agx_is_indirect(draw_.b) || draw_.indexed, false))
         return;
      if (hk_cond_render_applies(cmd)) {
         draw_ = hk_apple9_predicate_draw(cmd, draw_);
         if (!draw_.b.ptr)
            return;
      }
      draw_.restart = dyn->ia.primitive_restart_enable && draw_.indexed;
      draw_.index_size = cmd->state.gfx.index.size;
      uint64_t ia_verts = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT);
      uint64_t ia_prims = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT);
      uint64_t vs_inv = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT);
      uint64_t c_inv = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);
      uint64_t c_prims = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);
      if (hk_stat_enabled(ia_verts) || hk_stat_enabled(ia_prims) ||
          hk_stat_enabled(vs_inv) || hk_stat_enabled(c_inv) ||
          hk_stat_enabled(c_prims)) {
         hk_ia_update(cmd, draw_, ia_verts, ia_prims, vs_inv, c_prims, c_inv);
      }
      if (cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL])
         hk_apple9_merge_tess_info(cmd, false);
      hk_handle_passthrough_gs(cmd, draw_);
      if (!hk_sw_vs_outputs_fit(cmd, draw_))
         return;
      /* Geometry and tessellation emulation decompose fans themselves. */
      cmd->state.gfx.apple9_fan_unrolled =
         dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT &&
         !cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] &&
         !cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
      if (cmd->state.gfx.apple9_fan_unrolled)
         draw_ = hk_apple9_unroll_fan(cmd, draw_);
      /* Software stages read the draw ID from the root table they upload,
       * when one of the bound stages they run reads it. */
      if (hk_apple9_reads_draw_id(&cmd->state.gfx) &&
          cmd->state.gfx.descriptors.root.draw.draw_id != draw_id) {
         cmd->state.gfx.descriptors.root.draw.draw_id = draw_id;
         cmd->state.gfx.descriptors.root_dirty = true;
      }
      bool adjacency =
         !cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] &&
         !cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] &&
         mesa_prim_has_adjacency(vk_topology_to_mesa(dyn->ia.primitive_topology));
      hk_foreach_view(cmd) {
         struct agx_draw draw = draw_;
         if (adjacency)
            draw = hk_apple9_draw_without_adj(cmd, draw);
         else if (!cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] &&
                  !cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] &&
                  hk_apple9_needs_index_copy(cmd, draw)) {
            uint64_t t = hk_cpu_sub_begin();
            draw = hk_apple9_robust_index(cmd, draw);
            hk_cpu_sub_end(t, HK_CPU_SUB_ROBUST_INDEX);
         }
         if (cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL]) {
            uint64_t t = hk_cpu_sub_begin();
            draw = hk_apple9_launch_tess(cmd, draw, false);
            hk_cpu_sub_end(t, HK_CPU_SUB_TESS);
         }
         if (cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY]) {
            uint64_t t = hk_cpu_sub_begin();
            draw = hk_apple9_launch_gs(cmd, draw);
            hk_cpu_sub_end(t, HK_CPU_SUB_GS);
         }
         uint64_t t = hk_cpu_sub_begin();
         hk_apple9_draw(cmd, draw_id, draw);
         hk_cpu_sub_end(t, HK_CPU_SUB_A9DRAW);
      }
      return;
   }

   draw_.restart = dyn->ia.primitive_restart_enable && draw_.indexed;
   draw_.index_size = cmd->state.gfx.index.size;

   uint64_t stat_ia_verts = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT);

   uint64_t stat_ia_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT);

   uint64_t stat_vs_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT);

   uint64_t stat_c_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);

   uint64_t stat_c_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);

   bool ia_stats = hk_stat_enabled(stat_ia_verts) ||
                   hk_stat_enabled(stat_ia_prims) ||
                   hk_stat_enabled(stat_vs_inv) ||
                   hk_stat_enabled(stat_c_inv) || hk_stat_enabled(stat_c_prims);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   hk_foreach_view(cmd) {
      struct agx_draw draw = draw_;
      struct hk_cs *cs = hk_flush_gfx_state(cmd, draw_id, draw);
      /* If we failed to allocate a control stream, we've already lost the
       * device. Just drop the draw so we don't crash.
       */
      if (!cs)
         return;

      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
      bool needs_idx_robust = hk_needs_index_robustness(cmd, &draw);
      bool adj = mesa_prim_has_adjacency(
         vk_topology_to_mesa(dyn->ia.primitive_topology));
      adj &= !geom;
      needs_idx_robust &= !adj;

      struct hk_cs *ccs = NULL;
      uint8_t *out = cs->current;
      assert(cs->current + 0x1000 < cs->end);

      if (tess && HK_PERF(dev, NOTESS))
         continue;

      cs->stats.calls++;

      if (geom || tess) {
         ccs =
            hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true, HK_SYNC_ALL);
         if (!ccs)
            return;
      }

      if (ia_stats) {
         hk_ia_update(cmd, draw, stat_ia_verts, stat_ia_prims, stat_vs_inv,
                      stat_c_prims, stat_c_inv);
      }

      if (tess) {
         draw = hk_launch_tess(cmd, ccs, draw, geom ? 0 : stat_c_prims,
                               geom ? 0 : stat_c_inv);
      }

      if (geom) {
         draw = hk_launch_gs_prerast(cmd, ccs, draw);
      }

      if (adj) {
         assert(!geom && "geometry shaders handle adj directly");
         enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

         if (draw.restart) {
            draw = hk_draw_without_restart(cmd, draw, 1);
            prim = u_decomposed_prim(prim);
         }

         if (agx_is_indirect(draw.b)) {
            const size_t size = sizeof(VkDrawIndexedIndirectCommand);
            static_assert(sizeof(VkDrawIndexedIndirectCommand) >
                             sizeof(VkDrawIndirectCommand),
                          "allocation size is conservative");

            uint64_t out_draw = hk_pool_alloc(cmd, size, 4).gpu;
            struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

            libagx_draw_without_adj(
               cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw,
               draw.b.ptr, desc->root.draw.vertex_params, draw.index_buffer,
               draw.indexed ? agx_draw_index_range_el(draw) : 0,
               draw.indexed ? agx_index_size_to_B(draw.index_size) : 0, prim);

            draw = agx_draw_indirect(out_draw);
         } else {
            unsigned count = libagx_remap_adj_count(draw.b.count[0], prim);

            draw = (struct agx_draw){
               .b = agx_3d(count, draw.b.count[1], 1),
            };
         }
      }

      enum agx_primitive topology = cmd->state.gfx.topology;
      if (needs_idx_robust) {
         assert(!geom && !tess && !adj);
         perf_debug(cmd, "lowering robust index buffer");

         cs->current = out;

         draw = hk_draw_as_indexed_indirect(cmd, draw);

         size_t size_B = libagx_draw_robust_index_vdm_size();
         uint64_t target = hk_cs_alloc_for_indirect(cs, size_B);

         libagx_draw_robust_index(cmd, agx_1d(32), AGX_BARRIER_ALL | AGX_PREGFX,
                                  target, hk_heap(cmd), draw.b.ptr,
                                  draw.index_buffer, draw.index_buffer_range_B,
                                  draw.restart, topology, draw.index_size);
      } else {
         cs->current = (void *)agx_vdm_draw((uint32_t *)out, dev->dev.chip,
                                            draw, topology);
      }

      if (cs->primitive_count < AGX_PRIMITIVE_COUNT_UNKNOWN) {
         agx_add_primitive_count(
            &cs->primitive_count,
            agx_is_indirect(draw.b) || draw.restart || geom || tess || needs_idx_robust
               ? AGX_PRIMITIVE_COUNT_UNKNOWN
               : agx_count_primitives(topology, draw.b.count[0], draw.b.count[1]));
      }
      cs->stats.cmds++;
   }
}

static void
hk_init_mesh_heap(const void *data)
{
   struct hk_cmd_buffer *cmd = (struct hk_cmd_buffer *)data;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   size_t size = 512 * 1024 * 1024;
   dev->mesh_heap = agx_bo_create(&dev->dev, size, 0, 0, "Mesh heap");

   off_t off = dev->rodata.mesh_heap - dev->rodata.bo->va->addr;
   struct poly_heap *map = agx_bo_map(dev->rodata.bo) + off;
   *map = (struct poly_heap){
      .base = dev->mesh_heap ? dev->mesh_heap->va->addr : 0,
      .size = dev->mesh_heap ? size : 0,
   };
}

/* Records of mesh shader draws sized on the GPU are allocated from a heap,
 * freed after each command buffer.
 */
static uint64_t
hk_mesh_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   util_call_once_data(&dev->mesh_heap_init_once, hk_init_mesh_heap, cmd);
   if (!dev->mesh_heap)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   if (!cmd->uses_mesh_heap) {
      hk_queue_write(cmd, dev->rodata.mesh_heap + offsetof(struct poly_heap, bottom),
                     0, true /* after gfx */);
      cmd->uses_mesh_heap = true;
   }

   return dev->rodata.mesh_heap;
}

static uint64_t
hk_mesh_stat(struct hk_cmd_buffer *cmd, VkQueryPipelineStatisticFlagBits stat)
{
   uint64_t addr = hk_pipeline_stat_addr(cmd, stat);
   return hk_stat_enabled(addr) ? addr : 0;
}

/* Check state before resolving a deferred shader, so an ineligible draw only
 * compiles emulation. The final check also verifies the concrete program pair. */
static bool
hk_apple9_native_mesh_state_eligible(struct hk_cmd_buffer *cmd,
                                     const uint32_t *direct,
                                     uint64_t count_addr, bool cull_primitive)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!hk_apple9_native_mesh_available_device(hk_cmd_buffer_device(cmd)) ||
       !direct || count_addr ||
       !hk_apple9_native_mesh_grid_valid(direct) ||
       gfx->shaders[MESA_SHADER_TASK] || gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] || gfx->render.view_mask ||
       gfx->render.layer_count != 1 || hk_cond_render_applies(cmd) ||
       gfx->mesh_pg_query || gfx->descriptors.root.draw.pipeline_stats_flags ||
       gfx->occlusion.active_count || gfx->xfb_enabled ||
       dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
       MAX2(gfx->render.tilebuffer.nr_samples ?: dyn->ms.rasterization_samples,
            1) != 1)
      return false;

   /* Primitive and cull-distance culling use degenerate indices, which
    * conservative rasterization may still cover. */
   return !cull_primitive || dyn->rs.conservative_mode ==
                                VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT;
}

bool
hk_apple9_use_native_mesh(struct hk_cmd_buffer *cmd, const uint32_t *direct,
                          uint64_t count_addr)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_api_shader *mesh =
      hk_api_shader_native(gfx->shaders[MESA_SHADER_MESH]);
   const struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (!mesh)
      return false;
   if (fs && fs != hk_cmd_buffer_device(cmd)->null_fs) {
      fs = hk_api_shader_native(fs);
      if (!fs || !fs->apple9_native_leaf || !mesh->apple9_native_leaf ||
          memcmp(fs->apple9_native_pair_id, mesh->apple9_native_pair_id,
                 sizeof(mesh->apple9_native_pair_id)))
         return false;
   }
   const struct hk_shader *main = &mesh->variants[HK_MESH_VARIANT_MAIN];
   if (main->info.apple9_native_stage != HK_APPLE9_NATIVE_MESH)
      return false;
   return hk_apple9_native_mesh_state_eligible(
      cmd, direct, count_addr,
      (main->info.apple9_native.primitive_broadcast &
         VARYING_BIT_CULL_PRIMITIVE) ||
      main->info.apple9_native.cull_distances);
}

/* Mesh shader draws run the task and mesh workgroups as compute kernels, and
 * then draw every primitive slot of the mesh workgroups' output records. The
 * workgroup grid is either given (direct) or read from grid_addr. For draws
 * from a count buffer, draws past the count at count_addr are empty.
 */
static void
hk_apple9_draw_mesh(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                    const uint32_t *direct, uint64_t grid_addr,
                    uint64_t count_addr)
{
   /* Includes the hybrid native path: static pipeline feedback may change
    * after BeginRendering, and a failed decompression poisons recording. */
   if (!hk_feedback_prepare_draw(cmd))
      return;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;

   if (direct && (!direct[0] || !direct[1] || !direct[2]))
      return;

   const struct hk_api_shader *source =
      gfx->apple9_api[MESA_SHADER_MESH] ?: gfx->shaders[MESA_SHADER_MESH];
   bool prefer_native =
      source && p_atomic_read(&source->apple9_native_candidate) &&
      hk_apple9_native_mesh_state_eligible(
         cmd, direct, count_addr, source->apple9_native_cull_primitive);
   if (vk_command_buffer_has_error(&cmd->vk) ||
       !hk_apple9_resolve_shaders(cmd, prefer_native))
      return;

   struct hk_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];
   struct hk_api_shader *task = gfx->shaders[MESA_SHADER_TASK];
   if (!mesh)
      return;

   bool native = hk_apple9_use_native_mesh(cmd, direct, count_addr);
   if (!native && (mesh->apple9_native_leaf ||
                   (gfx->shaders[MESA_SHADER_FRAGMENT] &&
                    gfx->shaders[MESA_SHADER_FRAGMENT]->apple9_native_leaf))) {
      /* A native-only deferred result cannot serve an emulated draw. This
       * also makes a mismatched cached pair fall back before any work is
       * emitted, instead of failing the render-interface link at recording. */
      if (!hk_apple9_resolve_shaders(cmd, false))
         return;
      mesh = gfx->shaders[MESA_SHADER_MESH];
      task = gfx->shaders[MESA_SHADER_TASK];
      assert(mesh && !mesh->apple9_native_leaf);
   }
   if (!hk_apple9_prepare_draw(cmd, native || !direct || count_addr, native))
      return;
   struct hk_shader *main = &(native ? hk_api_shader_native(mesh) : mesh)
                                ->variants[HK_MESH_VARIANT_MAIN];
   if (debug_get_option_mesh_route_log()) {
      fprintf(stderr, "HK_MESH_ROUTE native=%u T=%u V=%u P=%u\n", native,
              native ? main->info.apple9_native.key.physical_threads : 0,
              main->info.mesh.max_vertices, main->info.mesh.max_primitives);
   }
   if (native) {
      /* One object workgroup publishes the nonempty direct child grid. */
      uint64_t grid = hk_pool_upload(cmd, direct, 3 * sizeof(uint32_t), 4);
      if (grid)
         hk_apple9_draw_native_mesh(cmd, draw_id, grid);
      return;
   }
   struct hk_shader *task_main = task ? hk_only_variant(task) : NULL;
   const struct hk_mesh_info *info = &main->info.mesh;
   unsigned vertices = MAX2(info->max_primitives, 1) *
                       (info->polygon_points ? HK_PP_MAX_VERTICES
                                             : mesa_vertices_per_prim(info->prim));
   struct hk_apple9_prerast_bindings bindings = {0};

   /* Draws without a task shader and with a known grid are set up here.
    * Predicated draws read a grid of zeroes when skipped.
    */
   bool predicated = hk_cond_render_applies(cmd);
   bool cpu_setup = direct && !task && !predicated;
   uint64_t direct_count =
      direct ? (uint64_t)direct[0] * direct[1] * direct[2] : 0;
   uint64_t records_B = direct_count * info->group_stride_B;
   if (cpu_setup && records_B > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   if (direct && !cpu_setup)
      grid_addr = hk_pool_upload(cmd, direct, 3 * sizeof(uint32_t), 4);
   if (predicated)
      grid_addr = hk_predicate_indirect(cmd, grid_addr, 3, true);
   /* Upload or predication failed: the command buffer is in error. */
   if (!cpu_setup && !grid_addr)
      return;

   desc->root.draw.draw_id = draw_id;

   /* Compute before graphics runs ahead of the whole render, so every
    * draw has its own records.
    */
   hk_foreach_view(cmd) {
      struct agx_ptr params =
         hk_pool_alloc(cmd, sizeof(struct libagx_mesh_params), 16);
      if (!params.gpu)
         return;

      struct libagx_mesh_params *p = params.cpu;
      *p = (struct libagx_mesh_params){
         .mesh_primitives = gfx->mesh_pg_query,
         .mesh_invocations = hk_mesh_stat(
            cmd, VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT),
         .task_invocations = hk_mesh_stat(
            cmd, VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT),
      };
      if (cpu_setup) {
         p->outputs = hk_pool_alloc(cmd, records_B, 16).gpu;
         p->mesh_count = direct_count;
      }
      for (unsigned v = 0; v < ARRAY_SIZE(p->primitive_varying_B); ++v) {
         p->primitive_varying_B[v] = hk_mesh_slot_offset_B(
            info, info->primitive_slots, VARYING_SLOT_VAR0 + v);
      }

      desc->root.draw.mesh_params = params.gpu;
      desc->root_dirty = true;
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

      if (!cpu_setup) {
         libagx_mesh_setup(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                           params.gpu, grid_addr, count_addr, draw_id,
                           hk_mesh_heap(cmd), task != NULL,
                           task_main ? task_main->info.task.payload_B : 0,
                           info->group_stride_B, vertices);
      }

      if (task_main) {
         hk_apple9_dispatch_prerast(
            cmd, task_main,
            agx_grid_indirect(params.gpu +
                              offsetof(struct libagx_mesh_params, task_dispatch)),
            &bindings);
         libagx_mesh_prefix(cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                            params.gpu, hk_mesh_heap(cmd), info->group_stride_B,
                            vertices);
      }

      struct agx_draw draw;
      if (cpu_setup) {
         hk_apple9_dispatch_prerast(
            cmd, main,
            agx_3d(direct[0] * main->info.apple9.local_size[0],
                   direct[1] * main->info.apple9.local_size[1],
                   direct[2] * main->info.apple9.local_size[2]),
            &bindings);
         draw = (struct agx_draw){.b = agx_3d(direct_count * vertices, 1, 1)};
      } else {
         hk_apple9_dispatch_prerast(
            cmd, main,
            agx_grid_indirect(params.gpu +
                              offsetof(struct libagx_mesh_params, mesh_dispatch)),
            &bindings);
         draw = agx_draw_indirect(params.gpu +
                                  offsetof(struct libagx_mesh_params, draw));
      }

      hk_apple9_draw(cmd, draw_id, draw);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX,
                       uint32_t groupCountY, uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   const uint32_t groups[3] = {groupCountX, groupCountY, groupCountZ};

   hk_apple9_draw_mesh(cmd, 0, groups, 0, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, uint32_t drawCount,
                               uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   for (uint32_t i = 0; i < drawCount; ++i) {
      hk_apple9_draw_mesh(cmd, i, NULL,
                          hk_buffer_address_ro(buffer, offset + i * stride), 0);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer,
                                    VkBuffer _buffer, VkDeviceSize offset,
                                    VkBuffer countBuffer,
                                    VkDeviceSize countBufferOffset,
                                    uint32_t maxDrawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   uint64_t count = hk_buffer_address_ro(count_buffer, countBufferOffset);
   for (uint32_t i = 0; i < maxDrawCount; ++i) {
      hk_apple9_draw_mesh(cmd, i, NULL,
                          hk_buffer_address_ro(buffer, offset + i * stride),
                          count);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;

   if (HK_TEST_INDIRECTS) {
      uint32_t data[] = {
         vertexCount,
         instanceCount,
         firstVertex,
         firstInstance,
      };

      draw = agx_draw_indirect(hk_pool_upload(cmd, data, sizeof(data), 4));
   } else {
      draw = (struct agx_draw){
         .b = agx_3d(vertexCount, instanceCount, 1),
         .start = firstVertex,
         .start_instance = firstInstance,
      };
   }

   hk_draw(cmd, 0, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                   const VkMultiDrawInfoEXT *pVertexInfo,
                   uint32_t instanceCount, uint32_t firstInstance,
                   uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   for (unsigned i = 0; i < drawCount; ++i) {
      struct agx_draw draw = {
         .b = agx_3d(pVertexInfo->vertexCount, instanceCount, 1),
         .start = pVertexInfo->firstVertex,
         .start_instance = firstInstance,
      };

      hk_draw(cmd, i, draw);
      pVertexInfo = ((void *)pVertexInfo) + stride;
   }
}

static void
hk_draw_indexed(VkCommandBuffer commandBuffer, uint16_t draw_id,
                uint32_t indexCount, uint32_t instanceCount,
                uint32_t firstIndex, int32_t vertexOffset,
                uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;
   struct hk_addr_range buf = cmd->state.gfx.index.buffer;

   if (HK_TEST_INDIRECTS && draw_id == 0) {
      uint32_t data[] = {
         indexCount, instanceCount, firstIndex, vertexOffset, firstInstance,
      };
      uint64_t addr = hk_pool_upload(cmd, data, sizeof(data), 4);

      draw = agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0);
   } else {
      draw =
         agx_draw_indexed(indexCount, instanceCount, firstIndex, vertexOffset,
                          firstInstance, buf.addr, buf.range, 0, 0);
   }

   hk_draw(cmd, draw_id, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                  uint32_t instanceCount, uint32_t firstIndex,
                  int32_t vertexOffset, uint32_t firstInstance)
{
   hk_draw_indexed(commandBuffer, 0, indexCount, instanceCount, firstIndex,
                   vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                          const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                          uint32_t instanceCount, uint32_t firstInstance,
                          uint32_t stride, const int32_t *pVertexOffset)
{
   for (unsigned i = 0; i < drawCount; ++i) {
      const uint32_t vertex_offset =
         pVertexOffset != NULL ? *pVertexOffset : pIndexInfo->vertexOffset;

      hk_draw_indexed(commandBuffer, i, pIndexInfo->indexCount, instanceCount,
                      pIndexInfo->firstIndex, vertex_offset, firstInstance);

      pIndexInfo = ((void *)pIndexInfo) + stride;
   }
}

static void
hk_draw_indirect_inner(VkCommandBuffer commandBuffer, uint64_t base,
                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndirect-drawCount-00476
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to sizeof(VkDrawIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = base + stride * draw_id;
      hk_draw(cmd, draw_id, agx_draw_indirect(addr));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                   VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indirect_inner(commandBuffer, hk_buffer_address_ro(buffer, offset),
                          drawCount, stride);
}

static void
hk_draw_indexed_indirect_inner(VkCommandBuffer commandBuffer, uint64_t buffer,
                               uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndexedIndirect-drawCount-00528
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to
    * sizeof(VkDrawIndexedIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndexedIndirectCommand));
   }

   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = buffer + stride * draw_id;
      struct hk_addr_range buf = cmd->state.gfx.index.buffer;

      hk_draw(cmd, draw_id,
              agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, uint32_t drawCount,
                          uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indexed_indirect_inner(
      commandBuffer, hk_buffer_address_ro(buffer, offset), drawCount, stride);
}

/*
 * To implement drawIndirectCount generically, we dispatch a compute kernel to
 * patch the indirect buffer and then we dispatch the predicated maxDrawCount
 * indirect draws.
 */
static void
hk_draw_indirect_count(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset, VkBuffer countBuffer,
                       VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                       uint32_t stride, bool indexed)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   perf_debug(cmd, "Draw indirect count");

   assert((stride % 4) == 0 && "aligned");

   size_t out_stride = sizeof(uint32_t) * (indexed ? 5 : 4);
   uint64_t patched = hk_pool_alloc(cmd, out_stride * maxDrawCount, 4).gpu;
   uint64_t in = hk_buffer_address_ro(buffer, offset);
   uint64_t count_addr = hk_buffer_address_ro(count_buffer, countBufferOffset);

   libagx_predicate_indirect(cmd, agx_1d(maxDrawCount),
                             AGX_BARRIER_ALL | AGX_PREGFX, patched, in,
                             count_addr, stride / 4, indexed);

   if (indexed) {
      hk_draw_indexed_indirect_inner(commandBuffer, patched, maxDrawCount,
                                     out_stride);
   } else {
      hk_draw_indirect_inner(commandBuffer, patched, maxDrawCount, out_stride);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                        VkDeviceSize offset, VkBuffer countBuffer,
                        VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                        uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, VkBuffer countBuffer,
                               VkDeviceSize countBufferOffset,
                               uint32_t maxDrawCount, uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer,
                               uint32_t instanceCount, uint32_t firstInstance,
                               VkBuffer counterBuffer,
                               VkDeviceSize counterBufferOffset,
                               uint32_t counterOffset, uint32_t vertexStride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, counter, counterBuffer);

   uint64_t indirect =
      hk_pool_alloc(cmd, sizeof(VkDrawIndirectCommand), 4).gpu;
   libagx_draw_xfb(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                   indirect, hk_buffer_address_ro(counter, counterBufferOffset),
                   counterOffset, vertexStride, instanceCount, firstInstance);
   hk_draw_indirect_inner(commandBuffer, indirect, 1, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer,
                                      uint32_t firstBinding,
                                      uint32_t bindingCount,
                                      const VkBuffer *pBuffers,
                                      const VkDeviceSize *pOffsets,
                                      const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;
      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;

      gfx->xfb[idx] = hk_buffer_addr_range(buffer, pOffsets[i], size, false);
   }
}

static void
hk_begin_end_xfb(VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
                 uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
                 const VkDeviceSize *pCounterBufferOffsets, bool begin)

{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   gfx->xfb_enabled = begin;

   /* If we haven't reserved XFB offsets yet for the command buffer, do so. */
   if (!gfx->xfb_offsets) {
      gfx->xfb_offsets = hk_pool_alloc(cmd, 4 * sizeof(uint32_t), 4).gpu;
   }

   struct libagx_xfb_counter_copy params = {};
   unsigned copies = 0;

   /* For CmdBeginTransformFeedbackEXT, we need to initialize everything */
   if (begin) {
      for (copies = 0; copies < 4; ++copies) {
         params.dest[copies] = gfx->xfb_offsets + copies * sizeof(uint32_t);
      }
   }

   /* A null pCounterBuffers means no counter buffers. */
   for (unsigned i = 0; pCounterBuffers && i < counterBufferCount; ++i) {
      if (pCounterBuffers[i] == VK_NULL_HANDLE)
         continue;

      VK_FROM_HANDLE(hk_buffer, buffer, pCounterBuffers[i]);

      uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[i] : 0;
      uint64_t cb_addr = hk_buffer_address_rw(buffer, offset);
      uint32_t cmd_idx = firstCounterBuffer + i;

      if (begin) {
         params.src[cmd_idx] = cb_addr;
      } else {
         params.dest[copies] = cb_addr;
         params.src[copies] = gfx->xfb_offsets + cmd_idx * sizeof(uint32_t);
         ++copies;
      }
   }

   if (begin)
      copies = 4;

   if (copies > 0) {
      perf_debug(cmd, "XFB counter copy");

      libagx_copy_xfb_counters(cmd, agx_1d(copies),
                               AGX_BARRIER_ALL | AGX_PREGFX,
                               hk_pool_upload(cmd, &params, sizeof(params), 8));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                                uint32_t firstCounterBuffer,
                                uint32_t counterBufferCount,
                                const VkBuffer *pCounterBuffers,
                                const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                              uint32_t firstCounterBuffer,
                              uint32_t counterBufferCount,
                              const VkBuffer *pCounterBuffers,
                              const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginConditionalRenderingEXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfoEXT *pConditionalRenderingBegin)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, pConditionalRenderingBegin->buffer);
   hk_cmd_render_merge_flush(cmd);

   uint64_t predicate =
      hk_buffer_address(buffer, pConditionalRenderingBegin->offset, true);
   cmd->state.cond_render.predicate = predicate;
   cmd->state.cond_render.active = true;
   cmd->state.cond_render.inverted =
      pConditionalRenderingBegin->flags &
      VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_cmd_render_merge_flush(cmd);
   cmd->state.cond_render.active = false;
}

bool
hk_cond_render_applies(struct hk_cmd_buffer *cmd)
{
   return cmd->state.cond_render.active &&
          (!cmd->in_meta || cmd->state.cond_render.meta);
}

/* Returns indirect arguments of `words` 32-bit words that are a copy of
 * `in`, or zero when the conditional rendering predicate fails. */
uint64_t
hk_predicate_indirect(struct hk_cmd_buffer *cmd, uint64_t in, unsigned words,
                      bool graphics)
{
   /* A failed allocation has already put the command buffer in error, and it
    * will not be submitted: return 0 rather than predicating to address 0. */
   uint64_t out = in ? hk_pool_alloc(cmd, words * sizeof(uint32_t), 16).gpu : 0;
   if (!out)
      return 0;
   libagx_predicate_indirect_apple9(
      cmd, agx_1d(1), AGX_BARRIER_ALL | (graphics ? AGX_PREGFX : 0), out, in,
      cmd->state.cond_render.predicate, cmd->state.cond_render.inverted, words);
   return out;
}

static struct agx_draw
hk_apple9_predicate_draw(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   unsigned words = draw.indexed ? 5 : 4;
   uint64_t in_draw = draw.b.ptr;
   if (!agx_is_indirect(draw.b)) {
      uint32_t desc[5] = {draw.b.count[0], draw.b.count[1], draw.start,
                          draw.indexed ? (uint32_t)draw.index_bias
                                       : draw.start_instance,
                          draw.start_instance};
      in_draw = hk_pool_upload(cmd, desc, sizeof(desc), 4);
   }
   uint64_t out_draw = hk_predicate_indirect(cmd, in_draw, words, true);
   if (draw.indexed) {
      return agx_draw_indexed_indirect(out_draw, draw.index_buffer,
                                       draw.index_buffer_range_B,
                                       draw.index_size, draw.restart);
   } else {
      return agx_draw_indirect(out_draw);
   }
}
