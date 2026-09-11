/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020-2021 Collabora, Ltd.
 * Copyright 2019 Sonny Jiang <sonnyj608@gmail.com>
 * Copyright 2019 Advanced Micro Devices, Inc.
 * Copyright 2014 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/layout/layout.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "gallium/auxiliary/util/u_blitter.h"
#include "gallium/auxiliary/util/u_dump.h"
#include "nir/pipe_nir.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_state.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_sampler.h"
#include "util/u_surface.h"
#include "agx_state.h"
#include "agx_apple9.h"
#include "glsl_types.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "shader_enums.h"

/* For block based blit kernels, we hardcode the maximum tile size which we can
 * always achieve. This simplifies our life.
 */
#define TILE_WIDTH  32
#define TILE_HEIGHT 32

static enum pipe_format
effective_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z32_FLOAT:
   case PIPE_FORMAT_Z24X8_UNORM:
      return PIPE_FORMAT_R32_FLOAT;
   case PIPE_FORMAT_Z16_UNORM:
      return PIPE_FORMAT_R16_UNORM;
   case PIPE_FORMAT_S8_UINT:
      return PIPE_FORMAT_R8_UINT;
   default:
      return format;
   }
}

static void *
asahi_blit_compute_shader(struct pipe_context *ctx, struct asahi_blit_key *key)
{
   const nir_shader_compiler_options *options =
      ctx->screen->nir_options[MESA_SHADER_COMPUTE];

   nir_builder b_ =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options, "blit_cs");
   nir_builder *b = &b_;
   b->shader->info.workgroup_size[0] = TILE_WIDTH;
   b->shader->info.workgroup_size[1] = TILE_HEIGHT;
   b->shader->info.num_ubos = 1;

   BITSET_SET(b->shader->info.textures_used, 0);
   BITSET_SET(b->shader->info.samplers_used, 0);
   BITSET_SET(b->shader->info.images_used, 0);

   nir_def *zero = nir_imm_int(b, 0);

   nir_def *params[4];
   b->shader->num_uniforms = ARRAY_SIZE(params);
   for (unsigned i = 0; i < b->shader->num_uniforms; ++i) {
      params[i] = nir_load_ubo(b, 2, 32, zero, nir_imm_int(b, i * 8),
                               .align_mul = 4, .range = ~0);
   }

   nir_def *trans_offs = params[0];
   nir_def *trans_scale = params[1];
   nir_def *dst_offs_2d = params[2];
   nir_def *dimensions_el_2d = params[3];

   nir_def *phys_id_el_nd = nir_trim_vector(
      b, nir_load_global_invocation_id(b, 32), key->array ? 3 : 2);
   nir_def *phys_id_el_2d = nir_trim_vector(b, phys_id_el_nd, 2);
   nir_def *layer = key->array ? nir_channel(b, phys_id_el_nd, 2) : NULL;

   /* Offset within the tile. We're dispatched for the entire tile but the
    * beginning might be out-of-bounds, so fix up.
    */
   nir_def *offs_in_tile_el_2d = nir_iand_imm(b, dst_offs_2d, 31);
   nir_def *logical_id_el_2d = nir_isub(b, phys_id_el_2d, offs_in_tile_el_2d);

   nir_def *image_pos_2d = nir_iadd(b, logical_id_el_2d, dst_offs_2d);
   nir_def *image_pos_nd = image_pos_2d;
   if (layer) {
      image_pos_nd =
         nir_vector_insert_imm(b, nir_pad_vector(b, image_pos_nd, 3), layer, 2);
   }

   nir_def *in_bounds;
   if (key->aligned) {
      in_bounds = nir_imm_true(b);
   } else {
      in_bounds = nir_ige(b, logical_id_el_2d, nir_imm_ivec2(b, 0, 0));
      in_bounds =
         nir_iand(b, in_bounds, nir_ilt(b, logical_id_el_2d, dimensions_el_2d));
   }

   unsigned bit_size = 32;
   nir_alu_type dst_type = nir_type_uint32;
   if (util_format_is_float16(key->dst_format)) {
      bit_size = 16;
      dst_type = nir_type_float16;
   }

   nir_def *colour0, *colour1;
   nir_push_if(b, nir_ball(b, in_bounds));
   {
      /* For pixels within the copy area, texture from the source */
      nir_def *coords_el_2d =
         nir_ffma_weak(b, nir_u2f32(b, logical_id_el_2d), trans_scale, trans_offs);

      nir_def *coords_el_nd = coords_el_2d;
      if (layer) {
         coords_el_nd = nir_vector_insert_imm(
            b, nir_pad_vector(b, coords_el_nd, 3), nir_u2f32(b, layer), 2);
      }

      colour0 = nir_tex(b, coords_el_nd, .texture_index = 0, .sampler_index = 0,
                        .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP,
                        .dim = GLSL_SAMPLER_DIM_2D, .is_array = key->array,
                        .dest_type = dst_type);
   }
   nir_push_else(b, NULL);
   {
      /* For out-of-bounds pixels, copy in the destination */
      colour1 = nir_image_load(
         b, 4, bit_size, nir_imm_int(b, 0), nir_pad_vec4(b, image_pos_nd), zero, zero,
         .image_array = key->array, .image_dim = GLSL_SAMPLER_DIM_2D,
         .access = ACCESS_IN_BOUNDS, .dest_type = dst_type);
   }
   nir_pop_if(b, NULL);
   nir_def *color = nir_if_phi(b, colour0, colour1);

   enum asahi_blit_clamp clamp = ASAHI_BLIT_CLAMP_NONE;
   bool src_sint = util_format_is_pure_sint(key->src_format);
   bool dst_sint = util_format_is_pure_sint(key->dst_format);
   if (util_format_is_pure_integer(key->src_format) &&
       util_format_is_pure_integer(key->dst_format)) {

      if (src_sint && !dst_sint)
         clamp = ASAHI_BLIT_CLAMP_SINT_TO_UINT;
      else if (!src_sint && dst_sint)
         clamp = ASAHI_BLIT_CLAMP_UINT_TO_SINT;
   }

   if (clamp == ASAHI_BLIT_CLAMP_SINT_TO_UINT)
      color = nir_imax(b, color, nir_imm_int(b, 0));
   else if (clamp == ASAHI_BLIT_CLAMP_UINT_TO_SINT)
      color = nir_umin(b, color, nir_imm_int(b, INT32_MAX));

   nir_def *local_offset = nir_imm_int(b, 0);
   nir_def *lid = nir_trim_vector(b, nir_load_local_invocation_id(b), 2);
   lid = nir_u2u16(b, lid);

   /* Pure integer formatss need to be clamped in software, at least in some
    * cases. We do so on store. Piglit gl-3.0-render-integer checks this, as
    * does KHR-GL33.packed_pixels.*.
    *
    * TODO: Make this common code somehow.
    */
   const struct util_format_description *desc =
      util_format_description(key->dst_format);
   unsigned c = util_format_get_first_non_void_channel(key->dst_format);

   if (desc->channel[c].size <= 16 &&
       util_format_is_pure_integer(key->dst_format)) {

      unsigned bits[4] = {
         desc->channel[0].size ?: desc->channel[0].size,
         desc->channel[1].size ?: desc->channel[0].size,
         desc->channel[2].size ?: desc->channel[0].size,
         desc->channel[3].size ?: desc->channel[0].size,
      };

      if (util_format_is_pure_sint(key->dst_format))
         color = nir_format_clamp_sint(b, color, bits);
      else
         color = nir_format_clamp_uint(b, color, bits);

      color = nir_u2u16(b, color);
   }

   /* The source texel has been converted into a 32-bit value. We need to
    * convert it to a tilebuffer format that can then be converted to the
    * destination format in the PBE hardware. That's the renderable format for
    * the destination format, which must exist along this path. This mirrors the
    * flow of fragment and end-of-tile shaders.
    */
   enum pipe_format tib_format =
      ail_pixel_format[effective_format(key->dst_format)].renderable;

   nir_store_local_pixel_agx(b, color, nir_imm_intN_t(b, 1, 16), lid, .base = 0,
                             .write_mask = 0xf, .format = tib_format,
                             .explicit_coord = true);

   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP);

   nir_push_if(b, nir_ball(b, nir_ieq_imm(b, lid, 0)));
   {
      nir_def *pbe_index = nir_imm_intN_t(b, 2, 16);
      nir_image_store_block_agx(
         b, pbe_index, local_offset, image_pos_nd, .format = tib_format,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = key->array,
         .explicit_coord = true);
   }
   nir_pop_if(b, NULL);
   b->shader->info.cs.image_block_size_per_thread_agx =
      util_format_get_blocksize(key->dst_format);

   return pipe_shader_from_nir(ctx, b->shader);
}

static bool
asahi_compute_blit_supported(const struct pipe_blit_info *info)
{
   return (info->src.box.depth == info->dst.box.depth) && !info->alpha_blend &&
          !info->num_window_rectangles && !info->sample0_only &&
          !info->scissor_enable && !info->window_rectangle_include &&
          !info->swizzle_enable && info->src.resource->nr_samples <= 1 &&
          info->dst.resource->nr_samples <= 1 &&
          !util_format_is_depth_and_stencil(info->src.format) &&
          !util_format_is_depth_and_stencil(info->dst.format) &&
          info->src.box.depth >= 0 &&
          info->mask == util_format_get_mask(info->src.format) &&
          /* XXX: texsubimage pbo failing otherwise, needs investigation */
          info->dst.format != PIPE_FORMAT_B5G6R5_UNORM &&
          info->dst.format != PIPE_FORMAT_B5G5R5A1_UNORM &&
          info->dst.format != PIPE_FORMAT_B5G5R5X1_UNORM &&
          info->dst.format != PIPE_FORMAT_R5G6B5_UNORM &&
          info->dst.format != PIPE_FORMAT_R5G5B5A1_UNORM &&
          info->dst.format != PIPE_FORMAT_R5G5B5X1_UNORM;
}

static void
asahi_compute_save(struct agx_context *ctx)
{
   struct asahi_blitter *blitter = &ctx->compute_blitter;
   struct agx_stage *stage = &ctx->stage[MESA_SHADER_COMPUTE];

   assert(!blitter->active && "recursion detected, driver bug");

   pipe_resource_reference(&blitter->saved_cb.buffer, stage->cb[0].buffer);
   memcpy(&blitter->saved_cb, &stage->cb[0],
          sizeof(struct pipe_constant_buffer));

   blitter->has_saved_image = stage->image_mask & BITFIELD_BIT(0);
   if (blitter->has_saved_image) {
      pipe_resource_reference(&blitter->saved_image.resource,
                              stage->images[0].resource);
      memcpy(&blitter->saved_image, &stage->images[0],
             sizeof(struct pipe_image_view));
   }

   pipe_sampler_view_reference(&blitter->saved_sampler_view,
                               &stage->textures[0]->base);

   blitter->saved_num_sampler_states = stage->sampler_count;
   memcpy(blitter->saved_sampler_states, stage->samplers,
          stage->sampler_count * sizeof(void *));

   blitter->saved_cs = stage->shader;
   blitter->active = true;
}

static void
asahi_compute_restore(struct agx_context *ctx)
{
   struct pipe_context *pctx = &ctx->base;
   struct asahi_blitter *blitter = &ctx->compute_blitter;

   if (blitter->has_saved_image) {
      pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 1, 0,
                              &blitter->saved_image);
      pipe_resource_reference(&blitter->saved_image.resource, NULL);
   }

   /* take_ownership=true so do not unreference */
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, &blitter->saved_cb);
   blitter->saved_cb.buffer = NULL;

   if (blitter->saved_sampler_view) {
      pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, 1, 0,
                              &blitter->saved_sampler_view);

      blitter->saved_sampler_view = NULL;
   }

   if (blitter->saved_num_sampler_states) {
      pctx->bind_sampler_states(pctx, MESA_SHADER_COMPUTE, 0,
                                blitter->saved_num_sampler_states,
                                blitter->saved_sampler_states);
   }

   pctx->bind_compute_state(pctx, blitter->saved_cs);
   blitter->saved_cs = NULL;
   blitter->active = false;
}

static void
asahi_compute_blit(struct pipe_context *ctx, const struct pipe_blit_info *info,
                   struct asahi_blitter *blitter)
{
   if (info->src.box.width == 0 || info->src.box.height == 0 ||
       info->dst.box.width == 0 || info->dst.box.height == 0)
      return;

   assert(asahi_compute_blit_supported(info));
   asahi_compute_save(agx_context(ctx));

   unsigned depth = info->dst.box.depth;
   bool array = depth > 1;

   struct pipe_resource *src = info->src.resource;
   struct pipe_resource *dst = info->dst.resource;
   struct pipe_sampler_view src_templ = {0}, *src_view;

   float src_width = (float)u_minify(src->width0, info->src.level);
   float src_height = (float)u_minify(src->height0, info->src.level);

   float x_scale =
      (info->src.box.width / (float)info->dst.box.width) / src_width;

   float y_scale =
      (info->src.box.height / (float)info->dst.box.height) / src_height;

   /* Expand the grid so destinations are in tiles */
   unsigned expanded_x0 = info->dst.box.x & ~(TILE_WIDTH - 1);
   unsigned expanded_y0 = info->dst.box.y & ~(TILE_HEIGHT - 1);
   unsigned expanded_x1 =
      align(info->dst.box.x + info->dst.box.width, TILE_WIDTH);
   unsigned expanded_y1 =
      align(info->dst.box.y + info->dst.box.height, TILE_HEIGHT);

   /* But clamp to the destination size to save some redundant threads */
   expanded_x1 =
      MIN2(expanded_x1, u_minify(info->dst.resource->width0, info->dst.level));
   expanded_y1 =
      MIN2(expanded_y1, u_minify(info->dst.resource->height0, info->dst.level));

   /* Calculate the width/height based on the expanded grid */
   unsigned width = expanded_x1 - expanded_x0;
   unsigned height = expanded_y1 - expanded_y0;

   unsigned data[] = {
      fui(0.5f * x_scale + (float)info->src.box.x / src_width),
      fui(0.5f * y_scale + (float)info->src.box.y / src_height),
      fui(x_scale),
      fui(y_scale),
      info->dst.box.x,
      info->dst.box.y,
      info->dst.box.width,
      info->dst.box.height,
   };

   struct pipe_constant_buffer cb = {
      .buffer_size = sizeof(data),
      .user_buffer = data,
   };
   ctx->set_constant_buffer(ctx, MESA_SHADER_COMPUTE, 0, &cb);

   struct pipe_image_view image = {
      .resource = dst,
      .access = PIPE_IMAGE_ACCESS_WRITE | PIPE_IMAGE_ACCESS_DRIVER_INTERNAL,
      .shader_access = PIPE_IMAGE_ACCESS_WRITE,
      .format = info->dst.format,
      .u.tex.level = info->dst.level,
      .u.tex.first_layer = info->dst.box.z,
      .u.tex.last_layer = info->dst.box.z + depth - 1,
      .u.tex.single_layer_view = !array,
   };
   ctx->set_shader_images(ctx, MESA_SHADER_COMPUTE, 0, 1, 0, &image);

   if (!blitter->sampler[info->filter]) {
      struct pipe_sampler_state sampler_state = {
         .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .min_img_filter = info->filter,
         .mag_img_filter = info->filter,
         .compare_func = PIPE_FUNC_ALWAYS,
         .seamless_cube_map = true,
         .max_lod = 31.0f,
      };

      blitter->sampler[info->filter] =
         ctx->create_sampler_state(ctx, &sampler_state);
   }

   ctx->bind_sampler_states(ctx, MESA_SHADER_COMPUTE, 0, 1,
                            &blitter->sampler[info->filter]);

   /* Initialize the sampler view. */
   u_sampler_view_default_template(&src_templ, src, src->format);
   src_templ.format = info->src.format;
   src_templ.target = array ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D;
   src_templ.swizzle_r = PIPE_SWIZZLE_X;
   src_templ.swizzle_g = PIPE_SWIZZLE_Y;
   src_templ.swizzle_b = PIPE_SWIZZLE_Z;
   src_templ.swizzle_a = PIPE_SWIZZLE_W;
   src_templ.u.tex.first_layer = info->src.box.z;
   src_templ.u.tex.last_layer = info->src.box.z + depth - 1;
   src_templ.u.tex.first_level = info->src.level;
   src_templ.u.tex.last_level = info->src.level;
   src_view = ctx->create_sampler_view(ctx, src, &src_templ);
   ctx->set_sampler_views(ctx, MESA_SHADER_COMPUTE, 0, 1, 0, &src_view);
   ctx->sampler_view_release(ctx, src_view);

   struct asahi_blit_key key = {
      .src_format = info->src.format,
      .dst_format = info->dst.format,
      .array = array,
      .aligned = info->dst.box.width == width && info->dst.box.height == height,
   };
   struct hash_entry *ent = _mesa_hash_table_search(blitter->blit_cs, &key);
   void *cs = NULL;

   if (ent) {
      cs = ent->data;
   } else {
      cs = asahi_blit_compute_shader(ctx, &key);
      _mesa_hash_table_insert(
         blitter->blit_cs, ralloc_memdup(blitter->blit_cs, &key, sizeof(key)),
         cs);
   }

   assert(cs != NULL);
   ctx->bind_compute_state(ctx, cs);

   struct pipe_grid_info grid_info = {
      .block = {TILE_WIDTH, TILE_HEIGHT, 1},
      .last_block = {width % TILE_WIDTH, height % TILE_HEIGHT, 1},
      .grid =
         {
            DIV_ROUND_UP(width, TILE_WIDTH),
            DIV_ROUND_UP(height, TILE_HEIGHT),
            depth,
         },
   };
   ctx->launch_grid(ctx, &grid_info);
   ctx->set_shader_images(ctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
   ctx->set_constant_buffer(ctx, MESA_SHADER_COMPUTE, 0, NULL);
   ctx->set_sampler_views(ctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);

   asahi_compute_restore(agx_context(ctx));
}

void
agx_blitter_save(struct agx_context *ctx, struct blitter_context *blitter,
                 enum asahi_blitter_op op)
{
   util_blitter_save_vertex_shader(blitter,
                                   ctx->stage[MESA_SHADER_VERTEX].shader);
   util_blitter_save_tessctrl_shader(blitter,
                                     ctx->stage[MESA_SHADER_TESS_CTRL].shader);
   util_blitter_save_tesseval_shader(blitter,
                                     ctx->stage[MESA_SHADER_TESS_EVAL].shader);
   util_blitter_save_geometry_shader(blitter,
                                     ctx->stage[MESA_SHADER_GEOMETRY].shader);
   util_blitter_save_so_targets(blitter, ctx->streamout.num_targets,
                                ctx->streamout.targets, MESA_PRIM_UNKNOWN);
   util_blitter_save_vertex_buffers(blitter, ctx->vertex_buffers,
                                    util_last_bit(ctx->vb_mask));
   util_blitter_save_vertex_elements(blitter, ctx->attributes);
   util_blitter_save_rasterizer(blitter, ctx->rast);
   util_blitter_save_viewport(blitter, &ctx->viewport[0]);

   if (op & ASAHI_SAVE_FRAGMENT_STATE) {
      util_blitter_save_blend(blitter, ctx->blend);
      util_blitter_save_depth_stencil_alpha(blitter, ctx->zs);
      util_blitter_save_stencil_ref(blitter, &ctx->stencil_ref);
      util_blitter_save_sample_mask(blitter, ctx->sample_mask, 0);

      util_blitter_save_scissor(blitter, &ctx->scissor[0]);
      util_blitter_save_fragment_shader(
         blitter, ctx->stage[MESA_SHADER_FRAGMENT].shader);

      if (op & ASAHI_SAVE_FRAGMENT_CONSTANT) {
         util_blitter_save_fragment_constant_buffer_slot(
            blitter, ctx->stage[MESA_SHADER_FRAGMENT].cb);
      }
   }

   if (op & ASAHI_SAVE_FRAMEBUFFER) {
      util_blitter_save_framebuffer(blitter, &ctx->framebuffer);
   }

   if (op & ASAHI_SAVE_TEXTURES) {
      util_blitter_save_fragment_sampler_states(
         blitter, ctx->stage[MESA_SHADER_FRAGMENT].sampler_count,
         (void **)(ctx->stage[MESA_SHADER_FRAGMENT].samplers));

      util_blitter_save_fragment_sampler_views(
         blitter, ctx->stage[MESA_SHADER_FRAGMENT].texture_count,
         (struct pipe_sampler_view **)ctx->stage[MESA_SHADER_FRAGMENT].textures);
   }

   if (!(op & ASAHI_DISABLE_RENDER_COND)) {
      util_blitter_save_render_condition(blitter,
                                         (struct pipe_query *)ctx->cond_query,
                                         ctx->cond_cond, ctx->cond_mode);
   }
}

static bool asahi_raw_resolve(struct pipe_context *, const struct pipe_blit_info *);

void
agx_blit(struct pipe_context *pipe, const struct pipe_blit_info *info)
{
   struct agx_context *ctx = agx_context(pipe);

   if (info->render_condition_enable && !agx_render_condition_check(ctx))
      return;

   /* Legalize compression /before/ calling into u_blitter to avoid recursion.
    * u_blitter bans recursive usage.
    */
   agx_legalize_compression(ctx, agx_resource(info->dst.resource),
                            info->dst.format);

   agx_legalize_compression(ctx, agx_resource(info->src.resource),
                            info->src.format);

   bool apple9 = agx_apple9_direct_render_enabled(agx_device(pipe->screen));
   if (apple9 && asahi_raw_resolve(pipe, info))
      return;
   if (apple9 && util_try_blit_via_copy_region(pipe, info, false))
      return;

   if (!apple9 && asahi_compute_blit_supported(info)) {
      asahi_compute_blit(pipe, info, &ctx->compute_blitter);
      return;
   }

   if (!util_blitter_is_blit_supported(ctx->blitter, info)) {
      fprintf(stderr, "\n");
      util_dump_blit_info(stderr, info);
      fprintf(stderr, "\n\n");
      UNREACHABLE("Unsupported blit");
   }

   /* Handle self-blits */
   agx_flush_writer(ctx, agx_resource(info->dst.resource), "Blit");

   agx_blitter_save(
      ctx, ctx->blitter,
      ASAHI_BLIT |
         (info->render_condition_enable ? 0 : ASAHI_DISABLE_RENDER_COND));
   util_blitter_blit(ctx->blitter, info, NULL);
}

static bool
try_copy_via_blit(struct pipe_context *pctx, struct pipe_resource *dst,
                  unsigned dst_level, unsigned dstx, unsigned dsty,
                  unsigned dstz, struct pipe_resource *src, unsigned src_level,
                  const struct pipe_box *src_box)
{
   struct agx_context *ctx = agx_context(pctx);

   if (dst->target == PIPE_BUFFER)
      return false;

   /* TODO: Handle these for rusticl copies */
   if (dst->target != src->target)
      return false;

   struct pipe_blit_info info = {
      .dst =
         {
            .resource = dst,
            .level = dst_level,
            .box.x = dstx,
            .box.y = dsty,
            .box.z = dstz,
            .box.width = src_box->width,
            .box.height = src_box->height,
            .box.depth = src_box->depth,
            .format = dst->format,
         },
      .src =
         {
            .resource = src,
            .level = src_level,
            .box = *src_box,
            .format = src->format,
         },
      .mask = util_format_get_mask(src->format),
      .filter = PIPE_TEX_FILTER_NEAREST,
      .scissor_enable = 0,
      .swizzle_enable = 0,
   };

   /* snorm formats don't round trip, so don't use them for copies */
   if (util_format_is_snorm(info.dst.format))
      info.dst.format = util_format_snorm_to_sint(info.dst.format);

   if (util_format_is_snorm(info.src.format))
      info.src.format = util_format_snorm_to_sint(info.src.format);

   if (util_blitter_is_blit_supported(ctx->blitter, &info) &&
       info.dst.format == info.src.format) {

      agx_blit(pctx, &info);
      return true;
   } else {
      return false;
   }
}

/* Raw 32-bit tiled-to-linear copies. Layout parameters are uniforms so one
 * normally compiled shader serves different image dimensions and strides. */
static nir_def *
asahi_spread_bits(nir_builder *b, nir_def *v)
{
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 8)), 0x00ff00ff);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 4)), 0x0f0f0f0f);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 2)), 0x33333333);
   return nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 1)), 0x55555555);
}

/* Sample-minor tiled storage is shared by rendering and resolves. Keeping the
 * addressing in NIR lets ordinary compute lowering handle both byte layouts. */
static nir_def *
asahi_tiled_pixel(nir_builder *b, nir_def *x, nir_def *y,
                  nir_def *tw_log2, nir_def *th_log2, nir_def *columns)
{
   nir_def *common = nir_umin(b, tw_log2, th_log2);
   nir_def *mask = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), common), -1);
   nir_def *ix = asahi_spread_bits(b, nir_iand(b, x, mask));
   nir_def *iy = asahi_spread_bits(b, nir_iand(b, y, mask));
   nir_def *major = nir_bcsel(b, nir_ult(b, th_log2, tw_log2), x, y);
   nir_def *tail = nir_ishl(b, nir_iand(b, nir_ushr(b, major, common),
      nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1),
         nir_isub(b, nir_umax(b, tw_log2, th_log2), common)), -1)),
      nir_imul_imm(b, common, 2));
   nir_def *tile = nir_iadd(b, nir_ushr(b, x, tw_log2),
      nir_imul(b, nir_ushr(b, y, th_log2), columns));
   return nir_iadd(b, nir_ishl(b, tile, nir_iadd(b, tw_log2, th_log2)),
      nir_ior(b, tail, nir_ior(b, ix, nir_ishl_imm(b, iy, 1))));
}

/* Uncompressed color reloads use the same sample-minor layout as the
 * compute resolve, including the single-sample case. The shader restores raw
 * tile words before application draws, preserving their original bits. The
 * same compiled entry reloads color after a partial render. Color clears use
 * ordinary draws, so the background entry can reload every attachment. */
struct agx_color_reload_key {
   enum pipe_format formats[8];
   unsigned samples, mask, count;
   bool store;
   unsigned tile_w_log2[8], tile_h_log2[8];
};

struct agx_color_reload {
   struct blitter_context *blitter;
   struct hash_table *shaders;
   void *vs, *rast, *velem;
};

static uint32_t
asahi_reload_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct agx_color_reload_key));
}

static bool
asahi_reload_equal(const void *a, const void *b)
{
   return !memcmp(a, b, sizeof(struct agx_color_reload_key));
}

static void *
asahi_reload_vs(struct pipe_context *pctx)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pctx->screen->nir_options[MESA_SHADER_VERTEX], "color reload triangle");
   nir_def *id = nir_load_vertex_id(&b);
   nir_def *x = nir_bcsel(&b, nir_ieq_imm(&b, id, 1),
                         nir_imm_float(&b, 3), nir_imm_float(&b, -1));
   nir_def *y = nir_bcsel(&b, nir_ieq_imm(&b, id, 2),
                         nir_imm_float(&b, 3), nir_imm_float(&b, -1));
   nir_store_output(&b, nir_vec4(&b, x, y, nir_imm_float(&b, 0),
                                nir_imm_float(&b, 1)), nir_imm_int(&b, 0),
      .io_semantics = {.location = VARYING_SLOT_POS}, .write_mask = 15);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return pipe_shader_from_nir(pctx, b.shader);
}

static void *
asahi_reload_fs(struct pipe_context *pctx, const struct agx_color_reload_key *key)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, pctx->screen->nir_options[MESA_SHADER_FRAGMENT],
      key->store ? "color store" : "color reload");
   b.shader->info.num_ubos = key->store ? 1 : key->count + 1;
   b.shader->info.num_ssbos = key->store ? key->count : 0;
   nir_def *xy = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *tile_xy = nir_imm_int(&b, 0);
   nir_variable *pixel_index = NULL;
   nir_def *i = NULL;
   if (key->store) {
      pixel_index =
         nir_local_variable_create(b.impl, glsl_uint_type(), "tile pixel");
      nir_store_var(&b, pixel_index, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b)->control = nir_loop_control_dont_unroll;
      i = nir_load_var(&b, pixel_index);
      nir_push_if(&b, nir_uge_imm(&b, i, key->samples == 4 ? 512 : 1024));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_def *x = nir_iand_imm(&b, i, 31);
      nir_def *y = nir_ushr_imm(&b, i, 5);
      tile_xy = nir_ior(&b, x, nir_ishl_imm(&b, y, 16));
      nir_def *group = nir_load_workgroup_id(&b);
      xy = nir_vec2(
         &b, nir_iadd(&b, x, nir_imul_imm(&b, nir_channel(&b, group, 0), 32)),
         nir_iadd(&b, y,
                  nir_imul_imm(&b, nir_channel(&b, group, 1),
                               key->samples == 4 ? 16 : 32)));
      nir_def *width =
         nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                      .align_mul = 4, .range = 128);
      nir_def *height =
         nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 4),
                      .align_mul = 4, .range = 128);
      nir_push_if(&b, nir_iand(&b, nir_ult(&b, nir_channel(&b, xy, 0), width),
                               nir_ult(&b, nir_channel(&b, xy, 1), height)));
   }
   unsigned offset = 0;
   for (unsigned rt = 0; rt < key->count; ++rt) {
      unsigned words = agx_apple9_color_words(key->formats[rt]);
      if (key->mask & BITFIELD_BIT(rt)) {
         nir_def *p[4];
         p[0] = nir_imm_int(&b, key->tile_w_log2[rt]);
         p[1] = nir_imm_int(&b, key->tile_h_log2[rt]);
         for (unsigned i = 2; i < 4; ++i)
            p[i] = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
               nir_imm_int(&b, 16 * rt + 4 * i), .align_mul = 4, .range = 128);
         nir_def *pixel = asahi_tiled_pixel(&b, nir_channel(&b, xy, 0),
            nir_channel(&b, xy, 1), p[0], p[1], p[2]);
         unsigned bytes = util_format_get_blocksize(key->formats[rt]);
         nir_def *base = nir_iadd(&b, p[3],
            nir_imul_imm(&b, pixel, key->samples * bytes));
         nir_variable *sample_index = nir_local_variable_create(b.impl, glsl_uint_type(), "reload sample");
         nir_store_var(&b, sample_index, nir_imm_int(&b, 0), 1);
         for (unsigned ss = 0; ss < (key->store ? key->samples : 1); ++ss) {
            if (!key->store)
               nir_push_loop(&b)->control = nir_loop_control_dont_unroll;
            nir_def *sample = key->store ? nir_imm_int(&b, ss)
                                         : nir_load_var(&b, sample_index);
            if (!key->store) {
               nir_push_if(&b, nir_uge_imm(&b, sample, key->samples));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);
            }
            {
               nir_def *v[2];
               if (key->store) {
                  nir_def *raw = nir_load_tile_pixel_agx(
                     &b, words, 32,
                     nir_u2u16(&b, nir_ishl(&b, nir_imm_int(&b, 1), sample)),
                     tile_xy, .base = offset,
                     .format = words == 2 ? PIPE_FORMAT_R32G32_UINT
                                          : PIPE_FORMAT_R32_UINT);
                  for (unsigned w = 0; w < words; ++w)
                     v[w] = nir_channel(&b, raw, w);
               } else
                  for (unsigned w = 0; w < words; ++w) {
                     /* R16F occupies a padded word in the tilebuffer. Reading
                      * its containing memory word also works for odd halfword
                      * offsets. */
                     nir_def *address = nir_iadd_imm(
                        &b, nir_iadd(&b, base, nir_imul_imm(&b, sample, bytes)),
                        4 * w);
                     v[w] = nir_load_ubo(
                        &b, 1, 32, nir_imm_int(&b, rt + 1),
                        bytes == 2 ? nir_iand_imm(&b, address, ~3u) : address,
                        .align_mul = 4, .range = ~0u);
                     if (bytes == 2)
                        v[w] = nir_iand_imm(
                           &b,
                           nir_ushr(&b, v[w],
                                    nir_imul_imm(
                                       &b, nir_iand_imm(&b, address, 2), 8)),
                           0xffff);
                  }
               if (key->formats[rt] == PIPE_FORMAT_B8G8R8A8_UNORM ||
                   key->formats[rt] == PIPE_FORMAT_B8G8R8X8_UNORM) {
                  /* PBE memory is BGRA; raw tile words use RGBA. */
                  v[0] = nir_ior(
                     &b, nir_iand_imm(&b, v[0], 0xff00ff00),
                     nir_ior(
                        &b, nir_ishl_imm(&b, nir_iand_imm(&b, v[0], 255), 16),
                        nir_iand_imm(&b, nir_ushr_imm(&b, v[0], 16), 255)));
                  if (key->formats[rt] == PIPE_FORMAT_B8G8R8X8_UNORM)
                     v[0] = nir_ior_imm(&b, v[0], 0xff000000);
               }
               if (key->store) {
                  for (unsigned w = 0; w < words; ++w) {
                     nir_def *address = nir_iadd_imm(
                        &b, nir_iadd(&b, base, nir_imul_imm(&b, sample, bytes)),
                        4 * w);
                     nir_store_ssbo(&b, bytes == 2 ? nir_u2u16(&b, v[w]) : v[w],
                                    nir_imm_int(&b, rt), address,
                                    .align_mul = MIN2(bytes, 4),
                                    .write_mask = 1,
                                    .access = ACCESS_NON_READABLE);
                  }
               }
               if (!key->store)
                  nir_store_local_pixel_agx(
                     &b, nir_vec(&b, v, words),
                     nir_u2u16(&b, nir_ishl(&b, nir_imm_int(&b, 1), sample)),
                     nir_undef(&b, 2, 16), .base = offset,
                     .format = words == 2 ? PIPE_FORMAT_R32G32_UINT
                                          : PIPE_FORMAT_R32_UINT,
                     .write_mask = BITFIELD_MASK(words));
            }
            if (!key->store) {
               nir_store_var(&b, sample_index, nir_iadd_imm(&b, sample, 1), 1);
               nir_pop_loop(&b, NULL);
            }
         }
      }
      offset += 4 * words;
   }
   if (pixel_index) {
      nir_pop_if(&b, NULL);
      nir_store_var(&b, pixel_index, nir_iadd_imm(&b, i, 1), 1);
      nir_pop_loop(&b, NULL);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return pipe_shader_from_nir(pctx, b.shader);
}

void
agx_destroy_color_reload(struct agx_context *ctx)
{
   struct agx_color_reload *reload = ctx->color_reload;
   if (!reload)
      return;
   struct pipe_context *pctx = &ctx->base;
   if (reload->shaders) {
      hash_table_foreach(reload->shaders, entry) {
         pctx->delete_fs_state(pctx, entry->data);
         free((void *)entry->key);
      }
      _mesa_hash_table_destroy(reload->shaders, NULL);
   }
   if (reload->vs) pctx->delete_vs_state(pctx, reload->vs);
   if (reload->rast) pctx->delete_rasterizer_state(pctx, reload->rast);
   if (reload->velem) pctx->delete_vertex_elements_state(pctx, reload->velem);
   if (reload->blitter) util_blitter_destroy(reload->blitter);
   free(reload);
   ctx->color_reload = NULL;
}

bool
agx_apple9_reload_color(struct agx_batch *batch)
{
   struct agx_context *ctx = batch->ctx;
   struct pipe_context *pctx = &ctx->base;
   if (batch->apple9_color_reloaded)
      return true;
   batch->apple9_color_reloaded = true;
   unsigned mask = 0;
   for (unsigned rt = 0; rt < batch->key.nr_cbufs; ++rt) {
      if (batch->key.cbufs[rt].texture)
         mask |= BITFIELD_BIT(rt);
   }
   if (!mask)
      return false;
   struct agx_color_reload_key key = {
      .samples = batch->tilebuffer_layout.nr_samples,
      .mask = mask, .count = batch->key.nr_cbufs,
   };
   uint32_t params[8][4] = {0};
   for (unsigned rt = 0; rt < key.count; ++rt) {
      const struct pipe_surface *surf = &batch->key.cbufs[rt];
      key.formats[rt] = surf->format == PIPE_FORMAT_NONE
                          ? PIPE_FORMAT_R8G8B8A8_UNORM : surf->format;
      if (!(mask & BITFIELD_BIT(rt)))
         continue;
      struct agx_resource *rsrc = agx_resource(surf->texture);
      const struct ail_layout *layout = &rsrc->layout;
      if (ail_is_level_logically_compressed(layout, surf->level) ||
          surf->first_layer != surf->last_layer)
         return false;
      struct ail_tile tile = layout->tilesize_el[surf->level];
      params[rt][0] = util_logbase2(tile.width_el);
      params[rt][1] = util_logbase2(tile.height_el);
      key.tile_w_log2[rt] = params[rt][0];
      key.tile_h_log2[rt] = params[rt][1];
      params[rt][2] = DIV_ROUND_UP(u_minify(surf->texture->width0, surf->level), tile.width_el);
      params[rt][3] = layout->level_offsets_B[surf->level] +
                     surf->first_layer * layout->layer_stride_B;
   }
   params[0][0] = batch->key.width;
   params[0][1] = batch->key.height;
   if (!ctx->color_reload) {
      ctx->color_reload = calloc(1, sizeof(*ctx->color_reload));
      if (!ctx->color_reload)
         return false;
      struct agx_color_reload *r = ctx->color_reload;
      r->blitter = util_blitter_create(pctx);
      r->shaders = _mesa_hash_table_create(NULL, asahi_reload_hash, asahi_reload_equal);
      r->vs = asahi_reload_vs(pctx);
      struct pipe_rasterizer_state rast = {
         .flatshade = true, .cull_face = PIPE_FACE_NONE,
         .fill_front = PIPE_POLYGON_MODE_FILL, .fill_back = PIPE_POLYGON_MODE_FILL,
         .depth_clip_near = true, .depth_clip_far = true,
         .half_pixel_center = true, .bottom_edge_rule = true, .multisample = true,
      };
      r->rast = pctx->create_rasterizer_state(pctx, &rast);
      r->velem = pctx->create_vertex_elements_state(pctx, 0, NULL);
      if (!r->blitter || !r->shaders || !r->vs || !r->rast || !r->velem) {
         agx_destroy_color_reload(ctx);
         return false;
      }
   }
   struct agx_color_reload *r = ctx->color_reload;
   struct hash_entry *entry = _mesa_hash_table_search(r->shaders, &key);
   if (!entry) {
      struct agx_color_reload_key *saved_key = malloc(sizeof(key));
      void *fs = asahi_reload_fs(pctx, &key);
      if (!saved_key || !fs) {
         free(saved_key);
         if (fs) pctx->delete_fs_state(pctx, fs);
         return false;
      }
      *saved_key = key;
      entry = _mesa_hash_table_insert(r->shaders, saved_key, fs);
   }
   struct hash_entry *store_entry = NULL;
   {
      struct agx_color_reload_key store_key = key;
      store_key.store = true;
      store_entry = _mesa_hash_table_search(r->shaders, &store_key);
      if (!store_entry) {
         struct agx_color_reload_key *saved_key = malloc(sizeof(store_key));
         void *fs = asahi_reload_fs(pctx, &store_key);
         if (!saved_key || !fs) {
            free(saved_key);
            if (fs)
               pctx->delete_fs_state(pctx, fs);
            return false;
         }
         *saved_key = store_key;
         store_entry = _mesa_hash_table_insert(r->shaders, saved_key, fs);
      }
   }
   struct pipe_constant_buffer saved[9] = {0};
   for (unsigned i = 0; i <= key.count; ++i)
      util_copy_constant_buffer(&saved[i], &ctx->stage[MESA_SHADER_FRAGMENT].cb[i]);
   bool queries = ctx->active_queries;
   agx_blitter_save(ctx, r->blitter, ASAHI_SAVE_FRAGMENT_STATE);
   util_blitter_common_clear_setup(r->blitter, batch->key.width,
      batch->key.height, mask * PIPE_CLEAR_COLOR0, NULL, NULL);
   pctx->set_active_query_state(pctx, false);
   pctx->bind_vs_state(pctx, r->vs);
   pctx->bind_fs_state(pctx, entry->data);
   pctx->bind_rasterizer_state(pctx, r->rast);
   pctx->bind_vertex_elements_state(pctx, r->velem);
   pctx->bind_tcs_state(pctx, NULL);
   pctx->bind_tes_state(pctx, NULL);
   pctx->bind_gs_state(pctx, NULL);
   pctx->set_stream_output_targets(pctx, 0, NULL, NULL, MESA_PRIM_UNKNOWN);
   struct pipe_viewport_state vp = {
      .scale = {batch->key.width * 0.5f, batch->key.height * 0.5f, 1},
      .translate = {batch->key.width * 0.5f, batch->key.height * 0.5f, 0},
   };
   pctx->set_viewport_states(pctx, 0, 1, &vp);
   struct pipe_constant_buffer cb = {.user_buffer = params, .buffer_size = sizeof(params)};
   pctx->set_constant_buffer(pctx, MESA_SHADER_FRAGMENT, 0, &cb);
   for (unsigned rt = 0; rt < key.count; ++rt) {
      if (!(mask & BITFIELD_BIT(rt)))
         continue;
      struct pipe_resource *resource = batch->key.cbufs[rt].texture;
      struct pipe_constant_buffer backing = {
         .buffer = resource, .buffer_size = agx_resource(resource)->layout.size_B,
      };
      pctx->set_constant_buffer(pctx, MESA_SHADER_FRAGMENT, rt + 1, &backing);
   }
   batch->clear |= mask * PIPE_CLEAR_COLOR0;
   struct pipe_draw_info info = {.mode = MESA_PRIM_TRIANGLES, .instance_count = 1};
   struct pipe_draw_start_count_bias draw = {.count = 3};
   /* Retain this helper's own launch record for background and partial-load
    * dispatch. Its resources remain live for the complete batch. */
   unsigned helper_draw = batch->apple9_uniform_draw_count;
   pctx->draw_vbo(pctx, &info, 0, NULL, &draw, 1);
   bool emitted = batch->apple9_uniform_draw_count == helper_draw + 1;
   if (emitted)
      batch->apple9_color_reload_draw = helper_draw;
   if (emitted && store_entry) {
      struct agx_stage *stage = &ctx->stage[MESA_SHADER_FRAGMENT];
      struct pipe_shader_buffer saved_ssbo[8] = {0}, targets[8] = {0};
      unsigned writable = stage->ssbo_writable_mask;
      for (unsigned rt = 0; rt < key.count; ++rt) {
         saved_ssbo[rt].buffer_offset = stage->ssbo[rt].buffer_offset;
         saved_ssbo[rt].buffer_size = stage->ssbo[rt].buffer_size;
         pipe_resource_reference(&saved_ssbo[rt].buffer,
                                 stage->ssbo[rt].buffer);
         if (!(key.mask & BITFIELD_BIT(rt)))
            continue;
         targets[rt].buffer = batch->key.cbufs[rt].texture;
         targets[rt].buffer_size =
            agx_resource(targets[rt].buffer)->layout.size_B;
      }
      pctx->set_shader_buffers(pctx, MESA_SHADER_FRAGMENT, 0, key.count,
                               targets, key.mask);
      pctx->bind_fs_state(pctx, store_entry->data);

      unsigned store_draw = batch->apple9_uniform_draw_count;
      batch->apple9_preparing_tile_store = true;
      pctx->draw_vbo(pctx, &info, 0, NULL, &draw, 1);
      batch->apple9_preparing_tile_store = false;
      emitted = batch->apple9_uniform_draw_count == store_draw + 1;
      if (emitted)
         batch->apple9_color_store_draw = store_draw;
      pctx->set_shader_buffers(pctx, MESA_SHADER_FRAGMENT, 0, key.count,
                               saved_ssbo, writable);
      for (unsigned rt = 0; rt < key.count; ++rt)
         pipe_resource_reference(&saved_ssbo[rt].buffer, NULL);
   }
   for (unsigned i = 0; i <= key.count; ++i) {
      pctx->set_constant_buffer(pctx, MESA_SHADER_FRAGMENT, i, &saved[i]);
      pipe_resource_reference(&saved[i].buffer, NULL);
   }
   util_blitter_restore_vertex_states(r->blitter);
   util_blitter_restore_fragment_states(r->blitter);
   util_blitter_restore_render_cond(r->blitter);
   util_blitter_unset_running_flag(r->blitter);
   pctx->set_active_query_state(pctx, queries);
   return emitted;
}

static void *
asahi_raw_resolve_shader(struct pipe_context *pctx, unsigned samples,
                         unsigned components, bool fp16)
{
   const nir_shader_compiler_options *options = pctx->screen->nir_options[MESA_SHADER_COMPUTE];
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options,
                                                  "tiled color resolve");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;
   b.shader->info.num_ubos = 1;
   nir_def *p[20];
   for (unsigned i = 0; i < ARRAY_SIZE(p); ++i)
      p[i] = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, i * 4),
                          .align_mul = 4, .range = sizeof(uint32_t) * ARRAY_SIZE(p));
   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, id, 0), *y = nir_channel(&b, id, 1);
   nir_push_if(&b, nir_iand(&b, nir_ult(&b, x, p[0]), nir_ult(&b, y, p[1])));
   nir_def *sx = nir_iadd(&b, p[2], nir_imul(&b, x, p[4]));
   nir_def *sy = nir_iadd(&b, p[3], nir_imul(&b, y, p[5]));
   nir_def *dx = nir_iadd(&b, p[6], nir_imul(&b, x, p[8]));
   nir_def *dy = nir_iadd(&b, p[7], nir_imul(&b, y, p[9]));
   nir_def *src = asahi_tiled_pixel(&b, sx, sy, p[10], p[11], p[12]);
   nir_def *dst = asahi_tiled_pixel(&b, dx, dy, p[14], p[15], p[16]);
   unsigned bytes = fp16 ? components * 2 : 4;
   src = nir_iadd(&b, p[13], nir_imul_imm(&b, src, samples * bytes));
   dst = nir_iadd(&b, p[17], nir_imul_imm(&b, dst, bytes));
   nir_def *sum[4];
   for (unsigned c = 0; c < components; ++c)
      sum[c] = nir_imm_float(&b, 0);
   for (unsigned sample = 0; sample < samples; ++sample) {
      nir_def *words[2];
      unsigned bits = bytes == 2 ? 16 : 32;
      for (unsigned w = 0; w < DIV_ROUND_UP(bytes, 4); ++w)
         words[w] = nir_u2u32(&b, nir_load_ssbo(&b, 1, bits, nir_imm_int(&b, 0),
            nir_iadd_imm(&b, src, sample * bytes + w * 4),
            .align_mul = bits / 8, .access = ACCESS_NON_WRITEABLE));
      for (unsigned c = 0; c < components; ++c) {
         nir_def *value = fp16
            ? nir_channel(&b, nir_unpack_half_2x16(&b, words[c / 2]), c & 1)
            : nir_u2f32(&b, nir_iand_imm(&b, nir_ushr_imm(&b, words[0], c * 8), 255));
         sum[c] = nir_fadd(&b, sum[c], value);
      }
   }
   for (unsigned c = 0; c < components; ++c)
      sum[c] = nir_fmul_imm(&b, sum[c], 1.0f / samples);
   nir_def *out[2] = {nir_imm_int(&b, 0), nir_imm_int(&b, 0)};
   if (fp16) {
      for (unsigned c = 0; c < components; c += 2)
         out[c / 2] = nir_pack_half_2x16(&b, nir_vec2(&b, sum[c],
            c + 1 < components ? sum[c + 1] : nir_imm_float(&b, 0)));
   } else {
      for (unsigned c = 0; c < components; ++c)
         out[0] = nir_ior(&b, out[0], nir_ishl_imm(&b,
            nir_f2u32(&b, nir_fround_even(&b, sum[c])), 8 * c));
   }
   for (unsigned w = 0; w < DIV_ROUND_UP(bytes, 4); ++w)
      nir_store_ssbo(&b, bytes == 2 ? nir_u2u16(&b, out[w]) : out[w],
         nir_imm_int(&b, 1), nir_iadd_imm(&b, dst, 4 * w),
         .align_mul = MIN2(bytes, 4), .write_mask = 1, .access = ACCESS_NON_READABLE);
   nir_pop_if(&b, NULL);
   return pipe_shader_from_nir(pctx, b.shader);
}

static bool
asahi_raw_resolve(struct pipe_context *pctx, const struct pipe_blit_info *info)
{
   struct agx_context *ctx = agx_context(pctx);
   struct pipe_resource *src = info->src.resource, *dst = info->dst.resource;
   struct agx_resource *s = agx_resource(src), *d = agx_resource(dst);
   bool fp16 = agx_apple9_color_is_half(info->src.format);
   if ((src->nr_samples != 2 && src->nr_samples != 4) || dst->nr_samples > 1 ||
       info->src.format != info->dst.format ||
       (!fp16 && info->src.format != PIPE_FORMAT_B8G8R8A8_UNORM &&
        info->src.format != PIPE_FORMAT_B8G8R8X8_UNORM) ||
       info->mask != util_format_get_mask(info->src.format) ||
       info->swizzle_enable || info->alpha_blend || info->num_window_rectangles ||
       info->window_rectangle_include || info->sample0_only ||
       abs(info->src.box.width) != abs(info->dst.box.width) ||
       abs(info->src.box.height) != abs(info->dst.box.height) ||
       info->src.box.depth != 1 || info->dst.box.depth != 1 ||
       !info->src.box.width || !info->src.box.height ||
       s->layout.compressed || d->layout.compressed ||
       s->layout.tiling != AIL_TILING_GPU || d->layout.tiling != AIL_TILING_GPU ||
       s->bo == d->bo || s->layout.size_B > UINT32_MAX || d->layout.size_B > UINT32_MAX)
      return false;
   unsigned levels[] = {info->src.level, info->dst.level};
   const struct pipe_box *boxes[] = {&info->src.box, &info->dst.box};
   struct agx_resource *resources[] = {s, d};
   uint32_t params[20] = {abs(info->src.box.width), abs(info->src.box.height)};
   for (unsigned i = 0; i < 2; ++i) {
      const struct pipe_box *box = boxes[i];
      struct agx_resource *r = resources[i];
      if (box->z < 0 || levels[i] > r->base.last_level ||
          box->z >= r->base.array_size)
         return false;
      struct ail_tile tile = r->layout.tilesize_el[levels[i]];
      if (!util_is_power_of_two_nonzero(tile.width_el) ||
          !util_is_power_of_two_nonzero(tile.height_el))
         return false;
      params[2 + 4 * i] = box->x - (box->width < 0);
      params[3 + 4 * i] = box->y - (box->height < 0);
      params[4 + 4 * i] = box->width < 0 ? -1 : 1;
      params[5 + 4 * i] = box->height < 0 ? -1 : 1;
      params[10 + 4 * i] = util_logbase2(tile.width_el);
      params[11 + 4 * i] = util_logbase2(tile.height_el);
      params[12 + 4 * i] = DIV_ROUND_UP(r->layout.stride_el[levels[i]], tile.width_el);
      params[13 + 4 * i] = ail_get_level_offset_B(&r->layout, levels[i]) +
                           r->layout.layer_stride_B * box->z;
   }
   /* Clip in copy coordinates so mirrored boxes and destination scissoring
    * preserve the one-to-one sample mapping. */
   for (unsigned axis = 0; axis < 2; ++axis) {
      int lo = 0, hi = params[axis];
      for (unsigned i = 0; i < 2; ++i) {
         struct pipe_resource *r = &resources[i]->base;
         int start = params[2 + 4 * i + axis];
         int step = (int32_t)params[4 + 4 * i + axis];
         int min = 0, max = u_minify(axis ? r->height0 : r->width0, levels[i]);
         if (i == 1 && info->scissor_enable) {
            min = axis ? info->scissor.miny : info->scissor.minx;
            max = MIN2(max, axis ? info->scissor.maxy : info->scissor.maxx);
         }
         lo = MAX2(lo, step > 0 ? min - start : start - max + 1);
         hi = MIN2(hi, step > 0 ? max - start : start - min + 1);
      }
      if (lo >= hi)
         return true;
      params[axis] = hi - lo;
      for (unsigned i = 0; i < 2; ++i)
         params[2 + 4 * i + axis] += lo * (int32_t)params[4 + 4 * i + axis];
   }
   unsigned components = agx_apple9_color_components(info->src.format);
   unsigned kind = fp16 ? (components == 1 ? 1 : components == 2 ? 2 : 3) : 0;
   void **shader = &ctx->compute_blitter.resolve_cs[src->nr_samples == 4][kind];
   if (!*shader)
      *shader = asahi_raw_resolve_shader(pctx, src->nr_samples, components, fp16);
   if (!*shader)
      return false;
   agx_flush_writer(ctx, s, "multisample resolve");
   agx_flush_writer(ctx, d, "multisample resolve destination");
   struct agx_stage *stage = &ctx->stage[MESA_SHADER_COMPUTE];
   struct pipe_shader_buffer saved[2] = {stage->ssbo[0], stage->ssbo[1]};
   for (unsigned i = 0; i < 2; ++i) {
      saved[i].buffer = NULL;
      pipe_resource_reference(&saved[i].buffer, stage->ssbo[i].buffer);
   }
   unsigned writable = stage->ssbo_writable_mask & 3;
   asahi_compute_save(ctx);
   struct pipe_constant_buffer cb = {.user_buffer = params, .buffer_size = sizeof(params)};
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, &cb);
   struct pipe_shader_buffer buffers[] = {
      {.buffer = src, .buffer_size = s->layout.size_B},
      {.buffer = dst, .buffer_size = d->layout.size_B},
   };
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, 2, buffers, 2);
   pctx->bind_compute_state(pctx, *shader);
   struct pipe_grid_info grid = {.block = {32, 1, 1},
      .grid = {DIV_ROUND_UP(params[0], 32), params[1], 1}};
   pctx->launch_grid(pctx, &grid);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, 2, saved, writable);
   for (unsigned i = 0; i < 2; ++i)
      pipe_resource_reference(&saved[i].buffer, NULL);
   asahi_compute_restore(ctx);
   BITSET_SET(d->data_valid, info->dst.level);
   d->linear_export_valid = false;
   return true;
}

static void *
asahi_detile_shader(struct pipe_context *pctx)
{
   const nir_shader_compiler_options *options = pctx->screen->nir_options[MESA_SHADER_COMPUTE];
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options,
                                                  "32-bit detile");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 2;
   b.shader->info.num_ubos = 1;
   nir_def *p[6];
   for (unsigned i = 0; i < ARRAY_SIZE(p); ++i)
      p[i] = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, i * 4),
                          .align_mul = 4, .range = sizeof(uint32_t) * 6);
   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, id, 0), *y = nir_channel(&b, id, 1);
   nir_push_if(&b, nir_iand(&b, nir_ult(&b, x, p[0]), nir_ult(&b, y, p[1])));
   nir_def *tw = nir_ishl(&b, nir_imm_int(&b, 1), p[2]);
   nir_def *th = nir_ishl(&b, nir_imm_int(&b, 1), p[3]);
   nir_def *ix = asahi_spread_bits(&b, nir_iand(&b, x, nir_iadd_imm(&b, tw, -1)));
   nir_def *iy = asahi_spread_bits(&b, nir_iand(&b, y, nir_iadd_imm(&b, th, -1)));
   nir_def *tile = nir_iadd(&b, nir_ushr(&b, x, p[2]),
      nir_imul(&b, nir_ushr(&b, y, p[3]), p[4]));
   nir_def *src = nir_iadd(&b, nir_ishl(&b, tile, nir_iadd(&b, p[2], p[3])),
      nir_ior(&b, ix, nir_ishl_imm(&b, iy, 1)));
   nir_def *dst = nir_iadd(&b, nir_imul(&b, y, p[5]), x);
   nir_def *value = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 0),
      nir_ishl_imm(&b, src, 2), .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
   nir_store_ssbo(&b, value, nir_imm_int(&b, 1), nir_ishl_imm(&b, dst, 2),
      .align_mul = 4, .write_mask = 1, .access = ACCESS_NON_READABLE);
   nir_pop_if(&b, NULL);
   return pipe_shader_from_nir(pctx, b.shader);
}

static bool
asahi_detile_copy(struct pipe_context *pctx, struct pipe_resource *dst,
                  unsigned dst_level, unsigned dstx, unsigned dsty, unsigned dstz,
                  struct pipe_resource *src, unsigned src_level,
                  const struct pipe_box *box)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_resource *s = agx_resource(src), *d = agx_resource(dst);
   if ((src->target != PIPE_TEXTURE_2D && src->target != PIPE_TEXTURE_RECT) ||
       (dst->target != PIPE_TEXTURE_2D && dst->target != PIPE_TEXTURE_RECT) ||
       src->format != dst->format || src->nr_samples > 1 || dst->nr_samples > 1 ||
       util_format_get_blocksize(src->format) != 4 ||
       util_format_get_blockwidth(src->format) != 1 ||
       util_format_get_blockheight(src->format) != 1 ||
       s->layout.compressed || d->layout.compressed ||
       s->layout.tiling != AIL_TILING_GPU || d->layout.tiling != AIL_TILING_LINEAR ||
       src_level || dst_level || dstx || dsty || dstz || box->x || box->y || box->z ||
       box->depth != 1 || box->width <= 0 || box->height <= 0 ||
       box->width != src->width0 || box->height != src->height0 ||
       dst->width0 < src->width0 || dst->height0 < src->height0 || s->bo == d->bo ||
       s->layout.size_B > UINT32_MAX || d->layout.size_B > UINT32_MAX)
      return false;

   perf_debug_ctx(ctx, "GPU detile %ux%u", src->width0, src->height0);
   struct ail_tile tile = s->layout.tilesize_el[0];
   if (!util_is_power_of_two_nonzero(tile.width_el) ||
       !util_is_power_of_two_nonzero(tile.height_el) ||
       tile.width_el > 65536 || tile.height_el > 65536 ||
       (d->layout.linear_stride_B & 3))
      return false;
   if (!ctx->compute_blitter.detile_cs)
      ctx->compute_blitter.detile_cs = asahi_detile_shader(pctx);
   if (!ctx->compute_blitter.detile_cs)
      return false;

   struct agx_stage *stage = &ctx->stage[MESA_SHADER_COMPUTE];
   struct pipe_shader_buffer saved[2] = {stage->ssbo[0], stage->ssbo[1]};
   for (unsigned i = 0; i < 2; i++) {
      saved[i].buffer = NULL;
      pipe_resource_reference(&saved[i].buffer, stage->ssbo[i].buffer);
   }
   unsigned writable = stage->ssbo_writable_mask & 3;
   asahi_compute_save(ctx);
   uint32_t params[] = {src->width0, src->height0,
      util_logbase2(tile.width_el), util_logbase2(tile.height_el),
      DIV_ROUND_UP(s->layout.stride_el[0], tile.width_el),
      d->layout.linear_stride_B / 4};
   struct pipe_constant_buffer cb = {.user_buffer = params, .buffer_size = sizeof(params)};
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, &cb);
   struct pipe_shader_buffer buffers[] = {
      {.buffer = src, .buffer_size = s->layout.size_B},
      {.buffer = dst, .buffer_size = d->layout.size_B},
   };
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, 2, buffers, 2);
   pctx->bind_compute_state(pctx, ctx->compute_blitter.detile_cs);
   struct pipe_grid_info grid = {.block = {32, 1, 1},
      .grid = {DIV_ROUND_UP(src->width0, 32), src->height0, 1}};
   pctx->launch_grid(pctx, &grid);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, 2, saved, writable);
   for (unsigned i = 0; i < 2; i++)
      pipe_resource_reference(&saved[i].buffer, NULL);
   asahi_compute_restore(ctx);
   /* Exports must be ready for the external consumer when flush_resource returns. */
   agx_flush_writer(ctx, d, "detile copy");
   return true;
}

void
agx_resource_copy_region(struct pipe_context *pctx, struct pipe_resource *dst,
                         unsigned dst_level, unsigned dstx, unsigned dsty,
                         unsigned dstz, struct pipe_resource *src,
                         unsigned src_level, const struct pipe_box *src_box)
{
   /* Compile eligible raw layout copies through the Apple9 compute path.
    * Other formats retain Gallium's synchronized mapped-copy fallback. */
   if (agx_apple9_direct_render_enabled(agx_device(pctx->screen))) {
      if (asahi_detile_copy(pctx, dst, dst_level, dstx, dsty, dstz, src, src_level, src_box))
         return;
      util_resource_copy_region(pctx, dst, dst_level, dstx, dsty, dstz, src,
                                 src_level, src_box);
      return;
   }

   if (dst->target == PIPE_BUFFER && src->target == PIPE_BUFFER) {
      struct agx_batch *batch = agx_get_compute_batch(agx_context(pctx));
      agx_batch_init_state(batch);
      assert(dst->format == src->format);
      unsigned bs = util_format_get_blocksize(dst->format);
      unsigned size = bs * src_box->width;
      unsigned dst_offset = dstx * bs;
      uint64_t dst_addr = agx_map_gpu(agx_resource(dst)) + dst_offset;
      uint64_t src_addr = agx_map_gpu(agx_resource(src)) + src_box->x * bs;

      agx_batch_reads(batch, agx_resource(src));
      agx_batch_writes_range(batch, agx_resource(dst), dst_offset, size);
      /* Use vectorized copies for as much of the buffer as possible. This requires
       * that dst, src, and size are all properly aligned. Failing to check for
       * alignment on the buffers causes subtle and hard-to-debug issues!
       */
      if (size >= 16 && (dst_addr & 0xf) == 0 && (src_addr & 0xf) == 0) {
         unsigned uint4s = size / 16;
         unsigned bytes = uint4s * 16;

         libagx_copy_uint4(batch, agx_1d(uint4s), AGX_BARRIER_ALL, dst_addr, src_addr);

         dst_addr += bytes;
         src_addr += bytes;
         size -= bytes;
      }

      if (size) {
         libagx_copy_uchar(batch, agx_1d(size), AGX_BARRIER_ALL, dst_addr, src_addr);
      }

      return;
   }

   if (try_copy_via_blit(pctx, dst, dst_level, dstx, dsty, dstz, src, src_level,
                         src_box))
      return;

   /* CPU fallback */
   util_resource_copy_region(pctx, dst, dst_level, dstx, dsty, dstz, src,
                             src_level, src_box);
}
