/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx_compile_apple9.h"
#include "compiler/nir/nir_builder.h"
#include "indices/u_primconvert.h"
#include "nir/pipe_nir.h"
#include "pipe/p_defines.h"
#include "poly/cl/libpoly.h"
#include "poly/prim.h"
#include "util/u_draw.h"
#include "util/u_dump.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_prim.h"
#include "util/u_upload_mgr.h"
#include "agx_state.h"

static struct pipe_stream_output_target *
agx_create_stream_output_target(struct pipe_context *pctx,
                                struct pipe_resource *prsc,
                                unsigned buffer_offset, unsigned buffer_size)
{
   struct agx_streamout_target *target =
      rzalloc(pctx, struct agx_streamout_target);

   if (!target)
      return NULL;

   pipe_reference_init(&target->base.reference, 1);
   pipe_resource_reference(&target->base.buffer, prsc);

   target->base.context = pctx;
   target->base.buffer_offset = buffer_offset;
   target->base.buffer_size = buffer_size;

   uint32_t zero = 0;
   target->offset = pipe_buffer_create_with_data(pctx, PIPE_BIND_GLOBAL,
                                                 PIPE_USAGE_DEFAULT, 4, &zero);

   return &target->base;
}

static void
agx_stream_output_target_destroy(struct pipe_context *pctx,
                                 struct pipe_stream_output_target *target)
{
   struct agx_streamout_target *tgt = agx_so_target(target);

   pipe_resource_reference(&tgt->base.buffer, NULL);
   pipe_resource_reference(&tgt->offset, NULL);
   ralloc_free(target);
}

static void
agx_set_stream_output_targets(struct pipe_context *pctx, unsigned num_targets,
                              struct pipe_stream_output_target **targets,
                              const unsigned *offsets,
                              enum mesa_prim output_prim)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_streamout *so = &ctx->streamout;

   assert(num_targets <= ARRAY_SIZE(so->targets));

   for (unsigned i = 0; i < num_targets; i++) {
      /* From the Gallium documentation:
       *
       *    -1 means the buffer should be appended to, and everything else sets
       *    the internal offset.
       *
       * We append regardless, so just check for != -1. Yes, using a negative
       * sentinel value with an unsigned type is bananas. But it's in the
       * Gallium contract and it will work out fine. Probably should be
       * redefined to be ~0 instead of -1 but it doesn't really matter.
       */
      if (offsets[i] != -1 && targets[i] != NULL) {
         pipe_buffer_write(pctx, agx_so_target(targets[i])->offset, 0, 4,
                           &offsets[i]);
      }

      pipe_so_target_reference(&so->targets[i], targets[i]);
   }

   for (unsigned i = num_targets; i < so->num_targets; i++)
      pipe_so_target_reference(&so->targets[i], NULL);

   so->num_targets = num_targets;
}

static struct pipe_stream_output_target *
get_target(struct agx_context *ctx, unsigned buffer)
{
   if (buffer < ctx->streamout.num_targets)
      return ctx->streamout.targets[buffer];
   else
      return NULL;
}

/*
 * Return the address of the indexed streamout buffer. This will be
 * pushed into the streamout shader.
 */
uint64_t
agx_batch_get_so_address(struct agx_batch *batch, unsigned buffer,
                         uint32_t *size)
{
   struct pipe_stream_output_target *target = get_target(batch->ctx, buffer);

   /* If there's no target, don't write anything */
   if (!target) {
      *size = 0;
      return 0;
   }

   /* Otherwise, write the target */
   struct agx_resource *rsrc = agx_resource(target->buffer);
   agx_batch_writes_range(batch, rsrc, target->buffer_offset,
                          target->buffer_size);

   *size = target->buffer_size;
   return agx_map_gpu(rsrc) + target->buffer_offset;
}

void
agx_draw_vbo_from_xfb(struct pipe_context *pctx,
                      const struct pipe_draw_info *info, unsigned drawid_offset,
                      const struct pipe_draw_indirect_info *indirect)
{
   perf_debug_ctx(agx_context(pctx), "draw auto");

   struct agx_streamout_target *so =
      agx_so_target(indirect->count_from_stream_output);

   unsigned offset_B = 0;
   pipe_buffer_read(pctx, so->offset, 0, 4, &offset_B);

   unsigned count = offset_B / so->stride;

   struct pipe_draw_start_count_bias draw = {
      .start = 0,
      .count = count,
   };

   pctx->draw_vbo(pctx, info, drawid_offset, NULL, &draw, 1);
}

static uint32_t
xfb_prims_for_vertices(enum mesa_prim mode, unsigned verts)
{
   uint32_t prims = u_decomposed_prims_for_vertices(mode, verts);

   /* The GL spec isn't super clear about this, but it implies that quads are
    * supposed to be tessellated into primitives and piglit
    * (ext_transform_feedback-tessellation quads) checks this.
    */
   if (u_decomposed_prim(mode) == MESA_PRIM_QUADS)
      prims *= 2;

   return prims;
}

/*
 * Count generated primitives on the CPU for transform feedback. This only works
 * in the absence of indirect draws, geometry shaders, or tessellation.
 */
void
agx_primitives_update_direct(struct agx_context *ctx,
                             const struct pipe_draw_info *info,
                             const struct pipe_draw_start_count_bias *draw)
{
   assert(ctx->active_queries && ctx->prims_generated[0] && "precondition");
   assert(!ctx->stage[MESA_SHADER_GEOMETRY].shader &&
          "Geometry shaders use their own counting");

   agx_query_increment_cpu(
      ctx, ctx->prims_generated[0],
      (uint64_t)xfb_prims_for_vertices(info->mode, draw->count) *
         info->instance_count);
}

void
agx_init_streamout_functions(struct pipe_context *ctx)
{
   ctx->create_stream_output_target = agx_create_stream_output_target;
   ctx->stream_output_target_destroy = agx_stream_output_target_destroy;
   ctx->set_stream_output_targets = agx_set_stream_output_targets;
}

/* Apple9 uses the same software primitive/capture model as the Apple8
 * passthrough-GS path. Poly helpers assemble input primitives on the GPU; the
 * API vertex program executes on the GPU and writes raw outputs through ordinary
 * SSBOs. Direct-draw offsets and counts are maintained on the CPU, like direct
 * primitive queries. No Apple8 executable or dispatch ABI is reused. */
/* Like Poly's restart unroller, search segments in order and emit their
 * decomposed primitives in parallel. Apple9 currently lacks the subgroup
 * ballot lowering used by that helper, so lanes perform the same scan and
 * distribute the output primitives. Only the resulting count returns to the
 * CPU; index data stays on the GPU. */
static void *
apple9_xfb_unroll_shader(struct pipe_context *pctx, enum mesa_prim mode,
                         unsigned index_size, bool flatshade_first)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, pctx->screen->nir_options[MESA_SHADER_COMPUTE],
      "transform feedback restart unroll");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 3;
   b.shader->info.num_ubos = 1;
   nir_def *params[3];
   for (unsigned i = 0; i < ARRAY_SIZE(params); ++i)
      params[i] =
         nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 4 * i),
                      .align_mul = 4, .range = ARRAY_SIZE(params) * 4);
   nir_def *lane = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_variable *needle =
      nir_local_variable_create(b.impl, glsl_uint_type(), "needle");
   nir_variable *end =
      nir_local_variable_create(b.impl, glsl_uint_type(), "end");
   nir_variable *output =
      nir_local_variable_create(b.impl, glsl_uint_type(), "output");
   nir_variable *primitive =
      nir_local_variable_create(b.impl, glsl_uint_type(), "primitive");
   nir_store_var(&b, needle, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, output, nir_imm_int(&b, 0), 1);
   nir_push_loop(&b)->control = nir_loop_control_dont_unroll;
   nir_def *start = nir_load_var(&b, needle);
   nir_break_if(&b, nir_uge(&b, start, params[0]));
   nir_store_var(&b, end, start, 1);
   nir_push_loop(&b)->control = nir_loop_control_dont_unroll;
   nir_def *next = nir_load_var(&b, end);
   nir_break_if(&b, nir_uge(&b, next, params[0]));
   nir_def *index =
      nir_load_ssbo(&b, 1, index_size * 8, nir_imm_int(&b, 0),
                    nir_iadd(&b, params[2], nir_imul_imm(&b, next, index_size)),
                    .align_mul = index_size, .access = ACCESS_NON_WRITEABLE);
   nir_break_if(&b, nir_ieq(&b, nir_u2u32(&b, index), params[1]));
   nir_store_var(&b, end, nir_iadd_imm(&b, next, 1), 1);
   nir_pop_loop(&b, NULL);
   nir_def *stop = nir_load_var(&b, end);
   nir_def *length = nir_isub(&b, stop, start);
   const struct u_prim_vertex_count *vc = u_prim_vertex_count(mode);
   nir_def *prims = nir_bcsel(
      &b, nir_uge_imm(&b, length, vc->min),
      nir_iadd_imm(
         &b,
         nir_udiv_imm(&b, nir_iadd_imm(&b, length, -(int)vc->min), vc->incr),
         1),
      nir_imm_int(&b, 0));
   if (mode == MESA_PRIM_LINE_LOOP)
      prims =
         nir_bcsel(&b, nir_uge_imm(&b, length, 2), length, nir_imm_int(&b, 0));
   unsigned vertices = mesa_vertices_per_prim(u_decomposed_prim(mode));
   nir_def *base = nir_load_var(&b, output);
   nir_store_var(&b, primitive, lane, 1);
   nir_push_loop(&b)->control = nir_loop_control_dont_unroll;
   nir_def *prim = nir_load_var(&b, primitive);
   nir_break_if(&b, nir_uge(&b, prim, prims));
   for (unsigned v = 0; v < vertices; ++v) {
      nir_def *id = nir_iadd_imm(&b, nir_imul_imm(&b, prim, vertices), v);
      if (vertices == 2)
         id = poly_vertex_id_for_line_class(&b, nir_imm_int(&b, mode), prim,
                                            nir_imm_int(&b, v), prims);
      else if (vertices == 3)
         id = poly_vertex_id_for_tri_class(&b, nir_imm_int(&b, mode), prim,
                                           nir_imm_int(&b, v),
                                           nir_imm_bool(&b, flatshade_first));
      index = nir_load_ssbo(
         &b, 1, index_size * 8, nir_imm_int(&b, 0),
         nir_iadd(&b, params[2],
                  nir_imul_imm(&b, nir_iadd(&b, start, id), index_size)),
         .align_mul = index_size, .access = ACCESS_NON_WRITEABLE);
      nir_def *dst = nir_iadd_imm(
         &b, nir_iadd(&b, base, nir_imul_imm(&b, prim, vertices)), v);
      nir_store_ssbo(&b, nir_u2u32(&b, index), nir_imm_int(&b, 1),
                     nir_imul_imm(&b, dst, 4), .write_mask = 1, .align_mul = 4);
   }
   nir_store_var(&b, primitive, nir_iadd_imm(&b, prim, 32), 1);
   nir_pop_loop(&b, NULL);
   nir_store_var(&b, output,
                 nir_iadd(&b, base, nir_imul_imm(&b, prims, vertices)), 1);
   nir_store_var(&b, needle, nir_iadd_imm(&b, stop, 1), 1);
   nir_pop_loop(&b, NULL);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_ssbo(&b, nir_load_var(&b, output), nir_imm_int(&b, 2),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   nir_pop_if(&b, NULL);
   return pipe_shader_from_nir(pctx, b.shader);
}

static void
apple9_xfb_unroll_restart(struct pipe_context *pctx,
                          const struct pipe_draw_info *info, unsigned drawid,
                          const struct pipe_draw_start_count_bias *draw)
{
   struct agx_context *ctx = agx_context(pctx);
   bool first = ctx->rast->base.flatshade_first;
   void **shader =
      &ctx->apple9_xfb_unroll[info->mode][util_logbase2(info->index_size)]
                             [first];
   if (!*shader)
      *shader =
         apple9_xfb_unroll_shader(pctx, info->mode, info->index_size, first);

   uint64_t max_vertices =
      (uint64_t)u_decomposed_prims_for_vertices(info->mode, draw->count) *
      mesa_vertices_per_prim(u_decomposed_prim(info->mode));
   if (max_vertices > UINT32_MAX / 4)
      abort();
   struct pipe_shader_buffer buffers[3] = {0};
   uint32_t params[] = {draw->count, info->restart_index,
                        draw->start * info->index_size};
   if (info->has_user_indices) {
      u_upload_data_ref(pctx->const_uploader, 0, draw->count * info->index_size,
                        16, (const uint8_t *)info->index.user + params[2],
                        &buffers[0].buffer_offset, &buffers[0].buffer);
      buffers[0].buffer_size = draw->count * info->index_size;
      params[2] = 0;
   } else {
      pipe_resource_reference(&buffers[0].buffer, info->index.resource);
      buffers[0].buffer_size = info->index.resource->width0;
   }
   buffers[1].buffer_size = MAX2(max_vertices * 4, 4);
   buffers[1].buffer =
      pipe_buffer_create(pctx->screen, PIPE_BIND_SHADER_BUFFER,
                         PIPE_USAGE_DEFAULT, buffers[1].buffer_size);
   buffers[2].buffer_size = 4;
   buffers[2].buffer = pipe_buffer_create(pctx->screen, PIPE_BIND_SHADER_BUFFER,
                                          PIPE_USAGE_DEFAULT, 4);
   if (!buffers[1].buffer || !buffers[2].buffer || !*shader)
      abort();
   /* Apple9 CDM dispatches within a batch do not have a memory barrier.
    * Submit an index producer (for example a GPU buffer copy) before the
    * scan consumes it. The resource tracker orders the batches on the GPU. */
   agx_flush_writer(ctx, agx_resource(buffers[0].buffer),
                     "Transform feedback restart indices");
   struct agx_stage *stage = &ctx->stage[MESA_SHADER_COMPUTE];
   void *saved_cs = stage->shader;
   struct pipe_shader_buffer saved[3] = {0};
   for (unsigned i = 0; i < 3; ++i) {
      saved[i] = stage->ssbo[i];
      saved[i].buffer = NULL;
      pipe_resource_reference(&saved[i].buffer, stage->ssbo[i].buffer);
   }
   unsigned writable = stage->ssbo_writable_mask & 7;
   bool saved_cb_bound = stage->cb_mask & 1;
   struct pipe_constant_buffer saved_cb = stage->cb[0];
   saved_cb.buffer = NULL;
   pipe_resource_reference(&saved_cb.buffer, stage->cb[0].buffer);
   struct pipe_constant_buffer cb = {.user_buffer = params,
                                     .buffer_size = sizeof(params)};
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, &cb);
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, 3, buffers, 6);
   pctx->bind_compute_state(pctx, *shader);
   struct pipe_grid_info grid = {.block = {32, 1, 1}, .grid = {1, 1, 1}};
   /* This dispatch implements input assembly, not an API compute dispatch. */
   struct agx_query *saved_cs_query =
      ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS];
   ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS] = NULL;
   pctx->launch_grid(pctx, &grid);
   ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS] = saved_cs_query;
   pctx->set_shader_buffers(pctx, MESA_SHADER_COMPUTE, 0, 3, saved, writable);
   for (unsigned i = 0; i < 3; ++i)
      pipe_resource_reference(&saved[i].buffer, NULL);
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0,
                             saved_cb_bound ? &saved_cb : NULL);
   pipe_resource_reference(&saved_cb.buffer, NULL);
   pctx->bind_compute_state(pctx, saved_cs);

   struct pipe_draw_info unrolled = *info;
   unrolled.mode = u_decomposed_prim(info->mode);
   unrolled.index_size = 4;
   unrolled.has_user_indices = false;
   unrolled.index.resource = buffers[1].buffer;
   unrolled.primitive_restart = false;
   struct pipe_draw_start_count_bias range = *draw;
   range.start = 0;
   pipe_buffer_read(pctx, buffers[2].buffer, 0, 4, &range.count);
   agx_apple9_capture_streamout(pctx, &unrolled, drawid, &range);
   for (unsigned i = 0; i < 3; ++i)
      pipe_resource_reference(&buffers[i].buffer, NULL);
}

void
agx_cleanup_streamout(struct pipe_context *pctx)
{
   struct agx_context *ctx = agx_context(pctx);
   if (ctx->apple9_primconvert)
      util_primconvert_destroy(ctx->apple9_primconvert);
   pipe_resource_reference(&ctx->apple9_xfb_target, NULL);
   if (ctx->apple9_xfb_fs)
      pctx->delete_fs_state(pctx, ctx->apple9_xfb_fs);
   if (ctx->apple9_xfb_rasterizer)
      pctx->delete_rasterizer_state(pctx, ctx->apple9_xfb_rasterizer);
   for (unsigned mode = 0; mode < MESA_PRIM_COUNT; ++mode)
      for (unsigned size = 0; size < 3; ++size)
         for (unsigned first = 0; first < 2; ++first)
            if (ctx->apple9_xfb_unroll[mode][size][first])
               pctx->delete_compute_state(
                  pctx, ctx->apple9_xfb_unroll[mode][size][first]);
}

void
agx_apple9_capture_streamout(struct pipe_context *pctx,
                             const struct pipe_draw_info *info, unsigned drawid,
                             const struct pipe_draw_start_count_bias *draw)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_uncompiled_shader *vs = ctx->stage[MESA_SHADER_VERTEX].shader;
   if (!draw->count || !info->instance_count)
      return;
   if (info->index_size && info->primitive_restart) {
      apple9_xfb_unroll_restart(pctx, info, drawid, draw);
      return;
   }
   bool flatshade_first = ctx->rast->base.flatshade_first;
   unsigned vertices = mesa_vertices_per_prim(u_decomposed_prim(info->mode));
   uint64_t prims_per_instance =
      u_decomposed_prims_for_vertices(info->mode, draw->count);
   uint64_t prims = prims_per_instance * info->instance_count;
   uint64_t written = prims;
   unsigned offsets[4] = {0};
   u_foreach_bit(i, vs->xfb_buffers_written) {
      struct pipe_stream_output_target *target = get_target(ctx, i);
      if (!target) {
         written = 0;
         continue;
      }
      unsigned stride = vs->xfb_strides[i];
      agx_so_target(target)->stride = stride;
      pipe_buffer_read(pctx, agx_so_target(target)->offset, 0, 4, &offsets[i]);
      /* Same complete-primitive bound as poly's pre-GS setup. Padding after
       * the last captured component does not need backing storage. */
      int64_t available = (int64_t)target->buffer_size + stride -
                          vs->xfb_output_end[i] - offsets[i];
      uint64_t capacity = MAX2(available, 0) / ((uint64_t)stride * vertices);
      written = MIN2(written, capacity);
   }
   if (ctx->active_queries) {
      agx_query_increment_cpu(ctx, ctx->tf_prims_generated[0], written);
      agx_query_increment_cpu(ctx, ctx->tf_overflow[0], written < prims);
      agx_query_increment_cpu(ctx, ctx->tf_any_overflow, written < prims);
      if (ctx->rast->base.rasterizer_discard)
         agx_query_increment_cpu(ctx, ctx->prims_generated[0], prims);
   }

   if (written) {
      if (!ctx->apple9_xfb_fs) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT,
            pctx->screen->nir_options[MESA_SHADER_FRAGMENT],
            "XFB capture raster discard");
         nir_store_output(
            &b, nir_imm_vec4(&b, 0, 0, 0, 0), nir_imm_int(&b, 0),
            .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
         ctx->apple9_xfb_fs = pipe_shader_from_nir(pctx, b.shader);
         struct pipe_resource target = {
            .target = PIPE_TEXTURE_2D,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .width0 = 1,
            .height0 = 1,
            .depth0 = 1,
            .array_size = 1,
            .bind = PIPE_BIND_RENDER_TARGET,
            .usage = PIPE_USAGE_DEFAULT,
         };
         ctx->apple9_xfb_target =
            pctx->screen->resource_create(pctx->screen, &target);
         if (!ctx->apple9_xfb_target)
            abort();
         struct pipe_rasterizer_state rs = {.rasterizer_discard = true,
                                            .point_size = 1,
                                            .line_width = 1};
         ctx->apple9_xfb_rasterizer = pctx->create_rasterizer_state(pctx, &rs);
      }
      struct pipe_shader_buffer index_buffer = {0};
      if (info->index_size) {
         if (info->has_user_indices) {
            u_upload_data_ref(
               pctx->const_uploader, 0, draw->count * info->index_size, 16,
               (const uint8_t *)info->index.user +
                  draw->start * info->index_size,
               &index_buffer.buffer_offset, &index_buffer.buffer);
            index_buffer.buffer_size = draw->count * info->index_size;
         } else {
            pipe_resource_reference(&index_buffer.buffer, info->index.resource);
            index_buffer.buffer_size = info->index.resource->width0;
         }
      }
      struct pipe_framebuffer_state saved_fb = {0};
      util_copy_framebuffer_state(&saved_fb, &ctx->framebuffer);
      struct pipe_framebuffer_state capture_fb = {
         .width = 1,
         .height = 1,
         .nr_cbufs = 1,
         .cbufs = {{.texture = ctx->apple9_xfb_target,
                    .format = PIPE_FORMAT_R8G8B8A8_UNORM}},
      };
      pctx->set_framebuffer_state(pctx, &capture_fb);
      void *saved_fs = ctx->stage[MESA_SHADER_FRAGMENT].shader;
      void *saved_rs = ctx->rast;
      bool saved_queries = ctx->active_queries;
      struct pipe_shader_buffer saved[6] = {0};
      for (unsigned i = 0; i < 6; ++i) {
         saved[i] =
            ctx->stage[MESA_SHADER_VERTEX].ssbo[AGX_APPLE9_XFB_PARAMS + i];
         saved[i].buffer = NULL;
         pipe_resource_reference(&saved[i].buffer,
                                 ctx->stage[MESA_SHADER_VERTEX]
                                    .ssbo[AGX_APPLE9_XFB_PARAMS + i]
                                    .buffer);
      }
      unsigned saved_writable =
         ctx->stage[MESA_SHADER_VERTEX].ssbo_writable_mask >>
         AGX_APPLE9_XFB_PARAMS;
      pctx->bind_fs_state(pctx, ctx->apple9_xfb_fs);
      pctx->bind_rasterizer_state(pctx, ctx->apple9_xfb_rasterizer);
      ctx->active_queries = false;
      ctx->apple9_xfb_capture = vs;
      ctx->apple9_xfb_mode = info->mode;
      ctx->apple9_xfb_index_size = info->index_size;
      ctx->apple9_xfb_flatshade_first = flatshade_first;
      ctx->dirty |= AGX_DIRTY_VS_PROG;

      /* Restart has already been expanded into a GPU index buffer. One
       * contiguous instance-major stream now covers both capture paths. */
      uint64_t done = 0, total = written * vertices;
      while (done < total) {
         /* Stay within the 32-bit vertex/byte-offset domains. Splits
          * preserve the invocation's API IDs and output ordering. */
         unsigned chunk = MIN2(total - done, 1024u * 1024u);
         struct agx_apple9_xfb_params params = {
            .first_vertex = info->has_user_indices && info->index_size
                               ? 0 : draw->start,
            .index_bias = draw->index_bias,
            .base_instance = info->start_instance,
            .draw_id = drawid,
            .first_capture = done,
            .vertices_per_instance = prims_per_instance * vertices,
         };
         struct pipe_shader_buffer buffers[6] = {0};
         u_upload_data_ref(pctx->const_uploader, 0, sizeof(params), 16,
                           &params, &buffers[0].buffer_offset,
                           &buffers[0].buffer);
         buffers[0].buffer_size = sizeof(params);
         u_foreach_bit(i, vs->xfb_buffers_written) {
            struct pipe_stream_output_target *target = get_target(ctx, i);
            buffers[i + 1] = (struct pipe_shader_buffer){
               .buffer = target->buffer,
               .buffer_offset = target->buffer_offset + offsets[i] +
                                done * vs->xfb_strides[i],
               .buffer_size = MIN2((uint64_t)chunk * vs->xfb_strides[i],
                                  (uint64_t)target->buffer_size - offsets[i] -
                                     done * vs->xfb_strides[i]),
            };
         }
         buffers[5] = index_buffer;
         pctx->set_shader_buffers(pctx, MESA_SHADER_VERTEX,
                                  AGX_APPLE9_XFB_PARAMS, 6, buffers,
                                  vs->xfb_buffers_written << 1);
         struct pipe_draw_info capture = {.mode = MESA_PRIM_POINTS,
                                          .instance_count = 1};
         struct pipe_draw_start_count_bias range = {.count = chunk};
         pctx->draw_vbo(pctx, &capture, 0, NULL, &range, 1);
         pipe_resource_reference(&buffers[0].buffer, NULL);
         done += chunk;
      }
      /* Capture outputs can be the next draw's vertex or uniform inputs.
       * End this producer batch so the ordinary resource tracker orders that
       * consumer; draws within one Apple9 VDM batch lack a memory barrier. */
      agx_flush_batch_for_reason(ctx, agx_get_batch(ctx),
                                 "Transform feedback producer");
      ctx->apple9_xfb_capture = NULL;
      ctx->active_queries = saved_queries;
      pctx->set_shader_buffers(pctx, MESA_SHADER_VERTEX, AGX_APPLE9_XFB_PARAMS,
                               6, saved, saved_writable);
      for (unsigned i = 0; i < 6; ++i)
         pipe_resource_reference(&saved[i].buffer, NULL);
      pipe_resource_reference(&index_buffer.buffer, NULL);
      pctx->bind_fs_state(pctx, saved_fs);
      pctx->bind_rasterizer_state(pctx, saved_rs);
      pctx->set_framebuffer_state(pctx, &saved_fb);
      util_unreference_framebuffer_state(&saved_fb);
      ctx->dirty |= AGX_DIRTY_VS_PROG;
   }

   u_foreach_bit(i, vs->xfb_buffers_written) {
      struct pipe_stream_output_target *target = get_target(ctx, i);
      if (target) {
         uint32_t offset = offsets[i] + written * vertices * vs->xfb_strides[i];
         pipe_buffer_write(pctx, agx_so_target(target)->offset, 0, 4, &offset);
      }
   }
}
