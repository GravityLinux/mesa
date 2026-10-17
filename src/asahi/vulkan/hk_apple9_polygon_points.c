/*
 * Copyright 2026 Asahi Linux contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9_polygon_points.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compile_apple9.h"

static nir_def *
pp_sysval(nir_builder *b, unsigned offset, unsigned comps)
{
   return nir_load_ubo(b, comps, 32,
                       nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
                       nir_imm_int(b, offset), .align_mul = 4,
                       .range_base = offset, .range = comps * 4);
}

static nir_def *
pp_cross(nir_builder *b, nir_def *x, nir_def *y)
{
   nir_def *yzx = nir_swizzle(b, x, (unsigned[]){1, 2, 0}, 3);
   nir_def *zxy = nir_swizzle(b, x, (unsigned[]){2, 0, 1}, 3);
   return nir_fsub(b, nir_fmul(b, yzx, nir_swizzle(b, y, (unsigned[]){2, 0, 1}, 3)),
                   nir_fmul(b, zxy, nir_swizzle(b, y, (unsigned[]){1, 2, 0}, 3)));
}

/* One of the three vertices' value of a slot, by dynamic index. */
static nir_def *
pp_select(nir_builder *b, nir_def *index, nir_def *v[3])
{
   return nir_bcsel(b, nir_ieq_imm(b, index, 0), v[0],
                    nir_bcsel(b, nir_ieq_imm(b, index, 1), v[1], v[2]));
}

/*
 * Polygon mode POINT for one triangle. The triangle is culled by the winding
 * of its homogeneous determinant, clipped against w > 0, the depth planes when
 * depth clipping is enabled and the enabled clip distances, and each vertex of
 * the clipped polygon is emitted as a point. Clipped vertices are carried as
 * barycentric weights of the triangle, so every varying is interpolated
 * linearly in clip space. Flat varyings come from the provoking vertex, the
 * polygon slot from the triangle, and depth bias from its slope. Clip and cull
 * distances stay published, as inside values, so the points keep the
 * producer's varying layout. The frustum's x and y planes are left to the
 * point path, which drops points centred outside the viewport, as a
 * guard-band rasterizer does.
 */
void
hk_apple9_pp_emit_triangle(nir_builder *b, const struct hk_pp_info *info,
                           hk_pp_load_fn load, void *data, nir_def *provoking,
                           nir_def *prim, nir_def *only)
{
   unsigned clip = info->clip, cull = info->cull;
   nir_def *pos[3];
   nir_def *dist[3][16];
   for (unsigned i = 0; i < 3; ++i) {
      pos[i] = load(b, data, i, VARYING_SLOT_POS, 0, 0, 4, nir_type_float32);
      for (unsigned k = 0; k < clip + cull; ++k) {
         dist[i][k] = load(b, data, i, VARYING_SLOT_CLIP_DIST0, k / 4, k % 4, 1,
                           nir_type_float32);
      }
   }

   nir_def *flags = pp_sysval(b, AGX_APPLE9_POLYGON_POINT_OFFSET, 1);
   nir_def *params = pp_sysval(b, AGX_APPLE9_POLYGON_POINT_OFFSET + 4, 4);
   nir_def *depth_range = pp_sysval(b, AGX_APPLE9_POLYGON_POINT_OFFSET + 20, 1);
   nir_def *vp = pp_sysval(b, AGX_APPLE9_VIEWPORT_TABLE_OFFSET, 4);
#define FLAG(f) nir_ine_imm(b, nir_iand_imm(b, flags, AGX_APPLE9_PP_##f), 0)

   /* Winding from the homogeneous determinant of (x, y, w), which keeps its
    * meaning with vertices behind the eye. Framebuffer area is positive for
    * counter-clockwise triangles when det and the y scale differ in sign. */
   nir_def *r[3];
   for (unsigned i = 0; i < 3; ++i)
      r[i] = nir_vec3(b, nir_channel(b, pos[i], 0), nir_channel(b, pos[i], 1),
                      nir_channel(b, pos[i], 3));
   nir_def *det = nir_fdot3(b, r[0], pp_cross(b, r[1], r[2]));
   nir_def *signed_area = nir_fmul(b, det, nir_channel(b, vp, 1));
   nir_def *front = nir_bcsel(b, FLAG(FRONT_CCW),
                              nir_fle_imm(b, signed_area, 0.0),
                              nir_fge_imm(b, signed_area, 0.0));
   nir_def *culled = nir_bcsel(b, front, FLAG(CULL_FRONT), FLAG(CULL_BACK));
   for (unsigned k = clip; k < clip + cull; ++k) {
      culled = nir_ior(b, culled,
                       nir_iand(b, nir_flt_imm(b, dist[0][k], 0.0),
                                nir_iand(b, nir_flt_imm(b, dist[1][k], 0.0),
                                         nir_flt_imm(b, dist[2][k], 0.0))));
   }

   nir_function_impl *impl = b->impl;
   const struct glsl_type *list_type =
      glsl_array_type(glsl_vec_type(3), HK_PP_MAX_VERTICES + 1, 0);
   nir_variable *list[2] = {
      nir_local_variable_create(impl, list_type, "pp_a"),
      nir_local_variable_create(impl, list_type, "pp_b"),
   };
   nir_variable *count = nir_local_variable_create(impl, glsl_uint_type(), "pp_n");
   nir_variable *out_n = nir_local_variable_create(impl, glsl_uint_type(), "pp_m");
   nir_variable *j = nir_local_variable_create(impl, glsl_uint_type(), "pp_j");

   if (!only)
      nir_push_if(b, nir_inot(b, culled));
   {
      nir_store_array_var_imm(b, list[0], 0, nir_imm_vec3(b, 1, 0, 0), 0x7);
      nir_store_array_var_imm(b, list[0], 1, nir_imm_vec3(b, 0, 1, 0), 0x7);
      nir_store_array_var_imm(b, list[0], 2, nir_imm_vec3(b, 0, 0, 1), 0x7);
      nir_store_var(b, count, nir_imm_int(b, 3), 1);

      /* Plane distances at the three vertices; disabled depth planes are
       * always inside. */
      unsigned planes = 3 + clip;
      nir_def *depth_clip = FLAG(DEPTH_CLIP);
      unsigned cur = 0;
      for (unsigned p = 0; p < planes; ++p) {
         nir_def *d[3];
         for (unsigned i = 0; i < 3; ++i) {
            nir_def *z = nir_channel(b, pos[i], 2), *w = nir_channel(b, pos[i], 3);
            if (p == 0)
               d[i] = nir_fadd_imm(b, w, -0x1p-20);
            else if (p == 1)
               d[i] = nir_bcsel(b, depth_clip, z, nir_imm_float(b, 1.0));
            else if (p == 2)
               d[i] = nir_bcsel(b, depth_clip, nir_fsub(b, w, z),
                                nir_imm_float(b, 1.0));
            else
               d[i] = dist[i][p - 3];
         }
         nir_def *dv = nir_vec3(b, d[0], d[1], d[2]);
         nir_variable *in = list[cur], *out = list[cur ^ 1];
         nir_store_var(b, out_n, nir_imm_int(b, 0), 1);
         nir_store_var(b, j, nir_imm_int(b, 0), 1);
         nir_def *n = nir_load_var(b, count);
         nir_push_loop(b);
         {
            nir_def *jj = nir_load_var(b, j);
            nir_break_if(b, nir_uge(b, jj, n));
            nir_def *next = nir_bcsel(b, nir_ieq(b, nir_iadd_imm(b, jj, 1), n),
                                      nir_imm_int(b, 0), nir_iadd_imm(b, jj, 1));
            nir_def *a = nir_load_array_var(b, in, jj);
            nir_def *c = nir_load_array_var(b, in, next);
            nir_def *da = nir_fdot3(b, a, dv), *dc = nir_fdot3(b, c, dv);
            nir_def *a_in = nir_fge_imm(b, da, 0.0);
            nir_def *c_in = nir_fge_imm(b, dc, 0.0);
            nir_push_if(b, a_in);
            {
               nir_def *m = nir_load_var(b, out_n);
               nir_store_array_var(b, out, m, a, 0x7);
               nir_store_var(b, out_n, nir_iadd_imm(b, m, 1), 1);
            }
            nir_pop_if(b, NULL);
            nir_push_if(b, nir_ine(b, a_in, c_in));
            {
               nir_def *t = nir_fdiv(b, da, nir_fsub(b, da, dc));
               nir_def *m = nir_load_var(b, out_n);
               nir_store_array_var(b, out, m, nir_flrp(b, a, c, t), 0x7);
               nir_store_var(b, out_n, nir_iadd_imm(b, m, 1), 1);
            }
            nir_pop_if(b, NULL);
            nir_store_var(b, j, nir_iadd_imm(b, jj, 1), 1);
         }
         nir_pop_loop(b, NULL);
         nir_store_var(b, count, nir_load_var(b, out_n), 1);
         cur ^= 1;
      }

      /* Depth bias from the triangle's window-space slope, applied to the
       * points' clip-space depth. */
      nir_def *bias = nir_imm_float(b, 0.0);
      nir_push_if(b, FLAG(DEPTH_BIAS));
      nir_def *biased;
      {
         nir_def *win[3];
         for (unsigned i = 0; i < 3; ++i) {
            nir_def *rcp = nir_frcp(b, nir_channel(b, pos[i], 3));
            win[i] = nir_vec3(b,
               nir_fmul(b, nir_fmul(b, nir_channel(b, pos[i], 0), rcp),
                        nir_channel(b, vp, 0)),
               nir_fmul(b, nir_fmul(b, nir_channel(b, pos[i], 1), rcp),
                        nir_channel(b, vp, 1)),
               nir_fmul(b, nir_fmul(b, nir_channel(b, pos[i], 2), rcp),
                        depth_range));
         }
         nir_def *e1 = nir_fsub(b, win[1], win[0]);
         nir_def *e2 = nir_fsub(b, win[2], win[0]);
         nir_def *nrm = pp_cross(b, e1, e2);
         nir_def *nz = nir_channel(b, nrm, 2);
         nir_def *safe = nir_bcsel(b, nir_feq_imm(b, nz, 0.0),
                                   nir_imm_float(b, 1.0), nz);
         nir_def *dzdx = nir_fabs(b, nir_fdiv(b, nir_channel(b, nrm, 0), safe));
         nir_def *dzdy = nir_fabs(b, nir_fdiv(b, nir_channel(b, nrm, 1), safe));
         nir_def *slope = nir_bcsel(b, nir_feq_imm(b, nz, 0.0),
                                    nir_imm_float(b, 0.0), nir_fmax(b, dzdx, dzdy));
         nir_def *maxz = nir_fmax(b, nir_fabs(b, nir_channel(b, win[0], 2)),
                                  nir_fmax(b, nir_fabs(b, nir_channel(b, win[1], 2)),
                                           nir_fabs(b, nir_channel(b, win[2], 2))));
         /* Float formats: 2^(exponent of the largest depth - 23). */
         nir_def *float_unit = nir_fexp2(b, nir_fadd_imm(
            b, nir_ffloor(b, nir_flog2(b, nir_fmax(b, maxz, nir_imm_float(b, 0x1p-126)))),
            -23.0));
         nir_def *unit = nir_bcsel(b, FLAG(DEPTH_BIAS_FLOAT), float_unit,
                                   nir_channel(b, params, 3));
         nir_def *o = nir_fadd(b, nir_fmul(b, nir_channel(b, params, 1), slope),
                               nir_fmul(b, nir_channel(b, params, 0), unit));
         nir_def *clamp = nir_channel(b, params, 2);
         o = nir_bcsel(b, nir_fgt_imm(b, clamp, 0.0), nir_fmin(b, o, clamp),
                       nir_bcsel(b, nir_flt_imm(b, clamp, 0.0),
                                 nir_fmax(b, o, clamp), o));
         biased = nir_bcsel(b, nir_feq_imm(b, depth_range, 0.0),
                            nir_imm_float(b, 0.0), nir_fdiv(b, o, depth_range));
      }
      nir_pop_if(b, NULL);
      bias = nir_if_phi(b, biased, bias);

      /* Per-slot values of the three vertices, loaded once. */
      unsigned nslots = 0;
      unsigned slots[VARYING_SLOT_MAX];
      nir_def *vals[VARYING_SLOT_MAX][3];
      u_foreach_bit64(loc, info->outputs) {
         if (loc == VARYING_SLOT_POS)
            continue;
         unsigned comps = info->output_components[loc];
         bool flat = (info->flat & BITFIELD64_BIT(loc)) ||
                     loc == VARYING_SLOT_LAYER || loc == VARYING_SLOT_VIEWPORT;
         for (unsigned i = 0; i < 3; ++i) {
            vals[nslots][i] = load(b, data, i, loc, 0, 0, comps,
                                   flat ? nir_type_uint32 : nir_type_float32);
         }
         slots[nslots++] = loc;
      }

      nir_variable *final = list[cur];
      nir_store_var(b, j, nir_imm_int(b, 0), 1);
      nir_def *n = nir_load_var(b, count);
      /* A single point, of a vertex shader drawing a fixed number of points
       * per triangle; missing vertices lie behind the eye. */
      nir_def *visible = NULL;
      if (only) {
         visible = nir_iand(b, nir_inot(b, culled), nir_ult(b, only, n));
         nir_store_var(b, j, nir_umin(b, only, nir_iadd_imm(b, nir_umax_imm(b, n, 1), -1)),
                       1);
      } else {
         nir_push_loop(b);
      }
      {
         nir_def *jj = nir_load_var(b, j);
         if (!only)
            nir_break_if(b, nir_uge(b, jj, n));
         nir_def *w = nir_load_array_var(b, final, jj);
         nir_def *p = nir_fadd(b, nir_fmul(b, pos[0], nir_channel(b, w, 0)),
                               nir_fadd(b, nir_fmul(b, pos[1], nir_channel(b, w, 1)),
                                        nir_fmul(b, pos[2], nir_channel(b, w, 2))));
         p = nir_vector_insert_imm(
            b, p, nir_fadd(b, nir_channel(b, p, 2),
                           nir_fmul(b, bias, nir_channel(b, p, 3))), 2);
         if (only)
            p = nir_bcsel(b, visible, p, nir_imm_vec4(b, 0.0, 0.0, 0.0, -1.0));
         nir_store_output(b, p, nir_imm_int(b, 0), .src_type = nir_type_float32,
                          .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         for (unsigned s = 0; s < nslots; ++s) {
            unsigned loc = slots[s];
            unsigned comps = info->output_components[loc];
            nir_io_semantics sem = {.location = loc, .num_slots = 1};
            bool flat = (info->flat & BITFIELD64_BIT(loc)) ||
                        loc == VARYING_SLOT_LAYER || loc == VARYING_SLOT_VIEWPORT;
            nir_def *value;
            if (flat) {
               value = pp_select(b, provoking, vals[s]);
            } else {
               value = nir_fadd(b, nir_fmul(b, vals[s][0], nir_channel(b, w, 0)),
                                nir_fadd(b, nir_fmul(b, vals[s][1], nir_channel(b, w, 1)),
                                         nir_fmul(b, vals[s][2], nir_channel(b, w, 2))));
            }
            for (unsigned c = 0; c < comps; ++c) {
               nir_store_output(b, nir_channel(b, value, c), nir_imm_int(b, 0),
                                .io_semantics = sem, .component = c,
                                .src_type = nir_type_uint32);
            }
         }
         for (unsigned k = 0; k < clip + cull; ++k) {
            nir_store_output(b, nir_imm_float(b, 1.0), nir_imm_int(b, k / 4),
                             .component = k % 4, .src_type = nir_type_float32,
                             .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0,
                                              .num_slots = clip + cull});
         }
         /* The polygon slot: back-facing, and the polygon's primitive ID in
          * place of the point's. */
         if (info->face_slot) {
            nir_store_output(b, nir_vec2(b, nir_b2i32(b, nir_inot(b, front)), prim),
                             nir_imm_int(b, 0), .src_type = nir_type_uint32,
                             .write_mask = 0x3,
                             .io_semantics = {.location = info->face_slot,
                                              .num_slots = 1});
         }
         if (!only) {
            nir_emit_vertex(b, 0);
            nir_end_primitive(b, 0);
            nir_store_var(b, j, nir_iadd_imm(b, jj, 1), 1);
         }
      }
      if (!only)
         nir_pop_loop(b, NULL);
   }
   if (!only)
      nir_pop_if(b, NULL);
#undef FLAG
}
