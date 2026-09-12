/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "compiler/nir/nir_builder.h"
#include "gallium/include/pipe/p_defines.h"
#include "agx_compile_apple9.h"

/* A texel offset is applied separately at each selected mip level, before
 * wrapping and filtering. Express that contract with ordinary NIR fetches;
 * the backend supplies its private nearest sampler for those fetches. */
static nir_def *
texture_info(nir_builder *b, unsigned texture, unsigned field)
{
   return nir_load_ubo(
      b, 1, 32, nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_TEXTURE_INFO_OFFSET +
                        texture * AGX_APPLE9_TEXTURE_INFO_STRIDE + field * 4),
      .align_mul = 4, .range = 4);
}

static nir_def *
wrap_texel(nir_builder *b, nir_def *i, nir_def *size, unsigned wrap)
{
   if (wrap == PIPE_TEX_WRAP_REPEAT)
      return nir_imod(b, i, size);
   if (wrap == PIPE_TEX_WRAP_MIRROR_REPEAT) {
      nir_def *period = nir_ishl_imm(b, size, 1);
      nir_def *j = nir_imod(b, i, period);
      return nir_imin(b, j, nir_isub(b, nir_iadd_imm(b, period, -1), j));
   }
   return nir_iclamp(b, i, nir_imm_int(b, 0), nir_iadd_imm(b, size, -1));
}

static nir_def *
compare_depth(nir_builder *b, nir_def *ref, nir_def *depth, unsigned func)
{
   nir_def *test;
   switch (func) {
   case PIPE_FUNC_NEVER:
      return nir_imm_float(b, 0);
   case PIPE_FUNC_ALWAYS:
      return nir_imm_float(b, 1);
   case PIPE_FUNC_LESS:
      test = nir_flt(b, ref, depth);
      break;
   case PIPE_FUNC_LEQUAL:
      test = nir_fge(b, depth, ref);
      break;
   case PIPE_FUNC_GREATER:
      test = nir_flt(b, depth, ref);
      break;
   case PIPE_FUNC_GEQUAL:
      test = nir_fge(b, ref, depth);
      break;
   case PIPE_FUNC_EQUAL:
      test = nir_feq(b, ref, depth);
      break;
   default:
      test = nir_fneu(b, ref, depth);
      break;
   }
   return nir_b2f32(b, test);
}

struct offset_sample {
   nir_tex_instr *tex;
   const struct agx_apple9_sampler_key *state;
   unsigned dims;
   nir_def *coord, *offset, *comparator, *layer;
   nir_def *base_size[3];
};

static nir_def *
fetch(nir_builder *b, const struct offset_sample *s, nir_def *coord,
      nir_def *lod)
{
   nir_tex_instr *txf = nir_tex_instr_create(b->shader, 2);
   txf->op = nir_texop_txf;
   txf->sampler_dim = s->tex->sampler_dim;
   txf->is_array = s->tex->is_array;
   txf->texture_index = s->tex->texture_index;
   txf->sampler_index = s->tex->sampler_index;
   txf->coord_components = coord->num_components;
   txf->dest_type = s->tex->dest_type;
   txf->src[0] = (nir_tex_src){.src_type = nir_tex_src_coord,
                               .src = nir_src_for_ssa(coord)};
   txf->src[1] =
      (nir_tex_src){.src_type = nir_tex_src_lod, .src = nir_src_for_ssa(lod)};
   nir_def_init(&txf->instr, &txf->def,
                s->tex->is_shadow ? 4 : s->tex->def.num_components, 32);
   nir_builder_instr_insert(b, &txf->instr);
   if (s->tex->is_shadow)
      return compare_depth(b, s->comparator, nir_channel(b, &txf->def, 0),
                           s->state->compare_func);
   return &txf->def;
}

static nir_def *
filter_level(nir_builder *b, const struct offset_sample *s, nir_def *lod,
             bool linear)
{
   nir_def *size[3], *base[3], *weight[3];
   for (unsigned c = 0; c < s->dims; ++c) {
      size[c] =
         nir_imax(b, nir_ushr(b, s->base_size[c], lod), nir_imm_int(b, 1));
      nir_def *p =
         nir_fmul(b, nir_channel(b, s->coord, c), nir_u2f32(b, size[c]));
      if (linear)
         p = nir_fadd_imm(b, p, -0.5);
      nir_def *floor = nir_ffloor(b, p);
      base[c] = nir_iadd(b, nir_f2i32(b, floor), nir_channel(b, s->offset, c));
      weight[c] = nir_fsub(b, p, floor);
   }
   unsigned count = linear ? 1u << s->dims : 1;
   nir_def *taps[8];
   for (unsigned tap = 0; tap < count; ++tap) {
      nir_def *coord[4];
      for (unsigned c = 0; c < s->dims; ++c)
         coord[c] = wrap_texel(b, nir_iadd_imm(b, base[c], (tap >> c) & 1),
                               size[c], s->state->wrap[c]);
      if (s->tex->is_array)
         coord[s->dims] = s->layer;
      taps[tap] =
         fetch(b, s, nir_vec(b, coord, s->dims + s->tex->is_array), lod);
   }
   /* Pair adjacent X taps, then Y and Z; no fixed registers or sampled values
    * escape this ordinary SSA filtering expression. */
   for (unsigned c = 0; linear && c < s->dims; ++c) {
      count /= 2;
      for (unsigned i = 0; i < count; ++i)
         taps[i] = nir_flrp(b, taps[2 * i], taps[2 * i + 1], weight[c]);
   }
   return taps[0];
}

static nir_def *
first_level(nir_builder *b, const struct offset_sample *s, nir_def *lod,
            nir_def *lambda)
{
   bool min = s->state->min_filter == PIPE_TEX_FILTER_LINEAR;
   bool mag = s->state->mag_filter == PIPE_TEX_FILTER_LINEAR;
   if (min == mag)
      return filter_level(b, s, lod, min);
   nir_push_if(b, nir_fge(b, nir_imm_float(b, 0), lambda));
   nir_def *magnified = filter_level(b, s, lod, mag);
   nir_push_else(b, NULL);
   nir_def *minified = filter_level(b, s, lod, min);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, magnified, minified);
}

static nir_def *
sample_lod(nir_builder *b, const struct offset_sample *s)
{
   nir_tex_instr *tex = s->tex;
   int explicit_lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *lod;
   if (explicit_lod >= 0) {
      lod = tex->src[explicit_lod].src.ssa;
   } else if (tex->op != nir_texop_txd &&
              b->shader->info.stage != MESA_SHADER_FRAGMENT) {
      lod = nir_imm_float(b, 0);
   } else {
      nir_def *dx, *dy;
      if (tex->op == nir_texop_txd) {
         dx = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddx)].src.ssa;
         dy = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddy)].src.ssa;
      } else {
         nir_def *spatial = nir_trim_vector(b, s->coord, s->dims);
         dx = nir_ddx(b, spatial);
         dy = nir_ddy(b, spatial);
      }
      nir_def *xx = nir_imm_float(b, 0), *yy = nir_imm_float(b, 0);
      for (unsigned c = 0; c < s->dims; ++c) {
         nir_def *scale = nir_u2f32(b, s->base_size[c]);
         nir_def *x = nir_fmul(b, nir_channel(b, dx, c), scale);
         nir_def *y = nir_fmul(b, nir_channel(b, dy, c), scale);
         xx = nir_fadd(b, xx, nir_fmul(b, x, x));
         yy = nir_fadd(b, yy, nir_fmul(b, y, y));
      }
      lod = nir_fmul_imm(b, nir_flog2(b, nir_fmax(b, xx, yy)), 0.5);
   }
   int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (bias >= 0)
      lod = nir_fadd(b, lod, tex->src[bias].src.ssa);
   lod = nir_fadd_imm(b, lod, s->state->lod_bias);
   return nir_fclamp(b, lod, nir_imm_float(b, s->state->min_lod),
                     nir_imm_float(b, s->state->max_lod));
}

bool
agx_nir_lower_apple9_texture_offsets(nir_shader *nir,
                                     const struct agx_apple9_sampler_key key[32],
                                     const char **reason)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   bool progress = false;
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
         if (offset < 0 || tex->op == nir_texop_txf)
            continue;
         if (tex->texture_index >= 32 || tex->sampler_index >= 32 ||
             (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
              tex->sampler_dim != GLSL_SAMPLER_DIM_3D)) {
            *reason =
               "Apple9 offset filtering requires a bound 2D, array, or 3D texture";
            return false;
         }
         struct offset_sample s = {
            .tex = tex,
            .state = &key[tex->sampler_index],
            .dims = tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 2,
            .coord =
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa,
            .offset = tex->src[offset].src.ssa};
         for (unsigned c = 0; c < s.dims; ++c) {
            unsigned wrap = s.state->wrap[c];
            if (wrap != PIPE_TEX_WRAP_REPEAT &&
                wrap != PIPE_TEX_WRAP_MIRROR_REPEAT &&
                wrap != PIPE_TEX_WRAP_CLAMP_TO_EDGE) {
               *reason =
                  "Apple9 offset filtering requires repeat, mirror repeat, or edge clamping";
               return false;
            }
         }
         b.cursor = nir_before_instr(instr);
         for (unsigned c = 0; c < s.dims; ++c)
            s.base_size[c] = texture_info(&b, tex->texture_index, c);
         if (tex->is_array) {
            s.layer = nir_f2i32(
               &b,
               nir_ffloor(
                  &b, nir_fadd_imm(&b, nir_channel(&b, s.coord, s.dims), 0.5)));
            s.layer = nir_iclamp(
               &b, s.layer, nir_imm_int(&b, 0),
               nir_iadd_imm(&b, texture_info(&b, tex->texture_index, 2), -1));
         }
         if (tex->is_shadow)
            s.comparator =
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_comparator)]
                  .src.ssa;
         nir_def *lambda = sample_lod(&b, &s);
         nir_def *last =
            nir_iadd_imm(&b, texture_info(&b, tex->texture_index, 3), -1);
         nir_def *level =
            nir_fclamp(&b, lambda, nir_imm_float(&b, 0), nir_i2f32(&b, last));
         nir_def *lod =
            s.state->mip_filter == PIPE_TEX_MIPFILTER_NONE ? nir_imm_int(&b, 0)
            : s.state->mip_filter == PIPE_TEX_MIPFILTER_NEAREST
               ? nir_imax(&b,
                          nir_iadd_imm(
                             &b,
                             nir_f2i32(&b, nir_fceil(&b, nir_fadd_imm(&b, level,
                                                                      0.5))),
                             -1),
                          nir_imm_int(&b, 0))
               : nir_f2i32(&b, nir_ffloor(&b, level));
         nir_def *value = first_level(&b, &s, lod, lambda);
         if (s.state->mip_filter == PIPE_TEX_MIPFILTER_LINEAR) {
            nir_def *next = nir_imin(&b, nir_iadd_imm(&b, lod, 1), last);
            nir_def *second = filter_level(
               &b, &s, next, s.state->min_filter == PIPE_TEX_FILTER_LINEAR);
            value = nir_flrp(&b, value, second, nir_ffract(&b, level));
         }
         nir_def_rewrite_uses(&tex->def, value);
         nir_instr_remove(instr);
         progress = true;
      }
   }
   nir->info.flrp_lowered = false;
   nir_progress(progress, impl, nir_metadata_none);
   return true;
}
