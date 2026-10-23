/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_nir_texture.h"
#include "compiler/nir/nir_builder.h"
#include "gallium/include/pipe/p_defines.h"
#include "agx_compile_apple9.h"

void
agx_nir_legalize_apple9_texture_sources(nir_shader *nir)
{
   /* Frontends and common image lowering can supply narrow LODs/coordinates.
    * Native lowering performs arithmetic in 32 bits before packing operands. */
   nir_tex_src_type_constraints constraints = {
      [nir_tex_src_coord] = {true, 32},
      [nir_tex_src_lod] = {true, 32},
      [nir_tex_src_bias] = {true, 32},
      [nir_tex_src_ms_index] = {true, 32},
      [nir_tex_src_comparator] = {true, 32},
      [nir_tex_src_ddx] = {true, 32},
      [nir_tex_src_ddy] = {true, 32},
      [nir_tex_src_offset] = {true, 32},
      [nir_tex_src_min_lod] = {true, 32},
   };
   nir_legalize_16bit_sampler_srcs(nir, constraints);
}

static bool
lower_texture_lod(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_txl && tex->op != nir_texop_txb &&
       tex->op != nir_texop_txf)
      return false;

   /* A separate source type makes this pass idempotent and prevents later
    * selection from treating packed integer bits as a floating-point LOD. */
   if (nir_tex_instr_src_index(tex, nir_tex_src_backend1) >= 0)
      return false;
   int source = nir_tex_instr_src_index(
      tex, tex->op == nir_texop_txb ? nir_tex_src_bias : nir_tex_src_lod);
   if (source < 0 || tex->src[source].src.ssa->bit_size != 32 ||
       tex->src[source].src.ssa->num_components != 1)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *lod = tex->src[source].src.ssa;
   nir_def *packed;
   if (tex->op == nir_texop_txf) {
      /* Integer levels occupy the integral part of the same signed Q6 field. */
      packed = nir_ishl_imm(b, lod, 22);
   } else {
      /* Preserve the established clamp order and floor, including negative
       * fractional inputs. Expose the entire conversion before preamble
       * extraction, so uniform LOD and bias do not repeat it in the main. */
      uint32_t fp_math_ctrl = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_fp_no_fast_math;
      nir_def *clamped = nir_fmin(b, nir_fmax(b, lod, nir_imm_float(b, -32.0f)),
                                  nir_imm_float(b, 2047.0f / 64.0f));
      nir_def *fixed =
         nir_f2i32(b, nir_ffloor(b, nir_fmul_imm(b, clamped, 64.0f)));
      packed = nir_ishl_imm(b, nir_iand_imm(b, fixed, 0xfff), 16);
      b->fp_math_ctrl = fp_math_ctrl;
   }
   nir_src_rewrite(&tex->src[source].src, packed);
   tex->src[source].src_type = nir_tex_src_backend1;
   return true;
}

bool
agx_nir_lower_apple9_texture_lod(nir_shader *shader)
{
   return nir_shader_instructions_pass(shader, lower_texture_lod,
                                       nir_metadata_control_flow, NULL);
}

/* A texel offset is applied separately at each selected mip level, before
 * wrapping and filtering. Express that contract with ordinary NIR fetches;
 * the backend supplies its private nearest sampler for those fetches. */
/* A handle-addressed descriptor carries the records otherwise published in
 * the per-draw sysvals. */
static nir_def *
descriptor_word(nir_builder *b, nir_def *handle, unsigned offset)
{
   nir_def *address = nir_pack_64_2x32_split(
      b, nir_iadd_imm(b, handle, offset),
      nir_imm_int(b, AGX_APPLE9_USC_BASE >> 32));
   /* EXP: descriptor records are constant for the draw; speculating them
    * lets the preamble hoist uniform sampler work out of control flow. */
   return nir_load_global_constant(b, 1, 32, address, .align_mul = 4,
      .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
}

nir_def *
agx_apple9_texture_info(nir_builder *b, const nir_tex_instr *tex, unsigned field)
{
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle >= 0)
      return descriptor_word(b, tex->src[handle].src.ssa,
                             AGX_APPLE9_DESCRIPTOR_TEXTURE_INFO + field * 4);
   return nir_load_ubo(
      b, 1, 32, nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_TEXTURE_INFO_OFFSET +
                        tex->texture_index * AGX_APPLE9_TEXTURE_INFO_STRIDE +
                        field * 4),
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
   nir_def *wrap[3], *min, *mag, *mip, *compare, *min_lod, *max_lod, *bias,
      *border;
};

nir_def *
agx_apple9_sampler_field(nir_builder *b, const nir_tex_instr *tex, unsigned field)
{
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (handle >= 0)
      return descriptor_word(b, tex->src[handle].src.ssa,
                             AGX_APPLE9_DESCRIPTOR_RUNTIME_SAMPLER + field * 4);
   return nir_load_ubo(
      b, 1, 32, nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_RUNTIME_SAMPLER_OFFSET +
                        tex->sampler_index * AGX_APPLE9_RUNTIME_SAMPLER_STRIDE +
                        4 * field),
      .align_mul = 4, .range = 4);
}

static bool
normalize_runtime_coordinates(nir_builder *b, nir_tex_instr *tex,
                              UNUSED void *data)
{
   /* Unnormalized samplers are restricted to single-level 1D/2D views.
    * The frontend has already expanded 1D to 2D with unit height, and may
    * have rewritten 2D as layer 0 of a one-layer array. Vulkan also forbids
    * them with ImplicitLod, Dref and Proj sampling and with any bias or
    * offset (VUID-vkCmdDraw-None-02703/02704), so those samples never take
    * pixel coordinates. */
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       (tex->is_array && !(tex->backend_flags & AGX_TEXTURE_FLAG_ARRAYED_2D)) ||
       nir_tex_instr_is_query(tex) || tex->op == nir_texop_txf ||
       tex->op == nir_texop_tex || tex->op == nir_texop_txb ||
       tex->is_shadow ||
       nir_tex_instr_src_index(tex, nir_tex_src_projector) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0)
      return false;
   int src = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (src < 0 || tex->texture_index >= 32 || tex->sampler_index >= 32)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *pixel =
      agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_PIXEL_COORDS);
   nir_def *coord = tex->src[src].src.ssa;
   nir_def *normalized[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < coord->num_components; ++c)
      normalized[c] = c < 2 ? nir_fdiv(b, nir_channel(b, coord, c),
                                       nir_u2f32(b, agx_apple9_texture_info(b, tex, c)))
                            : nir_channel(b, coord, c);
   nir_src_rewrite(
      &tex->src[src].src,
      nir_bcsel(b, nir_ine_imm(b, pixel, 0),
                nir_vec(b, normalized, coord->num_components), coord));
   return true;
}

/* Explicit-LOD sample of tex's view and sampler outside the image on every
 * axis (keeping an array layer), as a 32-bit vec4. */
static nir_def *
sample_border(nir_builder *b, nir_tex_instr *tex, nir_def *coord, unsigned dims)
{
   static const nir_tex_src_type copied[] = {
      nir_tex_src_texture_deref, nir_tex_src_sampler_deref,
      nir_tex_src_texture_offset, nir_tex_src_sampler_offset,
      nir_tex_src_texture_handle, nir_tex_src_sampler_handle,
   };
   unsigned count = 2;
   for (unsigned i = 0; i < ARRAY_SIZE(copied); ++i)
      count += nir_tex_instr_src_index(tex, copied[i]) >= 0;
   nir_tex_instr *bt = nir_tex_instr_create(b->shader, count);
   bt->op = nir_texop_txl;
   bt->sampler_dim = tex->sampler_dim;
   bt->is_array = tex->is_array;
   bt->coord_components = coord->num_components;
   bt->dest_type = nir_alu_type_get_base_type(tex->dest_type) | 32;
   bt->texture_index = tex->texture_index;
   bt->sampler_index = tex->sampler_index;
   bt->backend_flags = tex->backend_flags | AGX_APPLE9_TEXTURE_BIASED_LEVEL;
   bt->can_speculate = tex->can_speculate;
   nir_def *comps[4];
   for (unsigned c = 0; c < coord->num_components; ++c)
      comps[c] = c < dims ? nir_imm_float(b, -16.0f) : nir_channel(b, coord, c);
   unsigned n = 0;
   bt->src[n++] = (nir_tex_src){
      .src_type = nir_tex_src_coord,
      .src = nir_src_for_ssa(nir_vec(b, comps, coord->num_components))};
   bt->src[n++] = (nir_tex_src){.src_type = nir_tex_src_lod,
                                .src = nir_src_for_ssa(nir_imm_float(b, 0))};
   for (unsigned i = 0; i < ARRAY_SIZE(copied); ++i) {
      int src = nir_tex_instr_src_index(tex, copied[i]);
      if (src >= 0)
         bt->src[n++] = (nir_tex_src){.src_type = copied[i],
                                      .src = nir_src_for_ssa(tex->src[src].src.ssa)};
   }
   nir_def_init(&bt->instr, &bt->def, 4, 32);
   nir_builder_instr_insert(b, &bt->instr);
   return &bt->def;
}

static nir_def *
wrap_dynamic(nir_builder *b, nir_def *i, nir_def *size, nir_def *wrap)
{
   nir_def *edge = wrap_texel(b, i, size, PIPE_TEX_WRAP_CLAMP_TO_EDGE);
   nir_def *repeat = wrap_texel(b, i, size, PIPE_TEX_WRAP_REPEAT);
   nir_def *mirror = wrap_texel(b, i, size, PIPE_TEX_WRAP_MIRROR_REPEAT);
   nir_def *mirror_edge =
      nir_imin(b,
               nir_bcsel(b, nir_ilt_imm(b, i, 0),
                         nir_iadd_imm(b, nir_ineg(b, i), -1), i),
               nir_iadd_imm(b, size, -1));
   return nir_bcsel(
      b, nir_ieq_imm(b, wrap, PIPE_TEX_WRAP_REPEAT), repeat,
      nir_bcsel(
         b, nir_ieq_imm(b, wrap, PIPE_TEX_WRAP_MIRROR_REPEAT), mirror,
         nir_bcsel(b, nir_ieq_imm(b, wrap, PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE),
                   mirror_edge, edge)));
}

static nir_def *
fetch(nir_builder *b, const struct offset_sample *s, nir_def *coord,
      nir_def *lod, nir_def *border)
{
   int handle = nir_tex_instr_src_index(s->tex, nir_tex_src_texture_handle);
   nir_tex_instr *txf = nir_tex_instr_create(b->shader, 2 + (handle >= 0));
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
   if (handle >= 0) {
      txf->src[2] = (nir_tex_src){
         .src_type = nir_tex_src_texture_handle,
         .src = nir_src_for_ssa(s->tex->src[handle].src.ssa)};
   }
   nir_def_init(&txf->instr, &txf->def,
                s->tex->is_shadow ? 4 : s->tex->def.num_components, 32);
   nir_builder_instr_insert(b, &txf->instr);
   nir_def *value = nir_bcsel(
      b, border, nir_trim_vector(b, s->border, txf->def.num_components),
      &txf->def);
   if (!s->tex->is_shadow)
      return value;
   if (s->state)
      return compare_depth(b, s->comparator, nir_channel(b, value, 0),
                           s->state->compare_func);
   nir_def *result = nir_imm_float(b, 0);
   for (unsigned f = PIPE_FUNC_NEVER; f <= PIPE_FUNC_ALWAYS; ++f)
      result = nir_bcsel(
         b, nir_ieq_imm(b, s->compare, f),
         compare_depth(b, s->comparator, nir_channel(b, value, 0), f), result);
   return result;
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
      nir_def *border = nir_imm_false(b);
      for (unsigned c = 0; c < s->dims; ++c) {
         nir_def *i = nir_iadd_imm(b, base[c], (tap >> c) & 1);
         coord[c] = s->state ? wrap_texel(b, i, size[c], s->state->wrap[c])
                             : wrap_dynamic(b, i, size[c], s->wrap[c]);
         border = nir_ior(
            b, border,
            nir_iand(b,
                     nir_ieq_imm(b, s->wrap[c], PIPE_TEX_WRAP_CLAMP_TO_BORDER),
                     nir_uge(b, i, size[c])));
      }
      if (s->tex->is_array)
         coord[s->dims] = s->layer;
      taps[tap] = fetch(b, s, nir_vec(b, coord, s->dims + s->tex->is_array),
                        lod, border);
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
filter_dynamic(nir_builder *b, const struct offset_sample *s, nir_def *lod,
               nir_def *filter)
{
   /* Integer textures cannot be linearly filtered. Avoid emitting dead float
    * interpolation of integer texels even before constant folding. */
   if (!s->tex->is_shadow &&
       nir_alu_type_get_base_type(s->tex->dest_type) != nir_type_float)
      return filter_level(b, s, lod, false);
   nir_push_if(b, nir_ieq_imm(b, filter, PIPE_TEX_FILTER_LINEAR));
   nir_def *linear = filter_level(b, s, lod, true);
   nir_push_else(b, NULL);
   nir_def *nearest = filter_level(b, s, lod, false);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, linear, nearest);
}

static nir_def *
first_level(nir_builder *b, const struct offset_sample *s, nir_def *lod,
            nir_def *lambda)
{
   if (!s->state) {
      nir_def *filter =
         nir_bcsel(b, nir_fge(b, nir_imm_float(b, 0), lambda), s->mag, s->min);
      return filter_dynamic(b, s, lod, filter);
   }
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

/* textureGather returns one component of each texel in the level-zero 2x2
 * bilinear footprint, ordered (i0, j1), (i1, j1), (i1, j0), (i0, j0). Depth
 * comparisons apply to each texel.
 */
static nir_def *
gather(nir_builder *b, const struct offset_sample *s)
{
   nir_def *lod = nir_imm_int(b, 0);
   nir_def *base[2];
   for (unsigned c = 0; c < 2; ++c) {
      nir_def *p = nir_fadd_imm(b, nir_fmul(b, nir_channel(b, s->coord, c),
                                            nir_u2f32(b, s->base_size[c])),
                                -0.5);
      base[c] = nir_f2i32(b, nir_ffloor(b, p));
      if (s->offset)
         base[c] = nir_iadd(b, base[c], nir_channel(b, s->offset, c));
   }
   static const uint8_t taps[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
   nir_def *result[4];
   for (unsigned tap = 0; tap < 4; ++tap) {
      nir_def *coord[3];
      nir_def *border = nir_imm_false(b);
      for (unsigned c = 0; c < 2; ++c) {
         nir_def *i = nir_iadd_imm(b, base[c], taps[tap][c]);
         coord[c] = s->state ? wrap_texel(b, i, s->base_size[c], s->state->wrap[c])
                             : wrap_dynamic(b, i, s->base_size[c], s->wrap[c]);
         border = nir_ior(
            b, border,
            nir_iand(b,
                     nir_ieq_imm(b, s->wrap[c], PIPE_TEX_WRAP_CLAMP_TO_BORDER),
                     nir_uge(b, i, s->base_size[c])));
      }
      if (s->tex->is_array)
         coord[2] = s->layer;
      nir_def *value = fetch(b, s, nir_vec(b, coord, 2 + s->tex->is_array),
                             lod, border);
      result[tap] = s->tex->is_shadow ? value
                                      : nir_channel(b, value, s->tex->component);
   }
   return nir_vec(b, result, 4);
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
   lod = nir_fadd(b, lod, s->bias);
   /* A shader LOD clamp raises the sampler's minimum. */
   nir_def *min_lod = s->min_lod;
   int shader_min_lod = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (shader_min_lod >= 0)
      min_lod = nir_fmax(b, min_lod, tex->src[shader_min_lod].src.ssa);
   return nir_fclamp(b, lod, min_lod, s->max_lod);
}

/* A native explicit-LOD sample of one mip level, with the texel offset applied
 * as a coordinate shift. Vulkan adds offsets to integer texel coordinates
 * before wrapping and filtering, so at a single level the offset equals
 * offset / level_size in normalised coordinates; the shift's rounding error
 * (about one ulp of the coordinate) is far below the sub-texel precision.
 */
static nir_def *
native_offset_level(nir_builder *b, const struct offset_sample *s,
                    nir_def *level)
{
   nir_tex_instr *tex = s->tex;
   static const nir_tex_src_type copied[] = {
      nir_tex_src_texture_deref, nir_tex_src_sampler_deref,
      nir_tex_src_texture_offset, nir_tex_src_sampler_offset,
      nir_tex_src_texture_handle, nir_tex_src_sampler_handle,
      nir_tex_src_comparator,
   };
   unsigned count = 2;
   for (unsigned i = 0; i < ARRAY_SIZE(copied); ++i)
      count += nir_tex_instr_src_index(tex, copied[i]) >= 0;
   nir_def *comps[4];
   for (unsigned c = 0; c < s->coord->num_components; ++c) {
      comps[c] = nir_channel(b, s->coord, c);
      if (c < s->dims) {
         nir_def *size =
            nir_umax(b, nir_ushr(b, s->base_size[c], level), nir_imm_int(b, 1));
         nir_def *shift = nir_fmul(b, nir_i2f32(b, nir_channel(b, s->offset, c)),
                                   nir_frcp(b, nir_u2f32(b, size)));
         comps[c] = nir_fadd(b, comps[c], shift);
      }
   }
   nir_tex_instr *nt = nir_tex_instr_create(b->shader, count);
   nt->op = nir_texop_txl;
   nt->sampler_dim = tex->sampler_dim;
   nt->is_array = tex->is_array;
   nt->is_shadow = tex->is_shadow;
   nt->is_new_style_shadow = tex->is_new_style_shadow;
   nt->coord_components = s->coord->num_components;
   nt->dest_type = tex->dest_type;
   nt->texture_index = tex->texture_index;
   nt->sampler_index = tex->sampler_index;
   nt->backend_flags = tex->backend_flags | AGX_APPLE9_TEXTURE_BIASED_LEVEL;
   nt->can_speculate = tex->can_speculate;
   unsigned n = 0;
   nt->src[n++] = (nir_tex_src){
      .src_type = nir_tex_src_coord,
      .src = nir_src_for_ssa(nir_vec(b, comps, s->coord->num_components))};
   nt->src[n++] = (nir_tex_src){.src_type = nir_tex_src_lod,
                                .src = nir_src_for_ssa(nir_i2f32(b, level))};
   for (unsigned i = 0; i < ARRAY_SIZE(copied); ++i) {
      int src = nir_tex_instr_src_index(tex, copied[i]);
      if (src >= 0)
         nt->src[n++] = (nir_tex_src){.src_type = copied[i],
                                      .src = nir_src_for_ssa(tex->src[src].src.ssa)};
   }
   nir_def_init(&nt->instr, &nt->def, tex->def.num_components, tex->def.bit_size);
   nir_builder_instr_insert(b, &nt->instr);
   return &nt->def;
}

bool
agx_apple9_texture_native_offset(const nir_tex_instr *tex, uint16_t *bits)
{
   int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   /* The shape of an encodable offset. Whether the sample's final form has
    * a measured offset variant is decided by the backend, which lowers the
    * offset otherwise (AGX_APPLE9_TEXTURE_OFFSET_FALLBACK). */
   if (offset < 0 || !nir_src_is_const(tex->src[offset].src) ||
       tex->op == nir_texop_txf ||
       (tex->backend_flags & AGX_APPLE9_TEXTURE_OFFSET_FALLBACK) ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
        tex->op != nir_texop_txl && tex->op != nir_texop_txd &&
        tex->op != nir_texop_tg4) ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
        tex->sampler_dim != GLSL_SAMPLER_DIM_3D) ||
       /* The minimum-LOD forms are native only without an offset. */
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0)
      return false;
   unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 2;
   nir_src src = tex->src[offset].src;
   if (src.ssa->num_components < dims)
      return false;
   uint16_t packed = 0;
   for (unsigned c = 0; c < dims; ++c) {
      int64_t v = nir_src_comp_as_int(src, c);
      if (v < -8 || v > 7)
         return false;
      packed |= (uint16_t)(v & 15) << (4 * c);
   }
   if (bits)
      *bits = packed;
   return true;
}

bool
agx_nir_lower_apple9_texture_offsets(nir_shader *nir,
                                     const struct agx_apple9_sampler_key key[32],
                                     const char **reason)
{
   agx_nir_legalize_apple9_texture_sources(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   bool progress = false;
   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
         /* Gathers without offsets are native. */
         bool is_gather = tex->op == nir_texop_tg4;
         if (offset < 0 || tex->op == nir_texop_txf)
            continue;
         /* The sample instruction encodes these offsets itself unless the
          * backend finds no offset form for the sample's final form. It then
          * lowers the offset with runtime sampler state, which the driver
          * must supply for any stage that samples with an offset, static
          * key or not. */
         if (agx_apple9_texture_native_offset(tex, NULL))
            continue;
         if ((tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
              tex->op != nir_texop_txl && tex->op != nir_texop_txd &&
              !is_gather) ||
             tex->texture_index >= 32 || tex->sampler_index >= 32 ||
             (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
              (tex->sampler_dim != GLSL_SAMPLER_DIM_3D || is_gather))) {
            *reason =
               "Apple9 software sampling requires a bound 2D, array, or 3D texture";
            return false;
         }
         /* Static wraps are specialized only for the modes the software
          * filter evaluates without border colours. Others read the runtime
          * sampler state, whose border path covers every preset and custom
          * colour. */
         const struct agx_apple9_sampler_key *state =
            key ? &key[tex->sampler_index] : NULL;
         unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 2;
         for (unsigned c = 0; state && c < dims; ++c) {
            unsigned wrap = state->wrap[c];
            if (wrap != PIPE_TEX_WRAP_REPEAT &&
                wrap != PIPE_TEX_WRAP_MIRROR_REPEAT &&
                wrap != PIPE_TEX_WRAP_CLAMP_TO_EDGE)
               state = NULL;
         }
         struct offset_sample s = {
            .tex = tex,
            .state = state,
            .dims = dims,
            .coord =
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa,
            .offset = offset >= 0 ? tex->src[offset].src.ssa : NULL};
         b.cursor = nir_before_instr(instr);
         if (s.state) {
            for (unsigned c = 0; c < s.dims; ++c)
               s.wrap[c] = nir_imm_int(&b, s.state->wrap[c]);
            s.min_lod = nir_imm_float(&b, s.state->min_lod);
            s.max_lod = nir_imm_float(&b, s.state->max_lod);
            s.bias = nir_imm_float(&b, s.state->lod_bias);
            s.border = nir_imm_ivec4(&b, 0, 0, 0, 0);
         } else {
            for (unsigned c = 0; c < s.dims; ++c)
               s.wrap[c] = agx_apple9_sampler_field(&b, tex,
                                         AGX_APPLE9_SAMP_WRAP_S + c);
            s.min = agx_apple9_sampler_field(&b, tex, AGX_APPLE9_SAMP_MIN);
            s.mag = agx_apple9_sampler_field(&b, tex, AGX_APPLE9_SAMP_MAG);
            s.mip = agx_apple9_sampler_field(&b, tex, AGX_APPLE9_SAMP_MIP);
            s.compare =
               agx_apple9_sampler_field(&b, tex, AGX_APPLE9_SAMP_COMPARE);
            s.min_lod =
               agx_apple9_sampler_field(&b, tex, AGX_APPLE9_SAMP_MIN_LOD);
            s.max_lod =
               agx_apple9_sampler_field(&b, tex, AGX_APPLE9_SAMP_MAX_LOD);
            s.bias =
               agx_apple9_sampler_field(&b, tex, AGX_APPLE9_SAMP_BIAS);
            nir_def *preset =
               agx_apple9_sampler_field(&b, tex, AGX_APPLE9_SAMP_BORDER);
            bool floating =
               tex->is_shadow ||
               nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
            nir_def *one = floating ? nir_imm_float(&b, 1) : nir_imm_int(&b, 1);
            nir_def *rgb = nir_bcsel(&b, nir_ieq_imm(&b, preset, 2), one,
                                     nir_imm_int(&b, 0));
            nir_def *alpha = nir_bcsel(&b, nir_ine_imm(&b, preset, 0), one,
                                       nir_imm_int(&b, 0));
            /* Fetches use the view's composed format/component swizzle. Apply
             * that same mapping to software border taps before selecting a
             * gather component, including constant zero/one channels.
             */
            nir_def *swizzle = agx_apple9_texture_info(&b, tex, 7);
            nir_def *rgba[4];
            for (unsigned c = 0; c < 4; ++c) {
               nir_def *channel =
                  nir_iand_imm(&b, nir_ushr_imm(&b, swizzle, 3 * c), 7);
               nir_def *value = nir_bcsel(&b, nir_ieq_imm(&b, channel, PIPE_SWIZZLE_W),
                                          alpha, rgb);
               value = nir_bcsel(&b, nir_ieq_imm(&b, channel, PIPE_SWIZZLE_1), one, value);
               rgba[c] = nir_bcsel(&b, nir_ieq_imm(&b, channel, PIPE_SWIZZLE_0),
                                    nir_imm_int(&b, 0), value);
            }
            s.border = nir_vec(&b, rgba, 4);
            /* A custom border colour (preset 3) is known only to the native
             * sampler record, in the view's encoding. Take it from the
             * hardware: an explicit-LOD sample outside the image on every
             * axis returns the swizzled border. */
            if (!tex->is_shadow) {
               nir_push_if(&b, nir_ieq_imm(&b, preset, 3));
               nir_def *custom = sample_border(&b, tex, s.coord, s.dims);
               nir_pop_if(&b, NULL);
               s.border = nir_if_phi(&b, custom, s.border);
            }
         }
         for (unsigned c = 0; c < s.dims; ++c)
            s.base_size[c] = agx_apple9_texture_info(&b, tex, c);
         if (tex->is_array) {
            s.layer = nir_f2i32(
               &b,
               nir_ffloor(
                  &b, nir_fadd_imm(&b, nir_channel(&b, s.coord, s.dims), 0.5)));
            s.layer = nir_iclamp(
               &b, s.layer, nir_imm_int(&b, 0),
               nir_iadd_imm(&b, agx_apple9_texture_info(&b, tex, 2), -1));
         }
         if (tex->is_shadow)
            s.comparator =
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_comparator)]
                  .src.ssa;
         if (is_gather) {
            nir_def_replace(&tex->def, gather(&b, &s));
            progress = true;
            continue;
         }
         nir_def *lambda = sample_lod(&b, &s);
         nir_def *last =
            nir_iadd_imm(&b, agx_apple9_texture_info(&b, tex, 3), -1);
         nir_def *level =
            nir_fclamp(&b, lambda, nir_imm_float(&b, 0), nir_i2f32(&b, last));
         nir_def *nearest_lod = nir_imax(
            &b,
            nir_iadd_imm(
               &b, nir_f2i32(&b, nir_fceil(&b, nir_fadd_imm(&b, level, 0.5))),
               -1),
            nir_imm_int(&b, 0));
         nir_def *lod =
            !s.state
               ? nir_bcsel(
                    &b, nir_ieq_imm(&b, s.mip, PIPE_TEX_MIPFILTER_NONE),
                    nir_imm_int(&b, 0),
                    nir_bcsel(
                       &b, nir_ieq_imm(&b, s.mip, PIPE_TEX_MIPFILTER_NEAREST),
                       nearest_lod, nir_f2i32(&b, nir_ffloor(&b, level))))
            : s.state->mip_filter == PIPE_TEX_MIPFILTER_NONE
               ? nir_imm_int(&b, 0)
            : s.state->mip_filter == PIPE_TEX_MIPFILTER_NEAREST
               ? nir_imax(&b,
                          nir_iadd_imm(
                             &b,
                             nir_f2i32(&b, nir_fceil(&b, nir_fadd_imm(&b, level,
                                                                      0.5))),
                             -1),
                          nir_imm_int(&b, 0))
               : nir_f2i32(&b, nir_ffloor(&b, level));
         bool floating = tex->is_shadow || nir_alu_type_get_base_type(
                                              tex->dest_type) == nir_type_float;
         /* Runtime sampler state: let the hardware filter each level when
          * an explicit integer LOD selects exactly the level(s) the
          * software path would filter. That holds unless level 0 is
          * minified with differing min/mag filters (an explicit LOD of 0
          * selects the magnification filter), or the sampler's LOD clamp
          * would move the explicit LOD off the integer level.
          */
         nir_def *native = NULL;
         nir_def *fast = NULL;
         if (!s.state) {
            nir_def *next = nir_imin(&b, nir_iadd_imm(&b, lod, 1), last);
            nir_def *linear = nir_ieq_imm(&b, s.mip, PIPE_TEX_MIPFILTER_LINEAR);
            fast = nir_iand(
               &b,
               nir_ior(&b, nir_ieq(&b, s.min, s.mag),
                       nir_ior(&b, nir_fge(&b, nir_imm_float(&b, 0), lambda),
                               nir_ige_imm(&b, lod, 1))),
               nir_iand(&b, nir_fge(&b, nir_i2f32(&b, lod), s.min_lod),
                        nir_ior(&b, nir_inot(&b, linear),
                                nir_fge(&b, s.max_lod, nir_i2f32(&b, next)))));
            nir_push_if(&b, fast);
            native = native_offset_level(&b, &s, lod);
            if (floating) {
               nir_push_if(&b, linear);
               nir_def *second = native_offset_level(&b, &s, next);
               nir_def *mixed =
                  nir_flrp(&b, native, second, nir_ffract(&b, level));
               nir_pop_if(&b, NULL);
               native = nir_if_phi(&b, mixed, native);
            }
            nir_push_else(&b, NULL);
         }
         nir_def *value = first_level(&b, &s, lod, lambda);
         if (!s.state && floating) {
            nir_push_if(&b, nir_ieq_imm(&b, s.mip, PIPE_TEX_MIPFILTER_LINEAR));
            nir_def *next = nir_imin(&b, nir_iadd_imm(&b, lod, 1), last);
            nir_def *second = filter_dynamic(&b, &s, next, s.min);
            nir_def *mixed = nir_flrp(&b, value, second, nir_ffract(&b, level));
            nir_pop_if(&b, NULL);
            value = nir_if_phi(&b, mixed, value);
         } else if (s.state &&
                    s.state->mip_filter == PIPE_TEX_MIPFILTER_LINEAR) {
            nir_def *next = nir_imin(&b, nir_iadd_imm(&b, lod, 1), last);
            nir_def *second = filter_level(
               &b, &s, next, s.state->min_filter == PIPE_TEX_FILTER_LINEAR);
            value = nir_flrp(&b, value, second, nir_ffract(&b, level));
         }
         if (fast) {
            nir_pop_if(&b, NULL);
            value = nir_if_phi(&b, native, value);
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

/* Vulkan samplers are mutable descriptor contents, available at recording time.
 * Gallium continues to specialize the same filtering implementation with keys. */
bool
agx_nir_lower_apple9_runtime_sampler_state(nir_shader *nir, const char **reason)
{
   agx_nir_legalize_apple9_texture_sources(nir);
   nir_shader_tex_pass(nir, normalize_runtime_coordinates,
                       nir_metadata_control_flow, NULL);
   return agx_nir_lower_apple9_texture_offsets(nir, NULL, reason);
}
