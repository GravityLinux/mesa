/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include "compiler/nir/nir_builder.h"
#include "agx_compile_apple9.h"
#include "agx_apple9_machine.h"
#include "util/bitset.h"
#include "util/set.h"

static bool
apple9_const_operand(nir_scalar s, nir_op op, nir_scalar *value, uint32_t *c)
{
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != op)
      return false;
   nir_scalar x = nir_scalar_chase_alu_src(s, 0);
   nir_scalar y = nir_scalar_chase_alu_src(s, 1);
   if (nir_scalar_is_const(x)) {
      *value = y;
      *c = nir_scalar_as_uint(x);
      return true;
   }
   if (nir_scalar_is_const(y)) {
      *value = x;
      *c = nir_scalar_as_uint(y);
      return true;
   }
   return false;
}

static bool
bound_offset_add_cannot_wrap(nir_scalar offset, uint32_t addend)
{
   /* These operations leave known zero bits at the bottom of a byte
    * offset, even if their input arithmetic wraps. Use only that fact:
    * amul's relaxed overflow rules are not a reason to drop a robust check.
    */
   nir_scalar value;
   uint32_t c, maximum = UINT32_MAX;
   if (nir_scalar_is_const(offset)) {
      maximum = nir_scalar_as_uint(offset);
   } else if (apple9_const_operand(offset, nir_op_amul, &value, &c) ||
              apple9_const_operand(offset, nir_op_imul, &value, &c)) {
      maximum = c ? UINT32_MAX << (ffs(c) - 1) : 0;
   } else if (apple9_const_operand(offset, nir_op_iand, &value, &c)) {
      maximum = c;
   } else if (nir_scalar_is_alu(offset) &&
              nir_scalar_alu_op(offset) == nir_op_ishl &&
              nir_scalar_is_const(nir_scalar_chase_alu_src(offset, 1))) {
      c = nir_scalar_as_uint(nir_scalar_chase_alu_src(offset, 1));
      maximum = UINT32_MAX << (c & 31);
   }
   return addend <= UINT32_MAX - maximum;
}

static bool
opt_bounds(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_ult || alu->def.num_components != 1 ||
       alu->src[0].src.ssa->bit_size != 32)
      return false;

   nir_scalar lhs = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 0);
   nir_scalar clamped, offset;
   uint32_t last, ceiling;
   if (!apple9_const_operand(lhs, nir_op_iadd, &clamped, &last) || !last ||
       !apple9_const_operand(clamped, nir_op_umin, &offset, &ceiling) ||
       ceiling != UINT32_MAX - last)
      return false;

   /* Fold an in-field constant into the size limit only after proving the
    * original 32-bit offset addition cannot wrap. In particular, x*16+12
    * permits this but x*16+16 does not for unrestricted x.
    */
   nir_scalar base;
   uint32_t addend;
   while (apple9_const_operand(offset, nir_op_iadd, &base, &addend) &&
          addend <= UINT32_MAX - last &&
          bound_offset_add_cannot_wrap(base, addend)) {
      offset = base;
      last += addend;
   }

   /* min(offset, UINT_MAX-k) + k < size is exactly equivalent to
    * offset < max(size, k) - k, including short/empty buffers and offsets
    * whose last byte would overflow. Keep offset arithmetic unchanged:
    * an earlier wrapping add is not interchangeable with a 64-bit address
    * displacement. Moving the adjustment to size lets CSE and the preamble
    * share it across accesses to a uniform descriptor.
    */
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *size = nir_ssa_for_alu_src(b, alu, 1);
   nir_def *limit = nir_iadd_imm(b, nir_umax_imm(b, size, last), -last);
   nir_def_replace(&alu->def, nir_ult(b, nir_mov_scalar(b, offset), limit));
   return true;
}

bool
agx_nir_opt_apple9_bounds(nir_shader *shader)
{
   return nir_shader_alu_pass(shader, opt_bounds, nir_metadata_control_flow, NULL);
}

static nir_def *
flush_fp32(nir_builder *b, nir_def *x)
{
   /* Integer classification must distinguish subnormals from signed zero,
    * normals, infinities and NaNs without first consuming x as a float. */
   nir_def *tiny = nir_ult_imm(b, nir_iand_imm(b, x, 0x7fffffff), 0x00800000);
   return nir_bcsel(b, tiny, nir_iand_imm(b, x, 0x80000000), x);
}

static bool
fp32_result_is_canonical(nir_scalar s, unsigned depth,
                         const BITSET_WORD *canonical_uniforms)
{
   if (s.def->bit_size != 32 || depth == 8)
      return false;

   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s)) {
      uint32_t magnitude = nir_scalar_as_uint(s) & 0x7fffffff;
      return magnitude == 0 || magnitude >= 0x00800000;
   }

   if (nir_def_instr(s.def)->type == nir_instr_type_phi) {
      nir_foreach_phi_src(src, nir_def_as_phi(s.def)) {
         if (!fp32_result_is_canonical(nir_get_scalar(src->src.ssa, s.comp),
                                       depth + 1, canonical_uniforms))
            return false;
      }
      return true;
   }

   if (canonical_uniforms && nir_def_instr_type(s.def) == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_def_as_intrinsic(s.def);
      if (intr->intrinsic == nir_intrinsic_load_preamble &&
          nir_intrinsic_preamble_class(intr) == nir_preamble_class_general) {
         unsigned word = nir_intrinsic_base(intr) + s.comp;
         return word < AGX_APPLE9_UNIFORM_COUNT &&
                BITSET_TEST(canonical_uniforms, word);
      }
   }

   if (!nir_scalar_is_alu(s))
      return false;

   nir_alu_instr *alu = nir_def_as_alu(s.def);
   if (alu->op == nir_op_bcsel)
      return fp32_result_is_canonical(nir_scalar_chase_alu_src(s, 1), depth + 1,
                                      canonical_uniforms) &&
             fp32_result_is_canonical(nir_scalar_chase_alu_src(s, 2), depth + 1,
                                      canonical_uniforms);

   /* This query runs before rewriting any instruction. Under explicit FTZ,
    * floating ALU results are canonical either natively or after this pass.
    * The initial consumer check excludes sign/min/max operations. Only after
    * selecting all checks may pruning defer one to a bit-preserving user.
    */
   return nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) ==
          nir_type_float;
}

static bool
fp32_flush_is_unobservable(nir_def *def, unsigned depth,
                           const struct set *flush)
{
   if (depth == 8)
      return false;

   nir_foreach_use(src, def) {
      if (nir_src_is_if(src))
         return false;
      nir_instr *instr = nir_src_use_instr(src);
      if (flush && instr->type == nir_instr_type_phi) {
         if (!fp32_flush_is_unobservable(&nir_instr_as_phi(instr)->def,
                                         depth + 1, flush))
            return false;
         continue;
      }
      if (instr->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      /* All forms of these native FP32 operations consume subnormals as
       * signed zero. Min/max and sign operations do not: their selected or
       * modified operand bits can escape through a store or integer use. */
      case nir_op_fadd:
      case nir_op_fsub:
      case nir_op_fmul:
      case nir_op_ffma:
      case nir_op_ffma_weak:
      case nir_op_flt:
      case nir_op_fge:
      case nir_op_feq:
      case nir_op_fneu:
         break;
      case nir_op_bcsel:
         if (!flush || src == &alu->src[0].src ||
             !fp32_flush_is_unobservable(&alu->def, depth + 1, flush))
            return false;
         break;
      case nir_op_fneg:
      case nir_op_fabs:
      case nir_op_fmin:
      case nir_op_fmax:
      case nir_op_fsat:
      case nir_op_fcanonicalize:
         /* These commute with signed-zero canonicalization: sign changes
          * preserve magnitude, and native min/max comparisons use DAZ.
          * Delay an upstream check only to an independently retained check
          * or to a proven DAZ use of the entire chain. Do not use this rule
          * while constructing the initial canonical-producer proof. */
         if (!flush ||
             (!_mesa_set_search(flush, alu) &&
              !fp32_flush_is_unobservable(&alu->def, depth + 1, flush)))
            return false;
         break;
      default:
         if (!nir_op_is_vec_or_mov(alu->op) ||
             !fp32_flush_is_unobservable(&alu->def, depth + 1, flush))
            return false;
         break;
      }
   }
   return true;
}

static bool
fp32_minmax_has_normal_bound(nir_scalar s, bool lower,
                            const uint32_t *uniform_constants)
{
   s = nir_scalar_chase_movs(s);
   uint32_t value;
   if (nir_scalar_is_const(s)) {
      value = nir_scalar_as_uint(s);
   } else if (nir_def_instr_type(s.def) == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_def_as_intrinsic(s.def);
      if (intr->intrinsic != nir_intrinsic_load_preamble ||
          nir_intrinsic_preamble_class(intr) != nir_preamble_class_general)
         return false;
      unsigned word = nir_intrinsic_base(intr) + s.comp;
      if (word >= AGX_APPLE9_UNIFORM_COUNT)
         return false;
      value = uniform_constants[word];
   } else {
      return false;
   }

   /* max(x, positive-normal) and min(x, negative-normal) cannot select a
    * subnormal. Infinities are valid bounds; NaNs are not. Unknown uniform
    * constants are represented by zero, which never establishes a bound. */
   uint32_t magnitude = value & 0x7fffffff;
   return (value >> 31) == !lower && magnitude >= 0x00800000 &&
          magnitude <= 0x7f800000;
}

static bool
needs_fp32_flush(nir_alu_instr *alu, const BITSET_WORD *canonical_uniforms,
                 const uint32_t *uniform_constants)
{
   if (alu->def.bit_size != 32 ||
       nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) !=
          nir_type_float)
      return false;

   switch (alu->op) {
   /* T8132 exact-bit probes separate input DAZ from output FTZ for these
    * native operations, including compact, immediate and source-modifier
    * forms. FMA retains an underflowing product until the final rounding. */
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_ffma_weak:
      return false;
   /* Normal inputs producing subnormal exp2 results also flush on T8132. */
   case nir_op_fexp2:
      return false;
   /* These results are integral, or have magnitudes bounded away from the
    * FP32 subnormal interval by their input format. */
   case nir_op_b2f32:
   case nir_op_i2f32:
   case nir_op_u2f32:
   case nir_op_ffloor:
   case nir_op_fceil:
   case nir_op_ftrunc:
   case nir_op_fround_even:
   case nir_op_fsqrt:
   case nir_op_frsq:
   case nir_op_flog2:
      return false;
   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_fsat:
   case nir_op_fcanonicalize:
   case nir_op_fmin:
   case nir_op_fmax: {
      /* These operations cannot create a subnormal from canonical operands.
       * Prove every component, including swizzles, before eliding the check. */
      bool canonical = true;
      for (unsigned c = 0; c < alu->def.num_components; ++c) {
         nir_scalar result = nir_get_scalar(&alu->def, c);
         if ((alu->op == nir_op_fmin || alu->op == nir_op_fmax) &&
             (fp32_minmax_has_normal_bound(nir_scalar_chase_alu_src(result, 0),
                                           alu->op == nir_op_fmax, uniform_constants) ||
              fp32_minmax_has_normal_bound(nir_scalar_chase_alu_src(result, 1),
                                           alu->op == nir_op_fmax, uniform_constants)))
            continue;
         for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
            canonical &= fp32_result_is_canonical(
               nir_scalar_chase_alu_src(result, i), 0, canonical_uniforms);
      }
      if (canonical)
         return false;
      break;
   }
   case nir_op_f2f32:
      if (alu->src[0].src.ssa->bit_size == 16)
         return false;
      break;
   default:
      break;
   }

   return !fp32_flush_is_unobservable(&alu->def, 0, NULL);
}

static bool
lower_denorm(nir_builder *b, nir_alu_instr *alu, void *data)
{
   bool flush = _mesa_set_search(data, alu) != NULL;
   if (alu->op == nir_op_fcanonicalize && alu->def.bit_size == 32) {
      b->cursor = nir_before_instr(&alu->instr);
      nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
      nir_def_replace(&alu->def, flush ? flush_fp32(b, x) : x);
      return true;
   }
   if (!flush)
      return false;

   /* Run after algebraic optimization, which can replace arithmetic with
    * min/max or sign operations. Moves, selects, loads and stores are bit
    * preserving and deliberately do not enter this float-result path. */
   b->cursor = nir_after_instr(&alu->instr);
   nir_def *result = flush_fp32(b, &alu->def);
   nir_def_rewrite_uses_after(&alu->def, result);
   return true;
}

static bool
lower_denorm_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                       UNUSED void *data)
{
   bool affected;
   switch (intr->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
      affected = true;
      break;
   default:
      affected = nir_intrinsic_has_atomic_op(intr) &&
                 (nir_intrinsic_atomic_op(intr) == nir_atomic_op_fadd ||
                  nir_intrinsic_atomic_op(intr) == nir_atomic_op_fmin ||
                  nir_intrinsic_atomic_op(intr) == nir_atomic_op_fmax);
      break;
   }
   if (!affected || intr->def.bit_size != 32)
      return false;

   /* Floating atomic arithmetic must canonicalize its returned value too.
    * Integer atomics and bit-preserving exchange/load remain untouched. */
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *result = flush_fp32(b, &intr->def);
   nir_def_rewrite_uses_after(&intr->def, result);
   return true;
}

static bool
lower_denorm_reduction(nir_builder *b, nir_intrinsic_instr *intr,
                       UNUSED void *data)
{
   if ((intr->intrinsic != nir_intrinsic_reduce &&
        intr->intrinsic != nir_intrinsic_inclusive_scan &&
        intr->intrinsic != nir_intrinsic_exclusive_scan) ||
       intr->def.bit_size != 32 ||
       (nir_intrinsic_reduction_op(intr) != nir_op_fmin &&
        nir_intrinsic_reduction_op(intr) != nir_op_fmax))
      return false;

   /* A scan's first active lane, or a one-element cluster, may bypass every
    * ALU operation. Keep the original reduction's result semantics before
    * generic subgroup lowering turns it into bit-preserving shuffles/selects. */
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *result = flush_fp32(b, &intr->def);
   nir_def_rewrite_uses_after(&intr->def, result);
   return true;
}

bool
agx_nir_lower_apple9_subgroup_denorms(nir_shader *shader)
{
   return nir_is_denorm_flush_to_zero(
             shader->info.float_controls_execution_mode, 32) &&
          nir_shader_intrinsics_pass(shader, lower_denorm_reduction,
                                     nir_metadata_control_flow, NULL);
}

bool
agx_nir_lower_apple9_denorms(nir_shader *shader)
{
   /* Decide on the original SSA graph, before explicit integer flushes hide
    * the provenance of floating results. Keep this independent of fast-math
    * permissions: removing an unobservable flush preserves even exact math. */
   struct set *flush = _mesa_pointer_set_create(NULL);
   if (nir_is_denorm_flush_to_zero(shader->info.float_controls_execution_mode,
                                   32)) {
      BITSET_DECLARE(canonical_uniforms, AGX_APPLE9_UNIFORM_COUNT) = {0};
      uint32_t uniform_constants[AGX_APPLE9_UNIFORM_COUNT] = {0};
      nir_function *preamble = nir_shader_get_entrypoint(shader)->preamble;
      if (preamble) {
         /* Extraction replaces SSA edges by uniform transfers. Retain the
          * canonical-result proof across those transfers, including promoted
          * constants. A store is a raw-bit use, so its floating producer may
          * not defer a required flush to a consumer in the main shader. */
         nir_foreach_block(block, preamble->impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_preamble ||
                   nir_intrinsic_preamble_class(intr) != nir_preamble_class_general ||
                   intr->src[0].ssa->bit_size != 32)
                  continue;
               unsigned base = nir_intrinsic_base(intr);
               for (unsigned c = 0; c < intr->src[0].ssa->num_components; ++c) {
                  nir_scalar value = nir_scalar_chase_movs(
                     nir_get_scalar(intr->src[0].ssa, c));
                  if (base + c < AGX_APPLE9_UNIFORM_COUNT && nir_scalar_is_const(value))
                     uniform_constants[base + c] = nir_scalar_as_uint(value);
                  if (base + c < AGX_APPLE9_UNIFORM_COUNT &&
                      fp32_result_is_canonical(nir_get_scalar(intr->src[0].ssa, c),
                                                0, NULL))
                     BITSET_SET(canonical_uniforms, base + c);
               }
            }
         }
      }
      nir_foreach_function_impl(impl, shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu &&
                   needs_fp32_flush(nir_instr_as_alu(instr), canonical_uniforms,
                                     uniform_constants))
                  _mesa_set_add(flush, nir_instr_as_alu(instr));
            }
         }
      }
      /* Prune only after the original graph's checks have been selected.
       * A downstream check can absorb an upstream check through operations
       * commuting with canonicalization. Remove entries immediately so no
       * later proof can rely on a check already discarded. */
      nir_foreach_function_impl(impl, shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               struct set_entry *entry = _mesa_set_search(flush, alu);
               if (entry && fp32_flush_is_unobservable(&alu->def, 0, flush))
                  _mesa_set_remove(flush, entry);
            }
         }
      }
   }
   bool progress = nir_shader_alu_pass(shader, lower_denorm,
                                       nir_metadata_control_flow, flush);
   _mesa_set_destroy(flush, NULL);
   if (nir_is_denorm_flush_to_zero(shader->info.float_controls_execution_mode,
                                   32))
      progress |= nir_shader_intrinsics_pass(shader, lower_denorm_intrinsic,
                                             nir_metadata_control_flow, NULL);
   return progress;
}

/* Bounded-accuracy floating-point reduction, like the existing AGX backend.
 * Vulkan requires 2^-11 absolute error only inside [-pi, pi], in every stage,
 * so phase accuracy may degrade as |x| grows. The Apple9 factor consumes
 * quarter-turn units, so fold four quadrants into [-1, 1] before evaluating
 * sin(pi*p/2)/p. Work on the magnitude to avoid cancellation for small
 * negative sine inputs. */
static nir_def *
lower_sincos(nir_builder *b, nir_def *x, bool cosine)
{
   nir_def *turns = nir_fmul_imm(b, nir_fabs(b, x), 0x1.45f306p-3f);
   nir_def *q = nir_fmul_imm(b, nir_fsub(b, turns, nir_ffloor(b, turns)), 4.0f);
   if (cosine) {
      q = nir_fadd_imm(b, q, 1.0f);
      q = nir_bcsel(b, nir_fge_imm(b, q, 4.0f), nir_fadd_imm(b, q, -4.0f), q);
   }
   nir_def *p = nir_bcsel(b, nir_flt_imm(b, q, 1.0f), q,
      nir_bcsel(b, nir_flt_imm(b, q, 3.0f),
                 nir_fsub(b, nir_imm_float(b, 2.0f), q), nir_fadd_imm(b, q, -4.0f)));
   nir_def *result = nir_fmul(b, p, nir_fsin_factor_agx(b, p));
   return cosine ? result : nir_ixor(b, result, nir_iand_imm(b, x, 0x80000000));
}

static bool
math_filter(const nir_instr *instr, UNUSED const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op == nir_op_f2u8 || alu->op == nir_op_f2u16 ||
       alu->op == nir_op_f2i8 || alu->op == nir_op_f2i16)
      return true;
   if (alu->op == nir_op_b2f16 || alu->op == nir_op_f2f16_rtz ||
       (alu->op == nir_op_ldexp && alu->def.bit_size == 16))
      return true;
   if ((alu->op == nir_op_fsin || alu->op == nir_op_fcos) &&
       alu->def.bit_size == 16)
      return true;
   return alu->def.bit_size == 32 &&
          (alu->op == nir_op_iabs || alu->op == nir_op_fsin || alu->op == nir_op_fcos);
}

/* Round to nearest, then step one ulp toward zero if that rounded away. */
static nir_def *
lower_f2f16_rtz(nir_builder *b, nir_def *x)
{
   /* Keep algebraic optimization from folding the round trip. */
   uint32_t ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_no_fast_math;
   x = nir_f2f32(b, x);
   nir_def *h = nir_f2f16(b, x);
   nir_def *away = nir_flt(b, nir_fabs(b, x), nir_fabs(b, nir_f2f32(b, h)));
   nir_def *bits = nir_u2u16(b, nir_iadd_imm(b, nir_u2u32(b, h), -1));
   nir_def *result = nir_bcsel(b, away, bits, h);
   b->fp_math_ctrl = ctrl;
   return result;
}

static nir_def *
lower_math(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *source = nir_mov_alu(b, alu->src[0], alu->def.num_components);
   nir_def *components[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < alu->def.num_components; ++i) {
      nir_def *x = nir_channel(b, source, i);
      if (alu->op == nir_op_f2u8 || alu->op == nir_op_f2u16) {
         /* Narrow integer results come from the FP32 conversion. */
         components[i] = nir_u2uN(b, nir_f2u32(b, nir_f2fN(b, x, 32)),
                                  alu->def.bit_size);
      } else if (alu->op == nir_op_f2i8 || alu->op == nir_op_f2i16) {
         components[i] = nir_i2iN(b, nir_f2i32(b, nir_f2fN(b, x, 32)),
                                  alu->def.bit_size);
      } else if (alu->op == nir_op_b2f16) {
         components[i] = nir_bcsel(b, x, nir_imm_float16(b, 1.0f),
                                   nir_imm_float16(b, 0.0f));
      } else if (alu->op == nir_op_ldexp) {
         /* The generic binary16 expansion builds narrow exponent fields. */
         nir_def *exp = nir_channel(b, nir_mov_alu(b, alu->src[1],
                                                   alu->def.num_components), i);
         components[i] = nir_f2f16(b, nir_ldexp(b, nir_f2f32(b, x), exp));
      } else if (alu->op == nir_op_f2f16_rtz) {
         components[i] = lower_f2f16_rtz(b, x);
      } else if (alu->def.bit_size == 16) {
         /* Binary16 sine and cosine use the FP32 reduction. */
         nir_def *wide = nir_f2f32(b, x);
         wide = lower_sincos(b, wide, alu->op == nir_op_fcos);
         components[i] = nir_f2f16(b, wide);
      } else if (alu->op == nir_op_iabs) {
         components[i] = nir_imax(b, x, nir_ineg(b, x));
      } else {
         components[i] = lower_sincos(b, x, alu->op == nir_op_fcos);
      }
   }
   return nir_vec(b, components, alu->def.num_components);
}

/* T8132 F2I32 and F2U32 saturate to the destination range, convert NaN to
 * zero, and F2U32 converts negative values to zero (exact-bit probes with
 * +-2^31, +-2^32, +-1e30, +-inf and NaNs). Range checks that rebuild these
 * results around a conversion, such as DXVK's saturating ftoi/ftou, are
 * therefore redundant. NIR leaves out-of-range conversions undefined, so
 * fold them only here, at the backend boundary, and only for non-constant
 * inputs that constant folding cannot evaluate differently. */
static bool
float_const(nir_scalar s, float *value)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_const(s) || s.def->bit_size != 32)
      return false;
   *value = uif(nir_scalar_as_uint(s));
   return true;
}

/* cond selects k exactly when the saturating conversion of x yields k. */
static bool
conversion_select_redundant(nir_scalar cond, nir_scalar x, uint32_t k, bool sign)
{
   if (!nir_scalar_is_alu(cond))
      return false;
   nir_op op = nir_scalar_alu_op(cond);
   /* Only these per-component comparisons qualify; other conditions (such
    * as vector comparisons with sized sources) cannot be chased per scalar. */
   if (op != nir_op_fneu && op != nir_op_flt && op != nir_op_fge)
      return false;
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
   float c;
   if (op == nir_op_fneu && nir_scalar_equal(a, x) && nir_scalar_equal(b, x))
      return k == 0;
   /* x > c, with every float above c at or past the saturation bound. */
   if ((op == nir_op_flt && nir_scalar_equal(b, x) && float_const(a, &c)) ||
       (op == nir_op_fge && nir_scalar_equal(a, x) && float_const(b, &c) &&
        (c = nextafterf(c, -INFINITY), true))) {
      return sign ? (k == 0x7fffffff && c >= 2147483520.0f)
                  : (k == 0xffffffff && c >= 4294967040.0f);
   }
   /* x < c, with every float below c at or past the lower bound. */
   if ((op == nir_op_flt && nir_scalar_equal(a, x) && float_const(b, &c)) ||
       (op == nir_op_fge && nir_scalar_equal(b, x) && float_const(a, &c) &&
        (c = nextafterf(c, INFINITY), true))) {
      return sign ? (k == 0x80000000 && c <= -2147483648.0f)
                  : (k == 0 && c <= 0.0f);
   }
   return false;
}

/* The conversion at the core of a chain of redundant range selects. */
static bool
saturating_conversion(nir_scalar s, nir_scalar *x, bool *sign)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_f2i32 || op == nir_op_f2u32) {
      if (nir_scalar_chase_alu_src(s, 0).def->bit_size != 32)
         return false;
      *x = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      *sign = op == nir_op_f2i32;
      return !nir_scalar_is_const(*x);
   }
   if (op != nir_op_bcsel)
      return false;
   uint32_t k;
   nir_scalar sel = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   if (!nir_scalar_is_const(sel))
      return false;
   k = nir_scalar_as_uint(sel);
   return saturating_conversion(nir_scalar_chase_alu_src(s, 2), x, sign) &&
          conversion_select_redundant(nir_scalar_chase_alu_src(s, 0), *x, k, *sign);
}

static bool
fold_saturating_conversion(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->def.bit_size != 32 || alu->def.num_components != 1)
      return false;
   if (alu->op == nir_op_f2u32) {
      /* F2U32 already maps negative values and NaN to zero. */
      nir_scalar src = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 0);
      if (!nir_scalar_is_alu(src) || nir_scalar_alu_op(src) != nir_op_fmax ||
          src.def->bit_size != 32)
         return false;
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar zero = nir_scalar_chase_movs(nir_scalar_chase_alu_src(src, i));
         nir_scalar other = nir_scalar_chase_alu_src(src, 1 - i);
         if (nir_scalar_is_const(zero) && (nir_scalar_as_uint(zero) & 0x7fffffff) == 0 &&
             !nir_scalar_is_const(nir_scalar_chase_movs(other))) {
            b->cursor = nir_before_instr(&alu->instr);
            nir_src_rewrite(&alu->src[0].src, nir_mov_scalar(b, other));
            alu->src[0].swizzle[0] = 0;
            return true;
         }
      }
      return false;
   }
   if (alu->op != nir_op_bcsel)
      return false;
   nir_scalar x;
   bool sign;
   nir_scalar self = nir_get_scalar(&alu->def, 0);
   if (!saturating_conversion(self, &x, &sign))
      return false;
   nir_scalar inner = nir_scalar_chase_alu_src(self, 2);
   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, nir_mov_scalar(b, inner));
   return true;
}

bool
agx_nir_lower_apple9_math(nir_shader *shader)
{
   bool progress = agx_nir_lower_apple9_subgroup_denorms(shader);
   progress |= nir_shader_alu_pass(shader, fold_saturating_conversion,
                                   nir_metadata_control_flow, NULL);
   progress |= nir_shader_lower_instructions(shader, math_filter, lower_math, NULL);
   /* Apple9 has ordinary multiply/add/FMA, but no modeled interpolation
    * instruction. Use NIR's precision-aware lowering for FP32 mix. */
   progress |= nir_lower_flrp(shader, 16 | 32, false);
   if (progress) {
      nir_opt_constant_folding(shader);
      nir_opt_copy_prop(shader);
      nir_opt_dce(shader);
   }
   return progress;
}

/* Compute linear groups are consecutive sets of four local invocations.
 * Use the same subgroup lane transport as quad operations, with explicit
 * subtraction so no fragment-only derivative instruction is required.
 */
static bool
lower_compute_derivative(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   bool y, coarse;
   switch (intr->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddx_coarse:
      y = false;
      break;
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddy_coarse:
      y = true;
      break;
   default:
      return false;
   }
   coarse = intr->intrinsic == nir_intrinsic_ddx_coarse ||
            intr->intrinsic == nir_intrinsic_ddy_coarse;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *lane = nir_load_subgroup_invocation(b);
   unsigned step = y ? 2 : 1;
   nir_def *lo = nir_iand_imm(b, lane, ~(coarse ? 3u : step));
   nir_def *hi = nir_ior_imm(b, lo, step);
   nir_def *value = intr->src[0].ssa;
   nir_def *a = nir_shuffle(b, value, lo);
   nir_def *z = nir_shuffle(b, value, hi);
   nir_def_replace(&intr->def, nir_fsub(b, z, a));
   return true;
}

bool
agx_nir_lower_apple9_compute_derivatives(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.derivative_group != DERIVATIVE_GROUP_LINEAR)
      return false;

   return nir_shader_intrinsics_pass(shader, lower_compute_derivative,
                                      nir_metadata_control_flow, NULL);
}
