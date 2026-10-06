/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_compile_apple9.h"
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
fp32_result_is_canonical(nir_scalar s, unsigned depth)
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
                                       depth + 1))
            return false;
      }
      return true;
   }

   if (!nir_scalar_is_alu(s))
      return false;

   nir_alu_instr *alu = nir_def_as_alu(s.def);
   if (alu->op == nir_op_bcsel)
      return fp32_result_is_canonical(nir_scalar_chase_alu_src(s, 1), depth + 1) &&
             fp32_result_is_canonical(nir_scalar_chase_alu_src(s, 2), depth + 1);

   /* This query runs before rewriting any instruction. Under explicit FTZ,
    * floating ALU results are canonical either natively or after this pass.
    * The consumer check below deliberately excludes sign/min/max operations:
    * a producer cannot defer its flush to one of these bit-preserving users.
    */
   return nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) ==
          nir_type_float;
}

static bool
only_used_by_fp32_daz(nir_def *def, unsigned depth)
{
   if (depth == 8)
      return false;

   nir_foreach_use(src, def) {
      if (nir_src_is_if(src))
         return false;
      nir_instr *instr = nir_src_use_instr(src);
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
      default:
         if (!nir_op_is_vec_or_mov(alu->op) ||
             !only_used_by_fp32_daz(&alu->def, depth + 1))
            return false;
         break;
      }
   }
   return true;
}

static bool
needs_fp32_flush(nir_alu_instr *alu)
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
         for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
            canonical &= fp32_result_is_canonical(
               nir_scalar_chase_alu_src(result, i), 0);
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

   return !only_used_by_fp32_daz(&alu->def, 0);
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
      nir_foreach_function_impl(impl, shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu &&
                   needs_fp32_flush(nir_instr_as_alu(instr)))
                  _mesa_set_add(flush, nir_instr_as_alu(instr));
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

static nir_def *
select_word(nir_builder *b, nir_def **words, nir_def *index, unsigned offset)
{
   /* The reduced exponent makes index range from two through six. */
   nir_def *value = words[2 + offset];
   for (unsigned k = 3; k <= 6; ++k)
      value = nir_bcsel(b, nir_ieq_imm(b, index, k), words[k + offset], value);
   return value;
}

static nir_def *
extract_word(nir_builder *b, nir_def *lo, nir_def *hi, nir_def *shift)
{
   nir_def *joined =
      nir_ior(b, nir_ushr(b, lo, shift), nir_ishl(b, hi, nir_ineg(b, shift)));
   /* NIR masks a 32-bit shift count: shifting by 32 means shifting by zero. */
   return nir_bcsel(b, nir_ieq_imm(b, shift, 0), lo, joined);
}

static nir_def *
lower_sincos(nir_builder *b, nir_def *x, bool cosine)
{
   /* Independently constructed fixed-point argument reduction:
    *   abs(x) = M * 2^(E-23), C = floor((2/pi) * 2^256).
    * The nine-word integer product M*C supplies the quadrant and 64
    * fractional bits of x*2/pi, even at FLT_MAX. This avoids a floating
    * quotient losing its low quadrant bits for large arguments. Constants
    * below are the mathematical 2/pi expansion, least significant word first.
    * No proprietary polynomial or range-reduction sequence is used. */
   static const uint32_t two_over_pi[] = {
      0xdebbc561, 0xfe5163ab, 0x3c439041, 0xdb629599,
      0xf534ddc0, 0xfc2757d1, 0x4e441529, 0xa2f9836e,
   };
   nir_def *ax = nir_iand_imm(b, x, 0x7fffffff);
   nir_def *mantissa = nir_ior_imm(b, nir_iand_imm(b, ax, 0x7fffff), 0x800000);
   nir_def *carry = nir_imm_int(b, 0);
   nir_def *product[10];
   for (unsigned i = 0; i < 8; ++i) {
      nir_def *lo = nir_imul_imm(b, mantissa, two_over_pi[i]);
      nir_def *hi = nir_umul_high(b, mantissa, nir_imm_int(b, two_over_pi[i]));
      product[i] = nir_iadd(b, lo, carry);
      carry = nir_iadd(b, hi, nir_b2i32(b, nir_ult(b, product[i], lo)));
   }
   product[8] = carry;
   product[9] = nir_imm_int(b, 0);

   nir_def *exponent = nir_ushr_imm(b, ax, 23);
   exponent = nir_umin(b, nir_umax(b, exponent, nir_imm_int(b, 126)),
                       nir_imm_int(b, 254));
   /* The low end of the 64-bit fraction is product bit 342-biased_E. */
   nir_def *position = nir_isub(b, nir_imm_int(b, 342), exponent);
   nir_def *index = nir_ushr_imm(b, position, 5);
   nir_def *shift = nir_iand_imm(b, position, 31);
   nir_def *word[4];
   for (unsigned i = 0; i < 4; ++i)
      word[i] = select_word(b, product, index, i);
   nir_def *low = extract_word(b, word[0], word[1], shift);
   nir_def *high = extract_word(b, word[1], word[2], shift);
   nir_def *quadrant = extract_word(b, word[2], word[3], shift);
   nir_def *negative = nir_ine_imm(b, nir_iand_imm(b, high, 0x80000000), 0);
   quadrant = nir_iadd(b, quadrant, nir_b2i32(b, negative));

   /* Center the fraction on the nearest multiple of pi/2. Negate in integer
    * space before conversion, avoiding catastrophic cancellation near one. */
   nir_def *negative_high =
      nir_iadd(b, nir_inot(b, high), nir_b2i32(b, nir_ieq_imm(b, low, 0)));
   high = nir_bcsel(b, negative, negative_high, high);
   low = nir_bcsel(b, negative, nir_ineg(b, low), low);
   nir_def *fhi =
      nir_fmul_imm(b, nir_u2f32(b, nir_ushr_imm(b, high, 16)), 0x1p-16);
   nir_def *flo = nir_ffma(
      b, nir_u2f32(b, low), nir_imm_float(b, 0x1p-64),
      nir_fmul_imm(b, nir_u2f32(b, nir_iand_imm(b, high, 0xffff)), 0x1p-32));
   /* Keep the reduced phase in units of pi/2. For small arguments, a
    * two-part multiplication by 2/pi supplies the same representation. */
   const double inv_pi_hi = 0x1.45f306p-1, inv_pi_lo = 0x1.b93910p-26;
   nir_def *p0 = nir_fmul_imm(b, ax, inv_pi_hi);
   nir_def *error =
      nir_ffma(b, ax, nir_imm_float(b, inv_pi_hi), nir_fneg(b, p0));
   error = nir_ffma(b, ax, nir_imm_float(b, inv_pi_lo), error);
   nir_def *small = nir_ult_imm(b, ax, 0x3f490fdb);
   fhi = nir_bcsel(b, small, p0, fhi);
   flo = nir_bcsel(b, small, error, flo);
   negative = nir_iand(b, negative, nir_inot(b, small));
   quadrant = nir_bcsel(b, small, nir_imm_int(b, 0), quadrant);
   if (cosine)
      quadrant = nir_iadd_imm(b, quadrant, 1);

   /* EXP-M4-55: the SFU evaluates sin(pi*p/2)/p on [-1,1], with the
    * continuous value pi/2 at zero. The centered phase magnitude is at
    * most 1/2, so both it and its complement fit the hardware domain.
    * Correct for phase rounding and the subtraction in the complement;
    * first-order corrections suffice for these sub-ULP residuals. */
   nir_def *phase = nir_fadd(b, fhi, flo);
   nir_def *tail = nir_fadd(b, nir_fsub(b, fhi, phase), flo);
   nir_def *complement = nir_fsub(b, nir_imm_float(b, 1), phase);
   nir_def *complement_error =
      nir_fsub(b, nir_fsub(b, nir_imm_float(b, 1), complement), phase);
   nir_def *sine = nir_fmul(b, phase, nir_fsin_factor_agx(b, phase));
   nir_def *cos = nir_fmul(b, complement, nir_fsin_factor_agx(b, complement));
   nir_def *s = nir_ffma(b, nir_fmul_imm(b, tail, 0x1.921fb6p+0), cos, sine);
   nir_def *c = nir_ffma(
      b, nir_fmul_imm(b, nir_fsub(b, complement_error, tail), 0x1.921fb6p+0),
      sine, cos);
   s = nir_bcsel(b, negative, nir_fneg(b, s), s);
   nir_def *result =
      nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, quadrant, 1), 0), c, s);
   nir_def *sign = nir_ishl_imm(b, nir_iand_imm(b, quadrant, 2), 30);
   if (!cosine)
      sign = nir_ixor(b, sign, nir_iand_imm(b, x, 0x80000000));
   result = nir_ixor(b, result, sign);
   /* Preserve sin's signed zero and tiny inputs without a flushing ALU use. */
   result = nir_bcsel(b, nir_ult_imm(b, ax, 0x39800000),
                      cosine ? nir_imm_float(b, 1) : x, result);
   result = nir_bcsel(b, nir_uge_imm(b, ax, 0x7f800000),
                      nir_imm_int(b, 0x7fc00000), result);
   /* The small-input path above returns raw bits, so retain the original
    * fsin operation's denormal-result semantics before it disappears. */
   return nir_is_denorm_flush_to_zero(
             b->shader->info.float_controls_execution_mode, 32)
             ? flush_fp32(b, result) : result;
}

/* Graphics uses bounded-accuracy floating-point reduction, like the existing
 * AGX graphics backend. The Apple9 factor consumes quarter-turn units, so
 * fold four quadrants into [-1, 1] before evaluating sin(pi*p/2)/p.
 * Work on the magnitude to avoid cancellation for small negative sine inputs.
 * Unlike the compute reducer, phase accuracy degrades as |x| grows. */
static nir_def *
lower_graphics_sincos(nir_builder *b, nir_def *x, bool cosine)
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
         wide = b->shader->info.stage == MESA_SHADER_VERTEX ||
                      b->shader->info.stage == MESA_SHADER_FRAGMENT
                   ? lower_graphics_sincos(b, wide, alu->op == nir_op_fcos)
                   : lower_sincos(b, wide, alu->op == nir_op_fcos);
         components[i] = nir_f2f16(b, wide);
      } else if (alu->op == nir_op_iabs) {
         components[i] = nir_imax(b, x, nir_ineg(b, x));
      } else if (b->shader->info.stage == MESA_SHADER_VERTEX ||
                 b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         components[i] = lower_graphics_sincos(b, x, alu->op == nir_op_fcos);
      } else {
         components[i] = lower_sincos(b, x, alu->op == nir_op_fcos);
      }
   }
   return nir_vec(b, components, alu->def.num_components);
}

bool
agx_nir_lower_apple9_math(nir_shader *shader)
{
   bool progress = agx_nir_lower_apple9_subgroup_denorms(shader);
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
