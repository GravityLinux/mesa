/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_compile_apple9.h"

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
   return nir_bcsel(b, nir_uge_imm(b, ax, 0x7f800000),
                    nir_imm_int(b, 0x7fc00000), result);
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
   return alu->def.bit_size == 32 &&
          (alu->op == nir_op_iabs || alu->op == nir_op_fsin || alu->op == nir_op_fcos ||
           alu->op == nir_op_fcanonicalize);
}

static nir_def *
lower_math(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *source = nir_mov_alu(b, alu->src[0], alu->def.num_components);
   nir_def *components[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < alu->def.num_components; ++i) {
      nir_def *x = nir_channel(b, source, i);
      if (alu->op == nir_op_iabs) {
         components[i] = nir_imax(b, x, nir_ineg(b, x));
      } else if (alu->op == nir_op_fcanonicalize) {
         components[i] = nir_fmul_imm(b, x, 1.0f);
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
   bool progress =
      nir_shader_lower_instructions(shader, math_filter, lower_math, NULL);
   /* Apple9 has ordinary multiply/add/FMA, but no modeled interpolation
    * instruction. Use NIR's precision-aware lowering for FP32 mix. */
   progress |= nir_lower_flrp(shader, 32, false);
   if (progress) {
      nir_opt_constant_folding(shader);
      nir_opt_copy_prop(shader);
      nir_opt_dce(shader);
   }
   return progress;
}
