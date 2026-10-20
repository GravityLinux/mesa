/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * FP32 DenormPreserve for Apple9.
 *
 * Apple9 FP32 arithmetic treats subnormal inputs as signed zero and flushes
 * subnormal results; bit operations (moves, negate, absolute value, selects,
 * loads and stores) keep subnormal bits. For shaders that declare
 * DenormPreserve for 32-bit floats, each affected operation keeps its native
 * instruction and takes a software path only when an operand is subnormal or
 * the native result may have been flushed.
 *
 * The software path never feeds a subnormal to the hardware. It splits each
 * operand into a sign-carrying mantissa in [1, 2) and an integer exponent,
 * computes on mantissas natively (all normal), and rebuilds the result with
 * integer operations, rounding subnormal results to nearest-even using the
 * exact error of the native computation as a sticky term.
 *
 * Shaders without the execution mode are not modified.
 */

#include "compiler/nir/nir_builder.h"
#include "agx_compile_apple9.h"

#define F32_SIGN     0x80000000u
#define F32_EXP_MASK 0x7f800000u
#define F32_MAN_MASK 0x007fffffu
#define F32_ONE      0x3f800000u

static nir_def *
flip_sign(nir_builder *b, nir_def *x)
{
   return nir_ixor(b, x, nir_imm_int(b, F32_SIGN));
}

static nir_def *
exponent_field(nir_builder *b, nir_def *x)
{
   return nir_iand_imm(b, x, F32_EXP_MASK);
}

/* Nonzero subnormal. */
static nir_def *
is_subnormal(nir_builder *b, nir_def *x)
{
   return nir_iand(b, nir_ieq_imm(b, exponent_field(b, x), 0),
                  nir_ine_imm(b, nir_iand_imm(b, x, F32_MAN_MASK), 0));
}

/* Exponent field all zero: signed zero or subnormal. */
static nir_def *
is_zero_or_subnormal(nir_builder *b, nir_def *x)
{
   return nir_ieq_imm(b, exponent_field(b, x), 0);
}

static nir_def *
is_zero(nir_builder *b, nir_def *x)
{
   return nir_ieq_imm(b, nir_iand_imm(b, x, ~F32_SIGN), 0);
}

static nir_def *
is_inf_or_nan(nir_builder *b, nir_def *x)
{
   return nir_ieq_imm(b, exponent_field(b, x), F32_EXP_MASK);
}

/*
 * Split a finite nonzero x into a mantissa with x's sign and magnitude in
 * [1, 2), and an unbiased exponent, so that x = m * 2^e exactly. For zero the
 * result is (x, 0); callers handle zeros separately where it matters.
 */
static void
unpack(nir_builder *b, nir_def *x, nir_def **m, nir_def **e)
{
   nir_def *field = nir_ushr_imm(b, exponent_field(b, x), 23);
   nir_def *man = nir_iand_imm(b, x, F32_MAN_MASK);
   nir_def *sign = nir_iand_imm(b, x, F32_SIGN);

   /* Subnormal: shift the leading one into the implicit position. */
   nir_def *msb = nir_ufind_msb(b, man);
   nir_def *shift = nir_isub(b, nir_imm_int(b, 23), msb);
   nir_def *norm_man = nir_iand_imm(b, nir_ishl(b, man, shift), F32_MAN_MASK);
   nir_def *sub_e = nir_isub(b, nir_imm_int(b, -126), shift);

   nir_def *normal = nir_ine_imm(b, field, 0);
   nir_def *mant = nir_bcsel(b, normal, man, norm_man);
   *e = nir_bcsel(b, normal, nir_iadd_imm(b, field, -127), sub_e);
   *m = nir_bcsel(b, is_zero(b, x), x,
                  nir_ior(b, nir_ior_imm(b, sign, F32_ONE), mant));
}

/* A float with x's mantissa and sign and the given unbiased exponent, which
 * must be in the normal range. */
static nir_def *
with_exponent(nir_builder *b, nir_def *x, nir_def *e)
{
   nir_def *field = nir_ishl_imm(b, nir_iadd_imm(b, e, 127), 23);
   return nir_ior(b, nir_iand_imm(b, x, ~F32_EXP_MASK), field);
}

/*
 * Rebuild m * 2^E, where m is a normal (or zero) native float and err is the
 * exact signed error of m (exact value minus m), used only for its sign when
 * rounding a subnormal result to nearest-even.
 */
static nir_def *
pack(nir_builder *b, nir_def *m, nir_def *E, nir_def *err)
{
   nir_def *sign = nir_iand_imm(b, m, F32_SIGN);
   nir_def *field = nir_ushr_imm(b, exponent_field(b, m), 23);
   nir_def *T = nir_iadd(b, nir_iadd_imm(b, field, -127), E);

   nir_def *normal = with_exponent(b, m, T);
   nir_def *inf = nir_ior_imm(b, sign, F32_EXP_MASK);

   /* Subnormal: 24-bit significand shifted right by s = -126 - T. */
   nir_def *sig = nir_ior_imm(b, nir_iand_imm(b, m, F32_MAN_MASK), 0x00800000);
   nir_def *s = nir_isub(b, nir_imm_int(b, -126), T);
   nir_def *s_clamped = nir_umin(b, s, nir_imm_int(b, 31));
   nir_def *q = nir_ushr(b, sig, s_clamped);
   nir_def *rem = nir_iand(b, sig, nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), s_clamped), -1));
   nir_def *half = nir_ishl(b, nir_imm_int(b, 1), nir_iadd_imm(b, s_clamped, -1));

   /* Sign of (|exact| - |m|): err has the sign of exact - m. */
   nir_def *err_pos = nir_flt(b, nir_imm_float(b, 0.0), err);
   nir_def *err_neg = nir_flt(b, err, nir_imm_float(b, 0.0));
   nir_def *m_neg = nir_ine_imm(b, sign, 0);
   nir_def *above = nir_bcsel(b, m_neg, err_neg, err_pos);
   nir_def *below = nir_bcsel(b, m_neg, err_pos, err_neg);

   nir_def *up = nir_ior(b, nir_ult(b, half, rem),
                         nir_iand(b, nir_ieq(b, rem, half),
                                  nir_ior(b, above,
                                          nir_iand(b, nir_inot(b, below),
                                                   nir_ine_imm(b, nir_iand_imm(b, q, 1), 0)))));
   nir_def *rounded = nir_iadd(b, q, nir_b2i32(b, up));
   /* Beyond 24 bits of shift the value is below half the smallest subnormal. */
   nir_def *subnormal = nir_bcsel(b, nir_ilt_imm(b, s, 25),
                                  nir_ior(b, sign, rounded), sign);

   nir_def *result = nir_bcsel(b, nir_ilt_imm(b, T, -126), subnormal, normal);
   result = nir_bcsel(b, nir_ilt(b, nir_imm_int(b, 127), T), inf, result);
   return nir_bcsel(b, is_zero(b, m), m, result);
}

/* Error-free sum: s + t = a + b exactly, for normal a and b. */
static nir_def *
two_sum(nir_builder *b, nir_def *x, nir_def *y, nir_def **err)
{
   nir_def *s = nir_fadd(b, x, y);
   nir_def *bb = nir_fsub(b, s, x);
   *err = nir_fadd(b, nir_fsub(b, x, nir_fsub(b, s, bb)), nir_fsub(b, y, bb));
   return s;
}

/* Scale a mantissa m (|m| in [1, 2)) by 2^-d for d >= 0, keeping it normal:
 * beyond 60 only its sign matters to the rounding of a sum. */
static nir_def *
scale_down(nir_builder *b, nir_def *m, nir_def *d)
{
   nir_def *dc = nir_umin(b, d, nir_imm_int(b, 60));
   return nir_bcsel(b, is_zero(b, m), m, with_exponent(b, m, nir_ineg(b, dc)));
}

static nir_def *
soft_add(nir_builder *b, nir_def *x, nir_def *y)
{
   nir_def *mx, *ex, *my, *ey;
   unpack(b, x, &mx, &ex);
   unpack(b, y, &my, &ey);
   /* Zeros take the other operand's exponent so they never set the scale. */
   ex = nir_bcsel(b, is_zero(b, x), ey, ex);
   ey = nir_bcsel(b, is_zero(b, y), ex, ey);
   nir_def *E = nir_imax(b, ex, ey);
   nir_def *ax = scale_down(b, mx, nir_isub(b, E, ex));
   nir_def *ay = scale_down(b, my, nir_isub(b, E, ey));
   nir_def *err;
   nir_def *s = two_sum(b, ax, ay, &err);
   return pack(b, s, E, err);
}

static nir_def *
soft_mul(nir_builder *b, nir_def *x, nir_def *y)
{
   nir_def *mx, *ex, *my, *ey;
   unpack(b, x, &mx, &ex);
   unpack(b, y, &my, &ey);
   nir_def *p = nir_fmul(b, mx, my);
   nir_def *err = nir_ffma(b, mx, my, nir_fneg(b, p));
   nir_def *result = pack(b, p, nir_iadd(b, ex, ey), err);

   /* Infinity times a subnormal is infinity; the native DAZ product is NaN. */
   nir_def *sign = nir_iand_imm(b, nir_ixor(b, x, y), F32_SIGN);
   nir_def *inf = nir_ior_imm(b, sign, F32_EXP_MASK);
   nir_def *any_inf = nir_ior(b, nir_ieq_imm(b, nir_iand_imm(b, x, ~F32_SIGN), F32_EXP_MASK),
                              nir_ieq_imm(b, nir_iand_imm(b, y, ~F32_SIGN), F32_EXP_MASK));
   return nir_bcsel(b, any_inf, inf, result);
}

/*
 * x * y + z. The product is exact as hi + lo; the sum is compensated, giving
 * a faithfully rounded result whose subnormal rounding uses the residual.
 */
static nir_def *
soft_fma(nir_builder *b, nir_def *x, nir_def *y, nir_def *z)
{
   nir_def *mx, *ex, *my, *ey, *mz, *ez;
   unpack(b, x, &mx, &ex);
   unpack(b, y, &my, &ey);
   unpack(b, z, &mz, &ez);
   nir_def *ep = nir_iadd(b, ex, ey);
   nir_def *hi = nir_fmul(b, mx, my);
   nir_def *lo = nir_ffma(b, mx, my, nir_fneg(b, hi));
   nir_def *prod_zero = nir_ior(b, is_zero(b, x), is_zero(b, y));
   ep = nir_bcsel(b, prod_zero, ez, ep);
   ez = nir_bcsel(b, is_zero(b, z), ep, ez);
   nir_def *E = nir_iadd_imm(b, nir_imax(b, ep, ez), 1);

   /* hi in [1, 4): scale by 2^(ep - E) keeping lo's relation exact. */
   nir_def *dp = nir_isub(b, E, ep);
   nir_def *hf = nir_ushr_imm(b, exponent_field(b, hi), 23);
   nir_def *hs = nir_bcsel(b, prod_zero, nir_imm_int(b, 0),
                           with_exponent(b, hi, nir_isub(b, nir_iadd_imm(b, hf, -127),
                                                         nir_umin(b, dp, nir_imm_int(b, 60)))));
   nir_def *lf = nir_ushr_imm(b, exponent_field(b, lo), 23);
   nir_def *ls = nir_bcsel(b, nir_ior(b, prod_zero, is_zero(b, lo)), nir_imm_int(b, 0),
                           with_exponent(b, lo, nir_imax(b, nir_imm_int(b, -120),
                                                         nir_isub(b, nir_iadd_imm(b, lf, -127), dp))));
   nir_def *zs = scale_down(b, mz, nir_isub(b, E, ez));

   nir_def *e1, *e2, *rt;
   nir_def *s1 = two_sum(b, hs, zs, &e1);
   nir_def *t = two_sum(b, ls, e1, &rt);
   nir_def *s = two_sum(b, s1, t, &e2);
   nir_def *result = pack(b, s, E, nir_fadd(b, e2, rt));

   /* Infinity times a subnormal is infinity (native DAZ gives NaN); then
    * add the addend natively, which handles infinite and finite z. */
   nir_def *sign = nir_iand_imm(b, nir_ixor(b, x, y), F32_SIGN);
   nir_def *inf = nir_ior_imm(b, sign, F32_EXP_MASK);
   nir_def *any_inf = nir_ior(b, nir_ieq_imm(b, nir_iand_imm(b, x, ~F32_SIGN), F32_EXP_MASK),
                              nir_ieq_imm(b, nir_iand_imm(b, y, ~F32_SIGN), F32_EXP_MASK));
   return nir_bcsel(b, any_inf, nir_fadd(b, inf, z), result);
}

/* Integer key ordering finite floats and infinities, with -0 == +0. */
static nir_def *
order_key(nir_builder *b, nir_def *x)
{
   nir_def *mag = nir_iand_imm(b, x, ~F32_SIGN);
   return nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, x, F32_SIGN), 0),
                    nir_ineg(b, mag), mag);
}

static nir_def *
is_nan(nir_builder *b, nir_def *x)
{
   return nir_ult(b, nir_imm_int(b, F32_EXP_MASK), nir_iand_imm(b, x, ~F32_SIGN));
}

/* Positive subnormal x as m * 2^(2k) with m in [1, 4). */
static void
unpack_even(nir_builder *b, nir_def *x, nir_def **m, nir_def **k)
{
   nir_def *e;
   unpack(b, x, m, &e);
   *m = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, e, 1), 0),
                  nir_fmul_imm(b, *m, 2.0), *m);
   *k = nir_ishr_imm(b, e, 1);
}

static nir_def *
build_slow(nir_builder *b, nir_alu_instr *alu, nir_def **src)
{
   switch (alu->op) {
   case nir_op_frcp: {
      nir_def *m, *e;
      unpack(b, src[0], &m, &e);
      return pack(b, nir_frcp(b, m), nir_ineg(b, e), nir_imm_float(b, 0.0));
   }
   case nir_op_fsqrt:
   case nir_op_frsq: {
      nir_def *m, *k;
      unpack_even(b, src[0], &m, &k);
      return alu->op == nir_op_fsqrt
                ? pack(b, nir_fsqrt(b, m), k, nir_imm_float(b, 0.0))
                : pack(b, nir_frsq(b, m), nir_ineg(b, k), nir_imm_float(b, 0.0));
   }
   case nir_op_flog2: {
      nir_def *m, *e;
      unpack(b, src[0], &m, &e);
      return nir_fadd(b, nir_flog2(b, m), nir_i2f32(b, e));
   }
   case nir_op_fexp2:
      /* exp2(x) = exp2(x + 64) * 2^-64, rounded once more into the
       * subnormal range. */
      return pack(b, nir_fexp2(b, nir_fadd_imm(b, src[0], 64.0)),
                  nir_imm_int(b, -64), nir_imm_float(b, 0.0));
   case nir_op_fadd:
      return soft_add(b, src[0], src[1]);
   case nir_op_fsub:
      return soft_add(b, src[0], flip_sign(b, src[1]));
   case nir_op_fmul:
      return soft_mul(b, src[0], src[1]);
   case nir_op_ffma:
      return soft_fma(b, src[0], src[1], src[2]);
   case nir_op_flt:
   case nir_op_fge: {
      nir_def *lt = alu->op == nir_op_flt
                       ? nir_ilt(b, order_key(b, src[0]), order_key(b, src[1]))
                       : nir_ige(b, order_key(b, src[0]), order_key(b, src[1]));
      return nir_iand(b, lt, nir_inot(b, nir_ior(b, is_nan(b, src[0]), is_nan(b, src[1]))));
   }
   case nir_op_feq:
   case nir_op_fneu: {
      nir_def *eq = nir_iand(b, nir_ieq(b, order_key(b, src[0]), order_key(b, src[1])),
                             nir_inot(b, nir_ior(b, is_nan(b, src[0]), is_nan(b, src[1]))));
      return alu->op == nir_op_feq ? eq : nir_inot(b, eq);
   }
   case nir_op_fmin:
   case nir_op_fmax: {
      nir_def *lt = nir_ilt(b, order_key(b, src[0]), order_key(b, src[1]));
      nir_def *pick0 = alu->op == nir_op_fmin ? lt : nir_inot(b, lt);
      /* Equal keys (only +-0 or identical bits): min prefers -0, max +0. */
      nir_def *eq = nir_ieq(b, order_key(b, src[0]), order_key(b, src[1]));
      nir_def *neg0 = nir_ine_imm(b, nir_iand_imm(b, src[0], F32_SIGN), 0);
      pick0 = nir_bcsel(b, eq, alu->op == nir_op_fmin ? neg0 : nir_inot(b, neg0), pick0);
      return nir_bcsel(b, pick0, src[0], src[1]);
   }
   case nir_op_fsign:
      return nir_ior_imm(b, nir_iand_imm(b, src[0], F32_SIGN), F32_ONE);
   case nir_op_ffloor:
      return nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, src[0], F32_SIGN), 0),
                       nir_imm_int(b, 0xbf800000u), nir_imm_int(b, 0));
   case nir_op_fceil:
      return nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, src[0], F32_SIGN), 0),
                       nir_imm_int(b, F32_SIGN), nir_imm_int(b, F32_ONE));
   case nir_op_ftrunc:
   case nir_op_fround_even:
      return nir_iand_imm(b, src[0], F32_SIGN);
   case nir_op_ffract:
      /* x - floor(x): x itself when positive; 1 - |x| rounds to 1.0. */
      return nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, src[0], F32_SIGN), 0),
                       nir_imm_int(b, F32_ONE), src[0]);
   case nir_op_fsat:
      return nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, src[0], F32_SIGN), 0),
                       nir_imm_int(b, 0), src[0]);
   default:
      UNREACHABLE("filtered");
   }
}

/* Whether the native result may differ from the preserving one. */
static nir_def *
needs_slow(nir_builder *b, nir_alu_instr *alu, nir_def **src, nir_def *native)
{
   unsigned n = nir_op_infos[alu->op].num_inputs;
   nir_def *any_sub = nir_imm_false(b), *any_nan = nir_imm_false(b);
   nir_def *any_infnan = nir_imm_false(b);
   for (unsigned i = 0; i < n; ++i) {
      any_sub = nir_ior(b, any_sub, is_subnormal(b, src[i]));
      any_nan = nir_ior(b, any_nan, is_nan(b, src[i]));
      any_infnan = nir_ior(b, any_infnan, is_inf_or_nan(b, src[i]));
   }

   switch (alu->op) {
   case nir_op_fadd:
   case nir_op_fsub: {
      /* A flushed result reads as zero; exact cancellation is the only
       * legitimate zero from two finite nonzero operands. */
      nir_def *y = alu->op == nir_op_fsub ? flip_sign(b, src[1]) : src[1];
      nir_def *cancel = nir_ieq(b, src[0], flip_sign(b, y));
      nir_def *flushed = nir_iand(b, is_zero_or_subnormal(b, native),
                                  nir_inot(b, nir_ior(b, cancel,
                                                      nir_iand(b, is_zero(b, src[0]),
                                                               is_zero(b, y)))));
      /* Infinite or NaN operands already give the right native result. */
      return nir_iand(b, nir_ior(b, any_sub, flushed), nir_inot(b, any_infnan));
   }
   case nir_op_fmul: {
      nir_def *flushed = nir_iand(b, is_zero_or_subnormal(b, native),
                                  nir_inot(b, nir_ior(b, is_zero(b, src[0]),
                                                      is_zero(b, src[1]))));
      /* NaN propagates natively; infinity times subnormal is fixed below. */
      return nir_iand(b, nir_ior(b, any_sub, flushed), nir_inot(b, any_nan));
   }
   case nir_op_ffma: {
      nir_def *nonzero = nir_inot(b, nir_iand(b, nir_ior(b, is_zero(b, src[0]),
                                                         is_zero(b, src[1])),
                                              is_zero(b, src[2])));
      nir_def *finite = nir_inot(b, nir_ior(b, nir_ior(b, is_inf_or_nan(b, src[0]),
                                                       is_inf_or_nan(b, src[1])),
                                            is_inf_or_nan(b, src[2])));
      return nir_iand(b, nir_ior(b, any_sub,
                                 nir_iand(b, nir_iand(b, is_zero_or_subnormal(b, native),
                                                      nonzero), finite)),
                      nir_inot(b, any_nan));
   }
   case nir_op_fmin:
   case nir_op_fmax:
      /* With a NaN operand the native selection already returns the other
       * operand's bits. */
      return nir_iand(b, any_sub, nir_inot(b, any_nan));
   case nir_op_frcp:
      return any_sub;
   case nir_op_fsqrt:
   case nir_op_frsq:
   case nir_op_flog2:
      /* Negative inputs have no defined result; keep the native one. */
      return nir_ult(b, nir_iadd_imm(b, src[0], -1), nir_imm_int(b, F32_MAN_MASK));
   case nir_op_fexp2:
      /* Results in the subnormal range; the native result is flushed. A
       * subnormal input natively gives exp2(0) = 1, the correct rounding. */
      return nir_iand(b, nir_flt(b, src[0], nir_imm_float(b, -126.0)),
                      nir_fge(b, src[0], nir_imm_float(b, -151.0)));
   default:
      return any_sub;
   }
}

static bool
preserve_op(const nir_alu_instr *alu)
{
   switch (alu->op) {
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_fsign:
   case nir_op_ffloor:
   case nir_op_fceil:
   case nir_op_ftrunc:
   case nir_op_fround_even:
   case nir_op_ffract:
   case nir_op_fsat:
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_frcp:
   case nir_op_fsqrt:
   case nir_op_frsq:
   case nir_op_flog2:
   case nir_op_fexp2:
      return alu->def.bit_size == 32;
   case nir_op_flt:
   case nir_op_fge:
   case nir_op_feq:
   case nir_op_fneu:
      return alu->src[0].src.ssa->bit_size == 32;
   default:
      return false;
   }
}

static bool
filter(const nir_alu_instr *alu)
{
   return preserve_op(alu) && alu->def.num_components == 1;
}

static bool
scalar_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->def.num_components > 1 && preserve_op(alu);
}

static void
lower(nir_builder *b, nir_alu_instr *alu)
{
   /* Uses to redirect, gathered before any new user of the native result. */
   struct util_dynarray uses;
   util_dynarray_init(&uses, NULL);
   nir_foreach_use_including_if_safe(use, &alu->def)
      util_dynarray_append(&uses, use);

   b->cursor = nir_after_instr(&alu->instr);
   nir_def *src[3];
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
      src[i] = nir_ssa_for_alu_src(b, alu, i);
   nir_def *native = &alu->def;

   nir_def *slow_needed = needs_slow(b, alu, src, native);
   nir_def *slow;
   nir_push_if(b, slow_needed);
   {
      slow = build_slow(b, alu, src);
   }
   nir_push_else(b, NULL);
   nir_pop_if(b, NULL);
   nir_def *result = nir_if_phi(b, slow, native);

   util_dynarray_foreach(&uses, nir_src *, use)
      nir_src_rewrite(*use, result);
   util_dynarray_fini(&uses);
}

bool
agx_nir_lower_apple9_denorm_preserve(nir_shader *shader)
{
   if (!nir_is_denorm_preserve(shader->info.float_controls_execution_mode, 32))
      return false;

   bool progress = nir_lower_alu_to_scalar(shader, scalar_filter, NULL);
   nir_foreach_function_impl(impl, shader) {
      struct util_dynarray work;
      util_dynarray_init(&work, NULL);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                filter(nir_instr_as_alu(instr)))
               util_dynarray_append(&work, nir_instr_as_alu(instr));
         }
      }
      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&work, nir_alu_instr *, alu)
         lower(&b, *alu);
      progress |= work.size != 0;
      nir_progress(work.size != 0, impl, nir_metadata_none);
      util_dynarray_fini(&work);
   }
   return progress;
}
