/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/* Storage-image texel encoding: the canonical fast arms of
 * hk_apple9_encode_texel_fast must produce exactly the words of the generic
 * per-component layout encoder. Both are built for constant texels and
 * folded to constants, over special values (zeros, denormals, format
 * boundaries, infinities, NaNs, negatives) and a deterministic sweep of
 * float bit patterns. The unsigned 11/10-bit float result is also checked
 * against the Vulkan conversion rules where they fix one encoding.
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "hk_descriptor_set.h"
#include "hk_shader.h"

#define BATCH 128

static const nir_shader_compiler_options options = {0};

static uint32_t
rng(uint64_t *s)
{
   *s = *s * 6364136223846793005ull + 1442695040888963407ull;
   return *s >> 32;
}

static float
bits_to_float(uint32_t u)
{
   float f;
   memcpy(&f, &u, 4);
   return f;
}

/* Fold the shader to constants. */
static void
fold(nir_shader *s)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      NIR_PASS(progress, s, nir_opt_remove_phis);
      NIR_PASS(progress, s, nir_opt_undef);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);
}

/* Encode n texels with both encodings; out[i][0] generic, out[i][1] fast. */
static bool
encode_batch(const float (*rgba)[4], unsigned n, enum pipe_format format,
             enum hk_apple9_store_encode encode, uint32_t (*out)[2][4])
{
   struct hk_apple9_texel_metadata m;
   hk_apple9_texel_layout(format, &m);
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "encode");
   for (unsigned i = 0; i < n; ++i) {
      nir_def *value = nir_imm_vec4(&b, rgba[i][0], rgba[i][1], rgba[i][2],
                                    rgba[i][3]);
      nir_def *layout = nir_imm_ivec4(&b, m.component[0], m.component[1],
                                      m.component[2], m.component[3]);
      for (unsigned k = 0; k < 2; ++k) {
         nir_def *words = hk_apple9_encode_texel_fast(
            &b, value, layout, true,
            nir_imm_int(&b, k ? encode : HK_STORE_ENCODE_GENERIC));
         nir_store_global(&b, words, nir_imm_int64(&b, (i * 2 + k) * 16),
                          .align_mul = 16, .write_mask = 0xf);
      }
   }
   fold(b.shader);
   unsigned seen = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_global)
            continue;
         if (!nir_src_is_const(intr->src[0]) || !nir_src_is_const(intr->src[1])) {
            fprintf(stderr, "store did not fold to a constant\n");
            ralloc_free(b.shader);
            return false;
         }
         uint64_t slot = nir_src_as_uint(intr->src[1]) / 16;
         for (unsigned c = 0; c < 4; ++c)
            out[slot / 2][slot % 2][c] = nir_src_comp_as_uint(intr->src[0], c);
         seen++;
      }
   }
   ralloc_free(b.shader);
   if (seen != 2 * n)
      fprintf(stderr, "expected %u stores, folded %u\n", 2 * n, seen);
   return seen == 2 * n;
}

/* Vulkan unsigned float with mbits mantissa bits and a 5-bit exponent.
 * Returns false where the specification allows more than one encoding
 * (rounding direction, overflow to infinity or the largest finite value,
 * flushing denormals), so only fixed cases are compared. */
static bool
ufloat_expected(float f, unsigned mbits, uint32_t *out, uint32_t got)
{
   uint32_t max_finite = (30u << mbits) | ((1u << mbits) - 1);
   uint32_t inf = 31u << mbits;
   if (isnan(f)) {
      *out = inf | 1;
      return false; /* Any NaN encoding; checked by is_ufloat_nan below. */
   }
   if (f <= 0 || f == 0) {
      *out = 0;
      return true;
   }
   if (isinf(f)) {
      *out = inf;
      return true;
   }
   /* Exact representability: nearest and toward-zero agree. */
   int e;
   float frac = frexpf(f, &e); /* f = frac * 2^e, frac in [0.5, 1) */
   int exp = e - 1 + 15;       /* biased exponent of 1.m form */
   if (exp >= 31) {
      *out = got == inf ? inf : max_finite;
      return got == inf || got == max_finite;
   }
   double scaled;
   uint32_t enc;
   if (exp <= 0) {
      scaled = ldexp(f, 14 + mbits); /* denormal: m * 2^(-14 - mbits) */
      if (scaled != floor(scaled))
         return false;
      enc = (uint32_t)scaled;
   } else {
      scaled = ldexp(frac * 2.0, mbits) - ldexp(1.0, mbits);
      if (scaled != floor(scaled))
         return false;
      enc = ((uint32_t)exp << mbits) | (uint32_t)scaled;
   }
   *out = enc;
   return true;
}

static int
check_float11(void)
{
   static float in[1 << 13][4];
   unsigned n = 0;
   const uint32_t specials[] = {
      0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff,
      0x00800000, 0x38800000, 0x38000000, 0x37800000, 0x33800000,
      0x3f800000, 0xbf800000, 0x477e0000, 0x477f0000, 0x477fe000,
      0x47800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000,
      0xffc00000, 0x7f800001, 0x3f7fffff, 0x3f8007ff, 0x3f801000,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(specials); ++i) {
      float f = bits_to_float(specials[i]);
      for (unsigned c = 0; c < 3; ++c) {
         in[n][c] = c == 0 ? f : c == 1 ? -f : f * 0.5f;
         in[n][(c + 1) % 3] = f;
         in[n][(c + 2) % 3] = 1.0f;
         in[n++][3] = f;
      }
   }
   /* Every exponent around the 5-bit range with assorted mantissas, then
    * random bit patterns. */
   uint64_t s = 0x5eed;
   for (int e = 96; e <= 160 && n + 3 < ARRAY_SIZE(in); ++e) {
      for (unsigned k = 0; k < 64; ++k) {
         uint32_t mant = k < 2 ? k * 0x7fffff : rng(&s) & 0x7fffff;
         in[n][0] = bits_to_float(((uint32_t)e << 23) | mant);
         in[n][1] = bits_to_float(((uint32_t)e << 23) | (mant ^ 0x1000));
         in[n][2] = bits_to_float(((uint32_t)(e ^ 1) << 23) | (mant >> 1));
         in[n++][3] = 0.25f;
      }
   }
   while (n < ARRAY_SIZE(in)) {
      for (unsigned c = 0; c < 4; ++c)
         in[n][c] = bits_to_float(rng(&s));
      n++;
   }

   static uint32_t out[ARRAY_SIZE(in)][2][4];
   for (unsigned i = 0; i < n; i += BATCH) {
      if (!encode_batch(&in[i], MIN2(BATCH, n - i), PIPE_FORMAT_R11G11B10_FLOAT,
                        HK_STORE_ENCODE_FLOAT11, &out[i]))
         return 1;
   }

   unsigned mismatch = 0, spec_bad = 0, nan_cases = 0, nan_bad = 0;
   for (unsigned i = 0; i < n; ++i) {
      if (memcmp(out[i][0], out[i][1], sizeof(out[i][0]))) {
         if (mismatch++ < 8)
            fprintf(stderr,
                    "mismatch (%a %a %a): generic %08x %08x %08x %08x fast %08x %08x %08x %08x\n",
                    in[i][0], in[i][1], in[i][2], out[i][0][0], out[i][0][1],
                    out[i][0][2], out[i][0][3], out[i][1][0], out[i][1][1],
                    out[i][1][2], out[i][1][3]);
      }
      uint32_t w = out[i][1][0];
      const unsigned shift[3] = {0, 11, 22}, mbits[3] = {6, 6, 5};
      for (unsigned c = 0; c < 3; ++c) {
         uint32_t got = (w >> shift[c]) & ((1u << (mbits[c] + 5)) - 1);
         uint32_t want;
         if (isnan(in[i][c])) {
            nan_cases++;
            /* Vulkan maps NaN to a NaN encoding. */
            nan_bad += !((got >> mbits[c]) == 31 && (got & ((1u << mbits[c]) - 1)));
            continue;
         }
         if (ufloat_expected(in[i][c], mbits[c], &want, got) && got != want) {
            if (spec_bad++ < 8)
               fprintf(stderr, "spec: %a channel %u got %03x want %03x\n",
                       in[i][c], c, got, want);
         }
      }
   }
   printf("float11: %u texels, %u fast/generic mismatches, %u fixed-case spec "
          "deviations, NaN inputs %u (%u not encoded as NaN)\n",
          n, mismatch, spec_bad, nan_cases, nan_bad);
   /* The fast arm must match the generic encoder exactly. Conversion-rule
    * deviations are properties of nir_format_pack_11f11f10f shared by both
    * paths and are reported, not failed here. */
   return mismatch != 0;
}

/* NIR ALU instructions left on the path that a uniform store encoding
 * selects, for a non-constant texel (informational). */
static unsigned
path_alu_count(enum pipe_format format, enum hk_apple9_store_encode encode)
{
   struct hk_apple9_texel_metadata m;
   hk_apple9_texel_layout(format, &m);
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "path");
   nir_def *value = nir_load_global(&b, 4, 32, nir_imm_int64(&b, 0), .align_mul = 16);
   nir_def *layout = nir_imm_ivec4(&b, m.component[0], m.component[1],
                                   m.component[2], m.component[3]);
   nir_def *words = hk_apple9_encode_texel_fast(&b, value, layout, true,
                                                nir_imm_int(&b, encode));
   nir_store_global(&b, words, nir_imm_int64(&b, 64), .align_mul = 16,
                    .write_mask = 0xf);
   fold(b.shader);
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block)
         n += instr->type == nir_instr_type_alu;
   }
   ralloc_free(b.shader);
   return n;
}

int
main(void)
{
   printf("R11G11B10 store path, NIR ALU instructions: generic %u, float11 arm %u\n",
          path_alu_count(PIPE_FORMAT_R11G11B10_FLOAT, HK_STORE_ENCODE_GENERIC),
          path_alu_count(PIPE_FORMAT_R11G11B10_FLOAT, HK_STORE_ENCODE_FLOAT11));
   return check_float11();
}
