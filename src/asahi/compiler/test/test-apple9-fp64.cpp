/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include "agx_compile_apple9.h"
#include "compiler/nir/nir_builder.h"

namespace {

struct opcode_case {
   nir_op op;
   unsigned bits;
   unsigned components = 1;
};

class Apple9SoftFP64 : public ::testing::TestWithParam<opcode_case> {};

static nir_def *
input(nir_builder *b, unsigned binding, unsigned bits, unsigned components = 1)
{
   return nir_load_ubo(b, components, bits, nir_imm_int(b, binding),
                      nir_imm_int(b, 0), .align_mul = bits / 8,
                      .range = components * bits / 8);
}

static nir_def *
operation(nir_builder *b, opcode_case test)
{
   nir_def *a = input(b, 0, test.bits, test.components);
   nir_def *c = input(b, 1, test.bits, test.components);
   nir_def *shift = input(b, 2, 32);
   switch (test.op) {
   case nir_op_ffma:
      return nir_ffma(b, a, c, input(b, 2, test.bits));
   case nir_op_bcsel:
      return nir_bcsel(b, nir_ine_imm(b, shift, 0), a, c);
   case nir_op_ishl:
   case nir_op_ishr:
   case nir_op_ushr:
      return nir_build_alu2(b, test.op, a, shift);
   case nir_op_bitfield_insert: {
      nir_def *offset = nir_iand_imm(b, shift, 31);
      nir_def *bits = nir_umin(b, nir_iand_imm(b, input(b, 3, 32), 63),
                             nir_isub(b, nir_imm_int(b, 32), offset));
      return nir_bitfield_insert(b, a, c, offset, bits);
   }
   case nir_op_ubfe:
   case nir_op_ibfe:
      return nir_build_alu3(b, test.op, a, shift, input(b, 3, 32));
   case nir_op_ubitfield_extract:
   case nir_op_ibitfield_extract:
      return nir_build_alu3(b, test.op, a, nir_iand_imm(b, shift, 15),
                           nir_iand_imm(b, input(b, 3, 32), 15));
   default:
      return nir_build_alu(b, test.op, a,
                          nir_op_infos[test.op].num_inputs > 1 ? c : nullptr,
                          nullptr, nullptr);
   }
}

static void
store(nir_builder *b, nir_def *value)
{
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_store_ssbo(b, value, nir_imm_int(b, 0), nir_imm_int(b, 0),
                  .write_mask = BITFIELD_MASK(value->num_components),
                  .align_mul = unsigned(value->bit_size) / 8);
}

TEST_P(Apple9SoftFP64, Compiles)
{
   const opcode_case test = GetParam();
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "%s", nir_op_infos[test.op].name);
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *value = operation(&b, test);
   ASSERT_EQ(nir_def_as_alu(value)->op, test.op);
   store(&b, value);
   nir_validate_shader(b.shader, "software fp64 helper input");

   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   EXPECT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_EQ(reason, nullptr);
   EXPECT_NE(compiled.binary, nullptr);
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
   ralloc_free(b.shader);
}

/* Nonconstant UBO operands prevent constant folding from turning these into
 * tests of immediate loads. Include both halves of every 64-bit result. */
INSTANTIATE_TEST_SUITE_P(
   Helpers, Apple9SoftFP64,
   ::testing::Values(
      opcode_case{nir_op_uadd_carry, 32},
      opcode_case{nir_op_usub_borrow, 32},
      opcode_case{nir_op_umul_high, 32},
      opcode_case{nir_op_imul_high, 32},
      opcode_case{nir_op_imul_2x32_64, 32},
      opcode_case{nir_op_umul_2x32_64, 32},
      opcode_case{nir_op_ufind_msb, 32},
      opcode_case{nir_op_find_lsb, 32},
      opcode_case{nir_op_bitfield_insert, 32},
      opcode_case{nir_op_ubfe, 32},
      opcode_case{nir_op_ibfe, 32},
      opcode_case{nir_op_ubitfield_extract, 32},
      opcode_case{nir_op_ibitfield_extract, 32},
      opcode_case{nir_op_ishl, 32},
      opcode_case{nir_op_ishr, 32},
      opcode_case{nir_op_ushr, 32},
      opcode_case{nir_op_pack_64_2x32_split, 32},
      opcode_case{nir_op_unpack_64_2x32_split_x, 64},
      opcode_case{nir_op_unpack_64_2x32_split_y, 64},
      opcode_case{nir_op_bcsel, 32, 2},
      opcode_case{nir_op_u2u64, 32},
      opcode_case{nir_op_i2i64, 32},
      opcode_case{nir_op_u2u32, 64},
      opcode_case{nir_op_i2i32, 64},
      opcode_case{nir_op_f2i64, 32},
      opcode_case{nir_op_f2u64, 32},
      opcode_case{nir_op_i2f32, 64},
      opcode_case{nir_op_u2f32, 64},
      opcode_case{nir_op_iadd, 64},
      opcode_case{nir_op_isub, 64},
      opcode_case{nir_op_imul, 64},
      opcode_case{nir_op_imul_high, 64},
      opcode_case{nir_op_umul_high, 64},
      opcode_case{nir_op_idiv, 64},
      opcode_case{nir_op_udiv, 64},
      opcode_case{nir_op_imod, 64},
      opcode_case{nir_op_umod, 64},
      opcode_case{nir_op_iabs, 64},
      opcode_case{nir_op_isign, 64},
      opcode_case{nir_op_ineg, 64},
      opcode_case{nir_op_iand, 64},
      opcode_case{nir_op_ior, 64},
      opcode_case{nir_op_ixor, 64},
      opcode_case{nir_op_inot, 64},
      opcode_case{nir_op_imin, 64},
      opcode_case{nir_op_imax, 64},
      opcode_case{nir_op_umin, 64},
      opcode_case{nir_op_umax, 64},
      opcode_case{nir_op_ishl, 64},
      opcode_case{nir_op_ishr, 64},
      opcode_case{nir_op_ushr, 64},
      opcode_case{nir_op_bcsel, 64, 2},
      opcode_case{nir_op_ufind_msb, 64},
      opcode_case{nir_op_find_lsb, 64},
      opcode_case{nir_op_bit_count, 64},
      opcode_case{nir_op_bitfield_reverse, 64},
      opcode_case{nir_op_ubitfield_extract, 64},
      opcode_case{nir_op_ibitfield_extract, 64},
      opcode_case{nir_op_ieq, 64},
      opcode_case{nir_op_ine, 64},
      opcode_case{nir_op_ilt, 64},
      opcode_case{nir_op_ige, 64},
      opcode_case{nir_op_ult, 64},
      opcode_case{nir_op_uge, 64},
      opcode_case{nir_op_iadd_sat, 64},
      opcode_case{nir_op_uadd_sat, 64},
      opcode_case{nir_op_usub_sat, 64}),
   [](const ::testing::TestParamInfo<opcode_case> &info) {
      return std::string(nir_op_infos[info.param.op].name) + "_" +
             std::to_string(info.param.bits) + "_" +
             std::to_string(info.param.components);
   });

/* The state tracker supplies and inlines float64.glsl before the backend.
 * Direct raw doubles must fail, rather than silently selecting FP32 ALU.
 * agx_tests has no GLSL/GL-context link dependency; see N3.md for Tier A. */
class Apple9SoftFP64Rejection : public ::testing::TestWithParam<opcode_case> {};

TEST_P(Apple9SoftFP64Rejection, RequiresCallerLibrary)
{
   const opcode_case test = GetParam();
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unlowered double");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   store(&b, operation(&b, test));
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   EXPECT_FALSE(agx_compile_apple9_tiny(b.shader, &compiled, nullptr, &reason));
   EXPECT_STREQ(reason,
                "Apple9 buffer compiler encountered unsupported NIR");
   EXPECT_EQ(compiled.binary, nullptr);
   ralloc_free(b.shader);
}

INSTANTIATE_TEST_SUITE_P(
   RawDoubles, Apple9SoftFP64Rejection,
   ::testing::Values(
      opcode_case{nir_op_fadd, 64},
      opcode_case{nir_op_fmul, 64},
      opcode_case{nir_op_ffma, 64},
      opcode_case{nir_op_fdiv, 64},
      opcode_case{nir_op_fsqrt, 64},
      opcode_case{nir_op_ffract, 64},
      opcode_case{nir_op_fround_even, 64},
      opcode_case{nir_op_f2f64, 32},
      opcode_case{nir_op_f2f32, 64},
      opcode_case{nir_op_f2i32, 64},
      opcode_case{nir_op_i2f64, 32},
      opcode_case{nir_op_i2f64, 64},
      opcode_case{nir_op_f2i64, 64}),
   [](const ::testing::TestParamInfo<opcode_case> &info) {
      return std::string(nir_op_infos[info.param.op].name) + "_" +
             std::to_string(info.param.bits);
   });

} // namespace
