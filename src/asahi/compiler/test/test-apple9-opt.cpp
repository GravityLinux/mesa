/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_ir.h"
#include "agx_compile_apple9.h"
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>

class Apple9Optimization : public ::testing::Test {
protected:
   agx_apple9_vir_program p;
   void SetUp() override { agx_apple9_vir_init(&p); }
   void TearDown() override { agx_apple9_vir_finish(&p); }
   uint32_t imm(uint32_t n) {
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, n);
   }
   uint32_t add(uint32_t a, uint32_t b) {
      uint32_t src[] = {a, b};
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
         AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
   }
};

TEST_F(Apple9Optimization, FoldsWrappingArithmeticAndArithmeticRightShift)
{
   auto sum = add(imm(0xffffffff), imm(2));
   auto negative = imm(0x80000001);
   auto shift = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_ISHR,
      AGX_APPLE9_ENC_SHIFT_EXTENDED, &negative, 1, 31);
   auto result = add(sum, shift);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   EXPECT_EQ(agx_apple9_definition(&p, sum)->immediate, 1u);
   EXPECT_EQ(agx_apple9_definition(&p, shift)->immediate, 0xffffffffu);
   EXPECT_EQ(agx_apple9_definition(&p, result)->op, AGX_APPLE9_VIR_IMM);
   EXPECT_EQ(agx_apple9_definition(&p, result)->immediate, 0u);
}

TEST_F(Apple9Optimization, RewritesBothProductWordsButKeepsSignedness)
{
   uint32_t src[] = {agx_apple9_vir_input(&p, 4), agx_apple9_vir_input(&p, 5)};
   auto first = agx_apple9_vir_emit_mul_wide(&p, src, false);
   auto same = agx_apple9_vir_emit_mul_wide(&p, src, false);
   auto signed_product = agx_apple9_vir_emit_mul_wide(&p, src, true);
   auto low = add(first, same);
   auto high = add(same + 1, signed_product + 1);
   p.output = same + 1;
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   EXPECT_EQ(agx_apple9_definition(&p, low)->src[0], first);
   EXPECT_EQ(agx_apple9_definition(&p, low)->src[1], first);
   EXPECT_EQ(agx_apple9_definition(&p, high)->src[0], first + 1);
   EXPECT_EQ(agx_apple9_definition(&p, high)->src[1], signed_product + 1);
   EXPECT_EQ(p.output, first + 1);
}

TEST_F(Apple9Optimization, ActiveLaneTransitionsAndBlocksEndCseScope)
{
   auto input = agx_apple9_vir_input(&p, 4);
   auto constant = imm(7);
   auto before = add(input, constant);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_EXEC_MASK_ELSE,
      AGX_APPLE9_ENC_EXEC_MASK_ELSE, nullptr, 0, 0));
   auto after_mask = add(input, constant);
   agx_apple9_block_begin(&p, agx_apple9_block_create(&p));
   auto after_block = add(input, constant);
   auto result = add(after_mask, after_block);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   EXPECT_EQ(agx_apple9_definition(&p, result)->src[0], after_mask);
   EXPECT_EQ(agx_apple9_definition(&p, result)->src[1], after_block);
   EXPECT_NE(agx_apple9_definition(&p, result)->src[0], before);
}

TEST_F(Apple9Optimization, MemoryReadsAreNeverPureExpressionCse)
{
   auto index = imm(0);
   auto a = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_DEVICE_LOAD,
      AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
   auto b = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_DEVICE_LOAD,
      AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
   auto sum = add(a, b);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   EXPECT_EQ(agx_apple9_definition(&p, sum)->src[0], a);
   EXPECT_EQ(agx_apple9_definition(&p, sum)->src[1], b);
}

TEST(Apple9Packer, NativeHalfPairDefinesBothHalvesAndRetainsRepeatedInput)
{
   uint8_t phys[] = {95, 64};
   agx_apple9_vir_instr I = {};
   I.op = AGX_APPLE9_VIR_PACK_HALF_2X16;
   I.encoding = AGX_APPLE9_ENC_PACK_HALF_2X16;
   I.dest = 0; I.src[0] = I.src[1] = 1; I.nr_srcs = 2;
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t expected[] = {0xf1,0x81,0x54,0x81,0x00,0x13,
                               0xf1,0x01,0x5c,0x81,0x04,0x13};
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
   phys[1] = 95;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
}

TEST(Apple9Packer, WideMultiplyRequiresAlignedAdjacentProductWords)
{
   uint8_t phys[] = {94,95,64,80};
   agx_apple9_vir_instr I = {};
   I.op = AGX_APPLE9_VIR_IMUL_WIDE;
   I.encoding = AGX_APPLE9_ENC_INT_MUL_WIDE;
   I.dest = 0; I.dest_components = 2;
   I.src[0] = 2; I.src[1] = 3; I.nr_srcs = 2; I.immediate = 1;
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t expected[] = {0x9f,0x00,0x54,0xbc,0x02,0x00,0x81,0x02,0xe0,0x26,0x1e,0x00};
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
   phys[1] = 93;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   phys[0] = 93; phys[1] = 94;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
}

TEST(Apple9Packer, RegisterShiftsHaveIndependentSourceReleaseFlags)
{
   uint8_t phys[] = {80,95,64};
   agx_apple9_vir_instr I = {};
   I.op = AGX_APPLE9_VIR_ISHR;
   I.encoding = AGX_APPLE9_ENC_SHIFT_ARITH_REGISTER;
   I.dest = 0; I.src[0] = 1; I.src[1] = 2; I.nr_srcs = 2;
   I.live_after_mask = 3;
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t arithmetic[] = {0xa7,0x01,0x54,0xa0,0x02,0x7c,0x01,0x1a,0xe2,0x00};
   ASSERT_EQ(packed.length, sizeof(arithmetic));
   EXPECT_EQ(memcmp(packed.bytes, arithmetic, sizeof(arithmetic)), 0);
   I.op = AGX_APPLE9_VIR_USHR;
   I.encoding = AGX_APPLE9_ENC_SHIFT_LOGICAL_REGISTER;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t logical[] = {0xa7,0x00,0x54,0xa0,0x02,0x7c,0x01,0x02,0xf0,0x10,0x05,0x00};
   ASSERT_EQ(packed.length, sizeof(logical));
   EXPECT_EQ(memcmp(packed.bytes, logical, sizeof(logical)), 0);
   I.nr_srcs = 1; I.immediate = 32;
   I.encoding = AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
}

TEST_F(Apple9Optimization, ConstantRegisterShiftFollowsNativeAmountContract)
{
   auto input = imm(0x81234567);
   uint32_t src[] = {input, imm(33)};
   auto logical = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_USHR,
      AGX_APPLE9_ENC_SHIFT_LOGICAL_REGISTER, src, 2, 0);
   auto arithmetic = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_ISHR,
      AGX_APPLE9_ENC_SHIFT_ARITH_REGISTER, src, 2, 0);
   src[1] = imm(0x10004);
   auto low_half = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_USHR,
      AGX_APPLE9_ENC_SHIFT_LOGICAL_REGISTER, src, 2, 0);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   EXPECT_EQ(agx_apple9_definition(&p, logical)->immediate, 0u);
   EXPECT_EQ(agx_apple9_definition(&p, arithmetic)->immediate, 0xffffffffu);
   EXPECT_EQ(agx_apple9_definition(&p, low_half)->immediate, 0x08123456u);
}

TEST_F(Apple9Optimization, ComposesFloatModifiersAndKeepsDifferentSignsDistinct)
{
   auto a = agx_apple9_vir_input(&p, 4);
   auto b = agx_apple9_vir_input(&p, 5);
   auto bits = [&](agx_apple9_vir_opcode op, uint32_t x, uint32_t mask) {
      uint32_t src[] = {x, imm(mask)};
      return agx_apple9_vir_emit(&p, op, AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   };
   auto negate = bits(AGX_APPLE9_VIR_IXOR, a, 0x80000000u);
   auto absolute = bits(AGX_APPLE9_VIR_IAND, negate, 0x7fffffffu);
   auto negative_absolute = bits(AGX_APPLE9_VIR_IXOR, absolute, 0x80000000u);
   uint32_t src[] = {negative_absolute, b};
   auto x = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FSUB,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, src, 2, 0);
   src[0] = absolute;
   auto y = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FSUB,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, src, 2, 0);
   auto sum = add(x, y);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   const auto *first = agx_apple9_definition(&p, x);
   EXPECT_EQ(first->op, AGX_APPLE9_VIR_FADD);
   EXPECT_EQ(first->src[0], a);
   EXPECT_EQ(first->src_abs_mask, 1u);
   EXPECT_EQ(first->src_neg_mask, 3u);
   EXPECT_EQ(agx_apple9_definition(&p, y)->src_neg_mask, 2u);
   EXPECT_EQ(agx_apple9_definition(&p, sum)->src[1], y);
}

TEST_F(Apple9Optimization, SelectsOnlyExactlyRepresentableFloatImmediates)
{
   auto a = agx_apple9_vir_input(&p, 4);
   for (uint32_t value : {0u, 0x80000000u, 0x3fa00000u, 0xbfa00000u,
                          0x3f800001u, 0x7f800000u, 0x7fc00000u}) {
      uint32_t src[] = {a, imm(value)};
      auto result = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMUL,
         AGX_APPLE9_ENC_FLOAT2_COMPACT, src, 2, 0);
      ASSERT_TRUE(agx_apple9_optimize_vir(&p));
      ASSERT_TRUE(agx_apple9_analyze_uses(&p));
      const auto *I = agx_apple9_definition(&p, result);
      bool exact = (value & 0x7fffffffu) == 0 ||
                   (value & 0x7fffffffu) == 0x3fa00000u;
      EXPECT_EQ(I->op, exact ? AGX_APPLE9_VIR_FMUL_IMM : AGX_APPLE9_VIR_FMUL);
      if (exact) {
         EXPECT_EQ(I->immediate, value);
      }
   }
}

TEST(Apple9Packer, FloatAbsoluteMultiplyAndAddHaveDifferentLengths)
{
   uint8_t phys[] = {95,80,64,32};
   agx_apple9_vir_instr I = {};
   I.op = AGX_APPLE9_VIR_FADD; I.encoding = AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED;
   I.dest = 0; I.src[0] = 1; I.src[1] = 2; I.nr_srcs = 2;
   I.src_abs_mask = I.src_neg_mask = I.live_after_mask = 3;
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t add[] = {0xf9,0xa1,0x64,0x81,0x02,0x1d,0x02,0x80,0x03,0x00};
   ASSERT_EQ(packed.length, sizeof(add));
   EXPECT_EQ(memcmp(packed.bytes, add, sizeof(add)), 0);
   I.op = AGX_APPLE9_VIR_FMUL;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   I.encoding = AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t mul[] = {0xf9,0xa1,0x65,0x81,0x02,0x1d,0x02,0x80,0x03,0x00,0x00,0x00};
   ASSERT_EQ(packed.length, sizeof(mul));
   EXPECT_EQ(memcmp(packed.bytes, mul, sizeof(mul)), 0);
   I.op = AGX_APPLE9_VIR_FMA; I.encoding = AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED;
   I.src[2] = 3; I.nr_srcs = 3;
   I.src_abs_mask = I.live_after_mask = 7; I.src_neg_mask = 5;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t fma[] = {0xf9,0xa1,0x66,0x81,0x1b,0xc0,0x02,0x1d,0x00,0x80,0x03,0x00};
   ASSERT_EQ(packed.length, sizeof(fma));
   EXPECT_EQ(memcmp(packed.bytes, fma, sizeof(fma)), 0);
}

TEST_F(Apple9Optimization, UniformWritesEndUniformReadCseScope)
{
   auto zero = imm(0), value = agx_apple9_vir_input(&p, 4);
   auto before = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR_UNIFORM,
      AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1, AGX_APPLE9_PREAMBLE_BASE);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_STORE_UNIFORM,
      AGX_APPLE9_ENC_STORE_UNIFORM, &value, 1, AGX_APPLE9_PREAMBLE_BASE));
   auto after = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR_UNIFORM,
      AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1, AGX_APPLE9_PREAMBLE_BASE);
   auto result = add(before, after);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   EXPECT_EQ(agx_apple9_definition(&p, result)->src[0], before);
   EXPECT_EQ(agx_apple9_definition(&p, result)->src[1], after);
}
