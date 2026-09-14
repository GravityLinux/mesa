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

TEST(Apple9Compiler, TextureChannelSelectionReachesPackedNativeResults)
{
   /* Independently compiled T8132 read/sample shaders for the complete mask
    * set. These are only the channel bits, excluding resources and LOD. */
   const uint8_t channel3[] = {0,0xa0,0xa8,0xa8,0xb0,0xa0,0xa8,0xb0,
                              0xb8,0xa0,0xb0,0xa8,0xb8,0xa0,0xb0,0xb8};
   const uint8_t channel4[] = {0,0x90,0x90,0xb0,0x90,0x90,0x90,0xb0,
                              0x90,0xb0,0x90,0xb0,0x90,0xb0,0xb0,0xb0};
   const unsigned channel10 = BITFIELD_BIT(5) | BITFIELD_BIT(6) |
      BITFIELD_BIT(10) | BITFIELD_BIT(11) | BITFIELD_BIT(12) |
      BITFIELD_BIT(13) | BITFIELD_BIT(14);
   for (unsigned mask = 1; mask < 16; ++mask) {
      SCOPED_TRACE(mask);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "packed_texture_%u", mask);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
      tex->op = nir_texop_tex;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->texture_index = 3;
      tex->sampler_index = 5;
      tex->src[0].src_type = nir_tex_src_coord;
      tex->src[0].src = nir_src_for_ssa(nir_imm_vec2(&b, .375, .625));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_def *result = nullptr;
      for (unsigned c = 0; c < 4; ++c) {
         if (!(mask & BITFIELD_BIT(c)))
            continue;
         auto component = nir_fmul_imm(&b, nir_channel(&b, &tex->def, c), 1 << c);
         result = result ? nir_fadd(&b, result, component) : component;
      }
      nir_store_output(&b, nir_vec4(&b, result, result, result, nir_imm_float(&b, 1)),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
         << (reason ?: "");
      const auto *code = static_cast<const uint8_t *>(compiled.binary);
      unsigned samples = 0;
      for (unsigned i = 0; i + 14 <= compiled.info.binary_size; i += 2) {
         if ((code[i] & 7) != 5 || !(code[i + 1] & 0x80) ||
             code[i + 2] != 0x0c || code[i + 12] != 1)
            continue;
         EXPECT_EQ(code[i + 3], channel3[mask]);
         EXPECT_EQ(code[i + 4], channel4[mask]);
         EXPECT_EQ(bool(code[i + 10] & 0x40), bool(channel10 & BITFIELD_BIT(mask)));
         EXPECT_LE((code[i] >> 3) + util_bitcount(mask), 32u);
         ++samples;
      }
      EXPECT_EQ(samples, 1u);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

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

TEST_F(Apple9Optimization, UniformAndImmediateOperandsLeaveOnlyGprUses)
{
   auto zero = imm(0);
   auto uniform = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR_UNIFORM,
      AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1, 63);
   auto input = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_GET_GLOBAL_ID,
      AGX_APPLE9_ENC_GET_SR, nullptr, 0, 0);
   uint32_t add_src[] = {input, uniform};
   auto sum = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, add_src, 2, 0);
   uint32_t fma_src[] = {uniform, input, imm(0xbe800000)};
   p.output = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMA,
      AGX_APPLE9_ENC_FLOAT3_EXTENDED, fma_src, 3, 0);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   const auto *A = agx_apple9_definition(&p, sum);
   EXPECT_EQ(A->nr_srcs, 1);
   EXPECT_EQ(A->src[0], input);
   EXPECT_EQ(A->alu_src_uniform_mask, 2);
   EXPECT_EQ(A->alu_src_value[1], 63u);
   const auto *F = agx_apple9_definition(&p, p.output);
   EXPECT_EQ(F->nr_srcs, 1);
   EXPECT_EQ(F->src[0], input);
   EXPECT_EQ(F->alu_src_uniform_mask, 2);
   EXPECT_EQ(F->alu_src_immediate_mask, 4);
   EXPECT_EQ(F->alu_src_value[1], 63u);
   EXPECT_EQ(F->alu_src_value[2], 0xbe800000u);
   EXPECT_EQ(agx_apple9_uses(&p, uniform), nullptr);
   const char *reason = nullptr;
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "inline_sources");
   ASSERT_TRUE(agx_apple9_allocate_publications(&p, &reason)) << reason;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
   ASSERT_TRUE(agx_apple9_validate_vir_allocation(&p, &reason)) << reason;
   ralloc_free(nir.shader);
}

TEST_F(Apple9Optimization, UniformWritesBlockReadFoldingAndExpressionReuse)
{
   auto zero = imm(0);
   auto input = agx_apple9_vir_input(&p, 4);
   auto first = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR_UNIFORM,
      AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1, 48);
   agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_STORE_UNIFORM,
      AGX_APPLE9_ENC_STORE_UNIFORM, &input, 1, 48);
   uint32_t old_src[] = {first, input};
   auto old_sum = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, old_src, 2, 0);
   auto second = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR_UNIFORM,
      AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1, 48);
   uint32_t new_src[] = {second, input};
   p.output = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, new_src, 2, 0);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   EXPECT_EQ(agx_apple9_definition(&p, old_sum)->nr_srcs, 2);
   EXPECT_EQ(agx_apple9_definition(&p, old_sum)->alu_src_uniform_mask, 0);
   EXPECT_EQ(agx_apple9_definition(&p, p.output)->nr_srcs, 1);
   EXPECT_EQ(agx_apple9_definition(&p, p.output)->alu_src_uniform_mask, 1);
   EXPECT_NE(p.output, old_sum);
}

TEST_F(Apple9Optimization, SpecialFunctionsComposeAbsoluteThenNegation)
{
   auto input = agx_apple9_vir_input(&p, 83);
   uint32_t abs_src[] = {input, imm(0x7fffffff)};
   auto absolute = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IAND,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, abs_src, 2, 0);
   uint32_t neg_src[] = {absolute, imm(0x80000000)};
   auto negative = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, neg_src, 2, 0);
   uint32_t outputs[10]; unsigned count = 0;
   for (auto op : {AGX_APPLE9_VIR_FRCP, AGX_APPLE9_VIR_FRSQ,
         AGX_APPLE9_VIR_FSQRT_FACTOR, AGX_APPLE9_VIR_FSIN_FACTOR,
         AGX_APPLE9_VIR_FEXP2, AGX_APPLE9_VIR_FLOG2,
         AGX_APPLE9_VIR_FFLOOR, AGX_APPLE9_VIR_FCEIL,
         AGX_APPLE9_VIR_FTRUNC, AGX_APPLE9_VIR_FROUND_EVEN})
      outputs[count++] = agx_apple9_vir_emit(&p, op,
         AGX_APPLE9_ENC_FLOAT_SPECIAL, &negative, 1, 3);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   for (auto output : outputs) {
      const auto *I = agx_apple9_definition(&p, output);
      EXPECT_EQ(I->src[0], input);
      EXPECT_EQ(I->src_abs_mask, 1);
      EXPECT_EQ(I->src_neg_mask, 1);
   }
}

TEST_F(Apple9Optimization, FloatSelectCanonicalizesNegatedComparisons)
{
   auto a = agx_apple9_vir_input(&p, 4);
   auto b = agx_apple9_vir_input(&p, 5);
   auto c = agx_apple9_vir_input(&p, 6);
   auto d = agx_apple9_vir_input(&p, 7);
   for (auto condition : {AGX_APPLE9_SELECT_FEQ, AGX_APPLE9_SELECT_FGT,
                          AGX_APPLE9_SELECT_FLT}) {
      for (bool both : {false, true}) {
         agx_apple9_block_begin(&p, agx_apple9_block_create(&p));
         uint32_t neg_a_src[] = {a, imm(0x80000000)};
         auto neg_a = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
            AGX_APPLE9_ENC_LOGIC_EXTENDED, neg_a_src, 2, 0);
         uint32_t neg_b_src[] = {b, imm(0x80000000)};
         auto neg_b = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
            AGX_APPLE9_ENC_LOGIC_EXTENDED, neg_b_src, 2, 0);
         uint32_t src[] = {neg_a, both ? neg_b : b, c, d};
         p.output = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_SELECT,
            AGX_APPLE9_ENC_SELECT_GPR_WIDE, src, 4, condition);
         ASSERT_TRUE(agx_apple9_optimize_vir(&p));
         ASSERT_TRUE(agx_apple9_analyze_uses(&p));
         const auto *I = agx_apple9_definition(&p, p.output);
         EXPECT_EQ(I->src[0], both ? a : b);
         EXPECT_EQ(I->src[1], both ? b : a);
         EXPECT_EQ(I->src[2], c);
         EXPECT_EQ(I->src[3], d);
         EXPECT_EQ(I->src_neg_mask, both ? 0 : 2);
         auto reversed = condition == AGX_APPLE9_SELECT_FGT ? AGX_APPLE9_SELECT_FLT
            : condition == AGX_APPLE9_SELECT_FLT ? AGX_APPLE9_SELECT_FGT
            : AGX_APPLE9_SELECT_FEQ;
         EXPECT_EQ(I->immediate, reversed);
      }
   }
}

TEST_F(Apple9Optimization, MinmaxFoldsOnlyTheSupportedNegatedSecondOperand)
{
   auto a = agx_apple9_vir_input(&p, 4);
   auto b = agx_apple9_vir_input(&p, 5);
   uint32_t abs_src[] = {b, imm(0x7fffffff)};
   auto absolute = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IAND,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, abs_src, 2, 0);
   uint32_t neg_src[] = {absolute, imm(0x80000000)};
   auto negative = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, neg_src, 2, 0);
   uint32_t src[] = {a, negative};
   p.output = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMIN,
      AGX_APPLE9_ENC_MINMAX_COMPACT, src, 2, 0);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   const auto *I = agx_apple9_definition(&p, p.output);
   EXPECT_EQ(I->src[0], a);
   EXPECT_EQ(I->src[1], absolute);
   EXPECT_EQ(I->src_neg_mask, 2);
   EXPECT_EQ(I->src_abs_mask, 0);
}

TEST_F(Apple9Optimization, SaturationFusesOnlySupportedSoleUseProducers)
{
   struct {
      agx_apple9_vir_opcode op;
      agx_apple9_encoding encoding;
      unsigned sources, immediate;
      bool fused;
   } cases[] = {
      {AGX_APPLE9_VIR_FADD, AGX_APPLE9_ENC_FLOAT2_COMPACT, 2, 0, true},
      {AGX_APPLE9_VIR_FMUL, AGX_APPLE9_ENC_FLOAT2_COMPACT, 2, 0, true},
      {AGX_APPLE9_VIR_FMA, AGX_APPLE9_ENC_FLOAT3_EXTENDED, 3, 0, true},
      {AGX_APPLE9_VIR_FRCP, AGX_APPLE9_ENC_FLOAT_SPECIAL, 1, 3, true},
      {AGX_APPLE9_VIR_FRSQ, AGX_APPLE9_ENC_FLOAT_SPECIAL, 1, 3, true},
      {AGX_APPLE9_VIR_FSQRT_FACTOR, AGX_APPLE9_ENC_FLOAT_SPECIAL, 1, 3, true},
      {AGX_APPLE9_VIR_FSIN_FACTOR, AGX_APPLE9_ENC_FLOAT_SPECIAL, 1, 3, true},
      {AGX_APPLE9_VIR_FEXP2, AGX_APPLE9_ENC_FLOAT_SPECIAL, 1, 3, true},
      {AGX_APPLE9_VIR_FLOG2, AGX_APPLE9_ENC_FLOAT_SPECIAL, 1, 3, true},
      {AGX_APPLE9_VIR_DERIVATIVE, AGX_APPLE9_ENC_DERIVATIVE, 1, 0, true},
      {AGX_APPLE9_VIR_FMIN, AGX_APPLE9_ENC_MINMAX_COMPACT, 2, 0, false},
      {AGX_APPLE9_VIR_FMAX, AGX_APPLE9_ENC_MINMAX_COMPACT, 2, 0, false},
      {AGX_APPLE9_VIR_FFLOOR, AGX_APPLE9_ENC_FLOAT_SPECIAL, 1, 3, false},
   };
   uint32_t src[] = {agx_apple9_vir_input(&p, 4), agx_apple9_vir_input(&p, 5),
                     agx_apple9_vir_input(&p, 6)};
   for (auto test : cases) {
      SCOPED_TRACE(test.op);
      agx_apple9_block_begin(&p, agx_apple9_block_create(&p));
      uint32_t value = agx_apple9_vir_emit(&p, test.op, test.encoding,
         src, test.sources, test.immediate);
      uint32_t clamp = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FSAT,
         AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED, &value, 1, 0);
      p.output = clamp;
      ASSERT_TRUE(agx_apple9_optimize_vir(&p));
      ASSERT_TRUE(agx_apple9_analyze_uses(&p));
      const auto *I = agx_apple9_definition(&p, clamp);
      EXPECT_EQ(I->op, test.fused ? test.op : AGX_APPLE9_VIR_FADD_IMM);
      EXPECT_TRUE(I->saturate);
      EXPECT_EQ(I->src[0], test.fused ? src[0] : value);
   }
}

TEST_F(Apple9Optimization, SaturationKeepsUnclampedUsersAndActiveLaneScope)
{
   auto input = agx_apple9_vir_input(&p, 4);
   for (bool mask_transition : {false, true}) {
      agx_apple9_block_begin(&p, agx_apple9_block_create(&p));
      uint32_t value = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FRCP,
         AGX_APPLE9_ENC_FLOAT_SPECIAL, &input, 1, 3);
      if (mask_transition) {
         ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p,
            AGX_APPLE9_VIR_EXEC_MASK_ELSE, AGX_APPLE9_ENC_EXEC_MASK_ELSE,
            nullptr, 0, 0));
      }
      uint32_t clamp = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FSAT,
         AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED, &value, 1, 0);
      p.output = mask_transition ? clamp : add(clamp, value);
      ASSERT_TRUE(agx_apple9_optimize_vir(&p));
      ASSERT_TRUE(agx_apple9_analyze_uses(&p));
      const auto *I = agx_apple9_definition(&p, clamp);
      EXPECT_EQ(I->op, AGX_APPLE9_VIR_FADD_IMM);
      EXPECT_TRUE(I->saturate);
      EXPECT_FALSE(agx_apple9_definition(&p, value)->saturate);
   }
}

TEST_F(Apple9Optimization, SaturationIsPartOfTheExpressionIdentity)
{
   auto input = agx_apple9_vir_input(&p, 4);
   uint32_t plain = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FRCP,
      AGX_APPLE9_ENC_FLOAT_SPECIAL, &input, 1, 3);
   uint32_t clamped = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FRCP,
      AGX_APPLE9_ENC_FLOAT_SPECIAL, &input, 1, 3);
   p.instructions[p.instruction_count - 1]->saturate = true;
   p.output = add(plain, clamped);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   const auto *I = agx_apple9_definition(&p, p.output);
   EXPECT_EQ(I->src[0], plain);
   EXPECT_EQ(I->src[1], clamped);
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

TEST(Apple9Packer, UnormPackingDefinesBothHalvesWithIndependentSources)
{
   uint8_t phys[] = {95, 64, 31, 80, 0};
   agx_apple9_vir_instr I = {};
   I.op = AGX_APPLE9_VIR_PACK_UNORM_4X8;
   I.encoding = AGX_APPLE9_ENC_PACK_UNORM_4X8;
   I.dest = 0;
   I.nr_srcs = 4;
   for (unsigned s = 0; s < 4; ++s)
      I.src[s] = s + 1;
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t expected[] = {
      0x97,0x04,0x54,0xbe,0x02,0x00,0xf9,0x50,0x44,0xc2,
      0x97,0x04,0x54,0xbf,0x02,0x40,0x01,0x50,0x44,0xc2};
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
   /* The second pair must not read a value overwritten by the first. */
   phys[3] = 95;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   phys[3] = 96;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
}

TEST(Apple9Compiler, NativeUnormPackingSurvivesOrdinaryNirLowering)
{
   for (unsigned channels : {2u, 4u}) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
         &agx_nir_options, "native_unorm_%u", channels);
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.num_ssbos = 2;
      auto input = nir_load_ssbo(&b, channels, 32, nir_imm_int(&b, 1),
         nir_imm_int(&b, 0), .access = ACCESS_NON_WRITEABLE, .align_mul = 16);
      auto value = channels == 4 ? nir_pack_unorm_4x8(&b, input)
                                  : nir_pack_unorm_2x16(&b, input);
      nir_store_ssbo(&b, value, nir_imm_int(&b, 0), nir_imm_int(&b, 0), .align_mul = 4);
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, nullptr, &reason))
         << channels << ": " << (reason ?: "");
      unsigned native = 0;
      auto *code = static_cast<const uint8_t *>(compiled.binary);
      for (unsigned i = 0; i + 9 < compiled.info.main_size; ++i)
         native += code[i] == 0x97 && code[i + 1] == 0x04 &&
                   code[i + 9] == (channels == 4 ? 0xc2 : 0x82);
      EXPECT_EQ(native, channels / 2);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, NativeHalfPackingAndHighMultiplyReachOrdinaryNir)
{
   for (unsigned operation = 0; operation < 8; ++operation) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
         &agx_nir_options, "native_convert_%u", operation);
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.num_ssbos = 2;
      auto input = nir_load_ssbo(&b, 2, 32, nir_imm_int(&b, 1),
         nir_imm_int(&b, 0), .access = ACCESS_NON_WRITEABLE, .align_mul = 8);
      auto a = nir_channel(&b, input, 0), c = nir_channel(&b, input, 1);
      nir_def *result = operation == 0 ? nir_pack_half_2x16_split(&b, a, c)
         : operation == 1 ? nir_unpack_half_2x16(&b, a)
         : operation == 2 ? nir_u2u32(&b, nir_f2f16(&b, a))
         : operation == 3 ? nir_f2f32(&b, nir_u2u16(&b, a))
         : operation == 4 ? nir_umul_high(&b, a, c) : nir_imul_high(&b, a, c);
      if (operation >= 6) {
         auto product = operation == 6 ? nir_umul_2x32_64(&b, a, c)
                                       : nir_imul_2x32_64(&b, a, c);
         result = nir_vec2(&b, nir_unpack_64_2x32_split_x(&b, product),
                               nir_unpack_64_2x32_split_y(&b, product));
      }
      nir_store_ssbo(&b, result, nir_imm_int(&b, 0), nir_imm_int(&b, 0), .align_mul = 8);
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, nullptr, &reason))
         << operation << ": " << (reason ?: "");
      /* These all fit one short native operation plus the buffer interface.
       * The old software half conversions/high product exceeded this bound. */
      EXPECT_LT(compiled.info.main_size, 180u) << operation;
      free(compiled.binary);
      ralloc_free(b.shader);
   }
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
   const uint8_t add[] = {0xf9,0xa1,0x64,0x81,0x02,0x1d,0x02,0x00,0x03,0x00};
   ASSERT_EQ(packed.length, sizeof(add));
   EXPECT_EQ(memcmp(packed.bytes, add, sizeof(add)), 0);
   I.op = AGX_APPLE9_VIR_FMUL;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   I.encoding = AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t mul[] = {0xf9,0xa1,0x65,0x81,0x02,0x1d,0x02,0x00,0x03,0x00,0x00,0x00};
   ASSERT_EQ(packed.length, sizeof(mul));
   EXPECT_EQ(memcmp(packed.bytes, mul, sizeof(mul)), 0);
   I.op = AGX_APPLE9_VIR_FMA; I.encoding = AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED;
   I.src[2] = 3; I.nr_srcs = 3;
   I.src_abs_mask = I.live_after_mask = 7; I.src_neg_mask = 5;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
   const uint8_t fma[] = {0xf9,0xa1,0x66,0x81,0x1b,0xc0,0x02,0x1d,0x00,0x00,0x03,0x00};
   ASSERT_EQ(packed.length, sizeof(fma));
   EXPECT_EQ(memcmp(packed.bytes, fma, sizeof(fma)), 0);
}

TEST(Apple9Packer, FloatSelectModifierFieldsAreIndependent)
{
   uint8_t phys[] = {95, 64, 65, 66, 67};
   for (unsigned abs = 0; abs < 16; ++abs) {
      for (unsigned neg = 0; neg < 16; neg += 2) {
         agx_apple9_vir_instr I = {};
         I.op = AGX_APPLE9_VIR_SELECT;
         I.encoding = (abs & 3) ? AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED
                               : AGX_APPLE9_ENC_SELECT_GPR_WIDE;
         I.dest = 0; I.nr_srcs = 4;
         I.src[0] = 1; I.src[1] = 2; I.src[2] = 3; I.src[3] = 4;
         I.immediate = AGX_APPLE9_SELECT_FLT;
         I.src_abs_mask = abs; I.src_neg_mask = neg;
         agx_apple9_packed_instruction packed;
         const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
         EXPECT_EQ((packed.bytes[4] >> 3) & 3, ((abs >> 2) & 1) | (neg & 4 ? 2 : 0));
         EXPECT_EQ((packed.bytes[8] >> 3) & 3, ((abs >> 3) & 1) | (neg & 8 ? 2 : 0));
         EXPECT_EQ(bool(packed.bytes[7] & 8), bool(neg & 2));
         if (abs & 3) {
            EXPECT_EQ(packed.bytes[12] & 3, abs & 3);
         }
      }
   }
}

TEST(Apple9Packer, ExtendedSelectWaitSkipsTheFourthSourceDescriptor)
{
   uint8_t phys[] = {95, 64, 65, 66, 67};
   const uint8_t low[] = {0, 1, 2, 4, 0, 0, 0};
   const uint8_t high[] = {0, 0, 0, 0, 1, 2, 4};
   agx_apple9_vir_instr I = {};
   I.op = AGX_APPLE9_VIR_SELECT;
   I.encoding = AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED;
   I.dest = 0; I.nr_srcs = 4;
   I.src[0] = 1; I.src[1] = 2; I.src[2] = 3; I.src[3] = 4;
   I.immediate = AGX_APPLE9_SELECT_FLT;
   I.src_abs_mask = 15; I.src_neg_mask = 14; I.live_after_mask = 15;
   for (unsigned slot = 0; slot <= 6; ++slot) {
      I.scoreboard_slot = (agx_apple9_scoreboard_slot)slot;
      agx_apple9_packed_instruction packed;
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
      EXPECT_EQ(packed.bytes[7] >> 5, low[slot]);
      EXPECT_EQ(packed.bytes[11] >> 5, high[slot]);
      EXPECT_EQ(packed.bytes[9], 0x86);
      EXPECT_EQ(packed.bytes[12], 3);
   }
}

TEST(Apple9Packer, ExtendedFloatDependenciesAreSplitMasks)
{
   /* A distinct pending load on each of the six slots, with stale GPR and
    * destination canaries, distinguishes these masks from the compact
    * forms' three-bit slot index. Saturation and source modifiers survive. */
   const uint8_t low[] = {0, 1, 2, 4, 0, 0, 0};
   const uint8_t high[] = {0, 0, 0, 0, 1, 2, 4};
   for (auto enc : {AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED,
         AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED,
         AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED,
         AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED,
         AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED,
         AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED}) {
      bool fma = enc == AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED ||
                 enc == AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED;
      agx_apple9_vir_instr I = {};
      I.encoding = enc;
      I.op = fma ? AGX_APPLE9_VIR_FMA
         : enc == AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED ? AGX_APPLE9_VIR_FADD_IMM
         : enc == AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED ? AGX_APPLE9_VIR_FMUL
         : AGX_APPLE9_VIR_FADD;
      I.nr_srcs = fma ? 3 : I.op == AGX_APPLE9_VIR_FADD_IMM ? 1 : 2;
      I.dest = 0; I.src[0] = 1; I.src[1] = 2; I.src[2] = 3;
      I.saturate = true;
      uint8_t phys[] = {95, 64, 80, 32};
      for (unsigned slot = 0; slot <= 6; ++slot) {
         SCOPED_TRACE(enc);
         SCOPED_TRACE(slot);
         I.scoreboard_slot = (agx_apple9_scoreboard_slot)slot;
         agx_apple9_packed_instruction packed;
         const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
         unsigned first = fma ? 7 : 5;
         EXPECT_EQ(packed.bytes[first] >> 5, low[slot]);
         EXPECT_EQ(packed.bytes[first + 2] >> 5, high[slot]);
         EXPECT_TRUE(packed.bytes[first + 2] & 2);
      }
   }
}

TEST_F(Apple9Optimization, UniformWritesEndUniformReadCseScope)
{
   auto zero = imm(0), value = agx_apple9_vir_input(&p, 4);
   auto before = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR_UNIFORM,
      AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1, AGX_APPLE9_GRAPHICS_ROOT_WORDS);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_STORE_UNIFORM,
      AGX_APPLE9_ENC_STORE_UNIFORM, &value, 1, AGX_APPLE9_GRAPHICS_ROOT_WORDS));
   auto after = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR_UNIFORM,
      AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1, AGX_APPLE9_GRAPHICS_ROOT_WORDS);
   auto result = add(before, after);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   EXPECT_EQ(agx_apple9_definition(&p, result)->src[0], before);
   EXPECT_EQ(agx_apple9_definition(&p, result)->src[1], after);
}

TEST(Apple9Compiler, OversizePreambleFallsBackToOrdinaryMain)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "large_uniform_expression");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   auto u = nir_load_ubo(&b, 3, 32, nir_imm_int(&b, 7), nir_imm_int(&b, 0),
      .align_mul = 16, .range = 16);
   auto x = nir_channel(&b, u, 0), y = nir_channel(&b, u, 1), z = nir_channel(&b, u, 2);
   for (unsigned i = 0; i < AGX_APPLE9_MAX_PREAMBLE_BYTES / 4 + 1; ++i)
      x = nir_ffma(&b, x, y, z);
   auto gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_store_ssbo(&b, nir_ixor(&b, x, gid), nir_imm_int(&b, 0),
                  nir_imul_imm(&b, gid, 4), .align_mul = 4);
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason)) << reason;
   EXPECT_EQ(compiled.info.apple9_preamble_size, 0u);
   EXPECT_EQ(profile.preamble_size, 0u);
   EXPECT_GT(compiled.info.main_size, AGX_APPLE9_MAX_PREAMBLE_BYTES);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST_F(Apple9Optimization, Uniform255SurvivesTrackingAndIntegerSourceFolding)
{
   auto zero = imm(0);
   auto uniform = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR_UNIFORM,
      AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1, 255);
   auto input = agx_apple9_vir_input(&p, 64);
   uint32_t src[] = {uniform, input};
   p.output = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   ASSERT_TRUE(agx_apple9_optimize_vir(&p));
   ASSERT_TRUE(agx_apple9_analyze_uses(&p));
   const auto *I = agx_apple9_definition(&p, p.output);
   EXPECT_EQ(I->alu_src_uniform_mask, 1);
   EXPECT_EQ(I->alu_src_value[0], 255u);
   EXPECT_EQ(I->nr_srcs, 1);
   EXPECT_EQ(I->src[0], input);
   EXPECT_EQ(agx_apple9_uses(&p, uniform), nullptr);
}

TEST(Apple9Compiler, IntegerPreambleStoresFuseWithoutAllocatingUniformGprs)
{
   for (auto op : {AGX_APPLE9_VIR_IADD, AGX_APPLE9_VIR_ISUB,
                   AGX_APPLE9_VIR_IMAD, AGX_APPLE9_VIR_IMUL_WIDE}) {
      SCOPED_TRACE(op);
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      uint32_t src[3];
      for (unsigned i = 0; i < 3; ++i)
         src[i] = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_GET_GLOBAL_ID,
            AGX_APPLE9_ENC_GET_SR, nullptr, 0, i);
      bool wide = op == AGX_APPLE9_VIR_IMUL_WIDE;
      auto value = wide ? agx_apple9_vir_emit_mul_wide(&p, src, true)
         : agx_apple9_vir_emit(&p, op,
              op == AGX_APPLE9_VIR_IMAD ? AGX_APPLE9_ENC_INT_MAD_EXTENDED
                                       : AGX_APPLE9_ENC_INT_ADD_EXTENDED,
              src, op == AGX_APPLE9_VIR_IMAD ? 3 : 2, 0);
      for (unsigned c = 0; c < (wide ? 2u : 1u); ++c) {
         uint32_t v = value + c;
         ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_STORE_UNIFORM,
            AGX_APPLE9_ENC_STORE_UNIFORM, &v, 1, (wide ? 254 : 255) + c));
      }
      ASSERT_TRUE(agx_apple9_optimize_vir(&p));
      unsigned stores = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         auto *I = p.instructions[i];
         if (I->op != AGX_APPLE9_VIR_STORE_UNIFORM) continue;
         ++stores;
         EXPECT_EQ(I->uniform_op, op);
         EXPECT_EQ(I->uniform_signed, wide);
         EXPECT_EQ(I->dest, AGX_APPLE9_VREG_INVALID);
         EXPECT_EQ(I->nr_srcs, op == AGX_APPLE9_VIR_IMAD ? 3 : 2);
         EXPECT_EQ(I->immediate, wide ? 254u : 255u);
      }
      EXPECT_EQ(stores, 1u);
      auto b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
         &agx_nir_options, "uniform_destination");
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << (reason ?: "");
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         auto *I = p.instructions[i];
         if (I->op != AGX_APPLE9_VIR_STORE_UNIFORM) continue;
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason)) << (reason ?: "");
         EXPECT_EQ(packed.bytes[4] & 3, 1);
         EXPECT_EQ((packed.bytes[3] >> 1) | ((packed.bytes[4] & 1) << 7), wide ? 254 : 255);
      }
      ralloc_free(b.shader);
      agx_apple9_vir_finish(&p);
   }
}
