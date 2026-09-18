/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_ir.h"
#include "agx_apple9_machine.h"
#include "agx_compile_apple9.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "gallium/include/pipe/p_defines.h"

TEST(Apple9Machine, PhysicalModel)
{
   EXPECT_EQ(agx_apple9_machine.gpr_count, 96u);
   EXPECT_EQ(agx_apple9_machine.half_register_count, 192u);
   EXPECT_TRUE(agx_apple9_machine.hardware_register_interlocks);
   EXPECT_FALSE(agx_apple9_machine.software_waits);
   EXPECT_TRUE(agx_apple9_machine.spilling_supported);
   EXPECT_EQ(agx_apple9_machine.occupancy_model,
             AGX_APPLE9_OCCUPANCY_UNMEASURED);
}

TEST(Apple9Machine, EveryEncodingIsDescribed)
{
   for (unsigned i = 0; i < AGX_APPLE9_ENC_COUNT; ++i) {
      const auto *info =
         agx_apple9_encoding_info(static_cast<agx_apple9_encoding>(i));
      ASSERT_NE(info, nullptr);
      EXPECT_NE(info->name, nullptr);
      EXPECT_GT(info->length, 0u) << info->name;
      EXPECT_LE(info->operand_count, AGX_APPLE9_MAX_ENCODING_OPERANDS)
         << info->name;
   }
}

TEST(Apple9Machine, RawLiteralUsesSixBitScatteredDestination)
{
   for (unsigned gpr : {0u, 15u, 16u, 31u, 32u, 47u, 48u, 63u})
      EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
         AGX_APPLE9_ENC_MOV_IMM32, AGX_APPLE9_OPERAND_DEST, gpr, 32));

   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_MOV_IMM32, AGX_APPLE9_OPERAND_DEST, 64, 32));
   const auto *dst = agx_apple9_find_operand(AGX_APPLE9_ENC_MOV_IMM32,
                                             AGX_APPLE9_OPERAND_DEST);
   ASSERT_NE(dst, nullptr);
   EXPECT_TRUE(dst->flags & AGX_APPLE9_OPERAND_SCATTERED);
   EXPECT_FALSE(dst->flags & AGX_APPLE9_OPERAND_HARD_LOW);
}

TEST(Apple9Machine, CompactBinaryAluUsesScatteredFullRegisterMap)
{
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT2_COMPACT, AGX_APPLE9_OPERAND_DEST, 95, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT2_COMPACT, AGX_APPLE9_OPERAND_SRC0, 64, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT2_COMPACT, AGX_APPLE9_OPERAND_SRC1, 95, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT2_COMPACT, AGX_APPLE9_OPERAND_DEST, 96, 32));

   const auto *extended =
      agx_apple9_encoding_info(AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED);
   EXPECT_TRUE(extended->allocator_safe);
   const auto *dst = agx_apple9_find_operand(
      AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED, AGX_APPLE9_OPERAND_DEST);
   ASSERT_NE(dst, nullptr);
   EXPECT_EQ(dst->max_index, 95u);
   EXPECT_TRUE(dst->flags & AGX_APPLE9_OPERAND_SCATTERED);

   for (auto role : {AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_OPERAND_SRC0,
                     AGX_APPLE9_OPERAND_SRC1}) {
      EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(AGX_APPLE9_ENC_MINMAX_COMPACT,
                                                  role, 95, 32));
      EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
         AGX_APPLE9_ENC_MINMAX_COMPACT, role, 96, 32));
   }

   const auto *fma_dst = agx_apple9_find_operand(AGX_APPLE9_ENC_FLOAT3_EXTENDED,
                                                 AGX_APPLE9_OPERAND_DEST);
   ASSERT_NE(fma_dst, nullptr);
   EXPECT_EQ(fma_dst->max_index, 95u);
   EXPECT_TRUE(fma_dst->flags & AGX_APPLE9_OPERAND_SCATTERED);
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT3_EXTENDED, AGX_APPLE9_OPERAND_DEST, 64, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT3_EXTENDED, AGX_APPLE9_OPERAND_SRC0, 95, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT3_EXTENDED, AGX_APPLE9_OPERAND_SRC1, 64, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT3_EXTENDED, AGX_APPLE9_OPERAND_SRC2, 95, 32));
}

TEST(Apple9Machine, FullFileBoundaries)
{
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_GET_SR, AGX_APPLE9_OPERAND_DEST, 63, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_GET_SR, AGX_APPLE9_OPERAND_DEST, 64, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_DEVICE_LOAD, AGX_APPLE9_OPERAND_DEST, 95, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_DEVICE_LOAD, AGX_APPLE9_OPERAND_DEST, 96, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_DEVICE_LOAD, AGX_APPLE9_OPERAND_DEST, 96, 32));
   EXPECT_EQ(
      agx_apple9_encoding_info(AGX_APPLE9_ENC_DEVICE_LOAD)->operand_count, 2u);
   EXPECT_NE(agx_apple9_find_operand(AGX_APPLE9_ENC_DEVICE_LOAD,
                                     AGX_APPLE9_OPERAND_INDEX),
             nullptr);
}

TEST(Apple9Machine, SpecialFunctionsReachFullRegisterFile)
{
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_OPERAND_DEST, 95, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_OPERAND_DEST, 96, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_OPERAND_SRC0, 95, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_OPERAND_SRC0, 96, 32));
}

TEST(Apple9Packer, PerspectiveMultiplyNamesCoefficientAndBoundsRegisters)
{
   uint8_t phys[] = {15, 10, 6};
   agx_apple9_vir_instr project = {
      .op = AGX_APPLE9_VIR_FMUL_PROJECT,
      .encoding = AGX_APPLE9_ENC_FLOAT2_PROJECT,
      .dest = 0,
      .dest_components = 1,
      .src = {1, 2},
      .immediate = 2,
      .nr_srcs = 2,
      .live_after_mask = 2,
      .scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE,
   };
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&project, phys, &packed, &reason));
   const uint8_t own_msl[] = {0xf9, 0x15, 0x2f, 0x8d, 0x00, 0x02, 0x00, 0x00};
   ASSERT_EQ(packed.length, sizeof(own_msl));
   EXPECT_EQ(memcmp(packed.bytes, own_msl, sizeof(own_msl)), 0);
   for (unsigned cf = 0; cf <= 32; ++cf) {
      project.immediate = cf;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&project, phys, &packed, &reason));
      EXPECT_EQ(packed.bytes[5], cf);
   }
   project.immediate = AGX_APPLE9_MAX_VARYING_COMPONENTS + 1;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&project, phys, &packed, &reason));
   project.immediate = 2;
   phys[1] = 64;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&project, phys, &packed, &reason));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT2_PROJECT, AGX_APPLE9_OPERAND_SRC0, 63, 16));
}

TEST(Apple9Packer, ReciprocalPacksHandoffLifetimeAndNativeResultHint)
{
   const uint8_t phys[] = {18, 7};
   agx_apple9_vir_instr reciprocal = {
      .op = AGX_APPLE9_VIR_FRCP,
      .encoding = AGX_APPLE9_ENC_FLOAT_SPECIAL,
      .dest = 0,
      .dest_components = 1,
      .src = {1},
      .immediate = 0x02,
      .nr_srcs = 1,
      .scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6,
   };
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&reciprocal, phys, &packed, &reason))
      << (reason ? reason : "no diagnostic");
   static const uint8_t pending_release[] = {
      0xaf, 0x00, 0x56, 0x24, 0x02, 0x1c, 0x10, 0x48, 0x20, 0x00,
   };
   ASSERT_EQ(packed.length, sizeof(pending_release));
   EXPECT_EQ(memcmp(packed.bytes, pending_release, sizeof(pending_release)), 0);

   reciprocal.immediate = 0x03;
   reciprocal.live_after_mask = 1;
   reciprocal.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&reciprocal, phys, &packed, &reason))
      << (reason ? reason : "no diagnostic");
   static const uint8_t ordinary_retain[] = {
      0xaf, 0x00, 0x54, 0x24, 0x03, 0x1c, 0x00, 0x48, 0x20, 0x00,
   };
   ASSERT_EQ(packed.length, sizeof(ordinary_retain));
   EXPECT_EQ(memcmp(packed.bytes, ordinary_retain, sizeof(ordinary_retain)), 0);

   for (auto slot :
        {AGX_APPLE9_SCOREBOARD_SLOT_4, AGX_APPLE9_SCOREBOARD_SLOT_5}) {
      reciprocal.scoreboard_slot = slot;
      ASSERT_TRUE(
         agx_apple9_pack_vir_instruction(&reciprocal, phys, &packed, &reason))
         << reason;
      const unsigned encoded_mask =
         (packed.bytes[1] >> 4) | ((packed.bytes[2] & 0x3) << 4);
      EXPECT_EQ(encoded_mask, 1u << (slot - 1));
   }
}

TEST(Apple9Packer, RawLoadTokensPackIndependentlyOfIndexAndSequenceFlags)
{
   static const struct {
      uint16_t token;
      uint8_t byte8;
      uint8_t byte9;
   } cases[] = {
      {AGX_APPLE9_DEVICE_LOAD_TOKEN_5101, 0x51, 0x01},
      {AGX_APPLE9_DEVICE_LOAD_TOKEN_1100, 0x11, 0x00},
      {AGX_APPLE9_DEVICE_LOAD_TOKEN_5100, 0x51, 0x00},
      {AGX_APPLE9_DEVICE_LOAD_TOKEN_9100, 0x91, 0x00},
      {AGX_APPLE9_DEVICE_LOAD_TOKEN_D100, 0xd1, 0x00},
      {AGX_APPLE9_DEVICE_LOAD_TOKEN_1101, 0x11, 0x01},
   };

   for (const auto &test : cases) {
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_device_load_u32_raw(
         2, 1, 3, AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR,
         AGX_APPLE9_DEVICE_LOAD_HAS_NEXT, AGX_APPLE9_SCOREBOARD_SLOT_NONE, test.token, &packed));
      static const uint8_t fixed[] = {
         0x67, 0x00, 0x54, 0x04, 0x03, 0x01, 0x20,
         0x00, 0x00, 0x00, 0x00, 0x40, 0x46, 0x00,
      };
      ASSERT_EQ(packed.length, sizeof(fixed));
      EXPECT_EQ(memcmp(packed.bytes, fixed, 8), 0);
      EXPECT_EQ(packed.bytes[8], test.byte8);
      EXPECT_EQ(packed.bytes[9], test.byte9);
      EXPECT_EQ(memcmp(packed.bytes + 10, fixed + 10, 4), 0);
   }

   agx_apple9_packed_instruction packed = {};
   EXPECT_FALSE(agx_apple9_pack_device_load_u32_raw(
      2, AGX_APPLE9_GPR_COUNT, 0, AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR, 0, AGX_APPLE9_SCOREBOARD_SLOT_NONE, AGX_APPLE9_DEVICE_LOAD_TOKEN_5101, &packed));
   EXPECT_FALSE(agx_apple9_pack_device_load_u32_raw(
      2, 1, 0, AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR, 0, AGX_APPLE9_SCOREBOARD_SLOT_NONE, 0x7100,
      &packed));
}

TEST(Apple9Vir, DeviceLoadDirectIndexIsAnSsaSource)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t load =
      agx_apple9_vir_emit_device_load(&program, 0, index, &contract);
   ASSERT_NE(load, AGX_APPLE9_VREG_INVALID);
   ASSERT_EQ(program.instruction_count, 1u);
   EXPECT_EQ(program.instructions[0]->encoding, AGX_APPLE9_ENC_DEVICE_LOAD);
   ASSERT_EQ(program.instructions[0]->nr_srcs, 1u);
   EXPECT_EQ(program.instructions[0]->src[0], index);
   EXPECT_EQ(program.instructions[0]->device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_5101);

   ASSERT_TRUE(agx_apple9_vir_set_fixed_phys(&program, index, 1));
   ASSERT_TRUE(agx_apple9_vir_set_fixed_phys(&program, load, 0));
   ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, index));
   program.output = load;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(program.instructions[0],
                                               program.phys, &packed, &reason))
      << (reason ? reason : "");
   static const uint8_t expected[] = {
      0x67, 0x00, 0x44, 0x00, 0x00, 0x01, 0x20,
      0x00, 0x51, 0x01, 0x00, 0x40, 0x46, 0x00,
   };
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);

   EXPECT_EQ(agx_apple9_vir_emit_device_load(
                &program, 0, AGX_APPLE9_VREG_INVALID, &contract),
             AGX_APPLE9_VREG_INVALID);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, DeviceLoadComputedIndexIsIndependentOfDestination)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 2);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_COMPUTED_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t load =
      agx_apple9_vir_emit_device_load(&program, 0, index, &contract);
   ASSERT_NE(load, AGX_APPLE9_VREG_INVALID);

   uint8_t phys[] = {2, 4};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(program.instructions[0], phys,
                                               &packed, &reason))
      << (reason ? reason : "");
   static const uint8_t expected[] = {
      0x67, 0x00, 0x44, 0x08, 0x00, 0x82, 0x20,
      0x00, 0x51, 0x01, 0x00, 0x40, 0x46, 0x00,
   };
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
   EXPECT_EQ(agx_apple9_vir_emit_device_load(
                &program, 0, AGX_APPLE9_VREG_INVALID, &contract),
             AGX_APPLE9_VREG_INVALID);
   EXPECT_FALSE(agx_apple9_vir_set_device_load_index_kind(
      &program, load, static_cast<agx_apple9_device_load_index_kind>(3)));
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, DeviceLoadIndexLifetimeIsFinalizedFromLiveness)
{
   for (bool asserted_last_use : {false, true}) {
      SCOPED_TRACE(asserted_last_use);
      agx_apple9_vir_program program;
      agx_apple9_vir_init(&program);
      uint32_t index = agx_apple9_vir_input(&program, 1);
      const agx_apple9_device_load_contract contract = {
         .index_kind = asserted_last_use
                          ? AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR
                          : AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
         .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
      };
      uint32_t load =
         agx_apple9_vir_emit_device_load(&program, 0, index, &contract);
      ASSERT_NE(load, AGX_APPLE9_VREG_INVALID);
      ASSERT_TRUE(agx_apple9_vir_set_fixed_phys(&program, index, 1));
      ASSERT_TRUE(agx_apple9_vir_set_fixed_phys(&program, load, 0));

      /* Make the actual lifetime the opposite of the authored assertion. */
      if (asserted_last_use) {
         ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, index));
      }
      program.output = load;
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << (reason ?: "");
      EXPECT_EQ(program.instructions[0]->device_load_index_kind,
         asserted_last_use ? AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR
                           : AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR);
      /* A stale contract introduced after finalization is still rejected. */
      program.instructions[0]->device_load_index_kind = contract.index_kind;
      EXPECT_FALSE(agx_apple9_validate_vir_allocation(&program, &reason));
      ASSERT_NE(reason, nullptr);
      EXPECT_NE(strstr(reason, "lifetime"), nullptr) << reason;
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Packer, DeviceLoadInputCompletionIsIndependentOfOutputAndSequence)
{
   const uint16_t tokens[] = {0x1100, 0x5100, 0x9100, 0xd100, 0x1101, 0x5101};
   for (unsigned input = 0; input <= 6; ++input) {
      for (unsigned output = 1; output <= 6; ++output) {
         for (bool next : {false, true}) {
            agx_apple9_packed_instruction packed = {};
            ASSERT_TRUE(agx_apple9_pack_device_load_u32_raw(
               2, 1, 3, AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
               next ? AGX_APPLE9_DEVICE_LOAD_HAS_NEXT : 0,
               (agx_apple9_scoreboard_slot)input, tokens[output - 1], &packed));
            EXPECT_EQ((packed.bytes[1] >> 4) | ((packed.bytes[2] & 3) << 4),
                      input ? 1u << (input - 1) : 0);
            EXPECT_EQ(packed.bytes[2] & 0xfc, next ? 0x54 : 0x44);
            EXPECT_EQ(packed.bytes[8], tokens[output - 1] >> 8);
            EXPECT_EQ(packed.bytes[9], tokens[output - 1] & 0xff);
         }
      }
   }
   agx_apple9_packed_instruction packed = {};
   EXPECT_FALSE(agx_apple9_pack_device_load_u32_raw(
      2, 1, 3, AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR, 0,
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO, tokens[0], &packed));
}

TEST(Apple9Packer, NativeVectorMemoryWidthsMatchValidatedEncodings)
{
   struct vector_case {
      unsigned components;
      unsigned dst;
      unsigned index;
      uint8_t flags;
      agx_apple9_scoreboard_slot incoming_slot;
      uint8_t load[14];
      uint8_t store_load[14];
      uint8_t store_alu[14];
   };
   static const vector_case cases[] = {
      {2,
       0,
       2,
       0, AGX_APPLE9_SCOREBOARD_SLOT_1,
       {0x67, 0x10, 0x44, 0x00, 0x00, 0x02, 0x20, 0x00, 0x59, 0x01, 0x00, 0x40,
        0x48, 0x00},
       {0xe7, 0x00, 0x56, 0x00, 0x01, 0x02, 0x21, 0x00, 0x19, 0x00, 0x00, 0x10,
        0x12, 0x00},
       {0xe7, 0x00, 0x54, 0x00, 0x01, 0x02, 0x21, 0x00, 0x19, 0x00, 0x00, 0x10,
        0x12, 0x00}},
      {3,
       4,
       7,
       AGX_APPLE9_DEVICE_LOAD_HAS_NEXT, AGX_APPLE9_SCOREBOARD_SLOT_NONE,
       {0x67, 0x00, 0x54, 0x08, 0x00, 0x07, 0x20, 0x00, 0x5d, 0x01, 0x00, 0x40,
        0x40, 0x00},
       {0xe7, 0x00, 0x56, 0x00, 0x01, 0x07, 0x21, 0x00, 0x1d, 0x00, 0x00, 0x10,
        0x10, 0x00},
       {0xe7, 0x00, 0x54, 0x00, 0x01, 0x07, 0x21, 0x00, 0x1d, 0x00, 0x00, 0x10,
        0x10, 0x00}},
      {4,
       0,
       4,
       0, AGX_APPLE9_SCOREBOARD_SLOT_1,
       {0x67, 0x10, 0x44, 0x00, 0x00, 0x04, 0x20, 0x00, 0x57, 0x01, 0x00, 0x40,
        0x40, 0x00},
       {0xe7, 0x00, 0x56, 0x00, 0x01, 0x04, 0x21, 0x00, 0x17, 0x00, 0x00, 0x10,
        0x10, 0x00},
       {0xe7, 0x00, 0x54, 0x00, 0x01, 0x04, 0x21, 0x00, 0x17, 0x00, 0x00, 0x10,
        0x10, 0x00}},
   };

   agx_apple9_packed_instruction packed = {};
   for (const auto &test : cases) {
      SCOPED_TRACE(test.components);
      ASSERT_TRUE(agx_apple9_pack_device_load_vector_u32_raw(
         test.dst, test.index, 0, test.components,
         AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR, test.flags, test.incoming_slot,
         AGX_APPLE9_DEVICE_LOAD_TOKEN_5101, &packed));
      EXPECT_EQ(memcmp(packed.bytes, test.load, sizeof(test.load)), 0);

      ASSERT_TRUE(agx_apple9_pack_device_store_vector_u32(
         0, test.index, 1, test.components, AGX_APPLE9_SCOREBOARD_SLOT_6, true,
         &packed));
      EXPECT_EQ(memcmp(packed.bytes, test.store_load, sizeof(test.store_load)),
                0);

      ASSERT_TRUE(agx_apple9_pack_device_store_vector_u32(
         0, test.index, 1, test.components, AGX_APPLE9_SCOREBOARD_SLOT_NONE,
         true, &packed));
      EXPECT_EQ(memcmp(packed.bytes, test.store_alu, sizeof(test.store_alu)),
                0);
   }
}

TEST(Apple9Packer, NarrowScalarMemoryMatchesOwnMslCorpus)
{
   agx_apple9_packed_instruction packed = {};

   ASSERT_TRUE(agx_apple9_pack_device_load_scalar_raw(
      1, 0, 1, 8, AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      0, AGX_APPLE9_SCOREBOARD_SLOT_1,
      AGX_APPLE9_DEVICE_LOAD_TOKEN_5101, &packed));
   static const uint8_t load_u8[] = {
      0x67, 0x10, 0x44, 0x02, 0x01, 0x00, 0x20,
      0x00, 0x61, 0x01, 0x00, 0x40, 0x42, 0x00,
   };
   EXPECT_EQ(packed.length, sizeof(load_u8));
   EXPECT_EQ(memcmp(packed.bytes, load_u8, sizeof(load_u8)), 0);

   ASSERT_TRUE(agx_apple9_pack_device_load_scalar_raw(
      1, 0, 2, 16, AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      AGX_APPLE9_DEVICE_LOAD_HAS_NEXT, AGX_APPLE9_SCOREBOARD_SLOT_1, AGX_APPLE9_DEVICE_LOAD_TOKEN_5101, &packed));
   static const uint8_t load_u16[] = {
      0x67, 0x10, 0x54, 0x02, 0x02, 0x00, 0x20,
      0x00, 0x41, 0x01, 0x00, 0x40, 0x44, 0x00,
   };
   EXPECT_EQ(memcmp(packed.bytes, load_u16, sizeof(load_u16)), 0);

   ASSERT_TRUE(agx_apple9_pack_device_store_scalar(
      0, 0, 0, 8, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
   static const uint8_t store_u8[] = {
      0xe7, 0x00, 0x54, 0x00, 0x00, 0x00, 0x21,
      0x00, 0x21, 0x00, 0x00, 0x90, 0x10, 0x00,
   };
   EXPECT_EQ(memcmp(packed.bytes, store_u8, sizeof(store_u8)), 0);

   ASSERT_TRUE(agx_apple9_pack_device_store_scalar(
      0, 0, 0, 16, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
   static const uint8_t store_u16[] = {
      0xe7, 0x00, 0x54, 0x00, 0x00, 0x00, 0x21,
      0x00, 0x01, 0x00, 0x00, 0x10, 0x11, 0x00,
   };
   EXPECT_EQ(memcmp(packed.bytes, store_u16, sizeof(store_u16)), 0);
}

TEST(Apple9Packer, DeviceStoreEncodesAllocatedDataRegister)
{
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_device_store_scalar(
      17, 9, 3, 32, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
   EXPECT_EQ(packed.bytes[2], 0x54);
   EXPECT_EQ(packed.bytes[3], 34);
   EXPECT_EQ(packed.bytes[4], 3);
   EXPECT_EQ(packed.bytes[5], 9);

   ASSERT_TRUE(agx_apple9_pack_device_store_vector_u32(
      40, 7, 2, 4, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
   EXPECT_EQ(packed.bytes[3], 80);
   EXPECT_FALSE(agx_apple9_pack_device_store_scalar(
      96, 0, 0, 16, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
}

TEST(Apple9Packer, DeviceStoreIndexLifetimeMatchesNativeAccessDescriptor)
{
   agx_apple9_packed_instruction packed = {};

   ASSERT_TRUE(agx_apple9_pack_device_store_scalar(
      17, 9, 3, 32, AGX_APPLE9_SCOREBOARD_SLOT_NONE, false, &packed));
   EXPECT_EQ(packed.bytes[6], 0x20);

   ASSERT_TRUE(agx_apple9_pack_device_store_scalar(
      17, 9, 3, 32, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
   EXPECT_EQ(packed.bytes[6], 0x21);

   ASSERT_TRUE(agx_apple9_pack_device_store_vector_u32(
      40, 7, 2, 4, AGX_APPLE9_SCOREBOARD_SLOT_NONE, false, &packed));
   EXPECT_EQ(packed.bytes[6], 0x20);
}

TEST(Apple9Packer, DeviceAtomicEncodesOperationRegistersAndReturnMode)
{
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_device_atomic(
      23, 22, 5, AGX_APPLE9_ATOMIC_ADD, false,
      AGX_APPLE9_SCOREBOARD_SLOT_NONE, &packed));
   static const uint8_t add[] = {
      0x67, 0x01, 0x54, 0x00, 0x00, 0x05, 0x8b,
      0x8b, 0x00, 0x02, 0x00, 0x00, 0x60, 0x02,
   };
   ASSERT_EQ(packed.length, sizeof(add));
   EXPECT_EQ(memcmp(packed.bytes, add, sizeof(add)), 0);

   ASSERT_TRUE(agx_apple9_pack_device_atomic(
      25, 23, 5, AGX_APPLE9_ATOMIC_CMPXCHG, false,
      AGX_APPLE9_SCOREBOARD_SLOT_NONE, &packed));
   static const uint8_t cmpxchg[] = {
      0x67, 0x01, 0x54, 0x00, 0x00, 0x85, 0x8b,
      0x8c, 0x00, 0x02, 0x00, 0x00, 0x64, 0x02,
   };
   ASSERT_EQ(packed.length, sizeof(cmpxchg));
   EXPECT_EQ(memcmp(packed.bytes, cmpxchg, sizeof(cmpxchg)), 0);

   ASSERT_TRUE(agx_apple9_pack_device_atomic(
      5, 4, 9, AGX_APPLE9_ATOMIC_XOR, true,
      AGX_APPLE9_SCOREBOARD_SLOT_NONE, &packed));
   static const uint8_t discard[] = {
      0x67, 0x01, 0x54, 0x00, 0x00, 0x09, 0x82,
      0x82, 0x00, 0x40, 0x00, 0x00, 0x7e, 0x02,
   };
   ASSERT_EQ(packed.length, sizeof(discard));
   EXPECT_EQ(memcmp(packed.bytes, discard, sizeof(discard)), 0);

   ASSERT_TRUE(agx_apple9_pack_device_atomic(
      5, 4, 9, AGX_APPLE9_ATOMIC_XOR, true,
      AGX_APPLE9_SCOREBOARD_SLOT_6, &packed));
   EXPECT_EQ(packed.bytes[1], 0x01);
   EXPECT_EQ(packed.bytes[2], 0x56);

   EXPECT_FALSE(agx_apple9_pack_device_atomic(
      AGX_APPLE9_GPR_COUNT, 0, 0, AGX_APPLE9_ATOMIC_ADD, true,
      AGX_APPLE9_SCOREBOARD_SLOT_NONE, &packed));
   EXPECT_FALSE(agx_apple9_pack_device_atomic(
      0, 0, 0, AGX_APPLE9_ATOMIC_ADD, false,
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO, &packed));
}

TEST(Apple9Packer, DeviceAtomicResultEncodesFullDestinationAndAllSlots)
{
   static const uint8_t publication_code[] = {
      0, 2, 4, 3, 5, 6, 1,
   };

   for (unsigned destination = 0; destination < 96; ++destination) {
      for (unsigned slot = AGX_APPLE9_SCOREBOARD_SLOT_1;
           slot <= AGX_APPLE9_SCOREBOARD_SLOT_6; ++slot) {
         agx_apple9_vir_instr instruction = {};
         instruction.op = AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT;
         instruction.encoding = AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT;
         instruction.dest = AGX_APPLE9_VREG_INVALID;
         instruction.src[0] = 0;
         instruction.nr_srcs = 1;
         instruction.producer_scoreboard_slot =
            static_cast<agx_apple9_scoreboard_slot>(slot);
         const uint8_t phys[] = {static_cast<uint8_t>(destination)};
         agx_apple9_packed_instruction packed = {};
         const char *reason = nullptr;

         ASSERT_TRUE(agx_apple9_pack_vir_instruction(
            &instruction, phys, &packed, &reason))
            << "destination=" << destination << " slot=" << slot << " "
            << (reason ? reason : "");
         ASSERT_EQ(packed.length, 8u);
         EXPECT_EQ(packed.bytes[0],
                   ((destination & 0xf) << 4) | 0x0c);
         EXPECT_EQ(packed.bytes[1], 0x80);
         EXPECT_EQ(packed.bytes[2], 0x09 | (((destination >> 4) & 3) << 6));
         EXPECT_EQ(packed.bytes[5], publication_code[slot] << 5);
         EXPECT_EQ(packed.bytes[6], 0u);
         EXPECT_EQ(packed.bytes[7], (destination >> 6) << 4);
      }
   }

   agx_apple9_vir_instr invalid = {};
   invalid.op = AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT;
   invalid.encoding = AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT;
   invalid.dest = AGX_APPLE9_VREG_INVALID;
   invalid.src[0] = 0;
   invalid.nr_srcs = 1;
   invalid.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
   const uint8_t phys[] = {96};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   EXPECT_FALSE(
      agx_apple9_pack_vir_instruction(&invalid, phys, &packed, &reason));
}

TEST(Apple9Packer, GlobalInvocationIdAxesMatchNativeSelectors)
{
   static const uint8_t expected[3][4] = {
      {0x0c, 0xa0, 0x10, 0x06},
      {0x0c, 0xa1, 0x10, 0x06},
      {0x0c, 0xa2, 0x10, 0x06},
   };
   agx_apple9_packed_instruction packed;
   for (unsigned component = 0; component < 3; ++component) {
      ASSERT_TRUE(agx_apple9_pack_get_global_id(0, component, &packed));
      ASSERT_EQ(packed.length, sizeof(expected[component]));
      EXPECT_EQ(memcmp(packed.bytes, expected[component], packed.length), 0);
   }
   EXPECT_FALSE(agx_apple9_pack_get_global_id(0, 3, &packed));
}

TEST(Apple9Packer, NarrowSystemValuesUseNativeZeroExtendPair)
{
   static const uint8_t expected[] = {
      0x14, 0xa4, 0x10, 0x06, 0x13, 0x00, 0x00, 0x01,
   };
   agx_apple9_packed_instruction packed;
   ASSERT_TRUE(agx_apple9_pack_get_sr_zext16(1, 0xa4, &packed));
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
   EXPECT_FALSE(agx_apple9_pack_get_sr_zext16(64, 0xa4, &packed));

   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_GET_SR_ZEXT16, AGX_APPLE9_OPERAND_DEST, 63, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_GET_SR_ZEXT16, AGX_APPLE9_OPERAND_DEST, 64, 32));
}

TEST(Apple9Packer, ScalarConversionsAndArithmeticShiftMatchT8132Forms)
{
   uint8_t phys[2] = {16, 17};
   agx_apple9_vir_instr instruction = {};
   instruction.dest = 0;
   instruction.src[0] = 1;
   instruction.nr_srcs = 1;
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;

   instruction.op = AGX_APPLE9_VIR_I2F32;
   instruction.encoding = AGX_APPLE9_ENC_SINT_TO_FLOAT;
   static const uint8_t i2f[] = {0xa7, 0x07, 0x54, 0x20,
                                 0x03, 0x44, 0xac, 0x60};
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&instruction, phys, &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, i2f, sizeof(i2f)), 0);

   instruction.op = AGX_APPLE9_VIR_U2F32;
   instruction.encoding = AGX_APPLE9_ENC_UINT_TO_FLOAT;
   static const uint8_t u2f[] = {0xa7, 0x07, 0x54, 0x20,
                                 0x03, 0x44, 0xac, 0x20};
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&instruction, phys, &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, u2f, sizeof(u2f)), 0);

   instruction.live_after_mask = 1;
   static const uint8_t retained_u2f[] = {0xa7, 0x07, 0x54, 0x20,
                                          0x03, 0x44, 0x8c, 0x20};
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&instruction, phys, &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, retained_u2f, sizeof(retained_u2f)), 0);

   instruction.op = AGX_APPLE9_VIR_I2F32;
   instruction.encoding = AGX_APPLE9_ENC_SINT_TO_FLOAT;
   static const uint8_t retained_i2f[] = {0xa7, 0x07, 0x54, 0x20,
                                          0x03, 0x44, 0x8c, 0x60};
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&instruction, phys, &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, retained_i2f, sizeof(retained_i2f)), 0);
   instruction.live_after_mask = 0;

   instruction.op = AGX_APPLE9_VIR_F2I32;
   instruction.encoding = AGX_APPLE9_ENC_FLOAT_TO_SINT;
   static const uint8_t f2i[] = {0x27, 0x07, 0x54, 0x20, 0x03,
                                 0x44, 0xb4, 0x48, 0x03, 0x00};
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&instruction, phys, &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, f2i, sizeof(f2i)), 0);

   instruction.op = AGX_APPLE9_VIR_F2U32;
   instruction.encoding = AGX_APPLE9_ENC_FLOAT_TO_UINT;
   static const uint8_t f2u[] = {0x27, 0x07, 0x54, 0x20, 0x03,
                                 0x44, 0xb4, 0x08, 0x02, 0x00};
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&instruction, phys, &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, f2u, sizeof(f2u)), 0);

   instruction.op = AGX_APPLE9_VIR_ISHR;
   instruction.encoding = AGX_APPLE9_ENC_SHIFT_EXTENDED;
   instruction.immediate = 7;
   uint8_t shift_phys[2] = {12, 13};
   static const uint8_t ishr[] = {0xa7, 0x01, 0x54, 0x18, 0x02,
                                  0x34, 0x1c, 0x78, 0x62, 0x00};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&instruction, shift_phys,
                                               &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, ishr, sizeof(ishr)), 0);

   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_SHIFT_EXTENDED, AGX_APPLE9_OPERAND_DEST, 95, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_SHIFT_EXTENDED, AGX_APPLE9_OPERAND_DEST, 96, 32));
}

TEST(Apple9Machine, AllocatorSafeExtendedIntegerForms)
{
   EXPECT_TRUE(
      agx_apple9_encoding_info(AGX_APPLE9_ENC_INT_ADD_EXTENDED)->allocator_safe);
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_INT_ADD_EXTENDED, AGX_APPLE9_OPERAND_DEST, 95, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_INT_ADD_EXTENDED, AGX_APPLE9_OPERAND_SRC0, 95, 32));
   EXPECT_TRUE(
      agx_apple9_encoding_info(AGX_APPLE9_ENC_INT_MAD_EXTENDED)->allocator_safe);
   for (auto role : {AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_OPERAND_SRC0,
                     AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_OPERAND_SRC2}) {
      EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
         AGX_APPLE9_ENC_INT_MAD_EXTENDED, role, 95, 32));
   }
   EXPECT_TRUE(
      agx_apple9_encoding_info(AGX_APPLE9_ENC_MINMAX_COMPACT)->allocator_safe);
   for (auto role : {AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_OPERAND_SRC0,
                     AGX_APPLE9_OPERAND_SRC1}) {
      EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(AGX_APPLE9_ENC_MINMAX_COMPACT,
                                                  role, 95, 32));
      EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
         AGX_APPLE9_ENC_MINMAX_COMPACT, role, 96, 32));
   }
}

TEST(Apple9Machine, WideSelectReachesFullRegisterFile)
{
   const auto encoding = AGX_APPLE9_ENC_SELECT_GPR_WIDE;
   EXPECT_TRUE(agx_apple9_encoding_info(encoding)->allocator_safe);
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      encoding, AGX_APPLE9_OPERAND_DEST, 95, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      encoding, AGX_APPLE9_OPERAND_DEST, 96, 32));

   for (auto role : {AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_OPERAND_SRC1,
                     AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_OPERAND_SRC3}) {
      EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(encoding, role, 95, 32));
      EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(encoding, role, 96, 32));
   }
}

TEST(Apple9Machine, LogicFormsEncodeMeasuredRegisterBanks)
{
   const auto extended = AGX_APPLE9_ENC_LOGIC_EXTENDED;
   EXPECT_TRUE(agx_apple9_encoding_info(extended)->allocator_safe);
   for (auto role : {AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_OPERAND_SRC0,
                     AGX_APPLE9_OPERAND_SRC1}) {
      EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(extended, role, 95, 32));
   }

   const unsigned two_high[] = {95, 64, 2};
   const unsigned both_sources_high[] = {0, 64, 65};
   const unsigned all_high[] = {95, 64, 65};
   EXPECT_TRUE(
      agx_apple9_encoding_accepts_gpr_tuple(extended, two_high, 3, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr_tuple(extended,
                                                     both_sources_high, 3, 32));
   EXPECT_TRUE(
      agx_apple9_encoding_accepts_gpr_tuple(extended, all_high, 3, 32));
}

TEST(Apple9Machine, StoreDataAndIndexAreAllocatable)
{
   const auto *data = agx_apple9_find_operand(AGX_APPLE9_ENC_DEVICE_STORE,
                                              AGX_APPLE9_OPERAND_STORE_DATA);
   ASSERT_NE(data, nullptr);
   EXPECT_EQ(data->files, AGX_APPLE9_FILE_GPR);
   EXPECT_TRUE(data->flags & AGX_APPLE9_OPERAND_ALLOCATABLE);
   EXPECT_EQ(data->max_index, 95);
   EXPECT_TRUE(
      agx_apple9_encoding_info(AGX_APPLE9_ENC_DEVICE_STORE)->allocator_safe);
}

TEST(Apple9Machine, DependencyLayoutsAreEncodingProperties)
{
   static const struct {
      agx_apple9_encoding encoding;
      agx_apple9_dependency_layout layout;
   } cases[] = {
      {AGX_APPLE9_ENC_FLOAT2_COMPACT, AGX_APPLE9_DEPENDENCY_INDEX_45_47},
      {AGX_APPLE9_ENC_FLOAT3_EXTENDED, AGX_APPLE9_DEPENDENCY_INDEX_61_63},
      {AGX_APPLE9_ENC_INT_ADD_EXTENDED, AGX_APPLE9_DEPENDENCY_MASK_12_17},
      {AGX_APPLE9_ENC_INT_MAD_EXTENDED, AGX_APPLE9_DEPENDENCY_MASK_12_17},
      {AGX_APPLE9_ENC_UINT_TO_FLOAT, AGX_APPLE9_DEPENDENCY_MASK_12_17},
      {AGX_APPLE9_ENC_FLOAT_TO_UINT, AGX_APPLE9_DEPENDENCY_MASK_12_17},
      {AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_DEPENDENCY_MASK_12_17},
      {AGX_APPLE9_ENC_SHIFT_EXTENDED, AGX_APPLE9_DEPENDENCY_MASK_12_17},
      {AGX_APPLE9_ENC_DEVICE_STORE, AGX_APPLE9_DEPENDENCY_MASK_12_17},
      {AGX_APPLE9_ENC_LOGIC_EXTENDED, AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63},
      {AGX_APPLE9_ENC_DEVICE_LOAD, AGX_APPLE9_DEPENDENCY_MASK_12_17},
      {AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT, AGX_APPLE9_DEPENDENCY_MASK_12_17},
   };

   for (const auto &test : cases)
      EXPECT_EQ(agx_apple9_encoding_info(test.encoding)->dependency_layout,
                test.layout);
}

TEST(Apple9Packer, RegisterFormsMatchValidatedProbeTemplates)
{
   uint8_t phys[] = {64, 2, 95, 5};
   agx_apple9_vir_instr add = {
      .op = AGX_APPLE9_VIR_IADD,
      .encoding = AGX_APPLE9_ENC_INT_ADD_EXTENDED,
      .dest = 0,
      .src = {1, 2},
      .nr_srcs = 2,
   };
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&add, phys, &packed, &reason))
      << reason;
   ASSERT_EQ(packed.length, 10u);

   /* Same all-sources-dead envelope as EXP-M4-16's hardware probe. */
   uint8_t expected[] = {0x9f, 0x01, 0x54, 0x00, 0x02,
                         0x00, 0x00, 0xa8, 0x17, 0x05};
   uint64_t word = 0;
   memcpy(&word, expected, sizeof(word));
   word &= ~(((1ull << 7) - 1) << 25);
   word |= 64ull << 25;
   word &= ~(((1ull << 7) - 1) << 42);
   word |= 2ull << 42;
   word &= ~(((1ull << 7) - 1) << 51);
   word |= 95ull << 51;
   memcpy(expected, &word, sizeof(word));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);

   uint8_t select_phys[] = {15, 95, 64, 63, 94};
   agx_apple9_vir_instr select = {
      .op = AGX_APPLE9_VIR_SELECT,
      .encoding = AGX_APPLE9_ENC_SELECT_GPR_WIDE,
      .dest = 0,
      .src = {1, 2, 3, 4},
      .immediate = AGX_APPLE9_SELECT_ULT,
      .nr_srcs = 4,
      .scoreboard_slot = 6,
   };
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&select, select_phys, &packed, &reason))
      << reason;
   static const uint8_t expected_select[] = {
      0xf2, 0x3f, 0x1f, 0x01, 0x82, 0x7e, 0x05, 0xc5, 0xc0, 0x3c,
   };
   ASSERT_EQ(packed.length, sizeof(expected_select));
   for (unsigned i = 0; i < sizeof(expected_select); ++i)
      EXPECT_EQ(packed.bytes[i], expected_select[i]) << "byte=" << i;
}

TEST(Apple9Packer, CompactMoveImmediateStopsAtSevenBitBoundary)
{
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_mov_imm(15, 0x7f, &packed));
   static const uint8_t expected[] = {0xfc, 0x7f};
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
   EXPECT_FALSE(agx_apple9_pack_mov_imm(15, 0x80, &packed));
   EXPECT_FALSE(agx_apple9_pack_mov_imm(0, 0xff, &packed));
}

TEST(Apple9Packer, RawMoveImmediatePacksSixBitDestination)
{
   static const struct {
      unsigned dst;
      uint32_t value;
      uint8_t bytes[8];
   } cases[] = {
      {0, 0x12345678, {0x0c, 0xf8, 0x02, 0x12, 0x18, 0x08, 0xa2, 0x01}},
      {18, 0x3f800000, {0x2c, 0x80, 0x42, 0x3e, 0x00, 0x00, 0x00, 0x0c}},
      {34, 0x01020304, {0x2c, 0x84, 0x82, 0x00, 0x0c, 0x00, 0x10, 0x08}},
      {63, 0xffffffff, {0xfc, 0xff, 0xc2, 0xfe, 0x1e, 0x0c, 0xff, 0x0f}},
   };

   for (const auto &test : cases) {
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_mov_imm32(test.dst, test.value, &packed));
      ASSERT_EQ(packed.length, sizeof(test.bytes));
      EXPECT_EQ(memcmp(packed.bytes, test.bytes, sizeof(test.bytes)), 0);
   }

   agx_apple9_packed_instruction packed = {};
   EXPECT_FALSE(agx_apple9_pack_mov_imm32(64, 0x12345678, &packed));
}

TEST(Apple9Packer, ExtendedLogicCarriesSourceLiveness)
{
   uint8_t phys[] = {4, 2, 3};
   agx_apple9_vir_instr logic = {
      .op = AGX_APPLE9_VIR_IXOR,
      .encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED,
      .dest = 0,
      .src = {1, 2},
      .nr_srcs = 2,
   };

   static const struct {
      uint8_t mask;
      uint8_t bytes[10];
   } cases[] = {
      {0x0, {0x4b, 0x05, 0x1e, 0x07, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00}},
      {0x1, {0x4b, 0x85, 0x16, 0x07, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00}},
      {0x2, {0x4b, 0x05, 0x2e, 0x87, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00}},
      {0x3, {0x4b, 0x85, 0x26, 0x87, 0x02, 0x08, 0x00, 0x00, 0x00, 0x00}},
   };

   const char *reason = nullptr;
   for (const auto &test : cases) {
      logic.live_after_mask = test.mask;
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(
         agx_apple9_pack_vir_instruction(&logic, phys, &packed, &reason))
         << reason;
      ASSERT_EQ(packed.length, sizeof(test.bytes));
      EXPECT_EQ(memcmp(packed.bytes, test.bytes, sizeof(test.bytes)), 0)
         << "live_after_mask=" << unsigned(test.mask);
   }
}

TEST(Apple9Packer, ExtendedLogicUsesOneHotPendingSlotMask)
{
   uint8_t phys[] = {4, 2, 3};
   const char *reason = nullptr;
   for (auto op :
        {AGX_APPLE9_VIR_IAND, AGX_APPLE9_VIR_IOR, AGX_APPLE9_VIR_IXOR}) {
      agx_apple9_vir_instr logic = {
         .op = op,
         .encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED,
         .dest = 0,
         .src = {1, 2},
         .nr_srcs = 2,
      };

      for (unsigned slot = 0; slot <= 6; ++slot) {
         logic.scoreboard_slot = slot;
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(
            agx_apple9_pack_vir_instruction(&logic, phys, &packed, &reason))
            << reason;
         ASSERT_EQ(packed.length, 10u);
         const unsigned mask = ((packed.bytes[5] >> 5) & 0x7) |
                               (((packed.bytes[7] >> 5) & 0x7) << 3);
         EXPECT_EQ(mask, slot == 0 ? 0u : 1u << (slot - 1))
            << "op=" << unsigned(op) << " slot=" << slot;
      }
   }
}

TEST(Apple9Packer, IntegerFamilyUsesSharedOneHotDependencyLayout)
{
   uint8_t phys[] = {4, 2, 3, 5};
   agx_apple9_vir_instr instructions[] = {
      {
         .op = AGX_APPLE9_VIR_IADD,
         .encoding = AGX_APPLE9_ENC_INT_ADD_EXTENDED,
         .dest = 0,
         .src = {1, 2},
         .nr_srcs = 2,
      },
      {
         .op = AGX_APPLE9_VIR_IMAD,
         .encoding = AGX_APPLE9_ENC_INT_MAD_EXTENDED,
         .dest = 0,
         .src = {1, 2, 3},
         .nr_srcs = 3,
      },
      {
         .op = AGX_APPLE9_VIR_U2F32,
         .encoding = AGX_APPLE9_ENC_UINT_TO_FLOAT,
         .dest = 0,
         .src = {1},
         .nr_srcs = 1,
      },
      {
         .op = AGX_APPLE9_VIR_F2U32,
         .encoding = AGX_APPLE9_ENC_FLOAT_TO_UINT,
         .dest = 0,
         .src = {1},
         .nr_srcs = 1,
      },
      {
         .op = AGX_APPLE9_VIR_FRCP,
         .encoding = AGX_APPLE9_ENC_FLOAT_SPECIAL,
         .dest = 0,
         .src = {1},
         .immediate = 0x02,
         .nr_srcs = 1,
      },
      {
         .op = AGX_APPLE9_VIR_ISHR,
         .encoding = AGX_APPLE9_ENC_SHIFT_EXTENDED,
         .dest = 0,
         .src = {1},
         .immediate = 3,
         .nr_srcs = 1,
      },
   };

   const char *reason = nullptr;
   for (auto &instruction : instructions) {
      for (unsigned slot = 0; slot <= 6; ++slot) {
         instruction.scoreboard_slot = slot;
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&instruction, phys,
                                                     &packed, &reason))
            << "op=" << unsigned(instruction.op) << " slot=" << slot << " "
            << (reason ? reason : "");
         const unsigned mask =
            (packed.bytes[1] >> 4) | ((packed.bytes[2] & 0x3) << 4);
         EXPECT_EQ(mask, slot == 0 ? 0u : 1u << (slot - 1))
            << "op=" << unsigned(instruction.op) << " slot=" << slot;
      }
   }

   for (unsigned slot = 0; slot <= 6; ++slot) {
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_device_store_scalar(
         4, 2, 0, 32, (agx_apple9_scoreboard_slot)slot, true, &packed));
      const unsigned mask =
         (packed.bytes[1] >> 4) | ((packed.bytes[2] & 0x3) << 4);
      EXPECT_EQ(mask, slot == 0 ? 0u : 1u << (slot - 1))
         << "store slot=" << slot;
   }
}

TEST(Apple9Packer, IntegerMinmaxPreservesNativeAluState)
{
   uint8_t phys[] = {4, 2, 3};
   agx_apple9_vir_instr minmax = {
      .op = AGX_APPLE9_VIR_UMIN,
      .encoding = AGX_APPLE9_ENC_MINMAX_COMPACT,
      .dest = 0,
      .src = {1, 2},
      .nr_srcs = 2,
   };

   static const struct {
      uint8_t mask;
      uint8_t bytes[6];
   } cases[] = {
      {0x0, {0x42, 0x05, 0x3e, 0x07, 0x05, 0x00}},
      {0x1, {0x42, 0x85, 0x36, 0x07, 0x05, 0x00}},
      {0x2, {0x42, 0x05, 0x2e, 0x87, 0x05, 0x00}},
      {0x3, {0x42, 0x85, 0x26, 0x87, 0x05, 0x00}},
   };

   const char *reason = nullptr;
   for (const auto &test : cases) {
      minmax.live_after_mask = test.mask;
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(
         agx_apple9_pack_vir_instruction(&minmax, phys, &packed, &reason))
         << reason;
      ASSERT_EQ(packed.length, sizeof(test.bytes));
      EXPECT_EQ(memcmp(packed.bytes, test.bytes, sizeof(test.bytes)), 0)
         << "live_after_mask=" << unsigned(test.mask);
   }
}

TEST(Apple9Packer, CompactFloatCarriesNativeStateAndRelease)
{
   uint8_t phys[] = {4, 2, 3};
   agx_apple9_vir_instr add = {
      .op = AGX_APPLE9_VIR_FADD,
      .encoding = AGX_APPLE9_ENC_FLOAT2_COMPACT,
      .dest = 0,
      .src = {1, 2},
      .nr_srcs = 2,
   };
   agx_apple9_vir_instr minimum = add;
   minimum.op = AGX_APPLE9_VIR_FMIN;
   minimum.encoding = AGX_APPLE9_ENC_MINMAX_COMPACT;

   static const struct {
      uint8_t mask;
      uint8_t add[6];
      uint8_t minimum[6];
   } cases[] = {
      {0x0,
       {0x49, 0x05, 0x3c, 0x07, 0x00, 0x00},
       {0x42, 0x05, 0x3e, 0x07, 0x01, 0x00}},
      {0x1,
       {0x49, 0x85, 0x34, 0x07, 0x00, 0x00},
       {0x42, 0x85, 0x36, 0x07, 0x01, 0x00}},
      {0x2,
       {0x49, 0x05, 0x2c, 0x87, 0x00, 0x00},
       {0x42, 0x05, 0x2e, 0x87, 0x01, 0x00}},
      {0x3,
       {0x49, 0x85, 0x24, 0x87, 0x00, 0x00},
       {0x42, 0x85, 0x26, 0x87, 0x01, 0x00}},
   };

   const char *reason = nullptr;
   for (const auto &test : cases) {
      add.live_after_mask = test.mask;
      minimum.live_after_mask = test.mask;
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&add, phys, &packed, &reason))
         << reason;
      ASSERT_EQ(packed.length, sizeof(test.add));
      EXPECT_EQ(memcmp(packed.bytes, test.add, sizeof(test.add)), 0)
         << "fadd live_after_mask=" << unsigned(test.mask);

      ASSERT_TRUE(
         agx_apple9_pack_vir_instruction(&minimum, phys, &packed, &reason))
         << reason;
      ASSERT_EQ(packed.length, sizeof(test.minimum));
      EXPECT_EQ(memcmp(packed.bytes, test.minimum, sizeof(test.minimum)), 0)
         << "fmin live_after_mask=" << unsigned(test.mask);
   }

   add.live_after_mask = minimum.live_after_mask = 0x3;
   add.scoreboard_slot = minimum.scoreboard_slot = 6;
   static const uint8_t load_add[] = {
      0x49, 0x85, 0x24, 0x87, 0x00, 0xc0,
   };
   static const uint8_t load_minimum[] = {
      0x42, 0x85, 0x26, 0x87, 0x01, 0xc0,
   };
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&add, phys, &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, load_add, sizeof(load_add)), 0);
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&minimum, phys, &packed, &reason))
      << reason;
   EXPECT_EQ(memcmp(packed.bytes, load_minimum, sizeof(load_minimum)), 0);
}

TEST(Apple9Packer, CompactBinaryAluPacksAllScatteredRegisterBits)
{
   uint8_t phys[] = {95, 64, 79};
   agx_apple9_vir_instr add = {
      .op = AGX_APPLE9_VIR_FADD,
      .encoding = AGX_APPLE9_ENC_FLOAT2_COMPACT,
      .dest = 0,
      .src = {1, 2},
      .nr_srcs = 2,
   };
   agx_apple9_vir_instr minimum = add;
   minimum.op = AGX_APPLE9_VIR_UMIN;
   minimum.encoding = AGX_APPLE9_ENC_MINMAX_COMPACT;

   static const uint8_t expected_add[] = {
      0xf9, 0x01, 0x7c, 0x1f, 0x00, 0x15,
   };
   static const uint8_t expected_minimum[] = {
      0xf2, 0x01, 0x7e, 0x1f, 0x05, 0x15,
   };

   const char *reason = nullptr;
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&add, phys, &packed, &reason))
      << reason;
   ASSERT_EQ(packed.length, sizeof(expected_add));
   EXPECT_EQ(memcmp(packed.bytes, expected_add, sizeof(expected_add)), 0);

   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&minimum, phys, &packed, &reason))
      << reason;
   ASSERT_EQ(packed.length, sizeof(expected_minimum));
   EXPECT_EQ(memcmp(packed.bytes, expected_minimum, sizeof(expected_minimum)),
             0);
}

TEST(Apple9Packer, FmaCarriesNativeStateAndRelease)
{
   uint8_t phys[] = {4, 2, 3, 5};
   agx_apple9_vir_instr fma = {
      .op = AGX_APPLE9_VIR_FMA,
      .encoding = AGX_APPLE9_ENC_FLOAT3_EXTENDED,
      .dest = 0,
      .src = {1, 2, 3},
      .nr_srcs = 3,
   };

   static const struct {
      uint8_t mask;
      uint8_t bytes[8];
   } cases[] = {
      {0x0, {0x49, 0x05, 0x3e, 0x07, 0x81, 0x0a, 0x02, 0x00}},
      {0x1, {0x49, 0x85, 0x36, 0x07, 0x81, 0x0a, 0x02, 0x00}},
      {0x2, {0x49, 0x05, 0x2e, 0x87, 0x81, 0x0a, 0x02, 0x00}},
      {0x4, {0x49, 0x05, 0x3e, 0x07, 0x01, 0x8a, 0x02, 0x00}},
      {0x7, {0x49, 0x85, 0x26, 0x87, 0x01, 0x8a, 0x02, 0x00}},
   };

   const char *reason = nullptr;
   for (const auto &test : cases) {
      fma.live_after_mask = test.mask;
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&fma, phys, &packed, &reason))
         << reason;
      ASSERT_EQ(packed.length, sizeof(test.bytes));
      EXPECT_EQ(memcmp(packed.bytes, test.bytes, sizeof(test.bytes)), 0)
         << "live_after_mask=" << unsigned(test.mask);
   }

   fma.live_after_mask = 0;
   fma.scoreboard_slot = 6;
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&fma, phys, &packed, &reason))
      << reason;
   static const uint8_t load_fma[] = {
      0x49, 0x05, 0x3e, 0x07, 0x81, 0x0a, 0x02, 0xc0,
   };
   EXPECT_EQ(memcmp(packed.bytes, load_fma, sizeof(load_fma)), 0);
}

TEST(Apple9Packer, ImmediateFloatCarriesScoreboardSlot)
{
   uint8_t phys[] = {2, 2};
   agx_apple9_vir_instr add = {
      .op = AGX_APPLE9_VIR_FADD_IMM,
      .encoding = AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT,
      .dest = 0,
      .src = {1},
      .immediate = 0x40000000u,
      .nr_srcs = 1,
   };
   static const struct {
      uint8_t slot;
      uint8_t bytes[6];
   } cases[] = {
      {0, {0x29, 0xc1, 0x34, 0x05, 0x80, 0x00}},
      {1, {0x29, 0xc1, 0x34, 0x05, 0x80, 0x20}},
      {2, {0x29, 0xc1, 0x34, 0x05, 0x80, 0x40}},
      {3, {0x29, 0xc1, 0x34, 0x05, 0x80, 0x60}},
      {4, {0x29, 0xc1, 0x34, 0x05, 0x80, 0x80}},
      {5, {0x29, 0xc1, 0x34, 0x05, 0x80, 0xa0}},
      {6, {0x29, 0xc1, 0x34, 0x05, 0x80, 0xc0}},
   };

   const char *reason = nullptr;
   for (const auto &test : cases) {
      add.scoreboard_slot = test.slot;
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&add, phys, &packed, &reason))
         << reason;
      ASSERT_EQ(packed.length, sizeof(test.bytes));
      EXPECT_EQ(memcmp(packed.bytes, test.bytes, sizeof(test.bytes)), 0)
         << "slot=" << unsigned(test.slot);
   }

   add.scoreboard_slot = 7;
   agx_apple9_packed_instruction packed = {};
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&add, phys, &packed, &reason));
   ASSERT_NE(reason, nullptr);
   EXPECT_NE(strstr(reason, "dependency"), nullptr);
}

TEST(Apple9Vir, ScalarLoadAutoStartsAtSlot6)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t ordinary = agx_apple9_vir_input(&program, 2);
   uint32_t load =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                          AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, load, 0,
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   uint32_t sources[] = {load, ordinary};
   agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                       AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[0]->device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_5101);
   EXPECT_EQ(program.instructions[1]->scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, IndependentPendingGroupsAdvanceAndReuseSlots)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t ordinary = agx_apple9_vir_input(&program, 2);
   uint32_t loads[4];
   for (unsigned i = 0; i < 3; ++i) {
      loads[i] = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                     AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, i);
      ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
         &program, loads[i], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   }

   uint32_t sources0[] = {loads[0], ordinary};
   agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                       AGX_APPLE9_ENC_FLOAT2_COMPACT, sources0, 2, 0);
   loads[3] = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                  AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 3);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, loads[3], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   for (unsigned i = 1; i < 4; ++i) {
      uint32_t sources[] = {loads[i], ordinary};
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                          AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
   }

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   const uint8_t expected[] = {6, 1, 2, 6};
   unsigned seen = 0;
   for (unsigned i = 0; i < program.instruction_count; ++i) {
      if (program.instructions[i]->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         EXPECT_EQ(program.instructions[i]->producer_scoreboard_slot,
                   expected[seen++]);
      }
   }
   EXPECT_EQ(seen, 4u);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, ScalarLoadPreferenceUsesFirstFreeSlotAcrossGaps)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t ordinary = agx_apple9_vir_input(&program, 2);
   uint32_t loads[7];

   for (unsigned i = 0; i < 6; ++i) {
      loads[i] = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                     AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, i);
      ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
         &program, loads[i], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   }

   /* Slot 2 is third in the scalar-load preference and is the first slot
    * released.  Slots 6, 1, 3, 4, and 5 remain pending across the new load. */
   uint32_t release_sources[] = {loads[2], ordinary};
   agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                       AGX_APPLE9_ENC_FLOAT2_COMPACT, release_sources, 2, 0);
   loads[6] = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                  AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 6);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, loads[6], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));

   for (unsigned i : {0u, 1u, 3u, 4u, 5u, 6u}) {
      uint32_t sources[] = {loads[i], ordinary};
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                          AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
   }

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   const uint8_t expected[] = {6, 1, 2, 3, 4, 5, 2};
   unsigned seen = 0;
   for (unsigned i = 0; i < program.instruction_count; ++i) {
      if (program.instructions[i]->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         EXPECT_EQ(program.instructions[i]->producer_scoreboard_slot,
                   expected[seen++]);
      }
   }
   EXPECT_EQ(seen, 7u);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, SevenIndependentPendingGroupsMaterializeOldestHandoff)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t ordinary = agx_apple9_vir_input(&program, 2);
   uint32_t loads[7];

   for (unsigned i = 0; i < 7; ++i) {
      loads[i] = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                     AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, i);
      ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
         &program, loads[i], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   }
   program.instructions[6]->device_load_index_kind =
      AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR;
   uint32_t last_result = AGX_APPLE9_VREG_INVALID;
   for (unsigned i = 0; i < 7; ++i) {
      uint32_t sources[] = {loads[i], ordinary};
      last_result =
         agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                             AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
   }
   program.output = last_result;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   const uint8_t expected[] = {6, 1, 2, 3, 4, 5, 6};
   unsigned seen = 0, materialized = 0;
   for (unsigned i = 0; i < program.instruction_count; ++i) {
      if (program.instructions[i]->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         EXPECT_EQ(program.instructions[i]->producer_scoreboard_slot,
                   expected[seen++]);
      }
      materialized += program.instructions[i]->scoreboard_materialize;
   }
   EXPECT_EQ(seen, 7u);
   EXPECT_EQ(materialized, 1u);
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   bool general_materialization = false;
   for (unsigned i = 0; i < program.instruction_count; ++i) {
      if (program.instructions[i]->scoreboard_materialize) {
         EXPECT_EQ(program.instructions[i]->encoding,
                   AGX_APPLE9_ENC_LOGIC_EXTENDED);
         general_materialization |=
            program.phys[program.instructions[i]->dest] >= 16;
      }
   }
   EXPECT_TRUE(general_materialization);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, ReciprocalUsesSharedOneHotDependencySlots)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t ordinary = agx_apple9_vir_input(&program, 2);
   uint32_t loads[5];

   for (unsigned i = 0; i < ARRAY_SIZE(loads); ++i) {
      loads[i] = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                     AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, i);
      ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
         &program, loads[i], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   }

   uint32_t reciprocal =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FRCP,
                          AGX_APPLE9_ENC_FLOAT_SPECIAL, &loads[4], 1, 0x02);
   for (unsigned i = 0; i < 4; ++i) {
      uint32_t sources[] = {loads[i], ordinary};
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                          AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
   }
   program.output = reciprocal;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");

   unsigned seen_loads = 0, materializations = 0, reciprocals = 0;
   const uint8_t expected_slots[] = {6, 1, 2, 3, 4};
   for (unsigned i = 0; i < program.instruction_count; ++i) {
      const auto &instruction = *program.instructions[i];
      if (instruction.op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         EXPECT_EQ(instruction.producer_scoreboard_slot,
                   expected_slots[seen_loads++]);
      }
      materializations += instruction.scoreboard_materialize;
      if (instruction.op == AGX_APPLE9_VIR_FRCP) {
         ++reciprocals;
         EXPECT_EQ(instruction.scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_4);
      }
   }
   EXPECT_EQ(seen_loads, ARRAY_SIZE(loads));
   EXPECT_EQ(materializations, 0u);
   EXPECT_EQ(reciprocals, 1u);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, MultiSourcePendingLoadsShareOneSlot)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t loads[2];
   for (unsigned i = 0; i < 2; ++i) {
      loads[i] = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                     AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, i);
      ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
         &program, loads[i], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   }
   agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FMUL,
                       AGX_APPLE9_ENC_FLOAT2_COMPACT, loads, 2, 0);

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[1]->producer_scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[0]->device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_5101);
   EXPECT_EQ(program.instructions[1]->device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_5101);
   EXPECT_EQ(program.instructions[2]->scoreboard_slot, 6u);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, FirstHandoffRetainsGprForLaterReads)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t ordinary = agx_apple9_vir_input(&program, 2);
   uint32_t load =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                          AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, load, 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, index));
   uint32_t sources[] = {load, ordinary};
   agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                       AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
   uint32_t result =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FMUL,
                          AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
   program.output = result;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(program.instructions[1]->scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[2]->scoreboard_slot, 0u);
   EXPECT_NE(program.instructions[1]->live_after_mask & 1u, 0u);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, RawLoadTokenDefinesProducerScoreboardSlot)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t alu = agx_apple9_vir_input(&program, 2);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_9100,
   };
   uint32_t load =
      agx_apple9_vir_emit_device_load(&program, 0, index, &contract);
   ASSERT_NE(load, AGX_APPLE9_VREG_INVALID);
   uint32_t sources[] = {load, alu};
   agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                       AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(program.instructions[0]->device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_9100);
   EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_3);
   EXPECT_EQ(program.instructions[1]->scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_3);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, IntegerAddDirectlyConsumesSlot6)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t alu = agx_apple9_vir_input(&program, 2);
   uint32_t load =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                          AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
   ASSERT_NE(load, AGX_APPLE9_VREG_INVALID);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, load, 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));

   uint32_t sources[] = {load, alu};
   uint32_t result =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IADD,
                          AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources, 2, 0);
   program.output = result;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_EQ(program.instruction_count, 2u);
   EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[1]->scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[1]->op, AGX_APPLE9_VIR_IADD);
   EXPECT_EQ(program.instructions[1]->src[0], load);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, AllLogicDirectlyConsumePendingLoadGroups)
{
   for (auto op :
        {AGX_APPLE9_VIR_IAND, AGX_APPLE9_VIR_IOR, AGX_APPLE9_VIR_IXOR}) {
      agx_apple9_vir_program program;
      agx_apple9_vir_init(&program);
      uint32_t index = agx_apple9_vir_input(&program, 1);
      uint32_t loads[2];
      for (unsigned i = 0; i < 2; ++i) {
         loads[i] =
            agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, i);
         ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
            &program, loads[i], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
      }
      agx_apple9_vir_emit(&program, op, AGX_APPLE9_ENC_LOGIC_EXTENDED, loads, 2,
                          0);

      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
         << (reason ? reason : "") << " op=" << unsigned(op);
      ASSERT_EQ(program.instruction_count, 3u);
      EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot, 6u);
      EXPECT_EQ(program.instructions[1]->producer_scoreboard_slot, 6u);
      EXPECT_EQ(program.instructions[2]->scoreboard_slot, 6u);
      EXPECT_FALSE(program.instructions[2]->scoreboard_materialize);
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Vir, MaskedPhiEdgesShareOneAllocatedMergeDestination)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t then_value = agx_apple9_vir_input(&program, 2);
   uint32_t else_value = agx_apple9_vir_input(&program, 3);
   uint32_t merge = agx_apple9_vir_emit_phi(&program, nullptr);
   ASSERT_NE(merge, AGX_APPLE9_VREG_INVALID);
   ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&program, merge, then_value));
   ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&program, merge, else_value));
   program.output = merge;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");

   ASSERT_EQ(program.instruction_count, 3u);
   EXPECT_EQ(program.instructions[0]->op, AGX_APPLE9_VIR_PHI);
   for (unsigned i = 1; i < 3; ++i) {
      EXPECT_EQ(program.instructions[i]->op, AGX_APPLE9_VIR_PHI_SRC);
      EXPECT_EQ(program.instructions[i]->target, merge);
      EXPECT_EQ(program.phys[program.instructions[i]->target],
                program.phys[merge]);
   }
   EXPECT_GE(program.phys[merge], 16); /* Leave constrained coordinate tuples available. */
   EXPECT_NE(program.phys[merge], program.phys[then_value]);
   EXPECT_NE(program.phys[merge], program.phys[else_value]);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, ParallelPhiAssignmentsPreserveAllSourcesAfterAllocation)
{
   /* Every four-source assignment: chains, fanout, self copies, swaps,
    * disjoint cycles, and three/four-way rotations. Check allocated register
    * execution against simultaneous assignment, including arbitrary bit patterns. */
   const uint32_t initial[] = {0xdeadbeef, 0xbf800000, 0x7fc12345, 0x80000000};
   for (unsigned pattern = 0; pattern < 256; ++pattern) {
      SCOPED_TRACE(pattern);
      agx_apple9_vir_program program;
      agx_apple9_vir_init(&program);
      uint32_t values[4];
      for (unsigned i = 0; i < 4; ++i)
         values[i] = agx_apple9_vir_emit_phi(&program, nullptr);
      agx_apple9_vir_copy copies[4];
      for (unsigned i = 0; i < 4; ++i)
         copies[i] = {values[i], values[(pattern >> (2 * i)) & 3]};
      ASSERT_TRUE(agx_apple9_vir_emit_phi_edge(&program, copies, 4));
      program.output = agx_apple9_vir_emit_collect(&program, values, 4);
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
         << (reason ? reason : "");
      ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
         << (reason ? reason : "");

      uint32_t registers[64] = {};
      for (unsigned i = 0; i < 4; ++i)
         registers[program.phys[values[i]]] = initial[i];
      bool checked = false;
      for (unsigned i = 0; i < program.instruction_count; ++i) {
         const auto &ins = *program.instructions[i];
         switch (ins.op) {
         case AGX_APPLE9_VIR_PHI:
            break;
         case AGX_APPLE9_VIR_IOR:
            registers[program.phys[ins.dest]] =
               registers[program.phys[ins.src[0]]] |
               registers[program.phys[ins.src[1]]];
            break;
         case AGX_APPLE9_VIR_PHI_SRC:
            ASSERT_TRUE(agx_apple9_resolve_phi_edge(&program, &ins,
               [](void *data, bool swap, unsigned dest, unsigned source) {
                  auto *registers = static_cast<uint32_t *>(data);
                  if (swap) {
                     uint32_t old = registers[dest];
                     registers[dest] = registers[source];
                     registers[source] = old;
                  } else {
                     registers[dest] = registers[source];
                  }
                  return true;
               }, registers));
            break;
         case AGX_APPLE9_VIR_COLLECT:
            for (unsigned c = 0; c < 4; ++c)
               EXPECT_EQ(registers[program.phys[values[c]]],
                         initial[(pattern >> (2 * c)) & 3]);
            checked = true;
            break;
         default:
            FAIL() << "Unexpected opcode " << unsigned(ins.op);
         }
      }
      EXPECT_TRUE(checked);
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Vir, LogicNormalizesSinglePendingSourceIntoSourceA)
{
   for (auto op :
        {AGX_APPLE9_VIR_IAND, AGX_APPLE9_VIR_IOR, AGX_APPLE9_VIR_IXOR}) {
      agx_apple9_vir_program program;
      agx_apple9_vir_init(&program);
      uint32_t index = agx_apple9_vir_input(&program, 1);
      uint32_t ordinary = agx_apple9_vir_input(&program, 2);
      uint32_t load =
         agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                             AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
      ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
         &program, load, 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));

      /* Deliberately present the pending operand in source B. */
      uint32_t sources[] = {ordinary, load};
      agx_apple9_vir_emit(&program, op, AGX_APPLE9_ENC_LOGIC_EXTENDED, sources,
                          2, 0);

      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
         << (reason ? reason : "") << " op=" << unsigned(op);
      ASSERT_EQ(program.instruction_count, 2u);
      EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot, 6u);
      EXPECT_EQ(program.instructions[1]->scoreboard_slot, 6u);
      EXPECT_EQ(program.instructions[1]->src[0], load);
      EXPECT_EQ(program.instructions[1]->src[1], ordinary);
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Vir, ExplicitMultiSourceGroupRejectsDifferentSlots)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t alu_a = agx_apple9_vir_input(&program, 2);
   uint32_t alu_b = agx_apple9_vir_input(&program, 3);
   uint32_t load6 =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                          AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
   uint32_t load1 =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_DEVICE_LOAD,
                          AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 1);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, load6, 0, AGX_APPLE9_SCOREBOARD_SLOT_6));
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, load1, 0, AGX_APPLE9_SCOREBOARD_SLOT_1));
   uint32_t mixed_arms[4] = {alu_a, alu_b, load6, load1};
   agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_SELECT,
                       AGX_APPLE9_ENC_SELECT_GPR_WIDE, mixed_arms, 4,
                       AGX_APPLE9_SELECT_ULT);
   const char *reason = nullptr;
   EXPECT_FALSE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason));
   ASSERT_NE(reason, nullptr);
   EXPECT_STREQ(reason,
                "Apple9 multi-source scoreboard group uses different slots");
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, VectorStoreDirectlyConsumesItsLoadTuple)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 4);
   const agx_apple9_device_load_contract retained = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   const agx_apple9_device_load_contract second = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };

   uint32_t earlier =
      agx_apple9_vir_emit_device_load(&program, 0, index, &retained);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, earlier, AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   uint32_t vector =
      agx_apple9_vir_emit_device_load_vector(&program, 1, index, 4, &second);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, vector, 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   uint32_t lanes[] = {vector, vector + 1, vector + 2, vector + 3};
   ASSERT_TRUE(
      agx_apple9_vir_emit_device_store(&program, 2, index, lanes, 4, 32));
   uint32_t output_sources[] = {earlier, earlier};
   program.output =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IOR,
                          AGX_APPLE9_ENC_LOGIC_EXTENDED, output_sources, 2, 0);

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");

   unsigned collect_index = UINT_MAX;
   unsigned store_index = UINT_MAX;
   for (unsigned i = 0; i < program.instruction_count; ++i) {
      collect_index = program.instructions[i]->op == AGX_APPLE9_VIR_COLLECT
                         ? i
                         : collect_index;
      store_index = program.instructions[i]->op == AGX_APPLE9_VIR_DEVICE_STORE
                       ? i
                       : store_index;
   }
   ASSERT_EQ(collect_index, UINT_MAX);
   ASSERT_NE(store_index, UINT_MAX);
   EXPECT_EQ(program.instructions[store_index]->scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_1);
   for (unsigned c = 0; c < 4; ++c)
      EXPECT_EQ(program.instructions[store_index]->src[c], vector + c);

   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   for (unsigned c = 1; c < 4; ++c)
      EXPECT_EQ(program.phys[vector + c], program.phys[vector] + c);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, ReturningAtomicUsesAdjacentNativeResultMaterializer)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 2);
   uint32_t data = agx_apple9_vir_input(&program, 3);
   uint32_t result = AGX_APPLE9_VREG_INVALID;
   ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
      &program, 8, index, &data, 1, AGX_APPLE9_ATOMIC_ADD, false, &result));
   ASSERT_NE(result, AGX_APPLE9_VREG_INVALID);
   const uint32_t sources[] = {result, result};
   uint32_t durable = agx_apple9_vir_emit(
      &program, AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2,
      0);
   ASSERT_NE(durable, AGX_APPLE9_VREG_INVALID);
   program.output = durable;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_EQ(program.instruction_count, 3u);
   EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[1]->op,
             AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
   EXPECT_EQ(program.instructions[1]->scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_NONE);
   EXPECT_EQ(program.instructions[2]->scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);

   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   agx_apple9_packed_instruction atomic = {};
   agx_apple9_packed_instruction materialize = {};
   agx_apple9_packed_instruction copy = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      program.instructions[0], program.phys, &atomic, &reason));
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      program.instructions[1], program.phys, &materialize, &reason));
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      program.instructions[2], program.phys, &copy, &reason));
   EXPECT_EQ(atomic.bytes[0], 0x67);
   EXPECT_EQ(atomic.bytes[1], 0x01);
   EXPECT_EQ(atomic.bytes[2], 0x54);
   EXPECT_EQ(atomic.bytes[9], 0x02);
   ASSERT_EQ(materialize.length, 8u);
   EXPECT_EQ(materialize.bytes[0] & 0x0f, 0x0c);
   EXPECT_EQ(materialize.bytes[0] >> 4,
             program.phys[program.instructions[0]->dest]);
   EXPECT_EQ(materialize.bytes[1], 0x80);
   EXPECT_EQ(materialize.bytes[2], 0x09);
   EXPECT_EQ(copy.bytes[5] & 0xe0, 0u);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, ReturningAtomicReusesConsumedPendingInputSlot)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 2);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t data =
      agx_apple9_vir_emit_device_load(&program, 7, index, &contract);
   ASSERT_NE(data, AGX_APPLE9_VREG_INVALID);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, data, 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));

   uint32_t result = AGX_APPLE9_VREG_INVALID;
   ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
      &program, 8, index, &data, 1, AGX_APPLE9_ATOMIC_ADD, false, &result));
   const uint32_t sources[] = {result, result};
   uint32_t durable = agx_apple9_vir_emit(
      &program, AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2,
      0);
   ASSERT_NE(durable, AGX_APPLE9_VREG_INVALID);
   program.output = durable;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_EQ(program.instruction_count, 4u);
   EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[1]->scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[1]->producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[2]->producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[3]->scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);

   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      program.instructions[1], program.phys, &packed, &reason));
   EXPECT_EQ(((packed.bytes[1] >> 4) & 0x0f) |
                ((packed.bytes[2] & 0x03) << 4),
             1u << (AGX_APPLE9_SCOREBOARD_SLOT_6 - 1));
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, SixPendingLoadsFeedReturningAtomicsAtFullSlotPressure)
{
   static const uint8_t expected_slots[] = {6, 1, 2, 3, 4, 5};
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 2);
   uint32_t data[ARRAY_SIZE(expected_slots)];
   uint32_t result[ARRAY_SIZE(expected_slots)];
   unsigned load_instruction[ARRAY_SIZE(expected_slots)];
   unsigned atomic_instruction[ARRAY_SIZE(expected_slots)];
   unsigned publication_instruction[ARRAY_SIZE(expected_slots)];
   unsigned consumer_instruction[ARRAY_SIZE(expected_slots)];

   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(data); ++i) {
      load_instruction[i] = program.instruction_count;
      data[i] = agx_apple9_vir_emit_device_load(&program, 3 + i, index,
                                                &contract);
      ASSERT_NE(data[i], AGX_APPLE9_VREG_INVALID);
      ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
         &program, data[i], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   }

   for (unsigned i = 0; i < ARRAY_SIZE(result); ++i) {
      atomic_instruction[i] = program.instruction_count;
      ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
         &program, 10 + i, index, &data[i], 1, AGX_APPLE9_ATOMIC_ADD, false,
         &result[i]));
      publication_instruction[i] = atomic_instruction[i] + 1;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(result); ++i) {
      const uint32_t sources[] = {result[i], result[i]};
      consumer_instruction[i] = program.instruction_count;
      uint32_t durable = agx_apple9_vir_emit(
         &program, AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED,
         sources, ARRAY_SIZE(sources), 0);
      ASSERT_NE(durable, AGX_APPLE9_VREG_INVALID);
      program.output = durable;
   }

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   for (unsigned i = 0; i < ARRAY_SIZE(expected_slots); ++i) {
      EXPECT_EQ(program.instructions[load_instruction[i]]->producer_scoreboard_slot,
                expected_slots[i]);
      EXPECT_EQ(program.instructions[atomic_instruction[i]]->scoreboard_slot,
                expected_slots[i]);
      EXPECT_EQ(program.instructions[atomic_instruction[i]]->producer_scoreboard_slot,
                expected_slots[i]);
      EXPECT_EQ(program.instructions[publication_instruction[i]]->producer_scoreboard_slot,
                expected_slots[i]);
      EXPECT_EQ(program.instructions[consumer_instruction[i]]->scoreboard_slot,
                expected_slots[i]);
   }
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, DiscardedAtomicIsDestinationlessSideEffect)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 2);
   uint32_t data = agx_apple9_vir_input(&program, 3);
   const unsigned value_count = program.value_count;

   ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
      &program, 8, index, &data, 1, AGX_APPLE9_ATOMIC_XOR, true, nullptr));
   ASSERT_EQ(program.instruction_count, 1u);
   EXPECT_EQ(program.value_count, value_count);
   EXPECT_EQ(program.instructions[0]->op, AGX_APPLE9_VIR_DEVICE_ATOMIC);
   EXPECT_EQ(program.instructions[0]->dest, AGX_APPLE9_VREG_INVALID);
   EXPECT_TRUE(program.instructions[0]->atomic_discard);

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      program.instructions[0], program.phys, &packed, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(packed.bytes[1], 0x01);
   EXPECT_EQ(packed.bytes[9], 0x40);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, ReturningAtomicsUseTheCommonSixSlotAllocator)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 2);
   uint32_t data = agx_apple9_vir_input(&program, 3);
   uint32_t results[6];
   uint32_t durable[6];

   for (unsigned i = 0; i < ARRAY_SIZE(results); ++i) {
      ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
         &program, 8, index, &data, 1, AGX_APPLE9_ATOMIC_ADD, false,
         &results[i]));
   }
   for (unsigned i = 0; i < ARRAY_SIZE(results); ++i) {
      uint32_t sources[] = {results[i], results[i]};
      durable[i] = agx_apple9_vir_emit(
         &program, AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED,
         sources, 2, 0);
      ASSERT_NE(durable[i], AGX_APPLE9_VREG_INVALID);
      ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, durable[i]));
   }
   program.output = durable[5];

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");

   static const unsigned expected_slots[] = {6, 1, 2, 3, 4, 5};
   unsigned atomic_index = 0;
   for (unsigned i = 0; i < program.instruction_count; ++i) {
      const agx_apple9_vir_instr &instruction = *program.instructions[i];
      if (instruction.op != AGX_APPLE9_VIR_DEVICE_ATOMIC)
         continue;

      ASSERT_LT(atomic_index, ARRAY_SIZE(expected_slots));
      EXPECT_EQ(instruction.scoreboard_slot,
                AGX_APPLE9_SCOREBOARD_SLOT_NONE);
      EXPECT_EQ(instruction.producer_scoreboard_slot,
                expected_slots[atomic_index]);
      ASSERT_LT(i + 1, program.instruction_count);
      EXPECT_EQ(program.instructions[i + 1]->op,
                AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
      EXPECT_EQ(program.instructions[i + 1]->producer_scoreboard_slot,
                expected_slots[atomic_index]);
      ++atomic_index;
   }
   EXPECT_EQ(atomic_index, ARRAY_SIZE(expected_slots));

   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   bool saw_nonzero_landing = false;
   for (unsigned i = 0; i < ARRAY_SIZE(results); ++i) {
      EXPECT_LT(program.phys[results[i]], 64u);
      saw_nonzero_landing |= program.phys[results[i]] != 0;
   }
   EXPECT_TRUE(saw_nonzero_landing);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, SeventhReturningAtomicMaterializesOldestPendingGroup)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 2);
   uint32_t data = agx_apple9_vir_input(&program, 3);
   uint32_t results[7];

   for (unsigned i = 0; i < ARRAY_SIZE(results); ++i) {
      ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
         &program, 8, index, &data, 1, AGX_APPLE9_ATOMIC_ADD, false,
         &results[i]));
   }
   for (unsigned i = 0; i < ARRAY_SIZE(results); ++i) {
      uint32_t sources[] = {results[i], results[i]};
      uint32_t durable = agx_apple9_vir_emit(
         &program, AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED,
         sources, 2, 0);
      ASSERT_NE(durable, AGX_APPLE9_VREG_INVALID);
      ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, durable));
      program.output = durable;
   }

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   unsigned materializations = 0;
   for (unsigned i = 0; i < program.instruction_count; ++i)
      materializations += program.instructions[i]->scoreboard_materialize;
   EXPECT_EQ(materializations, 1u);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, CompareExchangeCollectsDesiredCompareTuple)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t desired = agx_apple9_vir_input(&program, 4);
   uint32_t spacer = agx_apple9_vir_input(&program, 8);
   uint32_t compare = agx_apple9_vir_input(&program, 6);
   uint32_t data[] = {desired, compare};
   uint32_t result = AGX_APPLE9_VREG_INVALID;
   ASSERT_TRUE(agx_apple9_vir_emit_device_atomic(
      &program, 8, index, data, 2, AGX_APPLE9_ATOMIC_CMPXCHG, false,
      &result));
   ASSERT_NE(result, AGX_APPLE9_VREG_INVALID);
   ASSERT_EQ(program.instructions[0]->op, AGX_APPLE9_VIR_COLLECT);
   const uint32_t copy_sources[] = {result, result};
   uint32_t durable = agx_apple9_vir_emit(
      &program, AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED,
      copy_sources, 2, 0);
   ASSERT_NE(durable, AGX_APPLE9_VREG_INVALID);
   ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, spacer));
   program.output = durable;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   const agx_apple9_vir_instr &atomic = *program.instructions[1];
   EXPECT_EQ(program.phys[atomic.src[1]], program.phys[atomic.src[0]] + 1);
   EXPECT_NE(program.phys[atomic.dest], program.phys[atomic.src[0]]);
   EXPECT_EQ(program.instructions[2]->op,
             AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Packer, LowRegisterSelectMatchesCallerOwnedCompilerForm)
{
   uint8_t phys[] = {0, 0, 2, 4, 3};
   agx_apple9_vir_instr select = {
      .op = AGX_APPLE9_VIR_SELECT,
      .encoding = AGX_APPLE9_ENC_SELECT_GPR_WIDE,
      .dest = 0,
      .src = {1, 2, 3, 4},
      .immediate = AGX_APPLE9_SELECT_ULT,
      .nr_srcs = 4,
   };
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&select, phys, &packed, &reason))
      << reason;
   static const uint8_t expected[] = {
      0x02, 0x01, 0x1f, 0x05, 0x82, 0x08, 0x05, 0x00, 0x80, 0x06,
   };
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);

   select.live_after_mask = 0x3;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&select, phys, &packed, &reason))
      << reason;
   static const uint8_t expected_live_ab[] = {
      0x02, 0x81, 0x07, 0x85, 0x82, 0x08, 0x05, 0x00, 0x80, 0x06,
   };
   ASSERT_EQ(packed.length, sizeof(expected_live_ab));
   EXPECT_EQ(memcmp(packed.bytes, expected_live_ab, sizeof(expected_live_ab)),
             0);

   select.live_after_mask = 0xf;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&select, phys, &packed, &reason))
      << reason;
   static const uint8_t expected_live_all[] = {
      0x02, 0x81, 0x07, 0x85, 0x02, 0x88, 0x05, 0x00, 0x00, 0x86,
   };
   ASSERT_EQ(packed.length, sizeof(expected_live_all));
   EXPECT_EQ(memcmp(packed.bytes, expected_live_all, sizeof(expected_live_all)),
             0);

   select.live_after_mask = 0;
   for (uint8_t slot = 0; slot <= 6; ++slot) {
      select.scoreboard_slot = slot;
      ASSERT_TRUE(
         agx_apple9_pack_vir_instruction(&select, phys, &packed, &reason))
         << reason;
      EXPECT_EQ((packed.bytes[7] >> 5) & 7, slot);
   }

   uint8_t high_phys[] = {0, 64, 65, 66, 67};
   for (uint8_t slot = 0; slot <= 6; ++slot) {
      select.scoreboard_slot = slot;
      ASSERT_TRUE(
         agx_apple9_pack_vir_instruction(&select, high_phys, &packed, &reason))
         << reason;
      EXPECT_EQ((packed.bytes[7] >> 5) & 7, slot);
   }

   select.scoreboard_slot = 7;
   EXPECT_FALSE(
      agx_apple9_pack_vir_instruction(&select, high_phys, &packed, &reason));
   ASSERT_NE(reason, nullptr);
   EXPECT_NE(strstr(reason, "dependency"), nullptr);
}

TEST(Apple9Packer, SelectConditionAndEqualityModeMatchHardwareSweeps)
{
   uint8_t phys[] = {0, 0, 2, 4, 3};
   agx_apple9_vir_instr select = {
      .op = AGX_APPLE9_VIR_SELECT,
      .encoding = AGX_APPLE9_ENC_SELECT_GPR_WIDE,
      .dest = 0,
      .src = {1, 2, 3, 4},
      .nr_srcs = 4,
   };
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;

   struct {
      uint32_t immediate;
      uint8_t mode;
      uint8_t condition;
   } cases[] = {
      {AGX_APPLE9_SELECT_FGT, 0x82, 0x02},
      {AGX_APPLE9_SELECT_FLT, 0x82, 0x03},
      {AGX_APPLE9_SELECT_UGT, 0x82, 0x04},
      {AGX_APPLE9_SELECT_ULT, 0x82, 0x05},
      {AGX_APPLE9_SELECT_IGT, 0x82, 0x06},
      {AGX_APPLE9_SELECT_ILT, 0x82, 0x07},
      {AGX_APPLE9_SELECT_FEQ | AGX_APPLE9_SELECT_EQUALITY, 0x86, 0x00},
   };

   for (const auto &test : cases) {
      select.immediate = test.immediate;
      ASSERT_TRUE(
         agx_apple9_pack_vir_instruction(&select, phys, &packed, &reason))
         << reason;
      ASSERT_EQ(packed.length, 10u);
      EXPECT_EQ(packed.bytes[4], test.mode);
      EXPECT_EQ(packed.bytes[6], test.condition);
   }
}

TEST(Apple9Packer, PredicateFormsEncodePolarityAndSourceLifetime)
{
   agx_apple9_vir_instr predicate = {
      .op = AGX_APPLE9_VIR_PREDICATE_COMPARE,
      .encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT,
      .dest = AGX_APPLE9_VREG_INVALID,
      .src = {0, 1},
      .immediate = AGX_APPLE9_PREDICATE_ILT,
      .nr_srcs = 2,
   };
   const uint8_t phys[] = {1, 2};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t expected[] = {0x0a, 0x03, 0x3a, 0x05, 0x07, 0xc0};
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);

   predicate.immediate |= AGX_APPLE9_PREDICATE_INVERT;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t inverted[] = {0x1a, 0x03, 0x3a, 0x05, 0x07, 0xc0};
   EXPECT_EQ(memcmp(packed.bytes, inverted, sizeof(inverted)), 0);

   predicate.immediate = AGX_APPLE9_PREDICATE_ILT;
   predicate.live_after_mask = BITFIELD_BIT(0);
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t retain_a[] = {0x0a, 0x03, 0x32, 0x05, 0x07, 0xc0};
   EXPECT_EQ(memcmp(packed.bytes, retain_a, sizeof(retain_a)), 0);

   predicate.live_after_mask = BITFIELD_BIT(1);
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t retain_b[] = {0x0a, 0x03, 0x2a, 0x05, 0x07, 0xc0};
   EXPECT_EQ(memcmp(packed.bytes, retain_b, sizeof(retain_b)), 0);

   predicate.live_after_mask = BITFIELD_BIT(0) | BITFIELD_BIT(1);
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t retain_both[] = {0x0a, 0x03, 0x22, 0x05, 0x07, 0xc0};
   EXPECT_EQ(memcmp(packed.bytes, retain_both, sizeof(retain_both)), 0);

   const uint8_t high_phys[] = {63, 62};
   predicate.immediate = AGX_APPLE9_PREDICATE_ULT;
   predicate.live_after_mask = 0;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, high_phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t high[] = {0x0a, 0x7f, 0x3a, 0x7d, 0x05, 0xc0};
   EXPECT_EQ(memcmp(packed.bytes, high, sizeof(high)), 0);

   predicate.encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
   predicate.immediate = AGX_APPLE9_PREDICATE_EXT_IEQ;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t integer_equal[] = {0x0a, 0x03, 0x3b, 0x05, 0x06,
                                    0x00, 0x07, 0xc0, 0x00, 0x00};
   ASSERT_EQ(packed.length, sizeof(integer_equal));
   EXPECT_EQ(memcmp(packed.bytes, integer_equal, sizeof(integer_equal)), 0);

   predicate.immediate = AGX_APPLE9_PREDICATE_EXT_FEQ;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t float_equal[] = {0x0a, 0x03, 0x3b, 0x05, 0x06,
                                  0x00, 0x00, 0xc0, 0x00, 0x00};
   EXPECT_EQ(memcmp(packed.bytes, float_equal, sizeof(float_equal)), 0);

   predicate.immediate =
      AGX_APPLE9_PREDICATE_EXT_FGE_SEQUENCE | AGX_APPLE9_PREDICATE_INVERT;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t float_ge[] = {0x1a, 0x03, 0x3b, 0x05, 0x06,
                               0x00, 0x02, 0xc0, 0x00, 0x00};
   EXPECT_EQ(memcmp(packed.bytes, float_ge, sizeof(float_ge)), 0);

   /* Native nested breaks keep the predicate at the target loop's level,
    * independent of how many conditional scopes are unwound. */
   predicate.encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP;
   predicate.immediate =
      AGX_APPLE9_PREDICATE_EXT_IEQ | AGX_APPLE9_PREDICATE_BANK(1);
   predicate.live_after_mask = 0;
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t loop_depth_one[] = {0x2a, 0x03, 0x23, 0x05, 0x06,
                                     0x00, 0x07, 0x00, 0x00, 0x00};
   EXPECT_EQ(memcmp(packed.bytes, loop_depth_one, sizeof(loop_depth_one)), 0);

   predicate.immediate =
      AGX_APPLE9_PREDICATE_EXT_IEQ | AGX_APPLE9_PREDICATE_BANK(2);
   ASSERT_TRUE(
      agx_apple9_pack_vir_instruction(&predicate, phys, &packed, &reason))
      << (reason ? reason : "");
   const uint8_t loop_depth_two[] = {0x4a, 0x03, 0x23, 0x05, 0x06,
                                     0x00, 0x07, 0x00, 0x00, 0x00};
   EXPECT_EQ(memcmp(packed.bytes, loop_depth_two, sizeof(loop_depth_two)), 0);
}

TEST(Apple9Packer, SimpleExecutionMaskScopeMatchesOwnSourceMetal)
{
   const struct {
      agx_apple9_vir_opcode op;
      agx_apple9_encoding encoding;
      uint8_t bytes[6];
      uint8_t length;
   } cases[] = {
      {AGX_APPLE9_VIR_EXEC_MASK_PUSH,
       AGX_APPLE9_ENC_EXEC_MASK_PUSH,
       {0x0f, 0x05, 0x54, 0x01},
       4},
      {AGX_APPLE9_VIR_EXEC_MASK_ELSE,
       AGX_APPLE9_ENC_EXEC_MASK_ELSE,
       {0x0f, 0x04, 0x04, 0x19},
       4},
      {AGX_APPLE9_VIR_EXEC_MASK_POP,
       AGX_APPLE9_ENC_EXEC_MASK_POP,
       {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00},
       6},
   };

   for (const auto &test : cases) {
      agx_apple9_vir_instr instruction = {
         .op = test.op,
         .encoding = test.encoding,
         .dest = AGX_APPLE9_VREG_INVALID,
         .immediate = test.op == AGX_APPLE9_VIR_EXEC_MASK_PUSH
                         ? AGX_APPLE9_EXEC_MASK_PREDICATE(0)
                         : 0,
      };
      agx_apple9_packed_instruction packed = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&instruction, nullptr,
                                                  &packed, &reason))
         << (reason ? reason : "");
      ASSERT_EQ(packed.length, test.length);
      EXPECT_EQ(memcmp(packed.bytes, test.bytes, test.length), 0);
   }

   agx_apple9_vir_instr inverted_push = {
      .op = AGX_APPLE9_VIR_EXEC_MASK_PUSH,
      .encoding = AGX_APPLE9_ENC_EXEC_MASK_PUSH,
      .dest = AGX_APPLE9_VREG_INVALID,
      .immediate =
         AGX_APPLE9_EXEC_MASK_PREDICATE(0) | AGX_APPLE9_EXEC_MASK_INVERT,
   };
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&inverted_push, nullptr, &packed,
                                               &reason))
      << (reason ? reason : "");
   const uint8_t expected[] = {0x0f, 0x05, 0x54, 0x21};
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
}

TEST(Apple9Packer, StructuredLoopControlMatchesOwnSourceMetal)
{
   const struct {
      agx_apple9_vir_opcode op;
      agx_apple9_encoding encoding;
      uint32_t immediate;
      uint8_t bytes[10];
      uint8_t length;
   } cases[] = {
      {AGX_APPLE9_VIR_LOOP_MASK_PUSH,
       AGX_APPLE9_ENC_LOOP_MASK_PUSH,
       0,
       {0x0f, 0x05, 0x54, 0x1a},
       4},
      {AGX_APPLE9_VIR_LOOP_MASK_UPDATE,
       AGX_APPLE9_ENC_LOOP_MASK_UPDATE,
       0x22,
       {0x8f, 0x04, 0x54, 0x22},
       4},
      {AGX_APPLE9_VIR_LOOP_MASK_UPDATE,
       AGX_APPLE9_ENC_LOOP_MASK_UPDATE,
       0x2a,
       {0x8f, 0x04, 0x54, 0x2a},
       4},
      {AGX_APPLE9_VIR_JMP_EXEC_ANY,
       AGX_APPLE9_ENC_JMP_EXEC_ANY,
       (uint32_t)(int32_t)-58,
       {0x0f, 0x00, 0x54, 0xc6, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00},
       10},
      {AGX_APPLE9_VIR_JMP_EXEC_NONE,
       AGX_APPLE9_ENC_JMP_EXEC_NONE,
       92,
       {0x0f, 0x01, 0x54, 0x5c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
       10},
      {AGX_APPLE9_VIR_BREAK_MASK_UNWIND,
       AGX_APPLE9_ENC_BREAK_MASK_UNWIND,
       AGX_APPLE9_BREAK_IMMEDIATE(3, 2),
       {0x8f, 0x05, 0x54, 0x03, 0x00, 0x02},
       6},
      {AGX_APPLE9_VIR_LOOP_MASK_POP,
       AGX_APPLE9_ENC_LOOP_MASK_POP,
       0,
       {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00},
       6},
   };

   for (const auto &test : cases) {
      agx_apple9_vir_instr instruction = {
         .op = test.op,
         .encoding = test.encoding,
         .dest = AGX_APPLE9_VREG_INVALID,
         .immediate = test.immediate,
      };
      agx_apple9_packed_instruction packed = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&instruction, nullptr,
                                                  &packed, &reason))
         << (reason ? reason : "");
      ASSERT_EQ(packed.length, test.length);
      EXPECT_EQ(memcmp(packed.bytes, test.bytes, test.length), 0);
   }
}

TEST(Apple9Allocator, ReleasesKilledSourcesAfterTheirConsumer)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t gid = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_GET_GLOBAL_ID,
                                      AGX_APPLE9_ENC_GET_SR, nullptr, 0, 0);
   uint32_t c =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IMM,
                          AGX_APPLE9_ENC_MOV_IMM_COMPACT, nullptr, 0, 7);
   uint32_t sources[] = {gid, c};
   uint32_t value =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IADD,
                          AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources, 2, 0);
   program.output = value;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
   EXPECT_LT(program.phys[gid], 64u);
   EXPECT_LT(program.phys[c], 16u);
   EXPECT_NE(program.phys[gid], program.phys[c]);
   EXPECT_NE(program.phys[value], program.phys[gid]);
   EXPECT_NE(program.phys[value], program.phys[c]);
   EXPECT_GE(program.phys[value], 16u);
   EXPECT_GE(program.max_phys_gpr, 16u);
   EXPECT_TRUE(agx_apple9_validate_vir_allocation(&program, &reason)) << reason;
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, ReportsNoSpillPressureLimit)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t values[17];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      values[i] =
         agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IMM,
                             AGX_APPLE9_ENC_MOV_IMM_COMPACT, nullptr, 0, i);
   }
   program.output = values[ARRAY_SIZE(values) - 1];
   for (unsigned i = 0; i + 1 < ARRAY_SIZE(values); ++i)
      ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, values[i]));

   const char *reason = nullptr;
   EXPECT_FALSE(agx_apple9_allocate_vir(&program, &reason));
   ASSERT_NE(reason, nullptr);
   EXPECT_STREQ(
      reason,
      "Apple9 no-spill allocator exhausted the compact destination bank");
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, UsesR16ThroughR63ForGeneralPressure)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t gid = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_GET_GLOBAL_ID,
                                      AGX_APPLE9_ENC_GET_SR, nullptr, 0, 0);
   uint32_t c =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IMM,
                          AGX_APPLE9_ENC_MOV_IMM_COMPACT, nullptr, 0, 1);
   uint32_t values[24];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      uint32_t sources[] = {gid, c};
      values[i] =
         agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IADD,
                             AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources, 2, 0);
   }

   uint32_t reduced = values[0];
   for (unsigned i = 1; i < ARRAY_SIZE(values); ++i) {
      uint32_t sources[] = {reduced, values[i]};
      reduced =
         agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IADD,
                             AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources, 2, 0);
   }
   program.output = reduced;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
   EXPECT_GE(program.peak_live_gprs, ARRAY_SIZE(values));
   EXPECT_GE(program.max_phys_gpr, 39u);
   EXPECT_LE(program.max_phys_gpr, 63u);
   EXPECT_TRUE(agx_apple9_validate_vir_allocation(&program, &reason)) << reason;
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, WideSelectDestinationDoesNotOverlapInputs)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);

   uint32_t sources[4];
   for (unsigned i = 0; i < 4; ++i) {
      sources[i] =
         agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IMM,
                             AGX_APPLE9_ENC_MOV_IMM_COMPACT, nullptr, 0, i + 1);
   }

   uint32_t value = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_SELECT,
                                        AGX_APPLE9_ENC_SELECT_GPR_WIDE, sources,
                                        4, AGX_APPLE9_SELECT_ULT);
   program.output = value;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
   for (unsigned i = 0; i < 4; ++i)
      EXPECT_NE(program.phys[value], program.phys[sources[i]]);
   EXPECT_TRUE(agx_apple9_validate_vir_allocation(&program, &reason)) << reason;
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, PrecoloredFragmentInputsAndOutputsShareLiveness)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t red = agx_apple9_vir_input(&program, 0);
   uint32_t green = agx_apple9_vir_input(&program, 4);
   uint32_t add_sources[] = {red, green};
   uint32_t sum =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                          AGX_APPLE9_ENC_FLOAT2_COMPACT, add_sources, 2, 0);
   uint32_t zero =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IMM,
                          AGX_APPLE9_ENC_MOV_IMM_COMPACT, nullptr, 0, 0);
   uint32_t copy_sources[] = {sum, zero};
   uint32_t output =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IOR,
                          AGX_APPLE9_ENC_LOGIC_EXTENDED, copy_sources, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_set_fixed_phys(&program, output, 0));
   program.output = output;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
   EXPECT_EQ(program.phys[red], 0u);
   EXPECT_EQ(program.phys[green], 4u);
   EXPECT_EQ(program.phys[output], 0u);
   EXPECT_NE(program.phys[sum], program.phys[red]);
   EXPECT_TRUE(agx_apple9_validate_vir_allocation(&program, &reason)) << reason;
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, NativeVectorLoadUsesOneAdjacentTupleAndOneSlot)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 4);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t vector =
      agx_apple9_vir_emit_device_load_vector(&program, 0, index, 4, &contract);
   ASSERT_NE(vector, AGX_APPLE9_VREG_INVALID);
   uint32_t sources[] = {vector + 2, vector + 2};
   uint32_t output =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_UMIN,
                          AGX_APPLE9_ENC_MINMAX_COMPACT, sources, 2, 0);
   program.output = output;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << reason;
   EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[1]->scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
   EXPECT_GE(program.phys[vector], 16u);
   for (unsigned c = 1; c < 4; ++c)
      EXPECT_EQ(program.phys[vector + c], program.phys[vector] + c);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, PendingVectorTupleCannotOverlapLaterAsyncDestination)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 4);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };

   uint32_t vector =
      agx_apple9_vir_emit_device_load_vector(&program, 0, index, 4, &contract);
   ASSERT_NE(vector, AGX_APPLE9_VREG_INVALID);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, vector, AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO));

   const agx_apple9_device_load_contract final_contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t scalar =
      agx_apple9_vir_emit_device_load(&program, 1, index, &final_contract);
   ASSERT_NE(scalar, AGX_APPLE9_VREG_INVALID);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
      &program, scalar, 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));

   uint32_t vector_sources[] = {vector, vector + 2};
   uint32_t first =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IXOR,
                          AGX_APPLE9_ENC_LOGIC_EXTENDED, vector_sources, 2, 0);
   uint32_t scalar_sources[] = {scalar, first};
   uint32_t output =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IXOR,
                          AGX_APPLE9_ENC_LOGIC_EXTENDED, scalar_sources, 2, 0);
   program.output = output;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");

   const unsigned vector_base = program.phys[vector];
   const unsigned scalar_reg = program.phys[scalar];
   EXPECT_FALSE(scalar_reg >= vector_base && scalar_reg < vector_base + 4);
   for (unsigned c = 1; c < 4; ++c)
      EXPECT_EQ(program.phys[vector + c], vector_base + c);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Vir, MemoryTupleLegalizationPreservesControlFlowTargets)
{
   /* COLLECT belongs to its memory consumer: a branch to that consumer must
    * execute the new COLLECT, while targets past it move forward. Exercise
    * both branch directions, repeated insertions, and the end boundary. */

   for (unsigned target = 0; target < 5; ++target) {
      SCOPED_TRACE(target);
      agx_apple9_vir_program program;
      agx_apple9_vir_init(&program);
      agx_apple9_block *blocks[5];
      for (auto &block : blocks)
         block = agx_apple9_block_create(&program);
      agx_apple9_block_begin(&program, blocks[0]);
      uint32_t sources[] = {agx_apple9_vir_input(&program, 2),
                            agx_apple9_vir_input(&program, 4),
                            agx_apple9_vir_input(&program, 6)};
      ASSERT_TRUE(agx_apple9_vir_emit_branch(
         &program, AGX_APPLE9_VIR_JMP_EXEC_NONE,
         AGX_APPLE9_ENC_JMP_EXEC_NONE, blocks[target]));
      for (unsigned i = 0; i < 2; ++i) {
         agx_apple9_block_begin(&program, blocks[i + 1]);
         /* Mimic scalarized sources after scoreboard legalization. */
         ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
            &program, AGX_APPLE9_VIR_DEVICE_STORE,
            AGX_APPLE9_ENC_DEVICE_STORE, sources, 3, 0));
         auto &store = *program.instructions[program.instruction_count - 1];
         store.memory_bits = 32;
         store.memory_components = 2;
      }
      agx_apple9_block_begin(&program, blocks[3]);
      ASSERT_TRUE(agx_apple9_vir_emit_branch(
         &program, AGX_APPLE9_VIR_JMP_EXEC_ANY,
         AGX_APPLE9_ENC_JMP_EXEC_ANY, blocks[target]));
      agx_apple9_block_begin(&program, blocks[4]);
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
         << (reason ? reason : "");
      ASSERT_EQ(program.instruction_count, 6u);
      EXPECT_EQ(program.instructions[0]->branch_target, blocks[target]);
      EXPECT_EQ(program.instructions[5]->branch_target, blocks[target]);
      EXPECT_EQ(program.instructions[1]->op, AGX_APPLE9_VIR_COLLECT);
      EXPECT_EQ(program.instructions[3]->op, AGX_APPLE9_VIR_COLLECT);
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Allocator, VectorStoreCollectsScalarSourcesBeforeAllocation)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t data[] = {
      agx_apple9_vir_input(&program, 2),
      agx_apple9_vir_input(&program, 4),
      agx_apple9_vir_input(&program, 6),
      agx_apple9_vir_input(&program, 8),
   };
   uint32_t index = agx_apple9_vir_input(&program, 10);
   ASSERT_TRUE(
      agx_apple9_vir_emit_device_store(&program, 0, index, data, 4, 32));
   ASSERT_EQ(program.instruction_count, 2u);
   const uint32_t tuple = program.instructions[0]->dest;
   EXPECT_EQ(program.instructions[0]->op, AGX_APPLE9_VIR_COLLECT);
   EXPECT_EQ(program.instructions[0]->dest_components, 4u);
   EXPECT_EQ(program.instructions[1]->op, AGX_APPLE9_VIR_DEVICE_STORE);
   for (unsigned c = 0; c < 4; ++c)
      EXPECT_EQ(program.instructions[1]->src[c], tuple + c);

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   for (unsigned c = 1; c < 4; ++c)
      EXPECT_EQ(program.phys[tuple + c], program.phys[tuple] + c);
   for (unsigned c = 0; c < 4; ++c)
      EXPECT_NE(program.phys[tuple + c], program.phys[data[c]]);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, StoreIndexLifetimeUsesAccessDescriptorWithoutCopy)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   uint32_t first_data = agx_apple9_vir_input(&program, 2);
   ASSERT_TRUE(
      agx_apple9_vir_emit_device_store(&program, 0, index, &first_data, 1, 32));

   uint32_t sources[] = {index, index};
   uint32_t second_data =
      agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IOR,
                          AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&program, 1, index,
                                                &second_data, 1, 32));

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(program.instruction_count, 3u);
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");

   const auto &first_store = *program.instructions[0];
   const auto &last_store = *program.instructions[2];
   EXPECT_NE(first_store.live_after_mask & (1u << 1), 0u);
   EXPECT_EQ(last_store.live_after_mask & (1u << 1), 0u);
   EXPECT_EQ(program.phys[first_store.src[1]], program.phys[last_store.src[1]]);

   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&first_store, program.phys,
                                               &packed, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(packed.bytes[6], 0x20);
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&last_store, program.phys,
                                               &packed, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(packed.bytes[6], 0x21);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, CollectCoalescesAnAlreadyAdjacentKilledTuple)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 13);
   uint32_t data[4];
   for (unsigned c = 0; c < 4; ++c) {
      data[c] =
         agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IMM,
                             AGX_APPLE9_ENC_MOV_IMM_COMPACT, nullptr, 0, c + 1);
   }
   ASSERT_TRUE(
      agx_apple9_vir_emit_device_store(&program, 0, index, data, 4, 32));
   const uint32_t tuple = program.instructions[4]->dest;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   for (unsigned c = 0; c < 4; ++c)
      EXPECT_EQ(program.phys[tuple + c], program.phys[data[c]]);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, CollectRespectsValuesLiveAcrossVectorStore)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t live = agx_apple9_vir_input(&program, 16);
   uint32_t data[] = {
      agx_apple9_vir_input(&program, 2),
      agx_apple9_vir_input(&program, 4),
      agx_apple9_vir_input(&program, 6),
      agx_apple9_vir_input(&program, 8),
   };
   uint32_t index = agx_apple9_vir_input(&program, 10);
   ASSERT_TRUE(
      agx_apple9_vir_emit_device_store(&program, 0, index, data, 4, 32));
   ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, live));
   const uint32_t tuple = program.instructions[0]->dest;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   for (unsigned c = 0; c < 4; ++c)
      EXPECT_NE(program.phys[tuple + c], program.phys[live]);
   agx_apple9_vir_finish(&program);
}

static nir_builder
apple9_compute_builder(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.num_ssbos = 1;
   return b;
}

static nir_def *
apple9_global_id_x(nir_builder *b)
{
   return nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
}

static void
apple9_store_output(nir_builder *b, nir_def *index, nir_def *value)
{
   nir_store_ssbo(b, value, nir_imm_int(b, 0), nir_imul_imm(b, index, 4));
}

static void
apple9_store_binding(nir_builder *b, unsigned binding, nir_def *index,
                     nir_def *value)
{
   nir_store_ssbo(b, value, nir_imm_int(b, binding), nir_imul_imm(b, index, 4));
}

static nir_shader *
apple9_simple_if_shader(bool with_else)
{
   nir_builder b =
      apple9_compute_builder(with_else ? "apple9_if_else" : "apple9_if");
   b.shader->info.num_ssbos = with_else ? 2 : 1;
   nir_def *gid = apple9_global_id_x(&b);
   nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, gid, 16));
   apple9_store_binding(&b, 0, gid, nir_iadd_imm(&b, gid, 0x100));
   if (with_else) {
      nir_push_else(&b, nif);
      apple9_store_binding(&b, 1, gid, nir_iadd_imm(&b, gid, 0x200));
   }
   nir_pop_if(&b, nif);
   return b.shader;
}

static nir_shader *
apple9_condition_shader(nir_op op)
{
   nir_builder b = apple9_compute_builder("apple9_condition");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *left = gid;
   nir_def *right = nir_imm_int(&b, 16);
   if (op == nir_op_feq || op == nir_op_fneu || op == nir_op_flt ||
       op == nir_op_fge) {
      left = nir_u2f32(&b, gid);
      right = nir_imm_float(&b, 16.0f);
   }

   nir_def *condition = nir_build_alu2(&b, op, left, right);
   nir_if *nif = nir_push_if(&b, condition);
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x100));
   nir_push_else(&b, nif);
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x200));
   nir_pop_if(&b, nif);
   return b.shader;
}

static nir_shader *
apple9_composed_boolean_if_shader(bool selected)
{
   nir_builder b = apple9_compute_builder("apple9_composed_boolean_if");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *low_half = nir_ult_imm(&b, gid, 16);
   nir_def *odd = nir_ine_imm(&b, nir_iand_imm(&b, gid, 1), 0);
   nir_def *condition =
      selected ? nir_bcsel(&b, odd, low_half, nir_inot(&b, low_half))
               : nir_iand(&b, low_half, odd);

   nir_if *nif = nir_push_if(&b, condition);
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x300));
   nir_push_else(&b, nif);
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x400));
   nir_pop_if(&b, nif);
   return b.shader;
}

static nir_shader *
apple9_predicate_lifetime_shader(nir_op op, unsigned live_sources,
                                 unsigned pressure_values)
{
   nir_builder b = apple9_compute_builder("apple9_predicate_lifetime");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *left = nir_iadd_imm(&b, gid, 0x10203);
   nir_def *right = nir_ixor(&b, gid, nir_imm_int(&b, 0x80004567u));

   nir_def *pressure[32];
   assert(pressure_values <= ARRAY_SIZE(pressure));
   for (unsigned i = 0; i < pressure_values; ++i) {
      pressure[i] = nir_iadd_imm(
         &b, nir_ixor(&b, gid, nir_imm_int(&b, 0x9e3779b9u * (i + 1))),
         i * 17 + 3);
   }

   nir_def *cmp_left = left;
   nir_def *cmp_right = right;
   if (op == nir_op_feq || op == nir_op_fneu || op == nir_op_flt ||
       op == nir_op_fge) {
      cmp_left = nir_u2f32(&b, left);
      cmp_right = nir_u2f32(&b, right);
   }

   nir_if *nif = nir_push_if(&b, nir_build_alu2(&b, op, cmp_left, cmp_right));
   apple9_store_binding(&b, 0, gid, nir_iadd_imm(&b, gid, 0x100));
   nir_push_else(&b, nif);
   apple9_store_binding(&b, 0, gid, nir_iadd_imm(&b, gid, 0x200));
   nir_pop_if(&b, nif);

   nir_def *after = nir_ixor(&b, gid, nir_imm_int(&b, 0xa5a55a5a));
   if (live_sources & BITFIELD_BIT(0))
      after = nir_iadd(&b, after, cmp_left);
   if (live_sources & BITFIELD_BIT(1))
      after = nir_ixor(&b, after, cmp_right);
   for (unsigned i = 0; i < pressure_values; ++i)
      after = nir_iadd(&b, after, pressure[i]);
   apple9_store_binding(&b, 1, gid, after);
   return b.shader;
}

enum apple9_region_shape {
   APPLE9_REGION_EMPTY,
   APPLE9_REGION_THEN_ONLY,
   APPLE9_REGION_ELSE_ONLY,
   APPLE9_REGION_BOTH,
};

static nir_shader *
apple9_single_region_shader(enum apple9_region_shape shape)
{
   nir_builder b = apple9_compute_builder("apple9_single_region");
   b.shader->info.num_ssbos = 4;
   nir_def *gid = apple9_global_id_x(&b);

   apple9_store_binding(&b, 0, gid,
                        nir_ixor(&b, gid, nir_imm_int(&b, 0x11111111)));
   nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, gid, 16));
   if (shape == APPLE9_REGION_THEN_ONLY || shape == APPLE9_REGION_BOTH)
      apple9_store_binding(&b, 1, gid, nir_iadd_imm(&b, gid, 0x22220000));
   nir_push_else(&b, nif);
   if (shape == APPLE9_REGION_ELSE_ONLY || shape == APPLE9_REGION_BOTH)
      apple9_store_binding(&b, 2, gid, nir_iadd_imm(&b, gid, 0x33330000));
   nir_pop_if(&b, nif);
   apple9_store_binding(&b, 3, gid,
                        nir_ixor(&b, gid, nir_imm_int(&b, 0x44444444)));
   return b.shader;
}

static nir_shader *
apple9_multiple_phi_shader()
{
   nir_builder b = apple9_compute_builder("apple9_multiple_phi");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, gid, 16));
   nir_def *then_scalar = nir_iadd_imm(&b, gid, 0x100);
   nir_def *then_vector =
      nir_vec4(&b, nir_iadd_imm(&b, gid, 1), nir_iadd_imm(&b, gid, 2),
               nir_iadd_imm(&b, gid, 3), nir_iadd_imm(&b, gid, 4));
   nir_push_else(&b, nif);
   nir_def *else_scalar = nir_iadd_imm(&b, gid, 0x200);
   nir_def *else_vector = nir_vec4(&b, nir_ixor(&b, gid, nir_imm_int(&b, 0x10)),
                                   nir_ixor(&b, gid, nir_imm_int(&b, 0x20)),
                                   nir_ixor(&b, gid, nir_imm_int(&b, 0x30)),
                                   nir_ixor(&b, gid, nir_imm_int(&b, 0x40)));
   nir_pop_if(&b, nif);

   nir_def *scalar = nir_if_phi(&b, then_scalar, else_scalar);
   nir_def *vector = nir_if_phi(&b, then_vector, else_vector);
   apple9_store_binding(&b, 0, gid, scalar);
   nir_store_ssbo(&b, vector, nir_imm_int(&b, 1), nir_imul_imm(&b, gid, 16));
   return b.shader;
}

static nir_shader *
apple9_simple_phi_shader()
{
   nir_builder b = apple9_compute_builder("apple9_if_phi");
   b.shader->info.num_ssbos = 1;
   nir_def *gid = apple9_global_id_x(&b);
   nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, gid, 16));
   nir_def *if_true = nir_iadd_imm(&b, gid, 0x100);
   nir_push_else(&b, nif);
   nir_def *if_false = nir_iadd_imm(&b, gid, 0x200);
   nir_pop_if(&b, nif);
   nir_def *selected = nir_if_phi(&b, if_true, if_false);
   apple9_store_output(&b, gid, selected);
   return b.shader;
}

static nir_shader *
apple9_nested_if_shader()
{
   nir_builder b = apple9_compute_builder("apple9_nested_if");
   nir_def *gid = apple9_global_id_x(&b);

   nir_if *outer = nir_push_if(&b, nir_ult_imm(&b, gid, 16));
   nir_if *then_inner =
      nir_push_if(&b, nir_ult_imm(&b, nir_iand_imm(&b, gid, 8), 1));
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x100));
   nir_push_else(&b, then_inner);
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x200));
   nir_pop_if(&b, then_inner);

   nir_push_else(&b, outer);
   nir_if *else_inner = nir_push_if(&b, nir_ult_imm(&b, gid, 24));
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x300));
   nir_push_else(&b, else_inner);
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x400));
   nir_pop_if(&b, else_inner);
   nir_pop_if(&b, outer);
   return b.shader;
}

static nir_shader *
apple9_deep_if_shader(unsigned depth)
{
   nir_builder b = apple9_compute_builder("apple9_deep_if");
   nir_def *gid = apple9_global_id_x(&b);
   nir_if *nested[32];

   assert(depth <= ARRAY_SIZE(nested));
   for (unsigned i = 0; i < depth; ++i) {
      nested[i] = nir_push_if(&b, nir_ult_imm(&b, gid, 32 - i));
      apple9_store_output(&b, nir_iadd_imm(&b, nir_ishl_imm(&b, gid, 5), i),
                          nir_iadd_imm(&b, gid, i));
   }

   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x100));
   for (unsigned i = depth; i > 0; --i)
      nir_pop_if(&b, nested[i - 1]);

   return b.shader;
}

static nir_shader *
apple9_nested_phi_shader()
{
   nir_builder b = apple9_compute_builder("apple9_nested_phi");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);

   nir_if *outer = nir_push_if(&b, nir_ult_imm(&b, gid, 20));
   nir_if *inner =
      nir_push_if(&b, nir_ult_imm(&b, nir_iand_imm(&b, gid, 3), 2));
   nir_def *inner_then =
      nir_vec4(&b, nir_iadd_imm(&b, gid, 1), nir_iadd_imm(&b, gid, 2),
               nir_iadd_imm(&b, gid, 3), nir_iadd_imm(&b, gid, 4));
   nir_push_else(&b, inner);
   nir_def *inner_else = nir_vec4(&b, nir_ixor(&b, gid, nir_imm_int(&b, 0x10)),
                                  nir_ixor(&b, gid, nir_imm_int(&b, 0x20)),
                                  nir_ixor(&b, gid, nir_imm_int(&b, 0x30)),
                                  nir_ixor(&b, gid, nir_imm_int(&b, 0x40)));
   nir_pop_if(&b, inner);
   nir_def *inner_value = nir_if_phi(&b, inner_then, inner_else);
   nir_def *outer_then = nir_iadd_imm(&b, inner_value, 0x1000);

   nir_push_else(&b, outer);
   nir_def *outer_else =
      nir_vec4(&b, nir_iadd_imm(&b, gid, 0x51), nir_iadd_imm(&b, gid, 0x52),
               nir_iadd_imm(&b, gid, 0x53), nir_iadd_imm(&b, gid, 0x54));
   nir_pop_if(&b, outer);
   nir_def *value = nir_if_phi(&b, outer_then, outer_else);

   nir_store_ssbo(&b, value, nir_imm_int(&b, 0), nir_imul_imm(&b, gid, 16));
   apple9_store_binding(
      &b, 1, gid,
      nir_ixor(&b, nir_channel(&b, value, 0), nir_channel(&b, value, 3)));
   return b.shader;
}

static nir_shader *
apple9_short_circuit_shader(bool is_or)
{
   nir_builder b =
      apple9_compute_builder(is_or ? "apple9_short_or" : "apple9_short_and");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *a = nir_ult_imm(&b, gid, 16);

   if (!is_or) {
      nir_if *outer = nir_push_if(&b, a);
      apple9_store_binding(&b, 1, gid, nir_iadd_imm(&b, gid, 0x500));
      nir_def *b_value = nir_ine_imm(&b, nir_iand_imm(&b, gid, 1), 0);
      nir_if *inner = nir_push_if(&b, b_value);
      apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x600));
      nir_pop_if(&b, inner);
      nir_pop_if(&b, outer);
   } else {
      nir_if *lhs = nir_push_if(&b, a);
      nir_def *then_value = nir_imm_true(&b);
      nir_push_else(&b, lhs);
      apple9_store_binding(&b, 1, gid, nir_iadd_imm(&b, gid, 0x700));
      nir_def *else_value = nir_ine_imm(&b, nir_iand_imm(&b, gid, 1), 0);
      nir_pop_if(&b, lhs);
      nir_def *value = nir_if_phi(&b, then_value, else_value);

      nir_if *result = nir_push_if(&b, value);
      apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x800));
      nir_pop_if(&b, result);
   }

   return b.shader;
}

static nir_shader *
apple9_counted_loop_shader(bool with_continue)
{
   nir_builder b = apple9_compute_builder(
      with_continue ? "apple9_loop_continue" : "apple9_counted_loop");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *limit = nir_iadd_imm(&b, nir_iand_imm(&b, gid, 7), 1);
   nir_def *initial = nir_imm_int(&b, 0);
   apple9_store_output(&b, gid, nir_imm_int(&b, 0xfeed0000));

   nir_loop *loop = nir_push_loop(&b);
   if (with_continue)
      nir_loop_add_continue_construct(loop);
   nir_block *entry = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_block *header = nir_loop_first_block(loop);
   nir_phi_instr *phi = nir_phi_instr_create(b.shader);
   nir_def_init(&phi->instr, &phi->def, 1, 32);
   nir_phi_instr_add_src(phi, entry, initial);

   nir_def *header_value =
      nir_iadd(&b, nir_imul_imm(&b, &phi->def, 0x101), gid);
   nir_break_if(&b, nir_uge(&b, &phi->def, limit));
   if (with_continue) {
      nir_if *skip =
         nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, &phi->def, 1), 0));
      nir_jump(&b, nir_jump_continue);
      nir_pop_if(&b, skip);
   }

   /* Deliberately reuse a value formed in the loop-test block after the
    * natural break. Its durable register must be refreshed at the latch. */
   nir_def *value = nir_iadd(&b, header_value,
                             nir_ixor(&b, gid, nir_imm_int(&b, 0x12340000)));
   apple9_store_output(&b, gid, value);

   if (with_continue)
      nir_push_continue(&b, loop);
   nir_def *next = nir_iadd_imm(&b, &phi->def, 1);
   nir_phi_instr_add_src(phi, nir_cursor_current_block(b.cursor), next);
   nir_pop_loop(&b, loop);

   b.cursor = nir_after_phis(header);
   nir_builder_instr_insert(&b, &phi->instr);
   nir_validate_shader(b.shader, "Apple9 counted loop test");
   return b.shader;
}

static nir_shader *
apple9_nested_loop_shader()
{
   nir_builder b = apple9_compute_builder("apple9_nested_loops");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *outer_initial = nir_imm_int(&b, 0);
   apple9_store_output(&b, gid, nir_imm_int(&b, 0));

   nir_loop *outer = nir_push_loop(&b);
   nir_block *outer_entry =
      nir_cf_node_as_block(nir_cf_node_prev(&outer->cf_node));
   nir_block *outer_header = nir_loop_first_block(outer);
   nir_phi_instr *outer_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&outer_phi->instr, &outer_phi->def, 1, 32);
   nir_phi_instr_add_src(outer_phi, outer_entry, outer_initial);
   nir_break_if(&b, nir_uge_imm(&b, &outer_phi->def, 3));

   nir_def *inner_initial = nir_imm_int(&b, 0);
   nir_loop *inner = nir_push_loop(&b);
   nir_block *inner_entry =
      nir_cf_node_as_block(nir_cf_node_prev(&inner->cf_node));
   nir_block *inner_header = nir_loop_first_block(inner);
   nir_phi_instr *inner_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&inner_phi->instr, &inner_phi->def, 1, 32);
   nir_phi_instr_add_src(inner_phi, inner_entry, inner_initial);
   nir_break_if(&b, nir_uge_imm(&b, &inner_phi->def, 2));
   apple9_store_output(
      &b, gid,
      nir_iadd(&b, nir_imul_imm(&b, &outer_phi->def, 16), &inner_phi->def));
   nir_def *inner_next = nir_iadd_imm(&b, &inner_phi->def, 1);
   nir_phi_instr_add_src(inner_phi, nir_cursor_current_block(b.cursor),
                         inner_next);
   nir_pop_loop(&b, inner);
   nir_cursor after_inner = b.cursor;
   b.cursor = nir_after_phis(inner_header);
   nir_builder_instr_insert(&b, &inner_phi->instr);
   b.cursor = after_inner;

   nir_def *outer_next = nir_iadd_imm(&b, &outer_phi->def, 1);
   nir_phi_instr_add_src(outer_phi, nir_cursor_current_block(b.cursor),
                         outer_next);
   nir_pop_loop(&b, outer);
   b.cursor = nir_after_phis(outer_header);
   nir_builder_instr_insert(&b, &outer_phi->instr);
   nir_validate_shader(b.shader, "Apple9 nested loop test");
   return b.shader;
}

static nir_shader *
apple9_general_break_loop_shader()
{
   nir_builder b = apple9_compute_builder("apple9_general_break_loop");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *initial = nir_imm_int(&b, 0);
   nir_def *limit = nir_iadd_imm(&b, nir_iand_imm(&b, gid, 7), 2);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *entry = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_block *header = nir_loop_first_block(loop);
   nir_phi_instr *phi = nir_phi_instr_create(b.shader);
   nir_def_init(&phi->instr, &phi->def, 1, 32);
   nir_phi_instr_add_src(phi, entry, initial);
   nir_break_if(&b, nir_uge(&b, &phi->def, limit));

   nir_def *selector = nir_iand_imm(&b, nir_iadd(&b, &phi->def, gid), 3);
   nir_if *conditional = nir_push_if(&b, nir_ieq_imm(&b, selector, 1));
   /* Observable work before the jump keeps this as a general nested break,
    * rather than the direct-break lowering used for `if (condition) break`. */
   apple9_store_output(&b, gid,
                       nir_ixor(&b, &phi->def, nir_imm_int(&b, 0x7b000000)));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, conditional);

   nir_def *next = nir_iadd_imm(&b, &phi->def, 1);
   nir_phi_instr_add_src(phi, nir_cursor_current_block(b.cursor), next);
   nir_pop_loop(&b, loop);
   b.cursor = nir_after_phis(header);
   nir_builder_instr_insert(&b, &phi->instr);
   nir_validate_shader(b.shader, "Apple9 general loop-break test");
   return b.shader;
}

/* Put the terminating edge between two unrelated conditional regions.  This
 * is intentionally not the source-shaped "condition at the header or latch"
 * pattern recognized by the old Apple9 loop matcher: the backend must walk
 * the structured NIR loop itself and lower the break wherever it occurs. */
static nir_shader *
apple9_mid_body_break_loop_shader()
{
   nir_builder b = apple9_compute_builder("apple9_mid_body_break_loop");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *initial = nir_imm_int(&b, 0);
   nir_def *limit = nir_iadd_imm(&b, nir_iand_imm(&b, gid, 3), 1);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *entry = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_block *header = nir_loop_first_block(loop);
   nir_phi_instr *phi = nir_phi_instr_create(b.shader);
   nir_def_init(&phi->instr, &phi->def, 1, 32);
   nir_phi_instr_add_src(phi, entry, initial);

   nir_if *prefix =
      nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, gid, 1), 0));
   apple9_store_output(&b, gid, nir_iadd_imm(&b, &phi->def, 0x100));
   nir_pop_if(&b, prefix);

   nir_break_if(&b, nir_uge(&b, &phi->def, limit));

   nir_if *suffix = nir_push_if(
      &b, nir_ine_imm(&b, nir_iand_imm(&b, nir_iadd(&b, gid, &phi->def), 2),
                      0));
   apple9_store_output(&b, gid, nir_iadd_imm(&b, &phi->def, 0x200));
   nir_pop_if(&b, suffix);

   nir_def *next = nir_iadd_imm(&b, &phi->def, 1);
   nir_phi_instr_add_src(phi, nir_cursor_current_block(b.cursor), next);
   nir_pop_loop(&b, loop);
   b.cursor = nir_after_phis(header);
   nir_builder_instr_insert(&b, &phi->instr);
   nir_validate_shader(b.shader, "Apple9 mid-body loop-break test");
   return b.shader;
}

enum apple9_conditional_load_shape {
   APPLE9_CONDITIONAL_LOAD_THEN_ONLY,
   APPLE9_CONDITIONAL_LOAD_BOTH_ARMS,
   APPLE9_CONDITIONAL_LOAD_FANOUT_AND_MERGE,
};

static nir_shader *
apple9_conditional_load_shader(enum apple9_conditional_load_shape shape)
{
   nir_builder b = apple9_compute_builder("apple9_conditional_load");
   b.shader->info.num_ssbos = shape == APPLE9_CONDITIONAL_LOAD_THEN_ONLY   ? 3
                              : shape == APPLE9_CONDITIONAL_LOAD_BOTH_ARMS ? 4
                                                                           : 6;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *offset = nir_imul_imm(&b, gid, 4);
   nir_def *condition = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), offset,
                                      .access = ACCESS_NON_WRITEABLE);

   nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, condition, 0x80000000u));
   nir_def *then_index =
      nir_iand_imm(&b, nir_iadd_imm(&b, nir_imul_imm(&b, gid, 5), 3), 63);
   nir_def *then_offset = nir_imul_imm(&b, then_index, 4);
   nir_def *then_value =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 2), then_offset,
                    .access = ACCESS_NON_WRITEABLE);
   if (shape == APPLE9_CONDITIONAL_LOAD_FANOUT_AND_MERGE)
      apple9_store_binding(&b, 5, gid, nir_ixor(&b, then_value, gid));
   then_value = nir_iadd_imm(&b, then_value, 0x13579bdfu);

   nir_push_else(&b, nif);
   nir_def *else_value = nir_iadd_imm(&b, gid, 0x2468ace0u);
   if (shape != APPLE9_CONDITIONAL_LOAD_THEN_ONLY) {
      nir_def *else_index =
         nir_iand_imm(&b, nir_iadd_imm(&b, nir_imul_imm(&b, gid, 7), 11), 63);
      nir_def *else_offset = nir_imul_imm(&b, else_index, 4);
      else_value = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 3), else_offset,
                                 .access = ACCESS_NON_WRITEABLE);
      if (shape == APPLE9_CONDITIONAL_LOAD_FANOUT_AND_MERGE)
         apple9_store_binding(&b, 5, gid, nir_iadd(&b, else_value, gid));
      else_value = nir_ixor(&b, else_value, nir_imm_int(&b, 0xa5a55a5au));
   }

   nir_pop_if(&b, nif);
   nir_def *merged = nir_if_phi(&b, then_value, else_value);
   if (shape == APPLE9_CONDITIONAL_LOAD_FANOUT_AND_MERGE) {
      nir_def *post_index =
         nir_iand_imm(&b, nir_iadd_imm(&b, nir_imul_imm(&b, gid, 13), 17), 63);
      nir_def *post_offset = nir_imul_imm(&b, post_index, 4);
      nir_def *post = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 4), post_offset,
                                    .access = ACCESS_NON_WRITEABLE);
      merged = nir_ixor(&b, merged, post);
   }
   apple9_store_output(&b, gid, merged);
   return b.shader;
}

static unsigned
apple9_binary_count_sequence(const struct agx_shader_part *compiled,
                             const uint8_t *sequence, unsigned length)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   unsigned count = 0;
   for (unsigned i = 0; i + length <= compiled->info.binary_size; ++i)
      count += memcmp(binary + i, sequence, length) == 0;
   return count;
}

static unsigned
apple9_binary_find_sequence(const struct agx_shader_part *compiled,
                            const uint8_t *sequence, unsigned length)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   for (unsigned i = 0; i + length <= compiled->info.binary_size; ++i) {
      if (memcmp(binary + i, sequence, length) == 0)
         return i;
   }
   return UINT_MAX;
}

static unsigned
apple9_binary_count_exec_pushes(const struct agx_shader_part *compiled)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   unsigned count = 0;
   for (unsigned i = 0; i + 4 <= compiled->info.binary_size; ++i) {
      if (binary[i] == 0x0f && binary[i + 1] == 0x05 && binary[i + 2] == 0x54 &&
          (binary[i + 3] & 3) == 1)
         ++count;
   }
   return count;
}

static unsigned
apple9_binary_device_load_offsets(const struct agx_shader_part *compiled,
                                  unsigned *offsets, unsigned capacity)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   unsigned count = 0;
   for (unsigned i = 0; i + 14 <= compiled->info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      if (bytes[0] != 0x67 || bytes[6] != 0x20 || bytes[7] != 0x00 ||
          (bytes[11] & 0xe0) != 0x40)
         continue;
      if (count < capacity)
         offsets[count] = i;
      ++count;
      i += 13;
   }
   return count;
}

static nir_shader *
apple9_constant_store_shader(uint32_t value)
{
   nir_builder b = apple9_compute_builder("apple9_constant_store");
   nir_def *gid = apple9_global_id_x(&b);
   apple9_store_output(&b, gid, nir_imm_int(&b, value));
   return b.shader;
}

static nir_shader *
apple9_large_constant_add_shader()
{
   nir_builder b = apple9_compute_builder("apple9_large_constant_add");
   nir_def *gid = apple9_global_id_x(&b);
   apple9_store_output(&b, gid, nir_iadd_imm(&b, gid, 0x12345678));
   return b.shader;
}

static nir_shader *
apple9_ssbo_reduce_shader(unsigned input_count, bool floating,
                          enum gl_access_qualifier access = ACCESS_NON_WRITEABLE)
{
   nir_builder b = apple9_compute_builder("apple9_ssbo_reduce");
   b.shader->info.num_ssbos = input_count + 1;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *offset = nir_imul_imm(&b, gid, 4);
   nir_def *value = floating ? nir_imm_float(&b, 1.0f) : nir_imm_int(&b, 1);

   for (unsigned binding = 1; binding <= input_count; ++binding) {
      nir_def *loaded = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, binding),
                                      offset, .access = access);
      value =
         floating ? nir_fadd(&b, value, loaded) : nir_iadd(&b, value, loaded);
   }

   nir_store_ssbo(&b, value, nir_imm_int(&b, 0), offset);
   return b.shader;
}

static nir_shader *
apple9_arbitrary_integer_shader()
{
   nir_builder b = apple9_compute_builder("apple9_arbitrary_integer");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *a = nir_imul_imm(&b, gid, 3);
   nir_def *mixed = nir_ixor(&b, a, nir_iadd_imm(&b, gid, 7));
   apple9_store_output(&b, gid, nir_iadd_imm(&b, mixed, 11));
   return b.shader;
}

static nir_shader *
apple9_arbitrary_float_shader()
{
   nir_builder b = apple9_compute_builder("apple9_arbitrary_float");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *a = nir_fmul(&b, gid, nir_imm_float(&b, 1.25f));
   nir_def *c = nir_fadd(&b, gid, nir_imm_float(&b, 7.5f));
   apple9_store_output(&b, gid, nir_ffma(&b, a, c, nir_imm_float(&b, -3.0f)));
   return b.shader;
}

enum apple9_reciprocal_shape {
   APPLE9_RECIPROCAL_DIRECT_STORE,
   APPLE9_RECIPROCAL_RETAIN_SOURCE,
   APPLE9_RECIPROCAL_MATERIALIZED_SOURCE,
   APPLE9_RECIPROCAL_RESULT_FANOUT,
};

static nir_shader *
apple9_reciprocal_shader(enum apple9_reciprocal_shape shape)
{
   nir_builder b = apple9_compute_builder("apple9_reciprocal");
   b.shader->info.num_ssbos =
      shape == APPLE9_RECIPROCAL_MATERIALIZED_SOURCE ? 3 : 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *offset = nir_imul_imm(&b, gid, 4);
   nir_def *x = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), offset,
                              .access = ACCESS_NON_WRITEABLE);
   nir_def *source = x;
   if (shape == APPLE9_RECIPROCAL_MATERIALIZED_SOURCE) {
      nir_def *y = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 2), offset,
                                 .access = ACCESS_NON_WRITEABLE);
      source = nir_fadd(&b, x, y);
   }

   nir_def *reciprocal = nir_frcp(&b, source);
   if (shape == APPLE9_RECIPROCAL_RESULT_FANOUT) {
      /* Make the first use a store and the later use an ALU operation.  The
       * native result hint describes the complete lifetime, not merely the
       * first consumer. */
      apple9_store_output(&b, gid, reciprocal);
      apple9_store_output(&b, nir_iadd_imm(&b, gid, 64),
                          nir_fadd(&b, reciprocal, source));
      return b.shader;
   }

   nir_def *result = shape == APPLE9_RECIPROCAL_RETAIN_SOURCE
                        ? nir_fadd(&b, reciprocal, source)
                        : reciprocal;
   apple9_store_output(&b, gid, result);
   return b.shader;
}

static nir_shader *
apple9_vector_load_shader()
{
   nir_builder b = apple9_compute_builder("apple9_vector_load");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *loaded =
      nir_load_ssbo(&b, 4, 32, nir_imm_int(&b, 1), nir_imul_imm(&b, gid, 16),
                    .access = ACCESS_NON_WRITEABLE, .align_mul = 16);
   nir_def *low =
      nir_ixor(&b, nir_channel(&b, loaded, 2), nir_channel(&b, loaded, 0));
   nir_def *high =
      nir_iand(&b, nir_channel(&b, loaded, 3), nir_channel(&b, loaded, 1));
   apple9_store_output(&b, gid, nir_iadd(&b, low, high));
   return b.shader;
}

static nir_shader *
apple9_vector_copy_shader(unsigned components)
{
   nir_builder b = apple9_compute_builder("apple9_vector_copy");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   const unsigned stride = components == 2 ? 8 : 16;
   nir_def *offset = nir_imul_imm(&b, gid, stride);
   nir_def *loaded =
      nir_load_ssbo(&b, components, 32, nir_imm_int(&b, 1), offset,
                    .access = ACCESS_NON_WRITEABLE, .align_mul = stride);
   nir_store_ssbo(&b, loaded, nir_imm_int(&b, 0), offset);
   return b.shader;
}

static nir_shader *
apple9_vector_alu_store_shader()
{
   nir_builder b = apple9_compute_builder("apple9_vector_alu_store");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *offset = nir_imul_imm(&b, gid, 16);
   nir_def *loaded =
      nir_load_ssbo(&b, 4, 32, nir_imm_int(&b, 1), offset,
                    .access = ACCESS_NON_WRITEABLE, .align_mul = 16);
   nir_def *lanes[4];
   for (unsigned c = 0; c < 4; ++c)
      lanes[c] = nir_iadd_imm(&b, nir_channel(&b, loaded, c), c + 1);
   nir_store_ssbo(&b, nir_vec(&b, lanes, 4), nir_imm_int(&b, 0), offset);
   return b.shader;
}

static nir_shader *
apple9_multiple_stores_one_binding_shader()
{
   nir_builder b = apple9_compute_builder("apple9_multiple_stores_one_binding");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *base = nir_imul_imm(&b, gid, 8);
   nir_store_ssbo(&b, nir_iadd_imm(&b, gid, 11), nir_imm_int(&b, 0), base);
   nir_store_ssbo(&b, nir_ixor(&b, gid, nir_imm_int(&b, 0x55)),
                  nir_imm_int(&b, 0), nir_iadd_imm(&b, base, 4));
   return b.shader;
}

static nir_shader *
apple9_multiple_output_bindings_shader(bool alias_input)
{
   nir_builder b = apple9_compute_builder("apple9_multiple_output_bindings");
   b.shader->info.num_ssbos = alias_input ? 2 : 3;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *offset = nir_imul_imm(&b, gid, 4);
   const unsigned input_binding = alias_input ? 0 : 2;
   nir_def *loaded =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, input_binding), offset,
                    .access = alias_input ? (enum gl_access_qualifier)0
                                          : ACCESS_NON_WRITEABLE);
   nir_store_ssbo(&b, nir_iadd_imm(&b, loaded, 10), nir_imm_int(&b, 0), offset);
   nir_store_ssbo(&b, nir_ixor(&b, loaded, nir_imm_int(&b, 0x55)),
                  nir_imm_int(&b, 1), offset);
   return b.shader;
}

static nir_shader *
apple9_multiple_scalar_vector_stores_shader()
{
   nir_builder b =
      apple9_compute_builder("apple9_multiple_scalar_vector_stores");
   b.shader->info.num_ssbos = 3;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *loaded =
      nir_load_ssbo(&b, 4, 32, nir_imm_int(&b, 2), nir_imul_imm(&b, gid, 16),
                    .access = ACCESS_NON_WRITEABLE, .align_mul = 16);
   nir_def *lanes[4];
   for (unsigned c = 0; c < 4; ++c)
      lanes[c] = nir_iadd_imm(&b, nir_channel(&b, loaded, c), c + 1);
   nir_store_ssbo(&b, nir_vec(&b, lanes, 4), nir_imm_int(&b, 0),
                  nir_imul_imm(&b, gid, 16));
   nir_store_ssbo(
      &b, nir_ixor(&b, nir_channel(&b, loaded, 0), nir_channel(&b, loaded, 3)),
      nir_imm_int(&b, 1), nir_imul_imm(&b, gid, 4));
   return b.shader;
}

static nir_shader *
apple9_narrow_load_shader(unsigned bits, bool sign_extend)
{
   nir_builder b = apple9_compute_builder("apple9_narrow_load");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *offset = nir_imul_imm(&b, gid, bits / 8);
   nir_def *loaded =
      nir_load_ssbo(&b, 1, bits, nir_imm_int(&b, 1), offset,
                    .access = ACCESS_NON_WRITEABLE, .align_mul = bits / 8);
   nir_def *extended =
      sign_extend ? nir_i2i32(&b, loaded) : nir_u2u32(&b, loaded);
   apple9_store_output(&b, gid, extended);
   return b.shader;
}

static nir_shader *
apple9_narrow_store_shader(unsigned bits)
{
   nir_builder b = apple9_compute_builder("apple9_narrow_store");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *loaded =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), nir_imul_imm(&b, gid, 4),
                    .access = ACCESS_NON_WRITEABLE);
   nir_def *narrowed = bits == 8 ? nir_u2u8(&b, loaded) : nir_u2u16(&b, loaded);
   nir_store_ssbo(&b, narrowed, nir_imm_int(&b, 0),
                  nir_imul_imm(&b, gid, bits / 8));
   return b.shader;
}

static nir_shader *
apple9_ubo_load_shader()
{
   nir_builder b = apple9_compute_builder("apple9_ubo_load");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *element = nir_iand_imm(&b, nir_iadd_imm(&b, gid, 7), 63);
   nir_def *loaded =
      nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imul_imm(&b, element, 4),
                   .align_mul = 4, .range = 64 * sizeof(uint32_t));
   apple9_store_output(&b, gid, nir_ixor(&b, loaded, nir_imm_int(&b, 0x5a)));
   return b.shader;
}

static nir_shader *
apple9_nested_dependent_load_shader()
{
   nir_builder b = apple9_compute_builder("apple9_nested_dependent_load");
   b.shader->info.num_ssbos = 4;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *linear_offset = nir_imul_imm(&b, gid, 4);
   nir_def *root = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), linear_offset,
                                 .access = ACCESS_NON_WRITEABLE);
   nir_def *j = nir_iand_imm(&b, nir_ixor(&b, root, gid), 63);
   nir_def *middle =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 2), nir_imul_imm(&b, j, 4),
                    .access = ACCESS_NON_WRITEABLE);
   nir_def *affine = nir_iadd_imm(&b, nir_imul_imm(&b, gid, 3), 1);
   nir_def *k = nir_iand_imm(&b, nir_ixor(&b, middle, affine), 63);
   nir_def *leaf =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 3), nir_imul_imm(&b, k, 4),
                    .access = ACCESS_NON_WRITEABLE);
   apple9_store_output(&b, gid, nir_ixor(&b, nir_ixor(&b, leaf, root), middle));
   return b.shader;
}

static nir_shader *
apple9_dynamic_scatter_shader()
{
   nir_builder b = apple9_compute_builder("apple9_dynamic_scatter");
   b.shader->info.num_ssbos = 3;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *linear_offset = nir_imul_imm(&b, gid, 4);
   nir_def *loaded_index =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), linear_offset,
                    .access = ACCESS_NON_WRITEABLE);
   nir_def *scatter_index = nir_iand_imm(&b, loaded_index, 7);
   nir_def *value = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 2),
                                  nir_imul_imm(&b, scatter_index, 4),
                                  .access = ACCESS_NON_WRITEABLE);
   nir_def *result = nir_iadd(&b, value, nir_imul_imm(&b, gid, 17));
   nir_store_ssbo(&b, result, nir_imm_int(&b, 0),
                  nir_imul_imm(&b, scatter_index, 4));
   return b.shader;
}

static nir_shader *
apple9_procedural_scatter_shader()
{
   nir_builder b = apple9_compute_builder("apple9_procedural_scatter");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *scatter_index = nir_iand_imm(&b, nir_imul_imm(&b, gid, 3), 7);
   nir_store_ssbo(&b, nir_iadd_imm(&b, gid, 100), nir_imm_int(&b, 0),
                  nir_imul_imm(&b, scatter_index, 4));
   return b.shader;
}

static nir_shader *
apple9_unbounded_scatter_shader()
{
   nir_builder b = apple9_compute_builder("apple9_unbounded_scatter");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *loaded_index =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), nir_imul_imm(&b, gid, 4),
                    .access = ACCESS_NON_WRITEABLE);
   nir_store_ssbo(&b, nir_iadd_imm(&b, gid, 100), nir_imm_int(&b, 0),
                  nir_imul_imm(&b, loaded_index, 4));
   return b.shader;
}

static nir_shader *
apple9_variable_shift_shader(nir_op op)
{
   nir_builder b = apple9_compute_builder("apple9_variable_shift");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *value =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), nir_imul_imm(&b, gid, 4),
                    .access = ACCESS_NON_WRITEABLE);
   nir_def *shifted = nir_build_alu(&b, op, value, gid, nullptr, nullptr);
   apple9_store_output(&b, gid, shifted);
   return b.shader;
}

static nir_shader *
apple9_system_load_index_shader()
{
   nir_builder b = apple9_compute_builder("apple9_system_load_index");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *local = nir_load_local_invocation_index(&b);
   nir_def *direct =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1), nir_imul_imm(&b, local, 4),
                    .access = ACCESS_NON_WRITEABLE);
   nir_def *derived_index = nir_iand_imm(&b, nir_iadd_imm(&b, local, 17), 63);
   nir_def *derived = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1),
                                    nir_imul_imm(&b, derived_index, 4),
                                    .access = ACCESS_NON_WRITEABLE);
   apple9_store_output(&b, gid, nir_ixor(&b, direct, derived));
   return b.shader;
}

enum apple9_test_system_value {
   APPLE9_TEST_LOCAL_ID,
   APPLE9_TEST_LOCAL_INDEX,
   APPLE9_TEST_WORKGROUP_ID,
   APPLE9_TEST_WORKGROUP_SIZE,
   APPLE9_TEST_SUBGROUP_INVOCATION,
   APPLE9_TEST_SUBGROUP_ID,
   APPLE9_TEST_SUBGROUP_SIZE,
};

static nir_shader *
apple9_system_value_shader(enum apple9_test_system_value system,
                           unsigned component)
{
   nir_builder b = apple9_compute_builder("apple9_system_value");
   nir_def *value = nullptr;
   switch (system) {
   case APPLE9_TEST_LOCAL_ID:
      value = nir_channel(&b, nir_load_local_invocation_id(&b), component);
      break;
   case APPLE9_TEST_LOCAL_INDEX:
      value = nir_load_local_invocation_index(&b);
      break;
   case APPLE9_TEST_WORKGROUP_ID:
      value = nir_channel(&b, nir_load_workgroup_id(&b), component);
      break;
   case APPLE9_TEST_WORKGROUP_SIZE:
      value = nir_channel(&b, nir_load_workgroup_size(&b), component);
      break;
   case APPLE9_TEST_SUBGROUP_INVOCATION:
      value = nir_load_subgroup_invocation(&b);
      break;
   case APPLE9_TEST_SUBGROUP_ID:
      value = nir_load_subgroup_id(&b);
      break;
   case APPLE9_TEST_SUBGROUP_SIZE:
      value = nir_load_subgroup_size(&b);
      break;
   }
   apple9_store_output(&b, apple9_global_id_x(&b), value);
   return b.shader;
}

static nir_shader *
apple9_num_workgroups_shader(bool variable_local_size, unsigned component,
                             bool atomic = false)
{
   nir_builder b = apple9_compute_builder("apple9_num_workgroups");
   if (variable_local_size) {
      b.shader->info.workgroup_size_variable = true;
      memset(b.shader->info.workgroup_size, 0,
             sizeof(b.shader->info.workgroup_size));
   } else {
      b.shader->info.workgroup_size[0] = 7;
      b.shader->info.workgroup_size[1] = 5;
      b.shader->info.workgroup_size[2] = 3;
   }

   nir_def *groups = nir_load_num_workgroups(&b);
   nir_def *value = nir_channel(&b, groups, component);
   if (atomic) {
      b.shader->info.num_ssbos = 2;
      value = nir_ssbo_atomic(&b, 32, nir_imm_int(&b, 1), nir_imm_int(&b, 0),
                             value, .atomic_op = nir_atomic_op_iadd);
   }
   apple9_store_output(&b, apple9_global_id_x(&b), value);
   return b.shader;
}

static nir_shader *
apple9_atomic_shader(nir_atomic_op op, bool discard, bool dynamic_index)
{
   nir_builder b = apple9_compute_builder("apple9_atomic");
   b.shader->info.num_ssbos = discard ? 1 : 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *binding = nir_imm_int(&b, discard ? 0 : 1);
   nir_def *offset = dynamic_index ? nir_imul_imm(&b, gid, 4)
                                   : nir_imm_int(&b, 0);
   nir_def *data = op == nir_atomic_op_fadd
                      ? nir_imm_float(&b, 1.25f)
                      : nir_iadd_imm(&b, gid, 7);
   nir_def *result =
      nir_ssbo_atomic(&b, 32, binding, offset, data, .atomic_op = op);
   if (!discard)
      apple9_store_output(&b, gid, result);
   return b.shader;
}

static nir_shader *
apple9_cmpxchg_shader()
{
   nir_builder b = apple9_compute_builder("apple9_cmpxchg");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *result = nir_ssbo_atomic_swap(
      &b, 32, nir_imm_int(&b, 1), nir_imul_imm(&b, gid, 4),
      nir_iadd_imm(&b, gid, 10), nir_iadd_imm(&b, gid, 1000),
      .atomic_op = nir_atomic_op_cmpxchg);
   apple9_store_output(&b, gid, result);
   return b.shader;
}

static nir_shader *
apple9_sequential_atomic_results_shader()
{
   nir_builder b = apple9_compute_builder("apple9_sequential_atomics");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *binding = nir_imm_int(&b, 1);
   nir_def *offset = nir_imul_imm(&b, gid, 4);
   nir_def *first = nir_ssbo_atomic(
      &b, 32, binding, offset, nir_iadd_imm(&b, gid, 3),
      .atomic_op = nir_atomic_op_iadd);
   nir_def *second = nir_ssbo_atomic(
      &b, 32, binding, offset, nir_imm_int(&b, 0x00ff00ff),
      .atomic_op = nir_atomic_op_ixor);
   apple9_store_output(&b, gid, nir_ixor(&b, first, second));
   return b.shader;
}

static nir_shader *
apple9_pending_load_atomic_shader(bool source_live_after,
                                  bool unrelated_pending_load)
{
   nir_builder b = apple9_compute_builder("apple9_pending_load_atomic");
   b.shader->info.num_ssbos = unrelated_pending_load ? 4 : 3;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *offset = nir_imul_imm(&b, gid, 4);
   nir_def *operand =
      nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 2), offset,
                    .access = ACCESS_NON_WRITEABLE);
   nir_def *unrelated = unrelated_pending_load
                           ? nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 3),
                                           offset,
                                           .access = ACCESS_NON_WRITEABLE)
                           : nullptr;
   nir_def *result =
      nir_ssbo_atomic(&b, 32, nir_imm_int(&b, 1), offset, operand,
                      .atomic_op = nir_atomic_op_iadd);

   if (unrelated_pending_load) {
      apple9_store_output(&b, gid, unrelated);
      apple9_store_output(&b, nir_iadd_imm(&b, gid, 64), result);
   } else {
      nir_def *output = source_live_after ? nir_ixor(&b, result, operand)
                                          : result;
      apple9_store_output(&b, gid, output);
   }
   return b.shader;
}

static unsigned
apple9_binary_count_atomics(const struct agx_shader_part *compiled,
                            enum agx_apple9_atomic_op op, bool discard)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   unsigned count = 0;
   for (size_t i = 0; i + 14 <= compiled->info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      count += bytes[0] == 0x67 &&
               (bytes[1] & 0x0f) == 0x01 && (bytes[2] & 0xfc) == 0x54 &&
               bytes[8] == 0x00 &&
               bytes[9] == (discard ? 0x40 : 0x02) &&
               bytes[12] == ((op << 1) | 0x40) && bytes[13] == 0x02;
   }
   return count;
}

static unsigned
apple9_binary_first_atomic_dependency(const struct agx_shader_part *compiled,
                                      enum agx_apple9_atomic_op op)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   for (size_t i = 0; i + 14 <= compiled->info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      if (bytes[0] == 0x67 && (bytes[1] & 0x0f) == 0x01 &&
          (bytes[2] & 0xfc) == 0x54 && bytes[8] == 0x00 &&
          bytes[12] == ((op << 1) | 0x40) && bytes[13] == 0x02)
         return ((bytes[1] >> 4) & 0x0f) | ((bytes[2] & 0x03) << 4);
   }
   return UINT_MAX;
}

static unsigned
apple9_binary_count_pending_stores(const struct agx_shader_part *compiled)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   unsigned count = 0;
   for (size_t i = 0; i + 14 <= compiled->info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      if (bytes[0] != 0xe7 || (bytes[2] & 0xfc) != 0x54)
         continue;
      const unsigned dependency =
         ((bytes[1] >> 4) & 0x0f) | ((bytes[2] & 0x03) << 4);
      count += dependency != 0;
   }
   return count;
}

static bool
apple9_binary_contains_get_sr_zext16(const struct agx_shader_part *compiled,
                                     uint8_t selector)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   for (size_t i = 0; i + 8 <= compiled->info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      if ((bytes[0] & 0xf) == 0x4 && bytes[1] == selector && bytes[2] == 0x10 &&
          bytes[3] == 0x06 && bytes[4] == ((bytes[0] & 0xf0) | 0x03) &&
          bytes[5] == 0 && bytes[6] == 0 && bytes[7] == 1)
         return true;
   }
   return false;
}

static bool
apple9_binary_contains_get_sr(const struct agx_shader_part *compiled,
                              uint8_t selector)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   for (size_t i = 0; i + 4 <= compiled->info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      if ((bytes[0] & 0xf) == 0xc && bytes[1] == selector && bytes[2] == 0x10 &&
          (bytes[3] & 0x1f) == 0x06)
         return true;
   }
   return false;
}

static unsigned
apple9_binary_count_reciprocals(const struct agx_shader_part *compiled)
{
   const uint8_t *binary = (const uint8_t *)compiled->binary;
   unsigned count = 0;
   for (size_t i = 0; i + 10 <= compiled->info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      count += bytes[0] == 0xaf && bytes[1] == 0x00 && bytes[7] == 0x48 &&
               bytes[8] == 0x20 && bytes[9] == 0x00;
   }
   return count;
}

static void
apple9_expect_compile(nir_shader *nir, enum agx_apple9_compute_abi expected_abi)
{
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(compiled.info.stage, MESA_SHADER_COMPUTE);
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_EQ(profile.abi, expected_abi);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, InternalComputeLowersTemporariesAndIntegerDivision)
{
   for (bool dynamic : {false, true}) {
      nir_builder b = apple9_compute_builder("internal_compute_loop");
      b.shader->info.num_ubos = 1;
      nir_def *limit = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                   nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
      nir_variable *counter = nir_local_variable_create(
         b.impl, glsl_uint_type(), "counter");
      nir_store_var(&b, counter, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b)->control = nir_loop_control_dont_unroll;
      nir_def *i = nir_load_var(&b, counter);
      nir_break_if(&b, nir_uge(&b, i, limit));
      nir_def *divisor = dynamic ? limit : nir_imm_int(&b, 3);
      apple9_store_output(&b, i, nir_udiv(&b, i, divisor));
      nir_store_var(&b, counter, nir_iadd_imm(&b, i, 1), 1);
      nir_pop_loop(&b, nullptr);
      apple9_expect_compile(b.shader, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   }
}

TEST(Apple9Compiler, ConstantStoreUsesGenericPipeline)
{
   apple9_expect_compile(apple9_constant_store_shader(42),
                         AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
}

TEST(Apple9Compiler, DeviceAtomicsCoverNativeOperationSelectors)
{
   const struct {
      nir_atomic_op nir_op;
      enum agx_apple9_atomic_op machine_op;
   } cases[] = {
      {nir_atomic_op_iadd, AGX_APPLE9_ATOMIC_ADD},
      {nir_atomic_op_isub, AGX_APPLE9_ATOMIC_SUB},
      {nir_atomic_op_imin, AGX_APPLE9_ATOMIC_SMIN},
      {nir_atomic_op_umin, AGX_APPLE9_ATOMIC_UMIN},
      {nir_atomic_op_imax, AGX_APPLE9_ATOMIC_SMAX},
      {nir_atomic_op_umax, AGX_APPLE9_ATOMIC_UMAX},
      {nir_atomic_op_iand, AGX_APPLE9_ATOMIC_AND},
      {nir_atomic_op_ior, AGX_APPLE9_ATOMIC_OR},
      {nir_atomic_op_ixor, AGX_APPLE9_ATOMIC_XOR},
      {nir_atomic_op_xchg, AGX_APPLE9_ATOMIC_XCHG},
      {nir_atomic_op_fadd, AGX_APPLE9_ATOMIC_FADD},
   };

   for (const auto &test : cases) {
      SCOPED_TRACE(test.nir_op);
      nir_shader *nir = apple9_atomic_shader(test.nir_op, false, true);
      struct agx_shader_part compiled = {};
      struct agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
      EXPECT_EQ(profile.atomic_frame_size, 4u);
      EXPECT_EQ(apple9_binary_count_atomics(&compiled, test.machine_op, false),
                1u);
      ASSERT_EQ(profile.resource_binding_count, 2u);
      EXPECT_EQ(profile.resource_binding[0], 1u);
      EXPECT_EQ(profile.resource_binding[1], 0u);
      EXPECT_EQ(profile.resource_read_mask, 1u);
      EXPECT_EQ(profile.resource_write_mask, 3u);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, SequentialAtomicReturnsUseGeneralPublicationSlots)
{
   nir_shader *nir = apple9_sequential_atomic_results_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   EXPECT_EQ(profile.atomic_frame_size, 4u);
   EXPECT_EQ(apple9_binary_count_atomics(&compiled, AGX_APPLE9_ATOMIC_ADD,
                                         false),
             1u);
   EXPECT_EQ(apple9_binary_count_atomics(&compiled, AGX_APPLE9_ATOMIC_XOR,
                                         false),
             1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, PendingLoadCompletesInReturningAtomic)
{
   nir_shader *nir = apple9_pending_load_atomic_shader(false, false);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(apple9_binary_first_atomic_dependency(
                &compiled, AGX_APPLE9_ATOMIC_ADD),
             1u << (AGX_APPLE9_SCOREBOARD_SLOT_6 - 1));
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, LiveAfterPendingAtomicOperandIsMaterializedSelectively)
{
   nir_shader *nir = apple9_pending_load_atomic_shader(true, false);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(apple9_binary_first_atomic_dependency(
                &compiled, AGX_APPLE9_ATOMIC_ADD),
             0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, AtomicAndStoresCompleteTheirPendingLoads)
{
   nir_shader *nir = apple9_pending_load_atomic_shader(false, true);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");
   unsigned dependency = apple9_binary_first_atomic_dependency(
      &compiled, AGX_APPLE9_ATOMIC_ADD);
   /* Independent loads may now overlap. The atomic still directly consumes
    * one pending completion, whose tag depends on the selected schedule. */
   EXPECT_NE(dependency, 0u);
   EXPECT_EQ(dependency & (dependency - 1), 0u);
   EXPECT_LE(dependency, 1u << (AGX_APPLE9_SCOREBOARD_SLOT_6 - 1));
   EXPECT_GE(apple9_binary_count_pending_stores(&compiled), 1u);
   EXPECT_LE(apple9_binary_count_pending_stores(&compiled), 2u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, CompareExchangeUsesTwoRegisterTuple)
{
   nir_shader *nir = apple9_cmpxchg_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(apple9_binary_count_atomics(&compiled,
                                         AGX_APPLE9_ATOMIC_CMPXCHG, false),
             1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, UnusedAtomicUsesNativeDiscardForm)
{
   nir_shader *nir = apple9_atomic_shader(nir_atomic_op_ixor, true, false);
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(apple9_binary_count_atomics(&compiled, AGX_APPLE9_ATOMIC_XOR,
                                         true),
             1u);
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   EXPECT_EQ(profile.atomic_frame_size, 0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, SimpleIfPredicatesOneStoreRegion)
{
   nir_shader *nir = apple9_simple_if_shader(false);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t push[] = {0x0f, 0x05, 0x54, 0x01};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, push, sizeof(push)), 1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, SimpleIfElseUsesNativeMaskTransition)
{
   nir_shader *nir = apple9_simple_if_shader(true);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t push[] = {0x0f, 0x05, 0x54, 0x01};
   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, push, sizeof(push)), 1u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, else_mask, sizeof(else_mask)),
      1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), 1u);

   /* A native if/else has one short predicate and one saved mask.  The
    * release bits depend on whether either source is used later. */
   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned predicates = 0;
   for (unsigned i = 0; i + 6 <= compiled.info.binary_size; ++i) {
      if (binary[i] != 0x0a || (binary[i + 2] & ~0x18) != 0x22 ||
          binary[i + 4] != AGX_APPLE9_PREDICATE_ULT || binary[i + 5] != 0xc0)
         continue;
      predicates++;
   }
   EXPECT_EQ(predicates, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, DirectConditionsUseProvenPredicateFamilies)
{
   struct condition_case {
      nir_op op;
      bool extended;
      uint8_t condition;
      bool predicate_inverted;
      bool push_inverted;
   } cases[] = {
      {nir_op_ult, false, AGX_APPLE9_PREDICATE_ULT, false, false},
      {nir_op_uge, false, AGX_APPLE9_PREDICATE_ULT, false, true},
      {nir_op_ilt, false, AGX_APPLE9_PREDICATE_ILT, false, false},
      {nir_op_ige, false, AGX_APPLE9_PREDICATE_ILT, false, true},
      {nir_op_ieq, true, AGX_APPLE9_PREDICATE_EXT_IEQ, false, false},
      {nir_op_ine, true, AGX_APPLE9_PREDICATE_EXT_IEQ, false, true},
      {nir_op_flt, false, AGX_APPLE9_PREDICATE_FLT, false, false},
      {nir_op_fge, true, AGX_APPLE9_PREDICATE_EXT_FGE_SEQUENCE, true, true},
      {nir_op_feq, true, AGX_APPLE9_PREDICATE_EXT_FEQ, false, false},
      {nir_op_fneu, true, AGX_APPLE9_PREDICATE_EXT_FEQ, false, true},
   };

   for (const auto &test : cases) {
      nir_shader *nir = apple9_condition_shader(test.op);
      struct agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
         << (reason ? reason : "no diagnostic") << " op=" << test.op;

      const uint8_t *binary = (const uint8_t *)compiled.binary;
      unsigned found = 0;
      for (unsigned i = 0;
           i + (test.extended ? 10 : 6) <= compiled.info.binary_size; ++i) {
         const uint8_t expected_opcode = test.predicate_inverted ? 0x1a : 0x0a;
         const bool header =
            binary[i] == expected_opcode &&
            (binary[i + 2] & ~0x18) == (test.extended ? 0x23 : 0x22);
         const bool tail =
            test.extended
               ? binary[i + 4] == 0x06 && binary[i + 5] == 0 &&
                    binary[i + 6] == test.condition && binary[i + 7] == 0xc0
               : binary[i + 4] == test.condition && binary[i + 5] == 0xc0;
         found += header && tail;
      }
      EXPECT_EQ(found, 1u) << "op=" << test.op;

      const uint8_t push[] = {0x0f, 0x05, 0x54,
                              (uint8_t)(test.push_inverted ? 0x21 : 0x01)};
      EXPECT_EQ(apple9_binary_count_sequence(&compiled, push, sizeof(push)), 1u)
         << "op=" << test.op;
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, PredicateSourceLifetimesReachBothEncodingFamilies)
{
   const struct {
      nir_op op;
      bool extended;
      uint8_t condition;
   } forms[] = {
      {nir_op_ult, false, AGX_APPLE9_PREDICATE_ULT},
      {nir_op_feq, true, AGX_APPLE9_PREDICATE_EXT_FEQ},
   };

   for (const auto &form : forms) {
      for (unsigned live_sources = 0; live_sources < 4; ++live_sources) {
         SCOPED_TRACE(testing::Message()
                      << "op=" << form.op << " live=" << live_sources);
         nir_shader *nir =
            apple9_predicate_lifetime_shader(form.op, live_sources, 0);
         struct agx_shader_part compiled = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
            << (reason ? reason : "no diagnostic");

         const uint8_t *binary = (const uint8_t *)compiled.binary;
         unsigned found = 0;
         for (unsigned i = 0;
              i + (form.extended ? 10 : 6) <= compiled.info.binary_size; ++i) {
            const bool header =
               binary[i] == 0x0a &&
               (binary[i + 2] & ~0x18) == (form.extended ? 0x23 : 0x22);
            const bool tail =
               form.extended
                  ? binary[i + 4] == 0x06 && binary[i + 5] == 0 &&
                       binary[i + 6] == form.condition && binary[i + 7] == 0xc0
                  : binary[i + 4] == form.condition && binary[i + 5] == 0xc0;
            if (!header || !tail)
               continue;

            const uint8_t expected_release =
               ((live_sources & BITFIELD_BIT(0)) ? 0 : 0x08) |
               ((live_sources & BITFIELD_BIT(1)) ? 0 : 0x10);
            EXPECT_EQ(binary[i + 2] & 0x18, expected_release);
            ++found;
         }
         EXPECT_EQ(found, 1u);
         free(compiled.binary);
         ralloc_free(nir);
      }
   }
}

TEST(Apple9Compiler, PredicateSourcesSurviveGeneralRegisterPressure)
{
   nir_shader *nir = apple9_predicate_lifetime_shader(nir_op_ult, 3, 20);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 200u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, SingleRegionSupportsEntryAndMergeStoresAndEmptyArms)
{
   const uint8_t push[] = {0x0f, 0x05, 0x54, 0x01};
   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   const struct {
      enum apple9_region_shape shape;
      unsigned pushes;
      unsigned elses;
      unsigned stores;
   } cases[] = {
      /* Both empty arms disappear; surrounding stores must survive. */
      {APPLE9_REGION_EMPTY, 0, 0, 2},
      {APPLE9_REGION_THEN_ONLY, 1, 1, 3},
      {APPLE9_REGION_ELSE_ONLY, 1, 1, 3},
      {APPLE9_REGION_BOTH, 1, 1, 4},
   };

   for (const auto &test : cases) {
      SCOPED_TRACE(test.shape);
      nir_shader *nir = apple9_single_region_shader(test.shape);
      struct agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_EQ(apple9_binary_count_sequence(&compiled, push, sizeof(push)),
                test.pushes);
      EXPECT_EQ(
         apple9_binary_count_sequence(&compiled, else_mask, sizeof(else_mask)),
         test.elses);
      EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)),
                test.pushes);

      unsigned stores = 0;
      const uint8_t *binary = (const uint8_t *)compiled.binary;
      for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i)
         stores += binary[i] == 0xe7;
      EXPECT_EQ(stores, test.stores);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, PureScalarAndVectorPhisBecomeSelects)
{
   nir_shader *nir = apple9_multiple_phi_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t push[] = {0x0f, 0x05, 0x54, 0x01};
   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, push, sizeof(push)), 0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, else_mask, sizeof(else_mask)),
      0u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), 0u);

   unsigned vector_stores = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
      vector_stores += bytes[0] == 0xe7 && bytes[8] == 0x17;
   }
   EXPECT_EQ(vector_stores, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, ArbitraryPureBooleansMaterializeThenCompareWithZero)
{
   for (bool selected : {false, true}) {
      nir_shader *nir = apple9_composed_boolean_if_shader(selected);
      struct agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
         << (reason ? reason : "no diagnostic");

      const uint8_t *binary = (const uint8_t *)compiled.binary;
      unsigned integer_equal = 0;
      for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i) {
         integer_equal += binary[i] == 0x0a &&
                          (binary[i + 2] & ~0x18) == 0x23 &&
                          binary[i + 4] == 0x06 && binary[i + 5] == 0 &&
                          binary[i + 6] == AGX_APPLE9_PREDICATE_EXT_IEQ &&
                          binary[i + 7] == 0xc0;
      }
      EXPECT_EQ(integer_equal, 1u);
      const uint8_t inverted_push[] = {0x0f, 0x05, 0x54, 0x21};
      EXPECT_EQ(apple9_binary_count_sequence(&compiled, inverted_push,
                                             sizeof(inverted_push)),
                1u);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, PurePhiBecomesDirectSelect)
{
   nir_shader *nir = apple9_simple_phi_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t push[] = {0x0f, 0x05, 0x54, 0x01};
   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, push, sizeof(push)), 0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, else_mask, sizeof(else_mask)),
      0u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), 0u);

   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned selects = 0;
   for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i)
      selects += (binary[i] & 0xf) == 0x02 && binary[i + 4] == 0x82;
   /* The comparison feeds one select directly, without a boolean temporary. */
   EXPECT_EQ(selects, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, NestedIfElseUsesImplicitMaskStack)
{
   nir_shader *nir = apple9_nested_if_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   const uint8_t bank_zero_push[] = {0x0f, 0x05, 0x54, 0x01};
   EXPECT_EQ(apple9_binary_count_exec_pushes(&compiled), 3u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, bank_zero_push,
                                          sizeof(bank_zero_push)),
             3u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, else_mask, sizeof(else_mask)),
      3u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), 3u);

   unsigned stores = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i)
      stores += ((const uint8_t *)compiled.binary)[i] == 0xe7;
   EXPECT_EQ(stores, 4u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, DeepIfNestingDoesNotConsumePredicateBanks)
{
   constexpr unsigned depth = 32;
   nir_shader *nir = apple9_deep_if_shader(depth);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t bank_zero_push[] = {0x0f, 0x05, 0x54, 0x01};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   EXPECT_EQ(apple9_binary_count_exec_pushes(&compiled), depth);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, bank_zero_push,
                                          sizeof(bank_zero_push)),
             depth);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), depth);

   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, NestedPureVectorPhisBecomeSelects)
{
   nir_shader *nir = apple9_nested_phi_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   EXPECT_EQ(apple9_binary_count_exec_pushes(&compiled), 0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, else_mask, sizeof(else_mask)),
      0u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), 0u);

   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned selects = 0;
   for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i)
      selects += (binary[i] & 0xf) == 0x02 && binary[i + 4] == 0x82;
   EXPECT_GT(selects, 0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, StructuredShortCircuitAndOrCompileWithoutSpeculation)
{
   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};

   for (bool is_or : {false, true}) {
      SCOPED_TRACE(is_or ? "or" : "and");
      nir_shader *nir = apple9_short_circuit_shader(is_or);
      struct agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
         << (reason ? reason : "no diagnostic");

      EXPECT_EQ(apple9_binary_count_exec_pushes(&compiled), 2u);
      EXPECT_EQ(
         apple9_binary_count_sequence(&compiled, else_mask, sizeof(else_mask)),
         2u);
      EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), 2u);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, CountedLoopCarriesSsaAndPatchesStartRelativeBackedge)
{
   nir_shader *nir = apple9_counted_loop_shader(false);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t loop_push[] = {0x0f, 0x05, 0x54, 0x1a};
   const uint8_t loop_update[] = {0x8f, 0x04, 0x54, 0x22};
   const uint8_t loop_pop[] = {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00};
   const uint8_t break_one_if[] = {0x8f, 0x05, 0x54, 0x03, 0x00, 0x01};
   const uint8_t exit_if_none[] = {0x0f, 0x01, 0x54};
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_push, sizeof(loop_push)),
      0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_update, sizeof(loop_update)),
      0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_pop, sizeof(loop_pop)), 1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, break_one_if,
                                          sizeof(break_one_if)),
             1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, exit_if_none,
                                          sizeof(exit_if_none)),
             0u);

   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned branches = 0;
   for (unsigned offset = 0; offset + 10 <= compiled.info.binary_size;
        ++offset) {
      if (binary[offset] != 0x0f || binary[offset + 1] != 0x00 ||
          binary[offset + 2] != 0x54 || binary[offset + 9] != 0x00)
         continue;
      int64_t displacement = 0;
      for (unsigned byte = 0; byte < 6; ++byte)
         displacement |= (int64_t)binary[offset + 3 + byte] << (8 * byte);
      if (displacement & (INT64_C(1) << 47))
         displacement |= ~((INT64_C(1) << 48) - 1);
      const int64_t target = (int64_t)offset + displacement;
      EXPECT_LT(target, (int64_t)offset);
      EXPECT_GT(target, 0);
      EXPECT_LT(target, (int64_t)compiled.info.binary_size);
      ++branches;
   }
   EXPECT_EQ(branches, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, NestedLoopsReuseConsumedBreakPredicates)
{
   nir_shader *nir = apple9_nested_loop_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t loop_push[] = {0x0f, 0x05, 0x54, 0x1a};
   const uint8_t outer_update[] = {0x8f, 0x04, 0x54, 0x22};
   const uint8_t inner_update[] = {0x8f, 0x04, 0x54, 0x26};
   const uint8_t loop_pop[] = {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00};
   const uint8_t outer_break[] = {0x8f, 0x05, 0x54, 0x03, 0x00, 0x01};
   const uint8_t inner_break[] = {0x8f, 0x05, 0x54, 0x03, 0x00, 0x01};
   const uint8_t exit_if_none[] = {0x0f, 0x01, 0x54};
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_push, sizeof(loop_push)),
      1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, outer_update,
                                          sizeof(outer_update)),
             0u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, inner_update,
                                          sizeof(inner_update)),
             0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_pop, sizeof(loop_pop)), 2u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, outer_break, sizeof(outer_break)),
      2u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, inner_break, sizeof(inner_break)),
      2u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, exit_if_none,
                                          sizeof(exit_if_none)),
             0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, LoopDepthDoesNotConsumePredicateBanks)
{
   for (unsigned depth : {6u, 16u}) {
      SCOPED_TRACE(depth);
      nir_builder b = apple9_compute_builder("deep_nested_loops");
      b.shader->info.num_ubos = 1;
      nir_def *limit = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                   nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
      std::vector<nir_loop *> loops;
      std::vector<nir_variable *> counters;
      for (unsigned level = 0; level < depth; ++level) {
         nir_variable *counter = nir_local_variable_create(
            b.impl, glsl_uint_type(), "counter");
         nir_store_var(&b, counter, nir_imm_int(&b, 0), 1);
         nir_loop *loop = nir_push_loop(&b);
         loop->control = nir_loop_control_dont_unroll;
         nir_break_if(&b, nir_uge(&b, nir_load_var(&b, counter), limit));
         loops.push_back(loop);
         counters.push_back(counter);
      }
      apple9_store_output(&b, nir_imm_int(&b, 0), limit);
      for (unsigned level = depth; level > 0; --level) {
         nir_variable *counter = counters[level - 1];
         nir_store_var(&b, counter,
            nir_iadd_imm(&b, nir_load_var(&b, counter), 1), 1);
         nir_pop_loop(&b, loops[level - 1]);
      }
      apple9_expect_compile(b.shader, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   }
}

TEST(Apple9Compiler, ContinueConstructLowersToStructuredMaskedLatch)
{
   nir_shader *nir = apple9_counted_loop_shader(true);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t loop_push[] = {0x0f, 0x05, 0x54, 0x1a};
   const uint8_t loop_update[] = {0x8f, 0x04, 0x54, 0x22};
   const uint8_t loop_pop[] = {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00};
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_push, sizeof(loop_push)),
      0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_update, sizeof(loop_update)),
      0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_pop, sizeof(loop_pop)), 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, StructuredLoopDoesNotRequireCanonicalTestPosition)
{
   nir_shader *nir = apple9_mid_body_break_loop_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t loop_push[] = {0x0f, 0x05, 0x54, 0x1a};
   const uint8_t loop_update[] = {0x8f, 0x04, 0x54, 0x22};
   const uint8_t loop_pop[] = {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00};
   const uint8_t exit_if_none[] = {0x0f, 0x01, 0x54};

   /* The top-level loop uses the hardware's implicit initial mask. All three
    * NIR conditionals retain their kind-1 scopes for SSA edge copies; the
    * break unwinds its scope, and the loop has one backedge and kind-2 pop. */
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_push, sizeof(loop_push)),
      0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_update, sizeof(loop_update)),
      0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_pop, sizeof(loop_pop)), 1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, exit_if_none,
                                          sizeof(exit_if_none)),
             0u);
   EXPECT_EQ(apple9_binary_count_exec_pushes(&compiled), 3u);

   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned backedges = 0;
   for (unsigned offset = 0; offset + 10 <= compiled.info.binary_size;
        ++offset) {
      if (binary[offset] != 0x0f || binary[offset + 1] != 0x00 ||
          binary[offset + 2] != 0x54 || binary[offset + 9] != 0x00)
         continue;
      ++backedges;
   }
   EXPECT_EQ(backedges, 1u);

   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, GeneralNestedBreakUsesNativeMaskUnwind)
{
   nir_shader *nir = apple9_general_break_loop_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t unwind[] = {0x8f, 0x05, 0x54, 0x03, 0x00, 0x01};
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, unwind, sizeof(unwind)),
             2u);
   /* The ordinary if populated p0. An unconditional jump must also publish
    * true to p1 before unwind; just finding an unwind opcode missed this bug.
    * Match the loop IEQ bank/type fields while allowing allocated operands. */
   unsigned loop_bank_one = 0;
   const uint8_t *bytes = (const uint8_t *)compiled.binary;
   for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i) {
      if (bytes[i] == 0x2a && bytes[i + 4] == 0x06 &&
          bytes[i + 5] == 0 && bytes[i + 6] == 0x07 &&
          bytes[i + 7] == 0 && bytes[i + 8] == 0 && bytes[i + 9] == 0)
         ++loop_bank_one;
   }
   EXPECT_EQ(loop_bank_one, 2u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, ConditionalLoadsAreCompletedInsideTheirMaskRegions)
{
   const uint8_t push[] = {0x0f, 0x05, 0x54, 0x01};
   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   const struct {
      enum apple9_conditional_load_shape shape;
      unsigned load_count;
      bool else_load;
      bool merge_load;
   } cases[] = {
      {APPLE9_CONDITIONAL_LOAD_THEN_ONLY, 2, false, false},
      {APPLE9_CONDITIONAL_LOAD_BOTH_ARMS, 3, true, false},
      {APPLE9_CONDITIONAL_LOAD_FANOUT_AND_MERGE, 4, true, true},
   };

   for (const auto &test : cases) {
      nir_shader *nir = apple9_conditional_load_shader(test.shape);
      struct agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
         << (reason ? reason : "no diagnostic");

      const unsigned push_offset =
         apple9_binary_find_sequence(&compiled, push, sizeof(push));
      const unsigned else_offset =
         apple9_binary_find_sequence(&compiled, else_mask, sizeof(else_mask));
      const unsigned pop_offset =
         apple9_binary_find_sequence(&compiled, pop, sizeof(pop));
      ASSERT_NE(push_offset, UINT_MAX);
      ASSERT_NE(else_offset, UINT_MAX);
      ASSERT_NE(pop_offset, UINT_MAX);
      ASSERT_LT(push_offset, else_offset);
      ASSERT_LT(else_offset, pop_offset);

      unsigned loads[4] = {};
      ASSERT_EQ(
         apple9_binary_device_load_offsets(&compiled, loads, ARRAY_SIZE(loads)),
         test.load_count);
      EXPECT_LT(loads[0], push_offset);
      EXPECT_GT(loads[1], push_offset);
      EXPECT_LT(loads[1], else_offset);
      if (test.else_load) {
         EXPECT_GT(loads[2], else_offset);
         EXPECT_LT(loads[2], pop_offset);
      }
      if (test.merge_load) {
         EXPECT_GT(loads[3], pop_offset);
      }

      /* Byte-offset conversion uses ordinary computed indices, including
       * the entry load. HAS_NEXT follows issue order across PUSH/ELSE/POP. */
      const uint8_t *binary = (const uint8_t *)compiled.binary;
      for (unsigned i = 0; i < test.load_count; ++i) {
         EXPECT_EQ(binary[loads[i] + 1] & 0x10, 0);
         EXPECT_EQ(binary[loads[i] + 2] & 0x10,
                   i + 1 < test.load_count ? 0x10 : 0x00);
      }

      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, LargeConstantUsesOneRawLiteralAndAllocatedStore)
{
   nir_shader *nir = apple9_constant_store_shader(0x12345678);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned literals = 0, stores = 0;
   unsigned literal_dst = UINT_MAX, store_data = UINT_MAX;
   for (unsigned i = 0; i + 8 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      if ((bytes[0] & 0x0f) == 0x0c && bytes[1] == 0xf8 &&
          (bytes[2] & 0x1f) == 0x02 && bytes[3] == 0x12 && bytes[4] == 0x18 &&
          bytes[5] == 0x08 && bytes[6] == 0xa2 && bytes[7] == 0x01) {
         literals++;
         literal_dst = (bytes[0] >> 4) | ((bytes[2] & 0xc0) >> 2);
      }
   }
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      if (bytes[0] == 0xe7 && bytes[1] == 0x00 && bytes[2] == 0x54) {
         stores++;
         store_data = bytes[3] >> 1;
      }
   }

   EXPECT_EQ(literals, 1u);
   EXPECT_EQ(stores, 1u);
   EXPECT_GE(literal_dst, 2u);
   EXPECT_LT(literal_dst, 64u);
   EXPECT_EQ(store_data, literal_dst);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, AluConstantUsesSixBitModeTwoLiteral)
{
   nir_shader *nir = apple9_large_constant_add_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned literals = 0;
   for (unsigned i = 0; i + 8 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = binary + i;
      if ((bytes[0] & 0x0f) == 0x0c && bytes[1] == 0xf8 &&
          (bytes[2] & 0x1f) == 0x02 && bytes[3] == 0x12 && bytes[4] == 0x18 &&
          bytes[5] == 0x08 && bytes[6] == 0xa2 && bytes[7] == 0x01) {
         literals++;
         const unsigned dst = (bytes[0] >> 4) | ((bytes[2] & 0xc0) >> 2);
         EXPECT_GE(dst, 2u);
         EXPECT_LT(dst, 64u);
      }
   }

   EXPECT_EQ(literals, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, IntegerAndFloatDoNotDependOnInputCount)
{
   static const enum agx_apple9_compute_abi abi[] = {
      AGX_APPLE9_COMPUTE_ABI_INVALID,
      AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS,
      AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS,
      AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS,
   };

   for (unsigned inputs = 1; inputs <= 3; ++inputs) {
      for (bool floating : {false, true}) {
         SCOPED_TRACE(testing::Message()
                      << "inputs=" << inputs << " floating=" << floating);
         apple9_expect_compile(apple9_ssbo_reduce_shader(inputs, floating),
                               abi[inputs]);
      }
   }
}

TEST(Apple9Compiler, RejectsVolatileAndCoherentAccess)
{
   for (enum gl_access_qualifier access : {ACCESS_VOLATILE, ACCESS_COHERENT}) {
      nir_shader *nir = apple9_ssbo_reduce_shader(2, false, access);
      struct agx_shader_part compiled = {};
      const char *reason = nullptr;
      EXPECT_FALSE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason));
      EXPECT_NE(reason, nullptr);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, GeneralIntegerAndFloatDagsCompile)
{
   apple9_expect_compile(apple9_arbitrary_integer_shader(),
                         AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   apple9_expect_compile(apple9_arbitrary_float_shader(),
                         AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
}

TEST(Apple9Compiler, ReciprocalUsesNativeHandoffAndLifetimeForms)
{
   static const struct {
      enum apple9_reciprocal_shape shape;
      uint8_t handoff;
      uint8_t result_hint;
      uint8_t source_lifetime;
   } cases[] = {
      {APPLE9_RECIPROCAL_DIRECT_STORE, 0x56, 0x03, 0x10},
      {APPLE9_RECIPROCAL_RETAIN_SOURCE, 0x56, 0x03, 0x00},
      {APPLE9_RECIPROCAL_MATERIALIZED_SOURCE, 0x54, 0x03, 0x10},
      {APPLE9_RECIPROCAL_RESULT_FANOUT, 0x56, 0x03, 0x00},
   };

   for (const auto &test : cases) {
      SCOPED_TRACE(testing::Message() << "shape=" << test.shape);
      nir_shader *nir = apple9_reciprocal_shader(test.shape);
      struct agx_shader_part compiled = {};
      struct agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");

      const uint8_t *binary = (const uint8_t *)compiled.binary;
      unsigned reciprocals = 0;
      for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i) {
         const uint8_t *bytes = binary + i;
         if (bytes[0] != 0xaf || bytes[1] != 0x00 || bytes[7] != 0x48 ||
             bytes[8] != 0x20 || bytes[9] != 0x00)
            continue;
         ++reciprocals;
         EXPECT_EQ(bytes[2], test.handoff);
         EXPECT_EQ(bytes[4], test.result_hint);
         EXPECT_EQ(bytes[6], test.source_lifetime);
         EXPECT_LT(bytes[3] >> 1, AGX_APPLE9_GPR_COUNT);
         EXPECT_LT(bytes[5] >> 2, 64u);
      }
      EXPECT_EQ(reciprocals, 1u);
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, SpecialFunctionsUseAllocatedOperandsAndLifetimes)
{
   const nir_op ops[] = {nir_op_frsq, nir_op_fsqrt, nir_op_fexp2, nir_op_flog2,
                         nir_op_ffloor, nir_op_fceil, nir_op_ftrunc,
                         nir_op_fround_even, nir_op_fsin_factor_agx};
   for (nir_op op : ops) {
      for (unsigned shape = 0; shape < 3; ++shape) {
         SCOPED_TRACE(testing::Message() << "op=" << op << " shape=" << shape);
         nir_builder b = apple9_compute_builder("apple9_special");
         b.shader->info.num_ssbos = 2;
         nir_def *gid = apple9_global_id_x(&b);
         nir_def *x = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1),
                                     nir_imul_imm(&b, gid, 4),
                                     .access = ACCESS_NON_WRITEABLE);
         if (shape == 2)
            x = nir_fadd_imm(&b, x, 0.25);
         nir_def *y = nir_build_alu(&b, op, x, NULL, NULL, NULL);
         apple9_store_output(&b, gid, y);
         if (shape == 1)
            apple9_store_output(&b, nir_iadd_imm(&b, gid, 64),
                                nir_fadd(&b, y, x));
         struct agx_shader_part compiled = {};
         struct agx_apple9_compute_profile profile = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
            << (reason ? reason : "no diagnostic");
         const uint8_t *bytes = (const uint8_t *)compiled.binary;
         unsigned found = 0;
         for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i) {
            const uint8_t *p = bytes + i;
            if ((p[0] != 0xaf && p[0] != 0x2f) || p[7] != 0x40 || p[9] != 0)
               continue;
            ++found;
            if (op == nir_op_fsqrt || op == nir_op_fsin_factor_agx) {
               EXPECT_EQ(p[0], 0x2f);
               EXPECT_EQ(p[1], op == nir_op_fsqrt ? 1 : 3);
               EXPECT_EQ(p[8], 0);
            }
            EXPECT_LT(p[3] >> 1, 96u);
            EXPECT_LT(p[5] >> 2, 64u);
            EXPECT_EQ(p[6], shape == 1 || op == nir_op_fsqrt ? 0x90 : 0xb0);
            EXPECT_EQ(p[2], shape < 2 ? 0x56 : 0x54);
         }
         EXPECT_EQ(found, 1u);
         free(compiled.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, IntegerAbsoluteLowersFromOrdinaryNir)
{
   nir_builder b = apple9_compute_builder("apple9_integer_absolute");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *value = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1),
      nir_imul_imm(&b, gid, 4), .access = ACCESS_NON_WRITEABLE);
   apple9_store_output(&b, gid, nir_iabs(&b, value));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ?: "");
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, SaturationKeepsNativeVectorSemantics)
{
   nir_builder b = apple9_compute_builder("apple9_saturate");
   nir_def *input = nir_imm_vec4(&b, -2.0, -0.0, 0.375, 2.0);
   nir_def *clamped = nir_fsat(&b, input);
   nir_intrinsic_instr *store =
      nir_store_ssbo(&b, clamped, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                     .write_mask = 0xf, .align_mul = 16);
   EXPECT_FALSE(agx_nir_lower_apple9_math(b.shader));
   ASSERT_EQ(nir_def_as_alu(clamped)->op, nir_op_fsat);
   ASSERT_TRUE(nir_opt_constant_folding(b.shader));
   ASSERT_TRUE(nir_src_is_const(store->src[0]));
   const nir_const_value *value = nir_src_as_const_value(store->src[0]);
   EXPECT_EQ(value[0].u32, 0u);
   EXPECT_EQ(value[1].u32, 0u);
   EXPECT_FLOAT_EQ(value[2].f32, 0.375f);
   EXPECT_FLOAT_EQ(value[3].f32, 1.0f);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, TrigonometryLowersFromOrdinaryNir)
{
   for (nir_op op : {nir_op_fsin, nir_op_fcos}) {
      nir_builder b = apple9_compute_builder("apple9_trigonometry");
      b.shader->info.num_ssbos = 2;
      nir_def *gid = apple9_global_id_x(&b);
      nir_def *x = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1),
                                  nir_imul_imm(&b, gid, 4),
                                  .access = ACCESS_NON_WRITEABLE);
      apple9_store_output(&b, gid, nir_build_alu(&b, op, x, NULL, NULL, NULL));
      /* Two independent reductions must fit together without CSE keeping
       * shared constants live across both complete expression trees. */
      nir_def *other = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1),
                                    nir_imul_imm(&b, nir_iadd_imm(&b, gid, 37), 4),
                                    .access = ACCESS_NON_WRITEABLE);
      apple9_store_output(&b, nir_iadd_imm(&b, gid, 64),
                          nir_build_alu(&b, op, other, NULL, NULL, NULL));
      struct agx_shader_part compiled = {};
      struct agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");
      const uint8_t *bytes = (const uint8_t *)compiled.binary;
      unsigned factors = 0;
      for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i) {
         const uint8_t *p = bytes + i;
         if (p[0] == 0x2f && p[1] == 3 && p[7] == 0x40 && p[9] == 0) {
            ++factors;
            EXPECT_EQ(p[6], 0x90); /* The multiply still needs its phase. */
         }
      }
      EXPECT_EQ(factors, 4u); /* Sine and complement for each reduction. */
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, VectorLoadsUseOneNativeScoreboardTuple)
{
   nir_shader *nir = apple9_vector_load_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, NativeVectorLoadsAndStoresCoverTwoThreeAndFourLanes)
{
   for (unsigned components : {2u, 3u, 4u}) {
      SCOPED_TRACE(components);
      nir_shader *nir = apple9_vector_copy_shader(components);
      struct agx_shader_part compiled = {};
      struct agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);

      unsigned vector_loads = 0, vector_stores = 0;
      unsigned load_data = UINT_MAX, store_data = UINT_MAX;
      const uint8_t load_token = components == 2   ? 0x59
                                 : components == 3 ? 0x5d
                                                   : 0x57;
      const uint8_t store_token = components == 2   ? 0x19
                                  : components == 3 ? 0x1d
                                                    : 0x17;
      for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
         const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
         if (bytes[0] == 0x67 && bytes[8] == load_token) {
            vector_loads++;
            load_data = bytes[3];
         }
         if (bytes[0] == 0xe7 && bytes[8] == store_token) {
            vector_stores++;
            store_data = bytes[3];
            EXPECT_EQ(bytes[2] & ~3u, 0x54u); /* The tuple was materialized after its load. */
         }
      }
      EXPECT_EQ(vector_loads, 1u);
      EXPECT_EQ(vector_stores, 1u);
      EXPECT_EQ(store_data, load_data);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, ComputedVectorStoreUsesPreRaTupleCollection)
{
   nir_shader *nir = apple9_vector_alu_store_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   unsigned vector_stores = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
      if (bytes[0] == 0xe7 && bytes[8] == 0x17) {
         vector_stores++;
         EXPECT_EQ(bytes[2], 0x54);
         EXPECT_EQ(bytes[3] & 1, 0u);
         EXPECT_LE((bytes[3] >> 1) + 4, 64u);
      }
   }
   EXPECT_EQ(vector_stores, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, MultipleStoresShareOneWritableResource)
{
   nir_shader *nir = apple9_multiple_stores_one_binding_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");

   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   EXPECT_EQ(profile.resource_binding_count, 1u);
   EXPECT_EQ(profile.resource_binding[0], 0u);
   EXPECT_EQ(profile.resource_read_mask, 0u);
   EXPECT_EQ(profile.resource_write_mask, 1u);

   unsigned stores = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
      if (bytes[0] == 0xe7 && bytes[2] == 0x54) {
         stores++;
         EXPECT_EQ(bytes[4], AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE);
      }
   }
   /* Adjacent stores coalesce while keeping one writable binding. */
   EXPECT_EQ(stores, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, MultipleWritableBindingsUseSemanticResourceMasks)
{
   for (bool alias_input : {false, true}) {
      SCOPED_TRACE(alias_input ? "read/write alias" : "separate input");
      nir_shader *nir = apple9_multiple_output_bindings_shader(alias_input);
      struct agx_shader_part compiled = {};
      struct agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");

      if (alias_input) {
         EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
         ASSERT_EQ(profile.resource_binding_count, 2u);
         EXPECT_EQ(profile.resource_binding[0], 1u);
         EXPECT_EQ(profile.resource_binding[1], 0u);
         EXPECT_EQ(profile.resource_read_mask, 0x2u);
         EXPECT_EQ(profile.resource_write_mask, 0x3u);
      } else {
         EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
         ASSERT_EQ(profile.resource_binding_count, 3u);
         EXPECT_EQ(profile.resource_binding[0], 2u);
         EXPECT_EQ(profile.resource_binding[1], 1u);
         EXPECT_EQ(profile.resource_binding[2], 0u);
         EXPECT_EQ(profile.resource_read_mask, 0x1u);
         EXPECT_EQ(profile.resource_write_mask, 0x6u);
      }

      unsigned stores = 0;
      uint8_t store_arguments[2] = {};
      uint8_t access_desc[2] = {};
      for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
         const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
         if (bytes[0] == 0xe7 && bytes[2] == 0x54 && stores < 2) {
            store_arguments[stores++] = bytes[4];
            access_desc[stores - 1] = bytes[6];
         }
      }
      EXPECT_EQ(stores, 2u);
      EXPECT_EQ(store_arguments[0], AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE +
                                       (alias_input ? 1u : 2u));
      EXPECT_EQ(store_arguments[1], AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE +
                                       (alias_input ? 0u : 1u));
      /* Each independently computed byte offset has its own last use. */
      /* Bit 0 retains a still-live data source; it is not the access mode. */
         EXPECT_EQ(access_desc[0] & ~1u, 0x20u);
      EXPECT_EQ(access_desc[1], 0x21);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, MultipleScalarAndVectorStoresUseAllocatedSources)
{
   nir_shader *nir = apple9_multiple_scalar_vector_stores_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   EXPECT_EQ(profile.resource_read_mask, 0x1u);
   EXPECT_EQ(profile.resource_write_mask, 0x6u);

   unsigned vector_stores = 0, scalar_stores = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
      if (bytes[0] != 0xe7 || bytes[2] != 0x54)
         continue;
      if (bytes[8] == 0x17) {
         vector_stores++;
         EXPECT_EQ(bytes[4], AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + 2u);
      } else if (bytes[8] == 0x11) {
         scalar_stores++;
         EXPECT_EQ(bytes[4], AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + 1u);
      }
   }
   EXPECT_EQ(vector_stores, 1u);
   EXPECT_EQ(scalar_stores, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, MaterializedScalarStoreUsesAllocatedSource)
{
   nir_shader *nir = apple9_ssbo_reduce_shader(2, false);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   unsigned stores = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
      if (bytes[0] == 0xe7 && bytes[2] == 0x54) {
         stores++;
         EXPECT_EQ(bytes[3] & 1, 0u);
         EXPECT_LT(bytes[3] >> 1, 64u);
      }
   }
   EXPECT_EQ(stores, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, NativeNarrowLoadsExtendAndStoresTruncate)
{
   for (unsigned bits : {8u, 16u}) {
      for (bool sign_extend : {false, true}) {
         SCOPED_TRACE(testing::Message()
                      << "load bits=" << bits << " signed=" << sign_extend);
         nir_shader *nir = apple9_narrow_load_shader(bits, sign_extend);
         struct agx_shader_part compiled = {};
         struct agx_apple9_compute_profile profile = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
            << (reason ? reason : "no diagnostic");
         EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);

         unsigned narrow_loads = 0;
         const uint8_t format = bits == 8 ? 0x21 : 0x01;
         const uint8_t tail = bits == 8 ? 0x42 : 0x44;
         for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
            const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
            narrow_loads += bytes[0] == 0x67 && (bytes[8] & 0x3f) == format &&
                            bytes[12] == tail;
         }
         EXPECT_EQ(narrow_loads, 1u);
         free(compiled.binary);
         ralloc_free(nir);
      }

      SCOPED_TRACE(testing::Message() << "store bits=" << bits);
      nir_shader *nir = apple9_narrow_store_shader(bits);
      struct agx_shader_part compiled = {};
      struct agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);

      unsigned narrow_stores = 0;
      const uint8_t format = bits == 8 ? 0x21 : 0x01;
      const uint8_t tail = bits == 8 ? 0x10 : 0x11;
      for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
         const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
         narrow_stores +=
            bytes[0] == 0xe7 && bytes[8] == format && bytes[12] == tail;
      }
      EXPECT_EQ(narrow_stores, 1u);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, UboLoadsUseTypedNativeResourceArguments)
{
   nir_shader *nir = apple9_ubo_load_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   ASSERT_EQ(profile.resource_binding_count, 2u);
   EXPECT_EQ(profile.resource_kind[0], AGX_APPLE9_COMPUTE_RESOURCE_UBO);
   EXPECT_EQ(profile.resource_binding[0], 0u);
   EXPECT_EQ(profile.resource_kind[1], AGX_APPLE9_COMPUTE_RESOURCE_SSBO);
   EXPECT_EQ(profile.resource_binding[1], 0u);
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, NestedDependentLoadsRetainEarlierResults)
{
   nir_shader *nir = apple9_nested_dependent_load_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, DynamicLoadIndexDrivesScatter)
{
   nir_shader *nir = apple9_dynamic_scatter_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");

   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   ASSERT_EQ(profile.resource_binding_count, 3u);
   EXPECT_EQ(profile.resource_binding[0], 2u);
   EXPECT_EQ(profile.resource_binding[1], 1u);
   EXPECT_EQ(profile.resource_binding[2], 0u);

   unsigned stores = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
      if (bytes[0] == 0xe7 && bytes[2] == 0x54 &&
          bytes[4] == AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + 2) {
         stores++;
         EXPECT_EQ(bytes[6], 0x21);
      }
   }
   EXPECT_EQ(stores, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, ProceduralDynamicScatterCompiles)
{
   nir_shader *nir = apple9_procedural_scatter_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");

   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   ASSERT_EQ(profile.resource_binding_count, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, ArbitraryLoadedIndexNeedsNoRangeProof)
{
   nir_shader *nir = apple9_unbounded_scatter_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
   EXPECT_EQ(profile.resource_binding_count, 2u);
   EXPECT_EQ(profile.resource_read_mask, 0x1u);
   EXPECT_EQ(profile.resource_write_mask, 0x2u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, VariableShiftsUseNativeRegisterOperands)
{
   for (nir_op op : {nir_op_ishl, nir_op_ishr, nir_op_ushr}) {
      nir_shader *nir = apple9_variable_shift_shader(op);
      struct agx_shader_part compiled = {};
      struct agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
      EXPECT_LT(compiled.info.binary_size, 100u);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, GenericComputeSystemRegisterTable)
{
   struct selector_case {
      enum apple9_test_system_value system;
      unsigned component;
      uint8_t selector;
      bool zext16;
   };
   static const selector_case cases[] = {
      {APPLE9_TEST_LOCAL_ID, 0, 0xa4, true},
      {APPLE9_TEST_LOCAL_ID, 1, 0xa5, true},
      {APPLE9_TEST_LOCAL_ID, 2, 0xa6, true},
      {APPLE9_TEST_LOCAL_INDEX, 0, 0xa7, true},
      {APPLE9_TEST_WORKGROUP_ID, 0, 0x9c, false},
      {APPLE9_TEST_WORKGROUP_ID, 1, 0x9d, false},
      {APPLE9_TEST_WORKGROUP_ID, 2, 0x9e, false},
      {APPLE9_TEST_WORKGROUP_SIZE, 0, 0x98, true},
      {APPLE9_TEST_WORKGROUP_SIZE, 1, 0x99, true},
      {APPLE9_TEST_WORKGROUP_SIZE, 2, 0x9a, true},
      {APPLE9_TEST_SUBGROUP_INVOCATION, 0, 0x82, true},
      {APPLE9_TEST_SUBGROUP_ID, 0, 0x85, true},
   };

   for (const auto &test : cases) {
      SCOPED_TRACE(testing::Message()
                   << "selector=" << (unsigned)test.selector);
      nir_shader *nir = apple9_system_value_shader(test.system, test.component);
      struct agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
         << (reason ? reason : "no diagnostic");

      if (test.zext16) {
         EXPECT_TRUE(
            apple9_binary_contains_get_sr_zext16(&compiled, test.selector));
      } else {
         EXPECT_TRUE(apple9_binary_contains_get_sr(&compiled, test.selector));
      }
      free(compiled.binary);
      ralloc_free(nir);
   }

   apple9_expect_compile(
      apple9_system_value_shader(APPLE9_TEST_SUBGROUP_SIZE, 0),
      AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);
}

TEST(Apple9Compiler, CachedSystemValueDominatesConditionalUses)
{
   nir_builder b = apple9_compute_builder("conditional_system_value");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, gid, 16));
   apple9_store_binding(&b, 0, gid,
                       nir_iadd_imm(&b, nir_load_local_invocation_index(&b), 100));
   nir_push_else(&b, nif);
   apple9_store_binding(&b, 1, gid,
                       nir_iadd_imm(&b, nir_load_local_invocation_index(&b), 200));
   nir_pop_if(&b, nif);

   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t push[] = {0x0f, 0x05, 0x54, 0x01};
   const uint8_t *binary = (const uint8_t *)compiled.binary;
   size_t first_push = 0;
   while (first_push + sizeof(push) <= compiled.info.binary_size &&
          memcmp(binary + first_push, push, sizeof(push)))
      ++first_push;
   ASSERT_LE(first_push + sizeof(push), compiled.info.binary_size);
   /* Inspect the unconditional prefix: lanes bypassing the first arm must
    * already have their local index before either arm consumes the cache. */
   struct agx_shader_part prefix = compiled;
   prefix.info.binary_size = first_push;
   EXPECT_TRUE(apple9_binary_contains_get_sr_zext16(&prefix, 0xa7));
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, NumWorkgroupsLoadsPublishedCounts)
{
   for (bool atomic : {false, true}) {
      for (bool variable : {false, true}) {
         for (unsigned component = 0; component < 3; ++component) {
            SCOPED_TRACE(testing::Message()
                         << "atomic=" << atomic << " variable=" << variable
                         << " component=" << component);
            nir_shader *nir =
               apple9_num_workgroups_shader(variable, component, atomic);
            struct agx_shader_part compiled = {};
            struct agx_apple9_compute_profile profile = {};
            const char *reason = nullptr;
            ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
               << (reason ? reason : "no diagnostic");

            EXPECT_EQ(profile.variable_local_size, variable);
            EXPECT_EQ(apple9_binary_count_reciprocals(&compiled), 0u);
            if (variable) {
               EXPECT_EQ(profile.local_size[component], 0u);
               EXPECT_FALSE(apple9_binary_contains_get_sr_zext16(
                  &compiled, 0x98 + component));
            } else {
               static const uint32_t expected[] = {7, 5, 3};
               EXPECT_EQ(profile.local_size[component], expected[component]);
            }

            free(compiled.binary);
            ralloc_free(nir);
         }
      }
   }
}

TEST(Apple9Compiler, SystemRegisterAndDerivedLoadIndicesShareTheSsaPath)
{
   nir_shader *nir = apple9_system_load_index_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS);

   bool saw_local_index = false;
   unsigned scalar_loads = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
      saw_local_index |= (bytes[0] & 0xf) == 0x4 && bytes[1] == 0xa7 &&
                         bytes[2] == 0x10 && bytes[3] == 0x06 &&
                         bytes[4] == ((bytes[0] & 0xf0) | 0x03);
      scalar_loads += bytes[0] == 0x67 && (bytes[8] & 0xf) == 1 &&
                      bytes[10] == 0 && bytes[11] == 0x40 &&
                      (bytes[12] & 0x3f) == 6;
   }
   EXPECT_TRUE(saw_local_index);
   /* The local and wrapped derived indices are distinct scalar accesses.
    * Count all producer slots, not only slot 6's 0x51 token. */
   EXPECT_EQ(scalar_loads, 2u);

   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Machine, RenderPublicationOperands)
{
   const uint8_t phys[] = {7, 16, 17};
   agx_apple9_vir_instr mul = {};
   mul.op = AGX_APPLE9_VIR_FMUL;
   mul.encoding = AGX_APPLE9_ENC_FLOAT2_EXPORT;
   mul.dest = 0;
   mul.src[0] = 1;
   mul.src[1] = 2;
   mul.nr_srcs = 2;
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&mul, phys, &packed, &reason));
   const uint8_t expected[] = {0x79, 0x21, 0x3d, 0x23, 0x41, 0, 0, 0};
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
   const uint8_t bad_dest[] = {128, 16, 17};
   EXPECT_FALSE(
      agx_apple9_pack_vir_instruction(&mul, bad_dest, &packed, &reason));
   const uint8_t bad_source[] = {7, 96, 17};
   EXPECT_FALSE(
      agx_apple9_pack_vir_instruction(&mul, bad_source, &packed, &reason));
}

static nir_shader *
apple9_render_test_shader(bool fragment, bool flat = false)
{
   nir_builder b = nir_builder_init_simple_shader(
      fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX, &agx_nir_options,
      "apple9_render_test");
   if (fragment) {
      nir_def *bary = nir_load_barycentric_pixel(
         &b, 32, .interp_mode = flat ? INTERP_MODE_FLAT : INTERP_MODE_SMOOTH);
      nir_def *color = nir_load_interpolated_input(
         &b, 3, 32, bary, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *r = nir_channel(&b, color, 0);
      nir_def *g = nir_channel(&b, color, 1);
      nir_def *blue = nir_channel(&b, color, 2);
      nir_store_output(
         &b,
         nir_vec4(&b, nir_fmul(&b, r, r), nir_fadd_imm(&b, g, .125),
                  nir_fmul_imm(&b, blue, .5), nir_imm_float(&b, 1)),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   } else {
      nir_def *id = nir_load_vertex_id(&b);
      nir_def *f = nir_u2f32(&b, id);
      nir_store_output(
         &b,
         nir_vec4(&b, nir_fmul_imm(&b, f, .5),
                  nir_fsub(&b, f, nir_imm_float(&b, 1)), nir_imm_float(&b, 0),
                  nir_imm_float(&b, 1)),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_output(
         &b,
         nir_vec3(&b, f, nir_b2f32(&b, nir_ieq_imm(&b, id, 1)),
                  nir_imm_float(&b, .25)),
         nir_imm_int(&b, 0), .write_mask = 7, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }
   b.shader->info.io_lowered = true;
   return b.shader;
}

TEST(Apple9Compiler, GraphicsVectorConstantsMatchSimplifiedScalarInputs)
{
   for (bool fragment : {false, true}) {
      unsigned sizes[2] = {};
      for (unsigned simplified = 0; simplified < 2; ++simplified) {
         nir_builder b = nir_builder_init_simple_shader(
            fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX,
            &agx_nir_options, "graphics_vector_constants");
         b.shader->info.num_ubos = 1;
         nir_def *value = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
            nir_imm_int(&b, 0), .align_mul = 16, .range = 16);
         if (simplified) {
            value = nir_vec4(&b, nir_channel(&b, value, 0),
               nir_imm_float(&b, .5), nir_imm_float(&b, .25),
               nir_channel(&b, value, 3));
         } else {
            value = nir_fadd(&b,
               nir_fmul(&b, value, nir_imm_vec4(&b, 1, 0, 0, 1)),
               nir_imm_vec4(&b, 0, .5, .25, 0));
         }
         nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = fragment ? unsigned(FRAG_RESULT_DATA0) : unsigned(VARYING_SLOT_POS)});
         b.shader->info.io_lowered = true;
         agx_shader_part compiled = {};
         const char *reason = nullptr;
         bool ok = fragment ? agx_compile_apple9_fragment(b.shader, &compiled, &reason)
                            : agx_compile_apple9_vertex(b.shader, &compiled, &reason);
         ASSERT_TRUE(ok) << (reason ?: "");
         sizes[simplified] = compiled.info.binary_size;
         free(compiled.binary);
         ralloc_free(b.shader);
      }
      EXPECT_EQ(sizes[0], sizes[1]) << "fragment=" << fragment;
   }
}

TEST(Apple9Compiler, VertexExportsStayDistinctThroughCompletion)
{
   nir_shader *nir = apple9_render_test_shader(false);
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_vertex(nir, &compiled, &reason))
      << (reason ?: "");
   const uint8_t *code = static_cast<const uint8_t *>(compiled.binary);
   uint64_t registers = 0;
   unsigned slots = 0, count = 0;
   bool vertex_id = false;
   for (unsigned i = 0; i + 4 <= compiled.info.binary_size; ++i) {
      if ((code[i] & 15) == 12 && code[i + 1] == 0xdd && code[i + 2] == 0x10) {
         vertex_id |= code[i + 3] == 0x06;
      }
   }
   for (unsigned i = 0; i + 8 <= compiled.info.binary_size; ++i) {
      if (code[i] != 0x57 || (code[i + 1] & 15) != 6 ||
          (code[i + 2] & 0xfc) != 0x54)
         continue;
      unsigned reg = code[i + 3] >> 1;
      unsigned slot = code[i + 4] >> 5;
      ASSERT_LT(reg, 64u);
      EXPECT_EQ(registers & (UINT64_C(1) << reg), 0u);
      registers |= UINT64_C(1) << reg;
      slots |= 1u << slot;
      ++count;
   }
   EXPECT_TRUE(vertex_id);
   EXPECT_EQ(count, 7u);
   EXPECT_EQ(slots, 0x7fu);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, FragmentPerspectiveCoefficientsAndArithmetic)
{
   nir_shader *nir = apple9_render_test_shader(true);
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(nir, &compiled, &reason))
      << (reason ?: "");
   EXPECT_EQ(compiled.info.stage, MESA_SHADER_FRAGMENT);
   EXPECT_EQ(compiled.info.varyings.fs.nr_cf, 4u);
   const uint8_t *code = static_cast<const uint8_t *>(compiled.binary);
   unsigned coefficients = 0, count = 0;
   for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i) {
      if (code[i] == 0x2f && (code[i + 1] == 5 || code[i + 1] == 13) &&
          code[i + 4] == 3 && code[i + 7] == 2) {
         EXPECT_EQ(code[i + 6], 0);
         coefficients |= 1u << (code[i + 5] >> 1);
         ++count;
      }
   }
   EXPECT_EQ(count, 4u);
   EXPECT_EQ(coefficients, 15u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, UnsupportedRenderInputsAndIncompleteOutputsFailClosed)
{
   nir_shader *nir = apple9_render_test_shader(true, true);
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   EXPECT_FALSE(agx_compile_apple9_fragment(nir, &compiled, &reason));
   EXPECT_NE(reason, nullptr);
   EXPECT_EQ(compiled.binary, nullptr);
   ralloc_free(nir);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "missing_position");
   nir_store_output(
      &b, nir_imm_float(&b, 1), nir_imm_int(&b, 0), .write_mask = 1,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   b.shader->info.io_lowered = true;
   EXPECT_FALSE(agx_compile_apple9_vertex(b.shader, &compiled, &reason));
   EXPECT_NE(reason, nullptr);
   EXPECT_EQ(compiled.binary, nullptr);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, GraphicsUboBindingAndConstantVectorOffsets)
{
   for (bool fragment : {false, true}) {
      nir_shader *nir = apple9_render_test_shader(fragment);
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_builder b = nir_builder_create(impl);
      nir_intrinsic_instr *output = nullptr;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_store_output) {
               output = nir_instr_as_intrinsic(instr);
               break;
            }
         }
         if (output)
            break;
      }
      ASSERT_NE(output, nullptr);
      b.cursor = nir_before_instr(&output->instr);
      nir_def *uniform = nir_load_ubo(
         &b, 4, 32, nir_imm_int(&b, 3), nir_imm_int(&b, 16), .align_mul = 16,
         .align_offset = 0, .range_base = 16, .range = 16);
      nir_src_rewrite(&output->src[0],
                      nir_fmul(&b, output->src[0].ssa, uniform));
      nir->info.num_ubos = 4;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(fragment
                     ? agx_compile_apple9_fragment(nir, &compiled, &reason)
                     : agx_compile_apple9_vertex(nir, &compiled, &reason))
         << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_ubo_mask, 1u << 3);
      EXPECT_GT(compiled.info.binary_size, 0u);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST(Apple9Compiler, GraphicsBufferAddressTables)
{
   for (bool fragment : {false, true}) {
      for (unsigned count : {5u, 16u, 32u}) {
         nir_shader *nir = apple9_render_test_shader(fragment);
         nir_function_impl *impl = nir_shader_get_entrypoint(nir);
         nir_builder b = nir_builder_create(impl);
         nir_intrinsic_instr *output = nullptr;
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_store_output) {
                  output = nir_instr_as_intrinsic(instr);
                  break;
               }
            }
            if (output)
               break;
         }
         ASSERT_NE(output, nullptr);
         b.cursor = nir_before_instr(&output->instr);
         nir_def *sum = nir_imm_float(&b, 0);
         for (unsigned i = 0; i < count; ++i) {
            /* Include the highest API binding even for the sparse cases. */
            unsigned binding = i == count - 1 ? 31 : i;
            nir_def *value = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, binding),
                                          nir_imm_int(&b, i * 4),
                                          .align_mul = 4, .range = 128);
            sum = nir_fadd(&b, sum, value);
         }
         nir_src_rewrite(&output->src[0],
                         nir_fmul(&b, output->src[0].ssa, sum));
         nir->info.num_ubos = 32;
         agx_shader_part compiled = {};
         const char *reason = nullptr;
         ASSERT_TRUE(fragment
                        ? agx_compile_apple9_fragment(nir, &compiled, &reason)
                        : agx_compile_apple9_vertex(nir, &compiled, &reason))
            << count << " " << (reason ?: "");
         EXPECT_EQ(compiled.info.apple9_resource_count, count);
         EXPECT_EQ(compiled.info.apple9_resource_binding[0], 31);
         EXPECT_EQ(compiled.info.apple9_ubo_mask,
                   (count == 32 ? UINT32_MAX
                                : ((1u << (count - 1)) - 1) | (1u << 31)));
         unsigned indirect = 0;
         const uint8_t *code = (const uint8_t *)compiled.binary;
         for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i)
            indirect += code[i] == 0x67 && (code[i + 2] & 0xec) == 0x44 &&
                        code[i + 11] == 0x40 && (code[i + 12] & 0x20);
         EXPECT_GE(indirect, count);
         free(compiled.binary);
         ralloc_free(nir);
      }
   }
}

TEST(Apple9Compiler,
     GraphicsRejectsInvalidBufferBindingWithoutDiagnosticPointer)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "invalid_graphics_ubo");
   nir_def *sum = nir_imm_float(&b, 0);
   for (unsigned binding = 0; binding < 33; ++binding) {
      nir_def *value =
         nir_load_ubo(&b, 1, 32, nir_imm_int(&b, binding), nir_imm_int(&b, 0),
                      .align_mul = 4, .range = 4);
      sum = nir_fadd(&b, sum, value);
   }
   nir_store_output(
      &b, nir_vec4(&b, sum, sum, sum, nir_imm_float(&b, 1)), nir_imm_int(&b, 0),
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.num_ubos = 33;
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   EXPECT_FALSE(agx_compile_apple9_fragment(b.shader, &compiled, nullptr));
   EXPECT_EQ(compiled.binary, nullptr);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, VertexInputsKeepAttributeAndUniformBindingsDistinct)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "vertex_buffers_and_uniforms");
   nir_def *position = nir_load_input(
      &b, 4, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VERT_ATTRIB_GENERIC3, .num_slots = 1});
   nir_def *color = nir_load_input(
      &b, 3, 32, nir_imm_int(&b, 0), .base = 1, .dest_type = nir_type_float32,
      .io_semantics = {.location = VERT_ATTRIB_GENERIC9, .num_slots = 1});
   nir_def *scale =
      nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 7), nir_imm_int(&b, 16),
                   .align_mul = 4, .range = 4);
   nir_store_output(
      &b, position, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_store_output(
      &b, nir_fmul(&b, color, scale), nir_imm_int(&b, 0), .write_mask = 7,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_apple9_vertex_layout layout = {};
   layout.stride[0] = 20;
   layout.format[0] = PIPE_FORMAT_R32G32B32_FLOAT; /* A vec4 shader input defaults W to one. */
   layout.stride[1] = 32;
   layout.format[1] = PIPE_FORMAT_R32G32B32A32_FLOAT;
   layout.buffer[1] = 1;
   layout.clip_halfz = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(
      agx_compile_apple9_vertex_inputs(b.shader, &layout, &compiled, &reason))
      << (reason ?: "");
   EXPECT_EQ(compiled.info.apple9_resource_count, 3);
   EXPECT_EQ(compiled.info.apple9_resource_binding[0], 33);
   EXPECT_EQ(compiled.info.apple9_resource_binding[1], 32);
   EXPECT_EQ(compiled.info.apple9_resource_binding[2], 7);
   EXPECT_EQ(compiled.info.apple9_ubo_mask, 1u << 7);
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            EXPECT_NE(nir_instr_as_intrinsic(instr)->intrinsic,
                      nir_intrinsic_load_input);
         }
      }
   }
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, PackedVertexInputsShareSparseBufferBinding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "vertex_buffers_and_uniforms");
   nir_def *position = nir_load_input(
      &b, 4, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VERT_ATTRIB_GENERIC3, .num_slots = 1});
   nir_def *color = nir_load_input(
      &b, 3, 32, nir_imm_int(&b, 0), .base = 1, .dest_type = nir_type_float32,
      .io_semantics = {.location = VERT_ATTRIB_GENERIC9, .num_slots = 1});
   nir_def *scale =
      nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 7), nir_imm_int(&b, 16),
                   .align_mul = 4, .range = 4);
   nir_store_output(
      &b, position, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_def *scaled = nir_fmul(&b, color, scale);
   nir_store_output(
      &b, nir_vec4(&b, nir_channel(&b, scaled, 0), nir_channel(&b, scaled, 1),
                   nir_channel(&b, scaled, 2), nir_undef(&b, 1, 32)),
      nir_imm_int(&b, 0), .write_mask = 7,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_store_output(&b, nir_imm_float(&b, 4), nir_imm_int(&b, 0),
                    .write_mask = 1, .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_PSIZ, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_apple9_vertex_layout layout = {};
   layout.stride[0] = 24;
   layout.offset[0] = 8;
   layout.buffer[0] = 7;
   layout.format[0] = PIPE_FORMAT_R16G16_UNORM; /* Missing Z/W default to zero/one. */
   layout.stride[1] = 24;
   layout.offset[1] = 12;
   layout.format[1] = PIPE_FORMAT_R8G8B8A8_SNORM;
   layout.buffer[1] = 7;
   layout.clip_halfz = true;
   layout.ignore_point_size = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(
      agx_compile_apple9_vertex_inputs(b.shader, &layout, &compiled, &reason))
      << (reason ?: "");
   EXPECT_EQ(compiled.info.apple9_resource_count, 2);
   EXPECT_EQ(compiled.info.apple9_resource_binding[0], 39);
   EXPECT_EQ(compiled.info.apple9_resource_binding[1], 7);
   EXPECT_EQ(compiled.info.apple9_ubo_mask, 1u << 7);
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            EXPECT_NE(nir_instr_as_intrinsic(instr)->intrinsic,
                      nir_intrinsic_load_input);
         }
      }
   }
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, SparseVaryingsLinkBySemanticComponent)
{
   nir_builder vs = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "sparse_vertex_outputs");
   nir_store_output(&vs, nir_vec4(&vs, nir_imm_float(&vs, 0), nir_imm_float(&vs, 0),
                                nir_imm_float(&vs, .5), nir_imm_float(&vs, 1)),
                    nir_imm_int(&vs, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   /* Write out of semantic order, with holes and a constant array offset. */
   nir_store_output(&vs, nir_vec3(&vs, nir_imm_float(&vs, .1),
                                nir_imm_float(&vs, .2), nir_imm_float(&vs, .3)),
                    nir_imm_int(&vs, 2), .write_mask = 7,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_VAR7, .num_slots = 3});
   nir_store_output(&vs, nir_vec4(&vs, nir_imm_float(&vs, 0), nir_imm_float(&vs, .4),
                                nir_imm_float(&vs, 0), nir_imm_float(&vs, .5)),
                    nir_imm_int(&vs, 0), .write_mask = 10,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 1});
   nir_store_output(&vs, nir_imm_float(&vs, .6), nir_imm_int(&vs, 0),
                    .write_mask = 1, .component = 3, .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_VAR31, .num_slots = 1});
   vs.shader->info.io_lowered = true;
   agx_shader_part vertex = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_vertex(vs.shader, &vertex, &reason)) << reason;
   EXPECT_EQ(vertex.info.apple9_varyings.count, 6u);
   EXPECT_EQ(vertex.info.apple9_varyings.mask[VARYING_SLOT_VAR3], 10u);
   EXPECT_EQ(vertex.info.apple9_varyings.mask[VARYING_SLOT_VAR9], 7u);
   EXPECT_EQ(vertex.info.apple9_varyings.mask[VARYING_SLOT_VAR31], 8u);
   const uint8_t *code = static_cast<const uint8_t *>(vertex.binary);
   unsigned slots = 0;
   for (unsigned i = 0; i + 8 <= vertex.info.binary_size; ++i) {
      if (code[i] == 0x57 && (code[i+1] & 15) == 6 && (code[i+2] & 0xfc) == 0x54)
         slots |= 1u << ((code[i+4] >> 5) | ((code[i+5] & 1) << 3));
   }
   EXPECT_EQ(slots, 0x3ffu);

   nir_builder fs = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "sparse_fragment_inputs");
   nir_def *bary = nir_load_barycentric_pixel(&fs, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *a = nir_load_interpolated_input(
      &fs, 1, 32, bary, nir_imm_int(&fs, 0), .component = 3,
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 1});
   nir_def *b = nir_load_interpolated_input(
      &fs, 1, 32, bary, nir_imm_int(&fs, 2), .component = 2,
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR7, .num_slots = 3});
   nir_def *c = nir_load_interpolated_input(
      &fs, 1, 32, bary, nir_imm_int(&fs, 0), .component = 3,
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR31, .num_slots = 1});
   nir_store_output(&fs, nir_vec4(&fs, a, b, c, nir_imm_float(&fs, 1)),
                    nir_imm_int(&fs, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   fs.shader->info.io_lowered = true;
   nir_shader *missing = nir_shader_clone(NULL, fs.shader);
   nir_shader *relocated = nir_shader_clone(NULL, fs.shader);
   agx_shader_part fragment = {};
   ASSERT_TRUE(agx_compile_apple9_fragment_inputs(fs.shader,
      &vertex.info.apple9_varyings, &fragment, &reason)) << reason;
   EXPECT_EQ(fragment.info.varyings.fs.nr_cf, 7u);
   code = static_cast<const uint8_t *>(fragment.binary);
   unsigned coefficients = 0;
   for (unsigned i = 0; i + 10 <= fragment.info.binary_size; ++i) {
      if (code[i] == 0x2f && (code[i+1] == 5 || code[i+1] == 13))
         coefficients |= 1u << (code[i+5] >> 1);
   }
   EXPECT_EQ(coefficients, (1u << 0) | (1u << 2) | (1u << 5) | (1u << 6));
   free(fragment.binary);
   /* The same FS has different coefficient indices with another producer,
    * including an output which this FS does not consume. */
   auto shifted = vertex.info.apple9_varyings;
   shifted.mask[VARYING_SLOT_VAR0] = 1;
   shifted.count++;
   ASSERT_TRUE(agx_compile_apple9_fragment_inputs(relocated, &shifted,
                                                  &fragment, &reason)) << reason;
   EXPECT_EQ(fragment.info.varyings.fs.nr_cf, 8u);
   code = static_cast<const uint8_t *>(fragment.binary);
   coefficients = 0;
   for (unsigned i = 0; i + 10 <= fragment.info.binary_size; ++i) {
      if (code[i] == 0x2f && (code[i+1] == 5 || code[i+1] == 13))
         coefficients |= 1u << (code[i+5] >> 1);
   }
   EXPECT_EQ(coefficients, (1u << 0) | (1u << 3) | (1u << 6) | (1u << 7));
   free(fragment.binary);
   auto invalid = vertex.info.apple9_varyings;
   invalid.mask[VARYING_SLOT_VAR31] = 0;
   invalid.count--;
   EXPECT_FALSE(agx_compile_apple9_fragment_inputs(missing, &invalid, &fragment, &reason));
   EXPECT_EQ(fragment.binary, nullptr);
   EXPECT_NE(strstr(reason, "not written"), nullptr);
   free(vertex.binary);
   ralloc_free(vs.shader);
   ralloc_free(fs.shader);
   ralloc_free(missing);
   ralloc_free(relocated);
}

TEST(Apple9Compiler, LegacyAndGenericVaryingsLinkBySemanticComponent)
{
   nir_builder vs = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "sparse_vertex_outputs");
   nir_store_output(&vs, nir_vec4(&vs, nir_imm_float(&vs, 0), nir_imm_float(&vs, 0),
                                nir_imm_float(&vs, .5), nir_imm_float(&vs, 1)),
                    nir_imm_int(&vs, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   /* Write out of semantic order, with holes and a constant array offset. */
   nir_store_output(&vs, nir_vec3(&vs, nir_imm_float(&vs, .1),
                                nir_imm_float(&vs, .2), nir_imm_float(&vs, .3)),
                    nir_imm_int(&vs, 2), .write_mask = 7,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_TEX0, .num_slots = 3});
   nir_store_output(&vs, nir_vec4(&vs, nir_imm_float(&vs, 0), nir_imm_float(&vs, .4),
                                nir_imm_float(&vs, 0), nir_imm_float(&vs, .5)),
                    nir_imm_int(&vs, 0), .write_mask = 10,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_COL0, .num_slots = 1});
   nir_store_output(&vs, nir_imm_float(&vs, .6), nir_imm_int(&vs, 0),
                    .write_mask = 1, .component = 3, .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_VAR31, .num_slots = 1});
   vs.shader->info.io_lowered = true;
   agx_shader_part vertex = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_vertex(vs.shader, &vertex, &reason)) << reason;
   EXPECT_EQ(vertex.info.apple9_varyings.count, 6u);
   EXPECT_EQ(vertex.info.apple9_varyings.mask[VARYING_SLOT_COL0], 10u);
   EXPECT_EQ(vertex.info.apple9_varyings.mask[VARYING_SLOT_TEX2], 7u);
   EXPECT_EQ(vertex.info.apple9_varyings.mask[VARYING_SLOT_VAR31], 8u);
   const uint8_t *code = static_cast<const uint8_t *>(vertex.binary);
   unsigned slots = 0;
   for (unsigned i = 0; i + 8 <= vertex.info.binary_size; ++i) {
      if (code[i] == 0x57 && (code[i+1] & 15) == 6 && (code[i+2] & 0xfc) == 0x54)
         slots |= 1u << ((code[i+4] >> 5) | ((code[i+5] & 1) << 3));
   }
   EXPECT_EQ(slots, 0x3ffu);

   nir_builder fs = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "sparse_fragment_inputs");
   nir_def *bary = nir_load_barycentric_pixel(&fs, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *a = nir_load_interpolated_input(
      &fs, 1, 32, bary, nir_imm_int(&fs, 0), .component = 3,
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_COL0, .num_slots = 1});
   nir_def *b = nir_load_interpolated_input(
      &fs, 1, 32, bary, nir_imm_int(&fs, 2), .component = 2,
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TEX0, .num_slots = 3});
   nir_def *c = nir_load_interpolated_input(
      &fs, 1, 32, bary, nir_imm_int(&fs, 0), .component = 3,
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR31, .num_slots = 1});
   nir_store_output(&fs, nir_vec4(&fs, a, b, c, nir_imm_float(&fs, 1)),
                    nir_imm_int(&fs, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   fs.shader->info.io_lowered = true;
   nir_shader *missing = nir_shader_clone(NULL, fs.shader);
   nir_shader *relocated = nir_shader_clone(NULL, fs.shader);
   agx_shader_part fragment = {};
   ASSERT_TRUE(agx_compile_apple9_fragment_inputs(fs.shader,
      &vertex.info.apple9_varyings, &fragment, &reason)) << reason;
   EXPECT_EQ(fragment.info.varyings.fs.nr_cf, 7u);
   code = static_cast<const uint8_t *>(fragment.binary);
   unsigned coefficients = 0;
   for (unsigned i = 0; i + 10 <= fragment.info.binary_size; ++i) {
      if (code[i] == 0x2f && (code[i+1] == 5 || code[i+1] == 13))
         coefficients |= 1u << (code[i+5] >> 1);
   }
   EXPECT_EQ(coefficients, (1u << 0) | (1u << 2) | (1u << 5) | (1u << 6));
   free(fragment.binary);
   /* The same FS has different coefficient indices with another producer,
    * including an output which this FS does not consume. */
   auto shifted = vertex.info.apple9_varyings;
   shifted.mask[VARYING_SLOT_COL0] |= 1;
   shifted.count++;
   ASSERT_TRUE(agx_compile_apple9_fragment_inputs(relocated, &shifted,
                                                  &fragment, &reason)) << reason;
   EXPECT_EQ(fragment.info.varyings.fs.nr_cf, 8u);
   code = static_cast<const uint8_t *>(fragment.binary);
   coefficients = 0;
   for (unsigned i = 0; i + 10 <= fragment.info.binary_size; ++i) {
      if (code[i] == 0x2f && (code[i+1] == 5 || code[i+1] == 13))
         coefficients |= 1u << (code[i+5] >> 1);
   }
   EXPECT_EQ(coefficients, (1u << 0) | (1u << 3) | (1u << 6) | (1u << 7));
   free(fragment.binary);
   auto invalid = vertex.info.apple9_varyings;
   invalid.mask[VARYING_SLOT_VAR31] = 0;
   invalid.count--;
   EXPECT_FALSE(agx_compile_apple9_fragment_inputs(missing, &invalid, &fragment, &reason));
   EXPECT_EQ(fragment.binary, nullptr);
   EXPECT_NE(strstr(reason, "not written"), nullptr);
   free(vertex.binary);
   ralloc_free(vs.shader);
   ralloc_free(fs.shader);
   ralloc_free(missing);
   ralloc_free(relocated);
}

TEST(Apple9Compiler, VaryingPublicationCapacityIsCheckedBeforeAllocation)
{
   for (unsigned count : {0u, 12u, 16u, 24u, 32u, 33u, 47u, 48u, 49u, 63u, 64u, 65u, 95u, 96u, 97u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "varying_capacity");
      nir_store_output(&b, nir_imm_vec4(&b, 0, 0, .5, 1), nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      for (unsigned i = 0; i < count; ++i) {
         nir_store_output(&b, nir_imm_float(&b, i * .0625), nir_imm_int(&b, 0),
                          .write_mask = 1, .component = i % 4,
                          .src_type = nir_type_float32,
                          .io_semantics = {.location = (uint8_t)(VARYING_SLOT_VAR0+i/4), .num_slots = 1});
      }
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      bool ok = agx_compile_apple9_vertex(b.shader, &compiled, &reason);
      if (count <= AGX_APPLE9_MAX_VARYING_COMPONENTS) {
         ASSERT_TRUE(ok) << reason;
         EXPECT_EQ(compiled.info.apple9_varyings.count, count);
         free(compiled.binary);
      } else {
         EXPECT_FALSE(ok);
         EXPECT_EQ(compiled.binary, nullptr);
         EXPECT_NE(strstr(reason, "96 user"), nullptr);
      }
      ralloc_free(b.shader);
   }
}

TEST(Apple9Encoding, TileReadMaskUsesDedicatedLastUseBit)
{
   /* EXP-M4-75: the mask addresses a 16-bit register half. Its last use is
    * instruction bit 69, separate from both the address and result slot. */
   agx_apple9_vir_instr load = {};
   load.op = AGX_APPLE9_VIR_TILE_LOAD;
   load.encoding = AGX_APPLE9_ENC_TILE_LOAD_MASK;
   load.dest = 0;
   load.src[0] = 1;
   load.nr_srcs = 1;
   uint8_t phys[] = {9, 7};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   const uint8_t slots[][2] = {
      {0x4e, 0}, {0xce, 0}, {0x4e, 1},
      {0xce, 1}, {0x4e, 2}, {0xce, 2},
   };
   for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
      load.producer_scoreboard_slot =
         (agx_apple9_scoreboard_slot)(AGX_APPLE9_SCOREBOARD_SLOT_1 + slot);
      for (unsigned live : {0u, 1u}) {
         load.live_after_mask = live;
         const uint8_t expected[] = {
            0x67, 0x06, 0x54, 18, 0, 0, 14, slots[slot][0],
            uint8_t(slots[slot][1] | (live ? 0 : 0x20)), 0, 0, 0x20,
         };
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
         ASSERT_EQ(packed.length, sizeof(expected));
         EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0)
            << "slot=" << slot + 1 << " live=" << live;
      }
   }
}

TEST(Apple9Encoding, TileReadHasAnExplicitResultSlot)
{
   agx_apple9_vir_instr load = {};
   load.op = AGX_APPLE9_VIR_TILE_LOAD;
   load.encoding = AGX_APPLE9_ENC_TILE_LOAD;
   load.dest = 0;
   load.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
   uint8_t phys[] = {0};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
   const uint8_t native[] = {0x67,0x0e,0x54,0,0,0,1,0xce,2,0,0,0};
   ASSERT_EQ(packed.length, sizeof(native));
   EXPECT_EQ(memcmp(packed.bytes, native, sizeof(native)), 0);
   load.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
   load.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1;
   phys[0] = 25;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
   EXPECT_EQ(packed.bytes[3], 50);
   EXPECT_EQ(packed.bytes[7], 0x4e);
   EXPECT_EQ(packed.bytes[8], 0);
   phys[0] = 95;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
   EXPECT_EQ(packed.bytes[3], 190);
   phys[0] = 96;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
}

TEST(Apple9Allocator, SharedTileMaskIsReleasedOnlyAfterItsLastUse)
{
   for (bool keep_mask : {false, true}) {
      agx_apple9_vir_program program;
      agx_apple9_vir_init(&program);
      uint32_t mask = agx_apple9_vir_input(&program, 16);
      uint32_t first = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_TILE_LOAD,
         AGX_APPLE9_ENC_TILE_LOAD_MASK, &mask, 1, 0);
      uint32_t last = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_TILE_LOAD,
         AGX_APPLE9_ENC_TILE_LOAD_MASK, &mask, 1, 1);
      auto *first_load = program.instructions[0];
      auto *last_load = program.instructions[1];
      first_load->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
      last_load->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
      uint32_t sources[] = {first, last};
      program.output = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IOR,
         AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
      if (keep_mask) {
         uint32_t later[] = {program.output, mask};
         program.output = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IOR,
            AGX_APPLE9_ENC_LOGIC_EXTENDED, later, 2, 0);
      }
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason)) << reason;
      ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
      ASSERT_TRUE(agx_apple9_validate_vir_allocation(&program, &reason)) << reason;
      for (auto *load : {first_load, last_load}) {
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(load, program.phys,
                                                   &packed, &reason)) << reason;
         EXPECT_EQ(packed.bytes[6], program.phys[mask] << 1);
         EXPECT_EQ(packed.bytes[8] & 0x20,
                   load == last_load && !keep_mask ? 0x20 : 0);
      }
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Allocator, TileReadNormalizesAndRetainsTheFirstLogicConsumer)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t mask = agx_apple9_vir_input(&program, 16);
   uint32_t tile = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_TILE_LOAD,
      AGX_APPLE9_ENC_TILE_LOAD, nullptr, 0, 0);
   program.instructions[0]->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
   uint32_t src[] = {mask, tile};
   uint32_t red = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IAND,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   uint32_t later[] = {tile, red};
   program.output = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, later, 2, 0);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason)) << reason;
   EXPECT_EQ(program.instructions[1]->src[0], tile);
   EXPECT_EQ(program.instructions[1]->scoreboard_slot,
             program.instructions[0]->producer_scoreboard_slot);
   EXPECT_EQ(program.instructions[2]->scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_NONE);
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
   EXPECT_NE(program.instructions[1]->live_after_mask & 1u, 0u);
   EXPECT_TRUE(agx_apple9_validate_vir_allocation(&program, &reason)) << reason;
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Allocator, TileReadMaterializesBeforeAStoreWithoutAWaitField)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t tile = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_TILE_LOAD,
      AGX_APPLE9_ENC_TILE_LOAD, nullptr, 0, 0);
   program.instructions[0]->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&program, AGX_APPLE9_VIR_TILE_STORE,
      AGX_APPLE9_ENC_TILE_STORE, &tile, 1, 0));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason)) << reason;
   ASSERT_EQ(program.instruction_count, 3u);
   EXPECT_TRUE(program.instructions[1]->scoreboard_materialize);
   EXPECT_EQ(program.instructions[1]->scoreboard_slot,
             program.instructions[0]->producer_scoreboard_slot);
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
   EXPECT_TRUE(agx_apple9_validate_vir_allocation(&program, &reason)) << reason;
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Compiler, GraphicsLoopResultsReachMergedOutputs)
{
   for (bool fragment : {false, true}) {
      for (bool continuation : {false, true}) {
         nir_builder b = nir_builder_init_simple_shader(
            fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX,
            &agx_nir_options, "graphics_loop_output");
         b.shader->info.num_ubos = 1;
         nir_def *limit = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
            nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
         nir_def *initial = nir_imm_int(&b, 0);
         nir_loop *loop = nir_push_loop(&b);
         if (continuation)
            nir_loop_add_continue_construct(loop);
         nir_block *entry = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
         nir_block *header = nir_loop_first_block(loop);
         nir_phi_instr *phi = nir_phi_instr_create(b.shader);
         nir_def_init(&phi->instr, &phi->def, 1, 32);
         nir_phi_instr_add_src(phi, entry, initial);
         nir_break_if(&b, nir_uge(&b, &phi->def, limit));
         if (continuation) {
            nir_if *skip = nir_push_if(&b, nir_ieq_imm(&b, &phi->def, 1));
            nir_jump(&b, nir_jump_continue);
            nir_pop_if(&b, skip);
            nir_push_continue(&b, loop);
         }
         nir_def *next = nir_iadd_imm(&b, &phi->def, 1);
         nir_phi_instr_add_src(phi, nir_cursor_current_block(b.cursor), next);
         nir_pop_loop(&b, loop);
         b.cursor = nir_after_phis(header);
         nir_builder_instr_insert(&b, &phi->instr);
         b.cursor = nir_after_cf_node(&loop->cf_node);
         nir_if *choose = nir_push_if(&b, nir_ult_imm(&b, &phi->def, 2));
         nir_def *yes = nir_fmul_imm(&b, nir_u2f32(&b, &phi->def), .125);
         nir_push_else(&b, choose);
         nir_def *no = nir_fmul_imm(&b, nir_u2f32(&b, &phi->def), .25);
         nir_pop_if(&b, choose);
         nir_def *value = nir_if_phi(&b, yes, no);
         nir_store_output(&b, nir_vec4(&b, value, value, value, nir_imm_float(&b, 1)),
            nir_imm_int(&b, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = unsigned(fragment ? FRAG_RESULT_DATA0 : VARYING_SLOT_POS)});
         b.shader->info.io_lowered = true;
         agx_shader_part compiled = {};
         const char *reason = nullptr;
         bool ok = fragment ? agx_compile_apple9_fragment(b.shader, &compiled, &reason)
                            : agx_compile_apple9_vertex(b.shader, &compiled, &reason);
         ASSERT_TRUE(ok) << (reason ?: "") << " fragment=" << fragment
                        << " continue=" << continuation;
         EXPECT_GT(compiled.info.binary_size, 4u);
         free(compiled.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, MultipleColorTargetsUseDistinctTileAddresses)
{
   for (unsigned count : {2u, 4u, 8u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "multiple_color_targets");
      b.shader->info.num_ubos = 1;
      nir_def *value = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .align_mul = 16, .range = 16);
      for (unsigned rt = 0; rt < count; ++rt) {
         nir_store_output(&b, nir_fadd_imm(&b, value, rt * .0625),
            nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = FRAG_RESULT_DATA0 + rt});
      }
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_mrt(
         b.shader, nullptr, nullptr, count, &compiled, &reason)) << (reason ?: "");
      unsigned stores[8] = {};
      const uint8_t *code = (const uint8_t *)compiled.binary;
      for (unsigned i = 0; i + 12 <= compiled.info.binary_size; ++i) {
         if (code[i] == 0xe7 && code[i + 1] == 0x16 && code[i + 2] == 0x54 &&
             code[i + 11] == 8 && code[i + 7] == 0x4e && code[i + 5] < 16)
            ++stores[code[i + 5] >> 1];
      }
      for (unsigned rt = 0; rt < 8; ++rt)
         EXPECT_EQ(stores[rt], unsigned(rt < count)) << "RT=" << rt;
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, UndefinedFragmentAlphaCanBeTrimmedBeforePacking)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "undefined_fragment_alpha");
   nir_store_output(&b,
                    nir_vec4(&b, nir_imm_float(&b, .5), nir_imm_float(&b, 1.5),
                             nir_imm_float(&b, 0), nir_undef(&b, 1, 32)),
                    nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0});
   b.shader->info.io_lowered = true;
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
      << (reason ?: "");
   EXPECT_GT(compiled.info.binary_size, 4u);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, LoweredLegacyFragColorBroadcastsToEveryTarget)
{
   for (unsigned count : {2u, 4u, 8u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "legacy_color_broadcast");
      nir_store_output(&b, nir_imm_vec4(&b, .25, .5, .75, 1),
                       nir_imm_int(&b, 0), .write_mask = 15,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_COLOR});
      b.shader->info.io_lowered = true;
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_COLOR);
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_mrt(b.shader, nullptr, nullptr,
                                                  count, &compiled, &reason))
         << (reason ?: "");
      unsigned stores = 0;
      const uint8_t *code = (const uint8_t *)compiled.binary;
      for (unsigned i = 0; i + 12 <= compiled.info.binary_size; ++i) {
         if (code[i] == 0xe7 && code[i + 1] == 0x16 && code[i + 2] == 0x54 &&
             code[i + 11] == 8 && code[i + 7] == 0x4e && code[i + 5] < 16)
            stores |= 1u << (code[i + 5] >> 1);
      }
      EXPECT_EQ(stores, (1u << count) - 1);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, LegacyFragColorUsesTheSingleRenderTarget)
{
   nir_shader *direct = apple9_render_test_shader(true);
   nir_shader *legacy = nir_shader_clone(NULL, direct);
   nir_foreach_block(block, nir_shader_get_entrypoint(legacy)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output) continue;
         nir_io_semantics semantics = nir_intrinsic_io_semantics(intr);
         semantics.location = FRAG_RESULT_COLOR;
         nir_intrinsic_set_io_semantics(intr, semantics);
      }
   }
   legacy->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_COLOR);
   agx_shader_part a = {}, b = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(direct, &a, &reason)) << reason;
   ASSERT_TRUE(agx_compile_apple9_fragment(legacy, &b, &reason)) << reason;
   ASSERT_EQ(a.info.binary_size, b.info.binary_size);
   EXPECT_EQ(memcmp(a.binary, b.binary, a.info.binary_size), 0);
   free(a.binary); free(b.binary);
   ralloc_free(direct); ralloc_free(legacy);
}

TEST(Apple9Compiler, FragmentPositionForms)
{
   /* Exercise live component tracking: a vec4 load with only XY live is legal,
    * and consuming Z/W uses dedicated rasterizer coefficients. */
   for (unsigned form = 0; form < 3; ++form) {
      for (unsigned component = 0; component < 4; ++component) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "fragment_position");
         nir_def *position;
         if (form == 0) {
            position = nir_load_frag_coord(&b);
         } else if (form == 1) {
            position = nir_load_interpolated_input(
               &b, 4, 32,
               nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH),
               nir_imm_int(&b, 0), .dest_type = nir_type_float32,
               .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         } else {
            position = nir_load_input(
               &b, 4, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
               .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         }
         nir_def *value = nir_channel(&b, position, component);
         nir_store_output(&b, nir_vec4(&b, value, value, value, nir_imm_float(&b, 1)),
                          nir_imm_int(&b, 0), .write_mask = 15,
                          .src_type = nir_type_float32,
                          .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
         b.shader->info.io_lowered = true;
         agx_shader_part out = {};
         const char *reason = nullptr;
         bool ok = agx_compile_apple9_fragment(b.shader, &out, &reason);
         EXPECT_TRUE(ok) << "form=" << form << " component=" << component
                                      << " reason=" << (reason ? reason : "none");
         if (ok) {
            EXPECT_EQ(out.info.apple9_varyings.count, 0u);
            EXPECT_EQ(out.info.apple9_reads_z, component == 2);
            EXPECT_EQ(out.info.varyings.fs.nr_cf, component == 2 ? 2u : 1u);
         }
         free(out.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, MixLowersDynamicEndpointsAndFactors)
{
   for (bool exact : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "mix_dynamic");
      nir_def *args[3];
      for (unsigned i = 0; i < 3; ++i)
         args[i] = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
                                nir_imm_int(&b, i * 16), .align_mul = 16,
                                .range_base = i * 16, .range = 16);
      nir_def *value = nir_flrp(&b, args[0], args[1], args[2]);
      nir_def_as_alu(value)->fp_math_ctrl = exact ? nir_fp_exact : nir_fp_fast_math;
      nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 15,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      b.shader->info.num_ubos = 1;
      agx_shader_part out = {};
      const char *reason = nullptr;
      EXPECT_TRUE(agx_compile_apple9_fragment(b.shader, &out, &reason)) << reason;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               EXPECT_NE(nir_instr_as_alu(instr)->op, nir_op_flrp);
            }
         }
      }
      free(out.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, GraphicsTrigUsesOneFactorPerOperation)
{
   for (bool fragment : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX,
         &agx_nir_options, "compact_graphics_trig");
      nir_def *x = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                               .align_mul = 4, .range = 4);
      nir_store_output(&b, nir_vec4(&b, nir_fsin(&b, x), nir_fcos(&b, x),
                                    nir_imm_float(&b, 0), nir_imm_float(&b, 1)),
                       nir_imm_int(&b, 0), .write_mask = 15,
                       .src_type = nir_type_float32,
                       .io_semantics = {.location = fragment ? unsigned(FRAG_RESULT_DATA0) : unsigned(VARYING_SLOT_POS), .num_slots = 1});
      b.shader->info.io_lowered = true;
      b.shader->info.num_ubos = 1;
      agx_shader_part out = {};
      const char *reason = nullptr;
      ASSERT_TRUE(fragment ? agx_compile_apple9_fragment(b.shader, &out, &reason)
                           : agx_compile_apple9_vertex(b.shader, &out, &reason)) << reason;
      unsigned factors = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu)
               factors += nir_instr_as_alu(instr)->op == nir_op_fsin_factor_agx;
         }
      }
      EXPECT_EQ(factors, 2u);
      EXPECT_LT(out.info.binary_size, 2048u);
      free(out.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, TextureCoordinatesAndResultUseAllocatedRegisters)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_texture");
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 3;
   tex->sampler_index = 5;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(nir_vec2(
      &b, nir_imm_float(&b, .375), nir_imm_float(&b, .875)));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_store_output(&b, &tex->def, nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason)) << (reason ?: "");
   EXPECT_EQ(compiled.info.apple9_texture_mask, 1u << 3);
   EXPECT_EQ(compiled.info.apple9_sampler_mask, 1u << 5);
   EXPECT_TRUE(compiled.info.disable_tri_merging);
   unsigned samples = 0;
   const auto *code = static_cast<const uint8_t *>(compiled.binary);
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      if ((code[i] & 7) == 5 && code[i+2] == 0x0c && code[i+3] == 0xb8 &&
          code[i+4] == 0xb0 && code[i+12] == 1) {
         EXPECT_LE(code[i] >> 3, 12);
         EXPECT_EQ(code[i+1] & 1, 0);
         EXPECT_LE(code[i+1] & 0x7f, 6);
         ++samples;
      }
   }
   EXPECT_EQ(samples, 1u);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, VertexTextureSamplingUsesExplicitLod)
{
   for (auto op : {nir_texop_tex, nir_texop_txl}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "apple9_vertex_texture");
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, op == nir_texop_txl ? 2 : 1);
      tex->op = op;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->texture_index = 3;
      tex->sampler_index = 5;
      tex->src[0].src_type = nir_tex_src_coord;
      tex->src[0].src = nir_src_for_ssa(nir_imm_vec2(&b, .375, .625));
      if (op == nir_texop_txl) {
         tex->src[1].src_type = nir_tex_src_lod;
         tex->src[1].src = nir_src_for_ssa(nir_imm_float(&b, 2));
      }
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_output(&b, &tex->def, nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_output(&b, nir_imm_vec4(&b, 0, 0, .5, 1), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_vertex(b.shader, &compiled, &reason))
         << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_texture_mask, 1u << 3);
      EXPECT_EQ(compiled.info.apple9_sampler_mask, 1u << 5);
      EXPECT_FALSE(compiled.info.disable_tri_merging);
      unsigned samples = 0;
      const auto *code = static_cast<const uint8_t *>(compiled.binary);
      for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
         if ((code[i] & 7) == 5 && code[i + 2] == 0x0c && code[i + 3] == 0xb8 &&
             code[i + 4] == 0xb0 && code[i + 12] == 1) {
            EXPECT_EQ(code[i + 6], 0);
            EXPECT_EQ(code[i + 7], 1); /* Explicit LOD, never quad derivatives. */
            ++samples;
         }
      }
      EXPECT_EQ(samples, 1u);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, VolumeCubeArrayAndShadowAcceptDynamicCoordinatesAndLod)
{
   for (unsigned kind = 0; kind < 5; ++kind)
   for (auto op : {nir_texop_txl, nir_texop_txb})
   for (bool reverse : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_texture_dimensions");
      b.shader->info.num_ubos = 1;
      nir_def *input = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .align_mul = 16, .range = 16);
      bool shadow = kind == 2 || kind == 3;
      bool scalar = kind == 3;
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, shadow ? 3 : 2);
      tex->op = op;
      tex->sampler_dim = kind == 0 ? GLSL_SAMPLER_DIM_3D :
                         kind == 1 ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->is_shadow = shadow;
      tex->is_array = kind == 4;
      tex->is_new_style_shadow = scalar;
      tex->coord_components = shadow ? 2 : 3;
      tex->texture_index = 3;
      tex->sampler_index = 7;
      unsigned c = reverse ? 1 : 0;
      tex->src[c].src_type = nir_tex_src_coord;
      tex->src[c].src = nir_src_for_ssa(nir_trim_vector(&b, input, tex->coord_components));
      tex->src[1-c].src_type = op == nir_texop_txl ? nir_tex_src_lod : nir_tex_src_bias;
      tex->src[1-c].src = nir_src_for_ssa(nir_channel(&b, input, 3));
      if (shadow) {
         tex->src[2].src_type = nir_tex_src_comparator;
         tex->src[2].src = nir_src_for_ssa(nir_channel(&b, input, 2));
      }
      nir_def_init(&tex->instr, &tex->def, scalar ? 1 : 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_def *color = scalar ? nir_vec4(&b, &tex->def, &tex->def,
                                         &tex->def, nir_imm_float(&b, 1)) : &tex->def;
      nir_store_output(&b, color, nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
         << "kind=" << kind << " op=" << op << " " << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_texture_mask, 1u << 3);
      EXPECT_EQ(compiled.info.apple9_sampler_mask, 1u << 7);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Vir, FixedTextureSlotRetiresConflictingAutomaticLoad)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   uint32_t coords[] = {agx_apple9_vir_input(&p, 2), agx_apple9_vir_input(&p, 3)};
   uint32_t one = agx_apple9_vir_input(&p, 4);
   uint32_t loads[2];
   for (unsigned i = 0; i < 2; ++i) {
      loads[i] = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_DEVICE_LOAD,
         AGX_APPLE9_ENC_DEVICE_LOAD, &one, 1, i);
      ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(
         &p, loads[i], 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   }
   uint32_t sample = agx_apple9_vir_emit_texture_sample(&p, coords, one, 0, 0);
   uint32_t sources[] = {loads[0], sample};
   uint32_t sum = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
   sources[0] = loads[1];
   sources[1] = sum;
   p.output = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&p, &reason)) << (reason ?: "");
   unsigned copies_before_sample = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      if (p.instructions[i]->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE)
         break;
      copies_before_sample += p.instructions[i]->scoreboard_materialize;
   }
   EXPECT_EQ(copies_before_sample, 1u);
   ASSERT_TRUE(agx_apple9_allocate_vir(&p, &reason)) << (reason ?: "");
   agx_apple9_vir_finish(&p);
}

TEST(Apple9Vir, SampleMaterializationPreservesProducerBlockAndBranchTargets)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   uint32_t coords[] = {agx_apple9_vir_input(&p, 2), agx_apple9_vir_input(&p, 3)};
   uint32_t one = agx_apple9_vir_input(&p, 4);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_JMP_EXEC_NONE,
      AGX_APPLE9_ENC_JMP_EXEC_NONE, nullptr, 0, 0));
   uint32_t sample = agx_apple9_vir_emit_texture_sample(&p, coords, one, 0, 0);
   ASSERT_NE(sample, AGX_APPLE9_VREG_INVALID);
   auto *marker = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, marker);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_LOOP_MASK_UPDATE,
      AGX_APPLE9_ENC_LOOP_MASK_UPDATE, nullptr, 0, 0x22));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_JMP_EXEC_ANY,
      AGX_APPLE9_ENC_JMP_EXEC_ANY, nullptr, 0, 0));
   p.instructions[0]->branch_target = marker;
   p.instructions[p.instruction_count - 1]->branch_target = marker;
   p.output = sample;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&p, &reason)) << (reason ?: "");
   unsigned copies = 0, jumps = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      auto &ins = *p.instructions[i];
      copies += ins.op == AGX_APPLE9_VIR_IOR;
      if (ins.op == AGX_APPLE9_VIR_JMP_EXEC_ANY || ins.op == AGX_APPLE9_VIR_JMP_EXEC_NONE) {
         ASSERT_EQ(ins.branch_target, marker);
         EXPECT_EQ(p.instructions[marker->start_index]->op, AGX_APPLE9_VIR_LOOP_MASK_UPDATE);
         ++jumps;
      }
   }
   EXPECT_EQ(copies, 4u);
   EXPECT_EQ(jumps, 2u);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9Compiler, BooleanUniformCanSelectFragmentValues)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "boolean_uniform");
   b.shader->info.num_ubos = 1;
   nir_def *storage = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
   nir_def *x = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
      nir_imm_int(&b, 4), .align_mul = 4, .range = 4);
   nir_def *condition = nir_ieq(&b, nir_b2b1(&b, storage),
                                nir_flt(&b, x, nir_imm_float(&b, .5)));
   nir_def *value = nir_bcsel(&b, condition,
                             nir_imm_float(&b, .25), nir_imm_float(&b, .75));
   nir_store_output(&b, nir_vec4(&b, value, value, value, nir_imm_float(&b, 1)),
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason)) << (reason ?: "");
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, SixteenIndependentTextureAndSamplerBindings)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_texture_bindings");
   nir_def *sum = nir_imm_vec4(&b, 0, 0, 0, 0);
   for (unsigned i = 0; i < 16; ++i) {
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
      tex->op = nir_texop_tex;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->texture_index = 2*i+1;
      tex->sampler_index = 2*(15-i);
      tex->src[0].src_type = nir_tex_src_coord;
      tex->src[0].src = nir_src_for_ssa(nir_vec2(
         &b, nir_imm_float(&b, .375), nir_imm_float(&b, .875)));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      sum = nir_fadd(&b, sum, &tex->def);
   }
   nir_store_output(&b, nir_fmul_imm(&b, sum, 1.0/16.0), nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason)) << (reason ?: "");
   EXPECT_EQ(compiled.info.apple9_texture_mask, 0xaaaaaaaau);
   EXPECT_EQ(compiled.info.apple9_sampler_mask, 0x55555555u);
   const auto *code = static_cast<const uint8_t *>(compiled.binary);
   unsigned seen = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      if ((code[i] & 7) != 5 || code[i+2] != 0x0c || code[i+3] != 0xb8 ||
          (code[i+4] & 0xf8) != 0xb0 || code[i+12] != 1)
         continue;
      unsigned texture = ((code[i+1] >> 3) & 7)*2 + (code[i+8] >> 7);
      unsigned sampler = (code[i+4] & 7)*2 + (code[i+9] & 1);
      EXPECT_EQ(code[i+8] & 1, 0) << "Later samples still need helper lanes";
      EXPECT_EQ(sampler, 15-texture);
      EXPECT_EQ(seen & (1u << texture), 0u);
      seen |= 1u << texture;
   }
   EXPECT_EQ(seen, 0xffffu);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, RejectsDescriptorCapacityBeforeEmission)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_too_many_textures");
   nir_def *sum = nir_imm_vec4(&b, 0, 0, 0, 0);
   for (unsigned i = 0; i < 17; ++i) {
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
      tex->op = nir_texop_tex;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->texture_index = i;
      tex->sampler_index = 0;
      tex->src[0].src_type = nir_tex_src_coord;
      tex->src[0].src = nir_src_for_ssa(nir_vec2(
         &b, nir_imm_float(&b, .375), nir_imm_float(&b, .875)));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      sum = nir_fadd(&b, sum, &tex->def);
   }
   nir_store_output(&b, sum, nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   EXPECT_FALSE(agx_compile_apple9_fragment(b.shader, &compiled, &reason));
   ASSERT_NE(reason, nullptr);
   EXPECT_NE(strstr(reason, "sixteen textures"), nullptr);
   EXPECT_EQ(compiled.binary, nullptr);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, FragmentInterpolationModesUseDistinctCoefficientContracts)
{
   for (unsigned mode : {INTERP_MODE_NONE, INTERP_MODE_SMOOTH,
                         INTERP_MODE_NOPERSPECTIVE, INTERP_MODE_FLAT}) {
      for (unsigned vectors : {8u, 12u, 16u, 17u, 24u}) {
         for (bool integer : {false, true}) {
            if (integer && mode != INTERP_MODE_FLAT)
               continue;
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_FRAGMENT, &agx_nir_options, "interpolation_modes");
            nir_def *input;
            if (mode == INTERP_MODE_FLAT) {
               input = nir_load_input(
                  &b, 1, 32, nir_imm_int(&b, 0), .component = 3,
                  .dest_type = integer ? nir_type_uint32 : nir_type_float32,
                  .io_semantics = {
                     .location = (uint8_t)(VARYING_SLOT_VAR0 + vectors - 1),
                     .num_slots = 1});
            } else {
               nir_def *bary =
                  nir_load_barycentric_pixel(&b, 32, .interp_mode = mode);
               input = nir_load_interpolated_input(
                  &b, 1, 32, bary, nir_imm_int(&b, 0), .component = 3,
                  .dest_type = nir_type_float32,
                  .io_semantics = {
                     .location = (uint8_t)(VARYING_SLOT_VAR0 + vectors - 1),
                     .num_slots = 1});
            }
            if (integer)
               input = nir_u2f32(&b, nir_iand_imm(&b, input, 255));
            nir_store_output(
               &b, nir_vec4(&b, input, input, input, nir_imm_float(&b, 1)),
               nir_imm_int(&b, 0), .write_mask = 15,
               .src_type = nir_type_float32,
               .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
            b.shader->info.io_lowered = true;
            agx_apple9_varying_layout producer = {};
            for (unsigned i = 0; i < vectors; ++i)
               producer.mask[VARYING_SLOT_VAR0 + i] = 15;
            producer.count = 4 * vectors;
            agx_shader_part out = {};
            const char *reason = nullptr;
            ASSERT_TRUE(agx_compile_apple9_fragment_inputs(b.shader, &producer,
                                                           &out, &reason))
               << reason;
            unsigned index = 4 * vectors - 1;
            EXPECT_EQ(out.info.apple9_linear_mask.lo,
                      mode == INTERP_MODE_NOPERSPECTIVE && index < 64
                         ? BITFIELD64_BIT(index) : 0u);
            EXPECT_EQ(out.info.apple9_linear_mask.hi,
                      mode == INTERP_MODE_NOPERSPECTIVE && index >= 64
                         ? BITFIELD64_BIT(index - 64) : 0u);
            EXPECT_EQ(out.info.apple9_flat_mask.lo,
                      mode == INTERP_MODE_FLAT && index < 64
                         ? BITFIELD64_BIT(index) : 0u);
            EXPECT_EQ(out.info.apple9_flat_mask.hi,
                      mode == INTERP_MODE_FLAT && index >= 64
                         ? BITFIELD64_BIT(index - 64) : 0u);
            EXPECT_EQ(out.info.varyings.fs.nr_cf, 4 * vectors + 1);
            free(out.binary);
            ralloc_free(b.shader);
         }
      }
   }
}

TEST(Apple9Compiler, CentroidAndCenterInterpolationShareOrdinaryAllocatedInputs)
{
   for (unsigned mode : {INTERP_MODE_SMOOTH, INTERP_MODE_NOPERSPECTIVE}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "centroid_inputs");
      nir_def *centroid = nir_load_barycentric_centroid(&b, 32, .interp_mode = mode);
      nir_def *center = nir_load_barycentric_pixel(&b, 32, .interp_mode = mode);
      nir_def *a = nir_load_interpolated_input(
         &b, 4, 32, centroid, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_def *c = nir_load_interpolated_input(
         &b, 4, 32, center, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR1, .num_slots = 1});
      nir_store_output(&b, nir_fadd(&b, a, c), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_apple9_varying_layout producer = {};
      producer.mask[VARYING_SLOT_VAR0] = producer.mask[VARYING_SLOT_VAR1] = 15;
      producer.count = 8;
      agx_apple9_blend blend = {};
      blend.rgb_src = blend.alpha_src = PIPE_BLENDFACTOR_ONE;
      blend.rgb_dst = blend.alpha_dst = PIPE_BLENDFACTOR_ZERO;
      blend.samples = 4;
      blend.colormask = 15;
      agx_shader_part out = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_mrt(
         b.shader, &producer, &blend, 1, &out, &reason)) << reason;
      free(out.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, IntegerVertexExportsPreserveBitPatterns)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "integer_exports");
   nir_store_output(
      &b, nir_imm_vec4(&b, 0, 0, .5, 1), nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_store_output(
      &b, nir_ixor(&b, nir_load_vertex_id(&b), nir_imm_int(&b, 0xffabcdefu)),
      nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part out = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_vertex(b.shader, &out, &reason)) << reason;
   EXPECT_EQ(out.info.apple9_varyings.count, 1u);
   free(out.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Allocator, FlatCoefficientTuplesUseScoreboardAndKeepUnusedLanesLive)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t a = agx_apple9_vir_emit_iter_flat(&program, 1);
   uint32_t b = agx_apple9_vir_emit_iter_flat(&program, 32);
   uint32_t aa[] = {a, a}, bb[] = {b, b};
   uint32_t x = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IOR,
                                    AGX_APPLE9_ENC_LOGIC_EXTENDED, aa, 2, 0);
   uint32_t y = agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_IOR,
                                    AGX_APPLE9_ENC_LOGIC_EXTENDED, bb, 2, 0);
   uint32_t xy[] = {x, y};
   program.output = agx_apple9_vir_emit(
      &program, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, xy, 2, 0);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << reason;
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
   ASSERT_TRUE(agx_apple9_validate_vir_allocation(&program, &reason)) << reason;
   EXPECT_NE(program.instructions[0]->producer_scoreboard_slot,
             program.instructions[1]->producer_scoreboard_slot);
   EXPECT_EQ(program.instructions[0]->producer_scoreboard_slot,
             program.instructions[2]->scoreboard_slot);
   EXPECT_EQ(program.instructions[1]->producer_scoreboard_slot,
             program.instructions[3]->scoreboard_slot);
   for (unsigned i = 0; i < 3; ++i)
      for (unsigned j = 0; j < 3; ++j)
         EXPECT_NE(program.phys[a - 2 + i], program.phys[b - 2 + j]);
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(program.instructions[1],
                                               program.phys, &packed, &reason))
      << reason;
   EXPECT_EQ(packed.bytes[4], 16);
   EXPECT_EQ(packed.bytes[5],
             program.instructions[1]->producer_scoreboard_slot - 1);
   agx_apple9_vir_finish(&program);
}

TEST(Apple9Packer, IntegerPublicationBoundsAllRegisterOperands)
{
   agx_apple9_vir_instr instruction = {};
   instruction.op = AGX_APPLE9_VIR_IXOR;
   instruction.encoding = AGX_APPLE9_ENC_LOGIC_EXPORT;
   instruction.dest = 0;
   instruction.src[0] = 1;
   instruction.src[1] = 2;
   instruction.nr_srcs = 2;
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   for (unsigned operand = 0; operand < 3; ++operand) {
      uint8_t phys[] = {16, 20, 21};
      phys[operand] = operand == 0 ? 127 : 95;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&instruction, phys, &packed, &reason));
      EXPECT_EQ(packed.bytes[4] & 0x40, 0x40);
      phys[operand] = operand == 0 ? 128 : 96;
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&instruction, phys, &packed, &reason));
   }
}

TEST(Apple9Allocator, PublicationsDoNotOccupyGprs)
{
   for (unsigned count : {1u, 36u, 64u, 65u, 96u, 128u, 129u}) {
      agx_apple9_vir_program program;
      agx_apple9_vir_init(&program);
      uint32_t publications[129];
      uint32_t ordinary[129];
      for (unsigned i = 0; i < count; ++i) {
         ordinary[i] = agx_apple9_vir_emit(
            &program, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32,
            nullptr, 0, 0x3f800000 + i);
         uint32_t sources[] = {ordinary[i], ordinary[i]};
         publications[i] = agx_apple9_vir_emit(
            &program, i % 2 ? AGX_APPLE9_VIR_IXOR : AGX_APPLE9_VIR_FMUL,
            i % 2 ? AGX_APPLE9_ENC_LOGIC_EXPORT : AGX_APPLE9_ENC_FLOAT2_EXPORT,
            sources, 2, 0);
      }
      /* Keep every publication pending while reusing ordinary storage. */
      for (unsigned i = 0; i < count; ++i)
         ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
            &program, AGX_APPLE9_VIR_VARY_STORE, AGX_APPLE9_ENC_VARY_STORE,
            &publications[i], 1, 0));
      const char *reason = nullptr;
      if (count > AGX_APPLE9_PUBLICATION_COUNT) {
         EXPECT_FALSE(agx_apple9_allocate_vir(&program, &reason));
         ASSERT_NE(reason, nullptr);
         EXPECT_NE(strstr(reason, "publication slots"), nullptr);
      } else {
         ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason)) << reason;
         EXPECT_EQ(program.publication_count, count);
         EXPECT_EQ(program.peak_live_gprs, 1u);
         EXPECT_EQ(program.max_phys_gpr, program.phys[ordinary[0]]);
         bool seen[AGX_APPLE9_PUBLICATION_COUNT] = {};
         for (unsigned i = 0; i < count; ++i) {
            EXPECT_TRUE(program.publication[publications[i]]);
            EXPECT_FALSE(program.publication[ordinary[i]]);
            EXPECT_EQ(program.phys[ordinary[i]], program.phys[ordinary[0]]);
            ASSERT_LT(program.phys[publications[i]], AGX_APPLE9_PUBLICATION_COUNT);
            unsigned index = program.phys[publications[i]];
            EXPECT_FALSE(seen[index]);
            seen[index] = true;
         }
         if (count == AGX_APPLE9_PUBLICATION_COUNT) {
            EXPECT_TRUE(seen[program.phys[ordinary[0]]]);
            program.phys[publications[1]] = program.phys[publications[0]];
            EXPECT_FALSE(agx_apple9_validate_vir_allocation(&program, &reason));
            EXPECT_NE(strstr(reason, "overlap"), nullptr);
         }
      }
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Allocator, RejectsCrossNamespaceOperands)
{
   for (bool publication_as_gpr : {false, true}) {
      agx_apple9_vir_program program;
      agx_apple9_vir_init(&program);
      uint32_t value = agx_apple9_vir_emit(
         &program, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32,
         nullptr, 0, 0x3f800000);
      uint32_t sources[] = {value, value};
      uint32_t publication = agx_apple9_vir_emit(
         &program, AGX_APPLE9_VIR_FMUL, AGX_APPLE9_ENC_FLOAT2_EXPORT,
         sources, 2, 0);
      uint32_t stored = publication_as_gpr ? publication : value;
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &program, AGX_APPLE9_VIR_VARY_STORE, AGX_APPLE9_ENC_VARY_STORE,
         &stored, 1, 0));
      if (publication_as_gpr) {
         uint32_t invalid[] = {publication, value};
         program.output = agx_apple9_vir_emit(
            &program, AGX_APPLE9_VIR_FMUL, AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED,
            invalid, 2, 0);
      }
      const char *reason = nullptr;
      EXPECT_FALSE(agx_apple9_allocate_vir(&program, &reason));
      ASSERT_NE(reason, nullptr);
      EXPECT_NE(strstr(reason, "namespace"), nullptr);
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Compiler, StandardBlendEquationsAndFactors)
{
   const unsigned factors[] = {
      PIPE_BLENDFACTOR_ZERO, PIPE_BLENDFACTOR_ONE,
      PIPE_BLENDFACTOR_SRC_COLOR, PIPE_BLENDFACTOR_INV_SRC_COLOR,
      PIPE_BLENDFACTOR_DST_COLOR, PIPE_BLENDFACTOR_INV_DST_COLOR,
      PIPE_BLENDFACTOR_SRC_ALPHA, PIPE_BLENDFACTOR_INV_SRC_ALPHA,
      PIPE_BLENDFACTOR_DST_ALPHA, PIPE_BLENDFACTOR_INV_DST_ALPHA,
      PIPE_BLENDFACTOR_CONST_COLOR, PIPE_BLENDFACTOR_INV_CONST_COLOR,
      PIPE_BLENDFACTOR_CONST_ALPHA, PIPE_BLENDFACTOR_INV_CONST_ALPHA,
      PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE,
   };
   for (unsigned factor : factors) {
      for (unsigned equation = PIPE_BLEND_ADD; equation <= PIPE_BLEND_MAX; ++equation) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "standard_blend");
         nir_store_output(&b, nir_imm_vec4(&b, .2, .4, .6, .8), nir_imm_int(&b, 0),
            .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
         b.shader->info.io_lowered = true;
         agx_apple9_blend blend = {};
         blend.rgb_src = blend.alpha_src = factor;
         blend.rgb_dst = blend.alpha_dst = PIPE_BLENDFACTOR_ONE;
         blend.rgb_func = blend.alpha_func = equation;
         blend.colormask = 15;
         agx_apple9_varying_layout varyings = {};
         agx_shader_part out = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_compile_apple9_fragment_blend(
            b.shader, &varyings, &blend, &out, &reason))
            << "factor=" << factor << " equation=" << equation << " " << reason;
         free(out.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, AdvancedBlendUsesOrdinaryColorLowering)
{
   for (unsigned mode = PIPE_ADVANCED_BLEND_MULTIPLY;
        mode <= PIPE_ADVANCED_BLEND_HSL_LUMINOSITY; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "advanced_blend");
      nir_store_output(&b, nir_imm_vec4(&b, .2, .4, .6, .8), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_apple9_blend blend = {};
      blend.advanced_mode = mode;
      blend.src_premultiplied = blend.dst_premultiplied = true;
      blend.colormask = 15;
      agx_apple9_varying_layout varyings = {};
      agx_shader_part out = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_blend(
         b.shader, &varyings, &blend, &out, &reason)) << mode << ": " << reason;
      free(out.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Allocator, TexturePublicationsHaveIndependentStorageAndPendingLifetime)
{
   for (unsigned placement : {0u, 1u, 2u}) {
      bool overlap = placement != 0;
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      uint32_t inputs[8];
      for (unsigned i = 0; i < 8; ++i) {
         inputs[i] = agx_apple9_vir_input(&p, i);
         ASSERT_TRUE(agx_apple9_vir_add_live_out(&p, inputs[i]));
      }
      uint32_t coords[] = {inputs[0], inputs[1]};
      ASSERT_NE(agx_apple9_vir_emit_texture_sample(&p, coords, inputs[2], 0, 0),
                AGX_APPLE9_VREG_INVALID);
      p.output = agx_apple9_vir_emit_texture_sample(&p, coords, inputs[2], 1, 0);
      ASSERT_NE(p.output, AGX_APPLE9_VREG_INVALID);
      /* Move the second parameter definition ahead of the first sample.
       * Both publications must coexist until the first result handoff. */
      if (placement == 1) {
         agx_apple9_vir_move_before(&p, p.instructions[2], p.instructions[1]);
      }
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&p, &reason)) << (reason ?: "");
      if (placement == 2) {
         /* More subtly, issuing the first sample does not end its parameter
          * lifetime. Put another publication before its result handoff. */
         unsigned second_params = 0;
         for (unsigned i = 1; i < p.instruction_count; ++i) {
            if (p.instructions[i]->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE) {
               second_params = i;
               break;
            }
         }
         ASSERT_GT(second_params, 2u);
         agx_apple9_vir_move_before(&p, p.instructions[second_params], p.instructions[2]);
      }
      ASSERT_TRUE(agx_apple9_allocate_vir(&p, &reason)) << (reason ?: "");
      ASSERT_TRUE(agx_apple9_validate_vir_allocation(&p, &reason)) << (reason ?: "");
      EXPECT_EQ(p.publication_count, overlap ? 4u : 2u);
      uint32_t pubs[2];
      unsigned count = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         auto &ins = *p.instructions[i];
         if (ins.op != AGX_APPLE9_VIR_PUBLICATION_TUPLE)
            continue;
         ASSERT_LT(count, 2u);
         pubs[count++] = ins.dest;
         EXPECT_TRUE(p.publication[ins.dest]);
         EXPECT_TRUE(p.publication[ins.dest + 1]);
         EXPECT_EQ(p.phys[ins.dest] & 1, 0);
      }
      ASSERT_EQ(count, 2u);
      for (unsigned i = 0; i < 8; ++i) {
         EXPECT_FALSE(p.publication[inputs[i]]);
         EXPECT_EQ(p.phys[inputs[i]], i);
      }
      if (overlap) {
         p.phys[pubs[1]] = p.phys[pubs[0]];
         p.phys[pubs[1] + 1] = p.phys[pubs[0] + 1];
         EXPECT_FALSE(agx_apple9_validate_vir_allocation(&p, &reason));
         ASSERT_NE(reason, nullptr);
         EXPECT_NE(strstr(reason, "overlap"), nullptr);
      }
      agx_apple9_vir_finish(&p);
   }
}

TEST(Apple9Compiler, TexelFetchUsesDynamicLodWithoutSamplerBinding)
{
   for (unsigned kind = 0; kind < 3; ++kind)
   for (bool reverse_sources : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_texel_fetch");
      b.shader->info.num_ubos = 1;
      nir_def *lod = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
      nir_def *coord = kind ? nir_vec3(&b, nir_imm_int(&b, 3), nir_imm_int(&b, 5), nir_imm_int(&b, 1))
                            : nir_vec2(&b, nir_imm_int(&b, 3), nir_imm_int(&b, 5));
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = nir_texop_txf;
      tex->sampler_dim = kind == 2 ? GLSL_SAMPLER_DIM_3D : GLSL_SAMPLER_DIM_2D;
      tex->is_array = kind == 1;
      tex->dest_type = nir_type_float32;
      tex->coord_components = kind ? 3 : 2;
      tex->texture_index = 7;
      tex->sampler_index = 31; /* Has no meaning for an integer fetch. */
      unsigned c = reverse_sources ? 1 : 0;
      tex->src[c].src_type = nir_tex_src_coord;
      tex->src[c].src = nir_src_for_ssa(coord);
      tex->src[1-c].src_type = nir_tex_src_lod;
      tex->src[1-c].src = nir_src_for_ssa(lod);
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_output(&b, &tex->def, nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
         << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_texture_mask, 1u << 7);
      EXPECT_EQ(compiled.info.apple9_sampler_mask, 0u);
      EXPECT_TRUE(compiled.info.apple9_uses_texel_fetch);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, SixteenApiSamplersLeaveRoomForTexelFetch)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "all_samplers_and_fetch");
   nir_def *sum = nir_imm_vec4(&b, 0, 0, 0, 0);
   for (unsigned i = 0; i < 17; ++i) {
      bool fetch = i == 16;
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = fetch ? nir_texop_txf : nir_texop_txl;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->texture_index = i % 16;
      tex->sampler_index = i % 16;
      tex->src[0].src_type = nir_tex_src_coord;
      tex->src[0].src = nir_src_for_ssa(fetch ? nir_imm_ivec2(&b, 0, 0)
                                            : nir_imm_vec2(&b, .5, .5));
      tex->src[1].src_type = nir_tex_src_lod;
      tex->src[1].src = nir_src_for_ssa(nir_imm_int(&b, 0));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      sum = nir_fadd(&b, sum, &tex->def);
   }
   nir_store_output(&b, nir_fmul_imm(&b, sum, 1.0 / 17.0), nir_imm_int(&b, 0),
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason)) << reason;
   EXPECT_EQ(compiled.info.apple9_texture_mask, 0xffffu);
   EXPECT_EQ(compiled.info.apple9_sampler_mask, 0xffffu);
   EXPECT_TRUE(compiled.info.apple9_uses_texel_fetch);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, TextureOffsetsLowerPerMipThroughOrdinaryFetches)
{
   for (unsigned kind = 0; kind < 3; ++kind)
   for (bool shadow : {false, true}) {
      if (kind == 2 && shadow)
         continue;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "offset_filtering");
      b.shader->info.num_ubos = 1;
      nir_def *lod = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, shadow ? 4 : 3);
      tex->op = nir_texop_txl;
      tex->sampler_dim = kind == 2 ? GLSL_SAMPLER_DIM_3D : GLSL_SAMPLER_DIM_2D;
      tex->is_array = kind == 1;
      tex->is_shadow = shadow;
      tex->is_new_style_shadow = shadow;
      tex->dest_type = nir_type_float32;
      tex->coord_components = kind ? 3 : 2;
      tex->texture_index = 7;
      tex->sampler_index = 5;
      tex->src[0].src_type = nir_tex_src_coord;
      tex->src[0].src = nir_src_for_ssa(kind ? nir_imm_vec3(&b, .375, .625, .25)
                                                   : nir_imm_vec2(&b, .375, .625));
      tex->src[1].src_type = nir_tex_src_lod;
      tex->src[1].src = nir_src_for_ssa(lod);
      tex->src[2].src_type = nir_tex_src_offset;
      tex->src[2].src = nir_src_for_ssa(kind == 2 ? nir_imm_ivec3(&b, 1, -2, 3)
                                                      : nir_imm_ivec2(&b, 1, -2));
      if (shadow) {
         tex->src[3].src_type = nir_tex_src_comparator;
         tex->src[3].src = nir_src_for_ssa(nir_imm_float(&b, .4));
      }
      nir_def_init(&tex->instr, &tex->def, shadow ? 1 : 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_def *color = shadow ? nir_replicate(&b, &tex->def, 4) : &tex->def;
      nir_store_output(&b, color, nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_apple9_sampler_key key[32] = {};
      key[5].wrap[0] = PIPE_TEX_WRAP_REPEAT;
      key[5].wrap[1] = PIPE_TEX_WRAP_MIRROR_REPEAT;
      key[5].wrap[2] = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      key[5].min_filter = PIPE_TEX_FILTER_LINEAR;
      key[5].mag_filter = PIPE_TEX_FILTER_NEAREST;
      key[5].mip_filter = PIPE_TEX_MIPFILTER_LINEAR;
      key[5].compare_func = PIPE_FUNC_LEQUAL;
      key[5].min_lod = -8;
      key[5].max_lod = 8;
      const char *reason = nullptr;
      ASSERT_TRUE(agx_nir_lower_apple9_texture_offsets(b.shader, key, &reason)) << (reason ?: "");
      nir_validate_shader(b.shader, "offsets lowered to filtered fetches");
      agx_shader_part compiled = {};
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason)) << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_texture_mask, 1u << 7);
      EXPECT_EQ(compiled.info.apple9_sampler_mask, 0u);
      EXPECT_TRUE(compiled.info.apple9_uses_texel_fetch);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, FilteredLodAndBiasUseSamplerBindings)
{
   for (auto op : {nir_texop_txl, nir_texop_txb})
   for (bool reverse_sources : {false, true})
   for (unsigned sampler : {5u, 29u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_texel_fetch");
      b.shader->info.num_ubos = 1;
      nir_def *lod = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
      nir_def *coord = nir_vec2(&b, nir_imm_float(&b, .375), nir_imm_float(&b, .625));
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = op;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->texture_index = 7;
      tex->sampler_index = sampler;
      unsigned c = reverse_sources ? 1 : 0;
      tex->src[c].src_type = nir_tex_src_coord;
      tex->src[c].src = nir_src_for_ssa(coord);
      tex->src[1-c].src_type = op == nir_texop_txb ? nir_tex_src_bias : nir_tex_src_lod;
      tex->src[1-c].src = nir_src_for_ssa(lod);
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_output(&b, &tex->def, nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
         << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_texture_mask, 1u << 7);
      EXPECT_EQ(compiled.info.apple9_sampler_mask, 1u << sampler);
      EXPECT_EQ(compiled.info.disable_tri_merging, op == nir_texop_txb);
      EXPECT_FALSE(compiled.info.apple9_uses_texel_fetch);
      ASSERT_EQ(compiled.info.apple9_resource_count, 2u);
      EXPECT_TRUE(compiled.info.apple9_resource_binding[0] == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING ||
                  compiled.info.apple9_resource_binding[1] == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

static nir_tex_instr *
apple9_lod_test_texture(nir_builder *b, nir_texop op, nir_def *lod)
{
   nir_def *coord = nir_vec2(b, nir_imm_float(b, .375), nir_imm_float(b, .625));
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = op;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[0] = (nir_tex_src){.src = nir_src_for_ssa(coord),
                              .src_type = nir_tex_src_coord};
   tex->src[1] = (nir_tex_src){
      .src = nir_src_for_ssa(lod),
      .src_type = op == nir_texop_txb ? nir_tex_src_bias : nir_tex_src_lod};
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   nir_store_output(b, &tex->def, nir_imm_int(b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b->shader->info.io_lowered = true;
   return tex;
}

TEST(Apple9Compiler, TextureLodPackingPreservesSignedQ6Boundaries)
{
   const float values[] = {
      -INFINITY, -100.0f, -32.0f, std::nextafter(-32.0f, 0.0f),
      -1.015625f, -1.0f, std::nextafter(0.0f, -1.0f), -0.0f, 0.0f,
      std::nextafter(0.015625f, 0.0f), 0.015625f,
      std::nextafter(0.015625f, 1.0f), 1.5f, 2047.0f / 64.0f,
      std::nextafter(2047.0f / 64.0f, INFINITY), 32.0f, INFINITY, NAN,
   };
   for (auto op : {nir_texop_txl, nir_texop_txb}) {
      for (float value : values) {
         SCOPED_TRACE(value);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_lod_rounding");
         nir_tex_instr *tex = apple9_lod_test_texture(&b, op, nir_imm_float(&b, value));
         ASSERT_TRUE(agx_nir_lower_apple9_texture_lod(b.shader));
         EXPECT_FALSE(agx_nir_lower_apple9_texture_lod(b.shader));
         nir_opt_constant_folding(b.shader);
         nir_validate_shader(b.shader, "packed LOD is a backend operand");
         int source = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
         ASSERT_GE(source, 0);
         ASSERT_TRUE(nir_src_is_const(tex->src[source].src));
         float clamped = std::fmin(std::fmax(value, -32.0f), 2047.0f / 64.0f);
         uint32_t expected = (uint32_t(int32_t(std::floor(clamped * 64.0f))) & 0xfff) << 16;
         EXPECT_EQ(nir_src_as_uint(tex->src[source].src), expected);
         EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_lod), -1);
         EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_bias), -1);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, IntegerFetchLodIsPackedWithoutFloatConversion)
{
   for (uint32_t level : {0u, 1u, 7u, 31u, UINT32_MAX}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_fetch_lod");
      nir_tex_instr *tex = apple9_lod_test_texture(
         &b, nir_texop_txf, nir_imm_int(&b, level));
      ASSERT_TRUE(agx_nir_lower_apple9_texture_lod(b.shader));
      nir_opt_constant_folding(b.shader);
      int source = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
      ASSERT_GE(source, 0);
      ASSERT_TRUE(nir_src_is_const(tex->src[source].src));
      EXPECT_EQ(nir_src_as_uint(tex->src[source].src), level << 22);
      ralloc_free(b.shader);
   }
}

static unsigned
apple9_binary_floor_count(const agx_shader_part *part)
{
   const uint8_t *code = (const uint8_t *)part->binary;
   unsigned count = 0;
   for (unsigned at = 0; at + 10 <= part->info.binary_size; at += 2) {
      const uint8_t *p = code + at;
      if (p[0] == 0x2f && p[1] == 0 && p[2] == 0x54 &&
          (p[6] & 0xd0) == 0x90 && (p[7] & 0x7f) == 0x40 && p[8] == 2)
         ++count;
   }
   return count;
}

TEST(Apple9Compiler, UniformLodPackingMovesToPreambleButVaryingLodStays)
{
   for (auto op : {nir_texop_txl, nir_texop_txb}) {
      for (bool uniform : {false, true}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_lod_preamble");
         b.shader->info.num_ubos = 1;
         nir_def *lod = uniform
            ? nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                           .align_mul = 4, .range = 4)
            : nir_channel(&b, nir_load_frag_coord(&b), 0);
         apple9_lod_test_texture(&b, op, lod);
         agx_shader_part compiled = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
            << (reason ?: "");
         agx_shader_part main = {}, preamble = {};
         main.binary = compiled.binary;
         main.info.binary_size = compiled.info.main_size;
         preamble.binary = (uint8_t *)compiled.binary + compiled.info.apple9_preamble_offset;
         preamble.info.binary_size = compiled.info.apple9_preamble_size;
         EXPECT_EQ(apple9_binary_floor_count(&main), uniform ? 0u : 1u);
         EXPECT_EQ(apple9_binary_floor_count(&preamble), uniform ? 1u : 0u);
         free(compiled.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, ExplicitGradientsUseEightWordPublications)
{
   for (bool reverse_sources : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "apple9_texel_fetch");
      b.shader->info.num_ubos = 1;
      nir_def *lod = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
      nir_def *coord = nir_vec2(&b, nir_imm_float(&b, .375), nir_imm_float(&b, .625));
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op = nir_texop_txd;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->texture_index = 7;
      tex->sampler_index = 5;
      unsigned c = reverse_sources ? 1 : 0;
      tex->src[c].src_type = nir_tex_src_coord;
      tex->src[c].src = nir_src_for_ssa(coord);
      tex->src[1-c].src_type = nir_tex_src_ddx;
      tex->src[1-c].src = nir_src_for_ssa(nir_vec2(&b, lod, nir_imm_float(&b, 0)));
      tex->src[2].src_type = nir_tex_src_ddy;
      tex->src[2].src = nir_src_for_ssa(nir_vec2(&b, nir_imm_float(&b, 0), lod));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_output(&b, &tex->def, nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
         << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_texture_mask, 1u << 7);
      EXPECT_EQ(compiled.info.apple9_sampler_mask, 1u << 5);
      EXPECT_FALSE(compiled.info.apple9_uses_texel_fetch);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Encoding, LodPublicationsDoNotIntroduceUnrequestedWaits)
{
   uint8_t phys[] = {4, 16, 20, 21};
   agx_apple9_vir_instr ins = {};
   ins.op = AGX_APPLE9_VIR_PUBLICATION_TUPLE;
   ins.encoding = AGX_APPLE9_ENC_TEXTURE_LOD_PARAMS;
   ins.dest = 0;
   ins.dest_components = 4;
   ins.nr_srcs = 3;
   ins.src[0] = 1;
   ins.src[1] = 2;
   ins.src[2] = 3;
   for (unsigned live = 0; live < 8; ++live) {
      ins.live_after_mask = live;
      agx_apple9_packed_instruction packed;
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
      ASSERT_EQ(packed.length, 30u);
      for (unsigned offset : {0u, 10u, 20u}) {
         EXPECT_EQ(packed.bytes[offset + 5] & 0xe0, 0);
         EXPECT_EQ(packed.bytes[offset + 7] & 0xe0, 0);
         EXPECT_NE(packed.bytes[offset + 2] & 0x20, 0);
      }
   }
}

TEST(Apple9Encoding, FloatToIntegerPreservesLiveSources)
{
   uint8_t phys[] = {16, 18};
   for (bool is_signed : {false, true}) {
      for (bool keep : {false, true}) {
         agx_apple9_vir_instr ins = {};
         ins.op = is_signed ? AGX_APPLE9_VIR_F2I32 : AGX_APPLE9_VIR_F2U32;
         ins.encoding = is_signed ? AGX_APPLE9_ENC_FLOAT_TO_SINT : AGX_APPLE9_ENC_FLOAT_TO_UINT;
         ins.dest = 0;
         ins.nr_srcs = 1;
         ins.src[0] = 1;
         ins.live_after_mask = keep;
         agx_apple9_packed_instruction packed;
         const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
         EXPECT_EQ(packed.bytes[6], keep ? 0x96 : 0xb4);
         EXPECT_EQ(packed.bytes[8], is_signed ? 0x03 : 0x02);
      }
   }
}

TEST(Apple9Compiler, RepeatedDemotionReusesCoveragePublication)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "repeated_demotion");
   nir_def *bary = nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *x = nir_load_interpolated_input(&b, 1, 32, bary, nir_imm_int(&b, 0),
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   for (unsigned i = 0; i < 20; ++i)
      nir_demote_if(&b, nir_flt(&b, x, nir_imm_float(&b, i / 40.0)));
   nir_store_output(&b, nir_vec4(&b, x, x, x, nir_imm_float(&b, 1)),
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   /* The compiler gathers discard use from the intrinsics. */
   agx_shader_part part = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment_inputs(b.shader, nullptr, &part, &reason)) << reason;
   const uint8_t *code = static_cast<const uint8_t *>(part.binary);
   unsigned coverage = 0, acquires = 0, releases = 0;
   const uint8_t acquire[] = {0x87, 2, 0x54, 1, 0, 0};
   const uint8_t release[] = {0x07, 2, 0x54, 1, 0, 0};
   for (unsigned i = 0; i + 6 <= part.info.binary_size; ++i) {
      acquires += memcmp(code + i, acquire, 6) == 0;
      releases += memcmp(code + i, release, 6) == 0;
      if (code[i] == 0x57 && code[i+1] == 0x14 && code[i+2] == 0x54) {
         EXPECT_EQ(code[i+3], 0u);
         ++coverage;
      }
   }
   /* The last demotion also finalizes tests; color stores do not repeat it. */
   EXPECT_EQ(coverage, 20u);
   EXPECT_EQ(acquires, 1u);
   EXPECT_EQ(releases, 1u);
   free(part.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, DepthWritesInitializeCoverageWithoutDiscard)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "fragment_depth_coverage");
   b.shader->info.num_ubos = 1;
   nir_def *z = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
   nir_store_output(&b, z, nir_imm_int(&b, 0),
      .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DEPTH, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
      << (reason ?: "");
   EXPECT_EQ(compiled.info.depth_layout, FRAG_DEPTH_LAYOUT_ANY);
   EXPECT_TRUE(compiled.info.apple9_uses_discard);
   const uint8_t *code = (const uint8_t *)compiled.binary;
   unsigned coverage_reads = 0;
   for (unsigned i = 0; i + 4 <= compiled.info.binary_size; ++i)
      coverage_reads += (code[i] & 0xf) == 0xc && code[i+1] == 0xc2 &&
                        code[i+2] == 0x10 && code[i+3] == 0x06;
   EXPECT_EQ(coverage_reads, 1u);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, DepthStencilOnlyFragmentHasNoColorStore)
{
   for (bool discard : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "depth_stencil_only");
      if (discard) {
         b.shader->info.num_ubos = 1;
         nir_def *x = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
            nir_imm_int(&b, 0), .align_mul = 4, .range = 4);
         nir_demote_if(&b, nir_flt(&b, x, nir_imm_float(&b, .5)));
      }
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason))
         << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_uses_discard, discard);
      ASSERT_NE(compiled.binary, nullptr);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, ComponentStoresPreservePreviouslyWrittenColor)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "component_color");
   nir_store_output(&b, nir_imm_float(&b, .4), nir_imm_int(&b, 0),
      .write_mask = 1, .component = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   nir_store_output(&b, nir_imm_vec3(&b, .2, 1, .6), nir_imm_int(&b, 0),
      .write_mask = 5, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   nir_store_output(&b, nir_imm_float(&b, .8), nir_imm_int(&b, 0),
      .write_mask = 1, .component = 3, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason)) << reason;
   unsigned stores = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
            ASSERT_TRUE(nir_src_is_const(intr->src[0]));
            EXPECT_EQ(nir_src_as_uint(intr->src[0]), 0xcc996633u);
            ++stores;
         }
      }
   }
   EXPECT_EQ(stores, 1u);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Machine, DerivativeAxisIsIndependentOfSourceLifetime)
{
   for (unsigned axis = 0; axis < 2; ++axis) {
      for (unsigned keep = 0; keep < 2; ++keep) {
         agx_apple9_vir_instr ins = {};
         ins.op = AGX_APPLE9_VIR_DERIVATIVE;
         ins.encoding = AGX_APPLE9_ENC_DERIVATIVE;
         ins.dest = 0;
         ins.src[0] = 1;
         ins.nr_srcs = 1;
         ins.immediate = axis;
         ins.live_after_mask = keep;
         uint8_t phys[] = {9, 23};
         agx_apple9_packed_instruction packed = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason)) << reason;
         EXPECT_EQ(packed.length, 10u);
         EXPECT_EQ(packed.bytes[1] & 7, axis ? 7 : 5);
         EXPECT_EQ(packed.bytes[3], 18);
         EXPECT_EQ(packed.bytes[5], 92);
         EXPECT_EQ(packed.bytes[6], keep ? 0x92 : 0x90);
         phys[1] = 95;
         ASSERT_TRUE(
            agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
         EXPECT_EQ(packed.bytes[5], 124);
         EXPECT_EQ(packed.bytes[6], keep ? 0x93 : 0x91);
         phys[1] = 96;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
      }
   }
}

TEST(Apple9Vir, PhiDefinitionStaysInSuccessorAndLivenessReachesFixedPoint)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto *entry = agx_apple9_block_create(&p);
   auto *header = agx_apple9_block_create(&p);
   auto *exit = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, entry);
   uint32_t invariant = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 17);
   uint32_t phi = agx_apple9_vir_emit_phi(&p, header);
   auto *definition = p.instructions[p.instruction_count - 1];
   agx_apple9_vir_copy initial[] = {{phi, invariant}};
   ASSERT_TRUE(agx_apple9_vir_emit_phi_edge(&p, initial, 1));
   agx_apple9_block_begin(&p, header);
   uint32_t sources[] = {phi, invariant};
   uint32_t sum = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
      AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_branch(&p, AGX_APPLE9_VIR_JMP_EXEC_NONE,
      AGX_APPLE9_ENC_JMP_EXEC_NONE, exit));
   /* Forces CFG construction to split the header at its early exit. */
   agx_apple9_vir_copy backedge[] = {{phi, sum}};
   ASSERT_TRUE(agx_apple9_vir_emit_phi_edge(&p, backedge, 1));
   ASSERT_TRUE(agx_apple9_vir_emit_branch(&p, AGX_APPLE9_VIR_JMP_EXEC_ANY,
      AGX_APPLE9_ENC_JMP_EXEC_ANY, header));
   agx_apple9_block_begin(&p, exit);
   p.output = phi;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_vir(&p, &reason)) << (reason ?: "");
   EXPECT_EQ(agx_apple9_instr_block(definition), header);
   EXPECT_EQ(p.instructions[header->start_index], definition);
   ASSERT_NE(header->next, exit);
   auto *latch = header->next;
   EXPECT_EQ(header->successors[0], latch);
   EXPECT_EQ(header->successors[1], exit);
   EXPECT_EQ(latch->successors[1], header);
   EXPECT_EQ(header->predecessor_count, 2u);
   EXPECT_TRUE(BITSET_TEST(header->live_in, invariant));
   EXPECT_TRUE(BITSET_TEST(latch->live_out, invariant));
   EXPECT_FALSE(BITSET_TEST(header->live_in, phi));
   EXPECT_NE(p.phys[invariant], p.phys[sum]);
   EXPECT_NE(p.phys[phi], p.phys[sum]);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9Vir, RemovingLastInstructionKeepsEmptyTargetBlockAlive)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto *entry = agx_apple9_block_create(&p);
   auto *target = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, entry);
   ASSERT_TRUE(agx_apple9_vir_emit_branch(&p, AGX_APPLE9_VIR_JMP_EXEC_NONE,
      AGX_APPLE9_ENC_JMP_EXEC_NONE, target));
   auto *branch = p.instructions[0];
   /* Growing the indexed view must not move instruction or block objects. */
   for (unsigned i = 0; i < 128; ++i)
      ASSERT_NE(agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, i), AGX_APPLE9_VREG_INVALID);
   agx_apple9_block_begin(&p, target);
   uint32_t index = agx_apple9_vir_input(&p, 2);
   uint32_t load = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_DEVICE_LOAD,
      AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(&p, load, 0,
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&p, &reason)) << (reason ?: "");
   EXPECT_EQ(p.instructions[0], branch);
   EXPECT_EQ(branch->branch_target, target);
   EXPECT_TRUE(list_is_empty(&target->instructions));
   EXPECT_EQ(target->start_index, p.instruction_count);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9Compiler, ComputedByteOffsetsDoNotRequireAnAffineShell)
{
   for (unsigned form = 0; form < 6; ++form) {
      SCOPED_TRACE(form);
      nir_builder b = apple9_compute_builder("computed_byte_offset");
      b.shader->info.num_ssbos = 2;
      nir_def *gid = apple9_global_id_x(&b);
      nir_def *offset;
      switch (form) {
      case 0: offset = nir_ineg(&b, nir_ishl_imm(&b, gid, 2)); break;
      case 1: offset = nir_inot(&b, nir_ior_imm(&b, gid, 3)); break;
      case 2: offset = nir_iand_imm(&b, nir_imul(&b, gid, gid), ~3u); break;
      case 3: offset = nir_bcsel(&b, nir_ult_imm(&b, gid, 8),
                      nir_ishl_imm(&b, gid, 2), nir_ishl_imm(&b, gid, 4)); break;
      case 4: offset = nir_ishl(&b, nir_ishl_imm(&b, gid, 2),
                               nir_iand_imm(&b, gid, 3)); break;
      default: offset = nir_iadd_imm(&b, nir_ishl_imm(&b, gid, 4), 0xfffffff0u); break;
      }
      nir_def *value = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 0), offset,
                                     .align_mul = 4);
      nir_store_ssbo(&b, value, nir_imm_int(&b, 1), offset, .align_mul = 4);
      nir_validate_shader(b.shader, "computed byte offsets");
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      agx_apple9_compute_profile profile = {};
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
         << (reason ?: "");
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Vir, SharedUseAnalysisTracksOperandAndLayoutEdits)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   uint32_t a = agx_apple9_vir_input(&p, 2), b = agx_apple9_vir_input(&p, 3);
   uint32_t src[] = {a, a};
   uint32_t x = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   auto *first = p.instructions[0];
   ASSERT_EQ(agx_apple9_definition(&p, x), first);
   ASSERT_EQ(agx_apple9_uses(&p, a)->instruction, first);
   ASSERT_EQ(agx_apple9_uses(&p, a)->next->source, 1u);
   EXPECT_EQ(agx_apple9_uses(&p, b), nullptr);
   agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   auto *second = p.instructions[1];
   agx_apple9_vir_move_before(&p, second, first);
   ASSERT_EQ(agx_apple9_uses(&p, a)->instruction, second);
   second->src[0] = second->src[1] = b;
   agx_apple9_invalidate_uses(&p);
   ASSERT_EQ(agx_apple9_uses(&p, a)->instruction, first);
   ASSERT_EQ(agx_apple9_uses(&p, b)->instruction, second);
   ASSERT_EQ(agx_apple9_definition(&p, x), first);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9Vir, LoopIndexReleaseUsesCfgRatherThanLastTextualRead)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   uint32_t index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 3);
   auto *header = agx_apple9_block_create(&p);
   auto *exit = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, header);
   uint32_t load = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_DEVICE_LOAD,
      AGX_APPLE9_ENC_DEVICE_LOAD, &index, 1, 0);
   ASSERT_TRUE(agx_apple9_vir_set_device_load_contract(&p, load, 0,
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   auto *memory = p.instructions[p.instruction_count - 1];
   uint32_t src[] = {load, load};
   p.output = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_branch(&p, AGX_APPLE9_VIR_JMP_EXEC_ANY,
      AGX_APPLE9_ENC_JMP_EXEC_ANY, header));
   agx_apple9_block_begin(&p, exit);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&p, &reason)) << (reason ?: "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&p, &reason)) << (reason ?: "");
   EXPECT_TRUE(memory->live_after_mask & 1);
   EXPECT_EQ(memory->device_load_index_kind, AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9Compiler, MemoryLegalizationHandlesUnalignedAndPartialVectors)
{
   for (unsigned alignment : {1u, 2u, 4u, 8u, 16u}) {
      SCOPED_TRACE(alignment);
      nir_builder b = apple9_compute_builder("general_memory_alignment");
      b.shader->info.num_ssbos = 2;
      nir_def *offset = nir_iadd_imm(&b,
         nir_ishl_imm(&b, apple9_global_id_x(&b), 4), alignment == 16 ? 0 : alignment);
      unsigned components = 4;
      nir_def *value = nir_load_ssbo(&b, components, 32, nir_imm_int(&b, 0), offset,
         .align_mul = alignment);
      nir_store_ssbo(&b, value, nir_imm_int(&b, 1), offset,
         .write_mask = 5, .align_mul = alignment);
      agx_shader_part compiled = {};
      agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
         << (reason ?: "");
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Vir, EditingAllocatedIrInvalidatesDependencyFinalization)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   p.output = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 3);
   auto *first = p.instructions[0];
   agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 4);
   auto *second = p.instructions[1];
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_vir(&p, &reason)) << (reason ?: "");
   EXPECT_TRUE(p.dependencies_finalized);
   agx_apple9_vir_move_before(&p, second, first);
   EXPECT_FALSE(p.dependencies_finalized);
   ASSERT_FALSE(agx_apple9_validate_vir_allocation(&p, &reason));
   ASSERT_NE(reason, nullptr);
   EXPECT_NE(strstr(reason, "stale"), nullptr);
   ASSERT_TRUE(agx_apple9_allocate_vir(&p, &reason)) << (reason ?: "");
   EXPECT_TRUE(p.dependencies_finalized);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9Compiler, PointSizeFollowsUserExports)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "point_size_export");
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, .5, 1), nir_imm_int(&b, 0),
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   for (unsigned i = 0; i < 8; ++i)
      nir_store_output(&b, nir_imm_vec4(&b, .1, .2, .3, .4), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = unsigned(VARYING_SLOT_VAR0 + i), .num_slots = 1});
   nir_store_output(&b, nir_imm_float(&b, 17), nir_imm_int(&b, 0),
      .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_PSIZ, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part out = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_vertex(b.shader, &out, &reason)) << reason;
   EXPECT_TRUE(out.info.apple9_writes_point_size);
   EXPECT_EQ(out.info.apple9_varyings.count, 32u);
   free(out.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, PointCoordinatesAndDepthHaveSeparateCoefficients)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "point_coords_depth");
   nir_def *p = nir_load_point_coord(&b);
   nir_def *z = nir_load_frag_coord_z(&b);
   nir_store_output(&b, nir_vec4(&b, nir_channel(&b, p, 0),
      nir_channel(&b, p, 1), z, nir_imm_float(&b, 1)), nir_imm_int(&b, 0),
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.io_lowered = true;
   agx_shader_part out = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &out, &reason)) << reason;
   EXPECT_TRUE(out.info.apple9_reads_point_coord);
   EXPECT_TRUE(out.info.apple9_reads_z);
   EXPECT_EQ(out.info.varyings.fs.nr_cf, 4u);
   free(out.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, DynamicIntegerDivisionUsesNormalLowering)
{
   for (bool is_signed : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "integer_division");
      nir_def *inputs = nir_load_ubo(&b, 2, 32, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .align_mul = 4, .range = 8);
      nir_def *q = is_signed
         ? nir_idiv(&b, nir_channel(&b, inputs, 0), nir_channel(&b, inputs, 1))
         : nir_udiv(&b, nir_channel(&b, inputs, 0), nir_channel(&b, inputs, 1));
      nir_def *f = is_signed ? nir_i2f32(&b, q) : nir_u2f32(&b, q);
      nir_store_output(&b, nir_vec4(&b, f, f, f, nir_imm_float(&b, 1)),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_shader_part out = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &out, &reason)) << reason;
      free(out.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Allocator, VaryingStoresWaitForPublicationCompletion)
{
   for (bool logic : {false, true}) {
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "varying_completion");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);
      for (unsigned i = 0; i < 2; ++i) {
         auto value = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
            AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x3f800000 + i);
         uint32_t sources[] = {value, value};
         auto publication = agx_apple9_vir_emit(&p,
            logic ? AGX_APPLE9_VIR_IOR : AGX_APPLE9_VIR_FMUL,
            logic ? AGX_APPLE9_ENC_LOGIC_EXPORT : AGX_APPLE9_ENC_FLOAT2_EXPORT,
            sources, 2, 0);
         ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p,
            AGX_APPLE9_VIR_VARY_STORE, AGX_APPLE9_ENC_VARY_STORE,
            &publication, 1, i));
      }
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
      unsigned producer_slot = 0, stores = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *ins = p.instructions[i];
         if (ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
             ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT) {
            EXPECT_EQ(producer_slot, 0u);
            producer_slot = ins->producer_scoreboard_slot;
            ASSERT_GE(producer_slot, 1u);
            ASSERT_LE(producer_slot, 6u);
            agx_apple9_packed_instruction packed = {};
            ASSERT_TRUE(agx_apple9_pack_vir_instruction(ins, p.phys, &packed, &reason));
            EXPECT_EQ((packed.bytes[7] >> 2) & 7, producer_slot - 1);
         } else if (ins->op == AGX_APPLE9_VIR_VARY_STORE) {
            ASSERT_NE(producer_slot, 0u);
            EXPECT_EQ(ins->scoreboard_slot, producer_slot);
            agx_apple9_packed_instruction packed = {};
            ASSERT_TRUE(agx_apple9_pack_vir_instruction(ins, p.phys, &packed, &reason));
            EXPECT_EQ((packed.bytes[1] >> 4) | ((packed.bytes[2] & 3) << 4),
                      1u << (producer_slot - 1));
            ++stores;
            producer_slot = 0;
         }
      }
      EXPECT_EQ(stores, 2u);
      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9Compiler, MultisampleOutputLoopsOnlyForDestinationDependentResults)
{
   for (unsigned samples : {2u, 4u}) {
      for (unsigned variant = 0; variant < 6; ++variant) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "sample_output");
         nir_store_output(&b, nir_load_frag_coord(&b), nir_imm_int(&b, 0),
            .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
         b.shader->info.io_lowered = true;
         agx_apple9_blend blend = {};
         blend.colormask = 15;
         blend.rgb_src = blend.alpha_src = PIPE_BLENDFACTOR_ONE;
         blend.rgb_dst = blend.alpha_dst = PIPE_BLENDFACTOR_ZERO;
         blend.format = variant == 1 ? PIPE_FORMAT_R16G16B16A16_FLOAT
                                     : PIPE_FORMAT_R8G8B8A8_UNORM;
         blend.samples = samples;
         if (variant == 1)
            blend.rgb_src = PIPE_BLENDFACTOR_SRC_ALPHA;
         else if (variant == 2)
            blend.rgb_src = PIPE_BLENDFACTOR_DST_COLOR;
         else if (variant == 3)
            blend.rgb_func = PIPE_BLEND_MIN;
         else if (variant == 4)
            blend.colormask = 7;
         else if (variant == 5) {
            blend.format = PIPE_FORMAT_R32G32B32A32_UINT;
            blend.rgb_dst = blend.alpha_dst = PIPE_BLENDFACTOR_ONE;
         }
         bool needs_destination = variant >= 2 && variant <= 4;
         agx_apple9_varying_layout varyings = {};
         agx_shader_part compiled = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_compile_apple9_fragment_mrt(b.shader, &varyings, &blend, 1,
                                                   &compiled, &reason))
            << "variant=" << variant << " samples=" << samples << ": " << reason;
         unsigned loads = 0, stores = 0, loop_bounds = 0, packs = 0;
         nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu) {
                  auto *alu = nir_instr_as_alu(instr);
                  packs += alu->op == nir_op_pack_unorm_4x8;
                  loop_bounds += alu->op == nir_op_uge &&
                     nir_src_is_const(alu->src[1].src) &&
                     nir_src_as_uint(alu->src[1].src) == samples;
               }
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               auto *intr = nir_instr_as_intrinsic(instr);
               loads += intr->intrinsic == nir_intrinsic_load_local_pixel_agx;
               if (intr->intrinsic != nir_intrinsic_store_local_pixel_agx)
                  continue;
               ++stores;
               if (!needs_destination) {
                  auto *mask = intr->src[1].ssa;
                  ASSERT_EQ(nir_def_instr_type(mask), nir_instr_type_intrinsic);
                  EXPECT_EQ(nir_def_as_intrinsic(mask)->intrinsic,
                            nir_intrinsic_load_sample_mask_in);
               }
            }
         }
         EXPECT_EQ(loads, needs_destination ? 1u : 0u) << variant;
         EXPECT_EQ(loop_bounds, needs_destination ? 1u : 0u) << variant;
         EXPECT_EQ(stores, 1u) << variant;
         EXPECT_EQ(packs, variant == 1 || variant == 5 ? 0u : 1u) << variant;
         free(compiled.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, MultisampleBlendLoopsSamplesAndPreservesFormatOffsets)
{
   for (unsigned samples : {2u, 4u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "sample_blend");
      for (unsigned rt = 0; rt < 2; ++rt)
         nir_store_output(&b, nir_imm_vec4(&b, .25, .5, .75, 1), nir_imm_int(&b, 0),
            .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = FRAG_RESULT_DATA0 + rt, .num_slots = 1});
      b.shader->info.io_lowered = true;
      agx_apple9_blend blend[2] = {};
      for (unsigned rt = 0; rt < 2; ++rt) {
         blend[rt].colormask = 15;
         blend[rt].rgb_src = blend[rt].rgb_dst = PIPE_BLENDFACTOR_ONE;
         blend[rt].alpha_src = blend[rt].alpha_dst = PIPE_BLENDFACTOR_ONE;
         blend[rt].format = rt ? PIPE_FORMAT_R16G16B16A16_FLOAT : PIPE_FORMAT_B8G8R8A8_UNORM;
         blend[rt].samples = samples;
      }
      agx_apple9_varying_layout varyings = {};
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_mrt(b.shader, &varyings, blend, 2,
                                                &compiled, &reason)) << reason;
      unsigned loads[2] = {}, stores[2] = {}, bounds = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op == nir_op_uge && nir_src_is_const(alu->src[1].src) &&
                   nir_src_as_uint(alu->src[1].src) == samples)
                  ++bounds;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            bool load = intr->intrinsic == nir_intrinsic_load_local_pixel_agx;
            if (!load && intr->intrinsic != nir_intrinsic_store_local_pixel_agx)
               continue;
            unsigned base = nir_intrinsic_base(intr);
            ASSERT_TRUE(base == 0 || base == 4);
            unsigned rt = base == 4;
            nir_def *mask = intr->src[!load].ssa;
            if (!load) {
               ASSERT_EQ(nir_def_instr_type(mask), nir_instr_type_alu);
               nir_alu_instr *alu = nir_def_as_alu(mask);
               ASSERT_EQ(alu->op, nir_op_iand);
               bool first_coverage = nir_def_instr_type(alu->src[0].src.ssa) ==
                                        nir_instr_type_intrinsic;
               nir_def *coverage = alu->src[first_coverage ? 0 : 1].src.ssa;
               ASSERT_EQ(nir_def_instr_type(coverage), nir_instr_type_intrinsic);
               EXPECT_EQ(nir_def_as_intrinsic(coverage)->intrinsic,
                         nir_intrinsic_load_sample_mask_in);
               mask = alu->src[first_coverage ? 1 : 0].src.ssa;
            }
            ASSERT_EQ(nir_def_instr_type(mask), nir_instr_type_alu);
            nir_alu_instr *shift = nir_def_as_alu(mask);
            ASSERT_EQ(shift->op, nir_op_ishl);
            ASSERT_TRUE(nir_src_is_const(shift->src[0].src));
            EXPECT_EQ(nir_src_as_uint(shift->src[0].src), 1u);
            EXPECT_EQ(nir_def_instr_type(shift->src[1].src.ssa), nir_instr_type_phi);
            if (load) ++loads[rt];
            else ++stores[rt];
         }
      }
      /* One transaction must cover every sample and render target. A release
       * between stores races the next overlapping fragment on hardware. */
      const uint8_t acquire[] = {0x87, 2, 0x54, 0x0c, 8, 0};
      const uint8_t release[] = {0x07, 2, 0x54, 0x0c, 2, 0};
      const uint8_t premature[] = {0x07, 2, 0x54, 0x0c, 0, 0};
      const auto *code = static_cast<const uint8_t *>(compiled.binary);
      unsigned acquires = 0, releases = 0;
      for (unsigned i = 0; i + 6 <= compiled.info.binary_size; ++i) {
         acquires += memcmp(code + i, acquire, 6) == 0;
         releases += memcmp(code + i, release, 6) == 0;
         EXPECT_NE(memcmp(code + i, premature, 6), 0);
      }
      EXPECT_EQ(acquires, 1u);
      EXPECT_EQ(releases, 1u);
      EXPECT_EQ(bounds, 2u);
      for (unsigned rt = 0; rt < 2; ++rt) {
         EXPECT_EQ(loads[rt], 1u);
         EXPECT_EQ(stores[rt], 1u);
      }
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Spilling, SplitOperandsAndIndependentCompletionTags)
{
   uint8_t phys[] = {82, 18};
   agx_apple9_vir_instr store = {};
   store.op = AGX_APPLE9_VIR_SPILL_STORE;
   store.encoding = AGX_APPLE9_ENC_SPILL_STORE;
   store.dest = AGX_APPLE9_VREG_INVALID;
   store.src[0] = 0; store.nr_srcs = 1; store.immediate = 71;
   store.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
   store.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_3;
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&store, phys, &packed, &reason));
   const uint8_t save_bytes[] = {0x7b,0x25,0x0f,0,0x22,0x91,0,0x14,0,0};
   ASSERT_EQ(packed.length, sizeof(save_bytes));
   EXPECT_EQ(memcmp(packed.bytes, save_bytes, sizeof(save_bytes)), 0);
   agx_apple9_vir_instr load = {};
   load.op = AGX_APPLE9_VIR_SPILL_LOAD; load.encoding = AGX_APPLE9_ENC_SPILL_LOAD;
   load.dest = 1; load.dest_components = 1; load.immediate = 71;
   load.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1;
   load.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
   const uint8_t fill_bytes[] = {0x2c,0x8e,0x41,0x01,0x02,0,0,0x80};
   ASSERT_EQ(packed.length, sizeof(fill_bytes));
   EXPECT_EQ(memcmp(packed.bytes, fill_bytes, sizeof(fill_bytes)), 0);
   store.live_after_mask = 1;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&store, phys, &packed, &reason));
}


TEST(Apple9Spilling, LoopCarriedPressureCompilesWithScratch)
{
   nir_builder b = apple9_compute_builder("spill_loop_pressure");
   b.shader->info.num_ssbos = 2;
   nir_def *gid = apple9_global_id_x(&b);
   const unsigned count = 112;
   nir_def *initial[count + 1];
   for (unsigned i = 0; i < count; ++i)
      initial[i] = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1),
         nir_iadd_imm(&b, nir_imul_imm(&b, gid, count * 4), i * 4),
         .access = ACCESS_NON_WRITEABLE, .align_mul = 4);
   initial[count] = nir_imm_int(&b, 0);
   nir_def *limit = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 1),
      nir_imm_int(&b, 128 * count * 4),
      .access = ACCESS_NON_WRITEABLE, .align_mul = 4);
   nir_loop *loop = nir_push_loop(&b);
   nir_block *entry = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_block *header = nir_loop_first_block(loop);
   nir_phi_instr *phi[count + 1];
   for (unsigned i = 0; i <= count; ++i) {
      phi[i] = nir_phi_instr_create(b.shader);
      nir_def_init(&phi[i]->instr, &phi[i]->def, 1, 32);
      nir_phi_instr_add_src(phi[i], entry, initial[i]);
   }
   nir_break_if(&b, nir_uge(&b, &phi[count]->def, limit));
   for (unsigned i = 0; i <= count; ++i) {
      nir_def *next = i == count ? nir_iadd_imm(&b, &phi[i]->def, 1) :
         i == 0 ? nir_ixor(&b, &phi[0]->def,
                            nir_iadd_imm(&b, &phi[count]->def, 0x9e3779b9)) :
                  nir_iadd(&b, &phi[i]->def, &phi[i - 1]->def);
      nir_phi_instr_add_src(phi[i], nir_cursor_current_block(b.cursor), next);
   }
   nir_pop_loop(&b, loop);
   b.cursor = nir_after_phis(header);
   for (unsigned i = 0; i <= count; ++i)
      nir_builder_instr_insert(&b, &phi[i]->instr);
   b.cursor = nir_after_cf_node(&loop->cf_node);
   nir_def *hash = nir_imm_int(&b, 0);
   for (unsigned i = 0; i < count; ++i)
      hash = nir_ixor(&b, nir_imul_imm(&b, hash, 33), &phi[i]->def);
   apple9_store_output(&b, gid, hash);
   nir_validate_shader(b.shader, "spill loop pressure");
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, nullptr, &reason))
      << (reason ? reason : "");
   EXPECT_GT(compiled.info.scratch_size, 0u);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, DirectResourceCapacityPreservesHighOwnershipBits)
{
   for (unsigned count : {9u, 15u, 16u, 18u, 19u}) {
      nir_shader *nir = apple9_ssbo_reduce_shader(count - 1, false);
      agx_shader_part compiled = {};
      agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      bool ok = agx_compile_apple9_tiny(nir, &compiled, &profile, &reason);
      if (count <= AGX_APPLE9_COMPUTE_MAX_RESOURCES) {
         ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
         EXPECT_EQ(profile.resource_binding_count, count);
         EXPECT_EQ(profile.resource_read_mask, (1u << (count - 1)) - 1);
         EXPECT_EQ(profile.resource_write_mask, 1u << (count - 1));
         EXPECT_EQ(profile.resource_binding[count - 1], 0);
         for (unsigned i = 0; i + 1 < count; ++i)
            EXPECT_EQ(profile.resource_binding[i], count - i - 1);
         free(compiled.binary);
      } else {
         EXPECT_FALSE(ok);
         ASSERT_NE(reason, nullptr);
         EXPECT_STREQ(reason, "Apple9 buffer resource capacity exceeded");
      }
      ralloc_free(nir);
   }
}

TEST(Apple9Allocator, SharedPublicationsCoverFullNamespace)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "publication_namespace");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);
   uint32_t publications[AGX_APPLE9_PUBLICATION_COUNT];
   for (unsigned i = 0; i < AGX_APPLE9_PUBLICATION_COUNT; ++i) {
      auto value = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x3f000000 + i);
      uint32_t sources[] = {value, value};
      publications[i] = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMUL,
         AGX_APPLE9_ENC_FLOAT2_EXPORT, sources, 2, 0);
   }
   for (auto value : publications)
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p,
         AGX_APPLE9_VIR_VARY_STORE, AGX_APPLE9_ENC_VARY_STORE, &value, 1, 0));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
   EXPECT_EQ(p.publication_count, AGX_APPLE9_PUBLICATION_COUNT);
   bool seen[AGX_APPLE9_PUBLICATION_COUNT] = {};
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->encoding != AGX_APPLE9_ENC_VARY_STORE)
         continue;
      ASSERT_LT(ins->src[0], p.value_count);
      ASSERT_TRUE(p.publication[ins->src[0]]);
      unsigned index = p.phys[ins->src[0]];
      ASSERT_LT(index, AGX_APPLE9_PUBLICATION_COUNT);
      EXPECT_FALSE(seen[index]);
      seen[index] = true;
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(ins, p.phys, &packed, &reason));
      EXPECT_EQ(packed.bytes[3], index << 1);
   }
   for (bool present : seen)
      EXPECT_TRUE(present);
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(AGX_APPLE9_ENC_FLOAT2_COMPACT,
      AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_GPR_COUNT, 32));
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9Encoding, ExplicitTileReadUsesCoordinatesAndImmediateSamples)
{
   agx_apple9_vir_instr load = {};
   load.op = AGX_APPLE9_VIR_TILE_LOAD;
   load.encoding = AGX_APPLE9_ENC_TILE_LOAD_COORDS;
   load.dest = 0;
   load.nr_srcs = 1;
   load.src[0] = 1;
   load.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1;
   uint8_t phys[] = {25, 8};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   for (unsigned sample = 0; sample < 4; ++sample) {
      load.tile_sample_mask = 1 << sample;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
      EXPECT_EQ(packed.length, 12u);
      EXPECT_EQ(packed.bytes[3], 50);
      EXPECT_EQ(packed.bytes[4], 8);
      EXPECT_EQ(packed.bytes[6], 1 << sample);
      EXPECT_EQ(packed.bytes[11], 0x10);
      EXPECT_EQ(packed.bytes[8], 0); /* Dead coordinates are not a dead mask. */
   }
   load.tile_sample_mask = 0;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
   load.tile_sample_mask = 16;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
   load.tile_sample_mask = 1;
   phys[1] = 95;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
   EXPECT_EQ(packed.bytes[4], 95);
   phys[1] = 96;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&load, phys, &packed, &reason));
}

TEST(Apple9Allocator, IndirectStoreRejectsBrokenAddressTuple)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   uint32_t address = agx_apple9_vir_input(&p, 8);
   agx_apple9_vir_input(&p, 9);
   uint32_t index = agx_apple9_vir_input(&p, 12);
   uint32_t value = agx_apple9_vir_input(&p, 14);
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &value, 1, 32));
   ASSERT_TRUE(agx_apple9_vir_set_device_store_address(&p, address));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&p, &reason)) << reason;
   ASSERT_TRUE(agx_apple9_allocate_vir(&p, &reason)) << reason;
   ASSERT_TRUE(agx_apple9_validate_vir_allocation(&p, &reason)) << reason;
   auto *store = p.instructions[p.instruction_count - 1];
   unsigned hi = store->src[store->memory_components + 2];
   p.phys[hi] = p.phys[store->src[store->memory_components + 1]] + 2;
   EXPECT_FALSE(agx_apple9_validate_vir_allocation(&p, &reason));
   agx_apple9_vir_finish(&p);
}

TEST(Apple9, EntryBranchAddressRange)
{
   struct agx_apple9_packed_instruction packed;
   for (int64_t displacement : {INT64_C(0xf0123456), -INT64_C(0xf0123456),
                                -(INT64_C(1) << 47), (INT64_C(1) << 47) - 2}) {
      ASSERT_TRUE(agx_apple9_pack_branch(true, displacement, &packed));
      ASSERT_EQ(packed.length, 10u);
      EXPECT_EQ(packed.bytes[0], 0x0f);
      EXPECT_EQ(packed.bytes[1], 0x00);
      EXPECT_EQ(packed.bytes[2], 0x54);
      for (unsigned i = 0; i < 6; i++)
         EXPECT_EQ(packed.bytes[3 + i],
                   ((uint64_t)displacement >> (8 * i)) & 0xff);
   }
   EXPECT_FALSE(agx_apple9_pack_branch(true, INT64_C(1) << 47, &packed));
   EXPECT_FALSE(agx_apple9_pack_branch(true, -(INT64_C(1) << 47) - 2, &packed));
   EXPECT_FALSE(agx_apple9_pack_branch(true, 3, &packed));
}

TEST(Apple9Compiler, BlockExportCompilesDynamicCoordinatesAndSparseImages)
{
   for (bool multisampled : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "block export");
      b.shader->info.num_images = 8;
      nir_def *group = nir_load_workgroup_id(&b);
      nir_def *xy = nir_imul_imm(&b, nir_trim_vector(&b, group, 2), 32);
      for (unsigned binding : {2u, 7u}) {
         nir_image_store_block_agx(&b, nir_imm_int(&b, binding),
            nir_imm_int(&b, binding == 2 ? 0 : 8), nir_pad_vec4(&b, xy),
            .image_dim = multisampled ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM);
      }
      b.shader->info.io_lowered = true;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment(b.shader, &compiled, &reason)) << (reason ?: "");
      EXPECT_EQ(compiled.info.apple9_image_mask, (1u << 2) | (1u << 7));
      EXPECT_EQ(compiled.info.apple9_texture_mask, 0u);
      unsigned exports = 0;
      const auto *code = static_cast<const uint8_t *>(compiled.binary);
      for (unsigned i = 0; i + 18 <= compiled.info.binary_size; ++i) {
         if (code[i] == 0x57 && code[i+8] == (multisampled ? 0x28 : 0xa8) &&
             code[i+9] == (multisampled ? 0x72 : 0x75)) {
            EXPECT_EQ(code[i+3] & 7, 0);
            EXPECT_LE(code[i+3], 24);
            EXPECT_EQ(code[i+5], exports << 4);
            EXPECT_EQ(code[i+12], 7);
            EXPECT_EQ(code[i+13], 0x12);
            ++exports;
         }
      }
      EXPECT_EQ(exports, 2u);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Encoding, BlockExportUsesAllocatedTupleAndImageSlot)
{
   agx_apple9_vir_instr store = {};
   store.op = AGX_APPLE9_VIR_BLOCK_IMAGE_STORE;
   store.encoding = AGX_APPLE9_ENC_BLOCK_IMAGE_STORE;
   store.nr_srcs = 3;
   store.src[0] = 0;
   store.src[1] = 1;
   store.src[2] = 2;
   store.texture_index = 9;
   store.immediate = 1;
   uint8_t phys[] = {8, 9, 10, 11};
   agx_apple9_packed_instruction packed = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&store, phys, &packed, &reason));
   EXPECT_EQ(packed.length, 18u);
   EXPECT_EQ(packed.bytes[3], 16);
   EXPECT_EQ(packed.bytes[5], 0x90);
   EXPECT_EQ(packed.bytes[9], 0x15);
   store.encoding = AGX_APPLE9_ENC_BLOCK_IMAGE_STORE_MS;
   store.nr_srcs = 4;
   store.src[3] = 3;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&store, phys, &packed, &reason));
   EXPECT_EQ(packed.bytes[3], 16);
   EXPECT_EQ(packed.bytes[5], 0x90);
   EXPECT_EQ(packed.bytes[8], 0x28);
   EXPECT_EQ(packed.bytes[9], 0x12);
   phys[3] = 12;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&store, phys, &packed, &reason));
   phys[3] = 11;
   store.nr_srcs = 3;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&store, phys, &packed, &reason));
   store.nr_srcs = 4;
   phys[1] = 10;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&store, phys, &packed, &reason));
   phys[1] = 9;
   store.texture_index = 16;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&store, phys, &packed, &reason));
}

TEST(Apple9Allocator, SharedLoadHandoffRetainsValuesAndCompletesVectorSiblings)
{
   for (unsigned components : {1u, 4u}) {
      for (bool logic : {false, true}) {
         nir_builder nir = nir_builder_init_simple_shader(
            MESA_SHADER_VERTEX, &agx_nir_options, "retained_load_handoff");
         nir_block logical = {};
         agx_apple9_vir_program p;
         agx_apple9_vir_init(&p);
         auto block = agx_apple9_block_create(&p);
         block->nir = &logical;
         agx_apple9_block_begin(&p, block);
         auto index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
            AGX_APPLE9_ENC_MOV_IMM_COMPACT, nullptr, 0, 0);
         auto ordinary = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
            AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x3f000000);
         agx_apple9_device_load_contract contract = {
            .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
            .flags = 0, .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
         };
         auto load = components == 1
            ? agx_apple9_vir_emit_device_load(&p, 0, index, &contract)
            : agx_apple9_vir_emit_device_load_vector(&p, 0, index, components, &contract);
         ASSERT_NE(load, AGX_APPLE9_VREG_INVALID);
         uint32_t sources[] = {load, ordinary};
         auto first = agx_apple9_vir_emit(&p,
            logic ? AGX_APPLE9_VIR_IXOR : AGX_APPLE9_VIR_FADD,
            logic ? AGX_APPLE9_ENC_LOGIC_EXTENDED : AGX_APPLE9_ENC_FLOAT2_COMPACT,
            sources, 2, 0);
         auto later = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMUL,
            AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
         auto output = [&](uint32_t value, unsigned slot) {
            uint32_t src[] = {value, value};
            auto pub = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR,
               AGX_APPLE9_ENC_LOGIC_EXPORT, src, 2, 0);
            EXPECT_TRUE(agx_apple9_vir_emit_side_effect(&p,
               AGX_APPLE9_VIR_VARY_STORE, AGX_APPLE9_ENC_VARY_STORE, &pub, 1, slot));
         };
         output(first, 0);
         output(later, 1);
         for (unsigned c = 1; c < components; ++c)
            output(load + c, c + 1);
         const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
         unsigned load_slot = 0, first_uses = 0, later_uses = 0;
         for (unsigned i = 0; i < p.instruction_count; ++i) {
            const auto *ins = p.instructions[i];
            if (ins->op == AGX_APPLE9_VIR_DEVICE_LOAD)
               load_slot = ins->producer_scoreboard_slot;
            if (ins->op == (logic ? AGX_APPLE9_VIR_IXOR : AGX_APPLE9_VIR_FADD)) {
               ASSERT_NE(load_slot, 0u);
               EXPECT_EQ(ins->scoreboard_slot, load_slot);
               EXPECT_NE(ins->live_after_mask & 1, 0u);
               ++first_uses;
            }
            if (ins->op == AGX_APPLE9_VIR_FMUL &&
                (ins->encoding == AGX_APPLE9_ENC_FLOAT2_COMPACT ||
                 ins->encoding == AGX_APPLE9_ENC_FLOAT2_BASE)) {
               EXPECT_EQ(ins->scoreboard_slot, 0u);
               ++later_uses;
            }
         }
         EXPECT_EQ(first_uses, 1u);
         EXPECT_EQ(later_uses, 1u);
         agx_apple9_vir_finish(&p);
         ralloc_free(nir.shader);
      }
   }
}

TEST(Apple9Packing, FloatExportInputMaskIsIndependentOfOutputCompletion)
{
   for (unsigned input = 1; input <= 6; ++input) {
      agx_apple9_vir_instr ins = {};
      ins.op = AGX_APPLE9_VIR_FMUL;
      ins.encoding = AGX_APPLE9_ENC_FLOAT2_EXPORT;
      ins.dest = 0; ins.dest_components = 1;
      ins.src[0] = 1; ins.src[1] = 2; ins.nr_srcs = 2;
      ins.scoreboard_slot = input;
      ins.producer_scoreboard_slot = 7 - input;
      uint8_t phys[] = {9, 4, 6};
      agx_apple9_packed_instruction packed = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason)) << reason;
      EXPECT_EQ(packed.length, 8u);
      EXPECT_EQ((packed.bytes[5] >> 5) | ((packed.bytes[7] >> 5) << 3),
                1u << (input - 1));
      EXPECT_EQ((packed.bytes[7] >> 2) & 7, 6u - input);
   }
}

class Apple9Completion : public ::testing::Test {
protected:
   nir_builder b;
   nir_block logical = {};
   agx_apple9_vir_program p;

   void SetUp() override
   {
      b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &agx_nir_options,
                                         "physical_completions");
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);
   }

   void TearDown() override
   {
      agx_apple9_vir_finish(&p);
      ralloc_free(b.shader);
   }

   uint32_t imm(uint32_t x)
   {
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, x);
   }

   uint32_t load(uint32_t index, unsigned binding, unsigned components = 1)
   {
      agx_apple9_device_load_contract c = {
         .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
         .flags = 0, .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
      };
      return components == 1
         ? agx_apple9_vir_emit_device_load(&p, binding, index, &c)
         : agx_apple9_vir_emit_device_load_vector(&p, binding, index, components, &c);
   }

   void output(uint32_t x, unsigned slot)
   {
      uint32_t src[] = {x, x};
      auto pub = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IOR,
         AGX_APPLE9_ENC_LOGIC_EXPORT, src, 2, 0);
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_VARY_STORE,
         AGX_APPLE9_ENC_VARY_STORE, &pub, 1, slot));
   }
};

TEST_F(Apple9Completion, PointerTupleRetiresAtRetainedAddressUse)
{
   auto zero = imm(0);
   auto address = load(zero, 0, 2);
   auto first = load(zero, 1);
   ASSERT_TRUE(agx_apple9_vir_set_load_address(&p, first, address));
   auto second = load(zero, 2);
   ASSERT_TRUE(agx_apple9_vir_set_load_address(&p, second, address));
   output(first, 0);
   output(second, 1);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned pointers = 0, slot = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         continue;
      if (ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD) {
         slot = ins->producer_scoreboard_slot;
         continue;
      }
      ASSERT_NE(slot, 0u);
      EXPECT_EQ(ins->scoreboard_slot, pointers ? 0u : slot);
      EXPECT_EQ(bool(ins->live_after_mask & 6), pointers == 0);
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(ins, p.phys, &packed, &reason)) << reason;
      EXPECT_EQ((packed.bytes[1] >> 4) | ((packed.bytes[2] & 3) << 4),
                pointers ? 0u : 1u << (slot - 1));
      EXPECT_EQ(bool(packed.bytes[9] & 4), pointers != 0);
      ++pointers;
   }
   EXPECT_EQ(pointers, 2u);
}

TEST_F(Apple9Completion, IndexCompletionDoesNotPreventLaterOrdinaryUse)
{
   auto zero = imm(0);
   auto index = load(zero, 0);
   auto value = load(index, 1);
   output(value, 0);
   output(index, 1);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned slot = 0, loads = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         continue;
      if (loads++ == 0) {
         slot = ins->producer_scoreboard_slot;
      } else {
         EXPECT_EQ(ins->scoreboard_slot, slot);
         EXPECT_NE(ins->live_after_mask & 1, 0u);
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(ins, p.phys, &packed, &reason)) << reason;
         EXPECT_EQ(packed.bytes[5] & 0x80, 0);
      }
   }
   EXPECT_EQ(loads, 2u);
}

TEST_F(Apple9Completion, AsyncSystemReadsRetireSlotConflictsAndFoldRealUses)
{
   auto zero = imm(0);
   auto a = load(zero, 0);
   auto occupied = load(zero, 1);
   auto id = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_GET_SR,
      AGX_APPLE9_ENC_GET_DRAW_ID, nullptr, 0, 0x10dd);
   uint32_t src[] = {id, zero};
   auto first = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   auto coverage = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_GET_SR,
      AGX_APPLE9_ENC_GET_COVERAGE, nullptr, 0, 0x10c2);
   src[0] = coverage;
   auto second = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   output(first, 0);
   output(second, 1);
   output(a, 2);
   output(occupied, 3);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned reads = 0, consumers = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->op == AGX_APPLE9_VIR_GET_SR) {
         EXPECT_EQ(ins->producer_scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_1);
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(ins, p.phys, &packed, &reason)) << reason;
         EXPECT_EQ(packed.length, 4u);
         auto unassigned = *ins;
         unassigned.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(
            &unassigned, p.phys, &packed, &reason));
         if (reads++ == 0) {
            ASSERT_GT(i, 0u);
            EXPECT_EQ(p.instructions[i - 1]->scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_1);
         }
      }
      if (ins->op == AGX_APPLE9_VIR_IXOR) {
         EXPECT_EQ(ins->scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_1);
         ++consumers;
      }
   }
   EXPECT_EQ(reads, 2u);
   EXPECT_EQ(consumers, 2u);
}

TEST_F(Apple9Completion, TextureNonleadingHandoffRetainsTupleForLaterReads)
{
   auto half = imm(0x3f000000);
   uint32_t coords[] = {half, half};
   auto sample = agx_apple9_vir_emit_texture_sample(&p, coords, half, 0, 0);
   ASSERT_NE(sample, AGX_APPLE9_VREG_INVALID);
   uint32_t src[] = {sample + 2, half};
   auto first = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, src, 2, 0);
   auto later = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FMUL,
      AGX_APPLE9_ENC_FLOAT2_COMPACT, src, 2, 0);
   output(first, 0);
   output(later, 1);
   output(sample, 2);
   output(sample + 1, 3);
   output(sample + 3, 4);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned samples = 0, first_uses = 0, later_uses = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE)
         ++samples;
      if (ins->op == AGX_APPLE9_VIR_FADD) {
         EXPECT_EQ(ins->scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_1);
         EXPECT_NE(ins->live_after_mask & 1, 0u);
         ++first_uses;
      }
      if (ins->op == AGX_APPLE9_VIR_FMUL &&
          (ins->encoding == AGX_APPLE9_ENC_FLOAT2_COMPACT ||
           ins->encoding == AGX_APPLE9_ENC_FLOAT2_BASE)) {
         EXPECT_EQ(ins->scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_NONE);
         ++later_uses;
      }
   }
   EXPECT_EQ(samples, 1u);
   EXPECT_EQ(first_uses, 1u);
   EXPECT_EQ(later_uses, 1u);
}

/* Golden results from independently authored T8132 cube-coordinate shaders. */
TEST(Apple9Packer, CubeWritesMagnitudeFacePairAndHalfCoordinates)
{
   const uint8_t expected[3][12] = {
      {0x17,1,0x54,4,3,0,5,0x12,0x54,0x21,0x92,0},
      {0x17,1,0x54,3,3,0,5,0x12,0x54,0x21,0x92,4},
      {0x17,1,0x54,0,3,0,4,0x10,0x50,0x2f,0x92,8},
   };
   for (unsigned mode = 0; mode < 3; ++mode) {
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      uint32_t src[] = {agx_apple9_vir_input(&p, 0),
                        agx_apple9_vir_input(&p, 1),
                        agx_apple9_vir_input(&p, 2)};
      uint32_t result = agx_apple9_vir_emit_cube(&p, src, mode);
      ASSERT_NE(result, AGX_APPLE9_VREG_INVALID);
      auto *ins = p.instructions[0];
      EXPECT_EQ(ins->dest_components, mode == 0 ? 2 : 1);
      EXPECT_EQ(p.value_count, 3 + ins->dest_components);
      ins->live_after_mask = mode == 2 ? 0 : 7;
      const uint8_t phys[] = {0, 1, 2, uint8_t(mode == 0 ? 4 : mode == 1 ? 3 : 0), 5};
      agx_apple9_packed_instruction packed;
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(ins, phys, &packed, &reason))
         << (reason ?: "");
      ASSERT_EQ(packed.length, 12);
      EXPECT_EQ(memcmp(packed.bytes, expected[mode], 12), 0);
      agx_apple9_vir_finish(&p);
   }
}

TEST(Apple9Packer, TextureCompletionTags)
{
   /* Paired tag bytes from controlled retags of our own M4 Metal shaders. */
   const uint8_t tags[] = {0x00, 0x24, 0x48, 0x6c, 0x90, 0xb4};
   for (auto encoding : {AGX_APPLE9_ENC_TEXTURE_SAMPLE,
                         AGX_APPLE9_ENC_TEXTURE_LOD,
                         AGX_APPLE9_ENC_TEXTURE_GRAD}) {
      agx_apple9_vir_instr ins = {};
      ins.op = AGX_APPLE9_VIR_TEXTURE_SAMPLE;
      ins.encoding = encoding;
      ins.nr_srcs = encoding == AGX_APPLE9_ENC_TEXTURE_GRAD ? 1 :
                    encoding == AGX_APPLE9_ENC_TEXTURE_LOD ? 4 : 2;
      ins.dest = 4;
      ins.dest_components = 4;
      for (unsigned i = 0; i < ins.nr_srcs; ++i)
         ins.src[i] = i;
      uint8_t phys[] = {0, 1, 2, 3, 4, 5, 6, 7};
      for (unsigned slot = 0; slot <= 7; ++slot) {
         ins.producer_scoreboard_slot = slot;
         agx_apple9_packed_instruction packed = {};
         const char *reason = nullptr;
         bool valid = agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason);
         ASSERT_EQ(valid, slot >= 1 && slot <= 6);
         if (valid) {
            EXPECT_EQ(packed.length, 14u);
            EXPECT_EQ(packed.bytes[5], tags[slot - 1]);
            for (unsigned base : {16u, 24u, 28u, 32u, 48u, 60u}) {
               for (unsigned c = 0; c < 4; ++c)
                  phys[4 + c] = base + c;
               ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
               EXPECT_EQ(packed.bytes[0], 5 | ((base & 31) << 3));
               EXPECT_EQ((packed.bytes[2] >> 6) & 1, base >> 5);
               EXPECT_EQ(packed.bytes[5], tags[slot - 1]);
            }
            for (unsigned base : {0u, 8u, 16u, 24u}) {
               for (unsigned c = 0; c < 4; ++c)
                  phys[c] = base + c;
               ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
               EXPECT_EQ(packed.bytes[1] & 7, base & 7);
               EXPECT_EQ(packed.bytes[3] & 3, base >> 3);
            }
            for (unsigned c = 0; c < 4; ++c)
               phys[c] = c;
            phys[4] = 61;
            EXPECT_FALSE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
            for (unsigned c = 0; c < 4; ++c)
               phys[4 + c] = 4 + c;
         }
      }
   }
}

TEST_F(Apple9Completion, IndependentTexturesWaitAtUse)
{
   auto half = imm(0x3f000000);
   uint32_t coords[] = {half, half};
   uint32_t samples[3];
   for (unsigned i = 0; i < 3; ++i)
      samples[i] = agx_apple9_vir_emit_texture_sample(&p, coords, half, i, i);
   for (unsigned i = 0; i < 3; ++i) {
      uint32_t src[] = {samples[i] + 2, half};
      auto first = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_FADD,
         AGX_APPLE9_ENC_FLOAT2_COMPACT, src, 2, 0);
      output(first, i);
      output(samples[i] + 3, i + 4);
   }
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned issued = 0, waited = 0, active = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->scoreboard_slot && (active & (1u << ins->scoreboard_slot))) {
         EXPECT_EQ(issued, 3u);
         EXPECT_NE(active & (1u << ins->scoreboard_slot), 0u);
         active &= ~(1u << ins->scoreboard_slot);
         ++waited;
      }
      if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
         EXPECT_EQ(ins->producer_scoreboard_slot, issued + 1);
         active |= 1u << ins->producer_scoreboard_slot;
         ++issued;
      }
   }
   EXPECT_EQ(issued, 3u);
   EXPECT_EQ(waited, 3u);
   EXPECT_EQ(active, 0u);
}

TEST_F(Apple9Completion, DependentTextureWaitsBeforeCoordinatePublication)
{
   auto half = imm(0x3f000000);
   uint32_t coords[] = {half, half};
   auto first = agx_apple9_vir_emit_texture_sample(&p, coords, half, 0, 0);
   coords[0] = first;
   coords[1] = first + 1;
   auto second = agx_apple9_vir_emit_texture_sample(&p, coords, half, 1, 1);
   output(second, 0);
   output(first + 3, 1);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned samples = 0, waits = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->scoreboard_slot && samples == 1)
         ++waits;
      if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
         if (samples++)
            EXPECT_EQ(waits, 1u);
      }
   }
   EXPECT_EQ(samples, 2u);
}

TEST_F(Apple9Completion, TexturesShareTagsWithMemoryAndRetireOnExhaustion)
{
   auto half = imm(0x3f000000), zero = imm(0);
   uint32_t values[5];
   for (unsigned i = 0; i < 5; ++i)
      values[i] = load(zero, i);
   uint32_t coords[] = {half, half};
   auto first = agx_apple9_vir_emit_texture_sample(&p, coords, half, 0, 0);
   auto second = agx_apple9_vir_emit_texture_sample(&p, coords, half, 1, 1);
   output(first, 0);
   output(second, 1);
   for (unsigned i = 0; i < 5; ++i)
      output(values[i], i + 2);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned active = 0, peak = 0, samples = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      if (ins->scoreboard_slot) {
         EXPECT_NE(active & (1u << ins->scoreboard_slot), 0u);
         active &= ~(1u << ins->scoreboard_slot);
      }
      if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ||
          ins->op == AGX_APPLE9_VIR_DEVICE_LOAD ||
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT) {
         unsigned bit = 1u << ins->producer_scoreboard_slot;
         EXPECT_EQ(active & bit, 0u);
         active |= bit;
         peak = MAX2(peak, util_bitcount(active));
         samples += ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE;
      }
   }
   EXPECT_EQ(samples, 2u);
   EXPECT_EQ(peak, 6u);
   EXPECT_EQ(active, 0u);
}

TEST(Apple9Compiler, TransformFeedbackUsesOrdinaryStoresAndPolyInputAssembly)
{
   for (auto mode : {MESA_PRIM_POINTS, MESA_PRIM_LINES, MESA_PRIM_LINE_STRIP,
                     MESA_PRIM_LINE_LOOP, MESA_PRIM_TRIANGLES,
                     MESA_PRIM_TRIANGLE_STRIP, MESA_PRIM_TRIANGLE_FAN}) {
      for (unsigned index_size : {0u, 1u, 2u, 4u}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_VERTEX, &agx_nir_options, "transform_feedback");
         nir_def *zero = nir_imm_int(&b, 0);
         nir_def *input = nir_load_input(
            &b, 4, 32, zero, .base = 0, .dest_type = nir_type_float32,
            .io_semantics = {.location = VERT_ATTRIB_GENERIC0, .num_slots = 1});
         nir_store_output(
            &b, input, zero, .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_def *ids =
            nir_vec2(&b, nir_load_vertex_id(&b), nir_load_instance_id(&b));
         nir_store_output(
            &b, ids, zero, .base = 1, .write_mask = 3,
            .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
         auto *xfb =
            (nir_xfb_info *)rzalloc_size(b.shader, nir_xfb_info_size(2));
         xfb->buffers_written = 5;
         xfb->streams_written = 1;
         xfb->output_count = 2;
         xfb->buffers[0].stride = 32;
         xfb->buffers[2].stride = 16;
         xfb->outputs[0] = {.buffer = 0,
                            .offset = 12,
                            .location = VARYING_SLOT_POS,
                            .component_mask = 15};
         xfb->outputs[1] = {.buffer = 2,
                            .offset = 8,
                            .location = VARYING_SLOT_VAR0,
                            .component_mask = 3};
         b.shader->xfb_info = xfb;
         b.shader->info.outputs_written =
            VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
         b.shader->info.inputs_read = BITFIELD64_BIT(VERT_ATTRIB_GENERIC0);
         b.shader->info.io_lowered = true;
         agx_apple9_vertex_layout layout = {};
         layout.capture_xfb = true;
         layout.xfb_mode = mode;
         layout.xfb_index_size = index_size;
         layout.format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;
         layout.stride[0] = 16;
         agx_shader_part out = {};
         const char *reason = nullptr;
         ASSERT_TRUE(
            agx_compile_apple9_vertex_inputs(b.shader, &layout, &out, &reason))
            << "mode=" << mode << " indices=" << index_size << " " << reason;
         unsigned written = 0;
         for (unsigned i = 0; i < out.info.apple9_resource_count; ++i) {
            if (out.info.apple9_resource_write_mask & (1u << i)) {
               unsigned binding = out.info.apple9_resource_binding[i];
               EXPECT_TRUE(binding == AGX_APPLE9_XFB_BUFFER_BASE ||
                           binding == AGX_APPLE9_XFB_BUFFER_BASE + 2);
               written |= 1u << (binding - AGX_APPLE9_XFB_BUFFER_BASE);
               EXPECT_NE(out.info.apple9_resource_ssbo_mask & (1u << i), 0u);
            }
         }
         EXPECT_EQ(written, 5u);
         EXPECT_EQ(out.info.apple9_varyings.count, 0u);
         free(out.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Packer, UnaryHighRegisterBitsPreserveLifetimesAndDependencies)
{
   static const struct {
      agx_apple9_vir_opcode op;
      agx_apple9_encoding encoding;
      unsigned immediate;
   } cases[] = {
      {AGX_APPLE9_VIR_BIT_COUNT, AGX_APPLE9_ENC_BIT_UNARY, 0},
      {AGX_APPLE9_VIR_UFIND_MSB, AGX_APPLE9_ENC_BIT_UNARY, 0},
      {AGX_APPLE9_VIR_BIT_REVERSE, AGX_APPLE9_ENC_BIT_UNARY, 0},
      {AGX_APPLE9_VIR_U2F32, AGX_APPLE9_ENC_UINT_TO_FLOAT, 0},
      {AGX_APPLE9_VIR_I2F32, AGX_APPLE9_ENC_SINT_TO_FLOAT, 0},
      {AGX_APPLE9_VIR_F2I32, AGX_APPLE9_ENC_FLOAT_TO_SINT, 0},
      {AGX_APPLE9_VIR_F2U32, AGX_APPLE9_ENC_FLOAT_TO_UINT, 0},
      {AGX_APPLE9_VIR_ISHR, AGX_APPLE9_ENC_SHIFT_EXTENDED, 7},
      {AGX_APPLE9_VIR_HRCP, AGX_APPLE9_ENC_HALF_SPECIAL, 2},
      {AGX_APPLE9_VIR_HRCP_F32, AGX_APPLE9_ENC_HALF_SPECIAL, 3},
      {AGX_APPLE9_VIR_HRSQ, AGX_APPLE9_ENC_HALF_SPECIAL, 2},
      {AGX_APPLE9_VIR_HSQRT_FACTOR, AGX_APPLE9_ENC_HALF_SPECIAL, 3},
      {AGX_APPLE9_VIR_HEXP2, AGX_APPLE9_ENC_HALF_SPECIAL, 2},
      {AGX_APPLE9_VIR_HLOG2, AGX_APPLE9_ENC_HALF_SPECIAL, 2},
      {AGX_APPLE9_VIR_HFLOOR, AGX_APPLE9_ENC_HALF_SPECIAL, 2},
      {AGX_APPLE9_VIR_HCEIL, AGX_APPLE9_ENC_HALF_SPECIAL, 2},
      {AGX_APPLE9_VIR_HTRUNC, AGX_APPLE9_ENC_HALF_SPECIAL, 2},
      {AGX_APPLE9_VIR_HROUND_EVEN, AGX_APPLE9_ENC_HALF_SPECIAL, 2},
      {AGX_APPLE9_VIR_FRCP, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FRSQ, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FSQRT_FACTOR, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FSIN_FACTOR, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FEXP2, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FLOG2, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FFLOOR, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FCEIL, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FTRUNC, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
      {AGX_APPLE9_VIR_FROUND_EVEN, AGX_APPLE9_ENC_FLOAT_SPECIAL, 3},
   };
   for (const auto &test : cases) {
      SCOPED_TRACE(test.op);
      for (unsigned live : {0u, 1u}) {
         for (unsigned slot = 0; slot <= 6; ++slot) {
            agx_apple9_vir_instr ins = {};
            ins.op = test.op;
            ins.encoding = test.encoding;
            ins.dest = 0;
            ins.src[0] = 1;
            ins.nr_srcs = 1;
            ins.immediate = test.immediate;
            ins.live_after_mask = live;
            ins.scoreboard_slot = static_cast<agx_apple9_scoreboard_slot>(slot);
            uint8_t phys[] = {31, 31};
            agx_apple9_packed_instruction low, high;
            const char *reason = nullptr;
            ASSERT_TRUE(
               agx_apple9_pack_vir_instruction(&ins, phys, &low, &reason))
               << reason;
            for (unsigned mask = 1; mask < 4; ++mask) {
               phys[0] = mask & 1 ? 95 : 31;
               phys[1] = mask & 2 ? 95 : 31;
               ASSERT_TRUE(
                  agx_apple9_pack_vir_instruction(&ins, phys, &high, &reason))
                  << reason;
               ASSERT_EQ(low.length, high.length);
               for (unsigned byte = 0; byte < low.length; ++byte) {
                  unsigned delta = byte == 3 && (mask & 1)   ? 0x80
                                   : byte == 6 && (mask & 2) ? 1
                                                             : 0;
                  EXPECT_EQ(high.bytes[byte], low.bytes[byte] ^ delta)
                     << "mask=" << mask << " byte=" << byte;
               }
            }
            phys[0] = 96;
            EXPECT_FALSE(
               agx_apple9_pack_vir_instruction(&ins, phys, &high, &reason));
            phys[0] = 31;
            phys[1] = 96;
            EXPECT_FALSE(
               agx_apple9_pack_vir_instruction(&ins, phys, &high, &reason));
         }
      }
   }
}

TEST(Apple9Packer, SystemRegisterAndZeroExtensionShareDestinationBits)
{
   for (unsigned dst = 0; dst < 64; ++dst) {
      agx_apple9_packed_instruction read, pair;
      ASSERT_TRUE(agx_apple9_pack_get_sr(dst, 0xa0, 0x10, &read));
      EXPECT_EQ(read.bytes[0], ((dst & 15) << 4) | 0x0c);
      EXPECT_EQ(read.bytes[1], 0xa0);
      EXPECT_EQ(read.bytes[2], 0x10 | ((dst >> 4) << 6));
      EXPECT_EQ(read.bytes[3], 6);
      ASSERT_TRUE(agx_apple9_pack_get_sr_zext16(dst, 0xa4, &pair));
      EXPECT_EQ(pair.bytes[2], read.bytes[2]);
      EXPECT_EQ(pair.bytes[4], ((dst & 15) << 4) | 3);
      EXPECT_EQ(pair.bytes[6], (dst >> 4) << 6);
      EXPECT_EQ(pair.bytes[7], 1);
   }
   agx_apple9_packed_instruction invalid;
   EXPECT_FALSE(agx_apple9_pack_get_sr(64, 0xa0, 0x10, &invalid));
   EXPECT_FALSE(agx_apple9_pack_get_sr(0, 0xa0, 0x50, &invalid));
   EXPECT_FALSE(agx_apple9_pack_get_sr_zext16(64, 0xa4, &invalid));
}

TEST(Apple9Compiler, PreambleRetainsCompleteResourceMap)
{
   nir_builder b = apple9_compute_builder("preamble_resource_map");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *uniform = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 7),
      nir_imm_int(&b, 16), .align_mul = 16, .range = 16);
   nir_def *value = nir_iadd_imm(&b,
      nir_imul_imm(&b, nir_channel(&b, uniform, 2), 13), 9);
   apple9_store_output(&b, gid, nir_ixor(&b, value, gid));
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.apple9_preamble_size, 0u);
   EXPECT_EQ(compiled.info.apple9_preamble_offset, compiled.info.main_size);
   EXPECT_EQ(compiled.info.binary_size,
             compiled.info.main_size + compiled.info.apple9_preamble_size);
   EXPECT_EQ(profile.preamble_size, compiled.info.apple9_preamble_size);
   EXPECT_EQ(profile.preamble_offset, compiled.info.apple9_preamble_offset);
   ASSERT_EQ(profile.resource_binding_count, 2u);
   bool found_ubo = false, found_output = false;
   for (unsigned i = 0; i < profile.resource_binding_count; ++i) {
      if (profile.resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_UBO) {
         EXPECT_EQ(profile.resource_binding[i], 7u);
         EXPECT_TRUE(profile.resource_read_mask & (1u << i));
         EXPECT_FALSE(profile.resource_write_mask & (1u << i));
         found_ubo = true;
      } else {
         EXPECT_EQ(profile.resource_binding[i], 0u);
         EXPECT_TRUE(profile.resource_write_mask & (1u << i));
         found_output = true;
      }
   }
   EXPECT_TRUE(found_ubo && found_output);
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, InvocationDependentUboStaysInMain)
{
   nir_shader *nir = apple9_ubo_load_shader();
   struct agx_shader_part compiled = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason));
   EXPECT_EQ(compiled.info.apple9_preamble_size, 0u);
   EXPECT_EQ(compiled.info.main_size, compiled.info.binary_size);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, PreambleShrinkingPreservesBufferByteAddresses)
{
   nir_builder b = apple9_compute_builder("preamble_vector_start");
   nir_def *gid = apple9_global_id_x(&b);
   nir_def *uniform = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 7),
      nir_imm_int(&b, 16), .align_mul = 16, .range_base = 16, .range = 16);
   nir_def *sum = nir_iadd(&b, nir_channel(&b, uniform, 1),
                              nir_channel(&b, uniform, 3));
   apple9_store_output(&b, gid, nir_ixor(&b, sum, gid));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ?: "");
   ASSERT_GT(compiled.info.apple9_preamble_size, 0u);

   /* The live words at 20 and 28 now fit a vec3 with byte displacement 20.
    * Its address must not be rounded to a 16-byte boundary. */
   agx_shader_part preamble = {};
   preamble.binary = (uint8_t *)compiled.binary + compiled.info.apple9_preamble_offset;
   preamble.info.binary_size = compiled.info.apple9_preamble_size;
   unsigned loads[4];
   ASSERT_EQ(apple9_binary_device_load_offsets(&preamble, loads, 4), 1u);
   const uint8_t *bytes = (const uint8_t *)preamble.binary + loads[0];
   EXPECT_EQ(bytes[8] & 0x0e, 0x0cu); /* Three dwords, starting at byte 20. */
   unsigned displacement = 0;
   for (unsigned bit = 0; bit < 16; ++bit)
      displacement |= ((bytes[(77 + bit) / 8] >> ((77 + bit) % 8)) & 1) << bit;
   EXPECT_EQ(displacement, 20u);
   EXPECT_EQ((bytes[12] >> 1) & 7, 1u); /* Unscaled byte index. */
   free(compiled.binary);
   ralloc_free(b.shader);
}

TEST(Apple9Compiler, PreambleUsesAllWordsAfterTheFullComputeResourceMap)
{
   for (unsigned resources : {2u, 3u, 18u}) {
      SCOPED_TRACE(resources);
      nir_builder b = apple9_compute_builder("preamble_argument_boundary");
      nir_def *gid = apple9_global_id_x(&b);
      unsigned base = 2 * (resources + 1);
      unsigned words = AGX_APPLE9_UNIFORM_COUNT - base;
      b.shader->info.num_ubos = resources - 1;
      for (unsigned i = 0; i < words; ++i) {
         nir_def *value = nir_load_ubo(
            &b, 1, 32, nir_imm_int(&b, i % (resources - 1)),
            nir_imm_int(&b, 4 * (i / (resources - 1))),
            .align_mul = 4, .range = 4);
         apple9_store_output(&b, nir_iadd_imm(&b, gid, i), nir_ixor(&b, value, gid));
      }
      agx_shader_part compiled = {};
      agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
         << (reason ?: "");
      ASSERT_EQ(profile.resource_binding_count, resources);
      ASSERT_GT(compiled.info.apple9_preamble_size, 0u);
      EXPECT_EQ(agx_apple9_compute_root_words(profile.resource_binding_count), base);

      /* Decode COPY-to-uniform destinations in the compiled setup. UBO
       * values feed invocation-dependent XORs, so every transferred word
       * remains live and none can be replaced by a constant or shared result. */
      const uint8_t *code = (const uint8_t *)compiled.binary +
                            compiled.info.apple9_preamble_offset;
      std::vector<bool> written(AGX_APPLE9_UNIFORM_COUNT, false);
      for (unsigned at = 0; at + 10 <= compiled.info.apple9_preamble_size; at += 2) {
         const uint8_t *p = code + at;
         if (p[0] == 0x9f && p[1] == 1 && (p[2] & 0xfd) == 0x54 &&
             (p[4] & 0xfe) == 0 && p[7] == 0xa8 && p[9] == 1) {
            unsigned word = (p[3] >> 1) | ((p[4] & 1) << 7);
            EXPECT_GE(word, base);
            written[word] = true;
         }
      }
      for (unsigned word = 0; word < AGX_APPLE9_UNIFORM_COUNT; ++word)
         EXPECT_EQ(written[word], word >= base) << "uniform word " << word;
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Packer, UniformLogicPreservesHighRegistersAndDependencies)
{
   agx_apple9_vir_instr ins = {};
   ins.op = AGX_APPLE9_VIR_IOR_UNIFORM;
   ins.encoding = AGX_APPLE9_ENC_LOGIC_UNIFORM;
   ins.dest = 0;
   ins.dest_components = 1;
   ins.src[0] = 1;
   ins.nr_srcs = 1;
   ins.immediate = 63;
   ins.live_after_mask = 1;
   ins.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
   const uint8_t phys[] = {64, 95};
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
   ASSERT_EQ(packed.length, 10u);
   auto bits = [&](unsigned start, unsigned width) {
      unsigned value = 0;
      for (unsigned i = 0; i < width; ++i)
         value |= ((packed.bytes[(start + i) / 8] >> ((start + i) % 8)) & 1u) << i;
      return value;
   };
   EXPECT_EQ(bits(8, 7), 127u);
   EXPECT_EQ(bits(4, 4) | bits(22, 2) << 4 | bits(44, 1) << 6, 64u);
   EXPECT_EQ(bits(25, 6) | bits(42, 1) << 6, 95u);
   EXPECT_EQ(bits(31, 1), 1u);
   EXPECT_EQ(bits(20, 1), 0u);
   EXPECT_EQ(bits(33, 1), 1u);
   EXPECT_EQ(bits(43, 1), 1u);
   EXPECT_EQ(bits(45, 3), 0u);
   EXPECT_EQ(bits(61, 3), 4u);
   for (unsigned u : {64u, 127u, 128u, 255u}) {
      ins.immediate = u;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
      EXPECT_EQ((bits(8, 7) >> 1) | bits(19, 1) << 6 | bits(40, 1) << 7, u);
      EXPECT_EQ(bits(25, 6) | bits(42, 1) << 6, 95u);
   }
   ins.immediate = 256;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
}

TEST(Apple9Packer, UniformStoreUsesEightBitDestinationAndFullGprSource)
{
   agx_apple9_vir_instr ins = {};
   ins.op = AGX_APPLE9_VIR_STORE_UNIFORM;
   ins.encoding = AGX_APPLE9_ENC_STORE_UNIFORM;
   ins.dest = AGX_APPLE9_VREG_INVALID;
   ins.src[0] = 0;
   ins.nr_srcs = 1;
   ins.immediate = 255;
   uint8_t phys[] = {95};
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
   const uint8_t expected[] = {0x9f, 0x01, 0x54, 0xfe, 0x01, 0x7c, 0x01, 0xa8, 0x13, 0x01};
   ASSERT_EQ(packed.length, sizeof(expected));
   EXPECT_EQ(memcmp(packed.bytes, expected, sizeof(expected)), 0);
   ins.immediate = 256;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
   ins.immediate = 255;
   phys[0] = 96;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&ins, phys, &packed, &reason));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(AGX_APPLE9_ENC_STORE_UNIFORM,
      AGX_APPLE9_OPERAND_SRC0, 95, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(AGX_APPLE9_ENC_STORE_UNIFORM,
      AGX_APPLE9_OPERAND_SRC0, 96, 32));
}

TEST_F(Apple9Completion, UniformWritePreservesSourcesUsedLater)
{
   auto input = load(imm(0), 0, 4);
   for (unsigned i = 0; i < 4; ++i) {
      uint32_t src = input + i;
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(&p, AGX_APPLE9_VIR_STORE_UNIFORM,
         AGX_APPLE9_ENC_STORE_UNIFORM, &src, 1, AGX_APPLE9_GRAPHICS_ROOT_WORDS + i));
   }
   output(input, 0);
   output(input + 3, 1);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned writes = 0;
   bool had_load = false, completed = false;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *ins = p.instructions[i];
      had_load |= ins->op == AGX_APPLE9_VIR_DEVICE_LOAD;
      completed |= had_load && ins->scoreboard_slot;
      if (ins->op != AGX_APPLE9_VIR_STORE_UNIFORM)
         continue;
      EXPECT_TRUE(completed);
      EXPECT_EQ(ins->live_after_mask, (writes == 0 || writes == 3) ? 1u : 0u);
      EXPECT_LT(p.phys[ins->src[0]], AGX_APPLE9_GPR_COUNT);
      ++writes;
   }
   EXPECT_EQ(writes, 4u);
}

TEST(Apple9Packer, UniformSourceBanksRemainIndependentOfHighGprs)
{
   struct Case { agx_apple9_vir_opcode op; agx_apple9_encoding encoding; };
   for (auto test : {Case{AGX_APPLE9_VIR_FADD, AGX_APPLE9_ENC_FLOAT2_COMPACT},
                    Case{AGX_APPLE9_VIR_FSUB, AGX_APPLE9_ENC_FLOAT2_COMPACT},
                    Case{AGX_APPLE9_VIR_FMUL, AGX_APPLE9_ENC_FLOAT2_COMPACT},
                    Case{AGX_APPLE9_VIR_IAND, AGX_APPLE9_ENC_LOGIC_EXTENDED},
                    Case{AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED},
                    Case{AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED},
                    Case{AGX_APPLE9_VIR_IMIN, AGX_APPLE9_ENC_MINMAX_COMPACT},
                    Case{AGX_APPLE9_VIR_IMAX, AGX_APPLE9_ENC_MINMAX_COMPACT},
                    Case{AGX_APPLE9_VIR_UMIN, AGX_APPLE9_ENC_MINMAX_COMPACT},
                    Case{AGX_APPLE9_VIR_UMAX, AGX_APPLE9_ENC_MINMAX_COMPACT},
                    Case{AGX_APPLE9_VIR_FMIN, AGX_APPLE9_ENC_MINMAX_COMPACT},
                    Case{AGX_APPLE9_VIR_FMAX, AGX_APPLE9_ENC_MINMAX_COMPACT}}) {
      for (unsigned u : {0u, 63u, 64u, 127u, 128u, 255u, 256u}) {
         for (unsigned role = 0; role < 2; ++role) {
            SCOPED_TRACE(::testing::Message() << test.op << " u" << u << " source " << role);
            agx_apple9_vir_instr I = {};
            I.op = test.op; I.encoding = test.encoding;
            I.dest = 0; I.src[0] = 1; I.nr_srcs = 1;
            I.live_after_mask = 1;
            I.alu_src_uniform_mask = 1 << role; I.alu_src_value[role] = u;
            const uint8_t phys[] = {95, 64};
            agx_apple9_packed_instruction packed;
            const char *reason = nullptr;
            bool ok = agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason);
            ASSERT_EQ(ok, u < 256) << (reason ?: "");
            if (!ok) continue;
            auto bit = [&](unsigned n) { return (packed.bytes[n / 8] >> (n % 8)) & 1; };
            unsigned index = ((packed.bytes[1 + 2 * role] & 127) >> 1) |
               (bit(role ? 20 : 19) << 6) | (bit(role ? 42 : 40) << 7);
            EXPECT_EQ(index, u);
            EXPECT_EQ(bit(role ? 41 : 39), 1);
            EXPECT_EQ(bit(role ? 40 : 42), 1); // Other source is GPR r64.
            EXPECT_EQ(bit(role ? 19 : 20), 0); // Its lifetime must be retained.
         }
      }
   }
}

TEST(Apple9Packer, FmaUniformBanksAndCompactBoundary)
{
   for (unsigned mask : {2u, 4u, 6u}) {
      agx_apple9_vir_instr I = {};
      I.op = AGX_APPLE9_VIR_FMA; I.encoding = AGX_APPLE9_ENC_FLOAT3_EXTENDED;
      I.dest = 0; I.src[0] = 1; I.src[1] = 2;
      I.nr_srcs = mask == 6 ? 1 : 2;
      I.live_after_mask = (1 << I.nr_srcs) - 1;
      I.alu_src_uniform_mask = mask;
      I.alu_src_value[1] = 255; I.alu_src_value[2] = 128;
      const uint8_t phys[] = {95, 64, 65};
      agx_apple9_packed_instruction packed;
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
      auto bit = [&](unsigned n) { return (packed.bytes[n / 8] >> (n % 8)) & 1; };
      EXPECT_EQ(bit(56), 1); // GPR A remains r64.
      if (mask & 2) {
         EXPECT_EQ(bit(57), 1);
         EXPECT_EQ(((packed.bytes[3] & 127) >> 1) | bit(20) << 6 | bit(58) << 7, 255);
      }
      if (mask & 4) {
         EXPECT_EQ(bit(37), 1);
         EXPECT_EQ(((packed.bytes[5] & 127) >> 1) | bit(39) << 6 | bit(38) << 7, 128);
      }
      I.alu_src_value[mask & 2 ? 1 : 2] = 256;
      EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   }
   agx_apple9_vir_instr I = {};
   I.op = AGX_APPLE9_VIR_FMA; I.encoding = AGX_APPLE9_ENC_FLOAT3_COMPACT;
   I.dest = 0; I.src[0] = 1; I.src[1] = 2; I.nr_srcs = 2;
   I.alu_src_uniform_mask = 2;
   const uint8_t phys[] = {4, 5, 6};
   agx_apple9_packed_instruction packed;
   const char *reason = nullptr;
   I.alu_src_value[1] = 63;
   EXPECT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   I.alu_src_value[1] = 64;
   EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
   I.encoding = AGX_APPLE9_ENC_FLOAT3_EXTENDED;
   EXPECT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
}


static unsigned
apple9_test_bits(const uint8_t *bytes, unsigned start, unsigned count)
{
   unsigned value = 0;
   for (unsigned i = 0; i < count; ++i)
      value |= ((bytes[(start + i) / 8] >> ((start + i) % 8)) & 1) << i;
   return value;
}

TEST(Apple9Packer, IntegerUniformFilesPreserveHighIndicesAndGprLifetimes)
{
   for (bool mad : {false, true}) {
      unsigned arity = mad ? 3 : 2;
      for (unsigned mask = 0; mask < (1u << arity); ++mask) {
         for (unsigned live = 0; live < (1u << arity); ++live) {
            agx_apple9_vir_instr I = {};
            I.op = mad ? AGX_APPLE9_VIR_IMAD : AGX_APPLE9_VIR_IADD;
            I.encoding = mad ? AGX_APPLE9_ENC_INT_MAD_EXTENDED : AGX_APPLE9_ENC_INT_ADD_EXTENDED;
            I.dest = 0; I.dest_components = 1;
            I.alu_src_uniform_mask = mask; I.live_after_mask = live;
            const unsigned uniforms[] = {255, 128, 127};
            const uint8_t phys[] = {95, 64, 65, 66};
            for (unsigned s = 0; s < arity; ++s) {
               I.alu_src_value[s] = uniforms[s];
               if (!(mask & (1u << s))) I.src[I.nr_srcs++] = s + 1;
            }
            const char *reason = nullptr;
            agx_apple9_packed_instruction packed;
            ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
            unsigned gpr = 0;
            for (unsigned s = 0; s < arity; ++s) {
               bool uniform = mask & (1u << s);
               EXPECT_EQ(apple9_test_bits(packed.bytes, (mad ? 81 : 72) + 2 * s, 1), !uniform);
               unsigned value = apple9_test_bits(packed.bytes, 42 + 9 * s, uniform ? 8 : 7);
               EXPECT_EQ(value, uniform ? uniforms[s] : phys[s + 1]);
               EXPECT_EQ(apple9_test_bits(packed.bytes, (mad ? 73 : 65) + s, 1),
                         uniform || !(live & (1u << gpr)));
               if (!uniform) ++gpr;
            }
            if (mask) {
               I.alu_src_value[ffs(mask) - 1] = 256;
               EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
            }
         }
      }
   }
}

TEST(Apple9Packer, FmaFirstUniformSupportsEveryBankAndExtendedModifierForm)
{
   for (auto encoding : {AGX_APPLE9_ENC_FLOAT3_EXTENDED,
                         AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED,
                         AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED}) {
      for (unsigned u : {0u, 63u, 64u, 127u, 128u, 255u}) {
         agx_apple9_vir_instr I = {};
         I.op = AGX_APPLE9_VIR_FMA; I.encoding = encoding;
         I.dest = 0; I.dest_components = 1;
         I.nr_srcs = 1; I.src[0] = 1; I.live_after_mask = 1;
         I.alu_src_uniform_mask = 3; I.alu_src_value[0] = u; I.alu_src_value[1] = 255;
         I.src_abs_mask = encoding == AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED ? 3 : 0;
         I.saturate = encoding != AGX_APPLE9_ENC_FLOAT3_EXTENDED;
         const uint8_t phys[] = {95, 64};
         agx_apple9_packed_instruction packed; const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
         EXPECT_EQ(apple9_test_bits(packed.bytes, 55, 1), 1u);
         EXPECT_EQ(apple9_test_bits(packed.bytes, 9, 6) |
                   apple9_test_bits(packed.bytes, 19, 1) << 6 |
                   apple9_test_bits(packed.bytes, 56, 1) << 7, u);
         EXPECT_EQ(apple9_test_bits(packed.bytes, 38, 1), 1u); // r64 addend.
         I.encoding = AGX_APPLE9_ENC_FLOAT3_COMPACT;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
      }
   }
}

TEST(Apple9Packer, NormalizedUnpackOwnsBothOutputsAndPreservesSourceHalf)
{
   for (unsigned mode : {2u, 3u, 4u, 6u}) {
      for (unsigned half : {0u, 1u}) {
         agx_apple9_vir_instr I = {};
         I.op = AGX_APPLE9_VIR_UNPACK_NORM; I.encoding = AGX_APPLE9_ENC_UNPACK_NORM;
         I.dest = 0; I.dest_components = 2; I.src[0] = 2; I.nr_srcs = 1;
         I.immediate = mode | (half ? AGX_APPLE9_UNPACK_HIGH_HALF : 0);
         uint8_t phys[] = {94, 95, 65};
         agx_apple9_packed_instruction packed; const char *reason = nullptr;
         bool valid = !half || mode == 3 || mode == 6;
         ASSERT_EQ(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason), valid);
         if (!valid) continue;
         EXPECT_EQ(packed.length, 8);
         EXPECT_EQ(apple9_test_bits(packed.bytes, 25, 7), 94u);
         EXPECT_EQ(apple9_test_bits(packed.bytes, 42, 7), 65u);
         EXPECT_EQ(apple9_test_bits(packed.bytes, 41, 1), half);
         EXPECT_EQ(apple9_test_bits(packed.bytes, 60, 3), mode);
         EXPECT_EQ(apple9_test_bits(packed.bytes, 52, 1), 1u);
         I.live_after_mask = 1;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
         EXPECT_EQ(apple9_test_bits(packed.bytes, 49, 1), 1u);
         EXPECT_EQ(apple9_test_bits(packed.bytes, 52, 1), 0u);
         phys[1] = 93;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
      }
   }
}

TEST(Apple9Compiler, NormalizedUnpackSurvivesLoweringAsPairs)
{
   for (auto op : {nir_op_unpack_unorm_4x8, nir_op_unpack_snorm_4x8,
                   nir_op_unpack_unorm_2x16, nir_op_unpack_snorm_2x16}) {
      nir_builder b = apple9_compute_builder("native_normalized_unpack");
      nir_def *gid = apple9_global_id_x(&b);
      nir_def *value = nir_build_alu(&b, op, gid, nullptr, nullptr, nullptr);
      const unsigned components = value->num_components;
      for (unsigned c = 0; c < components; ++c)
         apple9_store_output(&b, nir_iadd_imm(&b, nir_imul_imm(&b, gid, 4), c),
                             nir_channel(&b, value, c));
      agx_shader_part compiled = {}; agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason)) << reason;
      const auto *code = static_cast<const uint8_t *>(compiled.binary);
      unsigned count = 0;
      for (unsigned i = 0; i + 8 <= compiled.info.main_size; i += 2)
         count += (code[i] == 0x17 && (code[i + 1] & 15) == 4);
      EXPECT_EQ(count, components / 2);
      free(compiled.binary); ralloc_free(b.shader);
   }
}

TEST(Apple9Packer, MemoryScaleAndSignedOffsetAreIndependentOfVectorWidth)
{
   for (bool store : {false, true}) for (unsigned components : {1u, 2u, 3u, 4u}) {
      for (unsigned shift = 0; shift <= 4; ++shift) for (int offset : {-32768, -12, 0, 12, 32767}) {
         agx_apple9_vir_instr I = {};
         I.op = store ? AGX_APPLE9_VIR_DEVICE_STORE : AGX_APPLE9_VIR_DEVICE_LOAD;
         I.encoding = store ? AGX_APPLE9_ENC_DEVICE_STORE : AGX_APPLE9_ENC_DEVICE_LOAD;
         I.dest = store ? AGX_APPLE9_VREG_INVALID : 0;
         I.dest_components = components; I.memory_components = components; I.memory_bits = 32;
         I.memory_index_shift = shift; I.memory_offset = offset;
         I.nr_srcs = store ? components + 1 : 1;
         if (store) for (unsigned s = 0; s <= components; ++s) I.src[s] = s;
         else I.src[0] = 4;
         I.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
         I.device_load_raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101;
         I.device_load_index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR;
         const uint8_t phys[] = {60, 61, 62, 63, 64};
         agx_apple9_packed_instruction packed; const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason)) << reason;
         unsigned base = store ? 73 : 75;
         EXPECT_EQ((int16_t)apple9_test_bits(packed.bytes, base + 2, 16), offset);
         unsigned encoded_shift = apple9_test_bits(packed.bytes, base + 22, 3);
         EXPECT_EQ(encoded_shift ? encoded_shift - 1 : 4, shift);
         I.memory_index_shift = 5;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, &reason));
      }
   }
}

TEST(Apple9Compiler, ApiArrayAddressingUsesScaleAndDisplacementForVectors)
{
   for (unsigned components : {2u, 3u, 4u}) {
      for (unsigned form = 0; form < 6; ++form) {
         SCOPED_TRACE(components);
         SCOPED_TRACE(form);
         nir_builder b = apple9_compute_builder("array_address_fields");
         b.shader->info.num_ssbos = 2;
         nir_def *gid = apple9_global_id_x(&b);
         nir_def *index = nir_iand_imm(&b, gid, form == 5 ? 8191 : 1023);
         nir_def *word = form == 5
            ? nir_iadd_imm(&b, nir_ior_imm(&b, index, 4096), -8)
            : nir_iadd_imm(&b, nir_ishl_imm(&b, index, form), 1);
         /* The API frontend uses amul for the array's element size. */
         nir_def *offset = nir_amul(&b, word, nir_imm_int(&b, 4));
         nir_def *value = nir_load_ssbo(&b, components, 32, nir_imm_int(&b, 0), offset,
                                        .align_mul = 4);
         nir_store_ssbo(&b, value, nir_imm_int(&b, 1), nir_ishl_imm(&b, gid, 4),
                       .align_mul = 4);
         agx_shader_part compiled = {}; agx_apple9_compute_profile profile = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason)) << reason;
         unsigned loads[4];
         ASSERT_EQ(apple9_binary_device_load_offsets(&compiled, loads, 4), 1u);
         const uint8_t *bytes = static_cast<const uint8_t *>(compiled.binary) + loads[0];
         EXPECT_EQ((int16_t)apple9_test_bits(bytes, 77, 16), form == 5 ? -32 : 4);
         unsigned encoded = apple9_test_bits(bytes, 97, 3);
         EXPECT_EQ(encoded ? encoded - 1 : 4, form == 5 || form > 2 ? 2 : form + 2);
         EXPECT_EQ(bytes[8] & 14, components == 2 ? 8 : components == 3 ? 12 : 6);
         free(compiled.binary); ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, NativeIntegerUnarySurvivesMemoryLegalization)
{
   for (nir_op op : {nir_op_bit_count, nir_op_ufind_msb, nir_op_bitfield_reverse}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "integer unary");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_def *offset = nir_imul_imm(&b,
         nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0), 4);
      nir_def *input = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 0), offset,
                                     .align_mul = 4);
      nir_def *value = nir_build_alu(&b, op, input, nullptr, nullptr, nullptr);
      nir_store_ssbo(&b, value, nir_imm_int(&b, 1), offset,
                     .write_mask = 1, .align_mul = 4);
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, nullptr, &reason))
         << reason;
      unsigned count = 0;
      auto *bytes = static_cast<const uint8_t *>(compiled.binary);
      for (unsigned i = 0; i + 8 <= compiled.info.binary_size; ++i) {
         if (bytes[i] == (op == nir_op_bit_count ? 0x27 : 0xa7) &&
             (bytes[i + 1] & 0xf) == (op == nir_op_bitfield_reverse ? 4 : 5) &&
             (bytes[i + 2] & ~3u) == 0x54 && bytes[i + 4] == 2 &&
             bytes[i + 7] == 4)
            count++;
      }
      EXPECT_EQ(count, 1u) << nir_op_infos[op].name;
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Compiler, NativeHalfOperationsSurviveMemoryLegalization)
{
   for (nir_op op : {nir_op_fadd, nir_op_fsub, nir_op_fmul, nir_op_ffma,
                     nir_op_fmin, nir_op_fmax, nir_op_fsat, nir_op_frcp,
                     nir_op_frsq, nir_op_fsqrt, nir_op_fexp2, nir_op_flog2,
                     nir_op_ffloor, nir_op_fceil, nir_op_ftrunc,
                     nir_op_fround_even, nir_op_flt, nir_op_fge,
                     nir_op_feq, nir_op_fneu}) {
      SCOPED_TRACE(nir_op_infos[op].name);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "native half operations");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_def *offset = nir_imul_imm(&b,
         nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0), 4);
      nir_def *inputs[3];
      for (unsigned i = 0; i < 3; ++i)
         inputs[i] = nir_load_ssbo(&b, 1, 16, nir_imm_int(&b, i), offset,
                                    .align_mul = 4);
      nir_def *value = nir_build_alu(&b, op, inputs[0], inputs[1], inputs[2], nullptr);
      value = value->bit_size == 1 ? nir_b2i32(&b, value) : nir_f2f32(&b, value);
      nir_store_ssbo(&b, value, nir_imm_int(&b, 3), offset,
                     .write_mask = 1, .align_mul = 4);
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, nullptr, &reason))
         << reason;
      /* Loading FP16 data must reach native arithmetic directly. Widening
       * all inputs used to leave no 16-bit operation in this program. */
      const auto *bytes = static_cast<const uint8_t *>(compiled.binary);
      unsigned native = 0;
      for (unsigned i = 0; i + 6 <= compiled.info.main_size; i += 2) {
         if ((bytes[i] & 7) == 0 &&
             ((bytes[i + 2] & 7) == 4 || (bytes[i + 2] & 7) == 5 ||
              (bytes[i + 2] & 7) == 6))
            native++;
         if ((bytes[i] & 7) == 2 && !(bytes[i + 1] & 1) &&
             !(bytes[i + 3] & 1))
            native++;
         if (i + 10 <= compiled.info.main_size &&
             (bytes[i] & 0x7f) == 0x2f &&
             ((bytes[i + 6] & 0x18) == 8 ||
              (bytes[i + 1] & 0xf) == 0))
            native++;
      }
      EXPECT_GT(native, 0u);
      free(compiled.binary);
      ralloc_free(b.shader);
   }
}

TEST(Apple9Packer, HalfArithmeticKeepsRegisterExtensionsSeparateFromLifetime)
{
   for (agx_apple9_vir_opcode op : {AGX_APPLE9_VIR_HADD, AGX_APPLE9_VIR_HSUB,
                                    AGX_APPLE9_VIR_HMUL, AGX_APPLE9_VIR_HFMA,
                                    AGX_APPLE9_VIR_HMUL_MIXED}) {
      SCOPED_TRACE(op);
      bool fma = op == AGX_APPLE9_VIR_HFMA;
      unsigned count = fma ? 3 : 2;
      for (unsigned live = 0; live < (1u << count); ++live) {
         for (unsigned slot = 0; slot <= 6; ++slot) {
            agx_apple9_vir_instr I = {};
            I.op = op;
            I.encoding = fma ? AGX_APPLE9_ENC_HALF3 : AGX_APPLE9_ENC_HALF2;
            I.dest = 0;
            I.nr_srcs = count;
            for (unsigned i = 0; i < count; ++i)
               I.src[i] = i + 1;
            I.live_after_mask = live;
            I.scoreboard_slot = static_cast<agx_apple9_scoreboard_slot>(slot);
            uint8_t phys[] = {31, 30, 29, 28};
            agx_apple9_packed_instruction low, high;
            const char *reason = nullptr;
            ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &low, &reason))
               << reason;
            /* These independent bits are documented by the public FP16
             * encoding work and tested on T8132 with 80 live values. */
            const unsigned high_bits[] = {fma ? 60u : 44u, fma ? 56u : 40u,
                                           fma ? 58u : 42u, 38u};
            for (unsigned i = 0; i <= count; ++i) {
               phys[i] += 64;
               ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &high, &reason));
               for (unsigned byte = 0; byte < low.length; ++byte)
                  EXPECT_EQ(high.bytes[byte], low.bytes[byte] ^
                     (byte == high_bits[i] / 8 ? 1 << (high_bits[i] % 8) : 0));
               phys[i] = 96;
               EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &high, &reason));
               phys[i] = 31 - i;
            }
         }
      }
   }
}
