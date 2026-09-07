/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_ir.h"
#include "agx_apple9_machine.h"
#include "agx_compile_apple9.h"

#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>

TEST(Apple9Machine, PhysicalModel)
{
   EXPECT_EQ(agx_apple9_machine.gpr_count, 96u);
   EXPECT_EQ(agx_apple9_machine.half_register_count, 192u);
   EXPECT_TRUE(agx_apple9_machine.hardware_register_interlocks);
   EXPECT_FALSE(agx_apple9_machine.software_waits);
   EXPECT_FALSE(agx_apple9_machine.spilling_supported);
   EXPECT_EQ(agx_apple9_machine.occupancy_model,
             AGX_APPLE9_OCCUPANCY_PRESSURE_TIER);
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
   EXPECT_FALSE(extended->allocator_safe);
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
      AGX_APPLE9_ENC_GET_SR, AGX_APPLE9_OPERAND_DEST, 15, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_GET_SR, AGX_APPLE9_OPERAND_DEST, 16, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_DEVICE_LOAD, AGX_APPLE9_OPERAND_DEST, 63, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_DEVICE_LOAD, AGX_APPLE9_OPERAND_DEST, 64, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_DEVICE_LOAD, AGX_APPLE9_OPERAND_DEST, 96, 32));
   EXPECT_EQ(
      agx_apple9_encoding_info(AGX_APPLE9_ENC_DEVICE_LOAD)->operand_count, 2u);
   EXPECT_NE(agx_apple9_find_operand(AGX_APPLE9_ENC_DEVICE_LOAD,
                                     AGX_APPLE9_OPERAND_INDEX),
             nullptr);
}

TEST(Apple9Machine, ReciprocalHasAsymmetricRegisterFiles)
{
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_OPERAND_DEST, 95, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_OPERAND_DEST, 96, 32));
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_OPERAND_SRC0, 63, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_FLOAT_SPECIAL, AGX_APPLE9_OPERAND_SRC0, 64, 32));
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
   for (unsigned cf = 0; cf <= 12; ++cf) {
      project.immediate = cf;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(&project, phys, &packed, &reason));
      EXPECT_EQ(packed.bytes[5] & 15, cf);
   }
   project.immediate = 13;
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
         AGX_APPLE9_DEVICE_LOAD_HAS_NEXT, false, test.token, &packed));
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
      2, AGX_APPLE9_GPR_COUNT, 0, AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR, 0,
      false, AGX_APPLE9_DEVICE_LOAD_TOKEN_5101, &packed));
   EXPECT_FALSE(agx_apple9_pack_device_load_u32_raw(
      2, 1, 0, AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR, 0, false, 0x7100,
      &packed));
}

TEST(Apple9Vir, DeviceLoadDirectIndexIsAnSsaSource)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t index = agx_apple9_vir_input(&program, 1);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR,
      .flags = AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t load =
      agx_apple9_vir_emit_device_load(&program, 0, index, &contract);
   ASSERT_NE(load, AGX_APPLE9_VREG_INVALID);
   ASSERT_EQ(program.instruction_count, 1u);
   EXPECT_EQ(program.instructions[0].encoding, AGX_APPLE9_ENC_DEVICE_LOAD);
   ASSERT_EQ(program.instructions[0].nr_srcs, 1u);
   EXPECT_EQ(program.instructions[0].src[0], index);
   EXPECT_EQ(program.instructions[0].device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_5101);

   ASSERT_TRUE(agx_apple9_vir_set_fixed_phys(&program, index, 1));
   ASSERT_TRUE(agx_apple9_vir_set_fixed_phys(&program, load, 0));
   ASSERT_TRUE(agx_apple9_vir_add_live_out(&program, index));
   program.output = load;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&program.instructions[0],
                                               program.phys, &packed, &reason))
      << (reason ? reason : "");
   static const uint8_t expected[] = {
      0x67, 0x10, 0x44, 0x00, 0x00, 0x01, 0x20,
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
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(&program.instructions[0], phys,
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

TEST(Apple9Vir, DeviceLoadIndexLifetimeAssertionsFollowLiveness)
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
      EXPECT_FALSE(agx_apple9_allocate_vir(&program, &reason));
      ASSERT_NE(reason, nullptr);
      EXPECT_NE(strstr(reason, "lifetime"), nullptr) << reason;
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Packer, DeviceLoadSeparatesAddressSequenceFlagsFromScoreboardSlot)
{
   static const struct {
      uint8_t flags;
      enum agx_apple9_scoreboard_slot slot;
      uint8_t bytes[14];
   } cases[] = {
      {0,
       AGX_APPLE9_SCOREBOARD_SLOT_1,
       {0x67, 0x00, 0x44, 0x04, 0x03, 0x01, 0x20, 0x00, 0x11, 0x00, 0x00, 0x40,
        0x46, 0x00}},
      {0,
       AGX_APPLE9_SCOREBOARD_SLOT_2,
       {0x67, 0x00, 0x44, 0x04, 0x03, 0x01, 0x20, 0x00, 0x51, 0x00, 0x00, 0x40,
        0x46, 0x00}},
      {0,
       AGX_APPLE9_SCOREBOARD_SLOT_6,
       {0x67, 0x00, 0x44, 0x04, 0x03, 0x01, 0x20, 0x00, 0x51, 0x01, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX,
       AGX_APPLE9_SCOREBOARD_SLOT_1,
       {0x67, 0x10, 0x44, 0x04, 0x03, 0x01, 0x20, 0x00, 0x11, 0x00, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX,
       AGX_APPLE9_SCOREBOARD_SLOT_2,
       {0x67, 0x10, 0x44, 0x04, 0x03, 0x01, 0x20, 0x00, 0x51, 0x00, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX,
       AGX_APPLE9_SCOREBOARD_SLOT_6,
       {0x67, 0x10, 0x44, 0x04, 0x03, 0x01, 0x20, 0x00, 0x51, 0x01, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
       AGX_APPLE9_SCOREBOARD_SLOT_1,
       {0x67, 0x00, 0x54, 0x04, 0x03, 0x01, 0x20, 0x00, 0x11, 0x00, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
       AGX_APPLE9_SCOREBOARD_SLOT_2,
       {0x67, 0x00, 0x54, 0x04, 0x03, 0x01, 0x20, 0x00, 0x51, 0x00, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
       AGX_APPLE9_SCOREBOARD_SLOT_6,
       {0x67, 0x00, 0x54, 0x04, 0x03, 0x01, 0x20, 0x00, 0x51, 0x01, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX |
          AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
       AGX_APPLE9_SCOREBOARD_SLOT_1,
       {0x67, 0x10, 0x54, 0x04, 0x03, 0x01, 0x20, 0x00, 0x11, 0x00, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX |
          AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
       AGX_APPLE9_SCOREBOARD_SLOT_2,
       {0x67, 0x10, 0x54, 0x04, 0x03, 0x01, 0x20, 0x00, 0x51, 0x00, 0x00, 0x40,
        0x46, 0x00}},
      {AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX |
          AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
       AGX_APPLE9_SCOREBOARD_SLOT_6,
       {0x67, 0x10, 0x54, 0x04, 0x03, 0x01, 0x20, 0x00, 0x51, 0x01, 0x00, 0x40,
        0x46, 0x00}},
   };

   agx_apple9_packed_instruction packed = {};
   for (const auto &test : cases) {
      ASSERT_TRUE(agx_apple9_pack_device_load_u32(2, 1, 3, test.flags,
                                                  test.slot, &packed));
      ASSERT_EQ(packed.length, sizeof(test.bytes));
      EXPECT_EQ(memcmp(packed.bytes, test.bytes, sizeof(test.bytes)), 0)
         << "flags=" << unsigned(test.flags) << " slot=" << test.slot;
   }

   EXPECT_FALSE(agx_apple9_pack_device_load_u32(
      2, 1, 3, 0x80, AGX_APPLE9_SCOREBOARD_SLOT_6, &packed));
   EXPECT_FALSE(agx_apple9_pack_device_load_u32(
      2, 1, 3, 0, AGX_APPLE9_SCOREBOARD_SLOT_NONE, &packed));
   EXPECT_FALSE(agx_apple9_pack_device_load_u32(
      2, 1, 3, 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO, &packed));
}

TEST(Apple9Packer, NativeVectorMemoryWidthsMatchValidatedEncodings)
{
   struct vector_case {
      unsigned components;
      unsigned dst;
      unsigned index;
      uint8_t flags;
      uint8_t load[14];
      uint8_t store_load[14];
      uint8_t store_alu[14];
   };
   static const vector_case cases[] = {
      {2,
       0,
       2,
       AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX,
       {0x67, 0x10, 0x44, 0x00, 0x00, 0x02, 0x20, 0x00, 0x59, 0x01, 0x00, 0x40,
        0x48, 0x00},
       {0xe7, 0x00, 0x56, 0x00, 0x01, 0x02, 0x21, 0x00, 0x19, 0x00, 0x00, 0x10,
        0x12, 0x00},
       {0xe7, 0x00, 0x54, 0x00, 0x01, 0x02, 0x21, 0x00, 0x19, 0x00, 0x00, 0x10,
        0x12, 0x00}},
      {3,
       4,
       7,
       AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
       {0x67, 0x00, 0x54, 0x08, 0x00, 0x07, 0x20, 0x00, 0x5d, 0x01, 0x00, 0x40,
        0x40, 0x00},
       {0xe7, 0x00, 0x56, 0x00, 0x01, 0x07, 0x21, 0x00, 0x1d, 0x00, 0x00, 0x10,
        0x10, 0x00},
       {0xe7, 0x00, 0x54, 0x00, 0x01, 0x07, 0x21, 0x00, 0x1d, 0x00, 0x00, 0x10,
        0x10, 0x00}},
      {4,
       0,
       4,
       AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX,
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
         AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR, test.flags, false,
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
      AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX, false,
      AGX_APPLE9_DEVICE_LOAD_TOKEN_5101, &packed));
   static const uint8_t load_u8[] = {
      0x67, 0x10, 0x44, 0x02, 0x01, 0x00, 0x20,
      0x00, 0x61, 0x01, 0x00, 0x40, 0x42, 0x00,
   };
   EXPECT_EQ(packed.length, sizeof(load_u8));
   EXPECT_EQ(memcmp(packed.bytes, load_u8, sizeof(load_u8)), 0);

   ASSERT_TRUE(agx_apple9_pack_device_load_scalar_raw(
      1, 0, 2, 16, AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX | AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
      false, AGX_APPLE9_DEVICE_LOAD_TOKEN_5101, &packed));
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
      1, 0, 0, 16, AGX_APPLE9_SCOREBOARD_SLOT_NONE, true, &packed));
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

TEST(Apple9Packer, DeviceAtomicResultEncodesSixBitDestinationAndAllSlots)
{
   static const uint8_t publication_code[] = {
      0, 2, 4, 3, 5, 6, 1,
   };

   for (unsigned destination = 0; destination < 64; ++destination) {
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
         EXPECT_EQ(packed.bytes[2],
                   0x09 | ((destination >> 4) << 6));
         EXPECT_EQ(packed.bytes[5], publication_code[slot] << 5);
         EXPECT_EQ(packed.bytes[6], 0u);
      }
   }

   agx_apple9_vir_instr invalid = {};
   invalid.op = AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT;
   invalid.encoding = AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT;
   invalid.dest = AGX_APPLE9_VREG_INVALID;
   invalid.src[0] = 0;
   invalid.nr_srcs = 1;
   invalid.producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
   const uint8_t phys[] = {64};
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
   EXPECT_FALSE(agx_apple9_pack_get_sr_zext16(16, 0xa4, &packed));

   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_GET_SR_ZEXT16, AGX_APPLE9_OPERAND_DEST, 15, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_GET_SR_ZEXT16, AGX_APPLE9_OPERAND_DEST, 16, 32));
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
                                 0x44, 0xb4, 0x08, 0x03, 0x00};
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
      AGX_APPLE9_ENC_SHIFT_EXTENDED, AGX_APPLE9_OPERAND_DEST, 15, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      AGX_APPLE9_ENC_SHIFT_EXTENDED, AGX_APPLE9_OPERAND_DEST, 16, 32));
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

TEST(Apple9Machine, WideSelectHasAsymmetricReach)
{
   const auto encoding = AGX_APPLE9_ENC_SELECT_GPR_WIDE;
   EXPECT_TRUE(agx_apple9_encoding_info(encoding)->allocator_safe);
   EXPECT_TRUE(agx_apple9_encoding_accepts_gpr(
      encoding, AGX_APPLE9_OPERAND_DEST, 15, 32));
   EXPECT_FALSE(agx_apple9_encoding_accepts_gpr(
      encoding, AGX_APPLE9_OPERAND_DEST, 16, 32));

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
   EXPECT_FALSE(
      agx_apple9_encoding_accepts_gpr_tuple(extended, all_high, 3, 32));
}

TEST(Apple9Machine, StoreDataAndIndexAreAllocatable)
{
   const auto *data = agx_apple9_find_operand(AGX_APPLE9_ENC_DEVICE_STORE,
                                              AGX_APPLE9_OPERAND_STORE_DATA);
   ASSERT_NE(data, nullptr);
   EXPECT_EQ(data->files, AGX_APPLE9_FILE_GPR);
   EXPECT_TRUE(data->flags & AGX_APPLE9_OPERAND_ALLOCATABLE);
   EXPECT_EQ(data->max_index, 63);
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
      {AGX_APPLE9_ENC_DEVICE_LOAD, AGX_APPLE9_DEPENDENCY_NONE},
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
      0xf2, 0xbf, 0x67, 0x81, 0x02, 0xfe, 0x05, 0xd5, 0x40, 0xbc,
   };
   ASSERT_EQ(packed.length, sizeof(expected_select));
   EXPECT_EQ(memcmp(packed.bytes, expected_select, sizeof(expected_select)), 0);
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
      &program, load, AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX,
      AGX_APPLE9_SCOREBOARD_SLOT_AUTO));
   uint32_t sources[] = {load, ordinary};
   agx_apple9_vir_emit(&program, AGX_APPLE9_VIR_FADD,
                       AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   EXPECT_EQ(program.instructions[0].producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[0].device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_5101);
   EXPECT_EQ(program.instructions[1].scoreboard_slot,
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
      if (program.instructions[i].op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         EXPECT_EQ(program.instructions[i].producer_scoreboard_slot,
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
      if (program.instructions[i].op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         EXPECT_EQ(program.instructions[i].producer_scoreboard_slot,
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
   program.instructions[6].device_load_index_kind =
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
      if (program.instructions[i].op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         EXPECT_EQ(program.instructions[i].producer_scoreboard_slot,
                   expected[seen++]);
      }
      materialized += program.instructions[i].scoreboard_materialize;
   }
   EXPECT_EQ(seen, 7u);
   EXPECT_EQ(materialized, 1u);
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   bool general_materialization = false;
   for (unsigned i = 0; i < program.instruction_count; ++i) {
      if (program.instructions[i].scoreboard_materialize) {
         EXPECT_EQ(program.instructions[i].encoding,
                   AGX_APPLE9_ENC_LOGIC_EXTENDED);
         general_materialization |=
            program.phys[program.instructions[i].dest] >= 16;
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
      const auto &instruction = program.instructions[i];
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
   EXPECT_EQ(program.instructions[0].producer_scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[1].producer_scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[0].device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_5101);
   EXPECT_EQ(program.instructions[1].device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_5101);
   EXPECT_EQ(program.instructions[2].scoreboard_slot, 6u);
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
   EXPECT_EQ(program.instructions[1].scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[2].scoreboard_slot, 0u);
   EXPECT_NE(program.instructions[1].live_after_mask & 1u, 0u);
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
   EXPECT_EQ(program.instructions[0].device_load_raw_token,
             AGX_APPLE9_DEVICE_LOAD_TOKEN_9100);
   EXPECT_EQ(program.instructions[0].producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_3);
   EXPECT_EQ(program.instructions[1].scoreboard_slot,
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
   EXPECT_EQ(program.instructions[0].producer_scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[1].scoreboard_slot, 6u);
   EXPECT_EQ(program.instructions[1].op, AGX_APPLE9_VIR_IADD);
   EXPECT_EQ(program.instructions[1].src[0], load);
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
      EXPECT_EQ(program.instructions[0].producer_scoreboard_slot, 6u);
      EXPECT_EQ(program.instructions[1].producer_scoreboard_slot, 6u);
      EXPECT_EQ(program.instructions[2].scoreboard_slot, 6u);
      EXPECT_FALSE(program.instructions[2].scoreboard_materialize);
      agx_apple9_vir_finish(&program);
   }
}

TEST(Apple9Vir, MaskedPhiEdgesShareOneAllocatedMergeDestination)
{
   agx_apple9_vir_program program;
   agx_apple9_vir_init(&program);
   uint32_t then_value = agx_apple9_vir_input(&program, 2);
   uint32_t else_value = agx_apple9_vir_input(&program, 3);
   uint32_t merge = agx_apple9_vir_emit_merge(&program);
   ASSERT_NE(merge, AGX_APPLE9_VREG_INVALID);
   ASSERT_TRUE(agx_apple9_vir_emit_masked_copy(&program, merge, then_value));
   ASSERT_TRUE(agx_apple9_vir_emit_masked_copy(&program, merge, else_value));
   program.output = merge;

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");

   ASSERT_EQ(program.instruction_count, 3u);
   EXPECT_EQ(program.instructions[0].op, AGX_APPLE9_VIR_MERGE);
   for (unsigned i = 1; i < 3; ++i) {
      EXPECT_EQ(program.instructions[i].op, AGX_APPLE9_VIR_MASKED_COPY);
      EXPECT_EQ(program.instructions[i].target, merge);
      EXPECT_EQ(program.phys[program.instructions[i].target],
                program.phys[merge]);
   }
   EXPECT_NE(program.phys[merge], program.phys[then_value]);
   EXPECT_NE(program.phys[merge], program.phys[else_value]);
   agx_apple9_vir_finish(&program);
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
      EXPECT_EQ(program.instructions[0].producer_scoreboard_slot, 6u);
      EXPECT_EQ(program.instructions[1].scoreboard_slot, 6u);
      EXPECT_EQ(program.instructions[1].src[0], load);
      EXPECT_EQ(program.instructions[1].src[1], ordinary);
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
      collect_index = program.instructions[i].op == AGX_APPLE9_VIR_COLLECT
                         ? i
                         : collect_index;
      store_index = program.instructions[i].op == AGX_APPLE9_VIR_DEVICE_STORE
                       ? i
                       : store_index;
   }
   ASSERT_EQ(collect_index, UINT_MAX);
   ASSERT_NE(store_index, UINT_MAX);
   EXPECT_EQ(program.instructions[store_index].scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_1);
   for (unsigned c = 0; c < 4; ++c)
      EXPECT_EQ(program.instructions[store_index].src[c], vector + c);

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
   EXPECT_EQ(program.instructions[0].producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[1].op,
             AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
   EXPECT_EQ(program.instructions[1].scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_NONE);
   EXPECT_EQ(program.instructions[2].scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);

   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   agx_apple9_packed_instruction atomic = {};
   agx_apple9_packed_instruction materialize = {};
   agx_apple9_packed_instruction copy = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      &program.instructions[0], program.phys, &atomic, &reason));
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      &program.instructions[1], program.phys, &materialize, &reason));
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      &program.instructions[2], program.phys, &copy, &reason));
   EXPECT_EQ(atomic.bytes[0], 0x67);
   EXPECT_EQ(atomic.bytes[1], 0x01);
   EXPECT_EQ(atomic.bytes[2], 0x54);
   EXPECT_EQ(atomic.bytes[9], 0x02);
   ASSERT_EQ(materialize.length, 8u);
   EXPECT_EQ(materialize.bytes[0] & 0x0f, 0x0c);
   EXPECT_EQ(materialize.bytes[0] >> 4,
             program.phys[program.instructions[0].dest]);
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
   EXPECT_EQ(program.instructions[0].producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[1].scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[1].producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[2].producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[3].scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);

   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      &program.instructions[1], program.phys, &packed, &reason));
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
      EXPECT_EQ(program.instructions[load_instruction[i]]
                   .producer_scoreboard_slot,
                expected_slots[i]);
      EXPECT_EQ(program.instructions[atomic_instruction[i]].scoreboard_slot,
                expected_slots[i]);
      EXPECT_EQ(program.instructions[atomic_instruction[i]]
                   .producer_scoreboard_slot,
                expected_slots[i]);
      EXPECT_EQ(program.instructions[publication_instruction[i]]
                   .producer_scoreboard_slot,
                expected_slots[i]);
      EXPECT_EQ(program.instructions[consumer_instruction[i]].scoreboard_slot,
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
   EXPECT_EQ(program.instructions[0].op, AGX_APPLE9_VIR_DEVICE_ATOMIC);
   EXPECT_EQ(program.instructions[0].dest, AGX_APPLE9_VREG_INVALID);
   EXPECT_TRUE(program.instructions[0].atomic_discard);

   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&program, &reason))
      << (reason ? reason : "");
   ASSERT_TRUE(agx_apple9_allocate_vir(&program, &reason))
      << (reason ? reason : "");
   agx_apple9_packed_instruction packed = {};
   ASSERT_TRUE(agx_apple9_pack_vir_instruction(
      &program.instructions[0], program.phys, &packed, &reason))
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
      const agx_apple9_vir_instr &instruction = program.instructions[i];
      if (instruction.op != AGX_APPLE9_VIR_DEVICE_ATOMIC)
         continue;

      ASSERT_LT(atomic_index, ARRAY_SIZE(expected_slots));
      EXPECT_EQ(instruction.scoreboard_slot,
                AGX_APPLE9_SCOREBOARD_SLOT_NONE);
      EXPECT_EQ(instruction.producer_scoreboard_slot,
                expected_slots[atomic_index]);
      ASSERT_LT(i + 1, program.instruction_count);
      EXPECT_EQ(program.instructions[i + 1].op,
                AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
      EXPECT_EQ(program.instructions[i + 1].producer_scoreboard_slot,
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
      materializations += program.instructions[i].scoreboard_materialize;
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
   ASSERT_EQ(program.instructions[0].op, AGX_APPLE9_VIR_COLLECT);
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
   const agx_apple9_vir_instr &atomic = program.instructions[1];
   EXPECT_EQ(program.phys[atomic.src[1]], program.phys[atomic.src[0]] + 1);
   EXPECT_NE(program.phys[atomic.dest], program.phys[atomic.src[0]]);
   EXPECT_EQ(program.instructions[2].op,
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
   EXPECT_EQ(program.phys[gid], 0u);
   EXPECT_EQ(program.phys[c], 1u);
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
      .flags = AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX,
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
   EXPECT_EQ(program.instructions[0].producer_scoreboard_slot,
             AGX_APPLE9_SCOREBOARD_SLOT_6);
   EXPECT_EQ(program.instructions[1].scoreboard_slot,
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
   const uint32_t tuple = program.instructions[0].dest;
   EXPECT_EQ(program.instructions[0].op, AGX_APPLE9_VIR_COLLECT);
   EXPECT_EQ(program.instructions[0].dest_components, 4u);
   EXPECT_EQ(program.instructions[1].op, AGX_APPLE9_VIR_DEVICE_STORE);
   for (unsigned c = 0; c < 4; ++c)
      EXPECT_EQ(program.instructions[1].src[c], tuple + c);

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

   const auto &first_store = program.instructions[0];
   const auto &last_store = program.instructions[2];
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
   const uint32_t tuple = program.instructions[4].dest;

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
   const uint32_t tuple = program.instructions[0].dest;

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
   for (unsigned i = 0; i < depth; ++i)
      nested[i] = nir_push_if(&b, nir_ult_imm(&b, gid, 32 - i));

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
          bytes[11] != 0x40)
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
apple9_num_workgroups_shader(bool variable_local_size, unsigned component)
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
   apple9_store_output(&b, apple9_global_id_x(&b),
                       nir_channel(&b, groups, component));
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

TEST(Apple9Compiler, ConstantStoreUsesGenericPipeline)
{
   apple9_expect_compile(apple9_constant_store_shader(42),
                         AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_ATOMIC);
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
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_ATOMIC);
   EXPECT_EQ(apple9_binary_count_atomics(&compiled, AGX_APPLE9_ATOMIC_ADD,
                                         false),
             1u);
   EXPECT_EQ(apple9_binary_count_atomics(&compiled, AGX_APPLE9_ATOMIC_XOR,
                                         false),
             1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, FinalUsePendingLoadFeedsReturningAtomicDirectly)
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

TEST(Apple9Compiler, AtomicDoesNotDrainUnrelatedPendingLoads)
{
   nir_shader *nir = apple9_pending_load_atomic_shader(false, true);
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_NE(apple9_binary_first_atomic_dependency(
                &compiled, AGX_APPLE9_ATOMIC_ADD),
             0u);
   EXPECT_EQ(apple9_binary_count_pending_stores(&compiled), 2u);
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
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(apple9_binary_count_atomics(&compiled, AGX_APPLE9_ATOMIC_XOR,
                                         true),
             1u);
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
      {APPLE9_REGION_EMPTY, 0, 0, 2},
      {APPLE9_REGION_THEN_ONLY, 1, 0, 3},
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

TEST(Apple9Compiler, MultipleScalarAndVectorPhisUseMaskedEdgeCopies)
{
   nir_shader *nir = apple9_multiple_phi_shader();
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

TEST(Apple9Compiler, SimplePhiUsesMaskedEdgeCopies)
{
   nir_shader *nir = apple9_simple_phi_shader();
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

   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned selects = 0;
   for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i)
      selects += binary[i] == 0x02 && binary[i + 4] == 0x82;
   /* CFG phis are resolved by masked predecessor-edge copies. Explicit bcsel
    * remains a separate instruction-selection path, but this shader has none. */
   EXPECT_EQ(selects, 0u);
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

TEST(Apple9Compiler, NestedVectorPhisResolveAtEachReconvergence)
{
   nir_shader *nir = apple9_nested_phi_shader();
   struct agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, nullptr, &reason))
      << (reason ? reason : "no diagnostic");

   const uint8_t else_mask[] = {0x0f, 0x04, 0x04, 0x19};
   const uint8_t pop[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   EXPECT_EQ(apple9_binary_count_exec_pushes(&compiled), 2u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, else_mask, sizeof(else_mask)),
      2u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, pop, sizeof(pop)), 2u);

   const uint8_t *binary = (const uint8_t *)compiled.binary;
   unsigned selects = 0;
   for (unsigned i = 0; i + 10 <= compiled.info.binary_size; ++i)
      selects += binary[i] == 0x02 && binary[i + 4] == 0x82;
   EXPECT_EQ(selects, 0u);
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
         is_or ? 1u : 0u);
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
      1u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_pop, sizeof(loop_pop)), 1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, break_one_if,
                                          sizeof(break_one_if)),
             0u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, exit_if_none,
                                          sizeof(exit_if_none)),
             1u);

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
      EXPECT_GE(target, 0);
      EXPECT_LT(target, (int64_t)compiled.info.binary_size);
      ++branches;
   }
   EXPECT_EQ(branches, 1u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, NestedLoopsUseIndependentMaskDepthAndBreakTargets)
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
   const uint8_t inner_break[] = {0x8f, 0x05, 0x54, 0x03, 0x00, 0x02};
   const uint8_t exit_if_none[] = {0x0f, 0x01, 0x54};
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_push, sizeof(loop_push)),
      1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, outer_update,
                                          sizeof(outer_update)),
             1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, inner_update,
                                          sizeof(inner_update)),
             1u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_pop, sizeof(loop_pop)), 2u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, outer_break, sizeof(outer_break)),
      0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, inner_break, sizeof(inner_break)),
      0u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, exit_if_none,
                                          sizeof(exit_if_none)),
             2u);
   free(compiled.binary);
   ralloc_free(nir);
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
      1u);
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

   /* The top-level loop uses the hardware's implicit initial mask.  The two
    * ordinary conditionals have their own kind-1 scopes; the middle break
    * directly updates the loop mask, and the loop ends in one bare backedge
    * plus its kind-2 pop. */
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_push, sizeof(loop_push)),
      0u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_update, sizeof(loop_update)),
      1u);
   EXPECT_EQ(
      apple9_binary_count_sequence(&compiled, loop_pop, sizeof(loop_pop)), 1u);
   EXPECT_EQ(apple9_binary_count_sequence(&compiled, exit_if_none,
                                          sizeof(exit_if_none)),
             1u);
   EXPECT_EQ(apple9_binary_count_exec_pushes(&compiled), 2u);

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
             1u);
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

      /* The entry load consumes get_sr directly. Arm/merge indices pass
       * through ALU and use the ordinary address form. HAS_NEXT follows the
       * complete linear load sequence across PUSH/ELSE/POP. */
      const uint8_t *binary = (const uint8_t *)compiled.binary;
      for (unsigned i = 0; i < test.load_count; ++i) {
         EXPECT_EQ(binary[loads[i] + 1] & 0x10, i == 0 ? 0x10 : 0x00);
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
   EXPECT_GE(literal_dst, 16u);
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
         EXPECT_GE(dst, 16u);
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
      AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET,
      AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET,
      AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET,
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
                         AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
   apple9_expect_compile(apple9_arbitrary_float_shader(),
                         AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
}

TEST(Apple9Compiler, ReciprocalUsesNativeHandoffAndLifetimeForms)
{
   static const struct {
      enum apple9_reciprocal_shape shape;
      uint8_t handoff;
      uint8_t result_hint;
      uint8_t source_lifetime;
   } cases[] = {
      {APPLE9_RECIPROCAL_DIRECT_STORE, 0x56, 0x02, 0x10},
      {APPLE9_RECIPROCAL_RETAIN_SOURCE, 0x56, 0x03, 0x00},
      {APPLE9_RECIPROCAL_MATERIALIZED_SOURCE, 0x54, 0x02, 0x10},
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
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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
            EXPECT_EQ(p[2], shape == 2 ? 0x54 : 0x56);
         }
         EXPECT_EQ(found, 1u);
         free(compiled.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(Apple9Compiler, SaturationLowersVectorComponents)
{
   nir_builder b = apple9_compute_builder("apple9_saturate");
   nir_def *input = nir_imm_vec4(&b, -2.0, -0.0, 0.375, 2.0);
   nir_def *clamped = nir_fsat(&b, input);
   nir_intrinsic_instr *store =
      nir_store_ssbo(&b, clamped, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                     .write_mask = 0xf, .align_mul = 16);
   ASSERT_TRUE(agx_nir_lower_apple9_math(b.shader));
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
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);

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
            EXPECT_EQ(bytes[2], 0x56);
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

   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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
   EXPECT_EQ(stores, 2u);
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
         EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
         ASSERT_EQ(profile.resource_binding_count, 2u);
         EXPECT_EQ(profile.resource_binding[0], 1u);
         EXPECT_EQ(profile.resource_binding[1], 0u);
         EXPECT_EQ(profile.resource_read_mask, 0x2u);
         EXPECT_EQ(profile.resource_write_mask, 0x3u);
      } else {
         EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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
      EXPECT_EQ(access_desc[0], 0x20);
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
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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
         EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);

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
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);

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
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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

   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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

   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
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
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
   EXPECT_EQ(profile.resource_binding_count, 2u);
   EXPECT_EQ(profile.resource_read_mask, 0x1u);
   EXPECT_EQ(profile.resource_write_mask, 0x2u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST(Apple9Compiler, VariableShiftsUseGeneralValidatedLowering)
{
   for (nir_op op : {nir_op_ishl, nir_op_ishr, nir_op_ushr}) {
      nir_shader *nir = apple9_variable_shift_shader(op);
      struct agx_shader_part compiled = {};
      struct agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
      EXPECT_GT(compiled.info.binary_size, 100u);
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
      AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);
}

TEST(Apple9Compiler, NumWorkgroupsUsesRuntimeCeilingDivision)
{
   for (bool variable : {false, true}) {
      for (unsigned component = 0; component < 3; ++component) {
         SCOPED_TRACE(testing::Message()
                      << "variable=" << variable << " component=" << component);
         nir_shader *nir = apple9_num_workgroups_shader(variable, component);
         struct agx_shader_part compiled = {};
         struct agx_apple9_compute_profile profile = {};
         const char *reason = nullptr;
         ASSERT_TRUE(agx_compile_apple9_tiny(nir, &compiled, &profile, &reason))
            << (reason ? reason : "no diagnostic");

         EXPECT_EQ(profile.variable_local_size, variable);
         EXPECT_EQ(apple9_binary_count_reciprocals(&compiled), 1u);
         if (variable) {
            EXPECT_EQ(profile.local_size[component], 0u);
            EXPECT_TRUE(apple9_binary_contains_get_sr_zext16(&compiled,
                                                             0x98 + component));
         } else {
            static const uint32_t expected[] = {7, 5, 3};
            EXPECT_EQ(profile.local_size[component], expected[component]);
         }

         free(compiled.binary);
         ralloc_free(nir);
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
   EXPECT_EQ(profile.abi, AGX_APPLE9_COMPUTE_ABI_SSBO8_SUPERSET);

   bool saw_local_index = false;
   unsigned scalar_loads = 0;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      const uint8_t *bytes = (const uint8_t *)compiled.binary + i;
      saw_local_index |= (bytes[0] & 0xf) == 0x4 && bytes[1] == 0xa7 &&
                         bytes[2] == 0x10 && bytes[3] == 0x06 &&
                         bytes[4] == ((bytes[0] & 0xf0) | 0x03);
      scalar_loads += bytes[0] == 0x67 && (bytes[8] & 0xf8) == 0x50;
   }
   EXPECT_TRUE(saw_local_index);
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
   const uint8_t bad_dest[] = {16, 16, 17};
   EXPECT_FALSE(
      agx_apple9_pack_vir_instruction(&mul, bad_dest, &packed, &reason));
   const uint8_t bad_source[] = {7, 64, 17};
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
   unsigned registers = 0, slots = 0, count = 0;
   bool vertex_handoff = false;
   for (unsigned i = 0; i + 14 <= compiled.info.binary_size; ++i) {
      if ((code[i] & 15) == 12 && code[i + 1] == 0xdd && code[i + 2] == 0x10) {
         vertex_handoff |= (code[i + 9] & 0xe0) == 0x20;
      }
   }
   for (unsigned i = 0; i + 8 <= compiled.info.binary_size; ++i) {
      if (code[i] != 0x57 || code[i + 1] != 6 || code[i + 2] != 0x54)
         continue;
      unsigned reg = code[i + 3] >> 1;
      unsigned slot = code[i + 4] >> 5;
      ASSERT_LT(reg, 16u);
      EXPECT_EQ(registers & (1u << reg), 0u);
      registers |= 1u << reg;
      slots |= 1u << slot;
      ++count;
   }
   EXPECT_TRUE(vertex_handoff);
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

TEST(Apple9Compiler, GraphicsRejectsTooManyBuffersWithoutDiagnosticPointer)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "two_graphics_ubos");
   nir_def *sum = nir_imm_float(&b, 0);
   for (unsigned binding = 0; binding < 5; ++binding) {
      nir_def *value =
         nir_load_ubo(&b, 1, 32, nir_imm_int(&b, binding), nir_imm_int(&b, 0),
                      .align_mul = 4, .range = 4);
      sum = nir_fadd(&b, sum, value);
   }
   nir_store_output(
      &b, nir_vec4(&b, sum, sum, sum, nir_imm_float(&b, 1)), nir_imm_int(&b, 0),
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   b.shader->info.num_ubos = 5;
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
   layout.components[0] = 3; /* A vec4 shader input defaults W to one. */
   layout.stride[1] = 32;
   layout.components[1] = 4;
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
   EXPECT_EQ(vertex.info.apple9_varyings.mask[3], 10u);
   EXPECT_EQ(vertex.info.apple9_varyings.mask[9], 7u);
   EXPECT_EQ(vertex.info.apple9_varyings.mask[31], 8u);
   const uint8_t *code = static_cast<const uint8_t *>(vertex.binary);
   unsigned slots = 0;
   for (unsigned i = 0; i + 8 <= vertex.info.binary_size; ++i) {
      if (code[i] == 0x57 && code[i+1] == 6 && code[i+2] == 0x54)
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
   shifted.mask[0] = 1;
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
   invalid.mask[31] = 0;
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
   for (unsigned count : {0u, 12u, 13u}) {
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
      if (count <= 12) {
         ASSERT_TRUE(ok) << reason;
         EXPECT_EQ(compiled.info.apple9_varyings.count, count);
         free(compiled.binary);
      } else {
         EXPECT_FALSE(ok);
         EXPECT_EQ(compiled.binary, nullptr);
         EXPECT_NE(strstr(reason, "twelve"), nullptr);
      }
      ralloc_free(b.shader);
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
