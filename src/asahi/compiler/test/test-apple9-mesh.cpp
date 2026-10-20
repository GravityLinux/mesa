/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_ir.h"
#include "agx_apple9_mesh_encoding.h"
#include "agx_compile_apple9.h"
extern "C" {
#include "agx_nir.h"
}
#include <array>
#include <set>
#include <vector>
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>

/* The 58 unique singleton encodings in UNITS-DECODE-20261018. Sorted by
 * bytes, not execution order. No ordinary code or helper span is included. */
static const uint8_t census[][6] = {
   {0x57, 0x03, 0x54, 0x01, 0x0c, 0x30}, {0x57, 0x03, 0x54, 0x01, 0x0c, 0x34},
   {0x57, 0x03, 0x54, 0x01, 0x1c, 0x30}, {0x57, 0x03, 0x54, 0x01, 0x1c, 0x34},
   {0x57, 0x03, 0x54, 0x06, 0x02, 0x34}, {0x57, 0x03, 0x54, 0x06, 0x0e, 0x34},
   {0x57, 0x03, 0x54, 0x06, 0x14, 0x30}, {0x57, 0x03, 0x54, 0x06, 0x14, 0x34},
   {0x57, 0x03, 0x54, 0x06, 0x1c, 0x30}, {0x57, 0x03, 0x54, 0x07, 0x0c, 0x30},
   {0x57, 0x03, 0x54, 0x07, 0x0c, 0x34}, {0x57, 0x03, 0x54, 0x0a, 0x0c, 0x30},
   {0x57, 0x03, 0x54, 0x0a, 0x0c, 0x34}, {0x57, 0x03, 0x54, 0x0c, 0x1c, 0x34},
   {0x57, 0x03, 0x54, 0x0d, 0x0c, 0x30}, {0x57, 0x03, 0x54, 0x0d, 0x0c, 0x34},
   {0x57, 0x03, 0x54, 0x0e, 0x02, 0x30}, {0x57, 0x03, 0x54, 0x0e, 0x0c, 0x34},
   {0xd7, 0x06, 0x54, 0x05, 0x06, 0x00}, {0xd7, 0x06, 0x54, 0x05, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x06, 0x06, 0x00}, {0xd7, 0x06, 0x54, 0x07, 0x06, 0x00},
   {0xd7, 0x06, 0x54, 0x85, 0x06, 0x00}, {0xd7, 0x06, 0x54, 0x85, 0x0e, 0x00},
   {0xd7, 0x06, 0x54, 0x85, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x86, 0x0e, 0x00},
   {0xd7, 0x06, 0x54, 0x87, 0x06, 0x00}, {0xd7, 0x06, 0x54, 0x87, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x88, 0x0e, 0x00}, {0xd7, 0x06, 0x54, 0x88, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x89, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x8a, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x8b, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x8c, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x8d, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x8e, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x8f, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x90, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x91, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x92, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x93, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x94, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x95, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x96, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x97, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x98, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x99, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x9a, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x9b, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0x9c, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0x9e, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0xa0, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0xa2, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0xa4, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0xa6, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0xa8, 0x16, 0x00},
   {0xd7, 0x06, 0x54, 0xaa, 0x16, 0x00}, {0xd7, 0x06, 0x54, 0xac, 0x16, 0x00},
};

TEST(Apple9MeshEncoding, All58CensusSingletonsRoundTrip)
{
   static_assert(ARRAY_SIZE(census) == 58);
   unsigned control = 0, submit = 0;
   for (const auto &bytes : census) {
      agx_apple9_mesh_instruction ins = {};
      ASSERT_TRUE(agx_apple9_decode_mesh(bytes, &ins));
      uint8_t encoded[6] = {};
      ASSERT_TRUE(agx_apple9_encode_mesh(encoded, &ins));
      EXPECT_EQ(memcmp(bytes, encoded, 6), 0);
      /* Independently inspect the named field positions. */
      if (ins.kind == AGX_APPLE9_MESH_CONTROL) {
         ++control;
         EXPECT_EQ(2 * ins.count_gpr + ins.count_high, bytes[3]);
         EXPECT_EQ(2 * ins.auxiliary_gpr + ins.auxiliary_high,
                   (bytes[4] >> 1) | ((bytes[5] & 1) << 7));
         EXPECT_EQ(ins.control42, bool(bytes[5] & 4));
      } else {
         ++submit;
         EXPECT_EQ(ins.source_gpr, bytes[3] & 127);
         EXPECT_EQ(ins.release, bool(bytes[3] & 128));
         EXPECT_EQ(unsigned(ins.mode), (bytes[4] >> 3) & 3);
      }
   }
   EXPECT_EQ(control, 18u);
   EXPECT_EQ(submit, 40u);
}

TEST(Apple9MeshEncoding, RejectsUnknownControlsAndOutOfRangeRegisters)
{
   const uint8_t baseline[6] = {0x57, 3, 0x54, 6, 0x14, 0x34};
   for (unsigned bit : {12u, 17u, 23u, 32u, 41u, 43u, 44u, 45u, 46u, 47u}) {
      uint8_t changed[6];
      memcpy(changed, baseline, 6);
      changed[bit / 8] ^= 1u << (bit % 8);
      agx_apple9_mesh_instruction ins = {};
      EXPECT_FALSE(agx_apple9_decode_mesh(changed, &ins)) << bit;
   }
   agx_apple9_mesh_instruction ins = {};
   ins.kind = AGX_APPLE9_MESH_CONTROL;
   uint8_t bytes[6];
   ins.count_gpr = 96;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
   ins.count_gpr = 0;
   ins.auxiliary_gpr = 96;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
   ins = {};
   ins.kind = AGX_APPLE9_MESH_SUBMIT;
   for (auto mode : {AGX_APPLE9_MESH_SUBMIT_SCALAR,
                     AGX_APPLE9_MESH_SUBMIT_CHANNEL}) {
      ins.mode = mode;
      ins.source_gpr = 95;
      EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
      ins.source_gpr = 94;
      EXPECT_TRUE(agx_apple9_encode_mesh(bytes, &ins));
   }
   ins.mode = (agx_apple9_mesh_submit_mode)3;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
   ins.mode = (agx_apple9_mesh_submit_mode)-1;
   EXPECT_FALSE(agx_apple9_encode_mesh(bytes, &ins));
}

TEST(Apple9MeshEncoding, VirPacksAllocatedHalfwordsAndAdjacentPairs)
{
   for (unsigned a : {0u, 3u, 7u, 31u, 64u, 95u}) {
      for (unsigned halves = 0; halves < 4; ++halves) {
         agx_apple9_vir_instr I = {};
         I.op = AGX_APPLE9_VIR_MESH_CONTROL;
         I.encoding = AGX_APPLE9_ENC_MESH_CONTROL;
         I.dest = AGX_APPLE9_VREG_INVALID;
         I.src[0] = 0;
         I.src[1] = 1;
         I.nr_srcs = 2;
         I.src_half_mask = 3;
         I.src_hi_mask = halves;
         I.mesh_control42 = true;
         const uint8_t phys[] = {(uint8_t)a, (uint8_t)(95 - a)};
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(
            agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
         agx_apple9_mesh_instruction decoded = {};
         ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
         EXPECT_EQ(decoded.count_gpr, a);
         EXPECT_EQ(decoded.count_high, bool(halves & 1));
         EXPECT_EQ(decoded.auxiliary_gpr, 95 - a);
         EXPECT_EQ(decoded.auxiliary_high, bool(halves & 2));
         for (unsigned live : {1u, 2u, 3u}) {
            I.live_after_mask = live;
            EXPECT_FALSE(
               agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
         }
      }
   }
   for (bool channel : {false, true}) {
      agx_apple9_vir_instr I = {};
      I.op = channel ? AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL
                     : AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR;
      I.encoding = channel ? AGX_APPLE9_ENC_MESH_SUBMIT_CHANNEL
                           : AGX_APPLE9_ENC_MESH_SUBMIT_SCALAR;
      I.dest = AGX_APPLE9_VREG_INVALID;
      I.src[0] = 0;
      I.src[1] = 1;
      I.nr_srcs = 2;
      I.mesh_release = channel;
      for (unsigned base : {0u, 5u, 11u, 44u, 63u, 64u, 94u}) {
         uint8_t phys[] = {(uint8_t)base, (uint8_t)(base + 1)};
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
         agx_apple9_mesh_instruction decoded = {};
         ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
         EXPECT_EQ(decoded.source_gpr, base);
         EXPECT_EQ(decoded.mode, channel ? AGX_APPLE9_MESH_SUBMIT_CHANNEL
                                         : AGX_APPLE9_MESH_SUBMIT_SCALAR);
         EXPECT_EQ(decoded.release, channel);
         I.nr_srcs = 1;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
         I.nr_srcs = 2;
         phys[1] = base;
         EXPECT_FALSE(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr));
      }
   }
}

TEST(Apple9MeshIR, AllocationPreservesReadClearAndTupleInputsUsedLater)
{
   std::set<unsigned> count_locations, pair_locations;
   for (unsigned pressure : {0u, 12u, 112u}) {
      for (bool high : {false, true}) {
         for (unsigned primitive_count : {0u, 1u, 2u}) {
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_MESH, &agx_nir_options, "mesh_destructive_operands");
            nir_block logical = {};
            agx_apple9_vir_program p;
            agx_apple9_vir_init(&p);
            auto block = agx_apple9_block_create(&p);
            block->nir = &logical;
            agx_apple9_block_begin(&p, block);
            auto zero = agx_apple9_vir_emit(
               &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
            agx_apple9_device_load_contract load = {
               .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
               .flags = 0,
               .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
            };
            std::vector<uint32_t> values;
            for (unsigned i = 0; i < pressure + 4; ++i)
               values.push_back(
                  agx_apple9_vir_emit_device_load(&p, i, zero, &load));
            /* Keep every load before the mesh operations so the large case
             * exceeds the register file instead of scheduling loads later. */
            ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
               &p, AGX_APPLE9_VIR_WORKGROUP_BARRIER,
               AGX_APPLE9_ENC_WORKGROUP_BARRIER, nullptr, 0, 0));
            ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, values[0],
                                                         values[1], true));
            p.instructions[p.instruction_count - 1]->src_hi_mask = high;
            const auto *control = p.instructions[p.instruction_count - 1];
            EXPECT_FALSE(agx_apple9_instr_is_pure_alu(control));
            EXPECT_EQ(agx_apple9_find_operand(control->encoding,
                                              AGX_APPLE9_OPERAND_SRC0)
                         ->widths,
                      AGX_APPLE9_WIDTH_16);
            EXPECT_TRUE(agx_apple9_find_operand(control->encoding,
                                                AGX_APPLE9_OPERAND_SRC0)
                           ->flags &
                        AGX_APPLE9_OPERAND_CLOBBER);
            EXPECT_TRUE(agx_apple9_find_operand(control->encoding,
                                                AGX_APPLE9_OPERAND_SRC1)
                           ->flags & AGX_APPLE9_OPERAND_CLOBBER);
            uint32_t pair[] = {values[2], values[3]};
            ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, pair, true, true));
            uint32_t scalar[] = {values[1], zero};
            ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, scalar, false, false));
            ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
               &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
               &zero, 1, values.size()));
            /* Every original SSA value remains live across the destructive ops.
             */
            for (unsigned i = 0; i < values.size(); ++i)
               ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
                  &p, AGX_APPLE9_VIR_STORE_UNIFORM,
                  AGX_APPLE9_ENC_STORE_UNIFORM, &values[i], 1, i));
            const char *reason = nullptr;
            ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason))
               << reason;

            auto input = [&](unsigned i) -> uint32_t {
               if (i == 0)
                  return high ? (primitive_count << 16) | 0xcafe
                              : 0xcafe0000 | primitive_count;
               if (i == 1)
                  return 3;
               if (i == 2)
                  return 2;
               if (i == 3)
                  return 64;
               return 0x12340000 + i;
            };
            if (pressure == 112) {
               EXPECT_GT(p.scratch_size, 0u);
            }
            uint32_t regs[AGX_APPLE9_GPR_COUNT] = {}, uniform[256] = {};
            uint32_t scratch[AGX_APPLE9_SAVE_WORD_COUNT] = {};
            unsigned controls = 0, channels = 0, scalars = 0;
            for (unsigned i = 0; i < p.instruction_count; ++i) {
               const auto *I = p.instructions[i];
               agx_apple9_packed_instruction packed = {};
               ASSERT_TRUE(
                  agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
                  << "op=" << I->op << " " << (reason ?: "");
               switch (I->op) {
               case AGX_APPLE9_VIR_IMM:
                  regs[I->dest] = I->immediate;
                  break;
               case AGX_APPLE9_VIR_DEVICE_LOAD:
                  regs[I->dest] = input(I->immediate);
                  break;
               case AGX_APPLE9_VIR_IOR:
                  regs[I->dest] = regs[I->src[0]] | regs[I->src[1]];
                  break;
               case AGX_APPLE9_VIR_SPILL_STORE:
                  scratch[I->immediate] = regs[I->src[0]];
                  break;
               case AGX_APPLE9_VIR_SPILL_LOAD:
                  regs[I->dest] = scratch[I->immediate];
                  break;
               case AGX_APPLE9_VIR_WORKGROUP_BARRIER:
               case AGX_APPLE9_VIR_WAIT:
                  break;
               case AGX_APPLE9_VIR_MESH_CONTROL: {
                  agx_apple9_mesh_instruction decoded = {};
                  ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
                  count_locations.insert(decoded.count_gpr);
                  unsigned shift = decoded.count_high ? 16 : 0;
                  EXPECT_EQ((regs[decoded.count_gpr] >> shift) & 65535,
                            primitive_count);
                  EXPECT_EQ(regs[decoded.auxiliary_gpr] & 65535, 3u);
                  EXPECT_EQ(I->live_after_mask, 0u);
                  regs[decoded.count_gpr] &= ~(65535u << shift);
                  /* Worst case until hardware qualifies B preservation. */
                  regs[decoded.auxiliary_gpr] = 0xdeadbeef;
                  ++controls;
                  break;
               }
               case AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR:
               case AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL: {
                  agx_apple9_mesh_instruction decoded = {};
                  ASSERT_TRUE(agx_apple9_decode_mesh(packed.bytes, &decoded));
                  pair_locations.insert(decoded.source_gpr);
                  EXPECT_EQ(I->src[1], I->src[0] + 1);
                  bool channel = I->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL;
                  EXPECT_EQ(regs[decoded.source_gpr], channel ? 2u : 3u);
                  EXPECT_EQ(regs[decoded.source_gpr + 1], channel ? 64u : 0u);
                  EXPECT_EQ(I->live_after_mask, 0u);
                  /* Poison released storage: any use of the old SSA value
                   * must use the allocator's preserved copy. */
                  regs[decoded.source_gpr] = regs[decoded.source_gpr + 1] =
                     0xdeadbeef;
                  channel ? ++channels : ++scalars;
                  break;
               }
               case AGX_APPLE9_VIR_STORE_UNIFORM:
                  uniform[I->immediate] = regs[I->src[0]];
                  break;
               default:
                  FAIL() << "unhandled physical IR " << I->op;
               }
            }
            EXPECT_EQ(controls, 1u);
            EXPECT_EQ(channels, 1u);
            EXPECT_EQ(scalars, 1u);
            EXPECT_EQ(uniform[values.size()], 0u);
            for (unsigned i = 0; i < values.size(); ++i)
               EXPECT_EQ(uniform[i], input(i)) << "live value " << i;
            agx_apple9_vir_finish(&p);
            ralloc_free(b.shader);
         }
      }
   }
   EXPECT_GT(count_locations.size(), 1u);
   EXPECT_GT(pair_locations.size(), 1u);
}

TEST(Apple9MeshIR, CountOnlyEffectSurvivesAndDrainsLoads)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "mesh_count_only");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);
   auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                   AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   agx_apple9_device_load_contract load = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto count = agx_apple9_vir_emit_device_load(&p, 0, zero, &load);
   auto auxiliary = agx_apple9_vir_emit_device_load(&p, 1, zero, &load);
   ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, count, auxiliary, true));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   unsigned controls = 0, pending = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      agx_apple9_packed_instruction packed = {};
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason));
      if (I->scoreboard_slot)
         pending &= ~(1u << (I->scoreboard_slot - 1));
      if (I->op == AGX_APPLE9_VIR_WAIT)
         pending &= ~I->immediate;
      if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         ASSERT_GT(I->producer_scoreboard_slot, 0u);
         pending |= 1u << (I->producer_scoreboard_slot - 1);
      }
      if (I->op == AGX_APPLE9_VIR_MESH_CONTROL) {
         EXPECT_EQ(pending, 0u);
         ++controls;
      }
   }
   EXPECT_EQ(controls, 1u);
   agx_apple9_vir_finish(&p);
   ralloc_free(b.shader);
}
