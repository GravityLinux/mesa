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
            EXPECT_EQ(agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr),
                      live == 2u);
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
            EXPECT_FALSE(agx_apple9_find_operand(control->encoding,
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
                  EXPECT_EQ(I->live_after_mask & ~2u, 0u);
                  regs[decoded.count_gpr] &= ~(65535u << shift);
                  /* B remains available to its later SSA uses. */
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

/* The submit must order independent memory work, even when none of its
 * operands comes from that work. Exercise allocation and the physical
 * dependency pass, with zero, one and several occupied slots. */
TEST(Apple9MeshIR, SubmitsDrainIndependentProducers)
{
   for (bool channel : {false, true}) {
      for (unsigned count : {0u, 1u, 2u, 6u}) {
         SCOPED_TRACE(::testing::Message() << "channel=" << channel
                                          << " producers=" << count);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_MESH, &agx_nir_options, "mesh_completion_consumer");
         nir_block logical = {};
         agx_apple9_vir_program p;
         agx_apple9_vir_init(&p);
         auto block = agx_apple9_block_create(&p);
         block->nir = &logical;
         agx_apple9_block_begin(&p, block);
         auto zero = agx_apple9_vir_emit(
            &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
         auto one = agx_apple9_vir_emit(
            &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 1);
         agx_apple9_device_load_contract contract = {
            .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
            .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
         };
         std::vector<uint32_t> values;
         for (unsigned i = 0; i < count; ++i)
            values.push_back(agx_apple9_vir_emit_device_load(
               &p, i + 1, zero, &contract));
         uint32_t pair[] = {one, zero};
         ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, pair, channel, false));
         for (unsigned i = 0; i < count; ++i)
            ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
               &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
               &values[i], 1, i));
         const char *reason = nullptr;
         ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
         unsigned pending = 0, issued = 0, waited = 0, submits = 0;
         for (unsigned i = 0; i < p.instruction_count; ++i) {
            const auto *I = p.instructions[i];
            agx_apple9_packed_instruction packed = {};
            ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed,
                                                      &reason)) << reason;
            if (I->scoreboard_slot)
               pending &= ~BITFIELD_BIT(I->scoreboard_slot - 1);
            if (I->op == AGX_APPLE9_VIR_WAIT) {
               EXPECT_NE(I->immediate, 0u);
               EXPECT_EQ(I->immediate & ~pending, 0u);
               waited |= I->immediate;
               pending &= ~I->immediate;
            }
            if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
               ASSERT_GE(I->producer_scoreboard_slot, 1u);
               ASSERT_LE(I->producer_scoreboard_slot, 6u);
               unsigned bit = BITFIELD_BIT(I->producer_scoreboard_slot - 1);
               EXPECT_EQ(pending & bit, 0u);
               pending |= bit;
               issued |= bit;
            }
            if (I->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL ||
                I->op == AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR) {
               ++submits;
               EXPECT_EQ(pending, 0u);
               EXPECT_EQ(util_bitcount(issued), count);
               EXPECT_EQ(waited, issued);
               EXPECT_EQ(I->scoreboard_slot, 0u);
               EXPECT_EQ(I->producer_scoreboard_slot, 0u);
               if (count) {
                  ASSERT_GT(i, 0u);
                  EXPECT_EQ(p.instructions[i - 1]->op, AGX_APPLE9_VIR_WAIT);
                  EXPECT_EQ(p.instructions[i - 1]->immediate, issued);
               }
            }
         }
         EXPECT_EQ(submits, 1u);
         EXPECT_EQ(pending, 0u);
         agx_apple9_vir_finish(&p);
         ralloc_free(b.shader);
      }
   }
}

/* This documents the current handoff contract without treating a pending
 * mesh attribution experiment as proof that every input handoff is broken.
 * The consumed load remains live as an ordinary GPR across the submit. */
TEST(Apple9MeshIR, ConsumedLoadDoesNotCreateASubmitWait)
{
   for (bool channel : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_MESH, &agx_nir_options, "mesh_after_input_handoff");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);
      auto zero = agx_apple9_vir_emit(
         &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
      agx_apple9_device_load_contract contract = {
         .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
         .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
      };
      auto loaded = agx_apple9_vir_emit_device_load(&p, 1, zero, &contract);
      uint32_t src[] = {loaded, zero};
      auto value = agx_apple9_vir_emit(
         &p, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
      uint32_t pair[] = {value, zero};
      ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, pair, channel, false));
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &loaded, 1, 0));
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
      unsigned slot = 0, handoffs = 0, submits = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed,
                                                   &reason)) << reason;
         EXPECT_NE(I->op, AGX_APPLE9_VIR_WAIT);
         if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD)
            slot = I->producer_scoreboard_slot;
         if (I->op == AGX_APPLE9_VIR_IXOR) {
            ASSERT_NE(slot, 0u);
            EXPECT_EQ(I->scoreboard_slot, slot);
            ++handoffs;
         }
         if (I->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL ||
             I->op == AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR) {
            EXPECT_EQ(handoffs, 1u);
            EXPECT_EQ(I->scoreboard_slot, 0u);
            ++submits;
         }
      }
      EXPECT_EQ(handoffs, 1u);
      EXPECT_EQ(submits, 1u);
      agx_apple9_vir_finish(&p);
      ralloc_free(b.shader);
   }
}

/* Exhaust the encoding enum (including PSEUDO) with valid operand contracts
 * for the former opcode-based barrier list. An accepted encoding must carry
 * the scheduling flag, so a new alias cannot silently bypass completion. */
TEST(Apple9MeshEncoding, CompletionBarriersHaveOnlyFlaggedEncodings)
{
   const struct {
      agx_apple9_vir_opcode op;
      agx_apple9_encoding encoding;
   } barriers[] = {
      {AGX_APPLE9_VIR_OBJECT_CALL, AGX_APPLE9_ENC_OBJECT_CALL},
      {AGX_APPLE9_VIR_OBJECT_RETURN, AGX_APPLE9_ENC_OBJECT_RETURN},
      {AGX_APPLE9_VIR_OBJECT_RESUME, AGX_APPLE9_ENC_OBJECT_RESUME},
      {AGX_APPLE9_VIR_MESH_CONTROL, AGX_APPLE9_ENC_MESH_CONTROL},
      {AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR, AGX_APPLE9_ENC_MESH_SUBMIT_SCALAR},
      {AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL, AGX_APPLE9_ENC_MESH_SUBMIT_CHANNEL},
      {AGX_APPLE9_VIR_WORKGROUP_BARRIER, AGX_APPLE9_ENC_WORKGROUP_BARRIER},
      {AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE},
      {AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT},
      {AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS},
      {AGX_APPLE9_VIR_TILE_FENCE, AGX_APPLE9_ENC_TILE_FENCE},
      {AGX_APPLE9_VIR_COVERAGE, AGX_APPLE9_ENC_COVERAGE},
      {AGX_APPLE9_VIR_DEPTH_STORE, AGX_APPLE9_ENC_DEPTH_STORE},
   };
   const uint8_t phys[] = {0, 1};
   for (const auto &barrier : barriers) {
      SCOPED_TRACE(unsigned(barrier.op));
      agx_apple9_vir_instr I = {};
      I.op = barrier.op;
      I.dest = AGX_APPLE9_VREG_INVALID;
      I.src[0] = 0;
      I.src[1] = 1;
      switch (I.op) {
      case AGX_APPLE9_VIR_MESH_CONTROL:
         I.src_half_mask = 3;
         FALLTHROUGH;
      case AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR:
      case AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL:
         I.nr_srcs = 2;
         break;
      case AGX_APPLE9_VIR_COVERAGE:
         I.nr_srcs = 1;
         break;
      case AGX_APPLE9_VIR_DEPTH_STORE:
         I.nr_srcs = 2;
         I.immediate = 7;
         break;
      case AGX_APPLE9_VIR_TILE_ACCESS:
      case AGX_APPLE9_VIR_TILE_FENCE:
         I.immediate = 1;
         break;
      default:
         break;
      }
      unsigned accepted = 0;
      for (unsigned e = 0; e <= AGX_APPLE9_ENC_COUNT; ++e) {
         I.encoding = e == AGX_APPLE9_ENC_COUNT
            ? AGX_APPLE9_ENC_PSEUDO : agx_apple9_encoding(e);
         SCOPED_TRACE(unsigned(I.encoding));
         agx_apple9_packed_instruction packed = {};
         bool packs = agx_apple9_pack_vir_instruction(&I, phys, &packed, nullptr);
         EXPECT_EQ(packs, I.encoding == barrier.encoding);
         if (packs) {
            ASSERT_NE(I.encoding, AGX_APPLE9_ENC_PSEUDO);
            EXPECT_TRUE(agx_apple9_encoding_info(I.encoding)->requires_prior_completion);
            ++accepted;
         }
      }
      EXPECT_EQ(accepted, 1u);
   }
}

/* Exercise the physical dependency pass through normal shared allocation.
 * U0 produces no GPR and must never acquire a fabricated encoded output tag. */
class Apple9MeshCompletion : public ::testing::Test {
protected:
   nir_builder b;
   nir_block logical[4] = {};
   agx_apple9_vir_program p;
   std::vector<agx_apple9_block *> blocks;

   void SetUp() override
   {
      b = nir_builder_init_simple_shader(MESA_SHADER_MESH, &agx_nir_options,
                                         "mesh_completion_effect");
      agx_apple9_vir_init(&p);
      begin(0);
   }

   void TearDown() override
   {
      agx_apple9_vir_finish(&p);
      ralloc_free(b.shader);
   }

   void begin(unsigned index)
   {
      while (blocks.size() <= index) {
         auto block = agx_apple9_block_create(&p);
         block->nir = &logical[blocks.size()];
         blocks.push_back(block);
      }
      agx_apple9_block_begin(&p, blocks[index]);
   }

   uint32_t imm(unsigned x)
   {
      return agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                               AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, x);
   }

   void control()
   {
      auto count = imm(1), auxiliary = imm(3);
      ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, count, auxiliary, true));
   }

   void fence()
   {
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE,
         nullptr, 0, 0));
   }

   void submit(bool channel)
   {
      uint32_t pair[] = {imm(3), imm(0)};
      ASSERT_TRUE(agx_apple9_vir_emit_mesh_submit(&p, pair, channel, false));
   }

   void allocate(unsigned expected_waits)
   {
      allocate(std::vector<unsigned>(expected_waits, 0x03));
   }

   void allocate(const std::vector<unsigned> &expected_waits)
   {
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
      unsigned waits = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         const auto *I = p.instructions[i];
         agx_apple9_packed_instruction packed = {};
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed,
                                                   &reason)) << reason;
         if (I->op == AGX_APPLE9_VIR_WAIT) {
            ASSERT_LT(waits, expected_waits.size());
            EXPECT_EQ(I->immediate, expected_waits[waits++]);
         }
         if (I->op == AGX_APPLE9_VIR_MESH_CONTROL) {
            EXPECT_EQ(I->dest, AGX_APPLE9_VREG_INVALID);
            EXPECT_EQ(I->dest_components, 0u);
            EXPECT_EQ(I->producer_scoreboard_slot, 0u);
            EXPECT_EQ(I->scoreboard_slot, 0u);
         }
      }
      EXPECT_EQ(waits, expected_waits.size());
   }

   void wait_before(enum agx_apple9_vir_opcode op, unsigned occurrence = 0)
   {
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         if (p.instructions[i]->op != op || occurrence--)
            continue;
         ASSERT_GT(i, 0u);
         EXPECT_EQ(p.instructions[i - 1]->op, AGX_APPLE9_VIR_WAIT);
         EXPECT_EQ(p.instructions[i - 1]->immediate, 0x03u);
         return;
      }
      FAIL() << "missing consumer " << op;
   }
};

TEST_F(Apple9MeshCompletion, FenceRetiresResultlessControl)
{
   control();
   fence();
   allocate(1);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
}

TEST_F(Apple9MeshCompletion, SubmitsRetireControlOnlyOnce)
{
   control();
   submit(true);
   submit(false);
   allocate(1);
   wait_before(AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL);
}

TEST_F(Apple9MeshCompletion, ScalarSubmitRetiresControl)
{
   control();
   submit(false);
   allocate(1);
   wait_before(AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR);
}

TEST_F(Apple9MeshCompletion, AlreadyRetiredControlDoesNotWaitAgain)
{
   control();
   fence();
   begin(1);
   submit(true);
   fence();
   submit(false);
   allocate(1);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
}

TEST_F(Apple9MeshCompletion, RepeatedControlOrdersItsPredecessor)
{
   control();
   control();
   submit(false);
   allocate(2);
   wait_before(AGX_APPLE9_VIR_MESH_CONTROL, 1);
   wait_before(AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR);
}

TEST_F(Apple9MeshCompletion, BothBranchArmsReachTheJoin)
{
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   control();
   begin(2);
   control();
   begin(3);
   submit(false);
   allocate(2);
   wait_before(AGX_APPLE9_VIR_MESH_CONTROL, 1);
   wait_before(AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR);
}

TEST_F(Apple9MeshCompletion, RetiredBranchArmsDoNotWaitAtTheJoin)
{
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   control();
   fence();
   begin(2);
   control();
   fence();
   begin(3);
   submit(false);
   allocate(2);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE, 1);
}

TEST_F(Apple9MeshCompletion, JoinRetiresThePathBypassingAnEarlierFence)
{
   control();
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   fence();
   begin(2);
   begin(3);
   fence();
   submit(false);
   allocate(2);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE, 1);
}

TEST_F(Apple9MeshCompletion, BackedgeDrainsBeforeLogicalEnd)
{
   logical[0].successors[0] = &logical[1];
   logical[1].successors[0] = &logical[1];
   logical[1].successors[1] = &logical[2];
   begin(1);
   control();
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_JMP_EXEC_ANY, AGX_APPLE9_ENC_JMP_EXEC_ANY,
      nullptr, 0, 0));
   p.instructions[p.instruction_count - 1]->branch_target = blocks[1];
   begin(2);
   submit(false);
   allocate(1);
   wait_before(AGX_APPLE9_VIR_JMP_EXEC_ANY);
}

TEST_F(Apple9MeshCompletion, FixedProducerSlotRetiresTheWholeEffect)
{
   control();
   auto value = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_GET_SR, AGX_APPLE9_ENC_GET_DRAW_ID, nullptr, 0, 0x10dd);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &value, 1, 0));
   submit(false);
   allocate(1);
   wait_before(AGX_APPLE9_VIR_GET_SR);
}

TEST_F(Apple9MeshCompletion, PendingEffectExcludesSlotsOneAndTwo)
{
   control();
   auto index = imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t values[4];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = agx_apple9_vir_emit_device_load(&p, i + 1, index, &contract);
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &values[i], 1, i));
   fence();
   allocate(1);
   unsigned slots = 0, loads = 0;
   bool retired = false;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      retired |= I->op == AGX_APPLE9_VIR_WAIT;
      if (I->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         continue;
      EXPECT_FALSE(retired);
      ASSERT_GE(I->producer_scoreboard_slot, 3u);
      ASSERT_LE(I->producer_scoreboard_slot, 6u);
      unsigned bit = BITFIELD_BIT(I->producer_scoreboard_slot - 1);
      EXPECT_EQ(slots & bit, 0u);
      slots |= bit;
      ++loads;
   }
   EXPECT_EQ(loads, 4u);
   EXPECT_EQ(slots, 0x3cu);
   wait_before(AGX_APPLE9_VIR_DEVICE_FENCE);
}

TEST_F(Apple9MeshCompletion, ExhaustionRetiresTheEffectBeforeSlotReuse)
{
   control();
   auto index = imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t values[5];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = agx_apple9_vir_emit_device_load(&p, i + 1, index, &contract);
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
         &values[i], 1, i));
   fence();
   allocate(1);
   unsigned slots = 0, loads = 0, waits = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      waits += I->op == AGX_APPLE9_VIR_WAIT;
      if (I->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         continue;
      if (loads < 4) {
         EXPECT_EQ(waits, 0u);
         ASSERT_GE(I->producer_scoreboard_slot, 3u);
         ASSERT_LE(I->producer_scoreboard_slot, 6u);
         unsigned bit = BITFIELD_BIT(I->producer_scoreboard_slot - 1);
         EXPECT_EQ(slots & bit, 0u);
         slots |= bit;
      } else {
         EXPECT_EQ(slots, 0x3cu);
         EXPECT_EQ(waits, 1u);
         EXPECT_GE(I->producer_scoreboard_slot, 1u);
         EXPECT_LE(I->producer_scoreboard_slot, 2u);
      }
      ++loads;
   }
   EXPECT_EQ(loads, 5u);
   wait_before(AGX_APPLE9_VIR_DEVICE_LOAD, 4);
}

TEST_F(Apple9MeshCompletion, JoinedSlotOneResultRetiresTheWholeEffect)
{
   /* The load dominates both arms. One arm bypasses U0 and keeps its result
    * pending; the other retires it at U0 and issues the resultless effect.
    * The join must satisfy both possible obligations before reading it. */
   auto index = imm(0);
   agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto first = agx_apple9_vir_emit_device_load(&p, 1, index, &contract);
   auto joined = agx_apple9_vir_emit_device_load(&p, 2, index, &contract);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &first, 1, 0));
   logical[0].successors[0] = &logical[1];
   logical[0].successors[1] = &logical[2];
   logical[1].successors[0] = &logical[3];
   logical[2].successors[0] = &logical[3];
   begin(1);
   begin(2);
   control();
   begin(3);
   uint32_t src[] = {joined, imm(0x1234)};
   auto result = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, src, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
      &result, 1, 1));
   fence();
   allocate({0x01, 0x03});
   unsigned loads = 0, consumers = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      const auto *I = p.instructions[i];
      if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD)
         EXPECT_EQ(I->producer_scoreboard_slot, loads++ ? 1u : 6u);
      if (I->op == AGX_APPLE9_VIR_IXOR) {
         EXPECT_EQ(I->scoreboard_slot, AGX_APPLE9_SCOREBOARD_SLOT_NONE);
         ++consumers;
      }
   }
   EXPECT_EQ(loads, 2u);
   EXPECT_EQ(consumers, 1u);
   wait_before(AGX_APPLE9_VIR_IXOR);
}

TEST_F(Apple9MeshCompletion, EndOfProgramDrainsControl)
{
   control();
   allocate(1);
   ASSERT_GT(p.instruction_count, 0u);
   EXPECT_EQ(p.instructions[p.instruction_count - 1]->op, AGX_APPLE9_VIR_WAIT);
}

TEST(Apple9MeshIR, AuxiliaryClassRetainsLiveValueAndDetectsClobberMutation)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "mesh_auxiliary_preservation");
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
   auto a = agx_apple9_vir_emit_device_load(&p, 0, zero, &load);
   auto v = agx_apple9_vir_emit_device_load(&p, 1, zero, &load);
   ASSERT_TRUE(agx_apple9_vir_emit_mesh_control(&p, a, v, true));
   ASSERT_TRUE(agx_apple9_vir_emit_side_effect(
      &p, AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM, &v, 1, 0));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, b.shader, &reason)) << reason;
   ASSERT_TRUE(agx_apple9_validate_vir_allocation(&p, &reason)) << reason;

   /* Evaluate the actual allocated instructions twice. Only the second run
    * mutates U0's B effect to a destructive one. This must be observable: a
    * hidden allocator clobber copy would incorrectly mask the mutation. */
   uint32_t regs[2][AGX_APPLE9_GPR_COUNT] = {};
   uint32_t outputs[2] = {};
   unsigned controls = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      auto *I = p.instructions[i];
      for (unsigned m = 0; m < 2; ++m) {
         switch (I->op) {
         case AGX_APPLE9_VIR_IMM:
            regs[m][I->dest] = I->immediate;
            break;
         case AGX_APPLE9_VIR_DEVICE_LOAD:
            regs[m][I->dest] = I->immediate ? 3 : 1;
            break;
         case AGX_APPLE9_VIR_IOR:
            regs[m][I->dest] = regs[m][I->src[0]] | regs[m][I->src[1]];
            break;
         case AGX_APPLE9_VIR_IXOR:
            regs[m][I->dest] = regs[m][I->src[0]] ^ regs[m][I->src[1]];
            break;
         case AGX_APPLE9_VIR_WAIT:
            break;
         case AGX_APPLE9_VIR_MESH_CONTROL:
            EXPECT_EQ(I->live_after_mask, 2u);
            EXPECT_EQ(regs[m][I->src[1]], 3u);
            EXPECT_NE(I->src[0], I->src[1]);
            regs[m][I->src[0]] &= 0xffff0000u;
            if (m)
               regs[m][I->src[1]] = 0xdeadbeef;
            else {
               ++controls;
               /* An out-of-class physical operand must still be rejected. */
               auto saved = I->src[1];
               I->src[1] = AGX_APPLE9_GPR_COUNT;
               EXPECT_FALSE(agx_apple9_validate_vir_allocation(&p, &reason));
               I->src[1] = saved;
               EXPECT_TRUE(agx_apple9_validate_vir_allocation(&p, &reason));
            }
            break;
         case AGX_APPLE9_VIR_STORE_UNIFORM:
            outputs[m] = regs[m][I->src[0]];
            break;
         default:
            FAIL() << "unhandled allocated opcode " << I->op;
         }
      }
   }
   EXPECT_EQ(controls, 1u);
   EXPECT_EQ(outputs[0], 3u);
   EXPECT_EQ(outputs[1], 0xdeadbeefu);
   agx_apple9_vir_finish(&p);
   ralloc_free(b.shader);
}
