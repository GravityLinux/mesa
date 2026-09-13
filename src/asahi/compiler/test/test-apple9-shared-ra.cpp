/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>
#include "agx_test.h"

TEST(Apple9SharedRa, PlacementCostsLeaveSpaceForRestrictedDefinitions)
{
   unsigned moves[2] = {};
   for (unsigned policy = 0; policy < 2; ++policy) {
      void *memctx = ralloc_context(nullptr);
      auto ctx = rzalloc(memctx, agx_context);
      list_inithead(&ctx->blocks);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "scarce_bank_cost");
      agx_shader_key key = {};
      key.has_scratch = true;
      ctx->nir = nir.shader; ctx->stage = MESA_SHADER_COMPUTE; ctx->key = &key;
      ctx->ra_target.max_registers = 32;
      ctx->ra_target.reserved_registers = 4;
      ctx->ra_target.late_kill_sources = true;
      ctx->ra_target.cost_register_constraints = policy;
      ctx->ra_target.disable_occupancy_rematerialization = true;
      auto block = agx_test_block(ctx);
      auto b = agx_init_builder(ctx, agx_after_block(block));
      agx_index v[12];
      for (unsigned i = 0; i < 12; ++i)
         v[i] = agx_mov_imm(&b, 32, i + 1);
      auto low = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), v[0], v[1], 0);
      low->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, 3);
      low->reg_constraints[0] = {4,6,2};
      low->reg_constraints[1] = low->reg_constraints[2] = {0,30,2};
      auto sum = low->dest[0];
      for (unsigned i = 0; i < 12; ++i)
         sum = agx_iadd(&b, sum, v[i], 0);
      agx_export(&b, sum, 0);
      agx_ra(ctx);
      ASSERT_EQ(ctx->scratch_size_B, 0u);
      ASSERT_LT(ctx->max_reg, 32u);
      uint32_t regs[32] = {};
      agx_foreach_instr_in_block(block, I) {
         auto read = [&](agx_index x) {
            return x.type == AGX_INDEX_IMMEDIATE ? x.value : regs[x.value];
         };
         switch (I->op) {
         case AGX_OPCODE_MOV_IMM: regs[I->dest[0].value] = I->imm; break;
         case AGX_OPCODE_MOV:
            regs[I->dest[0].value] = read(I->src[0]); ++moves[policy]; break;
         case AGX_OPCODE_IADD:
            regs[I->dest[0].value] = read(I->src[0]) + read(I->src[1]); break;
         case AGX_OPCODE_SWAP: {
            auto a = I->src[0].value, c = I->src[1].value;
            std::swap(regs[a], regs[c]); ++moves[policy]; break;
         }
         case AGX_OPCODE_EXPORT: regs[I->imm] = read(I->src[0]); break;
         default: FAIL() << "unexpected opcode " << I->op;
         }
      }
      EXPECT_EQ(regs[0], 81u);
      ralloc_free(nir.shader); ralloc_free(memctx);
   }
   EXPECT_LT(moves[1], moves[0]);
}

TEST(Apple9SharedRa, InstructionConstraintsRelocateLiveValuesOutOfFullBanks)
{
   void *memctx = ralloc_context(nullptr);
   auto ctx = rzalloc(memctx, agx_context);
   list_inithead(&ctx->blocks);
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "constrained_banks");
   agx_shader_key key = {};
   key.has_scratch = true;
   ctx->nir = nir.shader;
   ctx->stage = MESA_SHADER_COMPUTE;
   ctx->key = &key;
   ctx->ra_target.max_registers = 128;
   ctx->ra_target.reserved_registers = 8;
   ctx->ra_target.late_kill_sources = true;
   ctx->ra_target.defer_spill_lowering = true;
   auto block = agx_test_block(ctx);
   auto b = agx_init_builder(ctx, agx_after_block(block));
   constexpr unsigned count = 56;
   agx_index values[count];
   uint32_t expected[count];
   for (unsigned i = 0; i < count; ++i) {
      expected[i] = (i * 0x9e3779b9u) ^ 0xa511e9b3u;
      values[i] = agx_mov_imm(&b, 32, expected[i]);
   }
   std::vector<agx_instr *> constrained;
   auto sum = agx_mov_imm(&b, 32, 0);
   uint32_t expected_sum = 0;
   for (unsigned i = 0; i < count; ++i) {
      /* All 56 values remain live. Both inputs and the output must fit in
       * r4-r7, so simply searching for an unused constrained register fails. */
      auto I = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32),
                           values[i], values[count - 1 - i], 0);
      I->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, 3);
      for (unsigned c = 0; c < 3; ++c)
         I->reg_constraints[c] = {8, 14, 2};
      constrained.push_back(I);
      sum = agx_iadd(&b, sum, I->dest[0], 0);
      expected_sum += expected[i] + expected[count - 1 - i];
   }
   /* Check every relocated value, not just the constrained results. */
   for (unsigned i = 0; i < count; ++i) {
      sum = agx_iadd(&b, values[i], sum, 5);
      expected_sum = expected[i] + expected_sum * 32;
   }
   agx_export(&b, sum, 120);
   agx_ra(ctx);
   EXPECT_EQ(ctx->scratch_size_B, 0u);
   for (auto I : constrained) {
      EXPECT_GE(I->dest[0].value, 8u);
      EXPECT_LE(I->dest[0].value, 14u);
      for (unsigned s = 0; s < 2; ++s) {
         EXPECT_GE(I->src[s].value, 8u);
         EXPECT_LE(I->src[s].value, 14u);
      }
   }
   uint32_t regs[128] = {};
   unsigned moves = 0;
   agx_foreach_instr_in_block(block, I) {
      auto read = [&](agx_index x) {
         return x.type == AGX_INDEX_IMMEDIATE ? x.value : regs[x.value];
      };
      switch (I->op) {
      case AGX_OPCODE_MOV_IMM:
         regs[I->dest[0].value] = I->imm;
         break;
      case AGX_OPCODE_IADD:
         regs[I->dest[0].value] = read(I->src[0]) + (read(I->src[1]) << I->shift);
         break;
      case AGX_OPCODE_MOV:
         regs[I->dest[0].value] = read(I->src[0]);
         ++moves;
         break;
      case AGX_OPCODE_SWAP: {
         auto old = read(I->src[0]);
         regs[I->src[0].value] = read(I->src[1]);
         regs[I->src[1].value] = old;
         break;
      }
      case AGX_OPCODE_EXPORT:
         regs[I->imm] = read(I->src[0]);
         break;
      default:
         FAIL() << "unexpected opcode " << I->op;
      }
   }
   EXPECT_EQ(regs[120], expected_sum);
   EXPECT_GT(moves, 0u);
   ralloc_free(nir.shader);
   ralloc_free(memctx);
}

TEST(Apple9SharedRa, SpilledLoopPhisPreserveParallelAssignments)
{
   void *memctx = ralloc_context(NULL);
   agx_context *ctx = rzalloc(memctx, agx_context);
   list_inithead(&ctx->blocks);
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "shared_spill_loop");
   nir.shader->info.workgroup_size[0] = 32;
   nir.shader->info.workgroup_size[1] = 1;
   nir.shader->info.workgroup_size[2] = 1;
   agx_shader_key key = {};
   key.has_scratch = true;
   ctx->nir = nir.shader;
   ctx->stage = MESA_SHADER_COMPUTE;
   ctx->key = &key;
   ctx->ra_target.max_registers = 128;
   ctx->ra_target.reserved_registers = 32;
   ctx->ra_target.spill_copy_register = 24;
   ctx->ra_target.spill_swap_registers[0] = 26;
   ctx->ra_target.spill_swap_registers[1] = 28;
   ctx->ra_target.late_kill_sources = true;
   ctx->ra_target.defer_spill_lowering = true;
   agx_block *entry = agx_test_block(ctx);
   agx_block *header = agx_test_block(ctx);
   agx_block *body = agx_test_block(ctx);
   agx_block *exit = agx_test_block(ctx);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);
   const unsigned count = 80;
   agx_index initial[count], value[count];
   agx_instr *phi[count];
   uint32_t expected_values[count];
   agx_builder b = agx_init_builder(ctx, agx_after_block(entry));
   for (unsigned i = 0; i < count; ++i) {
      expected_values[i] = (i * 0x9e3779b9u) ^ 0xa511e9b3u;
      initial[i] = agx_mov_imm(&b, 32, expected_values[i]);
   }
   b.cursor = agx_after_block(header);
   for (unsigned i = 0; i < count; ++i) {
      value[i] = agx_temp(ctx, AGX_SIZE_32);
      phi[i] = agx_phi_to(&b, value[i], 2);
      phi[i]->src[0] = initial[i];
   }
   b.cursor = agx_after_block(body);
   for (unsigned i = 0; i < count; ++i)
      phi[i]->src[1] = agx_iadd(&b, value[i], value[(i + 1) % count], 0);
   b.cursor = agx_after_block(exit);
   agx_index hash = value[0];
   for (unsigned i = 1; i < count; ++i)
      hash = agx_iadd(&b, value[i], hash, 5);
   agx_export(&b, hash, 120);
   agx_ra(ctx);
   ASSERT_GT(ctx->scratch_size_B, 0u);
   EXPECT_LE(ctx->scratch_size_B, 4096u);
   EXPECT_LT(ctx->max_reg, 128u);
   uint32_t regs[128] = {}, memory[2048] = {};
   unsigned stores = 0, loads = 0;
   auto execute = [&](agx_block *block) {
      agx_foreach_instr_in_block(block, I) {
         auto read = [&](agx_index x) {
            return x.type == AGX_INDEX_IMMEDIATE
                      ? x.value
                      : (x.memory ? memory : regs)[x.value];
         };
         uint32_t result;
         switch (I->op) {
         case AGX_OPCODE_MOV_IMM:
            result = I->imm;
            break;
         case AGX_OPCODE_IADD:
            result = read(I->src[0]) + (read(I->src[1]) << I->shift);
            break;
         case AGX_OPCODE_MOV:
            result = read(I->src[0]);
            stores += I->dest[0].memory;
            loads += I->src[0].memory;
            break;
         case AGX_OPCODE_EXPORT:
            regs[I->imm] = read(I->src[0]);
            continue;
         case AGX_OPCODE_SWAP: {
            uint32_t old = read(I->src[0]);
            regs[I->src[0].value] = read(I->src[1]);
            regs[I->src[1].value] = old;
            continue;
         }
         default:
            ADD_FAILURE() << "unexpected opcode " << I->op;
            return;
         }
         auto dst = I->dest[0];
         ASSERT_EQ(dst.type, AGX_INDEX_REGISTER);
         ASSERT_LT(dst.value, dst.memory ? 2048u : 128u);
         (dst.memory ? memory : regs)[dst.value] = result;
      }
   };
   execute(entry);
   for (unsigned iteration = 0; iteration < 3; ++iteration) {
      execute(header);
      execute(body);
      uint32_t first = expected_values[0];
      for (unsigned i = 0; i + 1 < count; ++i)
         expected_values[i] += expected_values[i + 1];
      expected_values[count - 1] += first;
   }
   execute(header);
   execute(exit);
   uint32_t expected = expected_values[0];
   for (unsigned i = 1; i < count; ++i)
      expected = expected * 32 + expected_values[i];
   EXPECT_EQ(regs[120], expected);
   EXPECT_GT(stores, 0u);
   EXPECT_GT(loads, 0u);
   ralloc_free(nir.shader);
   ralloc_free(memctx);
}

#include "agx_apple9_ir.h"

TEST(Apple9SharedRa, ShiftRetainsLiveInputWithoutCopies)
{
   for (bool retained : {false, true}) {
      SCOPED_TRACE(retained);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "shift_clobber");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);
      constexpr uint32_t input = 0xa511e9b3;
      auto base = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, input);
      auto shifted = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_ISHR,
         AGX_APPLE9_ENC_SHIFT_EXTENDED, &base, 1, 5);
      auto result = shifted;
      if (retained) {
         uint32_t src[] = {base, shifted};
         result = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
            AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
      }
      auto index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
      ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &result, 1, 32));
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
      EXPECT_EQ(p.scratch_size, 0u);
      EXPECT_TRUE(p.reserved_gprs[1]);
      EXPECT_FALSE(p.reserved_gprs[2]);
      uint32_t regs[96] = {}, output = 0;
      unsigned copies = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i) {
         auto I = p.instructions[i];
         switch (I->op) {
         case AGX_APPLE9_VIR_IMM:
            regs[I->dest] = I->immediate;
            break;
         case AGX_APPLE9_VIR_IOR:
            regs[I->dest] = regs[I->src[0]] | regs[I->src[1]];
            ++copies;
            break;
         case AGX_APPLE9_VIR_ISHR: {
            auto shifted_value = uint32_t(int32_t(regs[I->src[0]]) >> I->immediate);
            if (!(I->live_after_mask & 1))
               regs[I->src[0]] = 0xdeadbeef;
            regs[I->dest] = shifted_value;
            break;
         }
         case AGX_APPLE9_VIR_IADD:
            regs[I->dest] = regs[I->src[0]] + regs[I->src[1]];
            break;
         case AGX_APPLE9_VIR_DEVICE_STORE:
            output = regs[I->src[0]];
            break;
         default:
            FAIL() << "unexpected opcode " << I->op;
         }
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
            << reason;
      }
      EXPECT_EQ(output, uint32_t(int32_t(input) >> 5) + (retained ? input : 0));
      EXPECT_EQ(copies, 0u);
      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9SharedRa, LoadWaitFoldsIntoConsumerAfterIndependentArithmetic)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "late_load_wait");
   nir_block logical = {};
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto block = agx_apple9_block_create(&p);
   block->nir = &logical;
   agx_apple9_block_begin(&p, block);
   auto index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   const agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR,
      .flags = 0,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   auto loaded = agx_apple9_vir_emit_device_load(&p, 1, index, &contract);
   auto constant = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0x12345678);
   uint32_t add_src[] = {index, constant};
   auto independent = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
      AGX_APPLE9_ENC_INT_ADD_EXTENDED, add_src, 2, 0);
   uint32_t xor_src[] = {independent, loaded};
   auto result = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IXOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, xor_src, 2, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &result, 1, 32));
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
   unsigned load_position = ~0, arithmetic_position = ~0, wait_position = ~0;
   unsigned slot = 0, landing = ~0, copies = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      auto I = p.instructions[i];
      copies += I->op == AGX_APPLE9_VIR_IOR;
      if (I->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         load_position = i;
         slot = I->producer_scoreboard_slot;
         landing = I->dest;
      } else if (I->op == AGX_APPLE9_VIR_IADD) {
         arithmetic_position = i;
         EXPECT_EQ(I->scoreboard_slot, 0u);
      } else if (I->op == AGX_APPLE9_VIR_IXOR) {
         wait_position = i;
         EXPECT_EQ(I->scoreboard_slot, slot);
         EXPECT_EQ(I->src[0], landing); /* Pending logic operand normalized to A. */
      }
      agx_apple9_packed_instruction packed;
      ASSERT_TRUE(agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason)) << reason;
   }
   EXPECT_LT(load_position, arithmetic_position);
   EXPECT_LT(arithmetic_position, wait_position);
   EXPECT_EQ(copies, 0u);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

template <unsigned count>
static void
check_loop_register_pressure()
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "physical_spill_loop");
   nir.shader->info.workgroup_size[0] = 32;
   nir.shader->info.workgroup_size[1] = nir.shader->info.workgroup_size[2] = 1;
   nir_block logical[4] = {};
   logical[0].successors[0] = &logical[1];
   logical[1].successors[0] = &logical[2];
   logical[1].successors[1] = &logical[3];
   logical[2].successors[0] = &logical[1];
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   agx_apple9_block *blocks[4];
   for (unsigned i = 0; i < 4; ++i) {
      blocks[i] = agx_apple9_block_create(&p);
      blocks[i]->nir = &logical[i];
   }
   agx_apple9_block_begin(&p, blocks[0]);
   uint32_t initial[count], phi[count], expected[count];
   for (unsigned i = 0; i < count; ++i) {
      expected[i] = (i * 0x9e3779b9u) ^ 0xa511e9b3u;
      initial[i] =
         agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32,
                             nullptr, 0, expected[i]);
      phi[i] = agx_apple9_vir_emit_phi(&p, blocks[1]);
      ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&p, phi[i], initial[i]));
   }
   agx_apple9_block_begin(&p, blocks[1]);
   agx_apple9_block_begin(&p, blocks[2]);
   for (unsigned i = 0; i < count; ++i) {
      uint32_t src[] = {phi[i], phi[(i + 1) % count]};
      uint32_t next = agx_apple9_vir_emit(
         &p, AGX_APPLE9_VIR_IADD, AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
      ASSERT_TRUE(agx_apple9_vir_emit_phi_source(&p, phi[i], next));
   }
   agx_apple9_block_begin(&p, blocks[3]);
   uint32_t hash = phi[0];
   uint32_t multiplier = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 33);
   for (unsigned i = 1; i < count; ++i) {
      uint32_t src[] = {hash, multiplier, phi[i]};
      hash = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMAD,
                                 AGX_APPLE9_ENC_INT_MAD_EXTENDED, src, 3, 0);
   }
   uint32_t index = agx_apple9_vir_emit(
      &p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, index, &hash, 1, 32));
   agx_apple9_place_phis(&p);
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_assign_vir_scoreboard_slots(&p, &reason)) << reason;
   ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
   EXPECT_TRUE(p.physical);
   if (count > 96)
      ASSERT_GT(p.scratch_size, 0u);
   else
      ASSERT_EQ(p.scratch_size, 0u);
   EXPECT_LE(p.scratch_size, 4096u);
   uint32_t regs[96] = {}, memory[1024] = {}, result = 0;
   struct pending_write {
      bool memory;
      unsigned index, value, slot;
   };
   std::vector<pending_write> pending;
   unsigned saves = 0, fills = 0;
   auto execute = [&](agx_apple9_block *block) {
      unsigned end =
         block->next ? block->next->start_index : p.instruction_count;
      for (unsigned i = block->start_index; i < end; ++i) {
         auto I = p.instructions[i];
         ASSERT_EQ(agx_apple9_instr_block(I), block);
         for (auto it = pending.begin(); it != pending.end();) {
            if (it->slot == I->scoreboard_slot) {
               (it->memory ? memory : regs)[it->index] = it->value;
               it = pending.erase(it);
            } else
               ++it;
         }
         if (I->producer_scoreboard_slot) {
            for (const auto &write : pending)
               ASSERT_NE(write.slot, I->producer_scoreboard_slot);
         }
         uint32_t a = I->nr_srcs ? regs[I->src[0]] : 0;
         uint32_t b = I->nr_srcs > 1 ? regs[I->src[1]] : 0;
         switch (I->op) {
         case AGX_APPLE9_VIR_IMM:
            regs[I->dest] = I->immediate;
            break;
         case AGX_APPLE9_VIR_IADD:
            regs[I->dest] = a + b;
            break;
         case AGX_APPLE9_VIR_IMAD:
            regs[I->dest] = a * b + regs[I->src[2]];
            break;
         case AGX_APPLE9_VIR_IOR:
            regs[I->dest] = a | b;
            break;
         case AGX_APPLE9_VIR_SPILL_LOAD:
            ASSERT_LT(I->immediate, 1024u);
            pending.push_back({false, I->dest, memory[I->immediate],
                               I->producer_scoreboard_slot});
            ++fills;
            break;
         case AGX_APPLE9_VIR_SPILL_STORE:
            ASSERT_LT(I->immediate, 1024u);
            pending.push_back(
               {true, I->immediate, a, I->producer_scoreboard_slot});
            regs[I->src[0]] = 0xdeadbeef;
            ++saves;
            break;
         case AGX_APPLE9_VIR_DEVICE_STORE:
            result = a;
            break;
         default:
            FAIL() << "Unexpected physical opcode " << I->op;
         }
         agx_apple9_packed_instruction packed;
         ASSERT_TRUE(
            agx_apple9_pack_vir_instruction(I, p.phys, &packed, &reason))
            << "opcode " << I->op << " " << (reason ?: "");
      }
   };
   auto entry = p.blocks, header = entry->next, body = header->next,
        exit = body->next;
   execute(entry);
   for (unsigned iteration = 0; iteration < 3; ++iteration) {
      execute(header);
      execute(body);
      uint32_t first = expected[0];
      for (unsigned i = 0; i + 1 < count; ++i)
         expected[i] += expected[i + 1];
      expected[count - 1] += first;
   }
   execute(header);
   execute(exit);
   uint32_t expected_hash = expected[0];
   for (unsigned i = 1; i < count; ++i)
      expected_hash = expected_hash * 33 + expected[i];
   EXPECT_EQ(result, expected_hash);
   EXPECT_EQ(saves > 0, count > 96);
   EXPECT_EQ(fills > 0, count > 96);
   EXPECT_TRUE(pending.empty());
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

TEST(Apple9SharedRa, PhysicalSpillLoweringPreservesLiveSourcesAndLoopPhis)
{
   check_loop_register_pressure<80>();
   check_loop_register_pressure<112>();
}

TEST(Apple9SharedRa, TexturePublicationPressurePreservesEarlyHandoffs)
{
   for (unsigned count : {2u, 3u, 6u}) {
      SCOPED_TRACE(count);
      nir_builder nir = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "publication_pressure");
      nir_block logical = {};
      agx_apple9_vir_program p;
      agx_apple9_vir_init(&p);
      auto block = agx_apple9_block_create(&p);
      block->nir = &logical;
      agx_apple9_block_begin(&p, block);
      auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
      uint32_t coords[] = {zero, zero};
      std::vector<uint32_t> samples;
      for (unsigned t = 0; t < count; ++t)
         samples.push_back(agx_apple9_vir_emit_texture_lod(
            &p, coords, zero, t, 0, true));
      for (auto sample : samples) {
         uint32_t rgba[] = {sample, sample + 1, sample + 2, sample + 3};
         ASSERT_TRUE(agx_apple9_vir_emit_device_store(&p, 0, zero, rgba, 4, 32));
      }
      const char *reason = nullptr;
      ASSERT_TRUE(agx_apple9_allocate_publications(&p, &reason)) << reason;
      unsigned handoffs = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i)
         handoffs += p.instructions[i]->publication_handoff;
      EXPECT_EQ(handoffs, (count - 2) * 4);
      EXPECT_LE(p.publication_count, 8u);
      ASSERT_TRUE(agx_apple9_allocate_shared(&p, nir.shader, &reason)) << reason;
      unsigned retained = 0;
      for (unsigned i = 0; i < p.instruction_count; ++i)
         retained += p.instructions[i]->publication_handoff;
      EXPECT_EQ(retained, handoffs);
      agx_apple9_vir_finish(&p);
      ralloc_free(nir.shader);
   }
}

TEST(Apple9SharedRa, PublicationHandoffStaysInProducerBlock)
{
   agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   auto entry = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, entry);
   auto zero = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
      AGX_APPLE9_ENC_MOV_IMM32, nullptr, 0, 0);
   uint32_t coords[] = {zero, zero};
   auto first = agx_apple9_vir_emit_texture_lod(&p, coords, zero, 0, 0, true);
   auto successor = agx_apple9_block_create(&p);
   agx_apple9_block_begin(&p, successor);
   agx_apple9_vir_emit_texture_lod(&p, coords, zero, 1, 0, true);
   agx_apple9_vir_emit_texture_lod(&p, coords, zero, 2, 0, true);
   p.output = first;
   const char *reason = nullptr;
   ASSERT_TRUE(agx_apple9_allocate_publications(&p, &reason)) << reason;
   unsigned copies = 0;
   for (unsigned i = 0; i < p.instruction_count; ++i) {
      auto ins = p.instructions[i];
      if (ins->publication_handoff) {
         EXPECT_EQ(agx_apple9_instr_block(ins), entry);
         EXPECT_EQ(ins->src[0], first + copies);
         ++copies;
      }
   }
   EXPECT_EQ(copies, 4u);
   EXPECT_NE(p.output, first);
   agx_apple9_vir_finish(&p);
}

TEST(Apple9SharedRa, RematerializationPreservesHighOperandConstraints)
{
   void *memctx = ralloc_context(nullptr);
   auto ctx = rzalloc(memctx, agx_context);
   list_inithead(&ctx->blocks);
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "high_bank_rematerialization");
   agx_shader_key key = {};
   key.has_scratch = true;
   ctx->nir = nir.shader;
   ctx->stage = MESA_SHADER_COMPUTE;
   ctx->key = &key;
   ctx->ra_target.max_registers = 192;
   ctx->ra_target.reserved_registers = 8;
   ctx->ra_target.late_kill_sources = true;
   ctx->ra_target.defer_spill_lowering = true;
   auto block = agx_test_block(ctx);
   auto b = agx_init_builder(ctx, agx_after_block(block));
   agx_index values[72];
   uint32_t expected = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      uint32_t value = i * 0x9e3779b9u + 0xa511e9b3u;
      values[i] = agx_mov_imm(&b, 32, value);
      expected += value;
   }
   auto sum = agx_mov_imm(&b, 32, 0);
   std::vector<agx_instr *> constrained;
   for (auto value : values) {
      auto I = agx_iadd_to(&b, agx_temp(ctx, AGX_SIZE_32), value, sum, 0);
      I->reg_constraints = rzalloc_array(ctx, agx_reg_constraint, 3);
      I->reg_constraints[0] = {0, 190, 2};
      I->reg_constraints[1] = {128, 190, 2};
      I->reg_constraints[2] = {0, 190, 2};
      constrained.push_back(I);
      sum = I->dest[0];
   }
   agx_export(&b, sum, 8);
   agx_ra(ctx);
   EXPECT_EQ(ctx->scratch_size_B, 0u);
   for (auto I : constrained) {
      EXPECT_GE(I->src[0].value, 128u);
      EXPECT_LE(I->src[0].value, 190u);
   }
   uint32_t regs[192] = {};
   agx_foreach_instr_in_block(block, I) {
      auto read = [&](agx_index x) {
         return x.type == AGX_INDEX_IMMEDIATE ? x.value : regs[x.value];
      };
      switch (I->op) {
      case AGX_OPCODE_MOV_IMM:
         regs[I->dest[0].value] = I->imm;
         break;
      case AGX_OPCODE_IADD:
         regs[I->dest[0].value] = read(I->src[0]) + read(I->src[1]);
         break;
      case AGX_OPCODE_MOV:
         regs[I->dest[0].value] = read(I->src[0]);
         break;
      case AGX_OPCODE_SWAP:
         std::swap(regs[I->src[0].value], regs[I->src[1].value]);
         break;
      case AGX_OPCODE_EXPORT:
         regs[I->imm] = read(I->src[0]);
         break;
      default:
         FAIL() << "unexpected opcode " << I->op;
      }
   }
   EXPECT_EQ(regs[8], expected);
   ralloc_free(nir.shader);
   ralloc_free(memctx);
}
