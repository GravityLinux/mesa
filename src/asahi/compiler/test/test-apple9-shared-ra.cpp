/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include "compiler/nir/nir_builder.h"
#include <gtest/gtest.h>
#include "agx_test.h"

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
