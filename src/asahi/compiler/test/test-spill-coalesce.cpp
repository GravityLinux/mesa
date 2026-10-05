/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"

class SpillCoalesce : public testing::Test {
 protected:
   void *mem = ralloc_context(nullptr);
   agx_builder *b = agx_test_builder(mem);
   std::vector<std::pair<unsigned, unsigned>> homes;

   ~SpillCoalesce() { ralloc_free(mem); }

   agx_index value(unsigned reg, bool memory = false, unsigned home = 0,
                   enum agx_size size = AGX_SIZE_32, unsigned channels = 1)
   {
      agx_index x = agx_temp(b->shader, size);
      x.reg = reg;
      x.has_reg = true;
      x.memory = memory;
      x.channels_m1 = channels - 1;
      if (memory)
         homes.emplace_back(x.value, home);
      return x;
   }

   agx_index imm(unsigned reg, unsigned number)
   {
      auto x = value(reg);
      agx_mov_imm_to(b, x, number);
      return x;
   }

   agx_instr *load(agx_index x, unsigned reg = 8)
   {
      return agx_mov_to(b, value(reg, false, 0, x.size, agx_channels(x)), x);
   }

   bool run(unsigned *half_slots)
   {
      auto ctx = b->shader;
      ctx->spill_home = rzalloc_array(ctx, unsigned, ctx->alloc);
      ctx->spill_home_count = ctx->alloc;
      for (auto [ssa, home] : homes)
         ctx->spill_home[ssa] = home;
      agx_compute_liveness(ctx);
      auto graph = agx_spill_coalesce_prepare(ctx);
      EXPECT_NE(graph, nullptr);
      if (!graph)
         return false;
      agx_validate_ra(ctx);
      bool result = agx_spill_coalesce_apply(ctx, graph, half_slots);
      agx_validate_ra(ctx);
      ralloc_free(graph);
      return result;
   }
};

TEST_F(SpillCoalesce, DiamondCoalescesFramePhi)
{
   auto A = agx_start_block(b->shader);
   auto B = agx_test_block(b->shader);
   auto C = agx_test_block(b->shader);
   auto D = agx_test_block(b->shader);
   agx_block_add_successor(A, B);
   agx_block_add_successor(A, C);
   agx_block_add_successor(B, D);
   agx_block_add_successor(C, D);
   b->cursor = agx_after_block(A);
   auto x = imm(0, 7), y = imm(2, 11);
   auto left = value(0, true, 1), right = value(2, true, 1);
   auto joined = value(4, true, 1);
   b->cursor = agx_after_block(B);
   auto save_left = agx_mov_to(b, left, x);
   b->cursor = agx_after_block(C);
   auto save_right = agx_mov_to(b, right, y);
   b->cursor = agx_after_block(D);
   auto phi = agx_phi_to(b, joined, 2);
   phi->src[0] = left;
   phi->src[1] = right;
   auto fill = load(joined);
   unsigned half_slots = 6;
   ASSERT_TRUE(run(&half_slots));
   EXPECT_EQ(half_slots, 2u);
   EXPECT_EQ(phi->dest[0].reg, phi->src[0].reg);
   EXPECT_EQ(phi->dest[0].reg, phi->src[1].reg);
   EXPECT_EQ(fill->src[0].reg, save_left->dest[0].reg);
   EXPECT_EQ(fill->src[0].reg, save_right->dest[0].reg);
}

TEST_F(SpillCoalesce, SimultaneouslyLiveVersionsRemainSeparate)
{
   auto block = agx_start_block(b->shader);
   b->cursor = agx_after_block(block);
   auto x = imm(0, 7), y = imm(2, 11);
   auto a = value(0, true, 1), c = value(2, true, 1);
   auto sa = agx_mov_to(b, a, x);
   auto sc = agx_mov_to(b, c, y);
   auto la = load(a, 4), lc = load(c, 6);
   unsigned half_slots = 4;
   ASSERT_TRUE(run(&half_slots));
   EXPECT_EQ(half_slots, 4u);
   EXPECT_NE(sa->dest[0].reg, sc->dest[0].reg);
   EXPECT_EQ(la->src[0].reg, sa->dest[0].reg);
   EXPECT_EQ(lc->src[0].reg, sc->dest[0].reg);
}

TEST_F(SpillCoalesce, DeadPhiCannotClobberLiveMemory)
{
   auto A = agx_start_block(b->shader);
   auto B = agx_test_block(b->shader);
   auto C = agx_test_block(b->shader);
   auto D = agx_test_block(b->shader);
   agx_block_add_successor(A, B);
   agx_block_add_successor(A, C);
   agx_block_add_successor(B, D);
   agx_block_add_successor(C, D);
   b->cursor = agx_after_block(A);
   auto x = imm(0, 7);
   auto live = value(0, true, 1), left = value(2, true, 2);
   auto right = value(4, true, 3), dead = value(2, true, 1);
   auto save = agx_mov_to(b, live, x);
   b->cursor = agx_after_block(B);
   agx_mov_to(b, left, x);
   b->cursor = agx_after_block(C);
   agx_mov_to(b, right, x);
   b->cursor = agx_after_block(D);
   auto phi = agx_phi_to(b, dead, 2);
   phi->src[0] = left;
   phi->src[1] = right;
   load(live);
   unsigned half_slots = 6;
   ASSERT_TRUE(run(&half_slots));
   EXPECT_NE(phi->dest[0].reg, save->dest[0].reg);
}

TEST_F(SpillCoalesce, ContractionThatNeedsMoreScratchRollsBack)
{
   auto block = agx_start_block(b->shader);
   b->cursor = agx_after_block(block);
   auto x = imm(0, 7);
   auto a = value(0, true, 1), c = value(2, true, 2);
   auto d = value(0, true, 3), e = value(2, true, 1);
   auto sa = agx_mov_to(b, a, x);
   auto sc = agx_mov_to(b, c, x);
   load(a);
   auto sd = agx_mov_to(b, d, x);
   load(c);
   auto se = agx_mov_to(b, e, x);
   load(d);
   load(e);
   // Interference is a path a-c-d-e, colorable in two words. Contracting
   // the same-home endpoints a/e forms a triangle requiring three words.
   unsigned half_slots = 4;
   EXPECT_FALSE(run(&half_slots));
   EXPECT_EQ(half_slots, 4u);
   EXPECT_EQ(sa->dest[0].reg, 0u);
   EXPECT_EQ(sc->dest[0].reg, 2u);
   EXPECT_EQ(sd->dest[0].reg, 0u);
   EXPECT_EQ(se->dest[0].reg, 2u);
}

TEST_F(SpillCoalesce, LoopPhiIncludesTheBackedge)
{
   auto A = agx_start_block(b->shader);
   auto H = agx_test_block(b->shader);
   auto B = agx_test_block(b->shader);
   auto E = agx_test_block(b->shader);
   H->loop_header = true;
   agx_block_add_successor(A, H);
   agx_block_add_successor(H, B);
   agx_block_add_successor(H, E);
   agx_block_add_successor(B, H);
   auto init = value(0, true, 1), carried = value(2, true, 1);
   auto next = value(4, true, 1);
   b->cursor = agx_after_block(A);
   agx_mov_to(b, init, imm(0, 7));
   b->cursor = agx_after_block(H);
   auto phi = agx_phi_to(b, carried, 2);
   phi->src[0] = init;
   phi->src[1] = next;
   b->cursor = agx_after_block(B);
   auto v = load(carried);
   agx_mov_to(b, next, v->dest[0]);
   b->cursor = agx_after_block(E);
   load(carried);
   unsigned half_slots = 6;
   ASSERT_TRUE(run(&half_slots));
   EXPECT_EQ(half_slots, 2u);
   EXPECT_EQ(phi->dest[0].reg, phi->src[0].reg);
   EXPECT_EQ(phi->dest[0].reg, phi->src[1].reg);
}

TEST_F(SpillCoalesce, MixedWidthsKeepAlignmentAndDoNotOverlap)
{
   auto block = agx_start_block(b->shader);
   b->cursor = agx_after_block(block);
   auto x = imm(0, 7);
   auto a = value(0, true, 1);
   auto narrow = value(4, true, 2, AGX_SIZE_16);
   auto sa = agx_mov_to(b, a, x);
   auto sn = agx_mov_to(b, narrow, agx_register(0, AGX_SIZE_16));
   load(a);
   load(narrow);
   unsigned half_slots = 5;
   ASSERT_TRUE(run(&half_slots));
   EXPECT_LE(half_slots, 5u);
   EXPECT_EQ(sa->dest[0].reg % 2, 0u);
   EXPECT_TRUE(sn->dest[0].reg < sa->dest[0].reg ||
               sn->dest[0].reg >= sa->dest[0].reg + 2);
}
