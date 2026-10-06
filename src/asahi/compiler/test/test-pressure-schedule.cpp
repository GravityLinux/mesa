/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_test.h"
#include <gtest/gtest.h>
#include <vector>

class ResourceSchedule : public testing::Test {
protected:
   void *mem = ralloc_context(nullptr);
   agx_builder *b = agx_test_builder(mem);
   ResourceSchedule() {
      b->shader->schedule_resources = [](const agx_instr *I, uint32_t *r,
                                         uint32_t *w) {
         *r = I->imm & 0xff;
         *w = (I->imm >> 8) & 0xff;
      };
      b->shader->schedule_delay = [](const agx_instr *I) -> unsigned {
         return MAX2(1, (I->imm >> 16) & 0xff);
      };
   }
   ~ResourceSchedule() { ralloc_free(mem); }
   agx_instr *emit(unsigned read, unsigned write, unsigned delay = 1,
                   agx_index source = agx_null(), bool dest = false) {
      agx_instr *I = agx_alloc_instr(b, AGX_OPCODE_APPLE9, dest,
                                     !agx_is_null(source));
      I->imm = read | (write << 8) | (delay << 16);
      if (dest) I->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
      if (!agx_is_null(source)) I->src[0] = source;
      agx_builder_insert(&b->cursor, I);
      return I;
   }
   unsigned position(agx_instr *needle) {
      unsigned pos = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I == needle) return pos;
         ++pos;
      }
      return ~0u;
   }
};

TEST_F(ResourceSchedule, WriterOrdersEveryReaderWithoutSerializingReads)
{
   auto *first = emit(0, 1);
   auto *a = emit(1, 0), *b = emit(1, 0, 8), *c = emit(1, 0);
   auto *last = emit(0, 1);
   agx_pressure_schedule(this->b->shader);
   for (auto *reader : {a, b, c}) {
      EXPECT_LT(position(first), position(reader));
      EXPECT_LT(position(reader), position(last));
   }
   /* The longer read may move ahead of the other readers. A last-reader-only
    * edge would let a writer cross one of these reads. */
   EXPECT_LT(position(b), position(a));
}

TEST_F(ResourceSchedule, StateFenceOrdersOperationsUsingOtherResources)
{
   auto *before = emit(1 | 2, 0);
   auto *publication = emit(1, 4);
   auto *fence = emit(7, 7);
   auto *after = emit(1 | 4, 0);
   agx_pressure_schedule(b->shader);
   EXPECT_LT(position(before), position(fence));
   EXPECT_LT(position(publication), position(fence));
   EXPECT_LT(position(fence), position(after));
}

TEST_F(ResourceSchedule, SeparatesLongDependencyWithIndependentWork)
{
   auto *load = emit(1, 0, 8, agx_null(), true);
   auto *consumer = emit(0, 0, 1, load->dest[0]);
   std::vector<agx_instr *> independent;
   for (unsigned i = 0; i < 5; ++i) independent.push_back(emit(2, 0));
   agx_pressure_schedule(b->shader);
   EXPECT_LT(position(load), position(consumer));
   unsigned between = 0;
   for (auto *I : independent)
      between += position(load) < position(I) && position(I) < position(consumer);
   EXPECT_GT(between, 0u);
}

/* Resource bit 1 is memory below: loads read it, stores/atomics write it,
 * and a fence reads and writes every resource. */
TEST_F(ResourceSchedule, PressureSlackOverlapsIndependentLoads)
{
   for (unsigned slack : {0u, 2u}) {
      SCOPED_TRACE(slack);
      void *ctx = ralloc_context(nullptr);
      agx_builder *saved = b;
      b = agx_test_builder(ctx);
      b->shader->schedule_resources = saved->shader->schedule_resources;
      b->shader->schedule_delay = saved->shader->schedule_delay;
      b->shader->schedule_pressure_slack = slack;
      auto *first = emit(1, 0, 8, agx_null(), true);
      auto *use_first = emit(0, 0, 1, first->dest[0]);
      auto *second = emit(1, 0, 8, agx_null(), true);
      auto *use_second = emit(0, 0, 1, second->dest[0]);
      agx_pressure_schedule(b->shader);
      EXPECT_LT(position(first), position(use_first));
      EXPECT_LT(position(second), position(use_second));
      /* Overlap keeps both results live at once: one extra register. */
      bool overlapped = std::max(position(first), position(second)) <
                        std::min(position(use_first), position(use_second));
      EXPECT_EQ(overlapped, slack != 0);
      b = saved;
      ralloc_free(ctx);
   }
}

TEST_F(ResourceSchedule, PressureSlackKeepsLoadsBehindMemoryWriters)
{
   b->shader->schedule_pressure_slack = 16;
   /* An aliasing store, an atomic and a full fence each order a later load
    * after themselves, however much latency moving the load could hide. */
   for (unsigned blocker_write : {1u, 1u | 2u, 0xffu}) {
      SCOPED_TRACE(blocker_write);
      void *ctx = ralloc_context(nullptr);
      agx_builder *saved = b;
      b = agx_test_builder(ctx);
      b->shader->schedule_resources = saved->shader->schedule_resources;
      b->shader->schedule_delay = saved->shader->schedule_delay;
      b->shader->schedule_pressure_slack = 16;
      auto *first = emit(1, 0, 8, agx_null(), true);
      auto *use_first = emit(0, 0, 1, first->dest[0]);
      auto *blocker = emit(blocker_write, blocker_write);
      auto *second = emit(1, 0, 8, agx_null(), true);
      auto *use_second = emit(0, 0, 1, second->dest[0]);
      agx_pressure_schedule(b->shader);
      EXPECT_LT(position(first), position(blocker));
      EXPECT_LT(position(blocker), position(second));
      EXPECT_LT(position(second), position(use_second));
      (void)use_first;
      b = saved;
      ralloc_free(ctx);
   }
}

TEST_F(ResourceSchedule, PhiRemainsAtBlockEntryWithTargetResources)
{
   auto value = agx_temp(b->shader, AGX_SIZE_32);
   auto *phi = agx_phi_to(b, value, 0);
   for (unsigned i = 0; i < 5; ++i) emit(2, 0);
   /* Block-entry placement remains required even with no local SSA use (for
    * example, an entry value which is only live into a successor). */
   agx_pressure_schedule(b->shader);
   EXPECT_EQ(position(phi), 0u);
}

TEST(PressureSchedule, EqualCostReductionTreesStayDepthFirst)
{
   void *mem = ralloc_context(nullptr);
   auto b = agx_test_builder(mem);
   std::vector<agx_index> level;
   for (unsigned i = 0; i < 32; ++i)
      level.push_back(agx_mov_imm(b, 32, i + 1));
   while (level.size() > 1) {
      std::vector<agx_index> next;
      for (unsigned i = 0; i < level.size(); i += 2)
         next.push_back(agx_iadd(b, level[i], level[i + 1], 0));
      level = next;
   }
   agx_export(b, level[0], 0);
   agx_pressure_schedule(b->shader);

   /* A breadth-first backwards walk opens all 32 leaves before consuming
    * them. A depth-first walk needs only one value per level plus a leaf.
    * Native comparisons exposed this tie in a large 16-bit UBO reduction. */
   std::vector<bool> live(b->shader->alloc, false);
   unsigned pressure = 0, peak = 0;
   agx_foreach_block(b->shader, block) {
      agx_foreach_instr_in_block_rev(block, I) {
         agx_foreach_ssa_dest(I, d) {
            if (live[I->dest[d].value]) {
               live[I->dest[d].value] = false;
               --pressure;
            }
         }
         agx_foreach_ssa_src(I, s) {
            if (!live[I->src[s].value]) {
               live[I->src[s].value] = true;
               ++pressure;
            }
         }
         peak = MAX2(peak, pressure);
      }
   }
   EXPECT_LE(peak, 6u);
   EXPECT_EQ(pressure, 0u);
   ralloc_free(mem);
}
