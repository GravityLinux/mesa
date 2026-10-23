/* SPDX-License-Identifier: MIT */
#include "../hk_sync.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

class Sync : public testing::Test {
protected:
   hk_sync_state state;
   std::vector<hk_sync_stream> streams;
   std::vector<uint32_t> remap;
   std::vector<hk_sync_event> events;
   uint32_t next[2] = {};

   void SetUp() override { hk_sync_init(&state); }

   unsigned stream(hk_sync_engine engine)
   {
      unsigned id = streams.size();
      hk_sync_stream s = {};
      s.wait = hk_sync_empty();
      s.wait_tiling = hk_sync_empty();
      s.before[0] = next[0];
      s.before[1] = next[1];
      s.engine = engine;
      s.index = ++next[engine];
      streams.push_back(s);
      remap.push_back(id);
      return id;
   }

   void work(unsigned id, unsigned stages)
   {
      events.push_back({id, 0, stages});
   }

   void barrier(unsigned src, unsigned dst)
   {
      events.push_back({HK_SYNC_NONE, src, dst});
   }

   bool resolve(bool fallback = false)
   {
      return hk_sync_resolve(&state, events.data(), events.size(), remap.data(),
                             streams.data(), streams.size(), fallback);
   }

   void expect(unsigned id, uint32_t render, uint32_t compute)
   {
      EXPECT_EQ(streams[id].wait.engine[HK_SYNC_RENDER], render);
      EXPECT_EQ(streams[id].wait.engine[HK_SYNC_CDM], compute);
   }

   /* Render dependency needed before tiling; the rest of wait can be
    * deferred to the fragment stage.
    */
   void expect_tiling(unsigned id, uint32_t render)
   {
      EXPECT_EQ(streams[id].wait_tiling.engine[HK_SYNC_RENDER], render);
   }

   void vk_barrier(VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst)
   {
      barrier(hk_sync_stage_mask(src, true), hk_sync_stage_mask(dst, false));
   }

   /* Two render passes with a barrier between them. */
   void two_passes(VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst)
   {
      unsigned a = stream(HK_SYNC_RENDER);
      work(a, HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT);
      vk_barrier(src, dst);
      unsigned b = stream(HK_SYNC_RENDER);
      work(b, HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT);
      ASSERT_TRUE(resolve());
   }
};

TEST_F(Sync, FragmentConsumerDefersToFragment)
{
   two_passes(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
   expect(1, 1, HK_SYNC_NONE);
   expect_tiling(1, HK_SYNC_NONE);
}

TEST_F(Sync, AttachmentReuseDefersToFragment)
{
   two_passes(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT);
   expect(1, 1, HK_SYNC_NONE);
   expect_tiling(1, HK_SYNC_NONE);
}

TEST_F(Sync, VertexConsumerWaitsBeforeTiling)
{
   two_passes(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
              VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
   expect(1, 1, HK_SYNC_NONE);
   expect_tiling(1, 1);
}

TEST_F(Sync, VertexInputAndIndirectConsumersWaitBeforeTiling)
{
   two_passes(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
              VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT);
   expect_tiling(1, 1);
   SetUp();
   streams.clear(), remap.clear(), events.clear(), next[0] = next[1] = 0;
   two_passes(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
              VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
   expect_tiling(1, 1);
}

TEST_F(Sync, WriteAfterReadInTilingWaitsBeforeTiling)
{
   /* A later pass's vertex stage overwrites what an earlier fragment stage
    * read: an execution-only dependency, but tiling must still wait.
    */
   two_passes(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
              VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT);
   expect_tiling(1, 1);
}

TEST_F(Sync, AllGraphicsAndAllCommandsWaitBeforeTiling)
{
   two_passes(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
              VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
   expect_tiling(1, 1);
   SetUp();
   streams.clear(), remap.clear(), events.clear(), next[0] = next[1] = 0;
   two_passes(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
   expect_tiling(1, 1);
}

TEST_F(Sync, ComputeToFragmentKeepsComputeWaitButNoRenderWait)
{
   unsigned c = stream(HK_SYNC_CDM), r = stream(HK_SYNC_RENDER);
   work(c, HK_SYNC_COMPUTE);
   vk_barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
   work(r, HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT);
   ASSERT_TRUE(resolve());
   expect(r, HK_SYNC_NONE, 1);
   expect_tiling(r, HK_SYNC_NONE);
}

TEST_F(Sync, HelpersBetweenPassesWaitBeforeTiling)
{
   unsigned a = stream(HK_SYNC_RENDER), helper = stream(HK_SYNC_CDM);
   unsigned b = stream(HK_SYNC_RENDER);
   work(a, HK_SYNC_GRAPHICS);
   work(helper, HK_SYNC_ALL);
   work(b, HK_SYNC_GRAPHICS);
   ASSERT_TRUE(resolve());
   expect(b, 1, 1);
   expect_tiling(b, 1);
}

TEST_F(Sync, FallbackWaitsBeforeTiling)
{
   unsigned helper = stream(HK_SYNC_CDM), r = stream(HK_SYNC_RENDER);
   unsigned r2 = stream(HK_SYNC_RENDER);
   work(r, HK_SYNC_GRAPHICS);
   work(helper, HK_SYNC_ALL);
   work(r2, HK_SYNC_GRAPHICS);
   ASSERT_FALSE(resolve());
   expect(r2, 1, 1);
   expect_tiling(r2, 1);
}

TEST_F(Sync, IndependentEngines)
{
   unsigned c = stream(HK_SYNC_CDM), r = stream(HK_SYNC_RENDER);
   work(c, HK_SYNC_COMPUTE);
   work(r, HK_SYNC_GRAPHICS);
   ASSERT_TRUE(resolve());
   expect(c, HK_SYNC_NONE, HK_SYNC_NONE);
   expect(r, HK_SYNC_NONE, HK_SYNC_NONE);
}

TEST_F(Sync, ComputeToRenderAndRenderToCompute)
{
   unsigned a = stream(HK_SYNC_CDM), b = stream(HK_SYNC_RENDER);
   unsigned independent = stream(HK_SYNC_CDM), d = stream(HK_SYNC_CDM);
   work(a, HK_SYNC_COMPUTE);
   barrier(HK_SYNC_COMPUTE, HK_SYNC_GRAPHICS);
   work(b, HK_SYNC_GRAPHICS);
   work(independent, HK_SYNC_COMPUTE);
   barrier(HK_SYNC_GRAPHICS, HK_SYNC_COMPUTE);
   work(d, HK_SYNC_COMPUTE);
   ASSERT_TRUE(resolve());
   expect(b, HK_SYNC_NONE, 1);
   expect(independent, HK_SYNC_NONE, HK_SYNC_NONE);
   /* The compute producer is inherited through the dependency chain. */
   expect(d, 1, 1);
}

TEST_F(Sync, IndependentBarrierPairsDoNotCross)
{
   unsigned c = stream(HK_SYNC_CDM), r = stream(HK_SYNC_RENDER);
   unsigned c2 = stream(HK_SYNC_CDM), r2 = stream(HK_SYNC_RENDER);
   work(c, HK_SYNC_COMPUTE);
   work(r, HK_SYNC_GRAPHICS);
   barrier(HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
   barrier(HK_SYNC_GRAPHICS, HK_SYNC_GRAPHICS);
   work(c2, HK_SYNC_COMPUTE);
   work(r2, HK_SYNC_GRAPHICS);
   ASSERT_TRUE(resolve());
   expect(c2, HK_SYNC_NONE, 1);
   expect(r2, 1, HK_SYNC_NONE);
}

TEST_F(Sync, TransfersTrackTheirActualEngine)
{
   unsigned c = stream(HK_SYNC_CDM), r = stream(HK_SYNC_RENDER);
   unsigned read = stream(HK_SYNC_CDM);
   work(c, HK_SYNC_TRANSFER);
   work(r, HK_SYNC_TRANSFER);
   barrier(HK_SYNC_TRANSFER, HK_SYNC_COMPUTE);
   work(read, HK_SYNC_COMPUTE);
   ASSERT_TRUE(resolve());
   expect(read, 1, 1);
}

TEST_F(Sync, BarriersInOneDependencyInfoDoNotChain)
{
   unsigned c = stream(HK_SYNC_CDM), r = stream(HK_SYNC_RENDER);
   unsigned transfer = stream(HK_SYNC_CDM);
   work(c, HK_SYNC_COMPUTE);
   work(r, HK_SYNC_GRAPHICS);
   barrier(HK_SYNC_COMPUTE, HK_SYNC_GRAPHICS);
   events.back().barrier_group = 1;
   barrier(HK_SYNC_GRAPHICS, HK_SYNC_TRANSFER);
   events.back().barrier_group = 1;
   work(transfer, HK_SYNC_TRANSFER);
   ASSERT_TRUE(resolve());
   expect(transfer, 1, HK_SYNC_NONE);
}

TEST_F(Sync, BarrierOnlyCommandBufferAndChain)
{
   unsigned c = stream(HK_SYNC_CDM);
   work(c, HK_SYNC_COMPUTE);
   ASSERT_TRUE(resolve());
   events.clear();
   barrier(HK_SYNC_COMPUTE, HK_SYNC_GRAPHICS);
   barrier(HK_SYNC_GRAPHICS, HK_SYNC_TRANSFER);
   ASSERT_TRUE(resolve());
   events.clear();
   unsigned read = stream(HK_SYNC_RENDER);
   work(read, HK_SYNC_TRANSFER);
   ASSERT_TRUE(resolve());
   expect(read, HK_SYNC_NONE, 1);
}

TEST_F(Sync, BarrierScopeSurvivesNextSubmit)
{
   unsigned c = stream(HK_SYNC_CDM);
   work(c, HK_SYNC_COMPUTE);
   barrier(HK_SYNC_COMPUTE, HK_SYNC_GRAPHICS);
   ASSERT_TRUE(resolve());
   hk_sync_next_submit(&state);
   events.clear();
   unsigned r = stream(HK_SYNC_RENDER);
   work(r, HK_SYNC_GRAPHICS);
   ASSERT_TRUE(resolve());
   expect(r, HK_SYNC_NONE, 0);
}

TEST_F(Sync, MergedStreamsPreserveBarrierAndOutgoingFrontier)
{
   unsigned c = stream(HK_SYNC_CDM), merged = stream(HK_SYNC_CDM);
   unsigned r = stream(HK_SYNC_RENDER);
   streams[merged].index = 0;
   remap[merged] = c;
   streams[r].before[HK_SYNC_CDM] = 1;
   work(c, HK_SYNC_COMPUTE);
   barrier(HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
   work(merged, HK_SYNC_COMPUTE);
   barrier(HK_SYNC_COMPUTE, HK_SYNC_GRAPHICS);
   work(r, HK_SYNC_GRAPHICS);
   ASSERT_TRUE(resolve());
   expect(c, HK_SYNC_NONE, HK_SYNC_NONE);
   expect(r, HK_SYNC_NONE, 1);
}

TEST_F(Sync, RemovedEmptyStream)
{
   unsigned gone = stream(HK_SYNC_RENDER), c = stream(HK_SYNC_CDM);
   streams[gone].index = 0;
   remap[gone] = HK_SYNC_NONE;
   streams[c].before[HK_SYNC_RENDER] = 0;
   work(gone, HK_SYNC_GRAPHICS);
   work(c, HK_SYNC_TRANSFER);
   ASSERT_TRUE(resolve());
   expect(c, HK_SYNC_NONE, HK_SYNC_NONE);
}

TEST_F(Sync, HelpersOrderTheirConsumers)
{
   unsigned r = stream(HK_SYNC_RENDER), helper = stream(HK_SYNC_CDM);
   unsigned read = stream(HK_SYNC_RENDER);
   work(r, HK_SYNC_GRAPHICS);
   work(helper, HK_SYNC_ALL);
   work(read, HK_SYNC_GRAPHICS);
   ASSERT_TRUE(resolve());
   expect(helper, 1, HK_SYNC_NONE);
   expect(read, 1, 1);
}

TEST_F(Sync, ReorderedHelpersFallBackWithoutForwardWaits)
{
   unsigned helper = stream(HK_SYNC_CDM), r = stream(HK_SYNC_RENDER);
   /* Recording order differs from physical order. */
   work(r, HK_SYNC_GRAPHICS);
   work(helper, HK_SYNC_ALL);
   ASSERT_FALSE(resolve());
   expect(helper, 0, 0);
   expect(r, 0, 1);
}

TEST_F(Sync, RepeatedResolutionDoesNotMutateRecording)
{
   unsigned c = stream(HK_SYNC_CDM), r = stream(HK_SYNC_RENDER);
   work(c, HK_SYNC_COMPUTE);
   barrier(HK_SYNC_COMPUTE, HK_SYNC_GRAPHICS);
   work(r, HK_SYNC_GRAPHICS);
   auto original = events;
   ASSERT_TRUE(resolve());
   hk_sync_init(&state);
   for (auto &s : streams)
      s.wait = hk_sync_empty();
   ASSERT_TRUE(resolve());
   expect(r, HK_SYNC_NONE, 1);
   ASSERT_EQ(events.size(), original.size());
   for (unsigned i = 0; i < events.size(); ++i) {
      EXPECT_EQ(events[i].stream, original[i].stream);
      EXPECT_EQ(events[i].src, original[i].src);
      EXPECT_EQ(events[i].dst, original[i].dst);
   }
}

TEST(SyncScopes, NoneTopBottomAndAliases)
{
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_NONE, true), 0u);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, true), 0u);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, false), HK_SYNC_ALL);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, true), HK_SYNC_ALL);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, false), 0u);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, true), HK_SYNC_ALL);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, false), HK_SYNC_TRANSFER);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, true),
             HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, false),
             HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT);
}

TEST(SyncScopes, FragmentAndPreRasterScopes)
{
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, false),
             HK_SYNC_FRAGMENT);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, false),
             HK_SYNC_FRAGMENT);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, false),
             HK_SYNC_GRAPHICS);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, true),
             HK_SYNC_PRE_RASTER | HK_SYNC_DRAW_INDIRECT);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT, false),
             HK_SYNC_GRAPHICS);
}

TEST(SyncScopes, PipelineOrderingAndUnknownStages)
{
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, true),
             HK_SYNC_COMPUTE | HK_SYNC_DISPATCH_INDIRECT);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, true),
             HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, false),
             HK_SYNC_INDIRECT | HK_SYNC_COMPUTE | HK_SYNC_GRAPHICS);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_HOST_BIT, true), HK_SYNC_ALL);
   EXPECT_EQ(hk_sync_stage_mask(VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT, true),
             HK_SYNC_ALL);
   EXPECT_EQ(hk_sync_stage_mask(UINT64_C(1) << 63, false), HK_SYNC_ALL);
}

TEST(SyncSlicing, IncomingLocalNoneAndWideIndices)
{
   EXPECT_EQ(hk_sync_rebase(HK_SYNC_NONE, 64), HK_SYNC_NONE);
   EXPECT_EQ(hk_sync_rebase(0, 64), 0u);
   EXPECT_EQ(hk_sync_rebase(2, 64), 0u);
   EXPECT_EQ(hk_sync_rebase(64, 64), 0u);
   EXPECT_EQ(hk_sync_rebase(65, 64), 1u);
   EXPECT_EQ(hk_sync_rebase(100001, 100000), 1u);
}

TEST(SyncScopes, ExecutionOnlyAndOneSidedMemoryScopes)
{
   EXPECT_EQ(hk_sync_ordered_stages(HK_SYNC_COMPUTE, HK_SYNC_GRAPHICS, 0, 0),
             HK_SYNC_COMPUTE | HK_SYNC_GRAPHICS);
   EXPECT_EQ(hk_sync_ordered_stages(0, HK_SYNC_COMPUTE, 0, 0), 0u);
   EXPECT_EQ(hk_sync_ordered_stages(HK_SYNC_COMPUTE, 0, 0, 0), 0u);
   EXPECT_EQ(hk_sync_ordered_stages(HK_SYNC_COMPUTE, 0,
                                   VK_ACCESS_2_SHADER_WRITE_BIT, 0), HK_SYNC_COMPUTE);
   EXPECT_EQ(hk_sync_ordered_stages(0, HK_SYNC_COMPUTE, 0,
                                   VK_ACCESS_2_SHADER_READ_BIT), HK_SYNC_COMPUTE);
}

/* Compare the recording index with an independent linear oracle after every
 * insertion/move, including repeated work declarations and barriers. */
TEST(SyncIndex, ReorderedRecording)
{
   std::vector<hk_sync_event> events;
   std::vector<uint32_t> first(16, HK_SYNC_NONE);
   uint32_t random = 0x8132;
   auto next = [&]() { return random = random * 1664525u + 1013904223u; };

   for (unsigned step = 0; step < 2000; ++step) {
      unsigned old_end = events.size();
      unsigned count = 1 + next() % 4;
      for (unsigned j = 0; j < count; ++j) {
         uint32_t stream = (next() % 5 == 0) ? HK_SYNC_NONE : next() % 16;
         hk_sync_index_event(first.data(), stream, events.size());
         events.push_back({stream, 0, HK_SYNC_COMPUTE});
      }

      /* Hoist the newly recorded tail to an arbitrary earlier boundary,
       * matching both barrier insertion and merge-pre stream placement. */
      unsigned at = next() % (old_end + 1);
      std::rotate(events.begin() + at, events.begin() + old_end, events.end());
      hk_sync_reindex_events(first.data(), events.data(), at, events.size());

      std::vector<uint32_t> expected(first.size(), HK_SYNC_NONE);
      for (unsigned i = 0; i < events.size(); ++i) {
         uint32_t stream = events[i].stream;
         if (stream != HK_SYNC_NONE && expected[stream] == HK_SYNC_NONE)
            expected[stream] = i;
      }
      ASSERT_EQ(first, expected) << "after recording step " << step;
   }
}

TEST(SyncIndex, MergeAndRemove)
{
   uint32_t first[] = {10, 2, 7, HK_SYNC_NONE};
   /* Physical CDM ordering can differ from event order. */
   hk_sync_remap_first_event(first, 1, 0);
   EXPECT_EQ(first[0], 2u);
   EXPECT_EQ(first[1], HK_SYNC_NONE);
   hk_sync_remap_first_event(first, 2, 0);
   EXPECT_EQ(first[0], 2u);
   EXPECT_EQ(first[2], HK_SYNC_NONE);
   hk_sync_remap_first_event(first, 3, 0);
   EXPECT_EQ(first[0], 2u);
   hk_sync_remap_first_event(first, 0, HK_SYNC_NONE);
   EXPECT_EQ(first[0], HK_SYNC_NONE);

   /* Removing an empty render leaves its event in the log; a later move
    * must not make the removed stream live again. */
   struct hk_sync_event events[] = {{HK_SYNC_NONE, 0, 0}, {0, 0, 0}};
   hk_sync_reindex_events(first, events, 0, 2);
   EXPECT_EQ(first[0], HK_SYNC_NONE);
}
