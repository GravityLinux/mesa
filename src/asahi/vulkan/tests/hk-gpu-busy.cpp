/* SPDX-License-Identifier: MIT */
#include <algorithm>
#include <array>
#include <random>
#include <gtest/gtest.h>
#include "hk_gpu_busy_intervals.h"

namespace {
struct Ranges : hk_busy_ranges {
   Ranges(): hk_busy_ranges{}
   {
   }
   ~Ranges()
   {
      free(data);
   }
};
} // namespace

TEST(HKGpuBusy, OverlapAcrossEnginesAndOutOfOrder)
{
   Ranges all, vertex, fragment, compute;
   EXPECT_TRUE(hk_busy_union_add(&fragment, 200, 700));
   EXPECT_TRUE(hk_busy_union_add(&all, 200, 700));
   EXPECT_TRUE(hk_busy_union_add(&vertex, 100, 300));
   EXPECT_TRUE(hk_busy_union_add(&all, 100, 300));
   EXPECT_TRUE(hk_busy_union_add(&compute, 400, 600));
   EXPECT_TRUE(hk_busy_union_add(&all, 400, 600));
   EXPECT_EQ(hk_busy_union_size(&all), 600u);
   EXPECT_EQ(hk_busy_union_size(&vertex), 200u);
   EXPECT_EQ(hk_busy_union_size(&fragment), 500u);
   EXPECT_EQ(hk_busy_union_size(&compute), 200u);
   EXPECT_EQ(all.count, 1u);
}

TEST(HKGpuBusy, DuplicatesAdjacencyAndEmpty)
{
   Ranges r;
   EXPECT_TRUE(hk_busy_union_add(&r, 10, 20));
   EXPECT_TRUE(hk_busy_union_add(&r, 10, 20));
   EXPECT_TRUE(hk_busy_union_add(&r, 20, 40));
   EXPECT_TRUE(hk_busy_union_add(&r, 5, 10));
   EXPECT_TRUE(hk_busy_union_add(&r, 99, 99));
   EXPECT_TRUE(hk_busy_union_add(&r, 100, 99));
   EXPECT_EQ(hk_busy_union_size(&r), 35u);
   EXPECT_EQ(r.count, 1u);
   hk_busy_union_clear(&r);
   EXPECT_EQ(hk_busy_union_size(&r), 0u);
   EXPECT_FALSE(r.overflow);
}

TEST(HKGpuBusy, RandomizedBitmapOracle)
{
   std::mt19937 rng(0x8132);
   for (unsigned trial = 0; trial < 50; ++trial) {
      Ranges r;
      std::array<bool, 1000> busy{};
      for (unsigned sample = 0; sample < 2000; ++sample) {
         unsigned start = rng() % 1000;
         unsigned end = std::min(1000u, start + unsigned(rng() % 40));
         EXPECT_TRUE(hk_busy_union_add(&r, start, end));
         std::fill(busy.begin() + start, busy.begin() + end, true);
         uint64_t expected = std::count(busy.begin(), busy.end(), true);
         ASSERT_EQ(hk_busy_union_size(&r), expected);
         ASSERT_LE(hk_busy_union_size(&r), 1000u);
      }
   }
}

TEST(HKGpuBusy, BoundedStorageMarksMissingCoverage)
{
   Ranges r;
   for (unsigned i = 0; i < HK_BUSY_MAX_RANGES; ++i)
      ASSERT_TRUE(hk_busy_union_add(&r, i * 4, i * 4 + 1));
   EXPECT_FALSE(hk_busy_union_add(&r, HK_BUSY_MAX_RANGES * 4,
                                  HK_BUSY_MAX_RANGES * 4 + 1));
   EXPECT_TRUE(r.overflow);
   EXPECT_EQ(r.count, HK_BUSY_MAX_RANGES);
   EXPECT_EQ(hk_busy_union_size(&r), uint64_t(HK_BUSY_MAX_RANGES));
   /* A later overlapping interval can still be represented, without hiding
    * the fact that an earlier disjoint interval was omitted. */
   EXPECT_TRUE(hk_busy_union_add(&r, 0, HK_BUSY_MAX_RANGES * 4));
   EXPECT_EQ(r.count, 1u);
   EXPECT_TRUE(r.overflow);
   hk_busy_union_clear(&r);
   EXPECT_FALSE(r.overflow);
}

TEST(HKGpuBusy, NanosecondScaleSecondBoundary)
{
   Ranges first, second;
   const uint64_t ns = 1000000000;
   /* Split a command crossing a GPU-clock second into half-open bin ranges. */
   EXPECT_TRUE(hk_busy_union_add(&first, ns - 250, ns));
   EXPECT_TRUE(hk_busy_union_add(&second, 0, 400));
   EXPECT_TRUE(hk_busy_union_add(&second, 300, 800));
   EXPECT_EQ(hk_busy_union_size(&first), 250u);
   EXPECT_EQ(hk_busy_union_size(&second), 800u);
   EXPECT_TRUE(hk_busy_union_add(&second, 0, ns));
   EXPECT_EQ(hk_busy_union_size(&second), ns);
}

TEST(HKGpuBusy, CS2CardinalityAndThunkHeadroom)
{
   /* The 60 s baseline had 78,295 command interval pairs for 289 presents
    * (~271/frame), with a 2,141-pair GPU-clock second. Every pair may be
    * disjoint, so test cardinality rather than assuming engine overlap.
    * 8,192 ranges also covers ~30 frames at the observed mean pairs/frame. */
   Ranges r;
   for (unsigned i = 0; i < 8192; ++i) {
      ASSERT_TRUE(hk_busy_union_add(&r, 2 * i, 2 * i + 1));
      if (i + 1 == 2141 || i + 1 == 4096 || i + 1 == 8192) {
         EXPECT_EQ(hk_busy_union_size(&r), uint64_t(i + 1));
         EXPECT_EQ(r.count, i + 1);
         EXPECT_FALSE(r.overflow);
      }
   }
}
