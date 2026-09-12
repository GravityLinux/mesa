/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <array>
#include <cstring>
#include "gtest/gtest.h"

extern "C" {
#include "agx_apple9_launch.h"
#include "asahi/compiler/agx_apple9_profile.h"
}

static agx_apple9_launch_parameters
parameters()
{
   agx_apple9_launch_parameters p = {};
   p.shader_base = 0x10000000000ull;
   p.resource_table = p.shader_base + 0x208000;
   p.state = p.shader_base + 0x308020;
   p.entry_offset = 0x12340;
   p.publication_count = 9;
   p.frame_extent_a = 4;
   p.frame_extent_b = 4096;
   p.tile_bytes = 24;
   p.samples = 1;
   p.resource_count = 8;
   return p;
}

static unsigned
read16(const uint8_t *p)
{
   return p[0] | (p[1] << 8);
}

TEST(Apple9Launcher, AllStagesUseLiveEntryAndIndependentFrameExtents)
{
   auto p = parameters();
   for (auto stage : {AGX_APPLE9_LAUNCH_VERTEX, AGX_APPLE9_LAUNCH_FRAGMENT,
                      AGX_APPLE9_LAUNCH_COMPUTE}) {
      std::array<uint8_t, 1024> out;
      out.fill(0xa5);
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), stage, &p));
      unsigned call = agx_apple9_launch_call_offset(stage, p.resource_count);
      unsigned entry = out[call] | (out[call + 1] << 8) | (out[call + 2] << 16);
      EXPECT_EQ(entry, 0x246aau); /* Entry above the former 64 KiB boundary. */
      EXPECT_EQ(read16(out.data() + call + 11), 0x280u);
      EXPECT_EQ(read16(out.data() + call + 13), 4u);
      EXPECT_EQ(read16(out.data() + call + 15), 4096u);
      EXPECT_EQ(out[call + 17], 0);
      EXPECT_EQ(out.back(), 0);
   }
}

TEST(Apple9Launcher, ResourceCountsRespectAllocationBounds)
{
   auto p = parameters();
   std::array<uint8_t, 1056> guarded;
   for (unsigned resources = 1; resources <= 18; ++resources) {
      p.resource_count = resources;
      guarded.fill(0xa5);
      ASSERT_TRUE(agx_apple9_launch_build(guarded.data() + 16, 1024,
                                          AGX_APPLE9_LAUNCH_COMPUTE, &p));
      for (unsigned i = 0; i < 16; ++i) {
         EXPECT_EQ(guarded[i], 0xa5);
         EXPECT_EQ(guarded[1040 + i], 0xa5);
      }
      auto *out = guarded.data() + 16;
      unsigned call =
         agx_apple9_launch_call_offset(AGX_APPLE9_LAUNCH_COMPUTE, resources);
      EXPECT_EQ(read16(out + call + 13), p.frame_extent_a);
      EXPECT_EQ(read16(out + call + 15), p.frame_extent_b);
      /* The final state load reaches pending r56..r59 at the resource limit. */
      EXPECT_EQ(out[call - 21], 2 * (18 + 2 * (resources + AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE)));
   }
}

TEST(Apple9Launcher, InvalidParametersCannotPublishPartialCode)
{
   auto p = parameters();
   auto reject = [&](agx_apple9_launch_parameters bad,
                     agx_apple9_launch_stage stage = AGX_APPLE9_LAUNCH_COMPUTE,
                     size_t capacity = 1024) {
      std::array<uint8_t, 1024> out;
      out.fill(0xa5);
      const auto saved = out;
      EXPECT_FALSE(agx_apple9_launch_build(out.data(), capacity, stage, &bad));
      EXPECT_EQ(out, saved);
   };
   reject(p, AGX_APPLE9_LAUNCH_COMPUTE, 1023);
   reject(p, AGX_APPLE9_LAUNCH_VERTEX, 255);
   reject(p, static_cast<agx_apple9_launch_stage>(3));
   auto bad = p;
   bad.resource_table = p.shader_base - 1;
   reject(bad);
   bad.resource_table = p.shader_base + 0x20000000;
   reject(bad);
   bad = p;
   bad.state--;
   reject(bad);
   bad = p;
   bad.entry_offset++;
   reject(bad);
   bad.entry_offset = 0x1ffec;
   reject(bad);
   bad.entry_offset = UINT32_MAX;
   reject(bad);
   bad = p;
   bad.publication_count = 1023;
   reject(bad);
   bad = p;
   for (unsigned count : {0u, 19u, UINT32_MAX}) {
      bad.resource_count = count;
      reject(bad);
   }
   bad = p;
   for (unsigned bytes : {1u, 64u, 129u, 384u, 2048u, UINT32_MAX}) {
      bad.threadgroup_memory_bytes = bytes;
      reject(bad);
   }
   bad.threadgroup_memory_bytes = 128;
   reject(bad, AGX_APPLE9_LAUNCH_VERTEX);
   reject(bad, AGX_APPLE9_LAUNCH_FRAGMENT);
   bad = p;
   for (unsigned samples : {0u, 3u, 8u}) {
      bad.samples = samples;
      reject(bad, AGX_APPLE9_LAUNCH_FRAGMENT);
   }
   bad = p;
   for (unsigned bytes : {0u, 129u}) {
      bad.tile_bytes = bytes;
      reject(bad, AGX_APPLE9_LAUNCH_FRAGMENT);
   }
}

TEST(Apple9Launcher, PublicationCapacityCoversPairBoundaries)
{
   auto p = parameters();
   const unsigned counts[] = {0, 1, 2, 8, 9, 36, 1022};
   const unsigned words[] = {0, 0x80, 0x80, 0x200, 0x280, 0x900, 0xff80};
   std::array<uint8_t, 256> out;
   for (unsigned i = 0; i < 7; ++i) {
      p.publication_count = counts[i];
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                          AGX_APPLE9_LAUNCH_VERTEX, &p));
      EXPECT_EQ(read16(out.data() + 68 + 11), words[i]);
   }
}

TEST(Apple9Launcher, ThreadgroupAllocationDoesNotChangeEntryOrFrame)
{
   auto p = parameters();
   std::array<uint8_t, 1024> initial, out;
   ASSERT_TRUE(agx_apple9_launch_build(initial.data(), initial.size(),
                                       AGX_APPLE9_LAUNCH_COMPUTE, &p));
   unsigned call = agx_apple9_launch_call_offset(AGX_APPLE9_LAUNCH_COMPUTE,
                                                 p.resource_count);
   for (unsigned bytes : {128, 256, 512, 1024}) {
      p.threadgroup_memory_bytes = bytes;
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                          AGX_APPLE9_LAUNCH_COMPUTE, &p));
      EXPECT_EQ(read16(out.data() + call - 6), (bytes << 2) | 0x80);
      EXPECT_EQ(out[call - 4], 0);
      EXPECT_EQ(out[call - 3], 0);
      /* Only the allocation operand changes. */
      out[call - 6] = initial[call - 6];
      out[call - 5] = initial[call - 5];
      EXPECT_EQ(out, initial);
   }
}

TEST(Apple9Launcher, TileModesDoNotChangeEntryOrResources)
{
   auto p = parameters();
   std::array<uint8_t, 256> initial, out;
   ASSERT_TRUE(agx_apple9_launch_build(initial.data(), initial.size(),
                                       AGX_APPLE9_LAUNCH_FRAGMENT, &p));
   struct {
      unsigned samples, bytes, mode, low;
   } cases[] = {{1, 8, 0x4b, 0x43}, {2, 24, 0x4b, 0x46}, {4, 64, 0x43, 16},
               {1, 128, 0x4b, 0x61}};
   for (auto c : cases) {
      p.samples = c.samples;
      p.tile_bytes = c.bytes;
      ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(),
                                          AGX_APPLE9_LAUNCH_FRAGMENT, &p));
      EXPECT_EQ(out[68 - 7], c.low);
      EXPECT_EQ(out[68 - 6], c.mode);
      memcpy(out.data() + 68 - 10, initial.data() + 68 - 10, 8);
      EXPECT_EQ(out, initial);
   }
}
