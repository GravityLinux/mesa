/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <array>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"

extern "C" {
#include "agx_apple9_launch.h"
}

/* Synthetic opaque material. These tests exercise the interface and bounds,
 * without including external executable fixtures in the Mesa source tree. */
static std::array<uint8_t, 256>
opaque_input()
{
   std::array<uint8_t, 256> data;
   data.fill(0xcc);
   data[0x44 + 8] = 0xf7;
   data[0x44 + 9] = 0;
   data[0x44 + 10] = 0x2a;
   data[0x44 - 7] = 0x43;
   return data;
}

TEST(Apple9Launcher, RelocationsPreserveOpaqueBytesAndIndependentExtents)
{
   const auto input = opaque_input();
   agx_apple9_launch_recipe recipe;
   agx_apple9_launch_parameters params;
   ASSERT_TRUE(agx_apple9_launch_import(&recipe, &params,
                                        AGX_APPLE9_LAUNCH_VERTEX, input.data(),
                                        input.size(), 0x44));
   params.shader_base = 0x1000000000ull;
   params.resource_table = params.shader_base + 0x1fffffff;
   params.main_call = 0x3ffff;
   params.publication_word = 0x280;
   params.frame_extent_a = 4;
   params.frame_extent_b = 4096;
   std::array<uint8_t, 272> output;
   output.fill(0xa5);
   ASSERT_TRUE(
      agx_apple9_launch_build(output.data(), output.size(), &recipe, &params));
   EXPECT_EQ(output[1], 0xff);
   EXPECT_EQ(output[4], 0xde);
   EXPECT_EQ(output[5], 0xcc);
   EXPECT_EQ(output[6], 0xff);
   EXPECT_EQ(output[7], 0xff);
   EXPECT_EQ(output[0x44], 0xff);
   EXPECT_EQ(output[0x45], 0xff);
   EXPECT_EQ(output[0x46], 3);
   EXPECT_EQ(output[0x4f], 0x80);
   EXPECT_EQ(output[0x50], 2);
   EXPECT_EQ(output[0x51], 4);
   EXPECT_EQ(output[0x52], 0);
   EXPECT_EQ(output[0x53], 0);
   EXPECT_EQ(output[0x54], 16);
   for (unsigned i = 0; i < input.size(); ++i) {
      if (i == 1 || (i >= 4 && i < 8) || (i >= 0x44 && i < 0x47) ||
          (i >= 0x4f && i < 0x55))
         continue;
      EXPECT_EQ(output[i], input[i]) << i;
   }
   for (unsigned i = input.size(); i < output.size(); ++i)
      EXPECT_EQ(output[i], 0) << i;
}

TEST(Apple9Launcher, ComputeRootSetupDoesNotDependOnOpaquePrefixContents)
{
   auto input = opaque_input();
   agx_apple9_launch_recipe recipe;
   agx_apple9_launch_parameters params;
   ASSERT_TRUE(agx_apple9_launch_import(&recipe, &params,
                                        AGX_APPLE9_LAUNCH_COMPUTE, input.data(),
                                        input.size(), 0x44));
   params.shader_base = 0x10000000000ull;
   params.resource_table = params.shader_base + 0x5e14a0;
   params.state = params.shader_base + 0x448020;
   params.main_call = 0x7aa;
   std::array<uint8_t, 256> first, second;
   ASSERT_TRUE(agx_apple9_launch_build(first.data(), first.size(), &recipe,
                                       &params));
   /* No caller-owned code is needed for either root's four literals. */
   memset(input.data(), 0x37, 32);
   ASSERT_TRUE(agx_apple9_launch_build(second.data(), second.size(), &recipe,
                                       &params));
   EXPECT_EQ(first, second);
   const uint8_t expected[] = {
      0x2c, 0xa0, 0x02, 0x00, 0x12, 0x08, 0xf0, 0x02,
      0x3c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x8c, 0xa0, 0x82, 0x00, 0x00, 0x00, 0x24, 0x02,
      0x9c, 0x80, 0x82, 0x00, 0x04, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(memcmp(first.data(), expected, sizeof(expected)), 0);
   for (unsigned i = 32; i < 0x44 - 7; ++i)
      EXPECT_EQ(first[i], input[i]) << i;
}

TEST(Apple9Launcher, InvalidRelocationsCannotPublishPartialOutput)
{
   const auto input = opaque_input();
   agx_apple9_launch_recipe recipe;
   agx_apple9_launch_parameters params;
   ASSERT_TRUE(agx_apple9_launch_import(&recipe, &params,
                                        AGX_APPLE9_LAUNCH_COMPUTE, input.data(),
                                        input.size(), 0x44));
   params.shader_base = 0x1000000000ull;
   params.resource_table = params.shader_base + 0x100;
   params.main_call = 0x7aa;
   params.state = params.shader_base + 0x20;
   auto rejected = [&](agx_apple9_launch_parameters bad, size_t size) {
      std::array<uint8_t, 256> output;
      output.fill(0xa5);
      const auto original = output;
      EXPECT_FALSE(agx_apple9_launch_build(output.data(), size, &recipe, &bad));
      EXPECT_EQ(output, original);
   };
   rejected(params, 255);
   auto bad = params;
   bad.state -= 1;
   rejected(bad, 256);
   bad = params;
   bad.resource_table = params.shader_base - 1;
   rejected(bad, 256);
   bad.resource_table = params.shader_base + 0x20000000;
   rejected(bad, 256);
   bad = params;
   bad.main_call = 0x40000;
   rejected(bad, 256);
   /* The call's preceding control record cannot overlap generated roots. */
   recipe.prefix.size = 48;
   rejected(params, 256);
}

TEST(Apple9Launcher, FragmentSampleLayoutDoesNotChangeAdjacentOpaqueBytes)
{
   const auto input = opaque_input();
   agx_apple9_launch_recipe recipe;
   agx_apple9_launch_parameters params;
   ASSERT_TRUE(agx_apple9_launch_import(&recipe, &params,
                                        AGX_APPLE9_LAUNCH_FRAGMENT,
                                        input.data(), input.size(), 0x44));
   params.main_call = 0x7aa;
   params.tile_bytes = 24;
   std::array<uint8_t, 256> output;
   for (unsigned samples : {1, 2, 4}) {
      params.samples = samples;
      ASSERT_TRUE(agx_apple9_launch_build(output.data(), output.size(), &recipe,
                                          &params));
      EXPECT_EQ(output[0x3d], samples == 1 ? 0x47 : samples == 2 ? 0x46 : 6);
      EXPECT_EQ(output[0x3c], samples == 4 ? 0xaa : input[0x3c]);
      EXPECT_EQ(output[0x3e], samples == 4 ? 0x43 : input[0x3e]);
      EXPECT_EQ(output[0x3b], input[0x3b]);
      EXPECT_EQ(output[0x3f], input[0x3f]);
   }
}

TEST(Apple9Launcher, PublicationBudgetUsesPairsAndPreservesUnknownLowBits)
{
   const auto input = opaque_input();
   agx_apple9_launch_recipe recipe;
   agx_apple9_launch_parameters params;
   ASSERT_TRUE(agx_apple9_launch_import(&recipe, &params,
                                        AGX_APPLE9_LAUNCH_VERTEX, input.data(),
                                        input.size(), 0x44));
   params.main_call = 0x7aa;
   params.publication_word = 0x1203;
   params.publication_count_valid = true;
   const unsigned counts[] = {0, 8, 9, 36, 1022};
   const unsigned words[] = {3, 0x203, 0x283, 0x903, 0xff83};
   std::array<uint8_t, 256> output;
   for (unsigned i = 0; i < 5; ++i) {
      params.publication_count = counts[i];
      ASSERT_TRUE(agx_apple9_launch_build(output.data(), output.size(), &recipe,
                                          &params));
      EXPECT_EQ(output[0x4f] | (output[0x50] << 8), words[i]);
   }
   const auto original = output;
   params.publication_count = 1023;
   EXPECT_FALSE(
      agx_apple9_launch_build(output.data(), output.size(), &recipe, &params));
   EXPECT_EQ(output, original);
}

TEST(Apple9Launcher, ThreadgroupAllocationIsIndependentOfFrameAndMain)
{
   auto input = opaque_input();
   memcpy(input.data() + 0x44 - 10, "\x77\x00\x2a\x41", 4);
   agx_apple9_launch_recipe recipe;
   agx_apple9_launch_parameters params;
   ASSERT_TRUE(agx_apple9_launch_import(&recipe, &params,
                                        AGX_APPLE9_LAUNCH_COMPUTE, input.data(),
                                        input.size(), 0x44));
   params.main_call = 0x107aa;
   params.state = 0x20;
   params.frame_extent_a = 4;
   params.frame_extent_b = 4096;
   std::array<uint8_t, 256> output;
   for (unsigned bytes : {128, 256, 512, 1024}) {
      params.threadgroup_memory_bytes = bytes;
      ASSERT_TRUE(agx_apple9_launch_build(output.data(), output.size(), &recipe,
                                          &params));
      EXPECT_EQ(output[0x3e] | (output[0x3f] << 8), (bytes << 2) | 0x80);
      EXPECT_EQ(output[0x40], 0);
      EXPECT_EQ(output[0x41], 0);
      EXPECT_EQ(output[0x42], input[0x42]);
      EXPECT_EQ(output[0x43], input[0x43]);
      EXPECT_EQ(output[0x44], 0xaa);
      EXPECT_EQ(output[0x45], 7);
      EXPECT_EQ(output[0x46], 1);
      EXPECT_EQ(output[0x51], 4);
      EXPECT_EQ(output[0x53], 0);
      EXPECT_EQ(output[0x54], 16);
   }
   const auto original = output;
   for (unsigned bytes : {1u, 64u, 129u, 384u, 2048u, UINT32_MAX}) {
      params.threadgroup_memory_bytes = bytes;
      EXPECT_FALSE(agx_apple9_launch_build(output.data(), output.size(),
                                           &recipe, &params));
      EXPECT_EQ(output, original);
   }
}

static void
write32(std::vector<uint8_t> &data, unsigned at, unsigned value)
{
   for (unsigned i = 0; i < 4; ++i)
      data[at + i] = value >> (8 * i);
}

TEST(Apple9Launcher, ExternalLibraryChecksEverySpanBeforeReturningViews)
{
   for (unsigned version : {2u, 3u, 4u, 5u}) {
      const unsigned sizes[] = {version >= 4 ? 18u : 76u, version >= 4 ? 30u : 106u,
                                version == 2 ? 148u : version == 5 ? 18u : 32u, 58, version >= 4 ? 0u : 48u,
                                11, 17, 33, version == 2 ? 337u : 12u};
      std::vector<uint8_t> data(128);
      memcpy(data.data(), "A9LFRG02", 8);
      data[7] = '0' + version;
      write32(data, 8, 9);
      write32(data, 12, 3);
      for (unsigned i = 0; i < 9; ++i) {
         unsigned at = data.size();
         write32(data, 16 + 8 * i, at);
         write32(data, 20 + 8 * i, sizes[i]);
         data.resize(at + sizes[i]);
         if (i == 3) {
            data[at] = 0xf7;
            data[at + 2] = 0x2a;
         }
      }
      for (unsigned i = 0; i < 3; ++i) {
         data[88 + i * 12 + 6] = 0x43;
         data[88 + i * 12 + 11] = i == 2 ? 4 : 1;
      }
      agx_apple9_launch_library library = {};
      ASSERT_TRUE(
         agx_apple9_launch_library_open(&library, data.data(), data.size()));
      for (size_t size = 0; size < data.size(); ++size) {
         EXPECT_FALSE(agx_apple9_launch_library_open(&library, data.data(), size))
            << size;
      }
      auto corrupt = data;
      write32(corrupt, 16 + 8 * 8, UINT32_MAX);
      EXPECT_FALSE(
         agx_apple9_launch_library_open(&library, corrupt.data(), corrupt.size()));
      corrupt = data;
      write32(corrupt, 20 + 8 * 8, UINT32_MAX);
      EXPECT_FALSE(
         agx_apple9_launch_library_open(&library, corrupt.data(), corrupt.size()));
      agx_apple9_launch_recipe recipe;
      agx_apple9_launch_parameters params;
      EXPECT_FALSE(agx_apple9_launch_select(&library, AGX_APPLE9_LAUNCH_VERTEX,
                                            true, &recipe, &params));
      ASSERT_TRUE(agx_apple9_launch_select(&library, AGX_APPLE9_LAUNCH_FRAGMENT,
                                           true, &recipe, &params));
      EXPECT_EQ(recipe.prefix.size, version >= 4 ? 18u : 106u);
      EXPECT_EQ(recipe.coverage_setup.size, version >= 4 ? 30u : 0u);
      EXPECT_EQ(recipe.suffix.size, 33u);
      EXPECT_EQ(recipe.size, 256u);
      ASSERT_TRUE(agx_apple9_launch_select(&library, AGX_APPLE9_LAUNCH_COMPUTE,
                                           false, &recipe, &params));
      EXPECT_EQ(recipe.generated_resources, version >= 3);
      EXPECT_EQ(recipe.prefix.size, version == 5 ? 18u : version >= 3 ? 32u : 148u);
      EXPECT_EQ(recipe.generated_state_load, version == 5);
      EXPECT_EQ(recipe.suffix.size, version >= 3 ? 12u : 337u);
   }
}

TEST(Apple9Launcher, GeneratedResourcesMoveCallAndSelectStateLoading)
{
   for (bool generated_state : {false, true}) {
      std::array<uint8_t, 32> state;
      state.fill(0x35);
      std::array<uint8_t, 58> shared = {0xf7, 0, 0x2a};
      std::array<uint8_t, 12> ending;
      ending.fill(0xaa);
      agx_apple9_launch_recipe recipe = {};
      recipe.stage = AGX_APPLE9_LAUNCH_COMPUTE;
      recipe.generated_resources = true;
      recipe.generated_state_load = generated_state;
      recipe.prefix = {state.data(), generated_state ? 18u : state.size()};
      recipe.shared = {shared.data(), shared.size()};
      recipe.suffix = {ending.data(), ending.size()};
      recipe.size = 1024;
      agx_apple9_launch_parameters params = {};
      params.shader_base = 0x1000000000ull;
      params.resource_table = params.shader_base + 0x20d000;
      params.state = params.shader_base + 0x308020;
      params.main_call = 0x12abc;
      params.frame_extent_a = 4096;
      std::array<uint8_t, 1024> out;
      for (unsigned count = 1; count <= 18; ++count) {
         params.resource_count = count;
         ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), &recipe, &params));
         unsigned roots = count + 3;
         unsigned call = agx_apple9_launch_call_offset(&recipe, count);
         EXPECT_EQ(call, 56 + 14 * ((roots + 1) / 2));
         EXPECT_EQ(out[call], 0xbc);
         EXPECT_EQ(out[call + 1], 0x2a);
         EXPECT_EQ(out[call + 2], 1);
         EXPECT_EQ(out[call + 14], 16);
         unsigned state_start = call - 24;
         if (generated_state) {
            EXPECT_EQ(out[state_start + 3], 2 * (18 + roots * 2));
            EXPECT_EQ(out[state_start + 4], 40);
            EXPECT_EQ(out[state_start + 8], 0x57); /* Four words. */
            EXPECT_EQ(out[state_start + 10], 0);  /* No address offset. */
         } else {
            for (unsigned i = 0; i < 14; ++i)
               EXPECT_EQ(out[state_start + i], i == 3 ? 2 * (18 + roots * 2) : state[i]);
         }
         unsigned return_start = call + 66 + (2*roots + 4)*4;
         EXPECT_EQ(memcmp(out.data() + return_start, ending.data(), ending.size()), 0);
         for (unsigned i = return_start + ending.size(); i < out.size(); ++i)
            EXPECT_EQ(out[i], 0);
      }
      const auto saved = out;
      for (unsigned count : {0u, 19u, UINT32_MAX}) {
         params.resource_count = count;
         EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size(), &recipe, &params));
         EXPECT_EQ(out, saved);
      }
      params.resource_count = 18;
      EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size() - 1, &recipe, &params));
      EXPECT_EQ(out, saved);
      recipe.prefix.size--;
      EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size(), &recipe, &params));
      EXPECT_EQ(out, saved);
   }
}

TEST(Apple9Launcher, GeneratedGraphicsPreservesCoverageAndFullRootAddress)
{
   std::array<uint8_t, 18> pre;
   pre.fill(0xcc);
   std::array<uint8_t, 30> coverage;
   for (unsigned i = 0; i < coverage.size(); ++i)
      coverage[i] = 0x80 + i;
   std::array<uint8_t, 58> shared = {0xf7, 0, 0x2a};
   std::array<uint8_t, 17> ending;
   ending.fill(0xda);
   agx_apple9_launch_recipe recipe = {};
   recipe.prefix = {pre.data(), pre.size()};
   recipe.shared = {shared.data(), shared.size()};
   recipe.suffix = {ending.data(), ending.size()};
   recipe.generated_resources = true;
   recipe.size = 256;
   recipe.pre_call[0] = 0x43;
   agx_apple9_launch_parameters params = {};
   params.shader_base = 0x1000010000ull;
   params.resource_table = params.shader_base + 0xe100;
   params.main_call = 0x3abcd;
   params.tile_bytes = 16;
   std::array<uint8_t, 256> out;
   for (bool uses_coverage : {false, true}) {
      recipe.coverage_setup = uses_coverage
         ? agx_apple9_launch_fragment{coverage.data(), coverage.size()}
         : agx_apple9_launch_fragment{};
      for (auto stage : {AGX_APPLE9_LAUNCH_VERTEX, AGX_APPLE9_LAUNCH_FRAGMENT}) {
         recipe.stage = stage;
         if (uses_coverage && stage == AGX_APPLE9_LAUNCH_VERTEX)
            continue;
         for (unsigned samples : {1u, 2u, 4u}) {
            params.samples = samples;
            ASSERT_TRUE(agx_apple9_launch_build(out.data(), out.size(), &recipe, &params));
            unsigned call = agx_apple9_launch_call_offset(&recipe, 0);
            EXPECT_EQ(call, uses_coverage ? 98u : 68u);
            EXPECT_EQ(out[call], 0xcd);
            EXPECT_EQ(out[call + 1], 0xab);
            EXPECT_EQ(out[call + 2], 3);
            uint32_t low = (out[1] & 0x7f) | ((out[4] & 0x1e) << 6) |
                           ((out[5] & 0x0c) << 9) | (out[6] << 13) |
                           ((out[7] & 0x0f) << 21) | ((uint32_t)(out[3] & 0xfe) << 24);
            EXPECT_EQ(low, (uint32_t)params.resource_table);
            if (uses_coverage) {
               EXPECT_EQ(memcmp(out.data() + 16, coverage.data(), 16), 0);
               EXPECT_EQ(memcmp(out.data() + 74, coverage.data() + 16, 14), 0);
            }
            unsigned end = call + 8 + 58 + 48;
            EXPECT_EQ(memcmp(out.data() + end, ending.data(), ending.size()), 0);
            for (unsigned i = end + ending.size(); i < out.size(); ++i)
               EXPECT_EQ(out[i], 0);
         }
      }
   }
   const auto saved = out;
   recipe.coverage_setup.size = 29;
   EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size(), &recipe, &params));
   EXPECT_EQ(out, saved);
   recipe.coverage_setup.size = 30;
   recipe.coverage_setup.data = nullptr;
   EXPECT_FALSE(agx_apple9_launch_build(out.data(), out.size(), &recipe, &params));
   EXPECT_EQ(out, saved);
}
