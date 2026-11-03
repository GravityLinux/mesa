/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include <random>
#include <string.h>

extern "C" {
#include "asahi/compiler/agx_apple9_encoding.h"
#include "asahi/compiler/agx_apple9_profile.h"
#include "asahi/lib/agx_apple9_launch.h"
#include "asahi/lib/agx_apple9_layout.h"
}

/* A template emitted with per-launch fields (and, for compute, its root
 * words) must equal a full build. */
TEST(Apple9LaunchTemplate, MatchesBuild)
{
   std::mt19937_64 rng(0x1a9e);
   const uint64_t base = 1ull << 40;
   unsigned checked = 0;
   for (unsigned iter = 0; iter < 20000; ++iter) {
      enum agx_apple9_launch_stage stage =
         (enum agx_apple9_launch_stage)(rng() % 3);
      bool compute = stage == AGX_APPLE9_LAUNCH_COMPUTE;
      size_t capacity = compute ? AGX_APPLE9_COMPUTE_LAUNCH_SIZE
                                : AGX_APPLE9_GRAPHICS_LAUNCH_SIZE;
      struct agx_apple9_launch_parameters p = {};
      p.shader_base = base;
      p.publication_count = rng() % 64;
      p.frame_extent_a = (rng() % 4) ? 0 : (rng() % 4096);
      p.frame_extent_b = (rng() % 4) ? 0 : (rng() % 4096);
      p.resource_count = rng() % 12;
      if (compute && (rng() % 3) == 0)
         p.threadgroup_memory_bytes = 1024 * (1 + rng() % 32);
      if (stage == AGX_APPLE9_LAUNCH_FRAGMENT) {
         p.tile_bytes = rng() % 129;
         static const uint8_t samples[] = {1, 2, 4};
         p.samples = samples[rng() % 3];
      }
      if (rng() % 2)
         p.preamble_address = base + ((rng() % (1u << 30)) & ~1ull);

      struct agx_apple9_launch_template tmpl;
      bool ok = agx_apple9_launch_template_init(&tmpl, capacity, stage, &p);

      for (unsigned launch = 0; launch < 4; ++launch) {
         struct agx_apple9_launch_parameters q = p;
         q.resource_table = rng();
         q.entry_offset = (rng() % (AGX_APPLE9_ENTRY_MAX_OFFSET + 1)) & ~1u;
         q.launch_address = base + ((rng() % (1u << 31)) & ~63ull);
         uint32_t roots[2 * (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE +
                             AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES)];
         for (auto &w : roots)
            w = rng();
         q.root_words = roots;
         uint8_t a[AGX_APPLE9_COMPUTE_LAUNCH_SIZE], b[AGX_APPLE9_COMPUTE_LAUNCH_SIZE];
         memset(a, 0xa5, sizeof(a));
         memset(b, 0xa5, sizeof(b));
         bool built = agx_apple9_launch_build(a, capacity, stage, &q);
         ASSERT_EQ(built, ok) << "stage " << stage << " iter " << iter;
         if (!built)
            break;
         ASSERT_TRUE(agx_apple9_launch_template_emit(b, &tmpl, q.resource_table,
                                                     q.entry_offset,
                                                     q.launch_address));
         if (compute)
            ASSERT_TRUE(agx_apple9_launch_template_set_roots(b, &tmpl, roots));
         ASSERT_EQ(memcmp(a, b, capacity), 0)
            << "stage " << stage << " iter " << iter << " launch " << launch;
         ++checked;
      }
   }
   EXPECT_GT(checked, 20000u);
}

/* Compute launches materialize their root words: no table loads, and the
 * first argument transfer does not wait for one. A build without root words
 * is refused. */
TEST(Apple9LaunchTemplate, ComputeRootsAreLiterals)
{
   struct agx_apple9_launch_parameters p = {};
   p.shader_base = 1ull << 40;
   p.entry_offset = 0x1000;
   for (unsigned resources = 1; resources <= AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES;
        ++resources) {
      p.resource_count = resources;
      uint32_t roots[2 * (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE +
                          AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES)];
      for (unsigned i = 0; i < sizeof(roots) / sizeof(roots[0]); ++i)
         roots[i] = 0x01020304u * (i + 1);
      uint8_t out[AGX_APPLE9_COMPUTE_LAUNCH_SIZE];
      p.root_words = NULL;
      EXPECT_FALSE(agx_apple9_launch_build(out, sizeof(out), AGX_APPLE9_LAUNCH_COMPUTE, &p));
      p.root_words = roots;
      ASSERT_TRUE(agx_apple9_launch_build(out, sizeof(out), AGX_APPLE9_LAUNCH_COMPUTE, &p));
      const unsigned words = agx_apple9_compute_root_words(resources);
      for (unsigned i = 0; i < words; ++i) {
         uint8_t literal[8];
         ASSERT_TRUE(agx_apple9_encode_literal32(literal, 18 + i, roots[i]));
         EXPECT_EQ(memcmp(out + 16 + 8 * i, literal, 8), 0) << "word " << i;
      }
      const unsigned call = agx_apple9_launch_call_offset(AGX_APPLE9_LAUNCH_COMPUTE, resources);
      EXPECT_EQ(call, 16 + 8 * words + 8 + 2);
      /* Argument transfers follow the target (10) and frame (10) blocks. */
      EXPECT_EQ(out[call - 2 + 20 + 3] & 0x40, 0) << "first argument transfer waits";
   }
}
