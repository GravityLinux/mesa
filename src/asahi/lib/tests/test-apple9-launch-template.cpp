/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include <random>
#include <string.h>

extern "C" {
#include "asahi/lib/agx_apple9_launch.h"
#include "asahi/lib/agx_apple9_layout.h"
}

/* A template emitted with per-launch fields must equal a full build. */
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
         ASSERT_EQ(memcmp(a, b, capacity), 0)
            << "stage " << stage << " iter " << iter << " launch " << launch;
         ++checked;
      }
   }
   EXPECT_GT(checked, 20000u);
}
