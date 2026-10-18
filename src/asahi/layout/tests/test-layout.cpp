/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "layout.h"

TEST(Cubemap, Nonmipmapped)
{
   struct ail_layout layout = {
      .width_px = 512,
      .height_px = 512,
      .depth_px = 6,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.layer_stride_B, ALIGN_POT(512 * 512 * 4, 0x4000));
   EXPECT_EQ(layout.size_B, ALIGN_POT(512 * 512 * 4 * 6, 0x4000));
}

TEST(Cubemap, RoundsToOnePage)
{
   struct ail_layout layout = {
      .width_px = 63,
      .height_px = 63,
      .depth_px = 6,
      .sample_count_sa = 1,
      .levels = 6,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R32_FLOAT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.level_offsets_B[0], 0);
   EXPECT_EQ(layout.level_offsets_B[1], 0x4000);
   EXPECT_EQ(layout.level_offsets_B[2], 0x5000);
   EXPECT_EQ(layout.level_offsets_B[3], 0x5400);
   EXPECT_EQ(layout.level_offsets_B[4], 0x5500);
   EXPECT_TRUE(layout.page_aligned_layers);
   EXPECT_EQ(layout.layer_stride_B, 0x8000);
   EXPECT_EQ(layout.size_B, 0x30000);
}

TEST(Linear, SmokeTestBuffer)
{
   struct ail_layout layout = {
      .width_px = 81946,
      .height_px = 1,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_LINEAR,
      .format = PIPE_FORMAT_R8_UINT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.size_B, ALIGN_POT(81946, AIL_CACHELINE));
}

TEST(Miptree, AllMipLevels)
{
   struct ail_layout layout = {
      .width_px = 1024,
      .height_px = 1024,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 11,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UINT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.size_B, 0x555680);
   EXPECT_EQ(layout.sparse_folios_per_layer, 2);
   EXPECT_EQ(layout.sparse_table_size_B, 0x1000);
}

TEST(Miptree, SomeMipLevels)
{
   struct ail_layout layout = {
      .width_px = 1024,
      .height_px = 1024,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 4,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UINT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.size_B, 0x555680);
   EXPECT_EQ(layout.sparse_folios_per_layer, 2);
   EXPECT_EQ(layout.sparse_table_size_B, 0x1000);
}

TEST(Miptree, SmallPartialMiptree2DArray)
{
   struct ail_layout layout = {
      .width_px = 32,
      .height_px = 16,
      .depth_px = 64,
      .sample_count_sa = 1,
      .levels = 4,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R32_FLOAT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.layer_stride_B, 0xc00);
   EXPECT_EQ(layout.size_B, 0x30000);
   EXPECT_EQ(layout.sparse_folios_per_layer, 1);
   EXPECT_EQ(layout.sparse_table_size_B, 0x20000);
}

TEST(Miptree, SmallPartialMiptree3D)
{
   struct ail_layout layout = {
      .width_px = 32,
      .height_px = 16,
      .depth_px = 64,
      .sample_count_sa = 1,
      .levels = 4,
      .mipmapped_z = true,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R32_FLOAT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.layer_stride_B, 0xc80);
   EXPECT_EQ(layout.size_B, 0x32000);
   EXPECT_EQ(layout.sparse_folios_per_layer, 1);
   EXPECT_EQ(layout.sparse_table_size_B, 0x20000);
}

/* Native EXP-M4-07 byte offsets, reproduced by the shared AGX layout. */
TEST(Apple9Miptree, CapturedSquareOffsets)
{
   ail_layout layout = {
      .width_px = 384, .height_px = 384, .depth_px = 1,
      .sample_count_sa = 1, .levels = 9,
      .tiling = AIL_TILING_GPU, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   ail_make_miptree(&layout);
   const unsigned offsets[] = {0, 0x90000, 0xb4000, 0xc8000, 0xcc000,
                              0xcd000, 0xcd400, 0xcd500, 0xcd580, 0xcd600};
   for (unsigned l = 0; l < ARRAY_SIZE(offsets); ++l)
      EXPECT_EQ(layout.level_offsets_B[l], offsets[l]) << l;
   EXPECT_EQ(layout.mip_tail_first_lod, 3);
   EXPECT_EQ(layout.size_B, 0xcd600);
}

TEST(Apple9Miptree, TinyLevelsKeepMinimumSlots)
{
   ail_layout layout = {
      .width_px = 128, .height_px = 128, .depth_px = 1,
      .sample_count_sa = 1, .levels = 8,
      .tiling = AIL_TILING_GPU, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   ail_make_miptree(&layout);
   const unsigned offsets[] = {0, 0x10000, 0x14000, 0x15000, 0x15400,
                              0x15500, 0x15580, 0x15600, 0x15680};
   for (unsigned l = 0; l < ARRAY_SIZE(offsets); ++l)
      EXPECT_EQ(layout.level_offsets_B[l], offsets[l]) << l;
}

/* Hardware sampling of every level in the rectangular atlas regression
 * verifies the padding that square-only tests did not distinguish. */
TEST(Apple9Miptree, RectangularAtlasLargeLevelPadding)
{
   ail_layout layout = {
      .width_px = 994, .height_px = 950, .depth_px = 1,
      .sample_count_sa = 1, .levels = 10,
      .tiling = AIL_TILING_GPU, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   ail_make_miptree(&layout);
   const unsigned offsets[] = {0, 0x3c0000, 0x4d0000, 0x51c000, 0x530000,
                              0x534000, 0x535000, 0x535400, 0x535500,
                              0x535580, 0x535600};
   for (unsigned l = 0; l < ARRAY_SIZE(offsets); ++l)
      EXPECT_EQ(layout.level_offsets_B[l], offsets[l]) << l;
   EXPECT_EQ(layout.mip_tail_first_lod, 4);
   EXPECT_EQ(layout.size_B, 0x535600);
}

TEST(Apple9Miptree, WideAtlasDoesNotAddTailGap)
{
   ail_layout layout = {
      .width_px = 1846, .height_px = 760, .depth_px = 1,
      .sample_count_sa = 1, .levels = 11,
      .tiling = AIL_TILING_GPU, .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   ail_make_miptree(&layout);
   const unsigned offsets[] = {0, 0x570000, 0x6e4000, 0x744000, 0x76c000,
                              0x774000, 0x776000, 0x776800, 0x776a00,
                              0x776a80, 0x776b00, 0x776b80};
   for (unsigned l = 0; l < ARRAY_SIZE(offsets); ++l)
      EXPECT_EQ(layout.level_offsets_B[l], offsets[l]) << l;
   EXPECT_EQ(layout.mip_tail_first_lod, 4);
   EXPECT_EQ(layout.size_B, 0x776b80);
   // Sub-tile addressing uses square tiles, including elongated tiny levels.
   EXPECT_EQ(layout.tilesize_el[8].width_el, 2);
   EXPECT_EQ(layout.tilesize_el[8].height_el, 2);
}


/* Render views may select any mip and layer after allocation. All legal
 * color layouts must preserve the 16-byte descriptor address granularity. */
TEST(Layout, RenderViewAlignment)
{
   for (pipe_format format : {PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R16_FLOAT,
                             PIPE_FORMAT_R8G8B8A8_UNORM,
                             PIPE_FORMAT_R16G16B16A16_FLOAT,
                             PIPE_FORMAT_R32G32B32A32_FLOAT,
                             PIPE_FORMAT_R5G6B5_UNORM}) {
      for (unsigned width : {1u, 2u, 7u, 16u, 31u, 33u, 65u, 257u, 16384u}) {
         for (unsigned samples : {1u, 2u, 4u}) {
            for (bool volume : {false, true}) {
               ail_layout layout = {};
               layout.width_px = width;
               layout.height_px = MIN2(width, 33);
               layout.depth_px = 6;
               layout.sample_count_sa = samples;
               layout.levels = samples == 1 ? util_logbase2(width) + 1 : 1;
               layout.tiling = AIL_TILING_GPU;
               layout.format = format;
               layout.mipmapped_z = volume;
               layout.renderable = true;
               ail_make_miptree(&layout);
               EXPECT_EQ(layout.layer_stride_B & 15, 0u);
               for (unsigned level = 0; level < layout.levels; ++level) {
                  for (unsigned layer = 0; layer < 6; ++layer)
                     EXPECT_EQ((layout.level_offsets_B[level] +
                                ail_get_layer_offset_B(&layout, layer)) & 15, 0u);
               }
            }
         }
         for (unsigned offset : {0u, 16u, 65520u}) {
            for (bool imported_stride : {false, true}) {
               ail_layout layout = {};
               layout.width_px = width;
               layout.height_px = MIN2(width, 33);
               layout.depth_px = 6;
               layout.sample_count_sa = 1;
               layout.levels = 1;
               layout.tiling = AIL_TILING_LINEAR;
               layout.format = format;
               layout.renderable = true;
               layout.level_offsets_B[0] = offset;
               layout.linear_stride_B = imported_stride ? (1u << 20) : 0;
               ail_make_miptree(&layout);
               EXPECT_EQ(layout.linear_stride_B & 15, 0u);
               EXPECT_LE(layout.linear_stride_B, 1u << 20);
               EXPECT_EQ(layout.layer_stride_B & 15, 0u);
               for (unsigned layer = 0; layer < 6; ++layer)
                  EXPECT_EQ((layout.level_offsets_B[0] +
                             ail_get_layer_offset_B(&layout, layer)) & 15, 0u);
            }
         }
      }
   }
}
