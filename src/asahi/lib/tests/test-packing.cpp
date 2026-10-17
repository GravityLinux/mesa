/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_pack.h"
#include "agx_apple9.h"

#include <gtest/gtest.h>

const struct {
   float f;
   uint16_t encoded;
   bool inexact;
} lod_cases[] = {
   /* Lower bound clamp */
   {-INFINITY, 0x00, true},
   {-0.1, 0x00, true},
   {-0.0, 0x00, true},

   /* Exact bounds */
   {0.0, 0x00},
   {14.0, 0x380},

   /* Upper bound clamp */
   {14.1, 0x380, true},
   {18.1, 0x380, true},
   {INFINITY, 0x380, true},
};

const struct {
   uint32_t group_size;
   uint32_t length;
   uint32_t value;
   uint32_t encoded;
} group_cases[] = {
   /* Groups of 16 in a 4-bit word */
   {16, 4, 0, 0x1},   {16, 4, 1, 0x1},   {16, 4, 16, 0x1},  {16, 4, 17, 0x2},
   {16, 4, 31, 0x2},  {16, 4, 32, 0x2},  {16, 4, 33, 0x3},  {16, 4, 239, 0xF},
   {16, 4, 240, 0xF}, {16, 4, 241, 0x0}, {16, 4, 255, 0x0}, {16, 4, 256, 0x0},
};

TEST(LODClamp, Encode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(lod_cases); ++i)
      ASSERT_EQ(__gen_pack_lod(lod_cases[i].f, 0, 9), lod_cases[i].encoded);
}

TEST(LODClamp, Decode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(lod_cases); ++i) {
      if (lod_cases[i].inexact)
         continue;

      uint32_t cl;
      memcpy(&cl, &lod_cases[i].encoded, sizeof(lod_cases[i].encoded));

      ASSERT_EQ(__gen_unpack_lod(&cl, 0, 10), lod_cases[i].f);
   }
}

TEST(Groups, Encode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_cases); ++i) {
      ASSERT_EQ(__gen_to_groups(group_cases[i].value, group_cases[i].group_size,
                                group_cases[i].length),
                group_cases[i].encoded);
   }
}

TEST(Groups, Decode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_cases); ++i) {
      unsigned expected =
         ALIGN_POT(group_cases[i].value, group_cases[i].group_size);

      /* Clamp to minimum encodable */
      if (group_cases[i].value == 0)
         expected = group_cases[i].group_size;

      ASSERT_EQ(
         __gen_from_groups(group_cases[i].encoded, group_cases[i].group_size,
                           group_cases[i].length),
         expected);
   }
}

/* Apple9 keeps VA[47:4] below the sRGB and stride fields. Exercise each of
 * the eight address bits which cannot pass through a legacy descriptor.
 */
TEST(Apple9Texture, FullAddressDoesNotOverlapFlagsOrStride)
{
   struct AGX_TEXTURE tex = {AGX_TEXTURE_header};
   tex.dimension = AGX_TEXTURE_DIMENSION_2D;
   tex.layout = AGX_LAYOUT_LINEAR;
   tex.width = 1024;
   tex.height = 3;
   tex.stride = 4096 - 16;
   tex.srgb = true;
   tex.srgb_2_channel = true;

   for (unsigned bit = 40; bit < 48; ++bit) {
      tex.address = (UINT64_C(1) << bit) | 0x12345670;
      uint32_t words[AGX_APPLE9_TEXTURE_TABLE_STRIDE / 4] = {};
      ASSERT_TRUE(agx_apple9_pack_texture(words, &tex));
      uint64_t address_word = words[2] | (uint64_t(words[3]) << 32);
      EXPECT_EQ((address_word & ((UINT64_C(1) << 44) - 1)) << 4,
                tex.address);
      EXPECT_EQ((address_word >> 44) & 3, 3u);
      EXPECT_EQ(address_word >> 46, tex.stride >> 4);
   }
   uint32_t words[AGX_APPLE9_TEXTURE_TABLE_STRIDE / 4];
   tex.address = UINT64_C(1) << 48;
   EXPECT_FALSE(agx_apple9_pack_texture(words, &tex));
   tex.address = (UINT64_C(1) << 40) | 1;
   EXPECT_FALSE(agx_apple9_pack_texture(words, &tex));
}

TEST(Apple9Texture, ThreeDimensionalDepthHasElevenBits)
{
   struct AGX_TEXTURE tex = {AGX_TEXTURE_header};
   tex.dimension = AGX_TEXTURE_DIMENSION_3D;
   tex.layout = AGX_LAYOUT_TWIDDLED;
   tex.width = tex.height = 2;
   tex.address = UINT64_C(0x2fff000000);
   uint64_t words[AGX_APPLE9_TEXTURE_TABLE_STRIDE / 8];
   for (unsigned depth : {1u, 192u, 2047u, 2048u}) {
      tex.depth = depth;
      ASSERT_TRUE(agx_apple9_pack_texture(words, &tex));
      EXPECT_EQ((words[1] >> 46) & 0x7ff, depth - 1);
   }
   for (unsigned depth : {2049u, 4095u, 4096u, 8192u, 16384u}) {
      tex.depth = depth;
      EXPECT_FALSE(agx_apple9_pack_texture(words, &tex));
   }
   /* Array counts use a different hardware field; retain their range. */
   tex.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   tex.depth = 16384;
   EXPECT_TRUE(agx_apple9_pack_texture(words, &tex));
}

TEST(Apple9Texture, LinearR32fWithHighAddress)
{
   struct AGX_TEXTURE tex = {AGX_TEXTURE_header};
   tex.dimension = AGX_TEXTURE_DIMENSION_2D;
   tex.layout = AGX_LAYOUT_LINEAR;
   tex.channels = AGX_CHANNELS_R32;
   tex.type = AGX_TEXTURE_TYPE_FLOAT;
   tex.swizzle_r = AGX_CHANNEL_R;
   tex.swizzle_g = tex.swizzle_b = AGX_CHANNEL_0;
   tex.swizzle_a = AGX_CHANNEL_1;
   tex.width = 17;
   tex.height = 9;
   tex.address = UINT64_C(0x10000018000);
   uint32_t words[8];
   ASSERT_TRUE(agx_apple9_pack_texture(words, &tex));
   const uint32_t reference[8] = {
      0x09688842, 0x00002001, 0x00001800, 0x00000010, 0, 0, 0, 0,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(words); ++i)
      EXPECT_EQ(words[i], reference[i]) << "word " << i;
}
