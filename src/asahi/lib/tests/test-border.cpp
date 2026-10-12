/* SPDX-License-Identifier: MIT */

#include <gtest/gtest.h>

#include "agx_border.h"
#include "agx_pack.h"
#include "util/format/u_format.h"

TEST(Border, FormatChannelOrder)
{
   /* Distinct, exactly representable UNORM4 values distinguish a four-cycle
    * from its inverse. Vulkan B4G4R4A4_PACK16 is pipe A4R4G4B4 (low bits first).
    */
   const uint32_t rgba[] = {
      fui(1.0f / 15.0f), fui(4.0f / 15.0f), fui(9.0f / 15.0f), fui(14.0f / 15.0f),
   };
   const struct {
      enum pipe_format format;
      uint32_t memory_channels[4];
   } cases[] = {
      {PIPE_FORMAT_R4G4B4A4_UNORM, {1, 4, 9, 14}},
      {PIPE_FORMAT_B4G4R4A4_UNORM, {9, 4, 1, 14}},
      {PIPE_FORMAT_A4B4G4R4_UNORM, {14, 9, 4, 1}},
      {PIPE_FORMAT_A4R4G4B4_UNORM, {14, 1, 4, 9}},
   };

   for (const auto &c : cases) {
      SCOPED_TRACE(c.format);
      struct agx_border_packed packed;
      agx_pack_border(&packed, rgba, c.format);
      agx_unpack(stderr, &packed, BORDER, border);
      EXPECT_EQ(border.channel_0, c.memory_channels[0]);
      EXPECT_EQ(border.channel_1, c.memory_channels[1]);
      EXPECT_EQ(border.channel_2, c.memory_channels[2]);
      EXPECT_EQ(border.channel_3, c.memory_channels[3]);
   }
}

TEST(Border, StencilReadsRed)
{
   /* Vulkan passes a stencil border value in R; the S8 swizzle is _X__. */
   const uint32_t rgba[] = {173, 34, 0, 1};
   const enum pipe_format formats[] = {PIPE_FORMAT_S8_UINT,
                                       PIPE_FORMAT_X24S8_UINT,
                                       PIPE_FORMAT_X32_S8X24_UINT};
   for (enum pipe_format format : formats) {
      SCOPED_TRACE(format);
      struct agx_border_packed packed;
      agx_pack_border(&packed, rgba, format);
      agx_unpack(stderr, &packed, BORDER, border);
      const struct util_format_description *desc =
         util_format_description(format);
      for (unsigned i = 0; i < 4; ++i) {
         if (desc->swizzle[i] <= PIPE_SWIZZLE_W) {
            uint32_t ch[] = {border.channel_0, border.channel_1,
                             border.channel_2, border.channel_3};
            EXPECT_EQ(ch[desc->swizzle[i]], 173u);
         }
      }
   }
}

TEST(Border, ReplicatedChannelUsesFirstComponent)
{
   const uint32_t rgba[] = {fui(0.2f), fui(0.4f), fui(0.6f), fui(0.8f)};
   struct agx_border_packed packed;
   agx_pack_border(&packed, rgba, PIPE_FORMAT_L8A8_UNORM);
   agx_unpack(stderr, &packed, BORDER, border);
   EXPECT_EQ(border.channel_0, 51u);
   EXPECT_EQ(border.channel_1, 204u);
}
