/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "u_threaded_context.h"

TEST(ThreadedRenderpass, CompleteReadOnlySplit)
{
   tc_renderpass_info info = {};
   info.ended = true;
   info.cbuf_invalidate = 0xff;
   info.zsbuf_invalidate = true;
   info.zsbuf_read_dsa = true;
   tc_renderpass_info_reset(&info);
   EXPECT_TRUE(info.ended);
   EXPECT_TRUE(info.zsbuf_read_dsa);
   EXPECT_FALSE(info.zsbuf_clear_partial);
   EXPECT_FALSE(info.zsbuf_write_dsa);
   EXPECT_EQ(info.cbuf_load, 0xff);
   EXPECT_EQ(info.cbuf_invalidate, 0);
   EXPECT_FALSE(info.zsbuf_invalidate);
   tc_renderpass_info_reset(&info);
   EXPECT_FALSE(info.zsbuf_clear_partial);
}

TEST(ThreadedRenderpass, CompleteUnusedDepthSplit)
{
   tc_renderpass_info info = {};
   info.ended = true;
   tc_renderpass_info_reset(&info);
   EXPECT_FALSE(tc_renderpass_info_is_zsbuf_used(&info));
}

TEST(ThreadedRenderpass, IncompleteSplitStaysConservative)
{
   tc_renderpass_info info = {};
   info.zsbuf_read_dsa = true;
   tc_renderpass_info_reset(&info);
   EXPECT_FALSE(info.ended);
   EXPECT_TRUE(info.zsbuf_clear_partial);
}

TEST(ThreadedRenderpass, PreserveWritesAndLaterClears)
{
   for (unsigned kind = 0; kind < 3; ++kind) {
      tc_renderpass_info info = {};
      info.ended = true;
      info.zsbuf_write_dsa = kind == 0;
      info.zsbuf_write_fs = kind == 1;
      info.zsbuf_clear_partial = kind == 2;
      tc_renderpass_info_reset(&info);
      EXPECT_TRUE(tc_renderpass_info_is_zsbuf_used(&info));
      EXPECT_EQ(info.zsbuf_write_dsa, kind == 0);
      EXPECT_EQ(info.zsbuf_write_fs, kind == 1);
      EXPECT_EQ(info.zsbuf_clear_partial, kind == 2);
   }
}

TEST(ThreadedRenderpass, ConsumedFullClearIsNotRepeated)
{
   tc_renderpass_info info = {};
   info.ended = true;
   info.zsbuf_clear = true;
   info.cbuf_clear = 0xff;
   tc_renderpass_info_reset(&info);
   EXPECT_FALSE(info.zsbuf_clear);
   EXPECT_EQ(info.cbuf_clear, 0);
   EXPECT_FALSE(info.zsbuf_clear_partial);
}
