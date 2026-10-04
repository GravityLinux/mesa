/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_upload.h"
#include <gtest/gtest.h>
#include <array>

TEST(HkUpload, SmallUploadsFinishOldTail)
{
   std::array<uint8_t, 128 * 1024> a, b;
   hk_upload_chunk chunks[HK_UPLOAD_CHUNKS] = {};
   hk_upload_keep_chunk(chunks, {a.data(), 0, 96 * 1024});
   EXPECT_EQ(hk_upload_find_chunk(chunks, a.size(), 64 * 1024, 256), nullptr);
   hk_upload_keep_chunk(chunks, {b.data(), a.size(), 64 * 1024});
   auto *small = hk_upload_find_chunk(chunks, a.size(), 32 * 1024, 64);
   ASSERT_NE(small, nullptr);
   EXPECT_EQ(small->map, a.data());
   small->offset += 32 * 1024;
   auto *large = hk_upload_find_chunk(chunks, a.size(), 64 * 1024, 256);
   ASSERT_NE(large, nullptr);
   EXPECT_EQ(large->map, b.data());
   large->offset += 64 * 1024;
   EXPECT_EQ(hk_upload_find_chunk(chunks, a.size(), 1, 1), nullptr);
}

TEST(HkUpload, FitIncludesAlignment)
{
   uint8_t a, b;
   hk_upload_chunk chunks[HK_UPLOAD_CHUNKS] = {
      {&a, 0, 99}, {&b, 128, 64}};
   auto *chunk = hk_upload_find_chunk(chunks, 128, 16, 32);
   ASSERT_NE(chunk, nullptr);
   EXPECT_EQ(chunk->map, &b);
   EXPECT_EQ(hk_upload_find_chunk(chunks, 128, 65, 1), nullptr);
}

TEST(HkUpload, RetainLargestTails)
{
   uint8_t a, b, c;
   hk_upload_chunk chunks[HK_UPLOAD_CHUNKS] = {
      {&a, 0, 64}, {&b, 128, 96}};
   hk_upload_keep_chunk(chunks, {&c, 256, 112});
   EXPECT_EQ(chunks[0].map, &a);
   EXPECT_EQ(chunks[1].map, &b);
   hk_upload_keep_chunk(chunks, {&c, 256, 32});
   EXPECT_EQ(chunks[0].map, &a);
   EXPECT_EQ(chunks[1].map, &c);
}
