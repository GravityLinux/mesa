/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "gallium/drivers/asahi/agx_apple9.h"

#include <gtest/gtest.h>
#include <cstring>

static uint32_t
get_u32(const uint8_t *bytes, size_t offset)
{
   uint32_t value;
   memcpy(&value, bytes + offset, sizeof(value));
   return value;
}

static uint64_t
get_u64(const uint8_t *bytes, size_t offset)
{
   uint64_t value;
   memcpy(&value, bytes + offset, sizeof(value));
   return value;
}

TEST(Apple9ComputeGeometry, DirectPublishesThreadsAndIdentityScale)
{
   uint8_t record[0x100] = {};
   const uint64_t address = 0x1000200000ull;
   const agx_apple9_compute_geometry geometry = {
      .mode = AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT,
      .threads = {12, 10, 6},
      .local = {4, 5, 3},
   };
   ASSERT_TRUE(agx_apple9_build_compute_geometry_fields(
      record, sizeof(record), address, &geometry));
   EXPECT_EQ(get_u64(record, 0x00), address + 0xc0);
   EXPECT_EQ(get_u64(record, 0x08), address + 0xcc);
   for (unsigned d = 0; d < 3; ++d) {
      EXPECT_EQ(get_u32(record, 0xc0 + d * 4), geometry.threads[d]);
      EXPECT_EQ(get_u32(record, 0xcc + d * 4), 1u);
   }
}

TEST(Apple9ComputeGeometry, IndirectPublishesCallerRecordAndLocalScale)
{
   uint8_t record[0x100];
   memset(record, 0xa5, sizeof(record));
   const uint64_t address = 0x1000200000ull;
   const agx_apple9_compute_geometry geometry = {
      .mode = AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT,
      .group_counts = 0x2fff123400ull,
      .local = {4, 4, 1},
   };
   ASSERT_TRUE(agx_apple9_build_compute_geometry_fields(
      record, sizeof(record), address, &geometry));
   /* Every visible pointer remains intact even at the expanded capacity. */
   for (unsigned i = 0x10; i < 0xc0; ++i)
      EXPECT_EQ(record[i], 0xa5);
   for (unsigned i = 0xd8; i < sizeof(record); ++i)
      EXPECT_EQ(record[i], 0xa5);
   EXPECT_EQ(get_u64(record, 0x00), geometry.group_counts);
   EXPECT_EQ(get_u64(record, 0x08), address + 0xcc);
   for (unsigned d = 0; d < 3; ++d) {
      EXPECT_EQ(get_u32(record, 0xc0 + d * 4), 0u);
      EXPECT_EQ(get_u32(record, 0xcc + d * 4), geometry.local[d]);
   }
}

TEST(Apple9ComputeGeometry, RejectsMalformedIndirectGeometry)
{
   uint8_t record[0x100] = {};
   agx_apple9_compute_geometry geometry = {
      .mode = AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT,
      .group_counts = 0x2fff123402ull,
      .local = {4, 4, 1},
   };
   EXPECT_FALSE(agx_apple9_build_compute_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
   geometry.group_counts &= ~3ull;
   geometry.local[2] = 65;
   geometry.local[1] = 17;
   EXPECT_FALSE(agx_apple9_build_compute_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
   EXPECT_FALSE(agx_apple9_build_compute_geometry_fields(
      record, 0xd7, 0x1000200000ull, &geometry));
}

TEST(Apple9ComputeGeometry, RejectsMalformedDirectGeometry)
{
   uint8_t record[0x100] = {};
   agx_apple9_compute_geometry geometry = {
      .mode = AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT,
      .threads = {12, 10, 0},
      .local = {4, 5, 3},
   };
   EXPECT_FALSE(agx_apple9_build_compute_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));

   geometry.threads[2] = 6;
   geometry.local[0] = 1024;
   geometry.local[1] = 2;
   EXPECT_FALSE(agx_apple9_build_compute_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));

   geometry.mode = (agx_apple9_compute_geometry_mode)99;
   EXPECT_FALSE(agx_apple9_build_compute_geometry_fields(
      record, sizeof(record), 0x1000200000ull, &geometry));
}
