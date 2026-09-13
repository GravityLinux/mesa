/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <cstring>
#include "gtest/gtest.h"

extern "C" {
#include "agx_apple9_entries.h"
}

TEST(Apple9Entries, RepeatedDrawsShareTargets)
{
   agx_apple9_entry_table table = {};
   uint64_t code[2] = {0x10001000000ull, 0x10001004000ull};
   unsigned first[2], offsets[2];
   ASSERT_TRUE(agx_apple9_entry_table_add(&table, code, first));
   for (unsigned i = 0; i < 10000; i++) {
      ASSERT_TRUE(agx_apple9_entry_table_fits(&table, code));
      ASSERT_TRUE(agx_apple9_entry_table_add(&table, code, offsets));
      EXPECT_EQ(offsets[0], first[0]);
      EXPECT_EQ(offsets[1], first[1]);
   }
   EXPECT_EQ(table.count, 2u);
}

TEST(Apple9Entries, StagesShareIndependently)
{
   agx_apple9_entry_table table = {};
   uint64_t code[2] = {0x10001000000ull, 0x10001004000ull};
   unsigned first[2], second[2];
   ASSERT_TRUE(agx_apple9_entry_table_add(&table, code, first));
   code[1] += 0x4000;
   ASSERT_TRUE(agx_apple9_entry_table_add(&table, code, second));
   EXPECT_EQ(table.count, 3u);
   EXPECT_EQ(first[0], second[0]);
   EXPECT_NE(first[1], second[1]);

   /* The same address in a different stage has a distinct ABI key. */
   code[1] = code[0];
   ASSERT_TRUE(agx_apple9_entry_table_add(&table, code, second));
   EXPECT_EQ(table.count, 4u);
   EXPECT_NE(second[0], second[1]);
}

TEST(Apple9Entries, FullTableStillAcceptsExistingTargets)
{
   agx_apple9_entry_table table = {};
   uint64_t code[2] = {0x10001000000ull, 0x10001004000ull};
   unsigned offsets[2];
   ASSERT_TRUE(agx_apple9_entry_table_add(&table, code, offsets));
   while (table.count < AGX_APPLE9_RENDER_MAX_ENTRIES) {
      code[1] += 0x4000;
      ASSERT_TRUE(agx_apple9_entry_table_add(&table, code, offsets));
   }
   EXPECT_TRUE(agx_apple9_entry_table_fits(&table, code));
   EXPECT_TRUE(agx_apple9_entry_table_add(&table, code, offsets));
   EXPECT_LE(offsets[1] + 32u, AGX_APPLE9_RENDER_ENTRY_REGION_SIZE);

   auto saved = table;
   unsigned untouched[2] = {0xfeed, 0xbeef};
   code[1] += 0x4000;
   EXPECT_FALSE(agx_apple9_entry_table_fits(&table, code));
   EXPECT_FALSE(agx_apple9_entry_table_add(&table, code, untouched));
   EXPECT_EQ(std::memcmp(&saved, &table, sizeof(table)), 0);
   EXPECT_EQ(untouched[0], 0xfeedu);
   EXPECT_EQ(untouched[1], 0xbeefu);
}

TEST(Apple9Entries, TwoStageAdmissionIsAtomic)
{
   agx_apple9_entry_table table = {};
   uint64_t code[2] = {0x10001000000ull, 0x10001004000ull};
   unsigned offsets[2];
   while (table.count < AGX_APPLE9_RENDER_MAX_ENTRIES - 1) {
      code[1] += 0x4000;
      ASSERT_TRUE(agx_apple9_entry_table_add(&table, code, offsets));
   }
   auto saved = table;
   uint64_t fresh[2] = {0x10010000000ull, 0x10010004000ull};
   EXPECT_FALSE(agx_apple9_entry_table_add(&table, fresh, offsets));
   EXPECT_EQ(std::memcmp(&saved, &table, sizeof(table)), 0);
   fresh[0] = code[0];
   EXPECT_TRUE(agx_apple9_entry_table_add(&table, fresh, offsets));
   EXPECT_EQ(table.count, AGX_APPLE9_RENDER_MAX_ENTRIES);
}

TEST(Apple9Entries, BatchesOwnIndependentTables)
{
   agx_apple9_entry_table first = {}, next = {};
   uint64_t code[2] = {0x10001000000ull, 0x10001004000ull};
   unsigned a[2], b[2];
   ASSERT_TRUE(agx_apple9_entry_table_add(&first, code, a));
   code[0] += 0x8000;
   ASSERT_TRUE(agx_apple9_entry_table_add(&next, code, b));
   EXPECT_EQ(a[0], b[0]);
   EXPECT_NE(first.entries[0].code, next.entries[0].code);
   EXPECT_EQ(first.count, 2u);
}
