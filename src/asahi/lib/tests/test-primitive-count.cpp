/* SPDX-License-Identifier: MIT */
#include "agx_primitive_count.h"
#include <gtest/gtest.h>

TEST(PrimitiveCount, TopologiesAndIncompletePrimitives)
{
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_POINTS, 4, 1), 4);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_LINES, 9, 1), 4);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_LINE_STRIP, 5, 1), 4);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_LINE_LOOP, 4, 1), 4);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_TRIANGLES, 14, 1), 4);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_TRIANGLE_STRIP, 6, 1), 4);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_TRIANGLE_FAN, 6, 1), 4);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_LINE_LOOP, 1, 1), 0);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_LINE_STRIP, 0, 1), 0);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_TRIANGLES, 2, 1), 0);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_TRIANGLE_FAN, 1, 1), 0);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_QUADS, 4, 1), 5);
}

TEST(PrimitiveCount, DrawsInstancesAndMetaTriangles)
{
   uint8_t count = 0;
   EXPECT_FALSE(agx_has_few_primitives(count));
   agx_add_primitive_count(&count, agx_count_primitives(AGX_PRIMITIVE_TRIANGLES, 3, 2));
   EXPECT_EQ(count, 2);
   EXPECT_TRUE(agx_has_few_primitives(count));
   /* A rasterized meta rectangle contributes two ordinary triangles. */
   agx_add_primitive_count(&count, agx_count_primitives(AGX_PRIMITIVE_TRIANGLES, 6, 1));
   EXPECT_EQ(count, 4);
   EXPECT_TRUE(agx_has_few_primitives(count));
   agx_add_primitive_count(&count, agx_count_primitives(AGX_PRIMITIVE_POINTS, 1, 1));
   EXPECT_EQ(count, 5);
   EXPECT_FALSE(agx_has_few_primitives(count));
}

TEST(PrimitiveCount, UnknownAndOverflowStayIneligible)
{
   uint8_t count = 1;
   agx_add_primitive_count(&count, AGX_PRIMITIVE_COUNT_UNKNOWN);
   agx_add_primitive_count(&count, 0);
   EXPECT_FALSE(agx_has_few_primitives(count));
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_POINTS, UINT32_MAX, UINT32_MAX), 5);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_TRIANGLES, UINT32_MAX, UINT32_MAX), 5);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_POINTS, 1024, 1), 5);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_TRIANGLES, 3, 5), 5);
   EXPECT_EQ(agx_count_primitives(AGX_PRIMITIVE_POINTS, UINT32_MAX, 0), 0);
}
