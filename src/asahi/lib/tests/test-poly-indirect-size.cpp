/* SPDX-License-Identifier: MIT */
#include "compiler/shader_enums.h"
#include <gtest/gtest.h>
#include "poly-indirect-size.h"

/* Position plus one varying: 32 bytes of software VS output per vertex. */
static const uint64_t two_outputs = 0x3;
static const uint32_t heap_B = 128 * 1024 * 1024;

TEST(PolyIndirectSize, GeometryFitsOrdinaryDraws)
{
   EXPECT_TRUE(test_poly_gs_indirect_fits(
      MESA_PRIM_TRIANGLES, TEST_POLY_GS_SHAPE_DYNAMIC_INDEXED, 3, true, 4, 3000,
      4, two_outputs, heap_B));
   EXPECT_TRUE(test_poly_gs_indirect_fits(MESA_PRIM_POINTS,
                                          TEST_POLY_GS_SHAPE_STATIC_PER_PRIM, 1,
                                          false, 0, 1, 1, two_outputs, heap_B));
   /* Exactly the limit is allocatable. */
   EXPECT_TRUE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_STATIC_PER_PRIM, 1, false, 0,
      heap_B / 32, 1, two_outputs, heap_B));
   EXPECT_FALSE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_STATIC_PER_PRIM, 1, false, 0,
      heap_B / 32 + 1, 1, two_outputs, heap_B));
}

TEST(PolyIndirectSize, GeometryRejectsCountsThatWrap32Bit)
{
   /* UINT32_MAX * 32 B wraps to 0xffffffe0 in 32 bits, and products of
    * vertex and instance counts wrap to small values.
    */
   EXPECT_FALSE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_STATIC_PER_PRIM, 1, false, 0,
      UINT32_MAX, 1, two_outputs, heap_B));
   EXPECT_FALSE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_STATIC_PER_PRIM, 1, false, 0,
      1u << 16, 1u << 16, two_outputs, heap_B));
   EXPECT_FALSE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_STATIC_PER_PRIM, 1, false, 0,
      UINT32_MAX, UINT32_MAX, ~0ull, UINT32_MAX));
}

TEST(PolyIndirectSize, GeometryChecksCountAndIndexBuffers)
{
   /* No VS outputs, but per-primitive counts and dynamic indices still
    * scale with the draw.
    */
   EXPECT_FALSE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_STATIC_PER_PRIM, 1, true, 16,
      heap_B / 8, 1, 0, heap_B));
   EXPECT_FALSE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_DYNAMIC_INDEXED, 256, false, 0,
      1u << 20, 1, 0, heap_B));
   EXPECT_FALSE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_DYNAMIC_INDEXED, 4, false, 0,
      UINT32_MAX, 1, 0, heap_B));
   /* Without outputs or per-primitive buffers, no heap is needed. */
   EXPECT_TRUE(test_poly_gs_indirect_fits(
      MESA_PRIM_POINTS, TEST_POLY_GS_SHAPE_STATIC_PER_PRIM, 1, false, 0,
      UINT32_MAX, UINT32_MAX, 0, heap_B));
}

TEST(PolyIndirectSize, Tessellation)
{
   EXPECT_TRUE(
      test_poly_tess_indirect_fits(300, 2, 3, 32, two_outputs, heap_B));
   EXPECT_FALSE(
      test_poly_tess_indirect_fits(UINT32_MAX, 1, 3, 32, two_outputs, heap_B));
   EXPECT_FALSE(test_poly_tess_indirect_fits(UINT32_MAX, UINT32_MAX, 1, 1024,
                                             ~0ull, UINT32_MAX));
   /* Patch buffers alone: 1 << 22 patches of 136 B exceed 128 MiB. */
   EXPECT_FALSE(test_poly_tess_indirect_fits(4u << 22, 1, 4, 32, 0, heap_B));
}
