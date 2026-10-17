/* SPDX-License-Identifier: MIT */
/* geometry.h is C-only; expose its size checks to the C++ tests. */
#include "poly-indirect-size.h"
#include "poly/geometry.h"

static_assert(
   TEST_POLY_GS_SHAPE_DYNAMIC_INDEXED == POLY_GS_SHAPE_DYNAMIC_INDEXED &&
      TEST_POLY_GS_SHAPE_STATIC_INDEXED == POLY_GS_SHAPE_STATIC_INDEXED &&
      TEST_POLY_GS_SHAPE_STATIC_PER_PRIM == POLY_GS_SHAPE_STATIC_PER_PRIM &&
      TEST_POLY_GS_SHAPE_STATIC_PER_INSTANCE ==
         POLY_GS_SHAPE_STATIC_PER_INSTANCE,
   "shape values must match poly/geometry.h");

bool
test_poly_gs_indirect_fits(unsigned prim, unsigned shape, uint32_t max_indices,
                           bool prefix_sum, uint32_t count_stride_B,
                           uint32_t vertex_count, uint32_t instance_count,
                           uint64_t vs_outputs, uint32_t limit_B)
{
   return poly_gs_indirect_fits(
      (enum mesa_prim)prim, (enum poly_gs_shape)shape, max_indices, prefix_sum,
      count_stride_B, vertex_count, instance_count, vs_outputs, limit_B);
}

bool
test_poly_tess_indirect_fits(uint32_t vertex_count, uint32_t instance_count,
                             uint32_t input_patch_size, uint32_t tcs_stride_el,
                             uint64_t vs_outputs, uint32_t limit_B)
{
   return poly_tess_indirect_fits(vertex_count, instance_count,
                                  input_patch_size, tcs_stride_el, vs_outputs,
                                  limit_B);
}
