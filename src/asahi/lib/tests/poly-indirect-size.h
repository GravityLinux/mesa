/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Values of enum poly_gs_shape (poly/geometry.h is C-only). */
enum {
   TEST_POLY_GS_SHAPE_DYNAMIC_INDEXED,
   TEST_POLY_GS_SHAPE_STATIC_INDEXED,
   TEST_POLY_GS_SHAPE_STATIC_PER_PRIM,
   TEST_POLY_GS_SHAPE_STATIC_PER_INSTANCE,
};

bool test_poly_gs_indirect_fits(unsigned prim, unsigned shape,
                                uint32_t max_indices, bool prefix_sum,
                                uint32_t count_stride_B, uint32_t vertex_count,
                                uint32_t instance_count, uint64_t vs_outputs,
                                uint32_t limit_B);

bool test_poly_tess_indirect_fits(uint32_t vertex_count,
                                  uint32_t instance_count,
                                  uint32_t input_patch_size,
                                  uint32_t tcs_stride_el, uint64_t vs_outputs,
                                  uint32_t limit_B);

#ifdef __cplusplus
}
#endif
