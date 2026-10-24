/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"
#include "poly/tessellator.h"

#pragma once

/* One draw of a folded tessellator dispatch, which runs a pass of several
 * independent draws at once. The draws' workgroups are laid out back to back:
 * each draw has DIV_ROUND_UP(nr_patches, LIBAGX_TESS_FOLD_GROUP) of them,
 * starting at first_group, ascending with the draw index.
 */
struct libagx_tess_fold_draw {
   /* Address of the draw's struct poly_tess_params, read as constant by the
    * tessellator and as global by the prefix sum.
    */
   uint64_t p;
   uint32_t first_group;
   uint32_t nr_patches;
};
static_assert(sizeof(struct libagx_tess_fold_draw) == 16, "packed");

/* Workgroup size of the folded tessellator kernels */
#define LIBAGX_TESS_FOLD_GROUP 64
