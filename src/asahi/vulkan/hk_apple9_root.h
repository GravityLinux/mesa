/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "nir.h"

struct vk_pipeline_robustness_state;
struct vk_descriptor_set_layout;

#ifdef __cplusplus
extern "C" {
#endif

bool hk_nir_lower_buffer_descriptors(
   nir_shader *nir, const struct vk_pipeline_robustness_state *rs,
   uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts);

/* Resolve the native root UBO and descriptor-set pointers from that root. */
bool hk_nir_lower_apple9_root(nir_shader *nir);

/* Shared final conversion used by the driver and the pass tests. */
bool hk_nir_lower_global_constant_offsets(nir_shader *nir, bool soft_fault);

#ifdef __cplusplus
}
#endif
