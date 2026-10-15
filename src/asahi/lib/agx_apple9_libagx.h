/*
 * Copyright 2026 Niklas Sheth
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/nir/nir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Build tools may be 64-bit while the driver is 32-bit. Never embed a raw
 * shader_info here: its pointers and padding have different sizes. The NIR
 * metadata is followed by a NIR function stream, without shader-global state.
 * These kernel properties supplement gathered instruction information.
 */
struct agx_apple9_libagx_metadata {
   uint32_t workgroup_size[3];
   uint32_t shared_size;
   uint32_t scratch_size;
   uint32_t float_controls;
   uint32_t api_subgroup_size;
   uint32_t min_subgroup_size;
   uint32_t max_subgroup_size;
   uint32_t variable_local_size;
};

static_assert(sizeof(struct agx_apple9_libagx_metadata) == 40,
              "libagx kernel metadata must be independent of pointer size");

/* Deserialize a libagx program and lower its kernel inputs to UBO 0, ready
 * for agx_compile_apple9_tiny.
 */
nir_shader *agx_apple9_libagx_nir(unsigned program, void *mem_ctx);

#ifdef __cplusplus
}
#endif
