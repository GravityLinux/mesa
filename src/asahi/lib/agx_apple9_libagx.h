/*
 * Copyright 2026 Niklas Sheth
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/nir/nir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Deserialize a libagx program and lower its kernel inputs to UBO 0, ready
 * for agx_compile_apple9_tiny.
 */
nir_shader *agx_apple9_libagx_nir(unsigned program, void *mem_ctx);

#ifdef __cplusplus
}
#endif
