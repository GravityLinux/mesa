/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "nir.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "asahi/compiler/agx_nir.h"
#ifdef __cplusplus
}
#endif

/* Native compute uses subgroup32 (agx_compile_apple9.c). A residual matrix
 * means preprocessing failed, including after a preprocessed-NIR cache hit.
 * Never allow descriptor or scratch lowering to inspect its unsized types.
 */
static inline const char *
hk_apple9_coopmat_rejection(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE ||
       !nir->info.cs.has_cooperative_matrix)
      return NULL;

   const char *reason = NULL;
   agx_nir_lower_apple9_cooperative_matrix(nir, 32, &reason);
   return reason ? reason : "Cooperative matrix preprocessing required";
}
