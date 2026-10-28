/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_APPLE9_SOFTFP64_H
#define HK_APPLE9_SOFTFP64_H

#include "compiler/nir/nir.h"
#include "util/simple_mtx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Device-owned, lazily imported library. All accesses hold lock, including
 * inlining; finish runs after shader compilation has quiesced. */
struct hk_apple9_softfp64 {
   simple_mtx_t lock;
   nir_shader *library;
};

void hk_apple9_softfp64_init(struct hk_apple9_softfp64 *cache);
void hk_apple9_softfp64_finish(struct hk_apple9_softfp64 *cache);
bool hk_apple9_nir_uses_double(const nir_shader *nir);

/* Success, not progress. Unsupported operations fail before library inlining.
 * On failure reason names the residual opcode or a static load diagnostic.
 * Does not consume nir. No library is loaded for shaders without doubles. */
bool hk_apple9_lower_softfp64(struct hk_apple9_softfp64 *cache, nir_shader *nir,
                             const char **reason);

#ifdef __cplusplus
}
#endif
#endif
