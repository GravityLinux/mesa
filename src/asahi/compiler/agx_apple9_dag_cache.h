/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <string.h>
#include "agx_apple9_ir.h"
#include "agx_compile.h"

struct agx_apple9_export_memo {
   uint32_t value, publication;
   bool raw;
};

/* All synthetic DAG value caches belong here. Block identity alone does not
 * distinguish a closed callee from its caller. New members must use the empty
 * zero representation, or be explicitly initialized by reset below.
 */
struct agx_apple9_dag_cache {
   /* Remember every publication that can fit the bank, including late clip
    * distance writes that share values with rasterizer copies. */
   struct agx_apple9_export_memo export_memo[AGX_APPLE9_PUBLICATION_COUNT];
   unsigned export_memo_count;
   nir_block *export_memo_block;
   uint32_t system_vreg[256];
   struct agx_apple9_block *system_block;
   uint32_t zero_vreg;
   struct agx_apple9_block *zero_block;
   uint32_t entry_zero;
   struct agx_apple9_block *binding_pointer_block;
   uint32_t binding_pointer[AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
   uint32_t perspective_reciprocal;
   struct agx_apple9_block *perspective_block;
};

static inline void
agx_apple9_dag_cache_reset(struct agx_apple9_dag_cache *cache)
{
   /* Clear the complete aggregate, including future cache members, then set
    * invalid-value sentinels. Entry hoisting can run before a block exists. */
   memset(cache, 0, sizeof(*cache));
   cache->zero_vreg = AGX_APPLE9_VREG_INVALID;
   cache->entry_zero = AGX_APPLE9_VREG_INVALID;
   cache->perspective_reciprocal = AGX_APPLE9_VREG_INVALID;
   memset(cache->system_vreg, 0xff, sizeof(cache->system_vreg));
   memset(cache->binding_pointer, 0xff, sizeof(cache->binding_pointer));
}
