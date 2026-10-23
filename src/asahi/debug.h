/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Diagnostic-only flags. Keep code-generation controls out of this namespace. */
enum agx_apple9_debug {
   AGX_A9_DBG_ASM = BITFIELD_BIT(0),
   AGX_A9_DBG_BINARY = BITFIELD_BIT(1),
   AGX_A9_DBG_STATS = BITFIELD_BIT(2),
   AGX_A9_DBG_TRACE = BITFIELD_BIT(3),
   AGX_A9_DBG_NIR = BITFIELD_BIT(4),
   AGX_A9_DBG_FINAL_NIR = BITFIELD_BIT(5),
   AGX_A9_DBG_SHARED_RA = BITFIELD_BIT(6),
   AGX_A9_DBG_RA_AUDIT = BITFIELD_BIT(7),
   AGX_A9_DBG_RA_WEBS = BITFIELD_BIT(8),
   AGX_A9_DBG_SPILL_LOOPS = BITFIELD_BIT(9),
   AGX_A9_DBG_PRESSURE = BITFIELD_BIT(10),
   AGX_A9_DBG_VECTORIZE = BITFIELD_BIT(11),
   AGX_A9_DBG_EAGER = BITFIELD_BIT(12),
   AGX_A9_DBG_MERGE = BITFIELD_BIT(13),
   AGX_A9_DBG_UPLOAD = BITFIELD_BIT(14),
   AGX_A9_DBG_TILING = BITFIELD_BIT(15),
   AGX_A9_DBG_LAUNCH_CHECK = BITFIELD_BIT(16),
   AGX_A9_DBG_BO_LABELS = BITFIELD_BIT(17),
   AGX_A9_DBG_REPLAY = BITFIELD_BIT(18),
};

uint64_t agx_apple9_debug_flags(void);
bool agx_apple9_is_debug_env(const char *entry);

static inline bool
agx_apple9_debug_enabled(enum agx_apple9_debug flag)
{
   return (agx_apple9_debug_flags() & flag) != 0;
}

#ifdef __cplusplus
} /* extern C */
#endif
