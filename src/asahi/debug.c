/* SPDX-License-Identifier: MIT */
#include "debug.h"
#include <string.h>
#include "c11/threads.h"
#include "util/u_debug.h"

static const struct debug_named_value apple9_debug_options[] = {
   {"asm", AGX_A9_DBG_ASM, "Final packed instruction bytes"},
   {"binary", AGX_A9_DBG_BINARY, "Complete shader binary"},
   {"stats", AGX_A9_DBG_STATS, "Compiler statistics"},
   {"trace", AGX_A9_DBG_TRACE, "Compiler lowering trace"},
   {"nir", AGX_A9_DBG_NIR, "Intermediate NIR"},
   {"final_nir", AGX_A9_DBG_FINAL_NIR, "Final NIR"},
   {"shared_ra", AGX_A9_DBG_SHARED_RA, "Shared register allocator trace"},
   {"ra_audit", AGX_A9_DBG_RA_AUDIT, "Register allocation audit"},
   {"ra_webs", AGX_A9_DBG_RA_WEBS, "Phi-web audit (also enable ra_audit)"},
   {"spill_loops", AGX_A9_DBG_SPILL_LOOPS, "Loop spilling trace"},
   {"pressure", AGX_A9_DBG_PRESSURE, "NIR register pressure statistics"},
   {"vectorize", AGX_A9_DBG_VECTORIZE, "Load vectorization trace"},
   {"eager", AGX_A9_DBG_EAGER, "Eager variant trace"},
   {"merge", AGX_A9_DBG_MERGE, "Render merge statistics"},
   {"upload", AGX_A9_DBG_UPLOAD, "Command upload statistics"},
   {"tiling", AGX_A9_DBG_TILING, "Tiling dependency statistics"},
   {"launch_check", AGX_A9_DBG_LAUNCH_CHECK, "Compare launch template packing"},
   {"bo_labels", AGX_A9_DBG_BO_LABELS, "Buffer allocation labels"},
   {"replay", AGX_A9_DBG_REPLAY, "Capture replay addresses"},
   DEBUG_NAMED_VALUE_END,
};

/* Preserve existing worker scripts, including presence-only aliases whose
 * empty string and "0" values enable output. Boolean aliases retain Mesa's
 * boolean parser. Each alias only enables its corresponding diagnostic. */
static const struct {
   const char *name;
   uint64_t flag;
   bool boolean;
} apple9_debug_aliases[] = {
   {"AGX_APPLE9_ASM", AGX_A9_DBG_ASM, false},
   {"AGX_APPLE9_DUMP_BINARY", AGX_A9_DBG_BINARY, false},
   {"AGX_APPLE9_STATS", AGX_A9_DBG_STATS, false},
   {"AGX_APPLE9_TRACE", AGX_A9_DBG_TRACE, false},
   {"AGX_APPLE9_TRACE_NIR", AGX_A9_DBG_NIR, false},
   {"AGX_APPLE9_TRACE_FINAL_NIR", AGX_A9_DBG_FINAL_NIR, false},
   {"AGX_APPLE9_TRACE_SHARED_RA", AGX_A9_DBG_SHARED_RA, false},
   {"AGX_APPLE9_RA_AUDIT", AGX_A9_DBG_RA_AUDIT, false},
   {"AGX_APPLE9_RA_WEBS", AGX_A9_DBG_RA_WEBS, false},
   {"AGX_SPILL_TRACE_LOOPS", AGX_A9_DBG_SPILL_LOOPS, true},
   {"HK_APPLE9_PRESSURE_STATS", AGX_A9_DBG_PRESSURE, false},
   {"HK_APPLE9_TRACE_VECTORIZE", AGX_A9_DBG_VECTORIZE, false},
   {"HK_APPLE9_EAGER_TRACE", AGX_A9_DBG_EAGER, false},
   {"HK_APPLE9_MERGE_TRACE", AGX_A9_DBG_MERGE, true},
   {"HK_UPLOAD_STATS", AGX_A9_DBG_UPLOAD, true},
   {"HK_TILING_ELISION_STATS", AGX_A9_DBG_TILING, true},
   {"HK_APPLE9_LAUNCH_CHECK", AGX_A9_DBG_LAUNCH_CHECK, true},
   {"AGX_BO_TRACE_LABELS", AGX_A9_DBG_BO_LABELS, true},
   {"HK_REPLAY_DEBUG", AGX_A9_DBG_REPLAY, true},
};

static once_flag apple9_debug_once = ONCE_FLAG_INIT;
static uint64_t apple9_debug_value;

static void
apple9_debug_init(void)
{
   apple9_debug_value = debug_get_flags_option("AGX_APPLE9_DEBUG",
                                              apple9_debug_options, 0);
   for (unsigned i = 0; i < ARRAY_SIZE(apple9_debug_aliases); ++i) {
      const char *name = apple9_debug_aliases[i].name;
      bool enabled = apple9_debug_aliases[i].boolean
         ? debug_get_bool_option(name, false)
         : debug_get_option(name, NULL) != NULL;
      if (enabled)
         apple9_debug_value |= apple9_debug_aliases[i].flag;
   }
}

uint64_t
agx_apple9_debug_flags(void)
{
   call_once(&apple9_debug_once, apple9_debug_init);
   return apple9_debug_value;
}

/* Accept NAME or NAME=value, as used by the compiler cache environment hash. */
bool
agx_apple9_is_debug_env(const char *entry)
{
   size_t len = strcspn(entry, "=");
   if (len == strlen("AGX_APPLE9_DEBUG") &&
       !strncmp(entry, "AGX_APPLE9_DEBUG", len))
      return true;

   for (unsigned i = 0; i < ARRAY_SIZE(apple9_debug_aliases); ++i) {
      const char *name = apple9_debug_aliases[i].name;
      if (strlen(name) == len && !strncmp(entry, name, len))
         return true;
   }
   return false;
}
