/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_entries.h"
#include <assert.h>

static bool
plan_entries(const struct agx_apple9_entry_table *table, const uint64_t code[2],
             unsigned slots[2])
{
   assert(table->count <= AGX_APPLE9_RENDER_MAX_ENTRIES);
   unsigned next = table->count;
   for (unsigned stage = 0; stage < 2; stage++) {
      unsigned i;
      for (i = 0; i < table->count; i++) {
         if (table->entries[i].stage == stage &&
             table->entries[i].code == code[stage])
            break;
      }
      slots[stage] = i < table->count ? i : next++;
   }
   return next <= AGX_APPLE9_RENDER_MAX_ENTRIES;
}

bool
agx_apple9_entry_table_fits(const struct agx_apple9_entry_table *table,
                           const uint64_t code[2])
{
   unsigned slots[2];
   return plan_entries(table, code, slots);
}

bool
agx_apple9_entry_table_add(struct agx_apple9_entry_table *table,
                          const uint64_t code[2], unsigned offsets[2])
{
   unsigned slots[2];
   if (!plan_entries(table, code, slots))
      return false;

   for (unsigned stage = 0; stage < 2; stage++) {
      if (slots[stage] >= table->count) {
         assert(slots[stage] == table->count);
         table->entries[table->count++] =
            (struct agx_apple9_entry){.code = code[stage], .stage = stage};
      }
      offsets[stage] = agx_apple9_entry_offset(slots[stage]);
   }
   return true;
}
