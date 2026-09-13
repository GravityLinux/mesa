/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_ENTRIES_H
#define AGX_APPLE9_ENTRIES_H

#include <stdbool.h>
#include <stdint.h>
#include "asahi/lib/agx_apple9_layout.h"

/* Entries are immutable within a batch. The batch retains each shader BO,
 * so its address cannot be recycled while the table is in use. Stage is
 * part of the key to keep the vertex and fragment entry ABIs independent. */
struct agx_apple9_entry {
   uint64_t code;
   unsigned stage;
};

struct agx_apple9_entry_table {
   unsigned count;
   struct agx_apple9_entry entries[AGX_APPLE9_RENDER_MAX_ENTRIES];
};

static inline unsigned
agx_apple9_entry_offset(unsigned slot)
{
   return AGX_APPLE9_RENDER_ENTRY_HEADER_SIZE +
          slot * AGX_APPLE9_RENDER_ENTRY_BLOCK_SIZE +
          AGX_APPLE9_RENDER_ENTRY_CODE_OFFSET;
}

bool agx_apple9_entry_table_fits(const struct agx_apple9_entry_table *table,
                                const uint64_t code[2]);

/* Admit both stages together. Failure leaves the table and offsets unchanged. */
bool agx_apple9_entry_table_add(struct agx_apple9_entry_table *table,
                               const uint64_t code[2], unsigned offsets[2]);

#endif
