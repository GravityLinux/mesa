/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "hk_descriptor_memory.h"

/* Stable USC addresses: slabs are retained until device destruction. Entries
 * return to their slab when their view is destroyed. Caller serializes access. */
#define HK_APPLE9_PBE_SLAB_SIZE (64 * 1024)
struct hk_apple9_pbe_table {
   struct list_head slabs;
   const struct hk_descriptor_memory_ops *ops;
   void *data;
};

#ifdef __cplusplus
extern "C" {
#endif

void hk_apple9_pbe_table_init(struct hk_apple9_pbe_table *table,
                              const struct hk_descriptor_memory_ops *ops,
                              void *data);
bool hk_apple9_pbe_table_alloc(struct hk_apple9_pbe_table *table,
                               unsigned count,
                               struct hk_descriptor_allocation *out);
void hk_apple9_pbe_table_free(struct hk_apple9_pbe_table *table,
                              struct hk_descriptor_allocation *alloc);
void hk_apple9_pbe_table_finish(struct hk_apple9_pbe_table *table);

#ifdef __cplusplus
}
#endif
