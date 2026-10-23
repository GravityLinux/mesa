/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "hk_descriptor_memory.h"

/* Immutable shader and image-view state at stable USC addresses. Slabs remain
 * until device destruction; allocations return when their owner is destroyed.
 * Caller serializes access. */
#define HK_USC_HEAP_SLAB_SIZE (64 * 1024)
struct hk_usc_heap {
   struct list_head slabs;
   const struct hk_descriptor_memory_ops *ops;
   void *data;
};

#ifdef __cplusplus
extern "C" {
#endif

void hk_usc_heap_init(struct hk_usc_heap *table,
                              const struct hk_descriptor_memory_ops *ops,
                              void *data);
bool hk_usc_heap_alloc(struct hk_usc_heap *table,
                               uint64_t size, unsigned alignment,
                               struct hk_descriptor_allocation *out);
void hk_usc_heap_free(struct hk_usc_heap *table,
                              struct hk_descriptor_allocation *alloc);
void hk_usc_heap_finish(struct hk_usc_heap *table);

#ifdef __cplusplus
}
#endif
