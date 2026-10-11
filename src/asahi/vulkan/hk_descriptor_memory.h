/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "util/list.h"
#include "util/vma.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Descriptor pool memory. A pool whose worst case is small is backed by one
 * buffer created up front, as most pools are allocated from and freed at high
 * rates. A larger pool creates nothing up front: it may never need its worst
 * case (vkd3d-proton's mutable heaps declare no type list, so each descriptor
 * reserves the largest record while the sets it allocates use their layout's
 * smaller stride). Its large sets get a buffer of their own, and small sets
 * share chunks that persist until the pool is destroyed.
 */
#define HK_DESCRIPTOR_MEMORY_EAGER_MAX     (16ull << 20)
#define HK_DESCRIPTOR_MEMORY_DEDICATED_MIN (1ull << 20)
#define HK_DESCRIPTOR_MEMORY_CHUNK         (4ull << 20)

struct hk_descriptor_backing {
   void *bo;
   uint64_t addr;
   uint8_t *map;
   uint64_t size;
};

struct hk_descriptor_memory_ops {
   /* Create a mapped buffer of at least size bytes; false on failure. */
   bool (*create)(void *data, uint64_t size, struct hk_descriptor_backing *out);
   void (*destroy)(void *data, struct hk_descriptor_backing *backing);
};

struct hk_descriptor_chunk {
   struct list_head link;
   struct hk_descriptor_backing backing;
   struct util_vma_heap heap;
};

struct hk_descriptor_memory {
   const struct hk_descriptor_memory_ops *ops;
   void *data;
   bool lazy;
   /* Lazy pools: bytes handed out against the worst case. */
   uint64_t budget, used;
   struct list_head chunks;
};

/* One allocation. dedicated.bo is set when it owns its buffer. */
struct hk_descriptor_allocation {
   uint64_t addr;
   uint8_t *map;
   uint64_t size;
   struct hk_descriptor_backing dedicated;
};

enum hk_descriptor_memory_result {
   HK_DESCRIPTOR_MEMORY_OK,
   HK_DESCRIPTOR_MEMORY_OUT_OF_POOL,
   HK_DESCRIPTOR_MEMORY_OUT_OF_DEVICE,
};

enum hk_descriptor_memory_result
hk_descriptor_memory_init(struct hk_descriptor_memory *mem,
                          const struct hk_descriptor_memory_ops *ops,
                          void *data, uint64_t worst_case);

enum hk_descriptor_memory_result
hk_descriptor_memory_alloc(struct hk_descriptor_memory *mem, uint64_t size,
                           uint64_t alignment,
                           struct hk_descriptor_allocation *out);

void hk_descriptor_memory_free(struct hk_descriptor_memory *mem,
                               struct hk_descriptor_allocation *alloc);

void hk_descriptor_memory_finish(struct hk_descriptor_memory *mem);

#ifdef __cplusplus
}
#endif
