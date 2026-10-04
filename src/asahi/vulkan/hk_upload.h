/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdint.h>
#include "util/u_math.h"

struct hk_upload_chunk {
   uint8_t *map;
   uint64_t base;
   uint32_t offset;
};

/* Keep two independently usable tails. Large stream blocks can then share a
 * BO while small state uploads finish the tail of the previous BO.
 */
#define HK_UPLOAD_CHUNKS 2

static inline struct hk_upload_chunk *
hk_upload_find_chunk(struct hk_upload_chunk chunks[HK_UPLOAD_CHUNKS],
                     uint32_t capacity, uint32_t size, uint32_t alignment)
{
   struct hk_upload_chunk *best = NULL;
   for (unsigned i = 0; i < HK_UPLOAD_CHUNKS; ++i) {
      struct hk_upload_chunk *chunk = &chunks[i];
      uint32_t offset = align(chunk->offset, alignment);
      if (!chunk->map || offset > capacity || size > capacity - offset)
         continue;

      /* Best fit leaves the larger tail available for larger uploads. */
      if (!best || chunk->offset > best->offset)
         best = chunk;
   }
   return best;
}

static inline void
hk_upload_keep_chunk(struct hk_upload_chunk chunks[HK_UPLOAD_CHUNKS],
                     struct hk_upload_chunk chunk)
{
   unsigned smallest = 0;
   for (unsigned i = 0; i < HK_UPLOAD_CHUNKS; ++i) {
      if (!chunks[i].map) {
         chunks[i] = chunk;
         return;
      }
      if (chunks[i].offset > chunks[smallest].offset)
         smallest = i;
   }

   /* The new allocation may leave less space than either existing tail. */
   if (chunk.offset < chunks[smallest].offset)
      chunks[smallest] = chunk;
}
