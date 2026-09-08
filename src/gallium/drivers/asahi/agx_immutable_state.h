/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_IMMUTABLE_STATE_H
#define AGX_IMMUTABLE_STATE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* A state's differences from an immutable reference image. Construction
 * compares bytes once. Plans may name an exact predecessor or a common
 * baseline; the latter requires restoring the union of changed ranges. */
struct agx_state_range { uint32_t offset, size; };
struct agx_immutable_state {
   const uint8_t *data;
   uint32_t size, count;
   struct agx_state_range *ranges;
};

static inline bool
agx_immutable_state_init(struct agx_immutable_state *state, const void *data,
                         const void *baseline, uint32_t size)
{
   const uint8_t *src = data, *base = baseline;
   uint32_t count = 0;
   bool previous = false;
   for (uint32_t offset = 0; offset < size; offset += 64) {
      uint32_t len = size - offset < 64 ? size - offset : 64;
      bool changed = memcmp(src + offset, base + offset, len) != 0;
      count += changed && !previous;
      previous = changed;
   }
   struct agx_state_range *ranges = count ? calloc(count, sizeof(*ranges)) : NULL;
   if (count && !ranges)
      return false;
   uint32_t n = 0;
   previous = false;
   for (uint32_t offset = 0; offset < size; offset += 64) {
      uint32_t len = size - offset < 64 ? size - offset : 64;
      bool changed = memcmp(src + offset, base + offset, len) != 0;
      if (changed) {
         if (!previous)
            ranges[n++] = (struct agx_state_range){.offset = offset};
         ranges[n - 1].size += len;
      }
      previous = changed;
   }
   *state = (struct agx_immutable_state){src, size, count, ranges};
   return true;
}

/* Visit disjoint ranges to publish. NULL old means that destination storage
 * has not been initialized (or its contents were invalidated). */
static inline struct agx_state_range
agx_immutable_state_next(const struct agx_immutable_state *old,
                         const struct agx_immutable_state *next,
                         uint32_t *old_index, uint32_t *next_index)
{
   if (!old) {
      if ((*next_index)++ == 0)
         return (struct agx_state_range){0, next->size};
      return (struct agx_state_range){0, 0};
   }
   uint32_t start = UINT32_MAX, end = 0;
   for (;;) {
      const struct agx_state_range *a = *old_index < old->count ? &old->ranges[*old_index] : NULL;
      const struct agx_state_range *b = *next_index < next->count ? &next->ranges[*next_index] : NULL;
      bool take_old = a && (!b || a->offset <= b->offset);
      const struct agx_state_range *range = take_old ? a : b;
      if (!range || (start != UINT32_MAX && range->offset > end))
         break;
      if (start == UINT32_MAX)
         start = range->offset;
      if (range->offset + range->size > end)
         end = range->offset + range->size;
      if (take_old) (*old_index)++;
      else (*next_index)++;
   }
   return (struct agx_state_range){start == UINT32_MAX ? 0 : start, end ? end - start : 0};
}

#endif
