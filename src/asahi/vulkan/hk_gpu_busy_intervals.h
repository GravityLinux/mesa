/* SPDX-License-Identifier: MIT */
#ifndef HK_GPU_BUSY_INTERVALS_H
#define HK_GPU_BUSY_INTERVALS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Storage grows only when needed. Leave bounded headroom for hundreds of
 * command pairs per frame and multiple frames in one GPU-clock second. */
#define HK_BUSY_MAX_RANGES 32768

struct hk_busy_interval {
   uint64_t start, end;
};

struct hk_busy_ranges {
   struct hk_busy_interval *data;
   unsigned count, capacity;
   bool overflow;
};

/* Exact union of half-open intervals, including out-of-order completions.
 * Adjacent ranges merge, so dense workloads need very little storage.
 */
static inline bool
hk_busy_union_add(struct hk_busy_ranges *r, uint64_t start, uint64_t end)
{
   if (start >= end)
      return true;
   unsigned first = 0;
   while (first < r->count && r->data[first].end < start)
      ++first;
   unsigned last = first;
   while (last < r->count && r->data[last].start <= end) {
      if (r->data[last].start < start)
         start = r->data[last].start;
      if (r->data[last].end > end)
         end = r->data[last].end;
      ++last;
   }
   if (last == first && r->count == r->capacity) {
      unsigned capacity = r->capacity ? r->capacity * 2 : 16;
      if (capacity > HK_BUSY_MAX_RANGES) {
         r->overflow = true;
         return false;
      }
      void *data = realloc(r->data, capacity * sizeof(*r->data));
      if (!data) {
         r->overflow = true;
         return false;
      }
      r->data = (struct hk_busy_interval *)data;
      r->capacity = capacity;
   }
   unsigned removed = last - first;
   memmove(r->data + first + 1, r->data + last,
           (r->count - last) * sizeof(*r->data));
   r->data[first] = (struct hk_busy_interval){start, end};
   r->count = r->count + 1 - removed;
   return true;
}

static inline uint64_t
hk_busy_union_size(const struct hk_busy_ranges *r)
{
   uint64_t size = 0;
   for (unsigned i = 0; i < r->count; ++i)
      size += r->data[i].end - r->data[i].start;
   return size;
}

static inline void
hk_busy_union_clear(struct hk_busy_ranges *r)
{
   r->count = 0;
   r->overflow = false;
}
#endif
