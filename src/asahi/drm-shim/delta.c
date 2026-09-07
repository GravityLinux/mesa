/* SPDX-License-Identifier: MIT */
#include "delta.h"
#include <string.h>
#include <stdbool.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

int asahi_m1n1_equal(const void *a, const void *b, size_t size)
{
   return !memcmp(a, b, size);
}

/* Search without reading beyond the page/buffer. SSE2 is baseline on x86-64;
 * other hosts retain a portable path without changing the packet format. */
static uint64_t find_byte(const uint8_t *a, const uint8_t *b, uint64_t at,
                          uint64_t end, bool different)
{
#if defined(__SSE2__)
   while (end - at >= 16) {
      __m128i x = _mm_loadu_si128((const __m128i *)(a + at));
      __m128i y = _mm_loadu_si128((const __m128i *)(b + at));
      unsigned mask = _mm_movemask_epi8(_mm_cmpeq_epi8(x, y));
      if (different)
         mask ^= 0xffff;
      if (mask)
         return at + __builtin_ctz(mask);
      at += 16;
   }
#endif
   while (at < end && ((a[at] != b[at]) != different))
      ++at;
   return at;
}

static bool safe_gap(const struct asahi_cpu_spans *safe, uint64_t a, uint64_t b)
{
   if (!safe)
      return false;
   for (uint64_t i = 0; i < safe->count; ++i)
      if (safe->spans[i].start <= a && b <= safe->spans[i].end)
         return true;
   return false;
}

static void put64(uint8_t *out, uint64_t value)
{
   for (unsigned i = 0; i < 8; ++i)
      out[i] = value >> (8 * i);
}

/* Bounded resumable encoder. Commit a complete 4 KiB page at a time. Worst
 * case is alternating changed bytes: 2048 records of 24 bytes. No allocation,
 * per-range callback, target access or shadow mutation occurs here. */
int asahi_m1n1_encode_delta(const uint8_t *current, const uint8_t *previous,
   uint64_t size, uint64_t physical, const struct asahi_cpu_spans *safe,
   uint8_t *out, uint64_t capacity, uint64_t *cursor, uint64_t *written,
   uint64_t *payload, uint64_t *records)
{
   if (!current || !previous || !out || !cursor || !written || !payload || !records ||
       *cursor > size || (*cursor < size && (*cursor & 4095)) ||
       !physical || size > UINT64_MAX - physical || capacity < 49152)
      return -1;
   if (safe) {
      if (safe->count > ASAHI_CPU_SPANS)
         return -1;
      for (uint64_t i = 0; i < safe->count; ++i)
         if (safe->spans[i].start > safe->spans[i].end || safe->spans[i].end > size)
            return -1;
   }
   *written = *payload = *records = 0;
   while (*cursor < size && capacity - *written >= 49152) {
      uint64_t page = *cursor;
      uint64_t end = size - page < 4096 ? size : page + 4096;
      uint64_t start = find_byte(current, previous, page, end, true);
      while (start < end) {
         uint64_t stop = find_byte(current, previous, start, end, false);
         uint64_t next = find_byte(current, previous, stop, end, true);
         while (next < end && next - stop <= 16 && safe_gap(safe, stop, next)) {
            stop = find_byte(current, previous, next, end, false);
            next = find_byte(current, previous, stop, end, true);
         }
         uint64_t length = stop - start;
         uint64_t padded = (length + 7) & ~UINT64_C(7);
         put64(out + *written, physical + start);
         put64(out + *written + 8, length);
         memcpy(out + *written + 16, current + start, length);
         memset(out + *written + 16 + length, 0, padded - length);
         *written += 16 + padded;
         *payload += length;
         ++*records;
         start = next;
      }
      *cursor = end;
   }
   return 0;
}
