/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * hk_apple9_native_stream_admits: the split policy for batched native
 * renders. Every admitted stream must stay inside the root chunk without a
 * link and inside the paired decoder's 1024 words. Block sizes are those of
 * the recorded vectors (meshbatch vector record): a native block is
 * 156 bytes and a direct ordinary block 120, each including the terminator
 * that the next block overwrites. */
#include <stdio.h>
#include <stdlib.h>
#include "hk_apple9_native_batch.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

/* hk_get_cs root chunk and hk_ensure_cs_has_space reservation (padding plus
 * a link tag of at most 16 bytes). */
#define ROOT 4096
#define CAPACITY (ROOT - 0x800 - 16)

static const struct hk_apple9_native_profile mixed = {4, true, true};
static const struct hk_apple9_native_profile blocks = {4, true, false};
static const struct hk_apple9_native_profile legacy = {4, false, false};

static struct hk_apple9_draw_summary
native_mesh(void)
{
   return (struct hk_apple9_draw_summary){.native = true, .mesh = true};
}

static struct hk_apple9_draw_summary
vertex(void)
{
   return (struct hk_apple9_draw_summary){0};
}

int
main(void)
{
   CHECK(HK_APPLE9_ORDINARY_BLOCK_MAX_B <= HK_APPLE9_BATCH_BLOCK_MAX_B);
   CHECK(HK_APPLE9_BATCH_RESERVE_B >= HK_APPLE9_BATCH_ENSURE_B);
   struct hk_apple9_stream_summary s = {true, 1, 152, CAPACITY};
   struct hk_apple9_draw_summary d = native_mesh();

   /* The research profiles. */
   CHECK(hk_apple9_native_stream_admits(&s, &d, &mixed));
   CHECK(hk_apple9_native_stream_admits(&s, &d, &blocks));
   CHECK(!hk_apple9_native_stream_admits(&s, &d, &legacy));
   d = vertex();
   CHECK(hk_apple9_native_stream_admits(&s, &d, &mixed));
   CHECK(!hk_apple9_native_stream_admits(&s, &d, &blocks));
   CHECK(!hk_apple9_native_stream_admits(&s, &d, &legacy));

   /* Never append to an ordinary or empty stream, or with batch 1. */
   struct hk_apple9_stream_summary ordinary = {false, 0, 116, CAPACITY};
   CHECK(!hk_apple9_native_stream_admits(&ordinary, &d, &mixed));
   const struct hk_apple9_native_profile single = {1, true, true};
   d = native_mesh();
   CHECK(!hk_apple9_native_stream_admits(&s, &d, &single));

   /* Refusals that split. */
   const struct hk_apple9_draw_summary refused[] = {
      {.native = true, .mesh = true, .meta = true},
      {.native = true, .mesh = true, .queries = true},
      {.native = true, .mesh = true, .task = true},
      {.native = true, .mesh = true, .xfb = true},
      {.native = true, .mesh = true, .writes_memory = true},
      {.gs = true}, {.tess = true}, {.mesh = true}, {.xfb = true},
   };
   for (unsigned i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i)
      CHECK(!hk_apple9_native_stream_admits(&s, &refused[i], &mixed));

   /* The dispatch cap counts native blocks only. */
   s.dispatches = 4;
   d = native_mesh();
   CHECK(!hk_apple9_native_stream_admits(&s, &d, &mixed));
   d = vertex();
   CHECK(hk_apple9_native_stream_admits(&s, &d, &mixed));

   /* Budget edge: the reservation covers the ensure and the largest block. */
   s.dispatches = 1;
   s.bytes = CAPACITY - HK_APPLE9_BATCH_RESERVE_B;
   CHECK(hk_apple9_native_stream_admits(&s, &d, &mixed));
   s.bytes++;
   CHECK(!hk_apple9_native_stream_admits(&s, &d, &mixed));
   s.capacity = 1u << 20; /* decoder budget bounds a larger root too */
   s.bytes = HK_APPLE9_NATIVE_DECODE_BYTES - HK_APPLE9_BATCH_RESERVE_B + 1;
   CHECK(!hk_apple9_native_stream_admits(&s, &d, &mixed));

   /* Legacy repeated input: same GS/TES programs only. */
   s = (struct hk_apple9_stream_summary){true, 1, 152, CAPACITY};
   d = (struct hk_apple9_draw_summary){.native = true, .gs = true, .same_shaders = true};
   CHECK(hk_apple9_native_stream_admits(&s, &d, &legacy));
   d.same_shaders = false;
   CHECK(!hk_apple9_native_stream_admits(&s, &d, &legacy));

   /* Fill a mixed stream with worst-case blocks (native 160, ordinary
    * indexed 140) and check every admitted append stays in the root chunk
    * without a link and within the decoder budget. */
   for (unsigned pattern = 0; pattern < 4; ++pattern) {
      const struct hk_apple9_native_profile all = {16, true, true};
      s = (struct hk_apple9_stream_summary){true, 1, 160 - 4, CAPACITY};
      unsigned appended = 0;
      for (unsigned i = 0;; ++i) {
         bool native = pattern == 0 || (pattern == 1 && i % 2) ||
                       (pattern == 2 && i % 3 == 0);
         d = native ? native_mesh() : vertex();
         if (!hk_apple9_native_stream_admits(&s, &d, &all))
            break;
         CHECK(s.bytes + HK_APPLE9_BATCH_ENSURE_B <= CAPACITY);
         s.bytes += (native ? 160 : HK_APPLE9_ORDINARY_BLOCK_MAX_B) - 4;
         s.dispatches += native;
         CHECK(s.bytes + 4 <= ROOT - 0x800 && s.bytes + 4 <= HK_APPLE9_NATIVE_DECODE_BYTES);
         ++appended;
      }
      CHECK(appended >= 4);
      printf("pattern %u: %u blocks after the leading native block, %zu bytes\n",
             pattern, appended, s.bytes + 4);
   }
   /* Batched native graphics streams get an 8 KiB root (hk_get_cs): the
    * decoder budget, not a link, then bounds them, and 16 native draws fit
    * with ordinary draws between them. */
   {
      const struct hk_apple9_native_profile all = {16, true, true};
      s = (struct hk_apple9_stream_summary){true, 1, 160 - 4, 8192 - 0x800 - 16};
      unsigned natives = 1, ordinary = 0;
      for (unsigned i = 0; natives < 16; ++i) {
         d = i % 4 == 3 ? vertex() : native_mesh();
         CHECK(hk_apple9_native_stream_admits(&s, &d, &all));
         s.bytes += (d.native ? 160 : HK_APPLE9_ORDINARY_BLOCK_MAX_B) - 4;
         s.dispatches += d.native;
         natives += d.native;
         ordinary += !d.native;
      }
      CHECK(s.bytes + 4 <= HK_APPLE9_NATIVE_DECODE_BYTES);
      d = native_mesh();
      CHECK(!hk_apple9_native_stream_admits(&s, &d, &all));
      printf("8 KiB root: 16 native + %u ordinary blocks, %zu bytes\n", ordinary, s.bytes + 4);
   }
   /* The 4096-word research kernel (native_batch up to 64) with the 16 KiB
    * root: 64 native draws and 16 ordinary draws fit in one render. */
   {
      const struct hk_apple9_native_profile big = {64, true, true, 4096};
      s = (struct hk_apple9_stream_summary){true, 1, 160 - 4, 16384 - 0x800 - 16};
      unsigned natives = 1, ordinary = 0;
      for (unsigned i = 0; natives < 64; ++i) {
         d = i % 5 == 4 ? vertex() : native_mesh();
         CHECK(hk_apple9_native_stream_admits(&s, &d, &big));
         s.bytes += (d.native ? 160 : HK_APPLE9_ORDINARY_BLOCK_MAX_B) - 4;
         s.dispatches += d.native;
         natives += d.native;
         ordinary += !d.native;
      }
      CHECK(s.bytes + 4 <= 4 * 4096);
      d = native_mesh();
      CHECK(!hk_apple9_native_stream_admits(&s, &d, &big));
      /* A 1024-word profile refuses the same stream long before. */
      const struct hk_apple9_native_profile small = {64, true, true, 1024};
      CHECK(!hk_apple9_native_stream_admits(&s, &d, &small));
      printf("16 KiB root, 4096 words: 64 native + %u ordinary blocks, %zu bytes\n",
             ordinary, s.bytes + 4);
   }
   /* The production mapping from the latched kernel profile. */
   {
      const struct agx_native_blocks_support q = {true, 7, 64, true, 4096};
      struct hk_apple9_native_profile p = hk_apple9_native_profile_from_kernel(&q, true, false);
      CHECK(p.batch == 64 && p.blocks && p.mixed && p.words == 4096);
      struct agx_native_blocks_support v = q;
      v.mixed = false; v.words = 1024; v.max_native = 16;
      p = hk_apple9_native_profile_from_kernel(&v, true, false);
      CHECK(p.batch == 16 && p.blocks && !p.mixed && p.words == 1024);
      /* Not qualified, profile none, the single-block kernel, the kill
       * switches, or no qualified native path: one native block per render. */
      const struct agx_native_blocks_support none = {0};
      v = q; v.max_native = 1;
      const struct hk_apple9_native_profile single[] = {
         hk_apple9_native_profile_from_kernel(&none, true, false),
         hk_apple9_native_profile_from_kernel(&v, true, false),
         hk_apple9_native_profile_from_kernel(&q, false, false),
         hk_apple9_native_profile_from_kernel(&q, true, true),
      };
      for (unsigned i = 0; i < 4; ++i) {
         CHECK(single[i].batch == 1 && !single[i].blocks);
         s = (struct hk_apple9_stream_summary){true, 1, 152, CAPACITY};
         d = native_mesh();
         CHECK(!hk_apple9_native_stream_admits(&s, &d, &single[i]));
      }
      /* mixed=0 keeps ordinary draws out of a native render. */
      v = q; v.mixed = false;
      p = hk_apple9_native_profile_from_kernel(&v, true, false);
      s = (struct hk_apple9_stream_summary){true, 1, 152, CAPACITY};
      d = vertex();
      CHECK(!hk_apple9_native_stream_admits(&s, &d, &p));
      d = native_mesh();
      CHECK(hk_apple9_native_stream_admits(&s, &d, &p));
   }
   printf("PASS\n");
   return 0;
}
