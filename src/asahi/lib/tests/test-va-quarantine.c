/*
 * SPDX-License-Identifier: MIT
 *
 * The VA quarantine for closed GEM objects: no immediate reuse, the per-heap
 * byte cap, fixed-address conflicts, allocation-failure flushes, forked
 * children and teardown. No GPU or DRM is needed. Checks stay enabled in
 * release builds.
 */
#include "agx_device.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

#define MAIN_BASE 0x1000000000ull
#define USC_BASE  0x2000000000ull
#define PAGE      16384ull

static struct agx_device dev;

/* A main heap of `pages` pages (the guard page is part of every VA). */
static void
init(uint64_t pages)
{
   memset(&dev, 0, sizeof(dev));
   simple_mtx_init(&dev.vma_lock, mtx_plain);
   dev.guard_size = PAGE;
   util_vma_heap_init(&dev.main_heap, MAIN_BASE, pages * PAGE);
   util_vma_heap_init(&dev.usc_heap, USC_BASE, 1ull << 28);
   agx_va_quarantine_init(&dev, pages * PAGE, 1ull << 28, 0);
}

static void
finish(void)
{
   agx_va_quarantine_finish(&dev);
   CHECK(list_is_empty(&dev.va_quarantine.list));
   for (unsigned i = 0; i < ARRAY_SIZE(dev.va_quarantine.bytes); ++i)
      CHECK(dev.va_quarantine.bytes[i] == 0);
   util_vma_heap_finish(&dev.main_heap);
   util_vma_heap_finish(&dev.usc_heap);
   simple_mtx_destroy(&dev.vma_lock);
}

static struct agx_va *
alloc(uint64_t size, enum agx_va_flags flags, uint64_t fixed)
{
   return agx_va_alloc(&dev, size, PAGE, flags, fixed);
}

/* A retired VA is not handed out again while room remains elsewhere. */
static void
test_no_immediate_reuse(void)
{
   init(1024);
   struct agx_va *a = alloc(PAGE, 0, 0);
   uint64_t addr = a->addr;
   agx_va_retire(&dev, a);
   CHECK(dev.va_quarantine.bytes[0] == 2 * PAGE);
   struct agx_va *b = alloc(PAGE, 0, 0);
   CHECK(b->addr != addr);
   agx_va_free(&dev, b, false);
   finish();
}

/* Past the cap (heap / 8), the oldest retired VAs return to the heap. */
static void
test_cap_releases_oldest(void)
{
   init(1024); /* cap: 128 pages = 64 two-page VAs */
   struct agx_va *vas[80];
   uint64_t first = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(vas); ++i)
      vas[i] = alloc(PAGE, 0, 0);
   first = vas[0]->addr;
   for (unsigned i = 0; i < ARRAY_SIZE(vas); ++i)
      agx_va_retire(&dev, vas[i]);
   CHECK(dev.va_quarantine.bytes[0] <= dev.va_quarantine.cap[0]);
   CHECK(dev.va_quarantine.bytes[0] == 64 * 2 * PAGE);
   /* The oldest is free again: a fixed allocation there needs no flush. */
   uint64_t before = dev.va_quarantine.bytes[0];
   struct agx_va *f = alloc(PAGE, AGX_VA_FIXED, first);
   CHECK(f && f->addr == first && dev.va_quarantine.bytes[0] == before);
   agx_va_free(&dev, f, false);
   finish();
}

/* A fixed address still quarantined is released, and only it. */
static void
test_fixed_conflict(void)
{
   init(1024);
   struct agx_va *a = alloc(PAGE, 0, 0), *b = alloc(PAGE, 0, 0);
   uint64_t addr = a->addr, other = b->addr;
   agx_va_retire(&dev, a);
   agx_va_retire(&dev, b);
   struct agx_va *f = alloc(PAGE, AGX_VA_FIXED, addr);
   CHECK(f && f->addr == addr);
   CHECK(dev.va_quarantine.bytes[0] == 2 * PAGE); /* b is still quarantined */
   struct agx_va *g = alloc(PAGE, AGX_VA_FIXED, other);
   CHECK(g && g->addr == other && dev.va_quarantine.bytes[0] == 0);
   agx_va_free(&dev, f, false);
   agx_va_free(&dev, g, false);
   finish();
}

/* A fixed address that is live (not quarantined) still fails. */
static void
test_fixed_live_still_fails(void)
{
   init(1024);
   struct agx_va *a = alloc(PAGE, 0, 0), *b = alloc(PAGE, 0, 0);
   agx_va_retire(&dev, b);
   CHECK(!alloc(PAGE, AGX_VA_FIXED, a->addr));
   CHECK(dev.va_quarantine.bytes[0] == 2 * PAGE); /* nothing released */
   agx_va_free(&dev, a, false);
   finish();
}

/* When the heap is exhausted, the quarantine is flushed and the allocation
 * retried once.
 */
static void
test_exhaustion_flush(void)
{
   init(64); /* 32 two-page VAs; cap 8 pages */
   struct agx_va *vas[32];
   for (unsigned i = 0; i < 32; ++i)
      CHECK((vas[i] = alloc(PAGE, 0, 0)));
   CHECK(!alloc(PAGE, 0, 0));
   /* Retire 4 (within the cap): all stay quarantined, none free. */
   for (unsigned i = 0; i < 4; ++i)
      agx_va_retire(&dev, vas[i]);
   CHECK(dev.va_quarantine.bytes[0] == 4 * 2 * PAGE);
   struct agx_va *x = alloc(PAGE, 0, 0);
   CHECK(x && dev.va_quarantine.bytes[0] == 0);
   agx_va_free(&dev, x, false);
   for (unsigned i = 4; i < 32; ++i)
      agx_va_free(&dev, vas[i], false);
   finish();
}

/* Other heaps' quarantined VAs are not flushed for a main-heap failure. */
static void
test_flush_is_per_heap(void)
{
   init(64);
   struct agx_va *u = alloc(PAGE, AGX_VA_USC, 0);
   agx_va_retire(&dev, u);
   struct agx_va *vas[32];
   for (unsigned i = 0; i < 32; ++i)
      CHECK((vas[i] = alloc(PAGE, 0, 0)));
   CHECK(!alloc(PAGE, 0, 0));
   CHECK(dev.va_quarantine.bytes[1] == 2 * PAGE);
   for (unsigned i = 0; i < 32; ++i)
      agx_va_free(&dev, vas[i], false);
   finish();
}

/* A forked child never touches the heaps: no flush, the allocation fails. */
static void
test_forked_child_no_flush(void)
{
   init(64);
   struct agx_va *vas[32];
   for (unsigned i = 0; i < 32; ++i)
      CHECK((vas[i] = alloc(PAGE, 0, 0)));
   agx_va_retire(&dev, vas[0]);
   uint64_t addr = vas[0]->addr;
   dev.bo_cache.forked = true;
   CHECK(!alloc(PAGE, 0, 0));
   CHECK(!alloc(PAGE, AGX_VA_FIXED, addr));
   CHECK(dev.va_quarantine.bytes[0] == 2 * PAGE);
   dev.bo_cache.forked = false;
   for (unsigned i = 1; i < 32; ++i)
      agx_va_free(&dev, vas[i], false);
   finish();
}

/* Teardown returns everything, so the heaps finish fully free. */
static void
test_finish_releases_all(void)
{
   init(1024);
   for (unsigned i = 0; i < 10; ++i)
      agx_va_retire(&dev, alloc(PAGE, 0, 0));
   CHECK(dev.va_quarantine.bytes[0] == 10 * 2 * PAGE);
   agx_va_quarantine_finish(&dev);
   /* The whole heap is one hole again: a heap-sized fixed allocation fits. */
   struct agx_va *all = alloc(1024 * PAGE - PAGE, AGX_VA_FIXED, MAIN_BASE);
   CHECK(all);
   agx_va_free(&dev, all, false);
   finish();
}

int
main(void)
{
   test_no_immediate_reuse();
   test_cap_releases_oldest();
   test_fixed_conflict();
   test_fixed_live_still_fails();
   test_exhaustion_flush();
   test_flush_is_per_heap();
   test_forked_child_no_flush();
   test_finish_releases_all();
   printf("va-quarantine: 8 tests passed\n");
   return 0;
}
