/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "agx_bo.h"
#include "agx_device.h"

/* Main, USC, context, replay, replay USC: indexes agx_device::va_quarantine. */
static unsigned
agx_va_heap_index(enum agx_va_flags flags)
{
   assert(!(flags & AGX_VA_USC) || !(flags & AGX_VA_CONTEXT));
   if (flags & AGX_VA_CONTEXT)
      return 2;
   if (flags & AGX_VA_REPLAY)
      return (flags & AGX_VA_USC) ? 4 : 3;
   return (flags & AGX_VA_USC) ? 1 : 0;
}

static struct util_vma_heap *
agx_vma_heap(struct agx_device *dev, enum agx_va_flags flags)
{
   struct util_vma_heap *heaps[] = {&dev->main_heap, &dev->usc_heap,
                                    &dev->context_heap, &dev->replay_heap,
                                    &dev->replay_usc_heap};
   return heaps[agx_va_heap_index(flags)];
}

/*
 * A GEM close while the VM has work queued only marks the old mappings
 * closing: the kernel removes them once the work queued at the close has
 * completed, since it cannot tell which of that work used the object. A bind
 * over a closing range waits for all of it, so recycling a just-closed VA
 * (the heaps allocate top-down, so it is the next one handed out) stalls an
 * ordinary allocation for the whole backlog. Retired VAs therefore wait in a
 * FIFO and return to their heap once a per-heap byte cap is exceeded, or when
 * an allocation would otherwise fail.
 */
static void
agx_va_release_locked(struct agx_device *dev, struct agx_va *va)
{
   list_del(&va->link);
   dev->va_quarantine.bytes[agx_va_heap_index(va->flags)] -= va->size_B;
   dev->va_quarantine.heap_used_B[agx_va_heap_index(va->flags)] -= va->size_B;
   util_vma_heap_free(agx_vma_heap(dev, va->flags), va->addr, va->size_B);
   free(va);
}

/* Return quarantined VAs so an allocation can be retried: for a fixed
 * address, those overlapping it (in whichever heap: the ranges are
 * disjoint); otherwise all of the heap's. Returns whether any was released.
 */
static bool
agx_va_unquarantine_locked(struct agx_device *dev, enum agx_va_flags flags,
                           uint64_t fixed_va, uint64_t size_B)
{
   unsigned heap = agx_va_heap_index(flags);
   bool released = false;
   list_for_each_entry_safe(struct agx_va, va, &dev->va_quarantine.list, link) {
      if (flags & AGX_VA_FIXED) {
         if (va->addr >= fixed_va + size_B || fixed_va >= va->addr + va->size_B)
            continue;
      } else if (agx_va_heap_index(va->flags) != heap) {
         continue;
      }
      agx_va_release_locked(dev, va);
      if (flags & AGX_VA_FIXED)
         dev->va_quarantine.released_fixed++;
      else
         dev->va_quarantine.released_exhaustion++;
      released = true;
   }
   return released;
}

void
agx_va_quarantine_init(struct agx_device *dev, uint64_t main_B,
                       uint64_t usc_B, uint64_t context_B)
{
   const uint64_t max_B = 8ull << 30;
   memset(&dev->va_quarantine, 0, sizeof(dev->va_quarantine));
   list_inithead(&dev->va_quarantine.list);
   dev->va_quarantine.cap[0] = MIN2(main_B / 8, max_B);
   dev->va_quarantine.cap[1] = MIN2(usc_B / 8, max_B);
   dev->va_quarantine.cap[2] = MIN2(context_B / 8, max_B);
   dev->va_quarantine.context_size_B = context_B;
   /* Replay heaps keep cap 0: replay needs its exact addresses back. */
}

void
agx_va_quarantine_finish(struct agx_device *dev)
{
   if (getenv("AGX_VA_QUARANTINE_STATS")) {
      fprintf(stderr,
              "agx: va quarantine: retired %" PRIu64 ", high water %" PRIu64
              " B, released by cap %" PRIu64 ", exhaustion %" PRIu64
              ", fixed %" PRIu64 "\n",
              dev->va_quarantine.retired, dev->va_quarantine.high_water_B,
              dev->va_quarantine.released_cap,
              dev->va_quarantine.released_exhaustion,
              dev->va_quarantine.released_fixed);
      if (dev->va_quarantine.context_size_B) {
         fprintf(stderr,
                 "agx: context heap: high water %" PRIu64 " B of %" PRIu64
                 " B\n",
                 dev->va_quarantine.heap_high_water_B[2],
                 dev->va_quarantine.context_size_B);
      }
   }

   simple_mtx_lock(&dev->vma_lock);
   list_for_each_entry_safe(struct agx_va, va, &dev->va_quarantine.list, link)
      agx_va_release_locked(dev, va);
   simple_mtx_unlock(&dev->vma_lock);
}

/* Free the VA of a GEM object that has just been closed (no unbind: the close
 * removes the mappings, possibly later). See the comment above.
 */
void
agx_va_retire(struct agx_device *dev, struct agx_va *va)
{
   if (!va)
      return;

   unsigned heap = agx_va_heap_index(va->flags);
   simple_mtx_lock(&dev->vma_lock);
   list_addtail(&va->link, &dev->va_quarantine.list);
   dev->va_quarantine.bytes[heap] += va->size_B;
   dev->va_quarantine.retired++;
   uint64_t total = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(dev->va_quarantine.bytes); ++i)
      total += dev->va_quarantine.bytes[i];
   dev->va_quarantine.high_water_B = MAX2(dev->va_quarantine.high_water_B, total);
   list_for_each_entry_safe(struct agx_va, old, &dev->va_quarantine.list, link) {
      if (dev->va_quarantine.bytes[heap] <= dev->va_quarantine.cap[heap])
         break;
      if (agx_va_heap_index(old->flags) == heap) {
         agx_va_release_locked(dev, old);
         /* A zero cap (replay heaps) is no quarantine: not a cap release. */
         if (dev->va_quarantine.cap[heap])
            dev->va_quarantine.released_cap++;
      }
   }
   simple_mtx_unlock(&dev->vma_lock);
}

struct agx_va *
agx_va_alloc(struct agx_device *dev, uint64_t size_B, uint64_t align_B,
             enum agx_va_flags flags, uint64_t fixed_va)
{
   assert((fixed_va != 0) == !!(flags & AGX_VA_FIXED));
   assert((fixed_va % align_B) == 0);

   /* All allocations need a guard at the end to prevent overreads.
    *
    * TODO: Even with soft fault?
    */
   size_B += dev->guard_size;

   struct util_vma_heap *heap = agx_vma_heap(dev, flags);
   uint64_t addr = 0;

   simple_mtx_lock(&dev->vma_lock);
   const enum agx_va_flags requested = flags;
   for (unsigned attempt = 0; attempt < 2 && !addr; ++attempt) {
      /* Out of space, or a fixed address still quarantined: return the
       * quarantine (only what overlaps a fixed address) and retry once.
       * Never in a forked child: it must not touch the VA heaps.
       */
      if (attempt &&
          (dev->bo_cache.forked ||
           !agx_va_unquarantine_locked(dev, requested, fixed_va, size_B)))
         break;

      flags = requested;
      heap = agx_vma_heap(dev, flags);
      if (flags & AGX_VA_FIXED) {
         if (util_vma_heap_alloc_addr(heap, fixed_va, size_B)) {
            addr = fixed_va;
         } else if (flags & AGX_VA_REPLAY) {
            /* Captures made before the replay ranges existed hold addresses
             * in the ordinary heaps. Same address, other heap; never another.
             */
            flags &= ~AGX_VA_REPLAY;
            heap = agx_vma_heap(dev, flags);
            if (util_vma_heap_alloc_addr(heap, fixed_va, size_B))
               addr = fixed_va;
         }
      } else {
         addr = util_vma_heap_alloc(heap, size_B, align_B);
      }
   }
   if (addr) {
      unsigned index = agx_va_heap_index(flags);
      dev->va_quarantine.heap_used_B[index] += size_B;
      dev->va_quarantine.heap_high_water_B[index] =
         MAX2(dev->va_quarantine.heap_high_water_B[index],
              dev->va_quarantine.heap_used_B[index]);
   }
   simple_mtx_unlock(&dev->vma_lock);

   if (addr == 0)
      return NULL;

   struct agx_va *va = malloc(sizeof(struct agx_va));
   *va = (struct agx_va){
      .flags = flags,
      .size_B = size_B,
      .addr = addr,
   };
   return va;
}

void
agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind)
{
   if (!va)
      return;

   if (unbind) {
      agx_bo_bind(dev, NULL, va->addr, va->size_B, 0, DRM_ASAHI_BIND_UNBIND);
   }

   struct util_vma_heap *heap = agx_vma_heap(dev, va->flags);

   simple_mtx_lock(&dev->vma_lock);
   dev->va_quarantine.heap_used_B[agx_va_heap_index(va->flags)] -= va->size_B;
   util_vma_heap_free(heap, va->addr, va->size_B);
   simple_mtx_unlock(&dev->vma_lock);
   free(va);
}
