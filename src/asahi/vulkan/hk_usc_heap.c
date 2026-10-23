/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_usc_heap.h"
#include <assert.h>
#include <stdlib.h>
#include "util/macros.h"
#include "util/u_math.h"

void
hk_usc_heap_init(struct hk_usc_heap *table,
                         const struct hk_descriptor_memory_ops *ops, void *data)
{
   *table = (struct hk_usc_heap){.ops = ops, .data = data};
   list_inithead(&table->slabs);
}

static bool
alloc_from_slab(struct hk_descriptor_chunk *slab, uint64_t size, unsigned alignment,
                struct hk_descriptor_allocation *out)
{
   uint64_t addr =
      util_vma_heap_alloc(&slab->heap, size, alignment);
   if (!addr)
      return false;
   *out = (struct hk_descriptor_allocation){
      .addr = addr,
      .size = size,
      .map = slab->backing.map + (addr - slab->backing.addr),
   };
   return true;
}

bool
hk_usc_heap_alloc(struct hk_usc_heap *table, uint64_t size, unsigned alignment,
                          struct hk_descriptor_allocation *out)
{
   if (!size)
      return false;
   assert(util_is_power_of_two_nonzero(alignment));
   list_for_each_entry(struct hk_descriptor_chunk, slab, &table->slabs, link) {
      if (alloc_from_slab(slab, size, alignment, out))
         return true;
   }
   struct hk_descriptor_chunk *slab = calloc(1, sizeof(*slab));
   if (!slab)
      return false;
   if (!table->ops->create(table->data, MAX2(size, HK_USC_HEAP_SLAB_SIZE), 0,
                           &slab->backing)) {
      free(slab);
      return false;
   }
   assert(slab->backing.addr && slab->backing.size >= size);
   util_vma_heap_init(&slab->heap, slab->backing.addr, slab->backing.size);
   list_addtail(&slab->link, &table->slabs);
   ASSERTED bool ok = alloc_from_slab(slab, size, alignment, out);
   assert(ok);
   return true;
}

void
hk_usc_heap_free(struct hk_usc_heap *table,
                         struct hk_descriptor_allocation *alloc)
{
   if (!alloc->size)
      return;
   list_for_each_entry(struct hk_descriptor_chunk, slab, &table->slabs, link) {
      if (alloc->addr >= slab->backing.addr &&
          alloc->addr < slab->backing.addr + slab->backing.size) {
         assert(alloc->addr + alloc->size <=
                slab->backing.addr + slab->backing.size);
         util_vma_heap_free(&slab->heap, alloc->addr, alloc->size);
         *alloc = (struct hk_descriptor_allocation){0};
         return;
      }
   }
   UNREACHABLE("USC allocation must belong to its device's heap");
}

void
hk_usc_heap_finish(struct hk_usc_heap *table)
{
   list_for_each_entry_safe(struct hk_descriptor_chunk, slab, &table->slabs,
                            link) {
      util_vma_heap_finish(&slab->heap);
      table->ops->destroy(table->data, &slab->backing);
      list_del(&slab->link);
      free(slab);
   }
}
