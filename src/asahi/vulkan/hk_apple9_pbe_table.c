/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9_pbe_table.h"
#include <assert.h>
#include <stdlib.h>
#include "asahi/lib/agx_apple9_layout.h"
#include "util/macros.h"
#include "util/u_math.h"

void
hk_apple9_pbe_table_init(struct hk_apple9_pbe_table *table,
                         const struct hk_descriptor_memory_ops *ops, void *data)
{
   *table = (struct hk_apple9_pbe_table){.ops = ops, .data = data};
   list_inithead(&table->slabs);
}

static bool
alloc_from_slab(struct hk_descriptor_chunk *slab, uint64_t size,
                struct hk_descriptor_allocation *out)
{
   uint64_t addr =
      util_vma_heap_alloc(&slab->heap, size, AGX_APPLE9_TEXTURE_TABLE_STRIDE);
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
hk_apple9_pbe_table_alloc(struct hk_apple9_pbe_table *table, unsigned count,
                          struct hk_descriptor_allocation *out)
{
   if (!count)
      return false;
   uint64_t size = (uint64_t)count * AGX_APPLE9_TEXTURE_TABLE_STRIDE;
   list_for_each_entry(struct hk_descriptor_chunk, slab, &table->slabs, link) {
      if (alloc_from_slab(slab, size, out))
         return true;
   }
   struct hk_descriptor_chunk *slab = calloc(1, sizeof(*slab));
   if (!slab)
      return false;
   if (!table->ops->create(table->data, MAX2(size, HK_APPLE9_PBE_SLAB_SIZE), 0,
                           &slab->backing)) {
      free(slab);
      return false;
   }
   assert(slab->backing.addr && slab->backing.size >= size);
   util_vma_heap_init(&slab->heap, slab->backing.addr, slab->backing.size);
   list_addtail(&slab->link, &table->slabs);
   ASSERTED bool ok = alloc_from_slab(slab, size, out);
   assert(ok);
   return true;
}

void
hk_apple9_pbe_table_free(struct hk_apple9_pbe_table *table,
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
   UNREACHABLE("PBE allocation must belong to its device's table");
}

void
hk_apple9_pbe_table_finish(struct hk_apple9_pbe_table *table)
{
   list_for_each_entry_safe(struct hk_descriptor_chunk, slab, &table->slabs,
                            link) {
      util_vma_heap_finish(&slab->heap);
      table->ops->destroy(table->data, &slab->backing);
      list_del(&slab->link);
      free(slab);
   }
}
