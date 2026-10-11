/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "hk_descriptor_memory.h"

#include <assert.h>
#include <stdlib.h>
#include "util/macros.h"

static struct hk_descriptor_chunk *
hk_descriptor_chunk_create(struct hk_descriptor_memory *mem, uint64_t size)
{
   struct hk_descriptor_chunk *chunk = calloc(1, sizeof(*chunk));
   if (!chunk)
      return NULL;

   if (!mem->ops->create(mem->data, size, &chunk->backing)) {
      free(chunk);
      return NULL;
   }

   /* The buffer may be larger thanks to page alignment; use all of it. */
   assert(chunk->backing.size >= size && chunk->backing.addr);
   util_vma_heap_init(&chunk->heap, chunk->backing.addr, chunk->backing.size);
   list_addtail(&chunk->link, &mem->chunks);
   return chunk;
}

static bool
hk_descriptor_chunk_alloc(struct hk_descriptor_chunk *chunk, uint64_t size,
                          uint64_t alignment,
                          struct hk_descriptor_allocation *out)
{
   uint64_t addr = util_vma_heap_alloc(&chunk->heap, size, alignment);
   if (!addr)
      return false;

   assert(addr + size <= chunk->backing.addr + chunk->backing.size);
   *out = (struct hk_descriptor_allocation){
      .addr = addr,
      .map = chunk->backing.map + (addr - chunk->backing.addr),
      .size = size,
   };
   return true;
}

enum hk_descriptor_memory_result
hk_descriptor_memory_init(struct hk_descriptor_memory *mem,
                          const struct hk_descriptor_memory_ops *ops,
                          void *data, uint64_t worst_case)
{
   *mem = (struct hk_descriptor_memory){
      .ops = ops,
      .data = data,
      .lazy = worst_case > HK_DESCRIPTOR_MEMORY_EAGER_MAX,
      .budget = worst_case,
   };
   list_inithead(&mem->chunks);

   if (mem->lazy || !worst_case)
      return HK_DESCRIPTOR_MEMORY_OK;

   return hk_descriptor_chunk_create(mem, worst_case)
             ? HK_DESCRIPTOR_MEMORY_OK
             : HK_DESCRIPTOR_MEMORY_OUT_OF_DEVICE;
}

enum hk_descriptor_memory_result
hk_descriptor_memory_alloc(struct hk_descriptor_memory *mem, uint64_t size,
                           uint64_t alignment,
                           struct hk_descriptor_allocation *out)
{
   assert(size > 0);

   if (!mem->lazy) {
      list_for_each_entry(struct hk_descriptor_chunk, chunk, &mem->chunks,
                          link) {
         if (hk_descriptor_chunk_alloc(chunk, size, alignment, out))
            return HK_DESCRIPTOR_MEMORY_OK;
      }
      return HK_DESCRIPTOR_MEMORY_OUT_OF_POOL;
   }

   if (size > mem->budget - mem->used)
      return HK_DESCRIPTOR_MEMORY_OUT_OF_POOL;

   if (size >= HK_DESCRIPTOR_MEMORY_DEDICATED_MIN) {
      struct hk_descriptor_backing backing;
      if (!mem->ops->create(mem->data, size, &backing))
         return HK_DESCRIPTOR_MEMORY_OUT_OF_DEVICE;

      /* Buffers are page aligned, beyond any descriptor set alignment. */
      assert(backing.size >= size && !(backing.addr & (alignment - 1)));
      *out = (struct hk_descriptor_allocation){
         .addr = backing.addr,
         .map = backing.map,
         .size = size,
         .dedicated = backing,
      };
      mem->used += size;
      return HK_DESCRIPTOR_MEMORY_OK;
   }

   list_for_each_entry(struct hk_descriptor_chunk, chunk, &mem->chunks, link) {
      if (hk_descriptor_chunk_alloc(chunk, size, alignment, out)) {
         mem->used += size;
         return HK_DESCRIPTOR_MEMORY_OK;
      }
   }

   struct hk_descriptor_chunk *chunk =
      hk_descriptor_chunk_create(mem, HK_DESCRIPTOR_MEMORY_CHUNK);
   if (!chunk)
      return HK_DESCRIPTOR_MEMORY_OUT_OF_DEVICE;

   ASSERTED bool ok = hk_descriptor_chunk_alloc(chunk, size, alignment, out);
   assert(ok);
   mem->used += size;
   return HK_DESCRIPTOR_MEMORY_OK;
}

void
hk_descriptor_memory_free(struct hk_descriptor_memory *mem,
                          struct hk_descriptor_allocation *alloc)
{
   assert(alloc->size > 0);

   if (mem->lazy) {
      assert(mem->used >= alloc->size);
      mem->used -= alloc->size;
   }

   if (alloc->dedicated.bo) {
      mem->ops->destroy(mem->data, &alloc->dedicated);
   } else {
      bool found = false;
      list_for_each_entry(struct hk_descriptor_chunk, chunk, &mem->chunks,
                          link) {
         if (alloc->addr >= chunk->backing.addr &&
             alloc->addr < chunk->backing.addr + chunk->backing.size) {
            assert(alloc->addr + alloc->size <=
                   chunk->backing.addr + chunk->backing.size);
            util_vma_heap_free(&chunk->heap, alloc->addr, alloc->size);
            found = true;
            break;
         }
      }
      assert(found);
      (void)found;
   }

   *alloc = (struct hk_descriptor_allocation){0};
}

void
hk_descriptor_memory_finish(struct hk_descriptor_memory *mem)
{
   list_for_each_entry_safe(struct hk_descriptor_chunk, chunk, &mem->chunks,
                            link) {
      util_vma_heap_finish(&chunk->heap);
      mem->ops->destroy(mem->data, &chunk->backing);
      list_del(&chunk->link);
      free(chunk);
   }
}
