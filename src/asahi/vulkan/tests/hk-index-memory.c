/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include "../hk_buffer.h"
#include "../hk_device.h"
#include "../hk_device_memory.h"
#include "../hk_entrypoints.h"
#include "c11/threads.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         failures++;                                                           \
      }                                                                        \
   } while (0)
static unsigned failures, allocations, binds, frees;
static bool fail_alloc, fail_bind;
static uint64_t mapped_size, mapped_offset;
static const uint64_t alias_base = 0x10010000000ull;

struct agx_va *__wrap_agx_va_alloc(struct agx_device *dev, uint64_t size,
                                   uint64_t alignment, enum agx_va_flags flags,
                                   uint64_t fixed);
struct agx_va *
__wrap_agx_va_alloc(struct agx_device *dev, uint64_t size, uint64_t alignment,
                    enum agx_va_flags flags, uint64_t fixed)
{
   allocations++;
   CHECK((alignment == AIL_PAGESIZE || alignment == 65536) &&
         flags == AGX_VA_USC && fixed == 0);
   if (fail_alloc)
      return NULL;
   struct agx_va *va = calloc(1, sizeof(*va));
   *va = (struct agx_va){.addr = alias_base,
                         .size_B = size + dev->guard_size,
                         .flags = flags};
   return va;
}

int __wrap_agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t addr,
                       uint64_t size, uint64_t offset, uint32_t flags);
int
__wrap_agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t addr,
                   uint64_t size, uint64_t offset, uint32_t flags)
{
   binds++;
   mapped_size = size;
   mapped_offset = offset;
   CHECK(addr == alias_base);
   CHECK(flags == (DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE));
   return fail_bind ? -1 : 0;
}

void __wrap_agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind);
void
__wrap_agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind)
{
   frees++;
   CHECK(unbind);
   free(va);
}

struct thread_data {
   struct hk_device *dev;
   struct hk_device_memory *mem;
   bool success;
};
static int
query_thread(void *data)
{
   struct thread_data *t = data;
   t->success = true;
   for (unsigned i = 0; i < 100; i++)
      t->success &= hk_memory_index_address(t->dev, t->mem) == alias_base;
   return 0;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.guard_size = AIL_PAGESIZE;
   struct agx_va backing = {.addr = 0x4000000000ull};
   struct agx_bo bo = {.va = &backing, .size = 65536};
   struct hk_device_memory mem = {
      .vk = {.base = {.type = VK_OBJECT_TYPE_DEVICE_MEMORY}, .size = 49152},
      .bo = &bo,
   };
   simple_mtx_init(&mem.index_va_lock, mtx_plain);

   /* A cached BO can be bigger than the Vulkan allocation. Map only memory
    * owned by this allocation, and share that mapping across offsets.
    */
   CHECK(hk_memory_index_address(dev, &mem) == alias_base);
   CHECK(mapped_size == mem.vk.size && allocations == 1 && binds == 1);
   struct hk_buffer buffers[2] = {0};
   for (unsigned i = 0; i < 2; i++) {
      buffers[i].vk.base.type = VK_OBJECT_TYPE_BUFFER;
      buffers[i].apple9_index_va = true;
      VkBindBufferMemoryInfo info = {
         .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
         .buffer = hk_buffer_to_handle(&buffers[i]),
         .memory = hk_device_memory_to_handle(&mem),
         .memoryOffset = i * AIL_PAGESIZE,
      };
      CHECK(hk_BindBufferMemory2(hk_device_to_handle(dev), 1, &info) ==
            VK_SUCCESS);
      CHECK(buffers[i].vk.device_address == alias_base + info.memoryOffset);
      CHECK(buffers[i].va == NULL);
   }
   CHECK(allocations == 1 && binds == 1 && frees == 0);
   puts("shared allocation offsets: one mapping, no buffer-owned VA");
   hk_memory_free_index_va(dev, &mem);
   CHECK(dev->apple9_index_alias_bytes == 0);

   /* Concurrent bindings of different buffers to one allocation are legal. */
   thrd_t threads[4];
   struct thread_data data[4];
   unsigned before = allocations;
   for (unsigned i = 0; i < 4; i++) {
      data[i] = (struct thread_data){dev, &mem, false};
      CHECK(thrd_create(&threads[i], query_thread, &data[i]) == thrd_success);
   }
   for (unsigned i = 0; i < 4; i++) {
      thrd_join(threads[i], NULL);
      CHECK(data[i].success);
   }
   CHECK(allocations == before + 1);
   puts("concurrent bindings: one allocation and stable shared address");
   hk_memory_free_index_va(dev, &mem);
   CHECK(dev->apple9_index_alias_bytes == 0);

   /* The optimization must not turn USC exhaustion into an API failure. */
   unsigned free_before = frees;
   fail_alloc = true;
   CHECK(hk_memory_index_address(dev, &mem) == 0 && mem.index_va == NULL);
   CHECK(dev->apple9_index_alias_bytes == 0);
   fail_alloc = false;
   fail_bind = true;
   CHECK(hk_memory_index_address(dev, &mem) == 0 && mem.index_va == NULL);
   CHECK(frees == free_before + 1);
   CHECK(dev->apple9_index_alias_bytes == 0);
   fail_bind = false;
   CHECK(hk_memory_index_address(dev, &mem) == alias_base);
   hk_memory_free_index_va(dev, &mem);
   CHECK(dev->apple9_index_alias_bytes == 0);
   puts("alias failures: copy fallback and later retry remain available");

   /* The cap is inclusive. A larger allocation must reserve only the
    * buffer's range, including its original buffer-owned lifetime.
    */
   mem.vk.size = HK_INDEX_ALIAS_ALLOCATION_MAX;
   bo.size = mem.vk.size;
   CHECK(hk_memory_index_address(dev, &mem) == alias_base);
   CHECK(dev->apple9_index_alias_bytes == mem.vk.size + dev->dev.guard_size);
   hk_memory_free_index_va(dev, &mem);
   mem.vk.size += AIL_PAGESIZE;
   bo.size = mem.vk.size;
   for (unsigned limit = 0; limit < 2; limit++) {
      if (limit) {
         mem.vk.size = 49152;
         /* Less than size + guard is left in the shared-alias budget. */
         dev->apple9_index_alias_bytes = HK_INDEX_ALIAS_BUDGET - mem.vk.size;
      }
      uint64_t charged_before = dev->apple9_index_alias_bytes;
      before = allocations;
      CHECK(hk_memory_index_address(dev, &mem) == 0);
      CHECK(allocations == before && mem.index_va == NULL);
      struct hk_buffer fallback = {
         .vk = {.base = {.type = VK_OBJECT_TYPE_BUFFER}, .size = 128},
         .apple9_index_va = true,
      };
      VkBindBufferMemoryInfo info = {
         .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
         .buffer = hk_buffer_to_handle(&fallback),
         .memory = hk_device_memory_to_handle(&mem),
         .memoryOffset = AIL_PAGESIZE,
      };
      CHECK(hk_BindBufferMemory2(hk_device_to_handle(dev), 1, &info) ==
            VK_SUCCESS);
      CHECK(fallback.va && fallback.vk.device_address == alias_base);
      CHECK(mapped_offset == info.memoryOffset);
      CHECK(mapped_size == 2 * AIL_PAGESIZE);
      CHECK(mem.index_va == NULL &&
            dev->apple9_index_alias_bytes == charged_before);
      __wrap_agx_va_free(&dev->dev, fallback.va, true);
      dev->apple9_index_alias_bytes = 0;
   }
   puts("allocation cap and guard-inclusive budget: buffer-sized fallback");

   /* Another allocation may fill the budget after this one was mapped.
    * Existing aliases remain reusable, and freeing one returns its charge.
    */
   CHECK(hk_memory_index_address(dev, &mem) == alias_base);
   uint64_t own_charge = dev->apple9_index_alias_bytes;
   dev->apple9_index_alias_bytes = HK_INDEX_ALIAS_BUDGET;
   before = allocations;
   CHECK(hk_memory_index_address(dev, &mem) == alias_base);
   CHECK(allocations == before);
   hk_memory_free_index_va(dev, &mem);
   CHECK(dev->apple9_index_alias_bytes == HK_INDEX_ALIAS_BUDGET - own_charge);
   dev->apple9_index_alias_bytes = 0;
   puts(
      "existing aliases reuse a full budget; destruction releases the charge");

   /* Allocations already in USC need neither another VA nor another bind. */
   backing.flags = AGX_VA_USC;
   before = allocations;
   CHECK(hk_memory_index_address(dev, &mem) == backing.addr);
   CHECK(allocations == before && mem.index_va == NULL);
   simple_mtx_destroy(&mem.index_va_lock);
   free(dev);
   return failures != 0;
}
