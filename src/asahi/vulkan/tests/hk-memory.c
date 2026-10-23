/* SPDX-License-Identifier: MIT
 * Exercise real upload/reset/pool code with host-backed BOs. */
#include "../hk_cmd_buffer.h"
#include "../hk_entrypoints.h"
#include "vk_common_entrypoints.h"
#include "vk_alloc.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
static unsigned live_bos, allocations, trims;
struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned, enum agx_bo_flags, const char *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
void __wrap_agx_bo_cache_evict_all(struct agx_device *);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned alignment,
                     enum agx_bo_flags flags, const char *label)
{
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, size);
   CHECK(bo->va && bo->_map);
   bo->dev = dev; bo->size = size; bo->refcnt = 1;
   bo->va->addr = (++allocations) * (uint64_t)HK_CMD_BO_SIZE;
   ++live_bos;
   return bo;
}
void __wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo) return;
   CHECK(bo->refcnt == 1);
   free(bo->_map); free(bo->va); free(bo); --live_bos;
}
void __wrap_agx_bo_cache_evict_all(struct agx_device *dev) { ++trims; }
int main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->vk.dispatch_table.AllocateCommandBuffers = vk_common_AllocateCommandBuffers;
   dev->vk.dispatch_table.ResetCommandBuffer = hk_ResetCommandBuffer;
   dev->dev.chip = AGX_CHIP_G16G;
   VkDevice device = hk_device_to_handle(dev);
   const VkCommandPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool handle;
   CHECK(hk_CreateCommandPool(device, &ci, NULL, &handle) == VK_SUCCESS);
   struct hk_cmd_pool *pool = hk_cmd_pool_from_handle(handle);
   VkCommandBuffer cb;
   const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = handle, .commandBufferCount = 1,
   };
   CHECK(vk_common_AllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS);
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(cb);
   uint64_t last = 0;
   uint32_t words[128] = {0};
   for (unsigned i = 0; i < 6000; ++i) {
      words[0] = i;
      last = hk_pool_upload(cmd, words, sizeof(words), 64);
      CHECK(last);
   }
   /* All data, including uncached uploads, stays alive until reset. */
   unsigned before = live_bos;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE;
   hk_TrimCommandPool(device, handle, 0);
   CHECK(live_bos == before);
   CHECK(util_dynarray_resize_bytes(&cmd->sync_events, 1, 1024 * 1024));
   cmd->sync_events.size = 0;
   CHECK(hk_ResetCommandPool(device, handle, 0) == VK_SUCCESS);
   CHECK(cmd->sync_events.capacity >= 1024 * 1024);
   hk_TrimCommandPool(device, handle, 0);
   CHECK(!cmd->sync_events.capacity && !live_bos && trims == 2);
   /* A releasing reset frees CPU high-water marks on allocated buffers. */
   CHECK(util_dynarray_resize_bytes(&cmd->sync_events, 1, 1024 * 1024));
   cmd->sync_events.size = 0;
   CHECK(hk_ResetCommandBuffer(cb, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT) == VK_SUCCESS);
   CHECK(!cmd->sync_events.capacity);
   /* Free recycles the buffer. Pool RELEASE also drops these recycled objects. */
   CHECK(util_dynarray_resize_bytes(&cmd->sync_events, 1, 1024 * 1024));
   cmd->sync_events.size = 0;
   vk_common_FreeCommandBuffers(device, handle, 1, &cb);
   CHECK(!list_is_empty(&pool->vk.free_command_buffers[0]));
   CHECK(hk_ResetCommandPool(device, handle, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) == VK_SUCCESS);
   CHECK(list_is_empty(&pool->vk.free_command_buffers[0]));
   hk_DestroyCommandPool(device, handle, NULL);
   CHECK(!live_bos);
   free(dev);
   puts("PASS direct uploads; recording lifetime; reset, trim and recycled-buffer release");
}
