/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_pool.h"
#include "hk_cmd_buffer.h"
#include "asahi/lib/agx_bo.h"
#include "util/u_debug.h"

#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "vk_common_entrypoints.h"

enum agx_bo_flags
hk_cmd_pool_bo_flags(const struct hk_device *dev, enum hk_cmd_pool_kind kind)
{
   switch (kind) {
   case HK_CMD_POOL_MAIN:
      return 0;
   case HK_CMD_POOL_USC:
      return AGX_BO_LOW_VA |
             (dev->dev.chip == AGX_CHIP_G16G ? AGX_BO_WRITEBACK : 0);
   case HK_CMD_POOL_CONTEXT:
      assert(dev->dev.chip == AGX_CHIP_G16G);
      return AGX_BO_CONTEXT | AGX_BO_WRITEBACK;
   default:
      UNREACHABLE("Invalid command pool kind");
   }
}

static VkResult
hk_cmd_bo_create(struct hk_cmd_pool *pool, enum hk_cmd_pool_kind kind,
                 struct hk_cmd_bo **bo_out)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   struct hk_cmd_bo *bo;

   /* Device scope: an idle BO may be reused by any pool of the device. */
   bo = vk_zalloc(&dev->vk.alloc, sizeof(*bo), 8,
                  VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (bo == NULL)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);

   static const char *const labels[HK_CMD_POOL_COUNT] = {
      "Main command pool",
      "USC command pool",
      "Render context command pool",
   };
   uint64_t t_bo = hk_cpu_sub_begin();
   bo->bo = agx_bo_create(&dev->dev, HK_CMD_BO_SIZE, 0,
                          hk_cmd_pool_bo_flags(dev, kind), labels[kind]);
   hk_cpu_sub_end(t_bo, HK_CPU_SUB_CMD_BO);
   if (bo->bo == NULL) {
      vk_free(&dev->vk.alloc, bo);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   bo->map = agx_bo_map(bo->bo);

   *bo_out = bo;
   return VK_SUCCESS;
}

static void
hk_cmd_bo_destroy(struct hk_device *dev, struct hk_cmd_bo *bo)
{
   agx_bo_unreference(&dev->dev, bo->bo);
   vk_free(&dev->vk.alloc, bo);
}

/* Hand every idle BO back to the BO cache. */
void
hk_cmd_bo_cache_trim(struct hk_device *dev)
{
   struct hk_cmd_bo_cache *cache = &dev->cmd_bos;
   struct hk_cmd_bo *idle[HK_CMD_POOL_COUNT];

   simple_mtx_lock(&cache->lock);
   memcpy(idle, cache->free, sizeof(idle));
   memset(cache->free, 0, sizeof(cache->free));
   cache->bytes = 0;
   simple_mtx_unlock(&cache->lock);

   for (unsigned i = 0; i < HK_CMD_POOL_COUNT; ++i) {
      while (idle[i]) {
         struct hk_cmd_bo *bo = idle[i];
         idle[i] = bo->next_free;
         hk_cmd_bo_destroy(dev, bo);
      }
   }
}

void
hk_cmd_bo_cache_finish(struct hk_device *dev)
{
   hk_cmd_bo_cache_trim(dev);
   simple_mtx_destroy(&dev->cmd_bos.lock);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateCommandPool(VkDevice _device,
                     const VkCommandPoolCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkCommandPool *pCmdPool)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   struct hk_cmd_pool *pool;

   pool = vk_zalloc2(&device->vk.alloc, pAllocator, sizeof(*pool), 8,
                     VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pool == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_command_pool_init(&device->vk, &pool->vk, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&device->vk.alloc, pAllocator, pool);
      return result;
   }

   *pCmdPool = hk_cmd_pool_to_handle(pool);

   return VK_SUCCESS;
}

VkResult
hk_cmd_pool_alloc_bo(struct hk_cmd_pool *pool, enum hk_cmd_pool_kind kind,
                     struct hk_cmd_bo **bo_out)
{
   struct hk_cmd_bo_cache *cache = &hk_cmd_pool_device(pool)->cmd_bos;
   struct hk_cmd_bo *bo = NULL;

   simple_mtx_lock(&cache->lock);
   bo = cache->free[kind];
   if (bo) {
      cache->free[kind] = bo->next_free;
      cache->bytes -= HK_CMD_BO_SIZE;
   }
   simple_mtx_unlock(&cache->lock);

   if (bo) {
      *bo_out = bo;
      return VK_SUCCESS;
   }

   return hk_cmd_bo_create(pool, kind, bo_out);
}

/* Return a command buffer's BOs to the device's idle list, up to
 * HK_CMD_BO_CACHE_BYTES, or with release (a reset with RELEASE_RESOURCES) to
 * the BO cache. As in ANV, idle BOs are shared by every pool of the device;
 * unlike ANV's pool, the idle list is bounded, and the BO cache ages out what
 * does not fit. */
void
hk_cmd_pool_free_bo_list(struct hk_cmd_pool *pool, enum hk_cmd_pool_kind kind,
                         struct list_head *bos, bool release)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   struct hk_cmd_bo_cache *cache = &dev->cmd_bos;
   struct list_head excess;
   list_inithead(&excess);

   simple_mtx_lock(&cache->lock);
   list_for_each_entry_safe(struct hk_cmd_bo, bo, bos, link) {
      list_del(&bo->link);
      if (release || cache->bytes + HK_CMD_BO_SIZE > HK_CMD_BO_CACHE_BYTES) {
         list_addtail(&bo->link, &excess);
      } else {
         bo->next_free = cache->free[kind];
         cache->free[kind] = bo;
         cache->bytes += HK_CMD_BO_SIZE;
      }
   }
   simple_mtx_unlock(&cache->lock);

   list_for_each_entry_safe(struct hk_cmd_bo, bo, &excess, link) {
      hk_cpu_sub_count(release ? HK_CPU_SUB_CMD_BO_DESTROY_RELEASE
                               : HK_CPU_SUB_CMD_BO_DESTROY_CAP);
      hk_cmd_bo_destroy(dev, bo);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyCommandPool(VkDevice _device, VkCommandPool commandPool,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);

   if (!pool)
      return;

   vk_command_pool_finish(&pool->vk);
   vk_free2(&device->vk.alloc, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetCommandPool(VkDevice device, VkCommandPool commandPool,
                    VkCommandPoolResetFlags flags)
{
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);

   VkResult result = vk_common_ResetCommandPool(device, commandPool, flags);
   if (flags & VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) {
      /* The common reset visits allocated buffers, not recycled ones. */
      vk_command_pool_trim(&pool->vk, 0);
   }
   return result;
}

VKAPI_ATTR void VKAPI_CALL
hk_TrimCommandPool(VkDevice device, VkCommandPool commandPool,
                   VkCommandPoolTrimFlags flags)
{
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);
   struct hk_device *dev = hk_cmd_pool_device(pool);

   vk_command_pool_trim(&pool->vk, flags);
   list_for_each_entry(struct hk_cmd_buffer, cmd, &pool->vk.command_buffers,
                       vk.pool_link) {
      if (cmd->vk.state == MESA_VK_COMMAND_BUFFER_STATE_INITIAL)
         hk_trim_cmd_buffer(cmd);
   }

   /* An explicit trim is a request to return unused memory. Drop the
    * device's idle BOs (other pools only reallocate) and the BO cache's copies
    * of the same pages, rather than waiting for a later allocation or submit
    * to age them out. Ordinary resets stay cheap. */
   hk_cmd_bo_cache_trim(dev);
   agx_bo_cache_evict_all(&dev->dev);
}
