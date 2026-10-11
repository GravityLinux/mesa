/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_pool.h"
#include "asahi/lib/agx_bo.h"

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

   bo = vk_zalloc(&pool->vk.alloc, sizeof(*bo), 8,
                  VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bo == NULL)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);

   static const char *const labels[HK_CMD_POOL_COUNT] = {
      "Main command pool",
      "USC command pool",
      "Render context command pool",
   };
   bo->bo = agx_bo_create(&dev->dev, HK_CMD_BO_SIZE, 0,
                          hk_cmd_pool_bo_flags(dev, kind), labels[kind]);
   if (bo->bo == NULL) {
      vk_free(&pool->vk.alloc, bo);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   bo->map = agx_bo_map(bo->bo);

   *bo_out = bo;
   return VK_SUCCESS;
}

static void
hk_cmd_bo_destroy(struct hk_cmd_pool *pool, struct hk_cmd_bo *bo)
{
   struct hk_device *dev = hk_cmd_pool_device(pool);
   agx_bo_unreference(&dev->dev, bo->bo);
   vk_free(&pool->vk.alloc, bo);
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

   for (unsigned i = 0; i < HK_CMD_POOL_COUNT; ++i)
      list_inithead(&pool->free_bos[i]);

   *pCmdPool = hk_cmd_pool_to_handle(pool);

   return VK_SUCCESS;
}

static void
hk_cmd_pool_destroy_bos(struct hk_cmd_pool *pool)
{
   for (unsigned i = 0; i < HK_CMD_POOL_COUNT; ++i) {
      list_for_each_entry_safe(struct hk_cmd_bo, bo, &pool->free_bos[i], link)
         hk_cmd_bo_destroy(pool, bo);
      list_inithead(&pool->free_bos[i]);
      pool->num_free_bos[i] = 0;
   }
}

VkResult
hk_cmd_pool_alloc_bo(struct hk_cmd_pool *pool, enum hk_cmd_pool_kind kind,
                     struct hk_cmd_bo **bo_out)
{
   if (!list_is_empty(&pool->free_bos[kind])) {
      struct hk_cmd_bo *bo =
         list_first_entry(&pool->free_bos[kind], struct hk_cmd_bo, link);
      list_del(&bo->link);
      pool->num_free_bos[kind]--;
      *bo_out = bo;
      return VK_SUCCESS;
   }

   return hk_cmd_bo_create(pool, kind, bo_out);
}

/* Return a command buffer's BOs to the pool's free list (at most
 * HK_CMD_POOL_BO_MAX per kind), or with release (a reset with
 * RELEASE_RESOURCES) to the device. As in NVK and ANV, there is no device-wide
 * budget: a pool keeps its free list until trim, a releasing reset or destroy,
 * and the BO cache ages out what is returned. */
void
hk_cmd_pool_free_bo_list(struct hk_cmd_pool *pool, enum hk_cmd_pool_kind kind,
                         struct list_head *bos, bool release)
{
   list_for_each_entry_safe(struct hk_cmd_bo, bo, bos, link) {
      list_del(&bo->link);
      if (release || pool->num_free_bos[kind] >= HK_CMD_POOL_BO_MAX) {
         hk_cmd_bo_destroy(pool, bo);
      } else {
         list_addtail(&bo->link, &pool->free_bos[kind]);
         pool->num_free_bos[kind]++;
      }
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
   hk_cmd_pool_destroy_bos(pool);
   vk_free2(&device->vk.alloc, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetCommandPool(VkDevice device, VkCommandPool commandPool,
                    VkCommandPoolResetFlags flags)
{
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);

   VkResult result = vk_common_ResetCommandPool(device, commandPool, flags);
   if (flags & VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT)
      hk_cmd_pool_destroy_bos(pool);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
hk_TrimCommandPool(VkDevice device, VkCommandPool commandPool,
                   VkCommandPoolTrimFlags flags)
{
   VK_FROM_HANDLE(hk_cmd_pool, pool, commandPool);

   vk_command_pool_trim(&pool->vk, flags);
   hk_cmd_pool_destroy_bos(pool);
}
