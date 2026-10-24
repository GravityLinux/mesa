/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_feedback.h"
#include "hk_event.h"
#include "vulkan/vulkan_core.h"

#include "agx_bo.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "vk_alloc.h"
#include "util/bitset.h"

#define HK_EVENT_PAGE_SIZE 16384
#define HK_EVENTS_PER_PAGE (HK_EVENT_PAGE_SIZE / sizeof(VkResult))

struct hk_event_page {
   struct list_head link;
   struct agx_bo *bo;
   VkResult *map;
   unsigned used;
   BITSET_DECLARE(free, HK_EVENTS_PER_PAGE);
};

static VkResult
hk_event_alloc(struct hk_device *dev, struct hk_event *event)
{
   simple_mtx_lock(&dev->events.lock);
   struct hk_event_page *page;
   VkResult result = VK_SUCCESS;
   if (list_is_empty(&dev->events.available)) {
      page = vk_zalloc(&dev->vk.alloc, sizeof(*page), 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (!page) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto out;
      }
      page->bo = agx_bo_create(&dev->dev, HK_EVENT_PAGE_SIZE, 0,
                               AGX_BO_WRITEBACK, "Events");
      if (!page->bo) {
         vk_free(&dev->vk.alloc, page);
         result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
         goto out;
      }
      page->map = agx_bo_map(page->bo);
      if (!page->map) {
         agx_bo_unreference(&dev->dev, page->bo);
         vk_free(&dev->vk.alloc, page);
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto out;
      }
      BITSET_ONES(page->free);
      list_addtail(&page->link, &dev->events.available);
   } else {
      page = list_first_entry(&dev->events.available, struct hk_event_page, link);
   }

   unsigned slot = BITSET_FFS(page->free) - 1;
   assert(slot < HK_EVENTS_PER_PAGE);
   BITSET_CLEAR(page->free, slot);
   if (++page->used == HK_EVENTS_PER_PAGE)
      list_del(&page->link);
   event->page = page;
   event->slot = slot;
   event->status = &page->map[slot];
   event->addr = page->bo->va->addr + slot * sizeof(VkResult);

out:
   simple_mtx_unlock(&dev->events.lock);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateEvent(VkDevice device, const VkEventCreateInfo *pCreateInfo,
               const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_event *event;

   event = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*event),
                            VK_OBJECT_TYPE_EVENT);
   if (!event)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result = hk_event_alloc(dev, event);
   if (result != VK_SUCCESS) {
      vk_object_free(&dev->vk, pAllocator, event);
      return vk_error(dev, result);
   }

   *event->status = VK_EVENT_RESET;

   *pEvent = hk_event_to_handle(event);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyEvent(VkDevice device, VkEvent _event,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_event, event, _event);

   if (!event)
      return;

   struct hk_event_page *page = event->page;
   simple_mtx_lock(&dev->events.lock);
   assert(!BITSET_TEST(page->free, event->slot));
   BITSET_SET(page->free, event->slot);
   if (page->used-- == HK_EVENTS_PER_PAGE)
      list_addtail(&page->link, &dev->events.available);
   bool empty = page->used == 0;
   if (empty)
      list_del(&page->link);
   simple_mtx_unlock(&dev->events.lock);

   /* Pages never move. Release only after every event using them is gone. */
   if (empty) {
      agx_bo_unreference(&dev->dev, page->bo);
      vk_free(&dev->vk.alloc, page);
   }
   vk_object_free(&dev->vk, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetEventStatus(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   return *event->status;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_SetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   *event->status = VK_EVENT_SET;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   *event->status = VK_EVENT_RESET;

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_cmd_render_merge_flush(cmd);
   hk_apple9_end_native_batch(cmd);
   VK_FROM_HANDLE(hk_event, event, _event);

   perf_debug(cmd, "Set event");
   hk_cs_end_hint(cmd, HK_CS_END_EVENT);
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_WRITE_BIT,
                        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
   hk_queue_write(cmd, event->addr, VK_EVENT_SET, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                  VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_cmd_render_merge_flush(cmd);
   hk_apple9_end_native_batch(cmd);
   VK_FROM_HANDLE(hk_event, event, _event);

   perf_debug(cmd, "Reset event");
   hk_cs_end_hint(cmd, HK_CS_END_EVENT);
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_WRITE_BIT,
                        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
   hk_queue_write(cmd, event->addr, VK_EVENT_RESET, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount,
                  const VkEvent *pEvents,
                  const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (!cmd->in_meta)
      hk_feedback_invalidate(cmd);
   hk_cmd_render_merge_flush(cmd);
   hk_apple9_end_native_batch(cmd);
   perf_debug(cmd, "Wait events");

   /* The big hammer. Need to check if this is actually needed.
    *
    * XXX: perf.
    */
   hk_cs_end_hint(cmd, HK_CS_END_EVENT);
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_WRITE_BIT,
                        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}
