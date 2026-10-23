/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "asahi/debug.h"
#include "hk_device_memory.h"

#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include "asahi/lib/agx_bo.h"
#include "wsi_common.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"

#include <inttypes.h>
#include <sys/mman.h>

/* Supports opaque fd only */
const VkExternalMemoryProperties hk_opaque_fd_mem_props = {
   .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
   .exportFromImportedHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
};

/* Supports opaque fd and dma_buf. */
const VkExternalMemoryProperties hk_dma_buf_mem_props = {
   .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
   .exportFromImportedHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
};

static enum agx_bo_flags
hk_memory_type_flags(const VkMemoryType *type,
                     VkExternalMemoryHandleTypeFlagBits handle_types)
{
   unsigned flags = 0;

   if (handle_types)
      flags |= AGX_BO_SHARED | AGX_BO_SHAREABLE;

   return flags;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetMemoryFdPropertiesKHR(VkDevice device,
                            VkExternalMemoryHandleTypeFlagBits handleType,
                            int fd,
                            VkMemoryFdPropertiesKHR *pMemoryFdProperties)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct agx_bo *bo;

   switch (handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
      bo = agx_bo_import(&dev->dev, fd);
      if (bo == NULL)
         return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      break;
   default:
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }

   uint32_t type_bits = 0;
   for (unsigned t = 0; t < pdev->mem_type_count; t++) {
      const unsigned flags =
         hk_memory_type_flags(&pdev->mem_types[t], handleType);
      if (!(flags & ~bo->flags))
         type_bits |= (1 << t);
   }

   pMemoryFdProperties->memoryTypeBits = type_bits & hk_general_memory_types(pdev);

   agx_bo_unreference(&dev->dev, bo);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_AllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                  const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct hk_device_memory *mem;
   VkResult result = VK_SUCCESS;

   const VkImportMemoryFdInfoKHR *fd_info =
      vk_find_struct_const(pAllocateInfo->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   const VkExportMemoryAllocateInfo *export_info =
      vk_find_struct_const(pAllocateInfo->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   const VkMemoryType *type = &pdev->mem_types[pAllocateInfo->memoryTypeIndex];

   VkExternalMemoryHandleTypeFlagBits handle_types = 0;
   if (export_info != NULL)
      handle_types |= export_info->handleTypes;
   if (fd_info != NULL)
      handle_types |= fd_info->handleType;

   const unsigned flags = hk_memory_type_flags(type, handle_types);

   uint32_t alignment = 16384; /* Apple page size */

   struct hk_memory_heap *heap = &pdev->mem_heaps[type->heapIndex];
   if (p_atomic_read(&heap->used) > heap->size)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   const uint64_t aligned_size =
      align64(pAllocateInfo->allocationSize, alignment);

   mem = vk_device_memory_create(&dev->vk, pAllocateInfo, pAllocator,
                                 sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   simple_mtx_init(&mem->index_va_lock, mtx_plain);
   mem->index_va = NULL;
   mem->map = NULL;
   if (fd_info && fd_info->handleType) {
      assert(
         fd_info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT ||
         fd_info->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);

      mem->bo = agx_bo_import(&dev->dev, fd_info->fd);
      if (mem->bo == NULL) {
         result = vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
         goto fail_alloc;
      }
      assert(!(flags & ~mem->bo->flags));
   } else {
      enum agx_bo_flags flags = 0;
      if (handle_types)
         flags |= AGX_BO_SHAREABLE;
      if (type->propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
         flags |= AGX_BO_WRITEBACK;
      if (pAllocateInfo->memoryTypeIndex == pdev->apple9_descriptor_mem_type)
         flags |= AGX_BO_LOW_VA;

      /* D3D12 places resources in heaps at 64-KiB granularity and
       * guarantees the alignment of their GPU addresses. Translation layers
       * derive those addresses from the memory's, so align large ones.
       */
      unsigned va_align = aligned_size >= 65536 ? 65536 : 0;

      /* Capture-replay memory lives in the replay range, at its captured
       * address on replay. Buffers and images bound to it get their captured
       * addresses with it, and so do descriptors that embed them.
       */
      enum agx_va_flags va_flags = 0;
      uint64_t replay_addr = 0;
      if (mem->vk.alloc_flags &
          VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT) {
         const VkMemoryOpaqueCaptureAddressAllocateInfo *replay_info =
            vk_find_struct_const(pAllocateInfo->pNext,
                                 MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO);
         if (replay_info)
            replay_addr = replay_info->opaqueCaptureAddress;

         va_flags = AGX_VA_REPLAY | (replay_addr ? AGX_VA_FIXED : 0);
      }

      mem->bo = agx_bo_create_va(&dev->dev, aligned_size, va_align, flags,
                                 va_flags, replay_addr, "App memory");
      if (!mem->bo) {
         result = replay_addr
                     ? vk_errorf(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                                 "Replay address 0x%" PRIx64 " unavailable",
                                 replay_addr)
                     : vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto fail_alloc;
      }
   }

   /* Sparse buffers use a read-only shadow of their backing memory. Devices
    * without sparse residency never access it; in particular, rebinding a
    * cached BO's shadow is rejected by the experimental Apple9 kernel.
    */
   if (pdev->vk.supported_features.sparseResidencyBuffer &&
       !(mem->bo->flags & AGX_BO_LOW_VA)) {
      int ret = agx_bo_bind(&dev->dev, mem->bo,
                         agx_rw_addr_to_ro(&dev->dev, mem->bo->va->addr),
                         mem->bo->size, 0, DRM_ASAHI_BIND_READ);
      if (ret) {
         agx_bo_unreference(&dev->dev, mem->bo);
         result = vk_error(dev, VK_ERROR_UNKNOWN);
         goto fail_alloc;
      }
   }

   if (agx_apple9_debug_enabled(AGX_A9_DBG_REPLAY)) {
      const VkMemoryAllocateFlagsInfo *flags_info =
         vk_find_struct_const(pAllocateInfo->pNext, MEMORY_ALLOCATE_FLAGS_INFO);
      const VkMemoryOpaqueCaptureAddressAllocateInfo *opaque_info =
         vk_find_struct_const(pAllocateInfo->pNext,
                              MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO);
      fprintf(stderr,
              "hk-replay: memory size=0x%" PRIx64 " type=%u flags=0x%x "
              "requested=0x%" PRIx64 " got=0x%" PRIx64 "\n",
              (uint64_t)pAllocateInfo->allocationSize,
              pAllocateInfo->memoryTypeIndex,
              flags_info ? flags_info->flags : 0,
              opaque_info ? opaque_info->opaqueCaptureAddress : 0,
              mem->bo->va ? mem->bo->va->addr : 0);
   }

   /* Under virtio the host compositor reads WSI images with implicit sync,
    * which virglrenderer derives from a submission's external resources.
    * Other external memory uses explicit sync, as on native.
    */
   mem->implicit_sync = dev->dev.is_virtio &&
      vk_find_struct_const(pAllocateInfo->pNext,
                           WSI_MEMORY_ALLOCATE_INFO_MESA) != NULL;

   if (fd_info && fd_info->handleType) {
      /* From the Vulkan spec:
       *
       *    "Importing memory from a file descriptor transfers ownership of
       *    the file descriptor from the application to the Vulkan
       *    implementation. The application must not perform any operations on
       *    the file descriptor after a successful import."
       *
       * If the import fails, we leave the file descriptor open.
       */
      close(fd_info->fd);
   }

   uint64_t heap_used = p_atomic_add_return(&heap->used, mem->bo->size);
   p_atomic_add(&hk_process_heap_used[type->heapIndex], mem->bo->size);
   if (heap_used > heap->size) {
      hk_FreeMemory(device, hk_device_memory_to_handle(mem), pAllocator);
      return vk_errorf(device, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "Out of heap memory");
   }

   *pMem = hk_device_memory_to_handle(mem);

   return VK_SUCCESS;

fail_alloc:
   simple_mtx_destroy(&mem->index_va_lock);
   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
hk_FreeMemory(VkDevice device, VkDeviceMemory _mem,
              const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_device_memory, mem, _mem);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   if (!mem)
      return;

   const VkMemoryType *type = &pdev->mem_types[mem->vk.memory_type_index];
   struct hk_memory_heap *heap = &pdev->mem_heaps[type->heapIndex];
   p_atomic_add(&heap->used, -((int64_t)mem->bo->size));
   p_atomic_add(&hk_process_heap_used[type->heapIndex],
                -((int64_t)mem->bo->size));

   /* Freeing mapped memory implicitly unmaps it. */
   if (mem->map)
      munmap(mem->map, mem->map_size);

   hk_memory_free_index_va(dev, mem);
   simple_mtx_destroy(&mem->index_va_lock);
   agx_bo_unreference(&dev->dev, mem->bo);

   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
}

uint64_t
hk_memory_index_address(struct hk_device *dev, struct hk_device_memory *mem)
{
   if (mem->bo->va->flags & AGX_VA_USC)
      return mem->bo->va->addr;

   if (mem->vk.size > HK_INDEX_ALIAS_ALLOCATION_MAX)
      return 0;

   simple_mtx_lock(&mem->index_va_lock);
   if (!mem->index_va) {
      uint64_t size = align64(mem->vk.size, AIL_PAGESIZE);
      uint64_t charge = size + dev->dev.guard_size;
      uint64_t used = p_atomic_read(&dev->apple9_index_alias_bytes);
      for (;;) {
         if (charge > HK_INDEX_ALIAS_BUDGET ||
             used > HK_INDEX_ALIAS_BUDGET - charge) {
            simple_mtx_unlock(&mem->index_va_lock);
            return 0;
         }
         uint64_t previous = p_atomic_cmpxchg(&dev->apple9_index_alias_bytes,
                                             used, used + charge);
         if (previous == used)
            break;
         used = previous;
      }
      struct agx_va *va =
         agx_va_alloc(&dev->dev, size, AIL_PAGESIZE, AGX_VA_USC, 0);
      if (va) {
         if (agx_bo_bind(&dev->dev, mem->bo, va->addr, size, 0,
                         DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE) == 0)
            mem->index_va = va;
         else
            agx_va_free(&dev->dev, va, true);
      }
      if (!mem->index_va)
         p_atomic_add(&dev->apple9_index_alias_bytes, -((int64_t)charge));
   }
   uint64_t addr = mem->index_va ? mem->index_va->addr : 0;
   simple_mtx_unlock(&mem->index_va_lock);

   /* The caller uses a buffer-sized alias when sharing is unavailable. */
   return addr;
}

void
hk_memory_free_index_va(struct hk_device *dev, struct hk_device_memory *mem)
{
   if (!mem->index_va)
      return;

   uint64_t charge = mem->index_va->size_B;
   agx_va_free(&dev->dev, mem->index_va, true);
   p_atomic_add(&dev->apple9_index_alias_bytes, -((int64_t)charge));
   mem->index_va = NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_MapMemory2KHR(VkDevice device, const VkMemoryMapInfoKHR *pMemoryMapInfo,
                 void **ppData)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_device_memory, mem, pMemoryMapInfo->memory);

   if (mem == NULL) {
      *ppData = NULL;
      return VK_SUCCESS;
   }

   const VkDeviceSize offset = pMemoryMapInfo->offset;
   const VkDeviceSize size = vk_device_memory_range(
      &mem->vk, pMemoryMapInfo->offset, pMemoryMapInfo->size);

   void *fixed_addr = NULL;
   if (pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT) {
      const VkMemoryMapPlacedInfoEXT *placed_info = vk_find_struct_const(
         pMemoryMapInfo->pNext, MEMORY_MAP_PLACED_INFO_EXT);
      fixed_addr = placed_info->pPlacedAddress;
   }

   /* From the Vulkan spec version 1.0.32 docs for MapMemory:
    *
    *  * If size is not equal to VK_WHOLE_SIZE, size must be greater than 0
    *    assert(size != 0);
    *  * If size is not equal to VK_WHOLE_SIZE, size must be less than or
    *    equal to the size of the memory minus offset
    */
   assert(size > 0);
   assert(offset + size <= mem->bo->size);

   if (size != (size_t)size) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "requested size 0x%" PRIx64 " does not fit in %u bits",
                       size, (unsigned)(sizeof(size_t) * 8));
   }

   /* From the Vulkan 1.2.194 spec:
    *
    *    "memory must not be currently host mapped"
    */
   if (mem->map != NULL) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "Memory object already mapped.");
   }

   /* Map the allocation, not the BO, which is larger when it came from the BO
    * cache: a placed mapping must leave the pages after the allocation alone.
    */
   struct hk_physical_device *pdev = hk_device_physical(dev);
   size_t map_size =
      align64(mem->vk.size, pdev->vk.properties.minMemoryMapAlignment);
   assert(map_size <= mem->bo->size);

   mem->map = agx_bo_map_range(mem->bo, fixed_addr, map_size);
   if (mem->map == NULL)
      return vk_error(dev, VK_ERROR_MEMORY_MAP_FAILED);

   assert(!fixed_addr || mem->map == fixed_addr);
   mem->map_size = map_size;
   *ppData = mem->map + offset;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_UnmapMemory2KHR(VkDevice device,
                   const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(hk_device_memory, mem, pMemoryUnmapInfo->memory);

   if (mem == NULL)
      return VK_SUCCESS;

   if (mem->map == NULL)
      return VK_SUCCESS;

   if (pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT) {
      VK_FROM_HANDLE(hk_device, dev, device);

      void *err = mmap(mem->map, mem->map_size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      if (err == MAP_FAILED) {
         return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                          "Failed to map over original mapping");
      }
   } else {
      munmap(mem->map, mem->map_size);
   }

   mem->map = NULL;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_FlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                           const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_InvalidateMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                                const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory _mem,
                             VkDeviceSize *pCommittedMemoryInBytes)
{
   VK_FROM_HANDLE(hk_device_memory, mem, _mem);

   *pCommittedMemoryInBytes = mem->bo->size;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetMemoryFdKHR(VkDevice device, const VkMemoryGetFdInfoKHR *pGetFdInfo,
                  int *pFD)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_device_memory, memory, pGetFdInfo->memory);

   switch (pGetFdInfo->handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
      *pFD = agx_bo_export(&dev->dev, memory->bo);
      return VK_SUCCESS;
   default:
      assert(!"unsupported handle type");
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }
}

VKAPI_ATTR uint64_t VKAPI_CALL
hk_GetDeviceMemoryOpaqueCaptureAddress(
   UNUSED VkDevice device,
   const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo)
{
   VK_FROM_HANDLE(hk_device_memory, mem, pInfo->memory);

   /* Only capture-replay memory can be placed at a given address again. */
   if (!(mem->bo->va->flags & AGX_VA_REPLAY))
      return 0;

   return mem->bo->va->addr;
}

/* Device memory is never paged out, so priorities have no effect. */
VKAPI_ATTR void VKAPI_CALL
hk_SetDeviceMemoryPriorityEXT(VkDevice device, VkDeviceMemory memory,
                              float priority)
{
}
