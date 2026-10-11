/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "util/u_debug.h"
#include "hk_buffer.h"

#include "agx_bo.h"
#include "agx_device.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"

static uint32_t
hk_get_buffer_alignment(const struct hk_physical_device *pdev,
                        VkBufferUsageFlags2KHR usage_flags,
                        VkBufferCreateFlags create_flags)
{
   uint32_t alignment = 16;

   /* Native VDM index pointers are relative to the USC aperture. Mapping a
    * buffer there requires page-aligned memory bindings, not a CPU copy. */
   if (pdev->dev.chip == AGX_CHIP_G16G &&
       (usage_flags & VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT))
      alignment = 16384;

   if (usage_flags & VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR)
      alignment = MAX2(alignment, HK_MIN_UBO_ALIGNMENT);

   if (usage_flags & VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR)
      alignment = MAX2(alignment, hk_min_ssbo_alignment(&pdev->dev));

   if (usage_flags & (VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT_KHR |
                      VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT_KHR))
      alignment = MAX2(alignment, HK_MIN_TEXEL_BUFFER_ALIGNMENT);

   if (create_flags & (VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
                       VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT))
      alignment = MAX2(alignment, 16384);

   return alignment;
}

static uint64_t
hk_get_bda_replay_addr(const VkBufferCreateInfo *pCreateInfo)
{
   uint64_t addr = 0;
   vk_foreach_struct_const(ext, pCreateInfo->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO: {
         const VkBufferOpaqueCaptureAddressCreateInfo *bda = (void *)ext;
         if (bda->opaqueCaptureAddress != 0) {
#ifdef NDEBUG
            return bda->opaqueCaptureAddress;
#else
            assert(addr == 0 || bda->opaqueCaptureAddress == addr);
            addr = bda->opaqueCaptureAddress;
#endif
         }
         break;
      }

      case VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT: {
         const VkBufferDeviceAddressCreateInfoEXT *bda = (void *)ext;
         if (bda->deviceAddress != 0) {
#ifdef NDEBUG
            return bda->deviceAddress;
#else
            assert(addr == 0 || bda->deviceAddress == addr);
            addr = bda->deviceAddress;
#endif
         }
         break;
      }

      default:
         break;
      }
   }

   return addr;
}

DEBUG_GET_ONCE_BOOL_OPTION(hk_sparse_unmapped_raw_opt,
                           "HK_APPLE9_SPARSE_UNMAPPED_RAW", false)

bool
hk_sparse_unmapped_raw(void)
{
   return debug_get_option_hk_sparse_unmapped_raw_opt();
}

VkResult
hk_bind_scratch(struct hk_device *dev, struct agx_va *va, unsigned offset_B,
                size_t size_B)
{
   uint64_t addr = va->addr + offset_B;
   uint32_t flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_SINGLE_PAGE;

   /* Strict residency: unbound ranges map the zero page read-only in both
    * views, so reads are zero and soft faults drop writes, see
    * hk_sparse_unbound_unmapped. HK_APPLE9_SPARSE_UNMAPPED_RAW=1 leaves
    * them unmapped instead (soft-faulted reads, about 10x slower).
    */
   if (hk_sparse_unbound_unmapped(&dev->dev)) {
      if (hk_sparse_unmapped_raw())
         return VK_SUCCESS;
      if (agx_bo_bind(&dev->dev, dev->dev.zero_bo, addr, size_B, 0, flags) ||
          (hk_va_has_ro_shadow(dev, addr) &&
           agx_bo_bind(&dev->dev, dev->dev.zero_bo,
                       addr + dev->dev.sparse_ro_offset, size_B, 0, flags)))
         return VK_ERROR_UNKNOWN;
      return VK_SUCCESS;
   }

   /* Map read-write scratch to the primary (bottom half) VA range */
   int ret = agx_bo_bind(&dev->dev, dev->dev.scratch_bo, addr, size_B, 0,
                         flags | DRM_ASAHI_BIND_WRITE);
   if (ret)
      return VK_ERROR_UNKNOWN;

   /* Map read-only scratch to the secondary (top half) VA range */
   if (!hk_va_has_ro_shadow(dev, addr))
      return VK_SUCCESS;
   ret = agx_bo_bind(&dev->dev, dev->dev.zero_bo,
                     addr + dev->dev.sparse_ro_offset, size_B, 0, flags);
   if (ret)
      return VK_ERROR_UNKNOWN;

   return VK_SUCCESS;
}

/* Release a buffer's own VA. agx_va_free unbinds only the read-write range;
 * the read-only shadow (bound by hk_bind_scratch and buffer memory binds) must
 * be unbound too, or it keeps pointing at the old pages after the VA is reused.
 */
static void
hk_buffer_free_va(struct hk_device *dev, struct hk_buffer *buffer)
{
   if (!buffer->apple9_index_va && hk_va_has_ro_shadow(dev, buffer->va->addr))
      agx_bo_bind(&dev->dev, NULL, buffer->va->addr + dev->dev.sparse_ro_offset,
                  buffer->va->size_B, 0, DRM_ASAHI_BIND_UNBIND);
   agx_va_free(&dev->dev, buffer->va, true);
   buffer->va = NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateBuffer(VkDevice device, const VkBufferCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_buffer *buffer;

   if (pCreateInfo->size > HK_MAX_BUFFER_SIZE)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   buffer =
      vk_buffer_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*buffer));
   if (!buffer)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Sparse buffers keep the read-only zero mirror of the main heap. Their
    * indices, like those of any buffer outside the 4-GiB USC aperture, are
    * copied into it at draw time.
    */
   const bool sparse =
      buffer->vk.create_flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT;
   buffer->apple9_index_va = dev->dev.chip == AGX_CHIP_G16G && !sparse &&
      (buffer->vk.usage & VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT);
   if (buffer->vk.size > 0 &&
       (buffer->apple9_index_va || (buffer->vk.create_flags &
        (VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
         VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT)))) {

      const bool bda_capture_replay =
         buffer->vk.create_flags &
         VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
      const uint64_t bda_fixed_addr =
         bda_capture_replay ? hk_get_bda_replay_addr(pCreateInfo) : 0;

      uint32_t alignment = hk_get_buffer_alignment(
         hk_device_physical(dev), buffer->vk.usage, buffer->vk.create_flags);
      assert(alignment >= 16384);
      /* As for device memory, align large buffers to 64 KiB. A replayed
       * address keeps the size it was captured with: captures made with
       * 16 KiB rounding place the next buffer right after it.
       */
      if (buffer->vk.size >= 65536 && !bda_fixed_addr)
         alignment = MAX2(alignment, 65536);
      uint64_t vma_size_B = align64(buffer->vk.size, alignment);

      enum agx_va_flags flags = buffer->apple9_index_va ? AGX_VA_USC : 0;
      /* Shaders address descriptor records in the USC aperture. */
      if (dev->dev.chip == AGX_CHIP_G16G &&
          (buffer->vk.usage &
           (VK_BUFFER_USAGE_2_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
            VK_BUFFER_USAGE_2_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
            VK_BUFFER_USAGE_2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT)))
         flags |= AGX_VA_USC;
      if (bda_capture_replay)
         flags |= bda_fixed_addr != 0 ? AGX_VA_FIXED : AGX_VA_LOW;

      buffer->va =
         agx_va_alloc(&dev->dev, vma_size_B, alignment, flags, bda_fixed_addr);

      /* The USC aperture is shared with shaders. When it is full, place the
       * buffer like any other and copy its indices at draw time.
       */
      if (!buffer->va && buffer->apple9_index_va && !bda_fixed_addr) {
         buffer->apple9_index_va = false;
         flags &= ~AGX_VA_USC;
         if (bda_capture_replay)
            buffer->va = agx_va_alloc(&dev->dev, vma_size_B, alignment, flags,
                                      0);
         else
            goto done;
      }

      if (!buffer->va) {
         vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "Sparse VMA allocation failed");
      }
      buffer->vk.device_address = buffer->va->addr;

      /* Bind scratch pages to make read/write across the VA valid */
      if (!buffer->apple9_index_va &&
          hk_bind_scratch(dev, buffer->va, 0, vma_size_B) != VK_SUCCESS) {
         hk_buffer_free_va(dev, buffer);
         vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      /* Apple9 texel fetches report no residency, so shaders read it from
       * a table of the buffer's pages, all unbound initially.
       */
      if (dev->dev.chip == AGX_CHIP_G16G &&
          (buffer->vk.create_flags & VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT)) {
         size_t pages = DIV_ROUND_UP(vma_size_B, AIL_PAGESIZE);
         buffer->apple9_residency =
            agx_bo_create(&dev->dev, pages, 0, 0, "Apple9 buffer residency");
         if (!buffer->apple9_residency) {
            hk_buffer_free_va(dev, buffer);
            vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
            return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         }
         memset(agx_bo_map(buffer->apple9_residency), 0, pages);
      }
   }

done:
   *pBuffer = hk_buffer_to_handle(buffer);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyBuffer(VkDevice device, VkBuffer _buffer,
                 const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   if (!buffer)
      return;

   if (buffer->va)
      hk_buffer_free_va(dev, buffer);

   if (buffer->apple9_residency)
      agx_bo_unreference(&dev->dev, buffer->apple9_residency);

   vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceBufferMemoryRequirements(
   VkDevice device, const VkDeviceBufferMemoryRequirements *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   const uint32_t alignment = hk_get_buffer_alignment(
      hk_device_physical(dev), pInfo->pCreateInfo->usage,
      pInfo->pCreateInfo->flags);

   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = align64(pInfo->pCreateInfo->size, alignment),
      .alignment = alignment,
      .memoryTypeBits = hk_general_memory_types(pdev),
   };

   /* Descriptor buffers live in the USC aperture with descriptor pools. */
   const VkBufferUsageFlags2KHR descriptor_usage =
      VK_BUFFER_USAGE_2_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
      VK_BUFFER_USAGE_2_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
      VK_BUFFER_USAGE_2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT;
   if (pdev->apple9_descriptor_mem_type != UINT8_MAX &&
       (vk_buffer_usage_flags(pInfo->pCreateInfo) & descriptor_usage)) {
      pMemoryRequirements->memoryRequirements.memoryTypeBits =
         BITFIELD_BIT(pdev->apple9_descriptor_mem_type);
   }

   vk_foreach_struct_const(ext, pMemoryRequirements->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *dedicated = (void *)ext;
         dedicated->prefersDedicatedAllocation = false;
         dedicated->requiresDedicatedAllocation = false;
         break;
      }
      default:
         vk_debug_ignored_stype(ext->sType);
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceExternalBufferProperties(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceExternalBufferInfo *pExternalBufferInfo,
   VkExternalBufferProperties *pExternalBufferProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   /* The Vulkan 1.3.256 spec says:
    *
    *    VUID-VkPhysicalDeviceExternalBufferInfo-handleType-parameter
    *
    *    "handleType must be a valid VkExternalMemoryHandleTypeFlagBits value"
    *
    * This differs from VkPhysicalDeviceExternalImageFormatInfo, which
    * surprisingly permits handleType == 0.
    */
   assert(pExternalBufferInfo->handleType != 0);

   /* All of the current flags are for sparse which we don't support yet.
    * Even when we do support it, doing sparse on external memory sounds
    * sketchy.  Also, just disallowing flags is the safe option.
    */
   if (pExternalBufferInfo->flags)
      goto unsupported;

   /* Capability queries must agree with the device's external-memory API. */
   if (!pdev->vk.supported_extensions.KHR_external_memory_fd ||
       (pExternalBufferInfo->handleType ==
           VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT &&
        !pdev->vk.supported_extensions.EXT_external_memory_dma_buf))
      goto unsupported;

   switch (pExternalBufferInfo->handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
      pExternalBufferProperties->externalMemoryProperties =
         hk_dma_buf_mem_props;
      return;
   default:
      goto unsupported;
   }

unsupported:
   /* From the Vulkan 1.3.256 spec:
    *
    *    compatibleHandleTypes must include at least handleType.
    */
   pExternalBufferProperties->externalMemoryProperties =
      (VkExternalMemoryProperties){
         .compatibleHandleTypes = pExternalBufferInfo->handleType,
      };
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_BindBufferMemory2(VkDevice device, uint32_t bindInfoCount,
                     const VkBindBufferMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; ++i) {
      VK_FROM_HANDLE(hk_device_memory, mem, pBindInfos[i].memory);
      VK_FROM_HANDLE(hk_buffer, buffer, pBindInfos[i].buffer);

      buffer->implicit_sync_res =
         mem->implicit_sync ? mem->bo->uapi_handle : 0;

      if (buffer->va) {
         VK_FROM_HANDLE(hk_device, dev, device);
         size_t size = MIN2(mem->bo->size - pBindInfos[i].memoryOffset,
                            buffer->va->size_B);

         /* Lower mapping: read-write */
         int ret = agx_bo_bind(&dev->dev, mem->bo, buffer->vk.device_address,
                               size, pBindInfos[i].memoryOffset,
                               DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
         if (ret)
            return VK_ERROR_UNKNOWN;

         /* Apple9 index aliases and USC-aperture buffers need only one
          * read-write mapping. */
         if (!buffer->apple9_index_va &&
             hk_va_has_ro_shadow(dev, buffer->vk.device_address)) {
            /* Upper mapping: read-only */
            ret =
               agx_bo_bind(&dev->dev, mem->bo,
                           buffer->vk.device_address + dev->dev.sparse_ro_offset,
                           size, pBindInfos[i].memoryOffset, DRM_ASAHI_BIND_READ);
            if (ret)
               return VK_ERROR_UNKNOWN;
         }
      } else {
         assert(buffer->vk.device_address == 0);
         buffer->vk.device_address =
            mem->bo->va->addr + pBindInfos[i].memoryOffset;
      }

      const VkBindMemoryStatusKHR *status =
         vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
      if (status != NULL && status->pResult != NULL)
         *status->pResult = VK_SUCCESS;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR uint64_t VKAPI_CALL
hk_GetBufferOpaqueCaptureAddress(UNUSED VkDevice device,
                                 const VkBufferDeviceAddressInfo *pInfo)
{
   VK_FROM_HANDLE(hk_buffer, buffer, pInfo->buffer);

   return hk_buffer_address_rw(buffer, 0);
}

uint64_t
hk_buffer_address(const struct hk_buffer *buffer, uint64_t offset,
                  bool read_only)
{
   struct hk_device *dev = (struct hk_device *)buffer->vk.base.device;
   uint64_t addr = vk_buffer_address(&buffer->vk, offset);

   /* If we are accessing the buffer read-only, we want to return the read-only
    * shadow mapping so non-resident pages return zeroes. That only applies to
    * sparse resident buffers, which will have buffer->va != NULL. If buffer->va
    * is NULL, the buffer is not sparse resident, so we don't need the fix up...
    * and indeed, there may not be a shadow map available.
    */
   if (read_only && buffer->va && !buffer->apple9_index_va &&
       hk_va_has_ro_shadow(dev, addr)) {
      addr = agx_rw_addr_to_ro(&dev->dev, addr);
   }

   return addr;
}
