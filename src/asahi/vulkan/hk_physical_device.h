/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_device.h"
#include <sys/types.h>
#include "hk_private.h"
#include "vk_physical_device.h"
#include "vk_sync.h"
#include "wsi_common.h"

struct hk_instance;
struct hk_physical_device;

struct hk_queue_family {
   VkQueueFlags queue_flags;
   uint32_t queue_count;
};

struct hk_memory_heap {
   uint64_t size;
   uint64_t used;
   VkMemoryHeapFlags flags;
};

/* Heap usage of the whole process, across instances, for memory budgets. */
extern uint64_t hk_process_heap_used[VK_MAX_MEMORY_HEAPS];

struct hk_physical_device {
   struct vk_physical_device vk;

   /* Apple9 with hk_apple9_arrayed_2d_views (driconf, vkd3d-proton): sampled
    * 1D/2D views are one-layer arrays and 2D textures are sampled as arrays
    * (see pack_texture). */
   bool apple9_layered_2d;
   /* Apple9 with hk_apple9_d3d_mip_rounding (driconf, vkd3d-proton). */
   bool apple9_d3d_mip_rounding;
   /* Kill switch for the qualified native-index default, not research opt-ins. */
   bool disable_native_index;
   dev_t render_dev;
   int master_fd;

   /* Only used for VK_EXT_memory_budget */
   struct agx_device dev;

   struct wsi_device wsi_device;

   uint8_t device_uuid[VK_UUID_SIZE];

   // TODO: add mapable VRAM heap if possible
   struct hk_memory_heap mem_heaps[3];
   VkMemoryType mem_types[3];
   uint8_t mem_heap_count;
   uint8_t mem_type_count;
   /* G16 descriptor-buffer memory: shaders address descriptor records
    * relative to the USC aperture, so this type allocates there (write-back,
    * as descriptor pools). UINT8_MAX when absent. Only buffers with a
    * descriptor-buffer usage may (and must) use it. */
   uint8_t apple9_descriptor_mem_type;
   uint64_t sysmem;

   struct hk_queue_family queue_families[3];
   uint8_t queue_family_count;

   struct vk_sync_type syncobj_sync_type;
   const struct vk_sync_type *sync_types[2];

   simple_mtx_t debug_compile_lock;
};

VK_DEFINE_HANDLE_CASTS(hk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

static inline struct hk_instance *
hk_physical_device_instance(struct hk_physical_device *pdev)
{
   return (struct hk_instance *)pdev->vk.instance;
}

/* Memory types usable by images and by buffers without a descriptor-buffer
 * usage. */
static inline uint32_t
hk_general_memory_types(const struct hk_physical_device *pdev)
{
   uint32_t types = BITFIELD_MASK(pdev->mem_type_count);
   if (pdev->apple9_descriptor_mem_type != UINT8_MAX)
      types &= ~BITFIELD_BIT(pdev->apple9_descriptor_mem_type);
   return types;
}

VkResult hk_create_drm_physical_device(struct vk_instance *vk_instance,
                                       struct _drmDevice *drm_device,
                                       struct vk_physical_device **pdev_out);

void hk_physical_device_destroy(struct vk_physical_device *vk_device);

#if defined(VK_USE_PLATFORM_WAYLAND_KHR) ||                                    \
   defined(VK_USE_PLATFORM_XCB_KHR) || defined(VK_USE_PLATFORM_XLIB_KHR) ||    \
   defined(VK_USE_PLATFORM_DISPLAY_KHR)
#define HK_USE_WSI_PLATFORM
#endif
