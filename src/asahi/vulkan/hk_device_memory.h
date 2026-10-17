/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "vk_device_memory.h"

#include "util/list.h"
#include "util/simple_mtx.h"

struct hk_device;
struct hk_image_plane;

/* Leave most of USC available to shaders, descriptors and command uploads. */
#define HK_INDEX_ALIAS_ALLOCATION_MAX (256ull << 20)
#define HK_INDEX_ALIAS_BUDGET (1ull << 30)

struct hk_device_memory {
   struct vk_device_memory vk;

   struct agx_bo *bo;

   /* Budgeted USC mapping for ordinary Apple9 index buffers bound to this
    * allocation. Keep it until FreeMemory: destroying a short-lived buffer
    * must not unbind a mapping of an otherwise reusable allocation.
    */
   simple_mtx_t index_va_lock;
   struct agx_va *index_va;

   /* Virtio: WSI memory, which the host compositor accesses with implicit
    * sync. Submissions using it list it as an external resource. */
   bool implicit_sync;

   /* The API mapping, which the memory object owns. */
   void *map;
   size_t map_size;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

extern const VkExternalMemoryProperties hk_opaque_fd_mem_props;
extern const VkExternalMemoryProperties hk_dma_buf_mem_props;

uint64_t hk_memory_index_address(struct hk_device *dev,
                                struct hk_device_memory *mem);
void hk_memory_free_index_va(struct hk_device *dev,
                             struct hk_device_memory *mem);
