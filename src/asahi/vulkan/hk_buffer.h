/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hk_device_memory.h"
#include "hk_private.h"

#include "agx_abi.h"
#include "agx_bo.h"
#include "layout.h"
#include "vk_buffer.h"

struct hk_device_memory;
struct hk_physical_device;

/* Buffers placed at a replayable address of their own. */
#define HK_BUFFER_CREATE_CAPTURE_REPLAY_BITS                                   \
   (VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT |                       \
    VK_BUFFER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT)

struct hk_buffer {
   struct vk_buffer vk;

   /** Reserved VA for sparse/capture buffers or native Apple9 index buffers. */
   struct agx_va *va;
   bool apple9_index_va;

   /* Apple9 sparse residency: one byte per page, nonzero if bound. */
   struct agx_bo *apple9_residency;

   /* Virtio resource id of bound implicit-sync (WSI) memory, or 0. */
   uint32_t implicit_sync_res;
};

/* Descriptor residency pointer of a sparse residency buffer, biased so the
 * byte of the page holding address A is at the result + (A >> page shift).
 */
static inline uint64_t
hk_buffer_apple9_residency(const struct hk_buffer *buffer)
{
   if (!buffer->apple9_residency)
      return 0;

   return buffer->apple9_residency->va->addr -
          (buffer->va->addr / AIL_PAGESIZE);
}

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_buffer, vk.base, VkBuffer,
                               VK_OBJECT_TYPE_BUFFER)

uint64_t hk_buffer_address(const struct hk_buffer *buffer, uint64_t offset,
                           bool read_only);

static inline uint64_t
hk_buffer_address_rw(const struct hk_buffer *buffer, uint64_t offset)
{
   return hk_buffer_address(buffer, offset, false);
}

static inline uint64_t
hk_buffer_address_ro(const struct hk_buffer *buffer, uint64_t offset)
{
   return hk_buffer_address(buffer, offset, true);
}

static inline struct hk_addr_range
hk_buffer_addr_range(const struct hk_buffer *buffer, uint64_t offset,
                     uint64_t range, bool read_only)
{
   /* If range == 0, return a pointer to the zero page. That allows
    * eliding robustness2 bounds checks for index = 0, even without soft fault.
    */
   if (buffer == NULL || range == 0) {
      return (struct hk_addr_range){.addr = AGX_ZERO_PAGE_ADDRESS, .range = 0};
   }

   return (struct hk_addr_range){
      .addr = hk_buffer_address(buffer, offset, read_only),
      .range = vk_buffer_range(&buffer->vk, offset, range),
   };
}

VkResult hk_bind_scratch(struct hk_device *dev, struct agx_va *va,
                         unsigned offs_B, size_t size_B);
