/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_bo.h"
#include "hk_private.h"
#include "util/simple_mtx.h"

#include "vk_command_pool.h"

#define HK_CMD_BO_SIZE (1024 * 128)

/* Idle command BOs kept by the device across all pools, in bytes. Beyond it,
 * returned BOs go to the BO cache, which ages them out. */
#define HK_CMD_BO_CACHE_BYTES (32 * 1024 * 1024)

enum hk_cmd_pool_kind {
   HK_CMD_POOL_MAIN,
   HK_CMD_POOL_USC,
   HK_CMD_POOL_CONTEXT,
   HK_CMD_POOL_COUNT,
};

/* Recyclable command buffer BO, used for both push buffers and upload */
struct hk_cmd_bo {
   struct agx_bo *bo;

   void *map;

   /** Link in hk_cmd_buffer::bos */
   struct list_head link;

   /** Next idle BO in hk_cmd_bo_cache::free */
   struct hk_cmd_bo *next_free;
};

/* Idle command BOs shared by every pool of a device, most recently returned
 * first. As in ANV's batch BO pool, a pool keeps nothing of its own: memory
 * retained after a spike or by many pools (Zink keeps two per batch state) is
 * bounded by HK_CMD_BO_CACHE_BYTES for the whole device, not per pool.
 * Zero-initialized is empty. */
struct hk_cmd_bo_cache {
   simple_mtx_t lock;
   struct hk_cmd_bo *free[HK_CMD_POOL_COUNT];
   uint64_t bytes;
};

struct hk_cmd_pool {
   struct vk_command_pool vk;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_cmd_pool, vk.base, VkCommandPool,
                               VK_OBJECT_TYPE_COMMAND_POOL)

static inline struct hk_device *
hk_cmd_pool_device(struct hk_cmd_pool *pool)
{
   return (struct hk_device *)pool->vk.base.device;
}

enum agx_bo_flags hk_cmd_pool_bo_flags(const struct hk_device *dev,
                                       enum hk_cmd_pool_kind kind);

VkResult hk_cmd_pool_alloc_bo(struct hk_cmd_pool *pool,
                              enum hk_cmd_pool_kind kind,
                              struct hk_cmd_bo **bo_out);

void hk_cmd_bo_cache_finish(struct hk_device *dev);
void hk_cmd_bo_cache_trim(struct hk_device *dev);

void hk_cmd_pool_free_bo_list(struct hk_cmd_pool *pool,
                              enum hk_cmd_pool_kind kind,
                              struct list_head *bos, bool release);
