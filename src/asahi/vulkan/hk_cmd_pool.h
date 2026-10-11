/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_bo.h"
#include "hk_private.h"

#include "vk_command_pool.h"

#define HK_CMD_BO_SIZE     1024 * 128
#define HK_CMD_POOL_BO_MAX 32

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

   /** Link in hk_cmd_pool::free_bos or hk_cmd_buffer::bos */
   struct list_head link;
};

struct hk_cmd_pool {
   struct vk_command_pool vk;

   /** List of hk_cmd_bo */
   struct list_head free_bos[HK_CMD_POOL_COUNT];
   uint32_t num_free_bos[HK_CMD_POOL_COUNT];
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

void hk_cmd_pool_free_bo_list(struct hk_cmd_pool *pool,
                              enum hk_cmd_pool_kind kind,
                              struct list_head *bos, bool release);
