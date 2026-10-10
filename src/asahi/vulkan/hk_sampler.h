/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_private.h"

#include "vk_sampler.h"
#include "vk_ycbcr_conversion.h"

#include "vk_format.h"

struct hk_sampler {
   struct vk_sampler vk;
   VkClearColorValue custom_border;
   bool has_border;

   /* G16 samples custom border colours natively: the sampler record's custom
    * border mode reads this colour, packed for the sampler's border format,
    * from the 16 bytes after the record. */
   uint32_t apple9_border[4];

   uint8_t plane_count;
   uint16_t lod_bias_fp16;
   uint16_t txl_nudge_fp16;

   struct {
      struct hk_rc_sampler *hw;
   } planes[2];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_sampler, vk.base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)
