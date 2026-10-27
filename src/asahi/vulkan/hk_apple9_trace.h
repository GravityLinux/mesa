/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include "vulkan/vulkan_core.h"

enum hk_apple9_trace_category {
   HK_APPLE9_TRACE_RENDER,
   HK_APPLE9_TRACE_SHADER,
   HK_APPLE9_TRACE_DRAW,
   HK_APPLE9_TRACE_META,
   HK_APPLE9_TRACE_SUBMIT,
   HK_APPLE9_TRACE_REJECT,
   HK_APPLE9_TRACE_COUNT,
};

bool hk_apple9_trace(enum hk_apple9_trace_category category);
void hk_apple9_trace_rendering(const void *cmd, const VkRenderingInfo *info,
                               bool resolve);
void hk_apple9_trace_image(const char *role, VkImage image);
