/*
 * Copyright 2026 Asahi Linux contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/* G16 VK_EXT_descriptor_buffer record sizes. Each is a power of two so
 * layers can embed mutable descriptors, and each keeps the native texture and
 * sampler entries table aligned (32 B). Sampled records pad the 224 B
 * hk_apple9_image_descriptor; buffer addresses pad hk_buffer_address so the
 * offset alignment can stay at the image records' 32 B. */
#define HK_APPLE9_DB_SAMPLED_SIZE   256
#define HK_APPLE9_DB_SAMPLER_SIZE   256
#define HK_APPLE9_DB_STORAGE_SIZE   128
#define HK_APPLE9_DB_TEXEL_SIZE     128
#define HK_APPLE9_DB_BUFFER_SIZE    32
#define HK_APPLE9_DB_ALIGNMENT      32

/* Descriptor buffers share the 4 GiB USC aperture with shaders and pools. */
#define HK_APPLE9_DB_RANGE          (1u << 30)
#define HK_APPLE9_DB_ADDRESS_SPACE  (1u << 31)
