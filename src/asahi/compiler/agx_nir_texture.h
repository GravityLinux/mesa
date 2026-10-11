/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

/* Texture backend flags */
#define AGX_TEXTURE_FLAG_NO_CLAMP (1 << 0)

/* Indicates that the sampler should be overriden to clamp to 0 instead of 1 */
#define AGX_TEXTURE_FLAG_CLAMP_TO_0 (1 << 1)

/* A 2D access the driver rewrote as layer 0 of a one-layer array, so the
 * Apple9 texture lowering still treats it as 2D, for unnormalized samplers. */
#define AGX_TEXTURE_FLAG_ARRAYED_2D (1 << 2)

/* A size, level or sample query of a sampled view, answered from the view's
 * texture metadata (Apple9 texture_info) rather than its texture descriptor.
 * agx_nir_lower_texture_queries leaves it to the Apple9 backend. */
#define AGX_TEXTURE_FLAG_METADATA_QUERY (1 << 3)

/* Texel buffers lowered to (at most) 16384x16384 2D textures */
#define AGX_TEXTURE_BUFFER_WIDTH      16384
#define AGX_TEXTURE_BUFFER_MAX_HEIGHT 16384
#define AGX_TEXTURE_BUFFER_MAX_SIZE                                            \
   (AGX_TEXTURE_BUFFER_WIDTH * AGX_TEXTURE_BUFFER_MAX_HEIGHT)
