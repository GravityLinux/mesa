/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "asahi/lib/agx_tilebuffer.h"
#include "nir.h"
#include "vulkan/vulkan_core.h"

#ifdef __cplusplus
extern "C" {
#endif

bool hk_apple9_tilebuffer(enum pipe_format format, unsigned samples,
                          struct agx_tilebuffer_layout *tib);

void hk_apple9_set_sample_state(struct drm_asahi_cmd_render *cmd,
                                const struct agx_tilebuffer_layout *tib);

/* operation: 0 = clear, 1 = reload, 2 = store. No GPU allocations. */
nir_shader *
hk_apple9_tile_nir(unsigned operation, bool srgb, enum pipe_format format,
                  unsigned samples, const VkClearColorValue *clear,
                  const VkRect2D *clear_area);

#ifdef __cplusplus
}
#endif
