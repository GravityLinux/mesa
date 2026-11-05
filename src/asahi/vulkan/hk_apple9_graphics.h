/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_APPLE9_GRAPHICS_H
#define HK_APPLE9_GRAPHICS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "vulkan/vulkan_core.h"
#include "util/format/u_formats.h"

#include "asahi/lib/agx_apple9_compute.h"
#include "asahi/compiler/agx_apple9_profile.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/lib/agx_apple9_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Direct resource table slot size; sampler payloads occupy only eight bytes. */
#define AGX_APPLE9_TEXTURE_TABLE_STRIDE 32
#define AGX_APPLE9_SAMPLER_TABLE_STRIDE 32

/* Linear sampled textures encode (row_bytes / 16) - 1 in 16 bits. */
#define AGX_APPLE9_MAX_LINEAR_STRIDE (1u << 20)

/* Plan the exact byte range before narrowing it to the native 32-bit extent.
 * Keep this GPU-free: the caller copies out-of-aperture indices on the GPU.
 */
static inline bool
hk_apple9_index_range(uint64_t base, uint64_t range, uint32_t first,
                      uint32_t count, unsigned size, uint64_t *address,
                      uint32_t *extent, bool *needs_copy)
{
   if ((size != 1 && size != 2 && size != 4) || !count)
      return false;
   uint64_t offset = (uint64_t)first * size;
   uint64_t bytes = (uint64_t)count * size;
   if (offset >= range || bytes > range - offset ||
       bytes > UINT32_MAX - 3 || base > UINT64_MAX - offset ||
       base + offset > UINT64_MAX - bytes)
      return false;
   *address = base + offset;
   *extent = bytes;
   *needs_copy = *address < (UINT64_C(1) << 40) ||
                 *address + bytes > (UINT64_C(1) << 40) + (UINT64_C(1) << 32);
   return true;
}

struct agx_device;
struct agx_bo;
struct agx_pool;
struct agx_apple9_graphics;

struct hk_apple9_render_stage {
   /* Stable compiled-object identity; zero for externally supplied stages. */
   uint64_t program_id;
   /* Owned by the compiled shader; each using batch retains a BO reference. */
   struct agx_bo *bo;
   const uint8_t *binary;
   size_t binary_size;
   uint32_t preamble_offset, preamble_size;
   uint32_t ubo_mask;
   uint32_t scratch_size;
   uint16_t publication_count;
   bool publication_count_valid;
   uint32_t resource_ssbo_mask, resource_write_mask;
   uint8_t resource_count;
   uint8_t resource_binding[AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
   uint32_t texture_mask, sampler_mask, image_mask;
   struct agx_apple9_texture_mapping texture_mapping;
   bool uses_texel_fetch;
   bool uses_discard;
   bool writes_depth;
   bool reads_tile;
   bool disable_tri_merging;

   /* Scalar interface counts used by the bounded Apple9 stage linker. */
   uint8_t position_components;
   uint8_t varying_components;
   struct agx_apple9_varying_layout varyings;
   struct agx_apple9_interp_mask apple9_linear_mask, apple9_flat_mask;
   bool apple9_reads_z;
   bool reads_primitive_id;
   bool writes_point_size;
   bool writes_layer_viewport;
   uint8_t clip_distance_count;
   bool reads_point_coord;
   uint8_t render_targets;
};

struct hk_apple9_render_pipeline {
   /* Vertex fetch is lowered into the API vertex program. */
   struct hk_apple9_render_stage vertex;
   struct hk_apple9_render_stage fragment;

   /* Color surfaces in draw-buffer order; absent attachments have address 0. */
   uint64_t color_targets[8];
   enum pipe_format color_formats[8];
   uint8_t samples;

   /* Batch-owned PPP record, relative to the render-context aperture. */
   uint32_t ppp;

   /* Apple9 VDM linkage words produced by the bounded pipeline linker. */
   uint32_t pipeline_word;
   uint32_t vertex_launch;
   uint32_t vertex_state_class;

   uint64_t index_buffer;
   uint32_t index_extent;
   uint8_t index_size;
   uint8_t primitive;
   bool primitive_restart;
   bool flatshade_first;
   uint32_t restart_index;

};

/* CPU snapshot used while encoding a draw and for adjacent state reuse.
 * Encoded allocations belong to the batch's growing pools. */
struct hk_apple9_uniform_draw {
   uint64_t code[2];
   uint32_t launch[2], ppp, coefficients;
   uint64_t program_id[2];
   uint16_t tile_bytes, cf_count;
   uint8_t samples;
   /* CPU copy for bounded record reuse; never read back write-combined BOs. */
   uint8_t ppp_record[0x100];
   uint64_t buffers[2][AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
   uint8_t buffer_count[2];
   uint64_t vertex_table;
   uint64_t fragment_table;
   /* Graphics stage indices: vertex = 0, fragment = 1. */
   uint64_t texture_table[2], sampler_table[2];
   uint32_t depth_control, depth_face[2], stencil[2];
   uint32_t raster_control;
   uint8_t object_type;
   float viewport_translate[3], viewport_scale[3];
   uint32_t scissor_index;
   uint16_t depth_bias_index;
   uint16_t occlusion_index;
   uint8_t visibility_mode;
   uint16_t scissor_min[2], scissor_max[2];
   bool reads_tile;
   bool uses_discard;
   bool writes_depth;
   bool disable_tri_merging;
   bool flatshade_first;
};

bool hk_apple9_prepare_draw(struct agx_device *dev, struct agx_pool *usc_pool,
                             struct agx_pool *context_pool,
                             const struct hk_apple9_render_pipeline *pipeline,
                             struct hk_apple9_uniform_draw *draw,
                             const struct hk_apple9_uniform_draw *previous);

#define AGX_APPLE9_RENDER_CONTEXT_BASE UINT64_C(0x1000000000)

/*
 * Apple9 command and shader ABIs are intentionally kept outside the older
 * genxml path.  G16 and G17 share the Apple9 core shader ISA with each other;
 * neither is treated as the incompatible Apple8 ISA used by G13/G14.  Their
 * render packet and compiler-container ABIs also need Apple9-specific
 * handling.  This first encoder is deliberately narrow: it describes the
 * direct triangle lists, with optional 16- or 32-bit indices.
 */
bool
hk_apple9_link_render_pipeline(struct hk_apple9_render_pipeline *pipeline,
                                struct hk_apple9_render_stage vertex,
                                struct hk_apple9_render_stage fragment);

bool hk_apple9_link_render_pipeline_with_prolog(
   struct hk_apple9_render_pipeline *pipeline,
   struct hk_apple9_render_stage vertex_prolog,
   struct hk_apple9_render_stage vertex,
   struct hk_apple9_render_stage fragment);

size_t
hk_apple9_direct_draw_size(const struct hk_apple9_render_pipeline *pipeline);

uint8_t *hk_apple9_emit_direct_draw(
   uint8_t *out, const struct hk_apple9_render_pipeline *pipeline,
   unsigned vertex_count, unsigned instance_count, unsigned vertex_start);

struct hk_cmd_buffer;
struct hk_shader;
struct agx_draw;
struct VkRenderingInfo;
struct vk_color_blend_state;

/* Fragment blending is specialized into the API main, as in Gallium. */
bool hk_apple9_blend_state(const struct vk_color_blend_state *cb,
                           VkFormat format, unsigned samples,
                           struct agx_apple9_blend *out);

void hk_apple9_begin_rendering(struct hk_cmd_buffer *cmd,
                               const struct VkRenderingInfo *info);
void hk_apple9_draw(struct hk_cmd_buffer *cmd, struct agx_draw draw);
void hk_apple9_end_rendering(struct hk_cmd_buffer *cmd);
VkResult hk_apple9_graphics_reject(struct hk_cmd_buffer *cmd,
                                  const char *reason);

#ifdef __cplusplus
}
#endif

#endif
