/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "asahi/lib/agx_apple9_compute.h"
#include "nir.h"

struct hk_cmd_buffer;
struct hk_cs;
struct hk_shader;
struct agx_grid;
struct agx_workgroup;
struct agx_precompiled_shader;
struct vk_descriptor_set_layout;
struct agx_texture_packed;
struct AGX_TEXTURE;

#define HK_APPLE9_ROOT_BINDING 65
#define HK_APPLE9_DATA_BINDING 66
#define HK_APPLE9_BUFFER_SIZES_OFFSET \
   (sizeof(struct hk_root_descriptor_table))

struct hk_apple9_descriptor {
   uint32_t offset;
   uint32_t binding;
   uint32_t element;
   uint16_t dynamic_index;
   uint8_t set;
   uint8_t type;
};

struct hk_apple9_shader_mapping {
   struct hk_apple9_descriptor ubo[64];
   struct hk_apple9_descriptor ssbo[64];
   struct hk_apple9_descriptor textures[32];
   struct hk_apple9_descriptor samplers[32];
   struct hk_apple9_descriptor storage_images[32];
   uint8_t ubo_count, ssbo_count, texture_count, sampler_count;
   uint8_t storage_image_count;
   struct agx_apple9_vertex_layout vertex;
   uint32_t attributes;
   bool has_producer;
   struct agx_apple9_varying_layout producer;
   bool descriptor_snapshot;
};

#ifdef __cplusplus
extern "C" {
#endif

bool hk_apple9_formatless_store_supported(enum pipe_format format);
bool hk_apple9_decode_texture(const struct agx_texture_packed *input,
                             struct AGX_TEXTURE *decoded);

bool hk_nir_lower_apple9_descriptors(
   nir_shader *nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_apple9_shader_mapping *mapping, const char **reason);

bool hk_nir_lower_apple9_textures(
   nir_shader *nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_apple9_shader_mapping *mapping, const char **reason);

bool hk_nir_lower_apple9_io(
   nir_shader *nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_apple9_shader_mapping *mapping, const char **reason);

bool hk_apple9_dispatch_precompiled(struct hk_cmd_buffer *cmd,
                                    struct hk_cs *cs, struct agx_grid grid,
                                    struct agx_precompiled_shader *prog,
                                    const void *data, size_t data_size);

struct hk_apple9_uniform_draw;
bool hk_apple9_graphics_resources(struct hk_cmd_buffer *cmd,
                                  struct hk_shader *shader, unsigned stage,
                                  struct hk_apple9_uniform_draw *draw);

#ifdef __cplusplus
}
#endif
