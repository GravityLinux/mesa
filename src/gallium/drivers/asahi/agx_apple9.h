/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_APPLE9_H
#define AGX_APPLE9_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "util/format/u_formats.h"

#include "asahi/lib/agx_apple9_layout.h"
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

bool agx_apple9_texture_format_supported(enum pipe_format format);

struct agx_device;
struct agx_bo;
struct agx_pool;
struct agx_apple9_graphics;

struct agx_apple9_render_stage {
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

struct agx_apple9_render_pipeline {
   /* Vertex fetch is lowered into the API vertex program. */
   struct agx_apple9_render_stage vertex;
   struct agx_apple9_render_stage fragment;

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
struct agx_apple9_uniform_draw {
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

void agx_apple9_initialize_entries(struct agx_device *dev);

bool agx_apple9_prepare_draw(struct agx_device *dev, struct agx_pool *usc_pool,
                             struct agx_pool *context_pool,
                             const struct agx_apple9_render_pipeline *pipeline,
                             struct agx_apple9_uniform_draw *draw,
                             const struct agx_apple9_uniform_draw *previous);

#define AGX_APPLE9_COMPUTE_RESOURCE_STRIDE 0x20u
#define AGX_APPLE9_COMPUTE_SUPERSET_RESOURCE_STRIDE 0x100u
#define AGX_APPLE9_COMPUTE_GEOMETRY_GROUPS_OFFSET 0xc0u
#define AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE 0x2cu
#define AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE 0x28u
#define AGX_APPLE9_COMPUTE_INDIRECT_LOCAL_CDM_RECORD_SIZE 0x1cu
#define AGX_APPLE9_RENDER_CONTEXT_BASE UINT64_C(0x1000000000)

bool agx_apple9_compute_enabled(const struct agx_device *dev);

size_t agx_apple9_compute_launch_size(
   const struct agx_apple9_compute_profile *profile);

unsigned agx_apple9_compute_resource_count(
   const struct agx_apple9_compute_profile *profile);

/* Exact per-dispatch resource-record footprint selected by the package ABI.
 * The direct-buffer ABI uses one 0x100-byte record containing one hidden
 * pointer, up to eighteen visible pointers, and inline group counts. */
size_t agx_apple9_compute_resource_record_size(
   const struct agx_apple9_compute_profile *profile);

/* Exact total threadgroup-memory requirement carried by this profile.  The
 * value is validated against the selected opaque package ABI. */
uint32_t agx_apple9_compute_required_threadgroup_memory_bytes(
   const struct agx_apple9_compute_profile *profile);

/* Map a package argument record to the caller's Gallium buffer namespace and
 * binding.  Compiler-generated profiles may mix UBO inputs with SSBO inputs
 * and outputs without changing native package argument order. */
enum agx_apple9_compute_resource_kind agx_apple9_compute_resource_kind(
   const struct agx_apple9_compute_profile *profile, unsigned argument);

unsigned agx_apple9_compute_resource_binding(
   const struct agx_apple9_compute_profile *profile, unsigned argument);

uint32_t
agx_apple9_compute_read_mask(const struct agx_apple9_compute_profile *profile);

uint32_t
agx_apple9_compute_write_mask(const struct agx_apple9_compute_profile *profile);


/*
 * Direct CDM geometry is dispatch state.  Fixed-local-size profiles require
 * the compiled tuple, while variable-local-size profiles validate the tuple
 * supplied by the dispatch command.
 */
bool agx_apple9_compute_grid_supported(
   const struct agx_apple9_compute_profile *profile, const uint32_t global[3],
   const uint32_t local[3]);

/* Indirect dispatch is a launch/package ABI property. */
bool agx_apple9_compute_indirect_dispatch_supported(
   const struct agx_apple9_compute_profile *profile);

enum agx_apple9_compute_geometry_mode {
   AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT,
   AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT,
};

struct agx_apple9_compute_geometry {
   enum agx_apple9_compute_geometry_mode mode;
   union {
      uint32_t threads[3];
      uint64_t group_counts;
   };
   uint32_t local[3];
};

/* Populate the group-count pointer and inline counts in a resource record. */
bool agx_apple9_build_compute_geometry_fields(
   void *record, size_t record_size, uint64_t record_address,
   const struct agx_apple9_compute_geometry *geometry);

/* Build immutable resource and launch records in the batch's USC pool. */
bool agx_apple9_prepare_compute_dispatch(
   struct agx_device *dev, struct agx_pool *usc_pool, struct agx_bo *body,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count, uint64_t textures, uint64_t samplers,
   const struct agx_apple9_compute_geometry *geometry,
   uint64_t preamble_address, uint64_t *launch_address);

bool agx_apple9_emit_direct_dispatch(
   void *out, uint64_t launch, const uint32_t global[3],
   const uint32_t local[3], const struct agx_apple9_compute_profile *profile);

bool agx_apple9_emit_indirect_dispatch(
   void *out, uint64_t launch, uint64_t indirect, const uint32_t local[3],
   bool indirect_local, const struct agx_apple9_compute_profile *profile);

void agx_apple9_pack_r32f_texture(void *out, uint64_t address, uint32_t width,
                                  uint32_t height, uint32_t stride_B);

bool agx_apple9_sampler_wrap_supported(unsigned wrap);
void agx_apple9_pack_sampler(void *out, bool min_linear, bool mag_linear,
                              unsigned mip_filter, float min_lod, float max_lod,
                         unsigned wrap_s, unsigned wrap_t, unsigned max_anisotropy);
void agx_apple9_pack_nearest_sampler(void *out);

/*
 * Apple9 command and shader ABIs are intentionally kept outside the older
 * genxml path.  G16 and G17 share the Apple9 core shader ISA with each other;
 * neither is treated as the incompatible Apple8 ISA used by G13/G14.  Their
 * render packet and compiler-container ABIs also need Apple9-specific
 * handling.  This first encoder is deliberately narrow: it describes the
 * direct triangle lists, with optional 16- or 32-bit indices.
 */
bool
agx_apple9_link_render_pipeline(struct agx_apple9_render_pipeline *pipeline,
                                struct agx_apple9_render_stage vertex,
                                struct agx_apple9_render_stage fragment);

bool agx_apple9_link_render_pipeline_with_prolog(
   struct agx_apple9_render_pipeline *pipeline,
   struct agx_apple9_render_stage vertex_prolog,
   struct agx_apple9_render_stage vertex,
   struct agx_apple9_render_stage fragment);

size_t
agx_apple9_direct_draw_size(const struct agx_apple9_render_pipeline *pipeline);

uint8_t *agx_apple9_emit_direct_draw(
   uint8_t *out, const struct agx_apple9_render_pipeline *pipeline,
   unsigned vertex_count, unsigned instance_count, unsigned vertex_start);

uint8_t *agx_apple9_emit_indirect_draw(
   uint8_t *out, const struct agx_apple9_render_pipeline *pipeline,
   uint64_t indirect);

#ifdef __cplusplus
}
#endif

#endif
