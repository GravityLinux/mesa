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
#include "agx_apple9_entries.h"

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
   bool writes_point_size;
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

/* Attachment identity is independent of compiled shader identity. */
struct agx_apple9_framebuffer {
   uint64_t targets[8];
   enum pipe_format formats[8];
   uint16_t width, height;
   uint8_t count, samples;
};

struct agx_apple9_graphics *agx_apple9_graphics_create(struct agx_device *dev);
void agx_apple9_graphics_destroy(struct agx_apple9_graphics *graphics);
void agx_apple9_graphics_invalidate(struct agx_apple9_graphics *graphics);
struct agx_bo *agx_apple9_graphics_bo(struct agx_apple9_graphics *graphics);

bool agx_apple9_prepare_draw(struct agx_device *dev, struct agx_pool *usc_pool,
                             struct agx_pool *context_pool,
                             struct agx_apple9_entry_table *entries,
                             const struct agx_apple9_render_pipeline *pipeline,
                             struct agx_apple9_uniform_draw *draw,
                             const struct agx_apple9_uniform_draw *previous);

bool
agx_apple9_graphics_publish(struct agx_apple9_graphics *graphics,
                            const struct agx_apple9_framebuffer *framebuffer,
                            const struct agx_apple9_entry_table *entries,
                            unsigned varying_components,
                            const float clear_color[8][4]);

#define AGX_APPLE9_COMPUTE_PACKAGE_SIZE        0x100000u
#define AGX_APPLE9_COMPUTE_CODE_OFFSET         0x00000u
#define AGX_APPLE9_COMPUTE_CODE_SIZE           0x10000u
#define AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE 0x0340u
#define AGX_APPLE9_COMPUTE_BLOCK_HEADER_SIZE   0x0040u
#define AGX_APPLE9_COMPUTE_MAIN_OFFSET         0x03c0u
#define AGX_APPLE9_COMPUTE_STATE_OFFSET          0x18000u
#define AGX_APPLE9_COMPUTE_LAUNCH_OFFSET         0x90000u
#define AGX_APPLE9_COMPUTE_RESOURCE_OFFSET       0xe0000u
#define AGX_APPLE9_COMPUTE_RESOURCE_TABLE_OFFSET 0x14a0u
#define AGX_APPLE9_COMPUTE_LAUNCH_ALIGN          0x40u
#define AGX_APPLE9_COMPUTE_LAUNCH_REGION_END     0x98000u
#define AGX_APPLE9_COMPUTE_RESOURCE_STRIDE       0x20u
#define AGX_APPLE9_COMPUTE_SUPERSET_RESOURCE_STRIDE 0x100u
#define AGX_APPLE9_COMPUTE_GEOMETRY_GROUPS_OFFSET 0xc0u
#define AGX_APPLE9_COMPUTE_STATE_STRIDE          0x40u
#define AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE       0x2cu
#define AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE 0x28u

#define AGX_APPLE9_RENDER_HEADER_OFFSET         0x01000000u
#define AGX_APPLE9_RENDER_HEADER_APERTURE_SIZE  0x00400000u
#define AGX_APPLE9_RENDER_RESOURCE_OFFSET       0x00200000u
/* Attachment-state views in the fixed USC address space. */
#define AGX_APPLE9_RENDER_FIXED_TARGET_GRAPH_OFFSET  0x00160000u
#define AGX_APPLE9_RENDER_TARGET_GRAPH_SOURCE_OFFSET 0x00210000u
#define AGX_APPLE9_RENDER_ARCHIVE_HEADER_SIZE \
   AGX_APPLE9_RENDER_ENTRY_HEADER_SIZE
#define AGX_APPLE9_RENDER_BLOCK_HEADER_SIZE          0x0040u
#define AGX_APPLE9_RENDER_CONSTANT_RESERVED_SIZE     0x0040u
#define AGX_APPLE9_RENDER_FIRST_MAIN_OFFSET          0x03c0u
#define AGX_APPLE9_RENDER_CONTEXT_BASE               UINT64_C(0x1000000000)
#define AGX_APPLE9_RENDER_STATE_ADDRESS              UINT64_C(0x1000004000)
#define AGX_APPLE9_RENDER_STATE_SIZE                0x00068000u
/* The fixed USC table contains a helper directory and small stage entries.
 * Each entry transfers to an independently allocated compiler-generated body.
 * Headers retain the hardware archive grammar, without packing bodies here. */

bool agx_apple9_compute_enabled(const struct agx_device *dev);

/* Batch-local entries transfer to immutable independently allocated bodies. */
#define AGX_APPLE9_COMPUTE_MAX_ENTRIES                                         \
   ((AGX_APPLE9_COMPUTE_CODE_SIZE - AGX_APPLE9_COMPUTE_ARCHIVE_HEADER_SIZE) /  \
    0xc0u)
bool agx_apple9_build_compute_entry(
   void *mapping, unsigned dispatch, uint64_t shader_base, uint64_t body,
   const struct agx_apple9_compute_profile *profile, uint32_t *entry_offset);

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

uint32_t agx_apple9_compute_archive_call_offset(
   const struct agx_apple9_compute_profile *profile);

/* Whether this exact package ABI consumes a Dynamic-Caching state record.
 * This is an ABI property, not something inferred from the literal count. */
bool agx_apple9_compute_has_dynamic_state(
   const struct agx_apple9_compute_profile *profile);

/* Uniform interface selected by the capture-backed constant/launch pair.
 * Capacity is the number of caller-owned state words actually published,
 * not the storage remaining in the 0x40-byte state record. */
unsigned agx_apple9_compute_state_uniform_base(
   const struct agx_apple9_compute_profile *profile);

unsigned agx_apple9_compute_state_literal_capacity(
   const struct agx_apple9_compute_profile *profile);

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

/* Pure layout preflight used to roll a full batch before mutating it. */
bool agx_apple9_compute_dispatch_fits(
   size_t mapping_size, uint32_t launch_offset, uint32_t state_offset,
   uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile);

/* Persistent-state form used by Gallium.  State is pipeline-owned and only
 * launch/resource records consume space in the transient batch package. */
bool agx_apple9_compute_dispatch_fits_persistent(
   size_t mapping_size, uint32_t launch_offset, uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile);

/* Build one immutable Dynamic Caching state image transactionally. */
bool agx_apple9_build_compute_state(
   void *mapping, size_t mapping_size,
   const struct agx_apple9_compute_profile *profile);

/* A Dynamic Caching selector names the +0x20 payload half of an aligned
 * 0x40-byte state record inside the compact 512-MiB USC window. */
bool agx_apple9_compute_state_address_supported(uint64_t usc_exec_base,
                                                uint64_t state_address);

/*
 * Direct dispatches provide total threads and local size for CPU conversion
 * to group counts. Indirect dispatches provide the GPU group-count pointer.
 * Keep the alternatives tagged to avoid reading the wrong union member.
 */
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

/* Build dispatch state and launch records referring to an existing entry. */
bool agx_apple9_build_compute_dispatch(
   void *mapping, size_t mapping_size, uint64_t usc_exec_base,
   uint64_t package_base, uint32_t main_offset, uint32_t launch_offset,
   uint32_t state_offset, uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count,
   const struct agx_apple9_compute_geometry *geometry);

bool agx_apple9_build_compute_dispatch_persistent(
   void *mapping, size_t mapping_size, uint64_t usc_exec_base,
   uint64_t package_base, uint32_t main_offset, uint32_t launch_offset,
   uint64_t state_address, uint32_t resource_table_offset,
   const struct agx_apple9_compute_profile *profile, const uint64_t *resources,
   unsigned resource_count,
   const struct agx_apple9_compute_geometry *geometry);

bool agx_apple9_emit_direct_dispatch(
   void *out, uint64_t launch, const uint32_t global[3],
   const uint32_t local[3], const struct agx_apple9_compute_profile *profile);

bool agx_apple9_emit_indirect_dispatch(
   void *out, uint64_t launch, uint64_t indirect, const uint32_t local[3],
   const struct agx_apple9_compute_profile *profile);

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
bool agx_apple9_direct_render_enabled(const struct agx_device *dev);

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

#ifdef __cplusplus
}
#endif

#endif
