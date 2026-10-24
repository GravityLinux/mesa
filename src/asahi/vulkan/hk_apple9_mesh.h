/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "asahi/compiler/agx_apple9_mesh.h"
#include "asahi/lib/agx_apple9.h"
#include "asahi/lib/agx_apple9_launch.h"
#include "asahi/lib/agx_apple9_mesh.h"

/* Explicit userspace/kernel development pairing. The stable UAPI has no
 * native-mesh capability bit; enabling this requires the matching kernel. */
bool hk_apple9_native_mesh_enabled(void);
struct hk_physical_device;
struct hk_device;
/* Default publication policy is separate from the broader research opt-in. */
bool hk_apple9_native_index_enabled(const struct hk_physical_device *pdev);
bool hk_apple9_native_mesh_available(const struct hk_physical_device *pdev);
bool hk_apple9_native_mesh_available_device(struct hk_device *dev);
const char *hk_apple9_native_index_class_hash(void);
uint64_t hk_apple9_native_index_shapes(void);
/* Separate default-off GS route; also requires native mesh and exact classes. */
bool hk_apple9_native_gs_enabled(void);
bool hk_apple9_native_tess_enabled(void);

/* Research (HK_APPLE9_NATIVE_GS_PACK, default off): native GS runs up to
 * HK_APPLE9_NATIVE_GS_PACK input triangles per T32 mesh group, one GS
 * invocation per lane, in fixed per-lane output slots (T32 V24/P17 for 3
 * output vertices, T32 V32/P16 for 4). A packed body is recognised by its
 * T32 class alone. */
#define HK_APPLE9_NATIVE_GS_PACK 8
bool hk_apple9_native_gs_pack_enabled(void);

/* Stage policy is narrower than native-index qualification. These bits name
 * the API producer, not the generated mesh/object execution stages. */
#define HK_APPLE9_NATIVE_GS_BIT (1u << 0)
#define HK_APPLE9_NATIVE_TES_BIT (1u << 1)
#define HK_APPLE9_NATIVE_STAGE_BITS \
   (HK_APPLE9_NATIVE_GS_BIT | HK_APPLE9_NATIVE_TES_BIT)
uint8_t hk_apple9_native_stage_policy(const struct hk_physical_device *pdev);
bool hk_apple9_native_stage_available_device(struct hk_device *dev,
                                             mesa_shader_stage stage);
bool hk_apple9_native_stage_class(mesa_shader_stage stage, unsigned threads,
                                  unsigned vertices, unsigned primitives);

bool hk_apple9_native_mesh_attributes_enabled(void);
bool hk_apple9_native_mesh_primitive_lowering_enabled(void);

/* Explicit research selection, e.g. T16V3P2,T32V32P42, after qualification.
 * Empty, boolean/wildcard or malformed selections enable no new class.
 * The pure parser/predicate are shared with host tests. */
uint64_t hk_apple9_packed_mesh_shapes_parse(const char *selection);
uint64_t hk_apple9_native_mesh_packed_shapes(void);
bool hk_apple9_packed_mesh_class_selected(uint64_t selection,
                                         unsigned threads, unsigned vertices,
                                         unsigned primitives);
bool hk_apple9_native_mesh_packed_shape_eligible(const nir_shader *nir);
bool hk_apple9_native_mesh_workgroup_eligible(const nir_shader *nir);

enum hk_apple9_native_stage {
   HK_APPLE9_NATIVE_NONE,
   HK_APPLE9_NATIVE_OBJECT,
   HK_APPLE9_NATIVE_MESH,
};

/* Stage-local byte offsets. These are not GPU addresses or resource aliases.
 * Object and mesh have separate local allocations. */
#define HK_APPLE9_NATIVE_LOCAL_BYTES 128
#define HK_APPLE9_NATIVE_OUTPUT_ROOT 0
#define HK_APPLE9_NATIVE_GRID_ROOT 64
/* Fits alongside the entire admitted V3/P1 output interface (2080 bytes)
 * in one 32-KiB allocation. Larger output classes need a new storage model. */
#define HK_APPLE9_NATIVE_SHARED_BYTES 28672

/* One taskless object publishes one mesh-child grid. Apple's Metal Feature
 * Set Tables give Apple9 a total of 2^20 - 1 mesh groups per grid. Retain the
 * existing, more conservative Vulkan per-axis bound. These are grid limits,
 * independent of the physical object workgroup size in the command stream.
 * https://developer.apple.com/metal/capabilities/ */
#define HK_APPLE9_NATIVE_MAX_GROUP_COUNT 65535u
#define HK_APPLE9_NATIVE_MAX_GROUP_TOTAL ((1u << 20) - 1)

/* Validate a nonempty direct child grid without overflowing the product.
 * The draw entrypoint handles zero dimensions as a no-op before calling this. */
bool hk_apple9_native_mesh_grid_valid(const uint32_t groups[3]);

/* Zero rejects an invalid logical shape before any multiplication overflows. */
unsigned hk_apple9_native_mesh_threads(const uint16_t local_size[3]);
/* Research envelope beyond one publication round at physical T16/T32;
 * independent of the qualified class list. */
bool hk_apple9_native_mesh_round_counts_eligible(const uint16_t local_size[3],
                                                 unsigned vertices,
                                                 unsigned primitives);
/* Research envelope of the publication model at T32: every T16/T32
 * declaration up to V256/P256 except V3/P1. */
bool hk_apple9_native_mesh_model_counts_eligible(const uint16_t local_size[3],
                                                 unsigned vertices,
                                                 unsigned primitives);
bool hk_apple9_native_mesh_round_shape_eligible(const nir_shader *nir);

bool hk_apple9_native_shader_valid(
   const struct agx_shader_info *code,
   const struct agx_apple9_mesh_body_info *body,
   enum hk_apple9_native_stage stage);
bool hk_apple9_native_launch_parameters(
   struct agx_apple9_launch_parameters *out,
   const struct agx_shader_info *code,
   const struct agx_apple9_mesh_body_info *body,
   enum hk_apple9_native_stage stage, uint64_t shader_base,
   uint64_t code_address);

#define HK_APPLE9_MESH_COMMAND_BYTES 160
bool hk_apple9_mesh_command_build(
   void *out, size_t capacity,
   const struct agx_apple9_render_pipeline *pipeline,
   const uint32_t state[AGX_APPLE9_MESH_STATE_WORDS],
   const uint32_t object_groups[3], const uint32_t object_local_size[3],
   bool enable);
