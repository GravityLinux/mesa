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
bool hk_apple9_native_mesh_attributes_enabled(void);
bool hk_apple9_native_mesh_primitive_lowering_enabled(void);

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

/* Zero rejects an invalid logical shape before any multiplication overflows. */
unsigned hk_apple9_native_mesh_threads(const uint16_t local_size[3]);

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
