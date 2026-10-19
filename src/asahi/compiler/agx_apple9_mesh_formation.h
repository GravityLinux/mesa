/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "compiler/nir/nir_builder.h"
#include "agx_apple9_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bounded ordinary DATA fields; evidence and unknown-bit qualification are in
 * the native-mesh developer notes. OBS_* are measured invariants, not inferred
 * hardware semantics. Never extend their class solely from these names. */
enum {
   AGX_MESH_OBS_HEADER_FLAGS = 0x01000000,
   AGX_MESH_OBS_SHORT_LIST_TAG = 0x2026,
   AGX_MESH_OBS_SHORT_LIST_TAIL = 0x00a80001,
   AGX_MESH_OBS_SCENE_ROOT_FLAGS = 0x80000000,
   AGX_MESH_OBS_COMPACT_REFERENCE_TAG = 0x40,
   AGX_MESH_SHORT_LIST_PRIMITIVE_MASK_SHIFT = 8,
   AGX_MESH_ROOT_VALID_MASK_SHIFT = 27, /* Bit 123 of the 128-bit triple. */
   AGX_MESH_SCENE_ROOT_COUNT_SHIFT = 8,
   AGX_MESH_XY_FIXED_POINT_BIAS = 0x100000,
   AGX_MESH_DEPTH_FACE_LOW_MASK = (1u << 18) - 1,
};

/* Two references: one geometry allocation and one primitive-list allocation.
 * The general measured header count is n + floor(n / 3). */
#define AGX_MESH_SHORT_ROOT_REFERENCE_COUNT 2u
#define AGX_MESH_SHORT_ROOT_COUNT \
   (AGX_MESH_SHORT_ROOT_REFERENCE_COUNT + AGX_MESH_SHORT_ROOT_REFERENCE_COUNT / 3)
#define AGX_MESH_SHORT_ROOT_VALID_MASK \
   ((1u << AGX_MESH_SHORT_ROOT_REFERENCE_COUNT) - 1)

/* Semantic operands of ordinary record formation. These are compiler IR,
 * not a hardware argument ABI. A runtime adapter must resolve them from the
 * current reservation and draw state before machine code can be emitted. */
enum agx_apple9_mesh_formation_input {
   AGX_MESH_GEOMETRY_ADDRESS,
   AGX_MESH_LIST_ADDRESS,
   AGX_MESH_ROOT_ADDRESS,
   AGX_MESH_SCENE_ADDRESS,
   AGX_MESH_GEOMETRY_ID,
   AGX_MESH_LIST_ID,
   AGX_MESH_ROOT_ID,
   AGX_MESH_FRAGMENT_LAUNCH,
   AGX_MESH_DEPTH_CONTROL,
   AGX_MESH_DEPTH_FACE,
   AGX_MESH_DEPTH_FACE_2,
   AGX_MESH_STENCIL,
   AGX_MESH_SCISSOR_BIAS,
   AGX_MESH_TILE_CONTROL,
   AGX_MESH_FRONT_CCW,
   AGX_MESH_VIEWPORT_SCALE_X,
   AGX_MESH_VIEWPORT_SCALE_Y,
   AGX_MESH_VIEWPORT_SCALE_Z,
   AGX_MESH_VIEWPORT_OFFSET_X,
   AGX_MESH_VIEWPORT_OFFSET_Y,
   AGX_MESH_VIEWPORT_OFFSET_Z,
   AGX_MESH_RESERVATION_READY,
};

/* Development draw-data ABI: one little-endian 64-bit slot per semantic
 * above, in enum order. Non-address fields occupy the low 32 bits. The driver
 * owns and retains this read-only allocation through submission completion.
 * Its pointer is in argument uniform words AGX_APPLE9_MESH_DRAW_UNIFORM..+1.
 * Destinations are current reservations; no GPU addresses or IDs are baked
 * into the program. This ABI is private and does not alter the kernel UAPI. */
struct agx_apple9_mesh_draw {
   uint64_t input[AGX_MESH_RESERVATION_READY + 1];
};

struct agx_apple9_mesh_record_state {
   nir_def *fragment_launch, *depth_control, *depth_face, *depth_face_2;
   nir_def *stencil, *scissor_bias, *tile_control, *front_ccw;
};

struct agx_apple9_mesh_compaction {
   nir_def *id[3], *count;
};

/* Ten byte channels: three each for biased fixed-point X/Y, four for Z.
 * This finite encoder currently accepts exactly three compact references. */
struct agx_apple9_mesh_position_encoding {
   nir_def *modes[2], *words[8], *bits, *bytes;
};

struct agx_apple9_mesh_compaction
agx_nir_apple9_mesh_compact_triangle(nir_builder *b, nir_def *index[3]);

nir_def *agx_nir_apple9_mesh_triangle_tokens(
   nir_builder *b, const struct agx_apple9_mesh_compaction *compact,
   nir_def *backface);

struct agx_apple9_mesh_position_encoding
agx_nir_apple9_mesh_encode_positions(nir_builder *b, nir_def *position[3][3]);

/* Returns the byte offset of compact indices after the 44/48/52-byte header.
 * The known optional-word grammar is controlled by depth_control. */
nir_def *agx_nir_apple9_mesh_write_header(
   nir_builder *b, nir_def *address,
   const struct agx_apple9_mesh_record_state *state,
   const struct agx_apple9_mesh_position_encoding *positions,
   nir_def *compact_count);
void agx_nir_apple9_mesh_write_short_list(nir_builder *b, nir_def *address,
                                          nir_def *geometry_id);
void agx_nir_apple9_mesh_write_short_root(nir_builder *b, nir_def *address,
                                          nir_def *root_id,
                                          nir_def *geometry_id,
                                          nir_def *list_id);

/* Lower the understood Position-only, one-triangle family from normal NIR.
 * Includes ordinary body lowering/masking and collective rendezvous. Runtime
 * destinations/state remain semantic loads; no reservation/publication
 * mechanism is invented. Failure is a compile gate, not clipping behavior. */
bool agx_nir_lower_apple9_mesh_formation(
   nir_shader *nir, const struct agx_apple9_mesh_key *key,
   struct agx_apple9_mesh_body_info *info,
   const char **reason);

#ifdef __cplusplus
}
#endif
