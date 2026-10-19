/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_LAUNCH_H
#define AGX_APPLE9_LAUNCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum agx_apple9_launch_stage {
   AGX_APPLE9_LAUNCH_VERTEX,
   AGX_APPLE9_LAUNCH_FRAGMENT,
   AGX_APPLE9_LAUNCH_COMPUTE,
   /* Native absolute target and local allocation DATA model. Consumer
    * publication/retirement still requires the client's native-contract gate. */
   AGX_APPLE9_LAUNCH_TASK,
   AGX_APPLE9_LAUNCH_MESH,
};

enum {
   AGX_APPLE9_GRAPHICS_LAUNCH_SIZE = 256,
   AGX_APPLE9_COMPUTE_LAUNCH_SIZE = 1024,
};

struct agx_apple9_launch_parameters {
   uint64_t shader_base;
   uint64_t resource_table;
   /* Byte offset of the stage entry within the USC heap. */
   uint32_t entry_offset;
   /* Compiler high-water mark in 32-bit publication slots. */
   uint16_t publication_count;
   uint16_t frame_extent_a;
   uint16_t frame_extent_b;
   uint8_t tile_bytes;
   uint8_t samples;
   uint32_t threadgroup_memory_bytes;
   /* Visible compute buffers, after the group-count pointer. */
   unsigned resource_count;
   /* Optional tail transfer to a compiler-owned setup body ending in STOP.
    * Its BO has the same ownership/lifetime as the main shader. */
   uint64_t preamble_address;
   uint64_t launch_address;
};

/* Current T8132 execution evidence covers these allocation sizes. This does
 * not imply that the compiler implements shared load/store/barrier lowering. */
bool agx_apple9_launch_threadgroup_memory_supported(uint32_t bytes);

/* Offset of the encoded entry field, for diagnostics and frame inspection. */
size_t agx_apple9_launch_call_offset(enum agx_apple9_launch_stage stage,
                                     unsigned resource_count);

/* Build from stage parameters only; no executable input is read.
 * On invalid input leave the destination untouched; on success zero all unused
 * bytes through capacity. The fixed allocation sizes above include padding. */
bool agx_apple9_launch_build(uint8_t *out, size_t capacity,
                             enum agx_apple9_launch_stage stage,
                             const struct agx_apple9_launch_parameters *params);

/* A launch program built once, at pipeline creation, for repeated emission.
 * Only the resource table address, the entry offset and the program's own
 * address (which positions the preamble branch) vary between launches of a
 * pipeline; agx_apple9_launch_template_emit patches exactly those fields.
 * A plain value: it holds no pointers.
 */
struct agx_apple9_launch_template {
   uint8_t program[AGX_APPLE9_COMPUTE_LAUNCH_SIZE];
   uint64_t shader_base;
   uint64_t preamble_address;
   /* Encoded length of the program; emission zero-fills to capacity. */
   uint16_t length;
   uint16_t capacity;
   uint16_t target_at;
   /* Offset of the preamble branch, or 0 without a preamble. */
   uint16_t branch_at;
   enum agx_apple9_launch_stage stage;
   bool valid;
};

/* Build a template from stage parameters. resource_table, entry_offset and
 * launch_address are per-launch and ignored. Returns false (and leaves the
 * template invalid) when agx_apple9_launch_build would reject the stage
 * parameters. */
bool agx_apple9_launch_template_init(struct agx_apple9_launch_template *tmpl,
                                     size_t capacity,
                                     enum agx_apple9_launch_stage stage,
                                     const struct agx_apple9_launch_parameters *params);

/* Emit the program for one launch: the same bytes agx_apple9_launch_build
 * produces with the template's parameters and these per-launch fields. */
bool agx_apple9_launch_template_emit(
   uint8_t *out, const struct agx_apple9_launch_template *tmpl,
   uint64_t resource_table, uint32_t entry_offset, uint64_t launch_address);

#ifdef __cplusplus
}
#endif

#endif
