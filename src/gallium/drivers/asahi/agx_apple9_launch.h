/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_LAUNCH_H
#define AGX_APPLE9_LAUNCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum agx_apple9_launch_stage {
   AGX_APPLE9_LAUNCH_VERTEX,
   AGX_APPLE9_LAUNCH_FRAGMENT,
   AGX_APPLE9_LAUNCH_COMPUTE,
};

enum {
   AGX_APPLE9_GRAPHICS_LAUNCH_SIZE = 256,
   AGX_APPLE9_COMPUTE_LAUNCH_SIZE = 1024,
};

struct agx_apple9_launch_parameters {
   uint64_t shader_base;
   uint64_t resource_table;
   uint64_t state;
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
};

/* Current T8132 execution evidence covers these allocation sizes. This does
 * not imply that the compiler implements shared load/store/barrier lowering. */
bool agx_apple9_launch_threadgroup_memory_supported(uint32_t bytes);

/* Offset of the encoded entry field, for diagnostics and frame inspection. */
size_t agx_apple9_launch_call_offset(enum agx_apple9_launch_stage stage,
                                     unsigned resource_count);

/* Build from stage parameters only. No caller-owned executable input is read.
 * On invalid input leave the destination untouched; on success zero all unused
 * bytes through capacity. The fixed allocation sizes above include padding. */
bool agx_apple9_launch_build(uint8_t *out, size_t capacity,
                             enum agx_apple9_launch_stage stage,
                             const struct agx_apple9_launch_parameters *params);

#endif
