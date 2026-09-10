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

/* External compatibility material, never compiler instructions. The boundaries
 * are raw, call-relative byte ranges, not inferred instruction boundaries.
 * Views borrow their storage from the external blob owner. */
struct agx_apple9_launch_fragment {
   const uint8_t *data;
   size_t size;
};

struct agx_apple9_launch_recipe {
   enum agx_apple9_launch_stage stage;
   struct agx_apple9_launch_fragment prefix;
   struct agx_apple9_launch_fragment coverage_setup;
   struct agx_apple9_launch_fragment shared;
   struct agx_apple9_launch_fragment suffix_shared;
   struct agx_apple9_launch_fragment suffix;
   size_t size;
   bool generated_resources;
   bool generated_state_load;
   /* Preserved opaque stage differences, supplied by the external library. */
   uint8_t pre_call[2];
   uint8_t shared_tag;
};

struct agx_apple9_launch_parameters {
   uint64_t shader_base;
   uint64_t resource_table;
   uint64_t state;
   /* Already validated by the archive allocator, in its tested 18-bit range. */
   uint32_t main_call;
   uint16_t publication_word;
   uint16_t frame_extent_a;
   uint16_t frame_extent_b;
   uint8_t tile_bytes;
   uint8_t samples;
   /* Count 32-bit publication slots, including holes and temporary operands.
    * External mains without compiler metadata retain publication_word. */
   uint16_t publication_count;
   bool publication_count_valid;
   uint32_t threadgroup_memory_bytes;
   /* Visible compute buffers, after the three dispatch roots. */
   unsigned resource_count;
};

/* Versioned external material: three stage recipes, with optional coverage
 * fragments for the fragment stage. No executable bytes are embedded here. */
struct agx_apple9_launch_library {
   struct agx_apple9_launch_fragment fragments[9];
   const uint8_t *configurations;
   bool generated_resources;
   bool generated_state_load;
   bool generated_graphics_resources;
};

bool agx_apple9_launch_library_open(struct agx_apple9_launch_library *library,
                                    const uint8_t *data, size_t size);

/* Current T8132 execution evidence covers these allocation sizes. This does
 * not imply that the compiler implements shared load/store/barrier lowering. */
bool agx_apple9_launch_threadgroup_memory_supported(uint32_t bytes);
bool agx_apple9_launch_select(const struct agx_apple9_launch_library *library,
                              enum agx_apple9_launch_stage stage, bool coverage,
                              struct agx_apple9_launch_recipe *recipe,
                              struct agx_apple9_launch_parameters *params);

/* Transitional import of existing external carriers. The builder below only
 * consumes fragments and explicit parameters, not carrier names or captures. */
bool agx_apple9_launch_import(struct agx_apple9_launch_recipe *recipe,
                              struct agx_apple9_launch_parameters *defaults,
                              enum agx_apple9_launch_stage stage,
                              const uint8_t *data, size_t size,
                              unsigned call_offset);

/* On invalid inputs, leave the destination untouched. Source and destination
 * storage must not overlap. Bytes beyond the recipe size are zero-filled. */
size_t agx_apple9_launch_call_offset(const struct agx_apple9_launch_recipe *recipe,
                                     unsigned resource_count);

bool agx_apple9_launch_build(uint8_t *out, size_t capacity,
                             const struct agx_apple9_launch_recipe *recipe,
                             const struct agx_apple9_launch_parameters *params);

#endif
