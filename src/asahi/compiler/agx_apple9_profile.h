/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_APPLE9_PROFILE_H
#define AGX_APPLE9_PROFILE_H

#include <stdbool.h>
#include <stdint.h>
#include "agx_apple9_machine.h"

/*
 * A main program and its client package form one Apple9 ABI.  The current
 * generated entry setup supports up to eighteen visible resources. The
 * compiler compacts API bindings into the argument window after the group-count root.
 */
enum agx_apple9_compute_abi {
   AGX_APPLE9_COMPUTE_ABI_INVALID = 0,
   AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS,
};

/* Validated direct descriptor selection capacity, independent of API slots. */
#define AGX_APPLE9_GRAPHICS_MAX_TEXTURES 16
/* Sixteen API samplers plus the private nearest texel-fetch sampler. */
#define AGX_APPLE9_GRAPHICS_MAX_SAMPLERS 17

/* Graphics publishes six pointer roots. Compute publishes the group-count
 * pointer followed by its compacted resource pointers. Preamble results start
 * immediately after these roots and extend through the last uniform word. */
#define AGX_APPLE9_GRAPHICS_ROOT_WORDS 12
#define AGX_APPLE9_MAX_PREAMBLE_BYTES 8192

#define AGX_APPLE9_COMPUTE_MAX_RESOURCES 18
/* Conservative per-invocation limit exercised on T8132 in all three stages. */
#define AGX_APPLE9_MAX_SCRATCH_BYTES 4096
#define AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE 1

static inline unsigned
agx_apple9_compute_root_words(unsigned resource_count)
{
   return 2 * (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + resource_count);
}

enum agx_apple9_compute_resource_kind {
   AGX_APPLE9_COMPUTE_RESOURCE_SSBO = 0,
   AGX_APPLE9_COMPUTE_RESOURCE_UBO,
};

struct agx_apple9_compute_profile {
   enum agx_apple9_compute_abi abi;

   /* Mapping from native package arguments to API buffer bindings. */
   uint8_t resource_binding_count;
   uint8_t resource_binding[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
   enum agx_apple9_compute_resource_kind
      resource_kind[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
   /* Read/write ownership masks in native package-argument order. */
   uint32_t resource_read_mask;
   uint32_t resource_write_mask;

   /* Shader-local dispatch and linear invocation-index contract. */
   bool variable_local_size;
   uint32_t local_size[3];

   uint32_t required_threadgroup_memory_bytes;
   uint32_t scratch_size;
   uint32_t preamble_offset, preamble_size;
   /* Storage for returned 32-bit atomics, independently of RA spills. */
   uint16_t atomic_frame_size;

};

#define AGX_APPLE9_DIRECT_BUFFERS_COMPUTE_PROFILE                            \
   ((struct agx_apple9_compute_profile){                                     \
      .abi = AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS,                         \
      .local_size = {16, 1, 1},                                             \
   })

#endif
