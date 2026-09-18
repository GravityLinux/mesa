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
 * A main program and its client package form one Apple9 ABI. Buffer-only
 * compute may publish up to eighteen compact resource pointers directly.
 * Textured or larger compute programs publish six roots, with resource
 * pointers in an indirect table shared by the preamble and main program.
 */
enum agx_apple9_compute_abi {
   AGX_APPLE9_COMPUTE_ABI_INVALID = 0,
   AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS,
   AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES,
};

/* Validated direct descriptor selection capacity, independent of API slots. */
#define AGX_APPLE9_GRAPHICS_MAX_TEXTURES 16
/* Sixteen API samplers plus the private nearest texel-fetch sampler. */
#define AGX_APPLE9_GRAPHICS_MAX_SAMPLERS 17

/* Graphics and descriptor-table compute publish six pointer roots. Direct
 * compute publishes the group-count pointer then compact resource pointers.
 * Root publication owns
 * complete four-word blocks in the argument window. Preamble results start
 * after that padded window and extend through the last uniform word. */
#define AGX_APPLE9_GRAPHICS_ROOT_WORDS 12
#define AGX_APPLE9_MAX_PREAMBLE_BYTES 8192

#define AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES 18
#define AGX_APPLE9_COMPUTE_MAX_RESOURCES 32
/* Descriptor ABI roots: texture, sampler, buffer, group counts, shared, reserved. */
#define AGX_APPLE9_COMPUTE_GROUPS_ROOT 3
#define AGX_APPLE9_COMPUTE_SHARED_ARGUMENT 4
/* Conservative per-invocation limit exercised on T8132 in all three stages. */
#define AGX_APPLE9_MAX_SCRATCH_BYTES 32768
#define AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE 1
/* The threadgroup address space is selected by a tagged argument root. */
#define AGX_APPLE9_COMPUTE_SHARED_ROOT UINT64_C(0x80000000)

static inline unsigned
agx_apple9_compute_root_words(unsigned resource_count)
{
   return 2 * (AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + resource_count);
}

/* An odd pointer count leaves two words in the final root publication block.
 * T8132 root setup can overwrite preamble values in that partial block.
 * Reserve the complete block while keeping the resource table compact. */
static inline unsigned
agx_apple9_compute_preamble_base(unsigned resource_count)
{
   return (agx_apple9_compute_root_words(resource_count) + 3) & ~3u;
}

enum agx_apple9_compute_resource_kind {
   AGX_APPLE9_COMPUTE_RESOURCE_SSBO = 0,
   AGX_APPLE9_COMPUTE_RESOURCE_UBO,
   AGX_APPLE9_COMPUTE_RESOURCE_SHARED,
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
   bool writes_global;

   /* Shader-local dispatch and linear invocation-index contract. */
   bool variable_local_size;
   uint32_t local_size[3];

   uint32_t required_threadgroup_memory_bytes;
   uint32_t scratch_size;
   uint32_t preamble_offset, preamble_size;
   /* Storage for returned 32-bit atomics, independently of RA spills. */
   uint16_t atomic_frame_size;
   uint16_t publication_count;

};

#define AGX_APPLE9_DIRECT_BUFFERS_COMPUTE_PROFILE                            \
   ((struct agx_apple9_compute_profile){                                     \
      .abi = AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS,                         \
      .local_size = {16, 1, 1},                                             \
   })

#endif
