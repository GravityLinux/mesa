/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_NATIVE_MESH_STATE_H
#define AGX_APPLE9_NATIVE_MESH_STATE_H

#include "asahi/compiler/agx_apple9_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_APPLE9_MESH_STATE_WORDS 14

/* GPU DATA only. Launch handles are USC-relative offsets in 64-byte units.
 * The runtime separately validates its compiler and kernel contract. */
struct agx_apple9_mesh_state_parameters {
   const struct agx_apple9_mesh_layout *layout;
   uint32_t object_launch, mesh_launch;
   unsigned threads;
   /* Hardware object stage presence, including a taskless grid shim.
    * This is independent of application task-shader/payload presence. */
   bool has_object_stage;
   /* Application task payload and per-primitive outputs are different
    * interfaces. Neither has a qualified native state contract yet. */
   uint32_t task_payload_size_B;
   struct {
      uint32_t components, stride_B;
   } primitive_output;
};

bool
agx_apple9_mesh_state_build(uint32_t out[AGX_APPLE9_MESH_STATE_WORDS],
                            const struct agx_apple9_mesh_state_parameters *p);

/* Ten pointer-sized argument slots. The first six keep the graphics ABI;
 * local output/grid roots are explicit values, not pointers or GPR numbers.
 * The launch builder publishes the padded twenty-word window unchanged. */
struct agx_apple9_mesh_arguments {
   uint64_t graphics[AGX_APPLE9_GRAPHICS_ROOT_WORDS / 2];
   uint64_t output_root, grid_root, payload, draw;
};
bool agx_apple9_mesh_arguments_build(
   uint64_t out[AGX_APPLE9_MESH_ARGUMENT_WORDS / 2],
   const struct agx_apple9_mesh_arguments *p);

#ifdef __cplusplus
}
#endif
#endif
