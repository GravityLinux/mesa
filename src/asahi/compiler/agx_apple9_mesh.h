/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_MESH_H
#define AGX_APPLE9_MESH_H

#include "compiler/nir/nir.h"
#include "agx_compile_apple9.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Native stage-body ABI. These are argument selectors, not physical GPRs or
 * values taken from a capture. The launch builder must populate the selected
 * local root and the two consecutive payload-pointer uniform words. */
/* The driver and compiler share AGX_APPLE9_MESH_KEY. */
struct agx_apple9_mesh_key {
   /* Entry-owned uniform words which stage pointers may not overlap. */
   unsigned reserved_uniform_words;
   unsigned output_root;
   /* Width consumed by the stage's primitive-count handoff. */
   unsigned primitive_count_bits;
   unsigned payload_uniform;
   /* The object root points at this workgroup's grid record. The collective
    * export helper must consume the same root/offset after body completion. */
   unsigned grid_root;
   unsigned grid_offset_B;
   /* Physical completion width, or zero for the API workgroup width. Extra
    * invocations skip the user body and reconverge before native completion.
    * This is independent of the output allocation family's capacity. */
   unsigned physical_threads;
};

#define AGX_APPLE9_MESH_KEY                                                    \
   {                                                                           \
      .reserved_uniform_words = AGX_APPLE9_GRAPHICS_ROOT_WORDS,                \
      .output_root = AGX_APPLE9_MESH_OUTPUT_ROOT,                              \
      .primitive_count_bits = 16,                                              \
      .payload_uniform = AGX_APPLE9_MESH_PAYLOAD_UNIFORM,                      \
      .grid_root = AGX_APPLE9_MESH_GRID_ROOT,                                  \
   }

#define AGX_APPLE9_MESH_UNUSED UINT16_MAX

/* 32-bit component maps shared with the future launch and fragment linkers.
 * Local vertex outputs are component-major; primitive attributes are device
 * records. Indices and culling have distinct byte-addressed local regions. */
struct agx_apple9_mesh_layout {
   uint16_t vertex_component[VARYING_SLOT_MAX][4];
   uint16_t primitive_component[VARYING_SLOT_MAX][4];
   unsigned max_vertices, max_primitives;
   unsigned vertex_components, primitive_components;
   unsigned primitive_count_B, vertex_count_B, cull_B, position_B;
   unsigned vertex_pitch_B, primitive_stride_B, local_size_B;
   unsigned indices_per_primitive;
   bool culling;
};

struct agx_apple9_mesh_body_info {
   struct agx_apple9_mesh_layout layout;
   struct agx_apple9_mesh_key key;
   /* Unresolved pieces of the native contract. Completion lowering clears
    * its own requirement; launch integration must honor the remaining ones. */
   bool requires_native_entry;
   bool requires_object_completion;
   bool requires_mesh_completion;
   bool uses_vertex_count;
   unsigned argument_uniform_words;
};

static inline unsigned
agx_apple9_mesh_argument_words(const struct agx_apple9_mesh_key *key)
{
   return ALIGN_POT(MAX3(AGX_APPLE9_GRAPHICS_ROOT_WORDS,
                         2 * (MAX2(key->output_root, key->grid_root) + 1),
                         key->payload_uniform + 2),
                    4);
}

/* Validate supported normal mesh/task NIR and lower only the understood IO.
 * Returns success, not progress. Rejects unsupported IO before changing NIR.
 * IO must already use per-vertex/per-primitive intrinsics and explicit payload
 * offsets; dynamic vertex/primitive indices and component write masks work. */
bool agx_nir_lower_apple9_mesh(nir_shader *nir,
                               const struct agx_apple9_mesh_key *key,
                               struct agx_apple9_mesh_body_info *info,
                               const char **reason);

/* Preserve logical workgroup system values while masking extra completion
 * invocations between converged workgroup barriers. Call after IO/return
 * lowering and before ordinary compilation. Returns success, not progress.
 * Barriers nested in application branches/loops take the flag protocol in
 * the output count header, which needs the body layout. */
bool agx_nir_mask_apple9_mesh_body(nir_shader *nir,
                                   const struct agx_apple9_mesh_key *key,
                                   const struct agx_apple9_mesh_layout *layout,
                                   const char **reason);

/* Emit the understood dataflow of the generated object completion helper:
 * an output fence, then one local grid-word export per invocation 0/1/2.
 * The caller must provide a collective helper context. This is not a valid
 * inline task epilogue: call/frame/entry setup remains unresolved. */
struct nir_builder;
bool agx_nir_apple9_object_grid_export(struct nir_builder *b,
                                       const struct agx_apple9_mesh_key *key);

/* Compile the lowered body through ordinary instruction selection and RA.
 * The returned binary is deliberately body-only (see info), not a production
 * native launch path. Native compilation is separate from the compute-emulated
 * Vulkan backend. */
bool agx_compile_apple9_mesh_body(nir_shader *nir,
                                  const struct agx_apple9_mesh_key *key,
                                  struct agx_shader_part *out,
                                  struct agx_apple9_mesh_body_info *info,
                                  const char **reason);

/* Compile already lowered native IO plus caller-owned generated operations.
 * This retains the body-only upload gate; an executable client must supply
 * and audit its own entry/completion/launch contract before clearing it. */
bool agx_compile_apple9_mesh_lowered(nir_shader *nir,
                                     const struct agx_apple9_mesh_key *key,
                                     struct agx_shader_part *out,
                                     const char **reason);

/* Compiler-only native development gate, disabled unless explicitly enabled.
 * This accepts the qualified T16/Position/V3/P1 class. It emits the complete
 * mesh body, ordinary records, and all mesh submissions in a single compile.
 * The result still requires a native launch/object contract; generic shader
 * upload rejects apple9_body_only. This does not enable a Vulkan feature. */
bool agx_nir_lower_apple9_native_mesh(nir_shader *nir, bool enable_native_mesh,
                                      struct agx_apple9_mesh_body_info *info,
                                      const char **reason);
bool agx_compile_apple9_native_mesh(nir_shader *nir, bool enable_native_mesh,
                                    struct agx_shader_part *out,
                                    struct agx_apple9_mesh_body_info *info,
                                    const char **reason);

#ifdef __cplusplus
}
#endif
#endif
