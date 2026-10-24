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

/* Qualified native completion width, distinct from the 32-lane subgroup.
 * API invocations occupy its first 1..16 lanes. Wider launches need separate
 * hardware qualification of completion and object call participation. */
#define AGX_APPLE9_NATIVE_THREADS 16
/* Logical workgroups above T16 serialize at most four 32-lane subgroups.
 * This compiler model does not qualify physical multi-subgroup execution. */
#define AGX_APPLE9_NATIVE_SERIAL_THREADS 32
#define AGX_APPLE9_NATIVE_MAX_INVOCATIONS 128

bool agx_apple9_native_workgroup_supported(const nir_shader *nir,
                                          const char **reason);

/* Output declaration class shared by early driver IO lowering and the native
 * compiler. Other eligibility checks still depend on the body and shader key. */
bool agx_apple9_native_mesh_shape_eligible(const nir_shader *nir);

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
   /* User shared storage follows the output allocation on the same local
    * root. Keep output rows separate: mesh state must not count user data as
    * vertex planes. The allocation includes word padding for narrow accesses.
    * Launch integration must qualify this combined allocation independently. */
   unsigned shared_offset_B, shared_size_B, local_allocation_B;
   /* Declared API dimensions, before flattening to the physical launch. */
   uint16_t logical_size[3];
   /* Unresolved pieces of the native contract. Completion lowering clears
    * its own requirement; launch integration must honor the remaining ones. */
   bool requires_native_entry;
   bool requires_object_completion;
   bool requires_mesh_completion;
   bool uses_vertex_count;
   /* V3/P1 primitive outputs lowered to flat planes or a local cull flag. */
   uint64_t primitive_broadcast;
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
 * offsets; shared IO must also be explicit. Dynamic vertex/primitive indices
 * and component write masks work. Mesh shared IO is rebased after outputs;
 * task shared IO remains unsupported. */
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
/* Expand collective grid calls after NIR optimization and validate closed
 * regions before selection. Returns success, including when no calls exist. */
bool agx_nir_lower_apple9_object_calls(struct nir_shader *nir, const char **reason);

bool agx_nir_apple9_object_grid_export(struct nir_builder *b,
                                       const struct agx_apple9_mesh_key *key);

/* Driver-generated ordinary T1 task NIR. Its draw argument points to three
 * uint32_t mesh group counts passed to launch_mesh_workgroups. Compile through
 * native_object to mask the body and append the physical T16 collective. */
nir_shader *agx_nir_apple9_taskless_object(
   const struct agx_apple9_mesh_key *key);

/* Compile the taskless producer and a token-free collective grid call in one
 * allocation unit. The measured width-16/convention-1/POP2 contract remains
 * provisional pending portability and GPR-preservation results. body_only
 * stays set: native entry, frame backing and reservation/submission are still
 * driver promotion gates. The disabled gate neither mutates NIR nor output. */
bool agx_compile_apple9_taskless_object(
   bool enable_native_mesh, const struct agx_apple9_mesh_key *key,
   struct agx_shader_part *out, const char **reason);

/* Bounded ordinary task NIR: one top-level grid launch, no application shared
 * memory or payload. Mask application effects, then reconverge the physical
 * workgroup for the collective call. Both entrypoints retain body_only. */
bool agx_nir_lower_apple9_native_object(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_mesh_key *key,
   struct agx_apple9_mesh_body_info *info, const char **reason);
bool agx_compile_apple9_native_object(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_mesh_key *key, struct agx_shader_part *out,
   struct agx_apple9_mesh_body_info *info, const char **reason);

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

/* Fragment interpolation uses the same grouped scalar order as VS output.
 * NULL selects the position-only, zero-to-one-depth contract. */
struct agx_apple9_native_mesh_options {
   /* Zero uses the qualified T16 launch. T32 executes up to 128 logical
    * invocations as serialized subgroups; the driver must qualify that launch. */
   unsigned physical_threads;
   bool clip_halfz;
   bool attributes;
   /* Exact single-primitive lowering; no native primitive-record ABI. */
   bool primitive_broadcast;
   /* Provisional packed-index publication. Compiler-only opt-in until the
    * narrow declaration class and launch path receive device qualification. */
   bool packed_index_publication;
   uint64_t outputs_flat, outputs_linear;
};

/* Candidate number of publishing lanes for packed 8-bit topology indices.
 * Caller proves ceil(K * maximum_P / 4) fits the physical completion width.
 * The result is a prefix length, not a per-lane source offset. */
struct nir_builder;
nir_def *agx_nir_apple9_index_publication_lanes(struct nir_builder *b,
                                              nir_def *primitives, unsigned k);

/* Compiler-only native development gate, disabled unless explicitly enabled.
 * This accepts API sizes 1..16 at physical T16, or 1..128 at explicit T32,
 * and V3/P1. It emits the
 * complete mesh body and hardware output submissions in one compile.
 * clip_halfz converts final Z from [-W,W] to [0,W] before submission.
 * The result still requires a native launch/object contract; generic shader
 * upload rejects apple9_body_only. This does not enable a Vulkan feature. */
bool agx_nir_lower_apple9_native_mesh(nir_shader *nir, bool enable_native_mesh,
                                      const struct agx_apple9_native_mesh_options *options,
                                      struct agx_apple9_mesh_body_info *info,
                                      const char **reason);
bool agx_compile_apple9_native_mesh(nir_shader *nir, bool enable_native_mesh,
                                    const struct agx_apple9_native_mesh_options *options,
                                    struct agx_shader_part *out,
                                    struct agx_apple9_mesh_body_info *info,
                                    const char **reason);

#ifdef __cplusplus
}
#endif
#endif
