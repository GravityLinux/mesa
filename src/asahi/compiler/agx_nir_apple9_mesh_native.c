/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_mesh.h"
#include "compiler/nir/nir_builder.h"

static bool
fail(const char **reason, const char *message)
{
   if (reason)
      *reason = message;
   return false;
}

static bool
eligible(nir_shader *nir, const char **reason)
{
   if (nir->info.stage != MESA_SHADER_MESH ||
       nir->info.shared_size || nir->info.task_payload_size ||
       nir->info.mesh.max_vertices_out != 3 ||
       nir->info.mesh.max_primitives_out != 1 ||
       nir->info.mesh.primitive_type != MESA_PRIM_TRIANGLES)
      return fail(reason,
                  "native mesh development class requires V3 P1 triangles without shared memory or payload");
   if (!agx_apple9_native_workgroup_supported(nir, reason))
      return false;

   unsigned publications = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_set_vertex_and_primitive_count)
               continue;
            if (!nir_src_is_const(intr->src[0]) ||
                !nir_src_is_const(intr->src[1]) ||
                nir_src_as_uint(intr->src[0]) != 3 ||
                nir_src_as_uint(intr->src[1]) != 1)
               return fail(
                  reason,
                  "native mesh development class requires SetMeshOutputs(3, 1)");
            ++publications;
         }
      }
   }
   return publications == 1 ||
          fail(reason,
               "native mesh development class requires one count publication");
}

/* Keep the actual vertex count in compiler-owned local storage. The body
 * lowerer leaves the count marker so completion can consume both operands. */
static bool
consume_count(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_mesh_body_info *info = data;
   if (intr->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      nir_store_local_apple9_agx(b, intr->src[0].ssa,
                                 nir_imm_int(b, info->layout.vertex_count_B),
                                 .base = info->key.output_root, .bit_size = 32);
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }
   /* Point size does not participate in triangle rasterization. */
   if (intr->intrinsic == nir_intrinsic_store_per_vertex_output &&
       nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PSIZ) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   return false;
}

static void
device_fence(nir_builder *b)
{
   nir_barrier(b, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

/* Local planes stay in semantic/component order for shader output reads.
 * U channels use the fragment linker's interpolation groups instead. */
static bool
varying_layout(const struct agx_apple9_mesh_layout *layout,
               const struct agx_apple9_native_mesh_options *options,
               struct agx_apple9_varying_layout *varyings,
               const char **reason)
{
   memset(varyings, 0, sizeof(*varyings));
   if (layout->primitive_components || layout->culling)
      return fail(reason,
                  "native mesh primitive attributes/culling are not implemented");
   if (layout->vertex_components > 4 && (!options || !options->attributes))
      return fail(reason, "native mesh attributes development path is disabled");
   if (layout->vertex_components > 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS)
      return fail(reason, "native mesh exceeds 124 user varying scalars");
   for (unsigned slot = VARYING_SLOT_VAR0; slot < VARYING_SLOT_MAX; ++slot) {
      for (unsigned c = 0; c < 4; ++c) {
         if (layout->vertex_component[slot][c] == AGX_APPLE9_MESH_UNUSED)
            continue;
         if (slot >= ARRAY_SIZE(varyings->mask) ||
             !agx_apple9_varying_supported(slot))
            return fail(reason, "native mesh varying location is unsupported");
         varyings->mask[slot] |= BITFIELD_BIT(c);
         ++varyings->count;
      }
      if (slot < ARRAY_SIZE(varyings->mask) && varyings->mask[slot])
         varyings->group[slot] = agx_apple9_varying_group(
            options->outputs_flat, options->outputs_linear, slot);
   }
   return true;
}

bool
agx_nir_lower_apple9_native_mesh(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_native_mesh_options *options,
   struct agx_apple9_mesh_body_info *info, const char **reason)
{
   if (!enable_native_mesh)
      return fail(reason, "native mesh development path is disabled");
   if (!eligible(nir, reason))
      return false;

   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   key.physical_threads = AGX_APPLE9_NATIVE_THREADS;
   if (!agx_nir_lower_apple9_mesh(nir, &key, info, reason))
      return false;
   const struct agx_apple9_mesh_layout *layout = &info->layout;
   struct agx_apple9_varying_layout varyings;
   if (!varying_layout(layout, options, &varyings, reason))
      return false;
   nir_shader_intrinsics_pass(nir, consume_count, nir_metadata_none, info);
   nir_lower_returns(nir);
   if (!agx_nir_mask_apple9_mesh_body(nir, &key, layout, reason))
      return false;
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   /* Counts are the
    * actual shader publication, not maxima or register-descriptor constants.
    * U0 consumes and clears its A halfword; B's wider meaning is unresolved.
    * The qualified one-triangle class supplies actual vertex count to B. */
   nir_def *primitives = nir_load_local_apple9_agx(
      &b, nir_imm_int(&b, layout->primitive_count_B), .base = key.output_root);
   nir_def *vertices = nir_load_local_apple9_agx(
      &b, nir_imm_int(&b, layout->vertex_count_B), .base = key.output_root);
   nir_def *lane = nir_load_local_invocation_index(&b);
   if (options && options->clip_halfz) {
      /* Convert the completed homogeneous position, after all body writes and
       * output reads. Doing this at publication also handles scalar/partial
       * stores and different invocations writing Z and W. Keep W unchanged:
       * U consumes it for clipping and perspective division.
       * This is the vertex path's c=1/2 depth conversion, not a state-word
       * change. The body barrier above makes its final Z/W visible. */
      nir_push_if(&b, nir_ult(&b, lane, vertices));
      nir_def *offset = nir_imul_imm(&b, lane, 4);
      nir_def *z_address = nir_iadd_imm(
         &b, offset, layout->position_B + layout->vertex_pitch_B *
                        layout->vertex_component[VARYING_SLOT_POS][2]);
      nir_def *w_address = nir_iadd_imm(
         &b, offset, layout->position_B + layout->vertex_pitch_B *
                        layout->vertex_component[VARYING_SLOT_POS][3]);
      nir_def *z = nir_load_local_apple9_agx(&b, z_address,
                                            .base = key.output_root);
      nir_def *w = nir_load_local_apple9_agx(&b, w_address,
                                            .base = key.output_root);
      nir_def *half = nir_imm_float(&b, 0.5);
      nir_def *converted = nir_ffma_weak(
         &b, nir_fneg(&b, z), half, nir_ffma_weak(&b, w, half, z));
      nir_store_local_apple9_agx(&b, converted, z_address,
                                  .base = key.output_root, .bit_size = 32);
      nir_pop_if(&b, NULL);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
   }
   nir_mesh_control_apple9_agx(&b, nir_u2u16(&b, primitives),
                               nir_u2u16(&b, vertices), .base = 1);
   device_fence(&b);

   nir_push_if(&b, nir_ult(&b, lane, vertices));
   for (unsigned c = 0; c < 4; ++c) {
      unsigned plane =
         layout->position_B +
         layout->vertex_pitch_B * layout->vertex_component[VARYING_SLOT_POS][c];
      nir_mesh_submit_channel_apple9_agx(&b, nir_imm_int(&b, c),
                                         nir_imm_int(&b, plane));
      device_fence(&b);
   }
   for (unsigned slot = VARYING_SLOT_VAR0; slot < ARRAY_SIZE(varyings.mask); ++slot) {
      u_foreach_bit(c, varyings.mask[slot]) {
         unsigned channel = 4 + agx_apple9_varying_index(&varyings, slot, c);
         unsigned plane = layout->position_B + layout->vertex_pitch_B *
                             layout->vertex_component[slot][c];
         nir_mesh_submit_channel_apple9_agx(&b, nir_imm_int(&b, channel),
                                            nir_imm_int(&b, plane));
         device_fence(&b);
      }
   }
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_mesh_submit_scalar_apple9_agx(&b, vertices);
   device_fence(&b);
   nir_pop_if(&b, NULL);

   info->requires_mesh_completion = false;
   nir_progress(true, nir_shader_get_entrypoint(nir), nir_metadata_none);
   nir_validate_shader(nir, "native Apple9 mesh completion");
   return true;
}

bool
agx_compile_apple9_native_mesh(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_native_mesh_options *options,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   memset(out, 0, sizeof(*out));
   if (!agx_nir_lower_apple9_native_mesh(nir, enable_native_mesh, options,
                                         info, reason))
      return false;
   if (!agx_compile_apple9_mesh_lowered(nir, &info->key, out, reason))
      return false;
   return varying_layout(&info->layout, options, &out->info.apple9_varyings,
                         reason);
}
