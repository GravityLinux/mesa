/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_mesh_formation.h"

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

static bool
bind_draw_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_mesh_formation_input_apple9_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   /* This is an internal argument-table pointer, separate from task payload
    * and application descriptors. The runtime supplies the current draw. */
   nir_def *draw =
      nir_pack_64_2x32(b, nir_load_native_pointer_apple9_agx(
                             b, .base = AGX_APPLE9_MESH_DRAW_UNIFORM));
   unsigned offset = sizeof(uint64_t) * nir_intrinsic_base(intr);
   nir_def *value = nir_load_global(
      b, 1, intr->def.bit_size, nir_iadd_imm(b, draw, offset),
      .align_mul = sizeof(uint64_t), .access = ACCESS_NON_WRITEABLE);
   nir_def_replace(&intr->def, value);
   return true;
}

static void
device_fence(nir_builder *b)
{
   nir_barrier(b, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

bool
agx_nir_lower_apple9_native_mesh(nir_shader *nir, bool enable_native_mesh,
                                 struct agx_apple9_mesh_body_info *info,
                                 const char **reason)
{
   if (!enable_native_mesh)
      return fail(reason, "native mesh development path is disabled");
   if (!eligible(nir, reason))
      return false;

   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   key.physical_threads = AGX_APPLE9_NATIVE_THREADS;
   if (!agx_nir_lower_apple9_mesh_formation(nir, &key, info, reason))
      return false;
   const struct agx_apple9_mesh_layout *layout = &info->layout;
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

   /* Formation ends at a converged workgroup rendezvous. Counts are the
    * actual shader publication, not maxima or register-descriptor constants.
    * U0 consumes and clears its A halfword; B's wider meaning is unresolved.
    * The qualified one-triangle class supplies actual vertex count to B. */
   nir_def *primitives = nir_load_local_apple9_agx(
      &b, nir_imm_int(&b, layout->primitive_count_B), .base = key.output_root);
   nir_def *vertices = nir_load_local_apple9_agx(
      &b, nir_imm_int(&b, layout->vertex_count_B), .base = key.output_root);
   nir_mesh_control_apple9_agx(&b, nir_u2u16(&b, primitives),
                               nir_u2u16(&b, vertices), .base = 1);
   device_fence(&b);

   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b, nir_ult(&b, lane, vertices));
   for (unsigned c = 0; c < 4; ++c) {
      unsigned plane =
         layout->position_B +
         layout->vertex_pitch_B * layout->vertex_component[VARYING_SLOT_POS][c];
      nir_mesh_submit_channel_apple9_agx(&b, nir_imm_int(&b, c),
                                         nir_imm_int(&b, plane));
      device_fence(&b);
   }
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_mesh_submit_scalar_apple9_agx(&b, vertices);
   device_fence(&b);
   nir_pop_if(&b, NULL);

   nir_shader_intrinsics_pass(nir, bind_draw_input, nir_metadata_none, NULL);
   info->requires_mesh_completion = false;
   nir_progress(true, nir_shader_get_entrypoint(nir), nir_metadata_none);
   nir_validate_shader(nir, "native Apple9 mesh completion");
   return true;
}

bool
agx_compile_apple9_native_mesh(nir_shader *nir, bool enable_native_mesh,
                               struct agx_shader_part *out,
                               struct agx_apple9_mesh_body_info *info,
                               const char **reason)
{
   memset(out, 0, sizeof(*out));
   if (!agx_nir_lower_apple9_native_mesh(nir, enable_native_mesh, info, reason))
      return false;
   return agx_compile_apple9_mesh_lowered(nir, &info->key, out, reason);
}
