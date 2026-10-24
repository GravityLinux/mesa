/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "poly/tessellator.h"
#include "hk_shader.h"

/* The count pass publishes an indexed triangle draw. The object consumes its
 * index count directly, without a CPU copy or a second producer ABI. Native
 * routing admits one patch, whose tessellator output fits the hardware grid.
 * Use one guarded group for an empty patch; it publishes zero vertices and
 * primitives without relying on a zero-child-grid launch. */
nir_shader *
hk_nir_tes_mesh_object(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "TES triangle grid");
   b.shader->info.workgroup_size[0] = b.shader->info.workgroup_size[1] =
      b.shader->info.workgroup_size[2] = 1;
   nir_def *draw =
      nir_pack_64_2x32(&b, nir_load_native_pointer_apple9_agx(
                              &b, .base = AGX_APPLE9_MESH_DRAW_UNIFORM));
   nir_def *indices = nir_load_global(&b, 1, 32, draw, .align_mul = 4,
                                      .access = ACCESS_NON_WRITEABLE);
   nir_def *triangles = nir_udiv_imm(&b, indices, 3);
   nir_launch_mesh_workgroups(
      &b, nir_vec3(&b, nir_umax(&b, triangles, nir_imm_int(&b, 1)),
                   nir_imm_int(&b, 1), nir_imm_int(&b, 1)));
   nir_validate_shader(b.shader, "TES object draw-word consumer");
   return b.shader;
}

/* Consume the existing tessellator's triangle index stream. This compiler-only
 * contract does not dispatch work or replace tessellation factor generation.
 * Each workgroup evaluates one triangle; lanes 0..2 evaluate its corners.
 */
static bool
supported(const nir_shader *s, unsigned *vertices_in, const char **reason)
{
#define REJECT(msg)                                                            \
   do {                                                                        \
      if (reason)                                                              \
         *reason = msg;                                                        \
      return false;                                                            \
   } while (0)
   if (s->info.stage != MESA_SHADER_TESS_EVAL || !s->info.io_lowered ||
       (s->info.tess._primitive_mode != TESS_PRIMITIVE_TRIANGLES &&
        s->info.tess._primitive_mode != TESS_PRIMITIVE_QUADS) ||
       s->info.tess.point_mode)
      REJECT(
         "TES mesh probe needs triangle or quad domains without point mode");
   if (s->xfb_info || s->info.writes_memory ||
       s->info.clip_distance_array_size || s->info.cull_distance_array_size ||
       s->info.outputs_written != VARYING_BIT_POS ||
       (s->info.inputs_read & ~(VARYING_BIT_POS | VARYING_BIT_TESS_LEVEL_INNER |
                                VARYING_BIT_TESS_LEVEL_OUTER)) ||
       s->info.patch_inputs_read)
      REJECT(
         "TES mesh probe supports Position I/O and tess levels without side effects");

   *vertices_in = 1;
   unsigned implementations = 0;
   nir_foreach_function_impl(impl, s) {
      if (++implementations > 1)
         REJECT("TES mesh probe requires one entrypoint implementation");
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               REJECT("TES mesh probe requires inlined calls");
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_store_output:
               if (nir_intrinsic_io_semantics(intr).location !=
                      VARYING_SLOT_POS ||
                   !nir_src_is_const(intr->src[1]) ||
                   nir_src_as_uint(intr->src[1]) ||
                   intr->src[0].ssa->bit_size != 32 ||
                   nir_intrinsic_component(intr) +
                         intr->src[0].ssa->num_components >
                      4)
                  REJECT("unsupported TES output store");
               break;
            case nir_intrinsic_load_input: {
               unsigned loc = nir_intrinsic_io_semantics(intr).location;
               unsigned components =
                  loc == VARYING_SLOT_TESS_LEVEL_OUTER ? 4 : 2;
               if ((loc != VARYING_SLOT_TESS_LEVEL_OUTER &&
                    loc != VARYING_SLOT_TESS_LEVEL_INNER) ||
                   !nir_src_is_const(intr->src[0]) ||
                   nir_src_as_uint(intr->src[0]) || intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components >
                      components)
                  REJECT("unsupported TES patch input load");
               break;
            }
            case nir_intrinsic_load_per_vertex_input:
               if (nir_intrinsic_io_semantics(intr).location !=
                      VARYING_SLOT_POS ||
                   !nir_src_is_const(intr->src[0]) ||
                   nir_src_as_uint(intr->src[0]) >= 32 ||
                   !nir_src_is_const(intr->src[1]) ||
                   nir_src_as_uint(intr->src[1]) || intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components > 4)
                  REJECT("unsupported TES input load");
               *vertices_in =
                  MAX2(*vertices_in, nir_src_as_uint(intr->src[0]) + 1);
               break;
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
               if (!nir_deref_mode_is_one_of(
                      nir_src_as_deref(intr->src[0]),
                      nir_var_function_temp | nir_var_shader_temp))
                  REJECT(
                     "TES mesh probe requires lowered external dereferences");
               break;
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_tess_coord:
            case nir_intrinsic_load_tess_coord_xy:
            case nir_intrinsic_load_primitive_id:
            case nir_intrinsic_load_patch_vertices_in:
               break;
            default:
               REJECT("unsupported TES mesh probe intrinsic");
            }
         }
      }
   }
   return true;
#undef REJECT
}

struct tes_mesh_state {
   nir_def *lane, *index;
};

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct tes_mesh_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global_invocation_id:
      nir_def_replace(&intr->def, nir_vec3(b, s->index, nir_imm_int(b, 0),
                                           nir_imm_int(b, 0)));
      return true;
   case nir_intrinsic_store_output:
      nir_store_per_vertex_output(
         b, intr->src[0].ssa, s->lane, intr->src[1].ssa,
         .component = nir_intrinsic_component(intr),
         .write_mask = nir_intrinsic_write_mask(intr),
         .src_type = nir_intrinsic_src_type(intr),
         .io_semantics = nir_intrinsic_io_semantics(intr));
      nir_instr_remove(&intr->instr);
      return true;
   default:
      return false;
   }
}

nir_shader *
hk_nir_tes_mesh(const nir_shader *source, const char **reason)
{
   if (reason)
      *reason = NULL;
   unsigned vertices_in;
   if (!supported(source, &vertices_in, reason))
      return NULL;
   nir_shader *nir = nir_shader_clone(NULL, source);
   if (!nir) {
      if (reason)
         *reason = "out of memory cloning TES mesh probe";
      return NULL;
   }
   NIR_PASS(_, nir, nir_lower_returns);
   /* Preserve the existing index winding, coordinate and TCS-buffer ABI. */
   NIR_PASS(_, nir, poly_nir_lower_tes, false);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *group = nir_load_workgroup_id(&b);
   nir_def *in_grid =
      nir_iand(&b, nir_ieq_imm(&b, nir_channel(&b, group, 1), 0),
               nir_ieq_imm(&b, nir_channel(&b, group, 2), 0));
   /* Even metadata reads are suppressed for unused lanes and y/z slices. */
   nir_if *active =
      nir_push_if(&b, nir_iand(&b, nir_ult_imm(&b, lane, 3), in_grid));
   active->control = nir_selection_control_dont_flatten;
   nir_def *params = nir_load_tess_param_buffer_poly(&b);
   nir_def *draw = nir_load_global_constant(
      &b, 1, 64,
      nir_iadd_imm(&b, params, offsetof(struct poly_tess_params, out_draws)),
      .align_mul = 8);
   /* libagx_tess_alloc_draw writes the total index count to out_draws[0]. */
   nir_def *count = nir_load_global_constant(&b, 1, 32, draw, .align_mul = 4);
   nir_def *patch_size = nir_load_global_constant(
      &b, 1, 32,
      nir_iadd_imm(&b, params,
                   offsetof(struct poly_tess_params, output_patch_size)),
      .align_mul = 4);
   nir_def *valid = nir_iand(
      &b, nir_ult(&b, nir_channel(&b, group, 0), nir_udiv_imm(&b, count, 3)),
      nir_iand(&b, nir_uge_imm(&b, patch_size, vertices_in),
               nir_ule_imm(&b, patch_size, 32)));
   /* Bound the reads and stores themselves. Test before multiplication so
    * oversized groups cannot wrap to a valid index. Ignore a partial triangle.
    * Index contents and backing allocations retain the producer's contract. */
   nir_if *bounded = nir_push_if(&b, valid);
   bounded->control = nir_selection_control_dont_flatten;
   struct tes_mesh_state s = {
      .lane = lane,
      .index =
         nir_iadd(&b, nir_imul_imm(&b, nir_channel(&b, group, 0), 3), lane),
   };
   b.cursor = nir_cf_reinsert(&body, b.cursor);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .write_mask = 7, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
   nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_set_vertex_and_primitive_count(
      &b, nir_bcsel(&b, valid, nir_imm_int(&b, 3), nir_imm_int(&b, 0)),
      nir_b2i32(&b, valid), nir_imm_int(&b, 0));
   nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);
   /* Out-of-grid workgroups publish an empty mesh from lane zero. */
   nir_push_if(&b,
               nir_iand(&b, nir_ieq_imm(&b, lane, 0), nir_inot(&b, in_grid)));
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0),
                                      nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_pop_if(&b, NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower, nir_metadata_none, &s);

   memset(&nir->info.mesh, 0, sizeof(nir->info.mesh));
   nir->info.stage = MESA_SHADER_MESH;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir->info.mesh.max_vertices_out = 3;
   nir->info.mesh.max_primitives_out = 1;
   nir->info.workgroup_size_variable = false;
   nir->info.workgroup_size[0] = 16;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   nir_shader_gather_info(nir, impl);
   nir_validate_shader(nir, "TES mesh lowering");
   return nir;
}
