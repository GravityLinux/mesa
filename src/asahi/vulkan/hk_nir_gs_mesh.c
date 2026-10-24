/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "poly/geometry.h"
#include "hk_apple9_mesh.h"
#include "hk_shader.h"

/* First GS-to-mesh stage: one triangle-list input primitive, one GS
 * invocation, position-only triangle strips. This is a compiler contract;
 * draw routing must independently establish topology, ordering and state.
 * The public helper clones its input and never changes the fallback shader.
 */
static bool
supported(const nir_shader *s, const char **reason)
{
#define REJECT(msg)                                                            \
   do {                                                                        \
      if (reason)                                                              \
         *reason = msg;                                                        \
      return false;                                                            \
   } while (0)
   if (s->info.stage != MESA_SHADER_GEOMETRY || !s->info.io_lowered ||
       s->info.gs.input_primitive != MESA_PRIM_TRIANGLES ||
       s->info.gs.output_primitive != MESA_PRIM_TRIANGLE_STRIP ||
       s->info.gs.invocations != 1 || s->info.gs.active_stream_mask != 1 ||
       s->info.gs.vertices_out < 3 || s->info.gs.vertices_out > 32)
      REJECT(
         "GS mesh probe needs one triangle-list invocation and 3..32 output vertices");

   if (s->xfb_info || s->info.writes_memory ||
       s->info.clip_distance_array_size || s->info.cull_distance_array_size ||
       s->info.outputs_written != VARYING_BIT_POS ||
       (s->info.inputs_read & ~VARYING_BIT_POS))
      REJECT(
         "GS mesh probe supports only Position I/O without XFB or side effects");

   unsigned implementations = 0;
   nir_foreach_function_impl(impl, s) {
      if (++implementations > 1)
         REJECT("GS mesh probe requires one entrypoint implementation");
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               REJECT("GS mesh probe requires inlined calls");
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
                  REJECT("unsupported GS output store");
               break;
            case nir_intrinsic_load_per_vertex_input:
               if (nir_intrinsic_io_semantics(intr).location !=
                      VARYING_SLOT_POS ||
                   !nir_src_is_const(intr->src[0]) ||
                   nir_src_as_uint(intr->src[0]) >= 3 ||
                   !nir_src_is_const(intr->src[1]) ||
                   nir_src_as_uint(intr->src[1]) || intr->def.bit_size != 32 ||
                   nir_intrinsic_component(intr) + intr->def.num_components > 4)
                  REJECT("unsupported GS input load");
               break;
            case nir_intrinsic_emit_vertex:
            case nir_intrinsic_end_primitive:
               if (nir_intrinsic_stream_id(intr))
                  REJECT("GS mesh probe supports stream zero only");
               break;
            case nir_intrinsic_load_deref:
            case nir_intrinsic_store_deref:
               if (!nir_deref_mode_is_one_of(
                      nir_src_as_deref(intr->src[0]),
                      nir_var_function_temp | nir_var_shader_temp))
                  REJECT(
                     "GS mesh probe requires lowered external dereferences");
               break;
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_primitive_id:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_invocation_id:
               break;
            default:
               REJECT("unsupported GS mesh probe intrinsic");
            }
         }
      }
   }
   return true;
#undef REJECT
}

struct gs_mesh_state {
   unsigned max_vertices;
   /* Input triangle within the draw, and this invocation's first output
    * vertex and primitive slot: workgroup x and 0, 0 unpacked; with packing,
    * x * HK_APPLE9_NATIVE_GS_PACK + lane and the lane's fixed slots. */
   nir_def *primitive, *vertex_base, *primitive_base;
   nir_variable *position;
   nir_variable *vertices;
   nir_variable *primitives;
   nir_variable *strip;
};

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gs_mesh_state *s = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_store_output: {
      nir_def *value = nir_undef(b, 4, 32);
      unsigned component = nir_intrinsic_component(intr);
      for (unsigned i = 0; i < intr->src[0].ssa->num_components; ++i)
         value = nir_vector_insert_imm(
            b, value, nir_channel(b, intr->src[0].ssa, i), component + i);
      nir_store_var(b, s->position, value,
                    nir_intrinsic_write_mask(intr) << component);
      break;
   }
   case nir_intrinsic_emit_vertex: {
      nir_def *v = nir_load_var(b, s->vertices);
      /* GS over-emission has no effect. Bound the stores themselves, not
       * only the counts published after the body. Each primitive requires
       * three accepted vertices, bounding primitive indices as well. */
      nir_push_if(b, nir_ult_imm(b, v, s->max_vertices));
      {
         nir_def *p = nir_load_var(b, s->primitives);
         nir_def *n = nir_load_var(b, s->strip);
         nir_store_per_vertex_output(
            b, nir_load_var(b, s->position), nir_iadd(b, s->vertex_base, v), nir_imm_int(b, 0),
            .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_push_if(b, nir_uge_imm(b, n, 2));
         {
            /* Reset parity at EndPrimitive. A strip's second triangle reverses
             * its first two corners; mesh output is an explicit triangle list. */
            nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, n, 1), 0);
            nir_def *a = nir_iadd_imm(b, v, -2), *c = nir_iadd_imm(b, v, -1);
            nir_def *indices = nir_iadd(
               b, nir_vec3(b, nir_bcsel(b, odd, c, a), nir_bcsel(b, odd, a, c), v),
               s->vertex_base);
            nir_store_per_primitive_output(
               b, indices, nir_iadd(b, s->primitive_base, p), nir_imm_int(b, 0),
               .write_mask = 7,
               .src_type = nir_type_uint32,
               .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                                .num_slots = 1});
            nir_store_var(b, s->primitives, nir_iadd_imm(b, p, 1), 1);
         }
         nir_pop_if(b, NULL);
         nir_store_var(b, s->vertices, nir_iadd_imm(b, v, 1), 1);
         nir_store_var(b, s->strip, nir_iadd_imm(b, n, 1), 1);
         /* GS outputs become undefined after EmitVertex. */
         nir_store_var(b, s->position, nir_undef(b, 4, 32), 15);
      }
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_end_primitive:
      nir_store_var(b, s->strip, nir_imm_int(b, 0), 1);
      break;
   case nir_intrinsic_load_per_vertex_input: {
      /* Reuse poly's existing VS buffer layout. This narrow contract is a
       * triangle-list draw; dispatch x is primitive, y is instance. */
      nir_def *id = nir_load_workgroup_id(b);
      nir_def *vp = nir_load_vertex_param_buffer_poly(b);
      nir_def *verts = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, vp, offsetof(struct poly_vertex_params, grid[0])),
         .align_mul = 4);
      nir_def *vertex =
         nir_iadd(b, nir_imul(b, nir_channel(b, id, 1), verts),
                  nir_iadd(b, nir_imul_imm(b, s->primitive, 3), intr->src[0].ssa));
      /* Stage is still GS here so poly reads the producer's output mask. */
      nir_def_replace(&intr->def, poly_load_per_vertex_input(b, intr, vertex));
      return true;
   }
   case nir_intrinsic_load_invocation_id:
      nir_def_replace(&intr->def, nir_imm_int(b, 0));
      return true;
   case nir_intrinsic_load_primitive_id:
      nir_def_replace(&intr->def, s->primitive);
      return true;
   case nir_intrinsic_load_instance_id:
      nir_def_replace(&intr->def, nir_channel(b, nir_load_workgroup_id(b), 1));
      return true;
   default:
      return false;
   }
   nir_instr_remove(&intr->instr);
   return true;
}

nir_shader *
hk_nir_gs_mesh(const nir_shader *source, const char **reason)
{
   if (reason)
      *reason = NULL;
   if (!supported(source, reason))
      return NULL;
   nir_shader *nir = nir_shader_clone(NULL, source);
   if (!nir) {
      if (reason)
         *reason = "out of memory cloning GS mesh probe";
      return NULL;
   }
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   unsigned vertices = source->info.gs.vertices_out;
   /* Packing covers the qualified per-triangle shapes V3/P1 and V4/P2. */
   const unsigned pack =
      hk_apple9_native_gs_pack_enabled() && (vertices == 3 || vertices == 4)
         ? HK_APPLE9_NATIVE_GS_PACK : 1;
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *group = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   struct gs_mesh_state s = {
      .max_vertices = vertices,
      .primitive = pack > 1 ? nir_iadd(&b, nir_imul_imm(&b, group, pack), lane) : group,
      .vertex_base = pack > 1 ? nir_imul_imm(&b, lane, vertices) : nir_imm_int(&b, 0),
      .primitive_base =
         pack > 1 ? nir_imul_imm(&b, lane, vertices - 2) : nir_imm_int(&b, 0),
      .position =
         nir_local_variable_create(impl, glsl_vec4_type(), "gs_position"),
      .vertices =
         nir_local_variable_create(impl, glsl_uint_type(), "gs_vertices"),
      .primitives =
         nir_local_variable_create(impl, glsl_uint_type(), "gs_primitives"),
      .strip =
         nir_local_variable_create(impl, glsl_uint_type(), "gs_strip_vertices"),
   };
   nir_store_var(&b, s.position, nir_undef(&b, 4, 32), 15);
   nir_store_var(&b, s.vertices, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, s.primitives, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, s.strip, nir_imm_int(&b, 0), 1);
   nir_cursor after_setup = b.cursor;
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower, nir_metadata_none, &s);

   if (pack == 1) {
      b.cursor = nir_after_impl(impl);
      nir_def *p = nir_load_var(&b, s.primitives);
      nir_def *v = nir_bcsel(&b, nir_ine_imm(&b, p, 0),
                             nir_load_var(&b, s.vertices), nir_imm_int(&b, 0));
      nir_set_vertex_and_primitive_count(&b, v, p, nir_imm_int(&b, 0));

      nir_cf_list body;
      nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
      b.cursor = nir_before_impl(impl);
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
      b.cursor = nir_cf_reinsert(&body, b.cursor);
      nir_pop_if(&b, NULL);
   } else {
      /* Lanes 0..pack-1 run one GS invocation each, for consecutive input
       * triangles, so publication order is API order. Each owns fixed
       * slots; primitive slots its GS did not fill become degenerate
       * triangles on its first vertex, and the group publishes every slot
       * of its live lanes. */
      nir_cf_list body;
      nir_cf_extract(&body, after_setup, nir_after_impl(impl));
      b.cursor = after_setup;
      nir_def *vp = nir_load_vertex_param_buffer_poly(&b);
      nir_def *verts = nir_load_global_constant(
         &b, 1, 32,
         nir_iadd_imm(&b, vp, offsetof(struct poly_vertex_params, grid[0])),
         .align_mul = 4);
      nir_def *triangles = nir_udiv_imm(&b, verts, 3);
      nir_def *first = nir_imul_imm(&b, group, pack);
      nir_def *live = nir_umin(&b, nir_usub_sat(&b, triangles, first),
                               nir_imm_int(&b, pack));
      nir_if *active = nir_push_if(&b, nir_ult(&b, lane, live));
      active->control = nir_selection_control_dont_flatten;
      b.cursor = nir_cf_reinsert(&body, b.cursor);
      /* A lane whose GS emitted nothing still publishes its slots: give the
       * degenerate fill a finite vertex. */
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_var(&b, s.vertices), 0));
      nir_store_per_vertex_output(
         &b, nir_imm_vec4(&b, 0, 0, 0, 1), s.vertex_base, nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_pop_if(&b, NULL);
      nir_def *p = nir_load_var(&b, s.primitives);
      for (unsigned i = 0; i < vertices - 2; ++i) {
         nir_push_if(&b, nir_ule_imm(&b, p, i));
         nir_store_per_primitive_output(
            &b, nir_vec3(&b, s.vertex_base, s.vertex_base, s.vertex_base),
            nir_iadd_imm(&b, s.primitive_base, i), nir_imm_int(&b, 0),
            .write_mask = 7, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                             .num_slots = 1});
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, active);
      nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
      nir_set_vertex_and_primitive_count(
         &b, nir_imul_imm(&b, live, vertices),
         nir_imul_imm(&b, live, vertices - 2), nir_imm_int(&b, 0));
      nir_pop_if(&b, NULL);
   }

   memset(&nir->info.mesh, 0, sizeof(nir->info.mesh));
   nir->info.stage = MESA_SHADER_MESH;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   if (pack > 1) {
      /* The qualified classes T32 V24/P17 and T32 V32/P16. */
      nir->info.mesh.max_vertices_out = pack * vertices;
      nir->info.mesh.max_primitives_out = vertices == 3 ? 17 : 16;
      nir->info.workgroup_size[0] = 32;
   } else {
      nir->info.mesh.max_vertices_out = vertices;
      nir->info.mesh.max_primitives_out = vertices - 2;
      nir->info.workgroup_size[0] = vertices <= 16 ? 16 : 32;
   }
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   nir_shader_gather_info(nir, impl);
   nir_validate_shader(nir, "GS mesh emission lowering");
   return nir;
}
