/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Independent oracle for flat primitive transport using private vertices.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_shader.h"
#include "hk_apple9_mesh.h"
#include "asahi/compiler/agx_nir.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); exit(1); } } while (0)
#define LANES 32
#include "../../compiler/test/apple9-mesh-executor.h"

static unsigned indices_per_primitive = 3;

static nir_shader *
source(unsigned max_v, unsigned max_p, unsigned logical, bool cull)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "primitive vertex transport");
   nir_shader *nir = b.shader;
   nir->info.io_lowered = true;
   nir->info.workgroup_size[0] = logical;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.mesh.max_vertices_out = max_v;
   nir->info.mesh.max_primitives_out = max_p;
   nir->info.mesh.primitive_type = indices_per_primitive == 2 ? MESA_PRIM_LINES : MESA_PRIM_TRIANGLES;
   nir_def *counts = nir_load_num_workgroups(&b);
   nir_def *vertices = nir_channel(&b, counts, 0);
   nir_def *primitives = nir_channel(&b, counts, 1);
   nir_set_vertex_and_primitive_count(&b, vertices, primitives, nir_imm_int(&b, 0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   for (unsigned first = 0; first < max_v; first += logical) {
      nir_def *v = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult(&b, v, vertices));
      for (unsigned c = 0; c < 4; ++c) {
         nir_store_per_vertex_output(&b, nir_iadd_imm(&b, v, 0x10000 + 256 * c),
            v, nir_imm_int(&b, 0), .component = c, .write_mask = 1,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      }
      for (unsigned c = 0; c < 3; ++c) {
         nir_store_per_vertex_output(&b, nir_iadd_imm(&b, v, 0x20000 + 256 * c),
            v, nir_imm_int(&b, 0), .component = 2, .write_mask = 1,
            .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + c, .num_slots = 1});
      }
      nir_store_per_vertex_output(&b, nir_iadd_imm(&b, v, 0x30000),
         v, nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   for (unsigned first = 0; first < max_p; first += logical) {
      nir_def *p = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult(&b, p, primitives));
      nir_def *indices[3];
      for (unsigned c = 0; c < indices_per_primitive; ++c)
         indices[c] = nir_umod(&b, nir_iadd_imm(&b, nir_imul_imm(&b, p, 2), c), vertices);
      nir_store_per_primitive_output(&b, nir_vec(&b, indices, indices_per_primitive), p, nir_imm_int(&b, 0),
         .write_mask = BITFIELD_MASK(indices_per_primitive), .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      /* An array slot offset and sparse components exercise normal IO mapping. */
      nir_store_per_primitive_output(&b, nir_vec2(&b, nir_iadd_imm(&b, p, 0x40000),
                                                nir_iadd_imm(&b, p, 0x50000)),
         p, nir_imm_int(&b, 1), .component = 1, .write_mask = 3,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 2});
      nir_store_per_primitive_output(&b, nir_iadd_imm(&b, p, 0x60000),
         p, nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID, .num_slots = 1});
      if (cull) {
         nir_store_per_primitive_output(&b, nir_ieq_imm(&b, nir_umod_imm(&b, p, 3), 1),
            p, nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_bool1,
            .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE, .num_slots = 1});
      }
      nir_pop_if(&b, NULL);
   }
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir->info.clip_distance_array_size = 1;
   return nir;
}

static uint32_t
at(const struct machine *m, const struct agx_apple9_mesh_body_info *body,
   unsigned slot, unsigned c, unsigned v)
{
   const struct agx_apple9_mesh_layout *l = &body->layout;
   unsigned component = l->vertex_component[slot][c];
   CHECK(component != AGX_APPLE9_MESH_UNUSED);
   uint32_t value;
   memcpy(&value, m->local + l->position_B + component * l->vertex_pitch_B + 4 * v, 4);
   return value;
}

static void
check(unsigned max_v, unsigned max_p, unsigned logical,
      unsigned vertices, unsigned primitives, bool last, bool cull)
{
   nir_shader *nir = source(max_v, max_p, logical, cull);
   struct hk_apple9_graphics_key key = {
      .fs_reads_clip_distance = true,
      .vertex.outputs_flat = VARYING_BIT_VAR(1),
      .vertex.outputs_linear = VARYING_BIT_VAR(2),
   };
   const char *why = NULL;
   bool supported = false;
   CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &why) == VK_SUCCESS);
   if (!supported) fprintf(stderr, "%s\n", why);
   CHECK(supported);
   CHECK(nir->info.mesh.max_vertices_out == max_v);
   struct agx_apple9_native_mesh_options options = {
      .physical_threads = 32, .packed_index_publication = true,
      .publication_rounds = true, .attributes = true, .primitive_vertices = true,
      .link_clip_distances = true,
      .outputs_flat = VARYING_BIT_VAR(1), .outputs_linear = VARYING_BIT_VAR(2),
   };
   struct agx_apple9_mesh_body_info body;
   CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &options, &body, &why));
   CHECK(body.primitive_vertices == (VARYING_BIT_VAR(4) | VARYING_BIT_PRIMITIVE_ID));
   CHECK(body.layout.max_vertices == max_v + max_p && body.original_max_vertices == max_v);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_mesh_submit_channel_apple9_agx ||
             intr->intrinsic == nir_intrinsic_mesh_submit_scalar_apple9_agx)
            nir_instr_remove(instr);
         else if (intr->intrinsic == nir_intrinsic_load_provoking_last) {
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            nir_def_replace(&intr->def, nir_imm_int(&b, last ? -1 : 0));
         }
      }
   }
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   nir_validate_shader(nir, "primitive transport executor");
   struct machine m = {.variable_counts = true,
      .shared_begin = body.shared_offset_B, .shared_end = body.local_allocation_B,
      .expected_vertices = primitives ? vertices + primitives : 0,
      .expected_primitives = primitives, .num_groups = {vertices, primitives, 1}};
   memset(m.local, 0xa5, sizeof(m.local));
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.controls == ALL_LANES);
   for (unsigned p = 0; p < primitives; ++p) {
      bool culled = cull && p % 3 == 1;
      unsigned corner = last ? indices_per_primitive - 1 : 0;
      unsigned provoking = (2 * p + (culled ? 0 : corner)) % vertices;
      unsigned duplicate = vertices + p;
      for (unsigned c = 0; c < indices_per_primitive; ++c) {
         unsigned index = c == corner ? duplicate
                          : (2 * p + (culled ? 0 : c)) % vertices;
         CHECK(m.local[indices_per_primitive * p + c] == index);
      }
      for (unsigned c = 0; c < 3; ++c) {
         CHECK(at(&m, &body, VARYING_SLOT_VAR0 + c, 2, duplicate) ==
               0x20000 + 256 * c + provoking);
      }
      for (unsigned c = 0; c < 4; ++c)
         CHECK(at(&m, &body, VARYING_SLOT_POS, c, duplicate) == 0x10000 + 256 * c + provoking);
      CHECK(at(&m, &body, VARYING_SLOT_CLIP_DIST0, 0, duplicate) == 0x30000 + provoking);
      CHECK(at(&m, &body, AGX_APPLE9_CLIP_MIRROR_SLOT, 0, duplicate) == 0x30000 + provoking);
      CHECK(at(&m, &body, VARYING_SLOT_VAR4, 1, duplicate) == 0x40000 + p);
      CHECK(at(&m, &body, VARYING_SLOT_VAR4, 2, duplicate) == 0x50000 + p);
      CHECK(at(&m, &body, VARYING_SLOT_PRIMITIVE_ID, 0, duplicate) == 0x60000 + p);
   }
   for (unsigned v = 0; v < vertices; ++v)
      for (unsigned c = 0; c < 4; ++c)
         CHECK(at(&m, &body, VARYING_SLOT_POS, c, v) == 0x10000 + 256 * c + v);
   for (unsigned i = body.local_allocation_B; i < sizeof(m.local); ++i)
      CHECK(m.local[i] == 0xa5);
   free(m.value);
   ralloc_free(nir);

   nir = source(max_v, max_p, logical, cull);
   struct agx_shader_part code = {0};
   if (!hk_apple9_compile_native_mesh_program(nir, &key, &code, &body, &why)) {
      fprintf(stderr, "compile: %s\n", why);
      exit(1);
   }
   CHECK(hk_apple9_native_shader_valid(&code.info, &body, HK_APPLE9_NATIVE_MESH));
   CHECK(code.info.apple9_varyings.group[VARYING_SLOT_VAR4] == 1);
   CHECK(code.info.apple9_varyings.group[VARYING_SLOT_PRIMITIVE_ID] == 1);
   struct agx_apple9_mesh_body_info stale = body;
   stale.original_max_vertices++;
   CHECK(!hk_apple9_native_shader_valid(&code.info, &stale, HK_APPLE9_NATIVE_MESH));
   printf("L%u V%u/%u P%u/%u last%u cull%u: exact transport, binary %u\n",
      logical, vertices, max_v, primitives, max_p, last, cull, code.info.binary_size);
   free(code.binary);
   ralloc_free(nir);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const unsigned shapes[][3] = {{3, 2, 1}, {20, 10, 16}, {64, 124, 32}, {128, 128, 64}};
   for (unsigned s = 0; s < ARRAY_SIZE(shapes); ++s) {
      for (unsigned last = 0; last < 2; ++last) {
         unsigned v = shapes[s][0], p = shapes[s][1], l = shapes[s][2];
         check(v, p, l, v, p, last, false);
         check(v, p, l, v - 1, p - 1, last, true);
         check(v, p, l, 3, p, last, false);
         check(v, p, l, 0, 0, last, true);
      }
   }
   indices_per_primitive = 2;
   for (unsigned last = 0; last < 2; ++last) {
      check(64, 124, 1, 64, 124, last, false);
      check(64, 124, 1, 3, 65, last, false);
      check(64, 124, 1, 0, 0, last, false);
   }
   indices_per_primitive = 3;
   /* A primitive slot cannot alias a vertex slot; indirect slot offsets and
    * narrow values must decline on an owned clone before reading immediates. */
   for (unsigned bad = 0; bad < 4; ++bad) {
      nir_shader *nir = source(64, 124, 32, false);
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_per_primitive_output ||
                nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_VAR3)
               continue;
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            if (bad == 0) {
               nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
               sem.location = VARYING_SLOT_VAR0;
               nir_intrinsic_set_io_semantics(intr, sem);
            } else if (bad == 1) {
               nir_src_rewrite(&intr->src[2], nir_load_local_invocation_index(&b));
            } else if (bad == 2) {
               nir_src_rewrite(&intr->src[0], nir_u2u16(&b, intr->src[0].ssa));
            } else {
               nir->info.mesh.max_vertices_out = 133;
            }
         }
      }
      struct hk_apple9_graphics_key key = {0};
      bool supported = true;
      const char *why = NULL;
      CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &why) == VK_SUCCESS);
      CHECK(!supported && why);
      CHECK(nir->info.per_primitive_outputs & VARYING_BIT_VAR(4));
      ralloc_free(nir);
   }
   glsl_type_singleton_decref();
   return 0;
}
