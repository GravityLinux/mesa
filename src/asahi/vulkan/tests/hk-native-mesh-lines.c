/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Execute line bodies and every publication collective against a CPU oracle.
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

static uint32_t channel_mask[8][8], scalar_mask[4];
static uint32_t published[8][256];
static uint8_t indices[512];

static bool
publication(struct machine *m, nir_intrinsic_instr *n, unsigned lane)
{
   if (n->intrinsic != nir_intrinsic_mesh_submit_channel_apple9_agx &&
       n->intrinsic != nir_intrinsic_mesh_submit_scalar_apple9_agx)
      return false;
   uint32_t selector = src(m, n->src[0], lane)[0].u32;
   unsigned first = selector >> 16;
   unsigned offset = src(m, n->src[1], lane)[0].u32;
   CHECK(!(first % 32) && !(offset % 16));
   CHECK(m->controls == ALL_LANES && offset + 4 * lane + 4 <= sizeof(m->local));
   if (n->intrinsic == nir_intrinsic_mesh_submit_channel_apple9_agx) {
      unsigned c = selector & 0xffff;
      CHECK(c < 8 && first + lane < m->expected_vertices);
      CHECK(!(channel_mask[c][first / 32] & (1u << lane)));
      channel_mask[c][first / 32] |= 1u << lane;
      memcpy(&published[c][first + lane], m->local + offset + 4 * lane, 4);
   } else {
      CHECK((selector & 0xffff) == m->expected_vertices && offset == 4 * first);
      CHECK(first + lane < MAX2(DIV_ROUND_UP(2 * m->expected_primitives, 4), 1));
      CHECK(!(scalar_mask[first / 32] & (1u << lane)));
      scalar_mask[first / 32] |= 1u << lane;
      memcpy(indices + 4 * (first + lane), m->local + offset + 4 * lane, 4);
   }
   return true;
}

static nir_shader *
source(unsigned max_v, unsigned max_p, unsigned logical)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "native line publication");
   nir_shader *nir = b.shader;
   nir->info.io_lowered = true;
   nir->info.workgroup_size[0] = logical;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.mesh.max_vertices_out = max_v;
   nir->info.mesh.max_primitives_out = max_p;
   nir->info.mesh.primitive_type = MESA_PRIM_LINES;
   nir_def *counts = nir_load_num_workgroups(&b);
   nir_def *vertices = nir_channel(&b, counts, 0);
   nir_def *primitives = nir_channel(&b, counts, 1);
   nir_set_vertex_and_primitive_count(&b, vertices, primitives, nir_imm_int(&b, 0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   for (unsigned first = 0; first < max_v; first += logical) {
      nir_def *v = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult(&b, v, vertices));
      for (unsigned c = 0; c < 8; ++c)
         nir_store_per_vertex_output(&b, nir_iadd_imm(&b, v, 0x10000 * (c + 1)),
            v, nir_imm_int(&b, 0), .component = c % 4, .write_mask = 1,
            .src_type = nir_type_float32,
            .io_semantics = {.location = c < 4 ? VARYING_SLOT_POS : VARYING_SLOT_VAR0,
                             .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   for (unsigned first = 0; first < max_p; first += logical) {
      nir_def *p = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult(&b, p, primitives));
      /* Exercise high vertex indices, odd values and word-straddling pairs. */
      nir_def *i0 = nir_umod(&b, nir_imul_imm(&b, p, 3), vertices);
      nir_def *i1 = nir_isub(&b, nir_iadd_imm(&b, vertices, -1), i0);
      nir_store_per_primitive_output(&b, nir_vec2(&b, i0, i1), p,
         nir_imm_int(&b, 0), .write_mask = 3, .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return nir;
}

static void
check(unsigned max_v, unsigned max_p, unsigned logical,
      unsigned vertices, unsigned primitives)
{
   nir_shader *nir = source(max_v, max_p, logical);
   const struct agx_apple9_native_mesh_options options = {
      .physical_threads = 32, .packed_index_publication = true,
      .publication_rounds = true, .attributes = true,
   };
   const char *why = NULL;
   struct agx_apple9_mesh_body_info body;
   CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &options, &body, &why));
   CHECK(body.layout.indices_per_primitive == 2);
   CHECK(body.layout.primitive_count_B == ALIGN_POT(2 * max_p, 16));
   CHECK(body.layout.position_B == ALIGN_POT(2 * max_p, 16) + 16);
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct machine m = {
      .variable_counts = true, .num_groups = {vertices, primitives, 1},
      .expected_vertices = primitives && vertices ? vertices : 0,
      .expected_primitives = primitives && vertices ? primitives : 0,
      .shared_begin = body.shared_offset_B, .shared_end = body.local_allocation_B,
      .handle_intrinsic = publication,
   };
   memset(m.local, 0xa5, sizeof(m.local));
   memset(channel_mask, 0, sizeof(channel_mask));
   memset(scalar_mask, 0, sizeof(scalar_mask));
   memset(published, 0xa5, sizeof(published));
   memset(indices, 0xa5, sizeof(indices));
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.controls == ALL_LANES);
   for (unsigned c = 0; c < 8; ++c) {
      for (unsigned r = 0; r < 8; ++r) {
         unsigned active = MIN2(m.expected_vertices - MIN2(m.expected_vertices, 32 * r), 32);
         CHECK(channel_mask[c][r] == BITFIELD_MASK(active));
      }
      for (unsigned v = 0; v < m.expected_vertices; ++v)
         CHECK(published[c][v] == 0x10000 * (c + 1) + v);
      for (unsigned v = m.expected_vertices; v < 256; ++v)
         CHECK(published[c][v] == 0xa5a5a5a5);
   }
   unsigned words = MAX2(DIV_ROUND_UP(2 * m.expected_primitives, 4), 1);
   for (unsigned r = 0; r < 4; ++r) {
      unsigned active = MIN2(words - MIN2(words, 32 * r), 32);
      CHECK(scalar_mask[r] == BITFIELD_MASK(active));
   }
   for (unsigned p = 0; p < m.expected_primitives; ++p) {
      CHECK(indices[2 * p] == 3 * p % vertices);
      CHECK(indices[2 * p + 1] == vertices - 1 - 3 * p % vertices);
   }
   for (unsigned b = body.local_allocation_B; b < sizeof(m.local); ++b)
      CHECK(m.local[b] == 0xa5);
   free(m.value);
   ralloc_free(nir);

   nir = source(max_v, max_p, logical);
   struct hk_apple9_graphics_key key = {0};
   bool supported = false;
   CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &why) == VK_SUCCESS);
   CHECK(supported);
   key.line_stipple = true;
   CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &why) == VK_SUCCESS);
   CHECK(!supported);
   key.line_stipple = false;
   struct agx_shader_part part = {0};
   CHECK(hk_apple9_compile_native_mesh_program(nir, &key, &part, &body, &why));
   CHECK(hk_apple9_native_shader_valid(&part.info, &body, HK_APPLE9_NATIVE_MESH));
   struct agx_apple9_mesh_body_info stale = body;
   stale.layout.indices_per_primitive = 3;
   /* Padding can coincide for P1; changing the stride is then deliberately
    * tested on the larger declarations where the binary's layout differs. */
   if (ALIGN_POT(2 * max_p, 16) != ALIGN_POT(3 * max_p, 16))
      CHECK(!hk_apple9_native_shader_valid(&part.info, &stale, HK_APPLE9_NATIVE_MESH));
   stale = body; stale.cull_distances = 1;
   CHECK(!hk_apple9_native_shader_valid(&part.info, &stale, HK_APPLE9_NATIVE_MESH));
   stale = body; stale.publication_rounds = false;
   CHECK(!hk_apple9_native_shader_valid(&part.info, &stale, HK_APPLE9_NATIVE_MESH));
   uint32_t state[AGX_APPLE9_MESH_STATE_WORDS];
   const struct agx_apple9_mesh_state_parameters params = {
      .layout = &body.layout, .threads = 32,
   };
   CHECK(agx_apple9_mesh_state_build(state, &params));
   CHECK(state[10] == (max_p - 1 + ((DIV_ROUND_UP(max_p, 16) - 1) << 16)));
   CHECK(state[11] == 0x400820 && state[13] == 1 + DIV_ROUND_UP(max_v, 4) * 8);
   printf("lines L%u V%u/%u P%u/%u: exact publication, binary %u\n",
          logical, vertices, max_v, primitives, max_p, part.info.binary_size);
   free(part.binary);
   ralloc_free(nir);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const unsigned shapes[][3] = {
      {2, 1, 1}, {3, 1, 1}, {64, 126, 1}, {126, 64, 1},
      {128, 65, 16}, {256, 256, 32}, {64, 126, 64},
   };
   for (unsigned s = 0; s < ARRAY_SIZE(shapes); ++s) {
      unsigned v = shapes[s][0], p = shapes[s][1], l = shapes[s][2];
      check(v, p, l, v, p);
      check(v, p, l, v - 1, p - 1);
      check(v, p, l, v, 0);
      check(v, p, l, 0, 0);
   }
   /* Both line-cull forms and a third index component must decline before
    * any upload. Points remain outside this admission class. */
   for (unsigned bad = 0; bad < 4; ++bad) {
      nir_shader *nir = source(64, 126, 1);
      nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
      if (bad == 0)
         nir_store_per_primitive_output(&b, nir_imm_true(&b), nir_imm_int(&b, 0),
            nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_bool1,
            .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE, .num_slots = 1});
      else if (bad == 2)
         nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0),
            nir_imm_int(&b, 0), .write_mask = 7, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      else if (bad == 3)
         nir->info.mesh.primitive_type = MESA_PRIM_POINTS;
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      if (bad == 1)
         nir->info.cull_distance_array_size = 1;
      struct hk_apple9_graphics_key key = {0};
      bool supported = true;
      const char *why = NULL;
      CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &why) == VK_SUCCESS);
      CHECK(!supported && why);
      ralloc_free(nir);
   }
   glsl_type_singleton_decref();
   return 0;
}
