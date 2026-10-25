/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Execute lowered multi-primitive culling against an independent index oracle.
 * Publication collectives are covered by hk-native-mesh-rounds.
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

static nir_shader *
source(unsigned vertices, unsigned primitives, bool distance, bool boolean)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "multi-primitive native culling");
   nir_shader *nir = b.shader;
   nir->info.io_lowered = true;
   nir->info.workgroup_size[0] = 32;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.mesh.max_vertices_out = vertices;
   nir->info.mesh.max_primitives_out = primitives;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir->info.shared_size = 12;
   nir_def *counts = nir_load_num_workgroups(&b);
   nir_set_vertex_and_primitive_count(&b, nir_channel(&b, counts, 0),
                                    nir_channel(&b, counts, 1), nir_imm_int(&b, 0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_store_shared(&b, nir_imm_ivec3(&b, 0x12345678, 0xabcdef01, 0x76543210),
                    nir_imm_int(&b, 0), .align_mul = 4, .write_mask = 7);
   nir_pop_if(&b, NULL);
   for (unsigned first = 0; first < vertices; first += 32) {
      nir_def *v = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult_imm(&b, v, vertices));
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, .25, .5, .75, 1), v,
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      if (distance) {
         nir_def *d = nir_bcsel(&b, nir_ult_imm(&b, v, vertices / 4),
                               nir_imm_float(&b, -1), nir_imm_float(&b, 1));
         nir_store_per_vertex_output(&b, d, v, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0, .num_slots = 1});
      }
      nir_pop_if(&b, NULL);
   }
   for (unsigned first = 0; first < primitives; first += 32) {
      nir_def *p = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult_imm(&b, p, primitives));
      nir_def *indices[3];
      for (unsigned c = 0; c < 3; ++c)
         indices[c] = nir_umod_imm(&b, nir_iadd_imm(&b, nir_imul_imm(&b, p, 3), c), vertices);
      nir_store_per_primitive_output(&b, nir_vec(&b, indices, 3), p,
         nir_imm_int(&b, 0), .write_mask = 7, .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      nir_def *culled = nir_ieq_imm(&b, nir_umod_imm(&b, p, 3), 1);
      /* Leave some flags unwritten: poisoned LDS must still leave these
       * primitives visible, while the other flags exercise last-write wins. */
      nir_push_if(&b, nir_ine_imm(&b, nir_umod_imm(&b, p, 5), 2));
      /* Last write wins, including false overwriting true. */
      for (unsigned overwrite = 0; overwrite < 2; ++overwrite) {
         nir_def *flag = overwrite ? culled : nir_inot(&b, culled);
         if (!boolean)
            flag = nir_b2i32(&b, flag);
         nir_store_per_primitive_output(&b, flag, p, nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = boolean ? nir_type_bool1 : nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE, .num_slots = 1});
      }
      nir_pop_if(&b, NULL);
      nir_pop_if(&b, NULL);
   }
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir->info.cull_distance_array_size = distance;
   return nir;
}

static void
check(unsigned vertices, unsigned primitives, unsigned actual, bool distance,
      bool boolean)
{
   nir_shader *nir = source(vertices, primitives, distance, boolean);
   struct hk_apple9_graphics_key key = {0};
   const char *why = NULL;
   bool supported = false;
   CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &why) == VK_SUCCESS);
   if (!supported) fprintf(stderr, "%s\n", why);
   CHECK(supported);
   /* Capability probing must not consume the source output. */
   CHECK(nir->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE);
   struct agx_apple9_native_mesh_options options = {
      .physical_threads = 32, .packed_index_publication = true,
      .publication_rounds = true,
   };
   struct agx_apple9_mesh_body_info body;
   CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &options, &body, &why));
   CHECK(body.cull_primitive && body.cull_distances == distance);
   CHECK(!body.layout.culling && !body.layout.primitive_components);
   CHECK(body.cull_primitive_B == 12 + (distance ? vertices * 4 : 0));
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         CHECK(op != nir_intrinsic_store_per_primitive_output);
         if (op == nir_intrinsic_mesh_submit_channel_apple9_agx ||
             op == nir_intrinsic_mesh_submit_scalar_apple9_agx)
            nir_instr_remove(instr);
      }
   }
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   nir_validate_shader(nir, "executed native cull completion");
   struct machine m = {.variable_counts = true,
                       .shared_begin = body.shared_offset_B,
                       .shared_end = body.local_allocation_B,
                       .expected_vertices = actual ? vertices : 0,
                       .expected_primitives = actual,
                       .num_groups = {vertices, actual, 1}};
   memset(m.local, 0xa5, sizeof(m.local));
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.controls == ALL_LANES);
   for (unsigned p = 0; p < primitives; ++p) {
      unsigned v[3];
      bool all_negative = true;
      for (unsigned c = 0; c < 3; ++c) {
         v[c] = (3 * p + c) % vertices;
         all_negative &= v[c] < vertices / 4;
      }
      bool flag = p % 3 == 1 && p % 5 != 2;
      uint32_t stored_flag;
      memcpy(&stored_flag, m.local + body.shared_offset_B +
                          body.cull_primitive_B + 4 * p, sizeof(stored_flag));
      CHECK(stored_flag == flag);
      bool cull = p < actual && (flag || (distance && all_negative));
      for (unsigned c = 0; c < 3; ++c) {
         if (m.local[3 * p + c] != (cull ? v[0] : v[c]))
            fprintf(stderr, "V%u P%u actual%u d%u bool%u p%u c%u: %u != %u\n",
                    vertices, primitives, actual, distance, boolean, p, c,
                    m.local[3 * p + c], cull ? v[0] : v[c]);
         CHECK(m.local[3 * p + c] == (cull ? v[0] : v[c]));
      }
   }
   const uint32_t user_shared[] = {0x12345678, 0xabcdef01, 0x76543210};
   CHECK(!memcmp(m.local + body.shared_offset_B, user_shared, sizeof(user_shared)));
   for (unsigned i = body.local_allocation_B; i < sizeof(m.local); ++i)
      CHECK(m.local[i] == 0xa5);
   free(m.value);
   ralloc_free(nir);

   nir = source(vertices, primitives, distance, boolean);
   struct agx_shader_part code = {0};
   CHECK(hk_apple9_compile_native_mesh_program(nir, &key, &code, &body, &why));
   CHECK(hk_apple9_native_shader_valid(&code.info, &body, HK_APPLE9_NATIVE_MESH));
   struct agx_apple9_mesh_body_info stale = body;
   stale.cull_primitive_B = body.shared_size_B;
   CHECK(!hk_apple9_native_shader_valid(&code.info, &stale, HK_APPLE9_NATIVE_MESH));
   stale = body;
   stale.cull_primitive_B++;
   CHECK(!hk_apple9_native_shader_valid(&code.info, &stale, HK_APPLE9_NATIVE_MESH));
   CHECK(body.cull_primitive && !body.layout.culling);
   CHECK(code.info.apple9_varyings.count == 0 && !code.info.scratch_size);
   printf("V%u P%u actual%u distance%u bool%u: indices exact, binary %u bytes\n",
          vertices, primitives, actual, distance, boolean, code.info.binary_size);
   free(code.binary);
   ralloc_free(nir);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const unsigned shapes[][2] = {{3, 2}, {20, 10}, {64, 126}, {256, 256}};
   for (unsigned s = 0; s < ARRAY_SIZE(shapes); ++s) {
      for (unsigned d = 0; d < 2; ++d) {
         check(shapes[s][0], shapes[s][1], shapes[s][1], d, false);
         check(shapes[s][0], shapes[s][1], shapes[s][1] - 1, d, true);
         check(shapes[s][0], shapes[s][1], 0, d, false);
      }
   }
   /* Malformed/unsupported CullPrimitive IO must fail capability checking,
    * without mutating the caller or consuming an unchecked immediate. */
   for (unsigned bad = 0; bad < 2; ++bad) {
      nir_shader *nir = source(64, 126, false, false);
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_per_primitive_output ||
                nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_CULL_PRIMITIVE)
               continue;
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            if (bad)
               nir_src_rewrite(&intr->src[0], nir_u2u16(&b, intr->src[0].ssa));
            else
               nir_src_rewrite(&intr->src[2], nir_load_local_invocation_index(&b));
         }
      }
      struct hk_apple9_graphics_key key = {0};
      const char *why = NULL;
      bool supported = true;
      CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &why) == VK_SUCCESS);
      CHECK(!supported && why && strstr(why, "CullPrimitive"));
      CHECK(nir->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE);
      ralloc_free(nir);
   }
   glsl_type_singleton_decref();
   return 0;
}
