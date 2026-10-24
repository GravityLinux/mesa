/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Actual driver caller, exact qualification classes and launch admission.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_shader.h"
#include "hk_physical_device.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)

static nir_shader *
source(unsigned v, unsigned p, unsigned t)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "driver packed publication");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = t;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = v;
   b.shader->info.mesh.max_primitives_out = p;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, v),
                                    nir_imm_int(&b, p), nir_imm_int(&b, 0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b, nir_ult_imm(&b, lane, v));
   nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 0, 0, .5, 1), lane,
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_pop_if(&b, NULL);
   for (unsigned i = 0; i < p; ++i) {
      nir_push_if(&b, nir_ieq_imm(&b, lane, i % t));
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, i), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
check(nir_shader *nir, bool expected)
{
   struct hk_apple9_graphics_key key = {0};
   struct blob before, after;
   blob_init(&before); blob_init(&after);
   nir_serialize(&before, nir, false);
   bool supported = !expected;
   const char *reason = NULL;
   CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &reason) == VK_SUCCESS);
   CHECK(supported == expected);
   nir_serialize(&after, nir, false);
   CHECK(!before.out_of_memory && !after.out_of_memory && before.size == after.size);
   CHECK(!memcmp(before.data, after.data, before.size));
   blob_finish(&before); blob_finish(&after);

   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info body;
   CHECK(hk_apple9_compile_native_mesh_program(nir, &key, &part, &body, &reason) == expected);
   if (expected) {
      CHECK(hk_apple9_native_shader_valid(&part.info, &body, HK_APPLE9_NATIVE_MESH));
      struct agx_apple9_launch_parameters launch;
      CHECK(hk_apple9_native_launch_parameters(&launch, &part.info, &body,
         HK_APPLE9_NATIVE_MESH, 0, 0x1000));
      CHECK(launch.threadgroup_memory_bytes >= body.layout.local_size_B);
      struct agx_apple9_mesh_state_parameters state = {
         .layout = &body.layout, .threads = body.key.physical_threads,
         .has_object_stage = true,
      };
      uint32_t words[AGX_APPLE9_MESH_STATE_WORDS];
      CHECK(agx_apple9_mesh_state_build(words, &state));
      CHECK(words[10] == ((body.layout.max_primitives - 1) |
                           ((DIV_ROUND_UP(body.layout.max_primitives, 16) - 1) << 16)));
      CHECK(words[11] == (0x400800 | body.key.physical_threads));
      CHECK(words[13] == 1 + DIV_ROUND_UP(body.layout.max_vertices, 4) * body.layout.vertex_components);
      if (body.layout.max_primitives > 1) {
         struct agx_apple9_mesh_body_info bad = body;
         bad.layout.position_B += 16;
         CHECK(!hk_apple9_native_shader_valid(&part.info, &bad, HK_APPLE9_NATIVE_MESH));
         bad = body; bad.layout.max_primitives = 15;
         CHECK(!hk_apple9_native_shader_valid(&part.info, &bad, HK_APPLE9_NATIVE_MESH));
         bad = body; bad.logical_size[0] /= 2;
         CHECK(!hk_apple9_native_shader_valid(&part.info, &bad, HK_APPLE9_NATIVE_MESH));
      }
   }
   free(part.binary); ralloc_free(nir);
}

static void
selection(void)
{
   const char *invalid[] = {NULL, "", "1", "true", "all", "*", "T16V3P2,",
      "T16V3P2,,T16V4P2", " T16V3P2", "T16V3P2,T16V16P15", "T32V3P2", "T64V3P1"};
   for (unsigned i = 0; i < ARRAY_SIZE(invalid); ++i)
      CHECK(!hk_apple9_packed_mesh_shapes_parse(invalid[i]));
   uint64_t mask = hk_apple9_packed_mesh_shapes_parse("T16V3P2,T32V32P42");
   CHECK(mask);
   CHECK(hk_apple9_packed_mesh_class_selected(mask, 16, 3, 2));
   CHECK(hk_apple9_packed_mesh_class_selected(mask, 32, 32, 42));
   CHECK(!hk_apple9_packed_mesh_class_selected(mask, 16, 16, 16));
   CHECK(!hk_apple9_packed_mesh_class_selected(mask, 32, 3, 2));
   CHECK(!hk_apple9_packed_mesh_class_selected(~0ull, 16, 16, 15));
   CHECK(!hk_apple9_packed_mesh_class_selected(~0ull, 64, 3, 1));
}

int
main(int argc, char **argv)
{
   CHECK(argc == 2);
   glsl_type_singleton_init_or_ref();
   selection();
   bool on = hk_apple9_native_mesh_enabled();
   bool subset = !strcmp(argv[1], "subset"), capacity = !strcmp(argv[1], "capacity");
   bool v16 = !strcmp(argv[1], "v16");
   CHECK((hk_apple9_native_mesh_packed_shapes() != 0) == (subset || capacity || v16));
   check(source(3, 1, 1), on); /* Existing P1 path, unchanged logical expansion. */
   check(source(3, 2, 16), subset);
   check(source(4, 2, 16), subset);
   check(source(16, 21, 16), capacity);
   check(source(32, 42, 32), capacity);
   /* Critical review case: V<=W alone never enables V16/P16. */
   check(source(16, 16, 16), v16);
   check(source(16, 15, 16), false);
   check(source(3, 2, 32), false);
   check(source(3, 2, 8), false);
   check(source(3, 2, 64), false);
   nir_shader *nir = source(3, 2, 16);
   nir->info.workgroup_size[0] = 8; nir->info.workgroup_size[1] = 2;
   check(nir, false);
   nir = source(3, 2, 16);
   nir->info.task_payload_size = 16; check(nir, false);
   nir = source(3, 2, 16); nir->info.shared_size = 16; check(nir, false);
   nir = source(3, 2, 16); nir->info.mesh.primitive_type = MESA_PRIM_LINES; check(nir, false);
   nir = source(3, 2, 16);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_store_per_vertex_output(&b, nir_imm_float(&b, 1), nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   check(nir, false);
   struct hk_physical_device pdev = {0};
   printf("compiler_flags=%016llx\n", (unsigned long long)hk_physical_device_compiler_flags(&pdev));
   glsl_type_singleton_decref();
   puts("exact classes, immutable preflight, normal compiler caller and launch/state checks pass");
   return 0;
}
