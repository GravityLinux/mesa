/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Host tests of normal formation NIR with independent DATA and raster checks.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_apple9_mesh_formation.h"
#include "agx_nir.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)
#include "apple9-mesh-executor.h"
#include "mesh-formation-fixtures.h"
#include "mesh-h1-fixtures.h"

#include "apple9-mesh-data.h"

static nir_shader *
source(const struct formation_fixture *f)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "normal_mesh_fixture_inputs");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, f->vertices),
                                      nir_imm_int(&b, f->primitives),
                                      nir_imm_int(&b, 0));
   for (unsigned i = 0; i < 3; ++i) {
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), i));
      nir_def *v[4];
      for (unsigned c = 0; c < 4; ++c)
         v[c] = nir_imm_int(&b, f->positions[4 * i + c]);
      nir_store_per_vertex_output(
         &b, nir_vec(&b, v, 4), nir_imm_int(&b, i), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, f->indices[0], f->indices[1], f->indices[2]),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
   nir_pop_if(&b, NULL);
   return b.shader;
}

enum { PRESSURE_VALUES = 24 };
#define PRESSURE_INPUT  0x24000ull
#define PRESSURE_OUTPUT 0x28000ull

/* Test-only ordinary loads/stores keep independently supplied values live
 * across completion. No fixed GPRs or compiler allocation hooks are involved. */
static void
add_pressure(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_instr *control = NULL;
   nir_foreach_block(block, impl)
      nir_foreach_instr(i, block)
         if (i->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(i)->intrinsic ==
                nir_intrinsic_mesh_control_apple9_agx)
            control = i;
   CHECK(control);
   nir_builder b = nir_builder_at(nir_before_instr(control));
   nir_def *live[PRESSURE_VALUES];
   for (unsigned i = 0; i < PRESSURE_VALUES; ++i)
      live[i] =
         nir_load_global(&b, 1, 32, nir_imm_int64(&b, PRESSURE_INPUT + 4 * i),
                         .align_mul = 4, .access = ACCESS_VOLATILE);
   nir_barrier(&b, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   b.cursor = nir_after_impl(impl);
   nir_barrier(&b, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   nir_def *address = nir_iadd_imm(
      &b,
      nir_u2u64(&b, nir_imul_imm(&b, nir_load_local_invocation_index(&b),
                                 4 * PRESSURE_VALUES)),
      PRESSURE_OUTPUT);
   for (unsigned i = 0; i < PRESSURE_VALUES; ++i)
      nir_store_global(&b, live[i], nir_iadd_imm(&b, address, 4 * i),
                       .align_mul = 4);
   nir_progress(true, impl, nir_metadata_none);
   nir_validate_shader(nir, "test pressure across mesh completion");
}

static const char *export_dir;

static void
run_fixture(const struct formation_fixture *f, unsigned placement, bool ready,
            bool pressure, bool count_drops)
{
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   nir_shader *nir = source(f);
   CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &info, &reason));
   if (pressure)
      add_pressure(nir);
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   struct machine m = {0};
   uint32_t pressure_input[PRESSURE_VALUES],
      pressure_output[LANES][PRESSURE_VALUES] = {{0}};
   uint8_t pressure_written[sizeof(pressure_output)] = {0};
   for (unsigned i = 0; i < PRESSURE_VALUES; ++i)
      pressure_input[i] = 0x12340000 + 17 * i;
   m.region[4] = (struct region){PRESSURE_OUTPUT, sizeof(pressure_output),
                                 (uint8_t *)pressure_output, pressure_written};
   m.region[5] = (struct region){PRESSURE_INPUT, sizeof(pressure_input),
                                 (uint8_t *)pressure_input, NULL};
   uint32_t drops = 0;
   uint8_t drops_written[4] = {0};
   m.region[7] = (struct region){0x30000, 4, (uint8_t *)&drops, drops_written};
   m.formation_inputs[AGX_MESH_FORMATION_DROPS_ADDRESS] =
      count_drops ? 0x30000 : 0;
   for (unsigned i = 0; i < 4; ++i) {
      m.region[i] = (struct region){
         UINT64_C(0x1200000000) + placement * 0x80000 + i * 0x10000, 256,
         malloc(256), calloc(1, 256)};
      CHECK(m.region[i].data && m.region[i].written);
      memset(m.region[i].data, 0xa0 + i, 256);
      m.formation_inputs[AGX_MESH_GEOMETRY_ADDRESS + i] = m.region[i].base;
   }
   for (unsigned i = 0; i < 3; ++i)
      m.formation_inputs[AGX_MESH_GEOMETRY_ID + i] = f->ids[i];
   for (unsigned i = 0; i < 8; ++i)
      m.formation_inputs[AGX_MESH_FRAGMENT_LAUNCH + i] = f->state[i];
   for (unsigned i = 0; i < 6; ++i)
      m.formation_inputs[AGX_MESH_VIEWPORT_SCALE_X + i] = f->viewport[i];
   m.formation_inputs[AGX_MESH_RESERVATION_READY] = ready;
   m.region[6] = (struct region){INPUT_ADDR, sizeof(m.formation_inputs),
                                 (uint8_t *)m.formation_inputs, NULL};
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   uint32_t xyz[3][3];
   bool dropped;
   bool formed = source_geometry(f, xyz, &dropped) && ready;
   CHECK(drops == (ready && dropped && count_drops));
   CHECK(m.barriers == 2 && m.fences == (formed ? 9 : 7) + 2 * pressure);
   CHECK(m.controls == ALL_LANES && m.scalars == 1);
   for (unsigned c = 0; c < 4; ++c)
      CHECK(m.channels[c] == 7);
   if (pressure) {
      for (unsigned lane = 0; lane < LANES; ++lane)
         CHECK(!memcmp(pressure_output[lane], pressure_input,
                       sizeof(pressure_input)));
      for (unsigned i = 0; i < sizeof(pressure_written); ++i)
         CHECK(pressure_written[i]);
   }
   unsigned lengths[4] = {0, formed ? 12 : 0, formed ? 32 : 0, ready ? 48 : 0};
   if (formed) {
      while (lengths[0] < 256 && m.region[0].written[lengths[0]])
         ++lengths[0];
      check_geometry(f, m.region[0].data, lengths[0], xyz);
      CHECK(references_match(f, &m));
      m.region[2].data[16] ^= 1;
      CHECK(!references_match(f, &m));
      m.region[2].data[16] ^= 1;
   } else if (ready) {
      for (unsigned i = 0; i < 48; ++i)
         CHECK(m.region[3].data[i] == 0);
   }
   for (unsigned i = 0; i < 4; ++i) {
      for (unsigned j = 0; j < 256; ++j) {
         CHECK(m.region[i].written[j] == (j < lengths[i]));
         if (j >= lengths[i])
            CHECK(m.region[i].data[j] == 0xa0 + i);
      }
      free(m.region[i].data);
      free(m.region[i].written);
   }
   free(m.value);
   struct agx_shader_part code = {0};
   ralloc_free(nir);
   nir = source(f);
   bool ok = agx_nir_lower_apple9_native_mesh(nir, true, &info, &reason);
   if (ok && pressure)
      add_pressure(nir);
   if (ok)
      ok = agx_compile_apple9_mesh_lowered(nir, &info.key, &code, &reason);
   if (!ok)
      fprintf(stderr, "compile failed: %s\n", reason);
   CHECK(ok);
   CHECK(code.info.apple9_body_only && code.info.binary_size);
   CHECK(!info.requires_mesh_completion && info.requires_native_entry);
   CHECK(info.layout.primitive_count_B == 16 && info.layout.local_size_B == 96);
   if (export_dir && ready && !pressure && !placement &&
       !strncmp(f->name, "H1_", 3)) {
      char path[1024];
      CHECK(snprintf(path, sizeof(path), "%s/%s.bin", export_dir, f->name) <
            sizeof(path));
      FILE *file = fopen(path, "wb");
      CHECK(file && fwrite(code.binary, 1, code.info.binary_size, file) ==
                       code.info.binary_size);
      CHECK(!fclose(file));
      CHECK(snprintf(path, sizeof(path), "%s/%s.json", export_dir, f->name) <
            sizeof(path));
      file = fopen(path, "w");
      CHECK(file);
      fprintf(
         file,
         "{\"native_launch_required\":true,\"binary_size\":%u,"
         "\"gprs\":%u,\"scratch\":%u,\"atomic_frame\":%u,"
         "\"preamble_offset\":%u,\"preamble_size\":%u,"
         "\"local_output_bytes\":%u,\"argument_words\":%u,"
         "\"output_root\":%u,\"draw_uniform\":%u,\"abi_version\":%u,\"draw_bytes\":%zu}\n",
         code.info.binary_size, code.info.nr_gprs, code.info.scratch_size,
         code.info.apple9_atomic_frame_size, code.info.apple9_preamble_offset,
         code.info.apple9_preamble_size, info.layout.local_size_B,
         info.argument_uniform_words, info.key.output_root,
         AGX_APPLE9_MESH_DRAW_UNIFORM, AGX_APPLE9_MESH_ABI_VERSION,
         sizeof(struct agx_apple9_mesh_draw));
      CHECK(!fclose(file));
   }
   printf(
      "fixture=%s placement=%u ready=%u pressure=%u drops=%u records=decoded units=1+4+1 "
      "gprs=%u scratch=%u bytes=%u\n",
      f->name, placement, ready, pressure, drops, code.info.nr_gprs,
      code.info.scratch_size, code.info.binary_size);
   free(code.binary);
   ralloc_free(nir);
}

static void
reject_unsupported(void)
{
   for (unsigned variant = 0; variant < 5; ++variant) {
      struct formation_fixture f = h1_fixtures[0];
      if (variant == 1)
         f.primitives = 2;
      nir_shader *nir = source(&f);
      if (variant == 2)
         nir->info.workgroup_size[0] = 32;
      if (variant == 3)
         nir->info.task_payload_size = 16;
      if (variant == 4)
         nir->info.mesh.max_vertices_out = 4;
      struct agx_shader_part code = {0};
      struct agx_apple9_mesh_body_info info = {0};
      const char *reason = NULL;
      CHECK(!agx_compile_apple9_native_mesh(nir, variant != 0, &code, &info,
                                            &reason));
      CHECK(!code.binary && reason);
      if (!variant)
         CHECK(strstr(reason, "disabled"));
      ralloc_free(nir);
   }
}

int
main(int argc, char **argv)
{
   CHECK(argc == 1 || (argc == 3 && !strcmp(argv[1], "--export")));
   if (argc == 3)
      export_dir = argv[2];
   glsl_type_singleton_init_or_ref();
   reject_unsupported();
   for (unsigned i = 0; i < ARRAY_SIZE(formation_fixtures); ++i) {
      const struct formation_fixture *f = &formation_fixtures[i];
      if (f->vertices != 3 || f->primitives != 1)
         continue;
      for (unsigned placement = 0; placement < 2; ++placement)
         run_fixture(f, placement, true, false, true);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(h1_fixtures); ++i)
      for (unsigned placement = 0; placement < 2; ++placement)
         run_fixture(&h1_fixtures[i], placement, true, false, true);
   run_fixture(&h1_fixtures[0], 0, false, false, true);
   run_fixture(&h1_fixtures[0], 0, true, true, true);
   for (unsigned sign = 0; sign < 2; ++sign) {
      struct formation_fixture boundary = h1_fixtures[0];
      boundary.name = sign ? "CLIP_NEGATIVE_X" : "CLIP_POSITIVE_X";
      boundary.positions[0] = as_uint(sign ? -1.0f : 1.0f);
      run_fixture(&boundary, 0, true, false, true);
      run_fixture(&boundary, 0, true, false, false);
      run_fixture(&boundary, 0, false, false, true);
   }
   glsl_type_singleton_decref();
   return 0;
}
