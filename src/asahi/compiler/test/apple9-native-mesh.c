/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Host tests of local output publication and hardware mesh completion.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_apple9_mesh.h"
#include "agx_nir.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)
#include "apple9-mesh-executor.h"
#include "mesh-native-fixtures.h"
#include "mesh-clip-fixtures.h"


static nir_shader *
source(const struct native_mesh_fixture *f, unsigned threads)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "normal_mesh_fixture_inputs");
   b.shader->info.workgroup_size[0] = threads;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, f->vertices),
                                      nir_imm_int(&b, f->primitives),
                                      nir_imm_int(&b, 0));
   for (unsigned i = 0; i < 3; ++i) {
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), i % threads));
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
run_fixture(const struct native_mesh_fixture *f, bool pressure, bool clip_halfz,
            unsigned threads)
{
   const struct agx_apple9_native_mesh_options options = {.clip_halfz = clip_halfz};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   nir_shader *nir = source(f, threads);
   if (clip_halfz)
      nir_lower_io_to_scalar(nir, nir_var_shader_out, NULL, NULL);
   CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &options, &info, &reason));
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
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.barriers == 1 + clip_halfz && m.fences == 6 + 2 * pressure);
   CHECK(m.controls == ALL_LANES && m.scalars == 1);
   CHECK(m.writes == (pressure ? LANES * PRESSURE_VALUES : 0));
   for (unsigned c = 0; c < 4; ++c) {
      CHECK(m.channels[c] == 7);
      for (unsigned v = 0; v < 3; ++v) {
         uint32_t value;
         memcpy(&value, m.local + info.layout.position_B + 16 * c + 4 * v, 4);
         uint32_t expected = f->positions[4 * v + c];
         if (clip_halfz && c == 2) {
            float z, w;
            memcpy(&z, &f->positions[4 * v + 2], 4);
            memcpy(&w, &f->positions[4 * v + 3], 4);
            float converted = fmaf(-z, .5f, fmaf(w, .5f, z));
            memcpy(&expected, &converted, 4);
         }
         CHECK(value == expected);
         CHECK(m.submitted_position[v][c] == expected);
      }
   }
   for (unsigned i = 0; i < 3; ++i)
      CHECK(m.local[i] == f->indices[i]);
   if (pressure) {
      for (unsigned lane = 0; lane < LANES; ++lane)
         CHECK(!memcmp(pressure_output[lane], pressure_input,
                       sizeof(pressure_input)));
      for (unsigned i = 0; i < sizeof(pressure_written); ++i)
         CHECK(pressure_written[i]);
   }
   free(m.value);
   struct agx_shader_part code = {0};
   ralloc_free(nir);
   nir = source(f, threads);
   if (clip_halfz)
      nir_lower_io_to_scalar(nir, nir_var_shader_out, NULL, NULL);
   bool ok = agx_nir_lower_apple9_native_mesh(nir, true, &options, &info, &reason);
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
   if (export_dir && !pressure && !clip_halfz) {
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
         "\"output_root\":%u,\"draw_uniform\":%u,\"abi_version\":%u}\n",
         code.info.binary_size, code.info.nr_gprs, code.info.scratch_size,
         code.info.apple9_atomic_frame_size, code.info.apple9_preamble_offset,
         code.info.apple9_preamble_size, info.layout.local_size_B,
         info.argument_uniform_words, info.key.output_root,
         AGX_APPLE9_MESH_DRAW_UNIFORM, AGX_APPLE9_MESH_ABI_VERSION);
      CHECK(!fclose(file));
   }
   printf("fixture=%s pressure=%u halfz=%u implicit=1 units=1+4+1 gprs=%u scratch=%u bytes=%u\n",
          f->name, pressure, clip_halfz, code.info.nr_gprs, code.info.scratch_size,
          code.info.binary_size);
   free(code.binary);
   ralloc_free(nir);
}

struct attribute {
   unsigned slot, component, group;
};

static nir_shader *
attribute_source(const struct attribute *attrs, unsigned count)
{
   nir_shader *nir = source(&h1_fixtures[0], 16);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   for (unsigned v = 0; v < 3; ++v) {
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), v));
      for (unsigned i = 0; i < count; ++i)
         nir_store_per_vertex_output(
            &b, nir_imm_int(&b, 0x3f000000 + 256 * i + v),
            nir_imm_int(&b, v), nir_imm_int(&b, 0),
            .component = attrs[i].component, .write_mask = 1,
            .src_type = nir_type_float32,
            .io_semantics = {.location = attrs[i].slot, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   return nir;
}

static void
run_attributes(const char *name, const struct attribute *attrs, unsigned count)
{
   struct agx_apple9_native_mesh_options options = {.attributes = true};
   for (unsigned i = 0; i < count; ++i) {
      if (attrs[i].group == 1)
         options.outputs_flat |= BITFIELD64_BIT(attrs[i].slot);
      if (attrs[i].group == 2)
         options.outputs_linear |= BITFIELD64_BIT(attrs[i].slot);
   }
   const char *reason = NULL;
   struct agx_apple9_mesh_body_info info;
   nir_shader *nir = attribute_source(attrs, count);
   CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &options, &info, &reason));
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   struct machine m = {0};
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.barriers == 1 && m.fences == 6 + count);
   CHECK(m.controls == ALL_LANES && m.scalars == 1 && !m.writes);
   /* Derive packed order independently of the compiler's indexing helper. */
   unsigned channel[124];
   for (unsigned i = 0; i < count; ++i) {
      channel[i] = 4;
      for (unsigned j = 0; j < count; ++j)
         channel[i] += attrs[j].group < attrs[i].group ||
            (attrs[j].group == attrs[i].group &&
             (attrs[j].slot < attrs[i].slot ||
              (attrs[j].slot == attrs[i].slot &&
               attrs[j].component < attrs[i].component)));
      CHECK(m.channels[channel[i]] == 7);
      for (unsigned v = 0; v < 3; ++v)
         CHECK(m.submitted_position[v][channel[i]] == 0x3f000000 + 256 * i + v);
   }
   free(m.value);
   ralloc_free(nir);
   nir = attribute_source(attrs, count);
   struct agx_shader_part code = {0};
   CHECK(agx_compile_apple9_native_mesh(nir, true, &options, &code, &info, &reason));
   CHECK(code.info.apple9_varyings.count == count);
   CHECK(info.layout.local_size_B == 96 + 16 * count);
   for (unsigned i = 0; i < count; ++i) {
      CHECK(code.info.apple9_varyings.mask[attrs[i].slot] & BITFIELD_BIT(attrs[i].component));
      CHECK(code.info.apple9_varyings.group[attrs[i].slot] == attrs[i].group);
   }
   printf("fixture=%s pressure=0 halfz=0 gprs=%u channels=", name, code.info.nr_gprs);
   for (unsigned c = 0; c < 4; ++c)
      printf("%s%u:%u", c ? "," : "", c, 32 + 16 * c);
   /* Publication walks local semantic order, even when channel order differs. */
   for (unsigned slot = VARYING_SLOT_VAR0; slot <= VARYING_SLOT_VAR31; ++slot) {
      for (unsigned c = 0; c < 4; ++c) {
         for (unsigned i = 0; i < count; ++i) {
            if (attrs[i].slot == slot && attrs[i].component == c)
               printf(",%u:%u", channel[i], 32 + 16 * info.layout.vertex_component[slot][c]);
         }
      }
   }
   puts("");
   free(code.binary);
   ralloc_free(nir);
   nir = attribute_source(attrs, count);
   CHECK(!agx_compile_apple9_native_mesh(nir, true, NULL, &code, &info, &reason));
   CHECK(!code.binary && strstr(reason, "attributes development path is disabled"));
   ralloc_free(nir);
}

static void
check_attributes(void)
{
   for (unsigned g = 0; g < 3; ++g) {
      const struct attribute a = {VARYING_SLOT_VAR0, 0, g};
      run_attributes(g == 0 ? "ATTR_SMOOTH" : g == 1 ? "ATTR_FLAT" : "ATTR_LINEAR", &a, 1);
   }
   const struct attribute mixed[] = {
      {VARYING_SLOT_VAR0, 3, 2}, {VARYING_SLOT_VAR2, 1, 1},
      {VARYING_SLOT_VAR4, 2, 0}, {VARYING_SLOT_VAR4, 0, 0},
      {VARYING_SLOT_VAR7, 1, 2},
   };
   run_attributes("ATTR_MIXED_SPARSE", mixed, ARRAY_SIZE(mixed));
   struct attribute wide[124];
   for (unsigned i = 0; i < ARRAY_SIZE(wide); ++i)
      wide[i] = (struct attribute){VARYING_SLOT_VAR0 + i / 4, i % 4, (i / 4) % 3};
   run_attributes("ATTR_LIMIT", wide, ARRAY_SIZE(wide));
   const struct agx_apple9_native_mesh_options options = {.attributes = true};
   const char *reason = NULL;
   struct agx_shader_part code = {0};
   struct agx_apple9_mesh_body_info info;
   for (unsigned primitive = 0; primitive < 2; ++primitive) {
      nir_shader *nir = primitive ? source(&h1_fixtures[0], 16)
                                  : attribute_source(wide, ARRAY_SIZE(wide));
      nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
      if (primitive)
         nir_store_per_primitive_output(
            &b, nir_imm_float(&b, 1), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      else
         nir_store_per_vertex_output(
            &b, nir_imm_float(&b, 1), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
            .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR31, .num_slots = 1});
      CHECK(!agx_compile_apple9_native_mesh(nir, true, &options, &code, &info, &reason));
      CHECK(!code.binary && strstr(reason, primitive ? "primitive attributes" : "124 user"));
      ralloc_free(nir);
   }
}

static void
reject_unsupported(void)
{
   for (unsigned variant = 0; variant < 5; ++variant) {
      struct native_mesh_fixture f = h1_fixtures[0];
      if (variant == 1)
         f.primitives = 2;
      nir_shader *nir = source(&f, 16);
      if (variant == 2)
         nir->info.workgroup_size[0] = 32;
      if (variant == 3)
         nir->info.task_payload_size = 16;
      if (variant == 4)
         nir->info.mesh.max_vertices_out = 4;
      struct agx_shader_part code = {0};
      struct agx_apple9_mesh_body_info info = {0};
      const char *reason = NULL;
      CHECK(!agx_compile_apple9_native_mesh(nir, variant != 0, NULL, &code, &info,
                                            &reason));
      CHECK(!code.binary && reason);
      if (!variant)
         CHECK(strstr(reason, "disabled"));
      ralloc_free(nir);
   }
}

/* Runtime counts, including a return before the publication and two static
 * calls in mutually exclusive uniform branches. Keep inputs in memory so
 * neither the NIR interpreter nor the compiler can fold the tested paths. */
static nir_shader *
conditional_counts(unsigned mode, unsigned threads)
{
   nir_shader *nir = source(&h1_fixtures[0], threads);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_intrinsic_instr *count = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_set_vertex_and_primitive_count)
            count = nir_instr_as_intrinsic(instr);
      }
   }
   CHECK(count);
   nir_builder b = nir_builder_at(nir_before_instr(&count->instr));
   nir_def *v = nir_load_global(&b, 1, 32, nir_imm_int64(&b, INPUT_ADDR),
                                .align_mul = 4);
   nir_def *p = nir_load_global(&b, 1, 32, nir_imm_int64(&b, INPUT_ADDR + 4),
                                .align_mul = 4);
   nir_def *enabled = nir_ine_imm(
      &b, nir_load_global(&b, 1, 32, nir_imm_int64(&b, INPUT_ADDR + 8),
                           .align_mul = 4), 0);
   if (mode == 1) {
      nir_push_if(&b, nir_inot(&b, enabled));
      nir_jump(&b, nir_jump_return);
      nir_pop_if(&b, NULL);
   }
   if (mode == 2) {
      nir_push_if(&b, enabled);
      nir_set_vertex_and_primitive_count(&b, v, p, nir_imm_int(&b, 0));
      nir_push_else(&b, NULL);
      nir_set_vertex_and_primitive_count(
         &b, nir_imm_int(&b, 0), nir_imm_int(&b, 0), nir_imm_int(&b, 0));
      nir_pop_if(&b, NULL);
      v = nir_bcsel(&b, enabled, v, nir_imm_int(&b, 0));
      p = nir_bcsel(&b, enabled, p, nir_imm_int(&b, 0));
      nir_instr_remove(&count->instr);
   } else {
      nir_src_rewrite(&count->src[0], v);
      nir_src_rewrite(&count->src[1], p);
      b.cursor = nir_after_instr(&count->instr);
   }
   /* No output access on the empty paths, as required by the API. */
   nir_push_if(&b, nir_ior(&b, nir_ieq_imm(&b, v, 0), nir_ieq_imm(&b, p, 0)));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
   return nir;
}

static void
check_conditional_counts(void)
{
   const uint32_t inputs[][3] = {
      {3, 1, 1}, {0, 0, 1}, {3, 0, 1}, {3, 1, 0}, {0, 0, 0},
   };
   for (unsigned threads = 1; threads <= 16; threads *= 16) {
      for (unsigned mode = 0; mode < 3; ++mode) {
         for (unsigned i = 0; i < ARRAY_SIZE(inputs); ++i) {
            nir_shader *nir = conditional_counts(mode, threads);
            const char *reason = NULL;
            struct agx_apple9_mesh_body_info info;
            CHECK(agx_nir_lower_apple9_native_mesh(nir, true, NULL, &info,
                                                   &reason));
            nir_lower_vars_to_ssa(nir);
            nir_opt_dce(nir);
            bool enabled = !mode || inputs[i][2];
            unsigned vertices = enabled ? inputs[i][0] : 0;
            unsigned primitives = vertices ? inputs[i][1] : 0;
            struct machine m = {
               .variable_counts = true, .expected_vertices = vertices,
               .expected_primitives = primitives,
            };
            /* Detect an uninitialized count on the early-return path. */
            memset(m.local, 0xa5, sizeof(m.local));
            m.region[5] = (struct region){INPUT_ADDR, sizeof(inputs[i]),
                                          (uint8_t *)inputs[i], NULL};
            nir_function_impl *impl = nir_shader_get_entrypoint(nir);
            m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
            CHECK(m.value);
            execute(&m, &impl->body, ALL_LANES);
            CHECK(m.controls == ALL_LANES && m.scalars == 1);
            for (unsigned c = 0; c < 4; ++c)
               CHECK(m.channels[c] == BITFIELD_MASK(vertices));
            uint32_t stored_vertices;
            memcpy(&stored_vertices, m.local + info.layout.vertex_count_B, 4);
            CHECK(stored_vertices == vertices);
            struct agx_shader_part code = {0};
            CHECK(agx_compile_apple9_mesh_lowered(nir, &info.key, &code, &reason));
            CHECK(code.binary && !code.info.scratch_size);
            free(code.binary);
            free(m.value);
            ralloc_free(nir);
         }
      }
   }
   puts("native counts: 30 dynamic/empty/conditional paths executed and compiled");
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--counts")) {
      glsl_type_singleton_init_or_ref();
      check_conditional_counts();
      glsl_type_singleton_decref();
      return 0;
   }
   bool clip = argc == 3 && !strcmp(argv[1], "--export-clip");
   CHECK(argc == 1 || (argc == 3 && (!strcmp(argv[1], "--export") || clip)));
   if (argc == 3)
      export_dir = argv[2];
   glsl_type_singleton_init_or_ref();
   if (clip) {
      for (unsigned i = 0; i < ARRAY_SIZE(clip_fixtures); ++i) {
         struct native_mesh_fixture f = h1_fixtures[0];
         f.name = clip_fixtures[i].name;
         memcpy(f.positions, clip_fixtures[i].positions, sizeof(f.positions));
         /* Preserve all four position components, including non-unit W.
          * Hardware clipping is deliberately outside this host interpreter. */
         run_fixture(&f, false, false, 16);
      }
      glsl_type_singleton_decref();
      return 0;
   }
   reject_unsupported();
   check_attributes();
   for (unsigned threads = 1; threads < 16; ++threads) {
      struct native_mesh_fixture f = h1_fixtures[0];
      char name[32];
      snprintf(name, sizeof(name), "PARTIAL_T%u", threads);
      f.name = name;
      run_fixture(&f, false, false, threads);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(h1_fixtures); ++i)
      run_fixture(&h1_fixtures[i], false, false, 16);
   run_fixture(&h1_fixtures[0], true, false, 16);
   for (unsigned i = 0; i < ARRAY_SIZE(clip_fixtures); ++i) {
      struct native_mesh_fixture f = h1_fixtures[0];
      f.name = clip_fixtures[i].name;
      memcpy(f.positions, clip_fixtures[i].positions, sizeof(f.positions));
      run_fixture(&f, false, false, 16);
      run_fixture(&f, false, true, 16);
   }
   glsl_type_singleton_decref();
   return 0;
}
