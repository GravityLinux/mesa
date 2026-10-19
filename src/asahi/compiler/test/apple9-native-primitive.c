/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_apple9_mesh.h"
#include "agx_nir.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%u: %s\n", __LINE__, #x); exit(1); } } while (0)
#include "apple9-mesh-executor.h"

static nir_shader *
source(unsigned threads, unsigned builtin, bool collision)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "primitive flat broadcast");
   b.shader->info.workgroup_size[0] = threads;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                      nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   for (unsigned v = 0; v < 3; ++v) {
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, v == 1 ? .5 : -.5,
         v == 2 ? .5 : -.5, .5, 1), nir_imm_int(&b, v), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(&b, nir_imm_int(&b, 100 + v),
         nir_imm_int(&b, v), nir_imm_int(&b, 0), .component = 1,
         .write_mask = 1, .src_type = nir_type_uint32,
         .io_semantics = {.location = collision ? VARYING_SLOT_VAR3 : VARYING_SLOT_VAR5, .num_slots = 1});
   }
   nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 2, 0, 1),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), threads - 1));
   nir_def *input = nir_load_global(&b, 2, 32, nir_imm_int64(&b, INPUT_ADDR), .align_mul = 4);
   nir_store_per_primitive_output(&b, nir_channel(&b, input, 0),
      nir_imm_int(&b, 0), nir_imm_int(&b, 1), .component = 2,
      .write_mask = 1, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_VAR2, .num_slots = 2});
   nir_store_per_primitive_output(&b, nir_imm_int(&b, 37),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 1,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = builtin, .num_slots = 1});
   nir_store_per_primitive_output(&b, nir_ine_imm(&b, nir_channel(&b, input, 1), 0),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 1,
      .src_type = nir_type_bool1,
      .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE, .num_slots = 1});
   nir_pop_if(&b, NULL);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
      .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_shader_out);
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_def *read = nir_load_per_primitive_output(&b, 1, 32, nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .component = 2, .dest_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 1});
   nir_store_per_primitive_output(&b, nir_ixor(&b, read, nir_imm_int(&b, 0x1234)),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .component = 3,
      .write_mask = 1, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 1});
   nir_pop_if(&b, NULL);
   return b.shader;
}

static void
check(unsigned threads, unsigned cull)
{
   nir_shader *nir = source(threads, VARYING_SLOT_PRIMITIVE_ID, false);
   const struct agx_apple9_native_mesh_options options = {
      .attributes = true, .primitive_broadcast = true,
      /* Primitive semantics must override an incompatible interpolation mask. */
      .outputs_linear = VARYING_BIT_VAR(3),
   };
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &options, &info, &reason));
   CHECK(info.primitive_broadcast == (VARYING_BIT_VAR(3) |
      VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_CULL_PRIMITIVE));
   CHECK(!info.layout.culling && !info.layout.primitive_components);
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   struct machine m = {0};
   uint32_t input[2] = {0x7fc01234, cull};
   m.region[5] = (struct region){INPUT_ADDR, sizeof(input), (uint8_t *)input, NULL};
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.barriers + m.subgroup_barriers == 3);
   CHECK(m.controls == ALL_LANES && m.scalars == 1);
   const uint8_t indices[] = {2, 0, 1};
   for (unsigned i = 0; i < 3; ++i)
      CHECK(m.local[i] == (cull ? 0 : indices[i]));
   for (unsigned v = 0; v < 3; ++v) {
      CHECK(m.submitted_position[v][4] == 100 + v);
      CHECK(m.submitted_position[v][5] == 37);
      CHECK(m.submitted_position[v][6] == cull);
      CHECK(m.submitted_position[v][7] == input[0]);
      CHECK(m.submitted_position[v][8] == (input[0] ^ 0x1234));
   }
   free(m.value);
   ralloc_free(nir);
   nir = source(threads, VARYING_SLOT_PRIMITIVE_ID, false);
   struct agx_shader_part code = {0};
   if (!agx_compile_apple9_native_mesh(nir, true, &options, &code, &info, &reason)) {
      fprintf(stderr, "compile: %s\n", reason); exit(1);
   }
   CHECK(code.info.apple9_varyings.count == 5);
   CHECK(code.info.apple9_varyings.group[VARYING_SLOT_VAR3] == 1);
   CHECK(code.info.apple9_varyings.group[VARYING_SLOT_PRIMITIVE_ID] == 1);
   CHECK(info.layout.local_size_B == 176);
   free(code.binary);
   ralloc_free(nir);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   for (unsigned t = 0; t < 3; ++t)
      for (unsigned cull = 0; cull < 2; ++cull)
         check(t == 0 ? 1 : t == 1 ? 3 : 16, cull);
   const unsigned builtins[] = {VARYING_SLOT_LAYER, VARYING_SLOT_VIEWPORT,
                                VARYING_SLOT_PRIMITIVE_SHADING_RATE};
   for (unsigned k = 0; k < 6; ++k) {
      nir_shader *nir = source(1, k < 3 ? builtins[k] : VARYING_SLOT_PRIMITIVE_ID, k == 3);
      struct agx_apple9_native_mesh_options options = {
         .attributes = k != 4, .primitive_broadcast = k != 5,
      };
      struct agx_shader_part code = {0};
      struct agx_apple9_mesh_body_info info;
      const char *reason = NULL;
      CHECK(!agx_compile_apple9_native_mesh(nir, true, &options, &code, &info, &reason));
      CHECK(reason && !code.binary);
      ralloc_free(nir);
   }
   glsl_type_singleton_decref();
   puts("primitive broadcast: T1/T3/T16, cull true/false, sparse raw bits, cross-lane output reads, builtin/gate/alias rejection");
   return 0;
}
