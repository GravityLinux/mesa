/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_shader.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); exit(1); } } while (0)
#include "asahi/compiler/test/apple9-mesh-executor.h"

static const unsigned slots[] = {VARYING_SLOT_VAR0, VARYING_SLOT_VAR3, VARYING_SLOT_VAR5};
static const float positions[3][4] = {
   {-.5, -.5, 0, 1}, {1, -1, 1, 2}, {0, .25, .125, .5},
};

static nir_shader *
producer(bool mesh)
{
   nir_builder b = nir_builder_init_simple_shader(
      mesh ? MESA_SHADER_MESH : MESA_SHADER_VERTEX, &agx_nir_options,
      "hk native key producer");
   b.shader->info.io_lowered = true;
   if (mesh) {
      b.shader->info.workgroup_size[0] = b.shader->info.workgroup_size[1] =
         b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.max_vertices_out = 3;
      b.shader->info.mesh.max_primitives_out = 1;
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                         nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   }
   for (unsigned v = 0; v < (mesh ? 3 : 1); ++v) {
      nir_def *position = nir_imm_vec4(&b, positions[v][0], positions[v][1],
                                          positions[v][2], positions[v][3]);
      if (mesh)
         nir_store_per_vertex_output(&b, position, nir_imm_int(&b, v),
            nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      else
         nir_store_output(&b, position, nir_imm_int(&b, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      for (unsigned i = 0; i < ARRAY_SIZE(slots); ++i) {
         nir_def *value = nir_imm_float(&b, .25f * (i + v + 1));
         if (mesh)
            nir_store_per_vertex_output(&b, value, nir_imm_int(&b, v),
               nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
               .io_semantics = {.location = slots[i], .num_slots = 1});
         else
            nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 1,
               .src_type = nir_type_float32,
               .io_semantics = {.location = slots[i], .num_slots = 1});
      }
   }
   if (mesh)
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
consumer(unsigned flat, unsigned linear)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "hk native key consumer");
   b.shader->info.io_lowered = true;
   nir_def *values[4];
   for (unsigned i = 0; i < ARRAY_SIZE(slots); ++i) {
      if (i == flat)
         values[i] = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = slots[i], .num_slots = 1});
      else
         values[i] = nir_load_interpolated_input(&b, 1, 32,
            nir_load_barycentric_pixel(&b, 32,
               .interp_mode = i == linear ? INTERP_MODE_NOPERSPECTIVE : INTERP_MODE_SMOOTH),
            nir_imm_int(&b, 0), .dest_type = nir_type_float32,
            .io_semantics = {.location = slots[i], .num_slots = 1});
   }
   values[3] = nir_imm_float(&b, 1);
   nir_store_output(&b, nir_vec(&b, values, 4), nir_imm_int(&b, 0),
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
check_key(bool halfz, unsigned flat, unsigned linear)
{
   const struct hk_apple9_graphics_key key = {
      .clip_negative_one_to_one = halfz,
      .vertex.outputs_flat = BITFIELD64_BIT(slots[flat]),
      .vertex.outputs_linear = BITFIELD64_BIT(slots[linear]),
   };
   nir_shader *mesh = producer(true), *vs = producer(false);
   struct agx_shader_part native = {0}, vertex = {0}, fragment = {0};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &native, &info, &reason));
   CHECK(agx_compile_apple9_vertex_inputs(vs, &key.vertex, &vertex, &reason));
   for (unsigned i = 0; i < ARRAY_SIZE(slots); ++i) {
      unsigned expected = i == flat ? 1 : i == linear ? 2 : 0;
      CHECK(native.info.apple9_varyings.group[slots[i]] == expected);
      CHECK(vertex.info.apple9_varyings.group[slots[i]] == expected);
   }
   CHECK(!memcmp(&native.info.apple9_varyings, &vertex.info.apple9_varyings,
                  sizeof(native.info.apple9_varyings)));
   nir_shader *fs = consumer(flat, linear);
   CHECK(agx_compile_apple9_fragment_inputs(fs, &native.info.apple9_varyings,
                                           &fragment, &reason));
   CHECK(!memcmp(&fragment.info.apple9_varyings, &vertex.info.apple9_varyings,
                  sizeof(vertex.info.apple9_varyings)));

   /* Execute the lowered NIR returned by the real hk compilation call. This
    * checks the key's effect on publication, independently of its options
    * construction and without a device or shader-upload mock. */
   struct machine m = {0};
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   for (unsigned v = 0; v < 3; ++v) {
      float z, w;
      memcpy(&z, &m.submitted_position[v][2], 4);
      memcpy(&w, &m.submitted_position[v][3], 4);
      float expected = halfz ? fmaf(-positions[v][2], .5f,
                                    fmaf(positions[v][3], .5f, positions[v][2]))
                             : positions[v][2];
      CHECK(z == expected && w == positions[v][3]);
   }
   free(m.value);
   free(native.binary);
   free(vertex.binary);
   free(fragment.binary);
   ralloc_free(mesh);
   ralloc_free(vs);
   ralloc_free(fs);
}

int main(void)
{
   CHECK(hk_apple9_native_mesh_enabled());
   CHECK(hk_apple9_native_mesh_attributes_enabled());
   glsl_type_singleton_init_or_ref();
   for (unsigned halfz = 0; halfz < 2; ++halfz) {
      check_key(halfz, 0, 1);
      check_key(halfz, 2, 0);
   }
   glsl_type_singleton_decref();
   puts("hk native key: halfz/W values and VS/mesh/FS interpolation groups pass");
   return 0;
}
