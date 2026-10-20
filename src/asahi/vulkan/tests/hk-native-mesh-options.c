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

static void
check_primitive_key(void)
{
   CHECK(hk_apple9_native_mesh_primitive_lowering_enabled());
   nir_shader *mesh = producer(true);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(mesh)));
   nir_store_per_primitive_output(&b, nir_imm_int(&b, 37),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 1,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID, .num_slots = 1});
   struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part out = {0};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &out, &info, &reason));
   CHECK(info.primitive_broadcast == VARYING_BIT_PRIMITIVE_ID);
   CHECK(out.info.apple9_varyings.mask[VARYING_SLOT_PRIMITIVE_ID] == 1);
   CHECK(out.info.apple9_varyings.group[VARYING_SLOT_PRIMITIVE_ID] == 1);
   free(out.binary);
   ralloc_free(mesh);
}

static void
check_workgroup_key(const uint16_t size[3], bool shared, unsigned expected)
{
   nir_shader *mesh = producer(true);
   memcpy(mesh->info.workgroup_size, size, sizeof(mesh->info.workgroup_size));
   if (shared) {
      nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(mesh)));
      mesh->info.shared_size = 28672;
      nir_store_shared(&b, nir_imm_int(&b, 123), nir_imm_int(&b, 28668),
                        .align_mul = 4, .write_mask = 1);
   }
   const struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(hk_apple9_native_mesh_threads(size) == expected);
   CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &part, &info, &reason)
         == (expected != 0));
   if (expected) {
      CHECK(!memcmp(info.logical_size, size, sizeof(info.logical_size)));
      CHECK(info.key.physical_threads == expected);
      CHECK(part.info.workgroup_size[0] == expected);
      CHECK(part.info.workgroup_size[1] == 1 && part.info.workgroup_size[2] == 1);
      CHECK(hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
      struct agx_apple9_launch_parameters launch;
      CHECK(hk_apple9_native_launch_parameters(&launch, &part.info, &info,
         HK_APPLE9_NATIVE_MESH, 0, 0x1000));
      CHECK(launch.threadgroup_memory_bytes == (shared ? 32768 : 256));
      CHECK(info.shared_size_B == (shared ? 28672 : 0));
   } else {
      CHECK(reason && !part.binary);
   }
   free(part.binary);
   ralloc_free(mesh);
}

static unsigned
io_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static void
check_output_array(bool primitive, bool reverse)
{
   nir_shader *mesh = producer(true);
   mesh->info.io_lowered = false;
   const struct glsl_type *type = glsl_array_type(
      glsl_array_type(glsl_vec4_type(), 3, 0), primitive ? 1 : 3, 0);
   nir_variable *array = nir_variable_create(mesh, nir_var_shader_out, type,
                                             "output_array");
   array->data.location = VARYING_SLOT_VAR8;
   array->data.per_primitive = primitive;
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(mesh)));
   for (unsigned v = 0; v < (primitive ? 1 : 3); ++v) {
      for (unsigned j = 0; j < 3; ++j) {
         nir_def *index = nir_load_global(&b, 1, 32,
            nir_imm_int64(&b, INPUT_ADDR + j * 4), .align_mul = 4);
         nir_deref_instr *deref = nir_build_deref_array_imm(&b,
            nir_build_deref_var(&b, array), v);
         deref = nir_build_deref_array(&b, deref, index);
         nir_def *values[4];
         for (unsigned c = 0; c < 4; ++c)
            values[c] = nir_imm_int(&b, 0x80000 + 100 * v + 10 * j + c);
         nir_store_deref(&b, deref, nir_vec(&b, values, 4), 15);
      }
   }
   /* Exercise the normal deref-to-IO passes, then the actual driver compiler
    * boundary. The recording Vulkan regression checks pipeline wiring. */
   CHECK(nir_lower_indirect_derefs_to_if_else_trees(
      mesh, nir_var_shader_out, UINT32_MAX));
   nir_lower_io(mesh, nir_var_shader_out, io_size, 0);
   nir_opt_constant_folding(mesh);
   nir_opt_dce(mesh);
   nir_shader_gather_info(mesh, nir_shader_get_entrypoint(mesh));
   const struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(hk_apple9_compile_native_mesh_program(mesh, &key, &part, &info, &reason));
   CHECK(hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
   uint32_t indices[3];
   for (unsigned j = 0; j < 3; ++j)
      indices[j] = reverse ? 2 - j : j;
   struct machine m = {0};
   m.region[5] = (struct region){INPUT_ADDR, sizeof(indices), (uint8_t *)indices, NULL};
   nir_function_impl *impl = nir_shader_get_entrypoint(mesh);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   for (unsigned v = 0; v < 3; ++v) {
      for (unsigned j = 0; j < 3; ++j) {
         for (unsigned c = 0; c < 4; ++c) {
            unsigned slot = VARYING_SLOT_VAR8 + indices[j];
            unsigned channel = 4 + agx_apple9_varying_index(
               &part.info.apple9_varyings, slot, c);
            CHECK(m.submitted_position[v][channel] ==
                  0x80000 + 100 * (primitive ? 0 : v) + 10 * j + c);
            CHECK(part.info.apple9_varyings.group[slot] == (primitive ? 1 : 0));
         }
      }
   }
   free(m.value);
   free(part.binary);
   ralloc_free(mesh);
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
   check_primitive_key();
   for (unsigned primitive = 0; primitive < 2; ++primitive)
      for (unsigned reverse = 0; reverse < 2; ++reverse)
         check_output_array(primitive, reverse);
   const uint16_t shapes[][3] = {
      {1, 1, 1}, {16, 1, 1}, {17, 1, 1}, {32, 1, 1},
      {33, 1, 1}, {64, 1, 1}, {127, 1, 1}, {128, 1, 1},
      {1, 128, 1}, {1, 1, 128}, {4, 4, 4}, {2, 4, 16},
      {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {129, 1, 1},
      {1, 129, 1}, {1, 1, 129}, {16, 16, 1}, {65535, 65535, 65535},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i)
      for (unsigned shared = 0; shared < 2; ++shared)
         check_workgroup_key(shapes[i], shared, i < 2 ? 16 : i < 12 ? 32 : 0);
   CHECK(!hk_apple9_native_mesh_threads(NULL));
   glsl_type_singleton_decref();
   puts("hk native key: halfz/W, interpolation, output arrays, T128 shapes and shared allocation pass");
   return 0;
}
