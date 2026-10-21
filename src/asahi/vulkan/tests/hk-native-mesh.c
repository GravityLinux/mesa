/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_apple9_mesh.h"
#include "agx_pack.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); exit(1); } } while (0)

static nir_shader *
mesh_source(unsigned attributes)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "native runtime contract");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                      nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   for (unsigned i = 0; i < 3; ++i)
      nir_store_per_vertex_output(
         &b, nir_imm_vec4(&b, i == 0 ? -.5 : i == 1 ? .5 : 0,
                          i == 2 ? .5 : -.5, .5, 1),
         nir_imm_int(&b, i), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   for (unsigned i = 0; i < attributes; ++i)
      for (unsigned v = 0; v < 3; ++v)
         nir_store_per_vertex_output(
            &b, nir_imm_float(&b, i + v), nir_imm_int(&b, v), nir_imm_int(&b, 0),
            .component = i % 4, .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + i / 4, .num_slots = 1});
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .write_mask = 7, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   return b.shader;
}

static void
check_contract(bool object, unsigned attributes)
{
   const struct agx_apple9_mesh_key k = AGX_APPLE9_MESH_KEY;
   nir_shader *nir = object ? agx_nir_apple9_taskless_object(&k) : mesh_source(attributes);
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   const struct agx_apple9_native_mesh_options options = {.attributes = true};
   CHECK(object ? agx_compile_apple9_native_object(nir, true, &k, &part, &info, &reason)
                : agx_compile_apple9_native_mesh(nir, true, &options, &part, &info, &reason));
   enum hk_apple9_native_stage stage = object ? HK_APPLE9_NATIVE_OBJECT
                                             : HK_APPLE9_NATIVE_MESH;
   CHECK(hk_apple9_native_shader_valid(&part.info, &info, stage));
   CHECK(!hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_NONE));
   struct agx_shader_info bad = part.info;
   bad.apple9_body_only = false;
   CHECK(!hk_apple9_native_shader_valid(&bad, &info, stage));
   bad = part.info;
   bad.scratch_size = 4;
   CHECK(!hk_apple9_native_shader_valid(&bad, &info, stage));
   bad = part.info;
   bad.local_size = 128;
   CHECK(!hk_apple9_native_shader_valid(&bad, &info, stage));
   bad = part.info;
   bad.workgroup_size[0] = 32;
   CHECK(!hk_apple9_native_shader_valid(&bad, &info, stage));
   struct agx_apple9_mesh_body_info wrong = info;
   wrong.shared_size_B = 4;
   CHECK(!hk_apple9_native_shader_valid(&part.info, &wrong, stage));
   wrong = info;
   wrong.key.grid_offset_B = 4;
   CHECK(!hk_apple9_native_shader_valid(&part.info, &wrong, stage));
   wrong = info;
   wrong.requires_object_completion = true;
   CHECK(!hk_apple9_native_shader_valid(&part.info, &wrong, stage));

   const uint64_t base = UINT64_C(0x1200000000), code = base + 0x123400;
   struct agx_apple9_launch_parameters launch;
   CHECK(hk_apple9_native_launch_parameters(&launch, &part.info, &info,
                                             stage, base, code));
   CHECK(launch.frame_extent_a == (object ? 16 : 64));
   CHECK(launch.frame_extent_b == launch.frame_extent_a);
   CHECK(launch.threadgroup_memory_bytes == (object ? 128 :
      util_next_power_of_two(MAX2(128, 96 + 16 * attributes))));
   CHECK(launch.entry_offset == code - base);
   launch.resource_table = base + 0x445000;
   launch.launch_address = base + 0x448000;
   enum agx_apple9_launch_stage kind = object ? AGX_APPLE9_LAUNCH_TASK
                                             : AGX_APPLE9_LAUNCH_MESH;
   uint8_t bytes[256], templated[256];
   CHECK(agx_apple9_launch_build(bytes, sizeof(bytes), kind, &launch));
   size_t at = agx_apple9_launch_call_offset(kind, 0);
   uint64_t encoded = 0;
   for (unsigned i = 0; i < 6; ++i)
      encoded |= (uint64_t)bytes[at + 2 + i] << (8 * i);
   CHECK(encoded == 2 * code + 0x2a);
   CHECK(bytes[at + 8] == (object ? 2 : 6));
   struct agx_apple9_launch_template tmpl;
   CHECK(agx_apple9_launch_template_init(&tmpl, sizeof(bytes), kind, &launch));
   CHECK(agx_apple9_launch_template_emit(templated, &tmpl, launch.resource_table,
                                          launch.entry_offset, launch.launch_address));
   CHECK(!memcmp(bytes, templated, sizeof(bytes)));
   CHECK(!hk_apple9_native_launch_parameters(&launch, &part.info, &info,
                                              stage, base, base - 2));
   CHECK(!hk_apple9_native_launch_parameters(&launch, &part.info, &info,
                                              stage, base, base + (UINT64_C(1) << 32)));
   if (!object) {
      struct agx_apple9_mesh_state_parameters state = {
         .layout = &info.layout, .object_launch = 0x2400, .mesh_launch = 0x2000,
         .threads = 16, .has_object_stage = true,
      };
      uint32_t packet[AGX_APPLE9_MESH_STATE_WORDS];
      CHECK(agx_apple9_mesh_state_build(packet, &state));
      CHECK(packet[0] == 0x40000d2c && packet[1] == ((5 + attributes) << 8 | (4 + attributes)));
      CHECK(packet[6] == state.object_launch && packet[8] == state.mesh_launch);
      CHECK(packet[9] == 1 && packet[11] == 0x400810 && packet[13] == 5 + attributes);
      struct agx_apple9_render_pipeline pipeline = {
         .ppp = 0x4000, .primitive = AGX_PRIMITIVE_TRIANGLES,
         .ppp_viewport_offset = 0x100, .ppp_viewport_words = 10,
      };
      const uint32_t groups[] = {1, 1, 1}, wide[] = {2, 1, 1};
      const uint32_t local[] = {16, 1, 1}, scalar[] = {1, 1, 1};
      uint32_t command[HK_APPLE9_MESH_COMMAND_BYTES / 4], before[40];
      CHECK(hk_apple9_mesh_command_build(command, sizeof(command), &pipeline,
                                          packet, groups, local, true));
      CHECK(!memcmp(command, packet, sizeof(packet)));
      CHECK(command[32] == (0x70000000 | (AGX_PRIMITIVE_TRIANGLES << 8)));
      for (unsigned i = 0; i < 3; ++i) {
         CHECK(command[33 + i] == groups[i] * local[i]);
         CHECK(command[36 + i] == local[i]);
      }
      CHECK(command[39] == 0xc0000000);
      memcpy(before, command, sizeof(command));
      CHECK(!hk_apple9_mesh_command_build(command, sizeof(command), &pipeline,
                                           packet, groups, local, false));
      CHECK(!memcmp(before, command, sizeof(command)));
      CHECK(!hk_apple9_mesh_command_build(command, sizeof(command), &pipeline,
                                           packet, wide, local, true));
      CHECK(!memcmp(before, command, sizeof(command)));
      CHECK(!hk_apple9_mesh_command_build(command, sizeof(command), &pipeline,
                                           packet, groups, scalar, true));
      CHECK(!memcmp(before, command, sizeof(command)));
      state.task_payload_size_B = 4;
      CHECK(!agx_apple9_mesh_state_build(packet, &state));
      state.task_payload_size_B = 0;
      state.primitive_output.components = 1;
      CHECK(!agx_apple9_mesh_state_build(packet, &state));
   }
   free(part.binary);
   ralloc_free(nir);
}

/* Feed the packet's actual physical entry width to the independent encoded
 * mask checker. The logical size changes the NIR wrapper, never that width. */
static void
mask_fixtures(void)
{
   const unsigned sizes[] = {1, 3, 16};
   for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i) {
      const struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
      nir_shader *nir = agx_nir_apple9_taskless_object(&key);
      nir->info.workgroup_size[0] = sizes[i];
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info info;
      const char *reason = NULL;
      CHECK(agx_compile_apple9_native_object(nir, true, &key, &part, &info, &reason));
      CHECK(info.logical_size[0] == sizes[i]);
      uint32_t local[3];
      for (unsigned c = 0; c < 3; ++c)
         local[c] = part.info.workgroup_size[c];
      const uint32_t groups[] = {1, 1, 1};
      /* Header/PPP contents do not affect the dispatch-tail/mask contract. */
      uint32_t state[AGX_APPLE9_MESH_STATE_WORDS] = {0};
      struct agx_apple9_render_pipeline pipeline = {
         .ppp = 0x4000, .primitive = AGX_PRIMITIVE_TRIANGLES,
      };
      uint32_t command[HK_APPLE9_MESH_COMMAND_BYTES / 4];
      CHECK(hk_apple9_mesh_command_build(command, sizeof(command), &pipeline,
                                          state, groups, local, true));
      printf("object-mask logical=%u physical=%u dispatch=", sizes[i], local[0]);
      for (unsigned c = 33; c < 39; ++c)
         printf("%s%u", c == 33 ? "" : ",", command[c]);
      printf(" binary=");
      for (unsigned c = 0; c < part.info.binary_size; ++c)
         printf("%02x", ((const uint8_t *)part.binary)[c]);
      putchar('\n');
      free(part.binary);
      ralloc_free(nir);
   }
}

static void
check_bad_allocation(const struct agx_shader_info *code,
                     const struct agx_apple9_mesh_body_info *body)
{
   CHECK(!hk_apple9_native_shader_valid(code, body, HK_APPLE9_NATIVE_MESH));
   struct agx_apple9_launch_parameters launch, before;
   memset(&before, 0xa5, sizeof(before));
   memcpy(&launch, &before, sizeof(launch));
   CHECK(!hk_apple9_native_launch_parameters(&launch, code, body,
      HK_APPLE9_NATIVE_MESH, 0, 0x1000));
   CHECK(!memcmp(&launch, &before, sizeof(launch)));
}

static void
check_shared_admission(unsigned attributes, unsigned shared_bytes,
                       unsigned logical_threads)
{
   nir_shader *nir = mesh_source(attributes);
   nir->info.workgroup_size[0] = logical_threads;
   nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
   nir->info.shared_size = shared_bytes;
   if (shared_bytes)
      nir_store_shared(&b, nir_imm_int(&b, 123),
         nir_imm_int(&b, (shared_bytes - 4) & ~3u),
         .align_mul = 4, .write_mask = 1);
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   unsigned threads = logical_threads <= 16 ? 16 : 32;
   const struct agx_apple9_native_mesh_options options = {
      .attributes = true, .physical_threads = threads,
   };
   CHECK(agx_compile_apple9_native_mesh(nir, true, &options, &part, &info, &reason));
   unsigned output = 96 + 16 * attributes;
   unsigned total = output + ALIGN_POT(shared_bytes, 4);
   CHECK(info.shared_offset_B == (shared_bytes ? output : 0));
   CHECK(info.shared_size_B == shared_bytes && info.local_allocation_B == total);
   CHECK(part.info.local_size == (shared_bytes ? total : 0));
   CHECK(hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
   CHECK(!hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_OBJECT));
   struct agx_apple9_launch_parameters launch;
   CHECK(hk_apple9_native_launch_parameters(&launch, &part.info, &info,
      HK_APPLE9_NATIVE_MESH, 0, 0x1000));
   CHECK(launch.threadgroup_memory_bytes == util_next_power_of_two(MAX2(128, total)));
   /* User shared storage changes the launch allocation, not output rows. */
   struct agx_apple9_mesh_state_parameters state = {
      .layout = &info.layout, .object_launch = 0x2400, .mesh_launch = 0x2000,
      .threads = part.info.workgroup_size[0], .has_object_stage = true,
   };
   uint32_t packet[AGX_APPLE9_MESH_STATE_WORDS];
   CHECK(agx_apple9_mesh_state_build(packet, &state));
   CHECK(packet[11] == (0x400800 | threads) && packet[13] == 5 + attributes);

   struct agx_apple9_mesh_body_info bad = info;
   bad.shared_offset_B ^= 16;
   check_bad_allocation(&part.info, &bad);
   bad = info;
   bad.local_allocation_B -= 4;
   check_bad_allocation(&part.info, &bad);
   bad = info;
   bad.shared_size_B = UINT32_MAX;
   check_bad_allocation(&part.info, &bad);
   bad = info;
   bad.shared_size_B = 32768 - output + 1;
   bad.local_allocation_B = output + ALIGN_POT(bad.shared_size_B, 4);
   check_bad_allocation(&part.info, &bad);
   bad = info;
   bad.logical_size[0] = 0;
   check_bad_allocation(&part.info, &bad);
   bad = info;
   bad.logical_size[0] = 129;
   check_bad_allocation(&part.info, &bad);
   bad = info;
   bad.key.physical_threads = 64;
   check_bad_allocation(&part.info, &bad);
   struct agx_shader_info wrong = part.info;
   wrong.local_size ^= 4;
   check_bad_allocation(&wrong, &info);
   wrong = part.info;
   wrong.workgroup_size[0] = threads == 16 ? 32 : 16;
   check_bad_allocation(&wrong, &info);
   wrong = part.info;
   wrong.scratch_size = 4;
   check_bad_allocation(&wrong, &info);
   wrong = part.info;
   wrong.apple9_atomic_frame_size = 4;
   check_bad_allocation(&wrong, &info);
   free(part.binary);
   ralloc_free(nir);
}

static void
check_child_grids(void)
{
   const uint32_t valid[][3] = {
      {1, 1, 1}, {32, 1, 1}, {5, 6, 7},
      {65535, 16, 1}, {1, 65535, 16}, {16, 1, 65535},
      /* Exactly 2^20 - 1, with each axis below its separate limit. */
      {1023, 1025, 1},
   };
   const uint32_t invalid[][3] = {
      {0, 1, 1}, {1, 0, 1}, {1, 1, 0},
      {65536, 1, 1}, {1, 65536, 1}, {1, 1, 65536},
      {1024, 1024, 1}, {65535, 17, 1}, {65535, 65535, 65535},
      {UINT32_MAX, UINT32_MAX, UINT32_MAX},
   };
   CHECK(!hk_apple9_native_mesh_grid_valid(NULL));
   for (unsigned i = 0; i < ARRAY_SIZE(valid); ++i)
      CHECK(hk_apple9_native_mesh_grid_valid(valid[i]));
   for (unsigned i = 0; i < ARRAY_SIZE(invalid); ++i)
      CHECK(!hk_apple9_native_mesh_grid_valid(invalid[i]));
}

int
main(int argc, char **argv)
{
   CHECK(!hk_apple9_native_mesh_enabled());
   glsl_type_singleton_init_or_ref();
   if (argc == 2 && !strcmp(argv[1], "--mask-fixtures")) {
      mask_fixtures();
      glsl_type_singleton_decref();
      return 0;
   }
   CHECK(argc == 1);
   check_child_grids();
   check_contract(false, 0);
   check_contract(false, 1);
   check_contract(false, 5);
   check_contract(false, 124);
   check_contract(true, 0);
   const unsigned sizes[] = {1, 16, 17, 32, 33, 64, 127, 128};
   for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i) {
      check_shared_admission(0, 0, sizes[i]);
      check_shared_admission(0, 28672, sizes[i]);
   }
   check_shared_admission(0, 28671, 128);
   check_shared_admission(0, 32672, 128);
   check_shared_admission(124, 28672, 128);
   check_shared_admission(124, 30688, 128);
   struct agx_apple9_mesh_arguments args = {
      .graphics = {0x1000, 0x2000, 0x3000},
      .output_root = HK_APPLE9_NATIVE_OUTPUT_ROOT,
      .grid_root = HK_APPLE9_NATIVE_GRID_ROOT, .draw = 0x4000,
   };
   uint64_t words[10];
   CHECK(agx_apple9_mesh_arguments_build(words, &args));
   CHECK(words[0] == 0x1000 && words[1] == 0x2000 && words[2] == 0x3000);
   CHECK(words[6] == 0 && words[7] == 64 && words[8] == 0 && words[9] == 0x4000);
   glsl_type_singleton_decref();
   puts("native mesh: compiler contracts, implicit roots, launch and packet validation passed");
   return 0;
}
