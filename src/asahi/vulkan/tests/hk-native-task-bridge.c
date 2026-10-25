/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_buffer.h"
#include "asahi/libagx/mesh.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
#define LANES 1
#include "asahi/compiler/test/apple9-mesh-executor.h"

static nir_shader *
consumer(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "task bridge ownership");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = b.shader->info.workgroup_size[1] = 2;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   b.shader->info.task_payload_size = 16384;
   nir_def *id = nir_load_workgroup_id(&b);
   nir_def *count = nir_load_num_workgroups(&b);
   nir_def *global = nir_load_global_invocation_id(&b, 32);
   nir_store_global(&b, id, nir_imm_int64(&b, 0x60000), .align_mul = 4);
   nir_store_global(&b, count, nir_imm_int64(&b, 0x6000c), .align_mul = 4);
   nir_store_global(&b, global, nir_imm_int64(&b, 0x60018), .align_mul = 4);
   nir_def *first = nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 0),
                                         .align_mul = 4);
   nir_def *last = nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 16380),
                                        .align_mul = 4);
   nir_store_global(&b, nir_vec2(&b, first, last), nir_imm_int64(&b, 0x60024),
                    .align_mul = 4);
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                      nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
ownership(void)
{
   nir_shader *s = consumer();
   hk_nir_lower_native_mesh_task(s);
   CHECK(s->info.task_payload_size == 0);
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_index_ssa_defs(impl);
   struct hk_root_descriptor_table root = {.draw.mesh_params = 0x30000};
   /* Skip a zero-child parent between two nonempty parents. The final parent
    * emits a 2x2x2 grid, testing every axis of WorkgroupID/GlobalInvocationID. */
   uint32_t grids[][4] = {{1,1,1,1}, {0,0,0,1}, {2,2,2,9}};
   uint32_t payloads[3][4096] = {{0}};
   for (unsigned p = 0; p < 3; ++p) {
      payloads[p][0] = 0xa000 + p;
      payloads[p][4095] = 0xb000 + p;
   }
   struct libagx_mesh_params params = {
      .payloads = 0x40000, .task_grids = 0x50000, .task_count = 3,
      .payload_stride_B = 16384, .mesh_count = 9,
   };
   for (unsigned g = 0; g < 11; ++g) {
      uint32_t output[11];
      uint8_t written[sizeof(output)] = {0};
      memset(output, 0xcc, sizeof(output));
      struct machine m = {
         .ubo_base = 0x20000, .group_id = {g,0,0}, .num_groups = {9,1,1},
         .region = {
            {0x20000,sizeof(root),(uint8_t *)&root,NULL},
            {0x30000,sizeof(params),(uint8_t *)&params,NULL},
            {0x40000,sizeof(payloads),(uint8_t *)payloads,NULL},
            {0x50000,sizeof(grids),(uint8_t *)grids,NULL},
            {0x60000,sizeof(output),(uint8_t *)output,written},
         },
      };
      if (g == 10) {
         /* The zero-grid dummy may not even read a parent or payload. */
         params.mesh_count = params.task_count = 0;
         params.task_grids = params.payloads = 0;
         m.group_id[0] = 0;
      }
      m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
      CHECK(m.value);
      execute(&m, &impl->body, ALL_LANES);
      if (g < 9) {
         unsigned local = g ? g - 1 : 0, parent = g ? 2 : 0;
         const uint32_t expect[] = {
            local % 2, (local / 2) % 2, local / 4,
            g ? 2 : 1, g ? 2 : 1, g ? 2 : 1,
            2 * (local % 2), 2 * ((local / 2) % 2), local / 4,
            0xa000 + parent, 0xb000 + parent,
         };
         CHECK(!memcmp(output, expect, sizeof(output)));
         CHECK(m.actual_vertices == 3 && m.actual_primitives == 1);
         for (unsigned i = 0; i < sizeof(written); ++i) CHECK(written[i] == 1);
      } else {
         for (unsigned i = 0; i < ARRAY_SIZE(output); ++i) CHECK(output[i] == 0xcccccccc);
         CHECK(m.writes == 0 && m.actual_vertices == 0 && m.actual_primitives == 0);
      }
      free(m.value);
   }
   ralloc_free(s);
}

static void
bounds(void)
{
   for (unsigned c = 0; c < 6; ++c) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_TASK, &agx_nir_options, "task child bounds");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.max_subgroup_size = b.shader->info.min_subgroup_size = 32;
      nir_def *x = c == 0 ? nir_imm_int(&b, 3) :
         c == 1 ? nir_iand_imm(&b, nir_load_workgroup_id(&b), 7) :
         c == 2 ? nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000), .align_mul=4) :
         c == 3 ? nir_imm_int(&b, 0) :
         c == 4 ? nir_imm_int(&b, UINT32_MAX) :
                  nir_bit_count(&b, nir_load_local_invocation_index(&b));
      x = nir_channel(&b, x, 0);
      nir_launch_mesh_workgroups(&b, nir_vec3(&b, x,
         nir_imm_int(&b, 2), nir_imm_int(&b, 3)));
      const uint32_t expect[] = {18,42,UINT32_MAX,0,UINT32_MAX,30};
      uint32_t bound = hk_nir_task_mesh_bound(b.shader);
      if (bound != expect[c]) fprintf(stderr, "case %u bound=%u expected=%u\n", c, bound, expect[c]);
      CHECK(bound == expect[c]);
      ralloc_free(b.shader);
   }
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   bounds();
   ownership();
   glsl_type_singleton_decref();
   puts("task bounds, skipped parents, 3D child IDs, 16-KiB payload ownership and zero-child side effects PASS");
   return 0;
}
