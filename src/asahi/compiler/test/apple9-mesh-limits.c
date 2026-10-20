/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Body-layout capacity, separate from native completion capability.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_nir.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%u: %s\n", __LINE__, #x); exit(1); } } while (0)
#include "apple9-mesh-executor.h"

static nir_shader *
source(unsigned attributes, unsigned shared, bool cull,
       bool constant_counts, unsigned vertices, unsigned primitives)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "mesh output capacity");
   nir_shader *nir = b.shader;
   nir->info.workgroup_size[0] = LANES;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.mesh.max_vertices_out = 256;
   nir->info.mesh.max_primitives_out = 256;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir->info.shared_size = shared;
   nir_def *v = constant_counts ? nir_imm_int(&b, vertices)
      : nir_load_global(&b, 1, 32, nir_imm_int64(&b, INPUT_ADDR), .align_mul = 4);
   nir_def *p = constant_counts ? nir_imm_int(&b, primitives)
      : nir_load_global(&b, 1, 32, nir_imm_int64(&b, INPUT_ADDR + 4), .align_mul = 4);
   nir_set_vertex_and_primitive_count(&b, v, p, nir_imm_int(&b, 0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   for (unsigned start = 0; start < 256; start += LANES) {
      nir_def *i = nir_iadd_imm(&b, lane, start);
      nir_push_if(&b, nir_ult(&b, i, v));
      for (unsigned c = 0; c < 4 + attributes; ++c) {
         unsigned slot = c < 4 ? VARYING_SLOT_POS
                              : VARYING_SLOT_VAR0 + (c - 4) / 4;
         nir_store_per_vertex_output(&b,
            nir_iadd_imm(&b, i, 0x100000 + c * 4096), i, nir_imm_int(&b, 0),
            .component = c % 4, .write_mask = 1, .src_type = nir_type_uint32,
            .io_semantics = {.location = slot, .num_slots = 1});
      }
      nir_pop_if(&b, NULL);
      nir_push_if(&b, nir_iand(&b, nir_ult(&b, i, p), nir_ine_imm(&b, v, 0)));
      nir_def *indices[3];
      for (unsigned c = 0; c < 3; ++c)
         indices[c] = nir_umod(&b, nir_iadd_imm(&b, nir_imul_imm(&b, i, 3), c), v);
      nir_store_per_primitive_output(&b, nir_vec(&b, indices, 3), i,
         nir_imm_int(&b, 0), .write_mask = 7, .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      if (cull)
         nir_store_per_primitive_output(&b, nir_iand_imm(&b, i, 1), i,
            nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   if (shared) {
      nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
      nir_store_shared(&b, nir_imm_intN_t(&b, 0x5a, 8),
         nir_imm_int(&b, shared - 1), .align_mul = 1, .write_mask = 1);
      nir_pop_if(&b, NULL);
   }
   nir_validate_shader(nir, "mesh limits source");
   return nir;
}

static void
check(unsigned attributes, unsigned shared, bool cull,
      unsigned vertices, unsigned primitives)
{
   const struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   nir_shader *nir = source(attributes, shared, cull, false, 0, 0);
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(agx_nir_lower_apple9_mesh(nir, &key, &info, &reason));
   CHECK(info.requires_mesh_completion);
   CHECK(info.layout.max_vertices == 256 && info.layout.max_primitives == 256);
   /* Reference layout: 768 index bytes, 16 header bytes, optional cull words,
    * then 256 32-bit elements per component. Include untouched padding. */
   unsigned position = 784 + (cull ? 1024 : 0);
   unsigned output = position + 1024 * (4 + attributes);
   CHECK(info.layout.position_B == position && info.layout.local_size_B == output);
   CHECK(info.local_allocation_B == output + ALIGN_POT(shared, 4));
   nir_validate_shader(nir, "mesh limits lowered");
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   struct machine m = {0};
   uint8_t expected[sizeof(m.local)];
   memset(m.local, 0xa5, sizeof(m.local));
   memset(expected, 0xa5, sizeof(expected));
   uint32_t counts[] = {vertices, primitives};
   m.region[5] = (struct region){INPUT_ADDR, sizeof(counts), (uint8_t *)counts, NULL};
   m.shared_begin = output;
   m.shared_end = output + shared;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.count_lanes == ALL_LANES);
   CHECK(m.actual_vertices == vertices && m.actual_primitives == primitives);
   CHECK(!m.controls && !m.scalars);
   for (unsigned i = 0; i < vertices; ++i) {
      for (unsigned c = 0; c < 4 + attributes; ++c) {
         uint32_t value = 0x100000 + c * 4096 + i;
         memcpy(expected + position + c * 1024 + i * 4, &value, 4);
      }
   }
   for (unsigned i = 0; vertices && i < primitives; ++i) {
      for (unsigned c = 0; c < 3; ++c)
         expected[3 * i + c] = (3 * i + c) % vertices;
      if (cull)
         expected[784 + 4 * i] = i & 1;
   }
   uint16_t count = vertices ? primitives : 0;
   memcpy(expected + 768, &count, 2);
   if (shared)
      expected[output + shared - 1] = 0x5a;
   CHECK(!memcmp(expected, m.local, sizeof(expected)));
   free(m.value);
   ralloc_free(nir);

   /* Compile ordinary body NIR through allocation/encoding too. It must
    * retain the completion requirement and stay inadmissible as a native
    * complete stage even when all declared output bytes fit. */
   nir = source(attributes, shared, cull, false, 0, 0);
   struct agx_shader_part code = {0};
   CHECK(agx_compile_apple9_mesh_body(nir, &key, &code, &info, &reason));
   CHECK(code.binary && code.info.apple9_body_only && info.requires_mesh_completion);
   unsigned bytes = code.info.binary_size;
   free(code.binary);
   ralloc_free(nir);
   nir = source(attributes, shared, cull, false, 0, 0);
   code = (struct agx_shader_part){0};
   CHECK(!agx_compile_apple9_native_mesh(nir, true, NULL, &code, &info, &reason));
   CHECK(!code.binary && strstr(reason, "V3 P1"));
   ralloc_free(nir);
   printf("body V256 P256 attributes=%u cull=%u shared=%u actual=%u/%u local=%u code=%u\n",
          attributes, cull, shared, vertices, primitives, output + ALIGN_POT(shared, 4), bytes);
}

static void
reject(unsigned attributes, unsigned shared, unsigned vertices,
       unsigned primitives, const char *message)
{
   const struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   nir_shader *nir = source(attributes, shared, false, true, vertices, primitives);
   struct agx_apple9_mesh_body_info info, before;
   memset(&info, 0xa5, sizeof(info));
   before = info;
   char *before_nir = nir_shader_as_str(nir, NULL);
   const char *reason = NULL;
   CHECK(!agx_nir_lower_apple9_mesh(nir, &key, &info, &reason));
   CHECK(reason && strstr(reason, message));
   CHECK(!memcmp(&info, &before, sizeof(info)));
   char *after_nir = nir_shader_as_str(nir, NULL);
   CHECK(!strcmp(before_nir, after_nir));
   ralloc_free(before_nir);
   ralloc_free(after_nir);
   ralloc_free(nir);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check(0, 0, false, 256, 256);
   check(0, 0, true, 129, 17);
   check(3, 3, false, 3, 1);
   check(0, 0, false, 0, 256);
   check(0, 0, false, 256, 0);
   check(27, 240, false, 256, 256);
   reject(28, 0, 256, 256, "output layout exceeds");
   reject(27, 241, 256, 256, "shared allocations exceed");
   reject(0, 0, 257, 256, "counts exceed");
   reject(0, 0, 256, 257, "counts exceed");
   glsl_type_singleton_decref();
   puts("mesh body limits: 6 semantic/compile cases and 4 non-mutating rejections pass");
   return 0;
}
