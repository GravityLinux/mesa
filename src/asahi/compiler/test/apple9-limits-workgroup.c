/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Execute the native compiler's logical-workgroup lowering and completion.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_nir.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%u: %s\n", __LINE__, #x); exit(1); } } while (0)
#define LANES 32
#include "apple9-mesh-executor.h"

#define WORDS 32
#define GUARD 64
#define POISON 0xa5a5a5a5u

static nir_def *address(nir_builder *b, nir_def *index, unsigned word)
{
   return nir_iadd_imm(b, nir_u2u64(b, nir_imul_imm(b, index, WORDS * 4)),
                       INPUT_ADDR + (GUARD + word) * 4);
}

static void collective(nir_builder *b, nir_def *i, unsigned word)
{
   nir_def *values[] = {
      nir_ballot(b, 1, 32, nir_imm_true(b)),
      nir_reduce(b, i, .reduction_op = nir_op_iadd),
      nir_read_first_invocation(b, i), nir_b2i32(b, nir_elect(b, 1)),
   };
   for (unsigned c = 0; c < ARRAY_SIZE(values); ++c)
      nir_store_global(b, values[c], address(b, i, word + c), .align_mul = 4);
}

static nir_shader *source(unsigned x, unsigned y, unsigned z)
{
   unsigned total = x * y * z;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "native logical workgroups on one full subgroup");
   nir_shader *nir = b.shader;
   nir->info.workgroup_size[0] = x;
   nir->info.workgroup_size[1] = y;
   nir->info.workgroup_size[2] = z;
   nir->info.mesh.max_vertices_out = 3;
   nir->info.mesh.max_primitives_out = 1;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                    nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_def *i = nir_load_local_invocation_index(&b);
   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *size = nir_load_workgroup_size(&b);
   nir_global_atomic(&b, 32, address(&b, i, 0), nir_imm_int(&b, 1),
                     .atomic_op = nir_atomic_op_iadd);
   nir_def *values[] = {
      nir_iadd_imm(&b, i, 100), nir_channel(&b, id, 0),
      nir_channel(&b, id, 1), nir_channel(&b, id, 2),
      nir_channel(&b, size, 0), nir_channel(&b, size, 1),
      nir_channel(&b, size, 2), nir_load_subgroup_size(&b),
      nir_load_subgroup_id(&b), nir_load_subgroup_invocation(&b),
      nir_load_num_subgroups(&b),
   };
   for (unsigned c = 0; c < ARRAY_SIZE(values); ++c)
      nir_store_global(&b, values[c], address(&b, i, c + 1), .align_mul = 4);
   collective(&b, i, 12);
   nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, i, 1), 0));
   collective(&b, i, 16);
   nir_push_if(&b, nir_ieq_imm(&b, nir_iand_imm(&b, i, 3), 1));
   collective(&b, i, 20);
   nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);
   /* A converged nested barrier separates writes from reads by a different
    * logical subgroup. Keep the index SSA live across both barriers. */
   nir_push_if(&b, nir_ieq_imm(&b, nir_channel(&b, size, 0), x));
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
      .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
      .memory_modes = nir_var_mem_global);
   nir_def *neighbor = nir_umod_imm(&b, nir_iadd_imm(&b, i, 32), total);
   nir_def *cross = nir_load_global(&b, 1, 32, address(&b, neighbor, 1), .align_mul = 4);
   nir_store_global(&b, cross, address(&b, i, 24), .align_mul = 4);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
      .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
      .memory_modes = nir_var_mem_global);
   nir_def *again = nir_load_global(&b, 1, 32, address(&b, neighbor, 24), .align_mul = 4);
   nir_store_global(&b, again, address(&b, i, 25), .align_mul = 4);
   nir_pop_if(&b, NULL);
   /* The last logical invocation (including a one-lane final subgroup) owns
    * all geometry. Physical completion must run once after it has finished. */
   nir_push_if(&b, nir_ieq_imm(&b, i, total - 1));
   for (unsigned v = 0; v < 3; ++v) {
      nir_def *p[4];
      for (unsigned c = 0; c < 4; ++c)
         p[c] = nir_imm_int(&b, 0x3f000000 + 4 * v + c);
      nir_store_per_vertex_output(&b, nir_vec(&b, p, 4), nir_imm_int(&b, v),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   }
   nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_pop_if(&b, NULL);
   return nir;
}

static void check(unsigned x, unsigned y, unsigned z)
{
   unsigned total = x * y * z;
   nir_shader *nir = source(x, y, z);
   const struct agx_apple9_native_mesh_options options = {.physical_threads = 32};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &options, &info, &reason));
   CHECK(info.logical_size[0] == x && info.logical_size[1] == y && info.logical_size[2] == z);
   CHECK(info.key.physical_threads == 32 && !info.requires_mesh_completion);
   nir_lower_vars_to_ssa(nir);
   nir_validate_shader(nir, "native logical workgroup size lowering");
   CHECK(nir->info.workgroup_size[0] == 32 &&
         nir->info.workgroup_size[1] == 1 && nir->info.workgroup_size[2] == 1);
   uint32_t q[GUARD + 128 * WORDS + GUARD], expected[ARRAY_SIZE(q)];
   memset(q, 0xa5, sizeof(q));
   for (unsigned j = 0; j < 128; ++j) q[GUARD + WORDS * j] = 0;
   memcpy(expected, q, sizeof(q));
   for (unsigned j = 0; j < total; ++j) {
      unsigned sg = j / 32;
      uint32_t *row = expected + GUARD + WORDS * j;
      const uint32_t prefix[] = {1, 100 + j, j % x, j / x % y, j / (x * y),
         x, y, z, 32, sg, j % 32, DIV_ROUND_UP(total, 32)};
      memcpy(row, prefix, sizeof(prefix));
      for (unsigned group = 0; group < 3; ++group) {
         unsigned mask = group == 0 ? 0 : group == 1 ? 1 : 3;
         unsigned residue = group != 0;
         if ((j & mask) != residue) continue;
         uint32_t ballot = 0, sum = 0, first = UINT32_MAX;
         for (unsigned k = sg * 32; k < MIN2(total, (sg + 1) * 32); ++k) {
            if ((k & mask) != residue) continue;
            ballot |= 1u << (k % 32); sum += k; first = MIN2(first, k);
         }
         row[12 + 4 * group] = ballot;
         row[13 + 4 * group] = sum;
         row[14 + 4 * group] = first;
         row[15 + 4 * group] = j == first;
      }
      row[24] = 100 + (j + 32) % total;
      row[25] = 100 + (j + 64) % total;
   }
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   struct machine m = {0};
   uint8_t written[sizeof(q)] = {0};
   m.region[0] = (struct region){INPUT_ADDR, sizeof(q), (void *)q, written};
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.barriers + m.subgroup_barriers == 3);
   CHECK(m.controls == ALL_LANES && m.scalars == 1);
   for (unsigned v = 0; v < 3; ++v)
      for (unsigned c = 0; c < 4; ++c)
         CHECK(m.submitted_position[v][c] == 0x3f000000 + 4 * v + c);
   for (unsigned c = 0; c < ARRAY_SIZE(q); ++c) {
      if (q[c] != expected[c])
         fprintf(stderr, "T%u (%u,%u,%u) word %u: %#x != %#x\n",
                 total, x, y, z, c, q[c], expected[c]);
      CHECK(q[c] == expected[c]);
      bool touched = c >= GUARD && c < GUARD + total * WORDS &&
         expected[c] != POISON;
      for (unsigned byte = 0; byte < 4; ++byte)
         CHECK(written[4 * c + byte] == touched);
   }
   free(m.value);
   struct agx_shader_part part = {0};
   CHECK(agx_compile_apple9_mesh_lowered(nir, &info.key, &part, &reason));
   CHECK(part.info.apple9_body_only && part.info.binary_size && !part.info.scratch_size);
   CHECK(part.info.workgroup_size[0] == 32 && part.info.workgroup_size[1] == 1 && part.info.workgroup_size[2] == 1);
   printf("T%u (%u,%u,%u): native T32, semantic oracle passed, code=%u bytes, GPRs=%u\n",
          total, x, y, z, part.info.binary_size, part.info.nr_gprs);
   free(part.binary); ralloc_free(nir);
}

static void reject(void)
{
   const unsigned sizes[][3] = {{0,1,1}, {129,1,1}, {8,8,3}, {65535,65535,65535}};
   for (unsigned test = 0; test < 8; ++test) {
      nir_shader *nir = source(1,1,1);
      struct agx_apple9_native_mesh_options options = {.physical_threads = 32};
      if (test < ARRAY_SIZE(sizes))
         for (unsigned d = 0; d < 3; ++d) nir->info.workgroup_size[d] = sizes[test][d];
      else if (test == 4) nir->info.workgroup_size_variable = true;
      else if (test == 5) nir->info.api_subgroup_size = 16;
      else if (test == 6) options.physical_threads = 64;
      else nir->info.workgroup_size[0] = 17, options.physical_threads = 0;
      struct agx_apple9_mesh_body_info info;
      const char *reason = NULL;
      CHECK(!agx_nir_lower_apple9_native_mesh(nir, true, &options, &info, &reason));
      CHECK(reason);
      ralloc_free(nir);
   }
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   const unsigned shapes[][3] = {
      {1,1,1}, {3,1,1}, {17,1,1}, {31,1,1}, {32,1,1}, {33,1,1},
      {63,1,1}, {64,1,1}, {65,1,1}, {95,1,1}, {96,1,1}, {97,1,1},
      {127,1,1}, {128,1,1}, {1,128,1}, {1,1,128}, {3,3,3}, {3,3,7}, {4,4,4}, {5,5,5}, {1,8,16}};
   for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i)
      check(shapes[i][0], shapes[i][1], shapes[i][2]);
   reject();
   glsl_type_singleton_decref();
   puts("21 native T32 cases: partial and nested masks, subgroup operations, 3D IDs, two barrier phases, exactly-once effects, guards and one completion; 8 rejection cases");
   return 0;
}
