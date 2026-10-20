/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Native mesh shared/output separation and cross-subgroup dataflow.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_nir.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%u: %s\n", __LINE__, #x); exit(1); } } while (0)
#define LANES 32
#include "apple9-mesh-executor.h"

static void
barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
      .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
      .memory_modes = nir_var_mem_shared);
}

static nir_def *
q_address(nir_builder *b, nir_def *i, unsigned word)
{
   return nir_iadd_imm(b, nir_u2u64(b, nir_imul_imm(b, i, 16)),
                       INPUT_ADDR + 16 + word * 4);
}

static nir_shader *
source(unsigned x, unsigned y, unsigned z, unsigned bytes,
       unsigned attributes, bool declaration)
{
   unsigned total = x * y * z;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "native mesh shared/output allocation");
   nir_shader *nir = b.shader;
   nir->info.workgroup_size[0] = x;
   nir->info.workgroup_size[1] = y;
   nir->info.workgroup_size[2] = z;
   nir->info.mesh.max_vertices_out = 3;
   nir->info.mesh.max_primitives_out = 1;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_variable *var = declaration ? nir_variable_create(nir, nir_var_mem_shared,
      glsl_array_type(glsl_uint_type(), bytes / 4, 0), "user_shared") : NULL;
   if (!var)
      nir->info.shared_size = bytes;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                    nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   nir_def *i = nir_load_local_invocation_index(&b);
   nir_def *offset = nir_imul_imm(&b, i, 4);
   nir_def *initial = nir_iadd_imm(&b, i, 100);
   if (var)
      nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, var), i), initial, 1);
   else
      nir_store_shared(&b, initial, offset, .write_mask = 1, .align_mul = 4);
   /* Narrow accesses occupy the final three bytes, including an allocation
    * ending inside a word. No output/header byte may be used as padding. */
   nir_push_if(&b, nir_ieq_imm(&b, i, 0));
   nir_store_shared(&b, nir_imm_intN_t(&b, 0x5a, 8), nir_imm_int(&b, 0),
      .base = bytes - 1, .write_mask = 1, .align_mul = 1);
   nir_store_shared(&b, nir_imm_intN_t(&b, 0x1234, 16), nir_imm_int(&b, 0),
      .base = (bytes - 4) & ~1u, .write_mask = 1, .align_mul = 2);
   nir_pop_if(&b, NULL);
   barrier(&b);
   nir_def *neighbor = nir_umod_imm(&b, nir_iadd_imm(&b, i, 32), total);
   nir_def *cross = var
      ? nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, var), neighbor))
      : nir_load_shared(&b, 1, 32, nir_imul_imm(&b, neighbor, 4), .align_mul = 4);
   nir_store_global(&b, cross, q_address(&b, i, 0), .align_mul = 4);
   barrier(&b);
   nir_def *old = nir_shared_atomic(&b, 32, offset, nir_imm_int(&b, 7),
                                    .atomic_op = nir_atomic_op_iadd);
   nir_def *exchanged = nir_shared_atomic_swap(&b, 32, offset,
      nir_iadd_imm(&b, i, 107), nir_iadd_imm(&b, i, 200),
      .atomic_op = nir_atomic_op_cmpxchg);
   nir_store_global(&b, old, q_address(&b, i, 1), .align_mul = 4);
   nir_store_global(&b, exchanged, q_address(&b, i, 2), .align_mul = 4);
   barrier(&b);
   cross = nir_load_shared(&b, 1, 32, nir_imul_imm(&b, neighbor, 4), .align_mul = 4);
   nir_store_global(&b, cross, q_address(&b, i, 3), .align_mul = 4);
   /* Geometry uses values produced by other logical subgroups. The last
    * logical lane publishes, including a one-lane final partial subgroup. */
   nir_push_if(&b, nir_ieq_imm(&b, i, total - 1));
   nir_def *byte = nir_load_shared(&b, 1, 8, nir_imm_int(&b, 0),
                                   .base = bytes - 1, .align_mul = 1);
   nir_def *half = nir_load_shared(&b, 1, 16, nir_imm_int(&b, 0),
                                   .base = (bytes - 4) & ~1u, .align_mul = 2);
   for (unsigned v = 0; v < 3; ++v) {
      nir_def *p[] = {cross, nir_u2u32(&b, byte), nir_u2u32(&b, half), nir_imm_int(&b, 0x3f800000)};
      nir_store_per_vertex_output(&b, nir_vec(&b, p, 4), nir_imm_int(&b, v),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      for (unsigned c = 0; c < attributes; ++c)
         nir_store_per_vertex_output(&b, nir_iadd_imm(&b, cross, c), nir_imm_int(&b, v),
            nir_imm_int(&b, 0), .write_mask = 1, .component = c % 4,
            .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + c / 4, .num_slots = 1});
   }
   nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_pop_if(&b, NULL);
   return nir;
}

static void
check(unsigned x, unsigned y, unsigned z, unsigned bytes,
      unsigned attributes, bool declaration)
{
   unsigned total = x * y * z, output = 96 + 16 * attributes;
   nir_shader *nir = source(x, y, z, bytes, attributes, declaration);
   const struct agx_apple9_native_mesh_options options = {
      .physical_threads = 32, .attributes = true};
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   if (!agx_nir_lower_apple9_native_mesh(nir, true, &options, &info, &reason)) {
      fprintf(stderr, "lowering failed: %s\n", reason); exit(1);
   }
   CHECK(info.layout.local_size_B == output && info.shared_offset_B == output);
   CHECK(info.shared_size_B == bytes && info.local_allocation_B == output + ALIGN_POT(bytes, 4));
   CHECK(nir->info.shared_size == info.local_allocation_B);
   nir_lower_vars_to_ssa(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   nir_validate_shader(nir, "native mesh shared allocation");
   uint32_t q[4 + 128 * 4 + 4];
   uint8_t written[sizeof(q)] = {0};
   memset(q, 0xa5, sizeof(q));
   struct machine m = {.shared_begin = output, .shared_end = output + bytes};
   memset(m.local, 0xa5, sizeof(m.local));
   m.region[0] = (struct region){INPUT_ADDR, sizeof(q), (void *)q, written};
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   CHECK(m.controls == ALL_LANES && m.scalars == 1);
   for (unsigned j = 0; j < total; ++j) {
      unsigned neighbor = (j + 32) % total;
      const uint32_t expected[] = {100 + neighbor, 100 + j, 107 + j, 200 + neighbor};
      CHECK(!memcmp(q + 4 + 4 * j, expected, sizeof(expected)));
      uint32_t local; memcpy(&local, m.local + output + 4 * j, 4);
      CHECK(local == 200 + j);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(q); ++i) {
      bool touched = i >= 4 && i < 4 + total * 4;
      if (!touched) CHECK(q[i] == 0xa5a5a5a5);
      for (unsigned b = 0; b < 4; ++b) CHECK(written[4 * i + b] == touched);
   }
   for (unsigned v = 0; v < 3; ++v) {
      const uint32_t p[] = {200 + (total - 1 + 32) % total, 0x5a, 0x1234, 0x3f800000};
      CHECK(!memcmp(m.submitted_position[v], p, sizeof(p)));
      for (unsigned c = 0; c < attributes; ++c)
         CHECK(m.submitted_position[v][4 + c] == p[0] + c);
   }
   /* All gaps between outputs and touched user data, plus the allocation
    * tail, retain poison. This catches missing rebases and size rounding. */
   for (unsigned at = output + 4 * total; at < sizeof(m.local); ++at) {
      unsigned half = output + ((bytes - 4) & ~1u);
      if (at == output + bytes - 1 || at == half || at == half + 1) continue;
      CHECK(m.local[at] == 0xa5);
   }
   free(m.value);
   struct agx_shader_part part = {0};
   if (!agx_compile_apple9_mesh_lowered(nir, &info.key, &part, &reason)) {
      fprintf(stderr, "compilation failed: %s\n", reason); exit(1);
   }
   CHECK(part.info.apple9_body_only && part.info.binary_size);
   CHECK(part.info.local_size == info.local_allocation_B);
   CHECK(!part.info.apple9_resource_count && !part.info.apple9_ubo_mask &&
         !part.info.apple9_resource_ssbo_mask && !part.info.apple9_resource_write_mask);
   printf("shared_fixture=T%u_%u_%u_%u_bytes%u_attr%u_decl%u root=%u output=%u shared=%u total=%u code=%u scratch=%u\n",
      total, x, y, z, bytes, attributes, declaration, info.key.output_root,
      output, bytes, info.local_allocation_B, part.info.binary_size, part.info.scratch_size);
   free(part.binary); ralloc_free(nir);
}

static void
reject(void)
{
   const unsigned sizes[] = {0, 32673, 32768, 32769, UINT32_MAX};
   for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i) {
      nir_shader *nir = source(32, 1, 1, 28672, 0, false);
      nir->info.shared_size = sizes[i];
      const struct agx_apple9_native_mesh_options options = {.physical_threads = 32};
      struct agx_apple9_mesh_body_info info;
      const char *reason = NULL;
      CHECK(!agx_nir_lower_apple9_native_mesh(nir, true, &options, &info, &reason));
      CHECK(reason && strstr(reason, "shared"));
      ralloc_free(nir);
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const unsigned sizes[] = {1, 3, 16, 32, 33, 64, 127, 128};
   for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i)
      check(sizes[i], 1, 1, 28672, 0, false);
   check(4, 4, 4, 28672, 0, false);
   check(32, 1, 1, 28672, 0, true);
   check(32, 1, 1, 28671, 0, false);
   check(32, 1, 1, 32672, 0, false);
   check(32, 1, 1, 28672, 124, false);
   check(32, 1, 1, 30688, 124, false);
   reject();
   glsl_type_singleton_decref();
   puts("14 shared mesh programs: cross-subgroup exchange, atomic results, narrow tails, declarations, maximum attributes, exact 32-KiB boundary; 5 rejected allocations");
   return 0;
}
