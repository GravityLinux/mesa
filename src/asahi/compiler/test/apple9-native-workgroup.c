/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Normal NIR workgroup expansion, host semantics and device fixture exporter.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_apple9_mesh_formation.h"
#include "agx_nir.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)
#include "apple9-mesh-executor.h"
#include "mesh-formation-fixtures.h"
#include "mesh-h1-fixtures.h"

#include "apple9-mesh-data.h"

#define Q_ADDRESS 0x40000ull
#define Q_BYTES   2048
#define POISON    0xa5a5a5a5u

/* This test owns Q in the otherwise unused payload-pointer argument. It is
 * instrumentation, not an application task payload or production ABI change.
 * Every API lane performs a store and an atomic, with no source-level lane
 * guard. Lowering must exclude every extra physical invocation. */
static void
side_effects(nir_builder *b)
{
   nir_def *q =
      nir_pack_64_2x32(b, nir_load_native_pointer_apple9_agx(
                             b, .base = AGX_APPLE9_MESH_PAYLOAD_UNIFORM));
   nir_def *index = nir_load_local_invocation_index(b);
   nir_def *id = nir_load_local_invocation_id(b);
   nir_def *size = nir_load_workgroup_size(b);
   nir_def *fields[] = {
      index,
      nir_channel(b, id, 0),
      nir_channel(b, id, 1),
      nir_channel(b, id, 2),
      nir_channel(b, size, 0),
      nir_channel(b, size, 1),
      nir_channel(b, size, 2),
      nir_load_subgroup_size(b),
      nir_load_subgroup_invocation(b),
      nir_load_subgroup_id(b),
      nir_load_num_subgroups(b),
      nir_ballot(b, 1, 32, nir_imm_true(b)),
   };
   nir_def *address = nir_iadd(
      b, q, nir_u2u64(b, nir_iadd_imm(b, nir_imul_imm(b, index, 64), 256)));
   nir_global_atomic(b, 32, q, nir_imm_int(b, 1),
                     .atomic_op = nir_atomic_op_iadd);
   for (unsigned i = 0; i < ARRAY_SIZE(fields); ++i)
      nir_store_global(b, fields[i], nir_iadd_imm(b, address, 4 * i),
                       .align_mul = 4);
   /* Nested, API-uniform barrier exercises subgroup shrinking as well as
    * values live across the barrier. Width-16 stays a workgroup barrier. */
   nir_push_if(b, nir_ieq_imm(b, nir_channel(b, size, 1),
                              b->shader->info.workgroup_size[1]));
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   nir_store_global(b, nir_ixor(b, index, nir_imm_int(b, 0x12345678)),
                    nir_iadd_imm(b, address, 48), .align_mul = 4);
   nir_pop_if(b, NULL);
}

static nir_shader *
source(bool object, unsigned x, unsigned y, unsigned z, bool probe)
{
   nir_builder b = nir_builder_init_simple_shader(
      object ? MESA_SHADER_TASK : MESA_SHADER_MESH, &agx_nir_options,
      "native_workgroup_fixture");
   b.shader->info.workgroup_size[0] = x;
   b.shader->info.workgroup_size[1] = y;
   b.shader->info.workgroup_size[2] = z;
   if (probe)
      side_effects(&b);
   if (object) {
      nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 1, 1, 1));
      return b.shader;
   }
   const struct formation_fixture *f = &h1_fixtures[0];
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                      nir_imm_int(&b, 1), nir_imm_int(&b, 0));
   for (unsigned v = 0; v < 3; ++v) {
      /* Ordinary API work distribution: T1 writes all vertices itself. */
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b),
                                  v % (x * y * z)));
      nir_def *position[4];
      for (unsigned c = 0; c < 4; ++c)
         position[c] = nir_imm_int(&b, f->positions[4 * v + c]);
      nir_store_per_vertex_output(
         &b, nir_vec(&b, position, 4), nir_imm_int(&b, v), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_store_per_primitive_output(
      &b, nir_imm_ivec3(&b, 0, 1, 2), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .write_mask = 7, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                       .num_slots = 1});
   nir_pop_if(&b, NULL);
   return b.shader;
}

static bool
lower(nir_shader *nir, bool enable, struct agx_apple9_mesh_body_info *info,
      const char **reason)
{
   const struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   return nir->info.stage == MESA_SHADER_TASK
             ? agx_nir_lower_apple9_native_object(nir, enable, &key, info,
                                                  reason)
             : agx_nir_lower_apple9_native_mesh(nir, enable, info, reason);
}

static void
write_file(const char *dir, const char *name, const char *ext, const void *data,
           size_t bytes)
{
   char path[1024];
   CHECK(snprintf(path, sizeof(path), "%s/%s.%s", dir, name, ext) <
         sizeof(path));
   FILE *f = fopen(path, "wb");
   CHECK(f && fwrite(data, 1, bytes, f) == bytes && !fclose(f));
}

static void
check(bool object, unsigned x, unsigned y, unsigned z, bool probe,
      const char *dir)
{
   unsigned threads = x * y * z;
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   nir_shader *nir = source(object, x, y, z, probe);
   CHECK(lower(nir, true, &info, &reason));
   CHECK(info.logical_size[0] == x && info.logical_size[1] == y &&
         info.logical_size[2] == z);
   CHECK(nir->info.workgroup_size[0] == 16 &&
         nir->info.workgroup_size[1] == 1 && nir->info.workgroup_size[2] == 1);
   CHECK(info.key.physical_threads == 16);
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   struct machine m = {
      .side_effect_address = Q_ADDRESS,
      .local_root = object ? info.key.grid_root : info.key.output_root};
   memset(m.local, 0xcd, sizeof(m.local));
   uint32_t q[Q_BYTES / 4];
   uint8_t written[Q_BYTES] = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(q); ++i)
      q[i] = POISON;
   q[0] = 0;
   m.region[4] = (struct region){Q_ADDRESS, Q_BYTES, (uint8_t *)q, written};
   const struct formation_fixture *f = &h1_fixtures[0];
   for (unsigned i = 0; i < 4; ++i) {
      m.region[i] = (struct region){0x1200000000ull + i * 0x10000, 256,
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
   m.formation_inputs[AGX_MESH_RESERVATION_READY] = 1;
   m.region[6] = (struct region){INPUT_ADDR, sizeof(m.formation_inputs),
                                 (uint8_t *)m.formation_inputs, NULL};
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
   CHECK(m.value);
   execute(&m, &impl->body, ALL_LANES);
   free(m.value);
   const unsigned mask = (1u << threads) - 1;
   CHECK(q[0] == (probe ? threads : 0));
   CHECK(m.atomic_lanes == (probe ? mask : 0));
   for (unsigned i = 4; i < 256; ++i)
      CHECK(((uint8_t *)q)[i] == 0xa5 && !written[i]);
   for (unsigned lane = 0; lane < LANES; ++lane) {
      uint32_t expected[16] = {
         lane,
         lane % x,
         (lane / x) % y,
         lane / (x * y),
         x,
         y,
         z,
         32,
         lane,
         0,
         1,
         mask,
         lane ^ 0x12345678,
         POISON,
         POISON,
         POISON,
      };
      for (unsigned c = 0; c < 16; ++c) {
         unsigned offset = 256 + 64 * lane + 4 * c;
         bool store = probe && lane < threads && c < 13;
         CHECK(q[offset / 4] == (store ? expected[c] : POISON));
         for (unsigned byte = 0; byte < 4; ++byte)
            CHECK(written[offset + byte] == store);
      }
   }
   for (unsigned i = 256 + 64 * LANES; i < Q_BYTES; ++i)
      CHECK(((uint8_t *)q)[i] == 0xa5 && !written[i]);
   if (object) {
      CHECK(m.local_store_lanes == 1 && m.object_calls == 1);
      CHECK(m.grid[0] == 1 && m.grid[1] == 1 && m.grid[2] == 1);
      CHECK(m.grid_fences == ALL_LANES && m.grid_exports == 7);
      CHECK(m.global_store_lanes == (probe ? mask : 0));
   } else {
      CHECK(m.controls == ALL_LANES && m.scalars == 1);
      for (unsigned c = 0; c < 4; ++c)
         CHECK(m.channels[c] == 7);
      CHECK(m.local_store_lanes == ((1u << MIN2(threads, 3)) - 1));
      CHECK(m.global_store_lanes == (probe ? mask : 1));
      uint32_t xyz[3][3];
      bool dropped;
      CHECK(source_geometry(f, xyz, &dropped) && !dropped);
      unsigned length = 0;
      while (length < 256 && m.region[0].written[length])
         ++length;
      check_geometry(f, m.region[0].data, length, xyz);
      CHECK(references_match(f, &m));
   }
   char name[80];
   snprintf(name, sizeof(name), "%s_T%u_%ux%ux%u_%s",
            object ? "OBJECT" : "MESH", threads, x, y, z,
            probe ? "PROBE" : "PLAIN");
   if (dir) {
      write_file(dir, name, "expected-q.bin", q, sizeof(q));
      write_file(dir, name, "expected-local.bin", m.local, sizeof(m.local));
      if (!object) {
         for (unsigned i = 0; i < 4; ++i) {
            char ext[64];
            snprintf(ext, sizeof(ext), "expected-data%u.bin", i);
            write_file(dir, name, ext, m.region[i].data, 256);
         }
      }
   }
   for (unsigned i = 0; i < 4; ++i) {
      free(m.region[i].data);
      free(m.region[i].written);
   }
   ralloc_free(nir);
   nir = source(object, x, y, z, probe);
   CHECK(lower(nir, true, &info, &reason));
   struct agx_shader_part part = {0};
   if (!agx_compile_apple9_mesh_lowered(nir, &info.key, &part, &reason)) {
      fprintf(stderr, "%s: %s\n", name, reason);
      exit(1);
   }
   CHECK(part.info.apple9_body_only && part.info.binary_size);
   CHECK(part.info.workgroup_size[0] == 16 &&
         part.info.workgroup_size[1] == 1 && part.info.workgroup_size[2] == 1);
   CHECK(!part.info.scratch_size);
   if (dir) {
      write_file(dir, name, "bin", part.binary, part.info.binary_size);
      char meta[1024];
      int bytes = snprintf(
         meta, sizeof(meta),
         "{\"name\":\"%s\",\"logical_size\":[%u,%u,%u],\"physical_size\":[16,1,1],"
         "\"subgroup_size\":32,\"binary_size\":%u,\"gprs\":%u,\"scratch\":%u,"
         "\"atomic_frame\":%u,\"body_only\":true,\"probe\":%s,\"probe_pointer_uniform\":16,"
         "\"probe_bytes\":2048,\"draw_uniform\":18,\"local_bytes\":128,\"abi_version\":%u}\n",
         name, x, y, z, part.info.binary_size, part.info.nr_gprs,
         part.info.scratch_size, part.info.apple9_atomic_frame_size,
         probe ? "true" : "false", AGX_APPLE9_MESH_ABI_VERSION);
      CHECK(bytes > 0 && bytes < sizeof(meta));
      write_file(dir, name, "json", meta, bytes);
   }
   printf("%s: physical=16 subgroup=32 mask=%04x bytes=%u gprs=%u scratch=%u\n",
          name, mask, part.info.binary_size, part.info.nr_gprs,
          part.info.scratch_size);
   free(part.binary);
   ralloc_free(nir);
}

static void
reject(void)
{
   const unsigned sizes[][3] = {
      {0, 1, 1}, {17, 1, 1}, {32, 1, 1}, {2, 3, 3}, {65535, 65535, 65535},
      {1, 1, 1}};
   for (unsigned object = 0; object < 2; ++object) {
      for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i) {
         nir_shader *n = source(object, 1, 1, 1, false);
         for (unsigned d = 0; d < 3; ++d)
            n->info.workgroup_size[d] = sizes[i][d];
         n->info.workgroup_size_variable = i == ARRAY_SIZE(sizes) - 1;
         struct agx_apple9_mesh_body_info info;
         const char *reason = NULL;
         CHECK(!lower(n, true, &info, &reason));
         CHECK(reason &&
               (strstr(reason, "workgroup") || strstr(reason, "1..16")));
         ralloc_free(n);
      }
      nir_shader *n = source(object, 1, 1, 1, false);
      struct agx_apple9_mesh_body_info info;
      const char *reason = NULL;
      unsigned before = nir_shader_get_entrypoint(n)->ssa_alloc;
      CHECK(!lower(n, false, &info, &reason));
      CHECK(strstr(reason, "disabled") &&
            nir_shader_get_entrypoint(n)->ssa_alloc == before);
      CHECK(n->info.workgroup_size[0] == 1);
      ralloc_free(n);
   }
}

int
main(int argc, char **argv)
{
   CHECK(argc == 1 || (argc == 3 && !strcmp(argv[1], "--export")));
   const char *dir = argc == 3 ? argv[2] : NULL;
   glsl_type_singleton_init_or_ref();
   reject();
   for (unsigned object = 0; object < 2; ++object) {
      for (unsigned threads = 1; threads <= 16; ++threads)
         for (unsigned probe = 0; probe < 2; ++probe)
            check(object, threads, 1, 1, probe,
                  dir && (threads == 1 || threads == 3 || threads == 16)
                     ? dir
                     : NULL);
      const unsigned shapes[][3] = {{1, 3, 1}, {1, 1, 3}, {2, 2, 2},
                                    {2, 3, 2}, {2, 4, 2}, {1, 2, 8}};
      for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i)
         check(object, shapes[i][0], shapes[i][1], shapes[i][2], true, NULL);
   }
   glsl_type_singleton_decref();
   return 0;
}
