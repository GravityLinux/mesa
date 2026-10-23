/* SPDX-License-Identifier: MIT
 * Independent behavioral probes, compiled from NIR through normal selection
 * and register allocation. No Metal instruction sequence is an input.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/nir/nir_builder.h"
#include "agx_apple9_mesh.h"
#include "agx_nir.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

struct shape {
   unsigned vertices, primitives, threads, topology, control42, high_index;
   unsigned indices, planes, count_B, position_B, pitch_B, local_B;
};

static void
fence(nir_builder *b)
{
   nir_barrier(b, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

static void
rendezvous(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

static nir_variable *
begin_loop(nir_builder *b, nir_def *start, nir_def *limit)
{
   nir_variable *var =
      nir_local_variable_create(b->impl, glsl_uint_type(), "i");
   nir_store_var(b, var, start, 1);
   nir_push_loop(b);
   nir_break_if(b, nir_uge(b, nir_load_var(b, var), limit));
   return var;
}

static void
end_loop(nir_builder *b, nir_variable *var, unsigned step)
{
   nir_store_var(b, var, nir_iadd_imm(b, nir_load_var(b, var), step), 1);
   nir_pop_loop(b, NULL);
}

static void
local_store(nir_builder *b, nir_def *value, nir_def *address, unsigned bits)
{
   nir_store_local_apple9_agx(
      b, value, address, .base = AGX_APPLE9_MESH_OUTPUT_ROOT, .bit_size = bits);
}

static nir_def *
global_word(nir_builder *b, nir_def *q, unsigned offset)
{
   return nir_load_global(b, 1, 32, nir_iadd_imm(b, q, offset), .align_mul = 4);
}

static void
global_store(nir_builder *b, nir_def *q, nir_def *offset, nir_def *value)
{
   nir_store_global(b, value, nir_iadd(b, q, nir_u2u64(b, offset)),
                    .align_mul = 4);
}

static nir_shader *
source(const struct shape *s)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_MESH, &agx_nir_options,
                                     "independent native capability probe");
   b.shader->info.workgroup_size[0] = s->threads;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = s->vertices;
   b.shader->info.mesh.max_primitives_out = s->primitives;
   b.shader->info.mesh.primitive_type = s->topology == 6   ? MESA_PRIM_TRIANGLES
                                        : s->topology == 1 ? MESA_PRIM_LINES
                                                           : MESA_PRIM_POINTS;
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *q = nir_load_native_pointer_apple9_agx(
      &b, .base = AGX_APPLE9_MESH_PAYLOAD_UNIFORM);
   q = nir_pack_64_2x32(&b, q);
   nir_def *v = nir_umin_imm(&b, global_word(&b, q, 0), s->vertices);
   nir_def *p = nir_umin_imm(&b, global_word(&b, q, 4), s->primitives);
   nir_def *empty =
      nir_ior(&b, nir_ult_imm(&b, v, s->indices), nir_ieq_imm(&b, p, 0));
   v = nir_bcsel(&b, empty, nir_imm_int(&b, 0), v);
   p = nir_bcsel(&b, empty, nir_imm_int(&b, 0), p);
   nir_def *cookie = global_word(&b, q, 8);

   nir_def *visit = nir_iadd_imm(&b, nir_imul_imm(&b, lane, 16), 0x1000);
   global_store(&b, q, visit, nir_imm_int(&b, 0x43415032));
   global_store(&b, q, nir_iadd_imm(&b, visit, 4), lane);
   global_store(&b, q, nir_iadd_imm(&b, visit, 8), cookie);
   global_store(&b, q, nir_iadd_imm(&b, visit, 12),
                nir_imm_int(&b, s->threads));

   /* Define every local byte, including padding, so the copied input snapshot
    * has an exact oracle and never conceals uninitialized count/index data. */
   nir_variable *loop = begin_loop(&b, lane, nir_imm_int(&b, s->local_B / 4));
   local_store(&b, nir_imm_int(&b, 0),
               nir_imul_imm(&b, nir_load_var(&b, loop), 4), 32);
   end_loop(&b, loop, s->threads);
   rendezvous(&b);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   local_store(&b, nir_u2u16(&b, p), nir_imm_int(&b, s->count_B), 16);
   local_store(&b, v, nir_imm_int(&b, s->count_B + 4), 32);
   nir_def *header[] = {v,
                        p,
                        nir_imm_int(&b, s->vertices),
                        nir_imm_int(&b, s->primitives),
                        nir_imm_int(&b, s->threads),
                        nir_imm_int(&b, s->topology),
                        nir_imm_int(&b, s->planes),
                        nir_imm_int(&b, s->local_B)};
   for (unsigned i = 0; i < ARRAY_SIZE(header); ++i)
      global_store(&b, q, nir_imm_int(&b, 0x40 + i * 4), header[i]);
   nir_pop_if(&b, NULL);

   loop = begin_loop(&b, lane, v);
   nir_def *i = nir_load_var(&b, loop);
   nir_def *role = nir_umod_imm(&b, i, s->indices);
   role = nir_bcsel(&b, nir_ieq(&b, i, nir_iadd_imm(&b, v, -1)),
                    nir_imm_int(&b, s->indices - 1), role);
   nir_def *xyzw[5];
   if (s->topology == 6) {
      xyzw[0] = nir_bcsel(&b, nir_ieq_imm(&b, role, 1), nir_imm_float(&b, 3),
                          nir_imm_float(&b, -1));
      xyzw[1] = nir_bcsel(&b, nir_ieq_imm(&b, role, 2), nir_imm_float(&b, 3),
                          nir_imm_float(&b, -1));
   } else if (s->topology == 1) {
      xyzw[0] = nir_bcsel(&b, nir_ieq_imm(&b, role, 1),
                          nir_imm_float(&b, 1 + 1.0 / 64),
                          nir_imm_float(&b, -1 + 1.0 / 64));
      xyzw[1] = nir_imm_float(&b, -1.0 / 64);
   } else {
      xyzw[0] = nir_imm_float(&b, 1.0 / 64);
      xyzw[1] = nir_imm_float(&b, -1.0 / 64);
   }
   /* Independent high-index control: all earlier primitives are degenerate
    * (or outside the viewport for points). Only the final index can produce
    * a covered primitive, and it names the last active vertex. */
   if (s->high_index) {
      nir_def *last = nir_ieq(&b, i, nir_iadd_imm(&b, v, -1));
      if (s->topology == 6) {
         xyzw[1] =
            nir_bcsel(&b, last, nir_imm_float(&b, 3), nir_imm_float(&b, -1));
      } else if (s->topology == 1) {
         xyzw[0] =
            nir_bcsel(&b, last, nir_imm_float(&b, 3), nir_imm_float(&b, -3));
      } else {
         xyzw[0] = nir_bcsel(&b, last, xyzw[0], nir_imm_float(&b, 4));
         xyzw[1] = nir_bcsel(&b, last, xyzw[1], nir_imm_float(&b, 4));
      }
   }
   xyzw[2] = nir_imm_float(&b, 0);
   xyzw[3] = xyzw[4] = nir_imm_float(&b, 1);
   for (unsigned c = 0; c < s->planes; ++c)
      local_store(&b, xyzw[c],
                  nir_iadd_imm(&b, nir_imul_imm(&b, i, 4),
                               s->position_B + c * s->pitch_B),
                  32);
   end_loop(&b, loop, s->threads);

   loop = begin_loop(&b, lane, p);
   i = nir_load_var(&b, loop);
   nir_def *groups = nir_umax_imm(&b, nir_udiv_imm(&b, v, s->indices), 1);
   nir_def *first = s->high_index
                       ? nir_imm_int(&b, 0)
                       : nir_imul_imm(&b, nir_umod(&b, i, groups), s->indices);
   for (unsigned c = 0; c < s->indices; ++c) {
      nir_def *index = nir_iadd_imm(&b, first, c);
      if (c == s->indices - 1)
         index = nir_bcsel(&b, nir_ieq(&b, i, nir_iadd_imm(&b, p, -1)),
                           nir_iadd_imm(&b, v, -1), index);
      local_store(&b, nir_u2u8(&b, index),
                  nir_iadd_imm(&b, nir_imul_imm(&b, i, s->indices), c), 8);
   }
   end_loop(&b, loop, s->threads);
   rendezvous(&b);

   loop = begin_loop(&b, lane, nir_imm_int(&b, s->local_B / 4));
   nir_def *byte = nir_imul_imm(&b, nir_load_var(&b, loop), 4);
   nir_def *word =
      nir_load_local_apple9_agx(&b, byte, .base = AGX_APPLE9_MESH_OUTPUT_ROOT);
   global_store(&b, q, nir_iadd_imm(&b, byte, 0x4000), word);
   end_loop(&b, loop, s->threads);
   fence(&b);
   nir_mesh_control_apple9_agx(&b, nir_u2u16(&b, p), nir_u2u16(&b, v),
                               .base = s->control42);
   nir_push_if(&b, nir_ult(&b, lane, v));
   for (unsigned c = 0; c < s->planes; ++c) {
      nir_mesh_submit_channel_apple9_agx(
         &b, nir_imm_int(&b, c),
         nir_imm_int(&b, s->position_B + c * s->pitch_B));
      fence(&b);
   }
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
   nir_mesh_submit_scalar_apple9_agx(&b, v, nir_imm_int(&b, 0));
   fence(&b);
   nir_pop_if(&b, NULL);
   nir_validate_shader(b.shader, "independent native capability source");
   return b.shader;
}

static void
write_program(const char *prefix, const char *stage,
              const struct agx_shader_part *out)
{
   CHECK(out->info.nr_gprs <= 96 && out->info.binary_size);
   CHECK(!out->info.scratch_size && !out->info.apple9_atomic_frame_size &&
         !out->info.apple9_preamble_size);
   char path[4096];
   snprintf(path, sizeof(path), "%s.%s.bin", prefix, stage);
   FILE *f = fopen(path, "wb");
   CHECK(f &&
         fwrite(out->binary, 1, out->info.binary_size, f) ==
            out->info.binary_size &&
         !fclose(f));
   snprintf(path, sizeof(path), "%s.%s.json", prefix, stage);
   f = fopen(path, "w");
   CHECK(f);
   fprintf(
      f,
      "{\"bytes\":%u,\"gprs\":%u,\"scratch\":%u,\"frame\":%u,\"preamble\":%u,\"publications\":%u,\"compiler_abi\":%u}\n",
      out->info.binary_size, out->info.nr_gprs, out->info.scratch_size,
      out->info.apple9_atomic_frame_size, out->info.apple9_preamble_size,
      out->info.apple9_publication_count, AGX_APPLE9_MESH_ABI_VERSION);
   CHECK(!fclose(f));
}

int
main(int argc, char **argv)
{
   CHECK(argc == 8 || argc == 9);
   struct shape s = {.vertices = atoi(argv[2]),
                     .primitives = atoi(argv[3]),
                     .threads = atoi(argv[4]),
                     .topology = atoi(argv[5]),
                     .control42 = atoi(argv[6]),
                     .high_index = argc == 9 ? atoi(argv[8]) : 0};
   CHECK(s.vertices && s.vertices <= 256 && s.primitives &&
         s.primitives <= 512);
   CHECK(s.threads >= 16 && s.threads <= 256 &&
         util_is_power_of_two_nonzero(s.threads));
   CHECK(s.threads >= s.vertices && s.control42 <= 1 && s.high_index <= 1);
   CHECK(s.topology == 0 || s.topology == 1 || s.topology == 6);
   s.indices = s.topology == 6 ? 3 : s.topology == 1 ? 2 : 1;
   CHECK(!s.high_index || s.vertices > s.indices);
   s.planes = s.topology == 0 ? 5 : 4;
   s.count_B = ALIGN_POT(s.primitives * s.indices, 16);
   s.position_B = s.count_B + 16;
   s.pitch_B = ALIGN_POT(s.vertices, 4) * 4;
   s.local_B = s.position_B + s.pitch_B * s.planes;
   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   key.physical_threads = s.threads;
   struct agx_shader_part code = {0};
   const char *why = NULL;
   nir_shader *nir = source(&s);
   FILE *f = fopen(argv[7], "w");
   CHECK(f);
   nir_print_shader(nir, f);
   CHECK(!fclose(f));
   if (!agx_compile_apple9_mesh_lowered(nir, &key, &code, &why)) {
      fprintf(stderr, "mesh compile: %s\n", why);
      return 1;
   }
   write_program(argv[1], "mesh", &code);
   free(code.binary);
   ralloc_free(nir);
   key.physical_threads = 16;
   if (!agx_compile_apple9_taskless_object(true, &key, &code, &why)) {
      fprintf(stderr, "object compile: %s\n", why);
      return 1;
   }
   write_program(argv[1], "object", &code);
   free(code.binary);
   printf("V%u P%u T%u topology=%u local=%u\n", s.vertices, s.primitives,
          s.threads, s.topology, s.local_B);
   return 0;
}
