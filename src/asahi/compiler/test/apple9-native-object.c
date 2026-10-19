/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Host validation of generated object/grid calls. Hardware portability and
 * frame backing are deliberately not inferred from these host checks.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_apple9_mesh.h"
#include "agx_nir.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #x);                            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)
#include "apple9-mesh-executor.h"

static void
check_callee_local_index(const struct agx_shader_part *part, unsigned target)
{
   const uint8_t *bytes = part->binary;
   bool narrow = false;
   /* Search only the emitted callee, not the API-mask read in the caller.
    * The 16-bit SR result must have its zero-extension companion before
    * selection consumes it in a 32-bit comparison/address calculation. */
   for (unsigned i = target; i + 4 <= part->info.binary_size; i += 2) {
      if (!memcmp(bytes + i, "\x8f\x12\x54\x00", 4))
         break;
      if (bytes[i + 1] != 0xa7 || (bytes[i + 2] & 0x3f) != 0x10 ||
          bytes[i + 3] != 0x06)
         continue;
      CHECK((bytes[i] & 15) != 0x0c);
      if ((bytes[i] & 15) != 0x04)
         continue;
      CHECK(i + 8 <= part->info.binary_size);
      CHECK(bytes[i + 4] == ((bytes[i] & 0xf0) | 0x03));
      CHECK(bytes[i + 5] == 0 &&
            bytes[i + 6] == (bytes[i + 2] & 0xc0) && bytes[i + 7] == 1);
      narrow = true;
   }
   CHECK(narrow);
}

static void
run(nir_shader *nir, struct machine *m)
{
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   memset(m->previous, 0, sizeof(m->previous));
   m->value = calloc(impl->ssa_alloc, sizeof(*m->value));
   CHECK(m->value);
   execute(m, &impl->body, ALL_LANES);
   free(m->value);
   m->value = NULL;
}

static void
check_variant(unsigned root, unsigned offset, const uint32_t groups[3])
{
   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   key.grid_root = root;
   key.grid_offset_B = offset;
   nir_shader *body = agx_nir_apple9_taskless_object(&key);
   CHECK(body);
   CHECK(body->info.workgroup_size[0] == 1);
   struct agx_apple9_mesh_body_info info;
   const char *reason = NULL;
   CHECK(agx_nir_lower_apple9_native_object(body, true, &key, &info, &reason));
   struct machine m = {.local_root = root};
   memset(m.local, 0xcd, sizeof(m.local));
   m.region[6] = (struct region){INPUT_ADDR, 12, (uint8_t *)groups, NULL};
   run(body, &m);
   CHECK(m.barriers == 1 && m.grid_exports == 7 && m.object_calls == 1);
   CHECK(m.local_store_lanes == 1);
   CHECK(!memcmp(m.grid, groups, sizeof(m.grid)));
   for (unsigned i = 0; i < sizeof(m.local); ++i) {
      uint8_t expected = i >= offset && i < offset + 12
                            ? ((const uint8_t *)groups)[i - offset]
                            : 0xcd;
      CHECK(m.local[i] == expected);
   }
   /* Execute the same ordinary NIR callee that selection now sees. */
   CHECK(agx_nir_lower_apple9_object_calls(body, &reason));
   nir_validate_shader(body, "expanded closed object call");
   m.barriers = m.object_calls = m.grid_fences = m.grid_exports = 0;
   run(body, &m);
   CHECK(m.barriers == 1 && m.object_calls == 1);
   CHECK(m.grid_fences == ALL_LANES && m.grid_exports == 7);
   CHECK(!memcmp(m.grid, groups, sizeof(m.grid)));
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "test_object_export");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   CHECK(agx_nir_apple9_object_grid_export(&b, &key));
   m.grid_fences = m.grid_exports = 0;
   run(b.shader, &m);
   CHECK(m.grid_fences == ALL_LANES && m.grid_exports == 7);
   CHECK(!memcmp(m.grid, groups, sizeof(m.grid)));
   ralloc_free(body);
   ralloc_free(b.shader);

   struct agx_shader_part part = {0};
   CHECK(!agx_compile_apple9_taskless_object(false, &key, &part, &reason));
   CHECK(!part.binary && reason);
   CHECK(agx_compile_apple9_taskless_object(true, &key, &part, &reason));
   CHECK(part.info.apple9_body_only);
   CHECK(part.info.binary_size && part.info.nr_gprs);
   unsigned calls = 0, returns = 0;
   const uint8_t *bytes = part.binary;
   for (unsigned i = 0; i + 10 <= part.info.binary_size; i += 2) {
      if (memcmp(bytes + i, "\x8f\x10\x54", 3))
         continue;
      uint64_t raw = 0;
      for (unsigned j = 0; j < 6; ++j)
         raw |= (uint64_t)bytes[i + 3 + j] << (8 * j);
      int64_t target = i + ((int64_t)(raw << 16) >> 16);
      CHECK(target > i && target < part.info.binary_size);
      check_callee_local_index(&part, target);
      CHECK(!memcmp(bytes + i + 10, "\x0f\x06\x04\x02\x00\x00", 6));
      ++calls;
   }
   for (unsigned i = 0; i + 4 <= part.info.binary_size; i += 2)
      returns += !memcmp(bytes + i, "\x8f\x12\x54\x00", 4);
   CHECK(calls == 1 && returns == 1);
   printf(
      "object root=%u offset=%u groups=%u,%u,%u bytes=%u call=token-free-provisional\n",
      root, offset, groups[0], groups[1], groups[2],
      part.info.binary_size);
   free(part.binary);
}

static void
check_rejected_caller(unsigned width, bool divergent)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "unsupported_object_caller");
   b.shader->info.workgroup_size[0] = width;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   if (divergent)
      nir_push_if(&b, nir_ult_imm(&b, nir_load_local_invocation_index(&b), 3));
   nir_launch_mesh_workgroups_apple9_agx(&b, .base = 7);
   if (divergent)
      nir_pop_if(&b, NULL);
   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   struct agx_shader_part part = {0};
   const char *reason = NULL;
   CHECK(!agx_compile_apple9_mesh_lowered(b.shader, &key, &part, &reason));
   CHECK(!part.binary && reason);
   CHECK(!strcmp(reason, "object grid calls require a converged T16 leaf caller"));
   ralloc_free(b.shader);
}

static void
check_multiple_calls(unsigned pressure)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "object_calls_live_nir");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *address = nir_pack_64_2x32(&b,
      nir_load_native_pointer_apple9_agx(&b, .base = AGX_APPLE9_MESH_DRAW_UNIFORM));
   nir_def *values[40];
   for (unsigned i = 0; i < pressure; ++i)
      values[i] = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, address, 4 * i),
                                  .align_mul = 4);
   nir_launch_mesh_workgroups_apple9_agx(&b, .base = 7);
   nir_launch_mesh_workgroups_apple9_agx(&b, .base = 10, .range_base = 64);
   /* Both a callee-to-caller CFG edge and live values across two callees. */
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   for (unsigned i = 0; i < pressure; ++i)
      nir_store_global(&b, values[i], nir_iadd_imm(&b, address, 256 + 4 * i),
                       .align_mul = 4);
   nir_pop_if(&b, NULL);
   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   struct agx_shader_part part = {0};
   const char *reason = NULL;
   if (!agx_compile_apple9_mesh_lowered(b.shader, &key, &part, &reason)) {
      fprintf(stderr, "two calls: %s\n", reason);
      exit(1);
   }
   unsigned calls = 0, returns = 0;
   const uint8_t *bytes = part.binary;
   for (unsigned i = 0; i + 10 <= part.info.binary_size; i += 2) {
      if (!memcmp(bytes + i, "\x8f\x10\x54", 3)) {
         uint64_t raw = 0;
         for (unsigned j = 0; j < 6; ++j)
            raw |= (uint64_t)bytes[i + 3 + j] << (8 * j);
         int64_t target = i + ((int64_t)(raw << 16) >> 16);
         CHECK(target > i && target < part.info.binary_size);
         check_callee_local_index(&part, target);
         CHECK(!memcmp(bytes + i + 10, "\x0f\x06\x04\x02\x00\x00", 6));
         ++calls;
      }
   }
   for (unsigned i = 0; i + 4 <= part.info.binary_size; i += 2)
      returns += !memcmp(bytes + i, "\x8f\x12\x54\x00", 4);
   CHECK(calls == 2 && returns == 2 && !part.info.scratch_size);
   printf("two NIR calls: live=%u gprs=%u bytes=%u\n", pressure,
          part.info.nr_gprs, part.info.binary_size);
   free(part.binary);
   ralloc_free(b.shader);
}

static void
check_rejected_region(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "invalid_object_region");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   if (kind == 0) {
      nir_end_object_call_apple9_agx(&b);
   } else if (kind == 7) {
      nir_def *caller = nir_load_local_invocation_index(&b);
      nir_begin_object_call_apple9_agx(&b);
      nir_export_mesh_grid_word_apple9_agx(&b, caller);
      nir_end_object_call_apple9_agx(&b);
   } else {
      nir_begin_object_call_apple9_agx(&b);
      if (kind == 1) {
         nir_begin_object_call_apple9_agx(&b);
      } else if (kind == 2) {
         nir_push_loop(&b);
         nir_jump(&b, nir_jump_break);
         nir_pop_loop(&b, NULL);
      } else if (kind == 3 || kind == 4) {
         nir_def *value = nir_load_local_invocation_index(&b);
         if (kind == 4)
            value = nir_ieq_imm(&b, value, 0);
         nir_end_object_call_apple9_agx(&b);
         if (kind == 3) {
            nir_export_mesh_grid_word_apple9_agx(&b, value);
         } else {
            nir_push_if(&b, value);
            nir_fence_mesh_grid_apple9_agx(&b);
            nir_pop_if(&b, NULL);
         }
      } else if (kind == 5) {
         nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
         nir_def *yes = nir_imm_int(&b, 1);
         nir_push_else(&b, NULL);
         nir_def *no = nir_imm_int(&b, 2);
         nir_pop_if(&b, NULL);
         nir_export_mesh_grid_word_apple9_agx(&b, nir_if_phi(&b, yes, no));
      }
      /* kind 6 deliberately leaves the region unclosed. */
      if (kind != 3 && kind != 4 && kind != 6)
         nir_end_object_call_apple9_agx(&b);
   }
   const char *reason = NULL;
   CHECK(!agx_nir_lower_apple9_object_calls(b.shader, &reason));
   CHECK(reason && !strcmp(reason, "invalid closed NIR object call region"));
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const uint32_t groups[][3] = {{0, 1, 1}, {1, 1, 1}, {2, 3, 4}};
   for (unsigned r = 0; r < 2; ++r)
      for (unsigned off = 0; off <= 64; off += 64)
         for (unsigned i = 0; i < ARRAY_SIZE(groups); ++i)
            check_variant(r ? 10 : 7, off, groups[i]);
   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   key.grid_root = AGX_APPLE9_MESH_DRAW_UNIFORM / 2;
   CHECK(!agx_nir_apple9_taskless_object(&key));
   key.grid_root = 64;
   CHECK(!agx_nir_apple9_taskless_object(&key));
   key.grid_root = 7;
   key.grid_offset_B = 1;
   CHECK(!agx_nir_apple9_taskless_object(&key));
   key.grid_offset_B = 0;
   key.physical_threads = 32;
   CHECK(!agx_nir_apple9_taskless_object(&key));
   for (unsigned i = 0; i < 8; ++i)
      check_rejected_region(i);
   check_multiple_calls(1);
   check_multiple_calls(12);
   check_multiple_calls(40);
   check_rejected_caller(1, false);
   check_rejected_caller(32, false);
   check_rejected_caller(16, true);
   glsl_type_singleton_decref();
   return 0;
}
