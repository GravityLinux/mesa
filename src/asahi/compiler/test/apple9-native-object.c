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
   check_rejected_caller(1, false);
   check_rejected_caller(32, false);
   check_rejected_caller(16, true);
   glsl_type_singleton_decref();
   return 0;
}
