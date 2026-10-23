/* SPDX-License-Identifier: MIT
 * Compare a bounded allocation with the same completed allocation. The loop
 * keeps independently evolving phi values live, forcing real spill/repair.
 */
#include "../agx_apple9_ra.c"
#include "compiler/nir/nir_builder.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)

static void
check_pressure(unsigned count, bool banks, bool latency, bool masks)
{
   nir_builder nir = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "spill budget");
   nir.shader->info.workgroup_size[0] = 32;
   nir.shader->info.workgroup_size[1] = nir.shader->info.workgroup_size[2] = 1;
   nir_block logical[4] = {0};
   logical[0].successors[0] = &logical[1];
   logical[1].successors[0] = &logical[2];
   logical[1].successors[1] = &logical[3];
   logical[2].successors[0] = &logical[1];
   struct agx_apple9_vir_program p;
   agx_apple9_vir_init(&p);
   struct agx_apple9_block *blocks[4];
   for (unsigned i = 0; i < 4; ++i) {
      blocks[i] = agx_apple9_block_create(&p);
      blocks[i]->nir = &logical[i];
   }
   agx_apple9_block_begin(&p, blocks[0]);
   uint32_t phi[128];
   for (unsigned i = 0; i < count; ++i) {
      uint32_t initial =
         agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM, AGX_APPLE9_ENC_MOV_IMM32,
                             NULL, 0, (i * 0x9e3779b9u) ^ 0xa511e9b3u);
      phi[i] = agx_apple9_vir_emit_phi(&p, blocks[1]);
      CHECK(agx_apple9_vir_emit_phi_source(&p, phi[i], initial));
   }
   if (masks) {
      agx_apple9_block_begin(&p, agx_apple9_block_create(&p));
      CHECK(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_LOOP_MASK_PUSH, AGX_APPLE9_ENC_LOOP_MASK_PUSH,
         NULL, 0, 0));
   }
   agx_apple9_block_begin(&p, blocks[1]);
   agx_apple9_block_begin(&p, blocks[2]);
   for (unsigned i = 0; i < count; ++i) {
      uint32_t src[] = {phi[i], phi[(i + 1) % count]};
      uint32_t next = agx_apple9_vir_emit(
         &p, AGX_APPLE9_VIR_IADD, AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
      CHECK(agx_apple9_vir_emit_phi_source(&p, phi[i], next));
   }
   if (masks) {
      agx_apple9_block_begin(&p, agx_apple9_block_create(&p));
      CHECK(agx_apple9_vir_emit_side_effect(
         &p, AGX_APPLE9_VIR_LOOP_MASK_POP, AGX_APPLE9_ENC_LOOP_MASK_POP,
         NULL, 0, 0));
   }
   agx_apple9_block_begin(&p, blocks[3]);
   uint32_t hash = phi[0];
   for (unsigned i = 1; i < count; ++i) {
      uint32_t src[] = {hash, phi[i]};
      hash = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IADD,
                                 AGX_APPLE9_ENC_INT_ADD_EXTENDED, src, 2, 0);
   }
   uint32_t index = agx_apple9_vir_emit(&p, AGX_APPLE9_VIR_IMM,
                                        AGX_APPLE9_ENC_MOV_IMM32, NULL, 0, 0);
   CHECK(agx_apple9_vir_emit_device_store(&p, 0, index, &hash, 1, 32));
   agx_apple9_place_phis(&p);
   const char *reason = NULL;
   CHECK(agx_apple9_assign_vir_scoreboard_slots(&p, &reason));
   agx_apple9_vir_reindex(&p);
   CHECK(agx_apple9_allocate_publications(&p, &reason));
   struct agx_apple9_vir_program full, bounded;
   agx_apple9_vir_init(&full);
   agx_apple9_vir_init(&bounded);
   CHECK(apple9_allocate_candidate(&p, nir.shader, banks, latency, 0, true,
                                   true, UINT64_MAX, &full, &reason, NULL, NULL,
                                   NULL));
   uint64_t cost = apple9_allocation_cost(&full).spills;
   CHECK(count <= 96 || cost > 0);
   CHECK(apple9_allocate_candidate(&p, nir.shader, banks, latency, 0, true,
                                   true, cost, &bounded, &reason, NULL, NULL,
                                   NULL));
   CHECK(full.instruction_count == bounded.instruction_count);
   for (unsigned i = 0; i < full.instruction_count; ++i) {
      struct agx_apple9_packed_instruction a = {0}, b = {0};
      CHECK(agx_apple9_pack_vir_instruction(full.instructions[i], full.phys, &a,
                                            &reason));
      CHECK(agx_apple9_pack_vir_instruction(bounded.instructions[i],
                                            bounded.phys, &b, &reason));
      CHECK(a.length == b.length && !memcmp(a.bytes, b.bytes, a.length));
   }
   agx_apple9_vir_finish(&bounded);
   agx_apple9_vir_init(&bounded);
   if (cost) {
      CHECK(!apple9_allocate_candidate(&p, nir.shader, banks, latency, 0, true,
                                       true, 0, &bounded, &reason, NULL, NULL,
                                       NULL));
      CHECK(reason && strstr(reason, "spill budget"));
      CHECK(!bounded.instruction_count);
   }
   agx_apple9_vir_finish(&full);
   agx_apple9_vir_finish(&bounded);
   agx_apple9_vir_finish(&p);
   ralloc_free(nir.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const unsigned pressure[] = {8, 96, 128};
   for (unsigned i = 0; i < ARRAY_SIZE(pressure); ++i)
      for (unsigned banks = 0; banks < 2; ++banks)
         for (unsigned latency = 0; latency < 2; ++latency)
            for (unsigned masks = 0; masks < 2; ++masks)
               check_pressure(pressure[i], banks, latency, masks);
   glsl_type_singleton_decref();
   puts(
      "spill budgets preserve completed allocations and reject mandatory traffic above budget");
   return 0;
}
