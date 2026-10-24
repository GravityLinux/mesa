/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Destination-addressed publication rounds: research selection, emitted
 * collectives, upload validation and state words. CPU only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_shader.h"
#include "hk_apple9_mesh.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); exit(1); } } while (0)

static unsigned clip_distances, cull_distances;

static nir_shader *
source(unsigned logical, unsigned vertices, unsigned primitives, bool attributes,
       bool primitive_attribute)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "native mesh publication rounds");
   nir_shader *nir = b.shader;
   nir->info.io_lowered = true;
   nir->info.workgroup_size[0] = logical;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.mesh.max_vertices_out = vertices;
   nir->info.mesh.max_primitives_out = primitives;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, vertices),
                                    nir_imm_int(&b, primitives), nir_imm_int(&b, 0));
   /* Strided loops cover the declared range from every logical lane. */
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_variable *v = nir_local_variable_create(b.impl, glsl_uint_type(), "v");
   nir_store_var(&b, v, lane, 1);
   nir_push_loop(&b);
   nir_def *i = nir_load_var(&b, v);
   nir_break_if(&b, nir_uge_imm(&b, i, vertices));
   nir_store_per_vertex_output(&b, nir_imm_vec4(&b, -.5, .5, .5, 1), i,
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   if (attributes) {
      for (unsigned c = 0; c < 3; ++c)
         nir_store_per_vertex_output(&b, nir_imm_float(&b, .25 * (c + 1)), i,
            nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + c, .num_slots = 1});
   }
   if (clip_distances + cull_distances) {
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 1, -1, .5, 2), i,
         nir_imm_int(&b, 0), .write_mask = BITFIELD_MASK(MIN2(clip_distances + cull_distances, 4)),
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0, .num_slots = 1});
      if (clip_distances + cull_distances > 4)
         nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 1, 1, 1, 1), i,
            nir_imm_int(&b, 0), .write_mask = BITFIELD_MASK(clip_distances + cull_distances - 4),
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_CLIP_DIST1, .num_slots = 1});
   }
   nir_store_var(&b, v, nir_iadd_imm(&b, i, logical), 1);
   nir_pop_loop(&b, NULL);
   nir_store_var(&b, v, lane, 1);
   nir_push_loop(&b);
   i = nir_load_var(&b, v);
   nir_break_if(&b, nir_uge_imm(&b, i, primitives));
   nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2), i,
      nir_imm_int(&b, 0), .write_mask = 7, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   if (primitive_attribute)
      nir_store_per_primitive_output(&b, i, i, nir_imm_int(&b, 0), .write_mask = 1,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_VAR4, .num_slots = 1});
   nir_store_var(&b, v, nir_iadd_imm(&b, i, logical), 1);
   nir_pop_loop(&b, NULL);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir->info.clip_distance_array_size = clip_distances;
   nir->info.cull_distance_array_size = cull_distances;
   return nir;
}

/* Every channel collective of round r carries destination 32r in selector
 * bits 16+ and source offset plane + 128r; index round r carries 32r in the
 * count's bits 16+ and source offset 128r. */
static void
check_collectives(unsigned logical, unsigned v, unsigned p, unsigned physical)
{
   nir_shader *nir = source(logical, v, p, false, false);
   struct agx_apple9_native_mesh_options options = {
      .physical_threads = physical, .packed_index_publication = true,
      .publication_rounds = true,
   };
   struct agx_apple9_mesh_body_info body;
   const char *why = NULL;
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   bool ok = agx_nir_lower_apple9_native_mesh(nir, true, &options, &body, &why);
   if (!ok) fprintf(stderr, "L%u V%u P%u: %s\n", logical, v, p, why);
   CHECK(ok);
   const struct agx_apple9_mesh_layout *l = &body.layout;
   const unsigned round = MIN2(physical, AGX_APPLE9_PUBLICATION_ROUND);
   unsigned vertex_rounds = DIV_ROUND_UP(v, round);
   unsigned index_rounds = DIV_ROUND_UP(DIV_ROUND_UP(3 * p, 4), round);
   uint64_t channels[4] = {0}, scalars = 0;
   unsigned controls = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_mesh_control_apple9_agx) {
            controls++;
         } else if (intr->intrinsic == nir_intrinsic_mesh_submit_channel_apple9_agx) {
            unsigned selector = nir_src_as_uint(intr->src[0]);
            unsigned offset = nir_src_as_uint(intr->src[1]);
            unsigned c = selector & 0xffff, first = selector >> 16;
            CHECK(c < 4 && first % round == 0);
            unsigned r = first / round;
            CHECK(r < vertex_rounds && !(channels[c] & BITFIELD64_BIT(r)));
            CHECK(offset == l->position_B + 4 * first +
                            l->vertex_pitch_B * l->vertex_component[VARYING_SLOT_POS][c]);
            channels[c] |= BITFIELD64_BIT(r);
         } else if (intr->intrinsic == nir_intrinsic_mesh_submit_scalar_apple9_agx) {
            unsigned offset = nir_src_as_uint(intr->src[1]);
            CHECK(offset % (4 * round) == 0);
            unsigned r = offset / (4 * round);
            CHECK(r < index_rounds && !(scalars & BITFIELD64_BIT(r)));
            if (r) {
               nir_alu_instr *alu = nir_def_as_alu(intr->src[0].ssa);
               CHECK(alu->op == nir_op_ior &&
                     nir_src_as_uint(alu->src[1].src) == (offset / 4) << 16);
            }
            scalars |= BITFIELD64_BIT(r);
         }
      }
   }
   CHECK(controls == 1);
   for (unsigned c = 0; c < 4; ++c)
      CHECK(channels[c] == BITFIELD64_MASK(vertex_rounds));
   CHECK(scalars == BITFIELD64_MASK(index_rounds));
   printf("L%u V%u P%u T%u: vertex_rounds=%u index_rounds=%u\n",
          logical, v, p, physical, vertex_rounds, index_rounds);
   ralloc_free(nir);
}

static void
check(unsigned logical, unsigned v, unsigned p, bool attributes, bool prim,
      unsigned expected)
{
   nir_shader *nir = source(logical, v, p, attributes, prim);
   struct hk_apple9_graphics_key key = {
      .vertex.outputs_flat = BITFIELD64_BIT(VARYING_SLOT_VAR1),
      .vertex.outputs_linear = BITFIELD64_BIT(VARYING_SLOT_VAR2),
   };
   const char *why = NULL;
   bool supported = false;
   CHECK(hk_apple9_native_mesh_check(nir, &key, &supported, &why) == VK_SUCCESS);
   if (supported != (expected != 0))
      fprintf(stderr, "L%u V%u P%u: %s\n", logical, v, p, why);
   CHECK(supported == (expected != 0));
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info body;
   bool ok = hk_apple9_compile_native_mesh_program(nir, &key, &part, &body, &why);
   CHECK(ok == (expected != 0));
   if (!ok) {
      CHECK(why && !part.binary);
      printf("L%u V%u P%u: rejected: %s\n", logical, v, p, why);
   } else {
      CHECK(!body.primitive_broadcast && body.key.physical_threads == expected);
      CHECK(body.publication_rounds);
      CHECK(body.logical_size[0] == logical);
      CHECK(part.info.workgroup_size[0] == expected && !part.info.scratch_size);
      CHECK(hk_apple9_native_shader_valid(&part.info, &body, HK_APPLE9_NATIVE_MESH));
      CHECK(body.layout.vertex_components == (attributes ? 7 : 4));
      /* Uploads cannot reinterpret the binary under another width or size. */
      struct agx_apple9_mesh_body_info stale = body;
      stale.key.physical_threads = expected == 16 ? 32 : 16;
      CHECK(!hk_apple9_native_shader_valid(&part.info, &stale, HK_APPLE9_NATIVE_MESH));
      stale = body; stale.publication_rounds = false;
      CHECK(!hk_apple9_native_shader_valid(&part.info, &stale, HK_APPLE9_NATIVE_MESH));
      stale = body; stale.logical_size[0] = 129;
      CHECK(!hk_apple9_native_shader_valid(&part.info, &stale, HK_APPLE9_NATIVE_MESH));
      stale = body; stale.layout.max_vertices = 257;
      CHECK(!hk_apple9_native_shader_valid(&part.info, &stale, HK_APPLE9_NATIVE_MESH));
      struct agx_apple9_mesh_state_parameters state = {
         .layout = &body.layout, .threads = expected, .has_object_stage = true,
      };
      uint32_t words[AGX_APPLE9_MESH_STATE_WORDS];
      CHECK(agx_apple9_mesh_state_build(words, &state));
      CHECK(words[10] == ((p - 1) | ((DIV_ROUND_UP(p, 16) - 1) << 16)));
      CHECK(words[11] == (0x400800 | expected));
      CHECK(words[13] == 1 + DIV_ROUND_UP(v, 4) * body.layout.vertex_components);
      printf("L%u V%u P%u: T%u planes=%u bytes=%u gprs=%u\n", logical, v, p,
             expected, body.layout.vertex_components, part.info.binary_size,
             part.info.nr_gprs);
   }
   free(part.binary);
   ralloc_free(nir);
}

int
main(int argc, char **argv)
{
   bool research = argc == 2;
   glsl_type_singleton_init_or_ref();
   /* NVIDIA V64/P126 and Wicked V64/P124, Niagara L64 V64/P96, single-entry
    * boundaries, logical 1 and 16 (masked at T32) and the API maxima. */
   const unsigned samples[][3] = {
      {32, 33, 42}, {32, 32, 43}, {32, 64, 126}, {32, 64, 124}, {64, 64, 96},
      {128, 64, 126}, {1, 64, 126}, {16, 40, 20}, {16, 20, 10}, {32, 256, 256}, {100, 59, 101},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(samples); ++i) {
      unsigned logical = samples[i][0], v = samples[i][1], p = samples[i][2];
      /* Multi-round declarations always launch at T32. */
      unsigned physical = 32;
      uint16_t size[3] = {logical, 1, 1};
      CHECK(hk_apple9_native_mesh_round_counts_eligible(size, v, p) == research);
      check_collectives(logical, v, p, physical);
      for (unsigned attributes = 0; attributes < 2; ++attributes)
         check(logical, v, p, attributes, false, research ? physical : 0);
   }
   if (research) {
      /* Shapes within one round keep the class path; out-of-range shapes and
       * independent primitive records stay rejected. */
      uint16_t size[3] = {32, 1, 1};
      CHECK(!hk_apple9_native_mesh_round_counts_eligible(size, 32, 42));
      /* Single-round non-class shapes use the model too; V3/P1 does not. */
      CHECK(hk_apple9_native_mesh_model_counts_eligible(size, 32, 42));
      CHECK(!hk_apple9_native_mesh_model_counts_eligible(size, 3, 1));
      /* Within one round: attributes take the model, position-only stays
       * under the class list (V20/P10 is not a class). */
      check(32, 20, 10, true, false, 32);
      check(32, 20, 10, false, false, 0);
      CHECK(!hk_apple9_native_mesh_round_counts_eligible(size, 257, 126));
      CHECK(!hk_apple9_native_mesh_round_counts_eligible(size, 64, 257));
      size[0] = 129;
      CHECK(!hk_apple9_native_mesh_round_counts_eligible(size, 64, 126));
      check(32, 64, 126, true, true, 0);
      /* ClipDistance: planes linked as varyings plus rasterizer channels
       * 4 + varyings + d in every round; PPP sees the count. */
      for (clip_distances = 1; clip_distances <= 8; clip_distances += 7) {
         nir_shader *nir = source(32, 64, 126, false, false);
         struct hk_apple9_graphics_key key = {0};
         struct agx_shader_part part = {0};
         struct agx_apple9_mesh_body_info body;
         const char *why = NULL;
         bool ok = hk_apple9_compile_native_mesh_program(nir, &key, &part, &body, &why);
         if (!ok) fprintf(stderr, "clip %u: %s\n", clip_distances, why);
         CHECK(ok && body.clip_distances == clip_distances);
         CHECK(part.info.apple9_clip_distance_count == clip_distances);
         CHECK(part.info.apple9_varyings.count == 0 && body.layout.vertex_components == 4 + clip_distances);
         CHECK(hk_apple9_native_shader_valid(&part.info, &body, HK_APPLE9_NATIVE_MESH));
         struct agx_apple9_mesh_body_info stale = body;
         stale.clip_distances = 0;
         CHECK(!hk_apple9_native_shader_valid(&part.info, &stale, HK_APPLE9_NATIVE_MESH));
         printf("clip %u: T%u bytes=%u\n", clip_distances, body.key.physical_threads,
                part.info.binary_size);
         free(part.binary);
         ralloc_free(nir);
      }
      clip_distances = 0;
      /* Cull distances: evaluated at completion from compiler-owned shared
       * storage; not published, so the planes/PPP count exclude them. */
      const unsigned combos[][2] = {{0, 1}, {2, 2}, {1, 7}};
      for (unsigned k = 0; k < ARRAY_SIZE(combos); ++k) {
         clip_distances = combos[k][0];
         cull_distances = combos[k][1];
         nir_shader *nir = source(32, 64, 126, false, false);
         struct hk_apple9_graphics_key key = {0};
         struct agx_shader_part part = {0};
         struct agx_apple9_mesh_body_info body;
         const char *why = NULL;
         bool ok = hk_apple9_compile_native_mesh_program(nir, &key, &part, &body, &why);
         if (!ok) fprintf(stderr, "clip %u cull %u: %s\n", clip_distances, cull_distances, why);
         CHECK(ok && body.clip_distances == clip_distances &&
               body.cull_distances == cull_distances);
         CHECK(body.layout.vertex_components == 4 + clip_distances);
         CHECK(part.info.apple9_clip_distance_count == clip_distances);
         CHECK(body.shared_size_B >= 64 * 4 * cull_distances);
         CHECK(hk_apple9_native_shader_valid(&part.info, &body, HK_APPLE9_NATIVE_MESH));
         printf("clip %u cull %u: T%u bytes=%u shared=%u\n", clip_distances, cull_distances,
                body.key.physical_threads, part.info.binary_size, body.shared_size_B);
         free(part.binary);
         ralloc_free(nir);
      }
      clip_distances = cull_distances = 0;
      /* A fragment shader reading gl_ClipDistance links the distances as
       * varyings; the rasterizer copy reads mirror planes, so planes still
       * equal 4 + varyings + clip distances (the PPP output count). */
      for (clip_distances = 1; clip_distances <= 5; clip_distances += 4) {
         nir_shader *nir = source(32, 64, 126, false, false);
         struct hk_apple9_graphics_key key = {.fs_reads_clip_distance = true};
         struct agx_shader_part part = {0};
         struct agx_apple9_mesh_body_info body;
         const char *why = NULL;
         bool ok = hk_apple9_compile_native_mesh_program(nir, &key, &part, &body, &why);
         if (!ok) fprintf(stderr, "linked clip %u: %s\n", clip_distances, why);
         CHECK(ok && body.clip_linked && body.clip_distances == clip_distances);
         CHECK(part.info.apple9_varyings.count == clip_distances);
         CHECK(body.layout.vertex_components == 4 + 2 * clip_distances);
         CHECK(part.info.apple9_varyings.mask[VARYING_SLOT_CLIP_DIST0]);
         CHECK(hk_apple9_native_shader_valid(&part.info, &body, HK_APPLE9_NATIVE_MESH));
         printf("linked clip %u: planes=%u varyings=%u\n", clip_distances,
                body.layout.vertex_components, part.info.apple9_varyings.count);
         free(part.binary);
         ralloc_free(nir);
      }
      clip_distances = 0;
   }
   glsl_type_singleton_decref();
   return 0;
}
