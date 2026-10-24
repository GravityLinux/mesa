/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "hk_shader.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%d: %s\n", __LINE__, #x);                            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static nir_shader *
source(unsigned vertices, unsigned ends, bool inputs)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS mesh test");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = vertices;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   nir_def *count =
      nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                   .align_mul = 4, .range = 4);
   for (unsigned i = 0; i < vertices; ++i) {
      nir_push_if(&b, nir_ult(&b, nir_imm_int(&b, i), count));
      nir_def *pos =
         inputs
            ? nir_load_per_vertex_input(
                 &b, 4, 32, nir_imm_int(&b, i % 3), nir_imm_int(&b, 0),
                 .dest_type = nir_type_float32,
                 .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1})
            : nir_imm_vec4(&b, i, i + 1, 0.5, 1);
      /* Include scalar stores, as used by normalized GS I/O. */
      for (unsigned c = 0; c < 4; ++c)
         nir_store_output(
            &b, nir_channel(&b, pos, c), nir_imm_int(&b, 0), .component = c,
            .write_mask = 1, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_emit_vertex(&b, 0);
      nir_pop_if(&b, NULL);
      if (ends & BITFIELD_BIT(i))
         nir_end_primitive(&b, 0);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
lower_immutable(nir_shader *src, bool accepted)
{
   struct blob before, after;
   blob_init(&before);
   blob_init(&after);
   nir_serialize(&before, src, false);
   const char *reason = NULL;
   nir_shader *mesh = hk_nir_gs_mesh(src, &reason);
   if (!mesh && accepted)
      fprintf(stderr, "lowering: %s\n", reason);
   CHECK((mesh != NULL) == accepted);
   nir_serialize(&after, src, false);
   CHECK(before.size == after.size &&
         !memcmp(before.data, after.data, before.size));
   blob_finish(&before);
   blob_finish(&after);
   return mesh;
}

struct constants {
   unsigned count, lane;
};

static bool
specialize(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct constants *c = data;
   if (intr->intrinsic != nir_intrinsic_load_ubo &&
       intr->intrinsic != nir_intrinsic_load_local_invocation_index)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(
      &intr->def,
      nir_imm_int(
         b, intr->intrinsic == nir_intrinsic_load_ubo ? c->count : c->lane));
   return true;
}

static void
emission(unsigned vertices, unsigned count, unsigned ends, unsigned lane)
{
   nir_shader *src = source(MAX2(vertices, count), ends, false);
   src->info.gs.vertices_out = vertices;
   nir_shader *mesh = lower_immutable(src, true);
   CHECK(mesh->info.mesh.max_vertices_out == vertices);
   CHECK(mesh->info.mesh.max_primitives_out == vertices - 2);
   CHECK(mesh->info.workgroup_size[0] == (vertices <= 16 ? 16 : 32));
   struct constants c = {count, lane};
   NIR_PASS(_, mesh, nir_shader_intrinsics_pass, specialize, nir_metadata_none,
            &c);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, mesh, nir_opt_copy_prop);
      NIR_PASS(progress, mesh, nir_opt_constant_folding);
      NIR_PASS(progress, mesh, nir_opt_algebraic);
      NIR_PASS(progress, mesh, nir_opt_if, 0);
      NIR_PASS(progress, mesh, nir_opt_dead_cf);
      NIR_PASS(progress, mesh, nir_opt_dce);
   } while (progress);
   nir_validate_shader(mesh, "specialized GS mesh test");
   /* Check stores as well as final counts: clamping counts alone cannot
    * prevent an out-of-bounds mesh-output write. */
   count = MIN2(count, vertices);
   unsigned expected[32][3], prims = 0, strip = 0;
   for (unsigned i = 0; i < count; ++i) {
      if (strip >= 2) {
         expected[prims][0] = i - 2 + (strip & 1);
         expected[prims][1] = i - 1 - (strip & 1);
         expected[prims++][2] = i;
      }
      strip++;
      if (ends & BITFIELD_BIT(i))
         strip = 0;
   }
   unsigned seen_v = 0, seen_p = 0, seen_count = 0;
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_store_per_vertex_output:
               CHECK(nir_src_as_uint(intr->src[1]) == seen_v);
               CHECK(nir_src_comp_as_float(intr->src[0], 0) == seen_v);
               CHECK(nir_src_comp_as_float(intr->src[0], 1) == seen_v + 1);
               CHECK(nir_src_comp_as_float(intr->src[0], 2) == .5f);
               CHECK(nir_src_comp_as_float(intr->src[0], 3) == 1.f);
               seen_v++;
               break;
            case nir_intrinsic_store_per_primitive_output:
               CHECK(seen_p < prims && nir_src_as_uint(intr->src[1]) == seen_p);
               for (unsigned k = 0; k < 3; ++k)
                  CHECK(nir_src_comp_as_uint(intr->src[0], k) ==
                        expected[seen_p][k]);
               seen_p++;
               break;
            case nir_intrinsic_set_vertex_and_primitive_count:
               CHECK(nir_src_as_uint(intr->src[0]) == (prims ? count : 0));
               CHECK(nir_src_as_uint(intr->src[1]) == prims);
               seen_count++;
               break;
            default:
               CHECK(!"unlowered intrinsic");
            }
         }
      }
   }
   CHECK(seen_v == (lane ? 0 : count));
   CHECK(seen_p == (lane ? 0 : prims));
   CHECK(seen_count == !lane);
   ralloc_free(mesh);
   ralloc_free(src);
}

static void
compile(unsigned v, bool inputs)
{
   nir_shader *src = source(v, 0, inputs);
   struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   bool compiled =
      hk_apple9_compile_gs_mesh_program(src, &key, &part, &info, &reason);
   if (!compiled)
      fprintf(stderr, "compile V%u inputs=%u: %s\n", v, inputs, reason);
   CHECK(compiled && part.binary && part.info.binary_size);
   CHECK(info.layout.max_vertices == v && info.layout.max_primitives == v - 2);
   CHECK(
      hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
   free(part.binary);
   ralloc_free(src);
}

int
main(void)
{
   setenv("HK_APPLE9_NATIVE_MESH", "1", 1);
   setenv("HK_APPLE9_NATIVE_MESH_PACKED_SHAPES", "T16V3P1,T16V4P2", 1);
   glsl_type_singleton_init_or_ref();
   for (unsigned count = 0; count <= 6; ++count)
      for (unsigned ends = 0; ends < 64; ++ends)
         emission(6, count, ends, 0);
   emission(4, 4, 0, 1);
   emission(24, 24, 0, 0);
   for (unsigned ends = 0; ends < 64; ++ends) {
      emission(3, 6, ends, 0);
      emission(4, 6, ends, 0);
   }
   emission(4, 6, 0, 1);
   for (unsigned kind = 0; kind < 6; ++kind) {
      nir_shader *src = source(4, 0, false);
      switch (kind) {
      case 0:
         src->info.gs.invocations = 2;
         break;
      case 1:
         src->info.gs.active_stream_mask = 3;
         break;
      case 2:
         src->info.gs.input_primitive = MESA_PRIM_TRIANGLES_ADJACENCY;
         break;
      case 3:
         src->info.gs.output_primitive = MESA_PRIM_POINTS;
         break;
      case 4:
         src->info.outputs_written |= BITFIELD64_BIT(VARYING_SLOT_VAR0);
         break;
      case 5:
         src->info.writes_memory = true;
         break;
      }
      lower_immutable(src, false);
      ralloc_free(src);
   }
   compile(3, false);
   compile(4, false);
   compile(3, true);
   compile(4, true);
   glsl_type_singleton_decref();
   puts(
      "579 emission cases (including 129 over-emission cases), immutable rejection, dynamic counts and native compilation passed");
   return 0;
}
