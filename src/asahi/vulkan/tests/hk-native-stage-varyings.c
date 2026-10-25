/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "poly/geometry.h"
#include "poly/tessellator.h"
#include "hk_shader.h"
#include "hk_apple9_mesh.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); abort(); } } while (0)

static void
store(nir_builder *b, nir_def *v, unsigned slot, unsigned component, nir_alu_type type)
{
   nir_store_output(b, v, nir_imm_int(b, 0), .component = component,
                    .write_mask = BITFIELD_MASK(v->num_components), .src_type = type,
                    .io_semantics = {.location = slot, .num_slots = 1});
}

static nir_shader *
source(bool gs, unsigned vertices, bool quad)
{
   nir_builder b = nir_builder_init_simple_shader(
      gs ? MESA_SHADER_GEOMETRY : MESA_SHADER_TESS_EVAL, &agx_nir_options,
      "native stage varying test");
   b.shader->info.io_lowered = true;
   if (gs) {
      b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
      b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
      b.shader->info.gs.vertices_out = vertices;
      b.shader->info.gs.invocations = b.shader->info.gs.active_stream_mask = 1;
   } else {
      b.shader->info.tess._primitive_mode = quad ? TESS_PRIMITIVE_QUADS : TESS_PRIMITIVE_TRIANGLES;
      b.shader->info.tess.spacing = TESS_SPACING_EQUAL;
   }
   for (unsigned i = 0; i < (gs ? vertices : 1); ++i) {
      nir_def *input = nir_load_per_vertex_input(
         &b, 1, 32, nir_imm_int(&b, gs ? i % 3 : 2), nir_imm_int(&b, 0),
         .component = 1, .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR7, .num_slots = 1});
      nir_def *patch = gs ? nir_fadd_imm(&b, input, .5f) : nir_load_input(
         &b, 1, 32, nir_imm_int(&b, 0), .component = 1,
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_PATCH2, .num_slots = 1});
      nir_def *linear = gs ? nir_imm_float(&b, .25f * i)
                           : nir_channel(&b, nir_load_tess_coord(&b), 0);
      store(&b, nir_imm_vec4(&b, i, i + 1, .5, 1), VARYING_SLOT_POS, 0, nir_type_float32);
      /* A vector followed by a scalar overwrite must preserve component 1. */
      store(&b, gs ? nir_vec2(&b, input, nir_imm_float(&b, -1)) : input,
            VARYING_SLOT_VAR0, 1, nir_type_float32);
      store(&b, patch, VARYING_SLOT_VAR0, 2, nir_type_float32);
      store(&b, nir_imm_int(&b, 0x12345000u + i), VARYING_SLOT_VAR3, 2, nir_type_uint32);
      store(&b, linear, VARYING_SLOT_VAR5, 0, nir_type_float32);
      if (gs)
         nir_emit_vertex(&b, 0);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

struct region { uint64_t base; const void *data; size_t size; };
struct bindings {
   unsigned lane, group, reads;
   struct region regions[9];
};
enum { GP = 0x100000, VP = 0x200000, VS = 0x300000, TP = 0x400000,
       DRAW = 0x500000, INDICES = 0x600000, COORDS = 0x700000,
       ALLOCS = 0x800000, TCS = 0x900000 };

static bool
specialize(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct bindings *s = data;
   nir_def *v = NULL;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_workgroup_id: v = nir_imm_ivec3(b, s->group, 0, 0); break;
   case nir_intrinsic_load_local_invocation_index: v = nir_imm_int(b, s->lane); break;
   case nir_intrinsic_load_provoking_last: v = nir_imm_int(b, 1); break;
   case nir_intrinsic_load_geometry_param_buffer_poly: v = nir_imm_int64(b, GP); break;
   case nir_intrinsic_load_vertex_param_buffer_poly: v = nir_imm_int64(b, VP); break;
   case nir_intrinsic_load_tess_param_buffer_poly: v = nir_imm_int64(b, TP); break;
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_global: {
      if (!nir_src_is_const(intr->src[0]))
         return false;
      uint64_t addr = nir_src_as_uint(intr->src[0]);
      unsigned bytes = intr->def.bit_size / 8;
      for (unsigned i = 0; i < ARRAY_SIZE(s->regions); ++i) {
         struct region *r = &s->regions[i];
         unsigned size = bytes * intr->def.num_components;
         if (addr < r->base || size > r->size || addr - r->base > r->size - size)
            continue;
         nir_def *values[NIR_MAX_VEC_COMPONENTS];
         for (unsigned c = 0; c < intr->def.num_components; ++c) {
            uint64_t word = 0;
            memcpy(&word, (const uint8_t *)r->data + addr - r->base + c * bytes, bytes);
            values[c] = nir_imm_intN_t(b, word, intr->def.bit_size);
         }
         v = nir_vec(b, values, intr->def.num_components);
         s->reads += r->base == VS || r->base == TCS;
         break;
      }
      if (!v)
         fprintf(stderr, "unmapped stage input %#llx\n", (unsigned long long)addr);
      CHECK(v);
      break;
   }
   default: return false;
   }
   nir_def_replace(&intr->def, v);
   return true;
}

static void
check_values(bool gs, unsigned vertices, bool quad, unsigned group, unsigned lane)
{
   const bool packed = gs ? hk_apple9_native_gs_pack_enabled() : hk_apple9_native_tes_pack_enabled();
   unsigned pack = packed ? 8 : 1;
   nir_shader *src = source(gs, vertices, quad);
   struct blob before, after;
   blob_init(&before); blob_init(&after);
   nir_serialize(&before, src, false);
   const char *reason = NULL;
   nir_shader *mesh = gs ? hk_nir_gs_mesh(src, &reason) : hk_nir_tes_mesh(src, &reason);
   if (!mesh) fprintf(stderr, "lower: %s\n", reason);
   CHECK(mesh);
   nir_serialize(&after, src, false);
   CHECK(before.size == after.size && !memcmp(before.data, after.data, before.size));
   blob_finish(&before); blob_finish(&after);

   struct poly_geometry_params gp = {.input_topology = MESA_PRIM_TRIANGLES};
   struct poly_vertex_params vp = {.grid = {33, 1, 1}, .verts_per_instance = 33,
      .output_buffer = VS, .outputs = VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR7)};
   float vs[33 * 8], tcs[42];
   for (unsigned i = 0; i < ARRAY_SIZE(vs); ++i) vs[i] = i + .25f;
   for (unsigned i = 0; i < ARRAY_SIZE(tcs); ++i) tcs[i] = i + .125f;
   struct poly_tess_params tp = {.out_draws = DRAW, .output_patch_size = 3,
      .patches_per_instance = 1, .index_buffer = INDICES,
      .patch_coord_buffer = COORDS, .coord_allocs = ALLOCS, .tcs_buffer = TCS,
      .tcs_per_vertex_outputs = vp.outputs, .tcs_patch_constants = 3};
   uint32_t count = 33, alloc = 0, indices[33];
   struct poly_tess_point coords[3] = {{0, 16384}, {16384, 16384}, {32768, 16384}};
   for (unsigned i = 0; i < ARRAY_SIZE(indices); ++i) indices[i] = i % 3;
   struct bindings s = {.lane = lane, .group = group, .regions = {
      {GP, &gp, sizeof(gp)}, {VP, &vp, sizeof(vp)}, {VS, vs, sizeof(vs)},
      {TP, &tp, sizeof(tp)}, {DRAW, &count, sizeof(count)}, {INDICES, indices, sizeof(indices)},
      {COORDS, coords, sizeof(coords)}, {ALLOCS, &alloc, sizeof(alloc)}, {TCS, tcs, sizeof(tcs)}}};
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, mesh, nir_shader_intrinsics_pass, specialize, nir_metadata_none, &s);
      NIR_PASS(progress, mesh, nir_opt_copy_prop);
      NIR_PASS(progress, mesh, nir_opt_constant_folding);
      NIR_PASS(progress, mesh, nir_opt_algebraic);
      NIR_PASS(progress, mesh, nir_opt_if, 0);
      NIR_PASS(progress, mesh, nir_opt_dead_cf);
      NIR_PASS(progress, mesh, nir_opt_dce);
   } while (progress);
   nir_validate_shader(mesh, "specialized stage varyings");
   unsigned triangle = group * pack + (gs ? (packed ? lane : 0) : lane / 3);
   bool active = triangle < 11 && lane < (gs ? pack : 3 * pack);
   unsigned seen = 0;
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            CHECK(intr->intrinsic == nir_intrinsic_store_per_vertex_output ||
                  intr->intrinsic == nir_intrinsic_store_per_primitive_output ||
                  intr->intrinsic == nir_intrinsic_set_vertex_and_primitive_count);
            if (intr->intrinsic != nir_intrinsic_store_per_vertex_output) continue;
            unsigned loc = nir_intrinsic_io_semantics(intr).location;
            if (loc == VARYING_SLOT_POS) continue;
            CHECK(active);
            unsigned vertex = nir_src_as_uint(intr->src[1]);
            unsigned emitted = gs ? vertex - (packed ? lane * vertices : 0) : 0;
            CHECK(gs ? emitted < vertices : vertex == lane);
            u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
               unsigned component = nir_intrinsic_component(intr) + c;
               CHECK(nir_src_is_const(intr->src[0]));
               if (loc == VARYING_SLOT_VAR3) {
                  CHECK(component == 2 && nir_intrinsic_src_type(intr) == nir_type_uint32);
                  CHECK(nir_src_comp_as_uint(intr->src[0], c) == 0x12345000u + emitted);
               } else {
                  float expected;
                  if (loc == VARYING_SLOT_VAR0) {
                     CHECK(component == 1 || component == 2);
                     expected = gs ? (triangle * 3 + emitted % 3) * 8 + 5.25f
                                   : component == 1 ? 39.125f : 15.125f;
                     if (gs && component == 2) expected += .5f;
                  } else {
                     CHECK(loc == VARYING_SLOT_VAR5 && component == 0);
                     expected = .25f * (gs ? emitted : lane % 3);
                  }
                  float actual = nir_src_comp_as_float(intr->src[0], c);
                  if (actual != expected)
                     fprintf(stderr, "gs=%u packed=%u group=%u lane=%u vertex=%u loc=%u component=%u: %f != %f\n",
                             gs, packed, group, lane, vertex, loc, component, actual, expected);
                  CHECK(actual == expected);
               }
               seen++;
            }
         }
      }
   }
   CHECK(seen == (active ? 4 * (gs ? vertices : 1) : 0));
   /* Dead packed lanes must not read another primitive's user data. */
   CHECK(active ? s.reads > 0 : s.reads == 0);
   ralloc_free(mesh); ralloc_free(src);
}

static void
check_compile(bool gs, unsigned vertices, bool quad)
{
   nir_shader *src = source(gs, vertices, quad);
   struct hk_apple9_graphics_key key = {
      .vertex.outputs_flat = BITFIELD64_BIT(VARYING_SLOT_VAR3),
      .vertex.outputs_linear = BITFIELD64_BIT(VARYING_SLOT_VAR5)};
   struct agx_shader_part part = {0}, fragment = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   bool ok = gs ? hk_apple9_compile_gs_mesh_program(src, &key, &part, &info, &reason)
                : hk_apple9_compile_tes_mesh_program(src, &key, &part, &info, &reason);
   if (!ok) fprintf(stderr, "compile: %s\n", reason);
   CHECK(ok && part.binary && part.info.binary_size);
   CHECK(hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
   CHECK(part.info.apple9_varyings.mask[VARYING_SLOT_VAR0] == 6);
   CHECK(part.info.apple9_varyings.mask[VARYING_SLOT_VAR3] == 4);
   CHECK(part.info.apple9_varyings.mask[VARYING_SLOT_VAR5] == 1);
   CHECK(part.info.apple9_varyings.group[VARYING_SLOT_VAR0] == 0);
   CHECK(part.info.apple9_varyings.group[VARYING_SLOT_VAR3] == 1);
   CHECK(part.info.apple9_varyings.group[VARYING_SLOT_VAR5] == 2);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options, "stage varying consumer");
   b.shader->info.io_lowered = true;
   nir_def *smooth = nir_load_interpolated_input(&b, 2, 32,
      nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH),
      nir_imm_int(&b, 0), .component = 1, .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_def *flat = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0), .component = 2,
      .dest_type = nir_type_uint32, .io_semantics = {.location = VARYING_SLOT_VAR3, .num_slots = 1});
   nir_def *linear = nir_load_interpolated_input(&b, 1, 32,
      nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_NOPERSPECTIVE),
      nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR5, .num_slots = 1});
   store(&b, nir_vec4(&b, nir_channel(&b, smooth, 0), nir_channel(&b, smooth, 1),
                     nir_u2f32(&b, flat), linear), FRAG_RESULT_DATA0, 0, nir_type_float32);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   CHECK(agx_compile_apple9_fragment_inputs(b.shader, &part.info.apple9_varyings, &fragment, &reason));
   CHECK(!memcmp(&part.info.apple9_varyings, &fragment.info.apple9_varyings, sizeof(part.info.apple9_varyings)));
   free(fragment.binary); free(part.binary); ralloc_free(src); ralloc_free(b.shader);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   for (unsigned gs = 0; gs < 2; ++gs) {
      for (unsigned variant = 0; variant < 2; ++variant) {
         check_compile(gs, 3 + variant, variant);
         const unsigned groups[] = {0, 1, 2, 10, 11};
         for (unsigned g = 0; g < ARRAY_SIZE(groups); ++g) {
            /* Unpacked GS dispatches exactly the input primitive count. */
            if (gs && !hk_apple9_native_gs_pack_enabled() && groups[g] == 11)
               continue;
            for (unsigned lane = 0; lane < 32; ++lane)
               check_values(gs, 3 + variant, variant, groups[g], lane);
         }
      }
   }
   glsl_type_singleton_decref();
   puts("stage varyings: input addresses, component masks, integer types, packed tails, immutable sources and native/FS compilation passed");
}
