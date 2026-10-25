/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "poly/geometry.h"
#include "poly/tessellator.h"
#include "hk_shader.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%d: %s\n", __LINE__, #x);                            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static nir_shader *
source(bool quad, bool inputs, bool partial)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TESS_EVAL, &agx_nir_options, "TES mesh test");
   b.shader->info.io_lowered = true;
   b.shader->info.tess._primitive_mode =
      quad ? TESS_PRIMITIVE_QUADS : TESS_PRIMITIVE_TRIANGLES;
   b.shader->info.tess.spacing = TESS_SPACING_EQUAL;
   nir_def *coord = nir_load_tess_coord(&b);
   nir_def *pos =
      nir_vec4(&b, nir_channel(&b, coord, 0), nir_channel(&b, coord, 1),
               nir_channel(&b, coord, 2), nir_imm_float(&b, 1));
   if (inputs)
      pos = nir_fadd(
         &b, pos,
         nir_load_per_vertex_input(
            &b, 4, 32, nir_imm_int(&b, 2), nir_imm_int(&b, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1}));
   for (unsigned c = 0; c < 4; ++c) {
      if (partial && c == 2)
         continue;
      nir_store_output(
         &b, nir_channel(&b, pos, c), nir_imm_int(&b, 0), .component = c,
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
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
   nir_shader *mesh = hk_nir_tes_mesh(src, &reason);
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

/* Sparse test memory: resolve actual lowered load addresses, including the
 * producer's winding swap, instead of checking a second emission algorithm. */
struct constants {
   uint32_t count, group, lane, patch_size, y, z;
   bool ccw, quad, inputs, partial;
   unsigned phase, loads, index_reads, coord_reads, tcs_reads;
};

enum {
   PARAMS = 0x100000,
   DRAW = 0x200000,
   ALLOCS = 0x300000,
   INDICES = 0x400000,
   COORDS = 0x500000,
   TCS = 0x600000
};

static unsigned
corner(const struct constants *c)
{
   return c->ccw && c->lane ? 3 - c->lane : c->lane;
}

static bool
specialize(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct constants *c = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = NULL;
   if (c->phase == 0) {
      switch (intr->intrinsic) {
      case nir_intrinsic_load_local_invocation_index:
         value = nir_imm_int(b, c->lane);
         break;
      case nir_intrinsic_load_workgroup_id:
         value = nir_imm_ivec3(b, c->group, c->y, c->z);
         break;
      case nir_intrinsic_load_tess_param_buffer_poly:
         value = nir_imm_int64(b, PARAMS);
         break;
      default:
         return false;
      }
   } else {
      if ((intr->intrinsic != nir_intrinsic_load_global_constant &&
           intr->intrinsic != nir_intrinsic_load_global) ||
          !nir_src_is_const(intr->src[0]))
         return false;
      uint64_t addr = nir_src_as_uint(intr->src[0]);
      uint64_t word = 0;
      if (addr >= PARAMS && addr < PARAMS + sizeof(struct poly_tess_params)) {
         switch (addr - PARAMS) {
         case offsetof(struct poly_tess_params, out_draws):
            word = DRAW;
            break;
         case offsetof(struct poly_tess_params, output_patch_size):
            word = c->patch_size;
            break;
         case offsetof(struct poly_tess_params, ccw):
            word = c->ccw;
            break;
         case offsetof(struct poly_tess_params, index_buffer):
            word = INDICES;
            break;
         case offsetof(struct poly_tess_params, patch_coord_buffer):
            word = COORDS;
            break;
         case offsetof(struct poly_tess_params, coord_allocs):
            word = ALLOCS;
            break;
         case offsetof(struct poly_tess_params, tcs_patch_constants):
            word = 0;
            break;
         case offsetof(struct poly_tess_params, tcs_per_vertex_outputs):
            word = VARYING_BIT_POS;
            break;
         case offsetof(struct poly_tess_params, tcs_buffer):
            word = TCS;
            break;
         default:
            CHECK(!"unexpected parameter load");
         }
      } else if (addr == DRAW) {
         word = c->count;
      } else if (c->phase < 2) {
         return false;
      } else if (intr->intrinsic == nir_intrinsic_load_global &&
                 intr->def.num_components == 1 && addr >= INDICES) {
         CHECK(addr == INDICES + ((uint64_t)c->group * 3 + corner(c)) * 4);
         word = 5 * POLY_TES_PATCH_ID_STRIDE + corner(c);
         c->index_reads++;
      } else if (addr == ALLOCS + 5 * 4) {
         word = 8;
      } else if (intr->intrinsic == nir_intrinsic_load_global &&
                 intr->def.num_components == 2) {
         CHECK(addr == COORDS + (8 + corner(c)) * 8);
         value = nir_imm_ivec2(b, corner(c) * 16384, 16384);
         c->coord_reads++;
      } else if (intr->intrinsic == nir_intrinsic_load_global_constant &&
                 intr->def.num_components == 4) {
         CHECK(c->inputs);
         CHECK(addr == TCS + (5 * (6 + c->patch_size * 4) + 6 + 2 * 4) * 4);
         value = nir_imm_vec4(b, 10, 20, 30, 40);
         c->tcs_reads++;
      } else {
         fprintf(stderr, "unexpected memory load 0x%llx\n",
                 (unsigned long long)addr);
         CHECK(false);
      }
      if (!value)
         value = nir_imm_intN_t(b, word, intr->def.bit_size);
      c->loads++;
   }
   nir_def_replace(&intr->def, value);
   return true;
}

static void
optimize(nir_shader *s)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_if, 0);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);
}

static void
levels(void)
{
   /* The Vulkan frontend represents tess levels as ordinary load_input. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TESS_EVAL, &agx_nir_options, "TES levels test");
   b.shader->info.io_lowered = true;
   b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
   nir_def *outer = nir_load_input(
      &b, 4, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER,
                       .num_slots = 1});
   nir_def *inner = nir_load_input(
      &b, 2, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER,
                       .num_slots = 1});
   nir_def *pos =
      nir_vec4(&b, nir_channel(&b, outer, 0), nir_channel(&b, inner, 0),
               nir_imm_float(&b, 0), nir_imm_float(&b, 1));
   nir_store_output(
      &b, pos, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_shader *mesh = lower_immutable(b.shader, true);
   /* A short patch blocks tess-level payload loads as well as coordinates. */
   struct constants c = {.count = 3, .patch_size = 0};
   for (c.phase = 0; c.phase < 3; ++c.phase) {
      bool progress;
      do {
         optimize(mesh);
         progress = false;
         NIR_PASS(progress, mesh, nir_shader_intrinsics_pass, specialize,
                  nir_metadata_none, &c);
      } while (progress);
   }
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            CHECK(intr->intrinsic ==
                  nir_intrinsic_set_vertex_and_primitive_count);
            CHECK(nir_src_as_uint(intr->src[0]) == 0 &&
                  nir_src_as_uint(intr->src[1]) == 0);
         }
      }
   }
   ralloc_free(mesh);
   /* A vec2 spanning components 1..2 exceeds the two inner levels. */
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_input &&
                   nir_intrinsic_io_semantics(intr).location ==
                      VARYING_SLOT_TESS_LEVEL_INNER)
                  nir_intrinsic_set_component(intr, 1);
            }
         }
      }
   }
   CHECK(!lower_immutable(b.shader, false));
   ralloc_free(b.shader);
}

static unsigned bounds_cases;

static void
bounds(struct constants c)
{
   nir_shader *src = source(c.quad, c.inputs, c.partial);
   nir_shader *mesh = lower_immutable(src, true);
   CHECK(mesh->info.mesh.max_vertices_out == 3);
   CHECK(mesh->info.mesh.max_primitives_out == 1);
   CHECK(mesh->info.workgroup_size[0] == 16);
   bool valid = !c.y && !c.z && c.lane < 3 && c.group < c.count / 3 &&
                c.patch_size <= 32 && c.patch_size >= (c.inputs ? 3 : 1);
   for (c.phase = 0; c.phase < 3; ++c.phase) {
      bool progress;
      do {
         optimize(mesh);
         progress = false;
         NIR_PASS(progress, mesh, nir_shader_intrinsics_pass, specialize,
                  nir_metadata_none, &c);
      } while (progress);
   }
   optimize(mesh);
   nir_validate_shader(mesh, "specialized TES mesh bounds test");
   unsigned vertices = 0, primitives = 0, counts = 0, mask = 0;
   nir_foreach_function_impl(impl, mesh) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_store_per_vertex_output: {
               CHECK(valid && nir_src_as_uint(intr->src[1]) == c.lane);
               CHECK(nir_intrinsic_write_mask(intr) == 1);
               unsigned component = nir_intrinsic_component(intr);
               float coord[4] = {corner(&c) * .25f, .25f,
                                 c.quad ? 0 : .75f - corner(&c) * .25f, 1};
               float expected =
                  coord[component] + (c.inputs ? (component + 1) * 10 : 0);
               CHECK(nir_src_as_float(intr->src[0]) == expected);
               mask |= BITFIELD_BIT(component);
               vertices++;
               break;
            }
            case nir_intrinsic_store_per_primitive_output:
               CHECK(valid && c.lane == 0 &&
                     nir_src_as_uint(intr->src[1]) == 0);
               for (unsigned k = 0; k < 3; ++k)
                  CHECK(nir_src_comp_as_uint(intr->src[0], k) == k);
               primitives++;
               break;
            case nir_intrinsic_set_vertex_and_primitive_count:
               CHECK(!c.lane);
               CHECK(nir_src_as_uint(intr->src[0]) == (valid ? 3 : 0));
               CHECK(nir_src_as_uint(intr->src[1]) == (valid ? 1 : 0));
               counts++;
               break;
            default:
               CHECK(!"unresolved intrinsic or unexpected memory access");
            }
         }
      }
   }
   CHECK(vertices == (valid ? (c.partial ? 3 : 4) : 0));
   CHECK(mask == (valid ? (c.partial ? 11 : 15) : 0));
   CHECK(primitives == (valid && !c.lane));
   CHECK(counts == !c.lane);
   CHECK(valid ? c.index_reads > 0 && c.coord_reads > 0
               : c.index_reads == 0 && c.coord_reads == 0 && c.tcs_reads == 0);
   CHECK(c.tcs_reads == (valid && c.inputs));
   if (c.y || c.z || c.lane >= 3)
      CHECK(c.loads == 0);
   ralloc_free(mesh);
   ralloc_free(src);
   bounds_cases++;
}

static void
rejections(void)
{
   for (unsigned k = 0; k < 14; ++k) {
      nir_shader *src = source(false, true, false);
      switch (k) {
      case 0:
         src->info.stage = MESA_SHADER_VERTEX;
         break;
      case 1:
         src->info.tess._primitive_mode = TESS_PRIMITIVE_ISOLINES;
         break;
      case 2:
         src->info.tess.point_mode = true;
         break;
      case 3:
         src->info.writes_memory = true;
         break;
      case 4:
         src->info.outputs_written |= VARYING_BIT_LAYER;
         break;
      case 5:
         src->info.inputs_read |= VARYING_BIT_PSIZ;
         break;
      case 6:
         src->info.outputs_written_16bit = 1;
         break;
      case 7:
         src->info.clip_distance_array_size = 1;
         break;
      case 8:
         src->info.cull_distance_array_size = 1;
         break;
      case 9:
         src->info.io_lowered = false;
         break;
      case 11:
      case 12:
      case 13: {
         nir_function_impl *impl = nir_shader_get_entrypoint(src);
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         if (k == 12) {
            nir_load_tess_level_inner(&b);
         } else if (k == 13) {
            nir_load_tess_level_outer(&b);
         } else {
            nir_def *index = nir_imm_int(&b, 32);
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type == nir_instr_type_intrinsic &&
                      nir_instr_as_intrinsic(instr)->intrinsic ==
                         nir_intrinsic_load_per_vertex_input)
                     nir_src_rewrite(&nir_instr_as_intrinsic(instr)->src[0],
                                     index);
               }
            }
         }
         break;
      }
      case 10: {
         nir_function_impl *impl = nir_shader_get_entrypoint(src);
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         nir_def *dynamic = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                         nir_imm_int(&b, 0), .align_mul = 4);
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_load_per_vertex_input)
                  nir_src_rewrite(&nir_instr_as_intrinsic(instr)->src[0],
                                  dynamic);
            }
         }
         break;
      }
      }
      CHECK(!lower_immutable(src, false));
      ralloc_free(src);
   }
}

static void
compile(bool quad, bool inputs, bool partial)
{
   nir_shader *src = source(quad, inputs, partial);
   struct hk_apple9_graphics_key key = {0};
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   bool compiled =
      hk_apple9_compile_tes_mesh_program(src, &key, &part, &info, &reason);
   if (!compiled)
      fprintf(stderr, "compile quad=%u inputs=%u partial=%u: %s\n", quad,
              inputs, partial, reason);
   CHECK(compiled && part.binary && part.info.binary_size);
   CHECK(info.layout.max_vertices == 3 && info.layout.max_primitives == 1);
   CHECK(
      hk_apple9_native_shader_valid(&part.info, &info, HK_APPLE9_NATIVE_MESH));
   free(part.binary);
   ralloc_free(src);
}

/* Specialize the actual object NIR, so the test checks its address, load width
 * and launch dimensions rather than repeating its count-to-grid expression. */
struct object_case {
   uint32_t count;
   unsigned loads;
};

static bool
specialize_object(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct object_case *c = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value;
   if (intr->intrinsic == nir_intrinsic_load_native_pointer_apple9_agx) {
      CHECK(nir_intrinsic_base(intr) == AGX_APPLE9_MESH_DRAW_UNIFORM);
      value = nir_imm_ivec2(b, DRAW, 0);
   } else if (intr->intrinsic == nir_intrinsic_load_global &&
              nir_src_is_const(intr->src[0])) {
      CHECK(nir_src_as_uint(intr->src[0]) == DRAW);
      CHECK(intr->def.bit_size == 32 && intr->def.num_components == 1);
      CHECK(nir_intrinsic_align_mul(intr) == 4);
      ++c->loads;
      value = nir_imm_int(b, c->count);
   } else {
      return false;
   }
   nir_def_replace(&intr->def, value);
   return true;
}

static void
object_grid(void)
{
   const uint32_t counts[] = {0, 1, 2, 3, 4, 5, 6, 24576, 49152, UINT32_MAX};
   const uint32_t expected[] = {1, 1, 1, 1, 1, 1, 2, 8192, 16384, 1431655765};
   for (unsigned c = 0; c < ARRAY_SIZE(counts); ++c) {
      nir_shader *s = hk_nir_tes_mesh_object();
      CHECK(s && s->info.stage == MESA_SHADER_TASK);
      struct object_case state = {.count = counts[c]};
      bool progress;
      do {
         optimize(s);
         progress = false;
         NIR_PASS(progress, s, nir_shader_intrinsics_pass, specialize_object,
                  nir_metadata_none, &state);
      } while (progress);
      optimize(s);
      unsigned launches = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            CHECK(intr->intrinsic == nir_intrinsic_launch_mesh_workgroups);
            CHECK(nir_src_comp_as_uint(intr->src[0], 0) == expected[c]);
            CHECK(nir_src_comp_as_uint(intr->src[0], 1) == 1);
            CHECK(nir_src_comp_as_uint(intr->src[0], 2) == 1);
            ++launches;
         }
      }
      CHECK(state.loads == 1 && launches == 1);
      ralloc_free(s);
   }
   nir_shader *s = hk_nir_tes_mesh_object();
   const struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   struct agx_shader_part part = {0};
   struct agx_apple9_mesh_body_info info = {0};
   const char *reason = NULL;
   CHECK(
      agx_compile_apple9_native_object(s, true, &key, &part, &info, &reason));
   CHECK(part.binary && part.info.binary_size);
   CHECK(hk_apple9_native_shader_valid(&part.info, &info,
                                       HK_APPLE9_NATIVE_OBJECT));
   free(part.binary);
   ralloc_free(s);
}

int
main(void)
{
   setenv("HK_APPLE9_NATIVE_MESH", "1", 1);
   setenv("HK_APPLE9_NATIVE_MESH_PACKED_SHAPES", "T16V3P1", 1);
   glsl_type_singleton_init_or_ref();
   nir_shader *s = source(false, true, false);
   nir_shader *m = lower_immutable(s, true);
   if (getenv("TES_TEST_DUMP"))
      nir_print_shader(m, stderr);
   ralloc_free(m);
   ralloc_free(s);
   for (unsigned q = 0; q < 2; ++q)
      for (unsigned i = 0; i < 2; ++i)
         for (unsigned p = 0; p < 2; ++p)
            compile(q, i, p);
   for (unsigned count = 0; count <= 10; ++count)
      for (unsigned group = 0; group <= 4; ++group)
         for (unsigned lane = 0; lane <= 4; ++lane)
            for (unsigned ccw = 0; ccw < 2; ++ccw)
               bounds((struct constants){.count = count,
                                         .group = group,
                                         .lane = lane,
                                         .patch_size = 3,
                                         .ccw = ccw,
                                         .inputs = true,
                                         .partial = (group & 1),
                                         .quad = (count & 1)});
   for (unsigned lane = 0; lane < 4; ++lane)
      for (unsigned ccw = 0; ccw < 2; ++ccw) {
         bounds((struct constants){.count = UINT32_MAX,
                                   .group = UINT32_MAX / 3 - 1,
                                   .lane = lane,
                                   .patch_size = 32,
                                   .inputs = true,
                                   .ccw = ccw});
         bounds((struct constants){.count = UINT32_MAX,
                                   .group = UINT32_MAX,
                                   .lane = lane,
                                   .patch_size = 3,
                                   .inputs = true,
                                   .ccw = ccw});
      }
   for (unsigned size = 0; size <= 33; ++size)
      for (unsigned input = 0; input < 2; ++input)
         bounds((
            struct constants){.count = 3, .patch_size = size, .inputs = input});
   bounds((struct constants){.count = 6, .group = 2, .patch_size = 3});
   bounds((struct constants){.count = 6, .patch_size = 3, .y = 1});
   bounds((struct constants){.count = 6, .patch_size = 3, .z = 1});
   object_grid();
   rejections();
   levels();
   printf(
      "%u bounds/winding/partial-output cases, 15 immutable rejections, tess-level bounds, 8 native compilations, 10 object grid cases + object compilation passed\n",
      bounds_cases);
   glsl_type_singleton_decref();
   return 0;
}
