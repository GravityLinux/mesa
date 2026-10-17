/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/vulkan/hk_cmd_buffer.h"
#include "asahi/vulkan/hk_descriptor_set.h"
#include "asahi/vulkan/hk_shader.h"
#include "compiler/nir/nir_builder.h"
#include "poly/nir/poly_nir.h"

static void
store(nir_builder *b, unsigned loc, unsigned comp, unsigned n, unsigned word)
{
   nir_def *v =
      nir_load_ubo(b, n, 32, nir_imm_int(b, 0), nir_imm_int(b, word * 4),
                   .align_mul = 4, .range = n * 4);
   nir_store_output(
      b, v, nir_imm_int(b, 0), .write_mask = BITFIELD_MASK(n),
      .component = comp,
      .src_type = (loc == VARYING_SLOT_VIEWPORT || loc == VARYING_SLOT_LAYER)
                     ? nir_type_uint32
                     : nir_type_float32,
      .io_semantics = {.location = loc, .num_slots = 1});
}

/* Same root-pointer lowering as hk's private helper; leave real record loads
 * intact. This does not replace any shader output with a test constant. */
static bool
lower_root(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_root_agx)
      return false;
   unsigned off = offsetof(struct hk_root_descriptor_table, root_desc_addr);
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(
      &intr->def, nir_load_ubo(b, 1, 64, nir_imm_int(b, 0), nir_imm_int(b, off),
                               .align_mul = 8, .range_base = off, .range = 8));
   return true;
}

struct probe_vertex {
   unsigned vertex, last;
};

static bool
constant_vertex(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct probe_vertex *p = data;
   unsigned value;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_id:
      value = p->vertex;
      break;
   case nir_intrinsic_load_instance_id:
      value = 0;
      break;
   case nir_intrinsic_load_provoking_last:
      value = p->last;
      break;
   case nir_intrinsic_store_output:
      if (nir_intrinsic_io_semantics(intr).location !=
          VARYING_SLOT_CULL_PRIMITIVE) {
         nir_instr_remove(&intr->instr);
         return true;
      }
      return false;
   default:
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_int(b, value));
   return true;
}

static bool
mask_semantics(void)
{
   const enum mesa_prim modes[] = {
      MESA_PRIM_POINTS,
      MESA_PRIM_LINE_STRIP,
      MESA_PRIM_TRIANGLE_STRIP,
   };
   const unsigned masks[] = {255, 127, 255, 191, 255, 255};
   for (unsigned written = 0; written <= 8; written += 4)
      for (unsigned mode = 0; mode < ARRAY_SIZE(modes); ++mode)
         for (unsigned split = 0; split < 2; ++split)
            for (unsigned last = 0; last < 2; ++last)
               for (unsigned first = 0; first < 6; ++first) {
                  unsigned n = mesa_vertices_per_prim(modes[mode]);
                  if (first + n > 6 || (split && first < 3 && first + n > 3))
                     continue;
                  nir_builder b = nir_builder_init_simple_shader(
                     MESA_SHADER_GEOMETRY, &agx_nir_options,
                     "cull_mask_semantics");
                  b.shader->info.io_lowered = true;
                  b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
                  b.shader->info.gs.output_primitive = modes[mode];
                  b.shader->info.gs.vertices_in = 1;
                  b.shader->info.gs.vertices_out = 6;
                  b.shader->info.gs.invocations =
                     b.shader->info.gs.active_stream_mask = 1;
                  b.shader->info.cull_distance_array_size = 8;
                  for (unsigned v = 0; v < 6; ++v) {
                     nir_store_output(
                        &b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                        .src_type = nir_type_float32, .write_mask = 15,
                        .io_semantics.location = VARYING_SLOT_POS);
                     for (unsigned c = 0; c < written; ++c)
                        nir_store_output(
                           &b, nir_imm_float(&b, masks[v] & (1 << c) ? -1 : 1),
                           nir_imm_int(&b, c / 4), .src_type = nir_type_float32,
                           .write_mask = 1, .component = c % 4,
                           .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0,
                                            .num_slots = 2});
                     nir_emit_vertex(&b, 0);
                     if (split && v == 2)
                        nir_end_primitive(&b, 0);
                  }
                  nir_shader_gather_info(b.shader,
                                         nir_shader_get_entrypoint(b.shader));
                  nir_shader *count = NULL, *rast = NULL, *pre = NULL;
                  struct poly_gs_info info = {0};
                  poly_nir_lower_gs_cull_mask(b.shader, &count, &rast, &pre,
                                              &info);
                  struct probe_vertex p = {
                     .vertex = last ? first + n - 1 : first,
                     .last = last};
                  nir_shader_intrinsics_pass(rast, constant_vertex,
                                             nir_metadata_none, &p);
                  nir_lower_global_vars_to_local(rast);
                  bool progress;
                  do {
                     progress = false;
                     NIR_PASS(progress, rast, nir_lower_vars_to_ssa);
                     NIR_PASS(progress, rast, nir_opt_copy_prop);
                     NIR_PASS(progress, rast, nir_opt_constant_folding);
                     NIR_PASS(progress, rast, nir_opt_algebraic);
                     NIR_PASS(progress, rast, nir_opt_dead_cf);
                     NIR_PASS(progress, rast, nir_opt_dce);
                  } while (progress);
                  unsigned expected = BITFIELD_MASK(written), actual = 0,
                           stores = 0;
                  for (unsigned v = first; v < first + n; ++v)
                     expected &= masks[v];
                  nir_foreach_block(block, nir_shader_get_entrypoint(rast)) {
                     nir_foreach_instr(instr, block) {
                        if (instr->type != nir_instr_type_intrinsic)
                           continue;
                        nir_intrinsic_instr *intr =
                           nir_instr_as_intrinsic(instr);
                        if (intr->intrinsic == nir_intrinsic_store_output) {
                           if (!nir_src_is_const(intr->src[0]))
                              return false;
                           actual = nir_src_as_uint(intr->src[0]);
                           ++stores;
                        }
                     }
                  }
                  ralloc_free(b.shader);
                  ralloc_free(rast);
                  ralloc_free(pre);
                  ralloc_free(count);
                  if (stores != (written ? 1 : 0) || actual != expected) {
                     fprintf(
                        stderr,
                        "cull mask mode=%u split=%u last=%u first=%u: got %x (%u stores), expected %x\n",
                        modes[mode], split, last, first, actual, stores,
                        expected);
                     return false;
                  }
               }
   return true;
}

// clang-format off
int main(void)
{
   unsigned failed = 0, total = 0;
   glsl_type_singleton_init_or_ref();
   if (!mask_semantics() || HK_APPLE9_OUTPUT_COMPONENTS != 122) return 1;
   puts("stage,api_components,point,written_size,viewport,layer,clip,cull,clips_late,compile,publications,varyings,fragment_compile,reason");
   const unsigned budgets[] = {32, 122};
   for (unsigned t = 0; t < ARRAY_SIZE(budgets); ++t)
   for (unsigned stage = 0; stage < 3; ++stage)
   for (unsigned point = 0; point < 2; ++point)
   for (unsigned psiz = 0; psiz < 2; ++psiz)
   for (unsigned vp = 0; vp < 2; ++vp)
   for (unsigned layer = 0; layer < 2; ++layer)
   for (unsigned clips = 0; clips <= 8; clips += 4)
   for (unsigned culls = 0; culls <= 8; culls += 4)
   for (unsigned late = 0; late < 2; ++late) {
      if (clips + culls > 8) continue;
      unsigned users = budgets[t] - 4 - psiz - vp - layer - clips - culls;
      mesa_shader_stage s = stage == 0 ? MESA_SHADER_VERTEX :
         stage == 1 ? MESA_SHADER_TESS_EVAL : MESA_SHADER_GEOMETRY;
      nir_builder b = nir_builder_init_simple_shader(s, &agx_nir_options, "boundary122");
      if (stage == 1) {
         b.shader->info.tess.point_mode = point;
         b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
      }
      if (stage == 2) {
         b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
         b.shader->info.gs.output_primitive = point ? MESA_PRIM_POINTS : MESA_PRIM_TRIANGLE_STRIP;
         b.shader->info.gs.vertices_in = 1;
         b.shader->info.gs.vertices_out = point ? 1 : 3;
         b.shader->info.gs.invocations = b.shader->info.gs.active_stream_mask = 1;
      }
      b.shader->info.io_lowered = true;
      b.shader->info.num_ubos = 1;
      b.shader->info.clip_distance_array_size = clips;
      b.shader->info.cull_distance_array_size = culls;
      store(&b, VARYING_SLOT_POS, 0, 4, 0);
      unsigned word = 4;
      if (!late) for (unsigned i = 0; i < clips; ++i)
         store(&b, VARYING_SLOT_CLIP_DIST0 + i / 4, i % 4, 1, word++);
      for (unsigned i = 0; i < users; ++i)
         store(&b, VARYING_SLOT_VAR0 + i / 4, i % 4, 1, word++);
      if (psiz) store(&b, VARYING_SLOT_PSIZ, 0, 1, word++);
      if (vp) store(&b, VARYING_SLOT_VIEWPORT, 0, 1, word++);
      if (layer) store(&b, VARYING_SLOT_LAYER, 0, 1, word++);
      for (unsigned i = 0; i < culls; ++i)
         store(&b, VARYING_SLOT_CLIP_DIST0 + (clips + i) / 4, (clips + i) % 4, 1, word++);
      if (late) for (unsigned i = 0; i < clips; ++i)
         store(&b, VARYING_SLOT_CLIP_DIST0 + i / 4, i % 4, 1, word++);
      if (stage == 2) for (unsigned i = 0; i < (point ? 1 : 3); ++i) nir_emit_vertex(&b, 0);
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      nir_shader *compile = b.shader;
      nir_shader *count = NULL, *rast = NULL, *pre = NULL;
      if (stage == 1) poly_nir_lower_tes(b.shader, true);
      if (stage == 2) {
         struct poly_gs_info info = {0};
         poly_nir_lower_gs_cull_mask(b.shader, &count, &rast, &pre, &info);
         compile = rast;
         /* Original GS order is normalized by poly; exercise both legal
          * raster-output store orders without changing its record reads. */
         if (late) {
            nir_function_impl *impl = nir_shader_get_entrypoint(rast);
            nir_instr *moves[8]; unsigned nmoves = 0;
            nir_foreach_block(block, impl) nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic) continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_output) continue;
               unsigned loc = nir_intrinsic_io_semantics(intr).location;
               if (loc != VARYING_SLOT_CLIP_DIST0 && loc != VARYING_SLOT_CLIP_DIST1) continue;
               moves[nmoves++] = instr;
            }
            for (unsigned i = 0; i < nmoves; ++i) {
               nir_instr_remove(moves[i]);
               nir_builder rb = nir_builder_at(nir_after_impl(impl));
               nir_builder_instr_insert(&rb, moves[i]);
            }
         }
         poly_nir_lower_sysvals(rast);
         hk_lower_uvs_index(rast, MESA_SHADER_GEOMETRY, 0);
         nir_shader_intrinsics_pass(rast, lower_root, nir_metadata_control_flow, NULL);
      }
      struct agx_apple9_vertex_layout layout = {
         .cull_distance_mask = true, .rasterize_points = point, .ignore_point_size = !point,
         .clip_distance_enable = 0xff,
      };
      struct agx_shader_part out = {0}; const char *reason = NULL;
      bool ok = agx_compile_apple9_vertex_inputs(compile, &layout, &out, &reason);
      const char *fs_reason = NULL;
      bool fs_ok = false;
      if (ok) {
         nir_builder fs = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
            &agx_nir_options, "complete_boundary_consumer");
         fs.shader->info.io_lowered = true;
         nir_def *sum = nir_imm_float(&fs, 0);
         nir_def *bary = nir_load_barycentric_pixel(&fs, 32,
                                                   .interp_mode = INTERP_MODE_SMOOTH);
         for (unsigned loc = 0; loc < 64; ++loc) {
            if (loc == VARYING_SLOT_CULL_PRIMITIVE || loc == VARYING_SLOT_CULL_PRIMITIVE + 1)
               continue;
            u_foreach_bit(c, out.info.apple9_varyings.mask[loc]) {
               nir_def *v = loc == VARYING_SLOT_VIEWPORT ? nir_load_input(
                  &fs, 1, 32, nir_imm_int(&fs, 0), .component = c,
                  .dest_type = nir_type_uint32,
                  .io_semantics = {.location = loc, .num_slots = 1}) :
                  nir_load_interpolated_input(&fs, 1, 32, bary, nir_imm_int(&fs, 0),
                     .component = c, .dest_type = nir_type_float32,
                     .io_semantics = {.location = loc, .num_slots = 1});
               sum = nir_fadd(&fs, sum, loc == VARYING_SLOT_VIEWPORT ? nir_u2f32(&fs,v) : v);
            }
         }
         nir_store_output(&fs, nir_vec4(&fs, sum, sum, sum, nir_imm_float(&fs,1)),
            nir_imm_int(&fs,0), .write_mask=0xf, .src_type=nir_type_float32,
            .io_semantics={.location=FRAG_RESULT_DATA0,.num_slots=1});
         nir_shader_gather_info(fs.shader,nir_shader_get_entrypoint(fs.shader));
         struct agx_shader_part fragment = {0};
         fs_ok = agx_compile_apple9_fragment_inputs(fs.shader,
                         &out.info.apple9_varyings,&fragment,&fs_reason);
         free(fragment.binary);ralloc_free(fs.shader);
      }
      printf("%s,%u,%u,%u,%u,%u,%u,%u,%u,%s,%u,%u,%s,%s\n",
         stage == 0 ? "VS" : stage == 1 ? "TES" : "GS-rast", word,
         point, psiz, vp, layer, clips, culls, late, ok ? "yes" : "no",
         ok ? out.info.apple9_publication_count : 0, ok ? out.info.apple9_varyings.count : 0,
         ok ? (fs_ok ? "yes" : "no") : "not-run", reason ?: fs_reason ?: "none");
      ++total; failed += !ok || !fs_ok;
      if (ok && culls &&
          (!out.info.apple9_varyings.cull_distance_mask ||
           out.info.apple9_varyings.mask[VARYING_SLOT_CULL_PRIMITIVE] != 1 ||
           out.info.apple9_varyings.mask[VARYING_SLOT_CULL_PRIMITIVE + 1]))
         ++failed;
      free(out.binary);ralloc_free(count);ralloc_free(rast);ralloc_free(pre);ralloc_free(b.shader);
   }
   fprintf(stderr, "compiled=%u rejected=%u total=%u\n", total-failed, failed, total);
   glsl_type_singleton_decref();
   return failed ? 1 : 0;
}

// clang-format on
