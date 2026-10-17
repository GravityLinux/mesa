/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "asahi/compiler/agx_compile_apple9.h"
#include "compiler/nir/nir_builder.h"
#include "poly/nir/poly_nir.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); return 1; } } while (0)

static unsigned
output_components(nir_shader *nir)
{
   unsigned components = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_output)
               components += util_bitcount(nir_intrinsic_write_mask(intr));
         }
      }
   }
   return components;
}

static int
geometry_outputs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "gs_output_boundary");
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.io_lowered = true;
   nir_def *zero = nir_imm_int(&b, 0);
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), zero,
      .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   for (unsigned i = 0; i < 124; ++i) {
      nir_store_output(&b, nir_imm_float(&b, i), zero, .write_mask = 1,
         .component = i % 4, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0 + i / 4,
                          .num_slots = 1});
   }
   nir_emit_vertex(&b, 0);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info = {0};
   CHECK(poly_nir_lower_gs(b.shader, &count, &rast, &pre, &info));
   CHECK(rast && rast->info.stage == MESA_SHADER_VERTEX);
   CHECK(output_components(rast) == 129);
   CHECK(rast->info.outputs_written & VARYING_BIT_PSIZ);
   puts("GS lowering: 128 API components become 129 with default point size");
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   ralloc_free(b.shader);
   return 0;
}

/* A full API interface plus the default point-size output inserted by TES
 * lowering needs 129 publications. Keep the native limit until there is an
 * overflow path; compute-based pre-rasterization alone cannot raise it.
 */
int main(void)
{
   glsl_type_singleton_init_or_ref();
   CHECK(geometry_outputs() == 0);
   const unsigned counts[] = {116, 122, 123, 124};
   for (unsigned t = 0; t < ARRAY_SIZE(counts); ++t) {
      unsigned users = counts[t];
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_EVAL, &agx_nir_options, "tes_output_boundary");
      b.shader->info.tess.point_mode = true;
      b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
      b.shader->info.io_lowered = true;
      b.shader->info.num_ubos = 1;
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *position = nir_load_ubo(&b, 4, 32, zero, zero,
                                      .align_mul = 16, .range = 16);
      nir_store_output(&b, position, zero, .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      for (unsigned i = 0; i < users; ++i) {
         nir_def *value = nir_load_ubo(&b, 1, 32, zero,
            nir_imm_int(&b, 16 + i * 4), .align_mul = 4, .range = 4);
         nir_store_output(&b, value, zero, .write_mask = 1,
            .component = i % 4, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0 + i / 4,
                             .num_slots = 1});
      }
      nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
      CHECK(poly_nir_lower_tes(b.shader, true));
      CHECK(b.shader->info.stage == MESA_SHADER_VERTEX);
      CHECK(b.shader->info.outputs_written & VARYING_BIT_PSIZ);
      struct agx_apple9_vertex_layout layout = {.rasterize_points = true};
      struct agx_shader_part out = {0};
      const char *reason = NULL;
      bool ok = agx_compile_apple9_vertex_inputs(b.shader, &layout, &out, &reason);
      printf("TES API components=%u native components=%u compile=%s reason=%s\n",
             4 + users, 4 + users + 1, ok ? "yes" : "no", reason ?: "none");
      CHECK(ok == (4 + users + 1 <= AGX_APPLE9_PUBLICATION_COUNT));
      if (ok) CHECK(out.info.apple9_publication_count <= 128);
      free(out.binary);
      ralloc_free(b.shader);
   }
   glsl_type_singleton_decref();
   return 0;
}
