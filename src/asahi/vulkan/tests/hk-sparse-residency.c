/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/* Sparse sample lowering runs after descriptor lowering. Its new size query
 * must survive the common query pass, which decodes legacy descriptors. The
 * Apple9 backend then answers it from the sampled image's metadata.
 */
#include <stdio.h>

#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "compiler/nir/nir_builder.h"
#include "hk_shader.h"

static const nir_shader_compiler_options options = {0};

static bool
check(nir_texop op, bool array)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "sparse");
   bool fetch = op == nir_texop_txf;
   nir_def *handle = nir_imm_int(&b, 0x1000);
   nir_def *coord = fetch ? nir_imm_ivec3(&b, 64, 64, 0)
                          : nir_imm_vec3(&b, 0.125, 0.125, 0.0);
   coord = nir_trim_vector(&b, coord, array ? 3 : 2);
   nir_def *grad = nir_imm_vec2(&b, 0.01, 0.01);
   nir_def *value = nir_build_tex(
      &b, op, .coord = coord, .texture_handle = handle,
      .sampler_handle = nir_imm_int(&b, 0x2000),
      .lod = op == nir_texop_txl ? nir_imm_float(&b, 0.0)
             : fetch ? nir_imm_int(&b, 0) : NULL,
      .dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_float32,
      .is_array = array, .is_sparse = true);
   nir_tex_instr *sample = nir_def_as_tex(value);
   if (op == nir_texop_txd) {
      nir_tex_instr_add_src(sample, nir_tex_src_ddx, grad);
      nir_tex_instr_add_src(sample, nir_tex_src_ddy, grad);
   }
   nir_store_global(&b, value, nir_imm_int64(&b, 0), .align_mul = 4,
                    .write_mask = 0x1f);

   hk_apple9_lower_sparse_residency(b.shader);
   nir_validate_shader(b.shader, "after sparse residency lowering");
   agx_nir_lower_texture_queries(b.shader);

   unsigned queries = 0;
   bool ok = !sample->is_sparse && sample->def.num_components == 4;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->op != nir_texop_txs)
               continue;
            ++queries;
            int h = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
            int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
            ok &= (tex->backend_flags & AGX_TEXTURE_FLAG_METADATA_QUERY) != 0;
            ok &= h >= 0 && tex->src[h].src.ssa == handle;
            ok &= lod >= 0 && nir_src_is_const(tex->src[lod].src) &&
                  nir_src_as_uint(tex->src[lod].src) == 0;
            ok &= tex->is_array == array;
         }
      }
   }
   ok &= queries == (fetch ? 0 : 1);
   if (!ok)
      fprintf(stderr, "sparse op %u array %u lost its metadata size query\n",
              op, array);
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   const nir_texop ops[] = {nir_texop_txl, nir_texop_tex, nir_texop_txd,
                            nir_texop_tg4, nir_texop_txf};
   bool ok = true;
   for (unsigned i = 0; i < ARRAY_SIZE(ops); ++i) {
      ok &= check(ops[i], false);
      ok &= check(ops[i], true);
   }
   return ok ? 0 : 1;
}
