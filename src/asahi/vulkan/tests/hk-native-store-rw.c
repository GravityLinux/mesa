/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include "hk_descriptor_set.h"
#include "hk_shader.h"
#include "compiler/nir/nir_builder.h"

/* Stores stay native only when no invocation can read its own native write:
 * every read before every store, no loop holding both, no calls. */

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "failed: %s\n", #c); return 1; } } while (0)

static void
image_op(nir_builder *b, nir_intrinsic_op op)
{
   nir_intrinsic_instr *intr = nir_intrinsic_instr_create(b->shader, op);
   const nir_intrinsic_info *info = &nir_intrinsic_infos[op];
   intr->num_components = 4;
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   nir_intrinsic_set_format(intr, PIPE_FORMAT_R32_FLOAT);
   for (unsigned j = 0; j < info->num_srcs; ++j) {
      unsigned components = info->src_components[j] ? info->src_components[j] : 4;
      intr->src[j] = nir_src_for_ssa(nir_imm_zero(b, components, j == 0 ? 64 : 32));
   }
   if (info->has_dest)
      nir_def_init(&intr->instr, &intr->def, 4, 32);
   nir_builder_instr_insert(b, &intr->instr);
}

#define LOAD nir_intrinsic_bindless_image_load
#define STORE nir_intrinsic_bindless_image_store

static nir_builder
shader(void)
{
   static const nir_shader_compiler_options options = {0};
   return nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "rw");
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   /* Read, then store: native. */
   nir_builder b = shader();
   image_op(&b, LOAD);
   image_op(&b, STORE);
   CHECK(hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* Store, then read: the read could observe the native write. */
   b = shader();
   image_op(&b, STORE);
   image_op(&b, LOAD);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* An atomic after a store reads it too. */
   b = shader();
   image_op(&b, STORE);
   image_op(&b, nir_intrinsic_bindless_image_atomic);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* Read before store inside one loop: the next iteration reads the write. */
   b = shader();
   nir_push_loop(&b);
   image_op(&b, LOAD);
   image_op(&b, STORE);
   nir_jump(&b, nir_jump_break);
   nir_pop_loop(&b, NULL);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* Reads in a loop that ends before the stores: native. */
   b = shader();
   nir_push_loop(&b);
   image_op(&b, LOAD);
   nir_jump(&b, nir_jump_break);
   nir_pop_loop(&b, NULL);
   nir_push_loop(&b);
   image_op(&b, STORE);
   nir_jump(&b, nir_jump_break);
   nir_pop_loop(&b, NULL);
   CHECK(hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* Store in the then-branch, read in the else-branch: rejected (block order). */
   b = shader();
   nir_push_if(&b, nir_imm_true(&b));
   image_op(&b, STORE);
   nir_push_else(&b, NULL);
   image_op(&b, LOAD);
   nir_pop_if(&b, NULL);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   /* A second function: block order is no longer program order. */
   b = shader();
   image_op(&b, LOAD);
   image_op(&b, STORE);
   nir_function *callee = nir_function_create(b.shader, "callee");
   nir_function_impl_create(callee);
   CHECK(!hk_apple9_native_store_shader(b.shader));
   ralloc_free(b.shader);

   puts("native storage rw rule: order, loops, branches and calls pass");
   glsl_type_singleton_decref();
   return 0;
}
