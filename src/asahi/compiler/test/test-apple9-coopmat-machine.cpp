/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "agx_compile_apple9.h"
#include "asahi/vulkan/hk_apple9_coopmat.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"

class Apple9CoopmatMachine : public ::testing::Test {
protected:
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "coopmat_machine");

   void SetUp() override
   {
      glsl_type_singleton_init_or_ref();
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.cs.has_cooperative_matrix = true;
   }
   void TearDown() override
   {
      ralloc_free(b.shader);
      glsl_type_singleton_decref();
   }

   nir_deref_instr *matrix(glsl_base_type type, glsl_cmat_use use)
   {
      glsl_cmat_description d = {};
      d.rows = d.cols = 8;
      d.scope = SCOPE_SUBGROUP;
      d.element_type = type;
      d.use = use;
      return nir_build_deref_var(&b, nir_local_variable_create(
         b.impl, glsl_cmat_type(&d), "matrix"));
   }

   nir_deref_instr *buffer(unsigned binding, bool half)
   {
      return nir_build_deref_cast(
         &b, nir_vec2(&b, nir_imm_int(&b, binding), nir_imm_int(&b, 0)),
         nir_var_mem_ssbo, half ? glsl_float16_t_type() : glsl_float_type(),
         half ? 2 : 4);
   }

   void roundtrip()
   {
      blob data;
      blob_init(&data);
      nir_serialize(&data, b.shader, false);
      ASSERT_FALSE(data.out_of_memory);
      blob_reader reader;
      blob_reader_init(&reader, data.data, data.size);
      nir_shader *copy = nir_deserialize(nullptr, &agx_nir_options, &reader);
      ASSERT_NE(copy, nullptr);
      ASSERT_FALSE(reader.overrun);
      EXPECT_EQ(reader.current, reader.end);
      ralloc_free(b.shader);
      b.shader = copy;
      b.impl = nir_shader_get_entrypoint(copy);
      blob_finish(&data);
   }

   void check_lowered()
   {
      EXPECT_FALSE(b.shader->info.cs.has_cooperative_matrix);
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_function_temp_variable(var, impl)
            EXPECT_FALSE(glsl_type_is_cmat(glsl_without_array(var->type)));
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               EXPECT_NE(instr->type, nir_instr_type_cmat_call);
               if (instr->type == nir_instr_type_deref) {
                  EXPECT_FALSE(glsl_type_is_cmat(
                     glsl_without_array(nir_instr_as_deref(instr)->type)));
               }
               if (instr->type == nir_instr_type_intrinsic) {
                  EXPECT_NE(strncmp(nir_intrinsic_infos[
                     nir_instr_as_intrinsic(instr)->intrinsic].name,
                     "cmat_", 5), 0);
               }
            }
         }
      }
   }

   void compile(bool half)
   {
      auto type = half ? GLSL_TYPE_FLOAT16 : GLSL_TYPE_FLOAT;
      auto *a = matrix(type, GLSL_CMAT_USE_A);
      auto *bb = matrix(type, GLSL_CMAT_USE_B);
      auto *c = matrix(GLSL_TYPE_FLOAT, GLSL_CMAT_USE_ACCUMULATOR);
      auto *d = matrix(GLSL_TYPE_FLOAT, GLSL_CMAT_USE_ACCUMULATOR);
      auto *stride = nir_imm_int(&b, 8);
      nir_cmat_load(&b, &a->def, &buffer(0, half)->def, stride,
                    .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);
      nir_cmat_load(&b, &bb->def, &buffer(1, half)->def, stride,
                    .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);
      nir_cmat_load(&b, &c->def, &buffer(2, false)->def, stride,
                    .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);
      nir_cmat_muladd(&b, &d->def, &a->def, &bb->def, &c->def);
      nir_cmat_store(&b, &buffer(3, false)->def, &d->def, stride,
                     .matrix_layout = GLSL_MATRIX_LAYOUT_ROW_MAJOR);
      nir_validate_shader(b.shader, "machine before matrix lowering");
      const char *reason = nullptr;
      ASSERT_TRUE(agx_nir_lower_apple9_cooperative_matrix(b.shader, 32, &reason))
         << (reason ? reason : "");
      EXPECT_EQ(reason, nullptr);
      check_lowered();
      roundtrip();
      EXPECT_EQ(hk_apple9_coopmat_rejection(b.shader), nullptr);
      check_lowered();

      /* The ordinary agx_preprocess_nir variable passes, including the
       * formerly unsafe scratch sizing, followed by SSBO explicit IO.
       */
      nir_lower_vars_to_ssa(b.shader);
      nir_lower_vars_to_scratch(b.shader, 256,
                                glsl_get_natural_size_align_bytes,
                                glsl_get_word_size_align_bytes);
      nir_lower_indirect_derefs_to_if_else_trees(
         b.shader, nir_var_function_temp, ~0u);
      nir_split_var_copies(b.shader);
      nir_lower_global_vars_to_local(b.shader);
      nir_lower_var_copies(b.shader);
      nir_lower_array_deref_of_vec(
         b.shader, nir_var_function_temp, nullptr,
         nir_lower_array_deref_of_vec_options(
            nir_lower_indirect_array_deref_of_vec_load |
            nir_lower_indirect_array_deref_of_vec_store));
      nir_lower_vars_to_ssa(b.shader);
      nir_lower_explicit_io(b.shader, nir_var_mem_ssbo,
                            nir_address_format_32bit_index_offset);
      nir_validate_shader(b.shader, "machine ordinary IO lowering");
      agx_shader_part out = {};
      agx_apple9_compute_profile profile = {};
      bool ok = agx_compile_apple9_tiny(b.shader, &out, &profile, &reason);
      EXPECT_TRUE(ok) << (reason ? reason : "");
      EXPECT_EQ(reason, nullptr);
      EXPECT_NE(out.binary, nullptr);
      EXPECT_GT(out.info.binary_size, 0u);
      free(out.binary);
   }
};

TEST_F(Apple9CoopmatMachine, FP32NativeCode) { compile(false); }
TEST_F(Apple9CoopmatMachine, FP16NativeCode) { compile(true); }

TEST_F(Apple9CoopmatMachine, CachedIntegerRejectedBeforeGenericLowering)
{
   matrix(GLSL_TYPE_INT, GLSL_CMAT_USE_A);
   char *before = nir_shader_as_str(b.shader, nullptr);
   const char *reason = hk_apple9_coopmat_rejection(b.shader);
   ASSERT_NE(reason, nullptr);
   /* Check deterministic failure and untouched NIR, not diagnostic prose. */
   std::string first_reason(reason);
   char *after = nir_shader_as_str(b.shader, nullptr);
   EXPECT_STREQ(before, after);
   roundtrip();
   EXPECT_TRUE(b.shader->info.cs.has_cooperative_matrix);
   EXPECT_EQ(first_reason, hk_apple9_coopmat_rejection(b.shader));
   ralloc_free(before);
   ralloc_free(after);
}

/* Independent scalar oracle for the actual tile_matmul_f32 probe. Unlike
 * cooperative fragments this is ordinary row/column indexing. The entry
 * overwrites C; its initial nonzero sentinel is deliberately not accumulated.
 */
TEST(Apple9CoopmatOracle, TilematmulExpected)
{
   float a[8][16], bb[16][8], expected[8][8];
   for (unsigned r = 0; r < 8; ++r)
      for (unsigned k = 0; k < 16; ++k)
         a[r][k] = float(int((r * 17 + k * 3) % 23) - 11) * 0.125f;
   for (unsigned k = 0; k < 16; ++k)
      for (unsigned c = 0; c < 8; ++c)
         bb[k][c] = float(int((k * 5 + c * 13) % 29) - 14) * 0.0625f;
   for (unsigned r = 0; r < 8; ++r)
      for (unsigned c = 0; c < 8; ++c) {
         float value = 0;
         for (unsigned k = 0; k < 16; ++k)
            value = std::fma(a[r][k], bb[k][c], value);
         expected[r][c] = value;
         ASSERT_TRUE(std::isfinite(value));
      }
   EXPECT_NE(expected[0][1], expected[1][0]);
   if (const char *path = getenv("N5_WRITE_EXPECTED")) {
      FILE *file = fopen(path, "wb");
      ASSERT_NE(file, nullptr);
      EXPECT_EQ(fwrite(expected, sizeof(expected), 1, file), 1u);
      EXPECT_EQ(fclose(file), 0);
   }
}
