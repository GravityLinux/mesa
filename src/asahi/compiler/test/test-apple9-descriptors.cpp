/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_compile_apple9.h"
#include "compiler/nir/nir_builder.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/vulkan/hk_apple9.h"

/* These are post-lowering NIR fixtures, not an independent descriptor pass.
 * Nonzero root member offsets are representative literals: the Vulkan pass
 * suite exercises the real hk_root_descriptor_table offsets in driver code.
 */
class Apple9RootCompiler : public ::testing::Test {
protected:
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "post_lowering_root");
   nir_def *root = nullptr;
   nir_def *index = nullptr;

   void SetUp() override
   {
      b.shader->info.workgroup_size[0] = 32;
      root = nir_load_ubo(
         &b, 1, 64, nir_imm_int(&b, HK_APPLE9_ROOT_BINDING),
         nir_imm_int(&b, 0), .align_mul = 8, .range = 8);
      index = nir_load_subgroup_invocation(&b);
   }

   void TearDown() override { ralloc_free(b.shader); }

   nir_def *set(unsigned set)
   {
      return nir_load_global_constant(
         &b, 1, 64, nir_iadd_imm(&b, root, 8 + 8 * set), .align_mul = 8);
   }

   nir_def *descriptor(nir_def *base, unsigned binding_offset,
                       unsigned array_size, bool robust)
   {
      nir_def *element = array_size == 1 ? nir_imm_int(&b, 0)
         : nir_iadd(&b, index, index); /* resource_reindex */
      if (robust)
         element = nir_umin_imm(&b, element, array_size - 1);
      nir_def *offset = nir_iadd_imm(
         &b, nir_amul_imm(&b, element, 16), binding_offset);
      return nir_load_global_constant(
         &b, 4, 32, nir_iadd(&b, base, nir_u2u64(&b, offset)),
         .align_mul = 16);
   }

   nir_def *load(nir_def *desc, bool robust)
   {
      nir_def *base = nir_pack_64_2x32(&b, nir_trim_vector(&b, desc, 2));
      nir_def *offset = nir_amul_imm(&b, index, 4);
      if (robust) {
         nir_def *valid = nir_ult(
            &b, index, nir_udiv_imm(&b, nir_channel(&b, desc, 2), 4));
         base = nir_bcsel(&b, valid, base,
                         nir_imm_int64(&b, AGX_ZERO_PAGE_ADDRESS));
         offset = nir_amul_imm(
            &b, nir_bcsel(&b, valid, index, nir_imm_int(&b, 0)), 4);
      }
      return nir_load_global(
         &b, 1, 32, nir_iadd(&b, base, nir_u2u64(&b, offset)), .align_mul = 4);
   }

   void output(nir_def *value)
   {
      nir_def *desc = descriptor(set(0), 4096, 1, false);
      nir_def *base = nir_pack_64_2x32(&b, nir_trim_vector(&b, desc, 2));
      nir_store_global(&b, value, nir_iadd(
         &b, base, nir_u2u64(&b, nir_amul_imm(&b, index, 4))),
         .write_mask = 1, .align_mul = 4);
   }

   void compile()
   {
      nir_validate_shader(b.shader, "post-lowering root fixture");
      agx_shader_part compiled = {};
      agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
      if (!ok)
         nir_print_shader(b.shader, stderr);
      ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
      EXPECT_EQ(reason, nullptr);
      EXPECT_NE(compiled.binary, nullptr);
      EXPECT_GT(compiled.info.binary_size, 0u);
      EXPECT_LE(profile.resource_binding_count,
                AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES);
      ASSERT_EQ(profile.resource_binding_count, 1);
      EXPECT_EQ(profile.resource_binding[0], HK_APPLE9_ROOT_BINDING);
      EXPECT_EQ(profile.resource_kind[0], AGX_APPLE9_COMPUTE_RESOURCE_UBO);
      free(compiled.binary);
   }

   void mixed(bool robust)
   {
      nir_def *sum = load(descriptor(set(0), 0, 4, robust), robust);
      sum = nir_iadd(&b, sum,
                    load(descriptor(set(1), 64, 300, robust), robust));
      sum = nir_iadd(&b, sum,
                    load(descriptor(root, 1024, 4, robust), robust));
      /* Inline uniform block pointer, with its compile-time size. */
      nir_def *inline_base = nir_iadd_imm(&b, set(1), 2048);
      nir_def *inline_desc = nir_vec4(
         &b, nir_unpack_64_2x32_split_x(&b, inline_base),
         nir_unpack_64_2x32_split_y(&b, inline_base),
         nir_imm_int(&b, 256), nir_imm_int(&b, 0));
      sum = nir_iadd(&b, sum, load(inline_desc, robust));
      nir_def *push = nir_load_global_constant(
         &b, 1, 32, nir_iadd(&b, root, nir_u2u64(
            &b, nir_iadd_imm(&b, nir_amul_imm(&b, index, 4), 512))),
         .align_mul = 4);
      output(nir_iadd(&b, sum, push));
   }
};

TEST_F(Apple9RootCompiler, TwoSetsUnboundedBufferShapes)
{
   mixed(false);
   compile();
}

TEST_F(Apple9RootCompiler, TwoSetsRobustBufferShapes)
{
   mixed(true);
   compile();
}

TEST_F(Apple9RootCompiler, RuntimeArray300DynamicReindex)
{
   output(load(descriptor(set(1), 64, 300, false), false));
   compile();
}

TEST_F(Apple9RootCompiler, FortyBindingsUseOneVisibleRoot)
{
   nir_def *base = set(0);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned binding = 0; binding < 40; ++binding)
      sum = nir_iadd(&b, sum, load(
         descriptor(base, binding * 16, 1, false), false));
   output(sum);
   compile();
}

TEST(Apple9RootGraphicsCompiler, RootInGraphicsResourceMap)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "root_graphics_resource_map");
   nir_def *root = nir_load_ubo(
      &b, 1, 64, nir_imm_int(&b, HK_APPLE9_ROOT_BINDING),
      nir_imm_int(&b, 0), .align_mul = 8, .range = 8);
   nir_def *set = nir_load_global_constant(
      &b, 1, 64, nir_iadd_imm(&b, root, 8), .align_mul = 8);
   nir_def *desc = nir_load_global_constant(&b, 4, 32, set, .align_mul = 16);
   nir_def *address = nir_pack_64_2x32(&b, nir_trim_vector(&b, desc, 2));
   nir_def *position = nir_load_global_constant(
      &b, 4, 32, address, .align_mul = 16);
   nir_store_output(
      &b, position, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32, .io_semantics = {.location = VARYING_SLOT_POS});
   b.shader->info.io_lowered = true;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_vertex(b.shader, &compiled, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_EQ(reason, nullptr);
   EXPECT_NE(compiled.binary, nullptr);
   EXPECT_GT(compiled.info.binary_size, 0u);
   ASSERT_EQ(compiled.info.apple9_resource_count, 1);
   EXPECT_EQ(compiled.info.apple9_resource_binding[0], HK_APPLE9_ROOT_BINDING);
   EXPECT_EQ(compiled.info.apple9_resource_ssbo_mask, 0u);
   free(compiled.binary);
   ralloc_free(b.shader);
}
