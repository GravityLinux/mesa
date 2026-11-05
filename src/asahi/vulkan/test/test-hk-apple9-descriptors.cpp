/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include <cstdlib>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "hk_apple9.h"
#include "hk_apple9_root.h"
#include "asahi/genxml/agx_pack.h"
#include "hk_descriptor_set_layout.h"
#include "nir_builder.h"
#include "vk_pipeline.h"
#include "vk_shader.h"

TEST(Apple9Descriptors, LinearTextureDepthDoesNotAliasStride)
{
   for (bool linear : {false, true}) {
      SCOPED_TRACE(linear);
      struct agx_texture_packed descriptor;
      struct AGX_TEXTURE cfg = {AGX_TEXTURE_header};
      cfg.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
      cfg.layout = linear ? AGX_LAYOUT_LINEAR : AGX_LAYOUT_TWIDDLED;
      cfg.width = 1440;
      cfg.height = 900;
      cfg.address = 0x4000;
      cfg.extended = linear;
      if (linear) {
         cfg.stride = 5744;
         cfg.depth_linear = 1;
      } else {
         cfg.depth = 4;
      }
      AGX_TEXTURE_pack(reinterpret_cast<uint32_t *>(&descriptor), &cfg);
      struct AGX_TEXTURE decoded;
      ASSERT_TRUE(hk_apple9_decode_texture(&descriptor, &decoded));
      EXPECT_EQ(decoded.depth, linear ? 1u : 4u);
      if (linear) {
         EXPECT_EQ(decoded.stride, 5744u);
      }
   }
}

static const char legacy_golden[] = R"(shader: MESA_SHADER_COMPUTE
name: legacy_descriptor_golden
internal: true
workgroup_size: 1, 1, 1
max_subgroup_size: 128
min_subgroup_size: 1
decl_function main () (entrypoint)

impl main {
    block b0:   // preds:
    32     %4 = load_const (0x00000000)
    32    %11 = load_const (0x00000000 = 0.000000)
    32     %5 = @load_ssbo (%11 (0x0), %4 (0x0)) (access=none, align_mul=4, align_offset=0, offset_shift=0)
    32     %7 = load_const (0x00000004)
    32    %12 = load_const (0x00000000 = 0.000000)
                @store_ssbo (%5, %12 (0x0), %7 (0x4)) (wrmask=x, access=none, align_mul=4, align_offset=0, offset_shift=0)
                // succs: b1
    block b1:
}

)";

TEST(Apple9Descriptors, LegacyConstantSelection)
{
   auto *layout = static_cast<hk_descriptor_set_layout *>(
      calloc(1, sizeof(hk_descriptor_set_layout) +
                   sizeof(hk_descriptor_set_binding_layout)));
   layout->binding_count = 1;
   layout->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layout->binding[0].array_size = 4;
   layout->binding[0].stride = 16;
   vk_descriptor_set_layout *sets[] = {&layout->vk};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "legacy_descriptor_golden");
   nir_def *idx = nir_vulkan_resource_index(
      &b, 2, 32, nir_imm_int(&b, 2), .desc_set = 0, .binding = 0,
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      &b, 2, 32, idx, .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *value = nir_load_ssbo(
      &b, 1, 32, nir_channel(&b, desc, 0), nir_imm_int(&b, 0),
      .align_mul = 4);
   nir_store_ssbo(&b, value, nir_channel(&b, desc, 0), nir_imm_int(&b, 4),
                  .write_mask = 1, .align_mul = 4);
   hk_apple9_shader_mapping mapping = {};
   const char *reason = nullptr;
   ASSERT_TRUE(hk_nir_lower_apple9_descriptors(
      b.shader, 1, sets, &mapping, &reason)) << (reason ? reason : "");
   nir_opt_constant_folding(b.shader);
   nir_opt_dce(b.shader);
   EXPECT_EQ(mapping.ssbo_count, 1);
   EXPECT_EQ(mapping.ssbo[0].element, 2u);
   char *text = nir_shader_as_str(b.shader, nullptr);
   EXPECT_STREQ(text, legacy_golden);
   if (getenv("N4_CAPTURE_LEGACY"))
      printf("N4_GOLDEN_BEGIN\n%sN4_GOLDEN_END\n", text);
   ralloc_free(text);
   ralloc_free(b.shader);
   free(layout);
}

/* Walk the actual address graph, including loads of set and buffer pointers. */
struct dependency {
   nir_def *target;
   std::unordered_set<nir_instr *> visited;
};

static bool
independent(nir_src *src, void *data)
{
   auto *dep = static_cast<dependency *>(data);
   if (src->ssa == dep->target)
      return false;
   nir_instr *instr = nir_def_instr(src->ssa);
   return !dep->visited.insert(instr).second ||
          nir_foreach_src(instr, independent, data);
}

static bool
depends_on(nir_def *value, nir_def *target)
{
   dependency dep = {target, {}};
   nir_src src = nir_src_for_ssa(value);
   return !independent(&src, &dep);
}

class RootDescriptors : public ::testing::Test {
protected:
   hk_descriptor_set_layout *layouts[2] = {};
   vk_descriptor_set_layout *sets[2] = {};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "root_descriptors");
   nir_def *index = nullptr;
   vk_pipeline_robustness_state rs = {};

   void SetUp() override
   {
      for (unsigned s = 0; s < 2; ++s) {
         layouts[s] = static_cast<hk_descriptor_set_layout *>(calloc(
            1, sizeof(hk_descriptor_set_layout) +
                  2 * sizeof(hk_descriptor_set_binding_layout)));
         layouts[s]->binding_count = 2;
         sets[s] = &layouts[s]->vk;
      }
      layouts[0]->binding[0] = {
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, 4, 0, 16, 0, nullptr};
      layouts[0]->binding[1] = {
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT, 300, 64, 16, 0,
         nullptr};
      layouts[1]->binding[0] = {
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 0, 4, 0, 16, 0, nullptr};
      layouts[1]->vk.dynamic_descriptor_count = 4;
      layouts[1]->binding[1] = {
         VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK, 0, 256, 64, 1, 0, nullptr};
      rs = vk_robustness_disabled;
      b.shader->info.workgroup_size[0] = 32;
      index = nir_load_subgroup_invocation(&b);
   }

   void TearDown() override
   {
      ralloc_free(b.shader);
      for (auto *layout : layouts)
         free(layout);
   }

   nir_deref_instr *buffer(unsigned set, unsigned binding, bool reindex = true)
   {
      bool ssbo = layouts[set]->binding[binding].type ==
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                  layouts[set]->binding[binding].type ==
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      nir_descriptor_type type = ssbo ? nir_descriptor_type_storage_buffer
                                     : nir_descriptor_type_uniform_buffer;
      nir_def *resource = nir_vulkan_resource_index(
         &b, 4, 32, index, .desc_set = set, .binding = binding,
         .desc_type = type);
      if (reindex)
         resource = nir_vulkan_resource_reindex(
            &b, 32, resource, index, .desc_type = type);
      nir_def *desc = nir_load_vulkan_descriptor(
         &b, 4, 32, resource, .desc_type = type);
      auto mode = ssbo ? nir_var_mem_ssbo : nir_var_mem_ubo;
      nir_deref_instr *ptr = nir_build_deref_cast(
         &b, desc, mode, glsl_uint_type(), 4);
      return nir_build_deref_ptr_as_array(&b, ptr, index);
   }

   void shader()
   {
      nir_def *sum = nir_load_deref(&b, buffer(0, 0));
      auto *output = buffer(0, 1);
      sum = nir_iadd(&b, sum, nir_load_deref(&b, output));
      sum = nir_iadd(&b, sum, nir_load_deref(&b, buffer(1, 0)));
      sum = nir_iadd(&b, sum, nir_load_deref(&b, buffer(1, 1, false)));
      sum = nir_iadd(&b, sum, nir_load_push_constant(
         &b, 1, 32, nir_imul_imm(&b, index, 4), .base = 0, .range = 256));
      nir_store_deref(&b, output, sum, 1);
   }

   void lower(bool robust)
   {
      if (robust) {
         rs.storage_buffers =
            VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT;
         rs.uniform_buffers =
            VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT;
      }
      nir_validate_shader(b.shader, "before buffer descriptors");
      hk_nir_lower_buffer_descriptors(b.shader, &rs, 2, sets);
      hk_apple9_shader_mapping mapping = {};
      const char *reason = nullptr;
      ASSERT_TRUE(hk_nir_lower_apple9_textures(
         b.shader, 2, sets, &mapping, &reason)) << (reason ? reason : "");
      EXPECT_EQ(mapping.ubo_count, 0);
      EXPECT_EQ(mapping.ssbo_count, 0);
      EXPECT_FALSE(mapping.descriptor_snapshot);
      nir_lower_explicit_io(b.shader, nir_var_mem_global,
                           nir_address_format_64bit_global);
      auto format = robust ? nir_address_format_64bit_bounded_global
                           : nir_address_format_64bit_global_32bit_offset;
      nir_lower_explicit_io(b.shader, nir_var_mem_ssbo, format);
      nir_lower_explicit_io(b.shader, nir_var_mem_ubo, format);
      nir_opt_constant_folding(b.shader);
      nir_opt_copy_prop(b.shader);
      hk_nir_lower_global_constant_offsets(b.shader, false);
      hk_nir_lower_apple9_root(b.shader);
      nir_opt_constant_folding(b.shader);
      nir_opt_copy_prop(b.shader);
      nir_opt_dce(b.shader);
      nir_validate_shader(b.shader, "after native root descriptors");
   }

   unsigned alu_count(nir_op op)
   {
      unsigned count = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               ++count;
         }
      }
      return count;
   }

   void check_root()
   {
      std::vector<nir_def *> roots;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_vulkan_resource_index);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_vulkan_resource_reindex);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_load_vulkan_descriptor);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_load_root_agx);
            EXPECT_NE(intr->intrinsic, nir_intrinsic_load_descriptor_set_agx);
            if (intr->intrinsic == nir_intrinsic_load_ubo) {
               EXPECT_EQ(nir_src_as_uint(intr->src[0]), HK_APPLE9_ROOT_BINDING);
               EXPECT_EQ(nir_src_as_uint(intr->src[1]), 0u);
               EXPECT_EQ(intr->def.bit_size, 64);
               roots.push_back(&intr->def);
            }
         }
      }
      ASSERT_FALSE(roots.empty());
      unsigned globals = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            bool store = intr->intrinsic == nir_intrinsic_store_global;
            if (!store && intr->intrinsic != nir_intrinsic_load_global &&
                intr->intrinsic != nir_intrinsic_load_global_constant)
               continue;
            nir_def *addr = intr->src[store ? 1 : 0].ssa;
            EXPECT_EQ(addr->bit_size, 64);
            bool rooted = false;
            for (auto *root : roots)
               rooted |= depends_on(addr, root);
            EXPECT_TRUE(rooted);
            ++globals;
         }
      }
      EXPECT_GE(globals, 8u);
   }
};

TEST_F(RootDescriptors, TwoSetsUnboundedGlobals)
{
   shader();
   lower(false);
   check_root();
   EXPECT_EQ(alu_count(nir_op_umin), 0u);
   EXPECT_EQ(alu_count(nir_op_bcsel), 0u);
}

TEST_F(RootDescriptors, LegacyRootLoadsKeepRangeAndAlignment)
{
   nir_def *resource = nir_vulkan_resource_index(
      &b, 2, 32, nir_imm_int(&b, 0), .desc_set = 0, .binding = 1,
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *descriptor = nir_load_vulkan_descriptor(
      &b, 2, 32, resource, .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *binding = nir_channel(&b, descriptor, 0);
   nir_def *push = nir_load_push_constant(
      &b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 16,
      .align_mul = 256, .align_offset = 0);
   nir_def *group = nir_channel(&b, nir_load_base_workgroup_id(&b, 32), 0);
   nir_def *size = nir_get_ssbo_size(&b, 32, binding);
   nir_store_ssbo(&b, nir_iadd(&b, push, nir_iadd(&b, group, size)),
                  binding, nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   hk_apple9_shader_mapping mapping = {};
   const char *reason = nullptr;
   ASSERT_TRUE(hk_nir_lower_apple9_descriptors(
      b.shader, 2, sets, &mapping, &reason)) << (reason ? reason : "");
   nir_opt_constant_folding(b.shader);
   nir_opt_dce(b.shader);
   std::set<unsigned> ranges;
   unsigned loads = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_ubo)
            continue;
         ASSERT_TRUE(nir_src_is_const(intr->src[1]));
         unsigned address = nir_src_as_uint(intr->src[1]);
         ASSERT_GT(nir_intrinsic_range(intr), 0u);
         EXPECT_EQ(nir_intrinsic_range_base(intr), address);
         EXPECT_EQ(nir_intrinsic_align_offset(intr),
                   address & (nir_intrinsic_align_mul(intr) - 1));
         ranges.insert(nir_intrinsic_range(intr));
         ++loads;
      }
   }
   EXPECT_EQ(loads, 3u);
   EXPECT_EQ(ranges, (std::set<unsigned>{4, 12, 16}));
   nir_validate_shader(b.shader, "legacy root metadata");
}

TEST_F(RootDescriptors, RobustBoundsAndArrayClamps)
{
   shader();
   lower(true);
   check_root();
   EXPECT_GE(alu_count(nir_op_umin), 4u);
   EXPECT_GE(alu_count(nir_op_bcsel), 4u);
   unsigned fixed_clamps = 0, runtime_clamps = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         auto *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_umin || !nir_src_is_const(alu->src[1].src))
            continue;
         unsigned bound = nir_src_as_uint(alu->src[1].src);
         if (bound == 3)
            ++fixed_clamps;
         if (bound == 299) {
            EXPECT_TRUE(depends_on(alu->src[0].src.ssa, index));
            ++runtime_clamps;
         }
      }
   }
   EXPECT_EQ(fixed_clamps, 2u); /* UBO[4] and dynamic SSBO[4] */
   EXPECT_EQ(runtime_clamps, 1u); /* runtime SSBO, not a 64-slot table */
}

TEST_F(RootDescriptors, RuntimeArray300RemainsDynamic)
{
   shader();
   lower(false);
   check_root();
   unsigned dynamic_addresses = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_global_constant &&
             depends_on(intr->src[0].ssa, index))
            ++dynamic_addresses;
      }
   }
   EXPECT_GE(dynamic_addresses, 2u);
}

TEST_F(RootDescriptors, ActualDriverLoweringCompiles)
{
   shader();
   lower(true);
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   if (!ok)
      nir_print_shader(b.shader, stderr);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_EQ(reason, nullptr);
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_NE(compiled.binary, nullptr);
   ASSERT_EQ(profile.resource_binding_count, 1);
   EXPECT_EQ(profile.resource_binding[0], HK_APPLE9_ROOT_BINDING);
   EXPECT_EQ(profile.resource_kind[0], AGX_APPLE9_COMPUTE_RESOURCE_UBO);
   free(compiled.binary);
}

TEST_F(RootDescriptors, R32ImageStoreCompilesWithoutArtificialBufferOutput)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   layouts[0]->binding[0].stride = 48;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "storage_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_image_deref_store(
      &b, deref, nir_pad_vec4(&b, nir_vec2(&b, index, nir_imm_int(&b, 0))),
      nir_undef(&b, 1, 32),
      nir_vec4(&b, nir_iadd_imm(&b, index, 17), nir_imm_int(&b, 0),
                nir_imm_int(&b, 0), nir_imm_int(&b, 0)),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R32_UINT, .access = ACCESS_NON_READABLE,
      .src_type = nir_type_uint32);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_TRUE(profile.writes_global);
   free(compiled.binary);
}

TEST_F(RootDescriptors, R32ImageLoadCompilesSuccessfully)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   layouts[0]->binding[0].stride = 48;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "storage_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_def *loaded = nir_image_deref_load(
      &b, 4, 32, deref,
      nir_pad_vec4(&b, nir_vec2(&b, index, nir_imm_int(&b, 0))),
      nir_undef(&b, 1, 32), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R32_UINT, .access = ACCESS_NON_WRITEABLE,
      .dest_type = nir_type_uint32);
   /* Use the loaded value in an SSBO store so the compiler doesn't DCE it */
   layouts[0]->binding[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layouts[0]->binding[1].offset = 48;
   layouts[0]->binding[1].stride = 16;
   nir_store_ssbo(&b, nir_channel(&b, loaded, 0), nir_imm_int(&b, 1),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
}

TEST_F(RootDescriptors, Image1DArrayChecksLayerForLoadsAndStores)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   layouts[0]->binding[0].stride = 48;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_1D, true, GLSL_TYPE_UINT), "array_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_def *coord = nir_pad_vec4(&b, nir_vec2(&b, nir_imm_int(&b, 0), index));
   nir_image_deref_store(
      &b, deref, coord, nir_undef(&b, 1, 32), nir_imm_ivec4(&b, 17, 0, 0, 0),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_1D,
      .image_array = true, .format = PIPE_FORMAT_R32_UINT,
      .access = ACCESS_NON_READABLE, .src_type = nir_type_uint32);
   nir_def *loaded = nir_image_deref_load(
      &b, 4, 32, deref, coord, nir_undef(&b, 1, 32), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_1D, .image_array = true,
      .format = PIPE_FORMAT_R32_UINT, .access = ACCESS_NON_WRITEABLE,
      .dest_type = nir_type_uint32);
   layouts[0]->binding[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layouts[0]->binding[1].offset = 48;
   layouts[0]->binding[1].stride = 16;
   nir_store_ssbo(&b, nir_channel(&b, loaded, 0), nir_imm_int(&b, 1),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   unsigned layer_guards = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         auto *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_ult || alu->def.num_components != 2)
            continue;
         nir_scalar layer = nir_scalar_resolved(
            alu->src[0].src.ssa, alu->src[0].swizzle[1]);
         layer_guards += layer.def == index && layer.comp == 0;
      }
   }
   EXPECT_GE(layer_guards, 2u);
}

TEST_F(RootDescriptors, R32TexelBufferLoadCompiles)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
   layouts[0]->binding[0].stride = 48;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_BUF, false, GLSL_TYPE_UINT), "buffer_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *loaded = nir_image_deref_load(
      &b, 4, 32, &nir_build_deref_var(&b, var)->def, nir_pad_vec4(&b, index),
      nir_undef(&b, 1, 32), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_BUF, .format = PIPE_FORMAT_R32_UINT,
      .access = ACCESS_NON_WRITEABLE, .dest_type = nir_type_uint32);
   layouts[0]->binding[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layouts[0]->binding[1].offset = 48;
   layouts[0]->binding[1].stride = 16;
   nir_store_ssbo(&b, nir_channel(&b, loaded, 0), nir_imm_int(&b, 1),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   free(compiled.binary);
}

TEST_F(RootDescriptors, FormatlessStoreUsesDescriptorFormatAndTexelStride)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   layouts[0]->binding[0].stride = 48;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "formatless_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_NONE;
   nir_image_deref_store(
      &b, &nir_build_deref_var(&b, var)->def,
      nir_pad_vec4(&b, nir_vec2(&b, index, nir_imm_int(&b, 0))),
      nir_undef(&b, 1, 32),
      nir_imm_ivec4(&b, 0x12345678, 0x23456789, 0x3456789a, 0x456789ab),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_NONE, .access = ACCESS_NON_READABLE,
      .src_type = nir_type_uint32);

   ASSERT_NO_FATAL_FAILURE(lower(false));
   std::vector<nir_def *> descriptor_words;
   std::vector<nir_def *> addresses;
   bool wide_store = false;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_global_constant &&
             intr->def.bit_size == 32) {
            nir_instr *address = nir_def_instr(intr->src[0].ssa);
            if (address->type == nir_instr_type_intrinsic) {
               auto *root = nir_instr_as_intrinsic(address);
               if (root->intrinsic == nir_intrinsic_load_ubo &&
                   nir_src_is_const(root->src[0]) &&
                   nir_src_as_uint(root->src[0]) == HK_APPLE9_ROOT_BINDING &&
                   nir_src_is_const(root->src[1]) &&
                   /* The root self-pointer precedes the descriptor sets. */
                   nir_src_as_uint(root->src[1]) == sizeof(uint64_t))
                  descriptor_words.push_back(&intr->def);
            }
         }
         /* Exclude assertion/printf stores emitted by the layout helper. */
         if (intr->intrinsic == nir_intrinsic_store_global &&
             (nir_intrinsic_access(intr) & ACCESS_NON_READABLE)) {
            addresses.push_back(intr->src[1].ssa);
            wide_store |= intr->src[0].ssa->bit_size == 32;
         }
      }
   }
   ASSERT_FALSE(descriptor_words.empty());
   ASSERT_FALSE(addresses.empty());
   EXPECT_TRUE(wide_store);
   bool runtime_stride = false;
   for (auto *address : addresses) {
      for (auto *word : descriptor_words)
         runtime_stride |= depends_on(address, word);
   }
   EXPECT_TRUE(runtime_stride);
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_TRUE(profile.writes_global);
   free(compiled.binary);
}

TEST_F(RootDescriptors, FormatlessTexelBufferStoreCompiles)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
   layouts[0]->binding[0].stride = 48;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_BUF, false, GLSL_TYPE_UINT),
      "formatless_buffer");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_NONE;
   nir_image_deref_store(
      &b, &nir_build_deref_var(&b, var)->def, nir_pad_vec4(&b, index),
      nir_undef(&b, 1, 32),
      nir_imm_ivec4(&b, 0x12345678, 0x23456789, 0, 0),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_BUF,
      .format = PIPE_FORMAT_NONE, .access = ACCESS_NON_READABLE,
      .src_type = nir_type_uint32);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason))
      << (reason ? reason : "no diagnostic");
   EXPECT_TRUE(profile.writes_global);
   free(compiled.binary);
}

TEST_F(RootDescriptors, ImageSizeCompilesSuccessfully)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   layouts[0]->binding[0].stride = 48;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "storage_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_def *size = nir_image_deref_size(
      &b, 2, 32, deref, nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
      .format = PIPE_FORMAT_R32_UINT);
   /* Use the size in an SSBO store */
   layouts[0]->binding[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   layouts[0]->binding[1].offset = 48;
   layouts[0]->binding[1].stride = 16;
   nir_store_ssbo(&b, nir_channel(&b, size, 0), nir_imm_int(&b, 1),
                  nir_imm_int(&b, 0), .write_mask = 1, .align_mul = 4);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
}

TEST_F(RootDescriptors, RGBA32ImageStoreCompilesSuccessfully)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   layouts[0]->binding[0].stride = 48;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT),
      "storage_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   var->data.image.format = PIPE_FORMAT_R32G32B32A32_UINT;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_image_deref_store(
      &b, deref, nir_pad_vec4(&b, nir_vec2(&b, index, nir_imm_int(&b, 0))),
      nir_undef(&b, 1, 32),
      nir_vec4(&b, nir_iadd_imm(&b, index, 17), nir_imm_int(&b, 0),
                nir_imm_int(&b, 0), nir_imm_int(&b, 0)),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R32G32B32A32_UINT, .access = ACCESS_NON_READABLE,
      .src_type = nir_type_uint32);
   ASSERT_NO_FATAL_FAILURE(lower(false));
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   const char *reason = nullptr;
   bool ok = agx_compile_apple9_tiny(b.shader, &compiled, &profile, &reason);
   ASSERT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_GT(compiled.info.binary_size, 0u);
   EXPECT_TRUE(profile.writes_global);
   free(compiled.binary);
}

TEST_F(RootDescriptors, TextureDerefsKeepConstantSelection)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   layouts[0]->binding[0].stride = 64;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      "sampled_image");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_def *coord = nir_imm_vec2(&b, 0.5, 0.5);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(coord);
   tex->src[1].src_type = nir_tex_src_texture_deref;
   tex->src[1].src = nir_src_for_ssa(deref);
   tex->src[2].src_type = nir_tex_src_sampler_deref;
   tex->src[2].src = nir_src_for_ssa(deref);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   hk_nir_lower_buffer_descriptors(b.shader, &rs, 2, sets);
   EXPECT_GE(nir_tex_instr_src_index(tex, nir_tex_src_texture_deref), 0);
   EXPECT_GE(nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref), 0);
   hk_apple9_shader_mapping mapping = {};
   const char *reason = nullptr;
   ASSERT_TRUE(hk_nir_lower_apple9_textures(
      b.shader, 2, sets, &mapping, &reason)) << (reason ? reason : "");
   EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_texture_deref), -1);
   EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref), -1);
   EXPECT_EQ(tex->texture_index, 0u);
   EXPECT_EQ(tex->sampler_index, 0u);
   EXPECT_EQ(mapping.texture_count, 1);
   EXPECT_EQ(mapping.sampler_count, 1);
   EXPECT_EQ(mapping.ubo_count, 0);
   EXPECT_EQ(mapping.ssbo_count, 0);
}

TEST_F(RootDescriptors, UniformTexelBufferUsesNativeTextureTable)
{
   layouts[0]->binding[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
   layouts[0]->binding[0].stride = 32;
   nir_variable *var = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_BUF, false, false, GLSL_TYPE_FLOAT),
      "texel_buffer");
   var->data.descriptor_set = 0;
   var->data.binding = 0;
   nir_def *deref = &nir_build_deref_var(&b, var)->def;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 1;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(nir_imm_int(&b, 17));
   tex->src[1].src_type = nir_tex_src_texture_deref;
   tex->src[1].src = nir_src_for_ssa(deref);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   hk_nir_lower_buffer_descriptors(b.shader, &rs, 2, sets);
   hk_apple9_shader_mapping mapping = {};
   const char *reason = nullptr;
   ASSERT_TRUE(hk_nir_lower_apple9_textures(
      b.shader, 2, sets, &mapping, &reason)) << (reason ? reason : "");

   EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_texture_deref), -1);
   EXPECT_EQ(mapping.texture_count, 1);
   EXPECT_EQ(mapping.textures[0].type, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER);
   EXPECT_EQ(mapping.textures[0].set, 0u);
   EXPECT_EQ(mapping.sampler_count, 0);
}
