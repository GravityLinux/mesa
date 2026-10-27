/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include "hk_apple9.h"
#include "hk_apple9_root.h"
#include "hk_apple9_msaa.h"
#include "hk_apple9_graphics.h"
#include "hk_descriptor_set_layout.h"
#include "asahi/genxml/agx_pack.h"
#include "nir_builder.h"
#include "vk_meta_private.h"
#include "vk_pipeline.h"
#include "vk_shader.h"
#include "vk_graphics_state.h"

struct vk_physical_device;
extern "C" void hk_preprocess_nir_internal(struct vk_physical_device *, nir_shader *);

class Apple9MSAA : public ::testing::Test {
protected:
   void SetUp() override { glsl_type_singleton_init_or_ref(); }
   void TearDown() override { glsl_type_singleton_decref(); }
};

TEST_F(Apple9MSAA, TextureDescriptorPreservesSampleCountAndAddress)
{
   for (unsigned samples : {1u, 2u, 4u}) {
      struct AGX_TEXTURE cfg = {AGX_TEXTURE_header};
      cfg.dimension = samples == 1 ? AGX_TEXTURE_DIMENSION_2D
                                   : AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
      cfg.layout = AGX_LAYOUT_TWIDDLED;
      cfg.width = 33;
      cfg.height = 17;
      cfg.depth = 1;
      cfg.address = UINT64_C(0x1000024000);
      cfg.samples = samples == 4 ? AGX_SAMPLE_COUNT_4 : AGX_SAMPLE_COUNT_2;
      struct agx_texture_packed input;
      AGX_TEXTURE_pack(reinterpret_cast<uint32_t *>(&input), &cfg);
      uint8_t output[32];
      ASSERT_TRUE(hk_apple9_pack_texture(output, &input));
      uint32_t header;
      uint64_t address;
      memcpy(&header, output + 4, 4);
      memcpy(&address, output + 8, 8);
      EXPECT_EQ(!!(header & (1u << 24)), samples == 4);
      EXPECT_EQ(address, cfg.address >> 4);
      EXPECT_EQ(output[0] & 15, cfg.dimension);
   }
}

TEST_F(Apple9MSAA, RGBA8FourSampleGeometryMatchesNativeKernelContract)
{
   agx_tilebuffer_layout tib;
   ASSERT_TRUE(hk_apple9_tilebuffer(PIPE_FORMAT_R8G8B8A8_UNORM, 4, &tib));
   drm_asahi_cmd_render cmd = {};
   hk_apple9_set_sample_state(&cmd, &tib);
   EXPECT_EQ(tib.sample_size_B, 8);
   EXPECT_EQ(cmd.samples, 4);
   EXPECT_EQ(cmd.sample_size_B, 8);
   EXPECT_EQ(cmd.ppp_multisamplectl, UINT64_C(0xeaa26e26));
   EXPECT_EQ(cmd.utile_width_px, 32);
   EXPECT_EQ(cmd.utile_height_px, 16);
   EXPECT_EQ(agx_tilebuffer_total_size(&tib), 16384u);
   unsigned utile_config = (cmd.utile_width_px / 16) << 12 |
                           (cmd.utile_height_px / 16) << 14 |
                           util_logbase2(tib.nr_samples);
   unsigned blocks = DIV_ROUND_UP(agx_tilebuffer_total_size(&tib), 2048);
   EXPECT_EQ(utile_config, 0x6002u);
   EXPECT_EQ(blocks, 8u);
   EXPECT_EQ((tib.nr_samples << 16) | blocks, 0x40008u);
}

TEST_F(Apple9MSAA, OneAndTwoSamplesKeepNative32PixelTileHeight)
{
   for (unsigned samples : {1u, 2u}) {
      agx_tilebuffer_layout tib;
      ASSERT_TRUE(hk_apple9_tilebuffer(PIPE_FORMAT_R8G8B8A8_UNORM, samples, &tib));
      drm_asahi_cmd_render cmd = {};
      hk_apple9_set_sample_state(&cmd, &tib);
      EXPECT_EQ(cmd.samples, samples);
      EXPECT_EQ(cmd.sample_size_B, 8);
      EXPECT_EQ(cmd.utile_width_px, 32);
      EXPECT_EQ(cmd.utile_height_px, 32);
      EXPECT_EQ(cmd.ppp_multisamplectl, samples == 1 ? 0x88u : 0x44ccu);
   }
}

TEST_F(Apple9MSAA, NativeShapesRejectOverBudgetWithoutChangingSingleSample)
{
   const pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_B8G8R8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_B8G8R8A8_SRGB,
      PIPE_FORMAT_R10G10B10A2_UNORM, PIPE_FORMAT_B10G10R10A2_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_SINT,
   };
   for (auto format : formats)
   for (unsigned samples : {1u, 2u, 4u}) {
      SCOPED_TRACE(format);
      SCOPED_TRACE(samples);
      agx_tilebuffer_layout tib;
      bool wide = format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
                  format == PIPE_FORMAT_R32G32B32A32_UINT ||
                  format == PIPE_FORMAT_R32G32B32A32_SINT;
      bool accepted = hk_apple9_tilebuffer(format, samples, &tib);
      ASSERT_EQ(accepted, samples == 1 || !wide);
      if (!accepted)
         continue;
      EXPECT_EQ(tib.sample_size_B, wide ? 16 : 8);
      EXPECT_EQ(tib.tile_size, samples == 4 ? 32 * 16 : 32 * 32);
      EXPECT_LE(agx_tilebuffer_total_size(&tib), 32767u);
      if (samples == 1) {
         auto original = agx_build_tilebuffer_layout(&format, 1, 1, false);
         EXPECT_EQ(tib.sample_size_B, original.sample_size_B);
         EXPECT_EQ(tib.tile_size, original.tile_size);
      }
   }
}

TEST_F(Apple9MSAA, BlendSpecializationsDistinguishSampleCounts)
{
   vk_color_blend_state cb = {};
   cb.attachment_count = 1;
   cb.color_write_enables = 1;
   cb.attachments[0].write_mask = 15;
   agx_apple9_blend previous = {};
   for (unsigned samples : {1u, 2u, 4u}) {
      agx_apple9_blend blend = {};
      ASSERT_TRUE(hk_apple9_blend_state(&cb, VK_FORMAT_R8G8B8A8_UNORM,
                                       samples, &blend));
      EXPECT_EQ(blend.samples, samples);
      EXPECT_NE(memcmp(&previous, &blend, sizeof(blend)), 0);
      previous = blend;
   }
}

TEST_F(Apple9MSAA, ActualRuntimeResolveVertexCompiles)
{
   nir_shader *nir = vk_meta_draw_rects_vs_nir(nullptr, false);
   nir->options = &agx_nir_options;
   hk_preprocess_nir_internal(nullptr, nir);
   agx_apple9_vertex_layout layout = {};
   layout.format[0] = PIPE_FORMAT_R32G32B32A32_UINT;
   layout.stride[0] = 16;
   layout.draw_params_ubo = AGX_APPLE9_GRAPHICS_SYSVAL_BINDING;
   agx_shader_part compiled = {};
   const char *reason = nullptr;
   ASSERT_TRUE(agx_compile_apple9_vertex_inputs(nir, &layout, &compiled, &reason))
      << (reason ? reason : "");
   EXPECT_TRUE(compiled.info.writes_layer_viewport);
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
   ralloc_free(nir);
}

TEST_F(Apple9MSAA, TileHelpersCompileForEveryAdmittedColorFormat)
{
   const pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R10G10B10A2_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_SINT,
   };
   const VkClearColorValue clear = {{0.25f, 0.5f, 0.75f, 1.0f}};
   const VkRect2D area = {{1, 2}, {29, 13}};
   for (auto format : formats)
   for (unsigned samples : {1u, 2u, 4u})
   for (unsigned op : {0u, 1u, 2u, 3u}) {
      SCOPED_TRACE(format);
      SCOPED_TRACE(samples);
      SCOPED_TRACE(op);
      nir_shader *nir = hk_apple9_tile_nir(op == 3 ? 0 : op, false, format,
                                          samples, &clear, op == 3 ? &area : nullptr);
      /* Verify the NIR sample addressing before native compilation mutates it. */
      unsigned stores = 0, masks = 0, fetches = 0, exports = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               auto *tex = nir_instr_as_tex(instr);
               EXPECT_EQ(tex->op, samples > 1 ? nir_texop_txf_ms : nir_texop_txf);
               ++fetches;
            } else if (instr->type == nir_instr_type_intrinsic) {
               auto *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
                  ++stores;
                  masks |= nir_src_as_uint(intr->src[1]);
               } else if (intr->intrinsic == nir_intrinsic_image_store_block_agx) {
                  EXPECT_EQ(nir_intrinsic_image_dim(intr),
                            samples > 1 ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D);
                  ++exports;
               }
            }
         }
      }
      EXPECT_EQ(exports, op == 2 ? 1u : 0u);
      EXPECT_EQ(fetches, op == 1 || op == 3 ? samples : 0u);
      EXPECT_EQ(stores, op == 2 ? 0u : op == 0 ? 1u : samples);
      if (stores) {
         EXPECT_EQ(masks, BITFIELD_MASK(samples));
      }
      agx_apple9_blend blend = {};
      blend.samples = samples;
      agx_shader_part compiled = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_fragment_blend(
         nir, nullptr, &blend, &compiled, &reason)) << (reason ? reason : "");
      EXPECT_EQ(compiled.info.apple9_resource_count, 0);
      EXPECT_EQ(compiled.info.apple9_image_mask, op == 2 ? 1u : 0u);
      EXPECT_GT(compiled.info.binary_size, 0u);
      free(compiled.binary);
      ralloc_free(nir);
   }
}

TEST_F(Apple9MSAA, ActualRuntimeColorResolveCompilesWithNativeDescriptors)
{
   const VkFormat formats[] = {
      VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM,
      VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB,
      VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32,
      VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8_UNORM,
      VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT,
      VK_FORMAT_R32G32B32A32_SINT,
   };
   for (auto format : formats)
   for (auto samples : {VK_SAMPLE_COUNT_2_BIT, VK_SAMPLE_COUNT_4_BIT}) {
      SCOPED_TRACE(format);
      SCOPED_TRACE(samples);
      vk_meta_blit_key key = {};
      key.dim = GLSL_SAMPLER_DIM_MS;
      key.src_samples = samples;
      bool integer = format == VK_FORMAT_R32G32B32A32_UINT ||
                     format == VK_FORMAT_R32G32B32A32_SINT;
      key.resolve_mode = integer ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT
                                 : VK_RESOLVE_MODE_AVERAGE_BIT;
      key.dst_format = format;
      key.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
      nir_shader *nir = vk_meta_build_blit_shader(&key);
      nir->options = &agx_nir_options;
      hk_preprocess_nir_internal(nullptr, nir);
      nir_lower_explicit_io(nir, nir_var_mem_push_const,
                           nir_address_format_32bit_offset);
      auto *layout = static_cast<hk_descriptor_set_layout *>(calloc(
         1, sizeof(hk_descriptor_set_layout) +
            2 * sizeof(hk_descriptor_set_binding_layout)));
      layout->binding_count = 2;
      layout->binding[0].type = VK_DESCRIPTOR_TYPE_SAMPLER;
      layout->binding[0].array_size = 1;
      layout->binding[0].stride = 32;
      layout->binding[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      layout->binding[1].array_size = 1;
      layout->binding[1].stride = 64;
      vk_descriptor_set_layout *sets[] = {&layout->vk};
      hk_nir_lower_buffer_descriptors(nir, &vk_robustness_disabled, 1, sets);
      hk_apple9_shader_mapping mapping = {};
      const char *reason = nullptr;
      ASSERT_TRUE(hk_nir_lower_apple9_textures(nir, 1, sets, &mapping, &reason))
         << (reason ? reason : "");
      nir_lower_explicit_io(nir, nir_var_mem_global,
                           nir_address_format_64bit_global);
      nir_lower_explicit_io(nir, nir_var_mem_ubo | nir_var_mem_ssbo,
                           nir_address_format_64bit_global_32bit_offset);
      nir_opt_constant_folding(nir);
      nir_opt_copy_prop(nir);
      hk_nir_lower_global_constant_offsets(nir, false);
      hk_nir_lower_apple9_root(nir);
      nir_opt_constant_folding(nir);
      nir_opt_copy_prop(nir);
      nir_opt_dce(nir);
      agx_apple9_blend blend = {};
      vk_color_blend_state cb = {};
      cb.attachment_count = 1;
      cb.color_write_enables = 1;
      cb.attachments[0].write_mask = 15;
      ASSERT_TRUE(hk_apple9_blend_state(&cb, format, 1, &blend));
      agx_shader_part compiled = {};
      ASSERT_TRUE(agx_compile_apple9_fragment_blend(
         nir, nullptr, &blend, &compiled, &reason)) << (reason ? reason : "");
      EXPECT_EQ(compiled.info.apple9_texture_mask, 1u);
      EXPECT_TRUE(compiled.info.apple9_uses_texel_fetch);
      EXPECT_GT(compiled.info.binary_size, 0u);
      free(compiled.binary);
      free(layout);
      ralloc_free(nir);
   }
}
