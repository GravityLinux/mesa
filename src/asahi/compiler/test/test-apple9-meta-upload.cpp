/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <array>
#include "agx_compile_apple9.h"
#include "asahi/layout/layout.h"
#include "asahi/vulkan/hk_meta_upload.h"

class Apple9MetaUpload : public ::testing::TestWithParam<unsigned> {
protected:
   void SetUp() override { glsl_type_singleton_init_or_ref(); }
   void TearDown() override { glsl_type_singleton_decref(); }
};

static void
specialize(nir_shader *shader, const hk_meta_upload_push &push,
           std::array<unsigned, 3> coord)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         nir_def *value = nullptr;
         if (intr->intrinsic == nir_intrinsic_load_push_constant) {
            uint64_t raw = 0;
            memcpy(&raw, reinterpret_cast<const char *>(&push) +
                          nir_src_as_uint(intr->src[0]), intr->def.bit_size / 8);
            value = nir_imm_intN_t(&b, raw, intr->def.bit_size);
         } else if (intr->intrinsic == nir_intrinsic_load_global_invocation_id) {
            value = nir_imm_ivec3(&b, coord[0], coord[1], coord[2]);
         }
         if (value) {
            nir_def_rewrite_uses(&intr->def, value);
            nir_instr_remove(instr);
         }
      }
   }
   agx_nir_lower_apple9_math(shader);
   bool progress;
   do {
      progress = nir_opt_constant_folding(shader);
      progress |= nir_opt_copy_prop(shader);
      progress |= nir_opt_dce(shader);
      progress |= nir_opt_if(shader, nir_opt_if_optimize_phi_true_false);
      progress |= nir_opt_dead_cf(shader);
   } while (progress);
}

TEST_P(Apple9MetaUpload, PaddedAddressesAndRawWidthMatchAIL)
{
   const unsigned bytes = GetParam();
   const pipe_format formats[] = {
      PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R8G8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
   };
   pipe_format format = formats[util_logbase2(bytes)];
   /* Page tiles from ail_get_max_tile_size, plus rectangular mip-tail tiles.
    * Use inline AIL address helpers, not a duplicate of the GPU formula.
    */
   const struct ail_tile tiles[] = {
      {bytes <= 2 ? 128u : bytes <= 8 ? 64u : 32u,
       bytes == 1 ? 128u : bytes <= 4 ? 64u : 32u},
      {16, 8}, {32, 32},
   };
   for (bool linear : {false, true}) {
      for (auto tile : tiles) {
         ail_layout layout = {};
         layout.format = format;
         layout.tiling = linear ? AIL_TILING_LINEAR : AIL_TILING_GPU;
         layout.sample_count_sa = 1;
         layout.levels = linear ? 1 : 2;
         layout.layer_stride_B = 0x40000;
         layout.linear_stride_B = ALIGN_POT(256 * bytes, 128);
         unsigned level = linear ? 0 : 1;
         layout.level_offsets_B[level] = linear ? 0 : 0x4000;
         layout.stride_el[level] = 256;
         layout.tilesize_el[level] = tile;
         const uint64_t buffer = 0x100000010ull, image = 0x200000000ull;
         hk_meta_upload_push push = {
            .src = buffer,
            .dst = image + ail_get_layer_level_B(&layout, 1, level),
            .src_row_B = 72 * bytes, .src_layer_B = 72 * 38 * bytes,
            .dst_layer_B = layout.layer_stride_B,
            .dst_stride = linear ? layout.linear_stride_B : 256,
            .tile_width = linear ? 0u : tile.width_el,
            .tile_height = linear ? 0u : tile.height_el,
            .dst_x = 7, .dst_y = 11, .width = 67, .height = 35, .layers = 2,
         };
         for (bool readback : {false, true}) {
            for (auto coord : {std::array<unsigned, 3>{0, 0, 0},
                              {66, 34, 0}, {31, 21, 1}, {32, 22, 1},
                              {67, 0, 0}, {0, 35, 0}, {0, 0, 2}}) {
               SCOPED_TRACE(::testing::Message() << bytes << " linear=" << linear
                            << " readback=" << readback << " x=" << coord[0]
                            << " y=" << coord[1] << " layer=" << coord[2]);
               nir_shader *shader = hk_meta_upload_shader_sized(bytes, readback);
               specialize(shader, push, coord);
               unsigned x = coord[0] + 7, y = coord[1] + 11, z = coord[2] + 1;
               uint64_t image_offset;
               if (linear) {
                  image_offset = ail_get_linear_pixel_B(&layout, level, x, y, z);
               } else {
                  unsigned tx = x & ~(tile.width_el - 1);
                  unsigned ty = y & ~(tile.height_el - 1);
                  image_offset = ail_get_twiddled_block_B(&layout, level, tx, ty, z) +
                     bytes * (ail_space_bits(x - tx) + 2 * ail_space_bits(y - ty));
               }
               uint64_t buffer_address = buffer + coord[0] * bytes +
                  coord[1] * push.src_row_B + coord[2] * push.src_layer_B;
               uint64_t image_address = image + image_offset;
               unsigned loads = 0, stores = 0;
               nir_def *loaded = nullptr;
               nir_foreach_block(block, nir_shader_get_entrypoint(shader)) {
                  nir_foreach_instr(instr, block) {
                     if (instr->type != nir_instr_type_intrinsic)
                        continue;
                     auto *intr = nir_instr_as_intrinsic(instr);
                     if (intr->intrinsic == nir_intrinsic_load_global) {
                        ++loads;
                        loaded = &intr->def;
                        ASSERT_TRUE(nir_src_is_const(intr->src[0]));
                        EXPECT_EQ(nir_src_as_uint(intr->src[0]),
                                  readback ? image_address : buffer_address);
                        EXPECT_EQ(loaded->num_components * loaded->bit_size / 8, bytes);
                     } else if (intr->intrinsic == nir_intrinsic_store_global) {
                        ++stores;
                        ASSERT_TRUE(nir_src_is_const(intr->src[1]));
                        EXPECT_EQ(nir_src_as_uint(intr->src[1]),
                                  readback ? buffer_address : image_address);
                        EXPECT_EQ(intr->src[0].ssa, loaded);
                        EXPECT_EQ(util_bitcount(nir_intrinsic_write_mask(intr)) *
                                  intr->src[0].ssa->bit_size / 8, bytes);
                        EXPECT_EQ(nir_intrinsic_align_mul(intr), bytes);
                     }
                  }
               }
               unsigned count = coord[0] < 67 && coord[1] < 35 && coord[2] < 2;
               EXPECT_EQ(loads, count);
               EXPECT_EQ(stores, count);
               ralloc_free(shader);
            }
         }
         /* Production launches one layer at a time with Z=0, rebasing both
          * pointers. Verify the same base-layer/mip and padded layer spacing.
          */
         for (unsigned layer = 1; layer <= 3; ++layer) {
            EXPECT_EQ(push.src, buffer + (layer - 1) * 72 * 38 * bytes);
            EXPECT_EQ(push.dst, image + ail_get_layer_level_B(&layout, layer, level));
            hk_meta_upload_next_layer(&push);
         }
      }
   }
}

TEST_P(Apple9MetaUpload, UnalignedTiledRowsMatchRealAILLayout)
{
   const unsigned bytes = GetParam();
   const pipe_format formats[] = {
      PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R8G8_UNORM, PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
   };
   ail_layout layout = {
      .width_px = 300, .height_px = 260, .depth_px = 2,
      .sample_count_sa = 1, .levels = 1, .tiling = AIL_TILING_GPU,
      .format = formats[util_logbase2(bytes)],
   };
   ail_make_miptree(&layout);
   const struct ail_tile tile = layout.tilesize_el[0];
   const uint64_t buffer = 0x100000010ull, image = 0x200000000ull;
   const hk_meta_upload_push push = {
      .src = buffer, .dst = image,
      .src_row_B = 304 * bytes, .src_layer_B = 304 * 264 * bytes,
      .dst_layer_B = layout.layer_stride_B,
      .dst_stride = layout.stride_el[0],
      .tile_width = tile.width_el, .tile_height = tile.height_el,
      .width = 300, .height = 260, .layers = 2,
   };
   for (bool readback : {false, true}) {
      for (auto coord : {std::array<unsigned, 3>{0, 0, 0},
                        {tile.width_el + 1, tile.height_el + 1, 1},
                        {299, 259, 0}}) {
         SCOPED_TRACE(::testing::Message() << bytes << " readback=" << readback
                      << " x=" << coord[0] << " y=" << coord[1]);
         nir_shader *shader = hk_meta_upload_shader_sized(bytes, readback);
         specialize(shader, push, coord);
         const unsigned x = coord[0], y = coord[1];
         const uint64_t expected_image = image +
            ail_get_twiddled_block_B(&layout, 0, x & ~(tile.width_el - 1),
                                     y & ~(tile.height_el - 1), coord[2]) +
            bytes * (ail_space_bits(x & (tile.width_el - 1)) +
                     2 * ail_space_bits(y & (tile.height_el - 1)));
         const uint64_t expected_buffer = buffer + x * bytes +
            y * push.src_row_B + coord[2] * push.src_layer_B;
         unsigned loads = 0, stores = 0;
         nir_foreach_block(block, nir_shader_get_entrypoint(shader)) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_global) {
                  ++loads;
                  ASSERT_TRUE(nir_src_is_const(intr->src[0]));
                  EXPECT_EQ(nir_src_as_uint(intr->src[0]),
                            readback ? expected_image : expected_buffer);
               } else if (intr->intrinsic == nir_intrinsic_store_global) {
                  ++stores;
                  ASSERT_TRUE(nir_src_is_const(intr->src[1]));
                  EXPECT_EQ(nir_src_as_uint(intr->src[1]),
                            readback ? expected_buffer : expected_image);
               }
            }
         }
         EXPECT_EQ(loads, 1u);
         EXPECT_EQ(stores, 1u);
         ralloc_free(shader);
      }
   }
}

TEST_P(Apple9MetaUpload, DynamicUploadAndReadbackCompileNative)
{
   for (bool readback : {false, true}) {
      nir_shader *shader = hk_meta_upload_shader_sized(GetParam(), readback);
      shader->options = &agx_nir_options;
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_push_constant)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def *value = nir_load_ubo(
               &b, 1, intr->def.bit_size, nir_imm_int(&b, 0), intr->src[0].ssa,
               .align_mul = unsigned(intr->def.bit_size) / 8,
               .range = sizeof(hk_meta_upload_push));
            nir_def_rewrite_uses(&intr->def, value);
            nir_instr_remove(instr);
         }
      }
      agx_shader_part compiled = {};
      agx_apple9_compute_profile profile = {};
      const char *reason = nullptr;
      ASSERT_TRUE(agx_compile_apple9_tiny(shader, &compiled, &profile, &reason))
         << (reason ? reason : "no diagnostic");
      EXPECT_TRUE(profile.writes_global);
      EXPECT_GT(compiled.info.binary_size, 0u);
      free(compiled.binary);
      ralloc_free(shader);
   }
}

INSTANTIATE_TEST_SUITE_P(TexelBytes, Apple9MetaUpload,
                        ::testing::Values(1u, 2u, 4u, 8u, 16u));

TEST(Apple9MetaUploadSplit, LargeImageStagingCopyIsBoundedAndComplete)
{
   const uint64_t bytes = 4096ull * 4096 * 8;
   for (unsigned unit : {1u, 16u}) {
      unsigned local = unit == 16 ? 32 : 16;
      uint64_t copied = 0;
      unsigned launches = 0;
      while (copied < bytes) {
         unsigned count = hk_meta_upload_copy_count(bytes - copied, unit, local);
         ASSERT_GT(count, 0u);
         EXPECT_LE(DIV_ROUND_UP(count, local), 65535u);
         EXPECT_LE(uint64_t(count) * unit, bytes - copied);
         copied += uint64_t(count) * unit;
         ++launches;
      }
      EXPECT_EQ(copied, bytes);
      EXPECT_GT(launches, 1u);
   }
}
