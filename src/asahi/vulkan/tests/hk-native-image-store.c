/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <string.h>
#include "hk_descriptor_set.h"
#include "hk_descriptor_buffer.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_shader.h"
#include "compiler/nir/nir_builder.h"
#include "asahi/layout/layout.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "failed: %s\n", #c); return 1; } } while (0)

static int
test_image_storage(void)
{
   struct hk_image image = {0};
   CHECK(hk_apple9_native_store_image(&image, 0));

   /* Each sparse flag must independently rule out native stores, even when
    * there is no legacy sparse_map. This includes binding without residency.
    */
   for (unsigned flags = 1; flags < 8; ++flags) {
      image.vk.create_flags =
         ((flags & 1) ? VK_IMAGE_CREATE_SPARSE_BINDING_BIT : 0) |
         ((flags & 2) ? VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT : 0) |
         ((flags & 4) ? VK_IMAGE_CREATE_SPARSE_ALIASED_BIT : 0);
      CHECK(!image.planes[0].sparse_map);
      CHECK(!hk_apple9_native_store_image(&image, 0));
   }

   /* G16's residency table does not make its sparse layout native-addressable. */
   struct agx_bo residency = {0};
   image.planes[0].apple9_residency = &residency;
   image.vk.create_flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                           VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
   CHECK(!image.planes[0].sparse_map);
   CHECK(!hk_apple9_native_store_image(&image, 0));

   /* Ordinary aliasing, mutable formats and cube compatibility remain eligible;
    * compressed storage still takes the existing software fallback.
    */
   image.planes[0].apple9_residency = NULL;
   image.vk.create_flags = VK_IMAGE_CREATE_ALIAS_BIT |
                           VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                           VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   CHECK(hk_apple9_native_store_image(&image, 0));
   image.planes[0].layout.compressed = true;
   CHECK(!hk_apple9_native_store_image(&image, 0));
   /* Block-texel views of compressed images address blocks, not the layout's
    * texels; same-size reinterpretations of uncompressed images stay native. */
   struct ail_layout bc1 = {.format = PIPE_FORMAT_DXT1_RGB};
   CHECK(!hk_apple9_native_store_view_layout(&bc1, PIPE_FORMAT_R16G16B16A16_FLOAT));
   struct ail_layout astc = {.format = PIPE_FORMAT_ASTC_4x4};
   CHECK(!hk_apple9_native_store_view_layout(&astc, PIPE_FORMAT_R32G32B32A32_FLOAT));
   struct ail_layout rgba8 = {.format = PIPE_FORMAT_R8G8B8A8_UNORM};
   CHECK(hk_apple9_native_store_view_layout(&rgba8, PIPE_FORMAT_R8G8B8A8_UNORM));
   CHECK(hk_apple9_native_store_view_layout(&rgba8, PIPE_FORMAT_R32_FLOAT));
   CHECK(!hk_apple9_native_store_view_layout(&rgba8, PIPE_FORMAT_R16G16B16A16_FLOAT));
   puts("native storage: sparse layouts and VA use software stores");
   return 0;
}

int main(void)
{
   CHECK(test_image_storage() == 0);
   glsl_type_singleton_init_or_ref();
   const enum pipe_format supported[] = {
      PIPE_FORMAT_R11G11B10_FLOAT, PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R16G16_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32_UINT, PIPE_FORMAT_R32_SINT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R8G8B8A8_UINT, PIPE_FORMAT_R16G16B16A16_UINT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(supported); ++i)
      CHECK(hk_apple9_native_store_format(supported[i]));
   const enum pipe_format fallback[] = {
      PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_R8G8B8A8_SINT, PIPE_FORMAT_R16_UINT,
      PIPE_FORMAT_R16_FLOAT, PIPE_FORMAT_R9G9B9E5_FLOAT, PIPE_FORMAT_Z32_FLOAT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(fallback); ++i)
      CHECK(!hk_apple9_native_store_format(fallback[i]));
   CHECK(sizeof(struct hk_apple9_storage_descriptor) == 128);
   CHECK(HK_APPLE9_DB_STORAGE_SIZE == 128);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, native_ok) == 88);
   CHECK(offsetof(struct hk_apple9_storage_descriptor, store_encode) == 92);
   const uint32_t indices[] = {0, 1, 2048, HK_APPLE9_STORAGE_PBE_MASK};
   for (unsigned i = 0; i < ARRAY_SIZE(indices); ++i) {
      uint32_t packed = (indices[i] << HK_APPLE9_STORAGE_PBE_SHIFT) | 7;
      CHECK((packed & 7) == 7);
      CHECK(((packed >> HK_APPLE9_STORAGE_PBE_SHIFT) & HK_APPLE9_STORAGE_PBE_MASK) == indices[i]);
   }
   CHECK(HK_APPLE9_DB_STORAGE_SIZE >= sizeof(struct hk_apple9_storage_descriptor));
   CHECK(util_is_power_of_two_nonzero(HK_APPLE9_DB_STORAGE_SIZE));
   CHECK(!(HK_APPLE9_STORAGE_READ_VALID & HK_APPLE9_STORAGE_WRITE_VALID));
   CHECK(!(HK_APPLE9_STORAGE_READ_VALID & HK_APPLE9_STORAGE_LAYER_WRITE_VALID));

   /* Each face gets an immutable 2D entry; its address must advance by the
    * allocation's layer stride, including nonzero view base layers. */
   struct ail_layout layout = {
      .width_px = 8, .height_px = 8, .depth_px = 24, .levels = 1,
      .sample_count_sa = 1, .format = PIPE_FORMAT_R11G11B10_FLOAT,
      .tiling = AIL_TILING_TWIDDLED, .layer_stride_B = 0x4000,
   };
   struct agx_pbe_packed legacy = {0};
   ((uint8_t *)&legacy)[0] = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   struct agx_pbe_packed saved = legacy;
   uint8_t entries[12][AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   for (unsigned i = 0; i < 12; ++i) {
      uint64_t base = UINT64_C(0x2ffff00000) + ail_get_layer_offset_B(&layout, 3 + i);
      CHECK(hk_apple9_translate_storage_pbe(entries[i], &legacy, layout.format,
         &layout, base, 0));
      CHECK((entries[i][0] & 0xf) == AGX_TEXTURE_DIMENSION_2D);
      uint64_t address;
      memcpy(&address, entries[i] + 8, sizeof(address));
      CHECK((address & ((UINT64_C(1) << 40) - 1)) == base >> 4);
      CHECK(!(address >> 44)); /* No array operand or count on a 2D entry. */
   }
   CHECK(!memcmp(&legacy, &saved, sizeof(legacy)));
   /* Eligibility is frozen before any load is lowered/removed: a read after
    * a store must keep every store of the shader on the converter. */
   const nir_intrinsic_op reads[] = {
      nir_intrinsic_bindless_image_load, nir_intrinsic_bindless_image_atomic,
      nir_intrinsic_bindless_image_atomic_swap,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(reads); ++i) {
      const nir_shader_compiler_options options = {0};
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "read_write");
      CHECK(hk_apple9_native_store_shader(b.shader));
      const nir_intrinsic_info *store_info = &nir_intrinsic_infos[nir_intrinsic_bindless_image_store];
      nir_intrinsic_instr *store = nir_intrinsic_instr_create(b.shader, nir_intrinsic_bindless_image_store);
      store->num_components = 4;
      nir_intrinsic_set_image_dim(store, GLSL_SAMPLER_DIM_2D);
      nir_intrinsic_set_format(store, PIPE_FORMAT_R32_UINT);
      for (unsigned j = 0; j < store_info->num_srcs; ++j) {
         unsigned components = store_info->src_components[j] ? store_info->src_components[j] : 4;
         store->src[j] = nir_src_for_ssa(nir_imm_zero(&b, components, j == 0 ? 64 : 32));
      }
      nir_builder_instr_insert(&b, &store->instr);
      CHECK(hk_apple9_native_store_shader(b.shader));
      nir_intrinsic_instr *read = nir_intrinsic_instr_create(b.shader, reads[i]);
      read->num_components = reads[i] == nir_intrinsic_bindless_image_load ? 4 : 1;
      nir_intrinsic_set_image_dim(read, GLSL_SAMPLER_DIM_2D);
      nir_intrinsic_set_format(read, PIPE_FORMAT_R32_UINT);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[reads[i]];
      for (unsigned j = 0; j < info->num_srcs; ++j) {
         unsigned components = info->src_components[j] ? info->src_components[j] : read->num_components;
         read->src[j] = nir_src_for_ssa(nir_imm_zero(&b, components, j == 0 ? 64 : 32));
      }
      nir_def_init(&read->instr, &read->def, read->num_components, 32);
      nir_builder_instr_insert(&b, &read->instr);
      CHECK(!hk_apple9_native_store_shader(b.shader));
      ralloc_free(b.shader);
   }
   puts("native storage: eligibility, descriptor ABI and per-layer addresses pass");
   glsl_type_singleton_decref();
   return 0;
}
