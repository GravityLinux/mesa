/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <string.h>
#include "hk_descriptor_set.h"
#include "hk_descriptor_buffer.h"
#include "asahi/layout/layout.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "failed: %s\n", #c); return 1; } } while (0)

int main(void)
{
   const enum pipe_format supported[] = {
      PIPE_FORMAT_R11G11B10_FLOAT, PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R16G16_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(supported); ++i)
      CHECK(hk_apple9_native_store_format(supported[i]));
   const enum pipe_format fallback[] = {
      PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_R32_UINT, PIPE_FORMAT_R32_SINT,
      PIPE_FORMAT_R16_FLOAT, PIPE_FORMAT_R9G9B9E5_FLOAT, PIPE_FORMAT_Z32_FLOAT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(fallback); ++i)
      CHECK(!hk_apple9_native_store_format(fallback[i]));
   CHECK(sizeof(struct hk_apple9_storage_descriptor) % AGX_APPLE9_TEXTURE_TABLE_STRIDE == 0);
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
   uint8_t entries[12][AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   for (unsigned i = 0; i < 12; ++i) {
      uint64_t base = UINT64_C(0x2ffff00000) + ail_get_layer_offset_B(&layout, 3 + i);
      CHECK(agx_apple9_translate_pbe(entries[i], &legacy, layout.format,
         &layout, base, 0, 1, false));
      uint64_t address;
      memcpy(&address, entries[i] + 8, sizeof(address));
      CHECK((address & ((UINT64_C(1) << 40) - 1)) == base >> 4);
      CHECK(!(address >> 44)); /* No array operand or count on a 2D entry. */
   }
   CHECK(!memcmp(&legacy, &(struct agx_pbe_packed){0}, sizeof(legacy)));
   puts("native storage: eligibility, descriptor ABI and per-layer addresses pass");
   return 0;
}
