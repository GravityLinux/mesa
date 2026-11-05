/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "libagx.h"
#include "nir_builder.h"

/* Private, descriptor-free ABI for single-sample color transfers.
 * src is the buffer and dst the image, including bufferOffset and mip/base
 * layer, even for readback (where the memory operations are reversed).
 */
struct hk_meta_upload_push {
   uint64_t src, dst;
   uint64_t src_row_B, src_layer_B, dst_layer_B;
   uint32_t dst_stride, tile_width, tile_height;
   uint32_t dst_x, dst_y;
   uint32_t width, height, layers;
};

static inline void
hk_meta_upload_next_layer(struct hk_meta_upload_push *push)
{
   push->src += push->src_layer_B;
   push->dst += push->dst_layer_B;
}

/* The tiled case uses the shared image-atomic calculation. AIL records the
 * logical mip width; the twiddle helper needs complete tile rows. A zero
 * tile width selects strided linear memory (stride is then in bytes).
 */
static inline nir_def *
hk_meta_upload_offset_sized(nir_builder *b, nir_def *xy, nir_def *stride,
                            nir_def *tile_width, nir_def *tile_height,
                            unsigned texel_B)
{
   nir_push_if(b, nir_ine_imm(b, tile_width, 0));
   nir_def *tiled_stride = nir_iand(
      b, nir_iadd(b, stride, nir_iadd_imm(b, tile_width, -1)),
      nir_ineg(b, tile_width));
   nir_def *tiled = nir_imul_imm(
      b, nir_u2u64(b, libagx_twiddle_coordinates(
            b, nir_u2u16(b, xy), nir_u2u16(b, tile_width),
            nir_u2u16(b, tile_height), tiled_stride)), texel_B);
   nir_push_else(b, NULL);
   nir_def *linear = nir_iadd(
      b, nir_imul_imm(b, nir_u2u64(b, nir_channel(b, xy, 0)), texel_B),
      nir_imul(b, nir_u2u64(b, nir_channel(b, xy, 1)), nir_u2u64(b, stride)));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, tiled, linear);
}

static inline nir_def *
hk_meta_upload_offset(nir_builder *b, nir_def *xy, nir_def *stride,
                      nir_def *tile_width, nir_def *tile_height)
{
   return hk_meta_upload_offset_sized(b, xy, stride, tile_width, tile_height, 4);
}

/* Shared with the rebased native buffer-copy loop. No launch may exceed the
 * CDM's 16-bit X group count, even for a large image's staging/readback buffer.
 */
static inline unsigned
hk_meta_upload_copy_count(uint64_t bytes, unsigned unit_B, unsigned local)
{
   return MIN2(bytes / unit_B, (uint64_t)local * 65535);
}

static inline nir_shader *
hk_meta_upload_shader_sized(unsigned texel_B, bool readback)
{
   assert(texel_B == 1 || texel_B == 2 || texel_B == 4 ||
          texel_B == 8 || texel_B == 16);
   nir_builder build =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "hk-meta-upload");
   nir_builder *b = &build;
   b->shader->info.workgroup_size[0] = 32;
   b->shader->info.workgroup_size[1] = 32;
   b->shader->info.workgroup_size[2] = 1;

#define UPLOAD_PUSH(name)                                                      \
   nir_load_push_constant(                                                     \
      b, 1, sizeof(((struct hk_meta_upload_push *)0)->name) * 8,                \
      nir_imm_int(b, offsetof(struct hk_meta_upload_push, name)))

   nir_def *coord = nir_load_global_invocation_id(b, 32);
   nir_def *x = nir_channel(b, coord, 0);
   nir_def *y = nir_channel(b, coord, 1);
   nir_def *z = nir_u2u64(b, nir_channel(b, coord, 2));
   nir_def *extent = nir_vec3(b, UPLOAD_PUSH(width), UPLOAD_PUSH(height),
                            UPLOAD_PUSH(layers));
   nir_push_if(b, nir_ball(b, nir_ult(b, coord, extent)));
   {
      nir_def *src_offset = nir_iadd(
         b, nir_imul_imm(b, nir_u2u64(b, x), texel_B),
         nir_iadd(b, nir_imul(b, nir_u2u64(b, y), UPLOAD_PUSH(src_row_B)),
                  nir_imul(b, z, UPLOAD_PUSH(src_layer_B))));
      nir_def *xy = nir_vec2(b, nir_iadd(b, x, UPLOAD_PUSH(dst_x)),
                            nir_iadd(b, y, UPLOAD_PUSH(dst_y)));
      nir_def *dst_offset = hk_meta_upload_offset_sized(
         b, xy, UPLOAD_PUSH(dst_stride), UPLOAD_PUSH(tile_width),
         UPLOAD_PUSH(tile_height), texel_B);
      dst_offset = nir_iadd(b, dst_offset,
                           nir_imul(b, z, UPLOAD_PUSH(dst_layer_B)));
      nir_def *buffer = nir_iadd(b, UPLOAD_PUSH(src), src_offset);
      nir_def *image = nir_iadd(b, UPLOAD_PUSH(dst), dst_offset);
      nir_def *value = nir_load_global(
         b, DIV_ROUND_UP(texel_B, 4), MIN2(32, texel_B * 8),
         readback ? image : buffer, .align_mul = texel_B);
      nir_store_global(b, value, readback ? buffer : image,
                       .align_mul = texel_B);
   }
   nir_pop_if(b, NULL);
#undef UPLOAD_PUSH
   return b->shader;
}

static inline nir_shader *
hk_meta_upload_shader(void)
{
   return hk_meta_upload_shader_sized(4, false);
}
