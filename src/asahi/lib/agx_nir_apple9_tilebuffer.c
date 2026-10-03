/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020-2021 Collabora, Ltd.
 * Copyright 2019 Sonny Jiang <sonnyj608@gmail.com>
 * Copyright 2019 Advanced Micro Devices, Inc.
 * Copyright 2014 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir_apple9_tilebuffer.h"
#include "asahi/compiler/agx_compile_apple9.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/format/u_format.h"

nir_shader *
agx_nir_apple9_tilebuffer(const nir_shader_compiler_options *options,
                          const struct agx_apple9_tilebuffer_key *key)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, options,
      key->store ? "color store" : "color reload");
   if (!key->mask) {
      nir_shader_gather_info(b.shader, b.impl);
      return b.shader;
   }
   if (key->store) {
      b.shader->info.num_images = key->count;
      nir_def *group = nir_load_workgroup_id(&b);
      nir_def *xy =
         nir_vec2(&b, nir_imul_imm(&b, nir_channel(&b, group, 0), 32),
                  nir_imul_imm(&b, nir_channel(&b, group, 1),
                               key->samples == 4 ? 16 : 32));
      unsigned offset = 0;
      for (unsigned rt = 0; rt < key->count; ++rt) {
         if (key->mask & BITFIELD_BIT(rt)) {
            nir_image_store_block_agx(
               &b, nir_imm_int(&b, rt), nir_imm_int(&b, offset),
               key->layered
                  ? nir_vec4(&b, nir_channel(&b, xy, 0), nir_channel(&b, xy, 1),
                             nir_load_layer_id(&b), nir_imm_int(&b, 0))
                  : nir_pad_vec4(&b, xy),
               .image_array = key->layered,
               .format = agx_apple9_color_tile_format(key->formats[rt]),
               .image_dim =
                  key->samples > 1 ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D);
         }
         offset += 4 * agx_apple9_color_words(key->formats[rt]);
      }
   } else {
      b.shader->info.num_ubos = 1;
      nir_def *xy = nir_u2u32(&b, nir_load_pixel_coord(&b));
      unsigned offset = 0;
      for (unsigned rt = 0; rt < key->count; ++rt) {
         unsigned words = agx_apple9_color_words(key->formats[rt]);
         if (key->mask & BITFIELD_BIT(rt)) {
            bool array = key->array_mask & BITFIELD_BIT(rt);
            nir_def *params = nir_load_ubo(&b, 2, 32, nir_imm_int(&b, 0),
                                           nir_imm_int(&b, rt * 8),
                                           .align_mul = 4, .range = 64);
            nir_def *coords = xy;
            if (array) {
               nir_def *layer = nir_channel(&b, params, 1);
               if (key->layered)
                  layer = nir_iadd(&b, layer, nir_load_layer_id(&b));
               coords = nir_vec3(&b, nir_channel(&b, xy, 0),
                                 nir_channel(&b, xy, 1), layer);
            }
            enum pipe_format format = key->formats[rt];
            nir_alu_type type =
               util_format_is_pure_uint(format)   ? nir_type_uint32
               : util_format_is_pure_sint(format) ? nir_type_int32
                                                  : nir_type_float32;
            enum pipe_format tile_format =
               util_format_linear(agx_apple9_color_tile_format(format));
            for (unsigned sample = 0; sample < key->samples; ++sample) {
               nir_def *color;
               if (key->clear_mask & BITFIELD_BIT(rt)) {
                  color = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
                                       nir_imm_int(&b, 64 + rt * 16),
                                       .align_mul = 16, .range_base = 64,
                                       .range = 128);
                  if (util_format_is_srgb(format)) {
                     nir_def *alpha = nir_channel(&b, color, 3);
                     color = nir_vector_insert_imm(
                        &b, nir_format_linear_to_srgb(&b, color), alpha, 3);
                  }
               } else if (key->samples > 1) {
                  color = nir_txf_ms(&b, coords, nir_imm_int(&b, sample),
                                     .texture_index = rt, .sampler_index = rt,
                                     .dim = GLSL_SAMPLER_DIM_MS,
                                     .is_array = array, .dest_type = type);
               } else {
                  color = nir_txf(&b, coords, .lod = nir_channel(&b, params, 0),
                                  .texture_index = rt, .sampler_index = rt,
                                  .dim = GLSL_SAMPLER_DIM_2D, .is_array = array,
                                  .dest_type = type);
               }
               nir_def *packed = nir_format_pack_rgba(&b, tile_format, color);
               /* Narrow tile formats occupy a padded 32-bit word. */
               packed = nir_u2u32(&b, packed);
               nir_store_local_pixel_agx(
                  &b, nir_trim_vector(&b, packed, words),
                  nir_imm_intN_t(&b, 1u << sample, 16), nir_undef(&b, 2, 16),
                  .base = offset, .format = agx_apple9_color_raw_format(words),
                  .write_mask = BITFIELD_MASK(words));
            }
         }
         offset += 4 * words;
      }
   }
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}
