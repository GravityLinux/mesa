/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9_msaa.h"
#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "nir_builder.h"
#include "nir_format_convert.h"
#include "util/format/u_format.h"

bool
hk_apple9_tilebuffer(enum pipe_format format, unsigned samples,
                     struct agx_tilebuffer_layout *tib)
{
   assert(samples == 1 || samples == 2 || samples == 4);
   /* Match the native fragment launch allocation, as in Gallium agx_state. */
   unsigned sample_size_B = ALIGN_POT(4 * agx_apple9_color_words(format), 8);
   unsigned tile_size = samples == 4 ? 32 * 16 : 32 * 32;
   /* agx_tilebuffer.c reserves the last byte of its 32-KiB tile budget.
    * Our EOT supports only these native shapes, not a 16x16 fallback. */
   if (sample_size_B * samples * tile_size > 32767)
      return false;
   *tib = agx_build_tilebuffer_layout(&format, 1, samples, false);
   tib->sample_size_B = sample_size_B;
   tib->tile_size = tile_size;
   agx_tilebuffer_pack_usc(tib);
   return true;
}

void
hk_apple9_set_sample_state(struct drm_asahi_cmd_render *cmd,
                           const struct agx_tilebuffer_layout *tib)
{
   agx_tilebuffer_set_drm_cmd(cmd, tib);
   cmd->samples = tib->nr_samples;
   cmd->sample_size_B = tib->sample_size_B;
   cmd->ppp_multisamplectl = agx_default_sample_positions(tib->nr_samples);
}

nir_shader *
hk_apple9_tile_nir(unsigned operation, bool srgb, enum pipe_format format,
                  unsigned samples, const VkClearColorValue *clear,
                  const VkRect2D *clear_area)
{
   assert(samples == 1 || samples == 2 || samples == 4);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "Honeykrisp Apple9 tile");
   if (operation == 2) {
      b.shader->info.num_images = 1;
      nir_def *group = nir_load_workgroup_id(&b);
      nir_def *xy = nir_vec2(
         &b, nir_imul_imm(&b, nir_channel(&b, group, 0), 32),
         nir_imul_imm(&b, nir_channel(&b, group, 1), samples == 4 ? 16 : 32));
      nir_image_store_block_agx(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
         nir_pad_vec4(&b, xy), .format = format,
         .image_dim = samples > 1 ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D);
   } else {
      nir_def *clear_color = NULL;
      if (operation == 0) {
         clear_color = nir_imm_ivec4(&b, clear->int32[0], clear->int32[1],
                                    clear->int32[2], clear->int32[3]);
         if (srgb)
            clear_color = nir_vector_insert_imm(
               &b, nir_format_linear_to_srgb(&b, clear_color),
               nir_channel(&b, clear_color, 3), 3);
      }
      bool reload = operation == 1 || clear_area;
      nir_def *xy = reload ? nir_u2u32(&b, nir_load_pixel_coord(&b)) : NULL;
      nir_def *inside = NULL;
      if (clear_area) {
         inside = nir_imm_true(&b);
         for (unsigned c = 0; c < 2; ++c) {
            unsigned lo = c ? clear_area->offset.y : clear_area->offset.x;
            unsigned size = c ? clear_area->extent.height : clear_area->extent.width;
            inside = nir_iand(&b, inside, nir_ult_imm(
               &b, nir_iadd_imm(&b, nir_channel(&b, xy, c), -(int)lo), size));
         }
      }
      /* A full clear broadcasts; a reload preserves every distinct sample,
       * including the pixels outside a partial clear's boundary tiles. */
      for (unsigned s = 0; s < (reload ? samples : 1); ++s) {
         nir_def *color = clear_color;
         if (reload) {
            nir_alu_type type = util_format_is_pure_uint(format) ? nir_type_uint32 :
                                util_format_is_pure_sint(format) ? nir_type_int32 :
                                                                  nir_type_float32;
            nir_def *loaded = samples > 1
               ? nir_txf_ms(&b, xy, nir_imm_int(&b, s),
                    .texture_index = 0, .sampler_index = 0,
                    .dim = GLSL_SAMPLER_DIM_MS, .dest_type = type)
               : nir_txf(&b, xy, .lod = nir_imm_int(&b, 0),
                    .texture_index = 0, .sampler_index = 0,
                    .dim = GLSL_SAMPLER_DIM_2D, .dest_type = type);
            color = inside ? nir_bcsel(&b, inside, clear_color, loaded) : loaded;
         }
         unsigned words = agx_apple9_color_words(format);
         nir_def *packed = nir_format_pack_rgba(&b, format, color);
         nir_store_local_pixel_agx(&b, nir_u2u32(&b, packed),
            nir_imm_intN_t(&b, reload ? BITFIELD_BIT(s) : BITFIELD_MASK(samples), 16),
            nir_undef(&b, 2, 16), .format = agx_apple9_color_raw_format(words),
            .write_mask = BITFIELD_MASK(words));
      }
   }
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}
