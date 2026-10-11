/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_buffer_view.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "util/bitscan.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"

#include "agx_helpers.h"
#include "agx_nir_texture.h"
#include "agx_pack.h"
#include "hk_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"

VkFormatFeatureFlags2
hk_get_buffer_format_features(struct hk_physical_device *pdev,
                              VkFormat vk_format)
{
   VkFormatFeatureFlags2 features = 0;
   enum pipe_format p_format = hk_format_to_pipe_format(vk_format);

   if (p_format == PIPE_FORMAT_NONE)
      return 0;

   if (agx_vbo_supports_format(p_format))
      features |= VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT;

   if (ail_pixel_format[p_format].texturable &&
       !util_format_is_depth_or_stencil(p_format)) {

      features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;

      /* RGB32 specially supported for uniform texel buffers only. */
      if (util_is_power_of_two_nonzero(util_format_get_blocksize(p_format))) {
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT |
                     VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
      }

      if (p_format == PIPE_FORMAT_R32_UINT || p_format == PIPE_FORMAT_R32_SINT)
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   }

   if (pdev->dev.chip == AGX_CHIP_G16G) {
      /* Storage descriptors carry the texel layout for access without a
       * shader format. */
      if (features & VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT)
         features |= VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
      /* Texel buffer atomics address device memory, which has FP32
       * addition. */
      if (p_format == PIPE_FORMAT_R32_FLOAT)
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
      if (!hk_apple9_storage_format_supported(vk_format))
         features &= ~(VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT |
                       VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT |
                       VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                       VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT);
   }

   return features;
}

/* Byte offset [7:0], bit shift [12:8], width minus one [17:13] and
 * conversion kind [21:18] of each component of a texel of this format.
 */
void
hk_apple9_texel_layout(enum pipe_format format,
                       struct hk_apple9_texel_metadata *m)
{
   const struct util_format_description *desc = util_format_description(format);
   m->size_B = util_format_get_blocksize(format);
   for (unsigned c = 0; c < 4; ++c) {
      unsigned swizzle = desc->swizzle[c];
      unsigned kind, offset = 0, shift = 0, bits = 1;
      if (swizzle >= 4) {
         kind = swizzle == PIPE_SWIZZLE_1 ? HK_TEXEL_ONE : HK_TEXEL_ZERO;
      } else {
         const struct util_format_channel_description *chan =
            &desc->channel[swizzle];
         offset = (chan->shift / 32) * 4;
         shift = chan->shift % 32;
         bits = chan->size;
         if (format == PIPE_FORMAT_R9G9B9E5_FLOAT)
            kind = HK_TEXEL_RGB9E5;
         else if (format == PIPE_FORMAT_R11G11B10_FLOAT)
            kind = HK_TEXEL_FLOAT11;
         else if (chan->type == UTIL_FORMAT_TYPE_FLOAT)
            kind = bits == 16 ? HK_TEXEL_FLOAT16 : HK_TEXEL_FLOAT32;
         else if (chan->type == UTIL_FORMAT_TYPE_SIGNED)
            kind = chan->normalized ? HK_TEXEL_SNORM : HK_TEXEL_SINT;
         else
            kind = chan->normalized ? HK_TEXEL_UNORM : HK_TEXEL_UINT;
         if (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB && c < 3)
            kind = HK_TEXEL_SRGB;
      }
      m->component[c] = offset | (shift << 8) | ((bits - 1) << 13) |
                        (kind << 18);
   }
}

enum hk_apple9_store_encode
hk_apple9_store_encode(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB || !desc->nr_channels)
      return HK_STORE_ENCODE_GENERIC;
   const struct util_format_channel_description *first = &desc->channel[0];
   unsigned w = first->size;
   for (unsigned c = 0; c < desc->nr_channels; ++c) {
      const struct util_format_channel_description *chan = &desc->channel[c];
      if (desc->swizzle[c] != c || chan->size != w || chan->shift != c * w ||
          chan->type != first->type || chan->normalized != first->normalized ||
          chan->pure_integer != first->pure_integer)
         return HK_STORE_ENCODE_GENERIC;
   }
   if (w == 32 && !first->normalized &&
       (first->type == UTIL_FORMAT_TYPE_FLOAT ||
        first->type == UTIL_FORMAT_TYPE_UNSIGNED ||
        first->type == UTIL_FORMAT_TYPE_SIGNED))
      return HK_STORE_ENCODE_RAW32;
   if (w == 8 && first->normalized && first->type == UTIL_FORMAT_TYPE_UNSIGNED)
      return HK_STORE_ENCODE_UNORM8;
   if (w == 16 && first->type == UTIL_FORMAT_TYPE_FLOAT)
      return HK_STORE_ENCODE_FLOAT16;
   return HK_STORE_ENCODE_GENERIC;
}

/* Native texel-buffer state for elements of format at the given addresses
 * (read-only and read-write views of the same memory, any texel alignment),
 * shared by buffer views and descriptor-buffer descriptors. */
void
hk_pack_texel_buffer(struct hk_device *dev, enum pipe_format format,
                     uint64_t ro_addr, uint64_t rw_addr, uint32_t elements,
                     uint64_t residency, struct agx_texture_packed *tex,
                     struct agx_pbe_packed *pbe,
                     struct hk_apple9_texel_metadata *m,
                     uint8_t *apple9_texture)
{
   const struct util_format_description *desc = util_format_description(format);

   uint8_t format_swizzle[4] = {
      desc->swizzle[0],
      desc->swizzle[1],
      desc->swizzle[2],
      desc->swizzle[3],
   };

   if (util_format_is_depth_or_stencil(format)) {
      assert(!util_format_is_depth_and_stencil(format) &&
             "separate stencil always used");

      /* Broadcast depth and stencil */
      format_swizzle[0] = 0;
      format_swizzle[1] = 0;
      format_swizzle[2] = 0;
      format_swizzle[3] = 0;
   }

   /* Decompose the offset into a multiple of 16-bytes (which we can include in
    * the address) and an extra texel-aligned tail offset of up to 15 bytes.
    *
    * This lets us offset partially in the shader instead, getting
    * around alignment restrictions on the base address pointer.
    */
   uint64_t rw_base = rw_addr & ~0xfull;
   uint64_t ro_base = ro_addr & ~0xfull;
   uint32_t tail_offset_B = rw_addr & 0xf;
   assert((ro_addr & 0xf) == tail_offset_B);
   uint32_t tail_offset_el = tail_offset_B / util_format_get_blocksize(format);
   assert(tail_offset_el * util_format_get_blocksize(format) == tail_offset_B &&
          "must be texel aligned");

   const bool apple9 = dev->dev.chip == AGX_CHIP_G16G;
   struct AGX_TEXTURE texture = {AGX_TEXTURE_header};
   texture.dimension = AGX_TEXTURE_DIMENSION_2D;
   texture.layout = AGX_LAYOUT_LINEAR;
   texture.channels = ail_pixel_format[format].channels;
   texture.type = ail_pixel_format[format].type;
   texture.swizzle_r = agx_channel_from_pipe(format_swizzle[0]);
   texture.swizzle_g = agx_channel_from_pipe(format_swizzle[1]);
   texture.swizzle_b = agx_channel_from_pipe(format_swizzle[2]);
   texture.swizzle_a = agx_channel_from_pipe(format_swizzle[3]);

   texture.width = AGX_TEXTURE_BUFFER_WIDTH;
   texture.height = DIV_ROUND_UP(elements + tail_offset_el, texture.width);
   texture.first_level = texture.last_level = 0;

   texture.address = ro_base;
   texture.buffer_size_sw = elements;
   texture.buffer_offset_sw = tail_offset_el;

   texture.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
   texture.srgb_2_channel = texture.srgb && util_format_colormask(desc) == 0x3;

   texture.depth = 1;
   texture.stride = (texture.width * util_format_get_blocksize(format)) - 16;

   if (apple9) {
      bool packed = agx_apple9_pack_texture(apple9_texture, &texture);
      assert(packed);
      (void)packed;
      /* The common prefix is software query state on G16. Storage accesses
       * use the full address in apple9 metadata, not a legacy PBE pointer.
       */
      texture.address = 0;
   }
   agx_pack(tex, TEXTURE, cfg) {
      cfg = texture;
   }

   agx_pack(pbe, PBE, cfg) {
      cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
      cfg.layout = AGX_LAYOUT_LINEAR;
      cfg.channels = ail_pixel_format[format].channels;
      cfg.type = ail_pixel_format[format].type;
      cfg.srgb = util_format_is_srgb(format);

      assert(desc->nr_channels >= 1 && desc->nr_channels <= 4);

      for (unsigned i = 0; i < desc->nr_channels; ++i) {
         if (desc->swizzle[i] == 0)
            cfg.swizzle_r = i;
         else if (desc->swizzle[i] == 1)
            cfg.swizzle_g = i;
         else if (desc->swizzle[i] == 2)
            cfg.swizzle_b = i;
         else if (desc->swizzle[i] == 3)
            cfg.swizzle_a = i;
      }

      cfg.buffer = apple9 ? 0 : rw_base;
      cfg.buffer_offset_sw = tail_offset_el;

      cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
      cfg.height = DIV_ROUND_UP(elements, cfg.width);
      cfg.level = 0;
      cfg.stride = (cfg.width * util_format_get_blocksize(format)) - 4;
      cfg.layers = 1;
      cfg.levels = 1;
   };

   if (apple9) {
      m->base = rw_addr;
      m->elements = elements;
      hk_apple9_texel_layout(format, m);
      m->residency = residency;
   }

}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkBufferView *pBufferView)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_buffer, buffer, pCreateInfo->buffer);
   struct hk_buffer_view *view;

   view = vk_buffer_view_create(&device->vk, pCreateInfo, pAllocator,
                                sizeof(*view));
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   hk_pack_texel_buffer(device, hk_format_to_pipe_format(view->vk.format),
                        hk_buffer_address_ro(buffer, view->vk.offset),
                        hk_buffer_address_rw(buffer, view->vk.offset),
                        view->vk.elements,
                        device->dev.chip == AGX_CHIP_G16G
                           ? hk_buffer_apple9_residency(buffer) : 0,
                        &view->tex, &view->pbe, &view->apple9,
                        view->apple9_texture);

   *pBufferView = hk_buffer_view_to_handle(view);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyBufferView(VkDevice _device, VkBufferView bufferView,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_buffer_view, view, bufferView);

   if (!view)
      return;

   vk_buffer_view_destroy(&device->vk, pAllocator, &view->vk);
}
