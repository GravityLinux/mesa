/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "../hk_cmd_buffer.h"
#include "../hk_feedback.h"
#include "../hk_image.h"
#include "../hk_image_view.h"
#include "asahi/lib/agx_tilebuffer.h"
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #x);                            \
         return 1;                                                             \
      }                                                                        \
   } while (0)
int
main(void)
{
   const VkFormat formats[] = {VK_FORMAT_R8_UINT, VK_FORMAT_R16_UINT,
                               VK_FORMAT_R8G8_UINT, VK_FORMAT_R32_UINT,
                               VK_FORMAT_R16G16_UINT, VK_FORMAT_R32G32_UINT,
                               VK_FORMAT_R8G8B8A8_SRGB,
                               VK_FORMAT_R8G8B8A8_SNORM,
                               VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                               VK_FORMAT_R16_SFLOAT,
                               VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_FORMAT_R32G32B32A32_SFLOAT};
   struct agx_device dev = {.chip = AGX_CHIP_G16G};
   for (unsigned i = 0; i < ARRAY_SIZE(formats); ++i) {
      VkFormat raw = hk_feedback_raw_format(formats[i]);
      CHECK(vk_format_is_uint(raw));
      CHECK(vk_format_get_blocksize(raw) ==
            vk_format_get_blocksize(formats[i]));
      CHECK(ail_pixel_format[hk_format_to_pipe_format(raw)].channels ==
            ail_pixel_format[hk_format_to_pipe_format(formats[i])].channels);
      CHECK(hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, formats[i], 1));
      enum pipe_format pf = hk_format_to_pipe_format(raw);
      struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout(&pf, 1, 1, true);
      struct drm_asahi_cmd_render render = {0};
      agx_tilebuffer_set_drm_cmd(&render, &tib);
      CHECK(render.utile_width_px % 16 == 0);
      CHECK(render.utile_height_px % 16 == 0);
      CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_3D, formats[i], 1));
      CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, formats[i], 4));
   }
   CHECK(hk_feedback_raw_format(VK_FORMAT_B10G11R11_UFLOAT_PACK32) ==
         VK_FORMAT_UNDEFINED);
   CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, VK_FORMAT_D16_UNORM,
                                   1));
   dev.chip = AGX_CHIP_G14G;
   CHECK(!hk_feedback_can_compress(&dev, VK_IMAGE_TYPE_2D, formats[0], 1));

   struct hk_cmd_buffer cmd = {0}, other = {0};
   struct hk_image a = {0}, b = {0};
   a.planes[0].bo = b.planes[0].bo = (void *)0x1000;
   struct hk_image_view view = {.vk = {.image = &a.vk,
                                       .base_mip_level = 2,
                                       .base_array_layer = 3,
                                       .layer_count = 1}};
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 4));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 1, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &b, 2, 3));
   CHECK(!hk_feedback_is_raw(&cmd, &a, 2, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 3));
   CHECK(!hk_feedback_is_raw(&cmd, &b, 2, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 4));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 1, 3));
   CHECK(hk_feedback_is_raw(&cmd, &a, 2, 3));
   CHECK(!hk_feedback_is_raw(&other, &a, 2, 3));
   hk_feedback_invalidate_view(&cmd, &view);
   CHECK(!hk_feedback_is_raw(&cmd, &a, 2, 3));
   CHECK(hk_feedback_is_raw(&cmd, &a, 2, 4));
   CHECK(hk_feedback_is_raw(&cmd, &a, 1, 3));
   CHECK(!hk_feedback_is_raw(&cmd, &b, 2, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 3));
   hk_feedback_invalidate_range(&cmd, &a, 2, 1, 3, 1);
   CHECK(!hk_feedback_is_raw(&cmd, &a, 2, 3));
   CHECK(hk_feedback_is_raw(&cmd, &a, 2, 4));
   CHECK(hk_feedback_is_raw(&cmd, &a, 1, 3));
   CHECK(hk_feedback_mark_raw(&cmd, &a, 2, 3));
   hk_feedback_invalidate(&cmd);
   CHECK(!hk_feedback_is_raw(&cmd, &a, 2, 3));
   util_dynarray_fini(&cmd.feedback_raw);
   puts(
      "Integer compression classes and recording-local raw subresources: PASS");
   return 0;
}
