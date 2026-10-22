/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "../hk_image_view.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); return 1; } } while (0)

int
main(void)
{
   struct hk_render_merge_key first, next;
   struct vk_image color_image = {0}, depth_image = {0}, other_image = {0};
   struct hk_image_view color = {0}, depth = {0};
   color.vk.image = &color_image;
   color.vk.format = VK_FORMAT_R8G8B8A8_UNORM;
   depth.vk.image = &depth_image;
   depth.vk.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
   struct hk_image_view *colors[] = {&color};
   VkRect2D area = {.extent = {128, 128}};
   CHECK(hk_render_merge_key_init(&first, 0, 1, 0, 1, colors,
                                   &depth, &depth, area));

   /* Distinct views of the same subresource may continue the render. */
   struct hk_image_view alias = depth;
   CHECK(hk_render_merge_key_init(&next, 0, 1, 0, 1, colors,
                                   &alias, &alias, area));
   CHECK(!memcmp(&first, &next, sizeof(first)));

   /* Depth identity, mip, layer, format, aspect and render area all matter. */
   for (unsigned change = 0; change < 7; ++change) {
      alias = depth;
      VkRect2D altered = area;
      switch (change) {
      case 0: alias.vk.image = &other_image; break;
      case 1: alias.vk.base_mip_level = 1; break;
      case 2: alias.vk.base_array_layer = 1; break;
      case 3: alias.vk.format = VK_FORMAT_D24_UNORM_S8_UINT; break;
      case 4: altered.offset.x = 1; break;
      case 5: altered.extent.height = 64; break;
      default: break;
      }
      CHECK(hk_render_merge_key_init(&next, 0, 1, 0, 1, colors, &alias,
                                      change == 6 ? NULL : &alias, altered));
      CHECK(memcmp(&first, &next, sizeof(first)));
   }
   CHECK(!hk_render_merge_key_init(&next, VK_RENDERING_SUSPENDING_BIT,
                                    1, 0, 1, colors, &depth, &depth, area));
   CHECK(!hk_render_merge_key_init(&next, 0, 1, 1, 1, colors,
                                    &depth, &depth, area));
   CHECK(!hk_render_merge_key_init(&next, 0, 1, 0, 0, colors,
                                    &depth, &depth, area));

   /* Preserve the existing colour-only render-area freedom. */
   CHECK(hk_render_merge_key_init(&first, 0, 1, 0, 1, colors, NULL, NULL, area));
   area.offset.x = 1;
   CHECK(hk_render_merge_key_init(&next, 0, 1, 0, 1, colors, NULL, NULL, area));
   CHECK(!memcmp(&first, &next, sizeof(first)));
   puts("PASS depth/stencil render merge subresources, areas and exclusions");
   return 0;
}
