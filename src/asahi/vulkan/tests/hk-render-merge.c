/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "../hk_image_view.h"
#include "../hk_image.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); return 1; } } while (0)

static int
check_depth_barriers(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   struct hk_image *image = calloc(1, sizeof(*image));
   CHECK(dev && cmd && image);
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   uint8_t pool[4096] = {0}, bytes[4096] = {0};
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;

   for (unsigned kind = 0; kind < 3; ++kind) {
      struct hk_cs render = {
         .cmd = cmd, .type = HK_CS_VDM, .sync_id = 0,
         .current = bytes, .end = bytes + sizeof(bytes),
         .sync_stages = HK_SYNC_GRAPHICS, .stats.cmds = 1,
      };
      cmd->sync_render_continuation = false;
      util_dynarray_clear(&cmd->sync_events);
      hk_cmd_sync_work(cmd, &render, HK_SYNC_GRAPHICS);
      cmd->current_cs.gfx = cmd->merge_cs = &render;
      cmd->merge_event = 0;
      cmd->merge_reach = hk_merge_graphics_stages();
      cmd->merge_key.image[HK_MAX_RTS] = &image->vk;
      cmd->barrier_image = kind == 0 ? image : NULL;
      hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
         kind == 2 ? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT :
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         kind == 0 ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT :
         kind == 1 ? VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT :
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
      /* Sampling stored depth (including input attachments) ends the render;
       * a depth attachment -> depth attachment barrier stays in-tile. */
      CHECK((cmd->merge_cs == &render) == (kind == 2));
      CHECK((cmd->current_cs.gfx == &render) == (kind == 2));
      cmd->merge_cs = cmd->current_cs.gfx = NULL;
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   free(image);
   free(cmd);
   free(dev);
   return 0;
}

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
   CHECK(!check_depth_barriers());
   puts("PASS depth/stencil render merge subresources, areas and exclusions");
   return 0;
}
