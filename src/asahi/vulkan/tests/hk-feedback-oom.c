/* SPDX-License-Identifier: MIT */
/* Exercise the real BeginRendering/draw/end paths after a decompression
 * allocation failure. No GPU or live Vulkan device is needed. */
#include "../hk_cmd_draw.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
static unsigned attempts;

bool __wrap_hk_meta_decompress_view(struct hk_cmd_buffer *, const struct hk_image_view *, unsigned);
bool
__wrap_hk_meta_decompress_view(struct hk_cmd_buffer *cmd,
                               const struct hk_image_view *view, unsigned layer)
{
   attempts++;
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return false;
}

VkResult __wrap___vk_errorf(const void *, VkResult, const char *, int, const char *, ...);
VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   return error;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && cmd);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   cmd->vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   struct hk_image image = {.vk = {.format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .image_type = VK_IMAGE_TYPE_2D,
                                   .samples = 1}};
   image.planes[0].layout = (struct ail_layout){.compressed = true,
      .width_px = 64, .height_px = 64, .sample_count_sa = 1};
   struct hk_image_view view = {.vk = {.base.type = VK_OBJECT_TYPE_IMAGE_VIEW,
                                        .image = &image.vk}};
   const VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(&view),
      .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo info = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {64, 64}}, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &att};
   VkCommandBuffer handle = hk_cmd_buffer_to_handle(cmd);
   hk_CmdBeginRendering(handle, &info);
   CHECK(attempts == 1 && vk_command_buffer_has_error(&cmd->vk));
   CHECK(!cmd->rendering && !cmd->current_cs.gfx);

   /* Draw entry points must not inspect absent shaders/render state or try
    * decompression again once recording has failed, including native mesh. */
   cmd->current_cs.gfx = (void *)(uintptr_t)1;
   hk_CmdDraw(handle, 3, 1, 0, 0);
   hk_CmdDrawIndexed(handle, 3, 1, 0, 0, 0);
   hk_CmdDrawMeshTasksEXT(handle, 1, 1, 1);
   hk_CmdEndRendering2KHR(handle, NULL);
   hk_CmdBeginRendering(handle, &info);
   CHECK(attempts == 1 && cmd->current_cs.gfx == (void *)(uintptr_t)1);
   CHECK(hk_EndCommandBuffer(handle) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   puts("failed feedback BeginRendering: later classic/mesh draws and end are inert");
   free(cmd);
   free(dev);
   return 0;
}
