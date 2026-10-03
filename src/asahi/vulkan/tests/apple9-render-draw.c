/* SPDX-License-Identifier: MIT
 * Native Vulkan scissored triangle and guarded GPU readback test.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define CHECK(x)                                                               \
   do {                                                                        \
      VkResult r = (x);                                                        \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d (line %d)\n", #x, r, __LINE__);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)
int
main(int argc, char **argv)
{
   if ((argc != 3 && argc != 4) || (argc == 4 && strcmp(argv[3], "bounds") &&
                                    strcmp(argv[3], "bounds-wrap") &&
                                    strcmp(argv[3], "sample-mask"))) {
      fprintf(stderr,
              "Usage: %s vertex.spv fragment.spv [bounds|bounds-wrap|sample-mask]\n",
              argv[0]);
      return 1;
   }
   bool sample_mask = argc == 4 && !strcmp(argv[3], "sample-mask");
   bool bounds = argc == 4 && !sample_mask;
   bool far_vertex = bounds && !strcmp(argv[3], "bounds-wrap");
   setbuf(stdout, NULL);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "Apple9 offscreen draw",
                            .apiVersion = VK_API_VERSION_1_0};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t n = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &n, NULL));
   if (!n) {
      fprintf(stderr, "No Vulkan device\n");
      return 1;
   }
   VkPhysicalDevice *devices = calloc(n, sizeof(*devices));
   CHECK(vkEnumeratePhysicalDevices(instance, &n, devices));
   VkPhysicalDevice phy = devices[0];
   free(devices);
   VkPhysicalDeviceProperties prop;
   vkGetPhysicalDeviceProperties(phy, &prop);
   printf("device: %s vendor=%04x device=%04x API=%u.%u\n", prop.deviceName,
          prop.vendorID, prop.deviceID, VK_VERSION_MAJOR(prop.apiVersion),
          VK_VERSION_MINOR(prop.apiVersion));
   if (!strstr(prop.deviceName, "Apple M4")) {
      fprintf(stderr, "Expected native Apple M4 renderer\n");
      return 1;
   }
   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(phy, &nq, NULL);
   VkQueueFamilyProperties *qp = calloc(nq, sizeof(*qp));
   vkGetPhysicalDeviceQueueFamilyProperties(phy, &nq, qp);
   uint32_t qi;
   for (qi = 0; qi < nq; qi++)
      if (qp[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)
         break;
   free(qp);
   if (qi == nq)
      return 1;
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qi,
      .queueCount = 1,
      .pQueuePriorities = &priority};
   VkPhysicalDeviceFeatures features = {.robustBufferAccess = bounds};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .pEnabledFeatures = &features};
   VkDevice dev;
   CHECK(vkCreateDevice(phy, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qi, 0, &queue);
   enum { W = 37, H = 19, PIXELS = W * H, BYTES = PIXELS * 4, GUARD = 64 };
   VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                           .imageType = VK_IMAGE_TYPE_2D,
                           .format = VK_FORMAT_R8G8B8A8_UNORM,
                           .extent = {W, H, 1},
                           .mipLevels = 1,
                           .arrayLayers = 1,
                           .samples = VK_SAMPLE_COUNT_1_BIT,
                           .tiling = VK_IMAGE_TILING_OPTIMAL,
                           .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   VkImage image;
   CHECK(vkCreateImage(dev, &ii, NULL, &image));
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = BYTES + GUARD,
                            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer;
   CHECK(vkCreateBuffer(dev, &bi, NULL, &buffer));
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(phy, &mp);
   VkDeviceMemory memory[2];
   for (unsigned i = 0; i < 2; ++i) {
      VkMemoryRequirements mr;
      if (i)
         vkGetBufferMemoryRequirements(dev, buffer, &mr);
      else
         vkGetImageMemoryRequirements(dev, image, &mr);
      unsigned t;
      for (t = 0; t < mp.memoryTypeCount; ++t)
         if ((mr.memoryTypeBits & (1u << t)) &&
             (!i || (mp.memoryTypes[t].propertyFlags &
                     (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                       (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)))
            break;
      if (t == mp.memoryTypeCount)
         return 1;
      VkMemoryAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = mr.size,
         .memoryTypeIndex = t};
      CHECK(vkAllocateMemory(dev, &ai, NULL, &memory[i]));
   }
   CHECK(vkBindImageMemory(dev, image, memory[0], 0));
   CHECK(vkBindBufferMemory(dev, buffer, memory[1], 0));
   uint8_t *pixels;
   CHECK(vkMapMemory(dev, memory[1], 0, VK_WHOLE_SIZE, 0, (void **)&pixels));
   VkBuffer vertex_buffer = VK_NULL_HANDLE;
   VkDeviceMemory vertex_memory = VK_NULL_HANDLE;
   if (bounds) {
      VkBufferCreateInfo vb = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = 4,
                               .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
      CHECK(vkCreateBuffer(dev, &vb, NULL, &vertex_buffer));
      VkMemoryRequirements req;
      vkGetBufferMemoryRequirements(dev, vertex_buffer, &req);
      unsigned t;
      for (t = 0; t < mp.memoryTypeCount; ++t)
         if ((req.memoryTypeBits & (1u << t)) &&
             (mp.memoryTypes[t].propertyFlags &
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            break;
      if (t == mp.memoryTypeCount)
         return 1;
      VkMemoryAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size,
         .memoryTypeIndex = t};
      CHECK(vkAllocateMemory(dev, &ai, NULL, &vertex_memory));
      CHECK(vkBindBufferMemory(dev, vertex_buffer, vertex_memory, 0));
      float *mapped;
      CHECK(vkMapMemory(dev, vertex_memory, 0, VK_WHOLE_SIZE, 0,
                        (void **)&mapped));
      /* Only red belongs to the VkBuffer. Adjacent allocation bytes must not
       * leak into the other channels of the shader's vec4 attribute. */
      mapped[0] = 1.0f;
      mapped[1] = mapped[2] = mapped[3] = 0.75f;
      vkUnmapMemory(dev, vertex_memory);
   }
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = ii.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vi, NULL, &view));
   VkAttachmentDescription ad = {
      .format = ii.format,
      .samples = ii.samples,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
   VkAttachmentReference ar = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ar};
   VkRenderPassCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &ad,
      .subpassCount = 1,
      .pSubpasses = &sub};
   VkRenderPass render;
   CHECK(vkCreateRenderPass(dev, &ri, NULL, &render));

   VkShaderModule modules[2];
   for (unsigned i = 0; i < 2; ++i) {
      FILE *file = fopen(argv[i + 1], "rb");
      if (!file)
         return 1;
      fseek(file, 0, SEEK_END);
      size_t size = ftell(file);
      rewind(file);
      uint32_t *code = malloc(size);
      if (fread(code, 1, size, file) != size)
         return 1;
      fclose(file);
      VkShaderModuleCreateInfo mi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = size,
         .pCode = code};
      CHECK(vkCreateShaderModule(dev, &mi, NULL, &modules[i]));
      free(code);
   }
   VkPipelineLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &li, NULL, &layout));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = modules[1],
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = modules[0],
       .pName = "main"}};
   VkVertexInputBindingDescription binding = {0,
                                              far_vertex ? 2048
                                              : bounds   ? 0
                                                         : 8,
                                              VK_VERTEX_INPUT_RATE_VERTEX};
   VkVertexInputAttributeDescription attr = {
      0, 0, bounds ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R32G32_SFLOAT,
      0};
   VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = bounds,
      .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = bounds,
      .pVertexAttributeDescriptions = &attr};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport viewport = {0, 0, W, H, 0, 1};
   VkRect2D scissor = {{3, 2}, {W - 7, H - 5}};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ca = {
      .blendEnable = VK_FALSE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = 15};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &ca};
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_FALSE,
      .depthWriteEnable = VK_FALSE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   VkGraphicsPipelineCreateInfo graphics = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .pDepthStencilState = &ds,
      .layout = layout,
      .renderPass = render};
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &graphics, NULL,
                                   &pipeline));

   VkFramebufferCreateInfo fi = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render,
      .attachmentCount = 1,
      .pAttachments = &view,
      .width = W,
      .height = H,
      .layers = 1};
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fi, NULL, &fb));
   VkCommandPoolCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qi};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkFenceCreateInfo fence_info = {.sType =
                                      VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fence_info, NULL, &fence));
   for (unsigned pass = 0; pass < 3; ++pass) {
      const uint8_t expected[4] = {32 + pass * 31, 128 - pass * 17,
                                   223 - pass * 23, 255};
      memset(pixels, 0xa5, BYTES + GUARD);
      CHECK(vkResetCommandBuffer(cmd, 0));
      CHECK(vkResetFences(dev, 1, &fence));
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(cmd, &begin));
      VkClearValue clear;
      for (unsigned c = 0; c < 4; ++c)
         clear.color.float32[c] = expected[c] / 255.0f;
      VkRenderPassBeginInfo rb = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = render,
         .framebuffer = fb,
         .renderArea = {{0, 0}, {W, H}},
         .clearValueCount = 1,
         .pClearValues = &clear};
      vkCmdBeginRenderPass(cmd, &rb, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      if (bounds) {
         VkDeviceSize zero = 0;
         vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &zero);
      }
      vkCmdDraw(cmd, 3, 1, far_vertex ? (1u << 21) : 0, 0);
      vkCmdEndRenderPass(cmd);
      VkImageMemoryBarrier barrier = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = image,
         .subresourceRange = vi.subresourceRange};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                           1, &barrier);
      VkBufferImageCopy copy = {
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {W, H, 1}};
      vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             buffer, 1, &copy);
      VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0,
                           NULL);
      CHECK(vkEndCommandBuffer(cmd));
      VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &cmd};
      CHECK(vkQueueSubmit(queue, 1, &submit, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull));
      for (unsigned i = 0; i < BYTES; ++i) {
         unsigned x = (i / 4) % W, y = (i / 4) / W;
         bool inside = !sample_mask && x >= 3 && x < W - 4 && y >= 2 && y < H - 3;
         const uint8_t draw_color[4] = {far_vertex ? 0 : 255, bounds ? 0 : 64,
                                        bounds ? 0 : 128, 255};
         uint8_t wanted = inside ? draw_color[i % 4] : expected[i % 4];
         if (abs((int)pixels[i] - wanted) > 1) {
            fprintf(stderr, "pixel %u channel %u: got %u expected %u\n", i / 4,
                    i % 4, pixels[i], wanted);
            return 1;
         }
      }
      for (unsigned i = BYTES; i < BYTES + GUARD; ++i)
         if (pixels[i] != 0xa5)
            return 1;
      printf("PASS draw/scissor/readback %u (%u pixels, guards intact)\n", pass,
             PIXELS);
   }
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   for (unsigned i = 0; i < 2; ++i)
      vkDestroyShaderModule(dev, modules[i], NULL);
   vkDestroyRenderPass(dev, render, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkUnmapMemory(dev, memory[1]);
   vkDestroyBuffer(dev, buffer, NULL);
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, memory[1], NULL);
   vkFreeMemory(dev, memory[0], NULL);
   vkDestroyBuffer(dev, vertex_buffer, NULL);
   vkFreeMemory(dev, vertex_memory, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
