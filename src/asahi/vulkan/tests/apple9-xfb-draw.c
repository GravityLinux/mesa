/* SPDX-License-Identifier: MIT
 * Native Vulkan byte-count draw regression with guarded pixel readback.
 * Compile the adjacent apple9-xfb-draw.{vert,frag} with glslangValidator -V.
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
   if (argc != 3) { fprintf(stderr, "usage: %s xfb.vert.spv xfb.frag.spv\n", argv[0]); return 2; }
   const bool bounds = false, far_vertex = false;
   setbuf(stdout, NULL);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "Apple9 offscreen draw",
                            .apiVersion = VK_API_VERSION_1_1};
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
   VkPhysicalDeviceTransformFeedbackPropertiesEXT xfb = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT};
   VkPhysicalDeviceProperties2 p2 = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&xfb};
   vkGetPhysicalDeviceProperties2(phy, &p2);
   if (!xfb.transformFeedbackDraw) { puts("SKIP transformFeedbackDraw=false"); return 77; }
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
   VkPhysicalDeviceFeatures features = {.drawIndirectFirstInstance = true};
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xf = {.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,.transformFeedback=VK_TRUE};
   const char *extension = VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .pEnabledFeatures = &features, .pNext=&xf, .enabledExtensionCount=1, .ppEnabledExtensionNames=&extension};
   PFN_vkCmdDrawIndirectByteCountEXT draw_xfb;
   VkDevice dev;
   CHECK(vkCreateDevice(phy, &dci, NULL, &dev));
   draw_xfb=(PFN_vkCmdDrawIndirectByteCountEXT)vkGetDeviceProcAddr(dev,"vkCmdDrawIndirectByteCountEXT");
   if (!draw_xfb) return 2;
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
   VkBuffer counter; VkDeviceMemory counter_memory;
   VkBufferCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=64,
      .usage=VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT};
   CHECK(vkCreateBuffer(dev,&ci,NULL,&counter));
   VkMemoryRequirements cr; vkGetBufferMemoryRequirements(dev,counter,&cr);
   unsigned ct=0; while (!(cr.memoryTypeBits & (1u << ct))) ct++;
   CHECK(vkAllocateMemory(dev,&(VkMemoryAllocateInfo){.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=cr.size,.memoryTypeIndex=ct},NULL,&counter_memory));
   CHECK(vkBindBufferMemory(dev,counter,counter_memory,0));
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
   const struct { uint32_t bytes, offset, stride, instances, first; bool draw; } tests[] = {
      {0,12,20,1,0,false}, {11,12,20,1,0,false}, {12,12,20,1,0,false},
      {71,12,20,1,0,false}, {72,12,20,1,0,true}, {79,12,20,1,0,true},
      {132,12,20,1,0,true}, {79,12,20,0,0,false}, {79,12,20,1,3,true},
      {UINT32_MAX-1,UINT32_MAX-63,20,1,0,true},
      {UINT32_MAX-64,UINT32_MAX-63,20,1,0,false},
   };
   for (unsigned pass = 0; pass < sizeof(tests)/sizeof(tests[0]); ++pass) {
      const uint8_t expected[4] = {32, 128, 223, 255};
      memset(pixels, 0xa5, BYTES + GUARD);
      CHECK(vkResetCommandBuffer(cmd, 0));
      CHECK(vkResetFences(dev, 1, &fence));
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(cmd, &begin));
      vkCmdUpdateBuffer(cmd,counter,12,sizeof(uint32_t),&tests[pass].bytes);
      VkMemoryBarrier indirect_barrier={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.dstAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,1,&indirect_barrier,0,NULL,0,NULL);
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
      draw_xfb(cmd,tests[pass].instances,tests[pass].first,counter,12,tests[pass].offset,tests[pass].stride);
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
         bool inside = tests[pass].draw && x >= 3 && x < W - 4 && y >= 2 && y < H - 3;
         const uint8_t draw_color[4] = {(tests[pass].first + 1) * 16, 64, 128, 255};
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
   vkDestroyBuffer(dev,counter,NULL);
   vkFreeMemory(dev,counter_memory,NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
