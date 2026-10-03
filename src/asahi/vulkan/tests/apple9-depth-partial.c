/* SPDX-License-Identifier: MIT
 * Depth across tiler partial renders: NTRIS instanced full-screen triangles,
 * the first tenth near (green), the rest far (red), depth LESS. Any red pixel
 * means depth did not survive a mid-pass flush. ZSTORE=1 stores depth.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      VkResult r = (x);                                                        \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d (line %d)\n", #x, r, __LINE__);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static double
now_ms(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static uint32_t
find_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
          VkMemoryPropertyFlags want)
{
   for (uint32_t t = 0; t < mp->memoryTypeCount; ++t)
      if ((bits & (1u << t)) && (mp->memoryTypes[t].propertyFlags & want) == want)
         return t;
   fprintf(stderr, "no memory type\n");
   exit(1);
}

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   FILE *file = fopen(path, "rb");
   if (!file) {
      fprintf(stderr, "cannot open %s\n", path);
      exit(1);
   }
   fseek(file, 0, SEEK_END);
   size_t size = ftell(file);
   rewind(file);
   uint32_t *code = malloc(size);
   if (fread(code, 1, size, file) != size)
      exit(1);
   fclose(file);
   VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = size, .pCode = code};
   VkShaderModule module;
   CHECK(vkCreateShaderModule(dev, &mi, NULL, &module));
   free(code);
   return module;
}

int
main(int argc, char **argv)
{
   if (argc != 3) {
      fprintf(stderr, "Usage: %s vertex.spv fragment.spv\n", argv[0]);
      return 1;
   }
   const uint32_t ntris = getenv("NTRIS") ? atoi(getenv("NTRIS")) : 200000;
   const bool zstore = getenv("ZSTORE") && atoi(getenv("ZSTORE"));
   const uint32_t near_count = ntris / 10;
   enum { W = 1024, H = 1024, BYTES = W * H * 4 };

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "apple9-depth-partial",
                            .apiVersion = VK_API_VERSION_1_0};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t n = 1;
   VkPhysicalDevice phy;
   vkEnumeratePhysicalDevices(instance, &n, &phy);
   if (!n)
      return 1;
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(phy, &mp);
   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(phy, &nq, NULL);
   VkQueueFamilyProperties *qp = calloc(nq, sizeof(*qp));
   vkGetPhysicalDeviceQueueFamilyProperties(phy, &nq, qp);
   uint32_t qi = 0;
   while (qi < nq && !(qp[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      ++qi;
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = qi, .queueCount = 1,
                                  .pQueuePriorities = &priority};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   VkDevice dev;
   CHECK(vkCreateDevice(phy, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qi, 0, &queue);

   const VkFormat formats[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT};
   VkImage images[2];
   VkImageView views[2];
   for (unsigned i = 0; i < 2; ++i) {
      VkImageCreateInfo ii = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
         .format = formats[i], .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = i ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                    : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
      CHECK(vkCreateImage(dev, &ii, NULL, &images[i]));
      VkMemoryRequirements mr;
      vkGetImageMemoryRequirements(dev, images[i], &mr);
      VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = mr.size,
                                 .memoryTypeIndex = find_type(&mp, mr.memoryTypeBits, 0)};
      VkDeviceMemory mem;
      CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
      CHECK(vkBindImageMemory(dev, images[i], mem, 0));
      VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = images[i],
         .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = formats[i],
         .subresourceRange = {i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                              0, 1, 0, 1}};
      CHECK(vkCreateImageView(dev, &vi, NULL, &views[i]));
   }
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = BYTES, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer;
   CHECK(vkCreateBuffer(dev, &bi, NULL, &buffer));
   VkMemoryRequirements br;
   vkGetBufferMemoryRequirements(dev, buffer, &br);
   VkMemoryAllocateInfo bai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = br.size,
      .memoryTypeIndex = find_type(&mp, br.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, buffer, bmem, 0));
   uint8_t *pixels;
   CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&pixels));

   VkAttachmentDescription ad[2] = {
      {.format = formats[0], .samples = VK_SAMPLE_COUNT_1_BIT,
       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
       .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
       .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
       .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
      {.format = formats[1], .samples = VK_SAMPLE_COUNT_1_BIT,
       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = zstore ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
       .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
       .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
   VkAttachmentReference cref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkAttachmentReference dref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
   VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                               .colorAttachmentCount = 1, .pColorAttachments = &cref,
                               .pDepthStencilAttachment = &dref};
   VkRenderPassCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                .attachmentCount = 2, .pAttachments = ad,
                                .subpassCount = 1, .pSubpasses = &sub};
   VkRenderPass render;
   CHECK(vkCreateRenderPass(dev, &ri, NULL, &render));

   VkPushConstantRange range = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
   VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                    .pushConstantRangeCount = 1,
                                    .pPushConstantRanges = &range};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &li, NULL, &layout));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = load_module(dev, argv[1]),
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = load_module(dev, argv[2]),
       .pName = "main"}};
   VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport viewport = {0, 0, W, H, 0, 1};
   VkRect2D scissor = {{0, 0}, {W, H}};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1,
      .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ca = {.colorWriteMask = 15};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &ca};
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS};
   VkGraphicsPipelineCreateInfo graphics = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
      .pStages = stages, .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pColorBlendState = &cb, .pDepthStencilState = &ds, .layout = layout,
      .renderPass = render};
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &graphics, NULL, &pipeline));
   VkFramebufferCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                 .renderPass = render, .attachmentCount = 2,
                                 .pAttachments = views, .width = W, .height = H,
                                 .layers = 1};
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fi, NULL, &fb));
   VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                 .queueFamilyIndex = qi};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));

   memset(pixels, 0xa5, BYTES);
   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   VkClearValue clears[2] = {{.color = {{0, 0, 1, 1}}}, {.depthStencil = {1, 0}}};
   VkRenderPassBeginInfo rb = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                               .renderPass = render, .framebuffer = fb,
                               .renderArea = {{0, 0}, {W, H}},
                               .clearValueCount = 2, .pClearValues = clears};
   vkCmdBeginRenderPass(cmd, &rb, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &near_count);
   vkCmdDraw(cmd, 3, ntris, 0, 0);
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                             .imageExtent = {W, H, 1}};
   vkCmdCopyImageToBuffer(cmd, images[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
                          1, &copy);
   VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1, .pCommandBuffers = &cmd};
   double t0 = now_ms();
   CHECK(vkQueueSubmit(queue, 1, &submit, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 20000000000ull));
   double t1 = now_ms();

   unsigned green = 0, red = 0, blue = 0, other = 0;
   for (unsigned i = 0; i < W * H; ++i) {
      const uint8_t *p = pixels + 4 * i;
      if (p[0] < 8 && p[1] > 247 && p[2] < 8)
         ++green;
      else if (p[0] > 247 && p[1] < 8 && p[2] < 8)
         ++red;
      else if (p[0] < 8 && p[1] < 8 && p[2] > 247)
         ++blue;
      else
         ++other;
   }
   printf("ntris=%u near=%u zstore=%d time=%.1f ms green=%u red=%u blue=%u other=%u %s\n",
          ntris, near_count, zstore, t1 - t0, green, red, blue, other,
          green == W * H ? "PASS" : "FAIL");
   return green == W * H ? 0 : 2;
}
