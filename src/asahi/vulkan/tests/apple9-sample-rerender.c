/* SPDX-License-Identifier: MIT
 * Sampling a re-rendered image: render a hashed pattern into A, sample A into
 * B, render a new pattern into A, sample A into C. C must equal the new
 * pattern; wrong pixels that equal the previous pattern are stale texture
 * reads. Environment: WIDTH, HEIGHT, ROUNDS, FORMAT=rgba8|rgba16f, NTRIS (instanced
 * full-screen triangles per pattern draw, to force partial renders), LOADOP=load
 * (continue A with LOAD_OP_LOAD after the first round).
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define MIN2(a, b) ((a) < (b) ? (a) : (b))

#define CHECK(x)                                                               \
   do {                                                                        \
      VkResult r = (x);                                                        \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d (line %d)\n", #x, r, __LINE__);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

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

static uint32_t
pattern(uint32_t x, uint32_t y, uint32_t seed)
{
   uint32_t h = (x * 73856093u) ^ (y * 19349663u) ^ (seed * 83492791u);
   h ^= h >> 13;
   h *= 0x5bd1e995u;
   h ^= h >> 15;
   return h;
}

static VkImage
make_image(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp, VkFormat format,
           uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImageView *view)
{
   VkImageCreateInfo ii = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = format, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage};
   VkImage image;
   CHECK(vkCreateImage(dev, &ii, NULL, &image));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, image, &mr);
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = mr.size,
                              .memoryTypeIndex = find_type(mp, mr.memoryTypeBits, 0)};
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
   CHECK(vkBindImageMemory(dev, image, mem, 0));
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   CHECK(vkCreateImageView(dev, &vi, NULL, view));
   return image;
}

static VkPipeline
make_pipeline(VkDevice dev, VkShaderModule vs, VkShaderModule fs, VkPipelineLayout layout,
              VkFormat color, uint32_t w, uint32_t h, uint32_t ncolor, bool with_depth)
{
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"}};
   VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport viewport = {0, 0, w, h, 0, 1};
   VkRect2D scissor = {{0, 0}, {w, h}};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ca[3] = {{.colorWriteMask = 15}, {.colorWriteMask = 15},
                                               {.colorWriteMask = 15}};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = ncolor, .pAttachments = ca};
   VkFormat colors[3] = {color, color, color};
   VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = ncolor, .pColorAttachmentFormats = colors,
      .depthAttachmentFormat = with_depth ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_UNDEFINED,
      .stencilAttachmentFormat = with_depth ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_UNDEFINED};
   VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = with_depth, .depthWriteEnable = with_depth,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL};
   VkGraphicsPipelineCreateInfo graphics = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cb, .pDepthStencilState = &dss,
      .layout = layout};
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &graphics, NULL, &pipeline));
   return pipeline;
}

static void
render(VkCommandBuffer cmd, const VkImageView *views, uint32_t count, VkImageView depth_view,
       uint32_t w, uint32_t h, bool clear)
{
   VkRenderingAttachmentInfo color[3];
   for (uint32_t i = 0; i < count; ++i) {
      color[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = views[i],
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = {.color = {{0, 0, 0, 1}}}};
   }
   VkRenderingAttachmentInfo depth = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = depth_view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = {.depthStencil = {1, 0}}};
   VkRenderingInfo info = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                           .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                           .colorAttachmentCount = count, .pColorAttachments = color,
                           .pDepthAttachment = depth_view ? &depth : NULL,
                           .pStencilAttachment = depth_view ? &depth : NULL};
   vkCmdBeginRendering(cmd, &info);
}

static void
barrier(VkCommandBuffer cmd, VkImage image, VkAccessFlags src, VkAccessFlags dst,
        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
   VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                             .srcAccessMask = src, .dstAccessMask = dst,
                             .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                             .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
                             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

int
main(int argc, char **argv)
{
   if (argc != 4) {
      fprintf(stderr, "Usage: %s fullscreen.vert.spv pattern.frag.spv sample.frag.spv\n",
              argv[0]);
      return 1;
   }
   const uint32_t W = getenv("WIDTH") ? atoi(getenv("WIDTH")) : 1280;
   const uint32_t H = getenv("HEIGHT") ? atoi(getenv("HEIGHT")) : 720;
   const uint32_t rounds = getenv("ROUNDS") ? atoi(getenv("ROUNDS")) : 2;
   const bool rgba8 = getenv("FORMAT") && !strcmp(getenv("FORMAT"), "rgba8");
   const uint32_t ntris = getenv("NTRIS") ? atoi(getenv("NTRIS")) : 1;
   const bool loadop = getenv("LOADOP") && !strcmp(getenv("LOADOP"), "load");
   const uint32_t mrt = getenv("MRT") ? atoi(getenv("MRT")) : 1;
   const bool depth = getenv("DEPTH") && atoi(getenv("DEPTH"));
   const uint32_t att = getenv("ATT") ? atoi(getenv("ATT")) : 0;
   /* Instances below this count cover only the left part of the screen. */
   const uint32_t partial_count = getenv("PARTIAL") ? atoi(getenv("PARTIAL")) : 0;
   const VkFormat fa = rgba8 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R16G16B16A16_SFLOAT;
   const VkFormat fc = VK_FORMAT_R8G8B8A8_UNORM;
   const size_t bytes = (size_t)W * H * 4;

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "apple9-sample-rerender",
                            .apiVersion = VK_API_VERSION_1_3};
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
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .dynamicRendering = VK_TRUE};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f13,
                             .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   VkDevice dev;
   CHECK(vkCreateDevice(phy, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qi, 0, &queue);

   VkImageView va[3], vb, vc, vd = VK_NULL_HANDLE;
   VkImage a[3];
   for (uint32_t i = 0; i < mrt; ++i)
      a[i] = make_image(dev, &mp, fa, W, H,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &va[i]);
   if (depth) {
      VkImageCreateInfo di = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
         .format = VK_FORMAT_D32_SFLOAT_S8_UINT, .extent = {W, H, 1}, .mipLevels = 1,
         .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT};
      VkImage dimg;
      CHECK(vkCreateImage(dev, &di, NULL, &dimg));
      VkMemoryRequirements mr;
      vkGetImageMemoryRequirements(dev, dimg, &mr);
      VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = mr.size,
                                 .memoryTypeIndex = find_type(&mp, mr.memoryTypeBits, 0)};
      VkDeviceMemory mem;
      CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
      CHECK(vkBindImageMemory(dev, dimg, mem, 0));
      VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = dimg,
         .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_D32_SFLOAT_S8_UINT,
         .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}};
      CHECK(vkCreateImageView(dev, &vi, NULL, &vd));
      VkImageMemoryBarrier init = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = dimg,
                                   .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}};
      (void)init;
   }
   VkImage b = make_image(dev, &mp, fc, W, H,
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &vb);
   VkImage c = make_image(dev, &mp, fc, W, H,
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &vc);

   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes * 2,
                            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
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
   memset(pixels, 0xa5, bytes * 2);

   VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                             .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
                             .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                             .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                             .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &si, NULL, &sampler));
   VkDescriptorSetLayoutBinding binding = {.binding = 0,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                           .descriptorCount = 1,
                                           .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
   VkDescriptorSetLayoutCreateInfo sli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
      .pBindings = &binding};
   VkDescriptorSetLayout sl;
   CHECK(vkCreateDescriptorSetLayout(dev, &sli, NULL, &sl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
   VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                       .descriptorPool = dp, .descriptorSetCount = 1,
                                       .pSetLayouts = &sl};
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkDescriptorImageInfo dii = {sampler, va[att], VK_IMAGE_LAYOUT_GENERAL};
   VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
                                 .dstBinding = 0, .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 .pImageInfo = &dii};
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

   VkPushConstantRange range = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8};
   VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
   VkPipelineLayout pattern_layout;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &pattern_layout));
   VkPipelineLayoutCreateInfo sli2 = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                      .setLayoutCount = 1, .pSetLayouts = &sl,
                                      .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
   VkPipelineLayout sample_layout;
   CHECK(vkCreatePipelineLayout(dev, &sli2, NULL, &sample_layout));
   VkShaderModule vs = load_module(dev, argv[1]);
   VkPipeline pattern_pipe =
      make_pipeline(dev, vs, load_module(dev, argv[2]), pattern_layout, fa, W, H, mrt, depth);
   VkPipeline sample_pipe =
      make_pipeline(dev, vs, load_module(dev, argv[3]), sample_layout, fc, W, H, 1, false);

   VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                 .queueFamilyIndex = qi};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                      .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));

   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   /* Initial layouts: everything to GENERAL. */
   VkImage all[5] = {a[0], mrt > 1 ? a[1] : a[0], mrt > 2 ? a[2] : a[0], b, c};
   for (unsigned i = 0; i < 5; ++i) {
      VkImageMemoryBarrier init = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = all[i],
                                   .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL,
                           1, &init);
   }
   for (uint32_t round = 1; round <= rounds; ++round) {
      VkImageView target = round == rounds ? vc : vb;
      VkImage target_image = round == rounds ? c : b;
      render(cmd, va, mrt, vd, W, H, round == 1 || !loadop);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pattern_pipe);
      uint32_t push[2] = {round, partial_count};
      vkCmdPushConstants(cmd, pattern_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8, push);
      vkCmdDraw(cmd, 3, ntris, 0, 0);
      vkCmdEndRendering(cmd);
      for (uint32_t i = 0; i < mrt; ++i)
         barrier(cmd, a[i], VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
      render(cmd, &target, 1, VK_NULL_HANDLE, W, H, true);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sample_pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sample_layout, 0, 1, &ds,
                              0, NULL);
      uint32_t zero2[2] = {0, 0};
      vkCmdPushConstants(cmd, sample_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8, zero2);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRendering(cmd);
      barrier(cmd, target_image, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
              VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT);
      for (uint32_t i = 0; i < mrt; ++i)
         barrier(cmd, a[i], VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
   }
   VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                             .imageExtent = {W, H, 1}};
   vkCmdCopyImageToBuffer(cmd, c, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
   copy.bufferOffset = bytes;
   vkCmdCopyImageToBuffer(cmd, b, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
   VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                        &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                          .pCommandBuffers = &cmd};
   CHECK(vkQueueSubmit(queue, 1, &submit, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 20000000000ull));

   unsigned wrong = 0, stale = 0, zero = 0, early = 0, wrong_blocks = 0, wrong_b = 0;
   const uint32_t last = rounds, prev = rounds - 1;
   for (uint32_t by = 0; by < H / 4; ++by) {
      for (uint32_t bx = 0; bx < W / 8; ++bx) {
         bool block_wrong = false;
         for (uint32_t y = by * 4; y < by * 4 + 4; ++y) {
            for (uint32_t x = bx * 8; x < bx * 8 + 8; ++x) {
               const uint8_t *p = pixels + 4 * (y * W + x);
               uint32_t got = p[0] | (p[1] << 8) | (p[2] << 16);
               const uint32_t off = att * 1000000u;
               uint32_t want = pattern(x, y, last * 65536u + ntris - 1 + off) & 0xffffff;
               if (got != want) {
                  ++wrong;
                  block_wrong = true;
                  if (!got)
                     ++zero;
                  else if (prev && got == (pattern(x, y, prev * 65536u + ntris - 1 + off) & 0xffffff))
                     ++stale;
                  else {
                     /* An earlier instance's pattern: a partial-render snapshot. */
                     for (uint32_t k = ntris - 1; k-- > ntris - MIN2(ntris, 4096);) {
                        if (got == (pattern(x, y, last * 65536u + k + off) & 0xffffff)) {
                           ++early;
                           break;
                        }
                     }
                  }
               }
               const uint8_t *q = pixels + bytes + 4 * (y * W + x);
               uint32_t got_b = q[0] | (q[1] << 8) | (q[2] << 16);
               if (prev && got_b != (pattern(x, y, prev * 65536u + ntris - 1 + off) & 0xffffff))
                  ++wrong_b;
            }
         }
         wrong_blocks += block_wrong;
      }
   }
   printf("%ux%u %s mrt=%u depth=%d att=%u rounds=%u ntris=%u partial=%u load=%d: C wrong=%u (%.2f%%) zero=%u prev-round=%u earlier-instance=%u blocks=%u/%u; B wrong=%u %s\n",
          W, H, rgba8 ? "rgba8" : "rgba16f", mrt, depth, att, rounds, ntris, partial_count, loadop, wrong, 100.0 * wrong / (W * H), zero, stale, early,
          wrong_blocks, (W / 8) * (H / 4), wrong_b, wrong || wrong_b ? "FAIL" : "PASS");
   return wrong || wrong_b ? 2 : 0;
}
