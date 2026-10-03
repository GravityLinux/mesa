/* SPDX-License-Identifier: MIT
 * Native Vulkan graphics pipeline compilation/cache test. No rendering. No
 * driver-private interfaces.
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
   if (argc != 4) {
      fprintf(
         stderr,
         "usage: %s procedural.vert.spv attributes.vert.spv fragment.spv\n",
         argv[0]);
      return 1;
   }
   setbuf(stdout, NULL);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "Apple9 pipeline compile",
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
      if (qp[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)
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
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   VkDevice dev;
   CHECK(vkCreateDevice(phy, &dci, NULL, &dev));
   VkShaderModule modules[3];
   for (unsigned i = 0; i < 3; ++i) {
      FILE *file = fopen(argv[i + 1], "rb");
      if (!file || fseek(file, 0, SEEK_END))
         return 1;
      long size = ftell(file);
      if (size <= 0 || size % 4 || fseek(file, 0, SEEK_SET))
         return 1;
      uint32_t *code = malloc(size);
      if (!code || fread(code, 1, size, file) != (size_t)size)
         return 1;
      fclose(file);
      VkShaderModuleCreateInfo mi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = size,
         .pCode = code};
      CHECK(vkCreateShaderModule(dev, &mi, NULL, &modules[i]));
      free(code);
   }
   VkPushConstantRange push = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .size = 32,
   };
   VkPipelineLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push,
   };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &li, NULL, &layout));
   VkPipelineCache cache;
   VkPipelineCacheCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
   CHECK(vkCreatePipelineCache(dev, &ci, NULL, &cache));
   const VkFormat formats[] = {VK_FORMAT_R8G8B8A8_UNORM,
                               VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_FORMAT_D32_SFLOAT};
   unsigned passed = 0;
   for (unsigned round = 0; round < 2; ++round) {
      for (unsigned f = 0; f < 3; ++f) {
         VkAttachmentDescription attachment = {
            .format = formats[f],
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
         VkAttachmentReference ref = {0,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
         VkSubpassDescription subpass = {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &ref};
         if (f == 2) {
            attachment.finalLayout =
               VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            subpass.colorAttachmentCount = 0;
            subpass.pColorAttachments = NULL;
            subpass.pDepthStencilAttachment = &ref;
         }
         VkRenderPassCreateInfo ri = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass};
         VkRenderPass render;
         CHECK(vkCreateRenderPass(dev, &ri, NULL, &render));
         for (unsigned attributes = 0; attributes < 2; ++attributes) {
            for (unsigned blend = 0; blend < 2; ++blend) {
               VkPipelineShaderStageCreateInfo stages[2] = {
                  {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                   .module = modules[2],
                   .pName = "main"},
                  {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_VERTEX_BIT,
                   .module = modules[attributes],
                   .pName = "main"}};
               if (round) {
                  VkPipelineShaderStageCreateInfo tmp = stages[0];
                  stages[0] = stages[1];
                  stages[1] = tmp;
               }
               VkVertexInputBindingDescription binding = {
                  0, 8, VK_VERTEX_INPUT_RATE_VERTEX};
               VkVertexInputAttributeDescription attr = {
                  0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
               VkPipelineVertexInputStateCreateInfo vi = {
                  .sType =
                     VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                  .vertexBindingDescriptionCount = attributes,
                  .pVertexBindingDescriptions = &binding,
                  .vertexAttributeDescriptionCount = attributes,
                  .pVertexAttributeDescriptions = &attr};
               VkPipelineInputAssemblyStateCreateInfo ia = {
                  .sType =
                     VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                  .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
               VkViewport viewport = {0, 0, 64, 64, 0, 1};
               VkRect2D scissor = {{0, 0}, {64, 64}};
               VkPipelineViewportStateCreateInfo vp = {
                  .sType =
                     VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                  .viewportCount = 1,
                  .pViewports = &viewport,
                  .scissorCount = 1,
                  .pScissors = &scissor};
               VkPipelineRasterizationStateCreateInfo rs = {
                  .sType =
                     VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                  .polygonMode = VK_POLYGON_MODE_FILL,
                  .cullMode = VK_CULL_MODE_NONE,
                  .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                  .lineWidth = 1};
               VkPipelineMultisampleStateCreateInfo ms = {
                  .sType =
                     VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                  .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
               VkPipelineColorBlendAttachmentState ca = {
                  .blendEnable = blend,
                  .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                  .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                  .colorBlendOp = VK_BLEND_OP_ADD,
                  .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                  .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                  .alphaBlendOp = VK_BLEND_OP_ADD,
                  .colorWriteMask = 15};
               VkPipelineColorBlendStateCreateInfo cb = {
                  .sType =
                     VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                  .attachmentCount = 1,
                  .pAttachments = &ca};
               VkPipelineDepthStencilStateCreateInfo ds = {
                  .sType =
                     VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                  .depthTestEnable = f == 2,
                  .depthWriteEnable = f == 2,
                  .depthCompareOp = VK_COMPARE_OP_LESS,
               };
               if (f == 2)
                  cb.attachmentCount = 0;
               VkGraphicsPipelineCreateInfo pi = {
                  .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                  .stageCount = 2,
                  .pStages = stages,
                  .pVertexInputState = &vi,
                  .pInputAssemblyState = &ia,
                  .pViewportState = &vp,
                  .pRasterizationState = &rs,
                  .pMultisampleState = &ms,
                  .pColorBlendState = &cb,
                  .pDepthStencilState = &ds,
                  .layout = layout,
                  .renderPass = render};
               VkPipeline pipeline;
               CHECK(vkCreateGraphicsPipelines(dev, cache, 1, &pi, NULL,
                                               &pipeline));
               vkDestroyPipeline(dev, pipeline, NULL);
               printf("PASS round=%u format=%u attributes=%u blend=%u\n", round,
                      f, attributes, blend);
               ++passed;
            }
         }
         vkDestroyRenderPass(dev, render, NULL);
      }
      /* Recreate the cache from its serialized bytes to test deserialization. */
      size_t size = 0;
      CHECK(vkGetPipelineCacheData(dev, cache, &size, NULL));
      void *data = malloc(size);
      CHECK(vkGetPipelineCacheData(dev, cache, &size, data));
      vkDestroyPipelineCache(dev, cache, NULL);
      ci.initialDataSize = size;
      ci.pInitialData = data;
      CHECK(vkCreatePipelineCache(dev, &ci, NULL, &cache));
      free(data);
   }
   vkDestroyPipelineCache(dev, cache, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   for (unsigned i = 0; i < 3; ++i)
      vkDestroyShaderModule(dev, modules[i], NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf("%u pipeline creations passed; no draws submitted\n", passed);
   return 0;
}
