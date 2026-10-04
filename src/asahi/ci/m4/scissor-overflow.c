/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Offscreen Vulkan regression for bounded scissor/depth-bias tables.
 * scissor: 65,537 distinct four-viewport groups in one rendering instance.
 * bias: 65,537 distinct bias records with a fixed scissor; the final draw must
 * fail the depth test, exposing a wrapped bias index. No implementation knobs.
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "stress_vert.h"
#include "stress_frag.h"

#define CHECK(expr) do { \
   VkResult result = (expr); \
   if (result != VK_SUCCESS) { \
      fprintf(stderr, "%s: %d at line %d\n", #expr, result, __LINE__); \
      exit(1); \
   } \
} while (0)

static VkPhysicalDeviceMemoryProperties memory;
static VkDevice device;

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (memory.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }
   fprintf(stderr, "No suitable memory type\n");
   exit(1);
}

struct image {
   VkImage image;
   VkImageView view;
   VkDeviceMemory memory;
};

static struct image
image_create(uint32_t width, uint32_t height, VkFormat format,
             VkImageAspectFlags aspect, VkImageUsageFlags usage)
{
   struct image image;
   VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
   };
   CHECK(vkCreateImage(device, &info, NULL, &image.image));
   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, image.image, &requirements);
   VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   CHECK(vkAllocateMemory(device, &allocation, NULL, &image.memory));
   CHECK(vkBindImageMemory(device, image.image, image.memory, 0));
   VkImageViewCreateInfo view = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image.image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = {aspect, 0, 1, 0, 1},
   };
   CHECK(vkCreateImageView(device, &view, NULL, &image.view));
   return image;
}

static void
image_destroy(struct image image)
{
   vkDestroyImageView(device, image.view, NULL);
   vkDestroyImage(device, image.image, NULL);
   vkFreeMemory(device, image.memory, NULL);
}

static VkShaderModule
shader(const uint32_t *words, size_t size)
{
   VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = words,
   };
   VkShaderModule module;
   CHECK(vkCreateShaderModule(device, &info, NULL, &module));
   return module;
}

int
main(int argc, char **argv)
{
   if (argc != 2 || (strcmp(argv[1], "scissor") && strcmp(argv[1], "bias"))) {
      fprintf(stderr, "Usage: %s scissor|bias\n", argv[0]);
      return 2;
   }
   const int bias = !strcmp(argv[1], "bias");
   const uint32_t draws = 65537, viewports = bias ? 1 : 4;
   const uint32_t width = bias ? 4 : 512;
   const uint32_t height = bias ? 4 : (draws * viewports + width - 1) / width;
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "scissor-overflow", .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   if (!count) return 77;
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));
   VkPhysicalDevice physical = devices[0];
   free(devices);
   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(physical, &properties);
   fprintf(stderr, "Renderer: %s\n", properties.deviceName);
   vkGetPhysicalDeviceMemoryProperties(physical, &memory);

   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
   };
   VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &f13,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!features.features.multiViewport || !f12.shaderOutputViewportIndex ||
       !f13.dynamicRendering) {
      fprintf(stderr, "Required viewport/dynamic rendering features unavailable\n");
      return 77;
   }
   features.features = (VkPhysicalDeviceFeatures){.multiViewport = VK_TRUE};
   f12 = (VkPhysicalDeviceVulkan12Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &f13, .shaderOutputViewportIndex = VK_TRUE,
   };
   f13 = (VkPhysicalDeviceVulkan13Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE,
   };
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
   VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
   uint32_t family = 0;
   while (family < count && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      ++family;
   free(families);
   assert(family < count);
   float priority = 1;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
   };
   CHECK(vkCreateDevice(physical, &device_info, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   struct image color = image_create(width, height, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   struct image depth = image_create(width, height, VK_FORMAT_D32_SFLOAT,
      VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
   VkShaderModule vert = shader(stress_vert, sizeof(stress_vert));
   VkShaderModule frag = shader(stress_frag, sizeof(stress_frag));
   VkPipelineShaderStageCreateInfo stages[] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main"},
   };
   struct { float depth; uint32_t red; } push = {.depth = .25f};
   VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .size = sizeof(push),
   };
   VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range,
   };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));
   VkPipelineVertexInputStateCreateInfo vertex = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkPipelineViewportStateCreateInfo viewport = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = viewports, .scissorCount = viewports,
   };
   VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1,
      .depthBiasEnable = bias,
   };
   VkPipelineMultisampleStateCreateInfo samples = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineDepthStencilStateCreateInfo depth_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = !bias,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   VkPipelineColorBlendAttachmentState attachment = {.colorWriteMask = 15};
   VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &attachment,
   };
   const VkDynamicState states[] = {
      VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS,
   };
   VkPipelineDynamicStateCreateInfo dynamic = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 3, .pDynamicStates = states,
   };
   VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &format,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
   };
   VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vertex,
      .pInputAssemblyState = &assembly, .pViewportState = &viewport,
      .pRasterizationState = &raster, .pMultisampleState = &samples,
      .pDepthStencilState = &depth_state, .pColorBlendState = &blend,
      .pDynamicState = &dynamic, .layout = layout,
   };
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                  NULL, &pipeline));
   VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = width * height * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer buffer;
   CHECK(vkCreateBuffer(device, &buffer_info, NULL, &buffer));
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, buffer, &requirements);
   VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory readback_memory;
   CHECK(vkAllocateMemory(device, &allocation, NULL, &readback_memory));
   CHECK(vkBindBufferMemory(device, buffer, readback_memory, 0));
   uint8_t *pixels;
   CHECK(vkMapMemory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, (void **)&pixels));
   VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family,
   };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
   VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(device, &command_info, &cmd));
   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   VkImageMemoryBarrier barriers[2] = {
      {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
       .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
       .image = color.image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
      {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
       .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
       .image = depth.image, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}},
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 0, NULL, 0, NULL, 2, barriers);
   VkRenderingAttachmentInfo color_attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {0, 0, 1, 1},
   };
   VkRenderingAttachmentInfo depth_attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = depth.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .clearValue.depthStencil.depth = bias ? .4f : 1.f,
   };
   VkRenderingInfo render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {width, height}},
      .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &color_attachment, .pDepthAttachment = &depth_attachment,
   };
   vkCmdBeginRendering(cmd, &render);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   VkViewport vps[4];
   for (uint32_t j = 0; j < viewports; ++j)
      vps[j] = (VkViewport){0, 0, width, height, 0, 1};
   vkCmdSetViewport(cmd, 0, viewports, vps);
   for (uint32_t i = 0; i < draws; ++i) {
      VkRect2D scissors[4];
      for (uint32_t j = 0; j < viewports; ++j) {
         uint32_t pixel = bias ? j : i * viewports + j;
         scissors[j] = (VkRect2D){{pixel % width, pixel / width}, {1, 1}};
      }
      vkCmdSetScissor(cmd, 0, viewports, scissors);
      vkCmdSetDepthBias(cmd, bias ? i * 128.f : 0.f, 0, 0);
      push.red = bias && i == draws - 1;
      vkCmdPushConstants(cmd, layout, push_range.stageFlags, 0, sizeof(push), &push);
      vkCmdDraw(cmd, 3, viewports, 0, 0);
   }
   if (!bias) {
      /* A later triangle must fail against depth written before every split.
       * Untouched pixels turn red, while all stress-draw pixels stay green.
       */
      VkRect2D full[4];
      for (uint32_t j = 0; j < viewports; ++j)
         full[j] = (VkRect2D){{0, 0}, {width, height}};
      vkCmdSetScissor(cmd, 0, viewports, full);
      push.depth = .75f; push.red = 1;
      vkCmdPushConstants(cmd, layout, push_range.stageFlags, 0, sizeof(push), &push);
      vkCmdDraw(cmd, 3, viewports, 0, 0);
   }
   vkCmdEndRendering(cmd);
   barriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
   barriers[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, barriers);
   VkBufferImageCopy copy = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {width, height, 1},
   };
   vkCmdCopyImageToBuffer(cmd, color.image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
   VkMemoryBarrier host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   };
   VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(device, &fence_info, NULL, &fence));
   CHECK(vkQueueSubmit(queue, 1, &submit, fence));
   CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 120000000000ull));
   unsigned mismatches = 0;
   for (uint32_t i = 0; i < width * height; ++i) {
      const uint8_t green[4] = {0, 255, 0, 255};
      const uint8_t other[4] = {bias ? 0 : 255, 0, bias ? 255 : 0, 255};
      const uint8_t *expected = i < (bias ? 1 : draws * viewports) ? green : other;
      if (memcmp(pixels + 4 * i, expected, 4)) {
         if (mismatches < 5)
            fprintf(stderr, "pixel %u,%u: got %u,%u,%u,%u; expected %u,%u,%u,%u\n",
               i % width, i / width, pixels[4*i], pixels[4*i+1], pixels[4*i+2], pixels[4*i+3],
               expected[0], expected[1], expected[2], expected[3]);
         ++mismatches;
      }
   }
   printf("{\"test\":\"%s\",\"draws\":%u,\"viewports\":%u,\"pixels\":%u,\"mismatches\":%u,\"result\":\"%s\"}\n",
      argv[1], draws, viewports, width*height, mismatches, mismatches ? "FAIL" : "PASS");
   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, pool, NULL);
   vkUnmapMemory(device, readback_memory);
   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, readback_memory, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, vert, NULL);
   vkDestroyShaderModule(device, frag, NULL);
   image_destroy(color); image_destroy(depth);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return mismatches ? 1 : 0;
}
