/* SPDX-License-Identifier: MIT
 * Reproduce the Apple9 libagx compressed 4x-MSAA fast-clear compilation path.
 * Does not use a window system, shaders, Steam or a game.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult result = (call);                                                \
      if (result != VK_SUCCESS) {                                              \
         fprintf(stderr, "%s = %d at %d\n", #call, result, __LINE__);          \
         return 1;                                                             \
      }                                                                        \
   } while (0)
int
main(void)
{
   setbuf(stdout, NULL);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "Apple9 compressed fast-clear",
                            .apiVersion = VK_API_VERSION_1_0};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   if (!count)
      return 4;
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));
   VkPhysicalDevice physical = devices[0];
   free(devices);
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(physical, &props);
   printf("pointer_bits=%zu renderer=%s\n", sizeof(void *) * 8,
          props.deviceName);
   if (!strstr(props.deviceName, "Apple M4")) {
      fprintf(stderr, "Expected Apple M4\n");
      return 4;
   }
   VkImageFormatProperties format;
   CHECK(vkGetPhysicalDeviceImageFormatProperties(
      physical, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_TYPE_2D,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0,
      &format));
   if (!(format.sampleCounts & VK_SAMPLE_COUNT_4_BIT)) {
      fprintf(stderr, "No 4x MSAA support\n");
      return 4;
   }
   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, NULL);
   VkQueueFamilyProperties *families = calloc(nq, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, families);
   uint32_t family = 0;
   while (family < nq && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      family++;
   free(families);
   if (family == nq)
      return 4;
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                           .imageType = VK_IMAGE_TYPE_2D,
                           .format = VK_FORMAT_B8G8R8A8_UNORM,
                           .extent = {128, 128, 1},
                           .mipLevels = 1,
                           .arrayLayers = 1,
                           .samples = VK_SAMPLE_COUNT_4_BIT,
                           .tiling = VK_IMAGE_TILING_OPTIMAL,
                           .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT};
   VkImage image;
   CHECK(vkCreateImage(device, &ii, NULL, &image));
   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, image, &requirements);
   VkPhysicalDeviceMemoryProperties memory_props;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_props);
   uint32_t type = 0;
   while (type < memory_props.memoryTypeCount &&
          !(requirements.memoryTypeBits & (1u << type)))
      type++;
   if (type == memory_props.memoryTypeCount)
      return 4;
   VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = type};
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(device, &allocation, NULL, &memory));
   CHECK(vkBindImageMemory(device, image, memory, 0));
   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
   VkCommandBuffer command;
   CHECK(vkAllocateCommandBuffers(device, &cai, &command));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(command, &begin));
   VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .levelCount = 1,
                                    .layerCount = 1};
   VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = range};
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                        &barrier);
   VkClearColorValue clear = {.float32 = {0.25, 0.5, 0.75, 1}};
   puts(
      "clearing compressed B8G8R8A8, four samples: libagx_fast_clear variant12");
   vkCmdClearColorImage(command, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1,
                        &range);
   CHECK(vkEndCommandBuffer(command));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &command};
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   puts(
      "PASS: fast-clear command compiled, submitted and completed (no pixel-readback claim)");
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
