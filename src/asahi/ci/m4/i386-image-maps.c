/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 *
 * Manual Vulkan regression for image CPU mapping lifetime on 32-bit clients.
 */
#define _FILE_OFFSET_BITS 64
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <vulkan/vulkan.h>

#define MiB (1024u * 1024u)
#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult result = (call);                                                \
      if (result != VK_SUCCESS) {                                              \
         fprintf(stderr, "%s: VkResult %d at line %d\n", #call, result,        \
                 __LINE__);                                                    \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

static VkDevice device;
static VkPhysicalDeviceMemoryProperties memory;

struct image {
   VkImage image;
   VkDeviceMemory memory;
};

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   for (unsigned i = 0; i < memory.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (memory.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }
   fprintf(stderr, "No suitable memory type\n");
   exit(2);
}

static struct image
create_image(unsigned width, VkImageUsageFlags usage)
{
   struct image image;
   VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {width, width, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
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
   return image;
}

/* FEX exposes host mappings, including its own reservations. Count only the
 * guest's low-4-GiB DRM mappings, not the large high-address JIT reservations.
 */
static uint64_t
gpu_mapped_bytes(void)
{
   FILE *file = fopen("/proc/self/maps", "r");
   if (!file) {
      perror("maps");
      exit(2);
   }
   char line[1024];
   uint64_t bytes = 0;
   while (fgets(line, sizeof(line), file)) {
      uint64_t start, end;
      if (sscanf(line, "%" SCNx64 "-%" SCNx64, &start, &end) == 2 &&
          end <= UINT64_C(0x100000000) && strstr(line, "/dev/dri/"))
         bytes += end - start;
   }
   fclose(file);
   return bytes;
}

static VkImageSubresourceLayers
subresource(void)
{
   return (VkImageSubresourceLayers){
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .layerCount = 1,
   };
}

static void
host_copy_check(void)
{
   PFN_vkCopyMemoryToImageEXT to_image =
      (PFN_vkCopyMemoryToImageEXT)vkGetDeviceProcAddr(device,
                                                      "vkCopyMemoryToImageEXT");
   PFN_vkCopyImageToMemoryEXT to_memory =
      (PFN_vkCopyImageToMemoryEXT)vkGetDeviceProcAddr(device,
                                                      "vkCopyImageToMemoryEXT");
   PFN_vkCopyImageToImageEXT copy_image =
      (PFN_vkCopyImageToImageEXT)vkGetDeviceProcAddr(device,
                                                     "vkCopyImageToImageEXT");
   struct image a = create_image(64, VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT);
   struct image b = create_image(64, VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT);
   uint8_t input[64 * 64 * 4], output[sizeof(input)];
   for (unsigned i = 0; i < sizeof(input); ++i)
      input[i] = (uint8_t)(i * 17 + i / 256);

   VkMemoryToImageCopyEXT upload_region = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
      .pHostPointer = input,
      .imageSubresource = subresource(),
      .imageExtent = {64, 64, 1},
   };
   VkCopyMemoryToImageInfoEXT upload = {
      .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
      .dstImage = a.image,
      .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1,
      .pRegions = &upload_region,
   };
   VkImageCopy2 copy_region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = subresource(),
      .dstSubresource = subresource(),
      .extent = {64, 64, 1},
   };
   VkCopyImageToImageInfoEXT copy = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO_EXT,
      .srcImage = a.image,
      .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .dstImage = b.image,
      .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1,
      .pRegions = &copy_region,
   };
   VkImageToMemoryCopyEXT download_region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT,
      .pHostPointer = output,
      .imageSubresource = subresource(),
      .imageExtent = {64, 64, 1},
   };
   VkCopyImageToMemoryInfoEXT download = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT,
      .srcImage = b.image,
      .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1,
      .pRegions = &download_region,
   };
   PFN_vkTransitionImageLayoutEXT transition =
      (PFN_vkTransitionImageLayoutEXT)vkGetDeviceProcAddr(
         device, "vkTransitionImageLayoutEXT");
   VkHostImageLayoutTransitionInfoEXT transitions[2];
   for (unsigned i = 0; i < 2; ++i) {
      transitions[i] = (VkHostImageLayoutTransitionInfoEXT){
         .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
         .image = i ? b.image : a.image,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
   }
   CHECK(transition(device, 2, transitions));
   uint64_t before = gpu_mapped_bytes();
   for (unsigned i = 0; i < 32; ++i) {
      CHECK(to_image(device, &upload));
      CHECK(copy_image(device, &copy));
      CHECK(to_memory(device, &download));
      if (memcmp(input, output, sizeof(input))) {
         fprintf(stderr, "Host image copy readback mismatch\n");
         exit(2);
      }
   }
   if (gpu_mapped_bytes() != before) {
      fprintf(stderr, "Host copy retained CPU mappings\n");
      exit(2);
   }
   vkDestroyImage(device, b.image, NULL);
   vkFreeMemory(device, b.memory, NULL);
   vkDestroyImage(device, a.image, NULL);
   vkFreeMemory(device, a.memory, NULL);
   puts("Host copy: 32 upload/image-copy/download cycles verified");
}

int
main(int argc, char **argv)
{
   unsigned count = argc > 1 ? strtoul(argv[1], NULL, 0) : 150;
   unsigned reserve_mib = argc > 2 ? strtoul(argv[2], NULL, 0) : 3072;
   if (sizeof(void *) != 4 || count == 0 || count > 256 || reserve_mib > 3584) {
      fprintf(stderr,
              "Usage (i386): %s [images 1..256] [reserved MiB <=3584]\n",
              argv[0]);
      return 2;
   }
   setbuf(stdout, NULL);
   VkInstance instance;
   VkApplicationInfo application = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "i386-image-maps",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application,
   };
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t device_count = 1;
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &device_count, &physical));
   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(physical, &properties);
   printf("Device: %s\n", properties.deviceName);
   vkGetPhysicalDeviceMemoryProperties(physical, &memory);
   uint32_t queue_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, NULL);
   VkQueueFamilyProperties *families = calloc(queue_count, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, families);
   unsigned family = 0;
   while (family < queue_count &&
          !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      ++family;
   if (family == queue_count)
      return 2;
   free(families);
   float priority = 1;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *extensions[] = {VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME};
   VkPhysicalDeviceHostImageCopyFeaturesEXT host_copy = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
      .hostImageCopy = VK_TRUE,
   };
   VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &host_copy,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   CHECK(vkCreateDevice(physical, &device_info, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkCommandPool pool;
   VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   CHECK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
   VkCommandBuffer command;
   VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   CHECK(vkAllocateCommandBuffers(device, &command_info, &command));

   void *reservations[56];
   unsigned reserved = 0;
   while ((reserved + 1) * 64 <= reserve_mib) {
      void *map =
         mmap(NULL, 64 * MiB, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (map == MAP_FAILED)
         break;
      reservations[reserved++] = map;
   }
   printf("Reserved VA: %u MiB (no committed RAM)\n", reserved * 64);
   uint64_t before = gpu_mapped_bytes();
   struct image *images = calloc(count, sizeof(*images));
   if (!images)
      return 2;
   for (unsigned i = 0; i < count; ++i)
      images[i] = create_image(1024, VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   uint64_t after = gpu_mapped_bytes();
   printf("Bound %u images; added CPU GPU mappings: %" PRIu64 " bytes\n", count,
          after - before);
   bool retained = after > before + 16 * MiB;

   VkBuffer buffer;
   VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = count * 4,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   CHECK(vkCreateBuffer(device, &buffer_info, NULL, &buffer));
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, buffer, &requirements);
   VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory readback;
   CHECK(vkAllocateMemory(device, &allocation, NULL, &readback));
   CHECK(vkBindBufferMemory(device, buffer, readback, 0));
   uint8_t *pixels;
   CHECK(vkMapMemory(device, readback, 0, VK_WHOLE_SIZE, 0, (void **)&pixels));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   CHECK(vkBeginCommandBuffer(command, &begin));
   VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
   for (unsigned i = 0; i < count; ++i) {
      VkImageMemoryBarrier barrier = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = images[i].image,
         .subresourceRange = range,
      };
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                           1, &barrier);
      VkClearColorValue color = {.float32 = {i & 1, (i >> 1) & 1, 1, 1}};
      vkCmdClearColorImage(command, images[i].image, VK_IMAGE_LAYOUT_GENERAL,
                           &color, 1, &range);
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                           1, &barrier);
      VkBufferImageCopy copy = {
         .bufferOffset = 4 * i,
         .imageSubresource = subresource(),
         .imageExtent = {1, 1, 1},
      };
      vkCmdCopyImageToBuffer(command, images[i].image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1,
                             &copy);
   }
   VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, NULL, 0,
                        NULL);
   CHECK(vkEndCommandBuffer(command));
   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command,
   };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   for (unsigned i = 0; i < count; ++i) {
      const uint8_t expected[] = {(i & 1) * 255, ((i >> 1) & 1) * 255, 255,
                                  255};
      if (memcmp(pixels + 4 * i, expected, 4)) {
         fprintf(stderr, "Image %u GPU readback mismatch\n", i);
         return 2;
      }
   }
   printf("GPU readback: %u/%u correct, vkMapMemory allocation %" PRIu64
          " bytes\n",
          count, count, (uint64_t)requirements.size);
   host_copy_check();
   vkUnmapMemory(device, readback);
   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, readback, NULL);
   for (unsigned i = 0; i < count; ++i) {
      vkDestroyImage(device, images[i].image, NULL);
      vkFreeMemory(device, images[i].memory, NULL);
   }
   free(images);
   for (unsigned i = 0; i < reserved; ++i)
      munmap(reservations[i], 64 * MiB);
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   if (retained) {
      fputs("FAIL: binding retained image CPU mappings\n", stderr);
      return 2;
   }
   puts("PASS: image binding does not retain CPU mappings");
   return 0;
}
