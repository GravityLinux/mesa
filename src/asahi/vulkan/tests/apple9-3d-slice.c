/* SPDX-License-Identifier: MIT
 * Native Vulkan 3D sampling boundary regression.
 * Fill slices with their layer number using host image copy, blit a selected
 * slice to 2D, and check all pixels and readback guards. Usage: [depth [slice]].
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
   setbuf(stdout, NULL);
   unsigned depth = argc > 1 ? atoi(argv[1]) : 4096;
   unsigned slice = argc > 2 ? atoi(argv[2]) : 8;
   printf("depth=%u slice=%u\n", depth, slice);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "Apple9 render-pass clear",
                            .apiVersion = VK_API_VERSION_1_3};
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
   printf("maxImageDimension3D=%u\n", prop.limits.maxImageDimension3D);
   if (depth > prop.limits.maxImageDimension3D) {
      puts("SKIP depth exceeds advertised 3D image range"); return 77;
   }
   if (0) {
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
   VkPhysicalDeviceHostImageCopyFeaturesEXT hf={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,.hostImageCopy=VK_TRUE};
   const char *he=VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME;
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci, .pNext=&hf, .enabledExtensionCount=1, .ppEnabledExtensionNames=&he};
   VkDevice dev;
   CHECK(vkCreateDevice(phy, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qi, 0, &queue);
   enum { W = 2, H = 2, PIXELS = W * H, BYTES = PIXELS * 4, GUARD = 64 };
   VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                           .flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT,
                           .imageType = VK_IMAGE_TYPE_3D,
                           .format = VK_FORMAT_R8G8B8A8_UNORM,
                           .extent = {W, H, depth},
                           .mipLevels = 1,
                           .arrayLayers = 1,
                           .samples = VK_SAMPLE_COUNT_1_BIT,
                           .tiling = VK_IMAGE_TILING_OPTIMAL,
                           .usage = VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
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
   PFN_vkTransitionImageLayoutEXT transition=(PFN_vkTransitionImageLayoutEXT)vkGetDeviceProcAddr(dev,"vkTransitionImageLayoutEXT");
   PFN_vkCopyMemoryToImageEXT host_copy=(PFN_vkCopyMemoryToImageEXT)vkGetDeviceProcAddr(dev,"vkCopyMemoryToImageEXT");
   if (!transition || !host_copy) return 2;
   CHECK(transition(dev,1,&(VkHostImageLayoutTransitionInfoEXT){.sType=VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,.image=image,.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_GENERAL,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}}));
   uint8_t *source=malloc(16*depth);
   for(unsigned z=0;z<depth;z++)for(unsigned pixel=0;pixel<4;pixel++) {
      source[z*16+pixel*4+0]=z&255; source[z*16+pixel*4+1]=z>>8;
      source[z*16+pixel*4+2]=90; source[z*16+pixel*4+3]=255;
   }
   VkMemoryToImageCopyEXT region={.sType=VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,.pHostPointer=source,.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={W,H,depth}};
   CHECK(host_copy(dev,&(VkCopyMemoryToImageInfoEXT){.sType=VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,.dstImage=image,.dstImageLayout=VK_IMAGE_LAYOUT_GENERAL,.regionCount=1,.pRegions=&region}));
   free(source);
   VkImage staging_image; VkDeviceMemory staging_memory;
   VkImageCreateInfo si = ii; si.flags = 0; si.imageType = VK_IMAGE_TYPE_2D;
   si.extent.depth = 1; si.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   CHECK(vkCreateImage(dev, &si, NULL, &staging_image));
   VkMemoryRequirements sr; vkGetImageMemoryRequirements(dev, staging_image, &sr);
   unsigned st = 0; while (!(sr.memoryTypeBits & (1u << st))) st++;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=sr.size,.memoryTypeIndex=st},NULL,&staging_memory));
   CHECK(vkBindImageMemory(dev, staging_image, staging_memory, 0));
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = ii.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, slice, 1}};
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vi, NULL, &view));
   VkAttachmentDescription ad = {
      .format = ii.format,
      .samples = ii.samples,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
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
   for (unsigned pass = 0; pass < 1; ++pass) {
      const uint8_t expected[4] = {slice & 255, slice >> 8, 90, 255};
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
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                           1, &barrier);
      VkImageMemoryBarrier sb = barrier; sb.image = staging_image;
      sb.srcAccessMask=0; sb.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
      sb.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; sb.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&sb);
      VkImageBlit blit = {.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
        .srcOffsets={{0,0,slice},{W,H,slice+1}}, .dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
        .dstOffsets={{0,0,0},{W,H,1}}};
      vkCmdBlitImage(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,staging_image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,VK_FILTER_NEAREST);
      sb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; sb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
      sb.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; sb.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&sb);
      VkBufferImageCopy copy = {
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageOffset = {0, 0, 0},
         .imageExtent = {W, H, 1}};
      vkCmdCopyImageToBuffer(cmd, staging_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
      printf("SAMPLE rgba=%u,%u,%u,%u requested_slice=%u sampled_slice=%u\n",pixels[0],pixels[1],pixels[2],pixels[3],slice,pixels[0]+256u*pixels[1]);
      for (unsigned i = 0; i < BYTES; ++i) {
         if (abs((int)pixels[i] - expected[i % 4]) > 1) {
            fprintf(stderr, "pixel %u channel %u: got %u expected %u\n", i / 4,
                    i % 4, pixels[i], expected[i % 4]);
            return 1;
         }
      }
      for (unsigned i = BYTES; i < BYTES + GUARD; ++i)
         if (pixels[i] != 0xa5)
            return 1;
      printf("PASS clear/readback %u (%u pixels, guards intact)\n", pass,
             PIXELS);
   }
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, render, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkUnmapMemory(dev, memory[1]);
   vkDestroyBuffer(dev, buffer, NULL);
   vkDestroyImage(dev, staging_image, NULL);
   vkFreeMemory(dev, staging_memory, NULL);
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, memory[1], NULL);
   vkFreeMemory(dev, memory[0], NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
