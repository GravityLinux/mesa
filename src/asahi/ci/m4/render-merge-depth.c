/* SPDX-License-Identifier: MIT
 * Consecutive depth/stencil-only dynamic rendering instances. Normal mode
 * checks every defined texel on a real Vulkan implementation. --record-only
 * only records/submits commands, for a host DRM noop-shim stream comparison.
 * Build: cc -std=c11 -O2 render-merge-depth.c -lvulkan -lm -o render-merge-depth
 */
#include <vulkan/vulkan.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 64
#define H 64
#define PIXELS (W * H)
#define STRIDE (PIXELS * 8)
#define CK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
   fprintf(stderr, "%s: %d (line %u)\n", #x, r, __LINE__); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties memory;
static VkCommandBuffer cb;
static VkFormat format;
static VkImageAspectFlags aspects;
static VkQueue queue;
static VkBuffer readback;
static VkDeviceMemory readback_memory;
static void *mapped;
static int record_only;

struct image {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView views[2];
   float depth[PIXELS];
   uint8_t stencil[PIXELS], undefined[PIXELS];
};

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags wanted)
{
   for (unsigned i = 0; i < memory.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (memory.memoryTypes[i].propertyFlags & wanted) == wanted)
         return i;
   fprintf(stderr, "No suitable memory type\n"); exit(1);
}

static struct image
make_image(void)
{
   struct image im = {0};
   CK(vkCreateImage(dev, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
   }, NULL, &im.image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, im.image, &req);
   CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size, .memoryTypeIndex = memory_type(req.memoryTypeBits, 0),
   }, NULL, &im.memory));
   CK(vkBindImageMemory(dev, im.image, im.memory, 0));
   for (unsigned i = 0; i < 2; i++)
      CK(vkCreateImageView(dev, &(VkImageViewCreateInfo){
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = im.image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
         .subresourceRange = {aspects, 0, 1, 0, 1},
      }, NULL, &im.views[i]));
   return im;
}

static void
barrier(VkPipelineStageFlags src, VkPipelineStageFlags dst,
        VkAccessFlags src_access, VkAccessFlags dst_access)
{
   VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = src_access, .dstAccessMask = dst_access};
   vkCmdPipelineBarrier(cb, src, dst, VK_DEPENDENCY_BY_REGION_BIT,
                        1, &mb, 0, NULL, 0, NULL);
}

static void
full_barrier(void)
{
   barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
           VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

static void
init_image(struct image *im)
{
   VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = im->image, .subresourceRange = {aspects, 0, 1, 0, 1}};
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
   VkClearDepthStencilValue value = {.depth = 0.5f, .stencil = 19};
   vkCmdClearDepthStencilImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, &value,
                              1, &b.subresourceRange);
   for (unsigned i = 0; i < PIXELS; i++) { im->depth[i] = 0.5f; im->stencil[i] = 19; }
}

static void
pass(struct image *im, unsigned view, VkRect2D area, VkRect2D rect,
     float z, unsigned s, VkAttachmentLoadOp load, VkAttachmentStoreOp store)
{
   VkClearValue clear = {.depthStencil = {z, s}};
   VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = im->views[view], .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = load, .storeOp = store, .clearValue = clear,
   };
   VkRenderingInfo info = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = area, .layerCount = 1,
      .pDepthAttachment = (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? &att : NULL,
      .pStencilAttachment = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? &att : NULL};
   vkCmdBeginRendering(cb, &info);
   if (load != VK_ATTACHMENT_LOAD_OP_CLEAR) {
      VkClearAttachment ca = {.aspectMask = aspects, .clearValue = clear};
      VkClearRect cr = {.rect = rect, .layerCount = 1};
      vkCmdClearAttachments(cb, 1, &ca, 1, &cr);
   } else {
      rect = area;
   }
   vkCmdEndRendering(cb);
   for (unsigned y = rect.offset.y; y < rect.offset.y + rect.extent.height; y++)
      for (unsigned x = rect.offset.x; x < rect.offset.x + rect.extent.width; x++) {
         im->depth[y * W + x] = z; im->stencil[y * W + x] = s;
         im->undefined[y * W + x] = 0;
      }
   if (store == VK_ATTACHMENT_STORE_OP_DONT_CARE)
      for (unsigned y = area.offset.y; y < area.offset.y + area.extent.height; y++)
         for (unsigned x = area.offset.x; x < area.offset.x + area.extent.width; x++)
            im->undefined[y * W + x] = 1;
}

static void
snapshot(struct image *im, unsigned index)
{
   VkBufferImageCopy copy = {.bufferOffset = index * STRIDE,
      .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
      .imageExtent = {W, H, 1}};
   if (aspects & VK_IMAGE_ASPECT_DEPTH_BIT)
      vkCmdCopyImageToBuffer(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, readback, 1, &copy);
   if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) {
      copy.bufferOffset += PIXELS * 4;
      copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      vkCmdCopyImageToBuffer(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, readback, 1, &copy);
   }
}

static unsigned
check_image(struct image *im, unsigned index)
{
   const uint8_t *p = (const uint8_t *)mapped + index * STRIDE;
   unsigned bad = 0;
   for (unsigned i = 0; i < PIXELS; i++) {
      if (im->undefined[i]) continue;
      if (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) {
         float got = format == VK_FORMAT_D16_UNORM ? ((const uint16_t *)p)[i] / 65535.0f
                                                  : ((const float *)p)[i];
         float tolerance = format == VK_FORMAT_D16_UNORM ? 1.5f / 65535.0f : 0.000001f;
         if (!isfinite(got) || fabsf(got - im->depth[i]) > tolerance) bad++;
      }
      if ((aspects & VK_IMAGE_ASPECT_STENCIL_BIT) && p[PIXELS * 4 + i] != im->stencil[i]) bad++;
   }
   return bad;
}

static void
destroy_image(struct image *im)
{
   for (unsigned i = 0; i < 2; i++) vkDestroyImageView(dev, im->views[i], NULL);
   vkDestroyImage(dev, im->image, NULL); vkFreeMemory(dev, im->memory, NULL);
}

int
main(int argc, char **argv)
{
   if (argc < 3 || argc > 4) {
      fprintf(stderr, "usage: %s d16|d32|d32s8|s8 same-area|alias-view|empty-clear|area-change|clear-load|store-dontcare|attachment-change|sample-barrier|full-barrier [--record-only]\n", argv[0]);
      return 2;
   }
   record_only = argc == 4 && !strcmp(argv[3], "--record-only");
   if (argc == 4 && !record_only) return 2;
   const char *kind = argv[2];
   const char *cases[] = {"same-area", "alias-view", "empty-clear", "area-change", "clear-load", "store-dontcare", "attachment-change", "sample-barrier", "full-barrier"};
   int found = 0;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) found |= !strcmp(kind, cases[i]);
   if (!found) return 2;
   if (!strcmp(argv[1], "d16")) format = VK_FORMAT_D16_UNORM;
   else if (!strcmp(argv[1], "d32")) format = VK_FORMAT_D32_SFLOAT;
   else if (!strcmp(argv[1], "d32s8")) format = VK_FORMAT_D32_SFLOAT_S8_UINT;
   else if (!strcmp(argv[1], "s8")) format = VK_FORMAT_S8_UINT;
   else return 2;
   aspects = (format != VK_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
             ((format == VK_FORMAT_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
   VkInstance instance;
   CK(vkCreateInstance(&(VkInstanceCreateInfo){.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
         .pApplicationName = "render-merge-depth", .apiVersion = VK_API_VERSION_1_3}}, NULL, &instance));
   VkPhysicalDevice physical[8]; uint32_t n = 8;
   CK(vkEnumeratePhysicalDevices(instance, &n, physical)); if (!n) return 1;
   VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(physical[0], &props);
   printf("renderer=%s driver=%u format=%s case=%s mode=%s\n", props.deviceName, props.driverVersion, argv[1], kind, record_only ? "record-only" : "pixel-check");
   uint32_t families = 0; vkGetPhysicalDeviceQueueFamilyProperties(physical[0], &families, NULL);
   VkQueueFamilyProperties *qprops = calloc(families, sizeof(*qprops));
   vkGetPhysicalDeviceQueueFamilyProperties(physical[0], &families, qprops);
   uint32_t family = 0;
   while (family < families && !(qprops[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) family++;
   free(qprops); if (family == families) return 1;
   float priority = 1.0f;
   VkPhysicalDeviceVulkan13Features features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE};
   CK(vkCreateDevice(physical[0], &(VkDeviceCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features, .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority}}, NULL, &dev));
   vkGetDeviceQueue(dev, family, 0, &queue); vkGetPhysicalDeviceMemoryProperties(physical[0], &memory);
   VkCommandPool pool;
   CK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family}, NULL, &pool));
   CK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cb));
   CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 2 * STRIDE, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT}, NULL, &readback));
   VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, readback, &req);
   CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size, .memoryTypeIndex = memory_type(req.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}, NULL, &readback_memory));
   CK(vkBindBufferMemory(dev, readback, readback_memory, 0));
   CK(vkMapMemory(dev, readback_memory, 0, VK_WHOLE_SIZE, 0, &mapped)); memset(mapped, 0x5a, 2 * STRIDE);
   struct image a = make_image(), b = make_image();
   CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
   init_image(&a); init_image(&b); full_barrier();
   VkRect2D area = {{4, 4}, {56, 56}}, r1 = {{8, 8}, {24, 24}}, r2 = {{20, 20}, {24, 24}};
   if (!strcmp(kind, "empty-clear")) area = (VkRect2D){{0, 0}, {W, H}};
   pass(&a, 0, area, r1, .25f, 53, !strcmp(kind, "empty-clear") ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
   if (!strcmp(kind, "sample-barrier"))
      barrier(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
   else if (!strcmp(kind, "full-barrier")) full_barrier();
   else barrier(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
   if (!strcmp(kind, "area-change")) area = (VkRect2D){{16, 16}, {40, 40}};
   pass(!strcmp(kind, "attachment-change") ? &b : &a,
        !strcmp(kind, "alias-view"), area, r2, .75f, 167,
        !strcmp(kind, "clear-load") ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        !strcmp(kind, "store-dontcare") ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE);
   full_barrier(); snapshot(&a, 0); snapshot(&b, 1);
   barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
   CK(vkEndCommandBuffer(cb));
   VkFence fence; CK(vkCreateFence(dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}, NULL, &fence));
   CK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb}, fence));
   CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_C(10000000000)));
   unsigned bad = record_only ? 0 : check_image(&a, 0) + check_image(&b, 1);
   if (record_only) puts("RECORDED; GPU pixels were not checked");
   else printf("PIXELS mismatches=%u\n", bad);
   vkDestroyFence(dev, fence, NULL); vkDestroyCommandPool(dev, pool, NULL);
   destroy_image(&a); destroy_image(&b); vkUnmapMemory(dev, readback_memory);
   vkDestroyBuffer(dev, readback, NULL); vkFreeMemory(dev, readback_memory, NULL);
   vkDestroyDevice(dev, NULL); vkDestroyInstance(instance, NULL);
   return bad ? 1 : 0;
}
