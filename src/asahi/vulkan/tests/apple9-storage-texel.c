/* SPDX-License-Identifier: MIT
 * Native Vulkan 1.0 texel-buffer format, bounds, and descriptor replacement
 * probe. Usage: apple9-storage-texel unorm.spv uint.spv sint.spv snorm.spv
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r = (call);                                                     \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d at %d\n", #call, r, __LINE__);                \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

enum { FLOAT, UINT, SINT, SNORM, TEXELS = 7, OUTPUT_WORDS = 33 * 4 };
struct format_case {
   const char *name;
   VkFormat format;
   unsigned type, bytes;
   uint32_t data[2][4];
   double expected[2][4];
};
#define CASE(fmt, type_, bytes_, ...)                                          \
   {#fmt, VK_FORMAT_##fmt, type_, bytes_, __VA_ARGS__}
static const struct format_case cases[] = {
   CASE(A8B8G8R8_UNORM_PACK32, FLOAT, 4, {{0x4080ff00}, {0xff008040}},
        {{0, 1, 128. / 255, 64. / 255}, {64. / 255, 128. / 255, 0, 1}}),
   CASE(A8B8G8R8_UINT_PACK32, UINT, 4, {{0xffc38121}, {0x0b090705}},
        {{33, 129, 195, 255}, {5, 7, 9, 11}}),
   CASE(A8B8G8R8_SINT_PACK32, SINT, 4, {{0xffc38121}, {0x0b090705}},
        {{33, -127, -61, -1}, {5, 7, 9, 11}}),
   CASE(A8B8G8R8_SNORM_PACK32, SNORM, 4, {{0x417fc081}, {0x007f4040}},
        {{-1, -64. / 127, 1, 65. / 127}, {64. / 127, 64. / 127, 1, 0}}),
};

static VkShaderModule
load_shader(VkDevice dev, const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      perror(path);
      exit(1);
   }
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   rewind(f);
   if (size <= 0 || size % 4)
      exit(1);
   uint32_t *code = malloc(size);
   if (!code || fread(code, 1, size, f) != (size_t)size)
      exit(1);
   fclose(f);
   VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code};
   VkShaderModule result;
   CHECK(vkCreateShaderModule(dev, &ci, NULL, &result));
   free(code);
   return result;
}

int
main(int argc, char **argv)
{
   if (argc != 5) {
      fprintf(stderr, "usage: %s unorm.spv uint.spv sint.spv snorm.spv\n",
              argv[0]);
      return 1;
   }
   setbuf(stdout, NULL);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_0};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t n = 1;
   VkPhysicalDevice phy;
   CHECK(vkEnumeratePhysicalDevices(instance, &n, &phy));
   if (!n)
      return 1;
   VkPhysicalDeviceProperties prop;
   vkGetPhysicalDeviceProperties(phy, &prop);
   printf("renderer: %s\n", prop.deviceName);
   if (!strstr(prop.deviceName, "Apple M4"))
      return 1;
   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(phy, &nq, NULL);
   VkQueueFamilyProperties *queues = calloc(nq, sizeof(*queues));
   vkGetPhysicalDeviceQueueFamilyProperties(phy, &nq, queues);
   uint32_t qi;
   for (qi = 0; qi < nq && !(queues[qi].queueFlags & VK_QUEUE_COMPUTE_BIT);
        ++qi) {
   }
   free(queues);
   if (qi == nq)
      return 1;
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qi,
      .queueCount = 1,
      .pQueuePriorities = &priority};
   VkPhysicalDeviceFeatures features = {.robustBufferAccess = VK_TRUE};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .pEnabledFeatures = &features};
   VkDevice dev;
   CHECK(vkCreateDevice(phy, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qi, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(phy, &mp);
   VkBuffer buffers[2];
   VkDeviceMemory memory[2];
   void *maps[2];
   VkDeviceSize offset = prop.limits.minTexelBufferOffsetAlignment;
   if (offset < 16)
      offset = 16;
   for (unsigned b = 0; b < 2; ++b) {
      VkBufferCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = offset + 4096,
         .usage = b ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    : VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT};
      CHECK(vkCreateBuffer(dev, &ci, NULL, &buffers[b]));
      VkMemoryRequirements req;
      vkGetBufferMemoryRequirements(dev, buffers[b], &req);
      uint32_t type;
      for (type = 0; type < mp.memoryTypeCount; ++type)
         if ((req.memoryTypeBits & (1u << type)) &&
             (mp.memoryTypes[type].propertyFlags &
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            break;
      if (type == mp.memoryTypeCount)
         return 1;
      VkMemoryAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size,
         .memoryTypeIndex = type};
      CHECK(vkAllocateMemory(dev, &ai, NULL, &memory[b]));
      CHECK(vkBindBufferMemory(dev, buffers[b], memory[b], 0));
      CHECK(vkMapMemory(dev, memory[b], 0, VK_WHOLE_SIZE, 0, &maps[b]));
   }
   VkDescriptorSetLayoutBinding bindings[2] = {
      {.binding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
   VkDescriptorSetLayoutCreateInfo slci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings};
   VkDescriptorSetLayout sl;
   CHECK(vkCreateDescriptorSetLayout(dev, &slci, NULL, &sl));
   VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = sizes};
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp,
      .descriptorSetCount = 1,
      .pSetLayouts = &sl};
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkDescriptorBufferInfo output = {buffers[1], 0, OUTPUT_WORDS * 4};
   VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds,
      .dstBinding = 1,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &output};
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &sl};
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qi,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 4};
   VkCommandBuffer commands[4];
   VkPipeline pipelines[4];
   CHECK(vkAllocateCommandBuffers(dev, &cbai, commands));
   for (unsigned t = 0; t < 4; ++t) {
      VkShaderModule shader = load_shader(dev, argv[t + 1]);
      VkComputePipelineCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .layout = pl,
         .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                   .module = shader,
                   .pName = "main"}};
      CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, NULL,
                                     &pipelines[t]));
      vkDestroyShaderModule(dev, shader, NULL);
   }
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   unsigned passed = 0, skipped = 0;
   for (unsigned test = 0; test < sizeof(cases) / sizeof(cases[0]); ++test) {
      const struct format_case *c = &cases[test];
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(phy, c->format, &fp);
      if (!(fp.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT)) {
         printf("SKIP %s\n", c->name);
         ++skipped;
         continue;
      }
      memset(maps[0], 0xcc, offset + 4096);
      memset(maps[1], 0xcd, OUTPUT_WORDS * 4 + 64);
      for (unsigned i = 0; i < TEXELS; ++i)
         memcpy((uint8_t *)maps[0] + offset + i * c->bytes, c->data[i & 1],
                c->bytes);
      VkBufferViewCreateInfo vci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = buffers[0],
         .format = c->format,
         .offset = offset,
         .range = TEXELS * c->bytes};
      VkBufferView view;
      CHECK(vkCreateBufferView(dev, &vci, NULL, &view));
      write.dstBinding = 0;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
      write.pBufferInfo = NULL;
      write.pTexelBufferView = &view;
      vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);
      /* Vulkan 1.0 descriptor updates invalidate commands using the set. */
      CHECK(vkResetCommandBuffer(commands[c->type], 0));
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(commands[c->type], &begin));
      vkCmdBindPipeline(commands[c->type], VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelines[c->type]);
      vkCmdBindDescriptorSets(commands[c->type], VK_PIPELINE_BIND_POINT_COMPUTE,
                              pl, 0, 1, &ds, 0, NULL);
      vkCmdDispatch(commands[c->type], 1, 1, 1);
      VkMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                 .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                 .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(
         commands[c->type], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
      CHECK(vkEndCommandBuffer(commands[c->type]));
      VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &commands[c->type]};
      CHECK(vkQueueSubmit(queue, 1, &submit, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull));
      CHECK(vkResetFences(dev, 1, &fence));
      uint32_t *result = maps[1];
      unsigned bad = 0;
      for (unsigned i = 0; i < 32; ++i) {
         bool valid = i > 0 && i <= TEXELS;
         for (unsigned j = 0; j < 4; ++j) {
            double want = valid ? c->expected[(i - 1) & 1][j] : 0;
            uint32_t raw = result[i * 4 + j];
            float f;
            memcpy(&f, &raw, 4);
            double got = (c->type == FLOAT || c->type == SNORM) ? f
                         : c->type == SINT ? (double)(int32_t)raw
                                           : raw;
            if (!isfinite(got) ||
                fabs(got - want) >
                   ((c->type == FLOAT || c->type == SNORM) ? 0.00002 : 0)) {
               if (bad++ < 4)
                  fprintf(stderr, "%s [%u].%u got %.9g want %.9g\n", c->name, i,
                          j, got, want);
            }
         }
      }
      for (unsigned j = 0; j < 4; ++j)
         bad += result[128 + j] != (j ? 0 : TEXELS);
      for (unsigned j = OUTPUT_WORDS; j < OUTPUT_WORDS + 16; ++j)
         bad += result[j] != 0xcdcdcdcd;
      /* Every valid texel is written with reversed channels. Check the
       * entire buffer, including writes attempted outside the view range. */
      for (unsigned byte = 0; byte < offset + 4096; ++byte) {
         uint8_t want = 0xcc;
         if (byte >= offset && byte < offset + TEXELS * 4) {
            unsigned texel = (byte - offset) / 4;
            unsigned channel = (byte - offset) % 4;
            want = c->data[texel & 1][0] >> ((3 - channel) * 8);
         }
         bad += ((uint8_t *)maps[0])[byte] != want;
      }
      vkDestroyBufferView(dev, view, NULL);
      if (bad) {
         fprintf(stderr, "FAIL %s: %u mismatches\n", c->name, bad);
         return 1;
      }
      printf("PASS %s\n", c->name);
      ++passed;
   }
   printf("%u passed, %u skipped\n", passed, skipped);
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   for (unsigned t = 0; t < 4; ++t)
      vkDestroyPipeline(dev, pipelines[t], NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyDescriptorSetLayout(dev, sl, NULL);
   for (unsigned b = 0; b < 2; ++b) {
      vkUnmapMemory(dev, memory[b]);
      vkDestroyBuffer(dev, buffers[b], NULL);
      vkFreeMemory(dev, memory[b], NULL);
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   return passed ? 0 : 1;
}
