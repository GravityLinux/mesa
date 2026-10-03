/* SPDX-License-Identifier: MIT
 * Native Vulkan compute/readback smoke test. No driver-private interfaces.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#ifndef APPLE9_PROBE_COUNT
#define APPLE9_PROBE_COUNT 257
#endif
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
   if (argc < 2 || argc > 3) {
      fprintf(stderr, "usage: %s shader.spv [--transfers|--shared]\n", argv[0]);
      return 1;
   }
   bool transfers = argc == 3 && !strcmp(argv[2], "--transfers");
   bool shared = argc == 3 && !strcmp(argv[2], "--shared");
   if (argc == 3 && !transfers && !shared) {
      fprintf(stderr, "Unknown test mode: %s\n", argv[2]);
      return 1;
   }
   setbuf(stdout, NULL);
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "Apple9 compute smoke",
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
   VkQueue queue;
   vkGetDeviceQueue(dev, qi, 0, &queue);
   enum { COUNT = APPLE9_PROBE_COUNT, WORDS = (COUNT + 63) & ~63,
          BYTES = WORDS * 4 };
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(phy, &mp);
   VkBuffer buffers[3];
   VkDeviceMemory memory[3];
   uint32_t *maps[3];
   for (unsigned b = 0; b < 3; b++) {
      VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = BYTES,
                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT};
      CHECK(vkCreateBuffer(dev, &bi, NULL, &buffers[b]));
      VkMemoryRequirements req;
      vkGetBufferMemoryRequirements(dev, buffers[b], &req);
      uint32_t t;
      for (t = 0; t < mp.memoryTypeCount; t++)
         if ((req.memoryTypeBits & (1u << t)) &&
             (mp.memoryTypes[t].propertyFlags &
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            break;
      if (t == mp.memoryTypeCount)
         return 1;
      VkMemoryAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size,
         .memoryTypeIndex = t};
      CHECK(vkAllocateMemory(dev, &ai, NULL, &memory[b]));
      CHECK(vkBindBufferMemory(dev, buffers[b], memory[b], 0));
      CHECK(
         vkMapMemory(dev, memory[b], 0, VK_WHOLE_SIZE, 0, (void **)&maps[b]));
   }
   VkDescriptorSetLayoutBinding bindings[2];
   memset(bindings, 0, sizeof(bindings));
   for (unsigned b = 0; b < 2; b++)
      bindings[b] = (VkDescriptorSetLayoutBinding){
         .binding = b,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo slci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings};
   VkDescriptorSetLayout sl;
   CHECK(vkCreateDescriptorSetLayout(dev, &slci, NULL, &sl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &ps};
   VkDescriptorPool dp;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp,
      .descriptorSetCount = 1,
      .pSetLayouts = &sl};
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   for (unsigned b = 0; b < 2; b++) {
      VkDescriptorBufferInfo dbi = {buffers[b], 0, BYTES};
      VkWriteDescriptorSet w = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds,
         .dstBinding = b,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi};
      vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
   }
   VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &sl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pc};
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   FILE *f = fopen(argv[1], "rb");
   if (!f)
      return 1;
   fseek(f, 0, SEEK_END);
   long bytes = ftell(f);
   rewind(f);
   uint32_t *code = malloc(bytes);
   if (fread(code, 1, bytes, f) != (size_t)bytes)
      return 1;
   fclose(f);
   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = bytes,
      .pCode = code};
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   free(code);
   VkComputePipelineCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm,
                .pName = "main"},
      .layout = pl};
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, NULL, &pipe));
   puts("pipeline compiled");
   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qi};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   for (uint32_t run = 0; run < 3; run++) {
      for (uint32_t i = 0; i < WORDS; i++) {
         maps[0][i] = (i * 37u) ^ ((run + 1) * 0x1234567u);
         maps[1][i] = transfers ? 0xfeedfaceu : 0xdeadbeefu;
         maps[2][i] = 0xfeedfaceu;
      }
      if (run == 1) {
         maps[0][0] = (COUNT + 63) / 64;
         maps[0][1] = maps[0][2] = 1;
      }
      uint32_t push[2] = {COUNT, 0x13579bdfu + run};
      CHECK(vkResetCommandBuffer(cb, 0));
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(cb, &begin));
      if (transfers) {
         vkCmdFillBuffer(cb, buffers[1], 0, BYTES, 0xdeadbeefu);
         VkMemoryBarrier fill = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         };
         vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fill,
                              0, NULL, 0, NULL);
      }
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds,
                              0, NULL);
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
      if (run == 1)
         vkCmdDispatchIndirect(cb, buffers[0], 0);
      else
         vkCmdDispatch(cb, (COUNT + 63) / 64, 1, 1);
      VkMemoryBarrier barrier = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask =
            transfers ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_HOST_READ_BIT,
      };
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           transfers ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                     : VK_PIPELINE_STAGE_HOST_BIT,
                           0, 1, &barrier, 0, NULL, 0, NULL);
      if (transfers) {
         VkBufferCopy copy = {.size = BYTES};
         vkCmdCopyBuffer(cb, buffers[1], buffers[2], 1, &copy);
         barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
         barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
         vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                              NULL, 0, NULL);
      }
      CHECK(vkEndCommandBuffer(cb));
      VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &cb};
      CHECK(vkQueueSubmit(queue, 1, &submit, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10000000000ull));
      CHECK(vkResetFences(dev, 1, &fence));
      uint32_t *result = maps[transfers ? 2 : 1];
      unsigned bad = 0;
      for (uint32_t i = 0; i < WORDS; i++) {
         uint32_t want =
            i < COUNT
               ? (maps[0][shared ? (i ^ 1) : i] * 1664525u + 1013904223u) ^
                    push[1]
               : 0xdeadbeefu;
         if (result[i] != want) {
            if (bad < 8)
               fprintf(stderr, "run %u [%u] got=%08x want=%08x\n", run, i,
                       result[i], want);
            bad++;
         }
      }
      printf("run %u: %u outputs + %u guard words, %u mismatches\n", run, COUNT,
             WORDS - COUNT, bad);
      if (bad)
         return 1;
   }
   CHECK(vkDeviceWaitIdle(dev));
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyDescriptorSetLayout(dev, sl, NULL);
   for (unsigned b = 0; b < 3; b++) {
      vkUnmapMemory(dev, memory[b]);
      vkDestroyBuffer(dev, buffers[b], NULL);
      vkFreeMemory(dev, memory[b], NULL);
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   puts("PASS");
   return 0;
}
