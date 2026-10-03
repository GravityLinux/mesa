/* SPDX-License-Identifier: MIT
 * Attachmentless 1x/4x rasterization through core Vulkan 1.0 occlusion queries.
 * Pass the procedural fullscreen vertex shader apple9-render-draw.vert.spv.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(expr)                                                            \
   do {                                                                        \
      VkResult r = (expr);                                                     \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d at %d\n", #expr, r, __LINE__);                \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

int
main(int argc, char **argv)
{
   if (argc != 2)
      return 2;
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_0};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice phy;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &phy));
   if (!count)
      return 1;
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(phy, &props);
   printf("device: %s\n", props.deviceName);
   if (!strstr(props.deviceName, "Apple M4"))
      return 1;
   vkGetPhysicalDeviceQueueFamilyProperties(phy, &count, NULL);
   VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(phy, &count, families);
   unsigned family;
   for (family = 0; family < count; ++family)
      if (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)
         break;
   free(families);
   if (family == count)
      return 1;
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   VkDevice dev;
   CHECK(vkCreateDevice(phy, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, family, 0, &queue);

   FILE *file = fopen(argv[1], "rb");
   if (!file)
      return 1;
   fseek(file, 0, SEEK_END);
   long size = ftell(file);
   rewind(file);
   if (size <= 0 || size % 4)
      return 1;
   uint32_t *code = malloc(size);
   if (fread(code, 1, size, file) != (size_t)size)
      return 1;
   fclose(file);
   VkShaderModuleCreateInfo mi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code};
   VkShaderModule vs;
   CHECK(vkCreateShaderModule(dev, &mi, NULL, &vs));
   free(code);
   VkPipelineLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &li, NULL, &layout));
   VkSubpassDescription subpass = {.pipelineBindPoint =
                                      VK_PIPELINE_BIND_POINT_GRAPHICS};
   VkRenderPassCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .subpassCount = 1,
      .pSubpasses = &subpass};
   VkRenderPass render;
   CHECK(vkCreateRenderPass(dev, &ri, NULL, &render));
   VkFramebufferCreateInfo fi = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render,
      .width = 37,
      .height = 19,
      .layers = 1};
   VkFramebuffer fb;
   CHECK(vkCreateFramebuffer(dev, &fi, NULL, &fb));
   VkPipelineShaderStageCreateInfo stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vs,
      .pName = "main"};
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport viewport = {0, 0, 37, 19, 0, 1};
   VkRect2D scissor = {{3, 2}, {30, 14}};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
   VkGraphicsPipelineCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 1,
      .pStages = &stage,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .layout = layout,
      .renderPass = render};
   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = family};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkQueryPoolCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_OCCLUSION,
      .queryCount = 1};
   VkQueryPool query;
   CHECK(vkCreateQueryPool(dev, &qi, NULL, &query));
   unsigned failures = 0;
   for (unsigned samples = 1; samples <= 4; samples *= 4) {
      for (unsigned enabled = 0; enabled < 2; ++enabled) {
         VkSampleMask mask = enabled ? 1u << (samples - 1) : 0;
         ms.rasterizationSamples = samples;
         ms.pSampleMask = &mask;
         VkPipeline pipeline;
         CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, NULL,
                                         &pipeline));
         CHECK(vkResetCommandBuffer(cmd, 0));
         VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
         CHECK(vkBeginCommandBuffer(cmd, &bi));
         vkCmdResetQueryPool(cmd, query, 0, 1);
         VkRenderPassBeginInfo rb = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render,
            .framebuffer = fb,
            .renderArea = {{0, 0}, {37, 19}}};
         vkCmdBeginRenderPass(cmd, &rb, VK_SUBPASS_CONTENTS_INLINE);
         vkCmdBeginQuery(cmd, query, 0, 0);
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
         vkCmdDraw(cmd, 3, 1, 0, 0);
         vkCmdEndQuery(cmd, query, 0);
         vkCmdEndRenderPass(cmd);
         CHECK(vkEndCommandBuffer(cmd));
         VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .commandBufferCount = 1,
                                .pCommandBuffers = &cmd};
         CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
         CHECK(vkQueueWaitIdle(queue));
         uint64_t result[2] = {0, 0};
         CHECK(vkGetQueryPoolResults(
            dev, query, 0, 1, sizeof(result), result, sizeof(result),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT |
               VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
         /* Conservative occlusion guarantees zero/nonzero, not an exact count. */
         int pass = result[1] && (!!result[0] == enabled);
         printf("%ux mask=0x%x result=%llu available=%llu: %s\n", samples, mask,
                (unsigned long long)result[0], (unsigned long long)result[1],
                pass ? "PASS" : "FAIL");
         failures += !pass;
         vkDestroyPipeline(dev, pipeline, NULL);
      }
   }
   vkDestroyQueryPool(dev, query, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, render, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   return failures ? 1 : 0;
}
