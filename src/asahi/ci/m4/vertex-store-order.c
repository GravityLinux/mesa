/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Offscreen Vulkan regression: a vertex-stage store must wait for an earlier
 * render's fragment reads when a FRAGMENT_SHADER -> VERTEX_SHADER barrier
 * orders them (write-after-read across renders).
 *
 * Render 1 draws a full-screen triangle whose fragment shader spins, then
 * reads B.flag (initially 1) into an R32_UINT attachment. Render 2's vertex
 * shader stores B.flag = 2. Every pixel of render 1 must read 1.
 * Requires vertexPipelineStoresAndAtomics. No implementation knobs.
 *
 * Build:
 *   for s in r1.vert r1.frag r2.vert r2.frag; do
 *     glslangValidator -V --vn vso_${s%.*}_${s#*.} -o vertex-store-order-${s%.*}-${s#*.}.h \
 *       vertex-store-order-$s; done
 *   cc -std=c11 -O2 vertex-store-order.c -lvulkan -o vertex-store-order
 * Usage: vertex-store-order [devices] [iterations] [size]   (exit 0 = pass)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "vertex-store-order-r1-vert.h"
#include "vertex-store-order-r1-frag.h"
#include "vertex-store-order-r2-vert.h"
#include "vertex-store-order-r2-frag.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, r_); exit(2); } } while (0)

static VkDevice dev;
static VkPhysicalDevice pdev;
static VkPhysicalDeviceMemoryProperties memprops;

static uint32_t
memtype(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type\n");
   exit(2);
}

struct buf { VkBuffer b; VkDeviceMemory m; void *map; };

static struct buf
mkbuf(VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct buf r;
   VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
   CK(vkCreateBuffer(dev, &ci, NULL, &r.b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, r.b, &mr);
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   CK(vkAllocateMemory(dev, &ai, NULL, &r.m));
   CK(vkBindBufferMemory(dev, r.b, r.m, 0));
   CK(vkMapMemory(dev, r.m, 0, VK_WHOLE_SIZE, 0, &r.map));
   return r;
}

struct target { VkImage img; VkDeviceMemory m; VkImageView view; VkFramebuffer fb; uint32_t w, h; };

static VkRenderPass rp;

static struct target
mktarget(uint32_t w, uint32_t h)
{
   struct target t = {.w = w, .h = h};
   VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R32_UINT, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   CK(vkCreateImage(dev, &ci, NULL, &t.img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, t.img, &mr);
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   CK(vkAllocateMemory(dev, &ai, NULL, &t.m));
   CK(vkBindImageMemory(dev, t.img, t.m, 0));
   VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t.img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   CK(vkCreateImageView(dev, &vi, NULL, &t.view));
   VkFramebufferCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp,
      .attachmentCount = 1, .pAttachments = &t.view, .width = w, .height = h, .layers = 1};
   CK(vkCreateFramebuffer(dev, &fi, NULL, &t.fb));
   return t;
}

static VkShaderModule
mkshader(const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
   VkShaderModule m;
   CK(vkCreateShaderModule(dev, &ci, NULL, &m));
   return m;
}

static VkPipelineLayout layout;

static VkPipeline
mkpipe(VkShaderModule vs, VkShaderModule fs)
{
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vin = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &ba};
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn};
   VkGraphicsPipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = st, .pVertexInputState = &vin, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pColorBlendState = &cb, .pDynamicState = &ds, .layout = layout, .renderPass = rp};
   VkPipeline p;
   CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
   return p;
}

static void
draw(VkCommandBuffer cb, struct target *t, VkPipeline p)
{
   VkClearValue clear = {.color.uint32 = {0}};
   VkRenderPassBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
      .framebuffer = t->fb, .renderArea = {{0, 0}, {t->w, t->h}}, .clearValueCount = 1, .pClearValues = &clear};
   vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
   VkViewport v = {0, 0, (float)t->w, (float)t->h, 0, 1};
   VkRect2D s = {{0, 0}, {t->w, t->h}};
   vkCmdSetViewport(cb, 0, 1, &v);
   vkCmdSetScissor(cb, 0, 1, &s);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRenderPass(cb);
}

/* The application's dependency: R1's fragment reads of B before R2's vertex
 * stores to B (an execution dependency suffices for write-after-read; the
 * access masks are added anyway). */
static void
war_barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        0, 1, &mb, 0, NULL, 0, NULL);
}

/* One device, ROUNDS submissions. Returns the number of failing rounds,
 * 77 if unsupported. */
static int
run(uint32_t rounds, uint32_t iters, uint32_t size)
{

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vertex-store-order",
      .apiVersion = VK_API_VERSION_1_2};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || !n) { fprintf(stderr, "no device\n"); return 2; }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   VkPhysicalDeviceFeatures feats;
   vkGetPhysicalDeviceFeatures(pdev, &feats);
   printf("device %s, vertexPipelineStoresAndAtomics %u\n", props.deviceName, feats.vertexPipelineStoresAndAtomics);
   if (!feats.vertexPipelineStoresAndAtomics) { printf("SKIP: no vertex stores\n"); return 77; }
   vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);

   float prio = 1;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
      .queueCount = 1, .pQueuePriorities = &prio};
   VkPhysicalDeviceFeatures en = {.vertexPipelineStoresAndAtomics = VK_TRUE};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci, .pEnabledFeatures = &en};
   CK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue q;
   vkGetDeviceQueue(dev, 0, 0, &q);

   VkAttachmentDescription att = {.format = VK_FORMAT_R32_UINT, .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
   VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   VkSubpassDescription sp = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &ref};
   VkRenderPassCreateInfo rci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
      .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sp};
   CK(vkCreateRenderPass(dev, &rci, NULL, &rp));

   VkDescriptorSetLayoutBinding bind = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
   VkDescriptorSetLayoutCreateInfo dli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &bind};
   VkDescriptorSetLayout dsl;
   CK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8};
   VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
   CK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));

   struct buf b = mkbuf(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
      .poolSizeCount = 1, .pPoolSizes = &ps};
   VkDescriptorPool pool;
   CK(vkCreateDescriptorPool(dev, &dpi, NULL, &pool));
   VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
   VkDescriptorSet set;
   CK(vkAllocateDescriptorSets(dev, &dai, &set));
   VkDescriptorBufferInfo dbi = {b.b, 0, VK_WHOLE_SIZE};
   VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi};
   vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);

   VkShaderModule s_r1v = mkshader(vso_r1_vert, sizeof(vso_r1_vert)), s_r1f = mkshader(vso_r1_frag, sizeof(vso_r1_frag));
   VkShaderModule s_r2v = mkshader(vso_r2_vert, sizeof(vso_r2_vert));
   VkShaderModule s_r2f = mkshader(vso_r2_frag, sizeof(vso_r2_frag));
   VkPipeline p1 = mkpipe(s_r1v, s_r1f), p2 = mkpipe(s_r2v, s_r2f);

   struct target t1 = mktarget(size, size), t2 = mktarget(16, 16);
   struct buf rb = mkbuf((VkDeviceSize)t1.w * t1.h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
   VkCommandPool cpool;
   CK(vkCreateCommandPool(dev, &cpi, NULL, &cpool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CK(vkAllocateCommandBuffers(dev, &cai, &cb));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CK(vkCreateFence(dev, &fci, NULL, &fence));

   struct { uint32_t iters; float seed; } pc = {iters, 0.731f};
   uint64_t bad_rounds = 0;
   for (uint32_t r = 0; r < rounds; ++r) {
      *(volatile uint32_t *)b.map = 1;
      CK(vkResetCommandBuffer(cb, 0));
      VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      CK(vkBeginCommandBuffer(cb, &cbi));
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      draw(cb, &t1, p1);
      war_barrier(cb);
      draw(cb, &t2, p2);
      VkMemoryBarrier tb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                           0, 1, &tb, 0, NULL, 0, NULL);
      VkBufferImageCopy cp = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {t1.w, t1.h, 1}};
      vkCmdCopyImageToBuffer(cb, t1.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.b, 1, &cp);
      VkMemoryBarrier hb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
      CK(vkEndCommandBuffer(cb));
      VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
      CK(vkQueueSubmit(q, 1, &si, fence));
      CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000));
      CK(vkResetFences(dev, 1, &fence));

      const uint32_t *px = rb.map;
      uint64_t one = 0, two = 0, other = 0;
      for (uint64_t i = 0; i < (uint64_t)t1.w * t1.h; ++i) {
         if (px[i] == 1) one++; else if (px[i] == 2) two++; else other++;
      }
      printf("round %u: pixels read 1: %lu, read 2 (too late): %lu, other: %lu, final flag %u\n",
                r, one, two, other, *(volatile uint32_t *)b.map);
      bad_rounds += (two + other) != 0;
   }
   return bad_rounds;
}

/* Each round runs in a fresh process with its own device: whether a store
 * can overtake the earlier render may depend on the queue's history, so the
 * first submission of a device is the one most worth repeating. */
int
main(int argc, char **argv)
{
   uint32_t rounds = argc > 1 ? atoi(argv[1]) : 20;
   uint32_t iters = argc > 2 ? atoi(argv[2]) : 20000;
   uint32_t size = argc > 3 ? atoi(argv[3]) : 1024;
   uint32_t failed = 0;
   for (uint32_t r = 0; r < rounds; ++r) {
      fflush(stdout);
      pid_t pid = fork();
      if (pid == 0)
         _exit(run(1, iters, size));
      int status = 0;
      if (pid < 0 || waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
         fprintf(stderr, "round %u: child failed\n", r);
         return 2;
      }
      if (WEXITSTATUS(status) == 77 || WEXITSTATUS(status) == 2)
         return WEXITSTATUS(status);
      failed += WEXITSTATUS(status) != 0;
   }
   printf("%s: %u of %u devices had pixels reading the vertex store\n",
          failed ? "FAIL" : "PASS", failed, rounds);
   return failed ? 1 : 0;
}
