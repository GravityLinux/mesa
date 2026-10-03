/* SPDX-License-Identifier: MIT
 * Register-indexed sampling (DXVK style): a 2048-entry sampler heap in set 0
 * (update-after-bind, partially bound, variable count) indexed by a push
 * constant, and separate sampled images in set 1. A full-screen draw samples a
 * hashed pattern image into a target that is read back and compared exactly.
 * Environment: WIDTH, HEIGHT, DRAWS (full-screen draws, each read back), SAMPLER
 * (heap index to use), SAMPLERS (entries written), TEXSETS (image descriptor
 * sets, cycled per draw), NOISE (extra draws with other descriptor sets between
 * measured draws, not read back), POOLS (descriptor pools the image sets are
 * spread over), IMAGES (distinct source patterns, cycled by descriptor set),
 * HOSTWRITE (keep writing unused heap entries and spare sets from
 * the host while the GPU executes, as a late-bound heap does), NOISEIMM (noise
 * draws use a combined image sampler at a fixed binding, i.e. per-draw tables;
 * needs a third shader argument).
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

static uint32_t
find_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
          VkMemoryPropertyFlags want)
{
   for (uint32_t t = 0; t < mp->memoryTypeCount; ++t)
      if ((bits & (1u << t)) && (mp->memoryTypes[t].propertyFlags & want) == want)
         return t;
   fprintf(stderr, "no memory type\n");
   exit(1);
}

static VkShaderModule
load_module(VkDevice dev, const char *path)
{
   FILE *file = fopen(path, "rb");
   if (!file) {
      fprintf(stderr, "cannot open %s\n", path);
      exit(1);
   }
   fseek(file, 0, SEEK_END);
   size_t size = ftell(file);
   rewind(file);
   uint32_t *code = malloc(size);
   if (fread(code, 1, size, file) != size)
      exit(1);
   fclose(file);
   VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = size, .pCode = code};
   VkShaderModule module;
   CHECK(vkCreateShaderModule(dev, &mi, NULL, &module));
   free(code);
   return module;
}

static uint32_t
hash_pixel(uint32_t x, uint32_t y, uint32_t seed)
{
   uint32_t h = (x * 73856093u) ^ (y * 19349663u) ^ (seed * 83492791u);
   h ^= h >> 13;
   h *= 0x5bd1e995u;
   h ^= h >> 15;
   return h;
}

static VkImage
make_image(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp, uint32_t W,
           uint32_t H, VkImageUsageFlags usage, VkImageView *view)
{
   VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                           .imageType = VK_IMAGE_TYPE_2D,
                           .format = VK_FORMAT_R8G8B8A8_UNORM,
                           .extent = {W, H, 1},
                           .mipLevels = 1,
                           .arrayLayers = 1,
                           .samples = VK_SAMPLE_COUNT_1_BIT,
                           .tiling = VK_IMAGE_TILING_OPTIMAL,
                           .usage = usage};
   VkImage image;
   CHECK(vkCreateImage(dev, &ii, NULL, &image));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, image, &mr);
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = mr.size,
                              .memoryTypeIndex = find_type(mp, mr.memoryTypeBits,
                                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
   CHECK(vkBindImageMemory(dev, image, mem, 0));
   VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                               .image = image,
                               .viewType = VK_IMAGE_VIEW_TYPE_2D,
                               .format = VK_FORMAT_R8G8B8A8_UNORM,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   CHECK(vkCreateImageView(dev, &vi, NULL, view));
   return image;
}

static void
image_barrier(VkCommandBuffer cmd, VkImage image, VkAccessFlags src, VkAccessFlags dst,
              VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
   VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                             .srcAccessMask = src,
                             .dstAccessMask = dst,
                             .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                             .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                             .image = image,
                             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr, "usage: %s fullscreen.vert.spv bindless.frag.spv\n", argv[0]);
      return 2;
   }
   uint32_t W = getenv("WIDTH") ? atoi(getenv("WIDTH")) : 1280;
   uint32_t H = getenv("HEIGHT") ? atoi(getenv("HEIGHT")) : 720;
   uint32_t images = getenv("IMAGES") ? atoi(getenv("IMAGES")) : 1;
   uint32_t draws = getenv("DRAWS") ? atoi(getenv("DRAWS")) : 16;
   uint32_t sampler_index = getenv("SAMPLER") ? atoi(getenv("SAMPLER")) : 5;
   uint32_t samplers = getenv("SAMPLERS") ? atoi(getenv("SAMPLERS")) : 8;
   uint32_t texsets = getenv("TEXSETS") ? atoi(getenv("TEXSETS")) : 1;
   uint32_t noise = getenv("NOISE") ? atoi(getenv("NOISE")) : 0;
   uint32_t pools = getenv("POOLS") ? atoi(getenv("POOLS")) : 1;
   bool hostwrite = getenv("HOSTWRITE") && atoi(getenv("HOSTWRITE"));
   bool noise_imm = getenv("NOISEIMM") && atoi(getenv("NOISEIMM")) && argc > 3;
   if (!W || !H || W % 8 || H % 4 || !images || !draws || !texsets || !pools) {
      fprintf(stderr, "positive sizes required; WIDTH/HEIGHT must be multiples of 8/4\n");
      return 2;
   }
   const uint32_t spare_sets = 8;
   if (sampler_index >= samplers)
      samplers = sampler_index + 1;
   const uint32_t heap_size = 2048;

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t count = 1;
   VkPhysicalDevice pdev;
   vkEnumeratePhysicalDevices(inst, &count, &pdev);
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   printf("device=%s images=%u\n", props.deviceName, images);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   uint32_t qi = 0;
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = qi,
                                  .queueCount = 1,
                                  .pQueuePriorities = &prio};
   VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .descriptorIndexing = VK_TRUE,
      .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
      .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
      .descriptorBindingPartiallyBound = VK_TRUE,
      .descriptorBindingVariableDescriptorCount = VK_TRUE,
      .runtimeDescriptorArray = VK_TRUE};
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &f12,
      .dynamicRendering = VK_TRUE};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &f13,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qi, 0, &queue);

   /* Distinct source patterns also exercise descriptor and texture cache misses. */
   VkImageView *src_views = calloc(images, sizeof(*src_views));
   VkImage *sources = calloc(images, sizeof(*sources));
   for (unsigned i = 0; i < images; ++i)
      sources[i] = make_image(dev, &mp, W, H,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              &src_views[i]);
   VkImageView src_view = src_views[0];
   const size_t bytes = (size_t)W * H * 4;
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = bytes * (images + draws),
                            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer;
   CHECK(vkCreateBuffer(dev, &bi, NULL, &buffer));
   VkMemoryRequirements br;
   vkGetBufferMemoryRequirements(dev, buffer, &br);
   VkMemoryAllocateInfo bai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = br.size,
      .memoryTypeIndex = find_type(&mp, br.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, buffer, bmem, 0));
   uint8_t *pixels;
   CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&pixels));
   uint32_t *pattern = (uint32_t *)pixels;
   for (uint32_t i = 0; i < images; ++i)
      for (uint32_t y = 0; y < H; ++y)
         for (uint32_t x = 0; x < W; ++x)
            pattern[i * W * H + y * W + x] = hash_pixel(x, y, 1 + i) | 0xff000000u;
   memset(pixels + bytes * images, 0xa5, bytes * draws);

   /* Targets: two, alternating. */
   VkImageView target_view[2];
   VkImage target[2];
   for (unsigned i = 0; i < 2; ++i)
      target[i] = make_image(dev, &mp, W, H,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                             &target_view[i]);

   /* Sampler heap: set 0, binding 0, 2048 samplers, late-bound. */
   VkDescriptorBindingFlags heap_flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                         VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                         VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
   VkDescriptorSetLayoutBindingFlagsCreateInfo heap_bf = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = 1,
      .pBindingFlags = &heap_flags};
   VkDescriptorSetLayoutBinding heap_binding = {.binding = 0,
                                                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                                                .descriptorCount = heap_size,
                                                .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS};
   VkDescriptorSetLayoutCreateInfo heap_li = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &heap_bf,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
      .bindingCount = 1,
      .pBindings = &heap_binding};
   VkDescriptorSetLayout heap_layout;
   CHECK(vkCreateDescriptorSetLayout(dev, &heap_li, NULL, &heap_layout));
   VkDescriptorSetLayoutBinding tex_binding = {.binding = 0,
                                               .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                               .descriptorCount = 1,
                                               .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
   VkDescriptorSetLayoutCreateInfo tex_li = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &tex_binding};
   VkDescriptorSetLayout tex_layout;
   CHECK(vkCreateDescriptorSetLayout(dev, &tex_li, NULL, &tex_layout));

   VkDescriptorPoolSize heap_ps = {VK_DESCRIPTOR_TYPE_SAMPLER, heap_size};
   VkDescriptorPoolCreateInfo heap_pi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &heap_ps};
   VkDescriptorPool heap_pool;
   CHECK(vkCreateDescriptorPool(dev, &heap_pi, NULL, &heap_pool));
   VkDescriptorSetVariableDescriptorCountAllocateInfo heap_vc = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
      .descriptorSetCount = 1,
      .pDescriptorCounts = &heap_size};
   VkDescriptorSetAllocateInfo heap_ai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = &heap_vc,
      .descriptorPool = heap_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &heap_layout};
   VkDescriptorSet heap_set;
   CHECK(vkAllocateDescriptorSets(dev, &heap_ai, &heap_set));

   uint32_t tex_set_count = texsets + noise + spare_sets;
   uint32_t per_pool = (tex_set_count + pools - 1) / pools;
   VkDescriptorPoolSize tex_ps = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, per_pool};
   VkDescriptorPoolCreateInfo tex_pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = per_pool,
                                        .poolSizeCount = 1,
                                        .pPoolSizes = &tex_ps};
   VkDescriptorSet *tex_sets = calloc(tex_set_count, sizeof(*tex_sets));
   for (uint32_t i = 0; i < tex_set_count; ++i) {
      static VkDescriptorPool tex_pool;
      if (i % per_pool == 0)
         CHECK(vkCreateDescriptorPool(dev, &tex_pi, NULL, &tex_pool));
      VkDescriptorSetAllocateInfo tex_ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                            .descriptorPool = tex_pool,
                                            .descriptorSetCount = 1,
                                            .pSetLayouts = &tex_layout};
      CHECK(vkAllocateDescriptorSets(dev, &tex_ai, &tex_sets[i]));
   }

   /* Samplers: distinct filter/address combinations so entries differ. */
   VkSampler *sampler_objs = calloc(samplers, sizeof(*sampler_objs));
   VkDescriptorImageInfo *sampler_infos = calloc(samplers, sizeof(*sampler_infos));
   for (uint32_t i = 0; i < samplers; ++i) {
      VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                .magFilter = VK_FILTER_NEAREST,
                                .minFilter = VK_FILTER_NEAREST,
                                .mipmapMode = (i & 1) ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                      : VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                .addressModeU = (i & 2) ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                                        : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                .addressModeV = (i & 4) ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                                        : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                .maxLod = (float)(i & 7)};
      CHECK(vkCreateSampler(dev, &si, NULL, &sampler_objs[i]));
      sampler_infos[i].sampler = sampler_objs[i];
   }
   VkWriteDescriptorSet heap_write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                      .dstSet = heap_set,
                                      .dstBinding = 0,
                                      .descriptorCount = samplers,
                                      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                                      .pImageInfo = sampler_infos};
   vkUpdateDescriptorSets(dev, 1, &heap_write, 0, NULL);
   VkDescriptorImageInfo tex_info = {.imageView = src_view,
                                     .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
   for (uint32_t i = 0; i < tex_set_count; ++i) {
      tex_info.imageView = src_views[i % images];
      VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = tex_sets[i],
                                .dstBinding = 0,
                                .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                .pImageInfo = &tex_info};
      vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
   }

   VkPushConstantRange range = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
   VkDescriptorSetLayout set_layouts[2] = {heap_layout, tex_layout};
   VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 2,
                                     .pSetLayouts = set_layouts,
                                     .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &range};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));

   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = load_module(dev, argv[1]),
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = load_module(dev, argv[2]),
       .pName = "main"}};
   VkPipelineVertexInputStateCreateInfo vin = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport viewport = {0, 0, (float)W, (float)H, 0, 1};
   VkRect2D scissor = {{0, 0}, {W, H}};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState att = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &att};
   VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                       .colorAttachmentCount = 1,
                                       .pColorAttachmentFormats = &color_format};
   VkGraphicsPipelineCreateInfo gpi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                       .pNext = &ri,
                                       .stageCount = 2,
                                       .pStages = stages,
                                       .pVertexInputState = &vin,
                                       .pInputAssemblyState = &ia,
                                       .pViewportState = &vp,
                                       .pRasterizationState = &rs,
                                       .pMultisampleState = &ms,
                                       .pColorBlendState = &cb,
                                       .layout = layout};
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipe));

   /* Immediate-path pipeline: one combined image sampler at a fixed binding. */
   VkPipeline imm_pipe = VK_NULL_HANDLE;
   VkPipelineLayout imm_layout = VK_NULL_HANDLE;
   VkDescriptorSet imm_set = VK_NULL_HANDLE;
   if (noise_imm) {
      VkDescriptorSetLayoutBinding cb_binding = {.binding = 0,
                                                 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                 .descriptorCount = 1,
                                                 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
      VkDescriptorSetLayoutCreateInfo cb_li = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = 1,
         .pBindings = &cb_binding};
      VkDescriptorSetLayout cb_layout;
      CHECK(vkCreateDescriptorSetLayout(dev, &cb_li, NULL, &cb_layout));
      VkDescriptorPoolSize cb_ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
      VkDescriptorPoolCreateInfo cb_pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                          .maxSets = 1,
                                          .poolSizeCount = 1,
                                          .pPoolSizes = &cb_ps};
      VkDescriptorPool cb_pool;
      CHECK(vkCreateDescriptorPool(dev, &cb_pi, NULL, &cb_pool));
      VkDescriptorSetAllocateInfo cb_ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                           .descriptorPool = cb_pool,
                                           .descriptorSetCount = 1,
                                           .pSetLayouts = &cb_layout};
      CHECK(vkAllocateDescriptorSets(dev, &cb_ai, &imm_set));
      VkDescriptorImageInfo cb_info = {sampler_objs[0], src_view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet cb_w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                   .dstSet = imm_set,
                                   .dstBinding = 0,
                                   .descriptorCount = 1,
                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                   .pImageInfo = &cb_info};
      vkUpdateDescriptorSets(dev, 1, &cb_w, 0, NULL);
      VkPipelineLayoutCreateInfo cb_pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = 1,
                                           .pSetLayouts = &cb_layout};
      CHECK(vkCreatePipelineLayout(dev, &cb_pli, NULL, &imm_layout));
      VkPipelineShaderStageCreateInfo imm_stages[2] = {
         stages[0],
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = load_module(dev, argv[3]),
          .pName = "main"}};
      VkGraphicsPipelineCreateInfo imm_gpi = gpi;
      imm_gpi.pStages = imm_stages;
      imm_gpi.layout = imm_layout;
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &imm_gpi, NULL, &imm_pipe));
   }

   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .queueFamilyIndex = qi};
   VkCommandPool cpool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cpool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                      .commandPool = cpool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));

   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   for (unsigned i = 0; i < images + 2; ++i) {
      VkImage image = i < images ? sources[i] : target[i - images];
      VkImageMemoryBarrier init = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT |
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, 0, NULL, 0, NULL, 1, &init);
   }
   VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                             .imageExtent = {W, H, 1}};
   for (unsigned i = 0; i < images; ++i) {
      VkBufferImageCopy in = copy;
      in.bufferOffset = bytes * i;
      vkCmdCopyBufferToImage(cmd, buffer, sources[i], VK_IMAGE_LAYOUT_GENERAL, 1, &in);
      image_barrier(cmd, sources[i], VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
   }

   for (uint32_t d = 0; d < draws; ++d) {
      for (uint32_t n = 0; n <= noise; ++n) {
         bool measured = n == noise;
         VkImage image = target[d & 1];
         VkRenderingAttachmentInfo color = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = target_view[d & 1],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
         VkRenderingInfo rinfo = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                  .renderArea = {{0, 0}, {W, H}},
                                  .layerCount = 1,
                                  .colorAttachmentCount = 1,
                                  .pColorAttachments = &color};
         vkCmdBeginRendering(cmd, &rinfo);
         if (!measured && noise_imm) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, imm_pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, imm_layout, 0, 1,
                                    &imm_set, 0, NULL);
         } else {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            VkDescriptorSet sets[2] = {heap_set,
                                       measured ? tex_sets[d % texsets] : tex_sets[texsets + n]};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, sets, 0,
                                    NULL);
            uint32_t push[4] = {measured ? sampler_index : (d + n) % samplers, 0, 0, 0};
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, push);
         }
         vkCmdDraw(cmd, 3, 1, 0, 0);
         vkCmdEndRendering(cmd);
         if (!measured) {
            image_barrier(cmd, image, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
            continue;
         }
         image_barrier(cmd, image, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
         VkBufferImageCopy out = copy;
         out.bufferOffset = bytes * (images + d);
         vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &out);
         image_barrier(cmd, image, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
      }
   }
   VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                           .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                        &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &cmd};
   CHECK(vkQueueSubmit(queue, 1, &submit, fence));
   uint64_t host_writes = 0;
   if (hostwrite) {
      /* Late-bound updates of entries no submitted work reads. */
      while (vkGetFenceStatus(dev, fence) == VK_NOT_READY) {
         uint32_t slot = samplers + host_writes % (heap_size - samplers);
         VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                   .dstSet = heap_set,
                                   .dstBinding = 0,
                                   .dstArrayElement = slot,
                                   .descriptorCount = 1,
                                   .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                                   .pImageInfo = &sampler_infos[host_writes % samplers]};
         vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
         VkWriteDescriptorSet w2 = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    .dstSet = tex_sets[texsets + noise + host_writes % spare_sets],
                                    .dstBinding = 0,
                                    .descriptorCount = 1,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                    .pImageInfo = &tex_info};
         vkUpdateDescriptorSets(dev, 1, &w2, 0, NULL);
         ++host_writes;
      }
   }
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_C(30000000000)));

   /* Compare every draw against the pattern; report wrong pixels, zero pixels
    * and whole 8x4 blocks (one wave each). */
   uint64_t wrong = 0, zero = 0, blocks = 0, zero_blocks = 0;
   uint32_t bad_draws = 0;
   for (uint32_t d = 0; d < draws; ++d) {
      const uint32_t *got = (const uint32_t *)(pixels + bytes * (images + d));
      const uint32_t *expected = pattern + ((d % texsets) % images) * W * H;
      uint64_t draw_wrong = 0;
      for (uint32_t by = 0; by < H / 4; ++by) {
         for (uint32_t bx = 0; bx < W / 8; ++bx) {
            unsigned bad = 0, zeros = 0;
            for (uint32_t y = by * 4; y < by * 4 + 4; ++y) {
               for (uint32_t x = bx * 8; x < bx * 8 + 8; ++x) {
                  uint32_t g = got[y * W + x];
                  if (g != expected[y * W + x]) {
                     ++bad;
                     zeros += g == 0;
                  }
               }
            }
            draw_wrong += bad;
            zero += zeros;
            blocks += bad == 32;
            zero_blocks += zeros == 32;
            if (bad && bad < 32 && blocks + bad_draws < 4)
               fprintf(stderr, "draw %u partial block at (%u,%u): %u wrong, got %08x want %08x\n",
                       d, bx * 8, by * 4, bad, got[by * 4 * W + bx * 8],
                       expected[by * 4 * W + bx * 8]);
         }
      }
      if (draw_wrong)
         fprintf(stderr, "draw %u image %u: %llu wrong pixels\n", d,
                 (d % texsets) % images, (unsigned long long)draw_wrong);
      wrong += draw_wrong;
      bad_draws += draw_wrong != 0;
   }
   printf("%ux%u draws=%u sampler=%u/%u texsets=%u noise=%u pools=%u hostwrites=%llu: "
          "wrong=%llu zero=%llu blocks=%llu zero_blocks=%llu bad_draws=%u %s\n",
          W, H, draws, sampler_index, samplers, texsets, noise, pools,
          (unsigned long long)host_writes, (unsigned long long)wrong,
          (unsigned long long)zero, (unsigned long long)blocks,
          (unsigned long long)zero_blocks, bad_draws, wrong ? "FAIL" : "PASS");
   return wrong ? 1 : 0;
}
