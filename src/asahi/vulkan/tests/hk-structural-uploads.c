/* SPDX-License-Identifier: MIT
 * Exercise actual render/view ownership with host-backed command BOs. */
#include "../hk_cmd_apple9.c"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)

static struct agx_bo *bos[128];
static unsigned allocations, live_bos, view_cases;
struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned, enum agx_bo_flags, const char *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
void __wrap_agx_bo_cache_evict_all(struct agx_device *);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned alignment,
                     enum agx_bo_flags flags, const char *label)
{
   CHECK(allocations + 1 < ARRAY_SIZE(bos));
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, size);
   CHECK(bo->va && bo->_map);
   bo->dev = dev; bo->size = size; bo->refcnt = 1;
   bo->va->addr = (++allocations) * (uint64_t)HK_CMD_BO_SIZE;
   bos[allocations] = bo;
   ++live_bos;
   return bo;
}
void __wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo) return;
   CHECK(bo->refcnt == 1);
   bos[bo->va->addr / HK_CMD_BO_SIZE] = NULL;
   free(bo->_map); free(bo->va); free(bo); --live_bos;
}
void __wrap_agx_bo_cache_evict_all(struct agx_device *dev) { }
static void *mapped(uint64_t gpu)
{
   for (unsigned i = 1; i <= allocations; ++i)
      if (bos[i] && gpu >= bos[i]->va->addr && gpu < bos[i]->va->addr + bos[i]->size)
         return (uint8_t *)bos[i]->_map + gpu - bos[i]->va->addr;
   CHECK(false);
   return NULL;
}

static void
check_view(struct hk_device *dev, VkFormat format, unsigned samples,
           unsigned level, unsigned layers, bool linear, bool compressed)
{
   const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = {127, 65, 1}, .mipLevels = samples > 1 || linear ? 1 : 4,
      .arrayLayers = layers + 1, .samples = samples,
      .tiling = linear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   struct hk_image image = {0};
   vk_image_init(&dev->vk, &image.vk, &ci);
   image.plane_count = 1;
   image.planes[0].addr = UINT64_C(0x2000000000);
   struct ail_layout *layout = &image.planes[0].layout;
   *layout = (struct ail_layout){
      .width_px = ci.extent.width, .height_px = ci.extent.height,
      .depth_px = ci.arrayLayers, .levels = ci.mipLevels,
      .sample_count_sa = samples, .format = vk_format_to_pipe_format(format),
      .tiling = linear ? AIL_TILING_LINEAR : AIL_TILING_GPU,
      .renderable = true, .compressed = compressed,
   };
   ail_make_miptree(layout);
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = hk_image_to_handle(&image),
      .viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 1, layers},
   };
   VkImageView handle;
   CHECK(hk_CreateImageView(hk_device_to_handle(dev), &vi, NULL, &handle) == VK_SUCCESS);
   struct hk_image_view *view = hk_image_view_from_handle(handle);
   for (unsigned layered = 0; layered < 2; ++layered) {
      const struct hk_apple9_render_view *owned = &view->apple9_render[layered];
      agx_unpack(stderr, layered ? &view->planes[0].layered_background : &view->planes[0].background,
                  TEXTURE, legacy);
      CHECK(owned->dimensions[0] == u_minify(legacy.width, legacy.first_level));
      CHECK(owned->dimensions[1] == u_minify(legacy.height, legacy.first_level));
      CHECK(owned->dimensions[2] == legacy.depth);
      CHECK(owned->dimensions[3] == 1);
      CHECK(owned->array == (legacy.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
                             legacy.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED));
      legacy.srgb = legacy.srgb_2_channel = false;
      struct agx_texture_packed packed;
      agx_pack(&packed, TEXTURE, cfg) { cfg = legacy; }
      uint8_t expected[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
      bool valid = agx_apple9_translate_texture(expected, &packed);
      CHECK(owned->texture_valid == valid);
      if (valid) CHECK(!memcmp(owned->texture, expected, sizeof(expected)));
      valid = hk_apple9_image_view_eot(view, layered, expected);
      CHECK(owned->store_valid == valid);
      if (valid) CHECK(!memcmp(owned->store, expected, sizeof(expected)));
   }
   hk_DestroyImageView(hk_device_to_handle(dev), handle, NULL);
   vk_image_finish(&image.vk);
   ++view_cases;
}

int main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->vk.dispatch_table.AllocateCommandBuffers = vk_common_AllocateCommandBuffers;
   dev->vk.dispatch_table.ResetCommandBuffer = hk_ResetCommandBuffer;
   dev->dev.chip = AGX_CHIP_G16G;
   hk_device_init_usc_heap(dev);
   const VkFormat formats[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB,
      VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
      VK_FORMAT_R32G32B32A32_UINT};
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (unsigned layers = 1; layers <= 3; layers += 2) {
         for (unsigned samples = 1; samples <= 4; samples *= 2) {
            unsigned levels = samples == 1 ? 4 : 1;
            for (unsigned l = 0; l < levels; ++l)
               check_view(dev, formats[f], samples, l, layers, false, false);
         }
         check_view(dev, formats[f], 1, 0, layers, true, false);
      }
   }
   for (unsigned samples = 1; samples <= 4; samples *= 2)
      check_view(dev, VK_FORMAT_R8G8B8A8_UNORM, samples, 0, 3, false, true);

   VkDevice device = hk_device_to_handle(dev);
   const VkCommandPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool;
   CHECK(hk_CreateCommandPool(device, &ci, NULL, &pool) == VK_SUCCESS);
   VkCommandBuffer cb;
   const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .commandBufferCount = 1,
   };
   CHECK(vk_common_AllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS);
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(cb);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_image_view view = {0};
   uint32_t dimensions[4] = {127, 65, 3, 1};
   memcpy(view.apple9_render[1].dimensions, dimensions, sizeof(dimensions));
   VkRenderingAttachmentInfo att[3] = {0};
   att[0].clearValue.color = (VkClearColorValue){.uint32 = {1, 2, 3, 4}};
   att[2].clearValue.color = (VkClearColorValue){.uint32 = {5, 6, 7, 8}};
   VkRenderingInfo info = {.colorAttachmentCount = 3, .pColorAttachments = att};
   render->color_att_count = 3;
   render->color_att[0].iview = render->color_att[2].iview = &view;
   render->tilebuffer.nr_samples = 4;
   render->cr.layers = 3;
   CHECK(hk_apple9_render_inputs(cmd, &info));
   uint64_t params = render->apple9_arguments, sysvals = render->apple9_sysvals;
   uint32_t *values = mapped(params);
   CHECK(!memcmp(values + 16, &att[0].clearValue.color, 16));
   CHECK(!memcmp(values + 24, &att[2].clearValue.color, 16));
   CHECK(values[20] == 0);
   uint8_t *metadata = mapped(sysvals);
   for (unsigned rt = 0; rt < 3; ++rt) {
      uint32_t expected[5] = {127, 65, 3, 1, 4};
      if (rt == 1) memset(expected, 0, sizeof(expected));
      CHECK(!memcmp(metadata + AGX_APPLE9_TEXTURE_INFO_OFFSET +
                       rt * AGX_APPLE9_TEXTURE_INFO_STRIDE, expected, sizeof(expected)));
   }
   struct hk_upload_chunk saved[HK_UPLOAD_CHUNKS];
   memcpy(saved, cmd->uploader[HK_CMD_POOL_MAIN].chunks, sizeof(saved));
   for (unsigned i = 0; i < 12; ++i) {
      CHECK(hk_apple9_render_inputs(cmd, &info));
      CHECK(render->apple9_arguments == params && render->apple9_sysvals == sysvals);
      CHECK(!memcmp(saved, cmd->uploader[HK_CMD_POOL_MAIN].chunks, sizeof(saved)));
   }
   /* A new render owns new inputs; previous recording bytes stay immutable. */
   render->apple9_arguments = render->apple9_sysvals = 0;
   att[0].clearValue.color.uint32[0] = 99;
   CHECK(hk_apple9_render_inputs(cmd, &info));
   CHECK(render->apple9_arguments != params && render->apple9_sysvals != sysvals);
   CHECK(((uint32_t *)mapped(params))[16] == 1);
   CHECK(((uint32_t *)mapped(render->apple9_arguments))[16] == 99);
   hk_DestroyCommandPool(device, pool, NULL);
   hk_device_finish_usc_heap(dev);
   CHECK(live_bos == 0);
   free(pdev); free(dev);
   printf("PASS %u render view cases; per-render input reuse, new-render separation and lifetime\n", view_cases);
}
