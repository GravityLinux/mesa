/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "hk_entrypoints.h"
#include "hk_physical_device.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); return 1; } } while (0)

int main(void)
{
   struct hk_physical_device pdev = {0};
   vk_object_base_init(NULL, &pdev.vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   const enum agx_chip chips[] = {AGX_CHIP_G13G, AGX_CHIP_G16G};
   const VkImageType types[] = {VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D};
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      pdev.dev.chip = chips[c];
      for (unsigned t = 0; t < ARRAY_SIZE(types); ++t) {
         VkPhysicalDeviceImageFormatInfo2 info = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .type = types[t],
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         };
         VkImageFormatProperties2 props = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
         };
         CHECK(hk_GetPhysicalDeviceImageFormatProperties2(
            hk_physical_device_to_handle(&pdev), &info, &props) == VK_SUCCESS);
         const VkImageFormatProperties *p = &props.imageFormatProperties;
         unsigned max = chips[c] == AGX_CHIP_G16G && types[t] == VK_IMAGE_TYPE_3D ? 2048 : 16384;
         CHECK(p->maxExtent.width == max);
         CHECK(p->maxExtent.height == (types[t] == VK_IMAGE_TYPE_1D ? 1 : max));
         CHECK(p->maxExtent.depth == (types[t] == VK_IMAGE_TYPE_3D ? max : 1));
         CHECK(p->maxMipLevels == (max == 2048 ? 12 : 15));
         CHECK(p->maxArrayLayers == (types[t] == VK_IMAGE_TYPE_3D ? 1 : 2048));
      }
   }
   puts("G16 3D format limits match native depth; other image ranges retained");

   /* Sparse residency: 2D images only, matching the advertised features. */
   pdev.dev.chip = AGX_CHIP_G16G;
   pdev.vk.supported_features.sparseBinding = true;
   pdev.vk.supported_features.sparseResidencyImage2D = true;
   pdev.vk.supported_features.sparseResidencyImage3D = false;
   for (unsigned t = 0; t < ARRAY_SIZE(types); ++t) {
      VkPhysicalDeviceImageFormatInfo2 info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .type = types[t],
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
         .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                  VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
      };
      VkImageFormatProperties2 props = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      };
      VkResult r = hk_GetPhysicalDeviceImageFormatProperties2(
         hk_physical_device_to_handle(&pdev), &info, &props);
      CHECK((r == VK_SUCCESS) == (types[t] == VK_IMAGE_TYPE_2D));

      VkPhysicalDeviceSparseImageFormatInfo2 sparse = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2,
         .format = info.format,
         .type = info.type,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .usage = info.usage,
         .tiling = info.tiling,
      };
      uint32_t count = 0;
      hk_GetPhysicalDeviceSparseImageFormatProperties2(
         hk_physical_device_to_handle(&pdev), &sparse, &count, NULL);
      CHECK(count == (types[t] == VK_IMAGE_TYPE_2D ? 1 : 0));
   }
   puts("Sparse residency format queries report 2D only");
   return 0;
}
