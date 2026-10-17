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
   return 0;
}
