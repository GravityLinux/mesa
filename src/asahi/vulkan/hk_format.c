/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "drm-uapi/drm_fourcc.h"
#include "vulkan/vulkan_core.h"

#include "hk_buffer_view.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include "vk_enum_defines.h"
#include "vk_format.h"

/* Storage formats on the Apple9 path: the SPIR-V/Vulkan image-format
 * compatibility table, plus B8G8R8A8_UNORM, which has no SPIR-V format and is
 * only reached formatless. Formatless accesses decode and encode through the
 * descriptor's per-component texel layout, which follows the format swizzle,
 * and native stores through the PBE, which renders BGRA swapchains.
 * https://docs.vulkan.org/spec/latest/chapters/formats.html
 */
bool
hk_apple9_storage_format_supported(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   /* Packed ABGR has the same component representation as RGBA8 on this
    * little-endian device and is compatible with the same SPIR-V formats.
    */
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
   /* Swapchain format; compute writers ask for STORAGE swapchains. */
   case VK_FORMAT_B8G8R8A8_UNORM:
      return true;
   default:
      return false;
   }
}

uint64_t agx_best_modifiers[] = {
   DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
   DRM_FORMAT_MOD_APPLE_GPU_TILED,
   DRM_FORMAT_MOD_LINEAR,
};

static VkFormatFeatureFlags2
hk_modifier_features(const struct agx_device *dev, uint64_t mod,
                     VkFormat vk_format, const VkFormatProperties *props)
{
   /* There's no corresponding fourcc, so don't advertise modifiers */
   if (vk_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
       vk_format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) {
      return 0;
   }

   /* Don't advertise compression for the uncompressable */
   if (mod == DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED &&
       !hk_can_compress_format(dev, vk_format))
      return 0;

   if (mod == DRM_FORMAT_MOD_LINEAR)
      return props->linearTilingFeatures;

   VkFormatFeatureFlags2 features = props->optimalTilingFeatures;

   /* Compressed images are never storage images (ail writeable_image). */
   if (mod == DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED)
      features &= ~(VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
                    VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT |
                    VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                    VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT);

   return features;
}

static void
get_drm_format_modifier_properties_list(
   const struct hk_physical_device *physical_device, VkFormat vk_format,
   VkDrmFormatModifierPropertiesListEXT *list, const VkFormatProperties *props)
{
   VK_OUTARRAY_MAKE_TYPED(VkDrmFormatModifierPropertiesEXT, out,
                          list->pDrmFormatModifierProperties,
                          &list->drmFormatModifierCount);

   for (unsigned i = 0; i < ARRAY_SIZE(agx_best_modifiers); ++i) {
      uint64_t mod = agx_best_modifiers[i];
      VkFormatFeatureFlags2 flags =
         hk_modifier_features(&physical_device->dev, mod, vk_format, props);

      if (!flags)
         continue;

      vk_outarray_append_typed(VkDrmFormatModifierPropertiesEXT, &out,
                               out_props)
      {
         *out_props = (VkDrmFormatModifierPropertiesEXT){
            .drmFormatModifier = mod,
            .drmFormatModifierPlaneCount = vk_format_get_plane_count(vk_format),
            .drmFormatModifierTilingFeatures = flags,
         };
      };
   }
}

static void
get_drm_format_modifier_properties_list_2(
   const struct hk_physical_device *physical_device, VkFormat vk_format,
   VkDrmFormatModifierPropertiesList2EXT *list, const VkFormatProperties *props)
{
   VK_OUTARRAY_MAKE_TYPED(VkDrmFormatModifierProperties2EXT, out,
                          list->pDrmFormatModifierProperties,
                          &list->drmFormatModifierCount);

   for (unsigned i = 0; i < ARRAY_SIZE(agx_best_modifiers); ++i) {
      uint64_t mod = agx_best_modifiers[i];
      VkFormatFeatureFlags2 flags =
         hk_modifier_features(&physical_device->dev, mod, vk_format, props);

      if (!flags)
         continue;

      vk_outarray_append_typed(VkDrmFormatModifierProperties2EXT, &out,
                               out_props)
      {
         *out_props = (VkDrmFormatModifierProperties2EXT){
            .drmFormatModifier = mod,
            .drmFormatModifierPlaneCount = vk_format_get_plane_count(vk_format),
            .drmFormatModifierTilingFeatures = flags,
         };
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice,
                                      VkFormat format,
                                      VkFormatProperties2 *pFormatProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdevice, physicalDevice);

   VkFormatFeatureFlags2 linear2, optimal2, buffer2;
   linear2 =
      hk_get_image_format_features(pdevice, format, VK_IMAGE_TILING_LINEAR);
   optimal2 =
      hk_get_image_format_features(pdevice, format, VK_IMAGE_TILING_OPTIMAL);
   buffer2 = hk_get_buffer_format_features(pdevice, format);

   pFormatProperties->formatProperties = (VkFormatProperties){
      .linearTilingFeatures = vk_format_features2_to_features(linear2),
      .optimalTilingFeatures = vk_format_features2_to_features(optimal2),
      .bufferFeatures = vk_format_features2_to_features(buffer2),
   };

   vk_foreach_struct(ext, pFormatProperties->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3: {
         VkFormatProperties3 *p = (void *)ext;
         p->linearTilingFeatures = linear2;
         p->optimalTilingFeatures = optimal2;
         p->bufferFeatures = buffer2;
         break;
      }

      case VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT:
         get_drm_format_modifier_properties_list(
            pdevice, format, (void *)ext, &pFormatProperties->formatProperties);
         break;

      case VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT:
         get_drm_format_modifier_properties_list_2(
            pdevice, format, (void *)ext, &pFormatProperties->formatProperties);
         break;

      default:
         vk_debug_ignored_stype(ext->sType);
         break;
      }
   }
}
