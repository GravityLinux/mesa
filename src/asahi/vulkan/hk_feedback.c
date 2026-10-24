/* SPDX-License-Identifier: MIT */
#include "hk_feedback.h"
#include "asahi/lib/agx_apple9.h"
#include "util/format/u_format.h"
#include "hk_cmd_buffer.h"
#include "hk_image.h"
#include "hk_image_view.h"

/* Repartitioning compressed channels is not a bitcast. Keep the compression
 * class as well as the texel size. Packed minifloats have no integer view and
 * retain the feedback compression ban rather than canonicalizing their NaNs.
 */
VkFormat
hk_feedback_raw_format(VkFormat format)
{
   enum pipe_format pf = hk_format_to_pipe_format(format);
   if (!ail_is_valid_pixel_format(pf) || util_format_is_depth_or_stencil(pf) ||
       util_format_is_compressed(pf))
      return VK_FORMAT_UNDEFINED;
   switch (ail_pixel_format[pf].channels) {
   case AGX_CHANNELS_R8:
      return VK_FORMAT_R8_UINT;
   case AGX_CHANNELS_R16:
      return VK_FORMAT_R16_UINT;
   case AGX_CHANNELS_R8G8:
      return VK_FORMAT_R8G8_UINT;
   case AGX_CHANNELS_R32:
      return VK_FORMAT_R32_UINT;
   case AGX_CHANNELS_R16G16:
      return VK_FORMAT_R16G16_UINT;
   case AGX_CHANNELS_R8G8B8A8:
      return VK_FORMAT_R8G8B8A8_UINT;
   case AGX_CHANNELS_R10G10B10A2:
      return VK_FORMAT_A2B10G10R10_UINT_PACK32;
   case AGX_CHANNELS_R32G32:
      return VK_FORMAT_R32G32_UINT;
   case AGX_CHANNELS_R16G16B16A16:
      return VK_FORMAT_R16G16B16A16_UINT;
   case AGX_CHANNELS_R32G32B32A32:
      return VK_FORMAT_R32G32B32A32_UINT;
   default:
      return VK_FORMAT_UNDEFINED;
   }
}

bool
hk_feedback_can_compress(const struct agx_device *dev, VkImageType type,
                         VkFormat format, unsigned samples)
{
   return dev->chip == AGX_CHIP_G16G && type == VK_IMAGE_TYPE_2D &&
          samples == 1 && hk_feedback_raw_format(format) != VK_FORMAT_UNDEFINED;
}

struct hk_feedback_raw {
   const struct hk_image *image;
   unsigned level, layer;
   bool raw;
};

bool
hk_feedback_is_raw(struct hk_cmd_buffer *cmd, const struct hk_image *image,
                   unsigned level, unsigned layer)
{
   util_dynarray_foreach(&cmd->feedback_raw, struct hk_feedback_raw, entry) {
      if (entry->image == image && entry->level == level &&
          entry->layer == layer)
         return entry->raw;
   }
   return false;
}

bool
hk_feedback_mark_raw(struct hk_cmd_buffer *cmd, const struct hk_image *image,
                     unsigned level, unsigned layer)
{
   /* A raw store through another image alias can overwrite this image's
    * differently partitioned compression metadata. Do not reuse its state.
    */
   util_dynarray_foreach(&cmd->feedback_raw, struct hk_feedback_raw, entry) {
      if (entry->image != image &&
          entry->image->planes[0].bo == image->planes[0].bo)
         entry->raw = false;
   }
   util_dynarray_foreach(&cmd->feedback_raw, struct hk_feedback_raw, entry) {
      if (entry->image == image && entry->level == level &&
          entry->layer == layer) {
         entry->raw = true;
         return true;
      }
   }
   struct hk_feedback_raw *entry =
      util_dynarray_grow(&cmd->feedback_raw, struct hk_feedback_raw, 1);
   if (!entry) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }
   *entry = (struct hk_feedback_raw){image, level, layer, true};
   return true;
}

/* Cache only commands before the current recording point, never global image
 * state. Each CB starts unknown. Re-submission repeats its first decompression
 * and every subsequent compressed-write invalidation, in recorded order.
 */
void
hk_feedback_invalidate(struct hk_cmd_buffer *cmd)
{
   util_dynarray_clear(&cmd->feedback_raw);
}

void
hk_feedback_invalidate_range(struct hk_cmd_buffer *cmd,
                             const struct hk_image *image, unsigned first_level,
                             unsigned levels, unsigned first_layer,
                             unsigned layers)
{
   util_dynarray_foreach(&cmd->feedback_raw, struct hk_feedback_raw, entry) {
      if (entry->image == image) {
         if (entry->level >= first_level &&
             entry->level - first_level < levels &&
             entry->layer >= first_layer && entry->layer - first_layer < layers)
            entry->raw = false;
      } else if (entry->image->planes[0].bo == image->planes[0].bo) {
         /* Image aliases can have different mip/layer partitions. */
         entry->raw = false;
      }
   }
}

void
hk_feedback_invalidate_view(struct hk_cmd_buffer *cmd,
                            const struct hk_image_view *view)
{
   if (!view)
      return;
   const struct hk_image *image =
      container_of(view->vk.image, struct hk_image, vk);
   hk_feedback_invalidate_range(cmd, image, view->vk.base_mip_level, 1,
                                view->vk.base_array_layer,
                                view->vk.layer_count);
}
