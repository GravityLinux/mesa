/* SPDX-License-Identifier: MIT */
#pragma once
#include "hk_private.h"
struct hk_cmd_buffer;
struct hk_image;
struct hk_image_view;
struct agx_device;
VkFormat hk_feedback_raw_format(VkFormat format);
bool hk_feedback_is_zs_format(VkFormat format);
bool hk_feedback_can_compress(const struct agx_device *dev, VkImageType type,
                              VkFormat format, unsigned samples);
bool hk_feedback_is_raw(struct hk_cmd_buffer *cmd, const struct hk_image *image,
                        unsigned level, unsigned layer);
bool hk_feedback_mark_raw(struct hk_cmd_buffer *cmd,
                          const struct hk_image *image, unsigned level,
                          unsigned layer);
void hk_feedback_invalidate(struct hk_cmd_buffer *cmd);
void hk_feedback_invalidate_range(struct hk_cmd_buffer *cmd,
                                  const struct hk_image *image,
                                  unsigned first_level, unsigned levels,
                                  unsigned first_layer, unsigned layers);
void hk_feedback_invalidate_view(struct hk_cmd_buffer *cmd,
                                 const struct hk_image_view *view);
bool hk_meta_decompress_view(struct hk_cmd_buffer *cmd,
                             const struct hk_image_view *view, unsigned layer);
