/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_apple9_layout.h"
#include "agx_pack.h"
#include "hk_descriptor_memory.h"
#include "hk_descriptor_set.h"
#include "hk_private.h"
#include "vk_image.h"

struct hk_device;
struct hk_image;

#define HK_MAX_PLANES      3
#define HK_MAX_IMAGE_DESCS (10 * HK_MAX_PLANES)

struct hk_image_view {
   struct vk_image_view vk;

   /* A suballocation in the device's immutable native PBE table. */
   uint32_t apple9_store_flags, apple9_pbe_index;
   struct hk_descriptor_allocation apple9_pbes;

   struct hk_apple9_render_view {
      uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
      uint8_t store[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
      uint32_t dimensions[4];
      bool array, texture_valid, store_valid;
   } apple9_render[2]; /* single layer, layered */

   uint8_t plane_count;
   struct {
      uint8_t image_plane;

      /* Logical width in blocks of an uncompressed alias of a compressed level
       * whose descriptor carries the padded stride instead, else 0.
       */
      uint32_t alias_width_el;

      struct agx_texture_packed ia, sampled, ro_storage, background,
         layered_background, emrt_texture;

      struct agx_pbe_packed storage, eot, layered_eot, emrt_pbe;

      /* Immutable descriptor bodies. Residency is resolved at the write,
       * since image memory can be bound after creating a view. */
      struct hk_apple9_image_part apple9_sampled, apple9_ia;
   } planes[3];
   struct hk_apple9_storage_descriptor apple9_storage;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_image_view, vk.base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)

/* Address of the struct hk_apple9_residency table for a plane of the view. */
uint64_t hk_apple9_view_residency(const struct hk_device *dev,
                                  const struct hk_image_view *view,
                                  unsigned plane);

/* Encode an uncompressed native tile-store view for render submission. */
bool hk_apple9_image_view_eot(const struct hk_image_view *view, bool layered,
                            bool raw, void *out);

/* Image memory must be dense and uncompressed for native storage writes. */
bool hk_apple9_native_store_image(const struct hk_image *image, unsigned plane);
bool hk_apple9_native_store_view_layout(const struct ail_layout *layout,
                                        enum pipe_format view_format);

/* Native writes address an individual 2D plane of the storage PBE. */
bool hk_apple9_translate_storage_pbe(void *out, const struct agx_pbe_packed *state,
                                     enum pipe_format format, const struct ail_layout *layout,
                                     uint64_t base, unsigned level);
