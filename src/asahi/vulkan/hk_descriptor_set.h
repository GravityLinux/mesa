/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "hk_device.h"
#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/lib/agx_apple9.h"
#include "vk_descriptor_update_template.h"
#include "vk_object.h"

#include "util/list.h"
#include "util/vma.h"

struct hk_descriptor_set_layout;

struct hk_sampled_image_descriptor {
   struct agx_texture_packed tex;
   struct agx_sampler_packed sampler;
   uint16_t sampler_index;

   /* Negative if there is no border colour, else the clamp=0 sampler index used
    * for custom border colour emulation.
    */
   int16_t clamp_0_sampler_index_or_negative;

   /* We want LOD bias and min LOD together as fp16 in this order and 32-bit
    * aligned. This lets us keep everything vectorized on the shader side, since
    * this matches the AGX2 lod_bias + min_lod descriptor.
    */
   uint16_t lod_bias_fp16;
   uint16_t min_lod_fp16;

   /* This is only used for txf, it can be separate from the other stuff */
   uint16_t min_lod_uint16;

   /* Added to explicit LODs, see hk_lower_d3d_txl_rounding */
   uint16_t txl_nudge_fp16;
   union {
      /* TODO: This should probably be a heap! */
      uint32_t border[4];

      /* Apple9 has no custom border colors. Instead, the view records the
       * residency table of its image and its base subresource.
       */
      struct {
         uint32_t residency_lo, residency_hi;
         uint16_t base_level, base_layer;
         uint32_t pad;
      } apple9;
   };
   uint8_t pad2[4];
};
static_assert(sizeof(struct hk_sampled_image_descriptor) == 64,
              "hk_sampled_image_descriptor has no holes");

struct hk_storage_image_descriptor {
   struct agx_texture_packed tex;
   struct agx_pbe_packed pbe;
};
static_assert(sizeof(struct hk_storage_image_descriptor) == 48,
              "hk_storage_image_descriptor has no holes");

/* GPU-readable format metadata for Apple9 uniform texel-buffer loads. The
 * ordinary texture prefix remains available for descriptor queries.
 * A component word contains byte offset [7:0], bit shift [12:8], width minus
 * one [17:13], and conversion kind [21:18]. Swizzles are resolved per component.
 */
enum hk_apple9_texel_kind {
   HK_TEXEL_ZERO, HK_TEXEL_ONE, HK_TEXEL_UINT, HK_TEXEL_SINT,
   HK_TEXEL_UNORM, HK_TEXEL_SNORM, HK_TEXEL_FLOAT16, HK_TEXEL_FLOAT32,
   HK_TEXEL_FLOAT11, HK_TEXEL_RGB9E5, HK_TEXEL_SRGB,
};

/* Storage images have no base or element count. They instead record the
 * residency table of their image and the base level | base layer << 16.
 */
struct hk_apple9_texel_metadata {
   uint64_t base;
   uint32_t elements;
   uint32_t size_B;
   uint32_t component[4];
   /* Sparse residency buffers: the residency byte of the page holding
    * address A is at residency + A / AIL_PAGESIZE, see
    * hk_buffer::apple9_residency. Zero for other resources. */
   uint64_t residency;
};

/* Uniform texel buffers read every format but RGB32 through the texture
 * unit. Like hk_apple9_image_descriptor, the descriptor holds the native
 * texture entry for shaders that address it directly. */
struct hk_apple9_texel_descriptor {
   struct agx_texture_packed tex;
   struct hk_apple9_texel_metadata metadata;
   uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
};
static_assert(sizeof(struct hk_apple9_texel_descriptor) == 96,
              "Apple9 texel descriptor has no holes");
static_assert(offsetof(struct hk_apple9_texel_descriptor, texture) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native texture entry is where register indices expect it");

/* Apple9 storage images and texel buffers append their texel layout, so
 * shaders can access them without a declared format. Storage images also
 * hold the native texture entry of their read-only view: loads that no
 * write of the invocation can precede go through the texture unit. The
 * entry is one table slot past where hk_apple9_image_descriptor keeps its
 * own, so a handle 32 bytes past the descriptor addresses it. */
struct hk_apple9_storage_descriptor {
   struct hk_storage_image_descriptor common;
   struct hk_apple9_texel_metadata metadata;
   /* Nonzero when texture holds a valid native entry. */
   uint32_t native_ok;
   /* enum hk_apple9_store_encode: how a store without a shader format can
    * encode the texel, chosen on the host from the view format. */
   uint32_t store_encode;
   uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
};
static_assert(sizeof(struct hk_apple9_storage_descriptor) == 128,
              "Apple9 storage descriptor has no holes");
static_assert(offsetof(struct hk_apple9_storage_descriptor, texture) ==
                 (AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY + 1) *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native storage entry is one table slot past the image one");

void hk_apple9_texel_layout(enum pipe_format format,
                            struct hk_apple9_texel_metadata *metadata);

/* Encodings for stores to storage images without a shader format. A
 * canonical format packs its N channels in order, each W bits wide from bit
 * 0, so stores can encode it with constant shifts instead of decoding the
 * per-component layout. Everything else uses the layout (GENERIC).
 */
enum hk_apple9_store_encode {
   HK_STORE_ENCODE_GENERIC = 0,
   /* 32-bit channels: the texel's bits as they are (float, uint, sint). */
   HK_STORE_ENCODE_RAW32,
   /* 8-bit unsigned normalized channels. */
   HK_STORE_ENCODE_UNORM8,
   /* 16-bit float channels. */
   HK_STORE_ENCODE_FLOAT16,
};

enum hk_apple9_store_encode hk_apple9_store_encode(enum pipe_format format);

/* G16 image and sampler descriptors extend the common layout with native
 * state that shaders can address directly. Descriptor memory lives in the
 * USC aperture, so a 32-byte-aligned record at VA v is native table entry
 * (v - shader_base) / 32 when a stage's table pointer is shader_base.
 */
struct hk_apple9_image_descriptor {
   struct hk_sampled_image_descriptor common;
   uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   uint8_t sampler[AGX_APPLE9_SAMPLER_TABLE_STRIDE];
   /* Same layouts as the per-draw texture info and runtime sampler records. */
   uint32_t texture_info[AGX_APPLE9_TEXTURE_INFO_STRIDE / 4];
   uint32_t runtime_sampler[AGX_APPLE9_RUNTIME_SAMPLER_STRIDE / 4];
};
static_assert(sizeof(struct hk_apple9_image_descriptor) == 224,
              "Apple9 image descriptor has no holes");
static_assert(offsetof(struct hk_apple9_image_descriptor, texture) % 32 == 0,
              "native texture entry is table aligned");
static_assert(offsetof(struct hk_apple9_image_descriptor, sampler) % 32 == 0,
              "native sampler entry is table aligned");

/* G16 descriptor records, 128 bytes each so translation layers can copy
 * them with their 128-byte paths: an image part and a sampler part. A
 * combined image sampler is the image part followed by the sampler part.
 * Shaders address each part through its own handle; the native entries keep
 * the table slots AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY and
 * AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY past their part's base, as the texel
 * and storage descriptors do.
 */
struct hk_apple9_image_part {
   /* Same layout as the per-draw texture info record. */
   uint32_t texture_info[AGX_APPLE9_TEXTURE_INFO_STRIDE / 4];
   /* Sparse residency table and the view's base level and layer. */
   uint32_t residency_lo, residency_hi;
   uint16_t base_level, base_layer;
   uint16_t min_lod_fp16, min_lod_uint16;
   uint8_t pad0[16];
   uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE];
   uint8_t pad1[32];
};
static_assert(sizeof(struct hk_apple9_image_part) == 128, "128-byte record");
static_assert(offsetof(struct hk_apple9_image_part, texture_info) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_INFO,
              "texture info where shaders read it");
static_assert(offsetof(struct hk_apple9_image_part, texture) ==
                 AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY *
                    AGX_APPLE9_TEXTURE_TABLE_STRIDE,
              "native texture entry is where register indices expect it");

struct hk_apple9_sampler_part {
   /* Same layout as the per-draw runtime sampler record. */
   uint32_t runtime_sampler[AGX_APPLE9_RUNTIME_SAMPLER_STRIDE / 4];
   uint16_t lod_bias_fp16, txl_nudge_fp16;
   uint8_t pad0[28];
   uint8_t sampler[AGX_APPLE9_SAMPLER_TABLE_STRIDE];
};
static_assert(sizeof(struct hk_apple9_sampler_part) == 128, "128-byte record");
static_assert(offsetof(struct hk_apple9_sampler_part, runtime_sampler) ==
                 AGX_APPLE9_DESCRIPTOR_RUNTIME_SAMPLER,
              "runtime sampler where shaders read it");
static_assert(offsetof(struct hk_apple9_sampler_part, sampler) ==
                 AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY *
                    AGX_APPLE9_SAMPLER_TABLE_STRIDE,
              "native sampler entry is where register indices expect it");

/* A combined image sampler is an image part followed by a sampler part; a
 * multi-plane descriptor repeats the whole record per plane. */
#define HK_APPLE9_COMBINED_DESCRIPTOR_SIZE                                    \
   (sizeof(struct hk_apple9_image_part) + sizeof(struct hk_apple9_sampler_part))
#define HK_MAX_SAMPLED_DESCRIPTOR_SIZE (3 * HK_APPLE9_COMBINED_DESCRIPTOR_SIZE)

void hk_apple9_texture_info(uint32_t *data, const struct agx_texture_packed *state);
bool hk_apple9_runtime_sampler(uint32_t *data, const struct agx_sampler_packed *state,
                               bool clamp_to_edge);
/* A custom border sampler takes its packed colour from border. */
bool hk_apple9_native_sampler(void *out, const struct agx_sampler_packed *state,
                              bool clamp_to_edge, const uint32_t *border);

/* This has to match nir_address_format_64bit_bounded_global */
struct hk_buffer_address {
   uint64_t base_addr;
   uint32_t size;
   uint32_t zero; /* Must be zero! */
};

struct hk_device;
struct hk_descriptor_set_layout;
void hk_descriptor_pack_embedded_samplers(
   struct hk_device *dev, const struct hk_descriptor_set_layout *layout,
   void *map);

struct hk_descriptor_pool {
   struct vk_object_base base;

   struct list_head sets;

   struct agx_bo *bo;
   uint8_t *mapped_ptr;
   struct util_vma_heap heap;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)

struct hk_descriptor_set {
   struct vk_object_base base;

   /* Link in hk_descriptor_pool::sets */
   struct list_head link;

   struct hk_descriptor_set_layout *layout;
   void *mapped_ptr;
   uint64_t addr;
   uint32_t size;

   struct hk_buffer_address dynamic_buffers[];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)

static inline uint64_t
hk_descriptor_set_addr(const struct hk_descriptor_set *set)
{
   return set->addr;
}

struct hk_push_descriptor_set {
   uint8_t data[HK_PUSH_DESCRIPTOR_SET_SIZE];
};

void hk_push_descriptor_set_update(struct hk_push_descriptor_set *push_set,
                                   struct hk_descriptor_set_layout *layout,
                                   uint32_t write_count,
                                   const VkWriteDescriptorSet *writes);

void hk_push_descriptor_set_update_template(
   struct hk_push_descriptor_set *push_set,
   struct hk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data);
