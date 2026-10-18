/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set.h"
#include "hk_gpu_busy.h"
#include "asahi/lib/agx_bo.h"
#include "util/format/u_format.h"
#include "util/half_float.h"
#include "vulkan/vulkan_core.h"

#include "agx_helpers.h"
#include "hk_buffer.h"
#include "hk_buffer_view.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_descriptor_buffer.h"

static inline uint32_t
align_u32(uint32_t v, uint32_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

static inline void *
desc_ubo_data(struct hk_descriptor_set *set, uint32_t binding, uint32_t elem,
              uint32_t *size_out)
{
   const struct hk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   uint32_t offset = binding_layout->offset + elem * binding_layout->stride;
   assert(offset < set->size);

   if (size_out != NULL)
      *size_out = set->size - offset;

   return (char *)set->mapped_ptr + offset;
}

static void
write_desc(struct hk_descriptor_set *set, uint32_t binding, uint32_t elem,
           const void *desc_data, size_t desc_size)
{
   ASSERTED uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   assert(desc_size <= dst_size);
   memcpy(dst, desc_data, desc_size);
}

/* Pack a sampled image, sampler or combined descriptor, one record per plane,
 * into out (HK_MAX_SAMPLED_DESCRIPTOR_SIZE bytes); returns its size. */
static size_t
hk_pack_sampled_image(struct hk_device *dev, void *out,
                      VkDescriptorType descriptor_type,
                      const VkDescriptorImageInfo *const info,
                      struct hk_sampler *sampler)
{
   struct hk_sampled_image_descriptor desc[3] = {};

   /* "Zero" initialize */
   for (unsigned i = 0; i < 3; ++i) {
      agx_set_null_texture(&desc[i].tex);
   }

   uint8_t plane_count = 1;
   bool ia = (descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
   const uint32_t *border = NULL;

   if (descriptor_type != VK_DESCRIPTOR_TYPE_SAMPLER && info &&
       info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_image_view, view, info->imageView);

      plane_count = view->plane_count;
      for (uint8_t plane = 0; plane < plane_count; plane++) {
         desc[plane].tex =
            ia ? view->planes[plane].ia : view->planes[plane].sampled;

         float min_lod = MAX2(view->vk.min_lod - view->vk.base_mip_level, 0.0);

         desc[plane].min_lod_fp16 = _mesa_float_to_half(min_lod);
         desc[plane].min_lod_uint16 = min_lod;
      }
   }

   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      border = sampler->apple9_border;

      if (sampler->has_border)
         assert(plane_count == 1);
      else
         plane_count = MAX2(plane_count, sampler->plane_count);

      for (uint8_t plane = 0; plane < plane_count; plane++) {
         /* We need to replicate the last sampler plane out to all image
          * planes due to sampler table entry limitations. See
          * hk_CreateSampler in hk_sampler.c for more details.
          */
         uint8_t sampler_plane = MIN2(plane, sampler->plane_count - 1);
         assert(sampler->planes[sampler_plane].hw->index < (1 << 12));

         /* All bindless samplers are indexed from 28 in hardware, add here so
          * we don't have to care in the shader.
          */
         desc[plane].sampler_index =
            sampler->planes[sampler_plane].hw->index + 28;
         desc[plane].sampler = sampler->planes[sampler_plane].hw->key;
         desc[plane].lod_bias_fp16 = sampler->lod_bias_fp16;
         desc[plane].txl_nudge_fp16 = sampler->txl_nudge_fp16;
         desc[plane].clamp_0_sampler_index_or_negative = -1;
      }

      if (sampler->has_border) {
         assert(sampler->plane_count == 2);
         desc[0].clamp_0_sampler_index_or_negative =
            sampler->planes[1].hw->index + 28;

         assert(desc[0].clamp_0_sampler_index_or_negative >= 0 &&
                "we have a border colour");

         static_assert(sizeof(desc[0].border) == sizeof(sampler->custom_border),
                       "fixed format");

         memcpy(desc[0].border, sampler->custom_border.uint32,
                sizeof(sampler->custom_border));
      }
   }
   if (dev->dev.chip == AGX_CHIP_G16G) {
      /* One record per plane: an image part read through texture handles, a
       * sampler part read through sampler handles, or both in that order for
       * a combined image sampler. */
      const bool image = descriptor_type != VK_DESCRIPTOR_TYPE_SAMPLER;
      const bool sampled = descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
                           descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      const size_t record = (image ? sizeof(struct hk_apple9_image_part) : 0) +
                            (sampled ? sizeof(struct hk_apple9_sampler_part) : 0);
      memset(out, 0, record * plane_count);
      for (uint8_t plane = 0; plane < plane_count; plane++) {
         uint8_t *rec = (uint8_t *)out + plane * record;
         if (image) {
            struct hk_apple9_image_part *ip = (void *)rec;
            uint64_t residency = dev->apple9_resident->va->addr;
            if (info && info->imageView != VK_NULL_HANDLE) {
               VK_FROM_HANDLE(hk_image_view, view, info->imageView);
               residency = hk_apple9_view_residency(dev, view, plane);
               ip->base_level = view->vk.base_mip_level;
               ip->base_layer = view->vk.base_array_layer;
            }
            ip->residency_lo = residency;
            ip->residency_hi = residency >> 32;
            ip->min_lod_fp16 = desc[plane].min_lod_fp16;
            ip->min_lod_uint16 = desc[plane].min_lod_uint16;
            agx_apple9_translate_texture(ip->texture, &desc[plane].tex);
            hk_apple9_texture_info(ip->texture_info, &desc[plane].tex);
            /* A padded compressed alias reports its logical width. */
            if (info && info->imageView != VK_NULL_HANDLE) {
               VK_FROM_HANDLE(hk_image_view, view, info->imageView);
               if (view->planes[plane].alias_width_el)
                  ip->texture_info[0] = view->planes[plane].alias_width_el;
            }
         }
         if (sampled) {
            struct hk_apple9_sampler_part *sp =
               (void *)(rec + (image ? sizeof(struct hk_apple9_image_part) : 0));
            hk_apple9_native_sampler(sp->sampler, &desc[plane].sampler, false,
                                     border);
            hk_apple9_runtime_sampler(sp->runtime_sampler, &desc[plane].sampler,
                                      false);
            sp->lod_bias_fp16 = desc[plane].lod_bias_fp16;
            sp->txl_nudge_fp16 = desc[plane].txl_nudge_fp16;
         }
      }
      return record * plane_count;
   }

   memcpy(out, desc, sizeof(desc[0]) * plane_count);
   return sizeof(desc[0]) * plane_count;
}

static void
write_sampled_image_view_desc(struct hk_descriptor_set *set,
                              const VkDescriptorImageInfo *const info,
                              uint32_t binding, uint32_t elem,
                              VkDescriptorType descriptor_type)
{
   struct hk_sampler *sampler = NULL;
   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      const struct hk_descriptor_set_binding_layout *binding_layout =
         &set->layout->binding[binding];
      sampler = binding_layout->immutable_samplers
                   ? binding_layout->immutable_samplers[elem]
                   : hk_sampler_from_handle(info->sampler);
   }
   /* Push descriptor writes use a transient set without an object base. */
   struct hk_device *dev = (struct hk_device *)set->layout->vk.base.device;
   uint8_t out[HK_MAX_SAMPLED_DESCRIPTOR_SIZE];
   size_t size = hk_pack_sampled_image(dev, out, descriptor_type, info, sampler);
   write_desc(set, binding, elem, out, size);
}

bool
hk_apple9_native_store_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R11G11B10_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return true;
   default:
      return false;
   }
}

static size_t
hk_pack_storage_image(struct hk_device *dev, void *out,
                      const VkDescriptorImageInfo *const info)
{
   struct hk_storage_image_descriptor desc = {};

   if (info && info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_image_view, view, info->imageView);

      /* Storage images are always single plane */
      assert(view->plane_count == 1);
      uint8_t plane = 0;

      desc.tex = view->planes[plane].ro_storage;
      desc.pbe = view->planes[plane].storage;
   } else {
      agx_set_null_texture(&desc.tex);
      agx_set_null_pbe(&desc.pbe);
   }

   if (dev->dev.chip == AGX_CHIP_G16G) {
      struct hk_apple9_storage_descriptor ext = {.common = desc};
      ext.metadata.base = dev->apple9_resident->va->addr;
      if (info && info->imageView != VK_NULL_HANDLE) {
         VK_FROM_HANDLE(hk_image_view, view, info->imageView);
         enum pipe_format pformat = vk_format_to_pipe_format(view->vk.format);
         hk_apple9_texel_layout(pformat, &ext.metadata);
         ext.store_encode = hk_apple9_store_encode(pformat);
         hk_gpu_busy_storage_format(view->vk.format, ext.store_encode);
         ext.metadata.base = hk_apple9_view_residency(dev, view, 0);
         ext.metadata.elements =
            view->vk.base_mip_level | (view->vk.base_array_layer << 16);
         if (agx_apple9_translate_texture(ext.texture, &desc.tex))
            ext.native_ok |= HK_APPLE9_STORAGE_READ_VALID;
         /* Native accesses keep the padded alias width. Size queries read
          * the legacy texture: give it the logical width.
          */
         if (view->planes[0].alias_width_el) {
            agx_unpack(NULL, &ext.common.tex, TEXTURE, t);
            t.width = view->planes[0].alias_width_el;
            agx_pack(&ext.common.tex, TEXTURE, cfg) {
               cfg = t;
            }
         }
         ext.native_ok |= view->apple9_store_flags;
         ext.native_ok |= view->apple9_pbe_index << HK_APPLE9_STORAGE_PBE_SHIFT;
      }
      memcpy(out, &ext, sizeof(ext));
      return sizeof(ext);
   }

   memcpy(out, &desc, sizeof(desc));
   return sizeof(desc);
}

static void
write_storage_image_view_desc(struct hk_descriptor_set *set,
                              const VkDescriptorImageInfo *const info,
                              uint32_t binding, uint32_t elem)
{
   struct hk_device *dev = (struct hk_device *)set->layout->vk.base.device;
   struct hk_apple9_storage_descriptor out;
   static_assert(sizeof(out) >= sizeof(struct hk_storage_image_descriptor),
                 "room for either record");
   write_desc(set, binding, elem, &out,
              hk_pack_storage_image(dev, &out, info));
}

static struct hk_buffer_address
get_buffer_address(const VkDescriptorBufferInfo *const info, bool read_only)
{
   VK_FROM_HANDLE(hk_buffer, buffer, info->buffer);
   const struct hk_addr_range addr_range =
      hk_buffer_addr_range(buffer, info->offset, info->range, read_only);
   assert(addr_range.range <= UINT32_MAX);

   return (struct hk_buffer_address){
      .base_addr = addr_range.addr,
      .size = addr_range.range,
   };
}

static void
write_buffer_desc(struct hk_descriptor_set *set,
                  const VkDescriptorBufferInfo *const info, uint32_t binding,
                  uint32_t elem, bool read_only)
{
   const struct hk_buffer_address desc = get_buffer_address(info, read_only);

   write_desc(set, binding, elem, &desc, sizeof(desc));
}

static void
write_dynamic_buffer_desc(struct hk_descriptor_set *set,
                          const VkDescriptorBufferInfo *const info,
                          uint32_t binding, uint32_t elem, bool read_only)
{
   const struct hk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   struct hk_buffer_address *desc =
      &set->dynamic_buffers[binding_layout->dynamic_buffer_index + elem];

   *desc = get_buffer_address(info, read_only);
}

/* Pack a texel buffer descriptor from its native state (a buffer view or an
 * address range), NULL for a null descriptor; returns its size. */
static size_t
hk_pack_texel_buffer_desc(struct hk_device *dev, void *out,
                          VkDescriptorType type,
                          const struct agx_texture_packed *tex,
                          const struct agx_pbe_packed *pbe,
                          const struct hk_apple9_texel_metadata *m,
                          const uint8_t *apple9_texture)
{
   /* G16 uniform texel buffers carry load metadata, also in mutable bindings. */
   if (dev->dev.chip == AGX_CHIP_G16G &&
       type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
      struct hk_apple9_texel_descriptor desc = {};
      if (tex) {
         desc.tex = *tex;
         desc.metadata = *m;
         memcpy(desc.texture, apple9_texture, sizeof(desc.texture));
      } else {
         agx_set_null_texture(&desc.tex);
         agx_apple9_translate_texture(desc.texture, &desc.tex);
      }
      memcpy(out, &desc, sizeof(desc));
      return sizeof(desc);
   } else if (dev->dev.chip == AGX_CHIP_G16G) {
      struct hk_apple9_storage_descriptor desc = {};
      if (tex) {
         desc.common.tex = *tex;
         desc.common.pbe = *pbe;
         desc.metadata = *m;
      } else {
         agx_set_null_texture(&desc.common.tex);
         agx_set_null_pbe(&desc.common.pbe);
      }
      memcpy(out, &desc, sizeof(desc));
      return sizeof(desc);
   } else {
      struct hk_storage_image_descriptor desc = {};
      if (tex) {
         desc.tex = *tex;
         desc.pbe = *pbe;
      } else {
         agx_set_null_texture(&desc.tex);
         agx_set_null_pbe(&desc.pbe);
      }
      memcpy(out, &desc, sizeof(desc));
      return sizeof(desc);
   }
}

static void
write_buffer_view_desc(struct hk_descriptor_set *set,
                       const VkBufferView bufferView, uint32_t binding,
                       uint32_t elem, VkDescriptorType type)
{
   VK_FROM_HANDLE(hk_buffer_view, view, bufferView);
   struct hk_device *dev = (struct hk_device *)set->layout->vk.base.device;
   uint8_t out[HK_MAX_DESCRIPTOR_SIZE];
   size_t size = hk_pack_texel_buffer_desc(
      dev, out, type, view ? &view->tex : NULL, view ? &view->pbe : NULL,
      view ? &view->apple9 : NULL, view ? view->apple9_texture : NULL);
   write_desc(set, binding, elem, out, size);
}

static void
write_inline_uniform_data(struct hk_descriptor_set *set,
                          const VkWriteDescriptorSetInlineUniformBlock *info,
                          uint32_t binding, uint32_t offset)
{
   assert(set->layout->binding[binding].stride == 1);
   write_desc(set, binding, offset, info->pData, info->dataSize);
}

VKAPI_ATTR void VKAPI_CALL
hk_UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                        const VkWriteDescriptorSet *pDescriptorWrites,
                        uint32_t descriptorCopyCount,
                        const VkCopyDescriptorSet *pDescriptorCopies)
{
   VK_FROM_HANDLE(hk_device, dev, device);

   for (uint32_t w = 0; w < descriptorWriteCount; w++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[w];
      VK_FROM_HANDLE(hk_descriptor_set, set, write->dstSet);

      /* vkd3d-proton writes null descriptors of one type into every mutable
       * binding of its heap, including bindings whose type list holds only
       * buffers. An Apple9 image descriptor is larger than those slots and
       * would overwrite the following descriptors, so make such slots null
       * buffers instead.
       */
      const struct hk_descriptor_set_binding_layout *binding_layout =
         &set->layout->binding[write->dstBinding];
      if (binding_layout->type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT) {
         uint32_t stride, alignment;
         hk_descriptor_stride_align_for_type(hk_device_physical(dev),
                                             write->descriptorType, NULL,
                                             false, &stride, &alignment);
         if (stride > binding_layout->stride) {
            const VkDescriptorBufferInfo null_info = {.range = VK_WHOLE_SIZE};
            for (uint32_t j = 0; j < write->descriptorCount; j++) {
               write_buffer_desc(set, &null_info, write->dstBinding,
                                 write->dstArrayElement + j, false);
            }
            continue;
         }

         /* A smaller descriptor leaves the rest of the slot to whatever was
          * written before, such as an image's size and levels. D3D12 allows
          * reading a descriptor as another type, and translation layers
          * write null descriptors of one type for all of them. Clear the
          * slot to a null image first, so the trailing fields read as null.
          */
         uint32_t image_stride, image_alignment;
         hk_descriptor_stride_align_for_type(hk_device_physical(dev),
                                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                             NULL, false, &image_stride,
                                             &image_alignment);
         if (stride < binding_layout->stride &&
             image_stride <= binding_layout->stride) {
            const VkDescriptorImageInfo null_image = {0};
            for (uint32_t j = 0; j < write->descriptorCount; j++) {
               write_sampled_image_view_desc(set, &null_image,
                                             write->dstBinding,
                                             write->dstArrayElement + j,
                                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
            }
         }
      }

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(set, write->pTexelBufferView[j],
                                   write->dstBinding,
                                   write->dstArrayElement + j,
                                   write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(
               set, write->pBufferInfo + j, write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_dynamic_buffer_desc(
               set, write->pBufferInfo + j, write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType ==
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
         }
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
         const VkWriteDescriptorSetInlineUniformBlock *write_inline =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK);
         assert(write_inline->dataSize == write->descriptorCount);
         write_inline_uniform_data(set, write_inline, write->dstBinding,
                                   write->dstArrayElement);
         break;
      }

      default:
         break;
      }
   }

   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      VK_FROM_HANDLE(hk_descriptor_set, src, copy->srcSet);
      VK_FROM_HANDLE(hk_descriptor_set, dst, copy->dstSet);

      const struct hk_descriptor_set_binding_layout *src_binding_layout =
         &src->layout->binding[copy->srcBinding];
      const struct hk_descriptor_set_binding_layout *dst_binding_layout =
         &dst->layout->binding[copy->dstBinding];

      if (dst_binding_layout->stride > 0 && src_binding_layout->stride > 0) {
         for (uint32_t j = 0; j < copy->descriptorCount; j++) {
            ASSERTED uint32_t dst_max_size, src_max_size;
            void *dst_map = desc_ubo_data(
               dst, copy->dstBinding, copy->dstArrayElement + j, &dst_max_size);
            const void *src_map = desc_ubo_data(
               src, copy->srcBinding, copy->srcArrayElement + j, &src_max_size);
            const uint32_t copy_size =
               MIN2(dst_binding_layout->stride, src_binding_layout->stride);
            assert(copy_size <= dst_max_size && copy_size <= src_max_size);
            memcpy(dst_map, src_map, copy_size);
         }
      }

      if (vk_descriptor_type_is_dynamic(src_binding_layout->type)) {
         const uint32_t dst_dyn_start =
            dst_binding_layout->dynamic_buffer_index + copy->dstArrayElement;
         const uint32_t src_dyn_start =
            src_binding_layout->dynamic_buffer_index + copy->srcArrayElement;
         typed_memcpy(&dst->dynamic_buffers[dst_dyn_start],
                      &src->dynamic_buffers[src_dyn_start],
                      copy->descriptorCount);
      }
   }
}

void
hk_push_descriptor_set_update(struct hk_push_descriptor_set *push_set,
                              struct hk_descriptor_set_layout *layout,
                              uint32_t write_count,
                              const VkWriteDescriptorSet *writes)
{
   assert(layout->non_variable_descriptor_buffer_size < sizeof(push_set->data));
   struct hk_descriptor_set set = {
      .layout = layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };

   for (uint32_t w = 0; w < write_count; w++) {
      const VkWriteDescriptorSet *write = &writes[w];
      assert(write->dstSet == VK_NULL_HANDLE);

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               &set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(&set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(&set, write->pTexelBufferView[j],
                                   write->dstBinding,
                                   write->dstArrayElement + j,
                                   write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(
               &set, write->pBufferInfo + j, write->dstBinding,
               write->dstArrayElement + j,
               write->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      default:
         break;
      }
   }
}

static void
hk_descriptor_set_destroy(struct hk_device *dev,
                          struct hk_descriptor_pool *pool,
                          struct hk_descriptor_set *set)
{
   list_del(&set->link);
   if (set->size > 0)
      hk_descriptor_memory_free(&pool->mem, &set->alloc);
   vk_descriptor_set_layout_unref(&dev->vk, &set->layout->vk);

   vk_object_free(&dev->vk, NULL, set);
}

/* Descriptor memory is in the USC aperture on G16, where shaders address
 * image descriptors as native table entries. Map it write-back (coherent on
 * G16, as the USC command pool): vkUpdateDescriptorSets copies read
 * descriptor memory, and reads from a write-combined mapping are uncached. */
static bool
hk_descriptor_bo_create(void *data, uint64_t size, uint64_t reserve,
                        struct hk_descriptor_backing *out)
{
   struct hk_device *dev = data;
   const bool low_va = dev->dev.chip == AGX_CHIP_G16G;
   struct agx_bo *bo = agx_bo_create(&dev->dev, size, 0,
                                     low_va ? AGX_BO_LOW_VA | AGX_BO_WRITEBACK
                                            : 0,
                                     "Descriptor pool");
   if (!bo)
      return false;

   *out = (struct hk_descriptor_backing){
      .bo = bo,
      .addr = bo->va->addr,
      .map = agx_bo_map(bo),
      .size = bo->size,
   };

   /* Reads past the set up to the layout maximum: alias the BO at the start
    * of a reserved range and map the zero page read-only over the tail. */
   const uint64_t page = AIL_PAGESIZE;
   reserve = align64(reserve, page);
   if (reserve > bo->size && dev->dev.zero_bo) {
      struct agx_va *va = agx_va_alloc(&dev->dev, reserve, page,
                                       low_va ? AGX_VA_USC : 0, 0);
      if (va && !agx_bo_bind(&dev->dev, bo, va->addr, bo->size, 0,
                             DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE) &&
          !agx_bo_bind(&dev->dev, dev->dev.zero_bo, va->addr + bo->size,
                       reserve - bo->size, 0,
                       DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_SINGLE_PAGE)) {
         out->addr = va->addr;
         out->va = va;
      } else {
         /* Out of USC aperture: keep the count-sized mapping. With soft
          * faults a hoisted read past the count then reads zero. */
         if (va)
            agx_va_free(&dev->dev, va, true);
         static bool warned;
         if (!warned) {
            warned = true;
            mesa_logw("hk: no USC VA for a %" PRIu64 " MiB descriptor-set "
                      "reservation; using the set's own mapping",
                      reserve >> 20);
         }
      }
   }
   return true;
}

static void
hk_descriptor_bo_destroy(void *data, struct hk_descriptor_backing *backing)
{
   struct hk_device *dev = data;
   struct agx_bo *bo = backing->bo;
   if (backing->va)
      agx_va_free(&dev->dev, backing->va, true);
   agx_bo_unreference(&dev->dev, bo);
}

static const struct hk_descriptor_memory_ops hk_descriptor_bo_ops = {
   .create = hk_descriptor_bo_create,
   .destroy = hk_descriptor_bo_destroy,
};

static void
hk_destroy_descriptor_pool(struct hk_device *dev,
                           const VkAllocationCallbacks *pAllocator,
                           struct hk_descriptor_pool *pool)
{
   list_for_each_entry_safe(struct hk_descriptor_set, set, &pool->sets, link)
      hk_descriptor_set_destroy(dev, pool, set);

   hk_descriptor_memory_finish(&pool->mem);

   vk_object_free(&dev->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateDescriptorPool(VkDevice _device,
                        const VkDescriptorPoolCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct hk_descriptor_pool *pool;

   pool = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                           VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   pool->host_only =
      pCreateInfo->flags & VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_EXT;

   list_inithead(&pool->sets);

   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);

   uint32_t max_align = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_type(pdev, pCreateInfo->pPoolSizes[i].type,
                                          type_list, false, &stride, &alignment);
      max_align = MAX2(max_align, alignment);
   }

   uint64_t bo_size = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      hk_descriptor_stride_align_for_type(pdev, pCreateInfo->pPoolSizes[i].type,
                                          type_list, false, &stride, &alignment);
      bo_size +=
         MAX2(stride, max_align) * pCreateInfo->pPoolSizes[i].descriptorCount;
   }

   /* Individual descriptor sets are aligned to the min UBO alignment to
    * ensure that we don't end up with unaligned data access in any shaders.
    * This means that each descriptor buffer allocated may burn up to 16B of
    * extra space to get the right alignment.  (Technically, it's at most 28B
    * because we're always going to start at least 4B aligned but we're being
    * conservative here.)  Allocate enough extra space that we can chop it
    * into maxSets pieces and align each one of them to 32B.
    */
   bo_size += HK_MIN_UBO_ALIGNMENT * pCreateInfo->maxSets;

   if (hk_descriptor_memory_init(&pool->mem, &hk_descriptor_bo_ops, dev,
                                 bo_size) != HK_DESCRIPTOR_MEMORY_OK) {
      hk_destroy_descriptor_pool(dev, pAllocator, pool);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   *pDescriptorPool = hk_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

static VkResult
hk_descriptor_set_create(struct hk_device *dev, struct hk_descriptor_pool *pool,
                         struct hk_descriptor_set_layout *layout,
                         uint32_t variable_count,
                         struct hk_descriptor_set **out_set)
{
   struct hk_descriptor_set *set;
   VkResult result;

   uint32_t mem_size =
      sizeof(struct hk_descriptor_set) +
      layout->vk.dynamic_descriptor_count * sizeof(struct hk_buffer_address);

   set =
      vk_object_zalloc(&dev->vk, NULL, mem_size, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   if (!set)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   set->size = layout->non_variable_descriptor_buffer_size;

   if (layout->binding_count > 0 &&
       (layout->binding[layout->binding_count - 1].flags &
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)) {
      uint32_t stride = layout->binding[layout->binding_count - 1].stride;

      /* Off by one so we don't underallocate the end. Otherwise vkd3d-proton
       * descriptor-performance underallocates.
       */
      set->size += stride * (variable_count + 1);
   }

   set->size = align64(set->size, HK_MIN_UBO_ALIGNMENT);

   /* Shaders clamp a variable-count binding's index to the layout maximum,
    * not to this set's count, and preamble hoisting may read a descriptor
    * whose guarded index is past the count. Keep those reads mapped: a
    * dedicated allocation reserves VA up to the layout maximum, its tail
    * reading as zero. Chunk suballocations stay inside mapped chunks. */
   /* Host-only pools (vkd3d's CPU descriptor heaps) are never read by the
    * GPU and reserve nothing: at vkd3d's 1M-descriptor layout maximum a
    * reservation is 128 MiB of USC aperture per set. */
   uint64_t reserve = 0;
   if (!pool->host_only && layout->binding_count > 0 &&
       (layout->binding[layout->binding_count - 1].flags &
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)) {
      const struct hk_descriptor_set_binding_layout *b =
         &layout->binding[layout->binding_count - 1];
      reserve = align64(layout->non_variable_descriptor_buffer_size +
                           (uint64_t)b->stride * (b->array_size + 1),
                        HK_MIN_UBO_ALIGNMENT);
   }

   if (set->size > 0) {
      switch (hk_descriptor_memory_alloc(&pool->mem, set->size,
                                         HK_MIN_UBO_ALIGNMENT, reserve,
                                         &set->alloc)) {
      case HK_DESCRIPTOR_MEMORY_OK:
         break;
      case HK_DESCRIPTOR_MEMORY_OUT_OF_POOL:
         vk_object_free(&dev->vk, NULL, set);
         return VK_ERROR_OUT_OF_POOL_MEMORY;
      case HK_DESCRIPTOR_MEMORY_OUT_OF_DEVICE:
         vk_object_free(&dev->vk, NULL, set);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
      set->addr = set->alloc.addr;
      set->mapped_ptr = set->alloc.map;
   }

   vk_descriptor_set_layout_ref(&layout->vk);
   set->layout = layout;

   for (uint32_t b = 0; b < layout->binding_count; b++) {
      if (layout->binding[b].type != VK_DESCRIPTOR_TYPE_SAMPLER &&
          layout->binding[b].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         continue;

      if (layout->binding[b].immutable_samplers == NULL)
         continue;

      uint32_t array_size = layout->binding[b].array_size;
      if (layout->binding[b].flags &
          VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
         array_size = variable_count;

      for (uint32_t j = 0; j < array_size; j++) {
         write_sampled_image_view_desc(set, NULL, b, j,
                                       layout->binding[b].type);
      }
   }

   list_addtail(&set->link, &pool->sets);
   *out_set = set;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_AllocateDescriptorSets(VkDevice device,
                          const VkDescriptorSetAllocateInfo *pAllocateInfo,
                          VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, pAllocateInfo->descriptorPool);

   VkResult result = VK_SUCCESS;
   uint32_t i;

   struct hk_descriptor_set *set = NULL;

   const VkDescriptorSetVariableDescriptorCountAllocateInfo *var_desc_count =
      vk_find_struct_const(
         pAllocateInfo->pNext,
         DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

   /* allocate a set of buffers for each shader to contain descriptors */
   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(hk_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);
      /* If descriptorSetCount is zero or this structure is not included in
       * the pNext chain, then the variable lengths are considered to be zero.
       */
      const uint32_t variable_count =
         var_desc_count && var_desc_count->descriptorSetCount > 0
            ? var_desc_count->pDescriptorCounts[i]
            : 0;

      result =
         hk_descriptor_set_create(dev, pool, layout, variable_count, &set);
      if (result != VK_SUCCESS)
         break;

      pDescriptorSets[i] = hk_descriptor_set_to_handle(set);
   }

   if (result != VK_SUCCESS) {
      hk_FreeDescriptorSets(device, pAllocateInfo->descriptorPool, i,
                            pDescriptorSets);
      for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
         pDescriptorSets[i] = VK_NULL_HANDLE;
      }
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_FreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool,
                      uint32_t descriptorSetCount,
                      const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, descriptorPool);

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      VK_FROM_HANDLE(hk_descriptor_set, set, pDescriptorSets[i]);

      if (set)
         hk_descriptor_set_destroy(dev, pool, set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyDescriptorPool(VkDevice device, VkDescriptorPool _pool,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, _pool);

   if (!_pool)
      return;

   hk_destroy_descriptor_pool(dev, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                       VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_descriptor_pool, pool, descriptorPool);

   list_for_each_entry_safe(struct hk_descriptor_set, set, &pool->sets, link)
      hk_descriptor_set_destroy(dev, pool, set);

   return VK_SUCCESS;
}

static void
hk_descriptor_set_write_template(
   struct hk_descriptor_set *set,
   const struct vk_descriptor_update_template *template, const void *data)
{
   for (uint32_t i = 0; i < template->entry_count; i++) {
      const struct vk_descriptor_template_entry *entry = &template->entries[i];

      switch (entry->type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_sampled_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j,
                                          entry->type);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_storage_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkBufferView *bview =
               data + entry->offset + j * entry->stride;

            write_buffer_view_desc(set, *bview, entry->binding,
                                   entry->array_element + j, entry->type);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_buffer_desc(set, info, entry->binding,
                              entry->array_element + j,
                              entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_dynamic_buffer_desc(
               set, info, entry->binding, entry->array_element + j,
               entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
         }
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
         write_desc(set, entry->binding, entry->array_element,
                    data + entry->offset, entry->array_count);
         break;

      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_UpdateDescriptorSetWithTemplate(
   VkDevice device, VkDescriptorSet descriptorSet,
   VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void *pData)
{
   VK_FROM_HANDLE(hk_descriptor_set, set, descriptorSet);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  descriptorUpdateTemplate);

   hk_descriptor_set_write_template(set, template, pData);
}

void
hk_push_descriptor_set_update_template(
   struct hk_push_descriptor_set *push_set,
   struct hk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data)
{
   struct hk_descriptor_set tmp_set = {
      .layout = layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };
   hk_descriptor_set_write_template(&tmp_set, template, data);
}

/* Copy a packed record into application descriptor memory, zero-filling the
 * rest of the descriptor so no stale bytes of another type remain. The
 * destination is written only. */
static void
hk_store_descriptor(void *dst, size_t data_size, const void *src, size_t size)
{
   assert(size <= data_size);
   memcpy(dst, src, size);
   memset((uint8_t *)dst + size, 0, data_size - size);
}

static void
hk_get_texel_buffer_descriptor(struct hk_device *dev, VkDescriptorType type,
                               const VkDescriptorAddressInfoEXT *info,
                               size_t data_size, void *dst)
{
   uint8_t out[HK_MAX_DESCRIPTOR_SIZE];
   size_t size;
   if (info && info->address) {
      struct agx_texture_packed tex;
      struct agx_pbe_packed pbe;
      struct hk_apple9_texel_metadata m = {0};
      uint8_t texture[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
      enum pipe_format format = hk_format_to_pipe_format(info->format);
      uint32_t elements = info->range / util_format_get_blocksize(format);
      hk_pack_texel_buffer(dev, format, info->address, info->address, elements,
                           dev->dev.chip == AGX_CHIP_G16G
                              ? dev->apple9_resident->va->addr : 0,
                           &tex, &pbe, &m, texture);
      size = hk_pack_texel_buffer_desc(dev, out, type, &tex, &pbe, &m, texture);
   } else {
      size = hk_pack_texel_buffer_desc(dev, out, type, NULL, NULL, NULL, NULL);
   }
   hk_store_descriptor(dst, data_size, out, size);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDescriptorEXT(VkDevice _device, const VkDescriptorGetInfoEXT *info,
                    size_t dataSize, void *pDescriptor)
{
   VK_FROM_HANDLE(hk_device, dev, _device);

   switch (info->type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
      VkDescriptorImageInfo image = {0};
      struct hk_sampler *sampler = NULL;
      if (info->type == VK_DESCRIPTOR_TYPE_SAMPLER) {
         sampler = hk_sampler_from_handle(*info->data.pSampler);
      } else if (info->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
         image = *info->data.pCombinedImageSampler;
         sampler = hk_sampler_from_handle(image.sampler);
      } else if (info->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
         if (info->data.pSampledImage)
            image = *info->data.pSampledImage;
      } else if (info->data.pInputAttachmentImage) {
         image = *info->data.pInputAttachmentImage;
      }
      uint8_t out[HK_MAX_SAMPLED_DESCRIPTOR_SIZE];
      size_t size = hk_pack_sampled_image(dev, out, info->type, &image, sampler);
      /* Planes are already consecutive records. */
      hk_store_descriptor(pDescriptor, dataSize, out, size);
      break;
   }

   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
      struct hk_apple9_storage_descriptor out;
      VkDescriptorImageInfo image = {0};
      if (info->data.pStorageImage)
         image = *info->data.pStorageImage;
      hk_store_descriptor(pDescriptor, dataSize, &out,
                          hk_pack_storage_image(dev, &out, &image));
      break;
   }

   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      hk_get_texel_buffer_descriptor(dev, info->type,
                                     info->data.pUniformTexelBuffer, dataSize,
                                     pDescriptor);
      break;

   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      hk_get_texel_buffer_descriptor(dev, info->type,
                                     info->data.pStorageTexelBuffer, dataSize,
                                     pDescriptor);
      break;

   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
      const VkDescriptorAddressInfoEXT *a =
         info->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            ? info->data.pUniformBuffer
            : info->data.pStorageBuffer;
      struct hk_buffer_address desc = {0};
      if (a && a->address) {
         assert(a->range <= UINT32_MAX);
         desc.base_addr = a->address;
         desc.size = a->range;
      }
      hk_store_descriptor(pDescriptor, dataSize, &desc, sizeof(desc));
      break;
   }

   default:
      UNREACHABLE("Unsupported descriptor type");
   }
}

/* Pack the immutable samplers of an embedded-sampler layout into map, laid
 * out as a descriptor buffer of that layout. */
void
hk_descriptor_pack_embedded_samplers(struct hk_device *dev,
                                     const struct hk_descriptor_set_layout *layout,
                                     void *map)
{
   for (uint32_t b = 0; b < layout->binding_count; ++b) {
      const struct hk_descriptor_set_binding_layout *bl = &layout->binding[b];
      if (!bl->immutable_samplers)
         continue;
      for (uint32_t e = 0; e < bl->array_size; ++e) {
         uint8_t out[HK_MAX_SAMPLED_DESCRIPTOR_SIZE];
         size_t size = hk_pack_sampled_image(dev, out, VK_DESCRIPTOR_TYPE_SAMPLER,
                                             &(VkDescriptorImageInfo){0},
                                             bl->immutable_samplers[e]);
         hk_store_descriptor((uint8_t *)map + bl->offset + e * bl->stride,
                             bl->stride, out, MIN2(size, bl->stride));
      }
   }
}
