/* SPDX-License-Identifier: MIT */
#include "../hk_descriptor_set.h"
#include "../hk_descriptor_set_layout.h"
#include "../hk_entrypoints.h"
#include "../hk_image.h"
#include "../hk_image_view.h"
#include "../hk_sampler.h"
#include "agx_helpers.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); return false; } } while (0)

/* All public descriptor writers must compose the same bytes, including push
 * and immutable-sampler paths. Distinct prepacked patterns make accidentally
 * rebuilding a record or selecting the wrong plane visible. */
static bool
check_writers(struct hk_device *dev, VkDescriptorType type,
              VkDescriptorImageInfo info, const void *expected, size_t size)
{
   uint8_t data[HK_MAX_SAMPLED_DESCRIPTOR_SIZE + 32];
   bool texel = type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
                type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
   VkBufferView null_buffer = VK_NULL_HANDLE;
   const void *entry_data = texel ? (const void *)&null_buffer : &info;
   VkDescriptorGetInfoEXT get = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
      .type = type, .data.pCombinedImageSampler = &info};
   if (type == VK_DESCRIPTOR_TYPE_SAMPLER)
      get.data.pSampler = &info.sampler;
   else if (texel)
      get.data.pUniformTexelBuffer = NULL;
   memset(data, 0xa5, sizeof(data));
   hk_GetDescriptorEXT(hk_device_to_handle(dev), &get, size + 8, data + 8);
   CHECK(!memcmp(data + 8, expected, size));
   for (unsigned i = 0; i < 8; ++i) {
      CHECK(data[i] == 0xa5 && data[size + 8 + i] == 0);
      CHECK(data[size + 16 + i] == 0xa5);
   }

   struct hk_descriptor_set_layout *layout =
      calloc(1, sizeof(*layout) + sizeof(layout->binding[0]));
   struct vk_descriptor_update_template *templ =
      calloc(1, sizeof(*templ) + sizeof(templ->entries[0]));
   CHECK(layout && templ);
   layout->vk.base.device = &dev->vk;
   layout->binding_count = 1;
   layout->non_variable_descriptor_buffer_size = size + 16;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = type, .array_size = 1, .offset = 16, .stride = size};
   struct hk_descriptor_set set = {
      .base.type = VK_OBJECT_TYPE_DESCRIPTOR_SET,
      .layout = layout, .mapped_ptr = data, .size = sizeof(data)};
   VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = hk_descriptor_set_to_handle(&set), .descriptorCount = 1,
      .descriptorType = type, .pImageInfo = &info,
      .pTexelBufferView = &null_buffer};
   templ->base.type = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE;
   templ->entry_count = 1;
   templ->entries[0] = (struct vk_descriptor_template_entry){
      .type = type, .array_count = 1,
      .stride = texel ? sizeof(null_buffer) : sizeof(info)};
   struct hk_push_descriptor_set push;
   for (unsigned path = 0; path < 4; ++path) {
      memset(data, 0xa5, sizeof(data));
      memset(&push, 0xa5, sizeof(push));
      write.dstSet = path < 2 ? hk_descriptor_set_to_handle(&set) : VK_NULL_HANDLE;
      if (path == 0)
         hk_UpdateDescriptorSets(hk_device_to_handle(dev), 1, &write, 0, NULL);
      else if (path == 1)
         hk_UpdateDescriptorSetWithTemplate(hk_device_to_handle(dev),
            hk_descriptor_set_to_handle(&set),
            vk_descriptor_update_template_to_handle(templ), entry_data);
      else if (path == 2)
         hk_push_descriptor_set_update(&push, layout, 1, &write);
      else
         hk_push_descriptor_set_update_template(&push, layout, templ, entry_data);
      const uint8_t *result = path < 2 ? data : push.data;
      if (path >= 2)
         CHECK(push.size == layout->non_variable_descriptor_buffer_size);
      CHECK(!memcmp(result + 16, expected, size));
      for (unsigned i = 0; i < 16; ++i)
         CHECK(result[i] == 0xa5 && result[size + 16 + i] == 0xa5);
   }
   if (type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
       type == VK_DESCRIPTOR_TYPE_SAMPLER) {
      struct hk_sampler *immutable = hk_sampler_from_handle(info.sampler);
      layout->binding[0].immutable_samplers = &immutable;
      info.sampler = VK_NULL_HANDLE;
      write.dstSet = hk_descriptor_set_to_handle(&set);
      memset(data, 0xa5, sizeof(data));
      hk_UpdateDescriptorSets(hk_device_to_handle(dev), 1, &write, 0, NULL);
      CHECK(!memcmp(data + 16, expected, size));
   }
   free(templ);
   free(layout);
   return true;
}

static bool
check(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->dev.chip = AGX_CHIP_G16G;
   struct agx_va va[4] = {{.addr = 0x123400005678}, {.addr = 0x234500006789},
                           {.addr = 0x34560000789a}, {.addr = 0x4567000089ab}};
   struct agx_bo resident[4] = {{.va = &va[0]}, {.va = &va[1]},
                                {.va = &va[2]}, {.va = &va[3]}};
   dev->apple9_resident = &resident[0];
   struct hk_image image = {0};
   struct hk_image_view view = {.vk.base.type = VK_OBJECT_TYPE_IMAGE_VIEW,
      .vk.image = &image.vk, .vk.format = VK_FORMAT_R8G8B8A8_UNORM,
      .plane_count = 3};
   struct hk_sampler sampler = {.vk.base.type = VK_OBJECT_TYPE_SAMPLER,
      .plane_count = 2};
   for (unsigned p = 0; p < 3; ++p) {
      view.planes[p].image_plane = 2 - p;
      image.planes[p].apple9_residency = &resident[p + 1];
      memset(&view.planes[p].apple9_sampled, 0x31 + p,
             sizeof(view.planes[p].apple9_sampled));
      memset(&view.planes[p].apple9_ia, 0x41 + p,
             sizeof(view.planes[p].apple9_ia));
   }
   for (unsigned p = 0; p < 2; ++p)
      memset(&sampler.planes[p].apple9, 0x51 + p, sizeof(sampler.planes[p].apple9));
   VkDescriptorImageInfo info = {.imageView = hk_image_view_to_handle(&view),
      .sampler = hk_sampler_to_handle(&sampler)};
   struct hk_image_view saved = view;
   for (unsigned pass = 0; pass < 2; ++pass) {
      /* Binding residency after view creation must affect the next write. */
      if (pass)
         image.planes[2].apple9_residency = NULL;
      for (unsigned kind = 0; kind < 3; ++kind) {
         VkDescriptorType type = kind == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                 : kind == 1 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                             : VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
         uint8_t expected[HK_MAX_SAMPLED_DESCRIPTOR_SIZE];
         size_t size = 0;
         for (unsigned p = 0; p < 3; ++p) {
            struct hk_apple9_image_part part = kind == 2 ? view.planes[p].apple9_ia
                                                        : view.planes[p].apple9_sampled;
            uint64_t addr = pass && !p ? va[0].addr : va[3 - p].addr;
            part.residency_lo = addr;
            part.residency_hi = addr >> 32;
            memcpy(expected + size, &part, sizeof(part));
            size += sizeof(part);
            if (kind == 0) {
               memcpy(expected + size, &sampler.planes[MIN2(p, 1)].apple9, 128);
               size += 128;
            }
         }
         CHECK(check_writers(dev, type, info, expected, size));
         CHECK(!memcmp(&view, &saved, sizeof(view)));
      }
   }
   struct hk_apple9_sampler_part samplers[2] = {
      sampler.planes[0].apple9, sampler.planes[1].apple9};
   CHECK(check_writers(dev, VK_DESCRIPTOR_TYPE_SAMPLER, info, samplers, sizeof(samplers)));

   view.plane_count = 1;
   memset(&view.apple9_storage, 0x61, sizeof(view.apple9_storage));
   view.apple9_storage.native_ok = HK_APPLE9_STORAGE_READ_VALID;
   view.apple9_storage.store_encode = HK_STORE_ENCODE_UNORM8;
   view.apple9_store_flags = HK_APPLE9_STORAGE_LAYER_WRITE_VALID;
   view.apple9_pbe_index = 0x1234;
   struct hk_apple9_storage_descriptor storage = view.apple9_storage;
   storage.metadata.base = va[0].addr;
   storage.native_ok |= view.apple9_store_flags |
                       view.apple9_pbe_index << HK_APPLE9_STORAGE_PBE_SHIFT;
   CHECK(check_writers(dev, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info, &storage, sizeof(storage)));
   CHECK(view.apple9_storage.metadata.base == UINT64_C(0x6161616161616161));

   info.imageView = VK_NULL_HANDLE;
   struct hk_apple9_image_part null = {.residency_lo = va[0].addr,
                                     .residency_hi = va[0].addr >> 32};
   struct agx_texture_packed tex;
   agx_set_null_texture(&tex);
   agx_apple9_translate_texture(null.texture, &tex);
   hk_apple9_texture_info(null.texture_info, &tex);
   CHECK(check_writers(dev, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, info, &null, sizeof(null)));
   CHECK(check_writers(dev, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, info, &null, sizeof(null)));
   CHECK(null.texture_info[3] == 0 && null.min_lod_fp16 == 0);
   uint8_t native[AGX_APPLE9_TEXTURE_TABLE_STRIDE + 2];
   memset(native, 0xa5, sizeof(native));
   agx_apple9_set_null_texture(native + 1);
   CHECK(native[0] == 0xa5 && native[sizeof(native) - 1] == 0xa5);
   CHECK(!memcmp(native + 1, null.texture, sizeof(null.texture)));
   uint8_t combined[2 * (sizeof(null) + sizeof(sampler.planes[0].apple9))];
   for (unsigned p = 0; p < 2; ++p) {
      uint8_t *record = combined + p * (sizeof(null) + sizeof(sampler.planes[p].apple9));
      memcpy(record, &null, sizeof(null));
      memcpy(record + sizeof(null), &sampler.planes[p].apple9,
             sizeof(sampler.planes[p].apple9));
   }
   CHECK(check_writers(dev, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, info,
                       combined, sizeof(combined)));
   struct hk_apple9_texel_descriptor texel = {.tex = tex};
   memcpy(texel.texture, null.texture, sizeof(texel.texture));
   CHECK(check_writers(dev, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, info,
                       &texel, sizeof(texel)));
   struct hk_apple9_storage_descriptor storage_texel = {.common.tex = tex};
   agx_set_null_pbe(&storage_texel.common.pbe);
   CHECK(check_writers(dev, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, info,
                       &storage_texel, sizeof(storage_texel)));
   storage = (struct hk_apple9_storage_descriptor){.metadata.base = va[0].addr};
   agx_set_null_texture(&storage.common.tex);
   agx_set_null_pbe(&storage.common.pbe);
   CHECK(check_writers(dev, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info, &storage, sizeof(storage)));
   free(dev);
   return true;
}

int main(void) { return check() ? 0 : 1; }
