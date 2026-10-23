/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <string.h>

#include "hk_descriptor_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "compiler/nir/nir_builder.h"
#include "vk_alloc.h"

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); abort(); \
} } while (0)

/* Evaluate the actual query/address lowering, replacing only descriptor memory
 * reads. The expected image coordinates come from the API and AIL, not from a
 * second implementation of the descriptor decoder.
 */
static void
evaluate(nir_builder *b, const void *descriptor, size_t bytes,
         uint64_t *result, unsigned components)
{
   agx_nir_lower_images(b->shader);
   agx_nir_lower_texture_queries(b->shader);
   bool progress;
   do {
      progress = nir_opt_constant_folding(b->shader);
      nir_foreach_block(block, b->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            b->cursor = nir_before_instr(instr);
            if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
               nir_def_replace(&intr->def, nir_imm_int64(b, 0x1000));
               progress = true;
            } else if (intr->intrinsic == nir_intrinsic_load_global_constant &&
                       nir_src_is_const(intr->src[0])) {
               uint64_t offset = nir_src_as_uint(intr->src[0]) - 0x1000;
               unsigned width = intr->def.bit_size / 8;
               CHECK(width && offset + width * intr->def.num_components <= bytes);
               nir_const_value value[NIR_MAX_VEC_COMPONENTS] = {0};
               for (unsigned c = 0; c < intr->def.num_components; ++c)
                  memcpy(&value[c], (const uint8_t *)descriptor + offset + c * width,
                         width);
               nir_def_replace(&intr->def, nir_build_imm(b, intr->def.num_components,
                                                       intr->def.bit_size, value));
               progress = true;
            }
         }
      }
      progress |= nir_opt_dead_cf(b->shader);
      progress |= nir_opt_dce(b->shader);
      progress |= nir_opt_remove_phis(b->shader);
   } while (progress);
   unsigned stores = 0;
   nir_foreach_block(block, b->impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         CHECK(intr->intrinsic == nir_intrinsic_store_global);
         CHECK(nir_src_is_const(intr->src[0]));
         for (unsigned c = 0; c < components; ++c)
            result[c] = nir_src_comp_as_uint(intr->src[0], c);
         ++stores;
      }
   }
   CHECK(stores == 1);
   ralloc_free(b->shader);
}

static void
check_size(const struct agx_texture_packed *tex, unsigned w, unsigned h, unsigned d)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options, "sliced size");
   nir_def *size = nir_bindless_image_size(&b, 3, 32, nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_3D);
   nir_store_global(&b, size, nir_imm_int64(&b, 0x2000),
                    .align_mul = 4, .write_mask = 7);
   uint64_t value[3];
   evaluate(&b, tex, sizeof(*tex), value, 3);
   if (value[0] != w || value[1] != h || value[2] != d)
      fprintf(stderr, "query: got %" PRIu64 "x%" PRIu64 "x%" PRIu64
                      ", expected %ux%ux%u\n", value[0], value[1], value[2], w, h, d);
   CHECK(value[0] == w && value[1] == h && value[2] == d);
}

static void
check_address(const struct agx_pbe_packed *pbe, enum pipe_format format,
              unsigned x, unsigned y, unsigned z, uint64_t expected)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                   &agx_nir_options, "sliced address");
   nir_def *addr = nir_bindless_image_texel_address(&b, 64, nir_imm_int(&b, 0),
      nir_imm_ivec4(&b, x, y, z, 0), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_3D, .format = format);
   nir_store_global(&b, addr, nir_imm_int64(&b, 0x2000),
                    .align_mul = 8, .write_mask = 1);
   uint64_t value;
   evaluate(&b, pbe, sizeof(*pbe), &value, 1);
   CHECK(value == expected);
}

static void
get_descriptor(struct hk_device *dev, VkImageView view, VkDescriptorType type,
               void *out)
{
   VkDescriptorImageInfo image = {.imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
   VkDescriptorGetInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT, .type = type,
      .data.pStorageImage = &image,
   };
   hk_GetDescriptorEXT(hk_device_to_handle(dev), &info, 128, out);
}

static void
check_updates(struct hk_device *dev, VkImageView view,
              const struct hk_apple9_storage_descriptor *expected)
{
   struct hk_descriptor_set_layout *layout = calloc(1, sizeof(*layout) + sizeof(layout->binding[0]));
   struct vk_descriptor_update_template *templ = calloc(1, sizeof(*templ) + sizeof(templ->entries[0]));
   CHECK(layout && templ);
   vk_object_base_init(&dev->vk, &layout->vk.base, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT);
   vk_object_base_init(&dev->vk, &templ->base, VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE);
   layout->binding_count = 1;
   layout->non_variable_descriptor_buffer_size = 3 * 128;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .array_size = 3, .stride = 128,
   };
   templ->entry_count = 1;
   templ->entries[0] = (struct vk_descriptor_template_entry){
      .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .array_element = 1,
      .array_count = 1, .stride = sizeof(VkDescriptorImageInfo),
   };
   uint8_t data[3 * 128];
   struct hk_descriptor_set set = {.layout = layout, .mapped_ptr = data, .size = sizeof(data)};
   vk_object_base_init(&dev->vk, &set.base, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   VkDescriptorImageInfo image = {.imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
   VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = hk_descriptor_set_to_handle(&set), .dstArrayElement = 1,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &image,
   };
   for (unsigned mutable = 0; mutable < 2; ++mutable) {
      layout->binding[0].type = mutable ? VK_DESCRIPTOR_TYPE_MUTABLE_EXT : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      memset(data, 0x5a, sizeof(data));
      hk_UpdateDescriptorSets(hk_device_to_handle(dev), 1, &write, 0, NULL);
      CHECK(!memcmp(data + 128, expected, 128));
      CHECK(data[127] == 0x5a && data[256] == 0x5a);
      VkCopyDescriptorSet copy = {
         .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
         .srcSet = write.dstSet, .dstSet = write.dstSet,
         .srcArrayElement = 1, .dstArrayElement = 2, .descriptorCount = 1,
      };
      hk_UpdateDescriptorSets(hk_device_to_handle(dev), 0, NULL, 1, &copy);
      CHECK(!memcmp(data + 256, expected, 128));
   }
   layout->binding[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   memset(data, 0, sizeof(data));
   hk_UpdateDescriptorSetWithTemplate(hk_device_to_handle(dev), write.dstSet,
      vk_descriptor_update_template_to_handle(templ), &image);
   CHECK(!memcmp(data + 128, expected, 128));
   struct hk_push_descriptor_set push = {0};
   write.dstSet = VK_NULL_HANDLE;
   hk_push_descriptor_set_update(&push, layout, 1, &write);
   CHECK(!memcmp(push.data + 128, expected, 128));
   memset(&push, 0, sizeof(push));
   hk_push_descriptor_set_update_template(&push, layout, templ, &image);
   CHECK(!memcmp(push.data + 128, expected, 128));
   vk_object_base_finish(&set.base);
   vk_object_base_finish(&templ->base);
   vk_object_base_finish(&layout->vk.base);
   free(templ);
   free(layout);
}

static unsigned cases;
static void
check_view(struct hk_device *dev, VkExtent3D extent, unsigned levels, unsigned level,
           unsigned offset, unsigned count, VkFormat format, VkImageCreateFlags flags)
{
   const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags,
      .imageType = VK_IMAGE_TYPE_3D, .format = format, .extent = extent,
      .mipLevels = levels, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   struct hk_image image = {0};
   vk_image_init(&dev->vk, &image.vk, &ci);
   image.plane_count = 1;
   image.planes[0].addr = UINT64_C(0x2000000000);
   struct ail_layout *layout = &image.planes[0].layout;
   *layout = (struct ail_layout){
      .width_px = extent.width, .height_px = extent.height, .depth_px = extent.depth,
      .levels = levels, .mipmapped_z = true, .sample_count_sa = 1,
      .format = vk_format_to_pipe_format(format), .tiling = AIL_TILING_GPU,
      .writeable_image = true, .renderable = true,
   };
   ail_make_miptree(layout);
   VkImageViewSlicedCreateInfoEXT slice = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_SLICED_CREATE_INFO_EXT,
      .sliceOffset = offset, .sliceCount = count,
   };
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = hk_image_to_handle(&image), .viewType = VK_IMAGE_VIEW_TYPE_3D,
      .format = format, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1},
   };
   bool alias = vk_format_is_compressed(format);
   if (alias)
      vi.format = VK_FORMAT_R32G32B32A32_UINT;
   else if (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)
      vi.format = VK_FORMAT_R32_SFLOAT;
   if (flags & VK_IMAGE_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT)
      vi.flags = VK_IMAGE_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT;
   VkImageView full, sliced;
   CHECK(hk_CreateImageView(hk_device_to_handle(dev), &vi, NULL, &full) == VK_SUCCESS);
   vi.pNext = &slice;
   CHECK(hk_CreateImageView(hk_device_to_handle(dev), &vi, NULL, &sliced) == VK_SUCCESS);
   struct hk_image_view *f = hk_image_view_from_handle(full);
   struct hk_image_view *s = hk_image_view_from_handle(sliced);
   unsigned depth = u_minify(extent.depth, level);
   unsigned slices = count == VK_REMAINING_3D_SLICES_EXT ? depth - offset : count;
   CHECK(s->vk.storage.z_slice_offset == offset && s->vk.storage.z_slice_count == slices);
   CHECK(!memcmp(&f->planes[0].sampled, &s->planes[0].sampled, sizeof(f->planes[0].sampled)));
   CHECK(!memcmp(&f->planes[0].ia, &s->planes[0].ia, sizeof(f->planes[0].ia)));
   CHECK(!memcmp(&f->planes[0].background, &s->planes[0].background, sizeof(f->planes[0].background)));
   CHECK(!memcmp(&f->planes[0].eot, &s->planes[0].eot, sizeof(f->planes[0].eot)));
   CHECK(!memcmp(&f->planes[0].layered_eot, &s->planes[0].layered_eot, sizeof(f->planes[0].layered_eot)));
   CHECK(!memcmp(&f->planes[0].emrt_pbe, &s->planes[0].emrt_pbe, sizeof(f->planes[0].emrt_pbe)));
   struct hk_apple9_image_part full_sampled, sliced_sampled;
   get_descriptor(dev, full, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &full_sampled);
   get_descriptor(dev, sliced, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sliced_sampled);
   CHECK(!memcmp(&full_sampled, &sliced_sampled, sizeof(full_sampled)));
   struct hk_apple9_storage_descriptor desc;
   get_descriptor(dev, sliced, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &desc);
   CHECK(sizeof(desc) == HK_APPLE9_DB_STORAGE_SIZE);
   CHECK(desc.native_ok == (alias ? 0 : HK_APPLE9_STORAGE_READ_VALID));
   CHECK(!s->apple9_pbes.map);
   agx_unpack(NULL, &desc.common.tex, TEXTURE, query);
   agx_unpack(NULL, &desc.common.pbe, PBE, pbe);
   agx_unpack(NULL, &s->planes[0].ro_storage, TEXTURE, native);
   CHECK(native.depth == extent.depth);
   CHECK(query.first_level == (alias ? 0 : level) && pbe.level == (alias ? 0 : level));
   CHECK(pbe.layer_stride_sw == layout->layer_stride_B);
   CHECK(pbe.level_offset_sw == (alias ? 0 : layout->level_offsets_B[level]));
   CHECK(pbe.buffer == image.planes[0].addr + offset * (uint64_t)layout->layer_stride_B +
         (alias ? layout->level_offsets_B[level] : 0));
   CHECK(native.address == pbe.buffer);
   uint8_t native_expected[AGX_APPLE9_TEXTURE_TABLE_STRIDE] = {0};
   if (!alias)
      CHECK(agx_apple9_translate_texture(native_expected, &s->planes[0].ro_storage));
   CHECK(!memcmp(desc.texture, native_expected, sizeof(native_expected)));
   /* Independently inspect the native depth bits: a Z-major image must retain
    * its full mip-tree depth even when only one slice is visible to storage.
    */
   uint64_t native_address;
   memcpy(&native_address, desc.texture + 8, 8);
   if (!alias) {
      CHECK(((native_address >> 46) & 0x7ff) + 1 == extent.depth);
      CHECK((native_address & ((UINT64_C(1) << 40) - 1)) << 4 == pbe.buffer);
   }
   unsigned w = u_minify(extent.width, level), h = u_minify(extent.height, level);
   w = util_format_get_nblocksx(layout->format, w);
   h = util_format_get_nblocksy(layout->format, h);
   check_size(&desc.common.tex, w, h, slices);
   if (!alias)
      check_size(&f->planes[0].sampled, w, h, depth);
   enum pipe_format view_format = vk_format_to_pipe_format(vi.format);
   check_address(&desc.common.pbe, view_format, 0, 0, 0,
      image.planes[0].addr + ail_get_layer_level_B(layout, offset, level));
   check_address(&desc.common.pbe, view_format, 0, 0, slices - 1,
      image.planes[0].addr + ail_get_layer_level_B(layout, offset + slices - 1, level));
   /* Last tile origin exercises row/tile arithmetic in addition to Z and mip. */
   unsigned x = (w - 1) & ~(layout->tilesize_el[level].width_el - 1);
   unsigned y = (h - 1) & ~(layout->tilesize_el[level].height_el - 1);
   check_address(&desc.common.pbe, view_format, x, y, slices - 1,
      image.planes[0].addr + ail_get_twiddled_block_B(layout, level,
         x * util_format_get_blockwidth(layout->format),
         y * util_format_get_blockheight(layout->format), offset + slices - 1));
   check_updates(dev, sliced, &desc);
   hk_DestroyImageView(hk_device_to_handle(dev), sliced, NULL);
   hk_DestroyImageView(hk_device_to_handle(dev), full, NULL);
   vk_image_finish(&image.vk);
   ++cases;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device pdev = {0};
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.physical = &pdev.vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->dev.chip = pdev.dev.chip = AGX_CHIP_G16G;
   simple_mtx_init(&dev->usc_heap_lock, mtx_plain);
   struct agx_va resident_va = {.addr = UINT64_C(0x3000000000)};
   struct agx_bo resident = {.va = &resident_va};
   dev->apple9_resident = &resident;
   const VkExtent3D extents[] = {{127, 65, 33}, {8, 4, 129}, {2048, 8, 1}, {8, 4, 2048}};
   const VkFormat formats[] = {VK_FORMAT_R32_UINT, VK_FORMAT_R16G16B16A16_SFLOAT};
   for (unsigned e = 0; e < ARRAY_SIZE(extents); ++e) {
      unsigned levels = util_logbase2(MAX3(extents[e].width, extents[e].height, extents[e].depth)) + 1;
      for (unsigned level = 0; level < levels; ++level) {
         unsigned depth = u_minify(extents[e].depth, level);
         for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
            check_view(dev, extents[e], levels, level, 0, depth, formats[f], 0);
            check_view(dev, extents[e], levels, level, 0, 1, formats[f], 0);
            check_view(dev, extents[e], levels, level, depth - 1,
                       VK_REMAINING_3D_SLICES_EXT, formats[f], VK_IMAGE_CREATE_SPARSE_BINDING_BIT);
            if (depth > 2) {
               check_view(dev, extents[e], levels, level, 1, MIN2(depth - 1, 3), formats[f], 0);
               check_view(dev, extents[e], levels, level, depth / 2,
                          VK_REMAINING_3D_SLICES_EXT, formats[f], 0);
            }
         }
      }
   }
   check_view(dev, (VkExtent3D){128, 64, 32}, 1, 0, 3, 7, VK_FORMAT_R32_UINT,
              VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
   check_view(dev, (VkExtent3D){128, 64, 32}, 4, 2, 3, 2, VK_FORMAT_R32_UINT,
              VK_IMAGE_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT);
   for (unsigned level = 0; level < 5; ++level)
      check_view(dev, (VkExtent3D){513, 257, 129}, 8, level, 1, 3,
                 VK_FORMAT_BC3_UNORM_BLOCK, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                 VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
                 VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
   struct hk_apple9_storage_descriptor null;
   get_descriptor(dev, VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &null);
   check_size(&null.common.tex, 0, 0, 0);
   CHECK(null.native_ok == 0);
   simple_mtx_destroy(&dev->usc_heap_lock);
   free(dev);
   glsl_type_singleton_decref();
   printf("sliced 3D: %u view/mip/format cases; query/address lowering and descriptor paths pass\n", cases);
   return 0;
}
