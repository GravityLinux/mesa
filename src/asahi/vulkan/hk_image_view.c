/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_apple9.h"
#include "hk_image_view.h"
#include "util/format/u_format.h"
#include "vulkan/vulkan_core.h"

#include "agx_helpers.h"
#include "agx_nir_texture.h"
#include "agx_pack.h"
#include "hk_device.h"
#include "hk_descriptor_set.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include "layout.h"
#include "vk_format.h"
#include "vk_meta.h"

enum hk_desc_usage {
   HK_DESC_USAGE_SAMPLED,
   HK_DESC_USAGE_STORAGE,
   HK_DESC_USAGE_INPUT,
   HK_DESC_USAGE_BG_EOT,
   HK_DESC_USAGE_LAYERED_BG_EOT,
   HK_DESC_USAGE_EMRT,
};

static bool
hk_image_view_type_is_array(VkImageViewType view_type)
{
   switch (view_type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_2D:
   case VK_IMAGE_VIEW_TYPE_3D:
   case VK_IMAGE_VIEW_TYPE_CUBE:
      return false;

   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return true;

   default:
      UNREACHABLE("Invalid image view type");
   }
}

static enum agx_texture_dimension
translate_image_view_type(VkImageViewType view_type, bool msaa, bool layered,
                          enum hk_desc_usage usage)
{
   if (usage == HK_DESC_USAGE_EMRT || usage == HK_DESC_USAGE_INPUT ||
       (usage == HK_DESC_USAGE_LAYERED_BG_EOT && layered)) {
      return msaa ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D_ARRAY;
   }

   /* For background/EOT, we ignore the application-provided view type */
   if (usage == HK_DESC_USAGE_BG_EOT || usage == HK_DESC_USAGE_LAYERED_BG_EOT) {
      return msaa ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D;
   }

   bool cubes_to_2d = usage != HK_DESC_USAGE_SAMPLED;

   switch (view_type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_2D:
      return msaa ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D;

   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
      return msaa ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                  : AGX_TEXTURE_DIMENSION_2D_ARRAY;

   case VK_IMAGE_VIEW_TYPE_3D:
      assert(!msaa);
      return AGX_TEXTURE_DIMENSION_3D;

   case VK_IMAGE_VIEW_TYPE_CUBE:
      assert(!msaa);
      return cubes_to_2d ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                         : AGX_TEXTURE_DIMENSION_CUBE;

   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      assert(!msaa);
      return cubes_to_2d ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                         : AGX_TEXTURE_DIMENSION_CUBE_ARRAY;

   default:
      UNREACHABLE("Invalid image view type");
   }
}

static enum pipe_swizzle
vk_swizzle_to_pipe(VkComponentSwizzle swizzle)
{
   switch (swizzle) {
   case VK_COMPONENT_SWIZZLE_R:
      return PIPE_SWIZZLE_X;
   case VK_COMPONENT_SWIZZLE_G:
      return PIPE_SWIZZLE_Y;
   case VK_COMPONENT_SWIZZLE_B:
      return PIPE_SWIZZLE_Z;
   case VK_COMPONENT_SWIZZLE_A:
      return PIPE_SWIZZLE_W;
   case VK_COMPONENT_SWIZZLE_ONE:
      return PIPE_SWIZZLE_1;
   case VK_COMPONENT_SWIZZLE_ZERO:
      return PIPE_SWIZZLE_0;
   default:
      UNREACHABLE("Invalid component swizzle");
   }
}

static enum pipe_format
get_stencil_format(enum pipe_format format)
{
   /* Note: R8 is not a stencil format, but our copy code will internally cast
    * between R8 and S8. It's simplest to handle here.
    */
   switch (format) {
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_S8_UINT:
      return PIPE_FORMAT_S8_UINT;
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
      return PIPE_FORMAT_X24S8_UINT;
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
      return PIPE_FORMAT_S8X24_UINT;
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      return PIPE_FORMAT_X32_S8X24_UINT;
   default:
      UNREACHABLE("Unsupported depth/stencil format");
   }
}

struct hk_3d {
   unsigned x, y, z;
};

static struct hk_3d
view_denominator(struct hk_image_view *view)
{
   enum pipe_format view_format = hk_format_to_pipe_format(view->vk.format);
   enum pipe_format img_format =
      hk_format_to_pipe_format(view->vk.image->format);

   if (util_format_is_compressed(view_format)) {
      /*
       * We can do an uncompressed view of a compressed image but not the other
       * way around.
       */
      assert(util_format_is_compressed(img_format));
      assert(util_format_get_blockwidth(img_format) ==
             util_format_get_blockwidth(view_format));
      assert(util_format_get_blockheight(img_format) ==
             util_format_get_blockheight(view_format));
      assert(util_format_get_blockdepth(img_format) ==
             util_format_get_blockdepth(view_format));

      return (struct hk_3d){1, 1, 1};
   }

   if (!util_format_is_compressed(img_format)) {
      /* Both formats uncompressed */
      return (struct hk_3d){1, 1, 1};
   }

   /* Else, img is compressed but view is not */
   return (struct hk_3d){
      util_format_get_blockwidth(img_format),
      util_format_get_blockheight(img_format),
      util_format_get_blockdepth(img_format),
   };
}

/* Width in blocks of an uncompressed alias of one level of a block-compressed
 * image. The hardware tiles the alias from its width, so a level of the large
 * miptree that is padded on the left needs its padded block stride: otherwise
 * every tile row after the first lands one tile early. The descriptor has no
 * separate stride for tiled layouts, and the wider alias would change
 * normalized coordinates and size queries, so only driver-internal views
 * (copies, blits, clears), which address texels explicitly, use it.
 *
 * Application views are deliberately excluded for now and stay mis-tiled on
 * padded levels: fixing them needs the logical size kept in the shader (size
 * queries, normalized coordinates, edge clamping) while the descriptor gets
 * the padded width. Until then dEQP texel_view_compatible ...texture.astc_8x*
 * are expected failures (src/asahi/ci/asahi-agx2-fails.txt).
 */
static unsigned
alias_width_el(const struct hk_image_view *view, const struct ail_layout *layout,
               unsigned level, unsigned denom_x)
{
   unsigned width_el = DIV_ROUND_UP(u_minify(layout->width_px, level), denom_x);

   if (layout->tiling == AIL_TILING_GPU &&
       (view->vk.create_flags & VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA))
      width_el = MAX2(width_el, layout->stride_el[level]);

   return width_el;
}

static enum pipe_format
format_for_plane(const struct hk_image_view *view, unsigned view_plane)
{
   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(view->vk.format);

   assert(ycbcr_info || view_plane == 0);
   VkFormat plane_format =
      ycbcr_info ? ycbcr_info->planes[view_plane].format : view->vk.format;

   enum pipe_format p_format = hk_format_to_pipe_format(plane_format);
   if (view->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      p_format = get_stencil_format(p_format);

   return p_format;
}

static void
pack_texture(struct hk_image_view *view, unsigned view_plane,
             enum hk_desc_usage usage, struct agx_texture_packed *out)
{
   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   const uint8_t image_plane = view->planes[view_plane].image_plane;
   struct hk_image_plane *plane = &image->planes[image_plane];
   struct ail_layout *layout = &plane->layout;
   uint64_t base_addr = hk_image_base_address(image, image_plane);

   bool cubes_to_2d = usage != HK_DESC_USAGE_SAMPLED;

   unsigned level = view->vk.base_mip_level;
   unsigned layer = view->vk.base_array_layer;

   enum pipe_format p_format = format_for_plane(view, view_plane);
   const struct util_format_description *desc =
      util_format_description(p_format);

   struct hk_3d denom = view_denominator(view);

   uint8_t format_swizzle[4] = {
      desc->swizzle[0],
      desc->swizzle[1],
      desc->swizzle[2],
      desc->swizzle[3],
   };

   /* Different APIs have different depth/stencil swizzle rules. Vulkan expects
    * R001 behaviour, override here because Mesa's format table is not that.
    */
   if (util_format_is_depth_or_stencil(p_format)) {
      format_swizzle[0] = PIPE_SWIZZLE_X;
      format_swizzle[1] = PIPE_SWIZZLE_0;
      format_swizzle[2] = PIPE_SWIZZLE_0;
      format_swizzle[3] = PIPE_SWIZZLE_1;
   }

   /* We only have a single swizzle for the user swizzle and the format
    * fixup, so compose them now.
    */
   uint8_t out_swizzle[4];
   uint8_t view_swizzle[4] = {
      vk_swizzle_to_pipe(view->vk.swizzle.r),
      vk_swizzle_to_pipe(view->vk.swizzle.g),
      vk_swizzle_to_pipe(view->vk.swizzle.b),
      vk_swizzle_to_pipe(view->vk.swizzle.a),
   };

   unsigned layers = view->vk.layer_count;
   if (view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D) {
      layers = DIV_ROUND_UP(layout->depth_px, denom.z);
   } else if (!cubes_to_2d &&
              (view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
               view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {

      layers /= 6;
   }

   util_format_compose_swizzles(format_swizzle, view_swizzle, out_swizzle);

   agx_pack(out, TEXTURE, cfg) {
      cfg.dimension = translate_image_view_type(
         view->vk.view_type, view->vk.image->samples > 1, layers > 1, usage);
      /* Apple9 texture instructions must match the descriptor's arrayness:
       * a mismatch drops the layer and level coordinates. D3D12 binds 2D
       * views to array declarations and the reverse, so with the
       * hk_apple9_arrayed_2d_views driconf option sampled 1D/2D views are
       * one-layer arrays, and hk_apple9_lower_tex_2d_as_array samples every
       * 2D texture as one. */
      struct hk_device *dev =
         container_of(view->vk.base.device, struct hk_device, vk);
      if (hk_device_physical(dev)->apple9_layered_2d &&
          usage == HK_DESC_USAGE_SAMPLED) {
         if (cfg.dimension == AGX_TEXTURE_DIMENSION_2D)
            cfg.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
         else if (cfg.dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED)
            cfg.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
      }
      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[p_format].channels;
      cfg.type = ail_pixel_format[p_format].type;
      cfg.srgb = util_format_is_srgb(p_format);

      cfg.swizzle_r = agx_channel_from_pipe(out_swizzle[0]);
      cfg.swizzle_g = agx_channel_from_pipe(out_swizzle[1]);
      cfg.swizzle_b = agx_channel_from_pipe(out_swizzle[2]);
      cfg.swizzle_a = agx_channel_from_pipe(out_swizzle[3]);

      if (denom.x > 1) {
         assert(view->vk.level_count == 1);
         assert(view->vk.layer_count == 1);

         cfg.address = base_addr + ail_get_layer_level_B(layout, layer, level);
         cfg.width = alias_width_el(view, layout, level, denom.x);
         cfg.height = DIV_ROUND_UP(u_minify(layout->height_px, level), denom.y);
         cfg.first_level = 0;
         cfg.last_level = 1;
      } else {
         cfg.address = base_addr + ail_get_layer_offset_B(layout, layer);
         cfg.width = layout->width_px;
         cfg.height = layout->height_px;
         cfg.first_level = level;
         cfg.last_level = level + view->vk.level_count - 1;
      }

      /* To implement sparse resident textures, the hardware texture descriptor
       * can instead point to a secondary page table controlled in userspace.
       * This allows remapping pages and - crucially - disabling unmapped pages
       * to read zero and report non-resident with shader residency queries.
       * When we have a sparse map, we need to point to it here.
       *
       * However, there's a wrinkle: when handling uncompressed views of
       * compressed images in the above code, we need to offset the image
       * address to point to the specific mip level rather than use the hardware
       * "first level" field. This ensures the layouts are consistent despite us
       * munging the image dimensions. In that case, we need to also offset the
       * sparse page table accordingly. Of course, the sparse page table is in
       * terms of pages, so this trick only works when the mip level is
       * page-aligned.
       *
       * However, if the mip level is NOT page-aligned, it is in the mip tail by
       * definition. As the mip tail is always resident, there is no need for a
       * sparse page table. So either:
       *
       * 1. We are in the mip tail and don't need a sparse map, or
       * 2. We are not but the level is page-aligned in the sparse map.
       *
       * Either way we're okay.
       */
      if (plane->sparse_map && level < layout->mip_tail_first_lod) {
         unsigned page = 0;
         if (denom.x > 1) {
            page = ail_bytes_to_pages(layout->level_offsets_B[level]);
         }

         cfg.mode = AGX_IMAGE_MODE_SPARSE;
         cfg.address = plane->sparse_map->va->addr +
                       ail_page_to_sparse_index_el(layout, layer, page) *
                          AIL_SPARSE_ELSIZE_B;
      }

      cfg.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
      cfg.unk_mipmapped = layout->levels > 1;
      cfg.srgb_2_channel = cfg.srgb && util_format_colormask(desc) == 0x3;

      if (layout->compressed) {
         cfg.compressed = true;
         cfg.extended = true;
      }

      if (layout->compressed) {
         cfg.acceleration_buffer = base_addr + layout->metadata_offset_B +
                                   (layer * layout->compression_layer_stride_B);
      }

      if (layout->tiling == AIL_TILING_LINEAR &&
          (hk_image_view_type_is_array(view->vk.view_type))) {

         cfg.depth_linear = layers;
         cfg.layer_stride_linear = layout->layer_stride_B - 0x80;
         cfg.extended = true;
      } else {
         assert((layout->tiling != AIL_TILING_LINEAR) || (layers == 1));
         cfg.depth = layers;
      }

      if (view->vk.image->samples > 1) {
         cfg.samples = agx_translate_sample_count(view->vk.image->samples);
      }

      if (layout->tiling == AIL_TILING_LINEAR) {
         cfg.stride = ail_get_linear_stride_B(layout, 0) - 16;
      } else {
         cfg.page_aligned_layers = layout->page_aligned_layers;
      }
   }
}

static void
pack_pbe(struct hk_device *dev, struct hk_image_view *view, unsigned view_plane,
         enum hk_desc_usage usage, struct agx_pbe_packed *out)
{
   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   const uint8_t image_plane = view->planes[view_plane].image_plane;
   struct ail_layout *layout = &image->planes[image_plane].layout;
   uint64_t base_addr = hk_image_base_address(image, image_plane);

   unsigned level = view->vk.base_mip_level;
   unsigned layer = view->vk.base_array_layer;

   enum pipe_format p_format = format_for_plane(view, view_plane);
   const struct util_format_description *desc =
      util_format_description(p_format);

   bool eot =
      usage == HK_DESC_USAGE_BG_EOT || usage == HK_DESC_USAGE_LAYERED_BG_EOT;

   /* The tilebuffer is already in sRGB space if needed. Do not convert for
    * end-of-tile descriptors.
    */
   if (eot)
      p_format = util_format_linear(p_format);

   bool msaa = view->vk.image->samples > 1;
   struct hk_3d denom = view_denominator(view);

   unsigned layers = view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D
                        ? image->vk.extent.depth
                        : view->vk.layer_count;

   agx_pack(out, PBE, cfg) {
      cfg.dimension =
         translate_image_view_type(view->vk.view_type, msaa, layers > 1, usage);
      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[p_format].channels;
      cfg.type = ail_pixel_format[p_format].type;
      cfg.srgb = util_format_is_srgb(p_format);

      assert(desc->nr_channels >= 1 && desc->nr_channels <= 4);

      for (unsigned i = 0; i < desc->nr_channels; ++i) {
         if (desc->swizzle[i] == 0)
            cfg.swizzle_r = i;
         else if (desc->swizzle[i] == 1)
            cfg.swizzle_g = i;
         else if (desc->swizzle[i] == 2)
            cfg.swizzle_b = i;
         else if (desc->swizzle[i] == 3)
            cfg.swizzle_a = i;
      }

      cfg.buffer = base_addr + ail_get_layer_offset_B(layout, layer);
      cfg.unk_mipmapped = layout->levels > 1;

      if (msaa & !eot) {
         /* Multisampled images are bound like buffer textures, with
          * addressing arithmetic to determine the texel to write.
          *
          * Note that the end-of-tile program uses real multisample images
          * with image_write_block instructions.
          */
         unsigned blocksize_B = util_format_get_blocksize(p_format);
         unsigned size_px =
            (layout->size_B - layout->layer_stride_B * layer) / blocksize_B;

         cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
         cfg.layout = AGX_LAYOUT_LINEAR;
         cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
         cfg.height = DIV_ROUND_UP(size_px, cfg.width);
         cfg.stride = (cfg.width * blocksize_B) - 4;
         cfg.layers = 1;
         cfg.levels = 1;

         cfg.buffer += layout->level_offsets_B[level];
         cfg.level = 0;
      } else {
         if (denom.x > 1) {
            assert(denom.z == 1 && "todo how to handle?");
            assert(view->vk.level_count == 1);
            assert(view->vk.layer_count == 1);

            cfg.buffer =
               base_addr + ail_get_layer_level_B(layout, layer, level);
            cfg.width = alias_width_el(view, layout, level, denom.x);
            cfg.height =
               DIV_ROUND_UP(u_minify(layout->height_px, level), denom.y);
            cfg.level = 0;
         } else {
            cfg.buffer = base_addr + ail_get_layer_offset_B(layout, layer);
            cfg.width = layout->width_px;
            cfg.height = layout->height_px;
            cfg.level = level;
         }

         if (layout->tiling == AIL_TILING_LINEAR &&
             (hk_image_view_type_is_array(view->vk.view_type))) {

            cfg.depth_linear = layers;
            cfg.layer_stride_linear = (layout->layer_stride_B - 0x80);
            cfg.extended = true;
         } else {
            assert((layout->tiling != AIL_TILING_LINEAR) || (layers == 1));
            cfg.layers = layers;
         }

         cfg.levels = image->vk.mip_levels;

         if (layout->tiling == AIL_TILING_LINEAR) {
            cfg.stride = ail_get_linear_stride_B(layout, level) - 4;
            assert(cfg.levels == 1);
         } else {
            cfg.page_aligned_layers = layout->page_aligned_layers;
         }

         if (image->vk.samples > 1)
            cfg.samples = agx_translate_sample_count(image->vk.samples);
      }

      if (layout->compressed && usage != HK_DESC_USAGE_EMRT) {
         cfg.compressed = true;
         cfg.extended = true;

         cfg.acceleration_buffer = base_addr + layout->metadata_offset_B +
                                   (layer * layout->compression_layer_stride_B);
      }

      /* When the descriptor isn't extended architecturally, we use
       * the last 8 bytes as a sideband to accelerate image atomics.
       */
      if (!cfg.extended &&
          (layout->writeable_image || usage == HK_DESC_USAGE_EMRT)) {

         if (msaa) {
            assert(denom.x == 1 && "no MSAA of block-compressed");

            cfg.aligned_width_msaa_sw =
               align(u_minify(layout->width_px, level),
                     layout->tilesize_el[level].width_el);

            cfg.sample_count_log2_sw = util_logbase2(image->vk.samples);
         } else {
            cfg.level_offset_sw = ail_get_level_offset_B(layout, cfg.level);
         }

         if (layout->tiling != AIL_TILING_LINEAR) {
            struct ail_tile tile_size = layout->tilesize_el[level];
            cfg.tile_width_sw = tile_size.width_el;
            cfg.tile_height_sw = tile_size.height_el;

            cfg.layer_stride_sw = layout->layer_stride_B;
         }
      }
   };
}

static VkResult
hk_image_view_init(struct hk_device *dev, struct hk_image_view *view,
                   const VkImageViewCreateInfo *pCreateInfo)
{
   VK_FROM_HANDLE(hk_image, image, pCreateInfo->image);
   memset(view, 0, sizeof(*view));

   vk_image_view_init(&dev->vk, &view->vk, pCreateInfo);

   /* First, figure out which image planes we need. For depth/stencil, we only
    * have one aspect viewed at a time.
    */
   if (image->vk.aspects &
       (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {

      view->plane_count = 1;
      view->planes[0].image_plane =
         hk_image_aspects_to_plane(image, view->vk.aspects);
   } else {
      /* For other formats, retrieve the plane count from the aspect mask
       * and then walk through the aspect mask to map each image plane
       * to its corresponding view plane
       */
      assert(util_bitcount(view->vk.aspects) ==
             vk_format_get_plane_count(view->vk.format));
      view->plane_count = 0;
      u_foreach_bit(aspect_bit, view->vk.aspects) {
         view->planes[view->plane_count++].image_plane =
            hk_image_aspects_to_plane(image, 1u << aspect_bit);
      }
   }

   /* Fill in each view plane separately */
   for (unsigned p = 0; p < view->plane_count; p++) {
      pack_texture(view, p, HK_DESC_USAGE_SAMPLED, &view->planes[p].sampled);
      pack_texture(view, p, HK_DESC_USAGE_STORAGE, &view->planes[p].ro_storage);
      pack_texture(view, p, HK_DESC_USAGE_INPUT, &view->planes[p].ia);
      pack_texture(view, p, HK_DESC_USAGE_EMRT, &view->planes[p].emrt_texture);
      pack_texture(view, p, HK_DESC_USAGE_BG_EOT, &view->planes[p].background);
      pack_texture(view, p, HK_DESC_USAGE_LAYERED_BG_EOT,
                   &view->planes[p].layered_background);

      pack_pbe(dev, view, p, HK_DESC_USAGE_STORAGE, &view->planes[p].storage);
      pack_pbe(dev, view, p, HK_DESC_USAGE_BG_EOT, &view->planes[p].eot);
      pack_pbe(dev, view, p, HK_DESC_USAGE_EMRT, &view->planes[p].emrt_pbe);
      pack_pbe(dev, view, p, HK_DESC_USAGE_LAYERED_BG_EOT,
               &view->planes[p].layered_eot);
   }

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyImageView(VkDevice _device, VkImageView imageView,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   VK_FROM_HANDLE(hk_image_view, view, imageView);

   if (!view)
      return;

   simple_mtx_lock(&dev->apple9_pbe_lock);
   hk_apple9_pbe_table_free(&dev->apple9_pbes, &view->apple9_pbes);
   simple_mtx_unlock(&dev->apple9_pbe_lock);
   vk_image_view_finish(&view->vk);
   vk_free2(&dev->vk.alloc, pAllocator, view);
}

bool
hk_apple9_translate_storage_pbe(void *out, const struct agx_pbe_packed *state,
                                enum pipe_format format, const struct ail_layout *layout,
                                uint64_t base, unsigned level)
{
   return agx_apple9_translate_pbe(out, state, format, layout, base, level,
                                  1, false, AGX_TEXTURE_DIMENSION_2D);
}

bool
hk_apple9_native_store_image(const struct hk_image *image, unsigned plane)
{
   /* Sparse images use remapped VA and mip tails; native PBE stores have not
    * been qualified for that storage. Check creation flags, not sparse_map:
    * G16 sparse residency uses apple9_residency instead, and sparse binding
    * alone has neither residency table. Keep all sparse writes in software.
    */
   const VkImageCreateFlags sparse = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                                    VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
   return !(image->vk.create_flags & sparse) &&
          !image->planes[plane].layout.compressed;
}

/* Storage writes use the native representation of the existing PBE half.
 * Per-layer 2D entries keep layer selection in descriptor addressing. The
 * shader never modifies descriptors, so divergent layers cannot race. */
static void
hk_apple9_init_storage_pbe(struct hk_device *dev, struct hk_image_view *view)
{
   view->apple9_store_flags = 0;
   view->apple9_pbes = (struct hk_descriptor_allocation){0};
   view->apple9_pbe_index = 0;
   if (dev->dev.chip != AGX_CHIP_G16G || view->plane_count != 1 ||
       !(view->vk.usage & VK_IMAGE_USAGE_STORAGE_BIT) ||
       !hk_apple9_native_store_format(format_for_plane(view, 0)) ||
       view->vk.image->samples != 1)
      return;
   bool array = view->vk.view_type == VK_IMAGE_VIEW_TYPE_2D_ARRAY ||
                view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
                view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
   if (!array && view->vk.view_type != VK_IMAGE_VIEW_TYPE_2D)
      return;
   /* Opaque capture data does not preserve table allocation addresses. */
   if ((view->vk.create_flags &
                   VK_IMAGE_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT) ||
                 (view->vk.image->create_flags &
                   VK_IMAGE_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT))
      return;
   const struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   unsigned plane = view->planes[0].image_plane;
   const struct ail_layout *layout = &image->planes[plane].layout;
   if (!hk_apple9_native_store_image(image, plane))
      return;
   unsigned count = array ? view->vk.layer_count : 1;
   simple_mtx_lock(&dev->apple9_pbe_lock);
   bool allocated = hk_apple9_pbe_table_alloc(&dev->apple9_pbes, count, &view->apple9_pbes);
   simple_mtx_unlock(&dev->apple9_pbe_lock);
   if (!allocated)
      return;
   uint64_t index = (view->apple9_pbes.addr - dev->dev.shader_base) /
                     AGX_APPLE9_TEXTURE_TABLE_STRIDE;
   if (index + count - 1 > HK_APPLE9_STORAGE_PBE_MASK) {
      simple_mtx_lock(&dev->apple9_pbe_lock);
      hk_apple9_pbe_table_free(&dev->apple9_pbes, &view->apple9_pbes);
      simple_mtx_unlock(&dev->apple9_pbe_lock);
      return;
   }
   uint8_t *entries = view->apple9_pbes.map;
   for (unsigned i = 0; i < count; ++i) {
      uint64_t base = hk_image_base_address(image, plane) +
         ail_get_layer_level_B(layout, view->vk.base_array_layer + i, 0);
      if (!hk_apple9_translate_storage_pbe(entries + i * AGX_APPLE9_TEXTURE_TABLE_STRIDE,
             &view->planes[0].storage, format_for_plane(view, 0), layout, base,
             view->vk.base_mip_level)) {
         simple_mtx_lock(&dev->apple9_pbe_lock);
         hk_apple9_pbe_table_free(&dev->apple9_pbes, &view->apple9_pbes);
         simple_mtx_unlock(&dev->apple9_pbe_lock);
         return;
      }
   }
   view->apple9_pbe_index = index;
   view->apple9_store_flags = array ? HK_APPLE9_STORAGE_LAYER_WRITE_VALID
                                    : HK_APPLE9_STORAGE_WRITE_VALID;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateImageView(VkDevice _device, const VkImageViewCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_image_view *view;
   VkResult result;

   view = vk_alloc2(&dev->vk.alloc, pAllocator, sizeof(*view), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   view->apple9_pbes = (struct hk_descriptor_allocation){0};
   result = hk_image_view_init(dev, view, pCreateInfo);
   if (result != VK_SUCCESS) {
      hk_DestroyImageView(_device, hk_image_view_to_handle(view), pAllocator);
      return result;
   }

   hk_apple9_init_storage_pbe(dev, view);
   *pView = hk_image_view_to_handle(view);

   return VK_SUCCESS;
}

bool
hk_apple9_image_view_eot(const struct hk_image_view *view, bool layered, void *out)
{
   if (view->plane_count != 1)
      return false;
   const struct hk_image *image =
      container_of(view->vk.image, struct hk_image, vk);
   unsigned plane = view->planes[0].image_plane;
   const struct ail_layout *layout = &image->planes[plane].layout;
   uint64_t base = hk_image_base_address(image, plane) +
                   ail_get_layer_level_B(layout, view->vk.base_array_layer, 0);
   /* A 3D view has one array layer but spans all depth slices. The native
    * descriptor, like the legacy PBE above, stores the base-level depth. */
   unsigned layers = view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D
                        ? image->vk.extent.depth : view->vk.layer_count;
   const struct agx_pbe_packed *state =
      layered ? &view->planes[0].layered_eot : &view->planes[0].eot;
   /* Use the same dimension selection as pack_pbe without unpacking the
    * descriptor on the render submission path. */
   enum agx_texture_dimension dimension = translate_image_view_type(
      view->vk.view_type, view->vk.image->samples > 1, layers > 1,
      layered ? HK_DESC_USAGE_LAYERED_BG_EOT : HK_DESC_USAGE_BG_EOT);
   return agx_apple9_translate_pbe(
      out, state, format_for_plane(view, 0), layout, base, view->vk.base_mip_level,
      layered ? layers : 1, layered, dimension);
}

uint64_t
hk_apple9_view_residency(const struct hk_device *dev,
                         const struct hk_image_view *view, unsigned plane)
{
   const struct hk_image *image =
      container_of(view->vk.image, struct hk_image, vk);
   struct agx_bo *table =
      image->planes[view->planes[plane].image_plane].apple9_residency;

   return (table ? table : dev->apple9_resident)->va->addr;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetImageViewOpaqueCaptureDescriptorDataEXT(
   UNUSED VkDevice device,
   UNUSED const VkImageViewCaptureDescriptorDataInfoEXT *pInfo,
   void *pData)
{
   /* Capture/replay views do not allocate layer PBE caches; their descriptor
    * addresses remain image-owned. */
   memset(pData, 0, HK_IMAGE_VIEW_CAPTURE_DATA_SIZE);
   return VK_SUCCESS;
}
