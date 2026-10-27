/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "hk_apple9.h"
#include "hk_cmd_buffer.h"
#include "hk_apple9_graphics.h"

#include "asahi/lib/agx_apple9_compute.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_helpers.h"
#include "asahi/layout/layout.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "libagx.h"
#include "nir_builder.h"
#include "nir_format_convert.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "util/half_float.h"
#include "util/u_math.h"
#include "util/format/u_format.h"
#include "pipe/p_defines.h"
#include <stdio.h>
#include <stdlib.h>
#define HK_APPLE9_BUFFER_ADDRESSES_OFFSET \
   ALIGN_POT(HK_APPLE9_BUFFER_SIZES_OFFSET + 64 * sizeof(uint32_t), 8)
#define A9FAIL() do { if (getenv("AGX_APPLE9_TRACE_REJECT")) fprintf(stderr, "APPLE9_DISPATCH_FAIL %s:%d\n", __func__, __LINE__); return false; } while (0)

struct hk_apple9_nir_context {
   struct vk_descriptor_set_layout *const *set_layouts;
   uint32_t set_layout_count;
   struct hk_apple9_shader_mapping *mapping;
   const char **reason;
   bool io_only;
   bool textures_only;
};

enum hk_apple9_map_kind {
   HK_APPLE9_MAP_UBO,
   HK_APPLE9_MAP_SSBO,
   HK_APPLE9_MAP_TEXTURE,
   HK_APPLE9_MAP_SAMPLER,
   HK_APPLE9_MAP_STORAGE_IMAGE,
};

static bool
hk_apple9_fail(struct hk_apple9_nir_context *ctx, const char *reason)
{
   if (!*ctx->reason)
      *ctx->reason = reason;
   return false;
}

static const struct hk_descriptor_set_binding_layout *
hk_apple9_binding_layout(struct hk_apple9_nir_context *ctx, uint32_t set,
                         uint32_t binding)
{
   if (set >= ctx->set_layout_count || !ctx->set_layouts[set]) {
      hk_apple9_fail(ctx, "Apple9 compute referenced an invalid descriptor set");
      return NULL;
   }

   const struct hk_descriptor_set_layout *layout =
      vk_to_hk_descriptor_set_layout(ctx->set_layouts[set]);
   if (binding >= layout->binding_count) {
      hk_apple9_fail(ctx, "Apple9 compute referenced an invalid descriptor binding");
      return NULL;
   }
   return &layout->binding[binding];
}

static bool
hk_apple9_map_descriptor(struct hk_apple9_nir_context *ctx,
                         enum hk_apple9_map_kind kind, uint32_t set,
                         uint32_t binding, uint32_t element,
                         const struct hk_descriptor_set_binding_layout *layout,
                         uint32_t *slot)
{
   struct hk_apple9_descriptor *entries;
   uint8_t *count;
   unsigned limit;
   switch (kind) {
   case HK_APPLE9_MAP_UBO:
      entries = ctx->mapping->ubo;
      count = &ctx->mapping->ubo_count;
      limit = ARRAY_SIZE(ctx->mapping->ubo);
      break;
   case HK_APPLE9_MAP_SSBO:
      entries = ctx->mapping->ssbo;
      count = &ctx->mapping->ssbo_count;
      limit = ARRAY_SIZE(ctx->mapping->ssbo);
      break;
   case HK_APPLE9_MAP_TEXTURE:
      entries = ctx->mapping->textures;
      count = &ctx->mapping->texture_count;
      limit = ARRAY_SIZE(ctx->mapping->textures);
      break;
   case HK_APPLE9_MAP_STORAGE_IMAGE:
      entries = ctx->mapping->storage_images;
      count = &ctx->mapping->storage_image_count;
      limit = ARRAY_SIZE(ctx->mapping->storage_images);
      break;
   default:
      entries = ctx->mapping->samplers;
      count = &ctx->mapping->sampler_count;
      limit = ARRAY_SIZE(ctx->mapping->samplers);
      break;
   }

   for (unsigned i = 0; i < *count; ++i) {
      if (entries[i].set == set && entries[i].binding == binding &&
          entries[i].element == element && entries[i].type == layout->type) {
         *slot = i;
         return true;
      }
   }
   if (*count == limit)
      return hk_apple9_fail(ctx, "Apple9 compute descriptor capacity exceeded");

   *slot = (*count)++;
   entries[*slot] = (struct hk_apple9_descriptor){
      .offset = layout->offset + element * layout->stride,
      .binding = binding,
      .element = element,
      .dynamic_index = layout->dynamic_buffer_index,
      .set = set,
      .type = layout->type,
   };
   return true;
}

static bool
hk_apple9_resolve_binding(struct hk_apple9_nir_context *ctx, nir_def *def,
                          enum hk_apple9_map_kind kind, uint32_t *slot)
{
   nir_scalar original = nir_scalar_resolved(def, 0);
   if (nir_scalar_is_const(original)) {
      uint32_t value = nir_scalar_as_uint(original);
      unsigned count = kind == HK_APPLE9_MAP_UBO
         ? ctx->mapping->ubo_count : ctx->mapping->ssbo_count;
      if (value < count) {
         *slot = value;
         return true;
      }
   }

   uint32_t element = 0;
   nir_def *cursor = def;
   for (;;) {
      nir_src src = { .ssa = cursor };
      nir_intrinsic_instr *intr = nir_src_as_intrinsic(src);
      if (!intr) {
         if (getenv("AGX_APPLE9_TRACE_REJECT")) {
            fprintf(stderr, "APPLE9_DESCRIPTOR_REJECT io_only=%u count=%u constant=%u\n",
                    ctx->io_only,
                    kind == HK_APPLE9_MAP_UBO ? ctx->mapping->ubo_count : ctx->mapping->ssbo_count,
                    nir_scalar_is_const(original));
            nir_print_instr(nir_def_instr(cursor), stderr);
         }
         return hk_apple9_fail(ctx, "Apple9 descriptor selection is not constant");
      }

      if (intr->intrinsic == nir_intrinsic_vulkan_resource_reindex) {
         if (!nir_src_is_const(intr->src[1]))
            return hk_apple9_fail(ctx, "Apple9 descriptor array selection is not constant");
         element += nir_src_as_uint(intr->src[1]);
         cursor = intr->src[0].ssa;
         continue;
      }
      if (intr->intrinsic != nir_intrinsic_vulkan_resource_index)
         return hk_apple9_fail(ctx, "Apple9 descriptor selection is not constant");
      if (!nir_src_is_const(intr->src[0]))
         return hk_apple9_fail(ctx, "Apple9 descriptor array selection is not constant");
      element += nir_src_as_uint(intr->src[0]);

      uint32_t set = nir_intrinsic_desc_set(intr);
      uint32_t binding = nir_intrinsic_binding(intr);
      const struct hk_descriptor_set_binding_layout *layout =
         hk_apple9_binding_layout(ctx, set, binding);
      if (!layout)
         return false;
      if (layout->array_size == 0 || element >= layout->array_size)
         return hk_apple9_fail(ctx, "Apple9 descriptor array element is out of range");

      if (kind == HK_APPLE9_MAP_SSBO &&
          layout->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
          layout->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
         return hk_apple9_fail(ctx, "Apple9 SSBO binding has the wrong descriptor type");
      if (kind == HK_APPLE9_MAP_UBO &&
          layout->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
          layout->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC &&
          layout->type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
         return hk_apple9_fail(ctx, "Apple9 UBO binding has the wrong descriptor type");
      if (kind == HK_APPLE9_MAP_TEXTURE &&
          layout->type != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
          layout->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
          layout->type != VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT &&
          layout->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
         return hk_apple9_fail(ctx, "Apple9 texture binding has an unsupported descriptor type");
      if (kind == HK_APPLE9_MAP_SAMPLER &&
          layout->type != VK_DESCRIPTOR_TYPE_SAMPLER &&
          layout->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         return hk_apple9_fail(ctx, "Apple9 sampler binding has the wrong descriptor type");

      return hk_apple9_map_descriptor(ctx, kind, set, binding, element,
                                      layout, slot);
   }
}

static bool
hk_apple9_resolve_deref(struct hk_apple9_nir_context *ctx, nir_src *src,
                        enum hk_apple9_map_kind kind, uint32_t *slot)
{
   nir_deref_instr *deref = nir_src_as_deref(*src);
   if (!deref)
      return hk_apple9_fail(ctx, "Apple9 texture descriptor is not a deref");

   uint32_t element = 0;
   if (deref->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(deref->arr.index))
         return hk_apple9_fail(ctx, "Apple9 texture array selection is not constant");
      element = nir_src_as_uint(deref->arr.index);
      deref = nir_deref_instr_parent(deref);
   }
   if (!deref || deref->deref_type != nir_deref_type_var)
      return hk_apple9_fail(ctx, "Apple9 texture descriptor is not a variable");

   nir_variable *var = deref->var;
   uint32_t set = var->data.descriptor_set;
   uint32_t binding = var->data.binding;
   const struct hk_descriptor_set_binding_layout *layout =
      hk_apple9_binding_layout(ctx, set, binding);
   if (!layout)
      return false;
   if (layout->array_size == 0 || element >= layout->array_size)
      return hk_apple9_fail(ctx, "Apple9 texture array element is out of range");

   if (kind == HK_APPLE9_MAP_TEXTURE &&
      layout->type != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
      layout->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
      layout->type != VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT &&
      layout->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
      return hk_apple9_fail(ctx, "Apple9 texture binding has an unsupported descriptor type");
   if (kind == HK_APPLE9_MAP_SAMPLER &&
      layout->type != VK_DESCRIPTOR_TYPE_SAMPLER &&
      layout->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
      return hk_apple9_fail(ctx, "Apple9 sampler binding has the wrong descriptor type");
   if (kind == HK_APPLE9_MAP_STORAGE_IMAGE &&
       layout->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
       layout->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
      return hk_apple9_fail(ctx, "Apple9 storage image binding has the wrong descriptor type");

   return hk_apple9_map_descriptor(ctx, kind, set, binding, element,
                                   layout, slot);
}

static bool
hk_apple9_lower_tex(nir_builder *b, nir_tex_instr *tex, void *data)
{
   struct hk_apple9_nir_context *ctx = data;
   if (*ctx->reason)
      return false;

   int src = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (src >= 0 || nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0) {
      hk_apple9_fail(ctx, "Apple9 bindless texture handles are unsupported");
      return false;
   }

   src = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (src >= 0) {
      uint32_t slot;
      if (!hk_apple9_resolve_deref(ctx, &tex->src[src].src,
                                   HK_APPLE9_MAP_TEXTURE, &slot))
         return false;
      nir_tex_instr_remove_src(tex, src);
      tex->texture_index = slot;
      tex->texture_non_uniform = false;
   }

   src = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (src >= 0) {
      uint32_t slot;
      if (!hk_apple9_resolve_deref(ctx, &tex->src[src].src,
                                   HK_APPLE9_MAP_SAMPLER, &slot))
         return false;
      nir_tex_instr_remove_src(tex, src);
      tex->sampler_index = slot;
      tex->sampler_non_uniform = false;
   }

   src = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
   if (src >= 0) {
      if (!nir_src_is_const(tex->src[src].src) || nir_src_as_uint(tex->src[src].src)) {
         hk_apple9_fail(ctx, "Apple9 dynamic texture offsets are unsupported");
         return false;
      }
      nir_tex_instr_remove_src(tex, src);
   }
   src = nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset);
   if (src >= 0) {
      if (!nir_src_is_const(tex->src[src].src) || nir_src_as_uint(tex->src[src].src)) {
         hk_apple9_fail(ctx, "Apple9 dynamic sampler offsets are unsupported");
         return false;
      }
      nir_tex_instr_remove_src(tex, src);
   }
   if (tex->op == nir_texop_lod_bias) {
      b->cursor = nir_before_instr(&tex->instr);
      nir_def *bias = nir_load_ubo(
         b, 1, 32, nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_APPLE9_SAMPLER_BIAS_OFFSET + tex->sampler_index * 4),
          .align_mul = 4,
          .range_base = AGX_APPLE9_SAMPLER_BIAS_OFFSET + tex->sampler_index * 4,
          .range = 4);
      nir_def_replace(&tex->def, nir_f2fN(b, bias, tex->def.bit_size));
   } else if (tex->op == nir_texop_txl || tex->op == nir_texop_txb) {
      int lod = nir_tex_instr_src_index(
         tex, tex->op == nir_texop_txl ? nir_tex_src_lod : nir_tex_src_bias);
      if (lod >= 0 && tex->src[lod].src.ssa->bit_size != 32) {
         b->cursor = nir_before_instr(&tex->instr);
         nir_src_rewrite(&tex->src[lod].src,
                        nir_f2f32(b, tex->src[lod].src.ssa));
      }
   }
   return true;
}

/* Resolve the storage-image descriptor address from the root descriptor table.
 * Returns a 64-bit GPU address pointing at the hk_storage_image_descriptor. */
static nir_def *
hk_apple9_storage_image_descriptor_addr(nir_builder *b,
                                         struct hk_apple9_nir_context *ctx,
                                         nir_src *deref_src, uint32_t *slot)
{
   if (!hk_apple9_resolve_deref(ctx, deref_src,
                               HK_APPLE9_MAP_STORAGE_IMAGE, slot))
      return NULL;
   const struct hk_apple9_descriptor *entry = &ctx->mapping->storage_images[*slot];
   unsigned set_offset = hk_root_descriptor_offset(sets[entry->set]);
   nir_def *set = nir_load_ubo(
      b, 1, 64, nir_imm_int(b, HK_APPLE9_ROOT_BINDING),
      nir_imm_int(b, set_offset), .align_mul = 8,
      .range_base = set_offset, .range = sizeof(uint64_t));
   return nir_iadd_imm(b, set, entry->offset);
}

/* Vulkan's storage-without-format set. The descriptor supplies the concrete
 * format; the SPIR-V image type only supplies the shader's scalar type.
 */
static const enum pipe_format hk_apple9_formatless_store_formats[] = {
   PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_SNORM,
   PIPE_FORMAT_R8G8B8A8_UINT, PIPE_FORMAT_R8G8B8A8_SINT,
   PIPE_FORMAT_R32_UINT, PIPE_FORMAT_R32_SINT, PIPE_FORMAT_R32_FLOAT,
   PIPE_FORMAT_R32G32_UINT, PIPE_FORMAT_R32G32_SINT, PIPE_FORMAT_R32G32_FLOAT,
   PIPE_FORMAT_R32G32B32A32_UINT, PIPE_FORMAT_R32G32B32A32_SINT,
   PIPE_FORMAT_R32G32B32A32_FLOAT,
   PIPE_FORMAT_R16G16B16A16_UINT, PIPE_FORMAT_R16G16B16A16_SINT,
   PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R16G16_FLOAT,
   PIPE_FORMAT_R11G11B10_FLOAT, PIPE_FORMAT_R16_FLOAT,
   PIPE_FORMAT_R16G16B16A16_UNORM, PIPE_FORMAT_R10G10B10A2_UNORM,
   PIPE_FORMAT_R16G16_UNORM, PIPE_FORMAT_R8G8_UNORM,
   PIPE_FORMAT_R16_UNORM, PIPE_FORMAT_R8_UNORM,
   PIPE_FORMAT_R16G16B16A16_SNORM, PIPE_FORMAT_R16G16_SNORM,
   PIPE_FORMAT_R8G8_SNORM, PIPE_FORMAT_R16_SNORM, PIPE_FORMAT_R8_SNORM,
   PIPE_FORMAT_R16G16_SINT, PIPE_FORMAT_R8G8_SINT,
   PIPE_FORMAT_R16_SINT, PIPE_FORMAT_R8_SINT,
   PIPE_FORMAT_R10G10B10A2_UINT, PIPE_FORMAT_R16G16_UINT,
   PIPE_FORMAT_R8G8_UINT, PIPE_FORMAT_R16_UINT, PIPE_FORMAT_R8_UINT,
   PIPE_FORMAT_A8_UNORM,
};

bool
hk_apple9_formatless_store_supported(enum pipe_format format)
{
   for (unsigned i = 0; i < ARRAY_SIZE(hk_apple9_formatless_store_formats); ++i) {
      if (hk_apple9_formatless_store_formats[i] == format)
         return true;
   }
   return false;
}

static uint32_t
hk_apple9_image_format_key(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   struct agx_texture_packed texture;
   agx_pack(&texture, TEXTURE, cfg) {
      cfg.width = cfg.height = cfg.depth = 1;
      cfg.channels = ail_pixel_format[format].channels;
      cfg.type = ail_pixel_format[format].type;
      cfg.swizzle_r = agx_channel_from_pipe(desc->swizzle[0]);
      cfg.swizzle_g = agx_channel_from_pipe(desc->swizzle[1]);
      cfg.swizzle_b = agx_channel_from_pipe(desc->swizzle[2]);
      cfg.swizzle_a = agx_channel_from_pipe(desc->swizzle[3]);
   }
   uint32_t word;
   memcpy(&word, &texture, sizeof(word));
   return word & 0x0fffffc0;
}

static void
hk_apple9_store_image_texel(nir_builder *b, nir_def *packed,
                            nir_def *address, unsigned bpp, unsigned access)
{
   if (bpp <= 2) {
      nir_store_global(b, nir_u2uN(b, nir_channel(b, packed, 0), bpp * 8), address,
                       .write_mask = 1, .access = access, .align_mul = bpp);
   } else {
      unsigned count = MIN2(bpp / 4, packed->num_components);
      nir_store_global(b, nir_trim_vector(b, packed, count), address,
                       .write_mask = BITFIELD_MASK(count), .access = access,
                       .align_mul = MIN2(bpp, 16));
   }
}

/* Compute the global memory address for an image texel at coordinate xy,
 * using either the linear or tiled layout from the PBE descriptor.
 * bytes_per_pixel is the format block size. */
static nir_def *
hk_apple9_image_texel_addr(nir_builder *b, nir_def *descriptor,
                            nir_def *xy, nir_def *bytes_per_pixel,
                            bool is_1d, bool is_layered)
{
   nir_def *pbe = nir_iadd_imm(
      b, descriptor, offsetof(struct hk_storage_image_descriptor, pbe));
   nir_def *header = nir_load_global_constant(b, 1, 32, pbe, .align_mul = 8);
   nir_push_if(b, nir_ieq_imm(b, nir_ubfe_imm(b, header, 4, 2), AGX_LAYOUT_LINEAR));
   /* PBE bits 64..99 encode buffer>>4; bits 104..124 encode pitch-4.
    * Unlike the tiled atomic-address helper, linear 2D uses byte pitch. */
   nir_def *word = nir_load_global_constant(
      b, 1, 64, nir_iadd_imm(b, pbe, 8), .align_mul = 8);
   nir_def *base = nir_ishl_imm(b, nir_iand_imm(b, word, BITFIELD64_MASK(36)), 4);
   nir_def *pitch = nir_iadd_imm(
      b, nir_iand_imm(b, nir_ushr_imm(b, word, 40), BITFIELD_MASK(21)), 4);
   nir_def *linear = nir_iadd(
      b, base, nir_iadd(
         b, nir_imul(b, nir_u2u64(b, nir_channel(b, xy, 1)), pitch),
         nir_imul(b, nir_u2u64(b, nir_channel(b, xy, 0)),
                  nir_u2u64(b, bytes_per_pixel))));
   nir_push_else(b, NULL);
   nir_def *coord4 = is_layered
      ? (is_1d
            ? nir_vec4(b, nir_channel(b, xy, 0), nir_channel(b, xy, 2),
                       nir_imm_int(b, 0), nir_imm_int(b, 0))
            : nir_vec4(b, nir_channel(b, xy, 0), nir_channel(b, xy, 1),
                       nir_channel(b, xy, 2), nir_imm_int(b, 0)))
      : nir_vec4(b, nir_channel(b, xy, 0), nir_channel(b, xy, 1),
                 nir_imm_int(b, 0), nir_imm_int(b, 0));
   nir_def *tiled = libagx_image_texel_address(
      b, pbe, coord4,
      nir_imm_int(b, 0), bytes_per_pixel,
      nir_imm_bool(b, is_1d),
      nir_imm_bool(b, false), nir_imm_bool(b, is_layered),
      nir_imm_bool(b, false));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, linear, tiled);
}

static bool
hk_apple9_lower_image_store(nir_builder *b, nir_intrinsic_instr *intr,
                            struct hk_apple9_nir_context *ctx)
{
   enum pipe_format format = nir_intrinsic_format(intr);
   bool formatless = format == PIPE_FORMAT_NONE;
   unsigned bpp = util_format_get_blocksize(format);
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool is_buffer = dim == GLSL_SAMPLER_DIM_BUF;
   bool is_array = nir_intrinsic_image_array(intr);
   bool is_1d = dim == GLSL_SAMPLER_DIM_1D;
   bool is_3d = dim == GLSL_SAMPLER_DIM_3D;
   bool is_cube = dim == GLSL_SAMPLER_DIM_CUBE;
   bool is_layered = is_array || is_3d || is_cube;

   /* Reject unsupported configurations */
   if ((dim != GLSL_SAMPLER_DIM_2D && dim != GLSL_SAMPLER_DIM_1D &&
        dim != GLSL_SAMPLER_DIM_3D && dim != GLSL_SAMPLER_DIM_CUBE &&
        !is_buffer) ||
       bpp == 0 || bpp > 16 ||
       intr->src[3].ssa->bit_size != 32 ||
       !nir_src_is_const(intr->src[4]) || nir_src_as_uint(intr->src[4]) ||
       (nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE)))
      return hk_apple9_fail(ctx, "Apple9 image store: unsupported config");

   uint32_t slot;
   nir_def *descriptor = hk_apple9_storage_image_descriptor_addr(
      b, ctx, &intr->src[0], &slot);
   if (!descriptor)
      return false;

   nir_def *texture = nir_iadd_imm(
      b, descriptor, offsetof(struct hk_storage_image_descriptor, tex));
   nir_def *pixel_bytes = nir_imm_int(b, bpp);
   nir_def *packed = NULL;
   nir_def *format_valid = nir_imm_bool(b, true);
   if (formatless) {
      nir_def *word = nir_load_global_constant(
         b, 1, 32, texture, .align_mul = 4);
      nir_def *key = nir_iand_imm(b, word, 0x0fffffc0);
      pixel_bytes = nir_imm_int(b, 1);
      packed = nir_imm_zero(b, 4, 32);
      format_valid = nir_imm_bool(b, false);
      for (unsigned i = 0; i < ARRAY_SIZE(hk_apple9_formatless_store_formats); ++i) {
         enum pipe_format candidate = hk_apple9_formatless_store_formats[i];
         nir_def *match = nir_ieq_imm(b, key, hk_apple9_image_format_key(candidate));
         nir_def *value = nir_format_pack_rgba(b, candidate, intr->src[3].ssa);
         packed = nir_bcsel(b, match, nir_pad_vec4(b, value), packed);
         pixel_bytes = nir_bcsel(
            b, match, nir_imm_int(b, util_format_get_blocksize(candidate)), pixel_bytes);
         format_valid = nir_ior(b, format_valid, match);
      }
   }
   unsigned nr_size_comps =
      is_buffer ? 1 : (is_1d ? (is_array ? 2 : 1) : (is_layered ? 3 : 2));
   nir_def *size = libagx_txs(
      b, texture, nir_imm_intN_t(b, 0, 16), nir_imm_int(b, nr_size_comps),
      nir_imm_bool(b, is_buffer), nir_imm_bool(b, is_1d),
      nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_2D || is_cube),
      nir_imm_bool(b, is_cube), nir_imm_bool(b, is_array));

   /* Bounds check: suppress out-of-bounds stores (null descriptors have 0 dims) */
   nir_def *coord = nir_channels(b, intr->src[1].ssa,
                                  BITFIELD_MASK(nr_size_comps));
   nir_push_if(b, nir_iand(b, format_valid,
      nir_ball(b, nir_ult(b, coord,
                         nir_channels(b, size, BITFIELD_MASK(nr_size_comps))))));

   nir_def *xy = nir_channels(b, intr->src[1].ssa,
                               is_1d ? 0x1 : 0x3);
   if (is_1d)
      xy = nir_vec2(b, nir_channel(b, xy, 0), nir_imm_int(b, 0));

   nir_def *address = is_buffer ? libagx_buffer_texel_address(
      b, nir_iadd_imm(b, descriptor, offsetof(struct hk_storage_image_descriptor, pbe)),
      intr->src[1].ssa, pixel_bytes) : hk_apple9_image_texel_addr(
      b, descriptor, is_layered
         ? nir_vec3(b, nir_channel(b, xy, 0), nir_channel(b, xy, 1),
                    nir_channel(b, intr->src[1].ssa, is_1d ? 1 : 2))
         : xy,
      pixel_bytes, is_1d, is_layered);

   /* Write the data. Use nir_format_pack_rgba to handle format conversion
    * from the shader's 32-bit values to the texel format. For 32-bit pure
    * formats (R32, RG32, RGBA32) this is an identity operation. For narrower
    * formats (RGBA8, RG16, etc.) it packs the values appropriately. */
   if (formatless) {
      for (unsigned bytes = 1; bytes <= 16; bytes *= 2) {
         nir_push_if(b, nir_ieq_imm(b, pixel_bytes, bytes));
         hk_apple9_store_image_texel(
            b, packed, address, bytes, nir_intrinsic_access(intr));
         nir_pop_if(b, NULL);
      }
   } else {
      packed = nir_format_pack_rgba(b, format, intr->src[3].ssa);
      hk_apple9_store_image_texel(
         b, packed, address, bpp, nir_intrinsic_access(intr));
   }

   nir_pop_if(b, NULL);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
hk_apple9_lower_image_load(nir_builder *b, nir_intrinsic_instr *intr,
                            struct hk_apple9_nir_context *ctx)
{
   enum pipe_format format = nir_intrinsic_format(intr);
   if (format == PIPE_FORMAT_NONE)
      return hk_apple9_fail(ctx, "Apple9 formatless image loads are unsupported");
   unsigned bpp = util_format_get_blocksize(format);
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool is_buffer = dim == GLSL_SAMPLER_DIM_BUF;
   bool is_array = nir_intrinsic_image_array(intr);
   bool is_1d = dim == GLSL_SAMPLER_DIM_1D;
   bool is_3d = dim == GLSL_SAMPLER_DIM_3D;
   bool is_cube = dim == GLSL_SAMPLER_DIM_CUBE;
   bool is_layered = is_array || is_3d || is_cube;

   if ((dim != GLSL_SAMPLER_DIM_2D && dim != GLSL_SAMPLER_DIM_1D &&
        dim != GLSL_SAMPLER_DIM_3D && dim != GLSL_SAMPLER_DIM_CUBE &&
        !is_buffer) ||
       bpp == 0 || bpp > 16 ||
       (nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE)))
      return hk_apple9_fail(ctx, "Apple9 image load: unsupported config");

   uint32_t slot;
   nir_def *descriptor = hk_apple9_storage_image_descriptor_addr(
      b, ctx, &intr->src[0], &slot);
   if (!descriptor)
      return false;

   nir_def *texture = nir_iadd_imm(
      b, descriptor, offsetof(struct hk_storage_image_descriptor, tex));
   unsigned nr_size_comps =
      is_buffer ? 1 : (is_1d ? (is_array ? 2 : 1) : (is_layered ? 3 : 2));
   nir_def *size = libagx_txs(
      b, texture, nir_imm_intN_t(b, 0, 16), nir_imm_int(b, nr_size_comps),
      nir_imm_bool(b, is_buffer), nir_imm_bool(b, is_1d),
      nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_2D || is_cube),
      nir_imm_bool(b, is_cube), nir_imm_bool(b, is_array));

   /* Bounds check: return zero for out-of-bounds loads */
   nir_def *coord = nir_channels(b, intr->src[1].ssa,
                                  BITFIELD_MASK(nr_size_comps));
   nir_push_if(b, nir_ball(b, nir_ult(b, coord,
                                        nir_channels(b, size,
                                                     BITFIELD_MASK(nr_size_comps)))));

   nir_def *xy = nir_channels(b, intr->src[1].ssa,
                               is_1d ? 0x1 : 0x3);
   if (is_1d || is_buffer)
      xy = nir_vec2(b, nir_channel(b, xy, 0), nir_imm_int(b, 0));

   nir_def *address = is_buffer ? libagx_buffer_texel_address(
      b, nir_iadd_imm(b, descriptor, offsetof(struct hk_storage_image_descriptor, pbe)),
      intr->src[1].ssa, nir_imm_int(b, bpp)) : hk_apple9_image_texel_addr(
      b, descriptor, is_layered
         ? nir_vec3(b, nir_channel(b, xy, 0), nir_channel(b, xy, 1),
                    nir_channel(b, intr->src[1].ssa, is_1d ? 1 : 2))
         : xy,
      nir_imm_int(b, bpp), is_1d, is_layered);

   /* Read the data. Use nir_format_unpack_rgba to handle format conversion
    * from the texel format to the shader's 32-bit values. For 32-bit pure
    * formats this is identity. For narrower formats it unpacks appropriately. */
   nir_def *loaded;
   unsigned comp_bits = intr->def.bit_size;

   if (bpp <= 2) {
      /* Load 1-2 bytes, widen to 32-bit for the unpack helper */
      unsigned load_bits = (bpp <= 1) ? 8 : 16;
      nir_def *packed = nir_load_global(b, 1, load_bits, address,
                                         .align_mul = bpp);
      packed = nir_u2u32(b, packed);
      loaded = nir_format_unpack_rgba(b, packed, format);
   } else if (bpp < 16 || util_format_get_component_bits(
                               format, UTIL_FORMAT_COLORSPACE_RGB, 0) < 32) {
      /* Load as uint32 dwords, then unpack to shader values.
       * nir_format_unpack_rgba handles both single-dword packed formats
       * (RGBA8, R10G10B10A2) and multi-dword formats (RGBA16F, RGBA16)
       * where it uses nir_format_bitcast_uvec_unmasked on the full vector. */
      unsigned nr_dwords = (bpp + 3) / 4;
      nir_def *packed = nir_load_global(b, nr_dwords, 32, address,
                                         .align_mul = MIN2(bpp, 16));
      loaded = nir_format_unpack_rgba(b, packed, format);
   } else {
      unsigned nr_comps = bpp / (comp_bits / 8);
      if (nr_comps == 0) nr_comps = 1;
      if (nr_comps > intr->def.num_components)
         nr_comps = intr->def.num_components;

      loaded = nir_load_global(b, nr_comps, comp_bits, address,
                                .align_mul = MIN2(bpp, 16));

      /* Pad to the expected number of components with zeros */
      if (nr_comps < intr->def.num_components) {
         nir_def *comps[4];
         for (unsigned i = 0; i < intr->def.num_components; i++) {
            if (i < nr_comps)
               comps[i] = nir_channel(b, loaded, i);
            else
               comps[i] = nir_imm_intN_t(b, 0, comp_bits);
         }
         loaded = nir_vec(b, comps, intr->def.num_components);
      }
   }

   nir_push_else(b, NULL);
   nir_def *zero = nir_imm_zero(b, intr->def.num_components, comp_bits);
   nir_pop_if(b, NULL);
   nir_def *result = nir_if_phi(b, loaded, zero);
   nir_def_replace(&intr->def, result);
   return true;
}

static bool
hk_apple9_lower_image_size(nir_builder *b, nir_intrinsic_instr *intr,
                            struct hk_apple9_nir_context *ctx)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool is_buffer = dim == GLSL_SAMPLER_DIM_BUF;
   bool is_array = nir_intrinsic_image_array(intr);
   bool is_1d = dim == GLSL_SAMPLER_DIM_1D;
   bool is_cube = dim == GLSL_SAMPLER_DIM_CUBE;

   uint32_t slot;
   nir_def *descriptor = hk_apple9_storage_image_descriptor_addr(
      b, ctx, &intr->src[0], &slot);
   if (!descriptor)
      return false;

   nir_def *texture = nir_iadd_imm(
      b, descriptor, offsetof(struct hk_storage_image_descriptor, tex));
   nir_def *size = libagx_txs(
      b, texture, nir_imm_intN_t(b, 0, 16),
      nir_imm_int(b, intr->def.num_components),
      nir_imm_bool(b, is_buffer), nir_imm_bool(b, is_1d),
      nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_2D || is_cube),
      nir_imm_bool(b, is_cube), nir_imm_bool(b, is_array));

   nir_def *result = nir_channels(b, size,
                                   BITFIELD_MASK(intr->def.num_components));

   /* For cube images, the descriptor stores total array layers (6 per cube).
    * imageSize should return the number of cubes, not total layers. */
   if (is_cube && is_array && intr->def.num_components >= 3) {
      nir_def *comps[4];
      for (unsigned i = 0; i < intr->def.num_components; i++) {
         comps[i] = nir_channel(b, result, i);
      }
      comps[2] = nir_udiv_imm(b, comps[2], 6);
      result = nir_vec(b, comps, intr->def.num_components);
   }

   nir_def_replace(&intr->def, result);
   return true;
}

static bool
hk_apple9_lower_image_samples(nir_builder *b, nir_intrinsic_instr *intr,
                               struct hk_apple9_nir_context *ctx)
{
   /* Non-MSAA images always return 1 sample */
   nir_def_replace(&intr->def, nir_imm_int(b, 1));
   return true;
}

static bool
hk_apple9_lower_intrinsic(nir_builder *b, nir_instr *instr, void *data)
{
   struct hk_apple9_nir_context *ctx = data;
   if (*ctx->reason || instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(&intr->instr);

   if (!ctx->io_only && intr->intrinsic == nir_intrinsic_image_deref_store)
      return hk_apple9_lower_image_store(b, intr, ctx);

   if (!ctx->io_only && intr->intrinsic == nir_intrinsic_image_deref_load)
      return hk_apple9_lower_image_load(b, intr, ctx);

   if (!ctx->io_only && intr->intrinsic == nir_intrinsic_image_deref_size)
      return hk_apple9_lower_image_size(b, intr, ctx);

   if (!ctx->io_only && intr->intrinsic == nir_intrinsic_image_deref_samples)
      return hk_apple9_lower_image_samples(b, intr, ctx);

   if (ctx->textures_only) {
      switch (intr->intrinsic) {
      case nir_intrinsic_image_deref_load:
      case nir_intrinsic_image_deref_sparse_load:
      case nir_intrinsic_image_deref_store:
      case nir_intrinsic_image_deref_atomic:
      case nir_intrinsic_image_deref_atomic_swap:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_deref_samples:
      case nir_intrinsic_image_deref_store_block_agx:
         return hk_apple9_fail(ctx, "Apple9 storage images are not yet supported");
      default:
         return false;
      }
   }

   if (intr->intrinsic == nir_intrinsic_load_push_constant) {
      if (ctx->io_only)
         return false;
      unsigned base = nir_intrinsic_base(intr) + hk_root_descriptor_offset(push);
      nir_def *offset = nir_iadd_imm(b, intr->src[0].ssa, base);
      nir_def *value = nir_load_ubo(
         b, intr->def.num_components, intr->def.bit_size,
         nir_imm_int(b, HK_APPLE9_ROOT_BINDING), offset,
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = (nir_intrinsic_align_offset(intr) +
                          hk_root_descriptor_offset(push)) %
                         nir_intrinsic_align_mul(intr),
         .range_base = base, .range = nir_intrinsic_range(intr));
      nir_def_rewrite_uses(&intr->def, value);
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_load_base_workgroup_id) {
      if (ctx->io_only)
         return false;
      nir_def *value = nir_load_ubo(
         b, 3, 32, nir_imm_int(b, HK_APPLE9_ROOT_BINDING),
         nir_imm_int(b, hk_root_descriptor_offset(cs.base_group)),
         .align_mul = 4, .align_offset = 0,
         .range_base = hk_root_descriptor_offset(cs.base_group), .range = 12);
      nir_def_rewrite_uses(&intr->def, value);
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_get_ssbo_size) {
      if (ctx->io_only &&
          !nir_scalar_is_const(nir_scalar_resolved(intr->src[0].ssa, 0))) {
         nir_def *offset = nir_iadd_imm(
            b, nir_imul_imm(b, intr->src[0].ssa, sizeof(uint32_t)),
            HK_APPLE9_BUFFER_SIZES_OFFSET);
         nir_def *value = nir_load_ubo(
            b, 1, 32, nir_imm_int(b, HK_APPLE9_ROOT_BINDING), offset,
            .align_mul = 4, .range_base = HK_APPLE9_BUFFER_SIZES_OFFSET,
            .range = 64 * sizeof(uint32_t));
         nir_def_replace(&intr->def, value);
         return true;
      }
      uint32_t slot;
      if (!hk_apple9_resolve_binding(ctx, intr->src[0].ssa,
                                     HK_APPLE9_MAP_SSBO, &slot))
         return false;
      unsigned offset = HK_APPLE9_BUFFER_SIZES_OFFSET + slot * sizeof(uint32_t);
      nir_def *value = nir_load_ubo(
         b, 1, 32, nir_imm_int(b, HK_APPLE9_ROOT_BINDING),
         nir_imm_int(b, offset), .align_mul = 4, .align_offset = 0,
         .range_base = offset, .range = sizeof(uint32_t));
      nir_def_rewrite_uses(&intr->def, value);
      return true;
   }

   enum hk_apple9_map_kind kind;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
      kind = HK_APPLE9_MAP_UBO;
      break;
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      kind = HK_APPLE9_MAP_SSBO;
      break;
   case nir_intrinsic_load_vulkan_descriptor: {
      if (ctx->io_only)
         return hk_apple9_fail(ctx, "Apple9 could not resolve an SSBO descriptor");
      nir_descriptor_type type = nir_intrinsic_desc_type(intr);
      enum hk_apple9_map_kind kind;
      if (type == nir_descriptor_type_storage_buffer)
         kind = HK_APPLE9_MAP_SSBO;
      else if (type == nir_descriptor_type_uniform_buffer)
         kind = HK_APPLE9_MAP_UBO;
      else
         return hk_apple9_fail(ctx, "Apple9 descriptor type is unsupported");
      uint32_t slot;
      if (!hk_apple9_resolve_binding(ctx, intr->src[0].ssa, kind, &slot))
         return false;
      nir_def *index = nir_imm_intN_t(b, slot, intr->def.bit_size);
      nir_def *replacement;
      if (intr->def.num_components == 1) {
         replacement = index;
      } else if (intr->def.num_components == 2) {
         replacement = nir_vec2(b, index,
                                nir_imm_intN_t(b, 0, intr->def.bit_size));
      } else {
         return hk_apple9_fail(ctx, "Apple9 descriptor vector is unsupported");
      }
      nir_def_rewrite_uses(&intr->def, replacement);
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      /* These should already have been handled above. If we reach here,
       * we're in io_only mode which doesn't handle images. */
      return false;
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_store_block_agx:
      return hk_apple9_fail(ctx, "Apple9 image atomics/sparse/block are not yet supported");
   default:
      return false;
   }

   nir_src *index = nir_get_io_index_src(intr);
   if (!index)
      return hk_apple9_fail(ctx, "Apple9 buffer intrinsic has no resource index");
   if (intr->intrinsic == nir_intrinsic_load_ubo && nir_src_is_const(*index) &&
       nir_src_as_uint(*index) == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING)
      return false;
   if (ctx->io_only && intr->intrinsic == nir_intrinsic_load_ubo &&
       nir_src_is_const(*index) && nir_src_as_uint(*index) >= 64)
      return false;
   if (ctx->io_only && kind == HK_APPLE9_MAP_SSBO &&
       !nir_scalar_is_const(nir_scalar_resolved(index->ssa, 0)) &&
       (intr->intrinsic == nir_intrinsic_load_ssbo ||
        intr->intrinsic == nir_intrinsic_store_ssbo)) {
      nir_def *address_offset = nir_iadd_imm(
         b, nir_imul_imm(b, index->ssa, sizeof(uint64_t)),
         HK_APPLE9_BUFFER_ADDRESSES_OFFSET);
      nir_def *base = nir_load_ubo(
         b, 1, 64, nir_imm_int(b, HK_APPLE9_ROOT_BINDING), address_offset,
          .align_mul = 8, .range_base = HK_APPLE9_BUFFER_ADDRESSES_OFFSET,
          .range = 64 * sizeof(uint64_t));
      bool store = intr->intrinsic == nir_intrinsic_store_ssbo;
      nir_def *offset = intr->src[store ? 2 : 1].ssa;
      nir_def *address = nir_iadd(b, base, nir_u2u64(b, offset));
      if (store) {
         nir_store_global(b, intr->src[0].ssa, address,
            .write_mask = nir_intrinsic_write_mask(intr),
            .align_mul = nir_intrinsic_align_mul(intr),
            .align_offset = nir_intrinsic_align_offset(intr),
            .access = nir_intrinsic_access(intr));
         nir_instr_remove(instr);
      } else {
         nir_def *value = nir_load_global(
            b, intr->def.num_components, intr->def.bit_size, address,
            .align_mul = nir_intrinsic_align_mul(intr),
            .align_offset = nir_intrinsic_align_offset(intr),
            .access = nir_intrinsic_access(intr));
         nir_def_replace(&intr->def, value);
      }
      return true;
   }
   uint32_t slot;
   if (!hk_apple9_resolve_binding(ctx, index->ssa, kind, &slot))
      return false;
   nir_src_rewrite(index, nir_imm_int(b, slot));
   return true;
}

bool
hk_nir_lower_apple9_descriptors(
   nir_shader *nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_apple9_shader_mapping *mapping, const char **reason)
{
   mapping->descriptor_snapshot = true;
   struct hk_apple9_nir_context ctx = {
      .set_layouts = set_layouts,
      .set_layout_count = set_layout_count,
      .mapping = mapping,
      .reason = reason,
   };
   *reason = NULL;
   nir_shader_tex_pass(nir, hk_apple9_lower_tex,
                       nir_metadata_control_flow, &ctx);
   nir_shader_instructions_pass(nir, hk_apple9_lower_intrinsic,
                                nir_metadata_none, &ctx);
   if (!*reason)
      nir->info.num_images = 0;
   return !*reason;
}

bool
hk_nir_lower_apple9_textures(
   nir_shader *nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_apple9_shader_mapping *mapping, const char **reason)
{
   struct hk_apple9_nir_context ctx = {
      .set_layouts = set_layouts,
      .set_layout_count = set_layout_count,
      .mapping = mapping,
      .reason = reason,
      .textures_only = true,
   };
   mapping->descriptor_snapshot = false;
   *reason = NULL;
   nir_shader_tex_pass(nir, hk_apple9_lower_tex,
                       nir_metadata_control_flow, &ctx);
   nir_shader_instructions_pass(nir, hk_apple9_lower_intrinsic,
                                nir_metadata_none, &ctx);
   if (!*reason) {
      /* Image ops were lowered to global memory; clear the count so
       * downstream checks (e.g. hk_shader.c graphics rejection) don't
       * trigger on the stale num_images from before lowering. */
      nir->info.num_images = 0;
   }
   return !*reason;
}

bool
hk_nir_lower_apple9_io(nir_shader *nir, uint32_t set_layout_count,
                       struct vk_descriptor_set_layout *const *set_layouts,
                       struct hk_apple9_shader_mapping *mapping,
                       const char **reason)
{
   struct hk_apple9_nir_context ctx = {
      .set_layouts = set_layouts,
      .set_layout_count = set_layout_count,
      .mapping = mapping,
      .reason = reason,
      .io_only = true,
   };
   *reason = NULL;
   nir_shader_instructions_pass(nir, hk_apple9_lower_intrinsic,
                                nir_metadata_control_flow, &ctx);
   return !*reason;
}

static const void *
hk_apple9_descriptor_cpu(struct hk_descriptor_state *desc, uint32_t set,
                         uint32_t offset, uint32_t size)
{
   if (set >= HK_MAX_SETS || offset > desc->set_sizes[set] ||
       size > desc->set_sizes[set] - offset)
      return NULL;
   if (desc->sets[set])
      return (const uint8_t *)desc->sets[set]->mapped_ptr + offset;
   if (desc->push[set])
      return desc->push[set]->data + offset;
   return NULL;
}

static bool
hk_apple9_resolve_buffer(struct hk_descriptor_state *desc,
                         struct hk_apple9_shader_mapping *mapping,
                         enum agx_apple9_compute_resource_kind kind,
                         uint32_t binding, uint64_t *address, uint32_t *size)
{
   struct hk_apple9_descriptor *entry = kind == AGX_APPLE9_COMPUTE_RESOURCE_UBO
      ? &mapping->ubo[binding] : &mapping->ssbo[binding];
   if (entry->set >= HK_MAX_SETS)
      A9FAIL();

   if (entry->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
      if (!desc->root.sets[entry->set] ||
          entry->offset > desc->set_sizes[entry->set])
         A9FAIL();
      *address = desc->root.sets[entry->set] + entry->offset;
      if (size)
         *size = desc->set_sizes[entry->set] - entry->offset;
      return true;
   }

   struct hk_buffer_address buffer;
   if (entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
       entry->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
      uint32_t index = desc->root.set_dynamic_buffer_start[entry->set] +
                       entry->dynamic_index + entry->element;
      if (index >= HK_MAX_DYNAMIC_BUFFERS)
         A9FAIL();
      buffer = desc->root.dynamic_buffers[index];
   } else {
      const void *record = hk_apple9_descriptor_cpu(
         desc, entry->set, entry->offset, sizeof(buffer));
      if (!record)
         A9FAIL();
      memcpy(&buffer, record, sizeof(buffer));
   }
   if (!buffer.base_addr)
      A9FAIL();
   *address = buffer.base_addr;
   if (size)
      *size = buffer.size;
   return true;
}

bool
hk_apple9_decode_texture(const struct agx_texture_packed *input,
                         struct AGX_TEXTURE *decoded)
{
   if (!AGX_TEXTURE_unpack(NULL, (const uint8_t *)input, decoded))
      A9FAIL();
   if (decoded->layout == AGX_LAYOUT_LINEAR) {
      decoded->depth = decoded->depth_linear;
      /* Extended at bit 127 aliases byte-stride bit 21 in the old packet. */
      decoded->stride &= BITFIELD_MASK(21);
   }
   return !decoded->compressed && decoded->mode != AGX_IMAGE_MODE_SPARSE &&
          decoded->width && decoded->height && decoded->depth &&
          decoded->last_level >= decoded->first_level;
}

static bool
hk_apple9_pack_texture(void *output,
                       const struct hk_sampled_image_descriptor *input)
{
   struct AGX_TEXTURE tex;
   if (!hk_apple9_decode_texture(&input->tex, &tex))
      A9FAIL();

   uint8_t *out = output;
   memset(out, 0, 32);
   memcpy(out, &input->tex, 8);
   uint32_t dimensions;
   memcpy(&dimensions, out + 4, sizeof(dimensions));
   dimensions &= 0x00ffffff;
   if (tex.samples == AGX_SAMPLE_COUNT_4)
      dimensions |= 1u << 24;
   if (tex.unk_mipmapped)
      dimensions |= 1u << 26;
   memcpy(out + 4, &dimensions, sizeof(dimensions));

   uint64_t address = tex.address >> 4;
   if (tex.layout == AGX_LAYOUT_LINEAR) {
      uint32_t stride = tex.stride + 16;
      if (!stride || (stride & 15))
         A9FAIL();
      address |= (uint64_t)((stride >> 4) - 1) << 46;
   } else {
      switch (tex.dimension) {
      case AGX_TEXTURE_DIMENSION_1D_ARRAY:
      case AGX_TEXTURE_DIMENSION_2D_ARRAY:
      case AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED:
      case AGX_TEXTURE_DIMENSION_3D:
      case AGX_TEXTURE_DIMENSION_CUBE:
      case AGX_TEXTURE_DIMENSION_CUBE_ARRAY:
         address |= (uint64_t)(tex.depth - 1) << 46;
         break;
      default:
         break;
      }
      if (tex.page_aligned_layers)
         address |= UINT64_C(1) << 62;
   }
   if (tex.unk_mipmapped)
      address |= UINT64_C(1) << 63;
   if (tex.srgb)
      address |= UINT64_C(1) << 44;
   memcpy(out + 8, &address, sizeof(address));
   out[21] = tex.first_level << 4;
   out[22] = tex.last_level;
   return true;
}

static uint64_t
hk_apple9_upload_sysvals(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                        struct hk_descriptor_state *descriptors)
{
   struct agx_ptr table = hk_pool_alloc(
      cmd, AGX_APPLE9_GRAPHICS_SYSVAL_SIZE, 16);
   if (!table.cpu)
      return 0;
   memset(table.cpu, 0, AGX_APPLE9_GRAPHICS_SYSVAL_SIZE);
   /* Only vertex fetch reads the draw parameters; fragment blend lowering
    * reads the blend constant from the same words.
    */
   if (shader->info.stage == MESA_SHADER_FRAGMENT)
      memcpy(table.cpu, cmd->vk.dynamic_graphics_state.cb.blend_constants,
             sizeof(cmd->vk.dynamic_graphics_state.cb.blend_constants));
   else
      memcpy(table.cpu, cmd->state.gfx.apple9_draw_params,
             sizeof(cmd->state.gfx.apple9_draw_params));

   /* Size and level queries read this table without a hardware texture
    * slot, so a stage may query a texture it never samples (WebRender takes
    * textureSize in the vertex stage and samples in the fragment stage).
    * Fill every mapped texture; only sampled ones must decode.
    */
   uint32_t sampled = shader->b.info.apple9_texture_mask;
   u_foreach_bit(binding,
                 sampled | BITFIELD_MASK(shader->apple9_mapping.texture_count)) {
      struct hk_apple9_descriptor *entry = &shader->apple9_mapping.textures[binding];
      const struct hk_sampled_image_descriptor *desc = hk_apple9_descriptor_cpu(
         descriptors, entry->set, entry->offset, sizeof(*desc));
      struct AGX_TEXTURE tex;
      if (!desc || !hk_apple9_decode_texture(&desc->tex, &tex)) {
         if (sampled & BITFIELD_BIT(binding))
            return 0;
         continue;
      }

      uint32_t *info = (uint32_t *)((uint8_t *)table.cpu +
         AGX_APPLE9_TEXTURE_INFO_OFFSET + binding * AGX_APPLE9_TEXTURE_INFO_STRIDE);
      if (entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
         /* Buffer-texture lowering bounds-checks the logical element count.
          * RGB32 emulation and unaligned view tails remain rejected below.
          */
         info[0] = tex.buffer_size_sw;
         info[1] = info[2] = info[3] = info[4] = 1;
         continue;
      }
      info[0] = u_minify(tex.width, tex.first_level);
      info[1] = u_minify(tex.height, tex.first_level);
      info[2] = tex.dimension == AGX_TEXTURE_DIMENSION_3D ? tex.depth : 1;
      if (tex.dimension == AGX_TEXTURE_DIMENSION_1D_ARRAY ||
          tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
          tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED ||
          tex.dimension == AGX_TEXTURE_DIMENSION_CUBE_ARRAY)
         info[2] = tex.depth;
      info[3] = tex.last_level - tex.first_level + 1;
      info[4] = tex.samples == AGX_SAMPLE_COUNT_4 ? 4 :
         (tex.dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
          tex.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED ? 2 : 1);
   }
   u_foreach_bit(binding, shader->b.info.apple9_sampler_mask) {
      struct hk_apple9_descriptor *entry = &shader->apple9_mapping.samplers[binding];
      const struct hk_sampled_image_descriptor *desc = hk_apple9_descriptor_cpu(
         descriptors, entry->set, entry->offset, sizeof(*desc));
      if (!desc)
         return 0;
      float bias = _mesa_half_to_float(desc->lod_bias_fp16);
      memcpy((uint8_t *)table.cpu + AGX_APPLE9_SAMPLER_BIAS_OFFSET + binding * 4,
             &bias, sizeof(bias));
   }
   return table.gpu;
}

static uint64_t
hk_apple9_upload_root(struct hk_cmd_buffer *cmd,
                     struct hk_descriptor_state *desc, const uint32_t sizes[64],
                     const uint64_t buffer_addresses[64])
{
   bool snapshot = sizes != NULL;
   const uint32_t bytes = snapshot
      ? HK_APPLE9_BUFFER_ADDRESSES_OFFSET + 64 * sizeof(uint64_t)
      : sizeof(desc->root);
   struct agx_ptr table = hk_pool_alloc(cmd, bytes, 16);
   if (!table.cpu)
      return 0;
   desc->root.root_desc_addr = table.gpu;
   memcpy(table.cpu, &desc->root, sizeof(desc->root));
   if (snapshot) {
      memcpy((uint8_t *)table.cpu + HK_APPLE9_BUFFER_SIZES_OFFSET,
             sizes, 64 * sizeof(uint32_t));
      memcpy((uint8_t *)table.cpu + HK_APPLE9_BUFFER_ADDRESSES_OFFSET,
             buffer_addresses, 64 * sizeof(uint64_t));
   }
   return table.gpu;
}

static bool
hk_apple9_upload_textures(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                          struct hk_descriptor_state *descriptors,
                          uint64_t *textures, uint64_t *samplers)
{
   const struct agx_shader_info *info = &shader->b.info;
   *textures = *samplers = 0;
   /* apple9_image_mask tracks tile-level PBE ops (image_store_block_agx),
    * not the image_deref_* operations we lower to global memory. User compute
    * shaders should not have any PBE tile ops after our NIR lowering. */

   for (unsigned i = 0; i < shader->apple9_mapping.storage_image_count; ++i) {
      const struct hk_apple9_descriptor *entry =
         &shader->apple9_mapping.storage_images[i];
      const struct hk_storage_image_descriptor *desc = hk_apple9_descriptor_cpu(
         descriptors, entry->set, entry->offset, sizeof(*desc));
      struct AGX_TEXTURE tex;
      struct AGX_PBE pbe;
      if (!desc || !AGX_TEXTURE_unpack(NULL, (const uint8_t *)&desc->tex, &tex))
         A9FAIL();
      if (tex.address == AGX_ZERO_PAGE_ADDRESS)
         continue;
      if (!AGX_PBE_unpack(NULL, (const uint8_t *)&desc->pbe, &pbe) ||
          tex.compressed || pbe.compressed ||
          /* Internal copies use array views even for one linear layer.
           * Global addressing needs no layer stride when the only layer is 0.
           */
          (tex.extended &&
           (tex.layout != AGX_LAYOUT_LINEAR || tex.depth_linear != 1)) ||
          (pbe.extended &&
           (pbe.layout != AGX_LAYOUT_LINEAR || pbe.depth_linear != 1)) ||
          tex.mode == AGX_IMAGE_MODE_SPARSE || pbe.mode == AGX_IMAGE_MODE_SPARSE ||
          (pbe.layout != AGX_LAYOUT_LINEAR && pbe.layout != AGX_LAYOUT_GPU))
         A9FAIL();
   }

   unsigned texture_count = util_bitcount(info->apple9_texture_mask);
   unsigned sampler_count = util_bitcount(info->apple9_sampler_mask) +
                            info->apple9_uses_texel_fetch;
   if (texture_count > AGX_APPLE9_GRAPHICS_MAX_TEXTURES ||
       sampler_count > AGX_APPLE9_GRAPHICS_MAX_SAMPLERS)
      A9FAIL();

   if (texture_count) {
      struct agx_ptr table = hk_pool_alloc(
         cmd, texture_count * 32, 64);
      if (!table.cpu)
         A9FAIL();
      unsigned slot = 0;
      u_foreach_bit(binding, info->apple9_texture_mask) {
         struct hk_apple9_descriptor *entry =
            &shader->apple9_mapping.textures[binding];
         const struct hk_sampled_image_descriptor *desc =
            hk_apple9_descriptor_cpu(descriptors, entry->set,
                                     entry->offset, sizeof(*desc));
         if (!desc || !hk_apple9_pack_texture(
                         (uint8_t *)table.cpu + slot * 32, desc))
            A9FAIL();
         ++slot;
      }
      *textures = table.gpu;
   }

   if (sampler_count) {
      struct agx_ptr table = hk_pool_alloc(cmd, sampler_count * 32, 64);
      if (!table.cpu)
         A9FAIL();
      memset(table.cpu, 0, sampler_count * 32);
      unsigned slot = 0;
      u_foreach_bit(binding, info->apple9_sampler_mask) {
         struct hk_apple9_descriptor *entry = &shader->apple9_mapping.samplers[binding];
         const struct hk_sampled_image_descriptor *desc = hk_apple9_descriptor_cpu(
            descriptors, entry->set, entry->offset, sizeof(*desc));
         struct AGX_SAMPLER sampler;
         if (!desc || !AGX_SAMPLER_unpack(NULL, (const uint8_t *)&desc->sampler, &sampler))
            A9FAIL();
         if (sampler.minify > AGX_FILTER_LINEAR || sampler.magnify > AGX_FILTER_LINEAR ||
             sampler.compare_enable || sampler.pixel_coordinates ||
             sampler.seamful_cube_maps || desc->clamp_0_sampler_index_or_negative >= 0)
            A9FAIL();
         /* Map AGX wrap modes to pipe wrap modes for the native packer */
         static const unsigned agx_to_pipe_wrap[] = {
            [AGX_WRAP_CLAMP_TO_EDGE]   = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
            [AGX_WRAP_REPEAT]          = PIPE_TEX_WRAP_REPEAT,
            [AGX_WRAP_MIRRORED_REPEAT] = PIPE_TEX_WRAP_MIRROR_REPEAT,
            [AGX_WRAP_CLAMP_TO_BORDER] = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
            [AGX_WRAP_CLAMP_GL]        = PIPE_TEX_WRAP_CLAMP,
         };
         unsigned pipe_wrap_s = sampler.wrap_s < ARRAY_SIZE(agx_to_pipe_wrap) ?
            agx_to_pipe_wrap[sampler.wrap_s] : PIPE_TEX_WRAP_CLAMP_TO_EDGE;
         unsigned pipe_wrap_t = sampler.wrap_t < ARRAY_SIZE(agx_to_pipe_wrap) ?
            agx_to_pipe_wrap[sampler.wrap_t] : PIPE_TEX_WRAP_CLAMP_TO_EDGE;
         agx_apple9_pack_sampler((uint8_t *)table.cpu + slot++ * 32,
            sampler.minify == AGX_FILTER_LINEAR, sampler.magnify == AGX_FILTER_LINEAR,
            sampler.mip_filter, sampler.minimum_lod, sampler.maximum_lod,
            pipe_wrap_s, pipe_wrap_t,
            sampler.maximum_anisotropy);
      }
      if (info->apple9_uses_texel_fetch)
         agx_apple9_pack_nearest_sampler((uint8_t *)table.cpu + slot * 32);
      *samplers = table.gpu;
   }
   return true;
}

bool
hk_apple9_graphics_resources(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                             unsigned stage, struct hk_apple9_uniform_draw *draw)
{
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   const struct agx_shader_info *info = &shader->b.info;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   uint32_t sizes[64] = {0};
   uint64_t buffer_addresses[64] = {0};
   bool snapshot = shader->apple9_mapping.descriptor_snapshot;
   for (unsigned i = 0; snapshot && i < shader->apple9_mapping.ssbo_count; ++i) {
      if (!hk_apple9_resolve_buffer(desc, &shader->apple9_mapping,
                                    AGX_APPLE9_COMPUTE_RESOURCE_SSBO, i,
                                    &buffer_addresses[i], &sizes[i]))
         A9FAIL();
   }

   /* Retain the narrow sampled-image contract. Uniform texel buffers use
    * the linear 2D representation produced by hk_CreateBufferView.
    */
   u_foreach_bit(binding, info->apple9_texture_mask) {
      const struct hk_apple9_descriptor *entry =
         &shader->apple9_mapping.textures[binding];
      const struct hk_sampled_image_descriptor *image = hk_apple9_descriptor_cpu(
         desc, entry->set, entry->offset, sizeof(*image));
      struct AGX_TEXTURE tex;
      if (!image || !hk_apple9_decode_texture(&image->tex, &tex))
         A9FAIL();
      if (entry->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
         if (tex.layout != AGX_LAYOUT_LINEAR ||
             tex.dimension != AGX_TEXTURE_DIMENSION_2D ||
             tex.channels == AGX_CHANNELS_R32G32B32_EMULATED ||
             tex.buffer_offset_sw)
            A9FAIL();
      } else if ((tex.dimension != AGX_TEXTURE_DIMENSION_2D &&
                  tex.dimension != AGX_TEXTURE_DIMENSION_2D_ARRAY) ||
                 tex.first_level || tex.last_level || tex.depth != 1 ||
                 tex.samples != 0) {
         A9FAIL();
      }
   }

   uint64_t root = 0, sysvals = 0;
   draw->buffer_count[stage] = info->apple9_resource_count;
   for (unsigned i = 0; i < info->apple9_resource_count; ++i) {
      unsigned binding = info->apple9_resource_binding[i];
      bool ssbo = info->apple9_resource_ssbo_mask & BITFIELD_BIT(i);
      uint64_t *address = &draw->buffers[stage][i];
      if (!ssbo && binding == HK_APPLE9_ROOT_BINDING) {
         if (!root)
            root = hk_apple9_upload_root(cmd, desc, snapshot ? sizes : NULL,
                                         snapshot ? buffer_addresses : NULL);
         *address = root;
      } else if (!ssbo && binding == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING) {
         if (!sysvals)
            sysvals = hk_apple9_upload_sysvals(cmd, shader, desc);
         *address = sysvals;
      } else if (!ssbo && binding == AGX_APPLE9_VERTEX_RANGES_BINDING) {
         if (stage || !shader->apple9_mapping.vertex.robust)
            A9FAIL();
         struct agx_apple9_vertex_range ranges[32] = {0};
         for (unsigned j = 0; j < ARRAY_SIZE(cmd->state.gfx.vb); ++j) {
            ranges[j].base = cmd->state.gfx.vb[j].addr;
            ranges[j].size = cmd->state.gfx.vb[j].range;
         }
         *address = hk_pool_upload(cmd, ranges, sizeof(ranges), 64);
      } else if (!ssbo && binding >= 32 && binding < 64) {
         const struct hk_addr_range *vb = &cmd->state.gfx.vb[binding - 32];
         if (stage || !vb->range || !vb->addr)
            A9FAIL();
         *address = vb->addr;
      } else {
         if (!snapshot)
            A9FAIL();
         unsigned count = ssbo ? shader->apple9_mapping.ssbo_count
                               : shader->apple9_mapping.ubo_count;
         if (binding >= count ||
             !hk_apple9_resolve_buffer(
                desc, &shader->apple9_mapping,
                ssbo ? AGX_APPLE9_COMPUTE_RESOURCE_SSBO
                     : AGX_APPLE9_COMPUTE_RESOURCE_UBO,
                binding, address, NULL))
            A9FAIL();
      }
      if (!*address)
         A9FAIL();
   }
   uint64_t table = info->apple9_resource_count
      ? hk_pool_upload(cmd, draw->buffers[stage],
                       info->apple9_resource_count * sizeof(uint64_t), 64) : 0;
   if (info->apple9_resource_count && !table)
      A9FAIL();
   if (stage)
      draw->fragment_table = table;
   else
      draw->vertex_table = table;
   return hk_apple9_upload_textures(cmd, shader, desc,
                                    &draw->texture_table[stage],
                                    &draw->sampler_table[stage]);
}

static bool
hk_apple9_launch_compute(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                         struct agx_grid grid, struct agx_workgroup wg,
                         struct agx_bo *body,
                         const struct agx_apple9_compute_profile *profile,
                         const uint64_t *addresses, unsigned count,
                         uint64_t textures, uint64_t samplers,
                         uint64_t group_counts)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_apple9_compute_geometry geometry = {
      .local = {wg.x, wg.y, wg.z},
   };
   bool indirect = grid.mode != AGX_CDM_MODE_DIRECT;
   if (grid.mode != AGX_CDM_MODE_DIRECT &&
       grid.mode != AGX_CDM_MODE_INDIRECT_GLOBAL)
      A9FAIL();
   geometry.mode = indirect ? AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT
                            : AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT;
   if (indirect)
      geometry.group_counts = group_counts ?: grid.ptr;
   else
      memcpy(geometry.threads, grid.count, sizeof(geometry.threads));

   uint64_t launch;
   if (!agx_apple9_prepare_compute_dispatch(
          &dev->dev, &cmd->apple9_compute, body, profile, addresses, count,
          textures, samplers, &geometry,
          profile->preamble_size
             ? body->va->addr + profile->preamble_offset : 0,
          &launch))
      A9FAIL();

   unsigned size = indirect ? AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE
                            : AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE;
   hk_ensure_cs_has_space(cmd, cs, size + 4);
   bool ok = indirect
      ? agx_apple9_emit_indirect_dispatch(cs->current, launch, grid.ptr,
                                         geometry.local, false, profile)
      : agx_apple9_emit_direct_dispatch(cs->current, launch,
                                        geometry.threads, geometry.local, profile);
   if (!ok)
      A9FAIL();
   cs->current = (uint8_t *)cs->current + size;
   cs->stats.cmds++;
   return true;
}

bool
hk_apple9_dispatch_precompiled(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                               struct agx_grid grid,
                               struct agx_precompiled_shader *prog,
                               const void *data, size_t data_size)
{
   uint64_t uploaded = data_size
      ? hk_pool_upload(cmd, data, data_size, 8) : 0;
   const struct agx_apple9_compute_profile *profile = &prog->apple9_profile;
   uint64_t addresses[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
   for (unsigned i = 0; i < profile->resource_binding_count; ++i) {
      if (profile->resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_SHARED) {
         addresses[i] = AGX_APPLE9_COMPUTE_SHARED_ROOT;
      } else {
         if (profile->resource_kind[i] != AGX_APPLE9_COMPUTE_RESOURCE_UBO ||
             profile->resource_binding[i] != 0 || !uploaded)
            A9FAIL();
         addresses[i] = uploaded;
      }
   }
   return hk_apple9_launch_compute(
      cmd, cs, grid, prog->b.workgroup, prog->bo, profile, addresses,
      profile->resource_binding_count, 0, 0, 0);
}

bool
hk_apple9_dispatch_shader(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                          struct hk_shader *shader, struct agx_grid grid,
                          struct agx_workgroup local_size)
{
   const struct agx_apple9_compute_profile *profile = &shader->apple9_profile;
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   uint32_t sizes[64] = {0};
   uint64_t buffer_addresses[64] = {0};
   bool snapshot = shader->apple9_mapping.descriptor_snapshot;
   for (unsigned i = 0; snapshot && i < shader->apple9_mapping.ssbo_count; ++i) {
      if (!hk_apple9_resolve_buffer(desc, &shader->apple9_mapping,
                                    AGX_APPLE9_COMPUTE_RESOURCE_SSBO, i,
                                    &buffer_addresses[i], &sizes[i]))
         A9FAIL();
   }

   uint64_t root = 0, sysvals = 0, constant = 0;
   uint64_t addresses[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
   for (unsigned i = 0; i < profile->resource_binding_count; ++i) {
      if (profile->resource_kind[i] == AGX_APPLE9_COMPUTE_RESOURCE_SHARED) {
         addresses[i] = AGX_APPLE9_COMPUTE_SHARED_ROOT;
         continue;
      }

      uint32_t binding = profile->resource_binding[i];
      if (binding == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING) {
         if (!sysvals)
            sysvals = hk_apple9_upload_sysvals(cmd, shader, desc);
         addresses[i] = sysvals;
      } else if (binding == HK_APPLE9_ROOT_BINDING) {
         if (!root)
            root = hk_apple9_upload_root(cmd, desc, snapshot ? sizes : NULL,
                                         snapshot ? buffer_addresses : NULL);
         addresses[i] = root;
      } else if (binding == HK_APPLE9_DATA_BINDING) {
         if (!shader->data_size)
            A9FAIL();
         if (!constant)
            constant = hk_pool_upload(cmd, shader->data_ptr,
                                      shader->data_size, 16);
         addresses[i] = constant;
      } else {
         if (!snapshot || binding >= 64 ||
             !hk_apple9_resolve_buffer(desc, &shader->apple9_mapping,
                                       profile->resource_kind[i], binding,
                                       &addresses[i], NULL))
            A9FAIL();
      }
      if (!addresses[i])
         A9FAIL();
   }

   uint64_t textures, samplers;
   if (!hk_apple9_upload_textures(cmd, shader, desc, &textures, &samplers))
      A9FAIL();
   return hk_apple9_launch_compute(
      cmd, cs, grid, local_size, shader->bo, profile, addresses,
      profile->resource_binding_count, textures, samplers,
      desc->root.cs.group_count_addr);
}
