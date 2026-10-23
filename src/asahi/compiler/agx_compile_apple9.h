/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_COMPILE_APPLE9_H
#define AGX_COMPILE_APPLE9_H

#include <stdbool.h>
#include <stdint.h>

#include "agx_compile.h"
#include "util/blend.h"
#include "util/format/u_formats.h"
#include "util/format/u_format.h"
#include "agx_apple9_profile.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Whether tex's texel offset has a native immediate encoding: a constant
 * offset in [-8, 7] on a 2D (non-array) or 3D sample, gather or comparison.
 * If so, *bits receives x | y << 4 | z << 8 in signed 4-bit fields. */
struct nir_tex_instr;
bool agx_apple9_texture_native_offset(const struct nir_tex_instr *tex,
                                      uint16_t *bits);

/* The sample mode word (instruction bytes 6-7) for an ordinary sample form,
 * and whether that form has a measured offset variant (returned in
 * *offset_mode). Shared by the packer and the backend's offset fallback. */
uint16_t agx_apple9_texture_mode(bool explicit_lod, bool gradient, bool bias,
                                 unsigned dimension, bool shadow, bool min_lod);
bool agx_apple9_texture_offset_mode(uint16_t mode, uint16_t *offset_mode);

/* Draw-time compiles: allow one helper thread for independent RA candidates. */
void agx_apple9_set_parallel_ra(bool enable);

bool agx_nir_hoist_apple9_uniform_reads(nir_shader *shader);
bool agx_nir_lower_apple9_math(nir_shader *shader);
bool agx_nir_opt_apple9_bounds(nir_shader *shader);
bool agx_nir_lower_apple9_compute_derivatives(nir_shader *shader);
bool agx_nir_lower_apple9_denorms(nir_shader *shader);
bool agx_nir_lower_apple9_denorm_preserve(nir_shader *shader);
bool agx_nir_lower_apple9_subgroup_denorms(nir_shader *shader);

/* Schedule texture operations without increasing estimated scalar-word pressure. */
bool agx_nir_hoist_apple9_samples(nir_shader *shader);
/* While set on this thread (compiling a spill probe), register allocation
 * keeps its first candidate. */
void agx_apple9_ra_set_probe(bool probe);
/* Free the first allocations kept from the last probe. */
void agx_apple9_ra_release_probe_results(void);
bool agx_apple9_ra_is_probe(void);
bool agx_nir_hoist_apple9_derivative_sources(nir_shader *shader);

/* After generic texture lowering, backend1 holds a packed Apple9 LOD/bias
 * word rather than the original floating-point LOD or integer fetch level. */
bool agx_nir_lower_apple9_texture_lod(nir_shader *shader);

/* agx_compile_apple9_tiny's reason when a compute program has no observable
 * effect; the caller may run nothing for it instead of failing. */
extern const char agx_apple9_compute_without_effects[];

/*
 * Compile compute NIR through Apple9's semantic instruction selection and
 * register allocator. Arithmetic, memory operations, and structured control
 * flow share the same lowering used by graphics shaders. Unsupported NIR or
 * resource requirements return a diagnostic; no shader replay is substituted.
 *
 * On failure, *reason points at a static diagnostic string when reason is
 * non-NULL. On success, out owns a malloc-backed main/preamble binary and may
 * be released with free(out->binary), like agx_compile_shader_nir output.
 */
bool agx_compile_apple9_tiny(nir_shader *nir, struct agx_shader_part *out,
                             struct agx_apple9_compute_profile *profile,
                             const char **reason);

/*
 * Compile graphics NIR through the common semantic VIR and register allocator.
 * Vertex attributes and fragment varyings are packed by semantic location and
 * component; fragment variants use the producer's layout. Smooth inputs use
 * coefficient-aware perspective multiplication. Flat and noperspective inputs
 * have separate coefficient modes. Centroid and sample inputs evaluate at
 * an allocated position selected from coverage or the requested sample index.
 * The common sample loop lowers per-sample shading and output masks.
 *
 * Buffer resources use a shader-loaded address table. Texture and sampler
 * bindings are compacted independently. Cube projection and integer fetches
 * select native operations; queries and explicit gradients use ordinary NIR
 * lowering.
 * Structured branches and loops use the common execution-mask model. Fragment
 * outputs include format-aware color stores, blending, and depth export.
 *
 * The caller supplies hardware clip coordinates and compatible stage and
 * render-target state. The single-target entry point below retains RGBA8
 * defaults; the MRT entry point takes explicit attachment formats and state.
 */
bool agx_compile_apple9_fragment(nir_shader *nir,
                                 struct agx_shader_part *out,
                                 const char **reason);

/* Internal per-draw system values share the ordinary buffer-address table.
 * API UBOs use 0..31; vertex bindings use 32..63. */
#define AGX_APPLE9_GRAPHICS_SYSVAL_BINDING 64
#define AGX_APPLE9_SAMPLER_BIAS_OFFSET 16
#define AGX_APPLE9_SAMPLER_BIAS_COUNT 32
#define AGX_APPLE9_TEXTURE_INFO_OFFSET (16 + AGX_APPLE9_SAMPLER_BIAS_COUNT * sizeof(float))
#define AGX_APPLE9_TEXTURE_INFO_STRIDE 32
#define AGX_APPLE9_POINT_SIZE_OFFSET (AGX_APPLE9_TEXTURE_INFO_OFFSET + 32 * AGX_APPLE9_TEXTURE_INFO_STRIDE)
#define AGX_APPLE9_FIRST_FAN_OFFSET (AGX_APPLE9_POINT_SIZE_OFFSET + 4)
#define AGX_APPLE9_POLYGON_STIPPLE_OFFSET (AGX_APPLE9_POINT_SIZE_OFFSET + 16)
/* Line stipple: the packed pattern/factor word, then the viewport scale and
 * offset of viewport 0 as vec3s (window = ndc * scale + offset). */
#define AGX_APPLE9_LINE_STIPPLE_OFFSET (AGX_APPLE9_POLYGON_STIPPLE_OFFSET + 32 * 4)
#define AGX_APPLE9_VIEWPORT_SCALE_OFFSET (AGX_APPLE9_LINE_STIPPLE_OFFSET + 16)
#define AGX_APPLE9_VIEWPORT_OFFSET_OFFSET (AGX_APPLE9_LINE_STIPPLE_OFFSET + 32)
#define AGX_APPLE9_GRAPHICS_SYSVAL_SIZE (AGX_APPLE9_LINE_STIPPLE_OFFSET + 48)

/* Optional per-dispatch sampler state for APIs with mutable samplers. */
#define AGX_APPLE9_RUNTIME_SAMPLER_OFFSET AGX_APPLE9_GRAPHICS_SYSVAL_SIZE
#define AGX_APPLE9_RUNTIME_SAMPLER_STRIDE 64

/* Honeykrisp G16 image descriptors live in the fixed USC aperture. A
 * texture or sampler handle is the offset from the aperture base of the
 * descriptor's image or sampler part. The image part starts with the same
 * texture info record as the per-draw sysvals, the sampler part with the
 * runtime sampler record; each holds its native table entry at the slot
 * below.
 */
#define AGX_APPLE9_USC_BASE (1ull << 40)
#define AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY 2
#define AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY 3
#define AGX_APPLE9_DESCRIPTOR_TEXTURE_INFO 0
#define AGX_APPLE9_DESCRIPTOR_RUNTIME_SAMPLER 0
/* Native table index of the private texel-fetch sampler, for stages whose
 * samplers are register indexed. */
#define AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET (AGX_APPLE9_RUNTIME_SAMPLER_OFFSET + 32 * AGX_APPLE9_RUNTIME_SAMPLER_STRIDE)
/* Per-draw color attachment write enables, one bit per attachment. */
#define AGX_APPLE9_COLOR_WRITE_ENABLE_OFFSET (AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET + 4)
/* Packed sample positions, in the PPP multisample control format. */
#define AGX_APPLE9_SAMPLE_POSITIONS_OFFSET (AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET + 8)
/* Per-viewport window transform for vertex-stage line stipple, one vec4
 * (scale.xy, offset.xy) per viewport: window = ndc * scale + offset. */
#define AGX_APPLE9_VIEWPORT_TABLE_OFFSET (AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET + 16)
#define AGX_APPLE9_VIEWPORT_TABLE_COUNT 16
/* Rasterization state for software polygon points (struct
 * agx_apple9_polygon_point_params). */
#define AGX_APPLE9_POLYGON_POINT_OFFSET \
   (AGX_APPLE9_VIEWPORT_TABLE_OFFSET + AGX_APPLE9_VIEWPORT_TABLE_COUNT * 16)
#define AGX_APPLE9_RUNTIME_SYSVAL_SIZE (AGX_APPLE9_POLYGON_POINT_OFFSET + 32)

enum agx_apple9_polygon_point_flags {
   AGX_APPLE9_PP_CULL_FRONT = 1 << 0,
   AGX_APPLE9_PP_CULL_BACK = 1 << 1,
   AGX_APPLE9_PP_FRONT_CCW = 1 << 2,
   AGX_APPLE9_PP_DEPTH_CLIP = 1 << 3,
   AGX_APPLE9_PP_DEPTH_BIAS = 1 << 4,
   /* The depth bias unit depends on the polygon's depth (float formats). */
   AGX_APPLE9_PP_DEPTH_BIAS_FLOAT = 1 << 5,
};

struct agx_apple9_polygon_point_params {
   uint32_t flags;
   float bias_constant, bias_slope, bias_clamp;
   /* Minimum resolvable depth difference of unorm formats. */
   float bias_unit;
   /* Viewport 0 maxDepth - minDepth. */
   float depth_range;
   uint32_t pad[2];
};
enum agx_apple9_sampler_field {
   AGX_APPLE9_SAMP_WRAP_S, AGX_APPLE9_SAMP_WRAP_T, AGX_APPLE9_SAMP_WRAP_R,
   AGX_APPLE9_SAMP_MIN, AGX_APPLE9_SAMP_MAG, AGX_APPLE9_SAMP_MIP,
   AGX_APPLE9_SAMP_COMPARE, AGX_APPLE9_SAMP_MIN_LOD, AGX_APPLE9_SAMP_MAX_LOD,
   AGX_APPLE9_SAMP_BIAS, AGX_APPLE9_SAMP_BORDER, AGX_APPLE9_SAMP_PIXEL_COORDS,
   AGX_APPLE9_SAMP_MAX_ANISOTROPY,
};

/* RGB32 buffer views use three scalar texels in the physical descriptor. */
static inline bool
agx_apple9_texture_is_rgb32(enum pipe_format format)
{
   return format == PIPE_FORMAT_R32G32B32_FLOAT ||
          format == PIPE_FORMAT_R32G32B32_UINT ||
          format == PIPE_FORMAT_R32G32B32_SINT;
}

/* Standard independent RGB/alpha blending, plus the RGBA write mask. */
struct agx_apple9_blend {
   uint8_t rgb_src, rgb_dst, alpha_src, alpha_dst;
   uint8_t rgb_func, alpha_func, colormask, logicop_func;
   /* NONE retains the RGBA8 layout used by standalone compiler callers. */
   enum pipe_format format;
   uint8_t samples; /* Zero retains single-sample compiler callers. */
   uint8_t disabled_samples;
   uint8_t alpha_to_coverage : 1;
   uint8_t alpha_to_one : 1;
   uint8_t multisample_disabled : 1;
   uint8_t color_attachment_unused : 1;
   /* Vulkan sample-mask writes also apply to single-sample rendering. */
   uint8_t sample_mask_single_sample : 1;
   uint8_t logicop_enable : 1;
   /* Writes are also gated by the attachment's bit in the per-draw color
    * write enable sysval. */
   uint8_t dynamic_enable : 1;
   /* Sample positions are read from the per-draw sysval rather than the
    * standard pattern, for programmable sample locations. */
   uint8_t dynamic_sample_positions : 1;
   uint8_t advanced_mode, advanced_overlap;
   uint8_t src_premultiplied, dst_premultiplied;
   /* Optional per-draw depth range for shader-triggered early tests. */
   bool clamp_early_depth;
   uint8_t early_depth_clamp_ubo;
   /* GL's by-location dual-source workaround, only set on RT0 when SRC1
    * blending is active. This is part of the fragment variant key. */
   bool dual_source_by_location;
   uint32_t early_depth_clamp_offset_B;
};

static inline bool
agx_apple9_blend_uses_src1(const struct agx_apple9_blend *blend)
{
   if (blend->logicop_enable || blend->advanced_mode)
      return false;

   const uint8_t factors[] = {
      blend->rgb_src, blend->rgb_dst, blend->alpha_src, blend->alpha_dst,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(factors); ++i) {
      enum pipe_blendfactor factor =
         util_blendfactor_without_invert((enum pipe_blendfactor)factors[i]);
      if (factor == PIPE_BLENDFACTOR_SRC1_COLOR ||
          factor == PIPE_BLENDFACTOR_SRC1_ALPHA)
         return true;
   }
   return false;
}

static inline unsigned
agx_apple9_color_components(enum pipe_format format)
{
   return format == PIPE_FORMAT_NONE ? 4 :
      util_format_description(format)->nr_channels;
}

static inline bool
agx_apple9_color_is_half(enum pipe_format format)
{
   return format == PIPE_FORMAT_R16_FLOAT ||
          format == PIPE_FORMAT_R16G16_FLOAT ||
          format == PIPE_FORMAT_R16G16B16A16_FLOAT;
}

/* A render target's tile representation need not match its memory packing.
 * The PBE converts unpacked floating-point components to normalized memory
 * formats. Use FP32 for normalized 16-bit colors to preserve their precision. */
static inline enum pipe_format
agx_apple9_color_tile_format(enum pipe_format format)
{
   switch (format) {
   /* Ordinary tile components are in logical RGBA order. The PBE descriptor
    * performs the memory channel permutation when it exports the tile.
    */
   case PIPE_FORMAT_B8G8R8A8_UNORM:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return PIPE_FORMAT_R8G8B8X8_UNORM;
   case PIPE_FORMAT_B8G8R8A8_SRGB:
      return PIPE_FORMAT_R8G8B8A8_SRGB;
   case PIPE_FORMAT_B8G8R8X8_SRGB:
      return PIPE_FORMAT_R8G8B8X8_SRGB;
   case PIPE_FORMAT_B8G8R8A8_SNORM:
      return PIPE_FORMAT_R8G8B8A8_SNORM;
   case PIPE_FORMAT_B8G8R8A8_UINT:
      return PIPE_FORMAT_R8G8B8A8_UINT;
   case PIPE_FORMAT_B8G8R8A8_SINT:
      return PIPE_FORMAT_R8G8B8A8_SINT;
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R5G5B5A1_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_R4G4B4A4_UNORM:
   case PIPE_FORMAT_B4G4R4A4_UNORM:
   case PIPE_FORMAT_A4B4G4R4_UNORM:
   case PIPE_FORMAT_A4R4G4B4_UNORM:
      return PIPE_FORMAT_R16G16B16A16_FLOAT;
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_B10G10R10A2_UINT:
      return PIPE_FORMAT_R16G16B16A16_UINT;
   /* Padded 10- and 12-bit formats export like 16-bit normalized ones. */
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_X6R10_UNORM:
   case PIPE_FORMAT_X4R12_UNORM:
      return PIPE_FORMAT_R32_FLOAT;
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_X6R10X6G10_UNORM:
   case PIPE_FORMAT_X4R12X4G12_UNORM:
      return PIPE_FORMAT_R32G32_FLOAT;
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_X6R10X6G10X6B10X6A10_UNORM:
      return PIPE_FORMAT_R32G32B32A32_FLOAT;
   default:
      return format;
   }
}

static inline unsigned
agx_apple9_color_words(enum pipe_format format)
{
   if (format == PIPE_FORMAT_NONE)
      return 1;
   return DIV_ROUND_UP(
      util_format_get_blocksize(agx_apple9_color_tile_format(format)), 4);
}

/* Raw tile accesses preserve format bits, including exact integer outputs. */
static inline enum pipe_format
agx_apple9_color_raw_format(unsigned words)
{
   return words == 4 ? PIPE_FORMAT_R32G32B32A32_UINT :
          words == 3 ? PIPE_FORMAT_R32G32B32_UINT :
          words == 2 ? PIPE_FORMAT_R32G32_UINT : PIPE_FORMAT_R32_UINT;
}

static inline bool
agx_apple9_color_is_normalized(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB ||
       (desc->nr_channels != 1 && desc->nr_channels != 2 && desc->nr_channels != 4) ||
       !desc->channel[0].normalized ||
       (desc->channel[0].size != 8 && desc->channel[0].size != 16))
      return false;

   for (unsigned c = 1; c < desc->nr_channels; ++c) {
      if (desc->channel[c].size != desc->channel[0].size ||
          (desc->channel[c].type != UTIL_FORMAT_TYPE_VOID &&
           (!desc->channel[c].normalized ||
            desc->channel[c].type != desc->channel[0].type)))
         return false;
   }

   return true;
}

static inline bool
agx_apple9_color_is_wide(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   return desc->layout == UTIL_FORMAT_LAYOUT_PLAIN &&
          (desc->nr_channels == 1 || desc->nr_channels == 2 || desc->nr_channels == 4) &&
          ((desc->channel[0].pure_integer &&
            (desc->channel[0].size == 8 || desc->channel[0].size == 16 ||
             desc->channel[0].size == 32)) ||
           (desc->channel[0].type == UTIL_FORMAT_TYPE_FLOAT && desc->channel[0].size == 32));
}

static inline bool
agx_apple9_color_is_packed(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R5G5B5A1_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_R4G4B4A4_UNORM:
   case PIPE_FORMAT_B4G4R4A4_UNORM:
   case PIPE_FORMAT_A4B4G4R4_UNORM:
   case PIPE_FORMAT_A4R4G4B4_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_B10G10R10A2_UINT:
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return true;
   default:
      return false;
   }
}

/* The NIR format describes the physical tile representation. Signedness and
 * destination conversion come from the PBE descriptor. Only the packed tile
 * representations listed below can be exported without expansion. */
static inline unsigned
agx_apple9_block_export_format(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   if (format == PIPE_FORMAT_R10G10B10A2_UNORM ||
       format == PIPE_FORMAT_B10G10R10A2_UNORM ||
       format == PIPE_FORMAT_R11G11B10_FLOAT)
      return 7;
   if (!desc->channel[0].pure_integer && desc->channel[0].size == 8)
      return 7;
   switch (desc->channel[0].size) {
   case 8: return 5;
   case 16: return 1;
   case 32: return 3;
   default: return ~0u;
   }
}

enum agx_apple9_sampler_flags {
   AGX_APPLE9_CUSTOM_BORDER = 1 << 0,
   AGX_APPLE9_CLAMP_SHADOW_REFERENCE = 1 << 1,
};

struct agx_apple9_sampler_key {
   uint32_t flags;
   float border[4];
   uint8_t wrap[3];
   uint8_t min_filter, mag_filter, mip_filter, compare_func, seamless_cube_map;
   float min_lod, max_lod, lod_bias;
   unsigned max_anisotropy;
};

void agx_nir_legalize_apple9_texture_sources(nir_shader *nir);
struct nir_builder;
struct nir_tex_instr;
struct nir_def *agx_apple9_texture_info(struct nir_builder *b,
                                        const struct nir_tex_instr *tex,
                                        unsigned field);
struct nir_def *agx_apple9_sampler_field(struct nir_builder *b,
                                         const struct nir_tex_instr *tex,
                                         unsigned field);
bool agx_nir_lower_apple9_runtime_sampler_state(nir_shader *nir, const char **reason);
/* A NULL key reads the runtime sampler metadata; otherwise specialize to key. */

bool agx_nir_lower_apple9_texture_offsets(
   nir_shader *nir, const struct agx_apple9_sampler_key key[32],
   const char **reason);

struct agx_apple9_texture_mapping {
   uint8_t samplers[32];
   uint32_t white_samplers;
   uint32_t seamful_cubes;
   /* Samplers used by texel-offset samples. Offsets the sample instruction
    * cannot encode are lowered with runtime sampler state, so the driver
    * must fill these runtime sampler records of the graphics sysval table. */
   uint32_t runtime_samplers;
};

/* Mask of sampler indices sampled with texel offsets; all samplers when an
 * offset sample has no sampler index yet. */
uint32_t agx_nir_apple9_offset_samplers(const nir_shader *nir);

/* Apple9 backend texture flags. They share nir_tex_instr::backend_flags with
 * the AGX_TEXTURE_FLAG_* bits in agx_nir_texture.h, so keep them disjoint. */
#define AGX_APPLE9_TEXTURE_CUBE_AS_ARRAY (1 << 8)
#define AGX_APPLE9_TEXTURE_LOD_QUERY (1 << 9)
/* Texel fetch whose level, layer and sample are already bounded. */
#define AGX_APPLE9_TEXTURE_FETCH_BOUNDED (1 << 10)
/* Handle sources hold native table indices rather than descriptor offsets. */
#define AGX_APPLE9_TEXTURE_HANDLE_INDEX (1 << 11)
/* The sample's final form has no measured offset encoding: lower its texel
 * offset in NIR instead of encoding it. */
#define AGX_APPLE9_TEXTURE_OFFSET_FALLBACK (1 << 12)
/* Explicit-level sample from the offset lowering, whose level already
 * includes the sampler's LOD bias: the per-draw sampler bias must not be
 * applied again. */
#define AGX_APPLE9_TEXTURE_BIASED_LEVEL (1 << 13)

bool agx_nir_lower_apple9_sampler_state(
   nir_shader *nir, const struct agx_apple9_sampler_key key[32],
   struct agx_apple9_texture_mapping *mapping, const char **reason);

/* Per-target blend array contains MAX2(nr_targets, 1) entries, up to eight.
 * Entry zero also carries multisample state for depth-only rendering. */
bool agx_compile_apple9_fragment_mrt(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, unsigned nr_targets,
   struct agx_shader_part *out, const char **reason);

bool agx_compile_apple9_fragment_blend(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend,
   struct agx_shader_part *out, const char **reason);

/* Specialize fragment coefficient reads to the producing vertex layout. */
bool agx_compile_apple9_fragment_inputs(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   struct agx_shader_part *out, const char **reason);

bool agx_apple9_vertex_format_supported(enum pipe_format format);

/* Vertex pulling uses ordinary typed buffer loads in the API main. Buffer
 * addresses remain draw state; format, stride and attribute offset specialize
 * the fetch. Multiple attributes sharing a binding use one resource argument. */
/* Private bindings for the software transform-feedback vertex job. Public
 * Apple9 vertex shaders do not expose SSBOs; these use the normal resource ABI.
 */
#define AGX_APPLE9_XFB_PARAMS      26
#define AGX_APPLE9_XFB_BUFFER_BASE 27
#define AGX_APPLE9_XFB_INDICES     31

struct agx_apple9_xfb_params {
   uint32_t first_vertex, index_bias, base_instance, draw_id;
   uint32_t first_capture, vertices_per_instance, instance_bias, reserved;
};

/* Instance-rate attribute whose every instance reads the first instance's
 * element.
 */
#define AGX_APPLE9_VERTEX_DIVISOR_ZERO UINT32_MAX

struct agx_apple9_vertex_layout {
   uint64_t outputs_flat, outputs_linear;
   uint32_t stride[16];
   uint32_t divisor[16]; /* Zero selects the per-vertex stream. */
   /* UBO containing first vertex/base vertex and base instance, respectively.
    * It may point into an indirect descriptor written by the GPU. */
   uint16_t draw_params_ubo;
   /* Optional table of {uint64_t address, uint64_t size} per buffer binding.
    * Nonzero selects bounded global fetches, with full-width byte offsets. */
   uint16_t buffer_descriptor_ubo;
   /* The buffer table instead holds {uint64_t address, uint64_t size,
    * uint32_t stride, uint32_t pad[3]} per binding, and stride[] is ignored. */
   bool dynamic_stride;
   /* Vertex input robustness is disabled: descriptor-table fetches skip the
    * bounds check (an out-of-bounds attribute is undefined). Zero, the
    * default, keeps robust fetches. Requires dynamic_stride, and the driver
    * must describe a null binding as the zero page with size and stride 0. */
   bool unbounded;
   bool clip_halfz;      /* Convert GL [-w,w] depth to hardware [0,w]. */
   uint8_t clip_distance_enable;
   bool cull_distance_mask; /* Cull signs are combined by software geometry. */
   bool capture_xfb;
   bool rasterize_points;
   uint8_t xfb_mode;
   uint8_t xfb_index_size;
   bool xfb_flatshade_first;
   bool
      ignore_point_size; /* Valid only when rasterizing non-point primitives. */
   enum pipe_format format[16];
   uint32_t offset[16]; /* Attribute byte offset within its vertex buffer. */
   uint8_t
      buffer[16]; /* Gallium vertex-buffer binding, independent of location. */
};

bool agx_compile_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout,
   struct agx_shader_part *out, const char **reason);

/* The same vertex fetch lowering is used by hardware VS and Poly compute VS. */
bool agx_nir_lower_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout, bool software);

/* Compile the bounded procedural vertex stage described above. */
bool agx_compile_apple9_vertex(nir_shader *nir, struct agx_shader_part *out,
                               const char **reason);

/* Vertex-fetch packaging is not implemented; this entry point rejects. */
bool agx_compile_apple9_vertex_prolog(nir_shader *nir,
                                      struct agx_shader_part *out,
                                      const char **reason);

#ifdef __cplusplus
}
#endif

#endif
