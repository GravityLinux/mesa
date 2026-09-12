/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_COMPILE_APPLE9_H
#define AGX_COMPILE_APPLE9_H

#include <stdbool.h>
#include <stdint.h>

#include "agx_compile.h"
#include "util/format/u_formats.h"
#include "util/format/u_format.h"
#include "agx_apple9_profile.h"

#ifdef __cplusplus
extern "C" {
#endif

bool agx_nir_lower_apple9_math(nir_shader *shader);

/*
 * Compile the deliberately bounded Apple9 straight-line NIR subset.
 *
 * This is a real instruction selector, not a lookup of complete shaders.  It
 * The compositional profile accepts one scalar 32-bit store to buffer(0),
 * indexed by global_invocation_id.x.  A second, separately captured profile
 * accepts two scalar float loads from buffers 0/1, one fadd, and a scalar
 * float store to buffer 2 at that same index.  The compiler selects exact
 * Apple9 instructions for both profiles and rejects every other memory/control
 * graph; it never falls back to a plausible but unproved package. Control
 * flow, spilling, vectors, and general memory operations remain later
 * milestones.
 *
 * On failure, *reason points at a static diagnostic string when reason is
 * non-NULL.  On success, out owns a malloc-backed main-program binary and may
 * be released with free(out->binary), like agx_compile_shader_nir output.
 */
bool agx_compile_apple9_tiny(nir_shader *nir, struct agx_shader_part *out,
                             struct agx_apple9_compute_profile *profile,
                             const char **reason);

/*
 * Compile structured FP32 graphics NIR through the common semantic VIR and
 * register allocator. Fragment inputs support pixel-center smooth FP32 user
 * components; RT0 is a complete vec4 packed to RGBA8. Vertex inputs use vertex
 * ID and formatted vertex elements; exports cover position plus up to twelve
 * user scalars across VAR0..VAR31, compacted by semantic location and
 * component. Fragment variants use the producer's layout, including unused
 * outputs. User UVS values stay unprojected for clipping and require
 * perspective CF bindings. The FS uses coefficient-aware projective
 * multiplication, which handles both ordinary and primitive-constant
 * coefficient representations. Constant array offsets and component holes are
 * supported. Up to 32 buffer resources per stage use a shader-loaded address
 * table and common memory lowering. Distinct vertex buffers and UBOs share that
 * budget. The resource-binding array records address-table order;
 * apple9_ubo_mask records API UBO bindings. Branches and loops use the common
 * execution-mask model, including lowered continuation constructs. Fragment
 * window XY uses upper-left integer pixel coordinates, matching Gallium's
 * advertised convention; API center/origin transforms belong to the caller.
 * Fragment sampling supports up to sixteen 2D textures and sixteen samplers,
 * independently compacted from live API bindings. Window Z/W, SSBOs, other
 * interpolation modes, and MRT fail closed. The caller supplies hardware clip
 * coordinates and compatible stage and render-target state.
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
#define AGX_APPLE9_GRAPHICS_SYSVAL_SIZE (AGX_APPLE9_TEXTURE_INFO_OFFSET + 32 * AGX_APPLE9_TEXTURE_INFO_STRIDE)

/* Standard independent RGB/alpha blending, plus the RGBA write mask. */
struct agx_apple9_blend {
   uint8_t rgb_src, rgb_dst, alpha_src, alpha_dst;
   uint8_t rgb_func, alpha_func, colormask, unsupported;
   /* NONE retains the RGBA8 layout used by standalone compiler callers. */
   enum pipe_format format;
   uint8_t samples; /* Zero retains single-sample compiler callers. */
   uint8_t disabled_samples;
   uint8_t alpha_to_coverage, alpha_to_one;
   uint8_t advanced_mode, advanced_overlap;
   uint8_t src_premultiplied, dst_premultiplied;
};

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
 * The PBE converts unpacked 16-bit components to packed destinations. */
static inline enum pipe_format
agx_apple9_color_tile_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R5G5B5A1_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_R4G4B4A4_UNORM:
   case PIPE_FORMAT_B4G4R4A4_UNORM:
      return PIPE_FORMAT_R16G16B16A16_FLOAT;
   case PIPE_FORMAT_R10G10B10A2_UINT:
      return PIPE_FORMAT_R16G16B16A16_UINT;
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
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
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
   AGX_APPLE9_CLAMP_S = 1 << 0,
   AGX_APPLE9_CLAMP_T = 1 << 1,
   AGX_APPLE9_CLAMP_R = 1 << 2,
   AGX_APPLE9_CUSTOM_BORDER = 1 << 3,
};

struct agx_apple9_sampler_key {
   uint32_t flags;
   float border[4];
   uint8_t wrap[3];
   uint8_t min_filter, mag_filter, mip_filter, compare_func, reserved;
   float min_lod, max_lod, lod_bias;
};

bool agx_nir_lower_apple9_texture_offsets(
   nir_shader *nir, const struct agx_apple9_sampler_key key[32],
   const char **reason);

struct agx_apple9_texture_mapping {
   uint8_t samplers[32];
   uint32_t white_samplers;
};

bool agx_nir_lower_apple9_sampler_state(
   nir_shader *nir, const struct agx_apple9_sampler_key key[32],
   struct agx_apple9_texture_mapping *mapping, const char **reason);

/* Per-target blend array contains nr_targets entries, up to eight. */
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
struct agx_apple9_vertex_layout {
   uint32_t stride[16];
   uint32_t divisor[16]; /* Zero selects the per-vertex stream. */
   bool clip_halfz;
   bool ignore_point_size; /* Valid only when rasterizing non-point primitives. */ /* Convert GL [-w,w] depth to hardware [0,w]. */
   enum pipe_format format[16];
   uint32_t offset[16]; /* Attribute byte offset within its vertex buffer. */
   uint8_t buffer[16]; /* Gallium vertex-buffer binding, independent of location. */
};

bool agx_compile_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout,
   struct agx_shader_part *out, const char **reason);

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
