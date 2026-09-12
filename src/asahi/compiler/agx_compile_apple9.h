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
#define AGX_APPLE9_GRAPHICS_SYSVAL_SIZE (16 + AGX_APPLE9_SAMPLER_BIAS_COUNT * sizeof(float))

/* Standard independent RGB/alpha blending, plus the RGBA write mask. */
struct agx_apple9_blend {
   uint8_t rgb_src, rgb_dst, alpha_src, alpha_dst;
   uint8_t rgb_func, alpha_func, colormask, unsupported;
   /* NONE retains the RGBA8 layout used by standalone compiler callers. */
   enum pipe_format format;
   uint8_t samples; /* Zero retains single-sample compiler callers. */
   uint8_t disabled_samples;
   uint8_t alpha_to_coverage, alpha_to_one;
};

static inline unsigned
agx_apple9_color_components(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R16_FLOAT: return 1;
   case PIPE_FORMAT_R16G16_FLOAT: return 2;
   default: return 4;
   }
}

static inline bool
agx_apple9_color_is_half(enum pipe_format format)
{
   return format == PIPE_FORMAT_R16_FLOAT ||
          format == PIPE_FORMAT_R16G16_FLOAT ||
          format == PIPE_FORMAT_R16G16B16A16_FLOAT;
}

static inline unsigned
agx_apple9_color_words(enum pipe_format format)
{
   return format == PIPE_FORMAT_R16G16B16A16_FLOAT ? 2 : 1;
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
};

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
