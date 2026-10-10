/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/compiler/agx_compile.h"
#include "poly/nir/poly_nir.h"
#include "util/macros.h"
#include "agx_linker.h"
#include "asahi/compiler/agx_apple9_profile.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "agx_usc.h"
#include "agx_uvs.h"

#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_private.h"

#include "shader_enums.h"
#include "vk_pipeline_cache.h"

#include "nir_defines.h"

#include "vk_shader.h"

#define HK_APPLE9_DRAW_PARAMS_BINDING 65
#define HK_APPLE9_VERTEX_BUFFERS_BINDING 66

struct hk_physical_device;
struct hk_pipeline_compilation_ctx;
struct vk_descriptor_set_layout;
struct vk_graphics_pipeline_state;
struct vk_pipeline_cache;
struct vk_pipeline_layout;
struct vk_pipeline_robustness_state;
struct vk_shader_module;

struct hk_tess_info {
   enum tess_primitive_mode mode : 8;
   enum gl_tess_spacing spacing  : 8;
   bool points;
   bool ccw;
};
static_assert(sizeof(struct hk_tess_info) == 4, "packed");

static struct hk_tess_info
hk_tess_info_merge(struct hk_tess_info a, struct hk_tess_info b)
{
   static_assert(TESS_PRIMITIVE_UNSPECIFIED == 0, "zero state");
   static_assert(TESS_SPACING_UNSPECIFIED == 0, "zero state");

   /* Just merge by OR'ing the raw bits */
   uint32_t x, y;
   memcpy(&x, &a, sizeof(x));
   memcpy(&y, &b, sizeof(y));

   x |= y;

   struct hk_tess_info out;
   memcpy(&out, &x, sizeof(out));
   return out;
}

/* Static core descriptor bindings, resolved to table contents at recording. */
struct hk_apple9_texture_binding {
   uint32_t set;
   uint32_t offset_B;
   /* Samplers used with cube arrays, which sample the faces as a 2D array:
    * cube sampling ignores the address modes and clamps to the edge.
    */
   uint32_t clamp_to_edge;
};

/* Texture operations whose descriptors may change after binding, or be
 * partially bound, must read them at execution time on G16. */
#define HK_APPLE9_TEX_DESCRIPTOR_HANDLE (1u << 30)

/* On a nir_texop_lod_bias query: load the sampler's explicit-LOD nudge
 * instead of its LOD bias, see hk_lower_d3d_txl_rounding. */
#define HK_TEX_TXL_NUDGE (1u << 29)

struct hk_apple9_texture_bindings {
   uint8_t texture_count, sampler_count;
   /* Textures (and samplers) are addressed through their descriptors, and the
    * stage's table pointers name the USC aperture. */
   bool bindless_textures, bindless_samplers;
   struct hk_apple9_texture_binding textures[AGX_APPLE9_GRAPHICS_MAX_TEXTURES];
   struct hk_apple9_texture_binding samplers[AGX_APPLE9_GRAPHICS_MAX_SAMPLERS - 1];
};

/* Mesh shaders run as compute kernels. Each workgroup writes a record of
 * group_stride_B bytes: its vertex and primitive counts, padded to 16 bytes,
 * then max_vertices vertices and max_primitives primitives, each the varying
 * slots in the respective mask in slot order. Slots have the words of a vec4,
 * except for scalar builtins and the primitive indices, which have their
 * primitive's vertex count. The primitive slots include the primitive indices
 * and cull flag.
 */
struct hk_mesh_info {
   uint64_t vertex_slots;
   uint64_t primitive_slots;
   uint32_t group_stride_B;
   uint16_t max_vertices, max_primitives;
   uint8_t prim;
   uint8_t _pad[7];
};

/* Fragment shaders read user per-primitive varyings from the output records,
 * at the byte offset of the primitive, passed in this flat varying.
 */
#define HK_MESH_PRIMITIVE_SLOT VARYING_SLOT_VAR31

static inline bool
hk_mesh_varying_in_memory(unsigned slot)
{
   return slot >= VARYING_SLOT_VAR0 && slot < VARYING_SLOT_VAR31;
}

static inline unsigned
hk_mesh_slot_words(const struct hk_mesh_info *info, unsigned slot)
{
   switch (slot) {
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_PRIMITIVE_ID:
   case VARYING_SLOT_PRIMITIVE_SHADING_RATE:
   case VARYING_SLOT_CULL_PRIMITIVE:
      return 1;
   case VARYING_SLOT_PRIMITIVE_INDICES:
      return mesa_vertices_per_prim(info->prim);
   default:
      return 4;
   }
}

/* Offset of a slot within a vertex or primitive, or their size with
 * slot = 64.
 */
static inline unsigned
hk_mesh_slot_offset_B(const struct hk_mesh_info *info, uint64_t mask,
                      unsigned slot)
{
   unsigned offset_B = 0;
   u_foreach_bit64(s, mask & BITFIELD64_MASK(slot))
      offset_B += hk_mesh_slot_words(info, s) * 4;
   return offset_B;
}

static inline unsigned
hk_mesh_primitives_offset_B(const struct hk_mesh_info *info)
{
   return 16 + info->max_vertices *
                  hk_mesh_slot_offset_B(info, info->vertex_slots, 64);
}

void hk_mesh_gather_info(const nir_shader *nir, struct hk_mesh_info *info);
void hk_nir_lower_mesh_main(nir_shader *nir, const struct hk_mesh_info *info);
void hk_nir_lower_task_main(nir_shader *nir);
#define HK_MAX_PER_VERTEX_COPIES 4

/* Per-vertex fragment inputs read as flat copies: the VAR slot of each input,
 * and the first of the three consecutive VAR slots holding its values for API
 * vertices 0, 1 and 2. Barycentrics are read by interpolating the one-hot
 * weights of the API vertices from bary[0] (perspective) and bary[1]
 * (linear), as VAR index + 1 or 0 if unused.
 */
struct hk_per_vertex_copies {
   unsigned count;
   uint8_t location[HK_MAX_PER_VERTEX_COPIES];
   uint8_t copy_base[HK_MAX_PER_VERTEX_COPIES];
   uint8_t bary[2];
};

static inline bool
hk_per_vertex_copies_used(const struct hk_per_vertex_copies *map)
{
   return map->count || map->bary[0] || map->bary[1];
}

nir_shader *hk_nir_mesh_rast(const nir_shader *mesh,
                             const struct hk_mesh_info *info,
                             const struct hk_per_vertex_copies *copies);

struct hk_shader_info {
   uint8_t apple9_render_targets;
   struct hk_apple9_texture_bindings apple9_bindings;
   struct agx_apple9_compute_profile apple9;
   union {
      struct {
         uint32_t attribs_read;
         BITSET_DECLARE(attrib_components_read, AGX_MAX_ATTRIBS * 4);
         bool use_prolog;
         uint8_t _pad[7];
      } vs;

      struct {
         struct agx_interp_info interp;
         struct agx_fs_epilog_link_info epilog_key;

         bool reads_sample_mask;
         bool post_depth_coverage;
         bool uses_sample_shading;
         bool early_fragment_tests;
         bool writes_memory;

         /* Per-vertex inputs and barycentrics written by the passthrough
          * GS, as in struct hk_per_vertex_copies. */
         uint8_t per_vertex_copies;
         uint8_t per_vertex_location[HK_MAX_PER_VERTEX_COPIES];
         uint8_t per_vertex_copy_base[HK_MAX_PER_VERTEX_COPIES];
         uint8_t per_vertex_bary[2];
      } fs;

      struct {
         uint64_t tcs_per_vertex_outputs;
         uint32_t tcs_output_stride;
         uint8_t tcs_output_patch_size;
         uint8_t tcs_nr_patch_outputs;

         struct hk_tess_info info;
      } tess;

      struct poly_gs_info gs;

      struct hk_mesh_info mesh;

      struct {
         uint32_t payload_B;
      } task;

      /* Used to initialize the union for other stages */
      uint8_t _pad[32];
   };

   struct agx_unlinked_uvs_layout uvs;

   /* Transform feedback buffer strides */
   uint8_t xfb_stride[MAX_XFB_BUFFERS];

   mesa_shader_stage stage : 8;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;
   uint8_t set_count;

   /* XXX: is there a less goofy way to do this? I really don't want dynamic
    * allocation here.
    */
   nir_xfb_info xfb_info;
   nir_xfb_output_info xfb_outputs[64];
};

/*
 * Hash table keys for fast-linked shader variants. These contain the entire
 * prolog/epilog key so we only do 1 hash table lookup instead of 2 in the
 * general case where the linked shader is already ready.
 */
struct hk_fast_link_key_vs {
   struct agx_vs_prolog_key prolog;
};

struct hk_fast_link_key_fs {
   unsigned nr_samples_shaded;
   struct agx_fs_prolog_key prolog;
   struct agx_fs_epilog_key epilog;
};

struct hk_shader {
   struct agx_shader_part b;

   struct hk_shader_info info;
   struct agx_fragment_face_2_packed frag_face;
   struct agx_counts_packed counts;

   const void *code_ptr;
   uint32_t code_size;

   const void *data_ptr;
   uint32_t data_size;

   /* BO for any uploaded shader part */
   struct agx_bo *bo;

   /* G16 compute: launch program template, built once at upload from the
    * profile and preamble address and patched per dispatch. NULL falls back
    * to building the program every dispatch. */
   struct agx_apple9_launch_template *apple9_launch;

   /* Cache of fast linked variants */
   struct {
      simple_mtx_t lock;
      struct hash_table *ht;
   } linked;

   /* If there's only a single possibly linked variant, direct pointer. TODO:
    * Union with the cache to save some space?
    */
   struct hk_linked_shader *only_linked;

   /* Address to the uploaded preamble section. Preambles are uploaded
    * separately from fast-linked main shaders.
    */
   uint64_t preamble_addr;

   /* Address of the start of the shader data section */
   uint64_t data_addr;
};

enum hk_vs_variant {
   /* Hardware vertex shader, when next stage is fragment */
   HK_VS_VARIANT_HW,

   /* Hardware compute shader, when next is geometry/tessellation */
   HK_VS_VARIANT_SW,

   HK_VS_VARIANTS,
};

enum hk_gs_variant {
   /* Hardware vertex shader used for rasterization */
   HK_GS_VARIANT_RAST,

   /* Main compute shader */
   HK_GS_VARIANT_MAIN,

   /* Count compute shader */
   HK_GS_VARIANT_COUNT,

   /* Pre-GS compute shader */
   HK_GS_VARIANT_PRE,

   HK_GS_VARIANTS,
};

enum hk_mesh_variant {
   /* Compute kernel writing the output records */
   HK_MESH_VARIANT_MAIN,

   /* Hardware vertex shader rasterizing the records */
   HK_MESH_VARIANT_RAST,

   HK_MESH_VARIANTS,
};

/* clang-format off */
static const char *hk_gs_variant_name[] = {
   [HK_GS_VARIANT_RAST] = "Rasterization",
   [HK_GS_VARIANT_MAIN] = "Main",
   [HK_GS_VARIANT_COUNT] = "Count",
   [HK_GS_VARIANT_PRE] = "Pre-GS",
};
/* clang-format on */

static inline unsigned
hk_num_variants(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_EVAL:
      return HK_VS_VARIANTS;

   case MESA_SHADER_GEOMETRY:
      return HK_GS_VARIANTS;

   case MESA_SHADER_MESH:
      return HK_MESH_VARIANTS;

   default:
      return 1;
   }
}

/*
 * An hk_api shader maps 1:1 to a VkShader object. An hk_api_shader may contain
 * multiple hardware hk_shader's, built at shader compile time. This complexity
 * is required to efficiently implement the legacy geometry pipeline.
 */
/* Graphics stages of an Apple9 draw-time variant, indexed by stage. Compute
 * is never among them. */
#define HK_APPLE9_GFX_STAGES (MESA_SHADER_MESH + 1)

struct hk_apple9_deferred;

struct hk_api_shader {
   struct vk_shader vk;

   /* Is this an internal passthrough geometry shader? */
   bool is_passthrough;

   /* Apple9 graphics shaders whose key depends on state that is dynamic or
    * unknown at compile time keep their NIR and are compiled at draw time,
    * together with the other bound stages. Their variants are unused.
    */
   struct hk_apple9_deferred *apple9_deferred;

   struct hk_shader variants[];
};

struct vk_graphics_pipeline_state;

VkResult hk_apple9_resolve_graphics(
   struct hk_device *dev, struct hk_api_shader *const api[HK_APPLE9_GFX_STAGES],
   const struct vk_graphics_pipeline_state *state,
   struct hk_api_shader *out[HK_APPLE9_GFX_STAGES]);

bool hk_apple9_deferred_sample_shading(const struct hk_api_shader *fs);

void hk_apple9_variants_finish(struct hk_device *dev);
void hk_apple9_bodies_finish(struct hk_device *dev);

#define hk_foreach_variant(api_shader, var)                                    \
   for (struct hk_shader *var = api_shader->variants;                          \
        var < api_shader->variants + hk_num_variants(api_shader->vk.stage);    \
        ++var)

static const char *
hk_variant_name(struct hk_api_shader *obj, struct hk_shader *variant)
{
   unsigned i = variant - obj->variants;
   assert(i < hk_num_variants(obj->vk.stage));

   if (hk_num_variants(obj->vk.stage) == 1) {
      return NULL;
   } else if (obj->vk.stage == MESA_SHADER_GEOMETRY) {
      assert(i < ARRAY_SIZE(hk_gs_variant_name));
      return hk_gs_variant_name[i];
   } else if (obj->vk.stage == MESA_SHADER_MESH) {
      return i == HK_MESH_VARIANT_RAST ? "Rasterization" : "Main";
   } else {
      assert(i < 2);
      return i == HK_VS_VARIANT_SW ? "Software" : "Hardware";
   }
}

static struct hk_shader *
hk_only_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   assert(hk_num_variants(obj->vk.stage) == 1);
   return &obj->variants[0];
}

static struct hk_shader *
hk_any_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   return &obj->variants[0];
}

static struct hk_shader *
hk_main_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_MAIN];
}

static struct hk_shader *
hk_count_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_COUNT];
}

static struct hk_shader *
hk_pre_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_PRE];
}

#define HK_MAX_LINKED_USC_SIZE                                                 \
   (AGX_USC_PRESHADER_LENGTH + AGX_USC_FRAGMENT_PROPERTIES_LENGTH +            \
    AGX_USC_REGISTERS_LENGTH + AGX_USC_SHADER_LENGTH + AGX_USC_SHARED_LENGTH + \
    AGX_USC_SAMPLER_LENGTH + (AGX_USC_UNIFORM_LENGTH * 9))

struct hk_linked_shader {
   struct agx_linked_shader b;

   /* True if the VS prolog uses software indexing, either for geom/tess or
    * adjacency primitives.
    */
   bool sw_indexing;

   /* Distinct from hk_shader::counts due to addition of cf_binding_count, which
    * is delayed since it depends on cull distance.
    */
   struct agx_fragment_shader_word_0_packed fs_counts;

   /* Baked USC words to bind this linked shader */
   struct {
      uint8_t data[HK_MAX_LINKED_USC_SIZE];
      size_t size;
   } usc;
};

struct hk_linked_shader *hk_fast_link(struct hk_device *dev, bool fragment,
                                      struct hk_shader *main,
                                      struct agx_shader_part *prolog,
                                      struct agx_shader_part *epilog,
                                      unsigned nr_samples_shaded);

extern const struct vk_device_shader_ops hk_device_shader_ops;

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev);

static inline nir_address_format
hk_buffer_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness)
{
   switch (robustness) {
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT:
      return nir_address_format_64bit_global_32bit_offset;
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT:
      return nir_address_format_64bit_bounded_global;
   default:
      UNREACHABLE("Invalid robust buffer access behavior");
   }
}

bool hk_lower_uvs_index(nir_shader *s, mesa_shader_stage sw_stage,
                        unsigned nr_vbos);

bool
hk_nir_lower_descriptors(nir_shader *nir,
                         const struct vk_pipeline_robustness_state *rs,
                         uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts);

VkResult hk_compile_shader(struct hk_device *dev,
                           struct vk_shader_compile_info *info,
                           const struct vk_graphics_pipeline_state *state,
                           const struct vk_features *features,
                           const VkAllocationCallbacks *pAllocator,
                           struct hk_api_shader **shader_out);

void hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev,
                                nir_shader *nir);

void hk_api_shader_destroy(struct vk_device *vk_dev,
                           struct vk_shader *vk_shader,
                           const VkAllocationCallbacks *pAllocator);

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs);

VkResult hk_apple9_compile_internal_gs(struct hk_device *dev, nir_shader *nir,
                                       const uint64_t fs_interp[2],
                                       struct hk_api_shader **out);

struct hk_api_shader *hk_meta_shader(struct hk_device *dev,
                                     hk_internal_builder_t builder, void *data,
                                     size_t data_size);

VkResult hk_apple9_create_null_fs(struct hk_device *dev,
   const VkAllocationCallbacks *alloc, struct hk_api_shader **out);
