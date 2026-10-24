/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "asahi/debug.h"
#include <unistd.h>

#include "hk_shader.h"
#include "hk_gpu_busy.h"
#include "util/u_queue.h"

#include "poly/nir/poly_nir.h"
#include "poly/geometry.h"
#include "pipe/p_defines.h"
#include "agx_debug.h"
#include "hk_apple9_polygon_points.h"
#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/compiler/agx_apple9_mesh.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_nir_lower_vbo.h"
#include "glsl_types.h"
#include "hk_instance.h"
#include "libagx.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_format_convert.h"

#include "agx_bo.h"
#include "asahi/libagx/mesh.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "agx_timeline.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_shader.h"

#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_serialize.h"
#include "nir_xfb_info.h"
#include "shader_enums.h"
#include "vk_graphics_state.h"
#include "vk_blend.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_physical_device_features.h"
#include "vk_pipeline.h"
#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_shader_module.h"
#include "vk_util.h"
#include "vk_ycbcr_conversion.h"

#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_linker.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/lib/agx_uvs.h"
#include "compiler/spirv/nir_spirv.h"

#include "util/u_atomic.h"
#include "util/blob.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"

DEBUG_GET_ONCE_OPTION(cached_hk_apple9_dump_fragment_dir,
                      "HK_APPLE9_DUMP_FRAGMENT_DIR", NULL)
DEBUG_GET_ONCE_OPTION(cached_hk_apple9_eager_variants,
                      "HK_APPLE9_EAGER_VARIANTS", NULL)
DEBUG_GET_ONCE_BOOL_OPTION(apple9_variant_cache_verify,
                           "HK_APPLE9_VARIANT_CACHE_VERIFY", false)


/* Set while a pipeline-time eager variant compiles on this thread. */
static _Thread_local bool hk_apple9_eager_compile;

struct hk_fs_key {
   bool zs_self_dep;

   /** True if sample shading is forced on via an API knob such as
    * VkPipelineMultisampleStateCreateInfo::minSampleShading
    */
   bool force_sample_shading;

   uint8_t pad[2];
};
static_assert(sizeof(struct hk_fs_key) == 4, "packed");

struct hk_vs_key {
   struct agx_velem_key attribs[32];
   bool skip_prolog;
   bool static_strides;
   bool kill_psiz;
   bool pad[1];
};
static_assert(sizeof(struct hk_vs_key) == 260, "packed");

union hk_key {
   struct hk_vs_key vs;
   struct hk_fs_key fs;
};

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

/* Code-generation environment controls key the shader caches: a warm cache
 * must not hand one setting's shaders to another. Diagnostics and known
 * runtime-only controls are excluded. Zero when none is set keeps existing
 * default caches valid.
 */
static uint64_t hk_compiler_env_hash_value;
static once_flag hk_compiler_env_hash_once = ONCE_FLAG_INIT;

/* Environment knobs whose value changes generated code: their values key the
 * shader caches. Keep this list next to the knobs; any other AGX_* or
 * HK_APPLE9_* variable (traces, dumps, statistics, runtime switches, or a
 * codegen knob missing here) keys the caches by name only, so output paths
 * never split them while an unlisted knob still does, conservatively.
 */
static const char *const hk_codegen_env[] = {
   "AGX_MESA_DEBUG",
   "AGX_SPILL_COALESCE",
   "AGX_APPLE9_NO_COMPARE_LITERAL",
   "AGX_APPLE9_GATED",
   "AGX_APPLE9_HOIST_LOADS",
   "AGX_APPLE9_HOIST_SAMPLES",
   "AGX_APPLE9_LICM",
   "AGX_APPLE9_LICM_SPECULATE",
   "AGX_APPLE9_NO_DESCRIPTOR_HOIST",
   "AGX_APPLE9_NO_POINTER_LOAD",
   "AGX_APPLE9_NO_PROMOTE",
   "AGX_APPLE9_NO_SAMPLER_HOIST",
   "AGX_APPLE9_NO_UNIFORM_MOVE",
   "AGX_APPLE9_SAFE_HOIST_STAGES",
   "HK_APPLE9_DESCRIPTOR_BUFFER",
   "HK_APPLE9_MESH_SHADER",
   "HK_APPLE9_NATIVE_MESH",
   "HK_APPLE9_NATIVE_MESH_ATTRIBUTES",
   "HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING",
   "HK_APPLE9_NATIVE_MESH_PACKED_SHAPES",
   "HK_APPLE9_NATIVE_INDEX",
   "HK_APPLE9_NATIVE_STAGES",
   "HK_APPLE9_GS_MESH_PROBE",
   "HK_APPLE9_GS_NATIVE",
   "HK_APPLE9_TESS_MESH_PROBE",
   "HK_APPLE9_TESS_NATIVE",
};

/* These controls change diagnostics, memory backing, when a variant is
 * compiled, or command recording/submission. They do not change shader code.
 * Keep them available for runtime investigations without splitting caches. */
static const char *const hk_ignored_env[] = {
   "HK_APPLE9_VARIANT_CACHE_VERIFY", "HK_APPLE9_DUMP_FRAGMENT_DIR",
   "HK_APPLE9_EAGER_VARIANTS",       "HK_APPLE9_SPARSE_SCRATCH",
   "HK_APPLE9_TESS_BATCH",           "HK_APPLE9_TESS_SCAN_BLOCK",
   "HK_APPLE9_MERGE_RENDERS",        "HK_APPLE9_MERGE_RESUME",
   "HK_APPLE9_MERGE_HOIST_BARRIERS", "HK_APPLE9_MERGE_MOVE_COMPUTE",
   "HK_APPLE9_FRAGMENT_BARRIER",     "HK_APPLE9_NO_TILING_ELISION",
   "HK_APPLE9_MESH_ROUTE_LOG",       "HK_APPLE9_GS_ROUTE_LOG",
   "HK_APPLE9_TESS_ROUTE_LOG",
};

static bool
hk_env_named(const char *var, const char *const *names, unsigned count)
{
   const size_t len = strcspn(var, "=");
   for (unsigned i = 0; i < count; ++i) {
      if (strlen(names[i]) == len && !strncmp(var, names[i], len))
         return true;
   }
   return false;
}

/* How an environment entry keys the caches: 0 not at all, 1 by name, 2 by
 * value. */
static unsigned
hk_env_key_kind(const char *var)
{
   if (strncmp(var, "AGX_", 4) && strncmp(var, "HK_APPLE9_", 10))
      return 0;
   if (agx_apple9_is_debug_env(var) ||
       hk_env_named(var, hk_ignored_env, ARRAY_SIZE(hk_ignored_env)))
      return 0;
   return hk_env_named(var, hk_codegen_env, ARRAY_SIZE(hk_codegen_env)) ? 2 : 1;
}

static int
hk_compare_env(const void *a, const void *b)
{
   return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Native ABI changes invalidate both device/compiler and graphics-state
 * caches. The source NIR already keys stage-specific output maps/capacities;
 * this keys the shared layout contract without struct padding. Bump the ABI
 * version when its layout algorithm or draw-slot interpretation changes. */
static void
hk_hash_apple9_native_mesh_abi(struct mesa_blake3 *ctx)
{
   const struct agx_apple9_mesh_key k = AGX_APPLE9_MESH_KEY;
   uint64_t packed_shapes = hk_apple9_native_mesh_packed_shapes();
   const uint32_t abi[] = {
      AGX_APPLE9_MESH_ABI_VERSION,
      k.reserved_uniform_words,
      k.output_root,
      k.primitive_count_bits,
      k.payload_uniform,
      k.grid_root,
      k.grid_offset_B,
      k.physical_threads,
      AGX_APPLE9_MESH_DRAW_UNIFORM,
      AGX_APPLE9_MESH_ARGUMENT_WORDS,
      hk_apple9_native_mesh_attributes_enabled(),
      2, /* Explicit native pairing and on-demand deferred realizations. */
      packed_shapes,
      packed_shapes >> 32,
      hk_apple9_native_gs_enabled(),
      hk_apple9_native_tess_enabled(),
      1, /* P1 logical workgroups independent of physical output class. */
      2, /* Destination-addressed publication rounds beyond 32 entries. */
   };
   _mesa_blake3_update(ctx, abi, sizeof(abi));
}

static void
hk_compiler_env_hash_init(void)
{
   extern char **environ;
   unsigned count = 0;
   for (char **e = environ; e && *e; ++e)
      count += hk_env_key_kind(*e) != 0;
   if (!count)
      return;

   const char **vars = malloc(count * sizeof(*vars));
   if (!vars) {
      hk_compiler_env_hash_value = ~0ull;
      return;
   }
   unsigned n = 0;
   for (char **e = environ; e && *e && n < count; ++e) {
      if (hk_env_key_kind(*e))
         vars[n++] = *e;
   }
   qsort(vars, n, sizeof(*vars), hk_compare_env);

   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   for (unsigned i = 0; i < n; ++i) {
      /* Name-only knobs hash "NAME", value knobs "NAME=VALUE". */
      const size_t len = hk_env_key_kind(vars[i]) == 2 ? strlen(vars[i])
                                                       : strcspn(vars[i], "=");
      _mesa_blake3_update(&ctx, vars[i], len);
      _mesa_blake3_update(&ctx, "", 1);
   }
   if (debug_get_bool_option("HK_APPLE9_NATIVE_MESH", false))
      hk_hash_apple9_native_mesh_abi(&ctx);
   blake3_hash hash;
   _mesa_blake3_final(&ctx, hash);
   memcpy(&hk_compiler_env_hash_value, hash, sizeof(hk_compiler_env_hash_value));
   free(vars);
}

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev)
{
   call_once(&hk_compiler_env_hash_once, hk_compiler_env_hash_init);
   /* Soft faults follow a kernel parameter, so they can change between
    * boots with the same cache. */
   uint64_t flags = pdev->dev.debug |
                    ((uint64_t)pdev->apple9_layered_2d << 63) |
                    ((uint64_t)agx_has_soft_fault((struct agx_device *)&pdev->dev) << 62) |
                    ((uint64_t)pdev->apple9_d3d_mip_rounding << 61) |
                    ((uint64_t)hk_apple9_native_index_enabled(pdev) << 60) |
                    ((uint64_t)hk_apple9_native_stage_policy(pdev) << 58);
   return flags ^ hk_compiler_env_hash_value;
}

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs)
{
   return &agx_nir_options;
}

static struct spirv_to_nir_options
hk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs)
{
   const struct hk_physical_device *pdev =
      container_of(vk_pdev, struct hk_physical_device, vk);

   return (struct spirv_to_nir_options){
      .ssbo_addr_format = hk_buffer_addr_format(rs->storage_buffers),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = hk_buffer_addr_format(rs->uniform_buffers),
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ssbo_alignment = hk_min_ssbo_alignment(&pdev->dev),
      .min_ubo_alignment = HK_MIN_UBO_ALIGNMENT,
   };
}

void
hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev, nir_shader *nir)
{
   /* Must lower before io to temps */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_terminate_to_demote);
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   } else if (nir->info.stage == MESA_SHADER_TASK) {
      /* Launching mesh workgroups ends the task workgroup. */
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   }

   /* Unroll loops before lowering indirects via
    * nir_lower_io_vars_to_temporaries */
   UNUSED bool progress = false;
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);

   /* Treat position, point size, clip/cull distances and tessellation levels
    * as invariant, as RADV does. Whether NIR fuses a multiply into an add
    * depends on the multiply's other users, so the same position math rounds
    * differently in shaders that also write other outputs from it. Blender
    * EEVEE's motion-vector prepass and its material pass then disagree on
    * depth by a few ULPs, and the material pass's EQUAL test drops most of
    * its pixels.
    */
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_propagate_invariant, true);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      struct nir_lower_sysvals_to_varyings_options sysvals_opts = {
         .point_coord = true,
         .primitive_id = nir->info.stage == MESA_SHADER_FRAGMENT,
      };

      nir_lower_sysvals_to_varyings(nir, &sysvals_opts);
   }

   NIR_PASS(_, nir, nir_lower_system_values);

   /* Gather info before preprocess_nir but after some general lowering, so
    * inputs_read and system_values_read are accurately set.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* Mesh and tessellation control outputs are shared by the workgroup or
    * patch: each invocation may write a different element, so copying whole
    * temporaries back at the end would overwrite the other invocations'
    * writes. Without temporaries, lower component accesses before I/O
    * lowering.
    */
   if (nir->info.stage == MESA_SHADER_MESH ||
       nir->info.stage == MESA_SHADER_TESS_CTRL) {
      NIR_PASS(_, nir, nir_lower_array_deref_of_vec, nir_var_shader_out, NULL,
               nir_lower_direct_array_deref_of_vec_load |
                  nir_lower_indirect_array_deref_of_vec_load |
                  nir_lower_direct_array_deref_of_vec_store |
                  nir_lower_indirect_array_deref_of_vec_store);
   } else {
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
               nir_shader_get_entrypoint(nir), nir_var_shader_out);
   }

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);

   /* Extract immutable tables before agx_preprocess_nir lowers large local
    * arrays to scratch. The analysis requires explicit loads/stores rather
    * than copies. Keep the data attached to NIR through variant cloning.
    */
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   struct hk_physical_device *pdev =
      container_of(vk_pdev, struct hk_physical_device, vk);
   if (pdev->dev.chip == AGX_CHIP_G16G)
      NIR_PASS(_, nir, agx_nir_lower_apple9_math);
   agx_preprocess_nir(nir);

   /* Constant-offset reads can fold away after extraction. Do not upload
    * or bind an unused blob: besides wasting storage, that would extend
    * every draw's root for a table the program never reads.
    */
   if (nir->constant_data_size) {
      bool used = false;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               used |= instr->type == nir_instr_type_intrinsic &&
                       nir_instr_as_intrinsic(instr)->intrinsic ==
                          nir_intrinsic_load_constant;
            }
         }
      }
      if (!used) {
         ralloc_free(nir->constant_data);
         nir->constant_data = NULL;
         nir->constant_data_size = 0;
      }
   }

   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_var_copies);
}

static void
hk_preprocess_nir(struct vk_physical_device *vk_pdev, nir_shader *nir,
                  UNUSED const struct vk_pipeline_robustness_state *rs)
{
   hk_preprocess_nir_internal(vk_pdev, nir);
   /* Only compute dispatches have a base workgroup. */
   nir_lower_compute_system_values_options csv_options = {
      .has_base_workgroup_id = nir->info.stage == MESA_SHADER_COMPUTE,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
}

static void
hk_populate_vs_key(struct hk_vs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));

   if (state && state->ia &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)) {

      key->kill_psiz = state->ia->primitive_topology != MESA_PRIM_POINTS;
   }

   if (state && state->vi && !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI) &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDINGS_VALID)) {

      agx_fill_velem_keys(state->vi, ~0 /* compacted on use */, key->attribs);
      key->skip_prolog = true;
      key->static_strides =
         !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDING_STRIDES);

      if (!key->static_strides) {
         for (unsigned i = 0; i < ARRAY_SIZE(key->attribs); ++i) {
            key->attribs[i].stride = 0;
         }
      }
   }
}

static void
hk_populate_fs_key(struct hk_fs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));

   if (state == NULL)
      return;

   if (state->pipeline_flags &
       VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)
      key->zs_self_dep = true;

   /* We force per-sample interpolation whenever sampleShadingEnable is set
    * regardless of minSampleShading or rasterizationSamples.
    *
    * When sampleShadingEnable is set, few guarantees are made about the
    * location of interpolation of the inputs.  The only real guarantees are
    * that the inputs are interpolated within the pixel and that you get at
    * least `rasterizationSamples * minSampleShading` unique positions.
    * Importantly, it does not require that when `rasterizationSamples *
    * minSampleShading <= 1.0` that those positions are at the fragment
    * center.  Therefore, it's valid to just always do per-sample all the time.
    *
    * The one caveat here is that we have to be careful about gl_SampleMaskIn.
    * When `hk_fs_key::force_sample_shading = true` we also turn any reads of
    * gl_SampleMaskIn into `1 << gl_SampleID` because the hardware sample mask
    * is actually per-fragment, not per-pass.  We handle this by smashing
    * minSampleShading to 1.0 whenever gl_SampleMaskIn is read.
    */
   const struct vk_multisample_state *ms = state->ms;
   if (ms != NULL && ms->sample_shading_enable)
      key->force_sample_shading = true;
}

enum hk_feature_key {
   HK_FEAT_CUSTOM_BORDER = BITFIELD_BIT(0),
   /* Apple9 lowers the image view min LOD clamp only when it is enabled. */
   HK_FEAT_MIN_LOD = BITFIELD_BIT(1),
};

static enum hk_feature_key
hk_make_feature_key(const struct vk_features *feats)
{
   return ((!feats || feats->customBorderColors) ? HK_FEAT_CUSTOM_BORDER : 0) |
          ((!feats || feats->minLod) ? HK_FEAT_MIN_LOD : 0);
}

static bool
hk_apple9_lower_clip_z_coeff(nir_builder *b, nir_intrinsic_instr *intr,
                             UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_clip_z_coeff)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_float(b, 0.5));
   return true;
}

/* Route fragment outputs to the color attachments naming their locations. */
static bool
hk_apple9_remap_fs_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const uint8_t *map = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location < FRAG_RESULT_DATA0 || sem.dual_source_blend_index)
      return false;
   unsigned location = sem.location - FRAG_RESULT_DATA0;
   unsigned rt = 0;
   while (rt < 8 && map[rt] != location)
      ++rt;
   if (rt == 8) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (rt == location)
      return false;
   sem.location = FRAG_RESULT_DATA0 + rt;
   nir_intrinsic_set_io_semantics(intr, sem);
   return true;
}

static void
hk_apple9_remap_fs_outputs(nir_shader *nir, const uint8_t map[8])
{
   bool identity = true;
   for (unsigned rt = 0; rt < 8; ++rt)
      identity &= map[rt] == rt;
   if (identity)
      return;
   nir_shader_intrinsics_pass(nir, hk_apple9_remap_fs_output,
                              nir_metadata_control_flow, (void *)map);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

/* Pipeline sample shading shades each sample, so interpolate every non-flat
 * input at its sample, as FragCoord already is through its sample position.
 * Apple9 interpolates explicitly and does not move pixel or centroid
 * barycentrics when shading per sample.
 */
static bool
hk_apple9_lower_bary_to_sample(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   nir_def *sample;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
      sample = nir_load_barycentric_sample(
         b, intr->def.bit_size, .interp_mode = nir_intrinsic_interp_mode(intr));
      break;
   case nir_intrinsic_load_barycentric_coord_pixel:
   case nir_intrinsic_load_barycentric_coord_centroid:
      sample = nir_load_barycentric_coord_sample(
         b, intr->def.bit_size, .interp_mode = nir_intrinsic_interp_mode(intr));
      break;
   default:
      return false;
   }

   nir_def_replace(&intr->def, sample);
   return true;
}

/* Per-vertex inputs and barycentrics follow the API vertex order. Points read
 * their one vertex for every index and lines read their second vertex as the
 * third. The rasterizer orders a first-provoking triangle fan's vertices
 * (centre, then the rim), a rotation of the API order (rim, centre). The
 * primitive class (bits 0-1) and the rotation (bit 2) are draw state, so
 * select at run time from the root table.
 */
static nir_def *
hk_apple9_rasterized_prim(nir_builder *b)
{
   const unsigned offset = hk_root_descriptor_offset(draw.rasterized_prim);
   return nir_load_ubo(b, 1, 16, nir_imm_int(b, 0), nir_imm_int(b, offset),
                       .align_mul = 2, .range_base = offset, .range = 2);
}

static nir_def *
hk_apple9_load_vertex(nir_builder *b, nir_intrinsic_instr *intr,
                      unsigned vertex)
{
   nir_intrinsic_instr *load =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
   nir_def *index = nir_imm_int(b, vertex);
   nir_builder_instr_insert(b, &load->instr);
   nir_src_rewrite(&load->src[0], index);
   return &load->def;
}

static bool
hk_apple9_lower_api_vertex_order(nir_shader *nir)
{
   struct util_dynarray loads;
   util_dynarray_init(&loads, NULL);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if ((intr->intrinsic == nir_intrinsic_load_input_vertex &&
                 nir_src_is_const(intr->src[0])) ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_pixel ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_centroid ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_sample ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_at_sample ||
                intr->intrinsic == nir_intrinsic_load_barycentric_coord_at_offset)
               util_dynarray_append(&loads, intr);
         }
      }
   }

   util_dynarray_foreach(&loads, nir_intrinsic_instr *, it) {
      nir_intrinsic_instr *intr = *it;
      nir_builder b = nir_builder_at(nir_after_instr(&intr->instr));
      nir_def *prim = hk_apple9_rasterized_prim(&b);
      nir_def *rotate = nir_ine_imm(&b, nir_iand_imm(&b, prim, 4), 0);
      nir_def *result;

      if (intr->intrinsic != nir_intrinsic_load_input_vertex) {
         /* API weights (w0, w1, w2) are the rasterizer's (w1, w2, w0). */
         nir_def *w = &intr->def;
         nir_def *rotated = nir_vec3(&b, nir_channel(&b, w, 1),
                                     nir_channel(&b, w, 2), nir_channel(&b, w, 0));
         result = nir_bcsel(&b, rotate, rotated, w);
      } else {
         unsigned k = nir_src_as_uint(intr->src[0]);
         nir_def *class = nir_iand_imm(&b, prim, 3);
         nir_def *v[3];
         for (unsigned i = 0; i < 3; ++i)
            v[i] = i == k ? &intr->def : hk_apple9_load_vertex(&b, intr, i);
         result = nir_bcsel(&b, rotate, v[(k + 1) % 3], v[k]);
         if (k == 2)
            result = nir_bcsel(&b, nir_ieq_imm(&b, class, 1), v[1], result);
         if (k != 0)
            result = nir_bcsel(&b, nir_ieq_imm(&b, class, 2), v[0], result);
      }
      nir_def_rewrite_uses_after(&intr->def, result);
   }

   bool progress = loads.size != 0;
   util_dynarray_fini(&loads);
   return nir_progress(progress, nir_shader_get_entrypoint(nir), nir_metadata_control_flow);
}

/* Per-vertex inputs are read as flat copies: the stage feeding the
 * rasterizer writes the values of the primitive's three API vertices to every
 * vertex. Flat varyings keep the provoking vertex's values when the
 * rasterizer clips at its guard band, so this is exact for clipped
 * primitives, where the raw-vertex coefficients hold the clipper's vertices.
 * Likewise the hardware barycentrics are relative to the clipped piece, so
 * barycentrics interpolate one-hot weights of the API vertices instead. The
 * copies take three consecutive VAR slots and the weights one VAR slot per
 * interpolation mode, which neither the producer writes nor the fragment
 * shader reads.
 */
static bool
hk_is_barycentric_coord(nir_intrinsic_op op)
{
   return op == nir_intrinsic_load_barycentric_coord_pixel ||
          op == nir_intrinsic_load_barycentric_coord_centroid ||
          op == nir_intrinsic_load_barycentric_coord_sample ||
          op == nir_intrinsic_load_barycentric_coord_at_sample ||
          op == nir_intrinsic_load_barycentric_coord_at_offset;
}

static nir_def *
hk_barycentric_for_coord(nir_builder *b, nir_intrinsic_instr *intr)
{
   enum glsl_interp_mode mode = nir_intrinsic_interp_mode(intr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_coord_pixel:
      return nir_load_barycentric_pixel(b, 32, .interp_mode = mode);
   case nir_intrinsic_load_barycentric_coord_centroid:
      return nir_load_barycentric_centroid(b, 32, .interp_mode = mode);
   case nir_intrinsic_load_barycentric_coord_sample:
      return nir_load_barycentric_sample(b, 32, .interp_mode = mode);
   case nir_intrinsic_load_barycentric_coord_at_sample:
      return nir_load_barycentric_at_sample(b, 32, intr->src[0].ssa,
                                            .interp_mode = mode);
   case nir_intrinsic_load_barycentric_coord_at_offset:
      return nir_load_barycentric_at_offset(b, 32, intr->src[0].ssa,
                                            .interp_mode = mode);
   default:
      UNREACHABLE("not a barycentric coordinate");
   }
}

static bool
hk_apple9_lower_per_vertex_copies(nir_shader *fs, uint64_t producer_outputs,
                                  struct hk_per_vertex_copies *map)
{
   memset(map, 0, sizeof(*map));
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   uint64_t used = fs->info.inputs_read | producer_outputs;
   bool bary[2] = {false, false};

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (hk_is_barycentric_coord(intr->intrinsic)) {
            bary[nir_intrinsic_interp_mode(intr) ==
                 INTERP_MODE_NOPERSPECTIVE] = true;
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_load_input_vertex)
            continue;
         if (!nir_src_is_const(intr->src[0]) || !nir_src_is_const(intr->src[1]))
            return false;
         unsigned location = nir_intrinsic_io_semantics(intr).location +
                             nir_src_as_uint(intr->src[1]);
         if (location < VARYING_SLOT_VAR0 || location >= VARYING_SLOT_VAR0 + 32)
            return false;
         uint8_t var = location - VARYING_SLOT_VAR0;
         unsigned j;
         for (j = 0; j < map->count && map->location[j] != var; ++j);
         if (j == map->count) {
            if (map->count == HK_MAX_PER_VERTEX_COPIES)
               return false;
            map->location[map->count++] = var;
         }
      }
   }
   if (!map->count && !bary[0] && !bary[1])
      return false;

   /* Allocate from the top of the VAR range. */
   int next = 31;
   for (unsigned j = 0; j < map->count; ++j) {
      while (next >= 2 &&
             (used & BITFIELD64_RANGE(VARYING_SLOT_VAR0 + next - 2, 3)))
         --next;
      if (next < 2)
         return false;
      map->copy_base[j] = next - 2;
      used |= BITFIELD64_RANGE(VARYING_SLOT_VAR0 + next - 2, 3);
      next -= 3;
   }
   for (unsigned m = 0; m < 2; ++m) {
      if (!bary[m])
         continue;
      while (next >= 0 && (used & BITFIELD64_BIT(VARYING_SLOT_VAR0 + next)))
         --next;
      if (next < 0)
         return false;
      map->bary[m] = next + 1;
      used |= BITFIELD64_BIT(VARYING_SLOT_VAR0 + next);
      --next;
   }

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(&intr->instr);

         if (hk_is_barycentric_coord(intr->intrinsic)) {
            unsigned m = nir_intrinsic_interp_mode(intr) ==
                         INTERP_MODE_NOPERSPECTIVE;
            nir_def *weights = nir_load_interpolated_input(
               &b, 3, 32, hk_barycentric_for_coord(&b, intr),
               nir_imm_int(&b, 0), .dest_type = nir_type_float32,
               .io_semantics = {
                  .location = VARYING_SLOT_VAR0 + map->bary[m] - 1,
                  .num_slots = 1,
               });
            nir_def_replace(&intr->def, weights);
            continue;
         }

         if (intr->intrinsic != nir_intrinsic_load_input_vertex)
            continue;
         unsigned location = nir_intrinsic_io_semantics(intr).location +
                             nir_src_as_uint(intr->src[1]);
         unsigned j;
         for (j = 0; map->location[j] != location - VARYING_SLOT_VAR0; ++j);
         unsigned vertex = nir_src_as_uint(intr->src[0]);

         nir_def *flat = nir_load_input(
            &b, intr->def.num_components, intr->def.bit_size, nir_imm_int(&b, 0),
            .component = nir_intrinsic_component(intr),
            .dest_type = nir_intrinsic_dest_type(intr),
            .io_semantics = {
               .location = VARYING_SLOT_VAR0 + map->copy_base[j] + vertex,
               .num_slots = 1,
            });
         nir_def_replace(&intr->def, flat);
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
   nir_shader_gather_info(fs, impl);
   return true;
}

/* Triangle merging shades a pixel's samples from adjacent strip or fan
 * triangles together, interpolated on one triangle's plane. That is exact at
 * the pixel centre but not at the other triangle's samples, so per-sample
 * shading and centroid, sample or offset interpolation keep the triangles
 * apart.
 */
static bool
hk_apple9_interpolates_off_center(nir_shader *nir)
{
   if (nir->info.fs.uses_sample_shading)
      return true;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_barycentric_centroid:
            case nir_intrinsic_load_barycentric_sample:
            case nir_intrinsic_load_barycentric_at_sample:
            case nir_intrinsic_load_barycentric_at_offset:
            case nir_intrinsic_load_barycentric_coord_centroid:
            case nir_intrinsic_load_barycentric_coord_sample:
            case nir_intrinsic_load_barycentric_coord_at_sample:
            case nir_intrinsic_load_barycentric_coord_at_offset:
               return true;
            default:
               break;
            }
         }
      }
   }
   return false;
}

/* The one-hot barycentric weights of a vertex of a non-indexed triangle list:
 * its corner is its index within the draw, modulo 3.
 */
static void
hk_apple9_write_vertex_weights(nir_shader *vs,
                               const struct hk_per_vertex_copies *map)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(vs);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   /* The draw parameters hold the first vertex, as for load_first_vertex,
    * which the root lowering has already replaced. */
   nir_def *first = nir_load_ubo(
      &b, 1, 32, nir_imm_int(&b, HK_APPLE9_DRAW_PARAMS_BINDING),
      nir_imm_int(&b, 0), .align_mul = 4, .range_base = 0, .range = 4);
   nir_def *corner =
      nir_umod_imm(&b, nir_isub(&b, nir_load_vertex_id(&b), first), 3);
   nir_def *weights = nir_vec3(&b, nir_b2f32(&b, nir_ieq_imm(&b, corner, 0)),
                               nir_b2f32(&b, nir_ieq_imm(&b, corner, 1)),
                               nir_b2f32(&b, nir_ieq_imm(&b, corner, 2)));
   for (unsigned m = 0; m < 2; ++m) {
      if (!map->bary[m])
         continue;
      unsigned slot = VARYING_SLOT_VAR0 + map->bary[m] - 1;
      nir_store_output(&b, weights, nir_imm_int(&b, 0), .write_mask = 0x7,
                       .src_type = nir_type_float32,
                       .io_semantics.location = slot,
                       .io_semantics.num_slots = 1);
      vs->info.outputs_written |= BITFIELD64_BIT(slot);
   }
   nir_progress(true, impl, nir_metadata_control_flow);
}

/*
 * Line stipple. The last pre-rasterization stage exports two slots. The flat
 * start slot holds a segment's first vertex in window space, snapped to the
 * vertex grid, and the stipple counter there: zero for independent segments,
 * the length of the strip so far where a geometry stage continues one. The
 * noperspective axis slot holds the vertex's own window position, which
 * interpolates to a point on the ideal line. The fragment's counter is the
 * start's counter plus its distance from the start along the major axis, and
 * the interpolated point decides that axis: a wide line's fragments lie off
 * the line, where comparing the fragment's own distances could pick the wrong
 * one. The window transform is the vertex's own viewport's, from the runtime
 * sysvals.
 */
/* Record the stage's latest position and viewport index outputs in
 * variables, for the stipple start written after them. */
struct hk_apple9_stipple_tap {
   nir_variable *pos, *viewport;
};

static struct hk_apple9_stipple_tap
hk_apple9_tap_position(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   struct hk_apple9_stipple_tap tap = {
      .pos = nir_local_variable_create(impl, glsl_vec4_type(), "stipple_pos"),
      .viewport =
         nir_local_variable_create(impl, glsl_uint_type(), "stipple_viewport"),
   };
   nir_store_var(&b, tap.pos, nir_imm_vec4(&b, 0.0, 0.0, 0.0, 1.0), 0xf);
   nir_store_var(&b, tap.viewport, nir_imm_int(&b, 0), 1);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned location = nir_intrinsic_io_semantics(intr).location;
         b.cursor = nir_after_instr(instr);
         nir_def *value = intr->src[0].ssa;
         if (location == VARYING_SLOT_POS) {
            nir_def *v = nir_load_var(&b, tap.pos);
            unsigned first = nir_intrinsic_component(intr);
            u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
               v = nir_vector_insert_imm(&b, v, nir_channel(&b, value, c),
                                         first + c);
            }
            nir_store_var(&b, tap.pos, v, 0xf);
         } else if (location == VARYING_SLOT_VIEWPORT) {
            nir_store_var(&b, tap.viewport, nir_channel(&b, value, 0), 1);
         }
      }
   }
   return tap;
}

/* The snapped window position of the tapped vertex, in its own viewport. */
static nir_def *
hk_apple9_stipple_window(nir_builder *b, struct hk_apple9_stipple_tap tap)
{
   nir_def *p = nir_load_var(b, tap.pos);
   /* Vertices behind the eye clip the segment, where the counter's start is
    * implementation-dependent; keep the divide finite. */
   nir_def *w = nir_fmax(b, nir_channel(b, p, 3), nir_imm_float(b, 0x1p-16));
   nir_def *ndc = nir_fdiv(b, nir_trim_vector(b, p, 2), w);
   nir_def *index = nir_umin(b, nir_load_var(b, tap.viewport),
                             nir_imm_int(b, AGX_APPLE9_VIEWPORT_TABLE_COUNT - 1));
   nir_def *transform = nir_load_ubo(
      b, 4, 32, nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_iadd_imm(b, nir_imul_imm(b, index, 16),
                   AGX_APPLE9_VIEWPORT_TABLE_OFFSET),
      .align_mul = 16, .range_base = AGX_APPLE9_VIEWPORT_TABLE_OFFSET,
      .range = AGX_APPLE9_VIEWPORT_TABLE_COUNT * 16);
   return agx_nir_line_stipple_snap(
      b, nir_ffma(b, ndc, nir_channels(b, transform, 0x3),
                  nir_channels(b, transform, 0xc)));
}

/* The start slot holds the segment's first vertex and the stipple counter
 * there; the axis slot the vertex's own position. */
static void
hk_apple9_store_line_stipple(nir_builder *b, nir_shader *s,
                             struct hk_apple9_stipple_slots slots,
                             nir_def *start, nir_def *base, nir_def *window)
{
   nir_store_output(b, nir_vec3(b, nir_channel(b, start, 0),
                                nir_channel(b, start, 1), base),
                    nir_imm_int(b, 0), .write_mask = 0x7,
                    .src_type = nir_type_float32,
                    .io_semantics.location = slots.start,
                    .io_semantics.num_slots = 1);
   nir_store_output(b, window, nir_imm_int(b, 0), .write_mask = 0x3,
                    .src_type = nir_type_float32,
                    .io_semantics.location = slots.axis,
                    .io_semantics.num_slots = 1);
   s->info.outputs_written |=
      BITFIELD64_BIT(slots.start) | BITFIELD64_BIT(slots.axis);
}

/* Vertex and evaluation shaders: every vertex starts its segment, with the
 * counter at zero. Strips continue in the passthrough geometry shader. */
static void
hk_apple9_write_line_stipple_start(nir_shader *vs,
                                   struct hk_apple9_stipple_slots slots)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(vs);
   struct hk_apple9_stipple_tap tap = hk_apple9_tap_position(vs);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_def *window = hk_apple9_stipple_window(&b, tap);
   hk_apple9_store_line_stipple(&b, vs, slots, window, nir_imm_float(&b, 0.0),
                                window);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, vs, nir_lower_vars_to_ssa);
}

/*
 * Geometry shaders emitting line strips: the counter continues along each
 * output strip and restarts at EndPrimitive. A segment's flat values come
 * from its provoking vertex, so under the first-vertex mode each vertex
 * carries its own position and counter, and under the last-vertex mode the
 * previous vertex's.
 */
static void
hk_apple9_write_gs_line_stipple(nir_shader *gs,
                                struct hk_apple9_stipple_slots slots)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(gs);
   struct hk_apple9_stipple_tap tap = hk_apple9_tap_position(gs);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *prev =
      nir_local_variable_create(impl, glsl_vec_type(2), "stipple_prev");
   nir_variable *prev_base =
      nir_local_variable_create(impl, glsl_float_type(), "stipple_prev_base");
   nir_variable *have_prev =
      nir_local_variable_create(impl, glsl_bool_type(), "stipple_have_prev");
   nir_store_var(&b, prev, nir_imm_vec2(&b, 0.0, 0.0), 0x3);
   nir_store_var(&b, prev_base, nir_imm_float(&b, 0.0), 1);
   nir_store_var(&b, have_prev, nir_imm_false(&b), 1);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         if (intr->intrinsic == nir_intrinsic_end_primitive ||
             intr->intrinsic == nir_intrinsic_end_primitive_with_counter) {
            nir_store_var(&b, have_prev, nir_imm_false(&b), 1);
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_emit_vertex &&
             intr->intrinsic != nir_intrinsic_emit_vertex_with_counter)
            continue;
         nir_def *window = hk_apple9_stipple_window(&b, tap);
         nir_def *had = nir_load_var(&b, have_prev);
         nir_def *p = nir_load_var(&b, prev);
         nir_def *pb = nir_load_var(&b, prev_base);
         nir_def *d = nir_fabs(&b, nir_fsub(&b, window, p));
         nir_def *base = nir_bcsel(
            &b, had,
            nir_fadd(&b, pb, nir_fmax(&b, nir_channel(&b, d, 0),
                                      nir_channel(&b, d, 1))),
            nir_imm_float(&b, 0.0));
         nir_def *last =
            nir_iand(&b, had, nir_ine_imm(&b, nir_load_provoking_last(&b), 0));
         hk_apple9_store_line_stipple(&b, gs, slots,
                                      nir_bcsel(&b, last, p, window),
                                      nir_bcsel(&b, last, pb, base), window);
         nir_store_var(&b, prev, window, 0x3);
         nir_store_var(&b, prev_base, base, 1);
         nir_store_var(&b, have_prev, nir_imm_true(&b), 1);
      }
   }
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, gs, nir_lower_vars_to_ssa);
}

static void
hk_apple9_lower_line_stipple_fs(nir_shader *fs,
                                struct hk_apple9_stipple_slots slots)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *line = nir_load_input(
      &b, 3, 32, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = slots.start, .num_slots = 1});
   nir_def *start = nir_trim_vector(&b, line, 2);
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_NOPERSPECTIVE);
   nir_def *on_line = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
      .io_semantics = {.location = slots.axis, .num_slots = 1});
   nir_def *axis = nir_fabs(&b, nir_fsub(&b, on_line, start));
   nir_def *x_major =
      nir_fge(&b, nir_channel(&b, axis, 0), nir_channel(&b, axis, 1));
   nir_def *pixel = nir_fadd_imm(
      &b, nir_u2f32(&b, nir_u2u32(&b, nir_load_pixel_coord(&b))), 0.5);
   nir_def *d = nir_fabs(&b, nir_fsub(&b, pixel, start));
   nir_def *counter = nir_f2u32(
      &b, nir_fadd(&b, nir_channel(&b, line, 2),
                   nir_bcsel(&b, x_major, nir_channel(&b, d, 0),
                             nir_channel(&b, d, 1))));
   agx_nir_line_stipple_test(&b, counter);
   nir_progress(true, impl, nir_metadata_control_flow);
   nir_shader_gather_info(fs, impl);
}

/* Generic varying components a shader stores (outputs) or loads (inputs),
 * per slot and component, 64-bit values counting twice. Indirectly addressed
 * arrays count their whole slots. Clip and cull distances are ordinary output
 * components here, as in the API limit.
 */
static unsigned
hk_apple9_generic_components(const nir_shader *s, bool outputs)
{
   uint8_t mask[32] = {0};
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned bits, comps;
            switch (intr->intrinsic) {
            case nir_intrinsic_store_output:
            case nir_intrinsic_store_per_vertex_output:
            case nir_intrinsic_store_per_primitive_output:
               if (!outputs)
                  continue;
               bits = nir_src_bit_size(intr->src[0]);
               comps = nir_intrinsic_write_mask(intr);
               break;
            case nir_intrinsic_load_input:
            case nir_intrinsic_load_interpolated_input:
            case nir_intrinsic_load_per_vertex_input:
            case nir_intrinsic_load_per_primitive_input:
               if (outputs)
                  continue;
               bits = intr->def.bit_size;
               comps = BITFIELD_MASK(intr->def.num_components);
               break;
            default:
               continue;
            }
            nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            if (sem.location < VARYING_SLOT_VAR0 ||
                sem.location >= VARYING_SLOT_VAR0 + 32)
               continue;
            unsigned slot = sem.location - VARYING_SLOT_VAR0;
            uint32_t m = comps << nir_intrinsic_component(intr);
            if (bits == 64) {
               uint32_t wide = 0;
               u_foreach_bit(c, m)
                  wide |= 3u << (2 * c);
               m = wide;
            }
            if (sem.num_slots > 1) {
               for (unsigned i = 0; i < sem.num_slots && slot + i < 32; ++i)
                  mask[slot + i] = 0xf;
            } else {
               for (unsigned i = 0; m && slot + i < 32; ++i, m >>= 4)
                  mask[slot + i] |= m & 0xf;
            }
         }
      }
   }
   unsigned total = 0;
   for (unsigned i = 0; i < 32; ++i)
      total += util_bitcount(mask[i]);
   if (outputs) {
      total += s->info.clip_distance_array_size +
               s->info.cull_distance_array_size;
   }
   return total;
}

/* The stipple slots: the highest two generic slots neither stage uses,
 * provided the four stipple components fit beside both the last stage's
 * outputs and the fragment shader's inputs within the API output limit.
 * Otherwise zero: the line is drawn unstippled, a known limitation.
 */
struct hk_apple9_stipple_slots
hk_apple9_line_stipple_slots(const nir_shader *last, const nir_shader *fs)
{
   struct hk_apple9_stipple_slots slots = {0};
   unsigned used_components = MAX2(hk_apple9_generic_components(last, true),
                                   hk_apple9_generic_components(fs, false));
   if (used_components + HK_APPLE9_STIPPLE_COMPONENTS >
       HK_APPLE9_OUTPUT_COMPONENTS)
      return slots;
   uint64_t used = last->info.outputs_written | fs->info.inputs_read;
   for (int k = 31; k >= 0 && !slots.axis; --k) {
      unsigned slot = VARYING_SLOT_VAR0 + k;
      if (used & BITFIELD64_BIT(slot))
         continue;
      if (slots.start)
         slots.axis = slot;
      else
         slots.start = slot;
   }
   if (!slots.axis)
      slots.start = 0;
   return slots;
}

/*
 * Polygon points are rasterized as points: their hardware facing is always
 * front and their hardware primitive ID counts points. When the fragment
 * shader reads either, a flat uvec2 slot carries the polygon's (back-facing,
 * primitive ID). Vertex and evaluation shaders write (0, ~0): front, and the
 * rasterizer's own primitive ID; the software polygon stage writes the
 * polygon's values.
 */
static bool
hk_apple9_lower_polygon_slot(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   unsigned slot = *(const unsigned *)data;
   bool face = intr->intrinsic == nir_intrinsic_load_front_face;
   bool prim = intr->intrinsic == nir_intrinsic_load_input &&
               nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PRIMITIVE_ID;
   if (!face && !prim)
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *v = nir_load_input(
      b, 2, 32, nir_imm_int(b, 0), .dest_type = nir_type_uint32,
      .io_semantics = {.location = slot, .num_slots = 1});
   if (face) {
      nir_def_replace(&intr->def, nir_ieq_imm(b, nir_channel(b, v, 0), 0));
   } else {
      nir_def *id = nir_channel(b, v, 1);
      nir_def *sel = nir_bcsel(b, nir_ieq_imm(b, id, ~0u), &intr->def, id);
      nir_def_rewrite_uses_after(&intr->def, sel);
   }
   return true;
}

static void
hk_apple9_write_polygon_slot(nir_shader *s, unsigned slot)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_imm_ivec2(&b, 0, ~0), nir_imm_int(&b, 0),
                    .write_mask = 0x3, .src_type = nir_type_uint32,
                    .io_semantics = {.location = slot, .num_slots = 1});
   s->info.outputs_written |= BITFIELD64_BIT(slot);
   nir_progress(true, impl, nir_metadata_control_flow);
}

static void
hk_apple9_write_primitive_id(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_load_primitive_id(&b), nir_imm_int(&b, 0),
                    .write_mask = 0x1, .src_type = nir_type_uint32,
                    .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_ID,
                                     .num_slots = 1});
   s->info.outputs_written |= VARYING_BIT_PRIMITIVE_ID;
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* Decide whether a pipeline's polygon-mode points use the software stage:
 * its rasterized points publish the stage's outputs, clip and cull distances
 * included, the polygon slot when the fragment shader reads facing or the
 * primitive ID, point size and any layer or viewport word, within the
 * varying bank. Returns the polygon slot through face_slot (0 if unread).
 */
static bool
hk_apple9_polygon_point_budget(const nir_shader *last, const nir_shader *fs,
                               uint64_t reserved, unsigned *face_slot)
{
   *face_slot = 0;
   unsigned generic = MAX2(hk_apple9_generic_components(last, true),
                           hk_apple9_generic_components(fs, false));
   bool reads_face =
      BITSET_TEST(fs->info.system_values_read, SYSTEM_VALUE_FRONT_FACE) ||
      (fs->info.inputs_read & VARYING_BIT_PRIMITIVE_ID);
   /* The vertex shader rasterizing mesh records always writes the layer. */
   bool layer = (last->info.outputs_written &
                 (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT)) ||
                last->info.stage == MESA_SHADER_MESH;
   if (generic + 2 * reads_face + 1 + layer > AGX_APPLE9_MAX_VARYING_COMPONENTS)
      return false;
   if (reads_face) {
      uint64_t used = last->info.outputs_written | fs->info.inputs_read | reserved;
      for (int k = 31; k >= 0 && !*face_slot; --k) {
         if (!(used & BITFIELD64_BIT(VARYING_SLOT_VAR0 + k)))
            *face_slot = VARYING_SLOT_VAR0 + k;
      }
      if (!*face_slot)
         return false;
   }
   return true;
}

/* Whether a draw with this depth/stencil state can write the depth or
 * stencil aspects present. A depth/stencil feedback loop needs late tests
 * only then: an early test that writes nothing leaves the attachment the
 * fragment shader reads unchanged (turnip applies the same condition). */
bool
hk_apple9_zs_writes(const struct vk_depth_stencil_state *ds,
                    VkImageAspectFlags aspects)
{
   if (!ds)
      return true;
   struct vk_depth_stencil_state opt = *ds;
   opt.stencil.write_enable = true;
   vk_optimize_depth_stencil_state(&opt, aspects, true);
   return opt.depth.write_enable || opt.stencil.write_enable;
}

static void
hk_apple9_graphics_key(struct hk_apple9_graphics_key *key,
                       const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));
   key->vertex.draw_params_ubo = HK_APPLE9_DRAW_PARAMS_BINDING;
   key->vertex.buffer_descriptor_ubo = HK_APPLE9_VERTEX_BUFFERS_BINDING;
   /* Strides are always read from the bound buffer table, which supports
    * VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE without recompilation. */
   key->vertex.dynamic_stride = true;
   /* Vulkan enables every clip distance the shader writes. */
   key->vertex.clip_distance_enable = 0xff;
   key->blend[0].samples = 1;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->color_map); ++rt)
      key->color_map[rt] = rt;
   if (!state)
      return;
   if (state->cal) {
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->color_map); ++rt)
         key->color_map[rt] = state->cal->color_map[rt];
   }
   key->clip_negative_one_to_one =
      state->vp && state->vp->depth_clip_negative_one_to_one;
   key->sample_shading = state->ms && state->ms->sample_shading_enable;
   /* Draw-time resolution passes the flag only for loops the draw can
    * write; a pipeline compiled with static state applies the test here. */
   key->zs_self_dep = (state->pipeline_flags &
                       VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) &&
      hk_apple9_zs_writes(state->ds,
         (state->rp && state->rp->depth_attachment_format != VK_FORMAT_UNDEFINED
             ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
         (state->rp && state->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED
             ? VK_IMAGE_ASPECT_STENCIL_BIT : 0));
   key->line_stipple = state->rs && state->rs->line.stipple.enable;
   key->polygon_points = state->rs &&
                         state->rs->polygon_mode == VK_POLYGON_MODE_POINT;
   if (state->ia) {
      enum mesa_prim prim =
         u_reduced_prim(vk_topology_to_mesa(state->ia->primitive_topology));
      key->polygon_points &= prim != MESA_PRIM_POINTS && prim != MESA_PRIM_LINES;
   }

   if (state->ia) {
      key->vertex.rasterize_points =
         state->ia->primitive_topology == MESA_PRIM_POINTS;
      key->vertex.ignore_point_size = !key->vertex.rasterize_points;
   }
   if (state->vi) {
      for (unsigned a = 0; a < 16; ++a) {
         if (!(state->vi->attributes_valid & BITFIELD_BIT(a)))
            continue;
         const struct vk_vertex_attribute_state *attr =
            &state->vi->attributes[a];
         const struct vk_vertex_binding_state *binding =
            &state->vi->bindings[attr->binding];
         key->vertex.format[a] = vk_format_to_pipe_format(attr->format);
         key->vertex.offset[a] = attr->offset;
         key->vertex.buffer[a] = attr->binding;
         /* A zero instance divisor repeats the first instance's element. */
         key->vertex.divisor[a] =
            binding->input_rate != VK_VERTEX_INPUT_RATE_INSTANCE ? 0
            : binding->divisor ? binding->divisor
                               : AGX_APPLE9_VERTEX_DIVISOR_ZERO;
      }
   }
   key->targets = state->rp ? state->rp->color_attachment_count : 0;
   for (unsigned rt = 0; rt < MAX2(MIN2(key->targets, 8), 1); ++rt) {
      struct agx_apple9_blend *blend = &key->blend[rt];
      blend->format = rt < key->targets
         ? vk_format_to_pipe_format(state->rp->color_attachment_formats[rt])
         : PIPE_FORMAT_NONE;
      blend->color_attachment_unused = blend->format == PIPE_FORMAT_NONE;
      blend->sample_mask_single_sample = true;
      blend->dynamic_sample_positions = true;
      /* A shader-triggered early test (memory writes, discard or depth/stencil
       * writes with early fragment tests) tests the rasterizer's depth, which
       * the hardware clamps to [0, 1]. Clipping keeps it within the viewport's
       * depth range unless depth clamp is enabled, so only then is a clamped
       * depth export needed. The export tests the shader's interpolated
       * frag_coord.z, which differs from the rasterizer's depth by a few ULPs:
       * an EQUAL pass after a depth prepass without such a shader would fail
       * (Zink EEVEE's G-buffer pass). Draw-time variants see the current
       * dynamic state; pipelines with dynamic depth clamp keep the export. */
      blend->clamp_early_depth =
         !state->rs || state->rs->depth_clamp_enable ||
         BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_CLAMP_ENABLE);
      blend->early_depth_clamp_ubo = 0;
      blend->early_depth_clamp_offset_B =
         hk_root_descriptor_offset(draw.depth_clamp);
      blend->samples = state->ms ? state->ms->rasterization_samples : 1;
      blend->rgb_src = blend->alpha_src = PIPE_BLENDFACTOR_ONE;
      blend->rgb_dst = blend->alpha_dst = PIPE_BLENDFACTOR_ZERO;
      blend->rgb_func = blend->alpha_func = PIPE_BLEND_ADD;
      if (state->ms) {
         blend->disabled_samples = ~state->ms->sample_mask;
         blend->alpha_to_coverage = state->ms->alpha_to_coverage_enable;
         blend->alpha_to_one = state->ms->alpha_to_one_enable;
      }
      if (state->cb && rt < state->cb->attachment_count) {
         const struct vk_color_blend_attachment_state *cb =
            &state->cb->attachments[rt];
         blend->dynamic_enable = BITSET_TEST(
            state->dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
         blend->colormask = (blend->dynamic_enable ||
                             (state->cb->color_write_enables & BITFIELD_BIT(rt)))
                               ? cb->write_mask
                               : 0;
         blend->logicop_enable = state->cb->logic_op_enable;
         blend->logicop_func = vk_logic_op_to_pipe(state->cb->logic_op);
         if (cb->blend_enable && cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
            /* Advanced equations are blended in the shader from the tile. */
            blend->advanced_mode =
               vk_advanced_blend_op_to_pipe(cb->color_blend_op);
            blend->advanced_overlap =
               vk_blend_overlap_to_pipe(cb->blend_overlap);
            blend->src_premultiplied = cb->src_premultiplied;
            blend->dst_premultiplied = cb->dst_premultiplied;
         } else if (cb->blend_enable) {
            blend->rgb_src =
               vk_blend_factor_to_pipe(cb->src_color_blend_factor);
            blend->rgb_dst =
               vk_blend_factor_to_pipe(cb->dst_color_blend_factor);
            blend->alpha_src =
               vk_blend_factor_to_pipe(cb->src_alpha_blend_factor);
            blend->alpha_dst =
               vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor);
            blend->rgb_func = vk_blend_op_to_pipe(cb->color_blend_op);
            blend->alpha_func = vk_blend_op_to_pipe(cb->alpha_blend_op);
         }
      }
   }
}

static bool
hk_apple9_partial_stages(VkShaderStageFlags stages)
{
   return !(stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
          !(stages & (VK_SHADER_STAGE_VERTEX_BIT |
                      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                      VK_SHADER_STAGE_GEOMETRY_BIT |
                      VK_SHADER_STAGE_MESH_BIT_EXT));
}

/* Dynamic states read by the Apple9 graphics key. */
static const enum mesa_vk_dynamic_graphics_state hk_apple9_keyed_dynamic[] = {
   MESA_VK_DYNAMIC_VI,
   MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES,
   MESA_VK_DYNAMIC_MS_SAMPLE_MASK,
   MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE,
   MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE,
   MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE,
   MESA_VK_DYNAMIC_CB_LOGIC_OP,
   MESA_VK_DYNAMIC_CB_ATTACHMENT_COUNT,
   MESA_VK_DYNAMIC_CB_BLEND_ENABLES,
   MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS,
   MESA_VK_DYNAMIC_CB_WRITE_MASKS,
   MESA_VK_DYNAMIC_VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE,
   MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP,
   MESA_VK_DYNAMIC_RS_LINE_STIPPLE_ENABLE,
   MESA_VK_DYNAMIC_RS_POLYGON_MODE,
};

/* With unused attachments, a render's tile layout can differ from the one
 * pipeline formats imply: a null view occupies one word whatever the
 * pipeline format, and an attachment the pipeline leaves undefined occupies
 * its view's size. Such fragment shaders take their layout from the render.
 */
static bool
hk_apple9_color_layout_varies(const struct vk_graphics_pipeline_state *state,
                              const struct vk_features *features)
{
   if (!state || !state->rp || !features ||
       !features->dynamicRenderingUnusedAttachments)
      return false;
   for (unsigned rt = 0; rt < state->rp->color_attachment_count; ++rt) {
      VkFormat format = state->rp->color_attachment_formats[rt];
      if (format == VK_FORMAT_UNDEFINED ||
          agx_apple9_color_words(vk_format_to_pipe_format(format)) != 1)
         return true;
   }
   return false;
}

static void
hk_apple9_graphics_key_device(struct hk_apple9_graphics_key *key,
                               const struct vk_graphics_pipeline_state *state,
                               const struct hk_physical_device *pdev)
{
   hk_apple9_graphics_key(key, state);
   key->native_index = hk_apple9_native_index_enabled(pdev);
   key->native_stages = hk_apple9_native_stage_policy(pdev);
}

static void
hk_hash_graphics_state(struct vk_physical_device *device,
                       const struct vk_graphics_pipeline_state *state,
                       const struct vk_features *features,
                       VkShaderStageFlags stages, blake3_hash blake3_out)
{
   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);
   struct hk_physical_device *pdev = container_of(device, struct hk_physical_device, vk);
   if (pdev->dev.chip == AGX_CHIP_G16G && state) {
      struct hk_apple9_graphics_key key;
      hk_apple9_graphics_key_device(&key, state, pdev);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));

      /* Deferred shaders differ from precompiled ones for the same key. */
      uint32_t deferral = (!state->vi << 0) | (!state->rp << 1) |
                          (!state->ms << 2) | (!state->cb << 3) |
                          (!!(state->pipeline_flags &
                              VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) << 31) |
                          (hk_apple9_partial_stages(stages) << 30) |
                          (hk_apple9_color_layout_varies(state, features) << 29);
      for (unsigned i = 0; i < ARRAY_SIZE(hk_apple9_keyed_dynamic); ++i) {
         if (BITSET_TEST(state->dynamic, hk_apple9_keyed_dynamic[i]))
            deferral |= BITFIELD_BIT(4 + i);
      }
      _mesa_blake3_update(&blake3_ctx, &deferral, sizeof(deferral));
   }
   if (state && (stages & VK_SHADER_STAGE_VERTEX_BIT)) {
      struct hk_vs_key key;
      hk_populate_vs_key(&key, state);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   } else if (state && (stages & VK_SHADER_STAGE_FRAGMENT_BIT)) {
      struct hk_fs_key key;
      hk_populate_fs_key(&key, state);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   }

   if (state &&
       (stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT))) {
      const bool is_multiview = state->mv->view_mask != 0;
      _mesa_blake3_update(&blake3_ctx, &is_multiview, sizeof(is_multiview));
   }

   if (pdev->dev.chip == AGX_CHIP_G16G &&
       (stages & (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) &&
       hk_apple9_native_mesh_available(pdev))
      hk_hash_apple9_native_mesh_abi(&blake3_ctx);

   enum hk_feature_key feature_key = hk_make_feature_key(features);
   _mesa_blake3_update(&blake3_ctx, &feature_key, sizeof(feature_key));

   _mesa_blake3_final(&blake3_ctx, blake3_out);
}

/* Apple9 has no bounds_agx; it selects with a generic comparison. */
static nir_def *
bounds_check(nir_builder *b, nir_def *data, nir_def *offs, nir_def *bound,
             bool apple9)
{
   if (!apple9 && data->bit_size == 32 && data->num_components == 1) {
      return nir_bounds_agx(b, data, offs, bound);
   } else {
      /* TODO: Optimize */
      return nir_bcsel(b, nir_uge(b, bound, offs), data,
                       nir_imm_zero(b, data->num_components, data->bit_size));
   }
}

static bool
is_op(nir_scalar s, nir_op op)
{
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == op;
}

/* Whether the first load_size bytes of a bounded access are in bounds. */
static nir_def *
check_in_bounds_size(nir_builder *b, nir_intrinsic_instr *intr,
                     unsigned load_size)
{
   nir_def *offset = intr->src[1].ssa;
   nir_def *bound = intr->src[2].ssa;

   /* Try to bounds check in terms of elements */
   nir_scalar offset_s = nir_scalar_resolved(offset, 0);
   if (is_op(offset_s, nir_op_amul)) {
      nir_scalar srcs[] = {nir_scalar_chase_alu_src(offset_s, 0),
                           nir_scalar_chase_alu_src(offset_s, 1)};
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      if (nir_scalar_is_const(srcs[1 - i]) &&
          nir_scalar_as_uint(srcs[1 - i]) == load_size) {

         nir_def *index = nir_mov_scalar(b, srcs[i]);
         return nir_ult(b, index, nir_udiv_imm(b, bound, load_size));
      }
   }

   /* TODO: handle also the iadd(amul) pattern, this is important */

   /* Otherwise bounds check in bytes */
   nir_def *sat_offset = nir_umin_imm(b, offset, UINT32_MAX - (load_size - 1));
   return nir_ult(b, nir_iadd_imm(b, sat_offset, load_size - 1), bound);
}

static nir_def *
check_in_bounds(nir_builder *b, nir_intrinsic_instr *intr)
{
   unsigned bit_size = intr->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   return check_in_bounds_size(b, intr, bit_size / 8 * intr->num_components);
}

/* An amul whose scale is not a shift of 0..4. */
static bool
hk_unencodable_scale(nir_scalar s)
{
   if (!is_op(s, nir_op_amul))
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar k = nir_scalar_chase_alu_src(s, i);
      if (nir_scalar_is_const(k))
         return !util_is_power_of_two_nonzero(nir_scalar_as_uint(k)) ||
                nir_scalar_as_uint(k) > 16;
   }
   return true;
}

static nir_def *
bound_offset(nir_builder *b, nir_def *valid, nir_scalar offset,
             bool shift_only_scale)
{
   /* If we can, clamp the source of an amul instead of the result, to remain
    * compatible with the hardware address mode.
    */
   if (is_op(offset, nir_op_amul)) {
      nir_scalar srcs[] = {
         nir_scalar_chase_alu_src(offset, 0),
         nir_scalar_chase_alu_src(offset, 1),
      };
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      /* Apple9 scales an index only by a shift of 0..4. Any other scale
       * would multiply again after the select; selecting the product, which
       * the bounds check already computed, saves that multiply. */
      if (shift_only_scale &&
          !(nir_scalar_is_const(srcs[1 - i]) &&
            util_is_power_of_two_nonzero(nir_scalar_as_uint(srcs[1 - i])) &&
            nir_scalar_as_uint(srcs[1 - i]) <= 16))
         return nir_bcsel(b, valid, nir_mov_scalar(b, offset), nir_imm_int(b, 0));
      nir_def *x = nir_mov_scalar(b, srcs[i]);
      nir_def *y = nir_mov_scalar(b, srcs[1 - i]);

      return nir_amul(b, nir_bcsel(b, valid, x, nir_imm_int(b, 0)), y);
   }

   /* Similar case for when there's an addition (chain) in the way */
   if (is_op(offset, nir_op_iadd)) {
      nir_scalar x = nir_scalar_chase_alu_src(offset, 0);
      nir_scalar y = nir_scalar_chase_alu_src(offset, 1);

      /* With an Apple9 scale the index cannot encode, select the complete
       * offset the bounds check computed rather than each term. */
      if (shift_only_scale &&
          (hk_unencodable_scale(x) || hk_unencodable_scale(y)))
         return nir_bcsel(b, valid, nir_mov_scalar(b, offset), nir_imm_int(b, 0));

      if (is_op(x, nir_op_amul) || is_op(y, nir_op_amul)) {
         return nir_iadd(b, bound_offset(b, valid, x, shift_only_scale),
                         bound_offset(b, valid, y, shift_only_scale));
      }
   }

   nir_def *def = nir_mov_scalar(b, offset);

   /* If the offset fits within the zero page, clamping is pointless */
   if (nir_scalar_is_const(offset) &&
       (nir_scalar_as_uint(offset) + 16) < AGX_ZERO_PAGE_SIZE)
      return def;

   /* Otherwise, fallback on clamping the offset */
   return nir_bcsel(b, valid, def, nir_imm_int(b, 0));
}

static void
lower_load_global_bounded(nir_builder *b, nir_intrinsic_instr *intr,
                          bool per_component_bounds)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = intr->src[0].ssa;
   nir_def *offset = intr->src[1].ssa;

   /* A vector merged from separate accesses (bounds_checked_modes) keeps
    * each component's own check: robustness2 returns in-bounds components
    * even when a later one is out of bounds. Address the load by its first
    * component; the vectorizer only merges naturally aligned accesses, so
    * the read stays within one aligned block of the same page. */
   unsigned byte_size = intr->def.bit_size / 8;
   bool per_component = per_component_bounds && intr->def.num_components > 1;

   /* Bounds check compatibly with the hardware address mode */
   nir_def *valid = per_component ? check_in_bounds_size(b, intr, byte_size)
                                  : check_in_bounds(b, intr);
   base = nir_bcsel(b, valid, base, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   offset = bound_offset(b, valid, nir_scalar_resolved(offset, 0),
                         per_component_bounds);

   enum gl_access_qualifier access = nir_intrinsic_access(intr);

   if (intr->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      access |= ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   }

   nir_def *val = nir_load_global(
      b, intr->def.num_components, intr->def.bit_size,
      nir_iadd(b, base, nir_u2u64(b, offset)),
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr), .access = access);

   if (per_component) {
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      comps[0] = nir_channel(b, val, 0);
      for (unsigned c = 1; c < intr->def.num_components; ++c) {
         comps[c] = nir_bcsel(b, check_in_bounds_size(b, intr, (c + 1) * byte_size),
                              nir_channel(b, val, c),
                              nir_imm_zero(b, 1, intr->def.bit_size));
      }
      val = nir_vec(b, comps, intr->def.num_components);
   }

   nir_def_replace(&intr->def, val);
}

/* Merging loads issues the combined load at the first one, so every later
 * component stays live from there. Apple9 shaders that already reach the
 * register limit then need more copies and swaps, so only merge accesses that
 * are close together (HK_APPLE9_VECTORIZE_WINDOW).
 */
/* Index of the first use in the load's own block, or -1 if a use is in
 * another block or not an instruction. The vectorizer numbers each block's
 * instructions in order before calling back. */
static int64_t
hk_first_use_index(nir_intrinsic_instr *load)
{
   int64_t first = INT64_MAX;
   nir_foreach_use_including_if(use, &load->def) {
      if (nir_src_is_if(use))
         return -1;
      nir_instr *user = nir_src_use_instr(use);
      if (user->block != load->instr.block)
         return -1;
      first = MIN2(first, (int64_t)user->index);
   }
   return first == INT64_MAX ? -1 : first;
}

/* Windowed merging joins loads at most this many instructions apart. */
#define HK_APPLE9_VECTORIZE_WINDOW 4
/* Robust fields merge only when their first uses are this close. */
#define HK_APPLE9_VECTORIZE_USE_WINDOW 2
/* Largest merged robust load, in bytes. */
#define HK_APPLE9_ROBUST_MERGE_BYTES 16

/* Peak number of live 32-bit words in a shader, from NIR liveness: a
 * backward walk of each block from its live-out set. 16-bit values count as
 * half a word; constants and undefs are free (immediates).
 */
static unsigned
hk_def_halves(const nir_def *def)
{
   nir_instr *instr = nir_def_instr((nir_def *)def);
   if (instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_undef)
      return 0;
   return def->num_components * MAX2(def->bit_size, 16) / 16;
}

static bool
hk_index_def(nir_def *def, void *data)
{
   ((nir_def **)data)[def->index] = def;
   return true;
}

struct hk_live_walk {
   bool *live;
   unsigned *halves;
};

static bool
hk_add_live_src(nir_src *src, void *data)
{
   struct hk_live_walk *walk = data;
   if (!walk->live[src->ssa->index]) {
      walk->live[src->ssa->index] = true;
      *walk->halves += hk_def_halves(src->ssa);
   }
   return true;
}

static unsigned
hk_apple9_peak_live_words(nir_shader *nir)
{
   unsigned peak = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_live_defs);
      nir_def **defs = calloc(impl->ssa_alloc, sizeof(*defs));
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block)
            nir_foreach_def(instr, hk_index_def, defs);
      }
      bool *live = calloc(impl->ssa_alloc, sizeof(*live));
      nir_foreach_block(block, impl) {
         memset(live, 0, impl->ssa_alloc * sizeof(*live));
         unsigned halves = 0;
         U_SPARSE_BITSET_FOREACH_SET(&block->live_out, i) {
            live[i] = true;
            if (defs[i])
               halves += hk_def_halves(defs[i]);
         }
         peak = MAX2(peak, halves);
         nir_foreach_instr_reverse(instr, block) {
            if (instr->type == nir_instr_type_phi)
               break;
            nir_def *def = nir_instr_def(instr);
            if (def && live[def->index]) {
               live[def->index] = false;
               halves -= hk_def_halves(def);
            }
            struct hk_live_walk walk = {live, &halves};
            nir_foreach_src(instr, hk_add_live_src, &walk);
            peak = MAX2(peak, halves);
         }
      }
      free(live);
      free(defs);
   }
   return DIV_ROUND_UP(peak, 2);
}

/* The issue and first-use windows in hk_apple9_vectorize_cb protect shaders
 * near the register limit, where a merged load keeps later fields live from
 * the first one's issue. Across DXVK they also leave most robust fields of
 * a vector or matrix as separate loads with their own bounds checks and
 * waits. So apply them only when the shader's estimated peak of live words
 * exceeds HK_APPLE9_VECTORIZE_PRESSURE_BUDGET.
 */
struct hk_vectorize_policy {
   bool windowed;
};

static bool hk_apple9_vectorize_cb(unsigned align_mul, unsigned align_offset,
                                   unsigned bit_size, unsigned num_components,
                                   int64_t hole_size, nir_intrinsic_instr *low,
                                   nir_intrinsic_instr *high, void *data);

/* Peak of live words after vectorizing a copy of the shader with or
 * without the windows. */
static unsigned
hk_apple9_vectorized_peak(const nir_shader *nir, bool windowed)
{
   nir_shader *clone = nir_shader_clone(NULL, nir);
   struct hk_vectorize_policy policy = {.windowed = windowed};
   nir_opt_load_store_vectorize(clone,
                                &(const nir_load_store_vectorize_options){
                                   .modes = nir_var_mem_global,
                                   .bounds_checked_modes = nir_var_mem_global,
                                   .callback = hk_apple9_vectorize_cb,
                                   .cb_data = &policy,
                                });
   unsigned peak = hk_apple9_peak_live_words(clone);
   ralloc_free(clone);
   return peak;
}

/* Below this peak of live 32-bit words, loads merge without windows. */
#define HK_APPLE9_VECTORIZE_PRESSURE_BUDGET 96

static bool
hk_apple9_vectorize_cb(unsigned align_mul, unsigned align_offset,
                       unsigned bit_size, unsigned num_components,
                       int64_t hole_size, nir_intrinsic_instr *low,
                       nir_intrinsic_instr *high, void *data)
{
   if (!agx_mem_vectorize_cb(align_mul, align_offset, bit_size, num_components,
                             hole_size, low, high, NULL))
      return false;
   const struct hk_vectorize_policy *policy = data;
   int64_t window = policy->windowed ? HK_APPLE9_VECTORIZE_WINDOW : INT64_MAX;
   int64_t distance = (int64_t)high->instr.index - (int64_t)low->instr.index;
   if (llabs(distance) > window)
      return false;
   /* A merged load issues every field where the first one was. In shaders
    * near the register limit that keeps later fields live much longer, so
    * only merge robust fields whose first uses are close as well. */
   int64_t use_window = HK_APPLE9_VECTORIZE_USE_WINDOW;
   bool bounded = low->intrinsic == nir_intrinsic_load_global_bounded ||
                  low->intrinsic == nir_intrinsic_load_global_constant_bounded;
   /* Robust merges of fields whose first uses are adjacent shorten Lethal
    * Company's light loops without extra spills at the register limit. */
   int64_t robust_bytes = HK_APPLE9_ROBUST_MERGE_BYTES;
   if (bounded && num_components * bit_size / 8 > robust_bytes)
      return false;
   if (bounded && policy->windowed) {
      int64_t low_use = hk_first_use_index(low), high_use = hk_first_use_index(high);
      if (low_use < 0 || high_use < 0 || llabs(high_use - low_use) > use_window)
         return false;
   }

   /* A merged bounded load may read past the end of its buffer when only
    * its first components are in bounds. Natural alignment keeps that read
    * inside one aligned block, so it cannot cross into another page. */
   if (bounded) {
      unsigned bytes = util_next_power_of_two(num_components * bit_size / 8);
      if (hole_size > 0 || align_mul < bytes || align_offset % bytes)
         return false;
   }
   return true;
}

struct hk_bounded_lowering {
   bool soft_fault;
   bool apple9;
   /* Merged vector loads are checked per component (bounds_checked_modes). */
   bool per_component;
};

static bool
lower_load_global_constant_offset_instr(nir_builder *b,
                                        nir_intrinsic_instr *intrin, void *data)
{
   const struct hk_bounded_lowering *opts = data;
   const bool *has_soft_fault = &opts->soft_fault;

   if (intrin->intrinsic == nir_intrinsic_load_global_bounded ||
       (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded &&
        !(*has_soft_fault))) {

      lower_load_global_bounded(b, intrin, opts->per_component);
      return true;
   }

   if (intrin->intrinsic != nir_intrinsic_load_global_constant_offset &&
       intrin->intrinsic != nir_intrinsic_load_global_constant_bounded)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *base_addr = intrin->src[0].ssa;
   nir_def *offset = intrin->src[1].ssa;
   nir_def *bound = NULL;

   unsigned bit_size = intrin->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   unsigned byte_size = bit_size / 8;
   unsigned load_size = byte_size * intrin->num_components;

   if (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      bound = intrin->src[2].ssa;
   }

   unsigned align_mul = nir_intrinsic_align_mul(intrin);
   unsigned align_offset = nir_intrinsic_align_offset(intrin);

   /* Without robustness an access outside the buffer is undefined, so the
    * load itself is in bounds. The Apple9 backend relies on this to fold
    * constant offsets past an amul: the hardware adds the displacement in
    * 64 bits, which differs from the 32-bit offset only outside the buffer.
    */
   enum gl_access_qualifier access = nir_intrinsic_access(intrin);
   if (intrin->intrinsic == nir_intrinsic_load_global_constant_offset)
      access |= ACCESS_IN_BOUNDS;

   nir_def *val = nir_load_global_constant(
      b, intrin->def.num_components, intrin->def.bit_size,
      nir_iadd(b, base_addr, nir_u2u64(b, offset)), .align_mul = align_mul,
      .align_offset = align_offset, .access = access);

   if (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded &&
       opts->per_component && intrin->def.num_components > 1) {
      /* A vector merged from separate robust fields (bounds_checked_modes)
       * returns each in-bounds component, as in lower_load_global_bounded:
       * checking only its last byte would zero the in-bounds fields too. A
       * field split over two merged loads then reads (x, y, 0, 0) where
       * robustness2 allows only (x, y, z, 0) or zero (uniform_buffer
       * len_12/31/252).
       */
      nir_scalar offs = nir_scalar_resolved(offset, 0);
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned c = 0; c < intrin->def.num_components; ++c) {
         nir_def *end = nir_scalar_is_const(offs)
            ? nir_imm_int(b, nir_scalar_as_uint(offs) + (c + 1) * byte_size)
            : nir_iadd_imm(b, offset, (c + 1) * byte_size);
         comps[c] = bounds_check(b, nir_channel(b, val, c), end, bound,
                                 opts->apple9);
      }
      val = nir_vec(b, comps, intrin->def.num_components);
   } else if (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      nir_scalar offs = nir_scalar_resolved(offset, 0);
      if (nir_scalar_is_const(offs)) {
         /* Calculate last byte loaded */
         unsigned offs_imm = nir_scalar_as_uint(offs) + load_size;

         /* Compare the last byte loaded against the bound exactly. Bounds
          * are the descriptor's exact range, not rounded to 64 B, so rounding
          * the offset down would let loads past a short range through
          * (robustness2 uniform_buffer len_4..len_60).
          */
         val = bounds_check(b, val, nir_imm_int(b, offs_imm), bound,
                            opts->apple9);
      } else {
         offset = nir_iadd_imm(b, offset, load_size);
         val = bounds_check(b, val, offset, bound, opts->apple9);
      }
   }

   nir_def_replace(&intrin->def, val);
   return true;
}

/* Lower bounded and offset global loads. Apple9 checks each component of a
 * merged vector separately. */
bool
hk_lower_bounded_global_loads(nir_shader *nir, bool soft_fault, bool apple9)
{
   struct hk_bounded_lowering bounded = {
      .soft_fault = soft_fault,
      .apple9 = apple9,
      .per_component = apple9,
   };
   return nir_shader_intrinsics_pass(nir, lower_load_global_constant_offset_instr,
                                     nir_metadata_none, &bounded);
}

struct lower_ycbcr_state {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
};

static const struct vk_ycbcr_conversion_state *
lookup_ycbcr_conversion(const void *_state, uint32_t set, uint32_t binding,
                        uint32_t array_index)
{
   const struct lower_ycbcr_state *state = _state;
   assert(set < state->set_layout_count);
   assert(state->set_layouts[set] != NULL);
   const struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(state->set_layouts[set]);
   assert(binding < set_layout->binding_count);

   const struct hk_descriptor_set_binding_layout *bind_layout =
      &set_layout->binding[binding];

   if (bind_layout->immutable_samplers == NULL)
      return NULL;

   array_index = MIN2(array_index, bind_layout->array_size - 1);

   const struct hk_sampler *sampler =
      bind_layout->immutable_samplers[array_index];

   return sampler && sampler->vk.ycbcr_conversion
             ? &sampler->vk.ycbcr_conversion->state
             : NULL;
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/*
 * This is the world's worst multiview implementation. We simply duplicate each
 * draw on the CPU side, changing a uniform in between, and then plumb the view
 * index into the layer ID here. Whatever, it works.
 *
 * The "proper" implementation on AGX would use vertex amplification, but a
 * MacBook is not a VR headset.
 */
static void
hk_lower_multiview(nir_shader *nir)
{
   /* If there's an existing layer ID write, ignore it. This avoids validation
    * splat with vk_meta.
    */
   nir_variable *existing = nir_find_variable_with_location(
      nir, nir_var_shader_out, VARYING_SLOT_LAYER);

   if (existing) {
      existing->data.mode = nir_var_shader_temp;
      existing->data.location = 0;
      nir_fixup_deref_modes(nir);
   }

   /* Now write the view index as the layer */
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

   nir_variable *layer =
      nir_variable_create(nir, nir_var_shader_out, glsl_uint_type(), NULL);

   layer->data.location = VARYING_SLOT_LAYER;

   nir_store_var(&b, layer, nir_load_view_index(&b), nir_component_mask(1));
   b.shader->info.outputs_written |= VARYING_BIT_LAYER;
}

static nir_def *
query_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_custom_border_color_agx, 4,
                                  tex->dest_type, false, false);
}

static nir_def *
has_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_has_custom_border_color_agx,
                                  1, nir_type_bool1, false, false);
}

static bool
lower(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex))
      return false;

   /* XXX: this is a really weird edge case, is this even well-defined? */
   if (tex->is_shadow)
      return false;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *orig = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, orig);
   nir_def *clamp_to_1 = &nir_instr_as_tex(orig)->def;

   nir_push_if(b, has_custom);
   nir_def *replaced = NULL;
   {
      /* Sample again, this time with clamp-to-0 instead of clamp-to-1 */
      nir_instr *clone_instr = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, clone_instr);

      nir_tex_instr *tex_0 = nir_instr_as_tex(clone_instr);
      nir_def *clamp_to_0 = &tex_0->def;

      tex_0->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;

      /* Grab the border colour */
      nir_def *border = query_custom_border(b, tex_0);

      if (tex->op == nir_texop_tg4) {
         border = nir_replicate(b, nir_channel(b, border, tex->component), 4);
      }

      /* Combine together with the border */
      if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
          tex->op != nir_texop_tg4) {

         /* For floats, lerp together:
          *
          * For border texels:  (1 * border) + (0 * border      ) = border
          * For regular texels: (x * border) + (x * (1 - border)) = x.
          *
          * Linear filtering is linear (duh), so lerping is compatible.
          */
         replaced = nir_flrp(b, clamp_to_0, clamp_to_1, border);
         b->shader->info.flrp_lowered = false;
      } else {
         /* For integers, just select componentwise since there is no linear
          * filtering. Gathers also use this path since they are unfiltered in
          * each component.
          */
         replaced = nir_bcsel(b, nir_ieq(b, clamp_to_0, clamp_to_1), clamp_to_0,
                              border);
      }
   }
   nir_pop_if(b, NULL);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, replaced, clamp_to_1);
   nir_def_replace(&tex->def, phi);
   return true;
}

static bool
agx_nir_lower_custom_border(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower, nir_metadata_none, NULL);
}

static nir_def *
query_min_lod(nir_builder *b, nir_tex_instr *tex, bool int_coords)
{
   nir_alu_type T = int_coords ? nir_type_uint16 : nir_type_float16;
   return nir_build_texture_query(b, tex, nir_texop_image_min_lod_agx, 1, T,
                                  false, false);
}

static bool
lower_min_lod(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const bool *float_fetch = data;
   if (*float_fetch && tex->op == nir_texop_lod &&
       b->shader->info.stage == MESA_SHADER_COMPUTE &&
       b->shader->info.derivative_group == DERIVATIVE_GROUP_LINEAR) {
      /* The compute query consumes the view clamp after sampler/view level
       * clamping, before mip rounding. It is not a sampler minimum LOD. */
      b->cursor = nir_before_instr(&tex->instr);
      nir_tex_instr_add_src(tex, nir_tex_src_min_lod,
                            query_min_lod(b, tex, false));
      return true;
   }
   if (nir_tex_instr_is_query(tex))
      return false;

   /* Buffer textures don't have levels-of-detail */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return false;

   if (tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)
      return false;

   bool int_coords = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms ||
                     tex->op == nir_texop_tg4;

   b->cursor = nir_before_instr(&tex->instr);
   /* Apple9 legalizes the minimum-LOD source as a float, so fetches take
    * the float minimum and compare against its floor. */
   nir_def *min_lod = query_min_lod(b, tex, int_coords && !*float_fetch);
   nir_def *other_min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

   if (tex->op == nir_texop_tg4) {
      b->cursor = nir_after_instr(&tex->instr);

      /* The Vulkan spec section "Texel Gathering" says:
       *
       *    If levelbase < minLodIntegerimageView, then any values fetched are
       *    zero if the robustImageAccess2 feature is enabled.
       *
       * We currently always enable robustImageAccess2, so implement that
       * semantic here.
       *
       * We could probably optimize this with a special descriptor for this case
       * but tg4 is rare enough I'm not bothered.
       */
      nir_def *old = &tex->def;
      nir_def *oob = *float_fetch ? nir_fge_imm(b, min_lod, 1.0)
                                  : nir_ine_imm(b, min_lod, 0);
      nir_def *zero = nir_imm_zero(b, old->num_components, old->bit_size);
      nir_def *new_ = nir_bcsel(b, oob, zero, old);
      nir_def_rewrite_uses_after(old, new_);
   } else if (tex->op == nir_texop_txl) {
      assert(other_min_lod == NULL && "txl doesn't have an API min lod");

      nir_def *lod = nir_steal_tex_src(tex, nir_tex_src_lod);
      if (lod) {
         min_lod = nir_fmax(b, nir_f2fN(b, lod, min_lod->bit_size), min_lod);
      }

      nir_tex_instr_add_src(tex, nir_tex_src_lod, min_lod);
   } else {
      if (other_min_lod) {
         assert(!int_coords && "no API min lod");
         min_lod = nir_fmax(b, min_lod, nir_f2f16(b, other_min_lod));
      }

      nir_tex_instr_add_src(tex, nir_tex_src_min_lod, min_lod);
   }

   return true;
}

static bool
agx_nir_lower_image_view_min_lod(nir_shader *nir, bool float_fetch)
{
   return nir_shader_tex_pass(nir, lower_min_lod, nir_metadata_none,
                              &float_fetch);
}

/*
 * In Vulkan, the VIEWPORT should read 0 in the fragment shader if it is not
 * written by the vertex shader, but in our implementation, the varying would
 * otherwise be undefined. This small pass predicates VIEWPORT reads based on
 * whether the hardware vertex shader writes the VIEWPORT (nonzero UVS index).
 */
static bool
lower_viewport_fs(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_VIEWPORT)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *orig = &intr->def;

   nir_def *uvs = nir_load_uvs_index_agx(b, .io_semantics = sem);
   nir_def *def = nir_bcsel(b, nir_ine_imm(b, uvs, 0), orig, nir_imm_int(b, 0));

   nir_def_rewrite_uses_after(orig, def);
   return true;
}

/* The Apple9 rasterizer delivers the layer to the fragment layer system
 * value. The viewport index is an ordinary flat varying.
 */
static bool
hk_apple9_lower_layer_viewport_fs(nir_builder *b, nir_intrinsic_instr *intr,
                                  UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_LAYER)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *layer = nir_iand_imm(b, nir_load_layer_id(b), 0xffff);
   nir_def_replace(&intr->def, nir_u2uN(b, layer, intr->def.bit_size));
   return true;
}

static bool
lower_subpass_dim(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   else if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   else
      return false;

   return true;
}

static bool
should_lower_robust(const nir_intrinsic_instr *intr, const void *_)
{
   /* The hardware is robust, but our software image atomics are not. Unlike the
    * GL driver, we don't use the common buffer image lowering, using the
    * agx_nir_lower_texture lowering for robustImageAccess2 semantics.
    */
   return intr->intrinsic == nir_intrinsic_image_deref_atomic ||
          intr->intrinsic == nir_intrinsic_image_deref_atomic_swap;
}

/* Uniform texel formats are descriptor state, not shader specialization. The
 * texture unit converts every format but RGB32, which has no native format,
 * exactly as it does for images, so fetch through the view's linear 2D
 * descriptor; RGB32 loads its three words. Bounds and texel counts come from
 * the descriptor's metadata in GPU memory. */
static bool
hk_apple9_lower_texel_buffer(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF ||
       (tex->op != nir_texop_txf && tex->op != nir_texop_txs))
      return false;
   int handle_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle_idx < 0)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *ptr = nir_load_from_texture_handle_agx(b, tex->src[handle_idx].src.ssa);
   ptr = nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_descriptor, metadata));
   nir_def *count = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_metadata, elements)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   if (tex->op == nir_texop_txs) {
      nir_def_replace(&tex->def, count);
      return true;
   }

   int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   bool floating = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
   nir_push_if(b, nir_ult(b, coord, count));
   nir_def *base = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_metadata, base)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *size = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_metadata, size_B)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_push_if(b, nir_ieq_imm(b, size, 12));
   nir_def *xyz = nir_load_global(b, 3, 32,
      nir_iadd(b, base, nir_u2u64(b, nir_imul_imm(b, coord, 12))), .align_mul = 4);
   nir_def *rgb32 = nir_vec4(b, nir_channel(b, xyz, 0), nir_channel(b, xyz, 1),
                             nir_channel(b, xyz, 2),
                             floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1));
   nir_push_else(b, NULL);
   /* The descriptor starts at the 16-byte aligned address at or below the
    * first texel; other texel sizes are powers of two. */
   nir_def *tail = nir_ushr(b, nir_iand_imm(b, nir_u2u32(b, base), 15),
                            nir_ufind_msb(b, size));
   nir_def *index = nir_iadd(b, coord, tail);
   nir_tex_instr *fetch = nir_tex_instr_create(b->shader, 3);
   fetch->op = nir_texop_txf;
   fetch->sampler_dim = GLSL_SAMPLER_DIM_2D;
   fetch->dest_type = nir_alu_type_get_base_type(tex->dest_type) | 32;
   fetch->coord_components = 2;
   fetch->texture_index = tex->texture_index;
   fetch->texture_non_uniform = tex->texture_non_uniform;
   /* The bounds check above covers the fetch, whose level is always zero. */
   fetch->backend_flags = (tex->backend_flags & HK_APPLE9_TEX_DESCRIPTOR_HANDLE) |
                          AGX_APPLE9_TEXTURE_FETCH_BOUNDED;
   fetch->src[0] = (nir_tex_src){.src_type = nir_tex_src_texture_handle,
      .src = nir_src_for_ssa(tex->src[handle_idx].src.ssa)};
   fetch->src[1] = (nir_tex_src){.src_type = nir_tex_src_coord,
      .src = nir_src_for_ssa(nir_vec2(b,
         nir_iand_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH - 1),
         nir_ushr_imm(b, index, util_logbase2(AGX_TEXTURE_BUFFER_WIDTH))))};
   fetch->src[2] = (nir_tex_src){.src_type = nir_tex_src_lod,
      .src = nir_src_for_ssa(nir_imm_int(b, 0))};
   nir_def_init(&fetch->instr, &fetch->def, 4, 32);
   nir_builder_instr_insert(b, &fetch->instr);
   nir_pop_if(b, NULL);
   nir_def *result = nir_trim_vector(b, nir_if_phi(b, rgb32, &fetch->def),
                                     tex->def.num_components);
   nir_push_else(b, NULL);
   /* Out-of-bounds reads return zero, with a one in formats lacking alpha. */
   nir_def *zero = nir_imm_zero(b, tex->def.num_components, 32);
   if (tex->def.num_components == 4) {
      nir_def *alpha_kind = nir_ushr_imm(b, nir_load_global_constant(b, 1, 32,
         nir_iadd_imm(b, ptr, offsetof(struct hk_apple9_texel_metadata, component[3])),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX), 18);
      nir_def *one = floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
      zero = nir_vector_insert_imm(b, zero,
         nir_bcsel(b, nir_ieq_imm(b, alpha_kind, HK_TEXEL_ONE), one,
                   nir_imm_int(b, 0)), 3);
   }
   nir_pop_if(b, NULL);
   nir_def_replace(&tex->def, nir_if_phi(b, result, zero));
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* Loads or stores a texel of 1 to 16 bytes as zero-padded 32-bit words. */
static nir_def *
hk_apple9_load_texel_words(nir_builder *b, nir_def *addr, nir_def *size_B,
                           enum gl_access_qualifier access)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_push_if(b, nir_ult_imm(b, size_B, 4));
   nir_push_if(b, nir_ieq_imm(b, size_B, 1));
   nir_def *r8 = nir_u2u32(b, nir_load_global(b, 1, 8, addr, .align_mul = 1,
                                              .access = access));
   nir_push_else(b, NULL);
   nir_def *r16 = nir_u2u32(b, nir_load_global(b, 1, 16, addr, .align_mul = 2,
                                               .access = access));
   nir_pop_if(b, NULL);
   nir_def *small = nir_vec4(b, nir_if_phi(b, r8, r16), zero, zero, zero);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 4));
   nir_def *r32 = nir_vec4(b, nir_load_global(b, 1, 32, addr, .align_mul = 4,
                                              .access = access),
                           zero, zero, zero);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 8));
   nir_def *r64 = nir_pad_vector_imm_int(
      b, nir_load_global(b, 2, 32, addr, .align_mul = 8, .access = access), 0, 4);
   nir_push_else(b, NULL);
   nir_def *r128 = nir_load_global(b, 4, 32, addr, .align_mul = 16,
                                   .access = access);
   nir_pop_if(b, NULL);
   nir_def *wide = nir_if_phi(b, r64, r128);
   nir_pop_if(b, NULL);
   nir_def *large = nir_if_phi(b, r32, wide);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, small, large);
}

static void
hk_apple9_store_texel_words(nir_builder *b, nir_def *addr, nir_def *size_B,
                            nir_def *words, enum gl_access_qualifier access)
{
   nir_def *w0 = nir_channel(b, words, 0);
   nir_push_if(b, nir_ieq_imm(b, size_B, 1));
   nir_store_global(b, nir_u2u8(b, w0), addr, .align_mul = 1, .access = access);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 2));
   nir_store_global(b, nir_u2u16(b, w0), addr, .align_mul = 2, .access = access);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 4));
   nir_store_global(b, w0, addr, .align_mul = 4, .access = access);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size_B, 8));
   nir_store_global(b, nir_trim_vector(b, words, 2), addr, .align_mul = 8,
                    .write_mask = 0x3, .access = access);
   nir_push_else(b, NULL);
   nir_store_global(b, words, addr, .align_mul = 16, .write_mask = 0xf,
                    .access = access);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static nir_def *
hk_apple9_texel_word(nir_builder *b, nir_def *words, nir_def *index)
{
   nir_def *w = nir_channel(b, words, 0);
   for (unsigned i = 1; i < 4; ++i)
      w = nir_bcsel(b, nir_ieq_imm(b, index, i), nir_channel(b, words, i), w);
   return w;
}

/* Decode one component of a texel whose first component has kind cls.
 * Other components are either the same kind or constant zero/one. */
static nir_def *
hk_apple9_decode_component(nir_builder *b, nir_def *words, nir_def *fmt,
                           unsigned c, unsigned cls, bool floating)
{
   nir_def *kind = nir_ushr_imm(b, fmt, 18);
   nir_def *value;
   if (cls == HK_TEXEL_FLOAT11 || cls == HK_TEXEL_RGB9E5) {
      nir_def *w0 = nir_channel(b, words, 0);
      value = c < 3 ? nir_channel(b, cls == HK_TEXEL_FLOAT11
                                        ? nir_format_unpack_11f11f10f(b, w0)
                                        : nir_format_unpack_r9g9b9e5(b, w0), c)
                    : nir_imm_float(b, 1.0);
   } else {
      nir_def *raw = hk_apple9_texel_word(
         b, words, nir_ushr_imm(b, nir_iand_imm(b, fmt, 0xff), 2));
      if (cls == HK_TEXEL_FLOAT32) {
         value = raw;
      } else {
         nir_def *shift = nir_iand_imm(b, nir_ushr_imm(b, fmt, 8), 31);
         nir_def *width = nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, fmt, 13), 31), 1);
         nir_def *extend = nir_isub(b, nir_imm_int(b, 32), width);
         nir_def *mask = nir_ushr(b, nir_imm_int(b, -1), extend);
         nir_def *u = nir_iand(b, nir_ushr(b, raw, shift), mask);
         nir_def *i = nir_ishr(b, nir_ishl(b, u, extend), extend);
         switch (cls) {
         case HK_TEXEL_SINT: value = i; break;
         case HK_TEXEL_UNORM:
            value = nir_fdiv(b, nir_u2f32(b, u), nir_u2f32(b, mask));
            break;
         case HK_TEXEL_SNORM:
            value = nir_fmax(b, nir_imm_float(b, -1.0),
               nir_fdiv(b, nir_i2f32(b, i), nir_u2f32(b, nir_ushr_imm(b, mask, 1))));
            break;
         case HK_TEXEL_FLOAT16:
            value = nir_channel(b, nir_unpack_half_2x16(b, u), 0);
            break;
         default: value = u; break;
         }
      }
   }
   value = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_ZERO), nir_imm_int(b, 0), value);
   return nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_ONE),
                    floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1), value);
}

static nir_def *hk_apple9_decode_texel_select(nir_builder *b, nir_def *words,
                                              nir_def *layout, bool floating);

/* The format kinds come from the descriptor, so they are normally uniform.
 * Branch on the class of the texel (the first component's kind, with any
 * other component of that kind or constant zero/one) and convert only that
 * format; other layouts fall back to evaluating every kind. */
static nir_def *
hk_apple9_decode_texel_classes(nir_builder *b, nir_def *words, nir_def *layout,
                               bool floating, const unsigned *classes,
                               unsigned count)
{
   if (!count)
      return hk_apple9_decode_texel_select(b, words, layout, floating);
   nir_def *in_class = nir_imm_true(b);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *k = nir_ushr_imm(b, nir_channel(b, layout, c), 18);
      nir_def *ok = nir_ieq_imm(b, k, classes[0]);
      if (c)
         ok = nir_ior(b, ok, nir_ior(b, nir_ieq_imm(b, k, HK_TEXEL_ZERO),
                                     nir_ieq_imm(b, k, HK_TEXEL_ONE)));
      in_class = nir_iand(b, in_class, ok);
   }
   nir_push_if(b, in_class);
   nir_def *rgba[4];
   for (unsigned c = 0; c < 4; ++c)
      rgba[c] = hk_apple9_decode_component(b, words, nir_channel(b, layout, c),
                                           c, classes[0], floating);
   nir_def *then_value = nir_vec(b, rgba, 4);
   nir_push_else(b, NULL);
   nir_def *else_value = hk_apple9_decode_texel_classes(
      b, words, layout, floating, classes + 1, count - 1);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, then_value, else_value);
}

static nir_def *
hk_apple9_decode_texel(nir_builder *b, nir_def *words, nir_def *layout,
                       bool floating)
{
   static const unsigned float_classes[] = {
      HK_TEXEL_UNORM, HK_TEXEL_FLOAT32, HK_TEXEL_FLOAT16, HK_TEXEL_SNORM,
      HK_TEXEL_FLOAT11, HK_TEXEL_RGB9E5,
   };
   static const unsigned int_classes[] = {HK_TEXEL_UINT, HK_TEXEL_SINT};
   return floating
      ? hk_apple9_decode_texel_classes(b, words, layout, true, float_classes,
            ARRAY_SIZE(float_classes))
      : hk_apple9_decode_texel_classes(b, words, layout, false, int_classes,
            ARRAY_SIZE(int_classes));
}

/* Decode a texel from the per-component layout of its descriptor, as
 * written by hk_apple9_texel_layout, evaluating every format kind. */
static nir_def *
hk_apple9_decode_texel_select(nir_builder *b, nir_def *words, nir_def *layout,
                              bool floating)
{
   nir_def *rgba[4];
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *fmt = nir_channel(b, layout, c);
      nir_def *kind = nir_ushr_imm(b, fmt, 18);
      nir_def *raw = hk_apple9_texel_word(
         b, words, nir_ushr_imm(b, nir_iand_imm(b, fmt, 0xff), 2));
      nir_def *shift = nir_iand_imm(b, nir_ushr_imm(b, fmt, 8), 31);
      nir_def *width = nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, fmt, 13), 31), 1);
      nir_def *extend = nir_isub(b, nir_imm_int(b, 32), width);
      nir_def *mask = nir_ushr(b, nir_imm_int(b, -1), extend);
      nir_def *u = nir_iand(b, nir_ushr(b, raw, shift), mask);
      nir_def *i = nir_ishr(b, nir_ishl(b, u, extend), extend);
      nir_def *value;
      if (floating) {
         nir_def *unorm = nir_fdiv(b, nir_u2f32(b, u), nir_u2f32(b, mask));
         nir_def *snorm = nir_fmax(b, nir_imm_float(b, -1.0),
            nir_fdiv(b, nir_i2f32(b, i), nir_u2f32(b, nir_ushr_imm(b, mask, 1))));
         value = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_SNORM), snorm, unorm);
         value = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_FLOAT16),
                           nir_channel(b, nir_unpack_half_2x16(b, u), 0), value);
         value = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_FLOAT32), raw, value);
         if (c < 3) {
            nir_def *w0 = nir_channel(b, words, 0);
            value = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_FLOAT11),
               nir_channel(b, nir_format_unpack_11f11f10f(b, w0), c), value);
            value = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_RGB9E5),
               nir_channel(b, nir_format_unpack_r9g9b9e5(b, w0), c), value);
         }
      } else {
         value = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_SINT), i, u);
      }
      value = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_ZERO), nir_imm_int(b, 0), value);
      rgba[c] = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_ONE),
                          floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1),
                          value);
   }
   return nir_vec(b, rgba, 4);
}

/* The common unsigned-float packer clamps with maximumNumber, which turns
 * NaNs into zero. Classify the original bits and preserve NaNs, including
 * payloads lost by mantissa truncation. Finite and infinity encodings retain
 * the common packer's behavior. */
static nir_def *
hk_apple9_pack_float11(nir_builder *b, nir_def *rgb)
{
   nir_def *packed = nir_format_pack_11f11f10f(b, rgb);
   const unsigned shifts[] = {0, 11, 22};
   const uint32_t masks[] = {0x7ff, 0x7ff, 0x3ff};
   const uint32_t nans[] = {0x7e0, 0x7e0, 0x3f0};
   for (unsigned c = 0; c < 3; ++c) {
      nir_def *is_nan = nir_ult(b, nir_imm_int(b, 0x7f800000),
         nir_iand_imm(b, nir_channel(b, rgb, c), 0x7fffffff));
      nir_def *repaired = nir_ior_imm(b,
         nir_iand_imm(b, packed, ~(masks[c] << shifts[c])),
         nans[c] << shifts[c]);
      packed = nir_bcsel(b, is_nan, repaired, packed);
   }
   return packed;
}

/* Encode a texel into zero-padded 32-bit words for its descriptor layout. */
static nir_def *
hk_apple9_encode_texel_kind(nir_builder *b, nir_def *value, nir_def *layout,
                            bool floating, int cls)
{
   nir_def *words[4] = {
      nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0),
   };
   nir_def *rgb = nir_trim_vector(b, value, 3);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *fmt = nir_channel(b, layout, c);
      nir_def *kind = nir_ushr_imm(b, fmt, 18);
      nir_def *index = nir_ushr_imm(b, nir_iand_imm(b, fmt, 0xff), 2);
      nir_def *shift = nir_iand_imm(b, nir_ushr_imm(b, fmt, 8), 31);
      nir_def *width = nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, fmt, 13), 31), 1);
      nir_def *mask = nir_ushr(b, nir_imm_int(b, -1),
                               nir_isub(b, nir_imm_int(b, 32), width));
      nir_def *v = nir_channel(b, value, c);
      nir_def *bits = v;
      if (floating && cls < 0) {
         nir_def *unorm = nir_f2u32(b, nir_fround_even(b,
            nir_fmul(b, nir_fsat(b, v), nir_u2f32(b, mask))));
         nir_def *snorm = nir_f2i32(b, nir_fround_even(b,
            nir_fmul(b, nir_fmin(b, nir_fmax(b, v, nir_imm_float(b, -1.0)),
                                 nir_imm_float(b, 1.0)),
                     nir_u2f32(b, nir_ushr_imm(b, mask, 1)))));
         bits = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_SNORM), snorm, unorm);
         bits = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_FLOAT16),
                          nir_pack_half_2x16_split(b, v, nir_imm_float(b, 0)), bits);
         bits = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_FLOAT32), v, bits);
      } else if (floating && cls == HK_TEXEL_UNORM) {
         bits = nir_f2u32(b, nir_fround_even(b,
            nir_fmul(b, nir_fsat(b, v), nir_u2f32(b, mask))));
      } else if (floating && cls == HK_TEXEL_SNORM) {
         bits = nir_f2i32(b, nir_fround_even(b,
            nir_fmul(b, nir_fmin(b, nir_fmax(b, v, nir_imm_float(b, -1.0)),
                                 nir_imm_float(b, 1.0)),
                     nir_u2f32(b, nir_ushr_imm(b, mask, 1)))));
      } else if (floating && cls == HK_TEXEL_FLOAT16) {
         bits = nir_pack_half_2x16_split(b, v, nir_imm_float(b, 0));
      }
      bits = nir_ishl(b, nir_iand(b, bits, mask), shift);
      /* Packed float formats encode all three components at once. */
      if (floating && c == 0 && (cls < 0 || cls == HK_TEXEL_FLOAT11 ||
                                  cls == HK_TEXEL_RGB9E5)) {
         if (cls < 0 || cls == HK_TEXEL_FLOAT11)
            bits = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_FLOAT11),
                             hk_apple9_pack_float11(b, rgb), bits);
         if (cls < 0 || cls == HK_TEXEL_RGB9E5)
            bits = nir_bcsel(b, nir_ieq_imm(b, kind, HK_TEXEL_RGB9E5),
                             nir_format_pack_r9g9b9e5(b, rgb), bits);
      } else if (floating && c < 3 && (cls < 0 || cls == HK_TEXEL_FLOAT11 ||
                                       cls == HK_TEXEL_RGB9E5)) {
         bits = nir_bcsel(b, nir_ior(b, nir_ieq_imm(b, kind, HK_TEXEL_FLOAT11),
                                     nir_ieq_imm(b, kind, HK_TEXEL_RGB9E5)),
                          nir_imm_int(b, 0), bits);
      }
      bits = nir_bcsel(b, nir_ior(b, nir_ieq_imm(b, kind, HK_TEXEL_ZERO),
                                  nir_ieq_imm(b, kind, HK_TEXEL_ONE)),
                       nir_imm_int(b, 0), bits);
      for (unsigned w = 0; w < 4; ++w) {
         words[w] = nir_bcsel(b, nir_ieq_imm(b, index, w),
                              nir_ior(b, words[w], bits), words[w]);
      }
   }
   return nir_vec(b, words, 4);
}

static nir_def *
hk_apple9_encode_texel_classes(nir_builder *b, nir_def *value, nir_def *layout,
                               bool floating, const unsigned *classes,
                               unsigned count)
{
   if (!count)
      return hk_apple9_encode_texel_kind(b, value, layout, floating, -1);
   nir_def *in_class = nir_imm_true(b);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *k = nir_ushr_imm(b, nir_channel(b, layout, c), 18);
      nir_def *ok = nir_ieq_imm(b, k, classes[0]);
      if (c)
         ok = nir_ior(b, ok, nir_ior(b, nir_ieq_imm(b, k, HK_TEXEL_ZERO),
                                     nir_ieq_imm(b, k, HK_TEXEL_ONE)));
      in_class = nir_iand(b, in_class, ok);
   }
   nir_push_if(b, in_class);
   nir_def *then_words = hk_apple9_encode_texel_kind(b, value, layout, floating,
                                                     classes[0]);
   nir_push_else(b, NULL);
   nir_def *else_words = hk_apple9_encode_texel_classes(
      b, value, layout, floating, classes + 1, count - 1);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, then_words, else_words);
}

/* As for decoding, branch on the uniform format class of the texel. */
static nir_def *
hk_apple9_encode_texel(nir_builder *b, nir_def *value, nir_def *layout,
                       bool floating)
{
   static const unsigned float_classes[] = {
      HK_TEXEL_UNORM, HK_TEXEL_FLOAT32, HK_TEXEL_FLOAT16, HK_TEXEL_SNORM,
      HK_TEXEL_FLOAT11, HK_TEXEL_RGB9E5,
   };
   return floating
      ? hk_apple9_encode_texel_classes(b, value, layout, true, float_classes,
            ARRAY_SIZE(float_classes))
      : hk_apple9_encode_texel_kind(b, value, layout, false, -1);
}

/* Encode with the constant shifts of a canonical format when the descriptor
 * names one (enum hk_apple9_store_encode), otherwise from the per-component
 * layout. The branches are uniform, and each fast arm computes exactly what
 * hk_apple9_encode_texel computes for those formats.
 */

nir_def *
hk_apple9_encode_texel_fast(nir_builder *b, nir_def *value, nir_def *layout,
                            bool floating, nir_def *encode)
{
   nir_push_if(b, nir_ieq_imm(b, encode, HK_STORE_ENCODE_RAW32));
   nir_def *raw = value;
   nir_push_else(b, NULL);
   nir_def *rest;
   if (floating) {
      nir_push_if(b, nir_ieq_imm(b, encode, HK_STORE_ENCODE_UNORM8));
      nir_def *word = nir_imm_int(b, 0);
      for (unsigned c = 0; c < 4; ++c) {
         nir_def *u = nir_f2u32(b, nir_fround_even(b,
            nir_fmul_imm(b, nir_fsat(b, nir_channel(b, value, c)), 255.0)));
         word = nir_ior(b, word, nir_ishl_imm(b, u, 8 * c));
      }
      nir_def *unorm8 = nir_vec4(b, word, nir_imm_int(b, 0), nir_imm_int(b, 0),
                                 nir_imm_int(b, 0));
      nir_push_else(b, NULL);
      nir_push_if(b, nir_ieq_imm(b, encode, HK_STORE_ENCODE_FLOAT16));
      nir_def *float16 = nir_vec4(b,
         nir_pack_half_2x16_split(b, nir_channel(b, value, 0), nir_channel(b, value, 1)),
         nir_pack_half_2x16_split(b, nir_channel(b, value, 2), nir_channel(b, value, 3)),
         nir_imm_int(b, 0), nir_imm_int(b, 0));
      nir_push_else(b, NULL);
      nir_push_if(b, nir_ieq_imm(b, encode, HK_STORE_ENCODE_FLOAT11));
      nir_def *float11 = nir_vec4(b,
         hk_apple9_pack_float11(b, nir_trim_vector(b, value, 3)),
         nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0));
      nir_push_else(b, NULL);
      nir_def *generic = hk_apple9_encode_texel(b, value, layout, true);
      nir_pop_if(b, NULL);
      nir_def *float11_or_generic = nir_if_phi(b, float11, generic);
      nir_pop_if(b, NULL);
      nir_def *half_or_generic = nir_if_phi(b, float16, float11_or_generic);
      nir_pop_if(b, NULL);
      rest = nir_if_phi(b, unorm8, half_or_generic);
   } else {
      rest = hk_apple9_encode_texel(b, value, layout, false);
   }
   nir_pop_if(b, NULL);
   return nir_if_phi(b, raw, rest);
}

/* The storage texel metadata carries the full address, including the view's
 * byte offset. The legacy PBE prefix is retained only for image queries.
 */
static nir_def *
hk_apple9_buffer_texel_address(nir_builder *b, nir_def *meta, nir_def *coord,
                               nir_def *size_B)
{
   nir_def *base = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, meta, offsetof(struct hk_apple9_texel_metadata, base)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *offset = nir_imul(b, nir_channel(b, coord, 0), size_B);
   return nir_iadd(b, base, nir_u2u64(b, offset));
}

static bool
hk_apple9_lower_buffer_address(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   bool atomic = intr->intrinsic == nir_intrinsic_bindless_image_atomic;
   bool swap = intr->intrinsic == nir_intrinsic_bindless_image_atomic_swap;
   if (!atomic && !swap &&
       intr->intrinsic != nir_intrinsic_bindless_image_texel_address)
      return false;
   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *pbe = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
   nir_def *meta = nir_iadd_imm(b, pbe,
      offsetof(struct hk_apple9_storage_descriptor, metadata) -
      offsetof(struct hk_storage_image_descriptor, pbe));
   unsigned size_B = atomic || swap ? intr->def.bit_size / 8 :
      util_format_get_blocksize(nir_intrinsic_format(intr));
   nir_def *addr = hk_apple9_buffer_texel_address(
      b, meta, intr->src[1].ssa, nir_imm_int(b, size_B));
   nir_def *result = addr;
   if (atomic || swap) {
      result = swap ? nir_global_atomic_swap(b, intr->def.bit_size, addr,
         intr->src[3].ssa, intr->src[4].ssa,
         .atomic_op = nir_intrinsic_atomic_op(intr)) :
         nir_global_atomic(b, intr->def.bit_size, addr, intr->src[3].ssa,
                           .atomic_op = nir_intrinsic_atomic_op(intr));
      /* Preserve the memory-to-texture fence used by common image lowering. */
      if (!(nir_intrinsic_access(intr) & ACCESS_NON_READABLE))
         nir_fence_mem_to_tex_agx(b);
   }
   nir_def_replace(&intr->def, result);
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* Storage access without a shader format reads the texel layout from the
 * Apple9 storage descriptor, then converts each component at run time.
 */
/* Storage-image loads can read through the texture unit, whose cache does
 * not observe shader writes. That is only correct for a load that no image
 * write or atomic of the same invocation can precede: none earlier in
 * program order, none in a loop around it, and no image barrier that could
 * make another invocation's write visible. Volatile loads stay on memory,
 * as do coherent ones when the shader writes images at all.
 */
static bool
hk_apple9_image_write(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
      return true;
   default:
      return false;
   }
}

static nir_loop *
hk_outermost_loop(nir_block *block)
{
   nir_loop *loop = NULL;
   for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
      if (n->type == nir_cf_node_loop)
         loop = nir_cf_node_as_loop(n);
   }
   return loop;
}

static void
hk_apple9_texture_route_loads(nir_shader *nir, struct set *safe)
{
   if (nir->info.internal)
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct set *write_loops = _mesa_pointer_set_create(NULL);
   bool writes = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             hk_apple9_image_write(nir_instr_as_intrinsic(instr))) {
            writes = true;
            nir_loop *loop = hk_outermost_loop(block);
            if (loop)
               _mesa_set_add(write_loops, loop);
         }
      }
   }
   bool written = false;
   nir_foreach_block(block, impl) {
      nir_loop *loop = hk_outermost_loop(block);
      bool loop_writes = loop && _mesa_set_search(write_loops, loop);
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (hk_apple9_image_write(intr)) {
            written = true;
         } else if (intr->intrinsic == nir_intrinsic_barrier) {
            if (writes && (nir_intrinsic_memory_modes(intr) & nir_var_image))
               written = true;
         } else if (intr->intrinsic == nir_intrinsic_bindless_image_load) {
            enum gl_access_qualifier access = nir_intrinsic_access(intr);
            enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
            if (!written && !loop_writes && !(access & ACCESS_VOLATILE) &&
                !(writes && (access & ACCESS_COHERENT)) &&
                dim != GLSL_SAMPLER_DIM_BUF && dim != GLSL_SAMPLER_DIM_MS &&
                dim != GLSL_SAMPLER_DIM_SUBPASS && dim != GLSL_SAMPLER_DIM_SUBPASS_MS)
               _mesa_set_add(safe, intr);
         }
      }
   }
   _mesa_set_destroy(write_loops, NULL);
}

/* Fetch a storage texel through the native entry of the descriptor that
 * load handle `tex` points at, see hk_apple9_storage_descriptor. */
static nir_def *
hk_apple9_texture_load(nir_builder *b, nir_intrinsic_instr *intr,
                       nir_intrinsic_instr *handle, nir_def *coord)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool cube = dim == GLSL_SAMPLER_DIM_CUBE;
   bool array = nir_intrinsic_image_array(intr) || cube;
   unsigned coords = (dim == GLSL_SAMPLER_DIM_1D ? 1 : dim == GLSL_SAMPLER_DIM_3D ? 3 : 2) +
                     (array && dim != GLSL_SAMPLER_DIM_3D);
   nir_def *native = nir_bindless_image_agx(
      b, nir_iadd_imm(b, handle->src[0].ssa,
                      offsetof(struct hk_apple9_storage_descriptor, texture) -
                         AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY *
                            AGX_APPLE9_TEXTURE_TABLE_STRIDE),
      .desc_set = nir_intrinsic_desc_set(handle));
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = cube ? GLSL_SAMPLER_DIM_2D : dim;
   tex->is_array = array;
   tex->coord_components = coords;
   tex->dest_type = nir_intrinsic_dest_type(intr);
   /* The caller has bounds-checked every coordinate, including the layer.
    * The common and Apple9 fetch bounds would read texture state relative
    * to the handle, which here addresses the storage descriptor's native
    * entry and has no such state beside it. */
   tex->backend_flags |= AGX_TEXTURE_FLAG_NO_CLAMP |
                         AGX_APPLE9_TEXTURE_FETCH_BOUNDED;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
      nir_trim_vector(b, nir_u2u32(b, coord), coords));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, native);
   nir_def_init(&tex->instr, &tex->def, 4, intr->def.bit_size);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

/* Whether the descriptor behind load handle `tex` has a native entry. */
static nir_def *
hk_apple9_native_ok(nir_builder *b, nir_def *tex)
{
   return nir_ine_imm(b, nir_iand_imm(b,
      nir_load_global_constant(b, 1, 32,
         nir_iadd_imm(b, nir_load_from_texture_handle_agx(b, tex),
                      offsetof(struct hk_apple9_storage_descriptor, native_ok)),
         .align_mul = 4), HK_APPLE9_STORAGE_READ_VALID), 0);
}

static nir_cf_node *
hk_apple9_outer_loop(nir_block *block)
{
   nir_cf_node *loop = NULL;
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_loop)
         loop = node;
   }
   return loop;
}

/* PBE export completion makes writes visible after a dispatch, but our
 * same-invocation check did not observe a native write through global image
 * memory, so no invocation may read after its own native store. That holds
 * when every image read precedes every image store in program order and no
 * loop holds both; a read by another invocation after a store needs a barrier
 * the read would then follow, so the same order covers it. Block order is
 * program order only within one inlined entry point, so refuse calls. Decide
 * before lowering removes any image loads. */
bool
hk_apple9_native_store_shader(const nir_shader *nir)
{
   unsigned impls = 0;
   nir_foreach_function_impl(impl, nir)
      ++impls;
   if (impls != 1)
      return false;
   bool stored = false, ok = true;
   struct set *read_loops = _mesa_pointer_set_create(NULL);
   struct set *store_loops = _mesa_pointer_set_create(NULL);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               ok = false;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_cf_node *loop = hk_apple9_outer_loop(block);
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_bindless_image_atomic:
            case nir_intrinsic_bindless_image_atomic_swap:
               ok &= !stored;
               if (loop)
                  _mesa_set_add(read_loops, loop);
               break;
            case nir_intrinsic_bindless_image_store:
               stored = true;
               if (loop)
                  _mesa_set_add(store_loops, loop);
               break;
            default:
               break;
            }
         }
      }
   }
   set_foreach(store_loops, entry)
      ok &= !_mesa_set_search(read_loops, entry->key);
   _mesa_set_destroy(read_loops, NULL);
   _mesa_set_destroy(store_loops, NULL);
   return ok;
}

struct hk_apple9_storage_lowering {
   struct set *texture_loads;
   bool native_stores;
};

/* The image-write data type must be known: untyped stores (meta copies of raw
 * texels through canonical uint views) would be encoded as float data. */
static bool
hk_apple9_native_store_type(nir_alu_type type)
{
   nir_alu_type base = nir_alu_type_get_base_type(type);
   return base == nir_type_float || base == nir_type_int || base == nir_type_uint;
}

/* 3D views have one 2D entry per depth slice, addressed like array layers. */
static bool
hk_apple9_can_native_store(const nir_builder *b, const nir_intrinsic_instr *intr,
                            bool native_stores)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   enum pipe_format format = nir_intrinsic_format(intr);
   return native_stores &&
          (format == PIPE_FORMAT_NONE || hk_apple9_native_store_format(format)) &&
          b->shader->info.stage == MESA_SHADER_COMPUTE &&
          (dim == GLSL_SAMPLER_DIM_2D || dim == GLSL_SAMPLER_DIM_CUBE ||
           dim == GLSL_SAMPLER_DIM_3D) &&
          intr->src[3].ssa->bit_size == 32 &&
          hk_apple9_native_store_type(nir_intrinsic_src_type(intr)) &&
          !(nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE));
}

static nir_def *
hk_apple9_native_write_ok(nir_builder *b, nir_def *tex, bool array)
{
   nir_def *flags = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, nir_load_from_texture_handle_agx(b, tex),
                   offsetof(struct hk_apple9_storage_descriptor, native_ok)),
      .align_mul = 4);
   unsigned mask = array ? HK_APPLE9_STORAGE_LAYER_WRITE_VALID
                         : HK_APPLE9_STORAGE_WRITE_VALID;
   return nir_ine_imm(b, nir_iand_imm(b, flags, mask), 0);
}

static void
hk_apple9_native_store(nir_builder *b, nir_intrinsic_instr *intr,
                        nir_def *pbe_addr, nir_def *coord, bool array)
{
   nir_def *packed = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, pbe_addr,
         offsetof(struct hk_apple9_storage_descriptor, native_ok) -
            offsetof(struct hk_storage_image_descriptor, pbe)), .align_mul = 4);
   nir_def *index = nir_iand_imm(b,
      nir_ushr_imm(b, packed, HK_APPLE9_STORAGE_PBE_SHIFT),
      HK_APPLE9_STORAGE_PBE_MASK);
   /* Storage cube coordinates already contain the flattened face layer, and
    * 3D views have one 2D entry per depth slice. */
   if (array)
      index = nir_iadd(b, index, nir_channel(b, coord, 2));
   nir_image_store_apple9_agx(b, index, nir_trim_vector(b, nir_u2u32(b, coord), 2),
      nir_pad_vector_imm_int(b, intr->src[3].ssa, 0, 4),
      .access = nir_intrinsic_access(intr), .src_type = nir_intrinsic_src_type(intr));
}

static bool
hk_apple9_lower_formatless_storage(nir_builder *b, nir_intrinsic_instr *intr,
                                   bool load, bool route, bool native_stores)
{
   if (nir_def_instr(intr->src[0].ssa)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *handle = nir_def_as_intrinsic(intr->src[0].ssa);
   if (handle->intrinsic != nir_intrinsic_bindless_image_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   unsigned set = nir_intrinsic_desc_set(handle);
   nir_def *offset = handle->src[0].ssa;
   /* Load descriptors point at tex, store descriptors at pbe. */
   nir_def *tex = load ? intr->src[0].ssa : nir_bindless_image_agx(
      b, nir_iadd_imm(b, offset, -sizeof(struct agx_texture_packed)), .desc_set = set);
   nir_def *pbe = load ? nir_bindless_image_agx(
      b, nir_iadd_imm(b, offset, sizeof(struct agx_texture_packed)), .desc_set = set)
      : intr->src[0].ssa;
   bool array = nir_intrinsic_image_array(intr);
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool cube = dim == GLSL_SAMPLER_DIM_CUBE;
   unsigned dimensions;
   switch (dim) {
   case GLSL_SAMPLER_DIM_BUF: dimensions = 1; break;
   case GLSL_SAMPLER_DIM_1D: dimensions = 1 + array; break;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_MS: dimensions = 2 + array; break;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE: dimensions = 3; break;
   default: return false;
   }
   enum gl_access_qualifier access = nir_intrinsic_access(intr);

   nir_def *meta = nir_iadd_imm(b, nir_load_from_texture_handle_agx(b, tex),
      offsetof(struct hk_apple9_storage_descriptor, metadata));
   nir_def *size_B = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, meta, offsetof(struct hk_apple9_texel_metadata, size_B)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *layout = nir_load_global_constant(b, 4, 32,
      nir_iadd_imm(b, meta, offsetof(struct hk_apple9_texel_metadata, component)),
      .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   /* Storage image descriptors also name a canonical store encoding. */
   nir_def *encode = NULL;
   if (!load && dim != GLSL_SAMPLER_DIM_BUF) {
      encode = nir_load_global_constant(b, 1, 32,
         nir_iadd_imm(b, nir_load_from_texture_handle_agx(b, tex),
                      offsetof(struct hk_apple9_storage_descriptor, store_encode)),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
   }

   /* Honeykrisp's storage cube descriptors are 2D arrays of faces. */
   nir_def *size = nir_bindless_image_size(b, dimensions, 32, tex, nir_imm_int(b, 0),
      .image_dim = cube ? GLSL_SAMPLER_DIM_2D : dim, .image_array = cube || array);
   nir_def *coord = intr->src[1].ssa;
   nir_def *inside = nir_imm_true(b);
   for (unsigned d = 0; d < dimensions; ++d)
      inside = nir_iand(b, inside,
         nir_ult(b, nir_channel(b, coord, d), nir_channel(b, size, d)));
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_bindless_image_samples(b, 32, tex,
         .image_dim = dim, .image_array = array);
      inside = nir_iand(b, inside, nir_ult(b, nir_u2u32(b, intr->src[2].ssa), samples));
   }
   /* The address is arithmetic on the descriptor, valid for any coordinate,
    * so compute it ahead of the bounds check: its uniform part can then
    * move to the preamble. Only the access below depends on the check. */
   bool native = !load && hk_apple9_can_native_store(b, intr, native_stores);
   bool native_array = array || cube || dim == GLSL_SAMPLER_DIM_3D;
   nir_def *ok = route ? hk_apple9_native_ok(b, tex) :
      native ? hk_apple9_native_write_ok(b, tex, native_array) : NULL;
   nir_def *pbe_addr = nir_load_from_texture_handle_agx(b, pbe);
   nir_def *addr = NULL;
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      addr = hk_apple9_buffer_texel_address(b, meta, coord, size_B);
   } else if (!route && !native) {
      bool layered = array || cube || dim == GLSL_SAMPLER_DIM_3D;
      addr = libagx_image_texel_address(
         b, pbe_addr, nir_pad_vector_imm_int(b, coord, 0, 4),
         nir_u2u32(b, intr->src[2].ssa), size_B,
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_MS), nir_imm_bool(b, layered),
         nir_imm_bool(b, false));
   }
   nir_push_if(b, inside);
   nir_def *value = NULL;
   if (load) {
      bool floating = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) ==
                      nir_type_float;
      nir_def *fetched = NULL;
      if (route) {
         nir_push_if(b, ok);
         fetched = hk_apple9_texture_load(b, intr, handle, coord);
         nir_push_else(b, NULL);
         bool layered = array || cube || dim == GLSL_SAMPLER_DIM_3D;
         addr = libagx_image_texel_address(
            b, pbe_addr, nir_pad_vector_imm_int(b, coord, 0, 4),
            nir_u2u32(b, intr->src[2].ssa), size_B,
            nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
            nir_imm_bool(b, false), nir_imm_bool(b, layered),
            nir_imm_bool(b, false));
      }
      nir_def *words = hk_apple9_load_texel_words(b, addr, size_B, access);
      value = hk_apple9_decode_texel(b, words, layout, floating);
      if (route) {
         nir_pop_if(b, NULL);
         value = nir_if_phi(b, fetched, value);
      }
   } else {
      if (native) {
         nir_push_if(b, ok);
         hk_apple9_native_store(b, intr, pbe_addr, coord, native_array);
         nir_push_else(b, NULL);
         addr = libagx_image_texel_address(
            b, pbe_addr, nir_pad_vector_imm_int(b, coord, 0, 4),
            nir_u2u32(b, intr->src[2].ssa), size_B,
            nir_imm_bool(b, false), nir_imm_bool(b, false),
            nir_imm_bool(b, native_array), nir_imm_bool(b, false));
      }
      bool floating = nir_alu_type_get_base_type(nir_intrinsic_src_type(intr)) ==
                      nir_type_float;
      nir_def *texel = nir_pad_vector_imm_int(b, intr->src[3].ssa, 0, 4);
      nir_def *words = encode
         ? hk_apple9_encode_texel_fast(b, texel, layout, floating, encode)
         : hk_apple9_encode_texel(b, texel, layout, floating);
      hk_apple9_store_texel_words(b, addr, size_B, words, access);
      if (native)
         nir_pop_if(b, NULL);
   }
   nir_def *zero = NULL;
   if (load) {
      /* Out-of-bounds reads return zero, with a one in formats lacking
       * alpha, matching the format's missing-component substitution. */
      nir_push_else(b, NULL);
      bool floating = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) ==
                      nir_type_float;
      nir_def *alpha_kind = nir_ushr_imm(b, nir_channel(b, layout, 3), 18);
      nir_def *one = floating ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
      zero = nir_vec4(b, nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0),
                      nir_bcsel(b, nir_ieq_imm(b, alpha_kind, HK_TEXEL_ONE), one,
                                nir_imm_int(b, 0)));
   }
   nir_pop_if(b, NULL);
   if (load) {
      value = nir_if_phi(b, value, zero);
      nir_def_replace(&intr->def, nir_trim_vector(b, value, intr->def.num_components));
   } else {
      nir_instr_remove(&intr->instr);
   }
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* Apple9 sampling reports no residency. Instead, find the page holding the
 * nearest texel of the nearest level in the image's residency table, see
 * struct hk_apple9_residency. Base is the view's level | layer << 16. A zero
 * residency code is resident.
 */
static nir_def *
hk_apple9_residency_code(nir_builder *b, nir_def *table, nir_def *base,
                         nir_def *level, nir_def *x, nir_def *y, nir_def *layer)
{
   nir_def *header = nir_load_global_constant(b, 2, 32, table, .align_mul = 16);
   nir_def *abs_level = nir_umin_imm(
      b, nir_iadd(b, nir_iand_imm(b, base, 0xffff), level), AIL_MAX_MIP_LEVELS - 1);
   nir_def *entry = nir_load_global_constant(b, 2, 32,
      nir_iadd(b, table, nir_u2u64(b, nir_iadd_imm(b, nir_imul_imm(b, abs_level, 8),
         offsetof(struct hk_apple9_residency, level)))), .align_mul = 8);
   nir_def *packed = nir_channel(b, entry, 1);
   nir_def *shift_x = nir_iand_imm(b, nir_ushr_imm(b, packed, 16), 0xff);
   nir_def *shift_y = nir_ushr_imm(b, packed, 24);
   nir_def *tile = nir_iadd(b,
      nir_imul(b, nir_ushr(b, y, shift_y), nir_iand_imm(b, packed, 0xffff)),
      nir_ushr(b, x, shift_x));
   nir_def *abs_layer = nir_iadd(b, nir_ushr_imm(b, base, 16), layer);
   nir_def *page = nir_iadd(b, nir_imul(b, abs_layer, nir_channel(b, header, 0)),
                            nir_iadd(b, nir_channel(b, entry, 0), tile));
   page = nir_umin(b, page, nir_iadd_imm(b, nir_channel(b, header, 1), -1));
   nir_def *word = nir_load_global_constant(b, 1, 32,
      nir_iadd(b, table, nir_u2u64(b, nir_iadd_imm(b, nir_iand_imm(b, page, ~3),
         offsetof(struct hk_apple9_residency, resident)))), .align_mul = 4);
   nir_def *byte = nir_iand_imm(
      b, nir_ushr(b, word, nir_imul_imm(b, nir_iand_imm(b, page, 3), 8)), 0xff);
   return nir_b2i32(b, nir_ieq_imm(b, byte, 0));
}

/* Residency code of texel x of a buffer view, from the table of its buffer's
 * pages (see hk_apple9_texel_metadata::residency). A zero table is resident.
 */
static nir_def *
hk_apple9_buffer_residency_code(nir_builder *b, nir_def *meta, nir_def *x)
{
   nir_def *table = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, meta, offsetof(struct hk_apple9_texel_metadata, residency)),
      .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
   nir_def *code;
   nir_push_if(b, nir_ine_imm(b, table, 0));
   {
      nir_def *base = nir_load_global_constant(b, 1, 64, meta, .align_mul = 8);
      nir_def *fields = nir_load_global_constant(b, 2, 32,
         nir_iadd_imm(b, meta, offsetof(struct hk_apple9_texel_metadata, elements)),
         .align_mul = 8, .access = ACCESS_UNIFORM_SAFE_AGX);
      nir_def *elements = nir_channel(b, fields, 0);
      nir_def *size_B = nir_channel(b, fields, 1);
      x = nir_umin(b, x, nir_iadd_imm(b, nir_umax_imm(b, elements, 1), -1));
      nir_def *addr = nir_iadd(b, base, nir_u2u64(b, nir_imul(b, x, size_B)));
      nir_def *entry = nir_iadd(b, table,
         nir_ushr_imm(b, addr, util_logbase2(AIL_PAGESIZE)));
      nir_def *word = nir_load_global_constant(b, 1, 32,
         nir_iand_imm(b, entry, ~3ull), .align_mul = 4);
      nir_def *byte = nir_iand_imm(b, nir_ushr(b, word,
         nir_imul_imm(b, nir_u2u32(b, nir_iand_imm(b, entry, 3)), 8)), 0xff);
      code = nir_b2i32(b, nir_ieq_imm(b, byte, 0));
   }
   nir_push_else(b, NULL);
   nir_def *resident = nir_imm_int(b, 0);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, code, resident);
}

/* The LOD a sparse sample selects, before the nearest level is chosen. */
static nir_def *
hk_apple9_sparse_lod(nir_builder *b, nir_tex_instr *tex, nir_def *xy,
                     nir_def *size)
{
   nir_def *lod;
   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (tex->op == nir_texop_tg4) {
      lod = nir_imm_float(b, 0.0);
   } else if (tex->op == nir_texop_txl) {
      lod = nir_f2f32(b, tex->src[lod_idx].src.ssa);
   } else if (tex->op == nir_texop_txd ||
              b->shader->info.stage == MESA_SHADER_FRAGMENT) {
      nir_def *dx, *dy;
      if (tex->op == nir_texop_txd) {
         dx = nir_f2f32(b, tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddx)].src.ssa);
         dy = nir_f2f32(b, tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddy)].src.ssa);
      } else {
         dx = nir_ddx(b, xy);
         dy = nir_ddy(b, xy);
      }
      dx = nir_fmul(b, nir_trim_vector(b, dx, 2), size);
      dy = nir_fmul(b, nir_trim_vector(b, dy, 2), size);
      lod = nir_flog2(b, nir_fsqrt(b, nir_fmax(b, nir_fdot2(b, dx, dx),
                                                  nir_fdot2(b, dy, dy))));
      int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
      if (bias >= 0)
         lod = nir_fadd(b, lod, nir_f2f32(b, tex->src[bias].src.ssa));
   } else {
      lod = nir_imm_float(b, 0.0);
   }
   int min_lod = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (min_lod >= 0)
      lod = nir_fmax(b, lod, nir_f2f32(b, tex->src[min_lod].src.ssa));
   return lod;
}

static bool
hk_apple9_lower_sparse_tex(nir_builder *b, nir_tex_instr *tex,
                           UNUSED void *data)
{
   if (!tex->is_sparse)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *code = nir_imm_int(b, 0);
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);

   /* Cube maps have no sparse residency here. */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF && handle >= 0) {
      nir_def *meta = nir_iadd_imm(b,
         nir_load_from_texture_handle_agx(b, tex->src[handle].src.ssa),
         offsetof(struct hk_apple9_texel_descriptor, metadata));
      nir_def *coord =
         tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa;
      code = hk_apple9_buffer_residency_code(b, meta,
                                             nir_u2u32(b, nir_channel(b, coord, 0)));
   } else if (tex->sampler_dim == GLSL_SAMPLER_DIM_2D && handle >= 0) {
      nir_def *desc =
         nir_load_from_texture_handle_agx(b, tex->src[handle].src.ssa);
      nir_def *fields = nir_load_global_constant(b, 3, 32,
         nir_iadd_imm(b, desc, offsetof(struct hk_apple9_image_part, residency_lo)),
         .align_mul = 4);
      nir_def *table = nir_pack_64_2x32_split(b, nir_channel(b, fields, 0),
                                              nir_channel(b, fields, 1));
      nir_def *coord =
         tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa;
      nir_def *level, *x, *y, *layer = nir_imm_int(b, 0);

      if (tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms) {
         coord = nir_u2u32(b, coord);
         int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
         level = lod >= 0 ? nir_u2u32(b, tex->src[lod].src.ssa) : nir_imm_int(b, 0);
         x = nir_channel(b, coord, 0);
         y = nir_channel(b, coord, 1);
         if (tex->is_array)
            layer = nir_channel(b, coord, 2);
      } else {
         coord = nir_f2f32(b, coord);
         nir_def *xy = nir_trim_vector(b, coord, 2);
         /* This query is inserted after descriptor lowering, which normally
          * marks sampled-image queries as using the Apple9 metadata. Without
          * the flag, the common query lowering decodes that metadata as a
          * legacy texture descriptor.
          */
         nir_def *size_query = nir_get_texture_size(b, tex);
         nir_def_as_tex(size_query)->backend_flags |= AGX_TEXTURE_FLAG_METADATA_QUERY;
         nir_def *size0 = nir_trim_vector(b, size_query, 2);
         nir_def *lod = hk_apple9_sparse_lod(b, tex, xy, nir_u2f32(b, size0));
         /* Nearest mipmapping selects ceil(lod + 1/2) - 1. */
         level = nir_f2u32(b, nir_fmax(b, nir_fceil(b, nir_fadd_imm(b, lod, -0.5)),
                                       nir_imm_float(b, 0.0)));
         nir_def *size = nir_umax_imm(b, nir_ushr(b, size0, level), 1);
         nir_def *texel = nir_fmin(b, nir_fmax(b, nir_fmul(b, xy, nir_u2f32(b, size)),
                                               nir_imm_zero(b, 2, 32)),
                                   nir_u2f32(b, nir_iadd_imm(b, size, -1)));
         texel = nir_f2u32(b, texel);
         x = nir_channel(b, texel, 0);
         y = nir_channel(b, texel, 1);
         if (tex->is_array) {
            layer = nir_f2u32(b, nir_fmax(b, nir_fround_even(b, nir_channel(b, coord, 2)),
                                          nir_imm_float(b, 0.0)));
         }
      }
      code = hk_apple9_residency_code(b, table, nir_channel(b, fields, 2),
                                      level, x, y, layer);
   }

   tex->is_sparse = false;
   tex->def.num_components--;
   b->cursor = nir_after_instr(&tex->instr);
   nir_scalar comps[5];
   for (unsigned c = 0; c < tex->def.num_components; ++c)
      comps[c] = nir_get_scalar(&tex->def, c);
   comps[tex->def.num_components] = nir_get_scalar(code, 0);
   nir_def *result = nir_vec_scalars(b, comps, tex->def.num_components + 1);
   nir_def_rewrite_uses_after(&tex->def, result);
   return true;
}

static bool
hk_apple9_lower_sparse_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                                 UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_is_sparse_texels_resident:
      nir_def_replace(&intr->def, nir_ieq_imm(b, intr->src[0].ssa, 0));
      return true;
   case nir_intrinsic_sparse_residency_code_and:
      nir_def_replace(&intr->def,
                      nir_ior(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;
   case nir_intrinsic_bindless_image_sparse_load:
      break;
   default:
      return false;
   }

   /* Storage views have a single level. Cube faces are array layers. */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   nir_def *code = nir_imm_int(b, 0);
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      nir_def *meta = nir_iadd_imm(b,
         nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
         offsetof(struct hk_apple9_storage_descriptor, metadata));
      code = hk_apple9_buffer_residency_code(b, meta,
         nir_u2u32(b, nir_channel(b, intr->src[1].ssa, 0)));
   } else if (dim == GLSL_SAMPLER_DIM_2D || dim == GLSL_SAMPLER_DIM_CUBE) {
      nir_def *meta = nir_iadd_imm(b,
         nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
         offsetof(struct hk_apple9_storage_descriptor, metadata));
      nir_def *table = nir_load_global_constant(b, 1, 64, meta, .align_mul = 8);
      nir_def *base = nir_load_global_constant(b, 1, 32,
         nir_iadd_imm(b, meta, offsetof(struct hk_apple9_texel_metadata, elements)),
         .align_mul = 4, .access = ACCESS_UNIFORM_SAFE_AGX);
      nir_def *coord = nir_u2u32(b, intr->src[1].ssa);
      bool layered = nir_intrinsic_image_array(intr) || dim == GLSL_SAMPLER_DIM_CUBE;
      code = hk_apple9_residency_code(b, table, base, nir_imm_int(b, 0),
                                      nir_channel(b, coord, 0),
                                      nir_channel(b, coord, 1),
                                      layered ? nir_channel(b, coord, 2)
                                              : nir_imm_int(b, 0));
   }

   intr->intrinsic = nir_intrinsic_bindless_image_load;
   intr->num_components--;
   intr->def.num_components--;
   b->cursor = nir_after_instr(&intr->instr);
   nir_scalar comps[5];
   for (unsigned c = 0; c < intr->def.num_components; ++c)
      comps[c] = nir_get_scalar(&intr->def, c);
   comps[intr->def.num_components] = nir_get_scalar(code, 0);
   nir_def *result = nir_vec_scalars(b, comps, intr->def.num_components + 1);
   nir_def_rewrite_uses_after(&intr->def, result);
   return true;
}

bool
hk_apple9_lower_sparse_residency(nir_shader *nir)
{
   bool progress = nir_shader_tex_pass(nir, hk_apple9_lower_sparse_tex,
                                       nir_metadata_none, NULL);
   progress |= nir_shader_intrinsics_pass(nir, hk_apple9_lower_sparse_intrinsic,
                                          nir_metadata_none, NULL);
   return progress;
}

/* Apple9 can access uncompressed storage images through the same
 * descriptor-based address calculation as image atomics. Descriptor contents
 * remain in GPU memory, and bounds checks precede every memory access.
 */
static bool
hk_apple9_lower_storage_image(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   const struct hk_apple9_storage_lowering *lower = data;
   bool load = intr->intrinsic == nir_intrinsic_bindless_image_load;
   if (!load && intr->intrinsic != nir_intrinsic_bindless_image_store)
      return false;
   enum pipe_format format = nir_intrinsic_format(intr);
   if (format == PIPE_FORMAT_NONE)
      return hk_apple9_lower_formatless_storage(b, intr, load,
                                                load && _mesa_set_search(lower->texture_loads, intr),
                                                lower->native_stores);
   const struct util_format_description *desc = util_format_description(format);
   unsigned size_B = util_format_get_blocksize(format);
   if ((desc->layout != UTIL_FORMAT_LAYOUT_PLAIN &&
        format != PIPE_FORMAT_R11G11B10_FLOAT && format != PIPE_FORMAT_R9G9B9E5_FLOAT) ||
       !util_is_power_of_two_nonzero(size_B) || size_B > 16 ||
       util_format_is_depth_or_stencil(format))
      return false;

   if (nir_def_instr(intr->src[0].ssa)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *handle = nir_def_as_intrinsic(intr->src[0].ssa);
   if (handle->intrinsic != nir_intrinsic_bindless_image_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   unsigned set = nir_intrinsic_desc_set(handle);
   nir_def *offset = handle->src[0].ssa;
   /* Load descriptors point at tex, store descriptors at pbe. */
   nir_def *tex = load ? intr->src[0].ssa : nir_bindless_image_agx(
      b, nir_iadd_imm(b, offset, -sizeof(struct agx_texture_packed)), .desc_set = set);
   nir_def *pbe = load ? nir_bindless_image_agx(
      b, nir_iadd_imm(b, offset, sizeof(struct agx_texture_packed)), .desc_set = set)
      : intr->src[0].ssa;
   bool array = nir_intrinsic_image_array(intr);
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool cube = dim == GLSL_SAMPLER_DIM_CUBE;
   unsigned dimensions;
   switch (dim) {
   case GLSL_SAMPLER_DIM_BUF: dimensions = 1; break;
   case GLSL_SAMPLER_DIM_1D: dimensions = 1 + array; break;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_MS: dimensions = 2 + array; break;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE: dimensions = 3; break;
   default: return false;
   }
   /* Honeykrisp's storage cube descriptors are 2D arrays of faces. Query
    * them as arrays to bounds-check the face coordinate without cube scaling.
    */
   nir_def *size = nir_bindless_image_size(b, dimensions, 32, tex, nir_imm_int(b, 0),
      .image_dim = cube ? GLSL_SAMPLER_DIM_2D : dim, .image_array = cube || array);
   nir_def *coord = intr->src[1].ssa;
   nir_def *inside = nir_imm_true(b);
   for (unsigned d = 0; d < dimensions; ++d)
      inside = nir_iand(b, inside,
         nir_ult(b, nir_channel(b, coord, d), nir_channel(b, size, d)));
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_bindless_image_samples(b, 32, tex,
         .image_dim = dim, .image_array = array);
      inside = nir_iand(b, inside, nir_ult(b, nir_u2u32(b, intr->src[2].ssa), samples));
   }
   /* Typed loads decode with a known format through global memory. */
   bool native = !load && hk_apple9_can_native_store(b, intr, lower->native_stores);
   bool native_array = array || cube || dim == GLSL_SAMPLER_DIM_3D;
   nir_def *ok = native ? hk_apple9_native_write_ok(b, tex, native_array) : NULL;
   /* See hk_apple9_lower_formatless_storage: address ahead of the check. */
   nir_def *addr = native ? NULL : nir_bindless_image_texel_address(b, 64, pbe, coord,
      intr->src[2].ssa, .image_dim = dim,
      .image_array = array, .format = format, .access = nir_intrinsic_access(intr));
   nir_push_if(b, inside);
   nir_def *value = NULL;
   if (load) {
      nir_def *raw = nir_load_global(b, DIV_ROUND_UP(size_B, 4), MIN2(32, size_B * 8),
         addr, .align_mul = size_B, .access = nir_intrinsic_access(intr));
      value = nir_format_unpack_rgba(b, raw, format);
   } else {
      if (native) {
         nir_push_if(b, ok);
         hk_apple9_native_store(b, intr, nir_load_from_texture_handle_agx(b, pbe),
                                 coord, native_array);
         nir_push_else(b, NULL);
         addr = nir_bindless_image_texel_address(b, 64, pbe, coord,
            intr->src[2].ssa, .image_dim = dim, .image_array = array,
            .format = format, .access = nir_intrinsic_access(intr));
      }
      nir_def *raw = format == PIPE_FORMAT_R11G11B10_FLOAT
         ? hk_apple9_pack_float11(b, nir_trim_vector(b, intr->src[3].ssa, 3))
         : nir_format_pack_rgba(b, format, intr->src[3].ssa);
      raw = size_B <= 4 ? nir_u2uN(b, raw, size_B * 8)
                        : nir_trim_vector(b, raw, size_B / 4);
      nir_store_global(b, raw, addr, .align_mul = size_B,
                       .write_mask = BITFIELD_MASK(raw->num_components),
                       .access = nir_intrinsic_access(intr));
      if (native)
         nir_pop_if(b, NULL);
   }
   nir_def *zero = NULL;
   if (load) {
      /* Out-of-bounds reads return zero, with a one in formats lacking
       * alpha, matching the format's missing-component substitution.
       */
      nir_push_else(b, NULL);
      bool alpha = desc->swizzle[3] <= PIPE_SWIZZLE_W;
      nir_alu_type type = nir_intrinsic_dest_type(intr);
      nir_def *one = nir_alu_type_get_base_type(type) == nir_type_float
                        ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
      zero = nir_vec4(b, nir_imm_int(b, 0), nir_imm_int(b, 0),
                      nir_imm_int(b, 0), alpha ? nir_imm_int(b, 0) : one);
   }
   nir_pop_if(b, NULL);
   if (load) {
      value = nir_if_phi(b, value, zero);
      nir_def_replace(&intr->def, nir_trim_vector(b, value, intr->def.num_components));
   } else {
      nir_instr_remove(&intr->instr);
   }
   b->shader->info.use_lowered_image_to_global = true;
   return true;
}

/* When every storage image access has become global memory, synchronize that
 * memory domain. Retain the texture fences if a storage image access through
 * the texture unit remains; those require their own cache-coherency
 * implementation. Sampled images and uniform texel buffers need not observe
 * writes from the same dispatch, and descriptor queries only read metadata,
 * so neither requires texture fences.
 */
static void
hk_apple9_lower_image_barriers(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (nir_intrinsic_has_image_dim(intr) &&
             intr->intrinsic != nir_intrinsic_image_size &&
             intr->intrinsic != nir_intrinsic_bindless_image_size &&
             intr->intrinsic != nir_intrinsic_image_samples &&
             intr->intrinsic != nir_intrinsic_bindless_image_samples &&
             intr->intrinsic != nir_intrinsic_image_levels &&
             intr->intrinsic != nir_intrinsic_bindless_image_levels)
            return;
      }
   }
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_barrier) {
            unsigned modes = nir_intrinsic_memory_modes(intr);
            if (modes & nir_var_image)
               nir_intrinsic_set_memory_modes(intr,
                  (modes & ~nir_var_image) | nir_var_mem_global);
         } else if (intr->intrinsic == nir_intrinsic_fence_pbe_to_tex_agx ||
                    intr->intrinsic == nir_intrinsic_fence_mem_to_tex_agx) {
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            nir_barrier(&b, .execution_scope = SCOPE_NONE,
                        .memory_scope = SCOPE_DEVICE,
                        .memory_semantics = NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE,
                        .memory_modes = nir_var_mem_global);
            nir_instr_remove(instr);
         }
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* storageInputOutput16: 16-bit stage inputs and outputs are carried as 32-bit
 * values at every interface, converting exactly at the access (f16 <-> f32,
 * 16-bit integers truncated on load and extended on store), as both sides of an
 * interface are widened alike. Slots and components are unchanged.
 */
static bool
hk_lower_io_16bit_to_32bit(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_per_primitive_input:
   case nir_intrinsic_load_input_vertex:
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output: {
      if (intr->def.bit_size != 16)
         return false;
      nir_alu_type type = nir_intrinsic_dest_type(intr);
      intr->def.bit_size = 32;
      nir_intrinsic_set_dest_type(intr, nir_alu_type_get_base_type(type) | 32);
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *narrow = nir_alu_type_get_base_type(type) == nir_type_float
                           ? nir_f2f16(b, &intr->def)
                           : nir_u2u16(b, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, narrow);
      return true;
   }
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output: {
      nir_def *value = intr->src[0].ssa;
      if (value->bit_size != 16)
         return false;
      nir_alu_type type = nir_intrinsic_src_type(intr);
      nir_alu_type base = nir_alu_type_get_base_type(type);
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *wide = base == nir_type_float ? nir_f2f32(b, value)
                      : base == nir_type_int  ? nir_i2i32(b, value)
                                              : nir_u2u32(b, value);
      nir_src_rewrite(&intr->src[0], wide);
      nir_intrinsic_set_src_type(intr, base | 32);
      return true;
   }
   default:
      return false;
   }
}

/* Mesh primitives rasterize with their per-primitive builtins as flat varyings
 * of each vertex. User per-primitive varyings are read from the records.
 */
static bool
hk_lower_per_primitive_input(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_per_primitive_input)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   bool native_broadcast = data && *(const bool *)data;
   if (!native_broadcast && hk_mesh_varying_in_memory(sem.location)) {
      nir_def *record = nir_load_input(
         b, 1, 32, nir_imm_int(b, 0), .range = 1, .dest_type = nir_type_uint32,
         .io_semantics.location = HK_MESH_PRIMITIVE_SLOT,
         .io_semantics.num_slots = 1);
      b->shader->info.inputs_read |= BITFIELD64_BIT(HK_MESH_PRIMITIVE_SLOT);

      unsigned params_offset = hk_root_descriptor_offset(draw.mesh_params);
      nir_def *params = nir_load_ubo(
         b, 1, 64, nir_imm_int(b, 0), nir_imm_int(b, params_offset),
         .align_mul = 8, .range_base = params_offset, .range = 8);
      nir_def *outputs = nir_load_global(
         b, 1, 64,
         nir_iadd_imm(b, params, offsetof(struct libagx_mesh_params, outputs)),
         .align_mul = 8);
      nir_def *varying = nir_load_global(
         b, 1, 16,
         nir_iadd_imm(b, params,
                      offsetof(struct libagx_mesh_params, primitive_varying_B) +
                         (sem.location - VARYING_SLOT_VAR0) * 2),
         .align_mul = 2);
      nir_def *offset = nir_iadd(
         b, nir_iadd(b, record, nir_u2u32(b, varying)),
         nir_iadd_imm(b, nir_imul_imm(b, intr->src[0].ssa, 16),
                      nir_intrinsic_component(intr) * 4));
      nir_def *value = nir_load_global(
         b, intr->def.num_components, 32,
         nir_iadd(b, outputs, nir_u2u64(b, offset)), .align_mul = 4);
      if (intr->def.bit_size != 32) {
         value = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) ==
                       nir_type_float
                    ? nir_f2fN(b, value, intr->def.bit_size)
                    : nir_u2uN(b, value, intr->def.bit_size);
      }
      nir_def_replace(&intr->def, value);
      return true;
   }

   nir_def *value = nir_load_input(
      b, intr->def.num_components, intr->def.bit_size, intr->src[0].ssa,
      .base = nir_intrinsic_base(intr), .range = sem.num_slots,
      .component = nir_intrinsic_component(intr),
      .dest_type = nir_intrinsic_dest_type(intr), .io_semantics = sem);
   nir_def_replace(&intr->def, value);
   return true;
}

/* With hk_apple9_d3d_mip_rounding (driconf, set for vkd3d-proton), add the
 * sampler's nudge to explicit LODs, so a NEAREST mip selection exactly
 * halfway between levels rounds up as in D3D. The nudge is zero for other
 * mipmap modes, see hk_CreateSampler.
 */
static bool
hk_lower_d3d_txl_rounding(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->op != nir_texop_txl || !nir_tex_instr_need_sampler(tex))
      return false;

   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (lod_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *nudge = nir_build_texture_query(b, tex, nir_texop_lod_bias, 1,
                                            nir_type_float16, false, false);
   nir_def_as_tex(nudge)->backend_flags |= HK_TEX_TXL_NUDGE;

   nir_def *lod = tex->src[lod_idx].src.ssa;
   nir_src_rewrite(&tex->src[lod_idx].src,
                   nir_fadd(b, lod, nir_f2fN(b, nudge, lod->bit_size)));
   return true;
}

/* With hk_apple9_arrayed_2d_views (driconf, set for vkd3d-proton), sampled
 * 1D and 2D views are one-layer 2D arrays on Apple9 (see pack_texture), since
 * its texture instructions must match the descriptor's arrayness. Sample
 * every non-array 2D texture, 1D having been lowered to 2D, as layer 0 of an
 * array. Layer 0 of a one-layer view is the view itself, and an array
 * declaration bound to a 2D view now honours its layer and level.
 */
static bool
hk_apple9_lower_tex_2d_as_array(nir_builder *b, nir_tex_instr *tex, void *data)
{
   if (tex->is_array || (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
                         tex->sampler_dim != GLSL_SAMPLER_DIM_MS))
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   int c = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (c >= 0) {
      nir_def *coord = tex->src[c].src.ssa;
      nir_def *layer = nir_tex_instr_src_type(tex, c) == nir_type_float
                          ? nir_imm_floatN_t(b, 0.0, coord->bit_size)
                          : nir_imm_intN_t(b, 0, coord->bit_size);
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < coord->num_components; ++i)
         comps[i] = nir_channel(b, coord, i);
      comps[coord->num_components] = layer;
      nir_src_rewrite(&tex->src[c].src, nir_vec(b, comps, coord->num_components + 1));
      tex->coord_components++;
   }
   tex->is_array = true;
   tex->backend_flags |= AGX_TEXTURE_FLAG_ARRAYED_2D;

   if (tex->op == nir_texop_txs) {
      /* Array sizes report the layer count too; return the 2D size. */
      unsigned old = tex->def.num_components;
      tex->def.num_components++;
      b->cursor = nir_after_instr(&tex->instr);
      nir_def *size = nir_trim_vector(b, &tex->def, old);
      nir_def_rewrite_uses_after(&tex->def, size);
   }
   return true;
}

static void
hk_lower_nir(struct hk_device *dev, nir_shader *nir,
             const struct vk_pipeline_robustness_state *rs, bool is_multiview,
             uint32_t set_layout_count,
             struct vk_descriptor_set_layout *const *set_layouts,
             enum hk_feature_key features)
{
   if (HK_PERF(dev, NOROBUST)) {
      rs = &vk_robustness_disabled;
   }

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_input_attachments,
               &(nir_input_attachment_options){
                  .use_view_id_for_layer = is_multiview,
               });

      NIR_PASS(_, nir, nir_shader_tex_pass, lower_subpass_dim, nir_metadata_all,
               NULL);
      NIR_PASS(_, nir, nir_lower_wpos_center);
   }

   /* XXX: should be last geometry stage, how do I get to that? */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      if (is_multiview)
         hk_lower_multiview(nir);
   }

   if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, nir, nir_lower_patch_vertices,
               nir->info.tess.tcs_vertices_out, NULL);
   }

   const struct lower_ycbcr_state ycbcr_state = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex, lookup_ycbcr_conversion,
            &ycbcr_state);

   /* Lower push constants before lower_descriptors */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   /* Turn cache flushes into image coherency bits while we still have derefs */
   NIR_PASS(_, nir, nir_lower_memory_model);

   NIR_PASS(_, nir, nir_lower_robust_access, should_lower_robust, NULL);

   /* We must do early lowering before hk_nir_lower_descriptors, since this will
    * create lod_bias instructions.
    */
   /* Apple9 bounds-checks buffer images in software against their element
    * count (hk_apple9_lower_storage_image). */
   NIR_PASS(_, nir, agx_nir_lower_texture_early, true /* support_lod_bias */,
            dev->dev.chip == AGX_CHIP_G16G);
   if (hk_device_physical(dev)->apple9_d3d_mip_rounding)
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_lower_d3d_txl_rounding,
               nir_metadata_control_flow, NULL);
   if (hk_device_physical(dev)->apple9_layered_2d)
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_apple9_lower_tex_2d_as_array,
               nir_metadata_control_flow, NULL);

   struct hk_instance *instance = hk_physical_device_instance(
      (struct hk_physical_device *)dev->vk.physical);

   /* Apple9 pays for the clamp only when the application enables the
    * extension. */
   if (dev->dev.chip == AGX_CHIP_G16G
          ? dev->vk.enabled_features.minLod
          : instance->drirc.misc.image_view_min_lod) {
      NIR_PASS(_, nir, agx_nir_lower_image_view_min_lod,
               dev->dev.chip == AGX_CHIP_G16G);
   }

   /* Apple9 exposes no custom border colors. */
   if ((features & HK_FEAT_CUSTOM_BORDER) && !HK_PERF(dev, NOBORDER) &&
       dev->dev.chip != AGX_CHIP_G16G) {
      NIR_PASS(_, nir, agx_nir_lower_custom_border);
   }

   NIR_PASS(_, nir, hk_nir_lower_descriptors, rs, set_layout_count,
            set_layouts, dev->dev.chip == AGX_CHIP_G16G);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            hk_buffer_addr_format(rs->storage_buffers));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            hk_buffer_addr_format(rs->uniform_buffers));

   /* Before inserting bounds checks, we want to do a fair bit of optimization.
    * lower_load_global_constant_offset_instr has special optimizations for
    * constant offsets, so we want as many offsets to be constant as possible.
    */
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   /* Apple9 vectorizes before the buffer base and offset become one 64-bit
    * address. Afterwards the offset is hidden inside u2u64, where adjacent
    * fields at x + c are no longer provably adjacent, and every field of a
    * dynamically indexed struct becomes a separate scalar load.
    */
   if (dev->dev.chip == AGX_CHIP_G16G) {
      /* Explicit I/O builds each access's base and offset afresh. The
       * vectorizer only merges accesses sharing those definitions. */
      NIR_PASS(_, nir, nir_opt_cse);
      if (agx_apple9_debug_enabled(AGX_A9_DBG_VECTORIZE) &&
          nir->info.stage == MESA_SHADER_FRAGMENT)
         nir_print_shader(nir, stderr);
      /* Below the pressure budget, merge without windows, unless that
       * raises the shader's peak of live words over windowed merging.
       */
      unsigned peak = hk_apple9_peak_live_words(nir);
      struct hk_vectorize_policy policy = {
         .windowed = peak > HK_APPLE9_VECTORIZE_PRESSURE_BUDGET};
      unsigned peak_windowed = 0, peak_free = 0;
      if (!policy.windowed) {
         peak_windowed = hk_apple9_vectorized_peak(nir, true);
         peak_free = hk_apple9_vectorized_peak(nir, false);
         policy.windowed = peak_free > peak_windowed;
      }
      if (agx_apple9_debug_enabled(AGX_A9_DBG_PRESSURE))
         fprintf(stderr,
                 "HK_PRESSURE stage=%u peak=%u windowed=%u merged_peaks=%u/%u\n",
                 nir->info.stage, peak,
                 policy.windowed, peak_windowed, peak_free);
      NIR_PASS(_, nir, nir_opt_load_store_vectorize,
               &(const nir_load_store_vectorize_options){
                  .modes = nir_var_mem_global,
                  .bounds_checked_modes = nir_var_mem_global,
                  .callback = hk_apple9_vectorize_cb,
                  .cb_data = &policy,
               });
   }

   nir_move_options opts = nir_move_load_ssbo | nir_move_load_ubo;
   NIR_PASS(progress, nir, nir_opt_sink, opts);
   NIR_PASS(progress, nir, nir_opt_move, opts);

   NIR_PASS(_, nir, hk_lower_bounded_global_loads,
            agx_has_soft_fault(&dev->dev), dev->dev.chip == AGX_CHIP_G16G);

   assert(nir->info.shared_size == 0);

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            shared_var_info);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   if (nir->info.stage == MESA_SHADER_TASK ||
       nir->info.stage == MESA_SHADER_MESH) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
               nir_var_mem_task_payload, shared_var_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_task_payload,
               nir_address_format_32bit_offset);
   }

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* Align everything up to 16B so we can write whole vec4s. */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size,
               16);

      /* We need to call lower_compute_system_values again because
       * nir_zero_initialize_shared_memory generates load_invocation_id which
       * has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   /* TODO: we can do indirect VS output */
   nir_variable_mode lower_indirect_modes = 0;
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      lower_indirect_modes |= nir_var_shader_out;
   else if (nir->info.stage == MESA_SHADER_VERTEX)
      lower_indirect_modes |= nir_var_shader_in | nir_var_shader_out;

   /* Apple9 interpolation packets address a fixed coefficient. Expand
    * dynamic fragment input indexing before lowering away the array derefs. */
   if (dev->dev.chip == AGX_CHIP_G16G &&
       nir->info.stage == MESA_SHADER_FRAGMENT)
      lower_indirect_modes |= nir_var_shader_in;

   /* Native mesh links a fixed component map. Resolve indirect output-array
    * slots while their declared bounds are still available as derefs. This
    * also covers arrays nested inside per-vertex/per-primitive blocks. */
   if (dev->dev.chip == AGX_CHIP_G16G &&
       hk_apple9_native_mesh_available(hk_device_physical(dev)) &&
       ((hk_apple9_native_mesh_enabled() &&
         agx_apple9_native_mesh_shape_eligible(nir)) ||
        hk_apple9_native_mesh_packed_shape_eligible(nir) ||
        hk_apple9_native_mesh_round_shape_eligible(nir)))
      lower_indirect_modes |= nir_var_shader_out;

   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            lower_indirect_modes, UINT32_MAX);

   /* nir_lower_io needs constant indices into compact arrays. Tess levels are
    * indexed dynamically (gl_TessLevelOuter[gl_InvocationID]), so turn them
    * into vectors and lower the component indexing, as RADV does.
    */
   if (nir->info.stage == MESA_SHADER_TESS_CTRL ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, nir, nir_lower_tess_level_array_vars_to_vec);
      NIR_PASS(_, nir, nir_lower_array_deref_of_vec,
               nir_var_shader_in | nir_var_shader_out, NULL,
               nir_lower_direct_array_deref_of_vec_load |
                  nir_lower_indirect_array_deref_of_vec_load |
                  nir_lower_direct_array_deref_of_vec_store |
                  nir_lower_indirect_array_deref_of_vec_store);
   }

   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            glsl_type_size,
            nir_lower_io_lower_64bit_to_32 |
               nir_lower_io_use_interpolated_input_intrinsics);
   if (dev->dev.chip == AGX_CHIP_G16G)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_lower_io_16bit_to_32bit,
               nir_metadata_control_flow, NULL);

   /* The qualified index profile publishes Position only. It cannot consume
    * per-primitive fragment inputs, so keep their original early lowering.
    * Only the broader research P1 interface defers these inputs to linking.
    * Delaying them for every enabled device would change excluded shaders. */
   bool native_primitive_inputs =
      dev->dev.chip == AGX_CHIP_G16G &&
      hk_apple9_native_mesh_available(hk_device_physical(dev)) &&
      hk_apple9_native_mesh_enabled();
   if (nir->info.stage == MESA_SHADER_FRAGMENT && !native_primitive_inputs) {
      /* The emulated path reads records; research P1 uses flat channels. */
      bool native_broadcast = false;
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               hk_lower_per_primitive_input, nir_metadata_control_flow,
               &native_broadcast);
   }

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               dev->dev.chip == AGX_CHIP_G16G
                  ? hk_apple9_lower_layer_viewport_fs
                  : lower_viewport_fs,
               nir_metadata_control_flow, NULL);
   }

   if (dev->dev.chip == AGX_CHIP_G16G) {
      /* Buffer residency lookups add control flow. */
      NIR_PASS(_, nir, hk_apple9_lower_sparse_residency);
      /* Texel-buffer bounds and RGB32 checks add control flow. */
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_apple9_lower_texel_buffer,
               nir_metadata_none, NULL);
      struct set *texture_loads = _mesa_pointer_set_create(NULL);
      hk_apple9_texture_route_loads(nir, texture_loads);
      struct hk_apple9_storage_lowering lower = {
         .texture_loads = texture_loads,
         .native_stores = hk_apple9_native_store_shader(nir),
      };
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_storage_image,
               nir_metadata_none, &lower);
      _mesa_set_destroy(texture_loads, NULL);
   }

   if (dev->dev.chip == AGX_CHIP_G16G) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_buffer_address,
               nir_metadata_control_flow, NULL);
      NIR_PASS(_, nir, agx_nir_lower_images);
      /* The ordinary descriptor prefix serves storage-image queries. Sampled
       * views' queries (AGX_TEXTURE_FLAG_METADATA_QUERY) are answered from
       * texture metadata by the Apple9 backend. */
      NIR_PASS(_, nir, agx_nir_lower_texture_queries);
   } else {
      NIR_PASS(_, nir, agx_nir_lower_texture);
   }
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);

   /* Keep radians until Apple9 range reduction. The common preprocessor
    * otherwise replaces sine/cosine with the older fsin_agx operation. */
   if (dev->dev.chip == AGX_CHIP_G16G)
      NIR_PASS(_, nir, agx_nir_lower_apple9_math);
   agx_preprocess_nir(nir);
   if (dev->dev.chip == AGX_CHIP_G16G)
      hk_apple9_lower_image_barriers(nir);

   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 0,
      .discard_ok = true,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   NIR_PASS(_, nir, nir_opt_if,
            nir_opt_if_optimize_phi_true_false | nir_opt_if_avoid_64bit_phis);
}

/* Byte-identical Apple9 bodies share one code allocation and shader entry.
 * Pipelines and draw-time variants often compile to the same code: in the
 * CS2 capture, three in four bodies duplicate another. The table keeps the
 * most recently uploaded bodies referenced, so a body may outlive its last
 * shader until evicted.
 */
#define HK_APPLE9_BODY_CACHE_SIZE 4096
/* Match the old count limit for minimum-size M4 BOs, but do not retain an
 * unbounded number of bytes when shader bodies span several pages. */
#define HK_APPLE9_BODY_CACHE_BYTES (64ull * 1024 * 1024)

#define HK_APPLE9_BODY_SLAB_SIZE (64 * 1024)
#define HK_APPLE9_BODY_ALIGNMENT 64
/* USC may prefetch beyond the last instruction. Keep the end of every slab
 * mapped but unallocated so its last body cannot prefetch into the VA guard. */
#define HK_APPLE9_BODY_TAIL 1024
#define HK_APPLE9_BODY_SLAB_USABLE \
   (HK_APPLE9_BODY_SLAB_SIZE - HK_APPLE9_BODY_TAIL)

struct hk_apple9_body_slab {
   struct agx_bo *bo;
   struct util_vma_heap heap;
   struct list_head link;
   unsigned live, cached;
};

struct hk_apple9_body {
   blake3_hash hash;
   uint64_t size, alloc_size, cache_charge;
   struct hk_apple9_body_slab *slab;
   struct agx_shader_body base;
   struct list_head link;
};

static uint32_t
hk_apple9_body_hash(const void *key)
{
   const struct hk_apple9_body *body = key;
   return _mesa_hash_data(body->hash, sizeof(body->hash));
}

static bool
hk_apple9_body_equal(const void *a, const void *b)
{
   const struct hk_apple9_body *x = a, *y = b;
   return x->size == y->size && !memcmp(x->hash, y->hash, sizeof(x->hash));
}

/* Heap and cache bookkeeping share apple9_bodies.lock. Body contents are
 * immutable before publication. Entry teardown releases its lock before this
 * callback, and BO release happens after dropping the body-cache lock. */
static void
hk_apple9_body_destroy(struct agx_device *adev, struct agx_shader_body *base)
{
   struct hk_device *dev = container_of(adev, struct hk_device, dev);
   struct hk_apple9_body *body =
      container_of(base, struct hk_apple9_body, base);
   struct hk_apple9_body_slab *slab = body->slab;
   bool release_bo = !slab;
   if (slab) {
      simple_mtx_lock(&dev->apple9_bodies.lock);
      util_vma_heap_free(
         &slab->heap,
         base->addr - slab->bo->va->addr + HK_APPLE9_BODY_ALIGNMENT,
         body->alloc_size);
      assert(slab->live);
      if (--slab->live == 0) {
         assert(!slab->cached);
         list_del(&slab->link);
         util_vma_heap_finish(&slab->heap);
         free(slab);
         release_bo = true;
      }
      simple_mtx_unlock(&dev->apple9_bodies.lock);
   }
   if (release_bo)
      agx_bo_unreference(adev, base->bo);
   free(body);
}

/* Offset zero is valid storage, but the VMA allocator uses it as failure.
 * Bias its offsets by one alignment unit and undo the bias for GPU addresses. */
static bool
hk_apple9_body_alloc(struct hk_device *dev, struct hk_apple9_body *body)
{
   uint64_t offset = 0;
   struct agx_bo *bo = NULL;
   struct agx_bo *failed_bo = NULL;
   body->alloc_size = ALIGN_POT(body->size, HK_APPLE9_BODY_ALIGNMENT);
   if (body->alloc_size <= HK_APPLE9_BODY_SLAB_USABLE) {
      simple_mtx_lock(&dev->apple9_bodies.lock);
      list_for_each_entry(struct hk_apple9_body_slab, slab,
                          &dev->apple9_bodies.slabs, link) {
         offset = util_vma_heap_alloc(&slab->heap, body->alloc_size,
                                      HK_APPLE9_BODY_ALIGNMENT);
         if (offset) {
            body->slab = slab;
            break;
         }
      }
      if (!body->slab) {
         struct hk_apple9_body_slab *slab = calloc(1, sizeof(*slab));
         if (slab) {
            slab->bo =
               agx_bo_create(&dev->dev, HK_APPLE9_BODY_SLAB_SIZE, 0,
                             AGX_BO_EXEC | AGX_BO_LOW_VA, "Apple9 shader slab");
            /* Map once before another uploader can reserve a slice. */
            if (slab->bo && agx_bo_map(slab->bo)) {
               util_vma_heap_init(&slab->heap, HK_APPLE9_BODY_ALIGNMENT,
                                  slab->bo->size - HK_APPLE9_BODY_TAIL);
               slab->heap.alloc_high = false;
               offset = util_vma_heap_alloc(&slab->heap, body->alloc_size,
                                            HK_APPLE9_BODY_ALIGNMENT);
               assert(offset);
               list_addtail(&slab->link, &dev->apple9_bodies.slabs);
               body->slab = slab;
            } else {
               failed_bo = slab->bo;
               free(slab);
            }
         }
      }
      if (body->slab) {
         body->slab->live++;
         bo = body->slab->bo;
         offset -= HK_APPLE9_BODY_ALIGNMENT;
      }
      simple_mtx_unlock(&dev->apple9_bodies.lock);
      agx_bo_unreference(&dev->dev, failed_bo);
   } else {
      bo = agx_bo_create(&dev->dev, body->alloc_size, 0,
                         AGX_BO_EXEC | AGX_BO_LOW_VA, "Apple9 shader");
   }
   if (!bo)
      return false;
   body->base = (struct agx_shader_body){
      .bo = bo,
      .addr = bo->va->addr + offset,
      .refcnt = 1,
      .destroy = hk_apple9_body_destroy,
   };
   /* Keep the previous page-rounded admission policy as well as a separate
    * bound on actual backing retained by cached bodies. */
   body->cache_charge =
      body->slab ? ALIGN_POT(body->size, AIL_PAGESIZE) : bo->size;
   return true;
}

static void
hk_apple9_body_uncache(struct hk_device *dev, struct hk_apple9_body *body)
{
   dev->apple9_bodies.count--;
   dev->apple9_bodies.bytes -= body->cache_charge;
   if (!body->slab || --body->slab->cached == 0)
      dev->apple9_bodies.backing_bytes -= body->base.bo->size;
}

static struct agx_shader_body *
hk_apple9_body_upload(struct hk_device *dev, const struct hk_shader *shader)
{
   uint64_t data_offset = ALIGN_POT((uint64_t)shader->code_size,
                                    HK_MIN_UBO_ALIGNMENT);
   uint64_t size = data_offset + shader->data_size;
   if (!size || size > SIZE_MAX - (HK_APPLE9_BODY_ALIGNMENT - 1))
      return NULL;
   struct hk_apple9_body key = {.size = size};
   blake3_hasher hash;
   blake3_hasher_init(&hash);
   blake3_hasher_update(&hash, &shader->code_size, sizeof(shader->code_size));
   blake3_hasher_update(&hash, shader->code_ptr, shader->code_size);
   blake3_hasher_update(&hash, shader->data_ptr, shader->data_size);
   blake3_hasher_finalize(&hash, key.hash, sizeof(key.hash));

   simple_mtx_lock(&dev->apple9_bodies.lock);
   if (!dev->apple9_bodies.ht) {
      dev->apple9_bodies.ht = _mesa_hash_table_create(
         NULL, hk_apple9_body_hash, hk_apple9_body_equal);
      list_inithead(&dev->apple9_bodies.lru);
   }
   struct hash_entry *entry =
      dev->apple9_bodies.ht ? _mesa_hash_table_search(dev->apple9_bodies.ht, &key)
                            : NULL;
   if (entry) {
      struct hk_apple9_body *body = entry->data;
      list_move_to(&body->link, &dev->apple9_bodies.lru);
      agx_shader_body_reference(&body->base);
      simple_mtx_unlock(&dev->apple9_bodies.lock);
      return &body->base;
   }
   simple_mtx_unlock(&dev->apple9_bodies.lock);

   struct hk_apple9_body *body = malloc(sizeof(*body));
   if (!body)
      return NULL;
   *body = key;
   if (!hk_apple9_body_alloc(dev, body)) {
      free(body);
      return NULL;
   }
   struct agx_bo *bo = body->base.bo;
   void *map = agx_bo_map(bo);
   if (!map) {
      agx_shader_body_unreference(&dev->dev, &body->base);
      return NULL;
   }
   uint64_t offset = body->base.addr - bo->va->addr;
   map = (char *)map + offset;
   memset(map, 0, body->alloc_size);
   memcpy(map, shader->code_ptr, shader->code_size);
   if (shader->data_size)
      memcpy((char *)map + data_offset,
             shader->data_ptr, shader->data_size);
   agx_bo_note_cpu_write(bo, offset, body->alloc_size);

   struct list_head evicted;
   list_inithead(&evicted);
   simple_mtx_lock(&dev->apple9_bodies.lock);
   if (dev->apple9_bodies.ht && bo->size <= HK_APPLE9_BODY_CACHE_BYTES &&
       !_mesa_hash_table_search(dev->apple9_bodies.ht, body) &&
       _mesa_hash_table_insert(dev->apple9_bodies.ht, body, body)) {
      agx_shader_body_reference(&body->base);
      list_add(&body->link, &dev->apple9_bodies.lru);
      dev->apple9_bodies.count++;
      dev->apple9_bodies.bytes += body->cache_charge;
      if (!body->slab || body->slab->cached++ == 0)
         dev->apple9_bodies.backing_bytes += bo->size;
      while (dev->apple9_bodies.count > HK_APPLE9_BODY_CACHE_SIZE ||
             dev->apple9_bodies.bytes > HK_APPLE9_BODY_CACHE_BYTES ||
             dev->apple9_bodies.backing_bytes > HK_APPLE9_BODY_CACHE_BYTES) {
         struct hk_apple9_body *old = list_last_entry(
            &dev->apple9_bodies.lru, struct hk_apple9_body, link);
         list_del(&old->link);
         _mesa_hash_table_remove_key(dev->apple9_bodies.ht, old);
         hk_apple9_body_uncache(dev, old);
         list_addtail(&old->link, &evicted);
      }
   }
   simple_mtx_unlock(&dev->apple9_bodies.lock);
   list_for_each_entry_safe(struct hk_apple9_body, old, &evicted, link) {
      list_del(&old->link);
      agx_shader_body_unreference(&dev->dev, &old->base);
   }
   /* A concurrent identical upload, oversized body or failed cache insertion
    * leaves the caller owning a valid body without a cache reference. */
   return &body->base;
}

void
hk_apple9_bodies_finish(struct hk_device *dev)
{
   if (!dev->apple9_bodies.ht)
      return;
   list_for_each_entry_safe(struct hk_apple9_body, body,
                            &dev->apple9_bodies.lru, link) {
      list_del(&body->link);
      hk_apple9_body_uncache(dev, body);
      agx_shader_body_unreference(&dev->dev, &body->base);
   }
   _mesa_hash_table_destroy(dev->apple9_bodies.ht, NULL);
   dev->apple9_bodies.ht = NULL;
   assert(!dev->apple9_bodies.count && !dev->apple9_bodies.bytes &&
          !dev->apple9_bodies.backing_bytes);
}

/* The native description of an uploaded G16 graphics stage. */
static void
hk_apple9_init_render_stage(struct hk_shader *s)
{
   bool fragment = s->info.stage == MESA_SHADER_FRAGMENT;
   const struct agx_shader_info *i = &s->b.info;
   struct agx_apple9_render_stage stage = {
      .program_id = s->apple9_program_id,
      .bo = s->bo,
      .body = s->apple9_body,
      .binary = s->code_ptr,
      .binary_size = i->binary_size,
      .preamble_offset = i->apple9_preamble_offset,
      .preamble_size = i->apple9_preamble_size,
      .publication_count = i->apple9_publication_count,
      .publication_count_valid = true,
      .native_mesh = s->info.apple9_native_stage == HK_APPLE9_NATIVE_MESH,
      .scratch_size = i->scratch_size,
      .atomic_frame_size = i->apple9_atomic_frame_size,
      .resource_count = i->apple9_resource_count,
      .resource_ssbo_mask = i->apple9_resource_ssbo_mask,
      .resource_write_mask = i->apple9_resource_write_mask,
      .position_components = fragment ? 0 : 4,
      .varying_components = i->apple9_varyings.count,
      .varyings = i->apple9_varyings,
      .apple9_linear_mask = i->apple9_linear_mask,
      .apple9_flat_mask = i->apple9_flat_mask,
      .apple9_reads_z = i->apple9_reads_z,
      .reads_primitive_id = i->apple9_reads_primitive_id,
      .reads_point_coord = i->apple9_reads_point_coord,
      .barycentric_mask = i->apple9_barycentric_mask,
      .per_vertex_count = i->apple9_per_vertex_count,
      .writes_point_size = i->apple9_writes_point_size,
      .writes_layer_viewport = i->writes_layer_viewport,
      .clip_distance_count = i->apple9_clip_distance_count,
      .render_targets = s->info.apple9_render_targets,
      .launch_cache = s->apple9_stage_launch,
      .texture_mask = i->apple9_texture_mask,
      .sampler_mask = i->apple9_sampler_mask,
      .image_mask = i->apple9_image_mask,
      .uses_texel_fetch = i->apple9_uses_texel_fetch,
      .reads_tile = i->apple9_reads_tile,
      .uses_discard = i->apple9_uses_discard,
      .writes_depth = i->depth_layout != FRAG_DEPTH_LAYOUT_UNCHANGED,
      .disable_tri_merging = i->disable_tri_merging,
      .pixel_interlock = i->apple9_pixel_interlock,
   };
   memcpy(stage.resource_binding, i->apple9_resource_binding,
          sizeof(stage.resource_binding));
   memcpy(stage.per_vertex, i->apple9_per_vertex, sizeof(stage.per_vertex));
   /* Every uploaded graphics stage has an id; without one, consecutive draws
    * silently stop reusing launches and coefficient tables. */
   assert(!s->bo || stage.program_id);
   s->apple9_stage = stage;
}

static VkResult
hk_upload_shader(struct hk_device *dev, struct hk_shader *shader)
{
   bool native = shader->info.apple9_native_stage != HK_APPLE9_NATIVE_NONE;
   if ((shader->b.info.apple9_body_only || native) &&
       !(dev->dev.chip == AGX_CHIP_G16G &&
         hk_apple9_native_mesh_available(hk_device_physical(dev)) &&
         shader->info.stage == (shader->info.apple9_native_stage ==
                                   HK_APPLE9_NATIVE_OBJECT
                                      ? MESA_SHADER_TASK : MESA_SHADER_MESH) &&
         hk_apple9_native_shader_valid(&shader->b.info,
                                       &shader->info.apple9_native,
                                       shader->info.apple9_native_stage)))
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apple9 native stage requires its native launch contract");
   if (dev->dev.chip == AGX_CHIP_G16G) {
      agx_apple9_dump_info(&shader->b.info);
      if (!shader->code_size ||
          (shader->info.stage == MESA_SHADER_COMPUTE &&
           agx_is_shader_empty(&shader->b)))
         return VK_SUCCESS;
      shader->apple9_body = hk_apple9_body_upload(dev, shader);
      if (!shader->apple9_body)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      shader->bo = shader->apple9_body->bo;
      if (shader->data_size)
         shader->data_addr = shader->apple9_body->addr +
                             ALIGN_POT((uint64_t)shader->code_size, HK_MIN_UBO_ALIGNMENT);
      hk_gpu_busy_shader(shader->apple9_body->addr, shader->code_ptr, shader->code_size,
                         shader->info.stage);
      if (shader->info.stage == MESA_SHADER_COMPUTE) {
         const struct agx_apple9_compute_profile *p = &shader->info.apple9;
         struct agx_apple9_launch_template *t = malloc(sizeof(*t));
         if (t && agx_apple9_compute_launch_template(
                     t, p, dev->dev.shader_base,
                     p->preamble_size ? shader->apple9_body->addr + p->preamble_offset
                                      : 0))
            shader->apple9_launch = t;
         else
            free(t);
      } else {
         shader->apple9_program_id =
            p_atomic_inc_return(&dev->apple9_last_program_id);
         shader->apple9_stage_launch =
            malloc(sizeof(*shader->apple9_stage_launch));
         if (shader->apple9_stage_launch)
            agx_apple9_stage_launch_cache_init(shader->apple9_stage_launch);
         hk_apple9_init_render_stage(shader);
         if (shader->info.stage == MESA_SHADER_FRAGMENT) {
            simple_mtx_lock(&dev->usc_heap_lock);
            bool allocated = hk_usc_heap_alloc(
               &dev->usc_heap,
               agx_apple9_coefficients_size(&shader->apple9_stage), 64,
               &shader->apple9_coefficients);
            simple_mtx_unlock(&dev->usc_heap_lock);
            if (!allocated)
               return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
            agx_apple9_init_coefficients(
               &dev->dev, &shader->apple9_stage,
               (struct agx_ptr){.cpu = shader->apple9_coefficients.map,
                                .gpu = shader->apple9_coefficients.addr});
         }
      }
      return VK_SUCCESS;
   }
   if (shader->b.info.has_preamble || shader->b.info.rodata.size_16 ||
       shader->data_size) {
      /* TODO: Do we wnat to compact? Revisit when we rework prolog/epilogs. */
      uint64_t data_offset = ALIGN_POT((uint64_t)shader->b.info.binary_size,
                                      HK_MIN_UBO_ALIGNMENT);
      uint64_t size = data_offset + shader->data_size;
      if (size > SIZE_MAX)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      assert(size > 0);

      shader->bo = agx_bo_create(&dev->dev, size, 0,
                                 AGX_BO_EXEC | AGX_BO_LOW_VA, "Preamble");
      if (!shader->bo)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      void *map = agx_bo_map(shader->bo);
      if (!map)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      memset(map, 0, size);
      memcpy(map, shader->b.binary, shader->b.info.binary_size);
      if (shader->data_size) {
         memcpy((char *)map + data_offset, shader->data_ptr, shader->data_size);
         shader->data_addr = shader->bo->va->addr + data_offset;
      }
      shader->preamble_addr =
         shader->bo->va->addr + shader->b.info.preamble_offset;
   }

   if (!shader->linked.ht) {
      /* If we only have a single variant, link now. */
      shader->only_linked = hk_fast_link(dev, false, shader, NULL, NULL, 0);
   }

   if (shader->info.stage == MESA_SHADER_FRAGMENT) {
      agx_pack_fragment_face_2(&shader->frag_face, 0, &shader->b.info);
   }

   agx_pack(&shader->counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->b.info.push_count;
      cfg.preshader_register_count = shader->b.info.nr_preamble_gprs;
      cfg.sampler_state_register_count = agx_translate_sampler_state_count(
         shader->b.info.sampler_state_count, false);
      cfg.texture_state_register_count = shader->b.info.texture_state_count;
   }
   return VK_SUCCESS;
}

DERIVE_HASH_TABLE(hk_fast_link_key_vs);
DERIVE_HASH_TABLE(hk_fast_link_key_fs);

static VkResult
hk_init_link_ht(struct hk_shader *shader, mesa_shader_stage sw_stage)
{
   simple_mtx_init(&shader->linked.lock, mtx_plain);

   bool multiple_variants =
      sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_FRAGMENT;

   if (!multiple_variants)
      return VK_SUCCESS;

   if (sw_stage == MESA_SHADER_VERTEX)
      shader->linked.ht = hk_fast_link_key_vs_table_create(NULL);
   else
      shader->linked.ht = hk_fast_link_key_fs_table_create(NULL);

   return (shader->linked.ht == NULL) ? VK_ERROR_OUT_OF_HOST_MEMORY
                                      : VK_SUCCESS;
}

static nir_intrinsic_instr *
hk_apple9_handle_intrinsic(nir_tex_instr *tex, unsigned sampler)
{
   int src = nir_tex_instr_src_index(
      tex, sampler ? nir_tex_src_sampler_handle : nir_tex_src_texture_handle);
   if (src < 0)
      return NULL;
   nir_instr *instr = nir_def_instr(tex->src[src].src.ssa);
   if (instr->type != nir_instr_type_intrinsic)
      return NULL;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == (sampler ? nir_intrinsic_bindless_sampler_agx
                                      : nir_intrinsic_bindless_image_agx)
             ? intr : NULL;
}

/* Address a descriptor relative to the USC aperture that holds G16
 * descriptor memory, see hk_apple9_image_part. */
static void
hk_apple9_use_descriptor_handle(nir_builder *b, nir_tex_instr *tex,
                                unsigned sampler)
{
   int src = nir_tex_instr_src_index(
      tex, sampler ? nir_tex_src_sampler_handle : nir_tex_src_texture_handle);
   nir_intrinsic_instr *handle = hk_apple9_handle_intrinsic(tex, sampler);
   b->cursor = nir_before_instr(&tex->instr);
   unsigned set = nir_intrinsic_desc_set(handle);
   unsigned root = hk_root_descriptor_offset(sets) + set * sizeof(uint64_t);
   nir_def *base = nir_load_ubo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, root),
                                .align_mul = 8, .range_base = root, .range = 4);
   /* Texture and sampler handles address their record part directly. */
   nir_def *offset = handle->src[0].ssa;
   nir_src_rewrite(&tex->src[src].src, nir_iadd(b, base, offset));
}

/* Debug: HK_APPLE9_DUMP_FRAGMENT_DIR saves each draw-time fragment compile's
 * input (serialized NIR, varying layout, blend states) as
 * <binary hash>-<size>.a9frag, named like the GPU pass log's shader hashes.
 * agx_apple9_fragment_replay recompiles it on the host. */
#define HK_APPLE9_FRAGMENT_DUMP_MAGIC "A9FRAG01"

static bool
hk_apple9_fragment_dump_begin(const nir_shader *nir, struct blob *blob)
{
   if (!debug_get_option_cached_hk_apple9_dump_fragment_dir())
      return false;
   blob_init(blob);
   nir_serialize(blob, nir, false);
   return !blob->out_of_memory;
}

static void
hk_apple9_fragment_dump_end(struct blob *nir_blob, bool compiled,
                            const struct agx_apple9_varying_layout *varyings,
                            const struct agx_apple9_blend *blend,
                            unsigned targets, const struct agx_shader_part *part)
{
   const char *dir = debug_get_option_cached_hk_apple9_dump_fragment_dir();
   if (compiled && dir) {
      const unsigned blends = MAX2(targets, 1);
      const uint32_t header[] = {
         sizeof(*varyings), sizeof(*blend), targets, blends, nir_blob->size,
      };
      char path[512];
      snprintf(path, sizeof(path), "%s/%08x-%u.a9frag", dir,
               _mesa_hash_data(part->binary, part->info.binary_size),
               part->info.binary_size);
      FILE *f = fopen(path, "wbx");
      if (f) {
         fwrite(HK_APPLE9_FRAGMENT_DUMP_MAGIC, 1, 8, f);
         fwrite(header, sizeof(header), 1, f);
         fwrite(varyings, sizeof(*varyings), 1, f);
         fwrite(blend, sizeof(*blend), blends, f);
         fwrite(nir_blob->data, 1, nir_blob->size, f);
         fclose(f);
      }
   }
   blob_finish(nir_blob);
}

/* Which memory a shader reads, from its intrinsics: texture and image
 * operations read images; loads from any buffer, global or descriptor
 * memory and vertex attributes read buffers. Queries and stores do not
 * read. Unknown memory loads count as buffers. */
bool
hk_nir_reads_draw_id(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_draw_id)
               return true;
         }
      }
   }
   return false;
}

/* Recognize the address expressions produced by descriptor lowering. Stop at
 * memory loads: a descriptor pointer loaded from a root is not root memory.
 * Unknown arithmetic/phis conservatively keep the full push region. */
enum hk_root_address_kind {
   HK_ROOT_ADDRESS_OTHER,
   HK_ROOT_ADDRESS_CONSTANT,
   HK_ROOT_ADDRESS_ROOT,
   HK_ROOT_ADDRESS_UNKNOWN,
};

struct hk_root_address {
   enum hk_root_address_kind kind;
   uint64_t offset;
};

static struct hk_root_address
hk_root_address(nir_def *def, unsigned depth)
{
   if (depth > 32)
      return (struct hk_root_address){HK_ROOT_ADDRESS_UNKNOWN};

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return (struct hk_root_address){
         HK_ROOT_ADDRESS_CONSTANT, nir_src_as_uint(nir_src_for_ssa(def))};

   if (instr->type == nir_instr_type_intrinsic) {
      return (struct hk_root_address){
         nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_root_agx
            ? HK_ROOT_ADDRESS_ROOT : HK_ROOT_ADDRESS_OTHER};
   }

   if (instr->type != nir_instr_type_alu)
      return (struct hk_root_address){HK_ROOT_ADDRESS_UNKNOWN};

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   struct hk_root_address a = hk_root_address(alu->src[0].src.ssa, depth + 1);
   if (alu->op == nir_op_mov || alu->op == nir_op_u2u64)
      return a;

   unsigned n = nir_op_infos[alu->op].num_inputs;
   if (alu->op == nir_op_iadd) {
      struct hk_root_address b = hk_root_address(alu->src[1].src.ssa, depth + 1);
      if ((a.kind == HK_ROOT_ADDRESS_CONSTANT &&
           b.kind == HK_ROOT_ADDRESS_CONSTANT) ||
          (a.kind == HK_ROOT_ADDRESS_ROOT &&
           b.kind == HK_ROOT_ADDRESS_CONSTANT) ||
          (a.kind == HK_ROOT_ADDRESS_CONSTANT &&
           b.kind == HK_ROOT_ADDRESS_ROOT)) {
         uint64_t offset = (a.offset + b.offset) & BITFIELD64_MASK(def->bit_size);
         return (struct hk_root_address){MAX2(a.kind, b.kind), offset};
      }
      return (struct hk_root_address){
         a.kind >= HK_ROOT_ADDRESS_ROOT || b.kind >= HK_ROOT_ADDRESS_ROOT
            ? HK_ROOT_ADDRESS_UNKNOWN : HK_ROOT_ADDRESS_OTHER};
   }

   bool root = a.kind >= HK_ROOT_ADDRESS_ROOT;
   for (unsigned i = 1; i < n; ++i)
      root |= hk_root_address(alu->src[i].src.ssa, depth + 1).kind >=
              HK_ROOT_ADDRESS_ROOT;
   return (struct hk_root_address){root ? HK_ROOT_ADDRESS_UNKNOWN
                                       : HK_ROOT_ADDRESS_OTHER};
}

static uint64_t
hk_push_range_mask(uint64_t start, uint64_t size)
{
   static_assert(HK_MAX_PUSH_SIZE == 64 * sizeof(uint32_t));
   if (start >= HK_MAX_PUSH_SIZE || size > HK_MAX_PUSH_SIZE - start)
      return UINT64_MAX;
   unsigned first = start / 4;
   unsigned end = DIV_ROUND_UP(start + size, 4);
   return BITFIELD64_RANGE(first, end - first);
}

uint64_t
hk_nir_push_constant_mask(const nir_shader *nir)
{
   uint64_t mask = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            unsigned bytes = in->num_components * in->def.bit_size / 8;
            if (in->intrinsic == nir_intrinsic_load_push_constant) {
               if (!nir_src_is_const(in->src[0]))
                  return UINT64_MAX;
               mask |= hk_push_range_mask((uint64_t)nir_intrinsic_base(in) +
                                             nir_src_as_uint(in->src[0]), bytes);
            } else if (in->intrinsic == nir_intrinsic_load_deref &&
                       nir_deref_mode_is(nir_src_as_deref(in->src[0]),
                                         nir_var_mem_push_const)) {
               return UINT64_MAX;
            } else if (in->intrinsic == nir_intrinsic_load_global_constant ||
                       in->intrinsic == nir_intrinsic_load_global ||
                       in->intrinsic == nir_intrinsic_load_global_constant_offset) {
               struct hk_root_address a = hk_root_address(in->src[0].ssa, 0);
               if (in->intrinsic == nir_intrinsic_load_global_constant_offset &&
                   a.kind >= HK_ROOT_ADDRESS_ROOT) {
                  if (!nir_src_is_const(in->src[1]))
                     return UINT64_MAX;
                  a.offset += nir_src_as_uint(in->src[1]);
               }
               if (a.kind == HK_ROOT_ADDRESS_UNKNOWN)
                  return UINT64_MAX;
               if (a.kind != HK_ROOT_ADDRESS_ROOT)
                  continue;
               const uint64_t start = hk_root_descriptor_offset(push);
               const uint64_t end = start + HK_MAX_PUSH_SIZE;
               if (a.offset < end && bytes > start - MIN2(a.offset, start)) {
                  uint64_t lo = MAX2(a.offset, start);
                  uint64_t hi = MIN2(a.offset + bytes, end);
                  mask |= hk_push_range_mask(lo - start, hi - lo);
               }
            }
         }
      }
   }
   return mask;
}

#ifndef NDEBUG
/* hk_apple9_lower_roots replaces load_root_agx with a load of root_desc_addr
 * from UBO 0. Recover that identity on a clone so the final-NIR check sees
 * both indirect root loads and direct UBO 0 reads. Other loaded pointers
 * still stop the address walk. The NIR passed to the backend is untouched.
 */
static bool
hk_debug_restore_root(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_ubo)
      return false;

   bool *unknown = data;
   if (!nir_src_is_const(intr->src[0])) {
      *unknown = true;
      return false;
   }
   if (nir_src_as_uint(intr->src[0]) != 0)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *root = nir_load_root_agx(b);
   nir_def *value;
   if (intr->num_components == 1 && intr->def.bit_size == 64 &&
       nir_src_is_const(intr->src[1]) &&
       nir_src_as_uint(intr->src[1]) ==
          hk_root_descriptor_offset(root_desc_addr)) {
      value = root;
   } else {
      nir_def *offset = nir_u2u64(b, intr->src[1].ssa);
      value = nir_load_global_constant(
         b, intr->num_components, intr->def.bit_size,
         nir_iadd(b, root, offset),
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr));
   }
   nir_def_replace(&intr->def, value);
   return true;
}

/* Geometry lowering introduces application-memory address phis and vector
 * expressions which the early root walk conservatively calls unknown. They
 * cannot refer to our root unless they depend on load_root_agx. Follow phis
 * with a visited set, but stop at memory loads just like the early analysis.
 */
struct hk_debug_root_walk {
   struct set *visited;
   bool found;
};

static bool
hk_debug_root_source(nir_src *src, void *data)
{
   struct hk_debug_root_walk *walk = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   bool found;
   _mesa_set_search_or_add(walk->visited, instr, &found);
   if (found)
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      walk->found |= nir_instr_as_intrinsic(instr)->intrinsic ==
                     nir_intrinsic_load_root_agx;
   } else {
      nir_foreach_src(instr, hk_debug_root_source, walk);
   }
   return !walk->found;
}

static bool
hk_debug_exclude_non_root_load(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global &&
       intr->intrinsic != nir_intrinsic_load_global_constant &&
       intr->intrinsic != nir_intrinsic_load_global_constant_offset)
      return false;
   struct hk_debug_root_walk walk = {
      .visited = _mesa_pointer_set_create(NULL),
   };
   assert(walk.visited);
   hk_debug_root_source(&intr->src[0], &walk);
   _mesa_set_destroy(walk.visited, NULL);
   if (walk.found)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[0], nir_imm_int64(b, 0));
   return true;
}

uint64_t
hk_nir_final_push_constant_mask(const nir_shader *nir)
{
   nir_shader *copy = nir_shader_clone(NULL, nir);
   assert(copy);
   bool unknown = false;
   NIR_PASS(_, copy, nir_shader_intrinsics_pass, hk_debug_restore_root,
            nir_metadata_control_flow, &unknown);
   /* Fold only the newly introduced integer conversions/adds, so direct
    * constant UBO offsets remain recognizable to the conservative walk. */
   NIR_PASS(_, copy, nir_opt_constant_folding);
   NIR_PASS(_, copy, nir_shader_intrinsics_pass, hk_debug_exclude_non_root_load,
            nir_metadata_control_flow, NULL);
   uint64_t mask = unknown ? UINT64_MAX : hk_nir_push_constant_mask(copy);
   ralloc_free(copy);
   return mask;
}

static void
hk_assert_final_push_mask(const nir_shader *nir, const struct hk_shader *shader)
{
   const struct hk_api_shader *obj = shader->debug_owner;
   assert(obj);
   uint64_t final = hk_nir_final_push_constant_mask(nir);
   assert((final & ~obj->push_constant_mask) == 0 ||
          obj->push_constant_mask == UINT64_MAX);
}
#else
#define hk_assert_final_push_mask(nir, shader) ((void)0)
#endif

uint8_t
hk_nir_pre_raster_reads(const nir_shader *nir)
{
   uint8_t reads = 0;
   if (nir->info.stage == MESA_SHADER_VERTEX && nir->info.inputs_read)
      reads |= HK_READS_BUFFERS;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               if (!nir_tex_instr_is_query(nir_instr_as_tex(instr)))
                  reads |= HK_READS_IMAGES;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_image_load:
            case nir_intrinsic_image_sparse_load:
            case nir_intrinsic_image_deref_load:
            case nir_intrinsic_image_deref_sparse_load:
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_load_input_attachment_coord:
               reads |= HK_READS_IMAGES;
               break;
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_ubo_vec4:
            case nir_intrinsic_load_ssbo:
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_global_2x32:
            case nir_intrinsic_load_global_bounded:
            case nir_intrinsic_load_global_constant_bounded:
            case nir_intrinsic_load_global_constant_offset:
            case nir_intrinsic_load_input:
            case nir_intrinsic_load_input_vertex:
            case nir_intrinsic_load_per_vertex_input:
               reads |= HK_READS_BUFFERS;
               break;
            case nir_intrinsic_load_deref: {
               nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
               if (nir_deref_mode_may_be(deref, nir_var_mem_ubo | nir_var_mem_ssbo |
                                                   nir_var_mem_global |
                                                   nir_var_shader_in))
                  reads |= HK_READS_BUFFERS;
               if (nir_deref_mode_may_be(deref, nir_var_image | nir_var_uniform))
                  reads |= HK_READS_IMAGES;
               break;
            }
            default:
               break;
            }
         }
      }
   }
   return reads;
}

static bool
hk_apple9_texture_bindings(nir_shader *nir)
{
   /* Every sampled descriptor is read through its memory (a descriptor
    * handle): native records live in set memory, written at update time, so
    * draws and dispatches build no texture/sampler tables and update-after-
    * bind needs nothing extra. The handle and the descriptor-record reads are
    * uniform and hoisted to the preamble. */
   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         for (unsigned sampler = 0; sampler < 2; ++sampler) {
            tex->backend_flags &= ~HK_APPLE9_TEX_DESCRIPTOR_HANDLE;
            if (nir_tex_instr_src_index(tex, sampler
                   ? nir_tex_src_sampler_handle : nir_tex_src_texture_handle) < 0)
               continue;
            if (!hk_apple9_handle_intrinsic(tex, sampler))
               return false;
            hk_apple9_use_descriptor_handle(&b, tex, sampler);
         }
      }
   }
   /* Handle loads were added and sources removed: liveness is stale. */
   nir_progress(true, nir_shader_get_entrypoint(nir), nir_metadata_control_flow);
   nir_opt_dce(nir);
   return true;
}

/* An early depth/stencil test updates the attachment before the fragment
 * shader runs, and the shader's input-attachment reads then observe this
 * fragment's own write. With a depth/stencil feedback loop, export the
 * interpolated depth at the end instead, so the test follows those reads. */
static void
hk_apple9_late_zs(nir_shader *nir)
{
   if (nir->info.fs.early_fragment_tests ||
       (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)))
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_store_output(&b, nir_load_frag_coord_z(&b), nir_imm_int(&b, 0),
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .src_type = nir_type_float32);
   nir->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* Hardware bounds shader-exported depth only to [0, 1]. Clamp depth exports,
 * including the side-effect pass's synthetic export, to the draw's clamp
 * range from the root table, as native Metal fragment shaders do with their
 * near/far uniforms.
 *
 * A NEVER depth test skips the fragment shader, which would drop side
 * effects, and a NaN export passes ALWAYS. Such draws therefore test LESS
 * (hk_cmd_apple9.c) against an exported depth above the storable range, so
 * every sample fails depth and still takes its stencil depth-fail op. */
static bool
hk_apple9_clamp_depth_export(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != FRAG_RESULT_DEPTH)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   unsigned offset = hk_root_descriptor_offset(draw.depth_clamp);
   nir_def *range = nir_load_ubo(b, 2, 32, nir_imm_int(b, 0),
                                 nir_imm_int(b, offset), .align_mul = 4,
                                 .range_base = offset, .range = 8);
   nir_def *z = intr->src[0].ssa;
   nir_def *clamped = nir_fmin(b, nir_fmax(b, z, nir_channel(b, range, 0)),
                               nir_channel(b, range, 1));
   unsigned never = hk_root_descriptor_offset(draw.force_never_in_shader);
   nir_def *fail = nir_load_ubo(b, 1, 16, nir_imm_int(b, 0),
                                nir_imm_int(b, never), .align_mul = 2,
                                .range_base = never, .range = 2);
   nir_src_rewrite(&intr->src[0],
                   nir_bcsel(b, nir_ine_imm(b, fail, 0),
                             nir_imm_float(b, 2.0), clamped));
   return true;
}

/* The Apple9 argument table carries one UBO pointing at Honeykrisp's
 * root descriptor table. Descriptor sets remain GPU addresses loaded from
 * that table; descriptor contents are never baked into compiled shaders. */
static bool
hk_apple9_lower_roots(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_depth_never_agx) {
      unsigned offset = hk_root_descriptor_offset(draw.force_never_in_shader);
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_load_ubo(b, 1, intr->def.bit_size,
         nir_imm_int(b, 0), nir_imm_int(b, offset), .align_mul = 2,
         .range_base = offset, .range = 2));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_draw_id) {
      unsigned offset = hk_root_descriptor_offset(draw.draw_id);
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_load_ubo(b, 1, 32,
         nir_imm_int(b, 0), nir_imm_int(b, offset), .align_mul = 4,
         .range_base = offset, .range = 4));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_base_vertex ||
       intr->intrinsic == nir_intrinsic_load_first_vertex ||
       intr->intrinsic == nir_intrinsic_load_base_instance) {
      unsigned offset = intr->intrinsic == nir_intrinsic_load_base_instance ? 4 : 0;
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *value = nir_load_ubo(b, 1, 32,
         nir_imm_int(b, HK_APPLE9_DRAW_PARAMS_BINDING), nir_imm_int(b, offset),
         .align_mul = 4, .range_base = offset, .range = 4);
      nir_def_replace(&intr->def, value);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
      if (nir_def_instr(intr->src[0].ssa)->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *handle = nir_def_as_intrinsic(intr->src[0].ssa);
      if (handle->intrinsic != nir_intrinsic_bindless_image_agx)
         return false;
      b->cursor = nir_before_instr(&intr->instr);
      unsigned slot = offsetof(struct hk_root_descriptor_table, sets) +
                      nir_intrinsic_desc_set(handle) * sizeof(uint64_t);
      nir_def *set = nir_load_ubo(b, 1, 64, nir_imm_int(b, 0),
                                 nir_imm_int(b, slot), .align_mul = 8,
                                 .range_base = slot, .range = 8);
      nir_def_replace(&intr->def,
         nir_iadd(b, set, nir_u2u64(b, handle->src[0].ssa)));
      return true;
   }

   unsigned offset;
   if (intr->intrinsic == nir_intrinsic_load_root_agx)
      offset = offsetof(struct hk_root_descriptor_table, root_desc_addr);
   else if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx)
      offset = offsetof(struct hk_root_descriptor_table, sets) +
               nir_intrinsic_desc_set(intr) * sizeof(uint64_t);
   else
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *ptr = nir_load_ubo(b, 1, 64, nir_imm_int(b, 0),
                              nir_imm_int(b, offset), .align_mul = 8,
                              .range_base = offset, .range = 8);
   nir_def_replace(&intr->def, ptr);
   return true;
}

static bool
lower_uniforms(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* Root is first, descriptor sets follow immediately. */
   unsigned *root_ = data;
   unsigned root = *root_;
   unsigned sets = root + 4;

   if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
      /* Change of units from sets to uniforms */
      nir_intrinsic_set_desc_set(intr,
                                 sets + (nir_intrinsic_desc_set(intr) * 4));
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_load_root_agx &&
       intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *rep;

   if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx) {
      unsigned s = nir_intrinsic_desc_set(intr);
      rep = nir_load_preamble(b, 1, 64, .base = sets + (4 * s));
   } else {
      rep = nir_load_preamble(b, 1, 64, .base = root);
   }

   nir_def_replace(&intr->def, rep);
   return true;
}

static bool
kill_psiz_write(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PSIZ)
      return false;

   if (nir_remove_sysval_output(intr, MESA_SHADER_FRAGMENT)) {
      b->shader->info.outputs_written &= ~VARYING_BIT_PSIZ;
   }

   return true;
}

static void
hk_lower_hw_vs(nir_shader *nir, struct hk_shader *shader, bool kill_psiz)
{
   /* Point size must be clamped, excessively large points don't render
    * properly on G13.
    *
    * Must be synced with pointSizeRange.
    */
   NIR_PASS(_, nir, nir_lower_point_size, 1.0f, 511.95f);

   if (kill_psiz) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, kill_psiz_write,
               nir_metadata_control_flow, NULL);
   }

   NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL, NULL);
   NIR_PASS(_, nir, agx_nir_lower_cull_distance_vs);

   NIR_PASS(_, nir, agx_nir_lower_uvs, &shader->info.uvs);
}

static VkResult
hk_copy_shader_constant_data(struct hk_device *dev, struct hk_shader *shader,
                             const nir_shader *nir)
{
   if (nir->constant_data_size > 0) {
      uint64_t data_size = ALIGN_POT((uint64_t)nir->constant_data_size,
                                     HK_MIN_UBO_ALIGNMENT);
      /* The shader cache stores data lengths in 32 bits. */
      if (data_size > UINT32_MAX)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      void *data = malloc(data_size);
      if (data == NULL)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      memcpy(data, nir->constant_data, nir->constant_data_size);

      assert(nir->constant_data_size <= data_size);
      memset(data + nir->constant_data_size, 0,
             data_size - nir->constant_data_size);

      shader->data_ptr = data;
      shader->data_size = data_size;
   }

   return VK_SUCCESS;
}

static VkResult
hk_compile_nir(struct hk_device *dev, const VkAllocationCallbacks *pAllocator,
               nir_shader *nir, VkShaderCreateFlagsEXT shader_flags,
               const struct vk_pipeline_robustness_state *rs,
               const union hk_key *key, enum hk_feature_key features,
               struct hk_shader *shader, mesa_shader_stage sw_stage, bool hw,
               nir_xfb_info *xfb_info, unsigned set_count)
{
   unsigned nr_vbos = 0;
   bool kill_psiz = false;

   /* For now, only shader objects are supported */
   if (sw_stage == MESA_SHADER_VERTEX) {
      nr_vbos = DIV_ROUND_UP(
         BITSET_LAST_BIT(shader->info.vs.attrib_components_read), 4);
      kill_psiz = key->vs.kill_psiz;
   } else if (sw_stage == MESA_SHADER_FRAGMENT) {
      shader->info.fs.writes_memory = nir->info.writes_memory;
      shader->info.fs.interp.linear = nir->info.linear_varyings;
      shader->info.fs.interp.flat =
         ~(nir->info.linear_varyings | nir->info.perspective_varyings);

      /* Discards must be lowering before lowering MSAA to handle discards */
      NIR_PASS(_, nir, agx_nir_lower_discard_zs_emit);
      NIR_PASS(_, nir, agx_nir_lower_fs_output_to_epilog,
               &shader->info.fs.epilog_key);
      NIR_PASS(_, nir, agx_nir_lower_sample_mask);

      if (nir->info.fs.uses_sample_shading) {
         /* Ensure the sample mask is preserved in register */
         nir_builder b =
            nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

         nir_def *mask =
            nir_load_exported_agx(&b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK);

         nir_export_agx(&b, mask, .base = AGX_ABI_FOUT_SAMPLE_MASK);

         NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
      }

      NIR_PASS(_, nir, agx_nir_lower_fs_active_samples_to_register);
      NIR_PASS(_, nir, agx_nir_lower_interpolation);
   } else if (sw_stage == MESA_SHADER_TESS_EVAL ||
              sw_stage == MESA_SHADER_TESS_CTRL) {

      shader->info.tess.info.ccw = nir->info.tess.ccw;
      shader->info.tess.info.points = nir->info.tess.point_mode;
      shader->info.tess.info.spacing = nir->info.tess.spacing;
      shader->info.tess.info.mode = nir->info.tess._primitive_mode;

      if (sw_stage == MESA_SHADER_TESS_CTRL) {
         shader->info.tess.tcs_output_patch_size =
            nir->info.tess.tcs_vertices_out;
         shader->info.tess.tcs_per_vertex_outputs =
            poly_tcs_per_vertex_outputs(nir);
         shader->info.tess.tcs_nr_patch_outputs =
            util_last_bit(nir->info.patch_outputs_written);
         shader->info.tess.tcs_output_stride = poly_tcs_output_stride(nir);
      } else {
         /* This destroys info so it needs to happen after the gather */
         NIR_PASS(_, nir, poly_nir_lower_tes, hw);
      }
   }

   /* Normally, vertex shaders need to write a default point size. However, if
    * we have a geometry/tessellation shader, the hardware vertex (software
    * GS/TES) will handle this itself instead.
    */
   if (sw_stage == MESA_SHADER_VERTEX && hw && !kill_psiz) {
      NIR_PASS(_, nir, nir_lower_default_point_size);
   }

   uint64_t outputs = nir->info.outputs_written;
   if (sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_TESS_EVAL) {
      if (hw) {
         hk_lower_hw_vs(nir, shader, kill_psiz);
      } else {
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));
         nir->xfb_info = NULL;
      }
   }

   if (dev->dev.chip == AGX_CHIP_G16G &&
       !hk_apple9_texture_bindings(nir)) {
      ralloc_free(nir);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apple9 texture operation without a descriptor handle");
   }

   if (dev->dev.chip == AGX_CHIP_G16G) {
      const char *reason = NULL;
      if (!agx_nir_lower_apple9_runtime_sampler_state(nir, &reason)) {
         ralloc_free(nir);
         return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT, "%s", reason);
      }
   }

   unsigned root = 0;
   if (sw_stage == MESA_SHADER_FRAGMENT)
      root = AGX_ABI_FUNI_ROOT;
   else if (sw_stage == MESA_SHADER_VERTEX)
      root = AGX_ABI_VUNI_COUNT_VK(nr_vbos);

   shader->info.set_count = set_count;

   /* XXX: rename */
   NIR_PASS(_, nir, hk_lower_uvs_index, sw_stage, nr_vbos);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            dev->dev.chip == AGX_CHIP_G16G ? hk_apple9_lower_roots : lower_uniforms,
            nir_metadata_control_flow, &root);

#if 0
   /* TODO */
   nir_variable_mode robust2_modes = 0;
   if (rs->uniform_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT)
      robust2_modes |= nir_var_mem_ubo;
   if (rs->storage_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT)
      robust2_modes |= nir_var_mem_ssbo;
#endif

   struct agx_shader_key backend_key = {
      .reserved_preamble = root + (4 * (1 + set_count)),
      .dev = agx_gather_device_key(&dev->dev),
      .no_stop = nir->info.stage == MESA_SHADER_FRAGMENT,
      .has_scratch = !nir->info.internal,
      .promote_constants = true,
      .promote_textures = true,
   };

   /* For now, sample shading is always dynamic. Indicate that. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.fs.uses_sample_shading)
      backend_key.fs.inside_sample_loop = true;

   simple_mtx_t *lock = NULL;
   if (agx_get_compiler_debug())
      lock = &hk_device_physical(dev)->debug_compile_lock;

   if (lock)
      simple_mtx_lock(lock);

   assert(nir->info.io_lowered);
   if (dev->dev.chip == AGX_CHIP_G16G) {
      hk_assert_final_push_mask(nir, shader);
      /* A compute entrypoint without side effects has no GPU work, even
       * when loads and arithmetic remain, for example once optimization has
       * proved every store unreachable. Keep the common Asahi empty-shader
       * marker so cached and freshly compiled shaders agree.
       */
      bool empty = sw_stage == MESA_SHADER_COMPUTE;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_barrier &&
                !(nir_intrinsic_infos[intr->intrinsic].flags &
                  NIR_INTRINSIC_CAN_ELIMINATE))
               empty = false;
         }
      }

      const char *reason = NULL;
      if (empty) {
         shader->b.info.stats.instrs = 1;
      } else if (sw_stage == MESA_SHADER_COMPUTE &&
                 agx_compile_apple9_tiny(nir, &shader->b, &shader->info.apple9,
                                         &reason)) {
         /* Compiled. */
      } else if (reason == agx_apple9_compute_without_effects) {
         /* Optimization removed every store: there is nothing to run. */
         memset(&shader->b, 0, sizeof(shader->b));
         memset(&shader->info.apple9, 0, sizeof(shader->info.apple9));
         shader->b.info.stats.instrs = 1;
      } else {
         if (agx_get_compiler_debug())
            fprintf(stderr, "Apple9 compute compilation failed: %s\n",
                    reason ?: "unsupported stage");
         if (lock)
            simple_mtx_unlock(lock);
         ralloc_free(nir);
         return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                          "Apple9 shader: %s", reason ? reason : "unsupported stage");
      }
   } else {
      agx_compile_shader_nir(nir, &backend_key, &shader->b);
   }

   if (lock)
      simple_mtx_unlock(lock);

   shader->code_ptr = shader->b.binary;
   shader->code_size = shader->b.info.binary_size;
   hk_gpu_busy_shader_source(shader->code_ptr, shader->code_size, nir->info.name,
                             nir->info.source_blake3, sw_stage);

   shader->info.stage = sw_stage;
   shader->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   shader->info.cull_distance_array_size = nir->info.cull_distance_array_size;
   shader->b.info.outputs = outputs;

   if (xfb_info) {
      assert(xfb_info->output_count < ARRAY_SIZE(shader->info.xfb_outputs));

      memcpy(&shader->info.xfb_info, xfb_info,
             nir_xfb_info_size(xfb_info->output_count));

      typed_memcpy(shader->info.xfb_stride, nir->info.xfb_stride, 4);
   }

   VkResult data_result = hk_copy_shader_constant_data(dev, shader, nir);
   if (data_result != VK_SUCCESS) {
      ralloc_free(nir);
      return data_result;
   }

   ralloc_free(nir);

   VkResult result = hk_init_link_ht(shader, sw_stage);
   if (result != VK_SUCCESS)
      return vk_error(dev, result);

   return hk_upload_shader(dev, shader);
}

static const struct vk_shader_ops hk_shader_ops;

static void
hk_destroy_linked_shader(struct hk_device *dev, struct hk_linked_shader *linked)
{
   agx_bo_unreference(&dev->dev, linked->b.bo);
   ralloc_free(linked);
}

#define HK_APPLE9_CACHED_VARIANTS_PER_SOURCE 64

struct hk_apple9_cached_variant {
   struct hk_apple9_cached_variant *next;
   struct vk_pipeline_cache_object *object;
};

struct hk_apple9_deferred {
   /* Serialized NIR after API lowering, before key-dependent compilation. */
   void *nir;
   size_t nir_size;
   uint32_t next_stage_mask;
   uint32_t set_layout_count;
   /* Static pipeline sample shading, which has no dynamic state. */
   bool sample_shading;
   /* Static depth/stencil feedback loop. The dynamic form reaches draw-time
    * variants through the dynamic feedback-loop state instead. */
   bool zs_self_dep;
   /* Vertex shader whose pipeline disables vertex input robustness. */
   bool vertex_unbounded;
   /* Color attachments the pipeline leaves undefined; writes are discarded
    * even when the render binds an image there. */
   uint8_t undefined_colors;
   /* A missing fragment stage needs a generated cull-distance consumer. */
   bool cull_distance;
   /* VkPipelineCache serializes these API shaders. Retain the variants seen
    * with this source so application-managed caches include draw-time work.
    * The list owns only serialized bytes, never other API shader objects. */
   simple_mtx_t variant_lock;
   struct hk_apple9_cached_variant *variants;
   unsigned variant_count;
};

static void
hk_apple9_remember_variant(struct hk_device *dev, struct hk_api_shader *api,
                            struct vk_pipeline_cache_object *object)
{
   struct hk_apple9_deferred *d = api->apple9_deferred;
   struct hk_apple9_cached_variant *evicted = NULL;
   simple_mtx_lock(&d->variant_lock);
   for (struct hk_apple9_cached_variant **p = &d->variants; *p;
        p = &(*p)->next) {
      if ((*p)->object == object) {
         struct hk_apple9_cached_variant *v = *p;
         *p = v->next;
         v->next = d->variants;
         d->variants = v;
         simple_mtx_unlock(&d->variant_lock);
         return;
      }
   }
   struct hk_apple9_cached_variant *v = malloc(sizeof(*v));
   if (v) {
      v->object = vk_pipeline_cache_object_ref(object);
      v->next = d->variants;
      d->variants = v;
      if (d->variant_count == HK_APPLE9_CACHED_VARIANTS_PER_SOURCE) {
         struct hk_apple9_cached_variant **tail = &d->variants;
         while ((*tail)->next)
            tail = &(*tail)->next;
         evicted = *tail;
         *tail = NULL;
      } else {
         ++d->variant_count;
      }
   }
   simple_mtx_unlock(&d->variant_lock);
   /* Cache serialization takes the runtime-cache lock before variant_lock.
    * A last unref takes that cache lock, so eviction must happen outside. */
   if (evicted) {
      vk_pipeline_cache_object_unref(&dev->vk, evicted->object);
      free(evicted);
   }
}

static bool
hk_apple9_serialize_variants(struct hk_apple9_deferred *d, struct blob *blob)
{
   intptr_t count_at = blob_reserve_uint32(blob);
   if (count_at < 0)
      return false;
   uint32_t count = 0;
   struct hk_apple9_cached_variant *variants[HK_APPLE9_CACHED_VARIANTS_PER_SOURCE];
   simple_mtx_lock(&d->variant_lock);
   for (struct hk_apple9_cached_variant *v = d->variants; v; v = v->next) {
      assert(count < ARRAY_SIZE(variants));
      variants[count++] = v;
   }
   /* Import remembers each entry at the head: write oldest first to preserve
    * recency when an application cache is loaded in another process. */
   for (unsigned i = count; i > 0; --i) {
      struct hk_apple9_cached_variant *v = variants[i - 1];
      struct vk_raw_data_cache_object *raw =
         container_of(v->object, struct vk_raw_data_cache_object, base);
      assert(v->object->key_size == BLAKE3_OUT_LEN);
      if (raw->data_size > UINT32_MAX) {
         blob->out_of_memory = true;
         break;
      }
      blob_write_bytes(blob, v->object->key_data, BLAKE3_OUT_LEN);
      blob_write_uint32(blob, raw->data_size);
      blob_write_bytes(blob, raw->data, raw->data_size);
   }
   simple_mtx_unlock(&d->variant_lock);
   blob_overwrite_uint32(blob, count_at, count);
   return !blob->out_of_memory;
}

static bool
hk_apple9_deserialize_variants(struct hk_device *dev,
                                struct hk_api_shader *api,
                                struct blob_reader *blob)
{
   uint32_t count = blob_read_uint32(blob);
   if (blob->overrun || count > (blob->end - blob->current) / (BLAKE3_OUT_LEN + 4))
      return false;
   for (unsigned i = 0; i < count; ++i) {
      const void *key = blob_read_bytes(blob, BLAKE3_OUT_LEN);
      uint32_t size = blob_read_uint32(blob);
      const void *data = blob_read_bytes(blob, size);
      if (blob->overrun || !size)
         return false;
      if (!dev->vk.mem_cache)
         continue;
      struct vk_pipeline_cache_object *object =
         vk_pipeline_cache_create_and_insert_object(
            dev->vk.mem_cache, key, BLAKE3_OUT_LEN, data, size,
            &vk_raw_data_cache_object_ops);
      if (object) {
         hk_apple9_remember_variant(dev, api, object);
         vk_pipeline_cache_object_unref(&dev->vk, object);
      }
   }
   return true;
}

static void hk_apple9_forget_deferred(struct hk_device *dev,
                                      struct hk_api_shader *obj);

static void
hk_shader_destroy(struct hk_device *dev, struct hk_shader *s)
{
   free((void *)s->code_ptr);
   free((void *)s->data_ptr);
   if (s->apple9_body)
      agx_shader_body_unreference(&dev->dev, s->apple9_body);
   else
      agx_bo_unreference(&dev->dev, s->bo);
   if (s->apple9_coefficients.size) {
      simple_mtx_lock(&dev->usc_heap_lock);
      hk_usc_heap_free(&dev->usc_heap, &s->apple9_coefficients);
      simple_mtx_unlock(&dev->usc_heap_lock);
   }
   free(s->apple9_launch);
   s->apple9_launch = NULL;
   if (s->apple9_stage_launch) {
      agx_apple9_stage_launch_cache_finish(s->apple9_stage_launch);
      free(s->apple9_stage_launch);
      s->apple9_stage_launch = NULL;
   }

   simple_mtx_destroy(&s->linked.lock);

   if (s->only_linked)
      hk_destroy_linked_shader(dev, s->only_linked);

   if (s->linked.ht) {
      hash_table_foreach(s->linked.ht, entry) {
         hk_destroy_linked_shader(dev, entry->data);
      }
      _mesa_hash_table_destroy(s->linked.ht, NULL);
   }
}

void
hk_api_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                      const VkAllocationCallbacks *pAllocator)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   if (obj->apple9_native) {
      assert(!obj->apple9_native->apple9_native);
      hk_api_shader_destroy(vk_dev, &obj->apple9_native->vk, pAllocator);
   }

   if (obj->apple9_deferred) {
      hk_apple9_forget_deferred(dev, obj);
      struct hk_apple9_cached_variant *v = obj->apple9_deferred->variants;
      while (v) {
         struct hk_apple9_cached_variant *next = v->next;
         vk_pipeline_cache_object_unref(&dev->vk, v->object);
         free(v);
         v = next;
      }
      simple_mtx_destroy(&obj->apple9_deferred->variant_lock);
      free(obj->apple9_deferred->nir);
      free(obj->apple9_deferred);
      vk_shader_free(&dev->vk, pAllocator, &obj->vk);
      return;
   }

   hk_foreach_variant(obj, shader) {
      hk_shader_destroy(dev, shader);
   }

   vk_shader_free(&dev->vk, pAllocator, &obj->vk);
}

struct hk_apple9_graphics_key;

/* Without a fragment stage nothing reads the rasterized varyings: keep only
 * the outputs the fixed-function rasterizer consumes, as other drivers do when
 * linking a position-only pre-rasterization stage. */
static bool
hk_apple9_strip_unread_output(nir_builder *b, nir_intrinsic_instr *intr,
                              UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   switch (nir_intrinsic_io_semantics(intr).location) {
   case VARYING_SLOT_POS:
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CLIP_DIST1:
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_CULL_DIST1:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_PRIMITIVE_SHADING_RATE:
      return false;
   default:
      nir_instr_remove(&intr->instr);
      return true;
   }
}

static void
hk_apple9_strip_unread_outputs(nir_shader *nir)
{
   if (!nir_shader_intrinsics_pass(nir, hk_apple9_strip_unread_output,
                                   nir_metadata_control_flow, NULL))
      return;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_dead_cf);
}

static VkResult
hk_apple9_compile_graphics(struct hk_device *dev, uint32_t count,
                           struct vk_shader_compile_info *infos,
                           const struct vk_graphics_pipeline_state *state,
                           const struct hk_apple9_graphics_key *key_in,
                           const uint64_t *fs_interp,
                           const VkAllocationCallbacks *alloc,
                           struct vk_shader **out);

VkResult
hk_compile_shader(struct hk_device *dev, struct vk_shader_compile_info *info,
                  const struct vk_graphics_pipeline_state *state,
                  const struct vk_features *vk_features,
                  const VkAllocationCallbacks *pAllocator,
                  struct hk_api_shader **shader_out)
{
   VkResult result;
   enum hk_feature_key features = hk_make_feature_key(vk_features);

   /* We consume the NIR, regardless of success or failure */
   nir_shader *nir = info->nir;
   if (dev->dev.chip == AGX_CHIP_G16G && info->stage == MESA_SHADER_GEOMETRY) {
      /* Internal geometry shaders, such as the transform feedback
       * passthrough, use the Apple9 geometry pipeline.
       */
      if (!nir->info.io_lowered) {
         hk_lower_nir(dev, nir, info->robustness, false, info->set_layout_count,
                      info->set_layouts, features);
      }
      struct vk_shader_compile_info gs_info = *info;
      struct vk_shader *out = NULL;
      VkResult result =
         hk_apple9_compile_graphics(dev, 1, &gs_info, state, NULL, NULL, pAllocator,
                                    &out);
      *shader_out = out ? container_of(out, struct hk_api_shader, vk) : NULL;
      return result;
   }
   if (dev->dev.chip == AGX_CHIP_G16G && info->stage != MESA_SHADER_COMPUTE) {
      ralloc_free(nir);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apple9 Honeykrisp currently supports compute shaders only");
   }

   size_t size = sizeof(struct hk_api_shader) +
                 sizeof(struct hk_shader) * hk_num_variants(info->stage);
   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, info->stage, pAllocator, size);

   if (obj == NULL) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   obj->pre_raster_reads = HK_READS_UNKNOWN;
   obj->reads_draw_id = hk_nir_reads_draw_id(nir);
   obj->push_constant_mask = hk_nir_push_constant_mask(nir);
#ifndef NDEBUG
   hk_foreach_variant(obj, variant)
      variant->debug_owner = obj;
#endif

   if (!nir->info.io_lowered) {
      hk_lower_nir(dev, nir, info->robustness, false, info->set_layout_count,
                   info->set_layouts, features);
   }

   mesa_shader_stage sw_stage = nir->info.stage;

   union hk_key key_tmp, *key = NULL;
   if (sw_stage == MESA_SHADER_FRAGMENT) {
      hk_populate_fs_key(&key_tmp.fs, state);
      key = &key_tmp;

      nir->info.fs.uses_sample_shading |= key->fs.force_sample_shading;

      /* Force late-Z for Z/S self-deps. TODO: There's probably a less silly way
       * to do this.
       */
      if (key->fs.zs_self_dep) {
         nir_builder b =
            nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
         nir_discard_if(&b, nir_imm_false(&b));
         nir->info.fs.uses_discard = true;
      }

      NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, false);
   } else if (sw_stage == MESA_SHADER_VERTEX) {
      hk_populate_vs_key(&key_tmp.vs, state);
      key = &key_tmp;
   } else if (sw_stage == MESA_SHADER_TESS_CTRL) {
      NIR_PASS(_, nir, poly_nir_lower_tcs, true);
   }

   /* Compile all variants up front */
   if (sw_stage == MESA_SHADER_GEOMETRY) {
      struct hk_shader *main_variant = hk_main_gs_variant(obj);
      struct hk_shader *count_variant = hk_count_gs_variant(obj);

      nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;

      NIR_PASS(_, nir, poly_nir_lower_gs, &count, &rast, &pre_gs,
               &count_variant->info.gs);

      agx_preprocess_nir(count);
      agx_preprocess_nir(rast);
      agx_preprocess_nir(pre_gs);

      struct hk_shader *shader = &obj->variants[HK_GS_VARIANT_RAST];
      hk_lower_hw_vs(rast, shader, false);
      shader->info.gs = count_variant->info.gs;
      main_variant->info.gs = count_variant->info.gs;

      struct {
         nir_shader *in;
         struct hk_shader *out;
      } variants[] = {
         {nir, hk_main_gs_variant(obj)},
         {pre_gs, hk_pre_gs_variant(obj)},
         {count, count_variant},
         {rast, &obj->variants[HK_GS_VARIANT_RAST]},
      };

      for (unsigned v = 0; v < ARRAY_SIZE(variants); ++v) {
         if (variants[v].in) {
            result =
               hk_compile_nir(dev, pAllocator, variants[v].in, info->flags,
                              info->robustness, NULL, features, variants[v].out,
                              sw_stage, true, NULL, info->set_layout_count);

            if (result != VK_SUCCESS) {
               hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
               ralloc_free(nir);
               ralloc_free(pre_gs);
               ralloc_free(count);
               ralloc_free(rast);
               return result;
            }
         }
      }
   } else if (sw_stage == MESA_SHADER_VERTEX ||
              sw_stage == MESA_SHADER_TESS_EVAL) {

      VkShaderStageFlags next_stage = info->next_stage_mask;

      /* Transform feedback is layered on top of geometry shaders. If there is
       * not a geometry shader in the pipeline, we will compile a geometry
       * shader for the purpose. Update the next_stage mask accordingly.
       */
      if (nir->xfb_info != NULL) {
         next_stage |= VK_SHADER_STAGE_GEOMETRY_BIT;
      }

      if (sw_stage == MESA_SHADER_VERTEX) {
         assert(
            !(nir->info.inputs_read & BITFIELD64_MASK(VERT_ATTRIB_GENERIC0)) &&
            "Fixed-function attributes not used in Vulkan");

         NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in);
      }

      /* Fold constant offset srcs for IO. */
      NIR_PASS(_, nir, nir_opt_constant_folding);

      for (enum hk_vs_variant v = 0; v < HK_VS_VARIANTS; ++v) {
         /* Only compile the software variant if we might use this shader with
          * geometry/tessellation. We need to compile the hardware variant
          * unconditionally to handle the VS -> null FS case, which does not
          * require setting the FRAGMENT bit.
          */
         if (v == HK_VS_VARIANT_SW &&
             !(next_stage & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                             VK_SHADER_STAGE_GEOMETRY_BIT)))
            continue;

         struct hk_shader *shader = &obj->variants[v];
         bool hw = v == HK_VS_VARIANT_HW;
         bool last = (v + 1) == HK_VS_VARIANTS;

         /* Each variant gets its own NIR. To save an extra clone, we use the
          * original NIR for the last stage.
          */
         nir_shader *clone = last ? nir : nir_shader_clone(NULL, nir);

         NIR_PASS(_, clone, agx_nir_gather_vs_inputs,
                  shader->info.vs.attrib_components_read);

         if (sw_stage == MESA_SHADER_VERTEX) {
            shader->info.vs.use_prolog = !(key && key->vs.skip_prolog);
            shader->info.vs.attribs_read =
               nir->info.inputs_read >> VERT_ATTRIB_GENERIC0;

            if (shader->info.vs.use_prolog) {
               NIR_PASS(_, clone, agx_nir_lower_vs_input_to_prolog);
            } else {
               struct agx_velem_key attribs[AGX_MAX_ATTRIBS];
               for (unsigned a = 0; a < AGX_MAX_ATTRIBS; ++a) {
                  if (key->vs.attribs[a].format) {
                     unsigned slot = util_bitcount64(
                        shader->info.vs.attribs_read & BITFIELD_MASK(a));

                     attribs[slot] = key->vs.attribs[a];
                  }
               }

               struct agx_robustness agx_rs = {
                  .soft_fault = agx_has_soft_fault(&dev->dev),

                  /* Correctly handling GPL + pipeline-robustness requires
                   * runtime changes, and I don't care enough to optimize this.
                   */
                  .level = AGX_ROBUSTNESS_D3D,
               };

               agx_nir_lower_vbo(clone, attribs, agx_rs,
                                 !key->vs.static_strides);

               unsigned nr = DIV_ROUND_UP(
                  BITSET_LAST_BIT(shader->info.vs.attrib_components_read), 4);
               agx_nir_lower_non_monolithic_uniforms(clone, nr);
            }
         }

         /* hk_compile_nir takes ownership of the clone */
         result =
            hk_compile_nir(dev, pAllocator, clone, info->flags,
                           info->robustness, key, features, shader, sw_stage,
                           hw, nir->xfb_info, info->set_layout_count);
         if (result != VK_SUCCESS) {
            hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
            ralloc_free(nir);
            return result;
         }
      }
   } else {
      struct hk_shader *shader = hk_only_variant(obj);

      /* hk_compile_nir takes ownership of nir */
      result = hk_compile_nir(dev, pAllocator, nir, info->flags,
                              info->robustness, key, features, shader, sw_stage,
                              true, NULL, info->set_layout_count);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   }

   hk_api_shader_init_constant_data(obj);
   *shader_out = obj;
   return VK_SUCCESS;
}

static void
nir_opts(nir_shader *nir, void *data)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);

      NIR_PASS(progress, nir, nir_opt_if, 0);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &(nir_opt_peephole_select_options){
                  .limit = 8,
                  .expensive_alu_ok = true,
                  .discard_ok = true,
               });

      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

/* Count fragment shader invocations for pipeline statistics queries. The
 * counter is behind a uniform test of the active query flags, so draws
 * without the query issue no atomics. With the query active, one lane adds
 * the wave's count: an atomic per fragment on the single counter serialized
 * every fragment of the draw, halving frame rates in games that keep
 * statistics queries open (Unigine Heaven through DXVK).
 */
static void
hk_apple9_lower_fs_statistics(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   unsigned offset = hk_root_descriptor_offset(draw.pipeline_stats_flags);
   nir_def *flags = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
                                 nir_imm_int(&b, offset), .align_mul = 4,
                                 .range_base = offset, .range = 4);
   /* The branches are uniform and usually not taken: keep if-conversion
    * from hoisting the coverage read and ballot into every invocation. */
   nir_if *active = nir_push_if(&b, nir_test_mask(&b, flags,
      VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT));
   active->control = nir_selection_control_dont_flatten;
   {
      nir_def *helper = nir_load_helper_invocation(&b, 1);
      nir_def *lanes = nir_ballot(&b, 1, 32, nir_inot(&b, helper));
      nir_if *counted = nir_push_if(&b, nir_inot(&b, helper));
      counted->control = nir_selection_control_dont_flatten;
      nir_push_if(&b, nir_elect(&b, 1));
      nir_def *addr = nir_load_stat_query_address_poly(
         &b, .base = PIPE_STAT_QUERY_PS_INVOCATIONS);
      nir_global_atomic(&b, 32, addr, nir_bit_count(&b, lanes),
                        .atomic_op = nir_atomic_op_iadd);
      nir_pop_if(&b, NULL);
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, hk_lower_uvs_index, MESA_SHADER_FRAGMENT, 0);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
}

/* Draw parameters come from the Apple9 draw-parameter UBO; the geometry
 * pipeline's poly sysvals live in the root table. Lower the former first so
 * hk_lower_uvs_index only sees the table-backed values.
 */
static void
hk_apple9_lower_prerast_sysvals(nir_shader *nir, mesa_shader_stage sw_stage)
{
   NIR_PASS(_, nir, poly_nir_lower_sysvals);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, hk_lower_uvs_index, sw_stage, 0);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
}

static VkResult
hk_apple9_finish_shader(struct hk_device *dev, struct hk_shader *shader,
                        const nir_shader *nir)
{
   shader->code_ptr = shader->b.binary;
   shader->code_size = shader->b.info.binary_size;
   shader->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   shader->info.cull_distance_array_size = nir->info.cull_distance_array_size;
   VkResult result = hk_copy_shader_constant_data(dev, shader, nir);
   return result == VK_SUCCESS ? hk_upload_shader(dev, shader) : result;
}

/* Compile a software pre-rasterization stage (vertex shader feeding a
 * geometry shader, or one of the geometry shader's compute variants) as an
 * Apple9 compute kernel. Does not take ownership of nir.
 */
static VkResult
hk_apple9_compile_prerast(struct hk_device *dev, nir_shader *nir,
                          struct hk_shader *shader, unsigned local_size,
                          const char **reason)
{
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.workgroup_size_variable = false;
   nir->info.workgroup_size[0] = local_size;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->xfb_info = NULL;
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_COMPUTE);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_dead_cf);
   NIR_PASS(_, nir, nir_opt_dce);

   hk_assert_final_push_mask(nir, shader);
   bool empty = true;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir))
      empty &= exec_list_is_empty(&block->instr_list);
   if (empty) {
      shader->b.info.stats.instrs = 1;
      return VK_SUCCESS;
   }

   if (!agx_compile_apple9_tiny(nir, &shader->b, &shader->info.apple9, reason)) {
      /* Static GS topology can leave the main program with nothing to store:
       * like an empty program, there is nothing to run. */
      if (*reason == agx_apple9_compute_without_effects) {
         memset(&shader->b, 0, sizeof(shader->b));
         memset(&shader->info.apple9, 0, sizeof(shader->info.apple9));
         shader->b.info.stats.instrs = 1;
         *reason = NULL;
         return VK_SUCCESS;
      }
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   return hk_apple9_finish_shader(dev, shader, nir);
}

static bool
hk_apple9_lower_sw_vertex_zero_base(nir_builder *b, nir_intrinsic_instr *intr,
                                    UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vertex_id_zero_base)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_channel(b, nir_load_global_invocation_id(b, 32), 0));
   return true;
}

/* The vertex shader in front of a geometry shader pulls its attributes and
 * indices in software, writing outputs to the vertex output buffer.
 */
static VkResult
hk_apple9_compile_sw_vs(struct hk_device *dev, nir_shader *nir,
                        const struct agx_apple9_vertex_layout *layout,
                        struct hk_shader *shader, const char **reason)
{
   if (!agx_nir_lower_apple9_vertex_inputs(nir, layout, true)) {
      *reason = "unsupported software vertex input";
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   uint64_t outputs = nir->info.outputs_written;
   NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
   NIR_PASS(_, nir, poly_nir_lower_sw_vs);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            hk_apple9_lower_sw_vertex_zero_base, nir_metadata_control_flow,
            NULL);
   VkResult result = hk_apple9_compile_prerast(dev, nir, shader, 64, reason);
   shader->b.info.outputs = outputs;
   return result;
}

static void
hk_apple9_gather_tess_info(struct hk_shader *shader, const nir_shader *nir)
{
   shader->info.tess.info.ccw = nir->info.tess.ccw;
   shader->info.tess.info.points = nir->info.tess.point_mode;
   shader->info.tess.info.spacing = nir->info.tess.spacing;
   shader->info.tess.info.mode = nir->info.tess._primitive_mode;
}

/* Tessellation control shaders run as compute kernels with one workgroup per
 * patch and one thread per output vertex, writing the patch buffer read by
 * the tessellator. Does not take ownership of nir.
 */
static VkResult
hk_apple9_compile_tcs(struct hk_device *dev, nir_shader *nir,
                      struct hk_shader *shader, const char **reason)
{
   hk_apple9_gather_tess_info(shader, nir);
   NIR_PASS(_, nir, poly_nir_lower_tcs, true);
   shader->info.tess.tcs_output_patch_size = nir->info.tess.tcs_vertices_out;
   shader->info.tess.tcs_per_vertex_outputs = poly_tcs_per_vertex_outputs(nir);
   shader->info.tess.tcs_nr_patch_outputs =
      util_last_bit(nir->info.patch_outputs_written);
   shader->info.tess.tcs_output_stride = poly_tcs_output_stride(nir);
   return hk_apple9_compile_prerast(dev, nir, shader,
                                    nir->info.tess.tcs_vertices_out, reason);
}

/* Cull distances reach the fragment shader as one linear value per plane,
 * zero where the distance is negative, when the interface has room for them:
 * that needs no geometry stage. Interfaces at the publication limit instead
 * publish one flat word of negative-plane bits per vertex, and a geometry
 * stage combines it over each primitive's vertices (hk_handle_passthrough_gs);
 * points use it directly. Sets *cull_mask to the transport used.
 */
static bool
hk_apple9_compile_vertex_culls(nir_shader *nir,
                               const struct agx_apple9_vertex_layout *vertex,
                               struct agx_shader_part *out, bool *cull_mask,
                               const char **reason)
{
   struct agx_apple9_vertex_layout layout = *vertex;
   layout.cull_distance_mask = *cull_mask = false;
   if (!nir->info.cull_distance_array_size)
      return agx_compile_apple9_vertex_inputs(nir, &layout, out, reason);

   nir_shader *planes = nir_shader_clone(ralloc_parent(nir), nir);
   if (agx_compile_apple9_vertex_inputs(planes, &layout, out, NULL)) {
      nir_shader_replace(nir, planes);
      return true;
   }
   ralloc_free(planes);
   free(out->binary);
   memset(out, 0, sizeof(*out));
   layout.cull_distance_mask = *cull_mask = true;
   return agx_compile_apple9_vertex_inputs(nir, &layout, out, reason);
}

DEBUG_GET_ONCE_BOOL_OPTION(tess_mesh_probe, "HK_APPLE9_TESS_MESH_PROBE", false)

bool
hk_apple9_compile_tes_mesh_program(
   const nir_shader *source, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   nir_shader *mesh = hk_nir_tes_mesh(source, reason);
   if (!mesh)
      return false;
   hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
   bool compiled =
      hk_apple9_compile_native_mesh_program(mesh, key, out, info, reason);
   ralloc_free(mesh);
   return compiled;
}

/* Evaluation shaders run as the hardware vertex program over the tessellated
 * domain, and as a software vertex program when a geometry shader or
 * transform feedback follows. Does not take ownership of nir.
 */
static VkResult
hk_apple9_compile_tes(struct hk_device *dev, nir_shader *nir,
                      const struct agx_apple9_vertex_layout *vertex,
                      struct hk_api_shader *obj, VkShaderStageFlags next,
                      const char **reason)
{
   /* Compile an immutable diagnostic clone; discard all probe code. */
   if (debug_get_option_tess_mesh_probe()) {
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info body = {0};
      const char *why = NULL;
      const struct hk_apple9_graphics_key key = {.vertex = *vertex};
      bool compiled = hk_apple9_compile_tes_mesh_program(
         nir, &key, &part, &body, &why);
      fprintf(stderr, "HK_TESS_MESH_PROBE compiled=%u bytes=%u submitted=0 reason=%s\n",
              compiled, part.info.binary_size, why ? why : "supported");
      free(part.binary);
   }
   struct hk_shader *hw = &obj->variants[HK_VS_VARIANT_HW];
   struct hk_shader *sw = &obj->variants[HK_VS_VARIANT_SW];
   hk_apple9_gather_tess_info(hw, nir);
   hk_apple9_gather_tess_info(sw, nir);
   sw->info.set_count = hw->info.set_count;
   sw->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   sw->info.cull_distance_array_size = nir->info.cull_distance_array_size;

   /* The software variant feeds a geometry stage: the application's, or a
    * passthrough for transform feedback or a cull mask. */
   nir_shader *clone = nir_shader_clone(NULL, nir);

   struct agx_apple9_vertex_layout layout = *vertex;
   layout.rasterize_points = nir->info.tess.point_mode;
   layout.ignore_point_size = !layout.rasterize_points;
   NIR_PASS(_, nir, poly_nir_lower_tes, true);
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_TESS_EVAL);
   if (agx_apple9_debug_enabled(AGX_A9_DBG_NIR))
      nir_print_shader(nir, stderr);
   bool cull_mask;
   hk_assert_final_push_mask(nir, hw);
   if (!hk_apple9_compile_vertex_culls(nir, &layout, &hw->b, &cull_mask,
                                       reason)) {
      ralloc_free(clone);
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   hw->info.cull_mask = sw->info.cull_mask = cull_mask;

   VkResult result = VK_SUCCESS;
   if ((next & VK_SHADER_STAGE_GEOMETRY_BIT) || clone->xfb_info || cull_mask) {
      if (clone->xfb_info) {
         memcpy(&sw->info.xfb_info, clone->xfb_info,
                nir_xfb_info_size(clone->xfb_info->output_count));
         typed_memcpy(sw->info.xfb_stride, clone->info.xfb_stride, 4);
      }
      NIR_PASS(_, clone, poly_nir_lower_tes, false);
      uint64_t outputs = clone->info.outputs_written;
      NIR_PASS(_, clone, poly_nir_lower_vs_before_gs);
      result = hk_apple9_compile_prerast(dev, clone, sw, 64, reason);
      sw->b.info.outputs = outputs;
   }
   ralloc_free(clone);
   if (result != VK_SUCCESS)
      return result;
   return hk_apple9_finish_shader(dev, hw, nir);
}

/* Application geometry shaders emitting triangle strips under polygon mode
 * POINT: every output store is tapped into the current vertex, the strip's two
 * previous vertices are kept, and each completed triangle is drawn as points
 * by hk_apple9_pp_emit_triangle instead of being emitted.
 */
struct hk_gs_pp {
   /* Per slot (clip/cull distances per element), the strip's two previous
    * vertices and the current one, as 32-bit words. */
   nir_variable *v[3][VARYING_SLOT_MAX + 8];
   nir_def *odd;
};

static unsigned
hk_gs_pp_key(unsigned location, unsigned offset)
{
   return location == VARYING_SLOT_CLIP_DIST0 ? VARYING_SLOT_MAX + offset
                                              : location;
}

static nir_def *
hk_gs_pp_load(nir_builder *b, void *data, unsigned vertex, unsigned location,
              unsigned offset, unsigned component, unsigned comps,
              nir_alu_type type)
{
   struct hk_gs_pp *pp = data;
   unsigned k = hk_gs_pp_key(location, offset);
   /* Odd strip triangles are wound (i + 1, i, i + 2). */
   nir_def *value;
   if (vertex == 2) {
      value = nir_load_var(b, pp->v[2][k]);
   } else {
      nir_def *a = nir_load_var(b, pp->v[vertex][k]);
      nir_def *c = nir_load_var(b, pp->v[vertex ^ 1][k]);
      value = nir_bcsel(b, pp->odd, c, a);
   }
   return nir_channels(b, value, BITFIELD_RANGE(component, comps));
}

/* Whether a geometry shader's triangle strips can be drawn as polygon
 * points: constant-offset 32-bit stream-0 outputs and an output count within
 * the geometry shader's vertex limit. */
static bool
hk_apple9_gs_polygon_points_ok(const nir_shader *gs)
{
   if (gs->info.gs.output_primitive != MESA_PRIM_TRIANGLE_STRIP)
      return false;
   unsigned per_triangle = 3 + 3 + gs->info.clip_distance_array_size;
   unsigned triangles =
      gs->info.gs.vertices_out > 2 ? gs->info.gs.vertices_out - 2 : 0;
   if (triangles * per_triangle > 1024)
      return false;
   nir_foreach_function_impl(impl, gs) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            if (!nir_src_is_const(intr->src[1]) ||
                nir_src_bit_size(intr->src[0]) != 32 ||
                nir_intrinsic_io_semantics(intr).gs_streams ||
                nir_intrinsic_io_semantics(intr).location >= VARYING_SLOT_MAX)
               return false;
         }
      }
   }
   return true;
}

static void
hk_apple9_lower_gs_polygon_points(nir_shader *gs, struct hk_pp_info info,
                                  bool point_size)
{
   assert(hk_apple9_gs_polygon_points_ok(gs));
   unsigned per_triangle = 3 + 3 + info.clip;
   unsigned triangles =
      gs->info.gs.vertices_out > 2 ? gs->info.gs.vertices_out - 2 : 0;

   nir_function_impl *impl = nir_shader_get_entrypoint(gs);
   unsigned components[VARYING_SLOT_MAX] = {0};
   uint64_t written = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned loc = nir_intrinsic_io_semantics(intr).location;
         written |= BITFIELD64_BIT(loc);
         components[loc] = MAX2(components[loc],
                                nir_intrinsic_component(intr) +
                                   util_last_bit(nir_intrinsic_write_mask(intr)));
      }
   }

   struct hk_gs_pp pp = {0};
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   for (unsigned k = 0; k < VARYING_SLOT_MAX + 8; ++k) {
      bool used = k < VARYING_SLOT_MAX
                     ? (written & BITFIELD64_BIT(k)) && k != VARYING_SLOT_CLIP_DIST0
                     : (written & VARYING_BIT_CLIP_DIST0) &&
                          k - VARYING_SLOT_MAX < DIV_ROUND_UP(info.clip + info.cull, 4);
      if (!used)
         continue;
      for (unsigned i = 0; i < 3; ++i) {
         pp.v[i][k] = nir_local_variable_create(impl, glsl_uvec4_type(), NULL);
         nir_store_var(&b, pp.v[i][k], nir_imm_ivec4(&b, 0, 0, 0, 0), 0xf);
      }
   }
   nir_variable *count =
      nir_local_variable_create(impl, glsl_uint_type(), "pp_strip");
   nir_store_var(&b, count, nir_imm_int(&b, 0), 1);

   info.outputs = written &
      ~(VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 |
        VARYING_BIT_CULL_DIST0 | VARYING_BIT_CULL_DIST1 |
        VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_BOUNDING_BOX0 |
        VARYING_BIT_BOUNDING_BOX1);
   /* Without polygonModePointSize, polygon points have size 1. */
   if (!point_size)
      info.outputs &= ~VARYING_BIT_PSIZ;
   info.output_components = components;

   /* Collect first: the triangle emission adds control flow and its own
    * emit_vertex intrinsics. */
   struct util_dynarray todo;
   util_dynarray_init(&todo, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         if (op == nir_intrinsic_store_output || op == nir_intrinsic_emit_vertex ||
             op == nir_intrinsic_end_primitive)
            util_dynarray_append(&todo, nir_instr_as_intrinsic(instr));
      }
   }
   util_dynarray_foreach(&todo, nir_intrinsic_instr *, it) {
      nir_intrinsic_instr *intr = *it;
      nir_instr *instr = &intr->instr;
      b.cursor = nir_before_instr(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_store_output: {
         unsigned k = hk_gs_pp_key(nir_intrinsic_io_semantics(intr).location,
                                   nir_src_as_uint(intr->src[1]));
         if (!pp.v[2][k]) {
            nir_instr_remove(instr);
            break;
         }
         unsigned first = nir_intrinsic_component(intr);
         nir_def *value = nir_load_var(&b, pp.v[2][k]);
         u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
            value = nir_vector_insert_imm(
               &b, value, nir_channel(&b, intr->src[0].ssa, c), first + c);
         }
         nir_store_var(&b, pp.v[2][k], value, 0xf);
         nir_instr_remove(instr);
         break;
      }
      case nir_intrinsic_end_primitive:
         nir_store_var(&b, count, nir_imm_int(&b, 0), 1);
         nir_instr_remove(instr);
         break;
      case nir_intrinsic_emit_vertex: {
         nir_def *n = nir_load_var(&b, count);
         nir_push_if(&b, nir_uge_imm(&b, n, 2));
         {
            pp.odd = nir_ine_imm(&b, nir_iand_imm(&b, n, 1), 0);
            /* Strip triangle i provokes from vertex i, or i + 2. */
            nir_def *last = nir_ine_imm(&b, nir_load_provoking_last(&b), 0);
            nir_def *provoking = nir_bcsel(
               &b, last, nir_imm_int(&b, 2),
               nir_bcsel(&b, pp.odd, nir_imm_int(&b, 1), nir_imm_int(&b, 0)));
            hk_apple9_pp_emit_triangle(&b, &info, hk_gs_pp_load, &pp,
                                       provoking, nir_load_primitive_id(&b), NULL);
         }
         nir_pop_if(&b, NULL);
         for (unsigned k = 0; k < VARYING_SLOT_MAX + 8; ++k) {
            if (!pp.v[2][k])
               continue;
            nir_store_var(&b, pp.v[0][k], nir_load_var(&b, pp.v[1][k]), 0xf);
            nir_store_var(&b, pp.v[1][k], nir_load_var(&b, pp.v[2][k]), 0xf);
         }
         nir_store_var(&b, count, nir_iadd_imm(&b, n, 1), 1);
         nir_instr_remove(instr);
         break;
      }
      default:
         break;
      }
   }
   util_dynarray_fini(&todo);

   gs->info.gs.output_primitive = MESA_PRIM_POINTS;
   gs->info.gs.vertices_out = MAX2(triangles * per_triangle, 1);
   gs->info.outputs_written &= ~VARYING_BIT_PRIMITIVE_ID;
   if (!point_size)
      gs->info.outputs_written &= ~VARYING_BIT_PSIZ;
   if (info.face_slot)
      gs->info.outputs_written |= BITFIELD64_BIT(info.face_slot);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, gs, nir_lower_vars_to_ssa);
}

/* Geometry shaders run as compute kernels writing an output vertex buffer,
 * followed by an ordinary hardware vertex program that rasterizes it.
 * Does not take ownership of nir.
 */
DEBUG_GET_ONCE_BOOL_OPTION(gs_mesh_probe, "HK_APPLE9_GS_MESH_PROBE", false)

bool
hk_apple9_compile_gs_mesh_program(
   const nir_shader *source, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   nir_shader *mesh = hk_nir_gs_mesh(source, reason);
   if (!mesh)
      return false;
   hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
   bool compiled = hk_apple9_compile_native_mesh_program(mesh, key, out, info, reason);
   ralloc_free(mesh);
   return compiled;
}

static VkResult
hk_apple9_compile_gs(struct hk_device *dev, nir_shader *nir,
                     const struct agx_apple9_vertex_layout *vertex,
                     bool cull_mask, struct hk_api_shader *obj,
                     const char **reason)
{
   /* This diagnostic probe does not select a submission path. The separate
    * default-off GS option and draw predicate own native routing. */
   if (debug_get_option_gs_mesh_probe()) {
      struct agx_shader_part part = {0};
      struct agx_apple9_mesh_body_info body = {0};
      const char *why = NULL;
      const struct hk_apple9_graphics_key key = {.vertex = *vertex};
      bool compiled = hk_apple9_compile_gs_mesh_program(
         nir, &key, &part, &body, &why);
      fprintf(stderr, "HK_GS_MESH_PROBE compiled=%u bytes=%u submitted=0 reason=%s\n",
              compiled, part.info.binary_size, why ? why : "supported");
      free(part.binary);
   }
   struct hk_shader *rast_shader = &obj->variants[HK_GS_VARIANT_RAST];
   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info = {0};
   if (cull_mask)
      NIR_PASS(_, nir, poly_nir_lower_gs_cull_mask, &count, &rast, &pre, &info);
   else
      NIR_PASS(_, nir, poly_nir_lower_gs, &count, &rast, &pre, &info);

   hk_foreach_variant(obj, variant) {
      variant->info.gs = info;
      variant->info.set_count = rast_shader->info.set_count;
   }

   VkResult result =
      hk_apple9_compile_prerast(dev, nir, hk_main_gs_variant(obj), 64, reason);
   if (result == VK_SUCCESS && count)
      result = hk_apple9_compile_prerast(dev, count, hk_count_gs_variant(obj),
                                         64, reason);
   if (result == VK_SUCCESS && pre)
      result = hk_apple9_compile_prerast(dev, pre, hk_pre_gs_variant(obj), 1,
                                         reason);
   if (result == VK_SUCCESS) {
      struct agx_apple9_vertex_layout layout = *vertex;
      layout.cull_distance_mask = cull_mask;
      layout.rasterize_points = info.mode == MESA_PRIM_POINTS;
      layout.ignore_point_size = !layout.rasterize_points;
      hk_apple9_lower_prerast_sysvals(rast, MESA_SHADER_GEOMETRY);
      if (agx_apple9_debug_enabled(AGX_A9_DBG_NIR))
         nir_print_shader(rast, stderr);
      hk_assert_final_push_mask(rast, rast_shader);
      if (agx_compile_apple9_vertex_inputs(rast, &layout, &rast_shader->b,
                                           reason))
         result = hk_apple9_finish_shader(dev, rast_shader, rast);
      else
         result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return result;
}

/* Mesh shaders run as compute kernels writing records, which a generated
 * vertex shader rasterizes.
 */
static VkResult
hk_apple9_compile_mesh(struct hk_device *dev, nir_shader *nir,
                       const struct agx_apple9_vertex_layout *vertex,
                       bool clip_negative_one_to_one,
                       const struct hk_per_vertex_copies *copies,
                       struct hk_apple9_stipple_slots stipple,
                       const struct hk_pp_info *polygon_points,
                       bool point_size, struct hk_api_shader *obj,
                       const char **reason)
{
   struct hk_shader *main = &obj->variants[HK_MESH_VARIANT_MAIN];
   struct hk_shader *rast = &obj->variants[HK_MESH_VARIANT_RAST];
   struct hk_mesh_info info;
   hk_mesh_gather_info(nir, &info);
   info.polygon_points = polygon_points != NULL;

   hk_foreach_variant(obj, variant) {
      variant->info.mesh = info;
      variant->info.set_count = main->info.set_count;
   }

   nir_shader *rast_nir = hk_nir_mesh_rast(nir, &info, copies, polygon_points,
                                           point_size);

   hk_nir_lower_mesh_main(nir, &info);
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->xfb_info = NULL;
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_COMPUTE);
   NIR_PASS(_, nir, nir_opt_dce);
   VkResult result = VK_SUCCESS;
   hk_assert_final_push_mask(nir, main);
   if (!agx_compile_apple9_tiny(nir, &main->b, &main->info.apple9, reason))
      result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   else
      result = hk_apple9_finish_shader(dev, main, nir);

   if (result == VK_SUCCESS) {
      struct agx_apple9_vertex_layout layout = *vertex;
      layout.rasterize_points =
         info.prim == MESA_PRIM_POINTS || info.polygon_points;
      layout.ignore_point_size = !layout.rasterize_points;
      if (stipple.start)
         hk_apple9_write_line_stipple_start(rast_nir, stipple);
      hk_apple9_lower_prerast_sysvals(rast_nir, MESA_SHADER_VERTEX);
      /* The generated vertex shader writes the rasterized position. */
      if (clip_negative_one_to_one) {
         NIR_PASS(_, rast_nir, nir_lower_clip_halfz_dynamic);
         NIR_PASS(_, rast_nir, nir_shader_intrinsics_pass,
                  hk_apple9_lower_clip_z_coeff, nir_metadata_control_flow,
                  NULL);
      }
      if (agx_apple9_debug_enabled(AGX_A9_DBG_NIR))
         nir_print_shader(rast_nir, stderr);
      hk_assert_final_push_mask(rast_nir, rast);
      if (agx_compile_apple9_vertex_inputs(rast_nir, &layout, &rast->b, reason))
         result = hk_apple9_finish_shader(dev, rast, rast_nir);
      else
         result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   ralloc_free(rast_nir);
   return result;
}

/* Compile the native program from the same graphics key used by the linked
 * pipeline. Keep key translation with the compilation so hk tests exercise
 * the production call boundary before device upload. */
static struct agx_apple9_native_mesh_options
hk_apple9_native_mesh_options(nir_shader *nir,
                              const struct hk_apple9_graphics_key *key)
{
   bool rounds = hk_apple9_native_mesh_round_shape_eligible(nir);
   bool packed = (key->native_index || hk_apple9_native_mesh_enabled()) &&
                 hk_apple9_native_mesh_packed_shape_eligible(nir);
   return (struct agx_apple9_native_mesh_options){
      .physical_threads = rounds ? AGX_APPLE9_NATIVE_SERIAL_THREADS
                                 : hk_apple9_native_mesh_threads(nir->info.workgroup_size),
      .clip_halfz = key->clip_negative_one_to_one,
      .attributes = (rounds || !packed) && hk_apple9_native_mesh_attributes_enabled(),
      .primitive_broadcast = !rounds && !packed && hk_apple9_native_mesh_primitive_lowering_enabled(),
      .packed_index_publication = rounds || packed,
      .publication_rounds = rounds,
      .link_clip_distances = key->fs_reads_clip_distance,
      .outputs_flat = key->vertex.outputs_flat,
      .outputs_linear = key->vertex.outputs_linear,
   };
}

/* Native incompatibility is a capability result, not a Vulkan error. Run the
 * actual lowering on a clone: some rejection sites occur after NIR mutation.
 * This runs at pipeline/library/shader creation even when final compilation
 * is deferred, so a shader's shape cannot first fail at EndCommandBuffer.
 */
VkResult
hk_apple9_native_mesh_check(nir_shader *nir,
                           const struct hk_apple9_graphics_key *key,
                           bool *supported, const char **reason)
{
   *supported = false;
   if (reason)
      *reason = NULL;
   if (!hk_apple9_native_mesh_enabled() &&
       !(key->native_index && hk_apple9_native_mesh_packed_shape_eligible(nir)))
      return VK_SUCCESS;

   struct agx_apple9_native_mesh_options options =
      hk_apple9_native_mesh_options(nir, key);
   if (!options.physical_threads) {
      if (reason)
         *reason = "native mesh supports 1..128 logical invocations";
      return VK_SUCCESS;
   }

   nir_shader *probe = nir_shader_clone(NULL, nir);
   if (!probe)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct agx_apple9_mesh_body_info info;
   *supported = agx_nir_lower_apple9_native_mesh(probe, true, &options,
                                                &info, reason);
   ralloc_free(probe);
   return VK_SUCCESS;
}

bool
hk_apple9_compile_native_mesh_program(
   nir_shader *nir, const struct hk_apple9_graphics_key *key,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   const struct agx_apple9_native_mesh_options options =
      hk_apple9_native_mesh_options(nir, key);
   if (!options.physical_threads) {
      if (reason)
         *reason = "native mesh supports 1..128 logical invocations";
      return false;
   }
   return agx_compile_apple9_native_mesh(
      nir, hk_apple9_native_mesh_enabled() ||
           (key->native_index && options.packed_index_publication),
      &options, out, info, reason);
}

/* GS capability is established on an immutable clone before normal
 * fallback lowering. Require an explicitly selected packed class even at P1;
 * enabling GS routing does not inherit old lane-0 qualification. */
DEBUG_GET_ONCE_BOOL_OPTION(gs_native_log, "HK_APPLE9_GS_ROUTE_LOG", false)

static VkResult
hk_apple9_native_gs_check(const nir_shader *source,
                         const struct hk_apple9_graphics_key *key,
                         bool *supported, const char **reason)
{
   *supported = false;
   bool production = key->native_stages & HK_APPLE9_NATIVE_GS_BIT;
   if (!production && !hk_apple9_native_gs_enabled())
      return VK_SUCCESS;
   /* Reject common declaration/interface exclusions before cloning NIR. The
    * normal lowering still validates every instruction in admitted sources. */
   if (production &&
       (source->info.gs.input_primitive != MESA_PRIM_TRIANGLES ||
        source->info.gs.output_primitive != MESA_PRIM_TRIANGLE_STRIP ||
        source->info.gs.invocations != 1 || source->info.gs.active_stream_mask != 1 ||
        source->info.gs.vertices_out < 3 || source->info.gs.vertices_out > 4 ||
        source->info.outputs_written != VARYING_BIT_POS ||
        (source->info.inputs_read & ~VARYING_BIT_POS) ||
        source->info.clip_distance_array_size || source->info.cull_distance_array_size ||
        source->xfb_info || source->info.writes_memory)) {
      if (reason)
         *reason = "GS declaration/interface is outside the admitted stage contract";
      return VK_SUCCESS;
   }
   const char *local_reason = NULL;
   if (!reason)
      reason = &local_reason;
   /* Early admission precedes graphics root lowering. Normalize an owned
    * clone just as the real graphics path does, so descriptor-backed counts
    * are checked as the loads that will reach the native backend. */
   nir_shader *normalized = nir_shader_clone(NULL, source);
   if (!normalized)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   NIR_PASS(_, normalized, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
   nir_shader *mesh = hk_nir_gs_mesh(normalized, reason);
   ralloc_free(normalized);
   if (!mesh) {
      if (debug_get_option_gs_native_log())
         fprintf(stderr, "HK_GS_CAP accepted=0 reason=%s\n", *reason ?: "lowering");
      return VK_SUCCESS;
   }
   VkResult result = VK_SUCCESS;
   if (hk_apple9_native_mesh_packed_shape_eligible(mesh) &&
       (!production || hk_apple9_native_stage_class(
          source->info.stage, mesh->info.workgroup_size[0],
          mesh->info.mesh.max_vertices_out, mesh->info.mesh.max_primitives_out))) {
      hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
      result = hk_apple9_native_mesh_check(mesh, key, supported, reason);
   } else {
      *reason = "GS declaration is outside the admitted stage classes";
   }
   if (debug_get_option_gs_native_log())
      fprintf(stderr, "HK_GS_CAP accepted=%u reason=%s\n", *supported, *reason ?: "supported");
   ralloc_free(mesh);
   return result;
}

DEBUG_GET_ONCE_BOOL_OPTION(tess_native_log, "HK_APPLE9_TESS_ROUTE_LOG", false)

static VkResult
hk_apple9_native_tess_check(const nir_shader *source,
                            const struct hk_apple9_graphics_key *key,
                            bool *supported, const char **reason)
{
   *supported = false;
   bool production = key->native_stages & HK_APPLE9_NATIVE_TES_BIT;
   if (!production && !hk_apple9_native_tess_enabled())
      return VK_SUCCESS;
   if (production &&
       ((source->info.tess._primitive_mode != TESS_PRIMITIVE_TRIANGLES &&
         source->info.tess._primitive_mode != TESS_PRIMITIVE_QUADS) ||
        source->info.tess.point_mode || source->info.patch_inputs_read ||
        source->info.outputs_written != VARYING_BIT_POS ||
        (source->info.inputs_read & ~(VARYING_BIT_POS | VARYING_BIT_TESS_LEVEL_INNER |
                                     VARYING_BIT_TESS_LEVEL_OUTER)) ||
        source->info.clip_distance_array_size || source->info.cull_distance_array_size ||
        source->xfb_info || source->info.writes_memory)) {
      if (reason)
         *reason = "TES declaration/interface is outside the admitted stage contract";
      return VK_SUCCESS;
   }
   const char *local_reason = NULL;
   if (!reason)
      reason = &local_reason;
   /* Early admission precedes graphics root lowering. Normalize an owned
    * clone just as the real graphics path does, so descriptor-backed counts
    * are checked as the loads that will reach the native backend. */
   nir_shader *normalized = nir_shader_clone(NULL, source);
   if (!normalized)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   NIR_PASS(_, normalized, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
            nir_metadata_control_flow, NULL);
   nir_shader *mesh = hk_nir_tes_mesh(normalized, reason);
   ralloc_free(normalized);
   if (!mesh) {
      if (debug_get_option_tess_native_log())
         fprintf(stderr, "HK_TESS_CAP accepted=0 reason=%s\n",
                 *reason ?: "lowering");
      return VK_SUCCESS;
   }
   VkResult result = VK_SUCCESS;
   if (hk_apple9_native_mesh_packed_shape_eligible(mesh) &&
       (!production || hk_apple9_native_stage_class(
          source->info.stage, mesh->info.workgroup_size[0],
          mesh->info.mesh.max_vertices_out, mesh->info.mesh.max_primitives_out))) {
      hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
      result = hk_apple9_native_mesh_check(mesh, key, supported, reason);
   } else {
      *reason = "TES declaration is outside the admitted stage classes";
   }
   if (debug_get_option_tess_native_log())
      fprintf(stderr, "HK_TESS_CAP accepted=%u reason=%s\n", *supported,
              *reason ?: "supported");
   ralloc_free(mesh);
   return result;
}

/* A native mesh API shader owns two ordinary compiled programs. The
 * generated object publishes the draw's mesh grid; it is not an API task shader.
 * Both BOs use the existing shader cache, upload and destruction paths. */
static VkResult
hk_apple9_compile_native_mesh(struct hk_device *dev, nir_shader *nir,
                               struct hk_api_shader *obj,
                               const struct hk_apple9_graphics_key *key,
                               const char **reason, bool *unsupported)
{
   struct hk_shader *mesh = hk_native_producer_main(obj);
   struct hk_shader *object = hk_native_producer_object(obj);
   hk_mesh_gather_info(nir, &mesh->info.mesh);
   /* The API object may be GS; this program executes at the mesh stage. */
   mesh->info.stage = MESA_SHADER_MESH;
   mesh->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   hk_assert_final_push_mask(nir, mesh);
   bool compiled = hk_apple9_compile_native_mesh_program(
      nir, key, &mesh->b, &mesh->info.apple9_native, reason);
   /* The backend may have allocated a binary before a final capability
    * check fails. Give it to the ordinary destruction path in either case. */
   mesh->code_ptr = mesh->b.binary;
   mesh->code_size = mesh->b.info.binary_size;
   if (!compiled) {
      *unsupported = true;
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   /* Register allocation may require a frame that the native launch cannot
    * yet back. Decline this optional realization before upload. */
   if (!hk_apple9_native_shader_valid(&mesh->b.info,
                                      &mesh->info.apple9_native,
                                      HK_APPLE9_NATIVE_MESH)) {
      *unsupported = true;
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   VkResult result = hk_apple9_finish_shader(dev, mesh, nir);
   if (result != VK_SUCCESS)
      return result;

   struct agx_apple9_mesh_key k = AGX_APPLE9_MESH_KEY;
   nir_shader *generated = obj->vk.stage == MESA_SHADER_TESS_EVAL
                              ? hk_nir_tes_mesh_object()
                              : agx_nir_apple9_taskless_object(&k);
   if (!generated)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   object->info.stage = MESA_SHADER_TASK;
   object->info.apple9_native_stage = HK_APPLE9_NATIVE_OBJECT;
   hk_assert_final_push_mask(generated, object);
   compiled = agx_compile_apple9_native_object(
      generated, hk_apple9_native_mesh_available(hk_device_physical(dev)),
      &k, &object->b,
      &object->info.apple9_native, reason);
   object->code_ptr = object->b.binary;
   object->code_size = object->b.info.binary_size;
   if (!compiled) {
      *unsupported = true;
      result = VK_ERROR_FEATURE_NOT_PRESENT;
   } else {
      if (!hk_apple9_native_shader_valid(&object->b.info,
                                         &object->info.apple9_native,
                                         HK_APPLE9_NATIVE_OBJECT)) {
         *unsupported = true;
         result = VK_ERROR_FEATURE_NOT_PRESENT;
      } else {
         result = hk_apple9_finish_shader(dev, object, generated);
      }
   }
   ralloc_free(generated);
   return result;
}

/* Task shaders run as compute kernels writing their payloads and the grids of
 * mesh workgroups they launch.
 */
static VkResult
hk_apple9_compile_task(struct hk_device *dev, nir_shader *nir,
                       struct hk_shader *shader, const char **reason)
{
   shader->info.task.payload_B = nir->info.task_payload_size;
   hk_nir_lower_task_main(nir);
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   hk_apple9_lower_prerast_sysvals(nir, MESA_SHADER_COMPUTE);
   NIR_PASS(_, nir, nir_opt_dce);
   hk_assert_final_push_mask(nir, shader);
   if (!agx_compile_apple9_tiny(nir, &shader->b, &shader->info.apple9, reason))
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   return hk_apple9_finish_shader(dev, shader, nir);
}

static void
hk_apple9_lower_primitive_fs(nir_shader *nir, bool native_mesh)
{
   bool broadcast =
      native_mesh && hk_apple9_native_mesh_primitive_lowering_enabled();
   if (!nir_shader_intrinsics_pass(nir, hk_lower_per_primitive_input,
                                   nir_metadata_control_flow, &broadcast))
      return;

   /* Layer reads introduced by primitive lowering use the same
    * fixed-function layer source as ordinary fragment inputs. */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            hk_apple9_lower_layer_viewport_fs, nir_metadata_control_flow, NULL);

   /* This NIR has already been preprocessed. The new input/record loads only
    * need dead-code cleanup and refreshed IO metadata before linking; the
    * backend performs ordinary algebraic/scalar cleanup later. Re-running
    * agx_preprocess_nir would nest the non-idempotent helper-write guards. */
   NIR_PASS(_, nir, nir_opt_dce);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

/* fs_interp optionally gives the {flat, linear} input masks of the fragment
 * shader consuming a separately compiled last pre-rasterization stage. */
static VkResult
hk_apple9_compile_graphics_path(struct hk_device *dev, uint32_t count,
                           struct vk_shader_compile_info *infos,
                           const struct vk_graphics_pipeline_state *state,
                           const struct hk_apple9_graphics_key *key_in,
                           const uint64_t *fs_interp,
                           const VkAllocationCallbacks *alloc,
                           struct vk_shader **out, bool native_mesh,
                           bool *native_unsupported)
{
   VkResult result = VK_SUCCESS;
   struct hk_apple9_graphics_key key;
   if (key_in)
      key = *key_in;
   else
      hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
   struct agx_apple9_varying_layout varyings = {0};
   const char *reason = NULL;
   simple_mtx_t *lock = agx_get_compiler_debug()
                           ? &hk_device_physical(dev)->debug_compile_lock
                           : NULL;
   if (lock)
      simple_mtx_lock(lock);

   memset(out, 0, count * sizeof(*out));
   if (native_mesh) {
      for (unsigned i = 0; i < count; ++i) {
         if (infos[i].stage == MESA_SHADER_TASK) {
            reason = "native mesh does not support API task shaders";
            result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT, "%s", reason);
            goto done;
         }
      }
   }

   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage != MESA_SHADER_FRAGMENT)
         continue;
      hk_apple9_lower_primitive_fs(infos[i].nir, native_mesh);
   }

   /* Per-vertex inputs are read as flat copies, written by the passthrough
    * geometry shader of vertex-shader pipelines, or by the vertex shader
    * rasterizing mesh records.
    */
   struct hk_per_vertex_copies per_vertex = {0};
   bool copies_from_gs = false;
   struct hk_apple9_stipple_slots stipple = {0};
   bool stipple_first = false;
   bool polygon_points = false;
   unsigned polygon_face_slot = 0;
   mesa_shader_stage polygon_stage = MESA_SHADER_NONE;
   mesa_shader_stage stipple_stage = MESA_SHADER_NONE;
   bool tes_primitive_id = false;
   {
      nir_shader *pre_nir = NULL, *fs_nir = NULL;
      nir_shader *stage_nir[MESA_VULKAN_SHADER_STAGES] = {NULL};
      bool other = false;
      for (unsigned i = 0; i < count; ++i) {
         stage_nir[infos[i].stage] = infos[i].nir;
         if (infos[i].stage == MESA_SHADER_VERTEX ||
             infos[i].stage == MESA_SHADER_MESH)
            pre_nir = infos[i].nir;
         else if (infos[i].stage == MESA_SHADER_FRAGMENT)
            fs_nir = infos[i].nir;
         else if (infos[i].stage != MESA_SHADER_TASK)
            other = true;
      }
      /* After tessellation without a geometry shader, the fragment shader's
       * primitive ID is the patch ID, numbered per instance. The hardware
       * counts the tessellated triangles of the whole draw instead, so the
       * evaluation shader publishes its own primitive ID (the patch ID) as
       * the flat varying the fragment shader then reads, as RADV's
       * export_prim_id does. */
      tes_primitive_id =
         stage_nir[MESA_SHADER_TESS_EVAL] && !stage_nir[MESA_SHADER_GEOMETRY] &&
         fs_nir && (fs_nir->info.inputs_read & VARYING_BIT_PRIMITIVE_ID) &&
         !(stage_nir[MESA_SHADER_TESS_EVAL]->info.outputs_written &
           VARYING_BIT_PRIMITIVE_ID);
      if (pre_nir && fs_nir && !other &&
          hk_apple9_lower_per_vertex_copies(fs_nir,
                                            pre_nir->info.outputs_written,
                                            &per_vertex))
         copies_from_gs = pre_nir->info.stage == MESA_SHADER_VERTEX;

      /* Stippled lines measure their counter from the segment start the last
       * pre-rasterization stage exports: a vertex shader, an evaluation shader
       * drawing isolines, or a geometry shader emitting line strips. */
      nir_shader *last = NULL;
      if (stage_nir[MESA_SHADER_GEOMETRY]) {
         if (stage_nir[MESA_SHADER_GEOMETRY]->info.gs.output_primitive ==
             MESA_PRIM_LINE_STRIP)
            last = stage_nir[MESA_SHADER_GEOMETRY];
      } else if (stage_nir[MESA_SHADER_TESS_EVAL]) {
         nir_shader *tes = stage_nir[MESA_SHADER_TESS_EVAL];
         nir_shader *tcs = stage_nir[MESA_SHADER_TESS_CTRL];
         enum tess_primitive_mode mode = tes->info.tess._primitive_mode;
         if (mode == TESS_PRIMITIVE_UNSPECIFIED && tcs)
            mode = tcs->info.tess._primitive_mode;
         bool points = tes->info.tess.point_mode ||
                       (tcs && tcs->info.tess.point_mode);
         if (mode == TESS_PRIMITIVE_ISOLINES && !points)
            last = tes;
      } else if (pre_nir && pre_nir->info.stage == MESA_SHADER_VERTEX) {
         last = pre_nir;
      } else if (pre_nir && pre_nir->info.stage == MESA_SHADER_MESH &&
                 pre_nir->info.mesh.primitive_type == MESA_PRIM_LINES) {
         /* Mesh lines are independent: the generated rasterizing vertex
          * shader exports the start as a vertex shader does. */
         last = pre_nir;
      }
      if (key.line_stipple && last && fs_nir) {
         stipple = hk_apple9_line_stipple_slots(last, fs_nir);
         if (stipple.start) {
            stipple_stage = last->info.stage;
         } else {
            static bool warned;
            if (!warned) {
               mesa_logw("Apple9: line stipple is ignored when the last "
                         "pre-rasterization stage uses more than %u of %u "
                         "output components or 31 generic slots",
                         HK_APPLE9_OUTPUT_COMPONENTS -
                            HK_APPLE9_STIPPLE_COMPONENTS,
                         HK_APPLE9_OUTPUT_COMPONENTS);
               warned = true;
            }
         }
         if (stipple.start) {
            hk_apple9_lower_line_stipple_fs(fs_nir, stipple);
            /* The stipple start needs the segment's first vertex. Under the
             * last-vertex mode it can still provoke when no input the
             * application can see is taken from the provoking vertex: no
             * flat varyings, viewport index or layer. Primitive ID is not
             * taken from a vertex. */
            const uint64_t provoked =
               VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT |
               BITFIELD64_RANGE(VARYING_SLOT_COL0, 2) |
               BITFIELD64_RANGE(VARYING_SLOT_BFC0, 2) |
               BITFIELD64_RANGE(VARYING_SLOT_TEX0, 8) |
               BITFIELD64_BIT(VARYING_SLOT_FOGC) |
               BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32);
            uint64_t flat = fs_nir->info.inputs_read &
                            ~(fs_nir->info.perspective_varyings |
                              fs_nir->info.linear_varyings) &
                            ~BITFIELD64_BIT(stipple.start);
            /* Geometry shaders choose the start for either mode. */
            stipple_first =
               stipple_stage != MESA_SHADER_GEOMETRY &&
               stipple_stage != MESA_SHADER_MESH && !(flat & provoked) &&
               !(last->info.outputs_written &
                 (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT));
         }
      }

      /* Polygon-mode points take the software polygon stage when the last
       * pre-rasterization stage's outputs fit: after a vertex or evaluation
       * shader, inside a geometry shader emitting triangle strips, or in the
       * vertex shader rasterizing mesh triangles. Otherwise they keep the
       * hardware polygon mode, a known limitation. */
      nir_shader *mesh_nir = stage_nir[MESA_SHADER_MESH];
      nir_shader *pp_last =
         mesh_nir
            ? (mesh_nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
                     !hk_per_vertex_copies_used(&per_vertex)
                  ? mesh_nir : NULL)
         : stage_nir[MESA_SHADER_GEOMETRY]
            ? (hk_apple9_gs_polygon_points_ok(stage_nir[MESA_SHADER_GEOMETRY])
                  ? stage_nir[MESA_SHADER_GEOMETRY] : NULL)
         : stage_nir[MESA_SHADER_TESS_EVAL] ? stage_nir[MESA_SHADER_TESS_EVAL]
         : stage_nir[MESA_SHADER_VERTEX];
      if (key.polygon_points && pp_last && fs_nir) {
         uint64_t reserved = 0;
         if (stipple.start) {
            reserved = BITFIELD64_BIT(stipple.start) | BITFIELD64_BIT(stipple.axis);
         }
         if (mesh_nir)
            reserved |= BITFIELD64_BIT(HK_MESH_PRIMITIVE_SLOT);
         polygon_points = hk_apple9_polygon_point_budget(pp_last, fs_nir, reserved,
                                                         &polygon_face_slot);
         if (polygon_points)
            polygon_stage = pp_last->info.stage;
         if (polygon_points && polygon_face_slot) {
            NIR_PASS(_, fs_nir, nir_shader_intrinsics_pass,
                     hk_apple9_lower_polygon_slot, nir_metadata_control_flow,
                     &polygon_face_slot);
            nir_shader_gather_info(fs_nir, nir_shader_get_entrypoint(fs_nir));
         }
         if (!polygon_points) {
            static bool warned;
            if (!warned) {
               mesa_logw("Apple9: polygon mode POINT keeps the hardware path "
                         "(size 1) when the stage's outputs leave no room for "
                         "point size and facing");
               warned = true;
            }
         }
      }
   }

   if (fs_interp) {
      key.vertex.outputs_flat = fs_interp[0];
      key.vertex.outputs_linear = fs_interp[1];
   }
   if (key.targets > 8) {
      reason = "more than eight color targets";
      result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
      goto done;
   }
   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_COMPUTE) {
         reason = "compute shaders are compiled separately";
         result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
         goto done;
      }
      if (infos[i].stage == MESA_SHADER_FRAGMENT) {
         key.vertex.outputs_flat = infos[i].nir->info.inputs_read &
                                   ~(infos[i].nir->info.perspective_varyings |
                                     infos[i].nir->info.linear_varyings);
         key.vertex.outputs_linear = infos[i].nir->info.linear_varyings;
         key.fs_reads_clip_distance = infos[i].nir->info.inputs_read &
            (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1);
      }
   }

   /* Tessellation modes may be declared by either tessellation stage, as
    * HLSL declares them on the control shader. Rasterization of the
    * evaluation shader's output depends on them.
    */
   nir_shader *tcs_nir = NULL, *tes_nir = NULL;
   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_TESS_CTRL)
         tcs_nir = infos[i].nir;
      else if (infos[i].stage == MESA_SHADER_TESS_EVAL)
         tes_nir = infos[i].nir;
   }
   if (tcs_nir && tes_nir) {
      tes_nir->info.tess.point_mode |= tcs_nir->info.tess.point_mode;
      tes_nir->info.tess.ccw |= tcs_nir->info.tess.ccw;
      if (tes_nir->info.tess.spacing == TESS_SPACING_UNSPECIFIED)
         tes_nir->info.tess.spacing = tcs_nir->info.tess.spacing;
      if (tes_nir->info.tess._primitive_mode == TESS_PRIMITIVE_UNSPECIFIED)
         tes_nir->info.tess._primitive_mode = tcs_nir->info.tess._primitive_mode;
   }

   bool has_fragment = false;
   for (unsigned i = 0; i < count; ++i)
      has_fragment |= infos[i].stage == MESA_SHADER_FRAGMENT;

   /* The last pre-rasterization stage maps [-w, w] depth to [0, w]. */
   mesa_shader_stage last_prerast = MESA_SHADER_NONE;
   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_VERTEX ||
          infos[i].stage == MESA_SHADER_TESS_EVAL ||
          infos[i].stage == MESA_SHADER_GEOMETRY)
         last_prerast = MAX2(last_prerast, (int)infos[i].stage);
      else if (infos[i].stage == MESA_SHADER_MESH)
         last_prerast = MESA_SHADER_MESH;
   }

   /* Compile the producer first, independent of the API stage-array order. */
   static const mesa_shader_stage order[] = {
      MESA_SHADER_VERTEX,    MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL,
      MESA_SHADER_GEOMETRY,  MESA_SHADER_TASK,      MESA_SHADER_MESH,
      MESA_SHADER_FRAGMENT,
   };
   for (unsigned pass = 0; pass < ARRAY_SIZE(order); ++pass) {
      mesa_shader_stage stage = order[pass];
      for (unsigned i = 0; i < count; ++i) {
         if (infos[i].stage != stage)
            continue;
         size_t size = sizeof(struct hk_api_shader) +
                       hk_num_variants(stage) * sizeof(struct hk_shader);
         struct hk_api_shader *obj =
            vk_shader_zalloc(&dev->vk, &hk_shader_ops, stage, alloc, size);
         if (!obj) {
            result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
            goto done;
         }
         out[i] = &obj->vk;
         hk_foreach_variant(obj, variant) {
            simple_mtx_init(&variant->linked.lock, mtx_plain);
            variant->info.stage = stage;
            variant->b.info.stats.instrs = 1;
         }
         struct hk_shader *shader = &obj->variants[0];
         nir_shader *nir = infos[i].nir;
         obj->writes_memory = nir->info.writes_memory;
         obj->pre_raster_reads = hk_nir_pre_raster_reads(nir);
         obj->reads_draw_id = hk_nir_reads_draw_id(nir);
         obj->push_constant_mask = hk_nir_push_constant_mask(nir);
#ifndef NDEBUG
         hk_foreach_variant(obj, variant)
            variant->debug_owner = obj;
#endif
         shader->info.set_count = infos[i].set_layout_count;
         shader->info.apple9_render_targets = key.targets;
         shader->info.apple9_line_stipple_first =
            stage == MESA_SHADER_FRAGMENT && stipple_first;
         if (!hk_apple9_texture_bindings(nir) ||
             !agx_nir_lower_apple9_runtime_sampler_state(nir, &reason)) {
            reason = reason ?: "texture operation without a descriptor handle";
            result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                               "Apple9 graphics textures: %s", reason);
            goto done;
         }
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_apple9_lower_roots,
                  nir_metadata_control_flow, NULL);
         if (key.clip_negative_one_to_one && stage == last_prerast &&
             !(native_mesh && (stage == MESA_SHADER_GEOMETRY ||
                               stage == MESA_SHADER_TESS_EVAL))) {
            NIR_PASS(_, nir, nir_lower_clip_halfz_dynamic);
            NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                     hk_apple9_lower_clip_z_coeff, nir_metadata_control_flow,
                     NULL);
         }
         /* Shadow and depth-prepass draws: a vertex shader rasterized without
          * a fragment stage computes only what the rasterizer reads. */
         if (stage == MESA_SHADER_VERTEX && stage == last_prerast &&
             !has_fragment && !nir->xfb_info &&
             !(infos[i].next_stage_mask &
               (VK_SHADER_STAGE_GEOMETRY_BIT |
                VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)) &&
             !hk_per_vertex_copies_used(&per_vertex))
            hk_apple9_strip_unread_outputs(nir);

         if (agx_apple9_debug_enabled(AGX_A9_DBG_NIR))
            nir_print_shader(nir, stderr);

         if (stage == MESA_SHADER_VERTEX) {
            /* Native vertex pulling indexes attributes by API location. */
            nir_foreach_function_impl(impl, nir) {
               nir_foreach_block(block, impl) {
                  nir_foreach_instr(instr, block) {
                     if (instr->type != nir_instr_type_intrinsic)
                        continue;
                     nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                     if (intr->intrinsic == nir_intrinsic_load_input) {
                        unsigned loc =
                           nir_intrinsic_io_semantics(intr).location;
                        nir_intrinsic_set_base(intr,
                                               loc - VERT_ATTRIB_GENERIC0);
                     }
                  }
               }
            }
            if (stipple_stage == MESA_SHADER_VERTEX)
               hk_apple9_write_line_stipple_start(nir, stipple);
            if (polygon_stage == MESA_SHADER_VERTEX && polygon_face_slot)
               hk_apple9_write_polygon_slot(nir, polygon_face_slot);
            /* Transform feedback and flat copies of per-vertex inputs are
             * layered on a passthrough geometry shader, which consumes the
             * software vertex shader.
             */
            /* The software variant feeds a geometry stage: the
             * application's, or a passthrough. A cull mask also needs one,
             * which is known once the hardware variant is compiled. */
            bool sw_needed =
               (infos[i].next_stage_mask &
                (VK_SHADER_STAGE_GEOMETRY_BIT |
                 VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)) ||
               nir->xfb_info || hk_per_vertex_copies_used(&per_vertex) ||
               stipple_stage == MESA_SHADER_VERTEX ||
               polygon_stage == MESA_SHADER_VERTEX;
            struct hk_shader *sw = &obj->variants[HK_VS_VARIANT_SW];
            nir_shader *sw_nir = NULL;
            if (sw_needed || nir->info.cull_distance_array_size) {
               sw->info.set_count = shader->info.set_count;
               sw->info.clip_distance_array_size =
                  nir->info.clip_distance_array_size;
               sw->info.cull_distance_array_size =
                  nir->info.cull_distance_array_size;
               if (nir->xfb_info) {
                  assert(nir->xfb_info->output_count <
                         ARRAY_SIZE(sw->info.xfb_outputs));
                  memcpy(&sw->info.xfb_info, nir->xfb_info,
                         nir_xfb_info_size(nir->xfb_info->output_count));
                  typed_memcpy(sw->info.xfb_stride, nir->info.xfb_stride, 4);
               }
               sw_nir = nir_shader_clone(NULL, nir);
            }
            /* Without per-vertex inputs, the hardware vertex shader writes
             * the one-hot barycentric weights itself, so non-indexed
             * triangle lists need no passthrough geometry shader. */
            if (!per_vertex.count && hk_per_vertex_copies_used(&per_vertex))
               hk_apple9_write_vertex_weights(nir, &per_vertex);
            bool cull_mask;
            hk_assert_final_push_mask(nir, shader);
            if (!hk_apple9_compile_vertex_culls(nir, &key.vertex, &shader->b,
                                                &cull_mask, &reason)) {
               ralloc_free(sw_nir);
               reason = reason ?: "unsupported vertex NIR";
               result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                                  "Apple9 vertex: %s", reason);
               goto done;
            }
            shader->info.cull_mask = sw->info.cull_mask = cull_mask;
            if (sw_needed || cull_mask) {
               result = hk_apple9_compile_sw_vs(dev, sw_nir, &key.vertex, sw,
                                                &reason);
            }
            ralloc_free(sw_nir);
            if (result != VK_SUCCESS)
               goto done;
            varyings = shader->b.info.apple9_varyings;
         } else if (stage == MESA_SHADER_TESS_CTRL) {
            result = hk_apple9_compile_tcs(dev, nir, shader, &reason);
            if (result != VK_SUCCESS)
               goto done;
            continue;
         } else if (stage == MESA_SHADER_TESS_EVAL) {
            if (native_mesh) {
               obj->apple9_native_tess = (struct hk_tess_info){
                  .mode = nir->info.tess._primitive_mode,
                  .spacing = nir->info.tess.spacing,
                  .ccw = nir->info.tess.ccw,
                  .points = nir->info.tess.point_mode,
               };
               nir_shader *mesh = hk_nir_tes_mesh(nir, &reason);
               if (!mesh) {
                  *native_unsupported = true;
                  result = VK_ERROR_FEATURE_NOT_PRESENT;
                  goto done;
               }
               hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
               result = hk_apple9_compile_native_mesh(
                  dev, mesh, obj, &key, &reason, native_unsupported);
               ralloc_free(mesh);
               if (result != VK_SUCCESS)
                  goto done;
               varyings = hk_native_producer_main(obj)->b.info.apple9_varyings;
               continue;
            }
            if (stipple_stage == MESA_SHADER_TESS_EVAL)
               hk_apple9_write_line_stipple_start(nir, stipple);
            if (polygon_stage == MESA_SHADER_TESS_EVAL && polygon_face_slot)
               hk_apple9_write_polygon_slot(nir, polygon_face_slot);
            if (tes_primitive_id)
               hk_apple9_write_primitive_id(nir);
            result = hk_apple9_compile_tes(
               dev, nir, &key.vertex, obj,
               infos[i].next_stage_mask |
                  (polygon_stage == MESA_SHADER_TESS_EVAL
                      ? VK_SHADER_STAGE_GEOMETRY_BIT : 0),
               &reason);
            if (result != VK_SUCCESS)
               goto done;
            varyings = shader->b.info.apple9_varyings;
            continue;
         } else if (stage == MESA_SHADER_GEOMETRY) {
            if (native_mesh) {
               nir_shader *mesh = hk_nir_gs_mesh(nir, &reason);
               if (!mesh) {
                  *native_unsupported = true;
                  result = VK_ERROR_FEATURE_NOT_PRESENT;
                  goto done;
               }
               hk_apple9_lower_prerast_sysvals(mesh, MESA_SHADER_COMPUTE);
               result = hk_apple9_compile_native_mesh(
                  dev, mesh, obj, &key, &reason, native_unsupported);
               ralloc_free(mesh);
               if (result != VK_SUCCESS)
                  goto done;
               varyings = hk_native_producer_main(obj)->b.info.apple9_varyings;
               continue;
            }
            if (stipple_stage == MESA_SHADER_GEOMETRY)
               hk_apple9_write_gs_line_stipple(nir, stipple);
            if (polygon_stage == MESA_SHADER_GEOMETRY) {
               struct hk_pp_info info = {
                  .clip = nir->info.clip_distance_array_size,
                  .cull = nir->info.cull_distance_array_size,
                  .flat = key.vertex.outputs_flat,
                  .face_slot = polygon_face_slot,
               };
               hk_apple9_lower_gs_polygon_points(
                  nir, info, hk_device_physical(dev)->vk.properties.polygonModePointSize);
            }
            result = hk_apple9_compile_gs(dev, nir, &key.vertex,
                                          !key.gs_cull_planes, obj, &reason);
            if (result != VK_SUCCESS)
               goto done;
            varyings = shader->b.info.apple9_varyings;
            continue;
         } else if (stage == MESA_SHADER_TASK) {
            result = hk_apple9_compile_task(dev, nir, shader, &reason);
            if (result != VK_SUCCESS)
               goto done;
            continue;
         } else if (stage == MESA_SHADER_MESH) {
            if (native_mesh) {
               result = hk_apple9_compile_native_mesh(
                  dev, nir, obj, &key, &reason, native_unsupported);
               if (result != VK_SUCCESS)
                  goto done;
               varyings = shader->b.info.apple9_varyings;
               continue;
            }

            struct hk_pp_info pp = {
               .clip = nir->info.clip_distance_array_size,
               .cull = nir->info.cull_distance_array_size,
               .flat = key.vertex.outputs_flat,
               .face_slot = polygon_face_slot,
            };
            result = hk_apple9_compile_mesh(
               dev, nir, &key.vertex, key.clip_negative_one_to_one, &per_vertex,
               stipple_stage == MESA_SHADER_MESH
                  ? stipple : (struct hk_apple9_stipple_slots){0},
               polygon_stage == MESA_SHADER_MESH ? &pp : NULL,
               hk_device_physical(dev)->vk.properties.polygonModePointSize,
               obj, &reason);
            if (result != VK_SUCCESS)
               goto done;
            varyings = obj->variants[HK_MESH_VARIANT_RAST].b.info.apple9_varyings;
            continue;
         } else {
            shader->info.fs.interp.linear = key.vertex.outputs_linear;
            shader->info.fs.interp.flat = key.vertex.outputs_flat;
            shader->info.fs.line_stipple_start = stipple.start;
            shader->info.fs.polygon_points = polygon_points;
            shader->info.fs.polygon_face_slot = polygon_face_slot;
            if (copies_from_gs) {
               shader->info.fs.per_vertex_copies = per_vertex.count;
               memcpy(shader->info.fs.per_vertex_bary, per_vertex.bary,
                      sizeof(per_vertex.bary));
            }
            memcpy(shader->info.fs.per_vertex_location, per_vertex.location,
                   sizeof(per_vertex.location));
            memcpy(shader->info.fs.per_vertex_copy_base, per_vertex.copy_base,
                   sizeof(per_vertex.copy_base));
            /* The statistics counter is the only memory write or wave-wide
             * operation of many shaders, but it runs only while a query is
             * active and counts correctly in merged waves. Keep triangle
             * merging unless the shader itself requires disabling it. */
            bool tri_merging = true;
            if (!nir->info.internal) {
               if (key.zs_self_dep)
                  hk_apple9_late_zs(nir);
               nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
               tri_merging = !nir->info.uses_wide_subgroup_intrinsics &&
                             !nir->info.fs.needs_coarse_quad_helper_invocations &&
                             !nir->info.writes_memory &&
                             !key.sample_shading &&
                             !hk_apple9_interpolates_off_center(nir);
               if (!key.no_fs_statistics)
                  hk_apple9_lower_fs_statistics(nir);
               NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                        hk_apple9_clamp_depth_export,
                        nir_metadata_control_flow, NULL);
            }
            shader->info.fs.writes_memory = nir->info.writes_memory;
            shader->info.fs.early_fragment_tests =
               nir->info.fs.early_fragment_tests;
            if (key.sample_shading) {
               nir->info.fs.uses_sample_shading = true;
               NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                        hk_apple9_lower_bary_to_sample,
                        nir_metadata_control_flow, NULL);
            }
            NIR_PASS(_, nir, hk_apple9_lower_api_vertex_order);
            hk_apple9_remap_fs_outputs(nir, key.color_map);
            /* The passthrough geometry shader produces the flat copies, and
             * per-vertex inputs read only through copies are not rasterized.
             * The mesh rasterization shader's layout already has the copies.
             */
            for (unsigned j = 0; copies_from_gs && j < per_vertex.count; ++j) {
               unsigned source = VARYING_SLOT_VAR0 + per_vertex.location[j];
               if (!(nir->info.inputs_read & BITFIELD64_BIT(source)) &&
                   varyings.mask[source]) {
                  varyings.count -= util_bitcount(varyings.mask[source]);
                  varyings.mask[source] = 0;
                  varyings.group[source] = 0;
               }
            }
            for (unsigned j = 0; copies_from_gs && j < per_vertex.count; ++j) {
               for (unsigned k = 0; k < 3; ++k) {
                  unsigned slot = VARYING_SLOT_VAR0 + per_vertex.copy_base[j] + k;
                  if (!varyings.mask[slot]) {
                     varyings.mask[slot] = 0xf;
                     varyings.group[slot] = 1;
                     varyings.count += 4;
                  }
               }
            }
            for (unsigned m = 0; copies_from_gs && m < 2; ++m) {
               unsigned slot = VARYING_SLOT_VAR0 + per_vertex.bary[m] - 1;
               if (per_vertex.bary[m] && !varyings.mask[slot]) {
                  varyings.mask[slot] = 0x7;
                  varyings.group[slot] = m ? 2 : 0;
                  varyings.count += 3;
               }
            }
            struct blob dump_nir;
            const bool dump = hk_apple9_fragment_dump_begin(nir, &dump_nir);
            hk_assert_final_push_mask(nir, shader);
            const bool compiled = agx_compile_apple9_fragment_mrt(
               nir, &varyings, key.blend, key.targets, &shader->b, &reason);
            if (dump)
               hk_apple9_fragment_dump_end(&dump_nir, compiled, &varyings,
                                           key.blend, key.targets, &shader->b);
            if (!compiled) {
               reason = reason ?: "unsupported fragment NIR";
               if (native_mesh)
                  *native_unsupported = true;
               result = native_mesh ? VK_ERROR_FEATURE_NOT_PRESENT :
                  vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                            "Apple9 fragment: %s", reason);
               goto done;
            }
            /* hk's decision covers the backend's reasons for the shader as
             * written (before the statistics counter, which counts correctly
             * in merged waves) and adds sample shading and off-centre
             * interpolation, which the backend can't see. */
            shader->b.info.disable_tri_merging = !tri_merging;
         }
         shader->code_ptr = shader->b.binary;
         shader->code_size = shader->b.info.binary_size;
         hk_gpu_busy_shader_source(shader->code_ptr, shader->code_size, nir->info.name,
                                   nir->info.source_blake3, nir->info.stage);
         shader->info.clip_distance_array_size =
            nir->info.clip_distance_array_size;
         shader->info.cull_distance_array_size =
            nir->info.cull_distance_array_size;
         result = hk_copy_shader_constant_data(dev, shader, nir);
         if (result == VK_SUCCESS)
            result = hk_upload_shader(dev, shader);
         if (result != VK_SUCCESS)
            goto done;
      }
   }

done:
   if (result != VK_SUCCESS && agx_get_compiler_debug())
      fprintf(stderr, "Apple9 graphics compilation failed: %s\n",
              reason ?: vk_Result_to_str(result));
   if (lock)
      simple_mtx_unlock(lock);
   for (unsigned i = 0; i < count; ++i) {
      ralloc_free(infos[i].nir);
      if (result != VK_SUCCESS && out[i]) {
         hk_api_shader_destroy(&dev->vk, out[i], alloc);
         out[i] = NULL;
      } else if (out[i]) {
         hk_api_shader_init_constant_data(
            container_of(out[i], struct hk_api_shader, vk));
      }
   }
   return result;
}

/* Match the whole native compile, including both interfaces, code and constant
 * tables. This identity survives independent per-stage cache round trips; a
 * pointer or a process-local counter would not. Nothing owns another stage. */
static void
hk_apple9_identify_native_pair(uint32_t count, struct vk_shader **shaders,
                               const struct hk_apple9_graphics_key *key)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, key, sizeof(*key));
   for (unsigned i = 0; i < count; ++i) {
      struct hk_api_shader *obj =
         container_of(shaders[i], struct hk_api_shader, vk);
      uint32_t stage = obj->vk.stage;
      _mesa_blake3_update(&ctx, &stage, sizeof(stage));
      if (stage == MESA_SHADER_TESS_EVAL)
         _mesa_blake3_update(&ctx, &obj->apple9_native_tess,
                             sizeof(obj->apple9_native_tess));
      hk_foreach_variant(obj, shader) {
         _mesa_blake3_update(&ctx, &shader->b.info.apple9_varyings,
                             sizeof(shader->b.info.apple9_varyings));
         _mesa_blake3_update(&ctx, &shader->code_size,
                             sizeof(shader->code_size));
         _mesa_blake3_update(&ctx, shader->code_ptr, shader->code_size);
         _mesa_blake3_update(&ctx, &shader->data_size,
                             sizeof(shader->data_size));
         if (shader->data_size)
            _mesa_blake3_update(&ctx, shader->data_ptr, shader->data_size);
      }
   }
   uint8_t stage_mask = 0;
   for (unsigned i = 0; i < count; ++i) {
      if (shaders[i]->stage == MESA_SHADER_GEOMETRY)
         stage_mask = key->native_stages & HK_APPLE9_NATIVE_GS_BIT;
      if (shaders[i]->stage == MESA_SHADER_TESS_EVAL)
         stage_mask = key->native_stages & HK_APPLE9_NATIVE_TES_BIT;
   }
   _mesa_blake3_update(&ctx, &stage_mask, sizeof(stage_mask));
   blake3_hash id;
   _mesa_blake3_final(&ctx, id);
   for (unsigned i = 0; i < count; ++i) {
      struct hk_api_shader *obj =
         container_of(shaders[i], struct hk_api_shader, vk);
      obj->apple9_native_leaf = true;
      obj->apple9_native_stage_mask = stage_mask;
      memcpy(obj->apple9_native_pair_id, id, sizeof(id));
      if (obj->vk.stage == MESA_SHADER_MESH ||
          obj->vk.stage == MESA_SHADER_GEOMETRY ||
          obj->vk.stage == MESA_SHADER_TESS_EVAL) {
         obj->apple9_native_candidate = true;
         obj->apple9_native_cull_primitive =
            (hk_native_producer_main(obj)->info.apple9_native.primitive_broadcast &
             VARYING_BIT_CULL_PRIMITIVE) ||
            hk_native_producer_main(obj)->info.apple9_native.cull_distances;
      }
   }
}

/* Eager pipelines own both realizations. Draw-time variants request one path:
 * native compiles only its own programs, falling back to emulation if the
 * native backend rejects the body. Independent NIR keeps either rejection
 * from leaving a half-lowered fallback behind. */
/* Whether a vertex shader's pipeline disables vertex input robustness, so
 * its attribute fetches may skip bounds checks. robustBufferAccess,
 * robustness2 and VK_EXT_pipeline_robustness all resolve into the stage's
 * vertex_inputs behaviour; shader objects take the device default. */
static bool
hk_apple9_vertex_unbounded(const struct vk_shader_compile_info *info)
{
   return info->robustness &&
          info->robustness->vertex_inputs ==
             VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT;
}

static VkResult
hk_apple9_compile_graphics_mode(struct hk_device *dev, uint32_t count,
                                struct vk_shader_compile_info *infos,
                                const struct vk_graphics_pipeline_state *state,
                                const struct hk_apple9_graphics_key *key_in,
                                const uint64_t *fs_interp,
                                const VkAllocationCallbacks *alloc,
                                struct vk_shader **out, bool native_only)
{
   assert(count <= HK_APPLE9_GFX_STAGES);
   struct hk_apple9_graphics_key key;
   if (key_in) {
      key = *key_in;
   } else {
      hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
      for (unsigned i = 0; i < count; ++i) {
         if (infos[i].stage == MESA_SHADER_VERTEX)
            key.vertex.unbounded = hk_apple9_vertex_unbounded(&infos[i]);
      }
   }

   nir_shader *mesh = NULL, *gs = NULL, *tes = NULL, *tcs = NULL;
   bool task = false, tess = false, per_vertex = false, fs_inputs = false;
   bool producer_effects = false;
   for (unsigned i = 0; i < count; ++i) {
      if (infos[i].stage != MESA_SHADER_FRAGMENT)
         producer_effects |= infos[i].nir->info.writes_memory || infos[i].nir->xfb_info;
      if (infos[i].stage == MESA_SHADER_MESH)
         mesh = infos[i].nir;
      if (infos[i].stage == MESA_SHADER_GEOMETRY)
         gs = infos[i].nir;
      if (infos[i].stage == MESA_SHADER_TESS_EVAL)
         tes = infos[i].nir;
      if (infos[i].stage == MESA_SHADER_TESS_CTRL)
         tcs = infos[i].nir;
      tess |= infos[i].stage == MESA_SHADER_TESS_CTRL ||
              infos[i].stage == MESA_SHADER_TESS_EVAL;
      task |= infos[i].stage == MESA_SHADER_TASK;
      if (infos[i].stage == MESA_SHADER_FRAGMENT)
         fs_inputs |= infos[i].nir->info.inputs_read & ~VARYING_BIT_POS;
      if (infos[i].stage != MESA_SHADER_FRAGMENT)
         continue;
      nir_foreach_function_impl(impl, infos[i].nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               per_vertex |= op == nir_intrinsic_load_input_vertex ||
                             op == nir_intrinsic_load_per_vertex_input ||
                             hk_is_barycentric_coord(op);
            }
         }
      }
   }

   bool native = false;
   VkResult result = VK_SUCCESS;
   const char *reason = NULL;
   if (mesh && !task && !per_vertex && !key.polygon_points)
      result = hk_apple9_native_mesh_check(mesh, &key, &native, &reason);
   else if (gs && !mesh && !task && !tess && !per_vertex && !fs_inputs &&
            !producer_effects &&
            !key.polygon_points && !key.line_stipple)
      result = hk_apple9_native_gs_check(gs, &key, &native, &reason);
   else if (tes && tcs && !gs && !mesh && !task && !per_vertex && !fs_inputs &&
            !producer_effects &&
            !key.polygon_points && !key.line_stipple)
      result = hk_apple9_native_tess_check(tes, &key, &native, &reason);

   struct vk_shader_compile_info native_infos[HK_APPLE9_GFX_STAGES] = {0};
   unsigned cloned = 0;
   if (result == VK_SUCCESS && native) {
      for (; cloned < count; ++cloned) {
         native_infos[cloned] = infos[cloned];
         native_infos[cloned].nir = nir_shader_clone(NULL, infos[cloned].nir);
         if (!native_infos[cloned].nir) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            break;
         }
      }
   }
   if (result != VK_SUCCESS) {
      for (unsigned i = 0; i < cloned; ++i)
         ralloc_free(native_infos[i].nir);
      for (unsigned i = 0; i < count; ++i)
         ralloc_free(infos[i].nir);
      memset(out, 0, count * sizeof(*out));
      return vk_error(dev, result);
   }

   memset(out, 0, count * sizeof(*out));
   if (!native_only || !native) {
      result = hk_apple9_compile_graphics_path(
         dev, count, infos, state, &key, fs_interp, alloc, out, false, NULL);
      if (result != VK_SUCCESS || !native) {
         for (unsigned i = 0; i < cloned; ++i)
            ralloc_free(native_infos[i].nir);
         return result;
      }
   }

   struct vk_shader *native_out[HK_APPLE9_GFX_STAGES] = {0};
   bool native_unsupported = false;
   result = hk_apple9_compile_graphics_path(dev, count, native_infos, state,
                                           &key, fs_interp, alloc,
                                           native_out, true,
                                           &native_unsupported);
   if (native_unsupported) {
      assert(result == VK_ERROR_FEATURE_NOT_PRESENT);
      /* Allocation failures remain real Vulkan errors. Only a native-only
       * capability failure compiles/uses the emulated realization. */
      return native_only ? hk_apple9_compile_graphics_path(
                              dev, count, infos, state, &key, fs_interp,
                              alloc, out, false, NULL)
                         : VK_SUCCESS;
   }
   if (native_only) {
      for (unsigned i = 0; i < count; ++i)
         ralloc_free(infos[i].nir);
   }
   if (result != VK_SUCCESS) {
      for (unsigned i = 0; i < count; ++i) {
         if (out[i])
            hk_api_shader_destroy(&dev->vk, out[i], alloc);
         out[i] = NULL;
      }
      return result;
   }

   hk_apple9_identify_native_pair(count, native_out, &key);
   if (native_only) {
      memcpy(out, native_out, count * sizeof(*out));
      return VK_SUCCESS;
   }
   for (unsigned i = 0; i < count; ++i) {
      struct hk_api_shader *obj = container_of(out[i], struct hk_api_shader, vk);
      obj->apple9_native = container_of(native_out[i], struct hk_api_shader, vk);
      obj->apple9_native_candidate = infos[i].stage == MESA_SHADER_MESH ||
                                     infos[i].stage == MESA_SHADER_GEOMETRY ||
                                     infos[i].stage == MESA_SHADER_TESS_EVAL;
      obj->apple9_native_cull_primitive =
         obj->apple9_native->apple9_native_cull_primitive;
      obj->apple9_native_stage_mask = obj->apple9_native->apple9_native_stage_mask;
      hk_api_shader_init_constant_data(obj);
   }
   return VK_SUCCESS;
}

static VkResult
hk_apple9_compile_graphics(struct hk_device *dev, uint32_t count,
                           struct vk_shader_compile_info *infos,
                           const struct vk_graphics_pipeline_state *state,
                           const struct hk_apple9_graphics_key *key_in,
                           const uint64_t *fs_interp,
                           const VkAllocationCallbacks *alloc,
                           struct vk_shader **out)
{
   return hk_apple9_compile_graphics_mode(dev, count, infos, state, key_in,
                                          fs_interp, alloc, out, false);
}

/* Even without an API fragment shader, the pipeline sample mask controls
 * depth/stencil writes and occlusion. Keep the empty program independent of
 * pipeline state by reading the normal draw root table.
 */
static nir_shader *
hk_apple9_null_fs_nir(bool sample_mask)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                 &agx_nir_options, "empty FS");
   /* With a full API sample mask the program has no work at all: no
    * coverage, no root reads, as Metal's empty fragment program. */
   if (!sample_mask)
      return b.shader;
   const unsigned mask_offset = hk_root_descriptor_offset(draw.api_sample_mask);
   nir_def *mask = nir_load_ubo(&b, 1, 16, nir_imm_int(&b, 0),
      nir_imm_int(&b, mask_offset), .align_mul = 2,
      .range_base = mask_offset, .range = 2);
   nir_demote_samples(&b, nir_inot(&b, mask));
   return b.shader;
}

VkResult
hk_apple9_create_null_fs(struct hk_device *dev,
                         const VkAllocationCallbacks *alloc, bool sample_mask,
                         struct hk_api_shader **out)
{
   struct vk_shader_compile_info info = {
      .nir = hk_apple9_null_fs_nir(sample_mask),
      .robustness = &vk_robustness_disabled,
      .stage = MESA_SHADER_FRAGMENT,
   };
   struct vk_shader *shader = NULL;
   VkResult result =
      hk_apple9_compile_graphics(dev, 1, &info, NULL, NULL, NULL, alloc, &shader);
   *out = shader ? container_of(shader, struct hk_api_shader, vk) : NULL;
   return result;
}

/* Internal geometry shaders, such as the transform feedback passthrough,
 * rasterize into the bound fragment shader's interpolation layout.
 */
VkResult
hk_apple9_compile_internal_gs(struct hk_device *dev, nir_shader *nir,
                              const uint64_t fs_interp[2], bool cull_mask,
                              struct hk_api_shader **out)
{
   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };
   hk_lower_nir(dev, nir, &rs, false, 0, NULL, 0);
   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_GEOMETRY,
      .nir = nir,
      .robustness = &rs,
   };
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key_device(&key, NULL, hk_device_physical(dev));
   key.gs_cull_planes = !cull_mask;
   struct vk_shader *shader = NULL;
   VkResult result =
      hk_apple9_compile_graphics(dev, 1, &info, NULL, &key, fs_interp, NULL, &shader);
   *out = shader ? container_of(shader, struct hk_api_shader, vk) : NULL;
   return result;
}

/* Apple9 graphics shaders specialise on state that the common AGX path
 * handles with prologs and epilogs: vertex formats, blending, attachment
 * formats and multisampling. When that state is dynamic, belongs to another
 * pipeline library, or is absent for shader objects, compile at draw time.
 */
static bool
hk_apple9_needs_deferral(uint32_t count,
                         const struct vk_shader_compile_info *infos,
                         const struct vk_graphics_pipeline_state *state,
                         const struct vk_features *features)
{
   if (!state)
      return true;

   /* Library stages link with stages compiled elsewhere, whose interfaces
    * and state are known only when the pipeline is bound. */
   if (state->pipeline_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)
      return true;

   /* Stages compiled apart from the rest of their pipeline, as fast-linked
    * libraries are, cannot agree on the varying layout until they are
    * linked. Pipelines without a fragment shader link the null one then. */
   VkShaderStageFlags compiled = 0;
   for (uint32_t i = 0; i < count; ++i)
      compiled |= mesa_to_vk_shader_stage(infos[i].stage);
   if (hk_apple9_partial_stages(compiled))
      return true;

   for (unsigned i = 0; i < ARRAY_SIZE(hk_apple9_keyed_dynamic); ++i) {
      if (BITSET_TEST(state->dynamic, hk_apple9_keyed_dynamic[i]))
         return true;
   }

   /* A depth/stencil feedback loop keys on whether the draw can write depth
    * or stencil (hk_apple9_zs_writes): dynamic state decides at draw time. */
   if (state->pipeline_flags &
       VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) {
      static const enum mesa_vk_dynamic_graphics_state ds_writes[] = {
         MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE,
         MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE,
         MESA_VK_DYNAMIC_DS_DEPTH_COMPARE_OP,
         MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE,
         MESA_VK_DYNAMIC_DS_STENCIL_OP,
         MESA_VK_DYNAMIC_DS_STENCIL_WRITE_MASK,
      };
      for (unsigned i = 0; i < ARRAY_SIZE(ds_writes); ++i) {
         if (BITSET_TEST(state->dynamic, ds_writes[i]))
            return true;
      }
   }

   for (uint32_t i = 0; i < count; ++i) {
      if (infos[i].stage == MESA_SHADER_VERTEX && !state->vi)
         return true;
      if (infos[i].stage == MESA_SHADER_FRAGMENT &&
          (!state->rp || !state->ms || !state->cb ||
           hk_apple9_color_layout_varies(state, features)))
         return true;
   }
   return false;
}

static VkResult
hk_apple9_defer_graphics(struct hk_device *dev, uint32_t count,
                         struct vk_shader_compile_info *infos,
                         const struct vk_graphics_pipeline_state *state,
                         const VkAllocationCallbacks *alloc,
                         struct vk_shader **out);

/* Debug only: a vertex shader writing zero to every generic varying a
 * fragment shader reads. Offline replay never links a fragment shader
 * compiled apart from its vertex shader; pairing it with this stub reaches
 * the backend with a matching varying layout. */
static nir_shader *
hk_apple9_stub_vs_nir(uint64_t inputs)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                 &agx_nir_options, "stub VS");
   nir_def *zero = nir_imm_vec4(&b, 0.0, 0.0, 0.0, 1.0);
   uint64_t written = VARYING_BIT_POS;
   u_foreach_bit64(slot, inputs) {
      if (slot >= VARYING_SLOT_VAR0 && slot < VARYING_SLOT_VAR0 + 32)
         written |= BITFIELD64_BIT(slot);
   }
   u_foreach_bit64(slot, written) {
      nir_store_output(&b, zero, nir_imm_int(&b, 0), .base = 0,
                       .write_mask = 0xf, .component = 0,
                       .src_type = nir_type_float32,
                       .io_semantics.location = slot,
                       .io_semantics.num_slots = 1);
   }
   b.shader->info.outputs_written = written;
   return b.shader;
}

static __thread bool hk_apple9_eager_busy;

/* Debug only (HK_APPLE9_EAGER_VARIANTS): compile a deferred pipeline's
 * variant when it is created, as a draw would, so offline replay that never
 * draws reaches the backend. State the pipeline leaves dynamic takes simple
 * defaults, so the variant can differ from one a real draw selects.
 */
static void
hk_apple9_eager_variant(struct hk_device *dev, uint32_t count,
                        struct vk_shader *const *shaders,
                        const struct vk_graphics_pipeline_state *pstate)
{
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {NULL};
   if (hk_apple9_eager_busy)
      return;
   for (uint32_t i = 0; i < count; ++i)
      api[shaders[i]->stage] = container_of(shaders[i], struct hk_api_shader, vk);

   /* A fragment shader compiled alone: pair it with a stub vertex shader. */
   struct hk_api_shader *stub = NULL;
   if (!api[MESA_SHADER_VERTEX] && !api[MESA_SHADER_MESH] &&
       api[MESA_SHADER_FRAGMENT] && api[MESA_SHADER_FRAGMENT]->apple9_deferred &&
       !(pstate && (pstate->pipeline_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR))) {
      const struct hk_apple9_deferred *d = api[MESA_SHADER_FRAGMENT]->apple9_deferred;
      struct blob_reader reader;
      blob_reader_init(&reader, d->nir, d->nir_size);
      nir_shader *fs = nir_deserialize(NULL, &agx_nir_options, &reader);
      if (fs) {
         struct vk_shader_compile_info info = {
            .stage = MESA_SHADER_VERTEX,
            .nir = hk_apple9_stub_vs_nir(fs->info.inputs_read),
            .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
            .robustness = &vk_robustness_disabled,
         };
         ralloc_free(fs);
         struct vk_shader *vs = NULL;
         hk_apple9_eager_busy = true;
         if (hk_apple9_defer_graphics(dev, 1, &info, NULL, NULL, &vs) == VK_SUCCESS)
            stub = container_of(vs, struct hk_api_shader, vk);
         hk_apple9_eager_busy = false;
         api[MESA_SHADER_VERTEX] = stub;
      }
   }
   if (agx_apple9_debug_enabled(AGX_A9_DBG_EAGER))
      fprintf(stderr, "HK_APPLE9_EAGER: %u stages vs=%d fs=%d flags=%#llx\n", count,
              !!api[MESA_SHADER_VERTEX], !!api[MESA_SHADER_FRAGMENT],
              pstate ? (unsigned long long)pstate->pipeline_flags : 0ull);
   if ((!api[MESA_SHADER_VERTEX] && !api[MESA_SHADER_MESH]) ||
       (pstate && (pstate->pipeline_flags &
                   VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)))
      return;

   struct vk_vertex_input_state vi = {.bindings_valid = 1};
   vi.bindings[0].stride = 16;
   vi.attributes_valid = BITFIELD_MASK(MESA_VK_MAX_VERTEX_ATTRIBUTES);
   for (unsigned a = 0; a < MESA_VK_MAX_VERTEX_ATTRIBUTES; ++a)
      vi.attributes[a].format = VK_FORMAT_R32G32B32A32_SFLOAT;
   struct vk_input_assembly_state ia = {
      .primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   struct vk_viewport_state vp = {.viewport_count = 1, .scissor_count = 1};
   struct vk_multisample_state ms = {
      .rasterization_samples = VK_SAMPLE_COUNT_1_BIT,
      .sample_mask = 0xffff,
   };
   struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   struct vk_color_blend_state cb = {0};
   struct vk_color_attachment_location_state cal;
   for (unsigned rt = 0; rt < MESA_VK_MAX_COLOR_ATTACHMENTS; ++rt) {
      cb.attachments[rt].write_mask = 0xf;
      cal.color_map[rt] = rt;
   }

   struct vk_graphics_pipeline_state state = {
      .vi = pstate && pstate->vi ? pstate->vi : &vi,
      .ia = pstate && pstate->ia ? pstate->ia : &ia,
      .vp = pstate && pstate->vp ? pstate->vp : &vp,
      .ms = pstate && pstate->ms ? pstate->ms : &ms,
      .rp = pstate && pstate->rp ? pstate->rp : &rp,
      .cal = pstate && pstate->cal ? pstate->cal : &cal,
      .pipeline_flags = pstate ? pstate->pipeline_flags : 0,
   };
   if (pstate && pstate->cb) {
      state.cb = pstate->cb;
   } else {
      cb.attachment_count = state.rp->color_attachment_count;
      state.cb = &cb;
   }
   BITSET_SET(state.dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);

   struct hk_api_shader *variant[HK_APPLE9_GFX_STAGES];
   hk_apple9_eager_compile = true;
   /* Warm the variant draws use outside statistics queries. */
   if (hk_apple9_resolve_graphics(dev, api, &state, variant, true, false) !=
       VK_SUCCESS)
      fprintf(stderr, "HK_APPLE9_EAGER_VARIANTS: variant compile failed\n");
   hk_apple9_eager_compile = false;
   if (stub)
      hk_api_shader_destroy(&dev->vk, &stub->vk, NULL);
}

static VkResult
hk_apple9_defer_graphics(struct hk_device *dev, uint32_t count,
                         struct vk_shader_compile_info *infos,
                         const struct vk_graphics_pipeline_state *state,
                         const VkAllocationCallbacks *alloc,
                         struct vk_shader **out)
{
   VkResult result = VK_SUCCESS;
   memset(out, 0, count * sizeof(*out));

   for (uint32_t i = 0; i < count; ++i) {
      bool native_candidate = false;
      if (infos[i].stage == MESA_SHADER_MESH) {
         struct hk_apple9_graphics_key key;
         hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
         result = hk_apple9_native_mesh_check(infos[i].nir, &key,
                                               &native_candidate, NULL);
         if (result != VK_SUCCESS)
            break;
      }
      if (infos[i].stage == MESA_SHADER_GEOMETRY) {
         struct hk_apple9_graphics_key key;
         hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
         result = hk_apple9_native_gs_check(infos[i].nir, &key,
                                           &native_candidate, NULL);
         if (result != VK_SUCCESS)
            break;
      }
      if (infos[i].stage == MESA_SHADER_TESS_EVAL) {
         struct hk_apple9_graphics_key key;
         hk_apple9_graphics_key_device(&key, state, hk_device_physical(dev));
         result = hk_apple9_native_tess_check(infos[i].nir, &key,
                                              &native_candidate, NULL);
         if (result != VK_SUCCESS)
            break;
      }
      size_t size = sizeof(struct hk_api_shader) +
                    hk_num_variants(infos[i].stage) * sizeof(struct hk_shader);
      struct hk_api_shader *obj =
         vk_shader_zalloc(&dev->vk, &hk_shader_ops, infos[i].stage, alloc, size);
      struct hk_apple9_deferred *deferred = calloc(1, sizeof(*deferred));
      if (!obj || !deferred) {
         if (obj)
            vk_shader_free(&dev->vk, alloc, &obj->vk);
         free(deferred);
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         break;
      }

      struct blob blob;
      blob_init(&blob);
      nir_serialize(&blob, infos[i].nir, false);
      if (blob.out_of_memory) {
         blob_finish(&blob);
         vk_shader_free(&dev->vk, alloc, &obj->vk);
         free(deferred);
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         break;
      }
      blob_finish_get_buffer(&blob, &deferred->nir, &deferred->nir_size);
      deferred->next_stage_mask = infos[i].next_stage_mask;
      deferred->set_layout_count = infos[i].set_layout_count;
      deferred->cull_distance = infos[i].nir->info.cull_distance_array_size != 0;
      deferred->sample_shading =
         state && state->ms && state->ms->sample_shading_enable;
      deferred->vertex_unbounded = infos[i].stage == MESA_SHADER_VERTEX &&
                                   hk_apple9_vertex_unbounded(&infos[i]);
      deferred->zs_self_dep =
         state && (state->pipeline_flags &
                   VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
      if (infos[i].stage == MESA_SHADER_FRAGMENT && state && state->rp) {
         for (unsigned rt = 0; rt < MIN2(state->rp->color_attachment_count, 8);
              ++rt) {
            if (state->rp->color_attachment_formats[rt] == VK_FORMAT_UNDEFINED)
               deferred->undefined_colors |= BITFIELD_BIT(rt);
         }
      }
      obj->apple9_deferred = deferred;
      obj->apple9_native_candidate = native_candidate;
      if (native_candidate) {
         uint8_t policy = hk_apple9_native_stage_policy(hk_device_physical(dev));
         if (infos[i].stage == MESA_SHADER_GEOMETRY)
            obj->apple9_native_stage_mask = policy & HK_APPLE9_NATIVE_GS_BIT;
         if (infos[i].stage == MESA_SHADER_TESS_EVAL)
            obj->apple9_native_stage_mask = policy & HK_APPLE9_NATIVE_TES_BIT;
      }
      obj->apple9_native_cull_primitive =
         infos[i].stage == MESA_SHADER_MESH &&
         ((infos[i].nir->info.outputs_written & VARYING_BIT_CULL_PRIMITIVE) ||
          infos[i].nir->info.cull_distance_array_size);
      simple_mtx_init(&deferred->variant_lock, mtx_plain);
      obj->writes_memory = infos[i].nir->info.writes_memory;
      obj->pre_raster_reads = hk_nir_pre_raster_reads(infos[i].nir);
      obj->reads_draw_id = hk_nir_reads_draw_id(infos[i].nir);
      obj->push_constant_mask = hk_nir_push_constant_mask(infos[i].nir);
      out[i] = &obj->vk;
   }

   for (uint32_t i = 0; i < count; ++i)
      ralloc_free(infos[i].nir);

   if (result != VK_SUCCESS) {
      for (uint32_t i = 0; i < count; ++i) {
         if (out[i])
            hk_api_shader_destroy(&dev->vk, out[i], alloc);
      }
      memset(out, 0, count * sizeof(*out));
   } else if (debug_get_option_cached_hk_apple9_eager_variants()) {
      hk_apple9_eager_variant(dev, count, out, state);
   }
   return result;
}

bool
hk_apple9_deferred_sample_shading(const struct hk_api_shader *fs)
{
   return fs && fs->apple9_deferred && fs->apple9_deferred->sample_shading;
}

/* Whether the fragment shader's pipeline declared a depth/stencil feedback
 * loop; draw-time resolution applies hk_apple9_zs_writes to it. */
bool
hk_apple9_deferred_zs_self_dep(const struct hk_api_shader *fs)
{
   return fs && fs->apple9_deferred && fs->apple9_deferred->zs_self_dep;
}

struct hk_apple9_variant {
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES];
   struct hk_apple9_graphics_key key;
   bool prefer_native;
   struct hk_api_shader *shaders[HK_APPLE9_GFX_STAGES];
   /* Keep the serialized object in the device's weak pipeline cache for the
    * lifetime of the live variant. The disk cache outlives both. */
   struct vk_pipeline_cache_object *cache_object;

   /* Signalled once the compile finished; result holds its outcome. The
    * variant is in the table from the start, so threads that need it wait
    * for this compile while others look up and compile variants freely. */
   struct util_queue_fence ready;
   VkResult result;
};

static uint32_t
hk_apple9_variant_hash(const void *key)
{
   return _mesa_hash_data(key, offsetof(struct hk_apple9_variant, shaders));
}

static bool
hk_apple9_variant_equal(const void *a, const void *b)
{
   return !memcmp(a, b, offsetof(struct hk_apple9_variant, shaders));
}

static void
hk_apple9_variant_destroy(struct hk_device *dev, struct hk_apple9_variant *v)
{
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      if (v->shaders[s])
         hk_api_shader_destroy(&dev->vk, &v->shaders[s]->vk, NULL);
   }
   util_queue_fence_destroy(&v->ready);
   if (v->cache_object)
      vk_pipeline_cache_object_unref(&dev->vk, v->cache_object);
   free(v);
}

static bool
hk_apple9_remove_generic_output(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location < VARYING_SLOT_VAR0)
      return false;
   nir_instr_remove(&intr->instr);
   return true;
}

/* Draw-time variants persist through the device pipeline cache (and so the
 * disk cache), as API shaders do through the runtime. The key covers every
 * input of the variant compile: the deferred stages exactly as the compile
 * reads them, the resolved graphics key, the device features shaders lower
 * against, and the compiler flags (debug, drirc, soft faults, environment
 * knobs).
 */
static bool
hk_apple9_variant_cache_key(struct hk_device *dev,
                            struct hk_api_shader *const api[HK_APPLE9_GFX_STAGES],
                            const struct hk_apple9_graphics_key *key,
                            bool prefer_native, blake3_hash out)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   static const char tag[] = "hk-apple9-variant-v3";
   _mesa_blake3_update(&ctx, tag, sizeof(tag));
   /* The UUID covers the driver build ID, root layout and compiler flags,
    * including when these entries travel inside an application cache. */
   const struct vk_properties *props = &dev->vk.physical->properties;
   _mesa_blake3_update(&ctx, props->pipelineCacheUUID, VK_UUID_SIZE);
   _mesa_blake3_update(&ctx, &props->shaderBinaryVersion,
                       sizeof(props->shaderBinaryVersion));
   _mesa_blake3_update(&ctx, &dev->dev.chip, sizeof(dev->dev.chip));
   const uint64_t flags =
      hk_physical_device_compiler_flags(hk_device_physical(dev));
   _mesa_blake3_update(&ctx, &flags, sizeof(flags));
   const enum hk_feature_key features =
      hk_make_feature_key(&dev->vk.enabled_features);
   _mesa_blake3_update(&ctx, &features, sizeof(features));
   _mesa_blake3_update(&ctx, key, sizeof(*key));
   /* Use the effective request after candidate collapse. Native leaves cannot
    * satisfy an emulated draw, even when every graphics-state field agrees. */
   _mesa_blake3_update(&ctx, &prefer_native, sizeof(prefer_native));
   hk_hash_apple9_native_mesh_abi(&ctx);
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      const uint32_t present = api[s] != NULL;
      _mesa_blake3_update(&ctx, &present, sizeof(present));
      if (!present)
         continue;
      const struct hk_apple9_deferred *d = api[s]->apple9_deferred;
      if (!d)
         return false;
      const uint64_t size = d->nir_size;
      const uint32_t fields[] = {d->next_stage_mask, d->set_layout_count,
                                 d->sample_shading, d->zs_self_dep,
                                 d->undefined_colors, d->cull_distance};
      _mesa_blake3_update(&ctx, &size, sizeof(size));
      _mesa_blake3_update(&ctx, d->nir, d->nir_size);
      _mesa_blake3_update(&ctx, fields, sizeof(fields));
   }
   _mesa_blake3_final(&ctx, out);
   return true;
}

static void
hk_apple9_variant_cache_store(struct hk_device *dev,
                              const unsigned *stages, unsigned count,
                              struct vk_shader *const *compiled,
                              struct blob *blob)
{
   blob_write_uint32(blob, 0x484b5602);
   blob_write_bytes(blob, dev->vk.physical->properties.pipelineCacheUUID,
                     VK_UUID_SIZE);
   blob_write_uint32(blob, dev->vk.physical->properties.shaderBinaryVersion);
   blob_write_uint32(blob, count);
   for (unsigned i = 0; i < count; ++i) {
      blob_write_uint32(blob, stages[i]);
      intptr_t size_at = blob_reserve_uint32(blob);
      const size_t start = blob->size;
      if (size_at < 0 ||
          !compiled[i]->ops->serialize(&dev->vk, compiled[i], blob)) {
         blob->out_of_memory = true;
         return;
      }
      if (blob->size - start > UINT32_MAX) {
         blob->out_of_memory = true;
         return;
      }
      blob_overwrite_uint32(blob, size_at, blob->size - start);
   }
}

/* Returns the number of stages loaded into out, or 0 on a miss. */
static unsigned
hk_apple9_variant_cache_load(struct hk_device *dev, const void *data,
                             size_t size, uint32_t expected_stages,
                             unsigned *stages,
                             struct vk_shader **out)
{
   for (unsigned i = 0; i < HK_APPLE9_GFX_STAGES; ++i)
      out[i] = NULL;
   struct blob_reader blob;
   blob_reader_init(&blob, data, size);
   const uint32_t magic = blob_read_uint32(&blob);
   uint8_t uuid[VK_UUID_SIZE];
   blob_copy_bytes(&blob, uuid, sizeof(uuid));
   const uint32_t version = blob_read_uint32(&blob);
   const unsigned count = blob_read_uint32(&blob);
   if (blob.overrun || magic != 0x484b5602 ||
       memcmp(uuid, dev->vk.physical->properties.pipelineCacheUUID, sizeof(uuid)) ||
       version != dev->vk.physical->properties.shaderBinaryVersion ||
       count == 0 || count != util_bitcount(expected_stages))
      return 0;
   uint32_t remaining = expected_stages;
   unsigned i;
   for (i = 0; i < count; ++i) {
      stages[i] = blob_read_uint32(&blob);
      const uint32_t len = blob_read_uint32(&blob);
      const uint8_t *bytes = blob_read_bytes(&blob, len);
      if (blob.overrun || stages[i] >= HK_APPLE9_GFX_STAGES ||
          stages[i] == MESA_SHADER_COMPUTE ||
          !(remaining & BITFIELD_BIT(stages[i])) || !len)
         break;
      remaining &= ~BITFIELD_BIT(stages[i]);
      struct blob_reader stage;
      blob_reader_init(&stage, bytes, len);
      if (dev->vk.shader_ops->deserialize(&dev->vk, &stage, version, NULL,
                                          &out[i]) != VK_SUCCESS ||
          !out[i] || out[i]->stage != stages[i] || stage.overrun ||
          stage.current != stage.end)
         break;
      /* Persistent draw variants contain compiled realizations only. Let the
       * one driver deserializer own the container format, then check the kind. */
      struct hk_api_shader *api = container_of(out[i], struct hk_api_shader, vk);
      if (api->apple9_deferred)
         break;
   }
   if (i == count && !remaining && blob.current == blob.end)
      return count;
   for (unsigned j = 0; j <= i && j < count; ++j) {
      if (out[j])
         hk_api_shader_destroy(&dev->vk, out[j], NULL);
      out[j] = NULL;
   }
   return 0;
}

VkResult
hk_apple9_resolve_graphics(
   struct hk_device *dev, struct hk_api_shader *const api[HK_APPLE9_GFX_STAGES],
   const struct vk_graphics_pipeline_state *state,
   struct hk_api_shader *out[HK_APPLE9_GFX_STAGES], bool prefer_native,
   bool fs_statistics)
{
   struct hk_apple9_variant search;
   memset(&search, 0, sizeof(search));
   memcpy(search.api, api, sizeof(search.api));
   hk_apple9_graphics_key_device(&search.key, state, hk_device_physical(dev));
   search.key.no_fs_statistics = !fs_statistics;
   const struct hk_api_shader *vs = api[MESA_SHADER_VERTEX];
   search.key.vertex.unbounded = vs && vs->apple9_deferred &&
                                 vs->apple9_deferred->vertex_unbounded;
   mesa_shader_stage producer =
      api[MESA_SHADER_MESH]        ? MESA_SHADER_MESH
      : api[MESA_SHADER_GEOMETRY]  ? MESA_SHADER_GEOMETRY
      : api[MESA_SHADER_TESS_EVAL] ? MESA_SHADER_TESS_EVAL
                                   : MESA_SHADER_MESH;
   /* Collapse ineligible requests onto the emulated cache entry. */
   search.prefer_native = prefer_native && api[producer] &&
                          p_atomic_read(&api[producer]->apple9_native_candidate);
   const struct hk_api_shader *fs = api[MESA_SHADER_FRAGMENT];
   if (fs && fs->apple9_deferred) {
      u_foreach_bit(rt, fs->apple9_deferred->undefined_colors)
         search.key.blend[rt].color_attachment_unused = true;
   }

retry:
   simple_mtx_lock(&dev->apple9_variants.lock);
   if (!dev->apple9_variants.ht) {
      dev->apple9_variants.ht = _mesa_hash_table_create(
         NULL, hk_apple9_variant_hash, hk_apple9_variant_equal);
   }
   struct hash_entry *entry =
      dev->apple9_variants.ht
         ? _mesa_hash_table_search(dev->apple9_variants.ht, &search)
         : NULL;
   if (!entry && dev->apple9_variants.ht && !search.prefer_native) {
      /* A rejected native request already owns emulated programs under its
       * original key. Reuse that completed entry without moving/freeing it:
       * waiters may still hold its fence, and another thread may already be
       * compiling the emulated key. Do not wait on unrelated in-flight work. */
      struct hk_apple9_variant alternate = search;
      alternate.prefer_native = true;
      struct hash_entry *fallback =
         _mesa_hash_table_search(dev->apple9_variants.ht, &alternate);
      if (fallback) {
         struct hk_apple9_variant *f = fallback->data;
         if (util_queue_fence_is_signalled(&f->ready) &&
             f->result == VK_SUCCESS && f->shaders[producer] &&
             !f->shaders[producer]->apple9_native_leaf)
            entry = fallback;
      }
   }
   struct hk_apple9_variant *v;
   if (entry) {
      v = entry->data;
      if (!util_queue_fence_is_signalled(&v->ready)) {
         simple_mtx_unlock(&dev->apple9_variants.lock);
         util_queue_fence_wait(&v->ready);
         goto retry;
      }
      if (v->result != VK_ERROR_OUT_OF_HOST_MEMORY &&
          v->result != VK_ERROR_OUT_OF_DEVICE_MEMORY) {
         VkResult result = v->result;
         if (result == VK_SUCCESS)
            memcpy(out, v->shaders, sizeof(v->shaders));
         simple_mtx_unlock(&dev->apple9_variants.lock);
         return result;
      }

      /* Retry temporary allocation failures in place. Other recording threads
       * may still be waiting on this fence, so removing/freeing the entry is
       * unsafe. Claim the next compile while holding the table lock.
       */
      util_queue_fence_reset(&v->ready);
      simple_mtx_unlock(&dev->apple9_variants.lock);
      goto compile;
   }
   if (!dev->apple9_variants.ht) {
      simple_mtx_unlock(&dev->apple9_variants.lock);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   /* Publish the variant before compiling, then compile without the lock:
    * a draw-time compile takes up to a second, and other recording threads
    * must not block on it for unrelated variants. */
   v = calloc(1, sizeof(*v));
   if (!v) {
      simple_mtx_unlock(&dev->apple9_variants.lock);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   memcpy(v, &search, offsetof(struct hk_apple9_variant, shaders));
   util_queue_fence_init(&v->ready);
   util_queue_fence_reset(&v->ready);
   _mesa_hash_table_insert(dev->apple9_variants.ht, v, v);
   simple_mtx_unlock(&dev->apple9_variants.lock);

compile:;
   /* HK_APPLE9_VARIANT_CACHE_VERIFY=1 compiles on a cache hit as well and
    * reports any difference, to catch inputs missing from the key. */
   const bool verify = debug_get_option_apple9_variant_cache_verify();
   blake3_hash cache_key;
   const bool cacheable =
      dev->vk.mem_cache &&
      hk_apple9_variant_cache_key(dev, api, &search.key, search.prefer_native,
                                    cache_key);
   struct vk_shader *cached[HK_APPLE9_GFX_STAGES];
   unsigned cached_stages[HK_APPLE9_GFX_STAGES];
   unsigned cached_count = 0;
   struct vk_pipeline_cache_object *cache_object = NULL;
   if (cacheable) {
      uint32_t expected_stages = 0;
      const struct hk_apple9_deferred *last = NULL;
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
         if (api[s]) {
            expected_stages |= BITFIELD_BIT(s);
            if (s != MESA_SHADER_FRAGMENT)
               last = api[s]->apple9_deferred;
         }
      }
      if (!api[MESA_SHADER_FRAGMENT] && last && last->cull_distance)
         expected_stages |= BITFIELD_BIT(MESA_SHADER_FRAGMENT);
      cache_object = vk_pipeline_cache_lookup_object(
         dev->vk.mem_cache, cache_key, sizeof(cache_key),
         &vk_raw_data_cache_object_ops, NULL);
      if (cache_object) {
         struct vk_raw_data_cache_object *raw =
            container_of(cache_object, struct vk_raw_data_cache_object, base);
         cached_count = hk_apple9_variant_cache_load(
            dev, raw->data, raw->data_size, expected_stages, cached_stages, cached);
         if (!cached_count) {
            vk_pipeline_cache_object_unref(&dev->vk, cache_object);
            cache_object = NULL;
         }
      }
   }
   if (cached_count && !verify) {
      for (unsigned i = 0; i < cached_count; ++i) {
         v->shaders[cached_stages[i]] =
            container_of(cached[i], struct hk_api_shader, vk);
      }
      memcpy(out, v->shaders, sizeof(v->shaders));
      v->cache_object = cache_object;
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
         if (api[s])
            hk_apple9_remember_variant(dev, api[s], cache_object);
      simple_mtx_lock(&dev->apple9_variants.lock);
      if (search.prefer_native && !v->shaders[producer]->apple9_native_leaf)
         p_atomic_set(&api[producer]->apple9_native_candidate, false);
      v->result = VK_SUCCESS;
      util_queue_fence_signal(&v->ready);
      simple_mtx_unlock(&dev->apple9_variants.lock);
      return VK_SUCCESS;
   }

   struct vk_shader_compile_info infos[HK_APPLE9_GFX_STAGES];
   unsigned stages[HK_APPLE9_GFX_STAGES];
   unsigned count = 0;
   VkResult result = VK_SUCCESS;
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s) {
      if (!api[s])
         continue;
      const struct hk_apple9_deferred *d = api[s]->apple9_deferred;
      if (!d) {
         /* Draw-time variants recompile every stage of the draw. */
         result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                            "Apple9: deferred and precompiled stages mixed");
         break;
      }
      struct blob_reader reader;
      blob_reader_init(&reader, d->nir, d->nir_size);
      nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &reader);
      if (!nir) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         break;
      }
      infos[count] = (struct vk_shader_compile_info){
         .stage = s,
         .nir = nir,
         .next_stage_mask = d->next_stage_mask,
         .robustness = &vk_robustness_disabled,
         .set_layout_count = d->set_layout_count,
      };
      stages[count++] = s;
   }

   /* Without a fragment shader nothing reads the generic varyings of a
    * vertex-only draw: drop them so their stores and the arithmetic feeding
    * them die. The variant is keyed on the bound stages, so the fragment
    * shader is known absent for every draw that uses it. Transform feedback
    * still captures them. This runs before the cull-distance null program
    * below is linked: it reads nothing, but as a consumer it would otherwise
    * keep every generic output.
    */
   if (result == VK_SUCCESS && !api[MESA_SHADER_FRAGMENT] && count == 1 &&
       infos[0].stage == MESA_SHADER_VERTEX && !infos[0].nir->xfb_info) {
      nir_shader *vs = infos[0].nir;
      if (nir_shader_intrinsics_pass(vs, hk_apple9_remove_generic_output,
                                     nir_metadata_control_flow, NULL)) {
         NIR_PASS(_, vs, nir_opt_dce);
         nir_shader_gather_info(vs, nir_shader_get_entrypoint(vs));
      }
   }

   /* Cull distances are applied by the fragment program, so a draw without
    * one links an empty program to the last pre-rasterization stage. */
   if (result == VK_SUCCESS && !api[MESA_SHADER_FRAGMENT] && count &&
       infos[count - 1].nir->info.cull_distance_array_size) {
      infos[count] = (struct vk_shader_compile_info){
         .stage = MESA_SHADER_FRAGMENT,
         .nir = hk_apple9_null_fs_nir(true),
         .robustness = &vk_robustness_disabled,
      };
      stages[count++] = MESA_SHADER_FRAGMENT;
   }

   struct vk_shader *compiled[HK_APPLE9_GFX_STAGES] = {NULL};
   if (result == VK_SUCCESS) {
      struct hk_cpu_compile_scope scope;
      hk_cpu_timing_compile_begin(&scope);
      /* A draw-time compile stalls the recording thread: let its register
       * allocation use one helper thread. Eager (pipeline-time) variants
       * stay serial so bulk compiles don't oversubscribe the CPU. */
      agx_apple9_set_parallel_ra(!hk_apple9_eager_compile);
      /* Creation-time incompatibility is final. Dynamic state can narrow
       * a candidate further, but cannot make an incompatible shader native. */
      if (!search.prefer_native) {
         result = hk_apple9_compile_graphics_path(
            dev, count, infos, NULL, &search.key, NULL, NULL, compiled,
            false, NULL);
      } else {
         result = hk_apple9_compile_graphics_mode(
            dev, count, infos, NULL, &search.key, NULL, NULL, compiled, true);
      }
      agx_apple9_set_parallel_ra(false);
      hk_cpu_timing_compile_end(&scope, true);
   } else {
      for (unsigned i = 0; i < count; ++i)
         ralloc_free(infos[i].nir);
   }

   /* Keep permanent failures, but allow a later caller to retry an allocation
    * failure once temporary memory pressure has passed. */
   if (result != VK_SUCCESS) {
      for (unsigned i = 0; i < count; ++i) {
         if (compiled[i])
            hk_api_shader_destroy(&dev->vk, compiled[i], NULL);
      }
   } else {
      if (cacheable) {
         struct blob blob;
         blob_init(&blob);
         hk_apple9_variant_cache_store(dev, stages, count, compiled,
                                       &blob);
         if (cached_count) {
            struct blob check;
            blob_init(&check);
            hk_apple9_variant_cache_store(dev, cached_stages,
                                          cached_count, cached, &check);
            if (blob.out_of_memory || check.out_of_memory ||
                blob.size != check.size ||
                memcmp(blob.data, check.data, blob.size)) {
               mesa_loge("hk: draw-time variant cache mismatch pid=%u (%zu vs %zu "
                         "bytes): an input is missing from the key",
                         (unsigned)getpid(), blob.size, check.size);
            } else {
               mesa_logi("hk: draw-time variant cache verified pid=%u (%zu bytes)",
                         (unsigned)getpid(), blob.size);
            }
            blob_finish(&check);
         } else if (!blob.out_of_memory) {
            cache_object = vk_pipeline_cache_create_and_insert_object(
                  dev->vk.mem_cache, cache_key, sizeof(cache_key), blob.data,
                  blob.size, &vk_raw_data_cache_object_ops);
         }
         blob_finish(&blob);
      }
      for (unsigned i = 0; i < count; ++i) {
         v->shaders[stages[i]] =
            container_of(compiled[i], struct hk_api_shader, vk);
      }
      memcpy(out, v->shaders, sizeof(v->shaders));
   }
   for (unsigned i = 0; i < cached_count; ++i)
      hk_api_shader_destroy(&dev->vk, cached[i], NULL);
   if (result != VK_SUCCESS && cache_object) {
      vk_pipeline_cache_object_unref(&dev->vk, cache_object);
      cache_object = NULL;
   }
   v->cache_object = cache_object;
   if (cache_object) {
      for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
         if (api[s])
            hk_apple9_remember_variant(dev, api[s], cache_object);
   }
   simple_mtx_lock(&dev->apple9_variants.lock);
   if (result == VK_SUCCESS && search.prefer_native &&
       !v->shaders[producer]->apple9_native_leaf)
      p_atomic_set(&api[producer]->apple9_native_candidate, false);
   v->result = result;
   util_queue_fence_signal(&v->ready);
   simple_mtx_unlock(&dev->apple9_variants.lock);
   return result;
}

/* Variants live as long as all of their API shaders. */
static void
hk_apple9_forget_deferred(struct hk_device *dev, struct hk_api_shader *obj)
{
   simple_mtx_lock(&dev->apple9_variants.lock);
   if (dev->apple9_variants.ht) {
      hash_table_foreach(dev->apple9_variants.ht, entry) {
         struct hk_apple9_variant *v = entry->data;
         bool uses = false;
         for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
            uses |= v->api[s] == obj;
         if (uses) {
            _mesa_hash_table_remove(dev->apple9_variants.ht, entry);
            hk_apple9_variant_destroy(dev, v);
         }
      }
   }
   simple_mtx_unlock(&dev->apple9_variants.lock);
}

void
hk_apple9_variants_finish(struct hk_device *dev)
{
   if (dev->apple9_variants.ht) {
      hash_table_foreach(dev->apple9_variants.ht, entry)
         hk_apple9_variant_destroy(dev, entry->data);
      _mesa_hash_table_destroy(dev->apple9_variants.ht, NULL);
      dev->apple9_variants.ht = NULL;
   }
}

static VkResult
hk_compile_shaders_timed(struct vk_device *vk_dev, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   agx_timeline_emit(100, 103, 0, shader_count, 0, 0, 0, 0, 0, 0);
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   nir_shader *shaders[shader_count];

   /* Lower shaders, notably lowering I/O. This is a prerequisite for
    * intershader optimization.
    */
   for (uint32_t i = 0; i < shader_count; i++) {
      /* Shader objects have no pipeline robustness, but still follow the
       * device's robustness features. The runtime passes them disabled.
       */
      if (infos[i].robustness == &vk_robustness_disabled)
         infos[i].robustness = &dev->vk.robustness_state;

      const struct vk_shader_compile_info *info = &infos[i];
      /* TODO: Multiview with ESO */
      const bool is_multiview = state && state->mv->view_mask != 0;
      enum hk_feature_key hk_features = hk_make_feature_key(features);
      nir_shader *nir = info->nir;

      hk_lower_nir(dev, nir, info->robustness, is_multiview,
                   info->set_layout_count, info->set_layouts, hk_features);

      if (nir->xfb_info) {
         /* Fold constant offset srcs for IO. */
         NIR_PASS(_, nir, nir_opt_constant_folding);

         nir_io_add_intrinsic_xfb_info(nir);
      }

      shaders[i] = nir;
   }

   /* Mesh primitive outputs link to plain flat inputs. */
   bool mesh = false;
   for (uint32_t i = 0; i < shader_count; i++)
      mesh |= infos[i].stage == MESA_SHADER_MESH;
   if (!mesh) {
      nir_opt_varyings_bulk(shaders, shader_count, true, UINT32_MAX,
                            UINT32_MAX, nir_opts, NULL);
   } else {
      /* Interpolation masks are otherwise gathered by the linker. */
      for (uint32_t i = 0; i < shader_count; i++)
         nir_shader_gather_info(shaders[i], nir_shader_get_entrypoint(shaders[i]));
   }

   if (dev->dev.chip == AGX_CHIP_G16G && shader_count &&
       infos[0].stage != MESA_SHADER_COMPUTE) {
      if (hk_apple9_needs_deferral(shader_count, infos, state, features))
         return hk_apple9_defer_graphics(dev, shader_count, infos, state,
                                         pAllocator, shaders_out);
      return hk_apple9_compile_graphics(dev, shader_count, infos, state,
                                       NULL, NULL, pAllocator, shaders_out);
   }

   for (uint32_t i = 0; i < shader_count; i++) {
      VkResult result =
         hk_compile_shader(dev, &infos[i], state, features, pAllocator,
                           (struct hk_api_shader **)&shaders_out[i]);
      if (result != VK_SUCCESS) {
         /* Clean up all the shaders before this point */
         for (uint32_t j = 0; j < i; j++)
            hk_api_shader_destroy(&dev->vk, shaders_out[j], pAllocator);

         /* Clean up all the NIR after this point */
         for (uint32_t j = i + 1; j < shader_count; j++)
            ralloc_free(infos[j].nir);

         /* Memset the output array */
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

         return result;
      }
   }

   return VK_SUCCESS;
}

static VkResult
hk_compile_shaders(struct vk_device *vk_dev, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   struct hk_cpu_compile_scope scope;
   hk_cpu_timing_compile_begin(&scope);
   VkResult result = hk_compile_shaders_timed(vk_dev, shader_count, infos,
                                              state, features, pAllocator,
                                              shaders_out);
   hk_cpu_timing_compile_end(&scope, false);
   return result;
}

static VkResult
hk_deserialize_shader(struct hk_device *dev, struct blob_reader *blob,
                      struct hk_shader *shader)
{
   struct hk_shader_info info;
   blob_copy_bytes(blob, &info, sizeof(info));

   struct agx_shader_info b_info;
   blob_copy_bytes(blob, &b_info, sizeof(b_info));

   const uint32_t code_size = blob_read_uint32(blob);
   const uint32_t data_size = blob_read_uint32(blob);
   /* Validate payload lengths before any code/data allocation, including
    * nested native realizations and shader binaries outside the variant cache.
    * Subtract successively so even a 32-bit host cannot overflow the sum. */
   if (blob->overrun || (unsigned)info.stage >= MESA_SHADER_MESH_STAGES ||
       code_size > blob->end - blob->current ||
       data_size > (blob->end - blob->current) - code_size)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   VkResult result = hk_init_link_ht(shader, info.stage);
   if (result != VK_SUCCESS)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   simple_mtx_init(&shader->linked.lock, mtx_plain);

   shader->b.info = b_info;
   shader->info = info;
   shader->code_size = code_size;
   shader->data_size = data_size;
   shader->b.info.binary_size = code_size;

   shader->code_ptr = malloc(MAX2(code_size, 1));
   if (shader->code_ptr == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   shader->data_ptr = malloc(MAX2(data_size, 1));
   if (shader->data_ptr == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   blob_copy_bytes(blob, (void *)shader->code_ptr, shader->code_size);
   blob_copy_bytes(blob, (void *)shader->data_ptr, shader->data_size);
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   shader->b.binary = (void *)shader->code_ptr;
   return hk_upload_shader(dev, shader);
}

/* Version the container independently of the hardware programs it owns. */
#define HK_SHADER_CONTAINER_VERSION 0x484b0007u

static VkResult
hk_deserialize_api_shader_inner(struct vk_device *vk_dev, struct blob_reader *blob,
                          uint32_t binary_version,
                          const VkAllocationCallbacks *pAllocator,
                          struct vk_shader **shader_out, bool allow_native)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);

   if (blob_read_uint32(blob) != HK_SHADER_CONTAINER_VERSION)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   mesa_shader_stage stage = blob_read_uint8(blob);
   bool native_candidate = blob_read_uint8(blob);
   if (blob->overrun || stage >= MESA_SHADER_MESH_STAGES)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   size_t size = sizeof(struct hk_api_shader) +
                 sizeof(struct hk_shader) * hk_num_variants(stage);

   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, stage, pAllocator, size);

   if (obj == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   obj->apple9_native_candidate = native_candidate;
   obj->apple9_native_cull_primitive = blob_read_uint8(blob);
   obj->apple9_native_leaf = blob_read_uint8(blob);
   obj->apple9_native_stage_mask = blob_read_uint8(blob);
   uint8_t stage_policy = hk_apple9_native_stage_policy(hk_device_physical(dev));
   uint8_t mask = obj->apple9_native_stage_mask;
   bool pair_stage =
      mask == 0 ||
      (mask == HK_APPLE9_NATIVE_GS_BIT &&
       (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY ||
        stage == MESA_SHADER_FRAGMENT)) ||
      (mask == HK_APPLE9_NATIVE_TES_BIT &&
       (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_CTRL ||
        stage == MESA_SHADER_TESS_EVAL || stage == MESA_SHADER_FRAGMENT));
   if (!pair_stage || (mask & ~stage_policy)) {
      hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   blob_copy_bytes(blob, obj->apple9_native_pair_id,
                   sizeof(obj->apple9_native_pair_id));
   bool gs_leaf = (hk_apple9_native_gs_enabled() ||
                   (obj->apple9_native_stage_mask == HK_APPLE9_NATIVE_GS_BIT)) &&
                  (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY);
   bool tess_leaf =
      (hk_apple9_native_tess_enabled() ||
       (obj->apple9_native_stage_mask == HK_APPLE9_NATIVE_TES_BIT)) &&
      (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_CTRL ||
       stage == MESA_SHADER_TESS_EVAL);
   if (blob->overrun ||
       (obj->apple9_native_leaf && stage != MESA_SHADER_MESH &&
        stage != MESA_SHADER_FRAGMENT && !gs_leaf && !tess_leaf)) {
      hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }

   if (blob_read_uint8(blob)) {
      if (obj->apple9_native_leaf ||
          (mask && stage != MESA_SHADER_GEOMETRY && stage != MESA_SHADER_TESS_EVAL)) {
         hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
      struct hk_apple9_deferred *d = calloc(1, sizeof(*d));
      if (!d) {
         vk_shader_free(&dev->vk, pAllocator, &obj->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      d->next_stage_mask = blob_read_uint32(blob);
      d->set_layout_count = blob_read_uint32(blob);
      d->sample_shading = blob_read_uint8(blob);
      d->zs_self_dep = blob_read_uint8(blob);
      d->vertex_unbounded = blob_read_uint8(blob);
      d->undefined_colors = blob_read_uint8(blob);
      d->cull_distance = blob_read_uint8(blob);
      obj->writes_memory = blob_read_uint8(blob);
      obj->pre_raster_reads = blob_read_uint8(blob);
      obj->reads_draw_id = blob_read_uint8(blob);
      blob_copy_bytes(blob, &obj->push_constant_mask,
                      sizeof(obj->push_constant_mask));
      d->nir_size = blob_read_uint32(blob);
      const void *nir = blob_read_bytes(blob, d->nir_size);
      d->nir = blob->overrun ? NULL : malloc(MAX2(d->nir_size, 1));
      if (!d->nir) {
         free(d);
         vk_shader_free(&dev->vk, pAllocator, &obj->vk);
         return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
      memcpy(d->nir, nir, d->nir_size);
      obj->apple9_deferred = d;
      simple_mtx_init(&d->variant_lock, mtx_plain);
      if (!hk_apple9_deserialize_variants(dev, obj, blob)) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
      *shader_out = &obj->vk;
      return VK_SUCCESS;
   }

   hk_foreach_variant(obj, shader) {
      VkResult result = hk_deserialize_shader(dev, blob, shader);

      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   }
   obj->writes_memory = blob_read_uint8(blob);
   if (obj->apple9_native_leaf && stage == MESA_SHADER_TESS_EVAL)
      blob_copy_bytes(blob, &obj->apple9_native_tess,
                      sizeof(obj->apple9_native_tess));
   obj->pre_raster_reads = blob_read_uint8(blob);
   obj->reads_draw_id = blob_read_uint8(blob);
   blob_copy_bytes(blob, &obj->push_constant_mask,
                   sizeof(obj->push_constant_mask));
   bool has_native = blob_read_uint8(blob);
   if (blob->overrun ||
       (has_native && (!allow_native || obj->apple9_native_leaf)) ||
       (mask && !has_native && !obj->apple9_native_leaf)) {
      hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   if (has_native) {
      struct vk_shader *native = NULL;
      VkResult result = hk_deserialize_api_shader_inner(
         vk_dev, blob, binary_version, pAllocator, &native, false);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
         return result;
      }
      obj->apple9_native = container_of(native, struct hk_api_shader, vk);
      if (native->stage != stage || obj->apple9_native->apple9_deferred ||
          !obj->apple9_native->apple9_native_leaf ||
          obj->apple9_native->apple9_native_stage_mask != mask) {
         hk_api_shader_destroy(vk_dev, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
   }

   hk_api_shader_init_constant_data(obj);
   *shader_out = &obj->vk;
   return VK_SUCCESS;
}

static VkResult
hk_deserialize_api_shader(struct vk_device *vk_dev, struct blob_reader *blob,
                          uint32_t binary_version,
                          const VkAllocationCallbacks *pAllocator,
                          struct vk_shader **shader_out)
{
   return hk_deserialize_api_shader_inner(vk_dev, blob, binary_version,
                                           pAllocator, shader_out, true);
}

static void
hk_shader_serialize(struct vk_device *vk_dev, const struct hk_shader *shader,
                    struct blob *blob)
{
   blob_write_bytes(blob, &shader->info, sizeof(shader->info));
   blob_write_bytes(blob, &shader->b.info, sizeof(shader->b.info));

   blob_write_uint32(blob, shader->code_size);
   blob_write_uint32(blob, shader->data_size);
   blob_write_bytes(blob, shader->code_ptr, shader->code_size);
   blob_write_bytes(blob, shader->data_ptr, shader->data_size);
}

static bool
hk_api_shader_serialize(struct vk_device *vk_dev,
                        const struct vk_shader *vk_shader, struct blob *blob)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   blob_write_uint32(blob, HK_SHADER_CONTAINER_VERSION);
   blob_write_uint8(blob, vk_shader->stage);
   blob_write_uint8(blob, p_atomic_read(&obj->apple9_native_candidate));
   blob_write_uint8(blob, obj->apple9_native_cull_primitive);
   blob_write_uint8(blob, obj->apple9_native_leaf);
   blob_write_uint8(blob, obj->apple9_native_stage_mask);
   blob_write_bytes(blob, obj->apple9_native_pair_id,
                    sizeof(obj->apple9_native_pair_id));

   struct hk_apple9_deferred *d = obj->apple9_deferred;
   blob_write_uint8(blob, d != NULL);
   if (d) {
      blob_write_uint32(blob, d->next_stage_mask);
      blob_write_uint32(blob, d->set_layout_count);
      blob_write_uint8(blob, d->sample_shading);
      blob_write_uint8(blob, d->zs_self_dep);
      blob_write_uint8(blob, d->vertex_unbounded);
      blob_write_uint8(blob, d->undefined_colors);
      blob_write_uint8(blob, d->cull_distance);
      blob_write_uint8(blob, obj->writes_memory);
      blob_write_uint8(blob, obj->pre_raster_reads);
      blob_write_uint8(blob, obj->reads_draw_id);
      blob_write_bytes(blob, &obj->push_constant_mask,
                       sizeof(obj->push_constant_mask));
      blob_write_uint32(blob, d->nir_size);
      blob_write_bytes(blob, d->nir, d->nir_size);
      return hk_apple9_serialize_variants(d, blob);
   }

   hk_foreach_variant(obj, shader) {
      hk_shader_serialize(vk_dev, shader, blob);
   }
   /* Preserve the API shader's memory effects on a cache hit. In particular,
    * a read-only vertex shader must not turn its render into a memory writer
    * and prevent later renders from overlapping tiling with fragment work.
    */
   blob_write_uint8(blob, obj->writes_memory);
   if (obj->apple9_native_leaf && obj->vk.stage == MESA_SHADER_TESS_EVAL)
      blob_write_bytes(blob, &obj->apple9_native_tess,
                       sizeof(obj->apple9_native_tess));
   blob_write_uint8(blob, obj->pre_raster_reads);
   blob_write_uint8(blob, obj->reads_draw_id);
   blob_write_bytes(blob, &obj->push_constant_mask,
                    sizeof(obj->push_constant_mask));

   blob_write_uint8(blob, obj->apple9_native != NULL);
   if (obj->apple9_native) {
      assert(!obj->apple9_native->apple9_native &&
             !obj->apple9_native->apple9_deferred &&
             obj->apple9_native->apple9_native_leaf &&
             !obj->apple9_native_leaf);
      hk_api_shader_serialize(vk_dev, &obj->apple9_native->vk, blob);
   }

   return !blob->out_of_memory;
}

static VkResult
hk_shader_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props)
   {
      props->stages = mesa_to_vk_shader_stage(obj->vk.stage);
      props->subgroupSize = 32;
      VK_COPY_STR(props->name, _mesa_shader_stage_to_string(obj->vk.stage));
      VK_PRINT_STR(props->description, "%s shader",
                   _mesa_shader_stage_to_string(obj->vk.stage));
   }

   return vk_outarray_status(&out);
}

static VkResult
hk_shader_get_executable_statistics(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, statistics,
                          statistic_count);

   assert(executable_index == 0);

   /* TODO: find a sane way to report multiple variants and have that play nice
    * with zink.
    */
   struct hk_shader *shader = hk_any_variant(obj);
   vk_add_agx2_stats(out, &shader->b.info.stats);
   return vk_outarray_status(&out);
}

static bool
write_ir_text(VkPipelineExecutableInternalRepresentationKHR *ir,
              const char *data)
{
   ir->isText = VK_TRUE;

   size_t data_len = strlen(data) + 1;

   if (ir->pData == NULL) {
      ir->dataSize = data_len;
      return true;
   }

   strncpy(ir->pData, data, ir->dataSize);
   if (ir->dataSize < data_len)
      return false;

   ir->dataSize = data_len;
   return true;
}

static VkResult
hk_shader_get_executable_internal_representations(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableInternalRepresentationKHR, out,
                          internal_representations,
                          internal_representation_count);
   bool incomplete_text = false;

   assert(executable_index == 0);

   /* TODO */
#if 0
   vk_outarray_append_typed(VkPipelineExecutableInternalRepresentationKHR, &out, ir) {
      VK_COPY_STR(ir->name, "AGX assembly");
      VK_COPY_STR(ir->description, "AGX assembly");
      if (!write_ir_text(ir, TODO))
         incomplete_text = true;
   }
#endif

   return incomplete_text ? VK_INCOMPLETE : vk_outarray_status(&out);
}

static const struct vk_shader_ops hk_shader_ops = {
   .destroy = hk_api_shader_destroy,
   .serialize = hk_api_shader_serialize,
   .get_executable_properties = hk_shader_get_executable_properties,
   .get_executable_statistics = hk_shader_get_executable_statistics,
   .get_executable_internal_representations =
      hk_shader_get_executable_internal_representations,
};

const struct vk_device_shader_ops hk_device_shader_ops = {
   .get_nir_options = hk_get_nir_options,
   .get_spirv_options = hk_get_spirv_options,
   .preprocess_nir = hk_preprocess_nir,
   .hash_state = hk_hash_graphics_state,
   .compile = hk_compile_shaders,
   .deserialize = hk_deserialize_api_shader,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .cmd_bind_shaders = hk_cmd_bind_shaders,
};

struct hk_linked_shader *
hk_fast_link(struct hk_device *dev, bool fragment, struct hk_shader *main,
             struct agx_shader_part *prolog, struct agx_shader_part *epilog,
             unsigned nr_samples_shaded)
{
   struct hk_linked_shader *s = rzalloc(NULL, struct hk_linked_shader);
   agx_fast_link(&s->b, &dev->dev, fragment, &main->b, prolog, epilog,
                 nr_samples_shaded);

   if (fragment) {
      unsigned samplers = main->b.info.sampler_state_count;
      if (s->b.uses_txf)
         samplers = MAX2(samplers, 1);

      agx_pack(&s->fs_counts, FRAGMENT_SHADER_WORD_0, cfg) {
         cfg.cf_binding_count = s->b.cf.nr_bindings;
         cfg.uniform_register_count = main->b.info.push_count;
         cfg.preshader_register_count = main->b.info.nr_preamble_gprs;
         cfg.texture_state_register_count = main->b.info.texture_state_count;
         cfg.sampler_state_register_count =
            agx_translate_sampler_state_count(samplers, false);
      }
   }

   /* Now that we've linked, bake the USC words to bind this program */
   struct agx_usc_builder b = agx_usc_builder(s->usc.data, sizeof(s->usc.data));

   if (main && main->b.info.rodata.size_16) {
      agx_usc_immediates(&b, &main->b.info.rodata, main->bo->va->addr);
   }

   if (s->b.uses_txf)
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);

   agx_usc_shared_non_fragment(&b, &main->b.info, 0);
   agx_usc_push_packed(&b, SHADER, s->b.shader);
   agx_usc_push_packed(&b, REGISTERS, s->b.regs);

   if (fragment)
      agx_usc_push_packed(&b, FRAGMENT_PROPERTIES, s->b.fragment_props);

   if (main && main->b.info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(&dev->dev, main->preamble_addr);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   s->usc.size = b.head - s->usc.data;
   return s;
}
