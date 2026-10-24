/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Compile, serialize, reload and destroy the real driver shader objects. Only
 * BO allocation is replaced with host memory; no command execution is claimed.
 */
#include "../hk_shader.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
#include "hk-test-usc.h"

static unsigned allocations, live_bos, fail_allocation;
static unsigned native_compiles, emulated_compiles, fragment_compiles;
static bool reject_native;

bool __real_agx_compile_apple9_native_mesh(
   nir_shader *, bool, const struct agx_apple9_native_mesh_options *,
   struct agx_shader_part *, struct agx_apple9_mesh_body_info *, const char **);
bool __wrap_agx_compile_apple9_native_mesh(
   nir_shader *, bool, const struct agx_apple9_native_mesh_options *,
   struct agx_shader_part *, struct agx_apple9_mesh_body_info *, const char **);
bool
__wrap_agx_compile_apple9_native_mesh(
   nir_shader *nir, bool enable,
   const struct agx_apple9_native_mesh_options *options,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   ++native_compiles;
   if (reject_native) {
      *reason = "injected native backend capability rejection";
      memset(out, 0, sizeof(*out));
      return false;
   }
   return __real_agx_compile_apple9_native_mesh(nir, enable, options, out, info,
                                                reason);
}

bool __real_agx_compile_apple9_tiny(nir_shader *, struct agx_shader_part *,
                                    struct agx_apple9_compute_profile *,
                                    const char **);
bool __wrap_agx_compile_apple9_tiny(nir_shader *, struct agx_shader_part *,
                                    struct agx_apple9_compute_profile *,
                                    const char **);
bool
__wrap_agx_compile_apple9_tiny(nir_shader *nir, struct agx_shader_part *out,
                               struct agx_apple9_compute_profile *profile,
                               const char **reason)
{
   ++emulated_compiles;
   return __real_agx_compile_apple9_tiny(nir, out, profile, reason);
}

bool __real_agx_compile_apple9_fragment_mrt(
   nir_shader *, const struct agx_apple9_varying_layout *,
   const struct agx_apple9_blend *, unsigned, struct agx_shader_part *,
   const char **);
bool __wrap_agx_compile_apple9_fragment_mrt(
   nir_shader *, const struct agx_apple9_varying_layout *,
   const struct agx_apple9_blend *, unsigned, struct agx_shader_part *,
   const char **);
bool
__wrap_agx_compile_apple9_fragment_mrt(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, unsigned nr_targets,
   struct agx_shader_part *out, const char **reason)
{
   ++fragment_compiles;
   return __real_agx_compile_apple9_fragment_mrt(nir, varyings, blend,
                                                 nr_targets, out, reason);
}

struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_reference(struct agx_bo *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
VkResult __wrap___vk_errorf(const void *, VkResult, const char *, int,
                            const char *, ...);

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   if (++allocations == fail_allocation)
      return NULL;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, size);
   CHECK(bo->va && bo->_map);
   bo->dev = dev;
   bo->size = size;
   bo->refcnt = 1;
   bo->va->addr = dev->shader_base + allocations * 0x100000;
   ++live_bos;
   return bo;
}

void
__wrap_agx_bo_reference(struct agx_bo *bo)
{
   CHECK(bo && bo->refcnt > 0);
   ++bo->refcnt;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   CHECK(bo->refcnt > 0);
   if (--bo->refcnt)
      return;
   free(bo->_map);
   free(bo->va);
   free(bo);
   --live_bos;
}

VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   if (format && !fail_allocation) {
      va_list args;
      va_start(args, format);
      vfprintf(stderr, format, args);
      va_end(args);
      fputc('\n', stderr);
   }
   return error;
}

static nir_shader *
source(bool fragment, unsigned vertices, unsigned primitives)
{
   nir_builder b = nir_builder_init_simple_shader(
      fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_MESH, &agx_nir_options,
      "hybrid ownership");
   b.shader->info.io_lowered = true;
   if (fragment) {
      nir_def *value = nir_load_per_primitive_input(&b, 4, 32,
         nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   } else {
      b.shader->info.workgroup_size[0] = b.shader->info.workgroup_size[1] =
         b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.max_vertices_out = vertices;
      b.shader->info.mesh.max_primitives_out = primitives;
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                         nir_imm_int(&b, 1), nir_imm_int(&b, 0));
      for (unsigned v = 0; v < 3; ++v)
         nir_store_per_vertex_output(&b,
            nir_imm_vec4(&b, v == 0 ? -.5 : .5, v == 2 ? .5 : -.5, .5, 1),
            nir_imm_int(&b, v), nir_imm_int(&b, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      nir_store_per_primitive_output(&b, nir_imm_vec4(&b, .2, .4, .6, 1),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }
   /* Both independently lowered realizations retain the original table. */
   b.shader->constant_data_size = 64;
   b.shader->constant_data = ralloc_size(b.shader, 64);
   memset(b.shader->constant_data, fragment ? 0x35 : 0x57, 64);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static struct hk_api_shader *
roundtrip(struct hk_device *dev, struct hk_api_shader *src)
{
   struct blob blob;
   blob_init(&blob);
   CHECK(hk_api_shader_serialize(&dev->vk, &src->vk, &blob));
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct vk_shader *result = NULL;
   CHECK(hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &result) == VK_SUCCESS);
   CHECK(!reader.overrun && reader.current == reader.end);
   blob_finish(&blob);
   struct hk_api_shader *copy = container_of(result, struct hk_api_shader, vk);
   CHECK(copy->apple9_native_leaf == src->apple9_native_leaf);
   CHECK(copy->apple9_native_stage_mask == src->apple9_native_stage_mask);
   CHECK(!memcmp(copy->apple9_native_pair_id, src->apple9_native_pair_id,
                 sizeof(copy->apple9_native_pair_id)));
   if (src->apple9_native) {
      CHECK(copy->apple9_native && copy->apple9_native->apple9_native_leaf);
      CHECK(!memcmp(copy->apple9_native->apple9_native_pair_id,
                    src->apple9_native->apple9_native_pair_id,
                    sizeof(copy->apple9_native_pair_id)));
   }
   return copy;
}

static void
check_programs(struct hk_api_shader *mesh, struct hk_api_shader *fs, bool native)
{
   CHECK(!mesh->apple9_deferred && !fs->apple9_deferred);
   if (!mesh->apple9_native_leaf) {
      CHECK(!fs->apple9_native_leaf);
      CHECK(mesh->variants[HK_MESH_VARIANT_MAIN].info.apple9_native_stage ==
            HK_APPLE9_NATIVE_NONE);
      CHECK(mesh->variants[HK_MESH_VARIANT_RAST].bo);
      CHECK(!mesh->variants[HK_MESH_VARIANT_RAST].b.info.apple9_body_only);
   }
   CHECK(!!hk_api_shader_native(mesh) == native &&
         !!hk_api_shader_native(fs) == native);
   CHECK(mesh->constant_data_addr && fs->constant_data_addr);
   if (native) {
      struct hk_api_shader *nm = hk_api_shader_native(mesh),
                           *nf = hk_api_shader_native(fs);
      CHECK(!nm->apple9_native && !nf->apple9_native);
      CHECK(nm->apple9_native_leaf && nf->apple9_native_leaf);
      const blake3_hash zero = {0};
      CHECK(memcmp(nm->apple9_native_pair_id, zero, sizeof(zero)));
      CHECK(!memcmp(nm->apple9_native_pair_id, nf->apple9_native_pair_id,
                    sizeof(nm->apple9_native_pair_id)));
      CHECK(nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native_stage == HK_APPLE9_NATIVE_MESH);
      CHECK(nm->variants[HK_MESH_VARIANT_OBJECT].info.apple9_native_stage == HK_APPLE9_NATIVE_OBJECT);
      CHECK(nm->variants[HK_MESH_VARIANT_MAIN].b.info.apple9_varyings.mask[VARYING_SLOT_VAR0] == 15);
      CHECK(nf->variants[0].b.info.apple9_varyings.mask[VARYING_SLOT_VAR0] == 15);
      CHECK(nm->variants[HK_MESH_VARIANT_MAIN].data_size == mesh->variants[HK_MESH_VARIANT_MAIN].data_size);
      CHECK(!memcmp(nm->variants[HK_MESH_VARIANT_MAIN].data_ptr,
                     mesh->variants[HK_MESH_VARIANT_MAIN].data_ptr,
                     mesh->variants[HK_MESH_VARIANT_MAIN].data_size));
      CHECK(nf->variants[0].data_size == fs->variants[0].data_size);
      CHECK(!memcmp(nf->variants[0].data_ptr, fs->variants[0].data_ptr, fs->variants[0].data_size));
   }
}

static unsigned
compile(struct hk_device *dev, bool deferred, bool native, unsigned failure)
{
   struct vk_shader_compile_info infos[] = {
      {.stage = MESA_SHADER_MESH, .nir = source(false, native ? 3 : 96, native ? 1 : 32),
       .robustness = &vk_robustness_disabled},
      {.stage = MESA_SHADER_FRAGMENT, .nir = source(true, 0, 0),
       .robustness = &vk_robustness_disabled},
   };
   struct vk_shader *out[2] = {0};
   const struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   const struct vk_color_blend_state cb = {
      .attachment_count = 1, .color_write_enables = 1,
      .attachments = {{.write_mask = 15}},
   };
   const struct vk_graphics_pipeline_state state = {.rp = &rp, .cb = &cb};
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key(&key, &state);
   allocations = 0;
   native_compiles = emulated_compiles = fragment_compiles = 0;
   fail_allocation = failure;
   VkResult result = deferred
      ? hk_apple9_defer_graphics(dev, 2, infos, NULL, NULL, out)
      : hk_apple9_compile_graphics(dev, 2, infos, NULL, &key, NULL, NULL, out);
   if (failure) {
      CHECK(result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
      CHECK(!out[0] && !out[1]);
   } else {
      if (result != VK_SUCCESS)
         fprintf(stderr, "compile deferred=%u native=%u result=%d\n", deferred, native, result);
      CHECK(result == VK_SUCCESS);
      struct hk_api_shader *mesh = container_of(out[0], struct hk_api_shader, vk);
      struct hk_api_shader *fs = container_of(out[1], struct hk_api_shader, vk);
      if (deferred) {
         CHECK(allocations == 0);
         CHECK(mesh->apple9_native_candidate == native);
         CHECK(mesh->apple9_deferred && !mesh->apple9_native);
      } else {
         check_programs(mesh, fs, native);
         CHECK(native_compiles == native && emulated_compiles == 1);
         CHECK(fragment_compiles == 1 + native);
      }
      struct hk_api_shader *mcopy = roundtrip(dev, mesh), *fcopy = roundtrip(dev, fs);
      if (deferred) {
         CHECK(mcopy->apple9_native_candidate == native);
         CHECK(allocations == 0);
         struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {
            [MESA_SHADER_MESH] = mcopy, [MESA_SHADER_FRAGMENT] = fcopy,
         }, *resolved[HK_APPLE9_GFX_STAGES] = {0};
         CHECK(hk_apple9_resolve_graphics(dev, api, &state, resolved, true, false) ==
               VK_SUCCESS);
         check_programs(resolved[MESA_SHADER_MESH], resolved[MESA_SHADER_FRAGMENT], native);
         CHECK(native_compiles == native && emulated_compiles == !native);
         CHECK(fragment_compiles == 1);
         struct hk_api_shader *first = resolved[MESA_SHADER_MESH];
         unsigned before = allocations;
         CHECK(hk_apple9_resolve_graphics(dev, api, &state, resolved, true, false) ==
               VK_SUCCESS);
         CHECK(resolved[MESA_SHADER_MESH] == first && allocations == before);
         CHECK(fragment_compiles == 1);
         CHECK(hk_apple9_resolve_graphics(dev, api, &state, resolved, false, false) ==
               VK_SUCCESS);
         check_programs(resolved[MESA_SHADER_MESH],
                        resolved[MESA_SHADER_FRAGMENT], false);
         CHECK(native_compiles == native && emulated_compiles == 1);
         CHECK(fragment_compiles == 1 + native);
         before = allocations;
         CHECK(hk_apple9_resolve_graphics(dev, api, &state, resolved, true, false) ==
               VK_SUCCESS);
         CHECK(resolved[MESA_SHADER_MESH] == first && allocations == before);
      } else {
         check_programs(mcopy, fcopy, native);
      }
      hk_api_shader_destroy(&dev->vk, &mcopy->vk, NULL);
      hk_api_shader_destroy(&dev->vk, &fcopy->vk, NULL);
      hk_api_shader_destroy(&dev->vk, out[0], NULL);
      hk_api_shader_destroy(&dev->vk, out[1], NULL);
   }
   hk_apple9_variants_finish(dev);
   hk_apple9_bodies_finish(dev);
   test_reset_usc_heap(dev);
   CHECK(live_bos == 0);
   return allocations;
}

static unsigned
compile_native_only(struct hk_device *dev, bool reject, unsigned failure)
{
   struct vk_shader_compile_info infos[] = {
      {.stage = MESA_SHADER_MESH,
       .nir = source(false, 3, 1),
       .robustness = &vk_robustness_disabled},
      {.stage = MESA_SHADER_FRAGMENT,
       .nir = source(true, 0, 0),
       .robustness = &vk_robustness_disabled},
   };
   struct hk_apple9_graphics_key key = {0};
   struct vk_shader *out[2] = {0};
   allocations = 0;
   native_compiles = emulated_compiles = fragment_compiles = 0;
   fail_allocation = failure;
   reject_native = reject;
   VkResult result = hk_apple9_compile_graphics_mode(dev, 2, infos, NULL, &key,
                                                     NULL, NULL, out, true);
   reject_native = false;
   if (failure) {
      CHECK(result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
      CHECK(!out[0] && !out[1]);
   } else {
      CHECK(result == VK_SUCCESS);
      CHECK(native_compiles == 1 && emulated_compiles == reject);
      CHECK(fragment_compiles == 1);
      struct hk_api_shader *mesh =
         container_of(out[0], struct hk_api_shader, vk);
      struct hk_api_shader *fs = container_of(out[1], struct hk_api_shader, vk);
      check_programs(mesh, fs, !reject);
      for (unsigned i = 0; i < 2; ++i)
         hk_api_shader_destroy(&dev->vk, out[i], NULL);
   }
   hk_apple9_bodies_finish(dev);
   test_reset_usc_heap(dev);
   CHECK(live_bos == 0);
   return allocations;
}

static void
deferred_rejection(struct hk_device *dev)
{
   struct vk_shader_compile_info infos[] = {
      {.stage = MESA_SHADER_MESH, .nir = source(false, 3, 1),
       .robustness = &vk_robustness_disabled},
      {.stage = MESA_SHADER_FRAGMENT, .nir = source(true, 0, 0),
       .robustness = &vk_robustness_disabled},
   };
   struct vk_shader *out[2] = {0};
   fail_allocation = 0;
   allocations = native_compiles = emulated_compiles = fragment_compiles = 0;
   CHECK(hk_apple9_defer_graphics(dev, 2, infos, NULL, NULL, out) == VK_SUCCESS);
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {
      [MESA_SHADER_MESH] = container_of(out[0], struct hk_api_shader, vk),
      [MESA_SHADER_FRAGMENT] = container_of(out[1], struct hk_api_shader, vk),
   }, *resolved[HK_APPLE9_GFX_STAGES] = {0};
   CHECK(api[MESA_SHADER_MESH]->apple9_native_candidate);
   reject_native = true;
   CHECK(hk_apple9_resolve_graphics(dev, api, NULL, resolved, true, false) == VK_SUCCESS);
   reject_native = false;
   CHECK(!api[MESA_SHADER_MESH]->apple9_native_candidate);
   check_programs(resolved[MESA_SHADER_MESH], resolved[MESA_SHADER_FRAGMENT], false);
   struct hk_api_shader *first = resolved[MESA_SHADER_MESH];
   unsigned before = allocations;
   for (unsigned i = 0; i < 4; ++i) {
      CHECK(hk_apple9_resolve_graphics(dev, api, NULL, resolved, i & 1, false) == VK_SUCCESS);
      CHECK(resolved[MESA_SHADER_MESH] == first && allocations == before);
      CHECK(native_compiles == 1 && emulated_compiles == 1 && fragment_compiles == 1);
   }
   struct hk_api_shader *copy = roundtrip(dev, api[MESA_SHADER_MESH]);
   CHECK(!copy->apple9_native_candidate);
   hk_api_shader_destroy(&dev->vk, &copy->vk, NULL);
   hk_api_shader_destroy(&dev->vk, out[0], NULL);
   hk_api_shader_destroy(&dev->vk, out[1], NULL);
   hk_apple9_variants_finish(dev);
   hk_apple9_bodies_finish(dev);
   test_reset_usc_heap(dev);
   CHECK(live_bos == 0);
}

/* Use real VS->GS input loads so the native path must retain the software VS
 * and its output mask. A constant fragment shader needs no varying copies. */
static nir_shader *
gs_source(mesa_shader_stage stage, unsigned vertices)
{
   nir_builder b = nir_builder_init_simple_shader(stage, &agx_nir_options,
                                                  "native GS ownership");
   b.shader->info.io_lowered = true;
   if (stage == MESA_SHADER_GEOMETRY) {
      b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
      b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
      b.shader->info.gs.vertices_in = 3;
      b.shader->info.gs.vertices_out = vertices;
      b.shader->info.gs.invocations = 1;
      b.shader->info.gs.active_stream_mask = 1;
      nir_def *count = nir_load_global_constant(
         &b, 1, 32, nir_load_descriptor_set_agx(&b, .desc_set = 0),
         .align_mul = 4);
      for (unsigned v = 0; v < vertices; ++v) {
         nir_push_if(&b, nir_ult(&b, nir_imm_int(&b, v), count));
         nir_def *pos = nir_load_per_vertex_input(
            &b, 4, 32, nir_imm_int(&b, v % 3), nir_imm_int(&b, 0),
            .dest_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_store_output(&b, pos, nir_imm_int(&b, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
         nir_emit_vertex(&b, 0);
         nir_pop_if(&b, NULL);
      }
      nir_end_primitive(&b, 0);
   } else {
      nir_store_output(&b, nir_imm_vec4(&b, .2, .4, .6, 1), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = stage == MESA_SHADER_VERTEX
                            ? VARYING_SLOT_POS : FRAG_RESULT_DATA0,
                          .num_slots = 1});
   }
   b.shader->constant_data_size = 64;
   b.shader->constant_data = ralloc_size(b.shader, 64);
   memset(b.shader->constant_data, 0x30 + stage, 64);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
check_gs_programs(struct hk_api_shader **api, bool native, unsigned vertices)
{
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX, MESA_SHADER_GEOMETRY, MESA_SHADER_FRAGMENT};
   struct hk_api_shader *ng = hk_api_shader_native(api[MESA_SHADER_GEOMETRY]);
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      struct hk_api_shader *s = api[stages[i]];
      CHECK(s && s->vk.stage == stages[i] && !s->apple9_deferred);
      CHECK(s->constant_data_addr);
      struct hk_api_shader *n = hk_api_shader_native(s);
      CHECK(!!n == native);
      if (!native)
         continue;
      CHECK(n->vk.stage == s->vk.stage && n->apple9_native_leaf);
      CHECK(!n->apple9_native && n->constant_data_addr);
      CHECK(!memcmp(n->apple9_native_pair_id, ng->apple9_native_pair_id,
                    sizeof(n->apple9_native_pair_id)));
      CHECK(n->variants[0].data_size == 64);
      for (unsigned j = 0; j < 64; ++j)
         CHECK(((uint8_t *)n->variants[0].data_ptr)[j] == 0x30 + stages[i]);
   }
   if (native) {
      CHECK(hk_native_producer_main(ng)->bo);
      CHECK(hk_native_producer_object(ng)->bo);
      CHECK(hk_native_producer_main(ng)->info.apple9_native_stage == HK_APPLE9_NATIVE_MESH);
      CHECK(hk_native_producer_object(ng)->info.apple9_native_stage == HK_APPLE9_NATIVE_OBJECT);
      /* Packed GS publishes HK_APPLE9_NATIVE_GS_PACK triangles' fixed slots. */
      const struct agx_apple9_mesh_layout *l = &hk_native_producer_main(ng)->info.apple9_native.layout;
      if (hk_apple9_native_gs_pack_enabled() && (vertices == 3 || vertices == 4)) {
         CHECK(l->max_vertices == HK_APPLE9_NATIVE_GS_PACK * vertices);
         CHECK(l->max_primitives == (vertices == 3 ? 17 : 16));
      } else {
         CHECK(l->max_vertices == vertices);
         CHECK(l->max_primitives == vertices - 2);
      }
      struct hk_api_shader *nv = hk_api_shader_native(api[MESA_SHADER_VERTEX]);
      CHECK(nv->variants[HK_VS_VARIANT_SW].bo && !nv->writes_memory);
      CHECK(nv->variants[HK_VS_VARIANT_SW].b.info.outputs & VARYING_BIT_POS);
   } else {
      CHECK(api[MESA_SHADER_GEOMETRY]->variants[HK_GS_VARIANT_RAST].bo);
      CHECK(!api[MESA_SHADER_GEOMETRY]->variants[0].info.apple9_native_stage);
   }
}

/* Exercise eager ownership, deferred native/fallback/native reuse, backend
 * rejection and every BO allocation failure, including the software VS. */
static unsigned
compile_gs(struct hk_device *dev, unsigned vertices, bool deferred,
           bool reject, unsigned failure)
{
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX, MESA_SHADER_GEOMETRY, MESA_SHADER_FRAGMENT};
   struct vk_shader_compile_info infos[3] = {0};
   struct vk_shader *out[3] = {0};
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {0};
   struct hk_api_shader *copies[HK_APPLE9_GFX_STAGES] = {0};
   struct hk_api_shader *resolved[HK_APPLE9_GFX_STAGES] = {0};
   for (unsigned i = 0; i < 3; ++i) {
      infos[i].stage = stages[i];
      infos[i].nir = gs_source(stages[i], vertices);
      infos[i].robustness = &vk_robustness_disabled;
      infos[i].next_stage_mask = i == 0 ? VK_SHADER_STAGE_GEOMETRY_BIT : 0;
   }
   allocations = native_compiles = emulated_compiles = fragment_compiles = 0;
   fail_allocation = failure;
   reject_native = reject;
   VkResult result = deferred
      ? hk_apple9_defer_graphics(dev, 3, infos, NULL, NULL, out)
      : hk_apple9_compile_graphics(dev, 3, infos, NULL, NULL, NULL, NULL, out);
   unsigned compiled_allocations = allocations;
   bool native = (hk_apple9_native_gs_enabled() ||
       (hk_apple9_native_stage_policy(hk_device_physical(dev)) & HK_APPLE9_NATIVE_GS_BIT)) && vertices <= 4 && !reject;
   if (!deferred && failure) {
      CHECK(result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
      CHECK(!out[0] && !out[1] && !out[2]);
   } else {
      CHECK(result == VK_SUCCESS);
      for (unsigned i = 0; i < 3; ++i) {
         api[stages[i]] = container_of(out[i], struct hk_api_shader, vk);
         copies[stages[i]] = roundtrip(dev, api[stages[i]]);
      }
      if (deferred) {
         CHECK(allocations == 0);
         result = hk_apple9_resolve_graphics(dev, copies, NULL, resolved, true, false);
         compiled_allocations = allocations;
         if (failure) {
            CHECK(result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
         } else {
            CHECK(result == VK_SUCCESS);
            check_gs_programs(resolved, native, vertices);
            CHECK(native_compiles == ((hk_apple9_native_gs_enabled() ||
       (hk_apple9_native_stage_policy(hk_device_physical(dev)) & HK_APPLE9_NATIVE_GS_BIT)) && vertices <= 4));
            unsigned before = allocations, compiles = native_compiles;
            struct hk_api_shader *first = resolved[MESA_SHADER_GEOMETRY];
            CHECK(hk_apple9_resolve_graphics(dev, copies, NULL, resolved, true, false) == VK_SUCCESS);
            CHECK(allocations == before && resolved[MESA_SHADER_GEOMETRY] == first);
            CHECK(hk_apple9_resolve_graphics(dev, copies, NULL, resolved, false, false) == VK_SUCCESS);
            check_gs_programs(resolved, false, vertices);
            before = allocations;
            CHECK(hk_apple9_resolve_graphics(dev, copies, NULL, resolved, true, false) == VK_SUCCESS);
            CHECK(allocations == before && resolved[MESA_SHADER_GEOMETRY] == first);
            CHECK(native_compiles == compiles);
         }
      } else {
         check_gs_programs(api, native, vertices);
         check_gs_programs(copies, native, vertices);
      }
      for (unsigned i = 0; i < 3; ++i) {
         hk_api_shader_destroy(&dev->vk, out[i], NULL);
         hk_api_shader_destroy(&dev->vk, &copies[stages[i]]->vk, NULL);
      }
   }
   reject_native = false;
   hk_apple9_variants_finish(dev);
   hk_apple9_bodies_finish(dev);
   test_reset_usc_heap(dev);
   CHECK(live_bos == 0);
   return compiled_allocations;
}

static nir_shader *
tess_source(mesa_shader_stage stage, unsigned vertices)
{
   if (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_FRAGMENT)
      return gs_source(stage, 0);
   nir_builder b = nir_builder_init_simple_shader(stage, &agx_nir_options,
                                                  "native TES ownership");
   b.shader->info.io_lowered = true;
   if (stage == MESA_SHADER_TESS_CTRL) {
      b.shader->info.tess.tcs_vertices_out = vertices;
      nir_def *id = nir_load_invocation_id(&b);
      nir_def *pos = nir_load_per_vertex_input(
         &b, 4, 32, id, nir_imm_int(&b, 0), .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(
         &b, pos, id, nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_push_if(&b, nir_ieq_imm(&b, id, 0));
      nir_store_output(
         &b, nir_imm_vec4(&b, 1, 1, 1, 1), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_OUTER,
                          .num_slots = 1});
      nir_store_output(
         &b, nir_imm_vec2(&b, 1, 1), nir_imm_int(&b, 0), .write_mask = 3,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_TESS_LEVEL_INNER,
                          .num_slots = 1});
      nir_pop_if(&b, NULL);
   } else {
      b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
      b.shader->info.tess.spacing = TESS_SPACING_EQUAL;
      nir_def *coord = nir_load_tess_coord(&b);
      nir_def *pos = nir_load_per_vertex_input(
         &b, 4, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      pos = nir_fadd(
         &b, pos,
         nir_vec4(&b, nir_channel(&b, coord, 0), nir_channel(&b, coord, 1),
                  nir_imm_float(&b, 0), nir_imm_float(&b, 0)));
      nir_store_output(
         &b, pos, nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   }
   b.shader->constant_data_size = 64;
   b.shader->constant_data = ralloc_size(b.shader, 64);
   memset(b.shader->constant_data, 0x30 + stage, 64);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static void
check_tess_programs(struct hk_api_shader **api, bool native, unsigned vertices)
{
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX, MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL,
      MESA_SHADER_FRAGMENT};
   struct hk_api_shader *ng = hk_api_shader_native(api[MESA_SHADER_TESS_EVAL]);
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      struct hk_api_shader *s = api[stages[i]];
      CHECK(s && s->vk.stage == stages[i] && !s->apple9_deferred);
      CHECK(s->constant_data_addr);
      struct hk_api_shader *n = hk_api_shader_native(s);
      CHECK(!!n == native);
      if (!native)
         continue;
      CHECK(n->vk.stage == s->vk.stage && n->apple9_native_leaf);
      CHECK(!n->apple9_native && n->constant_data_addr);
      CHECK(!memcmp(n->apple9_native_pair_id, ng->apple9_native_pair_id,
                    sizeof(n->apple9_native_pair_id)));
      CHECK(n->variants[0].data_size == 64);
      for (unsigned j = 0; j < 64; ++j)
         CHECK(((uint8_t *)n->variants[0].data_ptr)[j] == 0x30 + stages[i]);
   }
   if (native) {
      CHECK(hk_native_producer_main(ng)->bo);
      CHECK(hk_native_producer_object(ng)->bo);
      CHECK(hk_native_producer_main(ng)->info.apple9_native_stage ==
            HK_APPLE9_NATIVE_MESH);
      CHECK(hk_native_producer_object(ng)->info.apple9_native_stage ==
            HK_APPLE9_NATIVE_OBJECT);
      CHECK(
         hk_native_producer_main(ng)->info.apple9_native.layout.max_vertices ==
         3);
      CHECK(hk_native_producer_main(ng)
               ->info.apple9_native.layout.max_primitives == 1);
      struct hk_api_shader *nv = hk_api_shader_native(api[MESA_SHADER_VERTEX]);
      CHECK(nv->variants[HK_VS_VARIANT_SW].bo && !nv->writes_memory);
      CHECK(nv->variants[HK_VS_VARIANT_SW].b.info.outputs & VARYING_BIT_POS);
      struct hk_api_shader *nc =
         hk_api_shader_native(api[MESA_SHADER_TESS_CTRL]);
      CHECK(nc->variants[0].bo && !nc->writes_memory);
      CHECK(nc->variants[0].info.tess.tcs_output_patch_size == vertices);
      CHECK(ng->apple9_native_tess.mode == TESS_PRIMITIVE_TRIANGLES);
      CHECK(ng->apple9_native_tess.spacing == TESS_SPACING_EQUAL);
   } else {
      CHECK(api[MESA_SHADER_TESS_EVAL]->variants[HK_VS_VARIANT_HW].bo);
      CHECK(!api[MESA_SHADER_TESS_EVAL]->variants[0].info.apple9_native_stage);
   }
}

/* Exercise eager ownership, deferred native/fallback/native reuse, backend
 * rejection and every BO allocation failure, including the software VS. */
static unsigned
compile_tess(struct hk_device *dev, unsigned vertices, bool deferred,
             bool reject, unsigned failure)
{
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX, MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL,
      MESA_SHADER_FRAGMENT};
   struct vk_shader_compile_info infos[4] = {0};
   struct vk_shader *out[4] = {0};
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {0};
   struct hk_api_shader *copies[HK_APPLE9_GFX_STAGES] = {0};
   struct hk_api_shader *resolved[HK_APPLE9_GFX_STAGES] = {0};
   for (unsigned i = 0; i < 4; ++i) {
      infos[i].stage = stages[i];
      infos[i].nir = tess_source(stages[i], vertices);
      infos[i].robustness = &vk_robustness_disabled;
      infos[i].next_stage_mask =
         i == 0   ? VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
         : i == 1 ? VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
                  : 0;
   }
   allocations = native_compiles = emulated_compiles = fragment_compiles = 0;
   fail_allocation = failure;
   reject_native = reject;
   VkResult result =
      deferred ? hk_apple9_defer_graphics(dev, 4, infos, NULL, NULL, out)
               : hk_apple9_compile_graphics(dev, 4, infos, NULL, NULL, NULL,
                                            NULL, out);
   unsigned compiled_allocations = allocations;
   bool native = (hk_apple9_native_tess_enabled() ||
       (hk_apple9_native_stage_policy(hk_device_physical(dev)) & HK_APPLE9_NATIVE_TES_BIT)) && vertices <= 32 && !reject;
   if (!deferred && failure) {
      CHECK(result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
      CHECK(!out[0] && !out[1] && !out[2] && !out[3]);
   } else {
      CHECK(result == VK_SUCCESS);
      for (unsigned i = 0; i < 4; ++i) {
         api[stages[i]] = container_of(out[i], struct hk_api_shader, vk);
         copies[stages[i]] = roundtrip(dev, api[stages[i]]);
      }
      if (deferred) {
         CHECK(allocations == 0);
         result = hk_apple9_resolve_graphics(dev, copies, NULL, resolved, true,
                                             false);
         compiled_allocations = allocations;
         if (failure) {
            CHECK(result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
         } else {
            CHECK(result == VK_SUCCESS);
            check_tess_programs(resolved, native, vertices);
            CHECK(native_compiles ==
                  ((hk_apple9_native_tess_enabled() ||
       (hk_apple9_native_stage_policy(hk_device_physical(dev)) & HK_APPLE9_NATIVE_TES_BIT)) && vertices <= 32));
            unsigned before = allocations, compiles = native_compiles;
            struct hk_api_shader *first = resolved[MESA_SHADER_TESS_EVAL];
            CHECK(hk_apple9_resolve_graphics(dev, copies, NULL, resolved, true,
                                             false) == VK_SUCCESS);
            CHECK(allocations == before &&
                  resolved[MESA_SHADER_TESS_EVAL] == first);
            CHECK(hk_apple9_resolve_graphics(dev, copies, NULL, resolved, false,
                                             false) == VK_SUCCESS);
            check_tess_programs(resolved, false, vertices);
            before = allocations;
            CHECK(hk_apple9_resolve_graphics(dev, copies, NULL, resolved, true,
                                             false) == VK_SUCCESS);
            CHECK(allocations == before &&
                  resolved[MESA_SHADER_TESS_EVAL] == first);
            CHECK(native_compiles == compiles);
         }
      } else {
         check_tess_programs(api, native, vertices);
         check_tess_programs(copies, native, vertices);
      }
      for (unsigned i = 0; i < 4; ++i) {
         hk_api_shader_destroy(&dev->vk, out[i], NULL);
         hk_api_shader_destroy(&dev->vk, &copies[stages[i]]->vk, NULL);
      }
   }
   reject_native = false;
   hk_apple9_variants_finish(dev);
   hk_apple9_bodies_finish(dev);
   test_reset_usc_heap(dev);
   CHECK(live_bos == 0);
   return compiled_allocations;
}

static unsigned
store_guard_depth(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_store_global)
               continue;
            unsigned depth = 0;
            for (nir_cf_node *node = block->cf_node.parent; node;
                 node = node->parent)
               depth += node->type == nir_cf_node_if;
            return depth;
         }
      }
   }
   CHECK(!"store must survive preprocessing");
   return 0;
}

static void
check_fragment_fixup(void)
{
   for (unsigned native = 0; native < 2; ++native) {
      nir_shader *nir = source(true, 0, 0);
      nir_builder b =
         nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
      nir_def *address =
         nir_load_ubo(&b, 1, 64, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                      .align_mul = 8, .range = 8);
      nir_store_global(&b, nir_imm_int(&b, 0x57), address, .align_mul = 4,
                       .write_mask = 1);
      nir->info.writes_memory = true;
      nir->info.fs.uses_sample_shading = true;
      agx_preprocess_nir(nir);
      unsigned before = store_guard_depth(nir);
      CHECK(before == 1);
      hk_apple9_lower_primitive_fs(nir, native);
      CHECK(store_guard_depth(nir) == before);
      nir_validate_shader(nir,
                          "hybrid fragment fix-up preserves helper guards");
      ralloc_free(nir);
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check_fragment_fixup();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = 0x1200000000ull;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&dev->apple9_variants.lock, mtx_plain);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   list_inithead(&dev->apple9_bodies.slabs);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);
   for (unsigned v = 1; v <= 32; v = v == 1 ? 3 : v == 3 ? 32 : 33) {
      compile_tess(dev, v, false, false, 0);
      compile_tess(dev, v, true, false, 0);
   }
   if (hk_apple9_native_tess_enabled()) {
      compile_tess(dev, 3, false, true, 0);
      compile_tess(dev, 3, true, true, 0);
      for (unsigned deferred = 0; deferred < 2; ++deferred) {
         unsigned count = compile_tess(dev, 3, deferred, false, 0);
         for (unsigned i = 1; i <= count; ++i)
            compile_tess(dev, 3, deferred, false, i);
      }
      goto done;
   }
   for (unsigned v = 3; v <= 5; ++v) {
      compile_gs(dev, v, false, false, 0);
      compile_gs(dev, v, true, false, 0);
   }
   if (hk_apple9_native_gs_enabled()) {
      compile_gs(dev, 4, false, true, 0);
      compile_gs(dev, 4, true, true, 0);
      for (unsigned deferred = 0; deferred < 2; ++deferred) {
         unsigned count = compile_gs(dev, 4, deferred, false, 0);
         for (unsigned i = 1; i <= count; ++i)
            compile_gs(dev, 4, deferred, false, i);
      }
      goto done;
   }
   unsigned count = compile(dev, false, true, 0);
   compile(dev, false, false, 0);
   compile(dev, true, true, 0);
   compile(dev, true, false, 0);
   deferred_rejection(dev);
   for (unsigned i = 1; i <= count; ++i)
      compile(dev, false, true, i);
   count = compile_native_only(dev, false, 0);
   for (unsigned i = 1; i <= count; ++i)
      compile_native_only(dev, false, i);
   count = compile_native_only(dev, true, 0);
   for (unsigned i = 1; i <= count; ++i)
      compile_native_only(dev, true, i);
done:
   simple_mtx_destroy(&pdev->debug_compile_lock);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&dev->apple9_variants.lock);
   test_reset_usc_heap(dev);
   hk_device_finish_usc_heap(dev);
   free(dev);
   free(pdev);
   glsl_type_singleton_decref();
   puts(
      "hybrid pairs, single-path deferred compiles/cache reuse, helper guards, constants and OOM pass");
   return 0;
}
