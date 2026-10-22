/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Compile, serialize, reload and destroy the real driver shader objects. Only
 * BO allocation is replaced with host memory; no command execution is claimed.
 */
#include "../hk_shader.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)

static unsigned allocations, live_bos, fail_allocation;

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
   return container_of(result, struct hk_api_shader, vk);
}

static void
check_programs(struct hk_api_shader *mesh, struct hk_api_shader *fs, bool native)
{
   CHECK(!mesh->apple9_deferred && !fs->apple9_deferred);
   CHECK(mesh->variants[HK_MESH_VARIANT_MAIN].info.apple9_native_stage == HK_APPLE9_NATIVE_NONE);
   CHECK(mesh->variants[HK_MESH_VARIANT_RAST].bo);
   CHECK(!mesh->variants[HK_MESH_VARIANT_RAST].b.info.apple9_body_only);
   CHECK(!!mesh->apple9_native == native && !!fs->apple9_native == native);
   CHECK(mesh->constant_data_addr && fs->constant_data_addr);
   if (native) {
      struct hk_api_shader *nm = mesh->apple9_native, *nf = fs->apple9_native;
      CHECK(!nm->apple9_native && !nf->apple9_native);
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
      }
      struct hk_api_shader *mcopy = roundtrip(dev, mesh), *fcopy = roundtrip(dev, fs);
      if (deferred) {
         CHECK(mcopy->apple9_native_candidate == native);
         CHECK(allocations == 0);
         struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {
            [MESA_SHADER_MESH] = mcopy, [MESA_SHADER_FRAGMENT] = fcopy,
         }, *resolved[HK_APPLE9_GFX_STAGES] = {0};
         CHECK(hk_apple9_resolve_graphics(dev, api, &state, resolved) == VK_SUCCESS);
         check_programs(resolved[MESA_SHADER_MESH], resolved[MESA_SHADER_FRAGMENT], native);
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
   CHECK(live_bos == 0);
   return allocations;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = 0x1200000000ull;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   simple_mtx_init(&dev->apple9_variants.lock, mtx_plain);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);
   unsigned count = compile(dev, false, true, 0);
   compile(dev, false, false, 0);
   compile(dev, true, true, 0);
   compile(dev, true, false, 0);
   for (unsigned i = 1; i <= count; ++i)
      compile(dev, false, true, i);
   simple_mtx_destroy(&pdev->debug_compile_lock);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&dev->apple9_variants.lock);
   free(dev);
   free(pdev);
   glsl_type_singleton_decref();
   puts("hybrid eager/deferred pairs, primitive FS, cache reload, constants, ownership and upload OOM pass");
   return 0;
}
