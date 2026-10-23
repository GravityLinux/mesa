/* SPDX-License-Identifier: MIT
 * Real driver compilation/serialization and runtime memory/disk cache tests.
 * Only GPU allocations use host memory; no device is accessed.
 */
#include <unistd.h>
#include "../hk_shader.c"
#include "util/disk_cache.h"
#include "vk_common_entrypoints.h"
#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)
static unsigned allocations, live_bos, fail_allocation, nir_loads;

nir_shader *__real_nir_deserialize(void *, const nir_shader_compiler_options *,
                                   struct blob_reader *);
nir_shader *__wrap_nir_deserialize(void *, const nir_shader_compiler_options *,
                                   struct blob_reader *);
nir_shader *
__wrap_nir_deserialize(void *ctx, const nir_shader_compiler_options *opts,
                       struct blob_reader *blob)
{
   ++nir_loads;
   return __real_nir_deserialize(ctx, opts, blob);
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

static void VKAPI_CALL
properties(VkPhysicalDevice handle, VkPhysicalDeviceProperties *out)
{
   VK_FROM_HANDLE(vk_physical_device, pdev, handle);
   memset(out, 0, sizeof(*out));
   memcpy(out->pipelineCacheUUID, pdev->properties.pipelineCacheUUID,
          VK_UUID_SIZE);
}

static struct vk_pipeline_cache *
cache_create(struct hk_device *dev, const void *data, size_t size, bool disk)
{
   const VkPipelineCacheCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .initialDataSize = size,
      .pInitialData = data,
   };
   const struct vk_pipeline_cache_create_info create = {
      .pCreateInfo = &info,
      .weak_ref = !size,
      .skip_disk_cache = !disk,
      .force_enable = true,
   };
   struct vk_pipeline_cache *cache =
      vk_pipeline_cache_create(&dev->vk, &create, NULL);
   CHECK(cache);
   return cache;
}

static nir_shader *
source(unsigned stage, bool cull)
{
   nir_builder b =
      nir_builder_init_simple_shader(stage, &agx_nir_options, "variant cache");
   b.shader->info.io_lowered = true;
   nir_store_output(&b, nir_imm_vec4(&b, .25, .5, .75, 1), nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics = {.location = stage == MESA_SHADER_VERTEX
                                                    ? VARYING_SLOT_POS
                                                    : FRAG_RESULT_DATA0,
                                     .num_slots = 1});
   if (cull) {
      nir_store_output(
         &b, nir_imm_float(&b, 1), nir_imm_int(&b, 0), .write_mask = 1,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0, .num_slots = 1});
      b.shader->info.cull_distance_array_size = 1;
   }
   b.shader->constant_data_size = 64;
   b.shader->constant_data = ralloc_size(b.shader, 64);
   memset(b.shader->constant_data, 0x57 + stage, 64);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
mesh_source(bool fragment, unsigned vertices, unsigned primitives)
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

static void
api_create(struct hk_device *dev, struct hk_api_shader **api, bool cull)
{
   struct vk_shader_compile_info infos[] = {
      {.stage = MESA_SHADER_VERTEX,
       .nir = source(MESA_SHADER_VERTEX, cull),
       .robustness = &vk_robustness_disabled,
       .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT},
      {.stage = MESA_SHADER_FRAGMENT,
       .nir = cull ? NULL : source(MESA_SHADER_FRAGMENT, false),
       .robustness = &vk_robustness_disabled},
   };
   struct vk_shader *out[2] = {0};
   CHECK(hk_apple9_defer_graphics(dev, cull ? 1 : 2, infos, NULL, NULL, out) ==
         VK_SUCCESS);
   memset(api, 0, HK_APPLE9_GFX_STAGES * sizeof(*api));
   api[MESA_SHADER_VERTEX] = container_of(out[0], struct hk_api_shader, vk);
   if (!cull)
      api[MESA_SHADER_FRAGMENT] =
         container_of(out[1], struct hk_api_shader, vk);
}

static void
api_destroy(struct hk_device *dev, struct hk_api_shader **api)
{
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
      if (api[s])
         hk_api_shader_destroy(&dev->vk, &api[s]->vk, NULL);
}

static void
key_tests(struct hk_device *dev, struct hk_api_shader **api)
{
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key(&key, NULL);
   bool prefer_native = false;
   blake3_hash original, changed;
   CHECK(hk_apple9_variant_cache_key(dev, api, &key, prefer_native, original));
#define DIFFERENT(change, restore)                                             \
   do {                                                                        \
      change;                                                                  \
      CHECK(hk_apple9_variant_cache_key(dev, api, &key, prefer_native, changed));             \
      CHECK(memcmp(original, changed, sizeof(original)));                      \
      restore;                                                                 \
   } while (0)
   DIFFERENT(prefer_native = true, prefer_native = false);
   DIFFERENT(dev->vk.physical->properties.pipelineCacheUUID[0] ^= 1,
             dev->vk.physical->properties.pipelineCacheUUID[0] ^= 1);
   DIFFERENT(dev->vk.physical->properties.shaderBinaryVersion++,
             dev->vk.physical->properties.shaderBinaryVersion--);
   DIFFERENT(dev->dev.chip++, dev->dev.chip--);
   DIFFERENT(hk_device_physical(dev)->apple9_d3d_mip_rounding = true,
             hk_device_physical(dev)->apple9_d3d_mip_rounding = false);
   DIFFERENT(dev->vk.enabled_features.minLod = true,
             dev->vk.enabled_features.minLod = false);
   DIFFERENT(dev->vk.enabled_features.customBorderColors = true,
             dev->vk.enabled_features.customBorderColors = false);
   DIFFERENT(key.zs_self_dep = !key.zs_self_dep,
             key.zs_self_dep = !key.zs_self_dep);
   struct hk_apple9_deferred *d = api[MESA_SHADER_VERTEX]->apple9_deferred;
   DIFFERENT(((uint8_t *)d->nir)[d->nir_size - 1] ^= 1,
             ((uint8_t *)d->nir)[d->nir_size - 1] ^= 1);
   DIFFERENT(d->next_stage_mask ^= 1, d->next_stage_mask ^= 1);
   DIFFERENT(d->set_layout_count++, d->set_layout_count--);
   DIFFERENT(d->sample_shading = true, d->sample_shading = false);
   DIFFERENT(d->zs_self_dep = true, d->zs_self_dep = false);
   DIFFERENT(d->undefined_colors ^= 1, d->undefined_colors ^= 1);
   DIFFERENT(d->cull_distance = true, d->cull_distance = false);
   struct hk_api_shader *saved_fs = api[MESA_SHADER_FRAGMENT];
   DIFFERENT(api[MESA_SHADER_FRAGMENT] = NULL,
             api[MESA_SHADER_FRAGMENT] = saved_fs);
#undef DIFFERENT
   CHECK(hk_apple9_variant_cache_key(dev, api, &key, prefer_native, changed));
   CHECK(!memcmp(original, changed, sizeof(original)));
   api[MESA_SHADER_VERTEX]->apple9_deferred = NULL;
   CHECK(!hk_apple9_variant_cache_key(dev, api, &key, prefer_native, changed));
   api[MESA_SHADER_VERTEX]->apple9_deferred = d;
}

static void
expect_miss(struct hk_device *dev, const void *data, size_t size, unsigned mask)
{
   struct vk_shader *out[HK_APPLE9_GFX_STAGES];
   unsigned stages[HK_APPLE9_GFX_STAGES];
   unsigned before = live_bos;
   CHECK(!hk_apple9_variant_cache_load(dev, data, size, mask, stages, out));
   CHECK(live_bos == before);
   for (unsigned s = 0; s < HK_APPLE9_GFX_STAGES; ++s)
      CHECK(!out[s]);
}

static void
blob_tests(struct hk_device *dev, struct hk_api_shader **resolved,
           unsigned first_stage)
{
   unsigned stages[] = {first_stage, MESA_SHADER_FRAGMENT};
   struct vk_shader *shaders[] = {&resolved[stages[0]]->vk,
                                  &resolved[stages[1]]->vk};
   struct blob blob;
   blob_init(&blob);
   hk_apple9_variant_cache_store(dev, stages, 2, shaders, &blob);
   CHECK(!blob.out_of_memory);
   const unsigned mask = BITFIELD_BIT(stages[0]) | BITFIELD_BIT(stages[1]);
   unsigned loaded_stages[HK_APPLE9_GFX_STAGES];
   struct vk_shader *loaded[HK_APPLE9_GFX_STAGES];
   CHECK(hk_apple9_variant_cache_load(dev, blob.data, blob.size, mask,
                                      loaded_stages, loaded) == 2);
   struct blob copy;
   blob_init(&copy);
   hk_apple9_variant_cache_store(dev, loaded_stages, 2, loaded, &copy);
   CHECK(copy.size == blob.size && !memcmp(copy.data, blob.data, blob.size));
   for (unsigned i = 0; i < 2; ++i)
      hk_api_shader_destroy(&dev->vk, loaded[i], NULL);
   blob_finish(&copy);
   /* Every truncation, including failure after a complete first shader. */
   for (unsigned i = 0; i < blob.size; ++i)
      expect_miss(dev, blob.data, i, mask);
   expect_miss(dev, blob.data, blob.size, BITFIELD_BIT(MESA_SHADER_VERTEX));
   uint8_t *bad = malloc(blob.size + 1);
   CHECK(bad);
   memcpy(bad, blob.data, blob.size);
   bad[blob.size] = 0;
   expect_miss(dev, bad, blob.size + 1, mask);
   /* Header, UUID, binary version, count, stage, stage length, nested stage,
    * and deferred flag must all agree before a binary can be installed. */
   const unsigned corrupt[] = {0, 4, 20, 24, 28, 32, 36, 40};
   for (unsigned i = 0; i < ARRAY_SIZE(corrupt); ++i) {
      memcpy(bad, blob.data, blob.size);
      bad[corrupt[i]] ^= 0xff;
      expect_miss(dev, bad, blob.size, mask);
   }
   /* Duplicate the first stage in the second slot. */
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   blob_read_bytes(&reader, 32);
   uint32_t first_len = blob_read_uint32(&reader);
   blob_read_bytes(&reader, first_len);
   size_t second = ALIGN_POT(reader.current - reader.data, 4);
   memcpy(bad, blob.data, blob.size);
   memcpy(bad + second, &stages[0], sizeof(uint32_t));
   expect_miss(dev, bad, blob.size, mask);
   free(bad);
   blob_finish(&blob);
}


/* The common deserializer must reject impossible payload sizes before
 * allocating host or GPU program storage, independently of the outer cache. */
static void
payload_bounds_tests(struct hk_device *dev)
{
   const uint32_t lengths[][3] = {
      {UINT32_MAX, 0, 0}, {0, UINT32_MAX, 0},
      {4, 0, 3}, {0, 4, 3}, {2, 2, 3},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(lengths); ++i) {
      struct blob blob;
      blob_init(&blob);
      const struct hk_shader_info info = {.stage = MESA_SHADER_VERTEX};
      const struct agx_shader_info backend = {0};
      blob_write_bytes(&blob, &info, sizeof(info));
      blob_write_bytes(&blob, &backend, sizeof(backend));
      blob_write_uint32(&blob, lengths[i][0]);
      blob_write_uint32(&blob, lengths[i][1]);
      const uint32_t data = 0;
      blob_write_bytes(&blob, &data, lengths[i][2]);
      struct blob_reader reader;
      blob_reader_init(&reader, blob.data, blob.size);
      struct hk_shader shader = {0};
      unsigned before = allocations;
      CHECK(hk_deserialize_shader(dev, &reader, &shader) ==
            VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      CHECK(!shader.code_ptr && !shader.data_ptr && !shader.linked.ht);
      CHECK(allocations == before);
      blob_finish(&blob);
   }
}

static void
mesh_api_create(struct hk_device *dev, struct hk_api_shader **api)
{
   struct vk_shader_compile_info infos[] = {
      {.stage = MESA_SHADER_MESH, .nir = mesh_source(false, 3, 1),
       .robustness = &vk_robustness_disabled},
      {.stage = MESA_SHADER_FRAGMENT, .nir = mesh_source(true, 0, 0),
       .robustness = &vk_robustness_disabled},
   };
   struct vk_shader *out[2] = {0};
   CHECK(hk_apple9_defer_graphics(dev, 2, infos, NULL, NULL, out) == VK_SUCCESS);
   memset(api, 0, HK_APPLE9_GFX_STAGES * sizeof(*api));
   api[MESA_SHADER_MESH] = container_of(out[0], struct hk_api_shader, vk);
   api[MESA_SHADER_FRAGMENT] = container_of(out[1], struct hk_api_shader, vk);
   CHECK(api[MESA_SHADER_MESH]->apple9_native_candidate);
}

static void
native_cache_tests(struct hk_device *dev)
{
   dev->vk.mem_cache = cache_create(dev, NULL, 0, false);
   /* Hold sources so their raw variant objects survive this weak cache. */
   struct hk_api_shader *api[5][HK_APPLE9_GFX_STAGES];
   struct hk_api_shader *out[HK_APPLE9_GFX_STAGES];
   for (unsigned i = 0; i < ARRAY_SIZE(api); ++i) {
      mesh_api_create(dev, api[i]);
      bool native = i < 2;
      unsigned before = nir_loads;
      if (i == 4)
         api[i][MESA_SHADER_MESH]->apple9_native_candidate = false;
      CHECK(hk_apple9_resolve_graphics(dev, api[i], NULL, out,
                                      native || i == 4) == VK_SUCCESS);
      CHECK(nir_loads == before + ((i == 0 || i == 2) ? 2 : 0));
      CHECK(out[MESA_SHADER_MESH]->apple9_native_leaf == native);
      CHECK(out[MESA_SHADER_FRAGMENT]->apple9_native_leaf == native);
      CHECK(out[MESA_SHADER_MESH]->constant_data_addr);
      CHECK(out[MESA_SHADER_FRAGMENT]->constant_data_addr);
      if (native) {
         CHECK(!memcmp(out[MESA_SHADER_MESH]->apple9_native_pair_id,
                        out[MESA_SHADER_FRAGMENT]->apple9_native_pair_id,
                        sizeof(blake3_hash)));
      }
      if (i == 0 || i == 2)
         blob_tests(dev, out, MESA_SHADER_MESH);
   }
   CHECK(dev->vk.mem_cache->object_cache->entries == 2);
   /* A valid deferred container is not a compiled draw variant. */
   unsigned stages[] = {MESA_SHADER_MESH, MESA_SHADER_FRAGMENT};
   struct vk_shader *shaders[] = {&api[0][stages[0]]->vk,
                                  &api[0][stages[1]]->vk};
   struct blob blob;
   blob_init(&blob);
   hk_apple9_variant_cache_store(dev, stages, 2, shaders, &blob);
   expect_miss(dev, blob.data, blob.size,
               BITFIELD_BIT(stages[0]) | BITFIELD_BIT(stages[1]));
   blob_finish(&blob);
   for (unsigned i = 0; i < ARRAY_SIZE(api); ++i)
      api_destroy(dev, api[i]);
   hk_apple9_variants_finish(dev);
   hk_apple9_bodies_finish(dev);
   CHECK(live_bos == 0 && dev->vk.mem_cache->object_cache->entries == 0);
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   puts("native cache: separate native/emulated hits, collapsed candidate, pair IDs, constants and malformed hybrid containers pass");
}

static void
persistent_process(struct hk_device *dev, struct hk_api_shader **api,
                   const char *mode)
{
   struct hk_api_shader *out[HK_APPLE9_GFX_STAGES];
   struct hk_physical_device *pdev = hk_device_physical(dev);
   pdev->vk.disk_cache =
      disk_cache_create("hk-variant-process", "test-build", 0);
   CHECK(pdev->vk.disk_cache);
   dev->vk.mem_cache = cache_create(dev, NULL, 0, true);
   bool hit = !strcmp(mode, "--warm-process");
   unsigned before = nir_loads;
   CHECK(hk_apple9_resolve_graphics(dev, api, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads == before + (hit ? 0 : 2));
   api_destroy(dev, api);
   hk_apple9_variants_finish(dev);
   hk_apple9_bodies_finish(dev);
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   disk_cache_destroy(pdev->vk.disk_cache);
   CHECK(live_bos == 0);
   printf("separate process: %s\n", mode);
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--codegen-knobs")) {
      for (unsigned i = 0; i < ARRAY_SIZE(hk_codegen_env); ++i)
         puts(hk_codegen_env[i]);
      return 0;
   }
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   pdev->vk.base.type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
   dev->dev.chip = pdev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = 0x1200000000ull;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.shader_ops = &hk_device_shader_ops;
   pdev->vk.dispatch_table.GetPhysicalDeviceProperties = properties;
   memset(pdev->vk.properties.pipelineCacheUUID, 0x57, VK_UUID_SIZE);
   simple_mtx_init(&dev->apple9_variants.lock, mtx_plain);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES];
   struct hk_api_shader *again[HK_APPLE9_GFX_STAGES];
   struct hk_api_shader *out[HK_APPLE9_GFX_STAGES];
   payload_bounds_tests(dev);
   if (argc == 2 && !strcmp(argv[1], "--native-cache")) {
      native_cache_tests(dev);
      goto finish;
   }
   api_create(dev, api, false);
   if (argc == 2 && !strcmp(argv[1], "--key")) {
      struct hk_apple9_graphics_key key;
      hk_apple9_graphics_key(&key, NULL);
      blake3_hash hash;
      CHECK(hk_apple9_variant_cache_key(dev, api, &key, false, hash));
      for (unsigned i = 0; i < sizeof(hash); ++i)
         printf("%02x", hash[i]);
      puts("");
      api_destroy(dev, api);
      goto finish;
   }
   if (argc == 2 && (!strcmp(argv[1], "--cold-process") ||
                     !strcmp(argv[1], "--warm-process") ||
                     !strcmp(argv[1], "--verify-process"))) {
      persistent_process(dev, api, argv[1]);
      goto finish;
   }
   key_tests(dev, api);
   CHECK(getenv("MESA_SHADER_CACHE_DIR"));
   pdev->vk.disk_cache = disk_cache_create("hk-variant-test", "test-build", 0);
   CHECK(pdev->vk.disk_cache);
   dev->vk.mem_cache = cache_create(dev, NULL, 0, true);
   unsigned before = nir_loads;
   CHECK(hk_apple9_resolve_graphics(dev, api, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads == before + 2);
   blob_tests(dev, out, MESA_SHADER_VERTEX);
   /* These are the exact driver callbacks used by application VkPipelineCache
    * objects and shader binaries. Export after the draw-time variant exists. */
   struct blob api_blobs[2];
   const unsigned api_stages[] = {MESA_SHADER_VERTEX, MESA_SHADER_FRAGMENT};
   for (unsigned i = 0; i < 2; ++i) {
      blob_init(&api_blobs[i]);
      CHECK(hk_api_shader_serialize(&dev->vk, &api[api_stages[i]]->vk,
                                    &api_blobs[i]));
   }
   api_create(dev, again, false);
   before = nir_loads;
   CHECK(hk_apple9_resolve_graphics(dev, again, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads ==
         before); /* A different API object, same persistent key. */
   CHECK(dev->vk.mem_cache->object_cache->entries == 1);
   size_t size = 0;
   CHECK(vk_common_GetPipelineCacheData(
            vk_device_to_handle(&dev->vk),
            vk_pipeline_cache_to_handle(dev->vk.mem_cache), &size,
            NULL) == VK_SUCCESS);
   void *data = malloc(size);
   CHECK(data);
   CHECK(vk_common_GetPipelineCacheData(
            vk_device_to_handle(&dev->vk),
            vk_pipeline_cache_to_handle(dev->vk.mem_cache), &size,
            data) == VK_SUCCESS);
   api_destroy(dev, again);
   api_destroy(dev, api);
   hk_apple9_bodies_finish(dev);
   CHECK(live_bos == 0 && dev->vk.mem_cache->object_cache->entries == 0);
   hk_apple9_variants_finish(dev);
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   /* Export/import into a fresh runtime cache, with disk access disabled. */
   dev->vk.mem_cache = cache_create(dev, data, size, false);
   free(data);
   api_create(dev, api, false);
   before = nir_loads;
   CHECK(hk_apple9_resolve_graphics(dev, api, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads == before);
   api_destroy(dev, api);
   hk_apple9_variants_finish(dev);
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   /* Finish all asynchronous writes and reopen the actual disk cache. */
   disk_cache_destroy(pdev->vk.disk_cache);
   pdev->vk.disk_cache = disk_cache_create("hk-variant-test", "test-build", 0);
   dev->vk.mem_cache = cache_create(dev, NULL, 0, true);
   api_create(dev, api, false);
   before = nir_loads;
   CHECK(hk_apple9_resolve_graphics(dev, api, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads == before);
   /* Partial reload OOM falls back to compilation and keeps ownership sound. */
   api_create(dev, again, false);
   hk_apple9_bodies_finish(dev);
   fail_allocation = allocations + 2;
   CHECK(hk_apple9_resolve_graphics(dev, again, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads == before + 2);
   fail_allocation = 0;
   api_destroy(dev, again);
   api_destroy(dev, api);
   hk_apple9_variants_finish(dev);
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   dev->vk.mem_cache = cache_create(dev, NULL, 0, false);
   memset(api, 0, sizeof(api));
   /* Reload only the deferred API binaries, with an empty device cache and
    * disk reads disabled: they must carry the variants compiled later. */
   for (unsigned i = 0; i < 2; ++i) {
      struct blob_reader reader;
      blob_reader_init(&reader, api_blobs[i].data, api_blobs[i].size);
      struct vk_shader *shader = NULL;
      CHECK(hk_deserialize_api_shader(&dev->vk, &reader,
                                      pdev->vk.properties.shaderBinaryVersion,
                                      NULL, &shader) == VK_SUCCESS);
      CHECK(!reader.overrun && reader.current == reader.end);
      api[api_stages[i]] = container_of(shader, struct hk_api_shader, vk);
      blob_finish(&api_blobs[i]);
   }
   before = nir_loads;
   CHECK(hk_apple9_resolve_graphics(dev, api, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads == before);
   api_destroy(dev, api);
   CHECK(dev->vk.mem_cache->object_cache->entries == 0);
   /* A vertex-only cull-distance variant includes its generated fragment. */
   api_create(dev, api, true);
   before = nir_loads;
   CHECK(hk_apple9_resolve_graphics(dev, api, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads == before + 1 && out[MESA_SHADER_FRAGMENT]);
   api_create(dev, again, true);
   before = nir_loads;
   CHECK(hk_apple9_resolve_graphics(dev, again, NULL, out, false) == VK_SUCCESS);
   CHECK(nir_loads == before && out[MESA_SHADER_FRAGMENT]);
   api_destroy(dev, again);
   api_destroy(dev, api);
   hk_apple9_variants_finish(dev);
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   disk_cache_destroy(pdev->vk.disk_cache);
   hk_apple9_bodies_finish(dev);
   CHECK(live_bos == 0);
   puts(
      "variant cache: binary/constant round trip, pipeline export/import, deferred API binary persistence, disk reopen, malformed blobs, OOM, keys and generated fragment pass");
finish:
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&pdev->debug_compile_lock);
   simple_mtx_destroy(&dev->apple9_variants.lock);
   free(pdev);
   free(dev);
   glsl_type_singleton_decref();
   return 0;
}
