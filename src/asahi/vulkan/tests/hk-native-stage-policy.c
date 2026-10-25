/* SPDX-License-Identifier: MIT
 * Real shader ownership/cache and VDM recording; host BOs, no GPU execution.
 */
int hybrid_reference_main(void);
#define main hybrid_reference_main
#include "hk-native-mesh-hybrid.c"
#undef main
#include "../hk_cmd_apple9.c"
#include "hk_entrypoints.h"
#include "vk_common_entrypoints.h"
#include <sys/stat.h>
#include <time.h>

static uint64_t
cpu_time_ns(void)
{
   struct timespec t;
   CHECK(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t) == 0);
   return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

void __wrap_agx_bo_cache_evict_all(struct agx_device *dev);
void __wrap_agx_bo_cache_evict_all(UNUSED struct agx_device *dev) {}

static FILE *ledger;
static const char *directory;

static void
save_bytes(const char *name, const void *data, size_t size)
{
   char path[4096];
   CHECK(snprintf(path, sizeof(path), "%s/%s", directory, name) < sizeof(path));
   FILE *f = fopen(path, "wb"); CHECK(f);
   CHECK(fwrite(data, 1, size, f) == size && fclose(f) == 0);
}

static void
check_rejected_blob(struct hk_device *dev, const struct blob *blob)
{
   struct blob_reader reader;
   blob_reader_init(&reader, blob->data, blob->size);
   struct vk_shader *bad = NULL;
   CHECK(hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &bad) ==
         VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT && !bad);
}

static void
save_object(struct hk_device *dev, const char *name, unsigned i,
            struct hk_api_shader *api, bool expected)
{
   struct blob original, reloaded;
   blob_init(&original); blob_init(&reloaded);
   CHECK(hk_api_shader_serialize(&dev->vk, &api->vk, &original));
   struct hk_api_shader *copy = roundtrip(dev, api);
   CHECK(hk_api_shader_serialize(&dev->vk, &copy->vk, &reloaded));
   CHECK(original.size == reloaded.size &&
         !memcmp(original.data, reloaded.data, original.size));
   char file[256]; snprintf(file, sizeof(file), "%s-%u.bin", name, i);
   save_bytes(file, original.data, original.size);
   fprintf(ledger, "%s\t%u\t%u\t%u\t%zu\n", name, i, api->vk.stage,
           hk_api_shader_native(api) != NULL, original.size);
   CHECK(!!hk_api_shader_native(api) == expected);
   if (expected) {
      /* Version/stage/candidate/cull/leaf precede the stage provenance byte. */
      uint8_t *mask = (uint8_t *)original.data + 8;
      CHECK(*mask == api->apple9_native_stage_mask);
      const uint8_t saved_mask = *mask;
      const uint8_t invalid_masks[] = {0, 3, 4, 255,
         saved_mask == HK_APPLE9_NATIVE_GS_BIT ? HK_APPLE9_NATIVE_TES_BIT
                                               : HK_APPLE9_NATIVE_GS_BIT};
      for (unsigned j = 0; j < ARRAY_SIZE(invalid_masks); ++j) {
         *mask = invalid_masks[j];
         check_rejected_blob(dev, &original);
      }
      *mask = saved_mask;
      bool was_killed = hk_device_physical(dev)->disable_native_stages;
      hk_device_physical(dev)->disable_native_stages = true;
      struct blob_reader reader;
      blob_reader_init(&reader, original.data, original.size);
      struct vk_shader *bad = NULL;
      CHECK(hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &bad) ==
            VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT && !bad);
      hk_device_physical(dev)->disable_native_stages = was_killed;
   }
   hk_api_shader_destroy(&dev->vk, &copy->vk, NULL);
   blob_finish(&original); blob_finish(&reloaded);
   /* Every fallback variant, including complete shader info and data. This
    * is a separate object stream, not a masked digest of the native wrapper. */
   struct blob fallback;
   blob_init(&fallback);
   hk_foreach_variant(api, shader)
      hk_shader_serialize(&dev->vk, shader, &fallback);
   snprintf(file, sizeof(file), "%s-%u.fallback", name, i);
   save_bytes(file, fallback.data, fallback.size);
   blob_finish(&fallback);
   /* Pair IDs include the full policy key. Compare the complete native
    * program/state separately when selecting just one stage instead of both. */
   struct hk_api_shader *native = hk_api_shader_native(api);
   if (native) {
      struct blob programs;
      blob_init(&programs);
      hk_foreach_variant(native, shader)
         hk_shader_serialize(&dev->vk, shader, &programs);
      snprintf(file, sizeof(file), "%s-%u.native", name, i);
      save_bytes(file, programs.data, programs.size);
      blob_finish(&programs);
   }
}

static bool
route(struct hk_cmd_buffer *cmd, mesa_shader_stage producer, struct agx_draw draw)
{
   return producer == MESA_SHADER_GEOMETRY ? hk_apple9_use_native_gs(cmd, draw)
                                          : hk_apple9_use_native_tess(cmd, draw);
}

static VkCommandPool
record(struct hk_device *dev, const char *name, mesa_shader_stage producer,
       struct hk_api_shader **api, unsigned vertices, bool expected)
{
   VkDevice device = hk_device_to_handle(dev);
   const VkCommandPoolCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool;
   CHECK(hk_CreateCommandPool(device, &ci, NULL, &pool) == VK_SUCCESS);
   VkCommandBuffer cb;
   const VkCommandBufferAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool=pool, .commandBufferCount=1};
   CHECK(vk_common_AllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS);
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(cb);
   memcpy(cmd->state.gfx.shaders, api, sizeof(cmd->state.gfx.shaders));
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   cmd->state.gfx.render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   dyn->ia.primitive_topology = producer == MESA_SHADER_GEOMETRY
      ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
   dyn->ts.patch_control_points = vertices;
   struct agx_draw draw = {.b=agx_3d(producer == MESA_SHADER_GEOMETRY ? 3 : vertices, 1, 1)};
   CHECK(route(cmd, producer, draw) == expected);
   if (expected) {
#define FALLBACK(field, value) do { \
   __typeof__(field) saved = field; field = value; \
   CHECK(!route(cmd, producer, draw)); field = saved; \
   CHECK(route(cmd, producer, draw)); } while (0)
      if (producer != MESA_SHADER_GEOMETRY)
         FALLBACK(draw.indexed, true);
      if (producer == MESA_SHADER_GEOMETRY) {
         /* Native GS admits every triangle and instance of a direct list. */
         struct agx_draw wide = draw;
         wide.b.count[0] *= 2;
         wide.b.count[1] = 2;
         CHECK(route(cmd, producer, wide));
         FALLBACK(draw.b.count[0], draw.b.count[0] + 1);
      } else {
         FALLBACK(draw.b.count[0], draw.b.count[0] * 2);
         FALLBACK(draw.b.count[1], 2);
      }
      if (producer != MESA_SHADER_GEOMETRY)
         FALLBACK(draw.start, 1);
      FALLBACK(cmd->state.gfx.render.view_mask, 1);
      FALLBACK(cmd->state.gfx.render.layer_count, 2);
      FALLBACK(dyn->ms.rasterization_samples, 4);
      FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
      FALLBACK(dyn->rs.rasterizer_discard_enable, true);
      FALLBACK(dyn->rs.conservative_mode, VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
      FALLBACK(cmd->state.gfx.occlusion.active_count, 1);
      FALLBACK(cmd->state.gfx.xfb_enabled, true);
      FALLBACK(cmd->state.gfx.descriptors.root.draw.pipeline_stats_flags, 1);
      FALLBACK(cmd->state.cond_render.active, true);
      FALLBACK(dev->dev.native_mesh_qualified, false);
      FALLBACK(hk_device_physical(dev)->dev.native_mesh_qualified, false);
      CHECK(!route(cmd, producer, agx_draw_indirect(0x1000)));
      struct hk_api_shader *n = hk_api_shader_native(api[producer]);
      FALLBACK(n->apple9_native_stage_mask, 0);
      FALLBACK(hk_api_shader_native(api[MESA_SHADER_VERTEX])->writes_memory, true);
      struct agx_apple9_mesh_body_info *body = &hk_native_producer_main(n)->info.apple9_native;
      FALLBACK(body->key.physical_threads, 64);
      FALLBACK(body->layout.max_vertices, 16);
      /* No GS/TES class, packed (P16/P17) or not, publishes 15. */
      FALLBACK(body->layout.max_primitives, 15);
#undef FALLBACK
   }
   uint32_t words[256] = {0};
   struct hk_cs cs = {.cmd=cmd, .type=HK_CS_VDM, .start=(void *)words, .current=(void *)words};
   struct hk_api_shader *native = hk_api_shader_native(api[producer]);
   struct hk_shader *program = expected ? hk_native_producer_main(native)
      : &api[producer]->variants[producer == MESA_SHADER_GEOMETRY ? HK_GS_VARIANT_RAST : HK_VS_VARIANT_HW];
   const struct agx_apple9_render_stage stage = {.binary=program->code_ptr,.binary_size=program->code_size};
   const struct agx_apple9_render_pipeline pipeline = {.vertex=&stage,.fragment=&stage,
      .ppp=0x4000,.primitive=AGX_PRIMITIVE_TRIANGLES,.ppp_viewport_offset=0x100,.ppp_viewport_words=10};
   const struct agx_apple9_uniform_draw uniform = {0};
   uint64_t start = cpu_time_ns();
   CHECK(hk_apple9_record_draw(cmd, &cs, &pipeline, draw, expected ? 0x8000 : 0,
                              native, program, &uniform));
   uint64_t ns = cpu_time_ns() - start;
   CHECK(cs.apple9_native_mesh == expected && cmd->vk.record_result == VK_SUCCESS);
   char file[256]; snprintf(file,sizeof(file),"%s.packet",name);
   /* Include the terminal word, written without advancing the cursor. */
   save_bytes(file, words, cs.current - cs.start + 4);
   if (expected) {
      CHECK(words[0] == 0x40000d2c && words[39] == 0xc0000000);
      CHECK(program->apple9_body && program->apple9_body->entry_pins == 1);
   } else {
      uint8_t ordinary[1024] = {0};
      uint8_t *end = agx_apple9_emit_direct_draw(ordinary, &pipeline, draw.b.count[0], 1, 0);
      CHECK(cs.current-cs.start == end-ordinary-4 && !memcmp(words,ordinary,end-ordinary));
   }
   printf("RECORD %s native=%u cpu_ns=%"PRIu64" bytes=%zu\n",name,expected,ns,(size_t)(cs.current-cs.start+4));
   return pool;
}

static void
pipeline(struct hk_device *dev, mesa_shader_stage producer, unsigned vertices,
         unsigned negative)
{
   mesa_shader_stage stages[4] = {MESA_SHADER_VERTEX};
   unsigned count = 1;
   if (producer == MESA_SHADER_TESS_EVAL) stages[count++] = MESA_SHADER_TESS_CTRL;
   stages[count++] = producer; stages[count++] = MESA_SHADER_FRAGMENT;
   struct vk_shader_compile_info infos[4] = {0}; struct vk_shader *out[4] = {0};
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES] = {0};
   for (unsigned i=0;i<count;i++) {
      infos[i] = (struct vk_shader_compile_info){.stage=stages[i],
         .nir=producer==MESA_SHADER_GEOMETRY ? gs_source(stages[i],vertices) : tess_source(stages[i],vertices),
         .robustness=&vk_robustness_disabled,
         .next_stage_mask=i+1<count ? mesa_to_vk_shader_stage(stages[i+1]) : 0};
      if ((negative==1 && stages[i]==MESA_SHADER_VERTEX) ||
          (negative==2 && stages[i]==MESA_SHADER_TESS_CTRL)) {
         nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(infos[i].nir)));
         nir_store_global(&b,nir_imm_int(&b,1),nir_imm_int64(&b,0x1000), .align_mul=4, .write_mask=1);
         nir_shader_gather_info(infos[i].nir,nir_shader_get_entrypoint(infos[i].nir));
      }
   }
   for (unsigned i = 0; i < count; ++i) {
      nir_shader *nir = infos[i].nir;
      if (stages[i] == MESA_SHADER_GEOMETRY) {
         if (negative >= 3 && negative <= 8) {
            static const unsigned slots[] = {VARYING_SLOT_CLIP_DIST0,
               VARYING_SLOT_CLIP_DIST0, VARYING_SLOT_PSIZ, VARYING_SLOT_LAYER,
               VARYING_SLOT_VIEWPORT, VARYING_SLOT_PRIMITIVE_ID};
            nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
            nir_store_output(&b, nir_imm_int(&b, 1), nir_imm_int(&b, 0),
               .write_mask=1, .src_type=nir_type_uint32,
               .io_semantics={.location=slots[negative-3], .num_slots=1});
            if (negative == 3) nir->info.clip_distance_array_size = 1;
            if (negative == 4) nir->info.cull_distance_array_size = 1;
         }
         if (negative == 9) nir->info.gs.invocations = 33; /* above the 32 runs per primitive */
         if (negative == 10) {
            nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
            nir_emit_vertex(&b, .stream_id=1);
            nir_end_primitive(&b, .stream_id=1);
         }
      }
      if (stages[i] == MESA_SHADER_TESS_EVAL) {
         if (negative == 11) nir->info.tess.point_mode = true;
         if (negative == 12) nir->info.tess._primitive_mode = TESS_PRIMITIVE_ISOLINES;
         if (negative == 13) nir->info.tess._primitive_mode = TESS_PRIMITIVE_QUADS;
         if (negative == 14) {
            nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
            nir_def *value = nir_load_input(&b, 4, 32, nir_imm_int(&b, 0),
               .dest_type=nir_type_float32,
               .io_semantics={.location=VARYING_SLOT_PATCH0, .num_slots=1});
            nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask=15,
               .src_type=nir_type_float32,
               .io_semantics={.location=VARYING_SLOT_POS, .num_slots=1});
         }
      }
      if (negative == 14 && stages[i] == MESA_SHADER_TESS_CTRL) {
         nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
         nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
            .write_mask=15, .src_type=nir_type_float32,
            .io_semantics={.location=VARYING_SLOT_PATCH0, .num_slots=1});
      }
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      if (negative == 10 && stages[i] == MESA_SHADER_GEOMETRY)
         nir->info.gs.active_stream_mask = 3;
      if (stages[i] == producer) {
         struct hk_apple9_graphics_key key = {0};
         hk_apple9_graphics_key_device(&key, NULL, hk_device_physical(dev));
         bool supported = false;
         const char *reason = NULL;
         CHECK((producer == MESA_SHADER_GEOMETRY
            ? hk_apple9_native_gs_check(nir, &key, &supported, &reason)
            : hk_apple9_native_tess_check(nir, &key, &supported, &reason)) == VK_SUCCESS);
         printf("BOUNDARY %s v=%u case=%u candidate=%u reason=%s\n",
                producer == MESA_SHADER_GEOMETRY ? "gs" : "tes", vertices,
                negative, supported, reason ?: "policy/accepted");
      }
   }
   char name[128]; snprintf(name,sizeof(name),"%s-v%u-r%u",producer==MESA_SHADER_GEOMETRY?"gs":"tes",vertices,negative);
   allocations = 0;
   uint64_t start=cpu_time_ns();
   CHECK(hk_apple9_compile_graphics(dev,count,infos,NULL,NULL,NULL,NULL,out)==VK_SUCCESS);
   uint64_t ns=cpu_time_ns()-start;
   unsigned stage_bit = producer == MESA_SHADER_GEOMETRY
      ? HK_APPLE9_NATIVE_GS_BIT : HK_APPLE9_NATIVE_TES_BIT;
   bool expected=(hk_apple9_native_stage_policy(hk_device_physical(dev)) & stage_bit) &&
      (negative == 0 || negative == 13) &&
      (producer==MESA_SHADER_TESS_EVAL || vertices<=4);
   for(unsigned i=0;i<count;i++) {
      api[stages[i]]=container_of(out[i],struct hk_api_shader,vk);
      save_object(dev,name,i,api[stages[i]],expected);
   }
   printf("COMPILE %s native=%u cpu_ns=%"PRIu64"\n",name,expected,ns);
   bool recorded_native = expected && dev->dev.native_mesh_qualified;
   struct agx_shader_body *body = recorded_native
      ? hk_native_producer_main(hk_api_shader_native(api[producer]))->apple9_body : NULL;
   VkCommandPool pool = record(dev,name,producer,api,vertices,recorded_native);
   for(unsigned i=0;i<count;i++) hk_api_shader_destroy(&dev->vk,out[i],NULL);
   hk_apple9_variants_finish(dev); hk_apple9_bodies_finish(dev);
   /* Command storage owns the executable after API objects and the body
    * cache have gone away. Releasing the command pool releases the last pin. */
   if (body) CHECK(body->entry_pins == 1 && body->refcnt == 1 && body->bo->refcnt > 0);
   hk_DestroyCommandPool(hk_device_to_handle(dev),pool,NULL);
   hk_cmd_bo_cache_trim(dev); /* idle command BOs belong to the device */
   test_reset_usc_heap(dev);
   CHECK(live_bos==0);
}

int
main(int argc, char **argv)
{
   CHECK((argc==3 || argc==4) && (!getenv("HK_APPLE9_NATIVE_MESH") ||
         !strcmp(getenv("HK_APPLE9_NATIVE_MESH"), "0")) && !getenv("HK_APPLE9_GS_NATIVE") &&
         !getenv("HK_APPLE9_TESS_NATIVE") && !getenv("HK_APPLE9_NATIVE_MESH_PACKED_SHAPES"));
   directory=argv[2]; CHECK(mkdir(directory,0755)==0);
   char path[4096]; snprintf(path,sizeof(path),"%s/PROGRAMS.tsv",directory);
   ledger=fopen(path,"w");CHECK(ledger); fputs("case\tobject\tstage\tnative\tbytes\n",ledger);
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev=calloc(1,sizeof(*dev));struct hk_physical_device *pdev=calloc(1,sizeof(*pdev));CHECK(dev&&pdev);
   dev->dev.chip=pdev->dev.chip=AGX_CHIP_G16G;
   pdev->dev.native_mesh_qualified=strcmp(argv[1],"physical-negative")!=0;
   dev->dev.native_mesh_qualified=strcmp(argv[1],"logical-negative")!=0;
   pdev->disable_native_stages=!strcmp(argv[1],"drirc");
   dev->dev.shader_base=0x1200000000ull;
   dev->vk.base.type=VK_OBJECT_TYPE_DEVICE;
   dev->vk.alloc=*vk_default_allocator();dev->vk.physical=&pdev->vk;
   dev->vk.command_buffer_ops=&hk_cmd_buffer_ops;
   dev->vk.dispatch_table.AllocateCommandBuffers=vk_common_AllocateCommandBuffers;
   dev->vk.dispatch_table.ResetCommandBuffer=hk_ResetCommandBuffer;
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&dev->dev.apple9_entry_lock,mtx_plain);
   simple_mtx_init(&dev->apple9_variants.lock,mtx_plain);
   simple_mtx_init(&dev->apple9_bodies.lock,mtx_plain);list_inithead(&dev->apple9_bodies.slabs);
   simple_mtx_init(&pdev->debug_compile_lock,mtx_plain);
   unsigned policy = (!strcmp(argv[1], "on") || !strcmp(argv[1], "logical-negative")) ? 3
      : !strcmp(argv[1], "gs") ? 1 : !strcmp(argv[1], "tes") ? 2 : 0;
   CHECK(hk_apple9_native_stage_policy(pdev) == policy);
   const uint64_t flags = hk_physical_device_compiler_flags(pdev);
   bool was_killed = pdev->disable_native_stages;
   pdev->disable_native_stages = true;
   CHECK((flags ^ hk_physical_device_compiler_flags(pdev)) == (uint64_t)policy << 58);
   pdev->disable_native_stages = was_killed;
   printf("POLICY stages=%u index=%u flags=%016"PRIx64"\n", policy,
          hk_apple9_native_index_enabled(pdev), flags);
   if (argc == 4) {
      for (unsigned stage = 0; stage < 2; ++stage) {
         const unsigned bit = stage ? HK_APPLE9_NATIVE_TES_BIT : HK_APPLE9_NATIVE_GS_BIT;
         for (unsigned object = 0; object < (stage ? 4 : 3); ++object) {
            snprintf(path, sizeof(path), "%s/%s-v3-r0-%u.bin", argv[3], stage ? "tes" : "gs", object);
            FILE *f = fopen(path, "rb"); CHECK(f && fseek(f, 0, SEEK_END) == 0);
            long size = ftell(f); CHECK(size > 0 && fseek(f, 0, SEEK_SET) == 0);
            void *data = malloc(size); CHECK(data && fread(data, 1, size, f) == size); fclose(f);
            struct blob_reader reader; blob_reader_init(&reader, data, size);
            struct vk_shader *copy = NULL;
            VkResult result = hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &copy);
            CHECK(result == ((policy & bit) ? VK_SUCCESS : VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT));
            if (copy) hk_api_shader_destroy(&dev->vk, copy, NULL);
            free(data);
         }
      }
      hk_apple9_bodies_finish(dev); test_reset_usc_heap(dev); CHECK(live_bos == 0);
   }
   for(unsigned v=3;v<=5;v++)pipeline(dev,MESA_SHADER_GEOMETRY,v,0);
   pipeline(dev,MESA_SHADER_GEOMETRY,3,1);
   for (unsigned n=3; n<=10; ++n) pipeline(dev,MESA_SHADER_GEOMETRY,3,n);
   pipeline(dev,MESA_SHADER_TESS_EVAL,3,0);
   pipeline(dev,MESA_SHADER_TESS_EVAL,32,0);
   pipeline(dev,MESA_SHADER_TESS_EVAL,3,1);
   pipeline(dev,MESA_SHADER_TESS_EVAL,3,2);
   for (unsigned n=11; n<=14; ++n) pipeline(dev,MESA_SHADER_TESS_EVAL,3,n);
   /* Real deferred cache reload and native/fallback/native transitions. */
   compile_gs(dev,3,true,false,0);compile_gs(dev,4,true,false,0);compile_gs(dev,5,true,false,0);
   compile_tess(dev,3,true,false,0);compile_tess(dev,32,true,false,0);
   if(policy) {
      compile_gs(dev,3,true,true,0);compile_tess(dev,3,true,true,0);
   }
   fclose(ledger);
   simple_mtx_destroy(&pdev->debug_compile_lock);
   simple_mtx_destroy(&dev->apple9_bodies.lock);simple_mtx_destroy(&dev->apple9_variants.lock);
   simple_mtx_destroy(&dev->dev.apple9_entry_lock);
   test_reset_usc_heap(dev);hk_device_finish_usc_heap(dev);free(dev);free(pdev);glsl_type_singleton_decref();
   puts("PASS stage policy, complete serialization, cache reload, deferred transitions, VDM packets; GPU_execution=0");
   return 0;
}
