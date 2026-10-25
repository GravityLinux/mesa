/* SPDX-License-Identifier: MIT
 * Record real native/emulated draw packets with host-backed BOs. No GPU access.
 */
int structural_reference_main(void);
#define main structural_reference_main
#include "hk-structural-uploads.c"
#undef main
#include "compiler/nir/nir_builder.h"

static struct hk_api_shader *
api_shader(mesa_shader_stage stage)
{
   struct hk_api_shader *s = calloc(1, sizeof(*s) +
                                     hk_num_variants(stage) * sizeof(struct hk_shader));
   CHECK(s);
   s->vk.stage = stage;
   return s;
}

static nir_shader *
recording_mesh(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "qualified native draw recording");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 2;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                     nir_imm_int(&b, 2), nir_imm_int(&b, 0));
   for (unsigned i = 0; i < 3; ++i)
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 0, 0, .5, 1),
         nir_imm_int(&b, i), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   for (unsigned i = 0; i < 2; ++i)
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, i), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

int
main(int argc, char **argv)
{
   CHECK(argc == 2);
   CHECK(!getenv("HK_APPLE9_NATIVE_MESH"));
   CHECK(!getenv("HK_APPLE9_NATIVE_MESH_PACKED_SHAPES"));
   bool expected = !strcmp(argv[1], "enabled");
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->vk.dispatch_table.AllocateCommandBuffers = vk_common_AllocateCommandBuffers;
   dev->vk.dispatch_table.ResetCommandBuffer = hk_ResetCommandBuffer;
   dev->dev.chip = pdev->dev.chip = AGX_CHIP_G16G;
   pdev->dev.native_mesh_qualified = strcmp(argv[1], "kernel-absent") != 0;
   dev->dev.native_mesh_qualified = strcmp(argv[1], "vm-absent") != 0;
   pdev->disable_native_index = !strcmp(argv[1], "drirc");
   simple_mtx_init(&dev->dev.apple9_entry_lock, mtx_plain);
   hk_device_init_usc_heap(dev);
   VkDevice device = hk_device_to_handle(dev);
   const VkCommandPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool;
   CHECK(hk_CreateCommandPool(device, &ci, NULL, &pool) == VK_SUCCESS);
   VkCommandBuffer cb;
   const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .commandBufferCount = 1};
   CHECK(vk_common_AllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS);
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(cb);
   struct hk_api_shader *mesh = api_shader(MESA_SHADER_MESH);
   struct hk_api_shader *native = api_shader(MESA_SHADER_MESH);
   struct hk_api_shader *fs = api_shader(MESA_SHADER_FRAGMENT);
   struct hk_api_shader *native_fs = api_shader(MESA_SHADER_FRAGMENT);
   mesh->apple9_native = native;
   fs->apple9_native = native_fs;
   native->apple9_native_leaf = native_fs->apple9_native_leaf = true;
   memset(native->apple9_native_pair_id, 0x31, sizeof(native->apple9_native_pair_id));
   memcpy(native_fs->apple9_native_pair_id, native->apple9_native_pair_id,
          sizeof(native->apple9_native_pair_id));
   struct hk_shader *main = hk_native_producer_main(native);
   struct hk_shader *object = hk_native_producer_object(native);
   main->info.stage = MESA_SHADER_MESH;
   object->info.stage = MESA_SHADER_TASK;
   main->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   object->info.apple9_native_stage = HK_APPLE9_NATIVE_OBJECT;
   const struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   const struct agx_apple9_native_mesh_options options = {
      .physical_threads = 16, .packed_index_publication = true};
   nir_shader *nir = recording_mesh();
   const char *reason = NULL;
   CHECK(agx_compile_apple9_native_mesh(nir, true, &options, &main->b,
                                      &main->info.apple9_native, &reason));
   ralloc_free(nir);
   nir = agx_nir_apple9_taskless_object(&key);
   CHECK(agx_compile_apple9_native_object(nir, true, &key, &object->b,
                                        &object->info.apple9_native, &reason));
   ralloc_free(nir);
   main->bo = agx_bo_create(&dev->dev, 4096, 4096, 0, "mesh");
   object->bo = agx_bo_create(&dev->dev, 4096, 4096, 0, "object");
   agx_shader_body_init_bo(main->bo);
   agx_shader_body_init_bo(object->bo);
   main->apple9_body = &main->bo->apple9_body;
   object->apple9_body = &object->bo->apple9_body;
   cmd->state.gfx.shaders[MESA_SHADER_MESH] = mesh;
   cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = fs;
   cmd->state.gfx.render.layer_count = 1;
   cmd->vk.dynamic_graphics_state.rs.polygon_mode = VK_POLYGON_MODE_FILL;
   const uint32_t groups[] = {1, 1, 1};
   bool route = hk_apple9_use_native_mesh(cmd, groups, 0);
   CHECK(route == expected);
   uint32_t command[256] = {0};
   struct hk_cs cs = {.cmd = cmd, .type = HK_CS_VDM,
                     .start = (uint8_t *)command, .current = (uint8_t *)command};
   const struct agx_apple9_render_stage stage = {
      .binary = main->b.binary, .binary_size = main->b.info.binary_size};
   const struct agx_apple9_render_pipeline pipeline = {
      .vertex = &stage, .fragment = &stage,
      .ppp = 0x4000, .primitive = AGX_PRIMITIVE_TRIANGLES,
      .ppp_viewport_offset = 0x100, .ppp_viewport_words = 10};
   const struct agx_apple9_uniform_draw record = {0};
   const struct agx_draw draw = {.b = agx_3d(6, 1, 1)};
   CHECK(hk_apple9_record_draw(cmd, &cs, &pipeline, draw, route ? 0x8000 : 0,
                             native, main, &record));
   CHECK(cmd->vk.record_result == VK_SUCCESS);
   CHECK(cs.apple9_native_mesh == expected);
   if (expected) {
      CHECK(cs.current - cs.start == HK_APPLE9_MESH_COMMAND_BYTES - 4);
      CHECK(command[0] == 0x40000d2c);
      CHECK(command[10] == 1 && command[11] == 0x400810);
      CHECK(command[39] == 0xc0000000);
      CHECK(main->apple9_body->entry_pins == 1 && object->apple9_body->entry_pins == 1);
   } else {
      uint8_t ordinary[1024] = {0};
      uint8_t *end = agx_apple9_emit_direct_draw(ordinary, &pipeline, 6, 1, 0);
      CHECK(cs.current - cs.start == end - ordinary - 4);
      CHECK(!memcmp(command, ordinary, end - ordinary));
      CHECK(main->apple9_body->entry_pins == 0 && object->apple9_body->entry_pins == 0);
      uint32_t state[AGX_APPLE9_MESH_STATE_WORDS];
      CHECK(!hk_apple9_prepare_native_mesh(cmd, main, object, &record, 0x8000, state));
   }
   hk_DestroyCommandPool(device, pool, NULL);
   hk_cmd_bo_cache_finish(dev); /* idle command BOs belong to the device */
   CHECK(main->apple9_body->entry_pins == 0 && object->apple9_body->entry_pins == 0);
   CHECK(main->bo->refcnt == 1 && object->bo->refcnt == 1);
   agx_bo_unreference(&dev->dev, main->bo);
   agx_bo_unreference(&dev->dev, object->bo);
   free(main->b.binary); free(object->b.binary);
   free(native); free(mesh); free(native_fs); free(fs);
   hk_device_finish_usc_heap(dev);
   simple_mtx_destroy(&dev->dev.apple9_entry_lock);
   CHECK(live_bos == 0);
   free(pdev); free(dev);
   glsl_type_singleton_decref();
   printf("PASS route=%s; actual draw packet, preparation, launch, pin lifetime; GPU_execution=0\n",
          expected ? "native" : "emulated");
   return 0;
}
