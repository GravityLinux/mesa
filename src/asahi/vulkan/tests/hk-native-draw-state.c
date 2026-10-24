/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Draws reach hk_draw incomplete: hk_draw_indexed passes neither the bound
 * index size nor primitive restart. Native routing must see the draw
 * completed from command buffer state (Air 39b7da1c: an indexed native GS
 * fetched 32-bit indices as bytes). No GPU or live Vulkan device is needed.
 */
#include "../hk_cmd_draw.c"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)

VkResult __wrap___vk_errorf(const void *, VkResult, const char *, int, const char *, ...);
VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   return error;
}

static struct hk_api_shader *
shader(mesa_shader_stage stage)
{
   struct hk_api_shader *s =
      calloc(1, sizeof(*s) + hk_num_variants(stage) * sizeof(struct hk_shader));
   CHECK(s);
   s->vk.stage = stage;
   return s;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && cmd);
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* A native GS pipeline: GS 1->1 at T16 V3/P1, paired VS and FS. */
   struct hk_api_shader *gs = shader(MESA_SHADER_GEOMETRY), *ng = shader(MESA_SHADER_GEOMETRY);
   struct hk_api_shader *vs = shader(MESA_SHADER_VERTEX), *nv = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *fs = shader(MESA_SHADER_FRAGMENT), *nf = shader(MESA_SHADER_FRAGMENT);
   gs->apple9_native = ng; vs->apple9_native = nv; fs->apple9_native = nf;
   ng->apple9_native_leaf = nv->apple9_native_leaf = nf->apple9_native_leaf = true;
   memset(ng->apple9_native_pair_id, 0x5a, sizeof(ng->apple9_native_pair_id));
   memcpy(nv->apple9_native_pair_id, ng->apple9_native_pair_id, sizeof(ng->apple9_native_pair_id));
   memcpy(nf->apple9_native_pair_id, ng->apple9_native_pair_id, sizeof(ng->apple9_native_pair_id));
   struct hk_shader *main_ = hk_native_producer_main(ng);
   main_->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   hk_native_producer_object(ng)->info.apple9_native_stage = HK_APPLE9_NATIVE_OBJECT;
   main_->info.apple9_native.key.physical_threads = 16;
   main_->info.apple9_native.layout.max_vertices = 3;
   main_->info.apple9_native.layout.max_primitives = 1;
   gfx->shaders[MESA_SHADER_GEOMETRY] = gs;
   gfx->shaders[MESA_SHADER_VERTEX] = vs;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

   /* Bound state: a 32-bit index buffer, restart off. */
   gfx->index.size = AGX_INDEX_SIZE_U32;
   dyn->ia.primitive_restart_enable = false;

   /* Exactly what hk_draw_indexed builds. */
   struct agx_draw entry =
      agx_draw_indexed(6, 1, 3, -6, 0, 0x100000, 64, 0, 0);
   CHECK(entry.index_size != AGX_INDEX_SIZE_U32);

   struct agx_draw draw = hk_draw_complete_from_state(cmd, entry);
   CHECK(draw.index_size == AGX_INDEX_SIZE_U32 && !draw.restart);
   CHECK(agx_index_size_to_B(draw.index_size) == 4);
   CHECK(hk_apple9_use_native_gs(cmd, draw));

   /* Primitive restart from state keeps the draw on the fallback. */
   dyn->ia.primitive_restart_enable = true;
   draw = hk_draw_complete_from_state(cmd, entry);
   CHECK(draw.restart && !hk_apple9_use_native_gs(cmd, draw));

   /* Non-indexed draws never carry restart; the enabled state alone still
    * keeps native GS off. */
   draw = hk_draw_complete_from_state(cmd, (struct agx_draw){.b = agx_3d(6, 1, 1)});
   CHECK(!draw.restart && !hk_apple9_use_native_gs(cmd, draw));
   dyn->ia.primitive_restart_enable = false;
   CHECK(hk_apple9_use_native_gs(cmd, draw));

   free(gs); free(ng); free(vs); free(nv); free(fs); free(nf);
   free(cmd);
   free(dev);
   puts("native routing sees index size and restart from command buffer state");
   return 0;
}
