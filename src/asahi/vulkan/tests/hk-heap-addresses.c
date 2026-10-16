/* SPDX-License-Identifier: MIT */
/* Exercise the actual recording helpers, including private draw constructors.
 * Only shader execution is replaced; heap selection, uploads and addresses are
 * recorded by the driver. Including the implementation keeps test hooks out of
 * the production driver interface.
 */
#include "../hk_cmd_draw.c"

static unsigned failures;

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         failures++;                                                           \
      }                                                                        \
   } while (0)

void __wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                                enum agx_barrier barrier,
                                enum libagx_program idx, void *data,
                                size_t size);
void
__wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                           enum agx_barrier barrier, enum libagx_program idx,
                           void *data, size_t size)
{
   struct hk_cs **target =
      barrier & AGX_PREGFX ? &cmd->current_cs.pre_gfx : &cmd->current_cs.cs;
   CHECK(hk_cmd_buffer_get_cs_general(cmd, target, true, HK_SYNC_ALL));
}

static void
check_draw(const char *name, unsigned render, struct agx_draw draw,
           const struct agx_bo *bo, uint64_t heap_desc)
{
   struct poly_heap *heap = (void *)(uintptr_t)heap_desc;
   printf("render=%u %s descriptor=%#" PRIx64 " allocation_base=%#" PRIx64
          " draw_base=%#" PRIx64 " range=%u\n",
          render, name, heap_desc, heap->base, draw.index_buffer,
          draw.index_buffer_range_B);
   CHECK(draw.indexed && agx_is_indirect(draw.b));
   CHECK(draw.index_buffer == bo->va->addr);
   CHECK(draw.index_buffer_range_B == bo->size);
   /* The indirect record is a separate absolute upload, not heap-relative. */
   CHECK(draw.b.ptr != bo->va->addr && draw.b.ptr != heap_desc);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   if (!dev || !cmd || !pool)
      return 1;
   struct agx_va va = {.addr = 0x20000000};
   struct agx_bo bo = {.va = &va, .size = 128 * 1024 * 1024};
   struct poly_heap heap = {.base = va.addr, .size = bo.size};
   dev->dev.chip = AGX_CHIP_G16G;
   dev->heap = &bo;
   dev->heap_init_once.called = true;
   dev->rodata.heap = (uintptr_t)&heap;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = (uintptr_t)pool;
   list_inithead(&cmd->control_streams);

   struct hk_api_shader *tcs =
      calloc(1, sizeof(*tcs) + sizeof(struct hk_shader));
   struct hk_api_shader *gs =
      calloc(1, sizeof(*gs) + HK_GS_VARIANTS * sizeof(struct hk_shader));
   if (!tcs || !gs)
      return 1;
   tcs->vk.stage = MESA_SHADER_TESS_CTRL;
   tcs->variants[0].info.tess.tcs_output_stride = 16;
   tcs->variants[0].info.tess.tcs_output_patch_size = 3;
   gs->vk.stage = MESA_SHADER_GEOMETRY;
   gs->variants[HK_GS_VARIANT_COUNT].info.gs.shape =
      POLY_GS_SHAPE_DYNAMIC_INDEXED;
   cmd->state.gfx.shaders[MESA_SHADER_TESS_CTRL] = tcs;
   cmd->state.gfx.tess.info.spacing = TESS_SPACING_EQUAL;
   cmd->vk.dynamic_graphics_state.ts.patch_control_points = 3;
   cmd->vk.dynamic_graphics_state.ia.primitive_topology =
      VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;

   struct hk_cs gfx[4] = {0};
   uint64_t descriptors[4];
   for (unsigned render = 0; render < ARRAY_SIZE(gfx); ++render) {
      /* Model the recording boundary that end_graphics uses: the next render
       * uses the primary heap, with stable distinct graphics stream IDs. */
      cmd->uses_heap = false;
      gfx[render] = (struct hk_cs){
         .cmd = cmd,
         .type = HK_CS_VDM,
         .sync_id = util_dynarray_num_elements(&cmd->sync_remap, uint32_t)};
      cmd->current_cs.gfx = &gfx[render];
      list_addtail(&gfx[render].node, &cmd->control_streams);
      util_dynarray_append(&cmd->sync_remap, gfx[render].sync_id);
      hk_cmd_sync_work(cmd, &gfx[render], HK_SYNC_GRAPHICS);
      const struct agx_bo *selected = &bo;
      descriptors[render] = hk_heap(cmd);
      struct poly_heap *heap = (void *)(uintptr_t)descriptors[render];
      CHECK(heap->base == selected->va->addr && heap->size == selected->size);
      CHECK(hk_heap(cmd) == descriptors[render]);
      struct hk_cs *post = cmd->current_cs.post_gfx;
      CHECK(post && cmd->sync_fallback);
      CHECK(util_dynarray_num_elements(&post->imm_writes, struct libagx_imm_write) == render + 1);
      const struct libagx_imm_write *reset = util_dynarray_top_ptr(&post->imm_writes, struct libagx_imm_write);
      CHECK(reset->address == descriptors[render] + offsetof(struct poly_heap, bottom) && reset->value == 0);
      for (unsigned i = 0; i < render; ++i)
         CHECK(descriptors[i] == descriptors[render]);

      struct agx_draw direct = {.b = agx_3d(12, 1, 1)};
      check_draw("adjacency", render, hk_apple9_draw_without_adj(cmd, direct),
                 selected, descriptors[render]);
      struct agx_draw indexed = agx_draw_indexed(12, 1, 0, 0, 0, 0x40000000, 48,
                                                 AGX_INDEX_SIZE_U32, false);
      check_draw("robust-copy", render, hk_apple9_robust_index(cmd, indexed),
                 selected, descriptors[render]);
      check_draw("fan", render, hk_apple9_unroll_fan(cmd, direct), selected,
                 descriptors[render]);
      indexed = agx_draw_indexed_indirect((uintptr_t)pool, 0x40000000, 48,
                                          AGX_INDEX_SIZE_U32, true);
      check_draw("restart", render, hk_draw_without_restart(cmd, indexed, 1),
                 selected, descriptors[render]);

      for (unsigned indirect = 0; indirect < 2; ++indirect) {
         struct poly_tess_params params;
         hk_upload_tess_params(cmd, &params, indirect ? indexed : direct);
         printf("render=%u tess-%s descriptor=%#" PRIx64 " patch_base=%#" PRIx64
                "\n",
                render, indirect ? "indirect" : "direct", params.heap,
                params.patch_coord_buffer);
         CHECK(params.heap == descriptors[render]);
         CHECK(params.patch_coord_buffer == selected->va->addr);
         CHECK(params.out_draws != params.heap &&
               params.out_draws != selected->va->addr);
      }
      cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] = gs;
      hk_upload_geometry_params(cmd, indexed);
      printf("render=%u geometry index_base=%#" PRIx64 "\n", render,
             cmd->geom_index_buffer);
      CHECK(cmd->geom_index_buffer == selected->va->addr);
      CHECK(cmd->geom_index_count == selected->size);
      cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] = NULL;
      list_del(&cmd->current_cs.pre_gfx->node);
      free(cmd->current_cs.pre_gfx);
      cmd->current_cs.pre_gfx = NULL;
   }
   /* Every render uses the same immutable primary-heap descriptor. */
   for (unsigned i = 0; i < ARRAY_SIZE(gfx); ++i) {
      struct poly_heap *heap = (void *)(uintptr_t)descriptors[i];
      CHECK(heap->base == va.addr && heap->size == bo.size);
   }
   CHECK(cmd->heap_renders == 4);
   util_dynarray_fini(&cmd->current_cs.post_gfx->imm_writes);
   free(cmd->current_cs.post_gfx);
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   free(tcs);
   free(gs);
   free(pool);
   free(cmd);
   free(dev);
   return failures != 0;
}
