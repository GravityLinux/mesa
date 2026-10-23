/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_draw.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); failures++; } } while (0)
static unsigned failures, copies;

void __wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                               enum agx_barrier barrier,
                               enum libagx_program idx, void *data, size_t size);
void
__wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                          enum agx_barrier barrier,
                          enum libagx_program idx, void *data, size_t size)
{
   copies++;
   CHECK(barrier & AGX_PREGFX);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   if (!dev || !cmd || !pool)
      return 1;
   const uint64_t base = 0x1000000000ull, end = base + (1ull << 32);
   struct agx_va va = {.addr = base + 0x20000000};
   struct agx_bo bo = {.va = &va, .size = 128 * 1024 * 1024};
   struct poly_heap heap = {.base = va.addr, .size = bo.size};
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = base;
   dev->heap = &bo;
   dev->heap_init_once.called = true;
   dev->rodata.heap = (uintptr_t)&heap;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = (uintptr_t)pool;
   list_inithead(&cmd->control_streams);

   struct agx_draw draw = agx_draw_indexed(3, 1, 0, 0, 0, end + 0x4000, 64,
                                           AGX_INDEX_SIZE_U32, false);
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   struct agx_draw copied = hk_apple9_robust_index(cmd, draw);
   CHECK(copies == 1 && copied.indexed && agx_is_indirect(copied.b));
   CHECK(copied.index_buffer == va.addr && copied.index_buffer_range_B == bo.size);
   CHECK(copied.index_buffer >= base && copied.index_buffer < end);
   puts("high-VA binding: existing robust copy returns a reachable heap draw");

   draw.index_buffer = base - 64;
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   draw.index_buffer = end - 16;
   draw.start = 8;
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   puts("binding starts in aperture, firstIndex crosses its end: must copy");
   draw.start = 0;
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   draw.b = agx_grid_indirect(0x40000000);
   CHECK(hk_apple9_needs_index_copy(cmd, draw));
   draw.index_buffer = base + 0x4000;
   CHECK(!hk_apple9_needs_index_copy(cmd, draw));
   draw.index_buffer_range_B = 0;
   CHECK(!hk_apple9_needs_index_copy(cmd, draw));

   if (cmd->current_cs.post_gfx) {
      util_dynarray_fini(&cmd->current_cs.post_gfx->imm_writes);
      free(cmd->current_cs.post_gfx);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   free(pool);
   free(cmd);
   free(dev);
   return failures != 0;
}
