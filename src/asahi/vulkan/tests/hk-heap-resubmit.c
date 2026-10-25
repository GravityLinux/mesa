/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "libagx_shaders.h"
#include "query.h"
#include "poly/geometry.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%u: %s\n", \
                          __FILE__, __LINE__, #c); return 1; } } while (0)

static struct libagx_heap_reset_args reset;
static unsigned reset_count;
static bool reset_valid;

void __wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                                enum agx_barrier barrier,
                                enum libagx_program idx, void *data, size_t size);
void
__wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                           enum agx_barrier barrier,
                           enum libagx_program idx, void *data, size_t size)
{
   reset_valid = idx == LIBAGX_HEAP_RESET && size == sizeof(reset) &&
                 barrier == (AGX_BARRIER_ALL | AGX_PREGFX) &&
                 grid.count[0] == 1 && grid.count[1] == 1 && grid.count[2] == 1;
   if (!reset_valid)
      return;

   reset = *(struct libagx_heap_reset_args *)data;
   reset_count++;
   reset_valid = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.pre_gfx, true, HK_SYNC_ALL) != NULL;
}

static int
record_and_replay(bool graphics)
{
   reset_count = 0;
   reset_valid = false;
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);
   struct agx_va va = {.addr = 0x20000000};
   struct agx_bo bo = {.va = &va, .size = 128 * 1024 * 1024};
   dev->dev.chip = AGX_CHIP_G16G;
   dev->heap = &bo;
   dev->heap_va = &va;
   va.size_B = bo.size;
   dev->heap_init_once.called = true;
   struct poly_heap heap = {.base = va.addr, .size = bo.size};
   dev->rodata.heap = (uintptr_t)&heap;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = (uintptr_t)pool;
   list_inithead(&cmd->control_streams);

   struct hk_cs gfx = {.cmd = cmd, .type = HK_CS_VDM, .sync_id = 0};
   if (graphics) {
      cmd->current_cs.gfx = &gfx;
      list_addtail(&gfx.node, &cmd->control_streams);
      util_dynarray_append(&cmd->sync_remap, gfx.sync_id);
      util_dynarray_append(&cmd->sync_first_event, (uint32_t)HK_SYNC_NONE);
      hk_cmd_sync_work(cmd, &gfx, HK_SYNC_GRAPHICS);
   }

   uint64_t address = hk_heap(cmd);
   CHECK(address == (uintptr_t)&heap && hk_heap(cmd) == address);
   struct hk_cs *pre = cmd->current_cs.pre_gfx;
   CHECK(pre && !cmd->current_cs.post_gfx);
   CHECK(cmd->sync_fallback == !graphics);
   CHECK(pre->sync_before_gfx == graphics);
   CHECK(reset_valid && reset_count == 1);
   CHECK(reset.heap == address);
   if (graphics)
      CHECK(pre->node.next == &gfx.node);
   /* Record once, then replay N times with tessellation and geometry-sized
    * allocations. Uploads and the reset dispatch are never re-recorded.
    * Without an execution-time reset these exceed 128 MiB by submission 6.
    * The Air Vulkan regression also runs actual tessellation allocations.
    */
   const uint32_t mib = 1024 * 1024 / POLY_HEAP_UNIT_B;
   for (unsigned submit = 0; submit < 64; ++submit) {
      /* libagx_heap_reset */
      struct poly_heap *h = (struct poly_heap *)(uintptr_t)reset.heap;
      h->high_water = MAX2(h->high_water, h->bottom);
      h->bottom = 0;
      CHECK(poly_heap_needed_B(&heap) == (submit ? 24ull << 20 : 0));
      heap.bottom += 8 * mib;
      CHECK(poly_heap_needed_B(&heap) < heap.size);
      heap.bottom += 16 * mib;
      CHECK(poly_heap_needed_B(&heap) < heap.size);
   }
   CHECK(cmd->heap_renders == 1);

   free(pre);
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}

int
main(void)
{
   CHECK(record_and_replay(true) == 0);
   CHECK(record_and_replay(false) == 0);
   return 0;
}
