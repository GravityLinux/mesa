/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "libagx_shaders.h"
#include "query.h"
#include "poly/geometry.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%u: %s\n", \
                          __FILE__, __LINE__, #c); return 1; } } while (0)

static int
record_and_replay(bool graphics)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);
   struct agx_va va = {.addr = 0x20000000};
   struct agx_bo bo = {.va = &va, .size = 128 * 1024 * 1024};
   dev->dev.chip = AGX_CHIP_G16G;
   dev->heap = &bo;
   dev->heap_init_once.called = true;
   struct poly_heap heap = {.base = va.addr, .size = bo.size};
   dev->rodata.heap = (uintptr_t)&heap;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = (uintptr_t)pool;
   list_inithead(&cmd->control_streams);

   struct hk_cs gfx = {.cmd = cmd, .type = HK_CS_VDM, .sync_id = 0};
   if (graphics) {
      cmd->current_cs.gfx = &gfx;
      list_addtail(&gfx.node, &cmd->control_streams);
      util_dynarray_append(&cmd->sync_remap, gfx.sync_id);
      hk_cmd_sync_work(cmd, &gfx, HK_SYNC_GRAPHICS);
   }

   uint64_t address = hk_heap(cmd);
   CHECK(address == (uintptr_t)&heap && hk_heap(cmd) == address);
   struct hk_cs *post = cmd->current_cs.post_gfx;
   CHECK(post && cmd->sync_fallback);
   CHECK(util_dynarray_num_elements(&post->imm_writes, struct libagx_imm_write) == 1);
   struct libagx_imm_write *reset = post->imm_writes.data;
   CHECK(reset->address == address + offsetof(struct poly_heap, bottom));
   CHECK(reset->value == 0);
   /* Record once, then replay N times with tessellation and geometry-sized
    * allocations. Uploads and the reset dispatch are never re-recorded.
    * Without an execution-time reset these exceed 128 MiB by submission 6.
    * The Air Vulkan regression also runs actual tessellation allocations.
    */
   for (unsigned submit = 0; submit < 64; ++submit) {
      *(uint32_t *)(uintptr_t)reset->address = reset->value;
      CHECK(heap.bottom == 0);
      heap.bottom += 8 * 1024 * 1024;
      CHECK(heap.bottom < heap.size);
      heap.bottom += 16 * 1024 * 1024;
      CHECK(heap.bottom < heap.size);
   }
   CHECK(cmd->heap_renders == 1);

   util_dynarray_fini(&post->imm_writes);
   free(post);
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
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
