/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "libagx_shaders.h"
#include "poly/geometry.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%u: %s\n", \
                          __FILE__, __LINE__, #c); return 1; } } while (0)

/* Replace only shader execution, not heap/stream recording. Save the immutable
 * reset dispatch so the host can replay the recorded commands without DRM.
 */
static struct libagx_write_u32_args reset;
static enum agx_barrier reset_barrier;
static unsigned resets;
static bool invalid_dispatch;

void __wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                               enum agx_barrier barrier, enum libagx_program idx,
                               void *data, size_t size);

void
__wrap_hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                          enum agx_barrier barrier, enum libagx_program idx,
                          void *data, size_t size)
{
   if (idx != LIBAGX_WRITE_U32 || size != sizeof(reset)) {
      invalid_dispatch = true;
      return;
   }
   struct hk_cs **target = barrier & AGX_PREGFX
                             ? &cmd->current_cs.pre_gfx : &cmd->current_cs.cs;
   if (!hk_cmd_buffer_get_cs_general(cmd, target, true, HK_SYNC_ALL)) {
      invalid_dispatch = true;
      return;
   }
   memcpy(&reset, data, size);
   reset_barrier = barrier;
   resets++;
}

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

   resets = 0;
   invalid_dispatch = false;
   uint64_t address = hk_heap(cmd);
   CHECK(address && !invalid_dispatch && resets == 1);
   CHECK(hk_heap(cmd) == address && resets == 1);
   CHECK(reset.address == address + offsetof(struct poly_heap, bottom));
   CHECK(reset.value == 0 && (reset_barrier & AGX_BARRIER_ALL));
   CHECK(!!(reset_barrier & AGX_PREGFX) == graphics);
   CHECK(!cmd->current_cs.post_gfx && !cmd->sync_fallback);
   if (graphics)
      CHECK(cmd->current_cs.pre_gfx->pregfx_precise);

   struct poly_heap *heap = (void *)(uintptr_t)address;
   CHECK(heap->base == va.addr && heap->size == bo.size);
   /* Record once, then replay N times with tessellation and geometry-sized
    * allocations. Uploads and the reset dispatch are never re-recorded.
    * Without an execution-time reset these exceed 128 MiB by submission 6.
    * The Air Vulkan regression also runs actual tessellation allocations.
    */
   for (unsigned submit = 0; submit < 64; ++submit) {
      *(uint32_t *)(uintptr_t)reset.address = reset.value;
      CHECK(heap->bottom == 0);
      heap->bottom += 8 * 1024 * 1024;
      CHECK(heap->bottom < heap->size);
      heap->bottom += 16 * 1024 * 1024;
      CHECK(heap->bottom < heap->size);
   }
   CHECK(resets == 1 && cmd->heap_renders == 1);

   free(graphics ? cmd->current_cs.pre_gfx : cmd->current_cs.cs);
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
   setenv("HK_APPLE9_PREGFX_OVERLAP", "1", 1);
   CHECK(record_and_replay(true) == 0);
   CHECK(record_and_replay(false) == 0);
   return 0;
}
