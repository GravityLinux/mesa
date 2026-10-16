/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%u: %s\n", \
                          __FILE__, __LINE__, #c); return 1; } } while (0)

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   list_inithead(&cmd->control_streams);

   /* A pass without tess/GS has already recorded ordinary graphics work. */
   struct hk_cs gfx = {.cmd = cmd, .type = HK_CS_VDM, .sync_id = 0};
   cmd->current_cs.gfx = &gfx;
   list_addtail(&gfx.node, &cmd->control_streams);
   util_dynarray_append(&cmd->sync_remap, gfx.sync_id);
   hk_cmd_sync_work(cmd, &gfx, HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT);
   cmd->merge_cs = &gfx;
   cmd->merge_event = 0;

   /* Ending that pass keeps it open for merging. A kept attachment barrier
    * inserts an event before it, invalidating any cached absolute index.
    */
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   CHECK(cmd->merge_cs == &gfx && cmd->merge_event == 1);
   struct hk_sync_event *ev = cmd->sync_events.data;
   CHECK(ev[0].stream == HK_SYNC_NONE && ev[1].stream == gfx.sync_id);

   /* BeginRendering continues the same stream and consumes merge_cs. The
    * continued pass's first tess/GS draw creates its pre-gfx helper through
    * the real recording path, after the kept barrier but before all draws.
    */
   cmd->merge_cs = NULL;
   struct hk_cs *pre = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.pre_gfx, true, HK_SYNC_PRE_RASTER);
   CHECK(pre && cmd->sync_fallback);
   CHECK(util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event) == 3);
   ev = cmd->sync_events.data;
   CHECK(ev[0].stream == HK_SYNC_NONE);
   CHECK(ev[1].stream == gfx.sync_id);
   CHECK(ev[2].stream == pre->sync_id);
   CHECK(cmd->merge_event == 1);
   CHECK(cmd->control_streams.next == &pre->node && pre->node.next == &gfx.node);

   struct hk_sync_state state;
   hk_sync_init(&state);
   struct hk_sync_stream streams[2] = {
      {.engine = HK_SYNC_RENDER, .index = 1, .before = {0, 1}},
      {.engine = HK_SYNC_CDM, .index = 1, .before = {0, 0}},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(streams); ++i)
      streams[i].wait = streams[i].wait_tiling = hk_sync_empty();
   /* Helpers reordered around rendering request the normal full-ordering
    * fallback at recording time. Resolve their physical submission order. */
   CHECK(!hk_sync_resolve(&state, ev, 3, cmd->sync_remap.data, streams, 2,
                          cmd->sync_fallback));
   CHECK(streams[0].wait_tiling.engine[HK_SYNC_CDM] == 1);
   CHECK(streams[1].wait.engine[HK_SYNC_RENDER] == 0);

   /* A later request still records the additional scope on the same helper. */
   CHECK(hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true,
                                     HK_SYNC_DRAW_INDIRECT) == pre);
   CHECK(pre->sync_stages & HK_SYNC_DRAW_INDIRECT);
   CHECK(cmd->sync_fallback && cmd->merge_event == 1);

   free(pre);
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}
