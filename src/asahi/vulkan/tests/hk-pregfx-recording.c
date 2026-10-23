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
   CHECK(pre && pre->sync_before_gfx && !cmd->sync_fallback);
   CHECK(util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event) == 3);
   ev = cmd->sync_events.data;
   CHECK(ev[0].stream == HK_SYNC_NONE);
   CHECK(ev[1].stream == pre->sync_id && ev[1].dst == HK_SYNC_ALL);
   CHECK(ev[2].stream == gfx.sync_id);
   CHECK(cmd->merge_event == 2);
   CHECK(cmd->control_streams.next == &pre->node && pre->node.next == &gfx.node);

   struct hk_sync_state state;
   hk_sync_init(&state);
   /* An earlier render may still consume the shared geometry heap. */
   hk_sync_work(&state, HK_SYNC_GRAPHICS, HK_SYNC_RENDER, 5);
   struct hk_sync_stream streams[2] = {
      {.engine = HK_SYNC_RENDER, .index = 6, .before = {5, 1}},
      {.engine = HK_SYNC_CDM, .index = 1, .before = {5, 0}},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(streams); ++i)
      streams[i].wait = streams[i].wait_tiling = hk_sync_empty();
   /* Previous heap consumer -> reset/helper -> graphics consumer. The
    * logical events now match the physical submission order without fallback.
    */
   CHECK(hk_sync_resolve(&state, ev, 3, cmd->sync_remap.data, streams, 2,
                         cmd->sync_fallback));
   CHECK(streams[0].wait_tiling.engine[HK_SYNC_CDM] == 1);
   CHECK(streams[1].wait.engine[HK_SYNC_RENDER] == 5);

   /* Later helper requests use the same fully ordered event, never an event
    * after their graphics consumer. */
   CHECK(hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true,
                                     HK_SYNC_DRAW_INDIRECT) == pre);
   CHECK(pre->sync_stages & HK_SYNC_DRAW_INDIRECT);
   CHECK(!cmd->sync_fallback && cmd->merge_event == 2);
   CHECK(util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event) == 3);

   /* Resolving another execution uses new queue-relative indices without
    * mutating the immutable recorded event identities. */
   streams[0].index = 7;
   streams[0].before[0] = streams[1].before[0] = 6;
   streams[0].before[1] = 2;
   streams[1].index = 2;
   streams[1].before[1] = 1;
   for (unsigned i = 0; i < ARRAY_SIZE(streams); ++i)
      streams[i].wait = streams[i].wait_tiling = hk_sync_empty();
   CHECK(hk_sync_resolve(&state, ev, 3, cmd->sync_remap.data, streams, 2, false));
   CHECK(streams[0].wait_tiling.engine[HK_SYNC_CDM] == 2);
   CHECK(streams[1].wait.engine[HK_SYNC_RENDER] == 6);

   /* An unclassified post-graphics helper still requests fallback. */
   struct hk_cs *post = hk_cmd_buffer_get_cs_general(
      cmd, &cmd->current_cs.post_gfx, true, HK_SYNC_ALL);
   CHECK(post && !post->sync_before_gfx && cmd->sync_fallback);

   free(post);
   free(pre);
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   free(pool);
   free(cmd);
   free(dev);
   return 0;
}
