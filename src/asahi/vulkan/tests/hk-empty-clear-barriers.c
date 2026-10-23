/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "hk_entrypoints.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); return 1; } } while (0)

int main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && cmd);
   dev->dev.chip = AGX_CHIP_G16G;
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   struct hk_cs older = {.cmd = cmd, .type = HK_CS_CDM, .sync_id = 0};
   struct hk_cs clear = {.cmd = cmd, .type = HK_CS_VDM, .sync_id = 1};
   hk_cmd_sync_work(cmd, &older, HK_SYNC_COPY);
   hk_cmd_sync_work(cmd, &clear, HK_SYNC_GRAPHICS);
   cmd->merge_cs = cmd->current_cs.gfx = &clear;
   cmd->merge_empty = true;
   cmd->merge_event = 1;
   VkMemoryBarrier2 barriers[] = {
      {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
       .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
       .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
       .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
       .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT},
      {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
       .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
       .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
       .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
       .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT},
      {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
       .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
       .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
       .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
       .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(barriers); ++i) {
      hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(cmd), &(VkDependencyInfo){
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1, .pMemoryBarriers = &barriers[i]});
   }
   CHECK(cmd->merge_cs == &clear && cmd->empty_barriers.size);
   CHECK(util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event) == 2);
   hk_cmd_empty_barriers_flush(cmd, true);
   CHECK(!cmd->empty_barriers.size && cmd->merge_cs == &clear);
   CHECK(cmd->merge_event == 4 && cmd->merge_reach == ~(VkPipelineStageFlags2)0);
   struct hk_sync_event *ev = cmd->sync_events.data;
   CHECK(ev[0].stream == 0 && ev[4].stream == 1);
   for (unsigned i = 0; i < ARRAY_SIZE(barriers); ++i) {
      CHECK(ev[i + 1].stream == HK_SYNC_NONE);
      CHECK(ev[i + 1].barrier_group == i + 1);
      CHECK(ev[i + 1].src == hk_sync_stage_mask(barriers[i].srcStageMask, true));
      CHECK(ev[i + 1].dst == hk_sync_stage_mask(barriers[i].dstStageMask, false));
   }
   struct hk_sync_state state;
   hk_sync_init(&state);
   uint32_t remap[] = {0, 1};
   struct hk_sync_stream streams[] = {
      {.engine = HK_SYNC_CDM, .index = 1},
      {.engine = HK_SYNC_RENDER, .index = 1, .before = {0, 1}},
   };
   CHECK(hk_sync_resolve(&state, ev, 5, remap, streams, 2, false));
   CHECK(streams[1].wait.engine[HK_SYNC_CDM] == 1);

   /* Rejected fusion emits the original barrier after the clear event. It
    * must still order a subsequent shader reader, including a zero-access
    * execution dependency; it must never become an earlier barrier. */
   util_dynarray_clear(&cmd->sync_events);
   cmd->merge_cs = cmd->current_cs.gfx = NULL;
   cmd->merge_empty = false;
   hk_cmd_sync_work(cmd, &clear, HK_SYNC_GRAPHICS);
   struct hk_empty_barrier pending = {
      .src = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dst = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .group = 7};
   util_dynarray_append(&cmd->empty_barriers, pending);
   hk_cmd_empty_barriers_flush(cmd, false);
   struct hk_cs reader = {.cmd = cmd, .type = HK_CS_CDM, .sync_id = 0};
   hk_cmd_sync_work(cmd, &reader, HK_SYNC_COMPUTE);
   ev = cmd->sync_events.data;
   CHECK(ev[0].stream == 1 && ev[1].barrier_group == 7 && ev[2].stream == 0);
   hk_sync_init(&state);
   streams[0] = (struct hk_sync_stream){.engine = HK_SYNC_CDM, .index = 1, .before = {1, 0}};
   streams[1] = (struct hk_sync_stream){.engine = HK_SYNC_RENDER, .index = 1};
   CHECK(hk_sync_resolve(&state, ev, 3, remap, streams, 2, false));
   CHECK(streams[0].wait.engine[HK_SYNC_RENDER] == 1);
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->empty_barriers);
   vk_object_base_finish(&cmd->vk.base);
   free(cmd); free(dev);
   puts("PASS deferred clear barriers: predecessor ordering, grouping, fallback WAR");
   return 0;
}
