/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"

/* Exercise real recording and encoders without a DRM device. Streams and pool
 * storage are host-backed; no shaders or submissions are needed for barriers.
 */
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%u: %s\n", \
                          __FILE__, __LINE__, #c); return 1; } } while (0)

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && cmd);
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   uint8_t pool[4096] = {0};
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;

   uint8_t compute_bytes[4096] = {0}, render_bytes[4096] = {0};
   struct hk_cs compute = {
      .cmd = cmd, .type = HK_CS_CDM,
      .current = compute_bytes, .end = compute_bytes + sizeof(compute_bytes),
      .sync_stages = HK_SYNC_COMPUTE | HK_SYNC_DISPATCH_INDIRECT,
      .cdm_dirty = true,
   };
   struct hk_cs render = {
      .cmd = cmd, .type = HK_CS_VDM,
      .current = render_bytes, .end = render_bytes + sizeof(render_bytes),
      .sync_stages = HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT,
      .stats.cmds = 1,
   };
   cmd->current_cs.cs = &compute;
   cmd->current_cs.gfx = &render;

   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_NONE,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 0);
   CHECK(cmd->current_cs.cs == &compute && cmd->current_cs.gfx == &render);
   CHECK(compute.current == compute_bytes);

   /* Zero-access WAR still emits an execution barrier, without ending the
    * independent graphics stream or splitting compute into another job.
    */
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 0);
   CHECK(cmd->current_cs.cs == &compute && cmd->current_cs.gfx == &render);
   CHECK(compute.current == compute_bytes + AGX_CDM_BARRIER_LENGTH);
   /* G16 orders shader-to-shader dependencies with the compute-only mask and
    * leaves the full flush to consumers outside the shader core. */
   CHECK(*(uint32_t *)compute_bytes == 0x60000160);
   CHECK(compute.cdm_dirty && compute.cdm_compute_barrier &&
         compute.stats.flushes == 1);

   /* Indirect grids are fetched by the CDM itself: full flush. */
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
   CHECK(!compute.cdm_dirty && !compute.cdm_compute_barrier);
   CHECK(*(uint32_t *)(compute_bytes + AGX_CDM_BARRIER_LENGTH) != 0x60000160);
   compute.current = compute_bytes + AGX_CDM_BARRIER_LENGTH;
   compute.stats.flushes = 1;
   cmd->current_cs.cs = &compute;

   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   CHECK(cmd->current_cs.cs == &compute && cmd->current_cs.gfx == NULL);
   CHECK(render.current != render_bytes);

   render.current = render_bytes;
   cmd->current_cs.gfx = &render;
   compute.cdm_dirty = true;
   hk_cmd_sync_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   CHECK(cmd->current_cs.cs == NULL && cmd->current_cs.gfx == NULL);
   CHECK(!compute.cdm_dirty && compute.stats.flushes == 2);
   const struct hk_sync_event *event =
      util_dynarray_top_ptr(&cmd->sync_events, struct hk_sync_event);
   CHECK(event->src == (HK_SYNC_COMPUTE | HK_SYNC_DISPATCH_INDIRECT));
   CHECK(event->dst == HK_SYNC_FRAGMENT);

   /* A state-induced render split has an implicit store -> load dependency. */
   struct hk_cs resumed = {.cmd = cmd, .type = HK_CS_VDM, .sync_id = 7};
   size_t before = util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event);
   hk_cmd_sync_work(cmd, &resumed, HK_SYNC_GRAPHICS);
   CHECK(util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event) == before + 2);
   event = util_dynarray_element(&cmd->sync_events, struct hk_sync_event, before);
   CHECK(event->stream == HK_SYNC_NONE);
   CHECK(event->src & HK_SYNC_FRAGMENT);
   /* Attachment loads happen in the fragment stage; tiling need not wait. */
   CHECK(event->dst == HK_SYNC_FRAGMENT);
   CHECK(!cmd->sync_render_continuation);

   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   free(cmd);
   free(dev);
   return 0;
}
