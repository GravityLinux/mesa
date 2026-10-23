/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); return 1; } } while (0)

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && cmd);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = 1ull << 40;
   cmd->vk.base.device = &dev->vk;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   util_dynarray_append(&cmd->sync_remap, (uint32_t)0);
   util_dynarray_append(&cmd->sync_first_event, (uint32_t)HK_SYNC_NONE);
   uint8_t emitted[16384] = {0}, expected[16384] = {0};
   struct hk_cs cs = {
      .cmd = cmd, .type = HK_CS_VDM, .sync_id = 0,
      .start = emitted, .current = emitted, .end = emitted + sizeof(emitted),
   };
   struct hk_cs other = {0};
   cmd->current_cs.gfx = cmd->apple9_prev_cs = &cs;
   cmd->vk.dynamic_graphics_state.ia.primitive_topology =
      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   struct agx_draw draw = {.b = agx_3d(3, 1, 1)};
   struct hk_shader fragment = {0};
   static const uint8_t binary[] = {0};
   struct agx_apple9_render_stage stage = {.binary = binary, .binary_size = 1};
   cmd->apple9_draw_state.valid = true;
   cmd->apple9_draw_state.fragment = &fragment;
   cmd->apple9_draw_state.pipeline = (struct agx_apple9_render_pipeline){
      .vertex = &stage, .fragment = &stage, .ppp = 0x100,
      .primitive = AGX_PRIMITIVE_TRIANGLES, .pipeline_word = 0x1000000,
      .vertex_launch = 0x100, .vertex_state_class = 0x404,
   };
   CHECK(hk_apple9_draw_state_reusable(cmd, draw, 0));

   /* Replay an already prepared draw through the real fast path. It must
    * still emit both draws and declare the normal graphics execution scope.
    * No shader compilation or allocator is available in this host fixture. */
   uint8_t *end = expected;
   for (unsigned i = 0; i < 2; ++i) {
      end = agx_apple9_emit_direct_draw(end, &cmd->apple9_draw_state.pipeline,
                                       3, 1, 0) - 4;
      hk_apple9_draw(cmd, 0, draw);
   }
   CHECK((uint8_t *)cs.current - emitted == end - expected);
   CHECK(!memcmp(emitted, expected, end - expected));
   CHECK(cs.stats.cmds == 2 && cs.pass_count == 2 && cs.primitive_count == 2);
   CHECK(util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event) == 1);
   const struct hk_sync_event *event = cmd->sync_events.data;
   CHECK(event->stream == cs.sync_id);
   CHECK(*util_dynarray_element(&cmd->sync_first_event, uint32_t, 0) == 0);
   CHECK(event->dst == (HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT));

#define INVALIDATE(field, value) do { \
   __typeof__(field) saved = field; \
   field = value; CHECK(!hk_apple9_draw_state_reusable(cmd, draw, 0)); \
   field = saved; CHECK(hk_apple9_draw_state_reusable(cmd, draw, 0)); \
} while (0)
   INVALIDATE(cmd->apple9_draw_state.valid, false);
   INVALIDATE(cmd->current_cs.gfx, &other);
   INVALIDATE(cmd->merge_cs, &cs);
   INVALIDATE(cmd->state.gfx.render_gen, 1);
   INVALIDATE(cmd->state.gfx.dirty, HK_DIRTY_VB);
   INVALIDATE(cmd->state.gfx.shaders_dirty, BITFIELD_BIT(MESA_SHADER_VERTEX));
   BITSET_SET(cmd->vk.dynamic_graphics_state.dirty, MESA_VK_DYNAMIC_VP_SCISSORS);
   CHECK(!hk_apple9_draw_state_reusable(cmd, draw, 0));
   vk_dynamic_graphics_state_clear_dirty(&cmd->vk.dynamic_graphics_state);
   INVALIDATE(cmd->state.gfx.render.view_mask, 3);
   INVALIDATE(cmd->state.gfx.xfb_enabled, true);
   INVALIDATE(cmd->state.gfx.occlusion.active_count, 1);
   INVALIDATE(cmd->state.gfx.occlusion.segment_dirty, true);
   INVALIDATE(cmd->state.gfx.occlusion.mode, AGX_VISIBILITY_MODE_BOOLEAN);
   INVALIDATE(cmd->state.gfx.descriptors.root.draw.pipeline_stats_flags,
               VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT);
   INVALIDATE(cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY], (void *)1);
   INVALIDATE(cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL], (void *)1);
   INVALIDATE(cmd->state.gfx.shaders[MESA_SHADER_MESH], (void *)1);
   INVALIDATE(cmd->state.gfx.apple9_fan_unrolled, true);
   INVALIDATE(cmd->in_meta, true);
   INVALIDATE(draw.start, 3);
   INVALIDATE(draw.start_instance, 1);
   INVALIDATE(draw.indexed, true);
   CHECK(!hk_apple9_draw_state_reusable(cmd, draw, 0x1000));
   CHECK(!hk_apple9_draw_state_reusable(cmd, agx_draw_indirect(0x2000), 0));

   /* Index offsets, buffers, sizes, counts and indexed/non-indexed mode
    * feed only command emission. Vary them through the actual cache hit path
    * without any allocator, then compare the complete command stream. */
   for (unsigned i = 0; i < 24; ++i) {
      unsigned bytes = 1u << (i % 3);
      draw = agx_draw_indexed(3 + i, 1 + i % 2, i % 4, 0, 0,
         dev->dev.shader_base + 0x100 + (i % 2) * 32, 128,
         i % 3 == 0 ? AGX_INDEX_SIZE_U8 :
         i % 3 == 1 ? AGX_INDEX_SIZE_U16 : AGX_INDEX_SIZE_U32, false);
      cmd->state.gfx.index.restart = BITFIELD_MASK(bytes * 8);
      CHECK(hk_apple9_draw_state_reusable(cmd, draw, 0));
      struct agx_apple9_render_pipeline pipeline = cmd->apple9_draw_state.pipeline;
      pipeline.index_size = bytes;
      pipeline.index_buffer = draw.index_buffer + draw.start * bytes;
      pipeline.index_extent = MIN2(128 - draw.start * bytes, draw.b.count[0] * bytes);
      pipeline.restart_index = cmd->state.gfx.index.restart;
      end = agx_apple9_emit_direct_draw(end, &pipeline,
         draw.b.count[0], draw.b.count[1], draw.index_bias) - 4;
      hk_apple9_draw(cmd, 0, draw);
   }
   draw = (struct agx_draw){.b = agx_3d(6, 2, 1)};
   CHECK(hk_apple9_draw_state_reusable(cmd, draw, 0));
   struct agx_apple9_render_pipeline pipeline = cmd->apple9_draw_state.pipeline;
   pipeline.index_size = 0;
   end = agx_apple9_emit_direct_draw(end, &pipeline, 6, 2, 0) - 4;
   hk_apple9_draw(cmd, 0, draw);
   CHECK((uint8_t *)cs.current - emitted == end - expected);
   CHECK(!memcmp(emitted, expected, end - expected));
   CHECK(cs.stats.cmds == 27 && cs.pass_count == 27);

   /* Root/push changes update tables and launches in the reuse path; they
    * must not force rebuilding the unchanged raster state. */
   cmd->state.gfx.descriptors.root_dirty = true;
   cmd->state.gfx.descriptors.push_dirty = 1;
   CHECK(hk_apple9_draw_state_reusable(cmd, draw, 0));

   /* A null index binding needs its existing fresh zero-range allocation. */
   draw = agx_draw_indexed(3, 1, 0, 0, 0, 0, 0, AGX_INDEX_SIZE_U16, false);
   CHECK(!hk_apple9_draw_state_reusable(cmd, draw, 0));
   draw.index_buffer_range_B = 16;
   CHECK(hk_apple9_draw_state_reusable(cmd, draw, 0));
   draw.start = 8;
   CHECK(!hk_apple9_draw_state_reusable(cmd, draw, 0));

   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   free(cmd);
   free(dev);
   return 0;
}
