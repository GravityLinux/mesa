/* SPDX-License-Identifier: MIT */
#include "hk_sync.h"

unsigned
hk_sync_ordered_stages(unsigned src, unsigned dst,
                       VkAccessFlags2 src_access, VkAccessFlags2 dst_access)
{
   /* Execution-only dependencies still order both scopes. A one-sided scope
    * has no execution dependency, but may request availability or visibility.
    */
   if (src && dst)
      return src | dst;
   return (src_access ? src : 0) | (dst_access ? dst : 0);
}

/* Follow merged streams to the stream that carries their work. */
static uint32_t
hk_sync_remap(const uint32_t *remap, unsigned stream_count, uint32_t id)
{
   while (id != HK_SYNC_NONE) {
      assert(id < stream_count);
      uint32_t parent = remap[id];
      if (parent == id)
         break;
      assert(parent == HK_SYNC_NONE || parent < id);
      id = parent;
   }
   return id;
}

bool
hk_sync_resolve(struct hk_sync_state *state,
                const struct hk_sync_event *events, unsigned event_count,
                const uint32_t *remap, struct hk_sync_stream *streams,
                unsigned stream_count, bool fallback)
{
   struct hk_sync_state incoming = *state;
   struct hk_sync_state group_source = *state;
   uint32_t group = 0;
   if (!fallback) {
      for (unsigned i = 0; i < event_count; ++i) {
         const struct hk_sync_event *ev = &events[i];
         if (ev->stream == HK_SYNC_NONE) {
            if (ev->barrier_group == 0) {
               hk_sync_barrier(state, ev->src, ev->dst);
               group = 0;
            } else {
               /* Barriers in one VkDependencyInfo do not form chains with
                * each other. Snapshot incoming scopes, but include any helper
                * work emitted while closing their producer streams.
                */
               if (group != ev->barrier_group) {
                  group_source = *state;
                  group = ev->barrier_group;
               }
               for (unsigned s = 0; s < HK_SYNC_STAGE_COUNT; ++s)
                  group_source.frontier[s] = state->frontier[s];
               struct hk_sync_state scoped = group_source;
               hk_sync_barrier(&scoped, ev->src, ev->dst);
               for (unsigned s = 0; s < HK_SYNC_STAGE_COUNT; ++s) {
                  if (ev->dst & (1u << s))
                     hk_sync_join(&state->pending[s], scoped.pending[s]);
               }
            }
            continue;
         }

         uint32_t id = hk_sync_remap(remap, stream_count, ev->stream);
         if (id == HK_SYNC_NONE)
            continue;

         struct hk_sync_stream *s = &streams[id];
         /* Unclassified helpers keep their implicit dependencies, but do not
          * force full ordering for unrelated operations in the same buffer.
          */
         if (ev->dst == HK_SYNC_ALL)
            hk_sync_barrier(state, HK_SYNC_ALL, HK_SYNC_ALL);
         if (s->engine == HK_SYNC_RENDER) {
            hk_sync_join(&s->wait_tiling,
                         hk_sync_pending(state, ev->dst & HK_SYNC_TILING,
                                         s->engine, s->index));
         }
         struct hk_sync_dep dep = hk_sync_work(state, ev->dst, s->engine, s->index);
         hk_sync_join(&s->wait, dep);
         if (ev->after1) {
            uint32_t a = hk_sync_remap(remap, stream_count, ev->after1 - 1);
            if (a != HK_SYNC_NONE && a != id && streams[a].index) {
               struct hk_sync_dep after = hk_sync_empty();
               after.engine[streams[a].engine] = streams[a].index;
               hk_sync_join(&s->wait, after);
               if (s->engine == HK_SYNC_RENDER)
                  hk_sync_join(&s->wait_tiling, after);
            }
         }
         if (ev->dst == HK_SYNC_ALL)
            hk_sync_barrier(state, HK_SYNC_ALL, HK_SYNC_ALL);

         for (unsigned e = 0; e < HK_SYNC_ENGINE_COUNT; ++e) {
            if (s->wait.engine[e] != HK_SYNC_NONE &&
                s->wait.engine[e] > s->before[e])
               fallback = true;
         }
      }
   }

   if (fallback) {
      *state = incoming;
      for (unsigned i = 0; i < stream_count; ++i) {
         struct hk_sync_stream *s = &streams[i];
         if (!s->index)
            continue;
         s->wait = (struct hk_sync_dep){{s->before[0], s->before[1]}};
         s->wait_tiling = s->wait;
         hk_sync_work(state, HK_SYNC_ALL, s->engine, s->index);
      }
      hk_sync_barrier(state, HK_SYNC_ALL, HK_SYNC_ALL);
   }

   return !fallback;
}

unsigned
hk_sync_stage_mask(VkPipelineStageFlags2 stages, bool source)
{
   if (stages & (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
                 (source ? VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT :
                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)))
      return HK_SYNC_ALL;

   stages &= ~(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
               VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

   const VkPipelineStageFlags2 pre_raster =
      VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
      VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
      VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
      VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
      VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
      VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
      VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
      VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
      VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT;
   const VkPipelineStageFlags2 fragment =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
   const VkPipelineStageFlags2 graphics =
      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | pre_raster | fragment;
   const VkPipelineStageFlags2 transfer =
      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
      VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT |
      VK_PIPELINE_STAGE_2_RESOLVE_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
   const VkPipelineStageFlags2 compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
   const VkPipelineStageFlags2 indirect = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;

   /* Host, extensions without a classified implementation, and future stages
    * retain full ordering. NONE remains an empty execution scope.
    */
   if (stages & ~(graphics | transfer | compute | indirect))
      return HK_SYNC_ALL;

   unsigned mask = 0;
   if (stages & (VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | pre_raster))
      mask |= HK_SYNC_PRE_RASTER;
   if (stages & (VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | fragment))
      mask |= HK_SYNC_FRAGMENT;
   if (stages & compute)
      mask |= HK_SYNC_COMPUTE;
   if (stages & transfer)
      mask |= HK_SYNC_TRANSFER;
   if (stages & indirect)
      mask |= HK_SYNC_INDIRECT;
   if (stages & VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
      mask |= HK_SYNC_DRAW_INDIRECT;

   /* Source scopes include logically earlier stages; destination scopes
    * include logically later ones. A render command completes both of its
    * stages together, and only render dependencies can be deferred to the
    * fragment stage: compute -> fragment still waits before tiling.
    */
   if (source && (mask & HK_SYNC_FRAGMENT))
      mask |= HK_SYNC_PRE_RASTER;
   if (!source && (mask & HK_SYNC_PRE_RASTER))
      mask |= HK_SYNC_FRAGMENT;
   if (source && (mask & HK_SYNC_GRAPHICS))
      mask |= HK_SYNC_DRAW_INDIRECT;
   if (source && (mask & HK_SYNC_COMPUTE))
      mask |= HK_SYNC_DISPATCH_INDIRECT;
   if (!source && (mask & HK_SYNC_DRAW_INDIRECT))
      mask |= HK_SYNC_GRAPHICS;
   if (!source && (mask & HK_SYNC_DISPATCH_INDIRECT))
      mask |= HK_SYNC_COMPUTE;
   return mask;
}
