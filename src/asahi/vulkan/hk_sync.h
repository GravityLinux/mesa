/* SPDX-License-Identifier: MIT */
#ifndef HK_SYNC_H
#define HK_SYNC_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#ifdef __cplusplus
extern "C" {
#endif

unsigned hk_sync_stage_mask(VkPipelineStageFlags2 stages, bool source);
unsigned hk_sync_ordered_stages(unsigned src, unsigned dst,
                                VkAccessFlags2 src_access,
                                VkAccessFlags2 dst_access);

/* Logical execution domains, independent of the engine used to execute them.
 * A render command runs its pre-rasterization stages (tiling) and fragment
 * stages as separate firmware stages, so the two are tracked separately.
 * Transfers and indirect reads can run on either engine.
 */
enum hk_sync_stage {
   HK_SYNC_PRE_RASTER = 1 << 0,
   HK_SYNC_COMPUTE = 1 << 1,
   HK_SYNC_TRANSFER = 1 << 2,
   HK_SYNC_DRAW_INDIRECT = 1 << 3,
   HK_SYNC_DISPATCH_INDIRECT = 1 << 4,
   HK_SYNC_FRAGMENT = 1 << 5,
   HK_SYNC_GRAPHICS = HK_SYNC_PRE_RASTER | HK_SYNC_FRAGMENT,
   HK_SYNC_INDIRECT = HK_SYNC_DRAW_INDIRECT | HK_SYNC_DISPATCH_INDIRECT,
   /* Domains a render command's tiling stage executes or reads. */
   HK_SYNC_TILING = HK_SYNC_PRE_RASTER | HK_SYNC_DRAW_INDIRECT,
   HK_SYNC_ALL = (1 << 6) - 1,
   HK_SYNC_STAGE_COUNT = 6,
};

enum hk_sync_engine {
   HK_SYNC_RENDER,
   HK_SYNC_CDM,
   HK_SYNC_ENGINE_COUNT,
};

/* Keep indices wide until ioctl slicing. Zero means work from older submits;
 * NONE means no dependency, and must never participate in max().
 */
#define HK_SYNC_NONE UINT32_MAX

struct hk_sync_dep {
   uint32_t engine[HK_SYNC_ENGINE_COUNT];
};

struct hk_sync_state {
   struct hk_sync_dep frontier[HK_SYNC_STAGE_COUNT];
   struct hk_sync_dep pending[HK_SYNC_STAGE_COUNT];
};

struct hk_sync_event {
   /* HK_SYNC_NONE denotes a barrier; otherwise this is a stable stream ID. */
   uint32_t stream;
   unsigned src, dst;
   uint32_t barrier_group;
};

struct hk_sync_stream {
   struct hk_sync_dep wait;
   /* The subset of wait needed before a render command's tiling stage. The
    * rest may be deferred to its fragment stage.
    */
   struct hk_sync_dep wait_tiling;
   uint32_t before[HK_SYNC_ENGINE_COUNT];
   uint32_t index;
   enum hk_sync_engine engine;
};

bool hk_sync_resolve(struct hk_sync_state *state,
                     const struct hk_sync_event *events, unsigned event_count,
                     const uint32_t *remap, struct hk_sync_stream *streams,
                     unsigned stream_count, bool fallback);

static inline struct hk_sync_dep
hk_sync_empty(void)
{
   return (struct hk_sync_dep){{HK_SYNC_NONE, HK_SYNC_NONE}};
}

static inline void
hk_sync_join(struct hk_sync_dep *dst, struct hk_sync_dep src)
{
   for (unsigned e = 0; e < HK_SYNC_ENGINE_COUNT; ++e) {
      if (src.engine[e] != HK_SYNC_NONE &&
          (dst->engine[e] == HK_SYNC_NONE || dst->engine[e] < src.engine[e]))
         dst->engine[e] = src.engine[e];
   }
}

static inline void
hk_sync_init(struct hk_sync_state *state)
{
   for (unsigned s = 0; s < HK_SYNC_STAGE_COUNT; ++s) {
      state->frontier[s] = hk_sync_empty();
      state->pending[s] = hk_sync_empty();
   }
}

/* Each barrier is processed separately: unioning all source and destination
 * masks first invents dependencies between unrelated barrier pairs. Pending
 * scopes participate in the source so chains also work without intervening
 * action commands (including across empty command buffers).
 */
static inline void
hk_sync_barrier(struct hk_sync_state *state, unsigned src, unsigned dst)
{
   struct hk_sync_dep dep = hk_sync_empty();
   for (unsigned s = 0; s < HK_SYNC_STAGE_COUNT; ++s) {
      if (src & (1u << s)) {
         hk_sync_join(&dep, state->frontier[s]);
         hk_sync_join(&dep, state->pending[s]);
      }
   }

   for (unsigned s = 0; s < HK_SYNC_STAGE_COUNT; ++s) {
      if (dst & (1u << s))
         hk_sync_join(&state->pending[s], dep);
   }
}

/* Dependency of work in the given stages, without recording that work. */
static inline struct hk_sync_dep
hk_sync_pending(const struct hk_sync_state *state, unsigned stages,
                enum hk_sync_engine engine, uint32_t index)
{
   struct hk_sync_dep dep = hk_sync_empty();
   for (unsigned s = 0; s < HK_SYNC_STAGE_COUNT; ++s) {
      if (stages & (1u << s))
         hk_sync_join(&dep, state->pending[s]);
   }
   if (dep.engine[engine] == index)
      dep.engine[engine] = HK_SYNC_NONE;
   return dep;
}

static inline struct hk_sync_dep
hk_sync_work(struct hk_sync_state *state, unsigned stages,
             enum hk_sync_engine engine, uint32_t index)
{
   assert(index && index != HK_SYNC_NONE);
   struct hk_sync_dep dep = hk_sync_empty();
   for (unsigned s = 0; s < HK_SYNC_STAGE_COUNT; ++s) {
      if (stages & (1u << s)) {
         hk_sync_join(&dep, state->pending[s]);
         /* Subqueue order makes the latest producer cover older producers on
          * that engine, but says nothing about the other engine.
          */
         uint32_t *last = &state->frontier[s].engine[engine];
         if (*last == HK_SYNC_NONE || *last < index)
            *last = index;
      }
   }

   /* Multiple action commands can share a stream. Barriers requiring a real
    * dependency within that stream must split it or emit a CDM barrier.
    */
   if (dep.engine[engine] == index)
      dep.engine[engine] = HK_SYNC_NONE;
   return dep;
}

static inline uint32_t
hk_sync_rebase(uint32_t index, uint32_t submitted)
{
   if (index == HK_SYNC_NONE)
      return HK_SYNC_NONE;
   return index > submitted ? index - submitted : 0;
}

/* A later ioctl can only name the latest completion of an older subqueue. */
static inline void
hk_sync_next_submit(struct hk_sync_state *state)
{
   for (unsigned s = 0; s < HK_SYNC_STAGE_COUNT; ++s) {
      for (unsigned e = 0; e < HK_SYNC_ENGINE_COUNT; ++e) {
         if (state->frontier[s].engine[e] != HK_SYNC_NONE)
            state->frontier[s].engine[e] = 0;
         if (state->pending[s].engine[e] != HK_SYNC_NONE)
            state->pending[s].engine[e] = 0;
      }
   }
}

#ifdef __cplusplus
}
#endif

#endif
