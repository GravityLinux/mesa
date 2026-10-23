/*
 * Copyright 2026 Asahi Linux contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * Debug only (HK_CPU_TIMING=1): time hk's Vulkan entry points on the CPU.
 *
 * At device creation, selected entries of the device dispatch table are
 * replaced with wrappers that time the call and forward to the original.
 * Applications fetch device functions through vkGetDeviceProcAddr, which
 * reads that table, so this also covers runtime (vk_common_*) and WSI
 * entry points.
 *
 * Every wrapped call adds its wall time (CLOCK_MONOTONIC) to its category.
 * Thread CPU time (CLOCK_THREAD_CPUTIME_ID) is read only for submit,
 * present, wait and descriptor-update calls, because a thread-CPU clock read
 * per draw is expensive under FEX. HK_CPU_TIMING=2 reads it for every call.
 * Untimed entry points (dynamic state, queries, copies not listed below)
 * are not counted, so "record" is a lower bound.
 *
 * Once a second, the first present after the interval prints cumulative
 * counters to stderr, or to HK_CPU_TIMING_LOG if set, and device
 * destruction prints a final line:
 *
 *   HK_CPU cpu_ns=<CLOCK_MONOTONIC> frames=N
 *          <cat>_wall_ns=... <cat>_cpu_ns=... <cat>_calls=... (per category)
 *
 * Difference two lines to get per-frame values for a measurement window.
 *
 * Geometry-shader draws are counted at recording time, per poly output shape
 * (dyn = DYNAMIC_INDEXED, idx = STATIC_INDEXED, prim = STATIC_PER_PRIM,
 * inst = STATIC_PER_INSTANCE):
 *
 *   gs_<shape>_draws=...  draws recorded with that shape
 *   gs_<shape>_prims=...  input primitives x instances of the direct draws
 *   gs_<shape>_slots=...  prims x the GS's output index slots (max_indices),
 *                         an upper bound on rasterized vertices
 *   gs_indirect_draws=... indirect GS draws (no counts known on the CPU)
 *
 * Shader compiles are timed where hk compiles, per kind (api = vkCreate*
 * pipelines and shader objects, draw = Apple9 variants resolved while
 * recording a draw):
 *
 *   compile_<kind>_wall_ns=... compile_<kind>_cpu_ns=... compile_<kind>_calls=...
 *   compile_<kind>_max_ns=...  the longest single compile (a stutter bound)
 *
 * Draw sub-steps are timed with wall clock only (see hk_cpu_sub):
 *
 *   sub_<step>_ns=... sub_<step>_n=...
 */

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hk_device.h"
#include "util/u_math.h"
#include "vk_dispatch_table.h"

enum hk_cpu_cat {
   HK_CPU_SUBMIT,
   HK_CPU_PRESENT,
   HK_CPU_WAIT,
   HK_CPU_DESC,
   HK_CPU_DRAW,
   HK_CPU_DISPATCH,
   HK_CPU_RECORD,
   HK_CPU_CAT_COUNT,
};

static const char *hk_cpu_cat_name[HK_CPU_CAT_COUNT] = {
   "submit", "present", "wait", "desc", "draw", "dispatch", "record",
};

/* Per-thread accumulators: wrapped calls add to their own thread's block, with
 * no shared atomics (recording threads contending on one counter cost Dota
 * ~60% of its frame rate). The report sums every block; reads race with
 * writers but each 64-bit counter is read whole. Blocks live until exit. */
struct hk_cpu_block {
   uint64_t wall[HK_CPU_CAT_COUNT];
   uint64_t thread[HK_CPU_CAT_COUNT];
   uint64_t calls[HK_CPU_CAT_COUNT];
   struct hk_cpu_block *next;
};
static struct hk_cpu_block *_Atomic hk_cpu_blocks;
static __thread struct hk_cpu_block *hk_cpu_mine;

static struct hk_cpu_block *
hk_cpu_block(void)
{
   if (likely(hk_cpu_mine))
      return hk_cpu_mine;
   struct hk_cpu_block *b = calloc(1, sizeof(*b));
   if (!b)
      return NULL;
   struct hk_cpu_block *head = atomic_load(&hk_cpu_blocks);
   do {
      b->next = head;
   } while (!atomic_compare_exchange_weak(&hk_cpu_blocks, &head, b));
   return hk_cpu_mine = b;
}

static uint64_t
hk_cpu_sum(size_t offset, unsigned cat)
{
   uint64_t total = 0;
   for (struct hk_cpu_block *b = atomic_load(&hk_cpu_blocks); b; b = b->next)
      total += __atomic_load_n((uint64_t *)((char *)b + offset) + cat,
                               __ATOMIC_RELAXED);
   return total;
}
static _Atomic uint64_t hk_cpu_frames;
static _Atomic uint64_t hk_cpu_last_report;
/* Nonzero when HK_CPU_TIMING is set; read inline by the hooks in hk_device.h. */
int hk_cpu_level;
static _Atomic uint64_t hk_cpu_gs_draws[4], hk_cpu_gs_prims[4], hk_cpu_gs_slots[4];
static _Atomic uint64_t hk_cpu_gs_indirect;
static const char *hk_cpu_gs_shape_name[4] = {"dyn", "idx", "prim", "inst"};
static _Atomic uint64_t hk_cpu_compile_wall[2], hk_cpu_compile_thread[2];
static _Atomic uint64_t hk_cpu_compile_calls[2], hk_cpu_compile_max[2];
static const char *hk_cpu_compile_name[2] = {"api", "draw"};
static _Atomic uint64_t hk_cpu_sub_ns[HK_CPU_SUB_COUNT], hk_cpu_sub_n[HK_CPU_SUB_COUNT];
static const char *hk_cpu_sub_name[HK_CPU_SUB_COUNT] = {
   "resolve", "resolve_slow", "tess", "gs", "robust_index", "a9draw",
   "push_desc", "root", "textures", "tables", "prepare_alloc", "cmd_bo",
   "cmd_bo_destroy_release", "cmd_bo_destroy_cap",
};
/* Heap-using renders per command buffer and per submit: 1, 2, 3, 4+. */
static _Atomic uint64_t hk_cpu_heap_cmd[4], hk_cpu_heap_submit[4];
static _Atomic uint64_t hk_cpu_feedback[4];
static FILE *hk_cpu_out;

static inline uint64_t
hk_cpu_clock(clockid_t id)
{
   struct timespec ts;
   clock_gettime(id, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

struct hk_cpu_scope {
   enum hk_cpu_cat cat;
   bool thread;
   uint64_t wall, cpu;
};

static inline void
hk_cpu_begin(struct hk_cpu_scope *s, enum hk_cpu_cat cat)
{
   s->cat = cat;
   /* Thread CPU time is a syscall in some guests: at level 1 only for the
    * low-rate submit, present and wait calls. */
   s->thread = hk_cpu_level >= 2 || cat < HK_CPU_DESC;
   s->cpu = s->thread ? hk_cpu_clock(CLOCK_THREAD_CPUTIME_ID) : 0;
   s->wall = hk_cpu_clock(CLOCK_MONOTONIC);
}

static inline void
hk_cpu_end(struct hk_cpu_scope *s)
{
   struct hk_cpu_block *b = hk_cpu_block();
   if (!b)
      return;
   const uint64_t wall = hk_cpu_clock(CLOCK_MONOTONIC) - s->wall;
   __atomic_store_n(&b->wall[s->cat], b->wall[s->cat] + wall, __ATOMIC_RELAXED);
   if (s->thread) {
      const uint64_t cpu = hk_cpu_clock(CLOCK_THREAD_CPUTIME_ID) - s->cpu;
      __atomic_store_n(&b->thread[s->cat], b->thread[s->cat] + cpu,
                       __ATOMIC_RELAXED);
   }
   __atomic_store_n(&b->calls[s->cat], b->calls[s->cat] + 1, __ATOMIC_RELAXED);
}

static void
hk_cpu_report(bool force)
{
   uint64_t now = hk_cpu_clock(CLOCK_MONOTONIC);
   uint64_t last = atomic_load(&hk_cpu_last_report);
   if (!force && (now - last < 1000000000ull ||
                  !atomic_compare_exchange_strong(&hk_cpu_last_report, &last,
                                                  now)))
      return;

   char line[4096];
   int n = snprintf(line, sizeof(line), "HK_CPU cpu_ns=%" PRIu64 " frames=%" PRIu64,
                    now, atomic_load(&hk_cpu_frames));
   for (unsigned c = 0; c < HK_CPU_CAT_COUNT && n < (int)sizeof(line); ++c) {
      n += snprintf(line + n, sizeof(line) - n,
                    " %s_wall_ns=%" PRIu64 " %s_cpu_ns=%" PRIu64 " %s_calls=%" PRIu64,
                    hk_cpu_cat_name[c],
                    hk_cpu_sum(offsetof(struct hk_cpu_block, wall), c),
                    hk_cpu_cat_name[c],
                    hk_cpu_sum(offsetof(struct hk_cpu_block, thread), c),
                    hk_cpu_cat_name[c],
                    hk_cpu_sum(offsetof(struct hk_cpu_block, calls), c));
   }
   for (unsigned i = 0; i < 4 && n < (int)sizeof(line); ++i) {
      n += snprintf(line + n, sizeof(line) - n,
                    " gs_%s_draws=%" PRIu64 " gs_%s_prims=%" PRIu64 " gs_%s_slots=%" PRIu64,
                    hk_cpu_gs_shape_name[i], atomic_load(&hk_cpu_gs_draws[i]),
                    hk_cpu_gs_shape_name[i], atomic_load(&hk_cpu_gs_prims[i]),
                    hk_cpu_gs_shape_name[i], atomic_load(&hk_cpu_gs_slots[i]));
   }
   if (n < (int)sizeof(line)) {
      n += snprintf(line + n, sizeof(line) - n, " gs_indirect_draws=%" PRIu64,
                    atomic_load(&hk_cpu_gs_indirect));
   }
   for (unsigned i = 0; i < HK_CPU_SUB_COUNT && n < (int)sizeof(line); ++i) {
      n += snprintf(line + n, sizeof(line) - n,
                    " sub_%s_ns=%" PRIu64 " sub_%s_n=%" PRIu64,
                    hk_cpu_sub_name[i], atomic_load(&hk_cpu_sub_ns[i]),
                    hk_cpu_sub_name[i], atomic_load(&hk_cpu_sub_n[i]));
   }
   for (unsigned i = 0; i < 2 && n < (int)sizeof(line); ++i) {
      const char *k = hk_cpu_compile_name[i];
      n += snprintf(line + n, sizeof(line) - n,
                    " compile_%s_wall_ns=%" PRIu64 " compile_%s_cpu_ns=%" PRIu64
                    " compile_%s_calls=%" PRIu64 " compile_%s_max_ns=%" PRIu64,
                    k, atomic_load(&hk_cpu_compile_wall[i]),
                    k, atomic_load(&hk_cpu_compile_thread[i]),
                    k, atomic_load(&hk_cpu_compile_calls[i]),
                    k, atomic_load(&hk_cpu_compile_max[i]));
   }
   for (unsigned i = 0; i < 4 && n < (int)sizeof(line); ++i) {
      n += snprintf(line + n, sizeof(line) - n,
                    " heap_cmd_%u=%" PRIu64 " heap_submit_%u=%" PRIu64, i + 1,
                    atomic_load(&hk_cpu_heap_cmd[i]), i + 1,
                    atomic_load(&hk_cpu_heap_submit[i]));
   }
   fprintf(hk_cpu_out, "HK_FEEDBACK cpu_ns=%" PRIu64 " frames=%" PRIu64
           " calls=%" PRIu64 " bytes=%" PRIu64 " hits=%" PRIu64
           " misses=%" PRIu64 "\n", now, atomic_load(&hk_cpu_frames),
           atomic_load(&hk_cpu_feedback[0]), atomic_load(&hk_cpu_feedback[1]),
           atomic_load(&hk_cpu_feedback[2]), atomic_load(&hk_cpu_feedback[3]));
   fprintf(hk_cpu_out, "%s\n", line);
   fflush(hk_cpu_out);
}

void
hk_cpu_timing_note_heap(unsigned renders, bool submit)
{
   if (!renders)
      return;
   atomic_fetch_add_explicit(
      &(submit ? hk_cpu_heap_submit : hk_cpu_heap_cmd)[MIN2(renders, 4) - 1], 1,
      memory_order_relaxed);
}

void
hk_cpu_timing_note_feedback(const struct hk_feedback_stats *stats)
{
   atomic_fetch_add_explicit(&hk_cpu_feedback[0], stats->calls, memory_order_relaxed);
   atomic_fetch_add_explicit(&hk_cpu_feedback[1], stats->bytes, memory_order_relaxed);
   atomic_fetch_add_explicit(&hk_cpu_feedback[2], stats->hits, memory_order_relaxed);
   atomic_fetch_add_explicit(&hk_cpu_feedback[3], stats->misses, memory_order_relaxed);
}

#define HK_WRAP_VOID(cat, name, PARAMS, ARGS)                                  \
   static PFN_vk##name hk_cpu_orig_##name;                                     \
   static VKAPI_ATTR void VKAPI_CALL hk_cpu_##name PARAMS                      \
   {                                                                           \
      struct hk_cpu_scope hk_scope_;                                                   \
      hk_cpu_begin(&hk_scope_, cat);                                                   \
      hk_cpu_orig_##name ARGS;                                                 \
      hk_cpu_end(&hk_scope_);                                                          \
   }

#define HK_WRAP_RESULT(cat, name, PARAMS, ARGS)                                \
   static PFN_vk##name hk_cpu_orig_##name;                                     \
   static VKAPI_ATTR VkResult VKAPI_CALL hk_cpu_##name PARAMS                  \
   {                                                                           \
      struct hk_cpu_scope hk_scope_;                                                   \
      hk_cpu_begin(&hk_scope_, cat);                                                   \
      VkResult r = hk_cpu_orig_##name ARGS;                                    \
      hk_cpu_end(&hk_scope_);                                                          \
      return r;                                                                \
   }

/* Submission */
HK_WRAP_RESULT(HK_CPU_SUBMIT, QueueSubmit,
               (VkQueue q, uint32_t n, const VkSubmitInfo *s, VkFence f),
               (q, n, s, f))
HK_WRAP_RESULT(HK_CPU_SUBMIT, QueueSubmit2,
               (VkQueue q, uint32_t n, const VkSubmitInfo2 *s, VkFence f),
               (q, n, s, f))

/* Waits */
HK_WRAP_RESULT(HK_CPU_WAIT, WaitForFences,
               (VkDevice d, uint32_t n, const VkFence *f, VkBool32 all,
                uint64_t t),
               (d, n, f, all, t))
HK_WRAP_RESULT(HK_CPU_WAIT, WaitSemaphores,
               (VkDevice d, const VkSemaphoreWaitInfo *i, uint64_t t),
               (d, i, t))
HK_WRAP_RESULT(HK_CPU_WAIT, QueueWaitIdle, (VkQueue q), (q))
HK_WRAP_RESULT(HK_CPU_WAIT, DeviceWaitIdle, (VkDevice d), (d))

/* Descriptors */
HK_WRAP_VOID(HK_CPU_DESC, UpdateDescriptorSets,
             (VkDevice d, uint32_t nw, const VkWriteDescriptorSet *w,
              uint32_t nc, const VkCopyDescriptorSet *c),
             (d, nw, w, nc, c))
HK_WRAP_VOID(HK_CPU_DESC, UpdateDescriptorSetWithTemplate,
             (VkDevice d, VkDescriptorSet set, VkDescriptorUpdateTemplate t,
              const void *data),
             (d, set, t, data))
HK_WRAP_VOID(HK_CPU_DESC, CmdPushDescriptorSet,
             (VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
              uint32_t set, uint32_t n, const VkWriteDescriptorSet *w),
             (cb, bp, l, set, n, w))
HK_WRAP_VOID(HK_CPU_DESC, CmdBindDescriptorSets,
             (VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
              uint32_t first, uint32_t n, const VkDescriptorSet *sets,
              uint32_t nd, const uint32_t *dyn),
             (cb, bp, l, first, n, sets, nd, dyn))

/* Draws */
HK_WRAP_VOID(HK_CPU_DRAW, CmdDraw,
             (VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv,
              uint32_t fi),
             (cb, vc, ic, fv, fi))
HK_WRAP_VOID(HK_CPU_DRAW, CmdDrawIndexed,
             (VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi,
              int32_t vo, uint32_t finst),
             (cb, ic, inst, fi, vo, finst))
HK_WRAP_VOID(HK_CPU_DRAW, CmdDrawIndirect,
             (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n,
              uint32_t stride),
             (cb, b, o, n, stride))
HK_WRAP_VOID(HK_CPU_DRAW, CmdDrawIndexedIndirect,
             (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n,
              uint32_t stride),
             (cb, b, o, n, stride))
HK_WRAP_VOID(HK_CPU_DRAW, CmdDrawIndirectCount,
             (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
              VkDeviceSize co, uint32_t max, uint32_t stride),
             (cb, b, o, cbuf, co, max, stride))
HK_WRAP_VOID(HK_CPU_DRAW, CmdDrawIndexedIndirectCount,
             (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
              VkDeviceSize co, uint32_t max, uint32_t stride),
             (cb, b, o, cbuf, co, max, stride))

/* Dispatches */
HK_WRAP_VOID(HK_CPU_DISPATCH, CmdDispatch,
             (VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z),
             (cb, x, y, z))
HK_WRAP_VOID(HK_CPU_DISPATCH, CmdDispatchBase,
             (VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz,
              uint32_t x, uint32_t y, uint32_t z),
             (cb, bx, by, bz, x, y, z))
HK_WRAP_VOID(HK_CPU_DISPATCH, CmdDispatchIndirect,
             (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o), (cb, b, o))

/* Other recording */
HK_WRAP_RESULT(HK_CPU_RECORD, BeginCommandBuffer,
               (VkCommandBuffer cb, const VkCommandBufferBeginInfo *i),
               (cb, i))
HK_WRAP_RESULT(HK_CPU_RECORD, EndCommandBuffer, (VkCommandBuffer cb), (cb))
HK_WRAP_VOID(HK_CPU_RECORD, CmdBeginRendering,
             (VkCommandBuffer cb, const VkRenderingInfo *i), (cb, i))
HK_WRAP_VOID(HK_CPU_RECORD, CmdEndRendering, (VkCommandBuffer cb), (cb))
HK_WRAP_VOID(HK_CPU_RECORD, CmdPipelineBarrier2,
             (VkCommandBuffer cb, const VkDependencyInfo *i), (cb, i))
HK_WRAP_VOID(HK_CPU_RECORD, CmdBindPipeline,
             (VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p),
             (cb, bp, p))
HK_WRAP_VOID(HK_CPU_RECORD, CmdPushConstants,
             (VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st,
              uint32_t o, uint32_t sz, const void *v),
             (cb, l, st, o, sz, v))
HK_WRAP_VOID(HK_CPU_RECORD, CmdBindVertexBuffers2,
             (VkCommandBuffer cb, uint32_t first, uint32_t n,
              const VkBuffer *b, const VkDeviceSize *o, const VkDeviceSize *sz,
              const VkDeviceSize *st),
             (cb, first, n, b, o, sz, st))
HK_WRAP_VOID(HK_CPU_RECORD, CmdBindIndexBuffer,
             (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t),
             (cb, b, o, t))

/* Present: also counts frames and reports. */
static PFN_vkQueuePresentKHR hk_cpu_orig_QueuePresentKHR;
static VKAPI_ATTR VkResult VKAPI_CALL
hk_cpu_QueuePresentKHR(VkQueue q, const VkPresentInfoKHR *i)
{
   struct hk_cpu_scope s;
   hk_cpu_begin(&s, HK_CPU_PRESENT);
   VkResult r = hk_cpu_orig_QueuePresentKHR(q, i);
   hk_cpu_end(&s);
   atomic_fetch_add_explicit(&hk_cpu_frames, 1, memory_order_relaxed);
   hk_cpu_report(false);
   return r;
}

/* Device destruction prints the final counters. */
static PFN_vkDestroyDevice hk_cpu_orig_DestroyDevice;
static VKAPI_ATTR void VKAPI_CALL
hk_cpu_DestroyDevice(VkDevice d, const VkAllocationCallbacks *a)
{
   hk_cpu_report(true);
   hk_cpu_orig_DestroyDevice(d, a);
}

#define HK_INSTALL(name)                                                       \
   do {                                                                        \
      if (t->name && t->name != hk_cpu_##name) {                               \
         hk_cpu_orig_##name = t->name;                                         \
         t->name = hk_cpu_##name;                                              \
      }                                                                        \
   } while (0)

void
hk_cpu_timing_install(struct vk_device_dispatch_table *t)
{
   const char *level = getenv("HK_CPU_TIMING");
   if (!level || !atoi(level))
      return;
   hk_cpu_level = atoi(level);

   if (!hk_cpu_out) {
      const char *path = getenv("HK_CPU_TIMING_LOG");
      hk_cpu_out = path ? fopen(path, "a") : NULL;
      if (!hk_cpu_out)
         hk_cpu_out = stderr;
      atomic_store(&hk_cpu_last_report, hk_cpu_clock(CLOCK_MONOTONIC));
   }

   HK_INSTALL(QueueSubmit);
   HK_INSTALL(QueueSubmit2);
   HK_INSTALL(QueuePresentKHR);
   HK_INSTALL(DestroyDevice);
   HK_INSTALL(WaitForFences);
   HK_INSTALL(WaitSemaphores);
   HK_INSTALL(QueueWaitIdle);
   HK_INSTALL(DeviceWaitIdle);
   HK_INSTALL(UpdateDescriptorSets);
   HK_INSTALL(UpdateDescriptorSetWithTemplate);
   HK_INSTALL(CmdPushDescriptorSet);
   HK_INSTALL(CmdBindDescriptorSets);
   HK_INSTALL(CmdDraw);
   HK_INSTALL(CmdDrawIndexed);
   HK_INSTALL(CmdDrawIndirect);
   HK_INSTALL(CmdDrawIndexedIndirect);
   HK_INSTALL(CmdDrawIndirectCount);
   HK_INSTALL(CmdDrawIndexedIndirectCount);
   HK_INSTALL(CmdDispatch);
   HK_INSTALL(CmdDispatchBase);
   HK_INSTALL(CmdDispatchIndirect);
   HK_INSTALL(BeginCommandBuffer);
   HK_INSTALL(EndCommandBuffer);
   HK_INSTALL(CmdBeginRendering);
   HK_INSTALL(CmdEndRendering);
   HK_INSTALL(CmdPipelineBarrier2);
   HK_INSTALL(CmdBindPipeline);
   HK_INSTALL(CmdPushConstants);
   HK_INSTALL(CmdBindVertexBuffers2);
   HK_INSTALL(CmdBindIndexBuffer);
}

void
hk_cpu_timing_note_gs_slow(unsigned shape, bool indirect, uint64_t prims,
                           unsigned max_indices)
{
   if (shape >= 4)
      return;

   atomic_fetch_add_explicit(&hk_cpu_gs_draws[shape], 1, memory_order_relaxed);
   if (indirect) {
      atomic_fetch_add_explicit(&hk_cpu_gs_indirect, 1, memory_order_relaxed);
      return;
   }
   atomic_fetch_add_explicit(&hk_cpu_gs_prims[shape], prims,
                             memory_order_relaxed);
   atomic_fetch_add_explicit(&hk_cpu_gs_slots[shape], prims * max_indices,
                             memory_order_relaxed);
}

void
hk_cpu_timing_compile_begin(struct hk_cpu_compile_scope *s)
{
   s->active = hk_cpu_level != 0;
   if (!s->active)
      return;
   s->cpu = hk_cpu_clock(CLOCK_THREAD_CPUTIME_ID);
   s->wall = hk_cpu_clock(CLOCK_MONOTONIC);
}

void
hk_cpu_timing_compile_end(const struct hk_cpu_compile_scope *s, bool draw_time)
{
   if (!s->active)
      return;

   const unsigned k = draw_time;
   const uint64_t wall = hk_cpu_clock(CLOCK_MONOTONIC) - s->wall;
   atomic_fetch_add_explicit(&hk_cpu_compile_wall[k], wall, memory_order_relaxed);
   atomic_fetch_add_explicit(&hk_cpu_compile_thread[k],
                             hk_cpu_clock(CLOCK_THREAD_CPUTIME_ID) - s->cpu,
                             memory_order_relaxed);
   atomic_fetch_add_explicit(&hk_cpu_compile_calls[k], 1, memory_order_relaxed);
   uint64_t max = atomic_load_explicit(&hk_cpu_compile_max[k], memory_order_relaxed);
   while (wall > max &&
          !atomic_compare_exchange_weak(&hk_cpu_compile_max[k], &max, wall))
      ;
}

uint64_t
hk_cpu_sub_begin_slow(void)
{
   return hk_cpu_clock(CLOCK_MONOTONIC);
}

void
hk_cpu_sub_end_slow(uint64_t start, enum hk_cpu_sub sub)
{
   atomic_fetch_add_explicit(&hk_cpu_sub_ns[sub],
                             hk_cpu_clock(CLOCK_MONOTONIC) - start,
                             memory_order_relaxed);
   atomic_fetch_add_explicit(&hk_cpu_sub_n[sub], 1, memory_order_relaxed);
}

void
hk_cpu_sub_count_slow(enum hk_cpu_sub sub)
{
   atomic_fetch_add_explicit(&hk_cpu_sub_n[sub], 1, memory_order_relaxed);
}
