/* SPDX-License-Identifier: MIT
 * Diagnostic build only. Firmware timestamp pairs provide aggregate interval
 * unions, without retaining a command trace or changing application queries.
 */
#include "hk_gpu_busy.h"
#include "git_sha1.h"
#ifdef HK_GPU_BUSY_COUNTERS
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include "util/os_time.h"
#include "util/u_wait_watchdog.h"
#include <sys/stat.h>
#include "hk_device.h"
#include "hk_gpu_busy_intervals.h"
#include "hk_queue.h"

#define BUSY_SLOTS   8192
#define BUSY_SECONDS 64
#define SECOND_NS    UINT64_C(1000000000)
#define VTX          1
#define FRAG         2
#define COMPUTE      3

struct busy_slot {
   uint64_t point, submit_cpu_ns;
   unsigned engines;
};

struct busy_bin {
   uint64_t second, revision, pairs;
   bool valid, dirty;
   /* 0 is the union of all engines, then vertex/fragment/compute. */
   struct hk_busy_ranges ranges[4];
};

struct busy_device;
struct hk_gpu_busy_queue {
   struct hk_gpu_busy_queue *next;
   struct busy_device *device;
   struct hk_queue *queue;
   struct agx_bo *bo;
   volatile uint64_t *map;
   uint32_t handle;
   uint64_t submit_cpu_ns;
   struct busy_slot slots[BUSY_SLOTS];
   unsigned free_slots[BUSY_SLOTS], free_count;
};

struct busy_device {
   struct busy_device *next;
   struct hk_device *dev;
   struct hk_gpu_busy_queue *queues;
   FILE *file;
   uint64_t next_sample_ns;
   struct busy_bin bins[BUSY_SECONDS];
   uint64_t pairs_requested, pairs_written, pairs_collected;
   uint64_t app_pairs, full_pairs, initial_pairs, invalid_pairs, late_pairs;
   uint64_t queries, query_errors;
};

static pthread_mutex_t busy_lock = PTHREAD_MUTEX_INITIALIZER;
static struct busy_device *devices;

static FILE *
open_log(void)
{
   char path[4096];
   const char *setting = getenv("HK_GPU_BUSY_FILE");
   if (setting && *setting) {
      const char *pid = strstr(setting, "%p");
      if (pid)
         snprintf(path, sizeof(path), "%.*s%d%s", (int)(pid - setting), setting,
                  getpid(), pid + 2);
      else
         snprintf(path, sizeof(path), "%s", setting);
   } else {
      snprintf(path, sizeof(path), "/tmp/hk-gpu-busy-%d.log", getpid());
   }
   int fd =
      open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW, 0600);
   if (fd < 0)
      return NULL;
   struct stat st;
   if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != geteuid()) {
      close(fd);
      return NULL;
   }
   FILE *f = fdopen(fd, "a");
   if (!f)
      close(fd);
   return f;
}

static void
add_interval(struct busy_device *d, unsigned engine, uint64_t start,
             uint64_t end)
{
   if (!start || end <= start || end - start > BUSY_SECONDS * SECOND_NS) {
      d->invalid_pairs++;
      return;
   }
   d->pairs_collected++;
   while (start < end) {
      uint64_t second = start / SECOND_NS;
      uint64_t stop = (second + 1) * SECOND_NS;
      if (stop > end)
         stop = end;
      struct busy_bin *b = &d->bins[second % BUSY_SECONDS];
      if (b->valid && b->second > second) {
         d->late_pairs++;
         return;
      }
      if (!b->valid || b->second != second) {
         /* If completion delivery spans more than the bounded retention
          * window, report omitted dirty history instead of silently losing it. */
         if (b->valid && b->dirty)
            d->late_pairs += b->pairs;
         for (unsigned i = 0; i < 4; ++i)
            hk_busy_union_clear(&b->ranges[i]);
         b->valid = true;
         b->second = second;
         b->revision = b->pairs = 0;
      }
      uint64_t base = second * SECOND_NS;
      hk_busy_union_add(&b->ranges[engine], start - base, stop - base);
      hk_busy_union_add(&b->ranges[0], start - base, stop - base);
      b->pairs++;
      b->dirty = true;
      start = stop;
   }
}

static void
flush_log(struct busy_device *d, uint64_t now)
{
   for (unsigned i = 0; i < BUSY_SECONDS; ++i) {
      struct busy_bin *b = &d->bins[i];
      if (!b->valid || !b->dirty)
         continue;
      unsigned overflow_mask = 0;
      for (unsigned e = 0; e < 4; ++e) {
         if (b->ranges[e].overflow)
            overflow_mask |= 1u << e;
      }
      fprintf(
         d->file,
         "GPU_BUSY pid=%d device=%p cpu_ns=%" PRIu64 " gpu_second=%" PRIu64
         " revision=%" PRIu64 " union_ns=%" PRIu64 " vertex_ns=%" PRIu64
         " fragment_ns=%" PRIu64 " compute_ns=%" PRIu64 " pairs=%" PRIu64
         " overflow=%u overflow_mask=%u"
         " union_ranges=%u vertex_ranges=%u fragment_ranges=%u compute_ranges=%u\n",
         getpid(), (void *)d->dev, now, b->second, ++b->revision,
         hk_busy_union_size(&b->ranges[0]), hk_busy_union_size(&b->ranges[VTX]),
         hk_busy_union_size(&b->ranges[FRAG]),
         hk_busy_union_size(&b->ranges[COMPUTE]), b->pairs, overflow_mask != 0,
         overflow_mask, b->ranges[0].count, b->ranges[VTX].count,
         b->ranges[FRAG].count, b->ranges[COMPUTE].count);
      b->dirty = false;
   }
   uint64_t pending = 0, oldest_pending = UINT64_MAX;
   for (struct hk_gpu_busy_queue *q = d->queues; q; q = q->next) {
      pending += BUSY_SLOTS - q->free_count;
      for (unsigned i = 0; i < BUSY_SLOTS; ++i) {
         if (q->slots[i].engines && q->slots[i].submit_cpu_ns < oldest_pending)
            oldest_pending = q->slots[i].submit_cpu_ns;
      }
   }
   if (!pending)
      oldest_pending = 0;
   fprintf(d->file,
           "GPU_BUSY_SAMPLE pid=%d device=%p cpu_ns=%" PRIu64
           " requested=%" PRIu64 " written=%" PRIu64 " collected=%" PRIu64
           " app_owned=%" PRIu64 " pool_full=%" PRIu64 " initial=%" PRIu64
           " invalid=%" PRIu64 " too_late=%" PRIu64 " pending_slots=%" PRIu64
           " oldest_pending_cpu_ns=%" PRIu64 " queries=%" PRIu64
           " query_errors=%" PRIu64 "\n",
           getpid(), (void *)d->dev, now, d->pairs_requested, d->pairs_written,
           d->pairs_collected, d->app_pairs, d->full_pairs, d->initial_pairs,
           d->invalid_pairs, d->late_pairs, pending, oldest_pending, d->queries,
           d->query_errors);
   fflush(d->file);
}

/* One non-waiting completed-point query per live queue per second, independent
 * of the number of submissions. The queue progress fence protects slot reuse.
 */
static void
collect(struct busy_device *d, uint64_t now)
{
   for (struct hk_gpu_busy_queue *q = d->queues; q; q = q->next) {
      uint32_t handle = q->queue->drm.syncobj;
      uint64_t completed = 0;
      int ret = U_WATCHDOG_IOCTL(
         "GPU busy completed-point query", d->dev->dev.fd,
         drmSyncobjQuery(d->dev->dev.fd, &handle, &completed, 1));
      d->queries++;
      if (ret) {
         d->query_errors++;
         continue;
      }
      u_watchdog_complete(d->dev->dev.fd, handle, completed);
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      for (unsigned i = 0; i < BUSY_SLOTS; ++i) {
         struct busy_slot *slot = &q->slots[i];
         if (!slot->engines || slot->point > completed)
            continue;
         for (unsigned engine = VTX; engine <= COMPUTE; ++engine) {
            if (!(slot->engines & (1u << engine)))
               continue;
            unsigned offset = engine == FRAG ? 2 : 0;
            uint64_t start = q->map[4 * i + offset];
            uint64_t end = q->map[4 * i + offset + 1];
            add_interval(d, engine,
                         agx_gpu_timestamp_to_ns(&d->dev->dev, start),
                         agx_gpu_timestamp_to_ns(&d->dev->dev, end));
         }
         slot->engines = 0;
         q->free_slots[q->free_count++] = i;
      }
   }
   flush_log(d, now);
}

void
hk_gpu_busy_sample(struct hk_queue *queue)
{
   if (!queue->gpu_busy)
      return;
   uint64_t now = os_time_get_nano();
   pthread_mutex_lock(&busy_lock);
   struct busy_device *d = queue->gpu_busy->device;
   queue->gpu_busy->submit_cpu_ns = now;
   if (now >= d->next_sample_ns) {
      d->next_sample_ns = now + SECOND_NS;
      collect(d, now);
   }
   pthread_mutex_unlock(&busy_lock);
}

static bool
pair_unused(const struct drm_asahi_timestamps *pair)
{
   return !pair->start.handle && !pair->start.offset && !pair->end.handle &&
          !pair->end.offset;
}

static void
instrument(struct hk_queue *queue, struct drm_asahi_timestamps *first,
           struct drm_asahi_timestamps *second, unsigned first_engine)
{
   struct hk_gpu_busy_queue *q = queue->gpu_busy;
   if (!q)
      return;
   pthread_mutex_lock(&busy_lock);
   struct busy_device *d = q->device;
   bool use_first = pair_unused(first);
   bool use_second = second && pair_unused(second);
   unsigned count = use_first + use_second;
   d->pairs_requested += second ? 2 : 1;
   d->app_pairs += (second ? 2 : 1) - count;
   if (!count)
      goto out;
   /* HK initializes its progress syncobj to 1 before the first submission.
    * Don't treat that initial value as proof that the first job completed. */
   if (queue->drm.timeline_value <= 1) {
      d->initial_pairs += count;
      goto out;
   }
   if (!q->free_count) {
      d->full_pairs += count;
      goto out;
   }
   unsigned index = q->free_slots[--q->free_count];
   q->slots[index] = (struct busy_slot){
      .point = queue->drm.timeline_value,
      .submit_cpu_ns = q->submit_cpu_ns,
      .engines =
         (use_first ? 1u << first_engine : 0) | (use_second ? 1u << FRAG : 0),
   };
   for (unsigned i = 0; i < 4; ++i)
      q->map[4 * index + i] = 0;
   __atomic_thread_fence(__ATOMIC_RELEASE);
   unsigned offset = index * 4 * sizeof(uint64_t);
   if (use_first) {
      first->start = (struct drm_asahi_timestamp){q->handle, offset};
      first->end = (struct drm_asahi_timestamp){q->handle, offset + 8};
   }
   if (use_second) {
      second->start = (struct drm_asahi_timestamp){q->handle, offset + 16};
      second->end = (struct drm_asahi_timestamp){q->handle, offset + 24};
   }
   d->pairs_written += count;
out:
   pthread_mutex_unlock(&busy_lock);
}

void
hk_gpu_busy_render(struct hk_queue *queue, struct drm_asahi_cmd_render *cmd)
{
   instrument(queue, &cmd->ts_vtx, &cmd->ts_frag, VTX);
}

void
hk_gpu_busy_compute(struct hk_queue *queue, struct drm_asahi_cmd_compute *cmd)
{
   instrument(queue, &cmd->ts, NULL, COMPUTE);
}

void
hk_gpu_busy_init(struct hk_queue *queue)
{
   struct hk_device *dev = hk_queue_device(queue);
   if (!dev->dev.params.command_timestamp_frequency_hz)
      return;
   struct hk_gpu_busy_queue *q = calloc(1, sizeof(*q));
   if (!q)
      return;
   /* Match timestamp query pools: shared and uncached on G16. */
   q->bo = agx_bo_create(&dev->dev, BUSY_SLOTS * 4 * sizeof(uint64_t), 0,
                         AGX_BO_SHARED, "GPU busy timestamps");
   if (!q->bo)
      goto fail;
   if (agx_bind_timestamps(&dev->dev, q->bo, &q->handle))
      goto fail;
   q->map = agx_bo_map(q->bo);
   if (!q->map)
      goto fail;
   q->queue = queue;
   for (unsigned i = 0; i < BUSY_SLOTS; ++i)
      q->free_slots[q->free_count++] = i;
   pthread_mutex_lock(&busy_lock);
   struct busy_device *d;
   for (d = devices; d && d->dev != dev; d = d->next)
      ;
   if (!d) {
      d = calloc(1, sizeof(*d));
      if (!d) {
         pthread_mutex_unlock(&busy_lock);
         goto fail;
      }
      d->file = open_log();
      if (!d->file) {
         free(d);
         pthread_mutex_unlock(&busy_lock);
         goto fail;
      }
      d->dev = dev;
      d->next = devices;
      devices = d;
      struct timespec wall_before, wall_after;
      clock_gettime(CLOCK_REALTIME, &wall_before);
      uint64_t before = os_time_get_nano();
      uint64_t gpu = agx_get_gpu_timestamp(&dev->dev);
      uint64_t after = os_time_get_nano();
      clock_gettime(CLOCK_REALTIME, &wall_after);
      uint64_t unix_before =
         (uint64_t)wall_before.tv_sec * SECOND_NS + wall_before.tv_nsec;
      uint64_t unix_after =
         (uint64_t)wall_after.tv_sec * SECOND_NS + wall_after.tv_nsec;
      d->next_sample_ns = after + SECOND_NS;
      fprintf(d->file,
              "GPU_BUSY_CONFIG pid=%d device=%p gpu_anchor_ns=%" PRIu64
              " cpu_before_ns=%" PRIu64 " cpu_after_ns=%" PRIu64
              " unix_before_ns=%" PRIu64 " unix_after_ns=%" PRIu64
              " frequency_hz=%" PRIu64
              " slots_per_queue=%u bins=%u ranges_per_union=%u"
              " source=firmware-command-timestamps scope=VkDevice"
              " interval_union=exact revision_policy=replace"
              " build" MESA_GIT_SHA1 "\n",
              getpid(), (void *)dev, gpu, before, after, unix_before,
              unix_after,
              (uint64_t)dev->dev.params.command_timestamp_frequency_hz,
              BUSY_SLOTS, BUSY_SECONDS, HK_BUSY_MAX_RANGES);
      fflush(d->file);
   }
   q->device = d;
   q->next = d->queues;
   d->queues = q;
   queue->gpu_busy = q;
   pthread_mutex_unlock(&busy_lock);
   return;
fail:
   fprintf(stderr, "HK GPU busy counters unavailable for queue %u\n",
           queue->drm.id);
   if (q->handle)
      dev->dev.ops.bo_unbind_object(&dev->dev, q->handle);
   agx_bo_unreference(&dev->dev, q->bo);
   free(q);
}

void
hk_gpu_busy_finish(struct hk_queue *queue)
{
   struct hk_gpu_busy_queue *q = queue->gpu_busy;
   if (!q)
      return;
   pthread_mutex_lock(&busy_lock);
   struct busy_device *d = q->device;
   collect(d, os_time_get_nano());
   struct hk_gpu_busy_queue **link = &d->queues;
   while (*link != q)
      link = &(*link)->next;
   *link = q->next;
   queue->gpu_busy = NULL;
   if (!d->queues) {
      struct busy_device **device_link = &devices;
      while (*device_link != d)
         device_link = &(*device_link)->next;
      *device_link = d->next;
      for (unsigned i = 0; i < BUSY_SECONDS; ++i)
         for (unsigned e = 0; e < 4; ++e)
            free(d->bins[i].ranges[e].data);
      fclose(d->file);
      free(d);
   }
   pthread_mutex_unlock(&busy_lock);
   struct hk_device *dev = hk_queue_device(queue);
   dev->dev.ops.bo_unbind_object(&dev->dev, q->handle);
   agx_bo_unreference(&dev->dev, q->bo);
   free(q);
}
#endif
