/* SPDX-License-Identifier: MIT
 * Diagnostic build only. Firmware timestamp pairs provide aggregate interval
 * unions, without retaining a command trace or changing application queries.
 */
#include "hk_gpu_busy.h"
#include "agx_timeline.h"
#include "git_sha1.h"
#ifdef HK_GPU_BUSY_COUNTERS
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include "util/os_time.h"
#include "util/u_wait_watchdog.h"
#include <sys/stat.h>
#include "hk_device.h"
#include "hk_gpu_busy_intervals.h"
#include "hk_queue.h"
#include <string.h>
#include "util/hash_table.h"
#include "agx_bo.h"

#define BUSY_SLOTS   8192
#define BUSY_SECONDS 64
#define SECOND_NS    UINT64_C(1000000000)
#define VTX          1
#define FRAG         2
#define COMPUTE      3

struct busy_pass_key {
   uint64_t code;
   uint32_t grid[3];
   uint32_t count;
   uint32_t flags, reserved;
};

struct busy_slot {
   uint64_t point, submit_cpu_ns;
   unsigned engines;
   struct busy_pass_key pass;
};

#define PASS_SLOTS 16384
struct busy_pass {
   struct busy_pass_key key;
   bool used;
   uint64_t commands, ns[4];
};
static struct busy_pass passes[PASS_SLOTS];
static uint64_t passes_dropped, pass_next_dump_ns;
static struct busy_pass_key pending_pass;

static void
pass_add(const struct busy_pass_key *key, unsigned engine, uint64_t ns, bool first)
{
   /* Hash the whole key: code addresses are page aligned and render keys have
    * no grid or count, so a key-derived index lands most passes in a few
    * probe chains, and later passes are dropped once those fill. */
   uint32_t h = _mesa_hash_data(key, sizeof(*key));
   for (unsigned probe = 0; probe < 64; ++probe) {
      struct busy_pass *p = &passes[(h + probe) % PASS_SLOTS];
      if (!p->used) {
         p->used = true;
         p->key = *key;
      } else if (memcmp(&p->key, key, sizeof(*key))) {
         continue;
      }
      p->ns[engine] += ns;
      p->commands += first;
      return;
   }
   passes_dropped++;
}

struct busy_device;
static void busy_printf(struct busy_device *d, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

static void
pass_dump(struct busy_device *f, uint64_t now)
{
   busy_printf(f, "GPU_PASS_BEGIN cpu_ns=%" PRIu64 " dropped=%" PRIu64 "\n", now, passes_dropped);
   for (unsigned i = 0; i < PASS_SLOTS; ++i) {
      struct busy_pass *p = &passes[i];
      if (!p->used)
         continue;
      busy_printf(f, "GPU_PASS code=0x%" PRIx64 " grid=%u,%u,%u count=%u flags=%u commands=%" PRIu64
              " vertex_ns=%" PRIu64 " fragment_ns=%" PRIu64 " compute_ns=%" PRIu64 "\n",
              p->key.code, p->key.grid[0], p->key.grid[1], p->key.grid[2], p->key.count, p->key.flags,
              p->commands, p->ns[1], p->ns[2], p->ns[3]);
   }
   busy_printf(f, "GPU_PASS_END cpu_ns=%" PRIu64 "\n", now);
}

struct busy_bin {
   uint64_t second, revision, pairs;
   bool valid, dirty;
   /* 0 is the union of all engines, then vertex/fragment/compute. */
   struct hk_busy_ranges ranges[4];
};

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
   /* Log lines are formatted into out under busy_lock and written to fd by
    * the collector thread after it drops the lock, so neither the file
    * system (a virtio round trip per write under muvm) nor collection runs
    * on the submitting thread. */
   int fd;
   char *out;
   size_t out_len, out_cap;
   uint64_t next_sample_ns;
   struct busy_bin bins[BUSY_SECONDS];
   uint64_t pairs_requested, pairs_written, pairs_collected;
   uint64_t app_pairs, full_pairs, initial_pairs, invalid_pairs, late_pairs;
   uint64_t queries, query_errors;
   bool timeline_clock_calibrated;
};

static pthread_mutex_t busy_lock = PTHREAD_MUTEX_INITIALIZER;

static void
busy_printf(struct busy_device *d, const char *fmt, ...)
{
   for (;;) {
      va_list args;
      va_start(args, fmt);
      size_t room = d->out_cap - d->out_len;
      int n = vsnprintf(d->out ? d->out + d->out_len : NULL, room, fmt, args);
      va_end(args);
      if (n < 0)
         return;
      if ((size_t)n < room) {
         d->out_len += n;
         return;
      }
      size_t cap = MAX2(d->out_cap * 2, d->out_len + n + 4096);
      char *out = realloc(d->out, cap);
      if (!out)
         return;
      d->out = out;
      d->out_cap = cap;
   }
}

static void
write_all(int fd, const char *data, size_t len)
{
   while (len) {
      ssize_t n = write(fd, data, len);
      if (n < 0 && errno == EINTR)
         continue;
      if (n <= 0)
         return;
      data += n;
      len -= n;
   }
}

/* HK_GPU_BUSY_FILE enables busy counters, pass identity and shader identity;
 * without it every entry point returns at once. Read once. */
static bool
busy_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *file = getenv("HK_GPU_BUSY_FILE");
      enabled = file && *file;
   }
   return enabled;
}
static struct busy_device *devices;

static int
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
      return -1;
   struct stat st;
   if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != geteuid()) {
      close(fd);
      return -1;
   }
   return fd;
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
      busy_printf(
         d,
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
   busy_printf(d,
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
         /* The submit's progress point only orders its last command; wait for
          * this command's own end timestamps (pass-complete-overlay). */
         bool written = true;
         for (unsigned engine = VTX; engine <= COMPUTE; ++engine) {
            if (slot->engines & (1u << engine))
               written &= q->map[4 * i + (engine == FRAG ? 2 : 0) + 1] != 0;
         }
         if (!written && now - slot->submit_cpu_ns < 10 * SECOND_NS)
            continue;
         for (unsigned engine = VTX; engine <= COMPUTE; ++engine) {
            if (!(slot->engines & (1u << engine)))
               continue;
            unsigned offset = engine == FRAG ? 2 : 0;
            uint64_t start = q->map[4 * i + offset];
            uint64_t end = q->map[4 * i + offset + 1];
            /* HK_TIMELINE_FILE: per-command GPU intervals; f is the DRM fd,
             * which identifies the device across submit records. */
            agx_timeline_emit(TL_GPU_INTERVAL, engine, slot->point,
               q->queue->drm.id, ((uint64_t)q->handle << 32) | ((4*i+offset)*8),
               agx_gpu_timestamp_to_ns(&d->dev->dev, start),
               agx_gpu_timestamp_to_ns(&d->dev->dev, end), 1000000000,
               d->dev->dev.fd, 0);
            add_interval(d, engine,
                         agx_gpu_timestamp_to_ns(&d->dev->dev, start),
                         agx_gpu_timestamp_to_ns(&d->dev->dev, end));
            if (start && end > start)
               pass_add(&slot->pass, engine,
                        agx_gpu_timestamp_to_ns(&d->dev->dev, end) -
                           agx_gpu_timestamp_to_ns(&d->dev->dev, start),
                        engine != FRAG);
         }
         slot->engines = 0;
         q->free_slots[q->free_count++] = i;
      }
   }
   flush_log(d, now);
   if (now >= pass_next_dump_ns) {
      pass_next_dump_ns = now + 5 * SECOND_NS;
      pass_dump(d, now);
   }
}

void
hk_gpu_busy_set_pass(struct hk_queue *queue, uint64_t code,
                     const uint32_t grid[3], uint32_t count, uint32_t flags)
{
   if (!queue->gpu_busy)
      return;
   pthread_mutex_lock(&busy_lock);
   pending_pass = (struct busy_pass_key){
      .code = code, .grid = {grid[0], grid[1], grid[2]}, .count = count, .flags = flags, .reserved = 0};
   pthread_mutex_unlock(&busy_lock);
}

void
hk_gpu_busy_shader(struct agx_bo *bo, const void *code, uint32_t size,
                   unsigned stage)
{
   if (!bo || !busy_enabled())
      return;
   uint32_t hash = _mesa_hash_data(code, size);
   pthread_mutex_lock(&busy_lock);
   for (struct busy_device *d = devices; d; d = d->next) {
      busy_printf(d, "GPU_SHADER cpu_ns=%" PRIu64 " code=0x%" PRIx64
              " hash=0x%08x size=%u stage=%u\n", os_time_get_nano(),
              bo->va->addr, hash, size, stage);
      break;
   }
   pthread_mutex_unlock(&busy_lock);
   const char *dir = getenv("HK_GPU_PASS_SHADER_DIR");
   if (dir) {
      char path[512];
      snprintf(path, sizeof(path), "%s/%08x-%u.bin", dir, hash, size);
      FILE *f = fopen(path, "wbx");
      if (f) {
         fwrite(code, 1, size, f);
         fclose(f);
      }
   }
}

void
hk_gpu_busy_storage_format(unsigned vk_format, unsigned store_encode)
{
   static bool seen[1024];
   if (!busy_enabled() || vk_format >= ARRAY_SIZE(seen) || seen[vk_format])
      return;
   pthread_mutex_lock(&busy_lock);
   if (!seen[vk_format]) {
      seen[vk_format] = true;
      for (struct busy_device *d = devices; d; d = d->next) {
         busy_printf(d, "GPU_STORAGE_FORMAT vk=%u store_encode=%u\n",
                     vk_format, store_encode);
         break;
      }
   }
   pthread_mutex_unlock(&busy_lock);
}

void
hk_gpu_busy_shader_source(const void *code, uint32_t size, const char *name,
                          const uint8_t *blake3, unsigned stage)
{
   if (!code || !busy_enabled())
      return;
   char b3[65];
   for (unsigned i = 0; i < 32; ++i)
      snprintf(b3 + 2 * i, 3, "%02x", blake3[i]);
   char clean[96] = "";
   for (unsigned i = 0; name && name[i] && i + 1 < sizeof(clean); ++i)
      clean[i] = (name[i] <= ' ' || name[i] == '=') ? '_' : name[i];
   pthread_mutex_lock(&busy_lock);
   for (struct busy_device *d = devices; d; d = d->next) {
      busy_printf(d, "GPU_SHADER_SRC hash=0x%08x size=%u stage=%u blake3=%s name=%s\n",
              _mesa_hash_data(code, size), size, stage, b3, clean[0] ? clean : "-");
      break;
   }
   pthread_mutex_unlock(&busy_lock);
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
   /* HK_TIMELINE_FILE: calibrate GPU timestamps against CNTVCT (aarch64)
    * or CLOCK_MONOTONIC once recording is enabled. */
   if (agx_timeline_enabled() &&
#if defined(__aarch64__)
       !d->timeline_clock_calibrated
#else
       now >= d->next_sample_ns
#endif
       ) {
      d->timeline_clock_calibrated = true;
      for (unsigned i = 0; i < 32; i++) {
         uint64_t before = os_time_get_nano(), first = before, last = 0;
         uint64_t hz = 1000000000;
#if defined(__aarch64__)
         __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(first));
         __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(hz));
#endif
         uint64_t gpu = agx_get_gpu_timestamp(&d->dev->dev);
#if defined(__aarch64__)
         __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(last));
#endif
         uint64_t after = os_time_get_nano();
#if !defined(__aarch64__)
         last = after;
#endif
         agx_timeline_emit(TL_GPU_CLOCK,
#if defined(__aarch64__)
                           1,
#else
                           2,
#endif
                           queue->drm.id, gpu, before, after, 1000000000,
                           first, last, hz);
      }
   }
   pthread_mutex_unlock(&busy_lock);
}

/* Once a second: collect every device under busy_lock, then write what was
 * formatted with the lock dropped. */
static void *
collector_main(void *arg)
{
   for (;;) {
      struct timespec second = {.tv_sec = 1};
      while (nanosleep(&second, &second) && errno == EINTR)
         ;
      struct pending {
         int fd;
         char *data;
         size_t len;
      } pending[8];
      unsigned count = 0;
      pthread_mutex_lock(&busy_lock);
      uint64_t now = os_time_get_nano();
      for (struct busy_device *d = devices; d; d = d->next) {
         if (now >= d->next_sample_ns) {
            d->next_sample_ns = now + SECOND_NS;
            collect(d, now);
         }
         if (!d->out_len || count == ARRAY_SIZE(pending))
            continue;
         int fd = dup(d->fd);
         if (fd < 0)
            continue;
         pending[count++] = (struct pending){fd, d->out, d->out_len};
         d->out = NULL;
         d->out_len = d->out_cap = 0;
      }
      pthread_mutex_unlock(&busy_lock);
      for (unsigned i = 0; i < count; ++i) {
         write_all(pending[i].fd, pending[i].data, pending[i].len);
         close(pending[i].fd);
         free(pending[i].data);
      }
   }
   return NULL;
}

static void
start_collector(void)
{
   pthread_t thread;
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
   if (pthread_create(&thread, &attr, collector_main, NULL))
      fprintf(stderr, "HK GPU busy collector thread unavailable\n");
   pthread_attr_destroy(&attr);
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
      .pass = pending_pass,
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
   /* Off unless HK_GPU_BUSY_FILE is set: no timestamp BOs and no per-command
    * work otherwise. */
   if (!busy_enabled())
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
      d->fd = open_log();
      if (d->fd < 0) {
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
      busy_printf(d,
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
   }
   q->device = d;
   q->next = d->queues;
   d->queues = q;
   queue->gpu_busy = q;
   pthread_mutex_unlock(&busy_lock);
   static pthread_once_t collector_once = PTHREAD_ONCE_INIT;
   pthread_once(&collector_once, start_collector);
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
      /* collect() dumps the pass table only every 5 s; write the passes
       * since the last dump before the log closes. */
      pass_dump(d, os_time_get_nano());
      struct busy_device **device_link = &devices;
      while (*device_link != d)
         device_link = &(*device_link)->next;
      *device_link = d->next;
      for (unsigned i = 0; i < BUSY_SECONDS; ++i)
         for (unsigned e = 0; e < 4; ++e)
            free(d->bins[i].ranges[e].data);
      write_all(d->fd, d->out, d->out_len);
      close(d->fd);
      free(d->out);
      free(d);
   }
   pthread_mutex_unlock(&busy_lock);
   struct hk_device *dev = hk_queue_device(queue);
   dev->dev.ops.bo_unbind_object(&dev->dev, q->handle);
   agx_bo_unreference(&dev->dev, q->bo);
   free(q);
}
#endif
