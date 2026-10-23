/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "asahi/debug.h"
#include "agx_bo.h"
#include <inttypes.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include "util/u_debug.h"

#include "util/hash_table.h"
#include "util/ralloc.h"
#include "agx_device.h"
#include "decode.h"
#include "layout.h"

static pthread_mutex_t agx_reaper_devices_lock = PTHREAD_MUTEX_INITIALIZER;
static struct list_head agx_reaper_devices = {
   &agx_reaper_devices, &agx_reaper_devices,
};
static pthread_once_t agx_reaper_atfork_once = PTHREAD_ONCE_INIT;
static int agx_reaper_atfork_error;

static void
agx_reaper_fork_prepare(void)
{
   pthread_mutex_lock(&agx_reaper_devices_lock);
   list_for_each_entry(struct agx_device, dev, &agx_reaper_devices,
                       bo_cache.device_link) {
      pthread_mutex_lock(&dev->bo_map_lock);
      pthread_mutex_lock(&dev->bo_cache.lock);
   }
}

static void
agx_reaper_fork_parent(void)
{
   list_for_each_entry_rev(struct agx_device, dev, &agx_reaper_devices,
                           bo_cache.device_link) {
      pthread_mutex_unlock(&dev->bo_cache.lock);
      pthread_mutex_unlock(&dev->bo_map_lock);
   }
   pthread_mutex_unlock(&agx_reaper_devices_lock);
}

static void
agx_reaper_fork_child(void)
{
   list_for_each_entry(struct agx_device, dev, &agx_reaper_devices,
                       bo_cache.device_link) {
      /* GEM handles and GPU mappings still belong to the shared drm_file.
       * The child must not close handles or recycle the parent's VA, even
       * if a vanished worker had already removed a BO from the table.
       */
      dev->bo_cache.forked = true;
      dev->bo_cache.started = false;
      dev->bo_cache.busy = false;
      dev->bo_closing_handle = 0;
   }
   agx_reaper_fork_parent();
}

static void
agx_reaper_register_atfork(void)
{
   agx_reaper_atfork_error = pthread_atfork(agx_reaper_fork_prepare,
                                           agx_reaper_fork_parent,
                                           agx_reaper_fork_child);
}

/* Helper to calculate the bucket index of a BO */
static unsigned
agx_bucket_index(size_t size)
{
   /* Round down to POT to compute a bucket index */
   unsigned bucket_index = util_logbase2(size);

   /* Clamp to supported buckets. Huge allocations use the largest bucket */
   bucket_index = CLAMP(bucket_index, MIN_BO_CACHE_BUCKET, MAX_BO_CACHE_BUCKET);

   /* Reindex from 0 */
   return (bucket_index - MIN_BO_CACHE_BUCKET);
}

static struct list_head *
agx_bucket(struct agx_device *dev, size_t size)
{
   return &dev->bo_cache.buckets[agx_bucket_index(size)];
}

static void
agx_bo_cache_remove_locked(struct agx_device *dev, struct agx_bo *bo)
{
   list_del(&bo->bucket_link);
   list_del(&bo->lru_link);
   dev->bo_cache.size -= bo->size;
}

/* Tries to fetch a BO of sufficient size with the appropriate flags from the
 * BO cache. If it succeeds, it returns that BO and removes the BO from the
 * cache. If it fails, it returns NULL signaling the caller to allocate a new
 * BO. */

typedef int (*reused_fn)(int, uint32_t);

/* The m1n1 shim's reuse hook, looked up once. The pointer itself is the
 * published value, so a racing reader sees either "unresolved" or the hook. */
#define AGX_BO_REUSED_UNRESOLVED ((reused_fn)(uintptr_t)1)

static reused_fn
agx_bo_reused_hook(void)
{
   static reused_fn hook = AGX_BO_REUSED_UNRESOLVED;

   reused_fn fn = p_atomic_read(&hook);
   if (fn == AGX_BO_REUSED_UNRESOLVED) {
      fn = (reused_fn)dlsym(RTLD_DEFAULT, "asahi_m1n1_bo_reused");
      p_atomic_set(&hook, fn);
   }
   return fn;
}

struct agx_bo *
agx_bo_cache_fetch(struct agx_device *dev, size_t size, size_t align,
                   uint32_t flags, const bool dontwait)
{
   pthread_mutex_lock(&dev->bo_cache.lock);
   struct list_head *bucket = agx_bucket(dev, size);
   struct agx_bo *bo = NULL;

   /* Iterate the bucket looking for something suitable, most recently freed
    * first. Reusing the oldest entry would cycle a steady workload through
    * every cached BO, so that none ever ages out of the cache. */
   list_for_each_entry_rev(struct agx_bo, entry, bucket, bucket_link) {
      if (entry->size < size || entry->flags != flags)
         continue;

      /* Do not return more than 2x oversized BOs. */
      if (entry->size > 2 * size)
         continue;

      if (align > entry->align)
         continue;

      /* This one works, use it */
      agx_bo_cache_remove_locked(dev, entry);
      bo = entry;
      break;
   }
   pthread_mutex_unlock(&dev->bo_cache.lock);

   if (bo) {
      /* A cached allocation has a new owner. The remote copy may contain
       * writes by the previous GPU user even where the host bytes did not
       * change. Start a new upload baseline before exposing it for CPU use. */
      reused_fn reused = agx_bo_reused_hook();
      if (reused && reused(dev->fd, bo->uapi_handle)) {
         agx_bo_free(dev, bo);
         bo = NULL;
      }
   }

   return bo;
}

static void
agx_bo_cache_put_locked(struct agx_device *dev, struct agx_bo *bo)
{
   struct list_head *bucket = agx_bucket(dev, bo->size);
   struct timespec time;
   bool was_empty = list_is_empty(&dev->bo_cache.lru);

   /* Add us to the bucket */
   list_addtail(&bo->bucket_link, bucket);

   /* Add us to the LRU list and update the last_used field. */
   list_addtail(&bo->lru_link, &dev->bo_cache.lru);
   clock_gettime(CLOCK_MONOTONIC, &time);
   bo->last_used = time.tv_sec;

   /* Update statistics */
   dev->bo_cache.size += bo->size;

   if (0) {
      printf("BO cache: %zu KiB (+%zu KiB from %s, hit/miss %" PRIu64
             "/%" PRIu64 ")\n",
             DIV_ROUND_UP(dev->bo_cache.size, 1024),
             DIV_ROUND_UP(bo->size, 1024), bo->label,
             p_atomic_read(&dev->bo_cache.hits),
             p_atomic_read(&dev->bo_cache.misses));
   }

   /* Update label for debug */
   bo->label = "Unused (BO cache)";

   /* An existing head already has an earlier expiry timer armed. */
   if (was_empty)
      pthread_cond_signal(&dev->bo_cache.cond);
}

/* Tries to add a BO to the cache. Returns if it was successful */
static bool
agx_bo_cache_put(struct agx_device *dev, struct agx_bo *bo)
{
   if ((bo->flags & AGX_BO_SHARED) ||
       (bo->va && (bo->va->flags & (AGX_VA_REPLAY | AGX_VA_FIXED)))) {
      return false;
   } else {
      pthread_mutex_lock(&dev->bo_cache.lock);
      agx_bo_cache_put_locked(dev, bo);
      pthread_mutex_unlock(&dev->bo_cache.lock);

      return true;
   }
}

/* Queue an unreferenced BO for destruction. No allocation or ioctl is needed
 * here: the BO itself is the queue entry until the reaper takes ownership.
 */
void
agx_bo_free(struct agx_device *dev, struct agx_bo *bo)
{
   assert(!bo->refcnt && !bo->retired);
   if (unlikely(dev->bo_cache.forked)) {
      memset(bo, 0, sizeof(*bo));
      return;
   }

   pthread_mutex_lock(&dev->bo_cache.lock);
   bool was_empty = list_is_empty(&dev->bo_cache.retired);
   bo->retired = true;
   list_addtail(&bo->bucket_link, &dev->bo_cache.retired);
   if (was_empty)
      pthread_cond_signal(&dev->bo_cache.cond);
   pthread_mutex_unlock(&dev->bo_cache.lock);
}

/* Called by import with bo_map_lock held. Until the reaper starts closing a
 * shared BO, its original GEM handle and VA can be used again.
 */
void
agx_bo_reclaim(struct agx_device *dev, struct agx_bo *bo)
{
   assert(bo->flags & AGX_BO_SHARED);
   assert(!bo->refcnt && bo->retired);
   pthread_mutex_lock(&dev->bo_cache.lock);
   list_del(&bo->bucket_link);
   bo->retired = false;
   pthread_cond_broadcast(&dev->bo_cache.cond);
   pthread_mutex_unlock(&dev->bo_cache.lock);
   p_atomic_set(&bo->refcnt, 1);
}

/* Explicit trims queue all currently cached BOs. OOM recovery and teardown
 * additionally wait for the reaper; ordinary pool trims do not block on the
 * kernel. Entries returned to the cache later are not part of this trim.
 */
void
agx_bo_cache_evict_all(struct agx_device *dev)
{
   if (unlikely(dev->bo_cache.forked))
      return;

   pthread_mutex_lock(&dev->bo_cache.lock);
   bool wake = list_is_empty(&dev->bo_cache.retired) &&
               !list_is_empty(&dev->bo_cache.lru);
   list_for_each_entry_safe(struct agx_bo, bo, &dev->bo_cache.lru, lru_link) {
      agx_bo_cache_remove_locked(dev, bo);
      bo->retired = true;
      list_addtail(&bo->bucket_link, &dev->bo_cache.retired);
   }
   if (wake)
      pthread_cond_signal(&dev->bo_cache.cond);
   pthread_mutex_unlock(&dev->bo_cache.lock);
}

/* Return the oldest stale cache entry. Keep the others available for reuse
 * while a close is blocked, instead of filling a separate eviction queue.
 * The integer-second expiry matches the previous cache policy (> 2 s).
 */
static struct agx_bo *
agx_bo_cache_stale_locked(struct agx_device *dev)
{
   if (list_is_empty(&dev->bo_cache.lru))
      return NULL;

   struct agx_bo *bo =
      list_first_entry(&dev->bo_cache.lru, struct agx_bo, lru_link);
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   return now.tv_sec - bo->last_used > 2 ? bo : NULL;
}

static void
agx_bo_close(struct agx_device *dev, const struct agx_bo *bo)
{
   if (bo->_map)
      munmap(bo->_map, bo->size);
   if (bo->prime_fd != -1)
      close(bo->prime_fd);

   struct drm_gem_close args = {.handle = bo->handle};
   if (drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &args)) {
      fprintf(stderr, "DRM_IOCTL_GEM_CLOSE failed: %m\n");
      /* Keep the address reserved if the old mapping may still exist. */
      free(bo->va);
      return;
   }

   /* Do not recycle a VA until its old kernel mappings are gone. */
   agx_va_free(dev, bo->va, false);
}

static bool
agx_bo_reap_one(struct agx_device *dev)
{
   /* Same order as final unreference and import. Nothing takes bo_map_lock
    * while holding the cache lock. Neither lock covers a blocking close.
    */
   pthread_mutex_lock(&dev->bo_map_lock);
   pthread_mutex_lock(&dev->bo_cache.lock);
   struct agx_bo *bo;
   if (!list_is_empty(&dev->bo_cache.retired)) {
      bo = list_first_entry(&dev->bo_cache.retired, struct agx_bo, bucket_link);
      list_del(&bo->bucket_link);
   } else {
      bo = agx_bo_cache_stale_locked(dev);
      if (bo)
         agx_bo_cache_remove_locked(dev, bo);
   }

   if (!bo) {
      pthread_mutex_unlock(&dev->bo_cache.lock);
      pthread_mutex_unlock(&dev->bo_map_lock);
      return false;
   }

   assert(!bo->refcnt && !bo->writer);
   assert(!dev->bo_cache.busy && !dev->bo_closing_handle);
   dev->bo_cache.busy = true;
   if (bo->flags & AGX_BO_SHARED)
      dev->bo_closing_handle = bo->handle;

   /* The kernel can reuse a handle before GEM_CLOSE returns. Remove the old
    * table entry first, then operate solely on this private snapshot. Never
    * clear the slot after the ioctl: it may already belong to a new BO.
    */
   struct agx_bo dead = *bo;
   memset(bo, 0, sizeof(*bo));
   pthread_mutex_unlock(&dev->bo_cache.lock);
   pthread_mutex_unlock(&dev->bo_map_lock);

   agx_bo_close(dev, &dead);

   pthread_mutex_lock(&dev->bo_map_lock);
   dev->bo_closing_handle = 0;
   pthread_cond_broadcast(&dev->bo_map_cond);
   pthread_mutex_unlock(&dev->bo_map_lock);

   pthread_mutex_lock(&dev->bo_cache.lock);
   dev->bo_cache.busy = false;
   pthread_cond_broadcast(&dev->bo_cache.cond);
   pthread_mutex_unlock(&dev->bo_cache.lock);
   return true;
}

static int
agx_bo_reaper(void *data)
{
   struct agx_device *dev = data;
   u_thread_setname("agx-bo-reaper");

   for (;;) {
      pthread_mutex_lock(&dev->bo_cache.lock);
      while (list_is_empty(&dev->bo_cache.retired) &&
             !agx_bo_cache_stale_locked(dev)) {
         if (dev->bo_cache.stop) {
            pthread_mutex_unlock(&dev->bo_cache.lock);
            return 0;
         }

         if (list_is_empty(&dev->bo_cache.lru)) {
            pthread_cond_wait(&dev->bo_cache.cond, &dev->bo_cache.lock);
         } else {
            struct agx_bo *bo =
               list_first_entry(&dev->bo_cache.lru, struct agx_bo, lru_link);
            struct timespec expiry = {.tv_sec = bo->last_used + 3};
            pthread_cond_timedwait(&dev->bo_cache.cond, &dev->bo_cache.lock,
                                   &expiry);
         }
      }
      pthread_mutex_unlock(&dev->bo_cache.lock);
      agx_bo_reap_one(dev);
   }
}

void
agx_bo_cache_init(struct agx_device *dev)
{
   pthread_mutex_init(&dev->bo_cache.lock, NULL);
   pthread_condattr_t attr;
   pthread_condattr_init(&attr);
   pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
   pthread_cond_init(&dev->bo_cache.cond, &attr);
   pthread_condattr_destroy(&attr);
   pthread_cond_init(&dev->bo_map_cond, NULL);
   list_inithead(&dev->bo_cache.lru);
   list_inithead(&dev->bo_cache.retired);
   list_inithead(&dev->bo_cache.device_link);
   for (unsigned i = 0; i < ARRAY_SIZE(dev->bo_cache.buckets); ++i)
      list_inithead(&dev->bo_cache.buckets[i]);
}

bool
agx_bo_cache_start(struct agx_device *dev)
{
   assert(!dev->bo_cache.started);
   pthread_once(&agx_reaper_atfork_once, agx_reaper_register_atfork);
   if (agx_reaper_atfork_error || dev->bo_cache.forked)
      return false;

   pthread_mutex_lock(&agx_reaper_devices_lock);
   list_addtail(&dev->bo_cache.device_link, &agx_reaper_devices);
   dev->bo_cache.started = u_thread_create(&dev->bo_cache.thread, agx_bo_reaper,
                                           dev) == thrd_success;
   pthread_mutex_unlock(&agx_reaper_devices_lock);
   return dev->bo_cache.started;
}

void
agx_bo_cache_wait(struct agx_device *dev)
{
   /* Must not hold bo_map_lock: the worker needs it to complete a close. */
   if (unlikely(dev->bo_cache.forked))
      return;

   /* Device bootstrap can encounter an allocation failure before starting
    * the worker. No GPU work exists yet; reclaim synchronously in that case.
    */
   if (!dev->bo_cache.started) {
      while (agx_bo_reap_one(dev))
         ;
      return;
   }

   pthread_mutex_lock(&dev->bo_cache.lock);
   while (dev->bo_cache.busy || !list_is_empty(&dev->bo_cache.retired))
      pthread_cond_wait(&dev->bo_cache.cond, &dev->bo_cache.lock);
   pthread_mutex_unlock(&dev->bo_cache.lock);
}

void
agx_bo_cache_finish(struct agx_device *dev)
{
   if (!dev->bo_cache.forked) {
      agx_bo_cache_evict_all(dev);
      agx_bo_cache_wait(dev);
      if (dev->bo_cache.started) {
         pthread_mutex_lock(&dev->bo_cache.lock);
         dev->bo_cache.stop = true;
         pthread_cond_signal(&dev->bo_cache.cond);
         pthread_mutex_unlock(&dev->bo_cache.lock);
         thrd_join(dev->bo_cache.thread, NULL);
      }
   }

   pthread_mutex_lock(&agx_reaper_devices_lock);
   list_delinit(&dev->bo_cache.device_link);
   pthread_mutex_unlock(&agx_reaper_devices_lock);

   /* In the child, inherited condvars may still record vanished waiters;
    * even destroying those condvars could wait forever. No child uses them.
    */
   if (!dev->bo_cache.forked) {
      pthread_cond_destroy(&dev->bo_map_cond);
      pthread_cond_destroy(&dev->bo_cache.cond);
   }
   pthread_mutex_destroy(&dev->bo_cache.lock);
}

void
agx_bo_reference(struct agx_bo *bo)
{
   if (bo) {
      ASSERTED int count = p_atomic_inc_return(&bo->refcnt);
      assert(count != 1);
   }
}

struct label_stat {
   const char *label;
   uint32_t count;
   size_t alloc_B;
   size_t mapped_B;
};

static void
account_bo(struct label_stat *stat, struct agx_bo *bo)
{
   stat->count++;
   stat->alloc_B += bo->size;

   if (bo->_map != NULL)
      stat->mapped_B += bo->size;
}

static void
print_size(FILE *fp, size_t size_B)
{
   if (size_B >= (1024 * 1024 * 1024)) {
      fprintf(fp, "%.1f GiB", (double)size_B / (double)(1024 * 1024 * 1024));
   } else if (size_B >= (1024 * 1024)) {
      fprintf(fp, "%.1f MiB", (double)size_B / (double)(1024 * 1024));
   } else if (size_B >= 1024) {
      fprintf(fp, "%zu KiB", DIV_ROUND_UP(size_B, 1024));
   } else {
      fprintf(fp, "%zu B", size_B);
   }
}

static void
print_stat(FILE *fp, struct label_stat *stat)
{
   const char *BOLD = "\e[1m";
   const char *RESET = "\e[0m";

   fprintf(fp, "%s%s%s: ", BOLD, stat->label, RESET);
   print_size(fp, stat->alloc_B);

   if (stat->mapped_B) {
      fprintf(fp, ", mapped ");
      print_size(fp, stat->mapped_B);
   }

   fprintf(fp, ", %u BOs\n", stat->count);
}

static int
compare_size(const void *a_, const void *b_)
{
   const struct label_stat *const *label_a = a_;
   const struct label_stat *const *label_b = b_;

   size_t a = (*label_a)->alloc_B;
   size_t b = (*label_b)->alloc_B;

   return (a > b) ? 1 : (a < b) ? -1 : 0;
}

static void
agx_bo_dump_all(struct agx_device *dev)
{
   struct label_stat accum = {.label = "Total"};
   struct hash_table *totals = _mesa_string_hash_table_create(NULL);
   bool verbose = dev->debug & AGX_DBG_BODUMPVERBOSE;

   if (verbose)
      fprintf(stderr, "---\n");

   for (uint32_t handle = 0; handle < dev->max_handle; handle++) {
      struct agx_bo *bo = agx_lookup_bo(dev, handle);
      if (!bo->size)
         continue;

      if (verbose) {
         fprintf(stderr, "%u: %s %zu KiB\n", handle, bo->label,
                 bo->size / 1024);
      }

      account_bo(&accum, bo);

      assert(bo->label != NULL && "Everything must be labeled");

      struct hash_entry *ent = _mesa_hash_table_search(totals, bo->label);
      struct label_stat *ls;
      if (ent != NULL) {
         ls = ent->data;
      } else {
         ls = rzalloc(totals, struct label_stat);
         ls->label = bo->label;
         _mesa_hash_table_insert(totals, bo->label, ls);
      }

      account_bo(ls, bo);
   }

   if (verbose) {
      fprintf(stderr, "\n");
   }

   unsigned nr_labels = _mesa_hash_table_num_entries(totals);
   struct label_stat **stats =
      rzalloc_array(totals, struct label_stat *, nr_labels);
   unsigned i = 0;

   hash_table_foreach(totals, ent) {
      assert(i < nr_labels);
      stats[i++] = ent->data;
   }

   assert(i == nr_labels);

   /* Sort labels in ascending order of size */
   qsort(stats, nr_labels, sizeof(struct label_stat *), compare_size);

   for (i = 0; i < nr_labels; ++i) {
      print_stat(stderr, stats[i]);
   }

   print_stat(stderr, &accum);

   if (verbose)
      fprintf(stderr, "---\n\n");
   else
      fprintf(stderr, "\n");

   ralloc_free(totals);
}

static void
agx_bo_dump_all_periodic(struct agx_device *dev)
{
   if (likely(!(dev->debug & (AGX_DBG_BODUMP | AGX_DBG_BODUMPVERBOSE))))
      return;

   static time_t agx_last_dumped_time = 0;

   time_t now = time(NULL);
   if (now == agx_last_dumped_time)
      return;

   agx_bo_dump_all(dev);
   agx_last_dumped_time = now;
}

void
agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;

   assert(p_atomic_read(&bo->refcnt) > 0);

   /* Drop references other than the last without the lock. The last one is
    * dropped under bo_map_lock, which agx_bo_import also holds, so an import
    * can safely resurrect a queued shared BO before the reaper removes it.
    * Checking the count again after taking the lock is not enough: an import
    * and a release could both run in between, and two threads would then free
    * the same BO (as iris does, see iris_bo_unreference).
    */
   int c = p_atomic_read(&bo->refcnt), old;
   while (c != 1 && (old = p_atomic_cmpxchg(&bo->refcnt, c, c - 1)) != c)
      c = old;
   if (c != 1)
      return;

   pthread_mutex_lock(&dev->bo_map_lock);

   if (p_atomic_dec_zero(&bo->refcnt)) {
      if (unlikely(dev->bo_cache.forked)) {
         memset(bo, 0, sizeof(*bo));
         pthread_mutex_unlock(&dev->bo_map_lock);
         return;
      }

      assert(!p_atomic_read_relaxed(&bo->writer));

      if (bo->apple9_entry_offset || bo->apple9_entry_fixed) {
         simple_mtx_lock(&dev->apple9_entry_lock);
         if (bo->apple9_entry_offset) {
            util_vma_heap_free(&dev->apple9_entry_heap,
                              bo->apple9_entry_offset - AGX_APPLE9_ENTRY_CODE_OFFSET,
                              AGX_APPLE9_ENTRY_BLOCK_SIZE);
            bo->apple9_entry_offset = 0;
            list_del(&bo->apple9_entry_link);
         }
         bo->apple9_entry_fixed = false;
         simple_mtx_unlock(&dev->apple9_entry_lock);
      }

      if (dev->debug & AGX_DBG_TRACE)
         agxdecode_track_free(dev->agxdecode, bo);

      if (!agx_bo_cache_put(dev, bo))
         agx_bo_free(dev, bo);
   }

   agx_bo_dump_all_periodic(dev);

   pthread_mutex_unlock(&dev->bo_map_lock);
}

static void agx_bo_finish_create(struct agx_device *dev, struct agx_bo *bo,
                                 const char *label);

struct agx_bo *
agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
              enum agx_bo_flags flags, const char *label)
{
   struct agx_bo *bo;
   assert(size > 0);

   /* BOs are allocated in pages */
   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);

   /* See if we have a BO already in the cache */
   bo = agx_bo_cache_fetch(dev, size, align, flags, true);

   /* Update stats based on the first attempt to fetch */
   if (bo != NULL)
      p_atomic_inc(&dev->bo_cache.hits);
   else
      p_atomic_inc(&dev->bo_cache.misses);

   /* Otherwise, allocate a fresh BO. If allocation fails, we can try waiting
    * for something in the cache. But if there's no nothing suitable, we should
    * flush the cache to make space for the new allocation.
    */
   if (!bo)
      bo = dev->ops.bo_alloc(dev, size, align, flags, 0, 0);
   if (!bo)
      bo = agx_bo_cache_fetch(dev, size, align, flags, false);
   if (!bo) {
      agx_bo_cache_evict_all(dev);
      agx_bo_cache_wait(dev);
      bo = dev->ops.bo_alloc(dev, size, align, flags, 0, 0);
   }

   if (!bo) {
      fprintf(stderr, "BO creation failed\n");
      return NULL;
   }

   agx_bo_finish_create(dev, bo, label);
   return bo;
}

struct agx_bo *
agx_bo_create_va(struct agx_device *dev, size_t size, unsigned align,
                 enum agx_bo_flags flags, enum agx_va_flags va_flags,
                 uint64_t fixed_va, const char *label)
{
   if (!va_flags)
      return agx_bo_create(dev, size, align, flags, label);

   assert(size > 0);
   size = ALIGN_POT(size, AIL_PAGESIZE);
   align = MAX2(align, AIL_PAGESIZE);

   /* A cached BO keeps the VA it was created with, so these never come from
    * the cache (and never go back to it, see agx_bo_cache_put).
    */
   struct agx_bo *bo =
      dev->ops.bo_alloc(dev, size, align, flags, va_flags, fixed_va);
   if (!bo) {
      /* A recently released fixed/replay VA may still be closing. Only the
       * allocation-failure path waits; never expose an address too early. */
      agx_bo_cache_evict_all(dev);
      agx_bo_cache_wait(dev);
      bo = dev->ops.bo_alloc(dev, size, align, flags, va_flags, fixed_va);
   }
   if (!bo)
      return NULL;

   agx_bo_finish_create(dev, bo, label);
   return bo;
}

static void
agx_bo_finish_create(struct agx_device *dev, struct agx_bo *bo,
                     const char *label)
{
   if (bo->shim_cpu_epoch)
      *bo->shim_cpu_epoch = 0; /* Cache reuse must opt in again. */
   bo->label = label;
   if (agx_apple9_debug_enabled(AGX_A9_DBG_BO_LABELS))
      fprintf(stderr, "AGX_BO_LABEL fd=%d handle=%u flags=%u size=%zu label=%s\n",
              dev->fd, bo->uapi_handle, bo->flags, bo->size, label);
   p_atomic_set(&bo->refcnt, 1);

   if (dev->debug & AGX_DBG_TRACE) {
      agx_bo_map(bo);
      agxdecode_track_alloc(dev->agxdecode, bo);
   }

   if (unlikely(dev->debug & (AGX_DBG_BODUMP | AGX_DBG_BODUMPVERBOSE))) {
      pthread_mutex_lock(&dev->bo_map_lock);
      agx_bo_dump_all_periodic(dev);
      pthread_mutex_unlock(&dev->bo_map_lock);
   }
}

/* Native DRM mappings are coherent; the remote shim needs an explicit pull. */
bool
agx_bo_sync_cpu_read(struct agx_bo *bo)
{
   typedef int (*read_fn)(int, uint32_t);
   read_fn read = (read_fn)dlsym(RTLD_DEFAULT, "asahi_m1n1_cpu_read");
   return !read || read(bo->dev->fd, bo->uapi_handle) == 0;
}

void
agx_bo_note_cpu_write(struct agx_bo *bo, uint64_t offset, uint64_t size)
{
   typedef void (*note_fn)(int, uint32_t, uint64_t, uint64_t);
   note_fn note = (note_fn)dlsym(RTLD_DEFAULT, "asahi_m1n1_cpu_written");
   if (note)
      note(bo->dev->fd, bo->uapi_handle, offset, size);
}
