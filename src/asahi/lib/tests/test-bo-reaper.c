/*
 * SPDX-License-Identifier: MIT
 *
 * Exercise the real cache, refcount, import and reaper with a blocking DRM
 * close. No GPU is needed. Checks stay enabled in release builds.
 */
#include "agx_device.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

static struct {
   struct agx_device dev;
   pthread_mutex_t lock;
   pthread_cond_t cond;
   thrd_t main;
   bool hold, synchronous, fail_thread;
   unsigned block, entered[512], closed[512], failed;
   unsigned next_handle, alloc_calls, fail_allocs, import_calls;
   int import_fd, other_fd;
   unsigned import_handle, other_handle;
   unsigned wakes;
   pid_t owner;
} test;

int __wrap_drmIoctl(int fd, unsigned long request, void *arg);
int __wrap_drmPrimeFDToHandle(int fd, int prime, uint32_t *handle);
int __wrap_u_thread_create(thrd_t *thread, int (*run)(void *), void *data);
int __real_u_thread_create(thrd_t *thread, int (*run)(void *), void *data);
int __wrap_pthread_cond_signal(pthread_cond_t *cond);
int __real_pthread_cond_signal(pthread_cond_t *cond);
void __wrap_agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind);
void __real_agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind);

int
__wrap_pthread_cond_signal(pthread_cond_t *cond)
{
   if (cond == &test.dev.bo_cache.cond)
      p_atomic_inc(&test.wakes);
   return __real_pthread_cond_signal(cond);
}

void
__wrap_agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind)
{
   if (getpid() != test.owner)
      _exit(91);
   __real_agx_va_free(dev, va, unbind);
}

int
__wrap_u_thread_create(thrd_t *thread, int (*run)(void *), void *data)
{
   return test.fail_thread ? thrd_error
                           : __real_u_thread_create(thread, run, data);
}

static struct timespec
deadline(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_REALTIME, &ts);
   ts.tv_sec += 5;
   return ts;
}

static void
wait_count(unsigned *value, unsigned count)
{
   struct timespec until = deadline();
   pthread_mutex_lock(&test.lock);
   while (*value < count)
      CHECK(!pthread_cond_timedwait(&test.cond, &test.lock, &until));
   pthread_mutex_unlock(&test.lock);
}

int
__wrap_drmIoctl(int fd, unsigned long request, void *arg)
{
   if (getpid() != test.owner)
      _exit(92);
   CHECK(request == DRM_IOCTL_GEM_CLOSE);
   unsigned handle = ((struct drm_gem_close *)arg)->handle;
   CHECK(handle < ARRAY_SIZE(test.closed));
   CHECK(thrd_equal(thrd_current(), test.main) == test.synchronous);
   pthread_mutex_lock(&test.lock);
   test.entered[handle]++;
   pthread_cond_broadcast(&test.cond);
   struct timespec until = deadline();
   while (test.hold && handle == test.block)
      CHECK(!pthread_cond_timedwait(&test.cond, &test.lock, &until));
   test.closed[handle]++;
   if (test.import_handle == handle)
      test.import_handle = 300; /* A subsequent import gets a fresh handle. */
   bool fail = test.failed == handle;
   pthread_cond_broadcast(&test.cond);
   pthread_mutex_unlock(&test.lock);
   errno = EIO;
   return fail ? -1 : 0;
}

int
__wrap_drmPrimeFDToHandle(int fd, int prime, uint32_t *handle)
{
   pthread_mutex_lock(&test.lock);
   CHECK(prime == test.import_fd || prime == test.other_fd);
   *handle = prime == test.import_fd ? test.import_handle : test.other_handle;
   test.import_calls++;
   pthread_cond_broadcast(&test.cond);
   pthread_mutex_unlock(&test.lock);
   return 0;
}

static int
bind_bo(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
        uint32_t count)
{
   return 0;
}

static struct agx_bo *
new_bo(unsigned handle, size_t size, enum agx_bo_flags flags,
       enum agx_va_flags va_flags, uint64_t addr)
{
   struct agx_device *dev = &test.dev;
   pthread_mutex_lock(&dev->bo_map_lock);
   struct agx_bo *bo = agx_lookup_bo(dev, handle);
   CHECK(!bo->size);
   *bo = (struct agx_bo){
      .dev = dev,
      .handle = handle,
      .uapi_handle = handle,
      .size = size,
      .align = 16384,
      .flags = flags,
      .prime_fd = -1,
      .label = "test",
      .refcnt = 1,
   };
   dev->max_handle = MAX2(dev->max_handle, handle);
   pthread_mutex_unlock(&dev->bo_map_lock);
   bo->va = agx_va_alloc(dev, size, 16384, va_flags, addr);
   return bo;
}

static struct agx_bo *
alloc_bo(struct agx_device *dev, size_t size, size_t align,
         enum agx_bo_flags flags, enum agx_va_flags va_flags, uint64_t addr)
{
   pthread_mutex_lock(&test.lock);
   test.alloc_calls++;
   pthread_cond_broadcast(&test.cond);
   bool fail = test.fail_allocs > 0;
   if (fail)
      test.fail_allocs--;
   unsigned handle = test.next_handle++;
   pthread_mutex_unlock(&test.lock);
   if (fail)
      return NULL;
   struct agx_bo *bo = new_bo(handle, size, flags, va_flags, addr);
   bo->refcnt = 0;
   if (!bo->va) {
      agx_bo_free(dev, bo);
      return NULL;
   }
   return bo;
}

static void
init(bool start)
{
   memset(&test, 0, sizeof(test));
   pthread_mutex_init(&test.lock, NULL);
   pthread_cond_init(&test.cond, NULL);
   test.main = thrd_current();
   test.owner = getpid();
   test.synchronous = !start;
   test.next_handle = 100;
   test.import_fd = test.other_fd = -1;
   struct agx_device *dev = &test.dev;
   pthread_mutex_init(&dev->bo_map_lock, NULL);
   util_sparse_array_init(&dev->bo_map, sizeof(struct agx_bo), 512);
   simple_mtx_init(&dev->vma_lock, mtx_plain);
   util_vma_heap_init(&dev->main_heap, 0x1000000000ull, 1ull << 28);
   util_vma_heap_init(&dev->usc_heap, 0x2000000000ull, 1ull << 28);
   dev->guard_size = 16384;
   dev->ops.bo_alloc = alloc_bo;
   dev->ops.bo_bind = bind_bo;
   agx_bo_cache_init(dev);
   if (start)
      CHECK(agx_bo_cache_start(dev));
}

static void
finish(void)
{
   agx_bo_cache_finish(&test.dev);
   CHECK(!test.dev.bo_cache.size);
   for (unsigned i = 0; i <= test.dev.max_handle; ++i)
      CHECK(!agx_lookup_bo(&test.dev, i)->size);
   util_sparse_array_finish(&test.dev.bo_map);
   util_vma_heap_finish(&test.dev.main_heap);
   util_vma_heap_finish(&test.dev.usc_heap);
   simple_mtx_destroy(&test.dev.vma_lock);
   pthread_mutex_destroy(&test.dev.bo_map_lock);
   pthread_cond_destroy(&test.cond);
   pthread_mutex_destroy(&test.lock);
}

static void
block_close(unsigned handle)
{
   pthread_mutex_lock(&test.lock);
   test.block = handle;
   test.hold = true;
   pthread_mutex_unlock(&test.lock);
}

static void
release_close(void)
{
   pthread_mutex_lock(&test.lock);
   test.hold = false;
   pthread_cond_broadcast(&test.cond);
   pthread_mutex_unlock(&test.lock);
}

static void
test_blocked_close(void)
{
   init(true);
   struct agx_device *dev = &test.dev;
   struct agx_bo *old = new_bo(10, 16384, 0, 0, 0);
   uint64_t addr = old->va->addr;
   block_close(10);
   agx_bo_unreference(dev, old);
   agx_bo_cache_evict_all(dev);
   wait_count(&test.entered[10], 1);

   /* Allocation, last unreference, cache reuse and trim all progress while
    * the worker is inside GEM_CLOSE. VA reuse must not progress yet. */
   CHECK(!agx_va_alloc(dev, 16384, 16384, AGX_VA_FIXED, addr));
   struct agx_bo *live = new_bo(11, 32768, 0, 0, 0);
   agx_bo_unreference(dev, live);
   CHECK(agx_bo_create(dev, 32768, 0, 0, "reuse") == live);
   agx_bo_reference(live);
   agx_bo_unreference(dev, live);
   CHECK(live->refcnt == 1);
   agx_bo_unreference(dev, live);
   agx_bo_cache_evict_all(dev);
   CHECK(!test.closed[10] && !test.entered[11]);

   /* The kernel can allocate the old handle before close returns. */
   struct agx_bo *reused = new_bo(10, 65536, 0, 0, 0);
   release_close();
   agx_bo_cache_wait(dev);
   CHECK(reused->refcnt == 1 && reused->size == 65536);
   CHECK(test.closed[10] == 1 && test.closed[11] == 1);
   struct agx_va *va = agx_va_alloc(dev, 16384, 16384, AGX_VA_FIXED, addr);
   CHECK(va);
   agx_va_free(dev, va, false);
   agx_bo_unreference(dev, reused);
   finish();
   CHECK(test.closed[10] == 2);
}

static void
test_idle_expiry(void)
{
   init(true);
   struct agx_bo *bo = new_bo(12, 16384, 0, 0, 0);
   agx_bo_unreference(&test.dev, bo);
   pthread_mutex_lock(&test.dev.bo_cache.lock);
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   bo->last_used = now.tv_sec - 2;
   pthread_cond_signal(&test.dev.bo_cache.cond);
   pthread_mutex_unlock(&test.dev.bo_cache.lock);
   /* No further allocation, submission or release wakes the expiry timer. */
   wait_count(&test.entered[12], 1);
   finish();
   CHECK(test.closed[12] == 1);
}

static int
prime_fd(void)
{
   int fd = memfd_create("reaper-import", MFD_CLOEXEC);
   CHECK(fd >= 0 && !ftruncate(fd, 16384));
   return fd;
}

static void
test_resurrection(void)
{
   init(true);
   block_close(10);
   struct agx_bo *first = new_bo(10, 16384, AGX_BO_SHARED, 0, 0);
   agx_bo_unreference(&test.dev, first);
   wait_count(&test.entered[10], 1);

   test.import_fd = prime_fd();
   test.import_handle = 20;
   struct agx_bo *bo =
      new_bo(20, 16384, AGX_BO_SHARED | AGX_BO_SHAREABLE, 0, 0);
   bo->prime_fd = dup(test.import_fd);
   agx_bo_unreference(&test.dev, bo);
   CHECK(agx_bo_import(&test.dev, test.import_fd) == bo);
   CHECK(bo->refcnt == 1 && !bo->retired);
   release_close();
   agx_bo_cache_wait(&test.dev);
   CHECK(!test.entered[20]);
   agx_bo_unreference(&test.dev, bo);
   finish();
   CHECK(test.closed[20] == 1);
   close(test.import_fd);
}

struct operation {
   bool done;
   int fd;
   uint64_t fixed;
   struct agx_bo *bo;
};

static int
operate(void *data)
{
   struct operation *op = data;
   if (op->fd >= 0)
      op->bo = agx_bo_import(&test.dev, op->fd);
   else if (op->fixed)
      op->bo = agx_bo_create_va(&test.dev, 16384, 0, 0, AGX_VA_FIXED, op->fixed,
                                "fixed retry");
   else
      op->bo = agx_bo_create(&test.dev, 65536, 0, 0, "OOM retry");
   pthread_mutex_lock(&test.lock);
   op->done = true;
   pthread_cond_broadcast(&test.cond);
   pthread_mutex_unlock(&test.lock);
   return 0;
}

static void
test_import_during_close(void)
{
   init(true);
   test.import_fd = prime_fd();
   test.other_fd = prime_fd();
   test.import_handle = 20;
   test.other_handle = 21;
   unsigned flags = AGX_BO_SHARED | AGX_BO_SHAREABLE;
   struct agx_bo *old = new_bo(20, 16384, flags, 0, 0);
   old->prime_fd = dup(test.import_fd);
   struct agx_bo *other = new_bo(21, 16384, flags, 0, 0);
   other->prime_fd = dup(test.other_fd);
   block_close(20);
   agx_bo_unreference(&test.dev, old);
   wait_count(&test.entered[20], 1);
   struct operation op = {.fd = test.import_fd};
   thrd_t thread;
   CHECK(u_thread_create(&thread, operate, &op) == thrd_success);
   wait_count(&test.import_calls, 1);

   CHECK(agx_bo_import(&test.dev, test.other_fd) == other);
   agx_bo_unreference(&test.dev, other);
   pthread_mutex_lock(&test.lock);
   CHECK(!op.done);
   pthread_mutex_unlock(&test.lock);
   release_close();
   thrd_join(thread, NULL);
   CHECK(op.bo && op.bo->handle == 300);
   CHECK(test.import_calls >= 3);
   agx_bo_unreference(&test.dev, op.bo);
   agx_bo_unreference(&test.dev, other);
   finish();
   CHECK(test.closed[20] == 1 && test.closed[21] == 1 && test.closed[300] == 1);
   close(test.import_fd);
   close(test.other_fd);
}

static void
test_allocation_retry(bool fixed)
{
   init(true);
   struct agx_bo *bo = new_bo(25, 16384, 0, fixed ? AGX_VA_FIXED : 0,
                              fixed ? 0x1000000000ull : 0);
   CHECK(bo->va);
   struct operation op = {.fd = -1, .fixed = fixed ? bo->va->addr : 0};
   if (!fixed)
      test.fail_allocs = 1;
   block_close(25);
   agx_bo_unreference(&test.dev, bo);
   agx_bo_cache_evict_all(&test.dev);
   wait_count(&test.entered[25], 1);
   thrd_t thread;
   CHECK(u_thread_create(&thread, operate, &op) == thrd_success);
   wait_count(&test.alloc_calls, 1);
   pthread_mutex_lock(&test.lock);
   CHECK(!op.done);
   pthread_mutex_unlock(&test.lock);
   release_close();
   thrd_join(thread, NULL);
   CHECK(op.bo && test.closed[25] == 1 && test.alloc_calls == 2);
   if (fixed)
      CHECK(op.bo->va->addr == op.fixed);
   agx_bo_unreference(&test.dev, op.bo);
   finish();
}

static void
test_cache_reuse_and_backlog(void)
{
   init(true);
   block_close(10);
   agx_bo_unreference(&test.dev, new_bo(10, 16384, AGX_BO_SHARED, 0, 0));
   wait_count(&test.entered[10], 1);
   struct agx_bo *a = new_bo(11, 16384, 0, 0, 0);
   struct agx_bo *b = new_bo(12, 16384, 0, 0, 0);
   agx_bo_unreference(&test.dev, a);
   agx_bo_unreference(&test.dev, b);
   pthread_mutex_lock(&test.dev.bo_cache.lock);
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   a->last_used = b->last_used = now.tv_sec - 5;
   pthread_mutex_unlock(&test.dev.bo_cache.lock);
   CHECK(agx_bo_create(&test.dev, 16384, 0, 0, "MRU") == b);
   CHECK(agx_bo_create(&test.dev, 16384, 0, 0, "next") == a);
   agx_bo_unreference(&test.dev, a);
   agx_bo_unreference(&test.dev, b);
   /* More than the old 32-BO eviction batch, including uncached frees. */
   for (unsigned i = 30; i < 90; ++i)
      agx_bo_unreference(&test.dev, new_bo(i, 16384, AGX_BO_SHARED, 0, 0));
   agx_bo_cache_evict_all(&test.dev);
   release_close();
   finish();
   CHECK(test.closed[11] == 1 && test.closed[12] == 1);
   for (unsigned i = 30; i < 90; ++i)
      CHECK(test.closed[i] == 1);
}

static void
test_close_failure_and_bootstrap(void)
{
   init(false);
   test.fail_thread = true;
   CHECK(!agx_bo_cache_start(&test.dev));
   test.failed = 40;
   struct agx_bo *bo = new_bo(40, 16384, 0, 0, 0);
   uint64_t addr = bo->va->addr;
   agx_bo_unreference(&test.dev, bo);
   agx_bo_cache_evict_all(&test.dev);
   agx_bo_cache_wait(&test.dev);
   CHECK(test.closed[40] == 1);
   CHECK(!agx_va_alloc(&test.dev, 16384, 16384, AGX_VA_FIXED, addr));
   /* Free an allocation that failed before binding a VA. */
   bo = new_bo(41, 16384, 0, AGX_VA_FIXED, addr);
   CHECK(!bo->va);
   bo->refcnt = 0;
   agx_bo_free(&test.dev, bo);
   finish();
   CHECK(test.closed[41] == 1);
}

static void
test_invalid_import(void)
{
   init(true);
   int fd = memfd_create("invalid", MFD_CLOEXEC);
   CHECK(fd >= 0);
   CHECK(!agx_bo_import(&test.dev, fd));
   CHECK(!ftruncate(fd, 1));
   CHECK(!agx_bo_import(&test.dev, fd));
   CHECK(!agx_bo_import(&test.dev, -1));
   CHECK(!test.import_calls);
   close(fd);
   finish();
}

static void
test_transition_wakes(void)
{
   init(true);
   struct agx_device *dev = &test.dev;
   block_close(10);
   agx_bo_unreference(dev, new_bo(10, 16384, AGX_BO_SHARED, 0, 0));
   wait_count(&test.entered[10], 1);
   unsigned wakes = p_atomic_read(&test.wakes);

   agx_bo_unreference(dev, new_bo(11, 16384, 0, 0, 0));
   CHECK(p_atomic_read(&test.wakes) == ++wakes);
   agx_bo_unreference(dev, new_bo(12, 16384, 0, 0, 0));
   CHECK(p_atomic_read(&test.wakes) == wakes);
   agx_bo_unreference(dev, new_bo(13, 16384, AGX_BO_SHARED, 0, 0));
   CHECK(p_atomic_read(&test.wakes) == ++wakes);
   agx_bo_unreference(dev, new_bo(14, 16384, AGX_BO_SHARED, 0, 0));
   CHECK(p_atomic_read(&test.wakes) == wakes);
   agx_bo_cache_evict_all(dev);
   agx_bo_cache_evict_all(dev);
   CHECK(p_atomic_read(&test.wakes) == wakes);

   release_close();
   agx_bo_cache_wait(dev);
   block_close(15);
   agx_bo_unreference(dev, new_bo(15, 16384, AGX_BO_SHARED, 0, 0));
   wait_count(&test.entered[15], 1);
   agx_bo_unreference(dev, new_bo(16, 16384, 0, 0, 0));
   wakes = p_atomic_read(&test.wakes);
   agx_bo_cache_evict_all(dev);
   CHECK(p_atomic_read(&test.wakes) == wakes + 1);
   release_close();
   finish();
}

static void
test_fork_during_close(void)
{
   init(true);
   struct agx_device *dev = &test.dev;
   /* A second registered device also exercises an idle worker's inherited
    * condvar and removal of more than one device from the atfork list. */
   struct agx_device idle = {0};
   pthread_mutex_init(&idle.bo_map_lock, NULL);
   agx_bo_cache_init(&idle);
   CHECK(agx_bo_cache_start(&idle));

   block_close(10);
   agx_bo_unreference(dev, new_bo(10, 16384, AGX_BO_SHARED, 0, 0));
   wait_count(&test.entered[10], 1);
   struct agx_bo *cached = new_bo(11, 16384, 0, 0, 0);
   struct agx_bo *shared = new_bo(12, 16384, AGX_BO_SHARED, 0, 0);
   struct agx_bo *ordinary = new_bo(13, 32768, 0, 0, 0);
   uint64_t cached_addr = cached->va->addr, shared_addr = shared->va->addr;
   agx_bo_unreference(dev, cached);

   pid_t child = fork();
   CHECK(child >= 0);
   if (!child) {
      alarm(3);
      CHECK(!pthread_mutex_trylock(&dev->bo_map_lock));
      pthread_mutex_unlock(&dev->bo_map_lock);
      CHECK(!pthread_mutex_trylock(&dev->bo_cache.lock));
      pthread_mutex_unlock(&dev->bo_cache.lock);
      CHECK(!agx_bo_cache_fetch(dev, 16384, 16384, 0, true));
      agx_bo_unreference(dev, shared);
      agx_bo_unreference(dev, ordinary);
      CHECK(!shared->size && !ordinary->size);
      agx_bo_cache_wait(dev);
      agx_bo_cache_finish(dev);
      agx_bo_cache_wait(&idle);
      agx_bo_cache_finish(&idle);
      /* Any ioctl or VA free in this child exits with 91/92 above. */
      _exit(0);
   }

   int status;
   CHECK(waitpid(child, &status, 0) == child);
   CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
   CHECK(!test.closed[10]);
   CHECK(shared->refcnt == 1 && ordinary->refcnt == 1);
   CHECK(!agx_va_alloc(dev, 16384, 16384, AGX_VA_FIXED, cached_addr));
   CHECK(!agx_va_alloc(dev, 16384, 16384, AGX_VA_FIXED, shared_addr));
   agx_bo_unreference(dev, shared);
   agx_bo_unreference(dev, ordinary);
   agx_bo_cache_evict_all(dev);
   release_close();
   agx_bo_cache_wait(dev);
   for (unsigned i = 10; i <= 13; ++i)
      CHECK(test.closed[i] == 1);
   struct agx_va *va = agx_va_alloc(dev, 16384, 16384, AGX_VA_FIXED,
                                   cached_addr);
   CHECK(va);
   agx_va_free(dev, va, false);
   va = agx_va_alloc(dev, 16384, 16384, AGX_VA_FIXED, shared_addr);
   CHECK(va);
   agx_va_free(dev, va, false);
   agx_bo_cache_finish(&idle);
   pthread_mutex_destroy(&idle.bo_map_lock);
   finish();
}

int
main(void)
{
   alarm(30);
   test_blocked_close();
   test_idle_expiry();
   test_resurrection();
   test_import_during_close();
   test_allocation_retry(false);
   test_allocation_retry(true);
   test_cache_reuse_and_backlog();
   test_close_failure_and_bootstrap();
   test_invalid_import();
   test_transition_wakes();
   test_fork_during_close();
   puts("BO reaper: 11 cases passed (including fork and failed worker start)");
   return 0;
}
