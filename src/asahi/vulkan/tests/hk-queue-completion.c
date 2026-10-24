/* SPDX-License-Identifier: MIT */
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <xf86drm.h>
#include "../hk_queue_completion.h"
#include "util/libsync.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #x);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)

/* Mock the existing Linux interfaces, including immutable sync-file snapshots
 * and the distinction between aggregate and individual fence errors.
 */
struct mock_fence {
   int status, error_index;
   unsigned count;
};
static struct mock_fence fences[32768];
static struct mock_fence *handles[32768];
static struct {
   struct mock_fence *fence;
   bool open;
} files[32768];
static unsigned next_fd = 100, next_fence, creates, destroys, exports, closes;
static unsigned polls, infos, waits, transfers, live_fds, peak_fds;
static int fd_limit = 1024;
static int next_status = 1, export_error, info_error, poll_error, create_error;
static int transfer_error, wait_error;
static unsigned next_count = 1;
static bool alloc_fail;
static int realloc_fail_after = -1;

static struct mock_fence *
new_fence(void)
{
   CHECK(next_fence < ARRAY_SIZE(fences));
   struct mock_fence *f = &fences[next_fence++];
   *f = (struct mock_fence){next_status, -1, next_count};
   return f;
}

void *__real_realloc(void *, size_t);
void *__real_calloc(size_t, size_t);
void *__wrap_realloc(void *, size_t);
void *__wrap_calloc(size_t, size_t);
int __wrap_drmSyncobjCreate(int, uint32_t, uint32_t *);
int __wrap_drmSyncobjDestroy(int, uint32_t);
int __wrap_drmSyncobjTransfer(int, uint32_t, uint64_t, uint32_t, uint64_t, uint32_t);
int __wrap_drmSyncobjWait(int, uint32_t *, unsigned, int64_t, unsigned, uint32_t *);
int __wrap_drmSyncobjExportSyncFile(int, uint32_t, int *);
int __wrap_close(int);
int __wrap_poll(struct pollfd *, nfds_t, int);
int __wrap_ioctl(int, unsigned long, ...);
void *
__wrap_realloc(void *p, size_t size)
{
   if (realloc_fail_after == 0)
      return NULL;
   if (realloc_fail_after > 0)
      realloc_fail_after--;
   return alloc_fail ? NULL : __real_realloc(p, size);
}
void *
__wrap_calloc(size_t count, size_t size)
{
   return alloc_fail ? NULL : __real_calloc(count, size);
}
int
__wrap_drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle)
{
   CHECK(fd == 42 && !flags);
   if (create_error) {
      errno = create_error;
      return -1;
   }
   *handle = ++creates;
   CHECK(*handle < ARRAY_SIZE(handles));
   handles[*handle] = NULL;
   return 0;
}
int
__wrap_drmSyncobjDestroy(int fd, uint32_t handle)
{
   CHECK(fd == 42 && handle);
   destroys++;
   handles[handle] = NULL;
   return 0;
}
int
__wrap_drmSyncobjExportSyncFile(int fd, uint32_t handle, int *out)
{
   CHECK(fd == 42 && handle);
   if (export_error || live_fds == (unsigned)fd_limit) {
      errno = export_error ? export_error : EMFILE;
      return -1;
   }
   CHECK(next_fd < ARRAY_SIZE(files));
   *out = next_fd++;
   CHECK(handles[handle]);
   files[*out].fence = handles[handle];
   files[*out].open = true;
   exports++;
   live_fds++;
   peak_fds = MAX2(peak_fds, live_fds);
   return 0;
}
int
__wrap_drmSyncobjTransfer(int fd, uint32_t dst, uint64_t dst_point,
                         uint32_t src, uint64_t src_point, uint32_t flags)
{
   CHECK(fd == 42 && dst && src && !dst_point && !src_point && !flags);
   transfers++;
   if (transfer_error) {
      errno = transfer_error;
      return -1;
   }
   CHECK(handles[src]);
   handles[dst] = handles[src];
   return 0;
}
int
__wrap_drmSyncobjWait(int fd, uint32_t *objects, unsigned count, int64_t timeout,
                     unsigned flags, uint32_t *first)
{
   CHECK(fd == 42 && !timeout && !flags && count);
   waits++;
   if (wait_error) {
      errno = wait_error;
      wait_error = 0;
      return -1;
   }
   for (unsigned i = 0; i < count; ++i) {
      CHECK(handles[objects[i]]);
      if (handles[objects[i]]->status) {
         *first = i;
         return 0;
      }
   }
   errno = ETIME;
   return -1;
}
int
__wrap_close(int fd)
{
   CHECK(fd >= 100 && files[fd].open);
   files[fd].open = false;
   closes++;
   live_fds--;
   return 0;
}
int
__wrap_poll(struct pollfd *fds, nfds_t count, int timeout)
{
   CHECK(timeout == 0);
   polls++;
   if (poll_error) {
      errno = poll_error;
      poll_error = 0;
      return -1;
   }
   int ready = 0;
   for (unsigned i = 0; i < count; ++i) {
      fds[i].revents = 0;
      if (fds[i].fd < 0)
         continue;
      CHECK(files[fds[i].fd].open);
      if (files[fds[i].fd].fence->status) {
         fds[i].revents = POLLIN;
         ready++;
      }
   }
   return ready;
}
int
__wrap_ioctl(int fd, unsigned long request, ...)
{
   CHECK(request == SYNC_IOC_FILE_INFO && files[fd].open);
   infos++;
   if (info_error) {
      errno = info_error;
      return -1;
   }
   va_list ap;
   va_start(ap, request);
   struct sync_file_info *info = va_arg(ap, struct sync_file_info *);
   va_end(ap);
   unsigned count = files[fd].fence->count;
   if (info->num_fences && info->num_fences < count) {
      errno = EINVAL;
      return -1;
   }
   if (info->num_fences) {
      struct sync_fence_info *fences = (void *)(uintptr_t)info->sync_fence_info;
      for (unsigned i = 0; i < count; ++i)
         fences[i].status =
            (int)i == files[fd].fence->error_index ? -ECANCELED : files[fd].fence->status;
   }
   info->num_fences = count;
   info->status = files[fd].fence->status;
   return 0;
}

static void
submit(struct hk_queue_completion *c)
{
   CHECK(hk_queue_completion_begin(c) == 0);
   handles[c->syncobj] = new_fence();
   CHECK(hk_queue_completion_end(c, true) == 0);
}

static void
test_retirement(void)
{
   struct hk_queue_completion c;
   CHECK(hk_queue_completion_init(&c, 42) == 0);
   unsigned p = polls, i = infos, e = exports, cr = creates;
   for (unsigned n = 0; n < 1000; ++n)
      CHECK(hk_queue_completion_status(&c) == 1);
   CHECK(polls == p && infos == i && exports == e && creates == cr);

   next_status = 0;
   submit(&c);
   int first = next_fd - 1;
   for (unsigned n = 0; n < 1000; ++n)
      CHECK(hk_queue_completion_status(&c) == 0);
   CHECK(exports == e + 1 && infos == i && creates == cr);

   /* A later slice/submit finishes first. Keep the retirement frontier at
    * the earlier work, but never query the retired file again.
    */
   next_status = 1;
   submit(&c);
   CHECK(hk_queue_completion_status(&c) == 0);
   CHECK(c.completed == 0 && c.submitted == 2 && infos == i + 1);
   CHECK(hk_queue_completion_status(&c) == 0 && infos == i + 1);
   files[first].fence->status = 1;
   CHECK(hk_queue_completion_status(&c) == 1);
   CHECK(c.completed == 2 && c.submitted == 2);
   p = polls;
   i = infos;
   e = exports;
   for (unsigned n = 0; n < 1000; ++n)
      CHECK(hk_queue_completion_status(&c) == 1);
   CHECK(polls == p && infos == i && exports == e);

   /* The same storage and binary handle survive many completed submits. */
   struct hk_completion_fence *storage = c.pending;
   for (unsigned n = 0; n < 1000; ++n)
      submit(&c);
   CHECK(hk_queue_completion_status(&c) == 1);
   CHECK(c.pending == storage && creates == cr);
   hk_queue_completion_finish(&c);
}

static void
test_errors(void)
{
   struct hk_queue_completion c;
   CHECK(hk_queue_completion_init(&c, 42) == 0);
   next_status = 0;
   submit(&c);
   int first = next_fd - 1;
   next_status = 1;
   next_count = 129;
   submit(&c);
   int second = next_fd - 1;
   files[second].fence->error_index = 128;
   CHECK(hk_queue_completion_status(&c) == -ECANCELED);
   CHECK(c.completed == 0 && files[first].open);
   unsigned p = polls, i = infos;
   CHECK(hk_queue_completion_status(&c) == -ECANCELED);
   CHECK(hk_queue_completion_begin(&c) == -ECANCELED);
   CHECK(polls == p && infos == i);
   hk_queue_completion_finish(&c);
   CHECK(!files[first].open && !files[second].open);
   next_count = 1;

   CHECK(hk_queue_completion_init(&c, 42) == 0);
   next_status = 0;
   submit(&c);
   first = next_fd - 1;
   next_status = 1;
   submit(&c);
   files[first].fence->status = -EIO;
   CHECK(hk_queue_completion_status(&c) == -EIO);
   hk_queue_completion_finish(&c);
}

static void
test_failures(void)
{
   struct hk_queue_completion c;
   alloc_fail = true;
   CHECK(hk_queue_completion_init(&c, 42) == -ENOMEM);
   hk_queue_completion_finish(&c);
   alloc_fail = false;
   create_error = ENOMEM;
   CHECK(hk_queue_completion_init(&c, 42) == -ENOMEM);
   hk_queue_completion_finish(&c);
   create_error = 0;
   CHECK(hk_queue_completion_init(&c, 42) == 0);
   alloc_fail = true;
   CHECK(hk_queue_completion_begin(&c) == -ENOMEM);
   alloc_fail = false;
   CHECK(hk_queue_completion_begin(&c) == 0);
   CHECK(hk_queue_completion_end(&c, false) == -EIO);
   CHECK(hk_queue_completion_status(&c) == -EIO);
   hk_queue_completion_finish(&c);

   CHECK(hk_queue_completion_init(&c, 42) == 0);
   CHECK(hk_queue_completion_begin(&c) == 0);
   export_error = EMFILE;
   CHECK(hk_queue_completion_end(&c, true) == -EMFILE);
   export_error = 0;
   CHECK(hk_queue_completion_status(&c) == -EMFILE);
   hk_queue_completion_finish(&c);

   CHECK(hk_queue_completion_init(&c, 42) == 0);
   next_status = 1;
   submit(&c);
   poll_error = EINTR;
   CHECK(hk_queue_completion_status(&c) == 1);
   submit(&c);
   info_error = EIO;
   CHECK(hk_queue_completion_status(&c) == -EIO);
   info_error = 0;
   hk_queue_completion_finish(&c);

   CHECK(hk_queue_completion_init(&c, 42) == 0);
   next_count = 129;
   submit(&c);
   alloc_fail = true;
   CHECK(hk_queue_completion_status(&c) == -ENOMEM);
   alloc_fail = false;
   next_count = 1;
   hk_queue_completion_finish(&c);
}

struct concurrent {
   struct hk_queue_completion completion;
   atomic_bool done;
};
static void *
check_thread(void *arg)
{
   struct concurrent *c = arg;
   do {
      CHECK(hk_queue_completion_status(&c->completion) >= 0);
   } while (!atomic_load(&c->done));
   return NULL;
}
static void
test_concurrent(void)
{
   struct concurrent c = {.done = ATOMIC_VAR_INIT(false)};
   CHECK(hk_queue_completion_init(&c.completion, 42) == 0);
   pthread_t thread;
   CHECK(pthread_create(&thread, NULL, check_thread, &c) == 0);
   next_status = 1;
   for (unsigned n = 0; n < 2000; ++n)
      submit(&c.completion);
   atomic_store(&c.done, true);
   CHECK(pthread_join(thread, NULL) == 0);
   CHECK(hk_queue_completion_status(&c.completion) == 1);
   CHECK(c.completion.completed == 2000);
   hk_queue_completion_finish(&c.completion);
}

static void
test_deep_queue(void)
{
   for (unsigned error_case = 0; error_case < 2; ++error_case) {
      struct hk_queue_completion c;
      CHECK(hk_queue_completion_init(&c, 42) == 0);
      CHECK(live_fds == 0);
      peak_fds = 0;
      fd_limit = HK_QUEUE_COMPLETION_MAX_FDS + 1;
      next_status = 0;
      unsigned e = exports, tr = transfers;
      for (unsigned i = 0; i < 10000; ++i) {
         submit(&c);
         CHECK(c.open_fds <= HK_QUEUE_COMPLETION_MAX_FDS);
         CHECK(live_fds <= HK_QUEUE_COMPLETION_MAX_FDS);
      }
      CHECK(c.submitted == 10000 && c.completed == 0);
      CHECK(exports - e == HK_QUEUE_COMPLETION_MAX_FDS);
      CHECK(transfers - tr == 10000 - HK_QUEUE_COMPLETION_MAX_FDS);
      CHECK(peak_fds == HK_QUEUE_COMPLETION_MAX_FDS);

      if (error_case) {
         /* A later overflow fence fails while all older fds stay pending.
          * Preserve the original snapshot even if a newer fence, potentially
          * on the same kernel context, has replaced the output syncobj.
          * The error may be wrapped beyond the initial info-array capacity.
          */
         struct mock_fence *f = handles[c.pending[1000].syncobj];
         CHECK(f && f != handles[c.syncobj]);
         f->count = 129;
         f->status = 1;
         f->error_index = 128;
         CHECK(hk_queue_completion_status(&c) == -ECANCELED);
         CHECK(c.completed == 0 && c.open_fds == HK_QUEUE_COMPLETION_MAX_FDS);
         CHECK(peak_fds == HK_QUEUE_COMPLETION_MAX_FDS + 1);
         CHECK(hk_queue_completion_status(&c) == -ECANCELED);
      } else {
         for (unsigned i = 0; i < 10000; ++i) {
            struct hk_completion_fence *p = &c.pending[i];
            struct mock_fence *f = p->fd >= 0 ? files[p->fd].fence : handles[p->syncobj];
            f->status = 1;
         }
         wait_error = EINTR;
         CHECK(hk_queue_completion_status(&c) == 1);
         CHECK(c.completed == 10000 && c.open_fds == 0 && live_fds == 0);
         CHECK(exports - e == 10000);
         unsigned p = polls, w = waits, in = infos;
         CHECK(hk_queue_completion_status(&c) == 1);
         CHECK(polls == p && waits == w && infos == in);
      }
      hk_queue_completion_finish(&c);
      CHECK(live_fds == 0);
      fd_limit = 1024;
   }
}

static void
test_overflow_failures(void)
{
   /* Each independently grown array must leave the old capacity usable when
    * a later allocation fails, and partial init must still clean up.
    */
   for (unsigned i = 0; i < 3; ++i) {
      struct hk_queue_completion c;
      CHECK(hk_queue_completion_init(&c, 42) == 0);
      realloc_fail_after = i;
      CHECK(hk_queue_completion_begin(&c) == -ENOMEM);
      realloc_fail_after = -1;
      CHECK(hk_queue_completion_begin(&c) == 0);
      CHECK(hk_queue_completion_end(&c, false) == -EIO);
      hk_queue_completion_finish(&c);
   }

   for (unsigned failure = 0; failure < 4; ++failure) {
      struct hk_queue_completion c;
      CHECK(hk_queue_completion_init(&c, 42) == 0);
      next_status = 0;
      for (unsigned i = 0; i < HK_QUEUE_COMPLETION_MAX_FDS; ++i)
         submit(&c);
      if (failure == 0) {
         create_error = ENOMEM;
         CHECK(hk_queue_completion_begin(&c) == -ENOMEM);
         create_error = 0;
         CHECK(c.submitted == HK_QUEUE_COMPLETION_MAX_FDS);
      } else if (failure == 1) {
         CHECK(hk_queue_completion_begin(&c) == 0);
         handles[c.syncobj] = new_fence();
         transfer_error = EIO;
         CHECK(hk_queue_completion_end(&c, true) == -EIO);
         transfer_error = 0;
      } else {
         submit(&c);
         if (failure == 2) {
            wait_error = EIO;
         } else {
            handles[c.pending[HK_QUEUE_COMPLETION_MAX_FDS].syncobj]->status = 1;
            export_error = EIO;
         }
         CHECK(hk_queue_completion_status(&c) == -EIO);
         export_error = 0;
      }
      hk_queue_completion_finish(&c);
      CHECK(live_fds == 0);
   }
}

static void *
try_status_thread(void *arg)
{
   CHECK(hk_queue_completion_status(arg) == 1);
   return NULL;
}

static void
test_contended_status(void)
{
   struct hk_queue_completion c;
   CHECK(hk_queue_completion_init(&c, 42) == 0);
   CHECK(hk_queue_completion_begin(&c) == 0);
   unsigned p = polls, w = waits, in = infos;
   pthread_t thread;
   CHECK(pthread_create(&thread, NULL, try_status_thread, &c) == 0);
   /* Join while submission still owns the mutex. A blocking status path
    * would deadlock and fail the test's timeout, rather than pass silently.
    */
   CHECK(pthread_join(thread, NULL) == 0);
   CHECK(polls == p && waits == w && infos == in);
   CHECK(hk_queue_completion_end(&c, false) == -EIO);
   CHECK(hk_queue_completion_status(&c) == -EIO);
   hk_queue_completion_finish(&c);
}

int
main(void)
{
   test_retirement();
   test_errors();
   test_failures();
   test_concurrent();
   test_deep_queue();
   test_overflow_failures();
   test_contended_status();
   CHECK(creates == destroys && exports == closes && live_fds == 0);
   puts(
      "Queue retirement, 10k bounded-fd submits, error retention, failures, trylock: PASS");
   return 0;
}
