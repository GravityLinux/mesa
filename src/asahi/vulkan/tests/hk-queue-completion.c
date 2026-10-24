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
static struct {
   int status, error_index;
   unsigned count;
   bool open;
} files[8192];
static unsigned next_fd = 100, creates, destroys, exports, closes, polls, infos;
static int next_status = 1, export_error, info_error, poll_error, create_error;
static unsigned next_count = 1;
static bool alloc_fail;

void *__real_realloc(void *, size_t);
void *__real_calloc(size_t, size_t);
void *__wrap_realloc(void *, size_t);
void *__wrap_calloc(size_t, size_t);
int __wrap_drmSyncobjCreate(int, uint32_t, uint32_t *);
int __wrap_drmSyncobjDestroy(int, uint32_t);
int __wrap_drmSyncobjExportSyncFile(int, uint32_t, int *);
int __wrap_close(int);
int __wrap_poll(struct pollfd *, nfds_t, int);
int __wrap_ioctl(int, unsigned long, ...);
void *
__wrap_realloc(void *p, size_t size)
{
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
   return 0;
}
int
__wrap_drmSyncobjDestroy(int fd, uint32_t handle)
{
   CHECK(fd == 42 && handle);
   destroys++;
   return 0;
}
int
__wrap_drmSyncobjExportSyncFile(int fd, uint32_t handle, int *out)
{
   CHECK(fd == 42 && handle);
   if (export_error) {
      errno = export_error;
      return -1;
   }
   CHECK(next_fd < ARRAY_SIZE(files));
   *out = next_fd++;
   files[*out].status = next_status;
   files[*out].error_index = -1;
   files[*out].count = next_count;
   files[*out].open = true;
   exports++;
   return 0;
}
int
__wrap_close(int fd)
{
   CHECK(fd >= 100 && files[fd].open);
   files[fd].open = false;
   closes++;
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
      if (files[fds[i].fd].status) {
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
   unsigned count = files[fd].count;
   if (info->num_fences && info->num_fences < count) {
      errno = EINVAL;
      return -1;
   }
   if (info->num_fences) {
      struct sync_fence_info *fences = (void *)(uintptr_t)info->sync_fence_info;
      for (unsigned i = 0; i < count; ++i)
         fences[i].status =
            (int)i == files[fd].error_index ? -ECANCELED : files[fd].status;
   }
   info->num_fences = count;
   info->status = files[fd].status;
   return 0;
}

static void
submit(struct hk_queue_completion *c)
{
   CHECK(hk_queue_completion_begin(c) == 0);
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
   files[first].status = 1;
   CHECK(hk_queue_completion_status(&c) == 1);
   CHECK(c.completed == 2 && c.submitted == 2);
   p = polls;
   i = infos;
   e = exports;
   for (unsigned n = 0; n < 1000; ++n)
      CHECK(hk_queue_completion_status(&c) == 1);
   CHECK(polls == p && infos == i && exports == e);

   /* The same storage and binary handle survive many completed submits. */
   struct pollfd *storage = c.pending;
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
   files[second].error_index = 128;
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
   files[first].status = -EIO;
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

int
main(void)
{
   test_retirement();
   test_errors();
   test_failures();
   test_concurrent();
   CHECK(creates == destroys && exports == closes);
   puts(
      "Queue retirement, error retention, reuse, failures, concurrency: PASS");
   return 0;
}
