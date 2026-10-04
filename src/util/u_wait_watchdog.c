/* SPDX-License-Identifier: MIT
 * Test-build diagnostics. No periodic GPU queries, per-record I/O, or changes
 * to application wait timeouts. A separate thread diagnoses permanent waits.
 */
#include "u_wait_watchdog.h"

#ifdef HK_WAIT_WATCHDOG
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define WAIT_NS UINT64_C(5000000000)
#define HISTORY 16
#define SIGNALS 64
#define IOCTLS 64

struct submission {
   uint64_t id, time_ns, return_ns;
   int result, error;
   unsigned wait_count, signal_count;
   uint32_t completion_handle;
   uint64_t completion_point;
   struct { uint64_t object, point; } waits[SIGNALS], signals[SIGNALS];
   bool returned, completion_observed;
};

struct queue_history {
   struct queue_history *next;
   int fd;
   uint64_t queue;
   unsigned next_entry;
   struct submission entries[HISTORY];
};

struct ioctl_result {
   const char *name;
   uint64_t time_ns, tid;
   int fd, result, error;
};

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t wake;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_t thread;
static bool started, stopping;
static struct u_watchdog_wait *active;
static struct queue_history *queues;
static struct ioctl_result ioctls[IOCTLS];
static uint64_t next_id;
static unsigned next_ioctl;
static _Thread_local uint64_t cached_tid;

static uint64_t
thread_id(void)
{
   if (!cached_tid)
      cached_tid = syscall(SYS_gettid);
   return cached_tid;
}

static uint64_t
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static uint64_t
read_item(const void *array, size_t stride, unsigned i, bool small)
{
   if (!array)
      return 0;
   uint64_t value = 0;
   memcpy(&value, (const char *)array + stride * i, small ? 4 : 8);
   return value;
}

static void
print_objects(FILE *f, struct u_watchdog_objects o)
{
   for (unsigned i = 0; i < o.count; ++i)
      fprintf(f, "  object=0x%" PRIx64 " point=%" PRIu64 "\n",
              read_item(o.objects, o.object_stride, i, o.object32),
              read_item(o.points, o.point_stride, i, false));
}

/* Called with lock held: copy everything before allowing the waiting thread
 * to return and release the borrowed object arrays. File I/O is outside lock.
 */
static char *
snapshot(struct u_watchdog_wait *w, size_t *size)
{
   char *buffer = NULL;
   FILE *f = open_memstream(&buffer, size);
   if (!f)
      return NULL;
   fprintf(f, "HK LONG WAIT pid=%d tid=%" PRIu64 " wait=%s age_ms=%" PRIu64
           " start_ns=%" PRIu64 " fd=%d object=0x%" PRIx64
           " point=%" PRIu64 " flags=0x%" PRIx64 " deadline_ns=%" PRIu64 "\n",
           getpid(), w->tid, w->name, (now_ns() - w->start_ns) / 1000000,
           w->start_ns, w->fd, w->object, w->point, w->flags, w->deadline_ns);
   print_objects(f, w->objects);
   fprintf(f, "Completion is observed only via successful waits/queries on a private queue timeline. "
           "not-observed does NOT mean incomplete. No diagnostic GPU queries.\n");
   for (struct queue_history *q = queues; q; q = q->next) {
      fprintf(f, "QUEUE fd=%d queue=0x%" PRIx64 " last=%u\n", q->fd, q->queue,
              q->next_entry < HISTORY ? q->next_entry : HISTORY);
      for (unsigned j = 0; j < HISTORY; ++j) {
         struct submission *s = &q->entries[(q->next_entry + j) % HISTORY];
         if (!s->id)
            continue;
         fprintf(f, " submit=%" PRIu64 " time_ns=%" PRIu64 " returned=%d"
                 " return_ns=%" PRIu64 " result=%d errno=%d completion=%s\n",
                 s->id, s->time_ns, s->returned, s->return_ns, s->result, s->error,
                 s->completion_observed ? "observed" : "not-observed");
         fprintf(f, "  private-progress object=0x%x point=%" PRIu64 "\n",
                 s->completion_handle, s->completion_point);
         for (unsigned i = 0; i < s->wait_count && i < SIGNALS; ++i)
            fprintf(f, "  wait object=0x%" PRIx64 " point=%" PRIu64 "\n",
                    s->waits[i].object, s->waits[i].point);
         for (unsigned i = 0; i < s->signal_count && i < SIGNALS; ++i)
            fprintf(f, "  signal object=0x%" PRIx64 " point=%" PRIu64 "\n",
                    s->signals[i].object, s->signals[i].point);
         if (s->wait_count > SIGNALS || s->signal_count > SIGNALS)
            fprintf(f, "  truncated: total_waits=%u total_signals=%u cap=%u\n",
                    s->wait_count, s->signal_count, SIGNALS);
      }
   }
   fprintf(f, "RECENT IOCTL RETURNS (errno meaningful only on failure)\n");
   for (unsigned j = 0; j < IOCTLS; ++j) {
      struct ioctl_result *r = &ioctls[(next_ioctl + j) % IOCTLS];
      if (r->name)
         fprintf(f, " time_ns=%" PRIu64 " tid=%" PRIu64 " fd=%d %s ret=%d errno=%d\n",
                 r->time_ns, r->tid, r->fd, r->name, r->result, r->error);
   }
   fprintf(f, "END HK LONG WAIT\n");
   fclose(f);
   return buffer;
}

static void
write_all(int fd, const char *data, size_t size)
{
   while (size) {
      ssize_t written = write(fd, data, size);
      if (written < 0 && errno == EINTR)
         continue;
      if (written <= 0)
         break;
      data += written;
      size -= written;
   }
}

static void
emit(const char *data, size_t size)
{
   char path[4096];
   const char *setting = getenv("HK_WAIT_WATCHDOG_FILE");
   if (setting && setting[0]) {
      /* A %p placeholder avoids collisions between application processes. */
      const char *pid = strstr(setting, "%p");
      if (pid)
         snprintf(path, sizeof(path), "%.*s%d%s", (int)(pid - setting), setting,
                  getpid(), pid + 2);
      else
         snprintf(path, sizeof(path), "%s", setting);
   } else {
      snprintf(path, sizeof(path), "/tmp/hk-wait-watchdog-%d.log", getpid());
   }
   int fd = open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
   if (fd >= 0) {
      struct stat st;
      if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_uid == geteuid())
         write_all(fd, data, size);
      close(fd);
   }
   /* Persist first: a full application stderr pipe must not lose the file. */
   write_all(STDERR_FILENO, data, size);
}

static void *
watch(void *unused)
{
   (void)unused;
   /* Cancellation is allowed only during unlocked diagnostic I/O. */
   pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
   pthread_mutex_lock(&lock);
   while (!stopping) {
      uint64_t now = now_ns(), next = UINT64_MAX;
      struct u_watchdog_wait *due = NULL;
      for (struct u_watchdog_wait *w = active; w; w = w->next) {
         if (w->reported)
            continue;
         uint64_t fire = w->start_ns + WAIT_NS;
         if (fire <= now) {
            due = w;
            break;
         }
         if (fire < next)
            next = fire;
      }
      if (due) {
         size_t size = 0;
         char *data = snapshot(due, &size);
         due->reported = true;
         pthread_mutex_unlock(&lock);
         pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
         if (data) {
            emit(data, size);
            free(data);
         }
         pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
         pthread_mutex_lock(&lock);
      } else if (next == UINT64_MAX) {
         pthread_cond_wait(&wake, &lock);
      } else {
         struct timespec until = {next / 1000000000, next % 1000000000};
         pthread_cond_timedwait(&wake, &lock, &until);
      }
   }
   pthread_mutex_unlock(&lock);
   return NULL;
}

static void
initialize_wake(void)
{
   pthread_condattr_t attr;
   pthread_condattr_init(&attr);
   pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
   pthread_cond_init(&wake, &attr);
   pthread_condattr_destroy(&attr);
}

static void fork_prepare(void) { pthread_mutex_lock(&lock); }
static void fork_parent(void) { pthread_mutex_unlock(&lock); }
static void
fork_child(void)
{
   /* Parent threads and their stack scopes don't exist in the child. */
   active = NULL;
   queues = NULL;
   memset(ioctls, 0, sizeof(ioctls));
   next_ioctl = 0;
   cached_tid = 0;
   started = false;
   stopping = false;
   /* No parent waiter bookkeeping may be reused in the child. */
   wake = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
   initialize_wake();
   pthread_mutex_unlock(&lock);
}

static void
initialize(void)
{
   initialize_wake();
   pthread_atfork(fork_prepare, fork_parent, fork_child);
}

void
u_watchdog_begin(struct u_watchdog_wait *w)
{
   int saved_errno = errno;
   w->start_ns = now_ns();
   /* Short waits cannot reach the threshold; don't register ordinary polls. */
   if (w->deadline_ns != UINT64_MAX &&
       (w->deadline_ns <= w->start_ns || w->deadline_ns - w->start_ns < WAIT_NS)) {
      errno = saved_errno;
      return;
   }
   pthread_once(&once, initialize);
   pthread_mutex_lock(&lock);
   if (!started && !stopping)
      started = pthread_create(&thread, NULL, watch, NULL) == 0;
   if (started && !stopping) {
      w->tid = thread_id();
      w->active = true;
      w->next = active;
      active = w;
      pthread_cond_signal(&wake);
   }
   pthread_mutex_unlock(&lock);
   errno = saved_errno;
}

void
u_watchdog_end(struct u_watchdog_wait *w)
{
   if (!w->active)
      return;
   int saved_errno = errno;
   pthread_mutex_lock(&lock);
   struct u_watchdog_wait **link = &active;
   while (*link && *link != w)
      link = &(*link)->next;
   if (*link)
      *link = w->next;
   w->active = false;
   pthread_mutex_unlock(&lock);
   errno = saved_errno;
}

void
u_watchdog_ioctl(const char *name, int fd, int result, int error)
{
   int saved_errno = errno;
   pthread_once(&once, initialize);
   pthread_mutex_lock(&lock);
   ioctls[next_ioctl++ % IOCTLS] = (struct ioctl_result){
      .name = name, .fd = fd, .result = result, .error = error,
      .time_ns = now_ns(), .tid = thread_id(),
   };
   pthread_mutex_unlock(&lock);
   errno = saved_errno;
}

uint64_t
u_watchdog_submit(int fd, uint64_t queue, struct u_watchdog_objects waits,
                  struct u_watchdog_objects signals)
{
   int saved_errno = errno;
   pthread_once(&once, initialize);
   pthread_mutex_lock(&lock);
   struct queue_history *q;
   for (q = queues; q && (q->fd != fd || q->queue != queue); q = q->next)
      ;
   if (!q) {
      q = calloc(1, sizeof(*q));
      if (!q) {
         pthread_mutex_unlock(&lock);
         errno = saved_errno;
         return 0;
      }
      q->fd = fd;
      q->queue = queue;
      q->next = queues;
      queues = q;
   }
   struct submission *s = &q->entries[q->next_entry++ % HISTORY];
   *s = (struct submission){.id = ++next_id, .time_ns = now_ns(),
                           .wait_count = waits.count, .signal_count = signals.count};
   for (unsigned i = 0; i < waits.count && i < SIGNALS; ++i) {
      s->waits[i].object = read_item(waits.objects, waits.object_stride, i, waits.object32);
      s->waits[i].point = read_item(waits.points, waits.point_stride, i, false);
   }
   for (unsigned i = 0; i < signals.count && i < SIGNALS; ++i) {
      s->signals[i].object = read_item(signals.objects, signals.object_stride, i, signals.object32);
      s->signals[i].point = read_item(signals.points, signals.point_stride, i, false);
   }
   /* HK appends its private queue timeline as the final output signal.
    * Application semaphores may be host-signaled independently of GPU work. */
   if (signals.count) {
      s->completion_handle = read_item(signals.objects, signals.object_stride,
                                       signals.count - 1, signals.object32);
      s->completion_point = read_item(signals.points, signals.point_stride,
                                      signals.count - 1, false);
      /* HK initially signals this private object to 1. A query of that value
       * cannot establish completion of the first submission using point 1. */
      if (s->completion_point <= 1)
         s->completion_handle = 0;
   }
   uint64_t id = s->id;
   pthread_mutex_unlock(&lock);
   errno = saved_errno;
   return id;
}

void
u_watchdog_submit_result(uint64_t id, int result, int error)
{
   int saved_errno = errno;
   pthread_mutex_lock(&lock);
   for (struct queue_history *q = queues; q; q = q->next) {
      for (unsigned i = 0; i < HISTORY; ++i) {
         struct submission *s = &q->entries[i];
         if (id && s->id == id) {
            s->returned = true;
            s->return_ns = now_ns();
            s->result = result;
            s->error = error;
         }
      }
   }
   pthread_mutex_unlock(&lock);
   errno = saved_errno;
}

void
u_watchdog_complete(int fd, uint32_t handle, uint64_t point)
{
   /* Binary payloads can be reset/reimported, so handle matching alone is not
    * proof of a particular submission's completion. Only track timelines. */
   if (!point)
      return;
   pthread_mutex_lock(&lock);
   for (struct queue_history *q = queues; q; q = q->next) {
      if (q->fd != fd)
         continue;
      for (unsigned i = 0; i < HISTORY; ++i) {
         struct submission *s = &q->entries[i];
         if (!s->returned || s->result != 0)
            continue;
         if (s->completion_handle == handle && s->completion_point &&
             s->completion_point <= point)
            s->completion_observed = true;
      }
   }
   pthread_mutex_unlock(&lock);
}

void
u_watchdog_forget_object(int fd, uint32_t handle)
{
   pthread_mutex_lock(&lock);
   for (struct queue_history *q = queues; q; q = q->next) {
      if (q->fd != fd)
         continue;
      for (unsigned i = 0; i < HISTORY; ++i) {
         if (q->entries[i].completion_handle == handle)
            q->entries[i].completion_handle = 0;
      }
   }
   pthread_mutex_unlock(&lock);
}

void
u_watchdog_forget_queue(int fd, uint64_t queue)
{
   pthread_mutex_lock(&lock);
   struct queue_history **link = &queues;
   while (*link && ((*link)->fd != fd || (*link)->queue != queue))
      link = &(*link)->next;
   if (*link) {
      struct queue_history *q = *link;
      *link = q->next;
      free(q);
   }
   pthread_mutex_unlock(&lock);
}

__attribute__((destructor)) static void
finish(void)
{
   pthread_mutex_lock(&lock);
   stopping = true;
   if (started)
      pthread_cond_signal(&wake);
   pthread_mutex_unlock(&lock);
   if (started) {
      /* stderr may be a full pipe. Never let diagnostic I/O hold up dlclose. */
      pthread_cancel(thread);
      pthread_join(thread, NULL);
   }
   while (queues) {
      struct queue_history *next = queues->next;
      free(queues);
      queues = next;
   }
}
#endif
