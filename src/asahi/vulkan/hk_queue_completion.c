/* SPDX-License-Identifier: MIT */
#include "hk_queue_completion.h"
#include <limits.h>
#include <xf86drm.h>
#include "util/libsync.h"

/* Only inspect signaled sync files. An individual file's fence set cannot
 * grow, so after discovering its size a second query always fits. Keep the
 * allocation for later submissions and check every unwrapped fence: a chain
 * node's own status does not preserve its contained fence's error.
 */
static int
completion_fence_status(struct hk_queue_completion *c, int fd)
{
   struct sync_file_info info = {
      .num_fences = c->fence_capacity,
      .sync_fence_info = (uint64_t)(uintptr_t)c->fences,
   };
   if (ioctl(fd, SYNC_IOC_FILE_INFO, &info)) {
      if (errno != EINVAL)
         return -errno;

      info = (struct sync_file_info){0};
      if (ioctl(fd, SYNC_IOC_FILE_INFO, &info))
         return -errno;

      if (info.num_fences > SIZE_MAX / sizeof(*c->fences))
         return -ENOMEM;
      void *fences = realloc(c->fences, info.num_fences * sizeof(*c->fences));
      if (!fences)
         return -ENOMEM;
      c->fences = fences;
      c->fence_capacity = info.num_fences;
      info.sync_fence_info = (uint64_t)(uintptr_t)c->fences;
      if (ioctl(fd, SYNC_IOC_FILE_INFO, &info))
         return -errno;
   }

   int status = info.status;
   for (unsigned i = 0; i < info.num_fences; ++i) {
      if (c->fences[i].status < 0)
         return c->fences[i].status;
      if (c->fences[i].status == 0)
         status = 0;
   }
   return status;
}

static int
completion_status_locked(struct hk_queue_completion *c)
{
   if (c->error)
      return c->error;
   if (c->completed == c->submitted)
      return 1;

   unsigned count = c->submitted - c->completed;
   int ready;
   do {
      ready = poll(c->pending, count, 0);
   } while (ready < 0 && errno == EINTR);
   if (ready < 0)
      return c->error = -errno;
   if (!ready)
      return 0;

   /* Later independent work may fail while an earlier fence is still
    * pending. Inspect all ready files, not just the head of the queue.
    */
   for (unsigned i = 0; i < count; ++i) {
      struct pollfd *p = &c->pending[i];
      if (p->revents & (POLLNVAL | POLLHUP))
         return c->error = -EIO;
      if (!(p->revents & (POLLIN | POLLERR)))
         continue;
      int status = completion_fence_status(c, p->fd);
      if (status < 0)
         return c->error = status;
      if (status > 0) {
         close(p->fd);
         p->fd = -1;
      }
   }

   unsigned retired = 0;
   while (retired < count && c->pending[retired].fd < 0)
      retired++;
   memmove(c->pending, c->pending + retired,
           (count - retired) * sizeof(*c->pending));
   c->completed += retired;
   return c->completed == c->submitted;
}

int
hk_queue_completion_init(struct hk_queue_completion *c, int drm_fd)
{
   *c = (struct hk_queue_completion){.drm_fd = drm_fd};
   simple_mtx_init(&c->mutex, mtx_plain);
   c->fence_capacity = 4;
   c->fences = calloc(c->fence_capacity, sizeof(*c->fences));
   if (!c->fences)
      return -ENOMEM;
   if (drmSyncobjCreate(drm_fd, 0, &c->syncobj))
      return -errno;
   return 0;
}

void
hk_queue_completion_finish(struct hk_queue_completion *c)
{
   for (uint64_t i = 0; i < c->submitted - c->completed; ++i) {
      if (c->pending[i].fd >= 0)
         close(c->pending[i].fd);
   }
   free(c->pending);
   free(c->fences);
   if (c->syncobj)
      drmSyncobjDestroy(c->drm_fd, c->syncobj);
   simple_mtx_destroy(&c->mutex);
}

int
hk_queue_completion_begin(struct hk_queue_completion *c)
{
   simple_mtx_lock(&c->mutex);
   int status = completion_status_locked(c);
   if (status < 0)
      goto fail;

   uint64_t count = c->submitted - c->completed;
   if (count == c->capacity) {
      if (c->capacity > UINT_MAX / 2 ||
          c->capacity > SIZE_MAX / (2 * sizeof(*c->pending))) {
         status = -ENOMEM;
         goto fail;
      }
      unsigned capacity = MAX2(8, c->capacity * 2);
      void *pending = realloc(c->pending, capacity * sizeof(*c->pending));
      if (!pending) {
         status = -ENOMEM;
         goto fail;
      }
      c->pending = pending;
      c->capacity = capacity;
   }
   return 0;

fail:
   simple_mtx_unlock(&c->mutex);
   return status;
}

int
hk_queue_completion_end(struct hk_queue_completion *c, bool submitted)
{
   int fd;
   if (!submitted) {
      c->error = -EIO;
   } else if (drmSyncobjExportSyncFile(c->drm_fd, c->syncobj, &fd)) {
      /* Work is already queued. Losing its error observation is device loss. */
      c->error = -errno;
   } else {
      unsigned count = c->submitted - c->completed;
      c->pending[count] = (struct pollfd){.fd = fd, .events = POLLIN};
      c->submitted++;
   }
   int status = c->error;
   simple_mtx_unlock(&c->mutex);
   return status;
}

int
hk_queue_completion_status(struct hk_queue_completion *c)
{
   simple_mtx_lock(&c->mutex);
   int status = completion_status_locked(c);
   simple_mtx_unlock(&c->mutex);
   return status;
}
