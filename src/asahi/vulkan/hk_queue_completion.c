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
   struct pollfd files[HK_QUEUE_COMPLETION_MAX_FDS];
   unsigned indices[HK_QUEUE_COMPLETION_MAX_FDS];
   unsigned num_files = 0, num_handles = 0;
   for (unsigned i = 0; i < count; ++i) {
      struct hk_completion_fence *p = &c->pending[i];
      if (p->fd >= 0) {
         assert(num_files < ARRAY_SIZE(files));
         indices[num_files] = i;
         files[num_files++] = (struct pollfd){.fd = p->fd, .events = POLLIN};
      } else if (p->syncobj) {
         c->wait_handles[num_handles] = p->syncobj;
         c->wait_indices[num_handles++] = i;
      }
   }
   assert(num_files == c->open_fds);

   if (num_files) {
      int ready;
      do {
         ready = poll(files, num_files, 0);
      } while (ready < 0 && errno == EINTR);
      if (ready < 0)
         return c->error = -errno;
   }

   /* Later independent work may fail while an earlier fence is still
    * pending. Inspect all ready files, not just the head of the queue.
    */
   for (unsigned i = 0; i < num_files; ++i) {
      struct pollfd *p = &files[i];
      if (p->revents & (POLLNVAL | POLLHUP))
         return c->error = -EIO;
      if (!(p->revents & (POLLIN | POLLERR)))
         continue;
      int status = completion_fence_status(c, p->fd);
      if (status < 0)
         return c->error = status;
      if (status > 0) {
         close(p->fd);
         c->pending[indices[i]].fd = -1;
         c->open_fds--;
      }
   }

   /* Overflow snapshots own the same immutable fences in binary syncobjs.
    * A zero-timeout wait-any finds completed snapshots without exporting an
    * fd for every in-flight submission. Check each one's error before release.
    */
   while (num_handles) {
      uint32_t first;
      int ret = drmSyncobjWait(c->drm_fd, c->wait_handles, num_handles, 0, 0,
                              &first);
      if (ret) {
         if (errno == EINTR)
            continue;
         if (errno == ETIME)
            break;
         return c->error = -errno;
      }
      assert(first < num_handles);
      struct hk_completion_fence *p = &c->pending[c->wait_indices[first]];
      int fd;
      if (drmSyncobjExportSyncFile(c->drm_fd, p->syncobj, &fd))
         return c->error = -errno;
      int status = completion_fence_status(c, fd);
      close(fd);
      if (status <= 0)
         return c->error = status < 0 ? status : -EIO;

      drmSyncobjDestroy(c->drm_fd, p->syncobj);
      p->syncobj = 0;
      --num_handles;
      c->wait_handles[first] = c->wait_handles[num_handles];
      c->wait_indices[first] = c->wait_indices[num_handles];
   }

   unsigned retired = 0;
   while (retired < count && c->pending[retired].fd < 0 &&
          !c->pending[retired].syncobj)
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
      if (c->pending[i].syncobj)
         drmSyncobjDestroy(c->drm_fd, c->pending[i].syncobj);
   }
   free(c->pending);
   free(c->wait_handles);
   free(c->wait_indices);
   free(c->fences);
   if (c->reserved_syncobj)
      drmSyncobjDestroy(c->drm_fd, c->reserved_syncobj);
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
      void *handles = realloc(c->wait_handles, capacity * sizeof(*c->wait_handles));
      if (!handles) {
         status = -ENOMEM;
         goto fail;
      }
      c->wait_handles = handles;
      void *indices = realloc(c->wait_indices, capacity * sizeof(*c->wait_indices));
      if (!indices) {
         status = -ENOMEM;
         goto fail;
      }
      c->wait_indices = indices;
      c->capacity = capacity;
   }

   if (c->open_fds == HK_QUEUE_COMPLETION_MAX_FDS) {
      /* Reserve before submitting, so handle allocation failure cannot lose
       * error observation for newly published work. Do not merge sync files:
       * SYNC_IOC_MERGE can discard pending same-context fences, as well as
       * fences that signal between a readiness check and the merge.
       */
      if (drmSyncobjCreate(c->drm_fd, 0, &c->reserved_syncobj)) {
         status = -errno;
         goto fail;
      }
   }
   return 0;

fail:
   simple_mtx_unlock(&c->mutex);
   return status;
}

int
hk_queue_completion_end(struct hk_queue_completion *c, bool submitted)
{
   int fd = -1;
   uint32_t handle = c->reserved_syncobj;
   c->reserved_syncobj = 0;
   if (!submitted) {
      c->error = -EIO;
   } else if (handle) {
      if (drmSyncobjTransfer(c->drm_fd, handle, 0, c->syncobj, 0, 0))
         c->error = -errno;
   } else if (drmSyncobjExportSyncFile(c->drm_fd, c->syncobj, &fd)) {
      /* Work is already queued. Losing its error observation is device loss. */
      c->error = -errno;
   }

   if (!c->error) {
      unsigned count = c->submitted - c->completed;
      c->pending[count] = (struct hk_completion_fence){.fd = fd, .syncobj = handle};
      c->submitted++;
      c->open_fds += fd >= 0;
   } else if (handle) {
      drmSyncobjDestroy(c->drm_fd, handle);
   }
   int status = c->error;
   simple_mtx_unlock(&c->mutex);
   return status;
}

int
hk_queue_completion_status(struct hk_queue_completion *c)
{
   /* This is an error observation, not an application completion wait. The
    * submission path may hold the lock across a slow ioctl. Any error it
    * records remains sticky for the next uncontended observation.
    */
   if (!simple_mtx_trylock(&c->mutex))
      return 1;
   int status = completion_status_locked(c);
   simple_mtx_unlock(&c->mutex);
   return status;
}
