/*
 * Copyright © 2024 Google, Inc.
 * SPDX-License-Identifier: MIT
 */

#include "util/u_wait_watchdog.h"
#include "util/u_memory.h"
#include "util/u_sync_provider.h"
#include <xf86drm.h>
#include <errno.h>

struct util_sync_provider_drm {
   struct util_sync_provider base;
   int drm_fd;
};

static int
drm_fd(struct util_sync_provider *p)
{
   struct util_sync_provider_drm *d = (struct util_sync_provider_drm *)p;
   return d->drm_fd;
}

static int
drm_syncobj_create(struct util_sync_provider *p, uint32_t flags, uint32_t *handle)
{
   int ret = drmSyncobjCreate(drm_fd(p), flags, handle);
   u_watchdog_ioctl("drmSyncobjCreate", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_destroy(struct util_sync_provider *p, uint32_t handle)
{
   u_watchdog_forget_object(drm_fd(p), handle);
   int ret = drmSyncobjDestroy(drm_fd(p), handle);
   u_watchdog_ioctl("drmSyncobjDestroy", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_handle_to_fd(struct util_sync_provider *p, uint32_t handle, int *out_obj_fd)
{
   int ret = drmSyncobjHandleToFD(drm_fd(p), handle, out_obj_fd);
   u_watchdog_ioctl("drmSyncobjHandleToFD", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_fd_to_handle(struct util_sync_provider *p, int obj_fd, uint32_t *handle)
{
   int ret = drmSyncobjFDToHandle(drm_fd(p), obj_fd, handle);
   u_watchdog_ioctl("drmSyncobjFDToHandle", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_import_sync_file(struct util_sync_provider *p, uint32_t handle, int sync_file_fd)
{
   u_watchdog_forget_object(drm_fd(p), handle);
   int ret = drmSyncobjImportSyncFile(drm_fd(p), handle, sync_file_fd);
   u_watchdog_ioctl("drmSyncobjImportSyncFile", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_export_sync_file(struct util_sync_provider *p, uint32_t handle, int *out_sync_file_fd)
{
   int ret = drmSyncobjExportSyncFile(drm_fd(p), handle, out_sync_file_fd);
   u_watchdog_ioctl("drmSyncobjExportSyncFile", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_wait(struct util_sync_provider *p, uint32_t *handles, unsigned num_handles,
                 int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled)
{
   U_WATCHDOG_SCOPE(wd, "drm_syncobj_wait", drm_fd(p), 0, 0, flags,
                    (uint64_t)timeout_nsec,
                    ((struct u_watchdog_objects){
                       .objects = handles, .points = NULL,
                       .object_stride = sizeof(*handles), .point_stride = 8,
                       .count = num_handles, .object32 = true,
                    }));

   int ret = drmSyncobjWait(drm_fd(p), handles, num_handles, timeout_nsec, flags, first_signaled);
   u_watchdog_ioctl("drmSyncobjWait", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_reset(struct util_sync_provider *p, const uint32_t *handles, uint32_t handle_count)
{
   for (unsigned i = 0; i < handle_count; ++i)
      u_watchdog_forget_object(drm_fd(p), handles[i]);
   int ret = drmSyncobjReset(drm_fd(p), handles, handle_count);
   u_watchdog_ioctl("drmSyncobjReset", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_signal(struct util_sync_provider *p, const uint32_t *handles, uint32_t handle_count)
{
   int ret = drmSyncobjSignal(drm_fd(p), handles, handle_count);
   u_watchdog_ioctl("drmSyncobjSignal", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_timeline_signal(struct util_sync_provider *p, const uint32_t *handles,
                            uint64_t *points, uint32_t handle_count)
{
   int ret = drmSyncobjTimelineSignal(drm_fd(p), handles, points, handle_count);
   u_watchdog_ioctl("drmSyncobjTimelineSignal", drm_fd(p), ret, errno);
   return ret;
}

static int
drm_syncobj_timeline_wait(struct util_sync_provider *p, uint32_t *handles, uint64_t *points,
                          unsigned num_handles, int64_t timeout_nsec, unsigned flags,
                          uint32_t *first_signaled)
{
   U_WATCHDOG_SCOPE(wd, "drm_syncobj_timeline_wait", drm_fd(p), 0, 0, flags,
                    (uint64_t)timeout_nsec,
                    ((struct u_watchdog_objects){
                       .objects = handles, .points = points,
                       .object_stride = sizeof(*handles), .point_stride = 8,
                       .count = num_handles, .object32 = true,
                    }));

   int ret = drmSyncobjTimelineWait(drm_fd(p), handles, points, num_handles, timeout_nsec, flags, first_signaled);
   u_watchdog_ioctl("drmSyncobjTimelineWait", drm_fd(p), ret, errno);
   if (ret == 0 && !(flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE)) {
      if ((flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL) || num_handles == 1) {
         for (unsigned i = 0; i < num_handles; ++i)
            u_watchdog_complete(drm_fd(p), handles[i], points[i]);
      } else if (first_signaled && *first_signaled < num_handles) {
         u_watchdog_complete(drm_fd(p), handles[*first_signaled], points[*first_signaled]);
      }
   }
   return ret;
}

static int
drm_syncobj_query(struct util_sync_provider *p, uint32_t *handles, uint64_t *points,
                  uint32_t handle_count, uint32_t flags)
{
   int ret = drmSyncobjQuery2(drm_fd(p), handles, points, handle_count, flags);
   u_watchdog_ioctl("drmSyncobjQuery2", drm_fd(p), ret, errno);
   /* LAST_SUBMITTED is availability, not GPU completion. */
   if (ret == 0 && flags == 0) {
      for (unsigned i = 0; i < handle_count; ++i)
         u_watchdog_complete(drm_fd(p), handles[i], points[i]);
   }
   return ret;
}

static int
drm_syncobj_transfer(struct util_sync_provider *p, uint32_t dst_handle, uint64_t dst_point,
                     uint32_t src_handle, uint64_t src_point, uint32_t flags)
{
   int ret = drmSyncobjTransfer(drm_fd(p), dst_handle, dst_point, src_handle, src_point, flags);
   u_watchdog_ioctl("drmSyncobjTransfer", drm_fd(p), ret, errno);
   return ret;
}

static void
drm_syncobj_finalize(struct util_sync_provider *p)
{
   free(p);
}

static struct util_sync_provider *
drm_sync_provider_clone(struct util_sync_provider *p)
{
   return util_sync_provider_drm(drm_fd(p));
}


struct util_sync_provider *
util_sync_provider_drm(int drm_fd)
{
   struct util_sync_provider_drm *d = CALLOC_STRUCT(util_sync_provider_drm);

   d->drm_fd = drm_fd;
   d->base = (struct util_sync_provider) {
      .create = drm_syncobj_create,
      .destroy = drm_syncobj_destroy,
      .handle_to_fd = drm_syncobj_handle_to_fd,
      .fd_to_handle = drm_syncobj_fd_to_handle,
      .import_sync_file = drm_syncobj_import_sync_file,
      .export_sync_file = drm_syncobj_export_sync_file,
      .wait = drm_syncobj_wait,
      .reset = drm_syncobj_reset,
      .signal = drm_syncobj_signal,
      .query = drm_syncobj_query,
      .transfer = drm_syncobj_transfer,
      .finalize = drm_syncobj_finalize,
      .clone = drm_sync_provider_clone,
   };

   uint64_t cap;
   int err = drmGetCap(drm_fd, DRM_CAP_SYNCOBJ_TIMELINE, &cap);
   if (err == 0 && cap != 0) {
      d->base.timeline_signal = drm_syncobj_timeline_signal;
      d->base.timeline_wait = drm_syncobj_timeline_wait;
   }

   return &d->base;
}
