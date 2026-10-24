/* SPDX-License-Identifier: MIT */
#pragma once

#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include "util/simple_mtx.h"

struct sync_fence_info;

/* One extra, transient fd is needed when inspecting a completed overflow
 * snapshot. Binary syncobj handles are not subject to RLIMIT_NOFILE.
 */
#define HK_QUEUE_COMPLETION_MAX_FDS 64

struct hk_completion_fence {
   int fd;
   uint32_t syncobj;
};

/* Own the error-bearing fences until they have been checked. A syncobj
 * timeline alone cannot do this: it garbage-collects signaled chain nodes,
 * including failed ones. This is a retirement queue, not a point lookup table.
 */
struct hk_queue_completion {
   simple_mtx_t mutex;
   int drm_fd;
   uint32_t syncobj;
   int error;

   /* Serial numbers of submitted and retired kernel submissions. */
   uint64_t submitted, completed;
   struct hk_completion_fence *pending;
   unsigned capacity, open_fds;
   uint32_t reserved_syncobj;
   uint32_t *wait_handles;
   unsigned *wait_indices;

   struct sync_fence_info *fences;
   unsigned fence_capacity;
};

int hk_queue_completion_init(struct hk_queue_completion *c, int drm_fd);
void hk_queue_completion_finish(struct hk_queue_completion *c);

/* begin reserves storage and holds mutex across the submit and publication.
 * Every successful begin must be paired with end. The caller signals syncobj
 * in the kernel submission; end takes an immutable snapshot of that fence.
 */
int hk_queue_completion_begin(struct hk_queue_completion *c);
int hk_queue_completion_end(struct hk_queue_completion *c, bool submitted);

/* Negative errno: observed failure (sticky). Otherwise no error observed.
 * An uncontended check returns 1 if all retired, 0 if pending. A contended
 * check returns 1 without waiting for submission; this is not a completion
 * query and must not be used to satisfy application waits.
 */
int hk_queue_completion_status(struct hk_queue_completion *c);
