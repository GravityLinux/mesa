/* SPDX-License-Identifier: MIT */
#ifndef U_WAIT_WATCHDOG_H
#define U_WAIT_WATCHDOG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed, immutable arrays. Their lifetime includes that of the wait scope.
 * A NULL points array denotes binary objects. Objects may be 32-bit handles
 * or 64-bit identifiers; strides also permit arrays of driver structs.
 */
struct u_watchdog_objects {
   const void *objects, *points;
   size_t object_stride, point_stride;
   unsigned count;
   bool object32;
};

struct u_watchdog_wait {
   struct u_watchdog_wait *next;
   const char *name;
   int fd;
   uint64_t object, point, flags, deadline_ns, start_ns, tid;
   struct u_watchdog_objects objects;
   bool active, reported;
};

#ifdef HK_WAIT_WATCHDOG
void u_watchdog_begin(struct u_watchdog_wait *wait);
void u_watchdog_end(struct u_watchdog_wait *wait);
void u_watchdog_ioctl(const char *name, int fd, int result, int error);
uint64_t u_watchdog_submit(int fd, uint64_t queue,
                          struct u_watchdog_objects waits,
                          struct u_watchdog_objects signals);
void u_watchdog_submit_result(uint64_t id, int result, int error);
void u_watchdog_complete(int fd, uint32_t handle, uint64_t point);
void u_watchdog_forget_object(int fd, uint32_t handle);
void u_watchdog_forget_queue(int fd, uint64_t queue);

#define U_WATCHDOG_SCOPE(var, label, device_fd, obj, value, wait_flags, deadline, span) \
   struct u_watchdog_wait var __attribute__((cleanup(u_watchdog_end))) = { \
      .name = (label), .fd = (device_fd), .object = (uint64_t)(obj), \
      .point = (value), .flags = (wait_flags), .deadline_ns = (deadline), \
      .objects = (span), \
   }; \
   u_watchdog_begin(&var)

/* Preserve the wrapped call's result and errno exactly. */
#define U_WATCHDOG_IOCTL(label, device_fd, expression) ({ \
   U_WATCHDOG_SCOPE(_wd_ioctl, label, device_fd, 0, 0, 0, UINT64_MAX, \
                    ((struct u_watchdog_objects){0})); \
   int _wd_result = (expression); \
   u_watchdog_ioctl(label, device_fd, _wd_result, errno); \
   _wd_result; \
})
#else
#define U_WATCHDOG_SCOPE(...) ((void)0)
#define U_WATCHDOG_IOCTL(label, device_fd, expression) (expression)
#define u_watchdog_ioctl(...) ((void)0)
#define u_watchdog_submit(...) (UINT64_C(0))
#define u_watchdog_submit_result(...) ((void)0)
#define u_watchdog_complete(...) ((void)0)
#define u_watchdog_forget_object(...) ((void)0)
#define u_watchdog_forget_queue(...) ((void)0)
#endif

#ifdef __cplusplus
}
#endif
#endif
