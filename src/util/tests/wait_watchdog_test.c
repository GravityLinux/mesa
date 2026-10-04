/* SPDX-License-Identifier: MIT */
/* An indefinitely blocked driver-like wait. The Python test observes the
 * dump BEFORE releasing this process, so an after-return logger cannot pass.
 */
#include "util/u_wait_watchdog.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static void history(void)
{
   uint32_t signal = 77;
   for (uint64_t point = 1; point <= 20; ++point) {
      uint64_t id = u_watchdog_submit(42, 7, (struct u_watchdog_objects){0},
         (struct u_watchdog_objects){&signal, &point, 4, 8, 1, true});
      u_watchdog_submit_result(id, 0, 0);
   }
   u_watchdog_complete(42, 77, 18);
   uint64_t point = 1;
   uint64_t id = u_watchdog_submit(42, 9, (struct u_watchdog_objects){0},
      (struct u_watchdog_objects){&signal, &point, 4, 8, 1, true});
   u_watchdog_submit_result(id, -1, EBUSY);
   /* Successful availability/binary observations must never mark completion. */
   u_watchdog_complete(42, 77, 0);
   signal = 88;
   id = u_watchdog_submit(42, 10, (struct u_watchdog_objects){0},
      (struct u_watchdog_objects){&signal, &point, 4, 8, 1, true});
   u_watchdog_submit_result(id, 0, 0);
   u_watchdog_forget_object(42, signal);
   u_watchdog_complete(42, signal, 1);
   errno = EDOM;
   u_watchdog_ioctl("test-ioctl", 42, -1, EBUSY);
   assert(errno == EDOM);
}

static void blocked(const char *label)
{
   uint32_t handles[] = {77, 88};
   uint64_t points[] = {20, 33};
   U_WATCHDOG_SCOPE(wd, label, 42, UINT64_C(0xabcdef0123456789), 20,
                    0x123, UINT64_MAX,
                    ((struct u_watchdog_objects){handles, points, 4, 8, 2, true}));
   char byte;
   if (read(STDIN_FILENO, &byte, 1) != 1)
      abort();
}

static void short_waits(void)
{
   for (unsigned i = 0; i < 200; ++i) {
      U_WATCHDOG_SCOPE(wd, "short", 42, 77, 1, 0, now_ns() + 1000000,
                       ((struct u_watchdog_objects){0}));
   }
   {
      U_WATCHDOG_SCOPE(wd, "already-finished", 42, 77, 1, 0, UINT64_MAX,
                       ((struct u_watchdog_objects){0}));
   }
}

int main(int argc, char **argv)
{
   assert(argc == 2);
   if (!strcmp(argv[1], "blocked")) {
      history();
      blocked("selftest-blocked");
   } else if (!strcmp(argv[1], "short")) {
      short_waits();
      sleep(6);
   } else if (!strcmp(argv[1], "fork")) {
      short_waits(); /* Start the parent's watcher and let it sleep. */
      pid_t child = fork();
      if (!child) {
         blocked("selftest-fork-child");
         exit(0);
      }
      int status;
      waitpid(child, &status, 0);
      assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
   } else if (!strcmp(argv[1], "full-stderr")) {
      /* Enough output to fill a pipe, followed by ordinary process exit. */
      for (unsigned q = 0; q < 200; ++q)
         u_watchdog_submit(42, q, (struct u_watchdog_objects){0},
                           (struct u_watchdog_objects){0});
      blocked("selftest-full-stderr");
   } else {
      abort();
   }
   return 0;
}
