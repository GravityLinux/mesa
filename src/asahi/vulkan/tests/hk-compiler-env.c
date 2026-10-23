/* SPDX-License-Identifier: MIT */
#include "asahi/debug.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include <inttypes.h>
#include <stdio.h>

int
main(void)
{
   struct hk_physical_device pdev = {0};
   printf("%" PRIu64 " %" PRIu64 "\n",
          hk_physical_device_compiler_flags(&pdev), agx_apple9_debug_flags());
   return 0;
}
