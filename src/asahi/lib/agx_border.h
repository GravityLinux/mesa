/* SPDX-License-Identifier: MIT */

#pragma once

#include <stdint.h>
#include "util/format/u_formats.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx_border_packed;

void agx_pack_border(struct agx_border_packed *out, const uint32_t in[4],
                     enum pipe_format format);

#ifdef __cplusplus
}
#endif
