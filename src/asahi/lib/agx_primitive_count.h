/* SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/genxml/agx_pack.h"

/* macOS enables the few-primitives ISP mode for 1..4 statically known
 * primitives across all draws and instances in a render pass. Saturate at
 * five, which also represents an unknown count, so overflow cannot enable it.
 */
#define AGX_PRIMITIVE_COUNT_UNKNOWN 5

static inline uint8_t
agx_count_primitives(enum agx_primitive topology, uint32_t vertices,
                     uint32_t instances)
{
   uint64_t count;
   switch (topology) {
   case AGX_PRIMITIVE_POINTS:
      count = vertices;
      break;
   case AGX_PRIMITIVE_LINES:
      count = vertices / 2;
      break;
   case AGX_PRIMITIVE_LINE_STRIP:
      count = vertices > 1 ? vertices - 1 : 0;
      break;
   case AGX_PRIMITIVE_LINE_LOOP:
      count = vertices > 1 ? vertices : 0;
      break;
   case AGX_PRIMITIVE_TRIANGLES:
      count = vertices / 3;
      break;
   case AGX_PRIMITIVE_TRIANGLE_STRIP:
   case AGX_PRIMITIVE_TRIANGLE_FAN:
      count = vertices > 2 ? vertices - 2 : 0;
      break;
   default:
      return AGX_PRIMITIVE_COUNT_UNKNOWN;
   }

   return MIN2(count * instances, AGX_PRIMITIVE_COUNT_UNKNOWN);
}

static inline void
agx_add_primitive_count(uint8_t *total, uint8_t count)
{
   *total = MIN2((unsigned)*total + count, AGX_PRIMITIVE_COUNT_UNKNOWN);
}

static inline bool
agx_has_few_primitives(uint8_t count)
{
   return count > 0 && count < AGX_PRIMITIVE_COUNT_UNKNOWN;
}
