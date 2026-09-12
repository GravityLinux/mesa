/*
 * Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9.h"

#include <limits.h>
#include <string.h>

static void
put_u32(uint8_t *out, uint32_t value)
{
   memcpy(out, &value, sizeof(value));
}

static void
put_u64(uint8_t *out, uint64_t value)
{
   memcpy(out, &value, sizeof(value));
}

bool
agx_apple9_build_compute_geometry_fields(
   void *record_, size_t record_size, uint64_t record_address,
   const struct agx_apple9_compute_geometry *geometry)
{
   const unsigned offset = AGX_APPLE9_COMPUTE_GEOMETRY_GROUPS_OFFSET;
   if (!record_ || !geometry || record_size < offset + 12 ||
       record_address > UINT64_MAX - offset)
      return false;

   uint64_t local_threads = 1;
   for (unsigned d = 0; d < 3; ++d) {
      if (!geometry->local[d] || geometry->local[d] > 1024 / local_threads)
         return false;
      local_threads *= geometry->local[d];
   }

   uint32_t groups[3] = {0};
   uint64_t pointer;
   switch (geometry->mode) {
   case AGX_APPLE9_COMPUTE_GEOMETRY_DIRECT:
      for (unsigned d = 0; d < 3; ++d) {
         if (!geometry->threads[d])
            return false;
         /* CDM permits a partial final workgroup. Avoid overflowing the
          * numerator when rounding up on the CPU. */
         groups[d] = geometry->threads[d] / geometry->local[d] +
                     (geometry->threads[d] % geometry->local[d] != 0);
         if (groups[d] > 65535)
            return false;
      }
      pointer = record_address + offset;
      break;
   case AGX_APPLE9_COMPUTE_GEOMETRY_INDIRECT:
      pointer = geometry->group_counts;
      if (!pointer || (pointer & 3))
         return false;
      break;
   default:
      return false;
   }

   uint8_t *record = record_;
   put_u64(record, pointer);
   for (unsigned d = 0; d < 3; ++d)
      put_u32(record + offset + d * sizeof(uint32_t), groups[d]);
   return true;
}
