/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_launch.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "asahi/compiler/agx_apple9_encoding.h"
#include "asahi/compiler/agx_apple9_profile.h"

#include <stdatomic.h>
#include <string.h>

/* Process-wide experiment state packed as ceiling | encoding << 32 so a
 * reader always sees one consistent snapshot. */
static _Atomic uint64_t experiments_state =
   AGX_APPLE9_MAX_THREADGROUP_MEMORY_BYTES;

struct agx_apple9_experiments
agx_apple9_experiments_get(void)
{
   uint64_t packed = atomic_load_explicit(&experiments_state,
                                          memory_order_acquire);
   return (struct agx_apple9_experiments){
      .threadgroup_memory_ceiling = (uint32_t)packed,
      .threadgroup_encoding = (uint8_t)(packed >> 32),
   };
}

void
agx_apple9_experiments_set(const struct agx_apple9_experiments *experiments)
{
   atomic_store_explicit(
      &experiments_state,
      (uint64_t)experiments->threadgroup_memory_ceiling |
         ((uint64_t)experiments->threadgroup_encoding << 32),
      memory_order_release);
}

static void
write16(uint8_t *p, uint16_t value)
{
   p[0] = value;
   p[1] = value >> 8;
}

static void
write32(uint8_t *p, uint32_t value)
{
   write16(p, value);
   write16(p + 2, value >> 16);
}

static bool
threadgroup_memory_supported(uint32_t bytes,
                             struct agx_apple9_experiments experiments)
{
   return bytes == 0 ||
          (bytes >= 128 &&
           bytes <= experiments.threadgroup_memory_ceiling &&
           !(bytes & (bytes - 1)));
}

bool
agx_apple9_launch_threadgroup_memory_supported(uint32_t bytes)
{
   return threadgroup_memory_supported(bytes, agx_apple9_experiments_get());
}

size_t
agx_apple9_launch_call_offset(enum agx_apple9_launch_stage stage,
                              unsigned resource_count)
{
   if (stage == AGX_APPLE9_LAUNCH_VERTEX || stage == AGX_APPLE9_LAUNCH_FRAGMENT)
      return 68;
   if (stage != AGX_APPLE9_LAUNCH_COMPUTE || !resource_count ||
       resource_count > AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES)
      return 0;
   unsigned roots = resource_count + AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE;
   return 26 + 14 * ((roots + 1) / 2);
}

static void
encode_allocation(uint8_t out[8], enum agx_apple9_launch_stage stage,
                  const struct agx_apple9_launch_parameters *params)
{
   /* WIDE32 only differs from the production word above 8 KiB, where the
    * 16-bit word overflows; build() rejects those sizes otherwise. */
   memset(out, 0, 8);
   out[0] = 0x77;
   out[2] = 0x2a;
   out[3] = 0x41;
   if (stage == AGX_APPLE9_LAUNCH_COMPUTE &&
       params->threadgroup_memory_bytes >
          AGX_APPLE9_MAX_THREADGROUP_MEMORY_BYTES) {
      write32(out + 4, (params->threadgroup_memory_bytes << 2) | 0x80);
   } else if (stage == AGX_APPLE9_LAUNCH_COMPUTE &&
              params->threadgroup_memory_bytes) {
      write16(out + 4, (params->threadgroup_memory_bytes << 2) | 0x80);
   } else if (stage == AGX_APPLE9_LAUNCH_FRAGMENT) {
      unsigned units = (params->tile_bytes + 7) / 8;
      if (params->samples == 4) {
         out[2] = 0xaa;
         out[3] = 2 * units;
         out[4] = 0x43;
      } else {
         out[3] = (params->samples == 1 ? 0x41 : 0x40) + 2 * units;
         out[4] = 0x4b;
      }
   }
}

static void
encode_target(uint8_t out[10], uint32_t entry_offset)
{
   /* EXP200: the header and fixed control fields were constructed from live
    * bit-forcing/clearing tests with authored shaders, including multi-frame
    * returned-atomic checks. Reserved/modifier bits are zero. The two control
    * bits below remain required: clearing byte 7 bit 1 times out; clearing
    * byte 8 bit 7 can suppress a later dispatch. Their individual scheduling
    * semantics are not yet established. See README.apple9-launchers.md. */
   uint32_t entry = 2 * entry_offset + 0x2a;
   memset(out, 0, 10);
   write16(out, 0x177);
   out[2] = entry;
   out[3] = entry >> 8;
   out[4] = entry >> 16;
   out[7] = 2;
   out[8] = 0x80;
}

static void
encode_frame(uint8_t out[10], const struct agx_apple9_launch_parameters *params)
{
   /* A publication unit holds two 32-bit slots. The independent A/B extents
    * support scratch and returned atomics. No further shared setup is needed. */
   memset(out, 0, 10);
   out[0] = 0xf7;
   out[2] = 0x2a;
   write16(out + 3, ((params->publication_count + 1) / 2) << 7);
   write16(out + 5, params->frame_extent_a);
   write16(out + 7, params->frame_extent_b);
}

bool
agx_apple9_launch_build(uint8_t *out, size_t capacity,
                        enum agx_apple9_launch_stage stage,
                        const struct agx_apple9_launch_parameters *params)
{
   if (!out || !params || (unsigned)stage > AGX_APPLE9_LAUNCH_COMPUTE)
      return false;
   struct agx_apple9_experiments experiments = agx_apple9_experiments_get();
   bool compute = stage == AGX_APPLE9_LAUNCH_COMPUTE;
   size_t size = compute ? AGX_APPLE9_COMPUTE_LAUNCH_SIZE
                         : AGX_APPLE9_GRAPHICS_LAUNCH_SIZE;
   /* Sizes above the production word need the explicit WIDE32 candidate;
    * the shift must also fit 32 bits. */
   if (params->threadgroup_memory_bytes >
          AGX_APPLE9_MAX_THREADGROUP_MEMORY_BYTES &&
       (experiments.threadgroup_encoding !=
           AGX_APPLE9_TG_ENC_WIDE32 ||
        params->threadgroup_memory_bytes > (1u << 29)))
      return false;
   if (capacity < size || params->entry_offset > AGX_APPLE9_ENTRY_MAX_OFFSET ||
       (params->entry_offset & 1) || params->publication_count > 1022 ||
       (params->threadgroup_memory_bytes &&
        (!compute || !threadgroup_memory_supported(
                        params->threadgroup_memory_bytes, experiments))) ||
       (compute && !agx_apple9_launch_call_offset(stage, params->resource_count)) ||
       (stage == AGX_APPLE9_LAUNCH_FRAGMENT &&
        (params->tile_bytes > (params->samples == 1 ? 128 : 64) ||
         (params->samples != 1 && params->samples != 2 &&
          params->samples != 4))))
      return false;

   /* Build privately so even an encoder failure cannot publish partial code. */
   uint8_t program[AGX_APPLE9_COMPUTE_LAUNCH_SIZE] = {0};
   const unsigned root = 2, pending = 18;
   /* Table loads use a full address in a GPR pair. They have no compact
    * USC-relative pointer field or 512-MiB placement restriction. */
   agx_apple9_encode_literal32(program, root, params->resource_table);
   agx_apple9_encode_literal32(program + 8, root + 1,
                               params->resource_table >> 32);
   unsigned at = 16;
   unsigned words = compute
      ? agx_apple9_compute_root_words(params->resource_count)
      : AGX_APPLE9_GRAPHICS_ROOT_WORDS;
   unsigned roots = words / 2;
   for (int i = (roots - 1) & ~1; i >= 0; i -= 2) {
      if (!agx_apple9_encode_pointer_load(program + at, pending + 2 * i, root,
                                          i, i + 2 <= roots ? 2 : 1))
         return false;
      at += 14;
   }
   encode_allocation(program + at, stage, params);
   at += 8;
   encode_target(program + at, params->entry_offset);
   at += 10;
   encode_frame(program + at, params);
   at += 10;
   for (unsigned i = 0; i < words; ++i) {
      if (!agx_apple9_encode_argument_word(program + at, i, pending + i,
                                           i == 0))
         return false;
      at += 4;
   }
   if (params->preamble_address) {
      /* Keep compiled setup in the persistent shader BO. Recopying a long
       * body into every launch record defeats instruction-cache reuse. */
      if (((params->preamble_address | params->launch_address) & 1) ||
          params->launch_address < params->shader_base ||
          params->preamble_address < params->shader_base ||
          params->launch_address - params->shader_base > UINT32_MAX - at ||
          params->preamble_address - params->shader_base > UINT32_MAX)
         return false;
      int64_t displacement =
         (int64_t)(params->preamble_address - params->shader_base) -
         (int64_t)(params->launch_address - params->shader_base + at);
      if (!agx_apple9_encode_branch(program + at, true, displacement))
         return false;
      at += 10;
   }
   /* The preamble's STOP, or this STOP when no setup remains, completes the
    * argument context and starts the selected main shader. */
   write16(program + at, 0x0e);
   at += 4;
   if (at > size)
      return false;
   memcpy(out, program, at);
   memset(out + at, 0, capacity - at);
   return true;
}
