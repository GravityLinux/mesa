/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_launch.h"
#include "asahi/lib/agx_apple9_layout.h"
#include "asahi/compiler/agx_apple9_encoding.h"
#include "asahi/compiler/agx_apple9_profile.h"

#include <string.h>

static void
write16(uint8_t *p, uint16_t value)
{
   p[0] = value;
   p[1] = value >> 8;
}

bool
agx_apple9_launch_threadgroup_memory_supported(uint32_t bytes)
{
   return bytes == 0 ||
          (bytes >= 128 && bytes <= AGX_APPLE9_MAX_SHARED_BYTES && !(bytes & (bytes - 1)));
}

size_t
agx_apple9_launch_call_offset(enum agx_apple9_launch_stage stage,
                              unsigned resource_count)
{
   if (stage == AGX_APPLE9_LAUNCH_VERTEX || stage == AGX_APPLE9_LAUNCH_FRAGMENT)
      return 68;
   if (stage == AGX_APPLE9_LAUNCH_TASK || stage == AGX_APPLE9_LAUNCH_MESH)
      return 16 + 14 * (AGX_APPLE9_MESH_ARGUMENT_WORDS / 4) + 8;
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
   memset(out, 0, 8);
   out[0] = 0x77;
   out[2] = 0x2a;
   out[3] = 0x41;
   if ((stage == AGX_APPLE9_LAUNCH_COMPUTE ||
        stage == AGX_APPLE9_LAUNCH_TASK || stage == AGX_APPLE9_LAUNCH_MESH) &&
       params->threadgroup_memory_bytes) {
      write16(out + 4, (params->threadgroup_memory_bytes << 2) | 0x80);
      out[6] = params->threadgroup_memory_bytes >> 14;
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

static bool
encode_target(uint8_t out[10], enum agx_apple9_launch_stage stage,
              uint64_t shader_base, uint32_t entry_offset)
{
   if (entry_offset & 1)
      return false;
   if (stage == AGX_APPLE9_LAUNCH_TASK || stage == AGX_APPLE9_LAUNCH_MESH) {
      /* Native launch DATA exposes a full absolute target, unlike the
       * compact USC-relative graphics/compute target. No executable source
       * is used to construct this form. */
      const uint64_t limit = ((UINT64_C(1) << 48) - 0x2a) / 2;
      if ((shader_base & 1) || shader_base > limit ||
          entry_offset > limit - shader_base)
         return false;
      uint64_t target = 2 * (shader_base + entry_offset) + 0x2a;
      memset(out, 0, 10);
      write16(out, 0x177);
      for (unsigned i = 0; i < 6; ++i)
         out[2 + i] = target >> (8 * i);
      out[8] = stage == AGX_APPLE9_LAUNCH_MESH ? 6 : 2;
      return true;
   }
   if (entry_offset > AGX_APPLE9_ENTRY_MAX_OFFSET)
      return false;
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
   return true;
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

static bool
launch_build(uint8_t *out, size_t capacity, enum agx_apple9_launch_stage stage,
             const struct agx_apple9_launch_parameters *params,
             struct agx_apple9_launch_template *tmpl)
{
   if (!out || !params || (unsigned)stage > AGX_APPLE9_LAUNCH_MESH)
      return false;
   bool compute = stage == AGX_APPLE9_LAUNCH_COMPUTE;
   bool mesh =
      stage == AGX_APPLE9_LAUNCH_TASK || stage == AGX_APPLE9_LAUNCH_MESH;
   size_t size = compute ? AGX_APPLE9_COMPUTE_LAUNCH_SIZE
                         : AGX_APPLE9_GRAPHICS_LAUNCH_SIZE;
   if (capacity < size || params->publication_count > 1022 ||
       (params->threadgroup_memory_bytes &&
        (!(compute || mesh) || !agx_apple9_launch_threadgroup_memory_supported(
                        params->threadgroup_memory_bytes))) ||
       (compute && !agx_apple9_launch_call_offset(stage, params->resource_count)) ||
       (stage == AGX_APPLE9_LAUNCH_FRAGMENT &&
        (params->tile_bytes > 128 ||
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
                    : mesh ? AGX_APPLE9_MESH_ARGUMENT_WORDS
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
   if (tmpl)
      tmpl->target_at = at;
   if (!encode_target(program + at, stage, params->shader_base, params->entry_offset))
      return false;
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
      if (tmpl)
         tmpl->branch_at = at;
      at += 10;
   }
   /* The preamble's STOP, or this STOP when no setup remains, completes the
    * argument context and starts the selected main shader. */
   write16(program + at, 0x0e);
   at += 4;
   if (at > size)
      return false;
   if (tmpl)
      tmpl->length = at;
   memcpy(out, program, at);
   memset(out + at, 0, capacity - at);
   return true;
}

bool
agx_apple9_launch_build(uint8_t *out, size_t capacity,
                        enum agx_apple9_launch_stage stage,
                        const struct agx_apple9_launch_parameters *params)
{
   return launch_build(out, capacity, stage, params, NULL);
}

bool
agx_apple9_launch_template_init(struct agx_apple9_launch_template *tmpl,
                                size_t capacity,
                                enum agx_apple9_launch_stage stage,
                                const struct agx_apple9_launch_parameters *params)
{
   memset(tmpl, 0, sizeof(*tmpl));
   if (!params || capacity > sizeof(tmpl->program))
      return false;
   /* Placeholders for the per-launch fields: a valid even entry, and a
    * program address that keeps the preamble branch encodable. */
   struct agx_apple9_launch_parameters p = *params;
   p.resource_table = 0;
   p.entry_offset = 0;
   p.launch_address = p.preamble_address ? p.shader_base : 0;
   if (!launch_build(tmpl->program, capacity, stage, &p, tmpl))
      return false;
   tmpl->shader_base = p.shader_base;
   tmpl->preamble_address = p.preamble_address;
   tmpl->capacity = capacity;
   tmpl->stage = stage;
   tmpl->valid = true;
   return true;
}

bool
agx_apple9_launch_template_emit(
   uint8_t *out, const struct agx_apple9_launch_template *tmpl,
   uint64_t resource_table, uint32_t entry_offset, uint64_t launch_address)
{
   if (!tmpl->valid)
      return false;
   uint8_t head[16], target[10], branch[10];
   const unsigned root = 2;
   agx_apple9_encode_literal32(head, root, resource_table);
   agx_apple9_encode_literal32(head + 8, root + 1, resource_table >> 32);
   if (!encode_target(target, tmpl->stage, tmpl->shader_base, entry_offset))
      return false;
   if (tmpl->branch_at) {
      /* As launch_build: the displacement is relative to the branch. */
      uint64_t base = tmpl->shader_base;
      if ((launch_address & 1) || launch_address < base ||
          launch_address - base > UINT32_MAX - tmpl->branch_at)
         return false;
      int64_t displacement = (int64_t)(tmpl->preamble_address - base) -
                             (int64_t)(launch_address - base + tmpl->branch_at);
      if (!agx_apple9_encode_branch(branch, true, displacement))
         return false;
   }
   memcpy(out, tmpl->program, tmpl->length);
   memcpy(out, head, sizeof(head));
   memcpy(out + tmpl->target_at, target, sizeof(target));
   if (tmpl->branch_at)
      memcpy(out + tmpl->branch_at, branch, sizeof(branch));
   memset(out + tmpl->length, 0, tmpl->capacity - tmpl->length);
   return true;
}
