/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_mesh_encoding.h"
#include "agx_apple9_machine.h"

bool
agx_apple9_encode_mesh(uint8_t out[6],
                       const struct agx_apple9_mesh_instruction *ins)
{
   if (!out || !ins)
      return false;

   uint64_t word;
   switch (ins->kind) {
   case AGX_APPLE9_MESH_CONTROL:
      if (ins->count_gpr >= AGX_APPLE9_GPR_COUNT ||
          ins->auxiliary_gpr >= AGX_APPLE9_GPR_COUNT)
         return false;
      word = UINT64_C(0x300000540357) |
             ((uint64_t)(2 * ins->count_gpr + ins->count_high) << 24) |
             ((uint64_t)(2 * ins->auxiliary_gpr + ins->auxiliary_high) << 33) |
             ((uint64_t)ins->control42 << 42);
      break;
   case AGX_APPLE9_MESH_SUBMIT:
      if ((unsigned)ins->mode > AGX_APPLE9_MESH_SUBMIT_CHANNEL ||
          ins->source_gpr >= AGX_APPLE9_GPR_COUNT ||
          (ins->mode == AGX_APPLE9_MESH_SUBMIT_CHANNEL &&
           ins->source_gpr + 1 >= AGX_APPLE9_GPR_COUNT))
         return false;
      word = UINT64_C(0x0006005406d7) | ((uint64_t)ins->source_gpr << 24) |
             ((uint64_t)ins->release << 31) | ((uint64_t)ins->mode << 35);
      break;
   default:
      return false;
   }

   for (unsigned i = 0; i < 6; ++i)
      out[i] = word >> (8 * i);
   return true;
}

bool
agx_apple9_decode_mesh(const uint8_t in[6],
                       struct agx_apple9_mesh_instruction *ins)
{
   if (!in || !ins)
      return false;
   uint64_t word = 0;
   for (unsigned i = 0; i < 6; ++i)
      word |= (uint64_t)in[i] << (8 * i);

   struct agx_apple9_mesh_instruction decoded = {0};
   const uint64_t count_mask =
      (UINT64_C(255) << 24) | (UINT64_C(255) << 33) | (UINT64_C(1) << 42);
   const uint64_t submit_mask = (UINT64_C(255) << 24) | (UINT64_C(3) << 35);
   if ((word & ~count_mask) == UINT64_C(0x300000540357)) {
      unsigned a = (word >> 24) & 255, b = (word >> 33) & 255;
      decoded.kind = AGX_APPLE9_MESH_CONTROL;
      decoded.count_gpr = a / 2;
      decoded.count_high = a & 1;
      decoded.auxiliary_gpr = b / 2;
      decoded.auxiliary_high = b & 1;
      decoded.control42 = (word >> 42) & 1;
      if (decoded.count_gpr >= AGX_APPLE9_GPR_COUNT ||
          decoded.auxiliary_gpr >= AGX_APPLE9_GPR_COUNT)
         return false;
   } else if ((word & ~submit_mask) == UINT64_C(0x0006005406d7)) {
      decoded.kind = AGX_APPLE9_MESH_SUBMIT;
      decoded.source_gpr = (word >> 24) & 127;
      decoded.release = (word >> 31) & 1;
      decoded.mode = (word >> 35) & 3;
      if (decoded.mode > AGX_APPLE9_MESH_SUBMIT_CHANNEL ||
          decoded.source_gpr >= AGX_APPLE9_GPR_COUNT ||
          (decoded.mode == AGX_APPLE9_MESH_SUBMIT_CHANNEL &&
           decoded.source_gpr + 1 >= AGX_APPLE9_GPR_COUNT))
         return false;
   } else {
      return false;
   }

   *ins = decoded;
   return true;
}
