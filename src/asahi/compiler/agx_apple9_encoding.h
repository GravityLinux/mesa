/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_ENCODING_H
#define AGX_APPLE9_ENCODING_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* EXP-M4-37: native mode 2 materializes an arbitrary 32-bit value into
 * r0..r63. Shared by ordinary compiler lowering and stage ABI setup.
 * Destination bits 0..3 occupy byte 0's high nibble; bits 4..5 occupy
 * byte 2 bits 6..7. Byte 2 bit 5 is a separate native modifier, not a
 * destination extension; this established scalar form leaves it clear. */
static inline bool
agx_apple9_encode_literal32(uint8_t out[8], unsigned dst, uint32_t value)
{
   if (dst >= 64)
      return false;
   const uint8_t bytes[8] = {
      ((dst & 0xf) << 4) | 0x0c, 0x80 | (value & 0x7f),
      ((dst >> 4) << 6) | 0x02,  (value >> 24) & 0xfe,
      (value >> 6) & 0x1e,       (value >> 9) & 0x0c,
      (value >> 13) & 0xff,      (value >> 21) & 0x0f,
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* T8132 entry-table loads: base is a full address in a GPR pair, offset is
 * in eight-byte units, and results remain pending on dependency slot 2.
 * The established widths are one or two consecutive 64-bit pointers. */
static inline bool
agx_apple9_encode_pointer_load(uint8_t out[14], unsigned dst,
                              unsigned base, unsigned offset, unsigned count)
{
   if ((count != 1 && count != 2) || dst >= 64 || dst + 2 * count > 64 ||
       base >= 63 || offset > 255)
      return false;
   const uint8_t bytes[14] = {
      0x67, 0, 0x54, dst << 1, base, 0, 0, 0,
      count == 2 ? 0x57 : 0x59, 0, offset, 0x40, 0x26, 0,
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Signed, instruction-start-relative branch used by shader control flow and
 * persistent shader entries. The low bit is reserved for instruction alignment. */
static inline bool
agx_apple9_encode_branch(uint8_t out[10], bool any, int64_t displacement)
{
   if (!out || (displacement & 1) || displacement < -(INT64_C(1) << 47) ||
       displacement >= (INT64_C(1) << 47))
      return false;
   uint8_t bytes[10] = {0x0f, any ? 0x00 : 0x01, 0x54};
   for (unsigned byte = 0; byte < 6; ++byte)
      bytes[3 + byte] = (uint64_t)displacement >> (8 * byte);
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

/* Transfer one pending word to the main's argument window. A first transfer
 * waits for the table-load group on slot 2; subsequent transfers reuse it.
 * Argument word 2*i is the low half of pointer i, word 2*i+1 its high half. */
static inline bool
agx_apple9_encode_argument_word(uint8_t out[4], unsigned dst,
                               unsigned src, bool wait)
{
   if (dst >= 64 || src >= 64)
      return false;
   const uint8_t bytes[4] = {
      ((dst & 15) << 4) | 0x0b, (src << 1) & 0x7f,
      9 | ((dst >> 4) << 6), 4 | (wait ? 0x40 : 0),
   };
   memcpy(out, bytes, sizeof(bytes));
   return true;
}

#endif
