/* SPDX-License-Identifier: MIT */
#ifndef AGX_APPLE9_LAYOUT_H
#define AGX_APPLE9_LAYOUT_H

/* The launch target carries 2 * entry_offset + 0x2a in 24 bits. Entries
 * reside in this compact USC range and branch to independently allocated
 * shader bodies anywhere in the 4-GiB USC heap. An entry belongs to its
 * shader BO and cannot be recycled while any batch references that BO. */
#define AGX_APPLE9_ENTRY_ARENA_SIZE  0x00800000u
#define AGX_APPLE9_ENTRY_HEADER_SIZE 0x00000340u
#define AGX_APPLE9_ENTRY_BLOCK_SIZE  0x000000c0u
#define AGX_APPLE9_ENTRY_CODE_OFFSET 0x00000080u
#define AGX_APPLE9_ENTRY_ALIGNMENT   0x00000040u
#define AGX_APPLE9_ENTRY_MAX_OFFSET  ((0xffffffu - 0x2au) / 2u)

#endif
