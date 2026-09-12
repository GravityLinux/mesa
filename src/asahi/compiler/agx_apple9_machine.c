/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_machine.h"

#include <assert.h>
#include <stddef.h>

#define GPR(_role, _widths, _max, _align, _flags, _evidence)                   \
   {                                                                           \
      .role = (_role),                                                         \
      .files = AGX_APPLE9_FILE_GPR,                                            \
      .widths = (_widths),                                                     \
      .min_index = 0,                                                          \
      .max_index = (_max),                                                     \
      .alignment_halves = (_align),                                            \
      .flags = (_flags),                                                       \
      .evidence = (_evidence),                                                 \
   }

#define GPR_RANGE(_role, _widths, _min, _max, _align, _flags, _evidence)       \
   {                                                                           \
      .role = (_role),                                                         \
      .files = AGX_APPLE9_FILE_GPR,                                            \
      .widths = (_widths),                                                     \
      .min_index = (_min),                                                     \
      .max_index = (_max),                                                     \
      .alignment_halves = (_align),                                            \
      .flags = (_flags),                                                       \
      .evidence = (_evidence),                                                 \
   }

#define UNIFORM(_role, _widths, _max, _align, _flags, _evidence)               \
   {                                                                           \
      .role = (_role),                                                         \
      .files = AGX_APPLE9_FILE_UNIFORM,                                        \
      .widths = (_widths),                                                     \
      .min_index = 0,                                                          \
      .max_index = (_max),                                                     \
      .alignment_halves = (_align),                                            \
      .flags = (_flags),                                                       \
      .evidence = (_evidence),                                                 \
   }

#define UNIFORM_RANGE(_role, _widths, _min, _max, _align, _flags, _evidence)   \
   {                                                                           \
      .role = (_role),                                                         \
      .files = AGX_APPLE9_FILE_UNIFORM,                                        \
      .widths = (_widths),                                                     \
      .min_index = (_min),                                                     \
      .max_index = (_max),                                                     \
      .alignment_halves = (_align),                                            \
      .flags = (_flags),                                                       \
      .evidence = (_evidence),                                                 \
   }

#define IMPLICIT(_role, _widths, _evidence)                                    \
   {                                                                           \
      .role = (_role),                                                         \
      .files = AGX_APPLE9_FILE_IMPLICIT,                                       \
      .widths = (_widths),                                                     \
      .min_index = 0xff,                                                       \
      .max_index = 0xff,                                                       \
      .alignment_halves = 1,                                                   \
      .flags = AGX_APPLE9_OPERAND_IMPLICIT,                                    \
      .evidence = (_evidence),                                                 \
   }

const struct agx_apple9_machine agx_apple9_machine = {
   .gpr_count = AGX_APPLE9_GPR_COUNT,
   .half_register_count = AGX_APPLE9_HALF_REGISTER_COUNT,
   .hardware_register_interlocks = true,
   .software_waits = false,
   .spilling_supported = true,
   .occupancy_model = AGX_APPLE9_OCCUPANCY_PRESSURE_TIER,
};

/*
 * This table is deliberately conservative.  "allocator_safe" means every
 * register-bearing field required by the form is understood well enough for
 * a packer, not merely that the compiler has emitted examples of the form.
 * A 7-bit-looking byte is not allocator-safe when its high bits also carry
 * cache, liveness, or source-file state.
 */
static const struct agx_apple9_encoding_info encodings[] = {
   /* EXP-M4-60: ordinary 32-bit GPRs and a separate word-indexed scratch file.
    * SAVE/FILL each produce a completion tag and accept an incoming wait. */
   [AGX_APPLE9_ENC_SPILL_STORE] =
      {
         .name = "spill_store",
         .length = 10,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
         .operands = {GPR(
            AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
            AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
            AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_SPILL_LOAD] =
      {
         .name = "spill_load",
         .length = 8,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
         .operands = {GPR(
            AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
            AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
            AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_MOV_IMM_COMPACT] =
      {
         .name = "mov_imm_compact",
         .length = 2,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 15, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW |
                      AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_MOV_IMM32] =
      {
         .name = "mov_imm32",
         .length = 8,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               /* EXP-M4-37 hardware-validates the complete split six-bit
                * destination map.  Mode 2 reaches r0..r63; r64+ cannot be
                * represented by this form. */
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_GET_SR] =
      {
         .name = "get_sr",
         .length = 4,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               /* Native streams encode higher destinations, but the retained
                * G16 splice probes did not validate their publication to a
                * consumer.  Keep allocator-authored GET_SR values in the
                * hardware-proven compact bank and copy them out through the
                * independently validated extended logic form. */
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 15, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW |
                      AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_GET_SR_ZEXT16] =
      {
         .name = "get_sr_zext16",
         .length = 8,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 15, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW |
                      AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_UINT_TO_FLOAT] =
      {
         .name = "uint_to_float",
         .length = 8,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_SINT_TO_FLOAT] =
      {
         .name = "sint_to_float",
         .length = 8,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT_TO_SINT] =
      {
         .name = "float_to_sint",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT_TO_UINT] =
      {
         .name = "float_to_uint",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT] =
      {
         .name = "float2_immediate_compact",
         .length = 6,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 15,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT2_COMPACT] =
      {
         .name = "float2_compact",
         .length = 6,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC1,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED | AGX_APPLE9_OPERAND_UNIFORM_ALTERNATIVE,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT2_PROJECT] =
      {
         .name = "float2_project",
         .length = 8,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_NONE,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED] =
      {
         .name = "float2_modifier_extended",
         .length = 8,
         .operand_count = 3,
         .allocator_safe = false,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32,
                   95, 1, AGX_APPLE9_OPERAND_SCATTERED,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32,
                   95, 1, AGX_APPLE9_OPERAND_SCATTERED,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32,
                   95, 1, AGX_APPLE9_OPERAND_SCATTERED,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT3_EXTENDED] =
      {
         .name = "float3_extended",
         .length = 8,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_61_63,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_DERIVATIVE] =
      {
         .name = "derivative",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT_SPECIAL] =
      {
         .name = "float_special",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               /* EXP-M4-41 exhaustively validates the descriptor geometry:
                * byte 3 names r0..r95 as dst<<1, while byte 5 names only
                * r0..r63 as src<<2. */
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_INT_ADD_EXTENDED] =
      {
         .name = "int_add_extended",
         .length = 10,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32,
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32,
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC1,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_UNIFORM_ALTERNATIVE,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_INT_MAD_EXTENDED] =
      {
         .name = "int_mad_extended",
         .length = 12,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_MINMAX_COMPACT] =
      {
         .name = "minmax_compact",
         .length = 6,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC1,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_SELECT_GPR_WIDE] =
      {
         .name = "select_gpr_wide",
         .length = 10,
         .operand_count = 5,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_61_63,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 15,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_LOGIC_EXTENDED] =
      {
         .name = "logic_extended",
         .length = 10,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
         .max_high_gpr_operands = 2,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_LOGIC_EXPORT] =
      {
         .name = "logic_export",
         .length = 10,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
         .max_high_gpr_operands = 2,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                  AGX_APPLE9_PUBLICATION_COUNT - 1,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   /* The currently encoded shift form releases its source. Allocation must
    * preserve a surviving value; a retain encoding has not been established. */
   [AGX_APPLE9_ENC_SHIFT_EXTENDED] =
      {
         .name = "shift_extended",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   15, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 15,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT] =
      {
         .name = "predicate_compare_short",
         .length = 6,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               /* Both sources use the ordinary 32-bit descriptor
                * (gpr << 1) | 1.  Source lifetime is encoded independently
                * in byte 2. */
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED] =
      {
         .name = "predicate_compare_extended",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP] =
      {
         .name = "predicate_compare_loop",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_EXEC_MASK_PUSH] =
      {
         .name = "exec_mask_push",
         .length = 4,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_EXEC_MASK_ELSE] =
      {
         .name = "exec_mask_else",
         .length = 4,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_EXEC_MASK_POP] =
      {
         .name = "exec_mask_pop",
         .length = 6,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_LOOP_MASK_PUSH] =
      {
         .name = "loop_mask_push",
         .length = 4,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_LOOP_MASK_UPDATE] =
      {
         .name = "loop_mask_update",
         .length = 4,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_LOOP_MASK_POP] =
      {
         .name = "loop_mask_pop",
         .length = 6,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_JMP_EXEC_ANY] =
      {
         .name = "jmp_exec_any",
         .length = 10,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_JMP_EXEC_NONE] =
      {
         .name = "jmp_exec_none",
         .length = 10,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_BREAK_MASK_UNWIND] =
      {
         .name = "break_mask_unwind",
         .length = 6,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_GET_DRAW_ID] =
      {
         .name = "get_draw_id",
         .length = 14,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 15, 2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW,
                 AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_GET_COVERAGE] =
      {
         .name = "get_coverage",
         .length = 14,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 15, 2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW,
                 AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TEXTURE_COORDS] =
      {
         .name = "texture_coords",
         .length = 16,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          7, 4, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TEXTURE_SAMPLE] =
      {
         .name = "texture_sample",
         .length = 14,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          15, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          7, 4, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          7, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TEXTURE_FETCH_PARAMS] =
      {
         .name = "texture_fetch_params",
         .length = 32,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          7, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TEXTURE_FETCH] =
      {
         .name = "texture_fetch",
         .length = 14,
         .operand_count = 5,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          15, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          7, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          7, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          7, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32,
                          7, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TEXTURE_LOD_PARAMS] =
      {
         .name = "texture_lod_params",
         .length = 30,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          7, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TEXTURE_VOLUME_PARAMS] =
      {
         .name = "texture_volume_params",
         .length = 40,
         .operand_count = 5,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          7, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TEXTURE_LOD] =
      {
         .name = "texture_lod",
         .length = 14,
         .operand_count = 5,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          15, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          7, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          7, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          7, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32,
                          7, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TEXTURE_GRAD_PARAMS] =
      {
         .name = "texture_grad_params",
         .length = 60,
         .operand_count = 7,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   7, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC4, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC5, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_TEXTURE_GRAD] =
      {
         .name = "texture_grad",
         .length = 14,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   15, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   7, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT2_EXPORT] =
      {
         .name = "float2_export",
         .length = 8,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 AGX_APPLE9_PUBLICATION_COUNT - 1,
                 2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION,
                 AGX_APPLE9_EVIDENCE_HARDWARE),
             GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                 63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                 AGX_APPLE9_EVIDENCE_HARDWARE),
             GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                 63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                 AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_ITER_FLAT] =
      {
         .name = "iter_flat",
         .length = 6,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 61,
                          2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_CENTROID_POSITION] =
      {
         .name = "centroid_position",
         .length = 8,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands = {
            GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
            GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 4,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         },
      },
   [AGX_APPLE9_ENC_ITER_COORD] =
      {
         .name = "iter_coord",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
         .operands = {
            GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
            GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 1,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         },
      },
   [AGX_APPLE9_ENC_ITER] =
      {
         .name = "iter",
         .length = 10,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63,
                          2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_VARY_STORE] =
      {
         .name = "vary_store",
         .length = 8,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                 AGX_APPLE9_PUBLICATION_COUNT - 1,
                 2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION,
                 AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_COVERAGE] =
      {
         .name = "coverage",
         .length = 6,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63,
                          2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TILE_ACCESS] =
      {
         .name = "tile_access",
         .length = 6,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_TILE_LOAD] =
      {
         .name = "tile_load",
         .length = 12,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63,
                          2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TILE_STORE] =
      {
         .name = "tile_store",
         .length = 12,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63,
                          2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TILE_LOAD_MASK] =
      {
         .name = "tile_load_mask",
         .length = 12,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_BYTE_DIFF),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_BYTE_DIFF)},
      },
   [AGX_APPLE9_ENC_TILE_LOAD_COORDS] =
      {
         .name = "tile_load_coords",
         .length = 12,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TILE_STORE_MASK] =
      {
         .name = "tile_store_mask",
         .length = 12,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
         .operands = {GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_BYTE_DIFF),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_BYTE_DIFF)},
      },
   [AGX_APPLE9_ENC_TILE_FENCE] =
      {
         .name = "tile_fence",
         .length = 6,
         .operand_count = 0,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_DEVICE_LOAD] =
      {
         .name = "device_load",
         .length = 14,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32 |
                      AGX_APPLE9_WIDTH_64,
                   63, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT] =
      {
         .name = "device_load_indirect",
         .length = 14,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32 |
                      AGX_APPLE9_WIDTH_64,
                   63, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   62, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_BYTE_DIFF),
               GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_BYTE_DIFF),
            },
      },
   [AGX_APPLE9_ENC_DEVICE_STORE] =
      {
         .name = "device_store",
         .length = 14,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_STORE_DATA,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32 |
                      AGX_APPLE9_WIDTH_64,
                   63, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT] =
      {
         .name = "device_store_indirect",
         .length = 14,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_STORE_DATA,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32 |
                      AGX_APPLE9_WIDTH_64,
                   63, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   62, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_DEVICE_ATOMIC] =
      {
         .name = "device_atomic",
         .length = 14,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               /* The field is syntactically wider, but all native and
                * hardware-validated atomic-address witnesses currently use
                * the compact bank (r0..r6). Keep RA inside that proven bank
                * until a dedicated high-index splice establishes otherwise. */
               GPR(
                  AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32, 15,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_CLOBBER,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_ATOMIC_DATA, AGX_APPLE9_WIDTH_32, 15,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_CLOBBER,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT] =
      {
         .name = "device_atomic_result",
         .length = 8,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               /* The long low-nibble-c instruction following a returning
                * atomic has a hardware-validated six-bit destination. */
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
};

const struct agx_apple9_encoding_info *
agx_apple9_encoding_info(enum agx_apple9_encoding encoding)
{
   assert(encoding < AGX_APPLE9_ENC_COUNT);
   return &encodings[encoding];
}

const struct agx_apple9_operand_constraint *
agx_apple9_find_operand(enum agx_apple9_encoding encoding,
                        enum agx_apple9_operand_role role)
{
   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(encoding);

   for (unsigned i = 0; i < info->operand_count; ++i) {
      if (info->operands[i].role == role)
         return &info->operands[i];
   }

   return NULL;
}

bool
agx_apple9_encoding_accepts_gpr(enum agx_apple9_encoding encoding,
                                enum agx_apple9_operand_role role, unsigned gpr,
                                unsigned bits)
{
   const struct agx_apple9_operand_constraint *operand =
      agx_apple9_find_operand(encoding, role);

   if (operand == NULL || !(operand->files & AGX_APPLE9_FILE_GPR) ||
       !(operand->flags & AGX_APPLE9_OPERAND_ALLOCATABLE) ||
       gpr >= ((operand->flags & AGX_APPLE9_OPERAND_PUBLICATION)
                  ? AGX_APPLE9_PUBLICATION_COUNT : AGX_APPLE9_GPR_COUNT) ||
       gpr < operand->min_index ||
       gpr > operand->max_index)
      return false;

   unsigned width = bits == 16   ? AGX_APPLE9_WIDTH_16
                    : bits == 32 ? AGX_APPLE9_WIDTH_32
                    : bits == 64 ? AGX_APPLE9_WIDTH_64
                                 : 0;

   if (!(operand->widths & width))
      return false;

   unsigned first_half = 2 * gpr;
   unsigned natural_alignment = bits / 16;
   unsigned alignment = operand->alignment_halves > natural_alignment
                           ? operand->alignment_halves
                           : natural_alignment;
   return (first_half % alignment) == 0;
}

bool
agx_apple9_encoding_accepts_gpr_tuple(enum agx_apple9_encoding encoding,
                                      const unsigned *gprs,
                                      unsigned operand_count, unsigned bits)
{
   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(encoding);

   unsigned gpr_operand_count = 0;
   for (unsigned i = 0; i < info->operand_count; ++i)
      gpr_operand_count += !!(info->operands[i].files & AGX_APPLE9_FILE_GPR);

   if (operand_count != gpr_operand_count)
      return false;

   unsigned high = 0;
   unsigned gpr_index = 0;
   for (unsigned i = 0; i < info->operand_count; ++i) {
      const struct agx_apple9_operand_constraint *operand = &info->operands[i];
      if (!(operand->files & AGX_APPLE9_FILE_GPR))
         continue;
      if (!agx_apple9_encoding_accepts_gpr(encoding, operand->role,
                                           gprs[gpr_index], bits))
         return false;
      high += gprs[gpr_index++] >= 64;
   }

   return info->max_high_gpr_operands == 0 ||
          high <= info->max_high_gpr_operands;
}

static_assert((sizeof(encodings) / sizeof(encodings[0])) ==
                 AGX_APPLE9_ENC_COUNT,
              "every Apple9 encoding needs a machine-model entry");
