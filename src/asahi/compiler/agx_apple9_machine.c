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

#define ALU_SOURCE(_files, _umax, _role, _widths, _max, _align, _flags, _evidence)   \
   {                                                                      \
      .role = (_role),                                                     \
      .files = (_files),                                                   \
      .widths = (_widths), .min_index = 0, .max_index = (_max),             \
      .uniform_max_index = (_umax), .alignment_halves = (_align),                \
      .flags = (_flags), .evidence = (_evidence),                           \
   }

#define REGISTER_SOURCE(...)                                                 \
   ALU_SOURCE(AGX_APPLE9_FILE_GPR | AGX_APPLE9_FILE_UNIFORM, AGX_APPLE9_UNIFORM_COUNT - 1, __VA_ARGS__)
/* Sources whose descriptor also holds a literal: a zero-extended 8-bit
 * integer, or a minifloat for floating min/max (the packer checks which). */
#define INT_IMM_SOURCE(...)                                                  \
   ALU_SOURCE(AGX_APPLE9_FILE_GPR | AGX_APPLE9_FILE_UNIFORM |                \
              AGX_APPLE9_FILE_IMMEDIATE, AGX_APPLE9_UNIFORM_COUNT - 1, __VA_ARGS__)
#define FMA_SOURCE(...)                                                   \
   ALU_SOURCE(AGX_APPLE9_FILE_GPR | AGX_APPLE9_FILE_UNIFORM |                \
              AGX_APPLE9_FILE_IMMEDIATE, AGX_APPLE9_UNIFORM_COUNT - 1, __VA_ARGS__)

#define FMA_LOW_SOURCE(...) \
   ALU_SOURCE(AGX_APPLE9_FILE_GPR | AGX_APPLE9_FILE_UNIFORM | \
              AGX_APPLE9_FILE_IMMEDIATE, 63, __VA_ARGS__)

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
   .occupancy_model = AGX_APPLE9_OCCUPANCY_UNMEASURED,
};

/*
 * This table is deliberately conservative.  "allocator_safe" means every
 * register-bearing field required by the form is understood well enough for
 * a packer, not merely that the compiler has emitted examples of the form.
 * A 7-bit-looking byte is not allocator-safe when its high bits also carry
 * cache, liveness, or source-file state.
 */
static const struct agx_apple9_encoding_info encodings[] = {
   [AGX_APPLE9_ENC_MESH_CONTROL] = {
      .name = "mesh_control", .length = 6, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      /* T8132 r6: either slot 1 or slot 2 completes U0 at the first channel
       * submit. H4/H5 remain equivalent; choose slot 1, without claiming a
       * hidden token count. This is an implicit producer, not an output tag. */
      .fixed_producer_slot = 1,
      .fixed_producer_alternates = 0x02,
      .operands = {
         /* Whole-word allocation conservatively protects the other half.
          * The selected halfword is read and cleared by the instruction. */
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_16, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_16, 95, 2,
             /* Qualified U0 captures preserve the selected B halfword,
              * including low/high remaps and both control42 values. Let its
              * normal SSA uses retain it instead of manufacturing a dead
              * clobber copy. This does not impose a numeric register window. */
             AGX_APPLE9_OPERAND_ALLOCATABLE,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_MESH_SUBMIT_SCALAR] = {
      .name = "mesh_submit_scalar", .length = 6, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .requires_prior_completion = true,
      .operands = {
         /* Conservative even when release=0: preservation is unqualified. */
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 94, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         /* Implicitly addressed at base+1, explicitly tracked by allocation.
          * 0 and 3 pass; 0x10/0x20/0x30/0x40 suppress geometry on T8132.
          * The field meaning and preservation are still unqualified. */
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_MESH_SUBMIT_CHANNEL] = {
      .name = "mesh_submit_channel", .length = 6, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .requires_prior_completion = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 94, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_BYTE_DIFF),
      },
   },
   [AGX_APPLE9_ENC_HALT] = {
      .name = "halt", .length = 4,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .requires_prior_completion = true,
   },
   /* Waits for the scoreboard slots in its mask: byte 0 is 0x06 and bits
    * 10..15 hold the mask, bit i naming slot i + 1; the empty mask is a nop.
    * Hardware-validated as every standalone completion against no-wait,
    * empty-mask and wrong-slot controls. */
   [AGX_APPLE9_ENC_WAIT] = {
      .name = "wait", .length = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
   },
   /* Native T8132 Metal issues both the fence and the barrier with device
    * loads, returning atomics and threadgroup loads outstanding, including
    * before another invocation's store to a loaded location, and waits at
    * the results' consumers. The scheduler still drains the raster-order and
    * object-grid fence forms, which have no such evidence. */
   [AGX_APPLE9_ENC_DEVICE_FENCE] = {
      .name = "device_fence", .length = 6,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
      .requires_prior_completion = true,
      .prior_completion_exempt = AGX_APPLE9_COMPLETION_MEMORY,
   },
   [AGX_APPLE9_ENC_WORKGROUP_BARRIER] = {
      .name = "workgroup_barrier", .length = 6,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .requires_prior_completion = true,
      .prior_completion_exempt = AGX_APPLE9_COMPLETION_MEMORY,
   },
   [AGX_APPLE9_ENC_SHARED_LOAD] = {
      .name = "shared_load", .length = 14, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .fixed_producer_slot = 6,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SHARED_STORE] = {
      .name = "shared_store", .length = 14, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_LOCAL_STORE] = {
      .name = "local_store", .length = 14, .operand_count = 1,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         /* Preserve live data through the allocator, as for shared stores. */
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_LOCAL_STORE_INDEXED] = {
      .name = "local_store_indexed", .length = 14, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SHARED_ATOMIC] = {
      .name = "shared_atomic", .length = 12, .operand_count = 3,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .fixed_producer_slot = 1,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_ATOMIC_DATA, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_PRIVATE_LOAD] = {
      .name = "private_load", .length = 12, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .fixed_producer_slot = 6,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_PRIVATE_STORE] = {
      .name = "private_store", .length = 10, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SUBGROUP_SCAN_IADD] = {
      .name = "subgroup_scan_iadd", .length = 8, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SUBGROUP_REDUCE_UMAX] = {
      .name = "subgroup_reduce_umax", .length = 8, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SUBGROUP_BROADCAST] = {
      .name = "subgroup_broadcast", .length = 10, .operand_count = 3,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SUBGROUP_BALLOT] = {
      .name = "subgroup_ballot", .length = 10, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
             AGX_APPLE9_EVIDENCE_BYTE_DIFF),
      },
   },
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
               /* The short SR forms split the six-bit destination between
                * the first nibble and instruction bits 22..23. */
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE |
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
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE |
                      AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_BIT_UNARY] = {
      .name = "bit_unary",
      .length = 8,
      .operand_count = 2,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
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
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
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
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT_TO_HALF_ZEXT] = {
      .name = "float_to_half_zext",
      .length = 10,
      .operand_count = 2,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_PACK_HALF_2X16] = {
      .name = "pack_half_2x16",
      .length = 12,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_NONE,
      .split_result = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_PACK_UNORM_2X16] = {
      .name = "pack_unorm_2x16",
      .length = 10,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_NONE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_PACK_UNORM_4X8] = {
      .name = "pack_unorm_4x8",
      .length = 20,
      .operand_count = 5,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      /* Two native instructions, each with its own dependency fields. */
      .dependency_layout = AGX_APPLE9_DEPENDENCY_NONE,
      .split_result = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_PACK_FORMAT] = {
      .name = "pack_format",
      .length = 14,
      .operand_count = 5,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_PACK_FORMAT_RGB] = {
      .name = "pack_format_rgb",
      .length = 14,
      .operand_count = 4,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_UNPACK_NORM] = {
      .name = "unpack_norm",
      .length = 8,
      .operand_count = 2,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_64, 94, 4,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_HALF_TO_FLOAT] = {
      .name = "half_to_float",
      .length = 6,
      .operand_count = 2,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_HALF_MINMAX] = {
      .name = "half_minmax",
      .length = 6,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_HALF2] = {
      .name = "half2",
      .length = 6,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_HALF3] = {
      .name = "half3",
      .length = 8,
      .operand_count = 4,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_61_63,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
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
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED] = {
      .name = "float2_immediate_extended",
      .length = 8,
      .operand_count = 2,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
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
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC0,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC1,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED | AGX_APPLE9_OPERAND_UNIFORM_ALTERNATIVE,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT2_BASE] = {
      .name = "float2_base", .length = 4, .operand_count = 3,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_FLOAT3_COMPACT] = {
      .name = "float3_compact", .length = 6, .operand_count = 4,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         FMA_LOW_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
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
   [AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED] = {
      .name = "float2_modifier_extended",
      .length = 8,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED] = {
      .name = "float2_abs_extended",
      .length = 10,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED] = {
      .name = "float2_mul_abs_extended",
      .length = 12,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED] = {
      .name = "float3_modifier_extended",
      .length = 12,
      .operand_count = 4,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_61_63_77_79,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         FMA_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         FMA_SOURCE(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED] = {
      .name = "float3_saturate_extended",
      .length = 10,
      .operand_count = 4,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_61_63_77_79,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         FMA_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         FMA_SOURCE(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
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
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               FMA_SOURCE(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               FMA_SOURCE(
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
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_HALF_SPECIAL] =
      {
         .name = "half_special",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands =
            {
               /* Wide unary operations have seven-bit destination and
                * source fields at bits 25..31 and 42..48. The source field
                * crosses from byte 5 into byte 6 bit 0. */
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
         .dependency_retains_sources = true,
         .operands =
            {
               /* Wide unary operations have seven-bit destination and
                * source fields at bits 25..31 and 42..48. The source field
                * crosses from byte 5 into byte 6 bit 0. */
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT_SPECIAL_UNIFORM] =
      {
         .name = "float_special_uniform",
         .length = 10,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .dependency_retains_sources = true,
         .operands =
            {
               /* Wide unary operations have seven-bit destination and
                * source fields at bits 25..31 and 42..48. The source field
                * crosses from byte 5 into byte 6 bit 0. */
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32,
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(AGX_APPLE9_OPERAND_SRC0,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32,
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(
                  AGX_APPLE9_OPERAND_SRC1,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_UNIFORM_ALTERNATIVE,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_CUBE] =
      {
         .name = "cube",
         .length = 12,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
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
   [AGX_APPLE9_ENC_INT_MAD_EXTENDED] =
      {
         .name = "int_mad_extended",
         .length = 12,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_INT_MUL_WIDE] =
      {
         .name = "int_mul_wide",
         .length = 12,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .dependency_retains_sources = true,
         .operands = {
            GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_64, 94, 4,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
            GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
            GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
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
         /* Native T8132 integer and float min/max wait for a pending load
          * here while keeping that source live for a later read. */
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC0,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(
                  AGX_APPLE9_OPERAND_SRC1,
                  AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32, 95,
                  1, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_HALF_COMPARE_SELECT] =
      {
         .name = "half_compare_select",
         .length = 10,
         .operand_count = 5,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_61_63,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
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
   [AGX_APPLE9_ENC_SELECT_GPR_WIDE] =
      {
         .name = "select_gpr_wide",
         .length = 10,
         .operand_count = 5,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_61_63,
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(
                  AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(
                  AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED] =
      {
         .name = "select_modifier_extended",
         .length = 14,
         .operand_count = 5,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_61_63_93_95,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_COMPACT_PREFERRED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
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
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               REGISTER_SOURCE(
                  AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(
                  AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_LOGIC_UNIFORM] = {
      .name = "logic_uniform", .length = 10, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   /* Metal's four-byte uniform-to-register move (our own kernels): no
    * register source; the destination field reaches r63. */
   [AGX_APPLE9_ENC_UNIFORM_MOVE] = {
      .name = "uniform_move", .length = 4, .operand_count = 1,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_NONE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 63, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED,
             AGX_APPLE9_EVIDENCE_BYTE_DIFF),
      },
   },
   /* Uniform stores use the long integer destination file, independently of GPRs. */
   [AGX_APPLE9_ENC_STORE_UNIFORM] = {
      .name = "store_uniform", .length = 10, .operand_count = 2,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .dependency_retains_sources = true,
      .operands = {
         UNIFORM(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 AGX_APPLE9_UNIFORM_COUNT - 1, 2, 0,
                 AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_STORE_UNIFORM_ADD] = {
      .name = "store_uniform_add", .length = 10, .operand_count = 3,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .operands = {
         UNIFORM(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 AGX_APPLE9_UNIFORM_COUNT - 1, 2, 0,
                 AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_STORE_UNIFORM_MAD] = {
      .name = "store_uniform_mad", .length = 12, .operand_count = 4,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .operands = {
         UNIFORM(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 AGX_APPLE9_UNIFORM_COUNT - 1, 2, 0,
                 AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_STORE_UNIFORM_MUL_WIDE] = {
      .name = "store_uniform_mul_wide", .length = 12, .operand_count = 3,
      .allocator_safe = true, .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .operands = {
         UNIFORM(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_64,
                 AGX_APPLE9_UNIFORM_COUNT - 2, 4, 0,
                 AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
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
         .dependency_retains_sources = true,
         .operands =
            {
               GPR(
                  AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                  AGX_APPLE9_PUBLICATION_COUNT - 1,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION | AGX_APPLE9_OPERAND_SCATTERED,
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
   [AGX_APPLE9_ENC_SHIFT_EXTENDED] = {
      .name = "shift_extended",
      .length = 10,
      .operand_count = 2,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SHIFT_ARITH_REGISTER] = {
      .name = "shift_arith_register",
      .length = 10,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE] = {
      .name = "shift_logical_immediate",
      .length = 12,
      .operand_count = 2,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_SHIFT_LOGICAL_REGISTER] = {
      .name = "shift_logical_register",
      .length = 12,
      .operand_count = 3,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
      .dependency_retains_sources = true,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_HALF_PREDICATE_SHORT] =
      {
         .name = "half_predicate_short",
         .length = 6,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               /* Both sources use the ordinary 32-bit descriptor
                * (gpr << 1) | 1.  Source lifetime is encoded independently
                * in byte 2. */
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT] =
      {
         .name = "predicate_compare_short",
         .length = 6,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_45_47,
         .dependency_retains_sources = true,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               /* Both sources use the ordinary 32-bit descriptor
                * (gpr << 1) | 1.  Source lifetime is encoded independently
                * in byte 2. */
               REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_HALF_PREDICATE_EXTENDED] =
      {
         .name = "half_predicate_extended",
         .length = 10,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_61_63,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED] =
      {
         .name = "predicate_compare_extended",
         .length = 10,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_61_63,
         .dependency_retains_sources = true,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               REGISTER_SOURCE(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               INT_IMM_SOURCE(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP] =
      {
         .name = "predicate_compare_loop",
         .length = 10,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_INDEX_61_63,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
         .length = 4,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .fixed_producer_slot = 1,
         .operands =
            {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW,
                 AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_GET_COVERAGE] =
      {
         .name = "get_coverage",
         .length = 4,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .fixed_producer_slot = 1,
         .operands =
            {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_HARD_LOW,
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
                          31, 4, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          31, 4, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          31, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_BLOCK_STORE_PARAMS] =
      {
         .name = "block_store_params",
         .length = 30,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          15, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_BLOCK_STORE_EXTENDED_PARAMS] =
      {
         .name = "block_store_extended_params",
         .length = 40,
         .operand_count = 5,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          15, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                      GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_IMAGE_WRITE_PARAMS] = {
      .name = "image_write_params", .length = 80,
      .operand_count = 9, .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 24, 16,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC4, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC5, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC6, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC7, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   /* Synchronous PBE write: issue and join the export completion domain.
    * This shares that domain with block-image exports; ordinary GPR and
    * publication indices are independently allocated. */
   [AGX_APPLE9_ENC_IMAGE_WRITE] = {
      .name = "image_write", .length = 22,
      .operand_count = 10, .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 24, 16,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 25, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 26, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 27, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC4, AGX_APPLE9_WIDTH_32, 28, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC5, AGX_APPLE9_WIDTH_32, 29, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC6, AGX_APPLE9_WIDTH_32, 30, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC7, AGX_APPLE9_WIDTH_32, 31, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC8, AGX_APPLE9_WIDTH_32, 94, 4,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC9, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
      },
   },
   [AGX_APPLE9_ENC_BLOCK_IMAGE_STORE] = {
      .name = "block_image_store", .length = 18,
      .operand_count = 3, .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 15, 8,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 15, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 15, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE)},
   },
   [AGX_APPLE9_ENC_BLOCK_IMAGE_STORE_EXTENDED] = {
      .name = "block_image_store_extended", .length = 18,
      .operand_count = 4, .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 15, 8,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 15, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 15, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
         GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 15, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE)},
   },
   [AGX_APPLE9_ENC_TEXTURE_LOD_PARAMS] =
      {
         .name = "texture_lod_params",
         .length = 30,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                          31, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                          31, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                      GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                          63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          31, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          31, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                          31, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32,
                          31, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                   31, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
               GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC4, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC5, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_TEXTURE_GRAD_ARRAY_PARAMS] = {
      .name = "texture_grad_array_params",
      .length = 80,
      .operand_count = 9,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 31, 8,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC4, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC5, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC6, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC7, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
      },
   },
   [AGX_APPLE9_ENC_TEXTURE_GRAD_VOLUME_PARAMS] = {
      .name = "texture_grad_volume_params",
      .length = 90,
      .operand_count = 10,
      .allocator_safe = true,
      .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
      .operands = {
         GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 31, 8,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC3, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC4, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC5, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC6, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC7, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
         GPR(AGX_APPLE9_OPERAND_SRC8, AGX_APPLE9_WIDTH_32, 95, 2,
             AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
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
                   63, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   31, 8, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_FLOAT2_EXPORT] =
      {
         .name = "float2_export",
         .length = 8,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
         .dependency_retains_sources = true,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32,
                 AGX_APPLE9_PUBLICATION_COUNT - 1,
                 2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION,
                 AGX_APPLE9_EVIDENCE_HARDWARE),
             GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                 95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                 AGX_APPLE9_EVIDENCE_HARDWARE),
             GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                 95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                 AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_ITER_FLAT] =
      {
         .name = "iter_flat",
         .length = 6,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 93,
                          2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_INTERPOLATION_POSITION] =
      {
         .name = "interpolation_position",
         .length = 8,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operands = {
            GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
            /* The emitted form overwrites its sample-index input; a value
             * used afterwards needs a disposable copy (D32 MSAA readback). */
            GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 4,
                AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
                AGX_APPLE9_EVIDENCE_HARDWARE),
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
            GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_BYTE_DIFF),
            GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 1,
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
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
                          2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_VARY_STORE] =
      {
         .name = "vary_store",
         .length = 8,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                 AGX_APPLE9_PUBLICATION_COUNT - 1,
                 2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION,
                 AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_OBJECT_GRID_EXPORT] =
      {
         .name = "object_grid_export",
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
   [AGX_APPLE9_ENC_OBJECT_CALL] = {
      .name = "object_call", .length = 10, .allocator_safe = true,
      .requires_prior_completion = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
   },
   [AGX_APPLE9_ENC_OBJECT_RETURN] = {
      .name = "object_return", .length = 4, .allocator_safe = true,
      .requires_prior_completion = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
   },
   [AGX_APPLE9_ENC_OBJECT_RESUME] = {
      .name = "object_resume", .length = 6, .allocator_safe = true,
      .requires_prior_completion = true,
      .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
   },
   [AGX_APPLE9_ENC_PUBLICATION_PAIR] =
      {
         .name = "publication_pair",
         .length = 20,
         .operand_count = 3,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {
            GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 62, 4,
                AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION,
                AGX_APPLE9_EVIDENCE_HARDWARE),
            GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE),
            GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 95, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE, AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_DEPTH_STORE] =
      {
         .name = "depth_store",
         .length = 6,
         .operand_count = 2,
         .allocator_safe = true,
         .requires_prior_completion = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {
            GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 62, 4,
                AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION,
                AGX_APPLE9_EVIDENCE_HARDWARE),
            GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 63, 2,
                AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_PUBLICATION,
                AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_COVERAGE] =
      {
         .name = "coverage",
         .length = 6,
         /* Native captures carry byte 1 as 0x14 or 0x04: a slot-1 wait or
          * none, at the same bits as device memory's input mask. Native
          * fragments with discard or depth output wait there for the
          * coverage publication rather than before the tile access. */
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operand_count = 1,
         .allocator_safe = true,
         .requires_prior_completion = true,
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
         .requires_prior_completion = true,
         /* Native fragments open the test domain with the coverage
          * publication still pending; the coverage instruction waits. */
         .prior_completion_exempt = AGX_APPLE9_COMPLETION_PUBLICATION,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_TILE_LOAD] =
      {
         .name = "tile_load",
         .length = 12,
         .operand_count = 1,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands = {GPR(AGX_APPLE9_OPERAND_DEST, AGX_APPLE9_WIDTH_32, 95,
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
         .operands = {GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32, 95,
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
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_BYTE_DIFF),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE),
                      GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_HARDWARE)},
      },
   [AGX_APPLE9_ENC_TILE_STORE_MASK] =
      {
         .name = "tile_store_mask",
         .length = 12,
         /* Native dynamic-mask tile stores carry byte 1 as 0x16 or 0x06,
          * waiting for the coverage read that supplies their mask or not. */
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operand_count = 2,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
         .operands = {GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_BYTE_DIFF),
                      GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                          95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                          AGX_APPLE9_EVIDENCE_BYTE_DIFF)},
      },
   [AGX_APPLE9_ENC_TILE_FENCE] =
      {
         .name = "tile_fence",
         .length = 6,
         .operand_count = 0,
         .allocator_safe = true,
         .requires_prior_completion = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
      },
   [AGX_APPLE9_ENC_DEVICE_LOAD] =
      {
         .name = "device_load",
         .length = 14,
         .operand_count = 2,
         .allocator_safe = true,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .dependency_retains_sources = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32 |
                      AGX_APPLE9_WIDTH_64,
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .dependency_retains_sources = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32 |
                      AGX_APPLE9_WIDTH_64,
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   94, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_BYTE_DIFF),
               GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_BYTE_DIFF),
            },
      },
   /* A pointer-pair load without an index operand; the pair alone forms the
    * address, plus the byte displacement. */
   [AGX_APPLE9_ENC_DEVICE_LOAD_POINTER] =
      {
         .name = "device_load_pointer",
         .length = 14,
         .operand_count = 3,
         .allocator_safe = true,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .dependency_retains_sources = true,
         .evidence = AGX_APPLE9_EVIDENCE_BYTE_DIFF,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_DEST,
                   AGX_APPLE9_WIDTH_16 | AGX_APPLE9_WIDTH_32 |
                      AGX_APPLE9_WIDTH_64,
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   94, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_BYTE_DIFF),
               GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
                   95, 1, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32,
                   94, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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
               /* Both split seven-bit fields address the full GPR file.
                * Returning operations name their destination separately. */
               GPR(
                  AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_CLOBBER,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(
                  AGX_APPLE9_OPERAND_ATOMIC_DATA, AGX_APPLE9_WIDTH_32, 95,
                  2, AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED | AGX_APPLE9_OPERAND_CLOBBER,
                  AGX_APPLE9_EVIDENCE_HARDWARE),
            },
      },
   [AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT] =
      {
         .name = "device_atomic_indirect",
         .length = 14,
         .dependency_layout = AGX_APPLE9_DEPENDENCY_MASK_12_17,
         .operand_count = 4,
         .allocator_safe = true,
         .evidence = AGX_APPLE9_EVIDENCE_HARDWARE,
         .operands =
            {
               GPR(AGX_APPLE9_OPERAND_INDEX, AGX_APPLE9_WIDTH_32, 95, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED |
                      AGX_APPLE9_OPERAND_CLOBBER,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_ATOMIC_DATA, AGX_APPLE9_WIDTH_32, 95, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_SCATTERED |
                      AGX_APPLE9_OPERAND_CLOBBER,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC1, AGX_APPLE9_WIDTH_32, 94, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
                   AGX_APPLE9_EVIDENCE_HARDWARE),
               GPR(AGX_APPLE9_OPERAND_SRC2, AGX_APPLE9_WIDTH_32, 95, 2,
                   AGX_APPLE9_OPERAND_ALLOCATABLE | AGX_APPLE9_OPERAND_CLOBBER,
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
                * atomic has a seven-bit destination: four bits in the
                * first nibble, two at 22..23, and the last at bit 60. */
               GPR(AGX_APPLE9_OPERAND_SRC0, AGX_APPLE9_WIDTH_32,
                   95, 2, AGX_APPLE9_OPERAND_ALLOCATABLE,
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

   unsigned gpr_index = 0;
   for (unsigned i = 0; i < info->operand_count; ++i) {
      const struct agx_apple9_operand_constraint *operand = &info->operands[i];
      if (!(operand->files & AGX_APPLE9_FILE_GPR))
         continue;
      if (!agx_apple9_encoding_accepts_gpr(encoding, operand->role,
                                           gprs[gpr_index], bits))
         return false;
      ++gpr_index;
   }

   return true;
}

static_assert((sizeof(encodings) / sizeof(encodings[0])) ==
                 AGX_APPLE9_ENC_COUNT,
              "every Apple9 encoding needs a machine-model entry");
