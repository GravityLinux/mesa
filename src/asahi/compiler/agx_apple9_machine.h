/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_APPLE9_MACHINE_H
#define AGX_APPLE9_MACHINE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Apple9 uses two independently addressable 16-bit halves per 32-bit GPR. */
#define AGX_APPLE9_PUBLICATION_COUNT 128
#define AGX_APPLE9_GPR_COUNT           96
#define AGX_APPLE9_HALF_REGISTER_COUNT (2 * AGX_APPLE9_GPR_COUNT)

enum agx_apple9_evidence {
   AGX_APPLE9_EVIDENCE_UNRESOLVED = 0,
   AGX_APPLE9_EVIDENCE_BYTE_DIFF,
   AGX_APPLE9_EVIDENCE_HARDWARE,
};

enum agx_apple9_register_file {
   AGX_APPLE9_FILE_NONE = 0,
   AGX_APPLE9_FILE_GPR = 1 << 0,
   AGX_APPLE9_FILE_UNIFORM = 1 << 1,
   AGX_APPLE9_FILE_IMMEDIATE = 1 << 2,
   AGX_APPLE9_FILE_IMPLICIT = 1 << 3,
};

enum agx_apple9_operand_role {
   AGX_APPLE9_OPERAND_DEST,
   AGX_APPLE9_OPERAND_SRC0,
   AGX_APPLE9_OPERAND_SRC1,
   AGX_APPLE9_OPERAND_SRC2,
   AGX_APPLE9_OPERAND_SRC3,
   AGX_APPLE9_OPERAND_SRC4,
   AGX_APPLE9_OPERAND_SRC5,
   AGX_APPLE9_OPERAND_INDEX,
   AGX_APPLE9_OPERAND_STORE_DATA,
   AGX_APPLE9_OPERAND_ATOMIC_DATA,
};

enum agx_apple9_width {
   AGX_APPLE9_WIDTH_16 = 1 << 0,
   AGX_APPLE9_WIDTH_32 = 1 << 1,
   AGX_APPLE9_WIDTH_64 = 1 << 2,
};

enum agx_apple9_operand_flag {
   /* The bit packing is sufficiently understood for physical allocation. */
   AGX_APPLE9_OPERAND_ALLOCATABLE = 1 << 0,

   /* This is a hard encoding restriction, not merely a size preference. */
   AGX_APPLE9_OPERAND_HARD_LOW = 1 << 1,

   /* Prefer this range for code size, but another encoding may exist. */
   AGX_APPLE9_OPERAND_COMPACT_PREFERRED = 1 << 2,

   /* Register bits are distributed among source/liveness/cache fields. */
   AGX_APPLE9_OPERAND_SCATTERED = 1 << 3,

   /* The consumer obtains this value from the immediately preceding op. */
   AGX_APPLE9_OPERAND_IMPLICIT = 1 << 4,

   /* A separate encoding can select the uniform file in this operand slot. */
   AGX_APPLE9_OPERAND_UNIFORM_ALTERNATIVE = 1 << 5,

   /* The instruction destroys this input even when its value has later uses. */
   AGX_APPLE9_OPERAND_CLOBBER = 1 << 6,

   /* Scalar index addresses export storage rather than the ordinary GPR bank. */
   AGX_APPLE9_OPERAND_PUBLICATION = 1 << 7,
};

struct agx_apple9_operand_constraint {
   enum agx_apple9_operand_role role;
   uint8_t files;
   uint8_t widths;

   /* Inclusive index range in the selected file; 0xff when not applicable. */
   uint8_t min_index;
   uint8_t max_index;

   /* Alignment in 16-bit allocation units. */
   uint8_t alignment_halves;
   uint8_t flags;
   enum agx_apple9_evidence evidence;
};

/*
 * These are encoding forms, not NIR operations.  In particular, an operation
 * may have both a compact low-register form and a longer form with a different
 * source layout.  The allocator must choose a form before the packer commits
 * physical registers.
 */
enum agx_apple9_encoding {
   AGX_APPLE9_ENC_MOV_IMM_COMPACT,
   AGX_APPLE9_ENC_MOV_IMM32,
   AGX_APPLE9_ENC_GET_SR,
   AGX_APPLE9_ENC_GET_SR_ZEXT16,
   AGX_APPLE9_ENC_UINT_TO_FLOAT,
   AGX_APPLE9_ENC_SINT_TO_FLOAT,
   AGX_APPLE9_ENC_FLOAT_TO_SINT,
   AGX_APPLE9_ENC_FLOAT_TO_UINT,
   AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT,
   AGX_APPLE9_ENC_FLOAT2_COMPACT,
   AGX_APPLE9_ENC_FLOAT2_PROJECT,
   AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED,
   AGX_APPLE9_ENC_FLOAT3_EXTENDED,
   AGX_APPLE9_ENC_FLOAT_SPECIAL,
   AGX_APPLE9_ENC_DERIVATIVE,
   AGX_APPLE9_ENC_INT_ADD_EXTENDED,
   AGX_APPLE9_ENC_INT_MAD_EXTENDED,
   AGX_APPLE9_ENC_MINMAX_COMPACT,
   AGX_APPLE9_ENC_SELECT_GPR_WIDE,
   AGX_APPLE9_ENC_LOGIC_EXTENDED,
   AGX_APPLE9_ENC_LOGIC_EXPORT,
   AGX_APPLE9_ENC_SHIFT_EXTENDED,
   AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT,
   AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED,
   AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP,
   AGX_APPLE9_ENC_EXEC_MASK_PUSH,
   AGX_APPLE9_ENC_EXEC_MASK_ELSE,
   AGX_APPLE9_ENC_EXEC_MASK_POP,
   AGX_APPLE9_ENC_LOOP_MASK_PUSH,
   AGX_APPLE9_ENC_LOOP_MASK_UPDATE,
   AGX_APPLE9_ENC_LOOP_MASK_POP,
   AGX_APPLE9_ENC_JMP_EXEC_ANY,
   AGX_APPLE9_ENC_JMP_EXEC_NONE,
   AGX_APPLE9_ENC_BREAK_MASK_UNWIND,
   AGX_APPLE9_ENC_GET_VERTEX_ID,
   AGX_APPLE9_ENC_GET_COVERAGE,
   AGX_APPLE9_ENC_FLOAT2_EXPORT,
   AGX_APPLE9_ENC_TEXTURE_COORDS,
   AGX_APPLE9_ENC_TEXTURE_FETCH_PARAMS,
   AGX_APPLE9_ENC_TEXTURE_FETCH,
   AGX_APPLE9_ENC_TEXTURE_LOD_PARAMS,
   AGX_APPLE9_ENC_TEXTURE_VOLUME_PARAMS,
   AGX_APPLE9_ENC_TEXTURE_LOD,
   AGX_APPLE9_ENC_TEXTURE_GRAD_PARAMS,
   AGX_APPLE9_ENC_TEXTURE_GRAD,
   AGX_APPLE9_ENC_TEXTURE_SAMPLE,
   AGX_APPLE9_ENC_ITER,
   AGX_APPLE9_ENC_ITER_FLAT,
   AGX_APPLE9_ENC_VARY_STORE,
   AGX_APPLE9_ENC_COVERAGE,
   AGX_APPLE9_ENC_TILE_ACCESS,
   AGX_APPLE9_ENC_TILE_LOAD,
   AGX_APPLE9_ENC_TILE_STORE,
   AGX_APPLE9_ENC_TILE_LOAD_MASK,
   AGX_APPLE9_ENC_TILE_STORE_MASK,
   AGX_APPLE9_ENC_TILE_FENCE,
   AGX_APPLE9_ENC_DEVICE_LOAD,
   AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT,
   AGX_APPLE9_ENC_DEVICE_STORE,
   AGX_APPLE9_ENC_DEVICE_ATOMIC,
   AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT,
   AGX_APPLE9_ENC_SPILL_STORE,
   AGX_APPLE9_ENC_SPILL_LOAD,
   AGX_APPLE9_ENC_COUNT,

   /* Semantic IR pseudos have no machine encoding.  Keep them out of the
    * encoding table while making an accidental table lookup fail loudly. */
   AGX_APPLE9_ENC_PSEUDO = 0xff,
};

/*
 * Apple9 instructions encode pending-result dependencies in one of four
 * physical layouts. VIR names at most one logical slot for every consumer;
 * the selected machine encoding owns the translation into either a binary
 * slot index or a one-hot physical bit.
 */
enum agx_apple9_dependency_layout {
   AGX_APPLE9_DEPENDENCY_NONE = 0,
   AGX_APPLE9_DEPENDENCY_INDEX_45_47,
   AGX_APPLE9_DEPENDENCY_INDEX_61_63,
   AGX_APPLE9_DEPENDENCY_MASK_12_17,
   AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
};

#define AGX_APPLE9_MAX_ENCODING_OPERANDS 7

struct agx_apple9_encoding_info {
   const char *name;
   uint8_t length;
   uint8_t operand_count;
   bool allocator_safe;
   enum agx_apple9_evidence evidence;
   enum agx_apple9_dependency_layout dependency_layout;

   /* Zero means no cross-operand restriction. */
   uint8_t max_high_gpr_operands;
   struct agx_apple9_operand_constraint
      operands[AGX_APPLE9_MAX_ENCODING_OPERANDS];
};

enum agx_apple9_occupancy_model {
   AGX_APPLE9_OCCUPANCY_PRESSURE_TIER,
};

struct agx_apple9_machine {
   uint16_t gpr_count;
   uint16_t half_register_count;

   bool hardware_register_interlocks;
   bool software_waits;
   bool spilling_supported;
   enum agx_apple9_occupancy_model occupancy_model;
};

extern const struct agx_apple9_machine agx_apple9_machine;

const struct agx_apple9_encoding_info *
agx_apple9_encoding_info(enum agx_apple9_encoding encoding);

const struct agx_apple9_operand_constraint *
agx_apple9_find_operand(enum agx_apple9_encoding encoding,
                        enum agx_apple9_operand_role role);

bool agx_apple9_encoding_accepts_gpr(enum agx_apple9_encoding encoding,
                                     enum agx_apple9_operand_role role,
                                     unsigned gpr, unsigned bits);

/* Validate all GPR operands in descriptor order, including coupled limits. */
bool agx_apple9_encoding_accepts_gpr_tuple(enum agx_apple9_encoding encoding,
                                           const unsigned *gprs,
                                           unsigned operand_count,
                                           unsigned bits);

#ifdef __cplusplus
}
#endif

#endif
