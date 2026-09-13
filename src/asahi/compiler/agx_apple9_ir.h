/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX_APPLE9_IR_H
#define AGX_APPLE9_IR_H

#include <stdbool.h>
#include <stdint.h>
#include "util/bitset.h"
#include "util/list.h"

#include "agx_apple9_machine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_APPLE9_VREG_INVALID UINT32_MAX
#define AGX_APPLE9_PHYS_INVALID UINT8_MAX
#define AGX_APPLE9_MAX_VIR_SRCS 7
/*
 * A deliberately small semantic IR for the first real Apple9 compiler.
 *
 * Values are scalar 32-bit SSA virtual registers.  Consecutive values may be
 * grouped into an adjacent physical-register tuple by an explicit COLLECT
 * pseudo.  Encoding selection is explicit before allocation, so physical-
 * register constraints come from the Apple9 machine table instead of being
 * implicit in byte templates.
 */
enum agx_apple9_vir_opcode {
   AGX_APPLE9_VIR_IMM,
   AGX_APPLE9_VIR_GET_GLOBAL_ID,
   AGX_APPLE9_VIR_GET_SR,
   AGX_APPLE9_VIR_DEVICE_LOAD,
   AGX_APPLE9_VIR_PUBLICATION_TUPLE,
   AGX_APPLE9_VIR_TEXTURE_SAMPLE,
   AGX_APPLE9_VIR_BLOCK_IMAGE_STORE,
   AGX_APPLE9_VIR_U2F32,
   AGX_APPLE9_VIR_I2F32,
   AGX_APPLE9_VIR_F2I32,
   AGX_APPLE9_VIR_F2U32,
   AGX_APPLE9_VIR_IADD,
   AGX_APPLE9_VIR_IMUL,
   AGX_APPLE9_VIR_ISUB,
   AGX_APPLE9_VIR_IMAD,
   AGX_APPLE9_VIR_IAND,
   AGX_APPLE9_VIR_IOR,
   AGX_APPLE9_VIR_IXOR,
   AGX_APPLE9_VIR_ISHR,
   AGX_APPLE9_VIR_IMIN,
   AGX_APPLE9_VIR_IMAX,
   AGX_APPLE9_VIR_UMIN,
   AGX_APPLE9_VIR_UMAX,
   AGX_APPLE9_VIR_FADD,
   AGX_APPLE9_VIR_FSUB,
   AGX_APPLE9_VIR_FMUL,
   AGX_APPLE9_VIR_FMUL_PROJECT,
   AGX_APPLE9_VIR_FADD_IMM,
   AGX_APPLE9_VIR_FMUL_IMM,
   AGX_APPLE9_VIR_FMIN,
   AGX_APPLE9_VIR_FMAX,
   AGX_APPLE9_VIR_FMA,
   AGX_APPLE9_VIR_CUBE,
   AGX_APPLE9_VIR_FRCP,
   AGX_APPLE9_VIR_FRSQ,
   /* rsqrt-like factor with 1 at signed zero and +infinity. */
   AGX_APPLE9_VIR_FSQRT_FACTOR,
   /* sin(pi*x/2)/x on [-1,1], with limit pi/2 at zero. */
   AGX_APPLE9_VIR_FSIN_FACTOR,
   AGX_APPLE9_VIR_FEXP2,
   AGX_APPLE9_VIR_FLOG2,
   AGX_APPLE9_VIR_FFLOOR,
   AGX_APPLE9_VIR_FCEIL,
   AGX_APPLE9_VIR_FTRUNC,
   AGX_APPLE9_VIR_FROUND_EVEN,
   AGX_APPLE9_VIR_DERIVATIVE,
   AGX_APPLE9_VIR_SELECT,
   AGX_APPLE9_VIR_COLLECT,
   AGX_APPLE9_VIR_PHI,
   AGX_APPLE9_VIR_PHI_SRC,
   AGX_APPLE9_VIR_PREDICATE_COMPARE,
   AGX_APPLE9_VIR_EXEC_MASK_PUSH,
   AGX_APPLE9_VIR_EXEC_MASK_ELSE,
   AGX_APPLE9_VIR_EXEC_MASK_POP,
   AGX_APPLE9_VIR_LOOP_MASK_PUSH,
   AGX_APPLE9_VIR_LOOP_MASK_UPDATE,
   AGX_APPLE9_VIR_LOOP_MASK_POP,
   AGX_APPLE9_VIR_JMP_EXEC_ANY,
   AGX_APPLE9_VIR_JMP_EXEC_NONE,
   AGX_APPLE9_VIR_BREAK_MASK_UNWIND,
   AGX_APPLE9_VIR_CENTROID_POSITION,
   AGX_APPLE9_VIR_ITER,
   AGX_APPLE9_VIR_ITER_FLAT,
   AGX_APPLE9_VIR_VARY_STORE,
   AGX_APPLE9_VIR_COVERAGE,
   AGX_APPLE9_VIR_DEPTH_STORE,
   AGX_APPLE9_VIR_TILE_ACCESS,
   AGX_APPLE9_VIR_TILE_LOAD,
   AGX_APPLE9_VIR_TILE_STORE,
   AGX_APPLE9_VIR_TILE_FENCE,
   AGX_APPLE9_VIR_DEVICE_STORE,
   AGX_APPLE9_VIR_DEVICE_ATOMIC,
   AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT,
   AGX_APPLE9_VIR_SPILL_STORE,
   AGX_APPLE9_VIR_SPILL_LOAD,
};

/* Shared native five-bit operation selector used by the device and
 * threadgroup atomic families.  The current compiler emits the independently
 * hardware-validated per-lane device form. */
enum agx_apple9_atomic_op {
   AGX_APPLE9_ATOMIC_ADD = 0x10,
   AGX_APPLE9_ATOMIC_AND = 0x11,
   AGX_APPLE9_ATOMIC_CMPXCHG = 0x12,
   AGX_APPLE9_ATOMIC_FADD = 0x13,
   AGX_APPLE9_ATOMIC_SMAX = 0x14,
   AGX_APPLE9_ATOMIC_SMIN = 0x15,
   AGX_APPLE9_ATOMIC_OR = 0x16,
   AGX_APPLE9_ATOMIC_SUB = 0x1b,
   AGX_APPLE9_ATOMIC_UMAX = 0x1c,
   AGX_APPLE9_ATOMIC_UMIN = 0x1d,
   AGX_APPLE9_ATOMIC_XCHG = 0x1e,
   AGX_APPLE9_ATOMIC_XOR = 0x1f,
};

enum agx_apple9_predicate_condition {
   AGX_APPLE9_PREDICATE_FGT = 0x02,
   AGX_APPLE9_PREDICATE_FLT = 0x03,
   AGX_APPLE9_PREDICATE_UGT = 0x04,
   AGX_APPLE9_PREDICATE_ULT = 0x05,
   AGX_APPLE9_PREDICATE_IGT = 0x06,
   AGX_APPLE9_PREDICATE_ILT = 0x07,
};

enum agx_apple9_predicate_extended_condition {
   AGX_APPLE9_PREDICATE_EXT_FEQ = 0x00,
   AGX_APPLE9_PREDICATE_EXT_FGE_SEQUENCE = 0x02,
   AGX_APPLE9_PREDICATE_EXT_FLE_SEQUENCE = 0x03,
   AGX_APPLE9_PREDICATE_EXT_IEQ = 0x07,
};

/* EXP-M4-53 separates the six-entry predicate scratch bank from the implicit
 * execution-mask stack.  PREDICATE_INVERT complements a comparison result;
 * EXEC_MASK_INVERT complements the selected predicate when a push consumes
 * it.  Ordinary nested ifs may therefore reuse bank zero at every depth. */
#define AGX_APPLE9_PREDICATE_BANK_COUNT 6u
#define AGX_APPLE9_PREDICATE_INVERT     (1u << 8)
#define AGX_APPLE9_PREDICATE_BANK_SHIFT 16
#define AGX_APPLE9_PREDICATE_BANK_MASK                                        \
   (0x07u << AGX_APPLE9_PREDICATE_BANK_SHIFT)
#define AGX_APPLE9_PREDICATE_BANK(index)                                      \
   ((uint32_t)(index) << AGX_APPLE9_PREDICATE_BANK_SHIFT)

#define AGX_APPLE9_EXEC_MASK_INVERT (1u << 8)
#define AGX_APPLE9_EXEC_MASK_SOURCE(source) (0x01u + 4u * (source))
#define AGX_APPLE9_EXEC_MASK_PREDICATE(bank)                                  \
   AGX_APPLE9_EXEC_MASK_SOURCE(bank)
#define AGX_APPLE9_EXEC_MASK_TRUE  AGX_APPLE9_EXEC_MASK_SOURCE(6)
#define AGX_APPLE9_EXEC_MASK_FALSE AGX_APPLE9_EXEC_MASK_SOURCE(7)

/* The tested LOOP_MASK_UPDATE form selects the same predicate bank with
 * 0x02,0x06,0x0a,... .  Bit 0x20 is consumer polarity: EXP-M4-53 proves that
 * flipping it together with PREDICATE_INVERT preserves exact loop output. */
#define AGX_APPLE9_LOOP_MASK_INVERT 0x20u
#define AGX_APPLE9_LOOP_MASK_PREDICATE(bank) (0x02u + 4u * (bank))

/* BREAK_MASK_UNWIND packs the number of scopes crossed and the destination
 * loop nesting level into independent bytes. */
#define AGX_APPLE9_BREAK_SCOPE_TAG(immediate)  (((immediate) >> 8) & 0xffu)
#define AGX_APPLE9_BREAK_LOOP_DEPTH(immediate) ((immediate) & 0xffu)
#define AGX_APPLE9_BREAK_IMMEDIATE(scope_tag, loop_depth)                      \
   (((uint32_t)(scope_tag) << 8) | (uint32_t)(loop_depth))

enum agx_apple9_select_condition {
   AGX_APPLE9_SELECT_FEQ = 0x00,
   AGX_APPLE9_SELECT_FGT = 0x02,
   AGX_APPLE9_SELECT_FLT = 0x03,
   AGX_APPLE9_SELECT_UGT = 0x04,
   AGX_APPLE9_SELECT_ULT = 0x05,
   AGX_APPLE9_SELECT_IGT = 0x06,
   AGX_APPLE9_SELECT_ILT = 0x07,
};

#define AGX_APPLE9_SELECT_EQUALITY (1u << 8)

/*
 * Apple9 asynchronous producers publish through a six-entry scoreboard.
 * Slot 0 is the ordinary/materialized GPR path; slots 1--6 are transient
 * completion groups for outstanding asynchronous work.  AUTO is compiler IR only and must be resolved
 * after final instruction scheduling, before packing.
 */
enum agx_apple9_scoreboard_slot {
   AGX_APPLE9_SCOREBOARD_SLOT_NONE = 0,
   AGX_APPLE9_SCOREBOARD_SLOT_1 = 1,
   AGX_APPLE9_SCOREBOARD_SLOT_2 = 2,
   AGX_APPLE9_SCOREBOARD_SLOT_3 = 3,
   AGX_APPLE9_SCOREBOARD_SLOT_4 = 4,
   AGX_APPLE9_SCOREBOARD_SLOT_5 = 5,
   AGX_APPLE9_SCOREBOARD_SLOT_6 = 6,
   AGX_APPLE9_SCOREBOARD_SLOT_AUTO = 0xff,
};

enum agx_apple9_device_load_flags {
   /* Byte 2 bit 4 marks that another load follows in linear issue order.
    * Native Metal leaves it set across execution-mask transitions. */
   AGX_APPLE9_DEVICE_LOAD_HAS_NEXT = 1u << 1,
};

enum agx_apple9_device_load_index_kind {
   /* Byte 5's low seven bits name a real architectural GPR.  Bit 7 is clear
    * when that SSA value has a later consumer and set when this load is its
    * final consumer.  VIR packing derives the bit from allocator liveness;
    * the raw packer accepts the explicit form for byte-level tests. */
   AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
   AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,

   /* Compatibility spellings retained while the existing capture tiers are
    * migrated.  They describe lifetime, not how the index was computed. */
   AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR =
      AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
   AGX_APPLE9_DEVICE_LOAD_INDEX_COMPUTED_GPR =
      AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR,
};

/* Human-order names for the two raw bytes at instruction offsets 8 and 9.
 * For example, 0x5101 is serialized as 51:01, independent of host endian. */
enum agx_apple9_device_load_raw_token {
   AGX_APPLE9_DEVICE_LOAD_TOKEN_1100 = 0x1100,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_5100 = 0x5100,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_9100 = 0x9100,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_D100 = 0xd100,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_1101 = 0x1101,
   AGX_APPLE9_DEVICE_LOAD_TOKEN_5101 = 0x5101,
};

struct agx_apple9_device_load_contract {
   enum agx_apple9_device_load_index_kind index_kind;
   uint8_t flags;
   uint16_t raw_token;
};

struct agx_apple9_block;

struct agx_apple9_vir_instr {
   enum agx_apple9_vir_opcode op;
   enum agx_apple9_encoding encoding;
   uint32_t dest;
   /* Number of adjacent SSA values and physical GPRs defined at dest.
    * Scalar instructions define one value. Native vector memory loads and
    * COLLECT define one two-, three-, or four-register tuple. A native load
    * uses one scoreboard allocation for the complete tuple. */
   uint8_t dest_components;
   /* Scalar device-memory element width.  Zero is the established u32
    * default; compiler-authored narrow operations use 8 or 16 explicitly. */
   uint8_t memory_bits;
   /* Number of scalar lanes read or written by a memory instruction.  Loads
    * also define an adjacent dest_components tuple; stores have no SSA
    * destination and carry their ordered data values in src[0..N-1]. */
   uint8_t memory_components;
   uint8_t tile_sample_mask;
   enum agx_apple9_atomic_op atomic_op;
   bool atomic_discard;
   uint8_t texture_index, sampler_index;
   uint32_t src[AGX_APPLE9_MAX_VIR_SRCS];
   uint8_t texture_dimension;
   bool texture_shadow;
   bool texture_fetch;
   /* An incoming phi use names its successor's SSA definition. This is
    * edge metadata, not a mutable register assignment before allocation. */
   uint32_t target;
   uint32_t immediate;
   /* Stable destination identity; byte displacement is computed at packing. */
   struct agx_apple9_block *branch_target;
   /* Phi declaration and incoming edge uses remain SSA until packing. */
   struct agx_apple9_block *phi_block;
   struct agx_apple9_vir_instr *phi_edge;
   uint8_t nr_srcs;

   /* Scoreboard slot published by an asynchronous producer, including scalar
    * ALU exports to publication storage. Ordinary GPR ALU instructions leave
    * this at NONE. AUTO is resolved by the scheduled
    * scoreboard pass and is never serialized. */
   uint8_t producer_scoreboard_slot;
   bool producer_scoreboard_assigned, scoreboard_assigned;

   /* A scratch store has no register result. Its completion is consumed by
    * a scheduled instruction in the same block, independently of SSA uses. */
   struct agx_apple9_vir_instr *completion_consumer;

   /* Device-load sequence and index lifetime, independent of completion
    * tags. */
   uint8_t device_load_flags;
   enum agx_apple9_device_load_index_kind device_load_index_kind;

   /* Raw scalar-load instruction bytes 8:9.  These encode the producer's
    * scoreboard slot. */
   uint16_t device_load_raw_token;

   /* Bit s is set when src[s] has another consumer after this instruction. */
   uint8_t live_after_mask;

   /*
    * The one pending-result slot consumed by this instruction, or NONE.
    * Metal has not been observed making one instruction consume multiple
    * slots. The selected encoding translates this logical slot into either
    * a binary index or a one-hot physical field. This is independent of
    * per-source release bits.
    */
   uint8_t scoreboard_slot;

   /* Conservative identity bridge inserted when an asynchronous load's first
    * user has no proven slot-bearing form.  It consumes the assigned slot
    * immediately and leaves an ordinary GPR value for the original users. */
   bool scoreboard_materialize;
   /* Preserve a result handoff that releases scarce texture publications. */
   bool publication_handoff;

   /*
    * Fragment iterator/perspective producers are addressed by compact ALU
    * through operand tokens, not by their architectural GPR numbers.  A value
    * of 0xff means that this source uses the ordinary GPR encoding.  These
    * per-source tokens are distinct from the instruction-wide scoreboard
    * slot and are kept on the use until the graphics compiler is generalized.
    */
};

/* Instruction storage and block identity survive every insertion/removal.
 * Layout links are independent of ownership, allowing forward branch targets. */
struct agx_apple9_block {
   struct agx_apple9_block *next, *prev, *allocated_next;
   struct agx_apple9_vir_program *program;
   struct nir_block *nir; /* Logical CFG, distinct from execution-mask layout. */
   struct agx_apple9_block **predecessors;
   unsigned predecessor_count;
   struct list_head instructions;
   bool placed;
   unsigned index;
   struct agx_apple9_block *successors[2];
   BITSET_WORD *live_in, *live_out;
   unsigned start_index; /* Derived analysis position, never a branch identity. */
   unsigned offset;
};

/* Uses are ordered by machine layout. Rebuild after operand edits; pointers
 * remain valid only until the next analysis rebuild. */
struct agx_apple9_use {
   struct agx_apple9_vir_instr *instruction;
   unsigned source;
   struct agx_apple9_use *next;
};

struct agx_apple9_vir_program {
   /* Indexed analysis view of stable, block-owned instructions. */
   struct agx_apple9_vir_instr **instructions;
   struct agx_apple9_block *blocks, *last_block, *current_block;
   struct agx_apple9_block *allocated_blocks;
   struct agx_apple9_use_analysis *use_analysis;
   bool dependencies_finalized;
   unsigned instruction_count;
   unsigned instruction_capacity;
   unsigned value_count;
   uint32_t output;
   /* Fragment execution can update depth/stencil without explicit stores. */
   bool fragment_shader;
   bool physical;

   /*
    * Values may enter or leave the bounded program in fixed physical GPRs.
    * This is used by fragment interpolation/color packing, whose surrounding
    * stage instructions have an independently validated register contract.
    */
   uint8_t *fixed_phys;
   /* Optional per-value upper bound used to propagate a constrained
    * consumer's compact-register requirement back to its producer. */
   uint8_t *max_phys;
   uint32_t *live_out;
   unsigned live_out_count;
   unsigned live_out_capacity;
   bool reserved_gprs[AGX_APPLE9_GPR_COUNT];

   /* Filled by agx_apple9_allocate_vir(). */
   uint8_t *phys;
   /* Export publications have their own index namespace. Slots remain distinct
    * through completion; ordinary GPRs with the same index are independent. */
   bool *publication;
   unsigned publication_count;
   unsigned peak_live_gprs;
   unsigned max_phys_gpr;
   unsigned scratch_size; /* Bytes per invocation, rounded to 16. */
   unsigned spill_slots;
};

void agx_apple9_invalidate_uses(struct agx_apple9_vir_program *program);
bool agx_apple9_analyze_uses(struct agx_apple9_vir_program *program);
const struct agx_apple9_vir_instr *agx_apple9_definition(
   const struct agx_apple9_vir_program *program, uint32_t value);
const struct agx_apple9_use *agx_apple9_uses(
   const struct agx_apple9_vir_program *program, uint32_t value);

void agx_apple9_vir_init(struct agx_apple9_vir_program *program);
void agx_apple9_vir_finish(struct agx_apple9_vir_program *program);

struct agx_apple9_block *agx_apple9_block_create(struct agx_apple9_vir_program *program);
void agx_apple9_block_begin(struct agx_apple9_vir_program *program,
                          struct agx_apple9_block *block);
struct agx_apple9_block *agx_apple9_instr_block(const struct agx_apple9_vir_instr *instr);
void agx_apple9_vir_move_before(struct agx_apple9_vir_program *program,
                              struct agx_apple9_vir_instr *instr,
                              struct agx_apple9_vir_instr *before);
void agx_apple9_vir_reindex(struct agx_apple9_vir_program *program);

uint32_t agx_apple9_vir_emit(struct agx_apple9_vir_program *program,
                             enum agx_apple9_vir_opcode op,
                             enum agx_apple9_encoding encoding,
                             const uint32_t *src, unsigned nr_srcs,
                             uint32_t immediate);

bool agx_apple9_vir_emit_side_effect(struct agx_apple9_vir_program *program,
                                     enum agx_apple9_vir_opcode op,
                                     enum agx_apple9_encoding encoding,
                                     const uint32_t *src, unsigned nr_srcs,
                                     uint32_t immediate);

bool agx_apple9_vir_emit_branch(struct agx_apple9_vir_program *program,
                                enum agx_apple9_vir_opcode op,
                                enum agx_apple9_encoding encoding,
                                struct agx_apple9_block *target);

uint32_t agx_apple9_vir_input(struct agx_apple9_vir_program *program,
                              unsigned phys);
bool agx_apple9_vir_set_load_address(struct agx_apple9_vir_program *program,
                                     uint32_t value, uint32_t address);
uint32_t agx_apple9_vir_emit_iter_flat(struct agx_apple9_vir_program *program,
                                       unsigned coefficient);

uint32_t agx_apple9_vir_emit_device_load(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   const struct agx_apple9_device_load_contract *contract);
/* mode 0 returns magnitude and a face in the low 16 bits of the next GPR.
 * Modes 1/2 return the signed half-coordinate for U/V, respectively. */
uint32_t agx_apple9_vir_emit_cube(struct agx_apple9_vir_program *program,
                                 const uint32_t src[3], unsigned mode);

uint32_t agx_apple9_vir_emit_device_load_vector(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   unsigned components, const struct agx_apple9_device_load_contract *contract);
/* Publish an allocated FP32 coordinate pair and sample one bound 2D texture.
 * The current graphics package supplies eight coordinate publications. */
uint32_t
agx_apple9_vir_emit_publication_pair(struct agx_apple9_vir_program *program,
                                     const uint32_t src[2]);

bool agx_apple9_vir_emit_block_image_store(
   struct agx_apple9_vir_program *program, const uint32_t src[3],
   unsigned image, unsigned format, bool multisampled);

uint32_t agx_apple9_vir_emit_texture_sample(
   struct agx_apple9_vir_program *program, const uint32_t coords[2],
   uint32_t one, unsigned texture, unsigned sampler);

/* Form an adjacent register tuple from independent scalar SSA values before
 * register allocation. The pseudo is coalesced when possible and otherwise
 * lowered to copies after allocation, following the Apple8 AGX IR model. */
/* FP32 coordinates and signed Q6 LOD packed in bits16..27. */
uint32_t agx_apple9_vir_emit_texture_lod(
   struct agx_apple9_vir_program *program, const uint32_t coords[2],
   uint32_t packed_lod, unsigned texture, unsigned sampler, bool bias);

uint32_t agx_apple9_vir_emit_texture_volume(
   struct agx_apple9_vir_program *program, const uint32_t coords[3],
   uint32_t packed_lod, unsigned texture, unsigned sampler, unsigned dimension,
   bool bias, bool shadow);

/* Coordinates followed by ddx.xy and ddy.xy, all FP32. */
uint32_t agx_apple9_vir_emit_texture_grad(
   struct agx_apple9_vir_program *program, const uint32_t src[6],
   unsigned texture, unsigned sampler);

uint32_t agx_apple9_vir_emit_collect(struct agx_apple9_vir_program *program,
                                     const uint32_t *src, unsigned components);
void agx_apple9_place_phis(struct agx_apple9_vir_program *program);
/* Reuse the Apple8 physical parallel-copy scheduler. Indices are 32-bit GPRs. */
bool agx_apple9_resolve_phi_edge(const struct agx_apple9_vir_program *program,
   const struct agx_apple9_vir_instr *edge,
   bool (*emit)(void *data, bool swap, unsigned dest, unsigned source), void *data);

uint32_t agx_apple9_vir_emit_phi(struct agx_apple9_vir_program *program,
                                  struct agx_apple9_block *block);
bool agx_apple9_vir_emit_phi_source(struct agx_apple9_vir_program *program,
                                     uint32_t target, uint32_t source);
struct agx_apple9_vir_copy {
   uint32_t target, source;
};
bool agx_apple9_vir_emit_phi_edge(
   struct agx_apple9_vir_program *program,
   const struct agx_apple9_vir_copy *copies, unsigned count);

bool
agx_apple9_vir_set_device_store_address(struct agx_apple9_vir_program *program,
                                        uint32_t address);
bool agx_apple9_vir_emit_device_store(struct agx_apple9_vir_program *program,
                                      unsigned binding, uint32_t index,
                                      const uint32_t *data, unsigned components,
                                      unsigned bits);
bool agx_apple9_vir_emit_device_atomic(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   const uint32_t *data, unsigned data_components,
   enum agx_apple9_atomic_op op, bool discard_result, uint32_t *result_out);
bool agx_apple9_vir_set_device_load_contract(
   struct agx_apple9_vir_program *program, uint32_t value, uint8_t flags,
   enum agx_apple9_scoreboard_slot scoreboard_slot);
bool agx_apple9_vir_set_device_load_raw_contract(
   struct agx_apple9_vir_program *program, uint32_t value, uint8_t flags,
   uint16_t raw_token);
bool agx_apple9_vir_set_device_load_index_kind(
   struct agx_apple9_vir_program *program, uint32_t value,
   enum agx_apple9_device_load_index_kind index_kind);
bool agx_apple9_vir_set_fixed_phys(struct agx_apple9_vir_program *program,
                                   uint32_t value, unsigned phys);
bool agx_apple9_vir_add_live_out(struct agx_apple9_vir_program *program,
                                 uint32_t value);
/*
 * CFG liveness with conservative enclosing-interval allocation for scalar
 * and adjacent-tuple values. General encodings prefer
 * r16-r63 so the r0-r15 compact-result bank stays available to hard-low
 * instructions, then fall back to that low bank. Destinations remain distinct
 * from their inputs; killed sources become available to following
 * instructions. Spills use per-thread Dynamic-Caching scratch storage.
 */
struct nir_shader;
bool agx_apple9_allocate_shared(struct agx_apple9_vir_program *program,
                                 struct nir_shader *nir, const char **reason);
bool agx_apple9_allocate_publications(struct agx_apple9_vir_program *program,
                                      const char **reason);

bool agx_apple9_allocate_vir(struct agx_apple9_vir_program *program,
                             const char **reason);

bool
agx_apple9_validate_vir_allocation(const struct agx_apple9_vir_program *program,
                                   const char **reason);

/* Allocate pending asynchronous producers to scoreboard slots from final VIR
 * order.  The first capable consumer performs the handoff; later reads use
 * slot 0. */
bool
agx_apple9_assign_vir_scoreboard_slots(struct agx_apple9_vir_program *program,
                                       const char **reason);

/* One physical Apple9 instruction, used by the compiler and packer tests. */
struct agx_apple9_packed_instruction {
   /* Includes bounded multi-instruction parameter-publication pseudos. */
   uint8_t bytes[64];
   uint8_t length;
};

/* Start-relative byte displacement, encoded in the branch's signed 48-bit
 * field. Unlike local CFG offsets, entry transfers may cross the signed-32-bit
 * range. */
bool agx_apple9_pack_branch(bool any, int64_t displacement,
                            struct agx_apple9_packed_instruction *packed);

bool agx_apple9_pack_vir_instruction(
   const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
   struct agx_apple9_packed_instruction *packed, const char **reason);

bool
agx_apple9_pack_get_global_id(unsigned dst, unsigned component,
                              struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_get_sr(unsigned dst, uint8_t selector, uint8_t datapath,
                            struct agx_apple9_packed_instruction *packed);
bool
agx_apple9_pack_get_sr_zext16(unsigned dst, uint8_t selector,
                              struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_mov_imm(unsigned dst, unsigned value,
                             struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_mov_imm32(unsigned dst, uint32_t value,
                               struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_mov(unsigned dst, unsigned src,
                         struct agx_apple9_packed_instruction *packed);

bool
agx_apple9_pack_device_load_u32(unsigned dst, unsigned index, unsigned binding,
                                uint8_t flags,
                                enum agx_apple9_scoreboard_slot scoreboard_slot,
                                struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_load_u32_raw(
   unsigned dst, unsigned index, unsigned binding,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed);

/* Aligned native vector memory forms.  A load defines components consecutive
 * 32-bit GPRs beginning at dst and publishes the complete tuple through one
 * scoreboard slot.  A store consumes components consecutive GPRs beginning
 * at data and allocates no scoreboard slot of its own.  Device-store
 * access_desc bit 0 is the index last-use control: clear retains the index
 * GPR and set releases it after the address read. */
bool agx_apple9_pack_device_load_vector_u32_raw(
   unsigned dst, unsigned index, unsigned binding, unsigned components,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_load_scalar_raw(
   unsigned dst, unsigned index, unsigned binding, unsigned bits,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_store_scalar(
   unsigned data, unsigned index, unsigned binding, unsigned bits,
   enum agx_apple9_scoreboard_slot scoreboard_slot, bool release_index,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_store_vector_u32(
   unsigned data, unsigned index, unsigned binding, unsigned components,
   enum agx_apple9_scoreboard_slot scoreboard_slot, bool release_index,
   struct agx_apple9_packed_instruction *packed);
bool agx_apple9_pack_device_atomic(
   unsigned index, unsigned data, unsigned binding, enum agx_apple9_atomic_op op,
   bool discard_result,
   enum agx_apple9_scoreboard_slot input_dependency,
   struct agx_apple9_packed_instruction *packed);

#ifdef __cplusplus
}
#endif

#endif
