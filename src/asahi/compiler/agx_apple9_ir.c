/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_ir.h"
#include "util/u_dynarray.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define APPLE9_FIRST_ALLOCATABLE_GPR 0
#define APPLE9_FIRST_GENERAL_GPR     16
#define APPLE9_LAST_ALLOCATABLE_GPR  63

static void
set_bits(uint8_t *encoded, unsigned start, unsigned width, uint64_t value)
{
   assert(width <= 64);
   assert(width == 64 || value < (1ull << width));

   uint64_t mask = width == 64 ? UINT64_MAX : ((1ull << width) - 1);
   for (unsigned bit = 0; bit < width; ++bit) {
      unsigned byte = (start + bit) / 8;
      unsigned shift = (start + bit) % 8;
      encoded[byte] &= ~(1u << shift);
      encoded[byte] |= ((value & mask) >> bit & 1u) << shift;
   }
}

static bool
apple9_dependency_slot_valid(enum agx_apple9_dependency_layout layout,
                             uint8_t slot)
{
   if (slot > AGX_APPLE9_SCOREBOARD_SLOT_6)
      return false;

   switch (layout) {
   case AGX_APPLE9_DEPENDENCY_NONE:
      return slot == AGX_APPLE9_SCOREBOARD_SLOT_NONE;
   case AGX_APPLE9_DEPENDENCY_INDEX_45_47:
   case AGX_APPLE9_DEPENDENCY_INDEX_61_63:
   case AGX_APPLE9_DEPENDENCY_MASK_12_17:
   case AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63:
      return true;
   }

   return false;
}

static bool
apple9_pack_dependency(uint8_t *bytes, unsigned length,
                       enum agx_apple9_dependency_layout layout, uint8_t slot)
{
   if (!apple9_dependency_slot_valid(layout, slot))
      return false;

   switch (layout) {
   case AGX_APPLE9_DEPENDENCY_NONE:
      return true;
   case AGX_APPLE9_DEPENDENCY_INDEX_45_47:
      if (length * 8 < 48)
         return false;
      set_bits(bytes, 45, 3, slot);
      return true;
   case AGX_APPLE9_DEPENDENCY_INDEX_61_63:
      if (length * 8 < 64)
         return false;
      set_bits(bytes, 61, 3, slot);
      return true;
   case AGX_APPLE9_DEPENDENCY_MASK_12_17:
      if (length * 8 < 18)
         return false;
      set_bits(bytes, 12, 6,
               slot == AGX_APPLE9_SCOREBOARD_SLOT_NONE ? 0 : 1u << (slot - 1));
      return true;
   case AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63:
      if (length * 8 < 64)
         return false;
      set_bits(bytes, 45, 3,
               slot >= AGX_APPLE9_SCOREBOARD_SLOT_1 &&
                     slot <= AGX_APPLE9_SCOREBOARD_SLOT_3
                  ? 1u << (slot - 1)
                  : 0);
      set_bits(bytes, 61, 3,
               slot >= AGX_APPLE9_SCOREBOARD_SLOT_4
                  ? 1u << (slot - AGX_APPLE9_SCOREBOARD_SLOT_4)
                  : 0);
      return true;
   }

   return false;
}

void
agx_apple9_vir_init(struct agx_apple9_vir_program *program)
{
   memset(program, 0, sizeof(*program));
   program->output = AGX_APPLE9_VREG_INVALID;
}

void
agx_apple9_vir_finish(struct agx_apple9_vir_program *program)
{
   free(program->instructions);
   free(program->phys);
   free(program->fixed_phys);
   free(program->max_phys);
   free(program->live_out);
   agx_apple9_vir_init(program);
}

static uint32_t
apple9_vir_new_value(struct agx_apple9_vir_program *program,
                     unsigned fixed_phys)
{
   const unsigned new_count = program->value_count + 1;
   uint8_t *fixed = realloc(program->fixed_phys, new_count);
   if (fixed == NULL)
      return AGX_APPLE9_VREG_INVALID;
   program->fixed_phys = fixed;
   uint8_t *maximum = realloc(program->max_phys, new_count);
   if (maximum == NULL)
      return AGX_APPLE9_VREG_INVALID;
   program->max_phys = maximum;
   uint32_t value = program->value_count++;
   program->fixed_phys[value] = fixed_phys;
   program->max_phys[value] = AGX_APPLE9_PHYS_INVALID;
   return value;
}

static bool
apple9_vir_append_values(struct agx_apple9_vir_program *program, unsigned count)
{
   if (count == 0)
      return true;

   if (program->value_count > UINT32_MAX - count)
      return false;

   const unsigned old_count = program->value_count;
   const unsigned new_count = program->value_count + count;
   uint8_t *fixed = realloc(program->fixed_phys, new_count);
   if (fixed == NULL)
      return false;
   program->fixed_phys = fixed;
   uint8_t *maximum = realloc(program->max_phys, new_count);
   if (maximum == NULL)
      return false;
   program->max_phys = maximum;
   memset(&program->fixed_phys[old_count], AGX_APPLE9_PHYS_INVALID, count);
   memset(&program->max_phys[old_count], AGX_APPLE9_PHYS_INVALID, count);
   program->value_count += count;
   return true;
}

static struct agx_apple9_vir_instr *
apple9_vir_append_instruction(struct agx_apple9_vir_program *program)
{
   if (program->instruction_count == program->instruction_capacity) {
      unsigned capacity =
         program->instruction_capacity ? program->instruction_capacity * 2 : 16;
      void *resized = realloc(program->instructions,
                              capacity * sizeof(*program->instructions));
      if (resized == NULL)
         return NULL;
      program->instructions = resized;
      program->instruction_capacity = capacity;
   }

   return &program->instructions[program->instruction_count++];
}

uint32_t
agx_apple9_vir_emit(struct agx_apple9_vir_program *program,
                    enum agx_apple9_vir_opcode op,
                    enum agx_apple9_encoding encoding, const uint32_t *src,
                    unsigned nr_srcs, uint32_t immediate)
{
   assert(nr_srcs <= AGX_APPLE9_MAX_VIR_SRCS);
   uint32_t dest = apple9_vir_new_value(program, AGX_APPLE9_PHYS_INVALID);
   if (dest == AGX_APPLE9_VREG_INVALID)
      return dest;
   struct agx_apple9_vir_instr *instruction =
      apple9_vir_append_instruction(program);
   if (instruction == NULL) {
      --program->value_count;
      return AGX_APPLE9_VREG_INVALID;
   }
   *instruction = (struct agx_apple9_vir_instr){
      .op = op,
      .encoding = encoding,
      .dest = dest,
      .dest_components = 1,
      .target = AGX_APPLE9_VREG_INVALID,
      .immediate = immediate,
      .branch_target = AGX_APPLE9_VREG_INVALID,
      .nr_srcs = nr_srcs,
   };
   for (unsigned i = 0; i < nr_srcs; ++i)
      instruction->src[i] = src[i];
   return dest;
}

bool
agx_apple9_vir_emit_side_effect(struct agx_apple9_vir_program *program,
                                enum agx_apple9_vir_opcode op,
                                enum agx_apple9_encoding encoding,
                                const uint32_t *src, unsigned nr_srcs,
                                uint32_t immediate)
{
   if (program == NULL || nr_srcs > AGX_APPLE9_MAX_VIR_SRCS)
      return false;

   struct agx_apple9_vir_instr *instruction =
      apple9_vir_append_instruction(program);
   if (instruction == NULL)
      return false;

   *instruction = (struct agx_apple9_vir_instr){
      .op = op,
      .encoding = encoding,
      .dest = AGX_APPLE9_VREG_INVALID,
      .target = AGX_APPLE9_VREG_INVALID,
      .immediate = immediate,
      .branch_target = AGX_APPLE9_VREG_INVALID,
      .nr_srcs = nr_srcs,
   };
   for (unsigned i = 0; i < nr_srcs; ++i)
      instruction->src[i] = src[i];

   return true;
}

bool
agx_apple9_vir_emit_branch(struct agx_apple9_vir_program *program,
                           enum agx_apple9_vir_opcode op,
                           enum agx_apple9_encoding encoding,
                           uint32_t target_instruction)
{
   if ((op != AGX_APPLE9_VIR_JMP_EXEC_ANY ||
        encoding != AGX_APPLE9_ENC_JMP_EXEC_ANY) &&
       (op != AGX_APPLE9_VIR_JMP_EXEC_NONE ||
        encoding != AGX_APPLE9_ENC_JMP_EXEC_NONE))
      return false;

   if (!agx_apple9_vir_emit_side_effect(program, op, encoding, NULL, 0, 0))
      return false;

   program->instructions[program->instruction_count - 1].branch_target =
      target_instruction;
   return true;
}

uint32_t
agx_apple9_vir_input(struct agx_apple9_vir_program *program, unsigned phys)
{
   if (phys >= AGX_APPLE9_GPR_COUNT)
      return AGX_APPLE9_VREG_INVALID;

   return apple9_vir_new_value(program, phys);
}

static bool
apple9_device_load_raw_token_valid(uint16_t raw_token)
{
   switch (raw_token) {
   case AGX_APPLE9_DEVICE_LOAD_TOKEN_1100:
   case AGX_APPLE9_DEVICE_LOAD_TOKEN_5100:
   case AGX_APPLE9_DEVICE_LOAD_TOKEN_9100:
   case AGX_APPLE9_DEVICE_LOAD_TOKEN_D100:
   case AGX_APPLE9_DEVICE_LOAD_TOKEN_1101:
   case AGX_APPLE9_DEVICE_LOAD_TOKEN_5101:
      return true;
   default:
      return false;
   }
}

static bool
apple9_scalar_load_token_for_slot(enum agx_apple9_scoreboard_slot slot,
                                  uint16_t *raw_token)
{
   switch (slot) {
   case AGX_APPLE9_SCOREBOARD_SLOT_1:
      *raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1100;
      return true;
   case AGX_APPLE9_SCOREBOARD_SLOT_2:
      *raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5100;
      return true;
   case AGX_APPLE9_SCOREBOARD_SLOT_3:
      *raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_9100;
      return true;
   case AGX_APPLE9_SCOREBOARD_SLOT_4:
      *raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_D100;
      return true;
   case AGX_APPLE9_SCOREBOARD_SLOT_5:
      *raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_1101;
      return true;
   case AGX_APPLE9_SCOREBOARD_SLOT_6:
      *raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101;
      return true;
   default:
      return false;
   }
}

static bool
apple9_scalar_load_slot_for_token(uint16_t raw_token, uint8_t *slot)
{
   for (unsigned candidate = AGX_APPLE9_SCOREBOARD_SLOT_1;
        candidate <= AGX_APPLE9_SCOREBOARD_SLOT_6; ++candidate) {
      uint16_t token;
      if (apple9_scalar_load_token_for_slot(candidate, &token) &&
          token == raw_token) {
         *slot = candidate;
         return true;
      }
   }

   return false;
}

uint32_t
agx_apple9_vir_emit_device_load(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   const struct agx_apple9_device_load_contract *contract)
{
   if (program == NULL || contract == NULL || binding > UINT8_MAX ||
       (contract->flags & ~(AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX |
                            AGX_APPLE9_DEVICE_LOAD_HAS_NEXT)) ||
       !apple9_device_load_raw_token_valid(contract->raw_token))
      return AGX_APPLE9_VREG_INVALID;

   enum agx_apple9_encoding encoding = AGX_APPLE9_ENC_DEVICE_LOAD;
   const uint32_t *sources = &index;
   unsigned nr_srcs = 1;
   switch (contract->index_kind) {
   case AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR:
   case AGX_APPLE9_DEVICE_LOAD_INDEX_COMPUTED_GPR:
      if (index >= program->value_count)
         return AGX_APPLE9_VREG_INVALID;
      break;
   default:
      return AGX_APPLE9_VREG_INVALID;
   }

   uint32_t value = agx_apple9_vir_emit(program, AGX_APPLE9_VIR_DEVICE_LOAD,
                                        encoding, sources, nr_srcs, binding);
   if (value == AGX_APPLE9_VREG_INVALID)
      return value;

   struct agx_apple9_vir_instr *instruction =
      &program->instructions[program->instruction_count - 1];
   instruction->device_load_flags = contract->flags;
   instruction->device_load_index_kind = contract->index_kind;
   instruction->device_load_index_first_consumer =
      contract->index_first_load_consumer;
   instruction->device_load_raw_token = contract->raw_token;
   if (!apple9_scalar_load_slot_for_token(
          contract->raw_token, &instruction->producer_scoreboard_slot))
      return AGX_APPLE9_VREG_INVALID;
   return value;
}

uint32_t
agx_apple9_vir_emit_device_load_vector(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   unsigned components, const struct agx_apple9_device_load_contract *contract)
{
   if (components < 2 || components > 4)
      return AGX_APPLE9_VREG_INVALID;

   uint32_t value =
      agx_apple9_vir_emit_device_load(program, binding, index, contract);
   if (value == AGX_APPLE9_VREG_INVALID)
      return value;

   if (!apple9_vir_append_values(program, components - 1)) {
      --program->instruction_count;
      program->value_count = value;
      return AGX_APPLE9_VREG_INVALID;
   }

   program->instructions[program->instruction_count - 1].dest_components =
      components;
   return value;
}

uint32_t
agx_apple9_vir_emit_texture_sample(struct agx_apple9_vir_program *program,
                                  const uint32_t coords[2], uint32_t one)
{
   if (!program || !coords || coords[0] >= program->value_count ||
       coords[1] >= program->value_count || one >= program->value_count)
      return AGX_APPLE9_VREG_INVALID;

   uint32_t published = program->value_count;
   if (!apple9_vir_append_values(program, 2))
      return AGX_APPLE9_VREG_INVALID;
   struct agx_apple9_vir_instr *instruction = apple9_vir_append_instruction(program);
   if (!instruction)
      return AGX_APPLE9_VREG_INVALID;
   *instruction = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_TEXTURE_COORDS,
      .encoding = AGX_APPLE9_ENC_TEXTURE_COORDS,
      .dest = published, .dest_components = 2,
      .nr_srcs = 3, .src = {coords[0], coords[1], one},
   };

   uint32_t result = program->value_count;
   if (!apple9_vir_append_values(program, 4))
      return AGX_APPLE9_VREG_INVALID;
   instruction = apple9_vir_append_instruction(program);
   if (!instruction)
      return AGX_APPLE9_VREG_INVALID;
   *instruction = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_TEXTURE_SAMPLE,
      .encoding = AGX_APPLE9_ENC_TEXTURE_SAMPLE,
      .dest = result, .dest_components = 4,
      .nr_srcs = 2, .src = {published, published + 1},
      /* The serial sampling path uses handoff slot 1. Materialize each
       * result before issuing another sample or changing the lane mask. */
      .producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1,
   };
   return result;
}

uint32_t
agx_apple9_vir_emit_collect(struct agx_apple9_vir_program *program,
                            const uint32_t *src, unsigned components)
{
   if (program == NULL || src == NULL || components < 2 || components > 4)
      return AGX_APPLE9_VREG_INVALID;

   for (unsigned c = 0; c < components; ++c) {
      if (src[c] >= program->value_count)
         return AGX_APPLE9_VREG_INVALID;
   }

   const uint32_t dest = program->value_count;
   if (!apple9_vir_append_values(program, components))
      return AGX_APPLE9_VREG_INVALID;

   struct agx_apple9_vir_instr *instruction =
      apple9_vir_append_instruction(program);
   if (instruction == NULL) {
      program->value_count = dest;
      return AGX_APPLE9_VREG_INVALID;
   }

   *instruction = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_COLLECT,
      .encoding = AGX_APPLE9_ENC_PSEUDO,
      .dest = dest,
      .dest_components = components,
      .target = AGX_APPLE9_VREG_INVALID,
      .nr_srcs = components,
   };
   for (unsigned c = 0; c < components; ++c)
      instruction->src[c] = src[c];

   return dest;
}

uint32_t
agx_apple9_vir_emit_merge(struct agx_apple9_vir_program *program)
{
   return agx_apple9_vir_emit(program, AGX_APPLE9_VIR_MERGE,
                              AGX_APPLE9_ENC_PSEUDO, NULL, 0, 0);
}

bool
agx_apple9_vir_emit_masked_copy(struct agx_apple9_vir_program *program,
                                uint32_t target, uint32_t source)
{
   if (program == NULL || target >= program->value_count ||
       source >= program->value_count)
      return false;

   struct agx_apple9_vir_instr *instruction =
      apple9_vir_append_instruction(program);
   if (instruction == NULL)
      return false;

   *instruction = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_MASKED_COPY,
      .encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED,
      .dest = AGX_APPLE9_VREG_INVALID,
      .src = {source},
      .target = target,
      .nr_srcs = 1,
   };
   return true;
}

static bool
apple9_vir_values_form_tuple(const struct agx_apple9_vir_program *program,
                             const uint32_t *values, unsigned components)
{
   if (components < 2 || components > 4)
      return false;

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &program->instructions[i];
      if (instruction->dest != values[0] ||
          instruction->dest_components != components)
         continue;

      bool exact = true;
      for (unsigned c = 0; c < components; ++c)
         exact &= values[c] == instruction->dest + c;
      return exact;
   }

   return false;
}

bool
agx_apple9_vir_emit_device_store(struct agx_apple9_vir_program *program,
                                 unsigned binding, uint32_t index,
                                 const uint32_t *data, unsigned components,
                                 unsigned bits)
{
   if (program == NULL || data == NULL || components < 1 || components > 4 ||
       index >= program->value_count || binding > UINT8_MAX ||
       (bits != 8 && bits != 16 && bits != 32) ||
       (components > 1 && bits != 32))
      return false;

   for (unsigned c = 0; c < components; ++c) {
      if (data[c] >= program->value_count)
         return false;
   }

   const unsigned old_instruction_count = program->instruction_count;
   const unsigned old_value_count = program->value_count;
   uint32_t tuple = AGX_APPLE9_VREG_INVALID;
   if (components > 1 &&
       !apple9_vir_values_form_tuple(program, data, components)) {
      tuple = agx_apple9_vir_emit_collect(program, data, components);
      if (tuple == AGX_APPLE9_VREG_INVALID)
         return false;
   }

   struct agx_apple9_vir_instr *instruction =
      apple9_vir_append_instruction(program);
   if (instruction == NULL) {
      program->instruction_count = old_instruction_count;
      program->value_count = old_value_count;
      return false;
   }

   *instruction = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_DEVICE_STORE,
      .encoding = AGX_APPLE9_ENC_DEVICE_STORE,
      .dest = AGX_APPLE9_VREG_INVALID,
      .target = AGX_APPLE9_VREG_INVALID,
      .memory_bits = bits,
      .memory_components = components,
      .immediate = binding,
      .nr_srcs = components + 1,
   };
   for (unsigned c = 0; c < components; ++c)
      instruction->src[c] =
         tuple == AGX_APPLE9_VREG_INVALID ? data[c] : tuple + c;
   instruction->src[components] = index;
   return true;
}

static bool
apple9_atomic_op_valid(enum agx_apple9_atomic_op op)
{
   switch (op) {
   case AGX_APPLE9_ATOMIC_ADD:
   case AGX_APPLE9_ATOMIC_AND:
   case AGX_APPLE9_ATOMIC_CMPXCHG:
   case AGX_APPLE9_ATOMIC_FADD:
   case AGX_APPLE9_ATOMIC_SMAX:
   case AGX_APPLE9_ATOMIC_SMIN:
   case AGX_APPLE9_ATOMIC_OR:
   case AGX_APPLE9_ATOMIC_SUB:
   case AGX_APPLE9_ATOMIC_UMAX:
   case AGX_APPLE9_ATOMIC_UMIN:
   case AGX_APPLE9_ATOMIC_XCHG:
   case AGX_APPLE9_ATOMIC_XOR:
      return true;
   }

   return false;
}

bool
agx_apple9_vir_emit_device_atomic(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   const uint32_t *data, unsigned data_components,
   enum agx_apple9_atomic_op op, bool discard_result, uint32_t *result_out)
{
   if (result_out != NULL)
      *result_out = AGX_APPLE9_VREG_INVALID;

   const unsigned expected_components =
      op == AGX_APPLE9_ATOMIC_CMPXCHG ? 2 : 1;
   if (program == NULL || data == NULL || binding > UINT8_MAX ||
       index >= program->value_count || !apple9_atomic_op_valid(op) ||
       data_components != expected_components ||
       (!discard_result && result_out == NULL))
      return false;

   for (unsigned c = 0; c < data_components; ++c) {
      if (data[c] >= program->value_count)
         return false;
   }

   const unsigned old_instruction_count = program->instruction_count;
   const unsigned old_value_count = program->value_count;
   uint32_t tuple_values[2] = {data[0], data_components == 2 ? data[1] : 0};
   uint32_t tuple = AGX_APPLE9_VREG_INVALID;
   if (data_components > 1 &&
       !apple9_vir_values_form_tuple(program, tuple_values, data_components)) {
      tuple = agx_apple9_vir_emit_collect(program, tuple_values,
                                          data_components);
      if (tuple == AGX_APPLE9_VREG_INVALID)
         return false;
   }

   uint32_t sources[3];
   if (tuple != AGX_APPLE9_VREG_INVALID) {
      for (unsigned c = 0; c < data_components; ++c)
         sources[c] = tuple + c;
   } else {
      memcpy(sources, tuple_values, data_components * sizeof(sources[0]));
   }
   sources[data_components] = index;

   uint32_t result = AGX_APPLE9_VREG_INVALID;
   const bool emitted =
      discard_result
         ? agx_apple9_vir_emit_side_effect(
              program, AGX_APPLE9_VIR_DEVICE_ATOMIC,
              AGX_APPLE9_ENC_DEVICE_ATOMIC, sources, data_components + 1,
              binding)
         : ((result = agx_apple9_vir_emit(
                program, AGX_APPLE9_VIR_DEVICE_ATOMIC,
                AGX_APPLE9_ENC_DEVICE_ATOMIC, sources, data_components + 1,
                binding)) != AGX_APPLE9_VREG_INVALID);
   if (!emitted) {
      program->instruction_count = old_instruction_count;
      program->value_count = old_value_count;
      return false;
   }

   struct agx_apple9_vir_instr *instruction =
      &program->instructions[program->instruction_count - 1];
   instruction->memory_bits = 32;
   instruction->memory_components = data_components;
   instruction->atomic_op = op;
   instruction->atomic_discard = discard_result;
   /* Bits 12..17 are solely the input dependency mask. Ordinary GPR inputs
    * use mask zero. A direct pending-load input receives its actual producer
    * slot from the common scoreboard pass. This is independent of
    * returned-result publication below. */
   instruction->scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
   /* Returning atomics join the common six-slot allocator. The adjacent
    * DEVICE_ATOMIC_RESULT record names both the selected slot and landing GPR;
    * discarded atomics publish no result. */
   instruction->producer_scoreboard_slot =
      discard_result ? AGX_APPLE9_SCOREBOARD_SLOT_NONE
                     : AGX_APPLE9_SCOREBOARD_SLOT_AUTO;

   if (!discard_result &&
       !agx_apple9_vir_emit_side_effect(
          program, AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT,
          AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT, &result, 1, 0)) {
      program->instruction_count = old_instruction_count;
      program->value_count = old_value_count;
      return false;
   }

   if (!discard_result) {
      struct agx_apple9_vir_instr *publication =
         &program->instructions[program->instruction_count - 1];
      publication->producer_scoreboard_slot =
         instruction->producer_scoreboard_slot;
   }

   if (!discard_result)
      *result_out = result;
   return true;
}

bool
agx_apple9_vir_set_device_load_contract(
   struct agx_apple9_vir_program *program, uint32_t value, uint8_t flags,
   enum agx_apple9_scoreboard_slot scoreboard_slot)
{
   if (scoreboard_slot == AGX_APPLE9_SCOREBOARD_SLOT_AUTO) {
      if (flags & ~(AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX |
                    AGX_APPLE9_DEVICE_LOAD_HAS_NEXT))
         return false;

      for (unsigned i = 0; i < program->instruction_count; ++i) {
         struct agx_apple9_vir_instr *instruction = &program->instructions[i];
         if (instruction->dest != value)
            continue;
         if (instruction->op != AGX_APPLE9_VIR_DEVICE_LOAD)
            return false;

         instruction->device_load_flags = flags;
         instruction->device_load_raw_token = 0;
         instruction->producer_scoreboard_slot =
            AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         return true;
      }

      return false;
   }

   uint16_t raw_token;
   if (!apple9_scalar_load_token_for_slot(scoreboard_slot, &raw_token))
      return false;

   return agx_apple9_vir_set_device_load_raw_contract(program, value, flags,
                                                      raw_token);
}

bool
agx_apple9_vir_set_device_load_raw_contract(
   struct agx_apple9_vir_program *program, uint32_t value, uint8_t flags,
   uint16_t raw_token)
{
   uint8_t scoreboard_slot;
   if ((flags & ~(AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX |
                  AGX_APPLE9_DEVICE_LOAD_HAS_NEXT)) ||
       !apple9_scalar_load_slot_for_token(raw_token, &scoreboard_slot))
      return false;

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *instruction = &program->instructions[i];
      if (instruction->dest != value)
         continue;
      if (instruction->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         return false;

      instruction->device_load_flags = flags;
      instruction->device_load_raw_token = raw_token;
      instruction->producer_scoreboard_slot = scoreboard_slot;
      return true;
   }

   return false;
}

bool
agx_apple9_vir_set_device_load_index_kind(
   struct agx_apple9_vir_program *program, uint32_t value,
   enum agx_apple9_device_load_index_kind index_kind)
{
   if (index_kind != AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR &&
       index_kind != AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR)
      return false;

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *instruction = &program->instructions[i];
      if (instruction->dest != value)
         continue;
      if (instruction->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         return false;
      if (instruction->encoding != AGX_APPLE9_ENC_DEVICE_LOAD ||
          instruction->nr_srcs != 1) {
         return false;
      }
      instruction->device_load_index_kind = index_kind;
      return true;
   }

   return false;
}

bool
agx_apple9_vir_set_fixed_phys(struct agx_apple9_vir_program *program,
                              uint32_t value, unsigned phys)
{
   if (value >= program->value_count || phys >= AGX_APPLE9_GPR_COUNT)
      return false;

   program->fixed_phys[value] = phys;
   return true;
}

bool
agx_apple9_vir_add_live_out(struct agx_apple9_vir_program *program,
                            uint32_t value)
{
   if (value >= program->value_count)
      return false;

   for (unsigned i = 0; i < program->live_out_count; ++i) {
      if (program->live_out[i] == value)
         return true;
   }

   if (program->live_out_count == program->live_out_capacity) {
      unsigned capacity = MAX2(program->live_out_capacity * 2, 8);
      uint32_t *resized =
         realloc(program->live_out, capacity * sizeof(*program->live_out));
      if (resized == NULL)
         return false;
      program->live_out = resized;
      program->live_out_capacity = capacity;
   }

   program->live_out[program->live_out_count++] = value;
   return true;
}

static bool
encoding_tuple(const struct agx_apple9_vir_instr *instruction,
               const uint8_t *phys, unsigned *gprs, unsigned *count)
{
   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(instruction->encoding);
   if (!info->allocator_safe)
      return false;

   if (instruction->op == AGX_APPLE9_VIR_DEVICE_STORE) {
      const unsigned components = instruction->memory_components;
      if (components < 1 || components > 4 ||
          instruction->nr_srcs != components + 1)
         return false;

      /* Machine-table order is address then data base.  COLLECT has already
       * made the remaining vector lanes adjacent before allocation. */
      gprs[0] = phys[instruction->src[components]];
      gprs[1] = phys[instruction->src[0]];
      *count = 2;
      return true;
   }

   if (instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
      const unsigned components = instruction->memory_components;
      if (components < 1 || components > 2 ||
          instruction->nr_srcs != components + 1)
         return false;

      /* The packet names only the address index and RMW data tuple. A
       * returning atomic's adjacent publication record names its result;
       * a discarded atomic has no destination at all. */
      gprs[0] = phys[instruction->src[components]];
      gprs[1] = phys[instruction->src[0]];
      *count = 2;
      return true;
   }

   unsigned gpr_operand_count = 0;
   for (unsigned i = 0; i < info->operand_count; ++i)
      gpr_operand_count += !!(info->operands[i].files & AGX_APPLE9_FILE_GPR);

   const bool has_destination = instruction->dest != AGX_APPLE9_VREG_INVALID;
   if (gpr_operand_count != instruction->nr_srcs + has_destination)
      return false;

   unsigned base = 0;
   if (has_destination)
      gprs[base++] = phys[instruction->dest];
   for (unsigned i = 0; i < instruction->nr_srcs; ++i)
      gprs[base + i] = phys[instruction->src[i]];
   *count = gpr_operand_count;
   return true;
}

static unsigned
apple9_vir_dest_components(const struct agx_apple9_vir_instr *instruction)
{
   if (instruction->dest == AGX_APPLE9_VREG_INVALID)
      return 0;
   return instruction->dest_components ? instruction->dest_components : 1;
}

static const struct agx_apple9_vir_instr *
apple9_vir_producer_instruction(const struct agx_apple9_vir_program *program,
                                uint32_t value);

static bool
apple9_vir_is_graphics_output(enum agx_apple9_vir_opcode op)
{
   return op == AGX_APPLE9_VIR_VARY_STORE || op == AGX_APPLE9_VIR_TILE_ACCESS ||
          op == AGX_APPLE9_VIR_TILE_STORE || op == AGX_APPLE9_VIR_TILE_FENCE;
}

static bool
apple9_vir_is_control_side_effect(enum agx_apple9_vir_opcode op)
{
   return op == AGX_APPLE9_VIR_PREDICATE_COMPARE ||
          op == AGX_APPLE9_VIR_EXEC_MASK_PUSH ||
          op == AGX_APPLE9_VIR_EXEC_MASK_ELSE ||
          op == AGX_APPLE9_VIR_EXEC_MASK_POP ||
          op == AGX_APPLE9_VIR_LOOP_MASK_PUSH ||
          op == AGX_APPLE9_VIR_LOOP_MASK_UPDATE ||
          op == AGX_APPLE9_VIR_LOOP_MASK_POP ||
          op == AGX_APPLE9_VIR_JMP_EXEC_ANY ||
          op == AGX_APPLE9_VIR_JMP_EXEC_NONE ||
          op == AGX_APPLE9_VIR_BREAK_MASK_UNWIND;
}

bool
agx_apple9_validate_vir_allocation(const struct agx_apple9_vir_program *program,
                                   const char **reason)
{
   if (reason != NULL)
      *reason = NULL;
   if (program->phys == NULL) {
      if (reason != NULL)
         *reason = "Apple9 virtual IR has not been allocated";
      return false;
   }

   unsigned execution_mask_depth = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &program->instructions[i];
      const enum agx_apple9_dependency_layout dependency_layout =
         instruction->encoding == AGX_APPLE9_ENC_PSEUDO
            ? AGX_APPLE9_DEPENDENCY_NONE
            : agx_apple9_encoding_info(instruction->encoding)->dependency_layout;
      if (!apple9_dependency_slot_valid(dependency_layout,
                                        instruction->scoreboard_slot)) {
         if (reason != NULL)
            *reason = "Apple9 instruction has an invalid scoreboard dependency";
         return false;
      }

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_STORE) {
         const unsigned components = instruction->memory_components;
         const unsigned bits = instruction->memory_bits;
         if (instruction->dest != AGX_APPLE9_VREG_INVALID || components < 1 ||
             components > 4 || instruction->nr_srcs != components + 1 ||
             (bits != 8 && bits != 16 && bits != 32) ||
             (components > 1 && bits != 32) ||
             instruction->immediate > UINT8_MAX) {
            if (reason != NULL)
               *reason = "Apple9 device store has an invalid VIR contract";
            return false;
         }

         if (instruction->encoding != AGX_APPLE9_ENC_DEVICE_STORE) {
            if (reason != NULL)
               *reason =
                  "Apple9 device store form disagrees with its machine encoding";
            return false;
         }

         const unsigned allocation_bits = bits == 8 ? 16 : bits;
         const unsigned data = program->phys[instruction->src[0]];
         const unsigned index = program->phys[instruction->src[components]];
         bool adjacent = true;
         for (unsigned c = 1; c < components; ++c)
            adjacent &= program->phys[instruction->src[c]] == data + c;
         if (!agx_apple9_encoding_accepts_gpr(
                instruction->encoding, AGX_APPLE9_OPERAND_INDEX, index, 32) ||
             !agx_apple9_encoding_accepts_gpr(instruction->encoding,
                                              AGX_APPLE9_OPERAND_STORE_DATA,
                                              data, allocation_bits) ||
             !adjacent) {
            if (reason != NULL)
               *reason =
                  adjacent
                     ? "Apple9 device store violates an encoding constraint"
                     : "Apple9 vector store source is not an adjacent GPR tuple";
            return false;
         }

         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
         const unsigned components = instruction->memory_components;
         const bool compare_exchange =
            instruction->atomic_op == AGX_APPLE9_ATOMIC_CMPXCHG;
         const unsigned data = program->phys[instruction->src[0]];
         bool adjacent = true;
         for (unsigned c = 1; c < components; ++c)
            adjacent &= program->phys[instruction->src[c]] == data + c;
         const unsigned index = program->phys[instruction->src[components]];
         const bool scoreboard_valid =
            instruction->atomic_discard
               ? instruction->producer_scoreboard_slot ==
                    AGX_APPLE9_SCOREBOARD_SLOT_NONE
               : instruction->producer_scoreboard_slot >=
                       AGX_APPLE9_SCOREBOARD_SLOT_1 &&
                    instruction->producer_scoreboard_slot <=
                       AGX_APPLE9_SCOREBOARD_SLOT_6;
         const bool valid =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC &&
            instruction->memory_bits == 32 &&
            components == (compare_exchange ? 2 : 1) &&
            instruction->nr_srcs == components + 1 &&
            (instruction->atomic_discard
                ? instruction->dest == AGX_APPLE9_VREG_INVALID
                : instruction->dest < program->value_count) &&
            instruction->immediate <= UINT8_MAX &&
            apple9_atomic_op_valid(instruction->atomic_op) && adjacent &&
            agx_apple9_encoding_accepts_gpr(
               instruction->encoding, AGX_APPLE9_OPERAND_INDEX, index, 32) &&
            agx_apple9_encoding_accepts_gpr(
               instruction->encoding, AGX_APPLE9_OPERAND_ATOMIC_DATA, data,
               32) && scoreboard_valid;
         if (!valid) {
            if (reason != NULL)
               *reason = "Apple9 device atomic has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
         const bool valid =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT &&
            instruction->dest == AGX_APPLE9_VREG_INVALID &&
            instruction->nr_srcs == 1 && instruction->immediate == 0 &&
            instruction->src[0] < program->value_count && i > 0 &&
            program->instructions[i - 1].op ==
               AGX_APPLE9_VIR_DEVICE_ATOMIC &&
            !program->instructions[i - 1].atomic_discard &&
            program->instructions[i - 1].producer_scoreboard_slot >=
               AGX_APPLE9_SCOREBOARD_SLOT_1 &&
            program->instructions[i - 1].producer_scoreboard_slot <=
               AGX_APPLE9_SCOREBOARD_SLOT_6 &&
            instruction->producer_scoreboard_slot ==
               program->instructions[i - 1].producer_scoreboard_slot &&
            program->instructions[i - 1].dest == instruction->src[0] &&
            program->phys[instruction->src[0]] < 64;
         if (!valid) {
            if (reason != NULL)
               *reason =
                  "Apple9 atomic result materializer has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_MERGE) {
         const bool valid =
            instruction->encoding == AGX_APPLE9_ENC_PSEUDO &&
            instruction->dest != AGX_APPLE9_VREG_INVALID &&
            instruction->dest_components == 1 && instruction->nr_srcs == 0 &&
            instruction->target == AGX_APPLE9_VREG_INVALID &&
            instruction->immediate == 0 && instruction->memory_bits == 0 &&
            instruction->memory_components == 0 &&
            instruction->producer_scoreboard_slot ==
               AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
            instruction->scoreboard_slot == AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
            instruction->dest < program->value_count &&
            program->phys[instruction->dest] <= APPLE9_LAST_ALLOCATABLE_GPR;
         if (!valid) {
            if (reason != NULL)
               *reason = "Apple9 MERGE has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_MASKED_COPY) {
         const struct agx_apple9_vir_instr *target_producer =
            apple9_vir_producer_instruction(program, instruction->target);
         const unsigned target = instruction->target < program->value_count
                                    ? program->phys[instruction->target]
                                    : AGX_APPLE9_PHYS_INVALID;
         const unsigned source =
            instruction->nr_srcs == 1 &&
                  instruction->src[0] < program->value_count
               ? program->phys[instruction->src[0]]
               : AGX_APPLE9_PHYS_INVALID;
         const unsigned tuple[] = {target, source, source};
         const bool valid =
            instruction->encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED &&
            instruction->dest == AGX_APPLE9_VREG_INVALID &&
            instruction->nr_srcs == 1 &&
            instruction->target < program->value_count &&
            instruction->src[0] < program->value_count &&
            target_producer != NULL &&
            target_producer->op == AGX_APPLE9_VIR_MERGE &&
            instruction->immediate == 0 && instruction->memory_bits == 0 &&
            instruction->memory_components == 0 &&
            instruction->producer_scoreboard_slot ==
               AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
            agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                  ARRAY_SIZE(tuple), 32);
         if (!valid) {
            if (reason != NULL)
               *reason = "Apple9 MASKED_COPY has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (apple9_vir_is_control_side_effect(instruction->op)) {
         bool valid =
            instruction->dest == AGX_APPLE9_VREG_INVALID &&
            instruction->memory_bits == 0 &&
            instruction->memory_components == 0 &&
            instruction->producer_scoreboard_slot ==
               AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
            instruction->scoreboard_slot == AGX_APPLE9_SCOREBOARD_SLOT_NONE;

         switch (instruction->op) {
         case AGX_APPLE9_VIR_PREDICATE_COMPARE: {
            const unsigned condition = instruction->immediate & 0xff;
            const unsigned predicate_bank =
               (instruction->immediate & AGX_APPLE9_PREDICATE_BANK_MASK) >>
               AGX_APPLE9_PREDICATE_BANK_SHIFT;
            const bool short_form =
               instruction->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT;
            const bool extended_form =
               instruction->encoding ==
               AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
            const bool loop_form =
               instruction->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP;
            const bool valid_condition =
               short_form
                  ? (condition == AGX_APPLE9_PREDICATE_FGT ||
                     condition == AGX_APPLE9_PREDICATE_FLT ||
                     condition == AGX_APPLE9_PREDICATE_UGT ||
                     condition == AGX_APPLE9_PREDICATE_ULT ||
                     condition == AGX_APPLE9_PREDICATE_IGT ||
                     condition == AGX_APPLE9_PREDICATE_ILT)
                  : (extended_form || loop_form) &&
                       (condition == AGX_APPLE9_PREDICATE_EXT_FEQ ||
                        condition == AGX_APPLE9_PREDICATE_EXT_FGE_SEQUENCE ||
                        condition == AGX_APPLE9_PREDICATE_EXT_FLE_SEQUENCE ||
                        condition == AGX_APPLE9_PREDICATE_EXT_IEQ);
            valid &= (short_form || extended_form || loop_form) &&
                     instruction->nr_srcs == 2 &&
                     (instruction->immediate &
                      ~(0xffu | AGX_APPLE9_PREDICATE_INVERT |
                        AGX_APPLE9_PREDICATE_BANK_MASK)) == 0 &&
                     predicate_bank < AGX_APPLE9_PREDICATE_BANK_COUNT &&
                     valid_condition;
            break;
         }
         case AGX_APPLE9_VIR_EXEC_MASK_PUSH: {
            const unsigned selector = instruction->immediate & 0xffu;
            valid &= instruction->encoding == AGX_APPLE9_ENC_EXEC_MASK_PUSH &&
                     instruction->nr_srcs == 0 &&
                     (instruction->immediate &
                      ~(0xffu | AGX_APPLE9_EXEC_MASK_INVERT)) == 0 &&
                     selector >= AGX_APPLE9_EXEC_MASK_SOURCE(0) &&
                     selector <= AGX_APPLE9_EXEC_MASK_SOURCE(7) &&
                     ((selector - AGX_APPLE9_EXEC_MASK_SOURCE(0)) % 4) == 0;
            if (valid)
               ++execution_mask_depth;
            break;
         }
         case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
            valid &= instruction->encoding == AGX_APPLE9_ENC_EXEC_MASK_ELSE &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0 &&
                     execution_mask_depth > 0;
            break;
         case AGX_APPLE9_VIR_EXEC_MASK_POP:
            valid &= instruction->encoding == AGX_APPLE9_ENC_EXEC_MASK_POP &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0 &&
                     execution_mask_depth > 0;
            if (valid)
               --execution_mask_depth;
            break;
         case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
            valid &= instruction->encoding == AGX_APPLE9_ENC_LOOP_MASK_PUSH &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0;
            break;
         case AGX_APPLE9_VIR_LOOP_MASK_UPDATE: {
            const unsigned selector = instruction->immediate;
            const unsigned bank_selector =
               selector & ~AGX_APPLE9_LOOP_MASK_INVERT;
            valid &=
               instruction->encoding == AGX_APPLE9_ENC_LOOP_MASK_UPDATE &&
               instruction->nr_srcs == 0 &&
               (selector & ~(0x1fu | AGX_APPLE9_LOOP_MASK_INVERT)) == 0 &&
               bank_selector >= AGX_APPLE9_LOOP_MASK_PREDICATE(0) &&
               bank_selector <= AGX_APPLE9_LOOP_MASK_PREDICATE(
                                   AGX_APPLE9_PREDICATE_BANK_COUNT - 1) &&
               ((bank_selector - AGX_APPLE9_LOOP_MASK_PREDICATE(0)) % 4) == 0;
            break;
         }
         case AGX_APPLE9_VIR_LOOP_MASK_POP:
            valid &= instruction->encoding == AGX_APPLE9_ENC_LOOP_MASK_POP &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0;
            break;
         case AGX_APPLE9_VIR_JMP_EXEC_ANY:
         case AGX_APPLE9_VIR_JMP_EXEC_NONE:
            valid &= instruction->encoding ==
                        (instruction->op == AGX_APPLE9_VIR_JMP_EXEC_ANY
                            ? AGX_APPLE9_ENC_JMP_EXEC_ANY
                            : AGX_APPLE9_ENC_JMP_EXEC_NONE) &&
                     instruction->nr_srcs == 0 && instruction->immediate == 0 &&
                     instruction->branch_target <= program->instruction_count;
            break;
         case AGX_APPLE9_VIR_BREAK_MASK_UNWIND: {
            const unsigned scope_tag =
               AGX_APPLE9_BREAK_SCOPE_TAG(instruction->immediate);
            const unsigned loop_depth =
               AGX_APPLE9_BREAK_LOOP_DEPTH(instruction->immediate);
            valid &=
               instruction->encoding == AGX_APPLE9_ENC_BREAK_MASK_UNWIND &&
               instruction->nr_srcs == 0 && scope_tag >= 2 && loop_depth >= 1;
            break;
         }
         default:
            valid = false;
            break;
         }

         unsigned tuple[AGX_APPLE9_MAX_ENCODING_OPERANDS] = {0};
         unsigned tuple_count = 0;
         valid &=
            encoding_tuple(instruction, program->phys, tuple, &tuple_count) &&
            agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                  tuple_count, 32);
         if (!valid) {
            if (reason != NULL)
               *reason =
                  "Apple9 control-flow instruction has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_VARY_STORE ||
          instruction->op == AGX_APPLE9_VIR_TILE_ACCESS ||
          instruction->op == AGX_APPLE9_VIR_TILE_STORE ||
          instruction->op == AGX_APPLE9_VIR_TILE_FENCE) {
         unsigned gprs[AGX_APPLE9_MAX_ENCODING_OPERANDS], count;
         struct agx_apple9_packed_instruction packed;
         if (instruction->dest != AGX_APPLE9_VREG_INVALID ||
             !encoding_tuple(instruction, program->phys, gprs, &count) ||
             !agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, gprs,
                                                    count, 32) ||
             !agx_apple9_pack_vir_instruction(instruction, program->phys,
                                              &packed, reason)) {
            if (reason)
               *reason = "Apple9 graphics output has an invalid contract";
            return false;
         }
         continue;
      }

      const unsigned components = apple9_vir_dest_components(instruction);
      if (components > 4 || components > program->value_count ||
          instruction->dest > program->value_count - components ||
          (components > 1 && instruction->op != AGX_APPLE9_VIR_DEVICE_LOAD &&
           instruction->op != AGX_APPLE9_VIR_COLLECT &&
           instruction->op != AGX_APPLE9_VIR_TEXTURE_COORDS &&
           instruction->op != AGX_APPLE9_VIR_TEXTURE_SAMPLE)) {
         if (reason != NULL)
            *reason = "Apple9 instruction has an invalid destination tuple";
         return false;
      }
      for (unsigned c = 1; c < components; ++c) {
         if (program->phys[instruction->dest + c] !=
             program->phys[instruction->dest] + c) {
            if (reason != NULL)
               *reason =
                  "Apple9 vector destination is not an adjacent GPR tuple";
            return false;
         }
      }
      for (unsigned c = 0; c < components; ++c) {
         const unsigned maximum = program->max_phys[instruction->dest + c];
         if (maximum != AGX_APPLE9_PHYS_INVALID &&
             program->phys[instruction->dest + c] > maximum) {
            if (reason != NULL)
               *reason =
                  "Apple9 allocation violates a value register-class constraint";
            return false;
         }
      }

      if (instruction->op == AGX_APPLE9_VIR_COLLECT) {
         if (components < 2 || instruction->nr_srcs != components ||
             instruction->encoding != AGX_APPLE9_ENC_PSEUDO ||
             instruction->memory_bits != 0 ||
             instruction->memory_components != 0 ||
             instruction->producer_scoreboard_slot !=
                AGX_APPLE9_SCOREBOARD_SLOT_NONE ||
             instruction->scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE) {
            if (reason != NULL)
               *reason = "Apple9 COLLECT has an invalid VIR contract";
            return false;
         }

         bool identity = true;
         bool disjoint = true;
         for (unsigned d = 0; d < components; ++d) {
            identity &= program->phys[instruction->dest + d] ==
                        program->phys[instruction->src[d]];
            for (unsigned s = 0; s < components; ++s) {
               disjoint &= program->phys[instruction->dest + d] !=
                           program->phys[instruction->src[s]];
            }
         }
         if (!identity && !disjoint) {
            if (reason != NULL)
               *reason =
                  "Apple9 COLLECT allocation requires an unsupported overlapping shuffle";
            return false;
         }
         continue;
      }

      unsigned tuple[AGX_APPLE9_MAX_ENCODING_OPERANDS] = {0};
      unsigned tuple_count = 0;
      if (!encoding_tuple(instruction, program->phys, tuple, &tuple_count) ||
          !agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                 tuple_count, 32)) {
         if (getenv("AGX_APPLE9_TRACE") != NULL) {
            fprintf(stderr,
                    "APPLE9_VALIDATE_FAIL i=%u op=%u enc=%u dst=r%u src=", i,
                    instruction->op, instruction->encoding,
                    program->phys[instruction->dest]);
            for (unsigned s = 0; s < instruction->nr_srcs; ++s)
               fprintf(stderr, "%sr%u", s ? "," : "",
                       program->phys[instruction->src[s]]);
            fputc('\n', stderr);
         }
         if (reason != NULL)
            *reason = "Apple9 allocation violates an encoding constraint";
         return false;
      }

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         if (instruction->nr_srcs != 1 ||
             (instruction->device_load_index_kind !=
                 AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR &&
              instruction->device_load_index_kind !=
                 AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR)) {
            if (reason != NULL)
               *reason = "Apple9 device load has an invalid index contract";
            return false;
         }

         const bool last_use = !(instruction->live_after_mask & 1u);
         const bool expected_last_use =
            instruction->device_load_index_kind ==
            AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR;
         if (last_use != expected_last_use) {
            if (reason != NULL)
               *reason =
                  "Apple9 device-load index lifetime disagrees with VIR liveness";
            return false;
         }

         const struct agx_apple9_vir_instr *index_producer =
            apple9_vir_producer_instruction(program, instruction->src[0]);
         bool earlier_load_consumer = false;
         for (unsigned p = 0; p < i; ++p) {
            const struct agx_apple9_vir_instr *candidate =
               &program->instructions[p];
            if (candidate->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
                candidate->nr_srcs == 1 &&
                candidate->src[0] == instruction->src[0])
               earlier_load_consumer = true;
         }

         const bool expected_first_consumer =
            index_producer != NULL &&
            index_producer->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
            !earlier_load_consumer;
         if (instruction->device_load_index_first_consumer !=
             expected_first_consumer) {
            if (reason != NULL)
               *reason =
                  "Apple9 device-load dependent-index envelope violates its scoreboard contract";
            return false;
         }
      }
   }

   if (execution_mask_depth != 0) {
      if (reason != NULL)
         *reason = "Apple9 execution-mask stack is unbalanced";
      return false;
   }

   return true;
}

static bool
apple9_propagate_source_register_classes(struct agx_apple9_vir_program *program,
                                         const char **reason)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &program->instructions[i];
      if (instruction->op == AGX_APPLE9_VIR_MASKED_COPY) {
         if (instruction->target >= program->value_count ||
             instruction->nr_srcs != 1 ||
             instruction->src[0] >= program->value_count)
            goto invalid;

         const struct agx_apple9_operand_constraint *dest =
            agx_apple9_find_operand(instruction->encoding,
                                    AGX_APPLE9_OPERAND_DEST);
         const struct agx_apple9_operand_constraint *src0 =
            agx_apple9_find_operand(instruction->encoding,
                                    AGX_APPLE9_OPERAND_SRC0);
         const struct agx_apple9_operand_constraint *src1 =
            agx_apple9_find_operand(instruction->encoding,
                                    AGX_APPLE9_OPERAND_SRC1);
         if (dest == NULL || src0 == NULL || src1 == NULL)
            goto invalid;

         uint32_t target = instruction->target;
         uint32_t source = instruction->src[0];
         if (program->max_phys[target] == AGX_APPLE9_PHYS_INVALID ||
             dest->max_index < program->max_phys[target])
            program->max_phys[target] = dest->max_index;
         const uint8_t source_max = MIN2(src0->max_index, src1->max_index);
         if (program->max_phys[source] == AGX_APPLE9_PHYS_INVALID ||
             source_max < program->max_phys[source])
            program->max_phys[source] = source_max;
         continue;
      }
      if (instruction->encoding == AGX_APPLE9_ENC_PSEUDO)
         continue;

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_STORE) {
         const unsigned components = instruction->memory_components;
         if (components < 1 || components > 4 ||
             instruction->nr_srcs != components + 1)
            goto invalid;

         const struct agx_apple9_operand_constraint *data =
            agx_apple9_find_operand(instruction->encoding,
                                    AGX_APPLE9_OPERAND_STORE_DATA);
         const struct agx_apple9_operand_constraint *index =
            agx_apple9_find_operand(instruction->encoding,
                                    AGX_APPLE9_OPERAND_INDEX);
         if (data == NULL || index == NULL)
            goto invalid;

         for (unsigned c = 0; c < components; ++c) {
            const uint32_t value = instruction->src[c];
            if (value >= program->value_count || data->max_index + c >= 0xff)
               goto invalid;
            const uint8_t maximum = data->max_index + c;
            if (program->max_phys[value] == AGX_APPLE9_PHYS_INVALID ||
                maximum < program->max_phys[value])
               program->max_phys[value] = maximum;
         }

         const uint32_t value = instruction->src[components];
         if (value >= program->value_count)
            goto invalid;
         if (program->max_phys[value] == AGX_APPLE9_PHYS_INVALID ||
             index->max_index < program->max_phys[value])
            program->max_phys[value] = index->max_index;
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
         const unsigned components = instruction->memory_components;
         if (components < 1 || components > 2 ||
             instruction->nr_srcs != components + 1 ||
             (instruction->atomic_discard
                 ? instruction->dest != AGX_APPLE9_VREG_INVALID
                 : instruction->dest >= program->value_count))
            goto invalid;

         const struct agx_apple9_operand_constraint *data =
            agx_apple9_find_operand(instruction->encoding,
                                    AGX_APPLE9_OPERAND_ATOMIC_DATA);
         const struct agx_apple9_operand_constraint *index =
            agx_apple9_find_operand(instruction->encoding,
                                    AGX_APPLE9_OPERAND_INDEX);
         if (data == NULL || index == NULL)
            goto invalid;

         for (unsigned c = 0; c < components; ++c) {
            const uint32_t value = instruction->src[c];
            if (value >= program->value_count ||
                data->max_index + c >= AGX_APPLE9_PHYS_INVALID)
               goto invalid;
            const uint8_t maximum = data->max_index + c;
            if (program->max_phys[value] == AGX_APPLE9_PHYS_INVALID ||
                maximum < program->max_phys[value])
               program->max_phys[value] = maximum;
         }

         const uint32_t address = instruction->src[components];
         if (address >= program->value_count)
            goto invalid;
         if (program->max_phys[address] == AGX_APPLE9_PHYS_INVALID ||
             index->max_index < program->max_phys[address])
            program->max_phys[address] = index->max_index;
         continue;
      }

      const struct agx_apple9_encoding_info *info =
         agx_apple9_encoding_info(instruction->encoding);
      unsigned source = 0;
      for (unsigned operand = 0; operand < info->operand_count; ++operand) {
         const struct agx_apple9_operand_constraint *constraint =
            &info->operands[operand];
         if (!(constraint->files & AGX_APPLE9_FILE_GPR) ||
             constraint->role == AGX_APPLE9_OPERAND_DEST)
            continue;
         if (source >= instruction->nr_srcs ||
             instruction->src[source] >= program->value_count)
            goto invalid;

         const uint32_t value = instruction->src[source++];
         if (program->max_phys[value] == AGX_APPLE9_PHYS_INVALID ||
             constraint->max_index < program->max_phys[value])
            program->max_phys[value] = constraint->max_index;
      }
      if (source != instruction->nr_srcs)
         goto invalid;
   }

   return true;

invalid:
   if (reason != NULL)
      *reason = "Apple9 VIR has an invalid source register-class contract";
   return false;
}

static unsigned
apple9_vir_definition_index(const struct agx_apple9_vir_program *program,
                            uint32_t value)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &program->instructions[i];
      const unsigned components = apple9_vir_dest_components(instruction);
      if (components != 0 && value >= instruction->dest &&
          value - instruction->dest < components)
         return i;
   }

   /* Preloaded ABI inputs have no VIR definition. */
   return UINT_MAX;
}

bool
agx_apple9_allocate_vir(struct agx_apple9_vir_program *program,
                        const char **reason)
{
   if (reason != NULL)
      *reason = NULL;
   bool has_side_effect = false;
   for (unsigned i = 0; i < program->instruction_count; ++i)
      has_side_effect |=
         program->instructions[i].op == AGX_APPLE9_VIR_DEVICE_STORE ||
         program->instructions[i].op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
         apple9_vir_is_graphics_output(program->instructions[i].op);
   if ((program->output == AGX_APPLE9_VREG_INVALID && !has_side_effect) ||
       (program->output != AGX_APPLE9_VREG_INVALID &&
        program->output >= program->value_count)) {
      if (reason != NULL)
         *reason = "Apple9 virtual IR has neither an output nor a store";
      return false;
   }

   if (!apple9_propagate_source_register_classes(program, reason))
      return false;

   unsigned *last_use = calloc(program->value_count, sizeof(*last_use));
   bool *defined = calloc(program->value_count, sizeof(*defined));
   bool *seen_definition =
      calloc(program->value_count, sizeof(*seen_definition));
   bool *used = calloc(program->value_count, sizeof(*used));
   uint8_t *phys = malloc(program->value_count);
   if (last_use == NULL || defined == NULL || seen_definition == NULL ||
       used == NULL || phys == NULL) {
      free(last_use);
      free(defined);
      free(seen_definition);
      free(used);
      free(phys);
      if (reason != NULL)
         *reason = "out of memory allocating Apple9 virtual IR";
      return false;
   }
   memset(phys, AGX_APPLE9_PHYS_INVALID, program->value_count);

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &program->instructions[i];
      const uint32_t dest = instruction->dest;
      const unsigned components = apple9_vir_dest_components(instruction);
      if (components == 0) {
         if (instruction->op != AGX_APPLE9_VIR_DEVICE_STORE &&
             instruction->op != AGX_APPLE9_VIR_DEVICE_ATOMIC &&
             instruction->op != AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT &&
             instruction->op != AGX_APPLE9_VIR_MASKED_COPY &&
             !apple9_vir_is_control_side_effect(instruction->op) &&
             !apple9_vir_is_graphics_output(instruction->op)) {
            free(last_use);
            free(defined);
            free(seen_definition);
            free(used);
            free(phys);
            if (reason != NULL)
               *reason =
                  "Apple9 destination-less VIR instruction has side effects unknown to the allocator";
            return false;
         }
         continue;
      }
      if (components > 4 || components > program->value_count ||
          dest > program->value_count - components) {
         free(last_use);
         free(defined);
         free(seen_definition);
         free(used);
         free(phys);
         if (reason != NULL)
            *reason = "Apple9 virtual IR has an invalid SSA definition";
         return false;
      }
      for (unsigned c = 0; c < components; ++c) {
         if (defined[dest + c]) {
            free(last_use);
            free(defined);
            free(seen_definition);
            free(used);
            free(phys);
            if (reason != NULL)
               *reason = "Apple9 virtual IR has an invalid SSA definition";
            return false;
         }
         defined[dest + c] = true;
      }
   }

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &program->instructions[i];
      const unsigned components = apple9_vir_dest_components(instruction);
      if (components != 0) {
         for (unsigned c = 0; c < components; ++c) {
            seen_definition[instruction->dest + c] = true;
            used[instruction->dest + c] = true;
            last_use[instruction->dest + c] = i;
         }
      }
      for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
         if (instruction->src[s] >= program->value_count ||
             (defined[instruction->src[s]] &&
              !seen_definition[instruction->src[s]])) {
            free(last_use);
            free(defined);
            free(seen_definition);
            free(used);
            free(phys);
            if (reason != NULL)
               *reason = "Apple9 virtual IR is not in SSA definition order";
            return false;
         }
         used[instruction->src[s]] = true;
         last_use[instruction->src[s]] = i;
      }
      if (instruction->op == AGX_APPLE9_VIR_MASKED_COPY) {
         const uint32_t target = instruction->target;
         if (target >= program->value_count || !seen_definition[target]) {
            free(last_use);
            free(defined);
            free(seen_definition);
            free(used);
            free(phys);
            if (reason != NULL)
               *reason =
                  "Apple9 masked copy target is not defined before its edge";
            return false;
         }
         used[target] = true;
         last_use[target] = i;
      }
   }

   /* The allocator is interval-based, but loop bodies are emitted once and
    * reached repeatedly through an explicit backward JMP_EXEC_ANY.  Any
    * value defined before the header and read in the loop is therefore live
    * through the backedge, even if its final textual read occurs earlier.
    * This includes MERGE storage populated by MASKED_COPY on the backedge.
    *
    * Extending these intervals is the structured-CFG equivalent of the
    * standard live-in set at a loop header.  It prevents loop-carried and
    * invariant registers from being recycled by latch computations without
    * introducing fixed physical assignments. */
   for (unsigned branch = 0; branch < program->instruction_count; ++branch) {
      const struct agx_apple9_vir_instr *jump = &program->instructions[branch];
      if (jump->op != AGX_APPLE9_VIR_JMP_EXEC_ANY ||
          jump->branch_target >= branch)
         continue;

      const unsigned header = jump->branch_target;
      for (unsigned i = header; i < branch; ++i) {
         const struct agx_apple9_vir_instr *instruction =
            &program->instructions[i];
         for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
            const uint32_t value = instruction->src[s];
            const unsigned definition =
               apple9_vir_definition_index(program, value);
            if (definition == UINT_MAX || definition < header)
               last_use[value] = MAX2(last_use[value], branch);
         }

         if (instruction->op == AGX_APPLE9_VIR_MASKED_COPY) {
            const uint32_t target = instruction->target;
            const unsigned definition =
               apple9_vir_definition_index(program, target);
            if (definition == UINT_MAX || definition < header)
               last_use[target] = MAX2(last_use[target], branch);
         }
      }
   }

   /* A device load defines its whole architectural destination tuple when
    * the asynchronous return is handed off.  Even tuple lanes that have no
    * SSA user may therefore not be reused between the load and that handoff:
    * a later producer targeting the same GPR would race the pending return.
    *
    * Ordinary SSA liveness is sufficient for every used lane.  Extend only
    * otherwise-dead lanes through the first tuple consumer, which is the
    * instruction that consumes the load's scoreboard slot and makes the
    * returned tuple available through the ordinary GPR path. */
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *load = &program->instructions[i];
      const unsigned components = apple9_vir_dest_components(load);
      if (load->op != AGX_APPLE9_VIR_DEVICE_LOAD || components == 0)
         continue;

      unsigned handoff = UINT_MAX;
      for (unsigned j = i + 1;
           j < program->instruction_count && handoff == UINT_MAX; ++j) {
         const struct agx_apple9_vir_instr *candidate =
            &program->instructions[j];
         for (unsigned s = 0; s < candidate->nr_srcs; ++s) {
            if (candidate->src[s] >= load->dest &&
                candidate->src[s] - load->dest < components) {
               handoff = j;
               break;
            }
         }
      }

      if (handoff == UINT_MAX)
         continue;
      for (unsigned c = 0; c < components; ++c)
         last_use[load->dest + c] = MAX2(last_use[load->dest + c], handoff);
   }

   if (program->output != AGX_APPLE9_VREG_INVALID) {
      used[program->output] = true;
      last_use[program->output] = program->instruction_count;
   }

   for (unsigned i = 0; i < program->live_out_count; ++i) {
      uint32_t value = program->live_out[i];
      if (value >= program->value_count) {
         free(last_use);
         free(defined);
         free(seen_definition);
         free(used);
         free(phys);
         if (reason != NULL)
            *reason = "Apple9 virtual IR has an invalid live-out value";
         return false;
      }
      used[value] = true;
      last_use[value] = program->instruction_count;
   }

   bool fixed_gpr[AGX_APPLE9_GPR_COUNT] = {0};
   for (uint32_t value = 0; value < program->value_count; ++value) {
      if (program->fixed_phys[value] != AGX_APPLE9_PHYS_INVALID)
         fixed_gpr[program->fixed_phys[value]] = true;
   }

   int32_t owner[AGX_APPLE9_GPR_COUNT];
   for (unsigned gpr = 0; gpr < AGX_APPLE9_GPR_COUNT; ++gpr)
      owner[gpr] = (program->reserved_gprs[gpr] || fixed_gpr[gpr]) ? -2 : -1;
   unsigned live = 0;
   unsigned peak = 0;
   unsigned max_phys = 1;

   /* Preloaded inputs have no defining instruction and are live at entry. */
   for (uint32_t value = 0; value < program->value_count; ++value) {
      if (defined[value] || !used[value])
         continue;

      unsigned fixed = program->fixed_phys[value];
      if (fixed == AGX_APPLE9_PHYS_INVALID || fixed >= AGX_APPLE9_GPR_COUNT ||
          owner[fixed] >= 0) {
         free(last_use);
         free(defined);
         free(seen_definition);
         free(used);
         free(phys);
         if (reason != NULL)
            *reason = "Apple9 virtual IR has conflicting fixed inputs";
         return false;
      }

      phys[value] = fixed;
      owner[fixed] = value;
      ++live;
      if (live > peak)
         peak = live;
      if (fixed > max_phys)
         max_phys = fixed;
   }

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *instruction = &program->instructions[i];
      const unsigned components = apple9_vir_dest_components(instruction);

      instruction->live_after_mask = 0;
      for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
         if (last_use[instruction->src[s]] > i)
            instruction->live_after_mask |= 1u << s;
      }

      if (components == 0) {
         for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
            const uint32_t value = instruction->src[s];
            bool duplicate = false;
            for (unsigned earlier = 0; earlier < s; ++earlier)
               duplicate |= instruction->src[earlier] == value;
            if (last_use[value] == i && !duplicate) {
               if (owner[phys[value]] == (int32_t)value)
                  owner[phys[value]] = (program->reserved_gprs[phys[value]] ||
                                        fixed_gpr[phys[value]])
                                          ? -2
                                          : -1;
               assert(live > 0);
               --live;
            }
         }
         if (instruction->op == AGX_APPLE9_VIR_MASKED_COPY) {
            const uint32_t target = instruction->target;
            if (phys[target] == AGX_APPLE9_PHYS_INVALID ||
                owner[phys[target]] != (int32_t)target) {
               free(last_use);
               free(defined);
               free(seen_definition);
               free(used);
               free(phys);
               if (reason != NULL)
                  *reason = "Apple9 masked copy target is not live";
               return false;
            }
            if (last_use[target] == i) {
               owner[phys[target]] = (program->reserved_gprs[phys[target]] ||
                                      fixed_gpr[phys[target]])
                                        ? -2
                                        : -1;
               assert(live > 0);
               --live;
            }
         }

         /* CFG-induced interval ends (most notably a loop backedge) need not
          * be explicit machine operands. Release those values after the
          * control instruction that ends their interval. */
         for (uint32_t value = 0; value < program->value_count; ++value) {
            if (last_use[value] == i &&
                phys[value] != AGX_APPLE9_PHYS_INVALID &&
                owner[phys[value]] == (int32_t)value) {
               owner[phys[value]] = (program->reserved_gprs[phys[value]] ||
                                     fixed_gpr[phys[value]])
                                       ? -2
                                       : -1;
               assert(live > 0);
               --live;
            }
         }
         continue;
      }

      unsigned selected = AGX_APPLE9_PHYS_INVALID;
      bool has_fixed = false;
      for (unsigned c = 0; c < components; ++c) {
         const unsigned fixed = program->fixed_phys[instruction->dest + c];
         if (fixed == AGX_APPLE9_PHYS_INVALID)
            continue;
         if (fixed < c || (has_fixed && selected != fixed - c)) {
            selected = AGX_APPLE9_PHYS_INVALID;
            has_fixed = true;
            break;
         }
         selected = fixed - c;
         has_fixed = true;
      }

      bool coalesced_collect = false;
      if (instruction->op == AGX_APPLE9_VIR_COLLECT &&
          instruction->nr_srcs == components) {
         const unsigned candidate = phys[instruction->src[0]];
         bool exact = candidate != AGX_APPLE9_PHYS_INVALID &&
                      candidate + components <= AGX_APPLE9_GPR_COUNT &&
                      (!has_fixed || selected == candidate);
         for (unsigned c = 0; c < components && exact; ++c) {
            const uint32_t source = instruction->src[c];
            const unsigned maximum =
               program->max_phys[instruction->dest + c];
            exact &= phys[source] == candidate + c && last_use[source] == i &&
                     (maximum == AGX_APPLE9_PHYS_INVALID ||
                      candidate + c <= maximum) &&
                     !program->reserved_gprs[candidate + c] &&
                     !fixed_gpr[candidate + c];
         }
         if (exact) {
            selected = candidate;
            coalesced_collect = true;
         }
      }

      const bool is_pseudo_definition =
         instruction->op == AGX_APPLE9_VIR_COLLECT ||
         instruction->op == AGX_APPLE9_VIR_MERGE;
      const struct agx_apple9_operand_constraint *dest_constraint =
         is_pseudo_definition
            ? NULL
            : agx_apple9_find_operand(instruction->encoding,
                                      AGX_APPLE9_OPERAND_DEST);
      unsigned range_first[2] = {APPLE9_FIRST_ALLOCATABLE_GPR,
                                 APPLE9_FIRST_ALLOCATABLE_GPR};
      unsigned range_last[2] = {APPLE9_LAST_ALLOCATABLE_GPR,
                                APPLE9_LAST_ALLOCATABLE_GPR};
      unsigned range_count = 1;
      if (has_fixed || coalesced_collect) {
         range_first[0] = range_last[0] = selected;
      } else if ((is_pseudo_definition && dest_constraint == NULL) ||
                 (dest_constraint != NULL &&
                  dest_constraint->max_index >= APPLE9_FIRST_GENERAL_GPR)) {
         /* Keep the compact bank available for instructions whose result has
          * a genuine r0-r15 encoding limit.  General values start in r16 and
          * fall back to the low bank only after r16-r63 is occupied. This
          * includes MERGE/COLLECT pseudo-definitions: loop phis must not
          * occupy the entire constrained coordinate/result bank. */
         const unsigned maximum = dest_constraint ? dest_constraint->max_index
                                                   : APPLE9_LAST_ALLOCATABLE_GPR;
         range_first[0] = APPLE9_FIRST_GENERAL_GPR;
         range_last[0] = MIN2(maximum, APPLE9_LAST_ALLOCATABLE_GPR);
         range_first[1] = APPLE9_FIRST_ALLOCATABLE_GPR;
         range_last[1] = MIN2(maximum, APPLE9_FIRST_GENERAL_GPR - 1);
         range_count = 2;
      } else if (dest_constraint != NULL) {
         range_last[0] = MIN2((unsigned)dest_constraint->max_index,
                              APPLE9_LAST_ALLOCATABLE_GPR);
      }

      unsigned value_maximum = APPLE9_LAST_ALLOCATABLE_GPR;
      for (unsigned c = 0; c < components; ++c) {
         const unsigned maximum = program->max_phys[instruction->dest + c];
         if (maximum != AGX_APPLE9_PHYS_INVALID) {
            if (maximum < c)
               value_maximum = 0;
            else
               value_maximum = MIN2(value_maximum, maximum - c);
         }
      }
      for (unsigned range = 0; range < range_count; ++range)
         range_last[range] = MIN2(range_last[range], value_maximum);

      selected = AGX_APPLE9_PHYS_INVALID;
      for (unsigned range = 0;
           range < range_count && selected == AGX_APPLE9_PHYS_INVALID;
           ++range) {
         if (range_first[range] > range_last[range])
            continue;
         for (unsigned base = range_first[range]; base <= range_last[range];
              ++base) {
            if (base + components - 1 > APPLE9_LAST_ALLOCATABLE_GPR &&
                !has_fixed)
               break;
            if (base + components > AGX_APPLE9_GPR_COUNT ||
                (instruction->op != AGX_APPLE9_VIR_COLLECT &&
                 instruction->op != AGX_APPLE9_VIR_MERGE &&
                 instruction->op != AGX_APPLE9_VIR_DEVICE_ATOMIC &&
                 !agx_apple9_encoding_accepts_gpr(
                    instruction->encoding, AGX_APPLE9_OPERAND_DEST, base, 32)))
               continue;

            bool compatible = true;
            for (unsigned c = 0; c < components && compatible; ++c) {
               const unsigned gpr = base + c;
               const unsigned fixed =
                  program->fixed_phys[instruction->dest + c];
               if (fixed != AGX_APPLE9_PHYS_INVALID && fixed != gpr) {
                  compatible = false;
                  break;
               }
               if (owner[gpr] == -1 || (owner[gpr] == -2 && fixed == gpr))
                  continue;

               if (coalesced_collect) {
                  compatible &= owner[gpr] == (int32_t)instruction->src[c] &&
                                last_use[instruction->src[c]] == i;
                  continue;
               }

               if (!has_fixed || instruction->op == AGX_APPLE9_VIR_COLLECT) {
                  compatible = false;
                  break;
               }

               bool destructive_kill = false;
               for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
                  const uint32_t source = instruction->src[s];
                  destructive_kill |=
                     owner[gpr] == (int32_t)source && last_use[source] == i;
               }
               compatible &= destructive_kill;
            }
            if (compatible) {
               selected = base;
               break;
            }
         }
      }

      if (selected == AGX_APPLE9_PHYS_INVALID) {
         if (getenv("AGX_APPLE9_TRACE") != NULL) {
            fprintf(
               stderr,
               "APPLE9_ALLOC_FAIL i=%u op=%u dest=v%u width=%u fixed=%u live=%u peak=%u owners=",
               i, instruction->op, instruction->dest, components,
               program->fixed_phys[instruction->dest], live, peak);
            for (unsigned gpr = 0; gpr <= APPLE9_LAST_ALLOCATABLE_GPR; ++gpr)
               fprintf(stderr, "%s%d", gpr ? "," : "", owner[gpr]);
            fputc('\n', stderr);
            for (unsigned j = 0; j < program->instruction_count; ++j) {
               const struct agx_apple9_vir_instr *vir =
                  &program->instructions[j];
               fprintf(stderr, "APPLE9_ALLOC_VIR i=%u op=%u enc=%u dest=", j,
                       vir->op, vir->encoding);
               if (vir->dest == AGX_APPLE9_VREG_INVALID)
                  fputs("-", stderr);
               else
                  fprintf(stderr, "v%u fixed=%u", vir->dest,
                          program->fixed_phys[vir->dest]);
               if (vir->op == AGX_APPLE9_VIR_MASKED_COPY)
                  fprintf(stderr, " target=v%u", vir->target);
               fputs(" src=", stderr);
               for (unsigned s = 0; s < vir->nr_srcs; ++s)
                  fprintf(stderr, "%sv%u", s ? "," : "", vir->src[s]);
               fprintf(stderr, " imm=%#x\n", vir->immediate);
            }
         }
         free(last_use);
         free(defined);
         free(seen_definition);
         free(used);
         free(phys);
         if (reason != NULL)
            *reason =
               components == 1
                  ? (dest_constraint != NULL &&
                           dest_constraint->max_index < APPLE9_FIRST_GENERAL_GPR
                        ? "Apple9 no-spill allocator exhausted the compact destination bank"
                        : "Apple9 no-spill allocator exhausted r2-r63")
                  : "Apple9 no-spill allocator could not place an adjacent GPR tuple";
         return false;
      }

      unsigned destructive_replacements = 0;
      for (unsigned c = 0; c < components; ++c) {
         const int32_t previous_owner = owner[selected + c];
         if (previous_owner >= 0) {
            bool killed_here = false;
            for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
               const uint32_t source = instruction->src[s];
               killed_here |=
                  previous_owner == (int32_t)source && last_use[source] == i;
            }
            assert(killed_here);
            destructive_replacements++;
         }
         phys[instruction->dest + c] = selected + c;
         owner[selected + c] = instruction->dest + c;
      }
      /* A constrained in-place instruction can define its result in the same
       * fixed register as a source whose last use is this instruction.  That
       * replaces one live value rather than increasing pressure. */
      live += components - destructive_replacements;
      if (live > peak)
         peak = live;
      if (selected + components - 1 > max_phys)
         max_phys = selected + components - 1;

      /*
       * Allocate before releasing killed operands.  The operand matrix proves
       * their fields, but destructive destination/source overlap is not yet
       * validated for every Apple9 form.  Keep this first allocator strictly
       * non-destructive until each overlap is independently exercised.
       */
      for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
         uint32_t value = instruction->src[s];
         bool duplicate = false;
         for (unsigned earlier = 0; earlier < s; ++earlier)
            duplicate |= instruction->src[earlier] == value;
         if (last_use[value] == i && !duplicate) {
            if (owner[phys[value]] == (int32_t)value) {
               owner[phys[value]] = (program->reserved_gprs[phys[value]] ||
                                     fixed_gpr[phys[value]])
                                       ? -2
                                       : -1;
               assert(live > 0);
               --live;
            }
         }
      }

      /* A definition with no later SSA use dies at its defining instruction.
       * This is common for masked writes into a pinned merge register, but is
       * equally true for ordinary dead scalar results. The old tuple-only
       * cleanup leaked every such scalar allocation to the end of the
       * program and made repeated fixed-register writes appear to conflict. */
      for (unsigned c = 0; c < components; ++c) {
         const uint32_t value = instruction->dest + c;
         if (last_use[value] == i && owner[phys[value]] == (int32_t)value) {
            owner[phys[value]] =
               (program->reserved_gprs[phys[value]] || fixed_gpr[phys[value]])
                  ? -2
                  : -1;
            assert(live > 0);
            --live;
         }
      }
   }

   free(program->phys);
   program->phys = phys;
   program->peak_live_gprs = peak;
   program->max_phys_gpr = max_phys;
   free(last_use);
   free(defined);
   free(seen_definition);
   free(used);
   return agx_apple9_validate_vir_allocation(program, reason);
}

static const struct agx_apple9_vir_instr *
apple9_vir_producer_instruction(const struct agx_apple9_vir_program *program,
                                uint32_t value)
{
   if (value >= program->value_count)
      return NULL;

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &program->instructions[i];
      const unsigned components = apple9_vir_dest_components(instruction);
      if (components == 0)
         continue;
      if (value >= instruction->dest && value - instruction->dest < components)
         return instruction;
   }

   return NULL;
}

static bool
apple9_vir_is_pending_load(enum agx_apple9_vir_opcode op)
{
   return op == AGX_APPLE9_VIR_DEVICE_LOAD || op == AGX_APPLE9_VIR_TILE_LOAD ||
          op == AGX_APPLE9_VIR_TEXTURE_SAMPLE;
}

static bool
apple9_vir_is_load_token(const struct agx_apple9_vir_program *program,
                         uint32_t value, uint16_t raw_token)
{
   const struct agx_apple9_vir_instr *producer =
      apple9_vir_producer_instruction(program, value);
   return producer != NULL && producer->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
          producer->device_load_raw_token == raw_token;
}

static enum agx_apple9_dependency_layout
apple9_instruction_dependency_layout(
   const struct agx_apple9_vir_instr *instruction)
{
   if (instruction->encoding == AGX_APPLE9_ENC_PSEUDO)
      return AGX_APPLE9_DEPENDENCY_NONE;

   return agx_apple9_encoding_info(instruction->encoding)->dependency_layout;
}

static bool
apple9_instruction_accepts_scoreboard_slot(
   const struct agx_apple9_vir_instr *instruction)
{
   return apple9_instruction_dependency_layout(instruction) !=
          AGX_APPLE9_DEPENDENCY_NONE;
}

static bool
apple9_instruction_accepts_specific_scoreboard_slot(
   const struct agx_apple9_vir_instr *instruction, uint8_t slot)
{
   return apple9_instruction_accepts_scoreboard_slot(instruction) &&
          slot >= AGX_APPLE9_SCOREBOARD_SLOT_1 &&
          slot <= AGX_APPLE9_SCOREBOARD_SLOT_6;
}

static unsigned
apple9_first_consumer(const struct agx_apple9_vir_program *program,
                      unsigned producer_index);

static bool
apple9_materialize_pending_result_at(struct agx_apple9_vir_program *program,
                                     unsigned producer_index,
                                     unsigned insertion, const char **reason)
{
   const unsigned old_count = program->instruction_count;
   const struct agx_apple9_vir_instr *producer =
      &program->instructions[producer_index];
   const bool atomic = producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC;
   if (!apple9_vir_is_pending_load(producer->op) &&
       !(atomic && !producer->atomic_discard)) {
      if (reason != NULL)
         *reason =
            "Apple9 scoreboard materialization requires a pending producer";
      return false;
   }

   const uint32_t pending = producer->dest;
   const unsigned components =
      apple9_vir_dest_components(&program->instructions[producer_index]);
   const unsigned earliest_insertion = producer_index + (atomic ? 2 : 1);
   if (atomic &&
       (producer_index + 1 >= old_count ||
        program->instructions[producer_index + 1].op !=
           AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)) {
      if (reason != NULL)
         *reason = "Apple9 atomic publication record is not adjacent";
      return false;
   }
   if (insertion < earliest_insertion || insertion > old_count) {
      if (reason != NULL)
         *reason = "Apple9 pending-result materialization is out of order";
      return false;
   }

   uint32_t materialized[4];
   struct agx_apple9_vir_instr materialize_instructions[4];
   /* Consume the pending result through the general integer-logic form.
    *
    * The old umin(x, x) bridge was functionally correct, but its compact
    * destination is limited to r0-r15.  Under genuine scoreboard pressure
    * those registers can all remain live while r16-r63 are available, which
    * made materialization itself an artificial allocator bottleneck.  The
    * hardware-validated extended IOR form consumes the same pending slot,
    * implements the identical x | x bit-copy, and has a general destination.
   */
   for (unsigned c = 0; c < components; ++c) {
      uint32_t sources[] = {pending + c, pending + c};
      materialized[c] =
         agx_apple9_vir_emit(program, AGX_APPLE9_VIR_IOR,
                             AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
      if (materialized[c] == AGX_APPLE9_VREG_INVALID) {
         if (reason != NULL)
            *reason = "out of memory materializing an Apple9 pending result";
         return false;
      }
      materialize_instructions[c] = program->instructions[old_count + c];
   }

   materialize_instructions[0].scoreboard_materialize = true;

   memmove(&program->instructions[insertion + components],
           &program->instructions[insertion],
           (old_count - insertion) * sizeof(*program->instructions));
   memcpy(&program->instructions[insertion], materialize_instructions,
          components * sizeof(*program->instructions));

   /* Branch targets name VIR instruction boundaries. Inserting the return
    * handoff must preserve both loop backedges and forward loop exits. */
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *jump = &program->instructions[i];
      if ((jump->op == AGX_APPLE9_VIR_JMP_EXEC_ANY ||
           jump->op == AGX_APPLE9_VIR_JMP_EXEC_NONE) &&
          jump->branch_target != AGX_APPLE9_VREG_INVALID &&
          jump->branch_target >= insertion)
         jump->branch_target += components;
   }

   for (unsigned i = insertion + components; i < program->instruction_count;
        ++i) {
      struct agx_apple9_vir_instr *instruction = &program->instructions[i];
      for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
         for (unsigned c = 0; c < components; ++c) {
            if (instruction->src[s] == pending + c)
               instruction->src[s] = materialized[c];
         }
      }
      if (instruction->op == AGX_APPLE9_VIR_DEVICE_STORE)
         instruction->encoding = AGX_APPLE9_ENC_DEVICE_STORE;
   }
   for (unsigned c = 0; c < components; ++c) {
      if (program->output == pending + c)
         program->output = materialized[c];
   }
   for (unsigned i = 0; i < program->live_out_count; ++i) {
      for (unsigned c = 0; c < components; ++c) {
         if (program->live_out[i] == pending + c)
            program->live_out[i] = materialized[c];
      }
   }

   return true;
}

static bool
apple9_materialize_load(struct agx_apple9_vir_program *program,
                        unsigned producer_index, const char **reason)
{
   return apple9_materialize_pending_result_at(program, producer_index,
                                               producer_index + 1, reason);
}

static bool
apple9_materialize_unsupported_loads(struct agx_apple9_vir_program *program,
                                     const char **reason)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *producer = &program->instructions[i];
      if (producer->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE) {
         /* Separate the sample's fixed first-result handoff from subsequent
          * arithmetic and memory traffic, using ordinary SSA bit copies. */
         if (!apple9_materialize_load(program, i, reason))
            return false;
         i += 4;
         continue;
      }
      if (!apple9_vir_is_pending_load(producer->op) ||
          producer->producer_scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_AUTO)
         continue;

      const unsigned handoff = apple9_first_consumer(program, i);
      const unsigned components = apple9_vir_dest_components(producer);
      bool live_out = program->output != AGX_APPLE9_VREG_INVALID &&
                      program->output >= producer->dest &&
                      program->output - producer->dest < components;
      for (unsigned l = 0; l < program->live_out_count; ++l)
         live_out |= program->live_out[l] >= producer->dest &&
                     program->live_out[l] - producer->dest < components;

      if (handoff == UINT_MAX && !live_out) {
         for (unsigned c = 0; c < components; ++c)
            program->fixed_phys[producer->dest + c] = AGX_APPLE9_PHYS_INVALID;
         memmove(&program->instructions[i], &program->instructions[i + 1],
                 (program->instruction_count - i - 1) *
                    sizeof(*program->instructions));
         --program->instruction_count;
         for (unsigned j = 0; j < program->instruction_count; ++j) {
            struct agx_apple9_vir_instr *jump = &program->instructions[j];
            if ((jump->op == AGX_APPLE9_VIR_JMP_EXEC_ANY ||
                 jump->op == AGX_APPLE9_VIR_JMP_EXEC_NONE) &&
                jump->branch_target != AGX_APPLE9_VREG_INVALID &&
                jump->branch_target > i)
               --jump->branch_target;
         }
         --i;
         continue;
      }

      if (handoff != UINT_MAX && apple9_instruction_accepts_scoreboard_slot(
                                    &program->instructions[handoff]))
         continue;

      if (!apple9_materialize_load(program, i, reason))
         return false;

      /* The identity bridge occupies the next position. */
      i += 1;
   }

   return true;
}

static unsigned
apple9_first_consumer(const struct agx_apple9_vir_program *program,
                      unsigned producer_index)
{
   const struct agx_apple9_vir_instr *producer =
      &program->instructions[producer_index];
   const uint32_t first = producer->dest;
   const unsigned components = apple9_vir_dest_components(producer);
   for (unsigned i = producer_index + 1; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &program->instructions[i];

      /* This adjacent record only routes an atomic return into its named GPR.
       * It does not wait for or release the producer's scoreboard slot. */
      if (producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
          instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT &&
          instruction->nr_srcs == 1 && instruction->src[0] == first)
         continue;

      for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
         if (instruction->src[s] >= first &&
             instruction->src[s] - first < components)
            return i;
      }
   }

   return UINT_MAX;
}

/* Scoreboard legalization may replace a native vector load tuple with one
 * scalar materialization per lane.  Re-establish the ordinary vector-source
 * invariant after that rewrite. */
static bool
apple9_collect_vector_store_sources(struct agx_apple9_vir_program *program,
                                    const char **reason)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *store = &program->instructions[i];
      const unsigned components = store->memory_components;
      if (store->op != AGX_APPLE9_VIR_DEVICE_STORE || components < 2 ||
          apple9_vir_values_form_tuple(program, store->src, components))
         continue;

      uint32_t sources[4];
      for (unsigned c = 0; c < components; ++c)
         sources[c] = store->src[c];

      const unsigned old_count = program->instruction_count;
      const uint32_t tuple =
         agx_apple9_vir_emit_collect(program, sources, components);
      if (tuple == AGX_APPLE9_VREG_INVALID) {
         if (reason != NULL)
            *reason = "out of memory collecting Apple9 vector-store sources";
         return false;
      }

      const struct agx_apple9_vir_instr collect =
         program->instructions[old_count];
      memmove(&program->instructions[i + 1], &program->instructions[i],
              (old_count - i) * sizeof(*program->instructions));
      program->instructions[i] = collect;
      store = &program->instructions[i + 1];
      for (unsigned c = 0; c < components; ++c)
         store->src[c] = tuple + c;

      ++i;
   }

   return true;
}

static bool
apple9_materialize_scoreboard_pressure(struct agx_apple9_vir_program *program,
                                       const char **reason)
{
   for (;;) {
      bool changed = false;

      for (unsigned i = 0; i < program->instruction_count; ++i) {
         const struct agx_apple9_vir_instr *current = &program->instructions[i];
         const bool current_pending =
            (apple9_vir_is_pending_load(current->op) ||
             (current->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
              !current->atomic_discard)) &&
            current->producer_scoreboard_slot ==
               AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         if (!current_pending)
            continue;
         const uint32_t pressure_value = current->dest;

         unsigned pending_handoff[AGX_APPLE9_SCOREBOARD_SLOT_6 + 1];
         unsigned pending_count = 0;
         for (unsigned p = 0; p <= i; ++p) {
            const struct agx_apple9_vir_instr *producer =
               &program->instructions[p];
            const bool pending =
               (apple9_vir_is_pending_load(producer->op) ||
                (producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
                 !producer->atomic_discard)) &&
               producer->producer_scoreboard_slot ==
                  AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
            if (!pending)
               continue;

            unsigned handoff = apple9_first_consumer(program, p);
            if (handoff <= i || handoff == UINT_MAX)
               continue;

            bool duplicate = false;
            for (unsigned h = 0; h < pending_count; ++h)
               duplicate |= pending_handoff[h] == handoff;
            if (!duplicate)
               pending_handoff[pending_count++] = handoff;
            if (pending_count > AGX_APPLE9_SCOREBOARD_SLOT_6)
               break;
         }

         if (pending_count <= AGX_APPLE9_SCOREBOARD_SLOT_6)
            continue;

         /* Materialize every producer in the oldest pending multi-source
          * group immediately before the instruction that would exceed the
          * six-slot budget. Delaying the handoff preserves maximum latency
          * between an atomic publication and reuse of its slot while leaving
          * an ordinary GPR value for the original consumer. */
         const unsigned target_handoff = pending_handoff[0];
         struct util_dynarray values = UTIL_DYNARRAY_INIT;
         for (unsigned p = 0; p <= i; ++p) {
            const struct agx_apple9_vir_instr *producer =
               &program->instructions[p];
            const bool pending =
               (apple9_vir_is_pending_load(producer->op) ||
                (producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
                 !producer->atomic_discard)) &&
               producer->producer_scoreboard_slot ==
                  AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
            if (pending &&
                apple9_first_consumer(program, p) == target_handoff)
               util_dynarray_append(&values, producer->dest);
         }

         util_dynarray_foreach(&values, uint32_t, value) {
            unsigned producer = UINT_MAX;
            unsigned insertion = UINT_MAX;
            for (unsigned p = 0; p < program->instruction_count; ++p) {
               const struct agx_apple9_vir_instr *candidate =
                  &program->instructions[p];
               if ((apple9_vir_is_pending_load(candidate->op) ||
                    (candidate->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
                     !candidate->atomic_discard)) &&
                   program->instructions[p].dest == *value) {
                  producer = p;
               }
               if ((apple9_vir_is_pending_load(candidate->op) ||
                    (candidate->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
                     !candidate->atomic_discard)) &&
                   candidate->dest == pressure_value)
                  insertion = p;
            }
            if (producer == UINT_MAX || insertion == UINT_MAX ||
                !apple9_materialize_pending_result_at(
                   program, producer, insertion, reason)) {
               util_dynarray_fini(&values);
               if (reason != NULL && *reason == NULL)
                  *reason = "could not materialize Apple9 scoreboard pressure";
               return false;
            }
         }
         util_dynarray_fini(&values);
         changed = true;
         break;
      }

      if (!changed)
         return true;
   }
}

static bool
apple9_is_logic_handoff_source(const struct agx_apple9_vir_program *program,
                               unsigned consumer_index, uint32_t source)
{
   const struct agx_apple9_vir_instr *producer =
      apple9_vir_producer_instruction(program, source);
   return producer != NULL &&
          (apple9_vir_is_pending_load(producer->op) ||
           (producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
            !producer->atomic_discard)) &&
          apple9_first_consumer(program,
                                (unsigned)(producer - program->instructions)) ==
             consumer_index;
}

static void
apple9_normalize_logic_handoff_source(struct agx_apple9_vir_program *program,
                                      unsigned consumer_index)
{
   struct agx_apple9_vir_instr *consumer =
      &program->instructions[consumer_index];
   if (consumer->encoding != AGX_APPLE9_ENC_LOGIC_EXTENDED ||
       consumer->nr_srcs != 2 ||
       (consumer->op != AGX_APPLE9_VIR_IAND &&
        consumer->op != AGX_APPLE9_VIR_IOR &&
        consumer->op != AGX_APPLE9_VIR_IXOR))
      return;

   const bool source_a_pending =
      apple9_is_logic_handoff_source(program, consumer_index, consumer->src[0]);
   const bool source_b_pending =
      apple9_is_logic_handoff_source(program, consumer_index, consumer->src[1]);

   /* Native Apple9 ilogic uses its one-hot pending-result mask for source A.
    * Source B is an ordinary GPR unless both operands belong to the same
    * pending group.  AND/OR/XOR are commutative, so normalize a lone pending
    * source into the architectural role before allocation derives source
    * liveness.  This is required when a pending load reuses a register whose
    * old durable value is still present: leaving it in source B silently reads
    * that stale GPR value instead of the pending result. */
   if (!source_a_pending && source_b_pending) {
      const uint32_t temporary = consumer->src[0];
      consumer->src[0] = consumer->src[1];
      consumer->src[1] = temporary;
   }
}

bool
agx_apple9_assign_vir_scoreboard_slots(struct agx_apple9_vir_program *program,
                                       const char **reason)
{
   static const uint8_t scalar_load_preference[] = {6, 1, 2, 3, 4, 5};
   if (reason != NULL)
      *reason = NULL;
   if (program == NULL)
      return false;

   if (!apple9_materialize_unsupported_loads(program, reason))
      return false;

   if (!apple9_materialize_scoreboard_pressure(program, reason))
      return false;

   uint8_t *handoff_slot =
      calloc(program->instruction_count, sizeof(*handoff_slot));
   bool occupied[AGX_APPLE9_SCOREBOARD_SLOT_6 + 1] = {false};
   if (handoff_slot == NULL && program->instruction_count != 0) {
      if (reason != NULL)
         *reason = "out of memory allocating Apple9 scoreboard slots";
      return false;
   }

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *producer = &program->instructions[i];

      /* Consume an input handoff before allocating a result produced by the
       * same instruction. EXP-M4-51 proves that a returning atomic may reuse
       * its pending input's slot, including under full six-slot pressure. */
      if (handoff_slot[i] != AGX_APPLE9_SCOREBOARD_SLOT_NONE) {
         struct agx_apple9_vir_instr *consumer = &program->instructions[i];
         const uint8_t slot = handoff_slot[i];
         if (consumer->scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
             consumer->scoreboard_slot != slot) {
            if (reason != NULL)
               *reason =
                  "Apple9 consumer conflicts with allocated scoreboard slot";
            goto fail;
         }
         consumer->scoreboard_slot = slot;
         apple9_normalize_logic_handoff_source(program, i);
         occupied[slot] = false;
      }

      const bool is_load = apple9_vir_is_pending_load(producer->op);
      const bool is_returning_atomic =
         producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
         !producer->atomic_discard;
      if (is_load || is_returning_atomic) {
         const bool automatic =
            producer->producer_scoreboard_slot ==
            AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         const unsigned handoff = apple9_first_consumer(program, i);

         if (handoff == UINT_MAX) {
            if (automatic || is_returning_atomic) {
               if (reason != NULL)
                  *reason =
                     "Apple9 pending result has no scoreboard-capable handoff";
               goto fail;
            }
         } else if (!apple9_instruction_accepts_scoreboard_slot(
                       &program->instructions[handoff])) {
            if (automatic || is_returning_atomic) {
               if (reason != NULL)
                  *reason =
                     "Apple9 pending result requires an unsupported consumer form";
               goto fail;
            }
         } else {
            uint8_t slot = handoff_slot[handoff];
            if (slot == AGX_APPLE9_SCOREBOARD_SLOT_NONE) {
               if (automatic) {
                  for (unsigned p = 0; p < sizeof(scalar_load_preference) /
                                              sizeof(scalar_load_preference[0]);
                       ++p) {
                     const uint8_t candidate = scalar_load_preference[p];
                     if (!occupied[candidate] &&
                         apple9_instruction_accepts_specific_scoreboard_slot(
                            &program->instructions[handoff], candidate)) {
                        slot = candidate;
                        break;
                     }
                  }
               } else {
                  slot = producer->producer_scoreboard_slot;
               }

               if (!apple9_instruction_accepts_specific_scoreboard_slot(
                      &program->instructions[handoff], slot) ||
                   occupied[slot]) {
                  if (reason != NULL)
                     *reason = "Apple9 scoreboard has no compatible free slot";
                  goto fail;
               }
               occupied[slot] = true;
               handoff_slot[handoff] = slot;
            } else if (!automatic &&
                       producer->producer_scoreboard_slot != slot) {
               if (reason != NULL)
                  *reason =
                     "Apple9 multi-source scoreboard group uses different slots";
               goto fail;
            }

            if (automatic && producer->op == AGX_APPLE9_VIR_TILE_LOAD) {
               producer->producer_scoreboard_slot = slot;
            } else if (automatic && is_load) {
               uint16_t token;
               if (!apple9_scalar_load_token_for_slot(slot, &token)) {
                  if (reason != NULL)
                     *reason = "Apple9 scalar-load slot has no producer tag";
                  goto fail;
               }
               producer->producer_scoreboard_slot = slot;
               producer->device_load_raw_token = token;
            } else if (automatic) {
               producer->producer_scoreboard_slot = slot;
               assert(i + 1 < program->instruction_count);
               struct agx_apple9_vir_instr *publication =
                  &program->instructions[i + 1];
               assert(publication->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
               publication->producer_scoreboard_slot = slot;
            }
         }
      }

   }

   free(handoff_slot);
   return apple9_collect_vector_store_sources(program, reason);

fail:
   free(handoff_slot);
   return false;
}

static void
packed_init(struct agx_apple9_packed_instruction *packed, const uint8_t *bytes,
            unsigned length)
{
   memset(packed, 0, sizeof(*packed));
   memcpy(packed->bytes, bytes, length);
   packed->length = length;
}

bool
agx_apple9_pack_get_sr(unsigned dst, uint8_t selector, uint8_t datapath,
                       struct agx_apple9_packed_instruction *packed)
{
   if (dst >= AGX_APPLE9_GPR_COUNT)
      return false;
   uint8_t bytes[4] = {
      ((dst & 0xf) << 4) | 0x0c,
      selector,
      dst >= 64 ? datapath | 0x40 : datapath,
      0x06 | ((dst >> 4) << 5),
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_get_sr_zext16(unsigned dst, uint8_t selector,
                              struct agx_apple9_packed_instruction *packed)
{
   if (dst >= 16)
      return false;

   /*
    * EXP-M4-29's local-position, local-index, SIMD-lane, SIMD-group and
    * total-thread-count controls all use this pair.  The form-0 GET_SR
    * publishes a narrow result, and the immediately following X3 companion
    * zero-extends that result in place.  Selector-only hardware recodes
    * establish that the pair is shared by the complete scalar16 class.
    */
   const uint8_t bytes[8] = {
      (dst << 4) | 0x04, selector, 0x10, 0x06,
      (dst << 4) | 0x03, 0x00,     0x00, 0x01,
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_get_global_id(unsigned dst, unsigned component,
                              struct agx_apple9_packed_instruction *packed)
{
   if (component >= 3)
      return false;

   /* EXP-0092 and the Apple9 SR isolation corpus independently identify
    * thread_position_in_grid.{x,y,z} as selectors 0xa0/0xa1/0xa2. */
   return agx_apple9_pack_get_sr(dst, 0xa0 + component, 0x10, packed);
}

static bool
pack_i2f32(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
           struct agx_apple9_packed_instruction *packed)
{
   const unsigned dst = phys[instruction->dest];
   const unsigned src = phys[instruction->src[0]];
   if (instruction->nr_srcs != 1 || dst >= 64 || src >= 64 ||
       (instruction->op != AGX_APPLE9_VIR_U2F32 &&
        instruction->op != AGX_APPLE9_VIR_I2F32))
      return false;
   const bool retain_source = instruction->live_after_mask & 1u;

   /*
    * EXP-0013 and EXP-0144 hardware-validate the canonical 32-bit integer to
    * float form, including byte 7 bit 6 as the signed-source selector.
    *
    * Earlier code treated byte 1's high nibble as source lifetime because
    * Metal's retained-source cases used 0x17. The shared dependency model
    * identifies those bits as the one-hot slot field instead. Source lifetime
    * is carried independently by byte 6 (0x8c retained, 0xac last use).
    */
   const uint8_t bytes[] = {
      0xa7,
      0x07,
      0x54,
      dst << 1,
      0x03,
      src << 2,
      retain_source ? 0x8c : 0xac,
      instruction->op == AGX_APPLE9_VIR_I2F32 ? 0x60 : 0x20,
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_f2i32(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
           struct agx_apple9_packed_instruction *packed)
{
   const unsigned dst = phys[instruction->dest];
   const unsigned src = phys[instruction->src[0]];
   if (instruction->nr_srcs != 1 || dst >= 64 || src >= 64 ||
       (instruction->op != AGX_APPLE9_VIR_F2I32 &&
        instruction->op != AGX_APPLE9_VIR_F2U32))
      return false;

   /* EXP-0013 proves truncation toward zero and the signed/unsigned selector;
    * EXP-0144 independently locates dst=byte3>>1 and src=byte5>>2. */
   const uint8_t bytes[] = {
      0x27, 0x07,
      0x54, dst << 1,
      0x03, src << 2,
      0xb4, instruction->op == AGX_APPLE9_VIR_F2I32 ? 0x48 : 0x08,
      0x03, 0x00,
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_fspecial(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
              struct agx_apple9_packed_instruction *packed)
{
   const unsigned dst = phys[instruction->dest];
   const unsigned src = phys[instruction->src[0]];
   if (instruction->encoding != AGX_APPLE9_ENC_FLOAT_SPECIAL ||
       instruction->nr_srcs != 1 || dst >= AGX_APPLE9_GPR_COUNT || src >= 64 ||
       (instruction->immediate != 0x02 && instruction->immediate != 0x03))
      return false;

   /* The function class and datapath jointly select the operation. In
    * particular, class 0 on the ordinary SFU datapath is not reciprocal.
    * Source lifetime and pending dependencies are independent of selection.
    * EXP-M4-41/42 establish reciprocal and the mask; EXP-0026/0165/0237
    * document the other datapaths and direct-round selector family. */
   unsigned family = 0, function = 0;
   unsigned precision = 0x40, rounding = 0;
   switch (instruction->op) {
   case AGX_APPLE9_VIR_FRCP:
      family = 1;
      precision = 0x48;
      rounding = 0x20;
      break;
   case AGX_APPLE9_VIR_FRSQ:
      family = 1;
      function = 1;
      break;
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
      function = 1;
      break;
   case AGX_APPLE9_VIR_FSIN_FACTOR:
      function = 3;
      break;
   case AGX_APPLE9_VIR_FEXP2:
      family = 1;
      function = 2;
      break;
   case AGX_APPLE9_VIR_FLOG2:
      function = 2;
      break;
   case AGX_APPLE9_VIR_FFLOOR:
      rounding = 2;
      break;
   case AGX_APPLE9_VIR_FCEIL:
      rounding = 4;
      break;
   case AGX_APPLE9_VIR_FTRUNC:
      rounding = 6;
      break;
   case AGX_APPLE9_VIR_FROUND_EVEN:
      break;
   default:
      return false;
   }
   /* EXP-M4-54 T8132 matched Metal and destructive source-reuse tests:
    * ordinary SFU uses bit 5 for source release (0x90 -> 0xb0), whereas
    * reciprocal's distinct datapath uses bit 4 (0x00 -> 0x10).
    * Ordinary SFU bits [4:3] select BF16/FP16/FP32 as 0/1/2. Clearing bit 4
    * changes FP32 to BF16, reading the low 16 bits (often zero), not retaining
    * FP32. Reciprocal instead places source type in byte 7 bits [3:2].
    * Both forms consume pending producers through the independent mask. */
   const uint8_t bytes[10] = {
      0x2f | (family << 7), function, 0x54, dst << 1,
      instruction->immediate, src << 2,
      instruction->op == AGX_APPLE9_VIR_FRCP
         ? ((instruction->live_after_mask & 1u) ? 0x00 : 0x10)
         : ((instruction->live_after_mask & 1u) ? 0x90 : 0xb0),
      precision, rounding, 0,
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_ishr_imm(const struct agx_apple9_vir_instr *instruction,
              const uint8_t *phys, struct agx_apple9_packed_instruction *packed)
{
   const unsigned dst = phys[instruction->dest];
   const unsigned src = phys[instruction->src[0]];
   const unsigned amount = instruction->immediate;
   if (instruction->op != AGX_APPLE9_VIR_ISHR || instruction->nr_srcs != 1 ||
       dst >= 16 || src >= 16 || amount >= 32)
      return false;

   /* T8132 EXP-0139 validates arithmetic semantics and every amount 0..31.
    * The own-source corpus independently isolates byte3 as dst<<1, byte5 as
    * src<<2, and byte6 as amount<<2. Pending dependencies share the integer
    * one-hot field at bits 12..17. */
   const uint8_t bytes[] = {
      0xa7, 0x01, 0x54, dst << 1, 0x02, src << 2, amount << 2, 0x78, 0x62, 0x00,
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_mov_imm(unsigned dst, unsigned value,
                        struct agx_apple9_packed_instruction *packed)
{
   /* EXP-0128/EXP-0140 prove that bit 7 is not an immediate bit: values
    * 0x80..0xff do not write the destination. */
   if (dst >= 16 || value > 0x7f)
      return false;
   const uint8_t bytes[2] = {(dst << 4) | 0x0c, value};
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_mov_imm32(unsigned dst, uint32_t value,
                          struct agx_apple9_packed_instruction *packed)
{
   if (dst >= 64)
      return false;

   /* EXP-M4-37: native mode 2 is the eight-byte scalar raw-literal form.
    * Thirty-two own-source one-bit differences recover every payload bit,
    * and all 33 controls execute exactly on T8132.  The destination is six
    * bits: byte 0 carries bits 0..3 and byte 2 bits 6..7 carry bits 4..5.
    * Byte 2 bit 5 is a separate native modifier, not a destination bit. */
   const uint8_t bytes[8] = {
      ((dst & 0xf) << 4) | 0x0c, 0x80 | (value & 0x7f),
      ((dst >> 4) << 6) | 0x02,  (value >> 24) & 0xfe,
      (value >> 6) & 0x1e,       (value >> 9) & 0x0c,
      (value >> 13) & 0xff,      (value >> 21) & 0x0f,
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_mov(unsigned dst, unsigned src,
                    struct agx_apple9_packed_instruction *packed)
{
   if (dst >= 16 || src >= 64)
      return false;
   const uint8_t bytes[10] = {
      (dst << 4) | 0x0b,
      (src << 1) | 1,
      0x1f,
      0x01,
      0x02,
      0x00,
      0x00,
      0x80,
      0x00,
      0x00,
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_device_load_u32(unsigned dst, unsigned index, unsigned binding,
                                uint8_t flags,
                                enum agx_apple9_scoreboard_slot scoreboard_slot,
                                struct agx_apple9_packed_instruction *packed)
{
   uint16_t raw_token;
   if (!apple9_scalar_load_token_for_slot(scoreboard_slot, &raw_token))
      return false;

   return agx_apple9_pack_device_load_u32_raw(
      dst, index, binding, AGX_APPLE9_DEVICE_LOAD_INDEX_DIRECT_GPR, flags,
      false, raw_token, packed);
}

bool
agx_apple9_pack_device_load_u32_raw(
   unsigned dst, unsigned index, unsigned binding,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   bool index_first_load_consumer, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed)
{
   return agx_apple9_pack_device_load_scalar_raw(
      dst, index, binding, 32, index_kind, flags, index_first_load_consumer,
      raw_token, packed);
}

bool
agx_apple9_pack_device_load_scalar_raw(
   unsigned dst, unsigned index, unsigned binding, unsigned bits,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   bool index_first_load_consumer, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed)
{
   if (bits != 8 && bits != 16 && bits != 32)
      return false;
   if (!agx_apple9_pack_device_load_vector_u32_raw(
          dst, index, binding, 1, index_kind, flags, index_first_load_consumer,
          raw_token, packed))
      return false;

   static const uint8_t format[] = {0x21, 0x01, 0x11};
   static const uint8_t tail[] = {0x42, 0x44, 0x46};
   const unsigned size = bits == 8 ? 0 : bits == 16 ? 1 : 2;
   packed->bytes[8] = (packed->bytes[8] & 0xc0) | format[size];
   packed->bytes[12] = tail[size];
   return true;
}

bool
agx_apple9_pack_device_load_vector_u32_raw(
   unsigned dst, unsigned index, unsigned binding, unsigned components,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   bool index_first_load_consumer, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed)
{
   if (components < 1 || components > 4 || dst + components > 64 ||
       binding > UINT8_MAX ||
       (flags & ~(AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX |
                  AGX_APPLE9_DEVICE_LOAD_HAS_NEXT)) ||
       !apple9_device_load_raw_token_valid(raw_token))
      return false;

   uint8_t encoded_index;
   switch (index_kind) {
   case AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR:
      if (index >= AGX_APPLE9_GPR_COUNT)
         return false;
      encoded_index = index;
      break;
   case AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR:
      if (index >= AGX_APPLE9_GPR_COUNT)
         return false;
      encoded_index = 0x80 | index;
      break;
   default:
      return false;
   }

   /*
    * The raw-system-index address selector, linear HAS_NEXT marker, and
    * scalar-load scoreboard tag are independently encoded. All members of a
    * vector-result tuple carry the same slot tag.
    */
   uint8_t bytes[14] = {
      0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x20,
      0x00, 0x11, 0x00, 0x00, 0x40, 0x46, 0x00,
   };
   if (flags & AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX)
      bytes[1] = 0x10;
   if (flags & AGX_APPLE9_DEVICE_LOAD_HAS_NEXT)
      bytes[2] = 0x54;
   if (index_first_load_consumer)
      bytes[2] |= 0x02;
   bytes[3] = dst << 1;
   bytes[4] = binding;
   bytes[5] = encoded_index;
   static const uint8_t width_token_bits[] = {0x00, 0x08, 0x0c, 0x06};
   static const uint8_t width_tail[] = {0x46, 0x48, 0x40, 0x40};
   bytes[8] = (raw_token >> 8) | width_token_bits[components - 1];
   bytes[9] = raw_token & 0xff;
   bytes[12] = width_tail[components - 1];
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_device_store_scalar(
   unsigned data, unsigned index, unsigned binding, unsigned bits,
   enum agx_apple9_scoreboard_slot scoreboard_slot, bool release_index,
   struct agx_apple9_packed_instruction *packed)
{
   if (data >= 64 || index >= AGX_APPLE9_GPR_COUNT || binding > UINT8_MAX ||
       (bits != 8 && bits != 16 && bits != 32) || (bits != 32 && data != 0))
      return false;
   uint8_t bytes[14] = {
      0xe7, 0x00, 0x54, 0x00, 0x00, 0x00, 0x20,
      0x00, 0x11, 0x00, 0x00, 0x90, 0x11, 0x00,
   };
   bytes[3] = data << 1;
   bytes[4] = binding;
   bytes[5] = index;
   /* Native same-index store chains use 0x20 ... 0x21. */
   bytes[6] |= release_index;
   if (bits == 8) {
      bytes[8] = 0x21;
      bytes[11] = 0x90;
      bytes[12] = 0x10;
   } else if (bits == 16) {
      bytes[8] = 0x01;
      bytes[11] = 0x10;
      bytes[12] = 0x11;
   }
   if (!apple9_pack_dependency(bytes, sizeof(bytes),
                               AGX_APPLE9_DEPENDENCY_MASK_12_17,
                               scoreboard_slot))
      return false;
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_device_store_vector_u32(
   unsigned data, unsigned index, unsigned binding, unsigned components,
   enum agx_apple9_scoreboard_slot scoreboard_slot, bool release_index,
   struct agx_apple9_packed_instruction *packed)
{
   if (components < 1 || components > 4 || data + components > 64 ||
       index >= AGX_APPLE9_GPR_COUNT || binding > UINT8_MAX)
      return false;

   static const uint8_t width_token_bits[] = {0x00, 0x08, 0x0c, 0x06};
   static const uint8_t width_tail[] = {0x11, 0x12, 0x10, 0x10};

   /* EXP-M4-27 supplies the width-2/4 stores and the width-3 load fields.
    * The symmetric width-3 store composition was then validated by the
    * exact-output T8132 vector-suite hardware gate. */

   uint8_t bytes[14] = {
      0xe7,
      0x00,
      0x54,
      (uint8_t)(data << 1),
      (uint8_t)binding,
      (uint8_t)index,
      (uint8_t)(0x20 | release_index),
      0x00,
      (uint8_t)(0x11 | width_token_bits[components - 1]),
      0x00,
      0x00,
      0x10,
      width_tail[components - 1],
      0x00,
   };
   if (!apple9_pack_dependency(bytes, sizeof(bytes),
                               AGX_APPLE9_DEPENDENCY_MASK_12_17,
                               scoreboard_slot))
      return false;
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_device_atomic(
   unsigned index, unsigned data, unsigned binding, enum agx_apple9_atomic_op op,
   bool discard_result,
   enum agx_apple9_scoreboard_slot input_dependency,
   struct agx_apple9_packed_instruction *packed)
{
   const bool dependency_valid =
      input_dependency == AGX_APPLE9_SCOREBOARD_SLOT_NONE ||
      (input_dependency >= AGX_APPLE9_SCOREBOARD_SLOT_1 &&
       input_dependency <= AGX_APPLE9_SCOREBOARD_SLOT_6);
   if (index >= AGX_APPLE9_GPR_COUNT || data >= AGX_APPLE9_GPR_COUNT ||
       binding >= 128 ||
       !apple9_atomic_op_valid(op) || !dependency_valid)
      return false;

   const unsigned data_components =
      op == AGX_APPLE9_ATOMIC_CMPXCHG ? 2 : 1;
   if (data + data_components > AGX_APPLE9_GPR_COUNT)
      return false;

   /* Device atomics independently encode an address index and an RMW data
    * register. Compare-exchange reads an adjacent [desired, compare] pair.
    * A returning atomic's destination is carried by the adjacent
    * result-publication record rather than this packet. Bits 12..17 are an
    * input scoreboard dependency, independently established by EXP-M4-50.
    * EXP-M4-47 gives
    * direct witnesses for r0 data/r4 index,
    * r2 data/r1 index, and r3:r4 data/r5 index. Our inputs are ordinary
    * materialized GPRs use the native zero-mask 0x01/0x54 form. A pending
    * scalar load instead names its allocated slot in this same field. */
   uint8_t bytes[14] = {
      0x67,
      0x01,
      0x54,
      0x00,
      0x00,
      (uint8_t)(binding | ((data & 1) << 7)),
      (uint8_t)(((data >> 1) & 0x3f) | ((index & 1) << 7)),
      (uint8_t)(0x80 | ((index >> 1) & 0x3f)),
      0x00,
      discard_result ? 0x40 : 0x02,
      0x00,
      0x00,
      (uint8_t)((op << 1) | 0x40),
      0x02,
   };
   if (!apple9_pack_dependency(bytes, sizeof(bytes),
                               AGX_APPLE9_DEPENDENCY_MASK_12_17,
                               input_dependency))
      return false;

   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_iadd(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
          struct agx_apple9_packed_instruction *packed)
{
   /* Start with the no-dependency form. Source lifetime is carried by the
    * independently established per-source fields below; bits 12..17 are
    * reserved for the shared one-hot dependency encoding. */
   uint8_t bytes[10] = {0x9f, 0x01, 0x54, 0x00, 0x02,
                        0x00, 0x00, 0xa8, 0x17, 0x05};
   if (instruction->live_after_mask & (1u << 0)) {
      set_bits(bytes, 49, 1, 1);
      set_bits(bytes, 65, 1, 0);
   }
   if (instruction->live_after_mask & (1u << 1)) {
      set_bits(bytes, 58, 1, 1);
      set_bits(bytes, 66, 1, 0);
   }
   if (instruction->op == AGX_APPLE9_VIR_ISUB)
      bytes[0] = 0x1f;
   set_bits(bytes, 25, 7, phys[instruction->dest]);
   set_bits(bytes, 42, 7, phys[instruction->src[0]]);
   set_bits(bytes, 51, 7, phys[instruction->src[1]]);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_imad(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
          struct agx_apple9_packed_instruction *packed)
{
   /*
    * As with IADD, derive the contextual register-cache lifetime envelope
    * from the all-sources-dead hardware probe.  EXP-M4-16's ring and pressure
    * forms independently identify the A/B/C keep bits and their complementary
    * descriptor bits.
    */
   uint8_t bytes[12] = {0x9f, 0x00, 0x54, 0x00, 0x02, 0x00,
                        0x00, 0x00, 0xd0, 0x2f, 0x2a, 0x00};
   if (instruction->encoding != AGX_APPLE9_ENC_INT_MAD_EXTENDED ||
       instruction->nr_srcs != 3 || instruction->immediate != 0)
      return false;
   /* Source lifetime is carried by the independently established per-source
    * fields below; bits 12..17 are reserved for dependencies. */
   if (instruction->live_after_mask & (1u << 0)) {
      set_bits(bytes, 49, 1, 1);
      set_bits(bytes, 73, 1, 0);
   }
   if (instruction->live_after_mask & (1u << 1)) {
      set_bits(bytes, 58, 1, 1);
      set_bits(bytes, 74, 1, 0);
   }
   if (instruction->live_after_mask & (1u << 2)) {
      set_bits(bytes, 67, 1, 1);
      set_bits(bytes, 75, 1, 0);
   }
   set_bits(bytes, 25, 7, phys[instruction->dest]);
   set_bits(bytes, 42, 7, phys[instruction->src[0]]);
   set_bits(bytes, 51, 7, phys[instruction->src[1]]);
   set_bits(bytes, 60, 7, phys[instruction->src[2]]);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_compact_binary_gprs(uint8_t *bytes, unsigned dst, unsigned a, unsigned b)
{
   if (dst >= AGX_APPLE9_GPR_COUNT || a >= AGX_APPLE9_GPR_COUNT ||
       b >= AGX_APPLE9_GPR_COUNT)
      return false;

   /* EXP-M4-38: FALU2, integer min/max, native half ALU, and integer
    * logic use the same scattered physical-register map.  The descriptor
    * top bits at 15 and 31 are auxiliaries; the actual source bit 6 values
    * live at 40 and 42. */
   set_bits(bytes, 4, 4, dst & 0xf);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 44, 1, dst >> 6);
   set_bits(bytes, 9, 6, a & 0x3f);
   set_bits(bytes, 40, 1, a >> 6);
   set_bits(bytes, 25, 6, b & 0x3f);
   set_bits(bytes, 42, 1, b >> 6);
   return true;
}

static bool
pack_float2(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
            struct agx_apple9_packed_instruction *packed)
{
   if ((instruction->op == AGX_APPLE9_VIR_FMUL_PROJECT) !=
       (instruction->encoding == AGX_APPLE9_ENC_FLOAT2_PROJECT))
      return false;
   uint8_t bytes[8] = {0x09, 0x05, 0x1c, 0x01, 0x00, 0x00};
   if (instruction->op == AGX_APPLE9_VIR_FMUL ||
       instruction->op == AGX_APPLE9_VIR_FMUL_PROJECT)
      set_bits(bytes, 16, 3,
               instruction->op == AGX_APPLE9_VIR_FMUL_PROJECT ? 7 : 5);
   else
      set_bits(bytes, 16, 3, 4);
   if (instruction->op == AGX_APPLE9_VIR_FSUB)
      set_bits(bytes, 43, 1, 1);

   /*
    * Publish the compiler-native compact-float destination state and consume
    * direct ALU producers through native consumer state 0.  EXP-M4-17 proved
    * that the source release bits, not their adjacent descriptor auxiliaries,
    * control whether a source remains available to a later instruction.
    *
    * Keep the auxiliary bits at their measured compiler values.  They are
    * independently output-inert in the tested scalar form, so they must not
    * be relied upon for correctness.
    */
   set_bits(bytes, 21, 1, 1);
   if (instruction->live_after_mask & (1u << 0)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (instruction->live_after_mask & (1u << 1)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }

   if (!pack_compact_binary_gprs(bytes, phys[instruction->dest],
                                 phys[instruction->src[0]],
                                 phys[instruction->src[1]]))
      return false;
   if (instruction->encoding == AGX_APPLE9_ENC_FLOAT2_PROJECT) {
      if (phys[instruction->dest] >= 64 || phys[instruction->src[0]] >= 64 ||
          phys[instruction->src[1]] >= 64 || instruction->immediate > 12)
         return false;
      /* Source-authored Metal probes use opcode 7 and name the CF slot
       * alongside the two GPR operands. Perspective coefficients can be
       * primitive-constant; this form preserves their constant fast path
       * instead of unconditionally multiplying them by reciprocal 1/W.
       * The CF nibble shares byte 5 with the dependency field, so this
       * measured form has six-bit GPR operands. */
      bytes[5] = instruction->immediate;
      packed_init(packed, bytes, 8);
   } else if (instruction->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT) {
      /* EXP-M4-56: extended destination publication for vertex exports.
       * The ordinary ALU destination cannot be consumed by VARY_STORE.
       * Keep the proven low destination bank; its extension differs from
       * the ordinary compact ALU register map. */
      if (phys[instruction->dest] >= 16 || phys[instruction->src[0]] >= 64 ||
          phys[instruction->src[1]] >= 64)
         return false;
      bytes[4] |= 0x41;
      packed_init(packed, bytes, 8);
   } else {
      if (instruction->encoding != AGX_APPLE9_ENC_FLOAT2_COMPACT)
         return false;
      packed_init(packed, bytes, 6);
   }
   return true;
}

static bool
pack_float2_immediate(const struct agx_apple9_vir_instr *instruction,
                      const uint8_t *phys,
                      struct agx_apple9_packed_instruction *packed)
{
   /*
    * The source-B immediate is an exact eight-bit minifloat.  Its exponent
    * overload is deliberately guarded to 8..15: encodings below 8 select the
    * uniform register file instead of an immediate.
    */
   const uint32_t bits = instruction->immediate;
   const uint32_t magnitude = bits & 0x7fffffffu;
   uint8_t encoded = 0;
   bool found = false;

   for (unsigned exponent = 8; exponent < 16 && !found; ++exponent) {
      for (unsigned mantissa = 0; mantissa < 8; ++mantissa) {
         float value = exponent == 8 ? (mantissa / 8.0f) * 0.25f
                                     : (1.0f + mantissa / 8.0f) *
                                          (1u << (exponent - 8)) * 0.125f;
         uint32_t candidate;
         memcpy(&candidate, &value, sizeof(candidate));
         if (candidate == magnitude) {
            encoded = (exponent << 4) | (mantissa << 1) | 1;
            found = true;
            break;
         }
      }
   }
   if (!found)
      return false;

   /*
    * This form is the caller-compiler form measured for a converted
    * fragment position feeding fadd-immediate.  It also matches the
    * independently validated packed-immediate operand layout.
    */
   uint8_t bytes[6] = {0x09, encoded, 0x34, 0x01, 0x80, 0x00};
   if (instruction->op == AGX_APPLE9_VIR_FMUL_IMM)
      set_bits(bytes, 16, 3, 5);
   set_bits(bytes, 19, 1, bits >> 31);
   set_bits(bytes, 4, 4, phys[instruction->dest]);
   set_bits(bytes, 25, 6, phys[instruction->src[0]]);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_fma(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
         struct agx_apple9_packed_instruction *packed)
{
   uint8_t bytes[8] = {0x09, 0x01, 0x1e, 0x05, 0x81, 0x08, 0x02, 0x00};
   unsigned dst = phys[instruction->dest];
   unsigned a = phys[instruction->src[0]];
   unsigned b = phys[instruction->src[1]];
   unsigned c = phys[instruction->src[2]];

   /* Same compiler-native producer and source-release convention as falu2. */
   set_bits(bytes, 21, 1, 1);
   if (instruction->live_after_mask & (1u << 0)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (instruction->live_after_mask & (1u << 1)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }
   if (instruction->live_after_mask & (1u << 2)) {
      set_bits(bytes, 47, 1, 1);
      set_bits(bytes, 39, 1, 0);
   }

   set_bits(bytes, 4, 4, dst & 0xf);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 60, 1, dst >> 6);
   set_bits(bytes, 9, 6, a & 0x3f);
   set_bits(bytes, 56, 1, a >> 6);
   set_bits(bytes, 25, 6, b & 0x3f);
   set_bits(bytes, 58, 1, b >> 6);
   set_bits(bytes, 41, 6, c & 0x3f);
   set_bits(bytes, 38, 1, c >> 6);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_logic(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
           struct agx_apple9_packed_instruction *packed)
{
   if (instruction->encoding != AGX_APPLE9_ENC_LOGIC_EXTENDED ||
       instruction->nr_srcs != 2)
      return false;

   uint8_t bytes[10] = {0x0b, 0x05, 0x1f, 0x01, 0x00,
                        0x00, 0x00, 0x80, 0x00, 0x00};
   if (instruction->op == AGX_APPLE9_VIR_IOR) {
      bytes[4] = 0x02;
      bytes[5] = 0x08;
   } else if (instruction->op == AGX_APPLE9_VIR_IXOR) {
      bytes[2] = 0x1e;
      bytes[4] = 0x02;
      bytes[5] = 0x08;
   }

   /*
    * The extended logic form carries source lifetime state in and beside its
    * two register descriptors.  These transitions are exact across the
    * caller-owned EXP-M4-16 pressure kernels (source A survives) and the
    * non-coalesced rings (both sources survive):
    *
    *    source A live: bit 15 = 1, bit 19 = 0
    *    source B live: bit 31 = 1, bit 20 = 0, bit 21 = 1
    *
    * Without these bits the instruction still computes its destination, but
    * the hardware may discard a source whose SSA value is used later.
    */
   /*
    * Unlike FALU's binary three-bit slot selector, integer logic uses a
    * six-bit one-hot pending-result mask split across bits 45..47 and
    * 61..63.
    */
   if (instruction->live_after_mask & (1u << 0)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (instruction->live_after_mask & (1u << 1)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
      set_bits(bytes, 21, 1, 1);
   }
   if (!pack_compact_binary_gprs(bytes, phys[instruction->dest],
                                 phys[instruction->src[0]],
                                 phys[instruction->src[1]]))
      return false;
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_minmax(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
            struct agx_apple9_packed_instruction *packed)
{
   uint8_t select;
   switch (instruction->op) {
   case AGX_APPLE9_VIR_FMAX:
      select = 0;
      break;
   case AGX_APPLE9_VIR_FMIN:
      select = 1;
      break;
   case AGX_APPLE9_VIR_UMAX:
      select = 4;
      break;
   case AGX_APPLE9_VIR_UMIN:
      select = 5;
      break;
   case AGX_APPLE9_VIR_IMAX:
      select = 6;
      break;
   case AGX_APPLE9_VIR_IMIN:
      select = 7;
      break;
   default:
      return false;
   }

   uint8_t bytes[6] = {0x02, 0x01, 0x1e, 0x05, select, 0x00};

   /*
    * Publish the native ALU-producer destination and consumer states.  The
    * adjacent source fields were inert in the load-fed umin probe, but native
    * ALU-fed integer and float min/max both correlate them with source last
    * use. Preserve that compiler-native state here; only the float form is
    * currently described architecturally as release/retain.
    */
   set_bits(bytes, 21, 1, 1);
   if (instruction->live_after_mask & (1u << 0)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (instruction->live_after_mask & (1u << 1)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }

   if (!pack_compact_binary_gprs(bytes, phys[instruction->dest],
                                 phys[instruction->src[0]],
                                 phys[instruction->src[1]]))
      return false;
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_select(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
            struct agx_apple9_packed_instruction *packed)
{
   unsigned dst = phys[instruction->dest];
   unsigned cmp_a = phys[instruction->src[0]];
   unsigned cmp_b = phys[instruction->src[1]];
   unsigned if_true = phys[instruction->src[2]];
   unsigned if_false = phys[instruction->src[3]];

   /*
    * Apple9 has distinct low-register and high-pressure envelopes for the
    * same explicit-false select fields.  The low form comes from the exact
    * caller-owned computed_select_values MSL used by the DAG hardware test;
    * EXP-M4-16 supplies the independently validated high-register form.
    */
   uint8_t bytes[10];
   bool low_form = cmp_a < 64 && cmp_b < 64 && if_true < 64 && if_false < 64;
   if (low_form) {
      /* Explicit-GPR baseline for four ALU-produced, last-use operands. */
      const uint8_t low[10] = {0x02, 0x01, 0x1f, 0x01, 0x82,
                               0x00, 0x05, 0x00, 0x80, 0x00};
      memcpy(bytes, low, sizeof(bytes));
   } else {
      const uint8_t extended[10] = {0x12, 0x85, 0x67, 0x87, 0x02,
                                    0xd8, 0x05, 0xd0, 0x00, 0xda};
      memcpy(bytes, extended, sizeof(bytes));
   }

   const unsigned condition = instruction->immediate & 0xff;
   if (condition != AGX_APPLE9_SELECT_FEQ &&
       condition != AGX_APPLE9_SELECT_FGT &&
       condition != AGX_APPLE9_SELECT_FLT &&
       condition != AGX_APPLE9_SELECT_UGT &&
       condition != AGX_APPLE9_SELECT_ULT &&
       condition != AGX_APPLE9_SELECT_IGT && condition != AGX_APPLE9_SELECT_ILT)
      return false;

   /*
    * RT-1a independently swept all seven compare condition codes.  Bit 2 of
    * the mode selects the equality-family form in the capture-derived float
    * path.  Integer equality and complementary relations are synthesized from
    * executed mode-2 less-than selects; byte 5/9 are operand descriptors.
    */
   bytes[6] = condition;
   if (instruction->immediate & AGX_APPLE9_SELECT_EQUALITY)
      bytes[4] |= 0x04;

   if (low_form) {
      /*
       * This baseline already carries native destination state 1 and
       * consumer state 0 for four ALU-produced operands.  Load-fed EXP-M4-17
       * probes found every adjacent source bit output-inert, but caller-owned
       * ALU-fed select programs correlate each pair with source last use.
       * Preserve those native source-state pairs without transplanting the
       * producer bit at instruction bit 21 from another ALU family.  Both
       * caller-owned unsigned and signed low-GPR selects leave that bit clear;
       * setting it is output-inert for the unsigned condition but breaks the
       * signed condition on T8132.
       */
      if (instruction->live_after_mask & (1u << 0)) {
         set_bits(bytes, 15, 1, 1);
         set_bits(bytes, 19, 1, 0);
      }
      if (instruction->live_after_mask & (1u << 1)) {
         set_bits(bytes, 31, 1, 1);
         set_bits(bytes, 20, 1, 0);
      }
      if (instruction->live_after_mask & (1u << 2)) {
         set_bits(bytes, 47, 1, 1);
         set_bits(bytes, 39, 1, 0);
      }
      if (instruction->live_after_mask & (1u << 3)) {
         set_bits(bytes, 79, 1, 1);
         set_bits(bytes, 71, 1, 0);
      }
   }

   set_bits(bytes, 4, 4, dst);
   set_bits(bytes, 9, 6, cmp_a & 0x3f);
   set_bits(bytes, 56, 1, cmp_a >> 6);
   set_bits(bytes, 25, 6, cmp_b & 0x3f);
   set_bits(bytes, 58, 1, cmp_b >> 6);
   set_bits(bytes, 41, 6, if_true & 0x3f);
   set_bits(bytes, 38, 1, if_true >> 6);
   set_bits(bytes, 73, 6, if_false & 0x3f);
   set_bits(bytes, 70, 1, if_false >> 6);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_predicate_compare(const struct agx_apple9_vir_instr *instruction,
                       const uint8_t *phys,
                       struct agx_apple9_packed_instruction *packed)
{
   const bool short_form =
      instruction->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT;
   const bool extended_form =
      instruction->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
   const bool loop_form =
      instruction->encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP;
   if ((!short_form && !extended_form && !loop_form) ||
       instruction->nr_srcs != 2 ||
       (instruction->immediate & ~(0xffu | AGX_APPLE9_PREDICATE_INVERT |
                                   AGX_APPLE9_PREDICATE_BANK_MASK)) != 0)
      return false;

   const unsigned condition = instruction->immediate & 0xff;
   if (short_form) {
      if (condition != AGX_APPLE9_PREDICATE_FGT &&
          condition != AGX_APPLE9_PREDICATE_FLT &&
          condition != AGX_APPLE9_PREDICATE_UGT &&
          condition != AGX_APPLE9_PREDICATE_ULT &&
          condition != AGX_APPLE9_PREDICATE_IGT &&
          condition != AGX_APPLE9_PREDICATE_ILT)
         return false;
   } else if (condition != AGX_APPLE9_PREDICATE_EXT_FEQ &&
              condition != AGX_APPLE9_PREDICATE_EXT_FGE_SEQUENCE &&
              condition != AGX_APPLE9_PREDICATE_EXT_FLE_SEQUENCE &&
              condition != AGX_APPLE9_PREDICATE_EXT_IEQ) {
      return false;
   }

   const unsigned src0 = phys[instruction->src[0]];
   const unsigned src1 = phys[instruction->src[1]];
   if (src0 >= 64 || src1 >= 64)
      return false;

   const unsigned predicate_bank =
      (instruction->immediate & AGX_APPLE9_PREDICATE_BANK_MASK) >>
      AGX_APPLE9_PREDICATE_BANK_SHIFT;
   if (predicate_bank >= AGX_APPLE9_PREDICATE_BANK_COUNT)
      return false;

   /* Byte 0 bits 5--7 select one of six predicate scratch banks.  Bit 4 is
    * independently validated producer polarity.  These are not execution-
    * mask-stack depths: EXP-M4-53 relocates producer/consumer pairs between
    * banks while preserving exact output. */
   const uint8_t opcode =
      0x0a | (predicate_bank << 5) |
      ((instruction->immediate & AGX_APPLE9_PREDICATE_INVERT) ? 0x10 : 0);
   uint8_t control = 0x22 | ((extended_form || loop_form) ? 0x01 : 0);
   /* EXP-M4-46's native loop latches use control 0x23 even when the compare
    * operands are dead at the latch.  The 0x08/0x10 source-release controls
    * are hardware-proven only for the ordinary 0xc0-tail predicate forms;
    * do not extrapolate them to the loop-tail form. */
   if (!loop_form) {
      if (!(instruction->live_after_mask & BITFIELD_BIT(0)))
         control |= 0x08;
      if (!(instruction->live_after_mask & BITFIELD_BIT(1)))
         control |= 0x10;
   }

   if (short_form) {
      const uint8_t bytes[6] = {opcode,    (uint8_t)((src0 << 1) | 1),
                                control,   (uint8_t)((src1 << 1) | 1),
                                condition, 0xc0};
      packed_init(packed, bytes, sizeof(bytes));
   } else {
      const uint8_t bytes[10] = {
         opcode,    (uint8_t)((src0 << 1) | 1),
         control,   (uint8_t)((src1 << 1) | 1),
         0x06,      0x00,
         condition, loop_form ? 0x00 : 0xc0,
         0x00,      0x00,
      };
      packed_init(packed, bytes, sizeof(bytes));
   }
   return true;
}

static bool
pack_vir_instruction_body(const struct agx_apple9_vir_instr *instruction,
                          const uint8_t *phys,
                          struct agx_apple9_packed_instruction *packed,
                          const char **reason)
{
   if (reason != NULL)
      *reason = NULL;

   switch (instruction->op) {
   case AGX_APPLE9_VIR_IMM:
      if (instruction->encoding == AGX_APPLE9_ENC_MOV_IMM_COMPACT)
         return instruction->immediate <= 0x7f &&
                agx_apple9_pack_mov_imm(phys[instruction->dest],
                                        instruction->immediate, packed);
      if (instruction->encoding == AGX_APPLE9_ENC_MOV_IMM32)
         return agx_apple9_pack_mov_imm32(phys[instruction->dest],
                                          instruction->immediate, packed);
      break;
   case AGX_APPLE9_VIR_GET_GLOBAL_ID:
      return agx_apple9_pack_get_global_id(phys[instruction->dest],
                                           instruction->immediate, packed);
   case AGX_APPLE9_VIR_GET_SR:
      if (instruction->encoding == AGX_APPLE9_ENC_GET_VERTEX_ID) {
         const unsigned dst = phys[instruction->dest];
         if (dst >= 16 || instruction->immediate != 0x10dd ||
             instruction->nr_srcs)
            return false;
         /* Vertex ID is a slot-1 return, unlike compute's synchronous SR
          * reads. Materialize it immediately with an allocated identity IOR.
          * This bounded pair also leaves the slot free for subsequent work. */
         struct agx_apple9_vir_instr copy = {
            .op = AGX_APPLE9_VIR_IOR,
            .encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED,
            .dest = 0,
            .src = {0, 0},
            .nr_srcs = 2,
         };
         uint8_t registers[] = {dst};
         struct agx_apple9_packed_instruction move;
         if (!pack_logic(&copy, registers, &move) ||
             !apple9_pack_dependency(move.bytes, move.length,
                                     AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63,
                                     AGX_APPLE9_SCOREBOARD_SLOT_1))
            return false;
         uint8_t bytes[14] = {(dst << 4) | 0x0c, 0xdd, 0x10, 0x06};
         memcpy(bytes + 4, move.bytes, move.length);
         packed_init(packed, bytes, sizeof(bytes));
         return true;
      }
      if (instruction->encoding == AGX_APPLE9_ENC_GET_SR_ZEXT16)
         return agx_apple9_pack_get_sr_zext16(
            phys[instruction->dest], instruction->immediate & 0xff, packed);
      if (instruction->encoding != AGX_APPLE9_ENC_GET_SR)
         break;
      return agx_apple9_pack_get_sr(
         phys[instruction->dest], instruction->immediate & 0xff,
         (instruction->immediate >> 8) & 0xff, packed);
   case AGX_APPLE9_VIR_DEVICE_LOAD: {
      const unsigned components =
         instruction->dest_components ? instruction->dest_components : 1;
      uint16_t expected_token;
      if (instruction->immediate > UINT8_MAX ||
          !apple9_device_load_raw_token_valid(
             instruction->device_load_raw_token) ||
          instruction->producer_scoreboard_slot <
             AGX_APPLE9_SCOREBOARD_SLOT_1 ||
          instruction->producer_scoreboard_slot >
             AGX_APPLE9_SCOREBOARD_SLOT_6 ||
          !apple9_scalar_load_token_for_slot(
             instruction->producer_scoreboard_slot, &expected_token) ||
          instruction->device_load_raw_token != expected_token)
         break;

      if ((instruction->device_load_index_kind !=
              AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR &&
           instruction->device_load_index_kind !=
              AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR) ||
          instruction->encoding != AGX_APPLE9_ENC_DEVICE_LOAD ||
          instruction->nr_srcs != 1)
         break;
      unsigned index = phys[instruction->src[0]];

      /* The raw packer takes an explicit lifetime assertion for byte-oracle
       * tests. VIR code derives it from SSA liveness. */
      enum agx_apple9_device_load_index_kind index_lifetime =
         (instruction->live_after_mask & 1u)
            ? AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR
            : AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR;

      const unsigned memory_bits =
         instruction->memory_bits ? instruction->memory_bits : 32;
      if (components == 1)
         return agx_apple9_pack_device_load_scalar_raw(
            phys[instruction->dest], index, instruction->immediate, memory_bits,
            index_lifetime, instruction->device_load_flags,
            instruction->device_load_index_first_consumer,
            instruction->device_load_raw_token, packed);
      if (memory_bits != 32)
         break;
      return agx_apple9_pack_device_load_vector_u32_raw(
         phys[instruction->dest], index, instruction->immediate, components,
         index_lifetime, instruction->device_load_flags,
         instruction->device_load_index_first_consumer,
         instruction->device_load_raw_token, packed);
   }
   case AGX_APPLE9_VIR_DEVICE_STORE: {
      const unsigned components = instruction->memory_components;
      if (components < 1 || components > 4 ||
          instruction->nr_srcs != components + 1 ||
          instruction->immediate > UINT8_MAX ||
          instruction->encoding != AGX_APPLE9_ENC_DEVICE_STORE)
         break;

      const unsigned data = phys[instruction->src[0]];
      const unsigned index = phys[instruction->src[components]];
      const bool release_index =
         !(instruction->live_after_mask & (1u << components));
      for (unsigned c = 1; c < components; ++c) {
         if (phys[instruction->src[c]] != data + c)
            return false;
      }

      if (components == 1)
         return agx_apple9_pack_device_store_scalar(
            data, index, instruction->immediate, instruction->memory_bits,
            instruction->scoreboard_slot, release_index, packed);
      return agx_apple9_pack_device_store_vector_u32(
         data, index, instruction->immediate, components,
         instruction->scoreboard_slot, release_index, packed);
   }
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
      if (instruction->encoding != AGX_APPLE9_ENC_DEVICE_ATOMIC ||
          instruction->memory_bits != 32 ||
          instruction->nr_srcs != instruction->memory_components + 1 ||
          instruction->memory_components < 1 ||
          instruction->memory_components > 2)
         break;
      for (unsigned c = 1; c < instruction->memory_components; ++c) {
         if (phys[instruction->src[c]] != phys[instruction->src[0]] + c)
            return false;
      }
      return agx_apple9_pack_device_atomic(
         phys[instruction->src[instruction->memory_components]],
         phys[instruction->src[0]], instruction->immediate,
         instruction->atomic_op, instruction->atomic_discard,
         instruction->scoreboard_slot, packed);
   case AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT: {
      if (instruction->encoding != AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT ||
          instruction->dest != AGX_APPLE9_VREG_INVALID ||
          instruction->nr_srcs != 1 || instruction->immediate != 0)
         break;

      const unsigned destination = phys[instruction->src[0]];
      const unsigned slot = instruction->producer_scoreboard_slot;
      if (destination >= 64 || slot < AGX_APPLE9_SCOREBOARD_SLOT_1 ||
          slot > AGX_APPLE9_SCOREBOARD_SLOT_6)
         return false;

      /* The publication record uses a compact three-bit code, not the
       * consumer families' six-bit one-hot dependency mask.  Native Metal
       * establishes 001/010/100 for slots 6/1/2; EXP-M4-49 established the
       * remaining codes for slots 3/4/5 with exact hardware results.  The
       * atomic packet retains its independently scheduled input dependency;
       * those bits do not select its returned result. */
      static const uint8_t publication_code[] = {
         [AGX_APPLE9_SCOREBOARD_SLOT_1] = 2,
         [AGX_APPLE9_SCOREBOARD_SLOT_2] = 4,
         [AGX_APPLE9_SCOREBOARD_SLOT_3] = 3,
         [AGX_APPLE9_SCOREBOARD_SLOT_4] = 5,
         [AGX_APPLE9_SCOREBOARD_SLOT_5] = 6,
         [AGX_APPLE9_SCOREBOARD_SLOT_6] = 1,
      };
      const unsigned publication = publication_code[slot];

      /* EXP-M4-47 returning device atomics all use this eight-byte landing
       * form.  The high nibble selects the destination; bit 15 makes it one
       * long instruction rather than the unrelated two-byte MOV_IMM form.
       * Keeping it explicit prevents instruction decoders and schedulers from
       * splitting the atomic handoff into two fictitious instructions. */
      const uint8_t bytes[] = {
         (uint8_t)(((destination & 0x0f) << 4) | 0x0c),
         0x80,
         (uint8_t)(0x09 | ((destination >> 4) << 6)),
         0xa7,
         0x00,
         (uint8_t)(publication << 5),
         0x00,
         0x00,
      };
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_ITER: {
      if (instruction->encoding != AGX_APPLE9_ENC_ITER ||
          instruction->nr_srcs || phys[instruction->dest] >= 64 ||
          (instruction->immediate & ~0xfu) ||
          (instruction->immediate & 15) > 12)
         return false;
      /* Ordinary center coefficients, including coefficient 0 for 1/W.
       * Keep byte1 bit3 clear: setting it on the first ITER makes implicit
       * texture LOD fail in partially covered quads (mipmap edge probe).
       * Its earlier interpretation as a first-iterator marker was wrong. */
      unsigned imm = instruction->immediate;
      const uint8_t bytes[] = {0x2f, 0x05,
                               0x54, phys[instruction->dest] << 1,
                               0x03, (imm & 0xff) << 1,
                               0,    2,
                               0x10, 0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_VARY_STORE: {
      if (instruction->encoding != AGX_APPLE9_ENC_VARY_STORE ||
          instruction->nr_srcs != 1 || instruction->immediate >= 16 ||
          phys[instruction->src[0]] >= 16)
         return false;
      unsigned slot = instruction->immediate;
      const uint8_t bytes[] = {0x57,
                               0x06,
                               0x54,
                               phys[instruction->src[0]] << 1,
                               (slot & 7) << 5,
                               0x40 | (slot >> 3),
                               0,
                               0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TILE_ACCESS:
   case AGX_APPLE9_VIR_TILE_FENCE: {
      const bool access = instruction->op == AGX_APPLE9_VIR_TILE_ACCESS;
      if (instruction->nr_srcs ||
          instruction->encoding != (access ? AGX_APPLE9_ENC_TILE_ACCESS
                                           : AGX_APPLE9_ENC_TILE_FENCE) ||
          (access ? instruction->immediate != 0x600 &&
                       instruction->immediate != 0x80c &&
                       instruction->immediate != 0x808
                  : instruction->immediate != 0x20c))
         return false;
      const uint8_t bytes[] = {
         instruction->op == AGX_APPLE9_VIR_TILE_ACCESS ? 0x87 : 0x07,
         2,
         0x54,
         instruction->immediate & 0xff,
         (instruction->immediate >> 8) & 0xff,
         0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TEXTURE_COORDS: {
      if (instruction->encoding != AGX_APPLE9_ENC_TEXTURE_COORDS ||
          instruction->nr_srcs != 3 || instruction->dest_components != 2 ||
          phys[instruction->dest] > 6 || (phys[instruction->dest] & 1))
         return false;
      uint8_t bytes[16];
      for (unsigned c = 0; c < 2; ++c) {
         uint8_t registers[] = {phys[instruction->dest] + c,
                               phys[instruction->src[c]],
                               phys[instruction->src[2]]};
         struct agx_apple9_vir_instr publish = {
            .op = AGX_APPLE9_VIR_FMUL, .encoding = AGX_APPLE9_ENC_FLOAT2_EXPORT,
            .dest = 0, .nr_srcs = 2, .src = {1, 2},
            .live_after_mask = ((instruction->live_after_mask >> c) & 1) |
                               ((c == 0 || (instruction->live_after_mask & 4)) ? 2 : 0),
         };
         struct agx_apple9_packed_instruction part;
         if (!pack_float2(&publish, registers, &part))
            return false;
         memcpy(bytes + c * 8, part.bytes, 8);
      }
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE: {
      if (instruction->encoding != AGX_APPLE9_ENC_TEXTURE_SAMPLE ||
          instruction->nr_srcs != 2 || instruction->dest_components != 4 ||
          instruction->immediate ||
          instruction->producer_scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_1)
         return false;
      unsigned dst = phys[instruction->dest], coord = phys[instruction->src[0]];
      if (dst > 12 || coord > 6 || (coord & 1) ||
          phys[instruction->src[1]] != coord + 1)
         return false;
      /* The four-byte prefix names the result tuple and coordinate
       * publication. The legacy database's sampler-byte 'coord' label is
       * not this operand. First bound texture/sampler, FP32 2D implicit LOD. */
      const uint8_t bytes[] = {5 | (dst << 3), 0x80 | coord, 0x0c, 0xb8,
                              0xb0, 0, 0, 0, 1, 0, 0x10, 0, 1, 0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TILE_LOAD: {
      if (instruction->encoding != AGX_APPLE9_ENC_TILE_LOAD ||
          instruction->nr_srcs || instruction->immediate ||
          phys[instruction->dest] >= 64 ||
          instruction->producer_scoreboard_slot < AGX_APPLE9_SCOREBOARD_SLOT_1 ||
          instruction->producer_scoreboard_slot > AGX_APPLE9_SCOREBOARD_SLOT_6)
         return false;
      /* Explicit framebuffer input in our M4 f_blend shader, EXP-M4-01.
       * Native tag 0x2ce publishes slot 6; Mesa-produced reads and matching
       * consumers have been hardware-validated for all six slots. */
      const unsigned tag = 0x4e | ((instruction->producer_scoreboard_slot - 1) << 7);
      const uint8_t bytes[] = {0x67, 0x0e, 0x54, phys[instruction->dest] << 1,
                              0, 0, 1, tag & 0xff, tag >> 8, 0, 0, 0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TILE_STORE: {
      if (instruction->encoding != AGX_APPLE9_ENC_TILE_STORE ||
          instruction->nr_srcs != 1 || instruction->immediate != 0 ||
          phys[instruction->src[0]] >= 64)
         return false;
      /* EXP-M4-56: format 0x4e consumes one RGBA8 word. Explicit integer
       * packing is independent of the native compiler's conversion code. */
      const uint8_t bytes[] = {0xe7, 6, 0x54, phys[instruction->src[0]] << 1,
                               0,    0, 1,    0x4e,
                               0,    0, 0,    0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_U2F32:
   case AGX_APPLE9_VIR_I2F32:
      return pack_i2f32(instruction, phys, packed);
   case AGX_APPLE9_VIR_F2I32:
   case AGX_APPLE9_VIR_F2U32:
      return pack_f2i32(instruction, phys, packed);
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
   case AGX_APPLE9_VIR_FFLOOR:
   case AGX_APPLE9_VIR_FCEIL:
   case AGX_APPLE9_VIR_FTRUNC:
   case AGX_APPLE9_VIR_FROUND_EVEN:
      return pack_fspecial(instruction, phys, packed);
   case AGX_APPLE9_VIR_ISHR:
      return pack_ishr_imm(instruction, phys, packed);
   case AGX_APPLE9_VIR_IMUL:
      break;
   case AGX_APPLE9_VIR_IADD:
      if (instruction->encoding != AGX_APPLE9_ENC_INT_ADD_EXTENDED ||
          instruction->nr_srcs != 2)
         break;
      return pack_iadd(instruction, phys, packed);
   case AGX_APPLE9_VIR_ISUB:
      if (instruction->encoding != AGX_APPLE9_ENC_INT_ADD_EXTENDED ||
          instruction->nr_srcs != 2)
         break;
      return pack_iadd(instruction, phys, packed);
   case AGX_APPLE9_VIR_IMAD:
      return pack_imad(instruction, phys, packed);
   case AGX_APPLE9_VIR_IAND:
   case AGX_APPLE9_VIR_IOR:
   case AGX_APPLE9_VIR_IXOR:
      return pack_logic(instruction, phys, packed);
   case AGX_APPLE9_VIR_IMIN:
   case AGX_APPLE9_VIR_IMAX:
   case AGX_APPLE9_VIR_UMIN:
   case AGX_APPLE9_VIR_UMAX:
   case AGX_APPLE9_VIR_FMIN:
   case AGX_APPLE9_VIR_FMAX:
      return pack_minmax(instruction, phys, packed);
   case AGX_APPLE9_VIR_FADD:
   case AGX_APPLE9_VIR_FSUB:
   case AGX_APPLE9_VIR_FMUL:
   case AGX_APPLE9_VIR_FMUL_PROJECT:
      return pack_float2(instruction, phys, packed);
   case AGX_APPLE9_VIR_FADD_IMM:
   case AGX_APPLE9_VIR_FMUL_IMM:
      return pack_float2_immediate(instruction, phys, packed);
   case AGX_APPLE9_VIR_FMA:
      return pack_fma(instruction, phys, packed);
   case AGX_APPLE9_VIR_SELECT:
      return pack_select(instruction, phys, packed);
   case AGX_APPLE9_VIR_PREDICATE_COMPARE:
      return pack_predicate_compare(instruction, phys, packed);
   case AGX_APPLE9_VIR_EXEC_MASK_PUSH: {
      const unsigned selector = instruction->immediate & 0xffu;
      if (instruction->encoding != AGX_APPLE9_ENC_EXEC_MASK_PUSH ||
          instruction->nr_srcs != 0 ||
          (instruction->immediate & ~(0xffu | AGX_APPLE9_EXEC_MASK_INVERT)) !=
             0 ||
          selector < AGX_APPLE9_EXEC_MASK_SOURCE(0) ||
          selector > AGX_APPLE9_EXEC_MASK_SOURCE(7) ||
          ((selector - AGX_APPLE9_EXEC_MASK_SOURCE(0)) % 4) != 0)
         break;
      const uint8_t bytes[] = {
         0x0f,
         0x05,
         0x54,
         (uint8_t)(selector |
                   ((instruction->immediate & AGX_APPLE9_EXEC_MASK_INVERT)
                       ? 0x20
                       : 0)),
      };
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_EXEC_MASK_ELSE: {
      if (instruction->encoding != AGX_APPLE9_ENC_EXEC_MASK_ELSE ||
          instruction->nr_srcs != 0 || instruction->immediate != 0)
         break;
      const uint8_t bytes[] = {0x0f, 0x04, 0x04, 0x19};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_EXEC_MASK_POP: {
      if (instruction->encoding != AGX_APPLE9_ENC_EXEC_MASK_POP ||
          instruction->nr_srcs != 0 || instruction->immediate != 0)
         break;
      const uint8_t bytes[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_LOOP_MASK_PUSH: {
      if (instruction->encoding != AGX_APPLE9_ENC_LOOP_MASK_PUSH ||
          instruction->nr_srcs != 0 || instruction->immediate != 0)
         break;
      const uint8_t bytes[] = {0x0f, 0x05, 0x54, 0x1a};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_LOOP_MASK_UPDATE: {
      if (instruction->encoding != AGX_APPLE9_ENC_LOOP_MASK_UPDATE ||
          instruction->nr_srcs != 0 || instruction->immediate > UINT8_MAX)
         break;
      const uint8_t bytes[] = {0x8f, 0x04, 0x54,
                               (uint8_t)instruction->immediate};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_LOOP_MASK_POP: {
      if (instruction->encoding != AGX_APPLE9_ENC_LOOP_MASK_POP ||
          instruction->nr_srcs != 0 || instruction->immediate != 0)
         break;
      const uint8_t bytes[] = {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_JMP_EXEC_ANY:
   case AGX_APPLE9_VIR_JMP_EXEC_NONE: {
      const bool any = instruction->op == AGX_APPLE9_VIR_JMP_EXEC_ANY;
      const enum agx_apple9_encoding expected =
         any ? AGX_APPLE9_ENC_JMP_EXEC_ANY : AGX_APPLE9_ENC_JMP_EXEC_NONE;
      if (instruction->encoding != expected || instruction->nr_srcs != 0)
         break;

      /* Compiler-authored branches fit in signed 32 bits. Sign-extension to
       * the architectural 48-bit start-relative displacement is explicit. */
      const int64_t displacement = (int32_t)instruction->immediate;
      uint8_t bytes[10] = {0x0f, any ? 0x00 : 0x01, 0x54};
      const uint64_t encoded = (uint64_t)displacement;
      for (unsigned byte = 0; byte < 6; ++byte)
         bytes[3 + byte] = encoded >> (8 * byte);
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_BREAK_MASK_UNWIND: {
      const unsigned scope_tag =
         AGX_APPLE9_BREAK_SCOPE_TAG(instruction->immediate);
      const unsigned loop_depth =
         AGX_APPLE9_BREAK_LOOP_DEPTH(instruction->immediate);
      if (instruction->encoding != AGX_APPLE9_ENC_BREAK_MASK_UNWIND ||
          instruction->nr_srcs != 0 || scope_tag < 2 || scope_tag > UINT8_MAX ||
          loop_depth == 0 || loop_depth > UINT8_MAX)
         break;
      const uint8_t bytes[] = {
         0x8f, 0x05, 0x54, (uint8_t)scope_tag, 0x00, (uint8_t)loop_depth};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_COLLECT:
   case AGX_APPLE9_VIR_MERGE:
   case AGX_APPLE9_VIR_MASKED_COPY:
      break;
   }

   if (reason != NULL)
      *reason = "Apple9 packer cannot encode this virtual instruction";
   return false;
}

bool
agx_apple9_pack_vir_instruction(const struct agx_apple9_vir_instr *instruction,
                                const uint8_t *phys,
                                struct agx_apple9_packed_instruction *packed,
                                const char **reason)
{
   if (reason != NULL)
      *reason = NULL;

   const enum agx_apple9_dependency_layout layout =
      instruction->encoding == AGX_APPLE9_ENC_PSEUDO
         ? AGX_APPLE9_DEPENDENCY_NONE
         : agx_apple9_encoding_info(instruction->encoding)->dependency_layout;
   if (!apple9_dependency_slot_valid(layout, instruction->scoreboard_slot)) {
      if (reason != NULL)
         *reason =
            "Apple9 instruction dependency does not fit its encoding layout";
      return false;
   }

   if (!pack_vir_instruction_body(instruction, phys, packed, reason))
      return false;

   if (!apple9_pack_dependency(packed->bytes, packed->length, layout,
                               instruction->scoreboard_slot)) {
      if (reason != NULL)
         *reason = "Apple9 could not encode the instruction dependency";
      return false;
   }

   return true;
}
