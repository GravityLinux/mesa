/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_ir.h"
#include "agx_compile.h"
#include "agx_apple9_encoding.h"
#include "agx_apple9_profile.h"
#include "agx_builder.h"
#include "util/u_dynarray.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define APPLE9_FIRST_ALLOCATABLE_GPR 0
#define APPLE9_FIRST_GENERAL_GPR     16
#define APPLE9_LAST_ALLOCATABLE_GPR  (AGX_APPLE9_GPR_COUNT - 1)

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
   case AGX_APPLE9_DEPENDENCY_MASK_61_63_77_79:
   case AGX_APPLE9_DEPENDENCY_MASK_61_63_93_95:
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
   case AGX_APPLE9_DEPENDENCY_MASK_61_63_77_79:
   case AGX_APPLE9_DEPENDENCY_MASK_61_63_93_95: {
      unsigned offset = layout == AGX_APPLE9_DEPENDENCY_MASK_45_47_61_63 ? 45 : 61;
      unsigned second = offset +
         (layout == AGX_APPLE9_DEPENDENCY_MASK_61_63_93_95 ? 32 : 16);
      if (length * 8 < second + 3)
         return false;
      set_bits(bytes, offset, 3,
               slot >= AGX_APPLE9_SCOREBOARD_SLOT_1 &&
                     slot <= AGX_APPLE9_SCOREBOARD_SLOT_3
                  ? 1u << (slot - 1)
                  : 0);
      set_bits(bytes, second, 3,
               slot >= AGX_APPLE9_SCOREBOARD_SLOT_4
                  ? 1u << (slot - AGX_APPLE9_SCOREBOARD_SLOT_4)
                  : 0);
      return true;
   }
   }

   return false;
}

struct agx_apple9_use_analysis {
   bool valid;
   struct agx_apple9_vir_instr **definitions;
   struct agx_apple9_use **first, **last;
   struct agx_apple9_use *uses;
};

void
agx_apple9_invalidate_uses(struct agx_apple9_vir_program *program)
{
   program->dependencies_finalized = false;
   if (program->use_analysis)
      program->use_analysis->valid = false;
}

static void
apple9_free_use_analysis(struct agx_apple9_use_analysis *analysis)
{
   if (!analysis)
      return;
   free(analysis->definitions);
   free(analysis->first);
   free(analysis->last);
   free(analysis->uses);
   free(analysis);
}

struct apple9_instruction_node {
   struct list_head link;
   struct agx_apple9_block *block;
   unsigned position;
   struct agx_apple9_vir_instr instruction;
};

static struct apple9_instruction_node *
apple9_node(const struct agx_apple9_vir_instr *instr)
{
   return container_of(instr, struct apple9_instruction_node, instruction);
}

struct agx_apple9_block *
agx_apple9_instr_block(const struct agx_apple9_vir_instr *instr)
{
   return apple9_node(instr)->block;
}

struct agx_apple9_block *
agx_apple9_block_create(struct agx_apple9_vir_program *program)
{
   struct agx_apple9_block *block = calloc(1, sizeof(*block));
   if (!block)
      return NULL;
   block->program = program;
   list_inithead(&block->instructions);
   block->allocated_next = program->allocated_blocks;
   program->allocated_blocks = block;
   return block;
}

void
agx_apple9_block_begin(struct agx_apple9_vir_program *program,
                      struct agx_apple9_block *block)
{
   assert(block && block->program == program && !block->placed);
   block->placed = true;
   block->prev = program->last_block;
   if (program->last_block)
      program->last_block->next = block;
   else
      program->blocks = block;
   program->last_block = program->current_block = block;
}

void
agx_apple9_vir_reindex(struct agx_apple9_vir_program *program)
{
   agx_apple9_invalidate_uses(program);
   unsigned count = 0, index = 0;
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      block->index = index++;
      block->start_index = count;
      list_for_each_entry(struct apple9_instruction_node, node, &block->instructions, link) {
         node->position = count;
         program->instructions[count++] = &node->instruction;
      }
   }
   assert(count == program->instruction_count);
}

void
agx_apple9_vir_move_before(struct agx_apple9_vir_program *program,
                   struct agx_apple9_vir_instr *instr,
                   struct agx_apple9_vir_instr *before)
{
   struct apple9_instruction_node *node = apple9_node(instr);
   struct apple9_instruction_node *next = apple9_node(before);
   list_del(&node->link);
   list_addtail(&node->link, &next->link);
   assert(next->block->program == program && node->block->program == program);
   node->block = next->block;
   agx_apple9_vir_reindex(program);
}

void
agx_apple9_vir_remove(struct agx_apple9_vir_program *program,
                          struct agx_apple9_vir_instr *instr)
{
   struct apple9_instruction_node *node = apple9_node(instr);
   list_del(&node->link);
   free(node);
   --program->instruction_count;
   agx_apple9_vir_reindex(program);
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
   for (struct agx_apple9_block *b = program->allocated_blocks, *next; b; b = next) {
      next = b->allocated_next;
      list_for_each_entry_safe(struct apple9_instruction_node, node, &b->instructions, link) {
         list_del(&node->link);
         free(node);
      }
      free(b->predecessors);
      free(b->live_in);
      free(b->live_out);
      free(b);
   }
   apple9_free_use_analysis(program->use_analysis);
   free(program->instructions);
   free(program->publication);
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
   agx_apple9_invalidate_uses(program);
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
   agx_apple9_invalidate_uses(program);
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
   agx_apple9_invalidate_uses(program);
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

   if (!program->current_block) {
      struct agx_apple9_block *block = agx_apple9_block_create(program);
      if (!block)
         return NULL;
      agx_apple9_block_begin(program, block);
   }
   struct apple9_instruction_node *node = calloc(1, sizeof(*node));
   if (!node)
      return NULL;
   node->block = program->current_block;
   list_addtail(&node->link, &node->block->instructions);
   program->instructions[program->instruction_count++] = &node->instruction;
   return &node->instruction;
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
      .branch_target = NULL,
      .memory_index_shift = op == AGX_APPLE9_VIR_DEVICE_LOAD ? 2 : 0,
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
      .branch_target = NULL,
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
                           struct agx_apple9_block *target)
{
   if ((op != AGX_APPLE9_VIR_JMP_EXEC_ANY ||
        encoding != AGX_APPLE9_ENC_JMP_EXEC_ANY) &&
       (op != AGX_APPLE9_VIR_JMP_EXEC_NONE ||
        encoding != AGX_APPLE9_ENC_JMP_EXEC_NONE))
      return false;

   if (!agx_apple9_vir_emit_side_effect(program, op, encoding, NULL, 0, 0))
      return false;

   program->instructions[program->instruction_count - 1]->branch_target =
      target;
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
agx_apple9_vir_emit_iter_flat(struct agx_apple9_vir_program *program,
                              unsigned coefficient)
{
   uint32_t base =
      agx_apple9_vir_emit(program, AGX_APPLE9_VIR_ITER_FLAT,
                          AGX_APPLE9_ENC_ITER_FLAT, NULL, 0, coefficient);
   if (base == AGX_APPLE9_VREG_INVALID)
      return base;
   if (!apple9_vir_append_values(program, 2)) {
      --program->instruction_count;
      program->value_count = base;
      return AGX_APPLE9_VREG_INVALID;
   }
   program->instructions[program->instruction_count - 1]->dest_components = 3;
   program->instructions[program->instruction_count - 1]->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
   return base + 2;
}

uint32_t
agx_apple9_vir_emit_device_load(
   struct agx_apple9_vir_program *program, unsigned binding, uint32_t index,
   const struct agx_apple9_device_load_contract *contract)
{
   if (program == NULL || contract == NULL || binding > UINT8_MAX ||
       (contract->flags & ~AGX_APPLE9_DEVICE_LOAD_HAS_NEXT) ||
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
      program->instructions[program->instruction_count - 1];
   instruction->device_load_flags = contract->flags;
   instruction->device_load_index_kind = contract->index_kind;
   instruction->device_load_raw_token = contract->raw_token;
   if (!apple9_scalar_load_slot_for_token(
          contract->raw_token, &instruction->producer_scoreboard_slot))
      return AGX_APPLE9_VREG_INVALID;
   return value;
}

uint32_t
agx_apple9_vir_emit_cube(struct agx_apple9_vir_program *program,
                         const uint32_t src[3], unsigned mode)
{
   if (mode > 2)
      return AGX_APPLE9_VREG_INVALID;
   uint32_t value = agx_apple9_vir_emit(program, AGX_APPLE9_VIR_CUBE,
                                       AGX_APPLE9_ENC_CUBE, src, 3, mode);
   if (value == AGX_APPLE9_VREG_INVALID)
      return value;
   if (mode == 0) {
      if (!apple9_vir_append_values(program, 1))
         return AGX_APPLE9_VREG_INVALID;
      program->instructions[program->instruction_count - 1]->dest_components = 2;
   }
   return value;
}

uint32_t
agx_apple9_vir_emit_unpack_norm(struct agx_apple9_vir_program *program,
                               uint32_t src, unsigned mode)
{
   uint32_t value = agx_apple9_vir_emit(program, AGX_APPLE9_VIR_UNPACK_NORM,
                                       AGX_APPLE9_ENC_UNPACK_NORM, &src, 1, mode);
   if (value == AGX_APPLE9_VREG_INVALID || !apple9_vir_append_values(program, 1))
      return AGX_APPLE9_VREG_INVALID;
   program->instructions[program->instruction_count - 1]->dest_components = 2;
   return value;
}

uint32_t
agx_apple9_vir_emit_mul_wide(struct agx_apple9_vir_program *program,
                            const uint32_t src[2], bool is_signed)
{
   uint32_t value = agx_apple9_vir_emit(
      program, AGX_APPLE9_VIR_IMUL_WIDE, AGX_APPLE9_ENC_INT_MUL_WIDE,
      src, 2, is_signed);
   if (value == AGX_APPLE9_VREG_INVALID ||
       !apple9_vir_append_values(program, 1))
      return AGX_APPLE9_VREG_INVALID;
   program->instructions[program->instruction_count - 1]->dest_components = 2;
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

   program->instructions[program->instruction_count - 1]->dest_components = components;
   program->instructions[program->instruction_count - 1]->memory_index_shift =
      components == 2 ? 3 : 4;
   return value;
}

static uint32_t
apple9_vir_emit_texture(struct agx_apple9_vir_program *program,
                        const uint32_t coords[2], uint32_t one,
                        unsigned texture, unsigned sampler, bool lod, bool bias)
{
   if (!program || !coords || coords[0] >= program->value_count ||
       coords[1] >= program->value_count || one >= program->value_count ||
       texture >= 16 || sampler >= AGX_APPLE9_GRAPHICS_MAX_SAMPLERS)
      return AGX_APPLE9_VREG_INVALID;

   unsigned parameters = lod ? 4 : 2;
   uint32_t published = program->value_count;
   if (!apple9_vir_append_values(program, parameters))
      return AGX_APPLE9_VREG_INVALID;
   struct agx_apple9_vir_instr *instruction = apple9_vir_append_instruction(program);
   if (!instruction)
      return AGX_APPLE9_VREG_INVALID;
   *instruction = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_PUBLICATION_TUPLE,
      .encoding = lod ? AGX_APPLE9_ENC_TEXTURE_LOD_PARAMS
                        : AGX_APPLE9_ENC_TEXTURE_COORDS,
      .dest = published, .dest_components = parameters,
      .nr_srcs = 3, .src = {coords[0], coords[1], one},
   };

   uint32_t result = program->value_count;
   if (!apple9_vir_append_values(program, 4))
      return AGX_APPLE9_VREG_INVALID;
   instruction = apple9_vir_append_instruction(program);
   if (!instruction)
      return AGX_APPLE9_VREG_INVALID;
   *instruction = (struct agx_apple9_vir_instr){
      .texture_index = texture, .sampler_index = sampler,
      .op = AGX_APPLE9_VIR_TEXTURE_SAMPLE,
      .encoding = lod ? AGX_APPLE9_ENC_TEXTURE_LOD
                        : AGX_APPLE9_ENC_TEXTURE_SAMPLE,
      .immediate = bias,
      .dest = result, .dest_components = 4,
      .nr_srcs = parameters,
      .src = {published, published + 1, published + 2, published + 3},
      /* Default for standalone VIR allocation. The shared allocator assigns
       * completion tags after physical register and publication allocation. */
      .producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1,
   };
   return result;
}

uint32_t
agx_apple9_vir_emit_publication_pair(struct agx_apple9_vir_program *program,
                                     const uint32_t src[2])
{
   if (!program || !src || src[0] >= program->value_count || src[1] >= program->value_count)
      return AGX_APPLE9_VREG_INVALID;
   uint32_t result = program->value_count;
   if (!apple9_vir_append_values(program, 2))
      return AGX_APPLE9_VREG_INVALID;
   struct agx_apple9_vir_instr *ins = apple9_vir_append_instruction(program);
   if (!ins)
      return AGX_APPLE9_VREG_INVALID;
   *ins = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_PUBLICATION_TUPLE,
      .encoding = AGX_APPLE9_ENC_PUBLICATION_PAIR,
      .dest = result, .dest_components = 2,
      .src = {src[0], src[1]}, .nr_srcs = 2,
   };
   return result;
}

/* Publish a coordinate/byte-offset tuple, then export the implicit tile.
 * The synchronous store contract releases its publication and joins the
 * memory operation before the next VIR instruction. */
bool
agx_apple9_vir_emit_block_image_store(
   struct agx_apple9_vir_program *program, const uint32_t src[3],
   unsigned image, unsigned format, bool multisampled)
{
   if (image >= 16 || format > 15)
      return false;
   for (unsigned i = 0; i < 3; ++i)
      if (src[i] >= program->value_count)
         return false;
   uint32_t fields[] = {src[0], src[1], src[2], src[2]};
   unsigned count = multisampled ? 4 : 3;
   if (multisampled) {
      fields[2] = agx_apple9_vir_emit(program, AGX_APPLE9_VIR_IMM,
         AGX_APPLE9_ENC_MOV_IMM_COMPACT, NULL, 0, 0);
      if (fields[2] == AGX_APPLE9_VREG_INVALID)
         return false;
   }
   uint32_t tuple = program->value_count;
   if (!apple9_vir_append_values(program, 4))
      return false;
   struct agx_apple9_vir_instr *ins = apple9_vir_append_instruction(program);
   if (!ins)
      return false;
   *ins = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_PUBLICATION_TUPLE,
      .encoding = multisampled ? AGX_APPLE9_ENC_BLOCK_STORE_MS_PARAMS
                               : AGX_APPLE9_ENC_BLOCK_STORE_PARAMS,
      .dest = tuple, .dest_components = 4,
      .nr_srcs = count,
      .src = {fields[0], fields[1], fields[2], fields[3]},
   };
   uint32_t sources[] = {tuple, tuple + 1, tuple + 2, tuple + 3};
   if (!agx_apple9_vir_emit_side_effect(program,
          AGX_APPLE9_VIR_BLOCK_IMAGE_STORE,
          multisampled ? AGX_APPLE9_ENC_BLOCK_IMAGE_STORE_MS
                       : AGX_APPLE9_ENC_BLOCK_IMAGE_STORE,
          sources, count, format))
      return false;
   program->instructions[program->instruction_count - 1]->texture_index = image;
   return true;
}

uint32_t
agx_apple9_vir_emit_texture_sample(struct agx_apple9_vir_program *program,
                                  const uint32_t coords[2], uint32_t one,
                                  unsigned texture, unsigned sampler)
{
   return apple9_vir_emit_texture(program, coords, one, texture, sampler, false, false);
}

uint32_t
agx_apple9_vir_emit_texture_lod(struct agx_apple9_vir_program *program,
                               const uint32_t coords[2], uint32_t packed_lod,
                               unsigned texture, unsigned sampler, bool bias)
{
   return apple9_vir_emit_texture(program, coords, packed_lod, texture, sampler,
                                  true, bias);
}

uint32_t
agx_apple9_vir_emit_texture_volume(struct agx_apple9_vir_program *program,
                                  const uint32_t coords[3], uint32_t packed_lod,
                                  unsigned texture, unsigned sampler,
                                  unsigned dimension, bool bias, bool shadow)
{
   if (!program || !coords || coords[2] >= program->value_count ||
       (shadow ? dimension > 2 : (dimension != 1 && dimension != 2 && dimension != 3)))
      return AGX_APPLE9_VREG_INVALID;
   uint32_t result = apple9_vir_emit_texture(program, coords, packed_lod,
      texture, sampler, true, bias);
   if (result == AGX_APPLE9_VREG_INVALID)
      return result;
   struct agx_apple9_vir_instr *params = program->instructions[program->instruction_count - 2];
   params->encoding = AGX_APPLE9_ENC_TEXTURE_VOLUME_PARAMS;
   params->nr_srcs = 4;
   params->src[2] = coords[2];
   params->src[3] = packed_lod;
   program->instructions[program->instruction_count - 1]->texture_dimension = dimension;
   program->instructions[program->instruction_count - 1]->texture_shadow = shadow;
   return result;
}

uint32_t
agx_apple9_vir_emit_texture_grad(struct agx_apple9_vir_program *program,
                                const uint32_t src[6],
                                unsigned texture, unsigned sampler)
{
   if (!program || !src || texture >= 16 || sampler >= AGX_APPLE9_GRAPHICS_MAX_SAMPLERS)
      return AGX_APPLE9_VREG_INVALID;
   for (unsigned i = 0; i < 6; ++i)
      if (src[i] >= program->value_count)
         return AGX_APPLE9_VREG_INVALID;
   uint32_t published = program->value_count;
   if (!apple9_vir_append_values(program, 8))
      return AGX_APPLE9_VREG_INVALID;
   struct agx_apple9_vir_instr *instruction = apple9_vir_append_instruction(program);
   if (!instruction)
      return AGX_APPLE9_VREG_INVALID;
   *instruction = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_PUBLICATION_TUPLE,
      .encoding = AGX_APPLE9_ENC_TEXTURE_GRAD_PARAMS,
      .dest = published, .dest_components = 8, .nr_srcs = 6,
   };
   memcpy(instruction->src, src, 6 * sizeof(src[0]));
   uint32_t result = program->value_count;
   if (!apple9_vir_append_values(program, 4))
      return AGX_APPLE9_VREG_INVALID;
   instruction = apple9_vir_append_instruction(program);
   if (!instruction)
      return AGX_APPLE9_VREG_INVALID;
   *instruction = (struct agx_apple9_vir_instr){
      .op = AGX_APPLE9_VIR_TEXTURE_SAMPLE,
      .encoding = AGX_APPLE9_ENC_TEXTURE_GRAD,
      .texture_index = texture, .sampler_index = sampler,
      .dest = result, .dest_components = 4,
      .nr_srcs = 1, .src = {published},
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
agx_apple9_vir_emit_phi(struct agx_apple9_vir_program *program,
                         struct agx_apple9_block *block)
{
   if (block && block->program != program)
      return AGX_APPLE9_VREG_INVALID;
   uint32_t value = agx_apple9_vir_emit(program, AGX_APPLE9_VIR_PHI,
                                        AGX_APPLE9_ENC_PSEUDO, NULL, 0, 0);
   if (value != AGX_APPLE9_VREG_INVALID)
      program->instructions[program->instruction_count - 1]->phi_block =
         block ? block : program->current_block;
   return value;
}

bool
agx_apple9_vir_emit_phi_source(struct agx_apple9_vir_program *program,
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
      .op = AGX_APPLE9_VIR_PHI_SRC,
      .encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED,
      .dest = AGX_APPLE9_VREG_INVALID,
      .src = {source},
      .target = target,
      .nr_srcs = 1,
      .phi_edge = instruction,
   };
   return true;
}

/* Record edge uses without ordering writes or introducing virtual copies.
 * phi_edge identifies the final use record, where the simultaneous assignment
 * executes under this predecessor's lane mask after register allocation. */
bool
agx_apple9_vir_emit_phi_edge(struct agx_apple9_vir_program *program,
   const struct agx_apple9_vir_copy *copies, unsigned count)
{
   unsigned first = program->instruction_count;
   for (unsigned i = 0; i < count; ++i) {
      if (copies[i].source >= program->value_count || copies[i].target >= program->value_count)
         return false;
      for (unsigned j = 0; j < i; ++j) {
         if (copies[i].target == copies[j].target)
            return false;
      }
   }
   for (unsigned i = 0; i < count; ++i) {
      if (!agx_apple9_vir_emit_phi_source(program, copies[i].target, copies[i].source))
         return false;
   }
   if (count) {
      struct agx_apple9_vir_instr *edge = program->instructions[program->instruction_count - 1];
      for (unsigned i = first; i < program->instruction_count; ++i)
         program->instructions[i]->phi_edge = edge;
   }
   return true;
}

void
agx_apple9_place_phis(struct agx_apple9_vir_program *program)
{
   /* Selection may encounter an incoming edge before the successor. The
    * declaration is created early for SSA lookup, then attached to its block
    * once structured selection has placed every block in layout order. */
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->op != AGX_APPLE9_VIR_PHI || !ins->phi_block)
         continue;
      assert(ins->phi_block->placed);
      struct apple9_instruction_node *node = apple9_node(ins);
      list_del(&node->link);
      list_add(&node->link, &ins->phi_block->instructions);
      node->block = ins->phi_block;
   }
   agx_apple9_vir_reindex(program);
}

bool
agx_apple9_resolve_phi_edge(const struct agx_apple9_vir_program *program,
   const struct agx_apple9_vir_instr *edge,
   bool (*emit)(void *data, bool swap, unsigned dest, unsigned source), void *data)
{
   if (edge->phi_edge != edge)
      return true;
   agx_context *ctx = rzalloc(NULL, agx_context);
   if (!ctx)
      return false;
   agx_block block = {0};
   list_inithead(&block.instructions);
   agx_builder builder = {.shader = ctx, .cursor = agx_after_block(&block)};
   struct agx_copy *copies = rzalloc_array(ctx, struct agx_copy, program->instruction_count);
   if (!copies) {
      ralloc_free(ctx);
      return false;
   }
   unsigned count = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->op == AGX_APPLE9_VIR_PHI_SRC && ins->phi_edge == edge) {
         copies[count++] = (struct agx_copy) {
            .dest = program->phys[ins->target] * 2,
            .src = agx_register(program->phys[ins->src[0]] * 2, AGX_SIZE_32),
         };
      }
   }
   agx_emit_parallel_copies(&builder, copies, count);
   bool ok = true;
   agx_foreach_instr_in_block(&block, ins) {
      assert(ins->op == AGX_OPCODE_MOV || ins->op == AGX_OPCODE_SWAP);
      assert(ins->dest[0].size == AGX_SIZE_32 && ins->src[0].size == AGX_SIZE_32);
      /* SWAP's second source is the other register; source zero is dest. */
      unsigned source = ins->src[ins->op == AGX_OPCODE_SWAP ? 1 : 0].value / 2;
      if (!emit(data, ins->op == AGX_OPCODE_SWAP, ins->dest[0].value / 2, source)) {
         ok = false;
         break;
      }
   }
   ralloc_free(ctx);
   return ok;
}

static bool
apple9_vir_values_form_tuple(const struct agx_apple9_vir_program *program,
                             const uint32_t *values, unsigned components)
{
   if (components < 2 || components > 4)
      return false;

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         program->instructions[i];
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
      while (program->instruction_count > old_instruction_count)
         agx_apple9_vir_remove(program, program->instructions[program->instruction_count - 1]);
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
      .memory_index_shift = util_logbase2(bits / 8) + (components == 1 ? 0 : components == 2 ? 1 : 2),
      .immediate = binding,
      .nr_srcs = components + 1,
   };
   for (unsigned c = 0; c < components; ++c)
      instruction->src[c] =
         tuple == AGX_APPLE9_VREG_INVALID ? data[c] : tuple + c;
   instruction->src[components] = index;
   return true;
}

bool
agx_apple9_vir_set_device_store_address(struct agx_apple9_vir_program *program,
                                        uint32_t address)
{
   if (!program || !program->instruction_count ||
       address >= program->value_count || address + 1 >= program->value_count)
      return false;
   struct agx_apple9_vir_instr *ins =
      program->instructions[program->instruction_count - 1];
   if (ins->op != AGX_APPLE9_VIR_DEVICE_STORE ||
       ins->encoding != AGX_APPLE9_ENC_DEVICE_STORE)
      return false;
   ins->encoding = AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
   ins->src[ins->nr_srcs++] = address;
   ins->src[ins->nr_srcs++] = address + 1;
   agx_apple9_invalidate_uses(program);
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
      while (program->instruction_count > old_instruction_count)
         agx_apple9_vir_remove(program, program->instructions[program->instruction_count - 1]);
      program->value_count = old_value_count;
      return false;
   }

   struct agx_apple9_vir_instr *instruction =
      program->instructions[program->instruction_count - 1];
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
      while (program->instruction_count > old_instruction_count)
         agx_apple9_vir_remove(program, program->instructions[program->instruction_count - 1]);
      program->value_count = old_value_count;
      return false;
   }

   if (!discard_result) {
      struct agx_apple9_vir_instr *publication =
         program->instructions[program->instruction_count - 1];
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
      if (flags & ~AGX_APPLE9_DEVICE_LOAD_HAS_NEXT)
         return false;

      for (unsigned i = 0; i < program->instruction_count; ++i) {
         struct agx_apple9_vir_instr *instruction = program->instructions[i];
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
   if ((flags & ~AGX_APPLE9_DEVICE_LOAD_HAS_NEXT) ||
       !apple9_scalar_load_slot_for_token(raw_token, &scoreboard_slot))
      return false;

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *instruction = program->instructions[i];
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
agx_apple9_vir_set_load_address(struct agx_apple9_vir_program *program,
                                uint32_t value, uint32_t address)
{
   if (address >= program->value_count || address + 1 >= program->value_count)
      return false;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *load = program->instructions[i];
      if (load->dest != value || load->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         continue;
      if (load->encoding != AGX_APPLE9_ENC_DEVICE_LOAD || load->nr_srcs != 1)
         return false;
      agx_apple9_invalidate_uses(program);
      load->encoding = AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT;
      load->nr_srcs = 3;
      load->src[1] = address;
      load->src[2] = address + 1;
      load->immediate = 0;
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
      struct agx_apple9_vir_instr *instruction = program->instructions[i];
      if (instruction->dest != value)
         continue;
      if (instruction->op != AGX_APPLE9_VIR_DEVICE_LOAD)
         return false;
      if ((instruction->encoding != AGX_APPLE9_ENC_DEVICE_LOAD &&
           instruction->encoding != AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT) ||
          (instruction->nr_srcs != 1 && instruction->nr_srcs != 3)) {
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
      const bool indirect =
         instruction->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
      if (components < 1 || components > 4 ||
          instruction->nr_srcs != components + (indirect ? 3 : 1))
         return false;

      /* Machine-table order is address then data base.  COLLECT has already
       * made the remaining vector lanes adjacent before allocation. */
      gprs[0] = phys[instruction->src[components]];
      gprs[1] = phys[instruction->src[0]];
      *count = indirect ? 4 : 2;
      if (indirect) {
         gprs[2] = phys[instruction->src[components + 1]];
         gprs[3] = phys[instruction->src[components + 2]];
      }
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

bool
agx_apple9_analyze_uses(struct agx_apple9_vir_program *program)
{
   if (program->use_analysis && program->use_analysis->valid)
      return true;
   agx_apple9_vir_reindex(program);
   struct agx_apple9_use_analysis *a = calloc(1, sizeof(*a));
   if (!a)
      return false;
   unsigned values = MAX2(program->value_count, 1), uses = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i)
      uses += program->instructions[i]->nr_srcs;
   a->definitions = calloc(values, sizeof(*a->definitions));
   a->first = calloc(values, sizeof(*a->first));
   a->last = calloc(values, sizeof(*a->last));
   a->uses = calloc(MAX2(uses, 1), sizeof(*a->uses));
   if (!a->definitions || !a->first || !a->last || !a->uses)
      goto fail;
   unsigned cursor = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      for (unsigned c = 0; c < apple9_vir_dest_components(ins); ++c) {
         if (ins->dest + c >= program->value_count || a->definitions[ins->dest + c])
            goto fail;
         a->definitions[ins->dest + c] = ins;
      }
      for (unsigned src = 0; src < ins->nr_srcs; ++src) {
         uint32_t value = ins->src[src];
         if (value >= program->value_count)
            goto fail;
         struct agx_apple9_use *use = &a->uses[cursor++];
         *use = (struct agx_apple9_use){.instruction = ins, .source = src};
         if (a->last[value])
            a->last[value]->next = use;
         else
            a->first[value] = use;
         a->last[value] = use;
      }
   }
   a->valid = true;
   apple9_free_use_analysis(program->use_analysis);
   program->use_analysis = a;
   return true;
fail:
   apple9_free_use_analysis(a);
   return false;
}

const struct agx_apple9_vir_instr *
agx_apple9_definition(const struct agx_apple9_vir_program *program, uint32_t value)
{
   if (value >= program->value_count ||
       !agx_apple9_analyze_uses((struct agx_apple9_vir_program *)program))
      return NULL;
   return program->use_analysis->definitions[value];
}

const struct agx_apple9_use *
agx_apple9_uses(const struct agx_apple9_vir_program *program, uint32_t value)
{
   if (value >= program->value_count ||
       !agx_apple9_analyze_uses((struct agx_apple9_vir_program *)program))
      return NULL;
   return program->use_analysis->first[value];
}

/* Final dependency decisions follow all operand/layout legalization and CFG
 * liveness. Only physical expansion and byte packing may follow this point. */
static bool
apple9_finalize_dependencies(struct agx_apple9_vir_program *program)
{
   if (!agx_apple9_analyze_uses(program))
      return false;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->op == AGX_APPLE9_VIR_DEVICE_LOAD) {
         ins->device_load_index_kind = (ins->live_after_mask & 1)
            ? AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR
            : AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR;
      }
      if (ins->encoding == AGX_APPLE9_ENC_FLOAT_SPECIAL) {
         /* Bit zero's high-pressure meaning remains unproven; retain the
          * observed memory-only versus ALU-consumed result distinction. */
         ins->immediate = 2;
         for (const struct agx_apple9_use *use = agx_apple9_uses(program, ins->dest);
              use; use = use->next) {
            if (use->instruction->op != AGX_APPLE9_VIR_DEVICE_STORE) {
               ins->immediate = 3;
               break;
            }
         }
      }
   }
   program->dependencies_finalized = true;
   return true;
}

static const struct agx_apple9_vir_instr *
apple9_vir_producer_instruction(const struct agx_apple9_vir_program *program,
                                uint32_t value);

static bool
apple9_vir_is_graphics_output(enum agx_apple9_vir_opcode op)
{
   return op == AGX_APPLE9_VIR_COVERAGE || op == AGX_APPLE9_VIR_DEPTH_STORE || op == AGX_APPLE9_VIR_VARY_STORE ||
          op == AGX_APPLE9_VIR_TILE_ACCESS ||
          op == AGX_APPLE9_VIR_TILE_STORE || op == AGX_APPLE9_VIR_TILE_FENCE ||
          op == AGX_APPLE9_VIR_BLOCK_IMAGE_STORE;
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

/* The publication selector changes the destination namespace, not its
 * numeric encoding. Varying exports and texture parameters share this file. */
static bool
apple9_vir_publishes(const struct agx_apple9_vir_instr *instruction)
{
   return instruction->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
          instruction->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT ||
          instruction->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE;
}

static unsigned
apple9_first_consumer(const struct agx_apple9_vir_program *program,
                      unsigned producer_index);

/* A texture bundle describes a pending operation. Its parameters remain live
 * through the first result consumer, which performs the scoreboard handoff.
 * Rewriting them between the bundle and that consumer changes the sample.
 * Varying exports remain published until shader completion. */
static unsigned
apple9_publication_end(const struct agx_apple9_vir_program *program,
                       unsigned producer)
{
   const struct agx_apple9_vir_instr *pub = program->instructions[producer];
   if (pub->op != AGX_APPLE9_VIR_PUBLICATION_TUPLE &&
       pub->encoding != AGX_APPLE9_ENC_LOGIC_EXPORT)
      return program->instruction_count;

   unsigned end = producer;
   for (unsigned c = 0; c < pub->dest_components; ++c) {
      for (const struct agx_apple9_use *use = agx_apple9_uses(program, pub->dest + c);
           use; use = use->next) {
         const struct agx_apple9_vir_instr *sample = use->instruction;
         unsigned position = apple9_node(sample)->position;
         if (sample->op == AGX_APPLE9_VIR_COVERAGE || sample->op == AGX_APPLE9_VIR_DEPTH_STORE ||
             sample->op == AGX_APPLE9_VIR_BLOCK_IMAGE_STORE) {
            end = MAX2(end, position);
            continue;
         }
         if (sample->op != AGX_APPLE9_VIR_TEXTURE_SAMPLE)
            return program->instruction_count;
         unsigned handoff = apple9_first_consumer(program, position);
         end = MAX2(end, handoff == UINT_MAX ? program->instruction_count : handoff);
      }
   }

   /* Publication storage is a separate register file, but its cross-block
    * lifetime follows the same CFG as ordinary values. */
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      unsigned boundary = block->next ? block->next->start_index :
                                         program->instruction_count;
      if (boundary > block->start_index && block->live_out &&
          BITSET_TEST(block->live_out, pub->dest))
         end = MAX2(end, boundary - 1);
   }
   return end;
}

static bool
apple9_validate_value_files(const struct agx_apple9_vir_program *program,
                           const char **reason)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->dest < program->value_count &&
          program->publication[ins->dest] != apple9_vir_publishes(ins))
         goto invalid;
      if (apple9_vir_publishes(ins) &&
          ((apple9_vir_dest_components(ins) != 1 &&
             ins->op != AGX_APPLE9_VIR_PUBLICATION_TUPLE) ||
           ins->dest >= program->value_count ||
           program->fixed_phys[ins->dest] != AGX_APPLE9_PHYS_INVALID))
         goto invalid;
      for (unsigned j = 0; j < ins->nr_srcs; ++j) {
         if (ins->src[j] >= program->value_count ||
             program->publication[ins->src[j]] !=
                (ins->op == AGX_APPLE9_VIR_COVERAGE || ins->op == AGX_APPLE9_VIR_DEPTH_STORE ||
                 ins->op == AGX_APPLE9_VIR_VARY_STORE ||
                 ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ||
                 ins->op == AGX_APPLE9_VIR_BLOCK_IMAGE_STORE))
            goto invalid;
      }
      if (ins->op == AGX_APPLE9_VIR_PHI_SRC &&
          ins->target < program->value_count &&
          program->publication[ins->target])
         goto invalid;
   }
   if (program->output < program->value_count &&
       program->publication[program->output])
      goto invalid;
   return true;
invalid:
   if (reason)
      *reason = "Apple9 value uses the wrong GPR/publication namespace";
   return false;
}

bool
agx_apple9_validate_vir_allocation(const struct agx_apple9_vir_program *program,
                                   const char **reason)
{
   if (reason != NULL)
      *reason = NULL;
   if (program->phys == NULL || program->publication == NULL) {
      if (reason != NULL)
         *reason = "Apple9 virtual IR has not been allocated";
      return false;
   }

   if (!program->dependencies_finalized) {
      if (reason)
         *reason = "Apple9 dependency metadata is stale after an IR edit";
      return false;
   }

   if (!apple9_validate_value_files(program, reason))
      return false;

   unsigned publication_end[AGX_APPLE9_PUBLICATION_COUNT] = {0};
   bool publication_used[AGX_APPLE9_PUBLICATION_COUNT] = {false};
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (!apple9_vir_publishes(ins))
         continue;
      unsigned end = apple9_publication_end(program, i);
      for (unsigned c = 0; c < apple9_vir_dest_components(ins); ++c) {
         unsigned index = program->phys[ins->dest + c];
         if (index >= AGX_APPLE9_PUBLICATION_COUNT ||
             (publication_used[index] && publication_end[index] >= i)) {
            if (reason)
               *reason = "Apple9 publication slots overlap or exceed the encoding";
            return false;
         }
         publication_used[index] = true;
         publication_end[index] = end;
      }
   }

   unsigned execution_mask_depth = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         program->instructions[i];
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

      if (instruction->op == AGX_APPLE9_VIR_STORE_UNIFORM) {
         struct agx_apple9_packed_instruction packed;
         if (instruction->dest != AGX_APPLE9_VREG_INVALID ||
             !agx_apple9_pack_vir_instruction(instruction, program->phys, &packed, reason))
            return false;
         continue;
      }
      if (instruction->op == AGX_APPLE9_VIR_SPILL_STORE ||
          instruction->op == AGX_APPLE9_VIR_SPILL_LOAD) {
         struct agx_apple9_packed_instruction packed;
         if (instruction->immediate >= program->scratch_size / 4 ||
             !agx_apple9_pack_vir_instruction(instruction, program->phys, &packed, reason)) {
            if (reason) *reason = "Apple9 spill exceeds its frame or has an invalid dependency";
            return false;
         }
         continue;
      }
      if (instruction->op == AGX_APPLE9_VIR_DEVICE_STORE) {
         const unsigned components = instruction->memory_components;
         const bool indirect =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
         const unsigned bits = instruction->memory_bits;
         if (instruction->dest != AGX_APPLE9_VREG_INVALID || components < 1 ||
             components > 4 ||
             instruction->nr_srcs != components + (indirect ? 3 : 1) ||
             (bits != 8 && bits != 16 && bits != 32) ||
             (components > 1 && bits != 32) ||
             instruction->immediate > UINT8_MAX) {
            if (reason != NULL)
               *reason = "Apple9 device store has an invalid VIR contract";
            return false;
         }

         if ((instruction->encoding != AGX_APPLE9_ENC_DEVICE_STORE &&
              !indirect)) {
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

         if (indirect) {
            unsigned address = program->phys[instruction->src[components + 1]];
            unsigned high = program->phys[instruction->src[components + 2]];
            if (!agx_apple9_encoding_accepts_gpr(instruction->encoding,
                                                 AGX_APPLE9_OPERAND_SRC1,
                                                 address, 32) ||
                !agx_apple9_encoding_accepts_gpr(
                   instruction->encoding, AGX_APPLE9_OPERAND_SRC2, high, 32) ||
                high != address + 1) {
               if (reason)
                  *reason =
                     "Apple9 indirect store requires an adjacent address pair";
               return false;
            }
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
            program->instructions[i - 1]->op ==
               AGX_APPLE9_VIR_DEVICE_ATOMIC &&
            !program->instructions[i - 1]->atomic_discard &&
            program->instructions[i - 1]->producer_scoreboard_slot >=
               AGX_APPLE9_SCOREBOARD_SLOT_1 &&
            program->instructions[i - 1]->producer_scoreboard_slot <=
               AGX_APPLE9_SCOREBOARD_SLOT_6 &&
            instruction->producer_scoreboard_slot ==
               program->instructions[i - 1]->producer_scoreboard_slot &&
            program->instructions[i - 1]->dest == instruction->src[0] &&
            program->phys[instruction->src[0]] < 64;
         if (!valid) {
            if (reason != NULL)
               *reason =
                  "Apple9 atomic result materializer has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_PHI) {
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
               *reason = "Apple9 PHI has an invalid VIR contract";
            return false;
         }
         continue;
      }

      if (instruction->op == AGX_APPLE9_VIR_PHI_SRC) {
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
            target_producer->op == AGX_APPLE9_VIR_PHI &&
            instruction->phi_edge != NULL &&
            instruction->phi_edge->op == AGX_APPLE9_VIR_PHI_SRC &&
            agx_apple9_instr_block(instruction) ==
               agx_apple9_instr_block(instruction->phi_edge) &&
            instruction->immediate == 0 && instruction->memory_bits == 0 &&
            instruction->memory_components == 0 &&
            instruction->producer_scoreboard_slot ==
               AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
            agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                  ARRAY_SIZE(tuple), 32);
         if (!valid) {
            if (reason != NULL)
               *reason = "Apple9 phi incoming edge has an invalid VIR contract";
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
                     instruction->branch_target && instruction->branch_target->placed;
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

      if (instruction->op == AGX_APPLE9_VIR_COVERAGE || instruction->op == AGX_APPLE9_VIR_DEPTH_STORE ||
          instruction->op == AGX_APPLE9_VIR_VARY_STORE ||
          instruction->op == AGX_APPLE9_VIR_TILE_ACCESS ||
          instruction->op == AGX_APPLE9_VIR_TILE_STORE ||
          instruction->op == AGX_APPLE9_VIR_TILE_FENCE ||
          instruction->op == AGX_APPLE9_VIR_BLOCK_IMAGE_STORE) {
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
      if (components > (instruction->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE ? 8 : 4) ||
          components > program->value_count ||
          instruction->dest > program->value_count - components ||
          (components > 1 && instruction->op != AGX_APPLE9_VIR_DEVICE_LOAD &&
           instruction->op != AGX_APPLE9_VIR_UNPACK_NORM &&
           instruction->op != AGX_APPLE9_VIR_ITER_FLAT &&
           instruction->op != AGX_APPLE9_VIR_COLLECT &&
           instruction->op != AGX_APPLE9_VIR_PUBLICATION_TUPLE &&
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

      /* Inline uniform/immediate operands have their own file and range.
       * The float packers validate those alongside the compact GPR list. */
      if (instruction->alu_src_uniform_mask || instruction->alu_src_immediate_mask) {
         struct agx_apple9_packed_instruction packed;
         if (!agx_apple9_pack_vir_instruction(instruction, program->phys,
                                             &packed, reason))
            return false;
         continue;
      }
      unsigned tuple[AGX_APPLE9_MAX_ENCODING_OPERANDS] = {0};
      unsigned tuple_count = 0;
      if (!encoding_tuple(instruction, program->phys, tuple, &tuple_count) ||
          !agx_apple9_encoding_accepts_gpr_tuple(instruction->encoding, tuple,
                                                 tuple_count, 32)) {
         if (agx_apple9_trace_enabled()) {
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
         const bool indirect =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT;
         if (instruction->nr_srcs != (indirect ? 3 : 1) ||
             (indirect && program->phys[instruction->src[2]] !=
                             program->phys[instruction->src[1]] + 1)) {
            if (reason)
               *reason = "Apple9 device-load address must be an adjacent pair";
            return false;
         }
         if ((instruction->nr_srcs != 1 && instruction->nr_srcs != 3) ||
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
         program->instructions[i];
      if (instruction->op == AGX_APPLE9_VIR_PHI_SRC) {
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
         const bool indirect =
            instruction->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
         if (components < 1 || components > 4 ||
             instruction->nr_srcs != components + (indirect ? 3 : 1))
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
         if (indirect) {
            for (unsigned c = 0; c < 2; ++c) {
               uint32_t address = instruction->src[components + 1 + c];
               if (address >= program->value_count)
                  goto invalid;
               if (program->max_phys[address] == AGX_APPLE9_PHYS_INVALID ||
                   program->max_phys[address] > 62 + c)
                  program->max_phys[address] = 62 + c;
            }
         }
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
             agx_apple9_inline_source_file(instruction, constraint->role) ||
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

static bool
apple9_materialize_pending_result_at(struct agx_apple9_vir_program *program,
                                     unsigned producer_index,
                                     unsigned insertion, const char **reason);

bool
agx_apple9_allocate_publications(struct agx_apple9_vir_program *program,
                                 const char **reason)
{
retry:
   free(program->publication);
   free(program->phys);
   program->publication = calloc(MAX2(program->value_count, 1), sizeof(bool));
   program->phys = malloc(MAX2(program->value_count, 1));
   if (!program->publication || !program->phys ||
       !agx_apple9_analyze_uses(program))
      return false;
   memset(program->phys, AGX_APPLE9_PHYS_INVALID, program->value_count);
   program->publication_count = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (apple9_vir_publishes(ins)) {
         for (unsigned c = 0; c < ins->dest_components; ++c)
            program->publication[ins->dest + c] = true;
      }
   }
   if (!apple9_validate_value_files(program, reason) ||
       !apple9_propagate_source_register_classes(program, reason))
      return false;
   unsigned end[AGX_APPLE9_PUBLICATION_COUNT] = {0};
   bool used[AGX_APPLE9_PUBLICATION_COUNT] = {false};
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (!apple9_vir_publishes(ins))
         continue;
      unsigned width = ins->dest_components;
      unsigned align = MAX2(agx_apple9_encoding_info(ins->encoding)
                              ->operands[0].alignment_halves / 2, 1);
      unsigned slot;
      for (slot = 0; slot + width <= AGX_APPLE9_PUBLICATION_COUNT; slot += align) {
         bool available = true;
         for (unsigned c = 0; c < width; ++c) {
            unsigned max = program->max_phys[ins->dest + c];
            available &= (!used[slot + c] || end[slot + c] < i) &&
               (max == AGX_APPLE9_PHYS_INVALID || slot + c <= max);
         }
         if (available)
            break;
      }
      if (slot + width > AGX_APPLE9_PUBLICATION_COUNT) {
         /* A texture borrows its parameter publication until its result
          * handoff. Under pressure, consume an outstanding result in the
          * producer's block, then retry with the shortened lifetime. The
          * explicit copies must survive shared-IR copy propagation and DCE.
          * Each repair consumes one previously pending result, so retries
          * terminate even when the pressure cannot be relieved. */
         for (unsigned t = 0; t < i; ++t) {
            const struct agx_apple9_vir_instr *sample = program->instructions[t];
            if (sample->op != AGX_APPLE9_VIR_TEXTURE_SAMPLE ||
                apple9_first_consumer(program, t) < i)
               continue;
            bool borrows_publication = false;
            for (unsigned s = 0; s < sample->nr_srcs; ++s)
               borrows_publication |= program->publication[sample->src[s]];
            if (!borrows_publication)
               continue;
            if (!apple9_materialize_pending_result_at(program, t, t + 1, reason))
               return false;
            /* One result read completes the entire tuple. Keep that handoff
             * even if its scalar result is dead; ordinary identity copies
             * for the other components can coalesce through shared SSA. */
            program->instructions[t + 1]->publication_handoff = true;
            goto retry;
         }
         *reason = "Apple9 allocator exhausted export publication slots";
         return false;
      }
      unsigned last = apple9_publication_end(program, i);
      for (unsigned c = 0; c < width; ++c) {
         program->phys[ins->dest + c] = slot + c;
         used[slot + c] = true;
         end[slot + c] = last;
      }
      program->publication_count = MAX2(program->publication_count, slot + width);
   }
   return true;
}

static unsigned
apple9_vir_definition_index(const struct agx_apple9_vir_program *program,
                            uint32_t value)
{
   const struct agx_apple9_vir_instr *definition = agx_apple9_definition(program, value);
   return definition ? apple9_node(definition)->position : UINT_MAX;
}

/* Build the machine CFG from stable block targets. Masked if/else regions
 * execute in layout order; explicit branches also have a fallthrough edge.
 * Split at each branch so instructions after an early loop exit belong to
 * its fallthrough, rather than incorrectly killing values on the exit edge. */
static bool
apple9_build_cfg(struct agx_apple9_vir_program *program)
{
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      list_for_each_entry(struct apple9_instruction_node, node, &block->instructions, link) {
         if (node->instruction.op != AGX_APPLE9_VIR_JMP_EXEC_ANY &&
             node->instruction.op != AGX_APPLE9_VIR_JMP_EXEC_NONE)
            continue;
         if (node->link.next == &block->instructions)
            break;
         struct agx_apple9_block *tail = agx_apple9_block_create(program);
         if (!tail)
            return false;
         tail->placed = true;
         tail->prev = block;
         tail->next = block->next;
         if (tail->next)
            tail->next->prev = tail;
         else
            program->last_block = tail;
         block->next = tail;
         while (node->link.next != &block->instructions) {
            struct apple9_instruction_node *next =
               list_entry(node->link.next, struct apple9_instruction_node, link);
            list_del(&next->link);
            list_addtail(&next->link, &tail->instructions);
            next->block = tail;
         }
         break;
      }
   }
   agx_apple9_vir_reindex(program);
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      block->successors[0] = block->next;
      block->successors[1] = NULL;
      if (!list_is_empty(&block->instructions)) {
         struct apple9_instruction_node *last =
            list_last_entry(&block->instructions, struct apple9_instruction_node, link);
         if (last->instruction.op == AGX_APPLE9_VIR_JMP_EXEC_ANY ||
             last->instruction.op == AGX_APPLE9_VIR_JMP_EXEC_NONE) {
            if (!last->instruction.branch_target || !last->instruction.branch_target->placed ||
                last->instruction.branch_target->program != program)
               return false;
            block->successors[1] = last->instruction.branch_target;
         }
      }
   }
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      free(b->predecessors);
      b->predecessors = NULL;
      b->predecessor_count = 0;
   }
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      if (b->program != program || !b->placed ||
          (b->next && b->next->prev != b))
         return false;
      for (unsigned i = 0; i < 2; ++i) {
         struct agx_apple9_block *succ = b->successors[i];
         if (!succ || (i && succ == b->successors[0]))
            continue;
         void *preds = realloc(succ->predecessors,
            (succ->predecessor_count + 1) * sizeof(*succ->predecessors));
         if (!preds)
            return false;
         succ->predecessors = preds;
         succ->predecessors[succ->predecessor_count++] = b;
      }
   }
   program->current_block = program->last_block;
   return true;
}

/* Standard backwards fixed-point liveness. The allocator currently uses
 * conservative enclosing intervals derived from these sets; no loop shape,
 * nesting depth, or particular branch opcode is special to the analysis. */
static bool
apple9_liveness(struct agx_apple9_vir_program *program, unsigned *last_use)
{
   const unsigned words = MAX2(BITSET_WORDS(program->value_count), 1);
   BITSET_WORD *live = calloc(words, sizeof(*live));
   if (!live)
      return false;
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      free(b->live_in);
      free(b->live_out);
      b->live_in = calloc(words, sizeof(*live));
      b->live_out = calloc(words, sizeof(*live));
      if (!b->live_in || !b->live_out) {
         free(live);
         return false;
      }
   }
   bool progress;
   do {
      progress = false;
      for (struct agx_apple9_block *b = program->last_block; b; b = b->prev) {
         memset(live, 0, words * sizeof(*live));
         for (unsigned s = 0; s < 2; ++s) {
            if (!b->successors[s])
               continue;
            for (unsigned w = 0; w < words; ++w)
               live[w] |= b->successors[s]->live_in[w];
         }
         if (!b->successors[0] && !b->successors[1]) {
            if (program->output != AGX_APPLE9_VREG_INVALID)
               BITSET_SET(live, program->output);
            for (unsigned i = 0; i < program->live_out_count; ++i)
               BITSET_SET(live, program->live_out[i]);
         }
         memcpy(b->live_out, live, words * sizeof(*live));
         list_for_each_entry_rev(struct apple9_instruction_node, node, &b->instructions, link) {
            struct agx_apple9_vir_instr *ins = &node->instruction;
            for (unsigned c = 0; c < apple9_vir_dest_components(ins); ++c)
               BITSET_CLEAR(live, ins->dest + c);
            for (unsigned i = 0; i < ins->nr_srcs; ++i)
               BITSET_SET(live, ins->src[i]);
            /* Masked writes preserve inactive lanes. Until phi lowering,
             * the edge destination remains live across the masked region. */
            if (ins->op == AGX_APPLE9_VIR_PHI_SRC)
               BITSET_SET(live, ins->target);
         }
         if (memcmp(live, b->live_in, words * sizeof(*live))) {
            memcpy(b->live_in, live, words * sizeof(*live));
            progress = true;
         }
      }
   } while (progress);
   for (struct agx_apple9_block *b = program->blocks; b; b = b->next) {
      unsigned end = b->next ? b->next->start_index : program->instruction_count;
      if (end == b->start_index)
         continue;
      for (unsigned v = 0; v < program->value_count; ++v) {
         if (BITSET_TEST(b->live_out, v))
            last_use[v] = MAX2(last_use[v], end - 1);
      }
   }
   free(live);
   return true;
}

struct apple9_allocation_event {
   struct agx_apple9_vir_instr *instruction;
   unsigned position;
   unsigned serial;
};

static int
apple9_compare_allocation_events(const void *a, const void *b)
{
   const struct apple9_allocation_event *x = a, *y = b;
   if (x->position != y->position)
      return x->position < y->position ? -1 : 1;
   return x->serial < y->serial ? -1 : x->serial > y->serial;
}

static void apple9_move_after(struct agx_apple9_vir_program *, struct agx_apple9_vir_instr *, struct agx_apple9_vir_instr *);

bool
agx_apple9_allocate_vir(struct agx_apple9_vir_program *program,
                          const char **reason)
{
   if (reason != NULL)
      *reason = NULL;
   agx_apple9_place_phis(program);
   if (!apple9_build_cfg(program)) {
      if (reason)
         *reason = "Apple9 CFG has an invalid target or could not be allocated";
      return false;
   }
   bool has_side_effect = false;
   for (unsigned i = 0; i < program->instruction_count; ++i)
      has_side_effect |=
         program->instructions[i]->op == AGX_APPLE9_VIR_STORE_UNIFORM ||
         program->instructions[i]->op == AGX_APPLE9_VIR_DEVICE_STORE ||
         program->instructions[i]->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
         apple9_vir_is_graphics_output(program->instructions[i]->op);
   if ((program->output == AGX_APPLE9_VREG_INVALID && !has_side_effect &&
        !program->fragment_shader) ||
       (program->output != AGX_APPLE9_VREG_INVALID &&
        program->output >= program->value_count)) {
      if (reason != NULL)
         *reason = "Apple9 virtual IR has neither an output nor a store";
      return false;
   }

   free(program->publication);
   /* Empty fragment entrypoints are valid; don't rely on malloc(0). */
   unsigned allocation_count = MAX2(program->value_count, 1);
   program->publication = calloc(allocation_count, sizeof(bool));
   program->publication_count = 0;
   if (!program->publication) {
      if (reason)
         *reason = "out of memory classifying Apple9 values";
      return false;
   }
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (apple9_vir_publishes(ins) && ins->dest < program->value_count) {
         for (unsigned c = 0; c < apple9_vir_dest_components(ins); ++c) {
            if (ins->dest + c >= program->value_count)
               return false;
            program->publication[ins->dest + c] = true;
         }
      }
   }
   if (!apple9_validate_value_files(program, reason))
      return false;

   if (!apple9_propagate_source_register_classes(program, reason))
      return false;

   unsigned *last_use = calloc(allocation_count, sizeof(*last_use));
   bool *defined = calloc(allocation_count, sizeof(*defined));
   bool *seen_definition =
      calloc(allocation_count, sizeof(*seen_definition));
   bool *used = calloc(allocation_count, sizeof(*used));
   uint8_t *phys = malloc(allocation_count);
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
         program->instructions[i];
      const uint32_t dest = instruction->dest;
      const unsigned components = apple9_vir_dest_components(instruction);
      if (components == 0) {
         if (instruction->op != AGX_APPLE9_VIR_DEVICE_STORE &&
             instruction->op != AGX_APPLE9_VIR_SPILL_STORE &&
             instruction->op != AGX_APPLE9_VIR_DEVICE_ATOMIC &&
             instruction->op != AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT &&
             instruction->op != AGX_APPLE9_VIR_PHI_SRC &&
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
      if (components > (instruction->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE ? 8 : 4) ||
          components > program->value_count ||
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
         program->instructions[i];
      const unsigned components = apple9_vir_dest_components(instruction);
      if (components != 0) {
         for (unsigned c = 0; c < components; ++c) {
            seen_definition[instruction->dest + c] = true;
            used[instruction->dest + c] = true;
            last_use[instruction->dest + c] = MAX2(last_use[instruction->dest + c], i);
         }
      }
      for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
         if (instruction->src[s] >= program->value_count ||
             (defined[instruction->src[s]] &&
              !seen_definition[instruction->src[s]] &&
              apple9_vir_producer_instruction(program, instruction->src[s])->op != AGX_APPLE9_VIR_PHI)) {
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
      if (instruction->op == AGX_APPLE9_VIR_PHI_SRC) {
         const uint32_t target = instruction->target;
         if (target >= program->value_count || !defined[target]) {
            free(last_use);
            free(defined);
            free(seen_definition);
            free(used);
            free(phys);
            if (reason != NULL)
               *reason =
                  "Apple9 phi edge has no SSA definition";
            return false;
         }
         used[target] = true;
         last_use[target] = i;
      }
   }

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->op != AGX_APPLE9_VIR_PHI_SRC)
         continue;
      for (unsigned j = i; j < program->instruction_count; ++j) {
         if (program->instructions[j] == ins->phi_edge) {
            last_use[ins->src[0]] = MAX2(last_use[ins->src[0]], j);
            last_use[ins->target] = MAX2(last_use[ins->target], j);
            break;
         }
      }
   }

   /* Validate external roots before liveness indexes its bitsets. */
   for (unsigned i = 0; i < program->live_out_count; ++i) {
      if (program->live_out[i] >= program->value_count) {
         free(last_use);
         free(defined);
         free(seen_definition);
         free(used);
         free(phys);
         if (reason)
            *reason = "Apple9 virtual IR has an invalid live-out value";
         return false;
      }
   }

   if (!apple9_liveness(program, last_use)) {
      free(last_use);
      free(defined);
      free(seen_definition);
      free(used);
      free(phys);
      if (reason)
         *reason = "out of memory computing Apple9 CFG liveness";
      return false;
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
      const struct agx_apple9_vir_instr *load = program->instructions[i];
      const unsigned components = apple9_vir_dest_components(load);
      if ((load->op != AGX_APPLE9_VIR_DEVICE_LOAD &&
           load->op != AGX_APPLE9_VIR_ITER_FLAT) ||
          components == 0)
         continue;

      unsigned handoff = apple9_first_consumer(program, i);

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

   unsigned publication_end[AGX_APPLE9_PUBLICATION_COUNT] = {0};
   bool publication_used[AGX_APPLE9_PUBLICATION_COUNT] = {false};

   struct apple9_allocation_event *events =
      calloc(MAX2(program->instruction_count, 1), sizeof(*events));
   if (!events) {
      free(last_use);
      free(defined);
      free(seen_definition);
      free(used);
      free(phys);
      if (reason)
         *reason = "out of memory ordering Apple9 allocations";
      return false;
   }
   unsigned event_count = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      ins->live_after_mask = 0;
      for (unsigned src = 0; src < ins->nr_srcs; ++src) {
         if (last_use[ins->src[src]] > i)
            ins->live_after_mask |= 1u << src;
      }
      if (!apple9_vir_dest_components(ins))
         continue;
      unsigned first = i;
      if (ins->op == AGX_APPLE9_VIR_PHI) {
         /* A phi is defined at the successor, but its assigned storage is
          * populated on incoming masked edges. Reserve it at the first edge
          * without moving its SSA definition out of the successor block. */
         for (unsigned j = 0; j < program->instruction_count; ++j) {
            const struct agx_apple9_vir_instr *edge = program->instructions[j];
            if (edge->op == AGX_APPLE9_VIR_PHI_SRC && edge->target == ins->dest)
               first = MIN2(first, j);
         }
      }
      events[event_count++] = (struct apple9_allocation_event){ins, first, i};
   }
   qsort(events, event_count, sizeof(*events), apple9_compare_allocation_events);
   for (unsigned event = 0; event < event_count; ++event) {
      struct agx_apple9_vir_instr *instruction = events[event].instruction;
      const unsigned i = events[event].position;
      const unsigned components = apple9_vir_dest_components(instruction);
      for (unsigned reg = 0; reg < AGX_APPLE9_GPR_COUNT; ++reg) {
         if (owner[reg] >= 0 && last_use[owner[reg]] < i) {
            owner[reg] = (program->reserved_gprs[reg] || fixed_gpr[reg]) ? -2 : -1;
            assert(live > 0);
            --live;
         }
      }

      if (program->publication[instruction->dest]) {
         unsigned index;
         for (index = 0; index + components <= AGX_APPLE9_PUBLICATION_COUNT; ++index) {
            unsigned alignment = agx_apple9_encoding_info(instruction->encoding)
                                    ->operands[0].alignment_halves / 2;
            if (index % MAX2(alignment, 1))
               continue;
            bool available = true;
            for (unsigned c = 0; c < components; ++c) {
               unsigned maximum = program->max_phys[instruction->dest + c];
               available &= (!publication_used[index + c] ||
                             publication_end[index + c] < i) &&
                            (maximum == AGX_APPLE9_PHYS_INVALID ||
                             index + c <= maximum);
            }
            if (available)
               break;
         }
         if (index + components > AGX_APPLE9_PUBLICATION_COUNT) {
            free(events);
            free(last_use);
            free(defined);
            free(seen_definition);
            free(used);
            free(phys);
            if (reason)
               *reason = "Apple9 allocator exhausted export publication slots";
            return false;
         }
         unsigned end = apple9_publication_end(program, i);
         for (unsigned c = 0; c < components; ++c) {
            phys[instruction->dest + c] = index + c;
            publication_used[index + c] = true;
            publication_end[index + c] = MAX2(end, last_use[instruction->dest + c]);
         }
         program->publication_count = MAX2(program->publication_count,
                                           index + components);
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
         instruction->op == AGX_APPLE9_VIR_PHI;
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
          * fall back to the low bank only after r16-r95 is occupied. This
          * includes PHI/COLLECT pseudo-definitions: loop phis must not
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
                 instruction->op != AGX_APPLE9_VIR_PHI &&
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
         if (agx_apple9_trace_enabled()) {
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
                  program->instructions[j];
               fprintf(stderr, "APPLE9_ALLOC_VIR i=%u op=%u enc=%u dest=", j,
                       vir->op, vir->encoding);
               if (vir->dest == AGX_APPLE9_VREG_INVALID)
                  fputs("-", stderr);
               else
                  fprintf(stderr, "v%u fixed=%u", vir->dest,
                          program->fixed_phys[vir->dest]);
               if (vir->op == AGX_APPLE9_VIR_PHI_SRC)
                  fprintf(stderr, " target=v%u", vir->target);
               fputs(" src=", stderr);
               for (unsigned s = 0; s < vir->nr_srcs; ++s)
                  fprintf(stderr, "%sv%u", s ? "," : "", vir->src[s]);
               fprintf(stderr, " imm=%#x\n", vir->immediate);
            }
         }
         free(events);
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
                        : "Apple9 no-spill allocator exhausted r2-r95")
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

   }
   free(events);

   free(program->phys);
   program->phys = phys;
   program->peak_live_gprs = peak;
   program->max_phys_gpr = max_phys;
   free(last_use);
   free(defined);
   free(seen_definition);
   free(used);
   if (!apple9_finalize_dependencies(program)) {
      if (reason)
         *reason = "could not analyze Apple9 dependency uses";
      return false;
   }
   return agx_apple9_validate_vir_allocation(program, reason);
}

static const struct agx_apple9_vir_instr *
apple9_vir_producer_instruction(const struct agx_apple9_vir_program *program,
                                uint32_t value)
{
   return agx_apple9_definition(program, value);
}

static bool
apple9_vir_is_pending_load(enum agx_apple9_vir_opcode op)
{
   return op == AGX_APPLE9_VIR_SPILL_LOAD ||
          op == AGX_APPLE9_VIR_DEVICE_LOAD || op == AGX_APPLE9_VIR_TILE_LOAD ||
          op == AGX_APPLE9_VIR_TEXTURE_SAMPLE || op == AGX_APPLE9_VIR_ITER_FLAT;
}

static bool
apple9_vir_is_async(const struct agx_apple9_vir_instr *ins)
{
   return apple9_vir_is_pending_load(ins->op) ||
          ins->op == AGX_APPLE9_VIR_SPILL_STORE ||
          (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC && !ins->atomic_discard);
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
   return instruction->op != AGX_APPLE9_VIR_PHI_SRC &&
          apple9_instruction_dependency_layout(instruction) !=
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

/* A SAVE has no register result. Prefer folding its completion into the next
 * instruction that can wait, but do not carry scratch writes across a mask or
 * block boundary. A boundary with no such instruction gets an ordinary SSA
 * identity operation on zero; its register is allocated like any other temp. */
static void
apple9_move_after(struct agx_apple9_vir_program *program,
                 struct agx_apple9_vir_instr *ins, struct agx_apple9_vir_instr *after)
{
   struct apple9_instruction_node *node = apple9_node(ins);
   struct apple9_instruction_node *previous = apple9_node(after);
   list_del(&node->link);
   list_add(&node->link, &previous->link);
   node->block = previous->block;
   agx_apple9_vir_reindex(program);
}

static bool
apple9_materialize_spill_completion(struct agx_apple9_vir_program *program,
                                   struct agx_apple9_vir_instr *store)
{
   uint32_t zero = agx_apple9_vir_emit(program, AGX_APPLE9_VIR_IMM,
                                      AGX_APPLE9_ENC_MOV_IMM32, NULL, 0, 0);
   if (zero == AGX_APPLE9_VREG_INVALID)
      return false;
   struct agx_apple9_vir_instr *constant = program->instructions[program->instruction_count - 1];
   uint32_t sources[] = {zero, zero};
   if (agx_apple9_vir_emit(program, AGX_APPLE9_VIR_IOR,
                           AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0) == AGX_APPLE9_VREG_INVALID)
      return false;
   struct agx_apple9_vir_instr *wait = program->instructions[program->instruction_count - 1];
   apple9_move_after(program, constant, store);
   apple9_move_after(program, wait, constant);
   store->completion_consumer = wait;
   return true;
}

static bool
apple9_schedule_spill_completions(struct agx_apple9_vir_program *program)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *store = program->instructions[i];
      if (store->op != AGX_APPLE9_VIR_SPILL_STORE)
         continue;
      store->completion_consumer = NULL;
      for (unsigned j = i + 1; j < program->instruction_count; ++j) {
         struct agx_apple9_vir_instr *next = program->instructions[j];
         if (agx_apple9_instr_block(next) != agx_apple9_instr_block(store) ||
             apple9_vir_is_control_side_effect(next->op) ||
             next->op == AGX_APPLE9_VIR_PHI_SRC)
            break;
         if (apple9_instruction_accepts_scoreboard_slot(next)) {
            store->completion_consumer = next;
            break;
         }
      }
      if (!store->completion_consumer && !apple9_materialize_spill_completion(program, store))
         return false;
   }
   return true;
}

static bool
apple9_materialize_pending_result_at(struct agx_apple9_vir_program *program,
                                     unsigned producer_index,
                                     unsigned insertion, const char **reason)
{
   const unsigned old_count = program->instruction_count;
   const struct agx_apple9_vir_instr *producer =
      program->instructions[producer_index];
   if (producer->op == AGX_APPLE9_VIR_SPILL_STORE)
      return apple9_materialize_spill_completion(program, program->instructions[producer_index]);
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
      apple9_vir_dest_components(program->instructions[producer_index]);
   const unsigned earliest_insertion = producer_index + (atomic ? 2 : 1);
   if (atomic &&
       (producer_index + 1 >= old_count ||
        program->instructions[producer_index + 1]->op !=
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

   assert(components > 0 && components <= 4);
   uint32_t materialized[4];
   struct agx_apple9_vir_instr *materialize_instructions[4];
   /* Consume the pending result through the general integer-logic form.
    *
    * The old umin(x, x) bridge was functionally correct, but its compact
    * destination is limited to r0-r15.  Under genuine scoreboard pressure
    * those registers can all remain live while r16-r95 are available, which
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
      /* Reload handoffs are part of the reload itself. Spilling them would
       * recreate the same pending reload and handoff indefinitely. */
   }

   materialize_instructions[0]->scoreboard_materialize = true;

   if (insertion == earliest_insertion) {
      /* A return handoff belongs to its producer, even when the next
       * instruction starts a loop header. Backedges must not repeat it. */
      struct apple9_instruction_node *after =
         apple9_node(program->instructions[insertion - 1]);
      for (unsigned c = 0; c < components; ++c) {
         struct apple9_instruction_node *node = apple9_node(materialize_instructions[c]);
         list_del(&node->link);
         list_add(&node->link, &after->link);
         node->block = after->block;
         after = node;
      }
      agx_apple9_vir_reindex(program);
   } else if (insertion < old_count) {
      struct agx_apple9_vir_instr *before = program->instructions[insertion];
      for (unsigned c = 0; c < components; ++c)
         agx_apple9_vir_move_before(program, materialize_instructions[c], before);
   }

   agx_apple9_invalidate_uses(program);
   for (unsigned i = insertion + components; i < program->instruction_count;
        ++i) {
      struct agx_apple9_vir_instr *instruction = program->instructions[i];
      for (unsigned s = 0; s < instruction->nr_srcs; ++s) {
         for (unsigned c = 0; c < components; ++c) {
            if (instruction->src[s] == pending + c)
               instruction->src[s] = materialized[c];
         }
      }
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
      struct agx_apple9_vir_instr *producer = program->instructions[i];
      unsigned existing_handoff = apple9_first_consumer(program, i);
      if (existing_handoff != UINT_MAX &&
          program->instructions[existing_handoff]->scoreboard_materialize)
         continue;
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
         agx_apple9_vir_remove(program, producer);
         --i;
         continue;
      }

      if (handoff != UINT_MAX && apple9_instruction_accepts_scoreboard_slot(
                                    program->instructions[handoff]))
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
      program->instructions[producer_index];
   if (producer->op == AGX_APPLE9_VIR_SPILL_STORE)
      return producer->completion_consumer ? apple9_node(producer->completion_consumer)->position : UINT_MAX;
   const uint32_t first = producer->dest;
   const unsigned components = apple9_vir_dest_components(producer);
   unsigned first_use = UINT_MAX;
   for (unsigned c = 0; c < components; ++c) {
      for (const struct agx_apple9_use *use = agx_apple9_uses(program, first + c);
           use; use = use->next) {
         if (use->instruction->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
            continue;
         unsigned position = apple9_node(use->instruction)->position;
         if (position > producer_index)
            first_use = MIN2(first_use, position);
         break;
      }
   }
   return first_use;
}

/* Scoreboard legalization may split a returned tuple into scalar copies.
 * Reassemble adjacent tuples for store data and indirect load addresses. */
static bool
apple9_collect_memory_source_tuples(struct agx_apple9_vir_program *program,
                                    const char **reason)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *store = program->instructions[i];
      const bool load_indirect =
         store->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT;
      const bool store_indirect =
         store->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
      for (unsigned tuple_index = 0; tuple_index < (store_indirect ? 2 : 1);
           ++tuple_index) {
         const bool indirect = load_indirect || tuple_index == 1;
         const unsigned first = load_indirect ? 1
                                : tuple_index == 1
                                   ? store->memory_components + 1
                                   : 0;
         const unsigned components = indirect ? 2 : store->memory_components;
         if ((!indirect && store->op != AGX_APPLE9_VIR_DEVICE_STORE) ||
             components < 2 ||
             apple9_vir_values_form_tuple(program, store->src + first,
                                          components))
            continue;

         uint32_t sources[4];
         for (unsigned c = 0; c < components; ++c)
            sources[c] = store->src[first + c];

         const unsigned old_count = program->instruction_count;
         const uint32_t tuple =
            agx_apple9_vir_emit_collect(program, sources, components);
         if (tuple == AGX_APPLE9_VREG_INVALID) {
            if (reason != NULL)
               *reason = "out of memory collecting Apple9 memory source tuples";
            return false;
         }

         struct agx_apple9_vir_instr *collect =
            program->instructions[old_count];
         agx_apple9_vir_move_before(program, collect, store);

         agx_apple9_invalidate_uses(program);
         store = program->instructions[i + 1];
         for (unsigned c = 0; c < components; ++c)
            store->src[first + c] = tuple + c;

         ++i;
      }
   }

   return true;
}

static const uint8_t scalar_load_preference[] = {6, 1, 2, 3, 4, 5};

static bool
apple9_materialize_scoreboard_pressure(struct agx_apple9_vir_program *program,
                                       const char **reason)
{
   for (;;) {
      bool changed = false;
      unsigned handoffs[AGX_APPLE9_SCOREBOARD_SLOT_6 + 1];
      unsigned owners[AGX_APPLE9_SCOREBOARD_SLOT_6 + 1];
      for (unsigned slot = 0; slot < ARRAY_SIZE(handoffs); ++slot)
         handoffs[slot] = owners[slot] = UINT_MAX;

      for (unsigned i = 0; i < program->instruction_count; ++i) {
         for (unsigned slot = 1; slot < ARRAY_SIZE(handoffs); ++slot) {
            if (handoffs[slot] == i)
               handoffs[slot] = owners[slot] = UINT_MAX;
         }
         const struct agx_apple9_vir_instr *current = program->instructions[i];
         if (!apple9_vir_is_async(current))
            continue;
         unsigned handoff = apple9_first_consumer(program, i);
         if (handoff == UINT_MAX)
            continue;
         bool automatic = current->producer_scoreboard_slot ==
                             AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         unsigned slot = current->producer_scoreboard_slot;
         unsigned shared = 0;
         for (unsigned candidate = 1; candidate < ARRAY_SIZE(handoffs); ++candidate) {
            if (handoffs[candidate] == handoff)
               shared = candidate;
         }
         /* The assignment pass diagnoses explicitly incompatible slots in
          * a shared handoff; that is not a pressure problem. */
         if (shared)
            continue;
         if (automatic) {
            slot = 0;
            for (unsigned p = 0; p < ARRAY_SIZE(scalar_load_preference); ++p) {
               unsigned candidate = scalar_load_preference[p];
               if (handoffs[candidate] == UINT_MAX) {
                  slot = candidate;
                  break;
               }
            }
         }
         if (!shared && slot >= 1 && slot < ARRAY_SIZE(handoffs) &&
             handoffs[slot] == UINT_MAX) {
            handoffs[slot] = handoff;
            owners[slot] = i;
            continue;
         }

         /* Simulate the allocator, including fixed texture-result slots.
          * A fixed producer can conflict before all six slots are occupied.
          * Retire the conflicting group (or the oldest group under full
          * pressure) through ordinary SSA copies before this producer. */
         unsigned victim = shared;
         if (!victim && slot >= 1 && slot < ARRAY_SIZE(handoffs))
            victim = slot;
         if (!victim) {
            for (unsigned candidate = 1; candidate < ARRAY_SIZE(handoffs); ++candidate) {
               if (owners[candidate] != UINT_MAX &&
                   (!victim || owners[candidate] < owners[victim]))
                  victim = candidate;
            }
         }
         if (!victim || handoffs[victim] == UINT_MAX) {
            if (reason)
               *reason = "Apple9 pending producer has no legal scoreboard slot";
            return false;
         }
         const unsigned target_handoff = handoffs[victim];
         const struct agx_apple9_vir_instr *pressure_instruction = current;
         struct util_dynarray values = UTIL_DYNARRAY_INIT;
         for (unsigned p = 0; p < i; ++p) {
            const struct agx_apple9_vir_instr *producer = program->instructions[p];
            if (apple9_vir_is_async(producer) &&
                apple9_first_consumer(program, p) == target_handoff)
               util_dynarray_append(&values, producer);
         }

         util_dynarray_foreach(&values, const struct agx_apple9_vir_instr *, value) {
            unsigned producer = UINT_MAX;
            unsigned insertion = UINT_MAX;
            for (unsigned p = 0; p < program->instruction_count; ++p) {
               const struct agx_apple9_vir_instr *candidate =
                  program->instructions[p];
               if (candidate == *value) {
                  producer = p;
               }
               if (candidate == pressure_instruction)
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
                                apple9_vir_definition_index(program, source)) ==
             consumer_index;
}

static void
apple9_normalize_logic_handoff_source(struct agx_apple9_vir_program *program,
                                      unsigned consumer_index)
{
   struct agx_apple9_vir_instr *consumer =
      program->instructions[consumer_index];
   if ((consumer->encoding != AGX_APPLE9_ENC_LOGIC_EXTENDED &&
        consumer->encoding != AGX_APPLE9_ENC_LOGIC_EXPORT) ||
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
      agx_apple9_invalidate_uses(program);
   }
}


bool
agx_apple9_assign_vir_scoreboard_slots(struct agx_apple9_vir_program *program,
                                       const char **reason)
{
   agx_apple9_invalidate_uses(program);
   if (reason != NULL)
      *reason = NULL;
   if (program == NULL)
      return false;

   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->scoreboard_assigned) {
         ins->scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
         ins->scoreboard_assigned = false;
      }
      if (ins->producer_scoreboard_assigned) {
         ins->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
         ins->producer_scoreboard_assigned = false;
      }
   }
   if (!apple9_schedule_spill_completions(program)) {
      if (reason) *reason = "out of memory scheduling Apple9 spill completion";
      return false;
   }

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
      struct agx_apple9_vir_instr *producer = program->instructions[i];

      /* Consume an input handoff before allocating a result produced by the
       * same instruction. EXP-M4-51 proves that a returning atomic may reuse
       * its pending input's slot, including under full six-slot pressure. */
      if (handoff_slot[i] != AGX_APPLE9_SCOREBOARD_SLOT_NONE) {
         struct agx_apple9_vir_instr *consumer = program->instructions[i];
         const uint8_t slot = handoff_slot[i];
         if (consumer->scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE &&
             consumer->scoreboard_slot != slot) {
            if (reason != NULL)
               *reason =
                  "Apple9 consumer conflicts with allocated scoreboard slot";
            goto fail;
         }
         consumer->scoreboard_slot = slot;
         consumer->scoreboard_assigned = true;
         apple9_normalize_logic_handoff_source(program, i);
         occupied[slot] = false;
      }

      const bool is_load = apple9_vir_is_pending_load(producer->op);
      const bool is_returning_atomic =
         producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
         !producer->atomic_discard;
      if (is_load || is_returning_atomic || producer->op == AGX_APPLE9_VIR_SPILL_STORE) {
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
                       program->instructions[handoff])) {
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
                            program->instructions[handoff], candidate)) {
                        slot = candidate;
                        break;
                     }
                  }
               } else {
                  slot = producer->producer_scoreboard_slot;
               }

               if (!apple9_instruction_accepts_specific_scoreboard_slot(
                      program->instructions[handoff], slot) ||
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

            producer->producer_scoreboard_assigned = automatic;
            if (automatic && (producer->op == AGX_APPLE9_VIR_TILE_LOAD ||
                              producer->op == AGX_APPLE9_VIR_ITER_FLAT ||
                              producer->op == AGX_APPLE9_VIR_SPILL_LOAD ||
                              producer->op == AGX_APPLE9_VIR_SPILL_STORE)) {
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
                  program->instructions[i + 1];
               assert(publication->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT);
               publication->producer_scoreboard_slot = slot;
            }
         }
      }

   }

   free(handoff_slot);
   return apple9_collect_memory_source_tuples(program, reason);

fail:
   free(handoff_slot);
   return false;
}

static void
packed_init(struct agx_apple9_packed_instruction *packed, const uint8_t *bytes,
            unsigned length)
{
   assert(length <= sizeof(packed->bytes));
   memset(packed, 0, sizeof(*packed));
   memcpy(packed->bytes, bytes, length);
   packed->length = length;
}

bool
agx_apple9_pack_get_sr(unsigned dst, uint8_t selector, uint8_t datapath,
                       struct agx_apple9_packed_instruction *packed)
{
   if (dst >= 64 || (datapath & 0xc0))
      return false;
   uint8_t bytes[4] = {
      ((dst & 0xf) << 4) | 0x0c,
      selector,
      datapath | ((dst >> 4) << 6),
      0x06,
   };
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_get_sr_zext16(unsigned dst, uint8_t selector,
                              struct agx_apple9_packed_instruction *packed)
{
   if (dst >= 64)
      return false;

   /*
    * EXP-M4-29's local-position, local-index, SIMD-lane, SIMD-group and
    * total-thread-count controls all use this pair.  The form-0 GET_SR
    * publishes a narrow result, and the immediately following X3 companion
    * zero-extends that result in place.  Selector-only hardware recodes
    * establish that the pair is shared by the complete scalar16 class.
    */
   const uint8_t bytes[8] = {
      ((dst & 15) << 4) | 0x04, selector, 0x10 | ((dst >> 4) << 6), 0x06,
      ((dst & 15) << 4) | 0x03, 0x00,     (dst >> 4) << 6,          0x01,
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
   if (instruction->nr_srcs != 1 || dst >= AGX_APPLE9_GPR_COUNT ||
       src >= AGX_APPLE9_GPR_COUNT ||
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
      (src & 63) << 2,
      (retain_source ? 0x8c : 0xac) | (src >> 6),
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
   if (instruction->nr_srcs != 1 || dst >= AGX_APPLE9_GPR_COUNT ||
       src >= AGX_APPLE9_GPR_COUNT ||
       (instruction->op != AGX_APPLE9_VIR_F2I32 &&
        instruction->op != AGX_APPLE9_VIR_F2U32))
      return false;

   /* EXP-0013 proves truncation toward zero and the signed/unsigned selector;
    * EXP-0144 independently locates dst=byte3>>1 and src=byte5>>2. The
    * continuation's format bit must also match signedness: the signed format
    * with an unsigned operation saturates at 65535, not UINT32_MAX. */
   const uint8_t bytes[] = {
      0x27,
      0x07,
      0x54,
      dst << 1,
      0x03,
      (src & 63) << 2,
      ((instruction->live_after_mask & 1) ? 0x96 : 0xb4) | (src >> 6),
      instruction->op == AGX_APPLE9_VIR_F2I32 ? 0x48 : 0x08,
      instruction->op == AGX_APPLE9_VIR_F2I32 ? 0x03 : 0x02,
      0x00,
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
       instruction->nr_srcs != 1 || dst >= AGX_APPLE9_GPR_COUNT ||
       src >= AGX_APPLE9_GPR_COUNT ||
       ((instruction->src_abs_mask | instruction->src_neg_mask) & ~1u) ||
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
   uint8_t bytes[10] = {
      0x2f | (family << 7),
      function,
      0x54,
      dst << 1,
      instruction->immediate,
      (src & 63) << 2,
      (src >> 6) | (instruction->op == AGX_APPLE9_VIR_FRCP
                       ? ((instruction->live_after_mask & 1u) ? 0x00 : 0x10)
                       : ((instruction->live_after_mask & 1u) ? 0x90 : 0xb0)),
      precision,
      rounding,
      0,
   };
   if (instruction->saturate)
      set_bits(bytes, instruction->op == AGX_APPLE9_VIR_FRCP ? 65 : 58, 1, 1);
   /* Reciprocal has a separate datapath. Both modifier layouts preserve
    * source lifetime, including FP32 exceptional values and high GPRs. */
   set_bits(bytes, instruction->op == AGX_APPLE9_VIR_FRCP ? 70 : 63,
            1, instruction->src_abs_mask);
   set_bits(bytes, instruction->op == AGX_APPLE9_VIR_FRCP ? 71 : 64,
            1, instruction->src_neg_mask);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_shift(const struct agx_apple9_vir_instr *I, const uint8_t *phys,
           struct agx_apple9_packed_instruction *packed)
{
   bool arithmetic = I->op == AGX_APPLE9_VIR_ISHR;
   bool variable = I->nr_srcs == 2;
   enum agx_apple9_encoding encoding = arithmetic
      ? (variable ? AGX_APPLE9_ENC_SHIFT_ARITH_REGISTER
                  : AGX_APPLE9_ENC_SHIFT_EXTENDED)
      : (variable ? AGX_APPLE9_ENC_SHIFT_LOGICAL_REGISTER
                  : AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE);
   unsigned dst = phys[I->dest], src = phys[I->src[0]];
   if (I->encoding != encoding || (I->nr_srcs != 1 && !variable) ||
       dst >= AGX_APPLE9_GPR_COUNT || src >= AGX_APPLE9_GPR_COUNT ||
       (!arithmetic && I->op != AGX_APPLE9_VIR_USHR &&
        I->op != AGX_APPLE9_VIR_ISHL) ||
       (variable ? (phys[I->src[1]] >= AGX_APPLE9_GPR_COUNT || I->immediate)
                 : I->immediate >= 32))
      return false;

   uint8_t bytes[12] = {0xa7, 0, 0x54, 0, 0x02, 0, 0, 0, 0xf0, 0x11, 0x01, 0};
   if (arithmetic) {
      bytes[1] = 1;
      bytes[7] = variable ? 0xd8 : 0x78;
      bytes[8] = variable ? 0xe2 : 0x62;
      bytes[9] = 0;
   } else {
      if (I->op == AGX_APPLE9_VIR_ISHL)
         bytes[0] = 0x27;
      if (variable) {
         bytes[9] = 0x13;
         bytes[10] = 0x05;
      }
   }
   set_bits(bytes, 25, 7, dst);
   set_bits(bytes, 42, 7, src);
   if (variable)
      set_bits(bytes, 51, 7, phys[I->src[1]]);
   else
      set_bits(bytes, 50, 5, I->immediate);

   /* These datapaths have dedicated release flags, distinct from IMAD's
    * descriptor auxiliaries. The register amount consumes its low 16 bits;
    * selection explicitly limits NIR's shift amounts to five bits. */
   if (I->live_after_mask & 1)
      set_bits(bytes, arithmetic ? 62 : 72, 1, 0);
   if (variable && (I->live_after_mask & 2))
      set_bits(bytes, arithmetic ? 63 : 73, 1, 0);
   packed_init(packed, bytes, arithmetic ? 10 : 12);
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
   uint8_t bytes[8];
   if (!agx_apple9_encode_literal32(bytes, dst, value))
      return false;
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
      AGX_APPLE9_SCOREBOARD_SLOT_NONE, raw_token, packed);
}

bool
agx_apple9_pack_device_load_u32_raw(
   unsigned dst, unsigned index, unsigned binding,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed)
{
   return agx_apple9_pack_device_load_scalar_raw(
      dst, index, binding, 32, index_kind, flags, incoming_slot,
      raw_token, packed);
}

bool
agx_apple9_pack_device_load_scalar_raw(
   unsigned dst, unsigned index, unsigned binding, unsigned bits,
   enum agx_apple9_device_load_index_kind index_kind, uint8_t flags,
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed)
{
   if (bits != 8 && bits != 16 && bits != 32)
      return false;
   if (!agx_apple9_pack_device_load_vector_u32_raw(
          dst, index, binding, 1, index_kind, flags, incoming_slot,
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
   enum agx_apple9_scoreboard_slot incoming_slot, uint16_t raw_token,
   struct agx_apple9_packed_instruction *packed)
{
   if (components < 1 || components > 4 ||
       dst + components > AGX_APPLE9_GPR_COUNT || binding > UINT8_MAX ||
       (flags & ~AGX_APPLE9_DEVICE_LOAD_HAS_NEXT) ||
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

   /* Input completion, linear load sequencing, and the result slot are
    * independent. Bits 12..17 form the common six-bit input mask, for both
    * dependent indices and pointer pairs (EXP-M4-63). */
   uint8_t bytes[14] = {
      0x67, 0x00, 0x44, 0x00, 0x00, 0x00, 0x20,
      0x00, 0x11, 0x00, 0x00, 0x40, 0x46, 0x00,
   };
   if (flags & AGX_APPLE9_DEVICE_LOAD_HAS_NEXT)
      bytes[2] = 0x54;
   bytes[3] = dst << 1;
   bytes[4] = binding;
   bytes[5] = encoded_index;
   static const uint8_t width_token_bits[] = {0x00, 0x08, 0x0c, 0x06};
   static const uint8_t width_tail[] = {0x46, 0x48, 0x40, 0x40};
   bytes[8] = (raw_token >> 8) | width_token_bits[components - 1];
   bytes[9] = raw_token & 0xff;
   bytes[12] = width_tail[components - 1];
   if (!apple9_pack_dependency(bytes, sizeof(bytes),
                               AGX_APPLE9_DEPENDENCY_MASK_12_17, incoming_slot))
      return false;
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_device_store_scalar(
   unsigned data, unsigned index, unsigned binding, unsigned bits,
   enum agx_apple9_scoreboard_slot scoreboard_slot, bool release_index,
   struct agx_apple9_packed_instruction *packed)
{
   if (data >= AGX_APPLE9_GPR_COUNT || index >= AGX_APPLE9_GPR_COUNT ||
       binding > UINT8_MAX || (bits != 8 && bits != 16 && bits != 32))
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
   if (components < 1 || components > 4 ||
       data + components > AGX_APPLE9_GPR_COUNT ||
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
apple9_alu_sources(const struct agx_apple9_vir_instr *I, const uint8_t *phys,
                  unsigned arity, uint32_t *values, uint8_t *live);

/* Fixed integer sources use the GPR cache bit as uniform bit 7. The
 * descriptor's discard bit distinguishes a uniform from an immediate when
 * the register-file bit is clear; GPR lifetime remains independent. */
static void
pack_integer_source(uint8_t *bytes, unsigned source, bool uniform, bool live,
                    unsigned value_bit, unsigned discard_bit, unsigned file_bit)
{
   set_bits(bytes, value_bit, 7, source & 0x7f);
   set_bits(bytes, value_bit + 7, 1, uniform ? source >> 7 : live);
   set_bits(bytes, discard_bit, 1, uniform || !live);
   set_bits(bytes, file_bit, 1, !uniform);
}

/* Load and store share byte addressing but place the fields two bits apart. */
static bool
pack_memory_address(const struct agx_apple9_vir_instr *I,
                    struct agx_apple9_packed_instruction *packed)
{
   /* T8132 validates shifts through four. The larger-field probe failed,
    * so larger scales remain explicit ALU expressions. */
   if (I->memory_index_shift > 4)
      return false;
   unsigned base = I->op == AGX_APPLE9_VIR_DEVICE_LOAD ? 75 : 73;
   set_bits(packed->bytes, base + 2, 16, (uint16_t)I->memory_offset);
   set_bits(packed->bytes, base + 22, 3,
            I->memory_index_shift == 4 ? 0 : I->memory_index_shift + 1);
   return true;
}

static bool
pack_iadd(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
          struct agx_apple9_packed_instruction *packed)
{
   uint32_t source[3];
   uint8_t live;
   if (instruction->encoding != AGX_APPLE9_ENC_INT_ADD_EXTENDED ||
       instruction->alu_src_immediate_mask ||
       !apple9_alu_sources(instruction, phys, 2, source, &live))
      return false;
   uint8_t bytes[10] = {0x9f, 0x01, 0x54, 0x00, 0x02,
                        0x00, 0x00, 0xa8, 0x17, 0x05};
   if (instruction->op == AGX_APPLE9_VIR_ISUB)
      bytes[0] = 0x1f;
   set_bits(bytes, 25, 7, phys[instruction->dest]);
   for (unsigned s = 0; s < 2; ++s)
      pack_integer_source(bytes, source[s], instruction->alu_src_uniform_mask & (1u << s),
                          live & (1u << s), 42 + 9 * s, 65 + s, 72 + 2 * s);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_cube(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
           struct agx_apple9_packed_instruction *packed)
{
   if (instruction->encoding != AGX_APPLE9_ENC_CUBE ||
       instruction->nr_srcs != 3 || instruction->immediate > 2 ||
       instruction->dest_components != (instruction->immediate == 0 ? 2 : 1))
      return false;
   uint8_t bytes[12] = {0x17, 1, 0x54, 0, 3, 0, 0, 0, 0x50, 0x2f, 0x92, 0};
   set_bits(bytes, 24, 7, phys[instruction->dest]);
   set_bits(bytes, 90, 2, instruction->immediate);
   for (unsigned i = 0; i < 3; ++i) {
      set_bits(bytes, 41 + 9 * i, 7, phys[instruction->src[i]]);
      if (instruction->live_after_mask & BITFIELD_BIT(i)) {
         set_bits(bytes, 48 + 9 * i, 1, 1);
         set_bits(bytes, 73 + i, 1, 0);
      }
   }
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_imad(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
          struct agx_apple9_packed_instruction *packed)
{
   uint32_t source[3];
   uint8_t live;
   if (instruction->encoding != AGX_APPLE9_ENC_INT_MAD_EXTENDED ||
       instruction->immediate || instruction->alu_src_immediate_mask ||
       !apple9_alu_sources(instruction, phys, 3, source, &live))
      return false;
   uint8_t bytes[12] = {0x9f, 0x00, 0x54, 0x00, 0x02, 0x00,
                        0x00, 0x00, 0xd0, 0x2f, 0x2a, 0x00};
   set_bits(bytes, 25, 7, phys[instruction->dest]);
   for (unsigned s = 0; s < 3; ++s)
      pack_integer_source(bytes, source[s], instruction->alu_src_uniform_mask & (1u << s),
                          live & (1u << s), 42 + 9 * s, 73 + s, 81 + 2 * s);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_mul_wide(const struct agx_apple9_vir_instr *I, const uint8_t *phys,
              struct agx_apple9_packed_instruction *packed)
{
   unsigned dst = phys[I->dest], a = phys[I->src[0]], b = phys[I->src[1]];
   if (I->encoding != AGX_APPLE9_ENC_INT_MUL_WIDE || I->nr_srcs != 2 ||
       I->dest_components != 2 || I->immediate > 1 || (dst & 1) ||
       dst + 1 >= AGX_APPLE9_GPR_COUNT || a >= AGX_APPLE9_GPR_COUNT ||
       b >= AGX_APPLE9_GPR_COUNT || phys[I->dest + 1] != dst + 1)
      return false;

   /* This form writes both product words, not just mul-high to dst.
    * The two multiplicand lifetime fields are shared with low IMAD; the
    * addend descriptor is disabled. Signedness affects the upper word. */
   uint8_t bytes[] = {0x9f, 0x00, 0x54, 0x00, 0x02, 0x00,
                     0x00, 0x00, 0xe0, 0x26, 0x0a, 0x00};
   if (I->immediate)
      bytes[10] = 0x1e;
   set_bits(bytes, 25, 7, dst);
   set_bits(bytes, 42, 7, a);
   set_bits(bytes, 51, 7, b);
   if (I->live_after_mask & 1) {
      set_bits(bytes, 49, 1, 1);
      set_bits(bytes, 73, 1, 0);
   }
   if (I->live_after_mask & 2) {
      set_bits(bytes, 58, 1, 1);
      set_bits(bytes, 74, 1, 0);
   }
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_compact_binary_gprs(uint8_t *bytes, unsigned dst, unsigned a, unsigned b,
                         unsigned destination_count)
{
   if (dst >= destination_count || a >= AGX_APPLE9_GPR_COUNT ||
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
apple9_alu_sources(const struct agx_apple9_vir_instr *I, const uint8_t *phys,
                      unsigned arity, uint32_t *values, uint8_t *live)
{
   unsigned inline_mask = I->alu_src_uniform_mask | I->alu_src_immediate_mask;
   if ((inline_mask & ~BITFIELD_MASK(arity)) ||
       (I->alu_src_uniform_mask & I->alu_src_immediate_mask))
      return false;
   unsigned source = 0;
   *live = 0;
   for (unsigned s = 0; s < arity; ++s) {
      enum agx_apple9_operand_role role = AGX_APPLE9_OPERAND_SRC0 + s;
      const struct agx_apple9_operand_constraint *constraint =
         agx_apple9_find_operand(I->encoding, role);
      unsigned file = agx_apple9_inline_source_file(I, role);
      if (!constraint || (file && !(constraint->files & file)))
         return false;
      if (file == AGX_APPLE9_FILE_UNIFORM) {
         if (I->alu_src_value[s] > constraint->uniform_max_index)
            return false;
         values[s] = I->alu_src_value[s];
      } else if (file == AGX_APPLE9_FILE_IMMEDIATE) {
         uint8_t encoded;
         if (!agx_apple9_encode_float_immediate(I->alu_src_value[s], &encoded))
            return false;
         values[s] = I->alu_src_value[s];
      } else {
         if (source >= I->nr_srcs ||
             !agx_apple9_encoding_accepts_gpr(I->encoding, role,
                                              phys[I->src[source]], 32))
            return false;
         values[s] = phys[I->src[source]];
         *live |= ((I->live_after_mask >> source) & 1) << s;
         ++source;
      }
   }
   return source == I->nr_srcs;
}

static bool
pack_float2(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
            struct agx_apple9_packed_instruction *packed)
{
   if ((instruction->op == AGX_APPLE9_VIR_FMUL_PROJECT) !=
       (instruction->encoding == AGX_APPLE9_ENC_FLOAT2_PROJECT))
      return false;
   uint32_t src[3];
   uint8_t live;
   if (!apple9_alu_sources(instruction, phys, 2, src, &live) ||
       instruction->alu_src_immediate_mask ||
       ((instruction->src_abs_mask | instruction->src_neg_mask) & ~3))
      return false;
   if ((instruction->encoding == AGX_APPLE9_ENC_FLOAT2_PROJECT ||
        instruction->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT) &&
       (instruction->src_abs_mask || instruction->src_neg_mask))
      return false;
   uint8_t bytes[12] = {0x09, 0x05, 0x1c, 0x01, 0x00, 0x00};
   if (instruction->op == AGX_APPLE9_VIR_FMUL ||
       instruction->op == AGX_APPLE9_VIR_FMUL_PROJECT)
      set_bits(bytes, 16, 3,
               instruction->op == AGX_APPLE9_VIR_FMUL_PROJECT ? 7 : 5);
   else
      set_bits(bytes, 16, 3, 4);
   if ((instruction->op == AGX_APPLE9_VIR_FSUB) ^
       !!(instruction->src_neg_mask & 2))
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
   if (live & (1u << 0)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (live & (1u << 1)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }

   if (!pack_compact_binary_gprs(bytes, phys[instruction->dest],
                                 instruction->alu_src_uniform_mask & 1 ? 0 : src[0],
                                 instruction->alu_src_uniform_mask & 2 ? 0 : src[1],
                                 (instruction->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ? AGX_APPLE9_PUBLICATION_COUNT : AGX_APPLE9_GPR_COUNT)))
      return false;
   if (instruction->alu_src_uniform_mask & 1) {
      bytes[1] = ((src[0] & 63) << 1) | 1;
      set_bits(bytes, 39, 1, 1);
      set_bits(bytes, 19, 1, (src[0] >> 6) & 1);
      set_bits(bytes, 40, 1, src[0] >> 7);
   }
   if (instruction->alu_src_uniform_mask & 2) {
      bytes[3] = ((src[1] & 63) << 1) | 1;
      set_bits(bytes, 41, 1, 1);
      set_bits(bytes, 20, 1, (src[1] >> 6) & 1);
      set_bits(bytes, 42, 1, src[1] >> 7);
   }
   if (instruction->encoding == AGX_APPLE9_ENC_FLOAT2_PROJECT) {
      if (phys[instruction->dest] >= 64 || phys[instruction->src[0]] >= 64 ||
          phys[instruction->src[1]] >= 64 || instruction->immediate > AGX_APPLE9_MAX_VARYING_COMPONENTS)
         return false;
      /* Source-authored Metal probes use opcode 7 and name the CF slot
       * alongside the two GPR operands. Perspective coefficients can be
       * primitive-constant; this form preserves their constant fast path
       * instead of unconditionally multiplying them by reciprocal 1/W.
       * Byte 5 contains the coefficient index, including bit 5 at index 32.
       * It must not be overwritten by the ordinary FALU dependency encoder.
       * Pending sources are materialized by the common scoreboard lowering;
       * this form currently uses six-bit GPR operands. */
      bytes[5] = instruction->immediate;
      packed_init(packed, bytes, 8);
   } else if (instruction->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT) {
      /* Authored large-interface probes use the scattered register
       * map with the extended export destination publication. */
      if (phys[instruction->dest] >= AGX_APPLE9_PUBLICATION_COUNT ||
          phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT ||
          phys[instruction->src[1]] >= AGX_APPLE9_GPR_COUNT)
         return false;
      bytes[4] |= 0x41;
      if (instruction->producer_scoreboard_slot > AGX_APPLE9_SCOREBOARD_SLOT_6)
         return false;
      if (instruction->producer_scoreboard_slot)
         set_bits(bytes, 58, 3, instruction->producer_scoreboard_slot - 1);
      packed_init(packed, bytes, 8);
   } else {
      bool base = instruction->encoding == AGX_APPLE9_ENC_FLOAT2_BASE;
      bool negate = instruction->encoding == AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED;
      bool abs_add = instruction->encoding == AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED;
      bool abs_mul = instruction->encoding == AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED;
      if (instruction->encoding != AGX_APPLE9_ENC_FLOAT2_COMPACT &&
          !base && !negate && !abs_add && !abs_mul)
         return false;
      if ((instruction->src_abs_mask && !abs_add && !abs_mul) ||
          (instruction->saturate && !negate && !abs_add && !abs_mul) ||
          ((instruction->src_neg_mask & 1) && !negate && !abs_add && !abs_mul) ||
          (abs_add && instruction->op == AGX_APPLE9_VIR_FMUL) ||
          (abs_mul && instruction->op != AGX_APPLE9_VIR_FMUL))
         return false;
      if (base && (phys[instruction->dest] >= 64 ||
                   instruction->op == AGX_APPLE9_VIR_FSUB ||
                   instruction->src_abs_mask || instruction->src_neg_mask))
         return false;
      unsigned length = base ? 4 : 6;
      if (base)
         set_bits(bytes, 18, 1, 0);
      if (negate || abs_add || abs_mul) {
         set_bits(bytes, 32, 2, negate ? 1 : 2);
         set_bits(bytes, 49, 1, instruction->src_neg_mask & 1);
         if (abs_add || abs_mul)
            set_bits(bytes, 64, 2, instruction->src_abs_mask);
         length = negate ? 8 : abs_add ? 10 : 12;
      }
      if (instruction->saturate)
         set_bits(bytes, 57, 1, 1);
      packed_init(packed, bytes, length);
   }
   return true;
}

bool
agx_apple9_encode_float_immediate(uint32_t bits, uint8_t *encoded)
{
   const uint32_t magnitude = bits & 0x7fffffffu;
   *encoded = 0;
   bool found = false;

   for (unsigned exponent = 8; exponent < 16 && !found; ++exponent) {
      for (unsigned mantissa = 0; mantissa < 8; ++mantissa) {
         float value = exponent == 8 ? (mantissa / 8.0f) * 0.25f
                                     : (1.0f + mantissa / 8.0f) *
                                          (1u << (exponent - 8)) * 0.125f;
         uint32_t candidate;
         memcpy(&candidate, &value, sizeof(candidate));
         if (candidate == magnitude) {
            *encoded = (exponent << 4) | (mantissa << 1) | 1;
            found = true;
            break;
         }
      }
   }
   return found;
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
   bool extended = instruction->encoding == AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED;
   uint8_t encoded;
   if (instruction->nr_srcs != 1 ||
       (!extended && instruction->encoding != AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT) ||
       (instruction->saturate && !extended) ||
       !agx_apple9_encode_float_immediate(bits, &encoded) ||
       instruction->src_abs_mask || (instruction->src_neg_mask & ~1))
      return false;

   /*
    * This form is the caller-compiler form measured for a converted
    * fragment position feeding fadd-immediate.  It also matches the
    * independently validated packed-immediate operand layout.
    */
   uint8_t bytes[8] = {0x09, encoded, 0x34, 0x01, 0x80, 0x00};
   if (instruction->op == AGX_APPLE9_VIR_FMUL_IMM)
      set_bits(bytes, 16, 3, 5);
   set_bits(bytes, 19, 1, bits >> 31);
   unsigned dst = phys[instruction->dest], src = phys[instruction->src[0]];
   set_bits(bytes, 4, 4, dst & 15);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 44, 1, dst >> 6);
   set_bits(bytes, 25, 6, src & 63);
   set_bits(bytes, 42, 1, src >> 6);
   set_bits(bytes, 43, 1, instruction->src_neg_mask & 1);
   if (instruction->live_after_mask & 1) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }
   if (extended) {
      bytes[4] |= 1;
      set_bits(bytes, 57, 1, instruction->saturate);
   }
   packed_init(packed, bytes, extended ? 8 : 6);
   return true;
}

static bool
pack_fma(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
         struct agx_apple9_packed_instruction *packed)
{
   uint32_t source[3];
   uint8_t live;
   bool compact = instruction->encoding == AGX_APPLE9_ENC_FLOAT3_COMPACT;
   bool extended = instruction->encoding == AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED;
   bool sat = instruction->encoding == AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED;
   if ((instruction->encoding != AGX_APPLE9_ENC_FLOAT3_EXTENDED && !compact && !extended && !sat) ||
       (instruction->saturate && !extended && !sat) ||
       !apple9_alu_sources(instruction, phys, 3, source, &live) ||
       (instruction->alu_src_immediate_mask & 1) ||
       ((instruction->src_abs_mask | instruction->src_neg_mask) & ~7) ||
       ((instruction->src_abs_mask & 3) && !extended))
      return false;
   if (compact && ((instruction->alu_src_uniform_mask & 1) ||
                   ((instruction->alu_src_uniform_mask & 2) && source[1] >= 64) ||
                   instruction->saturate || instruction->src_abs_mask ||
                   instruction->src_neg_mask ||
                   ((instruction->alu_src_immediate_mask & 2) && (source[1] >> 31))))
      return false;
   uint8_t bytes[12] = {0x09, 0x01, 0x1e, 0x05, 0x81, 0x08, 0x02, 0x00};
   unsigned dst = phys[instruction->dest];
   if (!agx_apple9_encoding_accepts_gpr(instruction->encoding,
                                        AGX_APPLE9_OPERAND_DEST, dst, 32))
      return false;
   unsigned inline_mask = instruction->alu_src_uniform_mask |
                          instruction->alu_src_immediate_mask;
   unsigned a = (inline_mask & 1) ? 0 : source[0];
   unsigned b = (inline_mask & 2) ? 0 : source[1];
   unsigned c = (inline_mask & 4) ? 0 : source[2];
   unsigned neg_mask = instruction->src_neg_mask;

   /* Same compiler-native producer and source-release convention as falu2. */
   set_bits(bytes, 21, 1, 1);
   if (live & (1u << 0)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (live & (1u << 1)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }
   if (live & (1u << 2)) {
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
   if (instruction->alu_src_uniform_mask & 1) {
      bytes[1] = ((source[0] & 63) << 1) | 1;
      set_bits(bytes, 55, 1, 1);
      set_bits(bytes, 19, 1, (source[0] >> 6) & 1);
      set_bits(bytes, 56, 1, source[0] >> 7);
   }
   for (unsigned s = 1; s < 3; ++s) {
      if (!(inline_mask & BITFIELD_BIT(s)))
         continue;
      uint8_t encoded;
      if (instruction->alu_src_immediate_mask & BITFIELD_BIT(s)) {
         if (!agx_apple9_encode_float_immediate(source[s], &encoded))
            return false;
         if (!(instruction->src_abs_mask & BITFIELD_BIT(s)))
            neg_mask ^= (source[s] >> 31) << s;
      } else {
         encoded = ((source[s] & 63) << 1) | 1;
      }
      if (s == 1) {
         bytes[3] = encoded;
         set_bits(bytes, 57, 1, 1);
         set_bits(bytes, 20, 1, (instruction->alu_src_uniform_mask & 2) ? (source[s] >> 6) & 1 : 0);
         set_bits(bytes, 58, 1, (instruction->alu_src_uniform_mask & 2) ? source[s] >> 7 : 0);
      } else {
         bytes[5] = encoded & ~1;
         set_bits(bytes, 37, 1, 1);
         set_bits(bytes, 39, 1, (instruction->alu_src_uniform_mask & 4) ? (source[s] >> 6) & 1 : 0);
         set_bits(bytes, 38, 1, (instruction->alu_src_uniform_mask & 4) ? source[s] >> 7 : 0);
      }
   }
   set_bits(bytes, 35, 1, !!(instruction->src_abs_mask & 4));
   set_bits(bytes, 36, 1, !!(neg_mask & 4));
   set_bits(bytes, 59, 1, !!(neg_mask & 1) ^ !!(neg_mask & 2));
   if (compact) {
      /* The six-byte form moves source B's alternative-file selector from
       * bit 57 to bit 37. Source C and all GPRs remain six-bit operands;
       * it cannot carry the extended modifiers or a completion wait. */
      set_bits(bytes, 32, 2, 0);
      set_bits(bytes, 37, 1, !!(inline_mask & 2));
   } else if (extended) {
      set_bits(bytes, 32, 2, 3);
      set_bits(bytes, 80, 2, instruction->src_abs_mask & 3);
   } else if (sat) {
      set_bits(bytes, 32, 2, 2);
   }
   if (instruction->saturate)
      set_bits(bytes, 73, 1, 1);
   packed_init(packed, bytes, compact ? 6 : extended ? 12 : sat ? 10 : 8);
   return true;
}

/* Uniform index bits overlap GPR lifetime fields. Pack the selected file last. */
static void
pack_binary_uniforms(uint8_t *bytes, const struct agx_apple9_vir_instr *I,
                     const uint32_t source[3])
{
   for (unsigned s = 0; s < 2; ++s) {
      if (!(I->alu_src_uniform_mask & (1u << s)))
         continue;
      bytes[1 + 2 * s] = ((source[s] & 63) << 1) | 1;
      set_bits(bytes, s ? 41 : 39, 1, 1);
      set_bits(bytes, s ? 20 : 19, 1, (source[s] >> 6) & 1);
      set_bits(bytes, s ? 42 : 40, 1, source[s] >> 7);
   }
}

static bool
pack_logic(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
           struct agx_apple9_packed_instruction *packed)
{
   uint32_t source[3];
   uint8_t live;
   if (!apple9_alu_sources(instruction, phys, 2, source, &live) ||
       instruction->alu_src_immediate_mask || instruction->alu_src_uniform_mask == 3)
      return false;
   if ((instruction->encoding != AGX_APPLE9_ENC_LOGIC_EXTENDED &&
        instruction->encoding != AGX_APPLE9_ENC_LOGIC_EXPORT) ||
       instruction->nr_srcs + util_bitcount(instruction->alu_src_uniform_mask) != 2)
      return false;

   if (instruction->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT &&
       (phys[instruction->dest] >= AGX_APPLE9_PUBLICATION_COUNT ||
        phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT ||
        phys[instruction->src[1]] >= AGX_APPLE9_GPR_COUNT))
      return false;

   /* Begin without a pending-result dependency, including when used inside
    * a multi-instruction parameter publisher. The outer packer adds waits. */
   uint8_t bytes[10] = {0x0b, 0x05, 0x1f, 0x01, 0x00,
                        0x00, 0x00, 0x00, 0x00, 0x00};
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
    * Unlike compact FALU's binary three-bit slot selector, integer logic uses a
    * six-bit one-hot pending-result mask split across bits 45..47 and
    * 61..63.
    */
   if (live & (1u << 0)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (live & (1u << 1)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
      set_bits(bytes, 21, 1, 1);
   }
   if (!pack_compact_binary_gprs(bytes, phys[instruction->dest],
                                 instruction->alu_src_uniform_mask & 1 ? 0 : source[0],
                                 instruction->alu_src_uniform_mask & 2 ? 0 : source[1],
                                 (instruction->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT ? AGX_APPLE9_PUBLICATION_COUNT : AGX_APPLE9_GPR_COUNT)))
      return false;
   if (instruction->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT) {
      bytes[4] |= 0x40;
      set_bits(bytes, 21, 1, 1);
      if (instruction->producer_scoreboard_slot > AGX_APPLE9_SCOREBOARD_SLOT_6)
         return false;
      if (instruction->producer_scoreboard_slot)
         set_bits(bytes, 58, 3, instruction->producer_scoreboard_slot - 1);
   }
   pack_binary_uniforms(bytes, instruction, source);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_uniform_store(const struct agx_apple9_vir_instr *I, const uint8_t *phys,
                   struct agx_apple9_packed_instruction *packed)
{
   bool copy = I->encoding == AGX_APPLE9_ENC_STORE_UNIFORM;
   bool add = I->encoding == AGX_APPLE9_ENC_STORE_UNIFORM_ADD;
   bool mad = I->encoding == AGX_APPLE9_ENC_STORE_UNIFORM_MAD;
   bool wide = I->encoding == AGX_APPLE9_ENC_STORE_UNIFORM_MUL_WIDE;
   if ((!copy && !add && !mad && !wide) ||
       I->immediate >= AGX_APPLE9_UNIFORM_COUNT ||
       I->nr_srcs != (copy ? 1 : mad ? 3 : 2) ||
       (add && I->uniform_op != AGX_APPLE9_VIR_IADD && I->uniform_op != AGX_APPLE9_VIR_ISUB) ||
       (mad && I->uniform_op != AGX_APPLE9_VIR_IMAD) ||
       (wide && (I->uniform_op != AGX_APPLE9_VIR_IMUL_WIDE ||
                 (I->immediate & 1) || I->immediate + 1 >= AGX_APPLE9_UNIFORM_COUNT)))
      return false;
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      if (phys[I->src[s]] >= AGX_APPLE9_GPR_COUNT)
         return false;
   }
   uint8_t bytes[12] = {0x9f, 0x01, 0x54, 0, 0, 0, 0, 0xa8, 0x17, 0x05};
   if (mad || wide) {
      const uint8_t multiply[] = {0x9f, 0, 0x54, 0, 0, 0, 0, 0, 0xd0, 0x2f, 0x2a, 0};
      memcpy(bytes, multiply, sizeof(bytes));
      if (wide) {
         bytes[8] = 0xe0; bytes[9] = 0x26;
         bytes[10] = I->uniform_signed ? 0x1e : 0x0a;
      }
   } else if (add && I->uniform_op == AGX_APPLE9_VIR_ISUB) {
      bytes[0] = 0x1f;
   }
   /* COPY is IADD with an inline integer zero, not a read of GPR r0. */
   if (copy) {
      bytes[9] = 1;          /* Source B uses the alternative operand file. */
      set_bits(bytes, 66, 1, 0); /* Select an integer immediate, not a uniform. */
   }
   set_bits(bytes, 25, 8, I->immediate);
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      set_bits(bytes, 42 + 9 * s, 7, phys[I->src[s]]);
      if (I->live_after_mask & (1u << s)) {
         set_bits(bytes, 49 + 9 * s, 1, 1);
         set_bits(bytes, (mad || wide ? 73 : 65) + s, 1, 0);
      }
   }
   packed_init(packed, bytes, mad || wide ? 12 : 10);
   return true;
}

static bool
pack_uniform(const struct agx_apple9_vir_instr *ins, const uint8_t *phys,
             struct agx_apple9_packed_instruction *packed)
{
   if (ins->op == AGX_APPLE9_VIR_STORE_UNIFORM)
      return pack_uniform_store(ins, phys, packed);
   if (ins->nr_srcs != 1 || ins->immediate >= AGX_APPLE9_UNIFORM_COUNT)
      return false;
   unsigned src = phys[ins->src[0]];
   unsigned dst = phys[ins->dest];
   if (ins->encoding != AGX_APPLE9_ENC_LOGIC_UNIFORM ||
       dst >= AGX_APPLE9_GPR_COUNT || src >= AGX_APPLE9_GPR_COUNT)
      return false;
   uint8_t bytes[10] = {0x0b, 0x01, 0x37, 0x01, 0x82, 0x08};
   set_bits(bytes, 8, 7, 2 * (ins->immediate & 63) + 1);
   set_bits(bytes, 19, 1, (ins->immediate >> 6) & 1);
   set_bits(bytes, 40, 1, ins->immediate >> 7);
   set_bits(bytes, 4, 4, dst & 15);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 44, 1, dst >> 6);
   set_bits(bytes, 25, 6, src & 63);
   set_bits(bytes, 42, 1, src >> 6);
   if (ins->live_after_mask & 1) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_minmax(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
            struct agx_apple9_packed_instruction *packed)
{
   uint32_t source[3];
   uint8_t live;
   if (!apple9_alu_sources(instruction, phys, 2, source, &live) ||
       instruction->alu_src_immediate_mask || instruction->alu_src_uniform_mask == 3)
      return false;
   bool floating = instruction->op == AGX_APPLE9_VIR_FMIN ||
                   instruction->op == AGX_APPLE9_VIR_FMAX;
   if (instruction->encoding != AGX_APPLE9_ENC_MINMAX_COMPACT ||
       instruction->nr_srcs + util_bitcount(instruction->alu_src_uniform_mask) != 2 || instruction->src_abs_mask ||
       (instruction->src_neg_mask & ~(floating ? 2u : 0u)))
      return false;
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
   if (live & (1u << 0)) {
      set_bits(bytes, 15, 1, 1);
      set_bits(bytes, 19, 1, 0);
   }
   if (live & (1u << 1)) {
      set_bits(bytes, 31, 1, 1);
      set_bits(bytes, 20, 1, 0);
   }

   if (!pack_compact_binary_gprs(bytes, phys[instruction->dest],
                                 instruction->alu_src_uniform_mask & 1 ? 0 : source[0],
                                 instruction->alu_src_uniform_mask & 2 ? 0 : source[1],
                                 AGX_APPLE9_GPR_COUNT))
      return false;
   if (instruction->src_neg_mask & 2) {
      set_bits(bytes, 18, 1, 0);
      set_bits(bytes, 43, 1, 1);
   }
   pack_binary_uniforms(bytes, instruction, source);
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

static bool
pack_select(const struct agx_apple9_vir_instr *instruction, const uint8_t *phys,
            struct agx_apple9_packed_instruction *packed)
{
   bool extended = instruction->encoding == AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED;
   if ((instruction->encoding != AGX_APPLE9_ENC_SELECT_GPR_WIDE && !extended) ||
       instruction->nr_srcs != 4 ||
       ((instruction->src_abs_mask | instruction->src_neg_mask) & ~15u) ||
       (instruction->src_neg_mask & 1) ||
       ((instruction->src_abs_mask & 3) && !extended))
      return false;
   unsigned dst = phys[instruction->dest];
   unsigned cmp_a = phys[instruction->src[0]];
   unsigned cmp_b = phys[instruction->src[1]];
   unsigned if_true = phys[instruction->src[2]];
   unsigned if_false = phys[instruction->src[3]];
   if (dst >= AGX_APPLE9_GPR_COUNT || cmp_a >= AGX_APPLE9_GPR_COUNT ||
       cmp_b >= AGX_APPLE9_GPR_COUNT || if_true >= AGX_APPLE9_GPR_COUNT ||
       if_false >= AGX_APPLE9_GPR_COUNT)
      return false;

   /* All register banks share the same source lifetime fields. Start with
    * last-use operands, then retain the sources whose SSA values survive. */
   uint8_t bytes[14] = {0x02, 0x01, 0x1f, 0x01, 0x82,
                        0x00, 0x05, 0x00, 0x80, 0x00};

   const unsigned condition = instruction->immediate & 0xff;
   if ((instruction->src_abs_mask || instruction->src_neg_mask) &&
       condition != AGX_APPLE9_SELECT_FEQ && condition != AGX_APPLE9_SELECT_FGT &&
       condition != AGX_APPLE9_SELECT_FLT)
      return false;
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

   {
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

   set_bits(bytes, 4, 4, dst & 15);
   set_bits(bytes, 22, 2, (dst >> 4) & 3);
   set_bits(bytes, 60, 1, dst >> 6);
   set_bits(bytes, 9, 6, cmp_a & 0x3f);
   set_bits(bytes, 56, 1, cmp_a >> 6);
   set_bits(bytes, 25, 6, cmp_b & 0x3f);
   set_bits(bytes, 58, 1, cmp_b >> 6);
   set_bits(bytes, 41, 6, if_true & 0x3f);
   set_bits(bytes, 38, 1, if_true >> 6);
   set_bits(bytes, 73, 6, if_false & 0x3f);
   set_bits(bytes, 70, 1, if_false >> 6);
   if (extended) {
      set_bits(bytes, 32, 2, 3);
      set_bits(bytes, 96, 2, instruction->src_abs_mask & 3);
   }
   if ((instruction->src_abs_mask | instruction->src_neg_mask) & 12)
      set_bits(bytes, 18, 1, 0);
   set_bits(bytes, 35, 1, !!(instruction->src_abs_mask & 4));
   set_bits(bytes, 36, 1, !!(instruction->src_neg_mask & 4));
   set_bits(bytes, 67, 1, !!(instruction->src_abs_mask & 8));
   set_bits(bytes, 68, 1, !!(instruction->src_neg_mask & 8));
   set_bits(bytes, 59, 1, !!(instruction->src_neg_mask & 2));
   packed_init(packed, bytes, extended ? 14 : 10);
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
   if (src0 >= AGX_APPLE9_GPR_COUNT || src1 >= AGX_APPLE9_GPR_COUNT)
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
      uint8_t bytes[6] = {opcode,    (uint8_t)(((src0 & 63) << 1) | 1),
                          control,   (uint8_t)(((src1 & 63) << 1) | 1),
                          condition, 0xc0};
      set_bits(bytes, short_form ? 40 : 56, 1, src0 >> 6);
      set_bits(bytes, short_form ? 42 : 58, 1, src1 >> 6);
      packed_init(packed, bytes, sizeof(bytes));
   } else {
      uint8_t bytes[10] = {
         opcode,    (uint8_t)(((src0 & 63) << 1) | 1),
         control,   (uint8_t)(((src1 & 63) << 1) | 1),
         0x06,      0x00,
         condition, loop_form ? 0x00 : 0xc0,
         0x00,      0x00,
      };
      set_bits(bytes, short_form ? 40 : 56, 1, src0 >> 6);
      set_bits(bytes, short_form ? 42 : 58, 1, src1 >> 6);
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
   case AGX_APPLE9_VIR_UNPACK_NORM: {
      unsigned mode = instruction->immediate & 0xff;
      bool narrow = mode == AGX_APPLE9_UNPACK_UNORM8 || mode == AGX_APPLE9_UNPACK_SNORM8;
      bool high = instruction->immediate & AGX_APPLE9_UNPACK_HIGH_HALF;
      unsigned dst = phys[instruction->dest];
      if (instruction->encoding != AGX_APPLE9_ENC_UNPACK_NORM ||
          instruction->nr_srcs != 1 || instruction->dest_components != 2 ||
          (instruction->immediate & ~(0xffu | AGX_APPLE9_UNPACK_HIGH_HALF)) ||
          (!narrow && mode != AGX_APPLE9_UNPACK_UNORM16 && mode != AGX_APPLE9_UNPACK_SNORM16) ||
          (high && !narrow) || (dst & 1) || dst + 1 >= AGX_APPLE9_GPR_COUNT ||
          phys[instruction->dest + 1] != dst + 1 ||
          phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT)
         return false;
      uint8_t bytes[8] = {0x17, 0x04, 0x54, 0, 0, 0, 0x04, 0x8a};
      set_bits(bytes, 25, 7, dst);
      set_bits(bytes, 41, 1, high);
      set_bits(bytes, 42, 7, phys[instruction->src[0]]);
      set_bits(bytes, 49, 1, instruction->live_after_mask & 1);
      set_bits(bytes, 51, 1, !narrow);
      set_bits(bytes, 52, 1, !(instruction->live_after_mask & 1));
      set_bits(bytes, 60, 3, mode);
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_PACK_UNORM_2X16:
   case AGX_APPLE9_VIR_PACK_UNORM_4X8: {
      bool bytes = instruction->op == AGX_APPLE9_VIR_PACK_UNORM_4X8;
      enum agx_apple9_encoding encoding = bytes ? AGX_APPLE9_ENC_PACK_UNORM_4X8
                                                : AGX_APPLE9_ENC_PACK_UNORM_2X16;
      unsigned dst = phys[instruction->dest];
      if (instruction->encoding != encoding || instruction->immediate ||
          instruction->nr_srcs != (bytes ? 4 : 2) ||
          dst >= AGX_APPLE9_GPR_COUNT)
         return false;

      /* The 8-bit form writes one destination half per pair of FP32 inputs;
       * the 16-bit form defines the complete word. Keep all inputs until
       * both halves are written, including repeated/swizzled components.
       * Shared RA's late-kill contract prevents cross-instruction aliasing.
       * T8132 probes validate both halves, all seven register bits, clamping
       * and round-to-nearest-even conversion of the exact scaled input. */
      uint8_t code[20] = {0};
      for (unsigned pair = 0; pair < (bytes ? 2 : 1); ++pair) {
         unsigned a = phys[instruction->src[2 * pair]];
         unsigned b = phys[instruction->src[2 * pair + 1]];
         if (a >= AGX_APPLE9_GPR_COUNT || b >= AGX_APPLE9_GPR_COUNT ||
             a == dst || b == dst)
            return false;
         uint8_t *part = code + 10 * pair;
         memcpy(part, (uint8_t[]){0x97, 0x04, 0x56, 0, 0x02,
                                  0, 0, 0x50, 0x44, 0x82}, 10);
         set_bits(part, 25, 7, dst);
         set_bits(part, 42, 7, a);
         set_bits(part, 51, 7, b);
         if (bytes) {
            part[2] = 0x54;
            set_bits(part, 24, 1, pair);
            part[9] = 0xc2;
         }
      }
      packed_init(packed, code, bytes ? 20 : 10);
      return true;
   }
   case AGX_APPLE9_VIR_F2F16:
   case AGX_APPLE9_VIR_PACK_HALF_2X16:
   case AGX_APPLE9_VIR_UNPACK_HALF: {
      bool pair = instruction->op == AGX_APPLE9_VIR_PACK_HALF_2X16;
      bool widen = instruction->op == AGX_APPLE9_VIR_UNPACK_HALF;
      enum agx_apple9_encoding encoding = pair ? AGX_APPLE9_ENC_PACK_HALF_2X16
         : widen ? AGX_APPLE9_ENC_HALF_TO_FLOAT
                 : AGX_APPLE9_ENC_FLOAT_TO_HALF_ZEXT;
      unsigned dst = phys[instruction->dest];
      if (instruction->encoding != encoding ||
          instruction->nr_srcs != (pair ? 2 : 1) ||
          dst >= (pair || widen ? AGX_APPLE9_GPR_COUNT : 64) ||
          instruction->immediate > (widen ? 1 : 0))
         return false;

      /* Compact FALU conversion shares the scattered GPR map. A narrowing
       * write changes just one destination half. Both halves are defined
       * here: either two conversions or the native zero-extension companion.
       * The paired form materializes pending sources before entering this
       * sequence, since its two conversions have separate dependency fields. */
      uint8_t bytes[12] = {0};
      for (unsigned s = 0; s < (pair ? 2 : 1); ++s) {
         unsigned src = phys[instruction->src[s]];
         if (src >= AGX_APPLE9_GPR_COUNT || (pair && src == dst))
            return false;
         uint8_t *part = bytes + 6 * s;
         memcpy(part, (uint8_t[]){0x01, 0x01, 0x1c, 0x81, 0x00, 0x02}, 6);
         set_bits(part, 4, 4, dst & 15);
         set_bits(part, 22, 2, (dst >> 4) & 3);
         set_bits(part, 44, 1, dst >> 6);
         set_bits(part, 9, 6, src & 63);
         set_bits(part, 40, 1, src >> 6);
         set_bits(part, 34, 1, s);
         if (widen) {
            set_bits(part, 0, 4, 9);
            set_bits(part, 8, 1, 0);
            set_bits(part, 35, 1, instruction->immediate);
         }
         if ((instruction->live_after_mask & (1u << s)) ||
             (pair && s == 0 && instruction->src[0] == instruction->src[1])) {
            set_bits(part, 15, 1, 1);
            set_bits(part, 19, 1, 0);
         }
      }
      if (!widen && !pair) {
         bytes[6] = ((dst & 15) << 4) | 3;
         bytes[8] = (dst >> 4) << 6;
         bytes[9] = 1;
      }
      packed_init(packed, bytes, pair ? 12 : widen ? 6 : 10);
      return true;
   }
   case AGX_APPLE9_VIR_IMUL_WIDE:
      return pack_mul_wide(instruction, phys, packed);
   case AGX_APPLE9_VIR_SPILL_STORE: {
      unsigned slot = instruction->producer_scoreboard_slot;
      unsigned word = instruction->immediate;
      if (instruction->encoding != AGX_APPLE9_ENC_SPILL_STORE ||
          instruction->nr_srcs != 1 || instruction->live_after_mask ||
          instruction->dest != AGX_APPLE9_VREG_INVALID || word >= 8192 ||
          slot < 1 || slot > 6 || phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT)
         return false;
      unsigned src = phys[instruction->src[0]];
      const uint8_t bytes[] = {0x0b | ((word & 15) << 4),
         1 | ((src & 63) << 1), 0x0f | (((word >> 4) & 3) << 6), 0, 0x22,
         (src >> 6) | (((word >> 6) & 1) << 4), (word >> 7) << 2,
         (slot - 1) << 2, 0, 0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_SPILL_LOAD: {
      unsigned slot = instruction->producer_scoreboard_slot;
      unsigned word = instruction->immediate;
      if (instruction->encoding != AGX_APPLE9_ENC_SPILL_LOAD || instruction->nr_srcs ||
          word >= 8192 || slot < 1 || slot > 6 || phys[instruction->dest] >= AGX_APPLE9_GPR_COUNT)
         return false;
      unsigned dst = phys[instruction->dest];
      /* Retain the scratch word: SSA uses and loop iterations may reload it. */
      const uint8_t bytes[] = {0x0c | ((dst & 15) << 4),
         0x80 | ((word & 63) << 1), 0x01 | (((dst >> 4) & 3) << 6),
         1 | ((slot - 1) << 5), (word >> 6) << 1, 0, 0, (dst >> 6) << 4};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
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
      if (instruction->encoding == AGX_APPLE9_ENC_GET_DRAW_ID ||
          instruction->encoding == AGX_APPLE9_ENC_GET_COVERAGE) {
         bool coverage = instruction->encoding == AGX_APPLE9_ENC_GET_COVERAGE;
         const unsigned dst = phys[instruction->dest];
         bool valid_selector = coverage ? instruction->immediate == 0x10c2
            : (instruction->immediate == 0x10dd || instruction->immediate == 0x10d8);
         if (dst >= 64 || !valid_selector || instruction->nr_srcs ||
             instruction->producer_scoreboard_slot !=
                AGX_APPLE9_SCOREBOARD_SLOT_1)
            return false;
         /* The physical scheduler owns the completion handoff. This SR
          * encoding publishes to slot 1; synchronous SR forms remain distinct. */
         const uint8_t bytes[4] = {
            ((dst & 15) << 4) | 0x0c,
            instruction->immediate & 0xff,
            0x10 | ((dst >> 4) << 6),
            0x06,
         };
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
          (instruction->encoding != AGX_APPLE9_ENC_DEVICE_LOAD &&
           instruction->encoding != AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT) ||
          (instruction->nr_srcs != 1 && instruction->nr_srcs != 3))
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
      bool ok;
      if (components == 1)
         ok = agx_apple9_pack_device_load_scalar_raw(
            phys[instruction->dest], index, instruction->immediate, memory_bits,
            index_lifetime, instruction->device_load_flags,
            AGX_APPLE9_SCOREBOARD_SLOT_NONE,
            instruction->device_load_raw_token, packed);
      else
         ok = memory_bits == 32 &&
              agx_apple9_pack_device_load_vector_u32_raw(
                 phys[instruction->dest], index, instruction->immediate,
                 components, index_lifetime, instruction->device_load_flags,
                 AGX_APPLE9_SCOREBOARD_SLOT_NONE,
                 instruction->device_load_raw_token, packed);
      if (!ok)
         return false;
      if (instruction->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT) {
         const unsigned address = phys[instruction->src[1]];
         if (address >= AGX_APPLE9_GPR_COUNT - 1 ||
             phys[instruction->src[2]] != address + 1)
            return false;
         /* Authored T8132 pointer-chasing mains: select a GPR address pair,
          * independently of the element index and the return scoreboard. */
         packed->bytes[4] = address;
         packed->bytes[12] |= 0x20;
         if (!(instruction->live_after_mask & 6))
            packed->bytes[9] |= 4;
      }
      return pack_memory_address(instruction, packed);
   }
   case AGX_APPLE9_VIR_DEVICE_STORE: {
      const unsigned components = instruction->memory_components;
      const bool indirect =
         instruction->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT;
      if (components < 1 || components > 4 ||
          instruction->nr_srcs != components + (indirect ? 3 : 1) ||
          instruction->immediate > UINT8_MAX ||
          (instruction->encoding != AGX_APPLE9_ENC_DEVICE_STORE && !indirect))
         break;

      const unsigned data = phys[instruction->src[0]];
      const unsigned index = phys[instruction->src[components]];
      const bool release_index =
         !(instruction->live_after_mask & (1u << components));
      for (unsigned c = 1; c < components; ++c) {
         if (phys[instruction->src[c]] != data + c)
            return false;
      }

      bool ok;
      if (components == 1)
         ok = agx_apple9_pack_device_store_scalar(
            data, index, instruction->immediate, instruction->memory_bits,
            instruction->scoreboard_slot, release_index, packed);
      else
         ok = agx_apple9_pack_device_store_vector_u32(
            data, index, instruction->immediate, components,
            instruction->scoreboard_slot, release_index, packed);
      if (!ok)
         return false;
      if (indirect) {
         unsigned address = phys[instruction->src[components + 1]];
         if (address >= AGX_APPLE9_GPR_COUNT - 1 ||
             phys[instruction->src[components + 2]] != address + 1)
            return false;
         packed->bytes[4] = address;
         packed->bytes[12] |= 0x08;
      }
      return pack_memory_address(instruction, packed);
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
      if (destination >= AGX_APPLE9_GPR_COUNT ||
          slot < AGX_APPLE9_SCOREBOARD_SLOT_1 ||
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
         (uint8_t)(0x09 | (((destination >> 4) & 3) << 6)),
         0xa7,
         0x00,
         (uint8_t)(publication << 5),
         0x00,
         (uint8_t)((destination >> 6) << 4),
      };
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_ITER_FLAT: {
      if (instruction->encoding != AGX_APPLE9_ENC_ITER_FLAT ||
          instruction->nr_srcs || instruction->dest_components != 3 ||
          phys[instruction->dest] > AGX_APPLE9_GPR_COUNT - 3 ||
          instruction->immediate > AGX_APPLE9_MAX_VARYING_COMPONENTS ||
          instruction->producer_scoreboard_slot <
             AGX_APPLE9_SCOREBOARD_SLOT_1 ||
          instruction->producer_scoreboard_slot > AGX_APPLE9_SCOREBOARD_SLOT_6)
         return false;
      /* Asynchronously read the coefficient tuple. Its third word is the
       * selected vertex's unmodified value, including integer bit patterns.
       * Byte 5 publishes the result to a zero-based scoreboard slot; the
       * first consumer must take the ordinary pending-result handoff. */
      unsigned index = instruction->immediate;
      const uint8_t bytes[] = {
         0x1f,
         0x03,
         0x54,
         phys[instruction->dest] | ((index & 1) << 7),
         index >> 1,
         instruction->producer_scoreboard_slot - AGX_APPLE9_SCOREBOARD_SLOT_1};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_DERIVATIVE: {
      if (instruction->encoding != AGX_APPLE9_ENC_DERIVATIVE ||
          instruction->nr_srcs != 1 || instruction->immediate > 1 ||
          phys[instruction->dest] >= AGX_APPLE9_GPR_COUNT ||
          phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT)
         return false;
      const uint8_t bytes[] = {
         0x37,
         0x05 | (instruction->immediate << 1),
         0x54,
         phys[instruction->dest] << 1,
         0x03,
         (phys[instruction->src[0]] & 63) << 2,
         0x90 | ((instruction->live_after_mask & 1) ? 2 : 0) |
            (phys[instruction->src[0]] >> 6),
         0x40 | (instruction->saturate ? 4 : 0),
         0,
         0,
      };
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_CENTROID_POSITION: {
      if (instruction->encoding != AGX_APPLE9_ENC_CENTROID_POSITION ||
          instruction->nr_srcs != 1 || instruction->immediate ||
          phys[instruction->dest] >= AGX_APPLE9_GPR_COUNT ||
          phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT)
         return false;
      /* Authored M4 centroid and mixed-interpolation shaders: coverage is an
       * ordinary GPR input; the result is a packed interpolation position. */
      const uint8_t bytes[] = {0xaf,
                               0x04,
                               0x54,
                               phys[instruction->dest] << 1,
                               0x03,
                               (phys[instruction->src[0]] & 63) << 2,
                               0x0a | (phys[instruction->src[0]] >> 6),
                               0x01};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_ITER: {
      bool explicit_coord = instruction->encoding == AGX_APPLE9_ENC_ITER_COORD;
      if ((!explicit_coord && instruction->encoding != AGX_APPLE9_ENC_ITER) ||
          instruction->nr_srcs != (explicit_coord ? 1 : 0) ||
          (explicit_coord &&
           phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT) ||
          phys[instruction->dest] >= AGX_APPLE9_GPR_COUNT ||
          instruction->immediate > AGX_APPLE9_MAX_VARYING_COMPONENTS + 3)
         return false;
      /* Ordinary center coefficients: 0 for 1/W, followed by user components
       * and optional depth and point-coordinate coefficients.
       * Keep byte1 bit3 clear: setting it on the first ITER makes implicit
       * texture LOD fail in partially covered quads (mipmap edge probe).
       * Its earlier interpretation as a first-iterator marker was wrong. */
      unsigned imm = instruction->immediate;
      const uint8_t bytes[] = {0x2f, 0x05,
                               0x54, phys[instruction->dest] << 1,
                               0x03, (imm & 0xff) << 1,
                               explicit_coord ? phys[instruction->src[0]] : 0, 2,
                               explicit_coord ? ((instruction->live_after_mask & 1) ? 0x08 : 0x20) : 0x10, 0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_VARY_STORE: {
      if (instruction->encoding != AGX_APPLE9_ENC_VARY_STORE ||
          instruction->nr_srcs != 1 || instruction->immediate >= AGX_APPLE9_MAX_VARYING_COMPONENTS + 5 ||
          phys[instruction->src[0]] >= AGX_APPLE9_PUBLICATION_COUNT)
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
   case AGX_APPLE9_VIR_DEPTH_STORE: {
      if (instruction->encoding != AGX_APPLE9_ENC_DEPTH_STORE ||
          instruction->nr_srcs != 2 || phys[instruction->src[0]] >= 63 ||
          (phys[instruction->src[0]] & 1) ||
          phys[instruction->src[1]] != phys[instruction->src[0]] + 1)
         return false;
      uint8_t bytes[] = {0xd7, 0x14, 0x54,
                         phys[instruction->src[0]] << 1, 0, 3};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_COVERAGE: {
      if (instruction->encoding != AGX_APPLE9_ENC_COVERAGE ||
          instruction->nr_srcs != 1 ||
          phys[instruction->src[0]] >= 64)
         return false;
      uint8_t bytes[] = {0x57, 0x14, 0x54,
                         phys[instruction->src[0]] << 1, 0, 1};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TILE_ACCESS:
   case AGX_APPLE9_VIR_TILE_FENCE: {
      const bool access = instruction->op == AGX_APPLE9_VIR_TILE_ACCESS;
      if (instruction->nr_srcs ||
          instruction->encoding != (access ? AGX_APPLE9_ENC_TILE_ACCESS
                                           : AGX_APPLE9_ENC_TILE_FENCE) ||
          (access ? instruction->immediate != 0x700 &&
                       instruction->immediate != 0xf00 &&
                       instruction->immediate != 0x600 &&
                       (instruction->immediate & ~0xfc) != 0x800 &&
                       instruction->immediate != 1
                  : instruction->immediate != 0x300 &&
                       (instruction->immediate & ~0xfc) != 0 &&
                       (instruction->immediate & ~0xfc) != 0x200 &&
                       instruction->immediate != 1))
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
   case AGX_APPLE9_VIR_PUBLICATION_TUPLE: {
      if (instruction->encoding == AGX_APPLE9_ENC_PUBLICATION_PAIR) {
         unsigned base = phys[instruction->dest];
         if (instruction->nr_srcs != 2 || instruction->dest_components != 2 ||
             base > 62 || (base & 1))
            return false;
         uint8_t bytes[20];
         for (unsigned c = 0; c < 2; ++c) {
            unsigned source = instruction->src[c];
            bool keep = (instruction->live_after_mask & (1u << c)) ||
                        (c == 0 && instruction->src[1] == source);
            uint8_t registers[] = {base + c, phys[source]};
            struct agx_apple9_vir_instr publish = {
               .op = AGX_APPLE9_VIR_IOR,
               .encoding = AGX_APPLE9_ENC_LOGIC_EXPORT,
               .dest = 0, .nr_srcs = 2, .src = {1, 1},
               .live_after_mask = keep ? 3 : 0,
            };
            struct agx_apple9_packed_instruction part;
            if (!pack_logic(&publish, registers, &part) || part.length != 10)
               return false;
            set_bits(part.bytes, 25, 6, 0);
            set_bits(part.bytes, 42, 1, 0);
            set_bits(part.bytes, 31, 1, 0);
            set_bits(part.bytes, 20, 1, 0);
            set_bits(part.bytes, 24, 1, 0);
            set_bits(part.bytes, 43, 1, 0);
            part.bytes[8] |= 0x10;
            memcpy(bytes + 10 * c, part.bytes, 10);
         }
         packed_init(packed, bytes, sizeof(bytes));
         return true;
      }

      if (instruction->encoding == AGX_APPLE9_ENC_BLOCK_STORE_PARAMS ||
          instruction->encoding == AGX_APPLE9_ENC_BLOCK_STORE_MS_PARAMS ||
          instruction->encoding == AGX_APPLE9_ENC_TEXTURE_LOD_PARAMS ||
          instruction->encoding == AGX_APPLE9_ENC_TEXTURE_VOLUME_PARAMS ||
          instruction->encoding == AGX_APPLE9_ENC_TEXTURE_GRAD_PARAMS) {
         bool block_ms = instruction->encoding == AGX_APPLE9_ENC_BLOCK_STORE_MS_PARAMS;
         bool block = block_ms || instruction->encoding == AGX_APPLE9_ENC_BLOCK_STORE_PARAMS;
         bool gradient = instruction->encoding == AGX_APPLE9_ENC_TEXTURE_GRAD_PARAMS;
         bool volume = instruction->encoding == AGX_APPLE9_ENC_TEXTURE_VOLUME_PARAMS;
         unsigned base = phys[instruction->dest];
         if (instruction->nr_srcs != (gradient ? 6 : (volume || block_ms) ? 4 : 3) ||
             instruction->dest_components != (gradient ? 8 : 4) ||
             base > (block ? 12 : gradient ? 24 : 28) || (base & 3))
            return false;
         uint8_t bytes[64];
         for (unsigned c = 0; c < (gradient ? 6 : (volume || block_ms) ? 4 : 3); ++c) {
            unsigned source = instruction->src[c];
            bool keep = instruction->live_after_mask & (1u << c);
            for (unsigned j = c + 1; j < instruction->nr_srcs; ++j)
               keep |= instruction->src[j] == source;
            uint8_t registers[] = {base + (block || volume ? c : gradient && c >= 2 ? c + 2 : c == 2 ? 3 : c), phys[source]};
            struct agx_apple9_vir_instr publish = {
               .op = AGX_APPLE9_VIR_IOR,
               .encoding = AGX_APPLE9_ENC_LOGIC_EXPORT,
               .dest = 0, .nr_srcs = 2, .src = {1, 1},
               .live_after_mask = keep ? 3 : 0,
            };
            struct agx_apple9_packed_instruction part;
            if (!pack_logic(&publish, registers, &part) || part.length != 10)
               return false;
            memcpy(bytes + 10 * c, part.bytes, 10);
         }
         packed_init(packed, bytes, gradient ? 60 : (volume || block_ms) ? 40 : 30);
         return true;
      }
      if (instruction->encoding != AGX_APPLE9_ENC_TEXTURE_COORDS ||
          instruction->nr_srcs != 3 || instruction->dest_components != 2 ||
          phys[instruction->dest] > 30 || (phys[instruction->dest] & 1))
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
   case AGX_APPLE9_VIR_BLOCK_IMAGE_STORE: {
      unsigned coord = phys[instruction->src[0]];
      bool multisampled = instruction->encoding == AGX_APPLE9_ENC_BLOCK_IMAGE_STORE_MS;
      if ((!multisampled && instruction->encoding != AGX_APPLE9_ENC_BLOCK_IMAGE_STORE) ||
          instruction->nr_srcs != (multisampled ? 4 : 3) || coord > 12 || (coord & 3) ||
          phys[instruction->src[1]] != coord + 1 ||
          phys[instruction->src[2]] != coord + 2 ||
          (multisampled && phys[instruction->src[3]] != coord + 3) ||
          instruction->texture_index >= 16 || instruction->immediate > 15)
         return false;
      /* Publication: (X, Y, offset << 16) for 2D, and
       * (X, Y, 0, offset << 16) for MSAA. The MSAA mode exports every
       * sample from the implicit tile. The PBE descriptor supplies layout,
       * bounds, conversion and sample count. Release the complete tuple
       * and synchronously join export slot 1. */
      unsigned mode = multisampled ? 0x228 : 0x5a8;
      const uint8_t bytes[] = {
         0x57, 0x10, 0x54, coord << 1, 0, instruction->texture_index << 4,
         0, 0x80, mode & 0xff,
         (instruction->immediate << 4) | (mode >> 8), 1, 0,
         0x07, 0x12, 0x54, 0, 3, 0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE: {
      bool lod = instruction->encoding == AGX_APPLE9_ENC_TEXTURE_LOD;
      bool gradient = instruction->encoding == AGX_APPLE9_ENC_TEXTURE_GRAD;
      bool extended = lod || gradient;
      unsigned mask = instruction->texture_result_mask ?
         instruction->texture_result_mask : 0xf;
      if ((!extended && instruction->encoding != AGX_APPLE9_ENC_TEXTURE_SAMPLE) ||
          instruction->nr_srcs != (gradient ? 1 : extended ? 4 : 2) ||
          mask > 0xf || instruction->dest_components != util_bitcount(mask) ||
          instruction->texture_index >= 16 || instruction->sampler_index >= AGX_APPLE9_GRAPHICS_MAX_SAMPLERS ||
          instruction->immediate > (lod ? 1u : 0u) ||
          instruction->producer_scoreboard_slot < AGX_APPLE9_SCOREBOARD_SLOT_1 ||
          instruction->producer_scoreboard_slot > AGX_APPLE9_SCOREBOARD_SLOT_6)
         return false;
      unsigned dst = phys[instruction->dest], coord = phys[instruction->src[0]];
      if (dst + instruction->dest_components > 64 ||
          coord > (gradient   ? 24
                   : extended ? 28
                              : 30) ||
          (coord & (extended ? 3 : 1)) ||
          (!gradient && phys[instruction->src[1]] != coord + 1) ||
          (!gradient && extended &&
           (phys[instruction->src[2]] != coord + 2 ||
            phys[instruction->src[3]] != coord + 3)))
         return false;
      /* The four-byte prefix names the result tuple and coordinate
       * publication. The legacy database's sampler-byte 'coord' label is
       * not this operand. Direct texture/sampler table indices are distinct
       * fields, isolated with dependent and crossed own-source M4 samples. */
      unsigned texture = instruction->texture_index;
      unsigned sampler = instruction->sampler_index;
      /* Keep helper invocations alive for subsequent samples and derivatives.
       * Byte 8 bit 0 retires them after this sample. Our authored multi-sample
       * shader sets it only on its last sample; like the common AGX packer,
       * leave it clear until helper liveness is explicitly modeled.
       */
      uint8_t bytes[] = {
         5 | ((dst & 31) << 3), 0x80 | (coord & 7) | ((texture >> 1) << 3),
         0x0c | ((dst >> 5) << 6), 0xb8 | (coord >> 3),
         0xb0 | (sampler >> 1), 0, 0, 0, (texture & 1) << 7,
         sampler & 1, 0x10, 0, 1, 0};
      /* Texture completion uses paired zero-based tag fields. Matching both
       * fields is the native form, validated by retagging our own T8132
       * read and filtered-sample shaders with multiple outstanding results. */
      unsigned tag = instruction->producer_scoreboard_slot - 1;
      bytes[5] = (tag << 2) | (tag << 5);
      if (gradient) {
         bytes[6] = 4;
         bytes[7] = 1;
         bytes[8] = (texture & 1) << 7;
      }
      if (lod) {
         bytes[6] = instruction->immediate ? 4 : 0;
         bytes[7] = instruction->immediate ? 0 : 1;
      }
      /* Array samples share word 3: uint16 layer in the low half,
       * signed Q6 LOD in bits 16..27. The array mode must retain the
       * dynamic LOD selector; the immediate-LOD form ignores those bits. */
      if (instruction->texture_dimension == 2) {
         bytes[6] = instruction->immediate ? 0x0c : 0x08;
         bytes[7] = instruction->immediate ? 4 : 5;
      }
      if (instruction->texture_dimension == 3) {
         bytes[6] = instruction->immediate ? 0x90 : 0x88;
         bytes[7] = instruction->immediate ? 0x01 : 0x21;
      } else if (instruction->texture_dimension == 1) {
         bytes[6] = instruction->immediate ? 0x14 : 0x00;
         bytes[7] = instruction->immediate ? 0x00 : 0x03;
      }
      if (instruction->texture_fetch) {
         if (!lod || instruction->texture_shadow || instruction->immediate ||
             instruction->texture_dimension == 1)
            return false;
         bytes[6] = instruction->texture_dimension == 2 ? 0xa0 :
                    instruction->texture_dimension == 3 ? 0x98 : 0x80;
         bytes[7] = instruction->texture_dimension == 3 ? 1 : 0x24;
         bytes[10] = 0;
      }
      if (instruction->texture_shadow) {
         bytes[6] |= 0x20;
         bytes[10] = 0;
      }
      /* T8132 authored read/sample probes cover all 15 nonempty masks.
       * This four-bit selector has a permuted encoding; it is independent
       * of the resource, sampler, coordinate, LOD and completion fields.
       * Results are dense and ordered R, G, B, A among the selected lanes. */
      static const uint8_t selectors[16] = {
         0, 0, 1, 5, 2, 8, 9, 6, 3, 4, 10, 13, 11, 12, 14, 7,
      };
      unsigned selector = selectors[mask];
      set_bits(bytes, 27, 2, selector & 3);
      set_bits(bytes, 37, 1, (selector >> 2) & 1);
      set_bits(bytes, 86, 1, selector >> 3);
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TILE_LOAD: {
      bool coords = instruction->encoding == AGX_APPLE9_ENC_TILE_LOAD_COORDS;
      bool dynamic = instruction->encoding == AGX_APPLE9_ENC_TILE_LOAD_MASK;
      if ((!coords && !dynamic &&
           instruction->encoding != AGX_APPLE9_ENC_TILE_LOAD) ||
          instruction->nr_srcs != (coords || dynamic ? 1 : 0) ||
          instruction->immediate >= 32 ||
          phys[instruction->dest] >= AGX_APPLE9_GPR_COUNT ||
          ((coords || dynamic) &&
           phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT) ||
          (coords && (!instruction->tile_sample_mask ||
                      instruction->tile_sample_mask > 15)) ||
          instruction->producer_scoreboard_slot <
             AGX_APPLE9_SCOREBOARD_SLOT_1 ||
          instruction->producer_scoreboard_slot > AGX_APPLE9_SCOREBOARD_SLOT_6)
         return false;
      unsigned tag = 0x4e | ((instruction->producer_scoreboard_slot - 1) << 7);
      /* The dynamic sample mask addresses a 16-bit register half. Our
       * word-allocated masks occupy the low half; the operand low bit selects
       * the other half, not last use. Bit 69 independently releases the mask
       * after this read (native Metal lifetime probes, EXP-M4-75). */
      unsigned mask = coords    ? instruction->tile_sample_mask
                      : dynamic ? (phys[instruction->src[0]] << 1)
                                : 1;
      const uint8_t bytes[] = {0x67,
                               coords    ? 0x16
                               : dynamic ? 0x06
                                         : 0x0e,
                               0x54,
                               phys[instruction->dest] << 1,
                               coords ? phys[instruction->src[0]] : 0,
                               instruction->immediate << 1,
                               mask,
                               tag & 0xff,
                               (tag >> 8) |
                                  (dynamic && !(instruction->live_after_mask & 1)
                                     ? 0x20 : 0),
                               0,
                               0,
                               coords    ? 0x10
                               : dynamic ? 0x20
                                         : 0};
      packed_init(packed, bytes, sizeof(bytes));
      return true;
   }
   case AGX_APPLE9_VIR_TILE_STORE: {
      bool dynamic = instruction->nr_srcs == 2;
      if (instruction->encoding != (dynamic ? AGX_APPLE9_ENC_TILE_STORE_MASK
                                            : AGX_APPLE9_ENC_TILE_STORE) ||
          instruction->nr_srcs < 1 || instruction->nr_srcs > 2 ||
          instruction->immediate >= 32 ||
          phys[instruction->src[0]] >= AGX_APPLE9_GPR_COUNT ||
          (dynamic && phys[instruction->src[1]] >= AGX_APPLE9_GPR_COUNT))
         return false;
      /* Raw tile words with an independently allocated sample-mask source. */
      unsigned mask = dynamic ? phys[instruction->src[1]] << 1 : 1;
      const uint8_t bytes[] = {0xe7, dynamic ? 0x16 : 0x06, 0x54,
         phys[instruction->src[0]] << 1, 0, instruction->immediate << 1,
         mask, 0x4e, dynamic && !(instruction->live_after_mask & 2) ? 8 : 0, 0, 0,
         dynamic ? 8 : 0};
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
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHL:
      return pack_shift(instruction, phys, packed);
   case AGX_APPLE9_VIR_IMUL:
      break;
   case AGX_APPLE9_VIR_IADD:
   case AGX_APPLE9_VIR_ISUB:
      return pack_iadd(instruction, phys, packed);
   case AGX_APPLE9_VIR_CUBE:
      return pack_cube(instruction, phys, packed);
   case AGX_APPLE9_VIR_IMAD:
      return pack_imad(instruction, phys, packed);
   case AGX_APPLE9_VIR_IAND:
   case AGX_APPLE9_VIR_IOR:
   case AGX_APPLE9_VIR_IXOR:
      return pack_logic(instruction, phys, packed);
   case AGX_APPLE9_VIR_IOR_UNIFORM:
   case AGX_APPLE9_VIR_STORE_UNIFORM:
      return pack_uniform(instruction, phys, packed);
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
   case AGX_APPLE9_VIR_FSAT: {
      struct agx_apple9_vir_instr clamp = *instruction;
      clamp.op = AGX_APPLE9_VIR_FADD_IMM;
      clamp.immediate = 0;
      clamp.saturate = true;
      return pack_float2_immediate(&clamp, phys, packed);
   }
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
      return agx_apple9_pack_branch(any, displacement, packed);
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
   case AGX_APPLE9_VIR_PHI:
   case AGX_APPLE9_VIR_PHI_SRC:
      break;
   }

   if (reason != NULL)
      *reason = "Apple9 packer cannot encode this virtual instruction";
   return false;
}

bool
agx_apple9_pack_branch(bool any, int64_t displacement,
                       struct agx_apple9_packed_instruction *packed)
{
   uint8_t bytes[10];
   if (!packed || !agx_apple9_encode_branch(bytes, any, displacement))
      return false;
   packed_init(packed, bytes, sizeof(bytes));
   return true;
}

bool
agx_apple9_pack_vir_instruction(const struct agx_apple9_vir_instr *instruction,
                                const uint8_t *phys,
                                struct agx_apple9_packed_instruction *packed,
                                const char **reason)
{
   if (reason != NULL)
      *reason = NULL;

   if ((instruction->alu_src_uniform_mask || instruction->alu_src_immediate_mask) &&
       instruction->op != AGX_APPLE9_VIR_FADD &&
       instruction->op != AGX_APPLE9_VIR_FSUB &&
       instruction->op != AGX_APPLE9_VIR_FMUL &&
       instruction->op != AGX_APPLE9_VIR_FMA &&
       instruction->op != AGX_APPLE9_VIR_IADD &&
       instruction->op != AGX_APPLE9_VIR_ISUB &&
       instruction->op != AGX_APPLE9_VIR_IMAD &&
       instruction->op != AGX_APPLE9_VIR_IAND &&
       instruction->op != AGX_APPLE9_VIR_IOR &&
       instruction->op != AGX_APPLE9_VIR_IXOR &&
       instruction->op != AGX_APPLE9_VIR_IMIN &&
       instruction->op != AGX_APPLE9_VIR_IMAX &&
       instruction->op != AGX_APPLE9_VIR_UMIN &&
       instruction->op != AGX_APPLE9_VIR_UMAX &&
       instruction->op != AGX_APPLE9_VIR_FMIN &&
       instruction->op != AGX_APPLE9_VIR_FMAX) {
      if (reason)
         *reason = "Apple9 operation does not admit inline ALU operands";
      return false;
   }

   if (instruction->saturate && !agx_apple9_supports_saturate(instruction)) {
      if (reason)
         *reason = "Apple9 operation has no supported saturation modifier";
      return false;
   }

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
