/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "agx_apple9_ir.h"
#include "agx_builder.h"
#include "agx_compiler.h"

/* Apple9 selection keeps its machine operands and encoding contracts. The
 * shared IR carries their SSA dataflow, including genuine CFG phis, through
 * the existing spiller, memory-slot allocator and parallel-copy lowering. */
struct apple9_shared_ra {
   struct agx_apple9_vir_program *program;
   agx_context *ctx;
   struct agx_shader_key key;
   agx_block **blocks;
   agx_index *values;
   agx_index *vectors;
   agx_instr **phis;
   bool *publication;
};

static enum agx_schedule_class
apple9_schedule_class(const agx_instr *I)
{
   if (I->op == AGX_OPCODE_APPLE9 || I->op == AGX_OPCODE_APPLE9_PURE) {
      /* GPR SSA captures arithmetic dependencies. Publications, memory and
       * coverage still have implicit resources; serialize those operations
       * until their complete dependency graph is exposed to shared IR. */
      return agx_apple9_instr_is_pure_alu(I->apple9)
                ? AGX_SCHEDULE_CLASS_NONE
                : AGX_SCHEDULE_CLASS_BARRIER;
   }
   return agx_opcodes_info[I->op].schedule_class;
}

enum apple9_schedule_resource {
   APPLE9_SCHED_STATE = BITFIELD_BIT(0),
   APPLE9_SCHED_MEMORY = BITFIELD_BIT(1),
   APPLE9_SCHED_PUBLICATION = BITFIELD_BIT(2),
   APPLE9_SCHED_INTERPOLATION = BITFIELD_BIT(3),
   APPLE9_SCHED_TILE = BITFIELD_BIT(4),
   APPLE9_SCHED_ALL = BITFIELD_MASK(5),
};

static void
apple9_schedule_resources(const agx_instr *I, uint32_t *reads, uint32_t *writes)
{
   /* Every lane operation observes the active execution/coverage state. */
   *reads = APPLE9_SCHED_STATE;
   *writes = 0;
   switch (I->op) {
   case AGX_OPCODE_MOV:
   case AGX_OPCODE_MOV_IMM:
   case AGX_OPCODE_COLLECT:
   case AGX_OPCODE_SPLIT:
   case AGX_OPCODE_PHI:
      return;
   case AGX_OPCODE_APPLE9:
   case AGX_OPCODE_APPLE9_PURE:
      break;
   default:
      *reads = *writes = APPLE9_SCHED_ALL;
      return;
   }
   const struct agx_apple9_vir_instr *ins = I->apple9;
   if (agx_apple9_instr_is_pure_alu(ins))
      return;
   if (ins->publication_handoff ||
       ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT ||
       ins->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE)
      *reads |= *writes = APPLE9_SCHED_PUBLICATION;
   switch (ins->op) {
   case AGX_APPLE9_VIR_GET_GLOBAL_ID:
   case AGX_APPLE9_VIR_GET_SR:
   case AGX_APPLE9_VIR_DERIVATIVE:
      return;
   case AGX_APPLE9_VIR_DEVICE_LOAD:
      *reads |= APPLE9_SCHED_MEMORY;
      return;
   case AGX_APPLE9_VIR_DEVICE_STORE:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT:
   case AGX_APPLE9_VIR_BLOCK_IMAGE_STORE:
      *reads |= APPLE9_SCHED_MEMORY;
      *writes |= APPLE9_SCHED_MEMORY;
      return;
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
      *reads |= APPLE9_SCHED_MEMORY | APPLE9_SCHED_PUBLICATION;
      *writes |= APPLE9_SCHED_PUBLICATION;
      return;
   case AGX_APPLE9_VIR_CENTROID_POSITION:
   case AGX_APPLE9_VIR_ITER:
   case AGX_APPLE9_VIR_ITER_FLAT:
   case AGX_APPLE9_VIR_FMUL_PROJECT:
   case AGX_APPLE9_VIR_CUBE:
      *reads |= APPLE9_SCHED_INTERPOLATION;
      *writes |= APPLE9_SCHED_INTERPOLATION;
      return;
   case AGX_APPLE9_VIR_PUBLICATION_TUPLE:
      return;
   default:
      /* ALU exports/handoffs have no extra implicit state. Everything else,
       * including predicate, tile, coverage and control transitions, remains
       * a full resource fence until a narrower contract has been established. */
      if (*writes == APPLE9_SCHED_PUBLICATION)
         return;
      *reads = *writes = APPLE9_SCHED_ALL;
      return;
   }
}

static unsigned
apple9_dependency_delay(const struct agx_apple9_vir_instr *I)
{
   /* Scheduling priorities, not measured T8132 cycle counts. Memory receives
    * the largest separation; transcendental pipelines come next. Candidate
    * selection also accounts for the completed allocation's copies/spills. */
   switch (I->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_SPILL_LOAD:
      return 16;
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
      return 4;
   default:
      return 1;
   }
}

static unsigned
apple9_schedule_delay(const agx_instr *I)
{
   return (I->op == AGX_OPCODE_APPLE9 || I->op == AGX_OPCODE_APPLE9_PURE)
             ? apple9_dependency_delay(I->apple9) : 1;
}

static bool
apple9_identity(const struct agx_apple9_vir_instr *ins)
{
   return ins->op == AGX_APPLE9_VIR_IOR &&
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED &&
          ins->nr_srcs == 2 && ins->src[0] == ins->src[1];
}

/* A hardware tuple is one vector operand in allocation IR. SPLIT/COLLECT
 * express its relationship to scalar SSA values without reserving a bank of
 * post-allocation repair registers. */
static unsigned
apple9_source_width(const struct agx_apple9_vir_instr *ins, unsigned source)
{
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
        ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) && source == 0)
      return MAX2(ins->memory_components, 1);
   if (ins->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT &&
       source == ins->memory_components + 1)
      return 2;
   if (ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT && source == 1)
      return 2;
   return 1;
}

static const struct agx_apple9_operand_constraint *
apple9_source_constraint(const struct agx_apple9_vir_instr *ins, unsigned source)
{
   if (ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
       ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
      if (ins->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT &&
          source > ins->memory_components)
         return agx_apple9_find_operand(ins->encoding,
                                        source == ins->memory_components + 1
                                           ? AGX_APPLE9_OPERAND_SRC1
                                           : AGX_APPLE9_OPERAND_SRC2);
      enum agx_apple9_operand_role role =
         source ? AGX_APPLE9_OPERAND_INDEX
         : ins->op == AGX_APPLE9_VIR_DEVICE_STORE ? AGX_APPLE9_OPERAND_STORE_DATA
                                                : AGX_APPLE9_OPERAND_ATOMIC_DATA;
      return agx_apple9_find_operand(ins->encoding, role);
   }
   const struct agx_apple9_encoding_info *info =
      agx_apple9_encoding_info(ins->encoding);
   for (unsigned o = 0, s = 0; o < info->operand_count; ++o) {
      const struct agx_apple9_operand_constraint *c = &info->operands[o];
      if (!(c->files & AGX_APPLE9_FILE_GPR) ||
          agx_apple9_inline_source_file(ins, c->role) ||
          c->role == AGX_APPLE9_OPERAND_DEST)
         continue;
      if (s++ == source)
         return c;
   }
   return NULL;
}

static struct agx_reg_constraint
apple9_register_constraint(const struct agx_apple9_operand_constraint *c)
{
   return (struct agx_reg_constraint){
      .min = c ? 2 * c->min_index : 0,
      .max = c ? 2 * c->max_index : AGX_APPLE9_HALF_REGISTER_COUNT - 2,
      .align = c ? MAX2(c->alignment_halves, 2) : 2,
      .clobber = c && (c->flags & AGX_APPLE9_OPERAND_CLOBBER),
   };
}

static bool
apple9_publication(const struct agx_apple9_vir_instr *ins)
{
   return ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT ||
          ins->op == AGX_APPLE9_VIR_PUBLICATION_TUPLE;
}

static bool
apple9_after_logical_end(const struct agx_apple9_vir_instr *ins)
{
   switch (ins->op) {
   case AGX_APPLE9_VIR_EXEC_MASK_PUSH:
   case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
   case AGX_APPLE9_VIR_EXEC_MASK_POP:
   case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
   case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
   case AGX_APPLE9_VIR_LOOP_MASK_POP:
   case AGX_APPLE9_VIR_JMP_EXEC_ANY:
   case AGX_APPLE9_VIR_JMP_EXEC_NONE:
   case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
      return true;
   default:
      return false;
   }
}

static struct agx_apple9_block *
apple9_nir_block(struct apple9_shared_ra *ra, nir_block *nir)
{
   if (!nir)
      return NULL;
   for (struct agx_apple9_block *block = ra->program->blocks; block;
        block = block->next) {
      if (block->nir == nir)
         return block;
   }
   return NULL;
}

static agx_block *
apple9_shared_block(struct apple9_shared_ra *ra, struct agx_apple9_block *block)
{
   return block ? ra->blocks[block->index] : NULL;
}

static bool
apple9_shared_dataflow(struct apple9_shared_ra *ra, nir_shader *nir,
                       const char **reason)
{
   struct agx_apple9_vir_program *p = ra->program;
   agx_context *ctx = ra->ctx;
   ctx->nir = nir;
   ctx->stage = nir->info.stage;
   ctx->key = &ra->key;
   ctx->schedule_class = apple9_schedule_class;
   ra->key.has_scratch = true;
   ctx->ra_target = (struct agx_ra_target){
      .max_registers = AGX_APPLE9_HALF_REGISTER_COUNT,
      .reserved_registers = 4,
      .spill_reserved_registers = 8,
      .spill_copy_register = 2,
      .spill_swap_registers = {4, 6},
      .late_kill_sources = true,
      .preserve_source_kills = true,
      .defer_spill_lowering = true,
      .disable_occupancy_rematerialization = true,
   };
   list_inithead(&ctx->blocks);
   unsigned block_count = p->last_block->index + 1;
   ra->blocks = rzalloc_array(ctx, agx_block *, block_count);
   ra->values = rzalloc_array(ctx, agx_index, p->value_count);
   ra->vectors = rzalloc_array(ctx, agx_index, p->value_count);
   ra->phis = rzalloc_array(ctx, agx_instr *, p->value_count);
   ra->publication = rzalloc_array(ctx, bool, p->value_count);
   if (!ra->blocks || !ra->values || !ra->vectors || !ra->phis || !ra->publication)
      return false;

   for (unsigned i = 0; i < p->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = p->instructions[i];
      if (apple9_publication(ins)) {
         for (unsigned c = 0; c < ins->dest_components; ++c)
            ra->publication[ins->dest + c] = true;
      }
   }
   for (unsigned v = 0; v < p->value_count; ++v) {
      if (!ra->publication[v])
         ra->values[v] = agx_temp(ctx, AGX_SIZE_32);
   }
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = p->instructions[i];
      if (apple9_identity(ins) && !ins->publication_handoff)
         ra->values[ins->dest] = ra->values[ins->src[0]];
   }

   for (struct agx_apple9_block *block = p->blocks; block;
        block = block->next) {
      /* Loop-mask restoration blocks have no SSA dataflow. Their original
       * machine placement is retained when emitting the allocated program. */
      if (!block->nir && block != p->blocks)
         continue;
      agx_block *common = rzalloc(ctx, agx_block);
      if (!common)
         return false;
      list_inithead(&common->instructions);
      util_dynarray_init(&common->predecessors, common);
      common->index = ctx->num_blocks++;
      list_addtail(&common->link, &ctx->blocks);
      ra->blocks[block->index] = common;
   }

   for (struct agx_apple9_block *block = p->blocks; block;
        block = block->next) {
      agx_block *common = apple9_shared_block(ra, block);
      if (!common)
         continue;
      if (block->nir) {
         for (unsigned s = 0; s < 2; ++s) {
            nir_block *succ = block->nir->successors[s];
            agx_block *next =
               apple9_shared_block(ra, apple9_nir_block(ra, succ));
            if (!next)
               continue;
            agx_block_add_successor(common, next);
            next->loop_header |= next->index <= common->index;
         }
      } else {
         struct agx_apple9_block *next = block->next;
         while (next && !next->nir)
            next = next->next;
         if (next)
            agx_block_add_successor(common, apple9_shared_block(ra, next));
      }
   }

   /* Declare phis before selecting instructions, since incoming backedges
    * can refer to values whose definitions occur later in source order. */
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = p->instructions[i];
      if (ins->op != AGX_APPLE9_VIR_PHI)
         continue;
      agx_block *block = apple9_shared_block(ra, ins->phi_block);
      if (!block || !agx_num_predecessors(block)) {
         *reason = "Apple9 shared allocation requires a logical phi CFG";
         return false;
      }
      agx_builder b = agx_init_builder(ctx, agx_after_block(block));
      ra->phis[ins->dest] =
         agx_phi_to(&b, ra->values[ins->dest], agx_num_predecessors(block));
   }

   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = p->instructions[i];
      struct agx_apple9_block *owner = agx_apple9_instr_block(ins);
      agx_block *block = apple9_shared_block(ra, owner);
      if (ins->op == AGX_APPLE9_VIR_PHI)
         continue;
      if (!block) {
         if (ins->dest_components || ins->nr_srcs) {
            *reason = "Apple9 synthetic control block has SSA dataflow";
            return false;
         }
         continue;
      }
      if (ins->op == AGX_APPLE9_VIR_PHI_SRC) {
         agx_instr *phi = ra->phis[ins->target];
         struct agx_apple9_vir_instr const *definition =
            agx_apple9_definition(p, ins->target);
         agx_block *succ = apple9_shared_block(ra, definition->phi_block);
         if (!phi || !succ) {
            *reason = "Apple9 phi edge has no shared destination";
            return false;
         }
         phi->src[agx_predecessor_index(succ, block)] = ra->values[ins->src[0]];
         continue;
      }
      if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT ||
          (apple9_identity(ins) && !ins->publication_handoff))
         continue;
      agx_builder b = agx_init_builder(ctx, agx_after_block(block));
      unsigned components = ins->dest_components;
      bool pub = components && ra->publication[ins->dest];
      agx_index dest = components > 1
                          ? agx_vec_temp(ctx, AGX_SIZE_32, components)
                       : components ? ra->values[ins->dest]
                                    : agx_null();
      if (components > 1 && !pub)
         ra->vectors[ins->dest] = dest;
      if (ins->op == AGX_APPLE9_VIR_IMM) {
         agx_mov_imm_to(&b, dest, ins->immediate);
         continue;
      }
      agx_index sources[AGX_APPLE9_MAX_VIR_SRCS];
      unsigned first_source[AGX_APPLE9_MAX_VIR_SRCS], nr_srcs = 0;
      for (unsigned s = 0; s < ins->nr_srcs;) {
         unsigned width = apple9_source_width(ins, s);
         first_source[nr_srcs] = s;
         if (width == 1) {
            sources[nr_srcs] = ra->values[ins->src[s]];
         } else {
            agx_index vector = ra->vectors[ins->src[s]];
            bool tuple = agx_channels(vector) == width;
            for (unsigned c = 1; c < width; ++c)
               tuple &= ins->src[s + c] == ins->src[s] + c;
            if (!tuple) {
               vector = agx_vec_temp(ctx, AGX_SIZE_32, width);
               agx_instr *collect = agx_collect_to(&b, vector, width);
               for (unsigned c = 0; c < width; ++c)
                  collect->src[c] = ra->values[ins->src[s + c]];
            }
            sources[nr_srcs] = vector;
         }
         ++nr_srcs;
         s += width;
      }
      bool pure = components && !pub && !ins->publication_handoff &&
                  ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC;
      agx_instr *common =
         agx_alloc_instr(&b,
                         ins->op == AGX_APPLE9_VIR_COLLECT ? AGX_OPCODE_COLLECT
                         : apple9_after_logical_end(ins) ? AGX_OPCODE_APPLE9_CF
                         : pure ? AGX_OPCODE_APPLE9_PURE
                                                         : AGX_OPCODE_APPLE9,
                         components && !pub ? 1 : 0, nr_srcs);
      common->apple9 = ins;
      if (common->nr_dests)
         common->dest[0] = dest;
      for (unsigned s = 0; s < nr_srcs; ++s)
         common->src[s] = sources[s];
      if (common->op != AGX_OPCODE_COLLECT) {
         common->reg_constraints = rzalloc_array(
            ctx, struct agx_reg_constraint, common->nr_dests + nr_srcs);
         if (common->nr_dests)
            common->reg_constraints[0] = apple9_register_constraint(
               ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC
                  ? agx_apple9_find_operand(AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT,
                                            AGX_APPLE9_OPERAND_SRC0)
                  : agx_apple9_find_operand(ins->encoding,
                                            AGX_APPLE9_OPERAND_DEST));
         for (unsigned s = 0; s < nr_srcs; ++s) {
            unsigned source = first_source[s];
            struct agx_reg_constraint c =
               apple9_register_constraint(apple9_source_constraint(ins, source));
            unsigned fixed = p->fixed_phys[ins->src[source]];
            if (fixed != AGX_APPLE9_PHYS_INVALID && !ra->publication[ins->src[source]])
               c.min = c.max = fixed * 2;
            common->reg_constraints[common->nr_dests + s] = c;
         }
      }
      agx_builder_insert(&b.cursor, common);
      if (components > 1 && !pub) {
         agx_instr *split = agx_split(&b, components, dest);
         for (unsigned c = 0; c < components; ++c)
            split->dest[c] = ra->values[ins->dest + c];
      }
   }
   agx_validate(ctx, "Apple9 shared allocation input");
   return true;
}

/* Canonical physical operands occupy separate GPR and publication ranges. */
#define PUB_BASE  AGX_APPLE9_GPR_COUNT
#define COPY_REG  1
#define WAIT_REG  1
#define SPILL_TAG 6

static bool
apple9_emit_physical(struct agx_apple9_vir_program *out,
                     const struct agx_apple9_vir_instr *ins)
{
   if (!agx_apple9_vir_emit_side_effect(out, ins->op, ins->encoding, NULL, 0,
                                        0))
      return false;
   *out->instructions[out->instruction_count - 1] = *ins;
   for (unsigned c = 0; c < ins->dest_components; ++c) {
      if (ins->dest < PUB_BASE)
         out->max_phys_gpr = MAX2(out->max_phys_gpr, ins->dest + c);
   }
   for (unsigned s = 0; s < ins->nr_srcs; ++s) {
      if (ins->src[s] < PUB_BASE)
         out->max_phys_gpr = MAX2(out->max_phys_gpr, ins->src[s]);
   }
   return true;
}

static struct agx_apple9_vir_instr
apple9_physical_instruction(enum agx_apple9_vir_opcode op,
                            enum agx_apple9_encoding encoding, unsigned dest)
{
   return (struct agx_apple9_vir_instr){
      .op = op,
      .encoding = encoding,
      .dest = dest,
      .dest_components = dest != AGX_APPLE9_VREG_INVALID,
      .producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE,
      .scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE,
   };
}

static bool
apple9_physical_copy(struct agx_apple9_vir_program *out, unsigned dst,
                     unsigned src, unsigned wait)
{
   if (dst == src && wait == AGX_APPLE9_SCOREBOARD_SLOT_NONE)
      return true;
   struct agx_apple9_vir_instr ins = apple9_physical_instruction(
      AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, dst);
   ins.src[0] = ins.src[1] = src;
   ins.nr_srcs = 2;
   ins.live_after_mask = 3;
   ins.scoreboard_slot = wait;
   return apple9_emit_physical(out, &ins);
}

static bool
apple9_physical_immediate(struct agx_apple9_vir_program *out, unsigned dst,
                          uint32_t value)
{
   /* Rematerialization and PHI copies happen after constrained allocation.
    * The immediate form only has six destination bits; use the reserved
    * parallel-copy temporary when the final destination is in the high bank.
    */
   if (dst >= 64)
      return apple9_physical_immediate(out, COPY_REG, value) &&
             apple9_physical_copy(out, dst, COPY_REG,
                                  AGX_APPLE9_SCOREBOARD_SLOT_NONE);

   struct agx_apple9_vir_instr ins = apple9_physical_instruction(
      AGX_APPLE9_VIR_IMM,
      dst < 16 && value < 0x80 ? AGX_APPLE9_ENC_MOV_IMM_COMPACT
                              : AGX_APPLE9_ENC_MOV_IMM32, dst);
   ins.immediate = value;
   return apple9_emit_physical(out, &ins);
}

static bool
apple9_physical_move(struct agx_apple9_vir_program *out, agx_index dst,
                     agx_index src)
{
   assert(dst.type == AGX_INDEX_REGISTER && dst.size == AGX_SIZE_32);
   if (src.type == AGX_INDEX_IMMEDIATE)
      return apple9_physical_immediate(out, dst.value / 2, src.value);
   assert(src.type == AGX_INDEX_REGISTER && src.size == AGX_SIZE_32);
   if (!dst.memory && !src.memory)
      return apple9_physical_copy(out, dst.value / 2, src.value / 2,
                                  AGX_APPLE9_SCOREBOARD_SLOT_NONE);
   assert(!dst.memory || !src.memory);
   if (src.memory) {
      struct agx_apple9_vir_instr load = apple9_physical_instruction(
         AGX_APPLE9_VIR_SPILL_LOAD, AGX_APPLE9_ENC_SPILL_LOAD, dst.value / 2);
      load.immediate = src.value / 2;
      load.producer_scoreboard_slot = SPILL_TAG;
      return apple9_emit_physical(out, &load);
   }
   /* SAVE releases its input. Use the target's parallel-copy temporary so
    * a spilled SSA definition can remain live in its allocated GPR. */
   struct agx_apple9_vir_instr save = apple9_physical_instruction(
      AGX_APPLE9_VIR_SPILL_STORE, AGX_APPLE9_ENC_SPILL_STORE,
      AGX_APPLE9_VREG_INVALID);
   save.src[0] = COPY_REG;
   save.nr_srcs = 1;
   save.immediate = dst.value / 2;
   save.producer_scoreboard_slot = SPILL_TAG;
   return apple9_physical_copy(out, COPY_REG, src.value / 2,
                               AGX_APPLE9_SCOREBOARD_SLOT_NONE) &&
          apple9_emit_physical(out, &save);
}

static bool
apple9_physical_selected(struct apple9_shared_ra *ra,
                         struct agx_apple9_vir_program *out, agx_instr *common,
                         struct agx_apple9_block **blocks, const char **reason)
{
   const struct agx_apple9_vir_instr *original = common->apple9;
   struct agx_apple9_vir_instr ins = *original;
   bool pub = apple9_publication(original);
   ins.dest = ins.dest_components
                 ? pub ? PUB_BASE + ra->program->phys[original->dest]
                       : common->dest[0].value / 2
                 : AGX_APPLE9_VREG_INVALID;
   bool dead_dest = ins.dest_components && !pub && agx_is_null(common->dest[0]);
   if (dead_dest && ins.publication_handoff) {
      /* Unused texture channels still need the read that releases their
       * borrowed publication. Consume them into the reserved copy temporary. */
      ins.dest = COPY_REG;
   } else if (dead_dest) {
      assert(ins.op == AGX_APPLE9_VIR_DEVICE_ATOMIC);
      ins.atomic_discard = true;
      ins.dest_components = 0;
      ins.dest = AGX_APPLE9_VREG_INVALID;
   }
   ins.live_after_mask = 0;
   for (unsigned s = 0, source = 0; s < common->nr_srcs; ++s) {
      unsigned width = apple9_source_width(original, source);
      for (unsigned c = 0; c < width; ++c, ++source) {
         ins.src[source] = ra->publication[original->src[source]]
                              ? PUB_BASE + ra->program->phys[original->src[source]]
                              : common->src[s].value / 2 + c;
         if (!common->src[s].kill)
            ins.live_after_mask |= BITFIELD_BIT(source);
      }
   }
   if (ins.branch_target)
      ins.branch_target = blocks[ins.branch_target->index];
   ins.phi_block = NULL;
   ins.phi_edge = ins.completion_consumer = NULL;
   ins.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
   ins.device_load_index_kind = (ins.live_after_mask & 1)
                                   ? AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR
                                   : AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR;

   if (!apple9_emit_physical(out, &ins))
      return false;
   if (ins.op == AGX_APPLE9_VIR_DEVICE_ATOMIC && !ins.atomic_discard) {
      struct agx_apple9_vir_instr result = apple9_physical_instruction(
         AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT,
         AGX_APPLE9_ENC_DEVICE_ATOMIC_RESULT, AGX_APPLE9_VREG_INVALID);
      result.src[0] = ins.dest;
      result.nr_srcs = 1;
      result.live_after_mask = 1;
      result.producer_scoreboard_slot = ins.producer_scoreboard_slot;
      if (!apple9_emit_physical(out, &result))
         return false;
   }
   return true;
}

static bool
apple9_init_physical(struct agx_apple9_vir_program *out)
{
   agx_apple9_vir_init(out);
   for (unsigned r = 0; r < PUB_BASE + AGX_APPLE9_PUBLICATION_COUNT; ++r) {
      /* Reserve every canonical ID without asking the ordinary-input API to
       * validate a publication index as a GPR. The maps below supply its
       * namespace and physical index. */
      if (agx_apple9_vir_input(out, r < PUB_BASE ? r : 0) != r)
         return false;
      if (r >= PUB_BASE)
         out->fixed_phys[r] = AGX_APPLE9_PHYS_INVALID;
   }
   out->phys = malloc(out->value_count);
   out->publication = calloc(out->value_count, sizeof(bool));
   if (!out->phys || !out->publication)
      return false;
   for (unsigned r = 0; r < out->value_count; ++r) {
      out->phys[r] = r < PUB_BASE ? r : r - PUB_BASE;
      out->publication[r] = r >= PUB_BASE;
   }
   return true;
}

struct apple9_pending {
   bool active;
   struct agx_apple9_vir_instr producer;
};

/* Scalar export producers complete asynchronously in the publication namespace.
 * Their completion tag is independent of the publication's storage index. */
static bool
apple9_scalar_publication(const struct agx_apple9_vir_instr *ins)
{
   return ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT;
}

static bool
apple9_async_sr(const struct agx_apple9_vir_instr *ins)
{
   return ins->encoding == AGX_APPLE9_ENC_GET_DRAW_ID ||
          ins->encoding == AGX_APPLE9_ENC_GET_COVERAGE;
}

static bool
apple9_async(const struct agx_apple9_vir_instr *ins)
{
   if (apple9_scalar_publication(ins) || apple9_async_sr(ins))
      return true;
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
   case AGX_APPLE9_VIR_TILE_LOAD:
   case AGX_APPLE9_VIR_ITER_FLAT:
   case AGX_APPLE9_VIR_SPILL_LOAD:
   case AGX_APPLE9_VIR_SPILL_STORE:
      return true;
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
      return !ins->atomic_discard;
   default:
      return false;
   }
}

static bool
apple9_pending_result(const struct apple9_pending *p, unsigned reg)
{
   return p->active && reg >= p->producer.dest &&
          reg - p->producer.dest < p->producer.dest_components;
}

static bool
apple9_retire(struct agx_apple9_vir_program *out,
              struct apple9_pending *pending, unsigned slot)
{
   if (!pending[slot].active)
      return true;
   const struct agx_apple9_vir_instr *producer = &pending[slot].producer;
   unsigned reg = producer->dest;
   if (producer->op == AGX_APPLE9_VIR_SPILL_STORE ||
       apple9_scalar_publication(producer)) {
      /* Publications have no GPR result to materialize. Wait using the
       * reserved scratch operand while retaining the publication itself. */
      reg = WAIT_REG;
      if (!apple9_physical_immediate(out, reg, 0))
         return false;
   }
   if (!apple9_physical_copy(out, reg, reg, slot))
      return false;
   pending[slot].active = false;
   return true;
}

static bool
apple9_float_consumer(enum agx_apple9_encoding encoding)
{
   switch (encoding) {
   case AGX_APPLE9_ENC_FLOAT2_COMPACT:
   case AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED:
   case AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED:
   case AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED:
   case AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT:
   case AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED:
   case AGX_APPLE9_ENC_FLOAT3_EXTENDED:
   case AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED:
   case AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED:
      return true;
   default:
      return false;
   }
}

static bool
apple9_can_fold_wait(const struct agx_apple9_vir_instr *ins,
                     const struct apple9_pending *p)
{
   if (agx_apple9_encoding_info(ins->encoding)->dependency_layout ==
       AGX_APPLE9_DEPENDENCY_NONE)
      return false;
   if (p->producer.op == AGX_APPLE9_VIR_SPILL_STORE)
      return true;
   if (apple9_scalar_publication(&p->producer))
      return ins->op == AGX_APPLE9_VIR_VARY_STORE &&
             ins->src[0] == p->producer.dest;
   if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
       (ins->atomic_discard || ins->atomic_op == AGX_APPLE9_ATOMIC_CMPXCHG ||
        p->producer.op != AGX_APPLE9_VIR_DEVICE_LOAD ||
        p->producer.dest_components != 1 || ins->src[0] != p->producer.dest))
      return false;
   /* An identity copy explicitly requests materialization, independently of
    * which tuple components or later uses need the completed value. */
   if (apple9_identity(ins) && ins->src[0] == p->producer.dest)
      return true;
   /* A float consumer can perform the tuple handoff directly. This includes
    * FMA and inline operands; liveness still names only actual GPR inputs. */
   bool texture = p->producer.op == AGX_APPLE9_VIR_TEXTURE_SAMPLE;
   if (texture && !apple9_float_consumer(ins->encoding))
      return false;
   /* Completion makes the producer tuple available. Keeping operands live
    * is independent of releasing its slot. These ALU and address forms encode
    * source retention explicitly (EXP-M4-34/35/63). */
   bool tuple = p->producer.op == AGX_APPLE9_VIR_DEVICE_LOAD ||
                apple9_async_sr(&p->producer) || texture;
   bool can_retain = tuple &&
      (apple9_float_consumer(ins->encoding) ||
       ins->encoding == AGX_APPLE9_ENC_FLOAT_SPECIAL ||
       ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       ins->encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED ||
       ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT ||
       ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD ||
       ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT);
   unsigned consumed = 0;
   for (unsigned s = 0; s < ins->nr_srcs; ++s) {
      if (!apple9_pending_result(p, ins->src[s]))
         continue;
      if ((ins->live_after_mask & BITFIELD_BIT(s)) && !can_retain)
         return false;
      consumed |= BITFIELD_BIT(ins->src[s] - p->producer.dest);
   }
   return consumed == BITFIELD_MASK(p->producer.dest_components) ||
          (tuple && consumed != 0);
}

static bool
apple9_completion_barrier(const struct agx_apple9_vir_instr *ins)
{
   return apple9_after_logical_end(ins) ||
          ins->op == AGX_APPLE9_VIR_TILE_ACCESS ||
          ins->op == AGX_APPLE9_VIR_TILE_FENCE ||
          ins->op == AGX_APPLE9_VIR_COVERAGE ||
          ins->op == AGX_APPLE9_VIR_DEPTH_STORE;
}

static bool
apple9_pending_hazard(const struct agx_apple9_vir_instr *ins,
                      const struct apple9_pending *p)
{
   if (!p->active)
      return false;
   if (apple9_completion_barrier(ins))
      return true;
   const struct agx_apple9_vir_instr *producer = &p->producer;
   for (unsigned c = 0; c < ins->dest_components; ++c) {
      unsigned reg = ins->dest + c;
      if (apple9_pending_result(p, reg))
         return true;
      /* Includes texture parameter publications, which remain borrowed
       * until the result handoff, and scratch SAVE's destructive input. */
      for (unsigned s = 0; s < producer->nr_srcs; ++s) {
         if (producer->src[s] == reg)
            return true;
      }
   }
   bool scratch = ins->op == AGX_APPLE9_VIR_SPILL_LOAD ||
                  ins->op == AGX_APPLE9_VIR_SPILL_STORE;
   bool previous_scratch = producer->op == AGX_APPLE9_VIR_SPILL_LOAD ||
                           producer->op == AGX_APPLE9_VIR_SPILL_STORE;
   if (scratch && previous_scratch && ins->immediate == producer->immediate &&
       (ins->op == AGX_APPLE9_VIR_SPILL_STORE ||
        producer->op == AGX_APPLE9_VIR_SPILL_STORE))
      return true;
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
        ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) &&
       (producer->op == AGX_APPLE9_VIR_DEVICE_LOAD ||
        producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) &&
       ins->immediate == producer->immediate)
      return true;
   /* This compound encoding contains a fixed slot-1 publication and wait. */
   return ins->encoding == AGX_APPLE9_ENC_BLOCK_IMAGE_STORE &&
          producer->producer_scoreboard_slot == AGX_APPLE9_SCOREBOARD_SLOT_1;
}

/* Schedule the final physical stream: allocation can introduce earlier uses,
 * register reuse and scratch traffic that do not exist in the original SSA.
 * Never carry pending work across a logical block or execution-mask change. */
static void
apple9_select_compact_alu(struct agx_apple9_vir_instr *I)
{
   if (I->scoreboard_slot || I->saturate || I->src_abs_mask || I->src_neg_mask ||
       I->dest >= 64)
      return;
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      if (I->src[s] >= 64)
         return;
   }
   unsigned inline_mask = I->alu_src_uniform_mask | I->alu_src_immediate_mask;
   if (I->encoding == AGX_APPLE9_ENC_FLOAT2_COMPACT && !inline_mask &&
       (I->op == AGX_APPLE9_VIR_FADD || I->op == AGX_APPLE9_VIR_FMUL)) {
      I->encoding = AGX_APPLE9_ENC_FLOAT2_BASE;
   } else if (I->encoding == AGX_APPLE9_ENC_FLOAT3_EXTENDED &&
              !(inline_mask & ~2) &&
              !((I->alu_src_immediate_mask & 2) && (I->alu_src_value[1] >> 31))) {
      I->encoding = AGX_APPLE9_ENC_FLOAT3_COMPACT;
   }
}

static bool
apple9_schedule_physical(struct agx_apple9_vir_program *program,
                          const char **reason)
{
   struct agx_apple9_vir_program out;
   if (!apple9_init_physical(&out)) {
      agx_apple9_vir_finish(&out);
      return false;
   }
   unsigned count = program->last_block->index + 1;
   struct agx_apple9_block **blocks = calloc(count, sizeof(*blocks));
   bool success = false;
   if (!blocks)
      goto cleanup;
   for (unsigned b = 0; b < count; ++b) {
      blocks[b] = agx_apple9_block_create(&out);
      if (!blocks[b])
         goto cleanup;
   }
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      agx_apple9_block_begin(&out, blocks[block->index]);
      struct apple9_pending pending[7] = {0};
      unsigned end = block->next ? block->next->start_index : program->instruction_count;
      for (unsigned i = block->start_index; i < end; ++i) {
         struct agx_apple9_vir_instr ins = *program->instructions[i];
         if (ins.branch_target)
            ins.branch_target = blocks[ins.branch_target->index];
         if (ins.op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT) {
            assert(out.instruction_count &&
                   out.instructions[out.instruction_count - 1]->op == AGX_APPLE9_VIR_DEVICE_ATOMIC);
            ins.producer_scoreboard_slot =
               out.instructions[out.instruction_count - 1]->producer_scoreboard_slot;
            if (!apple9_emit_physical(&out, &ins))
               goto cleanup;
            continue;
         }

         ins.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
         unsigned fold = 0;
         for (unsigned slot = 1; slot <= 6; ++slot) {
            if (!pending[slot].active)
               continue;
            bool needed = apple9_pending_hazard(&ins, &pending[slot]);
            for (unsigned s = 0; s < ins.nr_srcs; ++s)
               needed |= apple9_pending_result(&pending[slot], ins.src[s]);
            if (!needed)
               continue;
            if (!fold && apple9_can_fold_wait(&ins, &pending[slot])) {
               fold = slot;
            } else if (!apple9_retire(&out, pending, slot)) {
               goto cleanup;
            }
         }
         if (fold) {
            /* Logic's pending-result source is A. Its operators commute. */
            if ((ins.encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED ||
                 ins.encoding == AGX_APPLE9_ENC_LOGIC_EXPORT) && ins.nr_srcs == 2 &&
                !apple9_pending_result(&pending[fold], ins.src[0]) &&
                apple9_pending_result(&pending[fold], ins.src[1])) {
               unsigned src = ins.src[0];
               ins.src[0] = ins.src[1];
               ins.src[1] = src;
               ins.live_after_mask = ((ins.live_after_mask & 1) << 1) |
                                     ((ins.live_after_mask & 2) >> 1);
            }
            ins.scoreboard_slot = fold;
            pending[fold].active = false;
         }

         if (apple9_async(&ins)) {
            unsigned slot = 0;
            if (apple9_async_sr(&ins)) {
               slot = AGX_APPLE9_SCOREBOARD_SLOT_1;
            } else {
               /* Samples and memory traffic share six completion tags.
                * Keep independent operations pending until a real hazard,
                * consumer, or tag exhaustion requires their result handoff. */
               unsigned first = ins.op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ? 1 : 6;
               for (unsigned p = 0; p < 6; ++p) {
                  unsigned candidate = (first - 1 + p) % 6 + 1;
                  if (!pending[candidate].active) {
                     slot = candidate;
                     break;
                  }
               }
               if (!slot)
                  slot = first;
            }
            if (!apple9_retire(&out, pending, slot))
               goto cleanup;
            ins.producer_scoreboard_slot = slot;
            if (ins.op == AGX_APPLE9_VIR_DEVICE_LOAD) {
               const uint16_t tokens[] = {
                  AGX_APPLE9_DEVICE_LOAD_TOKEN_1100, AGX_APPLE9_DEVICE_LOAD_TOKEN_5100,
                  AGX_APPLE9_DEVICE_LOAD_TOKEN_9100, AGX_APPLE9_DEVICE_LOAD_TOKEN_D100,
                  AGX_APPLE9_DEVICE_LOAD_TOKEN_1101, AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
               };
               ins.device_load_raw_token = tokens[slot - 1];
            }
            pending[slot] = (struct apple9_pending){.active = true, .producer = ins};
         }
         apple9_select_compact_alu(&ins);
         if (!apple9_emit_physical(&out, &ins))
            goto cleanup;
      }
      for (unsigned slot = 1; slot <= 6; ++slot) {
         if (!apple9_retire(&out, pending, slot))
            goto cleanup;
      }
   }
   agx_apple9_vir_reindex(&out);
   agx_apple9_vir_finish(program);
   *program = out;
   for (struct agx_apple9_block *block = program->allocated_blocks; block;
        block = block->allocated_next)
      block->program = program;
   memset(&out, 0, sizeof(out));
   success = true;
cleanup:
   if (!success && reason)
      *reason = "Could not schedule Apple9 physical completions";
   free(blocks);
   agx_apple9_vir_finish(&out);
   return success;
}

static bool
apple9_allocate_candidate(struct agx_apple9_vir_program *program,
                          nir_shader *nir, bool cost_constraints, bool latency,
                          struct agx_apple9_vir_program *result,
                          const char **reason)
{
   struct apple9_shared_ra ra = {.program = program};
   ra.ctx = rzalloc(NULL, agx_context);
   struct agx_apple9_vir_program out;
   agx_apple9_vir_init(&out);
   bool success = false;
   if (!ra.ctx || !apple9_shared_dataflow(&ra, nir, reason))
      goto cleanup;
   ra.ctx->ra_target.cost_register_constraints = cost_constraints;
   if (latency) {
      ra.ctx->schedule_resources = apple9_schedule_resources;
      ra.ctx->schedule_delay = apple9_schedule_delay;
   }
   agx_dce(ra.ctx, true);
   agx_pressure_schedule(ra.ctx);
   agx_ra(ra.ctx);
   if (ra.ctx->scratch_size_B > 4096) {
      *reason =
         "Apple9 scratch frame exceeds the supported per-invocation size";
      goto cleanup;
   }
   if (!apple9_init_physical(&out))
      goto cleanup;
   unsigned count = program->last_block->index + 1;
   struct agx_apple9_block **blocks =
      rzalloc_array(ra.ctx, struct agx_apple9_block *, count);
   for (struct agx_apple9_block *block = program->blocks; block;
        block = block->next) {
      blocks[block->index] = agx_apple9_block_create(&out);
      if (!blocks[block->index])
         goto cleanup;
   }
   for (struct agx_apple9_block *block = program->blocks; block;
        block = block->next) {
      agx_apple9_block_begin(&out, blocks[block->index]);
      agx_block *common = apple9_shared_block(&ra, block);
      if (!common) {
         for (unsigned i = 0; i < program->instruction_count; ++i) {
            struct agx_apple9_vir_instr ins = *program->instructions[i];
            if (agx_apple9_instr_block(program->instructions[i]) != block)
               continue;
            if (ins.branch_target)
               ins.branch_target = blocks[ins.branch_target->index];
            if (!apple9_emit_physical(&out, &ins))
               goto cleanup;
         }
         continue;
      }
      agx_foreach_instr_in_block(common, I) {
         switch (I->op) {
         case AGX_OPCODE_MOV:
            if (!apple9_physical_move(&out, I->dest[0], I->src[0]))
               goto cleanup;
            break;
         case AGX_OPCODE_MOV_IMM:
            if (!apple9_physical_immediate(&out, I->dest[0].value / 2, I->imm))
               goto cleanup;
            break;
         case AGX_OPCODE_SWAP: {
            unsigned a = I->src[0].value / 2, b = I->src[1].value / 2;
            if (!apple9_physical_copy(&out, WAIT_REG, a,
                                      AGX_APPLE9_SCOREBOARD_SLOT_NONE) ||
                !apple9_physical_copy(&out, a, b,
                                      AGX_APPLE9_SCOREBOARD_SLOT_NONE) ||
                !apple9_physical_copy(&out, b, WAIT_REG,
                                      AGX_APPLE9_SCOREBOARD_SLOT_NONE))
               goto cleanup;
            break;
         }
         case AGX_OPCODE_APPLE9:
         case AGX_OPCODE_APPLE9_PURE:
         case AGX_OPCODE_APPLE9_CF:
            if (!apple9_physical_selected(&ra, &out, I, blocks, reason))
               goto cleanup;
            break;
         default:
            *reason =
               "Unexpected shared instruction after Apple9 register allocation";
            goto cleanup;
         }
      }
   }
   agx_apple9_vir_reindex(&out);
   if (!apple9_schedule_physical(&out, reason))
      goto cleanup;
   out.scratch_size = ALIGN_POT(ra.ctx->scratch_size_B, 16);
   out.spill_slots = out.scratch_size / 4;
   unsigned reserved = ra.ctx->has_spill_pcopy_reserved ? 4 : 2;
   for (unsigned r = 0; r < reserved; ++r)
      out.reserved_gprs[r] = true;
   out.peak_live_gprs = ra.ctx->max_reg / 2 + 1;
   out.publication_count = program->publication_count;
   out.fragment_shader = program->fragment_shader;
   out.dependencies_finalized = out.physical = true;
   *result = out;
   for (struct agx_apple9_block *block = result->allocated_blocks; block;
        block = block->allocated_next)
      block->program = result;
   memset(&out, 0, sizeof(out));
   success = true;
cleanup:
   agx_apple9_vir_finish(&out);
   ralloc_free(ra.ctx);
   return success;
}

struct apple9_allocation_cost {
   uint64_t spills, instructions, bytes, dependency_gaps;
   unsigned registers;
   bool has_copies;
};

static struct apple9_allocation_cost
apple9_allocation_cost(const struct agx_apple9_vir_program *p)
{
   struct apple9_allocation_cost cost = {.registers = p->max_phys_gpr + 1};
   unsigned nesting = 0;
   unsigned ready[AGX_APPLE9_GPR_COUNT] = {0}, issue = 0;
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *I = p->instructions[i];
      if (I->op == AGX_APPLE9_VIR_LOOP_MASK_POP && nesting)
         --nesting;
      /* Static loop weighting, not a cycle/occupancy estimate. It prevents
       * saving one setup move by adding a move on every loop iteration. */
      unsigned weight = 1u << (3 * MIN2(nesting, 3));
      unsigned earliest = issue;
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (I->src[s] < ARRAY_SIZE(ready))
            earliest = MAX2(earliest, ready[I->src[s]]);
      }
      cost.dependency_gaps += weight * (earliest - issue);
      for (unsigned c = 0; c < I->dest_components; ++c) {
         if (I->dest + c < ARRAY_SIZE(ready))
            ready[I->dest + c] = earliest + apple9_dependency_delay(I);
      }
      issue = earliest + 1;
      cost.spills += weight * (I->op == AGX_APPLE9_VIR_SPILL_STORE ||
                               I->op == AGX_APPLE9_VIR_SPILL_LOAD);
      cost.instructions += weight;
      cost.bytes += weight * agx_apple9_encoding_info(I->encoding)->length;
      cost.has_copies |= apple9_identity(I) && I->dest != I->src[0];
      if (I->op == AGX_APPLE9_VIR_LOOP_MASK_PUSH)
         ++nesting;
   }
   return cost;
}

static bool
apple9_allocation_cheaper(struct apple9_allocation_cost a,
                          struct apple9_allocation_cost b)
{
   if (a.spills != b.spills) return a.spills < b.spills;
   if (a.instructions != b.instructions) return a.instructions < b.instructions;
   if (a.bytes != b.bytes) return a.bytes < b.bytes;
   if (a.registers != b.registers) return a.registers < b.registers;
   return a.dependency_gaps < b.dependency_gaps;
}

bool
agx_apple9_allocate_shared(struct agx_apple9_vir_program *program,
                           nir_shader *nir, const char **reason)
{
   if (!agx_apple9_allocate_publications(program, reason))
      return false;
   struct agx_apple9_vir_program best, alternative;
   agx_apple9_vir_init(&best);
   agx_apple9_vir_init(&alternative);
   if (!apple9_allocate_candidate(program, nir, false, false, &best, reason))
      return false;

   struct apple9_allocation_cost cost = apple9_allocation_cost(&best);
   if (cost.has_copies || cost.spills) {
      /* Bank hints can conflict with later tuples and phi reconciliation.
       * Compare completed allocations so those costs are paid by the policy
       * which caused them, including rematerialization and spill lowering. */
      const char *unused_reason = NULL;
      if (apple9_allocate_candidate(program, nir, true, false, &alternative,
                                     &unused_reason) &&
          apple9_allocation_cheaper(apple9_allocation_cost(&alternative), cost))
         SWAP(best, alternative);
   }
   agx_apple9_vir_finish(&alternative);
   cost = apple9_allocation_cost(&best);
   unsigned long_dependencies = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i)
      long_dependencies += apple9_dependency_delay(program->instructions[i]) > 1;
   if (long_dependencies > 1) {
      for (unsigned banks = 0; banks < 2; ++banks) {
         agx_apple9_vir_init(&alternative);
         const char *unused_reason = NULL;
         if (apple9_allocate_candidate(program, nir, banks, true, &alternative,
                                        &unused_reason) &&
             apple9_allocation_cheaper(apple9_allocation_cost(&alternative), cost)) {
            SWAP(best, alternative);
            cost = apple9_allocation_cost(&best);
         }
         agx_apple9_vir_finish(&alternative);
      }
   }
   agx_apple9_vir_finish(program);
   *program = best;
   for (struct agx_apple9_block *block = program->allocated_blocks; block;
        block = block->allocated_next)
      block->program = program;
   return true;
}
