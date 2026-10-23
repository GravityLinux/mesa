#include "c11/threads.h"
#include "agx_compile_apple9.h"
/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "agx_apple9_ir.h"
#include "agx_apple9_profile.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "util/hash_table.h"
#include "util/mesa-blake3.h"
#include "util/u_debug.h"

DEBUG_GET_ONCE_OPTION(cached_agx_apple9_parallel_ra,
                      "AGX_APPLE9_PARALLEL_RA", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_trace_shared_ra,
                      "AGX_APPLE9_TRACE_SHARED_RA", NULL)

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
   APPLE9_SCHED_MESH = BITFIELD_BIT(5),
   APPLE9_SCHED_ALL = BITFIELD_MASK(6),
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
   case AGX_APPLE9_VIR_MESH_CONTROL:
   case AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR:
   case AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL:
      /* Mesh publication consumes local output and externally visible
       * formation records. Keep it ordered with both and with other mesh
       * operations; execution-mask state is already an input above. */
      *reads |= APPLE9_SCHED_MEMORY | APPLE9_SCHED_MESH;
      *writes |= APPLE9_SCHED_MEMORY | APPLE9_SCHED_MESH;
      return;
   case AGX_APPLE9_VIR_GET_GLOBAL_ID:
   case AGX_APPLE9_VIR_GET_SR:
   case AGX_APPLE9_VIR_DERIVATIVE:
      return;
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
      *reads |= APPLE9_SCHED_MEMORY;
      if (ins->memory_volatile)
         *writes |= APPLE9_SCHED_MEMORY;
      return;
   case AGX_APPLE9_VIR_DEVICE_STORE:
   case AGX_APPLE9_VIR_PRIVATE_STORE:
   case AGX_APPLE9_VIR_SHARED_STORE:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT:
      *reads |= APPLE9_SCHED_MEMORY;
      *writes |= APPLE9_SCHED_MEMORY;
      return;
   case AGX_APPLE9_VIR_IMAGE_STORE:
   case AGX_APPLE9_VIR_BLOCK_IMAGE_STORE:
      /* Coordinates and data live in the publication file, outside GPR
       * SSA. Keep the store after their publication and before a later
       * publication reuses the slots. Both store encodings join the write
       * before returning, so the publication is only borrowed at issue. */
      *reads |= APPLE9_SCHED_MEMORY | APPLE9_SCHED_PUBLICATION;
      *writes |= APPLE9_SCHED_MEMORY;
      return;
   case AGX_APPLE9_VIR_TEXTURE_SAMPLE:
      *reads |= APPLE9_SCHED_MEMORY | APPLE9_SCHED_PUBLICATION;
      *writes |= APPLE9_SCHED_PUBLICATION;
      return;
   case AGX_APPLE9_VIR_INTERPOLATION_POSITION:
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
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
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
   if ((ins->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL ||
        ins->op == AGX_APPLE9_VIR_MESH_SUBMIT_SCALAR) && source == 0)
      return 2;
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
        ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
        ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC) && source == 0)
      return MAX2(ins->memory_components, 1);
   if ((ins->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT ||
        ins->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT) &&
       source == ins->memory_components + 1)
      return 2;
   if ((ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT && source == 1) ||
       (ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER && source == 0))
      return 2;
   return 1;
}

/* T8132 register-table probes validate both operands through r95. The full
 * half-register number is scattered across the sample bundle: texture bits
 * 11..14, 68..70, 122; sampler bits 32..35, 65..67, 116. */
static const struct agx_apple9_operand_constraint apple9_table_index[2] = {
   {
      .role = AGX_APPLE9_OPERAND_SRC4,
      .files = AGX_APPLE9_FILE_GPR,
      .widths = AGX_APPLE9_WIDTH_32,
      .min_index = 0,
      .max_index = AGX_APPLE9_GPR_COUNT - 1,
      .alignment_halves = 2,
      .flags = AGX_APPLE9_OPERAND_ALLOCATABLE,
   },
   {
      .role = AGX_APPLE9_OPERAND_SRC5,
      .files = AGX_APPLE9_FILE_GPR,
      .widths = AGX_APPLE9_WIDTH_32,
      .min_index = 0,
      .max_index = AGX_APPLE9_GPR_COUNT - 1,
      .alignment_halves = 2,
      .flags = AGX_APPLE9_OPERAND_ALLOCATABLE,
   },
};

static const struct agx_apple9_operand_constraint *
apple9_source_constraint(const struct agx_apple9_vir_instr *ins, unsigned source)
{
   if (ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE &&
       source + ins->texture_bindless + ins->sampler_bindless >= ins->nr_srcs)
      return &apple9_table_index[ins->sampler_bindless &&
                                 source + 1 == ins->nr_srcs];
   if (ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
       ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC ||
       ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC) {
      if ((ins->encoding == AGX_APPLE9_ENC_DEVICE_STORE_INDIRECT ||
           ins->encoding == AGX_APPLE9_ENC_DEVICE_ATOMIC_INDIRECT) &&
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
   if (ins->local_control)
      return false;
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

static bool apple9_async(const struct agx_apple9_vir_instr *ins);
static bool apple9_reads_sources_at_issue(const struct agx_apple9_vir_instr *ins);

/* Set while a candidate runs as one of a parallel pair (see
 * apple9_candidate_jobs_run): such candidates must not write the shared
 * program. */
static _Thread_local bool apple9_in_candidate_job;

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
      /* No permanent reservation: spilling reserves r0..r4 for its copy,
       * index and memory-swap temporaries. */
      .reserved_registers = 0,
      .spill_reserved_registers = 10,
      .spill_copy_register = 2,
      .spill_swap_registers = {6, 8},
      .late_kill_sources = true,
      .preserve_source_kills = true,
      .defer_spill_lowering = true,
      .disable_occupancy_rematerialization = true,
      .spill_partial_phi_sources = true,
      .mov_imm_registers = 2 * 64,
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
         nir_block *exit = block->nir_exit ?: block->nir;
         for (unsigned s = 0; s < 2; ++s) {
            nir_block *succ = exit->successors[s];
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

   /* Blocks after infinite loops are unreachable yet still feed their
    * successors. Drop those edges, as the common compiler deletes such blocks:
    * RA requires live-in values to be live-out of every predecessor.
    */
   bool pruned;
   do {
      pruned = false;
      agx_foreach_block(ctx, block) {
         if (block == agx_start_block(ctx) || agx_num_predecessors(block))
            continue;
         for (unsigned s = 0; s < ARRAY_SIZE(block->successors); ++s) {
            agx_block *succ = block->successors[s];
            if (!succ)
               continue;
            struct util_dynarray *preds = &succ->predecessors;
            unsigned index = agx_predecessor_index(succ, block);
            unsigned remaining = agx_num_predecessors(succ) - (index + 1);
            memmove(util_dynarray_element(preds, agx_block *, index),
                    util_dynarray_element(preds, agx_block *, index + 1),
                    remaining * sizeof(agx_block *));
            preds->size -= sizeof(agx_block *);
            block->successors[s] = NULL;
            pruned = true;
         }
      }
   } while (pruned);

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

   /* Register tuples collected for operands that are only read, such as a
    * load's base address pair, are reused by later instructions of the same
    * block that read the same values, instead of collecting a copy each. */
   struct {
      uint32_t src[4];
      unsigned width, last;
      agx_index vector;
   } tuples[16];
   unsigned nr_tuples = 0;
   agx_block *tuple_block = NULL;
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *ins = p->instructions[i];
      struct agx_apple9_block *owner = agx_apple9_instr_block(ins);
      agx_block *block = apple9_shared_block(ra, owner);
      if (ins->op == AGX_APPLE9_VIR_PHI)
         continue;
      if (block != tuple_block) {
         tuple_block = block;
         nr_tuples = 0;
      }
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
         bool edge = false;
         agx_foreach_predecessor(succ, pred)
            edge |= *pred == block;
         if (edge)
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
            /* A destroyed operand needs a private tuple. A load's base pair
             * may be a longer tuple's leading components, such as a loaded
             * descriptor's address; its split components share those
             * registers and may outlive the tuple, so the load retains them
             * (EXP-M4-63 validates retained pointer pairs). */
            const struct agx_apple9_operand_constraint *operand =
               apple9_source_constraint(ins, s);
            const bool load = ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_INDIRECT ||
                              ins->encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER;
            agx_index vector = ra->vectors[ins->src[s]];
            bool tuple = !(operand && (operand->flags & AGX_APPLE9_OPERAND_CLOBBER)) &&
                         (load ? agx_channels(vector) >= width
                               : agx_channels(vector) == width);
            for (unsigned c = 1; c < width; ++c)
               tuple &= ins->src[s + c] == ins->src[s] + c;
            const bool shared =
               !(operand && (operand->flags & AGX_APPLE9_OPERAND_CLOBBER)) &&
               width <= 4;
            bool reused = false;
            if (!tuple && shared) {
               for (unsigned t = 0; t < nr_tuples && !reused; ++t) {
                  if (tuples[t].width != width)
                     continue;
                  bool same = true;
                  for (unsigned c = 0; c < width; ++c)
                     same &= tuples[t].src[c] == ins->src[s + c];
                  /* Only for a nearby reader: holding the tuple across a long
                   * stretch adds register pressure to save two copies. */
                  if (same && i - tuples[t].last <= 128) {
                     vector = tuples[t].vector;
                     tuples[t].last = i;
                     reused = true;
                  }
               }
            }
            if (!tuple && !reused) {
               vector = agx_vec_temp(ctx, AGX_SIZE_32, width);
               agx_instr *collect = agx_collect_to(&b, vector, width);
               for (unsigned c = 0; c < width; ++c)
                  collect->src[c] = ra->values[ins->src[s + c]];
               if (shared && nr_tuples < ARRAY_SIZE(tuples)) {
                  tuples[nr_tuples].width = width;
                  for (unsigned c = 0; c < width; ++c)
                     tuples[nr_tuples].src[c] = ins->src[s + c];
                  tuples[nr_tuples].last = i;
                  tuples[nr_tuples++].vector = vector;
               }
            } else if (tuple && load) {
               /* Every candidate derives the same bits; the first (serial)
                * one sets them, so parallel candidates only read them. */
               const uint32_t retain = BITFIELD_RANGE(s, width);
               /* A parallel candidate deriving new bits would race. */
               assert(!apple9_in_candidate_job ||
                      (ins->retain_mask & retain) == retain);
               if ((ins->retain_mask & retain) != retain)
                  ins->retain_mask |= retain;
            }
            sources[nr_srcs] = vector;
         }
         ++nr_srcs;
         s += width;
      }
      bool pure = components && !pub && !ins->publication_handoff &&
                  !ins->memory_volatile &&
                  ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC &&
                  ins->op != AGX_APPLE9_VIR_SHARED_ATOMIC;
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
         /* The register-table form extends the destination at bit 153.
          * The 14-byte immediate form has only six measured GPR bits. */
         if (common->nr_dests && ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE &&
             (ins->texture_bindless || ins->sampler_bindless))
            common->reg_constraints[0].max =
               AGX_APPLE9_HALF_REGISTER_COUNT - 2;
         /* A synchronous instruction reads its sources before publishing its
          * result, also when it releases them, and a memory load reads them
          * at issue, so a scalar result may be written over a source that
          * dies here. */
         if (common->nr_dests)
            common->reg_constraints[0].reuse_killed =
               components == 1 &&
               (!apple9_async(ins) ||
                (apple9_reads_sources_at_issue(ins) &&
                 ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC)) &&
               common->op != AGX_OPCODE_APPLE9_CF &&
               ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC &&
               !agx_apple9_encoding_info(ins->encoding)->split_result;
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
#define SPILL_INDEX_REG 2
#define PUB_BASE  AGX_APPLE9_GPR_COUNT
#define COPY_REG  1
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
apple9_physical_xor(struct agx_apple9_vir_program *out, unsigned dst,
                    unsigned a, unsigned b)
{
   struct agx_apple9_vir_instr ins = apple9_physical_instruction(
      AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, dst);
   ins.src[0] = a;
   ins.src[1] = b;
   ins.nr_srcs = 2;
   ins.live_after_mask = 3;
   return apple9_emit_physical(out, &ins);
}

static bool
apple9_physical_immediate(struct agx_apple9_vir_program *out, unsigned dst,
                          uint32_t value)
{
   /* Rematerialization and PHI copies happen after constrained allocation.
    * The long immediate form has six destination bits; a constant for a
    * higher register goes through the copy register, reserved only for
    * allocations that need it (apple9_needs_copy_register). */
   if (dst >= 64) {
      assert(out->reserved_gprs[COPY_REG]);
      return apple9_physical_immediate(out, COPY_REG, value) &&
             apple9_physical_copy(out, dst, COPY_REG,
                                  AGX_APPLE9_SCOREBOARD_SLOT_NONE);
   }
   struct agx_apple9_vir_instr ins = apple9_physical_instruction(
      AGX_APPLE9_VIR_IMM,
      dst < 16 && value < 0x80 ? AGX_APPLE9_ENC_MOV_IMM_COMPACT
                              : AGX_APPLE9_ENC_MOV_IMM32, dst);
   ins.immediate = value;
   return apple9_emit_physical(out, &ins);
}

/* Spill slots start frame_base words into the invocation frame. */
static bool
apple9_physical_move(struct agx_apple9_vir_program *out, agx_index dst,
                     agx_index src, unsigned frame_base)
{
   assert(dst.type == AGX_INDEX_REGISTER && dst.size == AGX_SIZE_32);
   if (src.type == AGX_INDEX_IMMEDIATE)
      return apple9_physical_immediate(out, dst.value / 2, src.value);
   assert(src.type == AGX_INDEX_REGISTER && src.size == AGX_SIZE_32);
   assert(agx_channels(dst) == agx_channels(src));
   if ((dst.memory || src.memory) && agx_channels(dst) > 1) {
      /* The shared spiller moves whole tuples. SAVE/FILL transfer one word,
       * so lower every component of the allocated memory/register range. */
      unsigned channels = agx_channels(dst);
      dst.channels_m1 = src.channels_m1 = 0;
      for (unsigned c = 0; c < channels; ++c) {
         if (!apple9_physical_move(out, dst, src, frame_base))
            return false;
         dst.value += 2;
         src.value += 2;
      }
      return true;
   }
   if (!dst.memory && !src.memory)
      return apple9_physical_copy(out, dst.value / 2, src.value / 2,
                                  AGX_APPLE9_SCOREBOARD_SLOT_NONE);
   assert(!dst.memory || !src.memory);
   const unsigned word = frame_base + (src.memory ? src : dst).value / 2;
   if (src.memory && word >= AGX_APPLE9_FILL_WORD_COUNT) {
      /* The compact FILL encoding aliases offsets above 4 KiB. Indexed
       * private loads cover the full frame but only address low-bank GPRs.
       * Use the reserved copy register for both the zero index and result,
       * then move the completed word to its allocated destination. */
      struct agx_apple9_vir_instr load = apple9_physical_instruction(
         AGX_APPLE9_VIR_PRIVATE_LOAD, AGX_APPLE9_ENC_PRIVATE_LOAD, COPY_REG);
      load.src[0] = COPY_REG;
      load.nr_srcs = 1;
      load.immediate = word;
      load.producer_scoreboard_slot = SPILL_TAG;
      return apple9_physical_immediate(out, COPY_REG, 0) &&
             apple9_emit_physical(out, &load) &&
             apple9_physical_copy(out, dst.value / 2, COPY_REG,
                                  AGX_APPLE9_SCOREBOARD_SLOT_NONE);
   }
   if (src.memory) {
      struct agx_apple9_vir_instr load = apple9_physical_instruction(
         AGX_APPLE9_VIR_SPILL_LOAD, AGX_APPLE9_ENC_SPILL_LOAD, dst.value / 2);
      load.immediate = word;
      load.producer_scoreboard_slot = SPILL_TAG;
      return apple9_emit_physical(out, &load);
   }
   if (word >= AGX_APPLE9_SAVE_WORD_COUNT) {
      /* Compact SAVE aliases above 8 KiB. Preserve its source through the
       * copy temporary and use a separate reserved zero index for the full
       * frame's indexed store. Parallel-copy cycle registers exclude both. */
      struct agx_apple9_vir_instr store = apple9_physical_instruction(
         AGX_APPLE9_VIR_PRIVATE_STORE, AGX_APPLE9_ENC_PRIVATE_STORE,
         AGX_APPLE9_VREG_INVALID);
      store.src[0] = COPY_REG;
      store.src[1] = SPILL_INDEX_REG;
      store.nr_srcs = 2;
      store.live_after_mask = 2;
      store.immediate = word;
      return apple9_physical_copy(out, COPY_REG, src.value / 2,
                                  AGX_APPLE9_SCOREBOARD_SLOT_NONE) &&
             apple9_physical_immediate(out, SPILL_INDEX_REG, 0) &&
             apple9_emit_physical(out, &store);
   }
   /* SAVE releases its input. A last use can consume the allocated GPR;
    * otherwise preserve the live definition through the copy temporary. */
   struct agx_apple9_vir_instr save = apple9_physical_instruction(
      AGX_APPLE9_VIR_SPILL_STORE, AGX_APPLE9_ENC_SPILL_STORE,
      AGX_APPLE9_VREG_INVALID);
   save.src[0] = src.kill ? src.value / 2 : COPY_REG;
   save.nr_srcs = 1;
   save.immediate = word;
   save.producer_scoreboard_slot = SPILL_TAG;
   return (src.kill || apple9_physical_copy(out, COPY_REG, src.value / 2,
                                           AGX_APPLE9_SCOREBOARD_SLOT_NONE)) &&
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
   /* Unused texture channels keep an allocated destination (see
    * apple9_allocate_candidate) for the read that releases their publication. */
   assert(!(dead_dest && ins.publication_handoff));
   if (dead_dest && ins.op != AGX_APPLE9_VIR_SHARED_ATOMIC) {
      assert(ins.op == AGX_APPLE9_VIR_DEVICE_ATOMIC);
      ins.atomic_discard = true;
      ins.dest_components = 0;
      ins.dest = AGX_APPLE9_VREG_INVALID;
   }
   ins.live_after_mask = 0;
   /* Operands beyond the encoded sources only extend register lifetimes. */
   for (unsigned s = 0, source = 0;
        s < common->nr_srcs && source < original->nr_srcs; ++s) {
      unsigned width = apple9_source_width(original, source);
      for (unsigned c = 0; c < width; ++c, ++source) {
         ins.src[source] = ra->publication[original->src[source]]
                              ? PUB_BASE + ra->program->phys[original->src[source]]
                              : common->src[s].value / 2 + c;
         if (!common->src[s].kill)
            ins.live_after_mask |= BITFIELD_BIT(source);
      }
   }
   ins.live_after_mask |= original->retain_mask;
   if (dead_dest && ins.op == AGX_APPLE9_VIR_SHARED_ATOMIC) {
      /* Shared atomics always return a value. The allocator gives their
       * consumed data a private dead copy, so reuse that register as the
       * unused result destination and retire it through normal completion.
       */
      ins.dest = ins.src[0];
   }
   /* Indexed shader-local arrays follow the allocator's spill region.
    * Spill reloads emitted by physical_move already address that region. */
   if (ins.op == AGX_APPLE9_VIR_PRIVATE_LOAD ||
       ins.op == AGX_APPLE9_VIR_PRIVATE_STORE)
      ins.immediate += ALIGN_POT(ra->ctx->scratch_size_B, 16) / 4;
   if (ins.branch_target)
      ins.branch_target = blocks[ins.branch_target->index];
   ins.phi_block = NULL;
   ins.phi_edge = ins.completion_consumer = NULL;
   ins.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
   ins.device_load_index_kind =
      ins.encoding == AGX_APPLE9_ENC_DEVICE_LOAD_POINTER
         ? AGX_APPLE9_DEVICE_LOAD_INDEX_NONE
      : (ins.live_after_mask & 1) ? AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR
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

/* A slot may be pending on more than one producer when control-flow paths
 * merge. Each path contributes its own producer; waiting covers all of them.
 * A merge beyond the tracked capacity forgets the producers and waits on
 * entry to the block. */
#define APPLE9_PENDING_MERGE 8

/* A slot counts its outstanding operations: native Metal issues up to seven
 * loads and texture reads on one slot and waits for all of them with a
 * single consumer dependency (G13 allows eight). Never exceed seven. */
#define APPLE9_SLOT_SHARE_LIMIT 7

struct apple9_pending {
   unsigned count;
   unsigned latest_issue;
   /* Input position of the first instruction in this block that needs any
    * of the producers, or UINT_MAX when unknown or outside the block. */
   unsigned first_need;
   bool unknown;
   struct agx_apple9_vir_instr producer[APPLE9_PENDING_MERGE];
};

/* Pending instructions include fixed producers with no GPR result. Each is
 * stored once, under the slot selected by its machine descriptor. */
struct apple9_completion {
   struct apple9_pending pending[7];
};

/* Scalar export producers complete asynchronously in the publication namespace.
 * Their completion tag is independent of the publication's storage index. */
static bool
apple9_scalar_publication(const struct agx_apple9_vir_instr *ins)
{
   return ins->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
          ins->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT;
}

static unsigned
apple9_fixed_producer_slot(const struct agx_apple9_vir_instr *ins)
{
   if (ins->encoding == AGX_APPLE9_ENC_PSEUDO)
      return AGX_APPLE9_SCOREBOARD_SLOT_NONE;

   unsigned slot =
      agx_apple9_encoding_info(ins->encoding)->fixed_producer_slot;
   assert(slot <= AGX_APPLE9_SCOREBOARD_SLOT_6);
   return slot;
}

static bool
apple9_async(const struct agx_apple9_vir_instr *ins)
{
   if (apple9_scalar_publication(ins) || apple9_fixed_producer_slot(ins))
      return true;
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
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
apple9_pending_active(const struct apple9_pending *p)
{
   return p->count || p->unknown;
}

static unsigned
apple9_producer_slots(const struct agx_apple9_vir_instr *producer)
{
   assert(producer->producer_scoreboard_slot >= 1 &&
          producer->producer_scoreboard_slot <= 6);
   unsigned slots = BITFIELD_BIT(producer->producer_scoreboard_slot - 1);
   if (apple9_fixed_producer_slot(producer))
      slots |=
         agx_apple9_encoding_info(producer->encoding)->fixed_producer_alternates;
   return slots;
}

/* Return the canonical pending slots that conflict with a new reservation.
 * U0 reserves both selectors until either completes it; this never aliases
 * unrelated ordinary slot-1 and slot-2 producers. Unknown joins are drained
 * before issuing any instruction in the block. */
static unsigned
apple9_conflicting_slots(const struct apple9_completion *state, unsigned slots)
{
   unsigned conflicts = 0;
   for (unsigned slot = 1; slot <= 6; ++slot) {
      const struct apple9_pending *p = &state->pending[slot];
      bool conflict = p->unknown && (slots & BITFIELD_BIT(slot - 1));
      for (unsigned i = 0; i < p->count; ++i)
         conflict |= (slots & apple9_producer_slots(&p->producer[i])) != 0;
      if (conflict)
         conflicts |= BITFIELD_BIT(slot - 1);
   }
   return conflicts;
}

/* Apply an actual input wait, folded or standalone, to each producer. At a
 * join, slot 1 may contain U0 on one path and an ordinary load on another:
 * waiting on slot 2 completes only the U0 alternative in that list. */
static void
apple9_complete(struct apple9_completion *state, unsigned mask)
{
   for (unsigned slot = 1; slot <= 6; ++slot) {
      struct apple9_pending *p = &state->pending[slot];
      if (mask & BITFIELD_BIT(slot - 1)) {
         *p = (struct apple9_pending){0};
         continue;
      }
      unsigned count = 0;
      for (unsigned i = 0; i < p->count; ++i) {
         if (!(apple9_producer_slots(&p->producer[i]) & mask))
            p->producer[count++] = p->producer[i];
      }
      p->count = count;
   }
}

static bool
apple9_producer_result(const struct agx_apple9_vir_instr *producer, unsigned reg)
{
   return reg >= producer->dest && reg - producer->dest < producer->dest_components;
}

static bool
apple9_pending_result(const struct apple9_pending *p, unsigned reg)
{
   for (unsigned i = 0; i < p->count; ++i) {
      if (apple9_producer_result(&p->producer[i], reg))
         return true;
   }
   return false;
}

/* Wait for every slot in the mask with one standalone WAIT. Its operand is
 * the slot mask itself; it reads and writes no register. */
static bool
apple9_retire(struct agx_apple9_vir_program *out,
              struct apple9_completion *state, unsigned mask)
{
   struct apple9_pending *pending = state->pending;
   unsigned active = 0;
   u_foreach_bit(bit, mask) {
      if (apple9_pending_active(&pending[bit + 1]))
         active |= BITFIELD_BIT(bit);
   }
   if (!active)
      return true;
   struct agx_apple9_vir_instr wait = apple9_physical_instruction(
      AGX_APPLE9_VIR_WAIT, AGX_APPLE9_ENC_WAIT, AGX_APPLE9_VREG_INVALID);
   wait.immediate = active;
   if (!apple9_emit_physical(out, &wait))
      return false;
   apple9_complete(state, active);
   return true;
}

static bool
apple9_can_fold_wait(const struct agx_apple9_vir_instr *ins,
                     const struct agx_apple9_vir_instr *producer)
{
   if (ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC)
      return false;
   if (agx_apple9_encoding_info(ins->encoding)->dependency_layout ==
       AGX_APPLE9_DEPENDENCY_NONE)
      return false;
   if (producer->op == AGX_APPLE9_VIR_SPILL_STORE)
      return true;
   bool reads = false;
   for (unsigned s = 0; s < ins->nr_srcs; ++s)
      reads |= apple9_producer_result(producer, ins->src[s]);
   /* An instruction that only orders against the producer (it overwrites
    * the result or a borrowed source, or accesses the same memory) waits in
    * its own dependency field before it executes, as a standalone WAIT
    * would. Atomics' dependency field gates their inputs (EXP-M4-50). */
   if (!reads)
      return ins->op != AGX_APPLE9_VIR_DEVICE_ATOMIC;
   /* Publication readers wait through their input mask (EXP-M4-61);
    * native coverage instructions wait for their publication the same way. */
   if (apple9_scalar_publication(producer))
      return (ins->op == AGX_APPLE9_VIR_VARY_STORE ||
              ins->op == AGX_APPLE9_VIR_OBJECT_GRID_EXPORT ||
              ins->op == AGX_APPLE9_VIR_COVERAGE) &&
             ins->src[0] == producer->dest;
   if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC &&
       (ins->atomic_discard || ins->atomic_op == AGX_APPLE9_ATOMIC_CMPXCHG ||
        producer->op != AGX_APPLE9_VIR_DEVICE_LOAD ||
        producer->dest_components != 1 || ins->src[0] != producer->dest))
      return false;
   /* An identity copy explicitly requests materialization, independently of
    * which tuple components or later uses need the completed value. */
   if (apple9_identity(ins) && ins->src[0] == producer->dest)
      return true;
   /* Completion makes the whole producer tuple available. Keeping operands
    * live is independent of releasing its slot, but needs a source release
    * flag the consumer can clear: EXP-M4-34/35/63 for device loads, system
    * values and textures; CTS probes for atomic, private, shared and
    * interpolation results, including texture results in integer consumers.
    * Scratch fills keep their materialized form. */
   bool tuple = producer->op != AGX_APPLE9_VIR_SPILL_LOAD;
   bool can_retain =
      tuple && agx_apple9_encoding_info(ins->encoding)->dependency_retains_sources;
   unsigned consumed = 0;
   for (unsigned s = 0; s < ins->nr_srcs; ++s) {
      if (!apple9_producer_result(producer, ins->src[s]))
         continue;
      if ((ins->live_after_mask & BITFIELD_BIT(s)) && !can_retain)
         return false;
      consumed |= BITFIELD_BIT(ins->src[s] - producer->dest);
   }
   return consumed == BITFIELD_MASK(producer->dest_components) ||
          (tuple && consumed != 0);
}

/* Producers merged from several paths share the slot, so one folded wait
 * retires whichever of them is outstanding. It must be valid for each. */
static bool
apple9_can_fold_pending(const struct agx_apple9_vir_instr *ins,
                        const struct apple9_pending *p)
{
   if (p->unknown || !p->count)
      return false;
   for (unsigned i = 0; i < p->count; ++i) {
      if (!apple9_can_fold_wait(ins, &p->producer[i]))
         return false;
   }
   return true;
}

/* Loop control repeats or leaves the body; results stay pending only across
 * forward execution-mask changes and jumps. */
static bool
apple9_loop_control(const struct agx_apple9_vir_instr *ins)
{
   switch (ins->op) {
   case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
   case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
   case AGX_APPLE9_VIR_LOOP_MASK_POP:
   case AGX_APPLE9_VIR_JMP_EXEC_ANY:
   case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
      return true;
   default:
      return false;
   }
}

static bool
apple9_completion_barrier(const struct agx_apple9_vir_instr *ins, bool drain)
{
   if (apple9_after_logical_end(ins))
      return drain || apple9_loop_control(ins);
   return ins->encoding != AGX_APPLE9_ENC_PSEUDO &&
          agx_apple9_encoding_info(ins->encoding)->requires_prior_completion;
}

/* Memory operations read their address, index and data registers at issue.
 * Native Metal overwrites a pending device load's address or index 3,744
 * times in 4,000 public pipelines without waiting. Overwriting atomic,
 * private-load and shared-load inputs right after issue passed the CTS
 * atomic, indexing and shared-memory groups. */
static bool
apple9_reads_sources_at_issue(const struct agx_apple9_vir_instr *ins)
{
   switch (ins->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_MESH_CONTROL:
      return true;
   default:
      return false;
   }
}

static bool
apple9_producer_hazard(const struct agx_apple9_vir_instr *ins,
                       const struct agx_apple9_vir_instr *producer)
{
   /* T8132 r6 isolates the first channel submit as U0's publication consumer.
    * The post-U0 store is not a consumer. A/B are read at issue, so ordinary
    * register reuse must not manufacture an earlier publication deadline. */
   if (producer->op == AGX_APPLE9_VIR_MESH_CONTROL &&
       ins->op == AGX_APPLE9_VIR_MESH_SUBMIT_CHANNEL)
      return true;
   /* SAVE destroys its source just as a destination write would. In
    * particular, wait for earlier asynchronous borrowers before consuming it. */
   bool save = ins->op == AGX_APPLE9_VIR_SPILL_STORE;
   for (unsigned c = 0; c < ins->dest_components + save; ++c) {
      unsigned reg = save ? ins->src[0] : ins->dest + c;
      if (apple9_producer_result(producer, reg))
         return true;
      /* Includes texture parameter publications, which remain borrowed
       * until the result handoff, and scratch SAVE's destructive input. */
      for (unsigned s = 0;
           !apple9_reads_sources_at_issue(producer) && s < producer->nr_srcs;
           ++s) {
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
   /* Bindings name distinct buffers; global addresses may alias each other. */
   if ((ins->op == AGX_APPLE9_VIR_DEVICE_STORE ||
        ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) &&
       (producer->op == AGX_APPLE9_VIR_DEVICE_LOAD ||
        producer->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) &&
       ins->memory_global == producer->memory_global &&
       (ins->memory_global || ins->immediate == producer->immediate))
      return true;
   if (ins->op == AGX_APPLE9_VIR_PRIVATE_STORE &&
       producer->op == AGX_APPLE9_VIR_PRIVATE_LOAD)
      return true;
   if ((ins->op == AGX_APPLE9_VIR_SHARED_STORE ||
        ins->op == AGX_APPLE9_VIR_SHARED_ATOMIC) &&
       (producer->op == AGX_APPLE9_VIR_SHARED_LOAD ||
        producer->op == AGX_APPLE9_VIR_SHARED_ATOMIC))
      return true;
   if (ins->op == AGX_APPLE9_VIR_SHARED_LOAD &&
       producer->op == AGX_APPLE9_VIR_SHARED_ATOMIC)
      return true;
   /* This compound encoding contains a fixed slot-1 publication and wait. */
   return (ins->encoding == AGX_APPLE9_ENC_BLOCK_IMAGE_STORE ||
           ins->encoding == AGX_APPLE9_ENC_IMAGE_WRITE) &&
          producer->producer_scoreboard_slot == AGX_APPLE9_SCOREBOARD_SLOT_1;
}

static enum agx_apple9_completion_class
apple9_completion_class(const struct agx_apple9_vir_instr *producer)
{
   if (apple9_scalar_publication(producer))
      return AGX_APPLE9_COMPLETION_PUBLICATION;
   /* Native T8132 Metal leaves device and threadgroup loads and returning
    * atomics pending across fences and barriers. Threadgroup and private
    * loads have fixed slots but are memory results too. Texture reads are
    * not: before a texture barrier Metal waits for a pending read_write
    * texture read, and our storage-image barriers have no such form. */
   switch (producer->op) {
   case AGX_APPLE9_VIR_DEVICE_LOAD:
   case AGX_APPLE9_VIR_DEVICE_ATOMIC:
   case AGX_APPLE9_VIR_PRIVATE_LOAD:
   case AGX_APPLE9_VIR_SHARED_LOAD:
   case AGX_APPLE9_VIR_SHARED_ATOMIC:
      return AGX_APPLE9_COMPLETION_MEMORY;
   default:
      return AGX_APPLE9_COMPLETION_OTHER;
   }
}

/* Producer classes a completion barrier leaves pending. Loop control and
 * drained block ends wait for everything, as do the fence forms other than
 * the plain device fence. */
static unsigned
apple9_completion_exempt(const struct agx_apple9_vir_instr *ins)
{
   if (apple9_after_logical_end(ins) || ins->encoding == AGX_APPLE9_ENC_PSEUDO)
      return 0;
   if (ins->op == AGX_APPLE9_VIR_DEVICE_FENCE &&
       ins->immediate != AGX_APPLE9_FENCE_DEVICE)
      return 0;
   return agx_apple9_encoding_info(ins->encoding)->prior_completion_exempt;
}

static bool
apple9_pending_hazard(const struct agx_apple9_vir_instr *ins,
                      const struct apple9_pending *p, bool barrier)
{
   if (!apple9_pending_active(p))
      return false;
   if (p->unknown)
      return true;
   const unsigned exempt = barrier ? apple9_completion_exempt(ins) : 0;
   for (unsigned i = 0; i < p->count; ++i) {
      if (barrier && !(exempt & apple9_completion_class(&p->producer[i])))
         return true;
      if (apple9_producer_hazard(ins, &p->producer[i]))
         return true;
   }
   return false;
}

static bool
apple9_same_producer(const struct agx_apple9_vir_instr *a,
                     const struct agx_apple9_vir_instr *b)
{
   if (a->op != b->op || a->encoding != b->encoding || a->dest != b->dest ||
       a->dest_components != b->dest_components || a->nr_srcs != b->nr_srcs ||
       a->immediate != b->immediate ||
       a->producer_scoreboard_slot != b->producer_scoreboard_slot)
      return false;
   for (unsigned s = 0; s < a->nr_srcs; ++s) {
      if (a->src[s] != b->src[s])
         return false;
   }
   return true;
}

/* Add one path's pending producers to a successor's entry state. */
static void
apple9_merge_pending(struct apple9_completion *to_state,
                     const struct apple9_completion *from_state)
{
   struct apple9_pending *to = to_state->pending;
   const struct apple9_pending *from = from_state->pending;
   for (unsigned slot = 1; slot <= 6; ++slot) {
      struct apple9_pending *t = &to[slot];
      const struct apple9_pending *f = &from[slot];
      t->unknown |= f->unknown;
      t->latest_issue = MAX2(t->latest_issue, f->latest_issue);
      t->first_need = UINT_MAX;
      for (unsigned i = 0; i < f->count; ++i) {
         bool present = false;
         for (unsigned j = 0; j < t->count; ++j)
            present |= apple9_same_producer(&t->producer[j], &f->producer[i]);
         if (present)
            continue;
         if (t->count == APPLE9_PENDING_MERGE)
            t->unknown = true;
         else
            t->producer[t->count++] = f->producer[i];
      }
      if (t->unknown)
         t->count = 0;
   }
}

/* Producers that may share a counting slot with each other. Fixed producers,
 * publications, atomics and scratch traffic keep a slot to themselves. */
static bool
apple9_shares_slot(const struct agx_apple9_vir_instr *ins)
{
   return !apple9_fixed_producer_slot(ins) &&
          (ins->op == AGX_APPLE9_VIR_DEVICE_LOAD ||
           ins->op == AGX_APPLE9_VIR_TEXTURE_SAMPLE);
}

/* The first instruction in [start, end) that needs the producer's completion,
 * by the same tests the scheduler applies, or UINT_MAX. */
static unsigned
apple9_first_need(const struct agx_apple9_vir_program *program, unsigned start,
                  unsigned end, const struct agx_apple9_vir_instr *producer,
                  bool drain)
{
   const unsigned limit = MIN2(end, start + 512);
   for (unsigned j = start; j < limit; ++j) {
      const struct agx_apple9_vir_instr *ins = program->instructions[j];
      if (ins->op == AGX_APPLE9_VIR_WAIT ||
          ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC_RESULT)
         continue;
      if (apple9_completion_barrier(ins, drain) &&
          !(apple9_completion_exempt(ins) & apple9_completion_class(producer)))
         return j;
      if (apple9_producer_hazard(ins, producer))
         return j;
      for (unsigned s = 0; s < ins->nr_srcs; ++s) {
         if (apple9_producer_result(producer, ins->src[s]))
            return j;
      }
   }
   return UINT_MAX;
}

/* Slots that fixed-slot producers issued in [start, end) will claim. A free
 * slot chosen for an ordinary producer should not be one of them, or the
 * fixed producer must retire it first. */
static unsigned
apple9_fixed_slots_ahead(const struct agx_apple9_vir_program *program,
                         unsigned start, unsigned end)
{
   unsigned slots = 0;
   for (unsigned j = start; j < MIN2(end, start + 512); ++j) {
      const struct agx_apple9_vir_instr *ins = program->instructions[j];
      unsigned slot = apple9_fixed_producer_slot(ins);
      if (slot) {
         slots |= BITFIELD_BIT(slot - 1) |
                  agx_apple9_encoding_info(ins->encoding)->fixed_producer_alternates;
      }
   }
   return slots;
}

/* Whether a sharing producer can be added to the slot without retiring it. */
static bool
apple9_slot_joinable(const struct apple9_completion *state, unsigned slot)
{
   const struct apple9_pending *p = &state->pending[slot];
   if (!p->count || p->unknown || p->count >= APPLE9_SLOT_SHARE_LIMIT)
      return false;
   for (unsigned i = 0; i < p->count; ++i) {
      if (!apple9_shares_slot(&p->producer[i]))
         return false;
   }
   return apple9_conflicting_slots(state, BITFIELD_BIT(slot - 1)) ==
          BITFIELD_BIT(slot - 1);
}

/* Choose a slot to join for a sharing producer first needed at need. With
 * free slots, only a slot whose first need is the same instruction: that
 * consumer waits once for both, and nothing waits earlier than it would.
 * When every slot is occupied, joining still beats retiring a slot now:
 * prefer the slot needed soonest at or after need (older producers are
 * waited for slightly early), else the one needed latest. */
static unsigned
apple9_join_slot(const struct apple9_completion *state, unsigned need,
                 bool exhausted)
{
   unsigned best = 0;
   for (unsigned slot = 1; slot <= 6; ++slot) {
      if (!apple9_slot_joinable(state, slot))
         continue;
      unsigned other = state->pending[slot].first_need;
      if (!exhausted) {
         if (need != UINT_MAX && other == need)
            return slot;
         continue;
      }
      if (!best) {
         best = slot;
         continue;
      }
      unsigned current = state->pending[best].first_need;
      bool after = other >= need, best_after = current >= need;
      if (after != best_after ? after
                              : after ? other < current : other > current)
         best = slot;
   }
   return best;
}

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
       !I->src_hi_mask &&
       (I->op == AGX_APPLE9_VIR_FADD || I->op == AGX_APPLE9_VIR_FMUL)) {
      I->encoding = AGX_APPLE9_ENC_FLOAT2_BASE;
   } else if (I->encoding == AGX_APPLE9_ENC_FLOAT3_EXTENDED &&
              !(inline_mask & ~2) && !I->src_half_mask &&
              (!(I->alu_src_uniform_mask & 2) || I->alu_src_value[1] < 64) &&
              !((I->alu_src_immediate_mask & 2) && (I->alu_src_value[1] >> 31))) {
      I->encoding = AGX_APPLE9_ENC_FLOAT3_COMPACT;
   }
}

/* Schedule the final physical stream: allocation can introduce earlier uses,
 * register reuse and scratch traffic that do not exist in the original SSA.
 * Pending producers flow forward across execution-mask changes and forward
 * jumps, as native Metal code keeps them; each successor starts from the
 * union of its predecessors' pending producers. Loop control, back edges and
 * entry to a loop header wait for everything outstanding. */
static bool
apple9_schedule_physical(struct agx_apple9_vir_program *program,
                          const char **reason)
{
   struct agx_apple9_vir_program out;
   if (!apple9_init_physical(&out)) {
      agx_apple9_vir_finish(&out);
      return false;
   }
   memcpy(out.reserved_gprs, program->reserved_gprs, sizeof(out.reserved_gprs));
   unsigned count = program->last_block->index + 1;
   struct agx_apple9_block **blocks = calloc(count, sizeof(*blocks));
   /* Machine successors (fall-through and jump target), then logical CFG
    * successors, which also name loop back edges. */
   struct agx_apple9_block **successors = calloc(4 * count, sizeof(*successors));
   unsigned *position = calloc(count, sizeof(*position));
   bool *header = calloc(count, sizeof(*header));
   struct apple9_completion *entry = calloc(count, sizeof(*entry));
   struct hash_table *nir_blocks = _mesa_pointer_hash_table_create(NULL);
   bool success = false;
   if (!blocks || !successors || !position || !header || !entry || !nir_blocks)
      goto cleanup;
   /* Several physical blocks may come from one NIR block. As with the old
    * layout scan, logical successors name its last physical block. Build the
    * mapping once instead of searching the whole shader for every CFG edge.
    */
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      if (block->nir)
         _mesa_hash_table_insert(nir_blocks, block->nir, block);
   }
   for (unsigned b = 0; b < count; ++b) {
      blocks[b] = agx_apple9_block_create(&out);
      if (!blocks[b])
         goto cleanup;
   }
   unsigned layout = 0;
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      blocks[block->index]->nir = block->nir;
      blocks[block->index]->nir_exit = block->nir_exit;
      position[block->index] = layout++;
      unsigned end = block->next ? block->next->start_index : program->instruction_count;
      struct agx_apple9_block **succ = &successors[4 * block->index];
      succ[0] = block->next;
      if (end > block->start_index) {
         const struct agx_apple9_vir_instr *last = program->instructions[end - 1];
         if (last->op == AGX_APPLE9_VIR_JMP_EXEC_ANY ||
             last->op == AGX_APPLE9_VIR_JMP_EXEC_NONE)
            succ[1] = last->branch_target;
      }
      nir_block *exit = block->nir_exit ?: block->nir;
      for (unsigned s = 0; exit && s < 2; ++s) {
         if (!exit->successors[s])
            continue;
         struct hash_entry *entry =
            _mesa_hash_table_search(nir_blocks, exit->successors[s]);
         if (entry)
            succ[2 + s] = entry->data;
      }
   }
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      for (unsigned i = 0; i < 4; ++i) {
         struct agx_apple9_block *succ = successors[4 * block->index + i];
         if (succ && position[succ->index] <= position[block->index])
            header[succ->index] = true;
      }
   }
   for (struct agx_apple9_block *block = program->blocks; block; block = block->next) {
      agx_apple9_block_begin(&out, blocks[block->index]);
      struct apple9_completion state = {0};
      if (!header[block->index])
         state = entry[block->index];
      struct apple9_pending *pending = state.pending;
      unsigned unknown = 0;
      for (unsigned slot = 1; slot <= 6; ++slot) {
         if (pending[slot].unknown)
            unknown |= BITFIELD_BIT(slot - 1);
      }
      if (!apple9_retire(&out, &state, unknown))
         goto cleanup;
      struct agx_apple9_block *const *succ = &successors[4 * block->index];
      bool drain = !succ[0] && !succ[1] && !succ[2] && !succ[3];
      for (unsigned i = 0; i < 4; ++i) {
         drain |= succ[i] && (header[succ[i]->index] ||
                              position[succ[i]->index] <= position[block->index]);
      }
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

         if (ins.op == AGX_APPLE9_VIR_WAIT) {
            apple9_complete(&state, ins.immediate);
            if (!apple9_emit_physical(&out, &ins))
               goto cleanup;
            continue;
         }

         ins.scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_NONE;
         const bool barrier = apple9_completion_barrier(&ins, drain);
         unsigned fold = 0, retire = 0;
         for (unsigned slot = 1; slot <= 6; ++slot) {
            const struct apple9_pending *p = &pending[slot];
            if (!apple9_pending_active(p))
               continue;
            bool needed = apple9_pending_hazard(&ins, p, barrier);
            for (unsigned s = 0; s < ins.nr_srcs; ++s)
               needed |= apple9_pending_result(p, ins.src[s]);
            if (!needed)
               continue;
            if (!fold && apple9_can_fold_pending(&ins, p))
               fold = slot;
            else
               retire |= BITFIELD_BIT(slot - 1);
         }
         if (!apple9_retire(&out, &state, retire))
            goto cleanup;
         /* An alternate selector in the standalone wait may already have
          * completed the producer chosen for folding. */
         if (fold && !apple9_pending_active(&pending[fold]))
            fold = 0;
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
            apple9_complete(&state, BITFIELD_BIT(fold - 1));
         }

         if (apple9_async(&ins)) {
            unsigned slot = apple9_fixed_producer_slot(&ins);
            const bool share = apple9_shares_slot(&ins);
            const unsigned need =
               share ? apple9_first_need(program, i + 1, end, &ins, drain)
                     : UINT_MAX;
            bool joined = false;
            if (share) {
               slot = apple9_join_slot(&state, need, false);
               joined = slot != 0;
            }
            if (!slot) {
               /* Samples and memory traffic share six completion tags.
                * Keep independent operations pending until a real hazard,
                * consumer, or tag exhaustion requires their result handoff. */
               unsigned first = ins.op == AGX_APPLE9_VIR_TEXTURE_SAMPLE ? 1 : 6;
               const unsigned avoid = apple9_fixed_slots_ahead(
                  program, i + 1, need == UINT_MAX ? end : need);
               for (unsigned pass = 0; !slot && pass < 2; ++pass) {
                  for (unsigned p = 0; p < 6; ++p) {
                     unsigned candidate = (first - 1 + p) % 6 + 1;
                     if ((!pass && (avoid & BITFIELD_BIT(candidate - 1))) ||
                         apple9_conflicting_slots(
                            &state, BITFIELD_BIT(candidate - 1)))
                        continue;
                     slot = candidate;
                     break;
                  }
               }
               if (!slot && share) {
                  slot = apple9_join_slot(&state, need, true);
                  joined = slot != 0;
               }
               if (!slot) {
                  /* All tags are occupied. Retire the oldest pending group,
                   * allowing recent loads to overlap instead of repeatedly
                   * draining the newest producer in the preferred slot.
                   * At a CFG merge use the latest issue on either path. */
                  slot = first;
                  for (unsigned candidate = 1; candidate <= 6; ++candidate) {
                     if (pending[candidate].latest_issue < pending[slot].latest_issue)
                        slot = candidate;
                  }
               }
            }
            unsigned reservations = BITFIELD_BIT(slot - 1);
            if (apple9_fixed_producer_slot(&ins))
               reservations |=
                  agx_apple9_encoding_info(ins.encoding)->fixed_producer_alternates;
            /* In particular, never issue another U0 while one is pending.
             * The fully waited double-U0 r6 control timed out, so this is a
             * compiler scheduling invariant, not a claim that replay is safe. */
            if (!joined &&
                !apple9_retire(&out, &state,
                               apple9_conflicting_slots(&state, reservations)))
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
            if (joined) {
               struct apple9_pending *p = &pending[slot];
               p->producer[p->count++] = ins;
               p->latest_issue = out.instruction_count;
               p->first_need = MIN2(p->first_need, need);
            } else {
               pending[slot] = (struct apple9_pending){
                  .count = 1, .latest_issue = out.instruction_count,
                  .first_need = need, .producer = {ins}};
            }
         }
         apple9_select_compact_alu(&ins);
         if (!apple9_emit_physical(&out, &ins))
            goto cleanup;
      }
      if (drain) {
         if (!apple9_retire(&out, &state, BITFIELD_MASK(6)))
            goto cleanup;
      } else {
         for (unsigned i = 0; i < 4; ++i) {
            bool repeated = false;
            for (unsigned j = 0; j < i; ++j)
               repeated |= succ[j] == succ[i];
            if (succ[i] && !repeated)
               apple9_merge_pending(&entry[succ[i]->index], &state);
         }
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
   free(successors);
   free(position);
   free(header);
   free(entry);
   _mesa_hash_table_destroy(nir_blocks, NULL);
   agx_apple9_vir_finish(&out);
   return success;
}

static const char apple9_copy_register_reason[] =
   "Apple9 allocation needs the copy register for a high-bank constant";

/* Whether any immediate move after allocation targets r64 or above. */
static bool
apple9_needs_copy_register(agx_context *ctx)
{
   agx_foreach_instr_global(ctx, I) {
      if ((I->op == AGX_OPCODE_MOV_IMM ||
           (I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_IMMEDIATE)) &&
          I->dest[0].type == AGX_INDEX_REGISTER && !I->dest[0].memory &&
          I->dest[0].value / 2 >= 64)
         return true;
   }
   return false;
}

/* The pop restores the mask that its push saved, so an else immediately
 * followed by its pop has no effect; native one-armed ifs are a push and a
 * pop. Only once allocation is complete is an empty arm known to stay empty,
 * since allocation can place edge copies there. */
static void
apple9_remove_empty_else(struct agx_apple9_vir_program *p)
{
   for (unsigned i = 0; i + 1 < p->instruction_count;) {
      if (p->instructions[i]->op == AGX_APPLE9_VIR_EXEC_MASK_ELSE &&
          p->instructions[i + 1]->op == AGX_APPLE9_VIR_EXEC_MASK_POP)
         agx_apple9_vir_remove(p, p->instructions[i]);
      else
         ++i;
   }
}

static bool
apple9_returns_device_atomic(const struct agx_apple9_vir_program *program)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = program->instructions[i];
      if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC && !ins->atomic_discard)
         return true;
   }
   return false;
}

static const char apple9_duplicate_candidate_reason[] =
   "Apple9 allocation candidate repeats an evaluated schedule";

/* Everything after scheduling depends only on the scheduled IR and these
 * policy inputs, so an equal digest means an identical allocation, which
 * can never be strictly cheaper than the one already compared. */
static _Thread_local bool agx_apple9_ra_probe;

void
agx_apple9_ra_set_probe(bool probe)
{
   agx_apple9_ra_probe = probe;
}

bool
agx_apple9_ra_is_probe(void)
{
   return agx_apple9_ra_probe;
}

static bool
apple9_seen_schedule(agx_context *ctx, struct util_dynarray *seen,
                     bool cost_constraints, bool copy_register,
                     bool uniform_remat)
{
   /* Fingerprint the scheduled IR from its fields rather than its printed
    * form: operands, register constraints and every instruction field from
    * the opcode on. Branch targets are blocks of this candidate's own context,
    * so they enter as block indices. */
   struct mesa_blake3 sha;
   _mesa_blake3_init(&sha);
   const uint8_t flags[] = {cost_constraints, copy_register, uniform_remat};
   _mesa_blake3_update(&sha, flags, sizeof(flags));
   struct hash_table *blocks = _mesa_pointer_hash_table_create(NULL);
   if (!blocks)
      return false;
   agx_foreach_block(ctx, block)
      _mesa_hash_table_insert(blocks, block, (void *)(uintptr_t)(block->index + 1));
   agx_foreach_block(ctx, block) {
      uint32_t index = block->index;
      _mesa_blake3_update(&sha, &index, sizeof(index));
      agx_foreach_instr_in_block(block, I) {
         _mesa_blake3_update(&sha, I->dest, sizeof(agx_index) * I->nr_dests);
         _mesa_blake3_update(&sha, I->src, sizeof(agx_index) * I->nr_srcs);
         const uint8_t constrained = I->reg_constraints != NULL;
         _mesa_blake3_update(&sha, &constrained, 1);
         if (constrained) {
            _mesa_blake3_update(&sha, I->reg_constraints,
                                sizeof(struct agx_reg_constraint) *
                                   (I->nr_dests + I->nr_srcs));
         }
         agx_instr copy = *I;
         struct hash_entry *target = _mesa_hash_table_search(blocks, copy.target);
         if (target)
            copy.imm = (uintptr_t)target->data | (1ull << 63);
         _mesa_blake3_update(&sha, &copy.op,
                             sizeof(agx_instr) - offsetof(agx_instr, op));
      }
   }
   _mesa_hash_table_destroy(blocks, NULL);
   blake3_hash digest;
   _mesa_blake3_final(&sha, digest);
   for (unsigned off = 0; off + BLAKE3_OUT_LEN <= seen->size;
        off += BLAKE3_OUT_LEN) {
      if (!memcmp((const uint8_t *)seen->data + off, digest, BLAKE3_OUT_LEN))
         return true;
   }
   util_dynarray_append_array(seen, uint8_t, digest, BLAKE3_OUT_LEN);
   return false;
}

static bool
apple9_allocate_candidate(struct agx_apple9_vir_program *program,
                          nir_shader *nir, bool cost_constraints, bool latency,
                          unsigned pressure_slack, bool copy_register, bool uniform_remat,
                          struct agx_apple9_vir_program *result,
                          const char **reason, bool *remat_used,
                          unsigned *rejected_increase,
                          struct util_dynarray *seen)
{
   struct apple9_shared_ra ra = {.program = program};
   ra.ctx = rzalloc(NULL, agx_context);
   struct agx_apple9_vir_program out;
   agx_apple9_vir_init(&out);
   bool success = false;
   if (!ra.ctx || !apple9_shared_dataflow(&ra, nir, reason))
      goto cleanup;
   ra.ctx->ra_target.cost_register_constraints = cost_constraints;
   ra.ctx->ra_target.disable_uniform_rematerialization = !uniform_remat;
   /* A returning device atomic stores its old value in invocation-frame word
    * 0, which its result record then loads (agx_apple9_pack_device_atomic);
    * that store overwrote spill slot 0. Spills and private arrays start one
    * 16-byte frame granule above it. */
   if (apple9_returns_device_atomic(program))
      ra.ctx->scratch_size_B = 16;
   if (copy_register)
      ra.ctx->ra_target.reserved_registers = 4;
   if (latency) {
      ra.ctx->schedule_resources = apple9_schedule_resources;
      ra.ctx->schedule_delay = apple9_schedule_delay;
      ra.ctx->schedule_pressure_slack = pressure_slack;
   }
   agx_dce(ra.ctx, true);
   /* Unused texture channels still need the read that releases their
    * borrowed publication; give that read a register of its own. */
   agx_foreach_instr_global(ra.ctx, I) {
      if ((I->op == AGX_OPCODE_APPLE9 || I->op == AGX_OPCODE_APPLE9_PURE) &&
          I->nr_dests == 1 && agx_is_null(I->dest[0]) &&
          I->apple9->publication_handoff)
         I->dest[0] = agx_temp(ra.ctx, AGX_SIZE_32);
   }
   agx_pressure_schedule(ra.ctx);
   if (seen && apple9_seen_schedule(ra.ctx, seen, cost_constraints,
                                    copy_register, uniform_remat)) {
      *reason = apple9_duplicate_candidate_reason;
      goto cleanup;
   }
   if (debug_get_option_cached_agx_apple9_trace_shared_ra() != NULL)
      agx_print_shader(ra.ctx, stderr);
   if (!agx_ra(ra.ctx)) {
      *reason = ra.ctx->ra_error;
      goto cleanup;
   }
   if (ra.ctx->scratch_size_B > AGX_APPLE9_MAX_SCRATCH_BYTES) {
      *reason =
         "Apple9 scratch frame exceeds the supported per-invocation size";
      goto cleanup;
   }
   if (!copy_register && !ra.ctx->has_spill_pcopy_reserved &&
       apple9_needs_copy_register(ra.ctx)) {
      *reason = apple9_copy_register_reason;
      goto cleanup;
   }
   if (!apple9_init_physical(&out))
      goto cleanup;
   /* Spills reserve r0..r4 (copy, index and memory-swap temporaries); the
    * high-bank constant bridge reserves r0..r1. */
   {
      unsigned reserved = ra.ctx->has_spill_pcopy_reserved ? 5
                          : copy_register                ? 2
                                                         : 0;
      for (unsigned r = 0; r < reserved; ++r)
         out.reserved_gprs[r] = true;
   }
   unsigned count = program->last_block->index + 1;
   struct agx_apple9_block **blocks =
      rzalloc_array(ra.ctx, struct agx_apple9_block *, count);
   for (struct agx_apple9_block *block = program->blocks; block;
        block = block->next) {
      blocks[block->index] = agx_apple9_block_create(&out);
      if (!blocks[block->index])
         goto cleanup;
      /* Physical scheduling follows the logical CFG's loop edges too. */
      blocks[block->index]->nir = block->nir;
      blocks[block->index]->nir_exit = block->nir_exit;
   }
   /* Instruction indices grouped by block, in program order, for blocks
    * emitted directly from VIR. */
   unsigned *block_first = rzalloc_array(ra.ctx, unsigned, count + 1);
   unsigned *block_order =
      ralloc_array(ra.ctx, unsigned, MAX2(program->instruction_count, 1));
   if (!block_first || !block_order)
      goto cleanup;
   for (unsigned i = 0; i < program->instruction_count; ++i)
      ++block_first[agx_apple9_instr_block(program->instructions[i])->index + 1];
   for (unsigned b = 0; b < count; ++b)
      block_first[b + 1] += block_first[b];
   {
      unsigned *fill = rzalloc_array(ra.ctx, unsigned, count);
      if (!fill)
         goto cleanup;
      for (unsigned i = 0; i < program->instruction_count; ++i) {
         unsigned b = agx_apple9_instr_block(program->instructions[i])->index;
         block_order[block_first[b] + fill[b]++] = i;
      }
   }
   for (struct agx_apple9_block *block = program->blocks; block;
        block = block->next) {
      agx_apple9_block_begin(&out, blocks[block->index]);
      agx_block *common = apple9_shared_block(&ra, block);
      if (!common) {
         for (unsigned k = block_first[block->index];
              k < block_first[block->index + 1]; ++k) {
            const unsigned i = block_order[k];
            struct agx_apple9_vir_instr ins = *program->instructions[i];
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
            if (!apple9_physical_move(&out, I->dest[0], I->src[0],
                                      ra.ctx->spill_base_B / 4))
               goto cleanup;
            break;
         case AGX_OPCODE_MOV_IMM:
            if (!apple9_physical_immediate(&out, I->dest[0].value / 2, I->imm))
               goto cleanup;
            break;
         case AGX_OPCODE_SWAP: {
            /* XOR swap needs no temporary register. */
            unsigned a = I->src[0].value / 2, b = I->src[1].value / 2;
            if (!apple9_physical_xor(&out, a, a, b) ||
                !apple9_physical_xor(&out, b, b, a) ||
                !apple9_physical_xor(&out, a, a, b))
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
   apple9_remove_empty_else(&out);
   if (!apple9_schedule_physical(&out, reason))
      goto cleanup;
   unsigned spill_bytes = ALIGN_POT(ra.ctx->scratch_size_B, 16);
   out.private_size = program->private_size;
   out.spill_slots = (spill_bytes - ra.ctx->spill_base_B) / 4;
   /* The atomic reservation only protects spills or private arrays. */
   if (!out.spill_slots && !out.private_size)
      spill_bytes = 0;
   out.scratch_size = spill_bytes + ALIGN_POT(out.private_size, 16);
   if (out.scratch_size > AGX_APPLE9_MAX_SCRATCH_BYTES) {
      *reason = "Apple9 private arrays and spills exceed the scratch frame";
      goto cleanup;
   }
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
   if (remat_used)
      *remat_used = ra.ctx->uniform_remat_used;
   if (rejected_increase)
      *rejected_increase = ra.ctx->schedule_rejected_increase;
   agx_apple9_vir_finish(&out);
   ralloc_free(ra.ctx);
   return success;
}

struct apple9_allocation_cost {
   uint64_t spills, instructions, bytes, dependency_gaps;
   unsigned registers, peak_registers;
   bool has_copies;
};

static struct apple9_allocation_cost
apple9_allocation_cost(const struct agx_apple9_vir_program *p)
{
   struct apple9_allocation_cost cost = {
      .registers = p->max_phys_gpr + 1,
      .peak_registers = MAX2(p->max_phys_gpr + 1, p->peak_live_gprs),
   };
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

/* A latency-scheduled candidate trades issued instructions for exposed
 * long-latency dependency slots. Spills still dominate. Apple9's occupancy
 * curve is unmeasured, so the established halfreg granules stand in for it:
 * never accept a candidate that drops into a lower tier or adds registers up
 * to its boundary. Within a tier, count
 * a modelled stall slot at a quarter of an issued instruction, since other
 * resident threads hide part of it. */
static bool
apple9_latency_cheaper(struct apple9_allocation_cost a,
                       struct apple9_allocation_cost b)
{
   if (a.spills != b.spills)
      return a.spills < b.spills;
   struct agx_occupancy tier_a = agx_occupancy_for_register_count(2 * a.peak_registers);
   struct agx_occupancy tier_b = agx_occupancy_for_register_count(2 * b.peak_registers);
   if (tier_a.max_threads != tier_b.max_threads)
      return tier_a.max_threads > tier_b.max_threads;
   /* The granules are a stand-in and may be off by one on Apple9. Do not let
    * added registers reach the boundary of the tier the program occupies. */
   if (a.peak_registers > b.peak_registers &&
       2 * a.peak_registers >= tier_b.max_registers)
      return false;
   return 4 * a.instructions + a.dependency_gaps <
          4 * b.instructions + b.dependency_gaps;
}

/* Draw-time compiles may run independent allocation candidates on a helper
 * thread (at most one extra), so the recording thread waits for about four
 * allocations instead of six on spilling programs. Each job deduplicates
 * against a private copy of the schedules seen so far; their new
 * fingerprints are merged in the serial order afterwards, and every result is
 * compared in that order, so the outcome equals the serial search. */
static _Thread_local bool apple9_parallel_ra;

void
agx_apple9_set_parallel_ra(bool enable)
{
   apple9_parallel_ra = enable;
}

static bool
apple9_parallel_ra_enabled(void)
{
   static int forced = -1;
   if (unlikely(forced < 0))
      forced = debug_get_option_cached_agx_apple9_parallel_ra() ? atoi(debug_get_option_cached_agx_apple9_parallel_ra()) : 2;
   return forced == 2 ? apple9_parallel_ra : forced != 0;
}

struct apple9_candidate_job {
   struct agx_apple9_vir_program *program;
   nir_shader *nir;
   bool cost_constraints, latency, copy_register, uniform_remat, retry_copy;
   unsigned slack;
   /* Results. */
   bool valid, ok, remat_used;
   unsigned increase;
   const char *reason;
   struct agx_apple9_vir_program result;
   struct util_dynarray seen;
   unsigned seen_start;
};

static int
apple9_candidate_job_run(void *data)
{
   struct apple9_candidate_job *job = data;
   apple9_in_candidate_job = true;
   agx_apple9_vir_init(&job->result);
   job->reason = NULL;
   job->increase = UINT_MAX;
   job->remat_used = false;
   job->ok = apple9_allocate_candidate(
      job->program, job->nir, job->cost_constraints, job->latency, job->slack,
      job->copy_register, job->uniform_remat, &job->result, &job->reason,
      &job->remat_used, &job->increase, &job->seen);
   if (!job->ok && job->retry_copy && !job->copy_register &&
       job->reason == apple9_copy_register_reason) {
      agx_apple9_vir_finish(&job->result);
      agx_apple9_vir_init(&job->result);
      job->ok = apple9_allocate_candidate(
         job->program, job->nir, job->cost_constraints, job->latency, job->slack,
         true, job->uniform_remat, &job->result, &job->reason, NULL, NULL,
         &job->seen);
   }
   apple9_in_candidate_job = false;
   return 0;
}

static void
apple9_candidate_job_init(struct apple9_candidate_job *job,
                          struct agx_apple9_vir_program *program, nir_shader *nir,
                          const struct util_dynarray *seen, bool cost_constraints,
                          bool latency, unsigned slack, bool copy_register,
                          bool uniform_remat, bool retry_copy)
{
   *job = (struct apple9_candidate_job){
      .program = program, .nir = nir, .cost_constraints = cost_constraints,
      .latency = latency, .slack = slack, .copy_register = copy_register,
      .uniform_remat = uniform_remat, .retry_copy = retry_copy, .valid = true,
   };
   util_dynarray_init(&job->seen, NULL);
   util_dynarray_append_array(&job->seen, uint8_t, seen->data, seen->size);
   job->seen_start = seen->size;
}

/* Run two jobs, the second on a helper thread when possible, and merge their
 * new schedule fingerprints into seen in serial order. */
static void
apple9_candidate_jobs_run(struct apple9_candidate_job jobs[2],
                          struct util_dynarray *seen)
{
   thrd_t thread;
   bool threaded = thrd_create(&thread, apple9_candidate_job_run, &jobs[1]) ==
                   thrd_success;
   apple9_candidate_job_run(&jobs[0]);
   if (threaded)
      thrd_join(thread, NULL);
   else
      apple9_candidate_job_run(&jobs[1]);
   const uint8_t *first = (const uint8_t *)jobs[0].seen.data + jobs[0].seen_start;
   const unsigned first_size = jobs[0].seen.size - jobs[0].seen_start;
   const uint8_t *second = (const uint8_t *)jobs[1].seen.data + jobs[1].seen_start;
   const unsigned second_size = jobs[1].seen.size - jobs[1].seen_start;
   /* The second candidate repeats the first's schedule: in serial order it
    * would have been rejected as a duplicate. */
   for (unsigned a = 0; jobs[1].ok && a < second_size; a += BLAKE3_OUT_LEN) {
      for (unsigned b = 0; b < first_size; b += BLAKE3_OUT_LEN) {
         if (!memcmp(second + a, first + b, BLAKE3_OUT_LEN)) {
            agx_apple9_vir_finish(&jobs[1].result);
            agx_apple9_vir_init(&jobs[1].result);
            jobs[1].ok = false;
            jobs[1].reason = apple9_duplicate_candidate_reason;
            break;
         }
      }
   }
   util_dynarray_append_array(seen, uint8_t, first, first_size);
   util_dynarray_append_array(seen, uint8_t, second, second_size);
   util_dynarray_fini(&jobs[0].seen);
   util_dynarray_fini(&jobs[1].seen);
}

/* Take a prefetched candidate matching these parameters, or allocate it. */
static bool
apple9_take_candidate(struct apple9_candidate_job *prefetched, unsigned count,
                      struct agx_apple9_vir_program *program, nir_shader *nir,
                      bool cost_constraints, bool latency, unsigned slack,
                      bool copy_register, bool uniform_remat,
                      struct agx_apple9_vir_program *result, const char **reason,
                      bool *remat_used, unsigned *increase,
                      struct util_dynarray *seen)
{
   for (unsigned i = 0; i < count; ++i) {
      struct apple9_candidate_job *job = &prefetched[i];
      if (!job->valid || job->cost_constraints != cost_constraints ||
          job->latency != latency || job->slack != slack ||
          job->copy_register != copy_register ||
          job->uniform_remat != uniform_remat)
         continue;
      job->valid = false;
      agx_apple9_vir_finish(result);
      *result = job->result;
      *reason = job->reason;
      if (remat_used)
         *remat_used = job->remat_used;
      if (increase)
         *increase = job->increase;
      return job->ok;
   }
   return apple9_allocate_candidate(program, nir, cost_constraints, latency,
                                    slack, copy_register, uniform_remat, result,
                                    reason, remat_used, increase, seen);
}

bool
agx_apple9_allocate_shared(struct agx_apple9_vir_program *program,
                           nir_shader *nir, const char **reason)
{
   /* Reject unsafe U0 repetition before allocation or scheduling can insert
    * waits that would disguise it as an ordered, but still unqualified, form. */
   agx_apple9_vir_reindex(program);
   if (!agx_apple9_validate_mesh_control(program, reason))
      return false;

   /* Apple9 bypasses agx_compile_shader_nir, which initializes these flags
    * for the older backend. The shared allocator still consumes them. */
   agx_compiler_debug = agx_get_compiler_debug();

   if (!agx_apple9_allocate_publications(program, reason))
      return false;
   struct agx_apple9_vir_program best, alternative;
   agx_apple9_vir_init(&best);
   agx_apple9_vir_init(&alternative);
   /* Prefer an allocation without reserved registers; constants placed above
    * r63 need the copy register, so retry with it reserved. */
   bool copy_register = false;
   struct util_dynarray seen;
   util_dynarray_init(&seen, NULL);
   /* Whether the successful rematerializing candidate for [latency][banks]
    * (slack 0, current copy_register) actually rematerialized a uniform. If
    * not, the non-rematerializing allocation is the same computation. */
   bool remat_known[2][2] = {{false}}, remat_used[2][2] = {{false}};
   if (!apple9_allocate_candidate(program, nir, false, false, 0, false, true, &best,
                                  reason, &remat_used[0][0], NULL, &seen)) {
      if (*reason != apple9_copy_register_reason) {
         util_dynarray_fini(&seen);
         return false;
      }
      agx_apple9_vir_finish(&best);
      agx_apple9_vir_init(&best);
      copy_register = true;
      if (!apple9_allocate_candidate(program, nir, false, false, 0, true, true, &best,
                                     reason, &remat_used[0][0], NULL, &seen)) {
         util_dynarray_fini(&seen);
         return false;
      }
   }
   remat_known[0][0] = true;
   /* A spill probe needs only the first allocation. */
   if (agx_apple9_ra_probe) {
      util_dynarray_fini(&seen);
      agx_apple9_vir_finish(&alternative);
      agx_apple9_vir_finish(program);
      *program = best;
      for (struct agx_apple9_block *block = program->allocated_blocks; block;
           block = block->allocated_next)
         block->program = program;
      return true;
   }

   struct apple9_allocation_cost cost = apple9_allocation_cost(&best);
   /* Bank-cost allocations pay off when the plain allocation spills: they
    * then remove copies and spill traffic. Without spills they almost never
    * beat it (ELDEN RING's 758 shaders: 16 wins in 1,889 allocations, 0.2%
    * of their instructions) yet cost a full allocation each. Conversely,
    * once the plain allocation spills, the later alternatives without bank
    * costs never beat the best bank-cost one there (0 spills, 24 of 107,659
    * instructions), so each family of alternatives is tried only where it
    * can win. */
   const bool bank_candidates = cost.spills != 0;
   unsigned long_dependencies = 0;
   for (unsigned i = 0; i < program->instruction_count; ++i)
      long_dependencies += apple9_dependency_delay(program->instructions[i]) > 1;
   struct apple9_candidate_job prefetched[2] = {0};
   const bool parallel = apple9_parallel_ra_enabled();
   if (parallel && bank_candidates && long_dependencies > 1) {
      apple9_candidate_job_init(&prefetched[0], program, nir, &seen, true, false,
                                0, copy_register, true, false);
      apple9_candidate_job_init(&prefetched[1], program, nir, &seen, true, true,
                                0, copy_register, true, false);
      apple9_candidate_jobs_run(prefetched, &seen);
   }
   if (bank_candidates) {
      /* Bank hints can conflict with later tuples and phi reconciliation.
       * Compare completed allocations so those costs are paid by the policy
       * which caused them, including rematerialization and spill lowering. */
      const char *unused_reason = NULL;
      bool built = apple9_take_candidate(prefetched, 2, program, nir, true, false, 0,
                                         copy_register, true, &alternative,
                                         &unused_reason, &remat_used[0][1], NULL, &seen);
      remat_known[0][1] = built;
      if (built &&
          apple9_allocation_cheaper(apple9_allocation_cost(&alternative), cost))
         SWAP(best, alternative);
   }
   agx_apple9_vir_finish(&alternative);
   cost = apple9_allocation_cost(&best);
   if (long_dependencies > 1) {
      /* Overlapping independent long-latency operations needs registers;
       * also offer schedules that may raise a block's peak pressure by eight
       * GPRs. Spills and the occupancy tier still decide between them. */
      static const unsigned slacks[] = {0, 16};
      /* A larger slack changes the schedule only if a smaller one rejected a
       * reorder whose increase it would now admit; otherwise its allocation
       * repeats the previous one exactly. */
      unsigned rejected[2] = {UINT_MAX, UINT_MAX};
      for (unsigned c = 0; c < 2 * ARRAY_SIZE(slacks); ++c) {
         const unsigned banks = c & 1, slack = slacks[c >> 1];
         if (!!banks != bank_candidates)
            continue;
         if (slack && rejected[banks] <= slacks[(c >> 1) - 1])
            continue;
         agx_apple9_vir_init(&alternative);
         const char *unused_reason = NULL;
         bool used = false;
         unsigned increase = UINT_MAX;
         bool built = apple9_take_candidate(prefetched, 2, program, nir, banks, true, slack,
                                        copy_register, true, &alternative,
                                        &unused_reason, &used, &increase, &seen);
         rejected[banks] = built ? increase : UINT_MAX;
         if (!slack) {
            remat_known[1][banks] = built;
            remat_used[1][banks] = used;
         }
         if (built &&
             apple9_latency_cheaper(apple9_allocation_cost(&alternative), cost)) {
            SWAP(best, alternative);
            cost = apple9_allocation_cost(&best);
         }
         agx_apple9_vir_finish(&alternative);
      }
   }
   /* Rebuilding uniform expressions can need temporary registers. Compare
    * the fully allocated alternatives when scratch remains, so reserving those
    * temporaries never wins merely by saving cheap moves while adding spills.
    * Keep invariant-store placement in both policies. */
   /* Unconsumed prefetches (a candidate the serial rules then skipped). */
   for (unsigned i = 0; i < 2; ++i) {
      if (prefetched[i].valid)
         agx_apple9_vir_finish(&prefetched[i].result);
      prefetched[i].valid = false;
   }
   /* Uniform-rematerialization-off retries: the latency-scheduled one has
    * never beaten the others (0 of 216 spilling allocations over ER, its
    * draw-time fragments, CS2, Lethal Company, Willems, CTS and robust
    * corpora), so only the plain-scheduled retry runs. */
   const unsigned remat_off_latency = 0;
   if (cost.spills && parallel) {
      unsigned n = 0;
      for (unsigned latency = 0; latency <= remat_off_latency; ++latency) {
         for (unsigned banks = 0; banks < 2; ++banks) {
            if ((bank_candidates && !banks) ||
                (remat_known[latency][banks] && !remat_used[latency][banks]))
               continue;
            if (n < 2)
               apple9_candidate_job_init(&prefetched[n], program, nir, &seen, banks,
                                         latency, 0, copy_register, false, true);
            ++n;
         }
      }
      if (n == 2)
         apple9_candidate_jobs_run(prefetched, &seen);
      else
         for (unsigned i = 0; i < 2; ++i) {
            if (prefetched[i].valid)
               util_dynarray_fini(&prefetched[i].seen);
            prefetched[i].valid = false;
         }
   }
   if (cost.spills) {
      for (unsigned latency = 0; latency <= remat_off_latency; ++latency) {
         for (unsigned banks = 0; banks < 2; ++banks) {
            if (bank_candidates && !banks)
               continue;
            if (remat_known[latency][banks] && !remat_used[latency][banks])
               continue;
            agx_apple9_vir_init(&alternative);
            const char *unused_reason = NULL;
            bool ok = apple9_take_candidate(prefetched, 2,
               program, nir, banks, latency, 0, copy_register, false,
               &alternative, &unused_reason, NULL, NULL, &seen);
            if (!ok && !copy_register &&
                unused_reason == apple9_copy_register_reason) {
               agx_apple9_vir_finish(&alternative);
               agx_apple9_vir_init(&alternative);
               ok = apple9_allocate_candidate(
                  program, nir, banks, latency, 0, true, false,
                  &alternative, &unused_reason, NULL, NULL, &seen);
            }
            if (ok && apple9_allocation_cheaper(
                         apple9_allocation_cost(&alternative), cost)) {
               SWAP(best, alternative);
               cost = apple9_allocation_cost(&best);
            }
            agx_apple9_vir_finish(&alternative);
         }
      }
   }
   for (unsigned i = 0; i < 2; ++i) {
      if (prefetched[i].valid)
         agx_apple9_vir_finish(&prefetched[i].result);
   }
   util_dynarray_fini(&seen);
   agx_apple9_vir_finish(program);
   *program = best;
   for (struct agx_apple9_block *block = program->allocated_blocks; block;
        block = block->allocated_next)
      block->program = program;
   return true;
}
