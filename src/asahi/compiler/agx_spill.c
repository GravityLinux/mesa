/*
 * Copyright 2023-2024 Alyssa Rosenzweig
 * Copyright 2023-2024 Valve Corporation
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "asahi/debug.h"
#include "util/bitset.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/sparse_bitset.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "util/u_qsort.h"
#include "agx_builder.h"
#include "agx_apple9_ir.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/*
 * An implementation of "Register Spilling and Live-Range Splitting for SSA-Form
 * Programs" by Braun and Hack.
 */

/*
 * Next-use distances are logically in ℤ ∪ {∞}. Modeled as saturating uint32 and
 * referred to as dist_t.
 *
 * next_uses represents a next-use map. This is a sparse data structure mapping
 * variable names to next-use dist_t's. Variables with no later use (infinite
 * next-use distance) are not stored explicitly, making the time/space
 * requirements O(live variables). This is important for performance and memory
 * usage on big shaders with many blocks.
 *
 * For now, next_uses is backed by a Mesa hash table, but it could be optimized
 * to something more specialized in the future.
 */
#define DIST_INFINITY (UINT32_MAX)
typedef uint32_t dist_t;

static dist_t
dist_sum(dist_t A, dist_t B)
{
   return (A + B < A) ? DIST_INFINITY : (A + B);
}

/*
 * Braun and Hack weight next-use distances that are only reached by leaving a
 * loop with a large constant, so inside a loop a value used only after it
 * looks far away. Without this, values the loop never touches compete with
 * its working set, and the loop spills and reloads its own values on every
 * iteration, often in a single branch arm, which makes the register state
 * differ at the join and forces parallel copies there.
 */
#define LOOP_EXIT_DIST 100000

/* Structured control flow places a loop's blocks contiguously in source
 * order, from the header to its last back-edge predecessor. loop_end[h] is
 * that predecessor's index for each loop header h, and 0 otherwise.
 */
static unsigned *
compute_loop_ends(agx_context *ctx, void *memctx)
{
   unsigned *loop_end = rzalloc_array(memctx, unsigned, ctx->num_blocks);
   agx_foreach_block(ctx, block) {
      if (!block->loop_header)
         continue;
      agx_foreach_predecessor(block, pred) {
         if ((*pred)->index >= block->index)
            loop_end[block->index] = MAX2(loop_end[block->index], (*pred)->index);
      }
   }
   return loop_end;
}

static bool
loop_contains(const unsigned *loop_end, unsigned header, unsigned block)
{
   return header <= block && block <= loop_end[header];
}

/* Number of loops the edge pred -> succ leaves. */
static unsigned
loops_exited(agx_context *ctx, const unsigned *loop_end, agx_block *pred,
             agx_block *succ)
{
   unsigned exits = 0;
   for (unsigned h = 0; h <= pred->index && h < ctx->num_blocks; ++h) {
      if (loop_end[h] && loop_contains(loop_end, h, pred->index) &&
          !loop_contains(loop_end, h, succ->index))
         ++exits;
   }
   return exits;
}


static bool
spill_trace_loops(void)
{
   return agx_apple9_debug_enabled(AGX_A9_DBG_SPILL_LOOPS);
}

struct next_uses {
   struct hash_table_u64 *ht;
};

static void
init_next_uses(struct next_uses *nu, void *memctx)
{
   nu->ht = _mesa_hash_table_u64_create(memctx);
}

static void
destroy_next_uses(struct next_uses *nu)
{
   _mesa_hash_table_u64_destroy(nu->ht);
}

static void
clear_next_uses(struct next_uses *nu)
{
   _mesa_hash_table_u64_clear(nu->ht);
}

static void
copy_next_uses(struct next_uses *nu, const struct next_uses *from)
{
   clear_next_uses(nu);

   hash_table_u64_foreach(from->ht, use) {
      _mesa_hash_table_u64_insert(nu->ht, use.key, use.data);
   }
}

static void
set_next_use(struct next_uses *nu, unsigned node, dist_t dist)
{
   if (dist == DIST_INFINITY) {
      _mesa_hash_table_u64_remove(nu->ht, node);
   } else {
      uintptr_t as_ptr = (uintptr_t)(dist + 1);
      assert(as_ptr != 0 && "non-NULL");

      _mesa_hash_table_u64_insert(nu->ht, node, (void *)as_ptr);
   }
}

static dist_t
search_next_uses(const struct next_uses *nu, unsigned node)
{
   void *ent = _mesa_hash_table_u64_search(nu->ht, node);
   if (!ent)
      return DIST_INFINITY;

   uintptr_t raw = (uintptr_t)ent;
   return raw - 1;
}

#define foreach_next_use(nu, node, dist)                                       \
   hash_table_u64_foreach((nu)->ht, use_)                                      \
      for (uint32_t _terminator = 1, node = use_.key,                          \
                    UNUSED dist = ((uintptr_t)use_.data) - 1;                  \
           _terminator != 0; _terminator = 0)

/*
 * Calculate the minimum of two next-use sets. Values absent from one of the
 * underlying sets are infinity so do not contribute to the minimum, instead
 * acting like a set union.
 */
static bool
minimum_next_uses(struct next_uses *nu, const struct next_uses *from)
{
   bool progress = false;

   foreach_next_use(from, node, from_dist) {
      dist_t nu_dist = search_next_uses(nu, node);

      if (from_dist < nu_dist) {
         set_next_use(nu, node, from_dist);
         progress = true;
      }
   }

   return progress;
}

static uint32_t
instr_cycles(const agx_instr *I)
{
   return 1;
}

struct spill_block {
   /* Set of values available in the register file at the end */
   unsigned W_exit[AGX_NUM_REGS];
   unsigned nW_exit;

   unsigned W_entry[AGX_NUM_REGS];
   unsigned nW_entry;

   /* Set of live-out spilled values at the end of the block */
   unsigned *S_exit;
   unsigned nS_exit;

   unsigned *S_entry;
   unsigned nS_entry;

   /* Estimate */
   uint32_t cycles;

   /* Next-use maps at the start/end of the block */
   struct next_uses next_use_in;
   struct next_uses next_use_out;
};

struct spill_ctx {
   void *memctx;
   agx_context *shader;
   agx_block *block;

   /* Set of values currently available in the register file */
   BITSET_WORD *W;

   /* |W| = Current register pressure */
   unsigned nW;

   /* Local IPs of next-use */
   dist_t *next_uses;

   /* Current local IP relative to the start of the block */
   uint32_t ip;

   /* Set of live values that have been spilled. Contrary to the paper, this
    * is not a subset of W: the definition in the paper is bogus.
    */
   BITSET_WORD *S;

   /* Widths of vectors */
   uint8_t *channels;
   enum agx_size *size;

   /* Mapping of rematerializable values to their definitions, or NULL for nodes
    * that are not materializable.
    */
   agx_instr **remat;

   /* Maximum register pressure allowed */
   unsigned k;

   /* Number of variables */
   unsigned n;

   /* Information on blocks indexed in source order */
   struct spill_block *blocks;

   /* Base memory index reserved for spilled indices */
   unsigned spill_base;

   /* If true, we cannot use scratch and must only rematerialize. */
   bool remat_only;

   /* Loop spans (see compute_loop_ends). */
   const unsigned *loop_end;

   /* Zeroed per-variable scratch for compute_w_entry, which restores it. A
    * fresh array per block would make each block cost O(n). */
   unsigned *freq;

   /* Per-variable scratch for listing a set's members. */
   unsigned *members;
};

static inline struct spill_block *
spill_block(struct spill_ctx *ctx, agx_block *block)
{
   return &ctx->blocks[block->index];
}

/* Calculate the register demand of a node. This is rounded up to a power-of-two
 * to match the equivalent calculations in RA.
 */
static unsigned
node_size(struct spill_ctx *ctx, unsigned node)
{
   return util_next_power_of_two(ctx->channels[node]) *
          agx_size_align_16(ctx->size[node]);
}

/*
 * Map a control flow edge to a block. Assumes no critical edges.
 */
static agx_block *
agx_edge_to_block(agx_block *pred, agx_block *succ)
{
   /* End of predecessor is unique if there's a single successor */
   if (agx_num_successors(pred) == 1)
      return pred;

   /* The predecessor has multiple successors, meaning this is not the only
    * edge leaving the predecessor. Therefore, it is the only edge entering
    * the successor (otherwise the edge would be critical), so the start of
    * the successor is unique.
    */
   assert(agx_num_predecessors(succ) == 1 && "critical edge detected");
   return succ;
}

/*
 * Get a cursor to insert along a control flow edge: either at the start of the
 * successor or the end of the predecessor. This relies on the control flow
 * graph having no critical edges.
 */
static agx_cursor
agx_along_edge(agx_block *pred, agx_block *succ)
{
   agx_block *to = agx_edge_to_block(pred, succ);

   if (to == pred)
      return agx_after_block_logical(pred);
   else
      return agx_before_block(succ);
}

static inline agx_index
agx_index_as_mem(agx_index idx, unsigned mem_base)
{
   assert(idx.type == AGX_INDEX_NORMAL);
   assert(!idx.memory);
   idx.memory = true;
   idx.value = mem_base + idx.value;
   return idx;
}

static unsigned
chase_mem_index(agx_index ref, unsigned mem_base)
{
   assert(ref.type == AGX_INDEX_NORMAL);
   return ref.memory ? (ref.value - mem_base) : ref.value;
}

static agx_index
reconstruct_index(struct spill_ctx *ctx, unsigned node)
{
   return agx_get_vec_index(node, ctx->size[node], ctx->channels[node]);
}

static bool
can_remat(agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_MOV_IMM:
   case AGX_OPCODE_GET_SR:
      return true;
   default:
      return false;
   }
}

/* Rebuild short scalar expressions whose complete input graph consists of
 * constants and immutable uniforms. No GPR input is kept live for this: every
 * dependency is reconstructed at the use. Limit the total instruction count,
 * including repeated dependencies, to avoid expanding large expression trees.
 */
static bool
apple9_remat_alu(const agx_instr *I)
{
   if (I->op != AGX_OPCODE_APPLE9_PURE || !I->apple9 || I->nr_dests != 1 ||
       I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
       !agx_apple9_instr_is_pure_alu(I->apple9))
      return false;

   switch (I->apple9->op) {
   case AGX_APPLE9_VIR_IOR_UNIFORM:
   case AGX_APPLE9_VIR_IADD:
   case AGX_APPLE9_VIR_ISUB:
   case AGX_APPLE9_VIR_IAND:
   case AGX_APPLE9_VIR_IOR:
   case AGX_APPLE9_VIR_IXOR:
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHL:
   case AGX_APPLE9_VIR_FADD:
   case AGX_APPLE9_VIR_FSUB:
   case AGX_APPLE9_VIR_FMUL:
   case AGX_APPLE9_VIR_FADD_IMM:
   case AGX_APPLE9_VIR_FMUL_IMM:
   case AGX_APPLE9_VIR_SELECT:
   case AGX_APPLE9_VIR_BIT_COUNT:
   case AGX_APPLE9_VIR_BIT_REVERSE:
   case AGX_APPLE9_VIR_UFIND_MSB:
   case AGX_APPLE9_VIR_U2F32:
   case AGX_APPLE9_VIR_I2F32:
   case AGX_APPLE9_VIR_F2I32:
   case AGX_APPLE9_VIR_F2U32:
      return true;
   default:
      return false;
   }
}

static unsigned
find_apple9_remat(agx_context *ctx, agx_instr **remat, void *memctx,
                  unsigned max_extra)
{
   if (ctx->ra_target.disable_uniform_rematerialization)
      return 0;
   /* Preamble writes make uniforms mutable. Treat that entire shader part
    * conservatively; the main shader's file is immutable. */
   agx_foreach_instr_global(ctx, I) {
      if ((I->op == AGX_OPCODE_APPLE9 || I->op == AGX_OPCODE_APPLE9_PURE) &&
          I->apple9 && I->apple9->op == AGX_APPLE9_VIR_STORE_UNIFORM)
         return 0;
   }

   uint8_t *cost = rzalloc_array(memctx, uint8_t, ctx->alloc);
   unsigned *peak = rzalloc_array(memctx, unsigned, ctx->alloc);
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV_IMM) {
         cost[I->dest[0].value] = 1;
         peak[I->dest[0].value] =
            util_next_power_of_two(agx_index_size_16(I->dest[0]));
      }
   }

   unsigned extra = 0;
   bool progress;
   do {
      progress = false;
      agx_foreach_instr_global(ctx, I) {
         if (!apple9_remat_alu(I) || cost[I->dest[0].value])
            continue;
         unsigned total = 1, held = 0, demand = 0;
         agx_foreach_src(I, s) {
            if (I->src[s].type != AGX_INDEX_NORMAL ||
                !cost[I->src[s].value]) {
               total = 5;
               break;
            }
            total += cost[I->src[s].value];
            demand = MAX2(demand, held + peak[I->src[s].value]);
            held += util_next_power_of_two(agx_index_size_16(I->src[s]));
         }
         if (total <= 4) {
            unsigned width = agx_index_size_16(I->dest[0]);
            demand = MAX2(demand, held + width);
            if (demand - width > max_extra)
               continue;
            peak[I->dest[0].value] = demand;
            extra = MAX2(extra, demand - width);
            cost[I->dest[0].value] = total;
            remat[I->dest[0].value] = I;
            ctx->uniform_remat_used = true;
            progress = true;
         }
      }
   } while (progress);
   return extra;
}

static agx_instr *
remat_to(agx_builder *b, agx_index dst, struct spill_ctx *ctx, unsigned node)
{
   agx_instr *I = ctx->remat[node];
   assert(can_remat(I) || apple9_remat_alu(I));

   switch (I->op) {
   case AGX_OPCODE_MOV_IMM:
      return agx_mov_imm_to(b, dst, I->imm);
   case AGX_OPCODE_GET_SR:
      return agx_get_sr_to(b, dst, I->sr);
   case AGX_OPCODE_APPLE9_PURE: {
      agx_instr *copy = agx_alloc_instr(b, I->op, 1, I->nr_srcs);
      copy->dest[0] = dst;
      copy->apple9 = I->apple9;
      copy->reg_constraints = I->reg_constraints;
      agx_foreach_src(I, s) {
         agx_index tmp = agx_temp_like(b->shader, I->src[s]);
         remat_to(b, tmp, ctx, I->src[s].value);
         copy->src[s] = tmp;
      }
      agx_builder_insert(&b->cursor, copy);
      return copy;
   }
   default:
      UNREACHABLE("invalid remat");
   }
}

static void
insert_spill(agx_builder *b, struct spill_ctx *ctx, unsigned node)
{
   if (!ctx->remat[node]) {
      assert(!ctx->remat_only && "must have remat something");

      agx_index idx = reconstruct_index(ctx, node);
      agx_mov_to(b, agx_index_as_mem(idx, ctx->spill_base), idx);

      /* We only need the extra registers reserved if we actually spilled
       * instead of just remat.
       */
      b->shader->has_spill_pcopy_reserved = true;
   }
}

static void
insert_reload(struct spill_ctx *ctx, agx_block *block, agx_cursor cursor,
              unsigned node)
{
   agx_builder b = agx_init_builder(ctx->shader, cursor);
   agx_index idx = reconstruct_index(ctx, node);

   /* Reloading breaks SSA, but agx_repair_ssa will repair */
   if (ctx->remat[node]) {
      remat_to(&b, idx, ctx, node);
   } else {
      agx_mov_to(&b, idx, agx_index_as_mem(idx, ctx->spill_base));
   }
}

/* Insert into the register file */
static void
insert_W(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);
   assert(!BITSET_TEST(ctx->W, v));

   BITSET_SET(ctx->W, v);
   ctx->nW += node_size(ctx, v);
}

/* Remove from the register file */
static void
remove_W(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);
   assert(BITSET_TEST(ctx->W, v));

   BITSET_CLEAR(ctx->W, v);
   ctx->nW -= node_size(ctx, v);
}

static void
remove_W_if_present(struct spill_ctx *ctx, unsigned v)
{
   assert(v < ctx->n);

   if (BITSET_TEST(ctx->W, v))
      remove_W(ctx, v);
}

/* The members of a set, in increasing order, as an exactly sized array. */
static unsigned *
bitset_members(struct spill_ctx *ctx, const BITSET_WORD *set, unsigned *count)
{
   unsigned n = 0;
   int i;
   BITSET_FOREACH_SET(i, set, ctx->n)
      ctx->members[n++] = i;

   unsigned *out = ralloc_array(ctx->memctx, unsigned, n);
   memcpy(out, ctx->members, n * sizeof(*out));
   *count = n;
   return out;
}

struct candidate {
   unsigned node;
   dist_t dist;
};

static int
cmp_dist(const void *left_, const void *right_, void *ctx_)
{
   struct spill_ctx *ctx = ctx_;
   const struct candidate *left = left_;
   const struct candidate *right = right_;

   /* We assume that rematerializing - even before every instruction - is
    * cheaper than spilling. As long as one of the nodes is rematerializable
    * (with distance > 0), we choose it over spilling. Within a class of nodes
    * (rematerializable or not), compare by next-use-distance.
    */
   bool remat_left = ctx->remat[left->node] != NULL && left->dist > 0;
   bool remat_right = ctx->remat[right->node] != NULL && right->dist > 0;

   if (remat_left != remat_right)
      return remat_left ? 1 : -1;
   else
      return (left->dist > right->dist) - (left->dist < right->dist);
}

/*
 * Limit the register file W to maximum size m by evicting registers.
 */
static ATTRIBUTE_NOINLINE void
limit(struct spill_ctx *ctx, agx_instr *I, unsigned m)
{
   /* Nothing to do if we're already below the limit */
   if (ctx->nW <= m)
      return;

   /* Gather candidates for eviction. Note that next_uses gives IPs whereas
    * cmp_dist expects relative distances. This requires us to subtract ctx->ip
    * to ensure that cmp_dist works properly. Even though logically it shouldn't
    * affect the sorted order, practically this matters for correctness with
    * rematerialization. See the dist=0 test in cmp_dist.
    */
   struct candidate *candidates = alloca(ctx->nW * sizeof(struct candidate));
   unsigned j = 0;

   int i;
   BITSET_FOREACH_SET(i, ctx->W, ctx->n) {
      assert(j < ctx->nW);

      candidates[j++] = (struct candidate){
         .node = i,
         .dist = ctx->next_uses[i] - ctx->ip,
      };
   }

   /* Sort by next-use distance */
   util_qsort_r(candidates, j, sizeof(struct candidate), cmp_dist, ctx);

   /* Evict what doesn't fit */
   unsigned new_weight = 0;

   for (i = 0; i < j; ++i) {
      unsigned v = candidates[i].node;
      unsigned comps = node_size(ctx, v);

      if ((new_weight + comps) <= m) {
         new_weight += comps;
      } else {
         /* Insert a spill if we haven't spilled before and there is
          * another use
          */
         if (!BITSET_TEST(ctx->S, v) && candidates[i].dist < DIST_INFINITY) {
            agx_builder b = agx_init_builder(ctx->shader, agx_before_instr(I));
            insert_spill(&b, ctx, v);
            BITSET_SET(ctx->S, v);
         }

         remove_W(ctx, v);

         /* We keep going in case we can pack in a scalar */
      }
   }
}

/*
 * Insert coupling code on block boundaries. This must ensure:
 *
 *    - anything live-in we expect to have spilled is spilled
 *    - anything live-in we expect to have filled is filled
 *    - phi sources are spilled if the destination is spilled
 *    - phi sources are filled if the destination is not spilled
 *
 * The latter two requirements ensure correct pressure calculations for phis.
 */
static ATTRIBUTE_NOINLINE void
insert_coupling_code(struct spill_ctx *ctx, agx_block *pred, agx_block *succ)
{
   struct spill_block *sp = spill_block(ctx, pred);
   struct spill_block *ss = spill_block(ctx, succ);

   /* These scratch sets are no longer needed by the per-block MIN pass.
    * Index the predecessor state once; repeated linear membership scans and
    * phi scans otherwise become quadratic at large control-flow joins. Both
    * sets are empty here and are emptied again through the same lists on
    * return, so an edge costs its sets' sizes rather than O(n). */
   for (unsigned i = 0; i < sp->nS_exit; ++i)
      BITSET_SET(ctx->S, sp->S_exit[i]);
   for (unsigned i = 0; i < sp->nW_exit; ++i)
      BITSET_SET(ctx->W, sp->W_exit[i]);

   agx_foreach_phi_in_block(succ, I) {
      if (!I->dest[0].memory)
         continue;

      agx_builder b =
         agx_init_builder(ctx->shader, agx_before_function(ctx->shader));

      unsigned s = agx_predecessor_index(succ, pred);

      /* Copy immediate/uniform phi sources to memory variables at the start of
       * the program, where pressure is zero and hence the copy is legal.
       */
      if (I->src[s].type != AGX_INDEX_NORMAL) {
         assert(I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].type == AGX_INDEX_UNIFORM);

         agx_index mem = agx_temp_like(ctx->shader, I->dest[0]);
         assert(mem.memory);

         agx_index gpr = agx_temp_like(ctx->shader, I->dest[0]);
         gpr.memory = false;

         if (I->src[s].type == AGX_INDEX_IMMEDIATE)
            agx_mov_imm_to(&b, gpr, I->src[s].value);
         else
            agx_mov_to(&b, gpr, I->src[s]);

         agx_mov_to(&b, mem, gpr);
         I->src[s] = mem;
         continue;
      }

      if (!BITSET_TEST(ctx->S, I->src[s].value)) {
         /* Spill the phi source. TODO: avoid redundant spills here */
         agx_builder b =
            agx_init_builder(ctx->shader, agx_after_block_logical(pred));

         insert_spill(&b, ctx, I->src[s].value);
      }

      if (ctx->remat[I->src[s].value]) {
         unsigned node = I->src[s].value;
         agx_index idx = reconstruct_index(ctx, node);
         agx_index tmp = agx_temp_like(ctx->shader, idx);

         remat_to(&b, tmp, ctx, node);
         agx_mov_to(&b, agx_index_as_mem(idx, ctx->spill_base), tmp);
      }

      /* Use the spilled version */
      I->src[s] = agx_index_as_mem(I->src[s], ctx->spill_base);
   }

   /* Anything assumed to be spilled at the start of succ must be spilled along
    * all edges.
    */
   agx_foreach_phi_in_block(succ, phi)
      BITSET_SET(ctx->S, chase_mem_index(phi->dest[0], ctx->spill_base));

   for (unsigned i = 0; i < ss->nS_entry; ++i) {
      unsigned v = ss->S_entry[i];

      /* Phi destinations are handled separately. */
      if (BITSET_TEST(ctx->S, v))
         continue;

      agx_builder b = agx_init_builder(ctx->shader, agx_along_edge(pred, succ));
      insert_spill(&b, ctx, v);
   }

   /* Register phi destinations are also defined along the edge. Keep W as
    * the original predecessor set for the final phi-source reloads below. */
   for (unsigned i = 0; i < sp->nS_exit; ++i)
      BITSET_CLEAR(ctx->S, sp->S_exit[i]);
   agx_foreach_phi_in_block(succ, phi)
      BITSET_CLEAR(ctx->S, chase_mem_index(phi->dest[0], ctx->spill_base));
   agx_foreach_phi_in_block(succ, phi) {
      if (phi->dest[0].value < ctx->n)
         BITSET_SET(ctx->S, phi->dest[0].value);
   }

   /* Variables in W at the start of succ must be defined along the edge. */
   for (unsigned i = 0; i < ss->nW_entry; ++i) {
      unsigned node = ss->W_entry[i];
      if (BITSET_TEST(ctx->W, node) || BITSET_TEST(ctx->S, node))
         continue;

      /* Otherwise, inserting a reload defines the variable along the edge */
      agx_block *reload_block = agx_edge_to_block(pred, succ);
      insert_reload(ctx, reload_block, agx_along_edge(pred, succ), node);
   }

   agx_foreach_phi_in_block(succ, I) {
      if (I->dest[0].memory)
         continue;

      unsigned s = agx_predecessor_index(succ, pred);

      /* Treat immediate/uniform phi sources as registers for pressure
       * accounting and phi lowering purposes. Parallel copy lowering can handle
       * a copy from a immediate/uniform to a register, but not from an
       * immediate/uniform directly to memory.
       */
      if (I->src[s].type != AGX_INDEX_NORMAL) {
         assert(I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].type == AGX_INDEX_UNIFORM);

         continue;
      }

      /* Fill the phi source in the predecessor. */
      if (!BITSET_TEST(ctx->W, I->src[s].value)) {
         agx_block *reload_block = agx_edge_to_block(pred, succ);
         insert_reload(ctx, reload_block, agx_along_edge(pred, succ),
                       I->src[s].value);
      }

      /* Leave as-is for the GPR version */
      assert(!I->src[s].memory);
   }

   agx_foreach_phi_in_block(succ, phi) {
      if (phi->dest[0].value < ctx->n)
         BITSET_CLEAR(ctx->S, phi->dest[0].value);
   }
   for (unsigned i = 0; i < sp->nW_exit; ++i)
      BITSET_CLEAR(ctx->W, sp->W_exit[i]);
}

/*
 * Produce an array of next-use IPs relative to the start of the block. This is
 * an array of dist_t scalars, representing the next-use IP of each SSA dest
 * (right-to-left) and SSA source (left-to-right) of each instruction in the
 * block (bottom-to-top). Its size equals the # of SSA sources in the block.
 */
static ATTRIBUTE_NOINLINE void
calculate_local_next_use(struct spill_ctx *ctx, struct util_dynarray *out)
{
   struct spill_block *sb = spill_block(ctx, ctx->block);
   unsigned ip = sb->cycles;

   *out = UTIL_DYNARRAY_INIT;

   struct next_uses nu;
   init_next_uses(&nu, NULL);

   foreach_next_use(&sb->next_use_out, i, dist) {
      set_next_use(&nu, i, dist_sum(ip, dist));
   }

   agx_foreach_instr_in_block_rev(ctx->block, I) {
      ip -= instr_cycles(I);

      if (I->op != AGX_OPCODE_PHI) {
         agx_foreach_ssa_src(I, s) {
            unsigned v = I->src[s].value;

            util_dynarray_append(out, search_next_uses(&nu, v));
            set_next_use(&nu, v, ip);
         }

         agx_foreach_ssa_dest_rev(I, d) {
            unsigned v = I->dest[d].value;

            util_dynarray_append(out, search_next_uses(&nu, v));
         }
      }
   }

   assert(ip == 0 && "cycle counting is consistent");
   destroy_next_uses(&nu);
}

static agx_cursor
agx_before_instr_logical(agx_block *block, agx_instr *I)
{
   if (I->op == AGX_OPCODE_EXPORT) {
      agx_instr *first = agx_first_instr(block);

      while (I != first && I->op == AGX_OPCODE_EXPORT) {
         I = agx_prev_op(I);
      }

      if (I == first && first->op == AGX_OPCODE_EXPORT) {
         return agx_before_block(block);
      } else {
         return agx_after_instr(I);
      }
   } else {
      return agx_before_instr(I);
   }
}

/*
 * Insert spills/fills for a single basic block, following Belady's algorithm.
 * Corresponds to minAlgorithm from the paper.
 */
static ATTRIBUTE_NOINLINE void
min_algorithm(struct spill_ctx *ctx)
{
   struct spill_block *sblock = spill_block(ctx, ctx->block);
   struct util_dynarray local_next_ip;
   calculate_local_next_use(ctx, &local_next_ip);

   /* next_uses gives the distance from the start of the block, so prepopulate
    * with next_use_in.
    */
   foreach_next_use(&sblock->next_use_in, key, dist) {
      assert(key < ctx->n);
      ctx->next_uses[key] = dist;
   }

   dist_t *next_ips = util_dynarray_element(&local_next_ip, dist_t, 0);
   unsigned next_use_cursor =
      util_dynarray_num_elements(&local_next_ip, dist_t);

   /* Iterate each instruction in forward order */
   agx_foreach_instr_in_block(ctx->block, I) {
      assert(ctx->nW <= ctx->k && "invariant");

      /* Debug to check against our RA demand calculations */
      if (0) {
         printf("%u: ", ctx->nW);
         agx_print_instr(I, stdout);
      }

      /* Phis are special since they happen along the edge. When we initialized
       * W and S, we implicitly chose which phis are spilled. So, here we just
       * need to rewrite the phis to write into memory.
       *
       * Phi sources are handled later.
       */
      if (I->op == AGX_OPCODE_PHI) {
         if (!BITSET_TEST(ctx->W, I->dest[0].value)) {
            I->dest[0] = agx_index_as_mem(I->dest[0], ctx->spill_base);
         }

         ctx->ip += instr_cycles(I);
         continue;
      }

      /* Any source that is not in W needs to be reloaded. Gather the set R of
       * such values.
       */
      unsigned R[AGX_MAX_NORMAL_SOURCES];
      unsigned nR = 0;

      agx_foreach_ssa_src(I, s) {
         unsigned node = I->src[s].value;
         if (BITSET_TEST(ctx->W, node))
            continue;

         /* Mark this variable as needing a reload. */
         assert(node < ctx->n);
         assert(BITSET_TEST(ctx->S, node) && "must have been spilled");
         assert(nR < ARRAY_SIZE(R) && "maximum source count");
         R[nR++] = node;

         /* The inserted reload will add the value to the register file. */
         insert_W(ctx, node);
      }

      /* Limit W to make space for the sources we just added */
      limit(ctx, I, ctx->k);

      agx_foreach_ssa_dest(I, d) {
         assert(next_use_cursor >= 1);
         unsigned next_ip = next_ips[--next_use_cursor];

         if (next_ip == DIST_INFINITY)
            remove_W_if_present(ctx, I->dest[d].value);
         else
            ctx->next_uses[I->dest[d].value] = next_ip;
      }

      /* Count how many registers we need for destinations. Because of
       * SSA form, destinations are unique.
       */
      unsigned dest_size = 0;
      agx_foreach_ssa_dest(I, d) {
         dest_size += node_size(ctx, I->dest[d].value);
      }

      /* Limit W to make space for the destinations. */
      limit(ctx, I, ctx->k - dest_size);

      /* Destinations are now in the register file */
      agx_foreach_ssa_dest(I, d) {
         insert_W(ctx, I->dest[d].value);
      }

      /* Update next-use distances for this instruction. Unlike the paper, we
       * prune dead values from W as we go. This doesn't affect correctness, but
       * it speeds up limit() on average.
       *
       * This happens after the above limit() calls to model sources as
       * late-kill. This is conservative and could be improved, but it matches
       * how we currently estimate register demand.
       */
      agx_foreach_ssa_src_rev(I, s) {
         assert(next_use_cursor >= 1);

         unsigned next_ip = next_ips[--next_use_cursor];
         assert((next_ip == DIST_INFINITY) == I->src[s].kill);

         if (I->src[s].kill) {
            remove_W_if_present(ctx, I->src[s].value);
         } else {
            ctx->next_uses[I->src[s].value] = next_ip;
         }
      }

      /* Add reloads for the sources in front of the instruction. We need to be
       * careful around exports, hoisting the reloads to before all exports.
       *
       * This is legal since all exports happen in parallel and all registers
       * are dead after the exports. The register file
       * must be big enough for everything exported, so it must be big enough
       * for all the reloaded values right before the parallel exports.
       */
      for (unsigned i = 0; i < nR; ++i) {
         insert_reload(ctx, ctx->block, agx_before_instr_logical(ctx->block, I),
                       R[i]);
      }

      ctx->ip += instr_cycles(I);
   }

   assert(next_use_cursor == 0 && "exactly sized");

   int i;
   BITSET_FOREACH_SET(i, ctx->W, ctx->n)
      sblock->W_exit[sblock->nW_exit++] = i;

   sblock->S_exit = bitset_members(ctx, ctx->S, &sblock->nS_exit);
   util_dynarray_fini(&local_next_ip);
}

/* Highest register demand at any instruction of a block not yet processed by
 * the spiller, in node_size units. Rematerializable values are evicted and
 * recreated for free, so they only count at the instructions using them.
 * Sources are late-kill, matching min_algorithm. Also records the values
 * the block uses.
 */
static unsigned
block_max_pressure(struct spill_ctx *ctx, agx_block *block, BITSET_WORD *live,
                   BITSET_WORD *used)
{
   memset(live, 0, BITSET_BYTES(ctx->n));
   unsigned pressure = 0;
   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, i) {
      if (i < ctx->n && !ctx->remat[i]) {
         BITSET_SET(live, i);
         pressure += node_size(ctx, i);
      }
   }
   unsigned max = pressure;
   agx_foreach_instr_in_block_rev(block, I) {
      if (I->op == AGX_OPCODE_PHI)
         break;
      unsigned defs = 0;
      agx_foreach_ssa_dest(I, d) {
         unsigned v = I->dest[d].value;
         if (v >= ctx->n)
            continue;
         defs += node_size(ctx, v);
         if (BITSET_TEST(live, v)) {
            BITSET_CLEAR(live, v);
            pressure -= node_size(ctx, v);
         }
      }
      unsigned remat_srcs = 0;
      agx_foreach_ssa_src(I, s) {
         unsigned v = I->src[s].value;
         if (v >= ctx->n)
            continue;
         BITSET_SET(used, v);
         if (ctx->remat[v]) {
            remat_srcs += node_size(ctx, v);
         } else if (!BITSET_TEST(live, v)) {
            BITSET_SET(live, v);
            pressure += node_size(ctx, v);
         }
      }
      max = MAX2(max, pressure + remat_srcs + defs);
   }
   return max;
}

/* Whether a live-in value of a loop header is in a register at the end of
 * every forward predecessor, which have already been processed.
 */
static bool
in_preheader_w_exit(struct spill_ctx *ctx, agx_block *header, unsigned node)
{
   /* The header's own phis are defined on the entry edge */
   agx_foreach_phi_in_block(header, phi) {
      if (phi->dest[0].value == node)
         return true;
   }

   agx_foreach_predecessor(header, pred) {
      if ((*pred)->index >= header->index)
         continue;
      struct spill_block *sp = spill_block(ctx, *pred);
      bool found = false;
      for (unsigned i = 0; i < sp->nW_exit && !found; ++i)
         found = sp->W_exit[i] == node;
      if (!found)
         return false;
   }
   return true;
}

/*
 * Section 4.2 of the paper: at a loop header, keep the values used inside the
 * loop in registers, nearest next use first. Values live through the loop but
 * unused in it (next use beyond a loop exit) are kept only in registers the
 * loop's own peak demand leaves free; otherwise they are spilled once, before
 * the loop, instead of being evicted and reloaded on every iteration.
 *
 * Without loop handling, fall back to the heuristic in Hack's thesis: sort the
 * live-in set (+ destinations of phis) by next-use distance.
 */
static ATTRIBUTE_NOINLINE void
compute_w_entry_loop_header(struct spill_ctx *ctx)
{
   agx_block *block = ctx->block;
   struct spill_block *sb = spill_block(ctx, block);

   unsigned nP = u_sparse_bitset_count(&block->live_in);
   struct candidate *candidates = calloc(nP, sizeof(struct candidate));
   unsigned j = 0;

   foreach_next_use(&sb->next_use_in, i, dist) {
      assert(j < nP);
      candidates[j++] = (struct candidate){.node = i, .dist = dist};
   }

   assert(j == nP);

   /* Sort by next-use distance */
   util_qsort_r(candidates, j, sizeof(struct candidate), cmp_dist, ctx);

   /* Values live through the loop but unused in it are live in every block
    * of the loop, so registers left for them are k minus the loop's peak
    * demand without them.
    */
   unsigned through_budget = ctx->k;
   BITSET_WORD *used = NULL;
   if (ctx->loop_end[block->index]) {
      unsigned end = ctx->loop_end[block->index];
      BITSET_WORD *live = malloc(BITSET_BYTES(ctx->n));
      used = calloc(BITSET_WORDS(ctx->n), sizeof(BITSET_WORD));
      unsigned peak = 0;
      agx_foreach_block_from(ctx->shader, block, b) {
         if (b->index > end)
            break;
         peak = MAX2(peak, block_max_pressure(ctx, b, live, used));
         agx_foreach_phi_in_block(b, phi) {
            agx_foreach_ssa_src(phi, s) {
               agx_block *pred =
                  *util_dynarray_element(&b->predecessors, agx_block *, s);
               if (loop_contains(ctx->loop_end, block->index, pred->index) &&
                   phi->src[s].value < ctx->n)
                  BITSET_SET(used, phi->src[s].value);
            }
         }
      }
      free(live);
      unsigned through = 0;
      for (unsigned i = 0; i < j; ++i) {
         unsigned v = candidates[i].node;
         if (!BITSET_TEST(used, v) && !ctx->remat[v])
            through += node_size(ctx, v);
      }
      unsigned own = peak > through ? peak - through : 0;
      through_budget = own < ctx->k ? ctx->k - own : 0;
      if (spill_trace_loops())
         fprintf(stderr,
                 "SPILL_LOOP header=%u end=%u k=%u live_in=%u peak=%u "
                 "through=%u budget=%u\n",
                 block->index, end, ctx->k, j, peak, through, through_budget);
   }

   /* Take as much as we can */
   unsigned through_taken = 0;
   for (unsigned i = 0; i < j; ++i) {
      unsigned node = candidates[i].node;
      unsigned comps = node_size(ctx, node);
      bool through = used && !BITSET_TEST(used, node);

      /* Reloading a value before a loop that does not use it never pays:
       * keep it only if the preheader still has it in a register, and only
       * in registers the loop leaves free.
       */
      bool fits = (ctx->nW + comps) <= ctx->k;
      if (through) {
         fits &= through_taken + comps <= through_budget &&
                 in_preheader_w_exit(ctx, block, node);
      }

      if (fits) {
         insert_W(ctx, node);
         sb->W_entry[sb->nW_entry++] = node;
         if (through)
            through_taken += comps;
      }
      if (spill_trace_loops())
         fprintf(stderr, "  %%%u size=%u dist=%u%s%s\n", node, comps,
                 candidates[i].dist, through ? " through" : "",
                 fits ? " W" : "");
   }

   assert(ctx->nW <= ctx->k);
   free(candidates);
   free(used);
}

/*
 * Compute W_entry for a block. Section 4.2 in the paper.
 */
static ATTRIBUTE_NOINLINE void
compute_w_entry(struct spill_ctx *ctx)
{
   agx_block *block = ctx->block;
   struct spill_block *sb = spill_block(ctx, block);

   /* Nothing to do for start blocks */
   if (agx_num_predecessors(block) == 0)
      return;

   /* Loop headers have a different heuristic */
   if (block->loop_header) {
      compute_w_entry_loop_header(ctx);
      return;
   }

   /* Usual blocks follow */
   unsigned *freq = ctx->freq;

   /* Record what's written at the end of each predecessor */
   agx_foreach_predecessor(ctx->block, P) {
      struct spill_block *sp = spill_block(ctx, *P);

      for (unsigned i = 0; i < sp->nW_exit; ++i) {
         unsigned v = sp->W_exit[i];
         freq[v]++;
      }
   }

   struct util_dynarray candidate_array;
   util_dynarray_init(&candidate_array, NULL);

   /* Variables that are in all predecessors are assumed in W_entry. Phis and
    * variables in some predecessors are scored by next-use.
    */
   foreach_next_use(&sb->next_use_in, i, dist) {
      if (freq[i] == agx_num_predecessors(ctx->block)) {
         insert_W(ctx, i);
      } else if (freq[i]) {
         util_dynarray_append(&candidate_array,
                              ((struct candidate){.node = i, .dist = dist}));
      }
   }

   agx_foreach_predecessor(ctx->block, P) {
      struct spill_block *sp = spill_block(ctx, *P);

      for (unsigned i = 0; i < sp->nW_exit; ++i)
         freq[sp->W_exit[i]] = 0;
   }

   agx_foreach_phi_in_block(ctx->block, I) {
      bool all_found = true, any_found = false;

      agx_foreach_predecessor(ctx->block, pred) {
         struct spill_block *sp = spill_block(ctx, *pred);
         bool found = false;

         agx_index src = I->src[agx_predecessor_index(ctx->block, *pred)];
         if (src.type != AGX_INDEX_NORMAL)
            continue;

         unsigned v = src.value;
         for (unsigned i = 0; i < sp->nW_exit; ++i) {
            if (sp->W_exit[i] == v) {
               found = true;
               break;
            }
         }

         all_found &= found;
         any_found |= found;
      }

      /* Heuristic: if any phi source is spilled, spill the whole phi. This is
       * suboptimal, but it massively reduces pointless fill/spill chains with
       * massive phi webs. A target may instead keep a phi whose source some
       * predecessor still holds; coupling code reloads it on the other edges.
       */
      if (!(ctx->shader->ra_target.spill_partial_phi_sources ? any_found
                                                              : all_found))
         continue;

      util_dynarray_append(
         &candidate_array,
         ((struct candidate){
            .node = I->dest[0].value,
            .dist = search_next_uses(&sb->next_use_in, I->dest[0].value),
         }));
   }

   struct candidate *candidates = candidate_array.data;
   unsigned j = util_dynarray_num_elements(&candidate_array, struct candidate);

   /* Sort by next-use distance */
   util_qsort_r(candidates, j, sizeof(struct candidate), cmp_dist, ctx);

   /* Take as much as we can */
   for (unsigned i = 0; i < j; ++i) {
      unsigned node = candidates[i].node;
      unsigned comps = node_size(ctx, node);

      if ((ctx->nW + comps) <= ctx->k) {
         insert_W(ctx, node);
         sb->W_entry[sb->nW_entry++] = node;
      }
   }

   assert(ctx->nW <= ctx->k && "invariant");

   util_dynarray_fini(&candidate_array);
}

/*
 * We initialize S with the union of S at the exit of (forward edge)
 * predecessors and the complement of W, intersected with the live-in set. The
 * former propagates S forward. The latter ensures we spill along the edge when
 * a live value is not selected for the entry W.
 */
static ATTRIBUTE_NOINLINE void
compute_s_entry(struct spill_ctx *ctx)
{
   unsigned v;

   agx_foreach_predecessor(ctx->block, pred) {
      struct spill_block *sp = spill_block(ctx, *pred);

      for (unsigned i = 0; i < sp->nS_exit; ++i) {
         v = sp->S_exit[i];

         if (u_sparse_bitset_test(&ctx->block->live_in, v))
            BITSET_SET(ctx->S, v);
      }
   }

   U_SPARSE_BITSET_FOREACH_SET(&ctx->block->live_in, v) {
      if (!BITSET_TEST(ctx->W, v))
         BITSET_SET(ctx->S, v);
   }

   /* Copy ctx->S to S_entry for later look-ups with coupling code */
   struct spill_block *sb = spill_block(ctx, ctx->block);
   sb->S_entry = bitset_members(ctx, ctx->S, &sb->nS_entry);
}

static int
cmp_unsigned(const void *a, const void *b)
{
   unsigned x = *(const unsigned *)a, y = *(const unsigned *)b;
   return x < y ? -1 : x > y;
}

static ATTRIBUTE_NOINLINE void
global_next_use_distances(agx_context *ctx, void *memctx,
                          struct spill_block *blocks, const unsigned *loop_end)
{
   u_worklist worklist;
   u_worklist_init(&worklist, ctx->num_blocks, NULL);

   agx_foreach_block(ctx, block) {
      struct spill_block *sb = &blocks[block->index];

      init_next_uses(&sb->next_use_in, memctx);
      init_next_uses(&sb->next_use_out, memctx);

      agx_foreach_instr_in_block(block, I) {
         sb->cycles += instr_cycles(I);
      }

      agx_worklist_push_head(&worklist, block);
   }

   /* A block's upward-exposed uses and its definitions do not change while
    * the dataflow iterates, so gather them once: the uses in first-use order
    * with their distance from the block entry, and the definitions in
    * ascending order. Each visit then applies the same updates as a rescan,
    * in the same order, without touching every variable of the shader.
    */
   struct block_facts {
      unsigned *gen_node;
      uint32_t *gen_dist;
      unsigned *kill;
      unsigned gen_count, kill_count;
   } *facts = rzalloc_array(memctx, struct block_facts, ctx->num_blocks);
   BITSET_WORD *defined = calloc(BITSET_WORDS(ctx->alloc), sizeof(BITSET_WORD));
   BITSET_WORD *used = calloc(BITSET_WORDS(ctx->alloc), sizeof(BITSET_WORD));
   struct util_dynarray gen_node, gen_dist, kill;
   util_dynarray_init(&gen_node, NULL);
   util_dynarray_init(&gen_dist, NULL);
   util_dynarray_init(&kill, NULL);

   agx_foreach_block(ctx, blk) {
      struct block_facts *f = &facts[blk->index];
      util_dynarray_clear(&gen_node);
      util_dynarray_clear(&gen_dist);
      util_dynarray_clear(&kill);
      uint32_t cycle = 0;

      agx_foreach_instr_in_block(blk, I) {
         if (I->op == AGX_OPCODE_PHI) {
            cycle++;
            continue;
         }

         /* Record first use before def. Phi sources are handled when
          * propagating, because they logically happen in the predecessor.
          */
         agx_foreach_ssa_src(I, s) {
            unsigned v = I->src[s].value;
            assert(v < ctx->alloc);
            if (BITSET_TEST(defined, v) || BITSET_TEST(used, v))
               continue;
            BITSET_SET(used, v);
            util_dynarray_append(&gen_node, v);
            util_dynarray_append(&gen_dist, cycle);
         }

         /* Record defs */
         agx_foreach_ssa_dest(I, d) {
            unsigned v = I->dest[d].value;
            assert(v < ctx->alloc);
            if (!BITSET_TEST(defined, v)) {
               BITSET_SET(defined, v);
               util_dynarray_append(&kill, v);
            }
         }

         cycle += instr_cycles(I);
      }

      f->gen_count = util_dynarray_num_elements(&gen_node, unsigned);
      f->kill_count = util_dynarray_num_elements(&kill, unsigned);
      f->gen_node = ralloc_array(memctx, unsigned, MAX2(f->gen_count, 1));
      f->gen_dist = ralloc_array(memctx, uint32_t, MAX2(f->gen_count, 1));
      f->kill = ralloc_array(memctx, unsigned, MAX2(f->kill_count, 1));
      memcpy(f->gen_node, gen_node.data, f->gen_count * sizeof(unsigned));
      memcpy(f->gen_dist, gen_dist.data, f->gen_count * sizeof(uint32_t));
      memcpy(f->kill, kill.data, f->kill_count * sizeof(unsigned));
      qsort(f->kill, f->kill_count, sizeof(unsigned), cmp_unsigned);

      for (unsigned i = 0; i < f->gen_count; ++i)
         BITSET_CLEAR(used, f->gen_node[i]);
      for (unsigned i = 0; i < f->kill_count; ++i)
         BITSET_CLEAR(defined, f->kill[i]);
   }

   free(defined);
   free(used);
   util_dynarray_fini(&gen_node);
   util_dynarray_fini(&gen_dist);
   util_dynarray_fini(&kill);

   struct next_uses dists, far;
   init_next_uses(&dists, NULL);
   init_next_uses(&far, NULL);

   /* Iterate the work list in reverse order since liveness is backwards */
   while (!u_worklist_is_empty(&worklist)) {
      agx_block *blk = agx_worklist_pop_head(&worklist);
      struct spill_block *sb = &blocks[blk->index];
      const struct block_facts *f = &facts[blk->index];

      /* Initialize all distances to infinity */
      clear_next_uses(&dists);

      /* Calculate dists. Phis are handled separately. */
      for (unsigned i = 0; i < f->gen_count; ++i)
         set_next_use(&dists, f->gen_node[i], f->gen_dist[i]);

      /* Apply transfer function to get our entry state. */
      foreach_next_use(&sb->next_use_out, node, dist) {
         set_next_use(&sb->next_use_in, node, dist_sum(dist, sb->cycles));
      }

      foreach_next_use(&dists, node, dist) {
         set_next_use(&sb->next_use_in, node, dist);
      }

      for (unsigned i = 0; i < f->kill_count; ++i)
         set_next_use(&sb->next_use_in, f->kill[i], DIST_INFINITY);

      /* Propagate the live in of the successor (blk) to the live out of
       * predecessors.
       *
       * Phi nodes are logically on the control flow edge and act in parallel.
       * To handle when propagating, we kill writes from phis and make live the
       * corresponding sources.
       */
      agx_foreach_predecessor(blk, pred) {
         struct spill_block *sp = &blocks[(*pred)->index];
         copy_next_uses(&dists, &sb->next_use_in);

         /* Kill write */
         agx_foreach_phi_in_block(blk, I) {
            assert(I->dest[0].type == AGX_INDEX_NORMAL);
            set_next_use(&dists, I->dest[0].value, DIST_INFINITY);
         }

         /* Uses reached only by leaving loops are far away inside them. */
         unsigned exits = loops_exited(ctx, loop_end, *pred, blk);
         if (exits) {
            clear_next_uses(&far);
            foreach_next_use(&dists, node, dist) {
               set_next_use(&far, node, dist_sum(dist, exits * LOOP_EXIT_DIST));
            }
            copy_next_uses(&dists, &far);
         }

         /* Make live the corresponding source */
         agx_foreach_phi_in_block(blk, I) {
            agx_index operand = I->src[agx_predecessor_index(blk, *pred)];
            if (operand.type == AGX_INDEX_NORMAL)
               set_next_use(&dists, operand.value, 0);
         }

         /* Join by taking minimum */
         if (minimum_next_uses(&sp->next_use_out, &dists))
            agx_worklist_push_tail(&worklist, *pred);
      }
   }

   u_worklist_fini(&worklist);
   destroy_next_uses(&dists);
   destroy_next_uses(&far);
}

static ATTRIBUTE_NOINLINE void
validate_next_use_info(UNUSED agx_context *ctx,
                       UNUSED struct spill_block *blocks)
{
#ifndef NDEBUG
   agx_foreach_block(ctx, blk) {
      struct spill_block *sb = &blocks[blk->index];

      /* Invariant: next-use distance is finite iff the node is live */
      U_SPARSE_BITSET_FOREACH_SET(&blk->live_in, i)
         assert(search_next_uses(&sb->next_use_in, i) < DIST_INFINITY);

      U_SPARSE_BITSET_FOREACH_SET(&blk->live_out, i)
         assert(search_next_uses(&sb->next_use_out, i) < DIST_INFINITY);

      foreach_next_use(&sb->next_use_in, i, _)
         assert(u_sparse_bitset_test(&blk->live_in, i));

      foreach_next_use(&sb->next_use_out, i, _)
         assert(u_sparse_bitset_test(&blk->live_out, i));
   }
#endif
}

/* A spilled loop-invariant SSA value has the same bits on every iteration.
 * If it is already available at the unique preheader exit, write its scratch
 * copy there once. Do not extend a register live range through the preheader:
 * require either an existing scratch copy or membership in its exit W set.
 * Loop-carried phis are excluded because their destination is not live-in.
 */
static void
hoist_loop_invariant_spills(agx_context *ctx, unsigned n,
                            const struct spill_block *blocks,
                            const unsigned *loop_end)
{
   if (!ctx->ra_target.defer_spill_lowering)
      return;

   agx_foreach_block(ctx, header) {
      unsigned end = loop_end[header->index];
      if (!end)
         continue;
      agx_block *preheader = NULL;
      unsigned entries = 0;
      agx_foreach_predecessor(header, pred) {
         if (!loop_contains(loop_end, header->index, (*pred)->index)) {
            preheader = *pred;
            ++entries;
         }
      }
      if (entries != 1 || preheader->successors[1] ||
          preheader->successors[0] != header)
         continue;

      BITSET_WORD *available = calloc(BITSET_WORDS(n), sizeof(BITSET_WORD));
      BITSET_WORD *stored = calloc(BITSET_WORDS(n), sizeof(BITSET_WORD));
      const struct spill_block *sb = &blocks[preheader->index];
      for (unsigned i = 0; i < sb->nW_exit; ++i)
         BITSET_SET(available, sb->W_exit[i]);
      for (unsigned i = 0; i < sb->nS_exit; ++i)
         BITSET_SET(stored, sb->S_exit[i]);

      agx_foreach_block_from(ctx, header, block) {
         if (block->index > end)
            break;
         agx_foreach_instr_in_block_safe(block, I) {
            if (I->op != AGX_OPCODE_MOV || !I->dest[0].memory ||
                I->src[0].memory || I->src[0].type != AGX_INDEX_NORMAL)
               continue;
            unsigned v = I->src[0].value;
            if (v >= n || I->dest[0].value != n + v ||
                !u_sparse_bitset_test(&header->live_in, v) ||
                (!BITSET_TEST(available, v) && !BITSET_TEST(stored, v)))
               continue;
            if (!BITSET_TEST(stored, v)) {
               agx_builder b = agx_init_builder(
                  ctx, agx_after_block_logical(preheader));
               agx_mov_to(&b, I->dest[0], I->src[0]);
               BITSET_SET(stored, v);
            }
            agx_remove_instruction(I);
         }
      }
      free(available);
      free(stored);
   }
}

void
agx_spill(agx_context *ctx, unsigned k, bool remat_only)
{
   void *memctx = ralloc_context(NULL);

   /* We need extra registers for memory-memory swaps */
   k -= (ctx->ra_target.reserved_registers ||
         ctx->ra_target.spill_reserved_registers)
           ? MAX2(ctx->ra_target.reserved_registers,
                  remat_only ? 0 : ctx->ra_target.spill_reserved_registers)
           : (remat_only ? 2 : 8);

   uint8_t *channels = rzalloc_array(memctx, uint8_t, ctx->alloc);
   dist_t *next_uses = rzalloc_array(memctx, dist_t, ctx->alloc);
   enum agx_size *sizes = rzalloc_array(memctx, enum agx_size, ctx->alloc);
   agx_instr **remat = rzalloc_array(memctx, agx_instr *, ctx->alloc);

   agx_foreach_instr_global(ctx, I) {
      if (can_remat(I))
         remat[I->dest[0].value] = I;

      /* Measure vectors */
      agx_foreach_ssa_dest(I, d) {
         assert(sizes[I->dest[d].value] == 0 && "broken SSA");
         assert(channels[I->dest[d].value] == 0 && "broken SSA");

         sizes[I->dest[d].value] = I->dest[d].size;
         channels[I->dest[d].value] = agx_channels(I->dest[d]);
      }
   }

   /* Reconstructed sources overlap their result under late-kill accounting.
    * Reserve their temporaries without making the remaining budget smaller
    * than one instruction's simultaneous sources and destinations. In that
    * case limit() could evict a current source, which cannot release its
    * register until after the instruction. This matters for forced spilling
    * where a vector load may already consume the entire available budget. */
   unsigned minimum = 0;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_PHI)
         continue;
      unsigned needed = 0;
      agx_foreach_ssa_dest(I, d)
         needed += util_next_power_of_two(agx_index_size_16(I->dest[d]));
      agx_foreach_ssa_src(I, s) {
         bool duplicate = false;
         for (unsigned prev = 0; prev < s; ++prev)
            duplicate |= agx_is_equiv(I->src[prev], I->src[s]);
         if (!duplicate)
            needed += util_next_power_of_two(agx_index_size_16(I->src[s]));
      }
      minimum = MAX2(minimum, needed);
   }
   k -= find_apple9_remat(ctx, remat, memctx,
                          k > minimum ? k - minimum : 0);

   struct spill_block *blocks =
      rzalloc_array(memctx, struct spill_block, ctx->num_blocks);

   /* Step 1. Compute global next-use distances */
   unsigned *loop_end = compute_loop_ends(ctx, memctx);
   global_next_use_distances(ctx, memctx, blocks, loop_end);
   validate_next_use_info(ctx, blocks);

   /* Reserve a memory variable for every regular variable */
   unsigned n = ctx->alloc;
   ctx->alloc *= 2;

   /* W and S start each block and each edge empty; their users empty them
    * through the block's exit lists instead of clearing all n bits. */
   BITSET_WORD *W = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   BITSET_WORD *S = rzalloc_array(memctx, BITSET_WORD, BITSET_WORDS(n));
   unsigned *freq = rzalloc_array(memctx, unsigned, n);
   unsigned *members = ralloc_array(memctx, unsigned, n);

   agx_foreach_block(ctx, block) {

      struct spill_ctx sctx = {
         .memctx = memctx,
         .shader = ctx,
         .n = n,
         .channels = channels,
         .size = sizes,
         .remat = remat,
         .next_uses = next_uses,
         .block = block,
         .blocks = blocks,
         .k = k,
         .W = W,
         .S = S,
         .spill_base = n,
         .remat_only = remat_only,
         .loop_end = loop_end,
         .freq = freq,
         .members = members,
      };

      compute_w_entry(&sctx);
      compute_s_entry(&sctx);
      min_algorithm(&sctx);

      /* W and S now hold exactly the exit sets. */
      struct spill_block *sb = spill_block(&sctx, block);
      for (unsigned i = 0; i < sb->nW_exit; ++i)
         BITSET_CLEAR(W, sb->W_exit[i]);
      for (unsigned i = 0; i < sb->nS_exit; ++i)
         BITSET_CLEAR(S, sb->S_exit[i]);
   }

   /* Now that all blocks are processed separately, stitch it together */
   agx_foreach_block(ctx, block) {
      struct spill_ctx sctx = {
         .memctx = memctx,
         .shader = ctx,
         .n = n,
         .channels = channels,
         .size = sizes,
         .remat = remat,
         .block = block,
         .blocks = blocks,
         .k = k,
         .W = W,
         .S = S,
         .spill_base = n,
      };

      agx_foreach_predecessor(block, pred) {
         /* After spilling phi sources, insert coupling code */
         insert_coupling_code(&sctx, *pred, block);
      }
   }

   hoist_loop_invariant_spills(ctx, n, blocks, loop_end);
   ralloc_free(memctx);

   /* Spilling breaks SSA, so we need to repair before validating */
   agx_repair_ssa(ctx);
   agx_validate(ctx, "Spilling");

   /* Remat can introduce dead code */
   agx_dce(ctx, false);

   if (spill_trace_loops()) {
      fprintf(stderr, "SPILL_RESULT_BEGIN\n");
      agx_print_shader(ctx, stderr);
      fprintf(stderr, "SPILL_RESULT_END\n");
   }
}
