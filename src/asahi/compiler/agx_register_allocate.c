/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/bitset.h"
#include "util/macros.h"
#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_qsort.h"
#include "agx_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_opcodes.h"
#include "shader_enums.h"

#include "test/cs2-pressure/ra-audit.h"

/* SSA-based register allocator */
struct phi_web_node {
   /* Parent index, or circular for root */
   uint32_t parent;

   /* Rank, at most log2(n) so need ~5-bits */
   uint8_t rank;
};
static_assert(sizeof(struct phi_web_node) == 8, "packed");

static unsigned
phi_web_find(struct phi_web_node *web, unsigned x)
{
   if (web[x].parent == x) {
      /* Root */
      return x;
   } else {
      /* Search up the tree */
      unsigned root = x;
      while (web[root].parent != root)
         root = web[root].parent;

      /* Compress path. Second pass ensures O(1) memory usage. */
      while (web[x].parent != x) {
         unsigned temp = web[x].parent;
         web[x].parent = root;
         x = temp;
      }

      return root;
   }
}

static void
phi_web_union(struct phi_web_node *web, unsigned x, unsigned y)
{
   x = phi_web_find(web, x);
   y = phi_web_find(web, y);

   if (x == y)
      return;

   /* Union-by-rank: ensure x.rank >= y.rank */
   if (web[x].rank < web[y].rank) {
      SWAP(x, y);
   }

   web[y].parent = x;

   /* Increment rank if necessary */
   if (web[x].rank == web[y].rank) {
      web[x].rank++;
   }
}

/* Only phi participants can be merged. Restrict the interference graph to
 * those values, while using the full CFG liveness already computed for RA.
 * Constants have private incoming-use members and need no graph vertices.
 */
static struct u_sparse_bitset *
phi_web_interference(agx_context *ctx, BITSET_WORD *constants, void *memctx)
{
   BITSET_WORD *participates = BITSET_RZALLOC(memctx, ctx->alloc);
   BITSET_WORD *memory = BITSET_RZALLOC(memctx, ctx->alloc);
   struct u_sparse_bitset *graph =
      rzalloc_array(memctx, struct u_sparse_bitset, ctx->alloc);

   for (unsigned i = 0; i < ctx->alloc; ++i)
      u_sparse_bitset_init(&graph[i], ctx->alloc, memctx);

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d) {
         if (I->dest[d].memory)
            BITSET_SET(memory, I->dest[d].value);
      }

      if (I->op != AGX_OPCODE_PHI)
         continue;

      BITSET_SET(participates, I->dest[0].value);
      agx_foreach_ssa_src(I, s) {
         if (!BITSET_TEST(constants, I->src[s].value))
            BITSET_SET(participates, I->src[s].value);
      }
   }

   struct u_sparse_bitset live;
   u_sparse_bitset_init(&live, ctx->alloc, memctx);

   agx_foreach_block(ctx, block) {
      U_SPARSE_BITSET_FOREACH_SET(&block->live_out, v) {
         if (BITSET_TEST(participates, v))
            u_sparse_bitset_set(&live, v);
      }

      agx_foreach_instr_in_block_rev(block, I) {
         if (I->op == AGX_OPCODE_PHI)
            break;

         /* All destinations are simultaneous writes, including dead results. */
         agx_foreach_ssa_dest(I, d) {
            if (BITSET_TEST(participates, I->dest[d].value))
               u_sparse_bitset_set(&live, I->dest[d].value);
         }

         agx_foreach_ssa_dest(I, d) {
            unsigned v = I->dest[d].value;
            if (!BITSET_TEST(participates, v))
               continue;

            U_SPARSE_BITSET_FOREACH_SET(&live, other) {
               if (v != other &&
                   !!BITSET_TEST(memory, v) == !!BITSET_TEST(memory, other)) {
                  u_sparse_bitset_set(&graph[v], other);
                  u_sparse_bitset_set(&graph[other], v);
               }
            }
         }

         agx_foreach_ssa_dest(I, d)
            u_sparse_bitset_clear(&live, I->dest[d].value);

         agx_foreach_ssa_src(I, s) {
            if (BITSET_TEST(participates, I->src[s].value))
               u_sparse_bitset_set(&live, I->src[s].value);
         }
      }

      /* Phis define a parallel bundle at block entry. Their inputs are uses on
       * predecessor edges, already represented by the predecessors' live_out.
       */
      agx_foreach_phi_in_block(block, phi)
         u_sparse_bitset_set(&live, phi->dest[0].value);

      agx_foreach_phi_in_block(block, phi) {
         unsigned v = phi->dest[0].value;
         U_SPARSE_BITSET_FOREACH_SET(&live, other) {
            if (v != other &&
                !!BITSET_TEST(memory, v) == !!BITSET_TEST(memory, other)) {
               u_sparse_bitset_set(&graph[v], other);
               u_sparse_bitset_set(&graph[other], v);
            }
         }
      }

      u_sparse_bitset_clear_all(&live);
   }

   return graph;
}

struct phi_web_members {
   unsigned head, tail, next, count;

   /* A private constant-use member keeps this edge's placement preference,
    * without joining the shared constant definition to other phi classes.
    */
   struct phi_preference *constant;
};

static void
phi_web_union_noninterfering(struct phi_web_node *web,
                            struct phi_web_members *members,
                            struct u_sparse_bitset *interference,
                            unsigned nr_ssa, unsigned x, unsigned y)
{
   x = phi_web_find(web, x);
   y = phi_web_find(web, y);
   if (x == y)
      return;

   /* Inspect the smaller class. An interference edge to any member of the
    * other class prevents the merge, not just an edge between the phi's own
    * source and destination. This preserves noninterference transitively.
    */
   if (members[x].count > members[y].count)
      SWAP(x, y);

   for (unsigned v = members[x].head; v != ~0u; v = members[v].next) {
      if (v >= nr_ssa) /* private constant-use affinity member */
         continue;

      U_SPARSE_BITSET_FOREACH_SET(&interference[v], other) {
         if (phi_web_find(web, other) == y)
            return;
      }
   }

   phi_web_union(web, x, y);
   unsigned root = phi_web_find(web, x);
   unsigned other = root == x ? y : x;
   members[members[root].tail].next = members[other].head;
   members[root].tail = members[other].tail;
   members[root].count += members[other].count;
}

/* Each phi operand is an edge preference. Keep the operand address so that
 * live-range splits on predecessor edges are reflected in its register.
 */
struct phi_preference {
   struct phi_preference *next;
   const agx_index *other;
   uint64_t weight;
};

/* A dying source can eliminate at most one copy in a parallel phi bundle.
 * Keep all destination choices, but charge the edge only once per register. */
struct phi_dead_destination {
   struct phi_dead_destination *next;
   const agx_index *index;
};

struct phi_dead_group {
   struct phi_dead_group *next;
   agx_block *successor, *predecessor;
   struct phi_dead_destination *destinations;
   uint64_t weight;
};

struct phi_reg_preference {
   unsigned reg;
   uint64_t dead, edge, affinity;
};

struct ra_ctx {
   agx_context *shader;
   agx_block *block;
   agx_instr *instr;
   uint16_t *ssa_to_reg;
   uint8_t *ncomps;
   uint8_t *ncomps_unrounded;
   enum agx_size *sizes;
   enum ra_class *classes;
   BITSET_WORD *visited;
   BITSET_WORD *used_regs[RA_CLASSES];

   /* Were any sources killed early this instruction? We assert this is not true
    * when shuffling.
    */
   bool early_killed;

   /* Maintained while assigning registers. Count of registers required, i.e.
    * the maximum register assigned + 1.
    */
   unsigned *count[RA_CLASSES];

   /* For affinities */
   agx_instr **src_to_collect;
   struct phi_web_node *phi_web;
   struct phi_web_members *web_members;
   struct phi_preference **phi_preferences;
   struct phi_dead_group **dead_groups;
   uint64_t *phi_weights;
   unsigned *last_reg;
   struct util_dynarray *reg_preferences;
   /* Index of each register's entry in reg_preferences, or -1. */
   int16_t *reg_preference_index;
   uint64_t dead_total, edge_total, affinity_total;

   /* If bit i of used_regs is set, and register i is the first consecutive
    * register holding an SSA value, then reg_to_ssa[i] is the SSA index of the
    * value currently in register  i.
    *
    * Only for GPRs. We can add reg classes later if we have a use case.
    */
   uint32_t reg_to_ssa[AGX_NUM_REGS];

   /* Maximum number of registers that RA is allowed to use */
   unsigned bound[RA_CLASSES];

   /* Operand placements committed for the current constrained instruction. */
   BITSET_WORD *pinned;
   /* While a constrained instruction is placed: the original register of
    * each value live before it that has since moved, in first-move order. */
   struct ra_move {
      unsigned ssa, reg;
   } *moved;
   unsigned nr_moved;

   /* Registers of the current instruction's dying sources that its
    * destination may take over, and the sources it took over. */
   BITSET_WORD *reusable;
   uint64_t reused_sources;

   /* Cached preferences for the value being placed. */
   unsigned affinity_ssa, affinity_reg;

   /* Soft placement hints. Instruction constraints remain the only hard
    * restrictions, since different uses can need incompatible banks. */
   const struct agx_reg_constraint *preferred;
   const unsigned *scarcity;
   /* scarcity_floor[l][r]: the least scarcity of any 1 << l register window
    * starting at r or above, a lower bound for placement_floor. */
   const unsigned (*scarcity_floor)[AGX_NUM_REGS + 1];
   unsigned demand;
};

/*
 * RA treats the nesting counter, the divergent shuffle temporary, and the
 * spiller temporaries as alive throughout if used anywhere. This could be
 * optimized. Using a single power-of-two reserved region at the start ensures
 * these registers are never shuffled.
 */
static unsigned
reserved_size(agx_context *ctx)
{
   if (ctx->ra_target.reserved_registers ||
       ctx->ra_target.spill_reserved_registers)
      return MAX2(ctx->ra_target.reserved_registers,
                  ctx->has_spill_pcopy_reserved ? ctx->ra_target.spill_reserved_registers : 0);
   if (ctx->has_spill_pcopy_reserved)
      return 8;
   else if (ctx->any_quad_divergent_shuffle)
      return 2;
   else if (ctx->any_cf)
      return 1;
   else
      return 0;
}

static unsigned preferred_phi_register(struct ra_ctx *rctx, unsigned ssa,
                                       bool available);

/* Opt-in evidence for failed phi-web affinities; never changes placement. */
static void
pressure_web_event(struct ra_ctx *rctx, unsigned ssa, const char *event,
                   unsigned chosen, const struct agx_reg_constraint *constraint)
{
   if (pressure_audit.ctx != rctx->shader ||
       !getenv("AGX_APPLE9_RA_WEBS") || rctx->classes[ssa] != RA_GPR)
      return;
   unsigned root = phi_web_find(rctx->phi_web, ssa);
   unsigned want = preferred_phi_register(rctx, ssa, false);
   unsigned count = rctx->ncomps[ssa];
   fprintf(stderr, "RA_WEB_%s id=%u block=%u loop=%u ssa=%u web=%u "
           "op=%s want=%u chosen=%u width=%u bound=%u min=%u max=%u align=%u owners=",
           event, pressure_audit.serial, rctx->block->index,
           rctx->block->loop_header, ssa, root,
           agx_opcodes_info[rctx->instr->op].name, want, chosen, count,
           rctx->bound[RA_GPR], constraint ? constraint->min : 0,
           constraint ? constraint->max : rctx->bound[RA_GPR] - count,
           constraint ? constraint->align : count);
   if (want != ~0u) {
      for (unsigned reg = reserved_size(rctx->shader);
           reg < rctx->bound[RA_GPR];) {
         if (!BITSET_TEST(rctx->used_regs[RA_GPR], reg)) {
            ++reg;
            continue;
         }
         unsigned owner = rctx->reg_to_ssa[reg];
         unsigned width = rctx->ncomps[owner];
         assert(width);
         if (reg < want + count && want < reg + width)
            fprintf(stderr, "%u:%u:%u:%u,", owner,
                    phi_web_find(rctx->phi_web, owner), reg, width);
         reg += width;
      }
   }
   fputc('\n', stderr);
}

UNUSED static void
print_reg_file(struct ra_ctx *rctx, FILE *fp)
{
   unsigned reserved = reserved_size(rctx->shader);

   /* Dump the contents */
   for (unsigned i = reserved; i < rctx->bound[RA_GPR]; ++i) {
      if (BITSET_TEST(rctx->used_regs[RA_GPR], i)) {
         uint32_t ssa = rctx->reg_to_ssa[i];
         unsigned n = rctx->ncomps[ssa];
         fprintf(fp, "h%u...%u: %u\n", i, i + n - 1, ssa);
         i += (n - 1);
      }
   }
   fprintf(fp, "\n");

   /* Dump a visualization of the sizes to understand what live range
    * splitting is up against.
    */
   for (unsigned i = 0; i < rctx->bound[RA_GPR]; ++i) {
      /* Space out 16-bit vec4s */
      if (i && (i % 4) == 0) {
         fprintf(fp, " ");
      }

      if (i < reserved) {
         fprintf(fp, "-");
      } else if (BITSET_TEST(rctx->used_regs[RA_GPR], i)) {
         uint32_t ssa = rctx->reg_to_ssa[i];
         unsigned n = rctx->ncomps[ssa];
         for (unsigned j = 0; j < n; ++j) {
            assert(n < 10);
            fprintf(fp, "%u", n);
         }

         i += (n - 1);
      } else {
         fprintf(fp, ".");
      }
   }
   fprintf(fp, "\n\n");
}

enum agx_size
agx_split_width(const agx_instr *I)
{
   enum agx_size width = ~0;

   agx_foreach_dest(I, d) {
      if (I->dest[d].type == AGX_INDEX_NULL)
         continue;
      else if (width != ~0)
         assert(width == I->dest[d].size);
      else
         width = I->dest[d].size;
   }

   assert(width != ~0 && "should have been DCE'd");
   return width;
}

/*
 * Calculate register demand in 16-bit registers, while gathering widths and
 * classes. Becuase we allocate in SSA, this calculation is exact in
 * linear-time. Depends on liveness information.
 */
static unsigned
agx_calc_register_demand(agx_context *ctx, bool remat)
{
   /* Print detailed demand calculation, helpful to debug spilling */
   bool debug = agx_compiler_debug & AGX_DBG_DEMAND;

   if (debug) {
      agx_print_shader(ctx, stdout);
   }

   uint8_t *widths = calloc(ctx->alloc, sizeof(uint8_t));
   enum ra_class *classes = calloc(ctx->alloc, sizeof(enum ra_class));

   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV_IMM && remat)
         continue;

      agx_foreach_ssa_dest(I, d) {
         unsigned v = I->dest[d].value;
         assert(widths[v] == 0 && "broken SSA");
         /* Round up vectors for easier live range splitting */
         widths[v] = util_next_power_of_two(agx_index_size_16(I->dest[d]));
         classes[v] = ra_class_for_index(I->dest[d]);
      }
   }

   /* Calculate demand at the start of each block based on live-in, then update
    * for each instruction processed. Calculate rolling maximum.
    */
   unsigned max_demand = 0;

   agx_foreach_block(ctx, block) {
      unsigned demand = reserved_size(ctx);

      /* Everything live-in */
      U_SPARSE_BITSET_FOREACH_SET(&block->live_in, i) {
         if (classes[i] == RA_GPR)
            demand += widths[i];
      }

      max_demand = MAX2(demand, max_demand);

      /* To handle late-kill sources, this counter tracks the number of
       * registers to be freed after 1 extra instruction.
       */
      unsigned late_kill_count = 0;

      if (debug) {
         printf("\n");
      }

      agx_foreach_instr_in_block(block, I) {
         /* Phis happen in parallel and are already accounted for in the live-in
          * set, just skip them so we don't double count.
          */
         if (I->op == AGX_OPCODE_PHI)
            continue;

         if (I->op == AGX_OPCODE_MOV_IMM && remat)
            continue;

         if (debug) {
            printf("%u: ", demand);
            agx_print_instr(I, stdout);
         }

         if (I->op == AGX_OPCODE_PRELOAD) {
            unsigned size = agx_size_align_16(I->src[0].size);
            max_demand = MAX2(max_demand, I->src[0].value + size);
         } else if (I->op == AGX_OPCODE_EXPORT) {
            unsigned size = agx_size_align_16(I->src[0].size);
            max_demand = MAX2(max_demand, I->imm + size);
         }

         /* Occupancy and rematerialization may shrink the register file.
          * A constrained operand still needs its physical range to exist,
          * even when very few SSA values are live at once.
          */
         if (I->reg_constraints) {
            for (unsigned o = 0; o < I->nr_dests + I->nr_srcs; ++o) {
               agx_index index = o < I->nr_dests
                                    ? I->dest[o]
                                    : I->src[o - I->nr_dests];
               if (index.type != AGX_INDEX_NORMAL || index.memory)
                  continue;
               struct agx_reg_constraint c = I->reg_constraints[o];
               unsigned width =
                  util_next_power_of_two(agx_index_size_16(index));
               unsigned first = ALIGN_POT(MAX2(c.min, reserved_size(ctx)),
                                         MAX2(width, c.align));
               max_demand = MAX2(max_demand, first + width);
            }
         }

         /* Handle late-kill registers from last instruction */
         demand -= late_kill_count;
         late_kill_count = 0;

         /* Late-kill sources the first time we see them. This simplifies RA. We
          * could optimize to early-kill in some situations if we wanted.
          */
         agx_foreach_src(I, s) {
            if (!I->src[s].kill)
               continue;
            assert(I->src[s].type == AGX_INDEX_NORMAL);
            if (ra_class_for_index(I->src[s]) != RA_GPR)
               continue;

            bool skip = false;

            for (unsigned backwards = 0; backwards < s; ++backwards) {
               if (agx_is_equiv(I->src[backwards], I->src[s])) {
                  skip = true;
                  break;
               }
            }

            if (!skip)
               late_kill_count += widths[I->src[s].value];
         }

         /* Make destinations live */
         agx_foreach_ssa_dest(I, d) {
            if (ra_class_for_index(I->dest[d]) != RA_GPR)
               continue;

            /* Live range splits allocate at power-of-two granularity. Round up
             * destination sizes (temporarily) to powers-of-two.
             */
            unsigned real_width = widths[I->dest[d].value];
            unsigned pot_width = util_next_power_of_two(real_width);

            demand += pot_width;
            late_kill_count += (pot_width - real_width);
         }

         max_demand = MAX2(demand, max_demand);
      }

      demand -= late_kill_count;
   }

   free(widths);
   free(classes);
   return max_demand;
}

/* Window sizes (1 << l halves) with a precomputed scarcity floor. */
#define RA_SCARCITY_FLOOR_SIZES 6

struct placement_cost {
   uint64_t dead, edge, affinity;
   unsigned moves, collect, preference, materialization, growth, scarcity, reg;
};

/* Collects retain their tuple placement preference separately from phi edges. */
static unsigned
collect_affinity(struct ra_ctx *rctx, unsigned ssa)
{
   unsigned want = ~0u;
   agx_instr *collect = rctx->src_to_collect[ssa];
   if (collect && collect->op == AGX_OPCODE_COLLECT) {
      unsigned ours = ~0u;
      agx_foreach_ssa_src(collect, s) {
         if (collect->src[s].value == ssa) {
            ours = s;
            break;
         }
      }
      unsigned width = agx_size_align_16(collect->src[ours].size);
      agx_foreach_ssa_src(collect, s) {
         unsigned other = collect->src[s].value;
         if (other == ssa || !BITSET_TEST(rctx->visited, other) ||
             rctx->ssa_to_reg[other] < s * width)
            continue;
         want = rctx->ssa_to_reg[other] - s * width + ours * width;
         break;
      }
      unsigned size = rctx->ncomps[collect->dest[0].value];
      for (unsigned base = ALIGN_POT(reserved_size(rctx->shader), size);
           want == ~0u && base + size <= rctx->bound[RA_GPR]; base += size) {
         if (!BITSET_TEST_COUNT(rctx->used_regs[RA_GPR], base, size))
            want = base + ours * width;
      }
   }
   return want;
}

static void
add_reg_preference(struct ra_ctx *rctx, unsigned reg, uint64_t edge,
                   uint64_t affinity, uint64_t dead)
{
   rctx->dead_total += dead;
   rctx->edge_total += edge;
   rctx->affinity_total += affinity;
   /* Spill-memory locations lie beyond the GPR index; search those. */
   if (reg >= AGX_NUM_REGS) {
      util_dynarray_foreach(rctx->reg_preferences, struct phi_reg_preference, p) {
         if (p->reg == reg) {
            p->dead += dead;
            p->edge += edge;
            p->affinity += affinity;
            return;
         }
      }
   } else {
      int16_t index = rctx->reg_preference_index[reg];
      if (index >= 0) {
         struct phi_reg_preference *p = util_dynarray_element(
            rctx->reg_preferences, struct phi_reg_preference, index);
         p->dead += dead;
         p->edge += edge;
         p->affinity += affinity;
         return;
      }
      rctx->reg_preference_index[reg] = util_dynarray_num_elements(
         rctx->reg_preferences, struct phi_reg_preference);
   }
   util_dynarray_append(rctx->reg_preferences,
                       ((struct phi_reg_preference){reg, dead, edge, affinity}));
}

static void
cache_phi_preferences(struct ra_ctx *rctx, unsigned ssa)
{
   if (rctx->affinity_ssa == ssa)
      return;

   util_dynarray_foreach(rctx->reg_preferences, struct phi_reg_preference, p) {
      if (p->reg < AGX_NUM_REGS)
         rctx->reg_preference_index[p->reg] = -1;
   }
   util_dynarray_clear(rctx->reg_preferences);
   rctx->dead_total = rctx->edge_total = rctx->affinity_total = 0;
   for (struct phi_preference *p = rctx->phi_preferences[ssa]; p; p = p->next) {
      if (p->other->has_reg)
         add_reg_preference(rctx, p->other->reg, p->weight, 0, 0);
   }

   /* Transitive class history helps before the other end of a direct edge has
    * been assigned. Retain every location, and refresh values when RA moves
    * them, rather than permanently pinning the class to its first assignment.
    * Direct edge matches take priority over this speculative history.
    */
   unsigned root = phi_web_find(rctx->phi_web, ssa);
   for (unsigned v = rctx->web_members[root].head; v != ~0u;
        v = rctx->web_members[v].next) {
      if (v >= rctx->shader->alloc) {
         struct phi_preference *p = rctx->web_members[v].constant;
         if (p && p->other->has_reg)
            add_reg_preference(rctx, p->other->reg, 0, p->weight, 0);
      } else if (v != ssa && rctx->last_reg[v] != ~0u) {
         add_reg_preference(rctx, rctx->last_reg[v], 0, rctx->phi_weights[v], 0);
      }
   }
   for (struct phi_dead_group *g = rctx->dead_groups[ssa]; g; g = g->next) {
      BITSET_DECLARE(locations, AGX_NUM_REGS) = {0};
      for (struct phi_dead_destination *d = g->destinations; d; d = d->next) {
         if (d->index->has_reg) {
            BITSET_SET(locations, d->index->reg);
            continue;
         }

         /* Forward-edge destinations have not been assigned yet. Use their
          * compatible class history as a prediction, without reserving it. */
         unsigned root = phi_web_find(rctx->phi_web, d->index->value);
         for (unsigned v = rctx->web_members[root].head; v != ~0u;
              v = rctx->web_members[v].next) {
            if (v < rctx->shader->alloc && v != ssa &&
                rctx->last_reg[v] != ~0u)
               BITSET_SET(locations, rctx->last_reg[v]);
         }
      }
      unsigned reg;
      BITSET_FOREACH_SET(reg, locations, AGX_NUM_REGS)
         add_reg_preference(rctx, reg, 0, 0, g->weight);
   }
   rctx->affinity_reg = collect_affinity(rctx, ssa);
   rctx->affinity_ssa = ssa;
}

static unsigned
preferred_phi_register(struct ra_ctx *rctx, unsigned ssa, bool available)
{
   cache_phi_preferences(rctx, ssa);
   enum ra_class cls = rctx->classes[ssa];
   unsigned count = rctx->ncomps[ssa], best = ~0u;
   uint64_t dead = 0, edge = 0, affinity = 0;
   util_dynarray_foreach(rctx->reg_preferences, struct phi_reg_preference, p) {
      if (p->reg % count || p->reg + count > rctx->bound[cls] ||
          (available && BITSET_TEST_COUNT(rctx->used_regs[cls], p->reg, count)))
         continue;
      if (best == ~0u || p->edge > edge ||
          (p->edge == edge && (p->dead > dead ||
           (p->dead == dead && (p->affinity > affinity ||
            (p->affinity == affinity && p->reg < best)))))) {
         best = p->reg;
         dead = p->dead;
         edge = p->edge;
         affinity = p->affinity;
      }
   }
   return best;
}

static struct placement_cost
placement_cost(struct ra_ctx *rctx, unsigned ssa, unsigned reg, unsigned count)
{
   struct placement_cost cost = {.reg = reg};
   cache_phi_preferences(rctx, ssa);
   cost.dead = rctx->dead_total;
   cost.edge = rctx->edge_total;
   cost.affinity = rctx->affinity_total;
   const struct phi_reg_preference *pref = NULL;
   if (reg < AGX_NUM_REGS) {
      int16_t index = rctx->reg_preference_index[reg];
      if (index >= 0)
         pref = util_dynarray_element(rctx->reg_preferences,
                                      struct phi_reg_preference, index);
   } else {
      util_dynarray_foreach(rctx->reg_preferences, struct phi_reg_preference, p) {
         if (p->reg == reg) {
            pref = p;
            break;
         }
      }
   }
   if (pref) {
      cost.dead -= pref->dead;
      cost.edge -= pref->edge;
      cost.affinity -= pref->affinity;
   }
   cost.collect = rctx->affinity_reg != ~0u && rctx->affinity_reg != reg;
   if (!rctx->preferred)
      return cost;
   struct agx_reg_constraint c = rctx->preferred[ssa];
   if (c.min <= c.max) {
      cost.preference = reg < c.min || reg > c.max || (reg % c.align);
   }

   /* A constant outside the immediate destination bank needs a staging
    * move. Prefer a directly encodable free register after honouring phi,
    * tuple and operand preferences, but before conserving scarce banks. */
   unsigned immediate_regs = rctx->shader->ra_target.mov_imm_registers;
   if (immediate_regs && rctx->instr->op == AGX_OPCODE_MOV_IMM &&
       rctx->instr->dest[0].value == ssa)
      cost.materialization = reg + count > immediate_regs;

   /* Leave scarce banks available without raising the footprint beyond the
    * pressure already required by this program (or an existing fixed use). */
   unsigned limit = MAX2(rctx->demand, *rctx->count[RA_GPR]);
   cost.growth = reg + count > limit ? reg + count - limit : 0;
   for (unsigned h = reg; h < reg + count; ++h)
      cost.scarcity += rctx->scarcity[h];
   return cost;
}

/* A lower bound of placement_cost for every register from reg upwards, given
 * per-field floors over the phi preferences and the highest preferred GPR.
 * Each field it raises only grows with the register: the tuple affinity is
 * one register, the preferred window ends at its maximum, and immediate
 * encodability and footprint growth are thresholds on the register's end. */
static struct placement_cost
placement_floor(struct ra_ctx *rctx, unsigned ssa, unsigned reg, unsigned count,
                const struct placement_cost *floors, unsigned last_preference)
{
   struct placement_cost cost = *floors;
   cost.reg = reg;
   if (reg > last_preference || last_preference == ~0u) {
      cost.dead = rctx->dead_total;
      cost.edge = rctx->edge_total;
      cost.affinity = rctx->affinity_total;
   }
   cost.collect = rctx->affinity_reg != ~0u && rctx->affinity_reg < reg;
   if (!rctx->preferred)
      return cost;
   struct agx_reg_constraint c = rctx->preferred[ssa];
   cost.preference = c.min <= c.max && reg > c.max;
   unsigned immediate_regs = rctx->shader->ra_target.mov_imm_registers;
   if (immediate_regs && rctx->instr->op == AGX_OPCODE_MOV_IMM &&
       rctx->instr->dest[0].value == ssa)
      cost.materialization = reg + count > immediate_regs;
   unsigned limit = MAX2(rctx->demand, *rctx->count[RA_GPR]);
   cost.growth = reg + count > limit ? reg + count - limit : 0;
   unsigned l = util_logbase2(count);
   if (rctx->scarcity_floor && l < RA_SCARCITY_FLOOR_SIZES && reg <= AGX_NUM_REGS)
      cost.scarcity = rctx->scarcity_floor[l][reg];
   return cost;
}

static bool
placement_cheaper(struct placement_cost a, struct placement_cost b)
{
   if (a.moves != b.moves) return a.moves < b.moves;
   if (a.edge != b.edge) return a.edge < b.edge;
   if (a.dead != b.dead) return a.dead < b.dead;
   if (a.affinity != b.affinity) return a.affinity < b.affinity;
   if (a.collect != b.collect) return a.collect < b.collect;
   if (a.preference != b.preference) return a.preference < b.preference;
   if (a.materialization != b.materialization)
      return a.materialization < b.materialization;
   if (a.growth != b.growth) return a.growth < b.growth;
   if (a.scarcity != b.scarcity) return a.scarcity < b.scarcity;
   return a.reg < b.reg;
}

static bool
find_regs_simple(struct ra_ctx *rctx, enum ra_class cls, unsigned count,
                 unsigned align, unsigned ssa, unsigned *out)
{
   bool found = false;
   struct placement_cost best = {.moves = ~0u};
   rctx->affinity_ssa = ~0u;
   for (unsigned reg = 0; reg + count <= rctx->bound[cls]; reg += align) {
      if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, count)) {
         if (!rctx->preferred || cls != RA_GPR) {
            *out = reg;
            return true;
         }
         struct placement_cost cost = placement_cost(rctx, ssa, reg, count);
         if (!found || placement_cheaper(cost, best)) {
            *out = reg;
            best = cost;
            found = true;
         }
      }
   }

   return found;
}

/*
 * Search the register file for the best contiguous aligned region of the given
 * size to evict when shuffling registers. The region must not contain any
 * register marked in the passed bitset.
 *
 * As a hint, this also takes in the set of registers from killed sources passed
 * to this instruction. These should be deprioritized, since they are more
 * expensive to use (extra moves to shuffle the contents away).
 *
 * Precondition: such a region exists.
 *
 * Postcondition: at least one register in the returned region is already free.
 */
static unsigned
find_best_region_to_evict(struct ra_ctx *rctx, unsigned size,
                          BITSET_WORD *already_evicted)
{
   assert(util_is_power_of_two_or_zero(size) && "precondition");
   assert((rctx->bound[RA_GPR] % size) == 0 &&
          "register file size must be aligned to the maximum vector size");

   /* Useful for testing RA */
   bool invert = false;

   unsigned best_base = ~0;
   unsigned best_moves = invert ? 0 : ~0;

   for (unsigned base = 0; base + size <= rctx->bound[RA_GPR]; base += size) {
      /* The first k registers are preallocated and unevictable, so must be
       * skipped. By itself, this does not pose a problem. We are allocating n
       * registers, but this region has at most n-k free.  Since there are at
       * least n free registers total, there is at least k free registers
       * outside this region. Choose any such free register. The region
       * containing it has at most n-1 occupied registers. In the worst case,
       * n-k of those registers are are moved to the beginning region and the
       * remaining (n-1)-(n-k) = k-1 registers are moved to the k-1 free
       * registers in other regions, given there are k free registers total.
       * These recursive shuffles work out because everything is power-of-two
       * sized and naturally aligned, so the sizes shuffled are strictly
       * descending. So, we do not need extra registers to handle "single
       * region" unevictability.
       */
      if (base < reserved_size(rctx->shader))
         continue;

      if (rctx->pinned && BITSET_TEST_COUNT(rctx->pinned, base, size))
         continue;

      /* Do not evict the same register multiple times. It's not necessary since
       * we're just shuffling, there are enough free registers elsewhere.
       */
      if (BITSET_TEST_COUNT(already_evicted, base, size))
         continue;

      /* Estimate the number of moves required if we pick this region */
      unsigned moves = 0;
      bool any_free = false;

      for (unsigned reg = base; reg < base + size; ++reg) {
         /* We need a move for each blocked register (TODO: we only need a
          * single move for 32-bit pairs, could optimize to use that instead.)
          */
         if (BITSET_TEST(rctx->used_regs[RA_GPR], reg))
            moves++;
         else
            any_free = true;
      }

      /* Pick the region requiring fewest moves as a heuristic. Regions with no
       * free registers are skipped, since the recursive splitting algorithm
       * requires at least one free register.
       */
      if (any_free && ((moves < best_moves) ^ invert)) {
         best_moves = moves;
         best_base = base;
      }
   }

   assert(best_base < rctx->bound[RA_GPR] &&
          "not enough registers (should have spilled already)");
   return best_base;
}

static void
set_ssa_to_reg(struct ra_ctx *rctx, unsigned ssa, unsigned reg)
{
   enum ra_class cls = rctx->classes[ssa];
   *(rctx->count[cls]) = MAX2(*(rctx->count[cls]), reg + rctx->ncomps[ssa]);

   if (rctx->moved && cls == RA_GPR) {
      /* Log values live before the instruction (not its destinations)
       * above the reserved prefix, the values copied back afterwards. */
      bool record = rctx->ssa_to_reg[ssa] >= reserved_size(rctx->shader);
      agx_foreach_ssa_dest(rctx->instr, d)
         record &= rctx->instr->dest[d].value != ssa;
      for (unsigned i = 0; record && i < rctx->nr_moved; ++i)
         record = rctx->moved[i].ssa != ssa;
      if (record) {
         assert(rctx->nr_moved < AGX_NUM_REGS);
         rctx->moved[rctx->nr_moved++] =
            (struct ra_move){ssa, rctx->ssa_to_reg[ssa]};
      }
   }
   rctx->ssa_to_reg[ssa] = reg;
   rctx->last_reg[ssa] = reg;
   rctx->affinity_ssa = ~0u;

   if (cls == RA_GPR) {
      rctx->reg_to_ssa[reg] = ssa;
   }
}

/*
 * Insert parallel copies to move an SSA variable `var` to a new register
 * `new_reg`. This may require scalarizing.
 */
static void
insert_copy(struct ra_ctx *rctx, struct util_dynarray *copies, unsigned new_reg,
            unsigned var)
{
   enum agx_size size = rctx->sizes[var];
   unsigned align = agx_size_align_16(size);

   for (unsigned i = 0; i < rctx->ncomps[var]; i += align) {
      struct agx_copy copy = {
         .dest = new_reg + i,
         .src = agx_register(rctx->ssa_to_reg[var] + i, size),
      };

      assert((copy.dest % align) == 0 && "new dest must be aligned");
      assert((copy.src.value % align) == 0 && "src must be aligned");
      util_dynarray_append(copies, copy);
   }
}

static unsigned
assign_regs_by_copying(struct ra_ctx *rctx, agx_index dest, const agx_instr *I,
                       struct util_dynarray *copies)
{
   BITSET_DECLARE(clobbered, AGX_NUM_REGS) = {0};
   assert(dest.type == AGX_INDEX_NORMAL);

   /* Initialize the worklist with the variable we're assigning */
   unsigned blocked_vars[16] = {dest.value};
   size_t nr_blocked = 1;

   while (nr_blocked > 0) {
      /* Grab the largest var. TODO: Consider not writing O(N^2) code. */
      uint32_t ssa = ~0, nr = 0, chosen_idx = ~0;
      for (unsigned i = 0; i < nr_blocked; ++i) {
         uint32_t this_ssa = blocked_vars[i];
         uint32_t this_nr = rctx->ncomps[this_ssa];

         if (this_nr > nr) {
            nr = this_nr;
            ssa = this_ssa;
            chosen_idx = i;
         }
      }

      assert(ssa != ~0 && nr > 0 && "must have found something");
      assert(chosen_idx < nr_blocked && "must have found something");

      /* Pop it from the work list by swapping in the last element */
      blocked_vars[chosen_idx] = blocked_vars[--nr_blocked];

      /* We need to shuffle some variables to make room. Look for a range of
       * the register file that is partially blocked.
       */
      unsigned new_reg = find_best_region_to_evict(rctx, nr, clobbered);

      /* Blocked registers need to get reassigned. Add them to the worklist. */
      for (unsigned i = 0; i < nr; ++i) {
         if (BITSET_TEST(rctx->used_regs[RA_GPR], new_reg + i)) {
            unsigned blocked_reg = new_reg + i;
            uint32_t blocked_ssa = rctx->reg_to_ssa[blocked_reg];
            uint32_t blocked_nr = rctx->ncomps[blocked_ssa];

            assert(blocked_nr >= 1 && "must be assigned");

            blocked_vars[nr_blocked++] = blocked_ssa;
            assert(
               rctx->ssa_to_reg[blocked_ssa] == blocked_reg &&
               "variable must start within the range, since vectors are limited");

            for (unsigned j = 0; j < blocked_nr; ++j) {
               assert(
                  BITSET_TEST(rctx->used_regs[RA_GPR], new_reg + i + j) &&
                  "variable is allocated contiguous and vectors are limited, "
                  "so evicted in full");
            }

            /* Skip to the next variable */
            i += blocked_nr - 1;
         }
      }

      /* We are going to allocate to this range, so it is now fully used. Mark
       * it as such so we don't reassign here later.
       */
      BITSET_SET_COUNT(rctx->used_regs[RA_GPR], new_reg, nr);

      /* The first iteration is special: it is the original allocation of a
       * variable. All subsequent iterations pick a new register for a blocked
       * variable. For those, copy the blocked variable to its new register.
       */
      if (ssa != dest.value) {
         insert_copy(rctx, copies, new_reg, ssa);
      }

      /* Mark down the set of clobbered registers, so that killed sources may be
       * handled correctly later.
       */
      BITSET_SET_COUNT(clobbered, new_reg, nr);

      /* Update bookkeeping for this variable */
      set_ssa_to_reg(rctx, ssa, new_reg);
   }

   return rctx->ssa_to_reg[dest.value];
}

static int
sort_by_size(const void *a_, const void *b_, void *sizes_)
{
   const enum agx_size *sizes = sizes_;
   const unsigned *a = a_, *b = b_;

   return sizes[*b] - sizes[*a];
}

/*
 * When shuffling registers to assign a phi destination, we can't simply insert
 * the required moves before the phi, since phis happen in parallel along the
 * edge. Instead, there are two cases:
 *
 * 1. The source of the copy is the destination of a phi. Since we are
 *    emitting shuffle code, there will be no more reads of that destination
 *    with the old register. Since the phis all happen in parallel and writes
 *    precede reads, there was no previous read of that destination either. So
 *    the old destination is dead. Just replace the phi's destination with the
 *    moves's destination instead.
 *
 * 2. Otherwise, the source of the copy is a live-in value, since it's
 *    live when assigning phis at the start of a block but it is not a phi.
 *    If we move in parallel with the phi, the phi will still read the correct
 *    old register regardless and the destinations can't alias. So, insert a phi
 *    to do the copy in parallel along the incoming edges.
 */
static void
agx_emit_move_before_phi(agx_context *ctx, agx_block *block,
                         struct agx_copy *copy)
{
   assert(!copy->dest_mem && !copy->src.memory && "no memory shuffles");

   /* Look for the phi writing the destination */
   agx_foreach_phi_in_block(block, phi) {
      if (agx_is_equiv(agx_as_register(phi->dest[0]), copy->src) &&
          !phi->dest[0].memory) {

         phi->dest[0].reg = copy->dest;
         return;
      }
   }

   /* There wasn't such a phi, so it's live-in. Insert a phi instead. */
   agx_builder b = agx_init_builder(ctx, agx_before_block(block));

   agx_instr *phi = agx_phi_to(&b, agx_register_like(copy->dest, copy->src),
                               agx_num_predecessors(block));
   pressure_tag_phi(ctx, phi, "eviction-reconcile");
   assert(!copy->src.kill);

   agx_foreach_src(phi, s) {
      phi->src[s] = copy->src;
   }
}

static unsigned
find_regs(struct ra_ctx *rctx, agx_instr *I, unsigned dest_idx, unsigned count,
          unsigned align)
{
   unsigned reg;
   assert(count == align);

   enum ra_class cls = ra_class_for_index(I->dest[dest_idx]);

   if (find_regs_simple(rctx, cls, count, align, I->dest[dest_idx].value, &reg)) {
      return reg;
   } else {
      assert(!rctx->early_killed && "no live range splits with early kill");
      assert(cls == RA_GPR && "no memory live range splits");

      struct util_dynarray copies = UTIL_DYNARRAY_INIT;

      reg = assign_regs_by_copying(rctx, I->dest[dest_idx], I, &copies);

      /* Insert the necessary copies. Phis need special handling since we can't
       * insert instructions before the phi.
       */
      if (I->op == AGX_OPCODE_PHI) {
         util_dynarray_foreach(&copies, struct agx_copy, copy) {
            agx_emit_move_before_phi(rctx->shader, rctx->block, copy);
         }
      } else {
         agx_builder b = agx_init_builder(rctx->shader, agx_before_instr(I));
         agx_emit_parallel_copies(
            &b, copies.data,
            util_dynarray_num_elements(&copies, struct agx_copy));
      }

      util_dynarray_fini(&copies);

      /* assign_regs asserts this is cleared, so clear to be reassigned */
      BITSET_CLEAR_COUNT(rctx->used_regs[cls], reg, count);
      return reg;
   }
}

static uint32_t
search_ssa_to_reg_out(struct agx_block *blk, unsigned ssa)
{
   unsigned lo = 0, hi = blk->nr_ssa_to_reg_out;

   while (lo < hi) {
      unsigned mid = lo + (hi - lo) / 2;
      if (blk->ssa_to_reg_out[mid].ssa < ssa)
         lo = mid + 1;
      else
         hi = mid;
   }

   assert(lo < blk->nr_ssa_to_reg_out &&
          blk->ssa_to_reg_out[lo].ssa == ssa && "variable not defined in block");
   return blk->ssa_to_reg_out[lo].reg;
}

/*
 * Loop over live-in values at the start of the block and mark their registers
 * as in-use. We process blocks in dominance order, so this handles everything
 * but loop headers.
 *
 * For loop headers, this handles the forward edges but not the back edge.
 * However, that's okay: we don't want to reserve the registers that are
 * defined within the loop, because then we'd get a contradiction. Instead we
 * leave them available and then they become fixed points of a sort.
 */
static void
reserve_live_in(struct ra_ctx *rctx)
{
   /* If there are no predecessors, there is nothing live-in */
   unsigned nr_preds = agx_num_predecessors(rctx->block);
   if (nr_preds == 0)
      return;

   agx_builder b =
      agx_init_builder(rctx->shader, agx_before_block(rctx->block));

   U_SPARSE_BITSET_FOREACH_SET(&rctx->block->live_in, i) {
      /* Skip values defined in loops when processing the loop header */
      if (!BITSET_TEST(rctx->visited, i))
         continue;

      unsigned base;
      enum ra_class cls = rctx->classes[i];
      enum agx_size size = rctx->sizes[i];

      /* We need to use the unrounded channel count, since the extra padding
       * will be uninitialized and would fail RA validation.
       */
      unsigned channels = rctx->ncomps_unrounded[i] / agx_size_align_16(size);

      /* If we split live ranges, the variable might be defined differently at
       * the end of each predecessor. Join them together with a phi inserted at
       * the start of the block.
       */
      if (nr_preds > 1) {
         /* We'll fill in the destination after, to coalesce one of the moves */
         agx_instr *phi = agx_phi_to(&b, agx_null(), nr_preds);
         pressure_tag_phi(rctx->shader, phi, "live-in-reconcile");

         agx_foreach_predecessor(rctx->block, pred) {
            unsigned pred_idx = agx_predecessor_index(rctx->block, *pred);

            phi->src[pred_idx] = agx_get_vec_index(i, size, channels);
            phi->src[pred_idx].memory = cls == RA_MEM;

            if ((*pred)->ssa_to_reg_out == NULL) {
               /* If this is a loop header, we don't know where the register
                * will end up. So, we create a phi conservatively but don't fill
                * it in until the end of the loop. Stash in the information
                * we'll need to fill in the real register later.
                */
               assert(rctx->block->loop_header);
            } else {
               /* Otherwise, we can build the phi now */
               phi->src[pred_idx].reg =
                  search_ssa_to_reg_out(*pred, i);
               phi->src[pred_idx].has_reg = true;
            }
         }

         /* Pick the phi destination to coalesce a move. Predecessor ordering is
          * stable, so this means all live-in values get their registers from a
          * particular predecessor. That means that such a register allocation
          * is valid here, because it was valid in the predecessor.
          */
         assert(phi->src[0].has_reg && "not loop source");
         phi->dest[0] = phi->src[0];
         base = phi->dest[0].reg;
      } else {
         /* If we don't emit a phi, there is already a unique register */
         assert(nr_preds == 1);

         agx_block **pred = util_dynarray_begin(&rctx->block->predecessors);
         base = search_ssa_to_reg_out(*pred, i);
      }

      set_ssa_to_reg(rctx, i, base);

      for (unsigned j = 0; j < rctx->ncomps[i]; ++j) {
         BITSET_SET(rctx->used_regs[cls], base + j);
      }
   }
}

static void
assign_regs(struct ra_ctx *rctx, agx_index v, unsigned reg)
{
   enum ra_class cls = ra_class_for_index(v);
   pressure_web_event(rctx, v.value, "ASSIGN", reg, NULL);
   assert(reg < rctx->bound[cls] && "must not overflow register file");
   assert(v.type == AGX_INDEX_NORMAL && "only SSA gets registers allocated");
   set_ssa_to_reg(rctx, v.value, reg);

   assert(!BITSET_TEST(rctx->visited, v.value) && "SSA violated");
   BITSET_SET(rctx->visited, v.value);

   unsigned nr = rctx->ncomps[v.value];
   assert(nr >= 1);
   assert(!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, nr) &&
          "no interference");

   BITSET_SET_COUNT(rctx->used_regs[cls], reg, nr);
}

static void
relocate_unconstrained(struct ra_ctx *rctx, unsigned ssa)
{
   unsigned reg, count = rctx->ncomps[ssa];
   if (find_regs_simple(rctx, RA_GPR, count, count, ssa, &reg)) {
      BITSET_SET_COUNT(rctx->used_regs[RA_GPR], reg, count);
      set_ssa_to_reg(rctx, ssa, reg);
   } else {
      /* This transaction emits one parallel copy from its original mapping
       * after all relocations, including recursive live-range splits. */
      struct util_dynarray ignored = UTIL_DYNARRAY_INIT;
      agx_index value = agx_get_vec_index(
         ssa, rctx->sizes[ssa],
         rctx->ncomps_unrounded[ssa] / agx_size_align_16(rctx->sizes[ssa]));
      assign_regs_by_copying(rctx, value, rctx->instr, &ignored);
      util_dynarray_fini(&ignored);
   }
}

/* Validate before rounding or subtracting unsigned register bounds. Fixed
 * source copies into the reserved prefix are checked separately by the caller. */
static bool
constraint_window(struct agx_reg_constraint c, unsigned count,
                  unsigned reserved, unsigned bound,
                  unsigned *first, unsigned *last)
{
   if (!util_is_power_of_two_nonzero(count) ||
       !util_is_power_of_two_nonzero(c.align) || c.min > c.max || count > bound)
      return false;

   unsigned align = MAX2(count, c.align);
   *first = ALIGN_POT(MAX2(c.min, reserved), align);
   *last = MIN2(c.max, bound - count);
   return *first <= *last;
}

static bool
validate_constraint_windows(agx_context *ctx, unsigned bound)
{
   agx_foreach_instr_global(ctx, I) {
      if (!I->reg_constraints)
         continue;
      for (unsigned o = 0; o < I->nr_dests + I->nr_srcs; ++o) {
         agx_index value = o < I->nr_dests ? I->dest[o] : I->src[o - I->nr_dests];
         if (value.type != AGX_INDEX_NORMAL || value.memory)
            continue;
         struct agx_reg_constraint c = I->reg_constraints[o];
         unsigned count = util_next_power_of_two(agx_index_size_16(value));
         unsigned reserved = reserved_size(ctx), first, last;
         if (o >= I->nr_dests && c.min == c.max && c.max < reserved) {
            /* A fixed copy must fit entirely inside the reserved prefix. */
            if (c.align && util_is_power_of_two_nonzero(c.align) &&
                !(c.min % MAX2(count, c.align)) && count <= reserved - c.min)
               continue;
         } else if (constraint_window(c, count, reserved, bound, &first, &last)) {
            continue;
         }
         ctx->ra_error = "Register operand has no aligned allocatable window";
         return false;
      }
   }
   return true;
}

static unsigned
place_constrained_value(struct ra_ctx *rctx, agx_index value,
                        struct agx_reg_constraint c, bool defined)
{
   unsigned ssa = value.value, count = rctx->ncomps[ssa];
   unsigned align = MAX2(count, c.align);
   unsigned first, last;
   if (!constraint_window(c, count, reserved_size(rctx->shader),
                          rctx->bound[RA_GPR], &first, &last)) {
      rctx->shader->ra_error = "Register operand has no aligned allocatable window";
      return ~0u;
   }

   rctx->affinity_ssa = ~0u;

   /* Each reusable tuple lies inside one candidate window, so a destination
    * there replaces it without moves. Elsewhere it stays occupied. */
   BITSET_WORD *reusable = defined ? NULL : rctx->reusable;
   if (reusable) {
      for (unsigned i = 0; i < BITSET_WORDS(AGX_NUM_REGS); ++i) {
         rctx->used_regs[RA_GPR][i] &= ~reusable[i];
         rctx->pinned[i] &= ~reusable[i];
      }
   }

   if (!defined)
      pressure_web_event(rctx, ssa, "CHOOSE", ~0u, &c);
   if (defined) {
      unsigned reg = rctx->ssa_to_reg[ssa];
      if (reg >= first && reg <= last && !(reg % align)) {
         BITSET_SET_COUNT(rctx->pinned, reg, count);
         return reg;
      }
   }

   /* Unlike unconstrained splitting, the legal bank may be completely full.
    * Choose a legal region, then move its occupants to the rest of the file.
    * Only this instruction's already-placed operands are pinned. */
   /* The base of the tuple occupying each register, recorded once and only
    * when an occupied candidate is costed, so each candidate inspects only
    * its own registers. */
   unsigned tuple_base[AGX_NUM_REGS];
   bool have_tuple_base = false;

   unsigned chosen = ~0;
   struct placement_cost best = {.moves = ~0u};
   /* Per-field lower bounds of the ranked cost. Once the best candidate is
    * cheaper than the bound for every remaining register, none of them can
    * win (ties keep the lower register), so the search stops without
    * changing the result. Phi preferences lower the first three fields only
    * up to the highest preferred register. */
   cache_phi_preferences(rctx, ssa);
   uint64_t max_edge = 0, max_dead = 0, max_affinity = 0;
   unsigned last_preference = ~0u;
   util_dynarray_foreach(rctx->reg_preferences, struct phi_reg_preference, p) {
      max_edge = MAX2(max_edge, p->edge);
      max_dead = MAX2(max_dead, p->dead);
      max_affinity = MAX2(max_affinity, p->affinity);
      if (p->reg < AGX_NUM_REGS &&
          (last_preference == ~0u || p->reg > last_preference))
         last_preference = p->reg;
   }
   const struct placement_cost floors = {
      .edge = rctx->edge_total - max_edge,
      .dead = rctx->dead_total - max_dead,
      .affinity = rctx->affinity_total - max_affinity,
   };
   /* Moves rank first: when any aligned window is free, only free windows
    * can win, so occupied ones need no costing. */
   bool any_free = false;
   for (unsigned reg = first; reg <= last && !any_free; reg += align) {
      any_free = !BITSET_TEST_COUNT(rctx->pinned, reg, count) &&
                 !BITSET_TEST_COUNT(rctx->used_regs[RA_GPR], reg, count);
   }
   for (unsigned reg = first; reg <= last; reg += align) {
      if (chosen != ~0u &&
          placement_cheaper(best, placement_floor(rctx, ssa, reg, count, &floors,
                                                  last_preference)))
         break;
      if (BITSET_TEST_COUNT(rctx->pinned, reg, count))
         continue;
      /* Moves rank first, so an occupied region never beats a free one. */
      bool occupied = BITSET_TEST_COUNT(rctx->used_regs[RA_GPR], reg, count);
      if (occupied && (any_free || (chosen != ~0u && !best.moves)))
         continue;
      struct placement_cost cost = placement_cost(rctx, ssa, reg, count);
      if (occupied && rctx->preferred && !have_tuple_base) {
         for (unsigned h = reserved_size(rctx->shader); h < rctx->bound[RA_GPR];) {
            if (!BITSET_TEST(rctx->used_regs[RA_GPR], h)) {
               ++h;
               continue;
            }
            unsigned width = rctx->ncomps[rctx->reg_to_ssa[h]];
            for (unsigned k = 0; k < width; ++k)
               tuple_base[h + k] = h;
            h += width;
         }
         have_tuple_base = true;
      }
      if (!occupied) {
         /* No register to move. */
      } else if (rctx->preferred) {
         /* Even a one-half overlap evicts the complete tuple. Counting only
          * the overlapping halves can prefer a much more expensive move. */
         unsigned previous = ~0u;
         for (unsigned h = reg; h < reg + count; ++h) {
            if (!BITSET_TEST(rctx->used_regs[RA_GPR], h) ||
                tuple_base[h] == previous)
               continue;
            previous = tuple_base[h];
            cost.moves += rctx->ncomps[rctx->reg_to_ssa[previous]];
         }
      } else {
         for (unsigned h = reg; h < reg + count; ++h)
            cost.moves += BITSET_TEST(rctx->used_regs[RA_GPR], h);
      }
      if (chosen == ~0u || placement_cheaper(cost, best)) {
         best = cost;
         chosen = reg;
      }
   }
   if (chosen == ~0u) {
      rctx->shader->ra_error = "Instruction register constraints are unsatisfiable";
      return ~0u;
   }

   /* A free region has no occupants to move. */
   unsigned victims[16], nr_victims = 0;
   for (unsigned reg = best.moves ? reserved_size(rctx->shader) : rctx->bound[RA_GPR];
        reg < rctx->bound[RA_GPR];) {
      if (!BITSET_TEST(rctx->used_regs[RA_GPR], reg)) {
         ++reg;
         continue;
      }
      unsigned owner = rctx->reg_to_ssa[reg];
      unsigned width = rctx->ncomps[owner];
      assert(rctx->ssa_to_reg[owner] == reg && width);
      if (reg < chosen + count && chosen < reg + width) {
         assert(nr_victims < ARRAY_SIZE(victims));
         victims[nr_victims++] = owner;
      }
      reg += width;
   }
   for (unsigned i = 0; i < nr_victims; ++i) {
      unsigned owner = victims[i];
      BITSET_CLEAR_COUNT(rctx->used_regs[RA_GPR], rctx->ssa_to_reg[owner],
                         rctx->ncomps[owner]);
   }
   if (defined)
      BITSET_CLEAR_COUNT(rctx->used_regs[RA_GPR], rctx->ssa_to_reg[ssa], count);

   BITSET_SET_COUNT(rctx->pinned, chosen, count);
   if (defined) {
      BITSET_SET_COUNT(rctx->used_regs[RA_GPR], chosen, count);
      set_ssa_to_reg(rctx, ssa, chosen);
   } else {
      assign_regs(rctx, value, chosen);
   }
   if (reusable) {
      for (unsigned i = 0; i < BITSET_WORDS(AGX_NUM_REGS); ++i) {
         rctx->used_regs[RA_GPR][i] |= reusable[i];
         rctx->pinned[i] |= reusable[i];
      }
   }

   /* Largest first, matching the power-of-two live-range splitter. */
   while (nr_victims) {
      unsigned largest = 0;
      for (unsigned i = 1; i < nr_victims; ++i) {
         if (rctx->ncomps[victims[i]] > rctx->ncomps[victims[largest]])
            largest = i;
      }
      unsigned victim = victims[largest];
      victims[largest] = victims[--nr_victims];
      relocate_unconstrained(rctx, victim);
   }
   return chosen;
}

static bool
assign_constrained_operands(struct ra_ctx *rctx, agx_instr *I)
{
   BITSET_DECLARE(pinned, AGX_NUM_REGS) = {0};
   rctx->pinned = pinned;
   /* Values usually stay put, so log the original register of those that
    * move instead of recording every owner up front. */
   struct ra_move moved[AGX_NUM_REGS];
   rctx->moved = moved;
   rctx->nr_moved = 0;

   agx_foreach_ssa_src(I, s) {
      struct agx_reg_constraint c = I->reg_constraints[I->nr_dests + s];
      if (c.min == c.max && c.max < reserved_size(rctx->shader))
         continue;

      /* Repeated uses of one SSA value must satisfy all their constraints. */
      bool seen = false;
      agx_foreach_ssa_src(I, t) {
         if (!agx_is_equiv(I->src[s], I->src[t]))
            continue;
         struct agx_reg_constraint other = I->reg_constraints[I->nr_dests + t];
         if (other.min == other.max && other.max < reserved_size(rctx->shader))
            continue;
         seen |= t < s;
         c.min = MAX2(c.min, other.min);
         c.max = MIN2(c.max, other.max);
         c.align = MAX2(c.align, other.align);
      }
      if (seen)
         continue;
      /* Unrestricted sources may move while placing other operands. */
      if (c.min <= reserved_size(rctx->shader) &&
          c.max >= rctx->bound[RA_GPR] - rctx->ncomps[I->src[s].value] &&
          c.align <= rctx->ncomps[I->src[s].value])
         continue;
      if (place_constrained_value(rctx, I->src[s], c, true) == ~0u) {
         rctx->moved = NULL;
         return false;
      }
   }
   /* Sources dying here, other than destroyed or fixed-register copies,
    * whose tuple fits the destination's aligned window. */
   BITSET_DECLARE(reusable, AGX_NUM_REGS) = {0};
   bool reuse = I->nr_dests == 1 && I->reg_constraints[0].reuse_killed;
   unsigned dest_count = reuse ? rctx->ncomps[I->dest[0].value] : 0;
   unsigned dest_align = MAX2(dest_count, reuse ? I->reg_constraints[0].align : 1);
   rctx->reused_sources = 0;
   agx_foreach_ssa_src(I, s) {
      struct agx_reg_constraint c = I->reg_constraints[I->nr_dests + s];
      if (!reuse || !I->src[s].kill || I->src[s].memory || c.clobber ||
          (c.min == c.max && c.max < reserved_size(rctx->shader)))
         continue;
      unsigned base = rctx->ssa_to_reg[I->src[s].value];
      unsigned width = rctx->ncomps[I->src[s].value];
      if (base + width <= base - (base % dest_align) + dest_count)
         BITSET_SET_COUNT(reusable, base, width);
   }
   rctx->reusable = reuse ? reusable : NULL;
   agx_foreach_ssa_dest(I, d) {
      unsigned reg =
         place_constrained_value(rctx, I->dest[d], I->reg_constraints[d], false);
      if (reg == ~0u) {
         rctx->moved = NULL;
         return false;
      }
      agx_foreach_ssa_src(I, s) {
         unsigned base = rctx->ssa_to_reg[I->src[s].value];
         if (reuse && I->src[s].kill && BITSET_TEST(reusable, base) &&
             base >= reg && base < reg + dest_count) {
            assert(s < 64);
            rctx->reused_sources |= BITFIELD64_BIT(s);
         }
      }
   }
   rctx->reusable = NULL;
   rctx->moved = NULL;

   /* Copy moved values in order of their original registers. */
   for (unsigned i = 1; i < rctx->nr_moved; ++i) {
      struct ra_move m = moved[i];
      unsigned j = i;
      for (; j && moved[j - 1].reg > m.reg; --j)
         moved[j] = moved[j - 1];
      moved[j] = m;
   }
   struct util_dynarray copies = UTIL_DYNARRAY_INIT;
   for (unsigned i = 0; i < rctx->nr_moved; ++i) {
      unsigned ssa = moved[i].ssa, reg = moved[i].reg;
      if (rctx->ssa_to_reg[ssa] == reg)
         continue;
      unsigned width = agx_size_align_16(rctx->sizes[ssa]);
      for (unsigned c = 0; c < rctx->ncomps_unrounded[ssa]; c += width) {
         struct agx_copy copy = {
            .dest = rctx->ssa_to_reg[ssa] + c,
            .src = agx_register(reg + c, rctx->sizes[ssa]),
         };
         util_dynarray_append(&copies, copy);
      }
   }
   agx_foreach_ssa_src(I, s) {
      struct agx_reg_constraint c = I->reg_constraints[I->nr_dests + s];
      if (c.min != c.max || c.max >= reserved_size(rctx->shader))
         continue;
      unsigned old = rctx->ssa_to_reg[I->src[s].value];
      for (unsigned i = 0; i < rctx->nr_moved; ++i) {
         if (moved[i].ssa == I->src[s].value)
            old = moved[i].reg;
      }
      assert(old >= reserved_size(rctx->shader) && old < rctx->bound[RA_GPR]);
      struct agx_copy copy = {
         .dest = c.min,
         .src = agx_register(old, I->src[s].size),
      };
      util_dynarray_append(&copies, copy);
   }
   agx_builder b = agx_init_builder(rctx->shader, agx_before_instr(I));
   agx_emit_parallel_copies(&b, copies.data,
                            util_dynarray_num_elements(&copies, struct agx_copy));
   util_dynarray_fini(&copies);
   rctx->pinned = NULL;
   return true;
}

static void
agx_set_sources(struct ra_ctx *rctx, agx_instr *I)
{
   assert(I->op != AGX_OPCODE_PHI);

   agx_foreach_ssa_src(I, s) {
      assert(BITSET_TEST(rctx->visited, I->src[s].value) && "no phis");

      I->src[s].reg = rctx->ssa_to_reg[I->src[s].value];
      if (I->reg_constraints) {
         struct agx_reg_constraint c = I->reg_constraints[I->nr_dests + s];
         if (c.min == c.max && c.max < reserved_size(rctx->shader))
            I->src[s].reg = c.min;
      }
      I->src[s].has_reg = true;
   }
}

static void
agx_set_dests(struct ra_ctx *rctx, agx_instr *I)
{
   agx_foreach_ssa_dest(I, s) {
      I->dest[s].reg = rctx->ssa_to_reg[I->dest[s].value];
      I->dest[s].has_reg = true;
   }
}

static unsigned
affinity_base_of_collect(struct ra_ctx *rctx, agx_instr *collect, unsigned src)
{
   unsigned src_reg = rctx->ssa_to_reg[collect->src[src].value];
   unsigned src_offset = src * agx_size_align_16(collect->src[src].size);

   if (src_reg >= src_offset)
      return src_reg - src_offset;
   else
      return ~0;
}

static bool
try_coalesce_with(struct ra_ctx *rctx, agx_index ssa, unsigned count,
                  bool may_be_unvisited, unsigned *out)
{
   assert(ssa.type == AGX_INDEX_NORMAL);
   if (!BITSET_TEST(rctx->visited, ssa.value)) {
      assert(may_be_unvisited);
      return false;
   }

   unsigned base = rctx->ssa_to_reg[ssa.value];
   enum ra_class cls = ra_class_for_index(ssa);

   if (BITSET_TEST_COUNT(rctx->used_regs[cls], base, count))
      return false;

   assert(base + count <= rctx->bound[cls] && "invariant");
   *out = base;
   return true;
}

static unsigned
pick_regs(struct ra_ctx *rctx, agx_instr *I, unsigned d)
{
   agx_index idx = I->dest[d];
   enum ra_class cls = ra_class_for_index(idx);
   assert(idx.type == AGX_INDEX_NORMAL);

   unsigned count = rctx->ncomps[idx.value];
   assert(count >= 1);

   unsigned align = count;

   pressure_web_event(rctx, idx.value, "CHOOSE", ~0u, NULL);
   unsigned preferred = preferred_phi_register(rctx, idx.value, true);
   if (preferred != ~0u)
      return preferred;

   /* Try to allocate moves compatibly with their sources */
   if (I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_NORMAL &&
       I->src[0].memory == I->dest[0].memory &&
       I->src[0].size == I->dest[0].size) {

      unsigned out;
      if (try_coalesce_with(rctx, I->src[0], count, false, &out))
         return out;
   }

   /* Try to allocate collects compatibly with their sources */
   if (I->op == AGX_OPCODE_COLLECT) {
      agx_foreach_ssa_src(I, s) {
         assert(BITSET_TEST(rctx->visited, I->src[s].value) &&
                "registers assigned in an order compatible with dominance "
                "and this is not a phi node, so we have assigned a register");

         unsigned base = affinity_base_of_collect(rctx, I, s);
         if (base >= rctx->bound[cls] || (base + count) > rctx->bound[cls])
            continue;

         /* Unaligned destinations can happen when dest size > src size */
         if (base % align)
            continue;

         if (!BITSET_TEST_COUNT(rctx->used_regs[cls], base, count))
            return base;
      }
   }

   /* Try to coalesce scalar exports */
   agx_instr *collect_use = rctx->src_to_collect[idx.value];
   if (collect_use && collect_use->op == AGX_OPCODE_EXPORT) {
      unsigned reg = collect_use->imm;

      if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, align) &&
          (reg % align) == 0)
         return reg;
   }

   /* Try to coalesce vector exports */
   if (collect_use && collect_use->op == AGX_OPCODE_SPLIT) {
      if (collect_use->dest[0].type == AGX_INDEX_NORMAL) {
         agx_instr *exp = rctx->src_to_collect[collect_use->dest[0].value];
         if (exp && exp->op == AGX_OPCODE_EXPORT) {
            unsigned reg = exp->imm;

            if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, align) &&
                (reg % align) == 0)
               return reg;
         }
      }
   }

   /* Try to allocate sources of collects contiguously */
   if (collect_use && collect_use->op == AGX_OPCODE_COLLECT) {
      agx_instr *collect = collect_use;

      assert(count == align && "collect sources are scalar");

      /* Find our offset in the collect. If our source is repeated in the
       * collect, this may not be unique. We arbitrarily choose the first.
       */
      unsigned our_source = ~0;
      agx_foreach_ssa_src(collect, s) {
         if (agx_is_equiv(collect->src[s], idx)) {
            our_source = s;
            break;
         }
      }

      assert(our_source < collect->nr_srcs && "source must be in the collect");

      /* See if we can allocate compatibly with any source of the collect */
      agx_foreach_ssa_src(collect, s) {
         if (!BITSET_TEST(rctx->visited, collect->src[s].value))
            continue;

         /* Determine where the collect should start relative to the source */
         unsigned base = affinity_base_of_collect(rctx, collect, s);
         if (base >= rctx->bound[cls])
            continue;

         unsigned our_reg = base + (our_source * align);

         /* Don't allocate past the end of the register file */
         if ((our_reg + align) > rctx->bound[cls])
            continue;

         /* If those registers are free, then choose them */
         if (!BITSET_TEST_COUNT(rctx->used_regs[cls], our_reg, align))
            return our_reg;
      }

      unsigned collect_align = rctx->ncomps[collect->dest[0].value];
      unsigned offset = our_source * align;

      /* Prefer ranges of the register file that leave room for all sources of
       * the collect contiguously.
       */
      for (unsigned base = 0;
           base + (collect->nr_srcs * align) <= rctx->bound[cls];
           base += collect_align) {
         if (!BITSET_TEST_COUNT(rctx->used_regs[cls], base,
                                collect->nr_srcs * align))
            return base + offset;
      }

      /* Try to respect the alignment requirement of the collect destination,
       * which may be greater than the sources (e.g. pack_64_2x32_split). Look
       * for a register for the source such that the collect base is aligned.
       */
      if (collect_align > align) {
         for (unsigned reg = offset; reg + collect_align <= rctx->bound[cls];
              reg += collect_align) {
            if (!BITSET_TEST_COUNT(rctx->used_regs[cls], reg, count))
               return reg;
         }
      }
   }

   /* Default to any contiguous sequence of registers */
   return find_regs(rctx, I, d, count, align);
}

static void
kill_source(struct ra_ctx *rctx, const agx_instr *I, unsigned s)
{
   enum ra_class cls = ra_class_for_index(I->src[s]);
   unsigned reg = rctx->ssa_to_reg[I->src[s].value];
   unsigned count = rctx->ncomps[I->src[s].value];

   assert(I->op != AGX_OPCODE_PHI && "phis don't use .kill");
   assert(count >= 1);

   BITSET_CLEAR_COUNT(rctx->used_regs[cls], reg, count);
}

static void
try_kill_early_sources(struct ra_ctx *rctx, const agx_instr *I,
                       unsigned first_source, unsigned last_source,
                       unsigned region_end, unsigned region_base)
{
   unsigned dest_size = util_next_power_of_two(rctx->ncomps[I->dest[0].value]);
   unsigned dest_end = region_base + dest_size;

   /* We can only early-kill a region if we can trivially allocate the
    * destination to it. That way we never shuffle killed sources.
    *
    * To ensure that, the region must be aligned and cover the destination.
    */
   if (region_base == region_end ||
       (rctx->ssa_to_reg[I->src[first_source].value] & (dest_size - 1)) ||
       ((region_end < dest_end) &&
        BITSET_TEST_RANGE(rctx->used_regs[RA_GPR], region_end, dest_end)))
      return;

   for (unsigned s = first_source; s <= last_source; ++s) {
      if (I->src[s].kill && !I->src[s].memory) {
         kill_source(rctx, I, s);
         rctx->early_killed = true;
         I->src[s].kill = false;
      }
   }
}

/** Assign registers to SSA values in a block. */

static bool
agx_ra_assign_local(struct ra_ctx *rctx)
{
   BITSET_DECLARE(used_regs_gpr, AGX_NUM_REGS) = {0};
   BITSET_DECLARE(used_regs_mem, AGX_NUM_MODELED_REGS) = {0};
   uint16_t *ssa_to_reg = calloc(rctx->shader->alloc, sizeof(uint16_t));

   agx_block *block = rctx->block;
   rctx->used_regs[RA_GPR] = used_regs_gpr;
   rctx->used_regs[RA_MEM] = used_regs_mem;
   rctx->ssa_to_reg = ssa_to_reg;

   reserve_live_in(rctx);

   if (rctx->shader->ra_target.reserved_registers ||
       rctx->shader->ra_target.spill_reserved_registers)
      BITSET_SET_COUNT(used_regs_gpr, 0, reserved_size(rctx->shader));

   /* Force the nesting counter r0l live throughout shaders using control flow.
    * This could be optimized (sync with agx_calc_register_demand).
    */
   if (rctx->shader->any_cf)
      BITSET_SET(used_regs_gpr, 0);

   /* Force the zero r0h live throughout shaders using divergent shuffles. */
   if (rctx->shader->any_quad_divergent_shuffle) {
      assert(rctx->shader->any_cf);
      BITSET_SET(used_regs_gpr, 1);
   }

   /* Reserve bottom registers as temporaries for parallel copy lowering */
   if (rctx->shader->has_spill_pcopy_reserved) {
      BITSET_SET_COUNT(used_regs_gpr, 0, 8);
   }

   agx_foreach_instr_in_block(block, I) {
      rctx->instr = I;
      /* Early killing frees allocation space, but deferred destructive spill
       * stores still need to know whether the original value survives. */
      bool dead_spill_source = rctx->shader->ra_target.preserve_source_kills &&
                               I->op == AGX_OPCODE_MOV && I->dest[0].memory &&
                               !I->src[0].memory && I->src[0].kill;

      /* Optimization: if a split contains the last use of a vector, the split
       * can be removed by assigning the destinations overlapping the source.
       */
      if (I->op == AGX_OPCODE_SPLIT && I->src[0].kill) {
         assert(ra_class_for_index(I->src[0]) == RA_GPR);
         unsigned reg = ssa_to_reg[I->src[0].value];
         unsigned width = agx_size_align_16(agx_split_width(I));

         agx_foreach_dest(I, d) {
            assert(ra_class_for_index(I->dest[0]) == RA_GPR);

            /* Free up the source */
            unsigned offset_reg = reg + (d * width);
            BITSET_CLEAR_COUNT(used_regs_gpr, offset_reg, width);

            /* Assign the destination where the source was */
            if (!agx_is_null(I->dest[d]))
               assign_regs(rctx, I->dest[d], offset_reg);
         }

         unsigned trail = rctx->ncomps[I->src[0].value] - (I->nr_dests * width);
         BITSET_CLEAR_COUNT(used_regs_gpr, reg + (I->nr_dests * width), trail);

         agx_set_sources(rctx, I);
         agx_set_dests(rctx, I);
         continue;
      } else if (I->op == AGX_OPCODE_PRELOAD) {
         /* We must coalesce all preload moves */
         assert(I->dest[0].size == I->src[0].size);
         assert(I->src[0].type == AGX_INDEX_REGISTER);

         /* r1l specifically is a preloaded register. It is reserved during
          * demand calculations to ensure we don't need live range shuffling of
          * spilling temporaries. But we can still preload to it. So if it's
          * reserved, just free it. It'll be fine.
          */
         if (I->src[0].value == 2) {
            BITSET_CLEAR(rctx->used_regs[RA_GPR], 2);
         }

         assign_regs(rctx, I->dest[0], I->src[0].value);
         agx_set_dests(rctx, I);
         continue;
      }

      /* Search for regions of contiguous killed sources to early-kill. */
      rctx->early_killed = false;

      if (I->nr_dests == 1 && !I->reg_constraints &&
          (!rctx->shader->ra_target.late_kill_sources ||
           I->op == AGX_OPCODE_MOV || I->op == AGX_OPCODE_COLLECT)) {
         unsigned first_src = 0;
         unsigned end = 0;
         unsigned start = 0;

         agx_foreach_ssa_src(I, s) {
            if (I->src[s].kill && !I->src[s].memory) {
               unsigned reg = rctx->ssa_to_reg[I->src[s].value];

               if (start == end || end != reg) {
                  try_kill_early_sources(rctx, I, first_src, s, end, start);
                  first_src = s;
                  start = reg;
               }

               end = reg + rctx->ncomps[I->src[s].value];
            }
         }

         try_kill_early_sources(rctx, I, first_src, I->nr_srcs - 1, end, start);
      }

      /* Next, assign destinations one at a time. This is always legal
       * because of the SSA form.
       */
      if (I->reg_constraints) {
         if (!assign_constrained_operands(rctx, I)) {
            free(ssa_to_reg);
            return false;
         }
      } else {
         agx_foreach_ssa_dest(I, d) {
            if (I->op == AGX_OPCODE_PHI && I->dest[d].has_reg)
               continue;

            assign_regs(rctx, I->dest[d], pick_regs(rctx, I, d));
         }
      }

      /* Free late-killed sources, except those the destination replaced. */
      agx_foreach_ssa_src(I, s) {
         if (I->src[s].kill &&
             !(I->reg_constraints && (rctx->reused_sources & BITFIELD64_BIT(s)))) {
            kill_source(rctx, I, s);
         }
      }

      /* Phi sources are special. Set in the corresponding predecessors */
      if (I->op != AGX_OPCODE_PHI)
         agx_set_sources(rctx, I);

      agx_set_dests(rctx, I);
      if (dead_spill_source)
         I->src[0].kill = true;
   }

   /* Sparse-bitset iteration is ordered by SSA name. Store just the live-out
    * assignments so each successor can binary-search them, instead of scanning
    * the register file once per live-in value (including the large spill file).
    */
   block->nr_ssa_to_reg_out = 0;
   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, i) {
      block->nr_ssa_to_reg_out++;
   }

   /* Keep the visited marker non-NULL even for blocks with no live-out values. */
   block->ssa_to_reg_out =
      malloc(MAX2(block->nr_ssa_to_reg_out, 1) * sizeof(*block->ssa_to_reg_out));
   unsigned live_out_index = 0;
   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, i) {
      block->ssa_to_reg_out[live_out_index].ssa = i;
      block->ssa_to_reg_out[live_out_index++].reg = rctx->ssa_to_reg[i];
   }

   /* Also set the sources for the phis in our successors, since that logically
    * happens now (given the possibility of live range splits, etc)
    */
   agx_foreach_successor(block, succ) {
      unsigned pred_idx = agx_predecessor_index(succ, block);

      agx_foreach_phi_in_block(succ, phi) {
         if (phi->src[pred_idx].type == AGX_INDEX_NORMAL &&
             !phi->src[pred_idx].has_reg) {
            /* This source needs a fixup */
            unsigned value = phi->src[pred_idx].value;
            phi->src[pred_idx].reg = rctx->ssa_to_reg[value];
            phi->src[pred_idx].has_reg = true;
         }
      }
   }

   free(rctx->ssa_to_reg);
   return true;
}

/*
 * Lower phis to parallel copies at the logical end of a given block. If a block
 * needs parallel copies inserted, a successor of the block has a phi node. To
 * have a (nontrivial) phi node, a block must have multiple predecessors. So the
 * edge from the block to the successor (with phi) is not the only edge entering
 * the successor. Because the control flow graph has no critical edges, this
 * edge must therefore be the only edge leaving the block, so the block must
 * have only a single successor.
 */
static void
agx_insert_parallel_copies(agx_context *ctx, agx_block *block)
{
   bool any_succ = false;
   unsigned nr_phi = 0;

   /* Phi nodes logically happen on the control flow edge, so parallel copies
    * are added at the end of the predecessor */
   agx_builder b = agx_init_builder(ctx, agx_after_block_logical(block));

   agx_foreach_successor(block, succ) {
      assert(nr_phi == 0 && "control flow graph has a critical edge");

      agx_foreach_phi_in_block(succ, phi) {
         assert(!any_succ && "control flow graph has a critical edge");
         nr_phi += agx_channels(phi->dest[0]);
      }

      any_succ = true;

      /* Nothing to do if there are no phi nodes */
      if (nr_phi == 0)
         continue;

      unsigned pred_index = agx_predecessor_index(succ, block);

      /* Create a parallel copy lowering all the phi nodes */
      struct agx_copy *copies = calloc(sizeof(*copies), nr_phi);

      unsigned i = 0;

      agx_foreach_phi_in_block(succ, phi) {
         pressure_audit_edge(ctx, block, succ, phi, pred_index);
         agx_index dest = phi->dest[0];
         agx_index src = phi->src[pred_index];

         if (src.type == AGX_INDEX_IMMEDIATE)
            src.size = dest.size;

         assert(dest.type == AGX_INDEX_REGISTER);
         assert(dest.size == src.size);

         /* Scalarize the phi, since the parallel copy lowering doesn't handle
          * vector phis. While we scalarize phis in NIR, we can generate vector
          * phis from spilling so must take care.
          */
         for (unsigned c = 0; c < agx_channels(phi->dest[0]); ++c) {
            agx_index src_ = src;
            unsigned offs = c * agx_size_align_16(src.size);

            if (src.type != AGX_INDEX_IMMEDIATE) {
               assert(src.type == AGX_INDEX_UNIFORM ||
                      src.type == AGX_INDEX_REGISTER);
               src_.value += offs;
               src_.channels_m1 = 1 - 1;
            }

            assert(i < nr_phi);
            copies[i++] = (struct agx_copy){
               .dest = dest.value + offs,
               .dest_mem = dest.memory,
               .src = src_,
            };
         }
      }

      agx_emit_parallel_copies(&b, copies, nr_phi);

      free(copies);
   }
}

static void
lower_exports(agx_context *ctx)
{
   struct agx_copy copies[AGX_NUM_REGS];
   unsigned nr = 0;
   agx_block *block = agx_exit_block(ctx);

   agx_foreach_instr_in_block_safe(block, I) {
      if (I->op != AGX_OPCODE_EXPORT)
         continue;

      assert(agx_channels(I->src[0]) == 1 && "scalarized in frontend");
      assert(nr < ARRAY_SIZE(copies));

      copies[nr++] = (struct agx_copy){
         .dest = I->imm,
         .src = I->src[0],
      };

      /* The export itself is now trivial, reflect that for correct last-use
       * tracking later.
       */
      I->src[0] = agx_register_like(I->imm, I->src[0]);

      /* We cannot use fewer registers than we export */
      ctx->max_reg =
         MAX2(ctx->max_reg, I->imm + agx_size_align_16(I->src[0].size));
   }

   agx_builder b = agx_init_builder(ctx, agx_after_block_logical(block));
   agx_emit_parallel_copies(&b, copies, nr);
}

static void
constraint_preferences(agx_context *ctx, struct agx_reg_constraint *preferred,
                       unsigned *scarcity, unsigned max_regs)
{
   for (unsigned v = 0; v < ctx->alloc; ++v)
      preferred[v] = (struct agx_reg_constraint){0, max_regs - 1, 1};

   agx_foreach_instr_global(ctx, I) {
      if (!I->reg_constraints)
         continue;
      agx_foreach_ssa_src(I, s) {
         struct agx_reg_constraint c = I->reg_constraints[I->nr_dests + s];
         if (I->src[s].memory ||
             (c.min == c.max && c.max < reserved_size(ctx)))
            continue;
         struct agx_reg_constraint *p = &preferred[I->src[s].value];
         p->min = MAX2(p->min, c.min);
         p->max = MIN2(p->max, c.max);
         p->align = MAX2(p->align, c.align);
      }
      for (unsigned o = 0; o < I->nr_dests + I->nr_srcs; ++o) {
         agx_index idx = o < I->nr_dests ? I->dest[o] : I->src[o - I->nr_dests];
         if (idx.type != AGX_INDEX_NORMAL || idx.memory)
            continue;
         struct agx_reg_constraint c = I->reg_constraints[o];
         unsigned width = util_next_power_of_two(agx_index_size_16(idx));
         unsigned end = MIN2(c.max + width, max_regs);
         if (end <= c.min || (c.min == 0 && end == max_regs))
            continue;
         unsigned weight = DIV_ROUND_UP(max_regs, end - c.min);
         for (unsigned h = c.min; h < end; ++h)
            scarcity[h] += weight;
      }
   }
}

bool
agx_ra(agx_context *ctx)
{
   pressure_audit_begin(ctx);
   ctx->ra_error = NULL;
   /* Destructive input operands get a private SSA lifetime. Ordinary move
    * coalescing removes the copy when the original value dies here. */
   agx_foreach_instr_global_safe(ctx, I) {
      if (!I->reg_constraints)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_foreach_ssa_src(I, s) {
         if (!I->reg_constraints[I->nr_dests + s].clobber)
            continue;
         agx_index copy = agx_vec_temp(ctx, I->src[s].size, agx_channels(I->src[s]));
         agx_mov_to(&b, copy, I->src[s]);
         I->src[s] = copy;
      }
   }
   bool force_spilling =
      (agx_compiler_debug & AGX_DBG_SPILL) && ctx->key->has_scratch;

   /* Determine maximum possible registers. We won't exceed this! */
   unsigned max_possible_regs = ctx->ra_target.max_registers ?: AGX_NUM_REGS;

   /* Compute shaders need to have their entire workgroup together, so our
    * register usage is bounded by the workgroup size.
    */
   if (mesa_shader_stage_is_compute(ctx->stage) &&
       !ctx->ra_target.max_registers) {
      unsigned threads_per_workgroup;

      /* If we don't know the workgroup size, worst case it. TODO: Optimize
       * this, since it'll decimate opencl perf.
       */
      if (ctx->nir->info.workgroup_size_variable) {
         threads_per_workgroup = 1024;
      } else {
         threads_per_workgroup = ctx->nir->info.workgroup_size[0] *
                                 ctx->nir->info.workgroup_size[1] *
                                 ctx->nir->info.workgroup_size[2];
      }

      max_possible_regs = MIN2(max_possible_regs,
         agx_max_registers_for_occupancy(threads_per_workgroup));
   }

   if (force_spilling) {
      /* Even when testing spilling, we need enough room for preloaded/exported
       * regs.
       */
      unsigned d = 24;
      unsigned max_ncomps = 8;

      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_PRELOAD) {
            unsigned size = agx_size_align_16(I->src[0].size);
            d = MAX2(d, I->src[0].value + size);
         } else if (I->op == AGX_OPCODE_EXPORT) {
            unsigned size = agx_size_align_16(I->src[0].size);
            d = MAX2(d, I->imm + size);
         } else if (I->op == AGX_OPCODE_IMAGE_WRITE) {
            /* vec4 source + vec4 coordinates + bindless handle + reserved */
            d = MAX2(d, 26);
         } else if (I->op == AGX_OPCODE_TEXTURE_SAMPLE &&
                    (I->lod_mode == AGX_LOD_MODE_LOD_GRAD ||
                     I->lod_mode == AGX_LOD_MODE_LOD_GRAD_MIN)) {
            /* as above but with big gradient */
            d = MAX2(d, 36);
         }

         agx_foreach_ssa_dest(I, v) {
            max_ncomps = MAX2(max_ncomps, agx_index_size_16(I->dest[v]));
         }

         /* Target instructions may need a tuple and all of its split results
          * live together. Forced spilling must still fit one instruction plus
          * the target's spill temporaries; no eviction can shrink that set. */
         if (ctx->ra_target.max_registers) {
            unsigned needed = MAX2(ctx->ra_target.reserved_registers,
                                   ctx->ra_target.spill_reserved_registers);
            agx_foreach_ssa_dest(I, v) {
               if (!I->dest[v].memory)
                  needed += util_next_power_of_two(agx_index_size_16(I->dest[v]));
            }
            agx_foreach_ssa_src(I, s) {
               if (I->src[s].memory)
                  continue;
               bool duplicate = false;
               for (unsigned prev = 0; prev < s; ++prev)
                  duplicate |= agx_is_equiv(I->src[prev], I->src[s]);
               if (!duplicate)
                  needed += util_next_power_of_two(agx_index_size_16(I->src[s]));
            }
            d = MAX2(d, needed);
         }
      }

      max_possible_regs = ALIGN_POT(d, util_next_power_of_two(max_ncomps));
   } else if (ctx->key->is_helper) {
      /* The helper program is unspillable and has a limited register file */
      max_possible_regs = 32;
   }

   if (!validate_constraint_windows(ctx, max_possible_regs)) {
      pressure_audit_end(ctx);
      return false;
   }

   /* Calculate the demand. We'll use it to determine if we need to spill and to
    * bound register assignment.
    */
   agx_compute_liveness(ctx);
   unsigned effective_demand = agx_calc_register_demand(ctx, false);
   bool spilling = (effective_demand > max_possible_regs);
   bool remat = false;

   /* If we need multiple waves, see if we can rematerialize constants to save
    * waves. If we only have a single wave regardless, this is pointless.
    */
   if (!ctx->ra_target.disable_occupancy_rematerialization &&
       effective_demand > agx_round_registers(1) && !spilling) {
      unsigned effective_demand_remat = agx_calc_register_demand(ctx, true);

      /* Worst-case assume we need 6 16-bit registers for constants, for a
       * four-source cmpsel where 3 sources are 32-bit constants. Rounded to
       * ensure live-range splitting works.
       */
      if ((effective_demand_remat + 6) < effective_demand) {
         unsigned l = agx_round_registers(
            align(agx_round_registers(effective_demand_remat + 6), 16));

         /* Only rematerialize if it actually lets us save a wave */
         if (l < agx_round_registers(effective_demand)) {
            remat = true;
            max_possible_regs = l;
         }
      }
   }

   if (spilling || remat) {
      assert((remat || ctx->key->has_scratch) &&
             "internal shaders are unspillable");

      agx_spill(ctx, max_possible_regs, remat);
      pressure_audit_spilled(ctx);

      if (!validate_constraint_windows(ctx, max_possible_regs)) {
         pressure_audit_end(ctx);
         return false;
      }

      /* After spilling, recalculate liveness and demand */
      agx_compute_liveness(ctx);
      effective_demand = agx_calc_register_demand(ctx, false);

      /* The resulting program can now be assigned registers */
      assert(effective_demand <= max_possible_regs && "spiller post-condition");
   }

   /* Constants shared by unrelated phis must not connect their affinity webs.
    * Give each incoming constant use a private affinity member. The actual SSA
    * source stays shared, so this does not increase pressure or emit extra
    * materializations. Local phi/source coalescing can still use its register.
    */
   BITSET_WORD *constants = BITSET_CALLOC(ctx->alloc);
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV_IMM &&
          I->dest[0].type == AGX_INDEX_NORMAL)
         BITSET_SET(constants, I->dest[0].value);
   }

   unsigned nr_web_members = ctx->alloc;
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, phi) {
         agx_foreach_ssa_src(phi, s) {
            nr_web_members += BITSET_TEST(constants, phi->src[s].value) != 0;
         }
      }
   }

   /* Initialize singleton members, then join phi destinations to their inputs.
    * Members above ctx->alloc represent incoming constant uses, not SSA defs.
    */
   struct phi_web_node *phi_web = calloc(nr_web_members, sizeof(*phi_web));
   void *web_memctx = ralloc_context(NULL);
   struct phi_web_members *web_members =
      ralloc_array(web_memctx, struct phi_web_members, nr_web_members);
   void *graph_memctx = ralloc_context(web_memctx);
   struct u_sparse_bitset *interference =
      phi_web_interference(ctx, constants, graph_memctx);

   for (unsigned i = 0; i < nr_web_members; ++i) {
      phi_web[i].parent = i;
      web_members[i] = (struct phi_web_members){i, i, ~0u, 1};
   }

   unsigned next_constant_member = ctx->alloc;
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, phi) {
         agx_foreach_ssa_src(phi, s) {
            unsigned member = phi->src[s].value;
            if (BITSET_TEST(constants, member))
               member = next_constant_member++;

            if (phi->dest[0].size != phi->src[s].size ||
                phi->dest[0].memory != phi->src[s].memory ||
                agx_channels(phi->dest[0]) != agx_channels(phi->src[s]))
               continue;

            phi_web_union_noninterfering(phi_web, web_members, interference,
                                        ctx->alloc, phi->dest[0].value, member);
         }
      }
   }
   assert(next_constant_member == nr_web_members);
#ifndef NDEBUG
   for (unsigned v = 0; v < ctx->alloc; ++v) {
      U_SPARSE_BITSET_FOREACH_SET(&interference[v], other) {
         assert(phi_web_find(phi_web, v) != phi_web_find(phi_web, other));
      }
   }
#endif
   struct phi_preference **phi_preferences =
      rzalloc_array(web_memctx, struct phi_preference *, ctx->alloc);
   struct phi_dead_group **dead_groups =
      rzalloc_array(web_memctx, struct phi_dead_group *, ctx->alloc);
   uint64_t *phi_weights = rzalloc_array(web_memctx, uint64_t, ctx->alloc);
   unsigned *last_reg = ralloc_array(web_memctx, unsigned, ctx->alloc);
   for (unsigned v = 0; v < ctx->alloc; ++v)
      last_reg[v] = ~0u;

   /* Structured source order gives each loop a header-to-last-backedge span.
    * Weight an edge by its predecessor's depth, so a loop recurrence outweighs
    * a one-time entry. Saturate depth to keep sums comfortably in uint64_t.
    */
   unsigned *depth = rzalloc_array(web_memctx, unsigned, ctx->num_blocks);
   agx_foreach_block(ctx, header) {
      if (!header->loop_header)
         continue;
      unsigned end = header->index;
      agx_foreach_predecessor(header, pred)
         end = MAX2(end, (*pred)->index);
      for (unsigned b = header->index; b <= end; ++b)
         ++depth[b];
   }
   unsigned next_constant_preference = ctx->alloc;
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, phi) {
         agx_foreach_predecessor(block, pred) {
            unsigned s = agx_predecessor_index(block, *pred);
            agx_index *src = &phi->src[s], *dst = &phi->dest[0];
            unsigned member = src->type == AGX_INDEX_NORMAL &&
                              BITSET_TEST(constants, src->value)
                                 ? next_constant_preference++ : ~0u;
            if (src->type != AGX_INDEX_NORMAL ||
                src->size != dst->size || src->memory != dst->memory ||
                agx_channels(*src) != agx_channels(*dst))
               continue;
            uint64_t weight = 1ull << (3 * MIN2(depth[(*pred)->index], 6));
            /* Successor live_in is after its parallel phi definitions. A
             * source redefined by that bundle denotes the old iteration. */
            bool redefined = false;
            agx_foreach_phi_in_block(block, other)
               redefined |= other->dest[0].value == src->value;
            bool dead = redefined ||
               !u_sparse_bitset_test(&block->live_in, src->value);
            if (dead && !src->memory &&
                !u_sparse_bitset_test(&interference[src->value], dst->value)) {
               struct phi_dead_group *g = dead_groups[src->value];
               while (g && (g->successor != block || g->predecessor != *pred))
                  g = g->next;
               if (!g) {
                  g = rzalloc(web_memctx, struct phi_dead_group);
                  *g = (struct phi_dead_group){dead_groups[src->value],
                                              block, *pred, NULL, weight};
                  dead_groups[src->value] = g;
               }
               struct phi_dead_destination *d =
                  ralloc(web_memctx, struct phi_dead_destination);
               *d = (struct phi_dead_destination){g->destinations, dst};
               g->destinations = d;
            }
            struct phi_preference *forward =
               ralloc(web_memctx, struct phi_preference);
            struct phi_preference *reverse =
               ralloc(web_memctx, struct phi_preference);
            *forward = (struct phi_preference){
               phi_preferences[dst->value], src, weight};
            *reverse = (struct phi_preference){
               phi_preferences[src->value], dst, weight};
            phi_preferences[dst->value] = forward;
            phi_preferences[src->value] = reverse;
            phi_weights[src->value] += weight;
            phi_weights[dst->value] += weight;
            if (member != ~0u)
               web_members[member].constant = forward;
         }
      }
   }
   assert(next_constant_preference == nr_web_members);
   ralloc_free(graph_memctx);
   free(constants);
   struct util_dynarray reg_preferences;
   util_dynarray_init(&reg_preferences, web_memctx);
   int16_t reg_preference_index[AGX_NUM_REGS];
   memset(reg_preference_index, 0xff, sizeof(reg_preference_index));

   uint8_t *ncomps = calloc(ctx->alloc, sizeof(uint8_t));
   uint8_t *ncomps_unrounded = calloc(ctx->alloc, sizeof(uint8_t));
   enum ra_class *classes = calloc(ctx->alloc, sizeof(enum ra_class));
   agx_instr **src_to_collect = calloc(ctx->alloc, sizeof(agx_instr *));
   enum agx_size *sizes = calloc(ctx->alloc, sizeof(enum agx_size));
   BITSET_WORD *visited = BITSET_CALLOC(ctx->alloc);
   unsigned max_ncomps = 1;

   agx_foreach_instr_global(ctx, I) {
      /* Record tuple and export uses; all phi uses are retained separately. */
      if (I->op == AGX_OPCODE_COLLECT || I->op == AGX_OPCODE_EXPORT ||
          I->op == AGX_OPCODE_SPLIT) {
         agx_foreach_ssa_src(I, s) {
            src_to_collect[I->src[s].value] = I;
         }
      }

      agx_foreach_ssa_dest(I, d) {
         unsigned v = I->dest[d].value;
         assert(ncomps[v] == 0 && "broken SSA");
         /* Round up vectors for easier live range splitting */
         ncomps_unrounded[v] = agx_index_size_16(I->dest[d]);
         ncomps[v] = util_next_power_of_two(ncomps_unrounded[v]);
         sizes[v] = I->dest[d].size;
         classes[v] = ra_class_for_index(I->dest[d]);

         max_ncomps = MAX2(max_ncomps, ncomps[v]);
      }
   }

   /* For live range splitting to work properly, ensure the register file is
    * aligned to the larger vector size. Most of the time, this is a no-op since
    * the largest vector size is usually 128-bit and the register file is
    * naturally 128-bit aligned. However, this is required for correctness with
    * 3D textureGrad, which can have a source vector of length 6x32-bit,
    * rounding up to 256-bit and requiring special accounting here.
    */
   unsigned reg_file_alignment = MAX2(max_ncomps, 8);
   assert(util_is_power_of_two_nonzero(reg_file_alignment));
   assert(reg_file_alignment <= 16 && "max size");

   unsigned demand = ALIGN_POT(effective_demand, reg_file_alignment);
   assert(demand <= max_possible_regs && "Invariant");

   /* Round up the demand to the maximum number of registers we can use without
    * affecting occupancy. This reduces live range splitting.
    */
   unsigned max_regs =
      (ctx->key->is_helper || force_spilling || ctx->ra_target.max_registers)
         ? max_possible_regs
         : agx_occupancy_for_register_count(demand).max_registers;

   max_regs = ROUND_DOWN_TO(max_regs, reg_file_alignment);

   /* Or, we can bound tightly for debugging */
   if (agx_compiler_debug & AGX_DBG_DEMAND)
      max_regs = ALIGN_POT(MAX2(demand, 12), reg_file_alignment);

   /* ...but not too tightly */
   assert((max_regs % reg_file_alignment) == 0 && "occupancy limits aligned");
   assert(max_regs >= (6 * 2) && "space for vertex shader preloading");
   assert(max_regs <= max_possible_regs);

   struct agx_reg_constraint *preferred = NULL;
   unsigned scarcity[AGX_NUM_REGS] = {0};
   if (ctx->ra_target.cost_register_constraints) {
      preferred = calloc(ctx->alloc, sizeof(*preferred));
      if (preferred)
         constraint_preferences(ctx, preferred, scarcity, max_regs);
   }

   unsigned (*scarcity_floor)[AGX_NUM_REGS + 1] = NULL;
   if (preferred) {
      scarcity_floor = malloc(RA_SCARCITY_FLOOR_SIZES * sizeof(*scarcity_floor));
      for (unsigned l = 0; scarcity_floor && l < RA_SCARCITY_FLOOR_SIZES; ++l) {
         unsigned width = 1u << l;
         scarcity_floor[l][AGX_NUM_REGS] = UINT_MAX;
         for (int r = AGX_NUM_REGS - 1; r >= 0; --r) {
            unsigned sum = UINT_MAX;
            if (r + width <= AGX_NUM_REGS) {
               sum = 0;
               for (unsigned h = r; h < r + width; ++h)
                  sum += scarcity[h];
            }
            scarcity_floor[l][r] = MIN2(sum, scarcity_floor[l][r + 1]);
         }
      }
   }

   struct agx_spill_interference *spill_graph =
      ctx->spill_home ? agx_spill_coalesce_prepare(ctx) : NULL;
   bool success = false;
   if (!validate_constraint_windows(ctx, max_regs))
      goto cleanup;

   unsigned reg_count = 0, mem_slot_count = 0;

   /* Assign registers in dominance-order. This coincides with source-order due
    * to a NIR invariant, so we do not need special handling for this.
    */
   agx_foreach_block(ctx, block) {
      if (!agx_ra_assign_local(&(struct ra_ctx){
         .shader = ctx,
         .block = block,
         .src_to_collect = src_to_collect,
         .phi_web = phi_web,
         .web_members = web_members,
         .phi_preferences = phi_preferences,
         .dead_groups = dead_groups,
         .phi_weights = phi_weights,
         .last_reg = last_reg,
         .reg_preferences = &reg_preferences,
         .reg_preference_index = reg_preference_index,
         .ncomps = ncomps,
         .ncomps_unrounded = ncomps_unrounded,
         .sizes = sizes,
         .classes = classes,
         .visited = visited,
         .bound[RA_GPR] = max_regs,
         .bound[RA_MEM] = AGX_NUM_MODELED_REGS,
         .count[RA_GPR] = &reg_count,
         .count[RA_MEM] = &mem_slot_count,
         .preferred = preferred,
         .scarcity = scarcity,
         .scarcity_floor = scarcity_floor,
         .demand = demand,
         .affinity_ssa = ~0u,
      }))
         goto cleanup;
   }

   if (spill_graph)
      agx_spill_coalesce_apply(ctx, spill_graph, &mem_slot_count);

   ctx->max_reg = reg_count ? (reg_count - 1) : 0;
   ctx->spill_base_B = ctx->scratch_size_B;
   ctx->scratch_size_B += mem_slot_count * 2;

   /* Vertex shaders preload the vertex/instance IDs (r5, r6) even if the shader
    * don't use them. Account for that so the preload doesn't clobber GPRs.
    * Hardware tessellation eval shaders preload patch/instance IDs there.
    */
   if (ctx->nir->info.stage == MESA_SHADER_VERTEX ||
       ctx->nir->info.stage == MESA_SHADER_TESS_EVAL)
      ctx->max_reg = MAX2(ctx->max_reg, 6 * 2);

   assert(ctx->max_reg <= max_regs);

   pressure_audit_phis(ctx);

   /* Validate RA after assigning registers just before lowering SSA */
   agx_validate_ra(ctx);

   agx_foreach_instr_global_safe(ctx, ins) {
      /* Lower away SSA */
      agx_foreach_ssa_dest(ins, d) {
         ins->dest[d] =
            agx_replace_index(ins->dest[d], agx_as_register(ins->dest[d]));
      }

      /* Apple9 consumes the shared liveness result when selecting physical
       * operand release bits. Duplicate operands have one SSA kill marker. */
      unsigned source_kills = 0;
      if (ctx->ra_target.preserve_source_kills) {
         agx_foreach_ssa_src(ins, s) {
            agx_foreach_ssa_src(ins, t) {
               if (ins->src[t].kill &&
                   ins->src[s].value == ins->src[t].value &&
                   ins->src[s].memory == ins->src[t].memory)
                  source_kills |= BITFIELD_BIT(s);
            }
         }
      }
      agx_foreach_ssa_src(ins, s) {
         agx_replace_src(ins, s, agx_as_register(ins->src[s]));
         if (ctx->ra_target.preserve_source_kills)
            ins->src[s].kill = (source_kills & BITFIELD_BIT(s)) != 0;
      }

      /* Lower away RA pseudo-instructions */
      agx_builder b = agx_init_builder(ctx, agx_after_instr(ins));

      if (ins->op == AGX_OPCODE_COLLECT) {
         assert(ins->dest[0].type == AGX_INDEX_REGISTER);
         assert(!ins->dest[0].memory);

         unsigned base = ins->dest[0].value;
         unsigned width = agx_size_align_16(ins->src[0].size);

         struct agx_copy *copies = alloca(sizeof(copies[0]) * ins->nr_srcs);
         unsigned n = 0;

         /* Move the sources */
         agx_foreach_src(ins, i) {
            if (agx_is_null(ins->src[i]) || ins->src[i].type == AGX_INDEX_UNDEF)
               continue;
            assert(ins->src[i].size == ins->src[0].size);

            assert(n < ins->nr_srcs);
            copies[n++] = (struct agx_copy){
               .dest = base + (i * width),
               .src = ins->src[i],
            };
         }

         agx_emit_parallel_copies(&b, copies, n);
         agx_remove_instruction(ins);
         continue;
      } else if (ins->op == AGX_OPCODE_SPLIT) {
         assert(ins->src[0].type == AGX_INDEX_REGISTER ||
                ins->src[0].type == AGX_INDEX_UNIFORM);

         struct agx_copy copies[8];
         assert(ins->nr_dests <= ARRAY_SIZE(copies));

         unsigned n = 0;
         unsigned width = agx_size_align_16(agx_split_width(ins));

         /* Move the sources */
         agx_foreach_dest(ins, i) {
            if (ins->dest[i].type != AGX_INDEX_REGISTER)
               continue;

            assert(!ins->dest[i].memory);

            agx_index src = ins->src[0];
            src.size = ins->dest[i].size;
            src.channels_m1 = 0;
            src.value += (i * width);

            assert(n < ARRAY_SIZE(copies));
            copies[n++] = (struct agx_copy){
               .dest = ins->dest[i].value,
               .src = src,
            };
         }

         /* Lower away */
         agx_builder b = agx_init_builder(ctx, agx_after_instr(ins));
         agx_emit_parallel_copies(&b, copies, n);
         agx_remove_instruction(ins);
         continue;
      }
   }

   /* Insert parallel copies lowering phi nodes and exports */
   agx_foreach_block(ctx, block) {
      agx_insert_parallel_copies(ctx, block);
   }

   lower_exports(ctx);

   agx_foreach_instr_global_safe(ctx, I) {
      switch (I->op) {
      /* Pseudoinstructions for RA must be removed now */
      case AGX_OPCODE_PHI:
      case AGX_OPCODE_PRELOAD:
         agx_remove_instruction(I);
         break;

      /* Coalesced moves can be removed */
      case AGX_OPCODE_MOV:
         if (I->src[0].type == AGX_INDEX_REGISTER &&
             I->dest[0].size == I->src[0].size &&
             I->src[0].value == I->dest[0].value &&
             I->src[0].memory == I->dest[0].memory) {

            assert(I->dest[0].type == AGX_INDEX_REGISTER);
            agx_remove_instruction(I);
         }
         break;

      default:
         break;
      }
   }

   if (spilling && !ctx->ra_target.defer_spill_lowering)
      agx_lower_spill(ctx);

   success = true;

cleanup:
   free(scarcity_floor);
   ralloc_free(spill_graph);
   agx_foreach_block(ctx, block) {
      free(block->ssa_to_reg_out);
      block->ssa_to_reg_out = NULL;
      block->nr_ssa_to_reg_out = 0;
   }

   pressure_audit_end(ctx);
   ralloc_free(web_memctx);
   free(phi_web);
   free(src_to_collect);
   free(ncomps);
   free(ncomps_unrounded);
   free(sizes);
   free(classes);
   free(visited);
   free(preferred);
   return success;
}
