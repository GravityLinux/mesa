/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Bounded, opt-in spill-home coalescing prototype. Build interference while
 * the program is SSA, then recolor only memory after normal register assignment.
 * If the contracted graph cannot fit the original frame, keep the original
 * assignment. No spill/reload placement or GPR assignment is changed here.
 */

#include "agx_compiler.h"
#include "agx_debug.h"

struct agx_spill_interference {
   unsigned n, stride, alloc;
   unsigned *node, *home, *width;
   BITSET_WORD *adj;
};

static BITSET_WORD *
neighbors(struct agx_spill_interference *g, unsigned n)
{
   return g->adj + n * g->stride;
}

static void
interfere(struct agx_spill_interference *g, unsigned x, unsigned y)
{
   if (x != y) {
      BITSET_SET(neighbors(g, x), y);
      BITSET_SET(neighbors(g, y), x);
   }
}

static void
clique(struct agx_spill_interference *g, BITSET_WORD *live)
{
   unsigned x;
   BITSET_FOREACH_SET(x, live, g->n) {
      BITSET_WORD *row = neighbors(g, x);
      /* Every live vertex gets the same live set. Union whole words instead
       * of inserting both directions of every pair separately. */
      for (unsigned word = 0; word < g->stride; ++word)
         row[word] |= live[word];
      BITSET_CLEAR(row, x);
   }
}

struct agx_spill_interference *
agx_spill_coalesce_prepare(agx_context *ctx)
{
   unsigned n = 0, v;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         n += I->dest[d].memory;
   }

   /* Keep the experimental dense graph bounded; large shaders fall back. */
   if (!ctx->spill_home || !n || n > 4096)
      return NULL;

   struct agx_spill_interference *g = rzalloc(NULL, struct agx_spill_interference);
   g->n = n;
   g->alloc = ctx->alloc;
   g->stride = BITSET_WORDS(n);
   g->node = ralloc_array(g, unsigned, ctx->alloc);
   memset(g->node, 0xff, ctx->alloc * sizeof(unsigned));
   g->home = ralloc_array(g, unsigned, n);
   g->width = ralloc_array(g, unsigned, n);
   g->adj = rzalloc_array(g, BITSET_WORD, n * g->stride);
   n = 0;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d) {
         agx_index dst = I->dest[d];
         if (!dst.memory)
            continue;
         g->node[dst.value] = n;
         g->home[n] = dst.value < ctx->spill_home_count
            ? ctx->spill_home[dst.value] : 0;
         g->width[n++] = util_next_power_of_two(agx_index_size_16(dst));
      }
   }

   BITSET_WORD *live = rzalloc_array(g, BITSET_WORD, g->stride);
   agx_foreach_block(ctx, block) {
      memset(live, 0, BITSET_BYTES(g->n));
      U_SPARSE_BITSET_FOREACH_SET(&block->live_out, ssa) {
         if (g->node[ssa] != ~0u)
            BITSET_SET(live, g->node[ssa]);
      }
      clique(g, live);
      agx_foreach_instr_in_block_rev(block, I) {
         /* Phi definitions act in parallel at block entry; their sources
          * already participate in the corresponding predecessors' live_out. */
         if (I->op == AGX_OPCODE_PHI)
            continue;
         agx_foreach_ssa_dest(I, d) {
            if (!I->dest[d].memory)
               continue;
            unsigned dst = g->node[I->dest[d].value];
            BITSET_FOREACH_SET(v, live, g->n)
               interfere(g, dst, v);
            BITSET_CLEAR(live, dst);
         }
         agx_foreach_ssa_src(I, s) {
            if (I->src[s].memory)
               BITSET_SET(live, g->node[I->src[s].value]);
         }
      }
      /* Even a dead phi writes its destination and must not clobber another
       * value live at entry. All phi destinations are simultaneous. */
      agx_foreach_phi_in_block(block, phi) {
         if (phi->dest[0].memory)
            BITSET_SET(live, g->node[phi->dest[0].value]);
      }
      clique(g, live);
   }
   ralloc_free(live);
   return g;
}

static bool
overlap(unsigned a, unsigned aw, unsigned b, unsigned bw)
{
   return a < b + bw && b < a + aw;
}

static bool
record_assignment(struct agx_spill_interference *g, unsigned *old, agx_index idx)
{
   if (!idx.memory)
      return true;
   /* Don't attempt to recolor a physical memory move or a live-range split. */
   if (idx.type != AGX_INDEX_NORMAL || !idx.has_reg || idx.value >= g->alloc)
      return false;
   unsigned node = g->node[idx.value];
   if (node == ~0u || (old[node] != ~0u && old[node] != idx.reg))
      return false;
   old[node] = idx.reg;
   return true;
}

bool
agx_spill_coalesce_apply(agx_context *ctx, struct agx_spill_interference *g,
                         unsigned *half_slots)
{
   unsigned y;
   void *memctx = ralloc_context(NULL);
   unsigned *old = ralloc_array(memctx, unsigned, g->n);
   memset(old, 0xff, g->n * sizeof(unsigned));
   bool success = false;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_dest(I, d) {
         if (!record_assignment(g, old, I->dest[d]))
            goto out;
      }
      agx_foreach_src(I, s) {
         if (!record_assignment(g, old, I->src[s]))
            goto out;
      }
   }

   /* Validate the graph against the existing assignment before changing it. */
   for (unsigned x = 0; x < g->n; ++x) {
      if (old[x] == ~0u || old[x] + g->width[x] > *half_slots)
         goto out;
      BITSET_FOREACH_SET(y, neighbors(g, x), g->n) {
         if (overlap(old[x], g->width[x], old[y], g->width[y]))
            goto out;
      }
   }

   unsigned *group = ralloc_array(memctx, unsigned, g->n);
   unsigned *representative = ralloc_array(memctx, unsigned, g->n);
   unsigned count = 0;
   for (unsigned x = 0; x < g->n; ++x) {
      unsigned c;
      for (c = 0; c < count; ++c) {
         unsigned r = representative[c];
         if (!g->home[x] || g->home[x] != g->home[r] || g->width[x] != g->width[r])
            continue;
         bool conflict = false;
         for (unsigned y = 0; y < x; ++y)
            conflict |= group[y] == c && BITSET_TEST(neighbors(g, x), y);
         if (!conflict)
            break;
      }
      group[x] = c;
      if (c == count)
         representative[count++] = x;
   }

   unsigned stride = BITSET_WORDS(count);
   BITSET_WORD *adj = rzalloc_array(memctx, BITSET_WORD, count * stride);
   unsigned *degree = rzalloc_array(memctx, unsigned, count);
   for (unsigned x = 0; x < g->n; ++x) {
      BITSET_FOREACH_SET(y, neighbors(g, x), g->n) {
         assert(group[x] != group[y]);
         BITSET_SET(adj + group[x] * stride, group[y]);
      }
   }
   for (unsigned c = 0; c < count; ++c) {
      BITSET_FOREACH_SET(y, adj + c * stride, count)
         ++degree[c];
   }

   /* DSATUR, using occupied half-slots for saturation so widths/alignment are
    * respected. A failure discards the entire recoloring, retaining the old
    * scratch footprint and assignment. */
   unsigned *color = ralloc_array(memctx, unsigned, count);
   memset(color, 0xff, count * sizeof(unsigned));
   unsigned color_stride = BITSET_WORDS(*half_slots);
   BITSET_WORD *blocked = rzalloc_array(memctx, BITSET_WORD, count * color_stride);
   unsigned *saturation = rzalloc_array(memctx, unsigned, count);
   unsigned high = 0;
   for (unsigned assigned = 0; assigned < count; ++assigned) {
      unsigned best = ~0u, best_sat = 0;
      for (unsigned c = 0; c < count; ++c) {
         if (color[c] != ~0u)
            continue;
         unsigned sat = saturation[c];
         if (best == ~0u || sat > best_sat ||
             (sat == best_sat && (degree[c] > degree[best] ||
              (degree[c] == degree[best] &&
               g->home[representative[c]] < g->home[representative[best]])))) {
            best = c;
            best_sat = sat;
         }
      }
      BITSET_WORD *used = blocked + best * color_stride;
      unsigned width = g->width[representative[best]], base;
      for (base = 0; base + width <= *half_slots; base += width) {
         if (!BITSET_TEST_COUNT(used, base, width))
            break;
      }
      if (base + width > *half_slots)
         goto out;
      color[best] = base;
      high = MAX2(high, base + width);
      BITSET_FOREACH_SET(y, adj + best * stride, count) {
         BITSET_WORD *blocked_y = blocked + y * color_stride;
         for (unsigned r = base; r < base + width; ++r) {
            if (!BITSET_TEST(blocked_y, r)) {
               BITSET_SET(blocked_y, r);
               ++saturation[y];
            }
         }
      }
   }

   /* Only now commit the complete, bounded coloring. Synthetic identity phis
    * inserted during allocation keep the same SSA identity and are rewritten
    * too. The block exit maps are no longer read after allocation completes. */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d) {
         if (I->dest[d].memory)
            I->dest[d].reg = color[group[g->node[I->dest[d].value]]];
      }
      agx_foreach_ssa_src(I, s) {
         if (I->src[s].memory)
            I->src[s].reg = color[group[g->node[I->src[s].value]]];
      }
   }
   if (agx_get_compiler_debug() & AGX_DBG_VERBOSE)
      fprintf(stderr, "AGX_SPILL_COALESCE values=%u groups=%u half_slots=%u->%u\n",
              g->n, count, *half_slots, high);
   *half_slots = high;
   success = true;

out:
   if (!success && (agx_get_compiler_debug() & AGX_DBG_VERBOSE))
      fprintf(stderr, "AGX_SPILL_COALESCE fallback values=%u half_slots=%u\n",
              g->n, *half_slots);
   ralloc_free(memctx);
   return success;
}
