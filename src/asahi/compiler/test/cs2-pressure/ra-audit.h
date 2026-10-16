/* SPDX-License-Identifier: MIT
 * Local, opt-in allocation diagnostics. No placement or lowering changes.
 */
#include "util/u_debug.h"
#include "util/hash_table.h"

DEBUG_GET_ONCE_OPTION(cached_agx_apple9_ra_audit, "AGX_APPLE9_RA_AUDIT", NULL)

struct pressure_phi_tag {
   unsigned id;
   const char *origin;
};
static _Thread_local struct {
   agx_context *ctx;
   struct hash_table *phis;
   unsigned serial, next_phi, next_copy;
} pressure_audit;

static void
pressure_tag_phi(agx_context *ctx, agx_instr *phi, const char *origin)
{
   if (pressure_audit.ctx != ctx)
      return;
   if (_mesa_hash_table_search(pressure_audit.phis, phi))
      return;
   struct pressure_phi_tag *tag = ralloc(pressure_audit.phis, struct pressure_phi_tag);
   *tag = (struct pressure_phi_tag){++pressure_audit.next_phi, origin};
   _mesa_hash_table_insert(pressure_audit.phis, phi, tag);
}

static void
pressure_audit_begin(agx_context *ctx)
{
   if (!debug_get_option_cached_agx_apple9_ra_audit() || ctx->stage != MESA_SHADER_FRAGMENT)
      return;
   assert(!pressure_audit.ctx);
   pressure_audit.ctx = ctx;
   pressure_audit.phis = _mesa_pointer_hash_table_create(NULL);
   pressure_audit.next_phi = pressure_audit.next_copy = 0;
   ++pressure_audit.serial;
   fprintf(stderr, "RA_AUDIT_BEGIN id=%u alloc=%u\n", pressure_audit.serial, ctx->alloc);
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, phi)
         pressure_tag_phi(ctx, phi, "original");
   }
}

static void
pressure_audit_spilled(agx_context *ctx)
{
   if (pressure_audit.ctx != ctx)
      return;
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, phi)
         pressure_tag_phi(ctx, phi, "spill-repair");
   }
   fprintf(stderr, "RA_AUDIT_POSTSPILL_BEGIN id=%u\n", pressure_audit.serial);
   agx_print_shader(ctx, stderr);
   fprintf(stderr, "RA_AUDIT_POSTSPILL_END id=%u\n", pressure_audit.serial);
}

static void
pressure_audit_phis(agx_context *ctx)
{
   if (pressure_audit.ctx != ctx)
      return;
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, phi) {
         struct hash_entry *entry = _mesa_hash_table_search(pressure_audit.phis, phi);
         struct pressure_phi_tag *tag = entry ? entry->data : NULL;
         fprintf(stderr, "RA_AUDIT_PHI id=%u phi=%u origin=%s block=%u loop=%u dst=%u reg=%u mem=%u width=%u src=",
                 pressure_audit.serial, tag ? tag->id : 0, tag ? tag->origin : "unknown",
                 block->index, block->loop_header, phi->dest[0].value,
                 phi->dest[0].reg, phi->dest[0].memory, agx_channels(phi->dest[0]));
         agx_foreach_src(phi, s)
            fprintf(stderr, "%s%u:%u:%u", s ? "," : "", phi->src[s].value,
                    phi->src[s].reg, phi->src[s].memory);
         fputc('\n', stderr);
      }
   }
}

static void
pressure_audit_edge(agx_context *ctx, agx_block *pred, agx_block *succ,
                    agx_instr *phi, unsigned s)
{
   if (pressure_audit.ctx != ctx)
      return;
   struct hash_entry *entry = _mesa_hash_table_search(pressure_audit.phis, phi);
   struct pressure_phi_tag *tag = entry ? entry->data : NULL;
   fprintf(stderr, "RA_AUDIT_EDGE id=%u phi=%u origin=%s pred=%u succ=%u dst=%u src=%u mem=%u srcmem=%u type=%u width=%u\n",
           pressure_audit.serial, tag ? tag->id : 0, tag ? tag->origin : "unknown",
           pred->index, succ->index, phi->dest[0].value, phi->src[s].value,
           phi->dest[0].memory, phi->src[s].memory, phi->src[s].type,
           agx_channels(phi->dest[0]));
}

static void
pressure_audit_copies(agx_builder *b, struct agx_copy *copies, unsigned n,
                      const char *where)
{
   if (pressure_audit.ctx != b->shader || !n) {
      agx_emit_parallel_copies(b, copies, n);
      return;
   }
   struct list_head *before = b->cursor.option == agx_cursor_after_block
      ? b->cursor.block->instructions.prev
      : b->cursor.option == agx_cursor_before_instr
         ? b->cursor.instr->link.prev : &b->cursor.instr->link;
   unsigned block_index = ~0u;
   agx_foreach_block(b->shader, block) {
      if (b->cursor.option == agx_cursor_after_block) {
         if (b->cursor.block == block) block_index = block->index;
      } else {
         agx_foreach_instr_in_block(block, I) {
            if (I == b->cursor.instr) block_index = block->index;
         }
      }
   }
   unsigned id = ++pressure_audit.next_copy;
   fprintf(stderr, "RA_AUDIT_PCOPY id=%u copy=%u where=%s block=%u n=%u pairs=",
           pressure_audit.serial, id, where, block_index, n);
   for (unsigned i = 0; i < n; ++i)
      fprintf(stderr, "%s%u:%u:%u:%u:%u:%u", i ? "," : "", copies[i].dest,
              copies[i].src.value, copies[i].dest_mem, copies[i].src.memory,
              copies[i].src.type, copies[i].src.size);
   fputc('\n', stderr);
   agx_emit_parallel_copies(b, copies, n);
   unsigned swaps = 0, moves = 0, saves = 0, fills = 0;
   if (b->cursor.option == agx_cursor_after_instr && &b->cursor.instr->link != before) {
      struct list_head *end = b->cursor.instr->link.next;
      for (struct list_head *link = before->next; link != end; link = link->next) {
         agx_instr *I = list_entry(link, agx_instr, link);
         swaps += I->op == AGX_OPCODE_SWAP;
         if (I->op == AGX_OPCODE_MOV) {
            moves += !I->src[0].memory && !I->dest[0].memory;
            saves += I->dest[0].memory;
            fills += I->src[0].memory;
         }
      }
   }
   fprintf(stderr, "RA_AUDIT_PCOPY_RESULT id=%u copy=%u where=%s block=%u swaps=%u moves=%u saves=%u fills=%u\n",
           pressure_audit.serial, id, where, block_index, swaps, moves, saves, fills);
}

static void
pressure_audit_end(agx_context *ctx)
{
   if (pressure_audit.ctx != ctx)
      return;
   fprintf(stderr, "RA_AUDIT_END id=%u\n", pressure_audit.serial);
   _mesa_hash_table_destroy(pressure_audit.phis, NULL);
   pressure_audit.ctx = NULL;
}

#define agx_emit_parallel_copies(b, c, n) pressure_audit_copies(b, c, n, __func__)
