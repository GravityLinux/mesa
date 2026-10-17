#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile the actual active-batch budget helper with small host fixtures."""
from pathlib import Path
import subprocess
import tempfile
root = next(p for p in Path(__file__).resolve().parents if (p / 'src/gallium/drivers/asahi/agx_batch.c').exists())
s = (root / 'src/gallium/drivers/asahi/agx_batch.c').read_text()
a = s.index('void\nagx_flush_over_budget_batches(')
b = s.index('\nvoid\nagx_flush_all(', a)
helper = s[a:b]
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
struct screen { uint64_t batch_memory_budget; };
struct agx_batch { uint64_t bo_bytes; };
struct agx_context {
 struct { struct screen *screen; } base;
 struct { struct agx_batch slots[4]; } batches;
 struct agx_batch *batch;
 unsigned draw_depth, active, flushed;
};
#define agx_screen(x) (x)
#define foreach_active(ctx,i) for ((i)=0; (i)<4; ++(i)) if ((ctx)->active & (1u<<(i)))
void agx_flush_over_budget_batches(struct agx_context *ctx);
static void agx_flush_batch_for_reason(struct agx_context *ctx, struct agx_batch *b, const char *why) {
 assert(!strcmp(why, "Batch memory budget")); (void)why;
 if (ctx->draw_depth != 1) abort();
 agx_flush_over_budget_batches(ctx); /* Simulate a recursive tile-helper draw. */
 unsigned i = b - ctx->batches.slots;
 ctx->flushed |= 1u<<i; ctx->active &= ~(1u<<i);
 if (ctx->batch == b) ctx->batch = 0;
}
''' + helper + r'''
/* CHECK remains active for both assert and NDEBUG compiler configurations. */
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
int main(void) {
 struct screen screen = { .batch_memory_budget = 512 };
 struct agx_context ctx = { .base.screen = &screen, .active = 3 };
 ctx.batches.slots[0].bo_bytes = 128; /* Cached upload/compute batch. */
 ctx.batches.slots[1].bo_bytes = 1024; /* Graphics batch left active. */
 ctx.batch = &ctx.batches.slots[0];
 agx_flush_over_budget_batches(&ctx);
 CHECK(ctx.flushed == 2 && ctx.active == 1 && ctx.batch == &ctx.batches.slots[0] && ctx.draw_depth == 0);
 ctx.active = 3; ctx.flushed = 0; ctx.batch = 0;
 agx_flush_over_budget_batches(&ctx); /* Upload flush cleared the cache. */
 CHECK(ctx.flushed == 2 && ctx.active == 1);
 ctx.active = 15; ctx.flushed = 0; ctx.draw_depth = 1;
 ctx.batches.slots[0].bo_bytes = 1024;
 ctx.batches.slots[2].bo_bytes = 512;
 ctx.batches.slots[3].bo_bytes = 4096;
 agx_flush_over_budget_batches(&ctx); /* Tile-helper recursion must not flush. */
 CHECK(ctx.flushed == 0 && ctx.active == 15);
 ctx.draw_depth = 0;
 agx_flush_over_budget_batches(&ctx);
 CHECK(ctx.flushed == 11 && ctx.active == 4); /* Exact boundary is unchanged. */
 ctx.flushed = 0;
 agx_flush_over_budget_batches(&ctx);
 CHECK(ctx.flushed == 0 && ctx.active == 4);
 return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='asahi-batch-budget-') as d:
 p=Path(d); (p/'test.c').write_text(fixture)
 for flags in [[], ['-DNDEBUG']]:
  subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror',*flags,str(p/'test.c'),'-o',str(p/'test')],check=True)
  subprocess.run([str(p/'test')],check=True)
print('PASS: compute/null cached batch, all active batches, boundary, recursion; assert + NDEBUG')
