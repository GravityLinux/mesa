/* SPDX-License-Identifier: MIT
 * Compare binding-only updates with complete preparation at fixed addresses.
 * The reference rebuilds all raster state; both arms emit real launch/VDM bytes.
 */
#include "agx_apple9.h"
#include "agx_device.h"

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); \
} } while (0)

struct arena {
   uint8_t bytes[16384];
   uint64_t gpu;
   unsigned used, calls, fail_at;
};

static struct agx_ptr
allocate(void *data, size_t size, size_t alignment)
{
   struct arena *a = data;
   if (++a->calls == a->fail_at)
      return (struct agx_ptr){0};
   unsigned off = ALIGN_POT(a->used, alignment);
   CHECK(off + size <= sizeof(a->bytes));
   a->used = off + size;
   return (struct agx_ptr){.cpu = a->bytes + off, .gpu = a->gpu + off};
}

static void
compare(const struct agx_apple9_render_pipeline *pipeline,
        const struct agx_apple9_uniform_draw draw[2],
        const struct arena usc[2], const struct arena context[2])
{
   CHECK(!memcmp(&draw[0], &draw[1], sizeof(draw[0])));
   CHECK(usc[0].used == usc[1].used && usc[0].calls == usc[1].calls);
   CHECK(context[0].used == context[1].used &&
         context[0].calls == context[1].calls);
   CHECK(!memcmp(usc[0].bytes, usc[1].bytes, sizeof(usc[0].bytes)));
   CHECK(!memcmp(context[0].bytes, context[1].bytes, sizeof(context[0].bytes)));
   uint8_t commands[2][256] = {0};
   for (unsigned i = 0; i < 2; ++i) {
      struct agx_apple9_render_pipeline p = *pipeline;
      p.ppp = draw[i].ppp;
      p.ppp_viewport_offset = draw[i].ppp_viewport_offset;
      p.ppp_viewport_words = draw[i].ppp_viewport_words;
      p.vertex_launch = draw[i].launch[0] / 64;
      p.pipeline_word = 0x01000000;
      CHECK(agx_apple9_emit_direct_draw(commands[i], &p, 3, 1, 0) -
               commands[i] == agx_apple9_direct_draw_size(&p));
   }
   CHECK(!memcmp(commands[0], commands[1], sizeof(commands[0])));
}

int
main(void)
{
   struct agx_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   dev->shader_base = 1ull << 40;
   simple_mtx_init(&dev->apple9_entry_lock, mtx_plain);
   list_inithead(&dev->apple9_entry_lru);
   struct agx_va va = {.addr = dev->shader_base + 0x10000};
   struct agx_bo bo = {.va = &va};
   agx_shader_body_init_bo(&bo);
   bo.apple9_body.entry_offset = 0x400;
   list_add(&bo.apple9_body.entry_link, &dev->apple9_entry_lru);
   uint8_t code[128] = {0};
   unsigned cases = 0, failures = 0;
   for (unsigned owned = 0; owned < 2; ++owned) {
      for (unsigned samples = 1; samples <= 4; samples *= 2) {
         for (unsigned viewports = 1; viewports <= 16; viewports *= 2) {
            for (unsigned state = 0; state < 8; ++state) {
               struct agx_apple9_render_stage vs = {
                  .bo = &bo, .binary = code, .binary_size = sizeof(code),
                  .publication_count_valid = true, .publication_count = 32,
                  .position_components = 4, .varying_components = 4,
                  .program_id = 1, .resource_count = 1,
               };
               struct agx_apple9_render_stage fs = vs;
               fs.program_id = 2;
               fs.position_components = 0;
               fs.render_targets = state & 1 ? 2 : 1;
               if (owned) {
                  fs.coefficients[0] = 0x60000;
                  fs.coefficients[1] = 0x602c0;
                  fs.coefficients[2] = 0x60580;
                  fs.coefficient_count = 4;
               }
               struct agx_apple9_render_pipeline pipeline;
               CHECK(agx_apple9_link_render_pipeline(&pipeline, &vs, &fs));
               pipeline.samples = samples;
               pipeline.primitive = state & 2 ? AGX_PRIMITIVE_TRIANGLE_FAN
                                              : AGX_PRIMITIVE_TRIANGLES;
               pipeline.color_formats[0] = AGX_CHANNELS_R8G8B8A8;
               pipeline.color_formats[1] = AGX_CHANNELS_R16G16B16A16;
               struct arena usc[2] = {{.gpu = dev->shader_base + 0x20000}};
               struct arena context[2] = {
                  {.gpu = AGX_APPLE9_RENDER_CONTEXT_BASE + 0x10000}};
               struct agx_apple9_uniform_draw draw[2] = {0};
               draw[0].vertex_table = 0x40000;
               draw[0].fragment_table = 0x50000;
               draw[0].texture_table[0] = 0x70000;
               draw[0].texture_table[1] = 0x80000;
               draw[0].sampler_table[0] = 0x90000;
               draw[0].sampler_table[1] = 0xa0000;
               draw[0].flatshade_first = state & 4;
               draw[0].object_type = AGX_OBJECT_TYPE_TRIANGLE;
               draw[0].reads_tile = state & 1;
               draw[0].uses_discard = state & 2;
               draw[0].writes_depth = state & 4;
               draw[0].pixel_interlock = state & 1;
               draw[0].disable_tri_merging = state & 4;
               draw[0].depth_control = 0x100;
               draw[0].depth_face[0] = 0x20000;
               draw[0].depth_face[1] = 0x30000;
               draw[0].stencil[0] = 0x100001;
               draw[0].stencil[1] = 0x200002;
               draw[0].raster_control = 0x10000;
               draw[0].occlusion_index = 3;
               draw[0].visibility_mode = 1;
               draw[0].scissor_index = 5;
               draw[0].depth_bias_index = 7;
               draw[0].scissor_min[0] = 17;
               draw[0].scissor_min[1] = 31;
               draw[0].scissor_max[0] = 100;
               draw[0].scissor_max[1] = 200;
               draw[0].viewport_count = viewports;
               for (unsigned i = 0; i < 3; ++i) {
                  draw[0].viewport_translate[i] = 0.5f + i;
                  draw[0].viewport_scale[i] = 2.5f + i;
               }
               for (unsigned v = 1; v < viewports; ++v)
                  for (unsigned i = 0; i < 6; ++i)
                     draw[0].viewports[v - 1][i] = v + i * 0.25f;
               CHECK(agx_apple9_prepare_draw_alloc(dev, &usc[0], allocate,
                  &context[0], allocate, &pipeline, &draw[0], NULL));
               usc[1] = usc[0]; context[1] = context[0]; draw[1] = draw[0];
               /* Repeated VS-only, FS-only, both and no-op updates, followed
                * by a full raster update, exercise predecessor ownership.
                */
               for (unsigned step = 0; step < 8; ++step) {
                  struct agx_apple9_uniform_draw previous = draw[0];
                  unsigned mask = step % 4;
                  uint64_t tables[] = {previous.vertex_table + ((mask & 1) ? 64 : 0),
                     previous.fragment_table + ((mask & 2) ? 64 : 0)};
                  unsigned usc_used = usc[0].used, ctx_used = context[0].used;
                  uint8_t old_usc[16384], old_ctx[16384];
                  memcpy(old_usc, usc[0].bytes, usc_used);
                  memcpy(old_ctx, context[0].bytes, ctx_used);
                  draw[0].vertex_table = tables[0];
                  draw[0].fragment_table = tables[1];
                  CHECK(agx_apple9_prepare_draw_alloc(dev, &usc[0], allocate,
                     &context[0], allocate, &pipeline, &draw[0], &previous));
                  CHECK(agx_apple9_update_draw_tables_alloc(dev, &usc[1], allocate,
                     &context[1], allocate, &pipeline, &draw[1], tables));
                  compare(&pipeline, draw, usc, context);
                  CHECK(!memcmp(old_usc, usc[1].bytes, usc_used));
                  CHECK(!memcmp(old_ctx, context[1].bytes, ctx_used));
                  ++cases;
               }
               for (unsigned arm = 0; arm < 2; ++arm) {
                  struct agx_apple9_uniform_draw previous = draw[arm];
                  draw[arm].viewport_scale[0] += 1.0f;
                  CHECK(agx_apple9_prepare_draw_alloc(dev, &usc[arm], allocate,
                     &context[arm], allocate, &pipeline, &draw[arm], &previous));
               }
               compare(&pipeline, draw, usc, context);
               for (unsigned fail = 0; fail < 5; ++fail) {
                  struct arena u = usc[1], c = context[1];
                  struct agx_apple9_uniform_draw d = draw[1];
                  const uint64_t tables[] = {d.vertex_table + 64, d.fragment_table + 64};
                  if (fail < 2) u.fail_at = u.calls + fail + 1;
                  if (fail == 2) c.fail_at = c.calls + 1;
                  if (fail == 3) c.gpu = AGX_APPLE9_RENDER_CONTEXT_BASE - c.used - 64;
                  if (fail == 4) c.gpu = AGX_APPLE9_RENDER_CONTEXT_BASE + UINT32_MAX - c.used;
                  CHECK(!agx_apple9_update_draw_tables_alloc(dev, &u, allocate,
                     &c, allocate, &pipeline, &d, tables));
                  CHECK(!memcmp(&d, &draw[1], sizeof(d)));
                  CHECK(!memcmp(u.bytes, usc[1].bytes, usc[1].used));
                  CHECK(!memcmp(c.bytes, context[1].bytes, context[1].used));
                  ++failures;
               }
            }
         }
      }
   }
   simple_mtx_destroy(&dev->apple9_entry_lock);
   free(dev);
   printf("PASS %u updates: exact launch/PPP/VDM bytes and allocation order; %u failure cases leave prepared state unchanged\n", cases, failures);
}
