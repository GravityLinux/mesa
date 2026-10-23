/* SPDX-License-Identifier: MIT
 * Compare object-owned interpolation state with draw-time packing using fixed
 * semantic allocation addresses. No shader compilation or GPU is involved. */
#include "agx_apple9.h"
#include "agx_device.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)

struct arena {
   uint8_t bytes[8192], coefficients[AGX_APPLE9_CF_BINDINGS_SIZE];
   uint64_t gpu, cf_gpu;
   unsigned used, cf_uploads;
};

static struct agx_ptr
allocate(void *data, size_t size, size_t alignment)
{
   struct arena *a = data;
   if (size == AGX_APPLE9_CF_BINDINGS_SIZE) {
      ++a->cf_uploads;
      return (struct agx_ptr){.cpu = a->coefficients, .gpu = a->cf_gpu};
   }
   unsigned off = ALIGN_POT(a->used, alignment);
   CHECK(off + size <= sizeof(a->bytes));
   a->used = off + size;
   return (struct agx_ptr){.cpu = a->bytes + off, .gpu = a->gpu + off};
}

int main(void)
{
   struct agx_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   dev->shader_base = 1ull << 40;
   simple_mtx_init(&dev->apple9_entry_lock, mtx_plain);
   list_inithead(&dev->apple9_entry_lru);
   struct agx_va va = {.addr = dev->shader_base + 0x10000};
   struct agx_bo bo = {.va = &va, .apple9_entry_offset = 0x400};
   list_add(&bo.apple9_entry_link, &dev->apple9_entry_lru);
   uint8_t code[128] = {0};
   unsigned cases = 0;
   const unsigned components[] = {0, 1, 4, 65, AGX_APPLE9_MAX_VARYING_COMPONENTS};
   for (unsigned c = 0; c < ARRAY_SIZE(components); ++c) {
      for (unsigned mode = 0; mode < 4; ++mode) {
         for (unsigned extras = 0; extras < 16; ++extras) {
            struct agx_apple9_render_stage vs = {
               .bo = &bo, .binary = code, .binary_size = sizeof(code),
               .publication_count_valid = true, .publication_count = 32,
               .position_components = 4, .varying_components = components[c],
               .program_id = 1,
            };
            struct agx_apple9_render_stage fs = vs;
            fs.position_components = 0;
            fs.program_id = 2;
            fs.apple9_reads_z = extras & 1;
            fs.reads_point_coord = extras & 2;
            fs.reads_primitive_id = extras & 4;
            fs.barycentric_mask = (extras & 8) ? 3 : 0;
            fs.per_vertex_count = MIN2(components[c], 3);
            for (unsigned i = 0; i < fs.per_vertex_count; ++i)
               fs.per_vertex[i] = i;
            for (unsigned i = 0; i < components[c]; ++i) {
               if (mode == 1 || (mode == 3 && i % 3 == 0))
                  agx_apple9_interp_mask_set(&fs.apple9_flat_mask, i);
               if (mode == 2 || (mode == 3 && i % 3 == 1))
                  agx_apple9_interp_mask_set(&fs.apple9_linear_mask, i);
            }
            struct agx_apple9_render_stage owned = fs;
            uint8_t immutable[AGX_APPLE9_CF_ALLOCATION_SIZE];
            memset(immutable, 0xa5, sizeof(immutable));
            uint64_t cf_gpu = dev->shader_base + 0x30000;
            unsigned size = agx_apple9_coefficients_size(&owned);
            agx_apple9_init_coefficients(dev, &owned,
               (struct agx_ptr){.cpu = immutable, .gpu = cf_gpu});
            for (unsigned i = size; i < sizeof(immutable); ++i)
               CHECK(immutable[i] == 0xa5);
            for (unsigned pv = 0; pv < 3; ++pv) {
               unsigned offset = owned.coefficients[pv] - (cf_gpu - dev->shader_base);
               struct arena usc[2] = {
                  {.gpu = dev->shader_base + 0x20000, .cf_gpu = cf_gpu + offset},
                  {.gpu = dev->shader_base + 0x20000, .cf_gpu = cf_gpu + offset},
               };
               struct arena context[2] = {
                  {.gpu = AGX_APPLE9_RENDER_CONTEXT_BASE + 0x10000},
                  {.gpu = AGX_APPLE9_RENDER_CONTEXT_BASE + 0x10000},
               };
               struct agx_apple9_uniform_draw draw[2] = {0};
               uint8_t commands[2][256] = {0};
               for (unsigned arm = 0; arm < 2; ++arm) {
                  struct agx_apple9_render_pipeline pipeline;
                  CHECK(agx_apple9_link_render_pipeline(&pipeline, &vs, arm ? &owned : &fs));
                  pipeline.primitive = pv == 1 ? AGX_PRIMITIVE_TRIANGLE_FAN : AGX_PRIMITIVE_TRIANGLES;
                  pipeline.flatshade_first = pv != 2;
                  draw[arm].flatshade_first = pv != 2;
                  draw[arm].object_type = AGX_OBJECT_TYPE_TRIANGLE;
                  draw[arm].scissor_max[0] = draw[arm].scissor_max[1] = 32;
                  draw[arm].vertex_table = 0x40000;
                  draw[arm].fragment_table = 0x50000;
                  CHECK(agx_apple9_prepare_draw_alloc(dev, &usc[arm], allocate,
                     &context[arm], allocate, &pipeline, &draw[arm], NULL));
                  CHECK(draw[arm].provoking_vertex == pv);
                  pipeline.ppp = draw[arm].ppp;
                  pipeline.vertex_launch = draw[arm].launch[0] / 64;
                  pipeline.pipeline_word = 0x01000000;
                  uint8_t *end = agx_apple9_emit_direct_draw(commands[arm], &pipeline, 3, 1, 0);
                  CHECK(end - commands[arm] == agx_apple9_direct_draw_size(&pipeline));
               }
               CHECK(usc[0].cf_uploads == 1 && usc[1].cf_uploads == 0);
               CHECK(!memcmp(usc[0].coefficients, immutable + offset, AGX_APPLE9_CF_BINDINGS_SIZE));
               CHECK(draw[0].cf_count == draw[1].cf_count);
               CHECK(!memcmp(draw[0].ppp_record, draw[1].ppp_record, sizeof(draw[0].ppp_record)));
               CHECK(usc[0].used == usc[1].used && context[0].used == context[1].used);
               CHECK(!memcmp(usc[0].bytes, usc[1].bytes, usc[0].used));
               CHECK(!memcmp(context[0].bytes, context[1].bytes, context[0].used));
               CHECK(!memcmp(commands[0], commands[1], sizeof(commands[0])));
               ++cases;
            }
         }
      }
   }
   simple_mtx_destroy(&dev->apple9_entry_lock);
   free(dev);
   printf("PASS %u interpolation/primitive cases: coefficient payload, roots, PPP and VDM bytes identical at fixed addresses\n", cases);
}
