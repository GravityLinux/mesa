/* SPDX-License-Identifier: MIT */
#include "agx_apple9.h"
#include "agx_device.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
struct allocation { uint8_t data[512]; uint64_t gpu; size_t size; };
static struct agx_ptr
allocate(void *priv, size_t size, size_t alignment)
{
   struct allocation *a = priv;
   CHECK(size <= sizeof(a->data) && alignment == 64);
   a->size = size;
   return (struct agx_ptr){a->data, a->gpu};
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
   struct agx_apple9_render_stage shader = {
      .bo = &bo, .publication_count_valid = true,
      .publication_count = 32, .binary_size = 128,
   };
   for (unsigned cached = 0; cached < 2; ++cached) {
      for (unsigned fragment = 0; fragment < 2; ++fragment) {
         for (unsigned samples = 1; samples <= 4; samples *= 2) {
            for (unsigned preamble = 0; preamble < 2; ++preamble) {
               struct agx_apple9_stage_launch_cache cache;
               agx_apple9_stage_launch_cache_init(&cache);
               shader.launch_cache = cached ? &cache : NULL;
               shader.preamble_size = preamble ? 64 : 0;
               shader.preamble_offset = preamble ? 64 : 0;
               struct allocation a;
               memset(&a, 0xa5, sizeof(a));
               a.gpu = dev->shader_base + 0x20000;
               uint32_t launch = 0;
               CHECK(agx_apple9_prepare_graphics_stage(
                  dev, &a, allocate, &shader, fragment, 0x30000, 0x40000,
                  0x50000, 16, samples, &launch));
               CHECK(a.size == 64 + AGX_APPLE9_GRAPHICS_LAUNCH_SIZE);
               CHECK(launch == a.gpu + 64 - dev->shader_base);
               const uint64_t roots[8] = {0x40000, 0x50000, 0x30000};
               CHECK(!memcmp(a.data, roots, sizeof(roots)));
               struct agx_apple9_launch_parameters p = {
                  .shader_base = dev->shader_base,
                  .entry_offset = bo.apple9_entry_offset,
                  .publication_count = shader.publication_count,
                  .resource_table = a.gpu, .launch_address = a.gpu + 64,
                  .tile_bytes = fragment ? 16 : 0,
                  .samples = fragment ? samples : 0,
                  .preamble_address = preamble ? va.addr + 64 : 0,
               };
               uint8_t reference[AGX_APPLE9_GRAPHICS_LAUNCH_SIZE];
               CHECK(agx_apple9_launch_build(reference, sizeof(reference),
                  fragment ? AGX_APPLE9_LAUNCH_FRAGMENT : AGX_APPLE9_LAUNCH_VERTEX, &p));
               CHECK(!memcmp(reference, a.data + 64, sizeof(reference)));
               for (unsigned i = a.size; i < sizeof(a.data); ++i)
                  CHECK(a.data[i] == 0xa5);
               agx_apple9_stage_launch_cache_finish(&cache);
            }
         }
      }
   }
   simple_mtx_destroy(&dev->apple9_entry_lock);
   free(dev);
   puts("PASS 24 graphics allocations: roots, full launch bytes, preambles and bounds");
}
