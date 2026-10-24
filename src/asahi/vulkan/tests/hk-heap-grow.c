/* SPDX-License-Identifier: MIT */
/* Geometry heap growth at submission: the backing grows to a command buffer's
 * minimum, and past what earlier work needed (counted as an overflow once per
 * new maximum), to the next power of two, never past the fixed VA range and
 * never moving it. BO creation and binding are wrapped: no GPU.
 */
#include "../hk_cmd_draw.c"

static unsigned failures, creates, binds, unbinds, live;
static bool fail_bind;
static uint64_t bound_B;
#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         failures++;                                                           \
      }                                                                        \
   } while (0)

struct agx_bo *__wrap_agx_bo_create(struct agx_device *dev, size_t size,
                                    unsigned align, enum agx_bo_flags flags,
                                    const char *label);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   CHECK(strcmp(label, "Geometry heap") == 0 && flags == 0);
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   bo->size = size;
   creates++;
   live++;
   return bo;
}

void __wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo);
void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   live--;
   free(bo);
}

int __wrap_agx_bo_bind(struct agx_device *dev, struct agx_bo *bo,
                       uint64_t addr, size_t size_B, uint64_t offset_B,
                       uint32_t flags);
int
__wrap_agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t addr,
                   size_t size_B, uint64_t offset_B, uint32_t flags)
{
   CHECK(addr == 1ull << 32);
   CHECK(offset_B == 0);
   if (flags & DRM_ASAHI_BIND_UNBIND) {
      unbinds++;
      bound_B = 0;
      return 0;
   }
   if (fail_bind && bo->size != bound_B) {
      fail_bind = false;
      return -1;
   }
   CHECK(size_B == bo->size);
   binds++;
   bound_B = size_B;
   return 0;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct agx_va va = {.flags = AGX_VA_USC, .addr = 1ull << 32,
                       .size_B = HK_HEAP_VA_MAX_B};
   struct agx_va rodata_va = {0};
   struct agx_bo rodata = {.va = &rodata_va};
   struct poly_heap map = {.base = va.addr};
   rodata._map = &map;
   dev->rodata.bo = &rodata;
   dev->heap_va = &va;
   list_inithead(&dev->vk.queues);

   const uint64_t mib = 1024 * 1024;
   CHECK(hk_heap_back(dev, HK_HEAP_MIN_B));
   CHECK(map.size == HK_HEAP_MIN_B - POLY_HEAP_SINK_B && map.base == va.addr);

   /* Fits: nothing happens. */
   map.bottom = (4 * mib) / POLY_HEAP_UNIT_B;
   hk_heap_prepare_submit(dev, HK_HEAP_MIN_B);
   CHECK(creates == 1 && dev->heap->size == HK_HEAP_MIN_B);

   /* A command buffer recording tessellation: its minimum, no overflow. */
   hk_heap_prepare_submit(dev, HK_HEAP_GEOMETRY_MIN_B);
   CHECK(dev->heap->size == HK_HEAP_GEOMETRY_MIN_B && !dev->heap_overflows);
   CHECK(map.size == HK_HEAP_GEOMETRY_MIN_B - POLY_HEAP_SINK_B);
   CHECK(live == 1 && unbinds == 1 && bound_B == HK_HEAP_GEOMETRY_MIN_B);

   /* An earlier render needed 200 MiB (kept by the reset in high_water):
    * one overflow, and the heap grows to 256 MiB. */
   map.high_water = (200 * mib) / POLY_HEAP_UNIT_B;
   map.bottom = 0;
   hk_heap_prepare_submit(dev, 0);
   CHECK(dev->heap_overflows == 1 && dev->heap->size == 256 * mib);
   hk_heap_prepare_submit(dev, 0);
   CHECK(dev->heap_overflows == 1 && creates == 3);

   /* Past the VA range: the backing stops at the range, and the same need is
    * counted once however many submissions see it. */
   map.bottom = (3000 * mib) / POLY_HEAP_UNIT_B;
   hk_heap_prepare_submit(dev, 0);
   hk_heap_prepare_submit(dev, 0);
   CHECK(dev->heap_overflows == 2 && dev->heap->size == HK_HEAP_VA_MAX_B);
   CHECK(map.size == HK_HEAP_VA_MAX_B - POLY_HEAP_SINK_B);

   /* A failed bind keeps the old backing mapped. */
   struct agx_bo *old = dev->heap;
   uint64_t old_size = map.size;
   dev->heap_va->size_B = 2 * HK_HEAP_VA_MAX_B;
   fail_bind = true;
   hk_heap_prepare_submit(dev, 0);
   CHECK(dev->heap == old && map.size == old_size && bound_B == old->size);
   CHECK(live == 1);

   /* Decay: a period has passed on an idle device (no queues here) needing
    * 4 MiB, so the 1 GiB backing shrinks to the minimum and high_water clears,
    * and a later spike is counted again. */
   dev->heap_va->size_B = HK_HEAP_VA_MAX_B;
   map.high_water = 0;
   map.bottom = (4 * mib) / POLY_HEAP_UNIT_B;
   dev->heap_period_need_B = 0;
   dev->heap_period_start_ns = 0;
   hk_heap_prepare_submit(dev, 0);
   CHECK(dev->heap->size == HK_HEAP_MIN_B);
   CHECK(map.size == HK_HEAP_MIN_B - POLY_HEAP_SINK_B && map.base == va.addr);
   CHECK(map.high_water == 0 && map.bottom == 0 && !dev->heap_reported_B);

   /* Within the new period, a tessellation command buffer grows it again, and
    * it does not shrink back yet. */
   hk_heap_prepare_submit(dev, HK_HEAP_GEOMETRY_MIN_B);
   hk_heap_prepare_submit(dev, 0);
   CHECK(dev->heap->size == HK_HEAP_GEOMETRY_MIN_B && live == 1);

   agx_bo_unreference(&dev->dev, dev->heap);
   printf("geometry heap growth: %u backings, %u overflows\n", creates,
          dev->heap_overflows);
   free(dev);
   return failures != 0;
}
