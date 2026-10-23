/* SPDX-License-Identifier: MIT
 * Exercise the body cache with page-rounded host BOs. No GPU execution. */
#include "../hk_shader.c"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
static unsigned live_bos;
struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned, enum agx_bo_flags, const char *);
void __wrap_agx_bo_reference(struct agx_bo *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned alignment,
                     enum agx_bo_flags flags, const char *label)
{
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->size = ALIGN_POT(size, 16384);
   bo->_map = calloc(1, bo->size);
   CHECK(bo->_map);
   bo->refcnt = 1; ++live_bos;
   return bo;
}
void __wrap_agx_bo_reference(struct agx_bo *bo) { CHECK(bo->refcnt); ++bo->refcnt; }
void __wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo) return;
   CHECK(bo->refcnt);
   if (--bo->refcnt) return;
   free(bo->_map); free(bo); --live_bos;
}
static void check_cache(struct hk_device *dev, size_t size, unsigned count)
{
   uint8_t *code = calloc(1, size);
   CHECK(code);
   struct hk_shader shader = {.code_ptr = code, .code_size = size};
   struct agx_bo *held = NULL, *last = NULL;
   for (unsigned i = 0; i < count; ++i) {
      memcpy(code, &i, sizeof(i));
      last = hk_apple9_body_bo(dev, &shader);
      CHECK(last && !memcmp(last->_map, code, size));
      if (!i) held = last;
      else agx_bo_unreference(&dev->dev, last);
      CHECK(dev->apple9_bodies.bytes <= HK_APPLE9_BODY_CACHE_BYTES);
      CHECK(dev->apple9_bodies.count <= HK_APPLE9_BODY_CACHE_SIZE);
   }
   /* The most recent entry still deduplicates. Oldest eviction never frees
    * memory still held by its shader/recording owner. */
   struct agx_bo *again = hk_apple9_body_bo(dev, &shader);
   CHECK(again == last);
   agx_bo_unreference(&dev->dev, again);
   CHECK(held->refcnt == 1 && *(uint32_t *)held->_map == 0);
   agx_bo_unreference(&dev->dev, held);
   hk_apple9_bodies_finish(dev);
   CHECK(!dev->apple9_bodies.bytes && !dev->apple9_bodies.count && !live_bos);
   free(code);
}
int main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   check_cache(dev, 64, HK_APPLE9_BODY_CACHE_SIZE + 1);
   check_cache(dev, 1024 * 1024, 70);
   /* A single body above the byte budget remains usable without admission. */
   size_t size = HK_APPLE9_BODY_CACHE_BYTES + 16384;
   void *code = calloc(1, size);
   CHECK(code);
   struct hk_shader shader = {.code_ptr = code, .code_size = size};
   struct agx_bo *bo = hk_apple9_body_bo(dev, &shader);
   CHECK(bo && bo->refcnt == 1 && !dev->apple9_bodies.count);
   CHECK(!dev->apple9_bodies.bytes);
   agx_bo_unreference(&dev->dev, bo);
   hk_apple9_bodies_finish(dev);
   CHECK(!live_bos);
   free(code);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   free(dev);
   puts("PASS body-cache count/byte limits, reuse, held references and oversized bodies");
}
