/* SPDX-License-Identifier: MIT
 * Exercise the body cache with page-rounded host BOs. No GPU execution. */
#include "../hk_shader.c"
#include "../hk_cmd_buffer.h"
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
   bo->va = calloc(1, sizeof(*bo->va));
   CHECK(bo->va);
   bo->va->addr = 0x100000;
   agx_shader_body_init_bo(bo);
   bo->refcnt = 1; ++live_bos;
   return bo;
}
void __wrap_agx_bo_reference(struct agx_bo *bo) { CHECK(bo->refcnt); ++bo->refcnt; }
void __wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo) return;
   CHECK(bo->refcnt);
   if (--bo->refcnt) return;
   free(bo->va); free(bo->_map); free(bo); --live_bos;
}
static void check_cache(struct hk_device *dev, size_t size, unsigned count)
{
   uint8_t *code = calloc(1, size);
   CHECK(code);
   struct hk_shader shader = {.code_ptr = code, .code_size = size};
   struct agx_shader_body *held = NULL, *last = NULL;
   for (unsigned i = 0; i < count; ++i) {
      memcpy(code, &i, sizeof(i));
      last = hk_apple9_body_upload(dev, &shader);
      CHECK(last && !memcmp(last->bo->_map, code, size));
      if (!i) held = last;
      else agx_shader_body_unreference(&dev->dev, last);
      CHECK(dev->apple9_bodies.bytes <= HK_APPLE9_BODY_CACHE_BYTES);
      CHECK(dev->apple9_bodies.count <= HK_APPLE9_BODY_CACHE_SIZE);
   }
   /* The most recent entry still deduplicates. Oldest eviction never frees
    * memory still held by its shader/recording owner. */
   struct agx_shader_body *again = hk_apple9_body_upload(dev, &shader);
   CHECK(again == last);
   agx_shader_body_unreference(&dev->dev, again);
   CHECK(held->refcnt == 1 && *(uint32_t *)held->bo->_map == 0);
   agx_shader_body_unreference(&dev->dev, held);
   hk_apple9_bodies_finish(dev);
   CHECK(!dev->apple9_bodies.bytes && !dev->apple9_bodies.count && !live_bos);
   free(code);
}
static void check_recording(struct hk_device *dev)
{
   uint32_t code[16] = {0x12345678};
   struct hk_shader shader = {.code_ptr = code, .code_size = sizeof(code)};
   struct agx_shader_body *body = hk_apple9_body_upload(dev, &shader);
   CHECK(body && body->refcnt == 2);
   struct hk_cmd_buffer *a = calloc(1, sizeof(*a));
   struct hk_cmd_buffer *b = calloc(1, sizeof(*b));
   CHECK(a && b);
   a->vk.base.device = b->vk.base.device = &dev->vk;
   CHECK(hk_apple9_pin_entry(a, body));
   CHECK(hk_apple9_pin_entry(a, body));
   CHECK(hk_apple9_pin_entry(b, body));
   CHECK(body->entry_pins == 2 && body->refcnt == 4);
   agx_shader_body_unreference(&dev->dev, body);
   hk_apple9_bodies_finish(dev);
   CHECK(body->refcnt == 2 && live_bos == 1);
   hk_apple9_unpin_entries(a);
   CHECK(body->entry_pins == 1 && body->refcnt == 1);
   CHECK(!a->apple9_pinned->entries && !a->apple9_last_pin[0]);
   hk_apple9_unpin_entries(b);
   CHECK(!live_bos);
   hk_apple9_unpin_entries(b);
   _mesa_set_destroy(a->apple9_pinned, NULL);
   _mesa_set_destroy(b->apple9_pinned, NULL);
   free(a);
   free(b);
}
int main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   simple_mtx_init(&dev->dev.apple9_entry_lock, mtx_plain);
   check_recording(dev);
   check_cache(dev, 64, HK_APPLE9_BODY_CACHE_SIZE + 1);
   check_cache(dev, 1024 * 1024, 70);
   /* A single body above the byte budget remains usable without admission. */
   size_t size = HK_APPLE9_BODY_CACHE_BYTES + 16384;
   void *code = calloc(1, size);
   CHECK(code);
   struct hk_shader shader = {.code_ptr = code, .code_size = size};
   struct agx_shader_body *bo = hk_apple9_body_upload(dev, &shader);
   CHECK(bo && bo->refcnt == 1 && !dev->apple9_bodies.count);
   CHECK(!dev->apple9_bodies.bytes);
   agx_shader_body_unreference(&dev->dev, bo);
   hk_apple9_bodies_finish(dev);
   CHECK(!live_bos);
   free(code);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&dev->dev.apple9_entry_lock);
   free(dev);
   puts("PASS body-cache count/byte limits, reuse, held references and oversized bodies");
}
