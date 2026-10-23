/* SPDX-License-Identifier: MIT
 * Exercise the body cache with page-rounded host BOs. No GPU execution. */
#include "../hk_cmd_buffer.h"
#include "../hk_shader.c"
#include "util/u_thread.h"
#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)
static unsigned live_bos;
static bool fail_create, fail_map;
static uint64_t next_va = 0x100000;
static bool gate_uploads;
static util_barrier upload_gate;
void __wrap_agx_bo_note_cpu_write(struct agx_bo *, uint64_t, uint64_t);
void
__wrap_agx_bo_note_cpu_write(struct agx_bo *bo, uint64_t offset, uint64_t size)
{
   if (bo)
      CHECK(offset + size <= bo->size);
   if (gate_uploads)
      util_barrier_wait(&upload_gate);
}
static void *
map_fail(struct agx_device *dev, struct agx_bo *bo, void *addr, size_t size)
{
   return NULL;
}
struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_reference(struct agx_bo *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned alignment,
                     enum agx_bo_flags flags, const char *label)
{
   if (fail_create)
      return NULL;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->size = ALIGN_POT(size, 16384);
   bo->dev = dev;
   bo->_map = fail_map ? NULL : calloc(1, bo->size);
   CHECK(fail_map || bo->_map);
   bo->va = calloc(1, sizeof(*bo->va));
   CHECK(bo->va);
   bo->va->addr = next_va;
   next_va += bo->size;
   agx_shader_body_init_bo(bo);
   bo->refcnt = 1;
   ++live_bos;
   return bo;
}
void
__wrap_agx_bo_reference(struct agx_bo *bo)
{
   CHECK(bo->refcnt);
   ++bo->refcnt;
}
void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   CHECK(bo->refcnt);
   if (--bo->refcnt)
      return;
   free(bo->va);
   free(bo->_map);
   free(bo);
   --live_bos;
}
static void
check_cache(struct hk_device *dev, size_t size, unsigned count)
{
   uint8_t *code = calloc(1, size);
   CHECK(code);
   struct hk_shader shader = {.code_ptr = code, .code_size = size};
   struct agx_shader_body *held = NULL, *last = NULL;
   for (unsigned i = 0; i < count; ++i) {
      memcpy(code, &i, sizeof(i));
      last = hk_apple9_body_upload(dev, &shader);
      CHECK(last &&
            !memcmp((char *)last->bo->_map + last->addr - last->bo->va->addr,
                    code, size));
      if (!i)
         held = last;
      else
         agx_shader_body_unreference(&dev->dev, last);
      CHECK(dev->apple9_bodies.bytes <= HK_APPLE9_BODY_CACHE_BYTES);
      CHECK(dev->apple9_bodies.count <= HK_APPLE9_BODY_CACHE_SIZE);
      CHECK(dev->apple9_bodies.backing_bytes <= HK_APPLE9_BODY_CACHE_BYTES);
   }
   /* The most recent entry still deduplicates. Oldest eviction never frees
    * memory still held by its shader/recording owner. */
   struct agx_shader_body *again = hk_apple9_body_upload(dev, &shader);
   CHECK(again == last);
   agx_shader_body_unreference(&dev->dev, again);
   CHECK(held->refcnt == 1 &&
         *(uint32_t *)((char *)held->bo->_map + held->addr -
                       held->bo->va->addr) == 0);
   agx_shader_body_unreference(&dev->dev, held);
   hk_apple9_bodies_finish(dev);
   CHECK(!dev->apple9_bodies.bytes && !dev->apple9_bodies.count && !live_bos);
   free(code);
}
static void
check_recording(struct hk_device *dev)
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
struct recording {
   struct hk_cmd_buffer cmd;
   uint8_t launch[512];
};
static struct agx_ptr
alloc_launch(void *priv, size_t size, size_t align)
{
   struct recording *r = priv;
   CHECK(size <= sizeof(r->launch));
   return (struct agx_ptr){r->launch, 0x80000};
}
static void
launch_body(struct hk_device *dev, struct recording *r,
            struct agx_shader_body *body)
{
   struct agx_apple9_render_stage stage = {
      .bo = body->bo,
      .body = body,
      .publication_count_valid = true,
      .publication_count = 4,
      .binary_size = 64,
   };
   uint32_t launch;
   CHECK(agx_apple9_prepare_graphics_stage(
      &dev->dev, &r->cmd, alloc_launch, &stage, false, 0, 0, 0, 0, 1, &launch));
}
static void
check_slices_and_entries(struct hk_device *dev)
{
   uint8_t code[65] = {0x42}, data[19] = {0x24};
   struct hk_shader shader = {.code_ptr = code,
                              .code_size = sizeof(code),
                              .data_ptr = data,
                              .data_size = sizeof(data)};
   struct agx_shader_body *a = hk_apple9_body_upload(dev, &shader);
   code[0]++;
   struct agx_shader_body *b = hk_apple9_body_upload(dev, &shader);
   CHECK(a && b && a != b && a->bo == b->bo && a->addr != b->addr);
   CHECK(!(a->addr % 64) && !(b->addr % 64) && live_bos == 1);
   char *map = (char *)b->bo->_map + b->addr - b->bo->va->addr;
   CHECK(!memcmp(map, code, sizeof(code)));
   CHECK(!memcmp(map + 128, data, sizeof(data)));
   for (unsigned i = 65; i < 128; ++i)
      CHECK(map[i] == 0);
   for (unsigned i = 147; i < 192; ++i)
      CHECK(map[i] == 0);
   struct recording *r = calloc(1, sizeof(*r));
   struct recording *other = calloc(1, sizeof(*other));
   CHECK(r && other);
   r->cmd.vk.base.device = other->cmd.vk.base.device = &dev->vk;
   dev->dev.apple9_pin_entry = hk_apple9_pin_entry;
   list_inithead(&dev->dev.apple9_entry_lru);
   /* Two entries force real entry eviction on the third distinct body. */
   util_vma_heap_init(&dev->dev.apple9_entry_heap, AGX_APPLE9_ENTRY_HEADER_SIZE,
                      2 * AGX_APPLE9_ENTRY_BLOCK_SIZE);
   dev->dev.apple9_entry_heap.alloc_high = false;
   dev->dev.apple9_entry_map = calloc(1, 4096);
   CHECK(dev->dev.apple9_entry_map);
   launch_body(dev, r, a);
   launch_body(dev, other, b);
   CHECK(a->entry_offset && b->entry_offset &&
         a->entry_offset != b->entry_offset);
   uint32_t a_entry = a->entry_offset, b_entry = b->entry_offset;
   hk_apple9_unpin_entries(&other->cmd);
   code[0]++;
   struct agx_shader_body *c = hk_apple9_body_upload(dev, &shader);
   launch_body(dev, other, c);
   CHECK(a->entry_offset == a_entry && !b->entry_offset);
   CHECK(c->entry_offset == b_entry);
   hk_apple9_unpin_entries(&other->cmd);
   uint64_t old_b_addr = b->addr;
   agx_shader_body_unreference(&dev->dev, b);
   agx_shader_body_unreference(&dev->dev, c);
   hk_apple9_bodies_finish(dev);
   CHECK(a->entry_pins == 1 && a->refcnt == 2 && live_bos == 1);
   /* Reuse a freed hole without moving or overwriting the pinned neighbor. */
   code[0]++;
   struct agx_shader_body *d = hk_apple9_body_upload(dev, &shader);
   CHECK(d->addr == old_b_addr && d->bo == a->bo);
   CHECK(*(uint8_t *)a->bo->_map == 0x42 && a->entry_offset == a_entry);
   agx_shader_body_unreference(&dev->dev, a);
   hk_apple9_unpin_entries(&r->cmd);
   agx_shader_body_unreference(&dev->dev, d);
   hk_apple9_bodies_finish(dev);
   CHECK(!live_bos && list_is_empty(&dev->dev.apple9_entry_lru));
   CHECK(list_is_empty(&dev->apple9_bodies.slabs));
   util_vma_heap_finish(&dev->dev.apple9_entry_heap);
   free(dev->dev.apple9_entry_map);
   dev->dev.apple9_entry_map = NULL;
   _mesa_set_destroy(r->cmd.apple9_pinned, NULL);
   _mesa_set_destroy(other->cmd.apple9_pinned, NULL);
   free(r);
   free(other);
}
static void
check_failures(struct hk_device *dev)
{
   uint32_t code[16] = {0x1234};
   struct hk_shader shader = {.code_ptr = code, .code_size = sizeof(code)};
   fail_create = true;
   CHECK(!hk_apple9_body_upload(dev, &shader));
   fail_create = false;
   fail_map = true;
   dev->dev.ops.bo_mmap = map_fail;
   CHECK(!hk_apple9_body_upload(dev, &shader));
   fail_map = false;
   CHECK(!live_bos && !dev->apple9_bodies.count);
   CHECK(list_is_empty(&dev->apple9_bodies.slabs));
   hk_apple9_bodies_finish(dev);
}
static void
check_fragmented_budget(struct hk_device *dev)
{
   /* Keep one tiny cached body per slab, with uncached owners filling the
    * rest. Rounded body charges alone would allow 4,096 such slabs (256 MiB).
    * The independent backing budget must evict the oldest cached slices. */
   const unsigned count =
      HK_APPLE9_BODY_CACHE_BYTES / HK_APPLE9_BODY_SLAB_SIZE + 1;
   struct agx_shader_body **held = calloc(count, sizeof(*held));
   uint8_t code[HK_APPLE9_BODY_SLAB_SIZE - 64] = {0};
   CHECK(held);
   for (unsigned i = 0; i < count; ++i) {
      memcpy(code, &i, sizeof(i));
      struct hk_shader shader = {.code_ptr = code, .code_size = sizeof(code)};
      /* Reserve a held, uncached filler using the production allocator. */
      struct hk_apple9_body *filler = calloc(1, sizeof(*filler));
      CHECK(filler);
      filler->size = sizeof(code);
      CHECK(hk_apple9_body_alloc(dev, filler));
      held[i] = &filler->base;
      shader.code_size = 64;
      struct agx_shader_body *small = hk_apple9_body_upload(dev, &shader);
      CHECK(small && small->bo == held[i]->bo);
      agx_shader_body_unreference(&dev->dev, small);
      CHECK(dev->apple9_bodies.backing_bytes <= HK_APPLE9_BODY_CACHE_BYTES);
   }
   CHECK(dev->apple9_bodies.count == count - 1);
   CHECK(dev->apple9_bodies.bytes == (count - 1) * AIL_PAGESIZE);
   hk_apple9_bodies_finish(dev);
   for (unsigned i = 0; i < count; ++i)
      agx_shader_body_unreference(&dev->dev, held[i]);
   free(held);
   CHECK(!live_bos && list_is_empty(&dev->apple9_bodies.slabs));
}
struct upload_thread {
   struct hk_device *dev;
   struct hk_shader shader;
   struct agx_shader_body *body;
};
static int
upload(void *arg)
{
   struct upload_thread *t = arg;
   t->body = hk_apple9_body_upload(t->dev, &t->shader);
   return 0;
}
static void
check_concurrent_publication(struct hk_device *dev)
{
   uint32_t code[16] = {0x76543210};
   struct upload_thread args[2];
   thrd_t threads[2];
   util_barrier_init(&upload_gate, 2);
   gate_uploads = true;
   for (unsigned i = 0; i < 2; ++i) {
      args[i] = (struct upload_thread){
         .dev = dev,
         .shader = {.code_ptr = code, .code_size = sizeof(code)},
      };
      CHECK(thrd_create(&threads[i], upload, &args[i]) == thrd_success);
   }
   for (unsigned i = 0; i < 2; ++i)
      CHECK(thrd_join(threads[i], NULL) == thrd_success);
   gate_uploads = false;
   util_barrier_destroy(&upload_gate);
   struct agx_shader_body *a = args[0].body, *b = args[1].body;
   CHECK(a && b && a != b && a->bo == b->bo && a->addr != b->addr);
   CHECK(a->refcnt + b->refcnt == 3 && dev->apple9_bodies.count == 1);
   for (unsigned i = 0; i < 2; ++i) {
      struct agx_shader_body *body = args[i].body;
      CHECK(!memcmp((char *)body->bo->_map + body->addr - body->bo->va->addr,
                    code, sizeof(code)));
      agx_shader_body_unreference(&dev->dev, body);
   }
   hk_apple9_bodies_finish(dev);
   CHECK(!live_bos && list_is_empty(&dev->apple9_bodies.slabs));
}
static void
check_boundary(struct hk_device *dev)
{
   uint8_t code[HK_APPLE9_BODY_SLAB_SIZE + 1] = {1};
   for (unsigned size = sizeof(code) - 1; size <= sizeof(code); ++size) {
      struct hk_shader shader = {.code_ptr = code, .code_size = size};
      struct agx_shader_body *base = hk_apple9_body_upload(dev, &shader);
      CHECK(base);
      struct hk_apple9_body *body =
         container_of(base, struct hk_apple9_body, base);
      CHECK(!!body->slab == (size == HK_APPLE9_BODY_SLAB_SIZE));
      CHECK(!memcmp(base->bo->_map, code, size));
      agx_shader_body_unreference(&dev->dev, base);
      hk_apple9_bodies_finish(dev);
      CHECK(!live_bos && list_is_empty(&dev->apple9_bodies.slabs));
   }
}
int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   list_inithead(&dev->apple9_bodies.slabs);
   simple_mtx_init(&dev->dev.apple9_entry_lock, mtx_plain);
   check_failures(dev);
   check_recording(dev);
   check_slices_and_entries(dev);
   check_fragmented_budget(dev);
   check_concurrent_publication(dev);
   check_boundary(dev);
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
   puts(
      "PASS packed shader bodies: budgets, bytes, slice reuse, entry eviction, recording lifetime, OOM and teardown");
}
