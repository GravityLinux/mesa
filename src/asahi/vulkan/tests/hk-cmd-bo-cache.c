/* SPDX-License-Identifier: MIT */
/* Command BO recycling contract: idle BOs are shared by every pool of a
 * device, the most recently returned BO is reused first, the idle list is
 * bounded by HK_CMD_BO_CACHE_BYTES for the whole device, and a releasing reset
 * or an explicit trim hands BOs back to the BO cache. BO creation and release
 * are wrapped: this host test has no GPU.
 */
#include "../hk_cmd_pool.c"

static unsigned failures, live, created;
#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static char fake_map[16];

struct agx_bo *__wrap_agx_bo_create(struct agx_device *dev, size_t size,
                                    unsigned align, enum agx_bo_flags flags,
                                    const char *label);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   CHECK(size == HK_CMD_BO_SIZE);
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   bo->dev = dev;
   bo->size = size;
   bo->_map = fake_map;
   live++;
   created++;
   return bo;
}

void __wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo);
void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   CHECK(live > 0);
   live--;
   free(bo);
}

void __wrap_agx_bo_cache_evict_all(struct agx_device *dev);
void
__wrap_agx_bo_cache_evict_all(struct agx_device *dev)
{
}

/* Record n BOs of a kind into a command buffer's list, as the uploader does. */
static void
record(struct hk_cmd_pool *pool, enum hk_cmd_pool_kind kind,
       struct list_head *bos, unsigned n)
{
   list_inithead(bos);
   for (unsigned i = 0; i < n; ++i) {
      struct hk_cmd_bo *bo;
      CHECK(hk_cmd_pool_alloc_bo(pool, kind, &bo) == VK_SUCCESS);
      list_addtail(&bo->link, bos);
   }
}

static int
run(void)
{
   const unsigned budget = HK_CMD_BO_CACHE_BYTES / HK_CMD_BO_SIZE;
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_pool *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b));
   if (!dev || !a || !b)
      return 1;
   dev->vk.alloc = *vk_default_allocator();
   dev->dev.chip = AGX_CHIP_G16G;
   a->vk.base.device = &dev->vk;
   b->vk.base.device = &dev->vk;

   /* A BO returned by one pool is reused by another, most recent first. */
   struct list_head bos;
   record(a, HK_CMD_POOL_MAIN, &bos, 2);
   struct hk_cmd_bo *last =
      list_last_entry(&bos, struct hk_cmd_bo, link);
   hk_cmd_pool_free_bo_list(a, HK_CMD_POOL_MAIN, &bos, false);
   CHECK(list_is_empty(&bos) && live == 2);
   CHECK(dev->cmd_bos.bytes == 2 * HK_CMD_BO_SIZE);
   struct hk_cmd_bo *reused;
   CHECK(hk_cmd_pool_alloc_bo(b, HK_CMD_POOL_MAIN, &reused) == VK_SUCCESS);
   CHECK(reused == last && created == 2);
   list_inithead(&bos);
   list_addtail(&reused->link, &bos);
   hk_cmd_pool_free_bo_list(b, HK_CMD_POOL_MAIN, &bos, false);

   /* Kinds are not mixed: a USC BO is never handed out as a main one. */
   struct hk_cmd_bo *usc;
   CHECK(hk_cmd_pool_alloc_bo(a, HK_CMD_POOL_USC, &usc) == VK_SUCCESS);
   CHECK(created == 3);
   list_inithead(&bos);
   list_addtail(&usc->link, &bos);
   hk_cmd_pool_free_bo_list(a, HK_CMD_POOL_USC, &bos, false);

   /* A spike in two pools: the device keeps at most the budget idle, and
    * everything past it is released at once. */
   struct list_head spike_a, spike_b;
   record(a, HK_CMD_POOL_MAIN, &spike_a, budget);
   record(b, HK_CMD_POOL_CONTEXT, &spike_b, budget);
   CHECK(dev->cmd_bos.bytes == HK_CMD_BO_SIZE); /* only the idle USC BO */
   hk_cmd_pool_free_bo_list(a, HK_CMD_POOL_MAIN, &spike_a, false);
   hk_cmd_pool_free_bo_list(b, HK_CMD_POOL_CONTEXT, &spike_b, false);
   CHECK(dev->cmd_bos.bytes == HK_CMD_BO_CACHE_BYTES);
   CHECK(live == budget);

   /* A releasing reset bypasses the idle list. */
   record(b, HK_CMD_POOL_MAIN, &bos, 3);
   const uint64_t idle = dev->cmd_bos.bytes;
   const unsigned before = live;
   hk_cmd_pool_free_bo_list(b, HK_CMD_POOL_MAIN, &bos, true);
   CHECK(live == before - 3 && dev->cmd_bos.bytes == idle);

   /* An explicit trim releases every idle BO of the device. */
   hk_cmd_bo_cache_trim(dev);
   CHECK(live == 0 && dev->cmd_bos.bytes == 0);
   for (unsigned i = 0; i < HK_CMD_POOL_COUNT; ++i)
      CHECK(dev->cmd_bos.free[i] == NULL);

   hk_cmd_bo_cache_finish(dev);
   printf("command BO cache: %u BOs created, budget %u BOs\n", created, budget);
   free(a);
   free(b);
   free(dev);
   return failures != 0;
}

int
main(void)
{
   return run();
}
