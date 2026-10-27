/* SPDX-License-Identifier: MIT */
#include <unistd.h>
#include "../hk_shader.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); failures++; } } while (0)
static unsigned failures, attempts;

nir_shader *__wrap_nir_deserialize(void *mem_ctx,
                                  const nir_shader_compiler_options *options,
                                  struct blob_reader *blob);
nir_shader *
__wrap_nir_deserialize(void *mem_ctx, const nir_shader_compiler_options *options,
                       struct blob_reader *blob)
{
   p_atomic_inc(&attempts);
   /* Leave a window for another recording thread to wait on this variant. */
   usleep(20000);
   return NULL;
}

VkResult __wrap___vk_errorf(const void *obj, VkResult error, const char *file,
                            int line, const char *format, ...);
VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   return error;
}

struct thread_data {
   struct hk_device *dev;
   struct hk_api_shader *api[HK_APPLE9_GFX_STAGES];
   VkResult result;
};

static int
resolve(void *data)
{
   struct thread_data *t = data;
   struct hk_api_shader *out[HK_APPLE9_GFX_STAGES] = {0};
   t->result = hk_apple9_resolve_graphics(t->dev, t->api, NULL, out, false, false);
   return 0;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_api_shader api = {0};
   struct hk_apple9_deferred deferred = {0};
   if (!dev)
      return 1;
   simple_mtx_init(&dev->apple9_variants.lock, mtx_plain);
   static const uint8_t empty;
   deferred.nir = hk_apple9_nir_create(&empty, 0);
   api.apple9_deferred = &deferred;
   struct thread_data t = {.dev = dev, .api = {[MESA_SHADER_VERTEX] = &api}};
   resolve(&t);
   CHECK(t.result == VK_ERROR_OUT_OF_HOST_MEMORY && attempts == 1);
   resolve(&t);
   CHECK(t.result == VK_ERROR_OUT_OF_HOST_MEMORY && attempts == 2);
   puts("same variant after transient host OOM: compiler attempted again");

   enum { N = 6 };
   thrd_t threads[N];
   struct thread_data concurrent[N];
   unsigned before = attempts;
   for (unsigned i = 0; i < N; ++i) {
      concurrent[i] = t;
      CHECK(thrd_create(&threads[i], resolve, &concurrent[i]) == thrd_success);
   }
   for (unsigned i = 0; i < N; ++i) {
      thrd_join(threads[i], NULL);
      CHECK(concurrent[i].result == VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   CHECK(attempts == before + N);
   puts("concurrent callers retry without freeing a variant used by waiters");

   /* A permanent unsupported-stage combination remains cached. */
   api.apple9_deferred = NULL;
   hk_apple9_variants_finish(dev);
   resolve(&t);
   CHECK(t.result == VK_ERROR_FEATURE_NOT_PRESENT);
   api.apple9_deferred = &deferred;
   before = attempts;
   resolve(&t);
   CHECK(t.result == VK_ERROR_FEATURE_NOT_PRESENT && attempts == before);
   puts("permanent compile failure remains cached");
   hk_apple9_variants_finish(dev);
   simple_mtx_destroy(&dev->apple9_variants.lock);
   hk_apple9_nir_unref(deferred.nir);
   free(dev);
   return failures != 0;
}
