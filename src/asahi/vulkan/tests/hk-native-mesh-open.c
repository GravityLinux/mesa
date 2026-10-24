/* SPDX-License-Identifier: MIT */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "hk_apple9_mesh.h"
#include "agx_native_mesh.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static jmp_buf stopped;
static unsigned generation, probes, vm_creates;
static const char *before_text, *after_text;
static struct agx_device *opening_device;

/* Exercise agx_open_device itself, stopping before its first GPU allocation. */
drmVersionPtr __wrap_drmGetVersion(int fd);
drmVersionPtr __wrap_drmGetVersion(int fd)
{
   static drmVersion version = {.name = "asahi"};
   CHECK(fd == 101);
   return &version;
}
void __wrap_drmFreeVersion(drmVersionPtr version);
void __wrap_drmFreeVersion(drmVersionPtr version) { (void)version; }
int __wrap_drmIoctl(int fd, unsigned long request, void *arg);
int __wrap_drmIoctl(int fd, unsigned long request, void *arg)
{
   CHECK(fd == 101);
   if (request == DRM_IOCTL_ASAHI_GET_PARAMS) {
      struct drm_asahi_get_params *get = arg;
      struct drm_asahi_params_global *p = (void *)(uintptr_t)get->pointer;
      CHECK(get->size == sizeof(*p));
      CHECK(p == &opening_device->params && opening_device->chip == 0);
      *p = (struct drm_asahi_params_global){
         .gpu_generation = generation, .gpu_variant = generation == 17 ? 'P' : 'G',
         .vm_start = 0, .vm_end = 1ull << 42, .vm_kernel_min_size = 1ull << 26,
         .command_timestamp_frequency_hz = 24000000,
      };
      return 0;
   }
   CHECK(request == DRM_IOCTL_ASAHI_VM_CREATE);
   CHECK(probes == 1);
   ((struct drm_asahi_vm_create *)arg)->vm_id = 9;
   vm_creates++;
   return 0;
}
struct agx_native_mesh_support __wrap_agx_native_mesh_probe(int fd, const char *classes);
struct agx_native_mesh_support __wrap_agx_native_mesh_probe(int fd, const char *classes)
{
   CHECK(fd == 101 && probes < 2 && vm_creates == probes);
   CHECK(opening_device->chip ==
         (generation == 17 ? AGX_CHIP_G17P : AGX_CHIP_G16G));
   const char *text = probes++ ? after_text : before_text;
   return agx_native_mesh_parse(text, strlen(text), classes);
}
struct agx_bo *__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                                   enum agx_bo_flags flags, const char *label);
struct agx_bo *__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                                   enum agx_bo_flags flags, const char *label)
{
   (void)size; (void)align; (void)flags; (void)label;
   CHECK(probes == 2 && vm_creates == 1 && dev->vm_id == 9);
   longjmp(stopped, 1);
}

static uint64_t
check(unsigned gen, const char *before, const char *after, bool qualified, bool kill)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(pdev);
   pdev->dev.fd = 101;
   pdev->dev.native_mesh_classes = hk_apple9_native_index_class_hash();
   pdev->disable_native_index = kill;
   opening_device = &pdev->dev;
   generation = gen; probes = vm_creates = 0;
   before_text = before; after_text = after;
   if (!setjmp(stopped)) {
      agx_open_device(NULL, &pdev->dev);
      CHECK(!"device open returned before allocation boundary");
   }
   CHECK(pdev->dev.native_mesh_qualified == qualified);
   CHECK(hk_apple9_native_index_enabled(pdev) == (qualified && !kill));
   uint64_t flags = hk_physical_device_compiler_flags(pdev);
   CHECK(!pdev->dev.bo_cache.started);
   agx_bo_cache_finish(&pdev->dev);
   agxdecode_destroy_context(pdev->dev.agxdecode);
   util_sparse_array_finish(&pdev->dev.bo_map);
   pthread_mutex_destroy(&pdev->dev.bo_map_lock);
   simple_mtx_destroy(&pdev->dev.vma_lock);
   simple_mtx_destroy(&pdev->dev.apple9_entry_lock);
   util_vma_heap_finish(&pdev->dev.main_heap);
   util_vma_heap_finish(&pdev->dev.replay_heap);
   util_vma_heap_finish(&pdev->dev.usc_heap);
   util_vma_heap_finish(&pdev->dev.replay_usc_heap);
   util_vma_heap_finish(&pdev->dev.context_heap);
   util_vma_heap_finish(&pdev->dev.apple9_entry_heap);
   glsl_type_singleton_decref();
   free(pdev);
   opening_device = NULL;
   return flags;
}

int main(void)
{
   char yes[180], changed[180], wrong[180];
   const char *hash = hk_apple9_native_index_class_hash();
   snprintf(yes, sizeof(yes), "asahi-native-mesh v1 generation=16 profile=qualified classes=%s\n", hash);
   snprintf(changed, sizeof(changed), "asahi-native-mesh v1 generation=17 profile=qualified classes=%s\n", hash);
   strcpy(wrong, yes); wrong[strlen(wrong)-2] ^= 1;
   const char *none = "asahi-native-mesh v1 generation=15 profile=none classes=-\n";
   uint64_t positive = check(16, yes, yes, true, false);
   uint64_t negative = check(16, none, none, false, false);
   CHECK((positive ^ negative) == (1ull << 60));
   CHECK(check(16, yes, yes, true, true) == negative);
   CHECK(check(16, yes, changed, false, false) == negative);
   CHECK(check(16, yes, wrong, false, false) == negative);
   CHECK(check(16, none, yes, false, false) == negative);
   check(17, yes, yes, false, false);
   puts("PASS: real device-open ordering, seven profile/policy cases, compiler bit 60; GPU execution=0");
   return 0;
}
