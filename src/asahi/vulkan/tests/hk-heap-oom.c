/* SPDX-License-Identifier: MIT */
/* Fault injection at the real BO allocation boundary. Error logging is stubbed
 * because this host test has no Vulkan instance or debug callbacks.
 */
#include <unistd.h>
#include "../hk_cmd_draw.c"
#include <sys/wait.h>

static unsigned failures, allocations;
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
   allocations++;
   /* Without geometry or tessellation bound, the heap starts small. */
   CHECK(size == HK_HEAP_MIN_B && align == 0);
   CHECK(flags == 0);
   CHECK(strcmp(label, "Geometry heap") == 0);
   return NULL;
}

/* The heap's fixed VA range, in the USC aperture. */
static struct agx_va heap_va;
static unsigned va_allocs, va_frees;

struct agx_va *__wrap_agx_va_alloc(struct agx_device *dev, uint64_t size_B,
                                   uint64_t align_B, enum agx_va_flags flags,
                                   uint64_t fixed_va);
struct agx_va *
__wrap_agx_va_alloc(struct agx_device *dev, uint64_t size_B, uint64_t align_B,
                    enum agx_va_flags flags, uint64_t fixed_va)
{
   va_allocs++;
   CHECK(size_B == HK_HEAP_VA_MAX_B && flags == AGX_VA_USC && !fixed_va);
   heap_va = (struct agx_va){.flags = flags, .addr = 1ull << 32,
                             .size_B = size_B};
   return &heap_va;
}

void __wrap_agx_va_free(struct agx_device *dev, struct agx_va *va,
                        bool unbind);
void
__wrap_agx_va_free(struct agx_device *dev, struct agx_va *va, bool unbind)
{
   CHECK(va == &heap_va);
   va_frees++;
}

VkResult __wrap___vk_errorf(const void *obj, VkResult error, const char *file,
                            int line, const char *format, ...);
VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   CHECK(error == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return error;
}

static int
run(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   if (!dev || !cmd)
      return 1;
   struct agx_va rodata_va = {0};
   struct agx_bo rodata = {.va = &rodata_va};
   struct poly_heap heap_map = {0};
   rodata._map = &heap_map;
   dev->rodata.bo = &rodata;
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   CHECK(hk_heap(cmd) == 0);
   CHECK(allocations == 1 && !cmd->uses_heap && cmd->heap_renders == 0);
   /* The failed backing releases its VA range. */
   CHECK(va_allocs == 1 && va_frees == 1 && !dev->heap_va);
   CHECK(hk_geometry_heap_va(cmd) == NULL);
   struct agx_draw draw =
      hk_heap_indexed_indirect(cmd, 0x40000000, AGX_INDEX_SIZE_U32, false);
   CHECK(draw.index_buffer == 0 && draw.index_buffer_range_B == 0);
   CHECK(allocations == 1); /* Failed once initialization remains failed. */
   CHECK(vk_command_buffer_end(&cmd->vk) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   CHECK(cmd->vk.state == MESA_VK_COMMAND_BUFFER_STATE_INVALID);
   puts("primary heap allocation failure: VK_ERROR_OUT_OF_DEVICE_MEMORY");
   free(cmd);
   free(dev);
   return failures != 0;
}

int
main(void)
{
   return run();
}
