/* SPDX-License-Identifier: MIT */
/* Fault injection at the real BO allocation boundary. Error logging is stubbed
 * because this host test has no Vulkan instance or debug callbacks.
 */
#include <unistd.h>
#include "../hk_cmd_draw.c"
#include <sys/wait.h>

static unsigned failures, allocations;
static bool alternate;
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
   CHECK(size == 128 * 1024 * 1024 && align == 0);
   CHECK(flags == AGX_BO_LOW_VA);
   CHECK(strcmp(label, alternate ? "Geometry heap (alternate)"
                                 : "Geometry heap") == 0);
   return NULL;
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
run(bool overlap, bool alt)
{
   alternate = alt;
   setenv("HK_APPLE9_PREGFX_OVERLAP", overlap ? "1" : "0", 1);
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   if (!dev || !cmd)
      return 1;
   struct agx_va va = {.addr = 0x20000000};
   struct agx_bo primary = {.va = &va,
                            .size = 128 * 1024 * 1024,
                            .flags = AGX_BO_LOW_VA};
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   if (alt) {
      dev->heap = &primary;
      dev->heap_init_once.called = true;
      cmd->heap_renders = 1;
      cmd->heap_bo = &primary; /* Previous render's BO must not leak through. */
   }
   CHECK(hk_heap(cmd) == 0);
   CHECK(allocations == 1 && !cmd->uses_heap && !cmd->heap_desc);
   CHECK(cmd->heap_renders == (alt ? 1 : 0));
   CHECK(hk_geometry_heap_bo(cmd) == NULL);
   struct agx_draw draw =
      hk_heap_indexed_indirect(cmd, 0x40000000, AGX_INDEX_SIZE_U32, false);
   CHECK(draw.index_buffer == 0 && draw.index_buffer_range_B == 0);
   CHECK(allocations == 1); /* Failed once initialization remains failed. */
   CHECK(vk_command_buffer_end(&cmd->vk) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   CHECK(cmd->vk.state == MESA_VK_COMMAND_BUFFER_STATE_INVALID);
   printf("%s %s allocation failure: VK_ERROR_OUT_OF_DEVICE_MEMORY\n",
          overlap ? "overlap-on" : "overlap-off",
          alt ? "alternate" : "primary");
   free(cmd);
   free(dev);
   return failures != 0;
}

int
main(void)
{
   /* The option is cached per process, so each mode gets a fresh child. */
   for (unsigned i = 0; i < 3; ++i) {
      pid_t pid = fork();
      if (pid < 0)
         return 1;
      if (pid == 0)
         exit(run(i != 0, i == 2));
      int status;
      if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
          WEXITSTATUS(status) != 0)
         return 1;
   }
   return 0;
}
