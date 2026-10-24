/* SPDX-License-Identifier: MIT */
/* A folded tessellation batch whose draw table cannot be allocated must fail
 * the command buffer: its draws' indirect draws would read counts that no
 * pass writes. Pool allocation fails at the real allocation boundary.
 */
#include "../hk_cmd_buffer.h"

static unsigned failures, allocations, errors;
#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c);               \
         failures++;                                                           \
      }                                                                        \
   } while (0)

struct agx_ptr __wrap_hk_pool_alloc_internal(struct hk_cmd_buffer *cmd,
                                             uint32_t size, uint32_t alignment,
                                             enum hk_cmd_pool_kind pool);
struct agx_ptr
__wrap_hk_pool_alloc_internal(struct hk_cmd_buffer *cmd, uint32_t size,
                              uint32_t alignment, enum hk_cmd_pool_kind pool)
{
   allocations++;
   CHECK(size == 2 * 16 && pool == HK_CMD_POOL_MAIN);
   return (struct agx_ptr){0};
}

VkResult __wrap___vk_errorf(const void *obj, VkResult error, const char *file,
                            int line, const char *format, ...);
VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   errors++;
   CHECK(error == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return error;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   if (!dev || !cmd)
      return 1;
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   for (unsigned i = 0; i < HK_TESS_BATCH_LANES; ++i)
      cmd->tess_batch.lane[i] = UTIL_DYNARRAY_INIT;
   cmd->tess_batch.fold = UTIL_DYNARRAY_INIT;

   /* Two folded draws; their VS and TCS records are not needed here. */
   struct hk_cs pre_gfx = {.cmd = cmd, .type = HK_CS_CDM};
   cmd->current_cs.pre_gfx = &pre_gfx;
   for (unsigned i = 0; i < 2; ++i) {
      struct hk_tess_fold f = {
         .params = 0x1000 + i * 0x100,
         .nr_patches = 3,
         .prim = TESS_PRIMITIVE_TRIANGLES,
      };
      util_dynarray_append(&cmd->tess_batch.fold, f);
   }
   cmd->tess_batch.draws = 2;

   hk_tess_batch_flush(cmd);
   CHECK(allocations == 1);
   CHECK(errors == 1);
   CHECK(vk_command_buffer_has_error(&cmd->vk));
   CHECK(cmd->tess_batch.draws == 0 && cmd->tess_batch.fold.size == 0);
   for (unsigned i = 0; i < HK_TESS_BATCH_LANES; ++i)
      CHECK(cmd->tess_batch.lane[i].size == 0);
   /* Nothing reached the stream. */
   CHECK(pre_gfx.current == NULL && pre_gfx.stats.cmds == 0);

   if (!failures)
      printf("hk-tess-fold-oom: PASS\n");
   return failures != 0;
}
