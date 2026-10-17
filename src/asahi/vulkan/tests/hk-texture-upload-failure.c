/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "../hk_shader.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); failures++; } } while (0)
static unsigned failures, uploads;
static bool fail_upload = true;

uint64_t __wrap_hk_pool_upload(struct hk_cmd_buffer *cmd, const void *data,
                               uint32_t size, uint32_t alignment);
uint64_t
__wrap_hk_pool_upload(struct hk_cmd_buffer *cmd, const void *data,
                      uint32_t size, uint32_t alignment)
{
   uploads++;
   CHECK(size == AGX_APPLE9_RUNTIME_SYSVAL_SIZE && alignment == 64);
   CHECK(!memcmp(data, cmd->vk.dynamic_graphics_state.cb.blend_constants, 16));
   return fail_upload ? 0 : 0x1000040000ull;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   struct hk_shader shader = {0};
   struct agx_va va = {.addr = 0x1000000000ull};
   struct agx_bo fetch = {.va = &va};
   if (!dev || !cmd)
      return 1;
   dev->dev.shader_base = va.addr;
   dev->apple9_fetch_sampler = &fetch;
   cmd->vk.base.device = &dev->vk;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   dyn->cb.blend_constants[0] = 0.25f;
   dyn->cb.color_write_enables = 3;
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   gfx->apple9_sysvals = 0x1000010000ull;
   gfx->apple9_sample_positions = 0x44cc;
   uint64_t textures = 0, samplers = 0, sysvals = 0;
   CHECK(!hk_apple9_upload_textures(cmd, &shader, true, true, false,
                                   &textures, &samplers, &sysvals));
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS));
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES));
   CHECK(gfx->apple9_sysvals == 0x1000010000ull);
   CHECK(gfx->apple9_sysvals_positions == 0 && !gfx->apple9_sysvals_first_fan);
   fail_upload = false;
   CHECK(hk_apple9_upload_textures(cmd, &shader, true, true, false,
                                  &textures, &samplers, &sysvals));
   CHECK(sysvals == 0x1000040000ull && gfx->apple9_sysvals == sysvals);
   CHECK(!BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS));
   CHECK(!BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES));
   CHECK(gfx->apple9_sysvals_positions == 0x44cc && gfx->apple9_sysvals_first_fan);
   CHECK(hk_apple9_upload_textures(cmd, &shader, true, true, false,
                                  &textures, &samplers, &sysvals));
   CHECK(uploads == 2);
   puts("failed upload preserves dirty bits/cache; successful retry publishes once");
   free(cmd);
   free(dev);
   return failures != 0;
}
