/* SPDX-License-Identifier: MIT
 * Exercise real upload/reset/pool code with host-backed BOs. */
#include "../hk_cmd_buffer.h"
#include "../hk_entrypoints.h"
#include "../hk_shader.h"
#include "../hk_descriptor_set.h"
#include "../hk_descriptor_set_layout.h"
#include "compiler/nir/nir_builder.h"
#include "vk_common_entrypoints.h"
#include "vk_alloc.h"
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
static unsigned live_bos, allocations, trims;
static bool fail_alloc;
static struct agx_bo *bos[32];
VkResult __wrap___vk_errorf(const void *, VkResult, const char *, int,
                            const char *, ...);
VkResult
__wrap___vk_errorf(const void *obj, VkResult error, const char *file, int line,
                   const char *format, ...)
{
   CHECK(fail_alloc);
   return error;
}
struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned, enum agx_bo_flags, const char *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);
void __wrap_agx_bo_cache_evict_all(struct agx_device *);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned alignment,
                     enum agx_bo_flags flags, const char *label)
{
   if (fail_alloc) return NULL;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, size);
   CHECK(bo->va && bo->_map);
   bo->dev = dev; bo->size = size; bo->refcnt = 1;
   bo->va->addr = (++allocations) * (uint64_t)HK_CMD_BO_SIZE;
   CHECK(allocations < ARRAY_SIZE(bos));
   bos[allocations] = bo;
   memset(bo->_map, 0xa5, size);
   ++live_bos;
   return bo;
}
void __wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo) return;
   CHECK(bo->refcnt == 1);
   bos[bo->va->addr / (HK_CMD_BO_SIZE)] = NULL;
   free(bo->_map); free(bo->va); free(bo); --live_bos;
}
void __wrap_agx_bo_cache_evict_all(struct agx_device *dev) { ++trims; }

static struct hk_root_descriptor_table *
mapped(uint64_t gpu)
{
   struct agx_bo *bo = bos[gpu / (HK_CMD_BO_SIZE)];
   CHECK(bo);
   return (void *)((uint8_t *)bo->_map + gpu % (HK_CMD_BO_SIZE));
}

static void
readers(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                 &agx_nir_options, "push mask");
   CHECK(hk_nir_push_constant_mask(b.shader) == 0);
   nir_load_push_constant(&b, 2, 32, nir_imm_int(&b, 8), .base = 4, .range = 256);
   CHECK(hk_nir_push_constant_mask(b.shader) == BITFIELD64_RANGE(3, 2));
   nir_load_push_constant(&b, 1, 16, nir_imm_int(&b, 254), .range = 256);
   CHECK(hk_nir_push_constant_mask(b.shader) ==
         (BITFIELD64_RANGE(3, 2) | BITFIELD64_BIT(63)));
   nir_load_push_constant(&b, 1, 32, nir_load_vertex_id(&b), .range = 256);
   CHECK(hk_nir_push_constant_mask(b.shader) == UINT64_MAX);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &agx_nir_options,
                                      "lowered push mask");
   nir_def *root = nir_load_root_agx(&b);
   nir_load_global_constant(&b, 4, 32,
      nir_iadd_imm(&b, root, hk_root_descriptor_offset(push) + 12), .align_mul = 4);
   CHECK(hk_nir_push_constant_mask(b.shader) == BITFIELD64_RANGE(3, 4));
   nir_def *set = nir_load_global_constant(&b, 1, 64,
      nir_iadd_imm(&b, root, hk_root_descriptor_offset(sets)), .align_mul = 8);
   nir_load_global_constant(&b, 4, 32,
      nir_iadd_imm(&b, set, hk_root_descriptor_offset(push) + 100), .align_mul = 4);
   CHECK(hk_nir_push_constant_mask(b.shader) == BITFIELD64_RANGE(3, 4));
   nir_load_global_constant_offset(&b, 1, 32, root,
      nir_imm_int(&b, hk_root_descriptor_offset(push) + 252), .align_mul = 4);
   CHECK(hk_nir_push_constant_mask(b.shader) ==
         (BITFIELD64_RANGE(3, 4) | BITFIELD64_BIT(63)));
   nir_load_global_constant(&b, 1, 32,
      nir_iadd(&b, root, nir_u2u64(&b, nir_load_vertex_id(&b))), .align_mul = 4);
   CHECK(hk_nir_push_constant_mask(b.shader) == UINT64_MAX);
   ralloc_free(b.shader);
}

#ifndef NDEBUG
static void
check_final_reader(void)
{
   const unsigned push = offsetof(struct hk_root_descriptor_table, push);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "final root reader");
   nir_def *root = nir_load_ubo(&b, 1, 64, nir_imm_int(&b, 0),
      nir_imm_int(&b, offsetof(struct hk_root_descriptor_table, root_desc_addr)),
      .align_mul = 8, .range = UINT32_MAX);
   nir_load_global_constant(&b, 4, 32, nir_iadd_imm(&b, root, push + 12),
                            .align_mul = 4);
   CHECK(hk_nir_push_constant_mask(b.shader) == 0);
   CHECK(hk_nir_final_push_constant_mask(b.shader) == BITFIELD64_RANGE(3, 4));
   /* The check works on a clone, preserving the actual backend input. */
   CHECK(hk_nir_push_constant_mask(b.shader) == 0);
   nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, push + 252),
                .align_mul = 4, .range = UINT32_MAX);
   CHECK(hk_nir_final_push_constant_mask(b.shader) ==
         (BITFIELD64_RANGE(3, 4) | BITFIELD64_BIT(63)));
   nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_load_vertex_id(&b),
                .align_mul = 4, .range = UINT32_MAX);
   CHECK(hk_nir_final_push_constant_mask(b.shader) == UINT64_MAX);
   ralloc_free(b.shader);

   b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &agx_nir_options,
                                      "final loaded descriptor boundary");
   nir_def *set = nir_load_ubo(&b, 1, 64, nir_imm_int(&b, 0),
      nir_imm_int(&b, offsetof(struct hk_root_descriptor_table, sets)),
      .align_mul = 8, .range = UINT32_MAX);
   nir_load_global_constant(&b, 4, 32, nir_iadd_imm(&b, set, push), .align_mul = 4);
   nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 2), nir_imm_int(&b, push),
                .align_mul = 4, .range = UINT32_MAX);
   CHECK(hk_nir_final_push_constant_mask(b.shader) == 0);
   nir_push_if(&b, nir_ine_imm(&b, nir_load_vertex_id(&b), 0));
   nir_def *left = nir_imm_int64(&b, 0x10000);
   nir_push_else(&b, NULL);
   nir_def *right = nir_imm_int64(&b, 0x20000);
   nir_pop_if(&b, NULL);
   nir_def *address = nir_if_phi(&b, left, right);
   nir_load_global_constant(&b, 1, 32, address, .align_mul = 4);
   CHECK(hk_nir_push_constant_mask(b.shader) == UINT64_MAX);
   CHECK(hk_nir_final_push_constant_mask(b.shader) == 0);
   nir_load_ubo(&b, 1, 32, nir_load_vertex_id(&b), nir_imm_int(&b, push),
                .align_mul = 4, .range = UINT32_MAX);
   CHECK(hk_nir_final_push_constant_mask(b.shader) == UINT64_MAX);
   ralloc_free(b.shader);
}
#endif

int main(void)
{
#ifndef NDEBUG
   check_final_reader();
#endif
   readers();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->vk.dispatch_table.AllocateCommandBuffers = vk_common_AllocateCommandBuffers;
   dev->vk.dispatch_table.ResetCommandBuffer = hk_ResetCommandBuffer;
   dev->dev.chip = AGX_CHIP_G16G;
   VkDevice device = hk_device_to_handle(dev);
   const VkCommandPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool handle;
   CHECK(hk_CreateCommandPool(device, &ci, NULL, &handle) == VK_SUCCESS);
   VkCommandBuffer cb;
   const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = handle, .commandBufferCount = 1,
   };
   CHECK(vk_common_AllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS);
   struct hk_cmd_buffer *cmd = hk_cmd_buffer_from_handle(cb);
   struct hk_api_shader vs = {0}, fs = {0}, cs = {0};
   cmd->state.gfx.shaders[MESA_SHADER_VERTEX] = &vs;
   cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = &fs;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   memset(desc->root.push, 0x17, sizeof(desc->root.push));
   desc->root.draw.draw_id = 99;
   uint64_t a = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(a && mapped(a)->root_desc_addr == a);
   CHECK(desc->root_last_size == hk_root_descriptor_offset(push));
   CHECK(mapped(a)->draw.draw_id == 99 && mapped(a)->push[0] == 0xa5);
   desc->root.push[0] = 88;
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == a);
   vs.push_constant_mask = BITFIELD64_RANGE(2, 2);
   fs.push_constant_mask = BITFIELD64_BIT(63);
   uint64_t b = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(b != a && desc->root_last_size == hk_root_descriptor_offset(push) + 256);
   CHECK(mapped(b)->push[0] == 0xa5 && mapped(b)->push[8] == 0x17 &&
         mapped(b)->push[252] == 0x17);
   /* A bind can expose a hole without changing the allocated extent. */
   vs.push_constant_mask |= BITFIELD64_BIT(0);
   uint64_t c = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(c != b && mapped(c)->push[0] == 88 && mapped(b)->push[0] == 0xa5);
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == c);
   desc->root.push[4] = 66;
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == c);
   desc->root.push[8] = 42;
   uint64_t d = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(d != c && mapped(d)->push[8] == 42 && mapped(c)->push[8] == 0x17);
   /* Later fields preserve their ABI offsets, with the unread hole untouched. */
   desc->root_dynamic_count = 1;
   desc->root.set_dynamic_buffer_start[0] = 7;
   desc->root.dynamic_buffers[0].base_addr = 0x12345678;
   vs.push_constant_mask = fs.push_constant_mask = 0;
   uint64_t e = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(mapped(e)->set_dynamic_buffer_start[0] == 7);
   CHECK(mapped(e)->dynamic_buffers[0].base_addr == 0x12345678);
   CHECK(mapped(e)->push[0] == 0xa5 && mapped(e)->push[252] == 0xa5);
   desc->root_constant_stages = BITFIELD_BIT(MESA_SHADER_VERTEX);
   desc->root.constant_data[MESA_SHADER_VERTEX] = 0x9876543210;
   e = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(mapped(e)->constant_data[MESA_SHADER_VERTEX] == 0x9876543210);
   uint64_t old_mask = desc->root_last_push_mask;
   uint32_t old_size = desc->root_last_size;
   /* Exhaust chunks and fail the next BO, then retry without publishing a key. */
   for (unsigned i = 0; i < HK_UPLOAD_CHUNKS; ++i)
      cmd->uploader[HK_CMD_POOL_MAIN].chunks[i].offset = HK_CMD_BO_SIZE;
   vs.push_constant_mask = UINT64_MAX;
   fail_alloc = true;
   CHECK(!hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS));
   CHECK(desc->root_last_gpu == e && desc->root_last_push_mask == old_mask &&
         desc->root_last_size == old_size);
   fail_alloc = false;
   uint64_t f = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(f && f != e && !memcmp(mapped(f)->push, desc->root.push, HK_MAX_PUSH_SIZE));
   /* Compute uses its own stage/read mask and root cache. */
   cmd->state.cs.shader = &cs;
   cs.push_constant_mask = BITFIELD64_BIT(4);
   memset(cmd->state.cs.descriptors.root.push, 0x31, HK_MAX_PUSH_SIZE);
   uint64_t g = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   CHECK(mapped(g)->push[0] == 0xa5 && mapped(g)->push[16] == 0x31);
   CHECK(cmd->state.cs.descriptors.root_last_size == hk_root_descriptor_offset(push) + 20);
   /* Push descriptors retain only the current layout's extent. */
   struct hk_push_descriptor_set *push = calloc(1, sizeof(*push));
   CHECK(push);
   desc->push[0] = push;
   struct hk_descriptor_set_layout layout = {
      .non_variable_descriptor_buffer_size = 80};
   memset(push->data, 0x37, sizeof(push->data));
   hk_push_descriptor_set_update(push, &layout, 0, NULL);
   CHECK(push->size == 80);
   desc->push_dirty = 1;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   uint64_t set_address = desc->root.sets[0];
   CHECK(set_address && desc->set_sizes[0] == 80 && !desc->push_dirty);
   CHECK(!memcmp(mapped(set_address), push->data, 80));
   CHECK(((uint8_t *)mapped(set_address))[80] == 0xa5);
   desc->root_dirty = false;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   CHECK(!desc->root_dirty && desc->root.sets[0] == set_address);
   for (unsigned i = 0; i < HK_UPLOAD_CHUNKS; ++i)
      cmd->uploader[HK_CMD_POOL_USC].chunks[i].offset = HK_CMD_BO_SIZE;
   layout.non_variable_descriptor_buffer_size = sizeof(push->data);
   hk_push_descriptor_set_update(push, &layout, 0, NULL);
   desc->push_dirty = 1;
   fail_alloc = true;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   CHECK(desc->push_dirty == 1 && desc->root.sets[0] == set_address &&
         desc->set_sizes[0] == 80 && !desc->root_dirty);
   fail_alloc = false;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   CHECK(!desc->push_dirty && desc->set_sizes[0] == sizeof(push->data));
   CHECK(desc->root.sets[0] != set_address);
   CHECK(!memcmp(mapped(desc->root.sets[0]), push->data, sizeof(push->data)));
   layout.non_variable_descriptor_buffer_size = 0;
   hk_push_descriptor_set_update(push, &layout, 0, NULL);
   desc->push_dirty = 1;
   hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   CHECK(!desc->push_dirty && !desc->set_sizes[0] && !desc->root.sets[0]);
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE;
   CHECK(hk_ResetCommandBuffer(cb, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT) == VK_SUCCESS);
   CHECK(!desc->root_last_valid && !live_bos);
   hk_DestroyCommandPool(device, handle, NULL);
   free(dev);
   puts("PASS push-reader masks, sparse copies, stage unions, exposed holes, immutable roots, tails, failure/retry and reset");
}
