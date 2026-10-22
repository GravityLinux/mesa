/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_buffer.h"
#include <inttypes.h>
#include "util/simple_mtx.h"
#include "asahi/lib/agx_apple9.h"

#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_linker.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "hk_buffer.h"
#include "hk_cmd_pool.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

#include "libagx_dgc.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_pipeline_layout.h"
#include "vk_synchronization.h"

#include "util/list.h"
#include "util/hash_table.h"
#include "util/u_debug.h"
#include "util/ralloc.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

static void
hk_descriptor_state_fini(struct hk_cmd_buffer *cmd,
                         struct hk_descriptor_state *desc)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);

   for (unsigned i = 0; i < HK_MAX_SETS; i++) {
      vk_free(&pool->vk.alloc, desc->push[i]);
      desc->push[i] = NULL;
   }
}

static void
hk_free_resettable_cmd_buffer(struct hk_cmd_buffer *cmd, bool release)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);
   struct hk_device *dev = hk_cmd_pool_device(pool);

   ralloc_free(cmd->upload_cache);
   cmd->upload_cache = NULL;
   memset(cmd->upload_hits, 0, sizeof(cmd->upload_hits));
   memset(cmd->upload_misses, 0, sizeof(cmd->upload_misses));
   for (unsigned i = 0; i < HK_UPLOAD_LAST_COUNT; ++i)
      cmd->upload_last[i].gpu = 0;
   cmd->apple9_prev_cs = NULL;
   cmd->apple9_linked_id[0] = cmd->apple9_linked_id[1] = 0;
   cmd->apple9_null_fs_vertex_id = 0;
   hk_descriptor_state_fini(cmd, &cmd->state.gfx.descriptors);
   hk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);

   for (unsigned i = 0; i < HK_CMD_POOL_COUNT; ++i)
      hk_cmd_pool_free_bo_list(pool, i, &cmd->uploader[i].bos, release);

   list_for_each_entry_safe(struct hk_cs, it, &cmd->control_streams, node) {
      list_del(&it->node);
      hk_cs_destroy(it);
   }

   util_dynarray_foreach(&cmd->large_bos, struct agx_bo *, bo) {
      agx_bo_unreference(&dev->dev, *bo);
   }

   if (cmd->apple9_pinned) {
      simple_mtx_lock(&dev->dev.apple9_entry_lock);
      set_foreach(cmd->apple9_pinned, entry)
         ((struct agx_bo *)entry->key)->apple9_entry_pins--;
      simple_mtx_unlock(&dev->dev.apple9_entry_lock);
      set_foreach(cmd->apple9_pinned, entry)
         agx_bo_unreference(&dev->dev, (struct agx_bo *)entry->key);
      _mesa_set_clear(cmd->apple9_pinned, NULL);
   }
   cmd->apple9_last_pin[0] = cmd->apple9_last_pin[1] = NULL;

   util_dynarray_clear(&cmd->large_bos);
   util_dynarray_clear(&cmd->implicit_sync_res);
   util_dynarray_clear(&cmd->query_adds);
   util_dynarray_clear(&cmd->query_writes);
   util_dynarray_clear(&cmd->occlusion_accum);
   util_dynarray_clear(&cmd->sync_events);
   util_dynarray_clear(&cmd->sync_remap);
}

/* agx_device::apple9_pin_entry. Every Apple9 entry is requested while
 * recording a command buffer, which passes itself as the allocator. */
bool
hk_apple9_pin_entry(void *allocator, struct agx_bo *bo)
{
   struct hk_cmd_buffer *cmd = allocator;
   if (cmd->apple9_last_pin[0] == bo || cmd->apple9_last_pin[1] == bo)
      return true;

   if (!cmd->apple9_pinned) {
      cmd->apple9_pinned = _mesa_pointer_set_create(NULL);
      if (!cmd->apple9_pinned)
         return false;
   }
   bool found;
   if (!_mesa_set_search_or_add(cmd->apple9_pinned, bo, &found))
      return false;
   if (!found) {
      agx_bo_reference(bo);
      bo->apple9_entry_pins++;
   }
   cmd->apple9_last_pin[1] = cmd->apple9_last_pin[0];
   cmd->apple9_last_pin[0] = bo;
   return true;
}

static void
hk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct hk_cmd_buffer, vk);
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);

   hk_free_resettable_cmd_buffer(cmd, false);
   _mesa_set_destroy(cmd->apple9_pinned, NULL);
   util_dynarray_fini(&cmd->large_bos);
   util_dynarray_fini(&cmd->implicit_sync_res);
   util_dynarray_fini(&cmd->query_adds);
   util_dynarray_fini(&cmd->query_writes);
   util_dynarray_fini(&cmd->occlusion_accum);
   util_dynarray_fini(&cmd->sync_events);
   for (unsigned i = 0; i < HK_TESS_BATCH_LANES; ++i)
      util_dynarray_fini(&cmd->tess_batch.lane[i]);
   util_dynarray_fini(&cmd->sync_remap);
   free(cmd->merge_empty_render);
   vk_command_buffer_finish(&cmd->vk);
   vk_free(&pool->vk.alloc, cmd);
}

static VkResult
hk_create_cmd_buffer(struct vk_command_pool *vk_pool,
                     VkCommandBufferLevel level,
                     struct vk_command_buffer **cmd_buffer_out)
{
   struct hk_cmd_pool *pool = container_of(vk_pool, struct hk_cmd_pool, vk);
   struct hk_device *dev = hk_cmd_pool_device(pool);
   struct hk_cmd_buffer *cmd;
   VkResult result;

   cmd = vk_zalloc(&pool->vk.alloc, sizeof(*cmd), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (cmd == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = vk_command_buffer_init_with_params(&cmd->vk,
                                               &(struct vk_command_buffer_init_params) {
                                                  .pool = &pool->vk,
                                                  .ops = &hk_cmd_buffer_ops,
                                                  .level = level,
                                                  .needs_cmd_queue = level == VK_COMMAND_BUFFER_LEVEL_SECONDARY,
                                               });
   if (result != VK_SUCCESS) {
      vk_free(&pool->vk.alloc, cmd);
      return result;
   }

   cmd->large_bos = UTIL_DYNARRAY_INIT;
   cmd->implicit_sync_res = UTIL_DYNARRAY_INIT;
   cmd->query_adds = UTIL_DYNARRAY_INIT;
   cmd->query_writes = UTIL_DYNARRAY_INIT;
   cmd->occlusion_accum = UTIL_DYNARRAY_INIT;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   for (unsigned i = 0; i < HK_TESS_BATCH_LANES; ++i)
      cmd->tess_batch.lane[i] = UTIL_DYNARRAY_INIT;

   cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations =
      &cmd->state.gfx._dynamic_sl;

   for (unsigned i = 0; i < HK_CMD_POOL_COUNT; ++i)
      list_inithead(&cmd->uploader[i].bos);
   list_inithead(&cmd->control_streams);

   *cmd_buffer_out = &cmd->vk;

   return VK_SUCCESS;
}

static void
hk_reset_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer,
                    VkCommandBufferResetFlags flags)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct hk_cmd_buffer, vk);

   vk_command_buffer_reset(&cmd->vk);
   hk_free_resettable_cmd_buffer(
      cmd, flags & VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

   for (unsigned i = 0; i < HK_CMD_POOL_COUNT; ++i) {
      memset(cmd->uploader[i].chunks, 0, sizeof(cmd->uploader[i].chunks));
   }

   cmd->current_cs.gfx = NULL;
   cmd->merge_cs = NULL;
   cmd->merge_pre_cs = NULL;
   cmd->merge_reach = 0;
   cmd->merge_move_stage = 0;
   cmd->current_cs.cs = NULL;
   cmd->current_cs.post_gfx = NULL;
   cmd->current_cs.pre_gfx = NULL;

   for (unsigned i = 0; i < HK_TESS_BATCH_LANES; ++i)
      util_dynarray_clear(&cmd->tess_batch.lane[i]);
   cmd->tess_batch.capture = NULL;
   cmd->tess_batch.draws = 0;

   assert(!cmd->in_meta);
   cmd->geom_index_buffer = 0;
   cmd->geom_index_count = 0;
   cmd->geom_instance_count = 0;
   cmd->uses_heap = false;
   cmd->heap_renders = 0;
   cmd->uses_mesh_heap = false;
   cmd->sync_fallback = false;
   cmd->sync_next_group = 0;
   cmd->sync_scope = 0;
   cmd->sync_render_stages = 0;
   cmd->sync_render_continuation = false;
   cmd->api_dispatch = false;

   memset(&cmd->state, 0, sizeof(cmd->state));
}

const struct vk_command_buffer_ops hk_cmd_buffer_ops = {
   .create = hk_create_cmd_buffer,
   .reset = hk_reset_cmd_buffer,
   .destroy = hk_destroy_cmd_buffer,
};

static VkResult
hk_cmd_buffer_alloc_bo(struct hk_cmd_buffer *cmd, enum hk_cmd_pool_kind kind,
                       struct hk_cmd_bo **bo_out)
{
   VkResult result = hk_cmd_pool_alloc_bo(hk_cmd_buffer_pool(cmd), kind, bo_out);
   if (result != VK_SUCCESS)
      return result;

   list_addtail(&(*bo_out)->link, &cmd->uploader[kind].bos);

   return VK_SUCCESS;
}

struct agx_ptr
hk_pool_alloc_internal(struct hk_cmd_buffer *cmd, uint32_t size,
                       uint32_t alignment, enum hk_cmd_pool_kind kind)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_uploader *uploader = &cmd->uploader[kind];

   /* Specially handle large allocations owned by the command buffer, e.g. used
    * for statically allocated vertex output buffers with geometry shaders.
    */
   if (size > HK_CMD_BO_SIZE) {
      struct agx_bo *bo =
         agx_bo_create(&dev->dev, size, alignment,
                       hk_cmd_pool_bo_flags(dev, kind), "Large pool allocation");
      if (!bo) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return (struct agx_ptr){0};
      }

      util_dynarray_append(&cmd->large_bos, bo);
      return (struct agx_ptr){
         .gpu = bo->va->addr,
         .cpu = agx_bo_map(bo),
      };
   }

   assert(size <= HK_CMD_BO_SIZE);
   assert(alignment > 0);

   struct hk_upload_chunk *chunk =
      hk_upload_find_chunk(uploader->chunks, HK_CMD_BO_SIZE, size, alignment);
   if (chunk) {
      uint32_t offset = align(chunk->offset, alignment);
      chunk->offset = offset + size;

      return (struct agx_ptr){
         .gpu = chunk->base + offset,
         .cpu = chunk->map + offset,
      };
   }

   struct hk_cmd_bo *bo;
   VkResult result = hk_cmd_buffer_alloc_bo(cmd, kind, &bo);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return (struct agx_ptr){0};
   }

   hk_upload_keep_chunk(uploader->chunks, (struct hk_upload_chunk){
      .map = bo->map,
      .base = bo->bo->va->addr,
      .offset = size,
   });

   return (struct agx_ptr){
      .gpu = bo->bo->va->addr,
      .cpu = bo->map,
   };
}

/* The key owns an ordinary CPU copy, avoiding reads from uncached uploads.
 * Type and pool are part of the key: canonical roots and launches have address
 * fixups, whereas raw uploads preserve the key bytes verbatim.
 */
struct hk_upload_key {
   uint32_t size, alignment, kind, type;
   const void *data;
};

struct hk_upload_entry {
   struct hk_upload_key key;
   struct agx_ptr ptr;
};

/* HK_STATE_CACHE=0 disables reuse. Comma-separated names enable individual
 * kinds for correctness diagnosis; all kinds are enabled by default.
 */
static const struct debug_named_value hk_state_cache_options[] = {
   {"all", ~0u, "All immutable state kinds"},
   {"launch", BITFIELD_BIT(AGX_APPLE9_STATE_LAUNCH), "Graphics launch state"},
   {"cf", BITFIELD_BIT(AGX_APPLE9_STATE_COEFFICIENTS), "Coefficient bindings"},
   {"ppp", BITFIELD_BIT(AGX_APPLE9_STATE_PPP), "PPP context state"},
   {"uploads", BITFIELD_BIT(HK_UPLOAD_RUNTIME), "Runtime, texture and sampler tables"},
   {"root", BITFIELD_BIT(HK_UPLOAD_ROOT), "Root descriptor tables"},
   {"tables", BITFIELD_BIT(HK_UPLOAD_TABLE), "Draw, vertex and resource tables"},
   DEBUG_NAMED_VALUE_END
};
DEBUG_GET_ONCE_FLAGS_OPTION(hk_state_cache, "HK_STATE_CACHE",
                           hk_state_cache_options, ~0u)

static bool
hk_upload_equal(const void *a_, const void *b_)
{
   const struct hk_upload_key *a = a_, *b = b_;
   return a->size == b->size && a->alignment == b->alignment &&
          a->kind == b->kind && a->type == b->type &&
          !memcmp(a->data, b->data, a->size);
}

static struct agx_ptr
hk_pool_alloc_cached(struct hk_cmd_buffer *cmd, enum hk_cmd_pool_kind kind,
                      uint32_t type, const void *contents, uint32_t size,
                      uint32_t alignment, bool *is_new)
{
   *is_new = true;
   if (!size)
      return (struct agx_ptr){0};

   if (!(debug_get_option_hk_state_cache() & BITFIELD_BIT(type)))
      return hk_pool_alloc_internal(cmd, size, alignment, kind);

   const struct hk_upload_key key = {size, alignment, kind, type, contents};
   uint32_t meta[] = {size, alignment, kind, type};
   uint32_t hash = _mesa_hash_data_with_seed(
      contents, size, _mesa_hash_data(meta, sizeof(meta)));
   if (!cmd->upload_cache)
      cmd->upload_cache = _mesa_hash_table_create(NULL, NULL, hk_upload_equal);
   if (cmd->upload_cache) {
      struct hash_entry *entry =
         _mesa_hash_table_search_pre_hashed(cmd->upload_cache, hash, &key);
      if (entry) {
         *is_new = false;
         cmd->upload_hits[MIN2(type, HK_UPLOAD_TABLE)]++;
         return ((struct hk_upload_entry *)entry->data)->ptr;
      }
   }
   cmd->upload_misses[MIN2(type, HK_UPLOAD_TABLE)]++;

   struct agx_ptr ptr = hk_pool_alloc_internal(cmd, size, alignment, kind);
   if (!ptr.cpu || !cmd->upload_cache)
      return ptr;

   struct hk_upload_entry *entry =
      ralloc_size(cmd->upload_cache, sizeof(*entry) + size);
   if (entry) {
      entry->key = key;
      entry->key.data = entry + 1;
      memcpy(entry + 1, contents, size);
      entry->ptr = ptr;
      if (!_mesa_hash_table_insert_pre_hashed(cmd->upload_cache, hash,
                                             &entry->key, entry))
         ralloc_free(entry);
   }
   return ptr;
}

uint64_t
hk_pool_upload_cached(struct hk_cmd_buffer *cmd, const void *data,
                       uint32_t size, uint32_t alignment,
                       enum hk_upload_type type)
{
   bool is_new;
   struct agx_ptr ptr = hk_pool_alloc_cached(
      cmd, HK_CMD_POOL_MAIN, type, data, size, alignment, &is_new);
   if (!ptr.cpu)
      return 0;
   if (is_new)
      memcpy(ptr.cpu, data, size);
   return ptr.gpu;
}

DEBUG_GET_ONCE_BOOL_OPTION(hk_upload_stats, "HK_UPLOAD_STATS", false)

/* Upload a per-draw table, reusing the previous upload of the same kind when
 * the contents are unchanged. Unlike the content cache this keeps one entry
 * per kind: these tables carry the root address, so they rarely repeat
 * except consecutively, and hashing each into a growing table costs more
 * than it saves. */
uint64_t
hk_pool_upload_last(struct hk_cmd_buffer *cmd, enum hk_upload_last slot,
                    const void *data, uint32_t size, uint32_t alignment)
{
   struct hk_upload_last_entry *last = &cmd->upload_last[slot];
   assert(size <= sizeof(last->data));
   if (last->gpu && last->size == size && !memcmp(last->data, data, size))
      return last->gpu;

   struct agx_ptr ptr = hk_pool_alloc(cmd, size, alignment);
   if (!ptr.cpu)
      return 0;
   memcpy(ptr.cpu, data, size);
   memcpy(last->data, data, size);
   last->size = size;
   last->gpu = ptr.gpu;
   return ptr.gpu;
}

struct agx_ptr
hk_apple9_state_cache_alloc(void *allocator, enum agx_apple9_state_type type,
                             const void *contents, size_t size,
                             size_t alignment, bool *is_new)
{
   enum hk_cmd_pool_kind kind = type == AGX_APPLE9_STATE_PPP
                                  ? HK_CMD_POOL_CONTEXT : HK_CMD_POOL_USC;
   /* A PPP record embeds its draw's launch addresses. Launches are emitted
    * fresh unless the previous draw's is reused, so a PPP record can only
    * repeat consecutively, which the previous-draw comparison already reuses:
    * a content lookup would always miss and grow the cache by one entry per
    * draw. */
   if (type == AGX_APPLE9_STATE_PPP) {
      *is_new = true;
      return hk_pool_alloc_internal(allocator, size, alignment, kind);
   }
   return hk_pool_alloc_cached(allocator, kind, type, contents, size,
                               alignment, is_new);
}

/* Records that embed this pass's own uploads (background/EOT stages) can never
 * match an earlier entry, so they skip the content lookup and its hashing. */
struct agx_ptr
hk_apple9_state_uncached_alloc(void *allocator, enum agx_apple9_state_type type,
                                const void *contents, size_t size,
                                size_t alignment, bool *is_new)
{
   *is_new = true;
   return hk_pool_alloc_internal(allocator, size, alignment,
                                 type == AGX_APPLE9_STATE_PPP
                                    ? HK_CMD_POOL_CONTEXT
                                    : HK_CMD_POOL_USC);
}

uint64_t
hk_pool_upload(struct hk_cmd_buffer *cmd, const void *data, uint32_t size,
               uint32_t alignment)
{
   struct agx_ptr T = hk_pool_alloc(cmd, size, alignment);
   if (unlikely(T.cpu == NULL))
      return 0;

   memcpy(T.cpu, data, size);
   return T.gpu;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                      const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_reset_cmd_buffer(&cmd->vk, 0);

   perf_debug(cmd, "Begin command buffer");
   hk_cmd_buffer_begin_compute(cmd, pBeginInfo);
   hk_cmd_buffer_begin_graphics(cmd, pBeginInfo);

   return VK_SUCCESS;
}

/*
 * Merge adjacent compute control streams. Except for reading timestamps, there
 * is no reason to submit two CDM streams back-to-back in the same command
 * buffer. However, it is challenging to avoid constructing such sequences due
 * to the gymnastics required to reorder compute around graphics. Merging at
 * EndCommandBuffer is cheap O(# of control streams) and lets us get away with
 * the sloppiness.
 */
static void
merge_control_streams(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *last = NULL;

   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->type == HK_CS_CDM && last && last->type == HK_CS_CDM &&
          !last->timestamp.end.handle) {

         hk_cs_merge_cdm(last, cs);
         last->sync_stages |= cs->sync_stages;
         *util_dynarray_element(&cmd->sync_remap, uint32_t, cs->sync_id) =
            last->sync_id;
         list_del(&cs->node);
         hk_cs_destroy(cs);
      } else {
         last = cs;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   hk_cmd_render_merge_flush(cmd);
   assert(cmd->current_cs.gfx == NULL && cmd->current_cs.pre_gfx == NULL &&
          "must end rendering before ending the command buffer");

   perf_debug(cmd, "End command buffer");
   hk_flush_query_writes(cmd);
   hk_cs_end_hint(cmd, HK_CS_END_CMDBUF);
   hk_cmd_buffer_end_compute(cmd);
   hk_cs_end_hint(cmd, HK_CS_END_CMDBUF);
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (likely(!(dev->dev.debug & AGX_DBG_NOMERGE))) {
      merge_control_streams(cmd);
   }

   /* We cannot terminate CDM control streams until after merging, since merging
    * needs to append stream links late. Now that we've merged, insert all the
    * missing stream terminates.
    */
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->type == HK_CS_CDM) {
         cs->current = agx_cdm_terminate(cs->current);
      }
   }

   /* Lookup keys are only needed while recording. Keep the immutable GPU
    * uploads until reset, but do not retain duplicate CPU copies in executable
    * command buffers.
    */
   if (cmd->upload_cache && debug_get_option_hk_upload_stats()) {
      size_t bytes = 0;
      unsigned per_type[HK_UPLOAD_TABLE + 1] = {0};
      hash_table_foreach(cmd->upload_cache, entry) {
         const struct hk_upload_key *key = entry->key;
         bytes += key->size;
         if (key->type <= HK_UPLOAD_TABLE)
            per_type[key->type]++;
      }
      fprintf(stderr, "HK_UPLOAD_STATS cache entries=%u bytes=%zu "
              "launch=%u coefficients=%u ppp=%u runtime=%u root=%u table=%u\n",
              cmd->upload_cache->entries, bytes,
              per_type[AGX_APPLE9_STATE_LAUNCH],
              per_type[AGX_APPLE9_STATE_COEFFICIENTS],
              per_type[AGX_APPLE9_STATE_PPP], per_type[HK_UPLOAD_RUNTIME],
              per_type[HK_UPLOAD_ROOT], per_type[HK_UPLOAD_TABLE]);
      static const char *names[] = {"launch", "coefficients", "ppp",
                                    "runtime", "root", "table"};
      for (unsigned t = 0; t <= HK_UPLOAD_TABLE; ++t) {
         if (cmd->upload_hits[t] + cmd->upload_misses[t])
            fprintf(stderr, "HK_UPLOAD_STATS %s hits=%u misses=%u\n", names[t],
                    cmd->upload_hits[t], cmd->upload_misses[t]);
      }
   }
   ralloc_free(cmd->upload_cache);
   cmd->upload_cache = NULL;

   return vk_command_buffer_get_record_result(&cmd->vk);
}

void
hk_cmd_sync_work(struct hk_cmd_buffer *cmd, struct hk_cs *cs, unsigned stages)
{
   if (cs->type == HK_CS_VDM && !cs->sync_stages &&
       cmd->sync_render_continuation) {
      /* Resuming a split render pass loads the preceding stream's attachment
       * stores. This dependency is implicit in Vulkan, including splits caused
       * by changes to state that is immutable within a hardware render. Only
       * the fragment stage loads attachments.
       */
      struct hk_sync_event event = {
         .stream = HK_SYNC_NONE,
         .src = cmd->sync_render_stages,
         .dst = (stages & HK_SYNC_FRAGMENT) ? HK_SYNC_FRAGMENT : stages,
      };
      util_dynarray_append(&cmd->sync_events, event);
      cmd->sync_render_continuation = false;
   }
   cs->sync_stages |= stages;

   /* A stream can contain many draws. Coalesce repeated scope declarations
    * until a barrier or another stream intervenes.
    */
   if (cmd->sync_events.size) {
      struct hk_sync_event *last =
         util_dynarray_top_ptr(&cmd->sync_events, struct hk_sync_event);
      if (last->stream == cs->sync_id && last->dst == stages)
         return;
   }

   struct hk_sync_event event = {.stream = cs->sync_id, .dst = stages};
   util_dynarray_append(&cmd->sync_events, event);
}

/* HK_APPLE9_MERGE_TRACE (diagnostic): counts what ends renders kept open
 * for merging, printed to stderr every 5000 events as MERGE_TRACE lines. */
void
hk_render_merge_trace(const char *why, uint64_t a, uint64_t b)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_APPLE9_MERGE_TRACE", false);
   if (!enabled)
      return;
   static simple_mtx_t lock = SIMPLE_MTX_INITIALIZER;
   static struct { const char *why; uint64_t a, b, n; } table[64];
   static unsigned events;
   simple_mtx_lock(&lock);
   unsigned i = 0;
   while (i < ARRAY_SIZE(table) && table[i].why &&
          (table[i].why != why || table[i].a != a || table[i].b != b))
      ++i;
   if (i < ARRAY_SIZE(table)) {
      table[i].why = why;
      table[i].a = a;
      table[i].b = b;
      ++table[i].n;
   }
   if (++events % 5000 == 0) {
      for (unsigned j = 0; j < ARRAY_SIZE(table) && table[j].why; ++j)
         fprintf(stderr, "MERGE_TRACE events=%u why=%s a=0x%" PRIx64 " b=0x%" PRIx64 " n=%" PRIu64 "\n",
                 events, table[j].why, table[j].a, table[j].b, table[j].n);
   }
   simple_mtx_unlock(&lock);
}

void
hk_cmd_render_merge_flush_why(struct hk_cmd_buffer *cmd, const char *why)
{
   if (!cmd->merge_cs)
      return;
   if (why)
      hk_render_merge_trace(why, 0, 0);

   assert(cmd->current_cs.gfx == cmd->merge_cs);
   hk_cs_end_hint(cmd, HK_CS_END_RENDER_PASS);
   hk_cmd_buffer_end_graphics(cmd);
   assert(!cmd->merge_cs);
}

/* Whether a barrier between two render passes only orders attachment access,
 * which a single hardware render already orders in each tile.
 */
static bool
hk_render_merge_barrier(VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst,
                        VkAccessFlags2 src_access, VkAccessFlags2 dst_access)
{
   const VkPipelineStageFlags2 attachment =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
   const VkAccessFlags2 attachment_access =
      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

   if (!src || !dst || (src & ~attachment) ||
       (dst & ~(attachment | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)))
      return false;

   return !(src_access & ~attachment_access) &&
          !(dst_access &
            ~(attachment_access | VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT));
}

/* Place the sync events recorded from index first onwards ahead of the open
 * render's events, after earlier ones placed there. */
static void
hk_render_merge_place_events(struct hk_cmd_buffer *cmd, unsigned first)
{
   unsigned count =
      util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event);
   unsigned n = count - first;
   if (!n || first == cmd->merge_event)
      goto done;
   struct hk_sync_event *events = cmd->sync_events.data;
   struct hk_sync_event moved[8];
   assert(n <= ARRAY_SIZE(moved));
   memcpy(moved, &events[first], n * sizeof(*events));
   memmove(&events[cmd->merge_event + n], &events[cmd->merge_event],
           (first - cmd->merge_event) * sizeof(*events));
   memcpy(&events[cmd->merge_event], moved, n * sizeof(*events));
done:
   cmd->merge_event += n;
}

/* The compute stream for work moved ahead of the open render. */
struct hk_cs *
hk_render_merge_pre_cs(struct hk_cmd_buffer *cmd, unsigned stages)
{
   unsigned first =
      util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event);
   bool created = !cmd->merge_pre_cs;
   struct hk_cs *cs =
      hk_cmd_buffer_get_cs_general(cmd, &cmd->merge_pre_cs, true, stages);
   if (created && cs) {
      perf_debug(cmd, "Moving compute ahead of the open render");
      hk_render_merge_trace("moved ahead", cmd->merge_move_stage, cmd->merge_reach);
   }
   hk_render_merge_place_events(cmd, first);
   return cs;
}

/* Pipeline stages in a barrier scope, with the shorthand stages expanded:
 * TOP_OF_PIPE as a source and BOTTOM_OF_PIPE as a destination name none;
 * BOTTOM_OF_PIPE as a source and TOP_OF_PIPE as a destination mean all. */
static VkPipelineStageFlags2
hk_merge_expand_stages(VkPipelineStageFlags2 stages, bool source)
{
   const VkPipelineStageFlags2 all = ~(VkPipelineStageFlags2)0;
   if (stages & (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
                 (source ? VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
                         : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)))
      return all;
   if (stages & VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)
      stages |= hk_merge_graphics_stages();
   if (stages & VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT)
      stages |= VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT |
                VK_PIPELINE_STAGE_2_RESOLVE_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
   return stages & ~(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
                     VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
}

static bool
hk_image_overlaps(const struct hk_image *a, const struct hk_image *b)
{
   for (unsigned p = 0; p < a->plane_count; ++p) {
      for (unsigned q = 0; q < b->plane_count; ++q) {
         uint64_t a0 = a->planes[p].addr, a1 = a0 + a->planes[p].layout.size_B;
         uint64_t b0 = b->planes[q].addr, b1 = b0 + b->planes[q].layout.size_B;
         if (a0 < b1 && b0 < a1)
            return true;
      }
   }
   return false;
}

/* Whether the image barrier being recorded may leave the open render R open
 * (HK_APPLE9_MERGE_HOIST_BARRIERS, default on). Its image is none of R's
 * attachments nor shares memory with them, and R's stream has not written
 * memory from a shader, so every write to the image precedes R in recording
 * order: placing the barrier ahead of R orders those writes before R, and
 * reads after the barrier in R see them. Later draws in R that write memory
 * end it first (hk_apple9_prepare_draw). A barrier from R's shader stages
 * also orders R's reads before later writes, so it ends R instead. */
static bool
hk_render_merge_image_barrier(struct hk_cmd_buffer *cmd, VkPipelineStageFlags2 src)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_APPLE9_MERGE_HOIST_BARRIERS", true);
   const struct hk_image *image = cmd->barrier_image;
   if (!enabled || !image || cmd->barrier_qfot || cmd->in_meta ||
       cmd->merge_cs->merge_writes || image->planes[0].sparse_map)
      return false;
   /* The barrier's execution scope covers all earlier work in its source
    * stages, not only accesses to its image. When the render's attachments
    * are in layouts that allow shader access, its passes may have read them
    * as textures; a source in shader stages may order those reads before a
    * later pass's attachment writes, which other tiles of the merged render
    * could perform first. */
   const VkPipelineStageFlags2 attachment_output =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
   if (cmd->merge_feedback && (src & ~attachment_output))
      return false;
   /* A source in the render's shader stages covers its own texture reads,
    * which may include the image: a later write to it must wait for the
    * render (write-after-read), and a barrier ahead of the render orders
    * nothing of it. Piglit arb_copy_image-formats --samples=4 re-uploads a
    * texture its previous upload draw sampled. A finer check could keep R
    * open when R does not sample the image, should this show in a profile. */
   const VkPipelineStageFlags2 shaders =
      hk_merge_graphics_stages() &
      ~(attachment_output | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
        VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
        VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
        VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT);
   if (hk_merge_expand_stages(src, true) & shaders)
      return false;
   for (unsigned i = 0; i < HK_MAX_RTS + 2; ++i) {
      const struct vk_image *target = cmd->merge_key.image[i];
      if (!target)
         continue;
      const struct hk_image *t = container_of(target, struct hk_image, vk);
      if (t == image || t->planes[0].sparse_map || hk_image_overlaps(t, image))
         return false;
   }
   return true;
}

/* HK_APPLE9_MERGE_MOVE_COMPUTE=0 keeps compute and transfer work in order
 * after a render kept open for merging, and ends the render at every barrier
 * that does not involve it. */
bool
hk_render_merge_move_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("HK_APPLE9_MERGE_MOVE_COMPUTE", true);
   return enabled;
}

void
hk_cmd_sync_barrier(struct hk_cmd_buffer *cmd,
                    VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst,
                    VkAccessFlags2 src_access, VkAccessFlags2 dst_access)
{
   /* No synchronization may come between a suspended render pass and the
    * one resuming it; if any does, end the render as before. */
   if (cmd->merge_cs && cmd->merge_suspended)
      hk_cmd_render_merge_flush(cmd);
   if (cmd->merge_cs) {
      const VkPipelineStageFlags2 xsrc = hk_merge_expand_stages(src, true);
      const VkPipelineStageFlags2 xdst = hk_merge_expand_stages(dst, false);
      bool image_barrier = false, unrelated = false;
      /* Depth/stencil input attachments are not read from the tilebuffer.
       * They need the preceding render's stored image, not an in-tile order.
       */
      bool ds_input = (cmd->merge_key.image[HK_MAX_RTS] ||
                       cmd->merge_key.image[HK_MAX_RTS + 1]) &&
                      (dst_access & VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
      if (!ds_input && ((!cmd->in_meta &&
           hk_render_merge_barrier(src, dst, src_access, dst_access)) ||
          (image_barrier = hk_render_merge_image_barrier(cmd, src)) ||
          (unrelated = hk_render_merge_move_enabled() && !cmd->in_meta &&
                       !(xsrc & cmd->merge_reach)))) {
         cmd->merge_cs->merge_hoisted |= image_barrier;
         if (image_barrier)
            perf_debug(cmd, "Hoisting an image barrier ahead of the open render");
         /* The execution scope of a barrier with colour-output or test
          * sources covers the render's own passes; only when they cannot
          * read their attachments as textures does it order nothing of
          * theirs that later work could depend on. */
         if (!unrelated && (!image_barrier || cmd->merge_feedback))
            cmd->merge_reach |= xdst;
         /* Work moved ahead of the render after this barrier must follow it
          * in its own stream. */
         if (unrelated && cmd->merge_pre_cs)
            hk_cmd_buffer_end_compute_internal(cmd, &cmd->merge_pre_cs);
         /* The next render pass may continue the open render, where the tile
          * order satisfies this barrier. Earlier work it orders must still
          * precede the whole render: place it before the render's first
          * event. Later streams order after the render by its own scope.
          */
         struct hk_sync_event event = {
            .stream = HK_SYNC_NONE,
            .src = hk_sync_stage_mask(src, true),
            .dst = hk_sync_stage_mask(dst, false),
         };
         unsigned count =
            util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event);
         assert(cmd->merge_event < count);
         util_dynarray_append(&cmd->sync_events, event);
         struct hk_sync_event *events = cmd->sync_events.data;
         memmove(&events[cmd->merge_event + 1], &events[cmd->merge_event],
                 (count - cmd->merge_event) * sizeof(*events));
         events[cmd->merge_event++] = event;
         hk_render_merge_trace("barrier kept", src, dst);
         return;
      }

      hk_render_merge_trace(cmd->in_meta ? "meta barrier" : "barrier",
                            src | (uint64_t)!!src_access << 63,
                            dst | (uint64_t)!!dst_access << 63);
      hk_cmd_render_merge_flush_why(cmd, NULL);
   }

   unsigned src_stages = hk_sync_stage_mask(src, true);
   unsigned dst_stages = hk_sync_stage_mask(dst, false);
   /* vk_meta barriers name implementation stages. Its operations are tracked
    * in the API command's logical scope, so retain full internal dependencies.
    */
   if (cmd->in_meta) {
      src_stages = src_stages ? HK_SYNC_ALL : 0;
      dst_stages = dst_stages ? HK_SYNC_ALL : 0;
   }

   /* Access scopes govern availability/visibility, not execution ordering:
    * even zero-access WAR barriers must split ordered streams. Stream endings
    * currently provide full cache maintenance, covering the filtered access
    * scopes as well as implicit attachment stores.
    */
   src_access = vk_filter_src_access_flags2(src, src_access);
   dst_access = vk_filter_dst_access_flags2(dst, dst_access);

   unsigned ordered = hk_sync_ordered_stages(src_stages, dst_stages,
                                             src_access, dst_access);
   const unsigned compute = HK_SYNC_COMPUTE | HK_SYNC_DISPATCH_INDIRECT;
   struct hk_cs *cs = cmd->current_cs.cs;
   if (!cmd->sync_fallback && cs && ordered && !(ordered & ~compute) &&
       !(cs->sync_stages & ~compute)) {
      /* The cache barrier is also an execution barrier. Keep compute-only
       * dependencies in one CDM stream; no firmware command boundary is needed.
       * Shader-to-shader dependencies take the compute-only barrier on G16;
       * indirect grids are fetched by the CDM itself and need the full flush.
       */
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      hk_ensure_cs_has_space(cmd, cs, AGX_CDM_BARRIER_LENGTH);
      if (dev->dev.chip == AGX_CHIP_G16G &&
          !(dst_stages & HK_SYNC_DISPATCH_INDIRECT))
         hk_cdm_compute_barrier(dev, cs);
      else
         hk_cdm_cache_flush(dev, cs);
   } else if (cmd->sync_fallback ||
       (cmd->current_cs.cs && (cmd->current_cs.cs->sync_stages & ordered))) {
      hk_cs_end_hint(cmd, HK_CS_END_BARRIER);
      hk_cmd_buffer_end_compute(cmd);
   }

   if (cmd->sync_fallback ||
       (cmd->current_cs.gfx && (cmd->current_cs.gfx->sync_stages & ordered))) {
      hk_cs_end_hint(cmd, HK_CS_END_BARRIER);
      hk_cmd_buffer_end_graphics(cmd);
   }

   struct hk_sync_event event = {
      .stream = HK_SYNC_NONE, .src = src_stages, .dst = dst_stages,
   };
   util_dynarray_append(&cmd->sync_events, event);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                       const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (HK_PERF(dev, NOBARRIER))
      return;

   perf_debug(cmd, "Pipeline barrier");

   /* Preserve individual pairs rather than creating a cross product of all
    * source and destination scopes. Resource ranges cannot narrow execution
    * scopes without alias analysis. Layouts have no transition work on AGX.
    */
   uint32_t group = ++cmd->sync_next_group;
   assert(group != 0);
#define BARRIER(b) do {                                                        \
      hk_cmd_sync_barrier(cmd, (b).srcStageMask, (b).dstStageMask,                \
                           (b).srcAccessMask, (b).dstAccessMask);               \
      util_dynarray_top_ptr(&cmd->sync_events,                                 \
                            struct hk_sync_event)->barrier_group = group;      \
   } while (0)
   for (unsigned i = 0; i < pDependencyInfo->memoryBarrierCount; ++i)
      BARRIER(pDependencyInfo->pMemoryBarriers[i]);
   for (unsigned i = 0; i < pDependencyInfo->bufferMemoryBarrierCount; ++i)
      BARRIER(pDependencyInfo->pBufferMemoryBarriers[i]);
   for (unsigned i = 0; i < pDependencyInfo->imageMemoryBarrierCount; ++i) {
      const VkImageMemoryBarrier2 *b = &pDependencyInfo->pImageMemoryBarriers[i];
      cmd->barrier_image = hk_image_from_handle(b->image);
      cmd->barrier_qfot = b->srcQueueFamilyIndex != b->dstQueueFamilyIndex;
      BARRIER(*b);
   }
   cmd->barrier_image = NULL;
#undef BARRIER
}

void
hk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                    const mesa_shader_stage *stages,
                    struct vk_shader **const shaders)
{
   struct hk_cmd_buffer *cmd = container_of(vk_cmd, struct hk_cmd_buffer, vk);

   for (uint32_t i = 0; i < stage_count; i++) {
      struct hk_api_shader *shader =
         container_of(shaders[i], struct hk_api_shader, vk);

      if (stages[i] == MESA_SHADER_COMPUTE || stages[i] == MESA_SHADER_KERNEL) {
         hk_cmd_bind_compute_shader(cmd, shader);
      } else {
         if (stages[i] < HK_APPLE9_GFX_STAGES)
            cmd->state.gfx.apple9_api[stages[i]] = shaders[i] ? shader : NULL;
         hk_cmd_bind_graphics_shader(cmd, stages[i], shaders[i] ? shader : NULL);
      }
   }
}

/* Graphics bind points, including mesh pipelines */
#define HK_GRAPHICS_STAGES                                                     \
   (VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT |              \
    VK_SHADER_STAGE_MESH_BIT_EXT)

static void
hk_bind_descriptor_sets(UNUSED struct hk_cmd_buffer *cmd,
                        struct hk_descriptor_state *desc,
                        const VkBindDescriptorSetsInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   /* Fro the Vulkan 1.3.275 spec:
    *
    *    "When binding a descriptor set (see Descriptor Set Binding) to
    *    set number N...
    *
    *    If, additionally, the previously bound descriptor set for set
    *    N was bound using a pipeline layout not compatible for set N,
    *    then all bindings in sets numbered greater than N are
    *    disturbed."
    *
    * This means that, if some earlier set gets bound in such a way that
    * it changes set_dynamic_buffer_start[s], this binding is implicitly
    * invalidated.
    */
   uint8_t dyn_buffer_start =
      pipeline_layout->dynamic_descriptor_offset[info->firstSet];

   uint32_t next_dyn_offset = 0;
   for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      unsigned s = i + info->firstSet;
      VK_FROM_HANDLE(hk_descriptor_set, set, info->pDescriptorSets[i]);

      if (desc->sets[s] != set) {
         if (set != NULL) {
            desc->root.sets[s] = hk_descriptor_set_addr(set);
            desc->set_sizes[s] = set->size;
         } else {
            desc->root.sets[s] = 0;
            desc->set_sizes[s] = 0;
         }
         desc->sets[s] = set;
         desc->sets_dirty |= BITFIELD_BIT(s);
         desc->descriptor_buffer_sets &= ~BITFIELD_BIT(s);

         /* Binding descriptors invalidates push descriptors */
         desc->push_dirty &= ~BITFIELD_BIT(s);
      }

      desc->root.set_dynamic_buffer_start[s] = dyn_buffer_start;

      if (pipeline_layout->set_layouts[s] != NULL) {
         const struct hk_descriptor_set_layout *set_layout =
            vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[s]);

         if (set != NULL && set_layout->vk.dynamic_descriptor_count > 0) {
            for (uint32_t j = 0; j < set_layout->vk.dynamic_descriptor_count; j++) {
               struct hk_buffer_address addr = set->dynamic_buffers[j];
               addr.base_addr += info->pDynamicOffsets[next_dyn_offset + j];
               desc->root.dynamic_buffers[dyn_buffer_start + j] = addr;
            }
            next_dyn_offset += set->layout->vk.dynamic_descriptor_count;
         }

         dyn_buffer_start += set_layout->vk.dynamic_descriptor_count;
      } else {
         assert(set == NULL);
      }
   }
   assert(dyn_buffer_start <= HK_MAX_DYNAMIC_BUFFERS);
   assert(next_dyn_offset <= info->dynamicOffsetCount);
   desc->root_dynamic_count = MAX2(desc->root_dynamic_count, dyn_buffer_start);

   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindDescriptorSets2KHR(
   VkCommandBuffer commandBuffer,
   const VkBindDescriptorSetsInfoKHR *pBindDescriptorSetsInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pBindDescriptorSetsInfo->stageFlags & HK_GRAPHICS_STAGES) {
      hk_bind_descriptor_sets(cmd, &cmd->state.gfx.descriptors,
                              pBindDescriptorSetsInfo);
   }

   if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      hk_bind_descriptor_sets(cmd, &cmd->state.cs.descriptors,
                              pBindDescriptorSetsInfo);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindDescriptorBuffersEXT(VkCommandBuffer commandBuffer,
                               uint32_t bufferCount,
                               const VkDescriptorBufferBindingInfoEXT *pBindingInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   assert(bufferCount <= HK_MAX_SETS);
   for (uint32_t i = 0; i < bufferCount; ++i) {
      cmd->state.gfx.descriptors.descriptor_buffers[i] = pBindingInfos[i].address;
      cmd->state.cs.descriptors.descriptor_buffers[i] = pBindingInfos[i].address;
   }
}

/* Point set s at descriptor memory without a VkDescriptorSet object. */
static void
hk_bind_descriptor_memory(struct hk_descriptor_state *desc, unsigned s,
                          uint64_t addr)
{
   desc->root.sets[s] = addr;
   desc->set_sizes[s] = HK_MAX_DESCRIPTOR_SET_SIZE;
   desc->sets[s] = NULL;
   desc->descriptor_buffer_sets |= BITFIELD_BIT(s);
   desc->sets_dirty |= BITFIELD_BIT(s);
   /* Binding descriptors invalidates push descriptors */
   desc->push_dirty &= ~BITFIELD_BIT(s);
   desc->root_dirty = true;
}

static void
hk_set_descriptor_buffer_offsets(struct hk_descriptor_state *desc,
                                 const VkSetDescriptorBufferOffsetsInfoEXT *info)
{
   for (uint32_t i = 0; i < info->setCount; ++i) {
      hk_bind_descriptor_memory(
         desc, info->firstSet + i,
         desc->descriptor_buffers[info->pBufferIndices[i]] + info->pOffsets[i]);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetDescriptorBufferOffsets2EXT(
   VkCommandBuffer commandBuffer,
   const VkSetDescriptorBufferOffsetsInfoEXT *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (pInfo->stageFlags & HK_GRAPHICS_STAGES)
      hk_set_descriptor_buffer_offsets(&cmd->state.gfx.descriptors, pInfo);
   if (pInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      hk_set_descriptor_buffer_offsets(&cmd->state.cs.descriptors, pInfo);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindDescriptorBufferEmbeddedSamplers2EXT(
   VkCommandBuffer commandBuffer,
   const VkBindDescriptorBufferEmbeddedSamplersInfoEXT *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, pInfo->layout);
   const struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[pInfo->set]);
   assert(set_layout->embedded_samplers);
   uint64_t addr = set_layout->embedded_samplers->va->addr;
   if (pInfo->stageFlags & HK_GRAPHICS_STAGES)
      hk_bind_descriptor_memory(&cmd->state.gfx.descriptors, pInfo->set, addr);
   if (pInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      hk_bind_descriptor_memory(&cmd->state.cs.descriptors, pInfo->set, addr);
}

static void
hk_push_constants(UNUSED struct hk_cmd_buffer *cmd,
                  struct hk_descriptor_state *desc,
                  const VkPushConstantsInfoKHR *info)
{
   memcpy(desc->root.push + info->offset, info->pValues, info->size);
   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushConstants2KHR(VkCommandBuffer commandBuffer,
                        const VkPushConstantsInfoKHR *pPushConstantsInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pPushConstantsInfo->stageFlags & HK_GRAPHICS_STAGES)
      hk_push_constants(cmd, &cmd->state.gfx.descriptors, pPushConstantsInfo);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      hk_push_constants(cmd, &cmd->state.cs.descriptors, pPushConstantsInfo);
}

static struct hk_push_descriptor_set *
hk_cmd_push_descriptors(struct hk_cmd_buffer *cmd,
                        struct hk_descriptor_state *desc, uint32_t set)
{
   assert(set < HK_MAX_SETS);
   if (unlikely(desc->push[set] == NULL)) {
      desc->push[set] =
         vk_zalloc(&cmd->vk.pool->alloc, sizeof(*desc->push[set]), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (unlikely(desc->push[set] == NULL)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
   }

   /* Pushing descriptors replaces whatever sets are bound */
   desc->sets[set] = NULL;
   desc->descriptor_buffer_sets &= ~BITFIELD_BIT(set);
   desc->push_dirty |= BITFIELD_BIT(set);

   return desc->push[set];
}

static void
hk_push_descriptor_set(struct hk_cmd_buffer *cmd,
                       struct hk_descriptor_state *desc,
                       const VkPushDescriptorSetInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   struct hk_push_descriptor_set *push_set =
      hk_cmd_push_descriptors(cmd, desc, info->set);
   if (unlikely(push_set == NULL))
      return;

   struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[info->set]);

   hk_push_descriptor_set_update(push_set, set_layout,
                                 info->descriptorWriteCount,
                                 info->pDescriptorWrites);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushDescriptorSet2KHR(
   VkCommandBuffer commandBuffer,
   const VkPushDescriptorSetInfoKHR *pPushDescriptorSetInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pPushDescriptorSetInfo->stageFlags & HK_GRAPHICS_STAGES) {
      hk_push_descriptor_set(cmd, &cmd->state.gfx.descriptors,
                             pPushDescriptorSetInfo);
   }

   if (pPushDescriptorSetInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      hk_push_descriptor_set(cmd, &cmd->state.cs.descriptors,
                             pPushDescriptorSetInfo);
   }
}

void
hk_cmd_buffer_flush_push_descriptors(struct hk_cmd_buffer *cmd,
                                     struct hk_descriptor_state *desc)
{
   u_foreach_bit(set_idx, desc->push_dirty) {
      struct hk_push_descriptor_set *push_set = desc->push[set_idx];
      /* G16 shaders address image descriptors within the USC aperture. */
      uint64_t push_set_addr;
      if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G) {
         struct agx_ptr t =
            hk_pool_usc_alloc(cmd, sizeof(push_set->data), HK_MIN_UBO_ALIGNMENT);
         if (t.cpu)
            memcpy(t.cpu, push_set->data, sizeof(push_set->data));
         push_set_addr = t.gpu;
      } else {
         push_set_addr = hk_pool_upload(
            cmd, push_set->data, sizeof(push_set->data), HK_MIN_UBO_ALIGNMENT);
      }

      desc->root.sets[set_idx] = push_set_addr;
      desc->set_sizes[set_idx] = sizeof(push_set->data);
   }

   desc->root_dirty = true;
   desc->push_dirty = 0;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushDescriptorSetWithTemplate2KHR(
   VkCommandBuffer commandBuffer, const VkPushDescriptorSetWithTemplateInfoKHR
                                     *pPushDescriptorSetWithTemplateInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  pPushDescriptorSetWithTemplateInfo->descriptorUpdateTemplate);
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout,
                  pPushDescriptorSetWithTemplateInfo->layout);

   struct hk_descriptor_state *desc =
      hk_get_descriptors_state(cmd, template->bind_point);
   struct hk_push_descriptor_set *push_set = hk_cmd_push_descriptors(
      cmd, desc, pPushDescriptorSetWithTemplateInfo->set);
   if (unlikely(push_set == NULL))
      return;

   struct hk_descriptor_set_layout *set_layout = vk_to_hk_descriptor_set_layout(
      pipeline_layout->set_layouts[pPushDescriptorSetWithTemplateInfo->set]);

   hk_push_descriptor_set_update_template(
      push_set, set_layout, template,
      pPushDescriptorSetWithTemplateInfo->pData);
}

uint64_t
hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   struct hk_descriptor_state *desc = hk_get_descriptors_state(cmd, bind_point);
   struct hk_root_descriptor_table *root = &desc->root;

   bool is_new = true;
   struct agx_ptr root_ptr;
   size_t size = sizeof(*root);
   if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G) {
      /* Normalize only the self-address and compare the root against the
       * last upload for this bind point: identical roots reuse it, any
       * difference uploads a fresh copy. No hashing on this per-draw path.
       * Roots end after the dynamic buffer slots bound so far: the 64-entry
       * table is most of the root, and layouts use a few. */
      root->root_desc_addr = 0;
      size = hk_root_descriptor_offset(dynamic_buffers) +
             desc->root_dynamic_count * sizeof(struct hk_buffer_address);
      if (desc->root_last_valid && desc->root_last_size == size &&
          !memcmp(root, &desc->root_last, size)) {
         root->root_desc_addr = desc->root_last_gpu;
         return desc->root_last_gpu;
      }
      root_ptr = hk_pool_alloc(cmd, size, 8);
      if (root_ptr.gpu) {
         memcpy(&desc->root_last, root, size);
         desc->root_last_size = size;
         desc->root_last_gpu = root_ptr.gpu;
         desc->root_last_valid = true;
      }
   } else {
      root_ptr = hk_pool_alloc(cmd, size, 8);
   }
   if (!root_ptr.gpu)
      return 0;

   root->root_desc_addr = root_ptr.gpu;
   if (is_new)
      memcpy(root_ptr.cpu, root, size);
   return root_ptr.gpu;
}

void
hk_usc_upload_spilled_rt_descs(struct agx_usc_builder *b,
                               struct hk_cmd_buffer *cmd)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* Upload texture/PBE descriptors for each render target so we can clear
    * spilled render targets.
    */
   struct agx_ptr descs =
      hk_pool_alloc(cmd, AGX_TEXTURE_LENGTH * 2 * render->color_att_count, 64);
   struct agx_texture_packed *desc = descs.cpu;
   if (!desc)
      return;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      struct hk_image_view *iview = render->color_att[i].iview;
      if (!iview) {
         /* XXX: probably should emit a null descriptor here...? */
         continue;
      }

      memcpy(&desc[(i * 2) + 0], &iview->planes[0].emrt_texture, sizeof(*desc));
      memcpy(&desc[(i * 2) + 1], &iview->planes[0].emrt_pbe, sizeof(*desc));
   }

   desc = descs.cpu;

   /* Bind the base as u0_u1 for bindless access */
   agx_usc_uniform(b, 0, 4, hk_pool_upload(cmd, &descs.gpu, 8, 8));
}

void
hk_reserve_scratch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                   struct hk_shader *s)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint32_t max_scratch_size =
      MAX2(s->b.info.scratch_size, s->b.info.preamble_scratch_size);

   /* Not scratch but this is the most convenient place for this... */
   cs->uses_sampler_heap |= s->b.info.uses_sampler_heap;

   /* Apple9 private memory is described by the native launch frame extents.
    * It does not use the older helper shader and scratch-buffer protocol. */
   if (max_scratch_size == 0 || dev->dev.chip == AGX_CHIP_G16G)
      return;

   unsigned preamble_size = (s->b.info.preamble_scratch_size > 0) ? 1 : 0;

   /* Note: this uses the hardware stage, not the software stage */
   hk_device_alloc_scratch(dev, s->b.info.stage, max_scratch_size);
   perf_debug(cmd, "Reserving %u (%u) bytes of scratch for stage %s",
              s->b.info.scratch_size, s->b.info.preamble_scratch_size,
              _mesa_shader_stage_to_abbrev(s->b.info.stage));

   switch (s->b.info.stage) {
   case MESA_SHADER_FRAGMENT:
      cs->scratch.fs.main = true;
      cs->scratch.fs.preamble = MAX2(cs->scratch.fs.preamble, preamble_size);
      break;
   case MESA_SHADER_VERTEX:
      cs->scratch.vs.main = true;
      cs->scratch.vs.preamble = MAX2(cs->scratch.vs.preamble, preamble_size);
      break;
   default:
      cs->scratch.cs.main = true;
      cs->scratch.cs.preamble = MAX2(cs->scratch.cs.preamble, preamble_size);
      break;
   }
}

uint32_t
hk_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                    struct hk_linked_shader *linked)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   mesa_shader_stage sw_stage = s->info.stage;

   unsigned constant_push_ranges = DIV_ROUND_UP(s->b.info.rodata.size_16, 64);
   unsigned push_ranges = 2;
   unsigned stage_ranges = 3;

   size_t usc_size =
      agx_usc_size(constant_push_ranges + push_ranges + stage_ranges + 4);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return 0;

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   uint64_t root_ptr;

   if (sw_stage == MESA_SHADER_COMPUTE) {
      root_ptr = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   } else {
      root_ptr = cmd->state.gfx.root;
   }

   static_assert(offsetof(struct hk_root_descriptor_table, root_desc_addr) == 0,
                 "self-reflective");

   unsigned root_unif = 0;
   if (sw_stage == MESA_SHADER_VERTEX) {
      unsigned count =
         DIV_ROUND_UP(BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);

      if (count) {
         agx_usc_uniform(
            &b, AGX_ABI_VUNI_VBO_BASE(0), 4 * count,
            root_ptr + hk_root_descriptor_offset(draw.attrib_base));

         agx_usc_uniform(
            &b, AGX_ABI_VUNI_VBO_CLAMP(count, 0), 2 * count,
            root_ptr + hk_root_descriptor_offset(draw.attrib_clamps));
      }

      if (cmd->state.gfx.draw_params) {
         agx_usc_uniform(&b, AGX_ABI_VUNI_FIRST_VERTEX(count), 4,
                         cmd->state.gfx.draw_params);
      }

      if (cmd->state.gfx.draw_id_ptr) {
         agx_usc_uniform(&b, AGX_ABI_VUNI_DRAW_ID(count), 1,
                         cmd->state.gfx.draw_id_ptr);
      }

      if (linked->sw_indexing) {
         agx_usc_uniform(
            &b, AGX_ABI_VUNI_VERTEX_PARAMS(count), 4,
            root_ptr + hk_root_descriptor_offset(draw.vertex_params));
      }

      root_unif = AGX_ABI_VUNI_COUNT_VK(count);
   } else if (sw_stage == MESA_SHADER_FRAGMENT) {
      if (agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer)) {
         hk_usc_upload_spilled_rt_descs(&b, cmd);
      }

      if (cmd->state.gfx.uses_blend_constant) {
         agx_usc_uniform(
            &b, 4, 8,
            root_ptr + hk_root_descriptor_offset(draw.blend_constant));
      }

      /* The SHARED state is baked into linked->usc for non-fragment shaders. We
       * don't pass around the information to bake the tilebuffer layout.
       *
       * TODO: We probably could with some refactor.
       */
      agx_usc_push_packed(&b, SHARED, &cmd->state.gfx.render.tilebuffer.usc);
      root_unif = AGX_ABI_FUNI_ROOT;
   }

   /* Address for the root and each set */
   agx_usc_uniform(&b, root_unif, 4 * (1 + s->info.set_count), root_ptr);

   agx_usc_push_blob(&b, linked->usc.data, linked->usc.size);
   return agx_usc_addr(&dev->dev, t.gpu);
}

void
hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                    enum agx_barrier barrier, enum libagx_program idx,
                    void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_precompiled_shader *prog = agx_get_precompiled(&dev->bg_eot, idx);
   /* Apple9 internal programs are uploaded on first use and can fail when
    * the USC aperture is full. */
   if (!prog) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;

   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(
      cmd, target, true, cmd->sync_scope ?: HK_SYNC_ALL);
   if (!cs)
      return;

   if (prog->apple9) {
      uint64_t args = hk_pool_upload(cmd, data, data_size, 8);
      hk_apple9_dispatch(cmd, cs, prog->bo, &prog->apple9_profile, grid, args, NULL,
                         NULL);
      return;
   }

   struct agx_ptr t = hk_pool_usc_alloc(cmd, agx_usc_size(15), 64);
   uint64_t uploaded_data = hk_pool_upload(cmd, data, data_size, 4);
   if (!t.cpu || !uploaded_data)
      return;

   agx_usc_words_precomp(t.cpu, &prog->b, uploaded_data, data_size);

   hk_dispatch_with_usc_launch(dev, cs, prog->b.launch,
                               agx_usc_addr(&dev->dev, t.gpu), grid,
                               prog->b.workgroup);
}

void
hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   uint8_t *map = cs->current;

   cs->tib = render->tilebuffer;
   /* Segment slots are per control stream; active queries continue in a new
    * segment here. */
   cs->occlusion_base = 0;
   cs->occlusion_segments = 0;
   if (cmd->state.gfx.occlusion.active_count)
      cmd->state.gfx.occlusion.segment_dirty = true;

   /* Assume this is not the first control stream of the render pass, so
    * initially use the partial background/EOT program and ZLS control.
    * hk_BeginRendering/hk_EndRendering will override.
    */
   const struct hk_image_view *c0 = render->color_att[0].iview;
   const struct hk_image_view *zs = render->depth_att.iview ?: render->stencil_att.iview;
   cs->diag.color0 = c0 ? (uintptr_t)c0->vk.image : 0;
   cs->diag.depth = zs ? (uintptr_t)zs->vk.image : 0;
   cs->diag.area = MIN2(render->area.extent.width, 0xffff) |
                   (MIN2(render->area.extent.height, 0xffff) << 16);
   const struct hk_image_view *v = c0 ?: zs;
   cs->diag.view = v ? (MIN2(v->vk.base_mip_level, 0xff) |
                        (MIN2(v->vk.base_array_layer, 0xfff) << 8) |
                        (MIN2(render->layer_count, 0xfff) << 20)) : 0;
   cs->diag.color0_format = render->color_att[0].vk_format;
   cs->diag.depth_format = render->depth_att.vk_format;
   cs->diag.color_count = render->color_att_count;
   cs->diag.samples = render->tilebuffer.nr_samples;
   /* clears: colour-0 cleared, depth cleared, colour-0 compressed, colour-0
    * storage usage, depth compressed. */
   const struct hk_image *c0i = c0 ? container_of(c0->vk.image, struct hk_image, vk) : NULL;
   const struct hk_image *zsi = zs ? container_of(zs->vk.image, struct hk_image, vk) : NULL;
   cs->diag.clears = render->color_att[0].clear | (render->depth_att.clear << 1) |
                     ((c0i && c0i->planes[0].layout.compressed) << 2) |
                     ((c0i && (c0i->vk.usage & VK_IMAGE_USAGE_STORAGE_BIT)) << 3) |
                     ((zsi && zsi->planes[0].layout.compressed) << 4);

   cs->cr = render->cr;
   cs->cr.bg.main = render->cr.bg.partial;
   cs->cr.eot.main = render->cr.eot.partial;
   cs->cr.zls_control = render->cr.zls_control_partial;

   if (hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G) {
      cs->scissor = UTIL_DYNARRAY_INIT;
      cs->depth_bias = UTIL_DYNARRAY_INIT;
      cs->ppp_multisamplectl = hk_default_sample_positions(cs->tib.nr_samples);
      hk_cmd_buffer_dirty_all(cmd);
      return;
   }

   /* Barrier to enforce GPU-CPU coherency, in case this batch is back to back
    * with another that caused stale data to be cached and the CPU wrote to it
    * in the meantime.
    */
   agx_push(map, VDM_BARRIER, cfg) {
      cfg.usc_cache_inval = true;
   }

   struct AGX_PPP_HEADER present = {
      .w_clamp = true,
      .occlusion_query_2 = true,
      .output_unknown = true,
      .varying_word_2 = true,
      .viewport_count = 1, /* irrelevant */
   };

   size_t size = agx_ppp_update_size(&present);
   struct agx_ptr T = hk_pool_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

   /* clang-format off */
   agx_ppp_push(&ppp, W_CLAMP, cfg) cfg.w_clamp = 1e-10;
   agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY_2, cfg);
   agx_ppp_push(&ppp, OUTPUT_UNKNOWN, cfg);
   agx_ppp_push(&ppp, VARYING_2, cfg);
   /* clang-format on */

   agx_ppp_fini(&map, &ppp);
   cs->current = map;

   cs->scissor = UTIL_DYNARRAY_INIT;
   cs->depth_bias = UTIL_DYNARRAY_INIT;

   /* All graphics state must be reemited in each control stream */
   hk_cmd_buffer_dirty_all(cmd);
}

void
hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       size_t space)
{
   bool vdm = cs->type == HK_CS_VDM;

   size_t link_length =
      vdm ? AGX_VDM_STREAM_LINK_LENGTH : AGX_CDM_STREAM_LINK_LENGTH;

   /* Assert that we have space for a link tag */
   assert((cs->current + link_length) <= cs->end && "Encoder overflowed");

   /* Always leave room for a link tag, in case we run out of space later,
    * plus padding because VDM apparently overreads?
    *
    * 0x200 is not enough. 0x400 seems to work. 0x800 for safety.
    */
   space += link_length + 0x800;

   /* If there is room in the command buffer, we're done */
   if (likely((cs->end - cs->current) >= space))
      return;

   /* Otherwise, we need to allocate a new command buffer. We use memory owned
    * by the batch to simplify lifetime management for the BO.
    */
   size_t size = 65536;
   struct agx_ptr T = hk_pool_cs_alloc(cmd, size, 256, !vdm);
   if (!T.cpu)
      return;

   /* Native VDM continuation pointers are relative to the render context. */
   uint64_t target = T.gpu;
   if (vdm && hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G) {
      assert(target >= AGX_APPLE9_RENDER_CONTEXT_BASE &&
             target - AGX_APPLE9_RENDER_CONTEXT_BASE <= UINT32_MAX);
      target -= AGX_APPLE9_RENDER_CONTEXT_BASE;
   }
   agx_cs_jump(cs->current, target, vdm);

   /* Swap out the control stream */
   cs->current = T.cpu;
   cs->end = cs->current + size;
   cs->chunk = T;
   cs->stream_linked = true;
}

static void
clear_attachment_as_image(struct hk_cmd_buffer *cmd,
                          struct hk_rendering_state *render,
                          struct hk_attachment *att, unsigned aspect)
{
   struct hk_image_view *view = att->iview;
   if (!att->clear || !view || !(view->vk.aspects & aspect))
      return;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   const VkImageSubresourceRange range = {
      .layerCount = layer_count,
      .levelCount = 1,
      .baseArrayLayer = view->vk.base_array_layer,
      .baseMipLevel = view->vk.base_mip_level,
      .aspectMask = view->vk.aspects & aspect,
   };

   assert(util_bitcount(range.aspectMask) == 1);
   VkFormat format = att->vk_format;

   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      format = vk_format_depth_only(format);
   } else if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
      format = vk_format_stencil_only(format);
   }

   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   hk_clear_image(cmd, image, vk_format_to_pipe_format(format),
                  att->clear_colour, &range, false /* partially clear 3D */);
}

void
hk_optimize_empty_vdm(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.gfx;
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      clear_attachment_as_image(cmd, render, &render->color_att[i], ~0);
   }

   clear_attachment_as_image(cmd, render, &render->depth_att,
                             VK_IMAGE_ASPECT_DEPTH_BIT);

   clear_attachment_as_image(cmd, render, &render->stencil_att,
                             VK_IMAGE_ASPECT_STENCIL_BIT);

   /* Remove the VDM control stream from the command buffer, now that it is
    * replaced by equivalent other operations.
    */
   list_del(&cs->node);
   *util_dynarray_element(&cmd->sync_remap, uint32_t, cs->sync_id) = HK_SYNC_NONE;
   hk_cs_destroy(cs);
}
