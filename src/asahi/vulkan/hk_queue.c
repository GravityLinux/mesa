/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "hk_gpu_busy.h"
#include "util/u_wait_watchdog.h"
#include "hk_queue.h"
#include "hk_buffer.h"

#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "decode.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include <xf86drm.h>
#include "util/list.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

#include "hk_private.h"
#include "layout.h"
#include "vk_drm_syncobj.h"
#include "vk_sync.h"

/*
 * We need to specially handle submits with no control streams. The kernel
 * can't accept empty submits, but we can end up here in Vulkan for
 * synchronization purposes only. Rather than submit a no-op job (slow),
 * we simply tie the fences together.
 */
static VkResult
queue_submit_empty(struct hk_device *dev, struct hk_queue *queue,
                   struct vk_queue_submit *submit)
{
   int fd = dev->dev.fd;
#ifdef HK_WAIT_WATCHDOG
   unsigned wd_count = submit->wait_count + submit->signal_count + 1;
   STACK_ARRAY(struct drm_asahi_sync, wd_syncs, wd_count);
   for (unsigned i = 0; i < submit->wait_count; ++i) {
      wd_syncs[i] = (struct drm_asahi_sync){
         .handle = vk_sync_as_drm_syncobj(submit->waits[i].sync)->syncobj,
         .timeline_value = submit->waits[i].wait_value,
      };
   }
   struct drm_asahi_sync *wd_signals = wd_syncs + submit->wait_count;
   for (unsigned i = 0; i < submit->signal_count; ++i) {
      wd_signals[i] = (struct drm_asahi_sync){
         .handle = vk_sync_as_drm_syncobj(submit->signals[i].sync)->syncobj,
         .timeline_value = submit->signals[i].signal_value,
      };
   }
   wd_signals[submit->signal_count] = (struct drm_asahi_sync){
      .handle = queue->drm.syncobj,
      .timeline_value = queue->drm.timeline_value + submit->wait_count,
   };
   struct u_watchdog_objects wd_waits = {
      .objects = &wd_syncs[0].handle, .points = &wd_syncs[0].timeline_value,
      .object_stride = sizeof(*wd_syncs), .point_stride = sizeof(*wd_syncs),
      .count = submit->wait_count, .object32 = true,
   };
   struct u_watchdog_objects wd_out = {
      .objects = &wd_signals[0].handle, .points = &wd_signals[0].timeline_value,
      .object_stride = sizeof(*wd_syncs), .point_stride = sizeof(*wd_syncs),
      .count = submit->signal_count + 1, .object32 = true,
   };
   uint64_t wd_id = u_watchdog_submit(fd, queue->drm.id, wd_waits, wd_out);
   U_WATCHDOG_SCOPE(wd, "HK empty submit transfers", fd, queue->drm.id,
                    wd_id, 0, UINT64_MAX, wd_waits);
#endif
   int wd_result = 0, wd_error = 0;


   /* Transfer the waits into the queue timeline. */
   for (unsigned i = 0; i < submit->wait_count; ++i) {
      struct vk_sync_wait *wait = &submit->waits[i];

      assert(vk_sync_type_is_drm_syncobj(wait->sync->type));
      const struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(wait->sync);

      int ret = drmSyncobjTransfer(fd, queue->drm.syncobj, ++queue->drm.timeline_value,
                                  syncobj->syncobj, wait->wait_value, 0);
      u_watchdog_ioctl("HK empty submit wait transfer", fd, ret, errno);
      if (ret) {
         wd_result = ret;
         wd_error = errno;
      }
   }

   /* Transfer the queue timeline into each out fence. They will all be
    * signalled when we reach this point.
    */
   for (unsigned i = 0; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];

      assert(vk_sync_type_is_drm_syncobj(signal->sync->type));
      const struct vk_drm_syncobj *syncobj =
         vk_sync_as_drm_syncobj(signal->sync);

      int ret = drmSyncobjTransfer(fd, syncobj->syncobj, signal->signal_value,
                                  queue->drm.syncobj, queue->drm.timeline_value, 0);
      u_watchdog_ioctl("HK empty submit signal transfer", fd, ret, errno);
      if (ret) {
         wd_result = ret;
         wd_error = errno;
      }
   }

   u_watchdog_submit_result(wd_id, wd_result, wd_error);
#ifdef HK_WAIT_WATCHDOG
   /* End the borrowed-array scope before releasing a heap-backed STACK_ARRAY. */
   u_watchdog_end(&wd);
   STACK_ARRAY_FINISH(wd_syncs);
#endif
   /* The timeline value was already advanced: a failed transfer would leave
    * waits on it hanging forever.
    */
   if (wd_result) {
      mesa_loge("empty submit syncobj transfer failed: %s", strerror(wd_error));
      return VK_ERROR_DEVICE_LOST;
   }

   return VK_SUCCESS;
}

static void
asahi_fill_cdm_command(struct hk_device *dev, struct hk_cs *cs,
                       struct drm_asahi_cmd_compute *cmd)
{
   size_t len = cs->stream_linked ? cs->root_size : (cs->current - cs->start);

   *cmd = (struct drm_asahi_cmd_compute){
      .cdm_ctrl_stream_base = cs->addr,
      .cdm_ctrl_stream_end = cs->addr + len,

      .ts.end.handle = cs->timestamp.end.handle,
      .ts.end.offset = cs->timestamp.end.offset_B,
   };

   if (cs->uses_sampler_heap) {
      cmd->sampler_heap = dev->samplers.table.bo->va->addr;
      cmd->sampler_count = dev->samplers.table.alloc;
   }

   if (cs->scratch.cs.main || cs->scratch.cs.preamble) {
      cmd->helper.data = dev->scratch.cs.buf->va->addr;
      cmd->helper.cfg = cs->scratch.cs.preamble ? (1 << 16) : 0;
      cmd->helper.binary = agx_helper_program(&dev->bg_eot);
   }
}

static void
asahi_fill_vdm_command(struct hk_device *dev, struct hk_cs *cs,
                       struct drm_asahi_cmd_render *c)
{
   memset(c, 0, sizeof(*c));

   c->vdm_ctrl_stream_base = cs->addr;
   if ((dev->dev.params.features & DRM_ASAHI_FEATURE_FEW_PRIMITIVES) &&
       agx_has_few_primitives(cs->primitive_count))
      c->flags |= DRM_ASAHI_RENDER_FEW_PRIMITIVES;

   agx_pack(&c->ppp_ctrl, CR_PPP_CONTROL, cfg) {
      cfg.enable_w_clamp = true;
      cfg.fixed_point_format = 1;
   }

   c->width_px = cs->cr.width;
   c->height_px = cs->cr.height;

   c->isp_bgobjdepth = cs->cr.isp_bgobjdepth;
   c->isp_bgobjvals = cs->cr.isp_bgobjvals;

   static_assert(sizeof(c->zls_ctrl) == sizeof(cs->cr.zls_control));
   memcpy(&c->zls_ctrl, &cs->cr.zls_control, sizeof(cs->cr.zls_control));

   agx_pack(&c->isp_zls_pixels, CR_ISP_ZLS_PIXELS, cfg) {
      cfg.x = cs->cr.zls_width;
      cfg.y = cs->cr.zls_height;
   }

   c->depth.base = cs->cr.depth.buffer;
   c->depth.stride = cs->cr.depth.stride;
   c->depth.comp_base = cs->cr.depth.meta;
   c->depth.comp_stride = cs->cr.depth.meta_stride;
   c->stencil.base = cs->cr.stencil.buffer;
   c->stencil.stride = cs->cr.stencil.stride;
   c->stencil.comp_base = cs->cr.stencil.meta;
   c->stencil.comp_stride = cs->cr.stencil.meta_stride;

   if (cs->cr.dbias_is_int == U_TRISTATE_YES) {
      c->flags |= DRM_ASAHI_RENDER_DBIAS_IS_INT;
   }

   if (dev->dev.debug & AGX_DBG_NOCLUSTER) {
      c->flags |= DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING;
   }

   agx_tilebuffer_set_drm_cmd(c, &cs->tib);

   /* Can be 0 for attachmentless rendering with no draws */
   c->samples = MAX2(cs->tib.nr_samples, 1);
   c->layers = cs->cr.layers;

   /* Drawing max size will OOM and fail submission. But vkd3d-proton does this
    * for emulating no-attachment rendering. Clamp to something reasonable and
    * hope this is good enough in practice. This only affects a case that would
    * otherwise be guaranteed broken.
    *
    * XXX: Hack for vkd3d-proton.
    */
   if (c->layers == 2048 && c->width_px == 16384 && c->height_px == 16384) {
      mesa_log(MESA_LOG_WARN, MESA_LOG_TAG, "Clamping massive framebuffer");
      c->layers = 32;
   }

   c->ppp_multisamplectl = cs->ppp_multisamplectl;
   c->sample_size_B = cs->tib.sample_size_B;

   float tan_60 = 1.732051f;
   c->isp_merge_upper_x = fui(tan_60 / cs->cr.width);
   c->isp_merge_upper_y = fui(tan_60 / cs->cr.height);

   c->bg.usc = cs->cr.bg.main.usc | 4;
   c->eot.usc = cs->cr.eot.main.usc | 4;
   c->partial_bg.usc = cs->cr.bg.partial.usc | 4;
   c->partial_eot.usc = cs->cr.eot.partial.usc | 4;

   memcpy(&c->bg.rsrc_spec, &cs->cr.bg.main.counts,
          sizeof(struct agx_counts_packed));

   memcpy(&c->eot.rsrc_spec, &cs->cr.eot.main.counts,
          sizeof(struct agx_counts_packed));

   memcpy(&c->partial_bg.rsrc_spec, &cs->cr.bg.partial.counts,
          sizeof(struct agx_counts_packed));

   memcpy(&c->partial_eot.rsrc_spec, &cs->cr.eot.partial.counts,
          sizeof(struct agx_counts_packed));

   if (dev->dev.chip == AGX_CHIP_G16G) {
      c->bg.usc = cs->cr.bg.main.usc;
      c->eot.usc = cs->cr.eot.main.usc;
      c->partial_bg.usc = cs->cr.bg.partial.usc;
      c->partial_eot.usc = cs->cr.eot.partial.usc;
      c->bg.rsrc_spec = c->eot.rsrc_spec = 0;
      c->partial_bg.rsrc_spec = c->partial_eot.rsrc_spec = 0;
   }

   c->isp_scissor_base = cs->uploaded_scissor;
   /* An empty upload may point one past the command pool allocation. There is
    * no table for the kernel to validate or the GPU to read in that case.
    */
   c->isp_dbias_base = cs->depth_bias.size ? cs->uploaded_zbias : 0;

   if (cs->uses_sampler_heap) {
      c->sampler_heap = dev->samplers.table.bo->va->addr;
      c->sampler_count = dev->samplers.table.alloc;
   }

   c->isp_oclqry_base = cs->occlusion_base;

   if (cs->cr.process_empty_tiles)
      c->flags |= DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES;

   if (cs->scratch.vs.main || cs->scratch.vs.preamble) {
      c->flags |= DRM_ASAHI_RENDER_VERTEX_SCRATCH;
      c->vertex_helper.data = dev->scratch.vs.buf->va->addr;
      c->vertex_helper.cfg = cs->scratch.vs.preamble ? (1 << 16) : 0;
      c->vertex_helper.binary = agx_helper_program(&dev->bg_eot);
   }

   if (cs->scratch.fs.main || cs->scratch.fs.preamble) {
      c->fragment_helper.data = dev->scratch.fs.buf->va->addr;
      c->fragment_helper.cfg = cs->scratch.fs.preamble ? (1 << 16) : 0;
      c->fragment_helper.binary = agx_helper_program(&dev->bg_eot);
   }

   if (cs->timestamp.end.handle) {
      c->ts_frag.end.handle = cs->timestamp.end.handle;
      c->ts_frag.end.offset = cs->timestamp.end.offset_B;
   }
}

static void
asahi_fill_sync(struct drm_asahi_sync *sync, struct vk_sync *vk_sync,
                uint64_t value)
{
   if (unlikely(!vk_sync_type_is_drm_syncobj(vk_sync->type))) {
      UNREACHABLE("Unsupported sync type");
      return;
   }

   const struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(vk_sync);
   *sync = (struct drm_asahi_sync){.handle = syncobj->syncobj};

   if (vk_sync->flags & VK_SYNC_IS_TIMELINE) {
      sync->sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ;
      sync->timeline_value = value;
   } else {
      sync->sync_type = DRM_ASAHI_SYNC_SYNCOBJ;
   }
}

union drm_asahi_cmd {
   struct drm_asahi_cmd_compute compute;
   struct drm_asahi_cmd_render render;
};

/* XXX: Batching multiple commands per submission is causing rare (7ppm) flakes
 * on the CTS once lossless compression is enabled. This needs to be
 * investigated before we can reenable this mechanism. We are likely missing a
 * cache flush or barrier somewhere.
 */
static inline unsigned
max_commands_per_submit(struct hk_device *dev)
{
   return HK_PERF(dev, BATCH) ?
      MAX2(1, MIN2(dev->dev.params.max_commands_per_submission,
                   DRM_ASAHI_BARRIER_NONE - 1)) : 1;
}

static VkResult
queue_submit_single(struct hk_device *dev, struct drm_asahi_submit *submit,
                    unsigned ring_idx)
{
   struct agx_submit_virt virt = {.ring_idx = ring_idx};

   if (dev->dev.is_virtio) {
      u_rwlock_rdlock(&dev->external_bos.lock);
      virt.extres_count = util_dynarray_num_elements(
         &dev->external_bos.list, struct asahi_ccmd_submit_res);
      virt.extres = util_dynarray_begin(&dev->external_bos.list);
   }

#ifdef HK_WAIT_WATCHDOG
   const struct drm_asahi_sync *wd_syncs = (const void *)(uintptr_t)submit->syncs;
   const struct drm_asahi_sync *wd_signals = wd_syncs + submit->in_sync_count;
   struct u_watchdog_objects wd_waits = {
      .objects = submit->in_sync_count ? &wd_syncs[0].handle : NULL,
      .points = submit->in_sync_count ? &wd_syncs[0].timeline_value : NULL,
      .object_stride = sizeof(*wd_syncs), .point_stride = sizeof(*wd_syncs),
      .count = submit->in_sync_count, .object32 = true,
   };
   struct u_watchdog_objects wd_out = {
      .objects = submit->out_sync_count ? &wd_signals[0].handle : NULL,
      .points = submit->out_sync_count ? &wd_signals[0].timeline_value : NULL,
      .object_stride = sizeof(*wd_syncs), .point_stride = sizeof(*wd_syncs),
      .count = submit->out_sync_count, .object32 = true,
   };
   uint64_t wd_id = u_watchdog_submit(dev->dev.fd, submit->queue_id, wd_waits, wd_out);
   U_WATCHDOG_SCOPE(wd, "HK submit ioctl", dev->dev.fd, submit->queue_id, wd_id,
                    0, UINT64_MAX, wd_waits);
#endif
   int ret = dev->dev.ops.submit(&dev->dev, submit, &virt);
   int submit_errno = errno;
   u_watchdog_submit_result(wd_id, ret, errno);
   u_watchdog_ioctl("HK submit", dev->dev.fd, ret, errno);

   if (dev->dev.is_virtio)
      u_rwlock_rdunlock(&dev->external_bos.lock);

   /* The kernel rejects submissions once the GPU has failed (EIO) or when
    * a command is invalid. Either way the queue cannot make progress: report
    * device loss rather than silently dropping the work.
    */
   if (ret) {
      mesa_loge("DRM_IOCTL_ASAHI_SUBMIT failed: %s", strerror(submit_errno));
      return VK_ERROR_DEVICE_LOST;
   }

   return VK_SUCCESS;
}

/*
 * The kernel/firmware jointly impose a limit on commands per submit ioctl, but
 * we can build up arbitrarily large command buffers. We handle this here by
 * looping the ioctl, submitting slices of the command buffers that are within
 * bounds.
 */
static VkResult
queue_submit_looped(struct hk_device *dev, struct drm_asahi_submit *submit,
                    unsigned command_count, unsigned ring_idx)
{
   uint8_t *cmdbuf = (uint8_t *)(uintptr_t)submit->cmdbuf;
   uint32_t offs = 0;
   unsigned commands_remaining = command_count;

   uint64_t out_syncs =
      submit->syncs + sizeof(struct drm_asahi_sync) * submit->in_sync_count;

   while (commands_remaining) {
      bool first = commands_remaining == command_count;
      bool last = commands_remaining <= max_commands_per_submit(dev);

      unsigned count = MIN2(commands_remaining, max_commands_per_submit(dev));
      commands_remaining -= count;

      assert(!last || commands_remaining == 0);
      assert(count > 0);

      unsigned base_offs = offs;

      /* Headers already contain slice-relative barriers. */
      for (unsigned i = 0; i < count; ++i) {
         struct drm_asahi_cmd_header *cmd = (void *)(cmdbuf + offs);
         offs += sizeof(*cmd) + cmd->size;

      }


      bool has_in_syncs = first;
      bool has_out_syncs = last;

      struct drm_asahi_submit submit_ioctl = {
         .flags = submit->flags,
         .queue_id = submit->queue_id,
         .cmdbuf = submit->cmdbuf + base_offs,
         .cmdbuf_size = offs - base_offs,

         .syncs = has_in_syncs ? submit->syncs : out_syncs,
         .in_sync_count = has_in_syncs ? submit->in_sync_count : 0,
         .out_sync_count = has_out_syncs ? submit->out_sync_count : 0,
      };

      VkResult result = queue_submit_single(dev, &submit_ioctl, ring_idx);
      if (result != VK_SUCCESS)
         return result;

   }

   return VK_SUCCESS;
}

struct hk_bind_builder {
   /* Initialized */
   struct hk_device *dev;
   struct vk_object_base *obj_base;
   struct agx_va *va;
   struct hk_image *image;
   struct hk_buffer *buffer;

   /* State */
   struct hk_device_memory *mem;
   VkDeviceSize resourceOffset;
   VkDeviceSize size;
   VkDeviceSize memoryOffset;
   VkResult result;

   /* Array of drm_asahi_gem_bind_op's */
   struct util_dynarray binds;
};

static inline struct hk_bind_builder
hk_bind_builder(struct hk_device *dev, struct vk_object_base *obj_base,
                struct agx_va *va, struct hk_image *image)
{
   struct hk_bind_builder b = {
      .dev = dev,
      .obj_base = obj_base,
      .va = va,
      .image = image,
   };

   b.binds = UTIL_DYNARRAY_INIT;
   return b;
}

static VkResult
hk_flush_bind(struct hk_bind_builder *b)
{
   if (b->result != VK_SUCCESS || b->size == 0) {
      return b->result;
   }

   perf_debug(b->dev, "Sparse bind");

   uint64_t va_addr = b->va->addr + b->resourceOffset;

   /* If we have an image with sparse residency, we have a userspace-managed
    * sparse page table map, which we need to keep in sync with the real
    * kernel-managed page table.  This ensures textures get strict residency
    * semantics, using the hardware sparse support.
    */
   /* Apple9 shaders instead look up residency by page in the resource. */
   if (b->buffer && b->buffer->apple9_residency != NULL) {
      uint8_t *r = agx_bo_map(b->buffer->apple9_residency);
      uint64_t first_page = b->resourceOffset / AIL_PAGESIZE;
      uint64_t end_page =
         MIN2(DIV_ROUND_UP(b->resourceOffset + b->size, AIL_PAGESIZE),
              DIV_ROUND_UP(b->buffer->va->size_B, AIL_PAGESIZE));

      for (uint64_t page = first_page; page < end_page; ++page)
         r[page] = b->mem != NULL;
   }

   if (b->image && b->image->planes[0].apple9_residency != NULL) {
      struct hk_apple9_residency *r =
         agx_bo_map(b->image->planes[0].apple9_residency);
      uint64_t first_page = ail_bytes_to_pages(b->resourceOffset);
      uint64_t end_page =
         MIN2(first_page + ail_bytes_to_pages(b->size), r->pages);

      for (uint64_t page = first_page; page < end_page; ++page)
         r->resident[page] = b->mem != NULL;
   }

   if (b->image && b->image->planes[0].sparse_map != NULL) {
      assert(b->image->plane_count == 1 && "multiplane sparse not supported");

      uint32_t *map = agx_bo_map(b->image->planes[0].sparse_map);
      uint64_t size_page = ail_bytes_to_pages(b->size);

      struct ail_layout *layout = &b->image->planes[0].layout;
      uint64_t layer_stride_page = ail_bytes_to_pages(layout->layer_stride_B);

      for (unsigned offs_page = 0; offs_page < size_page; offs_page++) {
         /* Determine the target page to bind */
         uint64_t target_page =
            ail_bytes_to_pages(b->resourceOffset) + offs_page;

         /* The page table is per-layer. Fortunately, layers are page-aligned,
          * so we can divide to find the layer & the page relative to the start
          * of the layer, which give us the index into the sparse map.
          *
          * Note that we can end up out-of-bounds since the hardware page size
          * (16k) is smaller than the Vulkan standard sparse block size (65k).
          * Just clamp out-of-bounds maps - there is sufficient VA space for
          * them but not sufficient sparse map space for them.
          */
         uint64_t z = target_page / layer_stride_page;
         if (z >= layout->depth_px)
            break;

         uint64_t page_in_layer = target_page % layer_stride_page;
         unsigned idx = ail_page_to_sparse_index_el(layout, z, page_in_layer);

         agx_pack(map + idx, SPARSE_BLOCK, cfg) {
            cfg.enabled = b->mem != NULL;
            cfg.unknown = cfg.enabled;

            if (cfg.enabled) {
               cfg.address = va_addr + (offs_page * AIL_PAGESIZE);
            }
         }
      }
   }

   /* When the app wants to unbind, replace the bound pages with scratch pages
    * so we don't leave a gap.
    */
   struct drm_asahi_gem_bind_op op;
   if (!b->mem && hk_sparse_unbound_unmapped(&b->dev->dev)) {
      /* Leave both views unmapped, see hk_sparse_unbound_unmapped. */
      op = (struct drm_asahi_gem_bind_op){
         .flags = DRM_ASAHI_BIND_UNBIND,
         .addr = b->va->addr + b->resourceOffset,
         .range = b->size,
      };
      util_dynarray_append(&b->binds, op);
      op.addr = agx_rw_addr_to_ro(&b->dev->dev, op.addr);
      util_dynarray_append(&b->binds, op);
      return VK_SUCCESS;
   } else if (!b->mem) {
      op = (struct drm_asahi_gem_bind_op){
         .handle = b->dev->dev.scratch_bo->uapi_handle,
         .flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE |
                  DRM_ASAHI_BIND_SINGLE_PAGE,
         .addr = b->va->addr + b->resourceOffset,
         .range = b->size,
      };
   } else {
      op = (struct drm_asahi_gem_bind_op){
         .handle = b->mem->bo->uapi_handle,
         .flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
         .addr = va_addr,
         .offset = b->memoryOffset,
         .range = b->size,
      };
   }

   util_dynarray_append(&b->binds, op);

   /* Shadow a read-only mapping to the upper half */
   op.flags &= ~DRM_ASAHI_BIND_WRITE;
   op.addr = agx_rw_addr_to_ro(&b->dev->dev, op.addr);

   if (!b->mem) {
      op.handle = b->dev->dev.zero_bo->uapi_handle;
   }

   util_dynarray_append(&b->binds, op);

   return VK_SUCCESS;
}

static int
hk_bind_builder_finish(struct hk_bind_builder *b)
{
   hk_flush_bind(b);

   /* Submit everything to the kernel at once */
   if (b->binds.size > 0) {
      b->dev->dev.ops.bo_bind(
         &b->dev->dev, b->binds.data,
         util_dynarray_num_elements(&b->binds, struct drm_asahi_gem_bind_op));
   }

   util_dynarray_fini(&b->binds);
   return b->result;
}

static void
hk_add_bind(struct hk_bind_builder *b, struct hk_device_memory *mem,
            VkDeviceSize resourceOffset, VkDeviceSize size,
            VkDeviceSize memoryOffset)
{
   /* Discard trivial binds to simplify the below logic. */
   if (size == 0)
      return;

   /* Try to merge with the previous bind */
   if (b->size && b->mem == mem &&
       resourceOffset == b->resourceOffset + b->size &&
       (!mem || memoryOffset == b->memoryOffset + b->size)) {

      b->size += size;
      return;
   }

   /* Otherwise, flush the previous bind and replace with the new one */
   hk_flush_bind(b);
   b->mem = mem;
   b->resourceOffset = resourceOffset;
   b->size = size;
   b->memoryOffset = memoryOffset;
}

static VkResult
hk_sparse_buffer_bind_memory(struct hk_device *device,
                             const VkSparseBufferMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_buffer, buffer, bind->buffer);

   struct hk_bind_builder b =
      hk_bind_builder(device, &buffer->vk.base, buffer->va, NULL);
   b.buffer = buffer;

   for (uint32_t i = 0; i < bind->bindCount; ++i) {
      struct hk_device_memory *cur_mem = NULL;

      if (bind->pBinds[i].memory != VK_NULL_HANDLE)
         cur_mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

      hk_add_bind(&b, cur_mem, bind->pBinds[i].resourceOffset,
                  bind->pBinds[i].size, bind->pBinds[i].memoryOffset);
   }

   return hk_bind_builder_finish(&b);
}

static VkResult
hk_sparse_image_opaque_bind_memory(
   struct hk_device *device, const VkSparseImageOpaqueMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_image, image, bind->image);

   struct hk_bind_builder b =
      hk_bind_builder(device, &image->vk.base, image->planes[0].va, image);

   for (uint32_t i = 0; i < bind->bindCount; ++i) {
      struct hk_device_memory *mem = NULL;
      if (bind->pBinds[i].memory != VK_NULL_HANDLE)
         mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

      VkDeviceSize resourceOffset = bind->pBinds[i].resourceOffset;

      /* Conceptually, the miptail is a single region at the end of the image,
       * possibly layered. However, due to alignment requirements we need to
       * use a non-layered miptail and internally fan out to each of the layers.
       * This is facilitated by the HK_MIP_TAIL_START_OFFSET magic offset, see
       * the comment where that is defined for more detail.
       */
      if (resourceOffset >= HK_MIP_TAIL_START_OFFSET) {
         assert(resourceOffset == HK_MIP_TAIL_START_OFFSET &&
                "must bind whole miptail... maybe...");

         const struct ail_layout *layout = &image->planes[0].layout;
         unsigned tail_offset_B =
            layout->level_offsets_B[layout->mip_tail_first_lod];

         for (unsigned z = 0; z < layout->depth_px; ++z) {
            uint64_t image_offs = tail_offset_B + (z * layout->layer_stride_B);
            uint64_t mem_offs =
               bind->pBinds[i].memoryOffset + (z * layout->mip_tail_stride);

            hk_add_bind(&b, mem, image_offs, layout->mip_tail_stride, mem_offs);
         }
      } else {
         hk_add_bind(&b, mem, bind->pBinds[i].resourceOffset,
                     bind->pBinds[i].size, bind->pBinds[i].memoryOffset);
      }
   }

   return hk_bind_builder_finish(&b);
}

static void
bind_hw_tile(struct hk_bind_builder *b, struct hk_device_memory *mem,
             struct ail_layout *layout, unsigned layer, unsigned level,
             VkOffset3D offset, VkExtent3D extent, struct ail_tile std_size_el,
             unsigned mem_offset, unsigned x, unsigned y, unsigned z)
{
   uint64_t bo_offset_B = ail_get_twiddled_block_B(
      layout, level, offset.x + x, offset.y + y, layer + offset.z + z);

   /* Consider the standard tiles in the bound memory to be in raster order, and
    * address accordingly in standard tiles.
    */
   unsigned mem_x_stl = x / std_size_el.width_el;
   unsigned mem_y_stl = y / std_size_el.height_el;
   unsigned extent_w_stl = DIV_ROUND_UP(extent.width, std_size_el.width_el);
   unsigned extent_y_stl = DIV_ROUND_UP(extent.height, std_size_el.height_el);
   unsigned mem_offs_stl = (extent_y_stl * extent_w_stl * z) +
                           (extent_w_stl * mem_y_stl) + mem_x_stl;

   /* There are 4 hardware tiles per standard tile, so offset
    * accordingly for each hardware tile.
    */
   unsigned mem_offset_B = mem_offset + (mem_offs_stl * 4 * AIL_PAGESIZE);

   if (x % std_size_el.width_el)
      mem_offset_B += AIL_PAGESIZE;

   if (y % std_size_el.height_el)
      mem_offset_B += (2 * AIL_PAGESIZE);

   hk_add_bind(b, mem, bo_offset_B, AIL_PAGESIZE, mem_offset_B);
}

static VkResult
hk_sparse_image_bind_memory(struct hk_device *device,
                            const VkSparseImageMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_image, image, bind->image);
   struct ail_layout *layout = &image->planes[0].layout;

   struct hk_bind_builder b =
      hk_bind_builder(device, &image->vk.base, image->planes[0].va, image);

   for (uint32_t i = 0; i < bind->bindCount; ++i) {
      struct hk_device_memory *mem = NULL;
      if (bind->pBinds[i].memory != VK_NULL_HANDLE)
         mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

      uint64_t mem_offset = bind->pBinds[i].memoryOffset;
      const uint32_t layer = bind->pBinds[i].subresource.arrayLayer;
      const uint32_t level = bind->pBinds[i].subresource.mipLevel;

      VkExtent3D bind_extent = bind->pBinds[i].extent;
      bind_extent.width = DIV_ROUND_UP(
         bind_extent.width, vk_format_get_blockwidth(image->vk.format));
      bind_extent.height = DIV_ROUND_UP(
         bind_extent.height, vk_format_get_blockheight(image->vk.format));

      VkOffset3D bind_offset = bind->pBinds[i].offset;
      bind_offset.x /= vk_format_get_blockwidth(image->vk.format);
      bind_offset.y /= vk_format_get_blockheight(image->vk.format);

      /* Hardware tiles are exactly one page (16K) */
      struct ail_tile tilesize_el = layout->tilesize_el[level];
      unsigned size_B = tilesize_el.width_el * tilesize_el.height_el *
                        ail_get_blocksize_B(layout);

      assert(size_B == AIL_PAGESIZE && "fundamental to AGX");

      /* Standard tiles are exactly 4 pages (65K), consisting of a 2x2 grid of
       * hardware tiles.
       */
      struct ail_tile std_size_el = tilesize_el;
      std_size_el.width_el *= 2;
      std_size_el.height_el *= 2;

      for (unsigned z = 0; z < bind_extent.depth; z += 1) {
         for (unsigned y = 0; y < bind_extent.height;
              y += tilesize_el.height_el) {
            for (unsigned x = 0; x < bind_extent.width;
                 x += tilesize_el.width_el) {
               bind_hw_tile(&b, mem, layout, layer, level, bind_offset,
                            bind_extent, std_size_el, mem_offset, x, y, z);
            }
         }
      }
   }

   return hk_bind_builder_finish(&b);
}

static VkResult
hk_queue_submit_bind_sparse_memory(struct hk_device *device,
                                   struct vk_queue_submit *submission)
{
   assert(submission->command_buffer_count == 0);

   for (uint32_t i = 0; i < submission->buffer_bind_count; ++i) {
      VkResult result =
         hk_sparse_buffer_bind_memory(device, submission->buffer_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   for (uint32_t i = 0; i < submission->image_opaque_bind_count; ++i) {
      VkResult result = hk_sparse_image_opaque_bind_memory(
         device, submission->image_opaque_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   for (uint32_t i = 0; i < submission->image_bind_count; ++i) {
      VkResult result =
         hk_sparse_image_bind_memory(device, submission->image_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

/* Resolve recording-time stream identities only in submission-local storage.
 * In particular, secondary replay, repeated submissions, and simultaneous use
 * must never write queue-relative indices into a recorded command buffer.
 */
static struct hk_sync_stream *
hk_resolve_sync(struct hk_cmd_buffer *cmd, struct hk_sync_state *state,
                uint32_t nr_vdm, uint32_t nr_cdm)
{
   unsigned count = util_dynarray_num_elements(&cmd->sync_remap, uint32_t);
   struct hk_sync_stream *streams = calloc(MAX2(count, 1), sizeof(*streams));
   if (!streams)
      return NULL;

   uint32_t next[HK_SYNC_ENGINE_COUNT] = {nr_vdm, nr_cdm};
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      struct hk_sync_stream *s = &streams[cs->sync_id];
      s->wait = hk_sync_empty();
      s->wait_tiling = hk_sync_empty();
      memcpy(s->before, next, sizeof(next));
      s->engine = cs->type == HK_CS_CDM ? HK_SYNC_CDM : HK_SYNC_RENDER;
      s->index = ++next[s->engine];
   }

   bool precise = hk_sync_resolve(
      state, cmd->sync_events.data,
      util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event),
      cmd->sync_remap.data, streams, count, cmd->sync_fallback);
   if (!precise)
      perf_debug(cmd, "Conservative synchronization for reordered helpers");

   return streams;
}

/* Defer a render command's render dependency to its fragment stage when its
 * tiling stage depends on no earlier render work. Tiling commands on one queue
 * run in order, and fragment work waits for the same render completions as
 * before. Opt-in until measured: without the kernel's render_barrier=2 the
 * deferred wait costs more per pass than the tiling wait it replaces.
 */
DEBUG_GET_ONCE_BOOL_OPTION(hk_apple9_fragment_barrier,
                           "HK_APPLE9_FRAGMENT_BARRIER", false)

static bool
hk_defer_render_wait(struct hk_device *dev, const struct hk_sync_stream *s)
{
   return (dev->dev.params.features & DRM_ASAHI_FEATURE_FRAGMENT_BARRIER) &&
          debug_get_option_hk_apple9_fragment_barrier() &&
          s->wait.engine[HK_SYNC_RENDER] != HK_SYNC_NONE &&
          s->wait_tiling.engine[HK_SYNC_RENDER] == HK_SYNC_NONE;
}

static uint16_t
hk_ioctl_barrier(uint32_t index, uint32_t submitted)
{
   uint32_t rebased = hk_sync_rebase(index, submitted);
   assert(rebased == HK_SYNC_NONE || rebased < DRM_ASAHI_BARRIER_NONE);
   return rebased == HK_SYNC_NONE ? DRM_ASAHI_BARRIER_NONE : rebased;
}

static VkResult
queue_submit(struct hk_device *dev, struct hk_queue *queue,
             struct vk_queue_submit *submit)
{
   /* TODO: Support asynchronous sparse queue? */
   if (submit->buffer_bind_count || submit->image_bind_count ||
       submit->image_opaque_bind_count) {

      VkResult result = hk_queue_submit_bind_sparse_memory(dev, submit);
      if (result != VK_SUCCESS)
         return result;
   }

   unsigned command_count = 0;

   /* Gather the number of individual commands to submit up front */
   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      command_count += list_length(&cmdbuf->control_streams);
   }

   perf_debug_dev(&dev->dev,
                  "Submitting %u control streams (%u command buffers)",
                  command_count, submit->command_buffer_count);

   struct hk_sync_state sync = queue->sync;
   if (command_count == 0) {
      for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
         struct hk_cmd_buffer *cmd = (void *)submit->command_buffers[i];
         struct hk_sync_stream *streams = hk_resolve_sync(cmd, &sync, 0, 0);
         if (!streams)
            return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         free(streams);
      }
      VkResult result = queue_submit_empty(dev, queue, submit);
      if (result == VK_SUCCESS) {
         hk_sync_next_submit(&sync);
         queue->sync = sync;
      }
      return result;
   }

   unsigned wait_count = 0;
   struct drm_asahi_sync *syncs =
      alloca((submit->wait_count + submit->signal_count + 1) *
             sizeof(struct drm_asahi_sync));

   for (unsigned i = 0; i < submit->wait_count; ++i) {
      /* The kernel rejects the submission if we try to wait on the same
       * timeline semaphore at multiple points.
       *
       * TODO: Can we relax the UAPI?
       *
       * XXX: This is quadratic time.
       */
      bool skip = false;
      if (submit->waits[i].sync->flags & VK_SYNC_IS_TIMELINE) {
         uint32_t v1 = submit->waits[i].wait_value;
         for (unsigned j = 0; j < submit->wait_count; ++j) {
            uint32_t v2 = submit->waits[j].wait_value;
            if (i != j && submit->waits[i].sync == submit->waits[j].sync &&
                (v1 < v2 || (v1 == v2 && i < j))) {
               skip = true;
               break;
            }
         }

         if (skip)
            continue;
      }

      asahi_fill_sync(&syncs[wait_count++], submit->waits[i].sync,
                      submit->waits[i].wait_value);
   }

   for (unsigned i = 0; i < submit->signal_count; ++i) {
      asahi_fill_sync(&syncs[wait_count + i], submit->signals[i].sync,
                      submit->signals[i].signal_value);
   }

   /* Signal progress on the queue itself */
   syncs[wait_count + submit->signal_count] = (struct drm_asahi_sync){
      .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
      .handle = queue->drm.syncobj,
      .timeline_value = ++queue->drm.timeline_value,
   };

   /* Now setup the command structs */
   struct util_dynarray payload = UTIL_DYNARRAY_INIT;

   unsigned nr_vdm = 0, nr_cdm = 0;
   unsigned slice_vdm = 0, slice_cdm = 0, emitted = 0;
   unsigned last_slice_offset = 0;
   const unsigned slice_limit = max_commands_per_submit(dev);

   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      struct hk_sync_stream *streams =
         hk_resolve_sync(cmdbuf, &sync, nr_vdm, nr_cdm);
      if (!streams) {
         util_dynarray_fini(&payload);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      list_for_each_entry(struct hk_cs, cs, &cmdbuf->control_streams, node) {
         if ((emitted++ % slice_limit) == 0) {
            slice_vdm = nr_vdm;
            slice_cdm = nr_cdm;
            last_slice_offset = payload.size;
         }
         const struct hk_sync_dep *wait = &streams[cs->sync_id].wait;
         struct drm_asahi_cmd_header header =
            agx_cmd_header(cs->type == HK_CS_CDM,
                           hk_ioctl_barrier(wait->engine[HK_SYNC_RENDER], slice_vdm),
                           hk_ioctl_barrier(wait->engine[HK_SYNC_CDM], slice_cdm));

         util_dynarray_append(&payload, header);

         if (cs->type == HK_CS_CDM) {
            perf_debug(
               cmdbuf,
               "%u: Submitting CDM with %u API calls, %u dispatches, %u flushes, %u merged",
               i, cs->stats.calls, cs->stats.cmds, cs->stats.flushes,
               cs->stats.merged);

            assert(cs->stats.cmds > 0 || cs->stats.flushes > 0 ||
                   cs->timestamp.end.handle);

            struct drm_asahi_cmd_compute cmd;
            asahi_fill_cdm_command(dev, cs, &cmd);
            hk_gpu_busy_compute(queue, &cmd);
            util_dynarray_append(&payload, cmd);
            nr_cdm++;
         } else {
            assert(cs->type == HK_CS_VDM);
            perf_debug(cmdbuf, "%u: Submitting VDM with %u API draws, %u draws",
                       i, cs->stats.calls, cs->stats.cmds);
            assert(cs->stats.cmds > 0 || cs->cr.process_empty_tiles ||
                   cs->timestamp.end.handle);

            struct drm_asahi_cmd_render cmd;
            asahi_fill_vdm_command(dev, cs, &cmd);
            if (hk_defer_render_wait(dev, &streams[cs->sync_id]))
               cmd.flags |= DRM_ASAHI_RENDER_VDM_BARRIER_FRAGMENT;
            hk_gpu_busy_render(queue, &cmd);
            util_dynarray_append(&payload, cmd);
            nr_vdm++;
         }
      }
      free(streams);
   }

   /* Output semaphores and the queue timeline cover all preceding work. Each
    * engine present in the final ioctl covers its own older work by subqueue
    * order. Join any absent engine explicitly, including for a single ioctl.
    */
   struct drm_asahi_cmd_header *join = payload.data + last_slice_offset;
   if (nr_vdm == slice_vdm && join->vdm_barrier == DRM_ASAHI_BARRIER_NONE)
      join->vdm_barrier = 0;
   if (nr_cdm == slice_cdm && join->cdm_barrier == DRM_ASAHI_BARRIER_NONE)
      join->cdm_barrier = 0;

   if (dev->dev.debug & AGX_DBG_TRACE) {
      agxdecode_drm_cmdbuf(dev->dev.agxdecode, &dev->dev.params, &payload,
                           true);

      agxdecode_next_frame();
   }

   struct drm_asahi_submit submit_ioctl = {
      .flags = 0,
      .queue_id = queue->drm.id,
      .in_sync_count = wait_count,
      .out_sync_count = submit->signal_count + 1,
      .cmdbuf_size = payload.size,
      .syncs = (uint64_t)(uintptr_t)(syncs),
      .cmdbuf = (uint64_t)(uintptr_t)(payload.data),
   };

   VkResult result;
   if (command_count <= max_commands_per_submit(dev)) {
      result =
         queue_submit_single(dev, &submit_ioctl, queue->drm.virt_ring_idx);
   } else {
      result = queue_submit_looped(dev, &submit_ioctl, command_count,
                                   queue->drm.virt_ring_idx);
   }

   util_dynarray_fini(&payload);
   if (result == VK_SUCCESS) {
      hk_sync_next_submit(&sync);
      queue->sync = sync;
   }
   return result;
}

static VkResult
hk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   struct hk_device *dev = hk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   hk_gpu_busy_sample(queue);
   VkResult result = queue_submit(dev, queue, submit);
   if (result != VK_SUCCESS)
      result = vk_queue_set_lost(&queue->vk, "Submit failed");

   if (dev->dev.debug & AGX_DBG_SYNC) {
      /* Wait for completion */
      U_WATCHDOG_SCOPE(wd, "HK debug queue wait", dev->dev.fd,
                       queue->drm.syncobj, queue->drm.timeline_value, 0,
                       UINT64_MAX, ((struct u_watchdog_objects){0}));
      int err = drmSyncobjTimelineWait(
         dev->dev.fd, &queue->drm.syncobj, &queue->drm.timeline_value, 1,
         INT64_MAX, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL);
      u_watchdog_ioctl("drmSyncobjTimelineWait", dev->dev.fd, err, errno);
      if (!err)
         u_watchdog_complete(dev->dev.fd, queue->drm.syncobj, queue->drm.timeline_value);

      if (err) {
         result = vk_queue_set_lost(&queue->vk, "Wait failed");
      } else {
         VkResult res = dev->vk.check_status(&dev->vk);
         if (result == VK_SUCCESS)
            result = res;
      }
   }

   return result;
}

static enum drm_asahi_priority
translate_priority(VkQueueGlobalPriorityKHR prio)
{
   switch (prio) {
   case VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR:
      return DRM_ASAHI_PRIORITY_REALTIME;

   case VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR:
      return DRM_ASAHI_PRIORITY_HIGH;

   case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR:
      return DRM_ASAHI_PRIORITY_MEDIUM;

   case VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR:
      return DRM_ASAHI_PRIORITY_LOW;

   default:
      UNREACHABLE("Invalid VkQueueGlobalPriorityKHR");
   }
}

VkResult
hk_queue_init(struct hk_device *dev, const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   VkResult result;

   assert(pCreateInfo->queueFamilyIndex < pdev->queue_family_count);

   struct hk_queue *queue = vk_zalloc(&dev->vk.alloc, sizeof(struct hk_queue),
                                      8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   const VkDeviceQueueGlobalPriorityCreateInfoKHR *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR);
   const VkQueueGlobalPriorityKHR priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;

   /* TODO: Lift when kernel side is ready and we can handle failures in
    * create_command_queue.
    */
   enum drm_asahi_priority drm_priority = translate_priority(priority);
   if (drm_priority >= DRM_ASAHI_PRIORITY_HIGH) {
      return VK_ERROR_NOT_PERMITTED_EXT;
   }

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   queue->vk.driver_submit = hk_queue_submit;
   hk_sync_init(&queue->sync);

   /* Under virtio, the host's native context gives each kernel queue the
    * fence timeline of its priority (ring = priority + 1) and retires a ring's
    * guest fences when the latest submission on that timeline completes, while
    * the guest kernel skips waits on fences from the submitting ring. Two
    * queues of equal priority on independent kernel queues would share a
    * timeline: a fence could retire on the other queue's work, and waits
    * between them would be dropped. Give the second family its own priority
    * bucket, and so its own timeline and ring. */
   if (dev->dev.is_virtio && pCreateInfo->queueFamilyIndex != 0) {
      drm_priority = drm_priority == DRM_ASAHI_PRIORITY_LOW
                        ? DRM_ASAHI_PRIORITY_MEDIUM
                        : DRM_ASAHI_PRIORITY_LOW;
   }

   queue->drm.id = agx_create_command_queue(&dev->dev, drm_priority);
   queue->drm.virt_ring_idx = drm_priority + 1;

   if (drmSyncobjCreate(dev->dev.fd, 0, &queue->drm.syncobj)) {
      mesa_loge("drmSyncobjCreate() failed %d\n", errno);
      agx_destroy_command_queue(&dev->dev, queue->drm.id);
      vk_queue_finish(&queue->vk);

      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "DRM_IOCTL_SYNCOBJ_CREATE failed: %m");
   }

   uint64_t initial_value = 1;
   if (drmSyncobjTimelineSignal(dev->dev.fd, &queue->drm.syncobj,
                                &initial_value, 1)) {
      hk_queue_finish(dev, queue);
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "DRM_IOCTL_TIMELINE_SYNCOBJ_SIGNAL failed: %m");
   }

   hk_gpu_busy_init(queue);
   return VK_SUCCESS;
}

void
hk_queue_finish(struct hk_device *dev, struct hk_queue *queue)
{
   hk_gpu_busy_finish(queue);
   u_watchdog_forget_queue(dev->dev.fd, queue->drm.id);
   drmSyncobjDestroy(dev->dev.fd, queue->drm.syncobj);
   agx_destroy_command_queue(&dev->dev, queue->drm.id);
   vk_queue_finish(&queue->vk);
   vk_free(&dev->vk.alloc, queue);
}
