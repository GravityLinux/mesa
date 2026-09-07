/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * Copyright 2020 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include "agx_device.h"
#include <inttypes.h>
#include "clc/asahi_clc.h"
#include "drm-uapi/asahi_drm.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/timespec.h"
#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_device_virtio.h"
#include "agx_scratch.h"
#include "decode.h"
#include "glsl_types.h"
#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>
#include "drm-uapi/dma-buf.h"
#include "util/blob.h"
#include "util/log.h"
#include "util/mesa-blake3.h"
#include "util/os_file.h"
#include "util/os_mman.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_math.h"
#include "util/u_printf.h"
#include "git_sha1.h"
#include "nir_serialize.h"
#include "vdrm.h"

static inline int
asahi_simple_ioctl(struct agx_device *dev, unsigned cmd, void *req)
{
   if (dev->is_virtio) {
      return agx_virtio_simple_ioctl(dev, cmd, req);
   } else {
      return drmIoctl(dev->fd, cmd, req);
   }
}

#define AGX_APPLE9_COMPUTE_STATE_SLAB_SIZE 0x4000u
#define AGX_APPLE9_COMPUTE_STATE_RECORD_SIZE 0x40u
#define AGX_APPLE9_COMPUTE_STATE_SELECTOR_OFFSET 0x20u

static_assert((AGX_APPLE9_COMPUTE_STATE_SLAB_SIZE %
               AGX_APPLE9_COMPUTE_STATE_RECORD_SIZE) == 0,
              "Apple9 state slabs contain whole records");
static_assert((AGX_APPLE9_COMPUTE_STATE_SLAB_SIZE /
               AGX_APPLE9_COMPUTE_STATE_RECORD_SIZE) == 256,
              "Apple9 state slabs contain 256 records");

/* The render carrier contains executable pages and GPU-writable resource
 * records in the same fixed USC window. Reserve the whole window so ordinary
 * USC allocations cannot occupy addresses still referenced by its helpers. */
static int
agx_apple9_bind_fixed_usc(struct agx_device *dev, struct agx_bo *bo)
{
   if (bo != dev->apple9_render_fixed_usc)
      return agx_bo_bind(dev, bo, dev->shader_base,
                         AGX_APPLE9_FIXED_USC_ARENA_SIZE, 0,
                         DRM_ASAHI_BIND_READ);

   static const struct {
      uint32_t start, end;
   } writable[] = {
      {0x018000, 0x038000}, {0x058000, 0x078000}, {0x080000, 0x184000},
      {0x220000, 0x228000}, {0x230000, 0x340000}, {0x348000, 0x350000},
   };
   uint32_t cursor = 0;
   for (unsigned i = 0; i <= ARRAY_SIZE(writable); ++i) {
      uint32_t end = i < ARRAY_SIZE(writable) ? writable[i].start
                                              : AGX_APPLE9_FIXED_USC_ARENA_SIZE;
      int ret = agx_bo_bind(dev, bo, dev->shader_base + cursor, end - cursor,
                            cursor, DRM_ASAHI_BIND_READ);
      if (ret)
         return ret;
      if (i == ARRAY_SIZE(writable))
         break;
      ret = agx_bo_bind(dev, bo, dev->shader_base + end, writable[i].end - end,
                        end, DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
      if (ret)
         return ret;
      cursor = writable[i].end;
   }
   return 0;
}

static bool
agx_apple9_switch_fixed_usc_locked(struct agx_device *dev,
                                   struct agx_bo *next)
{
   if (!next || next->size < AGX_APPLE9_FIXED_USC_ARENA_SIZE)
      return false;

   if (dev->apple9_fixed_usc_owner == next)
      return true;

   struct agx_bo *previous = dev->apple9_fixed_usc_owner;
   if (agx_bo_bind(dev, NULL, dev->shader_base,
                   AGX_APPLE9_FIXED_USC_ARENA_SIZE, 0,
                   DRM_ASAHI_BIND_UNBIND))
      return false;
   dev->apple9_fixed_usc_owner = NULL;

   if (agx_apple9_bind_fixed_usc(dev, next)) {
      /* Remove any partially installed view before restoring its predecessor. */
      if (!agx_bo_bind(dev, NULL, dev->shader_base,
                       AGX_APPLE9_FIXED_USC_ARENA_SIZE, 0,
                       DRM_ASAHI_BIND_UNBIND) &&
          previous && !agx_apple9_bind_fixed_usc(dev, previous))
         dev->apple9_fixed_usc_owner = previous;
      return false;
   }

   dev->apple9_fixed_usc_owner = next;
   return true;
}

bool
agx_apple9_install_compute_archive(struct agx_device *dev)
{
   if (!dev || !dev->apple9_compute_archive ||
       !dev->apple9_compute_archive_shadow)
      return false;

   simple_mtx_lock(&dev->apple9_archive_lock);
   bool populated = dev->apple9_archive_next != 0;
   if (populated &&
       dev->apple9_compute_archive_installed_generation !=
          dev->apple9_compute_archive_generation) {
      memcpy(agx_bo_map(dev->apple9_compute_archive),
             dev->apple9_compute_archive_shadow,
             AGX_APPLE9_COMPUTE_ARCHIVE_SIZE);
      dev->apple9_compute_archive_installed_generation =
         dev->apple9_compute_archive_generation;
   }
   bool installed = populated && agx_apple9_switch_fixed_usc_locked(
      dev, dev->apple9_compute_archive);
   simple_mtx_unlock(&dev->apple9_archive_lock);
   return installed;
}

bool
agx_apple9_install_render_archive(struct agx_device *dev)
{
   if (!dev || !dev->apple9_render_fixed_usc)
      return false;

   simple_mtx_lock(&dev->apple9_archive_lock);
   bool installed = agx_apple9_switch_fixed_usc_locked(
      dev, dev->apple9_render_fixed_usc);
   simple_mtx_unlock(&dev->apple9_archive_lock);
   return installed;
}

bool
agx_apple9_alloc_compute_state(struct agx_device *dev, struct agx_bo **bo,
                               void **record, uint64_t *selector)
{
   if (!dev || !bo || !record || !selector ||
       (dev->chip != AGX_CHIP_G16G && dev->chip != AGX_CHIP_G17P))
      return false;

   simple_mtx_lock(&dev->apple9_archive_lock);

   struct agx_bo *slab = dev->apple9_compute_state_current;
   if (!slab ||
       dev->apple9_compute_state_next >
          AGX_APPLE9_COMPUTE_STATE_SLAB_SIZE -
             AGX_APPLE9_COMPUTE_STATE_RECORD_SIZE) {
      slab = agx_bo_create(dev, AGX_APPLE9_COMPUTE_STATE_SLAB_SIZE,
                           AGX_APPLE9_COMPUTE_STATE_SLAB_SIZE,
                           AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                           "Apple9 compute state slab");
      if (!slab) {
         simple_mtx_unlock(&dev->apple9_archive_lock);
         return false;
      }

      memset(agx_bo_map(slab), 0, slab->size);
      util_dynarray_append(&dev->apple9_compute_state_bos, slab);
      dev->apple9_compute_state_current = slab;
      dev->apple9_compute_state_next = 0;
   }

   uint32_t offset = dev->apple9_compute_state_next;
   dev->apple9_compute_state_next += AGX_APPLE9_COMPUTE_STATE_RECORD_SIZE;

   /* The device owns the allocation reference; give the shader its own. */
   agx_bo_reference(slab);
   *bo = slab;
   *record = (uint8_t *)agx_bo_map(slab) + offset;
   *selector = slab->va->addr + offset +
               AGX_APPLE9_COMPUTE_STATE_SELECTOR_OFFSET;

   simple_mtx_unlock(&dev->apple9_archive_lock);
   return true;
}

/* clang-format off */
static const struct debug_named_value agx_debug_options[] = {
   {"trace",     AGX_DBG_TRACE,    "Trace the command stream"},
   {"bodump",    AGX_DBG_BODUMP,   "Periodically dump live BOs"},
   {"no16",      AGX_DBG_NO16,     "Disable 16-bit support"},
   {"perf",      AGX_DBG_PERF,     "Print performance warnings"},
#ifndef NDEBUG
   {"dirty",     AGX_DBG_DIRTY,    "Disable dirty tracking"},
#endif
   {"precompile",AGX_DBG_PRECOMPILE,"Precompile shaders for shader-db"},
   {"nocompress",AGX_DBG_NOCOMPRESS,"Disable lossless compression"},
   {"nocluster", AGX_DBG_NOCLUSTER,"Disable vertex clustering"},
   {"sync",      AGX_DBG_SYNC,     "Synchronously wait for all submissions"},
   {"stats",     AGX_DBG_STATS,    "Show command execution statistics"},
   {"resource",  AGX_DBG_RESOURCE, "Log resource operations"},
   {"batch",     AGX_DBG_BATCH,    "Log batches"},
   {"nowc",      AGX_DBG_NOWC,     "Disable write-combining"},
   {"synctvb",   AGX_DBG_SYNCTVB,  "Synchronous TVB growth"},
   {"smalltile", AGX_DBG_SMALLTILE,"Force 16x16 tiles"},
   {"feedback",  AGX_DBG_FEEDBACK, "Debug feedback loops"},
   {"nomsaa",    AGX_DBG_NOMSAA,   "Force disable MSAA"},
   {"noshadow",  AGX_DBG_NOSHADOW, "Force disable resource shadowing"},
   {"scratch",   AGX_DBG_SCRATCH,  "Debug scratch memory usage"},
   {"1queue",    AGX_DBG_1QUEUE,   "Force usage of a single queue for multiple contexts"},
   {"nosoft",    AGX_DBG_NOSOFT,   "Disable soft fault optimizations"},
   {"nomerge",   AGX_DBG_NOMERGE,  "Disable control stream merging"},
   {"bodumpverbose", AGX_DBG_BODUMPVERBOSE,   "Include extra info with dumps"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

void
agx_bo_free(struct agx_device *dev, struct agx_bo *bo)
{
   const uint64_t handle = bo->handle;

   if (bo->_map)
      munmap(bo->_map, bo->size);

   /* Free the VA. No need to unmap the BO or unbind the VA, as the kernel will
    * take care of that when we close it.
    */
   agx_va_free(dev, bo->va, false);

   if (bo->prime_fd != -1)
      close(bo->prime_fd);

   /* Reset the handle. This has to happen before the GEM close to avoid a race.
    */
   memset(bo, 0, sizeof(*bo));
   __sync_synchronize();

   struct drm_gem_close args = {.handle = handle};
   drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &args);
}

static int
agx_drm_bo_bind(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
                uint32_t count)
{
   struct drm_asahi_vm_bind vm_bind = {
      .num_binds = count,
      .vm_id = dev->vm_id,
      .userptr = (uintptr_t)ops,
      .stride = sizeof(*ops),
   };

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_VM_BIND, &vm_bind);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_VM_BIND failed\n");
   }

   return ret;
}

/*
 * Convenience helper to bind a single BO regardless of kernel module.
 */
int
agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t addr,
            size_t size_B, uint64_t offset_B, uint32_t flags)
{
   assert((size_B % 16384) == 0 && "alignment required");
   assert((offset_B % 16384) == 0 && "alignment required");
   assert((addr % 16384) == 0 && "alignment required");

   struct drm_asahi_gem_bind_op op = {
      .flags = flags,
      .handle = bo ? bo->uapi_handle : 0,
      .offset = offset_B,
      .range = size_B,
      .addr = addr,
   };

   return dev->ops.bo_bind(dev, &op, 1);
}

int
agx_bind_timestamps(struct agx_device *dev, struct agx_bo *bo, uint32_t *handle)
{
   struct drm_asahi_gem_bind_object bind = {
      .op = DRM_ASAHI_BIND_OBJECT_OP_BIND,
      .flags = DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS,
      .handle = bo->uapi_handle,
      .range = bo->size,
   };

   int ret = dev->ops.bo_bind_object(dev, &bind);
   *handle = bind.object_handle;
   return ret;
}

static struct agx_bo *
agx_bo_alloc(struct agx_device *dev, size_t size, size_t align,
             enum agx_bo_flags flags)
{
   struct agx_bo *bo;
   unsigned handle = 0;

   /*
    * AGX_BO_EXEC describes contents, not addressability.  Compact Apple8/9
    * users request LOW_VA explicitly. Apple9 immutable source packages may
    * use arbitrary storage while one separate resident archive stays bound in
    * the compact fixed-base aperture.
    */

   struct drm_asahi_gem_create gem_create = {.size = size};

   if (flags & AGX_BO_WRITEBACK)
      gem_create.flags |= DRM_ASAHI_GEM_WRITEBACK;

   if (!(flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE))) {
      gem_create.flags |= DRM_ASAHI_GEM_VM_PRIVATE;
      gem_create.vm_id = dev->vm_id;
   }

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_CREATE, &gem_create);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GEM_CREATE failed: %m\n");
      return NULL;
   }

   handle = gem_create.handle;

   pthread_mutex_lock(&dev->bo_map_lock);
   bo = agx_lookup_bo(dev, handle);
   dev->max_handle = MAX2(dev->max_handle, handle);
   pthread_mutex_unlock(&dev->bo_map_lock);

   /* Fresh handle */
   assert(!memcmp(bo, &((struct agx_bo){}), sizeof(*bo)));

   bo->dev = dev;
   bo->size = gem_create.size;
   bo->align = align;
   bo->flags = flags;
   bo->handle = bo->uapi_handle = handle;
   bo->prime_fd = -1;

   enum agx_va_flags va_flags = flags & AGX_BO_LOW_VA ? AGX_VA_USC : 0;
   bo->va = agx_va_alloc(dev, size, bo->align, va_flags, 0);
   if (!bo->va) {
      fprintf(stderr, "Failed to allocate BO VMA\n");
      agx_bo_free(dev, bo);
      return NULL;
   }

   uint32_t bind = DRM_ASAHI_BIND_READ;
   if (!(flags & AGX_BO_READONLY)) {
      bind |= DRM_ASAHI_BIND_WRITE;
   }

   ret = agx_bo_bind(dev, bo, bo->va->addr, bo->size, 0, bind);
   if (ret) {
      agx_bo_free(dev, bo);
      return NULL;
   }

   return bo;
}

static void
agx_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr)
{
   assert(bo->_map == NULL && "not double mapped");

   struct drm_asahi_gem_mmap_offset gem_mmap_offset = {.handle =
                                                          bo->uapi_handle};
   int ret, flags;

   ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &gem_mmap_offset);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_MMAP_BO failed: %m\n");
      assert(0);
   }

   flags = MAP_SHARED | (fixed_addr ? MAP_FIXED : 0);
   bo->_map = os_mmap(fixed_addr, bo->size, PROT_READ | PROT_WRITE, flags,
                      dev->fd, gem_mmap_offset.offset);
   if (bo->_map == MAP_FAILED) {
      bo->_map = NULL;
      fprintf(stderr,
              "mmap failed: result=%p size=0x%llx fd=%i offset=0x%llx %m\n",
              bo->_map, (long long)bo->size, dev->fd,
              (long long)gem_mmap_offset.offset);
   }
}

struct agx_bo *
agx_bo_import(struct agx_device *dev, int fd)
{
   struct agx_bo *bo;
   ASSERTED int ret;
   unsigned gem_handle;

   pthread_mutex_lock(&dev->bo_map_lock);

   ret = drmPrimeFDToHandle(dev->fd, fd, &gem_handle);
   if (ret) {
      fprintf(stderr, "import failed: Could not map fd %d to handle\n", fd);
      pthread_mutex_unlock(&dev->bo_map_lock);
      return NULL;
   }

   bo = agx_lookup_bo(dev, gem_handle);
   dev->max_handle = MAX2(dev->max_handle, gem_handle);

   if (!bo->size) {
      bo->dev = dev;
      bo->size = lseek(fd, 0, SEEK_END);
      bo->align = AIL_PAGESIZE;

      /* Sometimes this can fail and return -1. size of -1 is not
       * a nice thing for mmap to try mmap. Be more robust also
       * for zero sized maps and fail nicely too
       */
      if ((bo->size == 0) || (bo->size == (size_t)-1)) {
         goto error;
      }
      if (bo->size & (AIL_PAGESIZE - 1)) {
         fprintf(
            stderr,
            "import failed: BO is not a multiple of the page size (0x%llx bytes)\n",
            (long long)bo->size);
         goto error;
      }

      bo->flags = AGX_BO_SHARED | AGX_BO_SHAREABLE;
      bo->handle = gem_handle;
      bo->prime_fd = os_dupfd_cloexec(fd);
      bo->label = "Imported BO";
      assert(bo->prime_fd >= 0);

      p_atomic_set(&bo->refcnt, 1);
      bo->va = agx_va_alloc(dev, bo->size, bo->align, 0, 0);

      if (!bo->va) {
         fprintf(
            stderr,
            "import failed: Could not allocate from VMA heap (0x%llx bytes)\n",
            (long long)bo->size);
         abort();
      }

      if (dev->is_virtio) {
         bo->uapi_handle = vdrm_handle_to_res_id(dev->vdrm, bo->handle);
      } else {
         bo->uapi_handle = bo->handle;
      }

      ret = agx_bo_bind(dev, bo, bo->va->addr, bo->size, 0,
                        DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
      if (ret) {
         fprintf(stderr, "import failed: Could not bind BO at 0x%llx\n",
                 (long long)bo->va->addr);
         abort();
      }
   } else {
      /* bo->refcnt == 0 can happen if the BO
       * was being released but agx_bo_import() acquired the
       * lock before agx_bo_unreference(). In that case, refcnt
       * is 0 and we can't use agx_bo_reference() directly, we
       * have to re-initialize the refcnt().
       * Note that agx_bo_unreference() checks
       * refcnt value just after acquiring the lock to
       * make sure the object is not freed if agx_bo_import()
       * acquired it in the meantime.
       */
      if (p_atomic_read(&bo->refcnt) == 0)
         p_atomic_set(&bo->refcnt, 1);
      else
         agx_bo_reference(bo);

      /* If this bo came back to us via import, it had better
       * been marked shared to begin with.
       */
      if (bo->shim_cpu_epoch)
      *bo->shim_cpu_epoch = 0;
   assert(bo->flags & AGX_BO_SHAREABLE);
      assert(bo->flags & AGX_BO_SHARED);
      assert(bo->prime_fd != -1);
   }
   pthread_mutex_unlock(&dev->bo_map_lock);

   assert(bo->dev != NULL && "post-condition");

   if (dev->debug & AGX_DBG_TRACE) {
      agx_bo_map(bo);
      agxdecode_track_alloc(dev->agxdecode, bo);
   }

   return bo;

error:
   memset(bo, 0, sizeof(*bo));
   pthread_mutex_unlock(&dev->bo_map_lock);
   return NULL;
}

void
agx_bo_make_shared(struct agx_device *dev, struct agx_bo *bo)
{
   assert(bo->flags & AGX_BO_SHAREABLE);
   if (bo->flags & AGX_BO_SHARED) {
      assert(bo->prime_fd >= 0);
      return;
   }

   bo->flags |= AGX_BO_SHARED;
   assert(bo->prime_fd == -1);

   int ret = drmPrimeHandleToFD(dev->fd, bo->handle, DRM_CLOEXEC | DRM_RDWR,
                                &bo->prime_fd);
   assert(ret == 0);
   assert(bo->prime_fd >= 0);

   /* If there is a pending writer to this BO, import it into the buffer
    * for implicit sync.
    */
   uint64_t writer = p_atomic_read_relaxed(&bo->writer);
   if (writer) {
      int out_sync_fd = -1;
      int ret = drmSyncobjExportSyncFile(dev->fd, agx_bo_writer_syncobj(writer),
                                         &out_sync_fd);
      assert(ret >= 0);
      assert(out_sync_fd >= 0);

      ret = agx_import_sync_file(dev, bo, out_sync_fd);
      assert(ret >= 0);
      close(out_sync_fd);
   }
}

int
agx_bo_export(struct agx_device *dev, struct agx_bo *bo)
{
   agx_bo_make_shared(dev, bo);

   assert(bo->prime_fd >= 0);
   return os_dupfd_cloexec(bo->prime_fd);
}

static int
agx_bo_bind_object(struct agx_device *dev,
                   struct drm_asahi_gem_bind_object *bind)
{
   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, bind);
   if (ret) {
      fprintf(stderr,
              "DRM_IOCTL_ASAHI_GEM_BIND_OBJECT failed: %m (handle=%d)\n",
              bind->handle);
   }

   return ret;
}

static int
agx_bo_unbind_object(struct agx_device *dev, uint32_t object_handle)
{
   struct drm_asahi_gem_bind_object gem_bind = {
      .op = DRM_ASAHI_BIND_OBJECT_OP_UNBIND,
      .object_handle = object_handle,
   };

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, &gem_bind);
   if (ret) {
      fprintf(stderr,
              "DRM_IOCTL_ASAHI_GEM_BIND_OBJECT failed: %m (object_handle=%d)\n",
              object_handle);
   }

   return ret;
}

static ssize_t
agx_get_params(struct agx_device *dev, void *buf, size_t size)
{
   struct drm_asahi_get_params get_param = {
      .param_group = 0,
      .pointer = (uint64_t)(uintptr_t)buf,
      .size = size,
   };

   memset(buf, 0, size);

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GET_PARAMS, &get_param);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GET_PARAMS failed: %m\n");
      return -EINVAL;
   }

   return get_param.size;
}

static int
agx_submit(struct agx_device *dev, struct drm_asahi_submit *submit,
           struct agx_submit_virt *virt)
{
   return drmIoctl(dev->fd, DRM_IOCTL_ASAHI_SUBMIT, submit);
}

const agx_device_ops_t agx_device_drm_ops = {
   .bo_alloc = agx_bo_alloc,
   .bo_bind = agx_drm_bo_bind,
   .bo_mmap = agx_bo_mmap,
   .get_params = agx_get_params,
   .submit = agx_submit,
   .bo_bind_object = agx_bo_bind_object,
   .bo_unbind_object = agx_bo_unbind_object,
};

static uint64_t
gcd(uint64_t n, uint64_t m)
{
   while (n != 0) {
      uint64_t remainder = m % n;
      m = n;
      n = remainder;
   }

   return m;
}

static void
agx_init_timestamps(struct agx_device *dev)
{
   uint64_t user_ts_gcd =
      gcd(dev->params.command_timestamp_frequency_hz, NSEC_PER_SEC);

   dev->user_timestamp_to_ns.num = NSEC_PER_SEC / user_ts_gcd;
   dev->user_timestamp_to_ns.den =
      dev->params.command_timestamp_frequency_hz / user_ts_gcd;
}

bool
agx_open_device(void *memctx, struct agx_device *dev)
{
   dev->debug =
      debug_get_flags_option("ASAHI_MESA_DEBUG", agx_debug_options, 0);

   dev->ops = agx_device_drm_ops;

   ssize_t params_size = -1;

   /* DRM version check */
   {
      drmVersionPtr version = drmGetVersion(dev->fd);
      if (!version) {
         fprintf(stderr, "cannot get version: %s", strerror(errno));
         return NULL;
      }

      if (!strcmp(version->name, "asahi")) {
         dev->is_virtio = false;
         dev->ops = agx_device_drm_ops;
      } else if (!strcmp(version->name, "virtio_gpu")) {
         dev->is_virtio = true;
         if (!agx_virtio_open_device(dev)) {
            fprintf(
               stderr,
               "Error opening virtio-gpu device for Asahi native context\n");
            return false;
         }
      } else {
         return false;
      }

      drmFreeVersion(version);
   }

   params_size = dev->ops.get_params(dev, &dev->params, sizeof(dev->params));
   if (params_size <= 0) {
      assert(0);
      return false;
   }
   assert(params_size >= sizeof(dev->params));

   assert(dev->params.gpu_generation >= 13);
   const char *variant = " Unknown";
   switch (dev->params.gpu_variant) {
   case 'G':
      variant = "";
      break;
   case 'S':
      variant = " Pro";
      break;
   case 'P':
      /* Apple9 firmware reports the Pro variant as P. */
      variant = " Pro";
      break;
   case 'C':
      variant = " Max";
      break;
   case 'D':
      variant = " Ultra";
      break;
   }
   snprintf(dev->name, sizeof(dev->name), "Apple M%d%s (G%d%c %02X)",
            dev->params.gpu_generation - 12, variant,
            dev->params.gpu_generation, dev->params.gpu_variant,
            dev->params.gpu_revision + 0xA0);

   /* We need a large chunk of VA space carved out for robustness. Hardware
    * loads can shift an i32 by up to 2, for a total shift of 4. If the base
    * address is zero, 36-bits is therefore enough to trap any zero-extended
    * 32-bit index. For more generality we would need a larger carveout, but
    * this is already optimal for VBOs.
    *
    * TODO: Maybe this should be on top instead? Might be ok.
    */
   uint64_t reservation = (1ull << 36);

   /* Also reserve VA space for the printf buffer at a stable address, avoiding
    * the need for relocs in precompiled shaders.
    */
   assert(reservation == LIBAGX_PRINTF_BUFFER_ADDRESS);
   reservation += LIBAGX_PRINTF_BUFFER_SIZE;

   dev->guard_size = AIL_PAGESIZE;
   uint64_t default_shader_base =
      ALIGN_POT(MAX2(dev->params.vm_start, reservation), 0x100000000ull);
   bool fixed_apple9_usc_base = dev->params.gpu_generation == 16 ||
                                dev->params.gpu_generation == 17;

   /* Apple9 compact shader references resolve through the fixed native 1-TiB
    * USC aperture on both G16 and G17.  A relocatable G17 base was an early
    * shim assumption, not a hardware capability.  The 42-bit UAT leaves room
    * for the complete 4-GiB heap there, so use the hardware address directly
    * instead of requiring a kernel-side alias.
    */
   if (fixed_apple9_usc_base) {
      dev->shader_base = 1ull << 40;
   } else {
      // Put the USC heap at the bottom of the user address space, 4GiB aligned
      dev->shader_base = default_shader_base;
   }

   if (dev->shader_base < reservation) {
      /* Our robustness implementation requires the bottom unmapped */
      fprintf(stderr, "Unexpected address layout, can't cope\n");
      assert(0);
      return false;
   }

   uint64_t shader_size = 0x100000000ull;
   /* Apple9 leaves the ordinary resource/descriptor heap below the fixed USC
    * aperture. Besides avoiding a wasted 1-TiB hole, this temporarily keeps
    * unported generic pipeline records inside their legacy 39-bit pointer
    * field. Native Apple9 sampling uses full 64-bit Tier-2 resource pointers,
    * so this is a Mesa compatibility guard, not a hardware address limit.
    * Executable/AGX_BO_LOW_VA allocations still use usc_heap.
    */
   uint64_t user_start = fixed_apple9_usc_base
                            ? default_shader_base
                            : dev->shader_base + shader_size;

   assert(dev->shader_base >= dev->params.vm_start);
   assert(dev->shader_base + shader_size < dev->params.vm_end);
   assert(user_start < dev->params.vm_end);

   dev->agxdecode = agxdecode_new_context(dev->shader_base);

   agx_init_timestamps(dev);

   util_sparse_array_init(&dev->bo_map, sizeof(struct agx_bo), 512);
   pthread_mutex_init(&dev->bo_map_lock, NULL);

   simple_mtx_init(&dev->bo_cache.lock, mtx_plain);
   list_inithead(&dev->bo_cache.lru);

   for (unsigned i = 0; i < ARRAY_SIZE(dev->bo_cache.buckets); ++i)
      list_inithead(&dev->bo_cache.buckets[i]);

   /* The generic ABI puts a generous kernel-private heap at the top of the
    * VM.  G16 render Work instead addresses that backend-owned state relative
    * to the fixed 0x1000000000 context and has a 4-GiB reach.  Keep its UAPI
    * carveout after the small fixed client graph and before the ordinary Mesa
    * heap, with exactly the advertised minimum capacity. */
   uint64_t kernel_size = MAX2(dev->params.vm_kernel_min_size, 32ull << 30);
   struct drm_asahi_vm_create vm_create;
   if (dev->params.gpu_generation == 16 &&
       dev->params.gpu_variant == 'G') {
      kernel_size = dev->params.vm_kernel_min_size;
      vm_create.kernel_end =
         AGX_APPLE9_FIXED_RENDER_CONTEXT_BASE + (1ull << 32);
      vm_create.kernel_start = vm_create.kernel_end - kernel_size;
      if (kernel_size < 0x4000000ull ||
          vm_create.kernel_end >
             AGX_APPLE9_FIXED_RENDER_CONTEXT_BASE + (1ull << 32)) {
         fprintf(stderr,
                 "G16 kernel-private heap does not fit its compact render "
                 "context\n");
         return false;
      }
   } else {
      vm_create.kernel_start = dev->params.vm_end - kernel_size;
      vm_create.kernel_end = dev->params.vm_end;
   }

   /* Keep Apple9's ordinary heap and its sparse read-only mirror below the
    * legacy generic-pipeline ceiling until Apple9 Tier-2 resource binding is
    * used throughout. The separate USC heap remains at 1 TiB and is not part
    * of this partition.
    */
   uint64_t user_end =
      fixed_apple9_usc_base ? (1ull << 39) : vm_create.kernel_start;
   assert(user_start < user_end);
   uint64_t user_size = user_end - user_start;

   int ret = asahi_simple_ioctl(dev, DRM_IOCTL_ASAHI_VM_CREATE, &vm_create);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_VM_CREATE failed: %m\n");
      assert(0);
      return false;
   }

   /* Round the user VA window to powers-of-two... */
   user_start = util_next_power_of_two64(user_start);
   user_size = util_next_power_of_two64(user_size + 1) >> 1;

   /* ...so when we cut user size in half to emulate sparse buffers... */
   user_size /= 2;

   /* ...or maybe in quarters if necessary to disambiguate */
   if (user_size == user_start) {
      user_size /= 2;
   }

   /* ...we can distinguish the top/bottom half by an address bit */
   dev->sparse_ro_offset = user_size;
   assert((user_start & dev->sparse_ro_offset) == 0);
   assert(((user_start + (user_size - 1)) & dev->sparse_ro_offset) == 0);

   simple_mtx_init(&dev->vma_lock, mtx_plain);
   util_vma_heap_init(&dev->main_heap, user_start, user_size);
   util_vma_heap_init(&dev->usc_heap, dev->shader_base, shader_size);

   /*
    * Apple9 launch wrappers encode their address as a compact 8-KiB chunk
    * relative to the queue's USC base.  Allocate the queue archive and batch
    * packages from the bottom of the USC heap so they remain in that compact
    * window.  Older generations use 32-bit USC offsets and retain Mesa's
    * usual high-to-low allocation policy.
    */
   if (dev->params.gpu_generation >= 16)
      dev->usc_heap.alloc_high = false;

   dev->vm_id = vm_create.vm_id;

   glsl_type_singleton_init_or_ref();

   if (agx_gather_device_key(dev).needs_g13x_coherency == U_TRISTATE_YES) {
      dev->libagx_programs = libagx_g13x;
   } else {
      dev->libagx_programs = libagx_g13g;
   }

   switch (dev->params.gpu_generation) {
   case 13:
      dev->chip = dev->params.num_clusters_total > 1 ? AGX_CHIP_G13X
                                                     : AGX_CHIP_G13G;
      break;
   case 14:
   case 15:
      dev->chip = dev->params.num_clusters_total > 1 ? AGX_CHIP_G14X
                                                     : AGX_CHIP_G14G;
      break;
   case 16:
      assert(dev->params.gpu_variant == 'G');
      dev->chip = AGX_CHIP_G16G;
      break;
   case 17:
      assert(dev->params.gpu_variant == 'P');
      dev->chip = AGX_CHIP_G17P;
      break;
   default:
      UNREACHABLE("Unsupported AGX generation");
   }

   if (dev->chip == AGX_CHIP_G16G || dev->chip == AGX_CHIP_G17P) {
      simple_mtx_init(&dev->apple9_archive_lock, mtx_plain);
      util_dynarray_init(&dev->apple9_compute_state_bos, NULL);
      dev->apple9_compute_state_current = NULL;
      dev->apple9_compute_state_next = 0;
      /* The CPU authors this archive through its writable mmap, but the GPU
       * only executes/reads it.  Native T8132 uses the read-only fixed-USC PTE
       * class.  Hardware isolation also shows why the distinction matters:
       * appending a main after earlier execution is exact with READ-only
       * leaves, while a GPU-writable mapping can fetch stale zeroes from the
       * same bytes and address.
       */
      dev->apple9_compute_archive =
         agx_bo_create(dev, AGX_APPLE9_FIXED_USC_ARENA_SIZE,
                       AGX_APPLE9_FIXED_USC_ARENA_SIZE,
                       AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK |
                          AGX_BO_READONLY,
                       "Apple9 fixed USC arena");
      if (dev->apple9_compute_archive &&
          dev->apple9_compute_archive->va->addr != dev->shader_base) {
         struct agx_bo *archive = dev->apple9_compute_archive;
         agx_va_free(dev, archive->va, true);
         archive->va = agx_va_alloc(
            dev, archive->size, archive->align,
            AGX_VA_USC | AGX_VA_FIXED, dev->shader_base);
         if (archive->va) {
            int bind = agx_bo_bind(
               dev, archive, archive->va->addr, archive->size, 0,
               DRM_ASAHI_BIND_READ);
            if (bind) {
               agx_va_free(dev, archive->va, true);
               archive->va = NULL;
            }
         }
      }

      if (!dev->apple9_compute_archive ||
          !dev->apple9_compute_archive->va ||
          dev->apple9_compute_archive->va->addr != dev->shader_base) {
         fprintf(stderr,
                 "Failed to reserve Apple9 compute archive at USC base\n");
         return false;
      }

      memset(agx_bo_map(dev->apple9_compute_archive), 0,
             dev->apple9_compute_archive->size);
      dev->apple9_compute_archive_shadow =
         calloc(1, AGX_APPLE9_COMPUTE_ARCHIVE_SIZE);
      if (!dev->apple9_compute_archive_shadow) {
         fprintf(stderr,
                 "Failed to allocate Apple9 compute archive shadow\n");
         return false;
      }
      dev->apple9_compute_archive_generation = 0;
      dev->apple9_compute_archive_installed_generation = 0;
      dev->apple9_fixed_usc_owner = dev->apple9_compute_archive;

      dev->apple9_render_fixed_usc =
         agx_bo_create(dev, AGX_APPLE9_FIXED_USC_ARENA_SIZE,
                       AGX_APPLE9_FIXED_USC_ARENA_SIZE,
                       AGX_BO_EXEC | AGX_BO_WRITEBACK | AGX_BO_READONLY,
                       "Apple9 render fixed USC arena");
      if (!dev->apple9_render_fixed_usc) {
         fprintf(stderr,
                 "Failed to allocate Apple9 render fixed-USC arena\n");
         return false;
      }
      memset(agx_bo_map(dev->apple9_render_fixed_usc), 0,
             dev->apple9_render_fixed_usc->size);

      dev->apple9_render_context =
         agx_bo_create(dev, AGX_APPLE9_FIXED_RENDER_CONTEXT_SIZE, 0x4000,
                       AGX_BO_WRITEBACK,
                       "Apple9 fixed render context");
      int render_context_bind = dev->apple9_render_context
         ? agx_bo_bind(dev, dev->apple9_render_context,
                       AGX_APPLE9_FIXED_RENDER_CONTEXT_BASE,
                       AGX_APPLE9_FIXED_RENDER_CONTEXT_SIZE, 0,
                       DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE)
         : -1;
      if (!dev->apple9_render_context || render_context_bind) {
         fprintf(stderr,
                 "Failed to reserve Apple9 fixed render context\n");
         return false;
      }
      memset(agx_bo_map(dev->apple9_render_context), 0,
             dev->apple9_render_context->size);
      dev->apple9_archive_next = 0;
      dev->apple9_archive_entry_count = 0;
   }

   /* Bind read-only zero page at 2^32. This is in our reservation, and can be
    * addressed with only small integers in the low/high. That lets us do some
    * robustness optimization even without soft fault.
    */
   {
      void *bo = agx_bo_create(dev, 16384, 0, 0, "Zero page");
      int ret = agx_bo_bind(dev, bo, AGX_ZERO_PAGE_ADDRESS, 16384, 0,
                            DRM_ASAHI_BIND_READ);
      if (ret) {
         fprintf(stderr, "Failed to bind zero page");
         return false;
      }

      dev->zero_bo = bo;
   }

   {
      void *bo = agx_bo_create(dev, AIL_PAGESIZE, 0, 0, "Scratch page");
      int ret = agx_bo_bind(dev, bo, AGX_SCRATCH_PAGE_ADDRESS, AIL_PAGESIZE, 0,
                            DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
      if (ret) {
         fprintf(stderr, "Failed to bind zero page");
         return false;
      }

      dev->scratch_bo = bo;

      /* The contents of the scratch page are undefined, but making them nonzero
       * helps fuzz for bugs where we incorrectly read from the write section.
       */
      memset(agx_bo_map(dev->scratch_bo), 0xCA, AIL_PAGESIZE);
   }

   void *bo = agx_bo_create(dev, LIBAGX_PRINTF_BUFFER_SIZE, 0, AGX_BO_WRITEBACK,
                            "Printf/abort");

   /* Apple9's native render-context graph owns the legacy libagx printf
    * address.  The bounded Apple9 compiler does not lower printf intrinsics,
    * so retain the host-side diagnostic buffer at the first page after that
    * graph rather than colliding with the hardware ABI. */
   uint64_t printf_address =
      (dev->chip == AGX_CHIP_G16G || dev->chip == AGX_CHIP_G17P)
         ? AGX_APPLE9_FIXED_RENDER_CONTEXT_BASE +
              AGX_APPLE9_FIXED_RENDER_CONTEXT_SIZE
         : LIBAGX_PRINTF_BUFFER_ADDRESS;
   ret = agx_bo_bind(dev, bo, printf_address,
                     LIBAGX_PRINTF_BUFFER_SIZE, 0,
                     DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
   if (ret) {
      fprintf(stderr, "Failed to bind printf buffer");
      return false;
   }

   u_printf_init(&dev->printf, bo, agx_bo_map(bo));
   return true;
}

void
agx_close_device(struct agx_device *dev)
{
   if (dev->chip == AGX_CHIP_G16G || dev->chip == AGX_CHIP_G17P) {
      util_dynarray_foreach(&dev->apple9_compute_state_bos, struct agx_bo *, bo)
         agx_bo_unreference(dev, *bo);
      util_dynarray_fini(&dev->apple9_compute_state_bos);

      /* Both physical archives may also have ordinary construction VAs, but
       * exactly one has this extra fixed alias. Remove it while both BOs are
       * still alive so teardown does not depend on GEM-close side effects. */
      if (dev->apple9_fixed_usc_owner) {
         int ret = agx_bo_bind(dev, NULL, dev->shader_base,
                               AGX_APPLE9_FIXED_USC_ARENA_SIZE, 0,
                               DRM_ASAHI_BIND_UNBIND);
         if (ret)
            fprintf(stderr,
                    "Failed to unbind Apple9 fixed-USC alias at teardown\n");
         else
            dev->apple9_fixed_usc_owner = NULL;
      }
   }
   agx_bo_unreference(dev, dev->apple9_render_context);
   agx_bo_unreference(dev, dev->apple9_render_fixed_usc);
   agx_bo_unreference(dev, dev->apple9_compute_archive);
   free(dev->apple9_compute_archive_shadow);
   if (dev->chip == AGX_CHIP_G16G || dev->chip == AGX_CHIP_G17P)
      simple_mtx_destroy(&dev->apple9_archive_lock);
   agx_bo_unreference(dev, dev->printf.bo);
   agx_bo_unreference(dev, dev->zero_bo);
   agx_bo_unreference(dev, dev->scratch_bo);
   u_printf_destroy(&dev->printf);
   agx_bo_cache_evict_all(dev);
   util_sparse_array_finish(&dev->bo_map);
   agxdecode_destroy_context(dev->agxdecode);

   util_vma_heap_finish(&dev->main_heap);
   util_vma_heap_finish(&dev->usc_heap);
   glsl_type_singleton_decref();

   close(dev->fd);
}

uint32_t
agx_create_command_queue(struct agx_device *dev,
                         enum drm_asahi_priority priority)
{

   if (dev->debug & AGX_DBG_1QUEUE) {
      // Abuse this lock for this, it's debug only anyway
      simple_mtx_lock(&dev->vma_lock);
      if (dev->queue_id) {
         simple_mtx_unlock(&dev->vma_lock);
         return dev->queue_id;
      }
   }

   struct drm_asahi_queue_create queue_create = {
      .vm_id = dev->vm_id,
      .priority = priority,
      .usc_exec_base = dev->shader_base,
   };

   int ret =
      asahi_simple_ioctl(dev, DRM_IOCTL_ASAHI_QUEUE_CREATE, &queue_create);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_QUEUE_CREATE failed: %m\n");
      assert(0);
   }

   if (dev->debug & AGX_DBG_1QUEUE) {
      dev->queue_id = queue_create.queue_id;
      simple_mtx_unlock(&dev->vma_lock);
   }

   return queue_create.queue_id;
}

int
agx_destroy_command_queue(struct agx_device *dev, uint32_t queue_id)
{
   if (dev->debug & AGX_DBG_1QUEUE)
      return 0;

   struct drm_asahi_queue_destroy queue_destroy = {
      .queue_id = queue_id,
   };

   return asahi_simple_ioctl(dev, DRM_IOCTL_ASAHI_QUEUE_DESTROY,
                             &queue_destroy);
}

int
agx_import_sync_file(struct agx_device *dev, struct agx_bo *bo, int fd)
{
   struct dma_buf_import_sync_file import_sync_file_ioctl = {
      .flags = DMA_BUF_SYNC_WRITE,
      .fd = fd,
   };

   assert(fd >= 0);
   assert(bo->prime_fd != -1);

   int ret = drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_IMPORT_SYNC_FILE,
                      &import_sync_file_ioctl);
   assert(ret >= 0);

   return ret;
}

int
agx_export_sync_file(struct agx_device *dev, struct agx_bo *bo)
{
   struct dma_buf_export_sync_file export_sync_file_ioctl = {
      .flags = DMA_BUF_SYNC_RW,
      .fd = -1,
   };

   assert(bo->prime_fd != -1);

   int ret = drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE,
                      &export_sync_file_ioctl);
   assert(ret >= 0);
   assert(export_sync_file_ioctl.fd >= 0);

   return ret >= 0 ? export_sync_file_ioctl.fd : ret;
}

void
agx_debug_fault(struct agx_device *dev, uint64_t addr)
{
   pthread_mutex_lock(&dev->bo_map_lock);

   struct agx_bo *best = NULL;

   for (uint32_t handle = 0; handle < dev->max_handle; handle++) {
      struct agx_bo *bo = agx_lookup_bo(dev, handle);
      if (!bo->va)
         continue;

      uint64_t bo_addr = bo->va->addr;
      if (bo->flags & AGX_BO_LOW_VA)
         bo_addr += dev->shader_base;

      if (!bo->size || bo_addr > addr)
         continue;

      if (!best || bo_addr > best->va->addr)
         best = bo;
   }

   if (!best) {
      mesa_logw("Address 0x%" PRIx64 " is unknown\n", addr);
   } else {
      uint64_t start = best->va->addr;
      uint64_t end = best->va->addr + best->size;
      if (addr > (end + 1024 * 1024 * 1024)) {
         /* 1GiB max as a sanity check */
         mesa_logw("Address 0x%" PRIx64 " is unknown\n", addr);
      } else if (addr > end) {
         mesa_logw("Address 0x%" PRIx64 " is 0x%" PRIx64
                   " bytes beyond an object at 0x%" PRIx64 "..0x%" PRIx64
                   " (%s)\n",
                   addr, addr - end, start, end - 1, best->label);
      } else {
         mesa_logw("Address 0x%" PRIx64 " is 0x%" PRIx64
                   " bytes into an object at 0x%" PRIx64 "..0x%" PRIx64
                   " (%s)\n",
                   addr, addr - start, start, end - 1, best->label);
      }
   }

   pthread_mutex_unlock(&dev->bo_map_lock);
}

uint64_t
agx_get_gpu_timestamp(struct agx_device *dev)
{
   struct drm_asahi_get_time get_time = {.flags = 0};

   int ret = asahi_simple_ioctl(dev, DRM_IOCTL_ASAHI_GET_TIME, &get_time);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GET_TIME failed: %m\n");
   }

   return get_time.gpu_timestamp;
}

/* (Re)define UUID_SIZE to avoid including vulkan.h (or p_defines.h) here. */
#define UUID_SIZE 16

void
agx_get_device_uuid(const struct agx_device *dev, void *uuid)
{
   blake3_hasher blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   /* The device UUID uniquely identifies the given device within the machine.
    * Since we never have more than one device, this doesn't need to be a real
    * UUID, so we use BLAKE3("agx" + gpu_generation + gpu_variant + gpu_revision).
    */
   static const char *device_name = "agx";
   _mesa_blake3_update(&blake3_ctx, device_name, strlen(device_name));

   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_generation,
                     sizeof(dev->params.gpu_generation));
   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_variant,
                     sizeof(dev->params.gpu_variant));
   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_revision,
                     sizeof(dev->params.gpu_revision));

   uint8_t blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_final(&blake3_ctx, blake3);

   assert(BLAKE3_KEY_LEN >= UUID_SIZE);
   memcpy(uuid, blake3, UUID_SIZE);
}

void
agx_get_driver_uuid(void *uuid)
{
   const char *driver_id = PACKAGE_VERSION MESA_GIT_SHA1;

   /* The driver UUID is used for determining sharability of images and memory
    * between two Vulkan instances in separate processes, but also to
    * determining memory objects and sharability between Vulkan and OpenGL
    * driver. People who want to share memory need to also check the device
    * UUID.
    */
   blake3_hasher blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   _mesa_blake3_update(&blake3_ctx, driver_id, strlen(driver_id));

   uint8_t blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_final(&blake3_ctx, blake3);

   assert(BLAKE3_KEY_LEN >= UUID_SIZE);
   memcpy(uuid, blake3, UUID_SIZE);
}

unsigned
agx_get_num_cores(const struct agx_device *dev)
{
   unsigned n = 0;

   for (unsigned cl = 0; cl < dev->params.num_clusters_total; cl++) {
      n += util_bitcount(dev->params.core_masks[cl]);
   }

   return n;
}

struct agx_device_key
agx_gather_device_key(struct agx_device *dev)
{
   bool g13x_coh = (dev->params.gpu_generation == 13 &&
                    dev->params.num_clusters_total > 1) ||
                   dev->params.num_dies > 1;

   return (struct agx_device_key){
      .shader_isa = dev->params.gpu_generation >= 16
                       ? AGX_SHADER_ISA_APPLE9
                       : AGX_SHADER_ISA_APPLE8,
      .needs_g13x_coherency = u_tristate_make(g13x_coh),
      .soft_fault = agx_has_soft_fault(dev),
   };
}
