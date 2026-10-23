/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2018 Broadcom
 * SPDX-License-Identifier: MIT
 */

/* A DRM shim that accepts every submission without a GPU. It lets host CI
 * run the Gallium and Vulkan drivers through their CPU-side paths (shader
 * compilation, state emission, command stream construction). Rendered
 * results are meaningless.
 *
 * ASAHI_NOOP_GPU selects the reported GPU: g13g (default) or g16g.
 */

#include <stdlib.h>
#include <string.h>

#include "drm-shim/drm_shim.h"
#include "drm-uapi/asahi_drm.h"

static struct drm_asahi_params_global params = {
   .gpu_generation = 13,
   .gpu_variant = 'G',
   .gpu_revision = 0,
   .chip_id = 0x8103,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 8,
   .max_frequency_khz = 1278000,
   .core_masks = {0xff},
   .vm_start = 0x4000,
   .vm_end = 0x7fffff8000ull,
   .max_commands_per_submission = 64,
   .max_attachments = 64,
   .command_timestamp_frequency_hz = 1000000000,
};

struct asahi_bo {
   struct shim_bo base;
};

static int
asahi_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
asahi_ioctl_get_params(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_get_params *gp = arg;

   if (gp->param_group != 0)
      return -EINVAL;

   memcpy((void *)(uintptr_t)gp->pointer, &params,
          MIN2(gp->size, sizeof(params)));
   return 0;
}

static int
asahi_ioctl_vm_create(int fd, unsigned long request, void *arg)
{
   static uint32_t next_vm = 1;
   struct drm_asahi_vm_create *create = arg;
   create->vm_id = next_vm++;
   return 0;
}

static int
asahi_ioctl_queue_create(int fd, unsigned long request, void *arg)
{
   static uint32_t next_queue = 1;
   struct drm_asahi_queue_create *create = arg;
   create->queue_id = next_queue++;
   return 0;
}

static int
asahi_ioctl_gem_create(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_create *create = arg;
   struct asahi_bo *bo = calloc(1, sizeof(*bo));
   if (!bo)
      return -ENOMEM;

   int ret = drm_shim_bo_init(&bo->base, create->size);
   if (ret) {
      free(bo);
      return ret;
   }

   create->handle = drm_shim_bo_get_handle(shim_fd, &bo->base);
   drm_shim_bo_put(&bo->base);
   return 0;
}

static int
asahi_ioctl_gem_mmap_offset(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_mmap_offset *map = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, map->handle);
   if (!bo)
      return -ENOENT;

   map->offset = drm_shim_bo_get_mmap_offset(shim_fd, bo);
   drm_shim_bo_put(bo);
   return 0;
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_ASAHI_GET_PARAMS] = asahi_ioctl_get_params,
   [DRM_ASAHI_GET_TIME] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_CREATE] = asahi_ioctl_vm_create,
   [DRM_ASAHI_VM_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_BIND] = asahi_ioctl_noop,
   [DRM_ASAHI_GEM_CREATE] = asahi_ioctl_gem_create,
   [DRM_ASAHI_GEM_MMAP_OFFSET] = asahi_ioctl_gem_mmap_offset,
   [DRM_ASAHI_GEM_BIND_OBJECT] = asahi_ioctl_noop,
   [DRM_ASAHI_QUEUE_CREATE] = asahi_ioctl_queue_create,
   [DRM_ASAHI_QUEUE_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_SUBMIT] = asahi_ioctl_noop,
};

void
drm_shim_driver_init(void)
{
   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);

   const char *compatible = "apple,gpu-g13g";
   const char *gpu = getenv("ASAHI_NOOP_GPU");
   if (gpu && !strcmp(gpu, "g16g")) {
      params.gpu_generation = 16;
      params.chip_id = 0x8132;
      params.num_cores_per_cluster = 10;
      params.core_masks[0] = 0x3ff;
      /* 42-bit UAT with the fixed 1-TiB USC aperture, and a kernel-private
       * heap inside the compact render context. */
      params.vm_end = (1ull << 42) - 0x4000;
      params.vm_kernel_min_size = 0x10000000;
      compatible = "apple,gpu-g16g";
   }

   drm_shim_platform_device_setup("asahi", "/soc/agx", compatible);
}
