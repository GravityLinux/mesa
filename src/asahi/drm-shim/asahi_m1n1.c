/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2018 Broadcom
 * SPDX-License-Identifier: MIT
 */

/* Modern Asahi DRM-shim transport backed by m1n1's embedded Python driver. */

#include <Python.h>
#include "delta.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "drm-shim/drm_shim.h"
#include "drm-uapi/asahi_drm.h"

struct asahi_bo {
   struct shim_bo base;
   uint64_t client_id;
   uint32_t handle;
   bool python_registered;
   struct asahi_cpu_spans cpu_writes;
   uint64_t cpu_epoch; /* Zero means untracked CPU access. */
};

static PyObject *python_shim;
static bool python_ready;
static mtx_t python_call_lock;

__attribute__((visibility("default"))) int
asahi_m1n1_debug_stat(uint32_t key, uint64_t *value);

static int
asahi_client_id(int fd, uint64_t *client_id)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   if (!shim_fd)
      return -EBADF;
   *client_id = shim_fd->driver_id;
   return 0;
}

static struct asahi_bo *
asahi_bo(struct shim_bo *bo)
{
   return (struct asahi_bo *)bo;
}

/* Borrowed for the GEM lifetime; Mesa drops the pointer with its BO. */
__attribute__((visibility("default"))) uint64_t *
asahi_m1n1_cpu_epoch(int fd, uint32_t handle);

uint64_t *
asahi_m1n1_cpu_epoch(int fd, uint32_t handle)
{
   struct shim_fd *file = drm_shim_fd_lookup(fd);
   if (!file)
      return NULL;
   struct shim_bo *base = drm_shim_bo_lookup(file, handle);
   if (!base)
      return NULL;
   uint64_t *epoch = &asahi_bo(base)->cpu_epoch;
   drm_shim_bo_put(base);
   return epoch;
}

/* Permission to resend gaps comes from completed, explicit CPU writes. It
 * lasts only until the next successful upload, including all aliases. */
__attribute__((visibility("default"))) void
asahi_m1n1_cpu_written(int fd, uint32_t handle, uint64_t start, uint64_t size);

void asahi_m1n1_cpu_written(int fd, uint32_t handle, uint64_t start, uint64_t size)
{
   struct shim_fd *file = drm_shim_fd_lookup(fd);
   struct shim_bo *base = file ? drm_shim_bo_lookup(file, handle) : NULL;
   if (!base)
      return;
   struct asahi_cpu_spans *spans = &asahi_bo(base)->cpu_writes;
   if (!size || start > base->size || size > base->size - start)
      goto out;
   uint64_t end = start + size;
   /* Union only overlapping/touching writes; never authorize an unwritten gap. */
   for (uint64_t i = 0; i < spans->count;) {
      struct asahi_cpu_span r = spans->spans[i];
      if (r.start <= end && start <= r.end) {
         if (r.start < start) start = r.start;
         if (r.end > end) end = r.end;
         spans->spans[i] = spans->spans[--spans->count];
         i = 0;
      } else {
         ++i;
      }
   }
   /* Full table simply loses this optimization, never broadens permission. */
   if (spans->count < ASAHI_CPU_SPANS)
      spans->spans[spans->count++] = (struct asahi_cpu_span){start, end};
out:
   drm_shim_bo_put(base);
}

static int
python_call(const char *method, int64_t *result, size_t argument_count, ...)
{
   if (!python_ready || !python_shim) {
      fprintf(stderr, "m1n1 DRM shim: Python backend is unavailable\n");
      return -EIO;
   }

   /* pyserial may release the GIL while a proxy response is pending.  Keep
    * the single m1n1 request/reply stream locked for the whole callback. */
   mtx_lock(&python_call_lock);
   PyGILState_STATE gil = PyGILState_Ensure();
   PyObject *callable = PyObject_GetAttrString(python_shim, method);
   PyObject *arguments = NULL;
   PyObject *value = NULL;
   int ret = -EIO;

   if (!callable || !PyCallable_Check(callable)) {
      fprintf(stderr, "m1n1 DRM shim: missing Python callback %s\n", method);
      goto out;
   }

   arguments = PyTuple_New(argument_count);
   if (!arguments)
      goto out;

   va_list ap;
   va_start(ap, argument_count);
   for (size_t index = 0; index < argument_count; ++index) {
      uint64_t raw = va_arg(ap, uint64_t);
      PyObject *argument = PyLong_FromUnsignedLongLong(raw);
      if (!argument) {
         va_end(ap);
         goto out;
      }
      PyTuple_SET_ITEM(arguments, index, argument);
   }
   va_end(ap);

   value = PyObject_CallObject(callable, arguments);
   if (!value)
      goto out;

   int64_t converted = PyLong_AsLongLong(value);
   if (converted == -1 && PyErr_Occurred())
      goto out;

   if (result)
      *result = converted;
   ret = 0;

out:
   if (PyErr_Occurred()) {
      fprintf(stderr, "m1n1 DRM shim: Python callback %s failed\n", method);
      PyErr_Print();
   }
   Py_XDECREF(value);
   Py_XDECREF(arguments);
   Py_XDECREF(callable);
   PyGILState_Release(gil);
   mtx_unlock(&python_call_lock);
   return ret;
}

static int
python_status(const char *method, size_t argument_count, ...)
{
   if (!python_ready || !python_shim) {
      fprintf(stderr, "m1n1 DRM shim: Python backend is unavailable\n");
      return -EIO;
   }

   mtx_lock(&python_call_lock);
   PyGILState_STATE gil = PyGILState_Ensure();
   PyObject *callable = PyObject_GetAttrString(python_shim, method);
   PyObject *arguments = NULL;
   PyObject *value = NULL;
   int ret = -EIO;

   if (!callable || !PyCallable_Check(callable)) {
      fprintf(stderr, "m1n1 DRM shim: missing Python callback %s\n", method);
      goto out;
   }

   arguments = PyTuple_New(argument_count);
   if (!arguments)
      goto out;

   va_list ap;
   va_start(ap, argument_count);
   for (size_t index = 0; index < argument_count; ++index) {
      uint64_t raw = va_arg(ap, uint64_t);
      PyObject *argument = PyLong_FromUnsignedLongLong(raw);
      if (!argument) {
         va_end(ap);
         goto out;
      }
      PyTuple_SET_ITEM(arguments, index, argument);
   }
   va_end(ap);

   value = PyObject_CallObject(callable, arguments);
   if (!value)
      goto out;

   int64_t status = PyLong_AsLongLong(value);
   if (status == -1 && PyErr_Occurred())
      goto out;
   ret = (int)status;

out:
   if (PyErr_Occurred()) {
      fprintf(stderr, "m1n1 DRM shim: Python callback %s failed\n", method);
      PyErr_Print();
   }
   Py_XDECREF(value);
   Py_XDECREF(arguments);
   Py_XDECREF(callable);
   PyGILState_Release(gil);
   mtx_unlock(&python_call_lock);
   return ret;
}

static bool
python_backend_init(void)
{
   /* The interposer is already resident.  Do not recursively inject it into
    * helper processes spawned while importing m1n1 (notably llvm-config).
    */
   unsetenv("LD_PRELOAD");
   Py_Initialize();

   PyObject *module = PyImport_ImportModule("m1n1.g17p_shim_entry");
   if (!module)
      goto fail;

   PyObject *shim_class = PyObject_GetAttrString(module, "Shim");
   Py_DECREF(module);
   if (!shim_class)
      goto fail;

   PyObject *memfd = PyLong_FromLong(shim_device.mem_fd);
   if (!memfd) {
      Py_DECREF(shim_class);
      goto fail;
   }

   python_shim = PyObject_CallOneArg(shim_class, memfd);
   Py_DECREF(memfd);
   Py_DECREF(shim_class);
   if (!python_shim)
      goto fail;

   python_ready = true;
   int status = python_status("modern_enable", 0);
   if (status) {
      fprintf(stderr,
              "m1n1 DRM shim: Python modern entrypoint failed: %d\n",
              status);
      python_ready = false;
      goto fail;
   }

   fprintf(stderr, "m1n1 DRM shim: embedded Python backend ready\n");
   PyEval_SaveThread();
   return true;

fail:
   if (PyErr_Occurred()) {
      fprintf(stderr, "m1n1 DRM shim: could not initialize Python backend\n");
      PyErr_Print();
   }
   Py_XDECREF(python_shim);
   python_shim = NULL;
   python_ready = false;
   PyEval_SaveThread();
   return false;
}

static int
asahi_ioctl_get_params(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_get_params *get = arg;
   if (get->param_group || get->pad)
      return -EINVAL;

   return python_status("modern_get_params", 2,
                        (uint64_t)get->pointer, (uint64_t)get->size);
}

static int
asahi_ioctl_get_time(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_get_time *get = arg;
   if (get->flags)
      return -EINVAL;

   int64_t timestamp;
   int ret = python_call("modern_get_time", &timestamp, 0);
   if (ret)
      return ret;
   get->gpu_timestamp = timestamp;
   return 0;
}

static int
asahi_ioctl_vm_create(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_vm_create *create = arg;
   uint64_t client_id;
   int ret = asahi_client_id(fd, &client_id);
   if (ret)
      return ret;
   int64_t vm_id;
   ret = python_call("modern_vm_create", &vm_id, 3,
                         client_id, create->kernel_start,
                         create->kernel_end);
   if (ret)
      return ret;
   if (vm_id < 0)
      return (int)vm_id;
   create->vm_id = vm_id;
   return 0;
}

static int
asahi_ioctl_vm_destroy(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_vm_destroy *destroy = arg;
   uint64_t client_id;
   int ret = asahi_client_id(fd, &client_id);
   if (ret)
      return ret;
   return python_status("modern_vm_destroy", 2,
                        client_id, (uint64_t)destroy->vm_id);
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

   bo->client_id = shim_fd->driver_id;
   bo->handle = drm_shim_bo_get_handle(shim_fd, &bo->base);
   create->handle = bo->handle;

   ret = python_status("modern_gem_created", 8,
                       bo->client_id, (uint64_t)bo->handle,
                       bo->base.mem_addr, create->size,
                       (uint64_t)create->flags, (uint64_t)create->vm_id,
                       (uint64_t)(uintptr_t)&bo->cpu_epoch,
                       (uint64_t)(uintptr_t)&bo->cpu_writes);
   if (!ret)
      bo->python_registered = true;

   drm_shim_bo_put(&bo->base);
   return ret;
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

static int
asahi_ioctl_vm_bind(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_vm_bind *bind = arg;
   uint64_t client_id;
   int ret = asahi_client_id(fd, &client_id);
   if (ret)
      return ret;
   return python_status("modern_vm_bind", 5,
                        client_id, (uint64_t)bind->vm_id,
                        bind->userptr, (uint64_t)bind->num_binds,
                        (uint64_t)bind->stride);
}

static int
asahi_ioctl_gem_bind_object(int fd, unsigned long request, void *arg)
{
   uint64_t client_id;
   int ret = asahi_client_id(fd, &client_id);
   if (ret)
      return ret;
   return python_status("modern_gem_bind_object", 2,
                        client_id, (uint64_t)(uintptr_t)arg);
}

static int
asahi_ioctl_queue_create(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_queue_create *create = arg;
   uint64_t client_id;
   int ret = asahi_client_id(fd, &client_id);
   if (ret)
      return ret;
   int64_t queue_id;
   ret = python_call("modern_queue_create", &queue_id, 4,
                         client_id, (uint64_t)create->vm_id,
                         (uint64_t)create->priority, create->usc_exec_base);
   if (ret)
      return ret;
   if (queue_id < 0)
      return (int)queue_id;
   create->queue_id = queue_id;
   return 0;
}

static int
asahi_ioctl_queue_destroy(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_queue_destroy *destroy = arg;
   uint64_t client_id;
   int ret = asahi_client_id(fd, &client_id);
   if (ret)
      return ret;
   return python_status("modern_queue_destroy", 2,
                        client_id, (uint64_t)destroy->queue_id);
}

static int
asahi_ioctl_submit(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_submit *submit = arg;
   uint64_t client_id;
   int ret = asahi_client_id(fd, &client_id);
   if (ret)
      return ret;
   return python_status("modern_submit", 7,
                        client_id, (uint64_t)submit->queue_id,
                        submit->cmdbuf, (uint64_t)submit->cmdbuf_size,
                        submit->syncs, (uint64_t)submit->in_sync_count,
                        (uint64_t)submit->out_sync_count);
}

static void
asahi_bo_free(struct shim_bo *base)
{
   struct asahi_bo *bo = asahi_bo(base);
   if (!bo->python_registered)
      return;

   int ret = python_status("modern_gem_closed", 2,
                           bo->client_id, (uint64_t)bo->handle);
   if (ret)
      fprintf(stderr,
              "m1n1 DRM shim: GEM %u close callback failed: %d\n",
              bo->handle, ret);
}

static void
asahi_bo_handle_close(struct shim_bo *base, uint32_t handle)
{
   struct asahi_bo *bo = asahi_bo(base);
   if (!bo->python_registered || handle != bo->handle)
      return;

   int ret = python_status("modern_gem_closed", 2,
                           bo->client_id, (uint64_t)handle);
   if (ret) {
      fprintf(stderr,
              "m1n1 DRM shim: GEM %u handle-close callback failed: %d\n",
              handle, ret);
      return;
   }
   bo->python_registered = false;
}

static void
asahi_file_close(uint64_t client_id)
{
   int ret = python_status("modern_file_closed", 1, client_id);
   if (ret)
      fprintf(stderr,
              "m1n1 DRM shim: file %" PRIu64 " close callback failed: %d\n",
              client_id, ret);
}

/* Experiment-only diagnostic ABI used by the hardware resource stress gate. */
__attribute__((visibility("default"))) int
asahi_m1n1_debug_stat(uint32_t key, uint64_t *value)
{
   int64_t result;
   int ret = python_call("modern_debug_stat", &result, 1, (uint64_t)key);
   if (ret)
      return ret;
   *value = (uint64_t)result;
   return 0;
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_ASAHI_GET_PARAMS] = asahi_ioctl_get_params,
   [DRM_ASAHI_GET_TIME] = asahi_ioctl_get_time,
   [DRM_ASAHI_VM_CREATE] = asahi_ioctl_vm_create,
   [DRM_ASAHI_VM_DESTROY] = asahi_ioctl_vm_destroy,
   [DRM_ASAHI_VM_BIND] = asahi_ioctl_vm_bind,
   [DRM_ASAHI_GEM_CREATE] = asahi_ioctl_gem_create,
   [DRM_ASAHI_GEM_MMAP_OFFSET] = asahi_ioctl_gem_mmap_offset,
   [DRM_ASAHI_GEM_BIND_OBJECT] = asahi_ioctl_gem_bind_object,
   [DRM_ASAHI_QUEUE_CREATE] = asahi_ioctl_queue_create,
   [DRM_ASAHI_QUEUE_DESTROY] = asahi_ioctl_queue_destroy,
   [DRM_ASAHI_SUBMIT] = asahi_ioctl_submit,
};

void
drm_shim_driver_init(void)
{
   mtx_init(&python_call_lock, mtx_plain);
   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);
   shim_device.driver_bo_free = asahi_bo_free;
   shim_device.driver_bo_handle_close = asahi_bo_handle_close;
   shim_device.driver_file_close = asahi_file_close;

   const char *compatible = "apple,gpu";
   if (python_backend_init()) {
      int64_t chip_id = 0;
      if (!python_call("modern_chip_id", &chip_id, 0)) {
         if (chip_id == 0x8132)
            compatible = "apple,gpu-g16g";
         else if (chip_id == 0x8140)
            compatible = "apple,gpu-g17p";
      }
   }

   drm_shim_platform_device_setup("asahi", "/soc/agx", compatible);
}
