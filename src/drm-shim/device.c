/*
 * Copyright © 2018 Broadcom
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/** @file
 *
 * Implements core GEM support (particularly ioctls) underneath the libc ioctl
 * wrappers, and calls into the driver-specific code as necessary.
 */

#include <c11/threads.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "drm-uapi/drm.h"
#include "drm_shim.h"
#include "util/hash_table.h"
#include "util/u_atomic.h"

#define SHIM_MEM_SIZE (4ull * 1024 * 1024 * 1024)

#ifndef HAVE_MEMFD_CREATE
#include <sys/syscall.h>

static inline int
memfd_create(const char *name, unsigned int flags)
{
   return syscall(SYS_memfd_create, name, flags);
}
#endif

/* Global state for the shim shared between libc, core, and driver. */
struct shim_device shim_device;

long shim_page_size;

struct shim_mapping {
   uintptr_t addr;
   size_t length;
   struct shim_bo *bo;
   struct shim_mapping *next;
};

static uint32_t
uint_key_hash(const void *key)
{
   return (uintptr_t)key;
}

static bool
uint_key_compare(const void *a, const void *b)
{
   return a == b;
}

/**
 * Called when the first libc shim is called, to initialize GEM simulation
 * state (other than the shims themselves).
 */
void
drm_shim_device_init(void)
{
   shim_device.fd_map = _mesa_hash_table_create(NULL,
                                                uint_key_hash,
                                                uint_key_compare);

   shim_device.offset_map = _mesa_hash_table_u64_create(NULL);
   shim_device.prime_map = _mesa_hash_table_u64_create(NULL);

   mtx_init(&shim_device.lock, mtx_plain);

   shim_device.mem_fd = memfd_create("shim mem", MFD_CLOEXEC);
   assert(shim_device.mem_fd != -1);

   ASSERTED int ret = ftruncate(shim_device.mem_fd, SHIM_MEM_SIZE);
   assert(ret == 0);

   /* The man page for mmap() says
    *
    *    offset must be a multiple of the page size as returned by
    *    sysconf(_SC_PAGE_SIZE).
    *
    * Depending on the configuration of the kernel, this may not be 4096. Get
    * this page size once and use it as the page size throughout, ensuring that
    * are offsets are page-size aligned as required. Otherwise, mmap will fail
    * with EINVAL.
    */

   shim_page_size = sysconf(_SC_PAGESIZE);

   util_vma_heap_init(&shim_device.mem_heap, shim_page_size,
                      SHIM_MEM_SIZE - shim_page_size);

   drm_shim_driver_init();
}

static struct shim_fd *
drm_shim_file_create(int fd)
{
   struct shim_fd *shim_fd = calloc(1, sizeof(*shim_fd));

   shim_fd->fd = fd;
   p_atomic_set(&shim_fd->refcount, 1);
   mtx_init(&shim_fd->handle_lock, mtx_plain);
   shim_fd->handles = _mesa_hash_table_create(NULL,
                                              uint_key_hash,
                                              uint_key_compare);
   shim_fd->next_syncobj_handle = 1;

   return shim_fd;
}

/**
 * Called when the libc shims have interposed an open or dup of our simulated
 * DRM device.
 */
void drm_shim_fd_register(int fd, struct shim_fd *shim_fd)
{
   bool fresh = !shim_fd;
   if (fresh)
      shim_fd = drm_shim_file_create(fd);
   else
      p_atomic_inc(&shim_fd->refcount);

   mtx_lock(&shim_device.lock);
   if (fresh)
      shim_fd->driver_id = ++shim_device.next_driver_id;
   _mesa_hash_table_insert(shim_device.fd_map, (void *)(uintptr_t)(fd + 1), shim_fd);
   mtx_unlock(&shim_device.lock);
}

static void handle_delete_fxn(struct hash_entry *entry)
{
   struct shim_bo *bo = entry->data;
   if (shim_device.driver_bo_handle_close)
      shim_device.driver_bo_handle_close(
         bo, (uint32_t)(uintptr_t)entry->key);
   drm_shim_bo_put(bo);
}

void drm_shim_fd_unregister(int fd)
{
   if (fd == -1)
      return;

   mtx_lock(&shim_device.lock);
   struct hash_entry *entry =
         _mesa_hash_table_search(shim_device.fd_map, (void *)(uintptr_t)(fd + 1));
   if (!entry) {
      mtx_unlock(&shim_device.lock);
      return;
   }
   struct shim_fd *shim_fd = entry->data;
   _mesa_hash_table_remove(shim_device.fd_map, entry);
   mtx_unlock(&shim_device.lock);

   if (!p_atomic_dec_zero(&shim_fd->refcount))
      return;

   _mesa_hash_table_destroy(shim_fd->handles, handle_delete_fxn);
   if (shim_device.driver_file_close)
      shim_device.driver_file_close(shim_fd->driver_id);
   free(shim_fd);
}

struct shim_fd *
drm_shim_fd_lookup(int fd)
{
   if (!drm_shim_inited() || fd == -1)
      return NULL;

   mtx_lock(&shim_device.lock);
   struct hash_entry *entry =
      _mesa_hash_table_search(shim_device.fd_map, (void *)(uintptr_t)(fd + 1));

   struct shim_fd *result = entry ? entry->data : NULL;
   mtx_unlock(&shim_device.lock);

   return result;
}

/* ioctl used by drmGetVersion() */
static int
drm_shim_ioctl_version(int fd, unsigned long request, void *arg)
{
   struct drm_version *args = arg;
   const char *date = "20190320";
   const char *desc = "shim";

   args->version_major = shim_device.version_major;
   args->version_minor = shim_device.version_minor;
   args->version_patchlevel = shim_device.version_patchlevel;

   if (args->name)
      strncpy(args->name, shim_device.driver_name, args->name_len);
   if (args->date)
      strncpy(args->date, date, args->date_len);
   if (args->desc)
      strncpy(args->desc, desc, args->desc_len);
   args->name_len = strlen(shim_device.driver_name);
   args->date_len = strlen(date);
   args->desc_len = strlen(desc);

   return 0;
}

static int
drm_shim_ioctl_get_unique(int fd, unsigned long request, void *arg)
{
   struct drm_unique *gu = arg;

   if (gu->unique && shim_device.unique)
      strncpy(gu->unique, shim_device.unique, gu->unique_len);
   gu->unique_len = shim_device.unique ? strlen(shim_device.unique) : 0;

   return 0;
}

static int
drm_shim_ioctl_get_cap(int fd, unsigned long request, void *arg)
{
   struct drm_get_cap *gc = arg;

   switch (gc->capability) {
   case DRM_CAP_PRIME:
      gc->value = DRM_PRIME_CAP_IMPORT | DRM_PRIME_CAP_EXPORT;
      return 0;

   case DRM_CAP_SYNCOBJ:
   case DRM_CAP_SYNCOBJ_TIMELINE:
   case DRM_CAP_ADDFB2_MODIFIERS:
      gc->value = 1;
      return 0;

   default:
      fprintf(stderr, "DRM_IOCTL_GET_CAP: unhandled 0x%x\n",
              (int)gc->capability);
      return -1;
   }
}

static int
drm_shim_ioctl_gem_close(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_gem_close *c = arg;

   if (!c->handle)
      return 0;

   mtx_lock(&shim_fd->handle_lock);
   struct hash_entry *entry =
      _mesa_hash_table_search(shim_fd->handles, (void *)(uintptr_t)c->handle);
   if (!entry) {
      mtx_unlock(&shim_fd->handle_lock);
      return -EINVAL;
   }

   struct shim_bo *bo = entry->data;
   _mesa_hash_table_remove(shim_fd->handles, entry);
   mtx_unlock(&shim_fd->handle_lock);
   if (shim_device.driver_bo_handle_close)
      shim_device.driver_bo_handle_close(bo, c->handle);
   drm_shim_bo_put(bo);
   return 0;
}

static int
drm_shim_ioctl_syncobj_create(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_syncobj_create *create = arg;

   mtx_lock(&shim_fd->handle_lock);
   create->handle = shim_fd->next_syncobj_handle++;
   assert(create->handle != 0);
   mtx_unlock(&shim_fd->handle_lock);

   return 0;
}

static int
drm_shim_ioctl_syncobj_handle_to_fd(int fd, unsigned long request, void *arg)
{
   struct drm_syncobj_handle *handle = arg;

   /*
    * Driver-backed drm-shim submissions complete synchronously.  Represent
    * an exported sync_file by an already-signalled eventfd, which has the
    * pollable-fd semantics Mesa needs without inventing asynchronous work.
    */
   handle->fd = eventfd(1, EFD_CLOEXEC | EFD_NONBLOCK);
   return handle->fd >= 0 ? 0 : -errno;
}

static int
drm_shim_ioctl_syncobj_fd_to_handle(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_syncobj_handle *handle = arg;

   /* IMPORT_SYNC_FILE targets an existing handle.  Opaque imports allocate a
    * new handle in the syncobj namespace.
    */
   if (handle->handle)
      return 0;

   mtx_lock(&shim_fd->handle_lock);
   handle->handle = shim_fd->next_syncobj_handle++;
   assert(handle->handle != 0);
   mtx_unlock(&shim_fd->handle_lock);
   return 0;
}

/* PRIME: an exported BO is represented by a memfd of the BO's size, so that
 * lseek(fd, 0, SEEK_END) reports the size like a real dma-buf. Exported BOs
 * stay alive for the life of the process. Importing returns this file's
 * existing handle for the BO, as the kernel does.
 */
static int
drm_shim_ioctl_prime_handle_to_fd(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_prime_handle *args = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, args->handle);
   if (!bo)
      return -ENOENT;

   /* fstat() is intercepted and takes shim_device.lock: create the memfd
    * before locking, and drop it if another export won the race. */
   struct stat st;
   int memfd = -1;
   if (p_atomic_read(&bo->prime_fd) <= 0) {
      memfd = memfd_create("shim dma-buf", MFD_CLOEXEC);
      if (memfd < 0 || ftruncate(memfd, bo->size) || fstat(memfd, &st)) {
         if (memfd >= 0)
            close(memfd);
         drm_shim_bo_put(bo);
         return -ENOMEM;
      }
   }

   mtx_lock(&shim_device.lock);
   if (bo->prime_fd <= 0 && memfd >= 0) {
      bo->prime_fd = memfd;
      memfd = -1;
      drm_shim_bo_get(bo);
      _mesa_hash_table_u64_insert(shim_device.prime_map, st.st_ino, bo);
   }
   int prime_fd = bo->prime_fd;
   mtx_unlock(&shim_device.lock);
   if (memfd >= 0)
      close(memfd);
   /* fcntl() is intercepted too. The memfd lives as long as the process. */
   args->fd = fcntl(prime_fd, F_DUPFD_CLOEXEC, 0);
   drm_shim_bo_put(bo);
   return args->fd < 0 ? -errno : 0;
}

static int
drm_shim_ioctl_prime_fd_to_handle(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_prime_handle *args = arg;
   struct stat st;
   if (fstat(args->fd, &st))
      return -EBADF;

   mtx_lock(&shim_device.lock);
   struct shim_bo *bo = _mesa_hash_table_u64_search(shim_device.prime_map,
                                                    st.st_ino);
   mtx_unlock(&shim_device.lock);
   if (!bo)
      return -EINVAL;

   mtx_lock(&shim_fd->handle_lock);
   hash_table_foreach(shim_fd->handles, entry) {
      if (entry->data == bo) {
         args->handle = (uintptr_t)entry->key;
         mtx_unlock(&shim_fd->handle_lock);
         return 0;
      }
   }
   mtx_unlock(&shim_fd->handle_lock);
   args->handle = drm_shim_bo_get_handle(shim_fd, bo);
   return 0;
}

static int
drm_shim_ioctl_stub(int fd, unsigned long request, void *arg)
{
   return 0;
}

ioctl_fn_t core_ioctls[] = {
   [_IOC_NR(DRM_IOCTL_VERSION)] = drm_shim_ioctl_version,
   [_IOC_NR(DRM_IOCTL_GET_UNIQUE)] = drm_shim_ioctl_get_unique,
   [_IOC_NR(DRM_IOCTL_GET_CAP)] = drm_shim_ioctl_get_cap,
   [_IOC_NR(DRM_IOCTL_GEM_CLOSE)] = drm_shim_ioctl_gem_close,
   [_IOC_NR(DRM_IOCTL_PRIME_HANDLE_TO_FD)] = drm_shim_ioctl_prime_handle_to_fd,
   [_IOC_NR(DRM_IOCTL_PRIME_FD_TO_HANDLE)] = drm_shim_ioctl_prime_fd_to_handle,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_CREATE)] = drm_shim_ioctl_syncobj_create,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_DESTROY)] = drm_shim_ioctl_stub,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD)] =
      drm_shim_ioctl_syncobj_handle_to_fd,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE)] =
      drm_shim_ioctl_syncobj_fd_to_handle,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_WAIT)] = drm_shim_ioctl_stub,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TRANSFER)] = drm_shim_ioctl_stub,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_RESET)] = drm_shim_ioctl_stub,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL)] = drm_shim_ioctl_stub,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT)] = drm_shim_ioctl_stub,
   [_IOC_NR(DRM_IOCTL_SYNCOBJ_QUERY)] = drm_shim_ioctl_stub,
};

/**
 * Implements the GEM core ioctls, and calls into driver-specific ioctls.
 */
int
drm_shim_ioctl(int fd, unsigned long request, void *arg)
{
   ASSERTED int type = _IOC_TYPE(request);
   int nr = _IOC_NR(request);

   assert(type == DRM_IOCTL_BASE);

   if (nr >= DRM_COMMAND_BASE && nr < DRM_COMMAND_END) {
      int driver_nr = nr - DRM_COMMAND_BASE;

      if (driver_nr < shim_device.driver_ioctl_count &&
          shim_device.driver_ioctls[driver_nr]) {
         return shim_device.driver_ioctls[driver_nr](fd, request, arg);
      }
   } else {
      if (nr < ARRAY_SIZE(core_ioctls) && core_ioctls[nr]) {
         return core_ioctls[nr](fd, request, arg);
      }
   }

   if (nr >= DRM_COMMAND_BASE && nr < DRM_COMMAND_END) {
      fprintf(stderr,
              "DRM_SHIM: unhandled driver DRM ioctl %d (0x%x) (0x%08lx)\n",
              nr - DRM_COMMAND_BASE, nr - DRM_COMMAND_BASE, request);
   } else {
      fprintf(stderr,
              "DRM_SHIM: unhandled core DRM ioctl 0x%X (0x%08lx)\n",
              nr, request);
   }

   return -EINVAL;
}

int
drm_shim_bo_init(struct shim_bo *bo, size_t size)
{

   mtx_lock(&shim_device.lock);
   bo->mem_addr = util_vma_heap_alloc(&shim_device.mem_heap, size, shim_page_size);
   mtx_unlock(&shim_device.lock);

   if (!bo->mem_addr)
      return -ENOMEM;

   bo->size = size;
   drm_shim_bo_get(bo);

   return 0;
}

struct shim_bo *
drm_shim_bo_lookup(struct shim_fd *shim_fd, int handle)
{
   if (!handle)
      return NULL;

   mtx_lock(&shim_fd->handle_lock);
   struct hash_entry *entry =
      _mesa_hash_table_search(shim_fd->handles, (void *)(uintptr_t)handle);
   struct shim_bo *bo = entry ? entry->data : NULL;
   mtx_unlock(&shim_fd->handle_lock);

   if (bo)
      p_atomic_inc(&bo->refcount);

   return bo;
}

void
drm_shim_bo_get(struct shim_bo *bo)
{
   p_atomic_inc(&bo->refcount);
}

void
drm_shim_bo_put(struct shim_bo *bo)
{
   assert(p_atomic_read(&bo->refcount) > 0);

   if (p_atomic_dec_return(&bo->refcount) > 0)
      return;

   if (shim_device.driver_bo_free)
      shim_device.driver_bo_free(bo);

   mtx_lock(&shim_device.lock);
   _mesa_hash_table_u64_remove(shim_device.offset_map, bo->mem_addr);
   util_vma_heap_free(&shim_device.mem_heap, bo->mem_addr, bo->size);
   mtx_unlock(&shim_device.lock);
   free(bo);
}

int
drm_shim_bo_get_handle(struct shim_fd *shim_fd, struct shim_bo *bo)
{
   /* We should probably have some real datastructure for finding the free
    * number.
    */
   mtx_lock(&shim_fd->handle_lock);
   for (int new_handle = 1; ; new_handle++) {
      void *key = (void *)(uintptr_t)new_handle;
      if (!_mesa_hash_table_search(shim_fd->handles, key)) {
         drm_shim_bo_get(bo);
         _mesa_hash_table_insert(shim_fd->handles, key, bo);
         mtx_unlock(&shim_fd->handle_lock);
         return new_handle;
      }
   }
   mtx_unlock(&shim_fd->handle_lock);

   return 0;
}

/* Creates an mmap offset for the BO in the DRM fd.
 */
uint64_t
drm_shim_bo_get_mmap_offset(struct shim_fd *shim_fd, struct shim_bo *bo)
{
   mtx_lock(&shim_device.lock);
   _mesa_hash_table_u64_insert(shim_device.offset_map, bo->mem_addr, bo);
   mtx_unlock(&shim_device.lock);

   /* reuse the buffer address as the mmap offset: */
   return bo->mem_addr;
}

void
drm_shim_init_iomem_region(off64_t offset, size_t size,
                           void *(*mmap_handler)(size_t, int, int, off64_t))
{
   shim_device.iomem_region.mmap = mmap_handler;
   shim_device.iomem_region.start = offset;
   shim_device.iomem_region.size = size;
}

/* For mmap() on the DRM fd, look up the BO from the "offset" and map the BO's
 * fd.
 */
void *
drm_shim_mmap(struct shim_fd *shim_fd, size_t length, int prot, int flags,
              int fd, off64_t offset)
{
   if (shim_device.iomem_region.mmap &&
       offset >= shim_device.iomem_region.start &&
       offset + length <= shim_device.iomem_region.start + shim_device.iomem_region.size) {
      return shim_device.iomem_region.mmap(length, prot, flags, offset);
   }

   mtx_lock(&shim_device.lock);
   struct shim_bo *bo = _mesa_hash_table_u64_search(shim_device.offset_map, offset);
   if (bo)
      drm_shim_bo_get(bo);
   mtx_unlock(&shim_device.lock);

   if (!bo)
      return MAP_FAILED;

   if (length > bo->size) {
      drm_shim_bo_put(bo);
      return MAP_FAILED;
   }

   /* The offset we pass to mmap must be aligned to the page size */
   assert((bo->mem_addr & (shim_page_size - 1)) == 0);

   void *mapping = mmap(NULL, length, prot, flags,
                        shim_device.mem_fd, bo->mem_addr);
   if (mapping == MAP_FAILED) {
      drm_shim_bo_put(bo);
      return MAP_FAILED;
   }

   struct shim_mapping *record = malloc(sizeof(*record));
   if (!record) {
      munmap(mapping, length);
      drm_shim_bo_put(bo);
      errno = ENOMEM;
      return MAP_FAILED;
   }
   *record = (struct shim_mapping) {
      .addr = (uintptr_t)mapping,
      .length = length,
      .bo = bo,
   };
   mtx_lock(&shim_device.lock);
   record->next = shim_device.mappings;
   shim_device.mappings = record;
   mtx_unlock(&shim_device.lock);
   return mapping;
}

void
drm_shim_munmap_notify(void *addr, size_t length)
{
   uintptr_t start = (uintptr_t)addr;
   uintptr_t end = start + length;
   if (end < start)
      return;

   struct shim_mapping *released = NULL;
   mtx_lock(&shim_device.lock);
   struct shim_mapping **link = &shim_device.mappings;
   while (*link) {
      struct shim_mapping *record = *link;
      uintptr_t map_start = record->addr;
      uintptr_t map_end = map_start + record->length;
      if (end <= map_start || start >= map_end) {
         link = &record->next;
         continue;
      }

      uintptr_t cut_start = MAX2(start, map_start);
      uintptr_t cut_end = MIN2(end, map_end);
      if (cut_start == map_start && cut_end == map_end) {
         *link = record->next;
         record->next = released;
         released = record;
      } else if (cut_start == map_start) {
         record->addr = cut_end;
         record->length = map_end - cut_end;
         link = &record->next;
      } else if (cut_end == map_end) {
         record->length = cut_start - map_start;
         link = &record->next;
      } else {
         struct shim_mapping *right = malloc(sizeof(*right));
         if (!right)
            abort();
         drm_shim_bo_get(record->bo);
         *right = (struct shim_mapping) {
            .addr = cut_end,
            .length = map_end - cut_end,
            .bo = record->bo,
            .next = record->next,
         };
         record->length = cut_start - map_start;
         record->next = right;
         link = &right->next;
      }
   }
   mtx_unlock(&shim_device.lock);

   while (released) {
      struct shim_mapping *next = released->next;
      drm_shim_bo_put(released->bo);
      free(released);
      released = next;
   }
}
