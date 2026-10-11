/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_device.h"
#include "util/rwlock.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "agx_bg_eot.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "decode.h"
#include "vk_cmd_queue.h"
#include "vk_dispatch_table.h"

#include "hk_private.h"

#include "hk_descriptor_table.h"
#include "hk_queue.h"
#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct hk_physical_device;
struct vk_pipeline_cache;

typedef void (*hk_internal_builder_t)(struct nir_builder *b, const void *key);

struct hk_internal_key {
   hk_internal_builder_t builder;
   size_t key_size;
   uint8_t key[];
};

struct hk_internal_shaders {
   simple_mtx_t lock;
   struct hash_table *ht;
};

struct hk_rc_sampler {
   struct agx_sampler_packed key;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct hk_sampler_heap {
   simple_mtx_t lock;

   struct hk_descriptor_table table;

   /* Map of agx_sampler_packed to hk_rc_sampler */
   struct hash_table *ht;
};

struct hk_device {
   struct vk_device vk;
   struct agx_device dev;
   struct agxdecode_ctx *decode_ctx;

   struct hk_sampler_heap samplers;

   /* G16 private texel-fetch sampler, addressable as a native table entry
    * relative to the USC aperture. */
   struct agx_bo *apple9_fetch_sampler;
   /* Last G16 graphics program id handed out, see hk_shader::apple9_program_id. */
   uint64_t apple9_last_program_id;
   /* Runtime sysval block of every kernel, see hk_apple9_upload_textures. */
   struct agx_bo *apple9_compute_sysvals;

   /* G16 residency table of every image without sparse residency: all of
    * its levels and layers resolve to a single resident page. */
   struct agx_bo *apple9_resident;

   struct {
      simple_mtx_t lock;
      struct list_head available;
   } events;

   struct vk_meta_device meta;
   struct agx_bg_eot_cache bg_eot;

   struct {
      struct agx_bo *bo;
      uint64_t heap;
      /* struct poly_heap of mesh shader records sized on the GPU */
      uint64_t mesh_heap;
   } rodata;

   struct hk_internal_shaders prolog_epilog;
   struct hk_internal_shaders kernels;
   struct hk_api_shader *null_fs;

   /* Apple9 draw-time variants of deferred graphics shaders, keyed by the
    * bound shader objects and the resolved graphics key. */
   struct {
      simple_mtx_t lock;
      struct hash_table *ht;
   } apple9_variants;

   /* Apple9 shader code BOs shared by byte-identical bodies, see
    * hk_upload_shader. */
   struct {
      simple_mtx_t lock;
      struct hash_table *ht;
      struct list_head lru;
      unsigned count;
   } apple9_bodies;


   /* Apple9 tile load and store programs, keyed by their tile buffer key. */
   struct {
      simple_mtx_t lock;
      struct hash_table *ht;
   } apple9_tile_helpers;

   /* Indirected for common secondary emulation */
   struct vk_device_dispatch_table cmd_dispatch;

   /* Heap used for GPU-side memory allocation for geometry/tessellation.
    *
    * Control streams accessing the heap must be serialized. This is not
    * expected to be a legitimate problem. If it is, we can rework later.
    */
   struct agx_bo *heap;
   util_once_flag heap_init_once;
   struct agx_bo *mesh_heap;
   util_once_flag mesh_heap_init_once;

   struct {
      struct agx_scratch vs, fs, cs;
      simple_mtx_t lock;
   } scratch;

   uint32_t perftest;

};

VK_DEFINE_HANDLE_CASTS(hk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

/* Sparse resources shadow their pages read-only at sparse_ro_offset in the
 * main heap. VAs in the USC aperture (G16 sparse descriptor buffers) have no
 * such shadow. */
static inline bool
hk_va_has_ro_shadow(const struct hk_device *dev, uint64_t addr)
{
   return addr < dev->dev.shader_base ||
          addr >= dev->dev.shader_base + (1ull << 32);
}

enum hk_perftest {
   HK_PERF_NOTESS = BITFIELD_BIT(0),
   HK_PERF_NOBORDER = BITFIELD_BIT(1),
   HK_PERF_NOBARRIER = BITFIELD_BIT(2),
   HK_PERF_BATCH = BITFIELD_BIT(3),
   HK_PERF_NOROBUST = BITFIELD_BIT(4),
};

#define HK_PERF(dev, flag) unlikely((dev)->perftest &HK_PERF_##flag)

static inline struct hk_physical_device *
hk_device_physical(struct hk_device *dev)
{
   return (struct hk_physical_device *)dev->vk.physical;
}

VkResult hk_device_init_meta(struct hk_device *dev);
void hk_device_finish_meta(struct hk_device *dev);

VkResult hk_sampler_heap_add(struct hk_device *dev,
                             struct agx_sampler_packed desc,
                             struct hk_rc_sampler **out);

void hk_sampler_heap_remove(struct hk_device *dev, struct hk_rc_sampler *rc);

static inline struct agx_scratch *
hk_device_scratch_locked(struct hk_device *dev, mesa_shader_stage stage)
{
   simple_mtx_assert_locked(&dev->scratch.lock);

   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      return &dev->scratch.fs;
   case MESA_SHADER_VERTEX:
      return &dev->scratch.vs;
   default:
      return &dev->scratch.cs;
   }
}

static inline void
hk_device_alloc_scratch(struct hk_device *dev, mesa_shader_stage stage,
                        unsigned size)
{
   simple_mtx_lock(&dev->scratch.lock);
   agx_scratch_alloc(hk_device_scratch_locked(dev, stage), size, 0);
   simple_mtx_unlock(&dev->scratch.lock);
}

/* With soft faults, accesses to unmapped GPU pages read zero and drop
 * writes, which is strict sparse residency. Apple9 then leaves unbound
 * sparse ranges unmapped instead of mapping the shared scratch page, whose
 * contents reads would observe. HK_APPLE9_SPARSE_SCRATCH=1 restores scratch.
 */
/* HK_APPLE9_SPARSE_UNMAPPED_RAW, read once. */
bool hk_sparse_unmapped_raw(void);

static inline bool
hk_sparse_unbound_unmapped(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G16G && agx_has_soft_fault((struct agx_device *)dev) &&
          !getenv("HK_APPLE9_SPARSE_SCRATCH");
}
/* Debug only: HK_CPU_TIMING, see hk_cpu_timing.c */
struct vk_device_dispatch_table;
void hk_cpu_timing_install(struct vk_device_dispatch_table *table);
void hk_cpu_timing_note_gs(unsigned shape, bool indirect, uint64_t prims,
                           unsigned max_indices);
struct hk_cpu_compile_scope {
   bool active;
   uint64_t wall, cpu;
};
void hk_cpu_timing_compile_begin(struct hk_cpu_compile_scope *s);
void hk_cpu_timing_compile_end(const struct hk_cpu_compile_scope *s,
                               bool draw_time);
/* Wall time of draw sub-steps; begin returns 0 when timing is off. */
enum hk_cpu_sub {
   HK_CPU_SUB_RESOLVE,
   HK_CPU_SUB_RESOLVE_SLOW,
   HK_CPU_SUB_TESS,
   HK_CPU_SUB_GS,
   HK_CPU_SUB_ROBUST_INDEX,
   HK_CPU_SUB_A9DRAW,
   HK_CPU_SUB_PUSH_DESC,
   HK_CPU_SUB_ROOT,
   HK_CPU_SUB_TEXTURES,
   HK_CPU_SUB_TABLES,
   HK_CPU_SUB_PREPARE_ALLOC,
   HK_CPU_SUB_CMD_BO,
   HK_CPU_SUB_COUNT,
};
uint64_t hk_cpu_sub_begin(void);
void hk_cpu_sub_end(uint64_t start, enum hk_cpu_sub sub);
