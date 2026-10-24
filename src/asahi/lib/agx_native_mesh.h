/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The driver-local v1 advertisement is not a shared DRM UAPI capability. */
#define AGX_NATIVE_MESH_ADVERTISEMENT_MAX 160
/* bits/219 stores generation << 8 | mode in its atomic u64. */
#define AGX_NATIVE_MESH_GENERATION_MAX (UINT64_MAX >> 8)

struct agx_native_mesh_support {
   bool qualified;
   uint64_t generation;
};

struct agx_native_mesh_support
agx_native_mesh_parse(const char *text, size_t size, const char *classes);

struct agx_native_mesh_support
agx_native_mesh_probe(int drm_fd, const char *classes);

static inline bool
agx_native_mesh_same_qualification(struct agx_native_mesh_support before,
                                   struct agx_native_mesh_support after)
{
   return before.qualified && after.qualified &&
          before.generation == after.generation;
}
