/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "mesh.h"

/* Allocate from the geometry heap, or return zero if it is exhausted. Mesh
 * draws which do not fit draw nothing.
 */
static global uchar *
mesh_heap_alloc(global struct poly_heap *heap, uint64_t size_B)
{
   size_B = align(size_B, 16);
   if (size_B > heap->size)
      return 0;

   uint offs =
      atomic_fetch_add((volatile atomic_uint *)(&heap->bottom), (uint)size_B);
   if ((uint64_t)offs + size_B > heap->size)
      return 0;

   return heap->base + offs;
}

static void
mesh_set_records(global struct libagx_mesh_params *p,
                 global struct poly_heap *heap, uint count,
                 uint group_stride_B, uint vertices_per_group)
{
   p->outputs = mesh_heap_alloc(heap, (uint64_t)count * group_stride_B);
   if (!p->outputs)
      count = 0;

   p->mesh_count = count;
   p->draw[0] = count * vertices_per_group;
   p->draw[1] = 1;
   p->draw[2] = 0;
   p->draw[3] = 0;
}

/* Set up a mesh draw of the workgroup grid at grid, unless it is draw number
 * draw of a count buffer with fewer draws. With a task shader, allocate its
 * payloads and grids; otherwise, the mesh workgroups' records.
 */
KERNEL(1)
libagx_mesh_setup(global struct libagx_mesh_params *p, constant uint *grid,
                  constant uint *count, uint draw, global struct poly_heap *heap,
                  uint has_task, uint payload_B, uint group_stride_B,
                  uint vertices_per_group)
{
   uint x = grid[0], y = grid[1], z = grid[2];
   if (count && draw >= *count)
      x = y = z = 0;

   uint64_t groups = (uint64_t)x * y * z;
   if (groups > (1u << 22))
      x = y = z = groups = 0;

   if (has_task) {
      p->payload_stride_B = align(payload_B, 16);
      p->payloads = mesh_heap_alloc(heap, groups * p->payload_stride_B);
      p->task_grids = (global uint *)mesh_heap_alloc(heap, groups * 16);
      if ((payload_B && !p->payloads) || !p->task_grids)
         x = y = z = groups = 0;

      p->task_count = groups;
      p->task_dispatch[0] = x;
      p->task_dispatch[1] = y;
      p->task_dispatch[2] = z;
   } else {
      p->task_count = 0;
      mesh_set_records(p, heap, groups, group_stride_B, vertices_per_group);
      p->mesh_dispatch[0] = p->mesh_count ? x : 0;
      p->mesh_dispatch[1] = y;
      p->mesh_dispatch[2] = z;
   }
}

/* After the task kernel, sum the mesh workgroups each task workgroup
 * launched and allocate their records.
 */
KERNEL(1024)
libagx_mesh_prefix(global struct libagx_mesh_params *p,
                   global struct poly_heap *heap, uint group_stride_B,
                   uint vertices_per_group)
{
   local uint scratch[32];
   uint tasks = p->task_count;
   poly_prefix_sum(scratch, p->task_grids, tasks, 4 /* words */, 3 /* word */,
                   1024);

   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
   if (cl_local_id.x != 0)
      return;

   uint total = tasks ? p->task_grids[(tasks - 1) * 4 + 3] : 0;
   mesh_set_records(p, heap, total, group_stride_B, vertices_per_group);
   p->mesh_dispatch[0] = p->mesh_count;
   p->mesh_dispatch[1] = 1;
   p->mesh_dispatch[2] = 1;
}

/* Native children need no emulated output records. The CPU proves the sum
 * fits a one-dimensional native grid before recording any task work, and
 * allocates independent task records in the command pool. Keep payloads and
 * task-grid records until the whole native render retires. */
KERNEL(1024)
libagx_mesh_prefix_native(global struct libagx_mesh_params *p)
{
   local uint scratch[32];
   uint tasks = p->task_count;
   poly_prefix_sum(scratch, p->task_grids, tasks, 4, 3, 1024);
   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
   if (cl_local_id.x != 0)
      return;
   uint total = tasks ? p->task_grids[(tasks - 1) * 4 + 3] : 0;
   p->mesh_count = total;
   p->mesh_dispatch[0] = max(total, 1u);
   p->mesh_dispatch[1] = 1;
   p->mesh_dispatch[2] = 1;
}
