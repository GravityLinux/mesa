/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_abi.h"
#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "agx_pack.h"
#include "libagx_dgc.h"

/*
 * To implement drawIndirectCount generically, we dispatch a kernel to
 * clone-and-patch the indirect buffer, predicating out draws as appropriate.
 */
KERNEL(32)
libagx_predicate_indirect(global uint32_t *out, constant uint32_t *in,
                          constant uint32_t *draw_count, uint32_t stride_el,
                          uint indexed__2)
{
   uint draw = cl_global_id.x;
   uint words = indexed__2 ? 5 : 4;
   bool enabled = draw < *draw_count;
   out += draw * words;
   in += draw * stride_el;

   /* Copy enabled draws, zero predicated draws. */
   for (uint i = 0; i < words; ++i) {
      out[i] = enabled ? in[i] : 0;
   }
}

/* CDM indirect-local buffers contain thread counts and local sizes. Shader
 * system values still expose the rounded number of workgroups. */
KERNEL(1)
libagx_workgroups_from_threads(global uint32_t *out, constant uint32_t *grid)
{
   for (uint i = 0; i < 3; ++i) {
      uint group_size = grid[i + 3];
      out[i] = grid[i] / group_size + (grid[i] % group_size != 0);
   }
}

/*
 * Indexing/offseting is in software if necessary so we strip all
 * indexing/offset information.
 */
KERNEL(1)
libagx_draw_without_adj(global VkDrawIndirectCommand *out,
                        global VkDrawIndirectCommand *in,
                        global struct poly_vertex_params *vp,
                        uint64_t index_buffer, uint64_t index_buffer_range_el,
                        int index_size_B, enum mesa_prim prim)
{
   *out = (VkDrawIndirectCommand){
      .vertexCount = libagx_remap_adj_count(in->vertexCount, prim),
      .instanceCount = in->instanceCount,
   };

   /* TODO: Deduplicate */
   if (index_size_B) {
      uint offs = in->firstVertex;

      vp->index_buffer = poly_index_buffer(index_buffer, index_buffer_range_el,
                                           offs, index_size_B);

      vp->index_buffer_range_el =
         poly_index_buffer_range_el(index_buffer_range_el, offs);
   }
}

/*
 * Map the index of a vertex in a primitive list without adjacency to its
 * index in the adjacency topology, as in the VS prolog lowering.
 */
static uint
libagx_map_from_adj(uint id, enum mesa_prim prim, bool flatshade_first)
{
   if (prim == MESA_PRIM_LINES_ADJACENCY) {
      /* (1, 2), (5, 6), (9, 10), ... */
      return ((id & ~1) * 2) + (id & 1) + 1;
   } else if (prim == MESA_PRIM_LINE_STRIP_ADJACENCY) {
      /* (1, 2), (2, 3), (3, 4), ... */
      return (id / 2) + (id & 1) + 1;
   } else if (prim == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      /* (0, 2, 4), (2, 6, 4), (4, 6, 8), (6, 10, 8), ... with the first
       * vertex provoking. With the last vertex provoking, odd triangles are
       * (2, 0, 4), (6, 4, 8), ... so that vertex 2 + 2 * tri stays last.
       */
      uint tri = id / 3;
      uint vtx = id % 3;
      if ((tri % 2) == 1) {
         if (flatshade_first)
            vtx = vtx == 0 ? 0 : 3 - vtx;
         else
            vtx = vtx == 2 ? 2 : 1 - vtx;
      }
      return 2 * (tri + vtx);
   } else {
      /* (0, 2, 4), (6, 8, 10), ... */
      return id * 2;
   }
}

/*
 * Draw adjacency primitives without a geometry shader: only a geometry
 * shader observes the adjacent vertices. Write an index buffer of the
 * remaining vertices into the heap and an indexed indirect draw of it,
 * preserving the vertex offset and instances.
 */
KERNEL(1)
libagx_unroll_adjacency(global uint32_t *out_draw, global struct poly_heap *heap,
                        constant uint32_t *in_draw, uint64_t index_buffer,
                        uint32_t index_buffer_range_el, uint32_t index_size_B,
                        uint32_t prim, uint32_t flatshade_first)
{
   bool indexed = index_size_B != 0;
   uint count = libagx_remap_adj_count(in_draw[0], prim);
   uint offs_B = poly_heap_alloc_offs(heap, count * 4);
   global uint32_t *out = (global uint32_t *)(heap->base + offs_B);

   uint64_t in_indices = 0;
   uint in_range_el = 0;
   if (indexed) {
      in_indices = poly_index_buffer(index_buffer, index_buffer_range_el,
                                     in_draw[2], index_size_B);
      in_range_el = poly_index_buffer_range_el(index_buffer_range_el,
                                               in_draw[2]);
   }

   for (uint i = 0; i < count; ++i) {
      uint raw = libagx_map_from_adj(i, prim, flatshade_first);
      out[i] = indexed ? poly_load_index(in_indices, in_range_el, raw,
                                         index_size_B)
                       : in_draw[2] + raw;
   }

   out_draw[0] = count;
   out_draw[1] = in_draw[1];
   out_draw[2] = offs_B / 4;
   out_draw[3] = indexed ? in_draw[3] : 0;
   out_draw[4] = indexed ? in_draw[4] : in_draw[3];
}

/* Apple9 bounds the hardware index fetch in whole words. Copy the indices of a
 * draw whose binding ends inside a word, reading zero past the binding.
 */
KERNEL(1)
libagx_robust_index_apple9(global uint32_t *out_draw,
                           global struct poly_heap *heap,
                           constant uint32_t *in_draw, uint64_t index_buffer,
                           uint32_t index_buffer_range_el,
                           uint32_t index_size_B)
{
   uint first = in_draw[2];
   uint range_el = poly_index_buffer_range_el(index_buffer_range_el, first);
   uint64_t in = poly_index_buffer(index_buffer, index_buffer_range_el, first,
                                   index_size_B);

   /* Zeroes past a few primitives only repeat the same degenerate one. */
   uint count = min(in_draw[0], range_el + 32);
   uint offs_B = poly_heap_alloc_offs(heap, (count * index_size_B + 3) & ~3);
   uintptr_t out = (uintptr_t)heap->base + offs_B;

   for (uint i = 0; i < count; ++i) {
      poly_store_index(out, index_size_B, i,
                       poly_load_index(in, range_el, i, index_size_B));
   }

   out_draw[0] = count;
   out_draw[1] = in_draw[1];
   out_draw[2] = offs_B / index_size_B;
   out_draw[3] = in_draw[3];
   out_draw[4] = in_draw[4];
}

/* Conditional rendering: copy indirect draw or dispatch arguments, or zero
 * them when the predicate fails, so the command does no work.
 */
KERNEL(1)
libagx_predicate_indirect_apple9(global uint32_t *out, constant uint32_t *in,
                                 constant uint32_t *predicate,
                                 uint32_t inverted, uint32_t words)
{
   bool pass = (*predicate != 0) != (inverted != 0);
   for (uint i = 0; i < words; ++i)
      out[i] = pass ? in[i] : 0;
}

/* Precondition: len must be < the group size */
static void
libagx_memcpy_small(global uchar *dst, constant uchar *src, uint len, uint tid)
{
   if (tid < len) {
      dst[tid] = src[tid];
   }
}

static void
libagx_memcpy_aligned_uint4(global uint *dst, constant uint *src, uint len,
                            uint tid, uint group_size)
{
   for (uint i = tid; i < len; i += group_size) {
      vstore4(vload4(i, src), i, dst);
   }
}

static void
libagx_memcpy_to_aligned(global uint *dst, constant uchar *src, uint len,
                         uint tid, uint group_size)
{
   /* Copy a few bytes at the start */
   uint start_unaligned = ((uintptr_t)src) & 3;
   if (start_unaligned) {
      uint need = 4 - start_unaligned;
      libagx_memcpy_small((global uchar *)dst, src, need, tid);
      src += need;
      len -= need;
   }

   /* Copy a few bytes at the end */
   uint end_unaligned = len & 0xf;
   len -= end_unaligned;
   libagx_memcpy_small(((global uchar *)dst) + len, src + len, end_unaligned,
                       tid);

   /* Now both src and dst are word-aligned, and len is 16-aligned */
   libagx_memcpy_aligned_uint4(dst, (constant uint *)src, len / 16, tid,
                               group_size);
}

/* Precondition: len must be < the group size */
static void
libagx_memset_small(global uchar *dst, uchar b, int len, uint tid)
{
   if (tid < len) {
      dst[tid] = b;
   }
}

/*
 * AGX does not implement robustBufferAccess2 semantics for
 * index buffers, where out-of-bounds indices read as zero. When we
 * dynamically detect index buffer overread (this if-statement), we need
 * to clone the index buffer and zero-extend it to get robustness.
 *
 * We do this dynamically (generating a VDM draw to consume the result) to avoid
 * expensive allocations & memcpys in the happy path where no out-of-bounds
 * access occurs. Otherwise we could use a hardware indirect draw, rather than
 * generating VDM words directly in shader.
 *
 * TODO: Handle multiple draws in parallel.
 */
KERNEL(32)
libagx_draw_robust_index(global uint32_t *vdm, global struct poly_heap *heap,
                         constant VkDrawIndexedIndirectCommand *cmd,
                         uint64_t in_buf_ptr, uint32_t in_buf_range_B,
                         ushort restart, enum agx_primitive topology,
                         enum agx_index_size index_size__3)
{
   uint tid = get_sub_group_local_id();
   bool first = tid == 0;
   enum agx_index_size index_size = index_size__3;

   struct agx_draw draw = agx_draw_indexed(
      cmd->indexCount, cmd->instanceCount, cmd->firstIndex, cmd->vertexOffset,
      cmd->firstInstance, in_buf_ptr, in_buf_range_B, index_size, restart);

   if (agx_indices_to_B(cmd->firstIndex, index_size) >= in_buf_range_B) {
      /* If the entire draw is out-of-bounds, skip it. We handle this specially
       * for both performance and to avoid integer wrapping issues in the main
       * code path (where cmd->firstIndex could get treated as a negative).
       */
      draw.index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw.index_buffer_range_B = 4;
      draw.start = 0;
   } else if (agx_direct_draw_overreads_indices(draw)) {
      constant void *in_buf = (constant void *)agx_draw_index_buffer(draw);
      uint in_size_el = agx_draw_index_range_el(draw);
      uint in_size_B = agx_indices_to_B(in_size_el, index_size);

      /* After a small number of zeroes at the end, extra zeroes cannot change
       * rendering since they will duplicate the same degenerate primitive many
       * times. Therefore we clamp the number of zeroes we need to extend with.
       * This makes the memset constant time.
       */
      draw.b.count[0] = min(draw.b.count[0], in_size_el + 32);

      uint out_size_el = draw.b.count[0];
      uint out_size_B = agx_indices_to_B(out_size_el, index_size);

      /* Allocate memory for the shadow index buffer */
      global uchar *padded;
      if (first) {
         padded = poly_heap_alloc(heap, out_size_B);
      }
      padded = (global uchar *)sub_group_broadcast((uintptr_t)padded, 0);

      draw.index_buffer = (uintptr_t)padded;
      draw.index_buffer_range_B = out_size_B;
      draw.start = 0;

      /* Clone the index buffer. The destination is aligned as a post-condition
       * of poly_heap_alloc_nonatomic.
       */
      libagx_memcpy_to_aligned((global uint *)padded, in_buf, in_size_B, tid,
                               32);

      /* Extend with up to 32 zeroes with a small memset */
      libagx_memset_small(padded + in_size_B, 0, out_size_B - in_size_B, tid);
   }

   if (first) {
      agx_vdm_draw(vdm, 0, draw, topology);
   }
}
