/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_primitive_count.h"
#include "asahi/lib/agx_apple9.h"

#include "util/macros.h"

#include "util/list.h"
#include "agx_abi.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"
#include "agx_uvs.h"
#include "libagx_dgc.h"
#include "pool.h"
#include "shader_enums.h"

#include "hk_private.h"
#include "hk_shader.h"
#include "hk_sync.h"

#include "hk_cmd_pool.h"
#include "hk_upload.h"
#include "hk_descriptor_set.h"

#include "asahi/lib/agx_nir_lower_vbo.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

#include "vk_command_buffer.h"

#include <stdio.h>

/* Occlusion queries active at once, and segment slots per control stream. */
#define HK_MAX_ACTIVE_OCCLUSION_QUERIES 64
#define HK_OCCLUSION_SEGMENTS           512

struct hk_buffer;
struct hk_cmd_bo;
struct hk_cmd_pool;
struct agx_apple9_compute_profile;
struct hk_image_view;
struct hk_push_descriptor_set;
struct hk_shader;
struct hk_linked_shader;
struct agx_usc_builder;
struct vk_shader;

/** Root descriptor table. */
struct hk_root_descriptor_table {
   /* Address of this descriptor itself. Must be first for reflection. */
   uint64_t root_desc_addr;

   /* Descriptor set base addresses. Must follow root_desc_addr to match our
    * push layout.
    */
   uint64_t sets[HK_MAX_SETS];

   union {
      struct {
         uint32_t view_index;
         uint32_t ppp_multisamplectl;
         uint32_t draw_id;
         uint32_t _pad0;

         /* Vertex input state */
         uint64_t attrib_base[AGX_MAX_VBUFS];
         uint32_t attrib_clamps[AGX_MAX_VBUFS];
         uint32_t attrib_strides[AGX_MAX_VBUFS];

         /* Mask of outputs flowing VS->TCS, VS->GS, or TES->GS . */
         uint64_t vertex_outputs;

         /* Address of vertex param buffer if geom/tess is used, else 0 */
         uint64_t vertex_params;

         /* Address of tessellation param buffer if tessellation used, else 0 */
         uint64_t tess_params;

         /* Address of geometry param buffer if GS is used, else 0 */
         uint64_t geometry_params;

         /* struct libagx_mesh_params of a mesh shader draw */
         uint64_t mesh_params;

         /* Pipeline statistics queries. This is a base address with flags. */
         uint64_t pipeline_stats;
         VkQueryPipelineStatisticFlags pipeline_stats_flags;

         float blend_constant[4];
         uint16_t no_epilog_discard;
         uint16_t _pad1;
         uint16_t api_sample_mask;
         uint16_t _pad2;
         uint16_t force_never_in_shader;
         uint16_t _pad3;
         uint16_t provoking;
         /* Rasterized primitive class for per-vertex fragment inputs:
          * 0 triangles, 1 lines, 2 points; bit 2 if the rasterizer's vertex
          * order is the API order rotated (first-provoking fans). */
         uint16_t rasterized_prim;

         /* True if there is an API geometry shader. If false, there may still
          * be a geometry shader in use (notably for transform feedback) but it
          * should not contribute to pipeline statistics.
          */
         uint16_t api_gs;
         uint16_t _pad5;

         uint16_t rasterization_stream;
         uint16_t _pad6;

         /* Fragment depth exports clamp to this range: viewport 0's depth
          * range, or the user range, with depth clamping; [0, 1] without.
          */
         float depth_clamp[2];

         /* Mapping from varying slots written by the last vertex stage to UVS
          * indices. This mapping must be compatible with the fragment shader.
          */
         uint8_t uvs_index[VARYING_SLOT_MAX];
      } draw;
      struct {
         uint64_t group_count_addr;
         uint32_t base_group[3];
      } cs;
   };

   /* Client push constants */
   uint8_t push[HK_MAX_PUSH_SIZE];

   /* Start index in dynamic_buffers where each set starts */
   uint8_t set_dynamic_buffer_start[HK_MAX_SETS];

   /* Dynamic buffer bindings. Last, so that a root upload can stop after the
    * slots in use, see hk_cmd_buffer_upload_root. */
   struct hk_buffer_address dynamic_buffers[HK_MAX_DYNAMIC_BUFFERS];
};

/* Bumped whenever hk_root_descriptor_table offsets change: compiled shaders
 * bake them, so pipeline and disk cache keys include it. */
#define HK_ROOT_LAYOUT_VERSION 2

/* helper macro for computing root descriptor byte offsets */
#define hk_root_descriptor_offset(member)                                      \
   offsetof(struct hk_root_descriptor_table, member)

struct hk_descriptor_state {
   bool root_dirty;
   struct hk_root_descriptor_table root;

   /* VK_EXT_descriptor_buffer: bound buffer addresses, indexed by
    * vkCmdSetDescriptorBufferOffsets bufferIndices. */
   uint64_t descriptor_buffers[HK_MAX_SETS];
   /* Sets bound from descriptor memory rather than a set object or push
    * descriptors. */
   uint32_t descriptor_buffer_sets;

   /* Last uploaded root (self-address normalized to 0) and its GPU address.
    * An identical root reuses the upload instead of hashing into the
    * per-command-buffer upload cache. Zeroed with the rest of the state on
    * reset, i.e. whenever the upload pool is. */
   /* Dynamic buffer slots bound since the last reset: roots are uploaded up
    * to the end of these. */
   uint32_t root_dynamic_count;

   bool root_last_valid;
   uint32_t root_last_size;
   uint64_t root_last_gpu;
   struct hk_root_descriptor_table root_last;

   /* Last direct-dispatch group count upload, reused for identical counts so
    * the root (which holds its address) stays comparable across dispatches. */
   bool group_count_last_valid;
   uint32_t group_count_last[3];
   uint64_t group_count_last_addr;

   uint32_t set_sizes[HK_MAX_SETS];
   struct hk_descriptor_set *sets[HK_MAX_SETS];
   uint32_t sets_dirty;

   struct hk_push_descriptor_set *push[HK_MAX_SETS];
   uint32_t push_dirty;
};

struct hk_attachment {
   VkFormat vk_format;
   struct hk_image_view *iview;

   VkResolveModeFlagBits resolve_mode;
   struct hk_image_view *resolve_iview;
   VkRenderingAttachmentFlagsKHR flags;

   bool clear;
   uint32_t clear_colour[4];
};

struct hk_bg_eot {
   uint64_t usc;
   struct agx_counts_packed counts;
};

struct hk_render_registers {
   uint32_t width, height, layers;
   uint32_t zls_width, zls_height;
   uint32_t isp_bgobjdepth;
   uint32_t isp_bgobjvals;
   struct agx_zls_control_packed zls_control, zls_control_partial;
   uint32_t depth_dimensions;
   bool process_empty_tiles;
   enum u_tristate dbias_is_int;

   struct {
      uint32_t dimensions;
      uint64_t buffer, meta;
      uint32_t stride, meta_stride;
   } depth;

   struct {
      uint64_t buffer, meta;
      uint32_t stride, meta_stride;
   } stencil;

   struct {
      struct hk_bg_eot main;
      struct hk_bg_eot partial;
   } bg;

   struct {
      struct hk_bg_eot main;
      struct hk_bg_eot partial;
   } eot;
};

struct hk_rendering_state {
   /* Stages that may depend on the open render this pass continued. */
   VkPipelineStageFlags2 merge_reach;
   /* Colour attachments this render pass stores (bit per attachment). */
   uint32_t store_mask;
   /* An attachment is in a layout that also allows shader access (GENERAL
    * or a feedback loop): passes may read it as a texture. */
   bool feedback;

   /* Depth and stencil loads of a suspend/resume chain continued in one
    * hardware render: those of its first render pass. */
   bool chain_load[2];
   /* This render pass resumed a suspended one in its render. */
   bool resumed;
   /* A stored colour attachment is linear, so buffers aliasing its memory
    * can observe the render's writes (hk_queue's tiling-wait elision). */
   bool linear_store;
   VkRenderingFlagBits flags;

   VkRect2D area;
   uint32_t layer_count;
   uint32_t view_mask;

   uint32_t color_att_count;
   struct hk_attachment color_att[HK_MAX_RTS];
   struct hk_attachment depth_att;
   struct hk_attachment stencil_att;

   struct agx_tilebuffer_layout tilebuffer;
   struct hk_render_registers cr;

   /* Control streams other than the last store depth/stencil for the next
    * one to load. The last control stream honours the store operations:
    * zls_last if it is also the first, zls_partial_last otherwise. */
   struct hk_cs *first_cs;
   struct agx_zls_control_packed zls_last, zls_partial_last;
};

struct hk_index_buffer_state {
   struct hk_addr_range buffer;
   enum agx_index_size size;
   uint32_t restart;
};

/* Dirty tracking bits for state not tracked by vk_dynamic_graphics_state or
 * shaders_dirty.
 */
enum hk_dirty {
   HK_DIRTY_VB = BITFIELD_BIT(0),
   HK_DIRTY_OCCLUSION = BITFIELD_BIT(1),
   HK_DIRTY_PROVOKING = BITFIELD_BIT(2),
   HK_DIRTY_VARYINGS = BITFIELD_BIT(3),
};

struct hk_graphics_state {
   struct hk_rendering_state render;
   struct hk_descriptor_state descriptors;

   enum hk_dirty dirty;

   uint64_t root;
   uint64_t draw_params;
   uint64_t draw_id_ptr;

   uint32_t shaders_dirty;
   struct hk_api_shader *shaders[MESA_SHADER_MESH + 1];

   /* Graphics shaders bound through the API. On Apple9, deferred shaders are
    * resolved from these into concrete variants in shaders[] at draw time.
    */
   struct hk_api_shader *apple9_api[HK_APPLE9_GFX_STAGES];

   /* Last resolution of apple9_api, reused while none of the graphics key's
    * inputs change. render_gen counts render pass instances. */
   struct {
      bool valid;
      uint32_t render_gen;
      struct hk_api_shader *api[HK_APPLE9_GFX_STAGES];
      struct hk_api_shader *out[HK_APPLE9_GFX_STAGES];
   } apple9_resolved;
   uint32_t render_gen;

   /* The current draw is a last-provoking fan unrolled to triangles. */
   bool apple9_fan_unrolled;
   /* The bound vertex shader writes barycentric weights for this draw, a
    * non-indexed triangle list, without the passthrough geometry shader. */
   bool apple9_weights_by_vs;

   /* Vertex buffers */
   struct hk_addr_range vb[AGX_MAX_VBUFS];

   /* Transform feedback buffers */
   struct hk_addr_range xfb[4];

   /* Is transform feedback enabled? */
   bool xfb_enabled;

   /* Internal transform feedback offset vec4.
    *
    * TODO: Strictly could be global.
    */
   uint64_t xfb_offsets;

   /* Pointer to the GPU memory backing active transform feedback queries,
    * per-stream. Zero if no query is bound.
    */
   uint64_t xfb_query[4];

   /* GPU counters of active primitives generated queries, per stream. They
    * count the primitives of the passthrough or API geometry stage. */
   uint64_t pg_query[4];

   /* Mesh primitives generated query */
   uint64_t mesh_pg_query;

   /* Input assembly statistics of direct draws not yet added to the
    * counters at addr: vertices (IA vertices and VS invocations), primitives
    * (IA primitives) and clipper primitives (clipping invocations and
    * primitives, which geometry and tessellation count themselves). See
    * hk_flush_ia_stats. */
   struct hk_ia_stats {
      uint64_t addr[5];
      uint32_t vertices, prims, clipped;
   } ia_stats;

   /* Apple9 packed sample positions of the current draw. */
   uint32_t apple9_sample_positions;
   /* The runtime sysval block of the last draw and the per-draw inputs it
    * was built from, see hk_apple9_upload_textures. */
   uint64_t apple9_sysvals;
   uint32_t apple9_sysvals_positions;
   bool apple9_sysvals_first_fan;

   struct hk_index_buffer_state index;
   enum agx_primitive topology;
   enum agx_object_type object_type;

   /* Provoking vertex 0, 1, or 2. Usually 0 or 2 for FIRST/LAST. 1 can only be
    * set for tri fans.
    */
   uint8_t provoking;

   struct {
      /* Mode and hardware index the next draws use. */
      enum agx_visibility_mode mode;
      uint16_t index;

      /* Several occlusion queries may be active at once (vkd3d-proton's
       * virtual queries rely on it), but a draw counts into one visibility
       * slot. Draws therefore count into a per-render segment slot, and each
       * segment is added into every query active while it counted (see
       * hk_occlusion_prepare_draw). A new segment starts whenever the active
       * set changes or a new control stream begins.
       */
      bool segment_dirty;
      bool paused;
      enum agx_visibility_mode segment_mode;
      unsigned active_count;
      struct {
         uint64_t report;
         bool precise;
      } active[HK_MAX_ACTIVE_OCCLUSION_QUERIES];
   } occlusion;

   /* Fast linked shader data structures */
   uint64_t varyings;
   struct agx_varyings_vs linked_varyings;

   uint32_t linked_dirty;
   struct hk_linked_shader *linked[MESA_SHADER_STAGES];
   bool generate_primitive_id;

   /* Whether blend constants are required by the active blend state */
   bool uses_blend_constant;

   /* Tessellation state */
   struct {
      uint64_t out_draws;
      uint64_t grids;
      struct hk_tess_info info;
      enum mesa_prim prim;
   } tess;

   /* Needed by vk_command_buffer::dynamic_graphics_state */
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
};

struct hk_compute_state {
   struct hk_descriptor_state descriptors;
   struct hk_api_shader *shader;
};

struct hk_cmd_push {
   void *map;
   uint64_t addr;
   uint32_t range;
   bool no_prefetch;
};

struct hk_scratch_req {
   bool main;
   bool preamble;
};

/*
 * Represents a firmware timestamp request.  Handle is a kernel timestamp object
 * handle, not a GEM handle.
 *
 * The kernel/firmware uses the handle/offset_B to write. We use the address to
 * read the results back. We could deduplicate this, but this is convenient.
 */
struct agx_timestamp_req {
   uint64_t addr;
   uint32_t handle;
   uint32_t offset_B;
};

/*
 * hk_cs represents a single control stream, to be enqueued either to the
 * CDM or VDM for compute/3D respectively.
 */
enum hk_cs_type {
   HK_CS_CDM,
   HK_CS_VDM,
};

/* Why a control stream ended (HK_TIMELINE_FILE per-command records). */
enum hk_cs_end {
   HK_CS_END_UNKNOWN = 0,
   HK_CS_END_CMDBUF = 1,      /* vkEndCommandBuffer */
   HK_CS_END_RENDER_PASS = 2, /* vkCmdEndRendering */
   HK_CS_END_BARRIER = 3,     /* pipeline barrier that split the stream */
   HK_CS_END_EVENT = 4,       /* set/reset/wait event */
   HK_CS_END_QUERY = 5,       /* timestamp or occlusion segment */
   HK_CS_END_RENDER_SPLIT = 6,/* scissor/bias/state overflow, render split */
   HK_CS_END_BEGIN_RENDER = 7,/* compute ended by vkCmdBeginRendering */
   HK_CS_END_PASS_SPLIT = 8,  /* HK_GPU_PASS_SPLIT */
};

struct hk_cs {
   struct list_head node;

   /* GPU pass attribution (diagnostic): first shader, its grid, and count. */
   uint64_t pass_code;
   uint32_t pass_grid[3];
   uint32_t pass_count;

   /* Render merging: a draw of this stream writes memory, and an image
    * barrier was hoisted ahead of this stream (hk_cmd_sync_barrier). */
   bool merge_writes, merge_hoisted;

   /* enum hk_pre_raster_reads of the stream's draws: what their
    * pre-rasterization stages read (hk_queue's tiling-wait elision). */
   uint8_t pre_raster_reads;
   /* The stream's render stores a linear colour attachment. */
   bool linear_store;

   /* enum hk_cs_end, and whether an hk meta operation created the stream. */
   uint8_t end_reason;
   bool meta;

   /* Parent command buffer. Convenience. */
   struct hk_cmd_buffer *cmd;

   /* Data master */
   enum hk_cs_type type;

   uint32_t sync_id;
   unsigned sync_stages;
   bool cdm_dirty;

   /* The last dispatch ended with the compute-only barrier: later compute
    * dispatches are ordered after it, but anything else (indirect grid
    * fetches, the stream end, non-Apple9 launches) needs a full flush first.
    * cdm_dirty stays set so those paths flush.
    */
   bool cdm_compute_barrier;

   /* Address of the root control stream for the job */
   uint64_t addr;

   /* Fat pointer to the start of the current chunk of the control stream */
   struct agx_ptr chunk;

   /* Start pointer of the root control stream */
   void *start;

   /* Size of the root allocation, retained when continuation blocks are added. */
   size_t root_size;

   /* Occlusion segment slots for this graphics control stream, allocated on
    * the first counting draw, and the number used so far. */
   uint64_t occlusion_base;
   uint32_t occlusion_segments;

   /* Current pointer within the control stream */
   void *current;

   /* End pointer of the current chunk of the control stream */
   void *end;

   /* Whether there is more than just the root chunk */
   bool stream_linked;

   /* Whether the sampler heap is required. Although we always must maintain the
    * heap for correctness, it's often not necessary since we can push lots of
    * samplers (especially for GL/DX11-era engines).
    */
   bool uses_sampler_heap;

   /* Scratch requirements */
   struct {
      union {
         struct hk_scratch_req vs;
         struct hk_scratch_req cs;
      };

      struct hk_scratch_req fs;
   } scratch;

   /* Immediate writes, type libagx_imm_write. These all happen in parallel at
    * the end of the control stream. This accelerates queries. Implies CDM.
    */
   struct util_dynarray imm_writes;

   /* Statistics */
   struct {
      uint32_t calls, cmds, flushes, merged;
   } stats;

   /* Timestamp writes. Currently just compute end / fragment end. We could
    * flesh this out later if we want finer info. (We will, but it's not
    * required for conformance.)
    */
   struct {
      struct agx_timestamp_req end;
   } timestamp;

   /* Remaining state is for graphics only, ignored for compute */
   uint8_t primitive_count; /* Saturated at AGX_PRIMITIVE_COUNT_UNKNOWN. */
   struct agx_tilebuffer_layout tib;

   struct util_dynarray scissor, depth_bias;
   uint64_t uploaded_scissor, uploaded_zbias;

   /* We can only set ppp_multisamplectl once per batch. has_sample_locations
    * tracks if we've committed to a set of sample locations yet. vk_meta
    * operations do not set has_sample_locations since they don't care and it
    * would interfere with the app-provided samples.
    *
    */
   bool has_sample_locations;
   uint32_t ppp_multisamplectl;

   struct hk_render_registers cr;

   /* Active restart index if one is set. Zero if there is no restart index set
    * yet, since Vulkan does not allow zero restart indices (unlike OpenGL).
    * This is used in place of dirty tracking, because dirty tracking
    * restart indices is complicated and just checking the saved value is cheap.
    */
   uint32_t restart_index;

   /* HK_TIMELINE_FILE (TL_RENDER_INFO): what this render stream targets. */
   struct {
      uint64_t color0, depth;          /* image identity */
      uint32_t area;                   /* width | height << 16 */
      uint32_t view;                   /* base mip | base layer << 8 | layers << 20 */
      uint32_t color0_format, depth_format;
      uint8_t color_count, samples;
      /* bit 0 colour-0 cleared, 1 depth cleared, 2 colour-0 compressed,
       * 3 colour-0 storage usage, 4 depth compressed */
      uint8_t clears;
   } diag;
};

/*
 * Helper to merge two compute control streams, concatenating the second control
 * stream to the first one. Must sync with hk_cs.
 */
static inline void
hk_cs_merge_cdm(struct hk_cs *a, const struct hk_cs *b)
{
   assert(a->type == HK_CS_CDM && b->type == HK_CS_CDM);
   assert(a->cmd == b->cmd);
   assert(!a->timestamp.end.handle);

   agx_cdm_jump(a->current, b->addr);
   a->current = b->current;
   a->stream_linked = true;

   a->uses_sampler_heap |= b->uses_sampler_heap;
   a->scratch.cs.main |= b->scratch.cs.main;
   a->scratch.cs.preamble |= b->scratch.cs.preamble;

   a->timestamp = b->timestamp;

   a->stats.calls += b->stats.calls;
   a->stats.cmds += b->stats.cmds;
   a->stats.flushes += b->stats.flushes;
   a->stats.merged++;
}

static inline uint64_t
hk_cs_current_addr(struct hk_cs *cs)
{
   return cs->chunk.gpu + ((uint8_t *)cs->current - (uint8_t *)cs->chunk.cpu);
}

struct hk_uploader {
   /** List of hk_cmd_bo */
   struct list_head bos;

   struct hk_upload_chunk chunks[HK_UPLOAD_CHUNKS];
};

#define HK_TESS_BATCH_LANES 5

enum hk_upload_last {
   HK_UPLOAD_LAST_DRAW_PARAMS,
   HK_UPLOAD_LAST_VERTEX_BUFFERS,
   HK_UPLOAD_LAST_VERTEX_TABLE,
   HK_UPLOAD_LAST_FRAGMENT_TABLE,
   HK_UPLOAD_LAST_COUNT,
};

/* Identity of a render pass's attachments, for HK_APPLE9_MERGE_RENDERS. */
struct hk_render_merge_key {
   const struct vk_image *image[HK_MAX_RTS];
   uint32_t level[HK_MAX_RTS];
   uint32_t layer[HK_MAX_RTS];
   VkFormat format[HK_MAX_RTS];
   uint32_t color_count, layer_count, view_mask;
};

struct hk_cmd_buffer {
   struct vk_command_buffer vk;

   struct {
      struct hk_graphics_state gfx;
      struct hk_compute_state cs;

      /* VK_EXT_conditional_rendering: draws, dispatches and attachment
       * clears run only when the 32-bit predicate is non-zero (zero if
       * inverted). Meta operations other than clears are unpredicated.
       */
      struct {
         uint64_t predicate;
         bool active, inverted, meta;
      } cond_render;
   } state;

   struct hk_uploader uploader[HK_CMD_POOL_COUNT];

   /* Full-content immutable upload cache, discarded on reset. */
   struct hash_table *upload_cache;
   /* HK_UPLOAD_STATS: content-cache lookups per record kind. */
   uint32_t upload_hits[AGX_APPLE9_STATE_COUNT + 3], upload_misses[AGX_APPLE9_STATE_COUNT + 3];

   /* Per-draw tables whose contents usually change from one draw to the next
    * (they carry the root address): the last upload of each kind, reused only
    * when the next is identical, see hk_pool_upload_last. */
   struct hk_upload_last_entry {
      uint64_t gpu;
      uint32_t size;
      /* The largest table: vertex buffers or a stage's resource table. */
      uint8_t data[MAX2(AGX_MAX_VBUFS * 32,
                        AGX_APPLE9_MAX_GRAPHICS_BUFFERS * sizeof(uint64_t))];
   } upload_last[HK_UPLOAD_LAST_COUNT];

   /* The current and previous Apple9 draw records, alternating: a draw
    * compares against the previous draw in the same graphics control stream,
    * so an unchanged launch, coefficient table or PPP record is reused
    * without hashing. Records are allocated per control stream (batch). */
   struct agx_apple9_uniform_draw apple9_draw[2];
   unsigned apple9_draw_current;
   struct hk_cs *apple9_prev_cs;

   /* The last linked Apple9 (vertex, fragment) pair, by program id: shader
    * objects pair stages only at draw time, so link once per change. */
   struct agx_apple9_render_pipeline apple9_linked;
   uint64_t apple9_linked_id[2];

   /* The null fragment stage carrying the vertex stage's varying layout,
    * for the vertex program apple9_null_fs_vertex_id. */
   struct agx_apple9_render_stage apple9_null_fs;
   uint64_t apple9_null_fs_vertex_id;
   bool apple9_null_fs_trivial;

   /* List of all recorded control streams */
   struct list_head control_streams;

   /* Immutable after recording; resolved afresh for every queue submission. */
   struct util_dynarray sync_events;
   struct util_dynarray sync_remap;
   uint32_t sync_next_group;
   unsigned sync_scope;
   unsigned sync_render_stages;
   bool sync_render_continuation;
   bool sync_fallback;
   bool api_dispatch;

   /* HK_APPLE9_MERGE_RENDERS: the control stream of a render pass that has
    * ended but is kept open, so the next render pass on the same attachments
    * can continue it. Ending graphics, or any command other than a
    * render-pass-internal barrier, closes it (hk_cmd_render_merge_flush).
    */
   struct hk_cs *merge_cs;
   struct hk_render_merge_key merge_key;
   /* sync_events index of merge_cs's first event, where barriers recorded
    * between the merged passes go. */
   unsigned merge_event;
   bool merge_first_cs;
   /* Colour attachments the open render's passes stored: a continuing pass
    * must store them too, or the merged render would drop those stores. */
   uint32_t merge_store_mask;
   bool merge_feedback;
   /* The open render is a suspended render pass, which only its resuming
    * render pass may continue (HK_APPLE9_MERGE_RESUME). */
   bool merge_suspended;
   bool merge_chain_load[2];
   /* Compute or transfer work recorded while a render is kept open, which
    * nothing makes depend on the render, runs in this stream placed before
    * it. merge_reach: the stages that may depend on the open render (its
    * graphics stages and the destinations of barriers from them);
    * merge_move_stage: the stage of the command being recorded, if it may
    * move ahead of the render. */
   struct hk_cs *merge_pre_cs;
   VkPipelineStageFlags2 merge_reach, merge_move_stage;
   /* The kept-open render has no draws yet: its render state, which the
    * empty-render optimization needs if it ends without being continued. */
   struct hk_rendering_state *merge_empty_render;
   bool merge_empty;
   /* The image barrier being recorded by vkCmdPipelineBarrier2, if any, and
    * whether it transfers queue-family ownership. */
   struct hk_image *barrier_image;
   bool barrier_qfot;

   /* Current recorded control stream */
   struct {
      /* VDM stream for 3D */
      struct hk_cs *gfx;

      /* CDM stream for compute */
      struct hk_cs *cs;

      /* CDM stream that executes immediately before the current graphics
       * control stream. Used for geometry shading, tessellation, etc.
       */
      struct hk_cs *pre_gfx;

      /* CDM stream that will execute after the current graphics control stream
       * finishes. Used for queries.
       */
      struct hk_cs *post_gfx;
   } current_cs;

   /* Pre-rasterization dispatch records of batched tessellated draws, one
    * lane per stage (VS, TCS, count, prefix sum, emit). The draws are
    * independent, so each lane's records run without barriers between them
    * and the lanes are emitted in order into the pre-gfx stream with one
    * compute barrier after each (hk_tess_batch_flush). capture, if set,
    * redirects hk_apple9_dispatch into a lane.
    */
   struct {
      struct util_dynarray lane[HK_TESS_BATCH_LANES];
      struct util_dynarray *capture;
      unsigned draws;
   } tess_batch;

   /* Are we currently inside a vk_meta operation? This alters sample location
    * behaviour.
    */
   bool in_meta;

   /* enum hk_cs_end for the next stream end (timeline diagnostics). */
   uint8_t end_hint;

   /* XXX: move me?
    *
    * Indirect draw generated by the indirect GS translator.
    */
   uint64_t geom_index_buffer;
   uint32_t geom_index_count;
   uint32_t geom_instance_count;

   /* Does the command buffer use the geometry heap? */
   bool uses_heap;
   bool uses_mesh_heap;

   /* Query counter increments and availability writes (struct
    * libagx_imm_write) that are due once earlier rendering completes. They
    * are applied together before the queries can next be observed, rather
    * than splitting rendering with a compute dispatch after every render
    * pass. See hk_flush_query_writes. */
   struct util_dynarray query_adds, query_writes;

   /* Occlusion segment accumulations (struct libagx_occlusion_accum), applied
    * with the query writes. */
   struct util_dynarray occlusion_accum;

   /* Owned large BOs */
   struct util_dynarray large_bos;

   /* Virtio resource ids (uint32_t) of implicit-sync (WSI) memory this
    * recording accesses; submissions list them as external resources. */
   struct util_dynarray implicit_sync_res;

   /* Apple9 shader BOs whose arena entries this recording may encode, each
    * referenced and pinned once until reset, and the last two pinned. */
   struct set *apple9_pinned;
   struct agx_bo *apple9_last_pin[2];
};

VK_DEFINE_HANDLE_CASTS(hk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops hk_cmd_buffer_ops;

static inline struct hk_cmd_pool *
hk_cmd_buffer_pool(struct hk_cmd_buffer *cmd)
{
   return (struct hk_cmd_pool *)cmd->vk.pool;
}

/*
 * The hardware vertex shader is supplied by the last geometry stage. The
 * geometry pipeline is vertex->tess->geometry so we search backwards.
 */
static inline struct hk_shader *
hk_bound_hw_vs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   struct hk_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];

   if (mesh)
      return &mesh->variants[HK_MESH_VARIANT_RAST];
   else if (gs)
      return &gs->variants[HK_GS_VARIANT_RAST];
   else if (tes)
      return &tes->variants[HK_VS_VARIANT_HW];
   else
      return &vs->variants[HK_VS_VARIANT_HW];
}

static inline struct hk_shader *
hk_bound_sw_vs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);

   if (hw_vs == &vs->variants[HK_VS_VARIANT_HW])
      return hw_vs;
   else
      return &vs->variants[HK_VS_VARIANT_SW];
}

static inline struct hk_shader *
hk_bound_sw_vs_before_gs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct hk_api_shader *api = tes ?: vs;

   return &api->variants[HK_VS_VARIANT_SW];
}

struct agx_ptr hk_pool_alloc_internal(struct hk_cmd_buffer *cmd, uint32_t size,
                                      uint32_t alignment, enum hk_cmd_pool_kind kind);

uint64_t hk_pool_upload(struct hk_cmd_buffer *cmd, const void *data,
                        uint32_t size, uint32_t alignment);

enum hk_upload_type {
   HK_UPLOAD_RUNTIME = AGX_APPLE9_STATE_COUNT,
   HK_UPLOAD_ROOT,
   HK_UPLOAD_TABLE,
};

uint64_t hk_pool_upload_cached(struct hk_cmd_buffer *cmd, const void *data,
                               uint32_t size, uint32_t alignment,
                               enum hk_upload_type type);
uint64_t hk_pool_upload_last(struct hk_cmd_buffer *cmd, enum hk_upload_last slot,
                             const void *data, uint32_t size,
                             uint32_t alignment);
struct agx_ptr hk_apple9_state_cache_alloc(
   void *allocator, enum agx_apple9_state_type type, const void *contents,
   size_t size, size_t alignment, bool *is_new);
struct agx_ptr hk_apple9_state_uncached_alloc(
   void *allocator, enum agx_apple9_state_type type, const void *contents,
   size_t size, size_t alignment, bool *is_new);

static inline struct agx_ptr
hk_pool_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return hk_pool_alloc_internal(cmd, size, alignment, HK_CMD_POOL_MAIN);
}

static inline struct agx_ptr
hk_pool_usc_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return hk_pool_alloc_internal(cmd, size, alignment, HK_CMD_POOL_USC);
}

static inline struct agx_ptr
hk_pool_context_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return hk_pool_alloc_internal(cmd, size, alignment, HK_CMD_POOL_CONTEXT);
}

static inline struct agx_ptr
hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment,
                 bool compute)
{
   return !compute && hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G16G
             ? hk_pool_context_alloc(cmd, size, alignment)
             : hk_pool_alloc(cmd, size, alignment);
}

void hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_cmd_sync_work(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                      unsigned stages);
void hk_cmd_sync_barrier(struct hk_cmd_buffer *cmd,
                         VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst,
                         VkAccessFlags2 src_access, VkAccessFlags2 dst_access);
void hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs);

static inline void
hk_cmd_use_implicit_sync(struct hk_cmd_buffer *cmd, uint32_t res)
{
   if (!res)
      return;
   util_dynarray_foreach(&cmd->implicit_sync_res, uint32_t, r) {
      if (*r == res)
         return;
   }
   util_dynarray_append(&cmd->implicit_sync_res, res);
}
void hk_cdm_compute_barrier(struct hk_device *dev, struct hk_cs *cs);
uint32_t hk_default_sample_positions(unsigned nr_samples);
struct vk_sample_locations_state;
uint32_t hk_pack_ppp_multisamplectrl(const struct vk_sample_locations_state *sl);

/* The pipeline stages a render pass instance runs in. */
static inline VkPipelineStageFlags2
hk_merge_graphics_stages(void)
{
   return VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
          VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
          VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
          VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
          VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
          VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
          VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
          VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
          VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
          VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
          VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
          VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT |
          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
          VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
          VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR |
          VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT |
          VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
}

void hk_cmd_render_merge_flush_why(struct hk_cmd_buffer *cmd, const char *why);
struct hk_cs *hk_render_merge_pre_cs(struct hk_cmd_buffer *cmd, unsigned stages);
void hk_render_merge_trace(const char *why, uint64_t a, uint64_t b);
bool hk_render_merge_move_enabled(void);
#define hk_cmd_render_merge_flush(cmd) hk_cmd_render_merge_flush_why(cmd, __func__)

static inline struct hk_cs *
hk_cmd_buffer_get_cs_general(struct hk_cmd_buffer *cmd, struct hk_cs **ptr,
                             bool compute, unsigned stages)
{
   /* Work recorded outside a render pass ends a render kept open for
    * merging. A merged render pass takes the stream before drawing. */
   if (unlikely(cmd->merge_cs) && ptr != &cmd->merge_pre_cs) {
      if (compute && ptr == &cmd->current_cs.cs && cmd->merge_move_stage &&
          !cmd->merge_suspended &&
          !(cmd->merge_move_stage & cmd->merge_reach) &&
          hk_render_merge_move_enabled())
         return hk_render_merge_pre_cs(cmd, stages);
      hk_cmd_render_merge_flush(cmd);
   }

   if ((*ptr) == NULL) {
      /* Most control streams are short. Reserve only a small root block;
       * hk_ensure_cs_has_space() chains larger blocks for long streams.
       */
      size_t initial_size = 4096;
      struct agx_ptr root = hk_pool_cs_alloc(cmd, initial_size, 1024, compute);
      if (!root.cpu)
         return NULL;

      /* Allocate hk_cs for the new stream */
      struct hk_cs *cs = malloc(sizeof(*cs));
      if (!cs) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
      *cs = (struct hk_cs){
         .cmd = cmd,
         .type = compute ? HK_CS_CDM : HK_CS_VDM,
         .addr = root.gpu,
         .start = root.cpu,
         .root_size = initial_size,
         .chunk = root,
         .current = root.cpu,
         .end = root.cpu + initial_size,
         .sync_id = util_dynarray_num_elements(&cmd->sync_remap, uint32_t),
         .meta = cmd->in_meta,
      };

      util_dynarray_append(&cmd->sync_remap, cs->sync_id);

      list_inithead(&cs->node);

      bool before_gfx = (ptr == &cmd->current_cs.pre_gfx);
      /* These helpers are physically reordered around rendering. Until their
       * logical operations are split out, preserve the existing full ordering.
       */
      cmd->sync_fallback |= before_gfx || ptr == &cmd->current_cs.post_gfx;

      /* Insert into the command buffer. We usually append to the end of the
       * command buffer, except for pre-graphics streams which go right before
       * the graphics workload. (This implies a level of out-of-order processing
       * that's allowed by Vulkan and required for efficient
       * geometry/tessellation shaders.)
       */
      if ((before_gfx || ptr == &cmd->merge_pre_cs) && cmd->current_cs.gfx) {
         list_addtail(&cs->node, &cmd->current_cs.gfx->node);
      } else {
         list_addtail(&cs->node, &cmd->control_streams);
      }

      *ptr = cs;

      if (!compute)
         hk_cs_init_graphics(cmd, cs);
   }

   assert(*ptr != NULL);
   hk_cmd_sync_work(cmd, *ptr, stages);
   return *ptr;
}

static inline struct hk_cs *
hk_cmd_buffer_get_cs(struct hk_cmd_buffer *cmd, bool compute)
{
   struct hk_cs **ptr = compute ? &cmd->current_cs.cs : &cmd->current_cs.gfx;
   unsigned stages = cmd->sync_scope ? cmd->sync_scope :
      compute ? HK_SYNC_COMPUTE | HK_SYNC_DISPATCH_INDIRECT :
                HK_SYNC_GRAPHICS | HK_SYNC_DRAW_INDIRECT;
   return hk_cmd_buffer_get_cs_general(cmd, ptr, compute, stages);
}

void hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            size_t space);

static inline uint64_t
hk_cs_alloc_for_indirect(struct hk_cs *cs, size_t size_B)
{
   hk_ensure_cs_has_space(cs->cmd, cs, size_B);

   uint64_t addr = hk_cs_current_addr(cs);
   cs->current += size_B;
   return addr;
}

static void
hk_cmd_buffer_dirty_all(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   vk_dynamic_graphics_state_dirty_all(dyn);
   gfx->dirty = ~0;
   gfx->shaders_dirty = ~0;
   gfx->linked_dirty = ~0;
   gfx->descriptors.root_dirty = true;
}

static inline void
hk_cs_destroy(struct hk_cs *cs)
{
   if (cs->type == HK_CS_VDM) {
      util_dynarray_fini(&cs->scissor);
      util_dynarray_fini(&cs->depth_bias);
   } else {
      util_dynarray_fini(&cs->imm_writes);
   }

   free(cs);
}

void hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_tess_batch_flush(struct hk_cmd_buffer *cmd);

/* Record why the next stream end happens, unless a caller already did. */
static inline void
hk_cs_end_hint(struct hk_cmd_buffer *cmd, enum hk_cs_end reason)
{
   if (!cmd->end_hint)
      cmd->end_hint = reason;
}

static void
hk_cmd_buffer_end_compute_internal(struct hk_cmd_buffer *cmd,
                                   struct hk_cs **ptr)
{
   if (*ptr) {
      struct hk_cs *cs = *ptr;
      if (!cs->end_reason)
         cs->end_reason = cmd->end_hint;

      /* This control stream may write immediates as it ends. Queue the writes
       * now that we're done emitting everything else.
       */
      if (cs->imm_writes.size) {
         hk_dispatch_imm_writes(cmd, cs);
      }

      /* Availability at the stream boundary, including the last dispatch. */
      if (cs->cdm_dirty)
         hk_cdm_cache_flush(hk_cmd_buffer_device(cmd), cs);
   }

   *ptr = NULL;
}

static void
hk_cmd_buffer_end_compute(struct hk_cmd_buffer *cmd)
{
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.cs);
   cmd->end_hint = HK_CS_END_UNKNOWN;
}

void hk_optimize_empty_vdm(struct hk_cmd_buffer *cmd);

bool hk_cond_render_applies(struct hk_cmd_buffer *cmd);
uint64_t hk_predicate_indirect(struct hk_cmd_buffer *cmd, uint64_t in,
                               unsigned words, bool graphics);

static void
hk_cmd_buffer_end_graphics(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.gfx;
   /* An empty render kept open for merging ends after its render pass: the
    * empty-render optimization needs that pass's render state back. */
   const bool restore = cs && cs == cmd->merge_cs && cmd->merge_empty;
   cmd->merge_cs = NULL;
   cmd->merge_empty = false;
   cmd->merge_suspended = false;
   cmd->merge_reach = 0;
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->merge_pre_cs);
   if (cs && !cs->end_reason)
      cs->end_reason = cmd->end_hint;
   if (cs) {
      cmd->sync_render_stages = cs->sync_stages;
      cmd->sync_render_continuation = true;
   }

   if (cs && cs->stats.cmds == 0) {
      if (restore) {
         struct hk_rendering_state *render = &cmd->state.gfx.render;
         memcpy(render, cmd->merge_empty_render, sizeof(*render));
         hk_optimize_empty_vdm(cmd);
         memset(render, 0, sizeof(*render));
      } else {
         hk_optimize_empty_vdm(cmd);
      }
   } else if (cs) {
      /* Scissor and depth bias arrays are staged to dynamic arrays on the
       * CPU. When we end the control stream, they're done growing and are
       * ready for upload.
       */
      cs->uploaded_scissor =
         hk_pool_upload(cmd, cs->scissor.data, cs->scissor.size, 64);

      cs->uploaded_zbias =
         hk_pool_upload(cmd, cs->depth_bias.data, cs->depth_bias.size, 64);

      /* TODO: maybe free scissor/depth_bias now? */
      cs->current = agx_vdm_terminate(cs->current);
   }

   cmd->current_cs.gfx = NULL;

   hk_tess_batch_flush(cmd);
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.pre_gfx);
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   assert(cmd->current_cs.pre_gfx == NULL);
   assert(cmd->current_cs.post_gfx == NULL);
   cmd->end_hint = HK_CS_END_UNKNOWN;

   /* We just flushed out the heap use. If we want to use it again, we'll need
    * to queue a free for it again.
    */
   cmd->uses_heap = false;
   cmd->uses_mesh_heap = false;
}

static inline uint64_t
hk_pipeline_stat_addr(struct hk_cmd_buffer *cmd,
                      VkQueryPipelineStatisticFlagBits stat)
{
   struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   VkQueryPipelineStatisticFlags flags = root->draw.pipeline_stats_flags;

   if (flags & stat) {
      assert(!cmd->in_meta && "queries paused for meta");
      assert(util_bitcount(stat) == 1 && "by construction");

      /* Prefix sum to determine the compacted index in the query pool */
      uint32_t index = util_bitcount(flags & (stat - 1));

      return root->draw.pipeline_stats + (sizeof(uint64_t) * index);
   } else {
      /* Query disabled */
      return AGX_SCRATCH_PAGE_ADDRESS;
   }
}

static inline bool
hk_stat_enabled(uint64_t addr)
{
   return addr != AGX_SCRATCH_PAGE_ADDRESS;
}

void hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                                  const VkCommandBufferBeginInfo *pBeginInfo);
void hk_cmd_buffer_begin_compute(struct hk_cmd_buffer *cmd,
                                 const VkCommandBufferBeginInfo *pBeginInfo);

void hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd);
void hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd);

void hk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                         const mesa_shader_stage *stages,
                         struct vk_shader **const shaders);

void hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                                 const mesa_shader_stage stage,
                                 struct hk_api_shader *shader);

void hk_cmd_bind_compute_shader(struct hk_cmd_buffer *cmd,
                                struct hk_api_shader *shader);

void hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                               struct hk_addr_range addr_range);

static inline struct hk_descriptor_state *
hk_get_descriptors_state(struct hk_cmd_buffer *cmd,
                         VkPipelineBindPoint bind_point)
{
   switch (bind_point) {
   case VK_PIPELINE_BIND_POINT_GRAPHICS:
      return &cmd->state.gfx.descriptors;
   case VK_PIPELINE_BIND_POINT_COMPUTE:
      return &cmd->state.cs.descriptors;
   default:
      UNREACHABLE("Unhandled bind point");
   }
};

void hk_flush_ia_stats(struct hk_cmd_buffer *cmd);
void hk_flush_query_writes(struct hk_cmd_buffer *cmd);
struct hk_cs *hk_occlusion_prepare_draw(struct hk_cmd_buffer *cmd, struct hk_cs *cs);

void hk_cmd_buffer_flush_push_descriptors(struct hk_cmd_buffer *cmd,
                                          struct hk_descriptor_state *desc);

void hk_meta_resolve_rendering(struct hk_cmd_buffer *cmd,
                               const VkRenderingInfo *pRenderingInfo);

uint64_t hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                                   VkPipelineBindPoint bind_point);

void hk_reserve_scratch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s);

uint32_t hk_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                             struct hk_linked_shader *linked);

void hk_usc_upload_spilled_rt_descs(struct agx_usc_builder *b,
                                    struct hk_cmd_buffer *cmd);

void hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                                 struct agx_cdm_launch_word_0_packed launch,
                                 uint32_t usc, struct agx_grid grid,
                                 struct agx_workgroup local_size);

void hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                          struct agx_shader_info *info, uint32_t usc,
                          struct agx_grid grid,
                          struct agx_workgroup local_size);

static inline void
hk_dispatch_with_local_size(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            struct hk_shader *s, struct agx_grid grid,
                            struct agx_workgroup local_size)
{
   if (agx_is_shader_empty(&s->b))
      return;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint32_t usc = hk_upload_usc_words(cmd, s, s->only_linked);

   hk_reserve_scratch(cmd, cs, s);
   hk_dispatch_with_usc(dev, cs, &s->b.info, usc, grid, local_size);
}

/* Buffers bound for software pre-rasterization kernels, which read the
 * graphics descriptors and the draw's vertex input tables.
 */
struct hk_apple9_prerast_bindings {
   uint64_t draw_params;
   uint64_t vertex_buffers;
};

/* Native grid counts are threads; indirect grids point to group counts. */
void hk_apple9_dispatch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       struct agx_bo *body,
                       const struct agx_apple9_compute_profile *profile,
                       struct agx_grid grid, uint64_t root,
                       const struct hk_shader *shader,
                       const struct hk_apple9_prerast_bindings *prerast);

void hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                         enum agx_barrier barrier, enum libagx_program idx,
                         void *data, size_t data_size);

#define MESA_DISPATCH_PRECOMP hk_dispatch_precomp

void hk_queue_write(struct hk_cmd_buffer *cmd, uint64_t address, uint32_t value,
                    bool after_gfx);

void agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                         uint64_t attribs_read, struct agx_velem_key *keys);

struct agx_robustness hk_prolog_robustness(struct hk_device *dev);

void hk_apple9_tile_helpers_finish(struct hk_device *dev);
struct hk_bg_eot hk_apple9_build_bg_eot(
   struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
   bool store, bool partial, bool incomplete);

bool hk_apple9_pin_entry(void *allocator, struct agx_bo *bo);
/* reads_buffers: the draw's tiling reads buffer memory itself (indices or
 * indirect arguments). */
bool hk_apple9_prepare_draw(struct hk_cmd_buffer *cmd, bool reads_buffers);
void hk_apple9_draw(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                    struct agx_draw draw);
struct hk_apple9_prerast_bindings
hk_apple9_draw_bindings(struct hk_cmd_buffer *cmd, struct agx_draw draw);
void hk_apple9_compute_sysvals(struct hk_device *dev, void *out);
bool hk_apple9_upload_textures(struct hk_cmd_buffer *cmd,
   const struct hk_shader *shader, bool graphics, bool first_fan,
   uint64_t *textures, uint64_t *samplers, uint64_t *sysvals);
