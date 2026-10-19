/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/lib/agx_apple9.h"
#include "asahi/lib/agx_nir_apple9_tilebuffer.h"
#include "pipe/p_defines.h"
#include "util/u_math.h"
#include "util/u_prim.h"
#include "vk_util.h"
#include "hk_cmd_buffer.h"
#include "hk_apple9_mesh.h"
#include "hk_device.h"
#include "hk_image.h"
#include "hk_image_view.h"

static struct agx_ptr
hk_apple9_usc_alloc(void *data, size_t size, size_t alignment)
{
   return hk_pool_usc_alloc(data, size, alignment);
}

/* A tile helper program depends only on its key. Each render binds its own
 * clear values, reload textures and store images through tables. */
struct hk_apple9_tile_helper {
   struct agx_apple9_tilebuffer_key key;
   struct agx_bo *bo;
   struct agx_shader_info info;
};

static uint32_t
hk_apple9_tile_helper_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct agx_apple9_tilebuffer_key));
}

static bool
hk_apple9_tile_helper_equal(const void *a, const void *b)
{
   return !memcmp(a, b, sizeof(struct agx_apple9_tilebuffer_key));
}

static const struct hk_apple9_tile_helper *
hk_apple9_tile_helper(struct hk_device *dev,
                      const struct agx_apple9_tilebuffer_key *key,
                      const struct agx_apple9_blend *blend)
{
   const struct hk_apple9_tile_helper *helper = NULL;
   struct hk_apple9_tile_helper *created = NULL;
   struct agx_shader_part part = {0};
   nir_shader *nir = NULL;
   const char *reason = NULL;
   simple_mtx_lock(&dev->apple9_tile_helpers.lock);
   if (!dev->apple9_tile_helpers.ht) {
      dev->apple9_tile_helpers.ht = _mesa_hash_table_create(
         NULL, hk_apple9_tile_helper_hash, hk_apple9_tile_helper_equal);
   }
   struct hash_table *ht = dev->apple9_tile_helpers.ht;
   struct hash_entry *entry = ht ? _mesa_hash_table_search(ht, key) : NULL;
   if (entry) {
      helper = entry->data;
      goto done;
   }
   if (!ht)
      goto done;

   nir = agx_nir_apple9_tilebuffer(&agx_nir_options, key);
   if (!agx_compile_apple9_fragment_mrt(nir, NULL, blend, key->count, &part,
                                        &reason)) {
      fprintf(stderr, "Apple9 tile helper: %s\n",
              reason ?: "unsupported shader");
      goto done;
   }
   created = calloc(1, sizeof(*created));
   if (!created)
      goto done;
   created->key = *key;
   created->info = part.info;
   created->bo = agx_bo_create(&dev->dev, part.info.binary_size, 0,
                               AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                               "Vulkan Apple9 tile helper");
   if (!created->bo) {
      free(created);
      goto done;
   }
   memcpy(agx_bo_map(created->bo), part.binary, part.info.binary_size);
   agx_bo_note_cpu_write(created->bo, 0, part.info.binary_size);
   _mesa_hash_table_insert(ht, &created->key, created);
   helper = created;

done:
   free(part.binary);
   ralloc_free(nir);
   simple_mtx_unlock(&dev->apple9_tile_helpers.lock);
   return helper;
}

void
hk_apple9_tile_helpers_finish(struct hk_device *dev)
{
   if (!dev->apple9_tile_helpers.ht)
      return;
   hash_table_foreach(dev->apple9_tile_helpers.ht, entry) {
      struct hk_apple9_tile_helper *helper = entry->data;
      agx_bo_unreference(&dev->dev, helper->bo);
      free(helper);
   }
   _mesa_hash_table_destroy(dev->apple9_tile_helpers.ht, NULL);
   dev->apple9_tile_helpers.ht = NULL;
}

struct hk_bg_eot
hk_apple9_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                       bool store, bool partial, bool incomplete)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   /* Keys are hashed as bytes, including padding. */
   struct agx_apple9_tilebuffer_key key;
   memset(&key, 0, sizeof(key));
   key.count = render->color_att_count;
   key.samples = MAX2(render->tilebuffer.nr_samples, 1);
   key.store = store;
   key.layered = render->cr.layers > 1;
   struct agx_apple9_blend blend[8] = {0};
   struct agx_texture_packed legacy[8] = {0};
   uint32_t params[48] = {0};
   uint8_t metadata[AGX_APPLE9_RUNTIME_SYSVAL_SIZE] = {0};
   unsigned tile_bytes = 0;
   VkResult error = VK_ERROR_FEATURE_NOT_PRESENT;
   if (key.count > 8)
      goto fail;

   for (unsigned rt = 0; rt < key.count; ++rt) {
      struct hk_image_view *view = render->color_att[rt].iview;
      key.formats[rt] = view ? hk_format_to_pipe_format(view->vk.format)
                             : PIPE_FORMAT_R8G8B8A8_UNORM;
      blend[rt].format = key.formats[rt];
      blend[rt].samples = key.samples;
      tile_bytes += 4 * agx_apple9_color_words(key.formats[rt]);
      if (!view)
         continue;
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[rt];
      bool load =
         partial || incomplete || att->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
      bool clear = !load && att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
      bool write = partial || att->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                   att->resolveMode != VK_RESOLVE_MODE_NONE;
      if (store ? write : (load || clear))
         key.mask |= BITFIELD_BIT(rt);
      if (clear)
         key.clear_mask |= BITFIELD_BIT(rt);
      memcpy(&params[16 + rt * 4], &att->clearValue.color, 16);
      const struct agx_texture_packed *texture =
         key.layered ? &view->planes[0].layered_background
                     : &view->planes[0].background;
      agx_unpack(stderr, texture, TEXTURE, image);
      /* Tile words retain encoded sRGB. Hardware reloads must not decode it. */
      image.srgb = image.srgb_2_channel = false;
      agx_pack(&legacy[rt], TEXTURE, cfg) {
         cfg = image;
      }
      if (image.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
          image.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED)
         key.array_mask |= BITFIELD_BIT(rt);
      uint32_t dimensions[] = {
         u_minify(image.width, image.first_level),
         u_minify(image.height, image.first_level),
         image.depth,
         image.last_level - image.first_level + 1,
         key.samples,
      };
      memcpy(metadata + AGX_APPLE9_TEXTURE_INFO_OFFSET +
                rt * AGX_APPLE9_TEXTURE_INFO_STRIDE,
             dimensions, sizeof(dimensions));
   }
   blend[0].samples = key.samples;
   if (tile_bytes > 128) {
      fprintf(stderr, "Apple9 tile helper: %u tile bytes at %u samples\n",
              tile_bytes, key.samples);
      goto fail;
   }
   const struct hk_apple9_tile_helper *helper =
      hk_apple9_tile_helper(dev, &key, blend);
   if (!helper)
      goto fail;
   const struct agx_shader_info *part_info = &helper->info;
   error = VK_ERROR_OUT_OF_DEVICE_MEMORY;
   uint64_t arguments = hk_pool_upload(cmd, params, sizeof(params), 16);
   uint64_t sysvals = hk_pool_upload_cached(cmd, metadata, sizeof(metadata), 64,
                                            HK_UPLOAD_RUNTIME);
   struct agx_ptr textures =
      hk_pool_alloc(cmd, 8 * AGX_APPLE9_TEXTURE_TABLE_STRIDE, 64);
   struct agx_ptr sampler =
      hk_pool_alloc(cmd, AGX_APPLE9_SAMPLER_TABLE_STRIDE, 64);
   if (!arguments || !sysvals || !textures.cpu || !sampler.cpu)
      goto fail;
   memset(sampler.cpu, 0, AGX_APPLE9_SAMPLER_TABLE_STRIDE);
   agx_apple9_pack_sampler(sampler.cpu, false, false, 1, 0, 14,
                           PIPE_TEX_WRAP_CLAMP_TO_EDGE,
                           PIPE_TEX_WRAP_CLAMP_TO_EDGE, 1);
   unsigned slot = 0;
   error = VK_ERROR_FEATURE_NOT_PRESENT;
   u_foreach_bit(rt, part_info->apple9_texture_mask) {
      if (rt >= key.count ||
          !agx_apple9_translate_texture(
             (uint8_t *)textures.cpu + slot++ * AGX_APPLE9_TEXTURE_TABLE_STRIDE,
             &legacy[rt])) {
         fprintf(stderr, "Apple9 tile helper: unsupported reload texture %u (%s)\n", rt, util_format_short_name(key.formats[rt]));
         goto fail;
      }
   }
   u_foreach_bit(rt, part_info->apple9_image_mask) {
      if (rt >= key.count || !hk_apple9_image_view_eot(
                                render->color_att[rt].iview, key.layered,
                                (uint8_t *)textures.cpu +
                                   slot++ * AGX_APPLE9_TEXTURE_TABLE_STRIDE)) {
         fprintf(stderr, "Apple9 tile helper: unsupported store image %u\n", rt);
         goto fail;
      }
   }
   uint64_t resources[AGX_APPLE9_MAX_GRAPHICS_BUFFERS] = {0};
   for (unsigned i = 0; i < part_info->apple9_resource_count; ++i) {
      unsigned binding = part_info->apple9_resource_binding[i];
      if (binding == 0)
         resources[i] = arguments;
      else if (binding == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING)
         resources[i] = sysvals;
      else
         goto fail;
   }
   uint64_t table = hk_pool_upload(
      cmd, resources, part_info->apple9_resource_count * sizeof(uint64_t), 64);
   struct agx_apple9_render_stage shader = {
      .bo = helper->bo,
      .binary_size = part_info->binary_size,
      .preamble_offset = part_info->apple9_preamble_offset,
      .preamble_size = part_info->apple9_preamble_size,
      .publication_count = part_info->apple9_publication_count,
      .publication_count_valid = true,
      .scratch_size = part_info->scratch_size,
   };
   uint32_t launch;
   if (!agx_apple9_prepare_graphics_stage(
          &dev->dev, cmd, hk_apple9_usc_alloc, &shader, true, table,
          textures.gpu, sampler.gpu,
          agx_apple9_tile_sample_bytes(tile_bytes, key.samples), key.samples,
          &launch,
          hk_apple9_state_uncached_alloc))
      goto fail;
   return (struct hk_bg_eot){.usc = launch};

fail:
   vk_command_buffer_set_error(&cmd->vk, error);
   return (struct hk_bg_eot){0};
}

static_assert(HK_MAX_VIEWPORTS == AGX_APPLE9_MAX_VIEWPORTS,
              "Vulkan viewport limit matches the PPP viewport block");

static struct agx_ptr
hk_apple9_context_alloc(void *data, size_t size, size_t alignment)
{
   return hk_pool_context_alloc(data, size, alignment);
}

/* Draw parameters (first vertex or index bias, then base instance) and the
 * vertex buffer table, shared by the hardware vertex fetch and by software
 * vertex shaders in front of a geometry shader.
 */
struct hk_apple9_prerast_bindings
hk_apple9_draw_bindings(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t params[] = {draw.indexed ? draw.index_bias : draw.start,
                        draw.start_instance};
   uint64_t draw_params =
      agx_is_indirect(draw.b)
         ? draw.b.ptr +
              (draw.indexed
                  ? offsetof(VkDrawIndexedIndirectCommand, vertexOffset)
                  : offsetof(VkDrawIndirectCommand, firstVertex))
         : hk_pool_upload_last(cmd, HK_UPLOAD_LAST_DRAW_PARAMS, params,
                               sizeof(params), 4);
   uint64_t vertex_buffers[ARRAY_SIZE(gfx->vb)][4] = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(gfx->vb); ++i) {
      vertex_buffers[i][0] = gfx->vb[i].addr;
      vertex_buffers[i][1] = gfx->vb[i].range;
      vertex_buffers[i][2] = dyn->vi_binding_strides[i];
   }
   return (struct hk_apple9_prerast_bindings){
      .draw_params = draw_params,
      .vertex_buffers =
         hk_pool_upload_last(cmd, HK_UPLOAD_LAST_VERTEX_BUFFERS, vertex_buffers,
                             sizeof(vertex_buffers), 16),
   };
}

static enum agx_primitive
hk_apple9_rast_primitive(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   default:
      return AGX_PRIMITIVE_TRIANGLES;
   }
}

/* Split before recording any of a draw's pre-rasterization work. Ending a
 * graphics stream also closes its pre/post compute streams and heap lifetime.
 * Reserve for every emulated multiview draw so these stay together.
 */
/* Sample positions for a draw: the application's when enabled for this
 * sample count, the standard ones otherwise. */
static uint32_t
hk_apple9_draw_sample_positions(const struct vk_dynamic_graphics_state *dyn,
                                unsigned samples)
{
   return dyn->ms.sample_locations_enable && dyn->ms.sample_locations &&
                dyn->ms.sample_locations->per_pixel == samples
             ? hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations)
             : hk_default_sample_positions(samples);
}

bool
hk_apple9_prepare_draw(struct hk_cmd_buffer *cmd, bool reads_buffers)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return false;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   if (!cs)
      return false;

   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* Sample positions are render state (the per-render multisample control),
    * so the first application draw of a render commits them and a draw with
    * other positions starts a new render. That lets them vary within a
    * subpass (variableSampleLocations, reported to Zink only). Meta draws
    * neither commit nor change them.
    */
   if (!cmd->in_meta) {
      unsigned samples = MAX2(render->tilebuffer.nr_samples ?:
                              dyn->ms.rasterization_samples, 1);
      uint32_t positions = hk_apple9_draw_sample_positions(dyn, samples);
      if (cs->has_sample_locations && cs->ppp_multisamplectl != positions) {
         perf_debug(cmd, "Splitting Apple9 render: sample positions changed");
         hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false);
         if (!cs)
            return false;
      }
      cs->has_sample_locations = true;
      cs->ppp_multisamplectl = positions;
   }

   unsigned views = util_bitcount(render->view_mask ?: 1);
   unsigned viewports = dyn->rs.rasterizer_discard_enable
                           ? 1 : MAX2(dyn->vp.viewport_count, 1);
   bool bias = dyn->rs.depth_bias.enable;
   bool dbias_is_int = render->depth_att.vk_format == VK_FORMAT_D16_UNORM ||
      dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;

   bool overflow =
      cs->scissor.size / AGX_SCISSOR_LENGTH + viewports * views > UINT16_MAX ||
      (bias && cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH + views > UINT16_MAX);
   bool change_bias = bias && !u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
   if (unlikely(overflow || change_bias)) {
      perf_debug(cmd, "Splitting Apple9 render: %s",
                 overflow ? "scissor/depth bias overflow" : "depth bias representation");
      /* The existing partial-render background, EOT and ZLS state stores and
       * reloads attachments. Only the final stream uses the API store ops.
       */
      hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
      if (bias) {
         bool success = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(success);
         (void)success;
      }
   }

   /* A stream that continued across a hoisted image barrier must not write
    * memory: tiles of its earlier passes may still read what it writes. */
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   bool writes = gfx->xfb_enabled;
   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT + 1 && !writes; ++s) {
      if (s != MESA_SHADER_COMPUTE && gfx->shaders[s])
         writes = gfx->shaders[s]->writes_memory;
   }
   writes |= gfx->shaders[MESA_SHADER_TASK] && gfx->shaders[MESA_SHADER_TASK]->writes_memory;
   writes |= gfx->shaders[MESA_SHADER_MESH] && gfx->shaders[MESA_SHADER_MESH]->writes_memory;
   if (unlikely(writes && cs->merge_hoisted)) {
      perf_debug(cmd, "Splitting Apple9 render: memory writes after a hoisted barrier");
      hk_cs_end_hint(cmd, HK_CS_END_RENDER_SPLIT);
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false);
      if (!cs)
         return false;
      if (bias) {
         bool success = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(success);
         (void)success;
      }
   }
   cs->merge_writes |= writes;

   /* What the draw's pre-rasterization work reads. hk-internal pre-raster
    * work (tessellation, geometry, transform feedback) keeps the tiling
    * wait. Meta draws compile their pipelines like applications', so their
    * shaders carry a read class (vk_meta vertex shaders read push constants
    * and its own vertex data). */
   uint8_t reads = reads_buffers ? HK_READS_BUFFERS : 0;
   if (cmd->in_meta)
      reads |= HK_READS_META;
   if (gfx->xfb_enabled || gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] || gfx->shaders[MESA_SHADER_GEOMETRY])
      reads |= HK_READS_INTERNAL;
   for (unsigned s = 0; s < ARRAY_SIZE(gfx->shaders); ++s) {
      if (s == MESA_SHADER_FRAGMENT || s == MESA_SHADER_COMPUTE || !gfx->shaders[s])
         continue;
      reads |= gfx->shaders[s]->pre_raster_reads;
   }
   cs->pre_raster_reads |= reads;
   cs->linear_store |= gfx->render.linear_store;

   return true;
}

/* The compiler's local roots and ordinary graphics roots share one table.
 * No scene allocation, cursor, resource ID or alias crosses into userspace.
 * Command-pool storage and pinned shader BOs survive command-buffer reuse. */
static bool
hk_apple9_prepare_native_mesh(struct hk_cmd_buffer *cmd,
                               const struct hk_shader *mesh,
                               const struct hk_shader *object,
                               const struct agx_apple9_uniform_draw *draw,
                               uint64_t grid,
                               uint32_t state[AGX_APPLE9_MESH_STATE_WORDS])
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_apple9_native_mesh_enabled() || !grid || (grid & 3) ||
       mesh->info.apple9_native_stage != HK_APPLE9_NATIVE_MESH ||
       object->info.apple9_native_stage != HK_APPLE9_NATIVE_OBJECT)
      return false;
   struct agx_apple9_launch_parameters p[2];
   const struct hk_shader *shaders[] = {mesh, object};
   for (unsigned i = 0; i < 2; ++i) {
      const struct hk_shader *shader = shaders[i];
      if (!shader->bo ||
          !hk_apple9_native_launch_parameters(
             &p[i], &shader->b.info, &shader->info.apple9_native,
             shader->info.apple9_native_stage, dev->dev.shader_base,
             shader->bo->va->addr))
         return false;
      simple_mtx_lock(&dev->dev.apple9_entry_lock);
      bool pinned = hk_apple9_pin_entry(cmd, shader->bo);
      simple_mtx_unlock(&dev->dev.apple9_entry_lock);
      if (!pinned)
         return false;
   }
   struct agx_apple9_mesh_arguments args = {
      .graphics = {draw->texture_table[0], draw->sampler_table[0],
                   draw->vertex_table},
      .output_root = HK_APPLE9_NATIVE_OUTPUT_ROOT,
      .grid_root = HK_APPLE9_NATIVE_GRID_ROOT,
      .draw = grid,
   };
   uint64_t words[AGX_APPLE9_MESH_ARGUMENT_WORDS / 2];
   if (!agx_apple9_mesh_arguments_build(words, &args))
      return false;
   uint64_t arguments = hk_pool_upload(cmd, words, sizeof(words), 64);
   struct agx_ptr launch = hk_pool_usc_alloc(
      cmd, 2 * AGX_APPLE9_GRAPHICS_LAUNCH_SIZE, 128);
   if (!arguments || !launch.cpu || launch.gpu < dev->dev.shader_base ||
       launch.gpu - dev->dev.shader_base >
          UINT32_MAX - 2 * AGX_APPLE9_GRAPHICS_LAUNCH_SIZE)
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      p[i].resource_table = arguments;
      p[i].launch_address = launch.gpu + i * AGX_APPLE9_GRAPHICS_LAUNCH_SIZE;
      if (!agx_apple9_launch_build(
             (uint8_t *)launch.cpu + i * AGX_APPLE9_GRAPHICS_LAUNCH_SIZE,
             AGX_APPLE9_GRAPHICS_LAUNCH_SIZE,
             i ? AGX_APPLE9_LAUNCH_TASK : AGX_APPLE9_LAUNCH_MESH, &p[i]))
         return false;
   }
   struct agx_apple9_mesh_state_parameters params = {
      .layout = &mesh->info.apple9_native.layout,
      .mesh_launch = agx_usc_addr(&dev->dev, launch.gpu) / 64,
      .object_launch = agx_usc_addr(
         &dev->dev, launch.gpu + AGX_APPLE9_GRAPHICS_LAUNCH_SIZE) / 64,
      .threads = AGX_APPLE9_NATIVE_THREADS,
      .has_object_stage = true,
   };
   return agx_apple9_mesh_state_build(state, &params);
}

static void
hk_apple9_draw_impl(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                    struct agx_draw draw, uint64_t native_grid)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   /* Geometry shaders can produce empty rasterization draws. */
   if (!agx_is_indirect(draw.b) && (!draw.b.count[0] || !draw.b.count[1]))
      return;
   unsigned failure_line = 0;
#define UNSUPPORTED()                                                          \
   do {                                                                        \
      failure_line = __LINE__;                                                 \
      goto unsupported;                                                        \
   } while (0)
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   bool rasterizer_discard = dyn->rs.rasterizer_discard_enable;
   /* A geometry shader has already run as compute; its rasterization
    * variant is the hardware vertex program for its output primitives.
    */
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   /* A non-zero rasterization stream discards unless a multistream geometry
    * shader selects that stream.
    */
   if (dyn->rs.rasterization_stream != 0) {
      rasterizer_discard |=
         !gs || !gs->variants[HK_GS_VARIANT_COUNT].info.gs.multistream;
   }
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   /* Mesh shaders have written their records as compute. */
   struct hk_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];
   /* Without an API fragment shader, the null program only applies the API
    * sample mask. A full mask needs no fragment work at all. */
   const bool null_fs = !gfx->shaders[MESA_SHADER_FRAGMENT] ||
                        gfx->shaders[MESA_SHADER_FRAGMENT] == dev->null_fs;
   const unsigned null_samples = MAX2(gfx->render.tilebuffer.nr_samples ?:
                                      dyn->ms.rasterization_samples, 1);
   const bool null_trivial =
      null_fs && dev->null_fs_trivial &&
      (dyn->ms.sample_mask & BITFIELD_MASK(null_samples)) ==
         BITFIELD_MASK(null_samples);
   struct hk_shader *shaders[] = {
      mesh  ? &mesh->variants[native_grid ? HK_MESH_VARIANT_MAIN
                                         : HK_MESH_VARIANT_RAST]
      : gs  ? &gs->variants[HK_GS_VARIANT_RAST]
      : tes ? &tes->variants[HK_VS_VARIANT_HW]
            : hk_any_variant(gfx->shaders[MESA_SHADER_VERTEX]),
      hk_any_variant(null_trivial ? dev->null_fs_trivial
                     : gfx->shaders[MESA_SHADER_FRAGMENT] ?: dev->null_fs),
   };
   if (native_grid &&
       (!mesh || gfx->shaders[MESA_SHADER_TASK] || gs || tes ||
        shaders[0]->info.apple9_native_stage != HK_APPLE9_NATIVE_MESH ||
        dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
        gfx->render.view_mask))
      UNSUPPORTED();
   /* Geometry and tessellation have already run as compute, and draw their
    * own output topology. Adjacency primitives without a geometry shader
    * draw a generated list without the adjacent vertices.
    */
   enum mesa_prim api_prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   bool adjacency = !mesh && !gs && !tes && mesa_prim_has_adjacency(api_prim);
   bool emulated = mesh || gs || tes || adjacency;
   /* Unsupported state must fail recording before any job can be submitted. */
   if (!shaders[0] || !shaders[1] ||
       (!rasterizer_discard &&
        (!dyn->vp.viewport_count ||
         dyn->vp.viewport_count > AGX_APPLE9_MAX_VIEWPORTS ||
         dyn->vp.scissor_count != dyn->vp.viewport_count)) ||
       (!emulated && dyn->ia.primitive_topology > VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
        dyn->ia.primitive_topology !=
           VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA))
      UNSUPPORTED();
   enum mesa_prim gs_prim = mesh  ? (shaders[0]->info.mesh.polygon_points
                                        ? MESA_PRIM_POINTS
                                        : shaders[0]->info.mesh.prim)
                            : gs  ? shaders[0]->info.gs.mode
                            : tes ? gfx->tess.prim
                            : adjacency ? u_reduced_prim(api_prim)
                                  : MESA_PRIM_COUNT;
   bool points = emulated ? gs_prim == MESA_PRIM_POINTS
                          : dyn->ia.primitive_topology ==
                               VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   bool lines = emulated ? (gs_prim == MESA_PRIM_LINES ||
                            gs_prim == MESA_PRIM_LINE_STRIP)
                   : (dyn->ia.primitive_topology >=
                         VK_PRIMITIVE_TOPOLOGY_LINE_LIST &&
                      dyn->ia.primitive_topology <=
                         VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
   const struct agx_apple9_render_stage *vertex = &shaders[0]->apple9_stage;
   const struct agx_apple9_render_stage *fragment = &shaders[1]->apple9_stage;
   if (null_fs) {
      /* The empty fragment program reads no varyings. Retain the producer's
       * layout for the native interface linker, independent of its contents. */
      if (!vertex->program_id ||
          cmd->apple9_null_fs_vertex_id != vertex->program_id ||
          cmd->apple9_null_fs_trivial != null_trivial) {
         cmd->apple9_null_fs_trivial = null_trivial;
         cmd->apple9_null_fs = *fragment;
         cmd->apple9_null_fs.varyings = vertex->varyings;
         cmd->apple9_null_fs.varying_components = vertex->varying_components;
         cmd->apple9_null_fs_vertex_id = vertex->program_id;
      }
      fragment = &cmd->apple9_null_fs;
   }
   if (!vertex->program_id || !fragment->program_id ||
       cmd->apple9_linked_id[0] != vertex->program_id ||
       cmd->apple9_linked_id[1] != fragment->program_id ||
       cmd->apple9_linked.vertex != vertex ||
       cmd->apple9_linked.fragment != fragment) {
      if (!agx_apple9_link_render_pipeline(&cmd->apple9_linked, vertex,
                                           fragment)) {
         cmd->apple9_linked_id[0] = cmd->apple9_linked_id[1] = 0;
         UNSUPPORTED();
      }
      cmd->apple9_linked_id[0] = vertex->program_id;
      cmd->apple9_linked_id[1] = fragment->program_id;
      agx_apple9_build_ppp_static(&cmd->apple9_linked, cmd->apple9_ppp_static);
      cmd->apple9_linked.ppp_static = cmd->apple9_ppp_static;
   }
   struct agx_apple9_render_pipeline pipeline = cmd->apple9_linked;
   /* Attachmentless render passes acquire their sample count from the first
    * drawing pipeline. Keep the native tile layout and submission state in
    * agreement with the fragment launch, including a pre-existing query CS.
    */
   if (!gfx->render.tilebuffer.nr_samples && dyn->ms.rasterization_samples) {
      gfx->render.tilebuffer.nr_samples = dyn->ms.rasterization_samples;
      gfx->render.tilebuffer.sample_size_B = ALIGN_POT(
         agx_apple9_tile_sample_bytes(0, dyn->ms.rasterization_samples), 8);
      gfx->render.tilebuffer.tile_size =
         dyn->ms.rasterization_samples == 4 ? 32 * 16 : 32 * 32;
   }
   pipeline.samples = MAX2(gfx->render.tilebuffer.nr_samples, 1);
   for (unsigned rt = 0; rt < gfx->render.color_att_count; ++rt) {
      struct hk_image_view *view = gfx->render.color_att[rt].iview;
      pipeline.color_formats[rt] =
         view ? hk_format_to_pipe_format(view->vk.format)
              : PIPE_FORMAT_R8G8B8A8_UNORM;
   }
   static const uint8_t primitives[] = {
      AGX_PRIMITIVE_POINTS,         AGX_PRIMITIVE_LINES,
      AGX_PRIMITIVE_LINE_STRIP,     AGX_PRIMITIVE_TRIANGLES,
      AGX_PRIMITIVE_TRIANGLE_STRIP, AGX_PRIMITIVE_TRIANGLE_FAN,
   };
   /* Mesa meta rectangles are already expanded to two ordinary triangles. */
   pipeline.primitive =
      emulated ? hk_apple9_rast_primitive(gs_prim)
      : dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA ||
           gfx->apple9_fan_unrolled
         ? AGX_PRIMITIVE_TRIANGLES
         : primitives[dyn->ia.primitive_topology];
   /* Emulated stages emit their primitives for the same provoking mode.
    * Stippled lines without them take the segment start from the first
    * vertex, see hk_apple9_lower_line_stipple_fs. Nothing else observes the
    * change: such fragment shaders read no flat varyings, layer or viewport
    * index (primitive ID does not depend on the provoking vertex), and the
    * root table keeps the API mode for transform feedback and adjacency. */
   pipeline.flatshade_first =
      dyn->rs.provoking_vertex != VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ||
      (!mesh && !gs && !adjacency && shaders[1]->info.apple9_line_stipple_first);
   if (draw.indexed) {
      pipeline.index_size = agx_index_size_to_B(draw.index_size);
      uint64_t offset = agx_is_indirect(draw.b)
                           ? 0
                           : (uint64_t)draw.start * pipeline.index_size;
      uint64_t index_buffer = draw.index_buffer;
      uint64_t index_range = draw.index_buffer_range_B;
      /* Null bindings and reads past the binding fetch index zero. The
       * hardware bounds the fetch by the extent, and the source must lie in
       * the USC aperture, so use a small zeroed pool allocation.
       */
      if (offset >= index_range) {
         struct agx_ptr zero = hk_pool_usc_alloc(cmd, 16, 16);
         if (!zero.cpu)
            return;
         memset(zero.cpu, 0, 16);
         index_buffer = zero.gpu;
         index_range = 16;
         offset = 0;
      }
      pipeline.index_buffer = index_buffer + offset;
      if (pipeline.index_buffer < dev->dev.shader_base ||
          pipeline.index_buffer >= dev->dev.shader_base + (1ull << 32))
         UNSUPPORTED();
      pipeline.index_extent =
         agx_is_indirect(draw.b)
            ? index_range - offset
            : MIN2(index_range - offset,
                   (uint64_t)draw.b.count[0] * pipeline.index_size);
      pipeline.primitive_restart =
         emulated ? draw.restart : dyn->ia.primitive_restart_enable;
      pipeline.restart_index =
         emulated ? BITFIELD_MASK(pipeline.index_size * 8) : gfx->index.restart;
   }
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false);
   if (!cs)
      return;
   cs = hk_occlusion_prepare_draw(cmd, cs);
   if (!cs)
      return;
   /* The paired kernel parses a leading native state and one direct draw.
    * Reject mixed/multiple draws and unqualified render resource classes
    * before publishing a stream that the kernel could mistake for ordinary
    * rendering. The empty root has ample space without a stream link. */
   if (cs->apple9_native_mesh ||
       (native_grid && (cs->current != cs->start || pipeline.samples != 1 ||
                        gfx->render.layer_count != 1)))
      UNSUPPORTED();
   cs->tib = gfx->render.tilebuffer;
   const struct vk_dynamic_graphics_state *ms_dyn = &cmd->vk.dynamic_graphics_state;
   gfx->apple9_sample_positions =
      hk_apple9_draw_sample_positions(ms_dyn, pipeline.samples);
   /* hk_apple9_prepare_draw committed the render's positions; meta draws
    * only fill them in for a render that has none yet. */
   if (!cs->has_sample_locations)
      cs->ppp_multisamplectl = gfx->apple9_sample_positions;
   hk_reserve_scratch(cmd, cs, shaders[0]);
   hk_reserve_scratch(cmd, cs, shaders[1]);
   struct agx_apple9_uniform_draw *record =
      &cmd->apple9_draw[cmd->apple9_draw_current];
   agx_apple9_uniform_draw_reset(record);
   /* Rasterizer discard pipelines may omit viewport/scissor state entirely.
    * Supply harmless hardware state without consulting those ignored API
    * values or dropping the draw through the empty-scissor optimization.
    */
   const VkViewport discard_viewport = {.width = 1, .height = 1, .maxDepth = 1};
   const VkRect2D *area = &gfx->render.area;
   unsigned viewports = rasterizer_discard ? 1 : dyn->vp.viewport_count;
   record->viewport_count = viewports;
   /* Viewport scissors are consecutive; the shader's viewport index selects
    * both the transform and the scissor. The tile-granular region clip is
    * their union.
    */
   struct agx_scissor_packed scissors[AGX_APPLE9_MAX_VIEWPORTS];
   int bbox[4] = {INT_MAX, INT_MAX, INT_MIN, INT_MIN};
   for (unsigned v = 0; v < viewports; ++v) {
      const VkViewport *vp =
         rasterizer_discard ? &discard_viewport : &dyn->vp.viewports[v];
      float transform[6] = {
         vp->x + vp->width * 0.5f,  vp->width * 0.5f,
         vp->y + vp->height * 0.5f, vp->height * 0.5f,
         vp->minDepth,              vp->maxDepth - vp->minDepth,
      };
      if (v == 0) {
         for (unsigned axis = 0; axis < 3; ++axis) {
            record->viewport_translate[axis] = transform[axis * 2];
            record->viewport_scale[axis] = transform[axis * 2 + 1];
         }
      } else {
         memcpy(record->viewports[v - 1], transform, sizeof(transform));
      }
      const VkRect2D *sc = rasterizer_discard ? area : &dyn->vp.scissors[v];
      int minx = MAX2(sc->offset.x, area->offset.x);
      int miny = MAX2(sc->offset.y, area->offset.y);
      int maxx = MIN2((int64_t)sc->offset.x + sc->extent.width,
                      (int64_t)area->offset.x + area->extent.width);
      int maxy = MIN2((int64_t)sc->offset.y + sc->extent.height,
                      (int64_t)area->offset.y + area->extent.height);
      /* Clipping has a guard band, so primitives are also scissored to
       * their viewport, which may be inverted.
       */
      if (!rasterizer_discard) {
         minx = MAX2(minx, (int)floorf(MIN2(vp->x, vp->x + vp->width)));
         miny = MAX2(miny, (int)floorf(MIN2(vp->y, vp->y + vp->height)));
         maxx = MIN2(maxx, (int)ceilf(MAX2(vp->x, vp->x + vp->width)));
         maxy = MIN2(maxy, (int)ceilf(MAX2(vp->y, vp->y + vp->height)));
      }
      if (minx >= maxx || miny >= maxy) {
         /* Empty scissor: a zero-area rectangle. */
         maxx = minx = MAX2(minx, 0);
         maxy = miny = MAX2(miny, 0);
      } else {
         bbox[0] = MIN2(bbox[0], minx);
         bbox[1] = MIN2(bbox[1], miny);
         bbox[2] = MAX2(bbox[2], maxx);
         bbox[3] = MAX2(bbox[3], maxy);
      }
      /* Depth clamping bounds depth to the viewport's range, unless a range
       * is requested instead. Unclamped depth is bounded to [0, 1], which
       * is what depthClampZeroOne specifies when clamping is disabled.
       */
      float clamp_min = MIN2(vp->minDepth, vp->maxDepth);
      float clamp_max = MAX2(vp->minDepth, vp->maxDepth);
      if (dyn->vp.depth_clamp_mode ==
          VK_DEPTH_CLAMP_MODE_USER_DEFINED_RANGE_EXT) {
         clamp_min = dyn->vp.depth_clamp_range.minDepthClamp;
         clamp_max = dyn->vp.depth_clamp_range.maxDepthClamp;
      }
      agx_pack(&scissors[v], SCISSOR, cfg) {
         cfg.min_x = minx;
         cfg.min_y = miny;
         cfg.max_x = maxx;
         cfg.max_y = maxy;
         cfg.min_z = dyn->rs.depth_clamp_enable ? clamp_min : 0;
         cfg.max_z = dyn->rs.depth_clamp_enable ? clamp_max : 1;
      }
      /* The scissor range bounds fixed-function depth only; shaders clamp
       * their own depth exports (hk_apple9_clamp_depth_export). */
      if (v == 0) {
         gfx->descriptors.root.draw.depth_clamp[0] =
            dyn->rs.depth_clamp_enable ? clamp_min : 0;
         gfx->descriptors.root.draw.depth_clamp[1] =
            dyn->rs.depth_clamp_enable ? clamp_max : 1;
      }
   }
   if (bbox[0] >= bbox[2]) {
      /* Every viewport scissor is empty, so nothing rasterizes. The hardware
       * vertex program still owns pre-rasterization side effects: geometry
       * shader stores and atomics run in its rasterization variant (poly
       * emits a culled dummy primitive for them), and so do vertex shader
       * stores and transform feedback. Mesh records were written as compute.
       * Only drop draws without such effects; otherwise run against a
       * zero-area region.
       */
      bool effects = gfx->xfb_enabled;
      for (unsigned s = 0; s < MESA_SHADER_FRAGMENT && !effects; ++s)
         effects = gfx->shaders[s] && gfx->shaders[s]->writes_memory;
      effects |= native_grid && mesh && mesh->writes_memory;
      if (!effects)
         return;
      bbox[0] = bbox[1] = bbox[2] = bbox[3] = 0;
   }
   record->scissor_min[0] = bbox[0];
   record->scissor_min[1] = bbox[1];
   record->scissor_max[0] = bbox[2];
   record->scissor_max[1] = bbox[3];
   unsigned scissor_count = cs->scissor.size / AGX_SCISSOR_LENGTH;
   if (scissor_count >= viewports &&
       !memcmp((uint8_t *)cs->scissor.data +
                  (scissor_count - viewports) * AGX_SCISSOR_LENGTH,
               scissors, viewports * sizeof(scissors[0]))) {
      record->scissor_index = scissor_count - viewports;
   } else {
      assert(scissor_count + viewports <= UINT16_MAX);
      record->scissor_index = scissor_count;
      util_dynarray_append_array(&cs->scissor, struct agx_scissor_packed,
                                 scissors, viewports);
   }
   bool bias = dyn->rs.depth_bias.enable && !points && !lines;
   if (bias) {
      bool d16 = gfx->render.depth_att.vk_format == VK_FORMAT_D16_UNORM;
      bool force_unorm =
         dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
      bool exact = dyn->rs.depth_bias.exact;
      bool unorm = d16 || force_unorm;
      /* UNORM representations use a fixed depth unit: 2^-16 for D16 and
       * 2^-24 when forced, or twice that unless exact. Floating depth uses
       * the exponent-dependent unit 2^(e - 23) directly. The submission
       * selects the matching mode.
       */
      float scale = d16         ? 1.0f / (exact ? (1 << 16) : (1 << 15))
                    : force_unorm ? 1.0f / (exact ? (1 << 24) : (1 << 23))
                                  : 1.0f;
      assert(cs->cr.dbias_is_int == (unorm ? U_TRISTATE_YES : U_TRISTATE_NO));
      struct agx_depth_bias_packed depth_bias;
      agx_pack(&depth_bias, DEPTH_BIAS, cfg) {
         cfg.depth_bias = dyn->rs.depth_bias.constant_factor * scale;
         cfg.slope_scale = dyn->rs.depth_bias.slope_factor;
         cfg.clamp = dyn->rs.depth_bias.clamp;
      }
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      if (count && !memcmp((uint8_t *)cs->depth_bias.data +
                              (count - 1) * AGX_DEPTH_BIAS_LENGTH,
                           &depth_bias, sizeof(depth_bias))) {
         record->depth_bias_index = count - 1;
      } else {
         assert(count < UINT16_MAX);
         record->depth_bias_index = count;
         util_dynarray_append(&cs->depth_bias, depth_bias);
      }
   }
   bool stencil = dyn->ds.stencil.test_enable && gfx->render.stencil_att.iview;
   /* Without a depth attachment, the depth test is disabled. */
   bool depth_test = dyn->ds.depth.test_enable && gfx->render.depth_att.iview;
   record->depth_control = 0x200 | (1u << 16) | (bias ? 1u << 17 : 0) |
                          (stencil ? 3u << 18 : 0) |
                          (rasterizer_discard ? 1u << 21 : 0);
   record->occlusion_index = gfx->occlusion.index;
   record->visibility_mode = gfx->occlusion.mode;
   record->depth_control |= record->visibility_mode << 14;
   /* Early tests keep a hardware NEVER: the shader must not run at all. */
   bool force_never_in_shader =
      depth_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER &&
      shaders[1]->info.fs.writes_memory &&
      !shaders[1]->info.fs.early_fragment_tests;
   gfx->descriptors.root.draw.force_never_in_shader = force_never_in_shader;
   /* The shader fails a forced NEVER test itself (hk_apple9_clamp_depth_export). */
   VkCompareOp depth_compare = !depth_test ? VK_COMPARE_OP_ALWAYS
                               : force_never_in_shader ? VK_COMPARE_OP_LESS
                                                       : dyn->ds.depth.compare_op;
   record->depth_face[0] = record->depth_face[1] =
      (agx_pack_line_width(dyn->rs.line.width) << 8) |
      (depth_compare << 24) |
      ((depth_test && dyn->ds.depth.write_enable) ? 0 : (1 << 21)) |
      (dyn->rs.polygon_mode << 18);
   for (unsigned face = 0; face < 2; ++face) {
      const struct vk_stencil_test_face_state *st =
         face ? &dyn->ds.stencil.back : &dyn->ds.stencil.front;
      record->depth_face[face] |= st->reference & 0xff;
      agx_pack(&record->stencil[face], FRAGMENT_STENCIL, cfg) {
         cfg.compare = stencil ? st->op.compare : AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = stencil ? st->write_mask : 0xff;
         cfg.read_mask = stencil ? st->compare_mask : 0xff;
         cfg.depth_pass = stencil ? st->op.pass : AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = stencil ? st->op.depth_fail : AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = stencil ? st->op.fail : AGX_STENCIL_OP_KEEP;
      }
   }
   struct agx_cull_packed cull;
   agx_pack(&cull, CULL, cfg) {
      cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
      cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
      cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
      cfg.depth_clamp = !cfg.depth_clip;
      /* Bresenham lines cover every sample of a covered pixel. */
      cfg.primitive_msaa =
         lines && dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
   }
   memcpy(&record->raster_control, &cull, 4);
   bool front_ccw = dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
   /* The tessellator emits its domain's winding; a geometry shader emits
    * its own.
    */
   if (tes && !gs) {
      front_ccw ^= gfx->tess.info.ccw;
      front_ccw ^= dyn->ts.domain_origin ==
                   VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }
   record->raster_control |= front_ccw ? 1u << 16 : 0;
   record->object_type = points ? AGX_OBJECT_TYPE_POINT_SPRITE_UV01
                        : lines ? AGX_OBJECT_TYPE_LINE
                                : AGX_OBJECT_TYPE_TRIANGLE;
   record->reads_tile = fragment->reads_tile;
   record->uses_discard = fragment->uses_discard;
   record->writes_depth = fragment->writes_depth;
   record->disable_tri_merging = fragment->disable_tri_merging;
   record->pixel_interlock = fragment->pixel_interlock;
   record->flatshade_first = pipeline.flatshade_first;
   gfx->descriptors.root.draw.api_sample_mask = dyn->ms.sample_mask;
   gfx->descriptors.root.draw.rasterized_prim =
      (points ? 2 : lines ? 1 : 0) |
      (!emulated &&
       dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN &&
       dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
          ? 4 : 0);
   gfx->descriptors.root.draw.draw_id = draw_id;
   gfx->descriptors.root_dirty = true;
   uint64_t t_sub = hk_cpu_sub_begin();
   hk_cmd_buffer_flush_push_descriptors(cmd, &gfx->descriptors);
   hk_cpu_sub_end(t_sub, HK_CPU_SUB_PUSH_DESC);
   t_sub = hk_cpu_sub_begin();
   gfx->root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   hk_cpu_sub_end(t_sub, HK_CPU_SUB_ROOT);
   struct hk_apple9_prerast_bindings bindings =
      hk_apple9_draw_bindings(cmd, draw);
   uint64_t draw_params = bindings.draw_params;
   uint64_t vertex_table = bindings.vertex_buffers;
   for (unsigned stage = 0; stage < 2; ++stage) {
      const struct agx_apple9_render_stage *s =
         stage ? fragment : vertex;
      uint64_t sysvals = 0;
      t_sub = hk_cpu_sub_begin();
      bool textures_ok = hk_apple9_upload_textures(
         cmd, shaders[stage], true,
         pipeline.primitive == AGX_PRIMITIVE_TRIANGLE_FAN, lines,
         &record->texture_table[stage], &record->sampler_table[stage], &sysvals);
      hk_cpu_sub_end(t_sub, HK_CPU_SUB_TEXTURES);
      if (!textures_ok)
         UNSUPPORTED();
      t_sub = hk_cpu_sub_begin();
      for (unsigned slot = 0; slot < s->resource_count; ++slot) {
         unsigned binding = s->resource_binding[slot];
         uint64_t address;
         if (binding == 0)
            address = gfx->root;
         else if (binding == AGX_APPLE9_GRAPHICS_SYSVAL_BINDING)
            address = sysvals;
         else if (binding == HK_APPLE9_DRAW_PARAMS_BINDING)
            address = draw_params;
         else if (binding == HK_APPLE9_VERTEX_BUFFERS_BINDING)
            address = vertex_table;
         else if (binding >= 32 && binding < 32 + ARRAY_SIZE(gfx->vb))
            address = gfx->vb[binding - 32].addr;
         else
            UNSUPPORTED();
         record->buffers[stage][slot] = address;
      }
      uint64_t table = hk_pool_upload_last(
         cmd, stage ? HK_UPLOAD_LAST_FRAGMENT_TABLE : HK_UPLOAD_LAST_VERTEX_TABLE,
         record->buffers[stage], s->resource_count * sizeof(uint64_t), 64);
      if (stage)
         record->fragment_table = table;
      else
         record->vertex_table = table;
      hk_cpu_sub_end(t_sub, HK_CPU_SUB_TABLES);
   }
   const struct agx_apple9_uniform_draw *previous =
      cmd->apple9_prev_cs == cs ? &cmd->apple9_draw[!cmd->apple9_draw_current]
                                : NULL;
   t_sub = hk_cpu_sub_begin();
   bool prepared = agx_apple9_prepare_draw_alloc(
      &dev->dev, cmd, hk_apple9_usc_alloc, cmd, hk_apple9_context_alloc,
      &pipeline, record, previous, hk_apple9_state_cache_alloc);
   hk_cpu_sub_end(t_sub, HK_CPU_SUB_PREPARE_ALLOC);
   if (!prepared)
      UNSUPPORTED();
   cmd->apple9_draw_current = !cmd->apple9_draw_current;
   cmd->apple9_prev_cs = cs;
   pipeline.ppp = record->ppp;
   pipeline.ppp_viewport_offset = record->ppp_viewport_offset;
   pipeline.ppp_viewport_words = record->ppp_viewport_words;
   pipeline.vertex_launch = record->launch[0] / 0x40;
   pipeline.pipeline_word = 0x01000000;
   hk_ensure_cs_has_space(cmd, cs, 256);
   if (native_grid) {
      uint32_t state[AGX_APPLE9_MESH_STATE_WORDS];
      const uint32_t object_groups[] = {1, 1, 1};
      if (!hk_apple9_prepare_native_mesh(
             cmd, shaders[0], &mesh->variants[HK_MESH_VARIANT_OBJECT],
             record, native_grid, state) ||
          !hk_apple9_mesh_command_build(
             cs->current, HK_APPLE9_MESH_COMMAND_BYTES, &pipeline, state,
             object_groups, hk_apple9_native_mesh_enabled()))
         UNSUPPORTED();
      cs->current += HK_APPLE9_MESH_COMMAND_BYTES - 4;
      cs->apple9_native_mesh = true;
   } else if (agx_is_indirect(draw.b))
      cs->current =
         agx_apple9_emit_indirect_draw(cs->current, &pipeline, draw.b.ptr) - 4;
   else
      cs->current = agx_apple9_emit_direct_draw(
                       cs->current, &pipeline, draw.b.count[0], draw.b.count[1],
                       draw.indexed ? draw.index_bias : draw.start) -
                    4;
   if (cs->primitive_count < AGX_PRIMITIVE_COUNT_UNKNOWN) {
      agx_add_primitive_count(
         &cs->primitive_count,
         native_grid ? 1
         : agx_is_indirect(draw.b) || pipeline.primitive_restart || gs || tes || mesh
            ? AGX_PRIMITIVE_COUNT_UNKNOWN
            : agx_count_primitives(pipeline.primitive, draw.b.count[0],
                                   draw.b.count[1]));
   }
   cs->stats.cmds++;
   if (!cs->pass_count++)
      cs->pass_code = shaders[1]->bo ? shaders[1]->bo->va->addr : 0;
   return;
unsupported:
   fprintf(
      stderr,
      "Apple9 draw unsupported at %u: shaders=%d/%d indirect=%d vp=%u sc=%u stencil=%d bias=%d topology=%u\n",
      failure_line, !!shaders[0], !!shaders[1], agx_is_indirect(draw.b),
      dyn->vp.viewport_count, dyn->vp.scissor_count,
      dyn->ds.stencil.test_enable, dyn->rs.depth_bias.enable,
      dyn->ia.primitive_topology);
#undef UNSUPPORTED
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
}

void
hk_apple9_draw(struct hk_cmd_buffer *cmd, uint32_t draw_id, struct agx_draw draw)
{
   hk_apple9_draw_impl(cmd, draw_id, draw, 0);
}

void
hk_apple9_draw_native_mesh(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                            uint64_t grid)
{
   hk_apple9_draw_impl(cmd, draw_id,
                       (struct agx_draw){.b = agx_3d(3, 1, 1)}, grid);
}
