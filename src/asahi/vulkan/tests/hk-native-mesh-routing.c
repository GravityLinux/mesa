/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "../hk_cmd_buffer.h"
#include "../hk_shader.h"

#define CHECK(c)                                                               \
   do {                                                                        \
      if (!(c)) {                                                              \
         fprintf(stderr, "%u: %s\n", __LINE__, #c);                            \
         abort();                                                              \
      }                                                                        \
   } while (0)

static struct hk_api_shader *
shader(mesa_shader_stage stage)
{
   struct hk_api_shader *s =
      calloc(1, sizeof(*s) + hk_num_variants(stage) * sizeof(struct hk_shader));
   CHECK(s);
   s->vk.stage = stage;
   return s;
}

static void
routes(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *mesh = shader(MESA_SHADER_MESH);
   struct hk_api_shader *fs = shader(MESA_SHADER_FRAGMENT);
   struct hk_api_shader *nm = shader(MESA_SHADER_MESH),
                        *nf = shader(MESA_SHADER_FRAGMENT);
   mesh->apple9_native = nm;
   fs->apple9_native = nf;
   nm->apple9_native_leaf = nf->apple9_native_leaf = true;
   memset(nm->apple9_native_pair_id, 0x57, sizeof(nm->apple9_native_pair_id));
   memcpy(nf->apple9_native_pair_id, nm->apple9_native_pair_id,
          sizeof(nf->apple9_native_pair_id));
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native_stage =
      HK_APPLE9_NATIVE_MESH;
   gfx->shaders[MESA_SHADER_MESH] = mesh;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   const uint32_t direct[] = {2, 3, 4}, oversized[] = {1024, 1024, 1};
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, NULL, 0));
   CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0x1000));
   CHECK(!hk_apple9_use_native_mesh(cmd, oversized, 0));
#define FALLBACK(field, value)                                                 \
   do {                                                                        \
      __typeof__(field) saved = field;                                         \
      field = value;                                                           \
      CHECK(!hk_apple9_use_native_mesh(cmd, direct, 0));                       \
      field = saved;                                                           \
      CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));                        \
   } while (0)
   FALLBACK(mesh->apple9_native, NULL);
   FALLBACK(fs->apple9_native, NULL);
   FALLBACK(nf->apple9_native_pair_id[0], 0x35);
   FALLBACK(nf->apple9_native_pair_id[31], 0x35);
   FALLBACK(nf->apple9_native_leaf, false);
   /* Deferred native variants are themselves leaves. Pairing is checked
    * identically to the alternatives of an eagerly compiled pipeline. */
   gfx->shaders[MESA_SHADER_MESH] = nm;
   gfx->shaders[MESA_SHADER_FRAGMENT] = nf;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
   FALLBACK(nf->apple9_native_pair_id[7], 0x35);
   gfx->shaders[MESA_SHADER_MESH] = mesh;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   FALLBACK(gfx->shaders[MESA_SHADER_TASK], mesh);
   FALLBACK(gfx->shaders[MESA_SHADER_GEOMETRY], mesh);
   FALLBACK(gfx->shaders[MESA_SHADER_TESS_EVAL], mesh);
   FALLBACK(gfx->render.view_mask, 3);
   FALLBACK(gfx->render.layer_count, 2);
   FALLBACK(gfx->mesh_pg_query, 0x2000);
   FALLBACK(gfx->descriptors.root.draw.pipeline_stats_flags,
            VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT);
   FALLBACK(gfx->occlusion.active_count, 1);
   FALLBACK(cmd->state.cond_render.active, true);
   FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
   FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_POINT);
   FALLBACK(dyn->ms.rasterization_samples, 4);
   FALLBACK(gfx->render.tilebuffer.nr_samples, 4);
   FALLBACK(gfx->xfb_enabled, true);
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.primitive_broadcast =
      VARYING_BIT_CULL_PRIMITIVE;
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
   /* CullDistance alone also removes a whole triangle with degenerate
    * indices. A cached native pair must fall back under conservative raster. */
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.primitive_broadcast = 0;
   nm->variants[HK_MESH_VARIANT_MAIN].info.apple9_native.cull_distances = 2;
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT);
   gfx->shaders[MESA_SHADER_MESH] = nm;
   gfx->shaders[MESA_SHADER_FRAGMENT] = nf;
   FALLBACK(dyn->rs.conservative_mode,
            VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
   gfx->shaders[MESA_SHADER_MESH] = mesh;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->shaders[MESA_SHADER_FRAGMENT] = NULL;
   CHECK(hk_apple9_use_native_mesh(cmd, direct, 0));
#undef FALLBACK
   memset(gfx->shaders, 0, sizeof(gfx->shaders));
   free(mesh);
   free(nm);
   free(fs);
   free(nf);
}

static void
gs_routes(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *gs = shader(MESA_SHADER_GEOMETRY);
   struct hk_api_shader *vs = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *fs = shader(MESA_SHADER_FRAGMENT);
   struct hk_api_shader *ng = shader(MESA_SHADER_GEOMETRY);
   struct hk_api_shader *nv = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *nf = shader(MESA_SHADER_FRAGMENT);
   gs->apple9_native = ng; vs->apple9_native = nv; fs->apple9_native = nf;
   ng->apple9_native_leaf = nv->apple9_native_leaf = nf->apple9_native_leaf = true;
   memset(ng->apple9_native_pair_id, 0x71, sizeof(ng->apple9_native_pair_id));
   memcpy(nv->apple9_native_pair_id, ng->apple9_native_pair_id, sizeof(ng->apple9_native_pair_id));
   memcpy(nf->apple9_native_pair_id, ng->apple9_native_pair_id, sizeof(ng->apple9_native_pair_id));
   struct hk_shader *main = hk_native_producer_main(ng);
   main->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   hk_native_producer_object(ng)->info.apple9_native_stage = HK_APPLE9_NATIVE_OBJECT;
   main->info.apple9_native.key.physical_threads = 16;
   main->info.apple9_native.layout.max_vertices = 3;
   main->info.apple9_native.layout.max_primitives = 1;
   gfx->shaders[MESA_SHADER_GEOMETRY] = gs;
   gfx->shaders[MESA_SHADER_VERTEX] = vs;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   struct agx_draw draw = {.b = agx_3d(3, 1, 1)};
   bool enabled = hk_apple9_native_gs_enabled();
   CHECK(hk_apple9_use_native_gs(cmd, draw) == enabled);
   if (enabled) {
#define FALLBACK(field, value)                                                 \
   do {                                                                        \
      __typeof__(field) saved = field;                                         \
      field = value;                                                           \
      CHECK(!hk_apple9_use_native_gs(cmd, draw));                               \
      field = saved;                                                           \
      CHECK(hk_apple9_use_native_gs(cmd, draw));                                \
   } while (0)
      FALLBACK(cmd->in_meta, true);
      FALLBACK(gs->apple9_native, NULL);
      FALLBACK(vs->apple9_native, NULL);
      FALLBACK(fs->apple9_native, NULL);
      FALLBACK(ng->apple9_native_leaf, false);
      FALLBACK(nv->apple9_native_pair_id[0], 0x13);
      FALLBACK(nf->apple9_native_pair_id[31], 0x13);
      FALLBACK(gs->is_passthrough, true);
      FALLBACK(nv->writes_memory, true);
      FALLBACK(main->info.apple9_native_stage, HK_APPLE9_NATIVE_NONE);
      FALLBACK(hk_native_producer_object(ng)->info.apple9_native_stage, HK_APPLE9_NATIVE_NONE);
      FALLBACK(main->info.apple9_native.key.physical_threads, 32);
      FALLBACK(main->info.apple9_native.layout.max_vertices, 5);
      FALLBACK(draw.indexed, true);
      FALLBACK(draw.restart, true);
      FALLBACK(draw.start, 1);
      FALLBACK(draw.start_instance, 1);
      FALLBACK(draw.b.count[0], 6);
      FALLBACK(draw.b.count[1], 2);
      FALLBACK(draw.b.count[2], 2);
      CHECK(!hk_apple9_use_native_gs(cmd, agx_draw_indirect(0x1000)));
      FALLBACK(dyn->ia.primitive_restart_enable, true);
      FALLBACK(dyn->ia.primitive_topology, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
      FALLBACK(gfx->shaders[MESA_SHADER_TESS_CTRL], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_TESS_EVAL], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_TASK], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_MESH], gs);
      FALLBACK(gfx->render.view_mask, 3);
      FALLBACK(gfx->render.layer_count, 2);
      FALLBACK(cmd->state.cond_render.active, true);
      FALLBACK(gfx->descriptors.root.draw.pipeline_stats_flags, 1);
      FALLBACK(gfx->occlusion.active_count, 1);
      FALLBACK(gfx->mesh_pg_query, 0x1000);
      FALLBACK(gfx->pg_query[0], 0x1000);
      FALLBACK(gfx->xfb_query[3], 0x1000);
      FALLBACK(gfx->xfb_enabled, true);
      FALLBACK(dyn->rs.rasterization_stream, 1);
      FALLBACK(dyn->rs.rasterizer_discard_enable, true);
      FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
      FALLBACK(dyn->rs.conservative_mode, VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
      FALLBACK(dyn->ms.rasterization_samples, 4);
      FALLBACK(gfx->render.tilebuffer.nr_samples, 4);
#undef FALLBACK
      main->info.apple9_native.layout.max_vertices = 4;
      main->info.apple9_native.layout.max_primitives = 2;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      gfx->shaders[MESA_SHADER_GEOMETRY] = ng;
      gfx->shaders[MESA_SHADER_VERTEX] = nv;
      gfx->shaders[MESA_SHADER_FRAGMENT] = nf;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
      gfx->shaders[MESA_SHADER_FRAGMENT] = NULL;
      CHECK(hk_apple9_use_native_gs(cmd, draw));
   }
   memset(gfx->shaders, 0, sizeof(gfx->shaders));
   free(gs); free(vs); free(fs); free(ng); free(nv); free(nf);
}

static void
tess_routes(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_api_shader *gs = shader(MESA_SHADER_TESS_EVAL);
   struct hk_api_shader *tcs = shader(MESA_SHADER_TESS_CTRL);
   struct hk_api_shader *nc = shader(MESA_SHADER_TESS_CTRL);
   struct hk_api_shader *vs = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *fs = shader(MESA_SHADER_FRAGMENT);
   struct hk_api_shader *ng = shader(MESA_SHADER_TESS_EVAL);
   struct hk_api_shader *nv = shader(MESA_SHADER_VERTEX);
   struct hk_api_shader *nf = shader(MESA_SHADER_FRAGMENT);
   gs->apple9_native = ng;
   vs->apple9_native = nv;
   fs->apple9_native = nf;
   tcs->apple9_native = nc;
   nc->apple9_native_leaf = true;
   ng->apple9_native_leaf = nv->apple9_native_leaf = nf->apple9_native_leaf =
      true;
   memset(ng->apple9_native_pair_id, 0x71, sizeof(ng->apple9_native_pair_id));
   memcpy(nv->apple9_native_pair_id, ng->apple9_native_pair_id,
          sizeof(ng->apple9_native_pair_id));
   memcpy(nf->apple9_native_pair_id, ng->apple9_native_pair_id,
          sizeof(ng->apple9_native_pair_id));
   memcpy(nc->apple9_native_pair_id, ng->apple9_native_pair_id,
          sizeof(ng->apple9_native_pair_id));
   struct hk_shader *main = hk_native_producer_main(ng);
   main->info.apple9_native_stage = HK_APPLE9_NATIVE_MESH;
   hk_native_producer_object(ng)->info.apple9_native_stage =
      HK_APPLE9_NATIVE_OBJECT;
   main->info.apple9_native.key.physical_threads = 16;
   main->info.apple9_native.layout.max_vertices = 3;
   main->info.apple9_native.layout.max_primitives = 1;
   gfx->shaders[MESA_SHADER_TESS_EVAL] = gs;
   gfx->shaders[MESA_SHADER_TESS_CTRL] = tcs;
   gfx->shaders[MESA_SHADER_VERTEX] = vs;
   gfx->shaders[MESA_SHADER_FRAGMENT] = fs;
   gfx->render.layer_count = 1;
   dyn->rs.polygon_mode = VK_POLYGON_MODE_FILL;
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
   struct agx_draw draw = {.b = agx_3d(3, 1, 1)};
   dyn->ts.patch_control_points = 3;
   bool enabled = hk_apple9_native_tess_enabled();
   CHECK(hk_apple9_use_native_tess(cmd, draw) == enabled);
   if (enabled) {
#define FALLBACK(field, value)                                                 \
   do {                                                                        \
      __typeof__(field) saved = field;                                         \
      field = value;                                                           \
      CHECK(!hk_apple9_use_native_tess(cmd, draw));                            \
      field = saved;                                                           \
      CHECK(hk_apple9_use_native_tess(cmd, draw));                             \
   } while (0)
      FALLBACK(cmd->in_meta, true);
      FALLBACK(gs->apple9_native, NULL);
      FALLBACK(vs->apple9_native, NULL);
      FALLBACK(fs->apple9_native, NULL);
      FALLBACK(ng->apple9_native_leaf, false);
      FALLBACK(nv->apple9_native_pair_id[0], 0x13);
      FALLBACK(nf->apple9_native_pair_id[31], 0x13);
      FALLBACK(tcs->apple9_native, NULL);
      FALLBACK(nc->apple9_native_pair_id[0], 0x13);
      FALLBACK(nc->writes_memory, true);
      FALLBACK(dyn->ts.patch_control_points, 0);
      FALLBACK(dyn->ts.patch_control_points, 33);
      FALLBACK(nv->writes_memory, true);
      FALLBACK(main->info.apple9_native_stage, HK_APPLE9_NATIVE_NONE);
      FALLBACK(hk_native_producer_object(ng)->info.apple9_native_stage,
               HK_APPLE9_NATIVE_NONE);
      FALLBACK(main->info.apple9_native.key.physical_threads, 32);
      FALLBACK(main->info.apple9_native.layout.max_vertices, 5);
      FALLBACK(draw.indexed, true);
      FALLBACK(draw.restart, true);
      FALLBACK(draw.start, 1);
      FALLBACK(draw.start_instance, 1);
      FALLBACK(draw.b.count[0], 6);
      FALLBACK(draw.b.count[1], 2);
      FALLBACK(draw.b.count[2], 2);
      CHECK(!hk_apple9_use_native_tess(cmd, agx_draw_indirect(0x1000)));
      FALLBACK(dyn->ia.primitive_restart_enable, true);
      FALLBACK(dyn->ia.primitive_topology,
               VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
      FALLBACK(gfx->shaders[MESA_SHADER_TESS_CTRL], NULL);
      FALLBACK(gfx->shaders[MESA_SHADER_GEOMETRY], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_TASK], gs);
      FALLBACK(gfx->shaders[MESA_SHADER_MESH], gs);
      FALLBACK(gfx->render.view_mask, 3);
      FALLBACK(gfx->render.layer_count, 2);
      FALLBACK(cmd->state.cond_render.active, true);
      FALLBACK(gfx->descriptors.root.draw.pipeline_stats_flags, 1);
      FALLBACK(gfx->occlusion.active_count, 1);
      FALLBACK(gfx->mesh_pg_query, 0x1000);
      FALLBACK(gfx->pg_query[0], 0x1000);
      FALLBACK(gfx->xfb_query[3], 0x1000);
      FALLBACK(gfx->xfb_enabled, true);
      FALLBACK(dyn->rs.rasterization_stream, 1);
      FALLBACK(dyn->rs.rasterizer_discard_enable, true);
      FALLBACK(dyn->rs.polygon_mode, VK_POLYGON_MODE_LINE);
      FALLBACK(dyn->rs.conservative_mode,
               VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
      FALLBACK(dyn->ms.rasterization_samples, 4);
      FALLBACK(gfx->render.tilebuffer.nr_samples, 4);
#undef FALLBACK
      for (unsigned p = 1; p <= 32; ++p) {
         draw.b.count[0] = dyn->ts.patch_control_points = p;
         CHECK(hk_apple9_use_native_tess(cmd, draw));
      }
      gfx->shaders[MESA_SHADER_TESS_EVAL] = ng;
      gfx->shaders[MESA_SHADER_VERTEX] = nv;
      gfx->shaders[MESA_SHADER_FRAGMENT] = nf;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
      gfx->shaders[MESA_SHADER_FRAGMENT] = NULL;
      CHECK(hk_apple9_use_native_tess(cmd, draw));
   }
   memset(gfx->shaders, 0, sizeof(gfx->shaders));
   free(tcs);
   free(nc);
   free(gs);
   free(vs);
   free(fs);
   free(ng);
   free(nv);
   free(nf);
}

static void
mark_draw(struct hk_cs *cs, bool native)
{
   /* The classification/transition test owns CPU storage, not GPU execution.
    * Reserve representative stream bytes before the real next preparation. */
   cs->current += 160;
   cs->stats.cmds++;
   cs->apple9_native_mesh = native;
}

static void
transitions(struct hk_cmd_buffer *cmd)
{
   uint8_t *pool = calloc(2, HK_CMD_BO_SIZE);
   CHECK(pool);
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = 0x1400000000ull;
   cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].map = pool + HK_CMD_BO_SIZE;
   cmd->uploader[HK_CMD_POOL_CONTEXT].chunks[0].base = 0x1500000000ull;
   cmd->sync_events = UTIL_DYNARRAY_INIT;
   cmd->sync_remap = UTIL_DYNARRAY_INIT;
   cmd->sync_first_event = UTIL_DYNARRAY_INIT;
   list_inithead(&cmd->control_streams);
   struct hk_rendering_state *r = &cmd->state.gfx.render;
   r->cr.bg.partial.usc = 0x1000;
   r->cr.eot.partial.usc = 0x2000;
   memset(&r->cr.zls_control_partial, 0x35, sizeof(r->cr.zls_control_partial));
   CHECK(hk_apple9_prepare_draw(cmd, false, false));
   struct hk_cs *first = cmd->current_cs.gfx;
   CHECK(first && !first->stats.cmds);
   /* The first native draw can use an empty stream. */
   CHECK(hk_apple9_prepare_draw(cmd, true, true));
   CHECK(cmd->current_cs.gfx == first);
   mark_draw(first, true);
   const bool paths[] = {false, true, true, false, true, false};
   for (unsigned i = 0; i < ARRAY_SIZE(paths); ++i) {
      struct hk_cs *prev = cmd->current_cs.gfx;
      unsigned old_events =
         util_dynarray_num_elements(&cmd->sync_events, struct hk_sync_event);
      /* The final transition is an ordinary meta draw after native. */
      cmd->in_meta = i == ARRAY_SIZE(paths) - 1;
      CHECK(hk_apple9_prepare_draw(cmd, paths[i], paths[i]));
      struct hk_cs *next = cmd->current_cs.gfx;
      CHECK(next != prev && prev->end_reason == HK_CS_END_RENDER_SPLIT);
      CHECK(next->current == next->start && !next->apple9_native_mesh);
      CHECK(next->cr.bg.main.usc == r->cr.bg.partial.usc);
      CHECK(prev->cr.eot.main.usc == r->cr.eot.partial.usc);
      CHECK(!memcmp(&next->cr.zls_control, &r->cr.zls_control_partial,
                    sizeof(next->cr.zls_control)));
      const struct hk_sync_event *events = cmd->sync_events.data;
      CHECK(util_dynarray_num_elements(&cmd->sync_events,
                                       struct hk_sync_event) >= old_events + 2);
      CHECK(events[old_events].stream == HK_SYNC_NONE);
      CHECK(events[old_events].src == prev->sync_stages);
      CHECK(events[old_events].dst & HK_SYNC_FRAGMENT);
      CHECK(events[old_events + 1].stream == next->sync_id);
      mark_draw(next, paths[i]);
   }
   cmd->in_meta = false;
   struct hk_cs *last = cmd->current_cs.gfx;
   CHECK(hk_apple9_prepare_draw(cmd, false, false));
   CHECK(cmd->current_cs.gfx == last); /* ordinary draws still coalesce */
   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      list_del(&cs->node);
      hk_cs_destroy(cs);
   }
   util_dynarray_fini(&cmd->sync_events);
   util_dynarray_fini(&cmd->sync_remap);
   util_dynarray_fini(&cmd->sync_first_event);
   free(pool);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   CHECK(dev && cmd);
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_RECORDING;
   routes(cmd);
   gs_routes(cmd);
   tess_routes(cmd);
   transitions(cmd);
   free(cmd);
   free(dev);
   puts(
      "hybrid draw routing, partial attachment state and continuation ordering pass");
   return 0;
}
