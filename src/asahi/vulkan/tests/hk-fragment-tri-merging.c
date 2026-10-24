/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * Triangle merging shades adjacent strip and fan triangles on one plane, which
 * is inexact away from the pixel centre. Compile real fragment shaders (BO
 * allocation replaced with host memory) and check that off-centre
 * interpolation disables merging whether or not the statistics counter is
 * compiled in, that pixel-centre interpolation keeps it, and that the choice
 * survives serialization.
 */
#include "../hk_shader.c"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); abort(); } } while (0)
#include "hk-test-usc.h"

struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_reference(struct agx_bo *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   static unsigned allocations;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   CHECK(bo);
   bo->va = calloc(1, sizeof(*bo->va));
   bo->_map = calloc(1, size);
   CHECK(bo->va && bo->_map);
   bo->dev = dev;
   bo->size = size;
   bo->refcnt = 1;
   bo->va->addr = dev->shader_base + ++allocations * 0x100000;
   return bo;
}

void
__wrap_agx_bo_reference(struct agx_bo *bo)
{
   ++bo->refcnt;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo || --bo->refcnt)
      return;
   free(bo->_map);
   free(bo->va);
   free(bo);
}

static nir_shader *
mesh_source(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH, &agx_nir_options,
                                                  "tri merging mesh");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = b.shader->info.workgroup_size[1] =
      b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3), nir_imm_int(&b, 1),
                                      nir_imm_int(&b, 0));
   for (unsigned v = 0; v < 3; ++v) {
      nir_store_per_vertex_output(&b,
         nir_imm_vec4(&b, v == 0 ? -.5 : .5, v == 2 ? .5 : -.5, .5, 1),
         nir_imm_int(&b, v), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, v, 1, 2, 3),
         nir_imm_int(&b, v), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }
   nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static nir_shader *
fragment_source(bool centroid)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                                  "tri merging fragment");
   /* An application shader: hk does not gate internal ones. */
   b.shader->info.internal = false;
   b.shader->info.io_lowered = true;
   nir_def *bary = centroid
      ? nir_load_barycentric_centroid(&b, 32, .interp_mode = INTERP_MODE_SMOOTH)
      : nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *value = nir_load_interpolated_input(&b, 4, 32, bary, nir_imm_int(&b, 0),
      .dest_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static bool
roundtrip_disables(struct hk_device *dev, struct hk_api_shader *fs)
{
   struct blob blob;
   blob_init(&blob);
   CHECK(hk_api_shader_serialize(&dev->vk, &fs->vk, &blob));
   struct blob_reader reader;
   blob_reader_init(&reader, blob.data, blob.size);
   struct vk_shader *result = NULL;
   CHECK(hk_deserialize_api_shader(&dev->vk, &reader, 0, NULL, &result) == VK_SUCCESS);
   blob_finish(&blob);
   struct hk_api_shader *copy = container_of(result, struct hk_api_shader, vk);
   bool disable = copy->variants[0].b.info.disable_tri_merging;
   copy->vk.ops->destroy(&dev->vk, &copy->vk, NULL);
   return disable;
}

/* Returns whether the compiled fragment shader disables triangle merging. */
static bool
disables_merging(struct hk_device *dev, bool centroid, bool statistics)
{
   struct vk_shader_compile_info infos[] = {
      {.stage = MESA_SHADER_MESH, .nir = mesh_source(),
       .robustness = &vk_robustness_disabled},
      {.stage = MESA_SHADER_FRAGMENT, .nir = fragment_source(centroid),
       .robustness = &vk_robustness_disabled},
   };
   struct vk_shader *out[2] = {0};
   const struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   const struct vk_color_blend_state cb = {
      .attachment_count = 1, .color_write_enables = 1,
      .attachments = {{.write_mask = 15}},
   };
   const struct vk_graphics_pipeline_state state = {.rp = &rp, .cb = &cb};
   struct hk_apple9_graphics_key key;
   hk_apple9_graphics_key(&key, &state);
   key.no_fs_statistics = !statistics;
   CHECK(hk_apple9_compile_graphics(dev, 2, infos, NULL, &key, NULL, NULL, out) ==
         VK_SUCCESS);
   struct hk_api_shader *fs = container_of(out[1], struct hk_api_shader, vk);
   bool disable = fs->variants[0].b.info.disable_tri_merging;
   CHECK(roundtrip_disables(dev, fs) == disable);
   for (unsigned i = 0; i < 2; ++i)
      out[i]->ops->destroy(&dev->vk, out[i], NULL);
   return disable;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(dev && pdev);
   dev->dev.chip = AGX_CHIP_G16G;
   dev->dev.shader_base = 0x1200000000ull;
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &pdev->vk;
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&dev->apple9_variants.lock, mtx_plain);
   simple_mtx_init(&dev->apple9_bodies.lock, mtx_plain);
   list_inithead(&dev->apple9_bodies.slabs);
   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);

   /* Off-centre interpolation keeps triangles apart, counter or not. */
   CHECK(disables_merging(dev, true, false));
   CHECK(disables_merging(dev, true, true));
   /* Pixel-centre interpolation merges, also with the counter. */
   CHECK(!disables_merging(dev, false, false));
   CHECK(!disables_merging(dev, false, true));

   simple_mtx_destroy(&pdev->debug_compile_lock);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&dev->apple9_variants.lock);
   test_reset_usc_heap(dev);
   hk_device_finish_usc_heap(dev);
   free(dev);
   free(pdev);
   glsl_type_singleton_decref();
   puts("off-centre interpolation disables triangle merging with and without statistics");
   return 0;
}
