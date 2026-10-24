/* SPDX-License-Identifier: MIT
 * CPU-only identity corpus using the real driver compile/serialization paths.
 * Reuse the hybrid ownership test's mock BO allocator; never open a device.
 */
int hybrid_reference_main(void);
#define main hybrid_reference_main
#include "hk-native-mesh-hybrid.c"
#undef main
#include <sys/stat.h>

static nir_shader *
index_source(unsigned v, unsigned p, unsigned t)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "driver packed publication");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = t;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = v;
   b.shader->info.mesh.max_primitives_out = p;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, v),
                                    nir_imm_int(&b, p), nir_imm_int(&b, 0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b, nir_ult_imm(&b, lane, v));
   nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 0, 0, .5, 1), lane,
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir_pop_if(&b, NULL);
   for (unsigned i = 0; i < p; ++i) {
      nir_push_if(&b, nir_ieq_imm(&b, lane, i % t));
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, i), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static unsigned cases, objects, natives;
static bool lower_common;
static FILE *ledger;
static const char *directory;

static void
record(struct hk_device *dev, const char *name, unsigned count,
       struct vk_shader_compile_info *infos)
{
   const struct vk_render_pass_state rp = {
      .color_attachment_count = 1,
      .color_attachment_formats = {VK_FORMAT_R8G8B8A8_UNORM},
   };
   const struct vk_color_blend_state cb = {
      .attachment_count = 1, .color_write_enables = 1,
      .attachments = {{.write_mask = 15}},
   };
   const struct vk_graphics_pipeline_state state = {.rp = &rp, .cb = &cb};
   struct vk_shader *out[5] = {0};
   allocations = 0;
   if (lower_common) {
      for (unsigned i = 0; i < count; ++i)
         hk_lower_nir(dev, infos[i].nir, infos[i].robustness, false,
                      0, NULL, 0);
   }
   VkResult result = hk_apple9_compile_graphics(dev, count, infos, &state,
                                               NULL, NULL, NULL, out);
   if (result != VK_SUCCESS)
      fprintf(stderr, "%s: compile error %d\n", name, result);
   CHECK(result == VK_SUCCESS);
   for (unsigned i = 0; i < count; ++i) {
      struct hk_api_shader *api = container_of(out[i], struct hk_api_shader, vk);
      struct blob blob;
      blob_init(&blob);
      CHECK(hk_api_shader_serialize(&dev->vk, out[i], &blob));
      CHECK(!blob.out_of_memory);
      char path[4096];
      CHECK(snprintf(path, sizeof(path), "%s/%s-%u.bin", directory, name, i) < sizeof(path));
      FILE *f = fopen(path, "wb"); CHECK(f);
      CHECK(fwrite(blob.data, 1, blob.size, f) == blob.size);
      CHECK(fclose(f) == 0);
      fprintf(ledger, "%s\t%u\t%u\t%u\t%zu\n", name, i,
              api->vk.stage, api->apple9_native != NULL, blob.size);
      natives += api->apple9_native != NULL;
      objects++;
      blob_finish(&blob);
      hk_api_shader_destroy(&dev->vk, out[i], NULL);
   }
   hk_apple9_variants_finish(dev);
   hk_apple9_bodies_finish(dev);
   test_reset_usc_heap(dev);
   CHECK(live_bos == 0);
   fflush(ledger);
   cases++;
}

static void
mesh_case(struct hk_device *dev, unsigned t, unsigned v, unsigned p, unsigned rejection)
{
   nir_shader *nir = index_source(v, p, t);
   switch (rejection) {
   case 1: nir->info.shared_size = 16; break;
   case 2: nir->info.task_payload_size = 16; break;
   case 3: nir->info.workgroup_size[0] = t/2; nir->info.workgroup_size[1] = 2; break;
   case 4: nir->info.workgroup_size[0] = 64; break;
   case 6: {
      nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
      nir_store_per_primitive_output(&b, nir_imm_float(&b, 1), nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      break;
   }
   case 5: {
      nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
      nir_store_per_vertex_output(&b, nir_imm_float(&b, 1), nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      break;
   }
   default: break;
   }
   struct vk_shader_compile_info infos[] = {
      {.stage=MESA_SHADER_MESH,.nir=nir,.robustness=&vk_robustness_disabled},
      {.stage=MESA_SHADER_FRAGMENT,.nir=gs_source(MESA_SHADER_FRAGMENT,0),.robustness=&vk_robustness_disabled},
   };
   if (rejection == 6) {
      nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(infos[1].nir)));
      nir_def *value = nir_load_per_primitive_input(&b, 1, 32, nir_imm_int(&b, 0),
         .dest_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      nir_store_output(&b, nir_vec4(&b, value, value, value, nir_imm_float(&b, 1)),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      nir_shader_gather_info(infos[1].nir, nir_shader_get_entrypoint(infos[1].nir));
   }
   char name[80];snprintf(name,sizeof(name),"T%uV%uP%u-R%u",t,v,p,rejection);
   record(dev,name,ARRAY_SIZE(infos),infos);
}

int main(int argc, char **argv)
{
   CHECK(argc >= 2 && argc <= 4);
   bool drirc_kill = false;
   for (unsigned i = 2; i < argc; ++i) {
      CHECK(!strcmp(argv[i], "drirc-kill") || !strcmp(argv[i], "prelower"));
      drirc_kill |= !strcmp(argv[i], "drirc-kill");
      lower_common |= !strcmp(argv[i], "prelower");
   }
   directory=argv[1];CHECK(mkdir(directory,0700)==0);
   char path[4096];snprintf(path,sizeof(path),"%s/PROGRAMS.tsv",directory);
   ledger=fopen(path,"w");CHECK(ledger);
   fputs("case\tobject\tstage\tnative\tbytes\n",ledger);
   glsl_type_singleton_init_or_ref();
   struct hk_device *dev=calloc(1,sizeof(*dev));
   struct hk_physical_device *pdev=calloc(1,sizeof(*pdev));CHECK(dev&&pdev);
   dev->dev.chip=pdev->dev.chip=AGX_CHIP_G16G;
   pdev->dev.native_mesh_qualified = true;
   dev->dev.native_mesh_qualified = true;
   pdev->disable_native_index = drirc_kill;
   dev->dev.shader_base=0x1200000000ull;
   dev->vk.alloc=*vk_default_allocator();dev->vk.physical=&pdev->vk;
   hk_device_init_usc_heap(dev);
   simple_mtx_init(&dev->apple9_variants.lock,mtx_plain);
   simple_mtx_init(&dev->apple9_bodies.lock,mtx_plain);
   simple_mtx_init(&pdev->debug_compile_lock,mtx_plain);
   for (unsigned t=16;t<=32;t*=2) {
      for (unsigned v=3;v<=t;++v) {
         for (unsigned p=1;p<=4*t/3;++p)
            mesh_case(dev,t,v,p,0);
      }
   }
   for (unsigned t=1;t<=128;t*=2)
      if (t!=16&&t!=32)mesh_case(dev,t,3,1,0);
   /* Every catalog class with a rejected interface/shape variation. */
#define HK_PACKED_MESH_SHAPE(t,v,p) for(unsigned r=(lower_common ? 6 : 1);r<=(lower_common ? 6 : 5);++r)mesh_case(dev,t,v,p,r);
#include "../hk_apple9_packed_mesh_shapes.def"
#undef HK_PACKED_MESH_SHAPE
   for (unsigned v=3;v<=5;++v) {
      struct vk_shader_compile_info infos[3]={0};
      const mesa_shader_stage stages[]={MESA_SHADER_VERTEX,MESA_SHADER_GEOMETRY,MESA_SHADER_FRAGMENT};
      for(unsigned i=0;i<3;++i)infos[i]=(struct vk_shader_compile_info){.stage=stages[i],.nir=gs_source(stages[i],v),.robustness=&vk_robustness_disabled};
      char name[80];snprintf(name,sizeof(name),"GS%u",v);record(dev,name,3,infos);
   }
   const mesa_shader_stage tess_stages[]={MESA_SHADER_VERTEX,MESA_SHADER_TESS_CTRL,MESA_SHADER_TESS_EVAL,MESA_SHADER_FRAGMENT};
   for(unsigned v=1;v<=4;++v) {
      struct vk_shader_compile_info infos[4]={0};
      for(unsigned i=0;i<4;++i)infos[i]=(struct vk_shader_compile_info){.stage=tess_stages[i],.nir=tess_source(tess_stages[i],v),.robustness=&vk_robustness_disabled};
      char name[80];snprintf(name,sizeof(name),"TES%u",v);record(dev,name,4,infos);
   }
   struct vk_shader_compile_info ordinary[]={
      {.stage=MESA_SHADER_VERTEX,.nir=gs_source(MESA_SHADER_VERTEX,3),.robustness=&vk_robustness_disabled},
      {.stage=MESA_SHADER_FRAGMENT,.nir=gs_source(MESA_SHADER_FRAGMENT,3),.robustness=&vk_robustness_disabled}};
   record(dev,"VS-FS",2,ordinary);
   simple_mtx_destroy(&pdev->debug_compile_lock);
   simple_mtx_destroy(&dev->apple9_bodies.lock);
   simple_mtx_destroy(&dev->apple9_variants.lock);
   test_reset_usc_heap(dev);hk_device_finish_usc_heap(dev);
   free(dev);free(pdev);glsl_type_singleton_decref();fclose(ledger);
   printf("cases=%u shader_objects=%u native_objects=%u GPU_execution=0\n",cases,objects,natives);
   return 0;
}
