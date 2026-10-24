/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Candidate eligibility, normalized participation, and whole-program compile.
 */
#include <stdio.h>
#include <stdlib.h>
#include "agx_apple9_mesh.h"
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%d: %s\n",__LINE__,#x); exit(1); } } while (0)

static nir_shader *source(unsigned v, unsigned p, unsigned threads)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "packed index publication proposal");
   b.shader->info.workgroup_size[0] = threads;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = v;
   b.shader->info.mesh.max_primitives_out = p;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b,v),nir_imm_int(&b,p),nir_imm_int(&b,0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b,nir_ult_imm(&b,lane,v));
   nir_store_per_vertex_output(&b,nir_imm_vec4(&b,0,0,0,1),lane,nir_imm_int(&b,0),
      .write_mask=15,.src_type=nir_type_float32,
      .io_semantics={.location=VARYING_SLOT_POS,.num_slots=1});
   nir_pop_if(&b,NULL);
   /* All indices are authored, including when P exceeds the physical lanes. */
   for (unsigned i=0;i<p;++i) {
      nir_push_if(&b,nir_ieq_imm(&b,lane,i%threads));
      nir_store_per_primitive_output(&b,nir_imm_ivec3(&b,0,1,2),nir_imm_int(&b,i),nir_imm_int(&b,0),
         .write_mask=7,.src_type=nir_type_uint32,
         .io_semantics={.location=VARYING_SLOT_PRIMITIVE_INDICES,.num_slots=1});
      nir_pop_if(&b,NULL);
   }
   return b.shader;
}

static void participation(void)
{
   for (unsigned k=1;k<=3;++k) for (unsigned p=0;p<=128;++p) {
      nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,&agx_nir_options,"prefix count");
      nir_def *n=agx_nir_apple9_index_publication_lanes(&b,nir_imm_int(&b,p),k);
      nir_store_global(&b,n,nir_imm_int64(&b,0x10000),.align_mul=4);
      nir_opt_constant_folding(b.shader);
      nir_intrinsic_instr *store=NULL;
      nir_foreach_block(block,b.impl) nir_foreach_instr(ins,block)
         if (ins->type==nir_instr_type_intrinsic && nir_instr_as_intrinsic(ins)->intrinsic==nir_intrinsic_store_global)
            store=nir_instr_as_intrinsic(ins);
      CHECK(store && nir_src_is_const(store->src[0]));
      unsigned words=0; for(unsigned i=0;i<k*p;i+=4) ++words;
      CHECK(nir_src_as_uint(store->src[0])==(words?words:1));
      ralloc_free(b.shader);
   }
}

int main(void)
{
   glsl_type_singleton_init_or_ref();participation();
   unsigned accepted=0,rejected=0;
   const unsigned cases[][4]={{3,1,16,1},{3,2,16,1},{16,16,16,1},{16,21,16,1},
      /* Beyond one 16/32-entry round: destination-addressed rounds. */
      {32,32,32,1},{32,42,32,1},{16,22,16,1},{32,43,32,1},{17,2,16,1},{64,1,64,0}};
   for(unsigned i=0;i<ARRAY_SIZE(cases);++i) {
      unsigned v=cases[i][0],p=cases[i][1],t=cases[i][2];
      const struct agx_apple9_native_mesh_options opts={.physical_threads=t,.packed_index_publication=true};
      const char *why=NULL; struct agx_shader_part code={0}; struct agx_apple9_mesh_body_info info;
      nir_shader *nir=source(v,p,t);
      bool ok=agx_compile_apple9_native_mesh(nir,true,&opts,&code,&info,&why);
      if(ok!=cases[i][3])fprintf(stderr,"V%u P%u T%u: %s\n",v,p,t,why?why:"compiled");
      CHECK(ok==cases[i][3]); if(ok)++accepted;else ++rejected;
      free(code.binary);ralloc_free(nir);
      if(v!=3 || p!=1) {
         nir=source(v,p,t);struct agx_apple9_native_mesh_options off={.physical_threads=t};
         CHECK(!agx_nir_lower_apple9_native_mesh(nir,true,&off,&info,&why));
         ralloc_free(nir);
      }
      nir=source(v,p,t);
      CHECK(!agx_nir_lower_apple9_native_mesh(nir,false,&opts,&info,&why));
      ralloc_free(nir);
   }
   glsl_type_singleton_decref();
   printf("387 prefix counts; %u whole programs; %u declaration rejections; default gates retained\n",accepted,rejected);
   return 0;
}
