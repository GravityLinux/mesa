/* Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include "hk_shader.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "failed: %s\n", #c); return false; } } while (0)

/* Test normal task/mesh lowering, including empty output and nontrivial local
 * sizes. Counting is per invocation, not per emitted vertex or workgroup.
 */
static bool
check_stage(bool task, unsigned primitives)
{
   nir_builder b = nir_builder_init_simple_shader(
      task ? MESA_SHADER_TASK : MESA_SHADER_MESH, &agx_nir_options, "mesh queries");
   b.shader->info.workgroup_size[0] = 2;
   b.shader->info.workgroup_size[1] = 3;
   b.shader->info.workgroup_size[2] = 4;
   if (task) {
      /* Even a task group emitting no mesh groups must count its invocations. */
      nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 0, 0, 0));
      hk_nir_lower_task_main(b.shader);
   } else {
      b.shader->info.mesh.max_vertices_out = 3;
      b.shader->info.mesh.max_primitives_out = 7;
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                         nir_imm_int(&b, primitives),
                                         nir_imm_int(&b, 0));
      struct hk_mesh_info info;
      hk_mesh_gather_info(b.shader, &info);
      hk_nir_lower_mesh_main(b.shader, &info);
   }
   nir_opt_constant_folding(b.shader);
   nir_validate_shader(b.shader, "mesh query lowering");

   unsigned atomics = 0, invocations = 0, primitive_counts = 0, carries = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_global_atomic)
               continue;
            CHECK(intr->def.bit_size == 32);
            CHECK(nir_intrinsic_atomic_op(intr) == nir_atomic_op_iadd);
            /* Every counter update is guarded, never an unconditional write
             * through the inactive query's null address. */
            CHECK(block->cf_node.parent->type == nir_cf_node_if);
            CHECK(nir_src_is_const(intr->src[1]));
            unsigned add = nir_src_as_uint(intr->src[1]);
            atomics++;
            invocations += add == 24;
            primitive_counts += !task && add == primitives;
            carries += add == 1;
         }
      }
   }
   CHECK(atomics == (task ? 2 : 4));
   CHECK(invocations == 1);
   CHECK(primitive_counts == (task ? 0 : 1));
   CHECK(carries == (task ? 1 : 2));
   ralloc_free(b.shader);
   return true;
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = check_stage(true, 0) && check_stage(false, 7) && check_stage(false, 0);
   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}
