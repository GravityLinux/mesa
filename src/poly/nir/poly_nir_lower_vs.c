/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "poly/cl/libpoly.h"
#include "poly/nir/poly_nir.h"
#include "poly/geometry.h"
#include "nir.h"

/*
 * This file implements basic input assembly in software. It runs on software
 * vertex shaders, as part of geometry/tessellation lowering. It does not apply
 * the topology, which happens in the geometry shader.
 */
nir_def *
poly_nir_load_vertex_id(nir_builder *b, nir_def *id)
{
   /* If drawing with an index buffer, pull the vertex ID. Otherwise, the
    * vertex ID is just the index as-is.
    */
   nir_def *index_size = nir_load_index_size_poly(b);
   nir_def *index_buffer_id;
   nir_if *index_size_present = nir_push_if(b, nir_ine_imm(b, index_size, 0));
   {
      nir_def *p = nir_load_vertex_param_buffer_poly(b);
      index_buffer_id = poly_load_index_buffer(b, p, id, index_size);
   }
   nir_pop_if(b, index_size_present);
   nir_def *effective_id = nir_if_phi(b, index_buffer_id, id);

   /* Add the "start", either an index bias or a base vertex. This must happen
    * after indexing for proper index bias behaviour.
    */
   return nir_iadd(b, effective_id, nir_load_first_vertex(b));
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_def *vertex_id = data;

   if (intr->intrinsic == nir_intrinsic_load_vertex_id) {
      nir_def_replace(&intr->def, vertex_id);
      return true;
   } else if (intr->intrinsic == nir_intrinsic_load_instance_id) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def,
                      nir_channel(b, nir_load_global_invocation_id(b, 32), 1));
      return true;
   }

   return false;
}

static bool
reads_vertex_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   return intr->intrinsic == nir_intrinsic_load_vertex_id;
}

bool
poly_nir_lower_sw_vs(nir_shader *s)
{
   /* Vertex fetch reads the vertex ID once per attribute. Pull it from the
    * index buffer once, at the top, rather than once per read: each pull is
    * control flow, so later passes cannot merge the copies.
    */
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_def *vertex_id = NULL;
   if (nir_function_intrinsics_pass(impl, reads_vertex_id, nir_metadata_all,
                                    NULL)) {
      nir_builder b = nir_builder_at(nir_before_impl(impl));
      nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      vertex_id = poly_nir_load_vertex_id(&b, id);
   }

   return nir_function_intrinsics_pass(impl, lower, nir_metadata_none,
                                       vertex_id) ||
          vertex_id != NULL;
}
