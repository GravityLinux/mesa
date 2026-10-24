/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_mesh.h"
#include "compiler/nir/nir_builder.h"

static bool
fail(const char **reason, const char *message)
{
   if (reason)
      *reason = message;
   return false;
}

bool
agx_apple9_native_mesh_shape_eligible(const nir_shader *nir)
{
   return nir->info.stage == MESA_SHADER_MESH && !nir->info.task_payload_size &&
          nir->info.mesh.max_vertices_out == 3 &&
          nir->info.mesh.max_primitives_out == 1 &&
          nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES;
}

static bool
eligible(nir_shader *nir, unsigned physical_threads, bool packed_indices,
         const char **reason)
{
   if (packed_indices) {
      /* This compiler option does not widen the driver's native shape
       * predicate. Publication repeats in rounds of at most
       * AGX_APPLE9_PUBLICATION_ROUND entries, so the API maxima are covered,
       * subject to the local output allocation. */
      if (nir->info.stage != MESA_SHADER_MESH || nir->info.task_payload_size ||
          nir->info.mesh.primitive_type != MESA_PRIM_TRIANGLES ||
          !nir->info.mesh.max_vertices_out ||
          nir->info.mesh.max_vertices_out > 256 ||
          !nir->info.mesh.max_primitives_out ||
          nir->info.mesh.max_primitives_out > 256)
         return fail(reason, "packed index candidate requires V1..256 P1..256 triangles");
   } else if (!agx_apple9_native_mesh_shape_eligible(nir))
      return fail(reason,
                  "native mesh development class requires V3 P1 triangles without payload");
   if (physical_threads == AGX_APPLE9_NATIVE_THREADS) {
      if (!agx_apple9_native_workgroup_supported(nir, reason))
         return false;
   } else {
      if (physical_threads != AGX_APPLE9_NATIVE_SERIAL_THREADS)
         return fail(reason, "native mesh physical workgroup must be T16 or T32");
      if (nir->info.workgroup_size_variable)
         return fail(reason, "native mesh requires a fixed API workgroup size");
      if (nir->info.api_subgroup_size && nir->info.api_subgroup_size != 32)
         return fail(reason, "native mesh requires architectural subgroup size 32");
      unsigned threads = 1;
      for (unsigned d = 0; d < 3; ++d) {
         unsigned size = nir->info.workgroup_size[d];
         if (!size || size > AGX_APPLE9_NATIVE_MAX_INVOCATIONS / threads)
            return fail(reason, "native mesh supports 1..128 API invocations at physical T32");
         threads *= size;
      }
   }

   /* SetMeshOutputs operands need not be constants, and mutually exclusive
    * branches may each contain a call. The body lowerer checks constant
    * counts against the declaration and bounds runtime counts before
    * publishing lane zero's normalized values.
    * Vulkan requires at most one executed call under uniform control flow. */
   return true;
}

/* Keep the actual vertex count in compiler-owned local storage. The body
 * lowerer leaves the count marker so completion can consume both operands. */
static bool
consume_count(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_mesh_body_info *info = data;
   if (intr->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      nir_store_local_apple9_agx(b, intr->src[0].ssa,
                                 nir_imm_int(b, info->layout.vertex_count_B),
                                 .base = info->key.output_root, .bit_size = 32);
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }
   /* Point size does not participate in triangle rasterization. */
   if (intr->intrinsic == nir_intrinsic_store_per_vertex_output &&
       nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PSIZ) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   return false;
}

nir_def *
agx_nir_apple9_index_publication_lanes(nir_builder *b, nir_def *primitives,
                                       unsigned k)
{
   assert(k >= 1 && k <= 3);
   nir_def *bytes = nir_imul_imm(b, primitives, k);
   /* Counts are normalized before publication; one lane retains the empty
    * handoff used by the existing native path. */
   return nir_umax_imm(b, nir_ushr_imm(b, nir_iadd_imm(b, bytes, 3), 2), 1);
}

static void
device_fence(nir_builder *b)
{
   nir_barrier(b, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

/* With one primitive, every referenced vertex has the same primitive value.
 * Keep body reads/writes in vertex zero's local plane, then broadcast only after
 * all body writes complete. This is an ordinary flat-varying lowering, separate
 * from the unresolved native primitive-record linking plane.
 */
struct primitive_lower {
   uint64_t mask;
   unsigned root, cull_B;
};

static bool
lower_primitive_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct primitive_lower *lower = data;
   uint64_t mask = lower->mask;
   bool store = intr->intrinsic == nir_intrinsic_store_per_primitive_output;
   if (!store && intr->intrinsic != nir_intrinsic_load_per_primitive_output)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned slot = sem.location + nir_src_as_uint(intr->src[store ? 2 : 1]);
   if (slot >= 64 || !(mask & BITFIELD64_BIT(slot)))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   if (slot == VARYING_SLOT_CULL_PRIMITIVE) {
      /* A compiler-owned word in the count header, separate from both native
       * primitive records and the fragment compiler's CullDistance transport. */
      nir_def *address = nir_imm_int(b, lower->cull_B);
      if (store) {
         nir_def *value = intr->src[0].ssa;
         if (value->bit_size == 1)
            value = nir_b2i32(b, value);
         nir_store_local_apple9_agx(b, value, address,
                                    .base = lower->root, .bit_size = 32);
         nir_instr_remove(&intr->instr);
      } else {
         nir_def_replace(&intr->def,
            nir_load_local_apple9_agx(b, address, .base = lower->root));
      }
      return true;
   }
   sem.location = slot;
   sem.num_slots = 1;
   if (store) {
      nir_def *value = intr->src[0].ssa;
      nir_store_per_vertex_output(
         b, value, nir_imm_int(b, 0), nir_imm_int(b, 0),
         .base = nir_intrinsic_base(intr), .range = 1,
         .component = nir_intrinsic_component(intr),
         .write_mask = nir_intrinsic_write_mask(intr),
         .src_type = nir_intrinsic_src_type(intr),
         .io_semantics = sem);
      nir_instr_remove(&intr->instr);
   } else {
      nir_def *value = nir_load_per_vertex_output(
         b, intr->def.num_components, 32, nir_imm_int(b, 0), nir_imm_int(b, 0),
         .base = nir_intrinsic_base(intr), .range = 1,
         .component = nir_intrinsic_component(intr),
         .dest_type = nir_intrinsic_dest_type(intr), .io_semantics = sem);
      nir_def_replace(&intr->def, value);
   }
   return true;
}

static bool
primitive_outputs(nir_shader *nir, const struct agx_apple9_mesh_key *key,
                  uint64_t *mask, const char **reason)
{
   if (nir->info.mesh.max_primitives_out != 1 ||
       nir->info.mesh.max_vertices_out > 3)
      return fail(reason, "primitive broadcast requires P1 and at most V3");

   uint64_t vertex = 0;
   *mask = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            bool store = intr->intrinsic == nir_intrinsic_store_per_primitive_output ||
                         intr->intrinsic == nir_intrinsic_store_per_vertex_output;
            bool prim = intr->intrinsic == nir_intrinsic_store_per_primitive_output ||
                        intr->intrinsic == nir_intrinsic_load_per_primitive_output;
            if (!store && !prim && intr->intrinsic != nir_intrinsic_load_per_vertex_output)
               continue;
            unsigned offset = store ? 2 : 1;
            if (!nir_src_is_const(intr->src[offset]))
               return fail(reason, "primitive broadcast requires constant IO slots");
            uint64_t slot = nir_intrinsic_io_semantics(intr).location +
                            (uint64_t)nir_src_as_uint(intr->src[offset]);
            if (slot >= 64)
               return fail(reason, "primitive broadcast IO slot is unsupported");
            if (!prim) {
               vertex |= BITFIELD64_BIT(slot);
               continue;
            }
            if (slot == VARYING_SLOT_PRIMITIVE_INDICES)
               continue;
            bool user = slot >= VARYING_SLOT_VAR0 && slot <= VARYING_SLOT_VAR31;
            bool cull = slot == VARYING_SLOT_CULL_PRIMITIVE;
            if (!user && !cull && slot != VARYING_SLOT_PRIMITIVE_ID)
               return fail(reason, "primitive broadcast builtin is unsupported");
            unsigned bits = store ? nir_src_bit_size(intr->src[0]) : intr->def.bit_size;
            if (bits != 32 && !(cull && store && bits == 1))
               return fail(reason, "primitive broadcast requires 32-bit components");
            if (!user && (intr->num_components != 1 || nir_intrinsic_component(intr)))
               return fail(reason, "primitive broadcast builtin must be scalar");
            *mask |= BITFIELD64_BIT(slot);
         }
      }
   }
   if (*mask & vertex)
      return fail(reason, "primitive and vertex outputs cannot share a broadcast slot");
   struct primitive_lower lower = {
      .mask = *mask, .root = key->output_root,
      .cull_B = ALIGN_POT(nir->info.mesh.max_primitives_out * 3, 16) + 8,
   };
   nir_shader_intrinsics_pass(nir, lower_primitive_output, nir_metadata_none, &lower);
   nir->info.per_primitive_outputs &= ~*mask;
   return true;
}

static void
broadcast_primitive(nir_builder *b, const struct agx_apple9_mesh_body_info *info,
                    nir_def *lane, nir_def *vertices)
{
   const struct agx_apple9_mesh_layout *l = &info->layout;
   unsigned root = info->key.output_root;
   nir_push_if(b, nir_ult(b, lane, vertices));
   u_foreach_bit64(slot, info->primitive_broadcast) {
      for (unsigned c = 0; c < 4; ++c) {
         unsigned component = l->vertex_component[slot][c];
         if (component == AGX_APPLE9_MESH_UNUSED)
            continue;
         unsigned plane = l->position_B + l->vertex_pitch_B * component;
         nir_def *value = nir_load_local_apple9_agx(b, nir_imm_int(b, plane), .base = root);
         nir_store_local_apple9_agx(
            b, value, nir_iadd_imm(b, nir_imul_imm(b, lane, 4), plane),
            .base = root, .bit_size = 32);
      }
   }
   nir_pop_if(b, NULL);
   if (info->primitive_broadcast & VARYING_BIT_CULL_PRIMITIVE) {
      nir_push_if(b, nir_ieq_imm(b, lane, 0));
      nir_def *address = nir_imm_int(b, l->primitive_count_B + 8);
      nir_def *cull = nir_load_local_apple9_agx(
         b, address, .base = root);
      nir_push_if(b, nir_ine_imm(b, cull, 0));
      for (unsigned c = 0; c < 3; ++c)
         nir_store_local_apple9_agx(b, nir_imm_int(b, 0), nir_imm_int(b, c),
                                    .base = root, .bit_size = 8);
      nir_pop_if(b, NULL);
      /* Restore the count header before publication: the hardware's use of
       * this word is unknown, so the compiler-owned flag must not escape. */
      nir_store_local_apple9_agx(b, nir_imm_int(b, 0), address,
                                 .base = root, .bit_size = 32);
      nir_pop_if(b, NULL);
   }
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
}

/* Local planes stay in semantic/component order for shader output reads.
 * U channels use the fragment linker's interpolation groups instead. */
static bool
varying_layout(const struct agx_apple9_mesh_body_info *info,
               const struct agx_apple9_native_mesh_options *options,
               struct agx_apple9_varying_layout *varyings,
               const char **reason)
{
   const struct agx_apple9_mesh_layout *layout = &info->layout;
   memset(varyings, 0, sizeof(*varyings));
   if (layout->primitive_components || layout->culling)
      return fail(reason,
                  "native mesh primitive attributes/culling are not implemented");
   if (layout->vertex_components > 4 && (!options || !options->attributes))
      return fail(reason, "native mesh attributes development path is disabled");
   if (layout->vertex_components > 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS)
      return fail(reason, "native mesh exceeds 124 user varying scalars");
   for (unsigned slot = 0; slot < VARYING_SLOT_MAX; ++slot) {
      /* Clip distances are rasterizer outputs only, published once after the
       * user varyings (Air 2026-10-24: a linked copy as well makes the
       * vertex stride disagree with the PPP output count). */
      if (slot == VARYING_SLOT_POS || slot == VARYING_SLOT_CLIP_DIST0 ||
          slot == VARYING_SLOT_CLIP_DIST1)
         continue;
      for (unsigned c = 0; c < 4; ++c) {
         if (layout->vertex_component[slot][c] == AGX_APPLE9_MESH_UNUSED)
            continue;
         if (slot >= ARRAY_SIZE(varyings->mask) ||
             !agx_apple9_varying_supported(slot))
            return fail(reason, "native mesh varying location is unsupported");
         varyings->mask[slot] |= BITFIELD_BIT(c);
         ++varyings->count;
      }
      if (slot < ARRAY_SIZE(varyings->mask) && varyings->mask[slot])
         varyings->group[slot] = agx_apple9_varying_group(
            options->outputs_flat | info->primitive_broadcast,
            options->outputs_linear, slot);
   }
   return true;
}

bool
agx_nir_lower_apple9_native_mesh(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_native_mesh_options *options,
   struct agx_apple9_mesh_body_info *info, const char **reason)
{
   if (!enable_native_mesh)
      return fail(reason, "native mesh development path is disabled");
   unsigned physical_threads = options && options->physical_threads
                                  ? options->physical_threads
                                  : AGX_APPLE9_NATIVE_THREADS;
   bool packed_indices = options && options->packed_index_publication;
   if (!eligible(nir, physical_threads, packed_indices, reason))
      return false;

   if (packed_indices && options->primitive_broadcast)
      return fail(reason, "packed index candidate has no primitive broadcast");
   bool rounds = options && options->publication_rounds;
   if (rounds && (!packed_indices || physical_threads != AGX_APPLE9_NATIVE_SERIAL_THREADS))
      return fail(reason, "multi-round publication requires packed indices at T32");

   nir_remove_dead_variables(nir, nir_var_mem_shared, NULL);
   nir_lower_vars_to_explicit_types(nir, nir_var_mem_shared,
                                    glsl_get_natural_size_align_bytes);
   nir_lower_explicit_io(nir, nir_var_mem_shared, nir_address_format_32bit_offset);
   nir_remove_dead_variables(nir, nir_var_mem_shared, NULL);

   uint16_t logical_size[3];
   memcpy(logical_size, nir->info.workgroup_size, sizeof(logical_size));
   unsigned logical_threads = logical_size[0] * logical_size[1] * logical_size[2];
   if (logical_threads > physical_threads) {
      /* Preserve the application's subgroup width and execute its subgroups
       * in order between workgroup barriers. The common pass supplies masks
       * for the final partial subgroup and keeps logical IDs and values live
       * across barriers. Completion is appended only after this transform. */
      nir_lower_system_values(nir);
      nir_lower_halt_to_return(nir);
      nir_lower_returns(nir);
      nir_lower_workgroup_size(nir, physical_threads);
      bool progress;
      do {
         progress = nir_lower_vars_to_ssa(nir);
         progress |= nir_opt_copy_prop(nir);
         progress |= nir_opt_constant_folding(nir);
         progress |= nir_opt_dead_cf(nir);
         progress |= nir_opt_remove_phis(nir);
         progress |= nir_opt_dce(nir);
      } while (progress);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   }

   struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   uint64_t primitive_broadcast = 0;
   if (options && options->attributes && options->primitive_broadcast &&
       !primitive_outputs(nir, &key, &primitive_broadcast, reason))
      return false;

   key.physical_threads = physical_threads;
   if (!agx_nir_lower_apple9_mesh(nir, &key, info, reason))
      return false;
   memcpy(info->logical_size, logical_size, sizeof(logical_size));
   info->primitive_broadcast = primitive_broadcast;
   info->publication_rounds = rounds;
   const struct agx_apple9_mesh_layout *layout = &info->layout;
   struct agx_apple9_varying_layout varyings;
   if (!varying_layout(info, options, &varyings, reason))
      return false;
   /* Every declared clip distance needs its plane; cull distances keep the
    * emulated path until native primitive culling covers them. */
   info->clip_distances = nir->info.clip_distance_array_size;
   if (nir->info.cull_distance_array_size || info->clip_distances > 8)
      return fail(reason, "native mesh supports up to eight clip and no cull distances");
   for (unsigned d = 0; d < info->clip_distances; ++d) {
      if (layout->vertex_component[VARYING_SLOT_CLIP_DIST0 + d / 4][d % 4] ==
          AGX_APPLE9_MESH_UNUSED)
         return fail(reason, "native mesh clip distance is declared but not written");
   }
   if (4 + varyings.count + info->clip_distances > 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS)
      return fail(reason, "native mesh varyings and clip distances exceed the UVS");
   nir_shader_intrinsics_pass(nir, consume_count, nir_metadata_none, info);
   nir_lower_returns(nir);
   if (!agx_nir_mask_apple9_mesh_body(nir, &key, layout, reason))
      return false;
   /* A uniform return can bypass SetMeshOutputs entirely. Such a workgroup
    * emits nothing. Initialize both count words before the application mask;
    * only lane zero writes them, and completion reads after its workgroup
    * barrier. No initialization of undefined output attributes is needed. */
   nir_builder entry =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
   nir_push_if(&entry, nir_ieq_imm(&entry,
                                 nir_load_local_invocation_index(&entry), 0));
   nir_store_local_apple9_agx(
      &entry, nir_imm_int(&entry, 0),
      nir_imm_int(&entry, layout->primitive_count_B),
      .base = key.output_root, .bit_size = key.primitive_count_bits);
   nir_store_local_apple9_agx(
      &entry, nir_imm_int(&entry, 0),
      nir_imm_int(&entry, layout->vertex_count_B),
      .base = key.output_root, .bit_size = 32);
   nir_pop_if(&entry, NULL);
   if (primitive_broadcast & VARYING_BIT_CULL_PRIMITIVE) {
      nir_push_if(&entry, nir_ieq_imm(&entry,
                                    nir_load_local_invocation_index(&entry), 0));
      nir_store_local_apple9_agx(
         &entry, nir_imm_int(&entry, 0),
         nir_imm_int(&entry, layout->primitive_count_B + 8),
         .base = key.output_root, .bit_size = 32);
      nir_pop_if(&entry, NULL);
      /* All physical lanes rendezvous before any application cull store. */
      nir_barrier(&entry, .execution_scope = SCOPE_WORKGROUP,
                         .memory_scope = SCOPE_WORKGROUP,
                         .memory_semantics = NIR_MEMORY_ACQ_REL,
                         .memory_modes = nir_var_mem_shared);
   }
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   /* Counts are the
    * actual shader publication, not maxima or register-descriptor constants.
    * U0 consumes and clears its A halfword; B's wider meaning is unresolved.
    * The qualified one-triangle class supplies actual vertex count to B. */
   nir_def *primitives = nir_load_local_apple9_agx(
      &b, nir_imm_int(&b, layout->primitive_count_B), .base = key.output_root);
   nir_def *vertices = nir_load_local_apple9_agx(
      &b, nir_imm_int(&b, layout->vertex_count_B), .base = key.output_root);
   nir_def *lane = nir_load_local_invocation_index(&b);
   if (primitive_broadcast)
      broadcast_primitive(&b, info, lane, vertices);
   if (options && options->clip_halfz) {
      /* Convert the completed homogeneous position, after all body writes and
       * output reads. Doing this at publication also handles scalar/partial
       * stores and different invocations writing Z and W. Keep W unchanged:
       * U consumes it for clipping and perspective division.
       * This is the vertex path's c=1/2 depth conversion, not a state-word
       * change. The body barrier above makes its final Z/W visible. */
      for (unsigned first = 0; first < layout->max_vertices;
           first += physical_threads) {
      nir_def *vertex = nir_iadd_imm(&b, lane, first);
      nir_push_if(&b, nir_ult(&b, vertex, vertices));
      nir_def *offset = nir_imul_imm(&b, vertex, 4);
      nir_def *z_address = nir_iadd_imm(
         &b, offset, layout->position_B + layout->vertex_pitch_B *
                        layout->vertex_component[VARYING_SLOT_POS][2]);
      nir_def *w_address = nir_iadd_imm(
         &b, offset, layout->position_B + layout->vertex_pitch_B *
                        layout->vertex_component[VARYING_SLOT_POS][3]);
      nir_def *z = nir_load_local_apple9_agx(&b, z_address,
                                            .base = key.output_root);
      nir_def *w = nir_load_local_apple9_agx(&b, w_address,
                                            .base = key.output_root);
      nir_def *half = nir_imm_float(&b, 0.5);
      nir_def *converted = nir_ffma_weak(
         &b, nir_fneg(&b, z), half, nir_ffma_weak(&b, w, half, z));
      nir_store_local_apple9_agx(&b, converted, z_address,
                                  .base = key.output_root, .bit_size = 32);
      nir_pop_if(&b, NULL);
      }
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
   }
   /* Publish the completed output memory before issuing mesh allocation.
    * U0 consumes the counts but does not write these global output planes.
    * Keeping this fence before U0 lets the first channel submit consume U0's
    * separate completion dependency through its measured input-wait field. */
   device_fence(&b);
   nir_mesh_control_apple9_agx(&b, nir_u2u16(&b, primitives),
                               nir_u2u16(&b, vertices), .base = 1);

   /* Each channel/index collective publishes at most one round of entries.
    * Its lanes form a contiguous prefix supplying one common source byte
    * offset (second operand, S2's uniform-base invariant); the destination
    * entry is bits 16 and up of the first operand: the channel selector for
    * vertex channels, the vertex-count operand for packed index words. Later
    * rounds advance both. Air 2026-10-24: a repeat without the destination
    * aliases entry 0; with it V64/P126 at T32 is exact. Single-round classes
    * keep destination 0 and emit the same code. Rounds beyond the runtime
    * counts are skipped, never issued empty. */
   const unsigned round = MIN2(physical_threads, AGX_APPLE9_PUBLICATION_ROUND);
   const unsigned clip_distances = info->clip_distances;
   for (unsigned first = 0; first < layout->max_vertices; first += round) {
      nir_def *active = nir_ult(&b, nir_iadd_imm(&b, lane, first), vertices);
      if (round != physical_threads)
         active = nir_iand(&b, active, nir_ult_imm(&b, lane, round));
      nir_push_if(&b, active);
      const uint32_t destination = first << 16;
      for (unsigned c = 0; c < 4; ++c) {
         unsigned plane =
            layout->position_B +
            layout->vertex_pitch_B * layout->vertex_component[VARYING_SLOT_POS][c];
         nir_mesh_submit_channel_apple9_agx(&b, nir_imm_int(&b, c | destination),
                                            nir_imm_int(&b, plane + 4 * first));
         device_fence(&b);
      }
      for (unsigned slot = 0; slot < ARRAY_SIZE(varyings.mask); ++slot) {
         u_foreach_bit(c, varyings.mask[slot]) {
            unsigned channel = 4 + agx_apple9_varying_index(&varyings, slot, c);
            unsigned plane = layout->position_B + layout->vertex_pitch_B *
                                layout->vertex_component[slot][c];
            nir_mesh_submit_channel_apple9_agx(
               &b, nir_imm_int(&b, channel | destination),
               nir_imm_int(&b, plane + 4 * first));
            device_fence(&b);
         }
      }
      /* Rasterizer clip distances follow the user varyings, as in the
       * vertex path's UVS order (mesh writes no point size or layer word).
       * They are published once, so the mesh state's plane count equals the
       * PPP output count 4 + varyings + clip distances. */
      for (unsigned d = 0; d < clip_distances; ++d) {
         unsigned slot = VARYING_SLOT_CLIP_DIST0 + d / 4;
         unsigned plane = layout->position_B + layout->vertex_pitch_B *
                             layout->vertex_component[slot][d % 4];
         nir_mesh_submit_channel_apple9_agx(
            &b, nir_imm_int(&b, (4 + varyings.count + d) | destination),
            nir_imm_int(&b, plane + 4 * first));
         device_fence(&b);
      }
      nir_pop_if(&b, NULL);
   }
   /* Preserve the qualified first operand (actual vertex count) in the low
    * bits. Index words start at byte 0 of the output root. */
   if (packed_indices) {
      nir_def *words = agx_nir_apple9_index_publication_lanes(
         &b, primitives, layout->indices_per_primitive);
      unsigned max_words =
         MAX2(DIV_ROUND_UP(layout->max_primitives * layout->indices_per_primitive, 4), 1);
      for (unsigned first = 0; first < max_words; first += round) {
         nir_def *active = nir_ult(&b, nir_iadd_imm(&b, lane, first), words);
         if (round != physical_threads)
            active = nir_iand(&b, active, nir_ult_imm(&b, lane, round));
         nir_push_if(&b, active);
         nir_def *count = first ? nir_ior_imm(&b, vertices, first << 16) : vertices;
         nir_mesh_submit_scalar_apple9_agx(&b, count, nir_imm_int(&b, 4 * first));
         device_fence(&b);
         nir_pop_if(&b, NULL);
      }
   } else {
      nir_push_if(&b, nir_ieq_imm(&b, lane, 0));
      nir_mesh_submit_scalar_apple9_agx(&b, vertices, nir_imm_int(&b, 0));
      device_fence(&b);
      nir_pop_if(&b, NULL);
   }

   info->requires_mesh_completion = false;
   nir_progress(true, nir_shader_get_entrypoint(nir), nir_metadata_none);
   nir_validate_shader(nir, "native Apple9 mesh completion");
   return true;
}

bool
agx_compile_apple9_native_mesh(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_native_mesh_options *options,
   struct agx_shader_part *out, struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   memset(out, 0, sizeof(*out));
   if (!agx_nir_lower_apple9_native_mesh(nir, enable_native_mesh, options,
                                         info, reason))
      return false;
   if (!agx_compile_apple9_mesh_lowered(nir, &info->key, out, reason))
      return false;
   out->info.apple9_clip_distance_count = info->clip_distances;
   return varying_layout(info, options, &out->info.apple9_varyings,
                         reason);
}
