/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * Mesh shaders run as compute kernels, writing one output record per
 * workgroup (see struct hk_mesh_info). A generated hardware vertex shader then
 * rasterizes the records: every primitive slot of every workgroup is drawn
 * as an unindexed primitive, and slots past the workgroup's primitive count
 * or culled by it are placed outside the clip volume.
 */

#include "compiler/nir/nir_builder.h"
#include "util/bitscan.h"
#include "agx_nir.h"
#include "asahi/libagx/mesh.h"
#include "hk_apple9_polygon_points.h"
#include "hk_cmd_buffer.h"
#include "hk_shader.h"
#include "nir_intrinsics.h"

static bool
hk_mesh_is_primitive_slot(const nir_shader *nir, unsigned slot)
{
   return (nir->info.per_primitive_outputs & BITFIELD64_BIT(slot)) ||
          slot == VARYING_SLOT_PRIMITIVE_INDICES ||
          slot == VARYING_SLOT_CULL_PRIMITIVE;
}

/* Records hold the slots the shader stores. Declared but unwritten outputs,
 * such as a viewport index, read as zero after rasterization.
 */
void
hk_mesh_gather_info(const nir_shader *nir, struct hk_mesh_info *info)
{
   memset(info, 0, sizeof(*info));
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_per_vertex_output &&
                intr->intrinsic != nir_intrinsic_store_per_primitive_output)
               continue;
            nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
            if (sem.location == VARYING_SLOT_PRIMITIVE_COUNT)
               continue;
            uint64_t slots = BITFIELD64_RANGE(sem.location, sem.num_slots);
            if (hk_mesh_is_primitive_slot(nir, sem.location))
               info->primitive_slots |= slots;
            else
               info->vertex_slots |= slots;
         }
      }
   }
   info->primitive_slots |= BITFIELD64_BIT(VARYING_SLOT_PRIMITIVE_INDICES);
   info->max_vertices = nir->info.mesh.max_vertices_out;
   info->max_primitives = nir->info.mesh.max_primitives_out;
   info->prim = nir->info.mesh.primitive_type;
   info->group_stride_B =
      hk_mesh_primitives_offset_B(info) +
      info->max_primitives *
         hk_mesh_slot_offset_B(info, info->primitive_slots, 64);
}

/* The draw's struct libagx_mesh_params */
static nir_def *
hk_mesh_params(nir_builder *b)
{
   unsigned offset = hk_root_descriptor_offset(draw.mesh_params);
   return nir_load_ubo(b, 1, 64, nir_imm_int(b, 0), nir_imm_int(b, offset),
                       .align_mul = 8, .range_base = offset, .range = 8);
}

#define hk_mesh_param(b, params, comps, bits, field)                          \
   nir_load_global(b, comps, bits,                                             \
                   nir_iadd_imm(b, params,                                     \
                                offsetof(struct libagx_mesh_params, field)),   \
                   .align_mul = bits / 8)

/* Add to an active 64-bit query counter without requiring shader atomic int64.
 * Each low-word wrap contributes one carry. Queries are observed only after
 * the pre-raster work completes, so the two atomics need not be one transaction.
 */
static void
hk_mesh_count(nir_builder *b, nir_def *params, unsigned offset, nir_def *value)
{
   nir_def *counter = nir_load_global(b, 1, 64, nir_iadd_imm(b, params, offset),
                                      .align_mul = 8);
   nir_push_if(b, nir_ine_imm(b, counter, 0));
   nir_def *old = nir_global_atomic(b, 32, counter, value,
                                  .atomic_op = nir_atomic_op_iadd);
   nir_push_if(b, nir_ult(b, nir_iadd(b, old, value), old));
   nir_global_atomic(b, 32, nir_iadd_imm(b, counter, 4), nir_imm_int(b, 1),
                     .atomic_op = nir_atomic_op_iadd);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static unsigned
hk_workgroup_invocations(const nir_shader *nir)
{
   return nir->info.workgroup_size[0] * nir->info.workgroup_size[1] *
          nir->info.workgroup_size[2];
}

static nir_def *
hk_linear_workgroup(nir_builder *b, nir_def *id, nir_def *count)
{
   return nir_iadd(
      b, nir_channel(b, id, 0),
      nir_imul(b, nir_channel(b, count, 0),
               nir_iadd(b, nir_channel(b, id, 1),
                        nir_imul(b, nir_channel(b, count, 1),
                                 nir_channel(b, id, 2)))));
}

/* Byte address of a vec4 slot of a vertex or primitive in a record. */
static nir_def *
hk_mesh_slot(nir_builder *b, const struct hk_mesh_info *info, nir_def *group,
             bool primitive, nir_def *index, unsigned location, nir_def *offset)
{
   uint64_t mask = primitive ? info->primitive_slots : info->vertex_slots;
   unsigned stride_B = hk_mesh_slot_offset_B(info, mask, 64);
   unsigned base_B = primitive ? hk_mesh_primitives_offset_B(info) : 16;

   nir_def *byte = nir_iadd_imm(b, nir_imul_imm(b, index, stride_B),
                                base_B + hk_mesh_slot_offset_B(info, mask, location));
   if (offset)
      byte = nir_iadd(b, byte, nir_imul_imm(b, offset, 16));
   return nir_iadd(b, group, nir_u2u64(b, byte));
}

struct hk_lower_mesh_state {
   const struct hk_mesh_info *info;
   nir_def *params;
   nir_def *group;

   /* API workgroup, and the native values it is derived from */
   nir_def *workgroup_id, *num_workgroups;
   nir_def *native_id, *native_count;

   /* Task payload, and the task workgroup's mesh grid record */
   nir_def *payload, *task_grid;
};

/* Task payload accesses address the task workgroup's payload record. */
static bool
hk_lower_task_payload(nir_builder *b, nir_intrinsic_instr *intr,
                      struct hk_lower_mesh_state *state)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_barrier: {
      /* Payloads and mesh outputs are in global memory. */
      nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
      nir_variable_mode records = nir_var_mem_task_payload | nir_var_shader_out;
      if (!(modes & records))
         return false;
      nir_intrinsic_set_memory_modes(intr,
                                     (modes & ~records) | nir_var_mem_global);
      return true;
   }
   case nir_intrinsic_load_task_payload: {
      nir_def *addr = nir_iadd(
         b, state->payload,
         nir_u2u64(b, nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr))));
      nir_def_replace(&intr->def,
                      nir_load_global(b, intr->def.num_components,
                                      intr->def.bit_size, addr,
                                      .align_mul = nir_intrinsic_align_mul(intr),
                                      .align_offset = nir_intrinsic_align_offset(intr)));
      return true;
   }
   case nir_intrinsic_store_task_payload: {
      nir_def *addr = nir_iadd(
         b, state->payload,
         nir_u2u64(b, nir_iadd_imm(b, intr->src[1].ssa, nir_intrinsic_base(intr))));
      nir_store_global(b, intr->src[0].ssa, addr,
                       .write_mask = nir_intrinsic_write_mask(intr),
                       .align_mul = nir_intrinsic_align_mul(intr),
                       .align_offset = nir_intrinsic_align_offset(intr));
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_task_payload_atomic:
   case nir_intrinsic_task_payload_atomic_swap: {
      bool swap = intr->intrinsic == nir_intrinsic_task_payload_atomic_swap;
      nir_def *addr = nir_iadd(
         b, state->payload,
         nir_u2u64(b, nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr))));
      nir_def *value =
         swap ? nir_global_atomic_swap(b, intr->def.bit_size, addr,
                                       intr->src[1].ssa, intr->src[2].ssa,
                                       .atomic_op = nir_intrinsic_atomic_op(intr))
              : nir_global_atomic(b, intr->def.bit_size, addr, intr->src[1].ssa,
                                  .atomic_op = nir_intrinsic_atomic_op(intr));
      nir_def_replace(&intr->def, value);
      return true;
   }
   default:
      return false;
   }
}

/* Workgroup IDs and counts take their API values. */
static bool
hk_lower_mesh_workgroup(nir_builder *b, nir_intrinsic_instr *intr,
                        struct hk_lower_mesh_state *state)
{
   if (intr->intrinsic == nir_intrinsic_load_workgroup_id &&
       &intr->def != state->native_id) {
      nir_def_replace(&intr->def,
                      nir_u2uN(b, state->workgroup_id, intr->def.bit_size));
      return true;
   } else if (intr->intrinsic == nir_intrinsic_load_num_workgroups &&
              &intr->def != state->native_count) {
      nir_def_replace(&intr->def,
                      nir_u2uN(b, state->num_workgroups, intr->def.bit_size));
      return true;
   }
   return false;
}

static nir_def *
hk_mesh_to_word(nir_builder *b, nir_def *value, nir_alu_type type)
{
   if (value->bit_size == 1)
      return nir_b2i32(b, value);
   if (value->bit_size == 32)
      return value;
   return nir_alu_type_get_base_type(type) == nir_type_float
             ? nir_f2f32(b, value)
             : nir_alu_type_get_base_type(type) == nir_type_int
                  ? nir_i2i32(b, value)
                  : nir_u2u32(b, value);
}

static nir_def *
hk_mesh_from_word(nir_builder *b, nir_def *value, unsigned bit_size,
                  nir_alu_type type)
{
   if (bit_size == 1)
      return nir_ine_imm(b, value, 0);
   if (bit_size == 32)
      return value;
   return nir_alu_type_get_base_type(type) == nir_type_float
             ? nir_f2fN(b, value, bit_size)
             : nir_u2uN(b, value, bit_size);
}

static bool
hk_lower_mesh_intrinsic(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct hk_lower_mesh_state *state = data;
   b->cursor = nir_before_instr(&intr->instr);

   if (hk_lower_task_payload(b, intr, state) ||
       hk_lower_mesh_workgroup(b, intr, state))
      return true;

   switch (intr->intrinsic) {
   case nir_intrinsic_set_vertex_and_primitive_count: {
      /* The first invocation's counts are used. */
      nir_def *counts = nir_vec2(b, nir_u2u32(b, intr->src[0].ssa),
                                 nir_u2u32(b, intr->src[1].ssa));
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      nir_store_global(b, counts, state->group, .align_mul = 16,
                       .write_mask = 0x3);
      hk_mesh_count(b, state->params,
                    offsetof(struct libagx_mesh_params, mesh_primitives),
                    nir_channel(b, counts, 1));
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }

   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output: {
      nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
      if (sem.location == VARYING_SLOT_PRIMITIVE_COUNT) {
         nir_instr_remove(&intr->instr);
         return true;
      }
      bool primitive = hk_mesh_is_primitive_slot(b->shader, sem.location);
      nir_def *addr = hk_mesh_slot(b, state->info, state->group, primitive,
                                   intr->src[1].ssa, sem.location,
                                   intr->src[2].ssa);
      nir_def *value = intr->src[0].ssa;
      nir_alu_type type = nir_intrinsic_src_type(intr);
      unsigned component = nir_intrinsic_component(intr);
      u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
         nir_def *word = hk_mesh_to_word(b, nir_channel(b, value, c), type);
         nir_store_global(b, word,
                          nir_iadd_imm(b, addr, (component + c) * 4),
                          .align_mul = 4);
      }
      nir_instr_remove(&intr->instr);
      return true;
   }

   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_load_per_primitive_output: {
      nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
      bool primitive = hk_mesh_is_primitive_slot(b->shader, sem.location);
      nir_def *addr = hk_mesh_slot(b, state->info, state->group, primitive,
                                   intr->src[0].ssa, sem.location,
                                   intr->src[1].ssa);
      addr = nir_iadd_imm(b, addr, nir_intrinsic_component(intr) * 4);
      nir_def *value = nir_load_global(b, intr->def.num_components, 32, addr,
                                       .align_mul = 4);
      nir_def_replace(&intr->def,
                      hk_mesh_from_word(b, value, intr->def.bit_size,
                                        nir_intrinsic_dest_type(intr)));
      return true;
   }

   default:
      return false;
   }
}

/* Find the task workgroup which launched mesh workgroup g: the first whose
 * inclusive prefix sum of launched workgroups exceeds g.
 */
static nir_def *
hk_find_task(nir_builder *b, nir_def *grids, nir_def *tasks, nir_def *g)
{
   nir_variable *lo = nir_local_variable_create(b->impl, glsl_uint_type(), "lo");
   nir_variable *hi = nir_local_variable_create(b->impl, glsl_uint_type(), "hi");
   nir_store_var(b, lo, nir_imm_int(b, 0), 1);
   nir_store_var(b, hi, tasks, 1);

   nir_push_loop(b);
   {
      nir_def *l = nir_load_var(b, lo), *h = nir_load_var(b, hi);
      nir_break_if(b, nir_uge(b, l, h));

      nir_def *mid = nir_ushr_imm(b, nir_iadd(b, l, h), 1);
      nir_def *sum = nir_load_global(
         b, 1, 32,
         nir_iadd(b, grids, nir_u2u64(b, nir_iadd_imm(b, nir_imul_imm(b, mid, 16), 12))),
         .align_mul = 4);
      nir_def *above = nir_ult(b, g, sum);
      nir_store_var(b, hi, nir_bcsel(b, above, mid, h), 1);
      nir_store_var(b, lo, nir_bcsel(b, above, l, nir_iadd_imm(b, mid, 1)), 1);
   }
   nir_pop_loop(b, NULL);
   return nir_load_var(b, lo);
}

/* Lower a mesh shader to the compute kernel writing its records. Without a
 * task shader, the dispatch is the API grid. Otherwise, it is a linear grid of
 * all mesh workgroups launched by the task workgroups.
 */
void
hk_nir_lower_mesh_main(nir_shader *nir, const struct hk_mesh_info *info)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *params = hk_mesh_params(&b);
   nir_def *tasks = hk_mesh_param(&b, params, 1, 32, task_count);
   struct hk_lower_mesh_state state = {
      .info = info,
      .params = params,
      .native_id = nir_load_workgroup_id(&b),
      .native_count = nir_load_num_workgroups(&b),
   };

   nir_def *g_direct = hk_linear_workgroup(&b, state.native_id,
                                           state.native_count);
   nir_def *g_task = nir_channel(&b, state.native_id, 0);

   nir_push_if(&b, nir_ine_imm(&b, tasks, 0));
   nir_def *grids = hk_mesh_param(&b, params, 1, 64, task_grids);
   nir_def *t = hk_find_task(&b, grids, tasks, g_task);
   nir_def *grid = nir_load_global(
      &b, 4, 32, nir_iadd(&b, grids, nir_u2u64(&b, nir_imul_imm(&b, t, 16))),
      .align_mul = 16);
   nir_def *mx = nir_channel(&b, grid, 0), *my = nir_channel(&b, grid, 1);
   nir_def *mxy = nir_imul(&b, mx, my);
   nir_def *base = nir_isub(&b, nir_channel(&b, grid, 3),
                            nir_imul(&b, mxy, nir_channel(&b, grid, 2)));
   nir_def *local = nir_isub(&b, g_task, base);
   nir_def *task_id = nir_vec3(&b, nir_umod(&b, local, mx),
                               nir_umod(&b, nir_udiv(&b, local, mx), my),
                               nir_udiv(&b, local, mxy));
   nir_def *stride = hk_mesh_param(&b, params, 1, 32, payload_stride_B);
   nir_def *payload = nir_iadd(
      &b, hk_mesh_param(&b, params, 1, 64, payloads),
      nir_imul(&b, nir_u2u64(&b, t), nir_u2u64(&b, stride)));
   nir_def *task_count = nir_trim_vector(&b, grid, 3);
   nir_push_else(&b, NULL);
   nir_def *no_payload = nir_imm_int64(&b, 0);
   nir_pop_if(&b, NULL);

   state.workgroup_id = nir_if_phi(&b, task_id, state.native_id);
   state.num_workgroups = nir_if_phi(&b, task_count, state.native_count);
   state.payload = nir_if_phi(&b, payload, no_payload);
   nir_def *g = nir_if_phi(&b, g_task, g_direct);
   state.group = nir_iadd(&b, hk_mesh_param(&b, params, 1, 64, outputs),
                          nir_imul_imm(&b, nir_u2u64(&b, g),
                                       info->group_stride_B));

   /* Without SetMeshOutputsEXT, a workgroup emits nothing. Zero the counts
    * before any invocation can set them.
    */
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_store_global(&b, nir_imm_zero(&b, 2, 32), state.group, .align_mul = 16,
                    .write_mask = 0x3);
   hk_mesh_count(&b, params, offsetof(struct libagx_mesh_params, mesh_invocations),
                 nir_imm_int(&b, hk_workgroup_invocations(nir)));
   nir_pop_if(&b, NULL);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   nir_progress(true, impl, nir_metadata_none);

   nir_shader_intrinsics_pass(nir, hk_lower_mesh_intrinsic, nir_metadata_none,
                              &state);
}

static bool
hk_lower_task_intrinsic(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct hk_lower_mesh_state *state = data;
   b->cursor = nir_before_instr(&intr->instr);

   if (hk_lower_task_payload(b, intr, state))
      return true;
   if (intr->intrinsic != nir_intrinsic_launch_mesh_workgroups)
      return false;

   /* The first invocation's grid is launched. */
   nir_def *dims = intr->src[0].ssa;
   nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
   nir_def *product = nir_imul(b, nir_imul(b, nir_channel(b, dims, 0),
                                           nir_channel(b, dims, 1)),
                               nir_channel(b, dims, 2));
   nir_store_global(b, nir_vec4(b, nir_channel(b, dims, 0),
                                nir_channel(b, dims, 1),
                                nir_channel(b, dims, 2), product),
                    state->task_grid, .align_mul = 16);
   nir_pop_if(b, NULL);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Lower a task shader to the compute kernel writing its payload, and the
 * grid of mesh workgroups it launches, for its workgroup of the API grid.
 */
void
hk_nir_lower_task_main(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *params = hk_mesh_params(&b);
   nir_def *t = hk_linear_workgroup(&b, nir_load_workgroup_id(&b),
                                    nir_load_num_workgroups(&b));
   nir_def *stride = hk_mesh_param(&b, params, 1, 32, payload_stride_B);
   struct hk_lower_mesh_state state = {
      .payload = nir_iadd(&b, hk_mesh_param(&b, params, 1, 64, payloads),
                          nir_imul(&b, nir_u2u64(&b, t), nir_u2u64(&b, stride))),
      .task_grid = nir_iadd(&b, hk_mesh_param(&b, params, 1, 64, task_grids),
                            nir_imul_imm(&b, nir_u2u64(&b, t), 16)),
   };

   /* A workgroup which launches no mesh workgroups has an empty grid. */
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_store_global(&b, nir_imm_zero(&b, 4, 32), state.task_grid,
                    .align_mul = 16);
   hk_mesh_count(&b, params, offsetof(struct libagx_mesh_params, task_invocations),
                 nir_imm_int(&b, hk_workgroup_invocations(nir)));
   nir_pop_if(&b, NULL);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   nir_progress(true, impl, nir_metadata_none);

   nir_shader_intrinsics_pass(nir, hk_lower_task_intrinsic, nir_metadata_none,
                              &state);
}

struct hk_mesh_pp {
   const struct hk_mesh_info *info;
   nir_def *group, *p, *index[3], *visible;
};

static nir_def *
hk_mesh_pp_load(nir_builder *b, void *data, unsigned vertex, unsigned location,
                unsigned offset, unsigned component, unsigned comps,
                nir_alu_type type)
{
   struct hk_mesh_pp *pp = data;
   const struct hk_mesh_info *info = pp->info;
   bool primitive = info->primitive_slots & BITFIELD64_BIT(location);
   nir_def *value;
   if ((info->vertex_slots | info->primitive_slots) & BITFIELD64_BIT(location)) {
      value = nir_load_global(
         b, hk_mesh_slot_words(info, location), 32,
         hk_mesh_slot(b, info, pp->group, primitive,
                      primitive ? pp->p : pp->index[vertex], location,
                      offset ? nir_imm_int(b, offset) : NULL),
         .align_mul = 4);
      value = nir_pad_vector_imm_int(b, value, 0, 4);
   } else if (location == VARYING_SLOT_LAYER) {
      unsigned root = hk_root_descriptor_offset(draw.view_index);
      value = nir_pad_vector_imm_int(
         b,
         nir_load_ubo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, root),
                      .align_mul = 4, .range_base = root, .range = 4),
         0, 4);
   } else {
      value = nir_imm_vec4(b, 0.0, 0.0, 0.0, 1.0);
   }

   /* Invisible primitives lie behind the eye, and clip to nothing. */
   if (location == VARYING_SLOT_POS) {
      value = nir_bcsel(b, pp->visible, value,
                        nir_imm_vec4(b, 0.0, 0.0, 0.0, -1.0));
   }
   return nir_channels(b, value, BITFIELD_RANGE(component, comps));
}

/* Polygon mode POINT: each triangle draws HK_PP_MAX_VERTICES points, one per
 * vertex of the clipped polygon, the others behind the eye.
 */
static nir_shader *
hk_nir_mesh_rast_pp(const nir_shader *mesh, const struct hk_mesh_info *info,
                    const struct hk_pp_info *pp_info, bool point_size)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "%s rast points",
      mesh->info.name ? mesh->info.name : "mesh");
   nir_shader *s = b.shader;

   unsigned per_group = MAX2(info->max_primitives, 1) * HK_PP_MAX_VERTICES;
   nir_def *vid = nir_load_vertex_id(&b);
   nir_def *g = nir_udiv_imm(&b, vid, per_group);
   nir_def *rem = nir_umod_imm(&b, vid, per_group);
   nir_def *p = nir_udiv_imm(&b, rem, HK_PP_MAX_VERTICES);
   nir_def *point = nir_umod_imm(&b, rem, HK_PP_MAX_VERTICES);

   struct hk_mesh_pp pp = {.info = info, .p = p};
   nir_def *records = hk_mesh_param(&b, hk_mesh_params(&b), 1, 64, outputs);
   pp.group = nir_iadd(&b, records,
                       nir_imul_imm(&b, nir_u2u64(&b, g), info->group_stride_B));
   nir_def *prim_count =
      nir_load_global(&b, 1, 32, nir_iadd_imm(&b, pp.group, 4), .align_mul = 4);
   pp.visible = nir_ult(&b, p, prim_count);
   if (info->primitive_slots & VARYING_BIT_CULL_PRIMITIVE) {
      nir_def *cull = nir_load_global(
         &b, 1, 32,
         hk_mesh_slot(&b, info, pp.group, true, p, VARYING_SLOT_CULL_PRIMITIVE,
                      NULL),
         .align_mul = 4);
      pp.visible = nir_iand(&b, pp.visible, nir_ieq_imm(&b, cull, 0));
   }
   for (unsigned i = 0; i < 3; ++i) {
      nir_def *index = nir_load_global(
         &b, 1, 32,
         nir_iadd_imm(&b,
                      hk_mesh_slot(&b, info, pp.group, true, p,
                                   VARYING_SLOT_PRIMITIVE_INDICES, NULL),
                      4 * i),
         .align_mul = 4);
      pp.index[i] = nir_umin_imm(&b, index, MAX2(info->max_vertices, 1) - 1);
   }

   uint64_t outputs = (info->vertex_slots | info->primitive_slots) &
                      ~(VARYING_BIT_PRIMITIVE_INDICES |
                        VARYING_BIT_CULL_PRIMITIVE | VARYING_BIT_CLIP_DIST0 |
                        VARYING_BIT_CLIP_DIST1 | VARYING_BIT_CULL_DIST0 |
                        VARYING_BIT_CULL_DIST1);
   outputs |= VARYING_BIT_POS | VARYING_BIT_LAYER;
   uint64_t memory = 0;
   u_foreach_bit64(slot, info->primitive_slots) {
      if (hk_mesh_varying_in_memory(slot))
         memory |= BITFIELD64_BIT(slot);
   }
   outputs &= ~memory;
   /* Without polygonModePointSize, polygon points have size 1. */
   if (!point_size)
      outputs &= ~VARYING_BIT_PSIZ;

   unsigned components[VARYING_SLOT_MAX] = {0};
   u_foreach_bit64(slot, outputs)
      components[slot] = hk_mesh_slot_words(info, slot);

   struct hk_pp_info pp_out = *pp_info;
   pp_out.outputs = outputs;
   pp_out.output_components = components;
   /* Per-primitive values are equal at the three corners: keep them exact. */
   pp_out.flat |= info->primitive_slots;

   nir_def *last = nir_ieq_imm(
      &b,
      nir_load_ubo(&b, 1, 16, nir_imm_int(&b, 0),
                   nir_imm_int(&b, hk_root_descriptor_offset(draw.provoking)),
                   .align_mul = 2,
                   .range_base = hk_root_descriptor_offset(draw.provoking),
                   .range = 2),
      2);
   nir_def *provoking = nir_bcsel(&b, last, nir_imm_int(&b, 2), nir_imm_int(&b, 0));
   nir_def *prim_id =
      (info->primitive_slots & VARYING_BIT_PRIMITIVE_ID)
         ? hk_mesh_pp_load(&b, &pp, 0, VARYING_SLOT_PRIMITIVE_ID, 0, 0, 1,
                           nir_type_uint32)
         : nir_imm_int(&b, 0);
   hk_apple9_pp_emit_triangle(&b, &pp_out, hk_mesh_pp_load, &pp, provoking,
                              prim_id, point);

   if (memory) {
      nir_def *offset = nir_iadd_imm(
         &b,
         nir_iadd(&b, nir_imul_imm(&b, g, info->group_stride_B),
                  nir_imul_imm(&b, p,
                               hk_mesh_slot_offset_B(info, info->primitive_slots,
                                                     64))),
         hk_mesh_primitives_offset_B(info));
      nir_store_output(&b, offset, nir_imm_int(&b, 0), .write_mask = 0x1,
                       .src_type = nir_type_uint32,
                       .io_semantics.location = HK_MESH_PRIMITIVE_SLOT,
                       .io_semantics.num_slots = 1);
      outputs |= BITFIELD64_BIT(HK_MESH_PRIMITIVE_SLOT);
   }

   unsigned clip = mesh->info.clip_distance_array_size;
   unsigned cull = mesh->info.cull_distance_array_size;
   if (clip + cull)
      outputs |= VARYING_BIT_CLIP_DIST0 | (clip + cull > 4 ? VARYING_BIT_CLIP_DIST1 : 0);
   if (pp_out.face_slot)
      outputs |= BITFIELD64_BIT(pp_out.face_slot);
   s->info.outputs_written = outputs;
   s->info.clip_distance_array_size = clip;
   s->info.cull_distance_array_size = cull;
   s->info.io_lowered = true;
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   return s;
}

/* The hardware vertex shader rasterizing mesh records. */
nir_shader *
hk_nir_mesh_rast(const nir_shader *mesh, const struct hk_mesh_info *info,
                 const struct hk_per_vertex_copies *copies,
                 const struct hk_pp_info *polygon_points, bool point_size)
{
   if (polygon_points)
      return hk_nir_mesh_rast_pp(mesh, info, polygon_points, point_size);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "%s rast",
      mesh->info.name ? mesh->info.name : "mesh");
   nir_shader *s = b.shader;

   unsigned vpp = mesa_vertices_per_prim(info->prim);
   unsigned per_group = MAX2(info->max_primitives, 1) * vpp;
   nir_def *vid = nir_load_vertex_id(&b);
   nir_def *g = nir_udiv_imm(&b, vid, per_group);
   nir_def *rem = nir_umod_imm(&b, vid, per_group);
   nir_def *p = nir_udiv_imm(&b, rem, vpp);
   nir_def *corner = nir_umod_imm(&b, rem, vpp);

   nir_def *records = hk_mesh_param(&b, hk_mesh_params(&b), 1, 64, outputs);
   nir_def *group = nir_iadd(&b, records,
                             nir_imul_imm(&b, nir_u2u64(&b, g),
                                          info->group_stride_B));
   nir_def *prim_count =
      nir_load_global(&b, 1, 32, nir_iadd_imm(&b, group, 4), .align_mul = 4);
   nir_def *visible = nir_ult(&b, p, prim_count);
   if (info->primitive_slots & VARYING_BIT_CULL_PRIMITIVE) {
      nir_def *cull = nir_load_global(
         &b, 1, 32,
         hk_mesh_slot(&b, info, group, true, p, VARYING_SLOT_CULL_PRIMITIVE,
                      NULL),
         .align_mul = 4);
      visible = nir_iand(&b, visible, nir_ieq_imm(&b, cull, 0));
   }

   nir_def *index = nir_load_global(
      &b, 1, 32,
      nir_iadd(&b,
               hk_mesh_slot(&b, info, group, true, p,
                            VARYING_SLOT_PRIMITIVE_INDICES, NULL),
               nir_u2u64(&b, nir_imul_imm(&b, corner, 4))),
      .align_mul = 4);
   index = nir_umin_imm(&b, index, MAX2(info->max_vertices, 1) - 1);

   uint64_t outputs = (info->vertex_slots | info->primitive_slots) &
                      ~(VARYING_BIT_PRIMITIVE_INDICES |
                        VARYING_BIT_CULL_PRIMITIVE);
   outputs |= VARYING_BIT_POS | VARYING_BIT_LAYER;

   /* User per-primitive varyings are read from memory by the primitive's
    * offset in the records.
    */
   uint64_t memory = 0;
   u_foreach_bit64(slot, info->primitive_slots) {
      if (hk_mesh_varying_in_memory(slot))
         memory |= BITFIELD64_BIT(slot);
   }
   outputs &= ~memory;
   u_foreach_bit64(slot, outputs) {
      bool primitive = info->primitive_slots & BITFIELD64_BIT(slot);
      nir_def *value;
      if ((info->vertex_slots | info->primitive_slots) & BITFIELD64_BIT(slot)) {
         value = nir_load_global(
            &b, hk_mesh_slot_words(info, slot), 32,
            hk_mesh_slot(&b, info, group, primitive, primitive ? p : index,
                         slot, NULL),
            .align_mul = 4);
         value = nir_pad_vector_imm_int(&b, value, 0, 4);
      } else if (slot == VARYING_SLOT_LAYER) {
         /* Multiview draws each view to its layer. Mesh shaders do not write
          * the layer then, and otherwise the view index is zero.
          */
         unsigned offset = hk_root_descriptor_offset(draw.view_index);
         value = nir_pad_vector_imm_int(
            &b,
            nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, offset),
                         .align_mul = 4, .range_base = offset, .range = 4),
            0, 4);
      } else {
         value = nir_imm_vec4(&b, 0.0, 0.0, 0.0, 1.0);
      }

      /* Invisible primitives lie behind the eye. */
      if (slot == VARYING_SLOT_POS) {
         value = nir_bcsel(&b, visible, value,
                           nir_imm_vec4(&b, 0.0, 0.0, 0.0, -1.0));
      }

      bool scalar = slot == VARYING_SLOT_PSIZ || slot == VARYING_SLOT_LAYER ||
                    slot == VARYING_SLOT_VIEWPORT ||
                    slot == VARYING_SLOT_PRIMITIVE_ID ||
                    slot == VARYING_SLOT_PRIMITIVE_SHADING_RATE;
      bool integer = slot == VARYING_SLOT_LAYER ||
                     slot == VARYING_SLOT_VIEWPORT ||
                     slot == VARYING_SLOT_PRIMITIVE_ID ||
                     slot == VARYING_SLOT_PRIMITIVE_SHADING_RATE;
      if (scalar)
         value = nir_channel(&b, value, 0);

      nir_store_output(&b, value, nir_imm_int(&b, 0),
                       .write_mask = nir_component_mask(value->num_components),
                       .src_type = integer ? nir_type_int32 : nir_type_float32,
                       .io_semantics.location = slot,
                       .io_semantics.num_slots = 1);
   }

   /* Every vertex of the primitive carries flat copies of the per-vertex
    * fragment inputs of its corners, in API vertex order. Points and lines
    * repeat their last vertex.
    */
   for (unsigned j = 0; copies && j < copies->count; ++j) {
      unsigned slot = VARYING_SLOT_VAR0 + copies->location[j];
      for (unsigned k = 0; k < 3; ++k) {
         nir_def *value = nir_imm_zero(&b, 4, 32);
         if (info->vertex_slots & BITFIELD64_BIT(slot)) {
            nir_def *corner_index = nir_load_global(
               &b, 1, 32,
               nir_iadd_imm(&b,
                            hk_mesh_slot(&b, info, group, true, p,
                                         VARYING_SLOT_PRIMITIVE_INDICES, NULL),
                            4 * MIN2(k, vpp - 1)),
               .align_mul = 4);
            corner_index = nir_umin_imm(&b, corner_index,
                                        MAX2(info->max_vertices, 1) - 1);
            value = nir_load_global(
               &b, hk_mesh_slot_words(info, slot), 32,
               hk_mesh_slot(&b, info, group, false, corner_index, slot, NULL),
               .align_mul = 4);
            value = nir_pad_vector_imm_int(&b, value, 0, 4);
         }

         unsigned copy = VARYING_SLOT_VAR0 + copies->copy_base[j] + k;
         nir_store_output(&b, value, nir_imm_int(&b, 0), .write_mask = 0xf,
                          .src_type = nir_type_float32,
                          .io_semantics.location = copy,
                          .io_semantics.num_slots = 1);
         outputs |= BITFIELD64_BIT(copy);
      }
   }

   /* Barycentrics interpolate one-hot weights of the API vertices. */
   for (unsigned m = 0; copies && m < 2; ++m) {
      if (!copies->bary[m])
         continue;
      unsigned slot = VARYING_SLOT_VAR0 + copies->bary[m] - 1;
      nir_def *weights = nir_vec3(&b, nir_b2f32(&b, nir_ieq_imm(&b, corner, 0)),
                                  nir_b2f32(&b, nir_ieq_imm(&b, corner, 1)),
                                  nir_b2f32(&b, nir_ieq_imm(&b, corner, 2)));
      nir_store_output(&b, weights, nir_imm_int(&b, 0), .write_mask = 0x7,
                       .src_type = nir_type_float32,
                       .io_semantics.location = slot,
                       .io_semantics.num_slots = 1);
      outputs |= BITFIELD64_BIT(slot);
   }

   if (memory) {
      nir_def *offset = nir_iadd_imm(
         &b,
         nir_iadd(&b, nir_imul_imm(&b, g, info->group_stride_B),
                  nir_imul_imm(&b, p,
                               hk_mesh_slot_offset_B(info, info->primitive_slots,
                                                     64))),
         hk_mesh_primitives_offset_B(info));
      nir_store_output(&b, offset, nir_imm_int(&b, 0), .write_mask = 0x1,
                       .src_type = nir_type_uint32,
                       .io_semantics.location = HK_MESH_PRIMITIVE_SLOT,
                       .io_semantics.num_slots = 1);
      outputs |= BITFIELD64_BIT(HK_MESH_PRIMITIVE_SLOT);
   }

   s->info.outputs_written = outputs;
   s->info.clip_distance_array_size = mesh->info.clip_distance_array_size;
   s->info.cull_distance_array_size = mesh->info.cull_distance_array_size;
   s->info.io_lowered = true;
   return s;
}
