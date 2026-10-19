/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <string.h>
#include "compiler/nir/nir_builder.h"
#include "util/bitscan.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "agx_apple9_mesh.h"
#include "agx_nir.h"

static bool
fail(const char **reason, const char *text)
{
   if (reason)
      *reason = text;
   return false;
}

static bool
is_workgroup_barrier(nir_instr *instr)
{
   return instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(nir_instr_as_intrinsic(instr)) ==
             SCOPE_WORKGROUP;
}

bool
agx_apple9_native_workgroup_supported(const nir_shader *nir, const char **reason)
{
   if (nir->info.workgroup_size_variable)
      return fail(reason, "native mesh/object requires a fixed API workgroup size");
   if (nir->info.api_subgroup_size && nir->info.api_subgroup_size != 32)
      return fail(reason, "native mesh/object requires architectural subgroup size 32");
   unsigned threads = 1;
   for (unsigned d = 0; d < 3; ++d) {
      unsigned size = nir->info.workgroup_size[d];
      if (!size || size > AGX_APPLE9_NATIVE_THREADS / threads)
         return fail(reason,
                     "native mesh/object supports 1..16 API invocations at "
                     "physical T16; wider launches are unqualified");
      threads *= size;
   }
   return true;
}

static bool
is_output(nir_intrinsic_op op)
{
   return op == nir_intrinsic_store_per_vertex_output ||
          op == nir_intrinsic_store_per_primitive_output ||
          op == nir_intrinsic_load_per_vertex_output ||
          op == nir_intrinsic_load_per_primitive_output;
}

static bool
is_store(nir_intrinsic_op op)
{
   return op == nir_intrinsic_store_per_vertex_output ||
          op == nir_intrinsic_store_per_primitive_output;
}

static bool
is_primitive(nir_intrinsic_op op)
{
   return op == nir_intrinsic_store_per_primitive_output ||
          op == nir_intrinsic_load_per_primitive_output;
}

/* Resolve constant IO-slot offsets before building the linked component map.
 * A dynamic vertex/primitive index is independent of an IO-slot offset. */
static unsigned
location(const nir_intrinsic_instr *intr)
{
   unsigned offset_src = is_store(intr->intrinsic) ? 2 : 1;
   return nir_intrinsic_io_semantics(intr).location +
          nir_src_as_uint(intr->src[offset_src]);
}

static bool
validate(nir_shader *nir, const struct agx_apple9_mesh_key *key,
         struct agx_apple9_mesh_layout *layout, const char **reason)
{
   if (nir->info.stage != MESA_SHADER_MESH &&
       nir->info.stage != MESA_SHADER_TASK)
      return fail(reason, "Apple9 native IO requires mesh or task NIR");
   uint64_t logical_threads = 1;
   for (unsigned d = 0; d < 3; ++d)
      logical_threads *= nir->info.workgroup_size[d];
   bool expanded = key && key->physical_threads > logical_threads;
   if (key && key->physical_threads &&
       (nir->info.workgroup_size_variable || !logical_threads ||
        key->physical_threads < logical_threads || key->physical_threads > 256))
      return fail(reason, "invalid Apple9 mesh physical workgroup width");
   if (key && key->physical_threads && nir->info.stage == MESA_SHADER_TASK &&
       (key->physical_threads != AGX_APPLE9_NATIVE_THREADS ||
        !agx_apple9_native_workgroup_supported(nir, reason)))
      return fail(reason, "native object expansion requires physical T16 and 1..16 API invocations");
   if (expanded && nir->info.derivative_group != DERIVATIVE_GROUP_NONE)
      return fail(reason,
                  "expanded Apple9 mesh derivative groups are unsupported");
   /* No native allocator currently separates user shared memory from the
    * local grid/output allocation. Reject both explicit IO and declarations
    * before lowering; a frontend may not have assigned shared_size yet. */
   if (nir->info.shared_size)
      return fail(reason,
                  "Apple9 native bodies do not support user shared memory");
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_shared)
      return fail(reason,
                  "Apple9 native bodies do not support user shared memory");
   /* Local root N owns uniform words 2*N and 2*N+1. Check the entire
    * fixed graphics window, including its group-count and shared roots. */
   if (!key || key->output_root >= 64 || key->grid_root >= 64 ||
       (key->primitive_count_bits != 16 && key->primitive_count_bits != 32) ||
       2 * key->output_root < key->reserved_uniform_words ||
       2 * key->grid_root < key->reserved_uniform_words ||
       key->payload_uniform < key->reserved_uniform_words ||
       (key->payload_uniform & 1) || key->payload_uniform >= 255 ||
       key->output_root == key->grid_root ||
       key->payload_uniform / 2 == key->output_root ||
       key->payload_uniform / 2 == key->grid_root || (key->grid_offset_B & 3) ||
       key->grid_offset_B > AGX_APPLE9_MAX_SHARED_BYTES - 12)
      return fail(reason,
                  "Apple9 native stage arguments overlap or are out of range");

   memset(layout, 0, sizeof(*layout));
   memset(layout->vertex_component, 0xff, sizeof(layout->vertex_component));
   memset(layout->primitive_component, 0xff,
          sizeof(layout->primitive_component));
   if (nir->info.stage == MESA_SHADER_MESH) {
      layout->max_vertices = nir->info.mesh.max_vertices_out;
      layout->max_primitives = nir->info.mesh.max_primitives_out;
      if (!layout->max_vertices || layout->max_vertices > 256 ||
          !layout->max_primitives || layout->max_primitives > 512 ||
          nir->info.mesh.primitive_type != MESA_PRIM_TRIANGLES)
         return fail(
            reason,
            "Apple9 native body currently requires triangle output with u8 indices");
      layout->indices_per_primitive = 3;
   }

   bool payload = false, local_output = false;
   uint8_t vertex[VARYING_SLOT_MAX] = {0}, primitive[VARYING_SLOT_MAX] = {0};
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_shared ||
                intr->intrinsic == nir_intrinsic_store_shared ||
                intr->intrinsic == nir_intrinsic_shared_atomic ||
                intr->intrinsic == nir_intrinsic_shared_atomic_swap)
               return fail(
                  reason,
                  "Apple9 native bodies do not support user shared memory");
            payload |= intr->intrinsic == nir_intrinsic_load_task_payload ||
                       intr->intrinsic == nir_intrinsic_store_task_payload;
            local_output |=
               is_output(intr->intrinsic) ||
               intr->intrinsic ==
                  nir_intrinsic_set_vertex_and_primitive_count ||
               intr->intrinsic == nir_intrinsic_launch_mesh_workgroups;
            if (intr->intrinsic == nir_intrinsic_task_payload_atomic ||
                intr->intrinsic == nir_intrinsic_task_payload_atomic_swap ||
                intr->intrinsic ==
                   nir_intrinsic_launch_mesh_workgroups_with_payload_deref)
               return fail(
                  reason,
                  "Apple9 native body requires explicit payload loads/stores");

            if (intr->intrinsic ==
                   nir_intrinsic_set_vertex_and_primitive_count &&
                nir->info.stage != MESA_SHADER_MESH)
               return fail(reason,
                           "Apple9 mesh counts used outside a mesh stage");
            if (intr->intrinsic == nir_intrinsic_launch_mesh_workgroups &&
                (nir->info.stage != MESA_SHADER_TASK ||
                 nir_intrinsic_base(intr)))
               return fail(
                  reason,
                  "Apple9 task grid requires the workgroup's payload base");
            if (!is_output(intr->intrinsic))
               continue;
            if (nir->info.stage != MESA_SHADER_MESH)
               return fail(reason,
                           "Apple9 mesh output used outside a mesh stage");
            unsigned offset_src = is_store(intr->intrinsic) ? 2 : 1;
            if (!nir_src_is_const(intr->src[offset_src]))
               return fail(reason,
                           "Apple9 native IO slot offsets must be constant");
            if ((uint64_t)nir_intrinsic_io_semantics(intr).location +
                   nir_src_as_uint(intr->src[offset_src]) >=
                VARYING_SLOT_MAX)
               return fail(reason, "Apple9 native IO slot offset overflows");
            unsigned slot = location(intr),
                     comp = nir_intrinsic_component(intr);
            unsigned components = intr->num_components;
            if (slot >= VARYING_SLOT_MAX || comp + components > 4)
               return fail(reason,
                           "Apple9 native IO slot/component is out of range");
            bool prim = is_primitive(intr->intrinsic);
            if (slot == VARYING_SLOT_PRIMITIVE_INDICES ||
                slot == VARYING_SLOT_CULL_PRIMITIVE) {
               if (!prim || !is_store(intr->intrinsic))
                  return fail(
                     reason,
                     "Apple9 native index/cull output loads are not implemented");
               if ((slot == VARYING_SLOT_PRIMITIVE_INDICES &&
                    comp + components > 3) ||
                   (slot == VARYING_SLOT_CULL_PRIMITIVE &&
                    (comp || components != 1)))
                  return fail(reason,
                              "Apple9 index/cull component shape is invalid");
               layout->culling |= slot == VARYING_SLOT_CULL_PRIMITIVE;
               continue;
            }
            if (slot != VARYING_SLOT_POS && slot < VARYING_SLOT_VAR0)
               return fail(reason,
                           "Apple9 native output builtin is not implemented");
            unsigned bits = is_store(intr->intrinsic)
                               ? nir_src_bit_size(intr->src[0])
                               : intr->def.bit_size;
            if (bits != 32 || (slot == VARYING_SLOT_POS && prim))
               return fail(
                  reason,
                  "Apple9 native user/position outputs require 32-bit components");
            unsigned mask = is_store(intr->intrinsic)
                               ? nir_intrinsic_write_mask(intr)
                               : BITFIELD_MASK(components);
            (prim ? primitive : vertex)[slot] |= mask << comp;
         }
      }
   }
   unsigned local_root =
      nir->info.stage == MESA_SHADER_TASK ? key->grid_root : key->output_root;
   if (payload && local_output && key->payload_uniform / 2 == local_root)
      return fail(reason,
                  "Apple9 local root aliases the payload pointer argument");
   if (nir->info.stage == MESA_SHADER_TASK)
      return true;

   /* Output contract validated with the retained opaque mesh completion at
    * 256V/32P, 64V/512P and 64V/32P with primitive attributes (dynamic indices).
    * Culling remains provisional. Which fields fixed-function hardware reads
    * directly is unknown; revalidate when replacing that completion.
    * The index area is padded to a 16-byte count/header region. Position
    * and user values are component planes, each rounded to four vertices.
    * Culling reserves one four-byte slot per primitive before positions. */
   layout->primitive_count_B = ALIGN_POT(layout->max_primitives * 3, 16);
   layout->vertex_count_B = layout->primitive_count_B + 4;
   layout->cull_B = layout->primitive_count_B + 16;
   layout->position_B =
      layout->cull_B + (layout->culling ? layout->max_primitives * 4 : 0);
   layout->vertex_pitch_B = ALIGN_POT(layout->max_vertices, 4) * 4;
   for (unsigned c = 0; c < 4; ++c)
      layout->vertex_component[VARYING_SLOT_POS][c] =
         layout->vertex_components++;
   for (unsigned slot = VARYING_SLOT_VAR0; slot < VARYING_SLOT_MAX; ++slot) {
      u_foreach_bit(c, vertex[slot])
         layout->vertex_component[slot][c] = layout->vertex_components++;
      u_foreach_bit(c, primitive[slot])
         layout->primitive_component[slot][c] = layout->primitive_components++;
   }
   layout->primitive_stride_B = layout->primitive_components * 4;
   /* Primitive attributes have a separate device record. The local output
    * contract reserves one linking plane when such records are present. */
   layout->local_size_B =
      layout->position_B +
      layout->vertex_pitch_B *
         (layout->vertex_components + (layout->primitive_components != 0));
   if (layout->local_size_B > AGX_APPLE9_MAX_SHARED_BYTES)
      return fail(
         reason,
         "Apple9 native output layout exceeds the measured local allocation");
   return true;
}

static bool
freeze_logical_workgroup(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned threads = *(const unsigned *)data;
   if (intr->intrinsic != nir_intrinsic_load_workgroup_size &&
       intr->intrinsic != nir_intrinsic_load_num_subgroups)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = intr->intrinsic == nir_intrinsic_load_num_subgroups
                       ? nir_imm_int(b, DIV_ROUND_UP(threads, 32))
                       : nir_imm_ivec3(b, b->shader->info.workgroup_size[0],
                                       b->shader->info.workgroup_size[1],
                                       b->shader->info.workgroup_size[2]);
   nir_def_replace(&intr->def, value);
   return true;
}

/* Mask one application region, leaving the adjacent rendezvous converged.
 * Cross-region values are repaired after all regions have been extracted. */
static void
mask_region(nir_cursor start, nir_cursor end, unsigned threads)
{
   if (nir_cursors_equal(start, end))
      return;
   nir_cf_list user;
   start = nir_cf_extract(&user, start, end);
   nir_builder b = nir_builder_at(start);
   nir_push_if(&b,
               nir_ult_imm(&b, nir_load_local_invocation_index(&b), threads));
   b.cursor = nir_cf_reinsert(&user, b.cursor);
   nir_pop_if(&b, NULL);
}

/* A workgroup rendezvous that also orders local memory: the nested-barrier
 * flag below depends on its memory semantics, not only on execution. */
static void
workgroup_rendezvous(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

/* Barriers inside application control flow, with more than one simdgroup of
 * application lanes. Workgroup barriers are uniform across the application
 * lanes, so every simdgroup holding any of them executes the same number B.
 * Simdgroups are 32 consecutive invocation indices, so the side each takes
 * below is uniform within it.
 *
 * Application lanes count their barriers; afterwards invocation 0 publishes
 * B in a word that held a sentinel, then the application side closes with
 * one rendezvous (B + 1 in all). The other simdgroups count rendezvous k and
 * read the word after each: it can only change between rendezvous B and
 * B + 1, so they see it at k = B (racing the store) or at k = B + 1, and in
 * the first case take one more rendezvous. Each simdgroup decides on one
 * read (its first lane's), so its lanes cannot see the race differently. Both
 * sides then meet at a final rendezvous and the word is cleared. Nothing
 * depends on how hardware treats simdgroups that have exited. The word is the
 * unused last word of the output count header; it reads zero again before the
 * body ends. */
#define NESTED_PENDING 0xffffffffu

static void
mask_nested(nir_function_impl *impl, unsigned threads, unsigned root,
            unsigned word)
{
   nir_variable *count =
      nir_local_variable_create(impl, glsl_uint_type(), "mesh_barriers");
   /* Count every application workgroup barrier where it executes. */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (!is_workgroup_barrier(instr))
            continue;
         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_store_var(&b, count, nir_iadd_imm(&b, nir_load_var(&b, count), 1),
                       1);
      }
   }
   nir_cf_list user;
   nir_cf_extract(&user, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *index = nir_load_local_invocation_index(&b);
   nir_def *first = nir_ieq_imm(&b, index, 0);
   nir_def *offset = nir_imm_int(&b, word);
   nir_store_var(&b, count, nir_imm_int(&b, 0), 1);
   nir_push_if(&b, first);
   nir_store_local_apple9_agx(&b, nir_imm_int(&b, NESTED_PENDING), offset,
                              .base = root, .bit_size = 32);
   nir_pop_if(&b, NULL);
   workgroup_rendezvous(&b);
   nir_push_if(&b, nir_ult_imm(&b, nir_iand_imm(&b, index, ~31u), threads));
   {
      nir_push_if(&b, nir_ult_imm(&b, index, threads));
      b.cursor = nir_cf_reinsert(&user, b.cursor);
      nir_pop_if(&b, NULL);
      nir_push_if(&b, first);
      nir_store_local_apple9_agx(&b, nir_load_var(&b, count), offset,
                                 .base = root, .bit_size = 32);
      nir_pop_if(&b, NULL);
      workgroup_rendezvous(&b);
   }
   nir_push_else(&b, NULL);
   {
      nir_variable *seen =
         nir_local_variable_create(impl, glsl_uint_type(), "mesh_rendezvous");
      nir_variable *published =
         nir_local_variable_create(impl, glsl_uint_type(), "mesh_published");
      nir_store_var(&b, seen, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      {
         workgroup_rendezvous(&b);
         nir_store_var(&b, seen, nir_iadd_imm(&b, nir_load_var(&b, seen), 1),
                       1);
         /* One value per simdgroup: lanes racing the store must not split. */
         nir_def *value = nir_read_first_invocation(
            &b, nir_load_local_apple9_agx(&b, offset, .base = root));
         nir_store_var(&b, published, value, 1);
         nir_break_if(&b, nir_ine_imm(&b, value, NESTED_PENDING));
      }
      nir_pop_loop(&b, NULL);
      /* Seen at k = B: the application still owes its closing rendezvous. */
      nir_push_if(
         &b, nir_ieq(&b, nir_load_var(&b, seen), nir_load_var(&b, published)));
      workgroup_rendezvous(&b);
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   workgroup_rendezvous(&b);
   nir_push_if(&b, first);
   nir_store_local_apple9_agx(&b, nir_imm_int(&b, 0), offset, .base = root,
                              .bit_size = 32);
   nir_pop_if(&b, NULL);
}

bool
agx_nir_mask_apple9_mesh_body(nir_shader *nir,
                              const struct agx_apple9_mesh_key *key,
                              const struct agx_apple9_mesh_layout *layout,
                              const char **reason)
{
   unsigned threads = nir->info.workgroup_size[0] *
                      nir->info.workgroup_size[1] * nir->info.workgroup_size[2];
   if (!key->physical_threads ||
       (key->physical_threads == threads &&
        nir->info.workgroup_size[1] == 1 && nir->info.workgroup_size[2] == 1))
      return true;

   /* With every application lane in simdgroup 0, application barriers only
    * order lanes of that simdgroup: shrink them to subgroup scope (as RADV
    * does for small mesh workgroups), so extra simdgroups never rendezvous
    * with the application, wherever its barriers are. */
   bool nested = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (!is_workgroup_barrier(instr))
               continue;
            nir_intrinsic_instr *barrier = nir_instr_as_intrinsic(instr);
            if (threads <= 32) {
               nir_intrinsic_set_execution_scope(barrier, SCOPE_SUBGROUP);
               if (nir_intrinsic_memory_scope(barrier) == SCOPE_WORKGROUP)
                  nir_intrinsic_set_memory_scope(barrier, SCOPE_SUBGROUP);
            } else if (block->cf_node.parent != &impl->cf_node) {
               nested = true;
            }
         }
      }
   }
   if (nested && !layout)
      return fail(
         reason, "expanded native mesh nested barriers need the output layout");

   /* Flatten the physical launch without changing the application's 3D IDs,
    * WorkgroupSize, or subgroup count. Freeze newly inserted size loads too. */
   nir_lower_compute_system_values_options options = {
      .lower_cs_local_id_to_index = true,
   };
   nir_lower_compute_system_values(nir, &options);
   nir_shader_intrinsics_pass(nir, freeze_logical_workgroup,
                              nir_metadata_control_flow, &threads);
   nir_foreach_function_impl(impl, nir) {
      if (nested) {
         /* Unverified on hardware: nothing must read this header word while
          * the body runs (formation's own reads are checked on the host). */
         mask_nested(impl, threads, key->output_root,
                     layout->primitive_count_B + 12);
      } else {
         /* Converged barriers split the masked application regions.
          * CF extraction may split/merge blocks, but instruction pointers
          * survive: collect barriers first rather than iterating blocks
          * while mutating. */
         struct util_dynarray barriers;
         util_dynarray_init(&barriers, NULL);
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (is_workgroup_barrier(instr))
                  util_dynarray_append(&barriers, instr);
            }
         }
         nir_cursor start = nir_before_impl(impl);
         util_dynarray_foreach(&barriers, nir_instr *, barrier) {
            mask_region(start, nir_before_instr(*barrier), threads);
            start = nir_after_instr(*barrier);
         }
         mask_region(start, nir_after_impl(impl), threads);
         util_dynarray_fini(&barriers);
      }
      nir_progress(true, impl, nir_metadata_none);
      /* Values defined in one mask must still reach the later active region;
       * inactive lanes receive undef, which no application operation consumes. */
      nir_repair_ssa_impl(impl);
   }
   if (nested)
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   nir->info.workgroup_size[0] = key->physical_threads;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   return true;
}

struct lower {
   const struct agx_apple9_mesh_key *key;
   struct agx_apple9_mesh_body_info *info;
};

static nir_def *
payload_pointer(nir_builder *b, const struct lower *s, nir_def *offset)
{
   nir_def *base =
      nir_load_native_pointer_apple9_agx(b, .base = s->key->payload_uniform);
   return nir_iadd(b, nir_pack_64_2x32(b, base), nir_u2u64(b, offset));
}

static void
store_local(nir_builder *b, unsigned root, nir_def *value, nir_def *offset,
            unsigned bits)
{
   nir_store_local_apple9_agx(b, nir_u2uN(b, value, bits), nir_u2u32(b, offset),
                              .base = root, .bit_size = bits);
}

static bool
lower_intrinsic(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower *s = data;
   const struct agx_apple9_mesh_layout *l = &s->info->layout;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_task_payload: {
      nir_def *off =
         nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr));
      nir_def *v = nir_load_global(
         b, intr->def.num_components, intr->def.bit_size,
         payload_pointer(b, s, off), .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr),
         .access = nir_intrinsic_access(intr));
      nir_def_replace(&intr->def, v);
      return true;
   }
   case nir_intrinsic_store_task_payload: {
      nir_def *off =
         nir_iadd_imm(b, intr->src[1].ssa, nir_intrinsic_base(intr));
      nir_store_global(b, intr->src[0].ssa, payload_pointer(b, s, off),
                       .align_mul = nir_intrinsic_align_mul(intr),
                       .align_offset = nir_intrinsic_align_offset(intr),
                       .write_mask = nir_intrinsic_write_mask(intr),
                       .access = nir_intrinsic_access(intr));
      break;
   }
   case nir_intrinsic_barrier: {
      nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
      if (!(modes & (nir_var_mem_task_payload | nir_var_shader_out)))
         return false;
      if (modes & nir_var_mem_task_payload)
         modes = (modes & ~nir_var_mem_task_payload) | nir_var_mem_global;
      if (modes & nir_var_shader_out) {
         modes = (modes & ~nir_var_shader_out) | nir_var_mem_shared;
         if (l->primitive_components)
            modes |= nir_var_mem_global;
      }
      nir_intrinsic_set_memory_modes(intr, modes);
      return true;
   }
   case nir_intrinsic_set_vertex_and_primitive_count: {
      /* Actual vertex count has no separately established native store.
       * Retain its empty-output meaning; publish only the measured primitive
       * count. Do not invent another header word for a vertex-count field. */
      s->info->uses_vertex_count = true;
      nir_def *p = nir_bcsel(b, nir_ieq_imm(b, intr->src[0].ssa, 0),
                             nir_imm_int(b, 0), intr->src[1].ssa);
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      /* Every supported primitive capacity fits the 16-bit logical count.
       * Its memory handoff may reserve a wider zero-extended word. */
      nir_store_local_apple9_agx(
         b, nir_u2u16(b, p), nir_imm_int(b, l->primitive_count_B),
         .base = s->key->output_root, .bit_size = s->key->primitive_count_bits);
      nir_pop_if(b, NULL);
      /* Keep both logical count SSA operands until a complete native stage
       * contract consumes them. Body compilation emits the measured primitive
       * store but must not silently discard the actual vertex count. */
      return true;
   }
   case nir_intrinsic_launch_mesh_workgroups:
      /* Preserve the collective terminator as an explicit semantic marker.
       * Body compilation records it; a future generated completion ABI must
       * perform the three-lane export, not an inline lane-zero shortcut. */
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      for (unsigned c = 0; c < 3; ++c)
         store_local(b, s->key->grid_root, nir_channel(b, intr->src[0].ssa, c),
                     nir_imm_int(b, s->key->grid_offset_B + 4 * c), 32);
      nir_pop_if(b, NULL);
      /* A structured region permits a terminating return even when the
       * original intrinsic has unreachable instructions after it. */
      nir_push_if(b, nir_imm_true(b));
      nir_end_task_apple9_agx(b);
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   default:
      if (!is_output(intr->intrinsic))
         return false;
      bool store = is_store(intr->intrinsic),
           prim = is_primitive(intr->intrinsic);
      unsigned slot = location(intr), component = nir_intrinsic_component(intr);
      nir_def *index = intr->src[store ? 1 : 0].ssa;
      unsigned mask = store ? nir_intrinsic_write_mask(intr)
                            : BITFIELD_MASK(intr->def.num_components);
      nir_def *values[4] = {0};
      u_foreach_bit(c, mask) {
         nir_def *offset;
         unsigned bits = 32;
         if (slot == VARYING_SLOT_PRIMITIVE_INDICES) {
            offset =
               nir_iadd_imm(b, nir_imul_imm(b, index, l->indices_per_primitive),
                            component + c);
            bits = 8;
         } else if (slot == VARYING_SLOT_CULL_PRIMITIVE) {
            offset = nir_iadd_imm(b, nir_imul_imm(b, index, 4), l->cull_B);
            bits = 8;
         } else if (prim) {
            unsigned channel = l->primitive_component[slot][component + c];
            offset = nir_iadd_imm(
               b, nir_imul_imm(b, index, l->primitive_stride_B), channel * 4);
         } else {
            unsigned channel = l->vertex_component[slot][component + c];
            offset = nir_iadd_imm(b, nir_imul_imm(b, index, 4),
                                  l->position_B + channel * l->vertex_pitch_B);
         }
         nir_def *address = NULL;
         if (prim && slot != VARYING_SLOT_PRIMITIVE_INDICES &&
             slot != VARYING_SLOT_CULL_PRIMITIVE)
            address = nir_iadd(
               b,
               nir_pack_64_2x32(b, nir_load_mesh_primitive_base_apple9_agx(b)),
               nir_u2u64(b, offset));
         if (store) {
            nir_def *value = nir_channel(b, intr->src[0].ssa, c);
            if (slot == VARYING_SLOT_CULL_PRIMITIVE)
               value = value->bit_size == 1 ? nir_b2i32(b, value) : value;
            if (address)
               nir_store_global(b, value, address, .align_mul = 4);
            else
               store_local(b, s->key->output_root, value, offset, bits);
         } else {
            values[c] = address
                           ? nir_load_global(b, 1, 32, address, .align_mul = 4)
                           : nir_load_local_apple9_agx(
                                b, offset, .base = s->key->output_root);
         }
      }
      if (!store) {
         nir_def_replace(&intr->def,
                         nir_vec(b, values, intr->def.num_components));
         return true;
      }
      break;
   }
   nir_instr_remove(&intr->instr);
   return true;
}

bool
agx_nir_lower_apple9_mesh(nir_shader *nir,
                          const struct agx_apple9_mesh_key *key,
                          struct agx_apple9_mesh_body_info *info,
                          const char **reason)
{
   if (reason)
      *reason = NULL;
   struct agx_apple9_mesh_body_info result = {0};
   if (!info)
      return fail(reason, "Apple9 native IO requires body metadata output");
   if (!validate(nir, key, &result.layout, reason))
      return false;
   result.key = *key;
   memcpy(result.logical_size, nir->info.workgroup_size,
          sizeof(result.logical_size));
   result.argument_uniform_words = agx_apple9_mesh_argument_words(key);
   result.requires_native_entry = true;
   result.requires_object_completion = nir->info.stage == MESA_SHADER_TASK;
   result.requires_mesh_completion = nir->info.stage == MESA_SHADER_MESH;
   *info = result;
   struct lower s = {key, info};
   nir_shader_intrinsics_pass(nir, lower_intrinsic, nir_metadata_none, &s);
   return true;
}

bool
agx_nir_apple9_object_grid_export(nir_builder *b,
                                  const struct agx_apple9_mesh_key *key)
{
   if (b->shader->info.stage != MESA_SHADER_TASK || !key ||
       key->grid_root >= 64 ||
       2 * key->grid_root < key->reserved_uniform_words ||
       (key->grid_offset_B & 3) ||
       key->grid_offset_B > AGX_APPLE9_MAX_SHARED_BYTES - 12)
      return false;
   /* Re-expressed from the user-approved object-helper decode (2026-10-14)
    * and encodings of our authored MSL operations, through our own NIR and
    * encoders. This is not derived from the opaque mesh helper.
    * Lanes 0-2 share lane 0's SIMD group. The object-grid fence completes
    * lane 0's grid stores before these loads; another lane mapping would
    * require a workgroup barrier and a newly validated caller contract. */
   nir_def *index = nir_load_local_invocation_index(b);
   nir_fence_mesh_grid_apple9_agx(b);
   nir_push_if(b, nir_ult_imm(b, index, 3));
   nir_def *word = nir_load_local_apple9_agx(
      b, nir_iadd_imm(b, nir_imul_imm(b, index, 4), key->grid_offset_B),
      .base = key->grid_root);
   nir_export_mesh_grid_word_apple9_agx(b, word);
   nir_pop_if(b, NULL);
   return true;
}

/* Region IDs cover both instructions and if conditions, so the validator can
 * reject callee SSA escaping even through a condition in the caller. */
struct object_regions {
   struct hash_table *ids;
   unsigned active, next;
};

static bool
validate_object_regions(struct exec_list *list, struct object_regions *r)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            nir_intrinsic_op op = instr->type == nir_instr_type_intrinsic
                                     ? nir_instr_as_intrinsic(instr)->intrinsic
                                     : nir_intrinsic_nop;
            bool begin = op == nir_intrinsic_begin_object_call_apple9_agx;
            bool end = op == nir_intrinsic_end_object_call_apple9_agx;
            if (begin || end) {
               if (node->parent->type != nir_cf_node_function ||
                   (begin ? r->active != 0 : r->active == 0))
                  return false;
               r->active = begin ? ++r->next : 0;
            } else if (r->active) {
               if (instr->type == nir_instr_type_phi ||
                   instr->type == nir_instr_type_call ||
                   instr->type == nir_instr_type_jump)
                  return false;
               _mesa_hash_table_insert(r->ids, instr,
                                       (void *)(uintptr_t)r->active);
            }
         }
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (r->active)
            _mesa_hash_table_insert(r->ids, nif,
                                    (void *)(uintptr_t)r->active);
         if (!validate_object_regions(&nif->then_list, r) ||
             !validate_object_regions(&nif->else_list, r))
            return false;
      } else if (node->type == nir_cf_node_loop) {
         if (r->active ||
             !validate_object_regions(&nir_cf_node_as_loop(node)->body, r))
            return false;
      } else {
         return false;
      }
   }
   return true;
}

bool
agx_nir_lower_apple9_object_calls(nir_shader *nir, const char **reason)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct util_dynarray calls = UTIL_DYNARRAY_INIT;
   bool found = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         bool call = intr->intrinsic == nir_intrinsic_launch_mesh_workgroups_apple9_agx;
         if (!call && intr->intrinsic != nir_intrinsic_begin_object_call_apple9_agx &&
                      intr->intrinsic != nir_intrinsic_end_object_call_apple9_agx)
            continue;
         found = true;
         if (nir->info.stage != MESA_SHADER_TASK ||
             nir->info.workgroup_size_variable ||
             nir->info.workgroup_size[0] != 16 ||
             nir->info.workgroup_size[1] != 1 ||
             nir->info.workgroup_size[2] != 1 ||
             block->cf_node.parent->type != nir_cf_node_function) {
            util_dynarray_fini(&calls);
            return fail(reason, "object grid calls require a converged T16 leaf caller");
         }
         if (call)
            util_dynarray_append(&calls, intr);
      }
   }
   if (!found) {
      util_dynarray_fini(&calls);
      return true;
   }

   /* Expand only after all NIR motion/optimization. The semantic builder owns
    * the body; selection chooses its system values, compare and load forms. */
   nir_builder b = nir_builder_create(impl);
   util_dynarray_foreach(&calls, nir_intrinsic_instr *, call) {
      const struct agx_apple9_mesh_key key = {
         .grid_root = nir_intrinsic_base(*call),
         .grid_offset_B = nir_intrinsic_range_base(*call),
      };
      b.cursor = nir_before_instr(&(*call)->instr);
      nir_begin_object_call_apple9_agx(&b);
      if (!agx_nir_apple9_object_grid_export(&b, &key)) {
         util_dynarray_fini(&calls);
         return fail(reason, "invalid object grid local storage");
      }
      nir_end_object_call_apple9_agx(&b);
      nir_instr_remove(&(*call)->instr);
   }
   nir_progress(calls.size != 0, impl, nir_metadata_none);
   util_dynarray_fini(&calls);

   struct object_regions r = {.ids = _mesa_pointer_hash_table_create(NULL)};
   if (!r.ids)
      return fail(reason, "out of memory validating object call regions");
   bool valid = validate_object_regions(&impl->body, &r) && !r.active;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         struct hash_entry *entry = _mesa_hash_table_search(r.ids, instr);
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         nir_foreach_use_including_if(use, def) {
            const void *parent = nir_src_is_if(use)
                                    ? (void *)nir_src_use_if(use)
                                    : (void *)nir_src_use_instr(use);
            struct hash_entry *consumer = _mesa_hash_table_search(r.ids, parent);
            /* No SSA interface: inputs come from the callee's own semantic
             * loads. This also prevents lazy selection of a caller ALU def
             * inside the callee and later reuse of that virtual value. */
            valid &= entry ? consumer && consumer->data == entry->data
                           : consumer == NULL;
         }
      }
   }
   _mesa_hash_table_destroy(r.ids, NULL);
   return valid || fail(reason, "invalid closed NIR object call region");
}

nir_shader *
agx_nir_apple9_taskless_object(const struct agx_apple9_mesh_key *key)
{
   if (!key || key->grid_root >= 64 ||
       (key->physical_threads &&
        key->physical_threads != AGX_APPLE9_NATIVE_THREADS) ||
       2 * key->grid_root < key->reserved_uniform_words ||
       key->grid_root == AGX_APPLE9_MESH_DRAW_UNIFORM / 2 ||
       (key->grid_offset_B & 3) ||
       key->grid_offset_B > AGX_APPLE9_MAX_SHARED_BYTES - 12)
      return NULL;
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "apple9_taskless_object");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *address = nir_pack_64_2x32(
      &b, nir_load_native_pointer_apple9_agx(
             &b, .base = AGX_APPLE9_MESH_DRAW_UNIFORM));
   nir_def *count[3];
   for (unsigned i = 0; i < 3; ++i) {
      count[i] = nir_load_global(
         &b, 1, 32, nir_iadd_imm(&b, address, 4 * i), .align_mul = 4,
         .access = ACCESS_NON_WRITEABLE);
   }
   nir_launch_mesh_workgroups(&b, nir_vec(&b, count, 3));
   nir_validate_shader(b.shader, "Apple9 taskless T1 object");
   return b.shader;
}

bool
agx_nir_lower_apple9_native_object(
   nir_shader *nir, bool enable_native_mesh,
   const struct agx_apple9_mesh_key *key,
   struct agx_apple9_mesh_body_info *info, const char **reason)
{
   if (!enable_native_mesh)
      return fail(reason, "native object development path is disabled");
   if (!key || nir->info.stage != MESA_SHADER_TASK ||
       nir->info.task_payload_size || nir->info.shared_size ||
       (key->physical_threads &&
        key->physical_threads != AGX_APPLE9_NATIVE_THREADS))
      return fail(reason,
                  "native object requires physical T16 without payload or user shared memory");
   if (!agx_apple9_native_workgroup_supported(nir, reason))
      return false;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   unsigned launches = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump && !launches &&
             nir_instr_as_jump(instr)->type == nir_jump_return)
            return fail(reason, "native object return may bypass the grid launch");
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         if (op == nir_intrinsic_launch_mesh_workgroups) {
            if (block->cf_node.parent != &impl->cf_node)
               return fail(reason,
                           "native object requires one top-level API grid launch");
            ++launches;
         }
         if (op == nir_intrinsic_load_task_payload ||
             op == nir_intrinsic_store_task_payload)
            return fail(reason, "native object payload is unqualified");
      }
   }
   if (launches != 1)
      return fail(reason, "native object requires one top-level API grid launch");
   struct agx_apple9_mesh_key physical = *key;
   physical.physical_threads = AGX_APPLE9_NATIVE_THREADS;
   if (!agx_nir_lower_apple9_mesh(nir, &physical, info, reason))
      return false;
   nir_lower_returns(nir);
   if (!agx_nir_mask_apple9_mesh_body(nir, &physical, NULL, reason))
      return false;
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   /* Application stores (including all three grid words from API lane 0)
    * finish before the compiler-owned collective sees physical lanes 0..15.
    * The callee's three-word export is independent of the API lane mask. */
   workgroup_rendezvous(&b);
   nir_launch_mesh_workgroups_apple9_agx(&b, .base = key->grid_root,
                                        .range_base = key->grid_offset_B);
   info->requires_object_completion = false;
   nir_progress(true, impl, nir_metadata_none);
   nir_validate_shader(nir, "Apple9 native object workgroup expansion");
   return true;
}
