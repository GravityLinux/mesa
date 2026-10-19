/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Bounded test interpreter for the structured NIR used by mesh formation.
 * Tests supply all memory/semantic bindings. This is not a GPU emulator.
 */
#pragma once
#include "compiler/nir/nir_constant_expressions.h"
#include "agx_apple9_mesh_formation.h"

#define LANES      16
#define ALL_LANES  0xffff
#define INPUT_ADDR 0x20000ull

struct region {
   uint64_t base;
   uint32_t size;
   uint8_t *data, *written;
};
struct machine {
   nir_const_value (*value)[LANES][NIR_MAX_VEC_COMPONENTS];
   nir_block *previous[LANES];
   uint8_t local[128];
   uint64_t formation_inputs[AGX_MESH_FORMATION_DROPS_ADDRESS + 1];
   struct region region[8];
   unsigned barriers, subgroup_barriers, fences, writes;
   unsigned controls, scalars, channels[4];
   unsigned local_root, grid_fences, grid_exports;
   unsigned local_store_lanes, global_store_lanes, atomic_lanes, object_calls;
   uint32_t grid[3];
   uint64_t side_effect_address;
};

static void *
memory(struct machine *m, uint64_t address, unsigned size, bool write)
{
   for (unsigned i = 0; i < ARRAY_SIZE(m->region); ++i) {
      struct region *r = &m->region[i];
      if (address >= r->base && size <= r->size &&
          address - r->base <= r->size - size) {
         if (write) {
            CHECK(i < 5 || i == 7);
            memset(r->written + address - r->base, 1, size);
            m->writes++;
         }
         return r->data + address - r->base;
      }
   }
   fprintf(stderr, "unmapped %s 0x%llx + %u\n", write ? "store" : "load",
           (unsigned long long)address, size);
   exit(1);
}

static nir_const_value *
src(struct machine *m, nir_src s, unsigned lane)
{
   CHECK(s.ssa);
   return m->value[s.ssa->index][lane];
}

static void
instruction(struct machine *m, nir_instr *i, unsigned lane)
{
   switch (i->type) {
   case nir_instr_type_undef: {
      /* SSA repair supplies undefined values only for inactive lanes. Poison
       * them deterministically so accidental dataflow becomes visible. */
      nir_def *def = nir_instr_def(i);
      memset(m->value[def->index][lane], 0xa5,
             def->num_components * sizeof(nir_const_value));
      break;
   }
   case nir_instr_type_load_const: {
      nir_load_const_instr *c = nir_instr_as_load_const(i);
      memcpy(m->value[c->def.index][lane], c->value,
             c->def.num_components * sizeof(nir_const_value));
      break;
   }
   case nir_instr_type_alu: {
      nir_alu_instr *a = nir_instr_as_alu(i);
      nir_const_value values[NIR_ALU_MAX_INPUTS][NIR_MAX_VEC_COMPONENTS] = {0};
      nir_const_value *inputs[NIR_ALU_MAX_INPUTS];
      for (unsigned j = 0; j < nir_op_infos[a->op].num_inputs; ++j) {
         inputs[j] = values[j];
         unsigned components =
            nir_op_infos[a->op].input_sizes[j] ?: a->def.num_components;
         for (unsigned c = 0; c < components; ++c)
            values[j][c] = src(m, a->src[j].src, lane)[a->src[j].swizzle[c]];
      }
      unsigned bits = a->def.bit_size;
      if (nir_op_infos[a->op].output_type & NIR_ALU_TYPE_SIZE_MASK) {
         for (unsigned j = 0; j < nir_op_infos[a->op].num_inputs; ++j)
            if (!(nir_op_infos[a->op].input_types[j] & NIR_ALU_TYPE_SIZE_MASK))
               bits = a->src[j].src.ssa->bit_size;
      }
      uint16_t poison = 0;
      nir_eval_const_opcode(a->op, m->value[a->def.index][lane], &poison,
                            a->def.num_components, bits, inputs, 0);
      CHECK(poison == 0);
      break;
   }
   case nir_instr_type_phi: {
      nir_phi_instr *p = nir_instr_as_phi(i);
      bool found = false;
      nir_foreach_phi_src(s, p) {
         if (s->pred == m->previous[lane]) {
            memcpy(m->value[p->def.index][lane], src(m, s->src, lane),
                   p->def.num_components * sizeof(nir_const_value));
            found = true;
         }
      }
      CHECK(found);
      break;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *n = nir_instr_as_intrinsic(i);
      nir_const_value *out = m->value[n->def.index][lane];
      switch (n->intrinsic) {
      case nir_intrinsic_end_task_apple9_agx:
         break;
      case nir_intrinsic_load_local_invocation_id:
         out[0].u32 = lane;
         out[1].u32 = out[2].u32 = 0;
         break;
      case nir_intrinsic_load_workgroup_size:
         out[0].u32 = LANES;
         out[1].u32 = out[2].u32 = 1;
         break;
      case nir_intrinsic_load_subgroup_size:
         out[0].u32 = 32;
         break;
      case nir_intrinsic_load_subgroup_invocation:
         out[0].u32 = lane;
         break;
      case nir_intrinsic_load_subgroup_id:
         out[0].u32 = 0;
         break;
      case nir_intrinsic_load_num_subgroups:
         out[0].u32 = 1;
         break;
      case nir_intrinsic_fence_mesh_grid_apple9_agx:
         CHECK(!(m->grid_fences & (1u << lane)));
         m->grid_fences |= 1u << lane;
         break;
      case nir_intrinsic_export_mesh_grid_word_apple9_agx:
         CHECK(lane < 3 && m->grid_fences == ALL_LANES);
         CHECK(!(m->grid_exports & (1u << lane)));
         m->grid[lane] = src(m, n->src[0], lane)[0].u32;
         m->grid_exports |= 1u << lane;
         break;
      case nir_intrinsic_mesh_control_apple9_agx:
         CHECK(nir_intrinsic_base(n) == 1);
         CHECK(src(m, n->src[0], lane)[0].u16 == 1);
         CHECK(src(m, n->src[1], lane)[0].u16 == 3);
         CHECK(!(m->controls & (1u << lane)));
         m->controls |= 1u << lane;
         break;
      case nir_intrinsic_mesh_submit_channel_apple9_agx: {
         unsigned channel = src(m, n->src[0], lane)[0].u32;
         CHECK(channel < 4 && lane < 3);
         CHECK(src(m, n->src[1], lane)[0].u32 == 32 + channel * 16);
         CHECK(m->controls == ALL_LANES);
         CHECK(!(m->channels[channel] & (1u << lane)));
         m->channels[channel] |= 1u << lane;
         break;
      }
      case nir_intrinsic_mesh_submit_scalar_apple9_agx:
         CHECK(lane == 0 && src(m, n->src[0], lane)[0].u32 == 3);
         for (unsigned c = 0; c < 4; ++c)
            CHECK(m->channels[c] == 7);
         CHECK(!m->scalars++);
         break;
      case nir_intrinsic_load_local_invocation_index:
         out[0].u32 = lane;
         break;
      case nir_intrinsic_load_native_pointer_apple9_agx: {
         unsigned base = nir_intrinsic_base(n);
         CHECK(base == AGX_APPLE9_MESH_DRAW_UNIFORM ||
               (base == AGX_APPLE9_MESH_PAYLOAD_UNIFORM && m->side_effect_address));
         uint64_t address = base == AGX_APPLE9_MESH_DRAW_UNIFORM
                               ? INPUT_ADDR : m->side_effect_address;
         out[0].u32 = address;
         out[1].u32 = address >> 32;
         break;
      }
      case nir_intrinsic_load_global: {
         unsigned bytes = n->def.bit_size / 8;
         uint64_t address = src(m, n->src[0], lane)[0].u64;
         for (unsigned c = 0; c < n->def.num_components; ++c) {
            out[c].u64 = 0;
            memcpy(&out[c], memory(m, address + c * bytes, bytes, false),
                   bytes);
         }
         break;
      }
      case nir_intrinsic_global_atomic: {
         m->atomic_lanes |= 1u << lane;
         CHECK(nir_intrinsic_atomic_op(n) == nir_atomic_op_iadd);
         uint32_t *counter = memory(m, src(m, n->src[0], lane)[0].u64, 4, true);
         out[0].u32 = *counter;
         *counter += src(m, n->src[1], lane)[0].u32;
         break;
      }
      case nir_intrinsic_store_global: {
         m->global_store_lanes |= 1u << lane;
         unsigned bytes = n->src[0].ssa->bit_size / 8;
         uint64_t address = src(m, n->src[1], lane)[0].u64;
         for (unsigned c = 0; c < n->src[0].ssa->num_components; ++c)
            if (nir_intrinsic_write_mask(n) & BITFIELD_BIT(c)) {
               memcpy(memory(m, address + c * bytes, bytes, true),
                      &src(m, n->src[0], lane)[c], bytes);
            }
         break;
      }
      case nir_intrinsic_store_local_apple9_agx: {
         m->local_store_lanes |= 1u << lane;
         unsigned bytes = nir_intrinsic_bit_size(n) / 8;
         unsigned offset = src(m, n->src[1], lane)[0].u32;
         CHECK(nir_intrinsic_base(n) ==
               (m->local_root ?: AGX_APPLE9_MESH_OUTPUT_ROOT));
         CHECK(offset + bytes <= sizeof(m->local));
         memcpy(m->local + offset, src(m, n->src[0], lane), bytes);
         break;
      }
      case nir_intrinsic_load_local_apple9_agx: {
         unsigned offset = src(m, n->src[0], lane)[0].u32;
         CHECK(nir_intrinsic_base(n) ==
               (m->local_root ?: AGX_APPLE9_MESH_OUTPUT_ROOT));
         CHECK(offset + 4 <= sizeof(m->local));
         memcpy(out, m->local + offset, 4);
         break;
      }
      default:
         fprintf(stderr, "unsupported intrinsic %s\n",
                 nir_intrinsic_infos[n->intrinsic].name);
         exit(1);
      }
      break;
   }
   default:
      fprintf(stderr, "unsupported instruction %u\n", i->type);
      exit(1);
   }
}

static void
execute(struct machine *m, struct exec_list *list, unsigned active)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         nir_block *block = nir_cf_node_as_block(node);
         nir_foreach_instr(i, block) {
            if (i->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(i)->intrinsic ==
                   nir_intrinsic_launch_mesh_workgroups_apple9_agx) {
               nir_intrinsic_instr *call = nir_instr_as_intrinsic(i);
               CHECK(active == ALL_LANES && m->barriers);
               CHECK(nir_intrinsic_base(call) == m->local_root);
               CHECK(!m->object_calls++);
               unsigned offset = nir_intrinsic_range_base(call);
               CHECK(offset + 12 <= sizeof(m->local));
               memcpy(m->grid, m->local + offset, sizeof(m->grid));
               m->grid_fences = ALL_LANES;
               m->grid_exports = 7;
               continue;
            }
            if (i->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(i)->intrinsic == nir_intrinsic_ballot) {
               nir_intrinsic_instr *ballot = nir_instr_as_intrinsic(i);
               CHECK(ballot->def.num_components == 1 && ballot->def.bit_size == 32);
               unsigned bits = 0;
               for (unsigned lane = 0; lane < LANES; ++lane)
                  if ((active & (1u << lane)) && src(m, ballot->src[0], lane)[0].b)
                     bits |= 1u << lane;
               for (unsigned lane = 0; lane < LANES; ++lane)
                  if (active & (1u << lane))
                     m->value[ballot->def.index][lane][0].u32 = bits;
               continue;
            }
            if (i->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(i)->intrinsic == nir_intrinsic_barrier) {
               nir_intrinsic_instr *b = nir_instr_as_intrinsic(i);
               if (nir_intrinsic_execution_scope(b) == SCOPE_WORKGROUP) {
                  CHECK(active == ALL_LANES);
                  m->barriers++;
               } else if (nir_intrinsic_execution_scope(b) == SCOPE_SUBGROUP) {
                  if (active)
                     m->subgroup_barriers++;
               } else if (active) {
                  CHECK(nir_intrinsic_memory_scope(b) == SCOPE_DEVICE);
                  m->fences++;
               }
            } else {
               for (unsigned lane = 0; lane < LANES; ++lane)
                  if (active & BITFIELD_BIT(lane))
                     instruction(m, i, lane);
            }
         }
         for (unsigned lane = 0; lane < LANES; ++lane)
            if (active & BITFIELD_BIT(lane))
               m->previous[lane] = block;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         unsigned yes = 0;
         for (unsigned lane = 0; lane < LANES; ++lane)
            if ((active & BITFIELD_BIT(lane)) &&
                src(m, nif->condition, lane)[0].b)
               yes |= BITFIELD_BIT(lane);
         if (yes)
            execute(m, &nif->then_list, yes);
         if (active & ~yes)
            execute(m, &nif->else_list, active & ~yes);
      } else {
         CHECK(!"unsupported control flow");
      }
   }
}
