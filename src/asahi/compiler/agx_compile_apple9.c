/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile_apple9.h"
#include "agx_apple9_ir.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "gallium/include/pipe/p_defines.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Apple9 compute compilation uses one NIR-to-VIR-to-machine pipeline. */

struct apple9_emitter {
   struct util_dynarray bytes;
   unsigned instructions;
};

static bool
apple9_emit_packed(struct apple9_emitter *emitter,
                   const struct agx_apple9_packed_instruction *packed);

static void
apple9_emit(struct apple9_emitter *emitter, unsigned length,
            const uint8_t *encoded)
{
   memcpy(util_dynarray_grow_bytes(&emitter->bytes, 1, length), encoded,
          length);
   emitter->instructions++;
}

static void
apple9_emit_stop(struct apple9_emitter *emitter)
{
   const uint8_t encoded[4] = {0x0e, 0, 0, 0};
   apple9_emit(emitter, sizeof(encoded), encoded);
}

/*
 * Gallium's GLSL path keeps SSBO offsets as a low/high pair until late NIR.
 * Peel only representation-preserving wrappers plus additions by zero.
 */
static nir_scalar
apple9_chase_trivial(nir_scalar scalar)
{
   for (unsigned iteration = 0; iteration < 32; ++iteration) {
      scalar = nir_scalar_chase_movs(scalar);
      if (nir_def_instr_type(scalar.def) != nir_instr_type_alu)
         return scalar;

      nir_op op = nir_scalar_alu_op(scalar);
      if (op == nir_op_vec2 || op == nir_op_vec3 || op == nir_op_vec4) {
         scalar = nir_scalar_chase_alu_src(scalar, scalar.comp);
         continue;
      }

      if (op == nir_op_iadd) {
         nir_scalar left =
            apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 0));
         nir_scalar right =
            apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 1));
         if (nir_scalar_is_const(left) && nir_scalar_as_uint(left) == 0) {
            scalar = right;
            continue;
         }
         if (nir_scalar_is_const(right) && nir_scalar_as_uint(right) == 0) {
            scalar = left;
            continue;
         }
      }

      return scalar;
   }

   return scalar;
}

struct apple9_system_source {
   uint8_t selector;
   bool zext16;
   bool global_id;
};

/*
 * EXP-M4-29 establishes two regular compute system-register forms.  Global
 * and workgroup positions are direct 32-bit values.  Local position/index,
 * SIMD indices, and workgroup-size components use the same narrow GET_SR plus
 * zero-extension pair and differ only in the selector.  Native asymmetric-3D
 * executions establish selectors 0x98..0x9a as the local-size XYZ tuple.
 *
 * Selectors 0xa8..0xaa are a different dispatch-time tuple used by Metal's
 * load_num_workgroups lowering.  Their bare values track the CDM local tuple,
 * while the public group count is derived from caller-owned global-thread
 * metadata.  Do not alias that packaging contract to load_workgroup_size.
 */
static bool
apple9_system_source(nir_scalar scalar, struct apple9_system_source *source)
{
   scalar = apple9_chase_trivial(scalar);
   if (nir_def_instr_type(scalar.def) != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_op op = nir_def_as_intrinsic(scalar.def)->intrinsic;
   /* NIR pixel coordinates are 16-bit. The fragment SR reads a full word;
    * framebuffer coordinates fit in 16 bits (maximum dimension 16384). */
   if (scalar.def->bit_size != 32 &&
       !(op == nir_intrinsic_load_pixel_coord && scalar.def->bit_size == 16))
      return false;
   unsigned components = scalar.def->num_components;
   uint8_t base;
   bool zext16 = false;
   bool global_id = false;

   switch (op) {
   case nir_intrinsic_load_pixel_coord:
      base = 0xa0;
      components = 2;
      break;
   case nir_intrinsic_load_global_invocation_id:
      base = 0xa0;
      global_id = true;
      break;
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_vertex_id_zero_base:
      base = 0xdd;
      components = 1;
      break;
   case nir_intrinsic_load_workgroup_id:
      base = 0x9c;
      break;
   case nir_intrinsic_load_local_invocation_id:
      base = 0xa4;
      zext16 = true;
      break;
   case nir_intrinsic_load_workgroup_size:
      base = 0x98;
      zext16 = true;
      break;
   case nir_intrinsic_load_local_invocation_index:
      base = 0xa7;
      components = 1;
      zext16 = true;
      break;
   case nir_intrinsic_load_subgroup_invocation:
      base = 0x82;
      components = 1;
      zext16 = true;
      break;
   case nir_intrinsic_load_subgroup_id:
      base = 0x85;
      components = 1;
      zext16 = true;
      break;
   default:
      return false;
   }

   if (scalar.comp >= components || scalar.comp >= 3)
      return false;
   *source = (struct apple9_system_source){
      .selector = base + scalar.comp,
      .zext16 = zext16,
      .global_id = global_id,
   };
   return true;
}

static bool
apple9_const_u32(nir_scalar scalar, uint32_t *value)
{
   scalar = apple9_chase_trivial(scalar);
   if (!nir_scalar_is_const(scalar) || scalar.def->bit_size != 32)
      return false;

   *value = nir_scalar_as_uint(scalar);
   return true;
}

static bool
apple9_element_index(nir_def *offset, nir_scalar *index, unsigned *index_scale,
                     unsigned *index_add, unsigned element_size)
{
   if (element_size != 1 && element_size != 2 && element_size != 4)
      return false;
   nir_scalar candidate = apple9_chase_trivial(nir_get_scalar(offset, 0));
   uint64_t byte_stride = 1;
   uint64_t byte_add = 0;

   /* Canonicalize the complete constant affine shell, independent of whether
    * NIR spells it as (i * stride + add), ((i * inner) + add) * outer, or
    * shifts.  The resulting scale and add are emitted as ordinary integer
    * address calculations before the memory instruction. */
   for (unsigned depth = 0; depth < 32; ++depth) {
      candidate = apple9_chase_trivial(candidate);
      if (nir_def_instr_type(candidate.def) != nir_instr_type_alu)
         break;

      nir_op op = nir_scalar_alu_op(candidate);
      nir_scalar left =
         apple9_chase_trivial(nir_scalar_chase_alu_src(candidate, 0));
      nir_scalar right =
         apple9_chase_trivial(nir_scalar_chase_alu_src(candidate, 1));
      uint32_t constant;

      if (op == nir_op_iadd) {
         nir_scalar affine;
         if (apple9_const_u32(right, &constant))
            affine = left;
         else if (apple9_const_u32(left, &constant))
            affine = right;
         else
            break;

         if (constant > (UINT32_MAX - byte_add) / byte_stride)
            return false;
         byte_add += (uint64_t)constant * byte_stride;
         candidate = affine;
         continue;
      }

      uint32_t factor;
      nir_scalar affine;
      if (op == nir_op_ishl) {
         if (!apple9_const_u32(right, &constant) || constant >= 32)
            return false;
         factor = 1u << constant;
         affine = left;
      } else if (op == nir_op_imul || op == nir_op_amul) {
         if (apple9_const_u32(right, &factor))
            affine = left;
         else if (apple9_const_u32(left, &factor))
            affine = right;
         else
            break;
      } else {
         break;
      }

      if (factor == 0 || byte_stride > UINT32_MAX / factor)
         return false;
      byte_stride *= factor;
      candidate = affine;
   }

   uint32_t constant_index;
   if (apple9_const_u32(candidate, &constant_index)) {
      const uint64_t byte_offset =
         (uint64_t)constant_index * byte_stride + byte_add;
      if (byte_offset > UINT32_MAX || byte_offset % element_size != 0)
         return false;
      if (index != NULL)
         *index = candidate;
      if (index_scale != NULL)
         *index_scale = 0;
      if (index_add != NULL)
         *index_add = byte_offset / element_size;
      return true;
   }

   if (byte_stride == 1 && element_size != 1) {
      /* A raw scalar offset is already an element index only for byte data. */
      return false;
   }

   if (byte_stride % element_size != 0 || byte_add % element_size != 0) {
      return false;
   } else {
      candidate = apple9_chase_trivial(candidate);
   }
   if (candidate.def->bit_size != 32)
      return false;

   if (index != NULL)
      *index = candidate;
   if (index_scale != NULL)
      *index_scale = byte_stride / element_size;
   if (index_add != NULL)
      *index_add = byte_add / element_size;
   return true;
}

struct apple9_scalar_load {
   nir_intrinsic_instr *intr;
   nir_block *block;
   nir_scalar index;
   unsigned argument;
   unsigned component;
   unsigned index_scale;
   unsigned index_add;
   unsigned bit_size;
};

struct apple9_buffer_store {
   nir_intrinsic_instr *intr;
   nir_block *block;
   nir_scalar index;
   unsigned argument;
   unsigned components;
   unsigned index_scale;
   unsigned index_add;
   unsigned bit_size;
   uint32_t output[4];
   uint32_t lowered_index;
};

struct apple9_buffer_atomic {
   nir_intrinsic_instr *intr;
   nir_block *block;
   nir_scalar index;
   unsigned argument;
   unsigned index_scale;
   unsigned index_add;
   enum agx_apple9_atomic_op op;
};

static bool
apple9_validate_cf_list(struct exec_list *list, const char **reason)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (!apple9_validate_cf_list(&nif->then_list, reason) ||
             !apple9_validate_cf_list(&nif->else_list, reason))
            return false;
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (nir_loop_has_continue_construct(loop)) {
            *reason = "Apple9 loop continuation constructs were not lowered";
            return false;
         }
         if (!apple9_validate_cf_list(&loop->body, reason))
            return false;
         break;
      }
      default:
         *reason = "Apple9 compiler requires structured NIR control flow";
         return false;
      }
   }

   return true;
}

static bool
apple9_cf_list_has_control_flow(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if || node->type == nir_cf_node_loop)
         return true;
      if (node->type != nir_cf_node_block)
         return false;
   }

   return false;
}

static nir_block *
apple9_cf_list_last_block(struct exec_list *list)
{
   nir_block *last = NULL;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block)
         last = nir_cf_node_as_block(node);
   }
   return last;
}

static bool
apple9_instruction_is_in_subset(nir_instr *instr, bool graphics)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
      return true;
   case nir_instr_type_alu: {
      nir_op op = nir_instr_as_alu(instr)->op;
      switch (op) {
      case nir_op_mov:
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4:
      case nir_op_b2b1:
      case nir_op_b2i32:
      case nir_op_b2f32:
      case nir_op_bcsel:
      case nir_op_inot:
      case nir_op_ineg:
      case nir_op_fabs:
      case nir_op_fneg:
      case nir_op_u2f32:
      case nir_op_i2f32:
      case nir_op_f2i32:
      case nir_op_f2u32:
      case nir_op_i2i8:
      case nir_op_i2i16:
      case nir_op_i2i32:
      case nir_op_u2u8:
      case nir_op_u2u16:
      case nir_op_u2u32:
      case nir_op_iadd:
      case nir_op_isub:
      case nir_op_imul:
      case nir_op_amul:
      case nir_op_iand:
      case nir_op_ior:
      case nir_op_ixor:
      case nir_op_ishl:
      case nir_op_ishr:
      case nir_op_ushr:
      case nir_op_imin:
      case nir_op_imax:
      case nir_op_umin:
      case nir_op_umax:
      case nir_op_fadd:
      case nir_op_fsub:
      case nir_op_fmul:
      case nir_op_frsq:
      case nir_op_fsqrt:
      case nir_op_fsin_factor_agx:
      case nir_op_fexp2:
      case nir_op_flog2:
      case nir_op_ffloor:
      case nir_op_fceil:
      case nir_op_ftrunc:
      case nir_op_fround_even:
      case nir_op_frcp:
      case nir_op_fmin:
      case nir_op_fmax:
      case nir_op_ffma:
      case nir_op_ffma_weak:
      case nir_op_ieq:
      case nir_op_ine:
      case nir_op_ilt:
      case nir_op_ige:
      case nir_op_ult:
      case nir_op_uge:
      case nir_op_feq:
      case nir_op_fneu:
      case nir_op_flt:
      case nir_op_fge:
         return true;
      default:
         return false;
      }
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
      if (graphics && (op == nir_intrinsic_load_vertex_id ||
                       op == nir_intrinsic_load_vertex_id_zero_base ||
                       op == nir_intrinsic_load_pixel_coord ||
                       op == nir_intrinsic_load_barycentric_pixel ||
                       op == nir_intrinsic_load_interpolated_input ||
                       op == nir_intrinsic_load_output ||
                       op == nir_intrinsic_store_output))
         return true;
      return op == nir_intrinsic_load_global_invocation_id ||
             op == nir_intrinsic_load_workgroup_id ||
             op == nir_intrinsic_load_local_invocation_id ||
             op == nir_intrinsic_load_local_invocation_index ||
             op == nir_intrinsic_load_workgroup_size ||
             op == nir_intrinsic_load_num_workgroups ||
             op == nir_intrinsic_load_subgroup_invocation ||
             op == nir_intrinsic_load_subgroup_id ||
             op == nir_intrinsic_load_subgroup_size ||
             op == nir_intrinsic_load_ssbo || op == nir_intrinsic_load_ubo ||
             op == nir_intrinsic_store_ssbo ||
             op == nir_intrinsic_ssbo_atomic ||
             op == nir_intrinsic_ssbo_atomic_swap;
   }
   case nir_instr_type_phi:
      return true;
   case nir_instr_type_jump: {
      nir_jump_type type = nir_instr_as_jump(instr)->type;
      return type == nir_jump_break || type == nir_jump_continue;
   }
   default:
      return false;
   }
}

struct apple9_loop_context {
   struct apple9_loop_context *parent;
   nir_loop *nir;
   nir_block *exit;
   unsigned depth;
   unsigned mask_depth;
};

struct apple9_dag_lower {
   struct agx_apple9_vir_program program;
   nir_shader *nir;
   uint32_t *ssa_to_vreg;
   unsigned ssa_map_count;
   uint32_t system_vreg[256];
   uint32_t zero_vreg;
   uint32_t perspective_reciprocal;
   bool perspective_ready;
   const struct agx_apple9_varying_layout *varyings;
   unsigned position_mask;
   unsigned color_stores;
   bool tile_read;
   bool tile_access;
   struct apple9_scalar_load *loads;
   unsigned load_count;
   struct apple9_buffer_atomic *atomics;
   unsigned atomic_count;
   unsigned argument_base;
   unsigned load_instruction_count;
   unsigned emitted_load_count;
   nir_block *active_load_block;
   unsigned active_load_instruction_count;
   unsigned active_emitted_load_count;
   bool structured_cf;
   unsigned mask_depth;
   struct apple9_loop_context *loop;
   const char *reason;
};

static uint8_t
apple9_current_load_flags(struct apple9_dag_lower *lower,
                          const struct apple9_scalar_load *load,
                          bool native_vector)
{
   assert(lower->active_load_instruction_count > 0);
   assert(lower->active_emitted_load_count <
          lower->active_load_instruction_count);
   assert(lower->emitted_load_count < lower->load_instruction_count);

   uint8_t flags = lower->emitted_load_count + 1 < lower->load_instruction_count
                      ? AGX_APPLE9_DEVICE_LOAD_HAS_NEXT
                      : 0;

   /* Native Metal's byte-1 bit 4 is an address-form selector, not a group
    * boundary. It is set when an unmodified get_sr result supplies the
    * element index directly. Scalar affine lowering destroys that form;
    * native vector loads incorporate their natural tuple stride themselves. */
   struct apple9_system_source system;
   const unsigned element_add = load->index_add + load->component;
   if (apple9_system_source(load->index, &system) &&
       ((native_vector && load->index_add == 0) ||
        (!native_vector && load->index_scale == 1 && element_add == 0)))
      flags |= AGX_APPLE9_DEVICE_LOAD_RAW_SYSTEM_INDEX;

   return flags;
}

static uint32_t
apple9_dag_emit(struct apple9_dag_lower *lower, enum agx_apple9_vir_opcode op,
                enum agx_apple9_encoding encoding, const uint32_t *src,
                unsigned nr_srcs, uint32_t immediate)
{
   uint32_t value = agx_apple9_vir_emit(&lower->program, op, encoding, src,
                                        nr_srcs, immediate);
   if (value == AGX_APPLE9_VREG_INVALID)
      lower->reason = "out of memory building Apple9 virtual IR";
   return value;
}

/* Extended integer OR is the general Apple9 bit-copy form available to this
 * compiler: x | x preserves every bit, accepts any r0-r63 source, and can
 * place the result outside compact destination banks. */
static uint32_t
apple9_dag_general_copy(struct apple9_dag_lower *lower, uint32_t source)
{
   if (source == AGX_APPLE9_VREG_INVALID)
      return source;
   const uint32_t sources[] = {source, source};
   return apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR,
                          AGX_APPLE9_ENC_LOGIC_EXTENDED, sources,
                          ARRAY_SIZE(sources), 0);
}

static uint32_t
apple9_dag_emit_constrained(struct apple9_dag_lower *lower,
                            enum agx_apple9_vir_opcode op,
                            enum agx_apple9_encoding encoding,
                            const uint32_t *sources, unsigned nr_sources,
                            uint32_t immediate)
{
   return apple9_dag_general_copy(
      lower,
      apple9_dag_emit(lower, op, encoding, sources, nr_sources, immediate));
}

static uint32_t
apple9_dag_imm(struct apple9_dag_lower *lower, uint32_t value)
{
   /* Metal normally folds a scalar literal into the consuming ALU operand
    * descriptor.  Until those immediate forms are selected generally, use
    * the independently executed mode-2 raw write.  EXP-M4-37 validates its
    * complete six-bit destination field, so the allocator may place the
    * result anywhere in r0..r63. */
   if (value > 0x7f)
      return apple9_dag_emit(lower, AGX_APPLE9_VIR_IMM,
                             AGX_APPLE9_ENC_MOV_IMM32, NULL, 0, value);

   return apple9_dag_emit_constrained(lower, AGX_APPLE9_VIR_IMM,
                                      AGX_APPLE9_ENC_MOV_IMM_COMPACT, NULL, 0,
                                      value);
}

static uint32_t
apple9_dag_zero(struct apple9_dag_lower *lower)
{
   if (lower->zero_vreg == AGX_APPLE9_VREG_INVALID)
      lower->zero_vreg = apple9_dag_imm(lower, 0);
   return lower->zero_vreg;
}

static uint32_t
apple9_dag_system(struct apple9_dag_lower *lower,
                  struct apple9_system_source system)
{
   if (lower->system_vreg[system.selector] == AGX_APPLE9_VREG_INVALID) {
      enum agx_apple9_vir_opcode op = system.global_id
                                         ? AGX_APPLE9_VIR_GET_GLOBAL_ID
                                         : AGX_APPLE9_VIR_GET_SR;
      enum agx_apple9_encoding encoding =
         system.selector == 0xdd ? AGX_APPLE9_ENC_GET_VERTEX_ID
         : system.zext16         ? AGX_APPLE9_ENC_GET_SR_ZEXT16
                                 : AGX_APPLE9_ENC_GET_SR;
      uint32_t immediate =
         system.global_id
            ? system.selector - 0xa0
            : system.selector | (system.zext16 ? 0 : (0x10u << 8));

      /* Both GET_SR families have a proven low destination contract.
       * Immediately move the value to the general bank so the rest of
       * instruction selection does not inherit that constraint. */
      lower->system_vreg[system.selector] =
         apple9_dag_emit_constrained(lower, op, encoding, NULL, 0, immediate);
   }

   return lower->system_vreg[system.selector];
}

static enum agx_apple9_vir_opcode
apple9_dag_binary_opcode(nir_op op)
{
   switch (op) {
   case nir_op_iadd:
      return AGX_APPLE9_VIR_IADD;
   case nir_op_isub:
      return AGX_APPLE9_VIR_ISUB;
   case nir_op_iand:
      return AGX_APPLE9_VIR_IAND;
   case nir_op_ior:
      return AGX_APPLE9_VIR_IOR;
   case nir_op_ixor:
      return AGX_APPLE9_VIR_IXOR;
   case nir_op_imin:
      return AGX_APPLE9_VIR_IMIN;
   case nir_op_imax:
      return AGX_APPLE9_VIR_IMAX;
   case nir_op_umin:
      return AGX_APPLE9_VIR_UMIN;
   case nir_op_umax:
      return AGX_APPLE9_VIR_UMAX;
   case nir_op_fadd:
      return AGX_APPLE9_VIR_FADD;
   case nir_op_fsub:
      return AGX_APPLE9_VIR_FSUB;
   case nir_op_fmul:
      return AGX_APPLE9_VIR_FMUL;
   case nir_op_fmin:
      return AGX_APPLE9_VIR_FMIN;
   case nir_op_fmax:
      return AGX_APPLE9_VIR_FMAX;
   default:
      return (enum agx_apple9_vir_opcode) - 1;
   }
}

static enum agx_apple9_encoding
apple9_dag_binary_encoding(nir_op op)
{
   switch (op) {
   case nir_op_iadd:
   case nir_op_isub:
      return AGX_APPLE9_ENC_INT_ADD_EXTENDED;
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
      return AGX_APPLE9_ENC_LOGIC_EXTENDED;
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
   case nir_op_fmin:
   case nir_op_fmax:
      return AGX_APPLE9_ENC_MINMAX_COMPACT;
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_fmul:
      return AGX_APPLE9_ENC_FLOAT2_COMPACT;
   default:
      return AGX_APPLE9_ENC_COUNT;
   }
}

static uint32_t apple9_lower_dag_scalar(struct apple9_dag_lower *lower,
                                        nir_scalar scalar);
static uint32_t apple9_lower_bool_scalar(struct apple9_dag_lower *lower,
                                         nir_scalar scalar);
static uint32_t apple9_emit_dag_select_raw(struct apple9_dag_lower *lower,
                                           uint32_t cmp_a, uint32_t cmp_b,
                                           uint32_t if_true, uint32_t if_false,
                                           uint32_t immediate);
static uint32_t apple9_dag_shift_imm(struct apple9_dag_lower *lower, nir_op op,
                                     uint32_t source, unsigned amount);

static uint32_t
apple9_dag_hidden_load(struct apple9_dag_lower *lower, unsigned binding,
                       unsigned element)
{
   uint32_t index = apple9_dag_imm(lower, element);
   if (index == AGX_APPLE9_VREG_INVALID)
      return index;

   const struct agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t value = agx_apple9_vir_emit_device_load(&lower->program, binding,
                                                    index, &contract);
   if (value == AGX_APPLE9_VREG_INVALID ||
       !agx_apple9_vir_set_device_load_contract(
          &lower->program, value, 0, AGX_APPLE9_SCOREBOARD_SLOT_AUTO)) {
      lower->reason = "could not emit an Apple9 hidden-resource load";
      return AGX_APPLE9_VREG_INVALID;
   }

   return value;
}

/* Compute ceil(numerator / divisor) for the complete Apple9 dispatch domain.
 * T8132 exhaustively satisfies |D * frcp(float(D)) - 1| <= 2^-18 for
 * D=1..1024.  Together with N <= 65535*1024, this makes the rounded quotient
 * candidate either floor(N/D) or ceil(N/D); one exact integer comparison
 * selects the latter. */
static uint32_t
apple9_dag_ceil_udiv(struct apple9_dag_lower *lower, uint32_t numerator,
                     uint32_t divisor)
{
   if (numerator == AGX_APPLE9_VREG_INVALID ||
       divisor == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;

   uint32_t numerator_f =
      apple9_dag_emit(lower, AGX_APPLE9_VIR_U2F32, AGX_APPLE9_ENC_UINT_TO_FLOAT,
                      &numerator, 1, 0);
   uint32_t divisor_f =
      apple9_dag_emit(lower, AGX_APPLE9_VIR_U2F32, AGX_APPLE9_ENC_UINT_TO_FLOAT,
                      &divisor, 1, 0);
   uint32_t reciprocal =
      apple9_dag_emit(lower, AGX_APPLE9_VIR_FRCP,
                      AGX_APPLE9_ENC_FLOAT_SPECIAL, &divisor_f, 1, 0x03);
   uint32_t half = apple9_dag_imm(lower, 0x3f000000);
   uint32_t estimate_sources[3] = {numerator_f, reciprocal, half};
   uint32_t estimate =
      apple9_dag_emit(lower, AGX_APPLE9_VIR_FMA, AGX_APPLE9_ENC_FLOAT3_EXTENDED,
                      estimate_sources, ARRAY_SIZE(estimate_sources), 0);
   uint32_t quotient =
      apple9_dag_emit(lower, AGX_APPLE9_VIR_F2U32, AGX_APPLE9_ENC_FLOAT_TO_UINT,
                      &estimate, 1, 0);
   uint32_t zero = apple9_dag_zero(lower);
   uint32_t product_sources[3] = {quotient, divisor, zero};
   uint32_t product = apple9_dag_emit(
      lower, AGX_APPLE9_VIR_IMAD, AGX_APPLE9_ENC_INT_MAD_EXTENDED,
      product_sources, ARRAY_SIZE(product_sources), 0);
   uint32_t one = apple9_dag_imm(lower, 1);
   uint32_t increment_sources[2] = {quotient, one};
   uint32_t incremented = apple9_dag_emit(
      lower, AGX_APPLE9_VIR_IADD, AGX_APPLE9_ENC_INT_ADD_EXTENDED,
      increment_sources, ARRAY_SIZE(increment_sources), 0);

   return apple9_emit_dag_select_raw(lower, product, numerator, incremented,
                                     quotient, AGX_APPLE9_SELECT_ULT);
}

static uint32_t
apple9_dag_num_workgroups(struct apple9_dag_lower *lower, unsigned component)
{
   if (component >= 3) {
      lower->reason = "Apple9 load_num_workgroups has an invalid component";
      return AGX_APPLE9_VREG_INVALID;
   }

   /* Hidden resource 0 is q0 and resource 1 is q1.  Direct dispatches publish
    * total threads and 1; indirect dispatches publish group counts and local
    * size.  This is the native normalized package contract. */
   uint32_t q0 = apple9_dag_hidden_load(lower, 0, component);
   if (q0 == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;

   uint32_t divisor;
   if (lower->nir->info.workgroup_size_variable) {
      divisor = apple9_dag_system(lower, (struct apple9_system_source){
                                            .selector = 0x98 + component,
                                            .zext16 = true,
                                         });
   } else {
      const uint32_t local = lower->nir->info.workgroup_size[component];
      if (!local) {
         lower->reason = "Apple9 load_num_workgroups has an invalid local size";
         return AGX_APPLE9_VREG_INVALID;
      }
      if (local == 1)
         return q0;
      divisor = apple9_dag_imm(lower, local);
   }

   uint32_t q1 = apple9_dag_hidden_load(lower, 1, component);
   uint32_t direct = apple9_dag_ceil_udiv(lower, q0, divisor);
   uint32_t one = apple9_dag_imm(lower, 1);
   uint32_t mode_sources[2] = {q1, one};
   uint32_t mode =
      apple9_dag_emit(lower, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED,
                      mode_sources, ARRAY_SIZE(mode_sources), 0);
   if (q1 == AGX_APPLE9_VREG_INVALID || direct == AGX_APPLE9_VREG_INVALID ||
       one == AGX_APPLE9_VREG_INVALID || mode == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;

   /* q1 == 1 selects the direct ceiling quotient; otherwise q0 is already the
    * caller's indirect group count. */
   return apple9_emit_dag_select_raw(lower, mode, one, direct, q0,
                                     AGX_APPLE9_SELECT_ULT);
}

static uint32_t
apple9_lower_dag_source(struct apple9_dag_lower *lower, nir_scalar parent,
                        unsigned source)
{
   return apple9_lower_dag_scalar(
      lower, apple9_chase_trivial(nir_scalar_chase_alu_src(parent, source)));
}

enum apple9_compare_domain {
   APPLE9_COMPARE_FLOAT,
   APPLE9_COMPARE_SIGNED,
   APPLE9_COMPARE_UNSIGNED,
   APPLE9_COMPARE_INTEGER,
};

enum apple9_compare_relation {
   APPLE9_COMPARE_EQUAL,
   APPLE9_COMPARE_NOT_EQUAL,
   APPLE9_COMPARE_LESS,
   APPLE9_COMPARE_GREATER_EQUAL,
};

struct apple9_compare {
   enum apple9_compare_domain domain;
   enum apple9_compare_relation relation;
};

/* NIR has one canonical opcode for each ordered comparison.  Source-language
 * > and <= arrive as LESS and GREATER_EQUAL with their operands exchanged.
 * Keep this semantic normalization shared by value-producing SELECT and the
 * transient predicate/mask path so their signedness and IEEE behavior cannot
 * drift apart as either encoding family grows. */
static bool
apple9_normalize_compare(nir_op op, struct apple9_compare *compare)
{
   switch (op) {
   case nir_op_feq:
      *compare =
         (struct apple9_compare){APPLE9_COMPARE_FLOAT, APPLE9_COMPARE_EQUAL};
      break;
   case nir_op_fneu:
      *compare = (struct apple9_compare){APPLE9_COMPARE_FLOAT,
                                         APPLE9_COMPARE_NOT_EQUAL};
      break;
   case nir_op_flt:
      *compare =
         (struct apple9_compare){APPLE9_COMPARE_FLOAT, APPLE9_COMPARE_LESS};
      break;
   case nir_op_fge:
      *compare = (struct apple9_compare){APPLE9_COMPARE_FLOAT,
                                         APPLE9_COMPARE_GREATER_EQUAL};
      break;
   case nir_op_ieq:
      *compare =
         (struct apple9_compare){APPLE9_COMPARE_INTEGER, APPLE9_COMPARE_EQUAL};
      break;
   case nir_op_ine:
      *compare = (struct apple9_compare){APPLE9_COMPARE_INTEGER,
                                         APPLE9_COMPARE_NOT_EQUAL};
      break;
   case nir_op_ilt:
      *compare =
         (struct apple9_compare){APPLE9_COMPARE_SIGNED, APPLE9_COMPARE_LESS};
      break;
   case nir_op_ige:
      *compare = (struct apple9_compare){APPLE9_COMPARE_SIGNED,
                                         APPLE9_COMPARE_GREATER_EQUAL};
      break;
   case nir_op_ult:
      *compare =
         (struct apple9_compare){APPLE9_COMPARE_UNSIGNED, APPLE9_COMPARE_LESS};
      break;
   case nir_op_uge:
      *compare = (struct apple9_compare){APPLE9_COMPARE_UNSIGNED,
                                         APPLE9_COMPARE_GREATER_EQUAL};
      break;
   default:
      return false;
   }

   return true;
}

static bool
apple9_select_condition(const struct apple9_compare *compare,
                        uint32_t *immediate)
{
   if (compare->domain == APPLE9_COMPARE_FLOAT) {
      switch (compare->relation) {
      case APPLE9_COMPARE_EQUAL:
      case APPLE9_COMPARE_NOT_EQUAL:
         *immediate = AGX_APPLE9_SELECT_FEQ | AGX_APPLE9_SELECT_EQUALITY;
         return true;
      case APPLE9_COMPARE_LESS:
         *immediate = AGX_APPLE9_SELECT_FLT;
         return true;
      case APPLE9_COMPARE_GREATER_EQUAL:
         *immediate = AGX_APPLE9_SELECT_FGT | AGX_APPLE9_SELECT_EQUALITY;
         return true;
      }
   } else if (compare->domain == APPLE9_COMPARE_UNSIGNED &&
              compare->relation == APPLE9_COMPARE_LESS) {
      *immediate = AGX_APPLE9_SELECT_ULT;
      return true;
   }

   return false;
}

static uint32_t
apple9_emit_dag_select_raw(struct apple9_dag_lower *lower, uint32_t cmp_a,
                           uint32_t cmp_b, uint32_t if_true, uint32_t if_false,
                           uint32_t immediate)
{
   uint32_t sources[4] = {cmp_a, cmp_b, if_true, if_false};
   for (unsigned i = 0; i < ARRAY_SIZE(sources); ++i) {
      if (sources[i] == AGX_APPLE9_VREG_INVALID)
         return AGX_APPLE9_VREG_INVALID;
   }

   return apple9_dag_emit_constrained(lower, AGX_APPLE9_VIR_SELECT,
                                      AGX_APPLE9_ENC_SELECT_GPR_WIDE, sources,
                                      ARRAY_SIZE(sources), immediate);
}

static uint32_t
apple9_emit_dag_select(struct apple9_dag_lower *lower, nir_scalar predicate,
                       uint32_t if_true, uint32_t if_false)
{
   predicate = apple9_chase_trivial(predicate);
   if (nir_def_instr_type(predicate.def) != nir_instr_type_alu) {
      lower->reason = "Apple9 DAG select requires a supported comparison";
      return AGX_APPLE9_VREG_INVALID;
   }

   struct apple9_compare compare;
   if (!apple9_normalize_compare(nir_scalar_alu_op(predicate), &compare)) {
      lower->reason = "Apple9 DAG select requires a supported comparison";
      return AGX_APPLE9_VREG_INVALID;
   }

   uint32_t cmp_a = apple9_lower_dag_source(lower, predicate, 0);
   uint32_t cmp_b = apple9_lower_dag_source(lower, predicate, 1);
   if (cmp_a == AGX_APPLE9_VREG_INVALID || cmp_b == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;

   /* Flipping the sign bit maps two's-complement order to unsigned order. */
   if (compare.domain == APPLE9_COMPARE_SIGNED) {
      uint32_t sign_a = apple9_dag_imm(lower, 0x80000000u);
      uint32_t sign_b = apple9_dag_imm(lower, 0x80000000u);
      uint32_t biased_a_sources[2] = {cmp_a, sign_a};
      uint32_t biased_b_sources[2] = {cmp_b, sign_b};
      uint32_t biased_a = apple9_dag_emit(
         lower, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED,
         biased_a_sources, ARRAY_SIZE(biased_a_sources), 0);
      uint32_t biased_b = apple9_dag_emit(
         lower, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED,
         biased_b_sources, ARRAY_SIZE(biased_b_sources), 0);
      return apple9_emit_dag_select_raw(
         lower, biased_a, biased_b,
         compare.relation == APPLE9_COMPARE_GREATER_EQUAL ? if_false : if_true,
         compare.relation == APPLE9_COMPARE_GREATER_EQUAL ? if_true : if_false,
         AGX_APPLE9_SELECT_ULT);
   }

   if (compare.domain == APPLE9_COMPARE_UNSIGNED &&
       compare.relation == APPLE9_COMPARE_GREATER_EQUAL) {
      return apple9_emit_dag_select_raw(lower, cmp_a, cmp_b, if_false, if_true,
                                        AGX_APPLE9_SELECT_ULT);
   }

   if (compare.domain == APPLE9_COMPARE_INTEGER) {
      bool not_equal = compare.relation == APPLE9_COMPARE_NOT_EQUAL;
      uint32_t xor_sources[2] = {cmp_a, cmp_b};
      uint32_t difference = apple9_dag_emit(
         lower, AGX_APPLE9_VIR_IXOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, xor_sources,
         ARRAY_SIZE(xor_sources), 0);
      uint32_t one = apple9_dag_imm(lower, 1);
      return apple9_emit_dag_select_raw(
         lower, difference, one, not_equal ? if_false : if_true,
         not_equal ? if_true : if_false, AGX_APPLE9_SELECT_ULT);
   }

   uint32_t immediate;
   if (!apple9_select_condition(&compare, &immediate)) {
      lower->reason = "Apple9 DAG select requires a supported comparison";
      return AGX_APPLE9_VREG_INVALID;
   }

   if (compare.domain == APPLE9_COMPARE_FLOAT &&
       compare.relation == APPLE9_COMPARE_NOT_EQUAL) {
      uint32_t temporary = if_true;
      if_true = if_false;
      if_false = temporary;
   }

   return apple9_emit_dag_select_raw(lower, cmp_a, cmp_b, if_true, if_false,
                                     immediate);
}

/* Keep ordinary Boolean SSA distinct from the transient predicate/mask state.
 * Metal follows the same split: comparisons may feed control directly, while
 * Boolean values that fan out or are combined are materialized as 0/1 GPRs.
 * This routine handles only pure ALU DAGs. Side-effecting short-circuit
 * expressions remain structured NIR control flow and are never speculated. */
static uint32_t
apple9_lower_bool_scalar(struct apple9_dag_lower *lower, nir_scalar scalar)
{
   scalar = apple9_chase_trivial(scalar);
   if (scalar.def->bit_size != 1 || scalar.comp >= 4) {
      lower->reason = "Apple9 Boolean lowering requires a scalar i1 value";
      return AGX_APPLE9_VREG_INVALID;
   }

   const unsigned key = scalar.def->index * 4 + scalar.comp;
   if (key >= lower->ssa_map_count) {
      lower->reason =
         "Apple9 Boolean lowering encountered an invalid SSA index";
      return AGX_APPLE9_VREG_INVALID;
   }
   if (lower->ssa_to_vreg[key] != AGX_APPLE9_VREG_INVALID)
      return lower->ssa_to_vreg[key];

   uint32_t value = AGX_APPLE9_VREG_INVALID;
   if (nir_scalar_is_const(scalar)) {
      value = apple9_dag_imm(lower, nir_scalar_as_uint(scalar) != 0);
   } else if (nir_def_instr_type(scalar.def) == nir_instr_type_alu) {
      const nir_op op = nir_scalar_alu_op(scalar);
      struct apple9_compare compare;
      if (apple9_normalize_compare(op, &compare)) {
         uint32_t one = apple9_dag_imm(lower, 1);
         uint32_t zero = apple9_dag_zero(lower);
         if (one != AGX_APPLE9_VREG_INVALID && zero != AGX_APPLE9_VREG_INVALID)
            value = apple9_emit_dag_select(lower, scalar, one, zero);
      } else {
         switch (op) {
         case nir_op_b2b1: {
            /* Uniform Boolean storage is 32 bits; Boolean SSA uses 0/1. */
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            uint32_t one = apple9_dag_imm(lower, 1);
            uint32_t zero = apple9_dag_zero(lower);
            if (source != AGX_APPLE9_VREG_INVALID &&
                one != AGX_APPLE9_VREG_INVALID && zero != AGX_APPLE9_VREG_INVALID)
               value = apple9_emit_dag_select_raw(lower, zero, source, one, zero,
                                                   AGX_APPLE9_SELECT_ULT);
            break;
         }
         case nir_op_iand:
         case nir_op_ior:
         case nir_op_ixor: {
            uint32_t sources[2] = {
               apple9_lower_bool_scalar(lower,
                                        nir_scalar_chase_alu_src(scalar, 0)),
               apple9_lower_bool_scalar(lower,
                                        nir_scalar_chase_alu_src(scalar, 1)),
            };
            if (sources[0] != AGX_APPLE9_VREG_INVALID &&
                sources[1] != AGX_APPLE9_VREG_INVALID) {
               value = apple9_dag_emit(lower, apple9_dag_binary_opcode(op),
                                       AGX_APPLE9_ENC_LOGIC_EXTENDED, sources,
                                       ARRAY_SIZE(sources), 0);
            }
            break;
         }
         case nir_op_inot: {
            uint32_t source = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 0));
            uint32_t one = apple9_dag_imm(lower, 1);
            uint32_t sources[2] = {source, one};
            if (source != AGX_APPLE9_VREG_INVALID &&
                one != AGX_APPLE9_VREG_INVALID)
               value = apple9_dag_emit(lower, AGX_APPLE9_VIR_IXOR,
                                       AGX_APPLE9_ENC_LOGIC_EXTENDED, sources,
                                       ARRAY_SIZE(sources), 0);
            break;
         }
         case nir_op_bcsel: {
            uint32_t condition = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 0));
            uint32_t if_true = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 1));
            uint32_t if_false = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 2));
            uint32_t zero = apple9_dag_zero(lower);
            value = apple9_emit_dag_select_raw(lower, zero, condition, if_true,
                                               if_false, AGX_APPLE9_SELECT_ULT);
            break;
         }
         default:
            break;
         }
      }
   }

   if (value == AGX_APPLE9_VREG_INVALID && lower->reason == NULL)
      lower->reason =
         "Apple9 Boolean lowering encountered an unsupported value";
   if (value != AGX_APPLE9_VREG_INVALID)
      lower->ssa_to_vreg[key] = value;
   return value;
}

static uint32_t
apple9_dag_shift_imm(struct apple9_dag_lower *lower, nir_op op, uint32_t source,
                     unsigned amount)
{
   if (source == AGX_APPLE9_VREG_INVALID || amount >= 32)
      return AGX_APPLE9_VREG_INVALID;
   if (amount == 0)
      return source;

   if (op == nir_op_ishl) {
      uint32_t scale = apple9_dag_imm(lower, 1u << amount);
      uint32_t zero = apple9_dag_zero(lower);
      uint32_t sources[3] = {source, scale, zero};
      if (scale == AGX_APPLE9_VREG_INVALID || zero == AGX_APPLE9_VREG_INVALID)
         return AGX_APPLE9_VREG_INVALID;
      return apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
                             AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources,
                             ARRAY_SIZE(sources), 0);
   }

   /* The immediate ASHR fields are executed only in the proven compact bank.
    * A longer variable-shift graph exposed that the earlier byte-diff-derived
    * r16-r63 interpretation reads zero on T8132.  The shift's machine
    * constraints and source-class propagation place both values in an
    * arbitrary r0-r15 pair; copy the result back to the general bank. */
   const uint32_t copy_sources[2] = {source, source};
   uint32_t compact_source =
      apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED,
                      copy_sources, ARRAY_SIZE(copy_sources), 0);
   if (compact_source == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;
   uint32_t shifted =
      apple9_dag_emit(lower, AGX_APPLE9_VIR_ISHR, AGX_APPLE9_ENC_SHIFT_EXTENDED,
                      &compact_source, 1, amount);
   if (shifted == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;
   shifted = apple9_dag_general_copy(lower, shifted);
   if (op != nir_op_ushr || shifted == AGX_APPLE9_VREG_INVALID)
      return shifted;

   uint32_t mask = apple9_dag_imm(lower, UINT32_MAX >> amount);
   uint32_t sources[2] = {shifted, mask};
   return mask == AGX_APPLE9_VREG_INVALID
             ? AGX_APPLE9_VREG_INVALID
             : apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
                               AGX_APPLE9_ENC_LOGIC_EXTENDED, sources,
                               ARRAY_SIZE(sources), 0);
}

static uint32_t
apple9_dag_shift_variable(struct apple9_dag_lower *lower, nir_op op,
                          uint32_t source, uint32_t amount)
{
   if (source == AGX_APPLE9_VREG_INVALID || amount == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;

   uint32_t result = source;
   uint32_t zero = apple9_dag_zero(lower);
   if (zero == AGX_APPLE9_VREG_INVALID)
      return zero;

   /* NIR shifts consume the low five amount bits.  A five-stage conditional
    * barrel shift is a correctness lowering built entirely from validated
    * Apple9 operations; the incompletely decoded native register-shift form
    * can replace it later as an instruction-selection optimization. */
   for (unsigned bit = 0; bit < 5; ++bit) {
      uint32_t mask = apple9_dag_imm(lower, BITFIELD_BIT(bit));
      uint32_t condition_sources[2] = {amount, mask};
      uint32_t condition =
         mask == AGX_APPLE9_VREG_INVALID
            ? AGX_APPLE9_VREG_INVALID
            : apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
                              AGX_APPLE9_ENC_LOGIC_EXTENDED, condition_sources,
                              ARRAY_SIZE(condition_sources), 0);
      uint32_t shifted =
         apple9_dag_shift_imm(lower, op, result, BITFIELD_BIT(bit));
      if (condition == AGX_APPLE9_VREG_INVALID ||
          shifted == AGX_APPLE9_VREG_INVALID)
         return AGX_APPLE9_VREG_INVALID;
      result = apple9_emit_dag_select_raw(lower, zero, condition, shifted,
                                          result, AGX_APPLE9_SELECT_ULT);
      if (result == AGX_APPLE9_VREG_INVALID)
         return result;
   }

   return result;
}

/* Returns the packed user scalar index, excluding the four position words. */
static int
apple9_varying_index(const struct agx_apple9_varying_layout *layout,
                     unsigned location, unsigned component)
{
   if (!layout || location < VARYING_SLOT_VAR0 ||
       location >= VARYING_SLOT_VAR0 + 32 || component >= 4)
      return -1;
   unsigned semantic = location - VARYING_SLOT_VAR0;
   if (!(layout->mask[semantic] & BITFIELD_BIT(component)))
      return -1;
   unsigned index = util_bitcount(layout->mask[semantic] & BITFIELD_MASK(component));
   for (unsigned i = 0; i < semantic; ++i)
      index += util_bitcount(layout->mask[i]);
   return index;
}

static uint32_t
apple9_lower_interpolated_input(struct apple9_dag_lower *lower,
                                nir_scalar scalar)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(scalar.def);
   nir_intrinsic_instr *bary = nir_src_as_intrinsic(intr->src[0]);
   const unsigned component = nir_intrinsic_component(intr) + scalar.comp;
   if (lower->nir->info.stage != MESA_SHADER_FRAGMENT ||
       scalar.def->bit_size != 32 || component >= 4 ||
       !nir_src_is_const(intr->src[1]) ||
       !bary || bary->intrinsic != nir_intrinsic_load_barycentric_pixel ||
       nir_intrinsic_interp_mode(bary) != INTERP_MODE_SMOOTH) {
      lower->reason = "Apple9 fragment input requires center-smooth FP32 user varyings";
      return AGX_APPLE9_VREG_INVALID;
   }
   unsigned location = nir_intrinsic_io_semantics(intr).location +
                       nir_src_as_uint(intr->src[1]);
   int index = apple9_varying_index(lower->varyings, location, component);
   if (index < 0) {
      lower->reason = "Apple9 fragment input is not written by the vertex stage";
      return AGX_APPLE9_VREG_INVALID;
   }
   if (!lower->perspective_ready) {
      uint32_t denominator = apple9_dag_emit(
         lower, AGX_APPLE9_VIR_ITER, AGX_APPLE9_ENC_ITER, NULL, 0, 0);
      lower->perspective_reciprocal =
         apple9_dag_emit(lower, AGX_APPLE9_VIR_FRCP,
                         AGX_APPLE9_ENC_FLOAT_SPECIAL, &denominator, 1, 3);
      lower->perspective_ready = true;
   }
   uint32_t coefficient = apple9_dag_emit(
      lower, AGX_APPLE9_VIR_ITER, AGX_APPLE9_ENC_ITER, NULL, 0, index + 1);
   /* Keep raw varyings available to homogeneous clipping. Native shade-7
    * coefficients pair with a coefficient-aware projective multiply, which
    * handles the rasterizer's primitive-constant representation. */
   uint32_t src[] = {coefficient, lower->perspective_reciprocal};
   return apple9_dag_emit(lower, AGX_APPLE9_VIR_FMUL_PROJECT,
                          AGX_APPLE9_ENC_FLOAT2_PROJECT, src, 2, index + 1);
}

static uint32_t
apple9_lower_dag_scalar(struct apple9_dag_lower *lower, nir_scalar scalar)
{
   scalar = apple9_chase_trivial(scalar);
   if (scalar.def->bit_size == 1)
      return apple9_lower_bool_scalar(lower, scalar);
   if ((scalar.def->bit_size != 8 && scalar.def->bit_size != 16 &&
        scalar.def->bit_size != 32) ||
       scalar.comp >= 4) {
      lower->reason =
         "Apple9 DAG compiler supports 8-, 16- and 32-bit scalar components";
      return AGX_APPLE9_VREG_INVALID;
   }

   unsigned key = scalar.def->index * 4 + scalar.comp;
   if (key >= lower->ssa_map_count) {
      lower->reason = "Apple9 DAG compiler encountered an invalid SSA index";
      return AGX_APPLE9_VREG_INVALID;
   }
   if (lower->ssa_to_vreg[key] != AGX_APPLE9_VREG_INVALID)
      return lower->ssa_to_vreg[key];

   uint32_t value = AGX_APPLE9_VREG_INVALID;
   uint32_t constant;
   if (nir_scalar_is_const(scalar)) {
      constant = nir_scalar_as_uint(scalar);
      if (scalar.def->bit_size < 32)
         constant &= BITFIELD_MASK(scalar.def->bit_size);
      value = apple9_dag_imm(lower, constant);
   } else {
      struct apple9_system_source system;
      const bool subgroup_size =
         nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
         nir_def_as_intrinsic(scalar.def)->intrinsic ==
            nir_intrinsic_load_subgroup_size;
      if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
          nir_def_as_intrinsic(scalar.def)->intrinsic ==
             nir_intrinsic_load_interpolated_input) {
         value = apple9_lower_interpolated_input(lower, scalar);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_output) {
         nir_intrinsic_instr *intr = nir_def_as_intrinsic(scalar.def);
         if (lower->nir->info.stage != MESA_SHADER_FRAGMENT || scalar.comp ||
             intr->num_components != 1 || intr->def.bit_size != 32 ||
             nir_intrinsic_dest_type(intr) != nir_type_uint32 ||
             !nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) != 0 ||
             nir_intrinsic_io_semantics(intr).location != FRAG_RESULT_DATA0 ||
             lower->tile_read)
            return AGX_APPLE9_VREG_INVALID;
         lower->tile_read = true;
         if ((!lower->tile_access && !agx_apple9_vir_emit_side_effect(&lower->program,
                AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0, 0x600)) ||
             !agx_apple9_vir_emit_side_effect(&lower->program,
                AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0, 0x808))
            return AGX_APPLE9_VREG_INVALID;
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_TILE_LOAD,
                                AGX_APPLE9_ENC_TILE_LOAD, NULL, 0, 0);
         if (value != AGX_APPLE9_VREG_INVALID)
            lower->program.instructions[lower->program.instruction_count - 1].producer_scoreboard_slot =
               AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
      } else if (subgroup_size) {
         /* Native Metal materializes the architectural SIMD width. */
         value = apple9_dag_imm(lower, 32);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic ==
                    nir_intrinsic_load_num_workgroups) {
         value = apple9_dag_num_workgroups(lower, scalar.comp);
      } else if (apple9_system_source(scalar, &system)) {
         value = apple9_dag_system(lower, system);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 (nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_ssbo ||
                  nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_ubo)) {
         struct apple9_scalar_load *load = NULL;
         for (unsigned i = 0; i < lower->load_count; ++i) {
            if (&lower->loads[i].intr->def == scalar.def &&
                lower->loads[i].component == scalar.comp) {
               load = &lower->loads[i];
               break;
            }
         }
         if (load == NULL) {
            lower->reason =
               "Apple9 DAG load is absent from the resource ledger";
            return AGX_APPLE9_VREG_INVALID;
         }
         if (load->block != lower->active_load_block) {
            lower->reason =
               "Apple9 DAG load escaped its active control-flow region";
            return AGX_APPLE9_VREG_INVALID;
         }

         uint32_t index = load->index_scale == 0
                             ? apple9_dag_zero(lower)
                             : apple9_lower_dag_scalar(lower, load->index);
         if (index == AGX_APPLE9_VREG_INVALID)
            return index;

         unsigned read_count = 0;
         for (unsigned i = 0; i < lower->load_count; ++i)
            read_count += lower->loads[i].intr == load->intr;

         /* Preserve a native NIR vector access whenever at least two lanes
          * survive.  The memory format supplies the std430 element stride
          * (2 dwords for vec2, 4 for vec3/vec4), while one scoreboard slot
          * covers the complete adjacent destination tuple.  Loading an
          * unused lane is preferable to splitting one semantic vector access
          * into independently scheduled scalar producers. */
         if (read_count >= 2) {
            const unsigned components = load->intr->def.num_components;
            const unsigned expected_stride = components == 2 ? 2 : 4;
            if (components < 2 || components > 4 ||
                load->index_scale != expected_stride || load->index_add != 0) {
               lower->reason =
                  "Apple9 native vector load requires a std430 vector stride";
               return AGX_APPLE9_VREG_INVALID;
            }

            uint8_t flags = apple9_current_load_flags(lower, load, true);
            const struct agx_apple9_device_load_contract contract = {
               .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
               .flags = flags,
               .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
            };
            const uint32_t base = agx_apple9_vir_emit_device_load_vector(
               &lower->program,
               lower->argument_base + load->argument, index,
               components, &contract);
            if (base == AGX_APPLE9_VREG_INVALID ||
                !agx_apple9_vir_set_device_load_contract(
                   &lower->program, base, flags,
                   AGX_APPLE9_SCOREBOARD_SLOT_AUTO)) {
               lower->reason =
                  "could not describe an Apple9 native vector load";
               return AGX_APPLE9_VREG_INVALID;
            }

            for (unsigned c = 0; c < components; ++c) {
               const unsigned lane_key = scalar.def->index * 4 + c;
               lower->ssa_to_vreg[lane_key] = base + c;
            }
            ++lower->active_emitted_load_count;
            ++lower->emitted_load_count;
            return lower->ssa_to_vreg[key];
         }

         if (load->index_scale > 1) {
            uint32_t scale = apple9_dag_imm(lower, load->index_scale);
            uint32_t zero = apple9_dag_zero(lower);
            uint32_t sources[3] = {index, scale, zero};
            if (scale == AGX_APPLE9_VREG_INVALID ||
                zero == AGX_APPLE9_VREG_INVALID)
               return AGX_APPLE9_VREG_INVALID;
            index = apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
                                    AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources,
                                    ARRAY_SIZE(sources), 0);
            if (index == AGX_APPLE9_VREG_INVALID)
               return index;
         }

         const unsigned element_add = load->index_add + load->component;
         if (element_add != 0) {
            uint32_t component = apple9_dag_imm(lower, element_add);
            uint32_t sources[2] = {index, component};
            if (component == AGX_APPLE9_VREG_INVALID)
               return component;
            index = apple9_dag_emit(lower, AGX_APPLE9_VIR_IADD,
                                    AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources,
                                    ARRAY_SIZE(sources), 0);
            if (index == AGX_APPLE9_VREG_INVALID)
               return index;
         }

         const uint32_t source[] = {index};
         value = apple9_dag_emit(
            lower, AGX_APPLE9_VIR_DEVICE_LOAD, AGX_APPLE9_ENC_DEVICE_LOAD,
            source, 1,
            lower->argument_base + load->argument);
         if (value != AGX_APPLE9_VREG_INVALID)
            lower->program.instructions[lower->program.instruction_count - 1]
               .memory_bits = load->bit_size;
         uint8_t flags = apple9_current_load_flags(lower, load, false);
         if (value == AGX_APPLE9_VREG_INVALID ||
             !agx_apple9_vir_set_device_load_contract(
                &lower->program, value, flags,
                AGX_APPLE9_SCOREBOARD_SLOT_AUTO)) {
            lower->reason = lower->reason != NULL
                               ? lower->reason
                               : "could not describe an Apple9 device load";
            return AGX_APPLE9_VREG_INVALID;
         }
         ++lower->active_emitted_load_count;
         ++lower->emitted_load_count;
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_alu) {
         nir_op op = nir_scalar_alu_op(scalar);

         if (op == nir_op_i2i8 || op == nir_op_i2i16 || op == nir_op_i2i32 ||
             op == nir_op_u2u8 || op == nir_op_u2u16 || op == nir_op_u2u32) {
            nir_scalar source_scalar =
               apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 0));
            const unsigned source_bits = source_scalar.def->bit_size;
            const unsigned destination_bits = scalar.def->bit_size;
            uint32_t source = apple9_lower_dag_scalar(lower, source_scalar);
            if (source == AGX_APPLE9_VREG_INVALID)
               return source;

            if (destination_bits <= source_bits) {
               /* Narrowing is a semantic truncation.  Keep the low bits in
                * the same SSA value; a narrow store consumes exactly those
                * bits, while any later widening emits its own extension. */
               value = source;
            } else if (op == nir_op_u2u8 || op == nir_op_u2u16 ||
                       op == nir_op_u2u32) {
               uint32_t mask = apple9_dag_imm(
                  lower,
                  source_bits == 32 ? UINT32_MAX : BITFIELD_MASK(source_bits));
               uint32_t sources[2] = {source, mask};
               if (mask != AGX_APPLE9_VREG_INVALID)
                  value = apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
                                          AGX_APPLE9_ENC_LOGIC_EXTENDED,
                                          sources, ARRAY_SIZE(sources), 0);
            } else {
               const unsigned shift = 32 - source_bits;
               uint32_t scale = apple9_dag_imm(lower, 1u << shift);
               uint32_t zero = apple9_dag_zero(lower);
               uint32_t sources[3] = {source, scale, zero};
               uint32_t shifted =
                  scale == AGX_APPLE9_VREG_INVALID ||
                        zero == AGX_APPLE9_VREG_INVALID
                     ? AGX_APPLE9_VREG_INVALID
                     : apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
                                       AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources,
                                       ARRAY_SIZE(sources), 0);
               if (shifted != AGX_APPLE9_VREG_INVALID)
                  value =
                     apple9_dag_shift_imm(lower, nir_op_ishr, shifted, shift);
            }
         } else if (op == nir_op_bcsel) {
            uint32_t if_true = apple9_lower_dag_source(lower, scalar, 1);
            uint32_t if_false = apple9_lower_dag_source(lower, scalar, 2);
            uint32_t condition = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 0));
            uint32_t zero = apple9_dag_zero(lower);
            if (if_true != AGX_APPLE9_VREG_INVALID &&
                if_false != AGX_APPLE9_VREG_INVALID &&
                condition != AGX_APPLE9_VREG_INVALID &&
                zero != AGX_APPLE9_VREG_INVALID)
               value =
                  apple9_emit_dag_select_raw(lower, zero, condition, if_true,
                                             if_false, AGX_APPLE9_SELECT_ULT);
         } else if (op == nir_op_b2i32) {
            value = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 0));
         } else if (op == nir_op_b2f32) {
            uint32_t boolean = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 0));
            if (boolean != AGX_APPLE9_VREG_INVALID)
               value =
                  apple9_dag_emit(lower, AGX_APPLE9_VIR_U2F32,
                                  AGX_APPLE9_ENC_UINT_TO_FLOAT, &boolean, 1, 0);
         } else if (op == nir_op_fsqrt) {
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source != AGX_APPLE9_VREG_INVALID) {
               uint32_t factor = apple9_dag_emit(
                  lower, AGX_APPLE9_VIR_FSQRT_FACTOR,
                  AGX_APPLE9_ENC_FLOAT_SPECIAL, &source, 1, 0x03);
               uint32_t sources[] = {source, factor};
               value = apple9_dag_emit(lower, AGX_APPLE9_VIR_FMUL,
                                      AGX_APPLE9_ENC_FLOAT2_COMPACT, sources, 2, 0);
            }
         } else if (op == nir_op_frcp || op == nir_op_frsq ||
                    op == nir_op_fsin_factor_agx || op == nir_op_fexp2 ||
                    op == nir_op_flog2 || op == nir_op_ffloor ||
                    op == nir_op_fceil || op == nir_op_ftrunc ||
                    op == nir_op_fround_even) {
            enum agx_apple9_vir_opcode special =
               op == nir_op_frcp ? AGX_APPLE9_VIR_FRCP :
               op == nir_op_frsq ? AGX_APPLE9_VIR_FRSQ :
               op == nir_op_fsin_factor_agx ? AGX_APPLE9_VIR_FSIN_FACTOR :
               op == nir_op_fexp2 ? AGX_APPLE9_VIR_FEXP2 :
               op == nir_op_flog2 ? AGX_APPLE9_VIR_FLOG2 :
               op == nir_op_ffloor ? AGX_APPLE9_VIR_FFLOOR :
               op == nir_op_fceil ? AGX_APPLE9_VIR_FCEIL :
               op == nir_op_ftrunc ? AGX_APPLE9_VIR_FTRUNC :
                                    AGX_APPLE9_VIR_FROUND_EVEN;
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source != AGX_APPLE9_VREG_INVALID)
               value = apple9_dag_emit(lower, special,
                                       AGX_APPLE9_ENC_FLOAT_SPECIAL, &source,
                                       1, 0x03);
         } else if (op == nir_op_u2f32 || op == nir_op_i2f32 ||
                    op == nir_op_f2i32 || op == nir_op_f2u32) {
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source != AGX_APPLE9_VREG_INVALID) {
               enum agx_apple9_vir_opcode vir_op =
                  op == nir_op_u2f32   ? AGX_APPLE9_VIR_U2F32
                  : op == nir_op_i2f32 ? AGX_APPLE9_VIR_I2F32
                  : op == nir_op_f2i32 ? AGX_APPLE9_VIR_F2I32
                                       : AGX_APPLE9_VIR_F2U32;
               enum agx_apple9_encoding encoding =
                  op == nir_op_u2f32   ? AGX_APPLE9_ENC_UINT_TO_FLOAT
                  : op == nir_op_i2f32 ? AGX_APPLE9_ENC_SINT_TO_FLOAT
                  : op == nir_op_f2i32 ? AGX_APPLE9_ENC_FLOAT_TO_SINT
                                       : AGX_APPLE9_ENC_FLOAT_TO_UINT;
               value = apple9_dag_emit(lower, vir_op, encoding, &source, 1, 0);
            }
         } else if (op == nir_op_inot || op == nir_op_fneg ||
                    op == nir_op_fabs || op == nir_op_ineg) {
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source == AGX_APPLE9_VREG_INVALID)
               return source;
            uint32_t immediate = op == nir_op_fneg   ? 0x80000000u
                                 : op == nir_op_fabs ? 0x7fffffffu
                                 : op == nir_op_inot ? UINT32_MAX
                                                     : 0;
            uint32_t other = apple9_dag_imm(lower, immediate);
            if (other == AGX_APPLE9_VREG_INVALID)
               return other;
            uint32_t sources[2];
            enum agx_apple9_vir_opcode vir_op;
            enum agx_apple9_encoding encoding;
            if (op == nir_op_ineg) {
               sources[0] = other;
               sources[1] = source;
               vir_op = AGX_APPLE9_VIR_ISUB;
               encoding = AGX_APPLE9_ENC_INT_ADD_EXTENDED;
            } else {
               sources[0] = source;
               sources[1] = other;
               vir_op =
                  op == nir_op_fabs ? AGX_APPLE9_VIR_IAND : AGX_APPLE9_VIR_IXOR;
               encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED;
            }
            value = apple9_dag_emit(lower, vir_op, encoding, sources, 2, 0);
         } else if (op == nir_op_imul || op == nir_op_amul) {
            uint32_t sources[3] = {
               apple9_lower_dag_source(lower, scalar, 0),
               apple9_lower_dag_source(lower, scalar, 1),
               apple9_dag_zero(lower),
            };
            if (sources[0] != AGX_APPLE9_VREG_INVALID &&
                sources[1] != AGX_APPLE9_VREG_INVALID &&
                sources[2] != AGX_APPLE9_VREG_INVALID)
               value = apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
                                       AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources,
                                       3, 0);
         } else if (op == nir_op_ishl) {
            nir_scalar shift =
               apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 1));
            uint32_t amount;
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            value = apple9_const_u32(shift, &amount) && amount < 32
                       ? apple9_dag_shift_imm(lower, op, source, amount)
                       : apple9_dag_shift_variable(
                            lower, op, source,
                            apple9_lower_dag_scalar(lower, shift));
         } else if (op == nir_op_ishr || op == nir_op_ushr) {
            nir_scalar shift =
               apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 1));
            uint32_t amount;
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            value = apple9_const_u32(shift, &amount) && amount < 32
                       ? apple9_dag_shift_imm(lower, op, source, amount)
                       : apple9_dag_shift_variable(
                            lower, op, source,
                            apple9_lower_dag_scalar(lower, shift));
         } else if (op == nir_op_ffma || op == nir_op_ffma_weak) {
            uint32_t sources[3] = {
               apple9_lower_dag_source(lower, scalar, 0),
               apple9_lower_dag_source(lower, scalar, 1),
               apple9_lower_dag_source(lower, scalar, 2),
            };
            if (sources[0] != AGX_APPLE9_VREG_INVALID &&
                sources[1] != AGX_APPLE9_VREG_INVALID &&
                sources[2] != AGX_APPLE9_VREG_INVALID)
               value = apple9_dag_emit(lower, AGX_APPLE9_VIR_FMA,
                                       AGX_APPLE9_ENC_FLOAT3_EXTENDED, sources,
                                       3, 0);
         } else {
            enum agx_apple9_vir_opcode vir_op = apple9_dag_binary_opcode(op);
            enum agx_apple9_encoding encoding = apple9_dag_binary_encoding(op);
            if (encoding != AGX_APPLE9_ENC_COUNT) {
               uint32_t sources[2] = {
                  apple9_lower_dag_source(lower, scalar, 0),
                  apple9_lower_dag_source(lower, scalar, 1),
               };
               if (sources[0] != AGX_APPLE9_VREG_INVALID &&
                   sources[1] != AGX_APPLE9_VREG_INVALID) {
                  const struct agx_apple9_operand_constraint *destination =
                     agx_apple9_find_operand(encoding, AGX_APPLE9_OPERAND_DEST);
                  value = destination != NULL && (destination->flags &
                                                  AGX_APPLE9_OPERAND_HARD_LOW)
                             ? apple9_dag_emit_constrained(
                                  lower, vir_op, encoding, sources, 2, 0)
                             : apple9_dag_emit(lower, vir_op, encoding, sources,
                                               2, 0);
               }
            }
         }
      }
   }

   if (value == AGX_APPLE9_VREG_INVALID && lower->reason == NULL)
      lower->reason =
         "Apple9 DAG compiler encountered an unsupported scalar operation";
   if (value != AGX_APPLE9_VREG_INVALID)
      lower->ssa_to_vreg[key] = value;
   return value;
}

static bool
apple9_emit_packed(struct apple9_emitter *emitter,
                   const struct agx_apple9_packed_instruction *packed)
{
   if (packed->length == 0 || packed->length > sizeof(packed->bytes))
      return false;
   apple9_emit(emitter, packed->length, packed->bytes);
   return true;
}

struct apple9_buffer_resource {
   enum agx_apple9_compute_resource_kind kind;
   uint8_t binding;
   bool read;
   bool write;
};

struct apple9_buffer_map {
   unsigned count;
   uint8_t read_mask;
   uint8_t write_mask;
   struct apple9_buffer_resource resource[AGX_APPLE9_COMPUTE_MAX_RESOURCES];
};

static int
apple9_compare_buffer_resource(const void *a_, const void *b_)
{
   const struct apple9_buffer_resource *a = a_;
   const struct apple9_buffer_resource *b = b_;
   /* Preserve the original ABI ordering: read-only inputs first, writable
    * resources last.  Existing one-output programs therefore remain
    * byte-identical, while multiple outputs receive deterministic arguments. */
   if (a->write != b->write)
      return (int)a->write - (int)b->write;
   if (a->kind != b->kind)
      return (int)a->kind - (int)b->kind;
   return (int)b->binding - (int)a->binding;
}

static struct apple9_buffer_resource *
apple9_find_buffer_resource(struct apple9_buffer_map *map,
                            enum agx_apple9_compute_resource_kind kind,
                            uint32_t binding)
{
   for (unsigned i = 0; i < map->count; ++i) {
      if (map->resource[i].kind == kind && map->resource[i].binding == binding)
         return &map->resource[i];
   }
   return NULL;
}

static bool
apple9_is_ssbo_atomic(nir_intrinsic_op op)
{
   return op == nir_intrinsic_ssbo_atomic ||
          op == nir_intrinsic_ssbo_atomic_swap;
}

/* Collect semantic API bindings before selecting a package.  The native
 * launch programs publish a compact pointer table; read/write ownership is
 * host-side scheduling state, not a distinct pointer encoding. */
static bool
apple9_collect_buffer_map(nir_shader *nir, struct apple9_buffer_map *map,
                          const char **reason)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_ssbo &&
             intr->intrinsic != nir_intrinsic_load_ubo &&
             intr->intrinsic != nir_intrinsic_store_ssbo &&
             !apple9_is_ssbo_atomic(intr->intrinsic))
            continue;

         const bool store = intr->intrinsic == nir_intrinsic_store_ssbo;
         const bool atomic = apple9_is_ssbo_atomic(intr->intrinsic);
         nir_def *binding_def = store ? intr->src[1].ssa : intr->src[0].ssa;
         uint32_t binding;
         if (!apple9_const_u32(nir_get_scalar(binding_def, 0), &binding) ||
             binding > UINT8_MAX) {
            *reason = "Apple9 buffer compiler requires constant bindings";
            return false;
         }

         enum agx_apple9_compute_resource_kind kind =
            !store && intr->intrinsic == nir_intrinsic_load_ubo
               ? AGX_APPLE9_COMPUTE_RESOURCE_UBO
               : AGX_APPLE9_COMPUTE_RESOURCE_SSBO;
         struct apple9_buffer_resource *resource =
            apple9_find_buffer_resource(map, kind, binding);
         if (resource == NULL) {
            if (map->count == ARRAY_SIZE(map->resource)) {
               *reason =
                  "Apple9 compiler supports up to eight buffer resources";
               return false;
            }
            resource = &map->resource[map->count++];
            *resource = (struct apple9_buffer_resource){
               .kind = kind,
               .binding = binding,
            };
         }
         resource->read |= !store;
         resource->write |= store || atomic;
      }
   }

   bool has_store = false;
   for (unsigned i = 0; i < map->count; ++i)
      has_store |= map->resource[i].write;
   if (!has_store && nir->info.stage == MESA_SHADER_COMPUTE) {
      *reason = "Apple9 buffer compiler requires at least one SSBO store";
      return false;
   }

   qsort(map->resource, map->count, sizeof(map->resource[0]),
         apple9_compare_buffer_resource);
   for (unsigned i = 0; i < map->count; ++i) {
      if (map->resource[i].read)
         map->read_mask |= BITFIELD_BIT(i);
      if (map->resource[i].write)
         map->write_mask |= BITFIELD_BIT(i);
   }
   return true;
}

static unsigned
apple9_buffer_argument(const struct apple9_buffer_map *map,
                       enum agx_apple9_compute_resource_kind kind,
                       uint32_t binding)
{
   for (unsigned i = 0; i < map->count; ++i) {
      if (map->resource[i].kind == kind && map->resource[i].binding == binding)
         return i;
   }
   return UINT_MAX;
}

static bool
apple9_atomic_op(nir_atomic_op op, enum agx_apple9_atomic_op *out)
{
   switch (op) {
   case nir_atomic_op_iadd:
      *out = AGX_APPLE9_ATOMIC_ADD;
      return true;
   case nir_atomic_op_isub:
      *out = AGX_APPLE9_ATOMIC_SUB;
      return true;
   case nir_atomic_op_imin:
      *out = AGX_APPLE9_ATOMIC_SMIN;
      return true;
   case nir_atomic_op_umin:
      *out = AGX_APPLE9_ATOMIC_UMIN;
      return true;
   case nir_atomic_op_imax:
      *out = AGX_APPLE9_ATOMIC_SMAX;
      return true;
   case nir_atomic_op_umax:
      *out = AGX_APPLE9_ATOMIC_UMAX;
      return true;
   case nir_atomic_op_iand:
      *out = AGX_APPLE9_ATOMIC_AND;
      return true;
   case nir_atomic_op_ior:
      *out = AGX_APPLE9_ATOMIC_OR;
      return true;
   case nir_atomic_op_ixor:
      *out = AGX_APPLE9_ATOMIC_XOR;
      return true;
   case nir_atomic_op_xchg:
      *out = AGX_APPLE9_ATOMIC_XCHG;
      return true;
   case nir_atomic_op_fadd:
      *out = AGX_APPLE9_ATOMIC_FADD;
      return true;
   case nir_atomic_op_cmpxchg:
      *out = AGX_APPLE9_ATOMIC_CMPXCHG;
      return true;
   default:
      return false;
   }
}

/* Resource count selects package layout, never shader semantics. */
static bool
apple9_find_buffer_dag(nir_shader *nir, const struct apple9_buffer_map *map,
                       struct util_dynarray *loads,
                       struct util_dynarray *stores,
                       struct util_dynarray *atomics, const char **reason)
{
   bool uses_num_workgroups = false;

   if ((map->count < 1 && nir->info.stage == MESA_SHADER_COMPUTE) ||
       map->count > ARRAY_SIZE(map->resource)) {
      *reason = "Apple9 buffer compiler requires one to eight resources";
      return false;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!apple9_validate_cf_list(&impl->body, reason))
      return false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (!apple9_instruction_is_in_subset(
                instr, nir->info.stage != MESA_SHADER_COMPUTE)) {
            *reason = "Apple9 buffer compiler encountered unsupported NIR";
            return false;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         uses_num_workgroups |=
            intr->intrinsic == nir_intrinsic_load_num_workgroups;
         if (intr->intrinsic != nir_intrinsic_load_ssbo &&
             intr->intrinsic != nir_intrinsic_load_ubo &&
             intr->intrinsic != nir_intrinsic_store_ssbo &&
             !apple9_is_ssbo_atomic(intr->intrinsic))
            continue;
         const bool atomic = apple9_is_ssbo_atomic(intr->intrinsic);
         if (!atomic &&
             (nir_intrinsic_access(intr) & (ACCESS_COHERENT | ACCESS_VOLATILE))) {
            *reason = "Apple9 SSBO compiler rejects volatile access";
            return false;
         }

         const bool load = intr->intrinsic != nir_intrinsic_store_ssbo &&
                           !atomic;
         nir_def *offset = load || atomic ? intr->src[1].ssa : intr->src[2].ssa;
         nir_scalar index;
         unsigned index_scale, index_add;
         const unsigned bit_size = atomic ? intr->def.bit_size
                                   : load  ? intr->def.bit_size
                                           : intr->src[0].ssa->bit_size;
         if (!apple9_element_index(offset, &index, &index_scale, &index_add,
                                   bit_size / 8)) {
            *reason =
               "Apple9 buffer compiler requires naturally indexed scalar elements";
            return false;
         }

         nir_def *binding_def = load || atomic ? intr->src[0].ssa
                                               : intr->src[1].ssa;
         uint32_t binding;
         if (!apple9_const_u32(nir_get_scalar(binding_def, 0), &binding)) {
            *reason = "Apple9 buffer compiler requires constant bindings";
            return false;
         }

         if (atomic) {
            if (bit_size != 32 || intr->def.num_components != 1 ||
                intr->src[2].ssa->bit_size != 32 ||
                intr->src[2].ssa->num_components != 1 ||
                (intr->intrinsic == nir_intrinsic_ssbo_atomic_swap &&
                 (nir_intrinsic_atomic_op(intr) != nir_atomic_op_cmpxchg ||
                  intr->src[3].ssa->bit_size != 32 ||
                  intr->src[3].ssa->num_components != 1))) {
               *reason = "Apple9 supports scalar 32-bit SSBO atomics";
               return false;
            }

            enum agx_apple9_atomic_op op;
            if (!apple9_atomic_op(nir_intrinsic_atomic_op(intr), &op)) {
               *reason = "Apple9 encountered an unsupported SSBO atomic operation";
               return false;
            }

            unsigned argument = apple9_buffer_argument(
               map, AGX_APPLE9_COMPUTE_RESOURCE_SSBO, binding);
            if (argument == UINT_MAX) {
               *reason = "Apple9 atomic binding is absent from its resource map";
               return false;
            }

            struct apple9_buffer_atomic entry = {
               .intr = intr,
               .block = block,
               .index = index,
               .argument = argument,
               .index_scale = index_scale,
               .index_add = index_add,
               .op = op,
            };
            util_dynarray_append(atomics, entry);
         } else if (load) {
            if (intr->def.num_components == 0 || intr->def.num_components > 4 ||
                (intr->def.bit_size != 8 && intr->def.bit_size != 16 &&
                 intr->def.bit_size != 32) ||
                (intr->def.bit_size != 32 && intr->def.num_components != 1)) {
               *reason =
                  "Apple9 inputs must be scalar 8/16-bit or one-to-four-component 32-bit buffer loads";
               return false;
            }
            enum agx_apple9_compute_resource_kind kind =
               intr->intrinsic == nir_intrinsic_load_ubo
                  ? AGX_APPLE9_COMPUTE_RESOURCE_UBO
                  : AGX_APPLE9_COMPUTE_RESOURCE_SSBO;
            unsigned argument = apple9_buffer_argument(map, kind, binding);
            if (argument == UINT_MAX) {
               *reason = "Apple9 input binding is absent from its resource map";
               return false;
            }
            const nir_component_mask_t read_mask =
               nir_def_components_read(&intr->def);
            for (unsigned component = 0; component < intr->def.num_components;
                 ++component) {
               if (!(read_mask & BITFIELD_BIT(component)))
                  continue;

               if (index_add > UINT32_MAX - component) {
                  *reason = "Apple9 vector-load field offset exceeds 32 bits";
                  return false;
               }

               struct apple9_scalar_load load = {
                  .intr = intr,
                  .block = block,
                  .index = index,
                  .argument = argument,
                  .component = component,
                  .index_scale = index_scale,
                  .index_add = index_add,
                  .bit_size = intr->def.bit_size,
               };
               util_dynarray_append(loads, load);
            }
         } else {
            const unsigned components = intr->src[0].ssa->num_components;
            const unsigned expected_stride = components == 1   ? 1
                                             : components == 2 ? 2
                                                               : 4;
            if (components < 1 || components > 4 ||
                (components > 1 &&
                 (index_scale != expected_stride || index_add != 0))) {
               *reason =
                  "Apple9 store index must use its natural vector stride";
               return false;
            }
            if (nir_intrinsic_write_mask(intr) != BITFIELD_MASK(components) ||
                (bit_size != 8 && bit_size != 16 && bit_size != 32) ||
                (bit_size != 32 && components != 1)) {
               *reason = "Apple9 requires complete scalar or u32 tuple stores";
               return false;
            }
            unsigned argument = apple9_buffer_argument(
               map, AGX_APPLE9_COMPUTE_RESOURCE_SSBO, binding);
            if (argument == UINT_MAX) {
               *reason =
                  "Apple9 output binding is absent from its resource map";
               return false;
            }

            struct apple9_buffer_store store = {
               .intr = intr,
               .block = block,
               .index = index,
               .argument = argument,
               .components = components,
               .index_scale = index_scale,
               .index_add = index_add,
               .bit_size = bit_size,
               .lowered_index = AGX_APPLE9_VREG_INVALID,
            };
            util_dynarray_append(stores, store);
         }
      }
   }

   if (stores->size == 0 && atomics->size == 0 &&
       nir->info.stage == MESA_SHADER_COMPUTE) {
      *reason = "Apple9 requires at least one SSBO side effect";
      return false;
   }

   /* The own-source atomic package publishes its eight caller resources at
    * q0..q7 and has no hidden direct/indirect geometry tuple. The normal
    * superset carrier publishes that tuple at q0..q2. Until those two launch
    * contracts are unified, silently lowering load_num_workgroups would read
    * the first caller SSBO address as dispatch metadata. */
   if (atomics->size != 0 && uses_num_workgroups) {
      *reason =
         "Apple9 atomic package does not publish the num-workgroups metadata";
      return false;
   }

   return true;
}

static bool
apple9_emit_collect_vir(struct apple9_emitter *emitter,
                        const struct agx_apple9_vir_instr *instruction,
                        const uint8_t *phys, unsigned *emission_max_gpr,
                        const char **reason)
{
   const unsigned components = instruction->dest_components;
   if (instruction->op != AGX_APPLE9_VIR_COLLECT || components < 2 ||
       components > 4 || instruction->nr_srcs != components)
      goto invalid;

   for (unsigned c = 0; c < components; ++c) {
      const unsigned dst = phys[instruction->dest + c];
      const unsigned src = phys[instruction->src[c]];
      *emission_max_gpr = MAX2(*emission_max_gpr, MAX2(dst, src));
      if (dst == src)
         continue;

      /* Extended IOR(x, x) is the validated general-register bit copy used by
       * scoreboard materialization. COLLECT is a parallel-copy pseudo, but
       * the allocator permits only complete coalescing or a disjoint tuple,
       * so these lane copies cannot clobber a later source. */
      struct agx_apple9_vir_instr copy = {
         .op = AGX_APPLE9_VIR_IOR,
         .encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED,
         .dest = 0,
         .dest_components = 1,
         .src = {1, 1},
         .nr_srcs = 2,
         .live_after_mask =
            (instruction->live_after_mask & (1u << c)) ? 0x3 : 0,
      };
      const uint8_t copy_phys[] = {dst, src};
      struct agx_apple9_packed_instruction packed;
      if (!agx_apple9_pack_vir_instruction(&copy, copy_phys, &packed, reason) ||
          !apple9_emit_packed(emitter, &packed))
         goto invalid;
   }

   return true;

invalid:
   if (reason != NULL && *reason == NULL)
      *reason = "Apple9 COLLECT lowering failed";
   return false;
}

static bool
apple9_emit_masked_copy_vir(struct apple9_emitter *emitter,
                            const struct agx_apple9_vir_instr *instruction,
                            const uint8_t *phys, unsigned *emission_max_gpr,
                            const char **reason)
{
   if (instruction->op != AGX_APPLE9_VIR_MASKED_COPY ||
       instruction->target == AGX_APPLE9_VREG_INVALID ||
       instruction->nr_srcs != 1)
      goto invalid;

   const unsigned dst = phys[instruction->target];
   const unsigned src = phys[instruction->src[0]];
   *emission_max_gpr = MAX2(*emission_max_gpr, MAX2(dst, src));

   /* Apple8 resolves CFG phis with predecessor-edge parallel copies after
    * allocation.  This bounded Apple9 slice uses the same architecture: MERGE
    * owns one virtual/physical destination, and each active arm conditionally
    * copies its source into it. Extended IOR(x, x) is our hardware-validated
    * general bit-copy form. */
   struct agx_apple9_vir_instr copy = {
      .op = AGX_APPLE9_VIR_IOR,
      .encoding = AGX_APPLE9_ENC_LOGIC_EXTENDED,
      .dest = 0,
      .dest_components = 1,
      .src = {1, 1},
      .target = AGX_APPLE9_VREG_INVALID,
      .nr_srcs = 2,
      .live_after_mask = (instruction->live_after_mask & 1u) ? 0x3 : 0,
      .scoreboard_slot = instruction->scoreboard_slot,
   };
   const uint8_t copy_phys[] = {dst, src};
   struct agx_apple9_packed_instruction packed;
   if (!agx_apple9_pack_vir_instruction(&copy, copy_phys, &packed, reason) ||
       !apple9_emit_packed(emitter, &packed))
      goto invalid;
   return true;

invalid:
   if (reason != NULL && *reason == NULL)
      *reason = "Apple9 masked phi-edge copy lowering failed";
   return false;
}

static bool
apple9_emit_device_store_vir(struct apple9_emitter *emitter,
                             const struct agx_apple9_vir_instr *instruction,
                             const uint8_t *phys, const char **reason)
{
   const unsigned components = instruction->memory_components;
   if (instruction->op != AGX_APPLE9_VIR_DEVICE_STORE || components < 1 ||
       components > 4 || instruction->nr_srcs != components + 1)
      goto invalid;

   struct agx_apple9_packed_instruction packed;
   unsigned data[4];
   for (unsigned c = 0; c < components; ++c)
      data[c] = phys[instruction->src[c]];
   const unsigned index = phys[instruction->src[components]];
   const bool release_index =
      !(instruction->live_after_mask & (1u << components));

   bool adjacent = true;
   for (unsigned c = 1; c < components; ++c)
      adjacent &= data[c] == data[0] + c;

   if (!adjacent)
      goto invalid;

   bool packed_ok =
      components == 1
         ? agx_apple9_pack_device_store_scalar(
              data[0], index, instruction->immediate, instruction->memory_bits,
              (enum agx_apple9_scoreboard_slot)instruction->scoreboard_slot,
              release_index, &packed)
         : agx_apple9_pack_device_store_vector_u32(
              data[0], index, instruction->immediate, components,
              (enum agx_apple9_scoreboard_slot)instruction->scoreboard_slot,
              release_index, &packed);
   if (!packed_ok || !apple9_emit_packed(emitter, &packed))
      goto invalid;
   return true;

invalid:
   if (reason != NULL && *reason == NULL)
      *reason = "Apple9 VIR device-store legalization failed";
   return false;
}

static void
apple9_infer_device_load_index_contracts(struct agx_apple9_vir_program *program)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *load = &program->instructions[i];
      if (load->op != AGX_APPLE9_VIR_DEVICE_LOAD || load->nr_srcs != 1)
         continue;

      uint32_t index = load->src[0];
      bool retained = false;
      for (unsigned live = 0; live < program->live_out_count; ++live)
         retained |= program->live_out[live] == index;
      for (unsigned j = i + 1; j < program->instruction_count; ++j) {
         for (unsigned source = 0; source < program->instructions[j].nr_srcs;
              ++source)
            retained |= program->instructions[j].src[source] == index;
      }
      load->device_load_index_kind =
         retained ? AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR
                  : AGX_APPLE9_DEVICE_LOAD_INDEX_LAST_USE_GPR;

      const struct agx_apple9_vir_instr *producer = NULL;
      bool earlier_load_consumer = false;
      for (unsigned j = 0; j < i; ++j) {
         const struct agx_apple9_vir_instr *candidate =
            &program->instructions[j];
         const unsigned components =
            candidate->dest_components ? candidate->dest_components : 1;
         if (index >= candidate->dest && index - candidate->dest < components)
            producer = candidate;
         if (candidate->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
             candidate->nr_srcs == 1 && candidate->src[0] == index)
            earlier_load_consumer = true;
      }
      load->device_load_index_first_consumer =
         producer != NULL && producer->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
         !earlier_load_consumer;
   }
}

static void
apple9_infer_special_result_hints(struct agx_apple9_vir_program *program)
{
   for (unsigned i = 0; i < program->instruction_count; ++i) {
      struct agx_apple9_vir_instr *special = &program->instructions[i];
      if (special->encoding != AGX_APPLE9_ENC_FLOAT_SPECIAL)
         continue;

      /* Native Metal selects 0x02 when the result goes directly to memory
       * and 0x03 when an ALU-family instruction consumes it. Hardware shows
       * bit 0 is output-inert in the measured low-pressure forms, but keep
       * the native distinction until its pressure behavior is understood. */
      special->immediate = 0x02;
      for (unsigned j = i + 1; j < program->instruction_count; ++j) {
         const struct agx_apple9_vir_instr *consumer =
            &program->instructions[j];
         bool reads_result = false;
         for (unsigned s = 0; s < consumer->nr_srcs; ++s)
            reads_result |= consumer->src[s] == special->dest;
         if (!reads_result)
            continue;

         if (consumer->op != AGX_APPLE9_VIR_DEVICE_STORE) {
            special->immediate = 0x03;
            break;
         }
      }
   }
}

static bool
apple9_lower_buffer_store_operands(struct apple9_dag_lower *lower,
                                   struct apple9_buffer_store *store)
{
   for (unsigned c = 0; c < ARRAY_SIZE(store->output); ++c)
      store->output[c] = AGX_APPLE9_VREG_INVALID;

   for (unsigned c = 0; c < store->components; ++c) {
      store->output[c] = apple9_lower_dag_scalar(
         lower,
         apple9_chase_trivial(nir_get_scalar(store->intr->src[0].ssa, c)));
      if (store->output[c] == AGX_APPLE9_VREG_INVALID)
         return false;
   }

   uint32_t index = store->index_scale == 0
                       ? apple9_dag_zero(lower)
                       : apple9_lower_dag_scalar(lower, store->index);
   if (index == AGX_APPLE9_VREG_INVALID)
      return false;

   /* Native vector stores scale their tuple index in the memory format.
    * Scalar stores instead consume a scalar-element index, so only they need
    * an explicit affine address calculation here. */
   if (store->components == 1 && store->index_scale > 1) {
      uint32_t scale = apple9_dag_imm(lower, store->index_scale);
      uint32_t zero = apple9_dag_zero(lower);
      uint32_t sources[3] = {index, scale, zero};
      if (scale == AGX_APPLE9_VREG_INVALID || zero == AGX_APPLE9_VREG_INVALID)
         return false;
      index = apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
                              AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources,
                              ARRAY_SIZE(sources), 0);
   }
   if (store->components == 1 && store->index_add != 0 &&
       index != AGX_APPLE9_VREG_INVALID) {
      uint32_t add = apple9_dag_imm(lower, store->index_add);
      uint32_t sources[2] = {index, add};
      index = add == AGX_APPLE9_VREG_INVALID
                 ? AGX_APPLE9_VREG_INVALID
                 : apple9_dag_emit(lower, AGX_APPLE9_VIR_IADD,
                                   AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources,
                                   ARRAY_SIZE(sources), 0);
   }

   store->lowered_index = index;
   return index != AGX_APPLE9_VREG_INVALID;
}

static bool
apple9_emit_buffer_store(struct apple9_dag_lower *lower,
                         const struct apple9_buffer_store *store)
{
   if (store->lowered_index == AGX_APPLE9_VREG_INVALID)
      return false;

   return agx_apple9_vir_emit_device_store(
      &lower->program,
      lower->argument_base + store->argument,
      store->lowered_index, store->output, store->components, store->bit_size);
}

static const struct agx_apple9_vir_instr *
apple9_dag_value_producer(const struct apple9_dag_lower *lower,
                          uint32_t value);

static bool
apple9_atomic_can_consume_pending_data(
   const struct apple9_dag_lower *lower,
   const struct apple9_buffer_atomic *atomic, nir_scalar scalar,
   uint32_t value, bool discard)
{
   /* EXP-M4-51 proves the scalar, returning form for every input/result slot
    * pair. Keep compare-exchange's two-value tuple and discarded results on
    * the conservative materialized path until those complete shapes receive
    * equivalent coverage. */
   if (discard || atomic->op == AGX_APPLE9_ATOMIC_CMPXCHG || scalar.comp != 0 ||
       scalar.def != atomic->intr->src[2].ssa ||
       !list_is_singular(&scalar.def->uses))
      return false;

   nir_src *only_use =
      list_first_entry(&scalar.def->uses, nir_src, use_link);
   if (nir_src_is_if(only_use) ||
       nir_src_use_instr(only_use) != &atomic->intr->instr)
      return false;

   const struct agx_apple9_vir_instr *producer =
      apple9_dag_value_producer(lower, value);
   return producer != NULL && producer->op == AGX_APPLE9_VIR_DEVICE_LOAD &&
          producer->dest_components == 1;
}

static bool
apple9_emit_buffer_atomic(struct apple9_dag_lower *lower,
                          const struct apple9_buffer_atomic *atomic)
{
   uint32_t index = atomic->index_scale == 0
                       ? apple9_dag_zero(lower)
                       : apple9_lower_dag_scalar(lower, atomic->index);
   if (index == AGX_APPLE9_VREG_INVALID)
      return false;

   if (atomic->index_scale > 1) {
      uint32_t scale = apple9_dag_imm(lower, atomic->index_scale);
      uint32_t zero = apple9_dag_zero(lower);
      uint32_t sources[3] = {index, scale, zero};
      if (scale == AGX_APPLE9_VREG_INVALID || zero == AGX_APPLE9_VREG_INVALID)
         return false;
      index = apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
                              AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources,
                              ARRAY_SIZE(sources), 0);
   }
   if (atomic->index_add != 0 && index != AGX_APPLE9_VREG_INVALID) {
      uint32_t add = apple9_dag_imm(lower, atomic->index_add);
      uint32_t sources[2] = {index, add};
      index = add == AGX_APPLE9_VREG_INVALID
                 ? AGX_APPLE9_VREG_INVALID
                 : apple9_dag_emit(lower, AGX_APPLE9_VIR_IADD,
                                   AGX_APPLE9_ENC_INT_ADD_EXTENDED, sources,
                                   ARRAY_SIZE(sources), 0);
   }
   if (index == AGX_APPLE9_VREG_INVALID)
      return false;

   /* The index is destructive and pending-index consumption is not yet
    * characterized. Always give it an ordinary private copy. */
   index = apple9_dag_general_copy(lower, index);
   if (index == AGX_APPLE9_VREG_INVALID)
      return false;

   const bool discard = nir_def_components_read(&atomic->intr->def) == 0;
   uint32_t data[2];
   unsigned data_components = 1;
   if (atomic->op == AGX_APPLE9_ATOMIC_CMPXCHG) {
      /* Native tuple order is desired, compare. NIR spells the sources as
       * compare, desired. */
      data[0] = apple9_lower_dag_scalar(
         lower, apple9_chase_trivial(nir_get_scalar(atomic->intr->src[3].ssa,
                                                     0)));
      data[1] = apple9_lower_dag_scalar(
         lower, apple9_chase_trivial(nir_get_scalar(atomic->intr->src[2].ssa,
                                                     0)));
      data_components = 2;
      for (unsigned c = 0; c < data_components; ++c) {
         data[c] = apple9_dag_general_copy(lower, data[c]);
         if (data[c] == AGX_APPLE9_VREG_INVALID)
            return false;
      }
   } else {
      nir_scalar scalar = apple9_chase_trivial(
         nir_get_scalar(atomic->intr->src[2].ssa, 0));
      data[0] = apple9_lower_dag_scalar(lower, scalar);
      if (data[0] == AGX_APPLE9_VREG_INVALID)
         return false;

      /* A final-use scalar load can feed the proven atomic data role
       * directly. The scoreboard pass connects the producer slot to the
       * atomic dependency mask. Every other value gets a private copy because
       * the atomic consumes/destructively releases its data register. */
      if (!apple9_atomic_can_consume_pending_data(lower, atomic, scalar,
                                                  data[0], discard)) {
         data[0] = apple9_dag_general_copy(lower, data[0]);
         if (data[0] == AGX_APPLE9_VREG_INVALID)
            return false;
      }
   }

   uint32_t result = AGX_APPLE9_VREG_INVALID;
   if (!agx_apple9_vir_emit_device_atomic(
          &lower->program, lower->argument_base + atomic->argument, index,
          data, data_components, atomic->op, discard, &result)) {
      lower->reason = "could not emit an Apple9 VIR device atomic";
      return false;
   }

   if (!discard) {
      const unsigned key = atomic->intr->def.index * 4;
      if (key >= lower->ssa_map_count) {
         lower->reason = "Apple9 atomic has an invalid SSA index";
         return false;
      }
      /* Preserve the pending value until its real first consumer. Scoreboard
       * allocation selects a free slot and the ordinary register allocator
       * selects the result-publication landing GPR. */
      lower->ssa_to_vreg[key] = result;
   }

   return true;
}

static bool
apple9_block_has_store(struct util_dynarray *stores, nir_block *block)
{
   util_dynarray_foreach(stores, struct apple9_buffer_store, store) {
      if (store->block == block)
         return true;
   }
   return false;
}

static bool
apple9_block_has_atomic(const struct apple9_dag_lower *lower, nir_block *block)
{
   for (unsigned i = 0; i < lower->atomic_count; ++i) {
      if (lower->atomics[i].block == block)
         return true;
   }
   return false;
}

static bool
apple9_block_has_load(const struct apple9_dag_lower *lower, nir_block *block)
{
   for (unsigned i = 0; i < lower->load_count; ++i) {
      if (lower->loads[i].block == block)
         return true;
   }
   return false;
}

static bool
apple9_block_has_phi(nir_block *block)
{
   nir_foreach_phi(phi, block)
      return true;
   return false;
}

static bool
apple9_block_has_jump(nir_block *block)
{
   nir_instr *last = nir_block_last_instr(block);
   return last != NULL && last->type == nir_instr_type_jump;
}

static bool
apple9_cf_list_has_effects(const struct apple9_dag_lower *lower,
                           struct util_dynarray *stores, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         nir_block *block = nir_cf_node_as_block(node);
         if (apple9_block_has_load(lower, block) ||
             apple9_block_has_store(stores, block) ||
             apple9_block_has_atomic(lower, block) ||
             apple9_block_has_phi(block) || apple9_block_has_jump(block))
            return true;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         nir_block *merge =
            nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
         if (apple9_block_has_phi(merge) ||
             apple9_cf_list_has_effects(lower, stores, &nif->then_list) ||
             apple9_cf_list_has_effects(lower, stores, &nif->else_list))
            return true;
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (apple9_cf_list_has_effects(lower, stores, &loop->body))
            return true;
      }
   }

   return false;
}

static unsigned
apple9_load_instruction_count_in_block(const struct apple9_dag_lower *lower,
                                       nir_block *block)
{
   unsigned count = 0;
   for (unsigned i = 0; i < lower->load_count; ++i) {
      if (lower->loads[i].block != block)
         continue;

      bool first_component = true;
      for (unsigned earlier = 0; earlier < i; ++earlier)
         first_component &= lower->loads[earlier].intr != lower->loads[i].intr;
      count += first_component;
   }
   return count;
}

/* A pending asynchronous return must be handed off before its execution-mask
 * region ends. Otherwise a consumer after ELSE/POP could observe a value that
 * was published for only the other lane population. This correctness-first
 * slice copies every entry or arm-local load to a durable ordinary GPR while
 * the load's own region is still active. */
static const struct agx_apple9_vir_instr *
apple9_dag_value_producer(const struct apple9_dag_lower *lower, uint32_t value)
{
   const struct agx_apple9_vir_instr *producer = NULL;

   for (unsigned p = 0; p < lower->program.instruction_count; ++p) {
      const struct agx_apple9_vir_instr *candidate =
         &lower->program.instructions[p];
      const unsigned components = candidate->dest_components
                                     ? candidate->dest_components
                                     : 1;
      if (value >= candidate->dest && value - candidate->dest < components)
         producer = candidate;
   }

   return producer;
}

static bool
apple9_materialize_atomic_result(struct apple9_dag_lower *lower,
                                 const struct apple9_buffer_atomic *atomic)
{
   const unsigned key = atomic->intr->def.index * 4;
   if (key >= lower->ssa_map_count) {
      lower->reason = "Apple9 atomic has an invalid SSA index";
      return false;
   }

   const uint32_t value = lower->ssa_to_vreg[key];
   if (value == AGX_APPLE9_VREG_INVALID)
      return true;

   const struct agx_apple9_vir_instr *producer =
      apple9_dag_value_producer(lower, value);
   if (producer == NULL || producer->op != AGX_APPLE9_VIR_DEVICE_ATOMIC)
      return true;

   uint32_t durable = apple9_dag_general_copy(lower, value);
   if (durable == AGX_APPLE9_VREG_INVALID)
      return false;
   lower->ssa_to_vreg[key] = durable;
   return true;
}

static bool
apple9_materialize_async_in_block(struct apple9_dag_lower *lower,
                                  nir_block *block)
{
   for (unsigned i = 0; i < lower->load_count; ++i) {
      const struct apple9_scalar_load *load = &lower->loads[i];
      if (load->block != block)
         continue;

      const unsigned key = load->intr->def.index * 4 + load->component;
      if (key >= lower->ssa_map_count ||
          lower->ssa_to_vreg[key] == AGX_APPLE9_VREG_INVALID) {
         lower->reason = "Apple9 block load is absent from the SSA map";
         return false;
      }

      uint32_t durable =
         apple9_dag_general_copy(lower, lower->ssa_to_vreg[key]);
      if (durable == AGX_APPLE9_VREG_INVALID)
         return false;
      lower->ssa_to_vreg[key] = durable;
   }

   for (unsigned i = 0; i < lower->atomic_count; ++i) {
      const struct apple9_buffer_atomic *atomic = &lower->atomics[i];
      if (atomic->block == block &&
          !apple9_materialize_atomic_result(lower, atomic))
         return false;
   }

   return true;
}

static struct apple9_buffer_store *
apple9_find_store(struct util_dynarray *stores, nir_intrinsic_instr *intr)
{
   util_dynarray_foreach(stores, struct apple9_buffer_store, store) {
      if (store->intr == intr)
         return store;
   }
   return NULL;
}

static struct apple9_buffer_atomic *
apple9_find_atomic(struct apple9_dag_lower *lower, nir_intrinsic_instr *intr)
{
   for (unsigned i = 0; i < lower->atomic_count; ++i) {
      if (lower->atomics[i].intr == intr)
         return &lower->atomics[i];
   }
   return NULL;
}

static bool apple9_emit_phi_copies_for_edge(struct apple9_dag_lower *lower,
                                            nir_block *merge,
                                            nir_block *predecessor);

static bool
apple9_block_reaches(nir_block *block, nir_block *successor)
{
   return block != NULL && (block->successors[0] == successor ||
                            block->successors[1] == successor);
}

static bool
apple9_emit_jump(struct apple9_dag_lower *lower, nir_block *block,
                 nir_jump_instr *jump)
{
   if (jump->type == nir_jump_continue) {
      lower->reason =
         "Apple9 encountered a continue after continuation lowering";
      return false;
   }
   if (jump->type != nir_jump_break || lower->loop == NULL) {
      lower->reason = "Apple9 supports only structured loop break jumps";
      return false;
   }

   if (apple9_block_reaches(block, lower->loop->exit) &&
       !apple9_emit_phi_copies_for_edge(lower, lower->loop->exit, block))
      return false;

   if (lower->mask_depth < lower->loop->mask_depth) {
      lower->reason = "Apple9 loop break has an invalid mask-stack depth";
      return false;
   }

   /* The native six-byte break form counts conditional scopes relative to
    * the target loop independently of nested-loop depth.  A direct break is
    * tag 2, one enclosing if is tag 3, and so on. */
   const unsigned scope_tag = 2 + (lower->mask_depth - lower->loop->mask_depth);
   if (scope_tag > UINT8_MAX || lower->loop->depth > UINT8_MAX) {
      lower->reason = "Apple9 loop nesting exceeds the encoded break fields";
      return false;
   }

   const bool ok = agx_apple9_vir_emit_side_effect(
      &lower->program, AGX_APPLE9_VIR_BREAK_MASK_UNWIND,
      AGX_APPLE9_ENC_BREAK_MASK_UNWIND, NULL, 0,
      AGX_APPLE9_BREAK_IMMEDIATE(scope_tag, lower->loop->depth));
   if (!ok)
      lower->reason = "could not emit an Apple9 loop break";
   return ok;
}

static bool
apple9_emit_graphics_output(struct apple9_dag_lower *lower,
                            nir_intrinsic_instr *intr)
{
   if (!nir_src_is_const(intr->src[1]) ||
       intr->src[0].ssa->bit_size != 32)
      return false;
   const unsigned location = nir_intrinsic_io_semantics(intr).location +
                             nir_src_as_uint(intr->src[1]);
   const bool fragment = lower->nir->info.stage == MESA_SHADER_FRAGMENT;
   if (!fragment && nir_intrinsic_src_type(intr) != nir_type_float32)
      return false;
   if (fragment) {
      if (location != FRAG_RESULT_DATA0 || intr->num_components != 1 ||
          nir_intrinsic_component(intr) != 0 || lower->color_stores++)
         return false;
      uint32_t color =
         apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[0].ssa, 0));
      return color != AGX_APPLE9_VREG_INVALID &&
             (lower->tile_read || lower->tile_access || agx_apple9_vir_emit_side_effect(
                &lower->program, AGX_APPLE9_VIR_TILE_ACCESS,
                AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0, 0x0600)) &&
             agx_apple9_vir_emit_side_effect(
                &lower->program, AGX_APPLE9_VIR_TILE_ACCESS,
                AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0, 0x080c) &&
             agx_apple9_vir_emit_side_effect(
                &lower->program, AGX_APPLE9_VIR_TILE_STORE,
                AGX_APPLE9_ENC_TILE_STORE, &color, 1, 0) &&
             agx_apple9_vir_emit_side_effect(
                &lower->program, AGX_APPLE9_VIR_TILE_FENCE,
                AGX_APPLE9_ENC_TILE_FENCE, NULL, 0, 0x020c);
   }
   for (unsigned c = 0; c < intr->num_components; ++c) {
      if (!(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)))
         continue;
      unsigned component = nir_intrinsic_component(intr) + c;
      if (component >= 4)
         return false;
      unsigned slot = component;
      if (location != VARYING_SLOT_POS) {
         int index = apple9_varying_index(lower->varyings, location, component);
         if (index < 0)
            return false;
         slot = 4 + index;
      }
      uint32_t value =
         apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[0].ssa, c));
      uint32_t scale = apple9_dag_imm(lower, fui(1.0f));
      uint32_t sources[] = {value, scale};
      if (value == AGX_APPLE9_VREG_INVALID || scale == AGX_APPLE9_VREG_INVALID)
         return false;
      value = apple9_dag_emit(lower, AGX_APPLE9_VIR_FMUL,
                              AGX_APPLE9_ENC_FLOAT2_EXPORT, sources, 2, 0);
      /* Exports consume published values asynchronously. Until the export
       * completion/release controls are established, keep these destinations
       * distinct and live through shader completion. EXP-M4-56 shows that
       * reusing one destination collapses otherwise valid geometry. */
      if (value == AGX_APPLE9_VREG_INVALID ||
          !agx_apple9_vir_add_live_out(&lower->program, value))
         return false;
      if (value == AGX_APPLE9_VREG_INVALID ||
          !agx_apple9_vir_emit_side_effect(
             &lower->program, AGX_APPLE9_VIR_VARY_STORE,
             AGX_APPLE9_ENC_VARY_STORE, &value, 1, slot))
         return false;
      if (location == VARYING_SLOT_POS)
         lower->position_mask |= BITFIELD_BIT(component);
   }
   return true;
}

/* Emit one NIR block in its original instruction order. Pure SSA expressions
 * are still recursively selected, but dominance guarantees that recursion
 * cannot pull a definition across an earlier side effect. Device loads are
 * issued where their NIR instruction occurs and stores are appended exactly
 * where their intrinsic occurs. */
static bool
apple9_emit_block(struct apple9_dag_lower *lower, struct util_dynarray *stores,
                  nir_block *block)
{
   lower->active_load_block = block;
   lower->active_load_instruction_count =
      apple9_load_instruction_count_in_block(lower, block);
   lower->active_emitted_load_count = 0;
   bool loads_materialized = false;

   nir_foreach_instr(instr, block) {
      if (instr->type == nir_instr_type_phi) {
         nir_phi_instr *phi = nir_instr_as_phi(instr);
         for (unsigned c = 0; c < phi->def.num_components; ++c) {
            const unsigned key = phi->def.index * 4 + c;
            if (key >= lower->ssa_map_count ||
                lower->ssa_to_vreg[key] == AGX_APPLE9_VREG_INVALID) {
               lower->reason =
                  "Apple9 merge phi was not prepared before its block";
               return false;
            }
         }
         continue;
      }

      if (instr->type == nir_instr_type_jump) {
         if (!apple9_materialize_async_in_block(lower, block) ||
             !apple9_emit_jump(lower, block, nir_instr_as_jump(instr)))
            return false;
         loads_materialized = true;
         continue;
      }

      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_barycentric_pixel)
            continue; /* Interpolation mode is consumed by the input operation. */
         if (intr->intrinsic == nir_intrinsic_store_output) {
            if (!apple9_emit_graphics_output(lower, intr)) {
               if (!lower->reason)
                  lower->reason = "unsupported Apple9 graphics output";
               return false;
            }
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_ssbo) {
            struct apple9_buffer_store *store = apple9_find_store(stores, intr);
            if (store == NULL ||
                !apple9_lower_buffer_store_operands(lower, store) ||
                !apple9_emit_buffer_store(lower, store)) {
               if (lower->reason == NULL)
                  lower->reason = "could not emit an Apple9 VIR device store";
               return false;
            }
            continue;
         }
         if (apple9_is_ssbo_atomic(intr->intrinsic)) {
            struct apple9_buffer_atomic *atomic =
               apple9_find_atomic(lower, intr);
            if (atomic == NULL || !apple9_emit_buffer_atomic(lower, atomic)) {
               if (lower->reason == NULL)
                  lower->reason = "could not emit an Apple9 device atomic";
               return false;
            }
            continue;
         }
      }

      nir_def *def = nir_instr_def(instr);
      if (def == NULL || def->bit_size == 1 ||
          (def->bit_size != 8 && def->bit_size != 16 && def->bit_size != 32))
         continue;

      const nir_component_mask_t read = nir_def_components_read(def);
      for (unsigned c = 0; c < def->num_components; ++c) {
         if ((read & BITFIELD_BIT(c)) &&
             apple9_lower_dag_scalar(lower, nir_get_scalar(def, c)) ==
                AGX_APPLE9_VREG_INVALID)
            return false;
      }
   }

   if (lower->active_emitted_load_count !=
       lower->active_load_instruction_count) {
      lower->reason = "Apple9 block loads were not emitted completely";
      return false;
   }

   /* Pending asynchronous values are local to the current execution-mask
    * region. Keep the correctness-first invariant that none crosses a NIR
    * block boundary in a structured program. */
   if (lower->structured_cf && !loads_materialized &&
       !apple9_materialize_async_in_block(lower, block))
      return false;

   lower->active_load_block = NULL;
   return true;
}

struct apple9_predicate_plan {
   enum agx_apple9_encoding encoding;
   uint32_t immediate;
   bool invert_push;
};

static bool
apple9_predicate_condition(const struct apple9_compare *compare,
                           struct apple9_predicate_plan *plan)
{
   memset(plan, 0, sizeof(*plan));
   if (compare->domain == APPLE9_COMPARE_FLOAT &&
       compare->relation == APPLE9_COMPARE_LESS) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT;
      plan->immediate = AGX_APPLE9_PREDICATE_FLT;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_FLOAT &&
              compare->relation == APPLE9_COMPARE_GREATER_EQUAL) {
      /* Native Metal uses this double-inverted extended sequence.  Unlike
       * !(a < b), it preserves IEEE unordered/NaN behavior. */
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
      plan->immediate =
         AGX_APPLE9_PREDICATE_EXT_FGE_SEQUENCE | AGX_APPLE9_PREDICATE_INVERT;
      plan->invert_push = true;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_UNSIGNED &&
              compare->relation == APPLE9_COMPARE_LESS) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT;
      plan->immediate = AGX_APPLE9_PREDICATE_ULT;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_UNSIGNED &&
              compare->relation == APPLE9_COMPARE_GREATER_EQUAL) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT;
      plan->immediate = AGX_APPLE9_PREDICATE_ULT;
      plan->invert_push = true;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_SIGNED &&
              compare->relation == APPLE9_COMPARE_LESS) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT;
      plan->immediate = AGX_APPLE9_PREDICATE_ILT;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_SIGNED &&
              compare->relation == APPLE9_COMPARE_GREATER_EQUAL) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT;
      plan->immediate = AGX_APPLE9_PREDICATE_ILT;
      plan->invert_push = true;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_FLOAT &&
              compare->relation == APPLE9_COMPARE_EQUAL) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
      plan->immediate = AGX_APPLE9_PREDICATE_EXT_FEQ;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_FLOAT &&
              compare->relation == APPLE9_COMPARE_NOT_EQUAL) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
      plan->immediate = AGX_APPLE9_PREDICATE_EXT_FEQ;
      plan->invert_push = true;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_INTEGER &&
              compare->relation == APPLE9_COMPARE_EQUAL) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
      plan->immediate = AGX_APPLE9_PREDICATE_EXT_IEQ;
      return true;
   } else if (compare->domain == APPLE9_COMPARE_INTEGER &&
              compare->relation == APPLE9_COMPARE_NOT_EQUAL) {
      plan->encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED;
      plan->immediate = AGX_APPLE9_PREDICATE_EXT_IEQ;
      plan->invert_push = true;
      return true;
   }

   return false;
}

static bool
apple9_emit_condition_predicate(struct apple9_dag_lower *lower,
                                nir_def *condition, bool *invert_push)
{
   nir_scalar predicate = apple9_chase_trivial(nir_get_scalar(condition, 0));

   struct apple9_predicate_plan plan;
   struct apple9_compare compare;
   uint32_t sources[2];
   if (nir_def_instr_type(predicate.def) == nir_instr_type_alu &&
       apple9_normalize_compare(nir_scalar_alu_op(predicate), &compare) &&
       apple9_predicate_condition(&compare, &plan)) {
      sources[0] = apple9_lower_dag_source(lower, predicate, 0);
      sources[1] = apple9_lower_dag_source(lower, predicate, 1);
   } else {
      /* Metal materializes arbitrary pure Boolean expressions to a 0/1 GPR,
       * then controls execution with an integer compare against zero. */
      sources[0] = apple9_lower_bool_scalar(lower, predicate);
      sources[1] = apple9_dag_zero(lower);
      plan = (struct apple9_predicate_plan){
         .encoding = AGX_APPLE9_ENC_PREDICATE_COMPARE_EXTENDED,
         .immediate = AGX_APPLE9_PREDICATE_EXT_IEQ,
         .invert_push = true,
      };
   }
   if (sources[0] == AGX_APPLE9_VREG_INVALID ||
       sources[1] == AGX_APPLE9_VREG_INVALID)
      return false;

   /* The push consumes this condition immediately.  Enclosing lane state is
    * already saved by the implicit mask stack, so native Metal reuses
    * predicate bank zero at every ordinary-if nesting depth. */
   plan.immediate |= AGX_APPLE9_PREDICATE_BANK(0);
   if (!agx_apple9_vir_emit_side_effect(
          &lower->program, AGX_APPLE9_VIR_PREDICATE_COMPARE, plan.encoding,
          sources, ARRAY_SIZE(sources), plan.immediate)) {
      lower->reason = "could not emit an Apple9 predicate comparison";
      return false;
   }
   *invert_push = plan.invert_push;
   return true;
}

static bool
apple9_emit_if_predicate(struct apple9_dag_lower *lower, nir_if *nif,
                         bool *invert_push)
{
   return apple9_emit_condition_predicate(lower, nif->condition.ssa,
                                          invert_push);
}

static bool
apple9_emit_exec_mask(struct apple9_dag_lower *lower, bool push, bool invert)
{
   assert(push || !invert);
   const unsigned selector = push ? AGX_APPLE9_EXEC_MASK_PREDICATE(0) : 0;
   const bool ok = agx_apple9_vir_emit_side_effect(
      &lower->program,
      push ? AGX_APPLE9_VIR_EXEC_MASK_PUSH : AGX_APPLE9_VIR_EXEC_MASK_POP,
      push ? AGX_APPLE9_ENC_EXEC_MASK_PUSH : AGX_APPLE9_ENC_EXEC_MASK_POP, NULL,
      0, selector | (invert ? AGX_APPLE9_EXEC_MASK_INVERT : 0));
   if (!ok)
      lower->reason = "could not emit an Apple9 execution-mask operation";
   return ok;
}

static bool
apple9_emit_exec_mask_else(struct apple9_dag_lower *lower)
{
   const bool ok = agx_apple9_vir_emit_side_effect(
      &lower->program, AGX_APPLE9_VIR_EXEC_MASK_ELSE,
      AGX_APPLE9_ENC_EXEC_MASK_ELSE, NULL, 0, 0);
   if (!ok)
      lower->reason = "could not emit an Apple9 execution-mask else";
   return ok;
}

/* Allocate the SSA values that exist after reconvergence. MERGE emits no
 * machine instruction; each component gives register allocation a normal
 * definition whose storage remains live while both predecessor edges
 * conditionally write it. */
static bool
apple9_prepare_phis(struct apple9_dag_lower *lower, nir_block *merge)
{
   nir_foreach_phi(phi, merge) {
      if (phi->def.num_components < 1 || phi->def.num_components > 4 ||
          (phi->def.bit_size != 1 && phi->def.bit_size != 8 &&
           phi->def.bit_size != 16 && phi->def.bit_size != 32)) {
         lower->reason =
            "Apple9 requires one-to-four-component 1/8/16/32-bit phis";
         return false;
      }

      unsigned sources = 0;
      nir_foreach_phi_src(src, phi) {
         if (!apple9_block_reaches(src->pred, merge)) {
            lower->reason = "Apple9 phi source is not a CFG predecessor";
            return false;
         }
         ++sources;
      }
      if (sources == 0) {
         lower->reason = "Apple9 phi has no predecessor values";
         return false;
      }

      for (unsigned c = 0; c < phi->def.num_components; ++c) {
         const unsigned key = phi->def.index * 4 + c;
         if (key >= lower->ssa_map_count) {
            lower->reason = "Apple9 phi has an invalid SSA index";
            return false;
         }
         if (lower->ssa_to_vreg[key] != AGX_APPLE9_VREG_INVALID)
            continue;
         uint32_t merge_vreg = agx_apple9_vir_emit_merge(&lower->program);
         if (merge_vreg == AGX_APPLE9_VREG_INVALID) {
            lower->reason = "out of memory allocating an Apple9 phi value";
            return false;
         }
         lower->ssa_to_vreg[key] = merge_vreg;
      }
   }

   return true;
}

/* Resolve each phi exactly where its NIR predecessor executes.  The source
 * computation and the copy both run under that predecessor's lane mask, so
 * arbitrary non-speculatable arm expressions can feed the merge without an
 * eager select or a capture-assigned physical register. */
static bool
apple9_emit_phi_copies_for_edge(struct apple9_dag_lower *lower,
                                nir_block *merge, nir_block *predecessor)
{
   nir_foreach_phi(phi, merge) {
      nir_def *source = NULL;
      nir_foreach_phi_src(src, phi) {
         if (src->pred == predecessor) {
            source = src->src.ssa;
            break;
         }
      }

      if (source == NULL) {
         lower->reason = "Apple9 phi edge is incomplete";
         return false;
      }

      for (unsigned c = 0; c < phi->def.num_components; ++c) {
         const unsigned key = phi->def.index * 4 + c;
         if (key >= lower->ssa_map_count ||
             lower->ssa_to_vreg[key] == AGX_APPLE9_VREG_INVALID) {
            lower->reason = "Apple9 phi edge is incomplete";
            return false;
         }

         nir_scalar source_scalar = nir_get_scalar(source, c);
         uint32_t value = phi->def.bit_size == 1
                             ? apple9_lower_bool_scalar(lower, source_scalar)
                             : apple9_lower_dag_scalar(lower, source_scalar);
         if (value == AGX_APPLE9_VREG_INVALID ||
             !agx_apple9_vir_emit_masked_copy(&lower->program,
                                              lower->ssa_to_vreg[key], value)) {
            if (lower->reason == NULL)
               lower->reason = "could not emit an Apple9 phi-edge copy";
            return false;
         }
      }
   }

   return true;
}

static bool apple9_emit_cf_list(struct apple9_dag_lower *lower,
                                struct util_dynarray *stores,
                                struct exec_list *list);

static bool
apple9_emit_loop_mask_op(struct apple9_dag_lower *lower,
                         enum agx_apple9_vir_opcode op,
                         enum agx_apple9_encoding encoding, uint32_t immediate,
                         const char *failure)
{
   const bool ok = agx_apple9_vir_emit_side_effect(
      &lower->program, op, encoding, NULL, 0, immediate);
   if (!ok)
      lower->reason = failure;
   return ok;
}

static bool
apple9_emit_loop_backedge(struct apple9_dag_lower *lower,
                          unsigned header_instruction)
{
   if (!agx_apple9_vir_emit_branch(&lower->program, AGX_APPLE9_VIR_JMP_EXEC_ANY,
                                   AGX_APPLE9_ENC_JMP_EXEC_ANY,
                                   header_instruction)) {
      lower->reason = "could not emit an Apple9 loop backedge";
      return false;
   }
   return true;
}

/* Recognize the NIR shape for `if (condition) break`. It is lowered directly
 * into a loop-mask update rather than pushing a temporary conditional scope. */
static nir_block *
apple9_direct_break_arm(struct exec_list *list)
{
   nir_block *result = NULL;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type != nir_cf_node_block || result != NULL)
         return NULL;

      nir_block *block = nir_cf_node_as_block(node);
      unsigned instructions = 0;
      nir_foreach_instr(instr, block) {
         ++instructions;
         if (instr->type != nir_instr_type_jump ||
             nir_instr_as_jump(instr)->type != nir_jump_break)
            return NULL;
      }
      if (instructions != 1)
         return NULL;
      result = block;
   }
   return result;
}

static bool
apple9_emit_loop_predicate(struct apple9_dag_lower *lower,
                           nir_if *nif, bool break_on_true,
                           unsigned predicate_bank)
{
   /* Native loop breaks publish an extended integer-equality predicate before
    * LOOP_MASK_UPDATE. Keep this correctness-first lowering general by
    * materializing the NIR Boolean to 0/1, then comparing it with zero.  The
    * loop update removes lanes selected by this predicate, so select the
    * break edge itself: compare with one for a true-arm break and with zero
    * for a false-arm break.  In particular, do not extrapolate the ordinary
    * predicate's byte-0 inversion bit to this loop-tail form; native loop
    * predicates use an explicit scratch-bank convention.
    *
    * Metal often avoids the materialization by strength-reducing comparisons
    * to equality with a terminal value. That is an independent optimization. */
   nir_scalar condition =
      apple9_chase_trivial(nir_get_scalar(nif->condition.ssa, 0));
   uint32_t sources[2] = {
      apple9_lower_bool_scalar(lower, condition),
      apple9_dag_imm(lower, break_on_true ? 1 : 0),
   };
   if (sources[0] == AGX_APPLE9_VREG_INVALID ||
       sources[1] == AGX_APPLE9_VREG_INVALID)
      return false;

   if (!agx_apple9_vir_emit_side_effect(
          &lower->program, AGX_APPLE9_VIR_PREDICATE_COMPARE,
          AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP, sources, ARRAY_SIZE(sources),
          AGX_APPLE9_PREDICATE_EXT_IEQ |
             AGX_APPLE9_PREDICATE_BANK(predicate_bank))) {
      lower->reason = "could not emit an Apple9 loop completion predicate";
      return false;
   }
   return true;
}

/* A source-level `if (condition) break` consumes its condition directly; the
 * condition is not first pushed as another ordinary mask scope.  Directly in
 * a loop, native Metal updates the current loop-mask entry and skips to the
 * loop pop when no lanes remain.  Beneath additional observable conditionals,
 * the six-byte form removes the matching lanes from the target loop while
 * unwinding those active scopes.
 *
 * Branch targets are left unresolved here.  apple9_emit_loop() patches every
 * unresolved JMP_EXEC_NONE in this loop body to the loop-pop instruction once
 * its final instruction index is known. */
static bool
apple9_emit_direct_loop_break_if(struct apple9_dag_lower *lower,
                                 struct util_dynarray *stores, nir_if *nif,
                                 bool *handled)
{
   *handled = false;
   if (lower->loop == NULL)
      return true;

   nir_block *then_break = apple9_direct_break_arm(&nif->then_list);
   nir_block *else_break = apple9_direct_break_arm(&nif->else_list);
   const bool then_empty =
      !apple9_cf_list_has_effects(lower, stores, &nif->then_list);
   const bool else_empty =
      !apple9_cf_list_has_effects(lower, stores, &nif->else_list);
   if (!((then_break != NULL && else_empty) ||
         (else_break != NULL && then_empty)))
      return true;

   const bool break_on_true = then_break != NULL;
   nir_block *break_predecessor = break_on_true ? then_break : else_break;

   /* Exit phis must receive their breaking-edge values while precisely the
    * breaking lanes are active.  Use a temporary ordinary mask only for the
    * copies, then restore the loop mask before performing the direct update. */
   if (apple9_block_has_phi(lower->loop->exit)) {
      bool invert_break = false;
      if (!apple9_emit_if_predicate(lower, nif, &invert_break) ||
          !apple9_emit_exec_mask(lower, true, invert_break ^ !break_on_true))
         return false;
      ++lower->mask_depth;
      if (!apple9_emit_phi_copies_for_edge(lower, lower->loop->exit,
                                           break_predecessor) ||
          !apple9_emit_exec_mask(lower, false, false))
         return false;
      assert(lower->mask_depth > 0);
      --lower->mask_depth;
   }

   /* The direct update consumes the predicate associated with the current
    * loop-mask entry.  BREAK_MASK_UNWIND is different: native Metal keeps
    * the predicate tied to the target loop while additional conditional
    * scopes accumulate above it.  Thus one and two enclosing ifs both use
    * bank 1 for a depth-one loop; a break from an inner depth-two loop uses
    * bank 2. */
   const bool direct_update = lower->mask_depth == lower->loop->mask_depth;
   const unsigned predicate_bank =
      direct_update ? lower->loop->depth - 1 : lower->loop->depth;
   if (predicate_bank >= AGX_APPLE9_PREDICATE_BANK_COUNT) {
      lower->reason = "Apple9 loop control exhausted the predicate scratch bank";
      return false;
   }
   if (!apple9_emit_loop_predicate(lower, nif, break_on_true, predicate_bank))
      return false;

   if (direct_update) {
      const unsigned selector =
         AGX_APPLE9_LOOP_MASK_INVERT |
         AGX_APPLE9_LOOP_MASK_PREDICATE(predicate_bank);
      if (!apple9_emit_loop_mask_op(
             lower, AGX_APPLE9_VIR_LOOP_MASK_UPDATE,
             AGX_APPLE9_ENC_LOOP_MASK_UPDATE, selector,
             "could not emit an Apple9 direct-break loop-mask update") ||
          !agx_apple9_vir_emit_branch(
             &lower->program, AGX_APPLE9_VIR_JMP_EXEC_NONE,
             AGX_APPLE9_ENC_JMP_EXEC_NONE, AGX_APPLE9_VREG_INVALID)) {
         if (lower->reason == NULL)
            lower->reason = "could not emit an Apple9 direct-break branch";
         return false;
      }
   } else {
      const unsigned scope_tag =
         2 + (lower->mask_depth - lower->loop->mask_depth);
      if (scope_tag > UINT8_MAX || lower->loop->depth > UINT8_MAX ||
          !agx_apple9_vir_emit_side_effect(
             &lower->program, AGX_APPLE9_VIR_BREAK_MASK_UNWIND,
             AGX_APPLE9_ENC_BREAK_MASK_UNWIND, NULL, 0,
             AGX_APPLE9_BREAK_IMMEDIATE(scope_tag, lower->loop->depth))) {
         if (lower->reason == NULL)
            lower->reason = "could not emit an Apple9 nested loop break";
         return false;
      }
   }

   *handled = true;
   return true;
}

static bool
apple9_emit_loop(struct apple9_dag_lower *lower, struct util_dynarray *stores,
                 nir_loop *loop)
{
   nir_cf_node *previous = nir_cf_node_prev(&loop->cf_node);
   nir_cf_node *next = nir_cf_node_next(&loop->cf_node);
   if (previous == NULL || previous->type != nir_cf_node_block ||
       next == NULL || next->type != nir_cf_node_block) {
      lower->reason = "Apple9 loop lacks structured entry or exit blocks";
      return false;
   }

   nir_block *entry = nir_cf_node_as_block(previous);
   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   nir_block *exit = nir_cf_node_as_block(next);

   if (apple9_block_has_phi(header)) {
      if (!apple9_prepare_phis(lower, header) ||
          !apple9_emit_phi_copies_for_edge(lower, header, entry))
         return false;
   }
   if (apple9_block_has_phi(exit) && !apple9_prepare_phis(lower, exit))
      return false;

   /* Match Apple8's structured-loop architecture: the complete NIR body is
    * one implicit loop. Explicit break edges update or unwind the Apple9 loop
    * mask, and JMP_EXEC_ANY repeats while that mask still contains lanes.
    * There is no backend-recognized source-level top/bottom test.
    *
    * At top level the hardware supplies the initial loop mask implicitly,
    * matching native bottom-tested loops. A nested loop must save the
    * enclosing lane population in an explicit loop scope. */
   if (lower->mask_depth > 0 &&
       !apple9_emit_loop_mask_op(lower, AGX_APPLE9_VIR_LOOP_MASK_PUSH,
                                 AGX_APPLE9_ENC_LOOP_MASK_PUSH, 0,
                                 "could not emit an Apple9 loop-mask push"))
      return false;
   ++lower->mask_depth;

   const unsigned header_instruction = lower->program.instruction_count;
   struct apple9_loop_context context = {
      .parent = lower->loop,
      .nir = loop,
      .exit = exit,
      .depth = lower->loop ? lower->loop->depth + 1 : 1,
      .mask_depth = lower->mask_depth,
   };
   lower->loop = &context;

   if (!apple9_emit_cf_list(lower, stores, &loop->body))
      return false;

   if (apple9_block_reaches(latch, header)) {
      if (apple9_block_has_phi(header) &&
          !apple9_emit_phi_copies_for_edge(lower, header, latch))
         return false;
      if (!apple9_emit_loop_backedge(lower, header_instruction))
         return false;
   }

   lower->loop = context.parent;

   /* Direct breaks branch to this loop's kind-2 pop.  Nested loops have
    * already resolved their own branches, so the remaining unresolved
    * JMP_EXEC_NONE instructions in this body's range belong to this loop. */
   const unsigned loop_pop_instruction = lower->program.instruction_count;
   for (unsigned i = header_instruction; i < loop_pop_instruction; ++i) {
      struct agx_apple9_vir_instr *instruction =
         &lower->program.instructions[i];
      if (instruction->op == AGX_APPLE9_VIR_JMP_EXEC_NONE &&
          instruction->branch_target == AGX_APPLE9_VREG_INVALID)
         instruction->branch_target = loop_pop_instruction;
   }

   if (!apple9_emit_loop_mask_op(lower, AGX_APPLE9_VIR_LOOP_MASK_POP,
                                 AGX_APPLE9_ENC_LOOP_MASK_POP, 0,
                                 "could not emit an Apple9 loop-mask pop"))
      return false;
   assert(lower->mask_depth > 0);
   --lower->mask_depth;
   return true;
}

static bool
apple9_emit_if(struct apple9_dag_lower *lower, struct util_dynarray *stores,
               nir_if *nif)
{
   bool handled = false;
   if (!apple9_emit_direct_loop_break_if(lower, stores, nif, &handled) ||
       handled)
      return handled;

   nir_cf_node *next = nir_cf_node_next(&nif->cf_node);
   if (next == NULL || next->type != nir_cf_node_block) {
      lower->reason = "Apple9 if/else has no structured merge block";
      return false;
   }

   nir_block *merge = nir_cf_node_as_block(next);
   const bool has_phis = apple9_block_has_phi(merge);
   const bool has_then =
      has_phis || apple9_cf_list_has_effects(lower, stores, &nif->then_list);
   const bool has_else =
      has_phis || apple9_cf_list_has_effects(lower, stores, &nif->else_list);

   /* A condition with no observable arm and no escaping value is dead. */
   if (!has_then && !has_else)
      return true;

   if (has_phis && !apple9_prepare_phis(lower, merge))
      return false;

   bool invert_push = false;
   if (!apple9_emit_if_predicate(lower, nif, &invert_push) ||
       !apple9_emit_exec_mask(lower, true, invert_push))
      return false;
   ++lower->mask_depth;

   nir_block *then_pred = apple9_cf_list_last_block(&nif->then_list);
   nir_block *else_pred = apple9_cf_list_last_block(&nif->else_list);
   if ((has_then && !apple9_emit_cf_list(lower, stores, &nif->then_list)) ||
       (has_phis && apple9_block_reaches(then_pred, merge) &&
        !apple9_emit_phi_copies_for_edge(lower, merge, then_pred)))
      return false;

   if (has_else) {
      if (!apple9_emit_exec_mask_else(lower) ||
          !apple9_emit_cf_list(lower, stores, &nif->else_list) ||
          (has_phis && apple9_block_reaches(else_pred, merge) &&
           !apple9_emit_phi_copies_for_edge(lower, merge, else_pred)))
         return false;
   }

   if (!apple9_emit_exec_mask(lower, false, false))
      return false;
   assert(lower->mask_depth > 0);
   --lower->mask_depth;
   return true;
}

/* Apple9's mask operations address a recursive hardware mask structure.
 * Mirror NIR directly: emit each structured block, if, or loop in program
 * order while keeping VIR and allocation linear. */
static bool
apple9_emit_cf_list(struct apple9_dag_lower *lower,
                    struct util_dynarray *stores, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         if (!apple9_emit_block(lower, stores, nir_cf_node_as_block(node)))
            return false;
         break;
      case nir_cf_node_if:
         if (!apple9_emit_if(lower, stores, nir_cf_node_as_if(node)))
            return false;
         break;
      case nir_cf_node_loop:
         if (!apple9_emit_loop(lower, stores, nir_cf_node_as_loop(node)))
            return false;
         break;
      default:
         lower->reason = "Apple9 compiler requires structured control flow";
         return false;
      }
   }

   return true;
}

static bool
apple9_compile_dag(nir_shader *nir, struct agx_shader_part *out,
                   struct agx_apple9_compute_profile *profile,
                   const struct agx_apple9_varying_layout *varyings,
                   const char **reason)
{
   /* Undefined lanes can remain in vectors with partial output write masks.
    * Materialize them consistently before instruction selection. */
   nir_lower_undef_to_zero(nir, NULL);
   /* Format and shader-math lowering can create identical expressions. */
   bool progress;
   do {
      progress = nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_cse(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);

   /* Selection follows NIR order, so shorten live ranges after lowering and
    * before collecting loads or assigning virtual registers. Use the same
    * movement options as the Apple8 backend; NIR checks load reorderability.
    */
   nir_move_options move = nir_move_const_undef | nir_move_load_ubo |
                           nir_move_load_input | nir_move_load_frag_coord |
                           nir_move_comparisons | nir_move_copies |
                           nir_move_load_ssbo | nir_move_alu;
   nir_opt_sink(nir, move);
   nir_opt_move(nir, move);

   struct util_dynarray loads = UTIL_DYNARRAY_INIT;
   struct util_dynarray stores = UTIL_DYNARRAY_INIT;
   struct util_dynarray atomics = UTIL_DYNARRAY_INIT;
   struct apple9_buffer_map resource_map = {0};

   if (!apple9_collect_buffer_map(nir, &resource_map, reason))
      return false;

   if (!apple9_find_buffer_dag(nir, &resource_map, &loads, &stores, &atomics,
                               reason)) {
      util_dynarray_fini(&loads);
      util_dynarray_fini(&stores);
      util_dynarray_fini(&atomics);
      return false;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   struct apple9_dag_lower lower = {
      .nir = nir,
      .varyings = varyings,
      .zero_vreg = AGX_APPLE9_VREG_INVALID,
      .loads = loads.data,
      .load_count =
         util_dynarray_num_elements(&loads, struct apple9_scalar_load),
      .atomics = atomics.data,
      .atomic_count =
         util_dynarray_num_elements(&atomics, struct apple9_buffer_atomic),
      .argument_base = nir->info.stage == MESA_SHADER_COMPUTE && atomics.size == 0
                          ? AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE
                          : 0,
      .structured_cf = apple9_cf_list_has_control_flow(&impl->body),
   };
   for (unsigned i = 0; i < ARRAY_SIZE(lower.system_vreg); ++i)
      lower.system_vreg[i] = AGX_APPLE9_VREG_INVALID;
   for (unsigned i = 0; i < lower.load_count; ++i) {
      bool first = true;
      for (unsigned earlier = 0; earlier < i; ++earlier)
         first &= lower.loads[earlier].intr != lower.loads[i].intr;
      lower.load_instruction_count += first;
   }
   agx_apple9_vir_init(&lower.program);
   lower.ssa_map_count = impl->ssa_alloc * 4;
   lower.ssa_to_vreg = malloc(lower.ssa_map_count * sizeof(uint32_t));
   if (lower.ssa_to_vreg == NULL) {
      *reason = "out of memory indexing Apple9 NIR";
      goto fail;
   }
   for (unsigned i = 0; i < lower.ssa_map_count; ++i)
      lower.ssa_to_vreg[i] = AGX_APPLE9_VREG_INVALID;

   if (!apple9_emit_cf_list(&lower, &stores, &impl->body)) {
      *reason = lower.reason != NULL
                   ? lower.reason
                   : "could not emit Apple9 structured control flow";
      goto fail;
   }

   if ((nir->info.stage == MESA_SHADER_VERTEX && lower.position_mask != 15) ||
       (nir->info.stage == MESA_SHADER_FRAGMENT && lower.color_stores != 1)) {
      *reason = "Apple9 render requires complete position or RT0 output";
      goto fail;
   }

   if (lower.emitted_load_count != lower.load_instruction_count) {
      *reason = "Apple9 DAG contains an input load outside the store graph";
      goto fail;
   }

   apple9_infer_device_load_index_contracts(&lower.program);

   if (!agx_apple9_assign_vir_scoreboard_slots(&lower.program, reason))
      goto fail;

   apple9_infer_special_result_hints(&lower.program);

   /* The byte/subword source selector is not generalized yet.  Scoreboard
    * materialization may have replaced the original load SSA, so constrain
    * the store's final source rather than the pre-materialization value. */
   for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
      struct agx_apple9_vir_instr *store = &lower.program.instructions[i];
      if (store->op == AGX_APPLE9_VIR_DEVICE_STORE &&
          store->memory_bits != 32 &&
          !agx_apple9_vir_set_fixed_phys(&lower.program, store->src[0], 0)) {
         *reason = "could not reserve a proven narrow-store source";
         goto fail;
      }
   }

   if (!agx_apple9_allocate_vir(&lower.program, reason))
      goto fail;

   if (getenv("AGX_APPLE9_TRACE") != NULL) {
      fprintf(stderr, "APPLE9_ALLOC_STATS stage=%u peak_live_gprs=%u\n",
              nir->info.stage, lower.program.peak_live_gprs);
      for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
         const struct agx_apple9_vir_instr *instruction =
            &lower.program.instructions[i];
         fprintf(stderr, "APPLE9_ALLOC i=%u op=%u enc=%u dst=", i,
                 instruction->op, instruction->encoding);
         if (instruction->dest == AGX_APPLE9_VREG_INVALID)
            fputs("-", stderr);
         else
            fprintf(stderr, "r%u", lower.program.phys[instruction->dest]);
         if (instruction->op == AGX_APPLE9_VIR_MASKED_COPY)
            fprintf(stderr, " target=r%u",
                    lower.program.phys[instruction->target]);
         fputs(" src=", stderr);
         for (unsigned s = 0; s < instruction->nr_srcs; ++s)
            fprintf(stderr, "%sr%u", s ? "," : "",
                    lower.program.phys[instruction->src[s]]);
         fprintf(stderr, " imm=%#x live=%#x slot=%u producer_slot=%u\n",
                 instruction->immediate, instruction->live_after_mask,
                 instruction->scoreboard_slot,
                 instruction->producer_scoreboard_slot);
      }
   }

   struct apple9_emitter emitter = {.bytes = UTIL_DYNARRAY_INIT};
   unsigned emission_max_gpr = lower.program.max_phys_gpr;
   struct agx_apple9_packed_instruction packed;
   unsigned *vir_offsets =
      malloc((lower.program.instruction_count + 1) * sizeof(*vir_offsets));
   if (vir_offsets == NULL) {
      *reason = "out of memory recording Apple9 branch boundaries";
      goto fail;
   }

   for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
      vir_offsets[i] = emitter.bytes.size;
      const struct agx_apple9_vir_instr *instruction =
         &lower.program.instructions[i];
      if (instruction->op == AGX_APPLE9_VIR_MERGE)
         continue;
      if (instruction->op == AGX_APPLE9_VIR_COLLECT) {
         if (!apple9_emit_collect_vir(&emitter, instruction, lower.program.phys,
                                      &emission_max_gpr, reason)) {
            util_dynarray_fini(&emitter.bytes);
            free(vir_offsets);
            goto fail;
         }
         continue;
      }
      if (instruction->op == AGX_APPLE9_VIR_MASKED_COPY) {
         if (!apple9_emit_masked_copy_vir(&emitter, instruction,
                                          lower.program.phys, &emission_max_gpr,
                                          reason)) {
            util_dynarray_fini(&emitter.bytes);
            free(vir_offsets);
            goto fail;
         }
         continue;
      }
      if (instruction->op == AGX_APPLE9_VIR_DEVICE_STORE) {
         if (!apple9_emit_device_store_vir(&emitter, instruction,
                                           lower.program.phys, reason)) {
            util_dynarray_fini(&emitter.bytes);
            free(vir_offsets);
            goto fail;
         }
         continue;
      }
      if (!agx_apple9_pack_vir_instruction(instruction, lower.program.phys,
                                           &packed, reason) ||
          !apple9_emit_packed(&emitter, &packed)) {
         if (getenv("AGX_APPLE9_TRACE") != NULL) {
            fprintf(stderr, "APPLE9_PACK_FAIL i=%u op=%u enc=%u dst=", i,
                    instruction->op, instruction->encoding);
            if (instruction->dest == AGX_APPLE9_VREG_INVALID)
               fputs("-", stderr);
            else
               fprintf(stderr, "r%u", lower.program.phys[instruction->dest]);
            fputs(" src=", stderr);
            for (unsigned s = 0; s < instruction->nr_srcs; ++s)
               fprintf(stderr, "%sr%u", s ? "," : "",
                       lower.program.phys[instruction->src[s]]);
            fputc('\n', stderr);
         }
         if (*reason == NULL)
            *reason = "Apple9 DAG instruction pack failed";
         util_dynarray_fini(&emitter.bytes);
         free(vir_offsets);
         goto fail;
      }
      if (getenv("AGX_APPLE9_TRACE") != NULL) {
         fprintf(stderr, "APPLE9_VIR i=%u op=%u enc=%u dst=", i,
                 instruction->op, instruction->encoding);
         if (instruction->dest == AGX_APPLE9_VREG_INVALID)
            fputs("-", stderr);
         else
            fprintf(stderr, "r%u", lower.program.phys[instruction->dest]);
         fputs(" src=", stderr);
         for (unsigned s = 0; s < instruction->nr_srcs; ++s)
            fprintf(stderr, "%sr%u", s ? "," : "",
                    lower.program.phys[instruction->src[s]]);
         fprintf(stderr, " live=%#x slot=%u producer_slot=%u\n",
                 instruction->live_after_mask, instruction->scoreboard_slot,
                 instruction->producer_scoreboard_slot);
      }
   }
   vir_offsets[lower.program.instruction_count] = emitter.bytes.size;

   /* Branch displacements are relative to the start of the branch, not its
    * end. Resolve them only now: COLLECT and MASKED_COPY pseudos can emit a
    * variable number of physical instructions after allocation. */
   for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         &lower.program.instructions[i];
      if (instruction->op != AGX_APPLE9_VIR_JMP_EXEC_ANY &&
          instruction->op != AGX_APPLE9_VIR_JMP_EXEC_NONE)
         continue;

      if (instruction->branch_target > lower.program.instruction_count) {
         *reason = "Apple9 branch targets an invalid VIR boundary";
         util_dynarray_fini(&emitter.bytes);
         free(vir_offsets);
         goto fail;
      }
      const int64_t displacement =
         (int64_t)vir_offsets[instruction->branch_target] - vir_offsets[i];
      if (displacement < INT32_MIN || displacement > INT32_MAX) {
         *reason = "Apple9 branch displacement exceeds the compiler range";
         util_dynarray_fini(&emitter.bytes);
         free(vir_offsets);
         goto fail;
      }

      struct agx_apple9_vir_instr resolved = *instruction;
      resolved.immediate = (uint32_t)(int32_t)displacement;
      if (!agx_apple9_pack_vir_instruction(&resolved, lower.program.phys,
                                           &packed, reason) ||
          packed.length !=
             agx_apple9_encoding_info(instruction->encoding)->length) {
         if (*reason == NULL)
            *reason = "could not resolve an Apple9 branch displacement";
         util_dynarray_fini(&emitter.bytes);
         free(vir_offsets);
         goto fail;
      }
      memcpy((uint8_t *)emitter.bytes.data + vir_offsets[i], packed.bytes,
             packed.length);
   }
   free(vir_offsets);
   if (getenv("AGX_APPLE9_TRACE") != NULL) {
      fputs("APPLE9_BINARY ", stderr);
      const uint8_t *binary = emitter.bytes.data;
      for (unsigned i = 0; i < emitter.bytes.size; ++i)
         fprintf(stderr, "%02x", binary[i]);
      fputc('\n', stderr);
   }
   apple9_emit_stop(&emitter);

   out->binary = emitter.bytes.data;
   out->info.stage = nir->info.stage;
   if (nir->info.stage != MESA_SHADER_COMPUTE) {
      out->info.apple9_resource_count = resource_map.count;
      for (unsigned i = 0; i < resource_map.count; ++i) {
         unsigned binding = resource_map.resource[i].binding;
         out->info.apple9_resource_binding[i] = binding;
         if (binding < 32)
            out->info.apple9_ubo_mask |= BITFIELD_BIT(binding);
      }
   }
   if (varyings)
      out->info.apple9_varyings = *varyings;
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      out->info.varyings.fs.nr_cf = varyings->count + 1;
   out->info.main_size = emitter.bytes.size;
   out->info.binary_size = emitter.bytes.size;
   /* nr_gprs is the architectural GPR high-water mark for Apple9.  Keep the
    * established minimum tier, but do not collapse every program using r16+
    * back to the old 16-register capture tier. */
   out->info.nr_gprs = MAX2(emission_max_gpr + 1, 8);
   out->info.stats.instrs = emitter.instructions;
   for (unsigned d = 0; d < 3; ++d)
      out->info.workgroup_size[d] = nir->info.workgroup_size[d];

   if (profile != NULL) {
      const unsigned resource_count = resource_map.count;
      *profile = AGX_APPLE9_SSBO8_SUPERSET_COMPUTE_PROFILE;
      if (lower.atomic_count != 0)
         profile->abi = AGX_APPLE9_COMPUTE_ABI_SSBO8_ATOMIC;
      profile->resource_binding_count = resource_count;
      profile->resource_read_mask = resource_map.read_mask;
      profile->resource_write_mask = resource_map.write_mask;
      for (unsigned i = 0; i < profile->resource_binding_count; ++i) {
         profile->resource_binding[i] = resource_map.resource[i].binding;
         profile->resource_kind[i] = resource_map.resource[i].kind;
      }
      profile->variable_local_size = nir->info.workgroup_size_variable;
      for (unsigned d = 0; d < 3; ++d)
         profile->local_size[d] = nir->info.workgroup_size[d];
   }

   free(lower.ssa_to_vreg);
   agx_apple9_vir_finish(&lower.program);
   util_dynarray_fini(&loads);
   util_dynarray_fini(&stores);
   util_dynarray_fini(&atomics);
   return true;

fail:
   free(lower.ssa_to_vreg);
   agx_apple9_vir_finish(&lower.program);
   util_dynarray_fini(&loads);
   util_dynarray_fini(&stores);
   util_dynarray_fini(&atomics);
   return false;
}

bool
agx_compile_apple9_tiny(nir_shader *nir, struct agx_shader_part *out,
                        struct agx_apple9_compute_profile *profile,
                        const char **reason_out)
{
   agx_nir_lower_apple9_math(nir);

   /* Make every source-level continue an ordinary structured masked region
    * ending at the loop latch. This upstream NIR pass preserves SSA and leaves
    * one backedge, matching the Apple9 loop machine directly. */
   nir_lower_continue_constructs(nir);

   if (getenv("AGX_APPLE9_TRACE_NIR") != NULL)
      nir_print_shader(nir, stderr);

   const char *reason = NULL;
   memset(out, 0, sizeof(*out));
   if (profile != NULL)
      memset(profile, 0, sizeof(*profile));
   if (reason_out != NULL)
      *reason_out = NULL;

   if (nir->info.stage != MESA_SHADER_COMPUTE) {
      reason = "Apple9 bounded compiler supports only compute shaders";
      goto unsupported;
   }

   if (!nir->info.workgroup_size_variable) {
      uint64_t local_threads = 1;
      for (unsigned d = 0; d < 3; ++d) {
         if (!nir->info.workgroup_size[d] ||
             nir->info.workgroup_size[d] > 1024 / local_threads) {
            reason = "Apple9 compute requires a legal fixed local size";
            goto unsupported;
         }
         local_threads *= nir->info.workgroup_size[d];
      }
   }

   if (apple9_compile_dag(nir, out, profile, NULL, &reason))
      return true;

unsupported:
   if (reason_out != NULL)
      *reason_out = reason;
   return false;
}

static unsigned
apple9_io_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

struct apple9_color_lower {
   const struct agx_apple9_blend *blend;
   bool valid;
};

static nir_def *
apple9_blend_factor(nir_builder *b, unsigned factor, nir_def *alpha)
{
   switch (factor) {
   case PIPE_BLENDFACTOR_ZERO: return nir_imm_float(b, 0);
   case PIPE_BLENDFACTOR_ONE: return nir_imm_float(b, 1);
   case PIPE_BLENDFACTOR_SRC_ALPHA: return alpha;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA: return nir_fsub(b, nir_imm_float(b, 1), alpha);
   default: return NULL;
   }
}

/* The validated RT0 store consumes RGBA8. Express conversion as ordinary NIR
 * so register allocation and numerical behavior do not depend on a captured
 * native color-pack sequence. */
static bool
apple9_lower_color(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   struct apple9_color_lower *lower = data;
   if (nir_intrinsic_io_semantics(intr).location != FRAG_RESULT_DATA0 ||
       intr->src[0].ssa->bit_size != 32 || intr->num_components != 4 ||
       nir_intrinsic_component(intr) || nir_intrinsic_write_mask(intr) != 15) {
      lower->valid = false;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   const struct agx_apple9_blend *blend = lower->blend;
   nir_def *dst = NULL, *factors[4] = {0};
   if (blend) {
      if (blend->unsupported || blend->rgb_func != PIPE_BLEND_ADD ||
          blend->alpha_func != PIPE_BLEND_ADD) {
         lower->valid = false;
         return false;
      }
      nir_def *alpha = nir_fsat(b, nir_channel(b, intr->src[0].ssa, 3));
      unsigned requested[] = {blend->rgb_src, blend->rgb_dst,
                              blend->alpha_src, blend->alpha_dst};
      for (unsigned i = 0; i < 4; ++i) {
         factors[i] = apple9_blend_factor(b, requested[i], alpha);
         if (!factors[i]) { lower->valid = false; return false; }
      }
      if (blend->rgb_dst != PIPE_BLENDFACTOR_ZERO ||
          blend->alpha_dst != PIPE_BLENDFACTOR_ZERO || blend->colormask != 15)
         dst = nir_load_output(b, 1, 32, nir_imm_int(b, 0),
            .dest_type = nir_type_uint32,
            .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
   }
   nir_def *packed = nir_imm_int(b, 0);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *v = nir_channel(b, intr->src[0].ssa, c);
      if (blend) {
         nir_def *d = dst ? nir_fmul_imm(b, nir_u2f32(b,
            nir_iand_imm(b, nir_ushr_imm(b, dst, 8*c), 255)), 1.0/255.0)
            : nir_imm_float(b, 0);
         v = (blend->colormask & (1 << c))
            ? nir_fadd(b, nir_fmul(b, nir_fsat(b, v), factors[c == 3 ? 2 : 0]),
                          nir_fmul(b, d, factors[c == 3 ? 3 : 1])) : d;
      }
      v = nir_fmin(b, nir_fmax(b, v, nir_imm_float(b, 0)), nir_imm_float(b, 1));
      v = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, v, 255)));
      packed = nir_ior(b, packed, nir_ishl_imm(b, v, 8 * c));
   }
   nir_src_rewrite(&intr->src[0], packed);
   intr->num_components = 1;
   nir_intrinsic_set_write_mask(intr, 1);
   return true;
}

/* Scalarize graphics UBO vectors so constant matrix-column offsets and
 * dynamic array accesses use the same independently scheduled memory loads.
 * Native vector formation is an optional later optimization. */
static bool
apple9_scalarize_graphics_ubo(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_ubo || intr->num_components == 1)
      return false;
   if (intr->def.bit_size != 32 || nir_intrinsic_align_mul(intr) < 4)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *components[4];
   if (intr->num_components > ARRAY_SIZE(components))
      return false;
   for (unsigned c = 0; c < intr->num_components; ++c) {
      components[c] = nir_load_ubo(
         b, 1, 32, intr->src[0].ssa, nir_iadd_imm(b, intr->src[1].ssa, c * 4),
         .access = nir_intrinsic_access(intr), .align_mul = 4,
         .align_offset = nir_intrinsic_align_offset(intr) % 4,
         .range_base = nir_intrinsic_range_base(intr),
         .range = nir_intrinsic_range(intr));
   }
   nir_def_rewrite_uses(&intr->def,
                        nir_vec(b, components, intr->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

/* Naturally aligned homogeneous channels. Packed bitfields and integer
 * shader inputs can use the ordinary Mesa vertex-format fallback for now. */
bool
agx_apple9_vertex_format_supported(enum pipe_format format)
{
   if (format == PIPE_FORMAT_NONE)
      return false;
   const struct util_format_description *desc = util_format_description(format);
   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB ||
       desc->nr_channels < 1 || desc->nr_channels > 4)
      return false;
   for (unsigned c = 0; c < desc->nr_channels; ++c) {
      const struct util_format_channel_description *ch = &desc->channel[c];
      if (ch->type == UTIL_FORMAT_TYPE_VOID)
         continue;
      if (!ch->size || ch->pure_integer || (ch->shift % ch->size))
         return false;
      if (ch->type == UTIL_FORMAT_TYPE_FLOAT && ch->size == 32)
         continue;
      if ((ch->type == UTIL_FORMAT_TYPE_UNSIGNED || ch->type == UTIL_FORMAT_TYPE_SIGNED) &&
          (ch->size == 8 || ch->size == 16))
         continue;
      return false;
   }
   return true;
}

struct apple9_vertex_lower {
   const struct agx_apple9_vertex_layout *layout;
   bool valid;
};

static bool
apple9_lower_vertex_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct apple9_vertex_lower *lower = data;
   if (intr->intrinsic == nir_intrinsic_store_output && lower->layout &&
       lower->layout->clip_halfz &&
       nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_POS) {
      if (intr->num_components != 4 || nir_intrinsic_component(intr)) {
         lower->valid = false;
         return false;
      }
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *pos = intr->src[0].ssa;
      nir_def *z = nir_fmul_imm(b, nir_fadd(b, nir_channel(b, pos, 2),
                                           nir_channel(b, pos, 3)), .5f);
      nir_src_rewrite(&intr->src[0], nir_vector_insert_imm(b, pos, z, 2));
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;
   /* Gallium compacts elements independently of API attribute locations. */
   unsigned attribute = nir_intrinsic_base(intr);
   if (!lower->layout || attribute >= 16 || intr->def.bit_size != 32 ||
       intr->num_components < 1 || intr->num_components > 4 ||
       nir_intrinsic_component(intr) + intr->num_components > 4 ||
       !nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) ||
       lower->layout->buffer[attribute] >= 32 ||
       !agx_apple9_vertex_format_supported(lower->layout->format[attribute])) {
      lower->valid = false;
      return false;
   }
   const struct util_format_description *desc =
      util_format_description(lower->layout->format[attribute]);
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *index = nir_load_vertex_id(b);
   nir_def *offset = nir_iadd_imm(b,
      nir_imul_imm(b, index, lower->layout->stride[attribute]),
      lower->layout->offset[attribute]);
   nir_def *components[4];
   for (unsigned c = 0; c < intr->num_components; ++c) {
      unsigned component = nir_intrinsic_component(intr) + c;
      unsigned swizzle = desc->swizzle[component];
      if (swizzle >= PIPE_SWIZZLE_0) {
         components[c] = nir_imm_float(b, swizzle == PIPE_SWIZZLE_1 ? 1 : 0);
         continue;
      }
      const struct util_format_channel_description *ch = &desc->channel[swizzle];
      unsigned bytes = ch->size / 8;
      unsigned add = lower->layout->offset[attribute] + ch->shift / 8;
      if (!bytes || (lower->layout->stride[attribute] % bytes) || (add % bytes)) {
         lower->valid = false;
         return false;
      }
      nir_def *value = nir_load_ubo(b, 1, ch->size,
         nir_imm_int(b, 32 + lower->layout->buffer[attribute]),
         nir_iadd_imm(b, offset, ch->shift / 8),
         .align_mul = bytes, .range = ~0u);
      if (ch->type != UTIL_FORMAT_TYPE_FLOAT) {
         unsigned bits = ch->size;
         if (ch->type == UTIL_FORMAT_TYPE_SIGNED) {
            value = nir_i2i32(b, value);
            value = nir_i2f32(b, value);
            if (ch->normalized)
               value = nir_fmax(b, nir_fmul_imm(b, value,
                  1.0 / ((1u << (bits - 1)) - 1)), nir_imm_float(b, -1));
         } else {
            value = nir_u2u32(b, value);
            value = nir_u2f32(b, value);
            if (ch->normalized)
               value = nir_fmul_imm(b, value, 1.0 / ((1u << bits) - 1));
         }
      }
      components[c] = value;
   }
   nir_def_rewrite_uses(&intr->def, nir_vec(b, components, intr->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

/* EXP-0111 FS-01: fragment SR 0xa0/0xa1 are integer pixel coordinates.
 * Gallium advertises upper-left, integer centers; mesa/st performs the API
 * center/origin adjustment. Keep this separate from user interpolation slots.
 * Z/W need their own interpolation model and must not be fabricated. */
static bool
apple9_lower_window_position(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned component = 0;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_frag_coord:
   case nir_intrinsic_load_frag_coord_xy:
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_interpolated_input: {
      if (nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_POS)
         return false;
      unsigned offset = intr->intrinsic == nir_intrinsic_load_input ? 0 : 1;
      if (!nir_src_is_const(intr->src[offset]) ||
          nir_src_as_uint(intr->src[offset]) != 0) {
         *(bool *)data = false;
         return false;
      }
      component = nir_intrinsic_component(intr);
      break;
   }
   default:
      return false;
   }
   unsigned read = nir_def_components_read(&intr->def);
   if (intr->def.bit_size != 32 || component >= 4 ||
       ((read << component) & ~0x3u)) {
      *(bool *)data = false;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *xy = nir_u2f32(b, nir_load_pixel_coord(b));
   nir_def *values[4];
   for (unsigned i = 0; i < intr->def.num_components; ++i)
      values[i] = (read & BITFIELD_BIT(i))
                     ? nir_channel(b, xy, component + i)
                     : nir_undef(b, 1, 32);
   nir_def_rewrite_uses(&intr->def, nir_vec(b, values, intr->def.num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
apple9_collect_varyings(nir_shader *nir,
                        const struct agx_apple9_varying_layout *producer,
                        struct agx_apple9_varying_layout *layout,
                        const char **reason)
{
   bool fragment = nir->info.stage == MESA_SHADER_FRAGMENT;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != (fragment ? nir_intrinsic_load_interpolated_input
                                         : nir_intrinsic_store_output))
            continue;
         unsigned location = nir_intrinsic_io_semantics(intr).location;
         if (!fragment && location == VARYING_SLOT_POS)
            continue;
         if (fragment && location == VARYING_SLOT_POS) {
            *reason = "Apple9 fragment window-position input is not implemented";
            return false;
         }
         if (!nir_src_is_const(intr->src[1]) ||
             nir_src_as_uint(intr->src[1]) >= 32)
            goto unsupported;
         location += nir_src_as_uint(intr->src[1]);
         unsigned component = nir_intrinsic_component(intr);
         if ((fragment ? nir_intrinsic_dest_type(intr)
                       : nir_intrinsic_src_type(intr)) != nir_type_float32)
            goto unsupported;
         if (location < VARYING_SLOT_VAR0 || location >= VARYING_SLOT_VAR0 + 32 ||
             component + intr->num_components > 4)
            goto unsupported;
         unsigned mask = fragment ? nir_def_components_read(&intr->def)
                                  : nir_intrinsic_write_mask(intr);
         layout->mask[location - VARYING_SLOT_VAR0] |= mask << component;
      }
   }
   for (unsigned i = 0; i < 32; ++i) {
      if (producer && (layout->mask[i] & ~producer->mask[i])) {
         *reason = "Apple9 fragment input is not written by the vertex stage";
         return false;
      }
      layout->count += util_bitcount(layout->mask[i]);
   }
   if (producer)
      *layout = *producer;
   unsigned count = 0;
   for (unsigned i = 0; i < 32; ++i) {
      if (layout->mask[i] & ~0xf)
         goto unsupported;
      count += util_bitcount(layout->mask[i]);
   }
   if (layout->count != count || count > AGX_APPLE9_MAX_VARYING_COMPONENTS) {
      *reason = "Apple9 export publication currently supports twelve user scalars";
      return false;
   }
   return true;
unsupported:
   *reason = "Apple9 graphics requires constant-indexed FP32 user varyings";
   return false;
}

static bool
apple9_compile_graphics(nir_shader *nir, struct agx_shader_part *out,
                        const struct agx_apple9_vertex_layout *layout,
                        const struct agx_apple9_varying_layout *producer,
                        const struct agx_apple9_blend *blend,
                        const char **reason)
{
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   if (reason)
      *reason = NULL;
   nir_lower_io(nir, nir_var_shader_in | nir_var_shader_out, apple9_io_size,
                nir_lower_io_use_interpolated_input_intrinsics);
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      if (layout && layout->ignore_point_size)
         nir_remove_outputs(nir, MESA_SHADER_FRAGMENT, 0,
                            BITFIELD64_BIT(VARYING_SLOT_PSIZ));
      struct apple9_vertex_lower lower = {.layout = layout, .valid = true};
      nir_shader_intrinsics_pass(nir, apple9_lower_vertex_input,
                                 nir_metadata_control_flow, &lower);
      if (!lower.valid) {
         *reason = "Apple9 vertex inputs require supported naturally aligned formats";
         return false;
      }
   }
   /* Use the same structured mask/loop model as compute. Continuations become
    * masked regions with one backedge before instruction selection. */
   nir_lower_continue_constructs(nir);
   /* Frontends may supply vector NIR or pre-simplified scalar NIR. Expose
    * scalar constants to standard algebraic cleanup before expanding math
    * and output packing, rather than relying on a caller-specific pipeline.
    */
   nir_lower_vars_to_ssa(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   bool progress;
   do {
      progress = nir_opt_algebraic(nir);
      progress |= nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_dce(nir);
      progress |= nir_opt_cse(nir);
   } while (progress);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      struct apple9_color_lower color = {.blend = blend, .valid = true};
      nir_shader_intrinsics_pass(nir, apple9_lower_color,
                                 nir_metadata_control_flow, &color);
      if (!color.valid) {
         if (reason)
            *reason = "Apple9 render requires FP32 vec4 RT0 and supported ADD blend factors";
         return false;
      }
   }
   nir_shader_intrinsics_pass(nir, apple9_scalarize_graphics_ubo,
                              nir_metadata_control_flow, NULL);
   agx_nir_lower_apple9_math(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_opt_constant_folding(nir);
   nir_opt_copy_prop(nir);
   nir_opt_dce(nir);
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      bool valid = true;
      nir_shader_intrinsics_pass(nir, apple9_lower_window_position,
                                 nir_metadata_control_flow, &valid);
      if (!valid) {
         *reason = "Apple9 fragment window position currently supports XY only";
         return false;
      }
      nir_lower_alu_to_scalar(nir, NULL, NULL);
      nir_opt_copy_prop(nir);
      nir_opt_dce(nir);
   }
   if (nir->info.num_ssbos) {
      if (reason)
         *reason = "Apple9 render does not yet support SSBOs";
      return false;
   }
   struct apple9_buffer_map buffers = {0};
   if (!apple9_collect_buffer_map(nir, &buffers, reason))
      return false;
   bool valid_buffers = buffers.count <= 4;
   for (unsigned i = 0; i < buffers.count; ++i)
      valid_buffers &= buffers.resource[i].binding < (layout ? 64 : 32) &&
         buffers.resource[i].kind == AGX_APPLE9_COMPUTE_RESOURCE_UBO;
   if (!valid_buffers) {
      if (reason)
         *reason = "Apple9 graphics currently supports four buffer arguments per stage";
      return false;
   }
   if (getenv("AGX_APPLE9_TRACE"))
      nir_print_shader(nir, stderr);
   struct agx_apple9_varying_layout varyings = {0};
   if (!apple9_collect_varyings(nir, producer, &varyings, reason))
      return false;
   return apple9_compile_dag(nir, out, NULL, &varyings, reason);
}

bool
agx_compile_apple9_fragment(nir_shader *nir, struct agx_shader_part *out,
                            const char **reason)
{
   return nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, NULL, NULL, reason);
}

bool
agx_compile_apple9_fragment_blend(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, struct agx_shader_part *out,
   const char **reason)
{
   return nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, varyings, blend, reason);
}

bool
agx_compile_apple9_fragment_inputs(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   struct agx_shader_part *out, const char **reason)
{
   return nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, varyings, NULL, reason);
}

bool
agx_compile_apple9_vertex(nir_shader *nir, struct agx_shader_part *out,
                          const char **reason)
{
   return nir->info.stage == MESA_SHADER_VERTEX &&
          apple9_compile_graphics(nir, out, NULL, NULL, NULL, reason);
}

bool
agx_compile_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout,
   struct agx_shader_part *out, const char **reason)
{
   return nir->info.stage == MESA_SHADER_VERTEX &&
          apple9_compile_graphics(nir, out, layout, NULL, NULL, reason);
}

bool
agx_compile_apple9_vertex_prolog(nir_shader *nir, struct agx_shader_part *out,
                                 const char **reason)
{
   memset(out, 0, sizeof(*out));
   if (reason)
      *reason = "Apple9 vertex-fetch ABI is not implemented";
   return false;
}
