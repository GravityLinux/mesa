/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile_apple9.h"
#include "agx_apple9_ir.h"
#include "agx_nir.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_lower_blend.h"
#include "compiler/nir/nir_xfb_info.h"
#include "poly/cl/libpoly.h"
#include "compiler/nir/nir_format_convert.h"
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
   case nir_intrinsic_load_instance_id:
      base = 0xd8;
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

/* Native vectors need element alignment, independently of the index scale.
 * NIR still owns partial-store splitting, exact access extents and aliasing. */
static nir_mem_access_size_align
apple9_memory_format(nir_intrinsic_op op, uint8_t bytes, uint8_t bit_size,
                     uint32_t align_mul, uint32_t align_offset,
                     bool offset_is_const, enum gl_access_qualifier access,
                     const void *data)
{
   unsigned alignment = nir_combined_align(align_mul, align_offset);
   if (bit_size == 32 && alignment >= 4 && bytes >= 12)
      return (nir_mem_access_size_align){.num_components = MIN2(bytes / 4, 4),
                                         .bit_size = 32, .align = 4};
   if (bit_size == 32 && alignment >= 4 && bytes >= 8)
      return (nir_mem_access_size_align){.num_components = 2, .bit_size = 32, .align = 4};
   if (bit_size == 32 && bytes >= 4 &&
       (op == nir_intrinsic_load_ssbo || op == nir_intrinsic_load_ubo))
      return (nir_mem_access_size_align){.num_components = 1, .bit_size = 32,
                                         .align = 4, .shift = nir_mem_access_shift_method_scalar};
   unsigned size = MIN2(MIN2(alignment, bytes), 4);
   size = 1u << util_logbase2(size);
   return (nir_mem_access_size_align){.num_components = 1, .bit_size = size * 8,
                                      .align = size};
}

/* Keep format conversion in NIR's native half operations until selection.
 * The machine conversion preserves finite subnormals and signed zero, rounds
 * to nearest even, and returns a canonical quiet NaN for either NaN sign. */
static nir_def *
apple9_pack_half(nir_builder *b, nir_def *src)
{
   return nir_pack_half_2x16_split(b, src, nir_imm_float(b, 0));
}

static nir_def *
apple9_unpack_half(nir_builder *b, nir_def *src)
{
   return nir_channel(b, nir_unpack_half_2x16(b, src), 0);
}

/* Mesa's generic pack lowering leaves these scalar split operations for
 * backends with native halfword packing. Apple9 represents narrow integer
 * values in ordinary 32-bit GPRs, so expand the residual operations. */
static bool
apple9_lower_memory_pack(nir_builder *b, nir_alu_instr *alu, void *data)
{
   nir_def *value;
   b->cursor = nir_before_instr(&alu->instr);
   switch (alu->op) {
   case nir_op_uadd_sat: {
      /* Constant division and late algebraic optimization can introduce
       * saturating addition. Keep the carry test in the operand's width. */
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *c = nir_ssa_for_alu_src(b, alu, 1);
      nir_def *sum = nir_iadd(b, a, c);
      value = nir_bcsel(b, nir_ult(b, sum, a),
                       nir_imm_intN_t(b, -1, sum->bit_size), sum);
      break;
   }
   case nir_op_i2f32:
   case nir_op_u2f32: {
      nir_def *source = nir_ssa_for_alu_src(b, alu, 0);
      if (source->bit_size >= 32)
         return false;
      /* Algebraic optimization can fold a narrow integer extension into
       * the conversion. The machine conversion consumes a full 32-bit
       * register, so make that extension explicit after optimization. */
      value = alu->op == nir_op_i2f32
         ? nir_i2f32(b, nir_i2i32(b, source))
         : nir_u2f32(b, nir_u2u32(b, source));
      break;
   }
   case nir_op_extract_u8:
   case nir_op_extract_i8:
   case nir_op_extract_u16:
   case nir_op_extract_i16: {
      unsigned width = (alu->op == nir_op_extract_u8 || alu->op == nir_op_extract_i8) ? 8 : 16;
      bool is_signed = alu->op == nir_op_extract_i8 || alu->op == nir_op_extract_i16;
      nir_def *source = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *index = nir_ssa_for_alu_src(b, alu, 1);
      if (source->bit_size != 32 || alu->def.bit_size != 32)
         return false;
      nir_def *shift = nir_imul_imm(b, index, width);
      value = is_signed
         ? nir_ishr_imm(b, nir_ishl(b, source,
              nir_isub(b, nir_imm_int(b, 32 - width), shift)), 32 - width)
         : nir_iand_imm(b, nir_ushr(b, source, shift), BITFIELD_MASK(width));
      break;
   }
   case nir_op_pack_32_2x16_split:
      value = nir_ior(b, nir_u2u32(b, nir_ssa_for_alu_src(b, alu, 0)),
                        nir_ishl_imm(b, nir_u2u32(b, nir_ssa_for_alu_src(b, alu, 1)), 16));
      break;
   case nir_op_unpack_32_2x16_split_x:
      value = nir_u2u16(b, nir_ssa_for_alu_src(b, alu, 0));
      break;
   case nir_op_unpack_32_2x16_split_y:
      value = nir_u2u16(b, nir_ushr_imm(b, nir_ssa_for_alu_src(b, alu, 0), 16));
      break;
   default:
      return false;
   }
   nir_def_replace(&alu->def, value);
   return true;
}

/* The IR offset is a wrapping 32-bit byte address. Convert the completed
 * expression to the hardware format's index only after that arithmetic.
 * Access alignment is established by nir_lower_mem_access_bit_sizes. */
static bool
apple9_element_index(nir_intrinsic_instr *intr, nir_def *offset,
                     unsigned element_size, unsigned components,
                     nir_scalar *index, unsigned *index_scale, unsigned *index_add)
{
   if (offset->bit_size != 32 || (element_size != 1 && element_size != 2 && element_size != 4))
      return false;
   unsigned stride = components == 1 ? 1 : components == 2 ? 2 : 4;
   unsigned bytes = element_size * stride;
   /* The vector format scales its index by the full padded vector size.
    * Never silently discard low address bits if a later pass changes the
    * alignment established by memory legalization. */
   if (components > 1 && nir_intrinsic_align(intr) < bytes)
      return false;
   nir_builder b = nir_builder_at(nir_before_instr(&intr->instr));
   nir_scalar scalar = apple9_chase_trivial(nir_get_scalar(offset, 0));
   uint32_t constant;
   if (components == 1 && apple9_const_u32(scalar, &constant)) {
      if (constant % bytes)
         return false;
      *index = scalar;
      *index_scale = 0;
      *index_add = constant / bytes;
   } else {
      nir_def *element = bytes == 1 ? offset : nir_ushr_imm(&b, offset, util_logbase2(bytes));
      *index = nir_get_scalar(element, 0);
      *index_scale = stride;
      *index_add = 0;
   }
   return true;
}

/* Conservative lower bounds for unsigned expressions. These establish when
 * a negative displacement cannot underflow before widening the address. */
static uint32_t
apple9_address_lower_bound(nir_scalar value)
{
   value = apple9_chase_trivial(value);
   uint32_t constant;
   if (apple9_const_u32(value, &constant))
      return constant;
   if (!nir_scalar_is_alu(value) ||
       (nir_scalar_alu_op(value) != nir_op_ior &&
        nir_scalar_alu_op(value) != nir_op_umax))
      return 0;
   uint32_t minimum = 0;
   for (unsigned s = 0; s < 2; ++s)
      if (apple9_const_u32(nir_scalar_chase_alu_src(value, s), &constant))
         minimum = MAX2(minimum, constant);
   return minimum;
}

static uint32_t
apple9_address_upper_bound(nir_shader *nir, struct hash_table *bounds, nir_scalar value)
{
   uint32_t maximum = nir_unsigned_upper_bound(nir, bounds, value);
   if (nir_scalar_is_alu(value) && nir_scalar_alu_op(value) == nir_op_iadd) {
      for (unsigned s = 0; s < 2; ++s) {
         uint32_t constant;
         nir_scalar other = apple9_chase_trivial(nir_scalar_chase_alu_src(value, 1 - s));
         if (apple9_const_u32(nir_scalar_chase_alu_src(value, s), &constant) &&
             (int32_t)constant < 0 &&
             apple9_address_lower_bound(other) >= -(int64_t)(int32_t)constant) {
            maximum = MIN2(maximum, nir_unsigned_upper_bound(nir, bounds, other) + constant);
         }
      }
   }
   return maximum;
}

/* Extract only address arithmetic whose 32-bit wrap semantics are preserved.
 * Unknown sums stay in the GPR; a folded shift masks its input if it can wrap.
 * Reserving the trailing component bytes permits scalar selection from a
 * vector access without changing its original 64-bit component addressing. */
static bool
apple9_memory_address(nir_shader *nir, nir_intrinsic_instr *intr, nir_def *offset,
                      unsigned element_size, unsigned components, nir_scalar *index,
                      uint8_t *shift, int16_t *displacement)
{
   if (offset->bit_size != 32 || offset->num_components != 1 ||
       (element_size != 1 && element_size != 2 && element_size != 4))
      return false;
   nir_builder b = nir_builder_at(nir_before_instr(&intr->instr));
   nir_scalar value = apple9_chase_trivial(nir_get_scalar(offset, 0));
   int max_offset = INT16_MAX - (components - 1) * element_size;
   *shift = 0;
   *displacement = 0;
   struct hash_table *bounds = _mesa_pointer_hash_table_create(NULL);
   if (!bounds)
      return false;
   /* GLSL buffer indexing uses amul. It has the same wrapping product
    * semantics as imul; matching both keeps this reachable from API shaders.
    * Peel from the outside in, accumulating scale only while the remaining
    * expression can still be evaluated as a 32-bit value without overflow. */
   for (unsigned depth = 0; depth < 8; ++depth) {
      uint32_t constant;
      if (apple9_const_u32(value, &constant)) {
         int64_t total = *displacement + ((int64_t)constant << *shift);
         if (total <= max_offset) {
            *displacement = total;
            *shift = 0;
            value = nir_get_scalar(nir_imm_int(&b, 0), 0);
         }
         break;
      }
      if (!nir_scalar_is_alu(value))
         break;
      nir_op op = nir_scalar_alu_op(value);
      bool peeled = false;
      if (op == nir_op_iadd) {
         for (unsigned s = 0; s < 2; ++s) {
            nir_scalar term = apple9_chase_trivial(nir_scalar_chase_alu_src(value, s));
            nir_scalar other = apple9_chase_trivial(nir_scalar_chase_alu_src(value, 1 - s));
            if (!apple9_const_u32(term, &constant))
               continue;
            int64_t add = (int32_t)constant;
            int64_t total = *displacement + add * (1u << *shift);
            if (total < INT16_MIN || total > max_offset)
               break;
            bool safe = add >= 0 &&
               !nir_addition_might_overflow(nir, bounds, other, constant);
            if (add < 0 && apple9_address_lower_bound(other) >= -add)
               safe = true;
            if (safe) {
               *displacement = total;
               value = other;
               peeled = true;
            }
            break;
         }
      } else if (op == nir_op_ishl || op == nir_op_imul || op == nir_op_amul) {
         for (unsigned s = 0; s < (op == nir_op_ishl ? 1 : 2); ++s) {
            nir_scalar term = nir_scalar_chase_alu_src(value, op == nir_op_ishl ? 1 : s);
            nir_scalar other = apple9_chase_trivial(nir_scalar_chase_alu_src(value, op == nir_op_ishl ? 0 : 1 - s));
            if (!apple9_const_u32(term, &constant))
               continue;
            unsigned amount = op == nir_op_ishl ? (constant & 31) :
               util_is_power_of_two_nonzero(constant) ? util_logbase2(constant) : 32;
            if (amount + *shift > 4)
               break;
            uint32_t maximum = UINT32_MAX >> amount;
            *shift += amount;
            if (apple9_address_upper_bound(nir, bounds, other) > maximum) {
               nir_def *base = nir_channel(&b, other.def, other.comp);
               value = nir_get_scalar(nir_iand_imm(&b, base, maximum), 0);
               /* The mask must apply after the remaining arithmetic. */
            } else {
               value = other;
               peeled = true;
            }
            break;
         }
      }
      if (!peeled)
         break;
   }
   *index = value;
   _mesa_hash_table_destroy(bounds, NULL);
   return true;
}

struct apple9_scalar_load {
   nir_intrinsic_instr *intr;
   nir_block *block;
   nir_scalar index;
   unsigned argument;
   unsigned component;
   uint8_t index_shift;
   int16_t byte_offset;
   unsigned bit_size;
};

struct apple9_buffer_store {
   nir_intrinsic_instr *intr;
   nir_block *block;
   nir_scalar index;
   unsigned argument;
   unsigned components;
   uint8_t index_shift;
   int16_t byte_offset;
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

/* One compiler-private sampler follows the 32 API sampler bindings. It is
 * packed after the live API samplers and uses nearest filtering without API
 * LOD clamps, bias, or comparison state. */
#define APPLE9_FETCH_SAMPLER 32

static bool
apple9_texture_uses_fetch_sampler(const nir_tex_instr *tex)
{
   return tex->sampler_index == APPLE9_FETCH_SAMPLER;
}

static bool
apple9_texture_supported(const nir_tex_instr *tex)
{
   /* Offset filtering is expanded with sampler state before selection. */
   if (nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0)
      return false;
   bool explicit_lod = tex->op == nir_texop_txl || tex->op == nir_texop_txb ||
                       tex->op == nir_texop_txf;
   bool gradient = tex->op == nir_texop_txd;
   bool array = tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_2D;
   bool volume = array || tex->sampler_dim == GLSL_SAMPLER_DIM_3D ||
                 tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   if ((!explicit_lod && !gradient && tex->op != nir_texop_tex) ||
       (!volume && tex->sampler_dim != GLSL_SAMPLER_DIM_2D) || (tex->is_array && !array) ||
       (tex->is_shadow && (tex->sampler_dim == GLSL_SAMPLER_DIM_3D || gradient)) ||
       (volume && gradient) ||
       tex->coord_components != (volume ? 3 : 2) || tex->def.bit_size != 32 ||
       (tex->def.num_components != 4 && !(tex->is_shadow && tex->def.num_components == 1)) ||
       (tex->dest_type != nir_type_float32 && tex->dest_type != nir_type_int32 &&
        tex->dest_type != nir_type_uint32) ||
       tex->num_srcs != (gradient ? 3 : explicit_lod ? 2 : 1) + tex->is_shadow || tex->texture_index >= 32 ||
       (tex->sampler_index > APPLE9_FETCH_SAMPLER))
      return false;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int lod = nir_tex_instr_src_index(tex, tex->op == nir_texop_txb
                                             ? nir_tex_src_bias : nir_tex_src_lod);
   int packed_lod = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
   if (packed_lod >= 0) {
      if (!explicit_lod || lod >= 0)
         return false;
      lod = packed_lod;
   }
   if (gradient) {
      for (enum nir_tex_src_type type = nir_tex_src_ddx;
           type <= nir_tex_src_ddy; ++type) {
         int source = nir_tex_instr_src_index(tex, type);
         if (source < 0 || tex->src[source].src.ssa->bit_size != 32 ||
             tex->src[source].src.ssa->num_components != 2)
            return false;
      }
   }
   int comparator = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   if (tex->is_shadow && (comparator < 0 || tex->src[comparator].src.ssa->bit_size != 32))
      return false;
   return coord >= 0 && tex->src[coord].src.ssa->bit_size == 32 &&
          (!explicit_lod || (lod >= 0 && tex->src[lod].src.ssa->bit_size == 32 &&
                      tex->src[lod].src.ssa->num_components == 1));
}

static bool
apple9_instruction_is_in_subset(nir_instr *instr, bool graphics)
{
   switch (instr->type) {
   case nir_instr_type_tex:
      return graphics && apple9_texture_supported(nir_instr_as_tex(instr));
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
      case nir_op_f2f16:
      case nir_op_f2f32:
      case nir_op_pack_half_2x16:
      case nir_op_pack_half_2x16_split:
      case nir_op_pack_unorm_2x16:
      case nir_op_pack_unorm_4x8:
      case nir_op_unpack_unorm_2x16:
      case nir_op_unpack_snorm_2x16:
      case nir_op_unpack_unorm_4x8:
      case nir_op_unpack_snorm_4x8:
      case nir_op_unpack_half_2x16:
      case nir_op_i2i8:
      case nir_op_i2i16:
      case nir_op_i2i32:
      case nir_op_u2u8:
      case nir_op_u2u16:
      case nir_op_u2u32:
      case nir_op_iadd:
      case nir_op_isub:
      case nir_op_imul:
      case nir_op_imul_high:
      case nir_op_umul_high:
      case nir_op_imul_2x32_64:
      case nir_op_umul_2x32_64:
      case nir_op_unpack_64_2x32_split_x:
      case nir_op_unpack_64_2x32_split_y:
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
      case nir_op_fsat:
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
      if (graphics &&
          (op == nir_intrinsic_load_vertex_id ||
           op == nir_intrinsic_load_vertex_id_zero_base ||
           op == nir_intrinsic_load_instance_id ||
           op == nir_intrinsic_load_pixel_coord ||
           op == nir_intrinsic_load_frag_coord_w ||
           op == nir_intrinsic_load_frag_coord_z ||
           op == nir_intrinsic_load_point_coord ||
           op == nir_intrinsic_load_front_face || op == nir_intrinsic_ddx ||
           op == nir_intrinsic_ddy || op == nir_intrinsic_ddx_fine ||
           op == nir_intrinsic_ddy_fine ||
           op == nir_intrinsic_load_barycentric_pixel ||
           op == nir_intrinsic_load_barycentric_centroid ||
           op == nir_intrinsic_load_interpolated_input ||
           op == nir_intrinsic_load_input ||
           op == nir_intrinsic_load_tile_pixel_agx ||
           op == nir_intrinsic_load_local_pixel_agx ||
           op == nir_intrinsic_load_sample_mask_in ||
           op == nir_intrinsic_sample_mask_agx ||
           op == nir_intrinsic_store_zs_agx ||
           op == nir_intrinsic_store_output ||
           op == nir_intrinsic_image_store_block_agx ||
           op == nir_intrinsic_store_local_pixel_agx ||
           op == nir_intrinsic_demote || op == nir_intrinsic_demote_if))
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
             op == nir_intrinsic_load_preamble ||
             op == nir_intrinsic_store_preamble ||
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
   struct agx_apple9_block *exit_block;
   unsigned depth;
   unsigned mask_depth;
};

struct apple9_dag_lower {
   struct agx_apple9_vir_program program;
   nir_shader *nir;
   uint32_t *ssa_to_vreg;
   struct agx_apple9_block **nir_blocks;
   unsigned ssa_map_count;
   uint32_t system_vreg[256];
   uint32_t zero_vreg;
   struct agx_apple9_block *binding_pointer_block;
   uint32_t binding_pointer[AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
   uint32_t perspective_reciprocal;
   struct agx_apple9_interp_mask linear_mask, flat_mask;
   bool perspective_ready;
   bool reads_z;
   bool writes_z;
   bool reads_point_coord;
   const struct agx_apple9_varying_layout *varyings;
   unsigned position_mask;
   bool color_stores;
   bool tile_read;
   bool tile_write;
   bool tile_access;
   bool tile_coords;
   uint32_t coverage_control;
   uint32_t coverage_mask;
   struct apple9_scalar_load *loads;
   unsigned load_count;
   struct apple9_buffer_atomic *atomics;
   unsigned atomic_count;
   unsigned argument_base;
   unsigned preamble_base;
   uint32_t texture_mask, sampler_mask, image_mask;
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
apple9_current_load_flags(struct apple9_dag_lower *lower)
{
   assert(lower->active_load_instruction_count > 0);
   assert(lower->active_emitted_load_count <
          lower->active_load_instruction_count);
   assert(lower->emitted_load_count < lower->load_instruction_count);

   uint8_t flags = lower->emitted_load_count + 1 < lower->load_instruction_count
                      ? AGX_APPLE9_DEVICE_LOAD_HAS_NEXT
                      : 0;

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

   return apple9_dag_emit(lower, AGX_APPLE9_VIR_IMM,
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
apple9_graphics_buffer_address(struct apple9_dag_lower *lower, unsigned argument)
{
   assert(lower->nir->info.stage != MESA_SHADER_COMPUTE);
   assert(argument < ARRAY_SIZE(lower->binding_pointer));
   if (lower->binding_pointer_block != lower->program.current_block) {
      lower->binding_pointer_block = lower->program.current_block;
      memset(lower->binding_pointer, 0xff, sizeof(lower->binding_pointer));
   }
   if (lower->binding_pointer[argument] != AGX_APPLE9_VREG_INVALID)
      return lower->binding_pointer[argument];

   /* The draw's binding table is immutable even when its buffers are
    * writable. Cache only that pointer, and only within its defining block. */
   uint32_t index = apple9_dag_imm(lower, argument);
   if (index == AGX_APPLE9_VREG_INVALID)
      return index;
   const struct agx_apple9_device_load_contract contract = {
      .index_kind = AGX_APPLE9_DEVICE_LOAD_INDEX_RETAINED_GPR,
      .flags = AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
      .raw_token = AGX_APPLE9_DEVICE_LOAD_TOKEN_5101,
   };
   uint32_t address = agx_apple9_vir_emit_device_load_vector(
      &lower->program, lower->argument_base, index, 2, &contract);
   if (address == AGX_APPLE9_VREG_INVALID ||
       !agx_apple9_vir_set_device_load_contract(
          &lower->program, address, AGX_APPLE9_DEVICE_LOAD_HAS_NEXT,
          AGX_APPLE9_SCOREBOARD_SLOT_AUTO))
      return AGX_APPLE9_VREG_INVALID;
   lower->binding_pointer[argument] = address;
   return address;
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
         (system.selector == 0xdd || system.selector == 0xd8)
                                 ? AGX_APPLE9_ENC_GET_DRAW_ID
         : system.zext16         ? AGX_APPLE9_ENC_GET_SR_ZEXT16
                                 : AGX_APPLE9_ENC_GET_SR;
      uint32_t immediate =
         system.global_id
            ? system.selector - 0xa0
            : system.selector | (system.zext16 ? 0 : (0x10u << 8));

      /* Keep the system value in SSA; allocation applies the six-bit
       * destination field shared by the short SR forms. */
      lower->system_vreg[system.selector] =
         apple9_dag_emit(lower, op, encoding, NULL, 0, immediate);
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

/* Both dispatch modes publish a pointer to three 32-bit group counts. */
static uint32_t
apple9_dag_num_workgroups(struct apple9_dag_lower *lower, unsigned component)
{
   if (component >= 3) {
      lower->reason = "Apple9 load_num_workgroups has an invalid component";
      return AGX_APPLE9_VREG_INVALID;
   }

   return apple9_dag_hidden_load(lower, 0, component);
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

   return apple9_dag_emit(lower, AGX_APPLE9_VIR_SELECT,
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
   } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
              nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_front_face &&
              lower->nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* The word read of selector 0xc5 reports back-facing. Convert to
       * NIR's front-facing Boolean, as in the upstream AGX convention. */
      uint32_t facing = apple9_dag_system(lower,
         (struct apple9_system_source){.selector = 0xc5});
      uint32_t zero = apple9_dag_zero(lower);
      value = apple9_emit_dag_select_raw(lower, zero, facing,
         zero, apple9_dag_imm(lower, 1), AGX_APPLE9_SELECT_ULT);
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
   if (source == AGX_APPLE9_VREG_INVALID)
      return source;
   amount &= 31;
   if (amount == 0)
      return source;
   return apple9_dag_emit(lower,
      op == nir_op_ishr ? AGX_APPLE9_VIR_ISHR :
      op == nir_op_ushr ? AGX_APPLE9_VIR_USHR : AGX_APPLE9_VIR_ISHL,
      op == nir_op_ishr ? AGX_APPLE9_ENC_SHIFT_EXTENDED
                       : AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE,
      &source, 1, amount);
}

static uint32_t
apple9_dag_shift_variable(struct apple9_dag_lower *lower, nir_op op,
                          uint32_t source, nir_scalar shift)
{
   uint32_t amount = apple9_lower_dag_scalar(lower, shift);
   if (source == AGX_APPLE9_VREG_INVALID || amount == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;

   /* Hardware shifts saturate outside 0..31, whereas NIR masks the amount.
    * Reuse an existing five-bit AND when the frontend already supplied it. */
   bool bounded = false;
   if (nir_def_instr_type(shift.def) == nir_instr_type_alu &&
       nir_scalar_alu_op(shift) == nir_op_iand) {
      for (unsigned s = 0; s < 2; ++s) {
         uint32_t mask;
         if (apple9_const_u32(nir_scalar_chase_alu_src(shift, s), &mask) &&
             mask <= 31)
            bounded = true;
      }
   }
   if (!bounded) {
      uint32_t sources[] = {amount, apple9_dag_imm(lower, 31)};
      amount = apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
                               AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
   }
   if (amount == AGX_APPLE9_VREG_INVALID)
      return amount;
   uint32_t sources[] = {source, amount};
   return apple9_dag_emit(lower,
      op == nir_op_ishr ? AGX_APPLE9_VIR_ISHR :
      op == nir_op_ushr ? AGX_APPLE9_VIR_USHR : AGX_APPLE9_VIR_ISHL,
      op == nir_op_ishr ? AGX_APPLE9_ENC_SHIFT_ARITH_REGISTER
                       : AGX_APPLE9_ENC_SHIFT_LOGICAL_REGISTER,
      sources, 2, 0);
}

/* Returns the packed user scalar index, excluding the four position words. */
static int
apple9_varying_index(const struct agx_apple9_varying_layout *layout,
                     unsigned location, unsigned component)
{
   if (!layout || !agx_apple9_varying_supported(location) || component >= 4)
      return -1;
   unsigned semantic = location;
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
   bool flat = intr->intrinsic == nir_intrinsic_load_input;
   nir_intrinsic_instr *bary = flat ? NULL : nir_src_as_intrinsic(intr->src[0]);
   unsigned offset = flat ? 0 : 1;
   bool centroid = bary && bary->intrinsic == nir_intrinsic_load_barycentric_centroid;
   bool linear =
      bary && nir_intrinsic_interp_mode(bary) == INTERP_MODE_NOPERSPECTIVE;
   /* Legacy color inputs retain NONE for glShadeModel. The state tracker
    * lowers GL_FLAT to load_input because caps.flatshade is false; remaining
    * NONE inputs therefore use the same perspective contract as SMOOTH. */
   const unsigned component = nir_intrinsic_component(intr) + scalar.comp;
   if (lower->nir->info.stage != MESA_SHADER_FRAGMENT ||
       scalar.def->bit_size != 32 || component >= 4 ||
       !nir_src_is_const(intr->src[offset]) ||
       (!flat &&
        (!bary || (!centroid && bary->intrinsic != nir_intrinsic_load_barycentric_pixel) ||
         (!linear && nir_intrinsic_interp_mode(bary) != INTERP_MODE_SMOOTH &&
          nir_intrinsic_interp_mode(bary) != INTERP_MODE_NONE)))) {
      lower->reason =
         "Apple9 fragment input requires center/centroid smooth, linear, or flat 32-bit user varyings";
      return AGX_APPLE9_VREG_INVALID;
   }
   unsigned location = nir_intrinsic_io_semantics(intr).location +
                       nir_src_as_uint(intr->src[offset]);
   int index = apple9_varying_index(lower->varyings, location, component);
   if (index < 0) {
      lower->reason = "Apple9 fragment input is not written by the vertex stage";
      return AGX_APPLE9_VREG_INVALID;
   }
   if (flat) {
      agx_apple9_interp_mask_set(&lower->flat_mask, index);
      return agx_apple9_vir_emit_iter_flat(&lower->program, index + 1);
   }
   uint32_t position = centroid ? apple9_lower_dag_scalar(
      lower, nir_get_scalar(&bary->def, 0)) : AGX_APPLE9_VREG_INVALID;
   enum agx_apple9_encoding encoding = centroid ? AGX_APPLE9_ENC_ITER_COORD
                                               : AGX_APPLE9_ENC_ITER;
   if (linear) {
      agx_apple9_interp_mask_set(&lower->linear_mask, index);
      return apple9_dag_emit(lower, AGX_APPLE9_VIR_ITER, encoding,
                             centroid ? &position : NULL, centroid, index + 1);
   }
   uint32_t reciprocal;
   if (centroid || !lower->perspective_ready) {
      uint32_t denominator = apple9_dag_emit(
         lower, AGX_APPLE9_VIR_ITER, encoding,
         centroid ? &position : NULL, centroid, 0);
      reciprocal = apple9_dag_emit(lower, AGX_APPLE9_VIR_FRCP,
         AGX_APPLE9_ENC_FLOAT_SPECIAL, &denominator, 1, 3);
      if (!centroid) {
         lower->perspective_reciprocal = reciprocal;
         lower->perspective_ready = true;
      }
   } else {
      reciprocal = lower->perspective_reciprocal;
   }
   uint32_t coefficient = apple9_dag_emit(
      lower, AGX_APPLE9_VIR_ITER, encoding,
      centroid ? &position : NULL, centroid, index + 1);
   /* Keep raw varyings available to homogeneous clipping. Native shade-7
    * coefficients pair with a coefficient-aware projective multiply, which
    * handles the rasterizer's primitive-constant representation. */
   uint32_t src[] = {coefficient, reciprocal};
   return apple9_dag_emit(lower, AGX_APPLE9_VIR_FMUL_PROJECT,
                          AGX_APPLE9_ENC_FLOAT2_PROJECT, src, 2, index + 1);
}

static uint32_t
apple9_lower_dag_scalar(struct apple9_dag_lower *lower, nir_scalar scalar)
{
   scalar = apple9_chase_trivial(scalar);
   if (scalar.def->bit_size == 1)
      return apple9_lower_bool_scalar(lower, scalar);
   bool wide_product = scalar.def->bit_size == 64 &&
      nir_def_instr_type(scalar.def) == nir_instr_type_alu &&
      (nir_scalar_alu_op(scalar) == nir_op_imul_2x32_64 ||
       nir_scalar_alu_op(scalar) == nir_op_umul_2x32_64);
   if ((scalar.def->bit_size != 8 && scalar.def->bit_size != 16 &&
        scalar.def->bit_size != 32 && !wide_product) ||
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
   } else if (nir_def_instr_type(scalar.def) == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(nir_def_instr(scalar.def));
      if (!apple9_texture_supported(tex)) {
         lower->reason = "Apple9 requires a non-shadow FP32 2D sample or integer texel fetch";
         return AGX_APPLE9_VREG_INVALID;
      }
      int coord_source = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      uint32_t coords[3];
      for (unsigned c = 0; c < tex->coord_components; ++c)
         coords[c] = apple9_lower_dag_scalar(
            lower, nir_get_scalar(tex->src[coord_source].src.ssa, c));
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         uint32_t major = agx_apple9_vir_emit_cube(&lower->program, coords, 0);
         uint32_t u = agx_apple9_vir_emit_cube(&lower->program, coords, 1);
         uint32_t v = agx_apple9_vir_emit_cube(&lower->program, coords, 2);
         if (major == AGX_APPLE9_VREG_INVALID || u == AGX_APPLE9_VREG_INVALID ||
             v == AGX_APPLE9_VREG_INVALID)
            return AGX_APPLE9_VREG_INVALID;
         uint32_t reciprocal = apple9_dag_emit(lower, AGX_APPLE9_VIR_FRCP,
            AGX_APPLE9_ENC_FLOAT_SPECIAL, &major, 1, 2);
         uint32_t sources[] = {u, reciprocal, apple9_dag_imm(lower, 0x3f000000)};
         coords[0] = apple9_dag_emit(lower, AGX_APPLE9_VIR_FMA,
            AGX_APPLE9_ENC_FLOAT3_EXTENDED, sources, 3, 0);
         sources[0] = v;
         coords[1] = apple9_dag_emit(lower, AGX_APPLE9_VIR_FMA,
            AGX_APPLE9_ENC_FLOAT3_EXTENDED, sources, 3, 0);
         /* The face operation only defines the low half of its second GPR. */
         sources[0] = major + 1;
         sources[1] = apple9_dag_imm(lower, 0xffff);
         coords[2] = apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
            AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
      }
      uint32_t layer_or_face = tex->coord_components == 3 ? coords[2] : 0;
      if (tex->is_shadow) {
         int comparator = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
         coords[2] = apple9_lower_dag_scalar(lower,
            nir_get_scalar(tex->src[comparator].src.ssa, 0));
      }
      if (lower->nir->info.stage == MESA_SHADER_FRAGMENT &&
          !lower->tile_access && !lower->tile_read &&
          !agx_apple9_vir_emit_side_effect(&lower->program,
             AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0, 0x600))
         return AGX_APPLE9_VREG_INVALID;
      lower->tile_access |= lower->nir->info.stage == MESA_SHADER_FRAGMENT;
      unsigned texture = util_bitcount(lower->texture_mask &
                                       BITFIELD_MASK(tex->texture_index));
      uint32_t result;
      if (tex->op == nir_texop_txd) {
         uint32_t sources[6] = {coords[0], coords[1]};
         for (unsigned derivative = 0; derivative < 2; ++derivative) {
            int source = nir_tex_instr_src_index(
               tex, derivative ? nir_tex_src_ddy : nir_tex_src_ddx);
            for (unsigned c = 0; c < 2; ++c)
               sources[2 + 2 * derivative + c] = apple9_lower_dag_scalar(
                  lower, nir_get_scalar(tex->src[source].src.ssa, c));
         }
         result = agx_apple9_vir_emit_texture_grad(
            &lower->program, sources, texture,
            util_bitcount(lower->sampler_mask & BITFIELD_MASK(tex->sampler_index)));
      } else if (tex->op == nir_texop_txl || tex->op == nir_texop_txb ||
                 tex->op == nir_texop_txf) {
         bool bias = tex->op == nir_texop_txb;
         int lod_source = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
         if (lod_source < 0) {
            lower->reason = "Apple9 texture LOD was not packed before selection";
            return AGX_APPLE9_VREG_INVALID;
         }
         uint32_t packed_lod = apple9_lower_dag_scalar(
            lower, nir_get_scalar(tex->src[lod_source].src.ssa, 0));
         bool fetch = tex->op == nir_texop_txf;
         if (tex->coord_components == 3 || tex->is_shadow) {
            if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE || tex->is_array) {
               uint32_t fields[] = {packed_lod, layer_or_face};
               packed_lod = apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR,
                  AGX_APPLE9_ENC_LOGIC_EXTENDED, fields, 2, 0);
            }
            result = agx_apple9_vir_emit_texture_volume(&lower->program, coords,
               packed_lod, texture,
               util_bitcount(lower->sampler_mask & BITFIELD_MASK(tex->sampler_index)),
               tex->is_array ? 2 : tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? 1 :
                  tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 0,
               bias, tex->is_shadow);
         } else
            result = agx_apple9_vir_emit_texture_lod(
            &lower->program, coords, packed_lod, texture,
            util_bitcount(lower->sampler_mask & BITFIELD_MASK(tex->sampler_index)),
            bias);
         if (fetch && result != AGX_APPLE9_VREG_INVALID)
            lower->program.instructions[lower->program.instruction_count - 1]->texture_fetch = true;
      } else {
         uint32_t one = apple9_dag_imm(lower, 0x3f800000);
         result = agx_apple9_vir_emit_texture_sample(
            &lower->program, coords, one, texture,
            util_bitcount(lower->sampler_mask & BITFIELD_MASK(tex->sampler_index)));
      }
      if (result == AGX_APPLE9_VREG_INVALID) {
         lower->reason = "could not emit Apple9 texture sample";
         return result;
      }
      /* Texture returns are a packed tuple, not four fixed channel slots.
       * Preserve logical swizzles while allocating only the channels read
       * by NIR. In particular, a .zw use returns two adjacent registers. */
      unsigned mask = nir_def_components_read(scalar.def) & 0xf;
      assert(mask & BITFIELD_BIT(scalar.comp));
      struct agx_apple9_vir_instr *sample =
         lower->program.instructions[lower->program.instruction_count - 1];
      sample->texture_result_mask = mask;
      sample->dest_components = util_bitcount(mask);
      unsigned packed = 0;
      for (unsigned c = 0; c < 4; ++c) {
         if (mask & BITFIELD_BIT(c))
            lower->ssa_to_vreg[scalar.def->index * 4 + c] = result + packed++;
      }
      value = lower->ssa_to_vreg[key];
   } else {
      struct apple9_system_source system;
      const bool subgroup_size =
         nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
         nir_def_as_intrinsic(scalar.def)->intrinsic ==
            nir_intrinsic_load_subgroup_size;
      nir_intrinsic_instr *intr = nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic
         ? nir_def_as_intrinsic(scalar.def) : NULL;
      if (intr && (intr->intrinsic == nir_intrinsic_ddx ||
                   intr->intrinsic == nir_intrinsic_ddy ||
                   intr->intrinsic == nir_intrinsic_ddx_fine ||
                   intr->intrinsic == nir_intrinsic_ddy_fine)) {
         if (lower->nir->info.stage != MESA_SHADER_FRAGMENT)
            return AGX_APPLE9_VREG_INVALID;
         uint32_t source = apple9_lower_dag_scalar(lower,
            nir_get_scalar(intr->src[0].ssa, scalar.comp));
         if (source != AGX_APPLE9_VREG_INVALID)
            value = apple9_dag_emit(lower, AGX_APPLE9_VIR_DERIVATIVE,
               AGX_APPLE9_ENC_DERIVATIVE, &source, 1,
               intr->intrinsic == nir_intrinsic_ddy || intr->intrinsic == nir_intrinsic_ddy_fine);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
          nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_frag_coord_w) {
         /* Coefficient zero is interpolated reciprocal clip W. */
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_ITER,
                                 AGX_APPLE9_ENC_ITER, NULL, 0, 0);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_frag_coord_z) {
         lower->reads_z = true;
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_ITER,
                                 AGX_APPLE9_ENC_ITER, NULL, 0,
                                 1 + (lower->varyings ? lower->varyings->count : 0));
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_point_coord) {
         if (scalar.comp >= 2)
            return AGX_APPLE9_VREG_INVALID;
         lower->reads_point_coord = true;
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_ITER,
                                 AGX_APPLE9_ENC_ITER, NULL, 0,
                                 1 + lower->varyings->count + lower->reads_z + scalar.comp);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
          (nir_def_as_intrinsic(scalar.def)->intrinsic ==
              nir_intrinsic_load_interpolated_input ||
           nir_def_as_intrinsic(scalar.def)->intrinsic ==
              nir_intrinsic_load_input)) {
         value = apple9_lower_interpolated_input(lower, scalar);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 (nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_local_pixel_agx ||
                  nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_tile_pixel_agx)) {
         nir_intrinsic_instr *intr = nir_def_as_intrinsic(scalar.def);
         if (lower->nir->info.stage != MESA_SHADER_FRAGMENT ||
             scalar.comp >= intr->num_components || intr->num_components > 4 ||
             intr->def.bit_size != 32 ||
             (nir_intrinsic_format(intr) != PIPE_FORMAT_R32_UINT &&
              nir_intrinsic_format(intr) != PIPE_FORMAT_R32G32_UINT &&
              nir_intrinsic_format(intr) != PIPE_FORMAT_R32G32B32_UINT &&
              nir_intrinsic_format(intr) != PIPE_FORMAT_R32G32B32A32_UINT) ||
             (nir_intrinsic_base(intr) & 3) ||
             (intr->src[0].ssa->bit_size != 16 &&
              intr->src[0].ssa->bit_size != 32))
            return AGX_APPLE9_VREG_INVALID;
         assert(lower->tile_access && lower->tile_read);
         const bool coords =
            intr->intrinsic == nir_intrinsic_load_tile_pixel_agx;
         if (coords && (!nir_src_is_const(intr->src[0]) ||
                        !nir_src_as_uint(intr->src[0]) ||
                        nir_src_as_uint(intr->src[0]) > 15 ||
                        intr->src[1].ssa->bit_size != 32 ||
                        intr->src[1].ssa->num_components != 1))
            return AGX_APPLE9_VREG_INVALID;
         uint32_t source = apple9_lower_dag_scalar(
            lower, nir_get_scalar(intr->src[coords ? 1 : 0].ssa, 0));
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_TILE_LOAD,
                                 coords ? AGX_APPLE9_ENC_TILE_LOAD_COORDS
                                        : AGX_APPLE9_ENC_TILE_LOAD_MASK,
                                 &source, 1,
                                 nir_intrinsic_base(intr) / 4 + scalar.comp);
         if (value != AGX_APPLE9_VREG_INVALID) {
            struct agx_apple9_vir_instr *load =
               lower->program.instructions[lower->program.instruction_count - 1];
            load->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_AUTO;
            if (coords)
               load->tile_sample_mask = nir_src_as_uint(intr->src[0]);
         }
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic ==
                    nir_intrinsic_load_barycentric_centroid) {
         uint32_t coverage = apple9_dag_emit(lower, AGX_APPLE9_VIR_GET_SR,
            AGX_APPLE9_ENC_GET_COVERAGE, NULL, 0, 0x10c2);
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_CENTROID_POSITION,
            AGX_APPLE9_ENC_CENTROID_POSITION, &coverage, 1, 0);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic ==
                    nir_intrinsic_load_sample_mask_in) {
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_GET_SR,
            AGX_APPLE9_ENC_GET_COVERAGE, NULL, 0, 0x10c2);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic ==
                    nir_intrinsic_load_preamble) {
         nir_intrinsic_instr *intr = nir_def_as_intrinsic(scalar.def);
         unsigned word = nir_intrinsic_base(intr) + scalar.comp;
         if (scalar.def->bit_size != 32 ||
             word >= AGX_APPLE9_UNIFORM_COUNT - lower->preamble_base)
            return AGX_APPLE9_VREG_INVALID;
         uint32_t zero = apple9_dag_zero(lower);
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR_UNIFORM,
            AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1,
            lower->preamble_base + word);
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

         uint32_t index = apple9_lower_dag_scalar(lower, load->index);
         if (index == AGX_APPLE9_VREG_INVALID)
            return index;

         uint32_t address = AGX_APPLE9_VREG_INVALID;
         if (lower->nir->info.stage != MESA_SHADER_COMPUTE) {
            address = apple9_graphics_buffer_address(lower, load->argument);
            if (address == AGX_APPLE9_VREG_INVALID)
               return address;
         }

         unsigned read_count = 0;
         for (unsigned i = 0; i < lower->load_count; ++i)
            read_count += lower->loads[i].intr == load->intr;

         /* Preserve the semantic vector extent with one result tuple. Index
          * scale and byte displacement do not depend on that extent. */
         if (read_count >= 2) {
            const unsigned components = load->intr->def.num_components;
            uint8_t flags = apple9_current_load_flags(lower);
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

            struct agx_apple9_vir_instr *memory =
               lower->program.instructions[lower->program.instruction_count - 1];
            memory->memory_index_shift = load->index_shift;
            memory->memory_offset = load->byte_offset;
            if (address != AGX_APPLE9_VREG_INVALID &&
                !agx_apple9_vir_set_load_address(&lower->program, base,
                                                 address))
               return AGX_APPLE9_VREG_INVALID;

            for (unsigned c = 0; c < components; ++c) {
               const unsigned lane_key = scalar.def->index * 4 + c;
               lower->ssa_to_vreg[lane_key] = base + c;
            }
            ++lower->active_emitted_load_count;
            ++lower->emitted_load_count;
            return lower->ssa_to_vreg[key];
         }

         const uint32_t source[] = {index};
         value = apple9_dag_emit(
            lower, AGX_APPLE9_VIR_DEVICE_LOAD, AGX_APPLE9_ENC_DEVICE_LOAD,
            source, 1,
            lower->argument_base + load->argument);
         if (value != AGX_APPLE9_VREG_INVALID) {
            struct agx_apple9_vir_instr *memory =
               lower->program.instructions[lower->program.instruction_count - 1];
            memory->memory_bits = load->bit_size;
            memory->memory_index_shift = load->index_shift;
            memory->memory_offset = load->byte_offset + load->component * (load->bit_size / 8);
         }
         uint8_t flags = apple9_current_load_flags(lower);
         if (value == AGX_APPLE9_VREG_INVALID ||
             !agx_apple9_vir_set_device_load_contract(
                &lower->program, value, flags,
                AGX_APPLE9_SCOREBOARD_SLOT_AUTO)) {
            lower->reason = lower->reason != NULL
                               ? lower->reason
                               : "could not describe an Apple9 device load";
            return AGX_APPLE9_VREG_INVALID;
         }
         if (address != AGX_APPLE9_VREG_INVALID &&
             !agx_apple9_vir_set_load_address(&lower->program, value, address))
            return AGX_APPLE9_VREG_INVALID;
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
         } else if (op == nir_op_f2f16 || op == nir_op_f2f32 ||
                    op == nir_op_unpack_half_2x16) {
            nir_scalar source_scalar = nir_scalar_chase_alu_src(scalar, 0);
            unsigned half = 0;
            if (op == nir_op_unpack_half_2x16) {
               nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(scalar.def));
               source_scalar = nir_get_scalar(alu->src[0].src.ssa,
                                               alu->src[0].swizzle[0]);
               half = scalar.comp;
            }
            bool narrow = op == nir_op_f2f16;
            if ((narrow && source_scalar.def->bit_size != 32) ||
                (op == nir_op_f2f32 && source_scalar.def->bit_size != 16) ||
                half > 1) {
               lower->reason = "unsupported Apple9 float conversion width";
               return AGX_APPLE9_VREG_INVALID;
            }
            uint32_t source = apple9_lower_dag_scalar(lower, source_scalar);
            if (source != AGX_APPLE9_VREG_INVALID)
               value = apple9_dag_emit(lower,
                  narrow ? AGX_APPLE9_VIR_F2F16 : AGX_APPLE9_VIR_UNPACK_HALF,
                  narrow ? AGX_APPLE9_ENC_FLOAT_TO_HALF_ZEXT
                         : AGX_APPLE9_ENC_HALF_TO_FLOAT, &source, 1, half);
         } else if (op == nir_op_unpack_unorm_2x16 || op == nir_op_unpack_snorm_2x16 ||
                    op == nir_op_unpack_unorm_4x8 || op == nir_op_unpack_snorm_4x8) {
            bool narrow = op == nir_op_unpack_unorm_4x8 || op == nir_op_unpack_snorm_4x8;
            bool sign = op == nir_op_unpack_snorm_2x16 || op == nir_op_unpack_snorm_4x8;
            unsigned mode = narrow ? (sign ? AGX_APPLE9_UNPACK_SNORM8 : AGX_APPLE9_UNPACK_UNORM8)
                                   : (sign ? AGX_APPLE9_UNPACK_SNORM16 : AGX_APPLE9_UNPACK_UNORM16);
            if (narrow && scalar.comp >= 2)
               mode |= AGX_APPLE9_UNPACK_HIGH_HALF;
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source == AGX_APPLE9_VREG_INVALID)
               return source;
            uint32_t pair = agx_apple9_vir_emit_unpack_norm(&lower->program, source, mode);
            if (pair == AGX_APPLE9_VREG_INVALID)
               return pair;
            unsigned first = scalar.comp & ~1u;
            lower->ssa_to_vreg[scalar.def->index * 4 + first] = pair;
            lower->ssa_to_vreg[scalar.def->index * 4 + first + 1] = pair + 1;
            return pair + (scalar.comp & 1);
         } else if (op == nir_op_pack_unorm_2x16 ||
                    op == nir_op_pack_unorm_4x8) {
            bool bytes = op == nir_op_pack_unorm_4x8;
            nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(scalar.def));
            uint32_t sources[4];
            for (unsigned c = 0; c < (bytes ? 4 : 2); ++c) {
               nir_scalar input = nir_get_scalar(alu->src[0].src.ssa,
                                                 alu->src[0].swizzle[c]);
               sources[c] = apple9_lower_dag_scalar(lower, input);
               if (sources[c] == AGX_APPLE9_VREG_INVALID)
                  return AGX_APPLE9_VREG_INVALID;
            }
            value = apple9_dag_emit(lower,
               bytes ? AGX_APPLE9_VIR_PACK_UNORM_4X8 : AGX_APPLE9_VIR_PACK_UNORM_2X16,
               bytes ? AGX_APPLE9_ENC_PACK_UNORM_4X8 : AGX_APPLE9_ENC_PACK_UNORM_2X16,
               sources, bytes ? 4 : 2, 0);
         } else if (op == nir_op_pack_half_2x16 ||
                    op == nir_op_pack_half_2x16_split) {
            nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(scalar.def));
            nir_scalar inputs[2];
            for (unsigned c = 0; c < 2; ++c) {
               unsigned s = op == nir_op_pack_half_2x16_split ? c : 0;
               unsigned channel = op == nir_op_pack_half_2x16_split ? 0 : c;
               inputs[c] = nir_get_scalar(alu->src[s].src.ssa,
                                           alu->src[s].swizzle[channel]);
            }
            uint32_t source = apple9_lower_dag_scalar(lower, inputs[0]);
            if (source == AGX_APPLE9_VREG_INVALID)
               return source;
            uint32_t upper;
            if (apple9_const_u32(inputs[1], &upper) && upper == 0) {
               value = apple9_dag_emit(lower, AGX_APPLE9_VIR_F2F16,
                  AGX_APPLE9_ENC_FLOAT_TO_HALF_ZEXT, &source, 1, 0);
            } else {
               uint32_t sources[] = {source,
                  apple9_lower_dag_scalar(lower, inputs[1])};
               if (sources[1] != AGX_APPLE9_VREG_INVALID)
                  value = apple9_dag_emit(lower, AGX_APPLE9_VIR_PACK_HALF_2X16,
                     AGX_APPLE9_ENC_PACK_HALF_2X16, sources, 2, 0);
            }
         } else if (op == nir_op_bcsel) {
            uint32_t if_true = apple9_lower_dag_source(lower, scalar, 1);
            uint32_t if_false = apple9_lower_dag_source(lower, scalar, 2);
            nir_scalar predicate =
               apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 0));
            struct apple9_compare compare;
            if (if_true == AGX_APPLE9_VREG_INVALID ||
                if_false == AGX_APPLE9_VREG_INVALID) {
               value = AGX_APPLE9_VREG_INVALID;
            } else if (nir_def_instr_type(predicate.def) == nir_instr_type_alu &&
                       apple9_normalize_compare(nir_scalar_alu_op(predicate),
                                                &compare)) {
               /* SELECT compares its operands directly. Materialize a
                * boolean only for users that actually need its value. */
               value = apple9_emit_dag_select(lower, predicate, if_true, if_false);
            } else {
               uint32_t condition = apple9_lower_bool_scalar(lower, predicate);
               uint32_t zero = apple9_dag_zero(lower);
               if (condition != AGX_APPLE9_VREG_INVALID &&
                   zero != AGX_APPLE9_VREG_INVALID)
                  value = apple9_emit_dag_select_raw(
                     lower, zero, condition, if_true, if_false,
                     AGX_APPLE9_SELECT_ULT);
            }
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
         } else if (op == nir_op_unpack_64_2x32_split_x ||
                    op == nir_op_unpack_64_2x32_split_y) {
            /* Extended multiply exposes the hardware product tuple directly.
             * General 64-bit arithmetic remains outside this scalar backend. */
            uint32_t product = apple9_lower_dag_source(lower, scalar, 0);
            if (product != AGX_APPLE9_VREG_INVALID)
               value = product + (op == nir_op_unpack_64_2x32_split_y);
         } else if (op == nir_op_imul_high || op == nir_op_umul_high ||
                    op == nir_op_imul_2x32_64 || op == nir_op_umul_2x32_64) {
            uint32_t sources[] = {
               apple9_lower_dag_source(lower, scalar, 0),
               apple9_lower_dag_source(lower, scalar, 1),
            };
            if (sources[0] != AGX_APPLE9_VREG_INVALID &&
                sources[1] != AGX_APPLE9_VREG_INVALID) {
               uint32_t product = agx_apple9_vir_emit_mul_wide(
                  &lower->program, sources,
                  op == nir_op_imul_high || op == nir_op_imul_2x32_64);
               if (product != AGX_APPLE9_VREG_INVALID)
                  value = product + !wide_product;
            }
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
            value = apple9_const_u32(shift, &amount)
                       ? apple9_dag_shift_imm(lower, op, source, amount)
                       : apple9_dag_shift_variable(
                            lower, op, source, shift);
         } else if (op == nir_op_ishr || op == nir_op_ushr) {
            nir_scalar shift =
               apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 1));
            uint32_t amount;
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            value = apple9_const_u32(shift, &amount)
                       ? apple9_dag_shift_imm(lower, op, source, amount)
                       : apple9_dag_shift_variable(
                            lower, op, source, shift);
         } else if (op == nir_op_fsat) {
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source != AGX_APPLE9_VREG_INVALID)
               value = apple9_dag_emit(lower, AGX_APPLE9_VIR_FSAT,
                  AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED, &source, 1, 0);
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
            if (op == nir_op_iadd) {
               for (unsigned i = 0; i < 2; ++i) {
                  nir_scalar mul = apple9_chase_trivial(
                     nir_scalar_chase_alu_src(scalar, i));
                  if (nir_def_instr_type(mul.def) != nir_instr_type_alu ||
                      (nir_scalar_alu_op(mul) != nir_op_imul &&
                       nir_scalar_alu_op(mul) != nir_op_amul))
                     continue;
                  /* Low-product IMAD has the same wrapping arithmetic as
                   * NIR's multiply followed by add, including signed data. */
                  uint32_t sources[] = {
                     apple9_lower_dag_source(lower, mul, 0),
                     apple9_lower_dag_source(lower, mul, 1),
                     apple9_lower_dag_source(lower, scalar, 1 - i),
                  };
                  if (sources[0] != AGX_APPLE9_VREG_INVALID &&
                      sources[1] != AGX_APPLE9_VREG_INVALID &&
                      sources[2] != AGX_APPLE9_VREG_INVALID)
                     value = apple9_dag_emit(
                        lower, AGX_APPLE9_VIR_IMAD,
                        AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources, 3, 0);
                  lower->ssa_to_vreg[key] = value;
                  return value;
               }
            }
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
                             ? apple9_dag_emit(
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
   uint32_t read_mask;
   uint32_t write_mask;
   struct apple9_buffer_resource resource[AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
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
            if (map->count == (nir->info.stage == MESA_SHADER_COMPUTE
                                  ? AGX_APPLE9_COMPUTE_MAX_RESOURCES
                                  : ARRAY_SIZE(map->resource))) {
               *reason = "Apple9 buffer resource capacity exceeded";
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

   if ((map->count < 1 && nir->info.stage == MESA_SHADER_COMPUTE) ||
       map->count > ARRAY_SIZE(map->resource)) {
      *reason = "Apple9 buffer compiler requires one to eight resources";
      return false;
   }

   bool uniform_stores = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!apple9_validate_cf_list(&impl->body, reason))
      return false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (!apple9_instruction_is_in_subset(
                instr, nir->info.stage != MESA_SHADER_COMPUTE)) {
            *reason = instr->type == nir_instr_type_tex
                         ? "Apple9 texture sampling is not implemented"
                         : "Apple9 buffer compiler encountered unsupported NIR";
            return false;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         uniform_stores |= intr->intrinsic == nir_intrinsic_store_preamble;
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
         unsigned index_scale = 0, index_add = 0;
         uint8_t index_shift = 0;
         int16_t byte_offset = 0;
         const unsigned bit_size = atomic ? intr->def.bit_size
                                   : load  ? intr->def.bit_size
                                           : intr->src[0].ssa->bit_size;
         const unsigned components = atomic ? 1 : load ? intr->def.num_components
                                                       : intr->src[0].ssa->num_components;
         if (!(atomic ? apple9_element_index(intr, offset, bit_size / 8, components,
                                              &index, &index_scale, &index_add)
                      : apple9_memory_address(nir, intr, offset, bit_size / 8, components,
                                              &index, &index_shift, &byte_offset))) {
            *reason =
               "Apple9 buffer compiler requires a supported 32-bit byte offset";
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

               struct apple9_scalar_load load = {
                  .intr = intr,
                  .block = block,
                  .index = index,
                  .argument = argument,
                  .component = component,
                  .index_shift = index_shift,
                  .byte_offset = byte_offset,
                  .bit_size = intr->def.bit_size,
               };
               util_dynarray_append(loads, load);
            }
         } else {
            const unsigned components = intr->src[0].ssa->num_components;
            if (components < 1 || components > 4) {
               *reason = "Apple9 store requires one to four components";
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
               .index_shift = index_shift,
               .byte_offset = byte_offset,
               .bit_size = bit_size,
               .lowered_index = AGX_APPLE9_VREG_INVALID,
            };
            util_dynarray_append(stores, store);
         }
      }
   }

   if (!uniform_stores && stores->size == 0 && atomics->size == 0 &&
       nir->info.stage == MESA_SHADER_COMPUTE) {
      *reason = "Apple9 requires at least one SSBO side effect";
      return false;
   }

   return true;
}

static bool
apple9_emit_device_store_vir(struct apple9_emitter *emitter,
                             const struct agx_apple9_vir_instr *instruction,
                             const uint8_t *phys, const char **reason)
{
   struct agx_apple9_packed_instruction packed;
   return agx_apple9_pack_vir_instruction(instruction, phys, &packed, reason) &&
          apple9_emit_packed(emitter, &packed);
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

   uint32_t index = apple9_lower_dag_scalar(lower, store->index);
   store->lowered_index = index;
   return index != AGX_APPLE9_VREG_INVALID;
}

static bool
apple9_emit_buffer_store(struct apple9_dag_lower *lower,
                         const struct apple9_buffer_store *store)
{
   if (store->lowered_index == AGX_APPLE9_VREG_INVALID)
      return false;

   uint32_t address = AGX_APPLE9_VREG_INVALID;
   if (lower->nir->info.stage != MESA_SHADER_COMPUTE) {
      address = apple9_graphics_buffer_address(lower, store->argument);
      if (address == AGX_APPLE9_VREG_INVALID)
         return false;
   }
   if (!agx_apple9_vir_emit_device_store(&lower->program,
                                         lower->argument_base + store->argument,
                                         store->lowered_index, store->output,
                                         store->components, store->bit_size))
      return false;
   struct agx_apple9_vir_instr *memory =
      lower->program.instructions[lower->program.instruction_count - 1];
   memory->memory_index_shift = store->index_shift;
   memory->memory_offset = store->byte_offset;
   return address == AGX_APPLE9_VREG_INVALID ||
          agx_apple9_vir_set_device_store_address(&lower->program, address);
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
         if (data[c] == AGX_APPLE9_VREG_INVALID)
            return false;
      }
   } else {
      nir_scalar scalar = apple9_chase_trivial(
         nir_get_scalar(atomic->intr->src[2].ssa, 0));
      data[0] = apple9_lower_dag_scalar(lower, scalar);
      if (data[0] == AGX_APPLE9_VREG_INVALID)
         return false;


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
apple9_block_has_phi(nir_block *block)
{
   nir_foreach_phi(phi, block)
      return true;
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

   /* This jump is unconditional for the currently active lanes. Unwind still
    * consumes a predicate in the target loop's bank; the enclosing ordinary
    * if only populated bank zero. Publish true here, after the exit copies,
    * so neither stale predicates nor comparisons used by those copies can
    * decide which active lanes break. Inactive lanes remain masked out. */
   if (lower->loop->depth >= AGX_APPLE9_PREDICATE_BANK_COUNT) {
      lower->reason = "Apple9 loop break exhausted the predicate scratch bank";
      return false;
   }

   uint32_t zero = apple9_dag_zero(lower);
   uint32_t sources[] = {zero, zero};
   if (zero == AGX_APPLE9_VREG_INVALID ||
       !agx_apple9_vir_emit_side_effect(
          &lower->program, AGX_APPLE9_VIR_PREDICATE_COMPARE,
          AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP, sources, 2,
          AGX_APPLE9_PREDICATE_EXT_IEQ |
             AGX_APPLE9_PREDICATE_BANK(lower->loop->depth))) {
      lower->reason = "could not emit an Apple9 unconditional break predicate";
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

/* Coverage is a publication consumed by the raster backend. Demotion changes
 * that state without removing helper lanes from shader execution. Only the
 * final depth/stencil test acquires and releases its synchronization domain;
 * pure sample kills inside loops must leave that domain pending. */
static uint32_t
apple9_pack_coverage(struct apple9_dag_lower *lower, uint32_t affected,
                     uint32_t live)
{
   uint32_t low[] = {affected, lower->coverage_mask};
   affected = apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, low, 2, 0);
   uint32_t middle = apple9_dag_shift_imm(lower, nir_op_ishl, affected, 4);
   uint32_t middle_mask[] = {middle, lower->coverage_control};
   middle = apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, middle_mask, 2, 0);
   uint32_t pack[] = {affected, apple9_dag_imm(lower, 256), middle};
   uint32_t control = apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
      AGX_APPLE9_ENC_INT_MAD_EXTENDED, pack, 3, 0);
   uint32_t live_mask[] = {live, lower->coverage_mask};
   live = apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, live_mask, 2, 0);
   uint32_t sources[] = {control, live};
   control = apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR,
      AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
   return control;
}

static bool
apple9_emit_coverage(struct apple9_dag_lower *lower, uint32_t affected,
                     uint32_t live, bool tests)
{
   uint32_t control = apple9_pack_coverage(lower, affected, live);
   uint32_t publish[] = {control, control};
   control = apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR,
      AGX_APPLE9_ENC_LOGIC_EXPORT, publish, 2, 0);
   return control != AGX_APPLE9_VREG_INVALID &&
      (!tests || agx_apple9_vir_emit_side_effect(&lower->program,
         AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0, 1)) &&
      agx_apple9_vir_emit_side_effect(&lower->program,
         AGX_APPLE9_VIR_COVERAGE, AGX_APPLE9_ENC_COVERAGE, &control, 1, 0) &&
      (!tests || agx_apple9_vir_emit_side_effect(&lower->program,
         AGX_APPLE9_VIR_TILE_FENCE, AGX_APPLE9_ENC_TILE_FENCE, NULL, 0, 1));
}

static bool
apple9_emit_graphics_output(struct apple9_dag_lower *lower,
                            nir_intrinsic_instr *intr)
{
   const bool fragment = lower->nir->info.stage == MESA_SHADER_FRAGMENT;
   if (fragment) {
      if (intr->intrinsic != nir_intrinsic_store_local_pixel_agx ||
          (intr->src[1].ssa->bit_size != 16 &&
           intr->src[1].ssa->bit_size != 32) ||
          nir_intrinsic_explicit_coord(intr) ||
          (nir_intrinsic_format(intr) != PIPE_FORMAT_R32_UINT &&
              nir_intrinsic_format(intr) != PIPE_FORMAT_R32G32_UINT &&
              nir_intrinsic_format(intr) != PIPE_FORMAT_R32G32B32_UINT &&
              nir_intrinsic_format(intr) != PIPE_FORMAT_R32G32B32A32_UINT) ||
          (nir_intrinsic_base(intr) & 3) ||
          intr->src[0].ssa->bit_size != 32 ||
          intr->num_components < 1 || intr->num_components > 4)
         return false;
      lower->color_stores = true;
      assert(lower->tile_access && lower->tile_write);
      uint32_t sample_mask = apple9_lower_dag_scalar(lower,
         nir_get_scalar(intr->src[1].ssa, 0));
      if (sample_mask == AGX_APPLE9_VREG_INVALID)
         return false;
      for (unsigned word = 0; word < intr->num_components; ++word) {
         uint32_t color = apple9_lower_dag_scalar(
            lower, nir_get_scalar(intr->src[0].ssa, word));
         uint32_t store_sources[] = {color, sample_mask};
         if (color == AGX_APPLE9_VREG_INVALID ||
             !agx_apple9_vir_emit_side_effect(
                &lower->program, AGX_APPLE9_VIR_TILE_STORE,
                AGX_APPLE9_ENC_TILE_STORE_MASK, store_sources, 2,
                nir_intrinsic_base(intr) / 4 + word))
            return false;
      }
      return true;
   }
   if (!nir_src_is_const(intr->src[1]) || intr->src[0].ssa->bit_size != 32)
      return false;
   const unsigned location = nir_intrinsic_io_semantics(intr).location +
                             nir_src_as_uint(intr->src[1]);
   nir_alu_type type = nir_intrinsic_src_type(intr);
   bool integer = type == nir_type_int32 || type == nir_type_uint32;
   if (type != nir_type_float32 && !integer)
      return false;
   if (location == VARYING_SLOT_POS && integer)
      return false;
   for (unsigned c = 0; c < intr->num_components; ++c) {
      if (!(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)))
         continue;
      unsigned component = nir_intrinsic_component(intr) + c;
      if (component >= 4)
         return false;
      unsigned slot = component;
      if (location == VARYING_SLOT_PSIZ) {
         if (integer || component != 0)
            return false;
         slot = 4 + lower->varyings->count;
      } else if (location != VARYING_SLOT_POS) {
         int index = apple9_varying_index(lower->varyings, location, component);
         if (index < 0)
            return false;
         slot = 4 + index;
      }
      uint32_t value =
         apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[0].ssa, c));
      /* Mesa may erase user-varying types while lowering IO. Publish their
       * bits without FP arithmetic: multiplying an integer payload by float
       * one would quiet NaNs and flush subnormals. The UVS transport is untyped. */
      bool raw = integer || (location != VARYING_SLOT_POS && location != VARYING_SLOT_PSIZ);
      uint32_t scale = apple9_dag_imm(lower, raw ? 0 : fui(1.0f));
      uint32_t sources[] = {value, scale};
      if (value == AGX_APPLE9_VREG_INVALID || scale == AGX_APPLE9_VREG_INVALID)
         return false;
      value = apple9_dag_emit(
         lower, raw ? AGX_APPLE9_VIR_IXOR : AGX_APPLE9_VIR_FMUL,
         raw ? AGX_APPLE9_ENC_LOGIC_EXPORT : AGX_APPLE9_ENC_FLOAT2_EXPORT,
         sources, 2, 0);
      /* The allocator retains this publication independently of GPR liveness. */
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
         if (!apple9_emit_jump(lower, block, nir_instr_as_jump(instr)))
            return false;
         continue;
      }

      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_barycentric_pixel)
            continue; /* Interpolation mode is consumed by the input operation. */
         if (intr->intrinsic == nir_intrinsic_store_zs_agx) {
            if (nir_intrinsic_base(intr) != 1 || !nir_src_is_const(intr->src[0]) ||
                nir_src_as_uint(intr->src[0]) != 0xff)
               return false;
            uint32_t value = apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[1].ssa, 0));
            uint32_t clamp[] = {value, apple9_dag_imm(lower, 0)};
            value = apple9_dag_emit(lower, AGX_APPLE9_VIR_FMAX,
               AGX_APPLE9_ENC_MINMAX_COMPACT, clamp, 2, 0);
            clamp[0] = value; clamp[1] = apple9_dag_imm(lower, fui(1.0f));
            value = apple9_dag_emit(lower, AGX_APPLE9_VIR_FMIN,
               AGX_APPLE9_ENC_MINMAX_COMPACT, clamp, 2, 0);
            uint32_t mask = apple9_dag_imm(lower, 0xff);
            uint32_t control = apple9_pack_coverage(lower, mask, mask);
            uint32_t fields[] = {control, value};
            value = agx_apple9_vir_emit_publication_pair(&lower->program, fields);
            uint32_t pair[] = {value, value + 1};
            if (value == AGX_APPLE9_VREG_INVALID ||
                !agx_apple9_vir_emit_side_effect(&lower->program,
                   AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0, 1) ||
                !agx_apple9_vir_emit_side_effect(&lower->program,
                   AGX_APPLE9_VIR_DEPTH_STORE, AGX_APPLE9_ENC_DEPTH_STORE, pair, 2, 0) ||
                !agx_apple9_vir_emit_side_effect(&lower->program,
                   AGX_APPLE9_VIR_TILE_FENCE, AGX_APPLE9_ENC_TILE_FENCE, NULL, 0, 1))
               return false;
            lower->writes_z = true;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
            uint32_t affected = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[0].ssa, 0));
            uint32_t live = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[1].ssa, 0));
            if (lower->nir->info.stage != MESA_SHADER_FRAGMENT ||
                affected == AGX_APPLE9_VREG_INVALID || live == AGX_APPLE9_VREG_INVALID ||
                !apple9_emit_coverage(lower, affected, live,
                   !(nir_src_is_const(intr->src[1]) && !nir_src_as_uint(intr->src[1]))))
               return false;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_output ||
             intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
            if (!apple9_emit_graphics_output(lower, intr)) {
               if (!lower->reason)
                  lower->reason = "unsupported Apple9 graphics output";
               return false;
            }
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_image_store_block_agx) {
            bool multisampled = nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS;
            if ((!multisampled && nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D) ||
                nir_intrinsic_image_array(intr) || nir_intrinsic_explicit_coord(intr))
               return false;
            enum pipe_format format = nir_intrinsic_format(intr);
            unsigned tile_format = agx_apple9_block_export_format(format);
            if (tile_format > 15)
               return false;
            unsigned binding = nir_src_as_uint(intr->src[0]);
            unsigned image = util_bitcount(lower->texture_mask) +
               util_bitcount(lower->image_mask & BITFIELD_MASK(binding));
            uint32_t src[] = {
               apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[2].ssa, 0)),
               apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[2].ssa, 1)),
               apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[1].ssa, 0)),
            };
            /* The publication word combines a byte offset in its upper
             * half with mip level zero below. Destination views supply their
             * own level dimensions and base address. */
            src[2] = apple9_dag_shift_imm(lower, nir_op_ishl, src[2], 16);
            if (!agx_apple9_vir_emit_block_image_store(&lower->program, src, image,
                                                       tile_format, multisampled))
               return false;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_preamble) {
            nir_def *def = intr->src[0].ssa;
            unsigned base = nir_intrinsic_base(intr);
            if (def->bit_size != 32 ||
                base + def->num_components >
                   AGX_APPLE9_UNIFORM_COUNT - lower->preamble_base)
               return false;
            for (unsigned c = 0; c < def->num_components; ++c) {
               uint32_t value = apple9_lower_dag_scalar(lower, nir_get_scalar(def, c));
               if (value == AGX_APPLE9_VREG_INVALID ||
                   !agx_apple9_vir_emit_side_effect(&lower->program,
                      AGX_APPLE9_VIR_STORE_UNIFORM, AGX_APPLE9_ENC_STORE_UNIFORM,
                      &value, 1, lower->preamble_base + base + c))
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
      if (def && def->bit_size == 1) {
         /* A Boolean consumed in another block must be computed while its
          * defining execution mask is still active. In particular, deferring
          * a loop-carried value to the back edge skips its computation for
          * lanes taking an earlier break and leaves loop-exit uses undefined.
          * Keep block-local predicates available for direct selection. */
         bool crosses_block = false;
         nir_foreach_use(use, def) {
            if (nir_src_use_instr(use)->block != block)
               crosses_block = true;
         }
         nir_foreach_if_use(use, def) {
            if (nir_cf_node_prev(&nir_src_use_if(use)->cf_node) != &block->cf_node)
               crosses_block = true;
         }
         if (crosses_block) {
            const nir_component_mask_t read = nir_def_components_read(def);
            for (unsigned c = 0; c < def->num_components; ++c) {
               if ((read & BITFIELD_BIT(c)) &&
                   apple9_lower_bool_scalar(lower, nir_get_scalar(def, c)) ==
                      AGX_APPLE9_VREG_INVALID)
                  return false;
            }
         }
      }
      if (def == NULL || def->bit_size == 1 ||
          (def->bit_size != 8 && def->bit_size != 16 && def->bit_size != 32 &&
           def->bit_size != 64))
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

/* Declare phi SSA identities before selecting incoming edges. Their owning
 * blocks are already known; declarations are placed there after selection. */
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
         uint32_t merge_vreg = agx_apple9_vir_emit_phi(&lower->program, lower->nir_blocks[merge->index]);
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
   struct util_dynarray copies = {0};
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
         goto fail;
      }

      for (unsigned c = 0; c < phi->def.num_components; ++c) {
         const unsigned key = phi->def.index * 4 + c;
         if (key >= lower->ssa_map_count ||
             lower->ssa_to_vreg[key] == AGX_APPLE9_VREG_INVALID) {
            lower->reason = "Apple9 phi edge is incomplete";
            goto fail;
         }

         nir_scalar source_scalar = nir_get_scalar(source, c);
         uint32_t value = phi->def.bit_size == 1
                             ? apple9_lower_bool_scalar(lower, source_scalar)
                             : apple9_lower_dag_scalar(lower, source_scalar);
         if (value == AGX_APPLE9_VREG_INVALID)
            goto fail;
         util_dynarray_append(&copies,
                              ((struct agx_apple9_vir_copy){
                                 lower->ssa_to_vreg[key], value}));
      }
   }

   /* All sources are selected before recording this parallel edge. */
   if (!agx_apple9_vir_emit_phi_edge(
          &lower->program, util_dynarray_begin(&copies),
          util_dynarray_num_elements(&copies, struct agx_apple9_vir_copy))) {
      lower->reason = "could not record Apple9 phi incoming edge";
      goto fail;
   }
   util_dynarray_fini(&copies);
   return true;

fail:
   util_dynarray_fini(&copies);
   return false;
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
                          struct agx_apple9_block *header)
{
   if (!agx_apple9_vir_emit_branch(&lower->program, AGX_APPLE9_VIR_JMP_EXEC_ANY,
                                   AGX_APPLE9_ENC_JMP_EXEC_ANY,
                                   header)) {
      lower->reason = "could not emit an Apple9 loop backedge";
      return false;
   }
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

   struct agx_apple9_block *header_block = lower->nir_blocks[header->index];
   struct agx_apple9_block *exit_block = agx_apple9_block_create(&lower->program);
   if (!header_block || !exit_block) {
      lower->reason = "out of memory creating Apple9 loop blocks";
      return false;
   }
   agx_apple9_block_begin(&lower->program, header_block);
   struct apple9_loop_context context = {
      .parent = lower->loop,
      .nir = loop,
      .exit = exit,
      .exit_block = exit_block,
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
      if (!apple9_emit_loop_backedge(lower, header_block))
         return false;
   }

   lower->loop = context.parent;

   agx_apple9_block_begin(&lower->program, exit_block);

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
   nir_cf_node *next = nir_cf_node_next(&nif->cf_node);
   if (next == NULL || next->type != nir_cf_node_block) {
      lower->reason = "Apple9 if/else has no structured merge block";
      return false;
   }

   nir_block *merge = nir_cf_node_as_block(next);
   const bool has_phis = apple9_block_has_phi(merge);
   /* Preserve both NIR predecessor blocks. The shared allocator can insert
    * edge copies even when an arm has no selected arithmetic. */
   if (has_phis && !apple9_prepare_phis(lower, merge))
      return false;

   bool invert_push = false;
   if (!apple9_emit_if_predicate(lower, nif, &invert_push) ||
       !apple9_emit_exec_mask(lower, true, invert_push))
      return false;
   ++lower->mask_depth;

   nir_block *then_pred = apple9_cf_list_last_block(&nif->then_list);
   nir_block *else_pred = apple9_cf_list_last_block(&nif->else_list);
   if (!apple9_emit_cf_list(lower, stores, &nif->then_list) ||
       (has_phis && apple9_block_reaches(then_pred, merge) &&
        !apple9_emit_phi_copies_for_edge(lower, merge, then_pred)))
      return false;

   if (!apple9_emit_exec_mask_else(lower) ||
       !apple9_emit_cf_list(lower, stores, &nif->else_list) ||
       (has_phis && apple9_block_reaches(else_pred, merge) &&
        !apple9_emit_phi_copies_for_edge(lower, merge, else_pred)))
      return false;

   if (!apple9_emit_exec_mask(lower, false, false))
      return false;
   assert(lower->mask_depth > 0);
   --lower->mask_depth;
   return true;
}

/* Apple9's mask operations address a recursive hardware mask structure.
 * Mirror NIR directly: emit each structured block, if, or loop in program
 * order while keeping stable blocks and explicit phi predecessor uses. */
static bool
apple9_emit_cf_list(struct apple9_dag_lower *lower,
                    struct util_dynarray *stores, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         struct agx_apple9_block *block =
            lower->nir_blocks[nir_cf_node_as_block(node)->index];
         if (!block->placed)
            agx_apple9_block_begin(&lower->program, block);
         if (!apple9_emit_block(lower, stores, nir_cf_node_as_block(node)))
            return false;
         break;
      }
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

static void
apple9_legalize_buffer_accesses(nir_shader *nir)
{
   const nir_lower_mem_access_bit_sizes_options memory_options = {
      .modes = nir_var_mem_ssbo | nir_var_mem_ubo,
      .callback = apple9_memory_format,
   };
   nir_opt_load_store_update_alignments(nir);
   nir_lower_mem_access_bit_sizes(nir, &memory_options);
   /* Memory legalization can create pack/unpack operations after the common
    * preprocessing pipeline. Use Mesa's arithmetic lowering for Apple9's
    * currently supported scalar bit operations. */
   const nir_shader_compiler_options *saved_options = nir->options;
   nir_shader_compiler_options memory_alu_options = *saved_options;
   memory_alu_options.lower_extract_byte = true;
   memory_alu_options.lower_extract_word = true;
   memory_alu_options.has_pack_32_4x8 = false;
   nir->options = &memory_alu_options;
   nir_lower_pack(nir);
   nir_shader_alu_pass(nir, apple9_lower_memory_pack, nir_metadata_control_flow, NULL);
   nir->options = saved_options;

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

}

static void
apple9_prepare_dag(nir_shader *nir)
{
   /* Legalization preserves vectors it is given, but does not discover
    * adjacent scalar accesses. Canonicalize their addresses first and let
    * NIR retain its alignment, aliasing and wrapping-offset checks. */
   nir_opt_constant_folding(nir);
   nir_opt_cse(nir);
   nir_opt_dce(nir);
   const nir_load_store_vectorize_options vector_options = {
      .modes = nir_var_mem_ubo | nir_var_mem_ssbo,
      .callback = agx_mem_vectorize_cb,
   };
   nir_opt_load_store_vectorize(nir, &vector_options);

   apple9_legalize_buffer_accesses(nir);
   bool progress;

   /* Match the common Asahi if-conversion policy. NIR checks side effects
    * and speculation legality; phi removal must follow control-flow edits. */
   const nir_opt_peephole_select_options select_options = {
      .limit = 64,
      .expensive_alu_ok = true,
   };
   do {
      progress = nir_opt_peephole_select(nir, &select_options);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_remove_phis(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_cse(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);

   /* FMA formation and other late algebraic rules belong after lowering in
    * every shader, not only shaders which happened to contain division.
    * Re-legalize scalar packing/extraction introduced by those rules. */
   nir_opt_algebraic_late(nir);
   nir_shader_alu_pass(nir, apple9_lower_memory_pack,
                      nir_metadata_control_flow, NULL);
   do {
      progress = nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_cse(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);

}

static bool
apple9_compile_dag_body(nir_shader *nir, struct agx_shader_part *out,
                       struct agx_apple9_compute_profile *profile,
                       const struct agx_apple9_varying_layout *varyings,
                       const struct apple9_buffer_map *resources,
                       bool preamble, const char **reason)
{
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
   struct apple9_buffer_map resource_map = *resources;

   if (!apple9_find_buffer_dag(nir, &resource_map, &loads, &stores, &atomics,
                               reason)) {
      util_dynarray_fini(&loads);
      util_dynarray_fini(&stores);
      util_dynarray_fini(&atomics);
      return false;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   nir_index_blocks(impl);
   uint32_t texture_mask = 0, sampler_mask = 0, image_mask = 0;
   bool uses_texel_fetch = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_image_store_block_agx) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) >= 32) {
               *reason = "Apple9 block export requires a constant image binding";
               return false;
            }
            image_mask |= BITFIELD_BIT(nir_src_as_uint(intr->src[0]));
         }
         if (instr->type == nir_instr_type_tex) {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            texture_mask |= BITFIELD_BIT(tex->texture_index);
            uses_texel_fetch |= apple9_texture_uses_fetch_sampler(tex);
            if (!apple9_texture_uses_fetch_sampler(tex))
               sampler_mask |= BITFIELD_BIT(tex->sampler_index);
         }
      }
   }
   bool has_texture = texture_mask != 0;
   struct apple9_dag_lower lower = {
      .nir = nir,
      .texture_mask = texture_mask,
      .image_mask = image_mask,
      .sampler_mask = sampler_mask,
      .varyings = varyings,
      .zero_vreg = AGX_APPLE9_VREG_INVALID,
      .loads = loads.data,
      .load_count =
         util_dynarray_num_elements(&loads, struct apple9_scalar_load),
      .atomics = atomics.data,
      .atomic_count =
         util_dynarray_num_elements(&atomics, struct apple9_buffer_atomic),
      .argument_base = nir->info.stage == MESA_SHADER_COMPUTE
                          ? AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE
                          : 2,
      .preamble_base = nir->info.stage == MESA_SHADER_COMPUTE
                         ? agx_apple9_compute_root_words(resources->count)
                         : AGX_APPLE9_GRAPHICS_ROOT_WORDS,
      .structured_cf = apple9_cf_list_has_control_flow(&impl->body),
   };
   /* Reserve depth before assigning point-coordinate coefficients, independent
    * of the order in which the DAG visits the two built-ins. */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_frag_coord_z)
            lower.reads_z = true;
      }
   }
   for (unsigned i = 0; i < ARRAY_SIZE(lower.system_vreg); ++i)
      lower.system_vreg[i] = AGX_APPLE9_VREG_INVALID;
   for (unsigned i = 0; i < lower.load_count; ++i) {
      bool first = true;
      for (unsigned earlier = 0; earlier < i; ++earlier)
         first &= lower.loads[earlier].intr != lower.loads[i].intr;
      lower.load_instruction_count += first;
   }
   agx_apple9_vir_init(&lower.program);
   /* Synthetic values cached across NIR blocks must dominate their uses.
    * A lazy zero first requested inside a conditional is undefined for lanes
    * bypassing that arm, even if its GPR remains live at the later use. */
   if (lower.structured_cf && apple9_dag_zero(&lower) == AGX_APPLE9_VREG_INVALID) {
      *reason = "could not initialize Apple9 control-flow zero";
      goto fail;
   }
   /* System values are cached by selector across the whole shader. Read them
    * before divergent control flow so a first use in one arm cannot leave
    * lanes in another arm with an undefined cached value. This also applies
    * to instance IDs introduced by vertex attribute lowering. */
   if (lower.structured_cf) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_infos[intr->intrinsic].has_dest)
               continue;
            for (unsigned c = 0; c < intr->def.num_components; ++c) {
               struct apple9_system_source system;
               if (apple9_system_source(nir_get_scalar(&intr->def, c), &system) &&
                   apple9_dag_system(&lower, system) == AGX_APPLE9_VREG_INVALID) {
                  *reason = "could not initialize Apple9 system value";
                  goto fail;
               }
            }
         }
      }
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_FRONT_FACE) &&
       apple9_dag_system(&lower, (struct apple9_system_source){.selector = 0xc5}) ==
          AGX_APPLE9_VREG_INVALID) {
      *reason = "could not initialize Apple9 facing state";
      goto fail;
   }
   lower.nir_blocks = calloc(impl->num_blocks, sizeof(*lower.nir_blocks));
   if (!lower.nir_blocks) {
      *reason = "out of memory indexing Apple9 blocks";
      goto fail;
   }
   nir_foreach_block(block, impl) {
      lower.nir_blocks[block->index] = agx_apple9_block_create(&lower.program);
      if (!lower.nir_blocks[block->index]) {
         *reason = "out of memory creating Apple9 blocks";
         goto fail;
      }
      lower.nir_blocks[block->index]->nir = block;
   }
   lower.ssa_map_count = impl->ssa_alloc * 4;
   lower.ssa_to_vreg = malloc(lower.ssa_map_count * sizeof(uint32_t));
   if (lower.ssa_to_vreg == NULL) {
      *reason = "out of memory indexing Apple9 NIR";
      goto fail;
   }
   for (unsigned i = 0; i < lower.ssa_map_count; ++i)
      lower.ssa_to_vreg[i] = AGX_APPLE9_VREG_INVALID;

   /* Both demotion and explicit depth export consume raster coverage.
    * Do not let depth-only shaders use uninitialized coverage SSA indices. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       (nir->info.fs.uses_discard ||
        (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)))) {
      uint32_t coverage = apple9_dag_emit(&lower,
         AGX_APPLE9_VIR_GET_SR, AGX_APPLE9_ENC_GET_COVERAGE, NULL, 0, 0x10c2);
      uint32_t low[] = {coverage, apple9_dag_imm(&lower, 15)};
      uint32_t high[] = {coverage, apple9_dag_imm(&lower, 240)};
      uint32_t affected = apple9_dag_emit(&lower, AGX_APPLE9_VIR_IAND,
         AGX_APPLE9_ENC_LOGIC_EXTENDED, low, 2, 0);
      lower.coverage_mask = affected;
      uint32_t middle = apple9_dag_emit(&lower, AGX_APPLE9_VIR_IAND,
         AGX_APPLE9_ENC_LOGIC_EXTENDED, high, 2, 0);
      lower.coverage_control = middle;
      if (middle == AGX_APPLE9_VREG_INVALID)
         goto fail;
   }

   /* Hold the tile transaction across every sample and attachment. Opening
    * it in the entry block also covers stores in divergent control flow.
    * Releasing individual stores lets later fragments overtake this one. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      bool reads = false, writes = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            reads |= op == nir_intrinsic_load_local_pixel_agx ||
                     op == nir_intrinsic_load_tile_pixel_agx;
            lower.tile_coords |= op == nir_intrinsic_load_tile_pixel_agx;
            writes |= op == nir_intrinsic_store_local_pixel_agx;
         }
      }
      if (reads || writes || image_mask || has_texture || nir->info.fs.uses_discard) {
         if (!agx_apple9_vir_emit_side_effect(
                &lower.program, AGX_APPLE9_VIR_TILE_ACCESS,
                AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0,
                image_mask ? 0x700 : lower.tile_coords ? 0xf00 : 0x600))
            goto fail;
         lower.tile_access = true;
      }
      if ((reads || writes) && !lower.tile_coords) {
         if (!agx_apple9_vir_emit_side_effect(&lower.program,
                AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS,
                NULL, 0, writes ? 0x80c : 0x808))
            goto fail;
         lower.tile_read = reads;
         lower.tile_write = writes;
      }
      if (lower.tile_coords)
         lower.tile_read = reads;
   }


   if (!apple9_emit_cf_list(&lower, &stores, &impl->body)) {
      *reason = lower.reason != NULL
                   ? lower.reason
                   : "could not emit Apple9 structured control flow";
      goto fail;
   }

   if (!preamble && nir->info.stage == MESA_SHADER_VERTEX && lower.position_mask != 15) {
      *reason = "Apple9 render requires complete position";
      goto fail;
   }

   /* Finalize tile writes once after all color outputs. The release marks
    * completion of the fragment's tile transaction, including MRT stores. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       (lower.color_stores || lower.tile_read) &&
       !agx_apple9_vir_emit_side_effect(
          &lower.program, AGX_APPLE9_VIR_TILE_FENCE, AGX_APPLE9_ENC_TILE_FENCE,
          NULL, 0, lower.tile_coords ? 0x0300 : 0x020c))
      goto fail;

   if (lower.emitted_load_count != lower.load_instruction_count) {
      *reason = "Apple9 DAG contains an input load outside the store graph";
      goto fail;
   }

   agx_apple9_place_phis(&lower.program);
   lower.program.fragment_shader = nir->info.stage == MESA_SHADER_FRAGMENT;

   if (!agx_apple9_optimize_vir(&lower.program)) {
      *reason = "out of memory optimizing Apple9 virtual IR";
      goto fail;
   }
   if (!agx_apple9_allocate_shared(&lower.program, nir, reason))
      goto fail;

   if (getenv("AGX_APPLE9_TRACE") != NULL) {
      fprintf(stderr, "APPLE9_ALLOC_STATS stage=%u peak_live_gprs=%u publications=%u scratch_bytes=%u spill_slots=%u\n",
              nir->info.stage, lower.program.peak_live_gprs,
              lower.program.publication_count, lower.program.scratch_size,
              lower.program.spill_slots);
      for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
         const struct agx_apple9_vir_instr *instruction =
            lower.program.instructions[i];
         fprintf(stderr, "APPLE9_ALLOC i=%u op=%u enc=%u dst=", i,
                 instruction->op, instruction->encoding);
         if (instruction->dest == AGX_APPLE9_VREG_INVALID)
            fputs("-", stderr);
         else
            fprintf(stderr, "%c%u",
                    lower.program.publication[instruction->dest] ? 'p' : 'r',
                    lower.program.phys[instruction->dest]);
         if (instruction->op == AGX_APPLE9_VIR_PHI_SRC)
            fprintf(stderr, " target=r%u",
                    lower.program.phys[instruction->target]);
         fputs(" src=", stderr);
         for (unsigned s = 0; s < instruction->nr_srcs; ++s)
            fprintf(stderr, "%s%c%u", s ? "," : "",
                    lower.program.publication[instruction->src[s]] ? 'p' : 'r',
                    lower.program.phys[instruction->src[s]]);
         fprintf(stderr, " imm=%#x live=%#x slot=%u producer_slot=%u\n",
                 instruction->immediate, instruction->live_after_mask,
                 instruction->scoreboard_slot,
                 instruction->producer_scoreboard_slot);
      }
   }

   struct apple9_emitter emitter = {.bytes = UTIL_DYNARRAY_INIT};
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
         lower.program.instructions[i];
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
               fprintf(stderr, "%c%u",
                       lower.program.publication[instruction->dest] ? 'p' : 'r',
                       lower.program.phys[instruction->dest]);
            fputs(" src=", stderr);
            for (unsigned s = 0; s < instruction->nr_srcs; ++s)
               fprintf(stderr, "%s%c%u", s ? "," : "",
                       lower.program.publication[instruction->src[s]] ? 'p' : 'r',
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
            fprintf(stderr, "%c%u",
                    lower.program.publication[instruction->dest] ? 'p' : 'r',
                    lower.program.phys[instruction->dest]);
         fputs(" src=", stderr);
         for (unsigned s = 0; s < instruction->nr_srcs; ++s)
            fprintf(stderr, "%s%c%u", s ? "," : "",
                    lower.program.publication[instruction->src[s]] ? 'p' : 'r',
                    lower.program.phys[instruction->src[s]]);
         fprintf(stderr, " live=%#x slot=%u producer_slot=%u\n",
                 instruction->live_after_mask, instruction->scoreboard_slot,
                 instruction->producer_scoreboard_slot);
      }
   }
   vir_offsets[lower.program.instruction_count] = emitter.bytes.size;
   for (struct agx_apple9_block *block = lower.program.blocks; block; block = block->next)
      block->offset = vir_offsets[block->start_index];

   /* Branch displacements are relative to the start of the branch, not its
    * end. Resolve them only now: COLLECT and physical phi-edge copies can emit a
    * variable number of physical instructions after allocation. */
   for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
      const struct agx_apple9_vir_instr *instruction =
         lower.program.instructions[i];
      if (instruction->op != AGX_APPLE9_VIR_JMP_EXEC_ANY &&
          instruction->op != AGX_APPLE9_VIR_JMP_EXEC_NONE)
         continue;

      if (!instruction->branch_target || !instruction->branch_target->placed) {
         *reason = "Apple9 branch targets an unplaced block";
         util_dynarray_fini(&emitter.bytes);
         free(vir_offsets);
         goto fail;
      }
      const int64_t displacement =
         (int64_t)instruction->branch_target->offset - vir_offsets[i];
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
   /* List final packed bytes after branch fixups, without changing codegen.
    * A row can contain multiple instructions for an expanded VIR pseudo. */
   if (getenv("AGX_APPLE9_ASM") != NULL) {
      fprintf(stderr, "%s_ASM_BEGIN stage=%u\n",
              preamble ? "APPLE9_PREAMBLE" : "APPLE9", nir->info.stage);
      for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
         const struct agx_apple9_vir_instr *ins = lower.program.instructions[i];
         fprintf(stderr, "APPLE9_ASM offset=%u op=%u enc=%u dst=",
                 vir_offsets[i], ins->op, ins->encoding);
         if (ins->dest == AGX_APPLE9_VREG_INVALID)
            fputs("-", stderr);
         else
            fprintf(stderr, "%c%u", lower.program.publication[ins->dest] ? 'p' : 'r',
                    lower.program.phys[ins->dest]);
         fputs(" src=", stderr);
         for (unsigned j = 0; j < ins->nr_srcs; ++j)
            fprintf(stderr, "%s%c%u", j ? "," : "",
                    lower.program.publication[ins->src[j]] ? 'p' : 'r',
                    lower.program.phys[ins->src[j]]);
         fprintf(stderr, " imm=%#x wait=%u producer=%u bytes=", ins->immediate,
                 ins->scoreboard_slot, ins->producer_scoreboard_slot);
         for (unsigned j = vir_offsets[i]; j < vir_offsets[i + 1]; ++j)
            fprintf(stderr, "%02x", ((const uint8_t *)emitter.bytes.data)[j]);
         fputc('\n', stderr);
      }
      fputs("APPLE9_ASM_END\n", stderr);
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
   if (getenv("AGX_APPLE9_STATS") != NULL) {
      unsigned copies = 0, waits = 0, saves = 0, fills = 0;
      for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
         const struct agx_apple9_vir_instr *ins = lower.program.instructions[i];
         copies += ins->op == AGX_APPLE9_VIR_IOR &&
                   ins->encoding == AGX_APPLE9_ENC_LOGIC_EXTENDED &&
                   ins->nr_srcs == 2 && ins->src[0] == ins->src[1];
         waits += ins->scoreboard_slot != AGX_APPLE9_SCOREBOARD_SLOT_NONE;
         saves += ins->op == AGX_APPLE9_VIR_SPILL_STORE;
         fills += ins->op == AGX_APPLE9_VIR_SPILL_LOAD;
      }
      fprintf(stderr, "%s_CODEGEN stage=%u bytes=%u instructions=%u copies=%u waits=%u saves=%u fills=%u gprs=%u scratch=%u\n",
              preamble ? "APPLE9_PREAMBLE" : "APPLE9",
              nir->info.stage, emitter.bytes.size, lower.program.instruction_count,
              copies, waits, saves, fills, lower.program.peak_live_gprs,
              lower.program.scratch_size);
   }


   out->binary = emitter.bytes.data;
   out->info.stage = nir->info.stage;
   out->info.apple9_linear_mask = lower.linear_mask;
   out->info.apple9_reads_z = lower.reads_z;
   out->info.depth_layout = lower.writes_z ? FRAG_DEPTH_LAYOUT_ANY
                                          : FRAG_DEPTH_LAYOUT_UNCHANGED;
   out->info.apple9_reads_point_coord = lower.reads_point_coord;
   out->info.apple9_writes_point_size = nir->info.stage == MESA_SHADER_VERTEX &&
      (nir->info.outputs_written & BITFIELD64_BIT(VARYING_SLOT_PSIZ));
   out->info.apple9_reads_tile = lower.tile_read;
   out->info.apple9_flat_mask = lower.flat_mask;
   out->info.apple9_texture_mask = texture_mask;
   out->info.apple9_image_mask = image_mask;
   out->info.apple9_sampler_mask = sampler_mask;
   out->info.apple9_uses_texel_fetch = uses_texel_fetch;
   out->info.apple9_uses_discard = nir->info.stage == MESA_SHADER_FRAGMENT &&
                                    (nir->info.fs.uses_discard || lower.writes_z);
   out->info.disable_tri_merging = nir->info.stage == MESA_SHADER_FRAGMENT &&
      (nir->info.uses_wide_subgroup_intrinsics ||
       nir->info.fs.needs_coarse_quad_helper_invocations || nir->info.writes_memory);
   if (nir->info.stage != MESA_SHADER_COMPUTE) {
      out->info.apple9_resource_count = resource_map.count;
      out->info.apple9_resource_write_mask = resource_map.write_mask;
      for (unsigned i = 0; i < resource_map.count; ++i) {
         unsigned binding = resource_map.resource[i].binding;
         out->info.apple9_resource_binding[i] = binding;
         if (resource_map.resource[i].kind == AGX_APPLE9_COMPUTE_RESOURCE_SSBO)
            out->info.apple9_resource_ssbo_mask |= BITFIELD_BIT(i);
         else if (binding < 32)
            out->info.apple9_ubo_mask |= BITFIELD_BIT(binding);
      }
   }
   if (varyings)
      out->info.apple9_varyings = *varyings;
   if (!preamble && nir->info.stage == MESA_SHADER_FRAGMENT)
      out->info.varyings.fs.nr_cf = varyings->count + 1 + lower.reads_z +
                                  2 * lower.reads_point_coord;
   out->info.main_size = emitter.bytes.size;
   out->info.binary_size = emitter.bytes.size;
   /* nr_gprs is the architectural GPR high-water mark for Apple9.  Keep the
    * established minimum tier, but do not collapse every program using r16+
    * back to the old 16-register capture tier. */
   out->info.nr_gprs = MAX2(lower.program.max_phys_gpr + 1, 8);
   out->info.scratch_size = lower.program.scratch_size;
   out->info.apple9_publication_count = lower.program.publication_count;
   out->info.stats.instrs = emitter.instructions;
   for (unsigned d = 0; d < 3; ++d)
      out->info.workgroup_size[d] = nir->info.workgroup_size[d];

   if (profile != NULL) {
      const unsigned resource_count = resource_map.count;
      *profile = AGX_APPLE9_DIRECT_BUFFERS_COMPUTE_PROFILE;
      for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
         const struct agx_apple9_vir_instr *ins = lower.program.instructions[i];
         if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC && !ins->atomic_discard)
            profile->atomic_frame_size = 4;
      }
      profile->resource_binding_count = resource_count;
      profile->resource_read_mask = resource_map.read_mask;
      profile->resource_write_mask = resource_map.write_mask;
      for (unsigned i = 0; i < profile->resource_binding_count; ++i) {
         profile->resource_binding[i] = resource_map.resource[i].binding;
         profile->resource_kind[i] = resource_map.resource[i].kind;
      }
      profile->scratch_size = lower.program.scratch_size;
      profile->variable_local_size = nir->info.workgroup_size_variable;
      for (unsigned d = 0; d < 3; ++d)
         profile->local_size[d] = nir->info.workgroup_size[d];
   }

   free(lower.nir_blocks);
   free(lower.ssa_to_vreg);
   agx_apple9_vir_finish(&lower.program);
   util_dynarray_fini(&loads);
   util_dynarray_fini(&stores);
   util_dynarray_fini(&atomics);
   return true;

fail:
   free(lower.nir_blocks);
   free(lower.ssa_to_vreg);
   agx_apple9_vir_finish(&lower.program);
   util_dynarray_fini(&loads);
   util_dynarray_fini(&stores);
   util_dynarray_fini(&atomics);
   return false;
}

struct apple9_preamble_eligibility {
   /* 0 unknown, 1 rejected/visiting, 2 accepted. Memoize shared DAGs. */
   uint8_t *state;
};

static bool
apple9_preamble_source(nir_src *src, void *data);

static bool
apple9_preamble_instruction(const nir_instr *instr, const void *data)
{
   const struct apple9_preamble_eligibility *eligibility = data;
   nir_def *def = nir_instr_def((nir_instr *)instr);
   if (!def || (def->bit_size != 32 && def->bit_size != 16 && def->bit_size != 1))
      return false;
   uint8_t *state = &eligibility->state[def->index];
   if (*state)
      return *state == 2;
   *state = 1;
   bool accepted = false;
   switch (instr->type) {
   case nir_instr_type_load_const:
      accepted = true;
      break;
   case nir_instr_type_alu:
      accepted = apple9_instruction_is_in_subset((nir_instr *)instr, false) &&
                 nir_foreach_src((nir_instr *)instr, apple9_preamble_source, (void *)data);
      break;
   case nir_instr_type_intrinsic:
      accepted = nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_ubo &&
                 nir_foreach_src((nir_instr *)instr, apple9_preamble_source, (void *)data);
      break;
   default:
      break;
   }
   *state = accepted ? 2 : 1;
   return accepted;
}

static bool
apple9_preamble_source(nir_src *src, void *data)
{
   return apple9_preamble_instruction(nir_def_instr(src->ssa), data);
}

static bool
apple9_preamble_avoid(const nir_instr *instr, const void *data)
{
   nir_def *def = nir_instr_def((nir_instr *)instr);
   /* Boolean and half intermediates can move with an expression, but only
    * 32-bit results have a validated argument-window transfer. */
   return !def || def->bit_size != 32 || !apple9_preamble_instruction(instr, data);
}

static void
apple9_preamble_def_size(nir_def *def, unsigned *size, unsigned *align,
                        nir_preamble_class *class_)
{
   *size = def->num_components;
   *align = 1;
   *class_ = nir_preamble_class_general;
}

static float
apple9_preamble_cost(nir_instr *instr, const void *data)
{
   if (!apple9_preamble_instruction(instr, data))
      return 0;
   return instr->type == nir_instr_type_intrinsic ? 8 : 1;
}

static float
apple9_preamble_rewrite_cost(nir_def *def, const void *unused)
{
   return def->num_components;
}

static bool
apple9_compile_dag(nir_shader *nir, struct agx_shader_part *out,
                  struct agx_apple9_compute_profile *profile,
                  const struct agx_apple9_varying_layout *varyings,
                  const char **reason)
{
   apple9_prepare_dag(nir);
   struct apple9_buffer_map resources = {0};
   if (!apple9_collect_buffer_map(nir, &resources, reason))
      return false;

   nir_shader *main = nir_shader_clone(NULL, nir);
   if (!main)
      return apple9_compile_dag_body(nir, out, profile, varyings, &resources,
                                    false, reason);
   /* Avoid spending uniform words on unused components of frontend UBO
    * vectors. Removing leading components can also reduce the alignment:
    * vec4 at byte 16 may become vec3 at byte 20. Re-legalize that access
    * before selecting its memory format or extracting a preamble. */
   nir_opt_shrink_vectors(main, true);
   nir_opt_constant_folding(main);
   apple9_legalize_buffer_accesses(main);
   nir_function_impl *main_impl = nir_shader_get_entrypoint(main);
   nir_index_ssa_defs(main_impl);
   struct apple9_preamble_eligibility eligibility = {
      .state = calloc(MAX2(main_impl->ssa_alloc, 1), sizeof(uint8_t)),
   };
   if (!eligibility.state) {
      ralloc_free(main);
      return apple9_compile_dag_body(nir, out, profile, varyings, &resources,
                                    false, reason);
   }
   const nir_opt_preamble_options options = {
      .cb_data = &eligibility,
      .def_size = apple9_preamble_def_size,
      .preamble_storage_size[nir_preamble_class_general] =
         AGX_APPLE9_UNIFORM_COUNT -
         (nir->info.stage == MESA_SHADER_COMPUTE
             ? agx_apple9_compute_root_words(resources.count)
             : AGX_APPLE9_GRAPHICS_ROOT_WORDS),
      .instr_cost_cb = apple9_preamble_cost,
      .rewrite_cost_cb = apple9_preamble_rewrite_cost,
      .avoid_instr_cb = apple9_preamble_avoid,
   };
   unsigned sizes[nir_preamble_num_classes] = {0};
   struct agx_shader_part setup = {0}, candidate = {0};
   struct agx_apple9_compute_profile candidate_profile = {0};
   const char *candidate_reason = NULL;
   nir_shader *preamble = NULL;
   if (main && nir_opt_preamble(main, &options, sizes)) {
      nir_function_impl *impl = nir_shader_get_preamble(main);
      /* Setup currently admits ordinary linear ALU/UBO work. Masked setup and
       * scratch have independent execution contracts and are not assumed. */
      if (impl && !apple9_cf_list_has_control_flow(&impl->body)) {
         preamble = nir_shader_create(NULL, nir->info.stage, nir->options);
         preamble->info.stage = nir->info.stage;
         preamble->info.float_controls_execution_mode = nir->info.float_controls_execution_mode;
         memcpy(preamble->info.workgroup_size, nir->info.workgroup_size,
                sizeof(preamble->info.workgroup_size));
         nir_function *fn = nir_function_create(preamble, "main");
         fn->is_entrypoint = true;
         fn->impl = nir_function_impl_clone(preamble, impl);
         fn->impl->function = fn;
         nir_opt_dce(main);
         nir_opt_dce(preamble);
         if (apple9_compile_dag_body(preamble, &setup, NULL, NULL, &resources,
                                    true, &candidate_reason) &&
             !setup.info.scratch_size &&
             setup.info.main_size > 4 &&
             setup.info.main_size <= AGX_APPLE9_MAX_PREAMBLE_BYTES &&
             apple9_compile_dag_body(main, &candidate,
                                    profile ? &candidate_profile : NULL,
                                    varyings, &resources, false, &candidate_reason)) {
            unsigned size = setup.info.main_size;
            uint8_t *binary = realloc(candidate.binary, candidate.info.binary_size + size);
            if (binary) {
               memcpy(binary + candidate.info.binary_size, setup.binary, size);
               candidate.info.apple9_preamble_offset = candidate.info.binary_size;
               candidate.info.apple9_preamble_size = size;
               candidate.info.binary_size += size;
               candidate.binary = binary;
               *out = candidate;
               if (profile) {
                  candidate_profile.preamble_offset = out->info.apple9_preamble_offset;
                  candidate_profile.preamble_size = size;
                  *profile = candidate_profile;
               }
               free(eligibility.state);
               free(setup.binary);
               ralloc_free(preamble);
               ralloc_free(main);
               return true;
            }
         }
      }
   }
   if (getenv("AGX_APPLE9_TRACE") && candidate_reason)
      fprintf(stderr, "APPLE9_PREAMBLE_FALLBACK %s\n", candidate_reason);
   free(setup.binary);
   free(candidate.binary);
   free(eligibility.state);
   ralloc_free(preamble);
   ralloc_free(main);
   return apple9_compile_dag_body(nir, out, profile, varyings, &resources,
                                 false, reason);
}

/* Internal shaders can arrive with generic Asahi options. Preserve operations
 * handled by this backend when running algebraic cleanup after legalization. */
static bool
apple9_opt_algebraic(nir_shader *nir)
{
   const nir_shader_compiler_options *saved = nir->options;
   nir_shader_compiler_options options = *saved;
   options.lower_pack_unorm_2x16 = false;
   options.lower_pack_unorm_4x8 = false;
   options.lower_unpack_unorm_2x16 = false;
   options.lower_unpack_unorm_4x8 = false;
   options.lower_unpack_snorm_2x16 = false;
   options.lower_unpack_snorm_4x8 = false;
   nir->options = &options;
   bool progress = nir_opt_algebraic(nir);
   nir->options = saved;
   return progress;
}

static void
apple9_lower_idiv(nir_shader *nir)
{
   /* Optimize constants before quotient/refinement lowering. High products
    * remain semantic NIR and select the upper word of a native wide multiply. */
   bool optimized = nir_opt_idiv_const(nir, 32);
   const nir_lower_idiv_options idiv = {0};
   if (nir_lower_idiv(nir, &idiv) || optimized) {
      agx_nir_lower_accurate_frcp(nir);
      const nir_shader_compiler_options *original_options = nir->options;
      nir_shader_compiler_options alu_options = *original_options;
      alu_options.lower_mul_high = false;
      alu_options.lower_extract_byte = true;
      alu_options.lower_extract_word = true;
      alu_options.lower_iabs = true;
      alu_options.has_fcanonicalize = false;
      nir->options = &alu_options;
      nir_lower_alu(nir);
      apple9_opt_algebraic(nir);
      nir_opt_algebraic_late(nir);
      nir->options = original_options;
   }
}

bool
agx_compile_apple9_tiny(nir_shader *nir, struct agx_shader_part *out,
                        struct agx_apple9_compute_profile *profile,
                        const char **reason_out)
{
   /* Make every source-level continue an ordinary structured masked region
    * ending at the loop latch. This upstream NIR pass preserves SSA and leaves
    * one backedge, matching the Apple9 loop machine directly. */
   nir_lower_continue_constructs(nir);

   /* Internal compute shaders may use function temporaries just like API
    * frontends. Normalize them to the SSA form consumed by this backend. */
   nir_lower_vars_to_ssa(nir);
   bool progress;
   do {
      progress = nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_remove_phis(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);
   apple9_lower_idiv(nir);
   agx_nir_lower_apple9_math(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_lower_all_phis_to_scalar(nir);

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
   unsigned nr_targets;
   bool valid;
};

/* This backend emits complete fragment programs, so the common coverage pass
 * always finalizes depth/stencil tests in this shader part. */
static bool
apple9_lower_coverage_sysval(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_rewrite_uses(&intr->def, nir_imm_intN_t(b, ~0u, intr->def.bit_size));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
apple9_lower_blend_constant(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_blend_const_color_rgba)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = nir_load_ubo(b, 4, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING), nir_imm_int(b, 0),
      .align_mul = 16, .range = 16);
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Sampler bias is per-draw state, independent of a shader's explicit bias.
 * Keep this in FP32, matching the rest of the Apple9 texture operands. */
bool
agx_nir_lower_apple9_sampler_state(
   nir_shader *nir, const struct agx_apple9_sampler_key key[32],
   struct agx_apple9_texture_mapping *mapping, const char **reason)
{
   memset(mapping, 0, sizeof(*mapping));
   uint8_t sampler_alias[32];
   memset(sampler_alias, 0xff, sizeof(sampler_alias));
   for (unsigned i = 0; i < 32; ++i) {
      mapping->samplers[i] = i;
   }
   nir_lower_samplers(nir);
   const nir_lower_tex_options options = {.lower_txp = ~0u, .lower_1d = true};
   nir_lower_tex(nir, &options);
   if (!agx_nir_lower_apple9_texture_offsets(nir, key, reason))
      return false;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   uint32_t samplers = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         if (tex->texture_index >= 32 || tex->sampler_index >= 32) {
            *reason = "Apple9 texture binding exceeds 32 entries";
            return false;
         }
         samplers |= BITFIELD_BIT(tex->sampler_index);
      }
   }
   nir_builder b = nir_builder_create(impl);
   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         if (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
             tex->op != nir_texop_txl && tex->op != nir_texop_txd)
            continue;
         const struct agx_apple9_sampler_key *state = &key[tex->sampler_index];
         if (!state->flags)
            continue;
         if ((tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
              tex->sampler_dim != GLSL_SAMPLER_DIM_3D) || tex->is_array || tex->is_shadow ||
             tex->def.num_components != 4 || tex->def.bit_size != 32) {
            *reason = "Apple9 border lowering requires a float 2D sample";
            return false;
         }
         int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
         if (coord < 0) {
            *reason = "Apple9 border sample has no coordinates";
            return false;
         }
         b.cursor = nir_before_instr(instr);
         nir_def *xy[3];
         for (unsigned c = 0; c < tex->coord_components; ++c) {
            xy[c] = nir_channel(&b, tex->src[coord].src.ssa, c);
            if (state->flags & (AGX_APPLE9_CLAMP_S << c))
               xy[c] = nir_fsat(&b, xy[c]);
         }
         nir_src_rewrite(&tex->src[coord].src, nir_vec(&b, xy, tex->coord_components));
         progress = true;
         if (!(state->flags & AGX_APPLE9_CUSTOM_BORDER))
            continue;
         unsigned s = tex->sampler_index;
         if (sampler_alias[s] == 0xff) {
            if (samplers == UINT32_MAX) {
               *reason = "Apple9 custom border needs an additional sampler entry";
               return false;
            }
            unsigned alias = ffs(~samplers) - 1;
            sampler_alias[s] = alias;
            samplers |= BITFIELD_BIT(alias);
            mapping->samplers[alias] = s;
            mapping->white_samplers |= BITFIELD_BIT(alias);
         }
         b.cursor = nir_after_instr(instr);
         nir_tex_instr *weight = nir_instr_as_tex(nir_instr_clone(nir, instr));
         weight->sampler_index = sampler_alias[s];
         nir_builder_instr_insert(&b, &weight->instr);
         nir_def *border = nir_imm_vec4(&b, state->border[0], state->border[1],
                                       state->border[2], state->border[3]);
         /* Match hk_shader.c: filtering is linear, so interpolating samples
          * with borders 0 and 1 gives the requested custom border. */
         nir_def *result = nir_flrp(&b, &tex->def, &weight->def, border);
         nir->info.flrp_lowered = false;
         nir_def_rewrite_uses_after(&tex->def, result);
      }
   }
   nir_progress(progress, impl, nir_metadata_control_flow);
   return true;
}

static nir_def *
apple9_texture_info(nir_builder *b, unsigned texture, unsigned field)
{
   return nir_load_ubo(b, 1, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_TEXTURE_INFO_OFFSET +
                    texture * AGX_APPLE9_TEXTURE_INFO_STRIDE + field * 4),
      .align_mul = 4, .range = 4);
}

/* Array shadow sampling uses the explicit LOD parameter contract. Derive
 * LOD from only the spatial coordinates; neither layer nor comparator is
 * a spatial derivative. The common texture pass turns these gradients into
 * LOD while preserving the ordinary sampler bias. */
static bool
apple9_lower_array_shadow_gradients(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex ||
       b->shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (!tex->is_array || !tex->is_shadow ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb))
      return false;
   b->cursor = nir_before_instr(instr);
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *xy = nir_trim_vector(b, tex->src[coord].src.ssa, 2);
   nir_def *dx = nir_ddx(b, xy), *dy = nir_ddy(b, xy);
   int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (bias >= 0) {
      nir_def *scale = nir_fexp2(b, tex->src[bias].src.ssa);
      dx = nir_fmul(b, dx, scale);
      dy = nir_fmul(b, dy, scale);
      nir_tex_instr_remove_src(tex, bias);
   }
   tex->op = nir_texop_txd;
   nir_tex_instr_add_src(tex, nir_tex_src_ddx, dx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, dy);
   return true;
}

static bool
apple9_lower_texel_fetch(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_txf || tex->is_shadow ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
        tex->sampler_dim != GLSL_SAMPLER_DIM_3D))
      return false;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (coord < 0 || lod < 0)
      return false;
   /* The read instruction consumes integer spatial coordinates. It uses a
    * private sampler descriptor, just as the common AGX texel-fetch path does. */
   tex->sampler_index = APPLE9_FETCH_SAMPLER;
   return true;
}

static bool
apple9_lower_texture_queries_and_layers(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->texture_index >= 32)
      return false;
   b->cursor = nir_before_instr(instr);
   if (tex->op == nir_texop_txs || tex->op == nir_texop_query_levels ||
       tex->op == nir_texop_texture_samples) {
      nir_def *result;
      if (tex->op == nir_texop_txs) {
         int src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
         nir_def *lod = src < 0 ? nir_imm_int(b, 0) : tex->src[src].src.ssa;
         lod = nir_umin(b, lod, nir_imm_int(b, 31));
         nir_def *size[3];
         for (unsigned c = 0; c < tex->def.num_components; ++c) {
            bool layer = tex->is_array && c + 1 == tex->def.num_components;
            size[c] = apple9_texture_info(b, tex->texture_index, layer ? 2 : c);
            if (!layer)
               size[c] = nir_umax(b, nir_ushr(b, size[c], lod), nir_imm_int(b, 1));
         }
         result = nir_vec(b, size, tex->def.num_components);
      } else
         result = apple9_texture_info(b, tex->texture_index,
                                      tex->op == nir_texop_query_levels ? 3 : 4);
      nir_def_rewrite_uses(&tex->def, result);
      nir_instr_remove(instr);
      return true;
   }
   if (tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
       tex->op != nir_texop_txf) {
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (coord < 0)
         return false;
      nir_def *xyz = tex->src[coord].src.ssa;
      nir_def *layer = nir_f2u32(b, nir_fmax(b,
         nir_fround_even(b, nir_channel(b, xyz, 2)), nir_imm_float(b, 0)));
      layer = nir_umin(b, layer,
         nir_iadd_imm(b, apple9_texture_info(b, tex->texture_index, 2), -1));
      nir_src_rewrite(&tex->src[coord].src,
         nir_vec3(b, nir_channel(b, xyz, 0), nir_channel(b, xyz, 1), layer));
      return true;
   }
   return false;
}

static bool
apple9_lower_sampler_bias(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb &&
       tex->op != nir_texop_txd && tex->op != nir_texop_txl)
      return false;
   if (tex->sampler_index >= AGX_APPLE9_SAMPLER_BIAS_COUNT)
      return false; /* Rejected by texture binding validation. */

   b->cursor = nir_before_instr(instr);
   nir_def *bias = nir_load_ubo(b, 1, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_SAMPLER_BIAS_OFFSET + tex->sampler_index * 4),
      .align_mul = 4, .range = 4);
   if (tex->op == nir_texop_txd) {
      nir_def *scale = nir_fexp2(b, bias);
      for (unsigned i = 0; i < tex->num_srcs; ++i) {
         if (tex->src[i].src_type == nir_tex_src_ddx ||
             tex->src[i].src_type == nir_tex_src_ddy)
            nir_src_rewrite(&tex->src[i].src,
               nir_fmul(b, tex->src[i].src.ssa, scale));
      }
   } else if (tex->op == nir_texop_tex) {
      tex->op = nir_texop_txb;
      nir_tex_instr_add_src(tex, nir_tex_src_bias, bias);
   } else {
      int idx = nir_tex_instr_src_index(tex, tex->op == nir_texop_txl
                                              ? nir_tex_src_lod : nir_tex_src_bias);
      nir_src_rewrite(&tex->src[idx].src,
         nir_fadd(b, tex->src[idx].src.ssa, bias));
   }
   return true;
}

static bool
apple9_blend_factor_supported(unsigned factor)
{
   switch (factor) {
   case PIPE_BLENDFACTOR_ZERO:
   case PIPE_BLENDFACTOR_ONE:
   case PIPE_BLENDFACTOR_SRC_COLOR:
   case PIPE_BLENDFACTOR_INV_SRC_COLOR:
   case PIPE_BLENDFACTOR_DST_COLOR:
   case PIPE_BLENDFACTOR_INV_DST_COLOR:
   case PIPE_BLENDFACTOR_SRC_ALPHA:
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA:
   case PIPE_BLENDFACTOR_DST_ALPHA:
   case PIPE_BLENDFACTOR_INV_DST_ALPHA:
   case PIPE_BLENDFACTOR_CONST_COLOR:
   case PIPE_BLENDFACTOR_INV_CONST_COLOR:
   case PIPE_BLENDFACTOR_CONST_ALPHA:
   case PIPE_BLENDFACTOR_INV_CONST_ALPHA:
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
      return true;
   default:
      return false;
   }
}

/* The validated color store consumes RGBA8. Express conversion as ordinary NIR
 * so register allocation and numerical behavior do not depend on a captured
 * native color-pack sequence. */
static void
apple9_collect_color(nir_shader *nir)
{
   /* Output assignments are ordinary mutable shader state. Collect them
    * before packing or blending so component writes and branch-local writes
    * produce one final color for each surviving invocation. */
   nir_lower_returns(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_variable *color[8] = {0};
   nir_variable *depth = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_output &&
             nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_DEPTH) {
            if (!depth)
               depth = nir_local_variable_create(impl, glsl_float_type(), "depth");
            b.cursor = nir_before_instr(instr);
            nir_store_var(&b, depth, intr->src[0].ssa, 1);
            nir_instr_remove(instr);
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_store_output ||
             nir_intrinsic_io_semantics(intr).location < FRAG_RESULT_DATA0 ||
             nir_intrinsic_io_semantics(intr).location >= FRAG_RESULT_DATA0 + 8 ||
             intr->src[0].ssa->bit_size != 32 ||
             !nir_src_is_const(intr->src[1]) ||
             nir_intrinsic_component(intr) + intr->num_components > 4)
            continue;
         unsigned rt = nir_intrinsic_io_semantics(intr).location - FRAG_RESULT_DATA0 +
                       nir_src_as_uint(intr->src[1]);
         if (rt >= 8)
            continue;
         if (!color[rt])
            color[rt] = nir_local_variable_create(impl, glsl_vec4_type(), "color");
         b.cursor = nir_before_instr(instr);
         unsigned component = nir_intrinsic_component(intr);
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; ++c)
            channels[c] = c >= component && c < component + intr->num_components
               ? nir_channel(&b, intr->src[0].ssa, c - component)
               : nir_undef(&b, 1, 32);
         nir_store_var(&b, color[rt], nir_vec(&b, channels, 4),
                       nir_intrinsic_write_mask(intr) << component);
         nir_instr_remove(instr);
      }
   }
   if (depth) {
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, depth), nir_imm_int(&b, 0),
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DEPTH, .num_slots = 1});
   }
   for (int rt = 7; rt >= 0; --rt) {
      if (!color[rt])
         continue;
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, color[rt]), nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_DATA0 + rt, .num_slots = 1});
      nir_progress(true, impl, nir_metadata_control_flow);
   }
}

static bool
apple9_blend_channel_reads_destination(enum pipe_blend_func func,
                                       enum pipe_blendfactor src,
                                       enum pipe_blendfactor dst)
{
   if (func == PIPE_BLEND_MIN || func == PIPE_BLEND_MAX ||
       dst != PIPE_BLENDFACTOR_ZERO)
      return true;

   switch (util_blendfactor_without_invert(src)) {
   case PIPE_BLENDFACTOR_DST_COLOR:
   case PIPE_BLENDFACTOR_DST_ALPHA:
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
      return true;
   default:
      return false;
   }
}

static bool
apple9_lower_color(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   struct apple9_color_lower *lower = data;
   if (nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_DEPTH)
      return false;
   unsigned rt = nir_intrinsic_io_semantics(intr).location - FRAG_RESULT_DATA0;
   if (rt < 8 && rt >= lower->nr_targets) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   unsigned component = nir_intrinsic_component(intr);
   if (rt >= 8 || intr->src[0].ssa->bit_size != 32 || !intr->num_components ||
       component + intr->num_components > 4) {
      lower->valid = false;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   const struct agx_apple9_blend *blend =
      lower->blend ? &lower->blend[rt] : NULL;
   enum pipe_format format = blend && blend->format != PIPE_FORMAT_NONE
                               ? blend->format : PIPE_FORMAT_R8G8B8A8_UNORM;
   enum pipe_format tile_format = agx_apple9_color_tile_format(format);
   bool fp16 = agx_apple9_color_is_half(tile_format);
   bool integer = util_format_is_pure_integer(format);
   const struct util_format_channel_description *channel =
      &util_format_description(tile_format)->channel[0];
   unsigned bits = channel->size;
   bool raw = integer || bits == 32;
   bool packed_format = agx_apple9_color_is_packed(tile_format);
   unsigned components = agx_apple9_color_components(format);
   unsigned words = agx_apple9_color_words(format), offset = 0;
   for (unsigned i = 0; i < rt; ++i)
      offset += agx_apple9_color_words(lower->blend ? lower->blend[i].format
                                                  : PIPE_FORMAT_NONE);
   nir_def *color = intr->src[0].ssa;
   /* Undef optimization can trim unwritten channels from the final collected
    * output. Materialize those undefined values as zero before RGBA packing. */
   if (component || intr->num_components != 4 ||
       nir_intrinsic_write_mask(intr) != 15) {
      nir_def *channels[4];
      for (unsigned c = 0; c < 4; ++c) {
         bool written =
            c >= component && c < component + intr->num_components &&
            (nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c - component));
         channels[c] = written ? nir_channel(b, color, c - component)
                               : nir_imm_float(b, 0);
      }
      color = nir_vec(b, channels, 4);
   }
   if (blend && !blend->colormask && !blend->alpha_to_coverage) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (blend) {
      if (blend->unsupported || (!blend->advanced_mode &&
          (blend->rgb_func > PIPE_BLEND_MAX ||
          blend->alpha_func > PIPE_BLEND_MAX ||
          !apple9_blend_factor_supported(blend->rgb_src) ||
          !apple9_blend_factor_supported(blend->rgb_dst) ||
          !apple9_blend_factor_supported(blend->alpha_src) ||
          !apple9_blend_factor_supported(blend->alpha_dst)))) {
         lower->valid = false;
         return false;
      }
   }
   unsigned samples = blend ? MAX2(blend->samples, 1) : 1;
   if (samples != 1 && samples != 2 && samples != 4) {
      lower->valid = false;
      return false;
   }
   /* A raw tile word must preserve masked channels, and blending may need a
    * different destination for each sample. Otherwise the same packed result
    * can be written to every covered sample in one masked store. */
   unsigned component_mask = BITFIELD_MASK(components);
   bool reads_destination = blend &&
      (((blend->colormask & component_mask) != component_mask) ||
       (!integer &&
        (blend->advanced_mode ||
         apple9_blend_channel_reads_destination(blend->rgb_func,
                                                blend->rgb_src, blend->rgb_dst) ||
         apple9_blend_channel_reads_destination(blend->alpha_func,
                                                blend->alpha_src, blend->alpha_dst))));
   bool sample_loop = samples > 1 && reads_destination;
   enum pipe_format raw_format = agx_apple9_color_raw_format(words);
   nir_def *src_color = color;
   nir_def *coverage = samples > 1 ? nir_load_sample_mask_in(b) : nir_imm_int(b, 1);
   if (blend && blend->disabled_samples)
      coverage = nir_iand_imm(b, coverage, ~blend->disabled_samples);
   if (blend && blend->alpha_to_coverage && rt == 0) {
      nir_def *alpha = nir_channel(b, color, 3);
      alpha = nir_fmin(b, nir_fmax(b, alpha, nir_imm_float(b, 0)), nir_imm_float(b, 1));
      nir_def *count = nir_f2u32(b, nir_fadd_imm(b, nir_fmul_imm(b, alpha, samples), 0.5));
      nir_def *mask = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), count), -1);
      nir_demote_samples(b, nir_u2u16(b, nir_inot(b, mask)));
      coverage = nir_iand(b, coverage, mask);
   }
   if (blend && blend->alpha_to_one)
      src_color = nir_vector_insert_imm(b, src_color, nir_imm_float(b, 1), 3);
   nir_variable *sample_index = NULL;
   nir_def *sample_mask = coverage;
   if (sample_loop) {
      sample_index = nir_local_variable_create(b->impl, glsl_uint_type(), "color sample");
      nir_store_var(b, sample_index, nir_imm_int(b, 0), 1);
      nir_push_loop(b)->control = nir_loop_control_dont_unroll;
      nir_def *index = nir_load_var(b, sample_index);
      nir_push_if(b, nir_uge_imm(b, index, samples));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);
      sample_mask = nir_ishl(b, nir_imm_int(b, 1), index);
   }
   {
      color = src_color;
      if (blend) {
         nir_def *packed_dst = reads_destination
            ? nir_load_local_pixel_agx(b, words, 32, sample_mask,
                 .base = 4 * offset, .format = raw_format)
            : nir_undef(b, words, 32);
         nir_def *channels[4];
         nir_def *normalized_dst = !packed_format && !raw && !fp16 && bits == 8
            ? nir_unpack_unorm_4x8(b, packed_dst) : NULL;
         for (unsigned c = 0; c < (packed_format ? 0 : 4); ++c) {
            if (c >= components) {
               channels[c] = integer ? nir_imm_int(b, c == 3) : nir_imm_float(b, c == 3);
               continue;
            }
            nir_def *value = nir_channel(b, packed_dst, (bits * c) / 32);
            if (bits < 32) {
               value = nir_ushr_imm(b, value, (bits * c) % 32);
               value = integer && channel->type == UTIL_FORMAT_TYPE_SIGNED
                  ? nir_ishr_imm(b, nir_ishl_imm(b, value, 32 - bits), 32 - bits)
                  : nir_iand_imm(b, value, BITFIELD_MASK(bits));
            }
            channels[c] = normalized_dst ? nir_channel(b, normalized_dst, c)
               : raw ? value : fp16 ? apple9_unpack_half(b, value)
               : nir_fmul_imm(b, nir_u2f32(b, value), 1.0 / 255.0);
         }
         nir_def *dst;
         if (format == PIPE_FORMAT_R11G11B10_FLOAT) {
            nir_def *r = apple9_unpack_half(b, nir_ishl_imm(b, nir_iand_imm(b, packed_dst, 0x7ff), 4));
            nir_def *g = apple9_unpack_half(b, nir_ishl_imm(b, nir_iand_imm(b, nir_ushr_imm(b, packed_dst, 11), 0x7ff), 4));
            nir_def *bl = apple9_unpack_half(b, nir_ishl_imm(b, nir_ushr_imm(b, packed_dst, 22), 5));
            dst = nir_vec4(b, r, g, bl, nir_imm_float(b, 1));
         } else
            dst = packed_format ? nir_format_unpack_rgba(b, packed_dst, format)
                                : nir_vec(b, channels, 4);
         if (util_format_is_srgb(format))
            dst = nir_vector_insert_imm(b, nir_format_srgb_to_linear(b, dst),
                                       nir_channel(b, dst, 3), 3);
         const nir_lower_blend_rt rt = {
            .format = format,
            .rgb = {blend->rgb_func, blend->rgb_src, blend->rgb_dst},
            .alpha = {blend->alpha_func, blend->alpha_src, blend->alpha_dst},
            .colormask = blend->colormask,
            .advanced_blend = blend->advanced_mode != 0,
            .blend_mode = blend->advanced_mode,
            .overlap = blend->advanced_overlap,
            .src_premultiplied = blend->src_premultiplied,
            .dst_premultiplied = blend->dst_premultiplied,
         };
         if (!integer)
            color = nir_color_blend(b, color, NULL, dst, &rt, false);
         color = nir_color_mask(b, color, dst, blend->colormask);
      }
      if (util_format_is_srgb(format))
         color = nir_vector_insert_imm(b, nir_format_linear_to_srgb(b, color),
                                      nir_channel(b, color, 3), 3);
      nir_def *packed[4] = {nir_imm_int(b, 0), nir_imm_int(b, 0),
                            nir_imm_int(b, 0), nir_imm_int(b, 0)};
      bool unorm8 = !raw && !fp16 && !packed_format;
      if (unorm8) {
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; ++c)
            channels[c] = c < components ? nir_channel(b, color, c)
                                          : nir_imm_float(b, 0);
         packed[0] = nir_pack_unorm_4x8(b, nir_vec(b, channels, 4));
      }
      for (unsigned c = 0; c < (packed_format || unorm8 ? 0 : components); ++c) {
         nir_def *v = nir_channel(b, color, c);
         if (raw) {
            if (bits < 32) {
               if (channel->type == UTIL_FORMAT_TYPE_SIGNED)
                  v = nir_imin(b, nir_imax(b, v, nir_imm_int(b, -(1 << (bits - 1)))),
                               nir_imm_int(b, (1 << (bits - 1)) - 1));
               else
                  v = nir_umin(b, v, nir_imm_int(b, BITFIELD_MASK(bits)));
               v = nir_iand_imm(b, v, BITFIELD_MASK(bits));
            }
            packed[(bits * c) / 32] = nir_ior(b, packed[(bits * c) / 32],
               nir_ishl_imm(b, v, (bits * c) % 32));
         } else if (fp16) {
            v = apple9_pack_half(b, v);
            packed[c / 2] = nir_ior(b, packed[c / 2], nir_ishl_imm(b, v, 16 * (c & 1)));
         }
      }
      if (packed_format)
         packed[0] = nir_format_pack_rgba(b, format, color);
      nir_store_local_pixel_agx(b, nir_vec(b, packed, words),
         nir_iand(b, coverage, sample_mask), nir_undef(b, 2, 16),
         .base = 4 * offset, .format = raw_format,
         .write_mask = BITFIELD_MASK(words));
   }
   if (sample_loop) {
      nir_store_var(b, sample_index, nir_iadd_imm(b, nir_load_var(b, sample_index), 1), 1);
      nir_pop_loop(b, NULL);
   }
   nir_instr_remove(&intr->instr);
   return true;
}

/* Vertex channels preserve integer values; packed 32-bit formats are
 * extracted with ordinary NIR shifts and masks. */
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
      if (!ch->size)
         return false;
      if (desc->block.bits == 32 && ch->size <= 32 &&
          ch->shift + ch->size <= 32 &&
          (ch->type == UTIL_FORMAT_TYPE_UNSIGNED || ch->type == UTIL_FORMAT_TYPE_SIGNED))
         continue;
      if (ch->shift % ch->size)
         return false;
      if (ch->type == UTIL_FORMAT_TYPE_FLOAT && (ch->size == 16 || ch->size == 32))
         continue;
      if ((ch->type == UTIL_FORMAT_TYPE_UNSIGNED || ch->type == UTIL_FORMAT_TYPE_SIGNED) &&
          (ch->size == 8 || ch->size == 16 || ch->size == 32))
         continue;
      return false;
   }
   return true;
}

/* The capture job is an ordinary vertex program over the decomposed stream.
 * Poly's GPU input assembly preserves API vertex/instance identities, including
 * duplicate indexed and strip vertices, independently of the hardware cache. */
struct apple9_xfb_lower {
   nir_def *raw_id;
   nir_def *vertex, *instance, *base_instance, *draw_id;
};

static bool
apple9_lower_xfb_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data)
{
   struct apple9_xfb_lower *lower = data;
   nir_def *replacement = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_vertex_id_zero_base:
      if (&intr->def != lower->raw_id)
         replacement = lower->vertex;
      break;
   case nir_intrinsic_load_instance_id:
      replacement = lower->instance;
      break;
   case nir_intrinsic_load_base_instance:
      replacement = lower->base_instance;
      break;
   case nir_intrinsic_load_draw_id:
      replacement = lower->draw_id;
      break;
   default:
      break;
   }
   if (replacement) {
      nir_def_replace(&intr->def, replacement);
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_io_xfb xfb = nir_intrinsic_io_xfb(intr);
   for (unsigned c = 0; c < 4; ++c) {
      if (!xfb.out[c].num_components)
         continue;
      unsigned buffer = xfb.out[c].buffer;
      unsigned stride = b->shader->info.xfb_stride[buffer] * 4;
      nir_component_mask_t mask = nir_component_mask(xfb.out[c].num_components);
      mask = (mask << c) >> nir_intrinsic_component(intr);
      nir_def *value = nir_channels(b, intr->src[0].ssa, mask);
      nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, lower->raw_id, stride),
                                     xfb.out[c].offset * 4);
      nir_store_ssbo(b, value,
                     nir_imm_int(b, AGX_APPLE9_XFB_BUFFER_BASE + buffer),
                     offset, .align_mul = 4,
                     .write_mask = nir_component_mask(value->num_components));
   }
   /* Capture precedes clipping and rasterization. The capture-only pass has
    * no raster outputs, irrespective of the API program's varying interface. */
   nir_instr_remove(&intr->instr);
   return true;
}

static void
apple9_lower_xfb(nir_shader *nir, const struct agx_apple9_vertex_layout *layout)
{
   nir_io_add_intrinsic_xfb_info(nir);
   nir_builder b =
      nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
   struct apple9_xfb_lower lower = {0};
   lower.raw_id = nir_load_vertex_id_zero_base(&b);
   nir_def *params[8];
   for (unsigned i = 0; i < ARRAY_SIZE(params); ++i)
      params[i] =
         nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, AGX_APPLE9_XFB_PARAMS),
                       nir_imm_int(&b, i * 4), .align_mul = 4,
                       .access = ACCESS_NON_WRITEABLE);
   unsigned vertices = mesa_vertices_per_prim(layout->xfb_mode);
   nir_def *logical = nir_iadd(&b, lower.raw_id, params[4]);
   nir_def *instance = nir_udiv(&b, logical, params[5]);
   nir_def *within = nir_umod(&b, logical, params[5]);
   nir_def *prim = nir_udiv_imm(&b, within, vertices);
   nir_def *vert = nir_umod_imm(&b, within, vertices);
   nir_def *mode = nir_imm_int(&b, layout->xfb_mode);
   nir_def *id = within;
   if (vertices == 2)
      id = poly_vertex_id_for_line_class(&b, mode, prim, vert,
                                         nir_udiv_imm(&b, params[5], vertices));
   else if (vertices == 3)
      id = poly_vertex_id_for_tri_class(
         &b, mode, prim, vert, nir_imm_bool(&b, layout->xfb_flatshade_first));
   id = nir_iadd(&b, id, params[0]);
   if (layout->xfb_index_size) {
      id = nir_load_ssbo(&b, 1, layout->xfb_index_size * 8,
                         nir_imm_int(&b, AGX_APPLE9_XFB_INDICES),
                         nir_imul_imm(&b, id, layout->xfb_index_size),
                         .align_mul = layout->xfb_index_size,
                         .access = ACCESS_NON_WRITEABLE);
      id = nir_iadd(&b, nir_u2u32(&b, id), params[1]);
   }
   lower.vertex = id;
   lower.instance = nir_iadd(&b, instance, params[6]);
   lower.base_instance = params[2];
   lower.draw_id = params[3];
   nir_shader_intrinsics_pass(nir, apple9_lower_xfb_intrinsic,
                              nir_metadata_control_flow, &lower);
   b.cursor = nir_after_impl(nir_shader_get_entrypoint(nir));
   nir_store_output(
      &b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0), .write_mask = 15,
      .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   nir->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_POS);
   nir->xfb_info = NULL;
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
   unsigned divisor = lower->layout->divisor[attribute];
   nir_def *index = divisor ? nir_udiv_imm(b, nir_load_instance_id(b), divisor)
                            : nir_load_vertex_id(b);
   if (divisor && lower->layout->capture_xfb)
      index = nir_iadd(b, index, nir_load_base_instance(b));
   nir_def *offset = nir_iadd_imm(b,
      nir_imul_imm(b, index, lower->layout->stride[attribute]),
      lower->layout->offset[attribute]);
   nir_def *components[4];
   for (unsigned c = 0; c < intr->num_components; ++c) {
      unsigned component = nir_intrinsic_component(intr) + c;
      unsigned swizzle = desc->swizzle[component];
      if (swizzle >= PIPE_SWIZZLE_0) {
         unsigned value = swizzle == PIPE_SWIZZLE_1;
         components[c] = util_format_is_pure_integer(lower->layout->format[attribute])
                            ? nir_imm_int(b, value) : nir_imm_float(b, value);
         continue;
      }
      const struct util_format_channel_description *ch = &desc->channel[swizzle];
      bool packed = (ch->size % 8) || (ch->shift % 8);
      unsigned bytes = packed ? 4 : ch->size / 8;
      unsigned add = lower->layout->offset[attribute] + (packed ? 0 : ch->shift / 8);
      if (!bytes || (lower->layout->stride[attribute] % bytes) || (add % bytes)) {
         lower->valid = false;
         return false;
      }
      nir_def *value = nir_load_ubo(b, 1, packed ? 32 : ch->size,
         nir_imm_int(b, 32 + lower->layout->buffer[attribute]),
         nir_iadd_imm(b, offset, packed ? 0 : ch->shift / 8),
         .align_mul = bytes, .range = ~0u);
      if (packed) {
         value = nir_ushr_imm(b, value, ch->shift);
         value = ch->type == UTIL_FORMAT_TYPE_SIGNED
                    ? nir_ishr_imm(b, nir_ishl_imm(b, value, 32 - ch->size), 32 - ch->size)
                    : nir_iand_imm(b, value, BITFIELD_MASK(ch->size));
      }
      if (ch->type == UTIL_FORMAT_TYPE_FLOAT) {
         if (ch->size == 16)
            value = apple9_unpack_half(b, nir_u2u32(b, value));
      } else if (ch->pure_integer) {
         value = ch->type == UTIL_FORMAT_TYPE_SIGNED ? nir_i2i32(b, value)
                                                    : nir_u2u32(b, value);
      } else {
         unsigned bits = ch->size;
         if (ch->type == UTIL_FORMAT_TYPE_SIGNED) {
            value = nir_i2i32(b, value);
            value = nir_i2f32(b, value);
            if (ch->normalized)
               value = nir_fmax(b, nir_fmul_imm(b, value,
                  1.0 / ((UINT64_C(1) << (bits - 1)) - 1)), nir_imm_float(b, -1));
         } else {
            value = nir_u2u32(b, value);
            value = nir_u2f32(b, value);
            if (ch->normalized)
               value = nir_fmul_imm(b, value, 1.0 / ((UINT64_C(1) << bits) - 1));
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
       ((read << component) & ~0xfu)) {
      *(bool *)data = false;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *xy = nir_u2f32(b, nir_load_pixel_coord(b));
   nir_def *values[4];
   for (unsigned i = 0; i < intr->def.num_components; ++i) {
      if (!(read & BITFIELD_BIT(i)))
         values[i] = nir_undef(b, 1, 32);
      else if (component + i == 2)
         values[i] = nir_load_frag_coord_z(b);
      else if (component + i == 3)
         values[i] = nir_load_frag_coord_w(b);
      else
         values[i] = nir_channel(b, xy, component + i);
   }
   nir_def_rewrite_uses(&intr->def, nir_vec(b, values, intr->def.num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
apple9_lower_point_coord(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if ((intr->intrinsic != nir_intrinsic_load_input &&
        intr->intrinsic != nir_intrinsic_load_interpolated_input) ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PNTC)
      return false;
   unsigned offset = intr->intrinsic == nir_intrinsic_load_input ? 0 : 1;
   unsigned component = nir_intrinsic_component(intr);
   if (!nir_src_is_const(intr->src[offset]) || nir_src_as_uint(intr->src[offset]) ||
       intr->def.bit_size != 32 || component + intr->def.num_components > 4) {
      *(bool *)data = false;
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *xy = nir_load_point_coord(b);
   nir_def *values[4];
   for (unsigned i = 0; i < intr->def.num_components; ++i) {
      unsigned c = component + i;
      values[i] = c < 2 ? nir_channel(b, xy, c) : nir_imm_float(b, c == 3);
   }
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
         if (fragment
                ? (intr->intrinsic != nir_intrinsic_load_interpolated_input &&
                   intr->intrinsic != nir_intrinsic_load_input)
                : intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned location = nir_intrinsic_io_semantics(intr).location;
         if (!fragment && (location == VARYING_SLOT_POS || location == VARYING_SLOT_PSIZ))
            continue;
         if (fragment && location == VARYING_SLOT_POS) {
            *reason = "Apple9 fragment window-position input is not implemented";
            return false;
         }
         unsigned offset = intr->intrinsic == nir_intrinsic_load_input ? 0 : 1;
         if (!nir_src_is_const(intr->src[offset]) ||
             nir_src_as_uint(intr->src[offset]) >= 32)
            goto unsupported;
         location += nir_src_as_uint(intr->src[offset]);
         unsigned component = nir_intrinsic_component(intr);
         nir_alu_type type = fragment ? nir_intrinsic_dest_type(intr)
                                      : nir_intrinsic_src_type(intr);
         if (type != nir_type_float32 &&
             !((!fragment || intr->intrinsic == nir_intrinsic_load_input) &&
               (type == nir_type_int32 || type == nir_type_uint32)))
            goto unsupported;
         if (!agx_apple9_varying_supported(location) ||
             component + intr->num_components > 4)
            goto unsupported;
         unsigned mask = fragment ? nir_def_components_read(&intr->def)
                                  : nir_intrinsic_write_mask(intr);
         layout->mask[location] |= mask << component;
      }
   }
   for (unsigned i = 0; i < ARRAY_SIZE(layout->mask); ++i) {
      if (producer && (layout->mask[i] & ~producer->mask[i])) {
         *reason = "Apple9 fragment input is not written by the vertex stage";
         return false;
      }
      layout->count += util_bitcount(layout->mask[i]);
   }
   if (producer)
      *layout = *producer;
   unsigned count = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(layout->mask); ++i) {
      if ((layout->mask[i] & ~0xf) ||
          (layout->mask[i] && !agx_apple9_varying_supported(i)))
         goto unsupported;
      count += util_bitcount(layout->mask[i]);
   }
   if (layout->count != count || count > AGX_APPLE9_MAX_VARYING_COMPONENTS) {
      *reason = "Apple9 export publication currently supports 96 user scalars";
      return false;
   }
   return true;
unsupported:
   *reason = "Apple9 graphics requires constant-indexed 32-bit user varyings";
   return false;
}

static bool
apple9_compile_graphics(nir_shader *nir, struct agx_shader_part *out,
                        const struct agx_apple9_vertex_layout *layout,
                        const struct agx_apple9_varying_layout *producer,
                        const struct agx_apple9_blend *blend,
                        unsigned nr_targets, const char **reason)
{
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   if (reason)
      *reason = NULL;
   /* Legacy gl_FragColor broadcasts to each enabled draw buffer. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      nir_lower_fragcolor(nir, nr_targets);
   /* Move varying arrays into local storage before flattening the interface.
    * Hardware slots remain static while shader-selected array elements are
    * handled by the same SSA lowering as local arrays. */
   nir_lower_io_vars_to_temporaries(nir, nir_shader_get_entrypoint(nir),
      nir->info.stage == MESA_SHADER_VERTEX ? nir_var_shader_out : nir_var_shader_in);
   nir_lower_global_vars_to_local(nir);
   nir_split_var_copies(nir);
   nir_lower_var_copies(nir);
   nir_lower_io(nir, nir_var_shader_in | nir_var_shader_out, apple9_io_size,
                nir_lower_io_use_interpolated_input_intrinsics);
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            nir_io_semantics semantics = nir_intrinsic_io_semantics(intr);
            if (semantics.location == FRAG_RESULT_COLOR) {
               semantics.location = FRAG_RESULT_DATA0;
               nir_intrinsic_set_io_semantics(intr, semantics);
               for (unsigned rt = 1; rt < nr_targets; ++rt) {
                  nir_instr *copy = nir_instr_clone(nir, &intr->instr);
                  semantics.location = FRAG_RESULT_DATA0 + rt;
                  nir_intrinsic_set_io_semantics(nir_instr_as_intrinsic(copy),
                                                 semantics);
                  nir_instr_insert(nir_before_instr(&intr->instr), copy);
               }
            }
         }
      }
      if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_COLOR)) {
         nir->info.outputs_written &= ~BITFIELD64_BIT(FRAG_RESULT_COLOR);
         nir->info.outputs_written |= BITFIELD64_MASK(nr_targets)
                                      << FRAG_RESULT_DATA0;
      }
   }

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
      if (layout && layout->capture_xfb)
         apple9_lower_xfb(nir, layout);
   }
   /* Use the same structured mask/loop model as compute. Continuations become
    * masked regions with one backedge before instruction selection. */
   nir_lower_continue_constructs(nir);
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      apple9_collect_color(nir);
   /* Small local arrays can live in SSA registers. Expand variable indices
    * through the standard NIR decision trees; larger arrays remain subject
    * to the backend's resource checks until scratch support is available. */
   nir_lower_array_deref_of_vec(nir, nir_var_function_temp, NULL,
                               nir_lower_indirect_array_deref_of_vec_load |
                               nir_lower_indirect_array_deref_of_vec_store);
   nir_lower_indirect_derefs_to_if_else_trees(nir, nir_var_function_temp, 32);
   /* Frontends may supply vector NIR or pre-simplified scalar NIR. Expose
    * scalar constants to standard algebraic cleanup before expanding math
    * and output packing, rather than relying on a caller-specific pipeline.
    */
   nir_lower_vars_to_ssa(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_lower_all_phis_to_scalar(nir);
   bool progress;
   do {
      progress = apple9_opt_algebraic(nir);
      progress |= nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_remove_phis(nir);
      progress |= nir_opt_undef(nir);
      progress |= nir_opt_loop_unroll(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_dce(nir);
      progress |= nir_opt_cse(nir);
   } while (progress);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      struct apple9_color_lower color = {
         .blend = blend, .nr_targets = nr_targets, .valid = true};
      nir_shader_intrinsics_pass(nir, apple9_lower_color,
                                 nir_metadata_none, &color);
      nir_lower_vars_to_ssa(nir);
      nir_lower_all_phis_to_scalar(nir);
      if (blend && blend[0].disabled_samples) {
         nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
         nir_demote_samples(&b, nir_imm_intN_t(&b, blend[0].disabled_samples, 16));
      }
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      agx_nir_lower_discard_zs_emit(nir);
      agx_nir_lower_sample_mask(nir);
      nir_shader_intrinsics_pass(nir, apple9_lower_coverage_sysval,
                                 nir_metadata_control_flow, NULL);
      if (!color.valid) {
         if (reason)
            *reason = "Apple9 render requires FP32 color outputs and standard single-source blending";
         return false;
      }
   }
   nir_shader_intrinsics_pass(nir, apple9_lower_blend_constant,
                              nir_metadata_control_flow, NULL);
   nir_opt_dce(nir);
   nir_lower_samplers(nir);
   /* Compatibility GL supplies projective texture coordinates. Reuse the
    * common NIR quotient lowering before selecting ordinary 2D sampling. */
   const nir_lower_tex_options tex_options = {
      .lower_txp = ~0u, .lower_1d = true,
      .lower_invalid_implicit_lod = true,
      .lower_txf_offset = true,
      .lower_txd_array = true, .lower_txd_3d = true,
      .lower_txd_cube_map = true, .lower_txd_shadow = true,
   };
   nir_shader_instructions_pass(nir, apple9_lower_array_shadow_gradients,
                                nir_metadata_control_flow, NULL);
   nir_lower_tex(nir, &tex_options);
   nir_shader_instructions_pass(nir, apple9_lower_texel_fetch,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_texture_queries_and_layers,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_sampler_bias,
                                nir_metadata_control_flow, NULL);
   agx_nir_lower_apple9_texture_lod(nir);
   apple9_lower_idiv(nir);
   /* Format lowering can introduce powers and other high-level ALU ops. */
   apple9_opt_algebraic(nir);
   agx_nir_lower_apple9_math(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_opt_constant_folding(nir);
   nir_opt_copy_prop(nir);
   nir_opt_dce(nir);
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      bool valid = true;
      nir_shader_intrinsics_pass(nir, apple9_lower_window_position,
                                 nir_metadata_control_flow, &valid);
      nir_shader_intrinsics_pass(nir, apple9_lower_point_coord,
                                 nir_metadata_control_flow, &valid);
      if (!valid) {
         *reason = "Apple9 fragment window position requires FP32 XYZW components";
         return false;
      }
      nir_lower_alu_to_scalar(nir, NULL, NULL);
      nir_opt_copy_prop(nir);
      nir_opt_dce(nir);
   }
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_sample_mask_agx)
            nir->info.fs.uses_discard = true;
      }
   }
   uint32_t texture_mask = 0, sampler_mask = 0;
   bool uses_texel_fetch = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         if (!apple9_texture_supported(tex)) {
            *reason = "Apple9 requires supported FP32 texture operations";
            return false;
         }
         texture_mask |= BITFIELD_BIT(tex->texture_index);
         uses_texel_fetch |= apple9_texture_uses_fetch_sampler(tex);
         if (!apple9_texture_uses_fetch_sampler(tex))
            sampler_mask |= BITFIELD_BIT(tex->sampler_index);
      }
   }
   if (util_bitcount(texture_mask) > AGX_APPLE9_GRAPHICS_MAX_TEXTURES ||
       util_bitcount(sampler_mask) + uses_texel_fetch >
          AGX_APPLE9_GRAPHICS_MAX_SAMPLERS) {
      *reason = "Apple9 supports sixteen textures and seventeen physical samplers";
      return false;
   }
   struct apple9_buffer_map buffers = {0};
   if (!apple9_collect_buffer_map(nir, &buffers, reason))
      return false;
   bool valid_buffers = buffers.count <= AGX_APPLE9_MAX_GRAPHICS_BUFFERS;
   for (unsigned i = 0; i < buffers.count; ++i)
      valid_buffers &=
         buffers.resource[i].kind == AGX_APPLE9_COMPUTE_RESOURCE_SSBO
            ? buffers.resource[i].binding < 32
            : buffers.resource[i].kind == AGX_APPLE9_COMPUTE_RESOURCE_UBO &&
                 (buffers.resource[i].binding < (layout ? 64 : 32) ||
                  buffers.resource[i].binding ==
                     AGX_APPLE9_GRAPHICS_SYSVAL_BINDING);
   if (!valid_buffers) {
      if (reason)
         *reason =
            "Apple9 graphics requires at most 32 supported buffer bindings";
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
          apple9_compile_graphics(nir, out, NULL, NULL, NULL, 1, reason);
}

bool
agx_compile_apple9_fragment_blend(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, struct agx_shader_part *out,
   const char **reason)
{
   return nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, varyings, blend, 1, reason);
}

bool
agx_compile_apple9_fragment_mrt(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   const struct agx_apple9_blend *blend, unsigned nr_targets,
   struct agx_shader_part *out, const char **reason)
{
   return nr_targets > 0 && nr_targets <= 8 &&
          nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, varyings, blend, nr_targets, reason);
}

bool
agx_compile_apple9_fragment_inputs(
   nir_shader *nir, const struct agx_apple9_varying_layout *varyings,
   struct agx_shader_part *out, const char **reason)
{
   return nir->info.stage == MESA_SHADER_FRAGMENT &&
          apple9_compile_graphics(nir, out, NULL, varyings, NULL, 1, reason);
}

bool
agx_compile_apple9_vertex(nir_shader *nir, struct agx_shader_part *out,
                          const char **reason)
{
   return nir->info.stage == MESA_SHADER_VERTEX &&
          apple9_compile_graphics(nir, out, NULL, NULL, NULL, 1, reason);
}

bool
agx_compile_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout,
   struct agx_shader_part *out, const char **reason)
{
   return nir->info.stage == MESA_SHADER_VERTEX &&
          apple9_compile_graphics(nir, out, layout, NULL, NULL, 1, reason);
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
