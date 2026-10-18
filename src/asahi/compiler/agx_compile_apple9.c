/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "util/u_debug.h"
#include "agx_compile_apple9.h"
#include "agx_apple9_ir.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_lower_blend.h"
#include "compiler/nir/nir_xfb_info.h"
#include "poly/cl/libpoly.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/log.h"
#include "gallium/include/pipe/p_defines.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DEBUG_GET_ONCE_OPTION(cached_agx_apple9_asm, "AGX_APPLE9_ASM", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_dump_binary,
                      "AGX_APPLE9_DUMP_BINARY", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_gated, "AGX_APPLE9_GATED", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_hoist_samples,
                      "AGX_APPLE9_HOIST_SAMPLES", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_licm, "AGX_APPLE9_LICM", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_licm_speculate,
                      "AGX_APPLE9_LICM_SPECULATE", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_no_descriptor_hoist,
                      "AGX_APPLE9_NO_DESCRIPTOR_HOIST", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_no_promote,
                      "AGX_APPLE9_NO_PROMOTE", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_no_sampler_hoist,
                      "AGX_APPLE9_NO_SAMPLER_HOIST", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_no_uniform_move,
                      "AGX_APPLE9_NO_UNIFORM_MOVE", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_safe_hoist_stages,
                      "AGX_APPLE9_SAFE_HOIST_STAGES", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_stats, "AGX_APPLE9_STATS", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_trace_final_nir,
                      "AGX_APPLE9_TRACE_FINAL_NIR", NULL)
DEBUG_GET_ONCE_OPTION(cached_agx_apple9_trace_nir, "AGX_APPLE9_TRACE_NIR", NULL)

DEBUG_GET_ONCE_NUM_OPTION(apple9_hoist_loads, "AGX_APPLE9_HOIST_LOADS", 0)

/* Apple9 compute compilation uses one NIR-to-VIR-to-machine pipeline. */

struct apple9_emitter {
   struct util_dynarray bytes;
   unsigned instructions;
};

/* The backend proved a compute program has no observable effect: callers
 * may run nothing for it instead of failing the pipeline. */
const char agx_apple9_compute_without_effects[] =
   "Apple9 compute program has no stores or other side effects";

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
static bool
apple9_trivial_zero(nir_scalar scalar)
{
   for (unsigned iteration = 0; iteration < 32; ++iteration) {
      scalar = nir_scalar_chase_movs(scalar);
      if (nir_scalar_is_const(scalar))
         return nir_scalar_as_uint(scalar) == 0;
      if (nir_def_instr_type(scalar.def) != nir_instr_type_alu)
         return false;

      nir_op op = nir_scalar_alu_op(scalar);
      if (op == nir_op_vec2 || op == nir_op_vec3 || op == nir_op_vec4) {
         scalar = nir_scalar_chase_alu_src(scalar, scalar.comp);
         continue;
      }
      if (op != nir_op_iadd)
         return false;

      /* A sum is trivially zero only if both terms are. Test a constant term
       * first so ordinary address chains stop at their displacement. */
      nir_scalar left = nir_scalar_chase_movs(nir_scalar_chase_alu_src(scalar, 0));
      nir_scalar right = nir_scalar_chase_movs(nir_scalar_chase_alu_src(scalar, 1));
      if (nir_scalar_is_const(left) && nir_scalar_as_uint(left) != 0)
         return false;
      if (nir_scalar_is_const(right) && nir_scalar_as_uint(right) != 0)
         return false;
      if (!apple9_trivial_zero(left))
         return false;
      scalar = right;
   }
   return false;
}

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

      /* Only an addition with a term that reduces to zero is peeled; the
       * other term is chased further. Deciding that needs a zero test, not
       * a full chase of both terms. */
      if (op == nir_op_iadd) {
         nir_scalar left = nir_scalar_chase_alu_src(scalar, 0);
         nir_scalar right = nir_scalar_chase_alu_src(scalar, 1);
         if (apple9_trivial_zero(left)) {
            scalar = right;
            continue;
         }
         if (apple9_trivial_zero(right)) {
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
   case nir_intrinsic_load_layer_id:
      base = 0xc3;
      components = 1;
      break;
   case nir_intrinsic_load_tile_id_agx:
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
   if (op == nir_intrinsic_load_scratch || op == nir_intrinsic_store_scratch)
      return (nir_mem_access_size_align){.num_components = 1, .bit_size = 32,
         .align = 4, .shift = nir_mem_access_shift_method_scalar};
   unsigned alignment = nir_combined_align(align_mul, align_offset);
   /* 64-bit data, such as a descriptor's pointer, moves as dword tuples so
    * both halves arrive in one register pair. */
   const bool dwords = bit_size == 32 || bit_size == 64;
   if (dwords && alignment >= 4 && bytes >= 12)
      return (nir_mem_access_size_align){.num_components = MIN2(bytes / 4, 4),
                                         .bit_size = 32, .align = 4};
   if (dwords && alignment >= 4 && bytes >= 8)
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
   case nir_op_bit_count:
   case nir_op_ufind_msb:
   case nir_op_bitfield_reverse: {
      nir_def *source = nir_ssa_for_alu_src(b, alu, 0);
      if (source->bit_size >= 32)
         return false;
      /* A narrow SSA value need not define its containing GPR's high bits.
       * Extend before counting/scanning; reversal also removes those zeros
       * from the result. 64-bit variants use common pair lowering. */
      value = nir_build_alu(b, alu->op, nir_u2u32(b, source), NULL, NULL, NULL);
      if (alu->op == nir_op_bitfield_reverse)
         value = nir_ushr_imm(b, value, 32 - source->bit_size);
      value = nir_u2uN(b, value, alu->def.bit_size);
      break;
   }
   case nir_op_b2f16:
      value = nir_bcsel(b, nir_ssa_for_alu_src(b, alu, 0),
                        nir_imm_float16(b, 1.0f), nir_imm_float16(b, 0.0f));
      break;
   case nir_op_b2b32:
      /* Boolean values in memory use all bits set for true. Boolean SSA
       * values in this backend use one, so make the conversion explicit. */
      value = nir_i2iN(b, nir_ineg(b, nir_b2i32(b, nir_ssa_for_alu_src(b, alu, 0))),
                       alu->def.bit_size);
      break;
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
   case nir_op_iadd_sat:
   case nir_op_isub_sat: {
      /* Signed overflow flips the result's sign away from the operands'.
       * Saturate toward the first operand's sign. */
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *c = nir_ssa_for_alu_src(b, alu, 1);
      const unsigned bits = a->bit_size;
      bool add = alu->op == nir_op_iadd_sat;
      nir_def *result = add ? nir_iadd(b, a, c) : nir_isub(b, a, c);
      nir_def *overflow = add
         ? nir_iand(b, nir_ixor(b, result, a), nir_ixor(b, result, c))
         : nir_iand(b, nir_ixor(b, a, c), nir_ixor(b, a, result));
      nir_def *limit = nir_ixor(b, nir_ishr_imm(b, a, bits - 1),
                                nir_imm_intN_t(b, u_intN_max(bits), bits));
      value = nir_bcsel(b, nir_ilt_imm(b, overflow, 0), limit, result);
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

/* Int64 lowering expands a global address base64 + zext(offset32) into
 *    lo = iadd(base_lo, offset)
 *    hi = iadd(base_hi, b2i32(ult(lo, base_lo)))
 * which equals the pair (base_lo, base_hi) plus the zero-extended offset for
 * either low addend, so any match is exact. Successive constant additions
 * compare each low word with the previous one, while reassociation can fold
 * their constants into a single low addition from the original base:
 *    lo = iadd(base_lo, k2)
 *    hi = iadd(hi1, b2i32(ult(lo, lo1)))
 * where (lo1, hi1) is base64 + k1. This is base64 + k1 + (k2 - k1) with each
 * step zero-extended, so it equals base64 + zext(k2) when k2 >= k1. Recover
 * the base pair and the offset so the offset can use the memory instruction's
 * index; the base then stays a stable pair (a loaded descriptor, or
 * uniforms). */
static bool
apple9_match_global_address(nir_scalar lo, nir_scalar hi, unsigned depth,
                            nir_scalar pair[2], nir_scalar *offset,
                            bool *constant, uint32_t *value)
{
   lo = apple9_chase_trivial(lo);
   hi = apple9_chase_trivial(hi);
   if (!nir_scalar_is_alu(lo) || nir_scalar_alu_op(lo) != nir_op_iadd ||
       !nir_scalar_is_alu(hi) || nir_scalar_alu_op(hi) != nir_op_iadd ||
       lo.def->bit_size != 32 || hi.def->bit_size != 32)
      return false;
   for (unsigned h = 0; h < 2; ++h) {
      nir_scalar carry = nir_scalar_chase_movs(nir_scalar_chase_alu_src(hi, h));
      if (!nir_scalar_is_alu(carry) || nir_scalar_alu_op(carry) != nir_op_b2i32)
         continue;
      nir_scalar cmp = nir_scalar_chase_movs(nir_scalar_chase_alu_src(carry, 0));
      if (!nir_scalar_is_alu(cmp) || nir_scalar_alu_op(cmp) != nir_op_ult ||
          !nir_scalar_equal(apple9_chase_trivial(nir_scalar_chase_alu_src(cmp, 0)), lo))
         continue;
      nir_scalar previous = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cmp, 1));
      nir_scalar previous_hi = nir_scalar_chase_alu_src(hi, 1 - h);
      for (unsigned l = 0; l < 2; ++l) {
         if (!nir_scalar_equal(nir_scalar_chase_movs(nir_scalar_chase_alu_src(lo, l)),
                               previous))
            continue;
         pair[0] = previous;
         pair[1] = previous_hi;
         *offset = nir_scalar_chase_alu_src(lo, 1 - l);
         /* An unsigned sum overflows exactly when it is below either addend,
          * so a carry against the constant addend (as 64-bit add lowering
          * emits for base + zext(k)) equally names the other addend as the
          * base word. Keep the constant as the offset, which the memory
          * instruction encodes directly, and the pointer as the base pair. */
         uint32_t k;
         if (apple9_const_u32(previous, &k) &&
             !apple9_const_u32(*offset, value)) {
            pair[0] = apple9_chase_trivial(*offset);
            *offset = previous;
         }
         *constant = apple9_const_u32(*offset, value);
         return true;
      }

      uint32_t k1, k2 = 0;
      bool inner_constant;
      nir_scalar inner_offset, root = lo;
      if (depth >= 4 ||
          !apple9_match_global_address(previous, previous_hi, depth + 1, pair,
                                       &inner_offset, &inner_constant, &k1) ||
          !inner_constant)
         continue;
      while (nir_scalar_is_alu(root) && nir_scalar_alu_op(root) == nir_op_iadd) {
         uint32_t addend;
         unsigned s = 0;
         while (s < 2 && !apple9_const_u32(nir_scalar_chase_alu_src(root, s), &addend))
            ++s;
         if (s == 2)
            break;
         k2 += addend;
         root = apple9_chase_trivial(nir_scalar_chase_alu_src(root, 1 - s));
      }
      if (nir_scalar_equal(root, apple9_chase_trivial(pair[0])) && k2 >= k1) {
         *constant = true;
         *value = k2;
         return true;
      }
   }
   return false;
}

static bool
apple9_split_global_address(nir_builder *b, nir_def *address, nir_def **base,
                            nir_def **offset)
{
   nir_scalar pair[2], scalar;
   bool constant;
   uint32_t value;
   if (!apple9_match_global_address(nir_get_scalar(address, 0),
                                    nir_get_scalar(address, 1), 0, pair,
                                    &scalar, &constant, &value))
      return false;
   *base = nir_vec_scalars(b, pair, 2);
   *offset = constant ? nir_imm_int(b, value) : nir_mov_scalar(b, scalar);
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
            /* An amul cannot overflow the variable it indexes, so for an
             * in-bounds access the 32-bit sum equals the hardware's 64-bit
             * index + displacement. Robust accesses still load out of
             * bounds before their result is discarded, so keep the proof. */
            if ((nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS) &&
                nir_scalar_is_alu(other) &&
                nir_scalar_alu_op(other) == nir_op_amul)
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
   nir_def *address;
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
   nir_def *address;
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
   nir_def *address;
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

/* Register-indexed textures and samplers carry their native table index as
 * a handle source, see apple9_lower_texture_handles. */
static unsigned
apple9_texture_handle_count(const nir_tex_instr *tex)
{
   return (nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0) +
          (nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0);
}

/* Immediate table slots used by a texture operation. */
static void
apple9_texture_table_masks(const nir_tex_instr *tex, uint32_t *textures,
                           uint32_t *samplers, bool *fetch)
{
   if (nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) < 0)
      *textures |= BITFIELD_BIT(tex->texture_index);
   if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0)
      return;
   if (apple9_texture_uses_fetch_sampler(tex))
      *fetch = true;
   else
      *samplers |= BITFIELD_BIT(tex->sampler_index);
}

static bool
apple9_texture_supported(const nir_tex_instr *tex)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_MS) {
      return tex->op == nir_texop_txf_ms && tex->coord_components == 4 &&
             !tex->is_shadow &&
             tex->num_srcs == 1 + apple9_texture_handle_count(tex) &&
             tex->src[0].src_type == nir_tex_src_coord &&
             tex->src[0].src.ssa->bit_size == 32 &&
             tex->src[0].src.ssa->num_components == 4 &&
             tex->def.bit_size == 32 && tex->def.num_components == 4 &&
             tex->texture_index < 32 && tex->sampler_index == APPLE9_FETCH_SAMPLER;
   }
   /* Offsets without a native encoding are expanded with sampler state
    * before selection. */
   const bool offset = nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0;
   if (offset && !agx_apple9_texture_native_offset(tex, NULL))
      return false;
   bool gather = tex->op == nir_texop_tg4;
   bool explicit_lod = tex->op == nir_texop_txl || tex->op == nir_texop_txb ||
                       tex->op == nir_texop_txf;
   bool gradient = tex->op == nir_texop_txd;
   bool array = tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_2D;
   bool cube_array = tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   bool volume = array || tex->sampler_dim == GLSL_SAMPLER_DIM_3D ||
                 tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   if ((!explicit_lod && !gradient && !gather && tex->op != nir_texop_tex) ||
       (gather && (tex->def.num_components != 4 || tex->is_gather_implicit_lod ||
                   tex->component > 3)) ||
       (!volume && tex->sampler_dim != GLSL_SAMPLER_DIM_2D) ||
       (tex->is_array && !array && !cube_array) ||
       (tex->is_shadow && tex->sampler_dim == GLSL_SAMPLER_DIM_3D) ||
       (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && gradient) ||
       tex->coord_components != (cube_array ? 4 : volume ? 3 : 2) ||
       tex->def.bit_size != 32 ||
       (tex->def.num_components != 4 && !(tex->is_shadow && tex->def.num_components == 1)) ||
       (tex->dest_type != nir_type_float32 && tex->dest_type != nir_type_int32 &&
        tex->dest_type != nir_type_uint32) ||
       tex->num_srcs != (gradient ? 3 : explicit_lod ? 2 : 1) + tex->is_shadow +
                        offset + apple9_texture_handle_count(tex) +
                        (nir_tex_instr_src_index(tex, nir_tex_src_backend2) >= 0) ||
       tex->texture_index >= 32 ||
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
             tex->src[source].src.ssa->num_components !=
                (tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 2))
            return false;
      }
   }
   int comparator = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   if (tex->is_shadow && (comparator < 0 || tex->src[comparator].src.ssa->bit_size != 32))
      return false;
   /* A packed minimum LOD accompanies 2D and 2D array gradients and biased
    * samples. */
   int min_lod = nir_tex_instr_src_index(tex, nir_tex_src_backend2);
   if (min_lod >= 0 && ((!gradient && tex->op != nir_texop_txb) ||
                        tex->is_shadow ||
                        tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
                        tex->src[min_lod].src.ssa->bit_size != 32))
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
      return apple9_texture_supported(nir_instr_as_tex(instr));
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
      case nir_op_b2i8:
      case nir_op_b2i16:
      case nir_op_b2i32:
      case nir_op_b2f32:
      case nir_op_bcsel:
      case nir_op_bit_count:
      case nir_op_ufind_msb:
      case nir_op_bitfield_reverse:
      case nir_op_inot:
      case nir_op_ineg:
      case nir_op_fabs:
      case nir_op_fneg:
      case nir_op_u2f16:
      case nir_op_i2f16:
      case nir_op_u2f32:
      case nir_op_i2f32:
      case nir_op_f2i32:
      case nir_op_f2u32:
      case nir_op_f2f16:
      case nir_op_f2f16_rtne:
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
           op == nir_intrinsic_load_front_face || op == nir_intrinsic_load_back_face_agx ||
           op == nir_intrinsic_ddx ||
           op == nir_intrinsic_ddy || op == nir_intrinsic_ddx_fine ||
           op == nir_intrinsic_ddy_fine || op == nir_intrinsic_ddx_coarse ||
           op == nir_intrinsic_ddy_coarse ||
           op == nir_intrinsic_load_barycentric_pixel ||
           op == nir_intrinsic_load_barycentric_centroid ||
           op == nir_intrinsic_load_barycentric_at_sample ||
           op == nir_intrinsic_load_barycentric_coord_pixel ||
           op == nir_intrinsic_load_barycentric_coord_centroid ||
           op == nir_intrinsic_load_barycentric_coord_at_sample ||
           op == nir_intrinsic_load_input_vertex ||
           op == nir_intrinsic_load_interpolated_input ||
           op == nir_intrinsic_load_input ||
           op == nir_intrinsic_load_coefficients_agx ||
           op == nir_intrinsic_load_tile_pixel_agx ||
           op == nir_intrinsic_load_local_pixel_agx ||
           op == nir_intrinsic_pack_format_agx ||
           op == nir_intrinsic_load_sample_mask_in ||
           op == nir_intrinsic_load_sample_mask ||
           op == nir_intrinsic_sample_mask_agx ||
           op == nir_intrinsic_store_zs_agx ||
           op == nir_intrinsic_store_output ||
           op == nir_intrinsic_store_uvs_agx ||
           op == nir_intrinsic_image_store_block_agx ||
           op == nir_intrinsic_store_local_pixel_agx ||
           op == nir_intrinsic_demote || op == nir_intrinsic_demote_if))
         return true;
      return op == nir_intrinsic_image_store_apple9_agx ||
             op == nir_intrinsic_load_global_invocation_id ||
             op == nir_intrinsic_load_workgroup_id ||
             op == nir_intrinsic_load_tile_id_agx ||
             op == nir_intrinsic_load_layer_id ||
             op == nir_intrinsic_load_local_invocation_id ||
             op == nir_intrinsic_load_local_invocation_index ||
             op == nir_intrinsic_load_workgroup_size ||
             op == nir_intrinsic_load_num_workgroups ||
             op == nir_intrinsic_load_subgroup_invocation ||
             op == nir_intrinsic_load_subgroup_id ||
             op == nir_intrinsic_load_subgroup_size ||
             op == nir_intrinsic_inclusive_scan ||
             op == nir_intrinsic_exclusive_scan ||
             op == nir_intrinsic_read_invocation ||
             op == nir_intrinsic_shuffle ||
             op == nir_intrinsic_ballot ||
             op == nir_intrinsic_load_scratch ||
             op == nir_intrinsic_store_scratch ||
             op == nir_intrinsic_load_shared ||
             op == nir_intrinsic_store_shared ||
             op == nir_intrinsic_shared_atomic ||
             op == nir_intrinsic_shared_atomic_swap ||
             op == nir_intrinsic_barrier ||
             op == nir_intrinsic_begin_invocation_interlock ||
             op == nir_intrinsic_end_invocation_interlock ||
             op == nir_intrinsic_load_ssbo || op == nir_intrinsic_load_ubo ||
             op == nir_intrinsic_load_preamble ||
             op == nir_intrinsic_store_preamble ||
             op == nir_intrinsic_store_ssbo ||
             op == nir_intrinsic_load_global_2x32 ||
             op == nir_intrinsic_store_global_2x32 ||
             op == nir_intrinsic_global_atomic_2x32 ||
             op == nir_intrinsic_global_atomic_swap_2x32 ||
             op == nir_intrinsic_ssbo_atomic ||
             op == nir_intrinsic_ssbo_atomic_swap;
   }
   case nir_instr_type_phi:
      return true;
   case nir_instr_type_jump: {
      nir_jump_type type = nir_instr_as_jump(instr)->type;
      return type == nir_jump_break || type == nir_jump_continue ||
             type == nir_jump_halt;
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
   unsigned mask_depth;
};

/* Varying exports of the current block, by exported value. A legal shader
 * may first write clip distances after all other outputs. Remember every
 * publication that can fit the bank so rasterizer copies can always reuse it.
 */
#define APPLE9_EXPORT_MEMO AGX_APPLE9_PUBLICATION_COUNT
struct apple9_export_memo {
   uint32_t value, export;
   bool raw;
};

struct apple9_dag_lower {
   struct agx_apple9_vir_program program;
   /* Set when one export per varying store would exceed the bank. */
   bool share_exports;
   struct apple9_export_memo export_memo[APPLE9_EXPORT_MEMO];
   unsigned export_memo_count;
   nir_block *export_memo_block;
   nir_shader *nir;
   uint32_t *ssa_to_vreg;
   struct agx_apple9_block **nir_blocks;
   unsigned ssa_map_count;
   uint32_t system_vreg[256];
   struct agx_apple9_block *system_block;
   uint32_t zero_vreg;
   struct agx_apple9_block *zero_block;
   /* A zero defined in the entry block dominates every use. */
   uint32_t entry_zero;
   struct agx_apple9_block *binding_pointer_block;
   uint32_t binding_pointer[AGX_APPLE9_MAX_GRAPHICS_BUFFERS];
   uint32_t perspective_reciprocal;
   struct agx_apple9_interp_mask linear_mask, flat_mask;
   struct agx_apple9_block *perspective_block;
   bool reads_z;
   bool writes_z;
   bool writes_s;
   /* Depth/stencil tests inside loops defer their release past the loop. */
   unsigned loop_depth;
   bool zs_release_pending;
   bool zs_export;
   bool reads_point_coord;
   bool reads_primitive_id;
   /* Barycentric weights (bit 0 perspective, bit 1 linear) and the varying
    * scalars read per vertex, which own one raw-vertex coefficient each. */
   uint8_t barycentric_mask;
   uint8_t per_vertex_count;
   uint8_t per_vertex[AGX_APPLE9_MAX_PER_VERTEX];
   const struct agx_apple9_varying_layout *varyings;
   unsigned position_mask;
   bool color_stores;
   bool tile_read;
   bool tile_write;
   bool tile_access;
   bool tile_coords;
   /* Fragment shader interlock holds the pixel acquired for writing. */
   bool interlock;
   uint32_t coverage_control;
   uint32_t coverage_mask;
   struct apple9_scalar_load *loads;
   unsigned load_count;
   struct apple9_buffer_atomic *atomics;
   unsigned atomic_count;
   unsigned argument_base;
   bool buffer_table;
   unsigned groups_argument;
   unsigned shared_argument;
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
   /* One zero from the entry block serves every block. Rematerializing a
    * zero per block made each constant memory index rewrite the register a
    * preceding store or load was still reading, costing a wait. */
   if (lower->entry_zero != AGX_APPLE9_VREG_INVALID)
      return lower->entry_zero;
   /* Otherwise reuse the zero only within its block, which need not dominate
    * later blocks. */
   if (lower->zero_vreg == AGX_APPLE9_VREG_INVALID ||
       lower->zero_block != lower->program.current_block) {
      lower->zero_vreg = apple9_dag_imm(lower, 0);
      lower->zero_block = lower->program.current_block;
   }
   return lower->zero_vreg;
}

static uint32_t
apple9_buffer_address(struct apple9_dag_lower *lower, unsigned argument)
{
   assert(lower->buffer_table);
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
   /* A read first emitted in one arm does not dominate sibling arms. */
   if (lower->system_block != lower->program.current_block) {
      lower->system_block = lower->program.current_block;
      memset(lower->system_vreg, 0xff, sizeof(lower->system_vreg));
   }
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

/* A global address held by an even-aligned pair of preamble uniforms is the
 * memory instruction's uniform base, as native Metal addresses its buffer
 * arguments (device_load ..., u0_u1, r1). The base field names the pair by
 * half its first uniform, so no GPR pair is formed. */
static bool
apple9_uniform_address(const struct apple9_dag_lower *lower, nir_def *address,
                       unsigned *binding)
{
   unsigned word[2];
   for (unsigned c = 0; c < 2; ++c) {
      nir_scalar s = apple9_chase_trivial(nir_get_scalar(address, c));
      if (s.def->bit_size != 32 ||
          nir_def_instr_type(s.def) != nir_instr_type_intrinsic ||
          nir_def_as_intrinsic(s.def)->intrinsic != nir_intrinsic_load_preamble)
         return false;
      word[c] = lower->preamble_base +
                nir_intrinsic_base(nir_def_as_intrinsic(s.def)) + s.comp;
   }
   if ((word[0] & 1) || word[1] != word[0] + 1 ||
       word[1] >= AGX_APPLE9_UNIFORM_COUNT)
      return false;
   *binding = word[0] / 2;
   return true;
}

static uint32_t
apple9_lower_global_address(struct apple9_dag_lower *lower, nir_def *address)
{
   uint32_t pair[2];
   for (unsigned c = 0; c < 2; ++c) {
      pair[c] = apple9_lower_dag_scalar(lower, nir_get_scalar(address, c));
      if (pair[c] == AGX_APPLE9_VREG_INVALID)
         return pair[c];
   }
   /* Adjacent values, such as a loaded pointer or descriptor, already form
    * the pair when they begin a tuple (apple9_shared_dataflow). */
   if (pair[1] == pair[0] + 1)
      return pair[0];
   return agx_apple9_vir_emit_collect(&lower->program, pair, 2);
}

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

   return apple9_dag_hidden_load(lower, lower->groups_argument, component);
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
      /* The equality bit turns the integer less-than select into equality,
       * as Metal emits it. Inequality swaps the arms. Validated on T8132:
       * 11,271 operator/switch/loop/integer/subgroup CTS cases, while
       * IGT with the bit is not equality.
       */
      bool not_equal = compare.relation == APPLE9_COMPARE_NOT_EQUAL;
      return apple9_emit_dag_select_raw(
         lower, cmp_a, cmp_b, not_equal ? if_false : if_true,
         not_equal ? if_true : if_false,
         AGX_APPLE9_SELECT_ILT | AGX_APPLE9_SELECT_EQUALITY);
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

   uint32_t result = apple9_emit_dag_select_raw(lower, cmp_a, cmp_b,
                                                if_true, if_false, immediate);
   if (result != AGX_APPLE9_VREG_INVALID &&
       compare.domain == APPLE9_COMPARE_FLOAT &&
       nir_scalar_chase_alu_src(predicate, 0).def->bit_size == 16) {
      lower->program.instructions[lower->program.instruction_count - 1]->encoding =
         AGX_APPLE9_ENC_HALF_COMPARE_SELECT;
   }
   return result;
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
      /* Rematerialize constant Booleans at each use, such as a phi edge copy
       * in its predecessor arm, rather than keeping one register live across
       * the program.
       */
      return apple9_dag_imm(lower, nir_scalar_as_uint(scalar) != 0);
   } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
              (nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_front_face ||
               nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_back_face_agx) &&
              lower->nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* The word read of selector 0xc5 reports back-facing. Convert to
       * NIR's front-facing Boolean, as in the upstream AGX convention. */
      uint32_t facing = apple9_dag_system(lower,
         (struct apple9_system_source){.selector = 0xc5});
      uint32_t zero = apple9_dag_zero(lower);
      bool back = nir_def_as_intrinsic(scalar.def)->intrinsic ==
                  nir_intrinsic_load_back_face_agx;
      uint32_t one = apple9_dag_imm(lower, 1);
      value = apple9_emit_dag_select_raw(lower, zero, facing,
         back ? one : zero, back ? zero : one, AGX_APPLE9_SELECT_ULT);
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
            /* With a comparison operand, one SELECT replaces its 0/1
             * materialization and the logic op: a | b = cmp ? 1 : b and
             * a & b = cmp ? b : 0. Chains of compares ORed together are common
             * in translated vertex shaders. Prefer a comparison no other user
             * has materialized; signed comparisons need biasing, so fuse them
             * only when this is their only use.
             */
            int fuse = -1;
            for (unsigned s = 0; op != nir_op_ixor && s < 2; ++s) {
               nir_scalar src =
                  apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, s));
               struct apple9_compare compare;
               if (nir_def_instr_type(src.def) != nir_instr_type_alu ||
                   !apple9_normalize_compare(nir_scalar_alu_op(src), &compare) ||
                   (compare.domain == APPLE9_COMPARE_SIGNED &&
                    !list_is_singular(&src.def->uses)))
                  continue;
               const unsigned src_key = src.def->index * 4 + src.comp;
               if (fuse < 0 || (src_key < lower->ssa_map_count &&
                                lower->ssa_to_vreg[src_key] ==
                                   AGX_APPLE9_VREG_INVALID))
                  fuse = s;
            }
            if (fuse >= 0) {
               uint32_t other = apple9_lower_bool_scalar(
                  lower, nir_scalar_chase_alu_src(scalar, 1 - fuse));
               uint32_t constant = op == nir_op_ior ? apple9_dag_imm(lower, 1)
                                                    : apple9_dag_zero(lower);
               nir_scalar compare = nir_scalar_chase_alu_src(scalar, fuse);
               if (other != AGX_APPLE9_VREG_INVALID &&
                   constant != AGX_APPLE9_VREG_INVALID)
                  value = op == nir_op_ior
                     ? apple9_emit_dag_select(lower, compare, constant, other)
                     : apple9_emit_dag_select(lower, compare, other, constant);
               break;
            }
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
            uint32_t if_true = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 1));
            uint32_t if_false = apple9_lower_bool_scalar(
               lower, nir_scalar_chase_alu_src(scalar, 2));
            nir_scalar predicate =
               apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 0));
            while (nir_def_instr_type(predicate.def) == nir_instr_type_alu &&
                   nir_scalar_alu_op(predicate) == nir_op_inot) {
               uint32_t temporary = if_true;
               if_true = if_false;
               if_false = temporary;
               predicate = apple9_chase_trivial(
                  nir_scalar_chase_alu_src(predicate, 0));
            }
            struct apple9_compare compare;
            if (nir_def_instr_type(predicate.def) == nir_instr_type_alu &&
                apple9_normalize_compare(nir_scalar_alu_op(predicate), &compare)) {
               value = apple9_emit_dag_select(lower, predicate, if_true, if_false);
               break;
            }
            uint32_t condition = apple9_lower_bool_scalar(lower, predicate);
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

/* Unsigned bitfield extract (source >> offset) & (2^width - 1); width 0
 * keeps every bit above offset, a plain logical shift. */
static uint32_t
apple9_dag_extract(struct apple9_dag_lower *lower, uint32_t source,
                   unsigned offset, unsigned width)
{
   if (source == AGX_APPLE9_VREG_INVALID || (!offset && !width))
      return source;
   return apple9_dag_emit(lower, AGX_APPLE9_VIR_USHR,
                          AGX_APPLE9_ENC_SHIFT_LOGICAL_IMMEDIATE, &source, 1,
                          offset | (width << AGX_APPLE9_SHIFT_WIDTH_SHIFT));
}

static uint32_t
apple9_dag_shift_variable(struct apple9_dag_lower *lower, nir_op op,
                          uint32_t source, nir_scalar shift, unsigned bits)
{
   uint32_t amount = apple9_lower_dag_scalar(lower, shift);
   if (source == AGX_APPLE9_VREG_INVALID || amount == AGX_APPLE9_VREG_INVALID)
      return AGX_APPLE9_VREG_INVALID;

   /* Hardware shifts saturate outside 0..31, whereas NIR masks the amount
    * to the operand width. Reuse an existing sufficiently narrow AND. */
   bool bounded = false;
   if (nir_def_instr_type(shift.def) == nir_instr_type_alu &&
       nir_scalar_alu_op(shift) == nir_op_iand) {
      for (unsigned s = 0; s < 2; ++s) {
         uint32_t mask;
         if (apple9_const_u32(nir_scalar_chase_alu_src(shift, s), &mask) &&
             mask < bits)
            bounded = true;
      }
   }
   if (!bounded) {
      uint32_t sources[] = {amount, apple9_dag_imm(lower, bits - 1)};
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
   for (unsigned i = 0; i < ARRAY_SIZE(layout->mask); ++i) {
      if (layout->group[i] < layout->group[semantic] ||
          (layout->group[i] == layout->group[semantic] && i < semantic))
         index += util_bitcount(layout->mask[i]);
   }
   return index;
}

/* Coefficient tuples are one asynchronous load, even when NIR consumes
 * multiple words. Publish the entire SSA tuple for later scalar consumers. */
static uint32_t
apple9_lower_coefficients(struct apple9_dag_lower *lower, nir_scalar scalar)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(scalar.def);
   if (lower->nir->info.stage != MESA_SHADER_FRAGMENT ||
       scalar.def->bit_size != 32 || scalar.comp >= 3 ||
       nir_intrinsic_interp_mode(intr) != INTERP_MODE_FLAT ||
       !nir_src_is_const(intr->src[0]))
      return AGX_APPLE9_VREG_INVALID;
   unsigned location = nir_intrinsic_io_semantics(intr).location +
                       nir_src_as_uint(intr->src[0]);
   int index = apple9_varying_index(lower->varyings, location,
                                   nir_intrinsic_component(intr));
   if (index < 0)
      return AGX_APPLE9_VREG_INVALID;
   agx_apple9_interp_mask_set(&lower->flat_mask, index);
   uint32_t base = agx_apple9_vir_emit_coefficients(&lower->program, index + 1);
   if (base == AGX_APPLE9_VREG_INVALID)
      return base;
   for (unsigned c = 0; c < 3; ++c)
      lower->ssa_to_vreg[intr->def.index * 4 + c] = base + c;
   return base + scalar.comp;
}

static bool
apple9_is_barycentric_coord(nir_intrinsic_op op)
{
   return op == nir_intrinsic_load_barycentric_coord_pixel ||
          op == nir_intrinsic_load_barycentric_coord_centroid ||
          op == nir_intrinsic_load_barycentric_coord_sample ||
          op == nir_intrinsic_load_barycentric_coord_at_sample ||
          op == nir_intrinsic_load_barycentric_coord_at_offset;
}

/* Coefficients after 1/W, the varyings, primitive ID, depth and the point
 * coordinates: two perspective and then two linear barycentric weights, as
 * used, followed by one raw-vertex coefficient per per-vertex scalar.
 */
static unsigned
apple9_extra_coefficient_base(const struct apple9_dag_lower *lower)
{
   return 1 + lower->varyings->count + lower->reads_primitive_id +
          lower->reads_z + 2 * lower->reads_point_coord;
}

static int
apple9_per_vertex_slot(const struct apple9_dag_lower *lower, int index)
{
   for (unsigned i = 0; i < lower->per_vertex_count; ++i) {
      if (lower->per_vertex[i] == index)
         return i;
   }
   return -1;
}

/* Interpolate one coefficient at the pixel centre, or at `position` when it
 * is valid, perspective-correct unless `linear`. */
static uint32_t
apple9_emit_interpolation(struct apple9_dag_lower *lower, unsigned coefficient,
                          uint32_t position, bool linear)
{
   bool at_position = position != AGX_APPLE9_VREG_INVALID;
   enum agx_apple9_encoding encoding = at_position ? AGX_APPLE9_ENC_ITER_COORD
                                               : AGX_APPLE9_ENC_ITER;
   if (linear) {
      return apple9_dag_emit(lower, AGX_APPLE9_VIR_ITER, encoding,
                             at_position ? &position : NULL, at_position,
                             coefficient);
   }
   uint32_t reciprocal;
   /* A reciprocal first emitted in one arm does not dominate sibling arms.
    * Reuse it only within the block where it was computed. */
   if (at_position || lower->perspective_block != lower->program.current_block) {
      uint32_t denominator = apple9_dag_emit(
         lower, AGX_APPLE9_VIR_ITER, encoding,
         at_position ? &position : NULL, at_position, 0);
      reciprocal = apple9_dag_emit(lower, AGX_APPLE9_VIR_FRCP,
         AGX_APPLE9_ENC_FLOAT_SPECIAL, &denominator, 1, 3);
      if (!at_position) {
         lower->perspective_reciprocal = reciprocal;
         lower->perspective_block = lower->program.current_block;
      }
   } else {
      reciprocal = lower->perspective_reciprocal;
   }
   uint32_t value = apple9_dag_emit(
      lower, AGX_APPLE9_VIR_ITER, encoding,
      at_position ? &position : NULL, at_position, coefficient);
   /* Keep raw varyings available to homogeneous clipping. Native shade-7
    * coefficients pair with a coefficient-aware projective multiply, which
    * handles the rasterizer's primitive-constant representation. */
   uint32_t src[] = {value, reciprocal};
   return apple9_dag_emit(lower, AGX_APPLE9_VIR_FMUL_PROJECT,
                          AGX_APPLE9_ENC_FLOAT2_PROJECT, src, 2, coefficient);
}

static uint32_t
apple9_lower_interpolated_input(struct apple9_dag_lower *lower,
                                nir_scalar scalar)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(scalar.def);
   bool flat = intr->intrinsic == nir_intrinsic_load_input;
   nir_intrinsic_instr *bary = flat ? NULL : nir_src_as_intrinsic(intr->src[0]);
   unsigned offset = flat ? 0 : 1;
   bool at_position = bary &&
      (bary->intrinsic == nir_intrinsic_load_barycentric_centroid ||
       bary->intrinsic == nir_intrinsic_load_barycentric_at_sample);
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
        (!bary || (!at_position && bary->intrinsic != nir_intrinsic_load_barycentric_pixel) ||
         (!linear && nir_intrinsic_interp_mode(bary) != INTERP_MODE_SMOOTH &&
          nir_intrinsic_interp_mode(bary) != INTERP_MODE_NONE)))) {
      lower->reason =
         "Apple9 fragment input requires center/centroid/sample smooth, linear, or flat 32-bit user varyings";
      return AGX_APPLE9_VREG_INVALID;
   }
   unsigned location = nir_intrinsic_io_semantics(intr).location +
                       nir_src_as_uint(intr->src[offset]);
   int index = apple9_varying_index(lower->varyings, location, component);
   if (index < 0 && flat && location == VARYING_SLOT_PRIMITIVE_ID &&
       component == 0 && lower->reads_primitive_id)
      return agx_apple9_vir_emit_iter_flat(&lower->program, lower->varyings->count + 1);
   if (index < 0) {
      lower->reason = "Apple9 fragment input is not written by the vertex stage";
      return AGX_APPLE9_VREG_INVALID;
   }
   if (flat) {
      agx_apple9_interp_mask_set(&lower->flat_mask, index);
      return agx_apple9_vir_emit_iter_flat(&lower->program, index + 1);
   }
   uint32_t position = at_position ? apple9_lower_dag_scalar(
      lower, nir_get_scalar(&bary->def, 0)) : AGX_APPLE9_VREG_INVALID;
   if (linear)
      agx_apple9_interp_mask_set(&lower->linear_mask, index);
   return apple9_emit_interpolation(lower, index + 1, position, linear);
}

/* Interpolation position for centroid and per-sample interpolation. */
static uint32_t
apple9_interpolation_position(struct apple9_dag_lower *lower,
                              nir_intrinsic_instr *bary, bool at_sample)
{
   uint32_t coverage;
   if (at_sample) {
      coverage = apple9_lower_dag_scalar(lower,
                                         nir_get_scalar(bary->src[0].ssa, 0));
   } else {
      coverage = apple9_dag_emit(lower, AGX_APPLE9_VIR_GET_SR,
                                 AGX_APPLE9_ENC_GET_COVERAGE, NULL, 0, 0x10c2);
   }
   return apple9_dag_emit(lower, AGX_APPLE9_VIR_INTERPOLATION_POSITION,
                          AGX_APPLE9_ENC_INTERPOLATION_POSITION, &coverage, 1,
                          at_sample);
}

/* The first two barycentric weights come from the rasterizer's barycentric
 * coefficients; the third is computed in NIR. */
static uint32_t
apple9_lower_barycentric_weight(struct apple9_dag_lower *lower,
                                nir_scalar scalar)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(scalar.def);
   bool linear = nir_intrinsic_interp_mode(intr) == INTERP_MODE_NOPERSPECTIVE;
   if (!lower->varyings || scalar.def->bit_size != 32 || scalar.comp >= 2 ||
       !(lower->barycentric_mask & (linear ? 2 : 1))) {
      lower->reason = "Apple9 barycentric weights take two 32-bit components";
      return AGX_APPLE9_VREG_INVALID;
   }
   unsigned coefficient = apple9_extra_coefficient_base(lower) + scalar.comp +
      (linear && (lower->barycentric_mask & 1) ? 2 : 0);
   uint32_t position = AGX_APPLE9_VREG_INVALID;
   if (intr->intrinsic == nir_intrinsic_load_barycentric_coord_centroid ||
       intr->intrinsic == nir_intrinsic_load_barycentric_coord_at_sample) {
      position = apple9_interpolation_position(
         lower, intr,
         intr->intrinsic == nir_intrinsic_load_barycentric_coord_at_sample);
   }
   return apple9_emit_interpolation(lower, coefficient, position, linear);
}

/* A per-vertex scalar's coefficient holds the raw values of vertices 0, 1
 * and 2 as its plane terms. */
static uint32_t
apple9_lower_input_vertex(struct apple9_dag_lower *lower, nir_scalar scalar)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(scalar.def);
   if (!lower->varyings || scalar.def->bit_size != 32 ||
       !nir_src_is_const(intr->src[0]) || !nir_src_is_const(intr->src[1]) ||
       nir_src_as_uint(intr->src[0]) > 2) {
      lower->reason =
         "Apple9 per-vertex inputs need constant vertices and 32-bit scalars";
      return AGX_APPLE9_VREG_INVALID;
   }
   unsigned location = nir_intrinsic_io_semantics(intr).location +
                       nir_src_as_uint(intr->src[1]);
   int index = apple9_varying_index(lower->varyings, location,
                                    nir_intrinsic_component(intr) + scalar.comp);
   int slot = index < 0 ? -1 : apple9_per_vertex_slot(lower, index);
   if (slot < 0) {
      lower->reason = "Apple9 per-vertex input is not written by the vertex stage";
      return AGX_APPLE9_VREG_INVALID;
   }
   unsigned coefficient = apple9_extra_coefficient_base(lower) +
      2 * util_bitcount(lower->barycentric_mask) + slot;
   uint32_t base = agx_apple9_vir_emit_coefficients(&lower->program, coefficient);
   if (base == AGX_APPLE9_VREG_INVALID)
      return base;
   return base + nir_src_as_uint(intr->src[0]);
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
      value = constant == 0 && lower->entry_zero != AGX_APPLE9_VREG_INVALID
                 ? apple9_dag_zero(lower) : apple9_dag_imm(lower, constant);
   } else if (nir_def_instr_type(scalar.def) == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(nir_def_instr(scalar.def));
      if (!apple9_texture_supported(tex)) {
         lower->reason = "Apple9 requires a non-shadow FP32 2D sample or integer texel fetch";
         return AGX_APPLE9_VREG_INVALID;
      }
      int coord_source = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      uint32_t coords[4];
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
      uint32_t layer_or_face = tex->coord_components >= 3 ? coords[2] : 0;
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && tex->is_array) {
         uint32_t face = coords[2], cube = coords[3];
         uint32_t sources[] = {cube, apple9_dag_imm(lower, 6), face};
         if (tex->backend_flags & AGX_APPLE9_TEXTURE_CUBE_AS_ARRAY) {
            /* Seamful cube arrays sample faces as 2D array layers,
             * layer = 6 * cube + face. */
            layer_or_face = apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
               AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources, 3, 0);
            coords[2] = layer_or_face;
         } else {
            /* Native cube arrays read the cube index from bits 0-11 and
             * the face from bits 12-14 of the packed LOD word. */
            sources[0] = face;
            sources[1] = apple9_dag_imm(lower, 1u << 12);
            sources[2] = cube;
            layer_or_face = apple9_dag_emit(lower, AGX_APPLE9_VIR_IMAD,
               AGX_APPLE9_ENC_INT_MAD_EXTENDED, sources, 3, 0);
         }
      }
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
      /* Table indices are emitted before the sample that reads them. */
      uint32_t handles[2];
      unsigned handle_count = 0;
      for (unsigned h = 0; h < 2; ++h) {
         int src = nir_tex_instr_src_index(
            tex, h ? nir_tex_src_sampler_handle : nir_tex_src_texture_handle);
         if (src >= 0)
            handles[handle_count++] = apple9_lower_dag_scalar(
               lower, nir_get_scalar(tex->src[src].src.ssa, 0));
      }
      uint32_t result;
      if (tex->op == nir_texop_txf_ms) {
         result = agx_apple9_vir_emit_texture_ms(
            &lower->program, coords, texture,
            util_bitcount(lower->sampler_mask), tex->is_array);
      } else if (tex->op == nir_texop_txd) {
         unsigned dimension = tex->is_array ? 2 :
            tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 0;
         unsigned axes = dimension == 3 ? 3 : 2;
         int min_lod = nir_tex_instr_src_index(tex, nir_tex_src_backend2);
         unsigned prefix = dimension == 3 ? 3 :
            dimension || tex->is_shadow || min_lod >= 0 ? 4 : 2;
         uint32_t sources[10] = {coords[0], coords[1]};
         if (dimension == 3)
            sources[2] = coords[2];
         if (prefix == 4) {
            sources[2] = dimension == 3 || tex->is_shadow ? coords[2]
                                                         : apple9_dag_zero(lower);
            sources[3] = tex->is_array ? layer_or_face : apple9_dag_zero(lower);
            if (min_lod >= 0) {
               uint32_t fields[] = {sources[3], apple9_lower_dag_scalar(
                  lower, nir_get_scalar(tex->src[min_lod].src.ssa, 0))};
               sources[3] = tex->is_array ? apple9_dag_emit(lower,
                  AGX_APPLE9_VIR_IOR, AGX_APPLE9_ENC_LOGIC_EXTENDED, fields, 2, 0)
                  : fields[1];
            }
         }
         for (unsigned derivative = 0; derivative < 2; ++derivative) {
            int source = nir_tex_instr_src_index(
               tex, derivative ? nir_tex_src_ddy : nir_tex_src_ddx);
            for (unsigned c = 0; c < axes; ++c)
               sources[prefix + 2 * c + derivative] = apple9_lower_dag_scalar(
                  lower, nir_get_scalar(tex->src[source].src.ssa, c));
         }
         result = agx_apple9_vir_emit_texture_grad(
            &lower->program, sources, texture,
            util_bitcount(lower->sampler_mask & BITFIELD_MASK(tex->sampler_index)),
            dimension, tex->is_shadow, min_lod >= 0);
      } else if (tex->op == nir_texop_txl || tex->op == nir_texop_txb ||
                 tex->op == nir_texop_txf ||
                 (tex->op == nir_texop_tg4 &&
                  (tex->coord_components >= 3 || tex->is_shadow))) {
         bool bias = tex->op == nir_texop_txb;
         int lod_source = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
         if (lod_source < 0 && tex->op != nir_texop_tg4) {
            lower->reason = "Apple9 texture LOD was not packed before selection";
            return AGX_APPLE9_VREG_INVALID;
         }
         /* Gather reads the base level; its LOD word carries only the layer. */
         uint32_t packed_lod = lod_source < 0 ? apple9_dag_zero(lower) :
            apple9_lower_dag_scalar(
               lower, nir_get_scalar(tex->src[lod_source].src.ssa, 0));
         bool fetch = tex->op == nir_texop_txf;
         int min_lod = nir_tex_instr_src_index(tex, nir_tex_src_backend2);
         uint32_t min_word = min_lod < 0 ? AGX_APPLE9_VREG_INVALID :
            apple9_lower_dag_scalar(lower,
               nir_get_scalar(tex->src[min_lod].src.ssa, 0));
         if (tex->coord_components >= 3 || tex->is_shadow) {
            if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE || tex->is_array) {
               uint32_t fields[] = {packed_lod, layer_or_face};
               packed_lod = apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR,
                  AGX_APPLE9_ENC_LOGIC_EXTENDED, fields, 2, 0);
            }
            /* A biased array sample takes its minimum LOD (Q6 << 16) in the
             * third parameter word, which otherwise repeats the layer. */
            if (min_lod >= 0)
               coords[2] = min_word;
            result = agx_apple9_vir_emit_texture_volume(&lower->program, coords,
               packed_lod, texture,
               util_bitcount(lower->sampler_mask & BITFIELD_MASK(tex->sampler_index)),
               (tex->backend_flags & AGX_APPLE9_TEXTURE_CUBE_AS_ARRAY) ? 2 :
               tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? (tex->is_array ? 4 : 1) :
               tex->is_array ? 2 :
                  tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 3 : 0,
               bias, tex->is_shadow);
         } else if (min_lod >= 0) {
            uint32_t fields[] = {packed_lod, min_word};
            result = agx_apple9_vir_emit_texture_min_lod(
               &lower->program, coords, apple9_dag_zero(lower),
               apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR,
                  AGX_APPLE9_ENC_LOGIC_EXTENDED, fields, 2, 0),
               texture,
               util_bitcount(lower->sampler_mask & BITFIELD_MASK(tex->sampler_index)));
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
      if (handle_count) {
         sample->texture_bindless =
            nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0;
         sample->sampler_bindless =
            nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0;
         if (!sample->texture_bindless) {
            lower->reason = "Apple9 register sampler index needs a register texture index";
            return AGX_APPLE9_VREG_INVALID;
         }
         for (unsigned h = 0; h < handle_count; ++h)
            sample->src[sample->nr_srcs++] = handles[h];
         agx_apple9_invalidate_uses(&lower->program);
      }
      sample->texture_lod_query = tex->backend_flags & AGX_APPLE9_TEXTURE_LOD_QUERY;
      sample->texture_has_offset =
         agx_apple9_texture_native_offset(tex, &sample->texture_offset);
      if (tex->op == nir_texop_tg4) {
         /* A gather always returns its four footprint texels. */
         mask = 0xf;
         sample->texture_gather = true;
         sample->texture_gather_component = tex->is_shadow ? 0 : tex->component;
      }
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
      if (intr && intr->intrinsic == nir_intrinsic_ballot) {
         if (scalar.def->bit_size != 32 || scalar.def->num_components != 1)
            return AGX_APPLE9_VREG_INVALID;
         uint32_t source = apple9_lower_dag_scalar(lower,
            nir_get_scalar(intr->src[0].ssa, 0));
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_SUBGROUP_BALLOT,
                                 AGX_APPLE9_ENC_SUBGROUP_BALLOT, &source, 1, 0);
      } else if (intr && intr->intrinsic == nir_intrinsic_load_scratch) {
         if (scalar.def->bit_size != 32 || nir_intrinsic_align_mul(intr) < 4 ||
             (nir_intrinsic_align_offset(intr) & 3))
            return AGX_APPLE9_VREG_INVALID;
         uint32_t address = apple9_lower_dag_scalar(lower,
            nir_get_scalar(intr->src[0].ssa, 0));
         address = apple9_dag_shift_imm(lower, nir_op_ushr, address, 2);
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_PRIVATE_LOAD,
            AGX_APPLE9_ENC_PRIVATE_LOAD, &address, 1, scalar.comp);
         if (value != AGX_APPLE9_VREG_INVALID) {
            lower->program.private_size = lower->nir->scratch_size;
            lower->program.instructions[lower->program.instruction_count - 1]
               ->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
         }
      } else if (intr && intr->intrinsic == nir_intrinsic_load_shared) {
         if (scalar.def->bit_size != 32 ||
             nir_intrinsic_align_mul(intr) < 4 ||
             (nir_intrinsic_align_offset(intr) & 3) ||
             lower->shared_argument == UINT_MAX)
            return AGX_APPLE9_VREG_INVALID;
         uint32_t address = apple9_lower_dag_scalar(lower,
            nir_get_scalar(intr->src[0].ssa, 0));
         unsigned offset = nir_intrinsic_base(intr) + 4 * scalar.comp;
         if (offset) {
            uint32_t add[] = {address, apple9_dag_imm(lower, offset)};
            address = apple9_dag_emit(lower, AGX_APPLE9_VIR_IADD,
               AGX_APPLE9_ENC_INT_ADD_EXTENDED, add, 2, 0);
         }
         address = apple9_dag_shift_imm(lower, nir_op_ushr, address, 2);
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_SHARED_LOAD,
            AGX_APPLE9_ENC_SHARED_LOAD, &address, 1,
            lower->argument_base + lower->shared_argument);
         if (value != AGX_APPLE9_VREG_INVALID)
            lower->program.instructions[lower->program.instruction_count - 1]
               ->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_6;
      } else if (intr && (intr->intrinsic == nir_intrinsic_inclusive_scan ||
                   intr->intrinsic == nir_intrinsic_exclusive_scan)) {
         if (scalar.def->bit_size != 32 ||
             nir_intrinsic_reduction_op(intr) != nir_op_iadd) {
            lower->reason = "Apple9 subgroup scans require 32-bit integer addition";
            return AGX_APPLE9_VREG_INVALID;
         }
         uint32_t source = apple9_lower_dag_scalar(lower,
            nir_get_scalar(intr->src[0].ssa, scalar.comp));
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_SUBGROUP_SCAN_IADD,
            AGX_APPLE9_ENC_SUBGROUP_SCAN_IADD, &source, 1, 0);
         if (intr->intrinsic == nir_intrinsic_inclusive_scan) {
            uint32_t add[] = {value, source};
            value = apple9_dag_emit(lower, AGX_APPLE9_VIR_IADD,
               AGX_APPLE9_ENC_INT_ADD_EXTENDED, add, 2, 0);
         }
      } else if (intr && intr->intrinsic == nir_intrinsic_pack_format_agx) {
         const bool rgba = nir_intrinsic_format(intr) == PIPE_FORMAT_R10G10B10A2_UNORM;
         if (nir_intrinsic_format(intr) != PIPE_FORMAT_R11G11B10_FLOAT && !rgba)
            return AGX_APPLE9_VREG_INVALID;
         uint32_t sources[4];
         for (unsigned c = 0; c < (rgba ? 4 : 3); ++c) {
            sources[c] = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[0].ssa, c));
            if (sources[c] == AGX_APPLE9_VREG_INVALID)
               return AGX_APPLE9_VREG_INVALID;
         }
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_PACK_FORMAT,
            rgba ? AGX_APPLE9_ENC_PACK_FORMAT : AGX_APPLE9_ENC_PACK_FORMAT_RGB,
            sources, rgba ? 4 : 3,
            rgba ? AGX_APPLE9_PACK_R10G10B10A2_UNORM : AGX_APPLE9_PACK_R11G11B10_FLOAT);
      } else if (intr && (intr->intrinsic == nir_intrinsic_read_invocation ||
                          intr->intrinsic == nir_intrinsic_shuffle)) {
         if (scalar.def->bit_size != 32)
            return AGX_APPLE9_VREG_INVALID;
         uint32_t sources[] = {
            apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[0].ssa, scalar.comp)),
            apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[1].ssa, 0)),
         };
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_SUBGROUP_BROADCAST,
            AGX_APPLE9_ENC_SUBGROUP_BROADCAST, sources, 2, 0);
      } else if (intr && (intr->intrinsic == nir_intrinsic_ddx ||
                   intr->intrinsic == nir_intrinsic_ddy ||
                   intr->intrinsic == nir_intrinsic_ddx_fine ||
                   intr->intrinsic == nir_intrinsic_ddy_fine ||
                   intr->intrinsic == nir_intrinsic_ddx_coarse ||
                   intr->intrinsic == nir_intrinsic_ddy_coarse)) {
         if (lower->nir->info.stage != MESA_SHADER_FRAGMENT)
            return AGX_APPLE9_VREG_INVALID;
         uint32_t source = apple9_lower_dag_scalar(lower,
            nir_get_scalar(intr->src[0].ssa, scalar.comp));
         if (source != AGX_APPLE9_VREG_INVALID)
            value = apple9_dag_emit(lower, AGX_APPLE9_VIR_DERIVATIVE,
               AGX_APPLE9_ENC_DERIVATIVE, &source, 1,
               intr->intrinsic == nir_intrinsic_ddy ||
               intr->intrinsic == nir_intrinsic_ddy_fine ||
               intr->intrinsic == nir_intrinsic_ddy_coarse);
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
                                 1 + (lower->varyings ? lower->varyings->count : 0) + lower->reads_primitive_id);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_point_coord) {
         if (scalar.comp >= 2)
            return AGX_APPLE9_VREG_INVALID;
         lower->reads_point_coord = true;
         value = apple9_dag_emit(lower, AGX_APPLE9_VIR_ITER,
                                 AGX_APPLE9_ENC_ITER, NULL, 0,
                                 1 + lower->varyings->count + lower->reads_primitive_id + lower->reads_z + scalar.comp);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic == nir_intrinsic_load_coefficients_agx) {
         value = apple9_lower_coefficients(lower, scalar);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 apple9_is_barycentric_coord(
                    nir_def_as_intrinsic(scalar.def)->intrinsic)) {
         value = apple9_lower_barycentric_weight(lower, scalar);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 nir_def_as_intrinsic(scalar.def)->intrinsic ==
                    nir_intrinsic_load_input_vertex) {
         value = apple9_lower_input_vertex(lower, scalar);
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
                 (nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_barycentric_centroid ||
                  nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_barycentric_at_sample)) {
         nir_intrinsic_instr *bary = nir_def_as_intrinsic(scalar.def);
         value = apple9_interpolation_position(
            lower, bary,
            bary->intrinsic == nir_intrinsic_load_barycentric_at_sample);
      } else if (nir_def_instr_type(scalar.def) == nir_instr_type_intrinsic &&
                 (nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_sample_mask_in ||
                  /* Raster coverage. Per-sample shading uses this to mask
                   * helper invocations for uncovered samples. */
                  nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_sample_mask)) {
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
         /* A uniform move reads the word without a zero register
          * (AGX_APPLE9_NO_UNIFORM_MOVE=1 keeps the zero-sourced OR). */
         static int no_move = -1;
         if (no_move < 0)
            no_move = debug_get_option_cached_agx_apple9_no_uniform_move() != NULL;
         if (no_move) {
            uint32_t zero = apple9_dag_zero(lower);
            value = apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR_UNIFORM,
               AGX_APPLE9_ENC_LOGIC_UNIFORM, &zero, 1,
               lower->preamble_base + word);
         } else {
            value = apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR_UNIFORM,
               AGX_APPLE9_ENC_UNIFORM_MOVE, NULL, 0,
               lower->preamble_base + word);
         }
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
                     nir_intrinsic_load_ubo ||
                  nir_def_as_intrinsic(scalar.def)->intrinsic ==
                     nir_intrinsic_load_global_2x32)) {
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

         unsigned binding = lower->argument_base + load->argument;
         const bool uniform_base = load->address &&
            apple9_uniform_address(lower, load->address, &binding);
         uint32_t address = AGX_APPLE9_VREG_INVALID;
         if (load->address && !uniform_base)
            address = apple9_lower_global_address(lower, load->address);
         else if (!load->address && lower->buffer_table)
            address = apple9_buffer_address(lower, load->argument);
         if (!uniform_base && (load->address || lower->buffer_table) &&
             address == AGX_APPLE9_VREG_INVALID)
            return address;

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
               &lower->program, binding, index, components, &contract);
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
            memory->memory_global = uniform_base;
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
            source, 1, binding);
         if (value != AGX_APPLE9_VREG_INVALID) {
            struct agx_apple9_vir_instr *memory =
               lower->program.instructions[lower->program.instruction_count - 1];
            memory->memory_global = uniform_base;
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

         if (scalar.def->bit_size == 16 &&
             (op == nir_op_fadd || op == nir_op_fsub || op == nir_op_fmul ||
              op == nir_op_ffma || op == nir_op_ffma_weak ||
              op == nir_op_fmin || op == nir_op_fmax)) {
            bool fma = op == nir_op_ffma || op == nir_op_ffma_weak;
            nir_scalar input[3];
            for (unsigned s = 0; s < (fma ? 3 : 2); ++s)
               input[s] = apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, s));
            if (op == nir_op_fmul) {
               /* NIR expresses division as a multiply by reciprocal. Keep
                * the reciprocal in FP32 for this use: binary16 reciprocals
                * overflow for small divisors even when the quotient fits.
                * Both inputs still enter their native FP16 hardware ports. */
               for (unsigned s = 0; s < 2; ++s) {
                  if (nir_scalar_is_alu(input[s]) &&
                      nir_scalar_alu_op(input[s]) == nir_op_frcp) {
                     uint32_t divisor = apple9_lower_dag_scalar(lower,
                        nir_scalar_chase_alu_src(input[s], 0));
                     uint32_t numerator = apple9_lower_dag_scalar(lower,
                        input[1 - s]);
                     if (divisor == AGX_APPLE9_VREG_INVALID ||
                         numerator == AGX_APPLE9_VREG_INVALID)
                        return AGX_APPLE9_VREG_INVALID;
                     uint32_t sources[] = {
                        apple9_dag_emit(lower, AGX_APPLE9_VIR_HRCP_F32,
                           AGX_APPLE9_ENC_HALF_SPECIAL, &divisor, 1, 3),
                        numerator};
                     value = apple9_dag_emit(lower, AGX_APPLE9_VIR_HMUL_MIXED,
                        AGX_APPLE9_ENC_HALF2, sources, 2, 0);
                     lower->ssa_to_vreg[key] = value;
                     return value;
                  }
               }
            }
            if (op == nir_op_fadd) {
               /* The compact half add can negate its second physical source.
                * Give subtraction its ordinary a-b semantics in VIR. */
               for (unsigned s = 0; s < 2; ++s) {
                  if (nir_scalar_is_alu(input[s]) &&
                      nir_scalar_alu_op(input[s]) == nir_op_fneg) {
                     nir_scalar positive = input[1 - s];
                     nir_scalar negative = nir_scalar_chase_alu_src(input[s], 0);
                     input[0] = positive;
                     input[1] = negative;
                     op = nir_op_fsub;
                     break;
                  }
               }
            }
            uint32_t sources[3];
            for (unsigned s = 0; s < (fma ? 3 : 2); ++s) {
               sources[s] = apple9_lower_dag_scalar(lower, input[s]);
               if (sources[s] == AGX_APPLE9_VREG_INVALID)
                  return sources[s];
            }
            enum agx_apple9_vir_opcode vir_op =
               fma ? AGX_APPLE9_VIR_HFMA
               : op == nir_op_fadd ? AGX_APPLE9_VIR_HADD
               : op == nir_op_fsub ? AGX_APPLE9_VIR_HSUB
               : op == nir_op_fmin ? AGX_APPLE9_VIR_HMIN
               : op == nir_op_fmax ? AGX_APPLE9_VIR_HMAX
                                    : AGX_APPLE9_VIR_HMUL;
            value = apple9_dag_emit(lower, vir_op,
               fma ? AGX_APPLE9_ENC_HALF3
               : (op == nir_op_fmin || op == nir_op_fmax)
                  ? AGX_APPLE9_ENC_HALF_MINMAX : AGX_APPLE9_ENC_HALF2,
               sources, fma ? 3 : 2, 0);
         } else if (op == nir_op_i2i8 || op == nir_op_i2i16 || op == nir_op_i2i32 ||
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
         } else if (op == nir_op_f2f16 || op == nir_op_f2f16_rtne ||
                    op == nir_op_f2f32 ||
                    op == nir_op_unpack_half_2x16) {
            nir_scalar source_scalar = nir_scalar_chase_alu_src(scalar, 0);
            unsigned half = 0;
            if (op == nir_op_unpack_half_2x16) {
               nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(scalar.def));
               source_scalar = nir_get_scalar(alu->src[0].src.ssa,
                                               alu->src[0].swizzle[0]);
               half = scalar.comp;
            }
            bool narrow = op == nir_op_f2f16 || op == nir_op_f2f16_rtne;
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
            while (nir_def_instr_type(predicate.def) == nir_instr_type_alu &&
                   nir_scalar_alu_op(predicate) == nir_op_inot) {
               uint32_t temporary = if_true;
               if_true = if_false;
               if_false = temporary;
               predicate = apple9_chase_trivial(
                  nir_scalar_chase_alu_src(predicate, 0));
            }
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
         } else if (op == nir_op_b2i8 || op == nir_op_b2i16 || op == nir_op_b2i32) {
            /* Canonical zero/one fits every integer destination width. */
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
               bool half = scalar.def->bit_size == 16;
               uint32_t factor = apple9_dag_emit(
                  lower, half ? AGX_APPLE9_VIR_HSQRT_FACTOR
                              : AGX_APPLE9_VIR_FSQRT_FACTOR,
                  half ? AGX_APPLE9_ENC_HALF_SPECIAL
                       : AGX_APPLE9_ENC_FLOAT_SPECIAL, &source, 1, 3);
               uint32_t sources[] = {factor, source};
               value = apple9_dag_emit(lower,
                  half ? AGX_APPLE9_VIR_HMUL_MIXED : AGX_APPLE9_VIR_FMUL,
                  half ? AGX_APPLE9_ENC_HALF2 : AGX_APPLE9_ENC_FLOAT2_COMPACT,
                  sources, 2, 0);
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
            bool half = scalar.def->bit_size == 16;
            if (half) {
               switch (op) {
               case nir_op_frcp: special = AGX_APPLE9_VIR_HRCP; break;
               case nir_op_frsq: special = AGX_APPLE9_VIR_HRSQ; break;
               case nir_op_fexp2: special = AGX_APPLE9_VIR_HEXP2; break;
               case nir_op_flog2: special = AGX_APPLE9_VIR_HLOG2; break;
               case nir_op_ffloor: special = AGX_APPLE9_VIR_HFLOOR; break;
               case nir_op_fceil: special = AGX_APPLE9_VIR_HCEIL; break;
               case nir_op_ftrunc: special = AGX_APPLE9_VIR_HTRUNC; break;
               case nir_op_fround_even: special = AGX_APPLE9_VIR_HROUND_EVEN; break;
               default: return AGX_APPLE9_VREG_INVALID;
               }
            }
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source != AGX_APPLE9_VREG_INVALID)
               value = apple9_dag_emit(lower, special,
                                       half ? AGX_APPLE9_ENC_HALF_SPECIAL
                                            : AGX_APPLE9_ENC_FLOAT_SPECIAL,
                                       &source, 1, half ? 2 : 3);
         } else if (op == nir_op_bit_count || op == nir_op_ufind_msb ||
                    op == nir_op_bitfield_reverse) {
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source != AGX_APPLE9_VREG_INVALID) {
               enum agx_apple9_vir_opcode vir_op =
                  op == nir_op_bit_count ? AGX_APPLE9_VIR_BIT_COUNT
                  : op == nir_op_ufind_msb ? AGX_APPLE9_VIR_UFIND_MSB
                                           : AGX_APPLE9_VIR_BIT_REVERSE;
               value = apple9_dag_emit(lower, vir_op,
                                       AGX_APPLE9_ENC_BIT_UNARY, &source, 1, 0);
            }
         } else if (op == nir_op_u2f32 || op == nir_op_i2f32 ||
                    op == nir_op_u2f16 || op == nir_op_i2f16 ||
                    op == nir_op_f2i32 || op == nir_op_f2u32) {
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source != AGX_APPLE9_VREG_INVALID) {
               unsigned bits = nir_scalar_chase_alu_src(scalar, 0).def->bit_size;
               /* Narrow NIR values may retain arbitrary upper register bits.
                * Extend at selection: algebraic passes can fold away an
                * earlier i2i32/u2u32 before the float conversion. */
               if (bits < 32 && (op == nir_op_u2f16 || op == nir_op_i2f16 ||
                                 op == nir_op_u2f32 || op == nir_op_i2f32)) {
                  if (op == nir_op_u2f16 || op == nir_op_u2f32) {
                     uint32_t sources[] = {
                        source, apple9_dag_imm(lower, BITFIELD_MASK(bits))};
                     source = apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
                        AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
                  } else {
                     source = apple9_dag_shift_imm(lower, nir_op_ishl,
                                                   source, 32 - bits);
                     source = apple9_dag_shift_imm(lower, nir_op_ishr,
                                                   source, 32 - bits);
                  }
                  if (source == AGX_APPLE9_VREG_INVALID)
                     return source;
               }
               /* The integer conversions read an FP32 source. Widen binary16
                * values held in the low register half first. */
               if (bits == 16 && (op == nir_op_f2i32 || op == nir_op_f2u32)) {
                  source = apple9_dag_emit(lower, AGX_APPLE9_VIR_UNPACK_HALF,
                     AGX_APPLE9_ENC_HALF_TO_FLOAT, &source, 1, 0);
                  if (source == AGX_APPLE9_VREG_INVALID)
                     return source;
               }
               enum agx_apple9_vir_opcode vir_op =
                  (op == nir_op_u2f32 || op == nir_op_u2f16) ? AGX_APPLE9_VIR_U2F32
                  : (op == nir_op_i2f32 || op == nir_op_i2f16) ? AGX_APPLE9_VIR_I2F32
                  : op == nir_op_f2i32 ? AGX_APPLE9_VIR_F2I32
                                       : AGX_APPLE9_VIR_F2U32;
               enum agx_apple9_encoding encoding =
                  (op == nir_op_u2f32 || op == nir_op_u2f16) ? AGX_APPLE9_ENC_UINT_TO_FLOAT
                  : (op == nir_op_i2f32 || op == nir_op_i2f16) ? AGX_APPLE9_ENC_SINT_TO_FLOAT
                  : op == nir_op_f2i32 ? AGX_APPLE9_ENC_FLOAT_TO_SINT
                                       : AGX_APPLE9_ENC_FLOAT_TO_UINT;
               value = apple9_dag_emit(lower, vir_op, encoding, &source, 1, 0);
               if (value != AGX_APPLE9_VREG_INVALID &&
                   (op == nir_op_u2f16 || op == nir_op_i2f16)) {
                  source = value;
                  value = apple9_dag_emit(lower, AGX_APPLE9_VIR_F2F16,
                     AGX_APPLE9_ENC_FLOAT_TO_HALF_ZEXT, &source, 1, 0);
               }
            }
         } else if (op == nir_op_inot || op == nir_op_fneg ||
                    op == nir_op_fabs || op == nir_op_ineg) {
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source == AGX_APPLE9_VREG_INVALID)
               return source;
            bool half = scalar.def->bit_size == 16;
            uint32_t immediate = op == nir_op_fneg   ? (half ? 0x8000u : 0x80000000u)
                                 : op == nir_op_fabs ? (half ? 0x7fffu : 0x7fffffffu)
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
         } else if (op == nir_op_ishl || op == nir_op_ishr || op == nir_op_ushr) {
            nir_scalar shift =
               apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 1));
            unsigned bits = scalar.def->bit_size;
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source == AGX_APPLE9_VREG_INVALID)
               return source;
            /* Narrow SSA values occupy 32-bit registers whose high bits are
             * undefined. Right shifts observe those bits unless the input is
             * extended first. Left shifts only need the width-correct count:
             * their low bits are independent of the undefined high bits. */
            if (bits < 32 && op == nir_op_ushr) {
               uint32_t sources[] = {source, apple9_dag_imm(lower, BITFIELD_MASK(bits))};
               source = apple9_dag_emit(lower, AGX_APPLE9_VIR_IAND,
                  AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
            } else if (bits < 32 && op == nir_op_ishr) {
               source = apple9_dag_shift_imm(lower, nir_op_ishl, source, 32 - bits);
               source = apple9_dag_shift_imm(lower, nir_op_ishr, source, 32 - bits);
            }
            uint32_t amount;
            value = apple9_const_u32(shift, &amount)
                       ? apple9_dag_shift_imm(lower, op, source, amount & (bits - 1))
                       : apple9_dag_shift_variable(lower, op, source, shift, bits);
         } else if (op == nir_op_fsat) {
            uint32_t source = apple9_lower_dag_source(lower, scalar, 0);
            if (source != AGX_APPLE9_VREG_INVALID) {
               if (scalar.def->bit_size == 16) {
                  uint32_t sources[] = {source, apple9_dag_zero(lower)};
                  sources[0] = apple9_dag_emit(lower, AGX_APPLE9_VIR_HMAX,
                     AGX_APPLE9_ENC_HALF_MINMAX, sources, 2, 0);
                  sources[1] = apple9_dag_imm(lower, 0x3c00);
                  value = apple9_dag_emit(lower, AGX_APPLE9_VIR_HMIN,
                     AGX_APPLE9_ENC_HALF_MINMAX, sources, 2, 0);
               } else {
                  value = apple9_dag_emit(lower, AGX_APPLE9_VIR_FSAT,
                     AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED, &source, 1, 0);
               }
            }
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
            /* x & (2^w - 1) and (x >> k) & (2^w - 1) are one bitfield
             * extract: the logical shift-right immediate form with a width. */
            for (unsigned i = 0; op == nir_op_iand &&
                                 scalar.def->bit_size == 32 && i < 2; ++i) {
               uint32_t mask, amount;
               if (!apple9_const_u32(nir_scalar_chase_alu_src(scalar, i), &mask) ||
                   !mask || mask == UINT32_MAX || (mask & (mask + 1)))
                  continue;
               unsigned width = util_bitcount(mask), offset = 0;
               nir_scalar field =
                  apple9_chase_trivial(nir_scalar_chase_alu_src(scalar, 1 - i));
               uint32_t source;
               if (nir_def_instr_type(field.def) == nir_instr_type_alu &&
                   nir_scalar_alu_op(field) == nir_op_ushr &&
                   field.def->bit_size == 32 &&
                   apple9_const_u32(nir_scalar_chase_alu_src(field, 1), &amount)) {
                  offset = amount & 31;
                  source = apple9_lower_dag_source(lower, field, 0);
               } else {
                  source = apple9_lower_dag_source(lower, scalar, 1 - i);
               }
               value = apple9_dag_extract(lower, source, offset,
                                          offset + width >= 32 ? 0 : width);
               lower->ssa_to_vreg[key] = value;
               return value;
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
   bool writes_global;
   bool descriptor_tables;
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

static bool
apple9_is_global_atomic(nir_intrinsic_op op)
{
   return op == nir_intrinsic_global_atomic_2x32 ||
          op == nir_intrinsic_global_atomic_swap_2x32;
}

static bool
apple9_is_buffer_atomic(nir_intrinsic_op op)
{
   return apple9_is_ssbo_atomic(op) || apple9_is_global_atomic(op);
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
         map->writes_global |= intr->intrinsic == nir_intrinsic_store_global_2x32 ||
                               apple9_is_global_atomic(intr->intrinsic) ||
                               intr->intrinsic == nir_intrinsic_image_store_apple9_agx;
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

   bool has_store = map->writes_global;
   for (unsigned i = 0; i < map->count; ++i)
      has_store |= map->resource[i].write;
   if (!has_store && nir->info.stage == MESA_SHADER_COMPUTE) {
      *reason = agx_apple9_compute_without_effects;
      return false;
   }

   qsort(map->resource, map->count, sizeof(map->resource[0]),
         apple9_compare_buffer_resource);
   if (nir->info.shared_size) {
      if (nir->info.stage != MESA_SHADER_COMPUTE ||
          nir->info.shared_size > AGX_APPLE9_MAX_SHARED_BYTES ||
          map->count == AGX_APPLE9_COMPUTE_MAX_RESOURCES) {
         *reason = "Apple9 shared memory exceeds the supported compute allocation";
         return false;
      }
      /* Shared storage is local to a workgroup. Its root participates in the
       * argument ABI but carries no external-buffer hazard or API binding. */
      map->resource[map->count++] = (struct apple9_buffer_resource){
         .kind = AGX_APPLE9_COMPUTE_RESOURCE_SHARED,
      };
   }
   map->descriptor_tables = nir->info.stage != MESA_SHADER_COMPUTE ||
      map->count > AGX_APPLE9_COMPUTE_DIRECT_MAX_RESOURCES;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block)
         map->descriptor_tables |= instr->type == nir_instr_type_tex ||
            (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_image_store_apple9_agx);
   }
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
            if (debug_get_option_cached_agx_apple9_trace_nir() != NULL) {
               fprintf(stderr, "Apple9 unsupported instruction: ");
               nir_print_instr(instr, stderr);
               fprintf(stderr, "\n");
            }
            *reason = instr->type == nir_instr_type_tex
                         ? "Apple9 texture operation is unsupported"
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
             intr->intrinsic != nir_intrinsic_load_global_2x32 &&
             intr->intrinsic != nir_intrinsic_store_global_2x32 &&
             !apple9_is_buffer_atomic(intr->intrinsic))
            continue;
         const bool atomic = apple9_is_buffer_atomic(intr->intrinsic);

         const bool global = intr->intrinsic == nir_intrinsic_load_global_2x32 ||
                             intr->intrinsic == nir_intrinsic_store_global_2x32 ||
                             apple9_is_global_atomic(intr->intrinsic);
         const bool load = intr->intrinsic != nir_intrinsic_store_ssbo &&
                           intr->intrinsic != nir_intrinsic_store_global_2x32 && !atomic;
         nir_def *address = global ? intr->src[load || atomic ? 0 : 1].ssa : NULL;
         if (address && (address->bit_size != 32 || address->num_components != 2)) {
            *reason = "Apple9 global access requires a 64-bit address pair";
            return false;
         }
         nir_builder builder = nir_builder_at(nir_before_instr(&intr->instr));
         nir_def *offset = global ? nir_imm_int(&builder, 0) :
                           load || atomic ? intr->src[1].ssa : intr->src[2].ssa;
         nir_scalar index;
         unsigned index_scale = 0, index_add = 0;
         uint8_t index_shift = 0;
         int16_t byte_offset = 0;
         const unsigned bit_size = atomic ? intr->def.bit_size
                                   : load  ? intr->def.bit_size
                                           : intr->src[0].ssa->bit_size;
         const unsigned components = atomic ? 1 : load ? intr->def.num_components
                                                       : intr->src[0].ssa->num_components;
         /* Loads and stores fold the offset exactly. Atomics shift it to an
          * element index, which is exact only for an aligned base, so they
          * keep the complete address. */
         nir_def *split_base = NULL, *split_offset = NULL;
         if (global && !atomic &&
             apple9_split_global_address(&builder, address, &split_base,
                                         &split_offset) &&
             apple9_memory_address(nir, intr, split_offset, bit_size / 8,
                                   components, &index, &index_shift,
                                   &byte_offset)) {
            address = split_base;
         } else if (!(atomic ? apple9_element_index(intr, offset, bit_size / 8, components,
                                                     &index, &index_scale, &index_add)
                             : apple9_memory_address(nir, intr, offset, bit_size / 8, components,
                                                     &index, &index_shift, &byte_offset))) {
            *reason =
               "Apple9 buffer compiler requires a supported 32-bit byte offset";
            return false;
         }

         nir_def *binding_def = global ? NULL : load || atomic ? intr->src[0].ssa
                                                                 : intr->src[1].ssa;
         uint32_t binding = 0;
         if (!global && !apple9_const_u32(nir_get_scalar(binding_def, 0), &binding)) {
            *reason = "Apple9 buffer compiler requires constant bindings";
            return false;
         }

         if (atomic) {
            unsigned data_src = global ? 1 : 2;
            bool swap = intr->intrinsic == nir_intrinsic_ssbo_atomic_swap ||
                        intr->intrinsic == nir_intrinsic_global_atomic_swap_2x32;
            if (bit_size != 32 || intr->def.num_components != 1 ||
                intr->src[data_src].ssa->bit_size != 32 ||
                intr->src[data_src].ssa->num_components != 1 ||
                (swap &&
                 (nir_intrinsic_atomic_op(intr) != nir_atomic_op_cmpxchg ||
                  intr->src[data_src + 1].ssa->bit_size != 32 ||
                  intr->src[data_src + 1].ssa->num_components != 1))) {
               *reason = "Apple9 supports scalar 32-bit SSBO atomics";
               return false;
            }

            enum agx_apple9_atomic_op op;
            if (!apple9_atomic_op(nir_intrinsic_atomic_op(intr), &op)) {
               *reason = "Apple9 encountered an unsupported SSBO atomic operation";
               return false;
            }

            unsigned argument = global ? 0 : apple9_buffer_argument(
               map, AGX_APPLE9_COMPUTE_RESOURCE_SSBO, binding);
            if (argument == UINT_MAX) {
               *reason = "Apple9 atomic binding is absent from its resource map";
               return false;
            }

            struct apple9_buffer_atomic entry = {
               .intr = intr,
               .block = block,
               .index = index,
               .address = address,
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
            unsigned argument = global ? 0 : apple9_buffer_argument(map, kind, binding);
            if (argument == UINT_MAX) {
               *reason = "Apple9 input binding is absent from its resource map";
               return false;
            }
            const nir_component_mask_t read_mask =
               (nir_intrinsic_access(intr) & ACCESS_VOLATILE)
                  ? BITFIELD_MASK(intr->def.num_components)
                  : nir_def_components_read(&intr->def);
            for (unsigned component = 0; component < intr->def.num_components;
                 ++component) {
               if (!(read_mask & BITFIELD_BIT(component)))
                  continue;

               struct apple9_scalar_load load = {
                  .intr = intr,
                  .block = block,
                  .index = index,
                  .address = address,
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
            unsigned argument = global ? 0 : apple9_buffer_argument(
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
               .address = address,
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

   if (!uniform_stores && !map->writes_global && stores->size == 0 && atomics->size == 0 &&
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

   unsigned binding = lower->argument_base + store->argument;
   const bool uniform_base = store->address &&
      apple9_uniform_address(lower, store->address, &binding);
   uint32_t address = AGX_APPLE9_VREG_INVALID;
   if (store->address && !uniform_base)
      address = apple9_lower_global_address(lower, store->address);
   else if (!store->address && lower->buffer_table)
      address = apple9_buffer_address(lower, store->argument);
   if (!uniform_base && (store->address || lower->buffer_table) &&
       address == AGX_APPLE9_VREG_INVALID)
      return false;
   if (!agx_apple9_vir_emit_device_store(&lower->program, binding,
                                         store->lowered_index, store->output,
                                         store->components, store->bit_size))
      return false;
   struct agx_apple9_vir_instr *memory =
      lower->program.instructions[lower->program.instruction_count - 1];
   memory->memory_global = uniform_base;
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
   unsigned data_src = atomic->address ? 1 : 2;
   uint32_t data[2];
   unsigned data_components = 1;
   if (atomic->op == AGX_APPLE9_ATOMIC_CMPXCHG) {
      /* Native tuple order is desired, compare. NIR spells the sources as
       * compare, desired. */
      data[0] = apple9_lower_dag_scalar(
         lower, apple9_chase_trivial(nir_get_scalar(atomic->intr->src[data_src + 1].ssa,
                                                     0)));
      data[1] = apple9_lower_dag_scalar(
         lower, apple9_chase_trivial(nir_get_scalar(atomic->intr->src[data_src].ssa,
                                                     0)));
      data_components = 2;
      for (unsigned c = 0; c < data_components; ++c) {
         if (data[c] == AGX_APPLE9_VREG_INVALID)
            return false;
      }
   } else {
      nir_scalar scalar = apple9_chase_trivial(
         nir_get_scalar(atomic->intr->src[data_src].ssa, 0));
      data[0] = apple9_lower_dag_scalar(lower, scalar);
      if (data[0] == AGX_APPLE9_VREG_INVALID)
         return false;


   }

   uint32_t result = AGX_APPLE9_VREG_INVALID;
   unsigned binding = lower->argument_base + atomic->argument;
   const bool uniform_base = atomic->address &&
      apple9_uniform_address(lower, atomic->address, &binding);
   uint32_t address = atomic->address && !uniform_base
                         ? apple9_lower_global_address(lower, atomic->address)
                         : AGX_APPLE9_VREG_INVALID;
   if (!atomic->address && lower->buffer_table)
      address = apple9_buffer_address(lower, atomic->argument);
   if (!uniform_base && (atomic->address || lower->buffer_table) &&
       address == AGX_APPLE9_VREG_INVALID)
      return false;
   if (!agx_apple9_vir_emit_device_atomic(
          &lower->program, binding, index,
          data, data_components, atomic->op, discard, &result)) {
      lower->reason = "could not emit an Apple9 VIR device atomic";
      return false;
   }
   for (unsigned i = lower->program.instruction_count; uniform_base && i-- > 0;) {
      struct agx_apple9_vir_instr *ins = lower->program.instructions[i];
      if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC) {
         ins->memory_global = true;
         break;
      }
   }

   if (address != AGX_APPLE9_VREG_INVALID &&
       !agx_apple9_vir_set_device_atomic_address(&lower->program, address))
      return false;

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
   if (jump->type == nir_jump_halt)
      return agx_apple9_vir_emit_side_effect(&lower->program,
         AGX_APPLE9_VIR_HALT, AGX_APPLE9_ENC_HALT, NULL, 0, 0);
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
   if (scope_tag > UINT8_MAX) {
      lower->reason = "Apple9 loop nesting exceeds the encoded break fields";
      return false;
   }

   /* Unwind consumes a predicate temporary, not a loop nesting level. The
    * predicate dies at this instruction, so every break can reuse bank one.
    * Publish true after the exit copies, since they may compare values. The
    * separate scope tag unwinds the intervening execution-mask scopes. */
   const unsigned break_bank = 1;

   uint32_t zero = apple9_dag_zero(lower);
   uint32_t sources[] = {zero, zero};
   if (zero == AGX_APPLE9_VREG_INVALID ||
       !agx_apple9_vir_emit_side_effect(
          &lower->program, AGX_APPLE9_VIR_PREDICATE_COMPARE,
          AGX_APPLE9_ENC_PREDICATE_COMPARE_LOOP, sources, 2,
          AGX_APPLE9_PREDICATE_EXT_IEQ |
             AGX_APPLE9_PREDICATE_BANK(break_bank))) {
      lower->reason = "could not emit an Apple9 unconditional break predicate";
      return false;
   }

   const bool ok = agx_apple9_vir_emit_side_effect(
      &lower->program, AGX_APPLE9_VIR_BREAK_MASK_UNWIND,
      AGX_APPLE9_ENC_BREAK_MASK_UNWIND, NULL, 0,
      AGX_APPLE9_BREAK_IMMEDIATE(scope_tag, break_bank));
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

/* Releasing the depth/stencil domain ends testing for the fragment, so tests
 * repeated in a loop (per-sample shading) release once after the loop. */
static bool
apple9_emit_zs_release(struct apple9_dag_lower *lower)
{
   if (lower->loop_depth) {
      lower->zs_release_pending = true;
      return true;
   }
   return agx_apple9_vir_emit_side_effect(&lower->program,
      AGX_APPLE9_VIR_TILE_FENCE, AGX_APPLE9_ENC_TILE_FENCE, NULL, 0, 1);
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
      (!tests || apple9_emit_zs_release(lower));
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
   bool uvs = intr->intrinsic == nir_intrinsic_store_uvs_agx;
   const unsigned location = uvs ? VARYING_SLOT_MAX
      : nir_intrinsic_io_semantics(intr).location + nir_src_as_uint(intr->src[1]);
   nir_alu_type type = uvs ? nir_type_float32 : nir_intrinsic_src_type(intr);
   bool integer = type == nir_type_int32 || type == nir_type_uint32;
   if (type != nir_type_float32 && !integer)
      return false;
   /* Position and point size are floats even when IO lowering erased the
    * type, as in the transform feedback passthrough geometry shader.
    */
   if (location == VARYING_SLOT_POS || location == VARYING_SLOT_PSIZ)
      integer = false;
   for (unsigned c = 0; c < intr->num_components; ++c) {
      if (!uvs && !(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)))
         continue;
      unsigned component = (uvs ? 0 : nir_intrinsic_component(intr)) + c;
      if (component >= 4)
         return false;
      unsigned slot = component;
      if (uvs) {
         slot = nir_src_as_uint(intr->src[1]) + c;
      } else if (location == VARYING_SLOT_PSIZ) {
         if (component != 0)
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
      if (value == AGX_APPLE9_VREG_INVALID)
         return false;
      /* Export publications stay live until completion, so a value stored to
       * several UVS slots (a clip distance's varying and rasterizer copies)
       * publishes once. */
      if (lower->share_exports &&
          lower->export_memo_block != lower->active_load_block) {
         lower->export_memo_block = lower->active_load_block;
         lower->export_memo_count = 0;
      }
      uint32_t exported = AGX_APPLE9_VREG_INVALID;
      for (unsigned m = 0; lower->share_exports && m < lower->export_memo_count; ++m) {
         if (lower->export_memo[m].value == value && lower->export_memo[m].raw == raw)
            exported = lower->export_memo[m].export;
      }
      if (exported == AGX_APPLE9_VREG_INVALID) {
         uint32_t scale = apple9_dag_imm(lower, raw ? 0 : fui(1.0f));
         uint32_t sources[] = {value, scale};
         if (scale == AGX_APPLE9_VREG_INVALID)
            return false;
         exported = apple9_dag_emit(
            lower, raw ? AGX_APPLE9_VIR_IXOR : AGX_APPLE9_VIR_FMUL,
            raw ? AGX_APPLE9_ENC_LOGIC_EXPORT : AGX_APPLE9_ENC_FLOAT2_EXPORT,
            sources, 2, 0);
         if (lower->share_exports && exported != AGX_APPLE9_VREG_INVALID &&
             lower->export_memo_count < APPLE9_EXPORT_MEMO) {
            lower->export_memo[lower->export_memo_count++] =
               (struct apple9_export_memo){value, exported, raw};
         }
      }
      value = exported;
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
         /* Preserve distinct volatile accesses through native optimization
          * and scheduling. The device fence orders each access against
          * earlier memory operations.
          */
         if (nir_intrinsic_has_access(intr) &&
             (nir_intrinsic_access(intr) & ACCESS_VOLATILE) &&
             !agx_apple9_vir_emit_side_effect(&lower->program,
                AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE,
                NULL, 0, 0))
            return false;
         if (intr->intrinsic == nir_intrinsic_load_barycentric_pixel)
            continue; /* Interpolation mode is consumed by the input operation. */
         /* The pixel is acquired in raster order at entry and released at
          * the end, so the critical section spans the whole shader. Its end
          * only orders the section's memory accesses before the release. */
         if (intr->intrinsic == nir_intrinsic_begin_invocation_interlock)
            continue;
         if (intr->intrinsic == nir_intrinsic_end_invocation_interlock) {
            if (!agx_apple9_vir_emit_side_effect(&lower->program,
                   AGX_APPLE9_VIR_DEVICE_FENCE, AGX_APPLE9_ENC_DEVICE_FENCE,
                   NULL, 0, 1))
               return false;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_barrier) {
            unsigned scope = nir_intrinsic_execution_scope(intr);
            unsigned modes = nir_intrinsic_memory_modes(intr);
            if (scope > SCOPE_WORKGROUP ||
                (modes & ~(nir_var_mem_shared | nir_var_mem_global |
                           nir_var_mem_ssbo | nir_var_image))) {
               lower->reason = "Apple9 barrier scope or memory class is unsupported";
               return false;
            }
            /* As on Apple8, a subgroup executes in lockstep and shared memory
             * needs no fence. Storage images are lowered to global memory.
             */
            if (nir_intrinsic_memory_scope(intr) != SCOPE_NONE &&
                (modes & (nir_var_mem_global | nir_var_mem_ssbo |
                          nir_var_image)) &&
                !agx_apple9_vir_emit_side_effect(&lower->program,
                   AGX_APPLE9_VIR_DEVICE_FENCE,
                   AGX_APPLE9_ENC_DEVICE_FENCE, NULL, 0, 0))
               return false;
            if (scope == SCOPE_WORKGROUP &&
                !agx_apple9_vir_emit_side_effect(&lower->program,
                   AGX_APPLE9_VIR_WORKGROUP_BARRIER,
                   AGX_APPLE9_ENC_WORKGROUP_BARRIER, NULL, 0, 0))
               return false;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_scratch) {
            if (intr->src[0].ssa->bit_size != 32 ||
                nir_intrinsic_align_mul(intr) < 4 ||
                (nir_intrinsic_align_offset(intr) & 3))
               return false;
            uint32_t address = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[1].ssa, 0));
            address = apple9_dag_shift_imm(lower, nir_op_ushr, address, 2);
            for (unsigned c = 0; c < intr->num_components; ++c) {
               if (!(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)))
                  continue;
               uint32_t sources[] = {
                  apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[0].ssa, c)),
                  address,
               };
               if (!agx_apple9_vir_emit_side_effect(&lower->program,
                      AGX_APPLE9_VIR_PRIVATE_STORE, AGX_APPLE9_ENC_PRIVATE_STORE,
                      sources, 2, c))
                  return false;
            }
            lower->program.private_size = lower->nir->scratch_size;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_shared) {
            if (intr->src[0].ssa->bit_size != 32 ||
                nir_intrinsic_align_mul(intr) < 4 ||
                (nir_intrinsic_align_offset(intr) & 3) ||
                lower->shared_argument == UINT_MAX)
               return false;
            uint32_t base = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[1].ssa, 0));
            for (unsigned c = 0; c < intr->num_components; ++c) {
               if (!(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)))
                  continue;
               uint32_t address = base;
               unsigned offset = nir_intrinsic_base(intr) + 4 * c;
               if (offset) {
                  uint32_t add[] = {address, apple9_dag_imm(lower, offset)};
                  address = apple9_dag_emit(lower, AGX_APPLE9_VIR_IADD,
                     AGX_APPLE9_ENC_INT_ADD_EXTENDED, add, 2, 0);
               }
               address = apple9_dag_shift_imm(lower, nir_op_ushr, address, 2);
               uint32_t sources[] = {
                  apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[0].ssa, c)),
                  address,
               };
               if (!agx_apple9_vir_emit_side_effect(&lower->program,
                      AGX_APPLE9_VIR_SHARED_STORE, AGX_APPLE9_ENC_SHARED_STORE,
                      sources, 2, lower->argument_base + lower->shared_argument))
                  return false;
            }
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_shared_atomic ||
             intr->intrinsic == nir_intrinsic_shared_atomic_swap) {
            enum agx_apple9_atomic_op operation;
            bool swap = intr->intrinsic == nir_intrinsic_shared_atomic_swap;
            if (intr->def.bit_size != 32 || intr->def.num_components != 1 ||
                !apple9_atomic_op(nir_intrinsic_atomic_op(intr), &operation) ||
                (swap != (operation == AGX_APPLE9_ATOMIC_CMPXCHG)) ||
                operation == AGX_APPLE9_ATOMIC_FADD ||
                lower->shared_argument == UINT_MAX)
               return false;
            uint32_t address = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[0].ssa, 0));
            if (nir_intrinsic_base(intr)) {
               uint32_t add[] = {address,
                  apple9_dag_imm(lower, nir_intrinsic_base(intr))};
               address = apple9_dag_emit(lower, AGX_APPLE9_VIR_IADD,
                  AGX_APPLE9_ENC_INT_ADD_EXTENDED, add, 2, 0);
            }
            address = apple9_dag_shift_imm(lower, nir_op_ushr, address, 2);
            unsigned components = swap ? 2 : 1;
            uint32_t sources[3];
            /* The native compare-exchange tuple is desired, compare. */
            sources[0] = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[swap ? 2 : 1].ssa, 0));
            if (swap)
               sources[1] = apple9_lower_dag_scalar(lower,
                  nir_get_scalar(intr->src[1].ssa, 0));
            sources[components] = address;
            uint32_t result = apple9_dag_emit(lower,
               AGX_APPLE9_VIR_SHARED_ATOMIC, AGX_APPLE9_ENC_SHARED_ATOMIC,
               sources, components + 1, lower->argument_base + lower->shared_argument);
            if (result == AGX_APPLE9_VREG_INVALID)
               return false;
            struct agx_apple9_vir_instr *atomic =
               lower->program.instructions[lower->program.instruction_count - 1];
            atomic->atomic_op = operation;
            atomic->memory_components = components;
            atomic->producer_scoreboard_slot = AGX_APPLE9_SCOREBOARD_SLOT_1;
            lower->ssa_to_vreg[intr->def.index * 4] = result;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_zs_agx) {
            const unsigned zs = nir_intrinsic_base(intr);
            if (!zs || (zs & ~3u))
               return false;
            uint32_t mask = apple9_lower_dag_scalar(
               lower, nir_get_scalar(intr->src[0].ssa, 0));
            uint32_t control = apple9_pack_coverage(lower, mask, mask);
            /* The control word keeps coverage in its low half and carries a
             * 16-bit stencil reference in its high half. The second word is
             * depth. Mode 1 consumes stencil, 3 depth, and 7 both. */
            if (zs & 2) {
               uint32_t stencil = apple9_lower_dag_scalar(lower,
                  nir_get_scalar(intr->src[2].ssa, 0));
               uint32_t sources[] = {control,
                  apple9_dag_shift_imm(lower, nir_op_ishl, stencil, 16)};
               control = apple9_dag_emit(lower, AGX_APPLE9_VIR_IOR,
                  AGX_APPLE9_ENC_LOGIC_EXTENDED, sources, 2, 0);
            }
            /* Hardware bounds exported depth to [0, 1]; drivers clamp to a
             * narrower depth-clamp range in NIR. NaN passes through. */
            uint32_t depth = control;
            if (zs & 1)
               depth = apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[1].ssa, 0));
            unsigned mode = zs == 1 ? 3 : zs == 2 ? 1 : 7;
            uint32_t fields[] = {control, depth};
            /* Stencil-only publication reads the control word split by lane
             * half; depth modes read lane-major control and depth words. */
            uint32_t value = agx_apple9_vir_emit_publication_pair(
               &lower->program, fields, zs == 2);
            uint32_t pair[] = {value, value + 1};
            if (value == AGX_APPLE9_VREG_INVALID ||
                !agx_apple9_vir_emit_side_effect(&lower->program,
                   AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0, 1) ||
                !agx_apple9_vir_emit_side_effect(&lower->program,
                   AGX_APPLE9_VIR_DEPTH_STORE, AGX_APPLE9_ENC_DEPTH_STORE, pair, 2, mode) ||
                !apple9_emit_zs_release(lower))
               return false;
            lower->writes_z |= zs & 1;
            lower->writes_s |= !!(zs & 2);
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
            uint32_t affected = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[0].ssa, 0));
            uint32_t live = apple9_lower_dag_scalar(lower,
               nir_get_scalar(intr->src[1].ssa, 0));
            /* A depth/stencil export performs the fragment's only test.
             * Earlier coverage publications just record killed samples. */
            if (lower->nir->info.stage != MESA_SHADER_FRAGMENT ||
                affected == AGX_APPLE9_VREG_INVALID || live == AGX_APPLE9_VREG_INVALID ||
                !apple9_emit_coverage(lower, affected, live, !lower->zs_export &&
                   !(nir_src_is_const(intr->src[1]) && !nir_src_as_uint(intr->src[1]))))
               return false;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_store_output ||
             intr->intrinsic == nir_intrinsic_store_uvs_agx ||
             intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
            if (!apple9_emit_graphics_output(lower, intr)) {
               if (!lower->reason)
                  lower->reason = "unsupported Apple9 graphics output";
               return false;
            }
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_image_store_apple9_agx) {
            uint32_t index = apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[0].ssa, 0));
            uint32_t xy[2], rgba[4];
            for (unsigned c = 0; c < 2; ++c)
               xy[c] = apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[1].ssa, c));
            for (unsigned c = 0; c < 4; ++c)
               rgba[c] = apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[2].ssa, c));
            if (index == AGX_APPLE9_VREG_INVALID ||
                !agx_apple9_vir_emit_image_store(&lower->program, xy, rgba, index))
               return false;
            continue;
         }
         if (intr->intrinsic == nir_intrinsic_image_store_block_agx) {
            bool multisampled = nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS;
            if ((!multisampled && nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_2D) ||
                nir_intrinsic_explicit_coord(intr))
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
               nir_intrinsic_image_array(intr)
                  ? apple9_lower_dag_scalar(lower, nir_get_scalar(intr->src[2].ssa, 2))
                  : AGX_APPLE9_VREG_INVALID,
            };
            /* The last word combines the tile byte offset above the mip
             * level. Layered views retain the whole mip allocation so the
             * hardware derives the physical stride between array planes. */
            src[2] = apple9_dag_shift_imm(lower, nir_op_ishl, src[2], 16);
            if (!agx_apple9_vir_emit_block_image_store(&lower->program, src, image,
                                                       tile_format, multisampled,
                                                       nir_intrinsic_image_array(intr)))
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
         if (intr->intrinsic == nir_intrinsic_store_ssbo ||
             intr->intrinsic == nir_intrinsic_store_global_2x32) {
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
         if (apple9_is_buffer_atomic(intr->intrinsic)) {
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
      if (def && def->bit_size == 1 && instr->type != nir_instr_type_load_const) {
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

      const bool volatile_load = instr->type == nir_instr_type_intrinsic &&
         nir_intrinsic_has_access(nir_instr_as_intrinsic(instr)) &&
         (nir_intrinsic_access(nir_instr_as_intrinsic(instr)) & ACCESS_VOLATILE);
      const nir_component_mask_t read = volatile_load
         ? BITFIELD_MASK(def->num_components) : nir_def_components_read(def);
      const unsigned first = lower->program.instruction_count;
      for (unsigned c = 0; c < def->num_components; ++c) {
         if ((read & BITFIELD_BIT(c)) &&
             apple9_lower_dag_scalar(lower, nir_get_scalar(def, c)) ==
                AGX_APPLE9_VREG_INVALID)
            return false;
      }
      if (volatile_load) {
         for (unsigned i = first; i < lower->program.instruction_count; ++i) {
            struct agx_apple9_vir_instr *ins = lower->program.instructions[i];
            if (ins->op == AGX_APPLE9_VIR_DEVICE_LOAD)
               ins->memory_volatile = true;
         }
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
      if (compare.domain == APPLE9_COMPARE_FLOAT &&
          nir_scalar_chase_alu_src(predicate, 0).def->bit_size == 16) {
         plan.encoding = plan.encoding == AGX_APPLE9_ENC_PREDICATE_COMPARE_SHORT
            ? AGX_APPLE9_ENC_HALF_PREDICATE_SHORT
            : AGX_APPLE9_ENC_HALF_PREDICATE_EXTENDED;
      }
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

/* Estimated issue cost of a structured arm, weighted as for the common
 * backend's forward-branch heuristic: memory and texture operations stand in
 * for their latency. */
static unsigned
apple9_cf_list_cost(struct exec_list *list)
{
   unsigned cost = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               cost += 50;
            else if (instr->type == nir_instr_type_intrinsic)
               cost += 10;
            else if (instr->type == nir_instr_type_alu)
               cost += 1;
         }
      }
   }
   return cost;
}

/* Beyond this cost, an arm is skipped when no lane executes it. */
#define APPLE9_SKIP_COST 19

/* Skip an arm with no active lanes by jumping to a block that begins with the
 * following mask operation. An arm executed with an empty mask writes no
 * lane, so the logical CFG remains exact for allocation; the physical
 * scheduler follows the jump edge for pending results. */
static bool
apple9_emit_skip(struct apple9_dag_lower *lower, struct exec_list *arm,
                 struct agx_apple9_block **target)
{
   *target = NULL;
   if (apple9_cf_list_cost(arm) <= APPLE9_SKIP_COST)
      return true;
   *target = agx_apple9_block_create(&lower->program);
   if (!*target ||
       !agx_apple9_vir_emit_branch(&lower->program, AGX_APPLE9_VIR_JMP_EXEC_NONE,
                                   AGX_APPLE9_ENC_JMP_EXEC_NONE, *target)) {
      lower->reason = "could not emit an Apple9 forward branch";
      return false;
   }
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
   struct agx_apple9_block *else_target, *pop_target;
   if (!apple9_emit_if_predicate(lower, nif, &invert_push) ||
       !apple9_emit_exec_mask(lower, true, invert_push) ||
       !apple9_emit_skip(lower, &nif->then_list, &else_target))
      return false;
   ++lower->mask_depth;

   nir_block *then_pred = apple9_cf_list_last_block(&nif->then_list);
   nir_block *else_pred = apple9_cf_list_last_block(&nif->else_list);
   if (!apple9_emit_cf_list(lower, stores, &nif->then_list) ||
       (has_phis && apple9_block_reaches(then_pred, merge) &&
        !apple9_emit_phi_copies_for_edge(lower, merge, then_pred)))
      return false;

   if (else_target)
      agx_apple9_block_begin(&lower->program, else_target);
   if (!apple9_emit_exec_mask_else(lower) ||
       !apple9_emit_skip(lower, &nif->else_list, &pop_target) ||
       !apple9_emit_cf_list(lower, stores, &nif->else_list) ||
       (has_phis && apple9_block_reaches(else_pred, merge) &&
        !apple9_emit_phi_copies_for_edge(lower, merge, else_pred)))
      return false;

   if (pop_target)
      agx_apple9_block_begin(&lower->program, pop_target);
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
         if (lower->zs_release_pending && !lower->loop_depth) {
            lower->zs_release_pending = false;
            if (!apple9_emit_zs_release(lower))
               return false;
         }
         if (!apple9_emit_block(lower, stores, nir_cf_node_as_block(node)))
            return false;
         break;
      }
      case nir_cf_node_if:
         if (!apple9_emit_if(lower, stores, nir_cf_node_as_if(node)))
            return false;
         break;
      case nir_cf_node_loop:
         lower->loop_depth++;
         if (!apple9_emit_loop(lower, stores, nir_cf_node_as_loop(node)))
            return false;
         lower->loop_depth--;
         break;
      default:
         lower->reason = "Apple9 compiler requires structured control flow";
         return false;
      }
   }

   return true;
}

static bool
apple9_lower_global_address_pair(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   bool load = intr->intrinsic == nir_intrinsic_load_global ||
               intr->intrinsic == nir_intrinsic_load_global_constant;
   bool store = intr->intrinsic == nir_intrinsic_store_global;
   bool atomic = intr->intrinsic == nir_intrinsic_global_atomic ||
                 intr->intrinsic == nir_intrinsic_global_atomic_swap;
   if (!load && !store && !atomic)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_intrinsic_op op = load ? nir_intrinsic_load_global_2x32
                             : nir_intrinsic_store_global_2x32;
   if (atomic)
      op = intr->intrinsic == nir_intrinsic_global_atomic
              ? nir_intrinsic_global_atomic_2x32
              : nir_intrinsic_global_atomic_swap_2x32;
   nir_intrinsic_instr *repl = nir_intrinsic_instr_create(b->shader, op);
   repl->num_components = intr->num_components;
   nir_intrinsic_copy_const_indices(repl, intr);
   /* Constant memory is not written while the shader runs. The pair form
    * has no constant variant, so carry that as reorderability. */
   if (intr->intrinsic == nir_intrinsic_load_global_constant)
      nir_intrinsic_set_access(repl, nir_intrinsic_access(repl) | ACCESS_CAN_REORDER);
   for (unsigned s = 0; s < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++s)
      repl->src[s] = intr->src[s];
   unsigned address_src = store ? 1 : 0;
   repl->src[address_src] = nir_src_for_ssa(
      nir_unpack_64_2x32(b, intr->src[address_src].ssa));
   if (!store)
      nir_def_init(&repl->instr, &repl->def, intr->def.num_components, intr->def.bit_size);
   nir_builder_instr_insert(b, &repl->instr);
   if (!store)
      nir_def_rewrite_uses(&intr->def, &repl->def);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Keep native binary16 arithmetic, comparisons and special functions.
 * Remaining composite math keeps the common FP32 expansion. */
static unsigned
apple9_float_alu_width(const nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return 0;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   switch (alu->op) {
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_ffma_weak:
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_fsat:
   case nir_op_fsqrt:
   case nir_op_frcp:
   case nir_op_frsq:
   case nir_op_fexp2:
   case nir_op_flog2:
   case nir_op_ffloor:
   case nir_op_fceil:
   case nir_op_ftrunc:
   case nir_op_fround_even:
   case nir_op_fabs:
   case nir_op_fneg:
      return 0;
   default:
      break;
   }
   if (alu->def.bit_size == 16 &&
       nir_op_infos[alu->op].output_type == nir_type_float)
      return 32;
   /* Narrow integers occupy 32-bit registers without extension. Operations
    * whose result depends on the undefined high bits run at 32 bits with
    * NIR's sign/zero extension. Moves, selects, conversions and bitwise
    * operations only produce the defined low bits. */
   switch (alu->op) {
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec5:
   case nir_op_vec8:
   case nir_op_vec16:
   case nir_op_bcsel:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_inot:
   case nir_op_iadd:
   case nir_op_isub:
   case nir_op_imul:
   case nir_op_ineg:
   case nir_op_ishl:
      return 0;
   default:
      break;
   }
   if (nir_op_infos[alu->op].is_conversion)
      return 0;
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
      nir_alu_type type = nir_op_infos[alu->op].input_types[i];
      unsigned bits = alu->src[i].src.ssa->bit_size;
      if ((nir_alu_type_get_base_type(type) == nir_type_int ||
           nir_alu_type_get_base_type(type) == nir_type_uint) &&
          bits > 1 && bits < 32)
         return 32;
   }
   return 0;
}

/* Spread each input's low 16 bits into alternating bits of a 32-bit word.
 * Image address calculations use this Morton interleave operation.
 */
static bool
apple9_lower_interleave(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_interleave)
      return false;
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *spread[2];
   static const unsigned shifts[] = {8, 4, 2, 1};
   static const uint32_t masks[] = {0x00ff00ff, 0x0f0f0f0f, 0x33333333, 0x55555555};
   for (unsigned c = 0; c < 2; ++c) {
      spread[c] = nir_iand_imm(b, nir_u2u32(b, nir_ssa_for_alu_src(b, alu, c)), 0xffff);
      for (unsigned i = 0; i < ARRAY_SIZE(shifts); ++i)
         spread[c] = nir_iand_imm(b,
            nir_ior(b, spread[c], nir_ishl_imm(b, spread[c], shifts[i])), masks[i]);
   }
   nir_def_replace(&alu->def,
      nir_ior(b, spread[0], nir_ishl_imm(b, spread[1], 1)));
   return true;
}

static bool
apple9_lower_half_conversion(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (alu->op != nir_op_u2f16 && alu->op != nir_op_i2f16)
      return false;
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *source = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *wide = alu->op == nir_op_u2f16 ? nir_u2f32(b, source)
                                           : nir_i2f32(b, source);
   nir_def_replace(&alu->def, nir_f2f16(b, wide));
   return true;
}

/* Shared memory accesses take a 32-bit byte offset. Algebraic rules can
 * narrow the offset computation when its range is small.
 */
static bool
apple9_widen_shared_offset(nir_builder *b, nir_intrinsic_instr *intr,
                           UNUSED void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_shared:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
      break;
   default:
      return false;
   }
   nir_src *offset = nir_get_io_offset_src(intr);
   if (offset->ssa->bit_size >= 32)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(offset, nir_u2u32(b, offset->ssa));
   return true;
}

/* Shared and scratch memory are addressed in words. Narrow loads extract
 * from the containing word. Narrow shared stores update only their own bits
 * with an atomic AND/OR pair, so invocations storing neighbouring bytes do
 * not race; private scratch stores read, modify and write the word.
 */
static bool
apple9_lower_narrow_shared(nir_builder *b, nir_intrinsic_instr *intr,
                           UNUSED void *data)
{
   bool scratch = intr->intrinsic == nir_intrinsic_load_scratch ||
                  intr->intrinsic == nir_intrinsic_store_scratch;
   bool store = intr->intrinsic == nir_intrinsic_store_shared ||
                intr->intrinsic == nir_intrinsic_store_scratch;
   if (!store && intr->intrinsic != nir_intrinsic_load_shared &&
       intr->intrinsic != nir_intrinsic_load_scratch)
      return false;
   unsigned bits = store ? intr->src[0].ssa->bit_size : intr->def.bit_size;
   if (bits >= 32)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = nir_u2u32(b, nir_get_io_offset_src(intr)->ssa);
   if (!scratch)
      base = nir_iadd_imm(b, base, nir_intrinsic_base(intr));
   unsigned components = store ? intr->src[0].ssa->num_components
                               : intr->def.num_components;
   nir_def *values[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < components; ++c) {
      if (store && !(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)))
         continue;
      nir_def *offset = nir_iadd_imm(b, base, c * bits / 8);
      nir_def *word = nir_iand_imm(b, offset, ~3u);
      nir_def *shift = nir_imul_imm(b, nir_iand_imm(b, offset, 3), 8);
      if (store) {
         nir_def *mask = nir_ishl(b, nir_imm_int(b, BITFIELD_MASK(bits)), shift);
         nir_def *value = nir_ishl(b,
            nir_iand_imm(b, nir_u2u32(b, nir_channel(b, intr->src[0].ssa, c)),
                         BITFIELD_MASK(bits)), shift);
         if (scratch) {
            /* Scratch is private to the invocation: read-modify-write. */
            nir_def *old = nir_load_scratch(b, 1, 32, word, .align_mul = 4);
            nir_store_scratch(b, nir_ior(b, nir_iand(b, old, nir_inot(b, mask)),
                                         value), word, .align_mul = 4);
         } else {
            nir_shared_atomic(b, 32, word, nir_inot(b, mask),
                              .atomic_op = nir_atomic_op_iand);
            nir_shared_atomic(b, 32, word, value,
                              .atomic_op = nir_atomic_op_ior);
         }
      } else {
         nir_def *loaded = scratch
            ? nir_load_scratch(b, 1, 32, word, .align_mul = 4)
            : nir_load_shared(b, 1, 32, word, .align_mul = 4);
         values[c] = nir_u2uN(b, nir_ushr(b, loaded, shift), bits);
      }
   }
   if (store)
      nir_instr_remove(&intr->instr);
   else
      nir_def_replace(&intr->def, nir_vec(b, values, components));
   return true;
}

static void
apple9_legalize_buffer_accesses(nir_shader *nir)
{
   nir_shader_intrinsics_pass(nir, apple9_lower_narrow_shared,
                              nir_metadata_control_flow, NULL);
   nir_shader_alu_pass(nir, apple9_lower_interleave,
                       nir_metadata_control_flow, NULL);
   nir_shader_alu_pass(nir, apple9_lower_half_conversion,
                       nir_metadata_control_flow, NULL);
   nir_lower_bit_size(nir, apple9_float_alu_width, NULL);
   /* The shared zero/scratch-page ABI also applies to software geometry.
    * Resolve its pointer intrinsics before splitting 64-bit addresses. */
   agx_nir_lower_sink_address(nir);
   const nir_lower_mem_access_bit_sizes_options memory_options = {
      .modes = nir_var_mem_ssbo | nir_var_mem_ubo | nir_var_mem_global |
               nir_var_mem_constant | nir_var_mem_shared | nir_var_function_temp,
      .callback = apple9_memory_format,
   };
   nir_opt_load_store_update_alignments(nir);
   nir_lower_mem_access_bit_sizes(nir, &memory_options);
   nir_shader_intrinsics_pass(nir, apple9_lower_global_address_pair,
                             nir_metadata_control_flow, NULL);
   /* Memory legalization can create ALU operations after preprocessing.
    * Keep native unary bit operations and normalized conversions for
    * selection while expanding the remaining composite operations. */
   const nir_shader_compiler_options *saved_options = nir->options;
   nir_shader_compiler_options memory_alu_options = *saved_options;
   memory_alu_options.lower_bitfield_extract8 = true;
   memory_alu_options.lower_bitfield_extract16 = true;
   memory_alu_options.lower_bitfield_extract = true;
   memory_alu_options.has_bfe = false;
   memory_alu_options.lower_extract_byte = true;
   memory_alu_options.lower_extract_word = true;
   memory_alu_options.has_pack_32_4x8 = false;
   memory_alu_options.lower_bit_count = false;
   memory_alu_options.lower_ufind_msb = false;
   memory_alu_options.lower_ifind_msb = true;
   memory_alu_options.lower_find_lsb = true;
   memory_alu_options.lower_usub_sat = true;
   memory_alu_options.lower_pack_unorm_2x16 = false;
   memory_alu_options.lower_pack_unorm_4x8 = false;
   memory_alu_options.lower_unpack_unorm_2x16 = false;
   memory_alu_options.lower_unpack_unorm_4x8 = false;
   memory_alu_options.lower_unpack_snorm_2x16 = false;
   memory_alu_options.lower_unpack_snorm_4x8 = false;
   memory_alu_options.lower_int64_options |= nir_lower_iadd64 | nir_lower_bcsel64;
   nir->options = &memory_alu_options;
   nir_lower_int64(nir);
   nir_lower_64bit_phis(nir);
   nir_lower_alu(nir);
   nir_lower_pack(nir);
   nir_opt_algebraic(nir);
   nir_shader_alu_pass(nir, apple9_lower_memory_pack, nir_metadata_control_flow, NULL);

   /* Undefined lanes can remain in vectors with partial output write masks.
    * Materialize them consistently before instruction selection. */
   nir_lower_undef_to_zero(nir, NULL);
   /* Format and shader-math lowering can create identical expressions. */
   bool progress;
   do {
      progress = nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_algebraic(nir);
      progress |= nir_opt_cse(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);
   /* Algebraic optimization can reform 16-bit packing operations. Lower them
    * again without running the algebraic rules that would recombine them. */
   /* Algebraic rules also narrow comparisons of extended values again, so
    * repeat the width legalization after the last algebraic pass. */
   if (nir_lower_pack(nir) |
       nir_shader_alu_pass(nir, apple9_lower_memory_pack,
                           nir_metadata_control_flow, NULL) |
       nir_lower_bit_size(nir, apple9_float_alu_width, NULL)) {
      nir_opt_constant_folding(nir);
      nir_opt_copy_prop(nir);
      nir_opt_cse(nir);
      nir_opt_dce(nir);
   }
   nir->options = saved_options;
}

static void
apple9_prepare_dag(nir_shader *nir)
{
   /* Combine adjacent memory and execution barriers before instruction
    * selection. A shared-memory fence followed by a workgroup barrier is
    * implemented by the same workgroup synchronization instruction.
    */
   nir_opt_combine_barriers(nir, NULL, NULL);

   /* Legalization preserves vectors it is given, but does not discover
    * adjacent scalar accesses. Canonicalize their addresses first and let
    * NIR retain its alignment, aliasing and wrapping-offset checks. */
   nir_opt_constant_folding(nir);
   nir_opt_cse(nir);
   nir_opt_dce(nir);
   /* Vulkan buffers and descriptors are global memory; their 64-bit
    * addresses are still unsplit here. */
   const nir_load_store_vectorize_options vector_options = {
      .modes = nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_mem_global |
               nir_var_mem_constant,
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
   /* Constant folding can turn zero gradients into txl after texture lowering.
    * Normalize those new LOD operands before selection/preamble extraction. */
   if (agx_nir_lower_apple9_texture_lod(nir)) {
      nir_opt_constant_folding(nir);
      nir_opt_copy_prop(nir);
      nir_opt_dce(nir);
   }
   nir_shader_intrinsics_pass(nir, apple9_widen_shared_offset,
                              nir_metadata_control_flow, NULL);
}

/* At most this many hoisted samples are unconsumed at once: eight
 * four-register coordinate tuples fill p0-p31. AGX_APPLE9_HOIST_SAMPLES=0
 * disables hoisting. */
static unsigned
apple9_hoist_limit(void)
{
   const char *env = debug_get_option_cached_agx_apple9_hoist_samples();
   return env ? atoi(env) : 8;
}

/* Long-latency operations to issue early: samples that fetch from memory
 * (queries return without waiting). Buffer-load hoisting is opt-in with
 * AGX_APPLE9_HOIST_LOADS=1 until its register-pressure tradeoffs improve. */
static bool
apple9_hoist_sample(nir_instr *instr)
{
   if (instr->type == nir_instr_type_tex)
      return !nir_tex_instr_is_query(nir_instr_as_tex(instr));
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_global_2x32: {
      return debug_get_option_apple9_hoist_loads() &&
             nir_intrinsic_can_reorder(intr);
   }
   default:
      return false;
   }
}

/* Instructions that may move up with a hoisted operation, as part of the
 * slice computing its operands: pure ALU and reorderable intrinsics.
 */
static bool
apple9_hoist_movable(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
   case nir_instr_type_tex:
      return true;
   case nir_instr_type_intrinsic:
      return nir_intrinsic_can_reorder(nir_instr_as_intrinsic(instr));
   default:
      return false;
   }
}

/* Instructions a hoisted operation may not cross. Stores, atomics and shared
 * memory accesses cannot change what it reads: reorderable loads do not alias
 * them, and sampled images are not coherent with writes made by the same
 * invocation. Output stores and subgroup operations do not interact with it
 * either. Anything that changes which invocations run (discard, demote,
 * terminate), interlocks, barriers and unknown intrinsics stay fences.
 */
static bool
apple9_hoist_fence(nir_instr *instr)
{
   if (apple9_hoist_movable(instr))
      return false;
   if (instr->type != nir_instr_type_intrinsic)
      return true;
   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_uvs_agx:
   case nir_intrinsic_store_global_2x32:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_load_shared:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_global_atomic_2x32:
   case nir_intrinsic_global_atomic_swap_2x32:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_ballot:
   case nir_intrinsic_exclusive_scan:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_reduce:
      return false;
   default:
      return true;
   }
}

/* Scalar-word liveness is a conservative scheduling cost, not a physical
 * register bound: later uniform folding and operand constraints can change
 * allocation. Compare both orders with the same model before extending
 * texture-result lifetimes. */
struct apple9_word_pressure {
   unsigned *width;
   BITSET_WORD *live;
   unsigned demand;
};

static bool
apple9_record_width(nir_def *def, void *data)
{
   unsigned *width = data;
   width[def->index] = def->num_components * DIV_ROUND_UP(def->bit_size, 32);
   return true;
}

static bool
apple9_add_live(nir_src *src, void *data)
{
   struct apple9_word_pressure *s = data;
   if (!nir_src_is_undef(*src) && !BITSET_TEST(s->live, src->ssa->index)) {
      BITSET_SET(s->live, src->ssa->index);
      s->demand += s->width[src->ssa->index];
   }
   return true;
}

static bool
apple9_kill_live(nir_def *def, void *data)
{
   struct apple9_word_pressure *s = data;
   if (BITSET_TEST(s->live, def->index)) {
      BITSET_CLEAR(s->live, def->index);
      s->demand -= s->width[def->index];
   }
   return true;
}

static unsigned
apple9_schedule_pressure(nir_block *block, nir_instr **order, unsigned count,
                         unsigned *width, BITSET_WORD *live, unsigned words)
{
   memset(live, 0, words * sizeof(*live));
   struct apple9_word_pressure s = {.width = width, .live = live};
   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, id) {
      BITSET_SET(live, id);
      s.demand += width[id];
   }
   nir_if *following = nir_block_get_following_if(block);
   if (following)
      apple9_add_live(&following->condition, &s);
   unsigned peak = s.demand;
   while (count) {
      nir_instr *instr = order[--count];
      nir_foreach_def(instr, apple9_kill_live, &s);
      nir_foreach_src(instr, apple9_add_live, &s);
      peak = MAX2(peak, s.demand);
   }
   return peak;
}

struct apple9_hoist_state {
   nir_block *block;
   BITSET_WORD *emitted;
   /* Emitted samples whose results no emitted instruction has consumed. */
   BITSET_WORD *pending;
   /* Readiness walk memo: instructions already found ready in this query. */
   unsigned *visited;
   unsigned query;
   nir_instr **order;
   unsigned count;
};

static void apple9_hoist_emit(struct apple9_hoist_state *s, nir_instr *instr);

static bool
apple9_hoist_emit_src(nir_src *src, void *data)
{
   struct apple9_hoist_state *s = data;
   nir_instr *parent = nir_def_instr(src->ssa);
   if (parent->block == s->block && parent->type != nir_instr_type_phi &&
       !BITSET_TEST(s->emitted, parent->index))
      apple9_hoist_emit(s, parent);
   return true;
}

static bool
apple9_hoist_retire_src(nir_src *src, void *data)
{
   struct apple9_hoist_state *s = data;
   nir_instr *parent = nir_def_instr(src->ssa);
   if (parent->block == s->block)
      BITSET_CLEAR(s->pending, parent->index);
   return true;
}

/* Emit an instruction after its unemitted in-block dependencies. */
static void
apple9_hoist_emit(struct apple9_hoist_state *s, nir_instr *instr)
{
   nir_foreach_src(instr, apple9_hoist_emit_src, s);
   if (BITSET_TEST(s->emitted, instr->index))
      return;
   nir_foreach_src(instr, apple9_hoist_retire_src, s);
   BITSET_SET(s->emitted, instr->index);
   s->order[s->count++] = instr;
}

/* Whether a sample's unemitted dependency slice can move up to the current
 * position: it contains no fence and no other sample, and does not consume a
 * pending sample (that would wait for it early instead of overlapping).
 */
static bool
apple9_hoist_ready_src(nir_src *src, void *data)
{
   struct apple9_hoist_state *s = data;
   nir_instr *parent = nir_def_instr(src->ssa);
   if (parent->block != s->block || parent->type == nir_instr_type_phi)
      return true;
   if (BITSET_TEST(s->emitted, parent->index))
      return !BITSET_TEST(s->pending, parent->index);
   if (s->visited[parent->index] == s->query)
      return true;
   if (!apple9_hoist_movable(parent) || apple9_hoist_sample(parent))
      return false;
   s->visited[parent->index] = s->query;
   return nir_foreach_src(parent, apple9_hoist_ready_src, s);
}

static unsigned
apple9_hoist_inflight(struct apple9_hoist_state *s, unsigned words)
{
   unsigned n = 0;
   for (unsigned i = 0; i < words; ++i)
      n += util_bitcount(s->pending[i]);
   return n;
}

/* Issue texture samples early. Selection follows NIR order, so a sample
 * placed right before its first use waits for the whole fetch. Within each region between fences,
 * pull later samples forward, with the arithmetic computing their
 * coordinates, while fewer than AGX_APPLE9_HOIST_SAMPLES (default 8: eight
 * four-register coordinate tuples fill p0-p31) are unconsumed. Everything
 * else keeps its order.
 */
static bool apple9_has_samples(nir_shader *nir);

bool
agx_nir_hoist_apple9_samples(nir_shader *nir)
{
   unsigned limit = apple9_hoist_limit();
   /* How far ahead, in NIR instructions, to look for a sample to hoist: an
    * upper bound on how far one moves. */
   const unsigned horizon = 256;
   if (!limit || !apple9_has_samples(nir))
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_live_defs);
   unsigned total = nir_index_instrs(impl);
   unsigned words = BITSET_WORDS(total);
   void *ctx = ralloc_context(NULL);
   struct apple9_hoist_state s = {
      .emitted = rzalloc_array(ctx, BITSET_WORD, words + 1),
      .pending = rzalloc_array(ctx, BITSET_WORD, words + 1),
      .visited = rzalloc_array(ctx, unsigned, total + 1),
      .order = ralloc_array(ctx, nir_instr *, total + 1),
   };
   unsigned *width = rzalloc_array(ctx, unsigned, impl->ssa_alloc);
   nir_foreach_block(b, impl) {
      nir_foreach_instr(i, b)
         nir_foreach_def(i, apple9_record_width, width);
   }
   unsigned live_words = BITSET_WORDS(impl->ssa_alloc);
   BITSET_WORD *live = rzalloc_array(ctx, BITSET_WORD, live_words);
   nir_instr **original = ralloc_array(ctx, nir_instr *, total + 1);
   bool progress = false;

   nir_foreach_block(block, impl) {
      s.block = block;
      s.count = 0;
      bool block_progress = false;
      unsigned original_count = 0;
      nir_foreach_instr(i, block) {
         if (i->type != nir_instr_type_phi && i->type != nir_instr_type_jump)
            original[original_count++] = i;
      }
      nir_instr *jump = nir_block_last_instr(block);
      if (jump && jump->type != nir_instr_type_jump)
         jump = NULL;

      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_phi || instr == jump ||
             BITSET_TEST(s.emitted, instr->index))
            continue;

         /* Before an ordinary instruction, pull later samples of the same
          * region forward while there is room. */
         if (!apple9_hoist_fence(instr)) {
            unsigned inflight = apple9_hoist_inflight(&s, words);
            unsigned scanned = 0;
            for (nir_instr *next = nir_instr_next(instr);
                 next && next != jump && inflight < limit &&
                 scanned < horizon && !apple9_hoist_fence(next);
                 next = nir_instr_next(next), ++scanned) {
               if (!apple9_hoist_sample(next) ||
                   BITSET_TEST(s.emitted, next->index))
                  continue;
               ++s.query;
               if (!nir_foreach_src(next, apple9_hoist_ready_src, &s))
                  continue;
               apple9_hoist_emit(&s, next);
               BITSET_SET(s.pending, next->index);
               ++inflight;
               block_progress = true;
            }
         }

         if (BITSET_TEST(s.emitted, instr->index))
            continue;
         apple9_hoist_emit(&s, instr);
         if (apple9_hoist_sample(instr))
            BITSET_SET(s.pending, instr->index);
      }

      /* Samples still pending are consumed in later blocks. */
      memset(s.pending, 0, (words + 1) * sizeof(BITSET_WORD));

      if (!block_progress)
         continue;
      assert(original_count == s.count);
      if (apple9_schedule_pressure(block, s.order, s.count, width, live, live_words) >
          apple9_schedule_pressure(block, original, original_count, width, live, live_words))
         continue;

      progress = true;
      /* Rebuild the block: phis, the new order, then the jump. */
      nir_cursor cursor = nir_after_phis(block);
      for (unsigned i = 0; i < s.count; ++i) {
         nir_instr_move(cursor, s.order[i]);
         cursor = nir_after_instr(s.order[i]);
      }
   }
   ralloc_free(ctx);
   if (progress)
      nir_progress(true, impl, nir_metadata_control_flow);
   return progress;
}

/* Compiler dumps for the compile in progress. Trial compiles write to a
 * buffer, so only the result that is kept is dumped. */
/* Derivatives read every lane of the quad, including lanes that have left
 * divergent control flow: D3D runs returned lanes on as helpers, and Vulkan
 * leaves the result undefined. A lane computes a value only where it executes,
 * so a coordinate interpolated after an early return holds stale data in the
 * returned lanes. Compute the input-derived operands of derivatives and
 * implicit-LOD samples in divergent control flow in the start block, which
 * every lane executes.
 */
#define APPLE9_DERIVATIVE_HOIST_MAX 32

static bool
apple9_derivative_hoist_src(nir_src *src, void *data);

static bool
apple9_derivative_hoistable(nir_instr *instr, nir_block *start, unsigned *budget)
{
   if (instr->block == start || (instr->pass_flags & 1))
      return true;
   bool ok = *budget > 0;
   if (ok) {
      --*budget;
      switch (instr->type) {
      case nir_instr_type_alu:
      case nir_instr_type_load_const:
      case nir_instr_type_undef:
         break;
      case nir_instr_type_intrinsic:
         /* Fragment inputs and uniform state. Memory loads stay where
          * their control flow may guard their addresses. */
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_load_interpolated_input:
         case nir_intrinsic_load_input:
         case nir_intrinsic_load_coefficients_agx:
         case nir_intrinsic_load_barycentric_pixel:
         case nir_intrinsic_load_barycentric_centroid:
         case nir_intrinsic_load_barycentric_sample:
         case nir_intrinsic_load_frag_coord:
         case nir_intrinsic_load_frag_coord_z:
         case nir_intrinsic_load_frag_coord_w:
         case nir_intrinsic_load_pixel_coord:
         case nir_intrinsic_load_point_coord:
         case nir_intrinsic_load_preamble:
            break;
         default:
            ok = false;
         }
         break;
      default:
         ok = false;
      }
   }
   ok = ok && nir_foreach_src(instr, apple9_derivative_hoist_src, budget);
   if (!ok)
      return false;
   /* Sources are marked first, so the marked order is a valid move order. */
   instr->pass_flags |= 1;
   return true;
}

static bool
apple9_derivative_hoist_src(nir_src *src, void *data)
{
   unsigned *budget = data;
   nir_instr *parent = nir_def_instr(src->ssa);
   nir_function_impl *impl = nir_cf_node_get_function(&parent->block->cf_node);
   return apple9_derivative_hoistable(parent, nir_start_block(impl), budget);
}

static bool
apple9_in_divergent_cf(nir_block *block)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_if &&
          nir_src_is_divergent(&nir_cf_node_as_if(node)->condition))
         return true;
      if (node->type == nir_cf_node_loop &&
          nir_loop_is_divergent(nir_cf_node_as_loop(node)))
         return true;
   }
   return false;
}

static nir_src *
apple9_derivative_operand(nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_ddx:
      case nir_intrinsic_ddy:
      case nir_intrinsic_ddx_fine:
      case nir_intrinsic_ddy_fine:
      case nir_intrinsic_ddx_coarse:
      case nir_intrinsic_ddy_coarse:
         return &intr->src[0];
      default:
         return NULL;
      }
   }
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (coord >= 0 && nir_tex_instr_has_implicit_derivative(tex))
         return &tex->src[coord].src;
   }
   return NULL;
}

bool
agx_nir_hoist_apple9_derivative_sources(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_divergence_analysis(nir);
   nir_block *start = nir_start_block(impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      if (block == start || !apple9_in_divergent_cf(block))
         continue;
      nir_foreach_instr(instr, block) {
         nir_src *src = apple9_derivative_operand(instr);
         if (!src)
            continue;

         nir_instr *parent = nir_def_instr(src->ssa);
         if (parent->block == start)
            continue;

         /* Mark the operand's dependency slice, then move it in order. */
         nir_shader_clear_pass_flags(nir);
         unsigned budget = APPLE9_DERIVATIVE_HOIST_MAX;
         if (!apple9_derivative_hoistable(parent, start, &budget))
            continue;

         nir_cursor cursor = nir_after_block_before_jump(start);
         nir_foreach_block(b, impl) {
            if (b == start)
               continue;
            nir_foreach_instr_safe(i, b) {
               if (i->pass_flags & 1) {
                  nir_instr_move(cursor, i);
                  cursor = nir_after_instr(i);
                  progress = true;
               }
            }
         }
      }
   }

   return nir_progress(progress, impl, nir_metadata_control_flow);
}

static _Thread_local FILE *apple9_log_stream;

static FILE *
apple9_log(void)
{
   return apple9_log_stream ? apple9_log_stream : stderr;
}

struct apple9_dag_cost {
   uint64_t weighted_spills;
   unsigned spills, instructions, gprs;
};

static bool
apple9_compile_dag_once(nir_shader *nir, struct agx_shader_part *out,
                        struct agx_apple9_compute_profile *profile,
                        const struct agx_apple9_varying_layout *varyings,
                        const struct apple9_buffer_map *resources,
                        const struct agx_apple9_uniform_constants *constants,
                        bool preamble, bool hoist,
                        struct apple9_dag_cost *cost, const char **reason)
{
   /* Preamble preparation runs algebraic optimization after prepare_dag and
    * can recreate fcanonicalize or bit-preserving float operations. Lower
    * FTZ after that last optimization boundary, independently in each main
    * and preamble candidate. Do not run algebraic rules after materializing
    * the integer checks. */
   /* FP32 DenormPreserve takes the same boundary: guard every affected
    * operation once no later rule can recreate one. */
   bool denorm_progress = agx_nir_lower_apple9_denorms(nir);
   denorm_progress |= agx_nir_lower_apple9_denorm_preserve(nir);
   if (denorm_progress) {
      bool progress;
      do {
         progress = nir_opt_constant_folding(nir);
         progress |= nir_opt_copy_prop(nir);
         progress |= nir_opt_cse(nir);
         progress |= nir_opt_dce(nir);
      } while (progress);
   }

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
   agx_nir_hoist_apple9_derivative_sources(nir);
   if (hoist)
      agx_nir_hoist_apple9_samples(nir);
   if (debug_get_option_cached_agx_apple9_trace_final_nir() != NULL)
      nir_print_shader(nir, apple9_log());

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
   bool uses_texel_fetch = false, has_texture = false;
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
            apple9_texture_table_masks(tex, &texture_mask, &sampler_mask,
                                       &uses_texel_fetch);
            has_texture = true;
         }
      }
   }
   /* Every varying store exports its value, and exports stay published
    * until completion. Only when that would exceed the publication bank are
    * values stored to several slots exported once. */
   unsigned varying_exports = 0;
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
            if (store->intrinsic == nir_intrinsic_store_output)
               varying_exports += util_bitcount(nir_intrinsic_write_mask(store));
            else if (store->intrinsic == nir_intrinsic_store_uvs_agx)
               varying_exports += store->src[0].ssa->num_components;
         }
      }
   }
   struct apple9_dag_lower lower = {
      .share_exports = varying_exports > AGX_APPLE9_PUBLICATION_COUNT,
      .nir = nir,
      .texture_mask = texture_mask,
      .image_mask = image_mask,
      .sampler_mask = sampler_mask,
      .varyings = varyings,
      .zero_vreg = AGX_APPLE9_VREG_INVALID,
      .entry_zero = AGX_APPLE9_VREG_INVALID,
      .loads = loads.data,
      .load_count =
         util_dynarray_num_elements(&loads, struct apple9_scalar_load),
      .atomics = atomics.data,
      .atomic_count =
         util_dynarray_num_elements(&atomics, struct apple9_buffer_atomic),
      .argument_base = resources->descriptor_tables
                          ? 2 : AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE,
      .buffer_table = resources->descriptor_tables,
      .groups_argument = resources->descriptor_tables
                            ? AGX_APPLE9_COMPUTE_GROUPS_ROOT : 0,
      .shared_argument = resources->descriptor_tables
         ? AGX_APPLE9_COMPUTE_SHARED_ARGUMENT - 2
         : apple9_buffer_argument(resources, AGX_APPLE9_COMPUTE_RESOURCE_SHARED, 0),
      .preamble_base = resources->descriptor_tables
                         ? AGX_APPLE9_GRAPHICS_ROOT_WORDS
                         : agx_apple9_compute_preamble_base(resources->count),
      .structured_cf = apple9_cf_list_has_control_flow(&impl->body),
   };
   /* Reserve depth before assigning point-coordinate coefficients, independent
    * of the order in which the DAG visits the two built-ins. */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_frag_coord_z)
            lower.reads_z = true;
         if (nir->info.stage == MESA_SHADER_FRAGMENT &&
             intr->intrinsic == nir_intrinsic_load_input &&
             nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PRIMITIVE_ID &&
             !varyings->mask[VARYING_SLOT_PRIMITIVE_ID])
            lower.reads_primitive_id = true;
         if (nir->info.stage != MESA_SHADER_FRAGMENT || !varyings)
            continue;
         /* Coefficients after the point coordinates depend on it. */
         if (intr->intrinsic == nir_intrinsic_load_point_coord)
            lower.reads_point_coord = true;
         if (apple9_is_barycentric_coord(intr->intrinsic)) {
            lower.barycentric_mask |=
               nir_intrinsic_interp_mode(intr) == INTERP_MODE_NOPERSPECTIVE ? 2 : 1;
         }
         if (intr->intrinsic == nir_intrinsic_load_input_vertex &&
             nir_src_is_const(intr->src[1])) {
            unsigned location = nir_intrinsic_io_semantics(intr).location +
                                nir_src_as_uint(intr->src[1]);
            for (unsigned c = 0; c < intr->def.num_components; ++c) {
               int index = apple9_varying_index(
                  varyings, location, nir_intrinsic_component(intr) + c);
               if (index < 0 || apple9_per_vertex_slot(&lower, index) >= 0)
                  continue;
               if (lower.per_vertex_count == AGX_APPLE9_MAX_PER_VERTEX) {
                  *reason = "too many Apple9 per-vertex fragment inputs";
                  goto fail;
               }
               lower.per_vertex[lower.per_vertex_count++] = index;
            }
         }
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
   if (lower.structured_cf)
      lower.entry_zero = lower.zero_vreg;
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

   /* Demotion and explicit depth/stencil export consume raster coverage.
    * Do not let ZS-only shaders use uninitialized coverage SSA indices. */
   const bool zs_export = nir->info.stage == MESA_SHADER_FRAGMENT &&
      (nir->info.outputs_written & (BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                    BITFIELD64_BIT(FRAG_RESULT_STENCIL)));
   lower.zs_export = zs_export;
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       (nir->info.fs.uses_discard || zs_export)) {
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
            lower.interlock |= op == nir_intrinsic_begin_invocation_interlock;
         }
      }
      if (lower.interlock && lower.tile_coords) {
         *reason = "Apple9 fragment interlock with tile coordinates is unsupported";
         goto fail;
      }
      if (reads || writes || image_mask || has_texture || lower.interlock ||
          nir->info.fs.uses_discard || zs_export) {
         if (!agx_apple9_vir_emit_side_effect(
                &lower.program, AGX_APPLE9_VIR_TILE_ACCESS,
                AGX_APPLE9_ENC_TILE_ACCESS, NULL, 0,
                image_mask ? 0x700 : lower.tile_coords ? 0xf00 : 0x600))
            goto fail;
         lower.tile_access = true;
      }
      if ((reads || writes || lower.interlock) && !lower.tile_coords) {
         if (!agx_apple9_vir_emit_side_effect(&lower.program,
                AGX_APPLE9_VIR_TILE_ACCESS, AGX_APPLE9_ENC_TILE_ACCESS,
                NULL, 0, writes || lower.interlock ? 0x80c : 0x808))
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
       (lower.color_stores || lower.tile_read || lower.interlock) &&
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

   if (!agx_apple9_optimize_vir_with_constants(&lower.program, constants)) {
      *reason = "out of memory optimizing Apple9 virtual IR";
      goto fail;
   }
   if (!agx_apple9_allocate_shared(&lower.program, nir, reason))
      goto fail;

   if (agx_apple9_trace_enabled()) {
      fprintf(apple9_log(), "APPLE9_ALLOC_STATS stage=%u peak_live_gprs=%u publications=%u scratch_bytes=%u spill_slots=%u\n",
              nir->info.stage, lower.program.peak_live_gprs,
              lower.program.publication_count, lower.program.scratch_size,
              lower.program.spill_slots);
      for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
         const struct agx_apple9_vir_instr *instruction =
            lower.program.instructions[i];
         fprintf(apple9_log(), "APPLE9_ALLOC i=%u op=%u enc=%u dst=", i,
                 instruction->op, instruction->encoding);
         if (instruction->dest == AGX_APPLE9_VREG_INVALID)
            fputs("-", apple9_log());
         else
            fprintf(apple9_log(), "%c%u",
                    lower.program.publication[instruction->dest] ? 'p' : 'r',
                    lower.program.phys[instruction->dest]);
         if (instruction->op == AGX_APPLE9_VIR_PHI_SRC)
            fprintf(apple9_log(), " target=r%u",
                    lower.program.phys[instruction->target]);
         fputs(" src=", apple9_log());
         for (unsigned s = 0; s < instruction->nr_srcs; ++s)
            fprintf(apple9_log(), "%s%c%u", s ? "," : "",
                    lower.program.publication[instruction->src[s]] ? 'p' : 'r',
                    lower.program.phys[instruction->src[s]]);
         fprintf(apple9_log(), " imm=%#x live=%#x slot=%u producer_slot=%u\n",
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
         if (agx_apple9_trace_enabled()) {
            fprintf(apple9_log(), "APPLE9_PACK_FAIL i=%u op=%u enc=%u dst=", i,
                    instruction->op, instruction->encoding);
            if (instruction->dest == AGX_APPLE9_VREG_INVALID)
               fputs("-", apple9_log());
            else
               fprintf(apple9_log(), "%c%u",
                       lower.program.publication[instruction->dest] ? 'p' : 'r',
                       lower.program.phys[instruction->dest]);
            fputs(" src=", apple9_log());
            for (unsigned s = 0; s < instruction->nr_srcs; ++s)
               fprintf(apple9_log(), "%s%c%u", s ? "," : "",
                       lower.program.publication[instruction->src[s]] ? 'p' : 'r',
                       lower.program.phys[instruction->src[s]]);
            fprintf(apple9_log(), " tex=%u samp=%u dim=%u fetch=%u ms=%u shadow=%u "
                    "imm=%u comps=%u mask=%x slot=%u\n",
                    instruction->texture_index, instruction->sampler_index,
                    instruction->texture_dimension, instruction->texture_fetch,
                    instruction->texture_multisampled,
                    instruction->texture_shadow, instruction->immediate,
                    instruction->dest_components,
                    instruction->texture_result_mask,
                    instruction->producer_scoreboard_slot);
         }
         if (*reason == NULL)
            *reason = "Apple9 DAG instruction pack failed";
         util_dynarray_fini(&emitter.bytes);
         free(vir_offsets);
         goto fail;
      }
      if (agx_apple9_trace_enabled()) {
         fprintf(apple9_log(), "APPLE9_VIR i=%u op=%u enc=%u dst=", i,
                 instruction->op, instruction->encoding);
         if (instruction->dest == AGX_APPLE9_VREG_INVALID)
            fputs("-", apple9_log());
         else
            fprintf(apple9_log(), "%c%u",
                    lower.program.publication[instruction->dest] ? 'p' : 'r',
                    lower.program.phys[instruction->dest]);
         fputs(" src=", apple9_log());
         for (unsigned s = 0; s < instruction->nr_srcs; ++s)
            fprintf(apple9_log(), "%s%c%u", s ? "," : "",
                    lower.program.publication[instruction->src[s]] ? 'p' : 'r',
                    lower.program.phys[instruction->src[s]]);
         fprintf(apple9_log(), " live=%#x slot=%u producer_slot=%u\n",
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
   if (debug_get_option_cached_agx_apple9_asm() != NULL) {
      fprintf(apple9_log(), "%s_ASM_BEGIN stage=%u\n",
              preamble ? "APPLE9_PREAMBLE" : "APPLE9", nir->info.stage);
      for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
         const struct agx_apple9_vir_instr *ins = lower.program.instructions[i];
         fprintf(apple9_log(), "APPLE9_ASM offset=%u op=%u enc=%u dst=",
                 vir_offsets[i], ins->op, ins->encoding);
         if (ins->dest == AGX_APPLE9_VREG_INVALID)
            fputs("-", apple9_log());
         else
            fprintf(apple9_log(), "%c%u", lower.program.publication[ins->dest] ? 'p' : 'r',
                    lower.program.phys[ins->dest]);
         fputs(" src=", apple9_log());
         for (unsigned j = 0; j < ins->nr_srcs; ++j)
            fprintf(apple9_log(), "%s%c%u", j ? "," : "",
                    lower.program.publication[ins->src[j]] ? 'p' : 'r',
                    lower.program.phys[ins->src[j]]);
         fprintf(apple9_log(), " imm=%#x wait=%u producer=%u bytes=", ins->immediate,
                 ins->scoreboard_slot, ins->producer_scoreboard_slot);
         for (unsigned j = vir_offsets[i]; j < vir_offsets[i + 1]; ++j)
            fprintf(apple9_log(), "%02x", ((const uint8_t *)emitter.bytes.data)[j]);
         fputc('\n', apple9_log());
      }
      fputs("APPLE9_ASM_END\n", apple9_log());
   }
   free(vir_offsets);
   if (agx_apple9_trace_enabled()) {
      fputs("APPLE9_BINARY ", apple9_log());
      const uint8_t *binary = emitter.bytes.data;
      for (unsigned i = 0; i < emitter.bytes.size; ++i)
         fprintf(apple9_log(), "%02x", binary[i]);
      fputc('\n', apple9_log());
   }
   apple9_emit_stop(&emitter);
   if (cost) {
      *cost = (struct apple9_dag_cost){
         .instructions = lower.program.instruction_count,
         .gprs = lower.program.peak_live_gprs,
      };
      for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
         enum agx_apple9_vir_opcode op = lower.program.instructions[i]->op;
         if (op == AGX_APPLE9_VIR_SPILL_STORE ||
             op == AGX_APPLE9_VIR_SPILL_LOAD) {
            const struct agx_apple9_block *block =
               agx_apple9_instr_block(lower.program.instructions[i]);
            unsigned depth = 0;
            for (nir_cf_node *node = block && block->nir
                                       ? block->nir->cf_node.parent : NULL;
                 node; node = node->parent)
               depth += node->type == nir_cf_node_loop;
            /* A loop-depth heuristic, not measured invocation counts. */
            cost->weighted_spills += 1u << (3 * MIN2(depth, 6));
            ++cost->spills;
         }
      }
   }
   if (debug_get_option_cached_agx_apple9_stats() != NULL) {
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
      fprintf(apple9_log(), "%s_CODEGEN stage=%u bytes=%u instructions=%u copies=%u waits=%u saves=%u fills=%u gprs=%u scratch=%u\n",
              preamble ? "APPLE9_PREAMBLE" : "APPLE9",
              nir->info.stage, emitter.bytes.size, lower.program.instruction_count,
              copies, waits, saves, fills, lower.program.peak_live_gprs,
              lower.program.scratch_size);
   }


   out->binary = emitter.bytes.data;
   out->info.stage = nir->info.stage;
   out->info.apple9_linear_mask = lower.linear_mask;
   out->info.apple9_reads_z = lower.reads_z;
   out->info.apple9_reads_primitive_id = lower.reads_primitive_id;
   out->info.depth_layout = lower.writes_z ? FRAG_DEPTH_LAYOUT_ANY
                                          : FRAG_DEPTH_LAYOUT_UNCHANGED;
   out->info.apple9_reads_point_coord = lower.reads_point_coord;
   out->info.apple9_barycentric_mask = lower.barycentric_mask;
   out->info.apple9_per_vertex_count = lower.per_vertex_count;
   memcpy(out->info.apple9_per_vertex, lower.per_vertex,
          sizeof(out->info.apple9_per_vertex));
   out->info.apple9_writes_point_size = nir->info.stage == MESA_SHADER_VERTEX &&
      (nir->info.outputs_written & BITFIELD64_BIT(VARYING_SLOT_PSIZ));
   out->info.writes_layer_viewport = nir->info.stage == MESA_SHADER_VERTEX &&
      (nir->info.outputs_written & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT));
   out->info.apple9_clip_distance_count = nir->info.stage == MESA_SHADER_VERTEX
      ? nir->info.clip_distance_array_size : 0;
   /* Interlocked fragments need the same ordered tile access as blending. */
   out->info.apple9_reads_tile = lower.tile_read || lower.interlock;
   out->info.apple9_flat_mask = lower.flat_mask;
   out->info.apple9_texture_mask = texture_mask;
   out->info.apple9_image_mask = image_mask;
   out->info.apple9_sampler_mask = sampler_mask;
   out->info.apple9_uses_texel_fetch = uses_texel_fetch;
   out->info.apple9_uses_discard = nir->info.stage == MESA_SHADER_FRAGMENT &&
                                    (nir->info.fs.uses_discard || lower.writes_z ||
                                     lower.writes_s);
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
                                  lower.reads_primitive_id + 2 * lower.reads_point_coord +
                                  2 * util_bitcount(lower.barycentric_mask) +
                                  lower.per_vertex_count;
   out->info.main_size = emitter.bytes.size;
   out->info.binary_size = emitter.bytes.size;
   if (debug_get_option_cached_agx_apple9_dump_binary() != NULL) {
      fprintf(apple9_log(), "APPLE9_BINARY stage=%u size=%u ", nir->info.stage,
              (unsigned)emitter.bytes.size);
      for (unsigned i = 0; i < emitter.bytes.size; ++i)
         fprintf(apple9_log(), "%02x", ((const uint8_t *)emitter.bytes.data)[i]);
      fprintf(apple9_log(), "\n");
   }
   /* nr_gprs is the architectural GPR high-water mark for Apple9.  Keep the
    * established minimum tier, but do not collapse every program using r16+
    * back to the old 16-register capture tier. */
   out->info.nr_gprs = MAX2(lower.program.max_phys_gpr + 1, 8);
   out->info.scratch_size = lower.program.scratch_size;
   out->info.local_size = nir->info.shared_size;
   out->info.apple9_publication_count = lower.program.publication_count;
   for (unsigned i = 0; i < lower.program.instruction_count; ++i) {
      const struct agx_apple9_vir_instr *ins = lower.program.instructions[i];
      if (ins->op == AGX_APPLE9_VIR_DEVICE_ATOMIC && !ins->atomic_discard)
         out->info.apple9_atomic_frame_size = 4;
   }
   out->info.stats.instrs = emitter.instructions;
   for (unsigned d = 0; d < 3; ++d)
      out->info.workgroup_size[d] = nir->info.workgroup_size[d];

   if (profile != NULL) {
      const unsigned resource_count = resource_map.count;
      *profile = AGX_APPLE9_DIRECT_BUFFERS_COMPUTE_PROFILE;
      if (resources->descriptor_tables)
         profile->abi = AGX_APPLE9_COMPUTE_ABI_DESCRIPTOR_TABLES;
      profile->atomic_frame_size = out->info.apple9_atomic_frame_size;
      profile->resource_binding_count = resource_count;
      profile->resource_read_mask = resource_map.read_mask;
      profile->resource_write_mask = resource_map.write_mask;
      profile->writes_global = resource_map.writes_global;
      profile->required_threadgroup_memory_bytes = nir->info.shared_size
         ? util_next_power_of_two(MAX2(nir->info.shared_size, 128)) : 0;
      for (unsigned i = 0; i < profile->resource_binding_count; ++i) {
         profile->resource_binding[i] = resource_map.resource[i].binding;
         profile->resource_kind[i] = resource_map.resource[i].kind;
      }
      profile->scratch_size = lower.program.scratch_size;
      profile->publication_count = lower.program.publication_count;
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

static bool
apple9_has_samples(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (apple9_hoist_sample(instr))
               return true;
         }
      }
   }
   return false;
}

/* Compile with the dumps buffered in *log. */
static bool
apple9_compile_dag_trial(nir_shader *nir, struct agx_shader_part *out,
                         struct agx_apple9_compute_profile *profile,
                         const struct agx_apple9_varying_layout *varyings,
                         const struct apple9_buffer_map *resources,
                         const struct agx_apple9_uniform_constants *constants,
                         bool hoist,
                         struct apple9_dag_cost *cost, char **log,
                         size_t *log_size, const char **reason)
{
   FILE *stream = open_memstream(log, log_size);
   FILE *previous = apple9_log_stream;
   apple9_log_stream = stream;
   bool ok = apple9_compile_dag_once(nir, out, profile, varyings, resources,
                                     constants, false, hoist, cost, reason);
   apple9_log_stream = previous;
   if (stream)
      fclose(stream);
   return ok;
}

static bool
apple9_dag_cost_less(const struct apple9_dag_cost *a,
                     const struct apple9_dag_cost *b)
{
   return a->weighted_spills < b->weighted_spills ||
          (a->weighted_spills == b->weighted_spills &&
           (a->spills < b->spills ||
            (a->spills == b->spills && a->instructions < b->instructions)));
}

/* Hoisting samples lengthens live ranges. In a shader that already fills the
 * register file, the extra spills and copies can cost more than the latency
 * hidden, so compile such shaders both ways and keep the result with fewer
 * loop-weighted spills, then fewer static spills and instructions.
 */
static bool
apple9_compile_dag_body(nir_shader *nir, struct agx_shader_part *out,
                       struct agx_apple9_compute_profile *profile,
                       const struct agx_apple9_varying_layout *varyings,
                       const struct apple9_buffer_map *resources,
                       const struct agx_apple9_uniform_constants *constants,
                       bool preamble, struct apple9_dag_cost *cost,
                       const char **reason)
{
   nir_shader *plain = NULL;
   if (preamble || agx_apple9_ra_is_probe() || !apple9_hoist_limit() ||
       !apple9_has_samples(nir) || !(plain = nir_shader_clone(NULL, nir)))
      return apple9_compile_dag_once(nir, out, profile, varyings, resources,
                                     constants, preamble, true, cost, reason);

   struct apple9_dag_cost hoisted_cost = {0}, plain_cost = {0};
   char *hoisted_log = NULL, *plain_log = NULL;
   size_t hoisted_size = 0, plain_size = 0;
   bool ok = apple9_compile_dag_trial(nir, out, profile, varyings, resources,
                                      constants, true, &hoisted_cost, &hoisted_log,
                                      &hoisted_size, reason);
   bool use_plain = false;
   if (!ok || hoisted_cost.spills || hoisted_cost.gprs >= AGX_APPLE9_GPR_COUNT) {
      struct agx_shader_part plain_out = {0};
      struct agx_apple9_compute_profile plain_profile = {0};
      const char *plain_reason = NULL;
      bool plain_ok = apple9_compile_dag_trial(
         plain, &plain_out, profile ? &plain_profile : NULL, varyings,
         resources, constants, false, &plain_cost, &plain_log, &plain_size,
         &plain_reason);
      use_plain = plain_ok &&
                  (!ok || apple9_dag_cost_less(&plain_cost, &hoisted_cost));
      if (use_plain) {
         if (ok)
            free(out->binary);
         *out = plain_out;
         if (profile)
            *profile = plain_profile;
         ok = true;
      } else {
         free(plain_out.binary);
      }
   }
   const char *log = use_plain ? plain_log : hoisted_log;
   size_t size = use_plain ? plain_size : hoisted_size;
   if (log && size)
      fwrite(log, 1, size, apple9_log());
   if (ok && cost)
      *cost = use_plain ? plain_cost : hoisted_cost;
   free(hoisted_log);
   free(plain_log);
   ralloc_free(plain);
   return ok;
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
   /* The setup program is straight-line code, but nir_opt_preamble keeps
    * the control flow around work it hoists. Hoisting under control flow,
    * such as a counter address behind a uniform query test, would therefore
    * leave the whole shader without a preamble. */
   if (instr->block->cf_node.parent->type != nir_cf_node_function)
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
   case nir_instr_type_intrinsic: {
      /* Global reads of memory the shader cannot write, such as Honeykrisp
       * descriptors, keep buffer addresses in uniforms like Metal's arguments. */
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic((nir_instr *)instr);
      accepted = (intr->intrinsic == nir_intrinsic_load_ubo ||
                  (intr->intrinsic == nir_intrinsic_load_global_2x32 &&
                   (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER))) &&
                 nir_foreach_src((nir_instr *)instr, apple9_preamble_source, (void *)data);
      break;
   }
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
   /* Even-aligned tuples let a stored pointer serve as a memory instruction's
    * uniform base pair. */
   *size = def->num_components;
   *align = def->num_components > 1 ? 2 : 1;
   *class_ = nir_preamble_class_general;
}

/* Constants an ALU instruction can read from a uniform word. The backend
 * folds such preamble reads into uniform operands; a constant left in the
 * main program costs a mov_imm32 and a register, and above r63 a copy too.
 */
static bool
apple9_uniform_constant_use(nir_alu_instr *alu, unsigned s, uint32_t *value)
{
   bool is_float, comparison = false;
   switch (alu->op) {
   case nir_op_flt: case nir_op_fge: case nir_op_feq: case nir_op_fneu:
   case nir_op_ilt: case nir_op_ige: case nir_op_ieq: case nir_op_ine:
   case nir_op_ult: case nir_op_uge:
      comparison = true;
      /* Select/predicate sources have no folded float-immediate form yet. */
      is_float = false;
      break;
   case nir_op_ishl: case nir_op_ishr: case nir_op_ushr:
      /* Constant shift amounts already have an immediate encoding. */
      if (s != 0)
         return false;
      is_float = false;
      break;
   case nir_op_bcsel:
      /* The condition is a boolean; only the selected 32-bit values use
       * these validated uniform operands. */
      if (s == 0)
         return false;
      is_float = false;
      break;
   case nir_op_fadd: case nir_op_fmul: case nir_op_ffma:
   case nir_op_fmin: case nir_op_fmax:
      is_float = true;
      break;
   case nir_op_ffloor: case nir_op_fceil: case nir_op_ftrunc:
   case nir_op_fround_even: case nir_op_frsq: case nir_op_fexp2: case nir_op_flog2:
   case nir_op_bit_count: case nir_op_bitfield_reverse: case nir_op_ufind_msb:
   case nir_op_u2f32: case nir_op_i2f32: case nir_op_f2u32: case nir_op_f2i32:
   case nir_op_iadd: case nir_op_iand: case nir_op_ior: case nir_op_ixor:
   case nir_op_imin: case nir_op_imax: case nir_op_umin: case nir_op_umax:
      is_float = false;
      break;
   default:
      return false;
   }
   if (alu->def.num_components != 1 ||
       (alu->def.bit_size != 32 && !comparison) ||
       alu->src[s].src.ssa->bit_size != 32 ||
       !nir_src_is_const(alu->src[s].src))
      return false;
   *value = nir_src_comp_as_uint(alu->src[s].src, alu->src[s].swizzle[0]);
   /* Low-bit masks select as bitfield extracts, which need no operand. */
   if (alu->op == nir_op_iand && *value && *value != UINT32_MAX &&
       !(*value & (*value + 1)))
      return false;
   /* Packed float immediates are already free. */
   uint8_t encoded;
   if (is_float) {
      /* Floating min/max operands, FADD/FMUL operands and the FMA addend
       * also take a fourth mantissa bit (min/max commute to operand B). */
      bool low;
      bool wide = alu->op != nir_op_ffma || s == 2;
      return !(wide ? agx_apple9_encode_float_immediate_wide(*value, &encoded, &low)
                    : agx_apple9_encode_float_immediate(*value, &encoded));
   }
   /* So are 8-bit literals of integer add/subtract, logic and selected
    * values; a small negative addend is subtracted. */
   /* Compare operands take a literal as well (either side, by reversal). */
   if (comparison && !agx_apple9_compare_literals_disabled()) {
      bool low;
      bool float_compare = alu->op == nir_op_flt || alu->op == nir_op_fge ||
                           alu->op == nir_op_feq || alu->op == nir_op_fneu;
      return float_compare
                ? !agx_apple9_encode_float_immediate_wide(*value, &encoded, &low)
                : *value > UINT8_MAX;
   }
   switch (alu->op) {
   case nir_op_iadd:
      return *value > UINT8_MAX && -*value > UINT8_MAX;
   case nir_op_iand: case nir_op_ior: case nir_op_ixor: case nir_op_bcsel:
      return *value > UINT8_MAX;
   default:
      return true;
   }
}

struct apple9_constant_count {
   uint32_t value;
   unsigned uses;
};

static int
apple9_constant_count_cmp(const void *a, const void *b)
{
   const struct apple9_constant_count *x = a, *y = b;
   if (x->uses != y->uses)
      return x->uses < y->uses ? 1 : -1;
   return x->value < y->value ? -1 : x->value > y->value;
}

/* Store the most-used constants in free uniform words. Keep their literal
 * NIR values through instruction selection: opaque preamble loads would hide
 * address offsets, integer folds and source modifiers. Native optimization
 * uses this table to select uniforms for the remaining constant operands. */
static void
apple9_promote_constants(nir_shader *shader, nir_function_impl *preamble,
                         unsigned *used, unsigned budget,
                         struct agx_apple9_uniform_constants *constants)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   struct hash_table_u64 *counts = _mesa_hash_table_u64_create(NULL);
   struct util_dynarray list = UTIL_DYNARRAY_INIT;

   nir_foreach_block(block, impl) {
      unsigned depth = 0;
      for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent)
         depth += node->type == nir_cf_node_loop;
      /* Approximate repeated execution without assuming a known trip count.
       * Cap nesting and saturate the sum so large shaders cannot wrap it. */
      unsigned weight = 1u << (3 * MIN2(depth, 6));
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs; ++s) {
            uint32_t value;
            if (!apple9_uniform_constant_use(alu, s, &value))
               continue;
            uintptr_t index = (uintptr_t)_mesa_hash_table_u64_search(counts, value);
            if (!index) {
               util_dynarray_append(&list, ((struct apple9_constant_count){value, 0}));
               index = util_dynarray_num_elements(&list, struct apple9_constant_count);
               _mesa_hash_table_u64_insert(counts, value, (void *)index);
            }
            struct apple9_constant_count *entry =
               util_dynarray_element(&list, struct apple9_constant_count, index - 1);
            entry->uses += MIN2(weight, UINT_MAX - entry->uses);
         }
      }
   }

   unsigned count = util_dynarray_num_elements(&list, struct apple9_constant_count);
   struct apple9_constant_count *sorted = list.data;
   if (count)
      qsort(sorted, count, sizeof(*sorted), apple9_constant_count_cmp);

   /* A single use outside loops gains nothing: it trades its mov_imm32 for
    * a preamble store and spends a uniform word. Loop uses carry their
    * execution weight, allowing a lone static use to be promoted. */
   constants->base = *used;
   constants->count = 0;
   nir_builder pb = nir_builder_at(nir_after_impl(preamble));
   /* nir_opt_preamble has already reserved its result words. Spend only the
    * remaining words; the completed setup is checked against the compiler's
    * preamble code-size bound before it is attached to the main program. */
   for (unsigned i = 0; i < count && *used < budget &&
                        sorted[i].uses >= 2; ++i) {
      nir_store_preamble(&pb, nir_imm_int(&pb, sorted[i].value), .base = *used,
                         .preamble_class = nir_preamble_class_general);
      constants->values[constants->count++] = sorted[i].value;
      ++*used;
   }

   _mesa_hash_table_u64_destroy(counts);
   util_dynarray_fini(&list);
}

/* Index of the last instruction of a loop (instructions are indexed in
 * program order, so it is the largest in the loop). */
static unsigned
apple9_loop_end_index(nir_loop *loop)
{
   unsigned last = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_instr *instr = nir_block_last_instr(block);
      if (instr)
         last = MAX2(last, instr->index);
   }
   return last;
}

/* Whether a CF node lies inside the loop. */
static bool
apple9_cf_inside(nir_cf_node *node, nir_loop *loop)
{
   for (; node; node = node->parent) {
      if (node == &loop->cf_node)
         return true;
   }
   return false;
}

/* A value left in the main program holds a register from its definition to
 * its last use. When the uniform words run out, that span, not only the
 * instructions it takes to compute, is what a uniform word saves: long-lived
 * buffer bases and invariant values otherwise occupy GPRs across the whole
 * program and force spills. */
static unsigned
apple9_main_span(nir_instr *instr)
{
   nir_def *def = nir_instr_def(instr);
   if (!def)
      return 0;
   unsigned last = instr->index;
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src))
         continue;
      nir_instr *use = nir_src_use_instr(src);
      last = MAX2(last, use->index);
      /* A use inside a loop that does not contain the definition keeps the
       * value live to the end of that (outermost such) loop, except a phi in
       * the loop header, whose own value takes over at loop entry. */
      for (nir_cf_node *node = use->block->cf_node.parent; node;
           node = node->parent) {
         if (node->type != nir_cf_node_loop)
            continue;
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (use->type == nir_instr_type_phi &&
             use->block == nir_loop_first_block(loop))
            continue;
         if (!apple9_cf_inside(&instr->block->cf_node, loop))
            last = MAX2(last, apple9_loop_end_index(loop));
      }
   }
   return last - instr->index;
}

static float
apple9_preamble_cost(nir_instr *instr, const void *data)
{
   if (!apple9_preamble_instruction(instr, data))
      return 0;
   float cost = instr->type == nir_instr_type_intrinsic ? 8 : 1;
   /* One cost unit per 25 main-program instructions of span; measured on
    * ER's draw-time fragments (spills 4,349 -> 1,396). */
   cost += (float)apple9_main_span(instr) / 25;
   return cost;
}

static float
apple9_preamble_rewrite_cost(nir_def *def, const void *unused)
{
   return def->num_components;
}

/* Texture and sampler handles are descriptor offsets from the USC aperture
 * base, which the stage's texture and sampler table pointers name when
 * handles are used. Convert them to the native table indices read by the
 * texture unit, once all descriptor metadata has been loaded through them.
 * A stage with register-indexed samplers reaches the private texel-fetch
 * sampler the same way.
 */
/* Split an iadd tree into its non-constant terms (NULL when none) and the sum
 * of its constant terms. */
static nir_def *
apple9_split_addend(nir_builder *b, nir_scalar s, uint32_t *constant,
                    unsigned depth)
{
   if (nir_scalar_is_const(s)) {
      *constant += nir_scalar_as_uint(s);
      return NULL;
   }
   if (depth && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      nir_scalar x = nir_scalar_chase_alu_src(s, 0);
      nir_scalar y = nir_scalar_chase_alu_src(s, 1);
      uint32_t before = *constant;
      nir_def *xs = apple9_split_addend(b, x, constant, depth - 1);
      nir_def *ys = apple9_split_addend(b, y, constant, depth - 1);
      if (*constant == before && xs && ys)
         return nir_channel(b, s.def, s.comp);
      if (!xs || !ys)
         return xs ? xs : ys;
      return nir_iadd(b, xs, ys);
   }
   return nir_channel(b, s.def, s.comp);
}

/* Handles are byte offsets into the descriptor aperture, which never wraps,
 * so the multiple-of-32 part of a handle's constant terms moves only the
 * entry index. Fold it into the entry immediate: handles that differ by a
 * record-relative offset (the image and sampler parts of one combined
 * descriptor, planes, array elements at constant indices) then share one
 * shift. */
static nir_def *
apple9_descriptor_entry(nir_builder *b, nir_def *handle, unsigned entry)
{
   uint32_t constant = 0;
   nir_def *x = apple9_split_addend(b, nir_get_scalar(handle, 0), &constant, 4);
   int32_t entries = (int32_t)constant >> 5;
   uint32_t rest = constant & 31;
   if (!x)
      return nir_imm_int(b, entries + entry);
   if (rest)
      x = nir_iadd_imm(b, x, rest);
   return nir_iadd_imm(b, nir_ushr_imm(b, x, 5), entries + entry);
}

static bool
apple9_lower_texture_handles(nir_shader *nir)
{
   bool samplers = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex)
            samplers |= nir_tex_instr_src_index(nir_instr_as_tex(instr),
                                                nir_tex_src_sampler_handle) >= 0;
      }
   }

   bool progress = false;
   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
         if (texture < 0 || (tex->backend_flags & AGX_APPLE9_TEXTURE_HANDLE_INDEX))
            continue;
         b.cursor = nir_before_instr(instr);
         nir_src_rewrite(&tex->src[texture].src,
            apple9_descriptor_entry(&b, tex->src[texture].src.ssa,
                                    AGX_APPLE9_DESCRIPTOR_TEXTURE_ENTRY));
         int sampler = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
         if (sampler >= 0) {
            nir_src_rewrite(&tex->src[sampler].src,
               apple9_descriptor_entry(&b, tex->src[sampler].src.ssa,
                                       AGX_APPLE9_DESCRIPTOR_SAMPLER_ENTRY));
         } else if (samplers && apple9_texture_uses_fetch_sampler(tex)) {
            nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
               nir_load_ubo(&b, 1, 32,
                  nir_imm_int(&b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
                  nir_imm_int(&b, AGX_APPLE9_FETCH_SAMPLER_INDEX_OFFSET),
                  .align_mul = 4, .range = 4));
         }
         tex->backend_flags |= AGX_APPLE9_TEXTURE_HANDLE_INDEX;
         progress = true;
      }
   }
   return nir_progress(progress, nir_shader_get_entrypoint(nir),
                       nir_metadata_control_flow);
}

/* Read-only descriptor loads and uniform buffer-size limits can execute
 * before their original branch. Move only speculatable uniform dependency
 * chains, then let preamble
 * extraction choose which values are worth storing. Ordinary guarded buffer
 * reads lack CAN_SPECULATE and must remain guarded.
 */
struct apple9_uniform_hoist {
   nir_block *entry;
   uint8_t *state;
   nir_cursor cursor;
   struct hash_table *original_blocks;
   /* Whether apple9_preamble_safe_read reads may leave their branches. */
   bool safe_reads;
   /* Whether descriptor memory reads alone may (stages without safe_reads),
    * with the constant driver-UBO reads that form their handles. */
   bool descriptor_reads;
   bool descriptor_chain;
};

static bool
apple9_is_bounds_limit(nir_instr *instr)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_iadd || alu->def.bit_size != 32 ||
       alu->def.num_components != 1)
      return false;

   nir_scalar root = nir_get_scalar(&alu->def, 0);
   for (unsigned s = 0; s < 2; ++s) {
      uint32_t addend, minimum;
      nir_scalar max = nir_scalar_chase_alu_src(root, 1 - s);
      if (!apple9_const_u32(nir_scalar_chase_alu_src(root, s), &addend) ||
          !addend || !nir_scalar_is_alu(max) ||
          nir_scalar_alu_op(max) != nir_op_umax)
         continue;
      for (unsigned m = 0; m < 2; ++m) {
         if (apple9_const_u32(nir_scalar_chase_alu_src(max, m), &minimum) &&
             minimum == -addend)
            return true;
      }
   }
   return false;
}

/* Pressure gating (AGX_APPLE9_GATED) compiles with and without LICM and the
 * preamble-only reads; -1 defers to the environment. */
static _Thread_local int apple9_safe_reads_override = -1;

/* Reads that are always safe once per dispatch at a uniform address, but
 * that must not be generally speculatable: if-conversion or code motion
 * would then run them for every thread on paths that skipped them.
 * Driver UBOs (app UBOs are global by now), descriptor memory in the USC
 * aperture and reads the driver marks ACCESS_UNIFORM_SAFE_AGX qualify. */
/* Reads of descriptor memory: constant, read-only records in the USC
 * aperture, at a uniform address. */
static bool
apple9_descriptor_read(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_global_2x32 ||
       !(nir_intrinsic_access(intr) & ACCESS_CAN_REORDER))
      return false;
   nir_scalar hi = nir_scalar_chase_movs(nir_get_scalar(intr->src[0].ssa, 1));
   return nir_scalar_is_const(hi) &&
          nir_scalar_as_uint(hi) == (AGX_APPLE9_USC_BASE >> 32);
}

static bool
apple9_preamble_safe_read(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic == nir_intrinsic_load_ubo)
      return nir_src_is_const(intr->src[0]) && nir_src_is_const(intr->src[1]);
   if (intr->intrinsic != nir_intrinsic_load_global_2x32 ||
       !(nir_intrinsic_access(intr) & ACCESS_CAN_REORDER))
      return false;
   if (nir_intrinsic_access(intr) & ACCESS_UNIFORM_SAFE_AGX)
      return true;
   nir_scalar hi = nir_scalar_chase_movs(nir_get_scalar(intr->src[0].ssa, 1));
   return nir_scalar_is_const(hi) &&
          nir_scalar_as_uint(hi) == (AGX_APPLE9_USC_BASE >> 32);
}

/* The first block of a loop at function level runs once whenever the shader
 * reaches the loop: no break or continue can precede it. A uniform value it
 * computes from read-only memory is the same on every iteration, so it can
 * move to the entry like a function-level block. DXIL's single-trip
 * "loop { ...; break; }" early-exit wrapper puts whole shaders there. */
static bool
apple9_block_runs_once_per_loop_entry(nir_block *block)
{
   nir_cf_node *parent = block->cf_node.parent;
   return parent->type == nir_cf_node_loop &&
          parent->parent->type == nir_cf_node_function &&
          block == nir_loop_first_block(nir_cf_node_as_loop(parent));
}

static bool
apple9_can_hoist_uniform(nir_src *src, void *data)
{
   struct apple9_uniform_hoist *s = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   uint8_t *state = &s->state[src->ssa->index];
   if (*state)
      return *state == 2;
   *state = 1;
   if (instr->block != s->entry && !nir_instr_can_speculate(instr) &&
       !apple9_block_runs_once_per_loop_entry(instr->block) &&
       !(s->safe_reads && apple9_preamble_safe_read(instr)) &&
       !(s->descriptor_reads && apple9_descriptor_read(instr)) &&
       !(s->descriptor_chain && apple9_preamble_safe_read(instr) &&
         nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_ubo))
      return false;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_load_ubo &&
          !(intr->intrinsic == nir_intrinsic_load_global_2x32 &&
            (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER)))
         return false;
   } else if (instr->type != nir_instr_type_alu &&
              instr->type != nir_instr_type_load_const) {
      return false;
   }
   if (!nir_foreach_src(instr, apple9_can_hoist_uniform, s))
      return false;
   *state = 2;
   return true;
}

static bool
apple9_hoist_uniform_source(nir_src *src, void *data)
{
   struct apple9_uniform_hoist *s = data;
   nir_instr *instr = nir_def_instr(src->ssa);
   if (instr->block == s->entry)
      return true;
   nir_foreach_src(instr, apple9_hoist_uniform_source, s);
   if (s->original_blocks)
      _mesa_hash_table_insert(s->original_blocks, instr, instr->block);
   nir_instr_move(s->cursor, instr);
   s->cursor = nir_after_instr(instr);
   return true;
}

static bool
apple9_hoist_uniform_reads(nir_shader *nir, struct hash_table *original_blocks,
                            bool addresses, unsigned *address_hoists)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   /* Preamble-only reads are on for compute by default. In a large fragment
    * shader they raised register pressure. AGX_APPLE9_SAFE_HOIST_STAGES
    * selects =compute, =all or =none; hk chooses which reads qualify. */
   const char *env = debug_get_option_cached_agx_apple9_safe_hoist_stages();
   const char *stages = apple9_safe_reads_override > 0 ? "all"
                        : apple9_safe_reads_override == 0 ? NULL
                        : env ? env : "compute";
   bool compute = nir->info.stage == MESA_SHADER_COMPUTE ||
                  nir->info.stage == MESA_SHADER_KERNEL;
   struct apple9_uniform_hoist s = {
      .entry = nir_start_block(impl),
      .original_blocks = original_blocks,
      .safe_reads = stages && (!strcmp(stages, "all") ||
                               (!strcmp(stages, "compute") && compute)),
      /* Descriptor records read through handles (texture info, runtime
       * sampler state) are hoisted in every stage: unlike buffer reads,
       * leaving them in their branches did not save registers. */
      .descriptor_reads = !debug_get_option_cached_agx_apple9_no_descriptor_hoist(),
      .state = calloc(MAX2(impl->ssa_alloc, 1), 1),
      .cursor = nir_after_block_before_jump(nir_start_block(impl)),
   };
   if (!s.state)
      return false;
   bool progress = false;
   /* Sampler state of conditionally executed samples (packed LOD/bias words,
    * descriptor handles) is uniform: offer its whole chain, including the
    * conversions and packing, to preamble extraction. Unselected
    * instructions are restored. Like address hoists, this placement is
    * compared against the original one when it spills or its setup does not
    * fit. */
   const bool alu_seeds = addresses && original_blocks &&
                          !debug_get_option_cached_agx_apple9_no_sampler_hoist();
   if (alu_seeds) {
      nir_foreach_block(block, impl) {
         if (block == s.entry)
            continue;
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            for (unsigned i = 0; i < tex->num_srcs; ++i) {
               switch (tex->src[i].src_type) {
               case nir_tex_src_coord: case nir_tex_src_ddx: case nir_tex_src_ddy:
               case nir_tex_src_comparator: case nir_tex_src_projector:
                  continue;
               default:
                  break;
               }
               nir_instr *source = nir_def_instr(tex->src[i].src.ssa);
               if (source->block == s.entry || source->type != nir_instr_type_alu)
                  continue;
               /* Constant driver-UBO reads (descriptor-set and root
                * pointers) may leave their branches with it. */
               s.descriptor_chain = s.descriptor_reads;
               bool hoist = apple9_can_hoist_uniform(&tex->src[i].src, &s);
               s.descriptor_chain = false;
               if (hoist) {
                  apple9_hoist_uniform_source(&tex->src[i].src, &s);
                  progress = true;
                  if (address_hoists)
                     ++*address_hoists;
               }
            }
         }
      }
   }
   nir_foreach_block(block, impl) {
      if (block == s.entry)
         continue;
      nir_foreach_instr_safe(instr, block) {
         /* In a run-once loop head, offer whole uniform expressions (the
          * bounds selects of the reads), not just the reads. */
         if (!apple9_is_bounds_limit(instr) &&
             !(instr->type == nir_instr_type_intrinsic &&
               nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_global_2x32) &&
             !(instr->type == nir_instr_type_alu &&
               apple9_block_runs_once_per_loop_entry(block)))
            continue;
         nir_src src = nir_src_for_ssa(nir_instr_def(instr));
         s.descriptor_chain = s.descriptor_reads && apple9_descriptor_read(instr);
         bool hoist = apple9_can_hoist_uniform(&src, &s);
         s.descriptor_chain = false;
         if (hoist) {
            apple9_hoist_uniform_source(&src, &s);
            progress = true;
         } else if (addresses && instr->type == nir_instr_type_intrinsic &&
                    nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_global_2x32) {
            /* The guarded read itself may not speculate, while its uniform
             * address can. Hoist that pure dependency chain so preamble
             * extraction can select bounds checks and redirected pointers
             * without moving the memory access across its condition. */
            nir_src *address = &nir_instr_as_intrinsic(instr)->src[0];
            if (nir_def_instr(address->ssa)->block != s.entry &&
                apple9_can_hoist_uniform(address, &s)) {
               apple9_hoist_uniform_source(address, &s);
               progress = true;
               if (address_hoists)
                  ++*address_hoists;
            }
         }
      }
   }
   free(s.state);
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

bool
agx_nir_hoist_apple9_uniform_reads(nir_shader *nir)
{
   return apple9_hoist_uniform_reads(nir, NULL, true, NULL);
}

/* Exposing an address to preamble extraction must not leave every unselected
 * expression live from shader entry. Restore surviving hoisted instructions
 * toward their original blocks. CSE may have shared them with other branches,
 * so include all uses when choosing a dominating block. Reverse traversal
 * moves consumers before sources, preserving their order after the phis. */
static void
apple9_restore_unselected_hoists(nir_shader *nir, struct hash_table *original_blocks)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_dominance |
                              nir_metadata_dominance_lca);
   nir_block *entry = nir_start_block(impl);
   bool progress = false;
   nir_foreach_instr_reverse_safe(instr, entry) {
      /* Extraction creates load_preamble intrinsics and frees selected
       * definitions. Do not associate new allocations with old pointer keys. */
      if (instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_preamble)
         continue;
      struct hash_entry *found = _mesa_hash_table_search(original_blocks, instr);
      if (!found)
         continue;
      nir_def *def = nir_instr_def(instr);
      nir_block *target = found->data;
      nir_foreach_use_including_if(use, def)
         target = nir_dominance_lca(target, nir_src_get_block(use));
      if (target != entry) {
         nir_instr_move(nir_after_phis(target), instr);
         progress = true;
      }
   }
   nir_progress(progress, impl, nir_metadata_control_flow);
}

/* The main program every preamble attempt starts from: the attempts of both
 * placements share it, so it is prepared once per compile. */
static _Thread_local nir_shader *apple9_prepared_main;

static nir_shader *
apple9_prepare_preamble_main(nir_shader *nir)
{
   if (apple9_prepared_main)
      return nir_shader_clone(NULL, apple9_prepared_main);
   nir_shader *main = nir_shader_clone(NULL, nir);
   if (!main)
      return NULL;
   /* Avoid spending uniform words on unused components of frontend UBO
    * vectors. Removing leading components can also reduce the alignment:
    * vec4 at byte 16 may become vec3 at byte 20. Re-legalize that access
    * before selecting its memory format or extracting a preamble. */
   nir_opt_shrink_vectors(main, true);
   nir_opt_constant_folding(main);
   apple9_legalize_buffer_accesses(main);
   return main;
}

static bool
apple9_try_preamble(nir_shader *nir, struct agx_shader_part *out,
                   struct agx_apple9_compute_profile *profile,
                   const struct agx_apple9_varying_layout *varyings,
                   const struct apple9_buffer_map *resources,
                   unsigned uniform_budget, bool promote_constants,
                   bool hoist_addresses, struct apple9_dag_cost *cost,
                   unsigned *preamble_bytes, unsigned *preamble_words,
                   unsigned *address_hoists)
{
   nir_shader *main = apple9_prepare_preamble_main(nir);
   if (!main)
      return false;
   struct hash_table *original_blocks = _mesa_pointer_hash_table_create(NULL);
   if (!original_blocks) {
      ralloc_free(main);
      return false;
   }
   /* Remember where every hoisted read came from, in either trial, so a read
    * the preamble does not take returns to the branch that needed it rather
    * than running unconditionally for every thread. */
   if (apple9_hoist_uniform_reads(main, original_blocks,
                                  hoist_addresses, address_hoists))
      nir_opt_cse(main);
   nir_function_impl *main_impl = nir_shader_get_entrypoint(main);
   nir_index_ssa_defs(main_impl);
   struct apple9_preamble_eligibility eligibility = {
      .state = calloc(MAX2(main_impl->ssa_alloc, 1), sizeof(uint8_t)),
   };
   if (!eligibility.state) {
      _mesa_hash_table_destroy(original_blocks, NULL);
      ralloc_free(main);
      return false;
   }
   const nir_opt_preamble_options options = {
      .cb_data = &eligibility,
      .def_size = apple9_preamble_def_size,
      .preamble_storage_size[nir_preamble_class_general] =
         uniform_budget,
      .instr_cost_cb = apple9_preamble_cost,
      .rewrite_cost_cb = apple9_preamble_rewrite_cost,
      .avoid_instr_cb = apple9_preamble_avoid,
   };
   unsigned sizes[nir_preamble_num_classes] = {0};
   struct agx_shader_part setup = {0}, candidate = {0};
   struct agx_apple9_uniform_constants constants = {0};
   struct agx_apple9_compute_profile candidate_profile = {0};
   const char *candidate_reason = NULL;
   nir_shader *preamble = NULL;
   if (main)
      nir_index_instrs(nir_shader_get_entrypoint(main));
   if (main && nir_opt_preamble(main, &options, sizes)) {
      apple9_restore_unselected_hoists(main, original_blocks);
      nir_function_impl *impl = nir_shader_get_preamble(main);
      /* Setup currently admits ordinary linear ALU/UBO work. Masked setup and
       * scratch have independent execution contracts and are not assumed. */
      if (impl && !apple9_cf_list_has_control_flow(&impl->body)) {
         if (promote_constants)
            apple9_promote_constants(main, impl,
                                     &sizes[nir_preamble_class_general],
                                     options.preamble_storage_size[nir_preamble_class_general],
                                     &constants);
         constants.base += resources->descriptor_tables
            ? AGX_APPLE9_GRAPHICS_ROOT_WORDS
            : agx_apple9_compute_preamble_base(resources->count);
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
         if (apple9_compile_dag_body(preamble, &setup, NULL, NULL, resources,
                                    NULL, true, NULL, &candidate_reason) &&
             !setup.info.scratch_size &&
             setup.info.main_size > 4 &&
             setup.info.main_size <=
                agx_apple9_max_preamble_bytes(nir->info.stage == MESA_SHADER_COMPUTE) &&
             apple9_compile_dag_body(main, &candidate,
                                    profile ? &candidate_profile : NULL,
                                    varyings, resources, &constants, false, cost, &candidate_reason)) {
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
               _mesa_hash_table_destroy(original_blocks, NULL);
               free(eligibility.state);
               free(setup.binary);
               ralloc_free(preamble);
               ralloc_free(main);
               return true;
            }
         }
      }
   }
   if (agx_apple9_trace_enabled() && candidate_reason)
      fprintf(stderr, "APPLE9_PREAMBLE_FALLBACK %s\n", candidate_reason);
   *preamble_bytes = setup.info.main_size;
   *preamble_words = sizes[nir_preamble_class_general];
   free(setup.binary);
   free(candidate.binary);
   _mesa_hash_table_destroy(original_blocks, NULL);
   free(eligibility.state);
   ralloc_free(preamble);
   ralloc_free(main);
   return false;
}

static bool
apple9_preamble_trial(nir_shader *nir, struct agx_shader_part *out,
                     struct agx_apple9_compute_profile *profile,
                     const struct agx_apple9_varying_layout *varyings,
                     const struct apple9_buffer_map *resources,
                     bool hoist_addresses, struct apple9_dag_cost *cost,
                     char **log, size_t *log_size, unsigned *address_hoists)
{
   FILE *stream = open_memstream(log, log_size);
   FILE *previous = apple9_log_stream;
   apple9_log_stream = stream;
   bool ok = false;
   unsigned budget = AGX_APPLE9_UNIFORM_COUNT -
      (resources->descriptor_tables ? AGX_APPLE9_GRAPHICS_ROOT_WORDS
                                   : agx_apple9_compute_preamble_base(resources->count));
   bool promote_constants = !debug_get_option_cached_agx_apple9_no_promote();
   /* Constant stores must not discard an otherwise useful preamble when
    * they cross the code-size policy bound. Retry without them first.
    * If the extracted expressions alone exceed the bound, reduce their
    * uniform budget using the measured setup size. Every attempt starts
    * from the same prepared NIR and still checks the assembled code size.
    * Bound retries so pathological setup code cannot multiply compile time.
    */
   const unsigned max_bytes =
      agx_apple9_max_preamble_bytes(nir->info.stage == MESA_SHADER_COMPUTE);
   for (unsigned attempt = 0; attempt < 4 && budget; ++attempt) {
      unsigned bytes = 0, words = 0;
      if (apple9_try_preamble(nir, out, profile, varyings, resources, budget,
                             promote_constants, hoist_addresses, cost, &bytes, &words,
                             address_hoists)) {
         ok = true;
         break;
      }
      if (bytes <= max_bytes)
         break;
      if (promote_constants) {
         promote_constants = false;
      } else {
         budget = (uint64_t)MIN2(budget, words) *
                  max_bytes / bytes;
         /* Shared setup dependencies make code size nonlinear in result
          * words. Reserve the last retry for a substantial reduction if
          * the proportional estimate has not fitted the setup yet. */
         if (attempt == 2)
            budget = MIN2(budget, words / 2);
      }
   }
   apple9_log_stream = previous;
   if (stream)
      fclose(stream);
   return ok;
}

static bool
apple9_licm_filter(nir_instr *instr, nir_loop *loop, bool dominates_exit)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
      return true;
   case nir_instr_type_intrinsic:
   case nir_instr_type_tex:
      /* Hoisting memory or texture work out of a branch runs it on every
       * iteration of every thread, which costs far more than it saves in
       * per-pixel loops (Lethal Company lighting). Hoist only from blocks
       * every iteration executes; AGX_APPLE9_LICM_SPECULATE allows leaving
       * branches for speculatable accesses. */
      return dominates_exit ||
             (nir_instr_can_speculate(instr) &&
              debug_get_option_cached_agx_apple9_licm_speculate());
   default:
      return false;
   }
}

static bool
apple9_compile_dag_tail(nir_shader *nir, struct agx_shader_part *out,
                        struct agx_apple9_compute_profile *profile,
                        const struct agx_apple9_varying_layout *varyings,
                        struct apple9_dag_cost *chosen_cost, const char **reason);

static bool
apple9_compile_dag(nir_shader *nir, struct agx_shader_part *out,
                  struct agx_apple9_compute_profile *profile,
                  const struct agx_apple9_varying_layout *varyings,
                  const char **reason)
{
   /* Preamble extraction runs divergence analysis, which requires generic
    * sink pointers to have been lowered to this driver's concrete addresses.
    * In particular, writes to the read-only zero page use the scratch page.
    */
   agx_nir_lower_sink_address(nir);
   apple9_lower_texture_handles(nir);
   /* As in agx_optimize_nir, reassociate before forming the preamble:
    * ordering chains by rank exposes uniform subexpressions to it and lets
    * CSE share common pairs. Buffer addresses are still 64-bit here, so the
    * later 2x32 split sees the carry chains it expects. The loop runs
    * algebraic cleanup, so give it this backend's options: reordered
    * shifted-byte chains otherwise become pack_32_4x8, which Apple9 lacks.
    */
   {
      const nir_shader_compiler_options *saved = nir->options;
      nir_shader_compiler_options options = *saved;
      options.lower_pack_unorm_2x16 = false;
      options.lower_pack_unorm_4x8 = false;
      options.lower_unpack_unorm_2x16 = false;
      options.lower_unpack_unorm_4x8 = false;
      options.lower_unpack_snorm_2x16 = false;
      options.lower_unpack_snorm_4x8 = false;
      options.lower_bitfield_extract = true;
      options.has_bfe = false;
      options.has_pack_32_4x8 = false;
      nir->options = &options;
      nir_opt_reassociate_loop(
         nir, nir_reassociate_scalar_math | nir_reassociate_cse_heuristic);
      nir->options = saved;
   }
   apple9_prepare_dag(nir);
   /* Loads and samples cost far more than the registers that hold their
    * loop-invariant results. Hoist them, and their address math, out of
    * loops before preamble selection, which can then also take uniform
    * ones. */
   /* LICM keeps hoisted values live across the whole loop. In shaders at
    * the register limit that costs far more than it saves (Lethal Company's
    * hot fragment shader quintupled its spills), so it is opt-in until it
    * has a register-pressure model. AGX_APPLE9_LICM=1 enables it. */
   if (debug_get_option_cached_agx_apple9_gated()) {
      /* Compile the aggressive variant (LICM and preamble-only reads) and the
       * plain one, and keep the cheaper allocation: both raise register
       * pressure, which can spill shaders already at the limit. */
      nir_shader *plain = nir_shader_clone(NULL, nir);
      struct agx_shader_part aggressive_out = {0}, plain_out = {0};
      struct agx_apple9_compute_profile aggressive_profile = {0}, plain_profile = {0};
      struct apple9_dag_cost aggressive_cost = {0}, plain_cost = {0};
      bool licm = nir_opt_licm(nir, apple9_licm_filter);
      if (licm) {
         nir_opt_cse(nir);
         nir_opt_dce(nir);
      }
      apple9_safe_reads_override = 1;
      bool aggressive_ok = apple9_compile_dag_tail(
         nir, &aggressive_out, profile ? &aggressive_profile : NULL, varyings,
         &aggressive_cost, reason);
      apple9_safe_reads_override = 0;
      bool plain_ok = apple9_compile_dag_tail(
         plain, &plain_out, profile ? &plain_profile : NULL, varyings,
         &plain_cost, reason);
      apple9_safe_reads_override = -1;
      ralloc_free(plain);
      /* Loop-weighted spills decide: spills outside loops are cheaper than
       * the loads and instructions hoisting removes inside them. */
      bool use_plain = plain_ok &&
         (!aggressive_ok || apple9_dag_cost_less(&plain_cost, &aggressive_cost));
      if (agx_apple9_trace_enabled())
         fprintf(stderr, "APPLE9_GATED chose=%s aggressive=%llu/%u plain=%llu/%u\n",
                 use_plain ? "plain" : "aggressive",
                 (unsigned long long)aggressive_cost.weighted_spills, aggressive_cost.spills,
                 (unsigned long long)plain_cost.weighted_spills, plain_cost.spills);
      if (use_plain) {
         free(aggressive_out.binary);
         *out = plain_out;
         if (profile)
            *profile = plain_profile;
         return true;
      }
      free(plain_out.binary);
      if (!aggressive_ok)
         return false;
      *out = aggressive_out;
      if (profile)
         *profile = aggressive_profile;
      return true;
   }
   if (debug_get_option_cached_agx_apple9_licm() && nir_opt_licm(nir, apple9_licm_filter)) {
      nir_opt_cse(nir);
      nir_opt_dce(nir);
   }
   struct apple9_dag_cost cost = {0};
   return apple9_compile_dag_tail(nir, out, profile, varyings, &cost, reason);
}

static bool
apple9_compile_dag_tail(nir_shader *nir, struct agx_shader_part *out,
                        struct agx_apple9_compute_profile *profile,
                        const struct agx_apple9_varying_layout *varyings,
                        struct apple9_dag_cost *chosen_cost, const char **reason)
{
   struct apple9_buffer_map resources = {0};
   if (!apple9_collect_buffer_map(nir, &resources, reason))
      return false;

   struct apple9_dag_cost address_cost = {0}, original_cost = {0};
   char *address_log = NULL, *original_log = NULL;
   size_t address_size = 0, original_size = 0;
   unsigned address_hoists = 0;
   apple9_prepared_main = NULL;
   nir_shader *prepared = apple9_prepare_preamble_main(nir);
   apple9_prepared_main = prepared;
   /* Moving uniform addresses competes with descriptor values and constants
    * for preamble storage. If the resulting main spills, compare the old
    * placement using the same loop-weighted allocation cost as sample
    * hoisting, while also rejecting any static spill increase. Preserve a
    * usable original preamble if new setup does not fit. The common no-spill
    * case needs only one search. Without any address hoist the original
    * placement prepares the same main program, so its search could only
    * repeat this one.
    *
    * When the address placement spills, the original placement nearly always
    * spills less (ELDEN RING: 288 of 290 fragment shaders), so the address
    * trial's full candidate search is usually wasted. Probe the address
    * placement with one allocation; if that spills, compile the original
    * placement first. Keep it outright when it does not spill or clearly
    * spills less than the probe; otherwise complete the address trial and
    * compare as before. */
   struct agx_shader_part original = {0};
   struct agx_apple9_compute_profile original_profile = {0};
   bool have_original = false, original_ok = false, ok = false;
   bool use_original = false, address_done = false;
   {
      struct agx_shader_part probe = {0};
      struct apple9_dag_cost probe_cost = {0};
      char *probe_log = NULL;
      size_t probe_size = 0;
      unsigned probe_hoists = 0;
      agx_apple9_ra_set_probe(true);
      bool probe_ok = apple9_preamble_trial(nir, &probe, NULL, varyings, &resources,
                                            true, &probe_cost, &probe_log,
                                            &probe_size, &probe_hoists);
      agx_apple9_ra_set_probe(false);
      free(probe.binary);
      free(probe_log);
      if (probe_ok && probe_cost.spills && probe_hoists) {
         have_original = true;
         original_ok = apple9_preamble_trial(
            nir, &original, profile ? &original_profile : NULL, varyings,
            &resources, false, &original_cost, &original_log, &original_size, NULL);
         if (original_ok &&
             (!original_cost.spills ||
              original_cost.weighted_spills * 10 <= probe_cost.weighted_spills * 9)) {
            use_original = true;
            address_done = true;
         }
      }
   }
   if (!address_done) {
      ok = apple9_preamble_trial(nir, out, profile, varyings, &resources,
                                 true, &address_cost, &address_log, &address_size,
                                 &address_hoists);
      if (!ok || (address_cost.spills && address_hoists)) {
         if (!have_original) {
            original_ok = apple9_preamble_trial(
               nir, &original, profile ? &original_profile : NULL, varyings,
               &resources, false, &original_cost, &original_log, &original_size,
               NULL);
         }
         use_original = original_ok &&
                        (!ok || address_cost.spills > original_cost.spills ||
                         apple9_dag_cost_less(&original_cost, &address_cost));
      }
   }
   if (use_original) {
      if (ok)
         free(out->binary);
      *out = original;
      if (profile)
         *profile = original_profile;
      ok = true;
   } else {
      free(original.binary);
   }
   apple9_prepared_main = NULL;
   ralloc_free(prepared);
   *chosen_cost = use_original ? original_cost : address_cost;
   const char *log = use_original ? original_log : address_log;
   size_t size = use_original ? original_size : address_size;
   if (log && size)
      fwrite(log, 1, size, apple9_log());
   free(address_log);
   free(original_log);
   if (ok)
      return true;

   return apple9_compile_dag_body(nir, out, profile, varyings, &resources,
                                 NULL, false, NULL, reason);
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
   /* Use the native shifts and integer logic until a bitfield instruction
    * is represented by the Apple9 machine model. */
   options.lower_bitfield_extract = true;
   options.has_bfe = false;
   nir->options = &options;
   bool progress = nir_opt_algebraic(nir);
   nir->options = saved;
   return progress;
}

static unsigned
apple9_narrow_divide_width(const nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_alu)
      return 0;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   switch (alu->op) {
   case nir_op_idiv:
   case nir_op_udiv:
   case nir_op_imod:
   case nir_op_umod:
   case nir_op_irem:
      return alu->def.bit_size < 32 ? 32 : 0;
   default:
      return 0;
   }
}

static void
apple9_lower_idiv(nir_shader *nir)
{
   /* Narrow quotients use the FP32 quotient/refinement sequence on
    * extended operands, which is exact for 8- and 16-bit ranges.
    */
   nir_lower_bit_size(nir, apple9_narrow_divide_width, NULL);
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

static bool
apple9_lower_subgroup_filter(const nir_intrinsic_instr *intr,
                             UNUSED const void *data)
{
   /* Keep scans with a native implementation. */
   return !((intr->intrinsic == nir_intrinsic_exclusive_scan ||
             intr->intrinsic == nir_intrinsic_inclusive_scan) &&
            intr->def.bit_size == 32 && intr->def.num_components == 1 &&
            nir_intrinsic_reduction_op(intr) == nir_op_iadd);
}

static bool
apple9_lower_compute_subgroups(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_workgroup_size &&
       !b->shader->info.workgroup_size_variable) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_ivec3(b,
         b->shader->info.workgroup_size[0], b->shader->info.workgroup_size[1],
         b->shader->info.workgroup_size[2]));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_quad_ballot_agx) {
      /* Extract this quad's four bits from a full ballot. */
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *ballot = nir_ballot(b, 1, 32, intr->src[0].ssa);
      nir_def *shift = nir_iand_imm(b, nir_load_subgroup_invocation(b), ~3);
      nir_def *quad = nir_iand_imm(b, nir_ushr(b, ballot, shift), 0xf);
      nir_def_replace(&intr->def, nir_u2uN(b, quad, intr->def.bit_size));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_active_subgroup_count_agx) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *count =
         nir_bit_count(b, nir_ballot(b, 1, 32, nir_imm_true(b)));
      nir_def_replace(&intr->def, nir_u2uN(b, count, intr->def.bit_size));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_active_subgroup_invocation_agx) {
      /* Count the active lanes below this one. */
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *active = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def *below = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1),
                                    nir_load_subgroup_invocation(b)), -1);
      nir_def *count = nir_bit_count(b, nir_iand(b, active, below));
      nir_def_replace(&intr->def, nir_u2uN(b, count, intr->def.bit_size));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_shuffle ||
       (intr->intrinsic == nir_intrinsic_read_invocation &&
        intr->def.bit_size == 1)) {
      b->cursor = nir_before_instr(&intr->instr);
      /* The native shuffle moves 32-bit registers. */
      nir_def *data = intr->src[0].ssa;
      if (data->bit_size == 1)
         data = nir_b2i32(b, data);
      nir_def *target = intr->src[1].ssa;
      nir_def *result;
      if (intr->intrinsic == nir_intrinsic_read_invocation) {
         result = nir_read_invocation(b, data, target);
      } else {
         /* As on Apple8, the hardware shuffle uses a single low 2-bit index
          * for each quad. Read each index within the target quad with a
          * quad-uniform index and select the requested one.
          */
         nir_def *quad_start = nir_iand_imm(b, target, 0x1c);
         result = NULL;
         for (unsigned i = 0; i < 4; ++i) {
            nir_def *target_i = nir_iadd_imm(b, quad_start, i);
            nir_def *value = nir_read_invocation(b, data, target_i);
            result = result ? nir_bcsel(b, nir_ieq(b, target, target_i),
                                        value, result)
                            : value;
         }
      }
      if (intr->def.bit_size == 1)
         result = nir_ine_imm(b, result, 0);
      nir_def_replace(&intr->def, result);
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_num_subgroups)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *size = b->shader->info.workgroup_size_variable
      ? nir_load_workgroup_size(b)
      : nir_imm_ivec3(b, b->shader->info.workgroup_size[0],
                        b->shader->info.workgroup_size[1],
                        b->shader->info.workgroup_size[2]);
   nir_def *threads = nir_imul(b, nir_channel(b, size, 0),
                              nir_imul(b, nir_channel(b, size, 1),
                                          nir_channel(b, size, 2)));
   nir_def_replace(&intr->def, nir_ushr_imm(b, nir_iadd_imm(b, threads, 31), 5));
   return true;
}

static void apple9_lower_textures(nir_shader *nir);

/* Cross-lane operations move whole 32-bit registers. */
static unsigned
apple9_subgroup_width(const nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return 0;
   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      return intr->def.bit_size > 1 && intr->def.bit_size < 32 ? 32 : 0;
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
      return intr->src[0].ssa->bit_size > 1 && intr->src[0].ssa->bit_size < 32 ? 32 : 0;
   default:
      return 0;
   }
}

static void
apple9_lower_subgroups(nir_shader *nir)
{
   agx_nir_lower_apple9_subgroup_denorms(nir);
   nir_lower_bit_size(nir, apple9_subgroup_width, NULL);
   /* Subgroup operations are expressed through the native ballot and
    * simd_shuffle, which takes a per-lane source index. 32-bit integer
    * addition scans have a native instruction and are kept.
    */
   const nir_lower_subgroups_options subgroups = {
      .filter = apple9_lower_subgroup_filter,
      .subgroup_size = 32,
      .ballot_bit_size = 32,
      .ballot_components = 1,
      .lower_to_scalar = true,
      .lower_vote = true,
      .lower_vote_feq = true,
      .lower_vote_ieq = true,
      .lower_vote_bool_eq = true,
      .lower_first_invocation_to_ballot = true,
      .lower_read_first_invocation = true,
      .lower_subgroup_masks = true,
      .lower_elect = true,
      .lower_inverse_ballot = true,
      .lower_relative_shuffle = true,
      .lower_quad = true,
      .lower_quad_vote = true,
      .lower_rotate_to_shuffle = true,
      .lower_rotate_clustered_to_shuffle = true,
      .lower_reduce = true,
      .lower_shuffle_to_32bit = true,
   };
   /* Reduction lowering emits relative shuffles; lower those in turn. */
   nir_lower_subgroups(nir, &subgroups);
   nir_lower_subgroups(nir, &subgroups);
   nir_shader_intrinsics_pass(nir, apple9_lower_compute_subgroups,
                              nir_metadata_control_flow, NULL);
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

   agx_nir_lower_apple9_compute_derivatives(nir);
   apple9_lower_subgroups(nir);

   /* Internal compute shaders may use function temporaries just like API
    * frontends. Normalize them to the SSA form consumed by this backend. */
   agx_nir_lower_printf_buffer(nir);
   if (nir->scratch_size <= 128)
      nir_lower_scratch_to_var(nir);
   nir_lower_array_deref_of_vec(nir, nir_var_function_temp, NULL,
                               nir_lower_indirect_array_deref_of_vec_load |
                               nir_lower_indirect_array_deref_of_vec_store);
   nir_lower_vars_to_ssa(nir);
   nir_lower_vars_to_scratch(nir, 128, glsl_get_natural_size_align_bytes,
                              glsl_get_word_size_align_bytes);
   nir_lower_indirect_derefs_to_if_else_trees(nir, nir_var_function_temp, ~0u);
   nir_lower_vars_to_ssa(nir);
   nir_remove_dead_variables(nir, nir_var_mem_shared, NULL);
   nir_lower_vars_to_explicit_types(nir, nir_var_mem_shared,
                                    glsl_get_natural_size_align_bytes);
   nir_lower_explicit_io(nir, nir_var_mem_shared, nir_address_format_32bit_offset);
   bool progress;
   do {
      progress = nir_opt_constant_folding(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_remove_phis(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);
   apple9_lower_textures(nir);
   apple9_lower_idiv(nir);
   agx_nir_opt_apple9_bounds(nir);
   agx_nir_lower_apple9_math(nir);
   if (agx_nir_lower_apple9_compute_derivatives(nir))
      apple9_lower_subgroups(nir);
   nir_lower_alu_to_scalar(nir, NULL, NULL);
   nir_lower_all_phis_to_scalar(nir);

   if (debug_get_option_cached_agx_apple9_trace_nir() != NULL)
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
   if (debug_get_option_cached_agx_apple9_trace_nir() != NULL)
      fprintf(stderr, "Apple9 compute compilation failed: %s\n",
              reason ? reason : "unknown reason");
   return false;
}

static unsigned
apple9_io_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

struct apple9_color_lower {
   const struct agx_apple9_blend *blend;
   nir_def *src1;
   unsigned nr_targets;
   bool valid;
};

/* Specialize the common sample lowering to the raster state in this variant.
 * The position word is shared with PPP programming, preserving sample order. */
static bool
apple9_lower_sample_state(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_blend *blend = data;
   unsigned samples = blend ? MAX2(blend->samples, 1) : 1;
   unsigned raster_samples = blend && blend->multisample_disabled ? 1 : samples;
   uint32_t value;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_sample_positions_agx:
      if (blend && blend->dynamic_sample_positions && raster_samples > 1) {
         b->cursor = nir_before_instr(&intr->instr);
         nir_def_replace(&intr->def, nir_load_ubo(b, 1, 32,
            nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
            nir_imm_int(b, AGX_APPLE9_SAMPLE_POSITIONS_OFFSET),
            .align_mul = 4, .range = 4));
         return true;
      }
      value = agx_default_sample_positions(raster_samples);
      break;
   case nir_intrinsic_load_samples_log2_agx:
      value = util_logbase2(raster_samples);
      break;
   case nir_intrinsic_load_api_sample_mask_agx:
      value = BITFIELD_MASK(samples) & ~(blend ? blend->disabled_samples : 0);
      break;
   default:
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, intr->def.bit_size));
   return true;
}

/* This backend emits complete fragment programs, so the common coverage pass
 * always finalizes depth/stencil tests in this shader part. */
/* No native interpolation position form takes an arbitrary offset. Evaluate
 * the plane at the pixel center and step along its fine screen-space
 * derivatives. This is exact for linear interpolation and a first-order
 * approximation for perspective-correct interpolation.
 */
static bool
apple9_lower_interp_at_offset(nir_builder *b, nir_intrinsic_instr *intr,
                              UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;
   nir_intrinsic_instr *bary = nir_src_as_intrinsic(intr->src[0]);
   if (!bary || bary->intrinsic != nir_intrinsic_load_barycentric_at_offset)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *offset = bary->src[0].ssa;
   nir_def *center = nir_load_barycentric_pixel(
      b, 32, .interp_mode = nir_intrinsic_interp_mode(bary));
   nir_src_rewrite(&intr->src[0], center);

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *value = &intr->def;
   nir_def *result = nir_ffma(b, nir_ddy_fine(b, value),
                              nir_channel(b, offset, 1),
                              nir_ffma(b, nir_ddx_fine(b, value),
                                       nir_channel(b, offset, 0), value));
   nir_def_rewrite_uses_after(value, result);
   return true;
}

/* Barycentric coordinates use the same interpolation positions as varyings.
 * Single-sampled shaders interpolate at the pixel centre, sample
 * interpolation becomes interpolation at the sample ID, and offsets step
 * along the screen-space derivatives as for varyings. Only the first two
 * weights are interpolated; the third is the remainder.
 */
static bool
apple9_lower_barycentric_coord(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (!apple9_is_barycentric_coord(intr->intrinsic))
      return false;

   const bool single_sampled = *(const bool *)data;
   const enum glsl_interp_mode mode = nir_intrinsic_interp_mode(intr);
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *coord;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_barycentric_coord_at_offset: {
      nir_def *offset = intr->src[0].ssa;
      nir_def *center =
         nir_load_barycentric_coord_pixel(b, 32, .interp_mode = mode);
      coord = nir_ffma(b, nir_ddy_fine(b, center), nir_channel(b, offset, 1),
                       nir_ffma(b, nir_ddx_fine(b, center),
                                nir_channel(b, offset, 0), center));
      break;
   }
   case nir_intrinsic_load_barycentric_coord_centroid:
   case nir_intrinsic_load_barycentric_coord_sample:
   case nir_intrinsic_load_barycentric_coord_at_sample:
      if (single_sampled) {
         coord = nir_load_barycentric_coord_pixel(b, 32, .interp_mode = mode);
      } else if (intr->intrinsic == nir_intrinsic_load_barycentric_coord_sample) {
         coord = nir_load_barycentric_coord_at_sample(
            b, 32, nir_load_sample_id(b), .interp_mode = mode);
      } else {
         return false;
      }
      break;
   default:
      return false;
   }
   nir_def_replace(&intr->def, coord);
   return true;
}

static bool
apple9_expand_barycentric_coord(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   if (!apple9_is_barycentric_coord(intr->intrinsic) ||
       intr->def.num_components != 3)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *x = nir_channel(b, &intr->def, 0);
   nir_def *y = nir_channel(b, &intr->def, 1);
   nir_def *z = nir_fsub_imm(b, 1.0, nir_fadd(b, x, y));
   nir_def *vec = nir_vec3(b, x, y, z);
   nir_def_rewrite_uses_after(&intr->def, vec);
   return true;
}

static bool
apple9_lower_coverage_sysval(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* Run after single-sample lowering, which can introduce helper queries
    * while simplifying sample-mask inputs.
    */
   if (intr->intrinsic == nir_intrinsic_load_helper_invocation) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_build_lowered_load_helper_invocation(b));
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_rewrite_uses(&intr->def, nir_imm_intN_t(b, ~0u, intr->def.bit_size));
   nir_instr_remove(&intr->instr);
   return true;
}

/* Match the common AGX flat interpolation contract: a first-provoking
 * triangle fan consumes B, while ordinary flat modes consume C. Keep this
 * choice in draw sysvals so dynamic topology does not specialize the shader. */
static bool
apple9_lower_flat_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_varying_layout *varyings = data;
   if (intr->intrinsic != nir_intrinsic_load_input ||
       (nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PRIMITIVE_ID &&
        !varyings->mask[VARYING_SLOT_PRIMITIVE_ID]))
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *first_fan = nir_load_ubo(b, 1, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_FIRST_FAN_OFFSET),
      .align_mul = 4, .range_base = AGX_APPLE9_FIRST_FAN_OFFSET, .range = 4);
   first_fan = nir_ine_imm(b, first_fan, 0);
   nir_def *values[4];
   unsigned read = nir_def_components_read(&intr->def);
   for (unsigned c = 0; c < intr->def.num_components; ++c) {
      if (!(read & BITFIELD_BIT(c))) {
         values[c] = nir_undef(b, 1, 32);
         continue;
      }
      nir_def *cf = nir_load_coefficients_agx(b, intr->src[0].ssa,
         .component = nir_intrinsic_component(intr) + c,
         .interp_mode = INTERP_MODE_FLAT,
         .io_semantics = nir_intrinsic_io_semantics(intr));
      values[c] = nir_bcsel(b, first_fan, nir_channel(b, cf, 1),
                            nir_channel(b, cf, 2));
   }
   nir_def_replace(&intr->def, nir_vec(b, values, intr->def.num_components));
   return true;
}

static bool
apple9_lower_fragment_uniform(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_line_stipple_agx ||
       intr->intrinsic == nir_intrinsic_load_viewport_scale ||
       intr->intrinsic == nir_intrinsic_load_viewport_offset) {
      b->cursor = nir_before_instr(&intr->instr);
      unsigned offset =
         intr->intrinsic == nir_intrinsic_load_line_stipple_agx
            ? AGX_APPLE9_LINE_STIPPLE_OFFSET
         : intr->intrinsic == nir_intrinsic_load_viewport_scale
            ? AGX_APPLE9_VIEWPORT_SCALE_OFFSET
            : AGX_APPLE9_VIEWPORT_OFFSET_OFFSET;
      unsigned n = intr->def.num_components;
      nir_def *value = nir_load_ubo(b, n, 32,
         nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, offset), .align_mul = 16, .range_base = offset,
         .range = n * 4);
      nir_def_replace(&intr->def, value);
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_blend_const_color_rgba &&
       intr->intrinsic != nir_intrinsic_load_polygon_stipple_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   bool stipple = intr->intrinsic == nir_intrinsic_load_polygon_stipple_agx;
   nir_def *offset = stipple
      ? nir_iadd_imm(b, nir_ishl_imm(b, intr->src[0].ssa, 2),
                    AGX_APPLE9_POLYGON_STIPPLE_OFFSET)
      : nir_imm_int(b, 0);
   nir_def *value = nir_load_ubo(b, stipple ? 1 : 4, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING), offset,
      .align_mul = stipple ? 4 : 16,
      .range_base = stipple ? AGX_APPLE9_POLYGON_STIPPLE_OFFSET : 0,
      .range = stipple ? 128 : 16);
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Rectangle textures share the ordinary 2D descriptor. The common texture
 * pass normalizes their coordinates; explicit gradients need the same scale.
 * Integer fetches use level zero without any coordinate normalization. */
static bool
apple9_lower_rectangle(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_RECT)
      return false;

   b->cursor = nir_before_instr(instr);
   if (tex->op == nir_texop_txf) {
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      if (nir_tex_instr_src_index(tex, nir_tex_src_lod) < 0 &&
          nir_tex_instr_src_index(tex, nir_tex_src_backend1) < 0)
         nir_tex_instr_add_src(tex, nir_tex_src_lod, nir_imm_int(b, 0));
      return true;
   }

   if (tex->op != nir_texop_txd)
      return false;

   nir_def *scale = nir_frcp(b, nir_i2f32(b, nir_get_texture_size(b, tex)));
   for (unsigned i = 0; i < tex->num_srcs; ++i) {
      if (tex->src[i].src_type == nir_tex_src_ddx ||
          tex->src[i].src_type == nir_tex_src_ddy)
         nir_src_rewrite(&tex->src[i].src,
                        nir_fmul(b, tex->src[i].src.ssa, scale));
   }
   return true;
}

/* Compute has no implicit texture-unit footprint. Derive it from the declared
 * derivative group, using the same spatial coordinates as explicit gradients.
 * Array layers and shadow comparisons never participate in the footprint. */
static bool
apple9_lower_compute_lod_query(nir_builder *b, nir_instr *instr, void *data)
{
   if (b->shader->info.stage != MESA_SHADER_COMPUTE ||
       b->shader->info.derivative_group != DERIVATIVE_GROUP_LINEAR ||
       instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_lod)
      return false;

   b->cursor = nir_before_instr(instr);
   unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_1D ? 1 :
                   tex->sampler_dim == GLSL_SAMPLER_DIM_3D ||
                   tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? 3 : 2;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *p = nir_trim_vector(b, tex->src[coord].src.ssa, dims);
   nir_def *dx = nir_ddx(b, p), *dy = nir_ddy(b, p);
   bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   if (cube) {
      /* Project each derivative onto the selected face with the quotient
       * rule. Signs of the face axes do not affect the footprint length. */
      nir_def *a = nir_fabs(b, p);
      nir_def *x = nir_channel(b, a, 0), *y = nir_channel(b, a, 1);
      nir_def *z = nir_channel(b, a, 2);
      nir_def *use_z = nir_fge(b, z, nir_fmax(b, x, y));
      nir_def *use_y = nir_fge(b, y, nir_fmax(b, x, z));
      unsigned xzy[] = {0, 2, 1}, yzx[] = {1, 2, 0};
      nir_def *values[] = {p, dx, dy};
      for (unsigned i = 0; i < 3; ++i)
         values[i] = nir_bcsel(b, use_z, values[i],
            nir_bcsel(b, use_y, nir_swizzle(b, values[i], xzy, 3),
                                 nir_swizzle(b, values[i], yzx, 3)));
      nir_def *rcp = nir_frcp(b, nir_channel(b, values[0], 2));
      nir_def *uv = nir_fmul(b, nir_trim_vector(b, values[0], 2), rcp);
      for (unsigned i = 1; i < 3; ++i)
         values[i] = nir_fmul(b, nir_fmul_imm(b, rcp, 0.5),
            nir_fsub(b, nir_trim_vector(b, values[i], 2),
                        nir_fmul(b, uv, nir_channel(b, values[i], 2))));
      dx = values[1];
      dy = values[2];
      dims = 2;
   }
   nir_def *size[3];
   for (unsigned c = 0; c < dims; ++c)
      size[c] = nir_u2f32(b, agx_apple9_texture_info(b, tex, cube ? 0 : c));
   nir_def *extent = nir_vec(b, size, dims);
   dx = nir_fmul(b, dx, extent);
   dy = nir_fmul(b, dy, extent);
   nir_def *xx = nir_fdot(b, dx, dx), *yy = nir_fdot(b, dy, dy);
   nir_def *major = nir_fmax(b, xx, yy);
   /* Native 2D anisotropic sampling uses footprint area divided by its
    * longest derivative. This also handles nonorthogonal and rank-one
    * footprints; the shorter derivative's length would overestimate them.
    * Native 3D sampling is isotropic. Cube gradients are lowered to scalar
    * LODs by the common texture pass. */
   if (!cube && tex->sampler_dim != GLSL_SAMPLER_DIM_3D) {
      nir_def *area = dims == 1 ? nir_imm_float(b, 0) :
         nir_fsub(b, nir_fmul(b, nir_channel(b, dx, 0), nir_channel(b, dy, 1)),
                     nir_fmul(b, nir_channel(b, dx, 1), nir_channel(b, dy, 0)));
      nir_def *minor = nir_bcsel(b, nir_flt(b, nir_imm_float(b, 0), major),
         nir_fdiv(b, nir_fmul(b, area, area), major), nir_imm_float(b, 0));
      nir_def *aniso = agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_MAX_ANISOTROPY);
      aniso = nir_fmax(b, aniso, nir_imm_float(b, 1.0));
      major = nir_fmax(b, minor, nir_fdiv(b, major, nir_fmul(b, aniso, aniso)));
   }
   nir_def *lod = nir_fmul_imm(b, nir_flog2(b, major), 0.5);
   int bias_src = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   nir_def *bias = bias_src >= 0 ? nir_f2f32(b, tex->src[bias_src].src.ssa) :
                                 nir_imm_float(b, 0);
   if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) < 0) {
      nir_def *state_bias = nir_load_ubo(b, 1, 32,
         nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_APPLE9_SAMPLER_BIAS_OFFSET + tex->sampler_index * 4),
         .align_mul = 4, .range = 4);
      bias = nir_fadd(b, bias, state_bias);
   }
   /* Honeykrisp advertises maxSamplerLodBias = 15. */
   bias = nir_fclamp(b, bias, nir_imm_float(b, -15), nir_imm_float(b, 15));
   lod = nir_fadd(b, lod, bias);
   nir_def *level = nir_fclamp(b, lod,
      agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_MIN_LOD),
      agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_MAX_LOD));
   nir_def *mip = agx_apple9_sampler_field(b, tex, AGX_APPLE9_SAMP_MIP);
   nir_def *levels = agx_apple9_texture_info(b, tex, 3);
   level = nir_fclamp(b, level, nir_imm_float(b, 0),
                        nir_u2f32(b, nir_iadd_imm(b, levels, -1)));
   int view_min = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (view_min >= 0)
      level = nir_fmax(b, level, nir_f2f32(b, tex->src[view_min].src.ssa));
   level = nir_bcsel(b, nir_ieq_imm(b, mip, PIPE_TEX_MIPFILTER_NEAREST),
      nir_fadd_imm(b, nir_fceil(b, nir_fadd_imm(b, level, 0.5)), -1.0), level);
   level = nir_bcsel(b, nir_ieq_imm(b, mip, PIPE_TEX_MIPFILTER_NONE),
                       nir_imm_float(b, 0), level);
   nir_def *result = nir_bcsel(b, nir_ieq_imm(b, levels, 0),
      nir_imm_vec2(b, 0, 0), nir_vec2(b, level, lod));
   nir_def_rewrite_uses(&tex->def, result);
   nir_instr_remove(instr);
   return true;
}

/* LOD queries use the texture unit's derivative and clamp machinery. The
 * hardware returns continuous LODs; GL additionally applies mip selection to
 * the first component. Bias goes through the normal sample-bias operand. */
static bool
apple9_lower_texture_lod_query(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_lod)
      return false;
   /* LOD queries have spatial coordinates only, even for array/shadow
    * samplers. Supply the native array operand without reading a nonexistent
    * NIR component; comparison does not participate in a LOD query. */
   if (tex->is_array && (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
                         tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)) {
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (coord < 0)
         return false;
      b->cursor = nir_before_instr(instr);
      nir_def *coords = tex->src[coord].src.ssa;
      unsigned spatial = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? 3 : 2;
      nir_src_rewrite(&tex->src[coord].src,
         nir_pad_vector_imm_int(b, nir_trim_vector(b, coords, spatial), 0,
                                spatial + 1));
      tex->coord_components = spatial + 1;
   }
   tex->is_shadow = false;
   tex->is_new_style_shadow = false;
   /* Always query in the biased form, with a zero bias if there is none:
    * table-addressed samplers get their state bias added this way, and a
    * descriptor-addressed query in the unbiased form returns a different
    * LOD (glsl.texture_functions textureclamp *_bias, cube-array bias). */
   if (nir_tex_instr_src_index(tex, nir_tex_src_bias) < 0) {
      b->cursor = nir_before_instr(instr);
      nir_tex_instr_add_src(tex, nir_tex_src_bias, nir_imm_float(b, 0));
   }
   tex->op = nir_texop_txb;
   tex->backend_flags |= AGX_APPLE9_TEXTURE_LOD_QUERY;
   /* As a sample, the instruction has a sample's destination size; the
    * query result is its first two components. Leaving the two-component
    * destination fails NIR validation after the next pass with progress. */
   b->cursor = nir_after_instr(instr);
   tex->def.num_components = nir_tex_instr_dest_size(tex);
   nir_def *level = nir_channel(b, &tex->def, 0);
   if (data && tex->sampler_index < 32) {
      const struct agx_apple9_sampler_key *key = data;
      if (key[tex->sampler_index].mip_filter == PIPE_TEX_MIPFILTER_NONE)
         level = nir_imm_float(b, 0);
      else if (key[tex->sampler_index].mip_filter == PIPE_TEX_MIPFILTER_NEAREST)
         level = nir_fadd_imm(b, nir_fceil(b, nir_fadd_imm(b, level, 0.5)), -1.0);
   }
   nir_def *result = nir_vec2(b, level, nir_channel(b, &tex->def, 1));
   nir_def_rewrite_uses_after(&tex->def, result);
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
   nir_lower_global_vars_to_local(nir);
   nir_split_var_copies(nir);
   nir_lower_var_copies(nir);
   nir_lower_vars_to_ssa(nir);
   bool optimized;
   do {
      optimized = nir_opt_constant_folding(nir);
      optimized |= nir_opt_copy_prop(nir);
      optimized |= nir_opt_remove_phis(nir);
      optimized |= nir_opt_loop_unroll(nir);
      optimized |= nir_opt_dead_cf(nir);
      optimized |= nir_opt_dce(nir);
   } while (optimized);
   nir_lower_samplers(nir);
   nir_opt_constant_folding(nir);
   nir_shader_instructions_pass(nir, apple9_lower_rectangle,
                                nir_metadata_control_flow, NULL);
   const nir_lower_tex_options options = {
      .lower_txp = ~0u, .lower_1d = true,
      .lower_rect = true, .lower_rect_offset = true,
   };
   nir_lower_tex(nir, &options);
   nir_shader_instructions_pass(nir, apple9_lower_texture_lod_query,
                                nir_metadata_control_flow, (void *)key);
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
         if (tex->backend_flags & AGX_APPLE9_TEXTURE_LOD_QUERY)
            continue;
         /* Normalized depth formats clamp the comparison reference to [0,1].
          * Z24 uses float storage through u_transfer_helper, so the texture
          * descriptor cannot infer this API-format requirement. */
         if (tex->is_shadow &&
             (key[tex->texture_index].flags & AGX_APPLE9_CLAMP_SHADOW_REFERENCE)) {
            int cmp = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
            if (cmp >= 0) {
               b.cursor = nir_before_instr(instr);
               nir_src_rewrite(&tex->src[cmp].src,
                              nir_fsat(&b, tex->src[cmp].src.ssa));
               progress = true;
            }
         }
         const struct agx_apple9_sampler_key *state = &key[tex->sampler_index];
         bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
         /* Comparison samples stay native: as an array layer they take the
          * array-comparison form, whose implicit-LOD encoding returns the
          * stored depth instead of the comparison result. 2D arrays reroute
          * that through gradients; a cube face here keeps the native cube
          * sample and its per-face clamp. */
         /* Implicit-LOD samples stay native too. As an array layer, each
          * pixel picks its own face, so a quad straddling a face edge gets a
          * discontinuous derivative and selects too high a level; the native
          * cube derives the LOD from the direction vector. */
         bool implicit_lod = tex->op == nir_texop_tex || tex->op == nir_texop_txb;
         if (cube && !state->seamless_cube_map && !tex->is_shadow && !implicit_lod) {
            /* Cube filtering always clamps the selected face on Apple9.
             * Sampling that face as an array layer restores ordinary 2D
             * wrapping when seamless cube filtering is disabled. */
            tex->backend_flags |= AGX_APPLE9_TEXTURE_CUBE_AS_ARRAY;
            mapping->seamful_cubes |= BITFIELD_BIT(tex->texture_index);
            progress = true;
         }
         if (!(state->flags & AGX_APPLE9_CUSTOM_BORDER))
            continue;
         if ((tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
              tex->sampler_dim != GLSL_SAMPLER_DIM_3D && !cube) || tex->is_shadow ||
             tex->def.num_components != 4 || tex->def.bit_size != 32) {
            *reason = "Apple9 border lowering requires a non-shadow 32-bit sample";
            return false;
         }
         int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
         if (coord < 0) {
            *reason = "Apple9 border sample has no coordinates";
            return false;
         }
         progress = true;
         unsigned s = tex->sampler_index;
         if (sampler_alias[s] == 0xff) {
            /* Border values live in the shader. Samplers with otherwise equal
             * state can share the same all-white helper descriptor. */
            struct agx_apple9_sampler_key white = *state;
            memset(white.border, 0, sizeof(white.border));
            for (unsigned j = 0; j < 32; ++j) {
               if (sampler_alias[j] == 0xff)
                  continue;
               struct agx_apple9_sampler_key other = key[j];
               memset(other.border, 0, sizeof(other.border));
               if (!memcmp(&white, &other, sizeof(white))) {
                  sampler_alias[s] = sampler_alias[j];
                  break;
               }
            }
         }
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
         nir_def *border = nir_imm_ivec4(&b, fui(state->border[0]), fui(state->border[1]),
                                        fui(state->border[2]), fui(state->border[3]));
         nir_def *result;
         if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float) {
            /* Match hk_shader.c: filtering is linear, so interpolating samples
             * with borders 0 and 1 gives the requested custom border. */
            result = nir_flrp(&b, &tex->def, &weight->def, border);
            nir->info.flrp_lowered = false;
         } else {
            /* Integer sampling is nearest-only. The two native samples differ
             * exactly on border components; select their integer bits without
             * interpreting either the texel or the border as floating point. */
            result = nir_bcsel(&b, nir_ine(&b, &tex->def, &weight->def),
                              border, &tex->def);
         }
         nir_def_rewrite_uses_after(&tex->def, result);
      }
   }
   nir_progress(progress, impl, nir_metadata_control_flow);
   return true;
}


static nir_def *
apple9_fetch_buffer(nir_builder *b, const nir_tex_instr *orig, nir_def *index)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->texture_index = orig->texture_index;
   tex->sampler_index = APPLE9_FETCH_SAMPLER;
   tex->dest_type = orig->dest_type;
   tex->coord_components = 2;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(
      nir_vec2(b, nir_umod_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH),
                  nir_udiv_imm(b, index, AGX_TEXTURE_BUFFER_WIDTH)));
   tex->src[1].src_type = nir_tex_src_lod;
   tex->src[1].src = nir_src_for_ssa(nir_imm_int(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

/* Linear 2D descriptors cover buffer views independently of the underlying
 * buffer allocation. RGB32 has no native format: fetch three scalar words and
 * apply the logical view's swizzle after assembling the texel. */
static bool
apple9_lower_buffer_texture(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF || tex->op != nir_texop_txf)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *index = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *size = agx_apple9_texture_info(b, tex, 0);
   nir_if *valid = nir_push_if(b, nir_ult(b, index, size));
   nir_if *rgb32 = nir_push_if(b,
      nir_ine_imm(b, agx_apple9_texture_info(b, tex, 5), 0));
   nir_def *components[4];
   nir_def *scalar_index = nir_imul_imm(b, index, 3);
   for (unsigned c = 0; c < 3; ++c)
      components[c] = nir_channel(b,
         apple9_fetch_buffer(b, tex, nir_iadd_imm(b, scalar_index, c)), 0);
   bool floating = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
   components[3] = floating ? nir_imm_float(b, 1) : nir_imm_int(b, 1);
   nir_def *swizzle = agx_apple9_texture_info(b, tex, 6);
   nir_def *rgba[4];
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *channel = nir_iand_imm(b, nir_ushr_imm(b, swizzle, c * 3), 7);
      rgba[c] = nir_imm_int(b, 0);
      for (unsigned j = 0; j < 4; ++j)
         rgba[c] = nir_bcsel(b, nir_ieq_imm(b, channel, j), components[j], rgba[c]);
      rgba[c] = nir_bcsel(b, nir_ieq_imm(b, channel, PIPE_SWIZZLE_1),
                         components[3], rgba[c]);
   }
   nir_def *assembled = nir_vec4(b, rgba[0], rgba[1], rgba[2], rgba[3]);
   nir_push_else(b, rgb32);
   nir_def *native = apple9_fetch_buffer(b, tex, index);
   nir_pop_if(b, rgb32);
   nir_def *result = nir_if_phi(b, assembled, native);
   nir_push_else(b, valid);
   nir_def *zero = nir_imm_ivec4(b, 0, 0, 0, 0);
   nir_pop_if(b, valid);
   result = nir_if_phi(b, result, zero);
   nir_def_rewrite_uses(&tex->def, nir_trim_vector(b, result, tex->def.num_components));
   nir_instr_remove(instr);
   return true;
}

/* 2D gradient samples of non-shadow textures take a minimum LOD. */
static bool
apple9_native_min_lod(const nir_tex_instr *tex)
{
   return tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_shadow &&
          nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0 &&
          nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0;
}

/* Implicit and biased 2D and 2D array samples with a minimum LOD have a
 * native biased form; an implicit sample is one with zero bias. */
static bool
apple9_native_biased_min_lod(const nir_tex_instr *tex)
{
   return apple9_native_min_lod(tex) &&
          (tex->op == nir_texop_tex || tex->op == nir_texop_txb);
}

/* Route linear compute sampling, implicit array-shadow sampling,
 * and implicit 2D sampling with a minimum LOD through explicit gradients.
 * Only spatial coordinates contribute derivatives, excluding the layer and
 * comparator. An instruction bias scales gradients so the minimum LOD clamps
 * the biased LOD, while the sampler's bias is applied afterwards. */
/* Whether a texture operation's final form, after the remaining backend
 * rewrites (implicit 3D and comparison samples become bias-0 samples), has a
 * measured offset encoding. Mirrors selection (see the texture emitters). */
static bool
apple9_offset_form_encodable(const nir_tex_instr *tex)
{
   const bool volume = tex->coord_components >= 3 || tex->is_shadow;
   unsigned dimension = 0;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)
      return false;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_3D)
      dimension = 3;
   /* 2D arrays pack the layer with the LOD (dimension 2). */
   if (tex->is_array)
      dimension = 2;
   /* Minimum-LOD samples keep the offset lowering (see the shape check). */
   if (nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0)
      return false;
   bool explicit_lod = false, bias = false, gradient = false;
   switch (tex->op) {
   case nir_texop_tg4:
      /* Comparison gathers select the explicit-LOD comparison form (the
       * base level); plain 2D gathers the implicit form. */
      if (dimension == 3)
         return false;
      explicit_lod = tex->is_shadow;
      break;
   case nir_texop_txd:
      gradient = true;
      break;
   case nir_texop_txl:
      explicit_lod = true;
      break;
   case nir_texop_txb:
      explicit_lod = bias = true;
      break;
   case nir_texop_tex:
      explicit_lod = bias = volume;
      break;
   default:
      return false;
   }
   uint16_t mode = agx_apple9_texture_mode(explicit_lod, gradient, bias,
                                           dimension, tex->is_shadow, false);
   return agx_apple9_texture_offset_mode(mode, NULL);
}

static bool
apple9_flag_unencodable_offsets(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0 ||
       apple9_offset_form_encodable(tex))
      return false;
   tex->backend_flags |= AGX_APPLE9_TEXTURE_OFFSET_FALLBACK;
   *(bool *)data = true;
   return true;
}

static bool
apple9_lower_implicit_gradients(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   bool compute = b->shader->info.stage == MESA_SHADER_COMPUTE &&
                  b->shader->info.derivative_group == DERIVATIVE_GROUP_LINEAR;
   if (!compute && b->shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (!compute && !(tex->is_array && tex->is_shadow &&
                    tex->sampler_dim == GLSL_SAMPLER_DIM_2D) &&
       (!apple9_native_min_lod(tex) || apple9_native_biased_min_lod(tex)))
      return false;
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb)
      return false;
   b->cursor = nir_before_instr(instr);
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   /* Projection, including the shadow comparator, is already lowered. */
   assert(nir_tex_instr_src_index(tex, nir_tex_src_projector) < 0);
   unsigned spatial = tex->coord_components - tex->is_array;
   nir_def *position = nir_trim_vector(b, tex->src[coord].src.ssa, spatial);
   nir_def *dx = nir_ddx(b, position), *dy = nir_ddy(b, position);
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
   if (tex->op == nir_texop_txf_ms && tex->sampler_dim == GLSL_SAMPLER_DIM_MS) {
      /* Input-attachment lowering can retain an explicit zero LOD.
       * Multisample images have only one level and the native fetch has no
       * LOD operand.
       */
      int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      if (lod >= 0) {
         if (!nir_src_is_const(tex->src[lod].src) ||
             nir_src_as_uint(tex->src[lod].src) != 0)
            return false;
         nir_tex_instr_remove_src(tex, lod);
      }
      b->cursor = nir_before_instr(instr);
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      nir_def *xyz = tex->src[coord].src.ssa;
      nir_def *sample = nir_steal_tex_src(tex, nir_tex_src_ms_index);
      nir_def *layer = tex->is_array ? nir_channel(b, xyz, 2) : nir_imm_int(b, 0);
      /* Full-width X/Y are followed by a 12-bit layer and sample index in
       * bits 12..13 of the third publication word. The fourth word is unused. */
      coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      nir_src_rewrite(&tex->src[coord].src,
         nir_vec4(b, nir_channel(b, xyz, 0), nir_channel(b, xyz, 1),
                     nir_ior(b, layer, nir_ishl_imm(b, sample, 12)),
                     nir_imm_int(b, 0)));
      tex->coord_components = 4;
      tex->sampler_index = APPLE9_FETCH_SAMPLER;
      return true;
   }
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

/* Texel fetches outside the image read the border. The fetch sampler
 * clamps coordinates to it, but hardware clamps the level, layer and sample
 * instead; move such fetches out of bounds explicitly. This precedes the
 * native packing of layers and samples.
 */
static bool
apple9_bound_texel_fetch(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if ((tex->op != nir_texop_txf && tex->op != nir_texop_txf_ms) ||
       tex->sampler_dim == GLSL_SAMPLER_DIM_BUF || tex->texture_index >= 32 ||
       (tex->backend_flags & AGX_APPLE9_TEXTURE_FETCH_BOUNDED))
      return false;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *xyz = tex->src[coord].src.ssa;
   nir_def *valid = nir_imm_true(b);
   /* Multisample images have one level; their fetch keeps a literal zero
    * LOD, if any.
    */
   int lod = tex->op == nir_texop_txf ? nir_tex_instr_src_index(tex, nir_tex_src_lod)
                                      : -1;
   if (lod >= 0) {
      nir_def *level = tex->src[lod].src.ssa;
      valid = nir_iand(b, valid, nir_ult(b, level,
         agx_apple9_texture_info(b, tex, 3)));
      nir_src_rewrite(&tex->src[lod].src,
                      nir_bcsel(b, valid, level, nir_imm_int(b, 0)));
   }
   if (tex->is_array) {
      nir_def *layer = nir_channel(b, xyz, tex->coord_components - 1);
      valid = nir_iand(b, valid, nir_ult(b, layer,
         agx_apple9_texture_info(b, tex, 2)));
   }
   int ms = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   if (ms >= 0) {
      valid = nir_iand(b, valid, nir_ult(b, tex->src[ms].src.ssa,
         agx_apple9_texture_info(b, tex, 4)));
   }
   nir_def *x = nir_bcsel(b, valid, nir_channel(b, xyz, 0), nir_imm_int(b, -1));
   nir_src_rewrite(&tex->src[coord].src, nir_vector_insert_imm(b, xyz, x, 0));
   tex->backend_flags |= AGX_APPLE9_TEXTURE_FETCH_BOUNDED;
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
            size[c] = agx_apple9_texture_info(b, tex, layer ? 2 : c);
            if (!layer && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF)
               size[c] = nir_umax(b, nir_ushr(b, size[c], lod), nir_imm_int(b, 1));
         }
         result = nir_vec(b, size, tex->def.num_components);
      } else
         result = agx_apple9_texture_info(b, tex,
                                      tex->op == nir_texop_query_levels ? 3 : 4);
      /* A null view has no levels, and its size and sample queries return 0
       * (Vulkan's null descriptors). Buffer metadata has no level count. */
      if (tex->op != nir_texop_query_levels &&
          tex->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
         nir_def *null = nir_ieq_imm(b, agx_apple9_texture_info(b, tex, 3), 0);
         result = nir_bcsel(b, null, nir_imm_zero(b, result->num_components, 32),
                            result);
      }
      nir_def_rewrite_uses(&tex->def, result);
      nir_instr_remove(instr);
      return true;
   }
   if (tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
       tex->op != nir_texop_lod) {
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (coord < 0)
         return false;
      nir_def *xyzw = tex->src[coord].src.ssa;
      nir_def *layer = nir_f2u32(b, nir_fmax(b,
         nir_fround_even(b, nir_channel(b, xyzw, 3)), nir_imm_float(b, 0)));
      nir_def *cubes = agx_apple9_texture_info(b, tex, 2);
      layer = nir_umin(b, layer, nir_iadd_imm(b, cubes, -1));
      nir_src_rewrite(&tex->src[coord].src,
         nir_vec4(b, nir_channel(b, xyzw, 0), nir_channel(b, xyzw, 1),
                  nir_channel(b, xyzw, 2), layer));
      return true;
   }
   if (tex->is_array && tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
       tex->op != nir_texop_txf && tex->op != nir_texop_lod) {
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (coord < 0)
         return false;
      nir_def *xyz = tex->src[coord].src.ssa;
      nir_def *layer = nir_f2u32(b, nir_fmax(b,
         nir_fround_even(b, nir_channel(b, xyz, 2)), nir_imm_float(b, 0)));
      layer = nir_umin(b, layer,
         nir_iadd_imm(b, agx_apple9_texture_info(b, tex, 2), -1));
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
   /* Descriptor-addressed samplers have their bias applied upstream. */
   if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0)
      return false;

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

/* Selection encodes an implicit sample only in its two-coordinate form; the
 * volume, cube, array and comparison operands exist on the explicit forms.
 * Table-addressed samplers became txb in apple9_lower_sampler_bias; give
 * descriptor-addressed ones, and late implicit queries, the same zero-bias
 * form. */
static bool
apple9_lower_implicit_volume_tex(nir_builder *b, nir_instr *instr,
                                 UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_tex ||
       (tex->coord_components < 3 && !tex->is_array && !tex->is_shadow))
      return false;
   b->cursor = nir_before_instr(instr);
   tex->op = nir_texop_txb;
   nir_tex_instr_add_src(tex, nir_tex_src_bias, nir_imm_float(b, 0));
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
   case PIPE_BLENDFACTOR_SRC1_COLOR:
   case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
   case PIPE_BLENDFACTOR_SRC1_ALPHA:
   case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
      return true;
   default:
      return false;
   }
}

/* The validated color store consumes RGBA8. Express conversion as ordinary NIR
 * so register allocation and numerical behavior do not depend on a captured
 * native color-pack sequence. */
static void
apple9_collect_color(nir_shader *nir, bool dual_source_by_location)
{
   /* Output assignments are ordinary mutable shader state. Collect them
    * before packing or blending so component writes and branch-local writes
    * produce one final color for each surviving invocation. */
   nir_lower_returns(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_variable *color[9] = {0};
   nir_variable *depth = NULL, *sample_mask = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_output &&
             nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_SAMPLE_MASK) {
            if (!sample_mask)
               sample_mask = nir_local_variable_create(impl, glsl_int_type(), "sample mask");
            b.cursor = nir_before_instr(instr);
            nir_store_var(&b, sample_mask, intr->src[0].ssa, 1);
            nir_instr_remove(instr);
            continue;
         }
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
         if (nir_intrinsic_io_semantics(intr).dual_source_blend_index) {
            if (rt != 0)
               continue;
            rt = 8;
         } else if (dual_source_by_location && rt == 1) {
            /* Unlike the epilog ABI, monolithic blending keeps MRT1 and
             * SRC1 separate. Remap before unused render targets are removed. */
            rt = 8;
         }
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
   if (sample_mask) {
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, sample_mask), nir_imm_int(&b, 0),
         .write_mask = 1, .src_type = nir_type_int32,
         .io_semantics = {.location = FRAG_RESULT_SAMPLE_MASK, .num_slots = 1});
   }
   if (depth) {
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, depth), nir_imm_int(&b, 0),
         .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DEPTH, .num_slots = 1});
   }
   /* Collect the dual-source value first, then location 0 before the other
    * render targets so alpha-to-coverage precedes every tile store.
    */
   for (unsigned i = 0; i < 9; ++i) {
      unsigned rt = i == 0 ? 8 : i - 1;
      if (!color[rt])
         continue;
      b.cursor = nir_after_impl(impl);
      nir_store_output(&b, nir_load_var(&b, color[rt]), nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics = {.location = FRAG_RESULT_DATA0 + (rt == 8 ? 0 : rt),
                                        .dual_source_blend_index = rt == 8,
                                        .num_slots = 1});
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

static nir_def *apple9_pack_color(nir_builder *b, enum pipe_format format,
                                  nir_def *color);

static bool
apple9_lower_color(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   struct apple9_color_lower *lower = data;
   /* Depth and stencil exports are emitted by the ZS lowering. */
   if (nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_DEPTH ||
       nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_STENCIL)
      return false;
   unsigned rt = nir_intrinsic_io_semantics(intr).location - FRAG_RESULT_DATA0;
   bool alpha_to_coverage = rt == 0 && lower->blend &&
                            lower->blend[0].alpha_to_coverage;
   if (rt < 8 && rt >= lower->nr_targets && !alpha_to_coverage) {
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
   bool normalized = !raw && !fp16 && !packed_format;
   bool snorm = normalized && channel->type == UTIL_FORMAT_TYPE_SIGNED;
   assert(!normalized || bits == 8);
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
   if (nir_intrinsic_io_semantics(intr).dual_source_blend_index) {
      lower->src1 = color;
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (blend && !blend->colormask && !blend->alpha_to_coverage) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (blend) {
      if (!blend->logicop_enable && !blend->advanced_mode &&
          (blend->rgb_func > PIPE_BLEND_MAX ||
          blend->alpha_func > PIPE_BLEND_MAX ||
          !apple9_blend_factor_supported(blend->rgb_src) ||
          !apple9_blend_factor_supported(blend->rgb_dst) ||
          !apple9_blend_factor_supported(blend->alpha_src) ||
          !apple9_blend_factor_supported(blend->alpha_dst))) {
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
   bool logicop_reads_destination = blend && blend->logicop_enable &&
      !util_format_is_float(format) && !util_format_is_srgb(format) &&
      blend->logicop_func != PIPE_LOGICOP_CLEAR &&
      blend->logicop_func != PIPE_LOGICOP_COPY &&
      blend->logicop_func != PIPE_LOGICOP_COPY_INVERTED &&
      blend->logicop_func != PIPE_LOGICOP_SET;
   bool reads_destination = blend &&
      (((blend->colormask & component_mask) != component_mask) ||
       logicop_reads_destination || (!blend->logicop_enable && !integer &&
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
   if (blend && blend->dynamic_enable) {
      nir_def *enables = nir_load_ubo(b, 1, 32,
         nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
         nir_imm_int(b, AGX_APPLE9_COLOR_WRITE_ENABLE_OFFSET),
         .align_mul = 4, .range = 4);
      coverage = nir_bcsel(b, nir_i2b(b, nir_ubitfield_extract_imm(b, enables, rt, 1)),
                           coverage, nir_imm_int(b, 0));
   }
   /* Alpha-to-coverage applies even when location 0 has no attachment. */
   if (rt >= lower->nr_targets || (blend && blend->color_attachment_unused)) {
      nir_instr_remove(&intr->instr);
      return true;
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
         for (unsigned c = 0; c < (packed_format ? 0 : 4); ++c) {
            if (c >= components) {
               channels[c] = integer ? nir_imm_int(b, c == 3) : nir_imm_float(b, c == 3);
               continue;
            }
            nir_def *value = nir_channel(b, packed_dst, (bits * c) / 32);
            if (normalized) {
               nir_def *unpacked = snorm ? nir_unpack_snorm_4x8(b, value)
                                          : nir_unpack_unorm_4x8(b, value);
               channels[c] = nir_channel(b, unpacked, c);
               continue;
            }
            if (bits < 32) {
               value = nir_ushr_imm(b, value, (bits * c) % 32);
               value = integer && channel->type == UTIL_FORMAT_TYPE_SIGNED
                  ? nir_ishr_imm(b, nir_ishl_imm(b, value, 32 - bits), 32 - bits)
                  : nir_iand_imm(b, value, BITFIELD_MASK(bits));
            }
            channels[c] = raw ? value : apple9_unpack_half(b, value);
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
         if (blend->logicop_enable)
            color = nir_color_logicop(b, color, dst, blend->logicop_func, format);
         else if (!integer) {
#ifndef NDEBUG
            if (!lower->src1 && agx_apple9_blend_uses_src1(blend))
               mesa_logw_once("Apple9 dual-source blend has no SRC1 output");
#endif
            color = nir_color_blend(b, color, lower->src1 ? lower->src1 : nir_undef(b, 4, 32),
                                    dst, &rt, false);
         }
         color = nir_color_mask(b, color, dst, blend->colormask);
      }
      if (util_format_is_srgb(format))
         color = nir_vector_insert_imm(b, nir_format_linear_to_srgb(b, color),
                                      nir_channel(b, color, 3), 3);
      if (tile_format != format && util_format_is_unorm(format))
         color = nir_fsat(b, color);
      else if (tile_format != format && util_format_is_snorm(format))
         color = nir_fclamp(b, color, nir_imm_float(b, -1), nir_imm_float(b, 1));
      nir_def *packed[4] = {nir_imm_int(b, 0), nir_imm_int(b, 0),
                            nir_imm_int(b, 0), nir_imm_int(b, 0)};
      if (normalized) {
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; ++c)
            channels[c] = c < components ? nir_channel(b, color, c)
                                          : nir_imm_float(b, 0);
         nir_def *rgba = nir_vec(b, channels, 4);
         packed[0] = snorm ? nir_pack_snorm_4x8(b, rgba)
                           : nir_pack_unorm_4x8(b, rgba);
      }
      for (unsigned c = 0; c < (packed_format || normalized ? 0 : components); ++c) {
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
         packed[0] = apple9_pack_color(b, format, color);
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

/* Packed colour formats with a native single-instruction pack (G16: RG11B10
 * float and RGB10A2 unorm, component 0 in the low bits); others use NIR's
 * software pack. */

static nir_def *
apple9_pack_color(nir_builder *b, enum pipe_format format, nir_def *color)
{
   switch (format) {
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return nir_pack_format_agx(b, nir_trim_vector(b, color, 3),
                                 .format = PIPE_FORMAT_R11G11B10_FLOAT);
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return nir_pack_format_agx(b, color, .format = PIPE_FORMAT_R10G10B10A2_UNORM);
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return nir_pack_format_agx(b, nir_swizzle(b, color, (unsigned[]){2, 1, 0, 3}, 4),
                                 .format = PIPE_FORMAT_R10G10B10A2_UNORM);
   default:
      return nir_format_pack_rgba(b, format, color);
   }
}

/* Vertex channels preserve integer values; packed 32-bit formats are
 * extracted with ordinary NIR shifts and masks. */
bool
agx_apple9_vertex_format_supported(enum pipe_format format)
{
   if (format == PIPE_FORMAT_R11G11B10_FLOAT)
      return true;
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
   nir->info.clip_distance_array_size = 0;
   nir->info.cull_distance_array_size = 0;
   nir->xfb_info = NULL;
}

struct apple9_vertex_lower {
   const struct agx_apple9_vertex_layout *layout;
   bool software;
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
   if (lower->layout && attribute < 16 && intr->def.bit_size == 32 &&
       lower->layout->format[attribute] == PIPE_FORMAT_NONE) {
      /* An input without a vertex attribute reads (0, 0, 0, 1). */
      b->cursor = nir_before_instr(&intr->instr);
      bool integer = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) !=
                     nir_type_float;
      nir_def *one = integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0);
      nir_def *values[4];
      for (unsigned i = 0; i < intr->num_components; ++i) {
         values[i] = nir_intrinsic_component(intr) + i == 3 ? one
                                                           : nir_imm_int(b, 0);
      }
      nir_def_replace(&intr->def, nir_vec(b, values, intr->num_components));
      return true;
   }
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
   nir_def *index = divisor == AGX_APPLE9_VERTEX_DIVISOR_ZERO ? nir_imm_int(b, 0)
                  : divisor ? nir_udiv_imm(b, nir_load_instance_id(b), divisor)
                            : nir_load_vertex_id(b);
   nir_def *index64 = nir_u2u64(b, index);
   if (divisor) {
      nir_def *base_instance = lower->layout->capture_xfb || lower->software
         ? nir_load_base_instance(b)
         : nir_load_ubo(b, 1, 32,
                       nir_imm_int(b, lower->layout->draw_params_ubo),
                       nir_imm_int(b, 4), .align_mul = 4,
                       .range_base = 4, .range = 4);
      index = nir_iadd(b, index, base_instance);
      index64 = nir_iadd(b, index64, nir_u2u64(b, base_instance));
   }
   nir_def *offset = nir_iadd_imm(b,
      nir_imul_imm(b, index, lower->layout->stride[attribute]),
      lower->layout->offset[attribute]);
   nir_def *buffer_address = NULL, *buffer_size = NULL, *offset64 = NULL;
   if (lower->layout->buffer_descriptor_ubo) {
      unsigned descriptor_size = lower->layout->dynamic_stride ? 32 : 16;
      unsigned descriptor_offset =
         lower->layout->buffer[attribute] * descriptor_size;
      nir_def *descriptor = nir_load_ubo(b, 4, 32,
         nir_imm_int(b, lower->layout->buffer_descriptor_ubo),
         nir_imm_int(b, descriptor_offset), .align_mul = 16,
         .range_base = descriptor_offset, .range = 16);
      buffer_address = nir_pack_64_2x32_split(b, nir_channel(b, descriptor, 0),
                                                nir_channel(b, descriptor, 1));
      buffer_size = nir_pack_64_2x32_split(b, nir_channel(b, descriptor, 2),
                                             nir_channel(b, descriptor, 3));
      nir_def *stride;
      if (lower->layout->dynamic_stride) {
         stride = nir_load_ubo(b, 1, 32,
            nir_imm_int(b, lower->layout->buffer_descriptor_ubo),
            nir_imm_int(b, descriptor_offset + 16), .align_mul = 16,
            .range_base = descriptor_offset + 16, .range = 4);
      } else {
         stride = nir_imm_int(b, lower->layout->stride[attribute]);
      }
      offset64 = nir_iadd_imm(b, nir_imul(b, index64, nir_u2u64(b, stride)),
                              lower->layout->offset[attribute]);
      /* Robust vertex input treats a whole attribute as out of bounds when
       * any byte of it is. Such an attribute reads the zero sink instead of
       * its buffer, so every component loads unconditionally. */
      nir_def *end = nir_iadd_imm(b, offset64, desc->block.bits / 8);
      nir_def *valid = nir_uge(b, buffer_size, end);
      buffer_address = nir_bcsel(b, valid, buffer_address,
                                 nir_load_ro_sink_address_poly(b));
      offset64 = nir_bcsel(b, valid, offset64, nir_imm_int64(b, 0));
   }
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
      /* Vulkan aligns the effective address, including the bound buffer
       * offset. An attribute offset alone need not be naturally aligned.
       * The descriptor path loads from that complete address.
       */
      if (!bytes || (!buffer_address &&
                     ((lower->layout->stride[attribute] % bytes) || (add % bytes)))) {
         lower->valid = false;
         return false;
      }
      unsigned bit_size = packed ? 32 : ch->size;
      nir_def *value;
      if (buffer_address) {
         nir_def *component_offset = nir_iadd_imm(b, offset64, packed ? 0 : ch->shift / 8);
         value = nir_load_global_constant(b, 1, bit_size,
            nir_iadd(b, buffer_address, component_offset), .align_mul = bytes);
      } else {
         value = nir_load_ubo(b, 1, bit_size,
            nir_imm_int(b, 32 + lower->layout->buffer[attribute]),
            nir_iadd_imm(b, offset, packed ? 0 : ch->shift / 8),
            .align_mul = bytes, .range = ~0u);
      }
      if (packed) {
         value = nir_ushr_imm(b, value, ch->shift);
         value = ch->type == UTIL_FORMAT_TYPE_SIGNED
                    ? nir_ishr_imm(b, nir_ishl_imm(b, value, 32 - ch->size), 32 - ch->size)
                    : nir_iand_imm(b, value, BITFIELD_MASK(ch->size));
      }
      if (ch->type == UTIL_FORMAT_TYPE_FLOAT) {
         if (ch->size == 16)
            value = apple9_unpack_half(b, nir_u2u32(b, value));
         else if (ch->size == 10 || ch->size == 11) {
            /* Unsigned packed floats share the half-float exponent bias.
             * Insert the missing low mantissa bits and a zero sign bit.
             */
            value = apple9_unpack_half(b, nir_ishl_imm(b, value, 15 - ch->size));
         }
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

bool
agx_nir_lower_apple9_vertex_inputs(
   nir_shader *nir, const struct agx_apple9_vertex_layout *layout, bool software)
{
   struct apple9_vertex_lower lower = {
      .layout = layout, .software = software, .valid = true,
   };
   nir_shader_intrinsics_pass(nir, apple9_lower_vertex_input,
                              nir_metadata_control_flow, &lower);
   return lower.valid;
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
apple9_lower_unwritten_inputs(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_varying_layout *producer = data;
   if (!producer ||
       (intr->intrinsic != nir_intrinsic_load_input &&
        intr->intrinsic != nir_intrinsic_load_interpolated_input))
      return false;

   nir_src *offset = nir_get_io_offset_src(intr);
   if (!nir_src_is_const(*offset))
      return false;

   unsigned location = nir_intrinsic_io_semantics(intr).location +
                       nir_src_as_uint(*offset);
   unsigned component = nir_intrinsic_component(intr);
   if (location >= ARRAY_SIZE(producer->mask) ||
       location == VARYING_SLOT_PRIMITIVE_ID ||
       component + intr->num_components > 4)
      return false;

   unsigned written = producer->mask[location] >> component;
   if (!(nir_def_components_read(&intr->def) & ~written))
      return false;

   /* A viewport index the vertex stage does not write is zero. */
   if (location == VARYING_SLOT_VIEWPORT) {
      b->cursor = nir_after_instr(&intr->instr);
      nir_def_rewrite_uses_after(&intr->def,
         nir_imm_zero(b, intr->def.num_components, intr->def.bit_size));
      return true;
   }

   /* Unwritten outputs have undefined values. Preserve every produced
    * component without requiring storage for the absent interface slots. */
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *channels[4];
   for (unsigned c = 0; c < intr->num_components; ++c)
      channels[c] = written & BITFIELD_BIT(c)
         ? nir_channel(b, &intr->def, c) : nir_undef(b, 1, intr->def.bit_size);
   nir_def *replacement = nir_vec(b, channels, intr->num_components);
   nir_def_rewrite_uses_after(&intr->def, replacement);
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
         /* The viewport index also travels as a flat varying for fragment
          * shaders; the rasterizer's packed layer/viewport word does not
          * reach them.
          */
         if (!fragment && (location == VARYING_SLOT_POS ||
                           location == VARYING_SLOT_PSIZ ||
                           location == VARYING_SLOT_LAYER))
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
         /* Without a geometry output, the rasterizer generates this scalar. */
         if (fragment && location == VARYING_SLOT_PRIMITIVE_ID &&
             (!producer || !producer->mask[VARYING_SLOT_PRIMITIVE_ID]) && component == 0 &&
             intr->intrinsic == nir_intrinsic_load_input && intr->num_components == 1)
            continue;
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
      *reason = "Apple9 graphics supports at most 124 user varying scalars";
      return false;
   }
   return true;
unsupported:
   *reason = "Apple9 graphics requires constant-indexed 32-bit user varyings";
   return false;
}

struct apple9_clip_distance_state {
   unsigned base, enable;
};

static bool
apple9_lower_clip_distance(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location != VARYING_SLOT_CLIP_DIST0 && location != VARYING_SLOT_CLIP_DIST1)
      return false;

   const struct apple9_clip_distance_state *state = data;
   unsigned base = 4 * (location - VARYING_SLOT_CLIP_DIST0 +
                        nir_src_as_uint(intr->src[1])) + nir_intrinsic_component(intr);
   b->cursor = nir_after_instr(&intr->instr);
   for (unsigned c = 0; c < intr->num_components; ++c) {
      unsigned plane = base + c;
      if (!(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(c)) ||
          plane >= b->shader->info.clip_distance_array_size)
         continue;
      /* Retain the ordinary varying for fragment inputs and transform
       * feedback. Only the rasterizer copy is zeroed for disabled planes. */
      nir_def *distance = state->enable & BITFIELD_BIT(plane)
         ? nir_channel(b, intr->src[0].ssa, c) : nir_imm_float(b, 0);
      nir_store_uvs_agx(b, distance, nir_imm_int(b, state->base + plane));
   }
   return true;
}

/* Raster routing shares a packed layer/viewport word, after point size and
 * before clip distances: layer in the low half, viewport in the high half.
 * Either may be written anywhere, so collect both and publish at the end. */
static bool
apple9_collect_layer_viewport(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   nir_variable **vars = data;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location != VARYING_SLOT_LAYER && location != VARYING_SLOT_VIEWPORT)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_store_var(b, vars[location == VARYING_SLOT_VIEWPORT], intr->src[0].ssa, 1);
   /* Keep the viewport store as the fragment-visible varying. */
   if (location == VARYING_SLOT_LAYER)
      nir_instr_remove(&intr->instr);
   return true;
}

static void
apple9_lower_layer_viewport_output(nir_shader *nir, unsigned slot)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *vars[2] = {
      nir_local_variable_create(impl, glsl_uint_type(), "layer"),
      nir_local_variable_create(impl, glsl_uint_type(), "viewport"),
   };
   nir_store_var(&b, vars[0], nir_imm_int(&b, 0), 1);
   nir_store_var(&b, vars[1], nir_imm_int(&b, 0), 1);
   nir_shader_intrinsics_pass(nir, apple9_collect_layer_viewport,
                              nir_metadata_control_flow, vars);
   b.cursor = nir_after_impl(impl);
   nir_def *word = nir_ior(&b, nir_iand_imm(&b, nir_load_var(&b, vars[0]), 0xffff),
                           nir_ishl_imm(&b, nir_load_var(&b, vars[1]), 16));
   nir_store_uvs_agx(&b, word, nir_imm_int(&b, slot));
   nir_lower_vars_to_ssa(nir);
}

static nir_def *
apple9_load_fixed_point_size(nir_builder *b)
{
   return nir_load_ubo(b, 1, 32,
      nir_imm_int(b, AGX_APPLE9_GRAPHICS_SYSVAL_BINDING),
      nir_imm_int(b, AGX_APPLE9_POINT_SIZE_OFFSET),
      .align_mul = 4, .range = 4);
}

static bool
apple9_lower_point_size(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PSIZ)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *fixed = apple9_load_fixed_point_size(b);
   /* Points of 512 pixels or more do not rasterize; keep sizes within the
    * advertised [1, 511.9375] range. */
   nir_def *size = nir_fmin(b, nir_fmax(b, intr->src[0].ssa, nir_imm_float(b, 1)),
                            nir_imm_float(b, 511.9375f));
   nir_src_rewrite(&intr->src[0],
      nir_bcsel(b, nir_fgt_imm(b, fixed, 0), fixed, size));
   return true;
}

/* The biased cube-array compare form is not yet decoded, and misreads
 * the comparison. Take derivatives of the direction and let the common
 * cube-gradient lowering select an explicit LOD instead. */
static bool
apple9_lower_shadow_cube_array_implicit_lod(nir_builder *b, nir_instr *instr,
                                            UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE || !tex->is_array ||
       !tex->is_shadow ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb) ||
       b->shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *dir = nir_trim_vector(b, tex->src[coord].src.ssa, 3);
   nir_def *ddx = nir_ddx(b, dir), *ddy = nir_ddy(b, dir);
   int bias = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   if (bias >= 0) {
      nir_def *scale = nir_fexp2(b, tex->src[bias].src.ssa);
      ddx = nir_fmul(b, ddx, scale);
      ddy = nir_fmul(b, ddy, scale);
      nir_tex_instr_remove_src(tex, bias);
   }
   nir_tex_instr_add_src(tex, nir_tex_src_ddx, ddx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, ddy);
   tex->op = nir_texop_txd;
   return true;
}

/* Texel fetches below the integer part of an image view's minimum LOD are
 * out of bounds. Such fetches use an out-of-range level, which the fetch
 * bounds check then rejects.
 */
static bool
apple9_lower_fetch_min_lod(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_txf && tex->op != nir_texop_txf_ms)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
   if (!min_lod)
      return false;
   if (tex->op == nir_texop_txf_ms) {
      /* Multisample images have a single level. */
      int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      nir_def *xyz = tex->src[coord].src.ssa;
      nir_def *x = nir_bcsel(b, nir_fge_imm(b, nir_f2f32(b, min_lod), 1.0),
                             nir_imm_intN_t(b, -1, xyz->bit_size),
                             nir_channel(b, xyz, 0));
      nir_src_rewrite(&tex->src[coord].src, nir_vector_insert_imm(b, xyz, x, 0));
      return true;
   }
   int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *level = lod >= 0 ? nir_u2u32(b, tex->src[lod].src.ssa)
                             : nir_imm_int(b, 0);
   nir_def *min_level = nir_f2u32(b, nir_ffloor(b, nir_f2f32(b, min_lod)));
   level = nir_bcsel(b, nir_uge(b, level, min_level), level,
                     nir_imm_int(b, ~0));
   if (lod >= 0)
      nir_src_rewrite(&tex->src[lod].src, level);
   else
      nir_tex_instr_add_src(tex, nir_tex_src_lod, level);
   return true;
}

/* Gradient samples of 2D textures clamp the LOD in hardware to a minimum
 * given as signed Q6 in bits 16..27 of coordinate word 3. Pack it separately
 * from the array layer, which occupies the low half. Other samples take no
 * minimum LOD: select the LOD explicitly for implicit and biased sampling,
 * and let nir_lower_tex handle the remaining gradients.
 */
static bool
apple9_lower_min_lod(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0)
      return false;

   b->cursor = nir_before_instr(instr);
   const bool biased = b->shader->info.stage == MESA_SHADER_FRAGMENT &&
                       apple9_native_biased_min_lod(tex);
   if ((tex->op == nir_texop_txd && apple9_native_min_lod(tex)) || biased) {
      if (tex->op == nir_texop_tex) {
         nir_tex_instr_add_src(tex, nir_tex_src_bias, nir_imm_float(b, 0.0f));
         tex->op = nir_texop_txb;
      }
      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
      uint32_t fp_math_ctrl = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_fp_no_fast_math;
      nir_def *clamped = nir_fmin(b, nir_fmax(b, min_lod, nir_imm_float(b, 0.0f)),
                                  nir_imm_float(b, 2047.0f / 64.0f));
      nir_def *fixed = nir_f2u32(b, nir_ffloor(b, nir_fmul_imm(b, clamped, 64.0f)));
      b->fp_math_ctrl = fp_math_ctrl;
      /* Signed Q6: in bits 16..27 of its word for gradients and arrays,
       * in the low 12 bits beside the bias for biased 2D samples. */
      nir_tex_instr_add_src(tex, nir_tex_src_backend2,
                            biased && !tex->is_array ? fixed : nir_ishl_imm(b, fixed, 16));
      return true;
   }
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb)
      return false;

   nir_def *lod = nir_get_texture_lod(b, tex);
   nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
   if (bias)
      lod = nir_fadd(b, lod, nir_f2fN(b, bias, lod->bit_size));
   nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
   lod = nir_fmax(b, lod, nir_f2fN(b, min_lod, lod->bit_size));
   nir_tex_instr_add_src(tex, nir_tex_src_lod, lod);
   tex->op = nir_texop_txl;
   return true;
}

/* External (EGLImage) textures are ordinary 2D views, as on Apple8. */
static bool
apple9_lower_external(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_EXTERNAL)
      return false;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   return true;
}

static void
apple9_lower_textures(nir_shader *nir)
{
   nir_shader_instructions_pass(nir, apple9_lower_external,
                                nir_metadata_all, NULL);
   nir_lower_samplers(nir);
   nir_shader_instructions_pass(nir, apple9_lower_fetch_min_lod,
                                nir_metadata_control_flow, NULL);
   agx_nir_legalize_apple9_texture_sources(nir);
   /* Gradients must follow projection. Common NIR lowering also projects
    * shadow comparators and leaves array layers unchanged. */
   const nir_lower_tex_options projection = {.lower_txp = ~0u};
   nir_lower_tex(nir, &projection);
   nir_shader_instructions_pass(nir, apple9_lower_compute_lod_query,
                                nir_metadata_control_flow, NULL);
   const nir_lower_tex_options tex_options = {
      .lower_1d = true,
      .lower_rect = true, .lower_rect_offset = true,
      .lower_invalid_implicit_lod = true,
      .lower_txf_offset = true,
      .lower_txd_cube_map = true,
      .lower_txd_clamp = true,
   };
   nir_shader_instructions_pass(nir, apple9_lower_implicit_gradients,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_rectangle,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_shadow_cube_array_implicit_lod,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_min_lod,
                                nir_metadata_control_flow, NULL);
   nir_lower_tex(nir, &tex_options);
   /* Native texel offsets need a measured offset form for the sample's final
    * form; lower any other offset with runtime sampler state, never fail. */
   bool fallback = false;
   nir_shader_instructions_pass(nir, apple9_flag_unencodable_offsets,
                                nir_metadata_all, &fallback);
   if (fallback) {
      const char *offset_reason = NULL;
      agx_nir_lower_apple9_texture_offsets(nir, NULL, &offset_reason);
   }
   nir_shader_instructions_pass(nir, apple9_lower_buffer_texture,
                                nir_metadata_none, NULL);
   nir_shader_instructions_pass(nir, apple9_bound_texel_fetch,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_texel_fetch,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_texture_queries_and_layers,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_texture_lod_query,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_sampler_bias,
                                nir_metadata_control_flow, NULL);
   nir_shader_instructions_pass(nir, apple9_lower_implicit_volume_tex,
                                nir_metadata_control_flow, NULL);
   agx_nir_lower_apple9_texture_lod(nir);
}

/* Coverage-only tests clamp interpolated depth to [0, 1]. For a single
 * sample, replace the initial early test with a depth publication clamped
 * to the driver's range. Replacing it, rather than adding another test,
 * avoids testing twice or applying stencil operations twice.
 *
 * Multisample interpolation needs a depth value per sample and continues to
 * use the native coverage test. */
static bool
apple9_clamp_early_depth(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_blend *blend = data;
   if (intr->intrinsic != nir_intrinsic_sample_mask_agx ||
       !nir_src_is_const(intr->src[0]) ||
       !nir_src_is_const(intr->src[1]) ||
       nir_src_as_uint(intr->src[0]) != 0xff ||
       nir_src_as_uint(intr->src[1]) != 0xff)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   unsigned offset = blend->early_depth_clamp_offset_B;
   nir_def *range = nir_load_ubo(b, 2, 32,
      nir_imm_int(b, blend->early_depth_clamp_ubo), nir_imm_int(b, offset),
      .align_mul = 4, .range_base = offset, .range = 8);
   nir_def *z = nir_load_frag_coord_z(b);
   z = nir_fmin(b, nir_fmax(b, z, nir_channel(b, range, 0)),
                   nir_channel(b, range, 1));
   nir_store_zs_agx(b, intr->src[0].ssa, z, nir_undef(b, 1, 16), .base = 1);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Native depth publication carries both affected and live sample masks, and
 * tile stores take an explicit sample mask built from entry coverage. Neither
 * observes earlier demotions, so keep demoted samples out of subsequent
 * publications and stores. */
static void
apple9_preserve_demotions(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool depth = false, demotes = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         depth |= op == nir_intrinsic_store_zs_agx;
         demotes |= op == nir_intrinsic_sample_mask_agx;
      }
   }
   if (!demotes)
      return;
   /* A late depth/stencil publication must follow every coverage kill,
    * including alpha-to-coverage lowered with the colour output. Early tests
    * stay before shader side effects and discard. */
   nir_instr *last_kill = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_sample_mask_agx &&
             block->cf_node.parent == &impl->cf_node)
            last_kill = instr;
      }
   }
   if (last_kill && depth && !nir->info.fs.early_fragment_tests) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr == last_kill)
               goto moved;
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_zs_agx &&
                block->cf_node.parent == &impl->cf_node)
               nir_instr_move(nir_after_instr(last_kill), instr);
         }
      }
   }
moved:;
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *killed =
      nir_local_variable_create(impl, glsl_uint16_t_type(), "killed samples");
   nir_store_var(&b, killed, nir_imm_intN_t(&b, 0, 16), 1);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_sample_mask_agx &&
             intr->intrinsic != nir_intrinsic_store_zs_agx &&
             intr->intrinsic != nir_intrinsic_store_local_pixel_agx)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_def *dead = nir_load_var(&b, killed);
         if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
            /* A later coverage update must not revive a killed sample. */
            nir_src_rewrite(&intr->src[1],
                            nir_iand(&b, intr->src[1].ssa, nir_inot(&b, dead)));
            nir_def *new_dead = nir_iand(&b, intr->src[0].ssa,
                                        nir_inot(&b, intr->src[1].ssa));
            nir_store_var(&b, killed, nir_ior(&b, dead, new_dead), 1);
         } else if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
            nir_def *mask = intr->src[1].ssa;
            nir_src_rewrite(&intr->src[1],
               nir_iand(&b, mask,
                        nir_inot(&b, nir_u2uN(&b, dead, mask->bit_size))));
         } else {
            nir_src_rewrite(&intr->src[0],
               nir_iand(&b, intr->src[0].ssa, nir_inot(&b, dead)));
            /* The stencil reference is published as a zero-extended word. */
            if (nir_intrinsic_base(intr) & 2)
               nir_src_rewrite(&intr->src[2], nir_u2u32(&b, intr->src[2].ssa));
         }
      }
   }
   nir_progress(true, impl, nir_metadata_control_flow);
}

/* A per-vertex mask for the software geometry path. It never interpolates:
 * poly combines the original vertices' masks before rasterization, and the
 * fragment shader reads a flat primitive mask. Hardware-only points can use
 * the per-vertex mask directly. Keep the actual distance outputs unchanged.
 */
static void
apple9_pack_cull_distance(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *mask = nir_local_variable_create(impl, glsl_uint_type(), NULL);
   nir_store_var(&b, mask, nir_imm_int(&b, 0), 1);
   unsigned clip = nir->info.clip_distance_array_size;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         unsigned loc = nir_intrinsic_io_semantics(intr).location;
         if (loc != VARYING_SLOT_CLIP_DIST0 && loc != VARYING_SLOT_CLIP_DIST1)
            continue;
         unsigned c = 4 * (loc - VARYING_SLOT_CLIP_DIST0 +
                            nir_src_as_uint(intr->src[1])) +
                      nir_intrinsic_component(intr);
         if (c < clip || c >= clip + nir->info.cull_distance_array_size)
            continue;
         unsigned bit = BITFIELD_BIT(c - clip);
         b.cursor = nir_before_instr(instr);
         nir_def *v = nir_bcsel(&b, nir_flt_imm(&b, intr->src[0].ssa, 0),
                                    nir_imm_int(&b, bit), nir_imm_int(&b, 0));
         nir_store_var(&b, mask,
            nir_ior(&b, nir_iand_imm(&b, nir_load_var(&b, mask), ~bit), v), 1);
      }
   }
   b.cursor = nir_after_impl(impl);
   nir_store_output(&b, nir_load_var(&b, mask), nir_imm_int(&b, 0),
      .src_type = nir_type_uint32, .write_mask = 1,
      .io_semantics.location = VARYING_SLOT_CULL_PRIMITIVE);
   nir->info.outputs_written |= BITFIELD64_BIT(VARYING_SLOT_CULL_PRIMITIVE);
   nir_progress(true, impl, nir_metadata_none);
}

/* The vertex stage exports, per cull half-space, a linear varying that is
 * one where the cull distance is non-negative. A primitive is culled when
 * that plane is zero at every vertex: zero here with zero screen-space
 * derivatives. Interpolated zeros are exact, so partially culled primitives
 * keep a nonzero gradient. */
static void
apple9_lower_cull_distance_fs(nir_shader *nir,
                              const struct agx_apple9_varying_layout *producer)
{
   if (!producer->mask[VARYING_SLOT_CULL_PRIMITIVE] &&
       !producer->mask[VARYING_SLOT_CULL_PRIMITIVE + 1])
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   if (producer->cull_distance_mask) {
      nir_def *mask = nir_load_input(&b, 1, 32, nir_imm_int(&b, 0),
         .dest_type = nir_type_uint32,
         .io_semantics.location = VARYING_SLOT_CULL_PRIMITIVE);
      nir_demote_if(&b, nir_ine_imm(&b, mask, 0));
      nir->info.inputs_read |= BITFIELD64_BIT(VARYING_SLOT_CULL_PRIMITIVE);
      nir->info.fs.uses_discard = true;
      nir_progress(true, impl, nir_metadata_none);
      return;
   }
   nir_def *bary = nir_load_barycentric_pixel(&b, 32,
      .interp_mode = INTERP_MODE_NOPERSPECTIVE);
   nir_def *culled = nir_imm_false(&b);
   for (unsigned slot = 0; slot < 2; ++slot) {
      u_foreach_bit(c, producer->mask[VARYING_SLOT_CULL_PRIMITIVE + slot]) {
         nir_def *q = nir_load_interpolated_input(&b, 1, 32, bary,
            nir_imm_int(&b, 0), .component = c, .dest_type = nir_type_float32,
            .io_semantics.location = VARYING_SLOT_CULL_PRIMITIVE + slot,
            .io_semantics.num_slots = 1);
         nir_def *zero = nir_iand(&b, nir_feq_imm(&b, q, 0),
            nir_iand(&b, nir_feq_imm(&b, nir_ddx(&b, q), 0),
                         nir_feq_imm(&b, nir_ddy(&b, q), 0)));
         culled = nir_ior(&b, culled, zero);
      }
   }
   nir_demote_if(&b, culled);
   nir->info.inputs_read |= BITFIELD64_RANGE(VARYING_SLOT_CULL_PRIMITIVE, 2);
   nir->info.fs.uses_discard = true;
   nir_progress(true, impl, nir_metadata_none);
}

static nir_def *apple9_pack_color(nir_builder *b, enum pipe_format format,
                                  nir_def *color);

static bool
apple9_compile_graphics(nir_shader *nir, struct agx_shader_part *out,
                        const struct agx_apple9_vertex_layout *layout,
                        const struct agx_apple9_varying_layout *producer,
                        const struct agx_apple9_blend *blend,
                        unsigned nr_targets, const char **reason)
{
   bool preserves_color = false;
   const char *unused_reason = NULL;
   if (!reason)
      reason = &unused_reason;
   memset(out, 0, sizeof(*out));
   if (reason)
      *reason = NULL;
   /* Legacy gl_FragColor broadcasts to each enabled draw buffer. */
   agx_nir_lower_printf_buffer(nir);
   apple9_lower_subgroups(nir);
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
   if (nir->info.stage == MESA_SHADER_VERTEX &&
       nir->info.cull_distance_array_size) {
      nir_lower_io_to_scalar(nir, nir_var_shader_out, NULL, NULL);
      if (layout && layout->cull_distance_mask)
         apple9_pack_cull_distance(nir);
      else
         agx_nir_lower_cull_distance_vs(nir);
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT && producer)
      apple9_lower_cull_distance_fs(nir, producer);
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      /* Position may be wholly or partially undefined, including shaders
       * used only for side effects with rasterization disabled. The native
       * export record still requires four words. Choose zero for components
       * which the shader never writes. */
      unsigned written = 0;
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_output &&
                nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_POS)
               written |= nir_intrinsic_write_mask(intr) <<
                          nir_intrinsic_component(intr);
         }
      }
      if (written != 0xf) {
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 0), nir_imm_int(&b, 0),
            .write_mask = 0xf & ~written, .src_type = nir_type_float32,
            .io_semantics.location = VARYING_SLOT_POS,
            .io_semantics.num_slots = 1);
      }
   }
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
      if (!agx_nir_lower_apple9_vertex_inputs(nir, layout, false)) {
         *reason = "Apple9 vertex inputs require supported naturally aligned formats";
         return false;
      }
      if (layout && layout->capture_xfb)
         apple9_lower_xfb(nir, layout);
      if (layout && layout->rasterize_points && !layout->capture_xfb) {
         /* mesa/st cannot insert point size when the API varying interface
          * is full. The hardware still needs the separate system output. */
         if (!nir_shader_intrinsics_pass(nir, apple9_lower_point_size,
                                           nir_metadata_control_flow, NULL)) {
            nir_builder b = nir_builder_at(
               nir_after_impl(nir_shader_get_entrypoint(nir)));
            /* APIs without a fixed size (zero) default to one pixel. */
            nir_def *fixed = apple9_load_fixed_point_size(&b);
            fixed = nir_bcsel(&b, nir_fgt_imm(&b, fixed, 0), fixed,
                              nir_imm_float(&b, 1.0f));
            nir_store_output(&b, fixed, nir_imm_int(&b, 0),
               .io_semantics.location = VARYING_SLOT_PSIZ,
               .io_semantics.num_slots = 1, .write_mask = 1,
               .src_type = nir_type_float32);
         }
      }
   }
   /* Use the same structured mask/loop model as compute. Continuations become
    * masked regions with one backedge before instruction selection. */
   nir_lower_continue_constructs(nir);
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      apple9_collect_color(nir, blend && blend[0].dual_source_by_location);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      /* Opaque visibility can discard earlier fragments for the whole pixel.
       * An omitted or masked target still needs their values, even when this
       * shader has no explicit tile load. Use ordered tile access, as for
       * blending, to preserve those attachments. */
      for (unsigned rt = 0; rt < nr_targets; ++rt) {
         /* With VK_EXT_dynamic_rendering_unused_attachments, a pipeline can
          * leave an attachment of the render undeclared. The attachment is
          * still bound and holds earlier draws' values, so preserve it.
          * Only a target with no attachment needs nothing. */
         if (blend && blend[rt].color_attachment_unused) {
            preserves_color |= blend[rt].format != PIPE_FORMAT_NONE;
            continue;
         }
         bool written = nir->info.outputs_written &
                        BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
         preserves_color |= !written ||
            (blend && (!blend[rt].colormask || blend[rt].dynamic_enable));
      }
   }
   /* As on Apple8, promote direct accesses to SSA, keep small arrays in
    * registers, and lower larger dynamically indexed arrays to scratch. */
   nir_lower_array_deref_of_vec(nir, nir_var_function_temp, NULL,
                               nir_lower_indirect_array_deref_of_vec_load |
                               nir_lower_indirect_array_deref_of_vec_store);
   nir_lower_vars_to_ssa(nir);
   nir_lower_vars_to_scratch(nir, 128, glsl_get_natural_size_align_bytes,
                              glsl_get_word_size_align_bytes);
   nir_lower_indirect_derefs_to_if_else_trees(nir, nir_var_function_temp, ~0u);
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
      unsigned samples = blend ? MAX2(blend[0].samples, 1) : 1;
      bool single_sampled =
         samples == 1 || (blend && blend[0].multisample_disabled);
      if (single_sampled) {
         const nir_lower_single_sampled_options options = {0};
         nir_lower_single_sampled(nir, &options);
         nir->info.fs.uses_sample_shading = false;
      }
      nir_shader_intrinsics_pass(nir, apple9_lower_barycentric_coord,
                                 nir_metadata_control_flow, &single_sampled);
      nir_shader_intrinsics_pass(nir, apple9_expand_barycentric_coord,
                                 nir_metadata_control_flow, NULL);
      nir_shader_intrinsics_pass(nir, apple9_lower_interp_at_offset,
                                 nir_metadata_control_flow, NULL);
      agx_nir_lower_sample_intrinsics(nir,
         !(blend && blend[0].sample_mask_single_sample));
      nir_shader_intrinsics_pass(nir, apple9_lower_sample_state,
                                 nir_metadata_control_flow, (void *)blend);
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
      agx_nir_lower_monolithic_msaa(nir, samples);
      if (samples == 1 && blend && blend[0].clamp_early_depth &&
          nir->info.fs.early_fragment_tests) {
         nir_shader_intrinsics_pass(nir, apple9_clamp_early_depth,
                                    nir_metadata_control_flow, (void *)blend);
      }
      apple9_preserve_demotions(nir);
      nir_lower_vars_to_ssa(nir);
      nir_lower_all_phis_to_scalar(nir);
      nir_shader_intrinsics_pass(nir, apple9_lower_coverage_sysval,
                                 nir_metadata_control_flow, NULL);
      if (!color.valid) {
         if (reason)
            *reason = "Apple9 render requires FP32 color outputs and standard single-source blending";
         return false;
      }
   }
   nir_shader_intrinsics_pass(nir, apple9_lower_fragment_uniform,
                              nir_metadata_control_flow, NULL);
   nir_opt_dce(nir);
   apple9_lower_textures(nir);
   apple9_lower_idiv(nir);
   /* Format lowering can introduce powers and other high-level ALU ops. */
   apple9_opt_algebraic(nir);
   agx_nir_opt_apple9_bounds(nir);
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
      nir_shader_intrinsics_pass(nir, apple9_lower_unwritten_inputs,
                                 nir_metadata_control_flow, (void *)producer);
      if (!valid) {
         *reason = "Apple9 fragment window position requires FP32 XYZW components";
         return false;
      }
      nir_lower_alu_to_scalar(nir, NULL, NULL);
      nir_opt_copy_prop(nir);
      nir_opt_dce(nir);
   }
   uint32_t texture_mask = 0, sampler_mask = 0;
   bool uses_texel_fetch = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         if (!apple9_texture_supported(tex)) {
            if (agx_apple9_trace_enabled())
               nir_print_instr(instr, stderr);
            *reason = "Apple9 requires supported FP32 texture operations";
            return false;
         }
         apple9_texture_table_masks(tex, &texture_mask, &sampler_mask,
                                    &uses_texel_fetch);
      }
   }
   if (util_bitcount(texture_mask) > AGX_APPLE9_GRAPHICS_MAX_TEXTURES ||
       util_bitcount(sampler_mask) + uses_texel_fetch >
          AGX_APPLE9_GRAPHICS_MAX_SAMPLERS) {
      *reason = "Apple9 supports 32 textures and 17 physical samplers per stage";
      return false;
   }
   struct agx_apple9_varying_layout varyings = {0};
   if (!apple9_collect_varyings(nir, producer, &varyings, reason))
      return false;
   if (layout && layout->cull_distance_mask &&
       varyings.mask[VARYING_SLOT_CULL_PRIMITIVE])
      varyings.cull_distance_mask = true;
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      nir_shader_intrinsics_pass(nir, apple9_lower_flat_input,
                                 nir_metadata_control_flow, &varyings);
      nir_opt_cse(nir);
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
   struct apple9_buffer_map buffers = {0};
   if (!apple9_collect_buffer_map(nir, &buffers, reason))
      return false;
   bool valid_buffers = buffers.count <= AGX_APPLE9_MAX_GRAPHICS_BUFFERS;
   for (unsigned i = 0; i < buffers.count; ++i)
      valid_buffers &=
         buffers.resource[i].kind == AGX_APPLE9_COMPUTE_RESOURCE_SSBO
            ? buffers.resource[i].binding < 32
            : buffers.resource[i].kind == AGX_APPLE9_COMPUTE_RESOURCE_UBO;
   if (!valid_buffers) {
      if (reason)
         *reason =
            "Apple9 graphics requires at most 32 supported buffer bindings";
      return false;
   }
   if (agx_apple9_trace_enabled())
      nir_print_shader(nir, stderr);
   if (layout) {
      for (unsigned i = 0; i < ARRAY_SIZE(varyings.mask); ++i) {
         if (varyings.mask[i])
            varyings.group[i] = (layout->outputs_flat & BITFIELD64_BIT(i)) ? 1 :
               (layout->outputs_linear & BITFIELD64_BIT(i)) ? 2 : 0;
      }
   }
   if (varyings.cull_distance_mask)
      varyings.group[VARYING_SLOT_CULL_PRIMITIVE] = 1;
   bool writes_layer = nir->info.outputs_written &
                       (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT);
   if (nir->info.stage == MESA_SHADER_VERTEX && writes_layer) {
      apple9_lower_layer_viewport_output(nir, 4 + varyings.count +
         !!(nir->info.outputs_written & VARYING_BIT_PSIZ));
   }
   if (nir->info.stage == MESA_SHADER_VERTEX && nir->info.clip_distance_array_size) {
      if (nir->info.clip_distance_array_size > 8) {
         *reason = "Apple9 supports eight clip distances";
         return false;
      }
      const struct apple9_clip_distance_state state = {
         .base = 4 + varyings.count + !!(nir->info.outputs_written & VARYING_BIT_PSIZ) +
                 writes_layer,
         .enable = layout ? layout->clip_distance_enable : 0xff,
      };
      nir_shader_intrinsics_pass(nir, apple9_lower_clip_distance,
                                 nir_metadata_control_flow, (void *)&state);
   }
   if (!apple9_compile_dag(nir, out, NULL, &varyings, reason))
      return false;
   out->info.apple9_reads_tile |= preserves_color;
   return true;
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
   return nr_targets <= 8 &&
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
