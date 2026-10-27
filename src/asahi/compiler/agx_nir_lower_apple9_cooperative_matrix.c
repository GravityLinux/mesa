/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

/*
 * A deliberately bounded, arithmetic implementation, NOT a matrix ISA opcode.
 *
 * Metal2vk explicitly populates the two per-invocation matrix components.
 * Consequently their ownership is part of that producer's ABI:
 *
 *   row = ((lane >> 2) & 4) | ((lane >> 1) & 3)
 *   col = ((lane >> 1) & 4) | ((lane << 1) & 2) | component
 *
 * This is the SAME layout for A, B and C, for both float sizes. In particular
 * B must not be silently transposed as in the subgroup8 lavapipe lowering.
 * All 32 invocations must participate. No shared memory or barriers are added.
 *
 * Use constant-index read_invocation, not a divergent broadcast masquerading
 * as a shuffle. Apple9's native path currently accepts 32-bit broadcasts.
 * Widen half operands BEFORE communicating, and accumulate using FP32 FMA.
 */

static bool
fail(const char **reason, const char *message)
{
   if (reason)
      *reason = message;
   return false;
}

static bool
supported_desc(const struct glsl_cmat_description *d)
{
   return d->scope == SCOPE_SUBGROUP && d->rows == 8 && d->cols == 8 &&
          (d->element_type == GLSL_TYPE_FLOAT ||
           d->element_type == GLSL_TYPE_FLOAT16) &&
          (d->use == GLSL_CMAT_USE_A || d->use == GLSL_CMAT_USE_B ||
           (d->use == GLSL_CMAT_USE_ACCUMULATOR &&
            d->element_type == GLSL_TYPE_FLOAT));
}

static bool
contains_cmat(const struct glsl_type *type)
{
   if (glsl_type_is_array(type))
      return contains_cmat(glsl_get_array_element(type));
   if (glsl_type_is_struct_or_ifc(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); ++i) {
         if (contains_cmat(glsl_get_struct_field(type, i)))
            return true;
      }
      return false;
   }
   return glsl_type_is_cmat(type);
}

static bool
supported_type(const struct glsl_type *type)
{
   if (!contains_cmat(type))
      return true;
   if (glsl_type_is_array(type))
      return glsl_get_explicit_stride(type) == 0 &&
             supported_type(glsl_get_array_element(type));
   return glsl_type_is_cmat(type) &&
          supported_desc(glsl_get_cmat_description(type));
}

static const struct glsl_cmat_description *
src_desc(nir_src src)
{
   return glsl_get_cmat_description(nir_src_as_deref(src)->type);
}

static bool
is_cmat_intrinsic(nir_intrinsic_instr *intr)
{
   return strncmp(nir_intrinsic_infos[intr->intrinsic].name, "cmat_", 5) == 0;
}

static bool
supported_intrinsic(nir_intrinsic_instr *intr)
{
   if (!is_cmat_intrinsic(intr))
      return true;

   /* Validate descriptors before inspecting operand roles below. */
   for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i) {
      if (nir_def_is_deref(intr->src[i].ssa)) {
         const struct glsl_type *type = nir_src_as_deref(intr->src[i])->type;
         if (glsl_type_is_cmat(type) &&
             !supported_desc(glsl_get_cmat_description(type)))
            return false;
      }
   }

   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_construct:
   case nir_intrinsic_cmat_copy:
   case nir_intrinsic_cmat_extract:
   case nir_intrinsic_cmat_insert:
      return true;
   case nir_intrinsic_cmat_length:
   case nir_intrinsic_cmat_get_coordinate: {
      struct glsl_cmat_description d = nir_intrinsic_cmat_desc(intr);
      return supported_desc(&d);
   }
   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store: {
      bool load = intr->intrinsic == nir_intrinsic_cmat_load;
      nir_deref_instr *ptr = nir_src_as_deref(intr->src[load]);
      enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);
      return (layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR ||
              layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) &&
             (glsl_type_is_scalar(ptr->type) || glsl_type_is_vector(ptr->type)) &&
             (glsl_get_bit_size(ptr->type) == 16 ||
              glsl_get_bit_size(ptr->type) == 32) &&
             glsl_get_bit_size(ptr->type) ==
                glsl_base_type_bit_size(src_desc(intr->src[!load])->element_type) &&
             nir_deref_mode_is_one_of(ptr, nir_var_mem_ssbo |
                                             nir_var_mem_global |
                                             nir_var_mem_shared);
   }
   case nir_intrinsic_cmat_muladd: {
      const struct glsl_cmat_description *d = src_desc(intr->src[0]);
      const struct glsl_cmat_description *a = src_desc(intr->src[1]);
      const struct glsl_cmat_description *b = src_desc(intr->src[2]);
      const struct glsl_cmat_description *c = src_desc(intr->src[3]);
      return !nir_intrinsic_saturate(intr) &&
             !nir_intrinsic_cmat_signed_mask(intr) &&
             a->use == GLSL_CMAT_USE_A && b->use == GLSL_CMAT_USE_B &&
             c->use == GLSL_CMAT_USE_ACCUMULATOR &&
             d->use == GLSL_CMAT_USE_ACCUMULATOR &&
             a->element_type == b->element_type;
   }
   case nir_intrinsic_cmat_convert:
      return !nir_intrinsic_saturate(intr) &&
             !nir_intrinsic_cmat_signed_mask(intr);
   case nir_intrinsic_cmat_bitcast:
      return src_desc(intr->src[0])->element_type ==
             src_desc(intr->src[1])->element_type;
   case nir_intrinsic_cmat_unary_op:
      return nir_intrinsic_alu_op(intr) == nir_op_fneg;
   case nir_intrinsic_cmat_binary_op:
      return nir_intrinsic_alu_op(intr) == nir_op_fadd ||
             nir_intrinsic_alu_op(intr) == nir_op_fsub ||
             nir_intrinsic_alu_op(intr) == nir_op_fmul ||
             nir_intrinsic_alu_op(intr) == nir_op_fdiv;
   case nir_intrinsic_cmat_scalar_op:
      return nir_intrinsic_alu_op(intr) == nir_op_fmul;
   default:
      return false;
   }
}

/* Atomic preflight: an unsupported shader is left entirely unchanged. */
static bool
validate_shader(nir_shader *shader, unsigned subgroup_size, const char **reason)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE || subgroup_size != 32)
      return fail(reason, "Apple9 cooperative matrices require compute subgroup32");

   nir_foreach_variable_in_shader(var, shader) {
      if (!supported_type(var->type) ||
          (contains_cmat(var->type) &&
           (var->data.mode != nir_var_shader_temp || var->constant_initializer)))
         return fail(reason, "Unsupported Apple9 cooperative matrix variable");
   }

   nir_foreach_function_impl(impl, shader) {
      if (impl->function->num_params != 0)
         return fail(reason, "Inline functions before Apple9 matrix lowering");
      nir_foreach_function_temp_variable(var, impl) {
         if (!supported_type(var->type) || (contains_cmat(var->type) &&
                                           var->constant_initializer))
            return fail(reason, "Unsupported Apple9 cooperative matrix local");
      }
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_cmat_call ||
                instr->type == nir_instr_type_call)
               return fail(reason, "Unsupported call in Apple9 matrix lowering");
            if (instr->type == nir_instr_type_intrinsic &&
                !supported_intrinsic(nir_instr_as_intrinsic(instr)))
               return fail(reason, "Unsupported Apple9 cooperative matrix operation");
            if (instr->type == nir_instr_type_deref) {
               nir_deref_instr *deref = nir_instr_as_deref(instr);
               if (!supported_type(deref->type) ||
                   (contains_cmat(deref->type) &&
                    (deref->deref_type != nir_deref_type_var &&
                     deref->deref_type != nir_deref_type_array)))
                  return fail(reason, "Unsupported Apple9 cooperative matrix deref");
            }
         }
      }
   }
   return true;
}

static const struct glsl_type *
lower_type(const struct glsl_type *type)
{
   if (glsl_type_is_cmat(type))
      return glsl_vector_type(glsl_get_cmat_description(type)->element_type, 2);
   if (glsl_type_is_array(type) && contains_cmat(type))
      return glsl_array_type(lower_type(glsl_get_array_element(type)),
                             glsl_get_length(type), 0);
   return type;
}

static nir_def *
load_matrix(nir_builder *b, nir_src src)
{
   return nir_build_load_deref(
      b, 2, glsl_base_type_bit_size(src_desc(src)->element_type), src.ssa, 0);
}

static void
store_matrix(nir_builder *b, nir_src dst, nir_def *value)
{
   nir_store_deref(b, nir_src_as_deref(dst), value, 3);
}

static nir_def *
matrix_row(nir_builder *b, nir_def *lane)
{
   return nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 2), 4),
                     nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3));
}

static nir_def *
matrix_col(nir_builder *b, nir_def *lane)
{
   return nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 4),
                     nir_iand_imm(b, nir_ishl_imm(b, lane, 1), 2));
}

static unsigned
matrix_lane(unsigned row, unsigned col)
{
   return ((row & 3) << 1) | ((row & 4) << 2) |
          ((col & 2) >> 1) | ((col & 4) << 1);
}

static void
lower_muladd(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *a = nir_f2f32(b, load_matrix(b, intr->src[1]));
   nir_def *mat_b = nir_f2f32(b, load_matrix(b, intr->src[2]));
   nir_def *c = load_matrix(b, intr->src[3]);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *row = matrix_row(b, lane);
   nir_def *col = matrix_col(b, lane);
   nir_def *acc[2] = {nir_channel(b, c, 0), nir_channel(b, c, 1)};

   /* Never allow reassociation of the eight FP32 accumulation steps. */
   b->fp_math_ctrl = 0;
   for (unsigned k = 0; k < 8; ++k) {
      nir_def *a_part = nir_channel(b, a, k & 1);
      nir_def *a_value = NULL;
      for (unsigned r = 0; r < 8; ++r) {
         nir_def *v = nir_read_invocation(
            b, a_part, nir_imm_int(b, matrix_lane(r, k)));
         a_value = r == 0 ? v :
            nir_bcsel(b, nir_ieq_imm(b, row, r), v, a_value);
      }

      for (unsigned j = 0; j < 2; ++j) {
         nir_def *b_part = nir_channel(b, mat_b, j);
         nir_def *b_value = NULL;
         for (unsigned col_pair = 0; col_pair < 8; col_pair += 2) {
            nir_def *v = nir_read_invocation(
               b, b_part, nir_imm_int(b, matrix_lane(k, col_pair)));
            b_value = col_pair == 0 ? v :
               nir_bcsel(b, nir_ieq_imm(b, col, col_pair), v, b_value);
         }
         acc[j] = nir_ffma(b, a_value, b_value, acc[j]);
      }
   }
   store_matrix(b, intr->src[0], nir_vec2(b, acc[0], acc[1]));
}

static void
lower_load_store(nir_builder *b, nir_intrinsic_instr *intr)
{
   bool load = intr->intrinsic == nir_intrinsic_cmat_load;
   const struct glsl_cmat_description *desc = src_desc(intr->src[!load]);
   nir_deref_instr *ptr = nir_src_as_deref(intr->src[load]);
   unsigned scalar_bytes = glsl_base_type_bit_size(desc->element_type) / 8;
   unsigned pointer_bytes = glsl_get_bit_size(ptr->type) / 8 *
                            glsl_get_vector_elements(ptr->type);
   unsigned idx_bits = ptr->def.bit_size;
   nir_def *stride = nir_u2uN(b, intr->src[2].ssa, idx_bits);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *row = nir_u2uN(b, matrix_row(b, lane), idx_bits);
   nir_def *col = nir_u2uN(b, matrix_col(b, lane), idx_bits);
   nir_def *value = load ? NULL : load_matrix(b, intr->src[1]);
   nir_def *parts[2];

   /* SPIR-V stride is in POINTER elements, including vector width. Only the
    * contiguous minor coordinate is in matrix scalar elements. Ignore the
    * array's ArrayStride for this operation, but not the incoming base offset.
    */
   ptr = nir_build_deref_cast(b, &ptr->def, ptr->modes, ptr->type, pointer_bytes);
   for (unsigned i = 0; i < 2; ++i) {
      nir_def *major = row;
      nir_def *minor = nir_iadd_imm(b, col, i);
      if (nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR)
         SWAP(major, minor);
      nir_deref_instr *element = nir_build_deref_ptr_as_array(
         b, ptr, nir_imul(b, major, stride));
      element = nir_build_deref_cast(b, &element->def, element->modes,
                                     glsl_scalar_type(desc->element_type),
                                     scalar_bytes);
      element = nir_build_deref_ptr_as_array(b, element, minor);
      if (load)
         parts[i] = nir_load_deref(b, element);
      else
         nir_store_deref(b, element, nir_channel(b, value, i), 1);
   }
   if (load)
      store_matrix(b, intr->src[0], nir_vec2(b, parts[0], parts[1]));
}

static void
lower_intrinsic(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *value = NULL;
   b->fp_math_ctrl = 0;
   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_muladd:
      lower_muladd(b, intr);
      break;
   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store:
      lower_load_store(b, intr);
      break;
   case nir_intrinsic_cmat_construct:
      value = nir_replicate(b, intr->src[1].ssa, 2);
      break;
   case nir_intrinsic_cmat_copy:
   case nir_intrinsic_cmat_bitcast:
      value = load_matrix(b, intr->src[1]);
      break;
   case nir_intrinsic_cmat_convert:
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      value = nir_f2fN(b, load_matrix(b, intr->src[1]),
                      glsl_base_type_bit_size(src_desc(intr->src[0])->element_type));
      break;
   case nir_intrinsic_cmat_unary_op:
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      value = nir_build_alu1(b, nir_intrinsic_alu_op(intr),
                            load_matrix(b, intr->src[1]));
      break;
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op:
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      value = nir_build_alu2(b, nir_intrinsic_alu_op(intr),
                            load_matrix(b, intr->src[1]),
                            intr->intrinsic == nir_intrinsic_cmat_scalar_op ?
                               intr->src[2].ssa : load_matrix(b, intr->src[2]));
      break;
   case nir_intrinsic_cmat_insert:
      value = nir_vector_insert(b, load_matrix(b, intr->src[2]),
                                intr->src[1].ssa, intr->src[3].ssa);
      break;
   case nir_intrinsic_cmat_extract:
      value = nir_vector_extract(b, load_matrix(b, intr->src[0]), intr->src[1].ssa);
      nir_def_replace(&intr->def, value);
      return;
   case nir_intrinsic_cmat_length:
      nir_def_replace(&intr->def, nir_imm_int(b, 2));
      return;
   case nir_intrinsic_cmat_get_coordinate: {
      nir_def *lane = nir_load_subgroup_invocation(b);
      value = nir_vec2(b, matrix_row(b, lane),
                      nir_iadd(b, matrix_col(b, lane), intr->src[0].ssa));
      nir_def_replace(&intr->def, value);
      return;
   }
   default:
      UNREACHABLE("Preflight accepted an unsupported cooperative matrix operation");
   }
   if (value)
      store_matrix(b, intr->src[0], value);
   nir_instr_remove(&intr->instr);
}

bool
agx_nir_lower_apple9_cooperative_matrix(nir_shader *shader,
                                        unsigned subgroup_size,
                                        const char **reason)
{
   if (reason)
      *reason = NULL;
   if (!shader->info.cs.has_cooperative_matrix)
      return true;
   if (!validate_shader(shader, subgroup_size, reason))
      return false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!is_cmat_intrinsic(intr))
               continue;
            b.cursor = nir_before_instr(instr);
            lower_intrinsic(&b, intr);
         }
      }
   }

   /* Keep the descriptors available until EVERY use has been lowered,
    * including uses across blocks. Then fix variables and all deref types.
    */
   nir_foreach_variable_in_shader(var, shader)
      var->type = lower_type(var->type);
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_function_temp_variable(var, impl)
         var->type = lower_type(var->type);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref) {
               nir_deref_instr *deref = nir_instr_as_deref(instr);
               deref->type = lower_type(deref->type);
            }
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   shader->info.cs.has_cooperative_matrix = false;
   return true;
}
