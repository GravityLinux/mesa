/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9_softfp64.h"
#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_softfloat.h"
#include "compiler/spirv/nir_spirv.h"

#include "hk_float64_spv.h"

void
hk_apple9_softfp64_init(struct hk_apple9_softfp64 *cache)
{
   cache->library = NULL;
   simple_mtx_init(&cache->lock, mtx_plain);
}

void
hk_apple9_softfp64_finish(struct hk_apple9_softfp64 *cache)
{
   ralloc_free(cache->library);
   cache->library = NULL;
   simple_mtx_destroy(&cache->lock);
}

static nir_shader *
load_library(void)
{
   const struct spirv_to_nir_options options = {
      .create_library = true,
      .skip_os_break_in_debug_build = true,
      .ignore_replacement = true,
   };
   nir_shader *nir =
      spirv_to_nir(hk_float64_spv, ARRAY_SIZE(hk_float64_spv), NULL,
                   MESA_SHADER_COMPUTE, "main", &options, &agx_nir_options);
   if (!nir)
      return NULL;

   /* As in Turnip, initialize locals before inlining their callers. Preserve
    * exported (mangled) names for nir_lower_doubles' SPIR-V lookup. */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_exported(nir);
   nir_lower_constant_to_temp(nir);
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_all);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_deref);
   nir_validate_shader(nir, "Apple9 imported Mesa float64 library");
   return nir;
}

static bool
is_double(const nir_alu_instr *alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   if (alu->def.bit_size == 64 &&
       nir_alu_type_get_base_type(info->output_type) == nir_type_float)
      return true;

   for (unsigned i = 0; i < info->num_inputs; ++i) {
      if (alu->src[i].src.ssa->bit_size == 64 &&
          nir_alu_type_get_base_type(info->input_types[i]) == nir_type_float)
         return true;
   }
   return false;
}

bool
hk_apple9_nir_uses_double(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                is_double(nir_instr_as_alu(instr)))
               return true;
         }
      }
   }
   return false;
}

/* Keep this list aligned with nir_lower_double_ops.c. In particular ffma is
 * NOT fmad: Mesa's __fmad64 rounds multiply and add separately. */
static const char *
function_name(const nir_alu_instr *alu)
{
   switch (alu->op) {
   case nir_op_fadd: return "__fadd64(u641;u641;";
   case nir_op_fmul: return "__fmul64(u641;u641;";
   case nir_op_fdiv: return "__fdiv64(u641;u641;";
   case nir_op_ffma: return "__ffma64(u641;u641;u641;";
   case nir_op_fmad: return "__fmad64(u641;u641;u641;";
   case nir_op_fsqrt: return "__fsqrt64(u641;";
   case nir_op_fabs: return "__fabs64(u641;";
   case nir_op_fneg: return "__fneg64(u641;";
   case nir_op_fround_even: return "__fround64(u641;";
   case nir_op_ftrunc: return "__ftrunc64(u641;";
   case nir_op_ffloor: return "__ffloor64(u641;";
   case nir_op_ffract: return "__ffract64(u641;";
   case nir_op_fsign: return "__fsign64(u641;";
   case nir_op_feq: return "__feq64(u641;u641;";
   case nir_op_fneu: return "__fneu64(u641;u641;";
   case nir_op_flt: return "__flt64(u641;u641;";
   case nir_op_fge: return "__fge64(u641;u641;";
   case nir_op_fmin: return "__fmin64(u641;u641;";
   case nir_op_fmax: return "__fmax64(u641;u641;";
   case nir_op_fsat: return "__fsat64(u641;";
   case nir_op_fisfinite: return "__fisfinite64(u641;";
   case nir_op_b2f64: return "__bool_to_fp64(b1;";
   case nir_op_f2f64:
      return alu->src[0].src.ssa->bit_size == 32 ? "__fp32_to_fp64(f1;" : NULL;
   case nir_op_f2f32: return "__fp64_to_fp32(u641;";
   case nir_op_f2i32: return "__fp64_to_int(u641;";
   case nir_op_f2u32: return "__fp64_to_uint(u641;";
   case nir_op_f2i64: return "__fp64_to_int64(u641;";
   case nir_op_f2u64: return "__fp64_to_uint64(u641;";
   case nir_op_i2f64:
      return alu->src[0].src.ssa->bit_size == 64 ? "__int64_to_fp64(i641;" :
             alu->src[0].src.ssa->bit_size == 32 ? "__int_to_fp64(i1;" : NULL;
   case nir_op_u2f64:
      return alu->src[0].src.ssa->bit_size == 64 ? "__uint64_to_fp64(u641;" :
             alu->src[0].src.ssa->bit_size == 32 ? "__uint_to_fp64(u1;" : NULL;
   default: return NULL;
   }
}

static bool
is_local_library_op(const nir_instr *instr, UNUSED const void *data)
{
   return instr->type == nir_instr_type_alu &&
          (nir_instr_as_alu(instr)->op == nir_op_fsqrt ||
           nir_instr_as_alu(instr)->op == nir_op_fdiv ||
           nir_instr_as_alu(instr)->op == nir_op_ffma) &&
          nir_instr_as_alu(instr)->def.bit_size == 64;
}

static nir_def *
lower_local_library_op(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_function *func =
      nir_shader_get_function_for_name(data, function_name(alu));
   return nir_lower_softfloat_func(b, alu, func, glsl_uint64_t_type());
}

bool
hk_apple9_lower_softfp64(struct hk_apple9_softfp64 *cache, nir_shader *nir,
                        const char **reason)
{
   *reason = NULL;
   bool needed = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu ||
                !is_double(nir_instr_as_alu(instr)))
               continue;
            needed = true;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (!function_name(alu)) {
               *reason = nir_op_infos[alu->op].name;
               return false;
            }
         }
      }
   }
   if (!needed)
      return true;

   simple_mtx_lock(&cache->lock);
   if (!cache->library)
      cache->library = load_library();
   if (!cache->library) {
      *reason = "failed to import embedded Mesa float64 library";
      simple_mtx_unlock(&cache->lock);
      return false;
   }

   /* Generic lowering asserts on missing functions. Check the actual exports
    * first so a failed or incompatible library remains a fallible compile. */
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu ||
                !is_double(nir_instr_as_alu(instr)))
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            nir_function *func = nir_shader_get_function_for_name(
               cache->library, function_name(alu));
            if (!func || !func->impl ||
                func->num_params != nir_op_infos[alu->op].num_inputs + 1) {
               *reason = "missing Mesa float64 library export";
               simple_mtx_unlock(&cache->lock);
               return false;
            }
         }
      }
   }

   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   /* These exports have no generic software mapping. Inline them BEFORE
    * generic lowering can split fdiv or lose the fused semantics of ffma. */
   nir_foreach_function_impl(impl, nir) {
      if (nir_function_impl_lower_instructions(impl, is_local_library_op,
                                              lower_local_library_op,
                                              cache->library)) {
         nir_index_ssa_defs(impl);
         nir_progress(true, impl, nir_metadata_none);
         nir_opt_deref_impl(impl);
      }
   }
   NIR_PASS(_, nir, nir_lower_doubles, cache->library,
            nir_lower_fp64_full_software);
   simple_mtx_unlock(&cache->lock);

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   nir_validate_shader(nir, "Apple9 Mesa software double lowering");
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                is_double(nir_instr_as_alu(instr))) {
               *reason = nir_op_infos[nir_instr_as_alu(instr)->op].name;
               return false;
            }
         }
      }
   }
   return true;
}
