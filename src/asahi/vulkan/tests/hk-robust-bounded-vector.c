/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "asahi/compiler/agx_compile.h"
#include "asahi/vulkan/hk_shader.h"
#include "compiler/nir/nir_builder.h"

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return false; \
} } while (0)

static void
fold(nir_shader *nir)
{
   bool progress;
   do {
      progress = nir_opt_constant_folding(nir);
      progress |= nir_opt_algebraic(nir);
      progress |= nir_opt_copy_prop(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);
}

static bool
is_loaded_channel(nir_scalar s, unsigned c)
{
   if (!nir_scalar_is_intrinsic(s) || s.comp != c)
      return false;
   nir_intrinsic_op op = nir_scalar_intrinsic_op(s);
   return op == nir_intrinsic_load_global_constant || op == nir_intrinsic_load_global;
}

/* A robust vector of 32-bit components at `offset` in a buffer of `bound`
 * bytes returns every component that is entirely in bounds and zero for the
 * rest (robustness2). Merged robust loads must not zero their in-bounds
 * components because a later one is out of bounds. The offset and bound are
 * either constants or values loaded at run time; in the latter case they are
 * replaced by constants after lowering, so folding resolves the checks.
 */
static bool
check(unsigned components, unsigned offset, unsigned bound, bool dynamic,
      bool soft_fault)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "robust_vector");
   nir_def *runtime = nir_load_global(&b, 2, 32, nir_imm_int64(&b, 0x4000),
                                      .align_mul = 8);
   nir_def *off = dynamic ? nir_channel(&b, runtime, 0) : nir_imm_int(&b, offset);
   nir_def *bnd = dynamic ? nir_channel(&b, runtime, 1) : nir_imm_int(&b, bound);
   nir_def *value = nir_load_global_constant_bounded(
      &b, components, 32, nir_imm_int64(&b, 0x10000), off, bnd,
      .align_mul = 4 * util_next_power_of_two(components), .align_offset = 0);
   nir_store_global(&b, value, nir_imm_int64(&b, 0x8000),
                    .write_mask = BITFIELD_MASK(components), .align_mul = 4);

   CHECK(hk_lower_bounded_global_loads(b.shader, soft_fault, true));

   if (dynamic) {
      b.cursor = nir_after_instr(nir_def_instr(runtime));
      nir_def_rewrite_uses(runtime, nir_imm_ivec2(&b, offset, bound));
   }
   fold(b.shader);

   nir_intrinsic_instr *store = NULL;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_global)
            store = nir_instr_as_intrinsic(instr);
      }
   }
   CHECK(store);
   for (unsigned c = 0; c < components; ++c) {
      nir_scalar s = nir_scalar_resolved(store->src[0].ssa, c);
      bool in_bounds = offset + 4 * (c + 1) <= bound;
      /* Without soft faults an out-of-bounds first component reads the zero
       * page through the redirected address instead of selecting zero. */
      if (!soft_fault && c == 0 && !in_bounds)
         continue;
      if (in_bounds) {
         CHECK(is_loaded_channel(s, c));
      } else {
         CHECK(nir_scalar_is_const(s) && nir_scalar_as_uint(s) == 0);
      }
   }
   ralloc_free(b.shader);
   return true;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   static const struct {
      unsigned components, offset, bound;
   } cases[] = {
      {4, 0, 12},  /* three of four in bounds: robustness2 len_12 */
      {2, 8, 12},  /* the upper half of that vec4 merged separately */
      {4, 0, 16},  /* entirely in bounds */
      {4, 0, 4},
      {4, 16, 12}, /* entirely out of bounds */
      {2, 0, 4},
      {3, 0, 8},
   };
   bool ok = true;
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      for (unsigned dynamic = 0; dynamic < 2; ++dynamic) {
         for (unsigned soft = 0; soft < 2; ++soft) {
            if (!check(cases[i].components, cases[i].offset, cases[i].bound,
                       dynamic, soft)) {
               fprintf(stderr, "failed: vec%u offset %u bound %u dynamic %u soft_fault %u\n",
                       cases[i].components, cases[i].offset, cases[i].bound,
                       dynamic, soft);
               ok = false;
            }
         }
      }
   }
   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}
