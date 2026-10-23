/*
 * SPDX-License-Identifier: MIT
 */

/* Clamps the depth comparison reference of shadow lookups to [0, 1].
 *
 * GL clamps the reference for fixed-point depth textures. Drivers that store
 * fixed-point depth in a floating-point format (pipe_caps.depth_unorm_emulated)
 * lose the clamp the hardware would apply, so the state tracker applies it to
 * every shadow sampler in clamp_mask. Projected lookups must be lowered first,
 * since the clamp applies to the projected reference.
 *
 * This should run after the samplers are lowered.
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "st_nir.h"

/* Returns the range of sampler units the lookup may use, or false if it is
 * unknown (bindless).
 */
static bool
get_sampler_range(nir_tex_instr *tex, unsigned *first, unsigned *count)
{
   if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) >= 0)
      return false;

   int idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (idx < 0)
      idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (idx < 0) {
      if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset) >= 0)
         return false;
      *first = tex->sampler_index;
      *count = 1;
      return true;
   }

   nir_deref_instr *deref = nir_src_as_deref(tex->src[idx].src);
   unsigned offset = 0;
   bool dynamic = false;
   while (deref->deref_type != nir_deref_type_var) {
      if (deref->deref_type != nir_deref_type_array)
         return false;
      unsigned elems = MAX2(glsl_get_aoa_size(deref->type), 1);
      if (nir_src_is_const(deref->arr.index))
         offset += nir_src_as_uint(deref->arr.index) * elems;
      else
         dynamic = true;
      deref = nir_deref_instr_parent(deref);
   }

   *first = deref->var->data.binding;
   if (dynamic) {
      *count = MAX2(glsl_get_aoa_size(deref->var->type), 1);
   } else {
      *first += offset;
      *count = 1;
   }
   return true;
}

static bool
clamp_shadow_ref(nir_builder *b, nir_tex_instr *tex, void *data)
{
   uint32_t clamp_mask = *(uint32_t *)data;
   int comp = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   if (comp < 0)
      return false;

   unsigned first, count;
   if (!get_sampler_range(tex, &first, &count) || first + count > 32)
      return false;

   /* with a dynamic index, clamp only if every sampler it may use needs it */
   uint32_t range = BITFIELD_RANGE(first, count);
   if ((clamp_mask & range) != range)
      return false;

   assert(nir_tex_instr_src_index(tex, nir_tex_src_projector) < 0);
   b->cursor = nir_before_instr(&tex->instr);
   nir_src_rewrite(&tex->src[comp].src, nir_fsat(b, tex->src[comp].src.ssa));
   return true;
}

bool
st_nir_lower_shadow_ref_clamp(nir_shader *shader, uint32_t clamp_mask)
{
   return nir_shader_tex_pass(shader, clamp_shadow_ref,
                              nir_metadata_control_flow, &clamp_mask);
}
