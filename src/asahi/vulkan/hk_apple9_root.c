/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9_root.h"
#include "hk_cmd_buffer.h"
#include "nir_builder.h"

static bool
lower_set(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   unsigned set = nir_intrinsic_desc_set(intr);
   nir_def *address = nir_iadd_imm(
      b, nir_load_root_agx(b), hk_root_descriptor_offset(sets[set]));
   nir_def_replace(&intr->def, nir_load_global_constant(
      b, 1, 64, address, .align_mul = 8, .access = ACCESS_CAN_SPECULATE));
   return true;
}

static bool
lower_root(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_root_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *root = nir_load_ubo(
      b, 1, 64, nir_imm_int(b, HK_APPLE9_ROOT_BINDING),
      nir_imm_int(b, hk_root_descriptor_offset(root_desc_addr)),
      .align_mul = 8, .range = 8);
   nir_def_replace(&intr->def, root);
   return true;
}

bool
hk_nir_lower_apple9_root(nir_shader *nir)
{
   bool progress = nir_shader_intrinsics_pass(
      nir, lower_set, nir_metadata_control_flow, NULL);
   return nir_shader_intrinsics_pass(
             nir, lower_root, nir_metadata_control_flow, NULL) || progress;
}
