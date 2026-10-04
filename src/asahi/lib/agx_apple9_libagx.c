/*
 * Copyright 2026 Niklas Sheth
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_libagx.h"
#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "libagx_shaders.h"

/* Apple9 compute has no preamble uniforms; kernel inputs come from UBO 0. */
static bool
lower_kernel_input(nir_builder *b, nir_intrinsic_instr *intr,
                   UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_preamble)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   unsigned offset = nir_intrinsic_base(intr) * 2;
   nir_def_replace(&intr->def,
      nir_load_ubo(b, intr->def.num_components, intr->def.bit_size,
                   nir_imm_int(b, 0), nir_imm_int(b, offset),
                   .align_mul = 2, .range_base = offset,
                   .range = intr->def.num_components * intr->def.bit_size / 8));
   return true;
}

nir_shader *
agx_apple9_libagx_nir(unsigned program, void *mem_ctx)
{
   assert(program < LIBAGX_NUM_PROGRAMS);
   const uint32_t *serialized = libagx_nir[program];
   struct blob_reader reader;
   blob_reader_init(&reader, serialized + 1, serialized[0]);
   nir_shader *nir = nir_deserialize(mem_ctx, &agx_nir_options, &reader);
   nir_shader_intrinsics_pass(nir, lower_kernel_input,
                              nir_metadata_control_flow, NULL);
   return nir;
}
