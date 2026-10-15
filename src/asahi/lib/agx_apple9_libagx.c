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

static void
invalid_blob(const char *reason)
{
   fprintf(stderr,
           "Apple9 libagx: %s; rebuild asahi_clc and regenerate "
           "libagx shaders\n",
           reason);
   abort();
}

nir_shader *
agx_apple9_libagx_nir(unsigned program, void *mem_ctx)
{
   assert(program < LIBAGX_NUM_PROGRAMS);
   const uint32_t *serialized = libagx_nir[program];
   return agx_apple9_libagx_deserialize(serialized + 1, serialized[0], mem_ctx);
}

nir_shader *
agx_apple9_libagx_deserialize(const void *data, size_t size, void *mem_ctx)
{
   if (size < sizeof(struct agx_apple9_libagx_metadata))
      invalid_blob("truncated kernel metadata");

   struct blob_reader reader;
   blob_reader_init(&reader, data, size);
   struct agx_apple9_libagx_metadata metadata;
   blob_copy_bytes(&reader, &metadata, sizeof(metadata));
   /* An old tool emits either shader_info or unversioned metadata here.
    * Reject it before the function reader interprets any of its contents.
    */
   if (metadata.format != AGX_APPLE9_LIBAGX_FORMAT)
      invalid_blob("incompatible kernel blob format");

   nir_function *function =
      nir_deserialize_function(mem_ctx, &agx_nir_options, &reader);
   if (reader.overrun || reader.current != reader.end || !function)
      invalid_blob("invalid kernel function stream");
   nir_shader *nir = function->shader;
   nir->info.stage = MESA_SHADER_COMPUTE;
   nir->info.internal = true;
   for (unsigned i = 0; i < 3; ++i)
      nir->info.workgroup_size[i] = metadata.workgroup_size[i];
   nir->info.shared_size = metadata.shared_size;
   nir->scratch_size = metadata.scratch_size;
   nir->info.float_controls_execution_mode = metadata.float_controls;
   nir->info.api_subgroup_size = metadata.api_subgroup_size;
   nir->info.min_subgroup_size = metadata.min_subgroup_size;
   nir->info.max_subgroup_size = metadata.max_subgroup_size;
   nir->info.workgroup_size_variable = metadata.variable_local_size;
   nir_shader_intrinsics_pass(nir, lower_kernel_input,
                              nir_metadata_control_flow, NULL);
   nir_shader_gather_info(nir, function->impl);
   return nir;
}
