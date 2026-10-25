/* SPDX-License-Identifier: MIT
 * Host-only declaration/interface gate on unmodified application SPIR-V.
 * A false result proves default-native rejection. A true result needs ordinary
 * pipeline compilation, upload and draw-state validation before admission.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/spirv/nir_spirv.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_apple9_mesh.h"
#include "hk_apple9_mesh.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

static void
json_string(const char *s)
{
   if (!s) {
      fputs("null", stdout);
      return;
   }
   putchar('"');
   for (; *s; ++s) {
      if (*s == '"' || *s == '\\')
         putchar('\\');
      if ((unsigned char)*s < 32)
         printf("\\u%04x", (unsigned char)*s);
      else
         putchar(*s);
   }
   putchar('"');
}

static void
layout(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   *align = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   *size = *align * glsl_get_vector_elements(type);
}

static unsigned
io_slots(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

int main(int argc, char **argv)
{
   bool research = argc == 5 && !strcmp(argv[4], "--research");
   if (argc != 4 && !research) {
      fprintf(stderr, "usage: %s mesh|task entry file.spv [--research]\n", argv[0]);
      return 2;
   }
   if (!research && (getenv("HK_APPLE9_NATIVE_MESH") ||
       (getenv("HK_APPLE9_NATIVE_INDEX") &&
        strcmp(getenv("HK_APPLE9_NATIVE_INDEX"), "1"))))
      return 2;
   FILE *f = fopen(argv[3], "rb");
   if (!f) return 2;
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   rewind(f);
   uint32_t *words = malloc(size);
   if (size < 20 || size % 4 || !words || fread(words, 1, size, f) != size)
      return 2;
   fclose(f);
   glsl_type_singleton_init_or_ref();
   const struct spirv_to_nir_options opts = {
      .environment = NIR_SPIRV_VULKAN,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .shared_addr_format = nir_address_format_32bit_offset,
      .task_payload_addr_format = nir_address_format_32bit_offset,
   };
   mesa_shader_stage stage = !strcmp(argv[1], "mesh") ? MESA_SHADER_MESH : MESA_SHADER_TASK;
   nir_shader *nir = spirv_to_nir(words, size / 4, NULL, stage, argv[2],
                                 &opts, &agx_nir_options);
   free(words);
   if (!nir) return 3;
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_deref);
   nir_remove_non_entrypoints(nir);
   /* Match the common Vulkan import's interface normalization. Gathering
    * metadata on unsplit blocks loses members whose parent has no location.
    * Explicit IO also preserves per-primitive and 16-bit interface metadata.
    */
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out | nir_var_system_value, NULL);
   nir_gather_clip_cull_distance_sizes_from_vars(nir);
   NIR_PASS(_, nir, nir_merge_clip_cull_distance_vars);
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            io_slots, nir_lower_io_lower_64bit_to_32);
   nir->info.io_lowered = true;
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
            nir_var_mem_shared | nir_var_mem_task_payload, layout);
   /* Fold address arithmetic introduced by explicit IO before distinguishing
    * dynamic slot offsets from the independent vertex/primitive index. */
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   /* The 16-bit location mask tracks packed varying locations, not every
    * narrow scalar in an unlinked interface. Include the actual IO width. */
   bool narrow_outputs = nir->info.outputs_written_16bit != 0;
   unsigned dynamic_slots = 0, dynamic_vertices = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_per_vertex_output ||
             intr->intrinsic == nir_intrinsic_store_per_primitive_output) {
            narrow_outputs |= nir_src_bit_size(intr->src[0]) == 16;
            dynamic_vertices += !nir_src_is_const(intr->src[1]);
            dynamic_slots += !nir_src_is_const(intr->src[2]);
         }
      }
   }
   unsigned threads = hk_apple9_native_mesh_threads(nir->info.workgroup_size);
   bool class_match = stage == MESA_SHADER_MESH &&
      hk_apple9_packed_mesh_class_selected(hk_apple9_native_index_shapes(), threads,
         nir->info.mesh.max_vertices_out, nir->info.mesh.max_primitives_out);
   struct hk_physical_device pdev = {.dev.chip = AGX_CHIP_G16G,
                                    .dev.native_mesh_qualified = true};
   bool enabled = hk_apple9_native_index_enabled(&pdev);
   bool gate = enabled && hk_apple9_native_mesh_packed_shape_eligible(nir);
   /* Observe the existing body-layout model independently of completion.
    * This cannot admit a shader: native entry, publication and primitive
    * linking are still required even when the body lowering succeeds. */
   struct agx_apple9_mesh_body_info body = {0};
   const struct agx_apple9_mesh_key key = AGX_APPLE9_MESH_KEY;
   bool body_ok = false;
   bool native_ok = false;
   const char *body_reason = NULL, *native_reason = NULL;
   if (stage == MESA_SHADER_MESH) {
      nir_shader *copy = nir_shader_clone(NULL, nir);
      NIR_PASS(_, copy, nir_lower_explicit_io,
               nir_var_mem_shared | nir_var_mem_task_payload,
               nir_address_format_32bit_offset);
      NIR_PASS(_, copy, nir_remove_dead_variables,
               nir_var_mem_shared | nir_var_mem_task_payload, NULL);
      body_ok = agx_nir_lower_apple9_mesh(copy, &key, &body, &body_reason);
      if (!body_ok)
         fprintf(stderr, "body-layout unsupported: %s\n", body_reason ?: "unknown");
      ralloc_free(copy);
      if (research) {
         copy = nir_shader_clone(NULL, nir);
         NIR_PASS(_, copy, nir_lower_explicit_io,
                  nir_var_mem_shared | nir_var_mem_task_payload,
                  nir_address_format_32bit_offset);
         NIR_PASS(_, copy, nir_remove_dead_variables,
                  nir_var_mem_shared | nir_var_mem_task_payload, NULL);
         struct hk_apple9_graphics_key graphics = {
            .native_task_bridge = nir->info.task_payload_size != 0,
            .fs_reads_clip_distance = nir->info.clip_distance_array_size != 0,
         };
         VkResult result = hk_apple9_native_mesh_check(
            copy, &graphics, &native_ok, &native_reason);
         if (result != VK_SUCCESS)
            native_reason = "native capability allocation failed";
         ralloc_free(copy);
      }
   }
   const struct agx_apple9_mesh_layout *l = &body.layout;
   printf("{\"analysis_version\":4,\"qualified_policy_enabled\":%s,\"gate\":%s,\"class_match\":%s,\"workgroup_match\":%s,\"workgroup\":[%u,%u,%u],"
          "\"physical_threads\":%u,\"max_vertices\":%u,\"max_primitives\":%u,"
          "\"triangles\":%s,\"shared_bytes\":%u,\"payload_bytes\":%u,"
          "\"outputs_extra\":%s,\"outputs_16bit\":%s,\"per_primitive_extra\":%s,"
          "\"outputs_written\":\"0x%llx\",\"api_subgroup_size\":%u,\"variable_workgroup\":%s,"
          "\"body_layout_supported\":%s,\"vertex_components\":%u,\"primitive_components\":%u,"
          "\"index_words\":%u,\"position_offset\":%u,\"vertex_pitch\":%u,"
          "\"output_bytes\":%u,\"local_allocation_bytes\":%u,\"primitive_stride\":%u",
          enabled ? "true" : "false", gate ? "true" : "false", class_match ? "true" : "false",
          hk_apple9_native_mesh_workgroup_eligible(nir) ? "true" : "false",
          nir->info.workgroup_size[0], nir->info.workgroup_size[1], nir->info.workgroup_size[2],
          threads, stage == MESA_SHADER_MESH ? nir->info.mesh.max_vertices_out : 0,
          stage == MESA_SHADER_MESH ? nir->info.mesh.max_primitives_out : 0,
          stage == MESA_SHADER_MESH && nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES ? "true" : "false",
          nir->info.shared_size, nir->info.task_payload_size,
          nir->info.outputs_written & ~(VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES) ? "true" : "false",
          narrow_outputs ? "true" : "false",
          nir->info.per_primitive_outputs & ~VARYING_BIT_PRIMITIVE_INDICES ? "true" : "false",
          (unsigned long long)nir->info.outputs_written, nir->info.api_subgroup_size,
          nir->info.workgroup_size_variable ? "true" : "false",
          body_ok ? "true" : "false", l->vertex_components, l->primitive_components,
          DIV_ROUND_UP(l->indices_per_primitive * l->max_primitives, 4),
          l->position_B, l->vertex_pitch_B, l->local_size_B,
          body.local_allocation_B, l->primitive_stride_B);
   printf(",\"per_primitive_outputs\":\"0x%llx\",\"dynamic_io_slot_ops\":%u,\"dynamic_vertex_index_ops\":%u,"
          "\"clip_distances\":%u,\"cull_distances\":%u,"
          "\"research_requested\":%s,\"native_lowering_supported\":%s,\"body_reason\":",
          (unsigned long long)nir->info.per_primitive_outputs,
          dynamic_slots, dynamic_vertices, nir->info.clip_distance_array_size,
          nir->info.cull_distance_array_size, research ? "true" : "false",
          native_ok ? "true" : "false");
   json_string(body_reason);
   fputs(",\"native_reason\":", stdout);
   json_string(native_reason);
   puts("}");
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return 0;
}
