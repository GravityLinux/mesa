/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9_mesh.h"
#include "util/u_debug.h"
#include <string.h>

DEBUG_GET_ONCE_BOOL_OPTION(native_mesh, "HK_APPLE9_NATIVE_MESH", false)
/* Qualified attribute/P1 support belongs to the opt-in native path. These
 * switches remain available for explicit comparisons within that path. */
DEBUG_GET_ONCE_BOOL_OPTION(native_mesh_attributes, "HK_APPLE9_NATIVE_MESH_ATTRIBUTES", true)
DEBUG_GET_ONCE_BOOL_OPTION(native_mesh_primitive_lowering, "HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING", true)

bool
hk_apple9_native_mesh_attributes_enabled(void)
{
   return debug_get_option_native_mesh_attributes();
}

bool
hk_apple9_native_mesh_enabled(void)
{
   return debug_get_option_native_mesh();
}

bool
hk_apple9_native_mesh_primitive_lowering_enabled(void)
{
   return hk_apple9_native_mesh_enabled() &&
          hk_apple9_native_mesh_attributes_enabled() &&
          debug_get_option_native_mesh_primitive_lowering();
}

bool
hk_apple9_native_mesh_grid_valid(const uint32_t groups[3])
{
   if (!groups)
      return false;

   unsigned total = 1;
   for (unsigned i = 0; i < 3; ++i) {
      if (!groups[i] || groups[i] > HK_APPLE9_NATIVE_MAX_GROUP_COUNT ||
          groups[i] > HK_APPLE9_NATIVE_MAX_GROUP_TOTAL / total)
         return false;
      total *= groups[i];
   }
   return true;
}

static bool
logical_size_valid(const uint16_t size[3], unsigned limit)
{
   unsigned total = 1;
   for (unsigned i = 0; i < 3; ++i) {
      if (!size[i] || size[i] > limit / total)
         return false;
      total *= size[i];
   }
   return true;
}

unsigned
hk_apple9_native_mesh_threads(const uint16_t local_size[3])
{
   if (!local_size ||
       !logical_size_valid(local_size, AGX_APPLE9_NATIVE_MAX_INVOCATIONS))
      return 0;
   return logical_size_valid(local_size, AGX_APPLE9_NATIVE_THREADS)
             ? AGX_APPLE9_NATIVE_THREADS : AGX_APPLE9_NATIVE_SERIAL_THREADS;
}

static bool
local_allocation_valid(const struct agx_shader_info *code,
                       const struct agx_apple9_mesh_body_info *body)
{
   unsigned output = body->layout.local_size_B;
   if (output > AGX_APPLE9_MAX_SHARED_BYTES ||
       body->shared_size_B > AGX_APPLE9_MAX_SHARED_BYTES)
      return false;
   unsigned offset = body->shared_size_B ? ALIGN_POT(output, 16) : 0;
   unsigned shared = ALIGN_POT(body->shared_size_B, 4);
   if (shared > AGX_APPLE9_MAX_SHARED_BYTES - offset)
      return false;
   unsigned total = body->shared_size_B ? offset + shared : output;
   return body->shared_offset_B == offset &&
          body->local_allocation_B == total &&
          code->local_size == (body->shared_size_B ? total : 0);
}

bool
hk_apple9_native_shader_valid(const struct agx_shader_info *code,
                               const struct agx_apple9_mesh_body_info *body,
                               enum hk_apple9_native_stage stage)
{
   if (!code || !body || stage == HK_APPLE9_NATIVE_NONE ||
       stage > HK_APPLE9_NATIVE_MESH || !code->apple9_body_only ||
       !code->binary_size || code->scratch_size ||
       code->apple9_atomic_frame_size ||
       code->workgroup_size[1] != 1 || code->workgroup_size[2] != 1 ||
       body->requires_mesh_completion || body->requires_object_completion ||
       body->argument_uniform_words != AGX_APPLE9_MESH_ARGUMENT_WORDS)
      return false;
   const struct agx_apple9_mesh_key *k = &body->key;
   const struct agx_apple9_mesh_key expected = AGX_APPLE9_MESH_KEY;
   if (k->reserved_uniform_words != expected.reserved_uniform_words ||
       k->output_root != expected.output_root ||
       k->primitive_count_bits != expected.primitive_count_bits ||
       k->payload_uniform != expected.payload_uniform ||
       k->grid_root != expected.grid_root || k->grid_offset_B ||
       code->workgroup_size[0] != k->physical_threads ||
       !local_allocation_valid(code, body))
      return false;
   if (stage == HK_APPLE9_NATIVE_OBJECT)
      return k->physical_threads == AGX_APPLE9_NATIVE_THREADS &&
             body->logical_size[0] == 1 && body->logical_size[1] == 1 &&
             body->logical_size[2] == 1 && !body->shared_size_B &&
             !body->layout.local_size_B && !code->local_size;
   unsigned logical_limit =
      k->physical_threads == AGX_APPLE9_NATIVE_THREADS
         ? AGX_APPLE9_NATIVE_THREADS
         : k->physical_threads == AGX_APPLE9_NATIVE_SERIAL_THREADS
              ? AGX_APPLE9_NATIVE_MAX_INVOCATIONS : 0;
   if (!logical_limit || !logical_size_valid(body->logical_size, logical_limit))
      return false;
   const struct agx_apple9_mesh_layout *l = &body->layout;
   return l->max_vertices == 3 && l->max_primitives == 1 &&
          l->indices_per_primitive == 3 && l->vertex_components >= 4 &&
          l->vertex_components <= 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS &&
          code->apple9_varyings.count == l->vertex_components - 4 &&
          !l->primitive_components && !l->primitive_stride_B && !l->culling &&
          l->local_size_B == 32 + 16 * l->vertex_components && l->position_B == 32 &&
          l->vertex_pitch_B == 16;
}

bool
hk_apple9_native_launch_parameters(
   struct agx_apple9_launch_parameters *out,
   const struct agx_shader_info *code,
   const struct agx_apple9_mesh_body_info *body,
   enum hk_apple9_native_stage stage, uint64_t shader_base,
   uint64_t code_address)
{
   if (!out || !hk_apple9_native_shader_valid(code, body, stage) ||
       code_address < shader_base || code_address - shader_base > UINT32_MAX ||
       (code_address & 1) ||
       code->apple9_preamble_size > AGX_APPLE9_MAX_GRAPHICS_PREAMBLE_BYTES ||
       code->apple9_preamble_offset > code->binary_size ||
       code->apple9_preamble_size >
          code->binary_size - code->apple9_preamble_offset)
      return false;
   /* Native frame backing is an entry/call requirement independent of spill
    * allocation. The measured T16 class uses object 16/16 and mesh 64/64.
    * Scratch and returned-atomic frames need separate qualification. */
   unsigned frame = stage == HK_APPLE9_NATIVE_OBJECT ? 16 : 64;
   *out = (struct agx_apple9_launch_parameters){
      .shader_base = shader_base,
      .entry_offset = code_address - shader_base,
      .publication_count = code->apple9_publication_count,
      .frame_extent_a = frame,
      .frame_extent_b = frame,
      .threadgroup_memory_bytes = stage == HK_APPLE9_NATIVE_OBJECT
                                    ? HK_APPLE9_NATIVE_LOCAL_BYTES
                                    : util_next_power_of_two(MAX2(
                                         HK_APPLE9_NATIVE_LOCAL_BYTES,
                                         body->local_allocation_B)),
      .preamble_address = code->apple9_preamble_size
                             ? code_address + code->apple9_preamble_offset : 0,
   };
   return true;
}

bool
hk_apple9_mesh_command_build(void *out, size_t capacity,
                             const struct agx_apple9_render_pipeline *p,
                             const uint32_t state[AGX_APPLE9_MESH_STATE_WORDS],
                             const uint32_t groups[3],
                             const uint32_t object_local_size[3], bool enable)
{
   if (!enable || !out || !p || !state || !groups || !object_local_size ||
       capacity < HK_APPLE9_MESH_COMMAND_BYTES || groups[0] != 1 ||
       groups[1] != 1 || groups[2] != 1 ||
       object_local_size[0] != AGX_APPLE9_NATIVE_THREADS ||
       object_local_size[1] != 1 || object_local_size[2] != 1 ||
       p->primitive > 0xf || (p->ppp & 3) ||
       p->ppp > UINT32_MAX - AGX_APPLE9_PPP_RECORD_MAX ||
       p->ppp_viewport_words > 255 || (p->ppp_viewport_offset & 3) ||
       p->ppp_viewport_offset > AGX_APPLE9_PPP_RECORD_MAX - 12)
      return false;
   uint32_t words[HK_APPLE9_MESH_COMMAND_BYTES / 4];
   memcpy(words, state, AGX_APPLE9_MESH_STATE_WORDS * 4);
   unsigned at = AGX_APPLE9_MESH_STATE_WORDS;
   words[at++] = 0x500;
   static const uint16_t offsets[] = {0,    0x40, 0x5c, 0x70,
                                      0x8c, 0xc0, 0xa0, 0xac};
   static const uint16_t counts[] = {7, 5, 7, 5, 10, 3, 2, 2};
   for (unsigned i = 0; i < ARRAY_SIZE(offsets); ++i) {
      unsigned offset = offsets[i], count = counts[i];
      if (offset == 0x8c && p->ppp_viewport_words)
         count = p->ppp_viewport_words;
      if (offset == 0xc0 && p->ppp_viewport_offset)
         offset = p->ppp_viewport_offset;
      words[at++] = p->ppp + offset;
      words[at++] = count << 8;
   }
   words[at++] = p->ppp + 0xf0;
   words[at++] = 0x70000000 | (p->primitive << 8);
   /* The dispatch tail carries total threads, then local threads. The mesh
    * state count is independent of object entry participation. Launch the
    * compiler's physical workgroup so ordinary mask pops can reconverge all
    * completion lanes, even when only API lane zero runs the taskless body. */
   for (unsigned i = 0; i < 3; ++i)
      words[at++] = groups[i] * object_local_size[i];
   for (unsigned i = 0; i < 3; ++i)
      words[at++] = object_local_size[i];
   words[at++] = 0xc0000000;
   assert(at == ARRAY_SIZE(words));
   memcpy(out, words, sizeof(words));
   return true;
}
