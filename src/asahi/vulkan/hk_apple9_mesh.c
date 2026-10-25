/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9_mesh.h"
#include "hk_physical_device.h"
#include "hk_device.h"
#include "hk_apple9_native_index_classes.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include <string.h>

DEBUG_GET_ONCE_BOOL_OPTION(native_mesh, "HK_APPLE9_NATIVE_MESH", false)
/* Land the qualified path without enabling the 31-class default. */
DEBUG_GET_ONCE_BOOL_OPTION(native_index, "HK_APPLE9_NATIVE_INDEX", false)
DEBUG_GET_ONCE_OPTION(native_stages, "HK_APPLE9_NATIVE_STAGES", NULL)
DEBUG_GET_ONCE_BOOL_OPTION(native_gs, "HK_APPLE9_GS_NATIVE", false)
DEBUG_GET_ONCE_BOOL_OPTION(native_tess, "HK_APPLE9_TESS_NATIVE", false)
DEBUG_GET_ONCE_BOOL_OPTION(native_gs_pack, "HK_APPLE9_NATIVE_GS_PACK", false)
DEBUG_GET_ONCE_BOOL_OPTION(native_tes_pack, "HK_APPLE9_NATIVE_TES_PACK", false)

bool
hk_apple9_native_tes_pack_enabled(void)
{
   return debug_get_option_native_tes_pack();
}

bool
hk_apple9_native_gs_pack_enabled(void)
{
   return debug_get_option_native_gs_pack();
}

bool
hk_apple9_native_gs_packed_class(unsigned threads, unsigned vertices,
                                 unsigned primitives)
{
   return threads == 32 &&
          ((vertices == 3 * HK_APPLE9_NATIVE_GS_PACK && primitives == 17) ||
           (vertices == 4 * HK_APPLE9_NATIVE_GS_PACK && primitives == 16));
}

DEBUG_GET_ONCE_BOOL_OPTION(native_task_bridge, "HK_APPLE9_NATIVE_TASK_BRIDGE", false)

bool
hk_apple9_native_task_bridge_enabled(void)
{
   return debug_get_option_native_task_bridge();
}

bool
hk_apple9_native_tess_enabled(void)
{
   return hk_apple9_native_mesh_enabled() && debug_get_option_native_tess();
}

bool
hk_apple9_native_gs_enabled(void)
{
   return hk_apple9_native_mesh_enabled() && debug_get_option_native_gs();
}
/* Qualified attribute/P1 support belongs to the opt-in native path. These
 * switches remain available for explicit comparisons within that path. */
DEBUG_GET_ONCE_BOOL_OPTION(native_mesh_attributes, "HK_APPLE9_NATIVE_MESH_ATTRIBUTES", true)
DEBUG_GET_ONCE_BOOL_OPTION(native_mesh_primitive_lowering, "HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING", true)
DEBUG_GET_ONCE_OPTION(native_mesh_packed_shapes,
                      "HK_APPLE9_NATIVE_MESH_PACKED_SHAPES", NULL)

static const struct {
   unsigned threads, vertices, primitives;
   const char *name;
} packed_shapes[] = {
#define HK_PACKED_MESH_SHAPE(t, v, p) {t, v, p, "T" #t "V" #v "P" #p},
#include "hk_apple9_packed_mesh_shapes.def"
#undef HK_PACKED_MESH_SHAPE
};
static_assert(ARRAY_SIZE(packed_shapes) <= 64, "packed shape selection mask");

uint64_t
hk_apple9_packed_mesh_shapes_parse(const char *selection)
{
   uint64_t mask = 0;
   if (!selection || !*selection)
      return 0;

   do {
      size_t length = strcspn(selection, ",");
      unsigned i;
      for (i = 0; i < ARRAY_SIZE(packed_shapes); ++i) {
         if (strlen(packed_shapes[i].name) == length &&
             !strncmp(selection, packed_shapes[i].name, length)) {
            mask |= BITFIELD64_BIT(i);
            break;
         }
      }
      /* Reject the whole selection, rather than silently widen or truncate a
       * malformed qualification list. In particular, "1" and "all" fail. */
      if (i == ARRAY_SIZE(packed_shapes))
         return 0;
      selection += length;
      if (!*selection)
         return mask;
   } while (*++selection);
   return 0;
}

bool
hk_apple9_packed_mesh_class_selected(uint64_t selection, unsigned threads,
                                    unsigned vertices, unsigned primitives)
{
   for (unsigned i = 0; i < ARRAY_SIZE(packed_shapes); ++i) {
      if ((selection & BITFIELD64_BIT(i)) &&
          threads == packed_shapes[i].threads &&
          vertices == packed_shapes[i].vertices &&
          primitives == packed_shapes[i].primitives)
         return true;
   }
   return false;
}

static bool
packed_class_enabled(unsigned threads, unsigned vertices, unsigned primitives)
{
   return hk_apple9_packed_mesh_class_selected(
      hk_apple9_native_mesh_packed_shapes(), threads, vertices, primitives);
}

uint64_t
hk_apple9_native_mesh_packed_shapes(void)
{
   /* An explicit legacy research configuration keeps its original semantics,
    * including HK_APPLE9_NATIVE_MESH=0 as a complete opt-out. An explicit
    * shape list alone must not widen the production qualification. */
   if (getenv("HK_APPLE9_NATIVE_MESH"))
      return hk_apple9_native_mesh_enabled()
                ? hk_apple9_packed_mesh_shapes_parse(
                     debug_get_option_native_mesh_packed_shapes())
                : 0;
   return debug_get_option_native_index() ? hk_apple9_native_index_shapes() : 0;
}

uint64_t
hk_apple9_native_index_shapes(void)
{
   static const char names[] =
#define HK_NATIVE_INDEX_SHAPE(t, v, p) ",T" #t "V" #v "P" #p
#include "hk_apple9_native_index_shapes.def"
#undef HK_NATIVE_INDEX_SHAPE
      ;
   return hk_apple9_packed_mesh_shapes_parse(names + 1);
}

bool
hk_apple9_native_index_enabled(const struct hk_physical_device *pdev)
{
   return pdev && pdev->dev.chip == AGX_CHIP_G16G &&
          pdev->dev.native_mesh_qualified && !pdev->disable_native_index &&
          debug_get_option_native_index() &&
          !getenv("HK_APPLE9_NATIVE_MESH");
}

bool
hk_apple9_native_mesh_available(const struct hk_physical_device *pdev)
{
   return hk_apple9_native_mesh_enabled() || hk_apple9_native_index_enabled(pdev);
}

bool
hk_apple9_native_mesh_available_device(struct hk_device *dev)
{
   /* The discovery VM partitions physical-device compiler/cache policy. It
    * cannot authorize a distinct logical VM: recording also needs that VM's
    * own immutable qualification. Research keeps its explicit pairing. */
   return hk_apple9_native_mesh_enabled() ||
          (dev->dev.native_mesh_qualified &&
           hk_apple9_native_index_enabled(hk_device_physical(dev)));
}

uint8_t
hk_apple9_native_stage_policy(const struct hk_physical_device *pdev)
{
   /* Stage qualification and its default decision are independent of native
    * index. An explicit stage selection enables host/device qualification;
    * absence, zero or an unknown selection keeps production GS/TES off. */
   if (!hk_apple9_native_index_enabled(pdev) || pdev->disable_native_stages)
      return 0;
   const char *selection = debug_get_option_native_stages();
   if (!selection || !strcmp(selection, "0"))
      return 0;
   if (!strcmp(selection, "gs"))
      return HK_APPLE9_NATIVE_GS_BIT;
   if (!strcmp(selection, "tes"))
      return HK_APPLE9_NATIVE_TES_BIT;
   if (!strcmp(selection, "gs,tes"))
      return HK_APPLE9_NATIVE_STAGE_BITS;

   static bool warned_invalid;
   if (!p_atomic_xchg(&warned_invalid, true))
      fprintf(stderr, "HK_APPLE9_NATIVE_STAGES: unknown selection '%s'; "
                      "expected 0, gs, tes or gs,tes; stages disabled\n",
              selection);
   return 0;
}

bool
hk_apple9_native_stage_available_device(struct hk_device *dev,
                                        mesa_shader_stage stage)
{
   if (stage == MESA_SHADER_GEOMETRY && hk_apple9_native_gs_enabled())
      return true;
   if (stage == MESA_SHADER_TESS_EVAL && hk_apple9_native_tess_enabled())
      return true;
   unsigned bit = stage == MESA_SHADER_GEOMETRY ? HK_APPLE9_NATIVE_GS_BIT
                : stage == MESA_SHADER_TESS_EVAL ? HK_APPLE9_NATIVE_TES_BIT : 0;
   return dev->dev.native_mesh_qualified &&
          (hk_apple9_native_stage_policy(hk_device_physical(dev)) & bit);
}

bool
hk_apple9_native_stage_class(mesa_shader_stage stage, unsigned threads,
                             unsigned vertices, unsigned primitives)
{
   /* The initial GS/TES contract never inherits the wider index class list.
    * In particular W64 and T16V16P16 cannot enter through a generated stage. */
   /* Packed GS publishes 8 triangles' fixed slots through two qualified
    * index classes (Air 39b7da1: 2.2 vs 9.4 ns of TA per triangle). */
   if (stage == MESA_SHADER_GEOMETRY && threads == 32 &&
       hk_apple9_native_gs_pack_enabled())
      return hk_apple9_native_gs_packed_class(threads, vertices, primitives);
   if (stage == MESA_SHADER_TESS_EVAL && threads == 32 &&
       hk_apple9_native_tes_pack_enabled())
      return vertices == 3 * HK_APPLE9_NATIVE_GS_PACK && primitives == 17;
   if (threads != 16)
      return false;
   if (stage == MESA_SHADER_GEOMETRY)
      return (vertices == 3 && primitives == 1) ||
             (vertices == 4 && primitives == 2);
   return stage == MESA_SHADER_TESS_EVAL && vertices == 3 && primitives == 1;
}

const char *
hk_apple9_native_index_class_hash(void)
{
   return HK_APPLE9_NATIVE_INDEX_CLASSES_SHA256;
}

bool
hk_apple9_native_mesh_workgroup_eligible(const nir_shader *nir)
{
   unsigned threads = hk_apple9_native_mesh_threads(nir->info.workgroup_size);
   if (!threads || nir->info.workgroup_size_variable ||
       (nir->info.api_subgroup_size && nir->info.api_subgroup_size != 32))
      return false;

   if (nir->info.workgroup_size[0] == threads &&
       nir->info.workgroup_size[1] == 1 && nir->info.workgroup_size[2] == 1)
      return true;

   /* Output resource classes describe physical publication, independently
    * of the application's logical workgroup. The existing P1 lowering masks
    * 1..16 logical lanes onto T16 and serializes up to four architectural
    * subgroups onto T32. Keep the larger-output gate at its qualified shape
    * while this logical P1 range is validated through ordinary Vulkan.
    * Derivative-bearing expansion needs its own geometry-preservation proof.
    */
   return nir->info.stage == MESA_SHADER_MESH &&
          nir->info.mesh.max_vertices_out == 3 &&
          nir->info.mesh.max_primitives_out == 1 &&
          nir->info.derivative_group == DERIVATIVE_GROUP_NONE;
}

bool
hk_apple9_native_mesh_round_counts_eligible(const uint16_t size[3],
                                             unsigned vertices,
                                             unsigned primitives)
{
   /* Research only: declarations beyond one publication round of their
    * natural T16/T32 width. Shapes that fit one round keep the class path.
    * Rounds are qualified at T32 only (Air 2026-10-24: 16-entry rounds at T16
    * over-count primitives), so these shapes always launch at T32 and 1..16
    * API invocations run masked. */
   unsigned threads = hk_apple9_native_mesh_threads(size);
   return hk_apple9_native_mesh_enabled() && threads && vertices &&
          vertices <= 256 && primitives && primitives <= 256 &&
          (vertices > threads || DIV_ROUND_UP(3 * primitives, 4) > threads);
}

bool
hk_apple9_native_mesh_model_counts_eligible(const uint16_t size[3],
                                             unsigned vertices,
                                             unsigned primitives)
{
   /* Research: any T16/T32 declaration the publication model covers, launched
    * at T32. V3/P1 keeps the P1 path, which also lowers primitive outputs. */
   return hk_apple9_native_mesh_enabled() && hk_apple9_native_mesh_threads(size) &&
          vertices && vertices <= 256 && primitives && primitives <= 256 &&
          !(vertices == 3 && primitives == 1);
}

bool
hk_apple9_native_mesh_round_shape_eligible(const nir_shader *nir)
{
   return nir->info.stage == MESA_SHADER_MESH && !nir->info.task_payload_size &&
          nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
          !nir->info.workgroup_size_variable &&
          (!nir->info.api_subgroup_size || nir->info.api_subgroup_size == 32) &&
          nir->info.derivative_group == DERIVATIVE_GROUP_NONE &&
          !(nir->info.per_primitive_outputs & ~VARYING_BIT_PRIMITIVE_INDICES) &&
          !nir->info.outputs_written_16bit &&
          hk_apple9_native_mesh_model_counts_eligible(
             nir->info.workgroup_size, nir->info.mesh.max_vertices_out,
             nir->info.mesh.max_primitives_out) &&
          /* Within one round, position-only shapes stay under the explicit
           * class list (T16V16P16 is deliberately excluded); the model takes
           * only the outputs that path cannot carry. */
          (hk_apple9_native_mesh_round_counts_eligible(
              nir->info.workgroup_size, nir->info.mesh.max_vertices_out,
              nir->info.mesh.max_primitives_out) ||
           (nir->info.outputs_written &
            ~(VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES |
              VARYING_BIT_PRIMITIVE_COUNT | VARYING_BIT_PSIZ)));
}

bool
hk_apple9_native_mesh_packed_shape_eligible(const nir_shader *nir)
{
   unsigned threads = hk_apple9_native_mesh_threads(nir->info.workgroup_size);
   /* Match the qualified class, not every shape satisfying ceil(3P/4)<=W.
    * Larger output classes retain W x 1 x 1. P1 logical expansion and
    * serialization do not change the physical output allocation. */
   return nir->info.stage == MESA_SHADER_MESH &&
          nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
          !nir->info.task_payload_size && !nir->info.shared_size &&
          hk_apple9_native_mesh_workgroup_eligible(nir) &&
          !(nir->info.outputs_written &
            ~(VARYING_BIT_POS | VARYING_BIT_PRIMITIVE_INDICES)) &&
          !nir->info.outputs_written_16bit &&
          !(nir->info.per_primitive_outputs & ~VARYING_BIT_PRIMITIVE_INDICES) &&
          packed_class_enabled(threads, nir->info.mesh.max_vertices_out,
                                nir->info.mesh.max_primitives_out);
}

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
       !local_allocation_valid(code, body) ||
       code->apple9_clip_distance_count != body->clip_distances ||
       body->clip_distances > 8)
      return false;
   if (stage == HK_APPLE9_NATIVE_OBJECT)
      return k->physical_threads == AGX_APPLE9_NATIVE_THREADS &&
             body->logical_size[0] == 1 && body->logical_size[1] == 1 &&
             body->logical_size[2] == 1 && !body->shared_size_B &&
             !body->layout.local_size_B && !code->local_size;
   const struct agx_apple9_mesh_layout *l = &body->layout;
   /* The lowering flag binds the binary to this path: a class binary with
    * stale metadata cannot pass as a masked T32 round binary, or vice versa. */
   if (body->publication_rounds
          ? !hk_apple9_native_mesh_model_counts_eligible(body->logical_size,
                                                          l->max_vertices,
                                                          l->max_primitives)
          : hk_apple9_native_mesh_round_counts_eligible(body->logical_size,
                                                         l->max_vertices,
                                                         l->max_primitives))
      return false;
   if (body->publication_rounds) {
      unsigned count_B = ALIGN_POT(3 * l->max_primitives, 16);
      unsigned position_B = count_B + 16;
      unsigned pitch_B = ALIGN_POT(l->max_vertices, 4) * 4;
      return k->physical_threads == AGX_APPLE9_NATIVE_SERIAL_THREADS &&
             !body->primitive_broadcast && l->indices_per_primitive == 3 &&
             l->vertex_components >= 4 &&
             l->vertex_components <= 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS &&
             code->apple9_varyings.count == l->vertex_components - 4 - body->clip_distances &&
             !l->primitive_components && !l->primitive_stride_B && !l->culling &&
             l->primitive_count_B == count_B && l->vertex_count_B == count_B + 4 &&
             l->cull_B == position_B && l->position_B == position_B &&
             l->vertex_pitch_B == pitch_B &&
             l->local_size_B == position_B + l->vertex_components * pitch_B;
   }
  unsigned logical_limit =
      k->physical_threads == AGX_APPLE9_NATIVE_THREADS
         ? AGX_APPLE9_NATIVE_THREADS
         : k->physical_threads == AGX_APPLE9_NATIVE_SERIAL_THREADS
              ? AGX_APPLE9_NATIVE_MAX_INVOCATIONS : 0;
   if (!logical_limit || !logical_size_valid(body->logical_size, logical_limit))
      return false;
   if (hk_apple9_native_mesh_threads(body->logical_size) != k->physical_threads)
      return false;
   /* The default policy also checks the exact P1 declaration at upload;
    * a research P1 shader binary must not bypass the production scope. */
   if (l->max_vertices != 3 || l->max_primitives != 1 ||
       !hk_apple9_native_mesh_enabled()) {
      unsigned count_B = ALIGN_POT(3 * l->max_primitives, 16);
      unsigned position_B = count_B + 16;
      unsigned pitch_B = ALIGN_POT(l->max_vertices, 4) * 4;
      return packed_class_enabled(k->physical_threads, l->max_vertices,
                                   l->max_primitives) &&
             ((l->max_vertices == 3 && l->max_primitives == 1) ||
              (body->logical_size[0] == k->physical_threads &&
               body->logical_size[1] == 1 && body->logical_size[2] == 1)) &&
             !body->shared_size_B && !body->primitive_broadcast &&
             l->indices_per_primitive == 3 && l->vertex_components == 4 &&
             !code->apple9_varyings.count && !l->primitive_components &&
             !l->primitive_stride_B && !l->culling &&
             l->primitive_count_B == count_B &&
             l->vertex_count_B == count_B + 4 && l->cull_B == position_B &&
             l->position_B == position_B && l->vertex_pitch_B == pitch_B &&
             l->local_size_B == position_B + 4 * pitch_B;
   }
   return l->max_vertices == 3 && l->max_primitives == 1 &&
          l->indices_per_primitive == 3 && l->vertex_components >= 4 &&
          l->vertex_components <= 4 + AGX_APPLE9_MAX_VARYING_COMPONENTS &&
          code->apple9_varyings.count == l->vertex_components - 4 - body->clip_distances &&
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
   size_t bytes;
   return hk_apple9_mesh_command_build_delta(out, capacity, p, state, groups,
                                             object_local_size, enable, NULL,
                                             NULL, &bytes) &&
          bytes == HK_APPLE9_MESH_COMMAND_BYTES;
}

/* The PPP state blocks of a record, in update-chain order: byte offset and
 * length in words. The viewport block may move and grow (several viewports). */
static unsigned
hk_apple9_ppp_blocks(const struct agx_apple9_render_pipeline *p,
                     uint16_t offset[9], uint8_t words[9])
{
   static const uint16_t offsets[] = {0, 0x40, 0x5c, 0x70, 0x8c, 0xc0, 0xa0, 0xac, 0xf0};
   static const uint8_t lengths[] = {5, 7, 5, 7, 5, 10, 3, 2, 2};
   for (unsigned i = 0; i < 9; ++i) {
      offset[i] = offsets[i];
      words[i] = lengths[i];
   }
   if (p->ppp_viewport_words)
      words[5] = p->ppp_viewport_words;
   if (p->ppp_viewport_offset)
      offset[5] = p->ppp_viewport_offset;
   return 9;
}

/* As hk_apple9_mesh_command_build, but when the previous native block of the
 * same render left PPP state `previous` (CPU bytes of its record), update only
 * the blocks whose bytes changed, always including the first (its header):
 * PPP state persists across a render's blocks, and Metal's second block of a
 * render updates one block (Air D2 capture). *bytes is the block length. */
bool
hk_apple9_mesh_command_build_delta(void *out, size_t capacity,
                                   const struct agx_apple9_render_pipeline *p,
                                   const uint32_t state[AGX_APPLE9_MESH_STATE_WORDS],
                                   const uint32_t groups[3],
                                   const uint32_t object_local_size[3], bool enable,
                                   const uint8_t *previous, const uint8_t *record,
                                   size_t *bytes)
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
   /* Each update names a block; the word before it carries its length. */
   uint16_t offset[9];
   uint8_t length[9];
   unsigned blocks = hk_apple9_ppp_blocks(p, offset, length);
   for (unsigned i = 0; i < blocks; ++i) {
      if (previous && record && i &&
          !memcmp(previous + offset[i], record + offset[i], length[i] * 4u))
         continue;
      words[at++] = length[i] << 8;
      words[at++] = p->ppp + offset[i];
   }
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
   assert(at <= ARRAY_SIZE(words));
   memcpy(out, words, at * 4);
   *bytes = at * 4;
   return true;
}
