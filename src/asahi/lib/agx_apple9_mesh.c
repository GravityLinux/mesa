/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_mesh.h"
#include <string.h>

bool
agx_apple9_mesh_state_build(uint32_t out[AGX_APPLE9_MESH_STATE_WORDS],
                            const struct agx_apple9_mesh_state_parameters *p)
{
   if (!out || !p || !p->layout)
      return false;
   const struct agx_apple9_mesh_layout *l = p->layout;
   /* Task payload and per-primitive outputs need separate completion and
    * PPP contracts; state field encodings alone do not qualify them. */
   if (p->task_payload_size_B || p->primitive_output.components ||
       p->primitive_output.stride_B || l->primitive_components ||
       l->primitive_stride_B)
      return false;
   unsigned n = l->vertex_components + (l->primitive_components != 0);
   if (!l->max_vertices || l->max_vertices > 256 || !l->max_primitives ||
       l->max_primitives > 512 || l->indices_per_primitive != 3 ||
       l->vertex_components < 4 || l->vertex_components > 253 || n > 254 ||
       l->primitive_components > 128 || !p->threads || p->threads > 256 ||
       l->local_size_B > AGX_APPLE9_MAX_SHARED_BYTES ||
       l->local_size_B < l->cull_B || (l->local_size_B - l->cull_B) % 16 ||
       (p->object_launch & 1) || (p->mesh_launch & 1))
      return false;

   /* Measured output contract, shared with body lowering. Rows cover the
    * cull region and the vertex/link planes, excluding indices/count header.
    * Culling and broader output classes still need normal completion support. */
   uint32_t words[AGX_APPLE9_MESH_STATE_WORDS] = {
      0x40000d2c,
      ((n + 1) << 8) | n,
      l->primitive_components ? 0xd0 : 0x10,
      0,
      0,
      0x01000000,
      p->object_launch,
      0x01000000,
      p->mesh_launch,
      p->has_object_stage,
      (l->max_primitives - 1) |
         ((DIV_ROUND_UP(l->max_primitives, 16) - 1) << 16) |
         (l->culling ? BITFIELD_BIT(30) : 0),
      /* Direct count field: own-MSL DATA measures 0x400900 at 256 threads
       * (and 0x400a00 at 512). Bit 8 belongs to the count, not another field.
       * Large-output completion is independent of this encoding evidence. */
      0x400800 | p->threads,
      l->max_primitives * l->primitive_components,
      1 + (l->local_size_B - l->cull_B) / 16,
   };
   memcpy(out, words, sizeof(words));
   return true;
}

bool
agx_apple9_mesh_arguments_build(uint64_t out[AGX_APPLE9_MESH_ARGUMENT_WORDS / 2],
                                const struct agx_apple9_mesh_arguments *p)
{
   if (!out || !p)
      return false;
   uint64_t roots[AGX_APPLE9_MESH_ARGUMENT_WORDS / 2] = {0};
   memcpy(roots, p->graphics, sizeof(p->graphics));
   roots[AGX_APPLE9_MESH_OUTPUT_ROOT] = p->output_root;
   roots[AGX_APPLE9_MESH_GRID_ROOT] = p->grid_root;
   roots[AGX_APPLE9_MESH_PAYLOAD_UNIFORM / 2] = p->payload;
   roots[AGX_APPLE9_MESH_DRAW_UNIFORM / 2] = p->draw;
   memcpy(out, roots, sizeof(roots));
   return true;
}
