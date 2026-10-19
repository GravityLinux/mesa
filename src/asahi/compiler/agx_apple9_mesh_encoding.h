/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef AGX_APPLE9_MESH_ENCODING_H
#define AGX_APPLE9_MESH_ENCODING_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Two instruction families, not six fixed-register instructions. These names
 * describe the measured interface. In particular, control's second operand
 * and flag42 do not yet have a general semantic interpretation. */
enum agx_apple9_mesh_instruction_kind {
   AGX_APPLE9_MESH_CONTROL,
   AGX_APPLE9_MESH_SUBMIT,
};

enum agx_apple9_mesh_submit_mode {
   AGX_APPLE9_MESH_SUBMIT_SCALAR = 0,
   AGX_APPLE9_MESH_SUBMIT_CULL = 1,
   AGX_APPLE9_MESH_SUBMIT_CHANNEL = 2,
};

struct agx_apple9_mesh_instruction {
   enum agx_apple9_mesh_instruction_kind kind;
   /* CONTROL reads and clears the selected primitive-count halfword.
    * Each descriptor encodes 2*GPR+half; B's read contract remains provisional.
    * The current compiler owns a whole GPR for each halfword operand. */
   uint8_t count_gpr, auxiliary_gpr;
   bool count_high, auxiliary_high, control42;
   /* SUBMIT names a source or source tuple by its first 32-bit register.
    * CHANNEL consumes (channel selector, local plane offset). Mode CULL is
    * exposed for field decoding; the development NIR path does not emit it. */
   uint8_t source_gpr;
   enum agx_apple9_mesh_submit_mode mode;
   bool release;
};

/* Only the observed six-byte register forms are accepted. Unknown header,
 * file/width and control bits are rejected rather than silently normalized.
 * These encode individual instructions; no executable spans are accepted. */
bool agx_apple9_encode_mesh(uint8_t out[6],
                           const struct agx_apple9_mesh_instruction *instruction);
bool agx_apple9_decode_mesh(const uint8_t in[6],
                           struct agx_apple9_mesh_instruction *instruction);

#ifdef __cplusplus
}
#endif
#endif
