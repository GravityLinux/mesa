/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Ordinary Apple9 mesh DATA encoders. Inputs are current SSA values; captured
 * records belong exclusively to tests. Reservation/consumer binding is separate.
 */
#include "agx_apple9_mesh_formation.h"
#include "agx_nir.h"

static void
store(nir_builder *b, nir_def *address, unsigned offset, nir_def *value)
{
   nir_store_global(b, value, nir_iadd_imm(b, address, offset), .align_mul = 4);
}

static void
word(nir_builder *b, nir_def *address, unsigned offset, uint32_t value)
{
   store(b, address, offset, nir_imm_int(b, value));
}

static nir_def *
input(nir_builder *b, enum agx_apple9_mesh_formation_input name)
{
   return nir_load_mesh_formation_input_apple9_agx(
      b, name <= AGX_MESH_SCENE_ADDRESS ? 64 : 32, .base = name);
}

static void
fence(nir_builder *b)
{
   nir_barrier(b, .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

static void
rendezvous(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

/* Circular byte ranges. With three live references, enumerating source-value
 * bases covers the same finite candidate set as the DATA-derived codec. */
static nir_def *
delta(nir_builder *b, nir_def *v, nir_def *base)
{
   return nir_iand_imm(b, nir_isub(b, v, base), 255);
}

struct channel {
   nir_def *mode, *width, *prefix_count, *base[2], *symbol[3];
};

static struct channel
encode_channel(nir_builder *b, nir_def **v)
{
   nir_def *rank = nir_imm_int(b, UINT32_MAX);
   for (unsigned i = 0; i < 3; ++i) {
      nir_def *d = nir_imm_int(b, 0);
      for (unsigned j = 0; j < 3; ++j)
         d = nir_umax(b, d, delta(b, v[j], v[i]));
      nir_def *width = nir_iadd_imm(b, nir_ufind_msb(b, d), 1);
      nir_def *r = nir_ior(b, nir_ishl_imm(b, width, 24),
                           nir_ior_imm(b, nir_ishl_imm(b, v[i], 8), 1 << 16));
      rank = nir_umin(
         b, rank,
         nir_bcsel(b, nir_ult_imm(b, width, 8), r, nir_imm_int(b, UINT32_MAX)));
   }
   for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = i + 1; j < 3; ++j) {
         nir_def *lo = nir_umin(b, v[i], v[j]);
         nir_def *hi = nir_umax(b, v[i], v[j]);
         nir_def *d = nir_imm_int(b, 0);
         for (unsigned k = 0; k < 3; ++k)
            d = nir_umax(b, d,
                         nir_umin(b, delta(b, v[k], lo), delta(b, v[k], hi)));
         nir_def *width = nir_iadd_imm(b, nir_ufind_msb(b, d), 2);
         nir_def *r = nir_ior(
            b, nir_ishl_imm(b, width, 24),
            nir_ior(b, nir_ior_imm(b, nir_ishl_imm(b, lo, 8), 2 << 16), hi));
         nir_def *valid =
            nir_iand(b, nir_ine(b, lo, hi), nir_ult_imm(b, width, 8));
         rank = nir_umin(b, rank,
                         nir_bcsel(b, valid, r, nir_imm_int(b, UINT32_MAX)));
      }
   }
   nir_def *w = nir_ushr_imm(b, rank, 24);
   nir_def *n = nir_iand_imm(b, nir_ushr_imm(b, rank, 16), 255);
   nir_def *cost = nir_iadd(b, nir_imul_imm(b, n, 8), nir_imul_imm(b, w, 3));
   nir_def *raw = nir_uge(b, cost, nir_imm_int(b, 24));
   struct channel f = {
      .width = nir_bcsel(b, raw, nir_imm_int(b, 8), w),
      .prefix_count = nir_bcsel(b, raw, nir_imm_int(b, 0), n),
      .base = {nir_iand_imm(b, nir_ushr_imm(b, rank, 8), 255),
               nir_iand_imm(b, rank, 255)},
   };
   nir_def *dual = nir_ieq_imm(b, n, 2);
   f.mode = nir_bcsel(
      b, raw, nir_imm_int(b, 8),
      nir_iadd(b, w, nir_bcsel(b, dual, nir_imm_int(b, 8), nir_imm_int(b, 0))));
   for (unsigned i = 0; i < 3; ++i) {
      nir_def *d0 = delta(b, v[i], f.base[0]);
      nir_def *d1 = delta(b, v[i], f.base[1]);
      nir_def *limit = nir_ishl(b, nir_imm_int(b, 1), nir_iadd_imm(b, w, -1));
      nir_def *s0 = nir_bcsel(b, nir_ult(b, d0, limit), nir_ishl_imm(b, d0, 1),
                              nir_imm_int(b, UINT32_MAX));
      nir_def *s1 = nir_bcsel(b, nir_ult(b, d1, limit),
                              nir_ior_imm(b, nir_ishl_imm(b, d1, 1), 1),
                              nir_imm_int(b, UINT32_MAX));
      f.symbol[i] =
         nir_bcsel(b, raw, v[i], nir_bcsel(b, dual, nir_umin(b, s0, s1), d0));
   }
   return f;
}

/* Pack into SSA words. This avoids store/load ordering assumptions inside the
 * bit writer; only final words are stored. Zero-width fields contribute zero. */
static void
pack(nir_builder *b, nir_def **out, nir_def *value, nir_def *at, nir_def *width)
{
   nir_def *slot = nir_ushr_imm(b, at, 5);
   nir_def *shift = nir_iand_imm(b, at, 31);
   value = nir_bcsel(b, nir_ine_imm(b, width, 0), value, nir_imm_int(b, 0));
   nir_def *low = nir_ishl(b, value, shift);
   nir_def *cross = nir_ult(b, nir_imm_int(b, 32), nir_iadd(b, shift, width));
   nir_def *high = nir_bcsel(
      b, cross, nir_ushr(b, value, nir_isub(b, nir_imm_int(b, 32), shift)),
      nir_imm_int(b, 0));
   for (unsigned i = 0; i < 8; ++i) {
      out[i] =
         nir_ior(b, out[i],
                 nir_bcsel(b, nir_ieq_imm(b, slot, i), low, nir_imm_int(b, 0)));
      if (i)
         out[i] = nir_ior(
            b, out[i],
            nir_bcsel(b, nir_ieq_imm(b, slot, i - 1), high, nir_imm_int(b, 0)));
   }
}

struct agx_apple9_mesh_position_encoding
agx_nir_apple9_mesh_encode_positions(nir_builder *b, nir_def *p[3][3])
{
   struct channel f[10];
   nir_def *desc[2] = {nir_imm_int(b, 0), nir_imm_int(b, 0)};
   nir_def *out[8];
   for (unsigned i = 0; i < 8; ++i)
      out[i] = nir_imm_int(b, 0);
   nir_def *bit = nir_imm_int(b, 0);
   for (unsigned c = 0; c < 10; ++c) {
      unsigned component = c < 3 ? 0 : c < 6 ? 1 : 2;
      unsigned byte = c < 3 ? c : c < 6 ? c - 3 : c - 6;
      nir_def *v[3];
      for (unsigned i = 0; i < 3; ++i)
         v[i] =
            nir_iand_imm(b, nir_ushr_imm(b, p[i][component], byte * 8), 255);
      f[c] = encode_channel(b, v);
      desc[c / 8] =
         nir_ior(b, desc[c / 8], nir_ishl_imm(b, f[c].mode, 4 * (c % 8)));
      for (unsigned j = 0; j < 2; ++j) {
         nir_def *width = nir_bcsel(b, nir_ugt_imm(b, f[c].prefix_count, j),
                                    nir_imm_int(b, 8), nir_imm_int(b, 0));
         pack(b, out, f[c].base[j], bit, width);
         bit = nir_iadd(b, bit, width);
      }
   }
   for (unsigned i = 0; i < 3; ++i) {
      for (unsigned c = 0; c < 10; ++c) {
         pack(b, out, f[c].symbol[i], bit, f[c].width);
         bit = nir_iadd(b, bit, f[c].width);
      }
   }
   struct agx_apple9_mesh_position_encoding encoded = {
      .modes = {desc[0], desc[1]},
      .bits = bit,
      .bytes = nir_ishl_imm(b, nir_ushr_imm(b, nir_iadd_imm(b, bit, 31), 5), 2),
   };
   memcpy(encoded.words, out, sizeof(out));
   return encoded;
}

struct agx_apple9_mesh_compaction
agx_nir_apple9_mesh_compact_triangle(nir_builder *b, nir_def *index[3])
{
   nir_def *new1 = nir_ine(b, index[1], index[0]);
   nir_def *new2 = nir_iand(b, nir_ine(b, index[2], index[0]),
                            nir_ine(b, index[2], index[1]));
   return (struct agx_apple9_mesh_compaction){
      .id = {nir_imm_int(b, 0), nir_b2i32(b, new1),
             nir_bcsel(
                b, nir_ieq(b, index[2], index[0]), nir_imm_int(b, 0),
                nir_bcsel(b, nir_ieq(b, index[2], index[1]), nir_b2i32(b, new1),
                          nir_iadd_imm(b, nir_b2i32(b, new1), 1)))},
      .count = nir_iadd_imm(
         b, nir_iadd(b, nir_b2i32(b, new1), nir_b2i32(b, new2)), 1),
   };
}

nir_def *
agx_nir_apple9_mesh_triangle_tokens(
   nir_builder *b, const struct agx_apple9_mesh_compaction *compact,
   nir_def *backface)
{
   nir_def *tokens = nir_imm_int(b, 0);
   for (unsigned i = 0; i < 3; ++i)
      tokens =
         nir_ior(b, tokens,
                 nir_ishl_imm(b, nir_ior_imm(b, compact->id[i], AGX_MESH_OBS_COMPACT_REFERENCE_TAG), i * 8));
   return nir_ior(b, tokens, nir_ishl_imm(b, nir_b2i32(b, backface), 23));
}

static void
store_at(nir_builder *b, nir_def *address, nir_def *offset, nir_def *value)
{
   nir_store_global(b, value, nir_iadd(b, address, nir_u2u64(b, offset)),
                    .align_mul = 4);
}

nir_def *
agx_nir_apple9_mesh_write_header(
   nir_builder *b, nir_def *address,
   const struct agx_apple9_mesh_record_state *s,
   const struct agx_apple9_mesh_position_encoding *p, nir_def *compact_count)
{
   for (unsigned i = 0; i < 3; ++i)
      word(b, address, 4 * i, 0);
   word(b, address, 12, AGX_MESH_OBS_HEADER_FLAGS);
   store(b, address, 16, s->fragment_launch);
   store(b, address, 20, nir_ior(b, s->depth_control, s->front_ccw));
   store(b, address, 24,
         nir_ior(b, s->depth_face_2, nir_iand_imm(b, s->depth_face, AGX_MESH_DEPTH_FACE_LOW_MASK)));
   nir_def *stencil = nir_test_mask(b, s->depth_control, 1 << 18);
   nir_def *scissor = nir_test_mask(b, s->depth_control, (1 << 16) | (1 << 17));
   nir_push_if(b, stencil);
   store(b, address, 28, s->stencil);
   nir_pop_if(b, NULL);
   nir_def *offset =
      nir_iadd_imm(b, nir_ishl_imm(b, nir_b2i32(b, stencil), 2), 28);
   nir_push_if(b, scissor);
   store_at(b, address, offset, s->scissor_bias);
   nir_pop_if(b, NULL);
   offset = nir_iadd(b, offset, nir_ishl_imm(b, nir_b2i32(b, scissor), 2));
   store_at(b, address, offset, s->tile_control);
   store_at(b, address, nir_iadd_imm(b, offset, 4), nir_imm_int(b, 0));
   store_at(b, address, nir_iadd_imm(b, offset, 8), p->modes[0]);
   store_at(b, address, nir_iadd_imm(b, offset, 12),
            nir_ior(b, p->modes[1],
                    nir_ishl_imm(b, nir_iadd_imm(b, compact_count, -1), 8)));
   return nir_iadd_imm(b, offset, 16);
}

void
agx_nir_apple9_mesh_write_short_list(nir_builder *b, nir_def *address,
                                     nir_def *geometry_id)
{
   word(b, address, 0, AGX_MESH_OBS_SHORT_LIST_TAG);
   store(b, address, 4, nir_ishl_imm(b, geometry_id, 5));
   /* One surviving primitive, hence one set bit in the primitive mask. */
   word(b, address, 8, AGX_MESH_OBS_SHORT_LIST_TAIL |
                          (1u << AGX_MESH_SHORT_LIST_PRIMITIVE_MASK_SHIFT));
}

void
agx_nir_apple9_mesh_write_short_root(nir_builder *b, nir_def *address,
                                     nir_def *root_id, nir_def *geometry_id,
                                     nir_def *list_id)
{
   word(b, address, 0, AGX_MESH_SHORT_ROOT_COUNT);
   word(b, address, 4, 0);
   store(b, address, 8, nir_ishl_imm(b, root_id, 1));
   word(b, address, 12, 0);
   store(b, address, 16, geometry_id);
   store(b, address, 20, nir_ishl_imm(b, list_id, 9));
   store(b, address, 24, nir_ushr_imm(b, list_id, 23));
   word(b, address, 28, AGX_MESH_SHORT_ROOT_VALID_MASK <<
                           AGX_MESH_ROOT_VALID_MASK_SHIFT);
}

static bool
prepare(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_apple9_mesh_body_info *info = data;
   const struct agx_apple9_mesh_key *key = &info->key;
   if (intr->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      /* Compiler-owned local scratch in the unused count-header word. */
      nir_store_local_apple9_agx(b, intr->src[0].ssa, nir_imm_int(b, info->layout.primitive_count_B + 4),
                                 .base = key->output_root, .bit_size = 32);
      nir_pop_if(b, NULL);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_store_per_vertex_output &&
       nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PSIZ) {
      nir_instr_remove(&intr->instr);
      return true;
   }
   return false;
}

static nir_def *
finite_normal_or_zero(nir_builder *b, nir_def *v)
{
   nir_def *a = nir_iand_imm(b, v, 0x7fffffff);
   return nir_iand(
      b, nir_ult_imm(b, a, 0x7f800000),
      nir_ior(b, nir_ieq_imm(b, a, 0), nir_uge_imm(b, a, 0x800000)));
}


bool
agx_nir_lower_apple9_mesh_formation(
   nir_shader *nir, const struct agx_apple9_mesh_key *key,
   struct agx_apple9_mesh_body_info *info,
   const char **reason)
{
   if (nir->info.stage != MESA_SHADER_MESH || nir->info.shared_size ||
       nir->info.task_payload_size || nir->info.mesh.max_vertices_out != 3 ||
       nir->info.mesh.max_primitives_out != 1 ||
       nir->info.mesh.primitive_type != MESA_PRIM_TRIANGLES) {
      if (reason)
         *reason =
            "ordinary mesh formation currently supports Position-only V3/P1 triangles";
      return false;
   }
   if (!agx_nir_lower_apple9_mesh(nir, key, info, reason))
      return false;
   const struct agx_apple9_mesh_layout *l = &info->layout;
   nir_shader_intrinsics_pass(nir, prepare, nir_metadata_none, info);
   if (l->vertex_components != 4 || l->primitive_components || l->culling ||
       l->position_B != 32 || l->vertex_pitch_B != 16 ||
       l->local_size_B != 96) {
      if (reason)
         *reason =
            "ordinary mesh formation requires Position only and no primitive attributes";
      return false;
   }
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_set_vertex_and_primitive_count)
            nir_instr_remove(instr);
      }
   }
   nir_lower_returns(nir);
   if (!agx_nir_mask_apple9_mesh_body(nir, key, &info->layout, reason))
      return false;
   nir_builder b =
      nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   rendezvous(&b);
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_def *vertices = nir_load_local_apple9_agx(&b, nir_imm_int(&b, l->primitive_count_B + 4),
                                                 .base = key->output_root);
   nir_def *count =
      nir_iand_imm(&b,
                   nir_load_local_apple9_agx(&b, nir_imm_int(&b, l->primitive_count_B),
                                             .base = key->output_root),
                   0xffff);
   nir_def *ready = nir_ine_imm(&b, input(&b, AGX_MESH_RESERVATION_READY), 0);
   nir_push_if(&b, ready);
   nir_def *scene = input(&b, AGX_MESH_SCENE_ADDRESS);
   for (unsigned i = 0; i < 12; ++i)
      word(&b, scene, 4 * i, 0);
   nir_push_if(&b, nir_ine_imm(&b, count, 0));
   nir_def *indices = nir_load_local_apple9_agx(&b, nir_imm_int(&b, 0),
                                                .base = key->output_root);
   nir_def *index[3];
   nir_def *valid =
      nir_iand(&b, nir_ieq_imm(&b, count, 1), nir_ieq_imm(&b, vertices, 3));
   for (unsigned i = 0; i < 3; ++i) {
      index[i] = nir_iand_imm(&b, nir_ushr_imm(&b, indices, i * 8), 255);
      valid = nir_iand(&b, valid, nir_ult_imm(&b, index[i], 3));
   }
   struct agx_apple9_mesh_compaction compact =
      agx_nir_apple9_mesh_compact_triangle(&b, index);
   valid = nir_iand(&b, valid, nir_ieq_imm(&b, compact.count, 3));
   nir_push_if(&b, valid);
   nir_def *v[3][4], *p[3][3];
   nir_def *inside = nir_imm_true(&b);
   for (unsigned i = 0; i < 3; ++i) {
      for (unsigned c = 0; c < 4; ++c) {
         v[i][c] = nir_load_local_apple9_agx(
            &b,
            nir_iadd_imm(&b, nir_imul_imm(&b, index[i], 4),
                         l->position_B + l->vertex_pitch_B * c),
            .base = key->output_root);
         inside = nir_iand(&b, inside, finite_normal_or_zero(&b, v[i][c]));
      }
      inside = nir_iand(&b, inside, nir_flt(&b, nir_imm_float(&b, 0), v[i][3]));
      inside =
         nir_iand(&b, inside, nir_flt(&b, nir_fabs(&b, v[i][0]), v[i][3]));
      inside =
         nir_iand(&b, inside, nir_flt(&b, nir_fabs(&b, v[i][1]), v[i][3]));
      inside = nir_iand(&b, inside, nir_fge(&b, v[i][2], nir_imm_float(&b, 0)));
      inside = nir_iand(&b, inside, nir_fge(&b, v[i][3], v[i][2]));
   }
   /* Reject exceptional inputs before division and integer conversion. This
    * is a scope predicate for the known inside-clip class, not a clipper. */
   nir_push_if(&b, inside);
   nir_def *scale[3], *offset[3];
   for (unsigned c = 0; c < 3; ++c) {
      scale[c] = input(&b, AGX_MESH_VIEWPORT_SCALE_X + c);
      offset[c] = input(&b, AGX_MESH_VIEWPORT_OFFSET_X + c);
   }
   nir_def *screen[3][3];
   inside = nir_imm_true(&b);
   for (unsigned i = 0; i < 3; ++i) {
      for (unsigned c = 0; c < 3; ++c) {
         screen[i][c] =
            nir_ffma(&b, nir_fdiv(&b, v[i][c], v[i][3]), scale[c], offset[c]);
         inside = nir_iand(&b, inside, finite_normal_or_zero(&b, screen[i][c]));
         if (c < 2) {
            inside = nir_iand(
               &b, inside, nir_fge(&b, screen[i][c], nir_imm_float(&b, -4095)));
            inside = nir_iand(
               &b, inside, nir_flt(&b, screen[i][c], nir_imm_float(&b, 61439)));
         }
      }
   }
   nir_push_if(&b, inside);
   for (unsigned i = 0; i < 3; ++i) {
      for (unsigned c = 0; c < 2; ++c) {
         /* Measured fixed-point snapping: nearest, ties away from zero. */
         nir_def *magnitude = nir_f2u32(
            &b, nir_fadd_imm(
                   &b, nir_fmul_imm(&b, nir_fabs(&b, screen[i][c]), 256), .5));
         p[i][c] = nir_iadd_imm(
            &b,
            nir_bcsel(&b, nir_flt(&b, screen[i][c], nir_imm_float(&b, 0)),
                      nir_ineg(&b, magnitude), magnitude),
            AGX_MESH_XY_FIXED_POINT_BIAS);
      }
      p[i][2] = screen[i][2];
   }
   /* Products of signed 24-bit differences require 64 bits. */
   nir_def *area =
      nir_isub(&b,
               nir_imul(&b, nir_i2i64(&b, nir_isub(&b, p[1][0], p[0][0])),
                        nir_i2i64(&b, nir_isub(&b, p[2][1], p[0][1]))),
               nir_imul(&b, nir_i2i64(&b, nir_isub(&b, p[2][0], p[0][0])),
                        nir_i2i64(&b, nir_isub(&b, p[1][1], p[0][1]))));
   nir_push_if(&b, nir_ine_imm(&b, area, 0));
   struct agx_apple9_mesh_record_state state = {
      .fragment_launch = input(&b, AGX_MESH_FRAGMENT_LAUNCH),
      .depth_control = input(&b, AGX_MESH_DEPTH_CONTROL),
      .depth_face = input(&b, AGX_MESH_DEPTH_FACE),
      .depth_face_2 = input(&b, AGX_MESH_DEPTH_FACE_2),
      .stencil = input(&b, AGX_MESH_STENCIL),
      .scissor_bias = input(&b, AGX_MESH_SCISSOR_BIAS),
      .tile_control = input(&b, AGX_MESH_TILE_CONTROL),
      .front_ccw = input(&b, AGX_MESH_FRONT_CCW),
   };
   nir_def *g = input(&b, AGX_MESH_GEOMETRY_ADDRESS);
   struct agx_apple9_mesh_position_encoding encoded =
      agx_nir_apple9_mesh_encode_positions(&b, p);
   nir_def *header =
      agx_nir_apple9_mesh_write_header(&b, g, &state, &encoded, compact.count);
   nir_def *back =
      nir_ieq(&b, nir_b2i32(&b, nir_igt_imm(&b, area, 0)), state.front_ccw);
   store_at(&b, g, header,
            agx_nir_apple9_mesh_triangle_tokens(&b, &compact, back));
   for (unsigned i = 0; i < 8; ++i) {
      nir_push_if(&b, nir_ugt_imm(&b, encoded.bits, 32 * i));
      store_at(&b, g, nir_iadd_imm(&b, header, 4 + 4 * i), encoded.words[i]);
      nir_pop_if(&b, NULL);
   }
   nir_def *gid = input(&b, AGX_MESH_GEOMETRY_ID),
           *lid = input(&b, AGX_MESH_LIST_ID);
   nir_def *rid = input(&b, AGX_MESH_ROOT_ID);
   agx_nir_apple9_mesh_write_short_list(&b, input(&b, AGX_MESH_LIST_ADDRESS),
                                        gid);
   fence(&b);
   agx_nir_apple9_mesh_write_short_root(&b, input(&b, AGX_MESH_ROOT_ADDRESS),
                                        rid, gid, lid);
   store(&b, scene, 8, rid);
   word(&b, scene, 12, AGX_MESH_SHORT_ROOT_COUNT <<
                          AGX_MESH_SCENE_ROOT_COUNT_SHIFT);
   fence(&b);
   store(&b, scene, 0, rid);
   word(&b, scene, 4, AGX_MESH_OBS_SCENE_ROOT_FLAGS);
   for (unsigned i = 0; i < 6; ++i)
      nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);
   fence(&b);
   rendezvous(&b);
   nir_progress(true, nir_shader_get_entrypoint(nir), nir_metadata_none);
   nir_validate_shader(nir, "ordinary Apple9 mesh formation");
   return true;
}
