/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Independent DATA reader and source/raster oracle. This accepts every mode in
 * the bounded byte-channel grammar; it never predicts the encoder's choices.
 */
#pragma once
#include <math.h>

struct decoded_geometry {
   uint32_t position[3][3];
   uint32_t launch, control, face, stencil, scissor, tile;
   unsigned header, bytes;
   bool backface;
};

static uint32_t
read32(const uint8_t *p)
{
   return p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
          ((uint32_t)p[3] << 24);
}

static void
write32(uint8_t *p, uint32_t v)
{
   for (unsigned i = 0; i < 4; ++i)
      p[i] = v >> (8 * i);
}

static bool
decode_geometry(const uint8_t *data, unsigned size, struct decoded_geometry *g)
{
   memset(g, 0, sizeof(*g));
   if (size < 44 || read32(data) || read32(data + 4) || read32(data + 8) ||
       read32(data + 12) != 0x01000000)
      return false;
   g->launch = read32(data + 16);
   g->control = read32(data + 20);
   g->face = read32(data + 24);
   unsigned p = 28;
   if (g->control & (1u << 18)) {
      g->stencil = read32(data + p);
      p += 4;
   }
   if (g->control & (3u << 16)) {
      g->scissor = read32(data + p);
      p += 4;
   }
   if (size < p + 20)
      return false;
   g->tile = read32(data + p);
   if (read32(data + p + 4))
      return false;
   uint32_t high = read32(data + p + 12);
   if ((high >> 8) != 2) /* This reader's class has three references. */
      return false;
   uint64_t modes = read32(data + p + 8) | ((uint64_t)(high & 255) << 32);
   g->header = p + 16;
   const uint8_t *token = data + g->header;
   if (token[3] || (token[0] & 0xc0) != 0x40 || (token[1] & 0xc0) != 0x40 ||
       (token[2] & 0x40) != 0x40)
      return false;
   unsigned ref[3] = {token[0] & 63, token[1] & 63, token[2] & 63};
   if (ref[0] > 2 || ref[1] > 2 || ref[2] > 2 || ref[0] == ref[1] ||
       ref[1] == ref[2] || ref[2] == ref[0])
      return false;
   g->backface = token[2] & 0x80;
   unsigned width[10], mode[10], bases[10][2] = {{0}};
   p = g->header + 4;
   for (unsigned c = 0; c < 10; ++c) {
      mode[c] = (modes >> (4 * c)) & 15;
      width[c] = mode[c] <= 8 ? mode[c] : mode[c] - 8;
      unsigned count = mode[c] < 8 ? 1 : mode[c] == 8 ? 0 : 2;
      if (p + count > size)
         return false;
      for (unsigned j = 0; j < count; ++j)
         bases[c][j] = data[p++];
   }
   unsigned bit = p * 8;
   uint32_t xyz[3][3] = {{0}};
   for (unsigned r = 0; r < 3; ++r) {
      for (unsigned c = 0; c < 10; ++c) {
         unsigned symbol = 0;
         if (bit + width[c] > size * 8)
            return false;
         for (unsigned j = 0; j < width[c]; ++j, ++bit)
            symbol |= ((data[bit / 8] >> (bit % 8)) & 1) << j;
         unsigned byte = mode[c] == 8  ? symbol
                         : mode[c] < 8 ? bases[c][0] + symbol
                                       : bases[c][symbol & 1] + (symbol >> 1);
         unsigned axis = c < 3 ? 0 : c < 6 ? 1 : 2;
         unsigned shift = 8 * (c - (axis == 0 ? 0 : axis == 1 ? 3 : 6));
         xyz[r][axis] |= (byte & 255) << shift;
      }
   }
   g->bytes = ((bit + 31) / 32) * 4;
   if (g->bytes != size)
      return false;
   for (; bit < size * 8; ++bit)
      if ((data[bit / 8] >> (bit % 8)) & 1)
         return false;
   for (unsigned r = 0; r < 3; ++r)
      memcpy(g->position[r], xyz[ref[r]], sizeof(xyz[0]));
   return true;
}

static float
as_float(uint32_t u)
{
   float f;
   memcpy(&f, &u, 4);
   return f;
}

static uint32_t
as_uint(float f)
{
   uint32_t u;
   memcpy(&u, &f, 4);
   return u;
}

static int64_t
triangle_area(const uint32_t p[3][3])
{
   return ((int64_t)p[1][0] - p[0][0]) * ((int64_t)p[2][1] - p[0][1]) -
          ((int64_t)p[2][0] - p[0][0]) * ((int64_t)p[1][1] - p[0][1]);
}

/* Source-level projection with explicit float32 rounding, followed by the
 * measured coordinate quantization. No packet data enters this computation. */
static bool
source_geometry(const struct formation_fixture *f, uint32_t xyz[3][3],
                bool *drop)
{
   *drop = false;
   if (!f->primitives)
      return false;
   *drop = true;
   if (f->vertices != 3 || f->primitives != 1 || f->indices[0] > 2 ||
       f->indices[1] > 2 || f->indices[2] > 2 ||
       f->indices[0] == f->indices[1] || f->indices[1] == f->indices[2] ||
       f->indices[2] == f->indices[0])
      return false;
   for (unsigned r = 0; r < 3; ++r) {
      float v[4];
      for (unsigned c = 0; c < 4; ++c) {
         v[c] = as_float(f->positions[4 * f->indices[r] + c]);
         if (!isfinite(v[c]) || (v[c] != 0 && !isnormal(v[c])))
            return false;
      }
      if (!(v[3] > 0 && fabsf(v[0]) < v[3] && fabsf(v[1]) < v[3] && v[2] >= 0 &&
            v[2] <= v[3]))
         return false;
      for (unsigned c = 0; c < 3; ++c) {
         float projected = v[c] / v[3];
         float screen = fmaf(projected, as_float(f->viewport[c]),
                             as_float(f->viewport[c + 3]));
         if (!isfinite(screen) || (screen != 0 && !isnormal(screen)))
            return false;
         if (c < 2) {
            if (screen < -4095 || screen >= 61439)
               return false;
            xyz[r][c] =
               (int32_t)copysignf(floorf(fabsf(screen) * 256 + .5f), screen) +
               0x100000;
         } else {
            xyz[r][c] = as_uint(screen);
         }
      }
   }
   *drop = false;
   return triangle_area(xyz) != 0;
}

/* Independent software coverage at 64x64 sample centres, using integer edge
 * equations and a top-left tie rule. Hardware pixel evidence remains required. */
static void
raster(const uint32_t xyz[3][3], uint8_t pixels[64 * 64])
{
   int order[3] = {0, 1, 2};
   if (triangle_area(xyz) < 0) {
      order[1] = 2;
      order[2] = 1;
   }
   for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
         bool covered = true;
         for (unsigned e = 0; e < 3; ++e) {
            const uint32_t *a = xyz[order[e]], *b = xyz[order[(e + 1) % 3]];
            int64_t dx = (int64_t)b[0] - a[0], dy = (int64_t)b[1] - a[1];
            int64_t edge = dx * (0x100000 + 256 * y + 128 - (int64_t)a[1]) -
                           dy * (0x100000 + 256 * x + 128 - (int64_t)a[0]);
            covered &= edge > 0 || (edge == 0 && (dy < 0 || (!dy && dx > 0)));
         }
         pixels[y * 64 + x] = covered;
      }
   }
}

static bool
geometry_matches(const struct formation_fixture *f,
                 const struct decoded_geometry *g, const uint32_t xyz[3][3])
{
   return !memcmp(g->position, xyz, sizeof(g->position)) &&
          g->launch == f->state[0] &&
          g->control == (f->state[1] | f->state[7]) &&
          g->face == (f->state[3] | (f->state[2] & 0x3ffff)) &&
          g->stencil == ((f->state[1] & (1u << 18)) ? f->state[4] : 0) &&
          g->scissor == ((f->state[1] & (3u << 16)) ? f->state[5] : 0) &&
          g->tile == f->state[6] &&
          g->backface == ((triangle_area(xyz) > 0) == f->state[7]);
}

static void
check_geometry(const struct formation_fixture *f, const uint8_t *data,
               unsigned size, const uint32_t xyz[3][3])
{
   struct decoded_geometry g;
   CHECK(decode_geometry(data, size, &g));
   CHECK(geometry_matches(f, &g, xyz));
   uint8_t expected[4096], actual[4096];
   raster(xyz, expected);
   raster(g.position, actual);
   CHECK(!memcmp(expected, actual, sizeof(actual)));
   if (!strncmp(f->name, "H1_", 3)) {
      unsigned pixels = 0;
      for (unsigned i = 0; i < sizeof(actual); ++i)
         pixels += actual[i];
      CHECK(pixels == 512);
   }

   /* A different, all-raw encoding must pass the same oracle. This keeps mode
    * selection and prefix heuristics out of the production test contract. */
   uint8_t raw[88] = {0};
   memcpy(raw, data, g.header);
   write32(raw + g.header - 8, 0x88888888);
   write32(raw + g.header - 4, 0x288);
   raw[g.header] = 0x40;
   raw[g.header + 1] = 0x41;
   raw[g.header + 2] = 0x42 | (g.backface ? 0x80 : 0);
   unsigned p = g.header + 4;
   for (unsigned r = 0; r < 3; ++r)
      for (unsigned c = 0; c < 3; ++c)
         for (unsigned j = 0; j < (c < 2 ? 3 : 4); ++j)
            raw[p++] = xyz[r][c] >> (8 * j);
   CHECK(decode_geometry(raw, g.header + 36, &g));
   CHECK(geometry_matches(f, &g, xyz));
   raw[g.header + 4] ^= 1;
   CHECK(decode_geometry(raw, g.header + 36, &g));
   CHECK(!geometry_matches(f, &g, xyz));
   raw[g.header + 3] = 1;
   CHECK(!decode_geometry(raw, size, &g));
}

static bool
references_match(const struct formation_fixture *f, const struct machine *m)
{
   const uint8_t *list = m->region[1].data, *root = m->region[2].data;
   const uint8_t *scene = m->region[3].data;
   if (read32(list) != 0x2026 || read32(list + 4) != (f->ids[0] << 5) ||
       read32(list + 8) != 0x00a80101 || read32(root) != 2 ||
       read32(root + 4) || read32(root + 8) != (f->ids[2] << 1) ||
       read32(root + 12) || read32(root + 16) != f->ids[0] ||
       read32(root + 20) != (f->ids[1] << 9) ||
       read32(root + 24) != (f->ids[1] >> 23) ||
       read32(root + 28) != (3u << 27) || read32(scene) != f->ids[2] ||
       read32(scene + 4) != 0x80000000 || read32(scene + 8) != f->ids[2] ||
       read32(scene + 12) != 0x200)
      return false;
   for (unsigned i = 16; i < 48; ++i)
      if (scene[i])
         return false;
   return true;
}
