/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Application positions and indices. No runtime addresses or state records.
 */
struct native_mesh_fixture {
   const char *name;
   unsigned vertices, primitives;
   uint32_t positions[12], indices[3];
};
static const struct native_mesh_fixture h1_fixtures[] = {
   {"H1_BASE", 3, 1,
      {0xbf000000, 0xbf000000, 0x3f000000, 0x3f800000, 0x3f000000, 0xbf000000, 0x3f000000, 0x3f800000, 0x0, 0x3f000000, 0x3f000000, 0x3f800000},
      {0x0, 0x1, 0x2}},
   {"H1_SHIFT_X", 3, 1,
      {0xbec00000, 0xbf000000, 0x3f000000, 0x3f800000, 0x3f200000, 0xbf000000, 0x3f000000, 0x3f800000, 0x3e000000, 0x3f000000, 0x3f000000, 0x3f800000},
      {0x0, 0x1, 0x2}},
};
