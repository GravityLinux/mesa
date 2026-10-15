/*
 * Copyright 2026 Niklas Sheth
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx_compile_apple9.h"
#include "compiler/glsl_types.h"
#include "agx_apple9_libagx.h"
#include "agx_apple9.h"
#include <array>

#include <gtest/gtest.h>

/* Every libagx program a driver can dispatch on Apple9 must compile with the
 * Apple9 compute compiler, or the driver aborts the first time it is used.
 */
extern "C" {
unsigned test_libagx_num_programs(void);
bool test_libagx_expected_to_fail(unsigned program);
unsigned test_libagx_fast_clear_4x(void);
}

class LibagxApple9 : public testing::TestWithParam<unsigned> {
 protected:
   static void SetUpTestSuite() { glsl_type_singleton_init_or_ref(); }
   static void TearDownTestSuite() { glsl_type_singleton_decref(); }
};

TEST_P(LibagxApple9, Compiles)
{
   unsigned program = GetParam();
   nir_shader *nir = agx_apple9_libagx_nir(program, NULL);
   ASSERT_EQ(nir->info.stage, MESA_SHADER_COMPUTE);
   if (program == test_libagx_fast_clear_4x()) {
      EXPECT_EQ(nir->info.workgroup_size[0], 32u);
      EXPECT_EQ(nir->info.workgroup_size[1], 1u);
      EXPECT_EQ(nir->info.workgroup_size[2], 1u);
      EXPECT_EQ(nir->info.shared_size, 0u);
      EXPECT_EQ(nir->scratch_size, 0u);
   }
   struct agx_shader_part part = {};
   struct agx_apple9_compute_profile profile = {};
   const char *reason = NULL;
   bool ok = agx_compile_apple9_tiny(nir, &part, &profile, &reason);

   if (test_libagx_expected_to_fail(program))
      EXPECT_FALSE(ok) << "now compiles; remove it from expected_to_fail";
   else
      EXPECT_TRUE(ok) << (reason ? reason : "unknown reason");

   free(part.binary);
   ralloc_free(nir);
}

INSTANTIATE_TEST_SUITE_P(All, LibagxApple9,
                         testing::Range(0u, test_libagx_num_programs()));

/* Guard both ends: these sizes are also the command allocator reservation. */
TEST(Apple9Dispatch, PacketFootprints)
{
   agx_apple9_compute_profile profile = {};
   profile.abi = AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS;
   profile.local_size[0] = profile.local_size[1] = profile.local_size[2] = 1;
   profile.resource_binding_count = 1;
   profile.resource_kind[0] = AGX_APPLE9_COMPUTE_RESOURCE_SSBO;
   profile.resource_write_mask = 1;
   const uint32_t global[] = {3, 5, 7};
   const uint32_t local[] = {1, 1, 1};
   const unsigned sizes[] = {AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE,
      AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE,
      AGX_APPLE9_COMPUTE_INDIRECT_LOCAL_CDM_RECORD_SIZE};
   for (unsigned mode = 0; mode < 3; ++mode) {
      std::array<uint8_t, 76> out;
      out.fill(0xa5);
      bool ok = mode == 0
         ? agx_apple9_emit_direct_dispatch(out.data() + 16, 0x10000010000ull,
                                          global, local, &profile, true)
         : agx_apple9_emit_indirect_dispatch(out.data() + 16, 0x10000010000ull,
                                            0x200000000ull, local, mode == 2,
                                            &profile, true);
      ASSERT_TRUE(ok);
      for (unsigned i = 0; i < 16; ++i)
         EXPECT_EQ(out[i], 0xa5);
      for (unsigned i = 16 + sizes[mode]; i < out.size(); ++i)
         EXPECT_EQ(out[i], 0xa5);
      uint32_t tail;
      memcpy(&tail, out.data() + 16 + sizes[mode] - 4, sizeof(tail));
      EXPECT_EQ(tail, 0x600fffffu);
   }
}

/* Consecutive draws with the same compiled stages reuse launches and the
 * coefficient table; a stage without a program id never does (hk once left
 * every id at zero, which silently disabled the reuse). */
TEST(Apple9DrawReuse, StablePairReusesLaunchesAndCoefficients)
{
   agx_apple9_uniform_draw a = {}, b = {};
   for (unsigned s = 0; s < 2; ++s) {
      a.program_id[s] = b.program_id[s] = 7 + s;
      a.code[s] = b.code[s] = 0x10000100000ull + s * 0x1000;
      a.texture_table[s] = b.texture_table[s] = 0x10000000000ull;
      a.sampler_table[s] = b.sampler_table[s] = 0x10000000000ull;
   }
   a.vertex_table = b.vertex_table = 0x1000;
   a.fragment_table = b.fragment_table = 0x2000;
   a.samples = b.samples = 1;
   a.tile_bytes = b.tile_bytes = 4;
   a.provoking_vertex = b.provoking_vertex = 0;
   EXPECT_TRUE(agx_apple9_draw_reuses_launch(&a, &b, 0));
   EXPECT_TRUE(agx_apple9_draw_reuses_launch(&a, &b, 1));
   EXPECT_TRUE(agx_apple9_draw_reuses_coefficients(&a, &b));

   /* A changed stage table only affects that stage's launch. */
   b.vertex_table = 0x3000;
   EXPECT_FALSE(agx_apple9_draw_reuses_launch(&a, &b, 0));
   EXPECT_TRUE(agx_apple9_draw_reuses_launch(&a, &b, 1));
   EXPECT_TRUE(agx_apple9_draw_reuses_coefficients(&a, &b));
   b.vertex_table = a.vertex_table;

   /* Tile allocation and sample count only matter to the fragment launch. */
   b.tile_bytes = 8;
   EXPECT_TRUE(agx_apple9_draw_reuses_launch(&a, &b, 0));
   EXPECT_FALSE(agx_apple9_draw_reuses_launch(&a, &b, 1));
   b.tile_bytes = a.tile_bytes;

   EXPECT_FALSE(agx_apple9_draw_reuses_launch(nullptr, &b, 0));
   EXPECT_FALSE(agx_apple9_draw_reuses_coefficients(nullptr, &b));
}

TEST(Apple9DrawReuse, ZeroProgramIdNeverReuses)
{
   agx_apple9_uniform_draw a = {}, b = {};
   EXPECT_FALSE(agx_apple9_draw_reuses_launch(&a, &b, 0));
   EXPECT_FALSE(agx_apple9_draw_reuses_launch(&a, &b, 1));
   EXPECT_FALSE(agx_apple9_draw_reuses_coefficients(&a, &b));
   a.program_id[0] = b.program_id[0] = 3;
   EXPECT_TRUE(agx_apple9_draw_reuses_launch(&a, &b, 0));
   EXPECT_FALSE(agx_apple9_draw_reuses_coefficients(&a, &b));
}
