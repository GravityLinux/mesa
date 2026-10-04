/*
 * Copyright 2026 Niklas Sheth
 * SPDX-License-Identifier: MIT
 */

#include "asahi/compiler/agx_compile_apple9.h"
#include "compiler/glsl_types.h"
#include "agx_apple9_libagx.h"

#include <gtest/gtest.h>

/* Every libagx program a driver can dispatch on Apple9 must compile with the
 * Apple9 compute compiler, or the driver aborts the first time it is used.
 */
extern "C" {
unsigned test_libagx_num_programs(void);
bool test_libagx_expected_to_fail(unsigned program);
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
