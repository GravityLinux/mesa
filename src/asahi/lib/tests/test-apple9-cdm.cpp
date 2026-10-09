/* SPDX-License-Identifier: MIT */
#include "agx_apple9.h"
#include <algorithm>
#include <iterator>
#include <gtest/gtest.h>

TEST(Apple9CDM, OptionalBarrierKeepsLaunchAndOutputBounds)
{
   agx_apple9_compute_profile profile = {};
   profile.abi = AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS;
   profile.resource_binding_count = 1;
   profile.resource_write_mask = 1;
   profile.local_size[0] = profile.local_size[1] = profile.local_size[2] = 1;
   const uint32_t global[] = {8, 1, 1};
   const uint64_t launch = UINT64_C(1) << 40;

   for (unsigned mode = 0; mode < 3; ++mode) {
      uint32_t fenced[16], unfenced[16];
      std::fill(std::begin(fenced), std::end(fenced), 0xdeadbeefu);
      std::fill(std::begin(unfenced), std::end(unfenced), 0xdeadbeefu);
      bool ok_fenced, ok_unfenced;
      unsigned size;
      if (mode == 0) {
         size = AGX_APPLE9_COMPUTE_CDM_RECORD_SIZE;
         ok_fenced = agx_apple9_emit_direct_dispatch(
            fenced, launch, global, profile.local_size, &profile, true);
         ok_unfenced = agx_apple9_emit_direct_dispatch(
            unfenced, launch, global, profile.local_size, &profile, false);
      } else {
         size = mode == 1 ? AGX_APPLE9_COMPUTE_INDIRECT_CDM_RECORD_SIZE :
                            AGX_APPLE9_COMPUTE_INDIRECT_LOCAL_CDM_RECORD_SIZE;
         ok_fenced = agx_apple9_emit_indirect_dispatch(
            fenced, launch, 0x1234000, profile.local_size, mode == 2, &profile, true);
         ok_unfenced = agx_apple9_emit_indirect_dispatch(
            unfenced, launch, 0x1234000, profile.local_size, mode == 2, &profile, false);
      }
      ASSERT_TRUE(ok_fenced);
      ASSERT_TRUE(ok_unfenced);
      for (unsigned i = 0; i + 1 < size / 4; ++i)
         EXPECT_EQ(fenced[i], unfenced[i]);
      EXPECT_EQ(fenced[size / 4 - 1], 0x600fffffu);
      EXPECT_EQ(unfenced[size / 4 - 1], 0xdeadbeefu);
      EXPECT_EQ(fenced[size / 4], 0xdeadbeefu);
      EXPECT_EQ(unfenced[size / 4], 0xdeadbeefu);

      /* The compute-only barrier replaces just the trailing word. */
      agx_apple9_set_compute_barrier(fenced, size);
      for (unsigned i = 0; i + 1 < size / 4; ++i)
         EXPECT_EQ(fenced[i], unfenced[i]);
      EXPECT_EQ(fenced[size / 4 - 1], 0x60000160u);
      EXPECT_EQ(fenced[size / 4], 0xdeadbeefu);
   }
}
