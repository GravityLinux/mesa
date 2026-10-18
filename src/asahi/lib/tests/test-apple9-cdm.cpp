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

/* The emitters refuse exactly the records agx_apple9_dispatch_reject_reason
 * names, and the reason is what hk prints for "dispatch failed (6)". */
TEST(Apple9CDM, RejectReasonMatchesEmitters)
{
   agx_apple9_compute_profile profile = {};
   profile.abi = AGX_APPLE9_COMPUTE_ABI_DIRECT_BUFFERS;
   profile.resource_binding_count = 1;
   profile.resource_write_mask = 1;
   profile.local_size[0] = profile.local_size[1] = profile.local_size[2] = 1;
   const uint64_t launch = UINT64_C(1) << 40;
   const uint32_t global[] = {8, 1, 1}, zero_axis[] = {8, 0, 1};
   uint32_t out[16];

   EXPECT_EQ(agx_apple9_dispatch_reject_reason(launch, false, 0, global,
                                               profile.local_size, &profile),
             nullptr);
   EXPECT_EQ(agx_apple9_dispatch_reject_reason(launch, true, 0x1234000, NULL,
                                               profile.local_size, &profile),
             nullptr);

   struct {
      bool indirect;
      uint64_t launch, address;
      const uint32_t *global;
      const char *reason;
   } cases[] = {
      {true, launch, 0, NULL, "null indirect address"},
      {true, launch, 0x1234002, NULL, "indirect address not 4-byte aligned"},
      {false, launch, 0, zero_axis, "zero grid axis"},
      {false, launch + 0x20, 0, global, "launch address not 64-byte aligned"},
   };
   for (const auto &c : cases) {
      const char *reason = agx_apple9_dispatch_reject_reason(
         c.launch, c.indirect, c.address, c.global, profile.local_size, &profile);
      ASSERT_NE(reason, nullptr) << c.reason;
      EXPECT_STREQ(reason, c.reason);
      bool ok = c.indirect
         ? agx_apple9_emit_indirect_dispatch(out, c.launch, c.address,
                                             profile.local_size, false, &profile,
                                             false)
         : agx_apple9_emit_direct_dispatch(out, c.launch, c.global,
                                           profile.local_size, &profile, false);
      EXPECT_FALSE(ok) << c.reason;
   }
}
