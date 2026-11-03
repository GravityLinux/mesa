/* SPDX-License-Identifier: MIT */

#include <cstdlib>
#include <gtest/gtest.h>

#include "hk_drirc.h"

static bool
vertex_stores(const char *device, const char *engine)
{
   struct hk_drirc drirc;
   driConfigFileParseParams params = {};
   params.driverName = "hk";
   params.deviceName = device;
   params.engineName = engine;
   hk_parse_dri_options(&drirc, &params);
   bool enabled = drirc.misc.enable_vertex_pipeline_stores_atomics;
   driDestroyOptionCache(&drirc.options);
   driDestroyOptionInfo(&drirc.available_options);
   return enabled;
}

TEST(HkDrirc, VertexStoresDefaultOnG16Only)
{
   unsetenv("hk_enable_vertex_pipeline_stores_atomics");
   EXPECT_FALSE(vertex_stores(nullptr, "mesa zink"));
   EXPECT_FALSE(vertex_stores("G13", "mesa zink"));
   EXPECT_FALSE(vertex_stores("G14", "mesa zink"));
   EXPECT_FALSE(vertex_stores("G13", nullptr));
   EXPECT_TRUE(vertex_stores("G16", "mesa zink"));
   EXPECT_TRUE(vertex_stores("G16", nullptr));
   EXPECT_TRUE(vertex_stores("G16", "mesa zink other"));
   EXPECT_TRUE(vertex_stores("G16", "other mesa zink"));
}

TEST(HkDrirc, ProtonVertexStoresRemainEnabled)
{
   unsetenv("hk_enable_vertex_pipeline_stores_atomics");
   const char *devices[] = {nullptr, "G13", "G16"};
   for (const char *device : devices) {
      EXPECT_TRUE(vertex_stores(device, "DXVK"));
      EXPECT_TRUE(vertex_stores(device, "vkd3d"));
   }
}

TEST(HkDrirc, VertexStoresEnvironmentOverride)
{
   const char *option = "hk_enable_vertex_pipeline_stores_atomics";
   setenv(option, "false", 1);
   EXPECT_FALSE(vertex_stores("G16", "mesa zink"));
   EXPECT_FALSE(vertex_stores("G16", "DXVK"));
   unsetenv(option);

   setenv(option, "true", 1);
   EXPECT_TRUE(vertex_stores("G13", "mesa zink"));
   EXPECT_TRUE(vertex_stores("G16", nullptr));
   unsetenv(option);
}

TEST(HkDrirc, NativeIndexKillSwitch)
{
   const char *option = "hk_disable_native_index";
   for (const char *value : {"false", "true"}) {
      setenv(option, value, 1);
      struct hk_drirc drirc;
      driConfigFileParseParams params = {};
      params.driverName = "hk";
      params.deviceName = "G16";
      hk_parse_dri_options(&drirc, &params);
      EXPECT_EQ(drirc.misc.disable_native_index, value[0] == 't');
      driDestroyOptionCache(&drirc.options);
      driDestroyOptionInfo(&drirc.available_options);
   }
   unsetenv(option);
   struct hk_drirc drirc;
   driConfigFileParseParams params = {};
   params.driverName = "hk";
   params.deviceName = "G16";
   hk_parse_dri_options(&drirc, &params);
   EXPECT_FALSE(drirc.misc.disable_native_index);
   driDestroyOptionCache(&drirc.options);
   driDestroyOptionInfo(&drirc.available_options);
}

TEST(HkDrirc, NativeStageKillSwitch)
{
   const char *option = "hk_disable_native_stages";
   for (const char *value : {"false", "true"}) {
      setenv(option, value, 1);
      struct hk_drirc drirc;
      driConfigFileParseParams params = {};
      params.driverName = "hk";
      params.deviceName = "G16";
      hk_parse_dri_options(&drirc, &params);
      EXPECT_EQ(drirc.misc.disable_native_stages, value[0] == 't');
      driDestroyOptionCache(&drirc.options);
      driDestroyOptionInfo(&drirc.available_options);
   }
   unsetenv(option);
   struct hk_drirc drirc;
   driConfigFileParseParams params = {};
   params.driverName = "hk";
   params.deviceName = "G16";
   hk_parse_dri_options(&drirc, &params);
   EXPECT_FALSE(drirc.misc.disable_native_stages);
   driDestroyOptionCache(&drirc.options);
   driDestroyOptionInfo(&drirc.available_options);
}

TEST(HkDrirc, DefaultsLoadedForEveryApplication)
{
   const char *devices[] = {nullptr, "G13", "G16"};
   const char *engines[] = {nullptr, "mesa zink", "vkd3d", "other"};
   for (const char *device : devices) {
      for (const char *engine : engines) {
         struct hk_drirc drirc;
         driConfigFileParseParams params = {};
         params.driverName = "hk";
         params.deviceName = device;
         params.engineName = engine;
         hk_parse_dri_options(&drirc, &params);
         EXPECT_TRUE(drirc.misc.defaults_loaded);
         driDestroyOptionCache(&drirc.options);
         driDestroyOptionInfo(&drirc.available_options);
      }
   }
}
