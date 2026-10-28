/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include <spirv-tools/libspirv.hpp>
#include <string>
#include <vector>
#include <unistd.h>
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "asahi/compiler/agx_compile_apple9.h"

/* Historical filename; these are importer tests, not a raw module scanner.
 * Keep CooperativeMatrixKHR physically unsupported: the real importer warns
 * for implemented capabilities and still imports, as the shared runtime does.
 */
class Apple9SpirvImporter : public ::testing::Test {
protected:
   spirv_capabilities caps = {};
   spirv_to_nir_options options = {};
   std::string diagnostic;

   void SetUp() override
   {
      glsl_type_singleton_init_or_ref();
      caps.Shader = true;
      caps.VulkanMemoryModel = true;
      caps.VulkanMemoryModelDeviceScope = true;
      options.capabilities = &caps;
      options.ssbo_addr_format = nir_address_format_32bit_index_offset;
      options.shared_addr_format = nir_address_format_32bit_offset;
      options.skip_os_break_in_debug_build = true;
      options.ignore_replacement = true;
      options.debug.private_data = &diagnostic;
      options.debug.func = [](void *data, nir_spirv_debug_level level,
                              size_t, const char *message) {
         if (level == NIR_SPIRV_DEBUG_LEVEL_ERROR)
            *static_cast<std::string *>(data) += message;
      };
   }

   void TearDown() override { glsl_type_singleton_decref(); }

   std::vector<uint32_t> module(const std::string &load,
                                const std::string &store,
                                const std::string &decoration = "",
                                bool member = false, bool vulkan = false,
                                bool store_only = false)
   {
      std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixKHR
OpCapability VulkanMemoryModel
OpCapability VulkanMemoryModelDeviceScope
OpExtension "SPV_KHR_cooperative_matrix"
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical )" + std::string(vulkan ? "Vulkan\n" : "GLSL450\n") + R"(
OpEntryPoint GLCompute %main "main" %buf %dst
OpExecutionMode %main LocalSize 32 1 1
OpDecorate %array ArrayStride 4
OpMemberDecorate %block 0 Offset 0
OpDecorate %block Block
OpDecorate %buf DescriptorSet 0
OpDecorate %buf Binding 0
OpDecorate %dst DescriptorSet 0
OpDecorate %dst Binding 1
)";
      if (!decoration.empty())
         text += member ? "OpMemberDecorate %block 0 " + decoration + "\n"
                        : std::string("OpDecorate ") +
                             (store_only ? "%dst " : "%buf ") + decoration + "\n";
      text += R"(
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%float = OpTypeFloat 32
%zero = OpConstant %u32 0
%two = OpConstant %u32 2
%three = OpConstant %u32 3
%eight = OpConstant %u32 8
%sixtyfour = OpConstant %u32 64
%array = OpTypeArray %float %sixtyfour
%block = OpTypeStruct %array
%pblock = OpTypePointer StorageBuffer %block
%pf = OpTypePointer StorageBuffer %float
%buf = OpVariable %pblock StorageBuffer
%dst = OpVariable %pblock StorageBuffer
%mat = OpTypeCooperativeMatrixKHR %float %three %eight %eight %two
%main = OpFunction %void None %fn
%label = OpLabel
%p = OpAccessChain %pf %buf %zero %zero
%q = OpAccessChain %pf %dst %zero %zero
%m = OpCooperativeMatrixLoadKHR %mat %p %zero %eight )" + load + R"(
OpCooperativeMatrixStoreKHR %q %m %zero %eight )" + store + R"(
OpReturn
OpFunctionEnd
)";
      spvtools::SpirvTools assembler(SPV_ENV_VULKAN_1_2);
      std::vector<uint32_t> words;
      EXPECT_TRUE(assembler.Assemble(text, &words)) << text;
      return words;
   }

   nir_shader *import(const std::vector<uint32_t> &words, bool restrictive)
   {
      diagnostic.clear();
      options.require_cmat_memory_access_none_or_aligned = restrictive;
      return spirv_to_nir(words.data(), words.size(), nullptr,
                           MESA_SHADER_COMPUTE, "main", &options,
                           &agx_nir_options);
   }

   void accepted(const std::vector<uint32_t> &words, bool restrictive,
                 bool barrier = false)
   {
      nir_shader *nir = import(words, restrictive);
      ASSERT_NE(nir, nullptr) << diagnostic;
      EXPECT_TRUE(nir->info.cs.has_cooperative_matrix);
      nir_validate_shader(nir, "actual cooperative matrix importer");
      if (barrier) {
         bool found = false;
         nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
            nir_foreach_instr(instr, block)
               found |= instr->type == nir_instr_type_intrinsic &&
                        nir_instr_as_intrinsic(instr)->intrinsic ==
                           nir_intrinsic_barrier;
         }
         EXPECT_TRUE(found);
      }
      ralloc_free(nir);
   }

   void rejected(const std::vector<uint32_t> &words, bool diagnosed = true)
   {
      nir_shader *nir = import(words, true);
      EXPECT_EQ(nir, nullptr);
      if (diagnosed) {
         EXPECT_FALSE(diagnostic.empty());
      }
      ralloc_free(nir);
   }
};

TEST_F(Apple9SpirvImporter, OmittedNoneAndAligned)
{
   for (const char *access : {"", "None", "Aligned 16"}) {
      auto words = module(access, access);
      accepted(words, false);
      accepted(words, true);
   }
}

TEST_F(Apple9SpirvImporter, ExplicitUnsupportedMemoryMasks)
{
   for (const char *access : {"Volatile", "Nontemporal", "Volatile|Aligned 16",
                             "NonPrivatePointer",
                             "MakePointerVisible|NonPrivatePointer %two"}) {
      auto words = module(access, "None", "", false, true);
      accepted(words, false, std::string(access).find("Visible") != std::string::npos);
      rejected(words);
   }
   for (const char *access : {"Volatile", "Nontemporal", "Nontemporal|Aligned 16",
                             "NonPrivatePointer",
                             "MakePointerAvailable|NonPrivatePointer %two"}) {
      auto words = module("None", access, "", false, true);
      accepted(words, false, std::string(access).find("Available") != std::string::npos);
      rejected(words);
   }
}

TEST_F(Apple9SpirvImporter, VariableAndMemberPointerQualifiers)
{
   for (bool member : {false, true}) {
      for (const char *qualifier : {"Volatile", "Coherent"}) {
         auto words = module("None", "None", qualifier, member);
         accepted(words, false);
         rejected(words);
      }
   }
}

TEST_F(Apple9SpirvImporter, QualifiedStorePointer)
{
   /* Read-only load pointer and a separately qualified store access chain. */
   for (const char *qualifier : {"Volatile", "Coherent"}) {
      auto words = module("None", "None", qualifier, false, false, true);
      accepted(words, false);
      rejected(words);
   }
}

TEST_F(Apple9SpirvImporter, InvalidAndTruncatedRealBoundary)
{
   auto valid = module("None", "None");
   /* Isolate a malformed header so an importer crash is a reported failure,
    * not the loss of every other test's evidence. No fake pre-validation.
    */
   for (unsigned word : {0u, 1u, 4u}) {
      auto invalid = valid;
      invalid[word] = word == 4 ? 1 : 0;
      EXPECT_EXIT({
         nir_shader *nir = import(invalid, true);
         bool failed = nir != nullptr || diagnostic.empty();
         ralloc_free(nir);
         _exit(failed);
      }, ::testing::ExitedWithCode(0), "");
   }
   auto short_header = valid;
   short_header.resize(5);
   rejected(short_header, false);
   auto truncated = valid;
   /* End in a declared instruction whose operands are missing. */
   truncated.push_back((4u << 16) | SpvOpDecorate);
   truncated.push_back(1);
   rejected(truncated);
   auto zero_length = valid;
   zero_length[5] &= 0xffffu;
   rejected(zero_length);

   for (auto opcode : {SpvOpCooperativeMatrixLoadKHR,
                       SpvOpCooperativeMatrixStoreKHR}) {
      auto missing_alignment = module("Aligned 16", "Aligned 16");
      bool found = false;
      for (size_t at = 5; at < missing_alignment.size();) {
         unsigned count = missing_alignment[at] >> 16;
         if ((missing_alignment[at] & 0xffffu) == opcode) {
            missing_alignment.erase(missing_alignment.begin() + at + count - 1);
            missing_alignment[at] -= 1u << 16;
            found = true;
            break;
         }
         at += count;
      }
      ASSERT_TRUE(found);
      rejected(missing_alignment);
   }
}
