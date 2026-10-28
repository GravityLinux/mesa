/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include "hk_apple9_softfp64.h"
#include "asahi/compiler/agx_compile_apple9.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_constant_expressions.h"

/* hk_shader.h also includes C-only command-packing helpers. */
struct vk_physical_device;
extern "C" void hk_preprocess_nir_internal(struct vk_physical_device *,
                                          nir_shader *);

namespace {

struct double_case {
   nir_op op;
   unsigned input_bits;
   unsigned components = 1;
};

class Apple9MesaSoftFP64 : public ::testing::TestWithParam<double_case> {
protected:
   static hk_apple9_softfp64 cache;

   static void SetUpTestSuite()
   {
      glsl_type_singleton_init_or_ref();
      hk_apple9_softfp64_init(&cache);
   }

   static void TearDownTestSuite()
   {
      hk_apple9_softfp64_finish(&cache);
      glsl_type_singleton_decref();
   }
};

hk_apple9_softfp64 Apple9MesaSoftFP64::cache;

static nir_def *
input(nir_builder *b, unsigned binding, unsigned bits, unsigned components)
{
   return nir_load_ubo(b, components, bits, nir_imm_int(b, binding),
                      nir_imm_int(b, 0), .align_mul = bits / 8,
                      .range = components * bits / 8);
}

static nir_shader *
shader(double_case test)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "Mesa double %s",
      nir_op_infos[test.op].name);
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *a = input(&b, 0, test.input_bits, test.components);
   nir_def *c = input(&b, 1, test.input_bits, test.components);
   nir_def *d = input(&b, 2, test.input_bits, test.components);
   nir_def *result = nir_build_alu(
      &b, test.op, a, nir_op_infos[test.op].num_inputs > 1 ? c : nullptr,
      nir_op_infos[test.op].num_inputs > 2 ? d : nullptr, nullptr);
   EXPECT_EQ(nir_def_as_alu(result)->op, test.op);
   if (result->bit_size == 1)
      result = nir_b2i32(&b, result);
   nir_store_ssbo(&b, result, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .write_mask = BITFIELD_MASK(result->num_components),
                  .align_mul = unsigned(result->bit_size) / 8);
   nir_validate_shader(b.shader, "nonconstant raw Mesa double test");
   return b.shader;
}

TEST_P(Apple9MesaSoftFP64, LibraryLoweredNativeCompile)
{
   nir_shader *nir = shader(GetParam());
   ASSERT_TRUE(hk_apple9_nir_uses_double(nir));
   const char *reason = nullptr;
   bool lowered = hk_apple9_lower_softfp64(&cache, nir, &reason);
   EXPECT_TRUE(lowered) << (reason ? reason : "no diagnostic");
   EXPECT_EQ(reason, nullptr);
   if (!lowered) {
      ralloc_free(nir);
      return;
   }
   ASSERT_NE(cache.library, nullptr);
   nir_shader *library = cache.library;
   EXPECT_TRUE(hk_apple9_lower_softfp64(&cache, nir, &reason));
   EXPECT_EQ(cache.library, library);
   EXPECT_FALSE(hk_apple9_nir_uses_double(nir));
   /* Native Vulkan defers generic preprocessing until after this fallible
    * lowering, so unsupported fused operations cannot be split first. */
   hk_preprocess_nir_internal(nullptr, nir);
   nir_validate_shader(nir, "real library output before native compile");

   unsigned loads = 0, stores = 0, alu_count = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               ++alu_count;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               const nir_op_info *info = &nir_op_infos[alu->op];
               EXPECT_FALSE(alu->def.bit_size == 64 &&
                  nir_alu_type_get_base_type(info->output_type) == nir_type_float);
               for (unsigned i = 0; i < info->num_inputs; ++i)
                  EXPECT_FALSE(alu->src[i].src.ssa->bit_size == 64 &&
                     nir_alu_type_get_base_type(info->input_types[i]) ==
                        nir_type_float);
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               loads += op == nir_intrinsic_load_ubo;
               stores += op == nir_intrinsic_store_ssbo;
            }
         }
      }
   }
   EXPECT_GT(loads, 0u);
   EXPECT_GT(stores, 0u);
   EXPECT_GT(alu_count, 0u);
   agx_shader_part compiled = {};
   agx_apple9_compute_profile profile = {};
   bool ok = agx_compile_apple9_tiny(nir, &compiled, &profile, &reason);
   EXPECT_TRUE(ok) << (reason ? reason : "no diagnostic");
   EXPECT_EQ(reason, nullptr);
   EXPECT_NE(compiled.binary, nullptr);
   EXPECT_GT(compiled.info.binary_size, 0u);
   free(compiled.binary);
   ralloc_free(nir);
}

INSTANTIATE_TEST_SUITE_P(
   ActualLibrary, Apple9MesaSoftFP64,
   ::testing::Values(
      double_case{nir_op_fadd, 64},
      double_case{nir_op_fadd, 64, 2},
      double_case{nir_op_fmul, 64},
      double_case{nir_op_fdiv, 64},
      double_case{nir_op_fdiv, 64, 2},
      double_case{nir_op_ffma, 64},
      double_case{nir_op_ffma, 64, 2},
      double_case{nir_op_fsqrt, 64},
      double_case{nir_op_f2f64, 32},
      double_case{nir_op_f2f32, 64},
      double_case{nir_op_i2f64, 32},
      double_case{nir_op_i2f64, 64},
      double_case{nir_op_u2f64, 32},
      double_case{nir_op_u2f64, 64},
      double_case{nir_op_f2i32, 64},
      double_case{nir_op_f2u32, 64},
      double_case{nir_op_f2i64, 64},
      double_case{nir_op_f2u64, 64},
      double_case{nir_op_fabs, 64},
      double_case{nir_op_fneg, 64},
      double_case{nir_op_fround_even, 64},
      double_case{nir_op_ftrunc, 64},
      double_case{nir_op_ffloor, 64},
      double_case{nir_op_ffract, 64},
      double_case{nir_op_fsign, 64},
      double_case{nir_op_feq, 64},
      double_case{nir_op_fneu, 64},
      double_case{nir_op_flt, 64},
      double_case{nir_op_fge, 64},
      double_case{nir_op_fmin, 64},
      double_case{nir_op_fmax, 64},
      double_case{nir_op_fsat, 64},
      double_case{nir_op_fisfinite, 64}),
   [](const ::testing::TestParamInfo<double_case> &info) {
      return std::string(nir_op_infos[info.param.op].name) + "_" +
             std::to_string(info.param.input_bits) + "_" +
             std::to_string(info.param.components);
   });

TEST(Apple9MesaSoftFP64Cache, NoDoubleDoesNotLoad)
{
   glsl_type_singleton_init_or_ref();
   hk_apple9_softfp64 cache;
   hk_apple9_softfp64_init(&cache);
   nir_shader *nir = shader({nir_op_fadd, 32});
   const char *reason = nullptr;
   EXPECT_TRUE(hk_apple9_lower_softfp64(&cache, nir, &reason));
   EXPECT_EQ(reason, nullptr);
   EXPECT_EQ(cache.library, nullptr);
   ralloc_free(nir);
   hk_apple9_softfp64_finish(&cache);
   glsl_type_singleton_decref();
}

TEST(Apple9MesaSoftFP64Cache, MissingExportFailsWithoutInlining)
{
   glsl_type_singleton_init_or_ref();
   hk_apple9_softfp64 cache;
   hk_apple9_softfp64_init(&cache);
   nir_builder empty = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "missing library exports");
   cache.library = empty.shader;
   nir_shader *nir = shader({nir_op_fadd, 64});
   const char *reason = nullptr;
   EXPECT_FALSE(hk_apple9_lower_softfp64(&cache, nir, &reason));
   EXPECT_STREQ(reason, "missing Mesa float64 library export");
   nir_validate_shader(nir, "missing export rejection");
   ralloc_free(nir);
   hk_apple9_softfp64_finish(&cache);
   glsl_type_singleton_decref();
}

class Apple9MesaSoftFP64Unavailable :
   public ::testing::TestWithParam<nir_op> {};

TEST_P(Apple9MesaSoftFP64Unavailable, RejectsBeforeLibraryLoad)
{
   glsl_type_singleton_init_or_ref();
   hk_apple9_softfp64 cache;
   hk_apple9_softfp64_init(&cache);
   nir_shader *nir = shader({GetParam(), 64});
   const char *reason = nullptr;
   EXPECT_FALSE(hk_apple9_lower_softfp64(&cache, nir, &reason));
   EXPECT_STREQ(reason, nir_op_infos[GetParam()].name);
   EXPECT_EQ(cache.library, nullptr);
   nir_validate_shader(nir, "unsupported raw double rejection");
   ralloc_free(nir);
   hk_apple9_softfp64_finish(&cache);
   glsl_type_singleton_decref();
}

INSTANTIATE_TEST_SUITE_P(
   UnsupportedLibraryOps, Apple9MesaSoftFP64Unavailable,
   ::testing::Values(nir_op_frcp),
   [](const ::testing::TestParamInfo<nir_op> &info) {
      return std::string(nir_op_infos[info.param].name);
   });

/* Execute the imported production library's NIR, not a second implementation
 * of arithmetic. Follow actual CFG edges and use Mesa's opcode evaluator,
 * including lower-width float ALU introduced by integer-division lowering.
 * PHIs are simultaneous, including on loop backedges.
 */
class Apple9MesaDivision : public Apple9MesaSoftFP64 {
protected:
   nir_shader *nir = nullptr;
   unsigned comparisons = 0;
   nir_op operation = nir_op_fdiv;

   void SetUp() override
   {
      prepare(nir_op_fdiv, "__fdiv64(u641;u641;");
   }

   void prepare(nir_op op, const char *export_name,
                nir_shader *program = nullptr)
   {
      ASSERT_TRUE(std::numeric_limits<double>::is_iec559);
      ASSERT_EQ(std::fegetround(), FE_TONEAREST);
      operation = op;
      nir = program ? program : shader({op, 64});
      const char *reason = nullptr;
      ASSERT_TRUE(hk_apple9_lower_softfp64(&cache, nir, &reason))
         << (reason ? reason : "no diagnostic");
      ASSERT_NE(nir_shader_get_function_for_name(
         cache.library, export_name), nullptr);
      ASSERT_FALSE(hk_apple9_nir_uses_double(nir));
      hk_preprocess_nir_internal(nullptr, nir);
      nir_validate_shader(nir, "production arithmetic before CPU execution");
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }

   void TearDown() override
   {
      RecordProperty(operation == nir_op_ffma ? "fma_comparisons" :
                     operation == nir_op_fdiv ? "division_comparisons" :
                                               "raw_probe_comparisons",
                     comparisons);
      ralloc_free(nir);
   }

   void evaluate(const std::array<uint64_t, 3> &inputs, uint64_t *result)
   {
      using value = std::array<nir_const_value, NIR_MAX_VEC_COMPONENTS>;
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      std::vector<value> values(impl->ssa_alloc);
      std::vector<bool> defined(impl->ssa_alloc, false);
      nir_block *previous = nullptr;
      nir_block *block = nir_start_block(impl);
      unsigned blocks = 0, stores = 0;
      while (block != impl->end_block) {
         ASSERT_NE(block, nullptr);
         ASSERT_LT(++blocks, 10000u) << "library control flow did not terminate";
         struct phi_value {
            unsigned index;
            value data;
            bool defined;
         };
         std::vector<phi_value> phis;
         nir_foreach_phi(phi, block) {
            bool found = false;
            nir_foreach_phi_src(src, phi) {
               if (src->pred == previous) {
                  unsigned source = src->src.ssa->index;
                  phis.push_back({phi->def.index, values[source],
                                  defined[source]});
                  found = true;
                  break;
               }
            }
            ASSERT_TRUE(found);
         }
         for (const auto &phi : phis) {
            values[phi.index] = phi.data;
            defined[phi.index] = phi.defined;
         }
         nir_foreach_instr(instr, block) {
            switch (instr->type) {
            case nir_instr_type_phi:
               break;
            case nir_instr_type_load_const: {
               auto *load = nir_instr_as_load_const(instr);
               for (unsigned c = 0; c < load->def.num_components; ++c)
                  values[load->def.index][c] = load->value[c];
               defined[load->def.index] = true;
               break;
            }
            case nir_instr_type_undef:
               defined[nir_instr_as_undef(instr)->def.index] = false;
               break;
            case nir_instr_type_alu: {
               auto *alu = nir_instr_as_alu(instr);
               const nir_op_info &info = nir_op_infos[alu->op];
               ASSERT_FALSE(alu->def.bit_size == 64 &&
                  nir_alu_type_get_base_type(info.output_type) == nir_type_float)
                  << info.name;
               unsigned bits = nir_alu_type_get_type_size(info.output_type) ?
                                  0 : alu->def.bit_size;
               value sources[NIR_ALU_MAX_INPUTS] = {};
               nir_const_value *ptrs[NIR_ALU_MAX_INPUTS] = {};
               for (unsigned i = 0; i < info.num_inputs; ++i) {
                  ASSERT_FALSE(alu->src[i].src.ssa->bit_size == 64 &&
                     nir_alu_type_get_base_type(info.input_types[i]) ==
                        nir_type_float) << info.name;
                  ASSERT_TRUE(defined[alu->src[i].src.ssa->index]) << info.name;
                  if (!bits && !nir_alu_type_get_type_size(info.input_types[i]))
                     bits = alu->src[i].src.ssa->bit_size;
                  ptrs[i] = sources[i].data();
                  unsigned count = info.input_sizes[i] ?
                                      info.input_sizes[i] : alu->def.num_components;
                  for (unsigned c = 0; c < count; ++c)
                     sources[i][c] =
                        values[alu->src[i].src.ssa->index][alu->src[i].swizzle[c]];
               }
               uint16_t poison = 0;
               nir_eval_const_opcode(alu->op, values[alu->def.index].data(),
                                     &poison, alu->def.num_components,
                                     bits ? bits : 32, ptrs,
                                     nir->info.float_controls_execution_mode);
               ASSERT_EQ(poison, 0u) << info.name;
               defined[alu->def.index] = true;
               break;
            }
            case nir_instr_type_intrinsic: {
               auto *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_ubo) {
                  ASSERT_TRUE(nir_src_is_const(intr->src[0]));
                  ASSERT_EQ(nir_src_as_uint(intr->src[1]), 0u);
                  unsigned binding = nir_src_as_uint(intr->src[0]);
                  ASSERT_LT(binding, nir_op_infos[operation].num_inputs);
                  ASSERT_EQ(intr->def.bit_size, 64u);
                  ASSERT_EQ(intr->def.num_components, 1u);
                  values[intr->def.index][0].u64 = inputs[binding];
                  defined[intr->def.index] = true;
               } else if (intr->intrinsic == nir_intrinsic_store_ssbo) {
                  ASSERT_EQ(nir_src_as_uint(intr->src[1]), 0u);
                  ASSERT_EQ(nir_src_as_uint(intr->src[2]), 0u);
                  ASSERT_EQ(intr->num_components, 1u);
                  ASSERT_EQ(intr->src[0].ssa->bit_size, 64u);
                  ASSERT_EQ(nir_intrinsic_write_mask(intr), 1u);
                  ASSERT_TRUE(defined[intr->src[0].ssa->index]);
                  *result = values[intr->src[0].ssa->index][0].u64;
                  ++stores;
               } else {
                  FAIL() << "uninterpreted intrinsic "
                         << nir_intrinsic_infos[intr->intrinsic].name;
               }
               break;
            }
            case nir_instr_type_jump:
               ASSERT_TRUE(nir_instr_as_jump(instr)->type == nir_jump_break ||
                           nir_instr_as_jump(instr)->type == nir_jump_continue);
               break;
            default:
               FAIL() << "uninterpreted instruction " << instr->type;
            }
         }
         unsigned edge = 0;
         if (block->successors[1]) {
            nir_if *nif = nir_block_get_following_if(block);
            ASSERT_NE(nif, nullptr);
            ASSERT_TRUE(defined[nif->condition.ssa->index]);
            edge = !nir_const_value_as_bool(
               values[nif->condition.ssa->index][0],
               nif->condition.ssa->bit_size);
         }
         previous = block;
         block = block->successors[edge];
      }
      ASSERT_EQ(stores, 1u);
   }

   void compare(uint64_t a, uint64_t b)
   {
      SCOPED_TRACE(::testing::Message() << std::hex << a << " / " << b);
      double da, db;
      memcpy(&da, &a, sizeof(da));
      memcpy(&db, &b, sizeof(db));
      /* Volatile operands force the independent host IEEE divide at runtime. */
      volatile double va = da, vb = db;
      double expected = va / vb;
      uint64_t want, got = 0;
      memcpy(&want, &expected, sizeof(want));
      ASSERT_NO_FATAL_FAILURE(evaluate({a, b, 0}, &got));
      if ((want & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {
         EXPECT_GT(got & 0x7fffffffffffffffULL, 0x7ff0000000000000ULL);
         EXPECT_NE(got & 0x0008000000000000ULL, 0u);
      } else {
         EXPECT_EQ(got, want);
      }
      ++comparisons;
   }
};

TEST_F(Apple9MesaDivision, SpecialValuesAndBoundaries)
{
   const uint64_t magnitudes[] = {
      0, 1, 2, 3, 0x00000000ffffffffULL, 0x0000000100000000ULL,
      0x000ffffffffffffeULL, 0x000fffffffffffffULL, 0x0010000000000000ULL,
      0x0010000000000001ULL, 0x001fffffffffffffULL, 0x0020000000000000ULL,
      0x3fdfffffffffffffULL, 0x3fe0000000000000ULL, 0x3fefffffffffffffULL,
      0x3ff0000000000000ULL, 0x3ff0000000000001ULL, 0x3ff8000000000000ULL,
      0x3fffffffffffffffULL, 0x4000000000000000ULL, 0x4008000000000000ULL,
      0x7fdfffffffffffffULL, 0x7fe0000000000000ULL, 0x7fefffffffffffffULL,
      0x7ff0000000000000ULL, 0x7ff8000000000000ULL, 0x7ff0000000000001ULL,
      0x7fffffffffffffffULL,
   };
   for (uint64_t a : magnitudes)
      for (uint64_t b : magnitudes)
         for (unsigned signs = 0; signs < 4; ++signs)
            ASSERT_NO_FATAL_FAILURE(compare(a | (uint64_t(signs & 1) << 63),
                                            b | (uint64_t(signs >> 1) << 63)));
}

TEST_F(Apple9MesaDivision, SubnormalTiesToEven)
{
   /* All subnormal precisions, both retained parities, both signs. Division
    * by two makes exact half-way results; adjacent denominators straddle them.
    * The largest numerator also rounds across the minimum-normal boundary.
    */
   for (unsigned bit = 0; bit <= 52; ++bit) {
      for (uint64_t a : {(uint64_t(1) << bit) | 1,
                         (uint64_t(1) << bit) | 3,
                         uint64_t(0x001fffffffffffffULL)}) {
         for (uint64_t b : {0x3fffffffffffffffULL, 0x4000000000000000ULL,
                            0x4000000000000001ULL})
            for (unsigned signs = 0; signs < 4; ++signs)
               ASSERT_NO_FATAL_FAILURE(compare(a | (uint64_t(signs & 1) << 63),
                                               b | (uint64_t(signs >> 1) << 63)));
      }
   }
}

TEST_F(Apple9MesaDivision, DeterministicIEEEReference)
{
   uint64_t state = 0x243f6a8885a308d3ULL;
   auto next = [&]() {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      return state;
   };
   for (unsigned i = 0; i < 4096; ++i) {
      uint64_t a = next(), b = next();
      ASSERT_NO_FATAL_FAILURE(compare(a, b));
      /* Independently force subnormal operands and underflow-scale quotients. */
      ASSERT_NO_FATAL_FAILURE(compare(a & 0x800fffffffffffffULL, b));
      ASSERT_NO_FATAL_FAILURE(compare(a, b & 0x800fffffffffffffULL));
      ASSERT_NO_FATAL_FAILURE(compare((a & 0x800fffffffffffffULL) |
                                        0x0010000000000000ULL,
                                     (b & 0x800fffffffffffffULL) |
                                        0x4000000000000000ULL));
   }
}

class Apple9MesaFMA : public Apple9MesaDivision {
protected:
   void SetUp() override
   {
      prepare(nir_op_ffma, "__ffma64(u641;u641;u641;");
   }

   static double from_bits(uint64_t bits)
   {
      double value;
      memcpy(&value, &bits, sizeof(value));
      return value;
   }

   static uint64_t to_bits(double value)
   {
      uint64_t bits;
      memcpy(&bits, &value, sizeof(bits));
      return bits;
   }

   void compare(uint64_t a, uint64_t b, uint64_t c)
   {
      SCOPED_TRACE(::testing::Message() << std::hex << a << " * " << b
                                      << " + " << c);
      volatile double va = from_bits(a), vb = from_bits(b), vc = from_bits(c);
      uint64_t want = to_bits(std::fma(va, vb, vc)), got = 0;
      ASSERT_NO_FATAL_FAILURE(evaluate({a, b, c}, &got));
      if ((want & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {
         EXPECT_GT(got & 0x7fffffffffffffffULL, 0x7ff0000000000000ULL);
         EXPECT_NE(got & 0x0008000000000000ULL, 0u);
      } else {
         EXPECT_EQ(got, want);
      }
      ++comparisons;
   }
};

TEST_F(Apple9MesaFMA, SpecialValuesAndSigns)
{
   const uint64_t magnitudes[] = {
      0, 1, 0x000fffffffffffffULL, 0x0010000000000000ULL,
      0x3fefffffffffffffULL, 0x3ff0000000000000ULL, 0x3ff0000000000001ULL,
      0x4000000000000000ULL, 0x7fefffffffffffffULL,
      0x7ff0000000000000ULL, 0x7ff8000000000000ULL, 0x7ff0000000000001ULL,
   };
   for (uint64_t a : magnitudes)
      for (uint64_t b : magnitudes)
         for (uint64_t c : magnitudes)
            for (unsigned signs = 0; signs < 8; ++signs)
               ASSERT_NO_FATAL_FAILURE(compare(
                  a | (uint64_t(signs & 1) << 63),
                  b | (uint64_t((signs >> 1) & 1) << 63),
                  c | (uint64_t(signs >> 2) << 63)));
}

TEST_F(Apple9MesaFMA, FusedNotSplit)
{
   const struct {
      uint64_t a, b, c, result;
   } cases[] = {
      /* (1+2^-52)*(1-2^-52)-1 = -2^-104, not zero. */
      {0x3ff0000000000001ULL, 0x3feffffffffffffeULL,
       0xbff0000000000000ULL, 0xb970000000000000ULL},
      /* The same cancellation at the least subnormal and its half-way tie. */
      {0x21a0000000000001ULL, 0x219ffffffffffffeULL,
       0x8350000000000000ULL, 0x8000000000000001ULL},
      {0x2190000000000001ULL, 0x219ffffffffffffeULL,
       0x8340000000000000ULL, 0x8000000000000000ULL},
      /* A residual below the subnormal range retains its negative zero sign. */
      {0x2000000000000001ULL, 0x1ffffffffffffffeULL,
       0x8010000000000000ULL, 0x8000000000000000ULL},
      /* An overflowing intermediate product must not lose cancellation. */
      {0x7fefffffffffffffULL, 0x4000000000000000ULL,
       0xffefffffffffffffULL, 0x7fefffffffffffffULL},
      {0x7fefffffffffffffULL, 0x3ff0000000000001ULL,
       0xffefffffffffffffULL, 0x7cafffffffffffffULL},
   };
   for (const auto &test : cases) {
      volatile double a = from_bits(test.a), b = from_bits(test.b);
      volatile double c = from_bits(test.c);
      volatile double product = a * b;
      ASSERT_EQ(to_bits(std::fma(a, b, c)), test.result);
      ASSERT_NE(to_bits(product + c), test.result);
      ASSERT_NO_FATAL_FAILURE(compare(test.a, test.b, test.c));
   }
}

TEST_F(Apple9MesaFMA, AlignmentAndRoundingBoundaries)
{
   /* Exact and jammed alignment, on either side of every limb boundary.
    * The multiplication retains nonzero bits through the full 106-bit product.
    */
   for (int gap : {0, 1, 20, 21, 22, 23, 31, 32, 33, 63, 64, 65,
                  73, 74, 75, 95, 96, 97, 126, 127, 128, 129}) {
      for (int direction : {-1, 1}) {
         uint64_t power = uint64_t(1023 + direction * gap) << 52;
         for (int offset : {-1, 0, 1}) {
            for (unsigned signs = 0; signs < 8; ++signs)
               ASSERT_NO_FATAL_FAILURE(compare(
                  0x3ff0000000000001ULL | (uint64_t(signs & 1) << 63),
                  0x3ff0000000000001ULL | (uint64_t((signs >> 1) & 1) << 63),
                  (power + offset) | (uint64_t(signs >> 2) << 63)));
         }
      }
   }
   /* Half-ULP addends, above/below/exactly at the tie, and both retained
    * parities. The smallest half-ULP here is itself a subnormal.
    */
   for (int exponent : {-1021, -970, -53, -1, 0, 1, 53, 970, 1023}) {
      for (unsigned parity = 0; parity < 2; ++parity) {
         uint64_t a = (uint64_t(exponent + 1023) << 52) | parity;
         uint64_t half = to_bits(std::ldexp(1.0, exponent - 53));
         for (int offset : {-1, 0, 1})
            for (unsigned signs = 0; signs < 4; ++signs)
               ASSERT_NO_FATAL_FAILURE(compare(
                  a | (uint64_t(signs & 1) << 63), 0x3ff0000000000000ULL,
                  (half + offset) | (uint64_t(signs >> 1) << 63)));
      }
   }
   /* Round the full product plus addend once in subnormal units. */
   for (uint64_t a : {1ULL, 3ULL, 5ULL, 0x000fffffffffffffULL,
                      0x0010000000000001ULL}) {
      for (uint64_t c : {0ULL, 1ULL, 2ULL, 0x0010000000000000ULL})
         for (unsigned signs = 0; signs < 4; ++signs)
            ASSERT_NO_FATAL_FAILURE(compare(
               a | (uint64_t(signs & 1) << 63), 0x3fe0000000000000ULL,
               c | (uint64_t(signs >> 1) << 63)));
   }
}

TEST_F(Apple9MesaFMA, DeterministicRawAndCancellation)
{
   uint64_t state = 0x13198a2e03707344ULL;
   auto next = [&]() {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      return state;
   };
   for (unsigned i = 0; i < 4096; ++i) {
      uint64_t a = next(), b = next(), c = next();
      ASSERT_NO_FATAL_FAILURE(compare(a, b, c));
      for (uint64_t exponent : {0x3ff0000000000000ULL,
                                0x2000000000000000ULL}) {
         uint64_t aa = (a & 0x800fffffffffffffULL) | exponent;
         uint64_t bb = (b & 0x800fffffffffffffULL) | exponent;
         volatile double va = from_bits(aa), vb = from_bits(bb);
         uint64_t cancel = to_bits(va * vb) ^ 0x8000000000000000ULL;
         ASSERT_NO_FATAL_FAILURE(compare(aa, bb, cancel));
         ASSERT_NO_FATAL_FAILURE(compare(aa, bb, cancel + 1));
      }
   }
}

TEST(Apple9MesaSoftFP64Cache, MissingFusedExportFailsWithoutInlining)
{
   glsl_type_singleton_init_or_ref();
   hk_apple9_softfp64 cache;
   hk_apple9_softfp64_init(&cache);
   nir_builder empty = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "missing fused library export");
   cache.library = empty.shader;
   nir_shader *nir = shader({nir_op_ffma, 64});
   const char *reason = nullptr;
   EXPECT_FALSE(hk_apple9_lower_softfp64(&cache, nir, &reason));
   EXPECT_STREQ(reason, "missing Mesa float64 library export");
   EXPECT_TRUE(hk_apple9_nir_uses_double(nir));
   nir_validate_shader(nir, "missing fused export rejection");
   ralloc_free(nir);
   hk_apple9_softfp64_finish(&cache);
   glsl_type_singleton_decref();
}

class Apple9MesaRawArithmetic : public Apple9MesaDivision {
protected:
   void SetUp() override
   {
      nir_op op = GetParam().op;
      prepare(op, op == nir_op_fadd ? "__fadd64(u641;u641;" :
                  op == nir_op_fmul ? "__fmul64(u641;u641;" :
                                     "__fsqrt64(u641;");
   }

   void compare(uint64_t a, uint64_t b)
   {
      SCOPED_TRACE(::testing::Message() << nir_op_infos[operation].name
                                      << " " << std::hex << a << " " << b);
      if (operation == nir_op_fsqrt)
         a &= 0x7fffffffffffffffULL;
      double da, db;
      memcpy(&da, &a, sizeof(da));
      memcpy(&db, &b, sizeof(db));
      volatile double va = da, vb = db;
      double expected = operation == nir_op_fadd ? va + vb :
                        operation == nir_op_fmul ? va * vb : std::sqrt(va);
      uint64_t want, got = 0;
      memcpy(&want, &expected, sizeof(want));
      ASSERT_NO_FATAL_FAILURE(evaluate({a, b, 0}, &got));
      if ((want & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {
         EXPECT_GT(got & 0x7fffffffffffffffULL, 0x7ff0000000000000ULL);
         EXPECT_NE(got & 0x0008000000000000ULL, 0u);
      } else {
         EXPECT_EQ(got, want);
      }
      ++comparisons;
   }
};

TEST_P(Apple9MesaRawArithmetic, RoundingBoundariesAndRawInputs)
{
   const uint64_t pairs[][2] = {
      {0x3ff0000000000000ULL, 0x3ca0000000000000ULL},
      {0x3ff0000000000001ULL, 0x3ca0000000000000ULL},
      {0x3ff0000000000000ULL, 0x3ca0000000000001ULL},
      {0x3ff0000000000000ULL, 0x3c9fffffffffffffULL},
      {0x0010000000000001ULL, 0x3fe0000000000000ULL},
      {0x0010000000000003ULL, 0x3fe0000000000000ULL},
      {0x0010000000000001ULL, 0x3fe0000000000001ULL},
      {0x000fffffffffffffULL, 0x3ff0000000000001ULL},
      {1, 0x3fe0000000000000ULL},
      {3, 0x3fe0000000000000ULL},
      {0x7fefffffffffffffULL, 0x3fefffffffffffffULL},
      {0x7ff0000000000000ULL, 0},
   };
   for (const auto &pair : pairs)
      for (unsigned signs = 0; signs < 4; ++signs)
         ASSERT_NO_FATAL_FAILURE(compare(
            pair[0] | (uint64_t(signs & 1) << 63),
            pair[1] | (uint64_t(signs >> 1) << 63)));
   uint64_t state = 0x13198a2e03707344ULL;
   for (unsigned i = 0; i < 4096; ++i) {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      uint64_t a = state;
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      ASSERT_NO_FATAL_FAILURE(compare(a, state));
   }
}

INSTANTIATE_TEST_SUITE_P(
   RawProbe, Apple9MesaRawArithmetic,
   ::testing::Values(double_case{nir_op_fadd, 64},
                     double_case{nir_op_fmul, 64},
                     double_case{nir_op_fsqrt, 64}),
   [](const auto &info) { return std::string(nir_op_infos[info.param.op].name); });

class Apple9MesaConversion : public Apple9MesaDivision {
protected:
   void SetUp() override
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "raw double conversion column");
      b.shader->info.workgroup_size[0] = 16;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_def *v = input(&b, 0, 64, 1);
      nir_def *result = nir_fadd(
         &b, nir_i2f64(&b, nir_f2i32(&b, v)),
         nir_f2f64(&b, nir_f2f32(&b, v)));
      nir_store_ssbo(&b, result, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                     .write_mask = 1, .align_mul = 8);
      prepare(nir_op_fadd, "__fadd64(u641;u641;", b.shader);
   }
};

TEST_F(Apple9MesaConversion, RawProbeFloatAndIntegerRoundTrip)
{
   for (unsigned i = 0; i < 4096; ++i) {
      volatile double v = (i & 1 ? -1.0 : 1.0) *
                             (8388608.0 + double(i / 2) + 0.5);
      volatile float f = float(v);
      double expected = double(int32_t(v)) + double(f);
      double value = v;
      uint64_t bits, want, got = 0;
      memcpy(&bits, &value, sizeof(bits));
      memcpy(&want, &expected, sizeof(want));
      ASSERT_NO_FATAL_FAILURE(evaluate({bits, 0, 0}, &got));
      EXPECT_EQ(got, want) << "input " << i;
      ++comparisons;
   }
}

} // namespace
