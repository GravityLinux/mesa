/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

extern "C" {
#include "agx_nir.h"
}
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_constant_expressions.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/half_float.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

/* A standalone CPU test. No device, driver, Vulkan instance, or GPU simulator.
 * The interpreter executes the actual lowered NIR, including cross-lane reads;
 * the oracle is an independent ordinary row/column matrix multiplication.
 */
#define CHECK(x) do { \
   if (!(x)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      exit(1); \
   } \
} while (0)

static const nir_shader_compiler_options options = {};
using Lane = std::array<nir_const_value, NIR_MAX_VEC_COMPONENTS>;
using Wave = std::array<Lane, 32>;

struct shader {
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &options, "apple9_cooperative_matrix_test");

   shader()
   {
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.cs.has_cooperative_matrix = true;
   }
   ~shader() { ralloc_free(b.shader); }

   nir_deref_instr *matrix(glsl_base_type type, glsl_cmat_use use,
                          unsigned rows = 8, mesa_scope scope = SCOPE_SUBGROUP)
   {
      glsl_cmat_description d = {};
      d.rows = rows;
      d.cols = 8;
      d.element_type = type;
      d.scope = scope;
      d.use = use;
      return nir_build_deref_var(&b, nir_local_variable_create(
         b.impl, glsl_cmat_type(&d), "matrix"));
   }

   nir_def *input(unsigned base, unsigned bits = 32)
   {
      return nir_load_input(&b, 2, bits, nir_imm_int(&b, 0), .base = int(base));
   }

   void output(nir_def *value, unsigned base = 0)
   {
      nir_store_output(&b, value, nir_imm_int(&b, 0),
                       .base = int(base), .write_mask = BITFIELD_MASK(value->num_components));
   }

   void lower()
   {
      nir_validate_shader(b.shader, "before cooperative matrix lowering");
      const char *reason = nullptr;
      CHECK(agx_nir_lower_apple9_cooperative_matrix(b.shader, 32, &reason));
      CHECK(!reason);
      CHECK(!b.shader->info.cs.has_cooperative_matrix);
      nir_validate_shader(b.shader, "after cooperative matrix lowering");
   }
};

static bool
matrix_type(const glsl_type *type)
{
   return glsl_type_is_cmat(glsl_without_array(type));
}

static void
check_lowered(nir_shader *s)
{
   nir_foreach_variable_in_shader(var, s)
      CHECK(!matrix_type(var->type));
   nir_foreach_function_impl(impl, s) {
      nir_foreach_function_temp_variable(var, impl)
         CHECK(!matrix_type(var->type));
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            CHECK(instr->type != nir_instr_type_cmat_call);
            if (instr->type == nir_instr_type_deref)
               CHECK(!matrix_type(nir_instr_as_deref(instr)->type));
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            auto *intr = nir_instr_as_intrinsic(instr);
            CHECK(strncmp(nir_intrinsic_infos[intr->intrinsic].name, "cmat_", 5));
            if (intr->intrinsic == nir_intrinsic_read_invocation) {
               CHECK(intr->def.bit_size == 32);
               CHECK(nir_src_is_const(intr->src[1]));
               CHECK(nir_src_as_uint(intr->src[1]) < 32);
            }
         }
      }
   }
}

struct interpreter {
   std::unordered_map<const nir_def *, Wave> values;
   std::array<Wave, 8> inputs = {}, outputs = {};
   std::vector<uint8_t> memory = std::vector<uint8_t>(16384, 0xa5);
   std::vector<unsigned> writes = std::vector<unsigned>(16384, 0);
   unsigned broadcasts = 0;

   Lane &get(nir_src src, unsigned lane)
   {
      CHECK(values.count(src.ssa));
      return values.at(src.ssa)[lane];
   }

   void run(shader &s)
   {
      s.lower();
      check_lowered(s.b.shader);
      nir_lower_var_copies(s.b.shader);
      nir_lower_array_deref_of_vec(
         s.b.shader, nir_var_function_temp, nullptr,
         nir_lower_array_deref_of_vec_options(
            nir_lower_indirect_array_deref_of_vec_load |
            nir_lower_indirect_array_deref_of_vec_store));
      nir_lower_vars_to_ssa(s.b.shader);
      nir_lower_explicit_io(s.b.shader, nir_var_mem_global,
                            nir_address_format_64bit_global);
      bool progress;
      do {
         progress = nir_opt_copy_prop(s.b.shader);
         progress |= nir_opt_constant_folding(s.b.shader);
         progress |= nir_opt_dce(s.b.shader);
      } while (progress);
      nir_validate_shader(s.b.shader, "interpreter input");

      nir_foreach_block(block, s.b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_load_const) {
               auto *load = nir_instr_as_load_const(instr);
               Wave &v = values[&load->def];
               for (unsigned lane = 0; lane < 32; ++lane)
                  for (unsigned c = 0; c < load->def.num_components; ++c)
                     v[lane][c] = load->value[c];
            } else if (instr->type == nir_instr_type_undef) {
               values[&nir_instr_as_undef(instr)->def] = {};
            } else if (instr->type == nir_instr_type_alu) {
               auto *alu = nir_instr_as_alu(instr);
               Wave &v = values[&alu->def];
               const nir_op_info &info = nir_op_infos[alu->op];
               /* Constant evaluation sizes the first UNSIZED operand, not
                * necessarily the result (f2f32 has a sized result).
                */
               unsigned bits = nir_alu_type_get_type_size(info.output_type) ?
                                  0 : alu->def.bit_size;
               for (unsigned i = 0; i < info.num_inputs && !bits; ++i) {
                  if (!nir_alu_type_get_type_size(info.input_types[i]))
                     bits = alu->src[i].src.ssa->bit_size;
               }
               if (!bits)
                  bits = 32;
               for (unsigned lane = 0; lane < 32; ++lane) {
                  Lane sources[NIR_ALU_MAX_INPUTS] = {};
                  nir_const_value *ptrs[NIR_ALU_MAX_INPUTS] = {};
                  for (unsigned i = 0; i < info.num_inputs; ++i) {
                     ptrs[i] = sources[i].data();
                     unsigned count = info.input_sizes[i] ?: alu->def.num_components;
                     for (unsigned c = 0; c < count; ++c)
                        sources[i][c] = get(alu->src[i].src, lane)[alu->src[i].swizzle[c]];
                  }
                  uint16_t poison = 0;
                  nir_eval_const_opcode(alu->op, v[lane].data(), &poison,
                                         alu->def.num_components, bits,
                                         ptrs, s.b.shader->info.float_controls_execution_mode);
                  CHECK(!poison);
               }
            } else if (instr->type == nir_instr_type_intrinsic) {
               auto *intr = nir_instr_as_intrinsic(instr);
               Wave &v = values[&intr->def];
               if (intr->intrinsic == nir_intrinsic_read_invocation)
                  ++broadcasts;
               for (unsigned lane = 0; lane < 32; ++lane) {
                  switch (intr->intrinsic) {
                  case nir_intrinsic_load_subgroup_invocation:
                     v[lane][0].u32 = lane;
                     break;
                  case nir_intrinsic_load_input:
                     v[lane] = inputs[nir_intrinsic_base(intr)][lane];
                     break;
                  case nir_intrinsic_store_output:
                     for (unsigned c = 0; c < intr->num_components; ++c) {
                        if (nir_intrinsic_write_mask(intr) & (1u << c))
                           outputs[nir_intrinsic_base(intr)][lane][c] =
                              get(intr->src[0], lane)[c];
                     }
                     break;
                  case nir_intrinsic_read_invocation: {
                     unsigned index = get(intr->src[1], lane)[0].u32;
                     CHECK(index < 32);
                     CHECK(index == get(intr->src[1], 0)[0].u32);
                     v[lane] = get(intr->src[0], index);
                     break;
                  }
                  case nir_intrinsic_load_global:
                  case nir_intrinsic_store_global: {
                     bool load = intr->intrinsic == nir_intrinsic_load_global;
                     unsigned bytes = (load ? intr->def.bit_size :
                                       intr->src[0].ssa->bit_size) / 8;
                     uint64_t address = get(intr->src[!load], lane)[0].u64;
                     for (unsigned c = 0; c < intr->num_components; ++c) {
                        CHECK(address + bytes <= memory.size());
                        if (load) {
                           v[lane][c].u64 = 0;
                           memcpy(&v[lane][c], &memory[address], bytes);
                        } else if (nir_intrinsic_write_mask(intr) & (1u << c)) {
                           memcpy(&memory[address], &get(intr->src[0], lane)[c], bytes);
                           for (unsigned j = 0; j < bytes; ++j)
                              ++writes[address + j];
                        }
                        address += bytes;
                     }
                     break;
                  }
                  default:
                     fprintf(stderr, "Uninterpreted intrinsic: %s\n",
                             nir_intrinsic_infos[intr->intrinsic].name);
                     CHECK(false);
                  }
               }
            } else {
               fprintf(stderr, "Uninterpreted instruction type: %u\n", instr->type);
               CHECK(false);
            }
         }
      }
   }
};

/* Independent oracle mapping, transcribed from the producer's quad formulas. */
static unsigned row(unsigned lane) { return ((lane / 4) & 4) + (lane / 2) % 4; }
static unsigned col(unsigned lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }

static void
set_float(nir_const_value &dst, float value, bool half)
{
   dst = {};
   if (half)
      dst.u16 = _mesa_float_to_half(value);
   else
      dst.f32 = value;
}

static float
rounded(float value, bool half)
{
   return half ? _mesa_half_to_float(_mesa_float_to_half(value)) : value;
}

static void
test_muladd(bool half, unsigned pattern, bool alias)
{
   shader s;
   interpreter cpu;
   auto type = half ? GLSL_TYPE_FLOAT16 : GLSL_TYPE_FLOAT;
   auto *a = s.matrix(type, GLSL_CMAT_USE_A);
   auto *b = s.matrix(type, GLSL_CMAT_USE_B);
   auto *c = s.matrix(GLSL_TYPE_FLOAT, GLSL_CMAT_USE_ACCUMULATOR);
   auto *d = alias ? c : s.matrix(GLSL_TYPE_FLOAT, GLSL_CMAT_USE_ACCUMULATOR);
   auto *av = s.input(0, half ? 16 : 32);
   auto *bv = s.input(1, half ? 16 : 32);
   auto *cv = s.input(2);
   /* Exactly the insert/copy/extract representation emitted by vtn. */
   for (unsigned j = 0; j < 2; ++j) {
      auto *idx = nir_imm_int(&s.b, j);
      nir_cmat_insert(&s.b, &a->def, nir_channel(&s.b, av, j), &a->def, idx);
      nir_cmat_insert(&s.b, &b->def, nir_channel(&s.b, bv, j), &b->def, idx);
      nir_cmat_insert(&s.b, &c->def, nir_channel(&s.b, cv, j), &c->def, idx);
   }
   nir_cmat_muladd(&s.b, &d->def, &a->def, &b->def, &c->def);
   auto *x = nir_cmat_extract(&s.b, 32, &d->def, nir_imm_int(&s.b, 0));
   auto *y = nir_cmat_extract(&s.b, 32, &d->def, nir_imm_int(&s.b, 1));
   s.output(nir_vec2(&s.b, x, y));

   float A[8][8], B[8][8], C[8][8];
   for (unsigned r = 0; r < 8; ++r) {
      for (unsigned c = 0; c < 8; ++c) {
         A[r][c] = rounded(pattern == 0 ? float(r == c) :
                           float(int((r * 17 + c * 3) % 23) - 11) * 0.125f, half);
         B[r][c] = rounded(pattern == 1 ? float(r == c) :
                           float(int((r * 5 + c * 13) % 29) - 14) * 0.0625f, half);
         C[r][c] = float(2 * r + c) * 0.25f - 3.0f;
         if (pattern == 3) {
            /* FP32: a split multiply/add loses the tiny fused residual.
             * FP16: multiplying in half overflows before FP32 accumulation.
             */
            A[r][c] = r != c ? 0 : half ? 4096.0f : 1.0f + 0x1p-23f;
            B[r][c] = r != c ? 0 : half ? 4096.0f : 1.0f - 0x1p-23f;
            C[r][c] = r != c ? C[r][c] : half ? -16777215.0f : -1.0f;
         }
      }
   }
   unsigned coverage[8][8] = {};
   for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned j = 0; j < 2; ++j) {
         unsigned r = row(lane), c = col(lane) + j;
         CHECK(++coverage[r][c] == 1);
         set_float(cpu.inputs[0][lane][j], A[r][c], half);
         set_float(cpu.inputs[1][lane][j], B[r][c], half);
         set_float(cpu.inputs[2][lane][j], C[r][c], false);
      }
   }
   cpu.run(s);
   CHECK(cpu.broadcasts == 128);
   for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned j = 0; j < 2; ++j) {
         unsigned r = row(lane), c = col(lane) + j;
         float expected = C[r][c];
         for (unsigned k = 0; k < 8; ++k)
            expected = std::fma(A[r][k], B[k][c], expected);
         if (cpu.outputs[0][lane][j].f32 != expected) {
            fprintf(stderr, "half=%u pattern=%u alias=%u lane=%u slot=%u "
                            "actual=%a expected=%a\n",
                    half, pattern, alias, lane, j,
                    cpu.outputs[0][lane][j].f32, expected);
            CHECK(false);
         }
      }
   }
}

static nir_deref_instr *
memory_pointer(shader &s, unsigned address, glsl_base_type type, unsigned width)
{
   unsigned size = glsl_base_type_bit_size(type) / 8 * width;
   return nir_build_deref_cast(&s.b, nir_imm_int64(&s.b, address),
                               nir_var_mem_global, glsl_vector_type(type, width), size);
}

static void
test_memory(bool half, bool column_major, unsigned width, glsl_cmat_use use)
{
   shader s;
   interpreter cpu;
   auto type = half ? GLSL_TYPE_FLOAT16 : GLSL_TYPE_FLOAT;
   auto *mat = s.matrix(type, use);
   unsigned bytes = half ? 2 : 4;
   constexpr unsigned input_base = 256, output_base = 8192;
   constexpr unsigned stride = 13;
   auto *in = memory_pointer(s, input_base, type, width);
   auto *out = memory_pointer(s, output_base, type, width);
   auto layout = column_major ? GLSL_MATRIX_LAYOUT_COLUMN_MAJOR :
                                GLSL_MATRIX_LAYOUT_ROW_MAJOR;
   /* A runtime stride, not a folded constant. */
   auto *dynamic_stride = nir_channel(&s.b, s.input(0), 0);
   nir_cmat_load(&s.b, &mat->def, &in->def, dynamic_stride, .matrix_layout = layout);
   nir_cmat_store(&s.b, &out->def, &mat->def, dynamic_stride, .matrix_layout = layout);
   s.output(nir_vec2(&s.b,
      nir_cmat_extract(&s.b, half ? 16 : 32, &mat->def, nir_imm_int(&s.b, 0)),
      nir_cmat_extract(&s.b, half ? 16 : 32, &mat->def, nir_imm_int(&s.b, 1))));
   std::vector<bool> expected_write(cpu.memory.size(), false);
   for (unsigned r = 0; r < 8; ++r) {
      for (unsigned c = 0; c < 8; ++c) {
         unsigned major = column_major ? c : r, minor = column_major ? r : c;
         unsigned offset = (major * stride * width + minor) * bytes;
         nir_const_value value = {};
         set_float(value, float(r * 8 + c) + 0.5f, half);
         memcpy(&cpu.memory[input_base + offset], &value, bytes);
         for (unsigned i = 0; i < bytes; ++i)
            expected_write[output_base + offset + i] = true;
      }
   }
   auto before = cpu.memory;
   for (auto &lane : cpu.inputs[0])
      lane[0].u32 = stride;
   cpu.run(s);
   for (unsigned lane = 0; lane < 32; ++lane) {
      for (unsigned j = 0; j < 2; ++j) {
         float actual = half ? _mesa_half_to_float(cpu.outputs[0][lane][j].u16) :
                               cpu.outputs[0][lane][j].f32;
         CHECK(actual == float(row(lane) * 8 + col(lane) + j) + 0.5f);
      }
   }
   for (unsigned i = 0; i < cpu.memory.size(); ++i) {
      CHECK(cpu.writes[i] == unsigned(expected_write[i]));
      if (expected_write[i])
         CHECK(cpu.memory[i] == before[input_base + i - output_base]);
      else
         CHECK(cpu.memory[i] == before[i]);
   }
}

static void
test_element_operations()
{
   shader s;
   interpreter cpu;
   auto *a = s.matrix(GLSL_TYPE_FLOAT, GLSL_CMAT_USE_A);
   auto *h = s.matrix(GLSL_TYPE_FLOAT16, GLSL_CMAT_USE_B);
   auto *b = s.matrix(GLSL_TYPE_FLOAT, GLSL_CMAT_USE_B);
   auto *c = s.matrix(GLSL_TYPE_FLOAT, GLSL_CMAT_USE_ACCUMULATOR);
   auto *array_type = glsl_array_type(glsl_array_type(a->type, 3, 0), 2, 0);
   auto *array = nir_local_variable_create(s.b.impl, array_type, "nested_matrices");
   auto *elem = nir_build_deref_array_imm(&s.b,
      nir_build_deref_array_imm(&s.b, nir_build_deref_var(&s.b, array), 1), 2);
   nir_cmat_construct(&s.b, &a->def, nir_imm_float(&s.b, 1.2345));
   nir_cmat_copy(&s.b, &elem->def, &a->def);
   nir_cmat_convert(&s.b, &h->def, &elem->def);
   nir_cmat_convert(&s.b, &b->def, &h->def);
   nir_cmat_bitcast(&s.b, &c->def, &b->def);
   nir_cmat_unary_op(&s.b, &c->def, &c->def, .alu_op = nir_op_fneg);
   nir_cmat_binary_op(&s.b, &c->def, &c->def, &c->def, .alu_op = nir_op_fadd);
   nir_cmat_scalar_op(&s.b, &c->def, &c->def, nir_imm_float(&s.b, 3),
                      .alu_op = nir_op_fmul);
   auto *index = nir_iand_imm(&s.b, nir_load_subgroup_invocation(&s.b), 1);
   nir_cmat_insert(&s.b, &c->def, nir_imm_float(&s.b, 7), &c->def, index);
   s.output(nir_cmat_extract(&s.b, 32, &c->def, index), 0);
   s.output(nir_cmat_extract(&s.b, 32, &c->def,
                            nir_ixor(&s.b, index, nir_imm_int(&s.b, 1))), 1);
   auto desc = *glsl_get_cmat_description(a->type);
   s.output(nir_cmat_length(&s.b, .cmat_desc = desc), 2);
   s.output(nir_cmat_get_coordinate(&s.b, index, .cmat_desc = desc), 3);
   cpu.run(s);
   for (unsigned lane = 0; lane < 32; ++lane) {
      CHECK(cpu.outputs[0][lane][0].f32 == 7.0f);
      CHECK(cpu.outputs[1][lane][0].f32 == -6.0f * rounded(1.2345f, true));
      CHECK(cpu.outputs[2][lane][0].u32 == 2);
      CHECK(cpu.outputs[3][lane][0].u32 == row(lane));
      CHECK(cpu.outputs[3][lane][1].u32 == col(lane) + (lane & 1));
   }
}

static void
test_rejection(unsigned which)
{
   shader s;
   auto *good = s.matrix(GLSL_TYPE_FLOAT, GLSL_CMAT_USE_A);
   nir_cmat_construct(&s.b, &good->def, nir_imm_float(&s.b, 1));
   auto *a = s.matrix(which == 2 ? GLSL_TYPE_INT : GLSL_TYPE_FLOAT,
                      GLSL_CMAT_USE_A, which == 1 ? 16 : 8,
                      which == 3 ? SCOPE_WORKGROUP : SCOPE_SUBGROUP);
   auto *b = s.matrix(which == 4 ? GLSL_TYPE_FLOAT16 : GLSL_TYPE_FLOAT,
                      GLSL_CMAT_USE_B);
   auto *c = s.matrix(which == 5 ? GLSL_TYPE_FLOAT16 : GLSL_TYPE_FLOAT,
                      GLSL_CMAT_USE_ACCUMULATOR);
   if (which == 8) {
      nir_cmat_transpose(&s.b, &a->def, &a->def);
   } else {
      nir_cmat_muladd(&s.b, &c->def, &a->def, &b->def, &c->def,
                      .saturate = which == 6,
                      .cmat_signed_mask = which == 7 ? NIR_CMAT_A_SIGNED :
                                                      nir_cmat_signed(0));
   }
   char *before = nir_shader_as_str(s.b.shader, nullptr);
   const char *reason = nullptr;
   CHECK(!agx_nir_lower_apple9_cooperative_matrix(
      s.b.shader, which == 0 ? 16 : 32, &reason));
   CHECK(reason && *reason);
   char *after = nir_shader_as_str(s.b.shader, nullptr);
   CHECK(strcmp(before, after) == 0);
   CHECK(s.b.shader->info.cs.has_cooperative_matrix);
   ralloc_free(before);
   ralloc_free(after);
}

static unsigned
count_muladd(nir_shader *s)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic)
               count += nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_cmat_muladd;
         }
      }
   }
   return count;
}

static void
test_modules(const char *directory)
{
   const struct {
      const char *file, *entry;
      unsigned muladds, y, z;
   } cases[] = {
      {"tilematmul", "tile_matmul_f32", 2, 1, 1},
      {"tilematmul_rt", "tile_matmul_rt2x2_f32", 4, 1, 1},
      {"tilematmul_rt", "tile_matmul_rt4x2_f32", 8, 1, 1},
      {"tilematmul_rt", "tile_matmul_rt4x4_f32", 16, 1, 1},
      {"tilematmul_h", "tile_matmul_h2x2", 4, 1, 1},
      {"tilematmul_h", "tile_matmul_h4x4", 16, 1, 1},
      {"gemm", "gemm_f32_t64x64x32", 832, 2, 2},
   };
   for (auto &test : cases) {
      std::string path = std::string(directory) + "/" + test.file + ".spv";
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      CHECK(file.good());
      auto bytes = file.tellg();
      CHECK(bytes > 0 && bytes % 4 == 0);
      std::vector<uint32_t> words(size_t(bytes) / 4);
      file.seekg(0);
      file.read(reinterpret_cast<char *>(words.data()), bytes);
      CHECK(file.good());
      uint32_t sizes[3] = {32, test.y, test.z};
      nir_spirv_specialization_entry entries[3] = {};
      for (unsigned i = 0; i < 3; ++i) {
         entries[i].id = i;
         entries[i].size = 4;
         entries[i].data = reinterpret_cast<uint8_t *>(&sizes[i]);
      }
      nir_spirv_specialization spec = {3, entries};
      spirv_to_nir_options spv = {};
      spv.environment = NIR_SPIRV_VULKAN;
      spv.ssbo_addr_format = nir_address_format_32bit_index_offset;
      spv.ubo_addr_format = nir_address_format_32bit_index_offset;
      spv.shared_addr_format = nir_address_format_32bit_offset;
      spv.push_const_addr_format = nir_address_format_32bit_offset;
      spv.phys_ssbo_addr_format = nir_address_format_64bit_global;
      spv.group_non_uniform_subgroup_size = 32;
      spv.skip_os_break_in_debug_build = true;
      spv.ignore_replacement = true;
      /* Like spirv2nir, this exercises the importer, not device capabilities. */
      nir_shader *s = spirv_to_nir(words.data(), words.size(), &spec,
                                  MESA_SHADER_COMPUTE, test.entry, &spv, &options);
      CHECK(s);
      nir_validate_shader(s, "actual Metal2vk input");
      CHECK(count_muladd(s) == test.muladds);
      const char *reason = nullptr;
      CHECK(agx_nir_lower_apple9_cooperative_matrix(s, 32, &reason));
      CHECK(!reason);
      check_lowered(s);
      nir_validate_shader(s, "actual Metal2vk lowered");
      nir_lower_var_copies(s);
      nir_lower_vars_to_ssa(s);
      nir_opt_copy_prop(s);
      nir_opt_dce(s);
      nir_validate_shader(s, "actual Metal2vk after SSA");
      /* Idempotence is success, not a false failure/progress signal. */
      CHECK(agx_nir_lower_apple9_cooperative_matrix(s, 32, &reason));
      printf("PASS %s (%u muladds lowered)\n", test.entry, test.muladds);
      ralloc_free(s);
   }
}

int
main(int argc, char **argv)
{
   CHECK(argc == 1 || argc == 2);
   glsl_type_singleton_init_or_ref();
   for (bool half : {false, true}) {
      for (unsigned pattern = 0; pattern < 4; ++pattern)
         for (bool alias : {false, true})
            test_muladd(half, pattern, alias);
      for (bool column_major : {false, true})
         for (unsigned width : {1u, 4u})
            for (auto use : {GLSL_CMAT_USE_A, GLSL_CMAT_USE_B})
               test_memory(half, column_major, width, use);
   }
   for (bool column_major : {false, true})
      test_memory(false, column_major, 1, GLSL_CMAT_USE_ACCUMULATOR);
   test_element_operations();
   for (unsigned i = 0; i < 9; ++i)
      test_rejection(i);
   printf("PASS 16 multiply-add, 18 memory-layout, element-operation and 9 rejection cases\n");
   if (argc == 2)
      test_modules(argv[1]);
   glsl_type_singleton_decref();
   return 0;
}
