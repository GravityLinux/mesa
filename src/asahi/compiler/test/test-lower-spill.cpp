/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_apple9_ir.h"
#include <vector>
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include "util/macros.h"
#include <gtest/gtest.h>

#define CASE(instr, expected)                                                  \
   do {                                                                        \
      agx_builder *A = agx_test_builder(mem_ctx);                              \
      agx_builder *B = agx_test_builder(mem_ctx);                              \
      {                                                                        \
         agx_builder *b = A;                                                   \
         instr;                                                                \
      }                                                                        \
      {                                                                        \
         agx_builder *b = B;                                                   \
         expected;                                                             \
      }                                                                        \
      A->shader->scratch_size_B = 1024;                                        \
      agx_lower_spill(A->shader);                                              \
      ASSERT_SHADER_EQUAL(A->shader, B->shader);                               \
   } while (0)

class LowerSpill : public testing::Test {
 protected:
   LowerSpill()
   {
      mem_ctx = ralloc_context(NULL);

      wx = agx_register(0, AGX_SIZE_32);
      hy = agx_register(2, AGX_SIZE_16);

      mw4 = agx_memory_register(0, AGX_SIZE_32);
      mh4 = agx_memory_register(0, AGX_SIZE_16);
      mw4.channels_m1 = 4 - 1;
      mh4.channels_m1 = 4 - 1;

      wx4 = wx;
      wx4.channels_m1 = 4 - 1;

      hy4 = hy;
      hy4.channels_m1 = 4 - 1;
   }

   ~LowerSpill()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
   agx_index wx, hy, wx4, hy4;
   agx_index mw4, mh4;

   unsigned scalar = BITFIELD_MASK(1);
   unsigned vec4 = BITFIELD_MASK(4);

   enum agx_format i16 = AGX_FORMAT_I16;
   enum agx_format i32 = AGX_FORMAT_I32;
};

TEST_F(LowerSpill, ScalarSpills)
{
   CASE(agx_mov_to(b, agx_memory_register(11, AGX_SIZE_16), hy),
        agx_stack_store(b, hy, agx_immediate(22), i16, scalar));

   CASE(agx_mov_to(b, agx_memory_register(18, AGX_SIZE_32), wx),
        agx_stack_store(b, wx, agx_immediate(36), i32, scalar));
}

TEST_F(LowerSpill, ScalarFills)
{
   CASE(agx_mov_to(b, hy, agx_memory_register(11, AGX_SIZE_16)),
        agx_stack_load_to(b, hy, agx_immediate(22), i16, scalar));

   CASE(agx_mov_to(b, wx, agx_memory_register(18, AGX_SIZE_32)),
        agx_stack_load_to(b, wx, agx_immediate(36), i32, scalar));
}

TEST_F(LowerSpill, VectorSpills)
{
   CASE(agx_mov_to(b, mh4, hy4),
        agx_stack_store(b, hy4, agx_immediate(0), i16, vec4));

   CASE(agx_mov_to(b, mw4, wx4),
        agx_stack_store(b, wx4, agx_immediate(0), i32, vec4));
}

TEST_F(LowerSpill, VectorFills)
{
   CASE(agx_mov_to(b, hy4, mh4),
        agx_stack_load_to(b, hy4, agx_immediate(0), i16, vec4));

   CASE(agx_mov_to(b, wx4, mw4),
        agx_stack_load_to(b, wx4, agx_immediate(0), i32, vec4));
}

TEST_F(LowerSpill, ScalarSpill64)
{
   CASE(agx_mov_to(b, agx_memory_register(16, AGX_SIZE_64),
                   agx_register(8, AGX_SIZE_64)),
        agx_stack_store(b, agx_register(8, AGX_SIZE_64), agx_immediate(32), i32,
                        BITFIELD_MASK(2)));
}

TEST_F(LowerSpill, ScalarFill64)
{
   CASE(agx_mov_to(b, agx_register(16, AGX_SIZE_64),
                   agx_memory_register(8, AGX_SIZE_64)),
        agx_stack_load_to(b, agx_register(16, AGX_SIZE_64), agx_immediate(16),
                          i32, BITFIELD_MASK(2)));
}

TEST_F(LowerSpill, Vec6Spill)
{
   CASE(
      {
         agx_index mvec6 = agx_memory_register(16, AGX_SIZE_32);
         agx_index vec6 = agx_register(8, AGX_SIZE_32);
         vec6.channels_m1 = 6 - 1;
         mvec6.channels_m1 = 6 - 1;

         agx_mov_to(b, mvec6, vec6);
      },
      {
         agx_index vec4 = agx_register(8, AGX_SIZE_32);
         agx_index vec2 = agx_register(8 + (4 * 2), AGX_SIZE_32);
         vec4.channels_m1 = 4 - 1;
         vec2.channels_m1 = 2 - 1;

         agx_stack_store(b, vec4, agx_immediate(32), i32, BITFIELD_MASK(4));
         agx_stack_store(b, vec2, agx_immediate(32 + 4 * 4), i32,
                         BITFIELD_MASK(2));
      });
}

TEST_F(LowerSpill, Vec6Fill)
{
   CASE(
      {
         agx_index mvec6 = agx_memory_register(16, AGX_SIZE_32);
         agx_index vec6 = agx_register(8, AGX_SIZE_32);
         vec6.channels_m1 = 6 - 1;
         mvec6.channels_m1 = 6 - 1;

         agx_mov_to(b, vec6, mvec6);
      },
      {
         agx_index vec4 = agx_register(8, AGX_SIZE_32);
         agx_index vec2 = agx_register(8 + (4 * 2), AGX_SIZE_32);
         vec4.channels_m1 = 4 - 1;
         vec2.channels_m1 = 2 - 1;

         agx_stack_load_to(b, vec4, agx_immediate(32), i32, BITFIELD_MASK(4));
         agx_stack_load_to(b, vec2, agx_immediate(32 + 4 * 4), i32,
                           BITFIELD_MASK(2));
      });
}

/* Selected uniform expressions must reconstruct their complete input graph,
 * preserve machine constraints, and decline mutable uniforms. */
TEST(Apple9Rematerialization, BoundedImmutableExpressions)
{
   for (unsigned layers : {0u, 1u, 3u}) {
      for (bool mutable_uniforms : {false, true}) {
         SCOPED_TRACE(layers);
         SCOPED_TRACE(mutable_uniforms);
         void *mem = ralloc_context(nullptr);
         agx_builder *b = agx_test_builder(mem);
         std::vector<agx_index> values;
         auto zero = agx_mov_imm(b, 32, 0);
         struct agx_reg_constraint constraint = {.max = 126, .align = 2};
         for (unsigned i = 0; i < 12; ++i) {
            auto selected = rzalloc(mem, agx_apple9_vir_instr);
            selected->op = AGX_APPLE9_VIR_IOR_UNIFORM;
            selected->encoding = AGX_APPLE9_ENC_LOGIC_UNIFORM;
            selected->dest_components = 1;
            selected->nr_srcs = 1;
            selected->immediate = 12 + i;
            auto I = agx_alloc_instr(b, AGX_OPCODE_APPLE9_PURE, 1, 1);
            I->apple9 = selected;
            I->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
            I->src[0] = zero;
            I->reg_constraints = rzalloc_array(mem, agx_reg_constraint, 2);
            I->reg_constraints[0] = constraint;
            I->reg_constraints[1] = constraint;
            agx_builder_insert(&b->cursor, I);
            agx_index value = I->dest[0];
            for (unsigned j = 0; j < layers; ++j) {
               auto add = agx_alloc_instr(b, AGX_OPCODE_APPLE9_PURE, 1, 1);
               add->apple9 = rzalloc(mem, agx_apple9_vir_instr);
               add->apple9->op = AGX_APPLE9_VIR_FADD_IMM;
               add->apple9->dest_components = 1;
               add->apple9->nr_srcs = 1;
               add->apple9->immediate = 0x3f800000;
               add->reg_constraints = I->reg_constraints;
               add->src[0] = value;
               value = add->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
               agx_builder_insert(&b->cursor, add);
            }
            values.push_back(value);
         }
         if (mutable_uniforms) {
            auto selected = rzalloc(mem, agx_apple9_vir_instr);
            selected->op = AGX_APPLE9_VIR_STORE_UNIFORM;
            auto I = agx_alloc_instr(b, AGX_OPCODE_APPLE9, 0, 1);
            I->apple9 = selected;
            I->src[0] = zero;
            agx_builder_insert(&b->cursor, I);
         }
         for (auto value : values) {
            auto I = agx_alloc_instr(b, AGX_OPCODE_APPLE9, 0, 1);
            I->src[0] = value;
            I->apple9 = rzalloc(mem, agx_apple9_vir_instr);
            I->apple9->op = AGX_APPLE9_VIR_DEVICE_STORE;
            agx_builder_insert(&b->cursor, I);
         }
         agx_compute_liveness(b->shader);
         agx_spill(b->shader, 16, false);
         unsigned memory_moves = 0, imports = 0;
         agx_foreach_instr_global(b->shader, I) {
            if (I->op == AGX_OPCODE_MOV)
               memory_moves += I->dest[0].memory || I->src[0].memory;
            if (I->op == AGX_OPCODE_APPLE9_PURE) {
               imports += I->apple9->op == AGX_APPLE9_VIR_IOR_UNIFORM;
               ASSERT_NE(I->reg_constraints, nullptr);
               EXPECT_EQ(I->reg_constraints[0].max, constraint.max);
               EXPECT_EQ(I->reg_constraints[0].align, constraint.align);
               EXPECT_EQ(I->nr_srcs, 1u);
            }
         }
         EXPECT_EQ(imports, 12u);
         if (mutable_uniforms || layers == 3)
            EXPECT_GT(memory_moves, 0u);
         else
            EXPECT_EQ(memory_moves, 0u);
         ralloc_free(mem);
      }
   }
}
