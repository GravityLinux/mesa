/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#include "agx_apple9_ir.h"
#include "util/hash_table.h"
#include "util/u_memory.h"

/* Selection creates address arithmetic and packing which did not exist
 * during NIR optimization. Keep a semantic whitelist: publications, implicit
 * state, derivatives and memory operations are not ordinary scalar ALU. */
bool
agx_apple9_instr_is_pure_alu(const struct agx_apple9_vir_instr *I)
{
   if ((I->dest_components != 1 && I->op != AGX_APPLE9_VIR_IMUL_WIDE &&
        I->op != AGX_APPLE9_VIR_UNPACK_NORM) ||
       I->publication_handoff ||
       I->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       I->encoding == AGX_APPLE9_ENC_LOGIC_EXPORT)
      return false;

   switch (I->op) {
   case AGX_APPLE9_VIR_IMM:
   case AGX_APPLE9_VIR_U2F32:
   case AGX_APPLE9_VIR_I2F32:
   case AGX_APPLE9_VIR_F2I32:
   case AGX_APPLE9_VIR_F2U32:
   case AGX_APPLE9_VIR_F2F16:
   case AGX_APPLE9_VIR_PACK_HALF_2X16:
   case AGX_APPLE9_VIR_PACK_UNORM_2X16:
   case AGX_APPLE9_VIR_PACK_UNORM_4X8:
   case AGX_APPLE9_VIR_UNPACK_HALF:
   case AGX_APPLE9_VIR_UNPACK_NORM:
   case AGX_APPLE9_VIR_IADD:
   case AGX_APPLE9_VIR_IMUL:
   case AGX_APPLE9_VIR_ISUB:
   case AGX_APPLE9_VIR_IMAD:
   case AGX_APPLE9_VIR_IMUL_WIDE:
   case AGX_APPLE9_VIR_IAND:
   case AGX_APPLE9_VIR_IOR_UNIFORM:
   case AGX_APPLE9_VIR_IOR:
   case AGX_APPLE9_VIR_IXOR:
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHL:
   case AGX_APPLE9_VIR_IMIN:
   case AGX_APPLE9_VIR_IMAX:
   case AGX_APPLE9_VIR_UMIN:
   case AGX_APPLE9_VIR_UMAX:
   case AGX_APPLE9_VIR_FADD:
   case AGX_APPLE9_VIR_FSUB:
   case AGX_APPLE9_VIR_FMUL:
   case AGX_APPLE9_VIR_FADD_IMM:
   case AGX_APPLE9_VIR_FMUL_IMM:
   case AGX_APPLE9_VIR_FSAT:
   case AGX_APPLE9_VIR_FMIN:
   case AGX_APPLE9_VIR_FMAX:
   case AGX_APPLE9_VIR_FMA:
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
   case AGX_APPLE9_VIR_FFLOOR:
   case AGX_APPLE9_VIR_FCEIL:
   case AGX_APPLE9_VIR_FTRUNC:
   case AGX_APPLE9_VIR_FROUND_EVEN:
   case AGX_APPLE9_VIR_SELECT:
      return true;
   default:
      return false;
   }
}

struct expression {
   uint32_t op, encoding, immediate, nr_srcs, components;
   uint32_t src_abs, src_neg, saturate;
   uint32_t src_uniform, src_immediate, src_value[3];
   uint32_t src[AGX_APPLE9_MAX_VIR_SRCS];
};

bool
agx_apple9_supports_saturate(const struct agx_apple9_vir_instr *I)
{
   if (I->publication_handoff ||
       I->encoding == AGX_APPLE9_ENC_FLOAT2_EXPORT ||
       I->encoding == AGX_APPLE9_ENC_FLOAT2_PROJECT)
      return false;
   switch (I->op) {
   case AGX_APPLE9_VIR_FADD:
   case AGX_APPLE9_VIR_FSUB:
   case AGX_APPLE9_VIR_FMUL:
   case AGX_APPLE9_VIR_FADD_IMM:
   case AGX_APPLE9_VIR_FMUL_IMM:
   case AGX_APPLE9_VIR_FSAT:
   case AGX_APPLE9_VIR_FMA:
   case AGX_APPLE9_VIR_FRCP:
   case AGX_APPLE9_VIR_FRSQ:
   case AGX_APPLE9_VIR_FSQRT_FACTOR:
   case AGX_APPLE9_VIR_FSIN_FACTOR:
   case AGX_APPLE9_VIR_FEXP2:
   case AGX_APPLE9_VIR_FLOG2:
   case AGX_APPLE9_VIR_DERIVATIVE:
      return true;
   default:
      /* The ordinary SFU saturation bit is ineffective on direct rounding.
       * Min/max and selects also need an explicit clamping operation. */
      return false;
   }
}

static void
select_saturated_encoding(struct agx_apple9_vir_instr *I)
{
   if (!I->saturate)
      return;
   switch (I->encoding) {
   case AGX_APPLE9_ENC_FLOAT2_COMPACT:
      I->encoding = AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED;
      break;
   case AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT:
      I->encoding = AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED;
      break;
   case AGX_APPLE9_ENC_FLOAT3_EXTENDED:
      I->encoding = AGX_APPLE9_ENC_FLOAT3_SATURATE_EXTENDED;
      break;
   default:
      break;
   }
}

static uint32_t
expression_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct expression));
}

static bool
expression_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(struct expression)) == 0;
}

static uint32_t
resolve(uint32_t *replacement, uint32_t value)
{
   uint32_t root = value;
   while (replacement[root] != root)
      root = replacement[root];
   while (replacement[value] != root) {
      uint32_t next = replacement[value];
      replacement[value] = root;
      value = next;
   }
   return root;
}

static bool
fold_integer(struct agx_apple9_vir_instr *I, const bool *constant,
             const uint32_t *values)
{
   if (!I->nr_srcs || I->nr_srcs > 3)
      return false;
   uint32_t c[3] = {0};
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      if (!constant[I->src[s]])
         return false;
      c[s] = values[I->src[s]];
   }

   uint32_t value;
   switch (I->op) {
   case AGX_APPLE9_VIR_IADD: value = c[0] + c[1]; break;
   case AGX_APPLE9_VIR_IMUL: value = c[0] * c[1]; break;
   case AGX_APPLE9_VIR_ISUB: value = c[0] - c[1]; break;
   case AGX_APPLE9_VIR_IMAD: value = c[0] * c[1] + c[2]; break;
   case AGX_APPLE9_VIR_IAND: value = c[0] & c[1]; break;
   case AGX_APPLE9_VIR_IOR: value = c[0] | c[1]; break;
   case AGX_APPLE9_VIR_IXOR: value = c[0] ^ c[1]; break;
   case AGX_APPLE9_VIR_ISHR:
   case AGX_APPLE9_VIR_USHR:
   case AGX_APPLE9_VIR_ISHL: {
      unsigned shift = I->nr_srcs == 2 ? (c[1] & 0xffff) : I->immediate;
      if (shift >= 32) {
         value = I->op == AGX_APPLE9_VIR_ISHR && (c[0] & 0x80000000u)
            ? UINT32_MAX : 0;
         break;
      }
      value = I->op == AGX_APPLE9_VIR_ISHL ? c[0] << shift : c[0] >> shift;
      if (I->op == AGX_APPLE9_VIR_ISHR && shift && (c[0] & 0x80000000u))
         value |= UINT32_MAX << (32 - shift);
      break;
   }
   default:
      return false;
   }

   I->op = AGX_APPLE9_VIR_IMM;
   I->encoding = value <= 0x7f ? AGX_APPLE9_ENC_MOV_IMM_COMPACT
                              : AGX_APPLE9_ENC_MOV_IMM32;
   I->nr_srcs = 0;
   I->immediate = value;
   return true;
}

static void
select_float_sources(struct agx_apple9_vir_instr *I,
                     struct agx_apple9_vir_instr **definitions,
                     const bool *constant, const uint32_t *values)
{
   if (I->alu_src_uniform_mask || I->alu_src_immediate_mask)
      return;
   bool special = I->encoding == AGX_APPLE9_ENC_FLOAT_SPECIAL;
   bool select = I->op == AGX_APPLE9_VIR_SELECT;
   bool minmax = I->op == AGX_APPLE9_VIR_FMIN || I->op == AGX_APPLE9_VIR_FMAX;
   unsigned condition = I->immediate & 0xff;
   if (select && condition != AGX_APPLE9_SELECT_FEQ &&
       condition != AGX_APPLE9_SELECT_FGT && condition != AGX_APPLE9_SELECT_FLT)
      return;
   if (!special && !select && !minmax && I->op != AGX_APPLE9_VIR_FADD && I->op != AGX_APPLE9_VIR_FSUB &&
       I->op != AGX_APPLE9_VIR_FMUL && I->op != AGX_APPLE9_VIR_FMA)
      return;

   if (I->op == AGX_APPLE9_VIR_FSUB) {
      I->op = AGX_APPLE9_VIR_FADD;
      I->src_neg_mask ^= 2;
   }

   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      if (minmax && s == 0)
         continue;
      unsigned bit = 1u << s;
      while (true) {
         const struct agx_apple9_vir_instr *producer = definitions[I->src[s]];
         if (!producer || !agx_apple9_instr_is_pure_alu(producer) ||
             producer->nr_srcs != 2 ||
             (producer->op != AGX_APPLE9_VIR_IXOR &&
              producer->op != AGX_APPLE9_VIR_IAND))
            break;
         uint32_t mask = producer->op == AGX_APPLE9_VIR_IXOR
            ? 0x80000000u : 0x7fffffffu;
         int source = -1;
         for (unsigned c = 0; c < 2; ++c) {
            if (constant[producer->src[c]] && values[producer->src[c]] == mask)
               source = 1 - c;
         }
         if (source < 0)
            break;
         if (minmax && producer->op == AGX_APPLE9_VIR_IAND)
            break;
         /* Compose modifiers from the outside in. Absolute value discards
          * any negation inside it, but keeps an outer negation. */
         if (producer->op == AGX_APPLE9_VIR_IAND)
            I->src_abs_mask |= bit;
         else if (!(I->src_abs_mask & bit))
            I->src_neg_mask ^= bit;
         I->src[s] = producer->src[source];
      }
   }

   if (select) {
      /* Swap the ordered comparison for a negated first operand. When both
       * inputs are negated, reversing the relation removes both negations.
       * These identities preserve equality and unordered NaN comparisons. */
      if (I->src_neg_mask & 1) {
         if (condition == AGX_APPLE9_SELECT_FGT) condition = AGX_APPLE9_SELECT_FLT;
         else if (condition == AGX_APPLE9_SELECT_FLT) condition = AGX_APPLE9_SELECT_FGT;
         if (I->src_neg_mask & 2) {
            I->src_neg_mask &= ~3u;
         } else {
            uint32_t source = I->src[0];
            I->src[0] = I->src[1]; I->src[1] = source;
            I->src_neg_mask = (I->src_neg_mask & ~3u) | 2;
            I->src_abs_mask = (I->src_abs_mask & ~3u) |
               ((I->src_abs_mask & 1) << 1) | ((I->src_abs_mask & 2) >> 1);
         }
         I->immediate = (I->immediate & ~0xffu) | condition;
      }
      I->encoding = (I->src_abs_mask & 3) ? AGX_APPLE9_ENC_SELECT_MODIFIER_EXTENDED
                                          : AGX_APPLE9_ENC_SELECT_GPR_WIDE;
      return;
   }
   if (special || minmax)
      return;

   /* Inline constants use an exact minifloat. Preserve unrepresentable
    * constants and modifiers which this six-byte form cannot express. */
   if (I->nr_srcs == 2) {
      for (int c = 1; c >= 0; --c) {
         unsigned other = 1 - c;
         if (!constant[I->src[c]] || (I->src_abs_mask & (1u << other)))
            continue;
         uint32_t value = values[I->src[c]];
         if (I->src_abs_mask & (1u << c))
            value &= 0x7fffffffu;
         if (I->src_neg_mask & (1u << c))
            value ^= 0x80000000u;
         uint8_t encoded;
         if (!agx_apple9_encode_float_immediate(value, &encoded))
            continue;
         I->op = I->op == AGX_APPLE9_VIR_FMUL ? AGX_APPLE9_VIR_FMUL_IMM
                                             : AGX_APPLE9_VIR_FADD_IMM;
         I->encoding = I->saturate ? AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED
                                  : AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_COMPACT;
         I->immediate = value;
         I->src[0] = I->src[other];
         I->src_neg_mask = (I->src_neg_mask >> other) & 1;
         I->src_abs_mask = 0;
         I->nr_srcs = 1;
         return;
      }
   }

   if (I->op == AGX_APPLE9_VIR_FMA) {
      I->encoding = (I->src_abs_mask & 3)
         ? AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED
         : AGX_APPLE9_ENC_FLOAT3_EXTENDED;
   } else {
      I->encoding = I->src_abs_mask
         ? (I->op == AGX_APPLE9_VIR_FMUL ? AGX_APPLE9_ENC_FLOAT2_MUL_ABS_EXTENDED
                                       : AGX_APPLE9_ENC_FLOAT2_ABS_EXTENDED)
         : (I->src_neg_mask & 1) ? AGX_APPLE9_ENC_FLOAT2_MODIFIER_EXTENDED
                                : AGX_APPLE9_ENC_FLOAT2_COMPACT;
   }
   select_saturated_encoding(I);
}

static void
inline_float_sources(struct agx_apple9_vir_instr *I, const uint8_t *uniform_read,
                      const bool *constant, const uint32_t *values)
{
   bool fma = I->op == AGX_APPLE9_VIR_FMA;
   unsigned arity = fma ? 3 : 2;
   if ((!fma && I->op != AGX_APPLE9_VIR_FADD && I->op != AGX_APPLE9_VIR_FMUL) ||
       I->nr_srcs != arity || I->alu_src_uniform_mask || I->alu_src_immediate_mask)
      return;

   uint8_t file[3] = {0};
   uint32_t value[3] = {0};
   for (unsigned s = 0; s < arity; ++s) {
      if (uniform_read[I->src[s]]) {
         file[s] = AGX_APPLE9_FILE_UNIFORM;
         value[s] = uniform_read[I->src[s]] - 1;
      } else if (fma && constant[I->src[s]]) {
         uint8_t encoded;
         if (agx_apple9_encode_float_immediate(values[I->src[s]], &encoded)) {
            file[s] = AGX_APPLE9_FILE_IMMEDIATE;
            value[s] = values[I->src[s]];
         }
      }
   }
   /* FMA's second multiplicand has a native alternative file selector.
    * Commuting the product preserves its absolute/negate composition. */
   if (fma && file[0] && !file[1]) {
      uint32_t src = I->src[0];
      I->src[0] = I->src[1]; I->src[1] = src;
      file[1] = file[0]; value[1] = value[0];
      I->src_abs_mask = (I->src_abs_mask & 4) |
         ((I->src_abs_mask & 1) << 1) | ((I->src_abs_mask & 2) >> 1);
      I->src_neg_mask = (I->src_neg_mask & 4) |
         ((I->src_neg_mask & 1) << 1) | ((I->src_neg_mask & 2) >> 1);
   }
   if (fma || (file[0] && file[1]))
      file[0] = 0;

   unsigned source = 0;
   for (unsigned s = 0; s < arity; ++s) {
      if (file[s] == AGX_APPLE9_FILE_UNIFORM) {
         I->alu_src_uniform_mask |= BITFIELD_BIT(s);
         I->alu_src_value[s] = value[s];
      } else if (file[s] == AGX_APPLE9_FILE_IMMEDIATE) {
         if (I->src_abs_mask & BITFIELD_BIT(s)) value[s] &= 0x7fffffffu;
         if (I->src_neg_mask & BITFIELD_BIT(s)) value[s] ^= 0x80000000u;
         I->src_abs_mask &= ~BITFIELD_BIT(s);
         I->src_neg_mask &= ~BITFIELD_BIT(s);
         I->alu_src_immediate_mask |= BITFIELD_BIT(s);
         I->alu_src_value[s] = value[s];
      } else {
         I->src[source++] = I->src[s];
      }
   }
   I->nr_srcs = source;
   if (fma) {
      I->encoding = (I->src_abs_mask & 3)
         ? AGX_APPLE9_ENC_FLOAT3_MODIFIER_EXTENDED
         : AGX_APPLE9_ENC_FLOAT3_EXTENDED;
      select_saturated_encoding(I);
   }
}

static bool
fold_saturate(struct agx_apple9_vir_program *p, const uint32_t *replacement)
{
   struct agx_apple9_vir_instr **definitions =
      CALLOC(p->value_count, sizeof(*definitions));
   unsigned *uses = CALLOC(p->value_count, sizeof(*uses));
   unsigned *scope = CALLOC(p->value_count, sizeof(*scope));
   if (!definitions || !uses || !scope) {
      free(definitions); free(uses); free(scope);
      return false;
   }
   unsigned epoch = 0;
   struct agx_apple9_block *block = NULL;
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *I = p->instructions[i];
      if (block != agx_apple9_instr_block(I)) {
         block = agx_apple9_instr_block(I);
         ++epoch;
      }
      switch (I->op) {
      case AGX_APPLE9_VIR_EXEC_MASK_PUSH:
      case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
      case AGX_APPLE9_VIR_EXEC_MASK_POP:
      case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
      case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
      case AGX_APPLE9_VIR_LOOP_MASK_POP:
      case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
         ++epoch;
         break;
      default:
         break;
      }
      bool eliminated = I->dest != AGX_APPLE9_VREG_INVALID &&
                        replacement[I->dest] != I->dest;
      if (!eliminated) {
         for (unsigned s = 0; s < I->nr_srcs; ++s)
            ++uses[I->src[s]];
      }
      for (unsigned c = 0; c < I->dest_components; ++c) {
         definitions[I->dest + c] = I;
         scope[I->dest + c] = epoch;
      }
   }
   for (unsigned i = 0; i < p->live_out_count; ++i)
      ++uses[p->live_out[i]];
   if (p->output != AGX_APPLE9_VREG_INVALID)
      ++uses[p->output];

   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *I = p->instructions[i];
      if (I->op != AGX_APPLE9_VIR_FSAT)
         continue;
      unsigned source = I->src[0];
      struct agx_apple9_vir_instr *producer = definitions[source];
      /* Keep the unclamped value when it has another use. Moving a pure
       * producer into its sole clamp use preserves SSA and source lifetime;
       * an execution-mask transition would change its active lanes. */
      if (producer && producer->op != AGX_APPLE9_VIR_FSAT &&
          producer->dest_components == 1 && uses[source] == 1 &&
          uses[I->dest] && scope[source] == scope[I->dest] &&
          agx_apple9_supports_saturate(producer)) {
         uint32_t dest = I->dest;
         *I = *producer;
         I->dest = dest;
      } else {
         /* Adding positive zero applies the ordinary FP32 canonicalization
          * before clamping. This also handles a load, select or rounding
          * result whose producer has no supported saturation modifier. */
         I->op = AGX_APPLE9_VIR_FADD_IMM;
         I->encoding = AGX_APPLE9_ENC_FLOAT2_IMMEDIATE_EXTENDED;
         I->immediate = 0;
      }
      I->saturate = true;
      select_saturated_encoding(I);
   }
   free(definitions); free(uses); free(scope);
   return true;
}

bool
agx_apple9_optimize_vir(struct agx_apple9_vir_program *p)
{
   assert(!p->physical && !p->dependencies_finalized);
   if (!p->value_count)
      return true;

   uint32_t *replacement = MALLOC(p->value_count * sizeof(*replacement));
   uint32_t *values = CALLOC(p->value_count, sizeof(*values));
   bool *constant = CALLOC(p->value_count, sizeof(*constant));
   uint8_t *uniform_read = CALLOC(p->value_count, sizeof(*uniform_read));
   struct expression *keys = CALLOC(p->value_count, sizeof(*keys));
   struct agx_apple9_vir_instr **definitions =
      CALLOC(p->value_count, sizeof(*definitions));
   struct hash_table *expressions =
      _mesa_hash_table_create(NULL, expression_hash, expression_equal);
   bool success = replacement && values && constant && uniform_read && keys && definitions && expressions;
   if (!success)
      goto cleanup;
   for (unsigned v = 0; v < p->value_count; ++v)
      replacement[v] = v;

   struct agx_apple9_block *block = NULL;
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *I = p->instructions[i];
      if (block != agx_apple9_instr_block(I)) {
         block = agx_apple9_instr_block(I);
         _mesa_hash_table_clear(expressions, NULL);
         memset(uniform_read, 0, p->value_count);
      }
      for (unsigned s = 0; s < I->nr_srcs; ++s)
         I->src[s] = resolve(replacement, I->src[s]);
      switch (I->op) {
      case AGX_APPLE9_VIR_STORE_UNIFORM:
         /* Uniform reads are reusable only while their argument window is
          * unchanged. Setup may publish another value to the same word. */
         _mesa_hash_table_clear(expressions, NULL);
         memset(uniform_read, 0, p->value_count);
         break;
      case AGX_APPLE9_VIR_EXEC_MASK_PUSH:
      case AGX_APPLE9_VIR_EXEC_MASK_ELSE:
      case AGX_APPLE9_VIR_EXEC_MASK_POP:
      case AGX_APPLE9_VIR_LOOP_MASK_PUSH:
      case AGX_APPLE9_VIR_LOOP_MASK_UPDATE:
      case AGX_APPLE9_VIR_LOOP_MASK_POP:
      case AGX_APPLE9_VIR_BREAK_MASK_UNWIND:
         /* A synthetic mask transition may share a logical block. Values
          * produced under its old active lanes need not cover the new ones. */
         _mesa_hash_table_clear(expressions, NULL);
         memset(uniform_read, 0, p->value_count);
         break;
      default:
         break;
      }
      if (!agx_apple9_instr_is_pure_alu(I))
         continue;
      /* Fixed interface values are not interchangeable with ordinary SSA. */
      bool constrained = false;
      for (unsigned c = 0; c < I->dest_components; ++c)
         constrained |= p->fixed_phys[I->dest + c] != AGX_APPLE9_PHYS_INVALID ||
                        p->max_phys[I->dest + c] != AGX_APPLE9_PHYS_INVALID;
      if (constrained)
         continue;

      fold_integer(I, constant, values);
      select_float_sources(I, definitions, constant, values);
      inline_float_sources(I, uniform_read, constant, values);
      for (unsigned c = 0; c < I->dest_components; ++c)
         definitions[I->dest + c] = I;
      if (I->op == AGX_APPLE9_VIR_IMM) {
         constant[I->dest] = true;
         values[I->dest] = I->immediate;
      }
      if (I->op == AGX_APPLE9_VIR_IOR_UNIFORM && I->nr_srcs == 1 &&
          I->immediate < 64 && constant[I->src[0]] && !values[I->src[0]])
         uniform_read[I->dest] = I->immediate + 1;
      struct expression *key = &keys[I->dest];
      *key = (struct expression){
         .op = I->op, .encoding = I->encoding,
         .immediate = I->immediate, .nr_srcs = I->nr_srcs,
         .components = I->dest_components,
         .src_abs = I->src_abs_mask, .src_neg = I->src_neg_mask,
         .saturate = I->saturate,
         .src_uniform = I->alu_src_uniform_mask,
         .src_immediate = I->alu_src_immediate_mask,
      };
      for (unsigned s = 0; s < 3; ++s) {
         if ((key->src_uniform | key->src_immediate) & BITFIELD_BIT(s))
            key->src_value[s] = I->alu_src_value[s];
      }
      memcpy(key->src, I->src, I->nr_srcs * sizeof(I->src[0]));
      struct hash_entry *entry = _mesa_hash_table_search(expressions, key);
      if (entry) {
         struct agx_apple9_vir_instr *previous = entry->data;
         for (unsigned c = 0; c < I->dest_components; ++c)
            replacement[I->dest + c] = previous->dest + c;
      } else if (!_mesa_hash_table_insert(expressions, key, I)) {
         success = false;
         goto cleanup;
      }
   }

   /* Backedge phi uses can precede their definition in layout order.
    * Rewrite the complete use set once after collecting all replacements.
    * Shared DCE removes the now-unused definitions before allocation. */
   for (unsigned i = 0; i < p->instruction_count; ++i) {
      struct agx_apple9_vir_instr *I = p->instructions[i];
      for (unsigned s = 0; s < I->nr_srcs; ++s)
         I->src[s] = resolve(replacement, I->src[s]);
   }
   for (unsigned i = 0; i < p->live_out_count; ++i)
      p->live_out[i] = resolve(replacement, p->live_out[i]);
   if (p->output != AGX_APPLE9_VREG_INVALID)
      p->output = resolve(replacement, p->output);
   success = fold_saturate(p, replacement);

cleanup:
   free(uniform_read);
   agx_apple9_invalidate_uses(p);
   if (expressions)
      _mesa_hash_table_destroy(expressions, NULL);
   free(keys);
   free(definitions);
   free(constant);
   free(values);
   free(replacement);
   return success;
}
