/* SPDX-License-Identifier: MIT */
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "../hk_shader.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%u: %s\n", __LINE__, #c); failures++; } } while (0)
static unsigned failures;

/* A vertex shader storing the given number of components to consecutive
 * generic slots, four per slot, and a fragment shader reading them. */
static void
make(unsigned components, unsigned bit_size, nir_shader **vs, nir_shader **fs)
{
   nir_builder v = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &agx_nir_options, "vs");
   nir_builder f = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options, "fs");
   unsigned per_slot = bit_size == 64 ? 2 : 4;
   for (unsigned slot = 0; components; ++slot) {
      unsigned n = MIN2(components, per_slot);
      components -= n;
      unsigned loc = VARYING_SLOT_VAR0 + slot;
      nir_store_output(&v, nir_imm_zero(&v, n, bit_size), nir_imm_int(&v, 0),
                       .write_mask = BITFIELD_MASK(n),
                       .src_type = bit_size == 64 ? nir_type_float64 : nir_type_float32,
                       .io_semantics = {.location = loc, .num_slots = 1});
      v.shader->info.outputs_written |= BITFIELD64_BIT(loc);
      nir_load_input(&f, n, bit_size, nir_imm_int(&f, 0),
                     .io_semantics = {.location = loc, .num_slots = 1});
      f.shader->info.inputs_read |= BITFIELD64_BIT(loc);
   }
   *vs = v.shader;
   *fs = f.shader;
}

static struct hk_apple9_stipple_slots
slots_for(unsigned components, unsigned bit_size)
{
   nir_shader *vs, *fs;
   make(components, bit_size, &vs, &fs);
   struct hk_apple9_stipple_slots s = hk_apple9_line_stipple_slots(vs, fs);
   ralloc_free(vs);
   ralloc_free(fs);
   return s;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const unsigned limit = HK_APPLE9_OUTPUT_COMPONENTS - HK_APPLE9_STIPPLE_COMPONENTS;

   /* Fits: the highest free slots, start above axis. */
   struct hk_apple9_stipple_slots s = slots_for(limit - 1, 32);
   CHECK(s.start == VARYING_SLOT_VAR31 && s.axis == VARYING_SLOT_VAR30);
   s = slots_for(limit, 32);
   CHECK(s.start && s.axis);

   /* One component over the budget falls back to unstippled. */
   s = slots_for(limit + 1, 32);
   CHECK(!s.start && !s.axis);
   s = slots_for(HK_APPLE9_OUTPUT_COMPONENTS, 32);
   CHECK(!s.start && !s.axis);

   /* 64-bit components count twice. */
   s = slots_for((limit + 2) / 2, 64);
   CHECK(!s.start && !s.axis);
   s = slots_for(limit / 2 - 1, 64);
   CHECK(s.start && s.axis);

   /* Thirty-one used generic slots leave room for only one. */
   nir_builder v = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &agx_nir_options, "vs");
   nir_builder f = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options, "fs");
   for (unsigned k = 0; k < 31; ++k) {
      unsigned loc = VARYING_SLOT_VAR0 + k;
      nir_store_output(&v, nir_imm_float(&v, 0), nir_imm_int(&v, 0),
                       .write_mask = 1, .src_type = nir_type_float32,
                       .io_semantics = {.location = loc, .num_slots = 1});
      v.shader->info.outputs_written |= BITFIELD64_BIT(loc);
   }
   s = hk_apple9_line_stipple_slots(v.shader, f.shader);
   CHECK(!s.start && !s.axis);
   ralloc_free(v.shader);
   ralloc_free(f.shader);

   glsl_type_singleton_decref();
   if (!failures)
      puts("line stipple slots: exact component budget, fallback below it");
   return failures != 0;
}
