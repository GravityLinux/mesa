/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "agx_apple9_launch.h"
#include "asahi/compiler/agx_apple9_encoding.h"
#include "asahi/compiler/agx_apple9_profile.h"

#include <string.h>

/* Three stage recipes share the call/frame layout. Stage-dependent behavior
 * includes generated resource setup and established relocations. Unresolved
 * coverage, call/frame and stage-ending spans remain external. */
static const struct {
   bool state_root;
   bool tile_layout;
} stages[] = {
   [AGX_APPLE9_LAUNCH_VERTEX] = {false, false},
   [AGX_APPLE9_LAUNCH_FRAGMENT] = {false, true},
   [AGX_APPLE9_LAUNCH_COMPUTE] = {true, false},
};

static uint16_t
read16(const uint8_t *p)
{
   return p[0] | ((uint16_t)p[1] << 8);
}

static void
write16(uint8_t *p, uint16_t value)
{
   p[0] = value;
   p[1] = value >> 8;
}

static bool
pointer_fits(uint64_t base, uint64_t address)
{
   return address >= base && ((address - base) >> 13) <= UINT16_MAX;
}

bool
agx_apple9_launch_threadgroup_memory_supported(uint32_t bytes)
{
   return bytes == 0 ||
          (bytes >= 128 && bytes <= 1024 && !(bytes & (bytes - 1)));
}

static void
write_pointer(uint8_t *p, uint64_t relative)
{
   p[1] = 0x80 | (relative & 0x7f);
   p[4] = (p[4] & ~0x1f) | ((relative >> 6) & 0x1e);
   p[5] = (p[5] & ~0x0c) | ((relative >> 9) & 0x0c);
   write16(p + 6, relative >> 13);
}

bool
agx_apple9_launch_import(struct agx_apple9_launch_recipe *recipe,
                         struct agx_apple9_launch_parameters *defaults,
                         enum agx_apple9_launch_stage stage,
                         const uint8_t *data, size_t size, unsigned call_offset)
{
   if (!recipe || !defaults || !data ||
       (unsigned)stage >= sizeof(stages) / sizeof(stages[0]) ||
       call_offset < 8 || call_offset > size || size - call_offset < 66 ||
       memcmp(data + call_offset + 8, "\xf7\x00\x2a", 3))
      return false;

   *recipe = (struct agx_apple9_launch_recipe){
      .stage = stage,
      .prefix = {data, call_offset + 8},
      .shared = {data + call_offset + 8, 58},
      .suffix = {data + call_offset + 66, size - call_offset - 66},
      .size = size,
      .pre_call = {data[call_offset - 7], data[call_offset - 6]},
      .shared_tag = data[call_offset + 38],
   };
   *defaults = (struct agx_apple9_launch_parameters){
      .publication_word = read16(data + call_offset + 11),
      .frame_extent_a = read16(data + call_offset + 13),
      .frame_extent_b = read16(data + call_offset + 15),
      .samples = 1,
   };
   return true;
}

size_t
agx_apple9_launch_call_offset(const struct agx_apple9_launch_recipe *recipe,
                              unsigned resource_count)
{
   if (!recipe || recipe->prefix.size < 8)
      return 0;
   if (!recipe->generated_resources)
      return recipe->prefix.size - 8;
   if (recipe->stage != AGX_APPLE9_LAUNCH_COMPUTE)
      return recipe->coverage_setup.size ? 98 : 68;
   if (!resource_count || resource_count > AGX_APPLE9_COMPUTE_MAX_RESOURCES)
      return 0;
   unsigned roots = resource_count + AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE;
   return 32 + 14 * ((roots + 1) / 2) + 32 - 8;
}

/* Materialize the compute ABI from the actual number of bound resources.
 * State loads use the same pending-load form as pointer tables. Call setup
 * and return remain explicit external fragments. */
static bool
build_compute_resources(uint8_t *out, size_t capacity,
                        const struct agx_apple9_launch_recipe *recipe,
                        const struct agx_apple9_launch_parameters *params)
{
   if (recipe->stage != AGX_APPLE9_LAUNCH_COMPUTE ||
       !recipe->prefix.data || recipe->prefix.size != (recipe->generated_state_load ? 18 : 32) ||
       !recipe->suffix.data || recipe->suffix.size != 12 ||
       recipe->coverage_setup.size || recipe->suffix_shared.size || !params->resource_count ||
       params->resource_count > AGX_APPLE9_COMPUTE_MAX_RESOURCES)
      return false;
   unsigned roots =
      AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE + params->resource_count;
   const unsigned pending = 18, root = 2;
   enum { max_roots = AGX_APPLE9_COMPUTE_VISIBLE_ARGUMENT_BASE +
                      AGX_APPLE9_COMPUTE_MAX_RESOURCES };
   uint8_t prefix[32 + 14 * ((max_roots + 1) / 2) + 32] = {0};
   uint8_t suffix[4 * (2 * max_roots + 4) + 12];
   unsigned at = 32;
   for (int i = (roots - 1) & ~1; i >= 0; i -= 2) {
      if (!agx_apple9_encode_pointer_load(prefix + at, pending + 2 * i,
                                          root, i, i + 2 <= roots ? 2 : 1))
         return false;
      at += 14;
   }
   if (recipe->generated_state_load) {
      /* Four ordinary state words, consumed by the argument transfers below.
       * EXP-175 readbacks cover changing records and independent placement. */
      if (!agx_apple9_encode_pointer_load(prefix + at, pending + 2 * roots,
                                         40, 0, 2))
         return false;
      at += 14;
      memcpy(prefix + at, recipe->prefix.data, 18);
      at += 18;
   } else {
      memcpy(prefix + at, recipe->prefix.data, 32);
      prefix[at + 3] = 2 * (pending + 2 * roots);
      at += 32;
   }
   unsigned words = 2 * roots + 4;
   for (unsigned i = 0; i < words; ++i) {
      if (!agx_apple9_encode_argument_word(suffix + 4 * i, i, pending + i, i == 0))
         return false;
   }
   memcpy(suffix + 4 * words, recipe->suffix.data, 12);
   struct agx_apple9_launch_recipe generated = *recipe;
   generated.generated_resources = false;
   generated.prefix = (struct agx_apple9_launch_fragment){prefix, at};
   generated.suffix = (struct agx_apple9_launch_fragment){suffix, 4 * words + 12};
   return agx_apple9_launch_build(out, capacity, &generated, params);
}

/* The graphics entry table has six pointer slots. Texture, sampler and buffer
 * roots occupy the first three; the driver currently clears the other three.
 * Preserve coverage's measured auxiliary spans around the generated loads. */
static bool
build_graphics_resources(uint8_t *out, size_t capacity,
                         const struct agx_apple9_launch_recipe *recipe,
                         const struct agx_apple9_launch_parameters *params)
{
   bool coverage = recipe->coverage_setup.size != 0;
   if ((recipe->stage != AGX_APPLE9_LAUNCH_VERTEX &&
        recipe->stage != AGX_APPLE9_LAUNCH_FRAGMENT) ||
       !recipe->prefix.data || recipe->prefix.size != 18 ||
       recipe->suffix_shared.size ||
       (coverage && (recipe->stage != AGX_APPLE9_LAUNCH_FRAGMENT ||
                     !recipe->coverage_setup.data || recipe->coverage_setup.size != 30)))
      return false;
   uint8_t prefix[106] = {0}, transfers[48];
   const unsigned root = 2, pending = 18;
   unsigned at = 16;
   if (coverage) {
      memcpy(prefix + at, recipe->coverage_setup.data, 16);
      at += 16;
   }
   for (int pointer = 4; pointer >= 0; pointer -= 2) {
      if (!agx_apple9_encode_pointer_load(prefix + at, pending + 2 * pointer,
                                          root, pointer, 2))
         return false;
      at += 14;
   }
   if (coverage) {
      memcpy(prefix + at, recipe->coverage_setup.data + 16, 14);
      at += 14;
   }
   memcpy(prefix + at, recipe->prefix.data, 18);
   at += 18;
   for (unsigned word = 0; word < 12; ++word) {
      if (!agx_apple9_encode_argument_word(transfers + 4 * word, word,
                                           pending + word, word == 0))
         return false;
   }
   struct agx_apple9_launch_recipe generated = *recipe;
   generated.generated_resources = false;
   generated.coverage_setup = (struct agx_apple9_launch_fragment){0};
   generated.prefix = (struct agx_apple9_launch_fragment){prefix, at};
   generated.suffix_shared = (struct agx_apple9_launch_fragment){transfers, sizeof(transfers)};
   if (!agx_apple9_launch_build(out, capacity, &generated, params))
      return false;
   /* Replace the compatibility relocation only after all validation succeeds.
    * Both fixed entry registers are within the literal encoder's valid range. */
   agx_apple9_encode_literal32(out, root, params->resource_table);
   agx_apple9_encode_literal32(out + 8, root + 1, params->resource_table >> 32);
   return true;
}

bool
agx_apple9_launch_build(uint8_t *out, size_t capacity,
                        const struct agx_apple9_launch_recipe *recipe,
                        const struct agx_apple9_launch_parameters *params)
{
   if (!out || !recipe || !params)
      return false;
   if (recipe->generated_resources)
      return recipe->stage == AGX_APPLE9_LAUNCH_COMPUTE
                ? build_compute_resources(out, capacity, recipe, params)
                : build_graphics_resources(out, capacity, recipe, params);
   if ((unsigned)recipe->stage >= sizeof(stages) / sizeof(stages[0]) ||
       recipe->coverage_setup.size ||
       !recipe->prefix.data || !recipe->shared.data ||
       (recipe->suffix_shared.size && !recipe->suffix_shared.data) ||
       (recipe->suffix.size && !recipe->suffix.data) ||
       recipe->prefix.size < 16 || recipe->shared.size != 58 ||
       recipe->size > capacity || recipe->prefix.size > recipe->size ||
       recipe->shared.size > recipe->size - recipe->prefix.size ||
       recipe->suffix_shared.size >
          recipe->size - recipe->prefix.size - recipe->shared.size ||
       recipe->suffix.size > recipe->size - recipe->prefix.size -
                                recipe->shared.size -
                                recipe->suffix_shared.size ||
       memcmp(recipe->shared.data, "\xf7\x00\x2a", 3) ||
       params->main_call > 0x3ffff ||
       (params->publication_count_valid && params->publication_count > 1022) ||
       !pointer_fits(params->shader_base, params->resource_table))
      return false;

   unsigned stage = recipe->stage;
   size_t call = recipe->prefix.size - 8;
   if (params->threadgroup_memory_bytes &&
       (stage != AGX_APPLE9_LAUNCH_COMPUTE ||
        !agx_apple9_launch_threadgroup_memory_supported(
           params->threadgroup_memory_bytes) ||
        call < 10 ||
        memcmp(recipe->prefix.data + call - 10, "\x77\x00\x2a", 3) ||
        recipe->pre_call[0] != 0x41))
      return false;
   if (stages[stage].state_root &&
       (call < 32 + 10 ||
        !pointer_fits(params->shader_base, params->state) ||
        ((params->state - params->shader_base) & 0x3f) != 0x20))
      return false;

   if (stages[stage].tile_layout &&
       (!params->tile_bytes || params->tile_bytes > 64 ||
        (params->samples != 1 && params->samples != 2 && params->samples != 4) ||
        recipe->pre_call[0] != 0x43))
      return false;

   /* Validate first so callers can publish the result without a partial
    * launcher becoming visible on an invalid relocation. */
   memcpy(out, recipe->prefix.data, recipe->prefix.size);
   memcpy(out + recipe->prefix.size, recipe->shared.data, recipe->shared.size);
   size_t suffix = recipe->prefix.size + recipe->shared.size;
   if (recipe->suffix_shared.size)
      memcpy(out + suffix, recipe->suffix_shared.data,
             recipe->suffix_shared.size);
   suffix += recipe->suffix_shared.size;
   if (recipe->suffix.size)
      memcpy(out + suffix, recipe->suffix.data, recipe->suffix.size);
   memset(out + suffix + recipe->suffix.size, 0,
          capacity - suffix - recipe->suffix.size);

   out[call] = params->main_call;
   out[call + 1] = params->main_call >> 8;
   out[call + 2] = params->main_call >> 16;
   memcpy(out + call - 7, recipe->pre_call, 2);
   if (params->threadgroup_memory_bytes) {
      uint32_t word = (params->threadgroup_memory_bytes << 2) | 0x80;
      write16(out + call - 6, word);
      write16(out + call - 4, word >> 16);
   }
   out[call + 38] = recipe->shared_tag;
   uint16_t publication_word = params->publication_word;
   if (params->publication_count_valid) {
      /* T8132 32-bit publication boundary tests: one unit holds a pair.
       * Preserve low bits whose semantics have not been established. */
      publication_word = (publication_word & 0x7f) |
                         (((params->publication_count + 1) / 2) << 7);
   }
   write16(out + call + 11, publication_word);
   write16(out + call + 13, params->frame_extent_a);
   write16(out + call + 15, params->frame_extent_b);
   if (stages[stage].state_root) {
      /* T8132 compute entry ABI, established with authored memory accesses
       * and independently relocated roots: r2:r3 addresses the resource
       * record, r40:r41 addresses the state record. Each pair holds a full
       * GPU pointer, low word first. These are entry-setup allocation choices;
       * the resource load encoder also accepts independently tested pairs. */
      agx_apple9_encode_literal32(out, 2, params->resource_table);
      agx_apple9_encode_literal32(out + 8, 3, params->resource_table >> 32);
      agx_apple9_encode_literal32(out + 16, 40, params->state);
      agx_apple9_encode_literal32(out + 24, 41, params->state >> 32);
   } else {
      write_pointer(out, params->resource_table - params->shader_base);
   }
   if (stages[stage].tile_layout) {
      unsigned units = (params->tile_bytes + 7) / 8;
      if (params->samples == 4) {
         out[call - 8] = 0xaa;
         out[call - 7] = 2 * units;
         out[call - 6] = 0x43;
      } else {
         out[call - 7] = (params->samples == 1 ? 0x41 : 0x40) + 2 * units;
      }
   }
   return true;
}

static uint32_t
read32(const uint8_t *p)
{
   return read16(p) | ((uint32_t)read16(p + 2) << 16);
}

bool
agx_apple9_launch_library_open(struct agx_apple9_launch_library *library,
                               const uint8_t *data, size_t size)
{
   const size_t header_size = 16 + 9 * 8 + 3 * 12 + 4;
   if (!library || !data || size < header_size ||
       (memcmp(data, "A9LFRG02", 8) && memcmp(data, "A9LFRG03", 8) &&
        memcmp(data, "A9LFRG04", 8) && memcmp(data, "A9LFRG05", 8)) ||
       read32(data + 8) != 9 || read32(data + 12) != 3)
      return false;

   struct agx_apple9_launch_library result = {
      .configurations = data + 16 + 9 * 8,
      .generated_resources = memcmp(data, "A9LFRG02", 8) != 0,
      .generated_graphics_resources = !memcmp(data, "A9LFRG04", 8) ||
                                      !memcmp(data, "A9LFRG05", 8),
      .generated_state_load = !memcmp(data, "A9LFRG05", 8),
   };
   for (unsigned i = 0; i < 9; ++i) {
      size_t offset = read32(data + 16 + i * 8);
      size_t length = read32(data + 20 + i * 8);
      if (offset < header_size || offset > size || length > size - offset)
         return false;
      result.fragments[i] =
         (struct agx_apple9_launch_fragment){data + offset, length};
   }
   /* Fixed layouts for versions 2 through 5. Other layouts need a new schema,
    * not silent reinterpretation of an unknown executable input. */
   const bool graphics = result.generated_graphics_resources;
   const unsigned sizes[] = {graphics ? 18 : 76, graphics ? 30 : 106,
                             result.generated_state_load ? 18 :
                             result.generated_resources ? 32 : 148, 58,
                             graphics ? 0 : 48};
   if (result.generated_resources && result.fragments[8].size != 12)
      return false;
   for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
      if (result.fragments[i].size != sizes[i])
         return false;
   }
   if (memcmp(result.fragments[3].data, "\xf7\x00\x2a", 3))
      return false;
   for (unsigned i = 0; i < 3; ++i) {
      const uint8_t *config = result.configurations + i * 12;
      size_t prefix = i == AGX_APPLE9_LAUNCH_COMPUTE
                         ? result.fragments[2].size : 76;
      size_t used = prefix + 58 + (i == AGX_APPLE9_LAUNCH_COMPUTE ? 0 : 48);
      size_t output_size = read16(config + 10);
      unsigned tail = i == AGX_APPLE9_LAUNCH_COMPUTE ? 8 : 5 + i;
      if (config[9] ||
          output_size != (i == AGX_APPLE9_LAUNCH_COMPUTE ? 1024 : 256) ||
          used > output_size ||
          result.fragments[tail].size > output_size - used ||
          (i == AGX_APPLE9_LAUNCH_FRAGMENT && config[6] != 0x43))
         return false;
   }
   if (read16(result.configurations + 38) ||
       result.fragments[7].size > 256 - 106 - 58 - 48)
      return false;
   *library = result;
   return true;
}

bool
agx_apple9_launch_select(const struct agx_apple9_launch_library *library,
                         enum agx_apple9_launch_stage stage, bool coverage,
                         struct agx_apple9_launch_recipe *recipe,
                         struct agx_apple9_launch_parameters *params)
{
   if (!library || !library->configurations || !recipe || !params ||
       (unsigned)stage >= sizeof(stages) / sizeof(stages[0]) ||
       (coverage && stage != AGX_APPLE9_LAUNCH_FRAGMENT))
      return false;

   const uint8_t *config = library->configurations + stage * 12;
   bool compute = stage == AGX_APPLE9_LAUNCH_COMPUTE;
   *recipe = (struct agx_apple9_launch_recipe){
      .stage = stage,
      .generated_resources = compute ? library->generated_resources
                                     : library->generated_graphics_resources,
      .generated_state_load = compute && library->generated_state_load,
      .coverage_setup = coverage && library->generated_graphics_resources
                           ? library->fragments[1]
                           : (struct agx_apple9_launch_fragment){0},
      .prefix = library->fragments[compute    ? 2
                                   : coverage && !library->generated_graphics_resources ? 1
                                              : 0],
      .shared = library->fragments[3],
      .suffix_shared = compute ? (struct agx_apple9_launch_fragment){0}
                               : library->fragments[4],
      .suffix = library->fragments[compute    ? 8
                                   : coverage ? 7
                                              : 5 + stage],
      .size = read16(config + 10),
      .pre_call = {config[6], config[7]},
      .shared_tag = config[8],
   };
   *params = (struct agx_apple9_launch_parameters){
      .publication_word =
         read16(coverage ? library->configurations + 36 : config),
      .frame_extent_a = read16(config + 2),
      .frame_extent_b = read16(config + 4),
      .samples = 1,
   };
   return true;
}
