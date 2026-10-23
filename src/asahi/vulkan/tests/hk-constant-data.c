/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include "asahi/compiler/agx_compile_apple9.h"
#include "asahi/vulkan/hk_cmd_buffer.h"
#include "asahi/vulkan/hk_shader.h"
#include "asahi/vulkan/hk_physical_device.h"
#include "compiler/nir/nir_builder.h"
#include "vulkan/runtime/vk_pipeline.h"

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return false; \
} } while (0)

static void
fold(nir_shader *nir)
{
   bool progress;
   do {
      progress = nir_opt_constant_folding(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);
}

/* Exercise each stage's pointer slot, nonzero intrinsic base, varying index,
 * load width and end-of-allocation clamp. Constant folding is the oracle for
 * the resulting address; it does not substitute the lowering being tested.
 */
static bool
addresses(mesa_shader_stage stage, unsigned bits, unsigned components,
          unsigned offset, bool root_uniform)
{
   nir_builder b = nir_builder_init_simple_shader(stage, &agx_nir_options,
                                                  "constant_data_address");
   b.shader->constant_data_size = 256;
   b.shader->constant_data = rzalloc_size(b.shader, 256);
   nir_def *index = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x4000),
                                    .align_mul = 4);
   nir_def *value = nir_load_constant(&b, components, bits, index,
                                     .base = 16, .range = 240,
                                     .align_mul = bits / 8);
   nir_store_global(&b, value, nir_imm_int64(&b, 0x5000), .align_mul = 4,
                    .write_mask = BITFIELD_MASK(components));
   CHECK(hk_nir_lower_descriptors(b.shader, &vk_robustness_disabled, 0, NULL, root_uniform));
   b.cursor = nir_before_impl(b.impl);
   nir_def_rewrite_uses(index, nir_imm_int(&b, offset));

   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         CHECK(intr->intrinsic != nir_intrinsic_load_constant);
         if (intr->intrinsic == nir_intrinsic_load_root_agx) {
            b.cursor = nir_before_instr(instr);
            nir_def_replace(&intr->def, nir_imm_int64(&b, 0x100000));
         }
      }
   }
   fold(b.shader);

   unsigned roots = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (root_uniform && intr->intrinsic == nir_intrinsic_load_ubo) {
            CHECK(nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE);
            CHECK(nir_src_is_const(intr->src[0]) && nir_src_as_uint(intr->src[0]) == 0);
            CHECK(nir_src_is_const(intr->src[1]));
            CHECK(nir_src_as_uint(intr->src[1]) ==
                  hk_root_descriptor_offset(constant_data) + stage * 8);
            b.cursor = nir_before_instr(instr);
            nir_def_replace(&intr->def, nir_imm_int64(&b, 0x200000));
            ++roots;
         } else if (!root_uniform && intr->intrinsic == nir_intrinsic_load_global_constant &&
                    nir_src_is_const(intr->src[0])) {
            CHECK(nir_src_as_uint(intr->src[0]) == 0x100000 +
                  hk_root_descriptor_offset(constant_data) + stage * 8);
            CHECK(intr->def.bit_size == 64);
            b.cursor = nir_before_instr(instr);
            nir_def_replace(&intr->def, nir_imm_int64(&b, 0x200000));
            ++roots;
         }
      }
   }
   CHECK(roots == 1);
   struct hk_descriptor_state desc = {0};
   desc.root_constant_stages = BITFIELD_BIT(stage);
   CHECK(hk_descriptor_root_size(&desc) >=
         hk_root_descriptor_offset(constant_data) + (stage + 1) * 8);
   CHECK(hk_descriptor_root_size(&desc) <= sizeof(desc.root));
   desc.root_constant_stages = 0;
   CHECK(hk_descriptor_root_size(&desc) == hk_root_descriptor_offset(dynamic_buffers));
   desc.root_dynamic_count = 2;
   CHECK(hk_descriptor_root_size(&desc) >=
         hk_root_descriptor_offset(dynamic_buffers) + 2 * sizeof(struct hk_buffer_address));
   fold(b.shader);
   unsigned loads = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_global_constant) {
            unsigned last = 256 - components * bits / 8;
            unsigned expected = MIN2(offset + 16, last);
            CHECK(nir_src_is_const(intr->src[0]));
            CHECK(nir_src_as_uint(intr->src[0]) == 0x200000 + expected);
            CHECK(intr->def.bit_size == bits);
            CHECK(intr->def.num_components == components);
            CHECK(nir_intrinsic_align_mul(intr) == bits / 8);
            ++loads;
         }
      }
   }
   CHECK(loads == 1);
   nir_validate_shader(b.shader, "constant data address lowering");
   ralloc_free(b.shader);
   return true;
}

static bool
extract(bool mutable, bool dynamic)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                              &agx_nir_options, "constant_data_extraction");
   nir_variable *array = nir_local_variable_create(b.impl,
      glsl_array_type(glsl_vector_type(GLSL_TYPE_UINT, 2), 64, 0), "table");
   nir_deref_instr *d = nir_build_deref_var(&b, array);
   for (unsigned i = 0; i < 64; ++i) {
      nir_deref_instr *elem = nir_build_deref_array_imm(&b, d, i);
      nir_store_deref(&b, elem, nir_imm_ivec2(&b, i * 73 + 19, i * 31 + 7), 3);
   }
   nir_def *index = dynamic
      ? nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0)
      : nir_imm_int(&b, 17);
   nir_deref_instr *elem = nir_build_deref_array(&b, d, nir_iand_imm(&b, index, 63));
   if (mutable)
      nir_store_deref(&b, elem, nir_imm_ivec2(&b, 1, 2), 3);
   nir_def *value = nir_load_deref(&b, elem);
   nir_store_global(&b, value, nir_imm_int64(&b, 0x5000), .align_mul = 4, .write_mask = 3);
   struct hk_physical_device pdev = {0};
   pdev.dev.chip = AGX_CHIP_G16G;
   hk_preprocess_nir_internal(&pdev.vk, b.shader);
   CHECK(b.shader->constant_data_size == (mutable || !dynamic ? 0 : 512));
   if (!mutable && dynamic) {
      const uint32_t *data = b.shader->constant_data;
      for (unsigned i = 0; i < 64; ++i) {
         CHECK(data[2 * i] == i * 73 + 19);
         CHECK(data[2 * i + 1] == i * 31 + 7);
      }
      CHECK(hk_nir_lower_descriptors(b.shader, &vk_robustness_disabled, 0, NULL, true));
   }
   nir_validate_shader(b.shader, "constant data extraction");
   ralloc_free(b.shader);
   return true;
}

static bool
variants(void)
{
   struct hk_api_shader *s = calloc(1, sizeof(*s) + HK_GS_VARIANTS * sizeof(struct hk_shader));
   CHECK(s);
   s->vk.stage = MESA_SHADER_GEOMETRY;
   CHECK(hk_shader_constant_data(s) == 0);
   const uint32_t data[] = {0x1001, 0x2002};
   for (unsigned v = HK_GS_VARIANT_MAIN; v < HK_GS_VARIANTS; ++v) {
      s->variants[v].data_size = sizeof(data);
      s->variants[v].data_ptr = data;
      s->variants[v].data_addr = 0x800000 + v * 0x1000;
   }
   /* The generated rasterization variant has no original constant table. */
   hk_api_shader_init_constant_data(s);
   CHECK(hk_shader_constant_data(s) == 0x800000 + HK_GS_VARIANT_MAIN * 0x1000);
   CHECK(hk_shader_constant_data(NULL) == 0);
   free(s);
   return true;
}

/* Exercise real binds and root uploads, including the adjacent descriptor
 * buffer state. Address arithmetic alone missed the six-slot GL-sized array
 * being indexed by the task and mesh stages. */
static bool
bind_upload(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   uint8_t *pool = calloc(1, HK_CMD_BO_SIZE);
   CHECK(dev && cmd && pool);
   dev->dev.chip = AGX_CHIP_G16G;
   cmd->vk.base.device = &dev->vk;
   const uint64_t base = 0x1400000000ull;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].map = pool;
   cmd->uploader[HK_CMD_POOL_MAIN].chunks[0].base = base;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   uint64_t canary[HK_MAX_SETS];
   memset(canary, 0xa5, sizeof(canary));
   memcpy(desc->descriptor_buffers, canary, sizeof(canary));
   struct hk_api_shader shaders[MESA_SHADER_MESH_STAGES] = {0};
   for (unsigned stage = MESA_SHADER_VERTEX; stage <= MESA_SHADER_MESH; ++stage) {
      if (stage == MESA_SHADER_COMPUTE)
         continue;
      shaders[stage].vk.stage = stage;
      shaders[stage].constant_data_addr = 0x100000 + stage * 0x1000;
      hk_cmd_bind_graphics_shader(cmd, stage, &shaders[stage]);
      CHECK(!memcmp(desc->descriptor_buffers, canary, sizeof(canary)));
      CHECK(desc->root.constant_data[stage] == shaders[stage].constant_data_addr);
      uint64_t address = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
      CHECK(address >= base && address - base + sizeof(desc->root) <= HK_CMD_BO_SIZE);
      struct hk_root_descriptor_table *uploaded = (void *)(pool + address - base);
      CHECK(uploaded->constant_data[stage] == shaders[stage].constant_data_addr);
      CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == address);
   }
   /* Unbinding the last stage clears its slot without touching neighbours. */
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_MESH, NULL);
   CHECK(!desc->root.constant_data[MESA_SHADER_MESH]);
   CHECK(!(desc->root_constant_stages & BITFIELD_BIT(MESA_SHADER_MESH)));
   CHECK(!memcmp(desc->descriptor_buffers, canary, sizeof(canary)));
   /* Rebinding a native-only leaf updates its stage's constant root. */
   struct hk_api_shader leaf = {.apple9_native_leaf = true,
                                .constant_data_addr = 0x900000};
   leaf.vk.stage = MESA_SHADER_MESH;
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_MESH, &leaf);
   uint64_t address = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   struct hk_root_descriptor_table *uploaded = (void *)(pool + address - base);
   CHECK(uploaded->constant_data[MESA_SHADER_MESH] == leaf.constant_data_addr);
   CHECK(!memcmp(desc->descriptor_buffers, canary, sizeof(canary)));
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_MESH, NULL);
   CHECK(!desc->root.constant_data[MESA_SHADER_MESH]);
   CHECK(!(desc->root_constant_stages & BITFIELD_BIT(MESA_SHADER_MESH)));
   free(pool);
   free(cmd);
   free(dev);
   return true;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = extract(false, true) && extract(true, true) &&
             extract(false, false) && variants() && bind_upload();
   const unsigned offsets[] = {0, 8, 64, 240, 0xfffffff0};
   for (unsigned stage = MESA_SHADER_VERTEX; ok && stage <= MESA_SHADER_MESH; ++stage)
      for (unsigned bits = 8; ok && bits <= 64; bits *= 2)
         for (unsigned n = 1; ok && n <= 4; n *= 2)
            for (unsigned i = 0; ok && i < ARRAY_SIZE(offsets); ++i)
               for (unsigned uniform = 0; ok && uniform < 2; ++uniform)
                  ok = addresses(stage, bits, n, offsets[i], uniform);
   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}
