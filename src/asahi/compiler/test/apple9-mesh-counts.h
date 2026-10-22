/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 * Shared runtime-count tests for the T16 and T32 host interpreters.
 */

/* Runtime counts, including a return before publication and two static calls
 * in mutually exclusive uniform branches. Inputs stay in memory so neither
 * the NIR interpreter nor the compiler can fold the tested paths. */
static nir_shader *
conditional_counts(unsigned mode, unsigned threads)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "native runtime output counts");
   nir_shader *nir = b.shader;
   nir->info.workgroup_size[0] = threads;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.mesh.max_vertices_out = 3;
   nir->info.mesh.max_primitives_out = 1;
   nir->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_def *v = nir_load_global(&b, 1, 32, nir_imm_int64(&b, INPUT_ADDR),
                                .align_mul = 4);
   nir_def *p = nir_load_global(&b, 1, 32, nir_imm_int64(&b, INPUT_ADDR + 4),
                                .align_mul = 4);
   nir_def *enabled = nir_ine_imm(
      &b, nir_load_global(&b, 1, 32, nir_imm_int64(&b, INPUT_ADDR + 8),
                           .align_mul = 4), 0);
   if (mode == 1) {
      nir_push_if(&b, nir_inot(&b, enabled));
      nir_jump(&b, nir_jump_return);
      nir_pop_if(&b, NULL);
   }
   if (mode == 2) {
      nir_push_if(&b, enabled);
      nir_set_vertex_and_primitive_count(&b, v, p, nir_imm_int(&b, 0));
      nir_push_else(&b, NULL);
      nir_set_vertex_and_primitive_count(
         &b, nir_imm_int(&b, 0), nir_imm_int(&b, 0), nir_imm_int(&b, 0));
      nir_pop_if(&b, NULL);
      v = nir_bcsel(&b, enabled, v, nir_imm_int(&b, 0));
      p = nir_bcsel(&b, enabled, p, nir_imm_int(&b, 0));
   } else {
      nir_set_vertex_and_primitive_count(&b, v, p, nir_imm_int(&b, 0));
   }
   /* No output access on the empty paths. */
   nir_push_if(&b, nir_ior(&b, nir_ieq_imm(&b, v, 0), nir_ieq_imm(&b, p, 0)));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);
   /* The last logical invocation owns output, including for serialized
    * workgroups. Legal V1/V2 triangles use repeated indices below V; overflow
    * cases still write only within the declared V3/P1 allocation. */
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b),
                              threads - 1));
   for (unsigned i = 0; i < 3; ++i) {
      nir_push_if(&b, nir_ult(&b, nir_imm_int(&b, i), v));
      nir_def *position[4];
      for (unsigned c = 0; c < 4; ++c)
         position[c] = nir_imm_int(&b, 0x3f000000 + 4 * i + c);
      nir_store_per_vertex_output(
         &b, nir_vec(&b, position, 4), nir_imm_int(&b, i), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_pop_if(&b, NULL);
   }
   nir_def *indices[3];
   for (unsigned i = 0; i < 3; ++i)
      indices[i] = nir_umin(&b, nir_imm_int(&b, i), nir_iadd_imm(&b, v, -1));
   nir_store_per_primitive_output(
      &b, nir_vec(&b, indices, 3), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .write_mask = 7, .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   nir_pop_if(&b, NULL);
   nir_validate_shader(nir, "runtime mesh counts source");
   return nir;
}

static void
check_conditional_counts(void)
{
   /* Explicit expected V/P keeps the oracle independent of the lowering. */
   const struct {
      uint32_t input[3];
      unsigned vertices, primitives;
   } cases[] = {
      {{3, 1, 1}, 3, 1}, {{1, 1, 1}, 1, 1}, {{2, 1, 1}, 2, 1},
      {{0, 0, 1}, 0, 0}, {{0, 1, 1}, 0, 0}, {{3, 0, 1}, 0, 0},
      {{3, 1, 0}, 3, 1}, {{0, 0, 0}, 0, 0},
      {{4, 1, 1}, 3, 1}, {{3, 2, 1}, 3, 1}, {{4, 2, 1}, 3, 1},
      {{65536, 65536, 1}, 3, 1}, {{UINT32_MAX, UINT32_MAX, 1}, 3, 1},
      {{UINT32_MAX, 0, 1}, 0, 0}, {{0, UINT32_MAX, 1}, 0, 0},
   };
   const unsigned sizes[] = {
      1, 16,
#if LANES == 32
      17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128,
#endif
   };
   const struct agx_apple9_native_mesh_options options = {
      .physical_threads = LANES,
   };
   for (unsigned t = 0; t < ARRAY_SIZE(sizes); ++t) {
      for (unsigned mode = 0; mode < 3; ++mode) {
         for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
            nir_shader *nir = conditional_counts(mode, sizes[t]);
            const char *reason = NULL;
            struct agx_apple9_mesh_body_info info;
            CHECK(agx_nir_lower_apple9_native_mesh(nir, true, &options, &info,
                                                   &reason));
            CHECK(info.logical_size[0] == sizes[t]);
            CHECK(info.key.physical_threads == LANES);
            nir_lower_vars_to_ssa(nir);
            nir_opt_dce(nir);
            nir_validate_shader(nir, "runtime mesh counts lowered");
            bool enabled = !mode || cases[i].input[2];
            unsigned vertices = enabled ? cases[i].vertices : 0;
            unsigned primitives = enabled ? cases[i].primitives : 0;
            struct machine m = {
               .variable_counts = true, .expected_vertices = vertices,
               .expected_primitives = primitives,
            };
            /* Detect an uninitialized count on the early-return path. */
            memset(m.local, 0xa5, sizeof(m.local));
            m.region[5] = (struct region){INPUT_ADDR, sizeof(cases[i].input),
                                          (uint8_t *)cases[i].input, NULL};
            nir_function_impl *impl = nir_shader_get_entrypoint(nir);
            nir_index_ssa_defs(impl);
            m.value = calloc(impl->ssa_alloc, sizeof(*m.value));
            CHECK(m.value);
            execute(&m, &impl->body, ALL_LANES);
            CHECK(m.controls == ALL_LANES && m.scalars == 1);
            for (unsigned c = 0; c < 4; ++c) {
               CHECK(m.channels[c] == BITFIELD_MASK(vertices));
               for (unsigned v = 0; v < 3; ++v) {
                  uint32_t stored;
                  memcpy(&stored, m.local + info.layout.position_B +
                                  c * info.layout.vertex_pitch_B + 4 * v, 4);
                  CHECK(stored == (v < vertices ? 0x3f000000 + 4 * v + c
                                               : 0xa5a5a5a5u));
                  if (v < vertices)
                     CHECK(m.submitted_position[v][c] == stored);
               }
            }
            for (unsigned c = 0; c < 3; ++c)
               CHECK(m.local[c] == (vertices ? MIN2(c, vertices - 1) : 0xa5));
            uint32_t stored_vertices;
            uint16_t stored_primitives;
            memcpy(&stored_vertices, m.local + info.layout.vertex_count_B, 4);
            memcpy(&stored_primitives, m.local + info.layout.primitive_count_B, 2);
            CHECK(stored_vertices == vertices && stored_primitives == primitives);
            struct agx_shader_part code = {0};
            CHECK(agx_compile_apple9_mesh_lowered(nir, &info.key, &code, &reason));
            CHECK(code.binary && !code.info.scratch_size);
            free(code.binary);
            free(m.value);
            ralloc_free(nir);
         }
      }
   }
   printf("native counts T%u: %zu runtime/partial/empty/overflow paths executed and compiled\n",
          LANES, ARRAY_SIZE(sizes) * 3 * ARRAY_SIZE(cases));
}
