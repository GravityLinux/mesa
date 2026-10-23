# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# SPDX-License-Identifier: MIT

libasahi_agx_files = files(
  'agx_compile.c',
  'agx_apple9_machine.c',
  'agx_apple9_mesh_encoding.c',
  'agx_apple9_ir.c',
  'agx_apple9_opt.c',
  'agx_apple9_ra.c',
  'agx_compile_apple9.c',
  'agx_nir_lower_apple9_mesh.c',
  'agx_nir_apple9_mesh_native.c',
  'agx_nir_lower_apple9_texture.c',
  'agx_nir_lower_apple9_math.c',
  'agx_nir_lower_apple9_preserve.c',
  'agx_dce.c',
  'agx_liveness.c',
  'agx_insert_waits.c',
  'agx_nir_lower_address.c',
  'agx_nir_lower_cull_distance.c',
  'agx_nir_lower_fminmax.c',
  'agx_nir_lower_frag_sidefx.c',
  'agx_nir_lower_sample_mask.c',
  '../lib/agx_nir_lower_msaa.c',
  '../lib/agx_nir_lower_sample_intrinsics.c',
  'agx_nir_lower_discard_zs_emit.c',
  'agx_nir_lower_interpolation.c',
  'agx_nir_lower_shared_bitsize.c',
  'agx_nir_lower_subgroups.c',
  'agx_nir_lower_texture.c',
  'agx_nir_opt_preamble.c',
  'agx_lower_64bit.c',
  'agx_lower_divergent_shuffle.c',
  'agx_lower_parallel_copy.c',
  'agx_lower_pseudo.c',
  'agx_lower_spill.c',
  'agx_lower_uniform_sources.c',
  'agx_pack.c',
  'agx_performance.c',
  'agx_print.c',
  'agx_pressure_schedule.c',
  'agx_ir.c',
  'agx_opt_cse.c',
  'agx_opt_break_if.c',
  'agx_opt_empty_else.c',
  'agx_opt_jmp_none.c',
  'agx_opt_compact_constants.c',
  'agx_opt_promote_constants.c',
  'agx_opt_register_cache.c',
  'agx_optimizer.c',
  'agx_repair_ssa.c',
  'agx_reindex_ssa.c',
  'agx_spill.c',
  'agx_spill_coalesce.c',
  'agx_register_allocate.c',
  'agx_validate.c',
  'agx_validate_ra.c',
)

agx_nir_algebraic_c = custom_target(
  input : 'agx_nir_algebraic.py',
  output : 'agx_nir_algebraic.c',
  command : [
    prog_python, '@INPUT@', '-p', dir_compiler_nir,
  ],
  capture : true,
  depend_files : nir_algebraic_depends,
)

agx_opcodes_h = custom_target(
  input : ['agx_opcodes.h.py'],
  output : 'agx_opcodes.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

idep_agx_opcodes_h = declare_dependency(
  sources : [agx_opcodes_h],
  include_directories : include_directories('.'),
)

agx_opcodes_c = custom_target(
  input : ['agx_opcodes.c.py'],
  output : 'agx_opcodes.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

agx_builder_h = custom_target(
  input : ['agx_builder.h.py'],
  output : 'agx_builder.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

idep_agx_builder_h = declare_dependency(
  sources : [agx_builder_h],
  include_directories : include_directories('.'),
)

libasahi_compiler = static_library(
  'asahi_compiler',
  [libasahi_agx_files, agx_opcodes_c, agx_nir_algebraic_c],
  dependencies: [idep_nir, idep_agx_opcodes_h, idep_agx_builder_h, idep_mesautil, idep_libagx, idep_libpoly],
  link_with : [libagx2_disasm],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests or with_tools.contains('drm-shim')
  executable(
    'agx_apple9_tiny_tool',
    files('test/apple9-tiny-tool.c'),
    c_args : [no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  executable(
    'agx_apple9_fragment_replay',
    files('test/apple9-fragment-replay.c'),
    c_args : [no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )

endif

if with_tests
  apple9_mesh_limits_test = executable('apple9-mesh-limits',
    'test/apple9-mesh-limits.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  test('apple9-mesh-limits', apple9_mesh_limits_test, suite : ['asahi'])
  apple9_native_shared_test = executable('apple9-native-shared',
    'test/apple9-native-shared.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  test('apple9-native-shared', prog_python,
    args : [files('test/check-apple9-native-shared.py'), apple9_native_shared_test],
    suite : ['asahi'])
  apple9_native_workgroup_test = executable('apple9-native-workgroup',
    'test/apple9-native-workgroup.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  test('apple9-native-workgroup', apple9_native_workgroup_test, suite : ['asahi'])
  apple9_native_object_test = executable('apple9-native-object',
    'test/apple9-native-object.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  test('apple9-native-object', apple9_native_object_test, suite : ['asahi'])
  apple9_limits_workgroup_test = executable('apple9-limits-workgroup',
    'test/apple9-limits-workgroup.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  test('apple9-limits-workgroup', apple9_limits_workgroup_test, suite : ['asahi'])
  apple9_native_primitive_test = executable('apple9-native-primitive',
    'test/apple9-native-primitive.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  test('apple9-native-primitive', apple9_native_primitive_test, suite : ['asahi'])
  apple9_native_mesh_test = executable('apple9-native-mesh',
    'test/apple9-native-mesh.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  executable('apple9-native-capability-probe',
    'research/native-capability-20261022/compile_native.c',
    c_args : [c_msvc_compat_args, no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_compiler],
  )
  test('apple9-native-mesh', prog_python,
    args : [files('test/check-apple9-native-mesh.py'), apple9_native_mesh_test],
    suite : ['asahi'],
  )
  test('apple9-native-mesh-counts', apple9_native_mesh_test,
    args : ['--counts'],
    suite : ['asahi'],
  )
  test('apple9-native-mesh-groups', prog_python,
    args : [files('test/check-apple9-mesh-groups.py'), apple9_native_mesh_test],
    suite : ['asahi'],
  )
  test(
    'agx_tests',
    executable(
      'agx_tests',
      files(
         'test/test-compact-constants.cpp',
         'test/test-apple9.cpp',
         'test/test-apple9-mesh.cpp',
         'test/test-apple9-object.cpp',
         'test/test-apple9-opt.cpp',
         'test/test-apple9-shared-ra.cpp',
         'test/test-pressure-schedule.cpp',
         'test/test-minifloat.cpp',
         'test/test-optimizer.cpp',
         'test/test-lower-pseudo.cpp',
         'test/test-lower-parallel-copy.cpp',
         'test/test-lower-spill.cpp',
         'test/test-repair-ssa.cpp',
         'test/test-spill-coalesce.cpp',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [idep_gtest, idep_nir, idep_agx_opcodes_h, idep_agx_builder_h, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )
endif
