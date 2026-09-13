# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

files_asahi = files(
  'driinfo_asahi.h',
  'agx_batch.c',
  'agx_apple9.c',
  'agx_apple9_entries.c',
  'agx_apple9_launch.c',
  'agx_blit.c',
  'agx_disk_cache.c',
  'agx_fence.c',
  'agx_apple9_geometry.c',
  'agx_pipe.c',
  'agx_nir_lower_sysvals.c',
  'agx_nir_lower_bindings.c',
  'agx_nir_lower_point_size.c',
  'agx_query.c',
  'agx_state.c',
  'agx_streamout.c',
  'agx_uniforms.c',
)

libasahi = static_library(
  'asahi',
  files_asahi,
  include_directories : [inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
  c_args : [c_msvc_compat_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx, idep_libagx_shaders_h],
)

driver_asahi = declare_dependency(
  compile_args : '-DGALLIUM_ASAHI',
  link_with : [libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib, libasahi_layout, libasahi_decode]
)

if with_tests
  test(
    'agx_apple9_entry_tests',
    executable(
      'agx_apple9_entry_tests',
      files('test/test-apple9-entries.cpp', 'agx_apple9_entries.c'),
      include_directories : [inc_src],
      dependencies : [idep_gtest],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )
  test(
    'agx_apple9_launcher_tests',
    executable(
      'agx_apple9_launcher_tests',
      files('test/test-apple9-launch.cpp', 'agx_apple9_launch.c'),
      include_directories : [inc_src],
      dependencies : [idep_gtest],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )
  test(
    'agx_apple9_geometry_tests',
    executable(
      'agx_apple9_geometry_tests',
      files('test/test-apple9-geometry.cpp', 'agx_apple9_geometry.c'),
      include_directories : [inc_include, inc_src],
      dependencies : [idep_gtest, idep_nir],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )
endif
