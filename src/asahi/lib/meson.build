# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

libasahi_lib_files = files(
  'agx_apple9.c',
  'agx_apple9_geometry.c',
  'agx_apple9_launch.c',
  'agx_apple9_mesh.c',
  'agx_apple9_libagx.c',
  'agx_bo.c',
  'agx_border.c',
  'agx_device.c',
  'agx_device_virtio.c',
  'agx_linker.c',
  'agx_bg_eot.c',
  'agx_tilebuffer.c',
  'agx_nir_lower_tilebuffer.c',
  'agx_nir_apple9_tilebuffer.c',
  'agx_nir_lower_uvs.c',
  'agx_nir_lower_vbo.c',
  'agx_nir_prolog_epilog.c',
  'agx_ppp.h',
  'agx_scratch.c',
  'agx_va.c',
  'pool.c',
)

libasahi_decode_files = files(
  'decode.c',
)

libasahi_decode = static_library(
  'asahi_decode',
  [libasahi_decode_files, agx_pack],
  dependencies : [dep_valgrind, idep_mesautil],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  link_with : [libagx2_disasm],
  build_by_default : false,
)

libasahi_decode_shared = shared_library(
  'asahi_decode',
  [libasahi_decode_files, agx_pack],
  dependencies : [dep_valgrind, idep_mesautil],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  build_by_default : with_tools.contains('asahi'),
  link_with : [libagx2_disasm],
)

libagx_shaders = custom_target(
  input : libagx_spv,
  output : ['libagx_shaders.h', 'libagx_shaders.c'],
  command : [prog_asahi_clc, libagx_spv, '@OUTPUT0@', '@OUTPUT1@'],
)

idep_libagx_shaders_h = declare_dependency(
  sources : [libagx_shaders],
  include_directories : include_directories('.'),
)

libasahi_lib = static_library(
  'asahi_lib',
  [libasahi_lib_files, libagx_shaders, agx_pack, sha1_h],
  include_directories : [inc_gallium, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  link_with: [libasahi_decode, libvdrm, libpoly_nir],
  dependencies: [idep_asahi_debug, dep_libdrm, dep_valgrind, idep_nir, idep_mesautil, idep_libagx, idep_libpoly],
  build_by_default : false,
)

if with_tests
  test('apple9-draw-tables', executable('apple9-draw-tables',
    'tests/test-apple9-draw-tables.c',
    include_directories : [inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [idep_mesautil, idep_nir, idep_libagx],
    link_with : [libasahi_lib, libasahi_compiler]), suite : ['asahi'])
  test('bo-reaper', executable('bo-reaper', 'tests/test-bo-reaper.c',
    include_directories : [inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [idep_mesautil, idep_nir, idep_libagx],
    link_with : [libasahi_lib, libasahi_compiler],
    link_args : ['-Wl,--wrap=drmIoctl', '-Wl,--wrap=drmPrimeFDToHandle',
                 '-Wl,--wrap=u_thread_create', '-Wl,--wrap=pthread_cond_signal',
                 '-Wl,--wrap=agx_va_free']),
    suite : ['asahi'], timeout : 40)
  test('apple9-coefficients', executable('apple9-coefficients',
    'tests/test-apple9-coefficients.c',
    include_directories : [inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [idep_mesautil, idep_nir, idep_libagx],
    link_with : [libasahi_lib, libasahi_compiler]), suite : ['asahi'])
  test('apple9-graphics-roots', executable('apple9-graphics-roots',
    'tests/test-apple9-graphics-roots.c',
    include_directories : [inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [idep_mesautil, idep_nir, idep_libagx],
    link_with : [libasahi_lib, libasahi_compiler]), suite : ['asahi'])
  test(
    'libasahi_tests',
    executable(
      'libasahi_tests',
      files(
        'tests/test-packing.cpp',
        'tests/test-border.cpp',
        'tests/test-apple9-cdm.cpp',
        'tests/test-apple9-launch-template.cpp',
        'tests/test-tilebuffer.cpp',
        'tests/test-primitive-count.cpp',
        'tests/test-poly-indirect-size.cpp',
        'tests/poly-indirect-size.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [idep_gtest, idep_agx_pack, idep_mesautil, idep_nir],
      link_with : [libasahi_lib, libasahi_layout, libasahi_compiler],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )

  test(
    'libagx_apple9',
    executable(
      'libagx_apple9_test',
      files('tests/test-libagx-apple9.cpp',
            'tests/libagx-apple9-programs.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_asahi],
      dependencies: [idep_gtest, idep_nir, idep_mesautil, idep_libagx_shaders_h],
      link_with : [libasahi_lib, libasahi_compiler],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )
endif

if dep_iokit.found()
  libasahi_wrap = shared_library(
    'wrap',
    'wrap.c',
    include_directories : [inc_src, inc_asahi],
    dependencies : [idep_mesautil, dep_iokit],
    c_args : [no_override_init_args, '-Wno-missing-prototypes'],
    gnu_symbol_visibility : 'hidden',
    build_by_default : with_tools.contains('asahi'),
    link_with: libasahi_decode,
  )
endif
