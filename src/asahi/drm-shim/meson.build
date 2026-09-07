# Copyright 2019 Broadcom
# SPDX-License-Identifier: MIT

dep_python_embed = dependency('python3-embed')

libasahi_noop_drm_shim = shared_library(
  'asahi_noop_drm_shim',
  'asahi_m1n1.c',
  'delta.c',
  include_directories: [inc_include, inc_src],
  dependencies: [dep_drm_shim, dep_valgrind, dep_python_embed],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

if with_tools.contains('drm-shim')
  executable(
    't8132_apple9_compute_runner',
    files(
      't8132_apple9_compute_runner.c',
      't8132_apple9_compute_memory.c',
      't8132_apple9_compute_geometry.c',
      't8132_apple9_compute_math.c',
    ),
    dependencies : [dependency('egl'), dependency('glesv2'), dep_m],
    install : false,
  )
endif
