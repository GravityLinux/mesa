/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"

#pragma once

/* GPU state of a mesh shader draw, run as task and mesh compute kernels
 * followed by a draw of the mesh workgroups' output records.
 */
struct libagx_mesh_params {
   /* Output records of mesh workgroups, group_stride_B bytes each */
   DEVICE(uchar) outputs;

   /* Payloads of task workgroups, payload_stride_B bytes each */
   DEVICE(uchar) payloads;

   /* Per task workgroup, the mesh workgroup counts x, y and z it launches,
    * then the inclusive prefix sum of their products.
    */
   DEVICE(uint32_t) task_grids;

   /* Task workgroups of the draw, zero without a task shader */
   uint32_t task_count;
   uint32_t payload_stride_B;

   /* Indirect dispatches of the task and mesh kernels, in workgroups */
   uint32_t task_dispatch[3];
   uint32_t mesh_dispatch[3];

   /* Indirect unindexed draw of the output records */
   uint32_t draw[4];

   uint32_t mesh_count;
   uint32_t pad;

   /* 64-bit counters of an active query, or zero: mesh primitives generated,
    * and mesh and task shader invocations.
    */
   DEVICE(uint64_t) mesh_primitives;
   DEVICE(uint64_t) mesh_invocations;
   DEVICE(uint64_t) task_invocations;

   /* Offset in a primitive of each of the user varyings VAR0 to VAR31, which
    * fragment shaders read from the output records.
    */
   uint16_t primitive_varying_B[32];
} PACKED;
static_assert(sizeof(struct libagx_mesh_params) == 168, "packed");
