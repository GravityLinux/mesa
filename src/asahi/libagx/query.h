/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * Copyright 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"

#pragma once

struct libagx_xfb_counter_copy {
   DEVICE(uint32_t) dest[4];
   DEVICE(uint32_t) src[4];
};

struct libagx_imm_write {
   DEVICE(uint32_t) address;
   uint32_t value;
   /* OpenCL pointers are 8-byte aligned. Keep the same array stride on
    * 32-bit hosts, where uint64_t may only be 4-byte aligned.
    */
   uint32_t _pad;
};
static_assert(sizeof(struct libagx_imm_write) == 16, "query write array stride");

/* Adds one render's occlusion segment counter into a query report. */
struct libagx_occlusion_accum {
   DEVICE(uint64_t) src;
   DEVICE(uint64_t) dst;
};
static_assert(sizeof(struct libagx_occlusion_accum) == 16, "occlusion accumulate stride");

#define LIBAGX_QUERY_UNAVAILABLE (uint64_t)((int64_t)-1)
