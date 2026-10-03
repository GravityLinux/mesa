/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020-2021 Collabora, Ltd.
 * Copyright 2019 Sonny Jiang <sonnyj608@gmail.com>
 * Copyright 2019 Advanced Micro Devices, Inc.
 * Copyright 2014 Broadcom
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/format/u_formats.h"
#include "nir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Background and partial-load shaders fetch the attachment through the
 * texture unit, then pack its tile representation. EOT uses native PBE block
 * stores; both descriptors retain the allocation's full mip tree. */
struct agx_apple9_tilebuffer_key {
   enum pipe_format formats[8];
   unsigned samples, mask, count;
   bool store, layered;
   unsigned array_mask;
   unsigned clear_mask;
};

/* Reload UBO0 contains {mip level, first layer} pairs, eight bytes per target.
 * Clear colors follow at byte 64, sixteen bytes per target. clear_mask selects
 * those values instead of attachment loads.
 * Texture/image indices match render-target indices. Reload sampled views must
 * use the linear format: tile words retain encoded sRGB values.
 */
nir_shader *
agx_nir_apple9_tilebuffer(const nir_shader_compiler_options *options,
                          const struct agx_apple9_tilebuffer_key *key);

#ifdef __cplusplus
}
#endif
