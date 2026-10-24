/*
    Copyright (c) Microsoft Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/
#include "poly/cl/tessellator.h"
#include "tessellator_fold.h"

KERNEL(64)
libagx_tess_isoline(constant struct poly_tess_params *p,
                    enum poly_tess_mode mode__2)
{
   uint patch = cl_global_id.x;
   poly_tess_isoline_process(p, patch, mode__2);
}

KERNEL(64)
libagx_tess_tri(constant struct poly_tess_params *p, enum poly_tess_mode mode__2)
{
   uint patch = cl_global_id.x;
   poly_tess_tri_process(p, patch, mode__2);
}

KERNEL(64)
libagx_tess_quad(constant struct poly_tess_params *p, enum poly_tess_mode mode__2)
{
   uint patch = cl_global_id.x;
   poly_tess_quad_process(p, patch, mode__2);
}

/* The draw of a folded dispatch that workgroup group belongs to: the last one
 * starting at or before it.
 */
static uint
libagx_tess_fold_find(constant struct libagx_tess_fold_draw *draws,
                      uint nr_draws, uint group)
{
   uint lo = 0, hi = nr_draws;
   while (hi - lo > 1) {
      uint mid = (lo + hi) / 2;
      if (draws[mid].first_group <= group)
         lo = mid;
      else
         hi = mid;
   }
   return lo;
}

#define TESS_FOLD_KERNEL(name, process)                                        \
   KERNEL(LIBAGX_TESS_FOLD_GROUP)                                              \
   name(constant struct libagx_tess_fold_draw *draws, uint nr_draws,           \
        enum poly_tess_mode mode__2)                                           \
   {                                                                           \
      uint d = libagx_tess_fold_find(draws, nr_draws, cl_group_id.x);          \
      uint patch = (cl_group_id.x - draws[d].first_group) *                    \
                      LIBAGX_TESS_FOLD_GROUP +                                 \
                   cl_local_id.x;                                              \
      if (patch < draws[d].nr_patches)                                         \
         process((constant struct poly_tess_params *)draws[d].p, patch,       \
                 mode__2);                                                     \
   }

TESS_FOLD_KERNEL(libagx_tess_isoline_fold, poly_tess_isoline_process)
TESS_FOLD_KERNEL(libagx_tess_tri_fold, poly_tess_tri_process)
TESS_FOLD_KERNEL(libagx_tess_quad_fold, poly_tess_quad_process)
