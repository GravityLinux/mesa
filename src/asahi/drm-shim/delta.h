/* SPDX-License-Identifier: MIT */
#ifndef ASAHI_M1N1_DELTA_H
#define ASAHI_M1N1_DELTA_H
#include <stdint.h>
#include <stddef.h>
#define ASAHI_CPU_SPANS 64
struct asahi_cpu_span { uint64_t start, end; };
struct asahi_cpu_spans { uint64_t count; struct asahi_cpu_span spans[ASAHI_CPU_SPANS]; };
__attribute__((visibility("default"))) int asahi_m1n1_equal(const void *, const void *, size_t);
__attribute__((visibility("default"))) int asahi_m1n1_encode_delta(
   const uint8_t *, const uint8_t *, uint64_t, uint64_t,
   const struct asahi_cpu_spans *, uint8_t *, uint64_t,
   uint64_t *, uint64_t *, uint64_t *, uint64_t *);
#endif
