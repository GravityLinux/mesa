/* SPDX-License-Identifier: MIT */
#ifndef AGX_TIMELINE_H
#define AGX_TIMELINE_H
#include "agx_timeline_protocol.h"
#include <stdbool.h>
#include <stddef.h>
void agx_timeline_init(void);
void agx_timeline_error(unsigned error, uint64_t detail);
uint64_t agx_timeline_frame(void);
void agx_timeline_presented(void);
bool agx_timeline_enabled(void);
uint64_t agx_timeline_id(void);
uint64_t agx_timeline_emit(unsigned type, unsigned flags, uint64_t id,
 uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f, uint64_t g);
uint64_t agx_timeline_hash(const void *data, size_t size);
uint64_t agx_timeline_api(void);
uint64_t agx_timeline_driver(void);
uint64_t agx_timeline_set_driver(uint64_t id);
struct agx_timeline_scope { uint64_t id, parent; unsigned op; };
struct agx_timeline_scope agx_timeline_scope_begin(unsigned op, uint64_t object,
 uint64_t count, uint64_t fence, uint64_t timeout, uint64_t flags);
void agx_timeline_scope_end(struct agx_timeline_scope *scope, int result);
#endif
