/* SPDX-License-Identifier: MIT
 * Diagnostic UAPI identities. Payloads are unchanged across virtio transport.
 */
#ifndef AGX_TIMELINE_UAPI_H
#define AGX_TIMELINE_UAPI_H
#include "agx_timeline.h"
#include "drm-uapi/asahi_drm.h"
static inline uint64_t tl_timestamp_key(struct drm_asahi_timestamp t)
{ return ((uint64_t)t.handle << 32) | t.offset; }
static inline void
agx_timeline_submit(unsigned kind, uint64_t id, int fd,
                    const struct drm_asahi_submit *s, uint64_t parent,
                    const uint16_t *reasons)
{
 if (!id) return;
 const unsigned char *p=(const void *)(uintptr_t)s->cmdbuf;
 agx_timeline_emit(kind,0,id,parent,fd,s->queue_id,
  agx_timeline_hash(p,s->cmdbuf_size),s->cmdbuf_size,(uintptr_t)s,agx_timeline_frame());
 if (kind==TL_HOST_IOCTL_BEGIN) return;
 for (unsigned off=0,index=0;off<s->cmdbuf_size;index++) {
  if (s->cmdbuf_size-off<sizeof(struct drm_asahi_cmd_header)) break;
  const struct drm_asahi_cmd_header *h=(const void *)(p+off);
  unsigned bytes=sizeof(*h)+h->size;
  if(bytes>s->cmdbuf_size-off) break;
  uint64_t first=0,second=0,flags=0;
  if(h->cmd_type==DRM_ASAHI_CMD_RENDER && h->size>=sizeof(struct drm_asahi_cmd_render)) {
   const struct drm_asahi_cmd_render *c=(const void *)(h+1);
   first=tl_timestamp_key(c->ts_vtx.start);second=tl_timestamp_key(c->ts_frag.start);flags=c->flags;
  } else if(h->cmd_type==DRM_ASAHI_CMD_COMPUTE && h->size>=sizeof(struct drm_asahi_cmd_compute)) {
   const struct drm_asahi_cmd_compute *c=(const void *)(h+1);
   first=tl_timestamp_key(c->ts.start);flags=c->flags;
  }
  agx_timeline_emit(TL_IOCTL_CMD,h->cmd_type,id,s->queue_id,index,first,second,
    h->vdm_barrier | ((uint64_t)h->cdm_barrier<<32),flags,
    /* g: why the command ended (enum hk_cs_end) | meta << 8, 0 if unknown */
    reasons ? reasons[index] : 0);
  off+=bytes;
 }
}
#endif
