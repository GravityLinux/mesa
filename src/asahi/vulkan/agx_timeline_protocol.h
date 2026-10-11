/* SPDX-License-Identifier: MIT */
#ifndef AGX_TIMELINE_PROTOCOL_H
#define AGX_TIMELINE_PROTOCOL_H
#ifdef __KERNEL__
#include <linux/types.h>
typedef u64 tl_u64;
typedef u32 tl_u32;
typedef u8 tl_u8;
#else
#include <stdint.h>
typedef uint64_t tl_u64;
typedef uint32_t tl_u32;
typedef uint8_t tl_u8;
#endif
#define TL_MAGIC 0x31454d4954584741ull
#define TL_VERSION 2
#define TL_HEADER_SIZE 4096
#define TL_RECORD_SIZE 96
#define TL_DEFAULT_CAPACITY (1u << 21)
enum tl_clock { TL_MONOTONIC_NS = 1, TL_ARM_VIRTUAL_COUNTER = 2 };
enum tl_event {
 TL_CLOCK = 1,
 TL_API_BEGIN, TL_API_END, TL_FENCE_MEMBER, TL_FENCE_RESET,
 TL_FENCE_CREATE, TL_FENCE_DESTROY, TL_PRESENT_WAIT,
 TL_DRIVER_BEGIN, TL_DRIVER_END, TL_DRIVER_WAIT, TL_DRIVER_SIGNAL,
 TL_IOCTL_BEGIN, TL_IOCTL_END, TL_IOCTL_CMD, TL_GPU_INTERVAL,
 TL_HOST_IOCTL_BEGIN, TL_HOST_IOCTL_END,
 TL_KIOCTL_BEGIN, TL_KIOCTL_END, TL_KFENCE_INIT, TL_KFENCE_SIGNAL,
 TL_KJOB_QUEUE, TL_KJOB_RUN, TL_KJOB_DONE, TL_KVCPU_CLOCK,
 TL_GPU_CLOCK, TL_ERROR, TL_TRANSFER,
 /* hk: one per render Work at submit, id = its vertex timestamp key (as in
  * TL_IOCTL_CMD c): attachment identity, area, formats, draws, pass code.
  * flags: waits on earlier render (bit 0) / compute (1) work, and the same
  * from the tiling stage (2, 3). */
 TL_RENDER_INFO,
};
/* Append-only while enabled. The controller enables recording after warm-up.
 * Full buffers increment dropped rather than silently overwriting history.
 * seq is published last with release semantics; reader requires i+1.
 */
struct tl_header {
 tl_u64 magic, version, header_size, record_size, capacity, clock_domain;
 tl_u64 enabled, written, dropped, pid, frequency, errors;
 tl_u8 reserved[TL_HEADER_SIZE - 12 * 8];
};
struct tl_record {
 tl_u64 seq, stamp, id, a, b, c, d, e, f, g;
 tl_u32 pid, tid, type, flags;
};
#endif
