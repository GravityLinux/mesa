/* SPDX-License-Identifier: MIT */
#ifndef HK_GPU_BUSY_H
#define HK_GPU_BUSY_H

struct hk_queue;
struct drm_asahi_cmd_render;
struct drm_asahi_cmd_compute;

#ifdef HK_GPU_BUSY_COUNTERS
void hk_gpu_busy_init(struct hk_queue *queue);
void hk_gpu_busy_finish(struct hk_queue *queue);
void hk_gpu_busy_sample(struct hk_queue *queue);
void hk_gpu_busy_render(struct hk_queue *queue,
                        struct drm_asahi_cmd_render *cmd);
void hk_gpu_busy_compute(struct hk_queue *queue,
                         struct drm_asahi_cmd_compute *cmd);
#else
#define hk_gpu_busy_init(...)    ((void)0)
#define hk_gpu_busy_finish(...)  ((void)0)
#define hk_gpu_busy_sample(...)  ((void)0)
#define hk_gpu_busy_render(...)  ((void)0)
#define hk_gpu_busy_compute(...) ((void)0)
#endif
#endif
