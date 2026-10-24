/* SPDX-License-Identifier: MIT */
#ifndef HK_GPU_BUSY_H
#define HK_GPU_BUSY_H

struct hk_queue;
struct drm_asahi_cmd_render;
struct drm_asahi_cmd_compute;

#include <stdint.h>
struct agx_bo;
#ifdef HK_GPU_BUSY_COUNTERS
void hk_gpu_busy_set_pass(struct hk_queue *queue, uint64_t code,
                          const uint32_t grid[3], uint32_t count);
void hk_gpu_busy_shader(uint64_t address, const void *code, uint32_t size,
                        unsigned stage);
/* GPU_STORAGE_FORMAT: once per storage-image view format: its VkFormat and
 * the G16 store encoder class it selects. */
void hk_gpu_busy_storage_format(unsigned vk_format, unsigned store_encode);
void hk_gpu_busy_shader_source(const void *code, uint32_t size, const char *name,
                               const uint8_t *blake3, unsigned stage);
void hk_gpu_busy_init(struct hk_queue *queue);
void hk_gpu_busy_finish(struct hk_queue *queue);
void hk_gpu_busy_sample(struct hk_queue *queue);
void hk_gpu_busy_render(struct hk_queue *queue,
                        struct drm_asahi_cmd_render *cmd);
void hk_gpu_busy_compute(struct hk_queue *queue,
                         struct drm_asahi_cmd_compute *cmd);
#else
#define hk_gpu_busy_set_pass(...) ((void)0)
#define hk_gpu_busy_shader(...)  ((void)0)
#define hk_gpu_busy_shader_source(...) ((void)0)
#define hk_gpu_busy_storage_format(...) ((void)0)
#define hk_gpu_busy_init(...)    ((void)0)
#define hk_gpu_busy_finish(...)  ((void)0)
#define hk_gpu_busy_sample(...)  ((void)0)
#define hk_gpu_busy_render(...)  ((void)0)
#define hk_gpu_busy_compute(...) ((void)0)
#endif
#endif
