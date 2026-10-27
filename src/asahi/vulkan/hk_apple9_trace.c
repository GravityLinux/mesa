/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_apple9_trace.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "util/u_atomic.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

bool
hk_apple9_trace(enum hk_apple9_trace_category category)
{
   static unsigned counts[HK_APPLE9_TRACE_COUNT];
   static const char *names[] = {"render", "shader", "draw", "meta", "submit",
                                 "reject"};
   if (!getenv("AGX_APPLE9_TRACE_REJECT"))
      return false;
   /* Saturate so a long-running process cannot wrap and resume logging. */
   unsigned count = p_atomic_read(&counts[category]);
   while (count < 65) {
      unsigned old = p_atomic_cmpxchg(&counts[category], count, count + 1);
      if (old == count) {
         if (count == 64)
            fprintf(stderr, "APPLE9_TRACE_LIMIT pid=%ld category=%s limit=64\n",
                    (long)getpid(), names[category]);
         return count < 64;
      }
      count = old;
   }
   return false;
}

void
hk_apple9_trace_image(const char *role, VkImage handle)
{
   VK_FROM_HANDLE(hk_image, image, handle);
   if (!image) {
      fprintf(stderr, "APPLE9_IMAGE role=%s present=0\n", role);
      return;
   }
   fprintf(stderr,
           "APPLE9_IMAGE role=%s image=%p format=%u samples=%u extent=%ux%ux%u "
           "usage=0x%llx planes=%u tiling=%u compressed=%u sparse=%u size=%llu\n",
           role, (void *)image, image->vk.format, image->vk.samples,
           image->vk.extent.width, image->vk.extent.height, image->vk.extent.depth,
           (unsigned long long)image->vk.usage, image->plane_count,
           image->planes[0].layout.tiling, image->planes[0].layout.compressed,
           !!image->planes[0].sparse_map,
           (unsigned long long)image->planes[0].layout.size_B);
}

static void
trace_attachment(const char *aspect, unsigned slot,
                  const VkRenderingAttachmentInfo *att)
{
   if (!att)
      return;
   VK_FROM_HANDLE(hk_image_view, view, att->imageView);
   VK_FROM_HANDLE(hk_image_view, resolve, att->resolveImageView);
   fprintf(stderr,
           "APPLE9_ATTACHMENT aspect=%s slot=%u view=%p format=%u "
           "level=%u+%u layer=%u+%u layout=%u load=%u store=%u "
           "resolve=%u resolve_view=%p resolve_format=%u resolve_layout=%u\n",
           aspect, slot, (void *)view, view ? view->vk.format : 0,
           view ? view->vk.base_mip_level : 0, view ? view->vk.level_count : 0,
           view ? view->vk.base_array_layer : 0, view ? view->vk.layer_count : 0,
           att->imageLayout, att->loadOp, att->storeOp, att->resolveMode,
           (void *)resolve, resolve ? resolve->vk.format : 0,
           att->resolveImageLayout);
   if (view)
      hk_apple9_trace_image(aspect, vk_image_to_handle(view->vk.image));
   if (resolve)
      hk_apple9_trace_image("resolve", vk_image_to_handle(resolve->vk.image));
}

void
hk_apple9_trace_rendering(const void *cmd, const VkRenderingInfo *info,
                          bool resolve)
{
   static unsigned serial;
   if (!hk_apple9_trace(resolve ? HK_APPLE9_TRACE_META : HK_APPLE9_TRACE_RENDER))
      return;
   fprintf(stderr,
           "APPLE9_RENDER_SNAPSHOT pid=%ld seq=%u cmd=%p op=%s flags=0x%x views=%u "
           "layers=%u area=%d,%d+%ux%u colors=%u depth=%u stencil=%u\n",
           (long)getpid(), p_atomic_inc_return(&serial), cmd,
           resolve ? "resolve" : "begin", info->flags,
           info->viewMask, info->layerCount, info->renderArea.offset.x,
           info->renderArea.offset.y, info->renderArea.extent.width,
           info->renderArea.extent.height, info->colorAttachmentCount,
           !!info->pDepthAttachment, !!info->pStencilAttachment);
   for (unsigned i = 0; i < MIN2(info->colorAttachmentCount, 8); ++i)
      trace_attachment("color", i, &info->pColorAttachments[i]);
   trace_attachment("depth", 0, info->pDepthAttachment);
   trace_attachment("stencil", 0, info->pStencilAttachment);
}
