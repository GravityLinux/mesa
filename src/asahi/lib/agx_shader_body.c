/* SPDX-License-Identifier: MIT */
#include "agx_shader_body.h"
#include "agx_device.h"
#include "util/u_atomic.h"

void
agx_shader_body_init_bo(struct agx_bo *bo)
{
   bo->apple9_body = (struct agx_shader_body){
      .bo = bo,
      .addr = bo->va->addr,
   };
}

void
agx_shader_body_reference(struct agx_shader_body *body)
{
   if (body->destroy) {
      assert(p_atomic_read(&body->refcnt) > 0);
      p_atomic_inc(&body->refcnt);
   } else {
      agx_bo_reference(body->bo);
   }
}

void
agx_shader_body_finish(struct agx_device *dev, struct agx_shader_body *body)
{
   assert(!body->entry_pins);
   if (!body->entry_offset && !body->entry_fixed)
      return;

   simple_mtx_lock(&dev->apple9_entry_lock);
   if (body->entry_offset) {
      util_vma_heap_free(&dev->apple9_entry_heap,
                         body->entry_offset - AGX_APPLE9_ENTRY_CODE_OFFSET,
                         AGX_APPLE9_ENTRY_BLOCK_SIZE);
      body->entry_offset = 0;
      list_del(&body->entry_link);
   }
   body->entry_fixed = false;
   simple_mtx_unlock(&dev->apple9_entry_lock);
}

void
agx_shader_body_unreference(struct agx_device *dev, struct agx_shader_body *body)
{
   if (!body)
      return;
   if (!body->destroy) {
      agx_bo_unreference(dev, body->bo);
      return;
   }

   assert(p_atomic_read(&body->refcnt) > 0);
   if (p_atomic_dec_zero(&body->refcnt)) {
      agx_shader_body_finish(dev, body);
      /* Storage release may take the BO lock. Never nest it inside the entry
       * lock, which whole-BO destruction takes in the opposite order. */
      body->destroy(dev, body);
   }
}
