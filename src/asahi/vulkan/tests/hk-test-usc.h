/* SPDX-License-Identifier: MIT */
#pragma once
static void
test_reset_usc_heap(struct hk_device *dev)
{
   list_for_each_entry(struct hk_descriptor_chunk, slab, &dev->usc_heap.slabs, link) {
      uint64_t addr = util_vma_heap_alloc(&slab->heap, slab->backing.size, 1);
      CHECK(addr == slab->backing.addr);
      util_vma_heap_free(&slab->heap, addr, slab->backing.size);
   }
   hk_device_finish_usc_heap(dev);
   hk_device_init_usc_heap(dev);
}
