/* Copyright 2026 Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_memory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <map>
#include <random>
#include <vector>

namespace {

/* Fake buffers: page-aligned, nonzero addresses, mapped by host memory. */
struct FakeDevice {
   uint64_t next_addr = 1ull << 32;
   std::map<uint64_t, std::vector<uint8_t>> live;
   unsigned creates = 0;
   bool fail = false;

   static bool create(void *data, uint64_t size, uint64_t reserve,
                      hk_descriptor_backing *out)
   {
      auto *dev = static_cast<FakeDevice *>(data);
      if (dev->fail)
         return false;
      uint64_t pages = (size + 0x3fff) & ~0x3fffull;
      uint64_t addr = dev->next_addr;
      dev->next_addr += pages + 0x4000;
      auto &mem = dev->live[addr];
      mem.resize(pages);
      *out = {&mem, addr, mem.data(), pages};
      dev->creates++;
      return true;
   }

   static void destroy(void *data, hk_descriptor_backing *backing)
   {
      auto *dev = static_cast<FakeDevice *>(data);
      ASSERT_EQ(dev->live.count(backing->addr), 1u);
      dev->live.erase(backing->addr);
   }
};

const hk_descriptor_memory_ops fake_ops = {FakeDevice::create,
                                           FakeDevice::destroy};

} // namespace

TEST(HkDescriptorMemory, SmallPoolIsOneBufferUpFront)
{
   FakeDevice dev;
   hk_descriptor_memory mem;
   ASSERT_EQ(hk_descriptor_memory_init(&mem, &fake_ops, &dev, 1 << 20),
             HK_DESCRIPTOR_MEMORY_OK);
   EXPECT_FALSE(mem.lazy);
   EXPECT_EQ(dev.live.size(), 1u);

   hk_descriptor_allocation a, b;
   ASSERT_EQ(hk_descriptor_memory_alloc(&mem, 512 << 10, 64, 0, &a),
             HK_DESCRIPTOR_MEMORY_OK);
   ASSERT_EQ(hk_descriptor_memory_alloc(&mem, 512 << 10, 64, 0, &b),
             HK_DESCRIPTOR_MEMORY_OK);
   hk_descriptor_allocation c;
   EXPECT_EQ(hk_descriptor_memory_alloc(&mem, 64, 64, 0, &c),
             HK_DESCRIPTOR_MEMORY_OUT_OF_POOL);
   EXPECT_EQ(dev.live.size(), 1u);
   hk_descriptor_memory_free(&mem, &a);
   EXPECT_EQ(hk_descriptor_memory_alloc(&mem, 64, 64, 0, &c),
             HK_DESCRIPTOR_MEMORY_OK);
   hk_descriptor_memory_finish(&mem);
   EXPECT_TRUE(dev.live.empty());
}

TEST(HkDescriptorMemory, LargePoolBacksSetsOnDemand)
{
   FakeDevice dev;
   hk_descriptor_memory mem;
   /* vkd3d-proton's heap pool: 2M mutable descriptors at 256 B worst case. */
   ASSERT_EQ(hk_descriptor_memory_init(&mem, &fake_ops, &dev, 512ull << 20),
             HK_DESCRIPTOR_MEMORY_OK);
   EXPECT_TRUE(mem.lazy);
   EXPECT_TRUE(dev.live.empty());

   /* Its sets use the layout's 128 B stride: half the worst case. */
   hk_descriptor_allocation heap;
   ASSERT_EQ(hk_descriptor_memory_alloc(&mem, 128ull << 20, 64, 0, &heap),
             HK_DESCRIPTOR_MEMORY_OK);
   EXPECT_NE(heap.dedicated.bo, nullptr);
   EXPECT_EQ(heap.dedicated.size, 128ull << 20);

   hk_descriptor_allocation small;
   ASSERT_EQ(hk_descriptor_memory_alloc(&mem, 4096, 64, 0, &small),
             HK_DESCRIPTOR_MEMORY_OK);
   EXPECT_EQ(small.dedicated.bo, nullptr);
   EXPECT_EQ(dev.live.size(), 2u);
   small.map[0] = 1;
   small.map[4095] = 1;

   hk_descriptor_memory_free(&mem, &heap);
   EXPECT_EQ(dev.live.size(), 1u);
   hk_descriptor_memory_free(&mem, &small);
   EXPECT_EQ(mem.used, 0u);
   hk_descriptor_memory_finish(&mem);
   EXPECT_TRUE(dev.live.empty());
}

TEST(HkDescriptorMemory, LargePoolKeepsTheWorstCaseBudget)
{
   FakeDevice dev;
   hk_descriptor_memory mem;
   const uint64_t budget = 20ull << 20;
   ASSERT_EQ(hk_descriptor_memory_init(&mem, &fake_ops, &dev, budget),
             HK_DESCRIPTOR_MEMORY_OK);
   hk_descriptor_allocation a, b, c;
   ASSERT_EQ(hk_descriptor_memory_alloc(&mem, 16ull << 20, 64, 0, &a),
             HK_DESCRIPTOR_MEMORY_OK);
   ASSERT_EQ(hk_descriptor_memory_alloc(&mem, 4ull << 20, 64, 0, &b),
             HK_DESCRIPTOR_MEMORY_OK);
   EXPECT_EQ(hk_descriptor_memory_alloc(&mem, 64, 64, 0, &c),
             HK_DESCRIPTOR_MEMORY_OUT_OF_POOL);
   hk_descriptor_memory_free(&mem, &b);
   EXPECT_EQ(hk_descriptor_memory_alloc(&mem, 64, 64, 0, &c),
             HK_DESCRIPTOR_MEMORY_OK);

   dev.fail = true;
   hk_descriptor_memory_free(&mem, &a);
   EXPECT_EQ(hk_descriptor_memory_alloc(&mem, 2ull << 20, 64, 0, &a),
             HK_DESCRIPTOR_MEMORY_OUT_OF_DEVICE);
   EXPECT_EQ(mem.used, 64u);
   hk_descriptor_memory_free(&mem, &c);
   hk_descriptor_memory_finish(&mem);
   EXPECT_TRUE(dev.live.empty());
}

/* Many free/reallocate rounds with mixed sizes, as pools see across resets:
 * no leaked buffers, a bounded chunk count, and every chunk empty at the end.
 */
TEST(HkDescriptorMemory, ChurnDoesNotFragmentOrLeak)
{
   FakeDevice dev;
   hk_descriptor_memory mem;
   ASSERT_EQ(hk_descriptor_memory_init(&mem, &fake_ops, &dev, 256ull << 20),
             HK_DESCRIPTOR_MEMORY_OK);
   std::mt19937 rng(1234);
   const uint64_t sizes[] = {64, 192, 1024, 4096, 64 << 10, 900 << 10,
                             1 << 20, 3 << 20};
   std::vector<hk_descriptor_allocation> sets;
   unsigned max_chunks = 0;
   for (unsigned round = 0; round < 200; round++) {
      /* Allocate up to ~24 MiB of mixed sets, then free a random half, and
       * every tenth round everything (a pool reset). */
      uint64_t live = 0;
      for (auto &s : sets)
         live += s.size;
      while (live < (24ull << 20)) {
         uint64_t size = sizes[rng() % std::size(sizes)];
         hk_descriptor_allocation a;
         ASSERT_EQ(hk_descriptor_memory_alloc(&mem, size, 64, 0, &a),
                   HK_DESCRIPTOR_MEMORY_OK);
         ASSERT_EQ(a.addr % 64, 0u);
         sets.push_back(a);
         live += size;
      }
      unsigned chunks = 0;
      list_for_each_entry(hk_descriptor_chunk, chunk, &mem.chunks, link)
         chunks++;
      max_chunks = std::max(max_chunks, chunks);

      std::shuffle(sets.begin(), sets.end(), rng);
      size_t keep = round % 10 == 9 ? 0 : sets.size() / 2;
      while (sets.size() > keep) {
         hk_descriptor_memory_free(&mem, &sets.back());
         sets.pop_back();
      }
   }
   for (auto &s : sets)
      hk_descriptor_memory_free(&mem, &s);
   sets.clear();

   EXPECT_EQ(mem.used, 0u);
   /* Small sets live below ~12 MiB at any time; with first fit their chunks
    * stay bounded rather than growing with the number of rounds. */
   EXPECT_LE(max_chunks, 8u);
   /* Only chunks remain, and each is completely free again. */
   unsigned chunks = 0;
   list_for_each_entry(hk_descriptor_chunk, chunk, &mem.chunks, link) {
      chunks++;
      uint64_t addr = util_vma_heap_alloc(&chunk->heap,
                                          chunk->backing.size, 64);
      EXPECT_EQ(addr, chunk->backing.addr);
      if (addr)
         util_vma_heap_free(&chunk->heap, addr, chunk->backing.size);
   }
   EXPECT_EQ(dev.live.size(), chunks);
   hk_descriptor_memory_finish(&mem);
   EXPECT_TRUE(dev.live.empty());
}

#include "hk_usc_heap.h"

TEST(HkUscHeap, StableSharedSlabsAndReuse)
{
   FakeDevice dev;
   hk_usc_heap table;
   hk_usc_heap_init(&table, &fake_ops, &dev);
   EXPECT_EQ(dev.creates, 0u);
   std::vector<hk_descriptor_allocation> views(2048);
   for (auto &view : views)
      ASSERT_TRUE(hk_usc_heap_alloc(&table, 1 * 32, 32, &view));
   EXPECT_EQ(dev.creates, 1u); /* 2048 views share one 64 KiB BO. */
   uint64_t first = views[0].addr;
   views[0].map[0] = 0x5a;
   hk_descriptor_allocation array;
   ASSERT_TRUE(hk_usc_heap_alloc(&table, 2048 * 32, 32, &array));
   EXPECT_EQ(dev.creates, 2u);
   EXPECT_EQ(views[0].addr, first); /* Growth never relocates live entries. */
   EXPECT_EQ(views[0].map[0], 0x5a);
   for (auto &view : views)
      hk_usc_heap_free(&table, &view);
   hk_descriptor_allocation reused;
   ASSERT_TRUE(hk_usc_heap_alloc(&table, 2048 * 32, 32, &reused));
   EXPECT_EQ(dev.creates, 2u); /* Freed view slots coalesce and are reused. */
   EXPECT_EQ(reused.addr + reused.size, first + 32);
   hk_usc_heap_free(&table, &array);
   hk_usc_heap_free(&table, &reused);
   hk_usc_heap_finish(&table);
   EXPECT_TRUE(dev.live.empty());
}

TEST(HkUscHeap, AllocationFailureKeepsExistingEntries)
{
   FakeDevice dev;
   hk_usc_heap table;
   hk_usc_heap_init(&table, &fake_ops, &dev);
   hk_descriptor_allocation a{}, failed{};
   EXPECT_FALSE(hk_usc_heap_alloc(&table, 0 * 32, 32, &failed));
   ASSERT_TRUE(hk_usc_heap_alloc(&table, 2048 * 32, 32, &a));
   a.map[0] = 0x6b;
   dev.fail = true;
   EXPECT_FALSE(hk_usc_heap_alloc(&table, 1 * 32, 32, &failed));
   EXPECT_EQ(failed.size, 0u);
   EXPECT_EQ(a.map[0], 0x6b);
   hk_usc_heap_free(&table, &a);
   ASSERT_TRUE(hk_usc_heap_alloc(&table, 12 * 32, 32, &failed));
   EXPECT_EQ(dev.creates, 1u); /* Reuse needs no new BO even under OOM. */
   hk_usc_heap_free(&table, &failed);
   hk_usc_heap_finish(&table);
   EXPECT_TRUE(dev.live.empty());
}

TEST(HkUscHeap, MixedObjectSizesAlignmentAndLifetime)
{
   FakeDevice dev;
   hk_usc_heap heap;
   hk_usc_heap_init(&heap, &fake_ops, &dev);
   hk_descriptor_allocation view{}, smooth{}, flat{};
   ASSERT_TRUE(hk_usc_heap_alloc(&heap, 32, 32, &view));
   ASSERT_TRUE(hk_usc_heap_alloc(&heap, 704, 64, &smooth));
   ASSERT_TRUE(hk_usc_heap_alloc(&heap, 3 * 704, 64, &flat));
   EXPECT_EQ(dev.creates, 1u);
   EXPECT_EQ(smooth.addr % 64, 0u);
   EXPECT_EQ(flat.addr % 64, 0u);
   memset(view.map, 0x11, view.size);
   memset(smooth.map, 0x22, smooth.size);
   memset(flat.map, 0x33, flat.size);
   uint64_t address = smooth.addr;
   hk_usc_heap_free(&heap, &smooth);
   ASSERT_TRUE(hk_usc_heap_alloc(&heap, 704, 64, &smooth));
   EXPECT_EQ(smooth.addr, address);
   memset(smooth.map, 0x44, smooth.size);
   for (unsigned i = 0; i < view.size; ++i)
      EXPECT_EQ(view.map[i], 0x11);
   for (unsigned i = 0; i < flat.size; ++i)
      EXPECT_EQ(flat.map[i], 0x33);
   hk_usc_heap_free(&heap, &flat);
   hk_usc_heap_free(&heap, &view);
   hk_usc_heap_free(&heap, &smooth);
   hk_usc_heap_finish(&heap);
   EXPECT_TRUE(dev.live.empty());
}
