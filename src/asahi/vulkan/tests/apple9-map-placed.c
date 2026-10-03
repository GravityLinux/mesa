/* SPDX-License-Identifier: MIT
 * Placed memory maps of allocations whose BO came from the BO cache. A freed
 * allocation leaves its BO, still mapped, in the cache; a smaller allocation
 * may reuse it. Its placed mapping must cover exactly the allocation (the
 * pages after it stay the caller's) and must land at the requested address.
 * Checks VK_EXT_map_memory_placed with and without VK_MEMORY_UNMAP_RESERVE_BIT_EXT,
 * and freeing memory while it is mapped.
 */
#define _GNU_SOURCE
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      VkResult r = (x);                                                        \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d (line %d)\n", #x, r, __LINE__);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static VkDevice dev;
static uint32_t host_type;
static PFN_vkMapMemory2KHR map2;
static PFN_vkUnmapMemory2KHR unmap2;

static VkDeviceMemory
allocate(VkDeviceSize size, bool map_and_leak)
{
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = size,
                              .memoryTypeIndex = host_type};
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
   if (map_and_leak) {
      /* Write through an ordinary map and free while mapped. */
      void *p;
      CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &p));
      memset(p, 0x5a, size);
   }
   return mem;
}

/* Reserve guard | allocation | guard of the caller's own pages, fill them,
 * map the allocation placed in the middle and check every guard byte. */
static bool
check_placed(VkDeviceSize size, bool reserve, const char *what)
{
   const size_t page = sysconf(_SC_PAGESIZE);
   const size_t total = page + size + page;
   uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (base == MAP_FAILED) {
      perror("mmap");
      exit(1);
   }
   memset(base, 0xab, total);

   VkDeviceMemory mem = allocate(size, false);
   VkMemoryMapPlacedInfoEXT placed = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT,
      .pPlacedAddress = base + page};
   VkMemoryMapInfoKHR mi = {.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR,
                            .pNext = &placed,
                            .flags = VK_MEMORY_MAP_PLACED_BIT_EXT,
                            .memory = mem,
                            .size = VK_WHOLE_SIZE};
   void *p = NULL;
   CHECK(map2(dev, &mi, &p));
   bool ok = true;
   if (p != base + page) {
      printf("%s: mapped at %p, requested %p\n", what, p, (void *)(base + page));
      ok = false;
   } else {
      memset(p, 0xcd, size);
   }
   VkMemoryUnmapInfoKHR ui = {.sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR,
                              .flags = reserve ? VK_MEMORY_UNMAP_RESERVE_BIT_EXT : 0,
                              .memory = mem};
   CHECK(unmap2(dev, &ui));

   /* Guard pages must still be the caller's, readable and unchanged. A
    * mapping that overran them leaves PROT_NONE (reserve) or nothing, which
    * faults here, or the device memory's contents. */
   for (size_t i = 0; i < page && ok; ++i) {
      if (base[i] != 0xab || base[page + size + i] != 0xab) {
         printf("%s: guard byte %zu changed (before %02x, after %02x)\n", what, i,
                base[i], base[page + size + i]);
         ok = false;
      }
   }
   vkFreeMemory(dev, mem, NULL);
   munmap(base, total);
   printf("%s: %s\n", what, ok ? "ok" : "FAIL");
   return ok;
}

int
main(void)
{
   const char *exts[] = {VK_KHR_MAP_MEMORY_2_EXTENSION_NAME,
                         VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME};
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t count = 1;
   VkPhysicalDevice pdev;
   vkEnumeratePhysicalDevices(inst, &count, &pdev);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   host_type = UINT32_MAX;
   for (uint32_t t = 0; t < mp.memoryTypeCount && host_type == UINT32_MAX; ++t) {
      if (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
         host_type = t;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueCount = 1,
                                  .pQueuePriorities = &prio};
   VkPhysicalDeviceMapMemoryPlacedFeaturesEXT placed_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT,
      .memoryMapPlaced = VK_TRUE,
      .memoryUnmapReserve = VK_TRUE};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &placed_features,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .enabledExtensionCount = 2,
                             .ppEnabledExtensionNames = exts};
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   map2 = (PFN_vkMapMemory2KHR)vkGetDeviceProcAddr(dev, "vkMapMemory2KHR");
   unmap2 = (PFN_vkUnmapMemory2KHR)vkGetDeviceProcAddr(dev, "vkUnmapMemory2KHR");

   const VkDeviceSize page = sysconf(_SC_PAGESIZE);
   bool ok = true;

   /* Fresh BOs. */
   ok &= check_placed(4 * page, false, "fresh");
   ok &= check_placed(4 * page, true, "fresh reserve");

   /* A larger BO freed while mapped, then reused for a smaller allocation
    * (the cache returns BOs up to twice the requested size from the same
    * power-of-two bucket). */
   for (unsigned reserve = 0; reserve < 2; ++reserve) {
      for (unsigned shrink = 1; shrink <= 3; ++shrink) {
         VkDeviceSize big = 15 * page;
         vkFreeMemory(dev, allocate(big, true), NULL);
         char what[64];
         snprintf(what, sizeof(what), "cached %u/15%s", 15 - shrink,
                  reserve ? " reserve" : "");
         ok &= check_placed(big - shrink * page, reserve, what);
      }
   }

   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("%s\n", ok ? "PASS" : "FAIL");
   return ok ? 0 : 1;
}
