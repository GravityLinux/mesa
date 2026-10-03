/* SPDX-License-Identifier: MIT
 * Event page-boundary, independent GPU writes, and freed-slot reuse regression.
 * Build: cc apple9-events.c -lvulkan -o apple9-events
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT 10000
#define GPU_WRITES 128
#define CHECK(call) do { VkResult r = (call); if (r != VK_SUCCESS) { \
   fprintf(stderr, "%s: %d\n", #call, r); exit(1); } } while (0)

int main(void)
{
   VkInstance instance;
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   if (!count) return 1;
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(physical, &props);
   printf("device=%s\n", props.deviceName);
   if (!strstr(props.deviceName, "Apple M4")) return 1;
   uint32_t families = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, NULL);
   VkQueueFamilyProperties *fp = calloc(families, sizeof(*fp));
   if (!fp) return 1;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, fp);
   uint32_t family;
   for (family = 0; family < families; family++)
      if (fp[family].queueFlags & VK_QUEUE_COMPUTE_BIT) break;
   free(fp);
   if (family == families) return 1;
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkEvent *events = calloc(COUNT, sizeof(*events));
   VkResult *expected = calloc(COUNT, sizeof(*expected));
   if (!events || !expected) return 1;
   VkEventCreateInfo eci = {.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
   for (unsigned i = 0; i < COUNT; i++) {
      CHECK(vkCreateEvent(device, &eci, NULL, &events[i]));
      expected[i] = VK_EVENT_RESET;
      if (!(i % 3)) {
         CHECK(vkSetEvent(device, events[i]));
         expected[i] = VK_EVENT_SET;
      }
   }
   VkCommandPool pool;
   VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family};
   CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
   VkCommandBuffer cmd;
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   for (unsigned i = 0; i < GPU_WRITES; i++) {
      unsigned slot = i < 6 ? 4093 + i : (i * 79) % COUNT;
      if (i & 1) {
         vkCmdSetEvent(cmd, events[slot], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
         expected[slot] = VK_EVENT_SET;
      } else {
         vkCmdResetEvent(cmd, events[slot], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
         expected[slot] = VK_EVENT_RESET;
      }
   }
   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd};
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   for (unsigned phase = 0; phase < 2; phase++) {
      for (unsigned i = 0; i < COUNT; i++) {
         VkResult result = vkGetEventStatus(device, events[i]);
         if (result != expected[i]) {
            fprintf(stderr, "phase=%u event=%u got=%d expected=%d\n",
                    phase, i, result, expected[i]);
            return 1;
         }
      }
      if (!phase) {
         for (unsigned i = 0; i < COUNT; i += 2)
            vkDestroyEvent(device, events[i], NULL);
         for (unsigned i = 0; i < COUNT; i += 2) {
            CHECK(vkCreateEvent(device, &eci, NULL, &events[i]));
            expected[i] = VK_EVENT_RESET;
         }
      }
   }
   vkDestroyCommandPool(device, pool, NULL);
   for (unsigned i = 0; i < COUNT; i++) vkDestroyEvent(device, events[i], NULL);
   free(events); free(expected);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   printf("PASS: %u events, %u GPU writes, %u reused slots\n",
          COUNT, GPU_WRITES, COUNT / 2);
   return 0;
}
