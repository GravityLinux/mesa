/* SPDX-License-Identifier: MIT */
#include "agx_native_mesh.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

struct agx_native_mesh_support
agx_native_mesh_parse(const char *text, size_t size, const char *classes)
{
   const struct agx_native_mesh_support no = {0};
   static const char prefix[] = "asahi-native-mesh v1 generation=";
   static const char profile[] = " profile=qualified classes=";
   if (!text || !classes || strlen(classes) != 64 ||
       size > AGX_NATIVE_MESH_ADVERTISEMENT_MAX ||
       size <= sizeof(prefix) - 1 ||
       memcmp(text, prefix, sizeof(prefix) - 1))
      return no;

   for (unsigned i = 0; i < 64; ++i) {
      if (!((classes[i] >= '0' && classes[i] <= '9') ||
            (classes[i] >= 'a' && classes[i] <= 'f')))
         return no;
   }

   size_t pos = sizeof(prefix) - 1;
   if (text[pos] < '0' || text[pos] > '9')
      return no;
   bool leading_zero = text[pos] == '0';
   uint64_t generation = 0;
   size_t digits = 0;
   while (pos < size && text[pos] >= '0' && text[pos] <= '9') {
      unsigned digit = text[pos++] - '0';
      if (generation > (AGX_NATIVE_MESH_GENERATION_MAX - digit) / 10 ||
          (leading_zero && digits))
         return no;
      generation = generation * 10 + digit;
      ++digits;
   }

   if (size - pos != sizeof(profile) - 1 + 64 + 1 ||
       memcmp(text + pos, profile, sizeof(profile) - 1) ||
       memcmp(text + pos + sizeof(profile) - 1, classes, 64) ||
       text[size - 1] != '\n')
      return no;

   return (struct agx_native_mesh_support){true, generation};
}

struct agx_native_mesh_support
agx_native_mesh_probe(int drm_fd, const char *classes)
{
   const struct agx_native_mesh_support no = {0};
   struct stat st;
   if (!classes || fstat(drm_fd, &st) || !S_ISCHR(st.st_mode))
      return no;

   char path[128];
   int n = snprintf(path, sizeof(path),
                    "/sys/dev/char/%u:%u/device/asahi_native_mesh",
                    major(st.st_rdev), minor(st.st_rdev));
   if (n < 0 || (size_t)n >= sizeof(path))
      return no;

   int fd = open(path, O_RDONLY | O_CLOEXEC);
   if (fd < 0)
      return no;

   char text[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1];
   size_t size = 0;
   bool complete = false;
   while (size < sizeof(text)) {
      ssize_t count = read(fd, text + size, sizeof(text) - size);
      if (count < 0) {
         if (errno == EINTR)
            continue;
         break;
      }
      if (count == 0) {
         complete = true;
         break;
      }
      size += count;
   }
   close(fd);
   return complete ? agx_native_mesh_parse(text, size, classes) : no;
}
