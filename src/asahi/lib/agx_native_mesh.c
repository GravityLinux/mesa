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

/* Read the driver-local attribute NAME of the DRM device behind drm_fd.
 * Returns the byte count, or 0 on any error or an over-long attribute. */
static size_t
read_attribute(int drm_fd, const char *name, char *text, size_t capacity)
{
   struct stat st;
   if (fstat(drm_fd, &st) || !S_ISCHR(st.st_mode))
      return 0;

   char path[128];
   int n = snprintf(path, sizeof(path), "/sys/dev/char/%u:%u/device/%s",
                    major(st.st_rdev), minor(st.st_rdev), name);
   if (n < 0 || (size_t)n >= sizeof(path))
      return 0;

   int fd = open(path, O_RDONLY | O_CLOEXEC);
   if (fd < 0)
      return 0;

   size_t size = 0;
   bool complete = false;
   while (size < capacity) {
      ssize_t count = read(fd, text + size, capacity - size);
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
   return complete ? size : 0;
}

struct agx_native_mesh_support
agx_native_mesh_probe(int drm_fd, const char *classes)
{
   const struct agx_native_mesh_support no = {0};
   if (!classes)
      return no;
   char text[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1];
   size_t size = read_attribute(drm_fd, "asahi_native_mesh", text, sizeof(text));
   return size ? agx_native_mesh_parse(text, size, classes) : no;
}

/* A decimal field without leading zeros, at most max, advancing *pos. */
static bool
parse_decimal(const char *text, size_t size, size_t *pos, uint64_t max,
              uint64_t *value)
{
   size_t at = *pos;
   if (at >= size || text[at] < '0' || text[at] > '9' ||
       (text[at] == '0' && at + 1 < size && text[at + 1] >= '0' &&
        text[at + 1] <= '9'))
      return false;
   uint64_t v = 0;
   while (at < size && text[at] >= '0' && text[at] <= '9') {
      unsigned digit = text[at++] - '0';
      if (digit > max || v > (max - digit) / 10)
         return false;
      v = v * 10 + digit;
   }
   *pos = at;
   *value = v;
   return true;
}

static bool
parse_literal(const char *text, size_t size, size_t *pos, const char *literal)
{
   size_t n = strlen(literal);
   if (size - *pos < n || memcmp(text + *pos, literal, n))
      return false;
   *pos += n;
   return true;
}

/* asahi-native-blocks v1 generation=<G> profile=<none|qualified>
 * max_native=<N> mixed=<0|1> words=<W>\n, exactly, single spaces. Any other
 * line, including none with values other than 1/0/1024, is not qualified. */
struct agx_native_blocks_support
agx_native_blocks_parse(const char *text, size_t size)
{
   const struct agx_native_blocks_support no = {0};
   if (!text || size > AGX_NATIVE_MESH_ADVERTISEMENT_MAX || !size ||
       text[size - 1] != '\n')
      return no;
   size_t pos = 0, end = size - 1;
   uint64_t generation, max_native, mixed, words;
   bool qualified;
   if (!parse_literal(text, end, &pos, "asahi-native-blocks v1 generation=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_MESH_GENERATION_MAX, &generation))
      return no;
   if (parse_literal(text, end, &pos, " profile=qualified"))
      qualified = true;
   else if (parse_literal(text, end, &pos, " profile=none"))
      qualified = false;
   else
      return no;
   if (!parse_literal(text, end, &pos, " max_native=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_BLOCKS_MAX_NATIVE, &max_native) ||
       !parse_literal(text, end, &pos, " mixed=") ||
       !parse_decimal(text, end, &pos, 1, &mixed) ||
       !parse_literal(text, end, &pos, " words=") ||
       !parse_decimal(text, end, &pos, AGX_NATIVE_BLOCKS_MAX_WORDS, &words) ||
       pos != end || !max_native || words < AGX_NATIVE_BLOCKS_MIN_WORDS)
      return no;
   if (!qualified)
      return no;
   return (struct agx_native_blocks_support){
      true, generation, max_native, mixed, words};
}

struct agx_native_blocks_support
agx_native_blocks_probe(int drm_fd)
{
   const struct agx_native_blocks_support no = {0};
   char text[AGX_NATIVE_MESH_ADVERTISEMENT_MAX + 1];
   size_t size = read_attribute(drm_fd, "asahi_native_blocks", text, sizeof(text));
   return size ? agx_native_blocks_parse(text, size) : no;
}
