/* SPDX-License-Identifier: MIT */
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)
#include <stdlib.h>

static const char classes[] =
   "5550ecf263192c596c3f9d8c3b84c0324ac3d0c8ee76da37c378205247f1a3bd";
static char input[512];
static size_t input_size, offset;
static bool fail_stat, regular_fd, fail_open, fail_read, interrupt_read;
static unsigned opens, closes;

static int
test_fstat(int fd, struct stat *st)
{
   CHECK(fd == 42);
   if (fail_stat)
      return -1;
   memset(st, 0, sizeof(*st));
   st->st_mode = regular_fd ? S_IFREG : S_IFCHR;
   st->st_rdev = makedev(226, 137);
   return 0;
}

static int
test_open(const char *path, int flags, ...)
{
   CHECK(!strcmp(path, "/sys/dev/char/226:137/device/asahi_native_mesh"));
   CHECK(flags == (O_RDONLY | O_CLOEXEC));
   ++opens;
   offset = 0;
   return fail_open ? -1 : 43;
}

static ssize_t
test_read(int fd, void *data, size_t size)
{
   CHECK(fd == 43);
   if (fail_read || interrupt_read) {
      errno = interrupt_read ? EINTR : EIO;
      interrupt_read = false;
      return -1;
   }
   size_t count = input_size - offset;
   if (count > size)
      count = size;
   if (count > 17)
      count = 17; /* Exercise short reads, including the final newline. */
   memcpy(data, input + offset, count);
   offset += count;
   return count;
}

static int
test_close(int fd)
{
   CHECK(fd == 43);
   ++closes;
   return 0;
}

/* Exercise the exact fd-derived reader with deterministic filesystem errors. */
#define fstat test_fstat
#define open test_open
#define read test_read
#define close test_close
#include "../agx_native_mesh.c"
#undef fstat
#undef open
#undef read
#undef close

static struct agx_native_mesh_support
parse(const char *generation, const char *profile, const char *hash)
{
   input_size = snprintf(input, sizeof(input),
                         "asahi-native-mesh v1 generation=%s profile=%s classes=%s\n",
                         generation, profile, hash);
   return agx_native_mesh_parse(input, input_size, classes);
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--parse-stdin")) {
      char *line = NULL;
      size_t capacity = 0;
      ssize_t length;
      while ((length = getline(&line, &capacity, stdin)) >= 0) {
         struct agx_native_mesh_support s =
            agx_native_mesh_parse(line, length, classes);
         printf("%u %llu\n", s.qualified, (unsigned long long)s.generation);
      }
      free(line);
      return ferror(stdin) ? 1 : 0;
   }
   CHECK(argc == 1);
   struct agx_native_mesh_support zero = parse("0", "qualified", classes);
   CHECK(zero.qualified && zero.generation == 0);
   struct agx_native_mesh_support max =
      parse("72057594037927935", "qualified", classes);
   CHECK(max.qualified && max.generation == AGX_NATIVE_MESH_GENERATION_MAX);
   const char *bad_generations[] = {
      "", "00", "01", "-1", "+1", " 1", "1 ", "1x", "0x1",
      "72057594037927936", "9223372036854775807", "9223372036854775808",
      "18446744073709551615", "9999999999999999999999",
   };
   for (unsigned i = 0; i < sizeof(bad_generations) / sizeof(*bad_generations); ++i)
      CHECK(!parse(bad_generations[i], "qualified", classes).qualified);
   CHECK(!parse("1", "none", "-").qualified);
   CHECK(!parse("1", "research", "-").qualified);
   CHECK(!parse("1", "research", classes).qualified);
   CHECK(!parse("1", "future", classes).qualified);
   CHECK(!parse("1", "qualified", "-").qualified);
   char wrong[65];
   memcpy(wrong, classes, sizeof(wrong));
   wrong[63] = '0';
   CHECK(!parse("1", "qualified", wrong).qualified);

   parse("1", "qualified", classes);
   char good[512];
   memcpy(good, input, input_size);
   size_t good_size = input_size;
   for (size_t i = 0; i < good_size; ++i) {
      CHECK(!agx_native_mesh_parse(good, i, classes).qualified);
      memcpy(input, good, good_size);
      input[i] = '\0';
      CHECK(!agx_native_mesh_parse(input, good_size, classes).qualified);
   }
   memcpy(input, good, good_size);
   input[19] = '2'; /* v2 */
   CHECK(!agx_native_mesh_parse(input, good_size, classes).qualified);
   memcpy(input, good, good_size);
   input[good_size] = '\n';
   CHECK(!agx_native_mesh_parse(input, good_size + 1, classes).qualified);
   memset(input + good_size, 'x', sizeof(input) - good_size);
   CHECK(!agx_native_mesh_parse(input, 161, classes).qualified);
   CHECK(!agx_native_mesh_parse(good, good_size, NULL).qualified);
   CHECK(!agx_native_mesh_parse(NULL, good_size, classes).qualified);

   CHECK(agx_native_mesh_same_qualification(zero, zero));
   CHECK(agx_native_mesh_same_qualification(max, max));
   CHECK(!agx_native_mesh_same_qualification(zero, max));
   struct agx_native_mesh_support absent = {0};
   CHECK(!agx_native_mesh_same_qualification(absent, zero));
   CHECK(!agx_native_mesh_same_qualification(zero, absent));
   CHECK(!agx_native_mesh_same_qualification(absent, absent));

   parse("7", "qualified", classes);
   interrupt_read = true;
   struct agx_native_mesh_support readback = agx_native_mesh_probe(42, classes);
   CHECK(readback.qualified && readback.generation == 7);
   CHECK(opens == 1 && closes == 1);
   fail_stat = true;
   CHECK(!agx_native_mesh_probe(42, classes).qualified);
   fail_stat = false;
   regular_fd = true;
   CHECK(!agx_native_mesh_probe(42, classes).qualified);
   regular_fd = false;
   CHECK(opens == 1 && closes == 1);
   fail_open = true;
   CHECK(!agx_native_mesh_probe(42, classes).qualified);
   fail_open = false;
   CHECK(opens == 2 && closes == 1);
   fail_read = true;
   CHECK(!agx_native_mesh_probe(42, classes).qualified);
   fail_read = false;
   CHECK(opens == 3 && closes == 2);
   input_size = sizeof(input);
   memset(input, 'x', sizeof(input));
   CHECK(!agx_native_mesh_probe(42, classes).qualified);
   CHECK(offset == 161 && opens == 4 && closes == 3);
   puts("PASS: strict grammar, generation race, own-fd path, bounded reads and failures");
   return 0;
}
