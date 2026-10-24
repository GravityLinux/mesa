/* SPDX-License-Identifier: MIT
 * asahi_native_blocks: exact grammar, fail closed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../agx_native_mesh.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static struct agx_native_blocks_support
parse(const char *s)
{
   return agx_native_blocks_parse(s, strlen(s));
}

int
main(void)
{
   struct agx_native_blocks_support q = parse(
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=4096\n");
   CHECK(q.qualified && q.generation == 7 && q.max_native == 64 && q.mixed && q.words == 4096);
   q = parse("asahi-native-blocks v1 generation=0 profile=qualified max_native=16 mixed=0 words=1024\n");
   CHECK(q.qualified && !q.generation && q.max_native == 16 && !q.mixed && q.words == 1024);
   char max[160];
   snprintf(max, sizeof(max), "asahi-native-blocks v1 generation=%llu profile=qualified max_native=2 mixed=0 words=4096\n",
            (unsigned long long)AGX_NATIVE_MESH_GENERATION_MAX);
   CHECK(parse(max).qualified && parse(max).generation == AGX_NATIVE_MESH_GENERATION_MAX);

   static const char *bad[] = {
      /* profile none is never positive, whatever it carries */
      "asahi-native-blocks v1 generation=7 profile=none max_native=1 mixed=0 words=1024\n",
      "asahi-native-blocks v1 generation=7 profile=research max_native=64 mixed=1 words=4096\n",
      "asahi-native-blocks v2 generation=7 profile=qualified max_native=64 mixed=1 words=4096\n",
      "asahi-native-mesh v1 generation=7 profile=qualified max_native=64 mixed=1 words=4096\n",
      /* framing */
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=4096",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=4096\n\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=4096 \n",
      "asahi-native-blocks v1 generation=7  profile=qualified max_native=64 mixed=1 words=4096\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=4096 x=1\n",
      "asahi-native-blocks v1 generation=7 profile=qualified mixed=1 max_native=64 words=4096\n",
      /* numbers */
      "asahi-native-blocks v1 generation=07 profile=qualified max_native=64 mixed=1 words=4096\n",
      "asahi-native-blocks v1 generation= profile=qualified max_native=64 mixed=1 words=4096\n",
      "asahi-native-blocks v1 generation=-1 profile=qualified max_native=64 mixed=1 words=4096\n",
      "asahi-native-blocks v1 generation=72057594037927936 profile=qualified max_native=64 mixed=1 words=4096\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=0 mixed=1 words=4096\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=65 mixed=1 words=4096\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=064 mixed=1 words=4096\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=2 words=4096\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=1023\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=4097\n",
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=40960\n",
      "",
   };
   for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
      if (parse(bad[i]).qualified) {
         fprintf(stderr, "accepted: %s", bad[i]);
         return 1;
      }
   }
   CHECK(!agx_native_blocks_parse(NULL, 10).qualified);
   char over[200];
   memset(over, 'x', sizeof(over));
   CHECK(!agx_native_blocks_parse(over, 161).qualified);

   /* Latching across VM_CREATE: every field must match. */
   struct agx_native_blocks_support a = parse(
      "asahi-native-blocks v1 generation=7 profile=qualified max_native=64 mixed=1 words=4096\n");
   struct agx_native_blocks_support b = a;
   CHECK(agx_native_blocks_same(a, b));
   b.generation++;
   CHECK(!agx_native_blocks_same(a, b));
   b = a; b.max_native = 16;
   CHECK(!agx_native_blocks_same(a, b));
   b = a; b.mixed = false;
   CHECK(!agx_native_blocks_same(a, b));
   b = a; b.words = 1024;
   CHECK(!agx_native_blocks_same(a, b));
   CHECK(!agx_native_blocks_same((struct agx_native_blocks_support){0},
                                 (struct agx_native_blocks_support){0}));
   printf("PASS\n");
   return 0;
}
