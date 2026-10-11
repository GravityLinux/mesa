/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

/* Recompile a draw-time Apple9 fragment program saved by hk's
 * HK_APPLE9_DUMP_FRAGMENT_DIR (<binary hash>-<size>.a9frag) and print the
 * binary's hash and size, so a capture can be reproduced exactly on the host.
 * Usage: agx_apple9_fragment_replay FILE.a9frag [OUT.bin]
 * Compiler environment (AGX_*) applies as in the capturing process. */

#include "agx_compile_apple9.h"

#include "compiler/glsl_types.h"
#include "compiler/nir/nir_serialize.h"
#include "util/blob.h"
#include "util/hash_table.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s FILE.a9frag [OUT.bin]\n", argv[0]);
      return 64;
   }
   FILE *f = fopen(argv[1], "rb");
   if (!f) {
      perror(argv[1]);
      return 1;
   }
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (size < 28 || size > (256 << 20)) {
      fprintf(stderr, "%s: bad size %ld\n", argv[1], size);
      return 1;
   }
   uint8_t *data = malloc(size);
   if (!data || fread(data, 1, size, f) != (size_t)size)
      return 1;
   fclose(f);

   uint32_t header[5];
   memcpy(header, data + 8, sizeof(header));
   const uint32_t varyings_size = header[0], blend_size = header[1];
   const uint32_t targets = header[2], blends = header[3], nir_size = header[4];
   const size_t fixed = 8 + sizeof(header);
   if (memcmp(data, "A9FRAG01", 8) ||
       varyings_size != sizeof(struct agx_apple9_varying_layout) ||
       blend_size != sizeof(struct agx_apple9_blend) || blends < 1 ||
       blends > 8 || fixed + varyings_size + blends * blend_size + nir_size !=
                         (size_t)size) {
      fprintf(stderr, "%s: not an A9FRAG01 dump of this build\n", argv[1]);
      return 1;
   }
   struct agx_apple9_varying_layout varyings;
   struct agx_apple9_blend blend[8];
   memcpy(&varyings, data + fixed, varyings_size);
   memcpy(blend, data + fixed + varyings_size, blends * blend_size);

   glsl_type_singleton_init_or_ref();
   struct blob_reader reader;
   blob_reader_init(&reader, data + fixed + varyings_size + blends * blend_size,
                    nir_size);
   nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &reader);
   if (!nir || reader.overrun) {
      fprintf(stderr, "%s: NIR deserialization failed\n", argv[1]);
      return 1;
   }

   struct agx_shader_part part = {0};
   const char *reason = NULL;
   if (!agx_compile_apple9_fragment_mrt(nir, &varyings, blend, targets, &part,
                                        &reason)) {
      fprintf(stderr, "compile failed: %s\n", reason ? reason : "?");
      return 1;
   }
   printf("hash=%08x size=%u\n",
          _mesa_hash_data(part.binary, part.info.binary_size),
          part.info.binary_size);
   if (argc > 2) {
      FILE *out = fopen(argv[2], "wb");
      if (!out || fwrite(part.binary, 1, part.info.binary_size, out) !=
                     part.info.binary_size)
         return 1;
      fclose(out);
   }
   ralloc_free(nir);
   free(data);
   glsl_type_singleton_decref();
   return 0;
}
