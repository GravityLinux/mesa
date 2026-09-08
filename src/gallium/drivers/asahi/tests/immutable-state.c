/* SPDX-License-Identifier: MIT */
/* Standalone test: cc -std=c11 -O2 immutable-state.c -o immutable-state */
#include "../agx_immutable_state.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
   enum { SIZE = 4099, STATES = 17 };
   uint8_t data[STATES][SIZE], target[SIZE];
   struct agx_immutable_state states[STATES] = {0};
   for (unsigned i = 0; i < SIZE; i++) data[0][i] = i * 17;
   for (unsigned s = 1; s < STATES; s++) {
      memcpy(data[s], data[0], SIZE);
      /* Disjoint/adjacent/overlapping ranges and a partial last block. */
      for (unsigned i = s; i < SIZE; i += s * 39)
         data[s][i] ^= s;
      data[s][SIZE - 1] ^= s;
   }
   for (unsigned s = 0; s < STATES; s++)
      assert(agx_immutable_state_init(&states[s], data[s], data[0], SIZE));
   for (unsigned a = 0; a <= STATES; a++) for (unsigned b = 0; b < STATES; b++) {
      const struct agx_immutable_state *old = a == STATES ? NULL : &states[a];
      memset(target, 0xff, SIZE);
      if (old) memcpy(target, old->data, SIZE);
      uint32_t ai = 0, bi = 0, end = 0;
      struct agx_state_range r;
      while ((r = agx_immutable_state_next(old, &states[b], &ai, &bi)).size) {
         assert(r.offset >= end && r.offset + r.size <= SIZE);
         memcpy(target + r.offset, states[b].data + r.offset, r.size);
         end = r.offset + r.size;
      }
      assert(!memcmp(target, data[b], SIZE));
   }
   /* Exact predecessor plans also handle equal non-baseline fields. */
   for (unsigned a = 0; a < STATES; a++) {
      for (unsigned b = 0; b < STATES; b++) {
         struct agx_immutable_state transition, empty = {0};
         assert(agx_immutable_state_init(&transition, data[b], data[a], SIZE));
         memcpy(target, data[a], SIZE);
         uint32_t ai = 0, bi = 0;
         struct agx_state_range r;
         while ((r = agx_immutable_state_next(&empty, &transition, &ai, &bi)).size)
            memcpy(target + r.offset, transition.data + r.offset, r.size);
         assert(!memcmp(target, data[b], SIZE));
         free(transition.ranges);
      }
   }
   for (unsigned s = 0; s < STATES; s++) free(states[s].ranges);
   puts("PASS: 595 immutable-state transitions, including exact predecessor plans");
}
