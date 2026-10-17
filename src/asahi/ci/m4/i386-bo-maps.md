# 32-bit GL CPU mapping lifetime regression

Build on a multilib host:

```
cc -m32 -O2 i386-bo-maps.c -o i386-bo-maps -lEGL -lGLESv2
```

Run with the i686 Asahi EGL/GLES driver on the target (native FEX on a
4K-process-enabled kernel also works). No Steam or game data is needed:

```
i386-bo-maps 256 0x564000 3072
```

The final argument reserves 3072 MiB of CPU VA in 64 MiB PROT_NONE chunks, without
committing RAM. Each of the 256 GPU buffers is mapped, written and unmapped;
the buffers stay alive, and sampled contents are read back at the end.
Retaining each CPU map after GL unmap exhausts the remaining 32-bit VA.
Success requires all 256 iterations and sampled readback. The separate chunks tolerate normal library-layout fragmentation. Smaller
reservations may be used on unusually crowded runtime layouts.

Without a reservation, `i386-bo-maps 768` reproduces the same exhaustion on
#189 with Mesa eb9a23153cd. Keep physical memory available for the BOs even
though only one word per buffer is written. Do not run alongside a game.
