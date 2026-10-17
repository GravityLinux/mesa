# 32-bit GL CPU mapping lifetime regression

Build on a multilib host (without NDEBUG; checks use assert):

```
cc -m32 -O2 -Wall -Wextra i386-bo-maps.c -o i386-bo-maps -lEGL -lGLESv2
```

Run with the i686 Asahi EGL/GLES driver on the target (native FEX on a
4K-process-enabled kernel also works). No Steam or game data is needed:

```
i386-bo-maps buffer 256 0x564000 3072
i386-bo-maps texture 150 1024 3072
```

Arguments are mode, object count, buffer bytes or texture width, and reserved
MiB. The old buffer-only syntax without a mode is also accepted. Each mode
reserves CPU VA in separate 64 MiB PROT_NONE chunks without committing RAM;
this tolerates normal library-layout fragmentation. Smaller reservations may
be needed for unusually crowded runtime layouts, and must match across the
parent/candidate comparison.

Buffer mode maps, writes and unmaps each buffer, keeping the buffers alive.
Sampled words are read back at the end. Retained BO maps exhaust 32-bit VA.

Texture mode uses glCompressedTexSubImage2D with BC1 data. On G16 Mesa decodes
BC1 to RGBA8 and uploads through the tiled CPU transfer path, including the
unmap tile copy that failed in Portal. At width 1024, 150 textures use about
675 MiB (600 MiB RGBA8 and 75 MiB compressed shadows), plus bounded transient
staging and driver state. Readback state is compiled and warmed before the VA
reservation. Every seventeenth texture is sampled by a fragment shader and
its red, green or blue output is verified with glReadPixels. Keep enough real
memory for the BOs; run separately from games with a scoped memory cap.

Success requires every upload/map and sampled readback. A GL allocation error,
NULL map, failed assertion, or process fault is a failure. Parent texture
exhaustion may fault in its void transfer-unmap callback before returning a GL
error. Preserve that log as failure evidence rather than calling it a pass.
