# 32-bit Vulkan image CPU mapping regression

Build with multilib Vulkan headers and loader:

```
cc -m32 -O2 -Wall -Wextra -Werror i386-image-maps.c -o i386-image-maps -lvulkan
```

Run with the i686 hk ICD on native FEX on the 4K-process-enabled kernel:

```
HK_EXPERIMENTAL_APPLE9=1 VK_DRIVER_FILES=/path/to/i686-hk.json \
  exec-4k FEXInterpreter ./i386-image-maps 150 3072
```

The first argument is the number of 1024x1024 RGBA8 optimal images (1..256).
The second is CPU VA to reserve in separate 64 MiB PROT_NONE chunks, not RAM.
The program reports the achieved reservation; compare the same value on parent
and candidate. Keep the process in a 2 GiB MemoryMax scope. The default images
use about 600 MiB, plus driver state, with no game or Steam requirement.

The regression checks that binding GPU-only images does not retain a large
low-32-bit DRM CPU mapping footprint (16 MiB tolerance). It then clears every
image on the GPU and verifies a pixel from each through a single small
host-coherent vkMapMemory allocation. A binding API success alone is not a
pass: the parent can ignore failed eager CPU maps while returning VK_SUCCESS.
Its retained mapping footprint is the independent failure criterion.

It also performs 32 host upload/image-to-image/download cycles on two small
optimal images, verifies every byte, and checks that these operations do not
retain new CPU mappings. All three host-copy entrypoints and CPU tiling are
exercised with the VA reservation and large GPU images still alive. No shader
source or game data is required. Mapping errors, wrong pixels, retained maps,
or process faults are failures (exit 2); a complete pass returns 0.

The /proc/self/maps accounting filters /dev/dri mappings below 4 GiB. It is
intended for the native i386/FEX Linux path, whose map view includes host VMAs;
large FEX JIT reservations above the guest range are excluded. This test is
manual hardware coverage, not part of the device-free host unit suite.
