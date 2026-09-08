# Shader-loaded graphics buffer tables

Apple9 graphics shaders now load buffer addresses from an ordinary GPU table.
The existing opaque launcher supplies one root pointer per stage; the generated
main loads a 64-bit address and performs an indexed memory load through that
register pair. This removes the four-direct-argument packaging restriction.

The software table holds up to 32 live resources per stage, with UBO and vertex
buffer bindings compacted together. This is a driver capacity, not a measured
hardware maximum. Gallium's separate API caps still apply: this configuration
exposes fifteen named uniform blocks plus the ordinary-uniform buffer per stage.

Each draw allocates its own table in the batch pool and retains every referenced
resource through normal read/hazard tracking. Rebinding cannot overwrite an
older draw's table. A textured FS has three independent roots: textures, samplers,
and buffers. Compute retains its existing direct argument ABI.

## Reproduction

Reset and chainload display-enabled m1n1 before each hardware process. From the
Mesa tree, run:

```sh
T8132_BUFFER_VERTEX=1 T8132_BUFFER_VBOS=1 \
src/asahi/drm-shim/scenes/buffers/run.sh /absolute/path/to/new-output
```

This exercises **32 VS buffers and 16 FS buffers** simultaneously: fifteen named
uniform blocks plus ordinary uniforms in each stage, and sixteen distinct vertex
buffer bindings. Vertex buffers have 32-byte strides and a four-byte attribute
offset. Each frame contains sixteen scissored draws with fresh uniform values and
rotated UBO/VBO bindings, without a finish between draws. Individual bands isolate
each named uniform block; the final band averages all fifteen. Every vertex buffer
contributes to the vertex result. A small perspective gradient prevents the
linker's uniform-varying optimization from selecting currently unsupported
noperspective interpolation.

Two 256×256 attachments are checked against a CPU oracle. On T8132, 2026-09-07,
all **131,072 pixels matched exactly**. The earlier 16-VS/16-FS three-draw test
also matched exactly. Set `T8132_BUFFER_COUNT=1` through `15` to reduce named
blocks; omit `T8132_BUFFER_VBOS` to remove vertex buffers, or omit both mode
variables for a fragment-buffer-only test. The oracle uses the same environment.

## Compiler model and provenance

`DEVICE_LOAD_INDIRECT` has an element index and two explicit address-word SSA
sources. Register allocation keeps the address words adjacent. Scoreboard
legalization materializes asynchronous pointer results through the existing
handoff machinery, then recollects the pair when necessary. Address-source
release follows liveness; it is independent of index-source release.

Own-authored Metal fragment mains in `tmp/apple9-buffer-table/` isolate direct,
uniform-pointer, per-pixel-pointer, double-indirection, repeated-pointer, and
vector-load cases. Only `_agc.main` was inspected. No proprietary helper or
launcher was disassembled or reconstructed, and the existing runtime blobs are
unchanged. The register-address form sets bundle byte 12 bit 5, byte 4 selects
the low address register, and byte 9 bit 2 releases the address after its last
use. These statements describe the tested indexed load form.

Compiler tests cover 5, 16 and 32 buffers in both stages, sparse API binding 31,
and rejection beyond the supported namespace/capacity. The suite has 207 passing
tests. Existing uniform, indexed-mesh, and render-to-texture/alpha-blending
regressions also match exactly. Hardware evidence and retained failed diagnostic runs are under the
workspace's `tmp/apple9-buffer-table/` directory.

This change does not add graphics SSBO writes, dynamic API buffer-binding
selection, new vertex formats, or a larger varying interface. Shader-side pointer
indirection is now represented, but those features still need their own lowering,
state support and validation. The twelve-scalar VS→FS limit is unchanged.
