# Generated Apple9 render programs and state

The T8132 driver generates its render shader archive and package state without
an external render seed. Vertex fetch, fragment processing, color clears,
background reload, and tile export use Mesa-generated programs. The archive
contains only the live vertex and fragment blocks for each linked pipeline;
its header and helper directories use the same builder as compute.

## Tile helpers

The background shader reads every active uncompressed tiled attachment and
writes its raw words into the tilebuffer. The driver retains its compiled
entry and resource bindings for initial background and partial-load dispatch.
It also executes as the first ordinary utility draw, with application queries
disabled. Color clears then execute through ordinary Mesa clear shaders.

The export shader iterates over the pixels and samples in a tile, reads raw
tile words at explicit coordinates, and stores them to the attachment's tiled
memory layout. Under the current T8132 launch layout, one export invocation
handles a 32x32 tile for one/two samples or a 32x16 tile for four samples.
Workgroup IDs locate the tile; framebuffer dimensions guard edge pixels.
R16F uses halfword stores, RGBA16F uses two raw words, and BGRA memory receives
a channel swap. MRT layouts derive from the active formats and sample count.

Tile export prepares a compiled fragment entry and resource bindings without
issuing an ordinary draw. Initial and partial end-of-tile dispatch select that
entry. Both helper indices are validated before submission, and the batch
retains the programs and resources until it retires.

The compiler models explicit tile reads with an allocated packed uint32
coordinate (x in the low half, y in the high half) and a constant sample mask.
The current Apple9 lowering accepts the low four mask bits. Graphics buffer
stores load their destination pointer from the resource table and use an
allocated adjacent GPR pair. Normal allocation, dependency tracking, and
encoding validation apply; no capture-specific register sequence implements
the helpers.

## Removed dependency

The render path no longer reads `render-seed-v2.bin`, earlier seed versions,
or the original whole-image template. The seed loader, parser, exporter, and
parser-only tests have been retired. Original inputs remain outside the source
tree for reproducing older control builds. The old archive adapters, library
bodies, EOT program, fixed VBO table, and unused vertex-heap upload path are no
longer retained in the generated archive.

The shared `launcher-fragments-v5.bin` library is a separate development
input for the parameterized launch builder, documented in
`README.apple9-launchers.md`. Compute division data also remains external.
Eliminating the render seed does not eliminate those separate inputs.

## Archive control records

Seed independence does not establish the semantics of every short literal in
an archive builder. EXP-185 tested the helper tables and constant blocks on
T8132 using mutations of our generated archive, without disassembling Apple
executables.

The 16-byte helper slots contain the existing start-relative `JMP_EXEC_ANY`
encoding and padding. Moving their common target while retargeting the slots
preserves rendering; leaving the slots pointed at a stop suppresses it. The
terminal's second four-byte span controls continuation: a stop placed after
it is harmless, but becomes effective when that span is replaced by padding.
Its exact return/mask semantics and the first four bytes' semantics remain
unresolved. Replacing the first four bytes with padding passed the focused
graphics and compute probes; this does not establish a universal no-op.

Zeroing both 64-byte render constant blocks passed all 35 graphics probes and
1,000 frames / 6,000 draws of the varying probe. Explicit archive calls select
the main after these blocks. Their contents are therefore unnecessary in the
tested current render path, but alternative launch modes and removal of the
space reservation were not tested. The compute constant block was not zeroed.
The render builder now leaves these reservations zeroed and no longer emits
vertex constant-program literals or fragment stop/padding contents. Reservation
sizes and entry offsets are unchanged; compute retains its existing constant
helper. See the parent workspace's `tmp/helpers185/RESULTS.md` for the
investigation and `tmp/render-constants186/RESULTS.md` for cleanup validation.

## Archive layout

The render archive now reserves 128 KiB. `asahi/lib/agx_apple9_layout.h`
defines the archive boundary, the following compiler-state region, and its
attachment-descriptor offsets. Both Gallium's state builder and libasahi's
GPU mapping use this layout. Moving the archive boundary moves the adjacent
state records and their writable pages together; static assertions check page
alignment and separation from the next region. Shader-entry encoding retains
its independently validated range and is checked for every main entry.

EXP-189 moved state from +0x18000 to +0x20000 and increased the code archive
from 96 to 128 KiB. The previously rejected simultaneous 256-value vertex and
fragment spill shaders now pass at 1x, 2x and 4x sampling, including exact
readback on all four frames. All 35 graphics regressions, six pressured
partial-render color/depth cases, 32 multi-program pixel checks across 15
forced rollovers, and 271 compiler/launcher/geometry tests pass.

This centralizes relocation of the archive-adjacent state region, not the
entire render address space. The fixed USC base, separate context/encoder
addresses, later resource/launcher regions and selected package self-pointer
relocations remain. The context encoder's +0x18000 offset belongs to a distinct
address space and is intentionally unchanged. See the parent workspace's
`tmp/archive189/RESULTS.md` for validation and remaining constraints.

## Validation

The generated export integration passed 258 compiler tests, including explicit
sample/coordinate encoding and rejection of broken indirect-store address
pairs. The seed-free prototype and the compact two-program archive each passed
all 35 graphics probes and all 672 GLES2 vertex-array/draw CTS cases. The probes
cover half-float preservation/blending, 2x/4x MSAA, up to eight render targets,
depth, coverage, textures, and multiple programs.

A 12 KiB test archive forced 15 rollovers across two multi-program probes;
all 32 pixel checks passed. Full 128x128 scans verify every pixel is populated;
65x33 probes exercise framebuffer edges. These checks establish ordinary
render correctness.

EXP-187 subsequently confirmed partial-render activation with one large draw
and 96 live varying components. Substituting a clear shader only for the
partial-load or partial-store entry loses earlier accumulated colour under
pressure, while small-draw controls pass. The normal driver preserves RGBA8
colour and 24-bit depth ordering at 1x, 2x and 4x sampling. The proof uses
controlled shader-entry effects, not an inferred partial count from archive
rollover. MRT, stencil, half-float and compressed partial renders remain
outside this validation. See `tmp/partial187/RESULTS.md` in the parent workspace.

EXP-188 validates graphics register spilling through normal GLSL compilation:
vertex-only and fragment-only shaders pass with 80, 128 and 256 live uint32
values, and both stages together pass with 80, 128 and 192. All cases pass at
1x, 2x and 4x sampling, with emitted spill stores/reloads confirmed and every
RGBA8 pixel compared against a CPU loop/checksum oracle. Eight-value controls
use no scratch. The 36-configuration matrix passes 144 frames. Both stages at
256 exceed the current 96 KiB archive capacity before submission. Checksums
cross the varying path as two numerically converted 16-bit halves; raw integer
varying bit preservation is outside this result. Individual graphics frame
extent A/B semantics remain unisolated. See `tmp/graphics-spill188/RESULTS.md`
in the parent workspace for sizes, reproduction and limitations.

Exact builds and final integration results are recorded in the parent
workspace under `tmp/render-integrated183/RESULTS.md`, with the preceding
experiments in `tmp/render-store179/`, `tmp/render-store180/`,
`tmp/render-state181/`, and `tmp/render-seed182/`.
