# Formatted vertex input regression

Run `run.sh OUTPUT_DIRECTORY` against a freshly chainloaded target. This draws
and checks two 256x256 textured quads through ordinary GLES vertex arrays.

The scene covers sparse attribute locations, interleaved FP32 positions,
UNORM16 UVs, UNORM8 colors, default Z/W components, a disabled constant color
attribute, uniforms, nonzero buffer and draw offsets (including a byte-aligned color base before aligned
FP32/UNORM16 attributes), replacement VBOs, and
indexed drawing with an index-buffer offset. The second frame changes the
constant attribute as well as the VBO. `check.py` compares every pixel with an
independent CPU interpolation/sampling reference, allowing one byte of error.

The compiler accepts naturally aligned FP32 and signed/unsigned normalized or
scaled 8/16-bit channels. Gallium advertises this subset so its vertex
translation path can handle other formats. Attributes referencing the same
resource share a hardware buffer argument; separate resources and uniforms
still share the current four-argument stage limit. Instancing remains outside
this direct path.
