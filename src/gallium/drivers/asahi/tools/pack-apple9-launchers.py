#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Factor caller-owned launchers into external raw fragments, without decoding.

Requires the validated T8132 resource layouts. This checks every shared byte
and reconstructs every input before writing the versioned fragment library.
No instruction boundaries or control flow are inferred. Later schemas omit
spans replaced by independently generated and hardware-validated operations.
Remaining trimmed zero storage is restored within the output allocation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

INPUTS = [
    ('vertex', 'render_vertex_texture_varyings32_launch.bin', 256, 0x44),
    ('fragment', 'render_texture_coords8_launch.bin', 256, 0x44),
    ('fragment-coverage', 'render_coverage_texture_launch.bin', 256, 0x62),
    ('compute', 'carrier8/launch.bin', 1024, 0x8c),
]
FRAGMENTS = [
    'graphics-prefix', 'coverage-prefix', 'compute-prefix', 'common-frame-body',
    'graphics-common-tail', 'vertex-tail', 'fragment-tail', 'coverage-tail',
    'compute-tail',
]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def normalized_prefix(data, call):
    out = bytearray(data[:call+8])
    # Resource root, retaining all bits outside the established relocation.
    out[1] = 0x80
    out[4] &= ~0x1f
    out[5] &= ~0x0c
    out[6:8] = bytes(2)
    # Main entry and opaque stage-specific bytes are provided separately.
    out[call:call+3] = bytes(3)
    out[call-7:call-5] = bytes(2)
    return bytes(out)

def normalized_body(data, call):
    out = bytearray(data[call+8:call+66])
    if len(out) != 58 or out[:3] != bytes.fromhex('f7002a'):
        raise ValueError('unrecognized frame record')
    out[3:9] = bytes(6)
    out[30] = 0  # Unresolved stage byte; preserved in configuration metadata.
    return bytes(out)

def pack(input_dir, compute_return=None, generate_graphics=False, generate_compute_state=False):
    inputs = [(name, (input_dir/file).read_bytes(), size, call)
              for name, file, size, call in INPUTS]
    for name, data, size, _ in inputs:
        if len(data) != size:
            raise ValueError(f'{name}: expected {size} bytes, got {len(data)}')
    prefixes = [normalized_prefix(data, call) for _, data, _, call in inputs]
    bodies = [normalized_body(data, call) for _, data, _, call in inputs]
    if prefixes[0] != prefixes[1]:
        raise ValueError('vertex/fragment resource prefixes no longer match')
    if any(body != bodies[0] for body in bodies):
        raise ValueError('common frame/body differs outside known parameters')
    common_tails = [data[call+66:call+114] for _, data, _, call in inputs[:3]]
    if any(tail != common_tails[0] for tail in common_tails):
        raise ValueError('graphics common tails differ')
    tails = [data[call+66+(0 if i == 3 else 48):].rstrip(b'\0')
             for i, (_, data, _, call) in enumerate(inputs)]
    fragments = [prefixes[1], prefixes[2], prefixes[3], bodies[0],
                 common_tails[0], *tails]
    configs = []
    for _, data, size, call in inputs:
        configs.append(struct.pack('<HHHBBBBH',
            *struct.unpack_from('<HHH', data, call+11),
            data[call-7], data[call-6], data[call+38], 0, size))
    # Prove lossless factoring before emitting anything. Relocation fields
    # are restored from the input here; the driver supplies live values.
    for i, (name, data, size, call) in enumerate(inputs):
        rebuilt = bytearray(prefixes[i] + bodies[0] +
                            (b'' if i == 3 else common_tails[0]) + tails[i])
        rebuilt.extend(bytes(size-len(rebuilt)))
        rebuilt[1] = data[1]
        rebuilt[4:8] = data[4:8]
        rebuilt[call:call+3] = data[call:call+3]
        rebuilt[call-7:call-5] = data[call-7:call-5]
        rebuilt[call+11:call+17] = data[call+11:call+17]
        rebuilt[call+38] = data[call+38]
        if rebuilt != data:
            raise ValueError(f'{name}: factoring is not lossless')
    # Coverage changes only the publication default within the frame metadata.
    # Keep three stage recipes and store that one optional parameter separately.
    if configs[1][2:] != configs[2][2:]:
        raise ValueError('coverage differs from the fragment stage metadata')
    coverage_word = configs[2][:2]
    configs = [configs[0], configs[1], configs[3]]
    version = 2
    if compute_return is not None:
        ending = compute_return.read_bytes()
        if len(ending) != 12:
            raise ValueError('compute return fragment must have exactly 12 bytes')
        # Measured state-load and pre-call spans survive resource generation.
        fragments[2] = fragments[2][116:148]
        fragments[8] = ending
        version = 3
    if generate_graphics:
        if compute_return is None:
            raise ValueError('generated graphics requires the version 3 compute return input')
        if prefixes[1][-18:] != prefixes[2][-18:]:
            raise ValueError('graphics pre-call spans no longer match')
        if prefixes[1][16:58] != prefixes[2][32:74]:
            raise ValueError('coverage resource-load span differs')
        fragments[0] = prefixes[1][-18:]
        fragments[1] = prefixes[2][16:32] + prefixes[2][74:88]
        fragments[4] = b''
        version = 4
    if generate_compute_state:
        if not generate_graphics:
            raise ValueError('generated compute state requires generated graphics')
        # EXP-175 replaces four-word state loading with the established encoder.
        # Keep only the independently retained pre-call span.
        fragments[2] = fragments[2][14:]
        version = 5
    header_size = 16 + len(fragments)*8 + len(configs)*12 + 4
    directory, payload, cursor = [], bytearray(), header_size
    for fragment in fragments:
        directory.append(struct.pack('<II', cursor, len(fragment)))
        payload.extend(fragment)
        cursor += len(fragment)
    packed = struct.pack('<8sII', f'A9LFRG{version:02d}'.encode(), len(fragments), len(configs))
    packed += b''.join(directory) + b''.join(configs) + coverage_word + bytes(2) + payload
    manifest = dict(schema=version, method=__doc__, packed_sha256=sha(packed),
        packed_bytes=len(packed), opaque_bytes=sum(map(len, fragments)),
        input_bytes=sum(len(data) for _, data, _, _ in inputs),
        inputs=[dict(name=name, file=file, sha256=sha(data), bytes=size)
                for (name, file, size, _), (_, data, _, _) in zip(INPUTS, inputs)],
        fragments=[dict(name=name, bytes=len(data), sha256=sha(data))
                   for name, data in zip(FRAGMENTS, fragments)])
    if compute_return is not None:
        manifest['compute_return'] = dict(file=str(compute_return), sha256=sha(ending), bytes=12)
    return packed, manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument("--compute-return", type=Path, help="Validated external return fragment; emits version 3 with generated compute resources")
    parser.add_argument("--generate-graphics", action="store_true", help="Emit version 4 with generated graphics resources and external coverage spans")
    parser.add_argument("--generate-compute-state", action="store_true", help="Emit version 5 with generated compute state loads")
    args = parser.parse_args()
    packed, manifest = pack(args.input_dir, args.compute_return, args.generate_graphics, args.generate_compute_state)
    args.output.write_bytes(packed)
    args.manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Validated four configurations / three stages: {len(packed)} bytes, '
          f'{manifest["opaque_bytes"]} opaque bytes in nine shared fragments')

if __name__ == '__main__':
    main()
