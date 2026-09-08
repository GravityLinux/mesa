#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check two uniform-driven draws with independent VS and FS data against a CPU pixel oracle.

The PNG uses only detiled hardware bytes. The reference checks the complete
coverage mask and each triangle's independently interpolated vertex colors.
"""
import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

TRIANGLES = (
    ((-.875, -.5), (-.125, -.5), (-.5, .25)),
    ((.875, -.25), (.875, .5), (.125, .125)),
)
COLORS = (
    ((1, 0, 0), (0, .5, 0), (0, 0, .25)),
    ((.25, 0, 0), (0, .75, 0), (0, 0, 1)),
)
CLEAR = bytes((191, 186, 255, 255))  # glClearColor(0.75, 0.73, 1, 1).


def barycentric(x, y, vertices):
    (ax, ay), (bx, by), (cx, cy) = vertices
    d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    a = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / d
    b = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / d
    return a, b, 1 - a - b


def chunk(kind, body):
    return (struct.pack('>I', len(body)) + kind + body +
            struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('attachment', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--frame', type=int, default=0)
args = parser.parse_args()
raw = args.attachment.read_bytes()
size = 512
if len(raw) != size * size * 4:
    parser.error('expected an uncompressed 512x512 BGRA8 attachment')
linear = bytearray(len(raw))
covered = [0, 0]
coverage_errors = color_errors = max_error = 0
examples = []
for y in range(size):
    for x in range(size):
        morton = sum((((x >> bit) & 1) << (2 * bit)) |
                     (((y >> bit) & 1) << (2 * bit + 1)) for bit in range(6))
        offset = (((y // 64) * 8 + x // 64) * 4096 + morton) * 4
        bgra = raw[offset:offset + 4]
        rgba = bytes((bgra[2], bgra[1], bgra[0], bgra[3]))
        linear[(y * size + x) * 4:(y * size + x + 1) * 4] = rgba
        sx, sy = (x + .5) / 256 - 1, 1 - (y + .5) / 256
        expected, primitive = CLEAR, None
        for i, vertices in enumerate(TRIANGLES):
            weights = barycentric(sx, sy, vertices)
            if min(weights) > 0:
                assert primitive is None, 'fixture triangles must not overlap'
                primitive = i
                covered[i] += 1
                expected = [round(255 * min(1, sum(w * c[channel]
                                           for w, c in zip(weights, COLORS[i])) +
                                           (args.frame * .125 if channel == 0 else 0)))
                            for channel in range(3)] + [255]
        coverage_errors += (primitive is not None) != (rgba != CLEAR)
        error = max(abs(a - b) for a, b in zip(rgba, expected))
        max_error = max(max_error, error)
        if error > 1:
            color_errors += 1
            if len(examples) < 8:
                examples.append(dict(x=x, y=y, primitive=primitive,
                                     actual=list(rgba), expected=list(expected)))

report = dict(vertices_per_draw=3, draw_calls=2, frame=args.frame, pixels=size * size,
              covered_per_triangle=covered, coverage_errors=coverage_errors,
              color_errors=color_errors, max_channel_error=max_error,
              examples=examples, attachment_sha256=hashlib.sha256(raw).hexdigest())
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
png = b'\x89PNG\r\n\x1a\n'
png += chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress(b''.join(
    b'\0' + linear[y * size * 4:(y + 1) * size * 4] for y in range(size))))
png += chunk(b'IEND', b'')
args.output.with_suffix('.png').write_bytes(png)
print(json.dumps(report))
assert all(covered) and coverage_errors == color_errors == 0
