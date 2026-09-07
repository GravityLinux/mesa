#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse, json, math, struct
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
size = 512
reports = []
for frame in range(8):
    raw = (a.directory / f'render-{frame:04d}-attachment-0.bin').read_bytes()
    depth = (a.directory / f'render-{frame:04d}-attachment-1.bin').read_bytes()
    rounds = frame % 4
    errors = zerrors = 0
    maxerr = 0
    examples = []
    for y in range(size):
        for x in range(size):
            u, v = (x + .5) / size, (y + .5) / size
            count = rounds if u < .5 else rounds + 2
            total = 0
            for i in range(count):
                for j in range(rounds):
                    if j == 1: continue
                    if v < .5 and i > 1: break
                    total += 1/64
            values = [sum(1/32 for i in range(rounds + k) if i != 1) for k in range(3)]
            vertex_value = values[0]*(1-u/2-v/2) + values[1]*u/2 + values[2]*v/2
            expected = [total, vertex_value, v/4, .5]
            if frame >= 4:
                expected = [expected[c] * (1 if c == 3 else .5) + [4,5,15,255][c]/255*.5 for c in range(4)]
            expected = [round(255*c) for c in expected]
            m = sum((((x>>b)&1)<<(2*b)) | (((y>>b)&1)<<(2*b+1)) for b in range(6))
            off = (((y//64)*8 + x//64)*4096 + m)*4
            b,g,r,alpha = raw[off:off+4]
            actual = [r,g,b,alpha]
            err = max(abs(c-d) for c,d in zip(expected, actual))
            maxerr = max(maxerr,err)
            if err > 1:
                errors += 1
                if len(examples)<10: examples.append([x,y,expected,actual])
            z = struct.unpack_from('<f',depth,off)[0]
            zerrors += not math.isfinite(z) or abs(z-.5)>2e-6
    reports.append(dict(frame=frame,color_errors=errors,depth_errors=zerrors,max_color_error=maxerr,examples=examples))
(a.directory/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
assert all(not r['color_errors'] and not r['depth_errors'] for r in reports)
