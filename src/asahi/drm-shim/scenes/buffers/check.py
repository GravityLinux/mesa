#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU oracle for buffer bindings and per-draw snapshots."""
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
count = int(os.getenv('T8132_BUFFER_COUNT', '15'))
vertex = bool(os.getenv('T8132_BUFFER_VERTEX'))
vbos = bool(os.getenv('T8132_BUFFER_VBOS'))
reports = []
for frame in range(2):
    raw = (root / f'render-{frame:04d}-attachment-0.bin').read_bytes()
    errors = maximum = 0
    examples = []
    for y in range(256):
        draw = y//16
        selected = list(range(count)) if draw == 15 else ([draw] if draw < count else [])
        sums = []
        for stage in range(2):
            sums.append([sum((((stage*16+i+frame*5+draw*3)%32)*7+(i%4*4+c)*3+11)%64 / 64 for i in selected)/(count if draw==15 else 1) for c in range(3)])
        if not vertex:
            sums[0] = [.25,.5,.75]
        if vbos:
            for c in range(3): sums[0][c] += sum(((i*5+c*9+3)%64)/64 for i in range(16))/64
        shift = [draw/128,frame/16,(frame+draw)/256]
        expected = [round(min(1,(a+b+shift[c]*(2 if vertex else 1))*.5)*255) for c,(a,b) in enumerate(zip(*sums))] + [255]
        for x in range(256):
            morton = sum((((x>>b)&1)<<(2*b)) | (((y>>b)&1)<<(2*b+1)) for b in range(6))
            offset = ((y//64*4+x//64)*4096+morton)*4
            blue,green,red,alpha = raw[offset:offset+4]
            actual = [red,green,blue,alpha]
            l1=(x+.5)/512; l2=(y+.5)/512; l0=1-l1-l2
            gradient=(l1/1.25+2*l2/1.5)/(l0+l1/1.25+l2/1.5)/64
            pixel_expected=expected.copy()
            pixel_expected[0]=round(min(1,(sums[0][0]+sums[1][0]+shift[0]*(2 if vertex else 1)+gradient)*.5)*255)
            error = max(abs(a-b) for a,b in zip(actual,pixel_expected))
            maximum = max(maximum,error)
            if error > 1:
                errors += 1
                if len(examples)<8: examples.append([x,y,pixel_expected,actual])
    reports.append(dict(frame=frame,pixels=65536,errors=errors,max_error=maximum,examples=examples))
(root/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
assert all(r['errors']==0 for r in reports)
