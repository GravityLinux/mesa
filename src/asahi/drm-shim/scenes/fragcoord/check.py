#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import json, math, struct, sys
from pathlib import Path
root = Path(sys.argv[1]); w, h = map(int, sys.argv[2:4])
reports = []
for frame in range(3):
    raw = (root / f'render-{frame:04d}-attachment-0.bin').read_bytes()
    depth = (root / f'render-{frame:04d}-attachment-1.bin').read_bytes()
    errors = zerrors = maxerr = 0
    examples = []
    vx, vy, vw, vh = (37, 53, w-101, h-127) if frame == 2 else (0, 0, w, h)
    for y in range(h):
        for x in range(w):
            gx, gy = x+.5, y+.5
            covered = vx <= gx < vx+vw and vy <= gy < vy+vh
            if not covered:
                expected = [4,5,15,255]
            elif frame == 1:
                expected = [128,128,round(((x+3*int(gy))%17)/16*255),255]
            else:
                expected = [round(gx/w*255),round(gy/h*255),round((gy-vy)/vh*255),255]
            m = sum((((x>>b)&1)<<(2*b)) | (((y>>b)&1)<<(2*b+1)) for b in range(6))
            off = (((y//64)*((w+63)//64)+x//64)*4096+m)*4
            blue,green,red,alpha = raw[off:off+4]
            actual = [red,green,blue,alpha]
            err = max(abs(a-b) for a,b in zip(actual,expected))
            maxerr = max(maxerr,err)
            if err > 1:
                errors += 1
                if len(examples)<8: examples.append([x,y,expected,actual])
            z = struct.unpack_from('<f',depth,off)[0]
            zerrors += not math.isfinite(z) or abs(z-(.5 if covered else 1))>2e-6
    reports.append(dict(frame=frame,width=w,height=h,color_errors=errors,depth_errors=zerrors,max_error=maxerr,examples=examples))
(root/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
assert all(not r['color_errors'] and not r['depth_errors'] for r in reports)
