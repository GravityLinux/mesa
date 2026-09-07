#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import json, math, struct, sys
from pathlib import Path
root=Path(sys.argv[1]); w,h=map(int,sys.argv[2:4])
colors=[[4,5,15,255]]*(w*h); depths=[1.]*(w*h)
reports=[]
frames=int(sys.argv[4]) if len(sys.argv)>4 else 7
color_only=len(sys.argv)>5 and sys.argv[5]=="1"
def rect(x,y,rw,rh,color,z=None):
    c=[round(255*v) for v in color]
    for yy in range(max(0,y),min(h,y+rh)):
        for xx in range(max(0,x),min(w,x+rw)):
            at=yy*w+xx;colors[at]=c
            if z is not None: depths[at]=z
for frame in range(frames):
    if frame<5 or frame>=7:
        colors=[[4,5,15,255]]*(w*h);depths=[1.]*(w*h)
    if frame>=7:
        for i in range(31 if frame==7 else 40):
            rect(7+(i%8)*53,11+(i//8)*67,41,55,[.25+.25*(i%3),.125+.125*(i%5),.5,1],.5)
    tint=[.75,.25,.5,1]
    if frame==0: rect(37,53,w-101,h-127,tint,.5)
    elif frame==1: rect(13,19,91,73,tint,.5)
    elif frame==2: rect(-11,-7,43,39,tint,.5)
    elif frame==3:
        rect(17,23,103,89,tint,.5)
        rect(61,47,109,67,[.125,.75,.375,.5],.5)
        rect(101,89,47,59,[.5,.125,.875,.25],.5)
    elif frame==4: rect(29,41,107,83,[.2,.4,.6,.5])
    elif frame==5: rect(83,97,113,71,[.8,.6,.2,.25])
    raw=(root/f'render-{frame:04d}-attachment-0.bin').read_bytes()
    depth_path=root/f'render-{frame:04d}-attachment-1.bin'
    if color_only: assert not depth_path.exists()
    depth=None if color_only else depth_path.read_bytes()
    errors=zerrors=maxerr=0;examples=[]
    for y in range(h):
        for x in range(w):
            m=sum((((x>>b)&1)<<(2*b))|(((y>>b)&1)<<(2*b+1)) for b in range(6))
            off=(((y//64)*((w+63)//64)+x//64)*4096+m)*4
            blue,green,red,alpha=raw[off:off+4]
            actual=[red,green,blue,alpha];expected=colors[y*w+x]
            err=max(abs(a-b) for a,b in zip(actual,expected));maxerr=max(maxerr,err)
            if err>1:
                errors+=1
                if len(examples)<8:examples.append([x,y,expected,actual])
            if depth is not None:
                z=struct.unpack_from('<f',depth,off)[0]
                zerrors+=not math.isfinite(z) or abs(z-depths[y*w+x])>2e-6
    reports.append(dict(frame=frame,color_errors=errors,depth_errors=zerrors,max_error=maxerr,examples=examples))
(root/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
assert len(list(root.glob('render-*-attachment-0.bin')))==frames
assert all(not r['color_errors'] and not r['depth_errors'] for r in reports)
