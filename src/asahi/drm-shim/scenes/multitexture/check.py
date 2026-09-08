#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent per-pixel oracle for sixteen textures, sampler objects and rebinds."""
import json, math, sys, os
from pathlib import Path
root=Path(sys.argv[1]); reports=[]
lod=bool(os.getenv("T8132_MULTITEXTURE_LOD"))
def texel(image,x,y):
    x=max(0,min(7,x));y=max(0,min(7,y))
    return [(17+image*11+x*23+y*7)&255,
            (31+image*13+x*5+y*29)&255,
            (47+image*7+x*17+y*19)&255,255]
def sample(image,u,v,linear):
    if lod:
        level=linear*.25;low=min(4,math.floor(level));high=min(4,low+1);f=level-low
        def color(l):return [(17+image*11+l*53)&255,(31+image*13+l*37)&255,(47+image*7+l*61)&255,255]
        return [a*(1-f)+b*f for a,b in zip(color(low),color(high))]
    if not linear:return texel(image,math.floor(u*8),math.floor(v*8))
    x=u*8-.5;y=v*8-.5;ix=math.floor(x);iy=math.floor(y);fx=x-ix;fy=y-iy
    taps=[texel(image,ix,iy),texel(image,ix+1,iy),texel(image,ix,iy+1),texel(image,ix+1,iy+1)]
    weights=[(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy]
    return [sum(t[c]*w for t,w in zip(taps,weights)) for c in range(4)]
for frame in range(2):
    raw=(root/f'render-{frame:04d}-attachment-0.bin').read_bytes()
    errors=maximum=0;examples=[];rgba=bytearray()
    for y in range(256):
        draw=0 if y<85 else 1 if y<170 else 2
        for x in range(256):
            u=(x+.5)/256*.875+.0625;v=(y+.5)/256*.875+.0625
            units=range(16) if y<128 else [x//16]
            values=[sample(5 if draw==2 else (unit+frame*7+draw*3)%16,
                           u,v,(unit+draw+frame)%(16 if lod else 2)) for unit in units]
            expected=[round(sum(p[c] for p in values)/len(values)) for c in range(4)]
            m=sum((((x>>b)&1)<<(2*b))|(((y>>b)&1)<<(2*b+1)) for b in range(6))
            offset=((y//64*4+x//64)*4096+m)*4
            blue,green,red,alpha=raw[offset:offset+4];actual=[red,green,blue,alpha];rgba.extend(actual)
            err=max(abs(a-b) for a,b in zip(actual,expected));maximum=max(maximum,err)
            if err>1:
                errors+=1
                if len(examples)<12:examples.append([x,y,expected,actual])
    (root/f'frame-{frame}.rgba').write_bytes(rgba)
    reports.append(dict(frame=frame,pixels=65536,errors=errors,max_error=maximum,examples=examples))
(root/'validation.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
assert all(r['errors']==0 for r in reports)
