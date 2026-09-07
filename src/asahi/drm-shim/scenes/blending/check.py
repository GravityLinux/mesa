#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse
import json
import math
import re
import struct
import zlib
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--probe',action='store_true')
p.add_argument('--frames',type=int,default=8)
a=p.parse_args()
log_path = a.directory / 'run.log'
if log_path.exists():
    log = log_path.read_text()
    assert len(re.findall(r'G16G UAPI: publication=', log)) == a.frames
    sizes = [int(s) for s in re.findall(r'APPLE9_ENCODER_FIXED bytes=(\d+)', log)]
    assert sizes == [136 + 7 * 116 + 4] * a.frames, sizes
assert len(list(a.directory.glob('render-*-attachment-0.bin'))) == a.frames
size=512
transforms=[(0,0,.2,1),(-.2,0,0,.7),(.2,0,-.2,.7),(0,0,.8,.5),(0,.45,-.5,.2),(0,-.2,-.1,.5),(.35,-.4,-.1,.3)]
colors=[(.25,.5,.75,.8),(.9,.1,.2,.5),(.1,.8,.3,.25),(1,0,1,1),(.1,.2,.9,1),(1,1,1,0),(.8,.7,.9,1)]
vertices=[(-.75,-.75),(.75,-.75),(0,.75)]
clear=(4,5,15,255)

def png(path,rgba):
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    rows=b''.join(b'\0'+rgba[y*size*4:(y+1)*size*4] for y in reversed(range(size)))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',size,size,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))

reports=[]
for frame in range(a.frames):
    reference=[clear]*(size*size);depths=[1.]*(size*size);edges=[False]*(size*size)
    for d in ([0,6,5,2,1,3,4] if frame%2 else [0,1,2,5,6,3,4]):
        dx,dy,z,scale=transforms[d];src=colors[d];blend=d not in (0,4)
        screen=[((x*scale+dx+1)*size/2,(1+y*scale+dy)*size/2) for x,y in vertices]
        (x0,y0),(x1,y1),(x2,y2)=screen
        denom=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        grad=[math.hypot(y1-y2,x2-x1)/abs(denom),math.hypot(y2-y0,x0-x2)/abs(denom),math.hypot(y0-y1,x1-x0)/abs(denom)]
        for y in range(size):
            for x in range(size):
                q0=((y1-y2)*(x+.5-x2)+(x2-x1)*(y+.5-y2))/denom
                q1=((y2-y0)*(x+.5-x2)+(x0-x2)*(y+.5-y2))/denom
                qs=[q0,q1,1-q0-q1];idx=y*size+x
                if all(q>=-g/64 for q,g in zip(qs,grad)) and any(abs(q)<=g/64 for q,g in zip(qs,grad)):
                    edges[idx]=True
                if min(qs)<0 or (z+1)/2>=depths[idx]:continue
                dst=[c/255 for c in reference[idx]]
                if not blend:result=src;depths[idx]=(z+1)/2
                elif a.probe:result=dst
                elif (frame//2)%2==0:result=[src[c]*(1 if c==3 else src[3])+dst[c]*(1-src[3]) for c in range(4)]
                else:result=[src[c]*src[3]+dst[c] for c in range(4)]
                if blend and frame>=4:
                    result=[result[0],dst[1],result[2],dst[3]]
                reference[idx]=tuple(round(255*max(0,min(1,c))) for c in result)
    raw=(a.directory/f'render-{frame:04d}-attachment-0.bin').read_bytes()
    zraw=(a.directory/f'render-{frame:04d}-attachment-1.bin').read_bytes()
    rgba=bytearray(len(raw));errors=zerrors=0;maxerr=0;maxz=0.;examples=[]
    for y in range(size):
        for x in range(size):
            m=sum((((x>>b)&1)<<(2*b))|(((y>>b)&1)<<(2*b+1)) for b in range(6))
            off=(((y//64)*(size//64)+x//64)*4096+m)*4
            b,g,r,alpha=raw[off:off+4];actual=(r,g,b,alpha);idx=y*size+x
            rgba[4*idx:4*idx+4]=bytes(actual)
            if edges[idx]:continue
            error=max(abs(q-r) for q,r in zip(actual,reference[idx]));maxerr=max(maxerr,error)
            if error>1:
                errors+=1
                if len(examples)<10:examples.append([x,y,actual,reference[idx]])
            dz=abs(struct.unpack_from('<f',zraw,off)[0]-depths[idx]);maxz=max(maxz,dz)
            zerrors+=not math.isfinite(dz) or dz>2e-6
    png(a.directory/f'frame{frame:02d}.png',bytes(rgba))
    reports.append(dict(frame=frame,color_errors=errors,depth_errors=zerrors,max_color_error=maxerr,max_depth_error=maxz,examples=examples))
(a.directory/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
assert all(not r['color_errors'] and not r['depth_errors'] for r in reports)
