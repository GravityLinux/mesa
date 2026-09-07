#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent GLES triangle/depth reference and tiled attachment comparison."""
import json
import os
import re
import math
import struct
import sys
from pathlib import Path
import zlib

def png(path, rgba, size):
    def chunk(kind, data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
    rows=b''.join(b'\0'+rgba[y*size*4:(y+1)*size*4] for y in reversed(range(size)))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',size,size,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))

root = Path(sys.argv[1])
size = 512
seq = [0, 1, 0, 2, 1, 0]
transforms = [(-.3,-.1,.1,.8),(0,.15,.4,.9),(.4,-.2,-.2,.6),
              (-.45,.4,-.6,.3),(0,-.5,-.4,.4),(.45,.5,0,.35)]
tints = [(1,.5,.25),(.25,1,.5),(.5,.25,1),(1,1,1),(1,.5,1),(.5,1,1)]
split=bool(os.environ.get('T8132_GLES_PIPELINE_SPLIT'))
wide=bool(os.environ.get('T8132_GLES_PIPELINE_WIDE'))
pin=bool(os.environ.get('T8132_GLES_PIPELINE_PIN'))
if pin:
    seq=[2]*21
    transforms=[(-.8+(i%5)*.4,-.65+(i//5)*.4,0,.16) for i in range(20)]+[(0,.85,0,.16)]
    tints=[(.1+.03*i,.2+.02*i,.9-.03*i) for i in range(20)]+[(.1,.2,.9)]
vertices = [(-.75,-.75),(.75,-.75),(0,.75)]
# Match the fixture's glClearColor, now honored by the Apple9 path.
clear = (191,186,255,255)
owners = [-1] * (size*size)
expected = [clear] * (size*size)
depths = [1.] * (size*size)
edges = [False] * (size*size)
for draw, (program, (dx,dy,z,scale), tint) in enumerate(zip(seq,transforms,tints)):
    screen = [(((-x if program==1 else x)*scale+dx+1)*size/2,
               (1+y*scale+dy)*size/2) for x,y in vertices]
    (x0,y0),(x1,y1),(x2,y2) = screen
    denom = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
    gradients = [math.hypot(y1-y2,x2-x1)/abs(denom),
                 math.hypot(y2-y0,x0-x2)/abs(denom),
                 math.hypot(y0-y1,x1-x0)/abs(denom)]
    colors=[]
    for x,y in vertices:
        a = [x*.25+.5,y*.25+.5,(x+y)*.125+.375]
        b = [y*.125+.25,x*.125+.5,(x-y)*.0625+.625]
        c = [x*.0625+.125,y*.0625+.375,x*.03125+y*.015625+.75]
        colors.append((tint if pin else [.125,.75,1]) if program==2 else
                      [(a[k] if program==0 else a[k]*.5+b[k]*.25+c[k]*.125)*tint[k]
                       for k in range(3)])
    for y in range(size):
        for x in range(size):
            q0=((y1-y2)*(x+.5-x2)+(x2-x1)*(y+.5-y2))/denom
            q1=((y2-y0)*(x+.5-x2)+(x0-x2)*(y+.5-y2))/denom
            qs=[q0,q1,1-q0-q1]
            idx=y*size+x
            if all(q >= -g/64 for q,g in zip(qs,gradients)) and any(abs(q)<=g/64 for q,g in zip(qs,gradients)):
                edges[idx]=True
            if min(qs)<0 or (z+1)/2>=depths[idx]:
                continue
            owners[idx]=program
            depths[idx]=(z+1)/2
            value=[sum(q*c[k] for q,c in zip(qs,colors)) for k in range(3)]
            if wide:
                phase=draw if draw<20 else 0
                px,py=(x+.5)*.01,(y+.5)*.01
                waves=[math.sin(px+phase),math.cos(py+phase),math.sin(px+py+phase)]
                value=[v+.02*w for v,w in zip(value,waves)]
            expected[idx]=tuple(round(255*v) for v in value)+(255,)
log = (root/'run.log').read_text()
draw_count=int(os.environ.get('T8132_GLES_DRAWS','21' if pin else '6'))
# The utility clear consumes one indexed slot; application draws split only
# when the 32-slot arena is full. Check every command length and final image.
remaining = draw_count
first = min(31, remaining)
frame_sizes = [136 + 116*first + 4]
remaining -= first
while remaining:
    count = min(32, remaining)
    frame_sizes.append(116*count + 4)
    remaining -= count
submissions = len(frame_sizes)
if split:
    # Frame markers are flushed after glFinish. Associate every sub-batch
    # with its frame and validate that each indexed draw was emitted once.
    final_captures=[]
    previous=0
    encoder_sizes=[]
    publication_count=0
    for line in log.splitlines():
        match=re.search(r'APPLE9_ENCODER_FIXED bytes=(\d+)',line)
        if match: encoder_sizes.append(int(match[1]))
        if 'G16G UAPI: publication=' in line: publication_count+=1
        if 'T8132_PIPELINES_FRAME frame=' in line:
            frame_sizes=encoder_sizes[previous:]
            assert len(frame_sizes)>1, frame_sizes
            # Clear and application draws each use a 136-byte indexed record.
            assert all((n-4)%136==0 for n in frame_sizes), frame_sizes
            assert sum((n-4)//136 for n in frame_sizes)==draw_count+1, frame_sizes
            assert publication_count==len(encoder_sizes)
            final_captures.append(publication_count-1)
            previous=len(encoder_sizes)
    assert len(final_captures)==4
    assert len(re.findall(r'APPLE9_RENDER_ARCHIVE_SPLIT',log))==publication_count-4
    assert len(list(root.glob('render-*-attachment-0.bin')))==publication_count
    final_capture=lambda frame: final_captures[frame]
else:
    assert len(re.findall(r'G16G UAPI: publication=',log)) == 4*submissions
    assert len(list(root.glob('render-*-attachment-0.bin'))) == 4*submissions
    sizes=[int(s) for s in re.findall(r'APPLE9_ENCODER_FIXED bytes=(\d+)',log)]
    assert sizes == frame_sizes*4, sizes
    final_capture = lambda frame: (frame+1)*submissions-1
rollover=bool(os.environ.get('T8132_GLES_PIPELINE_ROLLOVER'))
if rollover:
    assert len(re.findall(r'APPLE9_RENDER_BATCH_ROLLOVER',log)) == 2
if pin:
    assert len(re.findall(r'APPLE9_RENDER_PACKAGE storage=',log)) >= 20
if wide and not split:
    calls=[int(v,16) for v in re.findall(r'APPLE9_RENDER_CACHE fs_call=(0x[0-9a-f]+)',log)]
    assert max(calls)>0x1ffff, "fixture must execute entries above the old 64 KiB boundary"
reports=[]
for frame in range(4):
    raw=(root/f'render-{final_capture(frame):04d}-attachment-0.bin').read_bytes()
    depth=(root/f'render-{final_capture(frame):04d}-attachment-1.bin').read_bytes()
    assert len(raw)==len(depth)==size*size*4
    rgba=bytearray(len(raw)); errors=depth_errors=0; max_color=0; max_depth=0.; examples=[]
    for y in range(size):
        for x in range(size):
            m=sum((((x>>b)&1)<<(2*b))|(((y>>b)&1)<<(2*b+1)) for b in range(6))
            off=(((y//64)*(size//64)+x//64)*4096+m)*4
            b,g,r,a=raw[off:off+4]; actual=(r,g,b,a); idx=y*size+x
            rgba[4*idx:4*idx+4]=bytes(actual)
            if edges[idx]: continue
            ref = (64,128,191,255) if rollover and frame//2==1 and owners[idx]==2 else expected[idx]
            error=max(abs(a-b) for a,b in zip(actual,ref));max_color=max(max_color,error)
            if error>1:
                errors+=1
                if len(examples)<5: examples.append([x,y,actual,expected[idx]])
            dz=abs(struct.unpack_from('<f',depth,off)[0]-depths[idx]);max_depth=max(max_depth,dz)
            depth_errors+=not math.isfinite(dz) or dz>2e-6
    png(root/f'frame{frame:02d}.png', bytes(rgba), size)
    reports.append(dict(frame=frame,color_errors=errors,depth_errors=depth_errors,
                        max_color_error=max_color,max_depth_error=max_depth,examples=examples))
(root/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
assert all(not r['color_errors'] and not r['depth_errors'] for r in reports)
# Reverse submission order must preserve the same visible surfaces.
for attachment in range(2):
    first=(root/f'render-{final_capture(0):04d}-attachment-{attachment}.bin').read_bytes()
    assert all((root/f'render-{final_capture(frame):04d}-attachment-{attachment}.bin').read_bytes()==first for frame in range(1,2 if rollover else 4))
    if rollover:
        assert (root/f'render-{final_capture(2):04d}-attachment-{attachment}.bin').read_bytes() == (root/f'render-{final_capture(3):04d}-attachment-{attachment}.bin').read_bytes()
