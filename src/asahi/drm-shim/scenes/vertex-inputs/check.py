#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import json,os,re,sys
from pathlib import Path
root=Path(sys.argv[1]);report=[]
batches=(int(os.getenv('T8132_VERTEX_STREAM_DRAWS','56'))+55)//56 if os.getenv('T8132_VERTEX_INDEPENDENT_STREAMS') else 1
for frame in range(2):
 publication=(frame+1)*batches-1
 raw=(root/f'render-{publication:04d}-attachment-0.bin').read_bytes()
 assert len(raw)==256*256*4
 errors=0;maximum=0;examples=[]
 for y in range(256):
  for x in range(256):
   m=sum((((x>>b)&1)<<(2*b))|(((y>>b)&1)<<(2*b+1)) for b in range(6))
   at=((y//64*4+x//64)*4096+m)*4
   b,g,r,a=raw[at:at+4];actual=[r,g,b,a]
   expected=[round((17+3*(x//4))*(32+192*(x+.5)/256)/255*(.5 if frame else 1)),
             round((23+3*(y//4))*(32+192*(y+.5)/256)/255*.875*(.75 if frame else 1)),
             round((31+x//4+y//4)*64/255*.75),255]
   error=max(abs(a-b) for a,b in zip(actual,expected));maximum=max(maximum,error)
   if error>1:
    errors+=1
    if len(examples)<8:examples.append([x,y,actual,expected])
 report.append(dict(frame=frame,checked_pixels=65536,errors=errors,max_error=maximum,examples=examples))
(root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));assert all(r['errors']==0 for r in report)

# This mode changes allocation spacing while preserving the fetch layout.
if os.getenv("T8132_VERTEX_INDEPENDENT_STREAMS"):
 log=(root/'run.log').read_text()
 assert 'APPLE9_RENDER_ARCHIVE_SPLIT' not in log
 draws=int(os.getenv('T8132_VERTEX_STREAM_DRAWS','56'))
 assert len(re.findall(r'G16G DRM RENDER: opaque TA\+3D publication', log)) == 2*((draws+55)//56)
 # Moving either stream must reuse the same installed vertex program.
 calls=re.findall(r'APPLE9_RENDER_CACHE .*?prolog_call=(0x[0-9a-f]+)',log)
 assert calls and len(set(calls)) == 1, calls
