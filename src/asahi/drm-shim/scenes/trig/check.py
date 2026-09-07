#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);w,h=map(int,sys.argv[2:4])
y,x=np.mgrid[:h,:w];m=np.zeros_like(x)
for b in range(6):m|=((x>>b)&1)<<(2*b);m|=((y>>b)&1)<<(2*b+1)
off=((y//64)*((w+63)//64)+x//64)*4096+m
index=(x//2+y*(w//2)).astype(np.float32)
radii=[np.float32(np.pi),32,512,8192,65536,np.float32(1e-6),512,1048576]
reports=[]
for frame,radius in enumerate(radii):
    radius=np.float32(radius);step=np.float32(np.float32(2)*radius/np.float32(w*h//2-1))
    arg=np.float32(-radius+np.float32(index*step))
    if frame==6:
        arg=np.float32(np.float32(np.floor(index/np.float32(1024))*np.float32(np.pi/2))+np.float32((np.mod(index,1024)-512)*np.float32(2**-20)))
    raw=np.frombuffer((root/f'render-{frame:04d}-attachment-0.bin').read_bytes(),dtype=np.uint8).reshape(-1,4)[off]
    bits=(raw[:,:,2].astype(np.uint32)|raw[:,:,1].astype(np.uint32)<<8|raw[:,:,0].astype(np.uint32)<<16|raw[:,:,3].astype(np.uint32)<<24)
    actual=bits.view(np.float32)
    reference=np.where(x%2==0,np.sin(arg.astype(float)),np.cos(arg.astype(float)))
    error=np.abs(actual.astype(float)-reference)
    # A range-dependent absolute envelope, fixed before the hardware run.
    # This records degradation at large magnitudes rather than claiming ULP accuracy.
    limit=5e-7+np.abs(arg.astype(float))*2e-7
    failed=~np.isfinite(actual)|(error>limit)
    depth=np.frombuffer((root/f'render-{frame:04d}-attachment-1.bin').read_bytes(),dtype='<f4')[off]
    report=dict(frame=frame,range=[float(arg.min()),float(arg.max())],samples=w*h,
                max_abs_error=float(error.max()),rms_error=float(np.sqrt(np.mean(error**2))),
                failed=int(failed.sum()),depth_errors=int(np.count_nonzero(depth!=.5)))
    for label,mask in [('sin',x%2==0),('cos',x%2==1)]:report[label+'_max_abs_error']=float(error[mask].max())
    reports.append(report)
(root/'validation.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
assert all(not r['failed'] and not r['depth_errors'] for r in reports)
