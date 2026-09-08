#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate hardware-only color bytes against independent indexed rasterization."""
import argparse,hashlib,json,math,struct,zlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('attachment',type=Path);p.add_argument('--mode',choices=['quad','depth','cube'],required=True)
p.add_argument('--depth-states',action='store_true');p.add_argument('--depth-attachment',type=Path);p.add_argument('--four-buffers',action='store_true');p.add_argument('--frame',type=int,default=0);p.add_argument('--output',type=Path,required=True)
p.add_argument('--interpolation', choices=['linear','flat','mixed','integer']);p.add_argument('--varyings',type=int,choices=[0,7,8,9,12,16,24,32]);p.add_argument('--perspective-varyings',action='store_true');p.add_argument('--clipped-varyings',action='store_true');p.add_argument('--depth-clipped-varyings',action='store_true')
a=p.parse_args();size=512;clear=(0,0,0,0)
if a.mode=='quad':
 vertices=[(-.75,-.5,0),(.75,-.5,0),(.75,.5,0),(-.75,.5,0)]
 indices=[2,0,1,0,2,3]
elif a.mode=='depth':
 vertices=[(x,y,z) for z in [-.5,.5] for x,y in [(-.75,-.75),(.75,-.75),(0,.75)]]
 indices=list(range(6))
else:
 vertices=[(-.6,-.6,-.6),(.6,-.6,-.6),(.6,.6,-.6),(-.6,.6,-.6),
           (-.6,-.6,.6),(.6,-.6,.6),(.6,.6,.6),(-.6,.6,.6)]
 indices=[0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5]
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
vertices=[tuple(map(f32,v)) for v in vertices]
colors=[]
for i,(x,y,z) in enumerate(vertices):
 colors.append(((1 if i<3 else 0),(0 if i<3 else .875),.25*.75)
               if a.mode=='depth' else ((x*.6+.5),(y*.6+.5)*.875,(z*.6+.5)*.75 if a.mode=='cube' else .375*.75))
if a.varyings is not None:
 assert a.mode == 'quad' and not a.four_buffers and not a.depth_states
 colors=[]
 for x,y,z in vertices:
  first=(x*.6+.5,y*.6+.5,.375)
  second=(x*.25+.3,y*.25+.4,z*.25+.5)
  third=(x*x*.25+.2,y*y*.25+.3,x*y*.25+.4)
  if a.varyings==7: third=(third[0],)*3
  if a.varyings==8: third=(third[0],third[1],third[0]+third[1])
  extra=(x*.125+.25,y*.125+.25,(x+y)*.125+.5)
  colors.append(tuple((.1*(k+1) if a.varyings==0 else
     first[k]*.2+second[k]*.3+third[k]*.4+(extra[k]*.1 if a.varyings==12 else 0))
     * [1,.875,.75][k] for k in range(3)))
if a.varyings in (16,24,32):
 colors=[]
 for x,y,z in vertices:
  c=[0.,0.,0.]
  for j in range(a.varyings//4):
   v=[q*(j+1)/32+b for q,b in zip((x,y,x*y,x*x+y*y+.3*x+.2*y),(.3,.4,.5,.6))]
   for k in range(3): c[k]+=(v[k]*.75+v[3]*.25)/(a.varyings//4)
  colors.append(tuple(c[k]*[1,.875,.75][k] for k in range(3)))
if a.interpolation:
 colors=[(x*.6+.5,(y*.6+.5)*.875,(x*y*.25+.4)*.75) for x,y,z in vertices]
if a.interpolation == 'integer':
 # Full-width comparisons in the FS catch corruption of every payload bit.
 colors=[(.95 if x>0 else .05, (.8 if y>0 else .2)*.875,
          (.7 if x*y>0 else .1)*.75) for x,y,z in vertices]
clip_w=[1+x*.5 if a.perspective_varyings or a.interpolation else 1. for x,y,z in vertices]
if a.clipped_varyings:
 assert a.mode == 'quad' and (a.interpolation or (a.varyings == 9 and a.perspective_varyings))
 vertices=[(x*2,y,z) for x,y,z in vertices]
if a.depth_clipped_varyings:
 assert a.mode == 'quad' and (a.interpolation or (a.varyings == 9 and a.perspective_varyings))
 vertices=[(x,y,4*x/w) for (x,y,z),w in zip(vertices,clip_w)]
if a.four_buffers:
 colors=[tuple(x*(1-i*.0625) for x in c) for i,c in enumerate(colors)]
if a.mode=='cube':
 # Reproduce the input float32 matrix values independently from the driver.
 angle=f32(f32(.35)+f32((a.frame//2)*f32(.45)))
 ca,sa=map(f32,(math.cos(angle),math.sin(angle)));cb,sb=map(f32,(math.cos(f32(.45)),math.sin(f32(.45))))
 # The GLSL matrix product is separate FP32 multiplies and adds. Keeping
 # CPU doubles here can move a vertex across a subpixel rounding boundary.
 rows=[(ca,f32(sa*sb),f32(sa*cb)),(0,cb,-sb),(-sa,f32(ca*sb),f32(ca*cb))]
 vertices=[tuple(f32(f32(f32(row[0]*x)+f32(row[1]*y))+f32(row[2]*z))
                 for row in rows) for x,y,z in vertices]
if a.frame%2:
 indices=[v for t in reversed([indices[i:i+3] for i in range(0,len(indices),3)]) for v in t]
greater=a.depth_states and a.frame//2==1
write_depth=not a.depth_states or a.frame//2<2
test_depth=a.mode!='quad' and (not a.depth_states or a.frame//2!=3)
# Raw attachment rows follow the GL viewport's positive Y scale. The mesh
# fixture uses glClear without glClearColor, hence default transparent black.
points=[(f32(f32(x+1)*256),f32(f32(y+1)*256),f32(f32(z+1)*.5)) for x,y,z in vertices]
# Rasterization snaps projected XY to 1/256 pixel before forming planes.
# Keep Z unquantized; snapping Z would hide depth interpolation errors.
# Positive halfway screen coordinates round upward, not ties-to-even.
points=[(math.floor(x*256+.5)/256,math.floor(y*256+.5)/256,z) for x,y,z in points]
expected=[clear]*(size*size);depth=[0. if greater else 1.]*(size*size);edge=[False]*(size*size)
for t in range(0,len(indices),3):
 ids=indices[t:t+3];v=[points[i] for i in ids];c=[colors[i] for i in ids]
 (ax,ay,az),(bx,by,bz),(cx,cy,cz)=v
 d=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
 if abs(d)<1e-12: continue
 # Boundary tolerance only: less than 1/64 pixel from a projected edge.
 lengths=[math.hypot(bx-cx,by-cy),math.hypot(cx-ax,cy-ay),math.hypot(ax-bx,ay-by)]
 for y in range(max(0,math.floor(min(ay,by,cy))-1),min(size,math.ceil(max(ay,by,cy))+1)):
  for x in range(max(0,math.floor(min(ax,bx,cx))-1),min(size,math.ceil(max(ax,bx,cx))+1)):
   u=((by-cy)*(x+.5-cx)+(cx-bx)*(y+.5-cy))/d
   w=((cy-ay)*(x+.5-cx)+(ax-cx)*(y+.5-cy))/d
   weights=(u,w,1-u-w);index=y*size+x
   distances=[q*abs(d)/length for q,length in zip(weights,lengths)]
   if min(distances)>=-1/64 and min(abs(q) for q in distances)<1/64: edge[index]=True
   if min(weights)<0: continue
   z=sum(q*vv[2] for q,vv in zip(weights,v))
   if a.depth_clipped_varyings:
    # Clip-plane coverage follows the affine screen-space depth plane.
    # Include its 1/64-pixel edge band in the usual boundary reporting.
    dzdx=((by-cy)*(az-cz)+(cy-ay)*(bz-cz))/d
    dzdy=((cx-bx)*(az-cz)+(ax-cx)*(bz-cz))/d
    clip_band=math.hypot(dzdx,dzdy)/64
    if min(abs(z),abs(z-1)) < clip_band: edge[index]=True
    if z<0 or z>1: continue
   if a.depth_states and a.frame//2==4:
    if ids[0]>=3: continue
   elif test_depth and (z<=depth[index] if greater else z>=depth[index]): continue
   if test_depth and write_depth: depth[index]=z
   color_weights=weights
   if a.perspective_varyings:
    weighted=[q/clip_w[i] for q,i in zip(weights,ids)]
    color_weights=[q/sum(weighted) for q in weighted]
   if a.interpolation:
    smooth=[q/clip_w[i] for q,i in zip(weights,ids)]
    smooth=[q/sum(smooth) for q in smooth]
    channel_weights = [weights]*3 if a.interpolation=='linear' else [(0,0,1)]*3 if a.interpolation in ('flat','integer') else [smooth,weights,(0,0,1)]
    expected[index]=tuple(round(255*max(0,min(1,sum(q*cc[k] for q,cc in zip(channel_weights[k],c))))) for k in range(3))+(255,)
    continue
   expected[index]=tuple(round(255*max(0,min(1,sum(q*cc[k] for q,cc in zip(color_weights,c))))) for k in range(3))+(255,)
raw=a.attachment.read_bytes();assert len(raw)==size*size*4
depth_raw=a.depth_attachment.read_bytes() if a.depth_attachment else None
if depth_raw is not None: assert len(depth_raw)==len(raw)
depth_errors=0;max_depth_error=0.
rgba=bytearray(len(raw));coverage_errors=color_errors=boundary_differences=max_error=covered=0;examples=[]
for y in range(size):
 for x in range(size):
  m=sum((((x>>b)&1)<<(2*b))|(((y>>b)&1)<<(2*b+1)) for b in range(6))
  off=(((y//64)*8+x//64)*4096+m)*4
  b,g,r,alpha=raw[off:off+4];actual=(r,g,b,alpha);i=y*size+x
  rgba[i*4:i*4+4]=bytes(actual);ref=expected[i];covered+=actual!=clear
  mismatch=(actual!=clear)!=(ref!=clear);err=max(abs(q-r) for q,r in zip(actual,ref))
  if depth_raw is not None and not edge[i]:
   actual_depth=struct.unpack_from('<f',depth_raw,off)[0]
   depth_err=abs(actual_depth-depth[i]);max_depth_error=max(max_depth_error,depth_err)
   depth_errors+=not math.isfinite(actual_depth) or depth_err>2e-6
  if edge[i] and (mismatch or err>1): boundary_differences+=1;continue
  coverage_errors+=mismatch;color_errors+=err>1;max_error=max(max_error,err)
  if (mismatch or err>1) and len(examples)<8: examples.append(dict(x=x,y=y,actual=actual,expected=ref))
report=dict(depth_errors=depth_errors,max_depth_error=max_depth_error,mode=a.mode,frame=a.frame,vertices=len(vertices),indices=len(indices),triangles=len(indices)//3,
            covered_pixels=covered,coverage_errors=coverage_errors,color_errors=color_errors,
            boundary_differences=boundary_differences,max_interior_channel_error=max_error,examples=examples,
            attachment_sha256=hashlib.sha256(raw).hexdigest())
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
def chunk(k,b):return struct.pack('>I',len(b))+k+b+struct.pack('>I',zlib.crc32(k+b)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',size,size,8,6,0,0,0))
png+=chunk(b'IDAT',zlib.compress(b''.join(b'\0'+rgba[y*size*4:(y+1)*size*4] for y in range(size))))+chunk(b'IEND',b'')
a.output.with_suffix('.png').write_bytes(png)
print(json.dumps(report));assert covered>0 and coverage_errors==color_errors==depth_errors==0
