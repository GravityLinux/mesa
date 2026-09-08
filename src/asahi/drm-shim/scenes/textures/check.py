#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent texture/filter/mipmap/atlas oracle for two rendered frames."""
from pathlib import Path
import json, sys, os, math, struct
root = Path(sys.argv[1]); reports = []
atlas_w = int(os.getenv("T8132_TEXTURE_ATLAS", "0"))
atlas_h = int(os.getenv("T8132_TEXTURE_ATLAS_HEIGHT", str(atlas_w)))
channels = int(os.getenv("T8132_TEXTURE_CHANNELS", "4"))
mip_size = int(os.getenv("T8132_TEXTURE_MIPS", "0"))
mip_height = int(os.getenv("T8132_TEXTURE_MIP_HEIGHT", str(mip_size)))
generate = bool(os.getenv("T8132_TEXTURE_GENERATE_MIPS"))
mip_linear = bool(os.getenv("T8132_TEXTURE_MIP_LINEAR"))
loops = int(os.getenv("T8132_TEXTURE_LOOPS", "0"))
repeat = bool(os.getenv("T8132_TEXTURE_REPEAT"))
blend_mode = os.getenv("T8132_TEXTURE_BLEND_MODE")
blend = bool(os.getenv("T8132_TEXTURE_BLEND")) or bool(blend_mode)
rtt = bool(os.getenv("T8132_TEXTURE_RTT"))
linear = bool(os.getenv("T8132_TEXTURE_LINEAR"))
wrap = os.getenv("T8132_TEXTURE_WRAP")
mixed = bool(os.getenv("T8132_GLES_TEXTURES_MIXED"))
texture_format = os.getenv("T8132_TEXTURE_FORMAT")
stencil = os.getenv("T8132_TEXTURE_STENCIL")
extra_submission = rtt or stencil == "persist"
for frame in range(2):
    submission = frame*2+1 if extra_submission else frame
    raw = (root / f'render-{submission:04d}-attachment-0.bin').read_bytes()
    stencil_raw = (root / f'render-{submission:04d}-attachment-{2 if stencil == "zfail" else 1}.bin').read_bytes() if stencil else None
    stencil_depth_raw = (root / f'render-{submission:04d}-attachment-1.bin').read_bytes() if stencil == "zfail" else None
    stencil_errors = 0
    errors = 0; maximum = 0; examples = []; rgba = bytearray()
    depth_errors = 0
    depth_raw = (root / f'render-{frame:04d}-attachment-1.bin').read_bytes() if os.getenv("T8132_TEXTURE_DEPTH") else None
    for y in range(256):
        for x in range(256):
            m = sum((((x >> b) & 1) << (2*b)) | (((y >> b) & 1) << (2*b+1)) for b in range(6))
            at = ((y//64*4+x//64)*4096 + m)*4
            blue, green, red, alpha = raw[at:at+4]
            actual = [red, green, blue, alpha]; rgba.extend(actual)
            tx, ty = x//4, y//4
            expected = [17+3*tx+7*frame, 23+3*ty+11*frame, 31+tx+ty+13*frame, 64+tx+ty]
            if mixed:
                if 120 <= x < 136:
                    expected = [64, 128, 191, 255]
                else:
                    tx, ty = max(0, min(63, x//2 - 32)), max(0, min(63, y//2 - 32))
                    image = frame if x < 120 else 1-frame
                    expected = [17+3*tx+7*image, 23+3*ty+11*image,
                                31+tx+ty+13*image, 64+tx+ty]
                    if x >= 136:
                        expected[0] = round(expected[0] * .5)
                        expected[2] = round(expected[2] * .25)
            if linear and not (mixed and 120 <= x < 136):
                scale, offset = (2, -.5) if mixed else (1, 0)
                tx = max(0, min(63, ((x+.5)/256*scale+offset)*64-.5))
                ty = max(0, min(63, ((y+.5)/256*scale+offset)*64-.5))
                image = (frame if x < 120 else 1-frame) if mixed else frame
                expected = [17+3*tx+7*image, 23+3*ty+11*image,
                            31+tx+ty+13*image, 64+tx+ty]
                if mixed and x >= 136:
                    expected[0] *= .5
                    expected[2] *= .25
                expected = [round(v) for v in expected]
            if texture_format:
                def texel(i,j):
                    i,j=max(0,min(63,i)),max(0,min(63,j))
                    if texture_format == "srgb":
                        v=[255 if (i+frame)&1 else 0,192 if (j+frame)&1 else 16,128 if (i^j^frame)&1 else 32]
                        v=[c/255 for c in v]
                        return [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in v]+[(64+i+j)/255]
                    if texture_format == "rgb565":
                        return [((17+3*i+7*frame)>>3)/31,((23+3*j+11*frame)>>2)/63,((31+i+j+13*frame)>>3)/31,1]
                    assert texture_format == "rgba16f"
                    v=[-1+2*i/63+frame*.0625,.25+4*j/63,.00390625+(i+j)/128,.5+frame*.125]
                    return [struct.unpack('<e',struct.pack('<e',c))[0] for c in v]
                fx,fy=(x+.5)/4,(y+.5)/4
                if linear:
                    fx-=.5;fy-=.5
                    ix,iy=math.floor(fx),math.floor(fy);wx,wy=fx-ix,fy-iy
                    a,b,c,d=texel(ix,iy),texel(ix+1,iy),texel(ix,iy+1),texel(ix+1,iy+1)
                    sampled=[(a[k]*(1-wx)+b[k]*wx)*(1-wy)+(c[k]*(1-wx)+d[k]*wx)*wy for k in range(4)]
                else:
                    sampled=texel(math.floor(fx),math.floor(fy))
                if texture_format == "rgba16f":
                    sampled[0]=(sampled[0]+1)/2
                    sampled[1]/=8
                expected=[round(255*max(0,min(1,c))) for c in sampled]
            if wrap:
                assert not (mixed or atlas_w or mip_size or repeat or loops or rtt)
                def wrapped(i):
                    if wrap == "repeat": return i % 64
                    i %= 128
                    return i if i < 64 else 127-i
                def texel(i,j):
                    i,j=wrapped(i),wrapped(j)
                    return [17+3*i+7*frame,23+3*j+11*frame,31+i+j+13*frame,64+i+j]
                fx,fy=((x+.5)/256*3.25-1.3)*64,((y+.5)/256*3.25-1.3)*64
                if not linear:
                    expected=texel(math.floor(fx),math.floor(fy))
                else:
                    fx-=.5; fy-=.5
                    ix,iy=math.floor(fx),math.floor(fy); wx,wy=fx-ix,fy-iy
                    a,b,c,d=texel(ix,iy),texel(ix+1,iy),texel(ix,iy+1),texel(ix+1,iy+1)
                    expected=[round((a[k]*(1-wx)+b[k]*wx)*(1-wy)+(c[k]*(1-wx)+d[k]*wx)*wy) for k in range(4)]
            if repeat:
                assert not mixed and not linear and not rtt
                tx, ty = x//4, y//4
                expected = [round(17+1.5*(tx+ty)+7*frame),
                            round(23+1.5*(tx+ty)+11*frame),
                            31+tx+ty+13*frame, 64+tx+ty]
            if loops and os.getenv("T8132_TEXTURE_DISCARD") != "loop":
                assert loops % 2 == 1 and not mixed and not linear and not rtt
                tx, ty = (x//4, y//4) if (x < 128) == (frame == 0) else (y//4, x//4)
                mid = loops//2
                tx = sum(max(0,min(63,tx+i-mid)) for i in range(loops))/loops
                ty = sum(max(0,min(63,ty+j-mid)) for j in range(loops))/loops
                expected = [round(17+3*tx+7*frame), round(23+3*ty+11*frame),
                            round(31+tx+ty+13*frame), round(64+tx+ty)]
            if mip_size:
                levels = max(mip_size,mip_height).bit_length()
                level = min(levels-1, ((x+1)*levels-1)//256)
                bias = float(os.getenv("T8132_TEXTURE_BIAS", "0"))
                bias += float(os.getenv("T8132_TEXTURE_SAMPLER_BIAS", "0")) * (-1 if frame else 1)
                bias = math.floor(max(-32, min(2047/64, bias))*64)/64
                if os.getenv("T8132_TEXTURE_ANISO"):
                    bias -= math.log2(float(os.environ["T8132_TEXTURE_ANISO"]))
                base = int(os.getenv("T8132_TEXTURE_BASE", "0"))
                last = int(os.getenv("T8132_TEXTURE_LAST", str(levels-1)))
                lod = max(base, min(last, base + level + (.5 if mip_linear else 0) + bias))
                image = 1-frame if os.getenv("T8132_TEXTURE_FETCH_MIXED") and level & 1 else frame
                expected = [round(17+19*lod+7*image), round(23+11*lod+11*image),
                            round(31+7*lod+13*image), 255]
                if generate:
                    assert not mip_linear
                    bit = (min(x,mip_size-1) ^ min(y,mip_height-1)) & 1
                    value = bit if level == 0 else .5
                    pairs = [(20+7*frame,200+7*frame), (40+11*frame,160+11*frame),
                             (60+13*frame,140+13*frame)]
                    if texture_format == "rgb565":
                        pairs = [(16+16*frame,192+16*frame), (40+8*frame,160+8*frame),
                                 (56+16*frame,136+16*frame)]
                    def decode(v):
                        v /= 255
                        return v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4
                    expected = []
                    for c,(a,b) in enumerate(pairs):
                        if texture_format == "srgb":
                            v = decode(a)*(1-value)+decode(b)*value
                            if level:
                                encoded = 12.92*v if v <= .0031308 else 1.055*v**(1/2.4)-.055
                                v = decode(round(encoded*255))
                            expected.append(round(v*255))
                        elif texture_format == "rgb565":
                            bits = 6 if c == 1 else 5; top = (1<<bits)-1
                            v = ((a>>(8-bits))*(1-value)+(b>>(8-bits))*value)/top
                            expected.append(round(math.floor(v*top + .500001)/top*255))
                        else:
                            expected.append(round(a*(1-value)+b*value))
                    expected.append(255)
            if atlas_w:
                tx, ty = int((x+.5)*atlas_w/256), int((y+.5)*atlas_h/256)
                def texel(tx,ty):
                    tx,ty=max(0,min(atlas_w-1,tx)),max(0,min(atlas_h-1,ty))
                    values = [(17+tx*3+ty*5+c*47)&255 for c in range(channels)]
                    if frame and atlas_w//3+3 <= tx < atlas_w//3+100 and atlas_h//3+5 <= ty < atlas_h//3+88:
                        values = [211-37*c for c in range(channels)]
                    return values if channels==4 else [255,255,255,values[0]] if channels==1 else [values[0]]*3+[values[1]]
                expected = texel(tx,ty)
                if linear:
                    fx,fy=(x+.5)*atlas_w/256-.5,(y+.5)*atlas_h/256-.5
                    ix,iy=math.floor(fx),math.floor(fy); wx,wy=fx-ix,fy-iy
                    a,b,c,d=texel(ix,iy),texel(ix+1,iy),texel(ix,iy+1),texel(ix+1,iy+1)
                    expected=[round((a[k]*(1-wx)+b[k]*wx)*(1-wy)+(c[k]*(1-wx)+d[k]*wx)*wy) for k in range(4)]
            if blend:
                src = [v/255 for v in expected]; dst=[64/255,128/255,191/255, .6 if blend_mode in ("dstalpha","saturate") else 1]
                if not blend_mode:
                    expected = [round(255*(src[c]*src[3]+dst[c]*(1-src[3]))) for c in range(3)]+[255]
                else:
                    const=[.75 if frame else .25,.5,.25,.75]
                    def channel(c):
                        a,b=src[c],dst[c]
                        if blend_mode=="color": return b*(1-a)
                        if blend_mode=="multiply": return a*b
                        if blend_mode=="dstalpha": return a*dst[3]+b*(1-dst[3])
                        if blend_mode=="constant": return a*const[c]+b*(1-const[c])
                        if blend_mode=="saturate": return a*(1 if c==3 else min(src[3],1-dst[3]))+b
                        if blend_mode=="subtract": return a-b
                        if blend_mode=="reverse": return b-a
                        if blend_mode=="min": return min(a,b)
                        if blend_mode=="max": return max(a,b)
                        raise ValueError(blend_mode)
                    expected=[round(255*max(0,min(1,channel(c)))) for c in range(4)]
            if os.getenv("T8132_TEXTURE_SAMPLE_MASK") and x < 128:
                expected = [64,128,191,255]
            discard = os.getenv("T8132_TEXTURE_DISCARD")
            if discard and x < (64 if discard == "loop" and y < 128 else 128):
                expected = [64,128,191,153 if blend_mode in ("dstalpha", "saturate") else 255]
            cull = os.getenv("T8132_TEXTURE_CULL")
            if cull and (cull == "both" or ((cull == "back") == bool(frame))):
                expected = [64,128,191,255]
            depth = os.getenv("T8132_TEXTURE_DEPTH")
            if depth:
                z = .75 if frame else .5
                passes = {"less":z<.5,"lequal":z<=.5,"greater":z>.5,
                          "gequal":z>=.5,"equal":z==.5,"notequal":z!=.5,
                          "never":False,"always":True}[depth]
                if not passes: expected=[64,128,191,255]
            if depth_raw is not None:
                killed = bool(discard) and x < (64 if discard == "loop" and y < 128 else 128)
                expected_z = z if passes and not killed else .5
                actual_z = struct.unpack_from("<f", depth_raw, at)[0]
                depth_errors += not math.isclose(actual_z, expected_z, abs_tol=1e-6)
            if stencil_raw is not None:
                initial = 255 if frame else 0
                values = [initial, 0, 166, 255 if frame else 1,
                          254 if frame else 0, 0 if frame else 1,
                          254 if frame else 255, 255-initial]
                mask = 0x5a if frame else 255
                expected_s = (values[x//32] & mask) | (initial & (~mask & 255))
                if stencil.startswith("discard") and y < 128:
                    expected_s = initial
                if stencil == "compare":
                    expected_s = 0xa5 if frame else 0x5a
                    ref, value = (0xb0, 0xa0) if frame else (9, 10)
                    passes = [False, ref < value, ref == value, ref <= value,
                              ref > value, ref != value, ref >= value, True][x//32]
                    if not passes: expected = [64, 128, 191, 255]
                if stencil_depth_raw is not None:
                    depth_errors += not math.isclose(struct.unpack_from("<f", stencil_depth_raw, at)[0], .5, abs_tol=1e-6)
                morton = sum((((x >> b) & 1) << (2*b)) | (((y >> b) & 1) << (2*b+1)) for b in range(7))
                offset = ((y//128)*2 + x//128)*16384 + morton
                stencil_errors += stencil_raw[offset] != expected_s
            error = max(abs(a-b) for a,b in zip(actual, expected)); maximum = max(maximum, error)
            if error > 1:
                errors += 1
                if len(examples) < 8: examples.append([x, y, expected, actual])
    reports.append(dict(frame=frame, checked_pixels=65536, errors=errors, depth_errors=depth_errors, stencil_errors=stencil_errors, max_error=maximum, examples=examples))
    (root / f'frame-{frame}.rgba').write_bytes(rgba)
(root/'validation.json').write_text(json.dumps(reports, indent=2)+'\n')
print(json.dumps(reports, indent=2))
assert len(list(root.glob('render-*-attachment-0.bin'))) == (4 if extra_submission else 2)
assert not any(r['errors'] or r['depth_errors'] or r['stencil_errors'] for r in reports)
