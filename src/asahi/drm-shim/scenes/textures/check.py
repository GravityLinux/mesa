#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent texture/filter/mipmap/atlas oracle for two rendered frames."""
from pathlib import Path
import json, sys, os, math
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
blend = bool(os.getenv("T8132_TEXTURE_BLEND"))
rtt = bool(os.getenv("T8132_TEXTURE_RTT"))
linear = bool(os.getenv("T8132_TEXTURE_LINEAR"))
mixed = bool(os.getenv("T8132_GLES_TEXTURES_MIXED"))
for frame in range(2):
    raw = (root / f'render-{(frame*2+1) if rtt else frame:04d}-attachment-0.bin').read_bytes()
    errors = 0; maximum = 0; examples = []; rgba = bytearray()
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
            if repeat:
                assert not mixed and not linear and not rtt
                tx, ty = x//4, y//4
                expected = [round(17+1.5*(tx+ty)+7*frame),
                            round(23+1.5*(tx+ty)+11*frame),
                            31+tx+ty+13*frame, 64+tx+ty]
            if loops:
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
                lod = min(levels-1, level + (.5 if mip_linear else 0))
                expected = [round(17+19*lod+7*frame), round(23+11*lod+11*frame),
                            round(31+7*lod+13*frame), 255]
                if generate:
                    assert not mip_linear
                    bit = (min(x,mip_size-1) ^ min(y,mip_height-1)) & 1
                    value = bit if level == 0 else .5
                    expected = [round(20+180*value+7*frame), round(40+120*value+11*frame),
                                round(60+80*value+13*frame), 255]
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
                a = expected[3]/255
                expected = [round(expected[c]*a+[64,128,191][c]*(1-a)) for c in range(3)]+[255]
            error = max(abs(a-b) for a,b in zip(actual, expected)); maximum = max(maximum, error)
            if error > 1:
                errors += 1
                if len(examples) < 8: examples.append([x, y, expected, actual])
    reports.append(dict(frame=frame, checked_pixels=65536, errors=errors, max_error=maximum, examples=examples))
    (root / f'frame-{frame}.rgba').write_bytes(rgba)
(root/'validation.json').write_text(json.dumps(reports, indent=2)+'\n')
print(json.dumps(reports, indent=2))
assert len(list(root.glob('render-*-attachment-0.bin'))) == (4 if rtt else 2)
assert not any(r['errors'] for r in reports)
