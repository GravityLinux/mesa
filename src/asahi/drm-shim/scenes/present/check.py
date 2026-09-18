#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the HDMI stream at explicit EGL surface boundaries, never the device."""
import io
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import sysconfig
import urllib.request
from PIL import Image, ImageStat

binary, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=False)
root = Path(os.environ.get('ASAHI_WORKSPACE', '/home/nsheth/Projects/asahi'))
build = Path(os.environ.get('MESA_BUILD', root/'mesa-m1n1-build'))
m1n1 = Path(os.environ.get('M1N1_SHIM_ROOT', root/'m1n1-m4-agx'))
env = os.environ.copy()
env.update(PYTHONPATH=f'{m1n1}/proxyclient:{sysconfig.get_paths()["purelib"]}',
    PYTHONUNBUFFERED='1', M1N1DEVICE='/dev/m1n1',
    LD_PRELOAD=str(build/'src/asahi/drm-shim/libasahi_noop_drm_shim.so'),
    LD_LIBRARY_PATH=f'{build}/src/egl:{build}/src/gallium/targets/dri',
    __EGL_VENDOR_LIBRARY_FILENAMES=str(build/'src/egl/50_mesa.json'),
    MESA_LOADER_DRIVER_OVERRIDE='asahi', MESA_SHADER_CACHE_DISABLE='true',
    ASAHI_MESA_DEBUG='nocompress', EGL_PLATFORM='surfaceless',
    G16G_RENDER_SOURCE='1', G16G_EXECUTE_MESA_RENDER='1',
    G16G_MESA_RENDER_STAGE='direct',
    G16G_RENDER_PRESENT='tiled-bgra8', G16G_RENDER_PRESENT_ON_SWAP='1')
expected = {'red': (255,0,0), 'still-red': (255,0,0), 'blue': (0,0,255),
    'blue-repeat': (0,0,255), 'yellow-small': (255,255,0), 'blue-restored': (0,0,255), 'cyan-pending': (0,255,255), 'cyan-invalid-swap': (0,255,255)}
counts = {'initial':0,'hidden-red-green':0,'red':1,'still-red':1,'blue':2,
          'blue-repeat':3,'yellow-small':4,'blue-restored':5,'cyan-pending':6,'cyan-invalid-swap':6}
reports=[]
# Log is unbuffered at the OS pipe; read bytes to avoid TextIO read-ahead defeating select.
p = subprocess.Popen([str(binary)],env=env,stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
sel=selectors.DefaultSelector();sel.register(p.stdout,selectors.EVENT_READ)
pending=b''; full=''; initial=None
try:
    with (output/'run.log').open('w') as log:
        while True:
            if not sel.select(120): raise RuntimeError('fixture stalled before a checkpoint')
            data=os.read(p.stdout.fileno(),65536)
            if not data: break
            pending+=data
            while b'\n' in pending:
                raw,pending=pending.split(b'\n',1)
                line=raw.decode(errors='replace')+'\n'
                log.write(line);log.flush();full+=line
                if not line.startswith('PRESENT_READY '): continue
                name=line.split()[1]
                # Fetch several snapshots to allow the shared encoder to publish the latest HDMI frame.
                import time
                time.sleep(.25)
                with urllib.request.urlopen('http://localhost:9191/snapshot',timeout=10) as response:
                    jpeg=response.read()
                (output/f'{name}.jpg').write_bytes(jpeg)
                image=Image.open(io.BytesIO(jpeg)).convert('RGB')
                w,h=image.size
                center=ImageStat.Stat(image.crop((w//2-16,h//2-16,w//2+16,h//2+16))).mean
                if name=='initial': initial=center
                target=initial if name=='hidden-red-green' else expected.get(name)
                if target is not None:
                    # HDMI capture YUV conversion changes saturated channel levels.
                    tolerance=12 if name=='hidden-red-green' else 40
                    assert max(abs(a-b) for a,b in zip(center,target)) < tolerance, (name,center,target)
                if name=='yellow-small':
                    border=ImageStat.Stat(image.crop((w//2+90,h//2-16,w//2+110,h//2+16))).mean
                    assert max(abs(a-b) for a,b in zip(border,expected['blue']))<40, ('previous display contents lost',border)
                presentations=full.count('G16G_SWAP_PRESENT ')
                assert presentations==counts[name], (name,presentations,counts[name])
                assert full.count('G16G HDMI present:')==presentations
                reports.append(dict(step=name,center_rgb=center,presentations=presentations))
                print(name,center,presentations,flush=True)
                p.stdin.write(b'\n');p.stdin.flush()
    assert p.wait(timeout=20)==0
    assert 'PRESENT_PASS' in full and len(reports)==len(counts)
    (output/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
finally:
    if p.poll() is None:
        p.terminate()
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired: p.kill();p.wait()
