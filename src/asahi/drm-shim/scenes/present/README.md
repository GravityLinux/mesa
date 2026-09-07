# Present the application's EGL surface

`G16G_RENDER_PRESENT_ON_SWAP=1` with `G16G_RENDER_PRESENT=tiled-bgra8` enables
explicit presentation for the m1n1-backed Asahi surfaceless and device-platform EGL pbuffer paths.
It suppresses the old attachment-0 display hook after individual submissions.
With the flag unset, the existing submission-based demo mode is unchanged.

The opt-in pbuffer `eglSwapBuffers` path flushes rendering and selects the
surface's actual DRI image. Mesa's single-buffered pbuffer uses the loader's
**front** image internally. Gallium's `flush_frontbuffer` receives that live
resource; the Asahi callback passes its DRM-file-scoped BO handle, dimensions,
stride and offset through the embedded Python bridge. The shim resolves its
current VM binding and copies pixels to HDMI entirely on the target. It does
not export the BO, read pixels back to the host, or infer the final target from
submission order. The EGL surface retains the resource for the synchronous
call. The current m1n1 transport completes submissions synchronously; an async
transport would also need a completion wait before display.

The private Asahi callback uses the out-of-band drawable handle as an integer
status output. This bridge is opt-in bring-up plumbing, not a kernel ioctl or
public EGL extension. Only plain pbuffer `eglSwapBuffers` on the surfaceless/device platforms is enabled; ordinary EGL
pbuffer no-op semantics remain in place without the flag. Current presentation
supports level-zero, layer-zero uncompressed tiled BGRA8/BGRX8 resources. A
change in surface dimensions clears the previous image's border on HDMI.

## SDL/LÖVE

SDL's offscreen backend may make `SDL_GL_SwapWindow` a no-op. Build
`sdl-present.c` as a small preload alongside the existing DRM shim:

```sh
cc -shared -fPIC -O2 sdl-present.c -o sdl-present.so -ldl -lEGL -lGLESv2
```

With swap mode enabled, it checks that the supplied SDL window owns the current
EGL context, then calls `eglSwapBuffers` for `eglGetCurrentSurface(EGL_DRAW)`.
It does not need the old unconditional `glFinish`. Optional
`APPLE9_TRACE_GAME_PRESENT=1` logs application boundary begin/end markers;
`G16G_SWAP_PRESENT` logs the selected BO, frame count and display time. The local
LÖVE launcher in `tmp/apple9-vertex-bindings/run-love.sh` enables this mode.

## Hardware regression

Reset and chainload display-enabled m1n1 before each GPU process. Keep the
shared uStreamer at `http://localhost:9191` running. From the workspace root:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror \
  mesa-m1n1-shim/src/asahi/drm-shim/scenes/present/present.c \
  -o /absolute/path/to/present -lEGL -lGLESv2
uv run --python 3.14 --with-requirements m1n1-m4-agx/requirements.txt \
  --with pillow python \
  mesa-m1n1-shim/src/asahi/drm-shim/scenes/present/check.py \
  /absolute/path/to/present /absolute/path/to/new-output
```

The controller captures uStreamer snapshots at ten acknowledged checkpoints:
no display before swap, FBO rendered last and still bound at swap, hidden new
rendering, repeated swap without drawing, a smaller second surface and cleared
border, restoration of the original surface after destroying the second,
pending rendering flushed by swap, and rejection of a destroyed surface.
Set `PRESENT_TEST_DEVICE=1` to repeat through EGL device enumeration (the SDL
offscreen route), instead of the default surfaceless platform.
It requires exactly six display operations, all at valid swaps. Primary-color
checks tolerate HDMI capture YUV conversion; unchanged-display checks use a
smaller tolerance. It never accesses the capture device directly.

Python tests in `tests/python/test_agx_present.py` also cover handle reuse,
invalid extents/stride, repeated-page bindings, and missing live mappings.
