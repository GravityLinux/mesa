# GPU-less noop DRM shim

`libasahi_null_drm_shim.so` (built from `asahi_noop.c`, not installed)
accepts every ioctl without a GPU, so host builds can run the Asahi
Gallium driver through shader compilation, state emission and command
construction. Submissions are dropped, so rendered results and readbacks
are meaningless.

`ASAHI_NOOP_GPU` selects the reported GPU: `g13g` (default) or `g16g`
(T8132, the Apple9 paths). Example, from a build directory:

```sh
LD_PRELOAD=$PWD/src/asahi/drm-shim/libasahi_null_drm_shim.so \
ASAHI_NOOP_GPU=g16g EGL_PLATFORM=surfaceless \
__EGL_VENDOR_LIBRARY_FILENAMES=$PWD/src/egl/devenv_glvnd_egl.json \
LD_LIBRARY_PATH=$PWD/src/egl:$PWD/src/gallium/targets/dri \
<GL program>
```

`libasahi_noop_drm_shim.so` is a different shim: it forwards to real
hardware through m1n1 (`asahi_m1n1.c`, see README.m1n1-g16g.md).
