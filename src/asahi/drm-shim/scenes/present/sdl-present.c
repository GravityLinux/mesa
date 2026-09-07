/* SPDX-License-Identifier: MIT */
/* SDL's offscreen video backend may implement swap as a no-op. Route the
 * explicit application boundary through our opt-in EGL pbuffer presenter. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void SDL_GL_SwapWindow(void *window)
{
   const char *enabled = getenv("G16G_RENDER_PRESENT_ON_SWAP");
   if (enabled && strcmp(enabled, "1") == 0) {
      static void *(*current_window)(void);
      if (!current_window)
         current_window = dlsym(RTLD_NEXT, "SDL_GL_GetCurrentWindow");
      if (!current_window || current_window() != window) {
         fputs("m1n1 SDL present requires the window's current EGL context\n", stderr);
         abort();
      }
      if (getenv("APPLE9_TRACE_GAME_PRESENT"))
         fputs("APPLE9_APP_PRESENT_BEGIN\n", stderr);
      EGLDisplay display = eglGetCurrentDisplay();
      EGLSurface surface = eglGetCurrentSurface(EGL_DRAW);
      if (display == EGL_NO_DISPLAY || surface == EGL_NO_SURFACE ||
          !eglSwapBuffers(display, surface)) {
         fprintf(stderr, "m1n1 SDL present failed: EGL=%#x\n", eglGetError());
         abort();
      }
      if (getenv("APPLE9_TRACE_GAME_PRESENT"))
         fputs("APPLE9_APP_PRESENT_END\n", stderr);
      return;
   }
   static void (*next)(void *);
   if (!next) next = dlsym(RTLD_NEXT, "SDL_GL_SwapWindow");
   glFinish();
   next(window);
}
