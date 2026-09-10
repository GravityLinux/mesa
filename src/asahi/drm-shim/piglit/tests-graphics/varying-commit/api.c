/* SPDX-License-Identifier: MIT */
#include <epoxy/egl.h>
#include <epoxy/gl.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GRID = 92, WIDTH = 128, HEIGHT = 112, MODES = 6 };
static GLuint shader(GLenum stage, const char *text)
{
   GLuint s = glCreateShader(stage);
   glShaderSource(s, 1, &text, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
      exit(2);
   }
   return s;
}

int main(void)
{
   EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (!eglInitialize(d, NULL, NULL) || !eglBindAPI(EGL_OPENGL_API)) return 2;
   const EGLint attr[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
      EGL_OPENGL_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8, EGL_NONE};
   EGLConfig cfg;
   EGLint n;
   if (!eglChooseConfig(d, attr, &cfg, 1, &n) || n != 1) return 2;
   EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, NULL);
   const EGLint dim[] = {EGL_WIDTH, 256, EGL_HEIGHT, 256, EGL_NONE};
   EGLSurface surface = eglCreatePbufferSurface(d, cfg, dim);
   if (!eglMakeCurrent(d, surface, surface, ctx)) return 2;
   const char *position_expr = "position";
   const char *expressions[MODES] = {"data", "ceil(data)", "-data",
      "vec4(lessThan(ivec4(data), ivec4(1)))", "(data * 2.0 + data) / 3.0", "data"};
   const char *names[MODES] = {"passthrough", "ceil", "negate", "integer-less", "arithmetic", "varying-clamped"};
   GLuint programs[MODES];
   GLuint fs = shader(GL_FRAGMENT_SHADER,
      "#version 120\nvarying vec4 color;void main(){gl_FragColor=color;}");
   for (unsigned m = 0; m < MODES; ++m) {
      char source[1024];
      snprintf(source, sizeof(source), "#version 120\nattribute vec4 position;"
         "attribute vec4 data;varying vec4 color;void main(){"
         "gl_Position=%s;color=%s;%s}", position_expr, expressions[m],
         m == 3 ? "color=vec4(.75+.25*color.x,.1*color.y,.1*color.z,1);" : "");
      GLuint vs = shader(GL_VERTEX_SHADER, source);
      programs[m] = glCreateProgram();
      glAttachShader(programs[m], vs);
      glAttachShader(programs[m], fs);
      glBindAttribLocation(programs[m], 0, "position");
      glBindAttribLocation(programs[m], 1, "data");
      glLinkProgram(programs[m]);
      GLint ok;
      glGetProgramiv(programs[m], GL_LINK_STATUS, &ok);
      if (!ok) return 2;
      glDeleteShader(vs);
   }

   const unsigned vertices = (GRID + 1) * (GRID + 1), count = GRID * GRID * 6;
   float *positions = malloc(vertices * 4 * sizeof(float));
   float *attributes = malloc(vertices * 4 * sizeof(float));
   uint16_t *indices = malloc(count * sizeof(uint16_t));
   uint8_t *pixels = malloc(WIDTH * HEIGHT * 4);
   if (!positions || !attributes || !indices || !pixels) return 2;
   for (unsigned y = 0; y <= GRID; ++y)
      for (unsigned x = 0; x <= GRID; ++x) {
         unsigned v = y * (GRID + 1) + x;
         positions[v * 4] = 2.f * x / GRID - 1.f;
         positions[v * 4 + 1] = 2.f * y / GRID - 1.f;
         positions[v * 4 + 2] = 0;
         positions[v * 4 + 3] = 1;
      }
   for (unsigned y = 0; y < GRID; ++y)
      for (unsigned x = 0; x < GRID; ++x) {
         unsigned at = (y * GRID + x) * 6, v = y * (GRID + 1) + x;
         indices[at] = v; indices[at + 1] = v + 1;
         indices[at + 2] = indices[at + 3] = v + GRID + 1;
         indices[at + 4] = v + 1; indices[at + 5] = v + GRID + 2;
      }
   glEnableVertexAttribArray(0); glEnableVertexAttribArray(1);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, positions);
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, attributes);
   glViewport(11, 19, WIDTH, HEIGHT);
   unsigned failed[MODES] = {0};
   unsigned frames = 1000;
   if (getenv("VARYING_COMMIT_FRAMES")) {
      char *end;
      unsigned long requested = strtoul(getenv("VARYING_COMMIT_FRAMES"), &end, 10);
      if (*end || requested == 0 || requested > 100000) return 2;
      frames = requested;
   }
   for (unsigned frame = 0; frame < frames; ++frame)
      for (unsigned m = 0; m < MODES; ++m) {
         for (unsigned v = 0; v < vertices; ++v)
            for (unsigned c = 0; c < 4; ++c) {
               float value = (c == 0 || c == 3) ? 1.f : 0.f;
               float coordinate = positions[v * 4 + (c & 1)];
               attributes[v * 4 + c] =
                  m == 5 ? (value ? 2.f : -2.f) + .25f * coordinate :
                  m == 2 ? -value :
                  m == 3 ? 2.f * coordinate + .1f * c : value;
            }
         glUseProgram(programs[m]);
         glClearColor(.125f, .25f, .5f, 1.f); glClear(GL_COLOR_BUFFER_BIT);
         glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_SHORT, indices);
         glReadPixels(11, 19, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
         unsigned bad = 0;
         for (unsigned p = 0; p < WIDTH * HEIGHT; ++p)
            /* The comparison varies per vertex. Its interpolated colors
             * stay in this red range, disjoint from the blue clear color. */
            bad += pixels[p * 4] < (m == 3 ? 190 : 254) ||
                   pixels[p * 4 + 1] > (m == 3 ? 27 : 1) ||
                   pixels[p * 4 + 2] > (m == 3 ? 27 : 1) || pixels[p * 4 + 3] < 254;
         if (glGetError() != GL_NO_ERROR) return 2;
         if (bad) {
            fprintf(stderr, "%s frame %u: %u bad pixels\n", names[m], frame, bad);
            failed[m]++;

         }
      }
   unsigned failures = 0;
   for (unsigned m = 0; m < MODES; ++m) {
      failures += failed[m];
      printf("PIGLIT: {\"subtest\": {\"varying-commit-%s\": \"%s\"}}\n",
             names[m], failed[m] ? "fail" : "pass");
   }
   printf("frames=%u draws=%u failed_frames=%u\n", frames, frames * MODES, failures);
   eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroySurface(d, surface); eglDestroyContext(d, ctx); eglTerminate(d);
   free(positions); free(attributes); free(indices); free(pixels);
   return failures ? 1 : 0;
}
