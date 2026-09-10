/* SPDX-License-Identifier: MIT */
#include <epoxy/egl.h>
#include <epoxy/gl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Small adjacent triangles expose lost vertex fetches after a CPU read of a
 * device buffer. Test the same geometry before and after read/write access;
 * readback must neither change the input bytes nor affect subsequent draws.
 * Exercise client vertices and indices independently as well as together. */
enum { GRID = 92, WIDTH = 128, HEIGHT = 112, FRAMES = 20 };

static GLuint
shader(GLenum stage, const char *source)
{
   GLuint result = glCreateShader(stage);
   glShaderSource(result, 1, &source, NULL);
   glCompileShader(result);
   GLint ok;
   glGetShaderiv(result, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(result, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
      exit(2);
   }
   return result;
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (!eglInitialize(display, NULL, NULL) || !eglBindAPI(EGL_OPENGL_API))
      return 2;
   const EGLint attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, attributes, &config, 1, &count) || count != 1)
      return 2;
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
   const EGLint dimensions[] = {EGL_WIDTH, 256, EGL_HEIGHT, 256, EGL_NONE};
   EGLSurface surface = eglCreatePbufferSurface(display, config, dimensions);
   if (!eglMakeCurrent(display, surface, surface, context))
      return 2;

   GLuint vs = shader(GL_VERTEX_SHADER,
      "#version 120\nattribute vec4 position;attribute vec4 data;"
      "varying vec4 color;void main(){gl_Position=position;color=ceil(data);}");
   GLuint fs = shader(GL_FRAGMENT_SHADER,
      "#version 120\nvarying vec4 color;void main(){gl_FragColor=color;}");
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glBindAttribLocation(program, 0, "position");
   glBindAttribLocation(program, 1, "data");
   glLinkProgram(program);
   GLint linked;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked)
      return 2;
   glUseProgram(program);

   const unsigned vertices = (GRID + 1) * (GRID + 1);
   const unsigned indices_count = GRID * GRID * 6;
   const unsigned bytes = vertices * 8 * sizeof(float);
   float *data = calloc(1, bytes), *copy = malloc(bytes);
   uint16_t *indices = malloc(indices_count * sizeof(uint16_t));
   uint8_t *pixels = malloc(WIDTH * HEIGHT * 4);
   if (!data || !copy || !indices || !pixels)
      return 2;
   for (unsigned y = 0; y <= GRID; ++y) {
      for (unsigned x = 0; x <= GRID; ++x) {
         float *v = data + (y * (GRID + 1) + x) * 8;
         v[0] = 2.f * x / GRID - 1.f;
         v[1] = 2.f * y / GRID - 1.f;
         v[3] = v[4] = v[7] = 1.f;
      }
   }
   for (unsigned y = 0; y < GRID; ++y) {
      for (unsigned x = 0; x < GRID; ++x) {
         unsigned at = (y * GRID + x) * 6, v = y * (GRID + 1) + x;
         indices[at] = v;
         indices[at + 1] = v + 1;
         indices[at + 2] = indices[at + 3] = v + GRID + 1;
         indices[at + 4] = v + 1;
         indices[at + 5] = v + GRID + 2;
      }
   }
   GLuint ibo;
   glGenBuffers(1, &ibo);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_count * sizeof(uint16_t),
                indices, GL_STATIC_DRAW);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glViewport(11, 19, WIDTH, HEIGHT);

   const char *names[] = {"unchanged", "read-only", "rewrite",
                          "client-vertices", "client-indices", "client-both"};
   unsigned failures = 0;
   for (unsigned mode = 0; mode < sizeof(names) / sizeof(names[0]); ++mode) {
      bool client_vertices = mode == 3 || mode == 5;
      bool client_indices = mode == 4 || mode == 5;
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, client_indices ? 0 : ibo);
      GLuint vbo;
      glGenBuffers(1, &vbo);
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferData(GL_ARRAY_BUFFER, bytes, data, GL_STREAM_DRAW);
      glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), NULL);
      glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                            (void *)(4 * sizeof(float)));
      if (client_vertices) {
         glBindBuffer(GL_ARRAY_BUFFER, 0);
         glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), data);
         glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), data + 4);
      }
      bool pass = true;
      for (unsigned frame = 0; frame < FRAMES; ++frame) {
         if (frame && mode == 1) {
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, bytes, copy);
            if (memcmp(copy, data, bytes)) {
               fprintf(stderr, "Input bytes changed after read-only access\n");
               pass = false;
            }
         } else if (frame && mode == 2) {
            glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data);
         }
         glClearColor(.125f, .25f, .5f, 1.f);
         glClear(GL_COLOR_BUFFER_BIT);
         glDrawElements(GL_TRIANGLES, indices_count, GL_UNSIGNED_SHORT,
                        client_indices ? indices : NULL);
         glReadPixels(11, 19, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
         unsigned bad = 0;
         for (unsigned p = 0; p < WIDTH * HEIGHT; ++p) {
            bad += pixels[p * 4] < 254 || pixels[p * 4 + 1] > 1 ||
                   pixels[p * 4 + 2] > 1 || pixels[p * 4 + 3] < 254;
         }
         if (bad) {
            fprintf(stderr, "%s frame %u: %u wrong pixels\n", names[mode], frame, bad);
            pass = false;
         }
         if (glGetError() != GL_NO_ERROR)
            return 2;
      }
      printf("PIGLIT: {\"subtest\": {\"buffer-readback-%s\": \"%s\"}}\n",
             names[mode], pass ? "pass" : "fail");
      failures += !pass;
      glDeleteBuffers(1, &vbo);
   }
   glDeleteBuffers(1, &ibo);
   glDeleteProgram(program);
   glDeleteShader(vs);
   glDeleteShader(fs);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroySurface(display, surface);
   eglDestroyContext(display, context);
   eglTerminate(display);
   free(data);
   free(copy);
   free(indices);
   free(pixels);
   return failures ? 1 : 0;
}
