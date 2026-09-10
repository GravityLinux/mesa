/* SPDX-License-Identifier: MIT
 * MRT API regression: independent attachments, holes, remapping and blending.
 * Uses EGL/epoxy so it can run on the hardware shim and a software reference.
 */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <epoxy/egl.h>
#include <epoxy/gl.h>

#define WIDTH   67
#define HEIGHT  73
#define TARGETS 8
static GLuint textures[TARGETS], framebuffer;
static unsigned cases, failures;
static const unsigned char initial[4] = {17, 34, 51, 68};

static void
fail(const char *message)
{
   fprintf(stderr, "%s (GL=%#x, EGL=%#x)\n", message, glGetError(),
           eglGetError());
   exit(1);
}

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
      fprintf(stderr, "%s\n%s\n", source, log);
      fail("shader compilation failed");
   }
   return result;
}

/* mode: distinct constants, legacy broadcast, sampled source, or discard. */
static GLuint
program(unsigned count, unsigned mode)
{
   const char *vs = "#version 120\nvoid main() { gl_Position = gl_Vertex; }\n";
   char fs[4096] = "#version 120\nuniform sampler2D source;\nvoid main() {\n";
   if (mode == 3)
      strcat(fs, "if (gl_FragCoord.x < 33.0) discard;\n");
   if (mode == 1) {
      strcat(fs, "gl_FragColor = vec4(0.25,0.5,0.75,1.0);\n");
   } else {
      for (unsigned rt = 0; rt < count; ++rt) {
         char line[256];
         snprintf(line, sizeof(line),
                  "gl_FragData[%u] = vec4(%.8f,%.8f,0.25,1.0)%s;\n", rt,
                  (rt + 1) / 16.0, (8 - rt) / 16.0,
                  mode == 2 ? " + texture2D(source, vec2(0.5))" : "");
         strcat(fs, line);
      }
   }
   strcat(fs, "}\n");
   GLuint v = shader(GL_VERTEX_SHADER, vs), f = shader(GL_FRAGMENT_SHADER, fs);
   GLuint p = glCreateProgram();
   glAttachShader(p, v);
   glAttachShader(p, f);
   glLinkProgram(p);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok)
      fail("program link failed");
   glDeleteShader(v);
   glDeleteShader(f);
   glUseProgram(p);
   GLint source = glGetUniformLocation(p, "source");
   if (source >= 0)
      glUniform1i(source, 8);
   return p;
}

static void
reset_targets(void)
{
   unsigned char pixels[WIDTH * HEIGHT * 4];
   for (unsigned i = 0; i < WIDTH * HEIGHT; ++i)
      memcpy(pixels + i * 4, initial, 4);
   glActiveTexture(GL_TEXTURE0);
   for (unsigned rt = 0; rt < TARGETS; ++rt) {
      glBindTexture(GL_TEXTURE_2D, textures[rt]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, pixels);
      glColorMaskIndexedEXT(rt, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      glDisableIndexedEXT(GL_BLEND, rt);
   }
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);
}

static void
draw(void)
{
   static const float vertices[] = {-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1};
   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(2, GL_FLOAT, 0, vertices);
   glDrawArrays(GL_TRIANGLES, 0, 6);
   glDisableClientState(GL_VERTEX_ARRAY);
}

static bool
probe(unsigned target, const float color[4], bool discarded)
{
   unsigned char pixels[WIDTH * HEIGHT * 4];
   glReadBuffer(GL_COLOR_ATTACHMENT0_EXT + target);
   glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         for (unsigned c = 0; c < 4; ++c) {
            int expected = discarded && x < 33
                              ? initial[c]
                              : lroundf(fminf(fmaxf(color[c], 0), 1) * 255);
            int observed = pixels[4 * (y * WIDTH + x) + c];
            if (abs(expected - observed) > 1) {
               fprintf(stderr, "RT%u (%u,%u) channel %u: expected %d, got %d\n",
                       target, x, y, c, expected, observed);
               return false;
            }
         }
      }
   }
   return true;
}

static void
result(const char *name, bool ok)
{
   GLenum error = glGetError();
   if (error != GL_NO_ERROR) {
      fprintf(stderr, "%s: GL error %#x\n", name, error);
      ok = false;
   }
   ++cases;
   failures += !ok;
   printf("PIGLIT: {\"subtest\": {\"%s\": \"%s\"}}\n", name,
          ok ? "pass" : "fail");
   fflush(stdout);
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (!eglInitialize(display, NULL, NULL) || !eglBindAPI(EGL_OPENGL_API))
      fail("EGL initialization failed");
   const EGLint attributes[] = {EGL_SURFACE_TYPE,
                                EGL_PBUFFER_BIT,
                                EGL_RENDERABLE_TYPE,
                                EGL_OPENGL_BIT,
                                EGL_RED_SIZE,
                                8,
                                EGL_GREEN_SIZE,
                                8,
                                EGL_BLUE_SIZE,
                                8,
                                EGL_NONE};
   EGLConfig config;
   EGLint count;
   if (!eglChooseConfig(display, attributes, &config, 1, &count) || count != 1)
      fail("EGL configuration failed");
   EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
   const EGLint dimensions[] = {EGL_WIDTH, WIDTH, EGL_HEIGHT, HEIGHT, EGL_NONE};
   EGLSurface surface = eglCreatePbufferSurface(display, config, dimensions);
   if (!eglMakeCurrent(display, surface, surface, context))
      fail("EGL context failed");
   printf("Renderer: %s\n", glGetString(GL_RENDERER));
   GLint targets;
   glGetIntegerv(GL_MAX_DRAW_BUFFERS, &targets);
   if (targets < TARGETS || !epoxy_has_gl_extension("GL_EXT_draw_buffers2"))
      fail("eight targets and indexed color masks are required");
   glDisable(GL_DITHER);
   glViewport(0, 0, WIDTH, HEIGHT);
   glGenTextures(TARGETS, textures);
   glGenFramebuffersEXT(1, &framebuffer);
   reset_targets();
   for (unsigned rt = 0; rt < TARGETS; ++rt)
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                                GL_COLOR_ATTACHMENT0_EXT + rt, GL_TEXTURE_2D,
                                textures[rt], 0);
   if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) !=
       GL_FRAMEBUFFER_COMPLETE_EXT)
      fail("incomplete framebuffer");

   GLuint source_texture;
   glActiveTexture(GL_TEXTURE8);
   glGenTextures(1, &source_texture);
   glBindTexture(GL_TEXTURE_2D, source_texture);
   const unsigned char source_pixel[4] = {16, 32, 48, 0};
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                source_pixel);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   for (unsigned mode = 0; mode < 6; ++mode) {
      reset_targets();
      GLenum buffers[TARGETS];
      for (unsigned rt = 0; rt < TARGETS; ++rt)
         buffers[rt] = mode == 4 && !(rt & 1) ? GL_NONE
                                              : GL_COLOR_ATTACHMENT0_EXT +
                                                   (mode == 5 ? 7 - rt : rt);
      glDrawBuffers(TARGETS, buffers);
      GLuint p = program(TARGETS, mode);
      draw();
      bool ok = true;
      for (unsigned rt = 0; rt < TARGETS; ++rt) {
         float color[4] = {(rt + 1) / 16.0f, (8 - rt) / 16.0f, .25f, 1};
         if (mode == 1) {
            const float broadcast[] = {.25f, .5f, .75f, 1};
            memcpy(color, broadcast, sizeof(color));
         }
         if (mode == 2) {
            for (unsigned c = 0; c < 4; ++c)
               color[c] += source_pixel[c] / 255.0f;
         }
         if (mode == 4 && !(rt & 1)) {
            for (unsigned c = 0; c < 4; ++c)
               color[c] = initial[c] / 255.0f;
         }
         ok &= probe(mode == 5 ? 7 - rt : rt, color, mode == 3);
      }
      const char *names[] = {"large-eight",    "legacy-broadcast-eight",
                             "textured-eight", "discard-eight",
                             "sparse-eight",   "remapped-eight"};
      result(names[mode], ok);
      glDeleteProgram(p);
   }

   reset_targets();
   GLenum buffers[TARGETS];
   for (unsigned rt = 0; rt < TARGETS; ++rt)
      buffers[rt] = GL_COLOR_ATTACHMENT0_EXT + rt;
   glDrawBuffers(TARGETS, buffers);
   GLuint p = program(TARGETS, 1);
   glBlendFunc(GL_ONE, GL_ONE);
   for (unsigned rt = 0; rt < TARGETS; ++rt) {
      glEnableIndexedEXT(GL_BLEND, rt);
      glColorMaskIndexedEXT(rt, (rt & 3) == 0, (rt & 3) == 1, (rt & 3) == 2,
                            (rt & 3) == 3);
   }
   draw();
   bool ok = true;
   for (unsigned rt = 0; rt < TARGETS; ++rt) {
      const float add[] = {.25f, .5f, .75f, 1};
      float color[4];
      for (unsigned c = 0; c < 4; ++c)
         color[c] = initial[c] / 255.0f + ((rt & 3) == c ? add[c] : 0);
      ok &= probe(rt, color, false);
   }
   result("indexed-blend-and-masks-eight", ok);
   glDeleteProgram(p);
   glDeleteFramebuffersEXT(1, &framebuffer);
   glDeleteTextures(TARGETS, textures);
   glDeleteTextures(1, &source_texture);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroySurface(display, surface);
   eglDestroyContext(display, context);
   eglTerminate(display);
   printf("MRT_API_RESULT cases=%u failures=%u\n", cases, failures);
   return failures ? 1 : 0;
}
