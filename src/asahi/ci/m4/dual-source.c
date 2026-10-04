/* SPDX-License-Identifier: MIT
 * Own-source GL regression for dual-source output routing.
 * cc dual-source.c -o dual-source -lEGL -lGL -lm
 * dual_color_blend_by_location=true ./dual-source
 */
#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glcorearb.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static GLuint
shader(GLenum type, const char *text)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &text, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
      exit(1);
   }
   return s;
}

static int
check(const char *name, const float *expected)
{
   float pixels[4 * 4 * 4];
   glReadPixels(0, 0, 4, 4, GL_RGBA, GL_FLOAT, pixels);
   GLenum error = glGetError();
   int fail = error != GL_NO_ERROR;
   for (unsigned i = 0; i < 64; ++i)
      fail |= !isfinite(pixels[i]) || fabsf(pixels[i] - expected[i % 4]) > 0.003f;
   printf("%s %s: %.6f %.6f %.6f %.6f (GL error 0x%x)\n",
          fail ? "FAIL" : "PASS", name, pixels[0], pixels[1], pixels[2],
          pixels[3], error);
   return fail;
}

int
main(void)
{
   EGLDisplay d = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                        EGL_DEFAULT_DISPLAY, NULL);
   EGLint major, minor, count;
   EGLConfig config;
   const EGLint attrs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                          EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
   const EGLint context_attrs[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3,
      EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
      EGL_NONE};
   if (!eglInitialize(d, &major, &minor) || !eglBindAPI(EGL_OPENGL_API) ||
       !eglChooseConfig(d, attrs, &config, 1, &count) || count != 1) {
      fprintf(stderr, "EGL initialization failed: 0x%x\n", eglGetError());
      return 1;
   }
   EGLContext ctx = eglCreateContext(d, config, EGL_NO_CONTEXT, context_attrs);
   if (!eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx))
      return 1;
   printf("Renderer: %s; version: %s\n", glGetString(GL_RENDERER),
          glGetString(GL_VERSION));
   GLuint vao, fbo, tex[2];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glGenTextures(2, tex);
   for (unsigned i = 0; i < 2; ++i) {
      glBindTexture(GL_TEXTURE_2D, tex[i]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGBA,
                   GL_FLOAT, NULL);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                             GL_TEXTURE_2D, tex[i], 0);
   }
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 1;
   glViewport(0, 0, 4, 4);
   GLuint vs = shader(GL_VERTEX_SHADER,
      "#version 330 core\n"
      "void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);"
      "gl_Position = vec4(p * 2.0 - 1.0, 0, 1); }\n");
   const char *declarations[] = {
      "layout(location=0,index=0) out vec4 primary;\n"
      "layout(location=0,index=1) out vec4 secondary;\n",
      "out vec4 primary; out vec4 secondary;\n",
      "layout(location=0) out vec4 primary;\n"
      "layout(location=1) out vec4 secondary;\n",
   };
   const char *names[] = {"explicit-index", "indexed-binding", "by-location"};
   const float primary[] = {.2f, .3f, .4f, .5f};
   const float secondary[] = {.6f, .7f, .8f, .9f};
   const float expected[] = {.16f, .27f, .38f, .49f};
   const GLenum attachments[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
   int fails = 0;
   for (unsigned i = 0; i < 3; ++i) {
      char source[1024];
      snprintf(source, sizeof(source), "#version 330 core\n%s"
         "void main() { primary=vec4(.2,.3,.4,.5);"
         "secondary=vec4(.6,.7,.8,.9); }\n", declarations[i]);
      GLuint fs = shader(GL_FRAGMENT_SHADER, source);
      GLuint p = glCreateProgram();
      glAttachShader(p, vs);
      glAttachShader(p, fs);
      if (i == 1) {
         glBindFragDataLocationIndexed(p, 0, 0, "primary");
         glBindFragDataLocationIndexed(p, 0, 1, "secondary");
      }
      glLinkProgram(p);
      GLint ok;
      glGetProgramiv(p, GL_LINK_STATUS, &ok);
      if (!ok) {
         char log[4096];
         glGetProgramInfoLog(p, sizeof(log), NULL, log);
         fprintf(stderr, "%s: %s\n", names[i], log);
         return 1;
      }
      glUseProgram(p);
      glDrawBuffers(1, attachments);
      glClearColor(.1f, .2f, .3f, .4f);
      glClear(GL_COLOR_BUFFER_BIT);
      glEnable(GL_BLEND);
      glBlendFuncSeparate(GL_SRC1_COLOR, GL_ONE_MINUS_SRC1_COLOR,
                          GL_SRC1_ALPHA, GL_ONE_MINUS_SRC1_ALPHA);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glReadBuffer(GL_COLOR_ATTACHMENT0);
      fails += check(names[i], expected);

      /* Reuse the by-location program with the same factors but disabled
       * blending, then with ordinary blending. Location 1 must remain MRT1. */
      if (i == 2) {
         glDisable(GL_BLEND);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         fails += check("disabled-blend-primary", primary);
         glBlendFunc(GL_ONE, GL_ZERO);
         glDrawBuffers(2, attachments);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         glReadBuffer(GL_COLOR_ATTACHMENT1);
         fails += check("disabled-blend-MRT1", secondary);
         glEnable(GL_BLEND);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         fails += check("ordinary-blend-MRT1", secondary);
      }
      glDeleteProgram(p);
      glDeleteShader(fs);
   }
   glDeleteShader(vs);
   eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(d, ctx);
   eglTerminate(d);
   return fails != 0;
}
