/* SPDX-License-Identifier: MIT */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <epoxy/egl.h>
#include <epoxy/gl.h>
static bool finish_each;
static unsigned failures;
static GLuint
shader(GLenum stage, const char *src)
{
   GLuint s = glCreateShader(stage);
   glShaderSource(s, 1, &src, NULL);
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
static GLuint
program(bool mix)
{
   const char *vs =
      "#version 120\nvarying vec2 uv;void main(){gl_Position=gl_Vertex;uv=gl_Vertex.xy*0.5+0.5;}";
   const char *fs =
      mix
         ? "#version 120\nuniform sampler2D a,b;varying vec2 uv;void main(){gl_FragColor=(texture2D(a,uv)+texture2D(b,uv))*0.5;}"
         : "#version 120\nuniform sampler2D a;varying vec2 uv;void main(){gl_FragColor=texture2D(a,uv);}";
   GLuint v = shader(GL_VERTEX_SHADER, vs), f = shader(GL_FRAGMENT_SHADER, fs),
          p = glCreateProgram();
   glAttachShader(p, v);
   glAttachShader(p, f);
   glLinkProgram(p);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok)
      exit(2);
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "a"), 0);
   if (mix)
      glUniform1i(glGetUniformLocation(p, "b"), 1);
   glDeleteShader(v);
   glDeleteShader(f);
   return p;
}
static void
draw(void)
{
   const GLfloat xy[] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(2, GL_FLOAT, 0, xy);
   glDrawArrays(GL_TRIANGLES, 0, 6);
   glDisableClientState(GL_VERTEX_ARRAY);
   if (finish_each)
      glFinish();
}
static void
bind(unsigned unit, GLuint t)
{
   glActiveTexture(GL_TEXTURE0 + unit);
   glBindTexture(GL_TEXTURE_2D, t);
}
static GLuint
texture(unsigned r, unsigned g, unsigned b)
{
   GLuint t;
   glGenTextures(1, &t);
   bind(0, t);
   unsigned char p[32 * 32 * 4];
   for (unsigned i = 0; i < 32 * 32; i++) {
      p[4 * i] = r;
      p[4 * i + 1] = g;
      p[4 * i + 2] = b;
      p[4 * i + 3] = 255;
   }
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                p);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   return t;
}
static bool
pixel(int x, int y, int r, int g, int b)
{
   unsigned char p[4];
   glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
   bool ok = abs(p[0] - r) <= 1 && abs(p[1] - g) <= 1 && abs(p[2] - b) <= 1;
   if (!ok)
      fprintf(stderr, "pixel %d,%d: %u %u %u expected %d %d %d\n", x, y, p[0],
              p[1], p[2], r, g, b);
   return ok;
}
static void
result(const char *name, bool ok)
{
   ok &= glGetError() == GL_NO_ERROR;
   failures += !ok;
   printf("PIGLIT: {\"subtest\": {\"%s\": \"%s\"}}\n", name,
          ok ? "pass" : "fail");
   fflush(stdout);
}
int
main(int argc, char **argv)
{
   for (int i = 1; i < argc; i++)
      finish_each |= !strcmp(argv[i], "--finish");
   EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (!eglInitialize(d, NULL, NULL) || !eglBindAPI(EGL_OPENGL_API))
      return 2;
   EGLint attrs[] = {EGL_SURFACE_TYPE,
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
   EGLConfig cfg;
   EGLint n;
   if (!eglChooseConfig(d, attrs, &cfg, 1, &n) || n != 1)
      return 2;
   EGLContext c = eglCreateContext(d, cfg, EGL_NO_CONTEXT, NULL);
   EGLint dims[] = {EGL_WIDTH, 32, EGL_HEIGHT, 32, EGL_NONE};
   EGLSurface s = eglCreatePbufferSurface(d, cfg, dims);
   if (!eglMakeCurrent(d, s, s, c))
      return 2;
   GLuint red = texture(255, 0, 0), green = texture(0, 255, 0),
          single = program(false), mix = program(true);
   /* Each draw must retain the texture binding present when it was queued. */
   glUseProgram(single);
   glViewport(0, 0, 16, 32);
   bind(0, red);
   draw();
   glViewport(16, 0, 16, 32);
   bind(0, green);
   draw();
   result("rebind-two-draws",
          pixel(8, 16, 255, 0, 0) & pixel(24, 16, 0, 255, 0));
   glUseProgram(mix);
   glViewport(0, 0, 32, 32);
   bind(0, red);
   bind(1, green);
   draw();
   result("two-samplers", pixel(16, 16, 128, 128, 0));
   GLuint a = texture(0, 0, 0), b = texture(0, 0, 0), fbo;
   /* GPU writes to independent textures must remain visible to later draws. */
   glGenFramebuffersEXT(1, &fbo);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                             GL_TEXTURE_2D, a, 0);
   glUseProgram(single);
   bind(0, red);
   draw();
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                             GL_TEXTURE_2D, b, 0);
   bind(0, green);
   draw();
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   glUseProgram(mix);
   bind(0, a);
   bind(1, b);
   draw();
   result("two-rendered-textures", pixel(16, 16, 128, 128, 0));
   /* Black matches the original upload but differs from the unread GPU red.
    * A transport comparing only against an old CPU shadow loses this write. */
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                             GL_TEXTURE_2D, a, 0);
   glUseProgram(single);
   bind(0, red);
   draw();
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   bind(0, a);
   unsigned char black[32 * 32 * 4] = {0};
   for (unsigned i = 0; i < 32 * 32; i++)
      black[4 * i + 3] = 255;
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 32, 32, GL_RGBA, GL_UNSIGNED_BYTE,
                   black);
   draw();
   result("rewrite-old-cpu-value-over-gpu-output", pixel(16, 16, 0, 0, 0));
   /* Updating one region while an earlier draw reads the texture can move
    * it to new storage. The untouched region must retain the GPU's pixels. */
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                             GL_TEXTURE_2D, a, 0);
   bind(0, red);
   draw();
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   glViewport(0, 0, 16, 32);
   bind(0, a);
   draw();
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 32, GL_RGBA, GL_UNSIGNED_BYTE,
                   black);
   glViewport(16, 0, 16, 32);
   draw();
   result("partial-update-preserves-gpu-data-with-pending-reader",
          pixel(8, 16, 255, 0, 0) & pixel(20, 16, 0, 0, 0) &
          pixel(28, 16, 255, 0, 0));
   eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroySurface(d, s);
   eglDestroyContext(d, c);
   eglTerminate(d);
   return failures ? 1 : 0;
}
