/* SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <epoxy/egl.h>
#include <epoxy/gl.h>
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
int
main(void)
{
   EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (!eglInitialize(d, NULL, NULL) || !eglBindAPI(EGL_OPENGL_API))
      return 2;
   const EGLint at[] = {EGL_SURFACE_TYPE,
                        EGL_PBUFFER_BIT,
                        EGL_RENDERABLE_TYPE,
                        EGL_OPENGL_BIT,
                        EGL_RED_SIZE,
                        8,
                        EGL_GREEN_SIZE,
                        8,
                        EGL_BLUE_SIZE,
                        8,
                        EGL_ALPHA_SIZE,
                        8,
                        EGL_NONE};
   EGLConfig cfg;
   EGLint n;
   if (!eglChooseConfig(d, at, &cfg, 1, &n) || n != 1)
      return 2;
   EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, NULL);
   const EGLint dim[] = {EGL_WIDTH, 8, EGL_HEIGHT, 8, EGL_NONE};
   EGLSurface surf = eglCreatePbufferSurface(d, cfg, dim);
   if (!eglMakeCurrent(d, surf, surf, ctx))
      return 2;
   GLuint vs = shader(
      GL_VERTEX_SHADER,
      "#version 120\nvarying vec2 uv;void main(){gl_Position=gl_Vertex;uv=gl_Vertex.xy*.5+.5;}");
   uint8_t texels[256], got[256];
   for (unsigned i = 0; i < 64; i++)
      for (unsigned c = 0; c < 4; c++)
         texels[4 * i + c] = 17 + (i * 11 + c * 43) % 220;
   GLuint tex;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                texels);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   const float vertices[] = {-1, -1, 3, -1, -1, 3};
   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(2, GL_FLOAT, 0, vertices);
   glViewport(0, 0, 8, 8);
   glDisable(GL_DITHER);
   unsigned failures = 0;
   for (unsigned mode = 0; mode < 5; mode++) {
      char fs[1024];
      unsigned c = mode % 4;
      snprintf(
         fs, sizeof(fs),
         "#version 120\nuniform sampler2D t;varying vec2 uv;void main(){vec4 a=texture2D(t,uv);float first=a[%u]*.5;%s gl_FragColor=vec4(first,a.x*.25,a[%u]*.25,a.w*.5+a.y*.25);}",
         c,
         mode == 4 ? "vec4 b=texture2D(t,vec2(first,.5));first+=b.y*.25;" : "",
         c);
      GLuint f = shader(GL_FRAGMENT_SHADER, fs), p = glCreateProgram();
      glAttachShader(p, vs);
      glAttachShader(p, f);
      glLinkProgram(p);
      GLint ok;
      glGetProgramiv(p, GL_LINK_STATUS, &ok);
      if (!ok)
         return 2;
      glUseProgram(p);
      glUniform1i(glGetUniformLocation(p, "t"), 0);
      unsigned bad = 0;
      for (unsigned frame = 0; frame < 100; frame++) {
         glDrawArrays(GL_TRIANGLES, 0, 3);
         glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, got);
         if (glGetError() != GL_NO_ERROR)
            return 2;
         for (unsigned i = 0; i < 64; i++) {
            float expect[4] = {
               texels[4 * i + c] * .5f, texels[4 * i] * .25f,
               texels[4 * i + c] * .25f,
               texels[4 * i + 3] * .5f + texels[4 * i + 1] * .25f};
            if (mode == 4) {
               unsigned x = (unsigned)(texels[4 * i] * .5f / 255 * 8);
               expect[0] += texels[4 * (32 + x) + 1] * .25f;
            }
            for (unsigned k = 0; k < 4; k++)
               if (fabsf(got[4 * i + k] - expect[k]) > 1.1f) {
                  if (bad < 3)
                     fprintf(stderr,
                             "mode%u frame%u pixel%u c%u got%u expected%.2f\n",
                             mode, frame, i, k, got[4 * i + k], expect[k]);
                  bad++;
               }
         }
      }
      printf("mode=%u draws=100 mismatches=%u %s\n", mode, bad,
             bad ? "FAIL" : "PASS");
      printf("PIGLIT: {\"subtest\": {\"completion-texture-%u\": \"%s\"}}\n",
             mode, bad ? "fail" : "pass");
      failures += !!bad;
      glDeleteProgram(p);
      glDeleteShader(f);
   }
   return !!failures;
}
