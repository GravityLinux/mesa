/* Line stipple readback probe (GL compatibility profile, surfaceless EGL).
 * Each case draws one 1-pixel GL_LINES segment with glLineStipple in a 64x64
 * FBO and prints, along the major axis, which columns (rows) are lit. Run it
 * against a reference driver and compare the output line by line.
 *
 * cc -O2 line-stipple.c -o line-stipple -lEGL -lGL
 */
#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define N 64

struct line_case {
   float x0, y0, x1, y1;
   int factor;
   unsigned pattern;
   int strip; /* 1: draw as a 3-vertex strip through the midpoint */
};

static const struct line_case cases[] = {
   {2.5, 10.5, 50.5, 10.5, 1, 0x00ff, 0},
   {2.5, 12.5, 50.5, 12.5, 2, 0x0f0f, 0},
   {2.5, 14.5, 50.5, 14.5, 3, 0xaaaa, 0},
   {50.5, 16.5, 2.5, 16.5, 1, 0x00ff, 0}, /* reversed */
   {10.5, 2.5, 10.5, 60.5, 1, 0x3333, 0}, /* vertical */
   {2.5, 2.5, 40.5, 30.5, 1, 0x00ff, 0},  /* x-major diagonal */
   {20.5, 2.5, 30.5, 60.5, 2, 0x0f0f, 0}, /* y-major diagonal */
   {2.5, 20.5, 50.5, 20.5, 1, 0xffff, 0}, /* solid */
   {2.5, 22.5, 50.5, 22.5, 1, 0x0001, 0}, /* sparse */
   {2.5, 24.5, 50.5, 24.5, 1, 0x00ff, 1}, /* strip, per-segment counter */
};

int
main(void)
{
   EGLDisplay d = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                        EGL_DEFAULT_DISPLAY, NULL);
   assert(d != EGL_NO_DISPLAY && eglInitialize(d, NULL, NULL));
   assert(eglBindAPI(EGL_OPENGL_API));
   EGLint attrs[] = {EGL_CONTEXT_OPENGL_PROFILE_MASK,
                     EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, EGL_NONE};
   EGLContext ctx = eglCreateContext(d, NULL, EGL_NO_CONTEXT, attrs);
   assert(ctx != EGL_NO_CONTEXT);
   assert(eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx));
   fprintf(stderr, "renderer=%s version=%s\n", glGetString(GL_RENDERER),
           glGetString(GL_VERSION));

   GLuint fbo, rb;
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, N, N);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_RENDERBUFFER, rb);
   assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

   glViewport(0, 0, N, N);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0, N, 0, N, -1, 1);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glEnable(GL_LINE_STIPPLE);

   int failures = 0;
   for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
      const struct line_case *l = &cases[c];
      glClearColor(0, 0, 0, 1);
      glClear(GL_COLOR_BUFFER_BIT);
      glLineStipple(l->factor, l->pattern);
      glColor3f(1, 1, 1);
      if (l->strip) {
         glBegin(GL_LINE_STRIP);
         glVertex2f(l->x0, l->y0);
         glVertex2f((l->x0 + l->x1) / 2, (l->y0 + l->y1) / 2);
         glVertex2f(l->x1, l->y1);
      } else {
         glBegin(GL_LINES);
         glVertex2f(l->x0, l->y0);
         glVertex2f(l->x1, l->y1);
      }
      glEnd();

      static unsigned char px[N * N * 4];
      glReadPixels(0, 0, N, N, GL_RGBA, GL_UNSIGNED_BYTE, px);
      if (glGetError())
         failures++;

      /* For each major-axis position, whether any pixel across it is lit. */
      int x_major = abs((int)(l->x1 - l->x0)) >= abs((int)(l->y1 - l->y0));
      char row[N + 1];
      int lit = 0;
      for (int m = 0; m < N; ++m) {
         int any = 0;
         for (int k = 0; k < N; ++k) {
            int x = x_major ? m : k, y = x_major ? k : m;
            any |= px[(y * N + x) * 4] > 127;
         }
         row[m] = any ? '#' : '.';
         lit += any;
      }
      row[N] = 0;
      printf("case %u lit=%2d %s\n", c, lit, row);
   }

   /* Multi-vertex strips, a loop and primitive restart, through client
    * vertex/index arrays. The counter continues along a strip or loop and
    * resets at each restart. */
   static const struct {
      GLenum mode;
      int factor;
      unsigned pattern;
      int count;
      float v[8][2];
      unsigned short idx[10];
      int nidx;
   } multi[] = {
      {GL_LINE_STRIP, 1, 0x00ff, 5,
       {{2.5, 26.5}, {12.5, 26.5}, {22.5, 26.5}, {32.5, 26.5}, {50.5, 26.5}}, {0}, 0},
      {GL_LINE_STRIP, 2, 0x0f0f, 4,
       {{2.5, 28.5}, {9.5, 28.5}, {30.5, 28.5}, {50.5, 28.5}}, {0}, 0},
      {GL_LINE_STRIP, 1, 0x0f0f, 4,
       {{2.5, 2.5}, {20.5, 30.5}, {40.5, 10.5}, {60.5, 50.5}}, {0}, 0},
      {GL_LINE_LOOP, 1, 0x00ff, 3,
       {{4.5, 40.5}, {56.5, 40.5}, {56.5, 60.5}}, {0}, 0},
      {GL_LINE_STRIP, 1, 0x00ff, 4,
       {{2.5, 34.5}, {26.5, 34.5}, {26.5, 34.5}, {50.5, 34.5}},
       {0, 1, 0xffff, 2, 3}, 5},
   };
   glEnableClientState(GL_VERTEX_ARRAY);
   for (unsigned c = 0; c < sizeof(multi) / sizeof(multi[0]); ++c) {
      glClear(GL_COLOR_BUFFER_BIT);
      glLineStipple(multi[c].factor, multi[c].pattern);
      glVertexPointer(2, GL_FLOAT, 0, multi[c].v);
      if (multi[c].nidx) {
         glEnable(GL_PRIMITIVE_RESTART);
         glPrimitiveRestartIndex(0xffff);
         glDrawElements(multi[c].mode, multi[c].nidx, GL_UNSIGNED_SHORT,
                        multi[c].idx);
         glDisable(GL_PRIMITIVE_RESTART);
      } else {
         glDrawArrays(multi[c].mode, 0, multi[c].count);
      }
      static unsigned char px[N * N * 4];
      glReadPixels(0, 0, N, N, GL_RGBA, GL_UNSIGNED_BYTE, px);
      if (glGetError())
         failures++;
      unsigned lit = 0, hash = 2166136261u;
      for (int y = 0; y < N; ++y) {
         for (int x = 0; x < N; ++x) {
            if (px[(y * N + x) * 4] > 127) {
               lit++;
               hash = (hash ^ (unsigned)(y * N + x)) * 16777619u;
            }
         }
      }
      printf("multi %u lit=%3u hash=%08x\n", c, lit, hash);
      /* Horizontal cases: also print the column projection. */
      if (c == 0 || c == 1 || c == 4) {
         char row[N + 1];
         int y = (int)multi[c].v[0][1];
         for (int x = 0; x < N; ++x)
            row[x] = px[(y * N + x) * 4] > 127 ? '#' : '.';
         row[N] = 0;
         printf("        row %s\n", row);
      }
   }

   /* Long strips: collinear horizontal vertices, continuous at or below the
    * strip-segment bound (identical to one stippled line), counted per
    * segment above it. The drawn pattern of the second depends on the
    * driver's fallback; this case checks that it completes. */
   static const int long_counts[] = {1000, 50000};
   for (unsigned c = 0; c < 2; ++c) {
      int n = long_counts[c];
      float *v = malloc(sizeof(float) * 2 * n);
      for (int i = 0; i < n; ++i) {
         v[2 * i] = 2.5f + 48.0f * i / (n - 1);
         v[2 * i + 1] = 30.5f;
      }
      glClear(GL_COLOR_BUFFER_BIT);
      glLineStipple(1, 0x00ff);
      glVertexPointer(2, GL_FLOAT, 0, v);
      glDrawArrays(GL_LINE_STRIP, 0, n);
      static unsigned char px[N * N * 4];
      glReadPixels(0, 0, N, N, GL_RGBA, GL_UNSIGNED_BYTE, px);
      if (glGetError())
         failures++;
      char row[N + 1];
      for (int x = 0; x < N; ++x)
         row[x] = px[(30 * N + x) * 4] > 127 ? '#' : '.';
      row[N] = 0;
      printf("long %d %s %s\n", n, c ? "fallback" : "cont    ", row);
      free(v);
   }

   /* Fill the varyings: a vertex shader writing every varying slot the
    * driver allows must keep its values whether or not stipple applies. */
   GLint max_vec;
   glGetIntegerv(GL_MAX_VARYING_VECTORS, &max_vec);
   int nv = max_vec - 1; /* gl_Position takes no user slot here */
   char vs[2048], fs[2048];
   snprintf(vs, sizeof(vs),
            "#version 130\nin vec2 p; out vec4 v[%d];\n"
            "void main() { for (int i = 0; i < %d; ++i) v[i] = vec4(i, i + 1, i + 2, i + 3);\n"
            " gl_Position = vec4(p.x / 32.0 - 1.0, p.y / 32.0 - 1.0, 0, 1); }\n",
            nv, nv);
   snprintf(fs, sizeof(fs),
            "#version 130\nin vec4 v[%d];\n"
            "void main() { bool ok = true; for (int i = 0; i < %d; ++i)\n"
            " ok = ok && v[i] == vec4(i, i + 1, i + 2, i + 3);\n"
            " gl_FragColor = ok ? vec4(1, 1, 1, 1) : vec4(0, 0, 1, 1); }\n",
            nv, nv);
   GLuint prog = glCreateProgram();
   const char *srcs[2] = {vs, fs};
   GLenum types[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
   for (int i = 0; i < 2; ++i) {
      GLuint sh = glCreateShader(types[i]);
      glShaderSource(sh, 1, &srcs[i], NULL);
      glCompileShader(sh);
      glAttachShader(prog, sh);
   }
   glBindAttribLocation(prog, 0, "p");
   glLinkProgram(prog);
   GLint linked;
   glGetProgramiv(prog, GL_LINK_STATUS, &linked);
   if (linked) {
      glUseProgram(prog);
      glDisableClientState(GL_VERTEX_ARRAY);
      static const float seg[4] = {2.5, 32.5, 50.5, 32.5};
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, seg);
      glClear(GL_COLOR_BUFFER_BIT);
      glLineStipple(1, 0x00ff);
      glDrawArrays(GL_LINES, 0, 2);
      static unsigned char px[N * N * 4];
      glReadPixels(0, 0, N, N, GL_RGBA, GL_UNSIGNED_BYTE, px);
      int white = 0, bad = 0;
      for (int i = 0; i < N * N; ++i) {
         white += px[4 * i] > 127;
         bad += px[4 * i] < 128 && px[4 * i + 2] > 127;
      }
      printf("varyings %d white=%d corrupted=%d\n", nv, white, bad);
      failures += bad != 0;
      glUseProgram(0);
   } else {
      printf("varyings %d link failed\n", nv);
   }
   return failures ? 1 : 0;
}
