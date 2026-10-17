/* Vertex shader input numbering regression (GL, surfaceless EGL).
 *
 * The vertex shader (reduced from Portal 2's vertexlit_and_unlit_generic_vs30)
 * reads v0, v1, v2 and v4, but v1 only feeds a branch whose result goes to a
 * varying the fragment shader does not read. It is eliminated only after the
 * input bases were assigned, so Gallium numbers the vertex elements
 * {v0, v2, v4} while the shader's stale bases still counted v1. A driver that
 * trusted those bases fetched v4's data for v2 (the game's green/yellow
 * vertex colour).
 * v2 is a BGRA unorm8 colour in its own buffer; the point must show it.
 *
 * cc -O2 vs-input-bases.c -o vs-input-bases -lEGL -lGL
 */
#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static const char *vs_src =
   "#version 120\n"
   "attribute vec4 v0;\n"
   "attribute vec4 v1;\n"
   "attribute vec4 v2;\n"
   "attribute vec4 v4;\n"
   "uniform vec4 vc[54];\n"
   "varying vec4 oT4;\n"
   "varying vec4 colour;\n"
   "void main()\n"
   "{\n"
   "   vec4 vd1 = vec4( -128.0, -64.0, 0.01587301679, 1.0 );\n"
   "   vec4 vd2 = vec4( -2.0, 1.0, 2.200000047684, 0.0 );\n"
   "   vec4 r0;\n"
   "   vec4 r1;\n"
   "   vec4 r2;\n"
   "   vec4 r3;\n"
   "   r0.x = abs( vc[50].x );\n"
   "   r0.y = float( -r0.x < r0.x );\n"
   "   if ( -r0.x < r0.x )\n"
   "   {\n"
   "      r0.xz = vd1.xx + v1.xy;\n"
   "      r1.xy = vec2( lessThan( r0.xz, vc[0].xx ) );\n"
   "      r0.xz = abs(r0.xz) + -r1.xy;\n"
   "      r0.xz = r0.xz + vd1.yy;\n"
   "      r0.w = r1.x * vd2.x + vd2.y;\n"
   "      r1.xy = vec2( lessThan( r0.xz, vc[0].xx ) );\n"
   "      r0.xz = abs(r0.xz) + -r1.xy;\n"
   "      r1.z = r0.x * -vd1.z + vd1.w;\n"
   "      r2.z = r0.z * -vd1.z + r1.z;\n"
   "      r1.xy = r1.xy * vd2.xx + vd2.yy;\n"
   "      r2.xy = r0.xz * vd1.zz;\n"
   "      r3.xyz = normalize( r2.xyz );\n"
   "      r1.xy = r1.xy * r3.xy;\n"
   "      r1.z = r0.w * r3.z;\n"
   "   }\n"
   "   else\n"
   "   {\n"
   "      r1.xyz = vc[0].xxx;\n"
   "   }\n"
   "   oT4 = vec4(r1.xyz, 0.0);\n"
   "   colour = r0.yyyy * v2;\n"
   "   gl_Position = vec4(v0.xy + v4.xy, 0.0, 1.0);\n"
   "}\n";

static const char *fs_src =
   "#version 120\n"
   "varying vec4 colour;\n"
   "void main() { gl_FragColor = colour; }\n";

static GLuint
shader(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   assert(ok);
   return s;
}

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

   GLuint rb, fbo;
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 4, 4);
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_RENDERBUFFER, rb);
   glViewport(0, 0, 4, 4);

   GLuint prog = glCreateProgram();
   glAttachShader(prog, shader(GL_VERTEX_SHADER, vs_src));
   glAttachShader(prog, shader(GL_FRAGMENT_SHADER, fs_src));
   glBindAttribLocation(prog, 0, "v0");
   glBindAttribLocation(prog, 1, "v1");
   glBindAttribLocation(prog, 2, "v2");
   glBindAttribLocation(prog, 4, "v4");
   glLinkProgram(prog);
   GLint ok;
   glGetProgramiv(prog, GL_LINK_STATUS, &ok);
   assert(ok);
   glUseProgram(prog);
   float vc[54][4] = {{0}};
   vc[50][0] = 1.0f; /* enables the branch and the colour */
   glUniform4fv(glGetUniformLocation(prog, "vc"), 54, &vc[0][0]);

   /* One interleaved stream for v0/v1/v4 (as the game's), colour apart. */
   static const float stream[8] = {-0.25f, -0.25f, 9.0f, 9.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f};
   static const unsigned char bgra[4] = {0x20, 0x60, 0xc0, 0xff};
   GLuint bo[2];
   glGenBuffers(2, bo);
   glBindBuffer(GL_ARRAY_BUFFER, bo[0]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(stream), stream, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 32, (void *)0);
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, (void *)8);
   glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, 32, (void *)16);
   glBindBuffer(GL_ARRAY_BUFFER, bo[1]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(bgra), bgra, GL_STATIC_DRAW);
   glVertexAttribPointer(2, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, 4, (void *)0);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glEnableVertexAttribArray(2);
   glEnableVertexAttribArray(4);

   glClearColor(0, 0, 0, 0);
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawArrays(GL_POINTS, 0, 1);

   unsigned char px[4 * 4 * 4];
   glReadPixels(0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, px);
   /* (-0.25, -0.25) in a 4x4 viewport is pixel (1, 1). BGRA bytes 20 60 c0
    * read back as R=c0 G=60 B=20. */
   const unsigned char *p = &px[(1 * 4 + 1) * 4];
   int pass = p[0] == 0xc0 && p[1] == 0x60 && p[2] == 0x20 && p[3] == 0xff;
   printf("pixel %02x %02x %02x %02x: %s\n", p[0], p[1], p[2], p[3],
          pass ? "PASS" : "FAIL");
   return pass ? 0 : 1;
}
