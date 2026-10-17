/* SPDX-License-Identifier: MIT */
/* Exercise 32-bit GL transfer mappings without game data. */
#define _GNU_SOURCE
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <sys/mman.h>

#define DXT1_RGBA 0x83F1

static GLuint
compile_shader(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint ok;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   assert(ok);
   return shader;
}

/* Allocate and compile readback state before reserving VA, so exhaustion in
 * unrelated shader/FBO allocations cannot be mistaken for a transfer failure.
 */
static GLuint
prepare_readback(void)
{
   const char *vs = "#version 300 es\n"
                    "void main() {\n"
                    "vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
                    "gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";
   const char *fs = "#version 300 es\n"
                    "precision highp float;\n"
                    "uniform sampler2D tex;\n"
                    "out vec4 color;\n"
                    "void main() { color = texelFetch(tex, ivec2(0), 0); }\n";
   GLuint vertex = compile_shader(GL_VERTEX_SHADER, vs);
   GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fs);
   GLuint program = glCreateProgram();
   glAttachShader(program, vertex);
   glAttachShader(program, fragment);
   glLinkProgram(program);
   GLint ok;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   assert(ok);
   glDeleteShader(vertex);
   glDeleteShader(fragment);
   glUseProgram(program);
   glUniform1i(glGetUniformLocation(program, "tex"), 0);

   GLuint target;
   glGenTextures(1, &target);
   glBindTexture(GL_TEXTURE_2D, target);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          target, 0);
   assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
   glViewport(0, 0, 4, 4);

   GLuint warm;
   uint8_t pixel[4] = {255, 0, 0, 255};
   glGenTextures(1, &warm);
   glBindTexture(GL_TEXTURE_2D, warm);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                pixel);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
   assert(glGetError() == GL_NO_ERROR);
   assert(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0 && pixel[3] == 255);
   glDeleteTextures(1, &warm);
   return program;
}

static unsigned
buffers(unsigned count, unsigned size)
{
   GLuint *bo = calloc(count, sizeof(*bo));
   assert(bo);
   glGenBuffers(count, bo);
   unsigned made = 0;
   for (unsigned i = 0; i < count; i++) {
      glBindBuffer(GL_ARRAY_BUFFER, bo[i]);
      glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_STATIC_DRAW);
      GLenum err = glGetError();
      if (err) {
         printf("FAIL glBufferData i=%u err=0x%x\n", i, err);
         break;
      }
      void *p =
         glMapBufferRange(GL_ARRAY_BUFFER, 0, size,
                          GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
      if (!p) {
         printf("FAIL map i=%u total=%llu err=0x%x\n", i,
                (unsigned long long)i * size, glGetError());
         break;
      }
      *(volatile unsigned *)p = i;
      assert(glUnmapBuffer(GL_ARRAY_BUFFER));
      made++;
      if (i % 20 == 0)
         printf("mapped_then_unmapped=%u bytes=%llu\n", i + 1,
                (unsigned long long)(i + 1) * size);
   }
   for (unsigned i = 0; i < made; i += 17) {
      glBindBuffer(GL_ARRAY_BUFFER, bo[i]);
      unsigned *p = glMapBufferRange(GL_ARRAY_BUFFER, 0, 4, GL_MAP_READ_BIT);
      assert(p && *p == i);
      assert(glUnmapBuffer(GL_ARRAY_BUFFER));
   }
   glDeleteBuffers(count, bo);
   free(bo);
   return made;
}

static unsigned
textures(unsigned count, unsigned width)
{
   /* BC1 is decoded by Mesa on G16, then uploaded through the tiled RGBA8
    * transfer path used by Portal. At 1024^2, each texture uses 4 MiB for
    * RGBA8 plus a 512 KiB compressed shadow: 675 MiB for 150 textures.
    */
   assert(width && width % 4 == 0);
   unsigned size = (width / 4) * (width / 4) * 8;
   uint16_t *data = malloc(size);
   GLuint *tex = calloc(count, sizeof(*tex));
   assert(data && tex);
   glGenTextures(count, tex);
   unsigned made = 0;
   for (unsigned i = 0; i < count; i++) {
      static const uint16_t colors[] = {0xf800, 0x07e0, 0x001f};
      for (unsigned j = 0; j < size / 2; j += 4) {
         data[j] = colors[i % 3];
         data[j + 1] = 0;
         data[j + 2] = 0;
         data[j + 3] = 0;
      }
      glBindTexture(GL_TEXTURE_2D, tex[i]);
      glTexStorage2D(GL_TEXTURE_2D, 1, DXT1_RGBA, width, width);
      GLenum err = glGetError();
      if (err) {
         printf("FAIL glTexStorage2D i=%u err=0x%x\n", i, err);
         break;
      }
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, width, DXT1_RGBA,
                                size, data);
      err = glGetError();
      if (err) {
         printf("FAIL glCompressedTexSubImage2D i=%u err=0x%x\n", i, err);
         break;
      }
      made++;
      if (i % 20 == 0)
         printf("uploaded=%u compressed_bytes_each=%u\n", made, size);
   }
   for (unsigned i = 0; i < made; i += 17) {
      uint8_t pixel[4] = {0};
      glBindTexture(GL_TEXTURE_2D, tex[i]);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
      assert(glGetError() == GL_NO_ERROR);
      for (unsigned c = 0; c < 4; c++)
         assert(pixel[c] == ((c == i % 3 || c == 3) ? 255 : 0));
      printf("readback texture=%u rgba=%u,%u,%u,%u verified\n", i, pixel[0],
             pixel[1], pixel[2], pixel[3]);
   }
   glDeleteTextures(count, tex);
   free(tex);
   free(data);
   return made;
}

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   bool texture = argc > 1 && !strcmp(argv[1], "texture");
   if (texture || (argc > 1 && !strcmp(argv[1], "buffer"))) {
      argc--;
      argv++;
   }
   unsigned count =
      argc > 1 ? strtoul(argv[1], NULL, 0) : (texture ? 150 : 256);
   unsigned size =
      argc > 2 ? strtoul(argv[2], NULL, 0) : (texture ? 1024 : 0x564000);
   unsigned reserve_mib = argc > 3 ? strtoul(argv[3], NULL, 0) : 3072;
   assert(sizeof(void *) == 4);
   EGLDisplay d = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                        EGL_DEFAULT_DISPLAY, NULL);
   assert(d != EGL_NO_DISPLAY);
   assert(eglInitialize(d, NULL, NULL));
   assert(eglBindAPI(EGL_OPENGL_ES_API));
   EGLint attrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
   EGLContext ctx = eglCreateContext(d, NULL, EGL_NO_CONTEXT, attrs);
   assert(ctx != EGL_NO_CONTEXT);
   assert(eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx));
   printf("pid=%d pointer_bits=%zu pagesize=%ld renderer=%s mode=%s\n",
          getpid(), sizeof(void *) * 8, sysconf(_SC_PAGESIZE),
          glGetString(GL_RENDERER), texture ? "texture" : "buffer");
   GLuint program = texture ? prepare_readback() : 0;
   void *reserved[64];
   unsigned reservations = 0;
   assert(reserve_mib % 64 == 0 && reserve_mib < 4096);
   for (; reservations < reserve_mib / 64; reservations++) {
      reserved[reservations] =
         mmap(NULL, 64 * 1024 * 1024, PROT_NONE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      assert(reserved[reservations] != MAP_FAILED);
   }
   printf("VA reservation=%u MiB in 64MiB chunks\n", reserve_mib);
   unsigned made = texture ? textures(count, size) : buffers(count, size);
   printf("completed=%u requested=%u; sampled readback verified\n", made,
          count);
   for (unsigned i = 0; i < reservations; i++)
      munmap(reserved[i], 64 * 1024 * 1024);
   if (program)
      glDeleteProgram(program);
   eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(d, ctx);
   eglTerminate(d);
   return made == count ? 0 : 2;
}
