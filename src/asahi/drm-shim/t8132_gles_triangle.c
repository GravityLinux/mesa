/* SPDX-License-Identifier: MIT */

/*
 * End-to-end EGL/GLES smoke test for the m1n1-backed Asahi DRM shim.
 *
 * This intentionally enters through EGL and Gallium.  It does not issue an
 * Asahi ioctl directly, so shader compilation, resource allocation, VM binds,
 * command construction, and submission all come from Mesa.
 */

#define _POSIX_C_SOURCE 200809L
#include <time.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
   FRAGMENT_MATRIX_PROGRAMS = 9,
   FRAGMENT_MATRIX_FRAMES_PER_PROGRAM = 4,
};

static unsigned
fragment_matrix_program_for_frame(unsigned frame, bool interleaved)
{
   if (interleaved && frame >= FRAGMENT_MATRIX_FRAMES_PER_PROGRAM) {
      return 1 + ((frame - FRAGMENT_MATRIX_FRAMES_PER_PROGRAM) %
                  (FRAGMENT_MATRIX_PROGRAMS - 1));
   }

   return (frame / FRAGMENT_MATRIX_FRAMES_PER_PROGRAM) %
          FRAGMENT_MATRIX_PROGRAMS;
}

static void
fail(const char *message)
{
   fprintf(stderr, "T8132_GLES_TRIANGLE_FAIL: %s (EGL=%#x GL=%#x)\n", message,
           eglGetError(), glGetError());
   exit(1);
}

static GLuint
compile_shader(GLenum stage, const char *source)
{
   GLuint shader = glCreateShader(stage);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);

   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      GLsizei length = 0;
      glGetShaderInfoLog(shader, sizeof(log), &length, log);
      fprintf(stderr, "shader compile failed: %.*s\n", length, log);
      fail("shader compile");
   }

   return shader;
}

enum vertex_variant {
   VERTEX_VARIANT_NORMAL = 0,
   VERTEX_VARIANT_SHIFTED,
   VERTEX_VARIANT_DEGENERATE,
};

static const char *
vertex_variant_name(enum vertex_variant variant)
{
   switch (variant) {
   case VERTEX_VARIANT_NORMAL:
      return "normal";
   case VERTEX_VARIANT_SHIFTED:
      return "shifted";
   case VERTEX_VARIANT_DEGENERATE:
      return "degenerate";
   }

   return "invalid";
}

static enum vertex_variant
read_vertex_variant(void)
{
   const char *value = getenv("T8132_GLES_VERTEX_VARIANT");
   if (!value || !value[0] || !strcmp(value, "normal"))
      return VERTEX_VARIANT_NORMAL;
   if (!strcmp(value, "shifted"))
      return VERTEX_VARIANT_SHIFTED;
   if (!strcmp(value, "degenerate"))
      return VERTEX_VARIANT_DEGENERATE;

   fail("invalid T8132_GLES_VERTEX_VARIANT");
   return VERTEX_VARIANT_NORMAL;
}

/* Optional source files let the bring-up fixture exercise ordinary new GLSL
 * programs without teaching either the driver or compiler about scene names. */
static char *
read_shader_source(const char *environment)
{
   const char *path = getenv(environment);
   if (!path || !path[0])
      return NULL;
   FILE *file = fopen(path, "rb");
   if (!file)
      fail("open shader source");
   if (fseek(file, 0, SEEK_END))
      fail("seek shader source");
   long size = ftell(file);
   if (size < 0 || size > 1024 * 1024 || fseek(file, 0, SEEK_SET))
      fail("shader source size");
   char *source = malloc((size_t)size + 1);
   if (!source || fread(source, 1, size, file) != (size_t)size)
      fail("read shader source");
   source[size] = 0;
   fclose(file);
   return source;
}

static GLuint
link_program(unsigned fragment_variant, bool vbo_vertex,
             enum vertex_variant vertex_variant, GLuint *shared_vs)
{
   static const char vertex_source[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "out vec3 color;\n"
      "const vec2 positions[3] = vec2[3](\n"
      "   vec2( 0.0,  0.82),\n"
      "   vec2(-0.82, -0.72),\n"
      "   vec2( 0.82, -0.72));\n"
      "const vec3 colors[3] = vec3[3](\n"
      "   vec3(1.0, 0.0, 0.0),\n"
      "   vec3(0.0, 0.0, 1.0),\n"
      "   vec3(0.0, 1.0, 0.0));\n"
      "void main() {\n"
      "   gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);\n"
      "   color = colors[gl_VertexID];\n"
      "}\n";
   static const char vertex_source_shifted[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "out vec3 color;\n"
      "const vec2 positions[3] = vec2[3](\n"
      "   vec2( 0.55,  0.82),\n"
      "   vec2(-0.25, -0.72),\n"
      "   vec2( 0.95, -0.72));\n"
      "const vec3 colors[3] = vec3[3](\n"
      "   vec3(1.0, 0.0, 0.0),\n"
      "   vec3(0.0, 0.0, 1.0),\n"
      "   vec3(0.0, 1.0, 0.0));\n"
      "void main() {\n"
      "   gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);\n"
      "   color = colors[gl_VertexID];\n"
      "}\n";
   static const char vertex_source_degenerate[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "out vec3 color;\n"
      "const vec2 positions[3] = vec2[3](\n"
      "   vec2(-0.75, -0.50),\n"
      "   vec2( 0.00,  0.00),\n"
      "   vec2( 0.75,  0.50));\n"
      "const vec3 colors[3] = vec3[3](\n"
      "   vec3(1.0, 0.0, 0.0),\n"
      "   vec3(0.0, 0.0, 1.0),\n"
      "   vec3(0.0, 1.0, 0.0));\n"
      "void main() {\n"
      "   gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);\n"
      "   color = colors[gl_VertexID];\n"
      "}\n";
   static const char vertex_source_vbo[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "layout(location = 0) in vec2 position;\n"
      "layout(location = 1) in vec3 vertex_color;\n"
      "out vec3 color;\n"
      "void main() {\n"
      "   gl_Position = vec4(position, 0.0, 1.0);\n"
      "   color = vertex_color;\n"
      "}\n";
   static const char vertex_source_split_varyings[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "out vec2 shade_uv;\n"
      "out float shade_weight;\n"
      "const vec2 positions[3] = vec2[3](\n"
      "      vec2(0.0, 0.82), vec2(-0.82, -0.72),\n"
      "      vec2(0.82, -0.72));\n"
      "const vec3 values[3] = vec3[3](\n"
      "      vec3(1.0, 0.0, 0.25), vec3(0.0, 0.0, 0.75),\n"
      "      vec3(0.0, 1.0, 0.5));\n"
      "void main() {\n"
      "   gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);\n"
      "   shade_uv = values[gl_VertexID].xy;\n"
      "   shade_weight = values[gl_VertexID].z;\n"
      "}\n";
   static const char fragment_source[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   float product = color.r * color.r;\n"
      "   out_color = vec4(product, product * product,\n"
      "                    product * color.r, 1.0);\n"
      "}\n";
   static const char fragment_source_vbo_passthrough[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color, 1.0);\n"
      "}\n";
   static const char fragment_source_passthrough[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color, 1.0);\n"
      "}\n";
   static const char fragment_source_long[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   float rg = color.r * color.g;\n"
      "   float bb = color.b * color.b;\n"
      "   out_color = vec4(rg, rg * 0.5, bb, 1.0);\n"
      "}\n";
   static const char fragment_source_square_rgb[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color.r * color.r, color.g * color.g,\n"
      "                    color.b * color.b, 1.0);\n"
      "}\n";
   static const char fragment_source_scalar_square[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   float square = color.r * color.r;\n"
      "   out_color = vec4(square, square, square, 1.0);\n"
      "}\n";
   static const char fragment_source_scalar_add[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   float sum = color.r + color.r;\n"
      "   out_color = vec4(sum, sum, sum, 1.0);\n"
      "}\n";
   static const char fragment_source_square_r_passthrough_gb[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color.r * color.r, color.g, color.b, 1.0);\n"
      "}\n";
   static const char fragment_source_square_g_passthrough_rb[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color.r, color.g * color.g, color.b, 1.0);\n"
      "}\n";
   static const char fragment_source_square_b_passthrough_rg[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color.r, color.g, color.b * color.b, 1.0);\n"
      "}\n";
   static const char fragment_source_square_rg_passthrough_b[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color.r * color.r, color.g * color.g,\n"
      "                    color.b, 1.0);\n"
      "}\n";
   static const char fragment_source_square_rb_passthrough_g[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color.r * color.r, color.g,\n"
      "                    color.b * color.b, 1.0);\n"
      "}\n";
   static const char fragment_source_square_gb_passthrough_r[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(color.r, color.g * color.g,\n"
      "                    color.b * color.b, 1.0);\n"
      "}\n";
   static const char fragment_source_cycle[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   float product = color.r * color.r;\n"
      "   out_color = vec4(product * product, product,\n"
      "                    product * color.r, 1.0);\n"
      "}\n";
   static const char fragment_source_position[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      /*
       * Keep this probe deliberately free of an additional multiply.  Mesa
       * lowers gl_FragCoord.x itself to the Apple9 pixel-coordinate read,
       * u32-to-f32 conversion, and required +0.5 pixel-centre bias.  Saturating
       * the result in the normalized render target makes that three-operation
       * input path visible without conflating it with generic FMUL lowering.
       */
      "   float x = gl_FragCoord.x;\n"
      "   out_color = vec4(x, color.r, x, 1.0);\n"
      "}\n";
   static const char fragment_source_split_varyings[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec2 shade_uv;\n"
      "in float shade_weight;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   out_color = vec4(shade_uv.x * shade_weight,\n"
      "                    shade_uv.y + shade_weight,\n"
      "                    max(shade_uv.x, shade_uv.y), 1.0);\n"
      "}\n";
   static const char fragment_source_compare_select[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "in vec3 color;\n"
      "layout(location = 0) out vec4 out_color;\n"
      "void main() {\n"
      "   float red = color.r < color.g ? color.r : color.b;\n"
      "   float green = color.g >= color.b ? color.g : color.r;\n"
      "   float blue = color.r == color.b ? color.b : color.g;\n"
      "   out_color = vec4(red, green, blue, 1.0);\n"
      "}\n";

   const char *selected_fragment_source = fragment_source;
   if (vbo_vertex && fragment_variant == 0)
      selected_fragment_source = fragment_source_vbo_passthrough;
   else if (fragment_variant == 1)
      selected_fragment_source = fragment_source_long;
   else if (fragment_variant == 2)
      selected_fragment_source = fragment_source_cycle;
   else if (fragment_variant == 3)
      selected_fragment_source = fragment_source_position;
   else if (fragment_variant == 4)
      selected_fragment_source = fragment_source_split_varyings;
   else if (fragment_variant == 5)
      selected_fragment_source = fragment_source_compare_select;
   else if (fragment_variant == 6)
      selected_fragment_source = fragment_source_square_rgb;
   else if (fragment_variant == 7)
      selected_fragment_source = fragment_source_passthrough;
   else if (fragment_variant == 8)
      selected_fragment_source = fragment_source_scalar_square;
   else if (fragment_variant == 9)
      selected_fragment_source = fragment_source_scalar_add;
   else if (fragment_variant == 10)
      selected_fragment_source = fragment_source_square_r_passthrough_gb;
   else if (fragment_variant == 11)
      selected_fragment_source = fragment_source_square_g_passthrough_rb;
   else if (fragment_variant == 12)
      selected_fragment_source = fragment_source_square_b_passthrough_rg;
   else if (fragment_variant == 13)
      selected_fragment_source = fragment_source_square_rg_passthrough_b;
   else if (fragment_variant == 14)
      selected_fragment_source = fragment_source_square_rb_passthrough_g;
   else if (fragment_variant == 15)
      selected_fragment_source = fragment_source_square_gb_passthrough_r;

   static const char vertex_source_procedural[] =
      "#version 300 es\n"
      "precision highp float;\n"
      "out vec3 color;\n"
      "void main() {\n"
      " int id = gl_VertexID;\n"
      " float x = id == 0 ? 0.0 : (id == 1 ? -0.82 : 0.82);\n"
      " float y = id == 0 ? 0.82 : -0.72;\n"
      " gl_Position = vec4(x, y, 0.0, 1.0);\n"
      " color = vec3(id == 0 ? 1.0 : 0.0, id == 2 ? 1.0 : 0.0, id == 1 ? 1.0 : 0.0);\n"
      "}\n";
   const char *selected_vertex_source = vertex_source;
   if (vbo_vertex)
      selected_vertex_source = vertex_source_vbo;
   else if (vertex_variant == VERTEX_VARIANT_SHIFTED)
      selected_vertex_source = vertex_source_shifted;
   else if (vertex_variant == VERTEX_VARIANT_DEGENERATE)
      selected_vertex_source = vertex_source_degenerate;
   if (fragment_variant == 4)
      selected_vertex_source = vertex_source_split_varyings;
   if (getenv("T8132_GLES_PROCEDURAL"))
      selected_vertex_source = vertex_source_procedural;
   char *external_vs = read_shader_source("T8132_GLES_VERTEX_SOURCE");
   char *external_fs = read_shader_source("T8132_GLES_FRAGMENT_SOURCE");
   if (external_vs)
      selected_vertex_source = external_vs;
   if (external_fs)
      selected_fragment_source = external_fs;
   GLuint vs = shared_vs && *shared_vs ? *shared_vs :
               compile_shader(GL_VERTEX_SHADER, selected_vertex_source);
   if (shared_vs && !*shared_vs)
      *shared_vs = vs;
   GLuint fs = compile_shader(GL_FRAGMENT_SHADER, selected_fragment_source);
   free(external_vs);
   free(external_fs);
   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);
   if (!shared_vs)
      glDeleteShader(vs);
   glDeleteShader(fs);

   GLint ok = GL_FALSE;
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      GLsizei length = 0;
      glGetProgramInfoLog(program, sizeof(log), &length, log);
      fprintf(stderr, "program link failed: %.*s\n", length, log);
      fail("program link");
   }

   return program;
}

static EGLDisplay
open_asahi_display(void)
{
   PFNEGLQUERYDEVICESEXTPROC query_devices =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
   PFNEGLQUERYDEVICESTRINGEXTPROC query_device_string =
      (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress(
         "eglQueryDeviceStringEXT");
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
         "eglGetPlatformDisplayEXT");

   if (!query_devices || !query_device_string || !get_platform_display)
      fail("EGL_EXT_device_enumeration unavailable");

   if (getenv("T8132_GLES_SOFTWARE_REFERENCE"))
      return get_platform_display(EGL_PLATFORM_SURFACELESS_MESA,
                                  EGL_DEFAULT_DISPLAY, NULL);

   EGLDeviceEXT devices[16];
   EGLint count = 0;
   if (!query_devices(16, devices, &count))
      fail("eglQueryDevicesEXT");

   for (EGLint i = 0; i < count; ++i) {
      const char *drm =
         query_device_string(devices[i], EGL_DRM_DEVICE_FILE_EXT);
      const char *render =
         query_device_string(devices[i], EGL_DRM_RENDER_NODE_FILE_EXT);
      const char *extensions = query_device_string(devices[i], EGL_EXTENSIONS);
      fprintf(stderr, "EGL device %d drm=%s render=%s extensions=%s\n", i,
              drm ? drm : "(none)", render ? render : "(none)",
              extensions ? extensions : "(none)");
      if ((!drm || !strstr(drm, "renderD")) &&
          (!render || !strstr(render, "renderD")))
         continue;

      EGLDisplay display =
         get_platform_display(EGL_PLATFORM_DEVICE_EXT, devices[i], NULL);
      if (display != EGL_NO_DISPLAY)
         return display;
   }

   fail("Asahi DRM-shim EGL device not found");
   return EGL_NO_DISPLAY;
}

static unsigned
read_dimension(const char *name, unsigned fallback)
{
   const char *value = getenv(name);
   if (!value || !value[0])
      return fallback;
   char *end = NULL;
   unsigned long parsed = strtoul(value, &end, 0);
   if (!end || end[0] || parsed == 0 || parsed > 0x4000)
      fail("invalid framebuffer dimension");
   return parsed;
}

static void
write_ppm(const char *path, const uint8_t *rgba,
          unsigned width, unsigned height)
{
   if (!path)
      return;

   FILE *file = fopen(path, "wb");
   if (!file)
      fail("open output image");

   fprintf(file, "P6\n%u %u\n255\n", width, height);
   for (int y = (int)height - 1; y >= 0; --y) {
      for (unsigned x = 0; x < width; ++x)
         fwrite(&rgba[((y * width) + x) * 4], 1, 3, file);
   }
   fclose(file);
}

#include "scenes/mesh/mesh.h"
#include "scenes/pipelines/pipelines.h"
#include "scenes/blending/blending.h"
#include "scenes/island/island.h"
#include "scenes/sunset/sunset.h"
#include "scenes/control-flow/control-flow.h"
#include "scenes/fragcoord/fragcoord.h"
#include "scenes/mix/mix.h"
#include "scenes/trig/trig.h"
#include "scenes/scissor/scissor.h"
#include "scenes/textures/textures.h"
#include "scenes/multitexture/multitexture.h"
#include "scenes/vertex-inputs/vertex-inputs.h"
#include "scenes/canvases/canvases.h"

int
main(int argc, char **argv)
{
   if (argc > 2) {
      fprintf(stderr, "usage: %s [output.ppm]\n", argv[0]);
      return 2;
   }

   unsigned width = read_dimension("T8132_GLES_WIDTH", 512);
   unsigned height = read_dimension("T8132_GLES_HEIGHT", 512);
   EGLDisplay display = open_asahi_display();
   EGLint major = 0, minor = 0;
   if (!eglInitialize(display, &major, &minor))
      fail("eglInitialize");
   bool desktop = getenv("T8132_DESKTOP_GL") != NULL;
   if (!eglBindAPI(desktop ? EGL_OPENGL_API : EGL_OPENGL_ES_API))
      fail("eglBindAPI");

   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE,
      EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE,
      desktop ? EGL_OPENGL_BIT : EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,
      8,
      EGL_GREEN_SIZE,
      8,
      EGL_BLUE_SIZE,
      8,
      EGL_ALPHA_SIZE,
      8,
      EGL_NONE,
   };
   EGLConfig config;
   EGLint config_count = 0;
   if (!eglChooseConfig(display, config_attributes, &config, 1,
                        &config_count) ||
       config_count != 1)
      fail("eglChooseConfig");

   const EGLint pbuffer_attributes[] = {
      EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE,
   };
   EGLSurface surface =
      eglCreatePbufferSurface(display, config, pbuffer_attributes);
   if (surface == EGL_NO_SURFACE)
      fail("eglCreatePbufferSurface");

   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION_KHR,
      3,
      EGL_CONTEXT_MINOR_VERSION_KHR,
      0,
      EGL_NONE,
   };
   EGLContext contexts[2] = {
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes),
      EGL_NO_CONTEXT,
   };
   if (contexts[0] == EGL_NO_CONTEXT)
      fail("eglCreateContext");
   if (!eglMakeCurrent(display, surface, surface, contexts[0]))
      fail("eglMakeCurrent");

   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("T8132_GLES_RENDERER %s\n", renderer ? renderer : "(null)");
   const char *expected = getenv("T8132_GLES_SOFTWARE_REFERENCE") ? "llvmpipe" : "Apple M4";
   if (!renderer || !strstr(renderer, expected))
      fail("unexpected GL renderer");

   if (getenv("T8132_GLES_CANVASES"))
      return run_canvases(width, height);

   if (getenv("T8132_GLES_CONTROL_FLOW"))
      return run_control_flow(width, height);
   if (getenv("T8132_GLES_FRAGCOORD"))
      return run_fragcoord(width, height);
   if (getenv("T8132_GLES_MIX"))
      return run_mix(width, height);
   if (getenv("T8132_GLES_TRIG"))
      return run_trig(width, height);
   if (getenv("T8132_GLES_VERTEX_INPUTS"))
      return run_vertex_inputs(width, height);
   if (getenv("T8132_GLES_MULTITEXTURE"))
      return run_multitexture(width, height);

   if (getenv("T8132_GLES_TEXTURES"))
      return run_textures(width, height);
   if (getenv("T8132_GLES_SCISSOR"))
      return run_scissor(width, height);

   if (getenv("T8132_GLES_BLEND"))
      return run_blending(width, height);

   if (getenv("T8132_GLES_PIPELINES"))
      return run_pipelines(width, height);

   if (getenv("T8132_GLES_ISLAND"))
      return getenv("T8132_GLES_SUNSET") ? run_sunset(width, height) : run_island(width, height);

   if (getenv("T8132_GLES_MESH"))
      return run_mesh(width, height);

   bool cycle_pipelines = getenv("T8132_GLES_PIPELINE_CYCLE") != NULL;
   bool cycle_vertices = getenv("T8132_GLES_VERTEX_CYCLE") != NULL;
   bool fragment_matrix = getenv("T8132_GLES_FRAGMENT_MATRIX") != NULL;
   bool interleave_fragment_matrix =
      getenv("T8132_GLES_FRAGMENT_MATRIX_INTERLEAVED") != NULL;
   bool vbo_vertex = getenv("T8132_GLES_VBO_VERTEX") != NULL;
   enum vertex_variant vertex_variant = read_vertex_variant();
   if (vbo_vertex && vertex_variant != VERTEX_VARIANT_NORMAL)
      fail("vertex variants require the inline gl_VertexID path");
   if (cycle_vertices &&
       (cycle_pipelines || vbo_vertex ||
        vertex_variant != VERTEX_VARIANT_NORMAL))
      fail("vertex cycle requires the isolated inline render path");
   if (fragment_matrix && (cycle_pipelines || cycle_vertices))
      fail("fragment matrix cannot be combined with another cycle mode");
   bool alternate_contexts =
      cycle_pipelines && getenv("T8132_GLES_TWO_CONTEXTS") != NULL;
   GLuint programs[FRAGMENT_MATRIX_PROGRAMS] = {0};
   GLuint vaos[2] = {0};
   unsigned first_fragment = 0;
   if (!cycle_pipelines && getenv("T8132_GLES_LONG_FRAGMENT") != NULL)
      first_fragment = 1;
   if (!cycle_pipelines && getenv("T8132_GLES_POSITION_FRAGMENT") != NULL)
      first_fragment = 3;
   if (!cycle_pipelines && getenv("T8132_GLES_SPLIT_VARYINGS") != NULL)
      first_fragment = 4;
   if (!cycle_pipelines && getenv("T8132_GLES_COMPARE_SELECT") != NULL)
      first_fragment = 5;
   if (!cycle_pipelines && getenv("T8132_GLES_SQUARE_RGB") != NULL)
      first_fragment = 6;
   if (!cycle_pipelines && getenv("T8132_GLES_PASSTHROUGH") != NULL)
      first_fragment = 7;
   if (!cycle_pipelines && getenv("T8132_GLES_SCALAR_SQUARE") != NULL)
      first_fragment = 8;
   if (!cycle_pipelines && getenv("T8132_GLES_SCALAR_ADD") != NULL)
      first_fragment = 9;
   if (!cycle_pipelines && getenv("T8132_GLES_SQUARE_R_PASSTHROUGH_GB") != NULL)
      first_fragment = 10;
   if (!cycle_pipelines && getenv("T8132_GLES_SQUARE_G_PASSTHROUGH_RB") != NULL)
      first_fragment = 11;
   if (!cycle_pipelines && getenv("T8132_GLES_SQUARE_B_PASSTHROUGH_RG") != NULL)
      first_fragment = 12;
   GLuint shared_matrix_vs = 0;
   if (fragment_matrix) {
      /* This is the complete component-wise square family captured from the
       * caller-owned Metal control: no consumers, one shared consumer,
       * every one-/two-component subset, and all three components. */
      static const unsigned variants[FRAGMENT_MATRIX_PROGRAMS] = {
         7, 8, 10, 11, 12, 13, 14, 15, 6,
      };
      for (unsigned i = 0; i < FRAGMENT_MATRIX_PROGRAMS; ++i)
         programs[i] = link_program(variants[i], vbo_vertex, vertex_variant,
                                    &shared_matrix_vs);
      glDeleteShader(shared_matrix_vs);
   } else {
      programs[0] = link_program(first_fragment, vbo_vertex, vertex_variant,
                                 NULL);
   }
   glGenVertexArrays(1, &vaos[0]);
   glBindVertexArray(vaos[0]);
   glUseProgram(programs[0]);
   glViewport(0, 0, width, height);
   if (cycle_vertices) {
      programs[1] = link_program(first_fragment, false,
                                 VERTEX_VARIANT_SHIFTED, NULL);
      programs[2] = link_program(first_fragment, false,
                                 VERTEX_VARIANT_DEGENERATE, NULL);
   }
   if (cycle_pipelines) {
      unsigned second = alternate_contexts ? 1 : 0;
      if (alternate_contexts) {
         contexts[1] = eglCreateContext(display, config, EGL_NO_CONTEXT,
                                        context_attributes);
         if (contexts[1] == EGL_NO_CONTEXT ||
             !eglMakeCurrent(display, surface, surface, contexts[1]))
            fail("second EGL context");
      }
      /*
       * Use the deliberately unrelated long fragment program for B.  This
       * is a stronger ownership/lifetime gate than merely permuting A's
       * output channels: B consumes all three interpolants, has two
       * independent expression roots, and materializes a nontrivial float
       * constant.  A/B/A must therefore switch between genuinely different
       * immutable compiler packages and then return to the original one.
       */
      programs[1] = link_program(1, vbo_vertex, vertex_variant, NULL);
      glGenVertexArrays(1, &vaos[second]);
      glBindVertexArray(vaos[second]);
      glUseProgram(programs[1]);
      glViewport(0, 0, width, height);
      if (alternate_contexts &&
          !eglMakeCurrent(display, surface, surface, contexts[0]))
         fail("restore first EGL context");
   }
   if (fragment_matrix) {
      /* One controlled ladder, ordered by the lifetime contract it adds:
       * interpolation only; one ALU result; one ALU result reused by three
       * outputs; one transformed plus two direct values; and three
       * independent transformed values.  Four adjacent frames per entry
       * leave three measured repeats even if the first publication only
       * establishes the render envelope. */
      glUseProgram(programs[0]);
   }

   GLuint vertex_buffer = 0;
   if (vbo_vertex) {
      static const float vertices[] = {
          0.0f,  0.82f, 1.0f, 0.0f, 0.0f,
         -0.82f, -0.72f, 0.0f, 0.0f, 1.0f,
          0.82f, -0.72f, 0.0f, 1.0f, 0.0f,
      };
      glGenBuffers(1, &vertex_buffer);
      glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
      glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                   GL_STATIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                            (const void *)0);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                            (const void *)(2 * sizeof(float)));
   }

   /* Compile/link/state construction probe.  This intentionally stops before
    * the first draw so diagnostic compiler bridges can never reach hardware.
    */
   if (getenv("T8132_GLES_COMPILE_ONLY") != NULL) {
      glFinish();
      printf("T8132_GLES_COMPILE_ONLY_OK renderer=\"%s\" EGL=%d.%d\n",
             renderer, major, minor);
      return 0;
   }
   /* Each inline vertex control gets a warm-up publication followed by its
    * measured publication.  The known G16 render envelope's first use writes
    * only the clear target; treating retirement alone as success would hide
    * precisely the no-output failure this control is designed to catch. */
   unsigned frame_count =
                          fragment_matrix ?
                             FRAGMENT_MATRIX_PROGRAMS *
                                FRAGMENT_MATRIX_FRAMES_PER_PROGRAM :
                          cycle_vertices ? 6 : cycle_pipelines ? 3 : 2;
   const char *frame_env = getenv("T8132_GLES_FRAMES");
   if (frame_env && frame_env[0]) {
      char *end = NULL;
      unsigned long parsed = strtoul(frame_env, &end, 0);
      if (!end || end[0] || parsed == 0 || parsed > UINT32_MAX)
         fail("invalid T8132_GLES_FRAMES");
      frame_count = parsed;
   }
   unsigned draws_per_frame = read_dimension("T8132_GLES_DRAWS", 1);
   unsigned vertex_count = read_dimension("T8132_GLES_VERTICES", 3);
   printf("T8132_GLES_DRAW vertices=%u draws_per_frame=%u\n",
          vertex_count, draws_per_frame);
   for (unsigned frame = 0; frame < frame_count; ++frame) {
      /*
       * A/B/A is the minimum dynamic-cache lifetime gate: A and B select
       * distinct immutable compiler packages, while the final A must reuse
       * the first interned generation rather than rebuild or mutate it.
       */
      unsigned selected = fragment_matrix ?
                             fragment_matrix_program_for_frame(
                                frame, interleave_fragment_matrix) :
                          cycle_vertices ? (frame / 2) % 3 :
                          cycle_pipelines && (frame % 3) == 1;
      unsigned context_index = alternate_contexts ? selected : 0;
      if (alternate_contexts &&
          !eglMakeCurrent(display, surface, surface,
                          contexts[context_index]))
         fail("alternate EGL context");
      glBindVertexArray(vaos[context_index]);
      glUseProgram(programs[selected]);
      glViewport(0, 0, width, height);
      glClearColor(0.75f, 0.73f, 1.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      for (unsigned draw = 0; draw < draws_per_frame; ++draw) {
         if (getenv("T8132_GLES_UNIFORMS")) {
            GLint transform =
               glGetUniformLocation(programs[selected], "u_transform");
            GLint tint = glGetUniformLocation(programs[selected], "u_tint");
            GLint time = glGetUniformLocation(programs[selected], "u_time");
            bool blocks = getenv("T8132_GLES_UNIFORM_BLOCKS") != NULL;
            if (!blocks && transform < 0 && tint < 0 && time < 0)
               fail("uniform test shader interface");
            const GLfloat matrix[2][16] = {
               {.5, 0, 0, 0, 0, .5, 0, 0, 0, 0, 1, 0, -.5, -.125, 0, 1},
               {0, .5, 0, 0, -.5, 0, 0, 0, 0, 0, 1, 0, .5, .125, 0, 1},
            };
            const GLfloat colors[2][4] = {{1, .5, .25, 1}, {.25, .75, 1, 1}};
            if (blocks) {
               static GLuint buffers[2];
               if (!buffers[0])
                  glGenBuffers(2, buffers);
               GLuint vs_block =
                  glGetUniformBlockIndex(programs[selected], "TransformBlock");
               GLuint fs_block =
                  glGetUniformBlockIndex(programs[selected], "TintBlock");
               if (vs_block == GL_INVALID_INDEX || fs_block == GL_INVALID_INDEX)
                  fail("uniform block test interface");
               glUniformBlockBinding(programs[selected], vs_block, 4);
               glUniformBlockBinding(programs[selected], fs_block, 7);
               if (draw == 0) {
                  /* Populate all immutable per-draw ranges before the first
                   * draw. Rebinding then exercises two stage-specific UBOs
                   * without triggering an unrelated in-place buffer hazard. */
                  unsigned char *data = calloc(draws_per_frame, 256);
                  if (!data)
                     fail("uniform block allocation");
                  for (unsigned stage = 0; stage < 2; ++stage) {
                     memset(data, 0, draws_per_frame * 256);
                     for (unsigned d = 0; d < draws_per_frame; ++d) {
                        memcpy(data + d * 256,
                               stage ? colors[d % 2] : matrix[d % 2],
                               stage ? sizeof(colors[0]) : sizeof(matrix[0]));
                        if (stage) {
                           GLfloat value = frame * .125f;
                           memcpy(data + d * 256 + 16, &value, sizeof(value));
                        }
                     }
                     glBindBuffer(GL_UNIFORM_BUFFER, buffers[stage]);
                     glBufferData(GL_UNIFORM_BUFFER, draws_per_frame * 256,
                                  data, GL_DYNAMIC_DRAW);
                  }
                  free(data);
               }
               glBindBufferRange(GL_UNIFORM_BUFFER, 4, buffers[0], draw * 256,
                                 64);
               glBindBufferRange(GL_UNIFORM_BUFFER, 7, buffers[1], draw * 256,
                                 32);
            }
            if (transform >= 0)
               glUniformMatrix4fv(transform, 1, GL_FALSE, matrix[draw % 2]);
            if (tint >= 0)
               glUniform4fv(tint, 1, colors[draw % 2]);
            if (time >= 0)
               glUniform1f(time, frame * .125f);
         }
         glDrawArrays(GL_TRIANGLES, 0, vertex_count);
      }
      /* Build the complete Gallium batch and its Apple9 compiler package,
       * but deliberately stop before glFinish submits it to drm-shim.  This
       * is an offline package-format diagnostic, unlike COMPILE_ONLY which
       * exits before draw-time pipeline construction. */
      if (getenv("T8132_GLES_BUILD_ONLY") != NULL) {
         printf("T8132_GLES_BUILD_ONLY_OK frame=%u draws=%u\n",
                frame, draws_per_frame);
         fflush(NULL);
         _Exit(0);
      }
      glFinish();
      if (glGetError() != GL_NO_ERROR)
         fail("draw");
   }

   if (argc == 2) {
      size_t image_size = (size_t)width * height * 4;
      uint8_t *pixels = malloc(image_size);
      if (!pixels)
         fail("allocate readback");
      glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
      if (glGetError() != GL_NO_ERROR)
         fail("glReadPixels");

      const uint8_t *corner = &pixels[0];
      const uint8_t *centre =
         &pixels[(((height / 2) * width) + (width / 2)) * 4];
      size_t coverage = 0;
      uint64_t x_sum = 0;
      unsigned min_x = width, min_y = height, max_x = 0, max_y = 0;
      for (unsigned y = 0; y < height; ++y) {
         for (unsigned x = 0; x < width; ++x) {
            const uint8_t *pixel = &pixels[((size_t)y * width + x) * 4];
            if (!memcmp(pixel, corner, 4))
               continue;

            coverage++;
            x_sum += x;
            if (x < min_x)
               min_x = x;
            if (y < min_y)
               min_y = y;
            if (x > max_x)
               max_x = x;
            if (y > max_y)
               max_y = y;
         }
      }
      double centroid_x = coverage ? (double)x_sum / coverage : 0.0;
      if (vertex_variant == VERTEX_VARIANT_DEGENERATE) {
         if (coverage != 0)
            fail("degenerate vertex program produced fragments");
      } else {
         if (coverage == 0)
            fail("vertex program produced no fragments");
         double normalized_x = centroid_x / width;
         if (vertex_variant == VERTEX_VARIANT_NORMAL &&
             (normalized_x < 0.40 || normalized_x > 0.60))
            fail("normal vertex program has unexpected coverage centroid");
         if (vertex_variant == VERTEX_VARIANT_SHIFTED && normalized_x < 0.62)
            fail("shifted vertex program did not move coverage right");
      }

      write_ppm(argv[1], pixels, width, height);
      printf("T8132_GLES_TRIANGLE_OK renderer=\"%s\" EGL=%d.%d "
             "size=%ux%u frames=%u draws=%u pipelines=%u contexts=%u "
             "cycle=%s vertex=%s coverage=%zu bbox=%u,%u-%u,%u "
             "centroid_x=%.2f "
             "readback=yes corner=%02x%02x%02x%02x "
             "centre=%02x%02x%02x%02x\n",
             renderer, major, minor, width, height, frame_count,
             draws_per_frame, fragment_matrix ? FRAGMENT_MATRIX_PROGRAMS :
                              cycle_vertices ? 3 :
                              cycle_pipelines ? 2 : 1,
             alternate_contexts ? 2 : 1,
             fragment_matrix ? "fragment-matrix" :
             cycle_vertices ? "vertex-NSD" :
             cycle_pipelines ? "ABA" : "none",
             vertex_variant_name(vertex_variant), coverage,
             coverage ? min_x : 0, coverage ? min_y : 0,
             coverage ? max_x : 0, coverage ? max_y : 0, centroid_x,
             corner[0], corner[1], corner[2], corner[3],
             centre[0], centre[1], centre[2], centre[3]);
      free(pixels);
   } else {
      /* Submission is synchronous in the m1n1 backend, which validates the
       * completed hardware attachment against its source-render oracle before
       * returning from glFinish.
       */
      printf("T8132_GLES_TRIANGLE_OK renderer=\"%s\" EGL=%d.%d "
             "size=%ux%u frames=%u draws=%u pipelines=%u contexts=%u "
             "cycle=%s vertex=%s "
             "readback=no\n",
             renderer, major, minor, width, height, frame_count,
             draws_per_frame, fragment_matrix ? FRAGMENT_MATRIX_PROGRAMS :
                              cycle_vertices ? 3 :
                              cycle_pipelines ? 2 : 1,
             alternate_contexts ? 2 : 1,
             fragment_matrix ? "fragment-matrix" :
             cycle_vertices ? "vertex-NSD" :
             cycle_pipelines ? "ABA" : "none",
             vertex_variant_name(vertex_variant));
   }
   for (unsigned i = 0; i < (alternate_contexts ? 2 : 1); ++i) {
      if (!eglMakeCurrent(display, surface, surface, contexts[i]))
         fail("cleanup EGL context");
      glDeleteVertexArrays(1, &vaos[i]);
      glDeleteProgram(programs[i]);
      if (!alternate_contexts && programs[1] && !fragment_matrix)
         glDeleteProgram(programs[1]);
   }
   if (cycle_vertices)
      glDeleteProgram(programs[2]);
   if (fragment_matrix) {
      for (unsigned i = 1; i < FRAGMENT_MATRIX_PROGRAMS; ++i)
         glDeleteProgram(programs[i]);
   }
   if (vertex_buffer)
      glDeleteBuffers(1, &vertex_buffer);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   if (contexts[1] != EGL_NO_CONTEXT)
      eglDestroyContext(display, contexts[1]);
   eglDestroyContext(display, contexts[0]);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return 0;
}
