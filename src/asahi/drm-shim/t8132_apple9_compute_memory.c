/* SPDX-License-Identifier: MIT */

/* Memory cases for the native Apple9 compute Piglit runner. */

#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "t8132_apple9_compute_cases.h"

#define VALUE_COUNT        64u
#define INPUT_EXTRA_VALUES (VALUE_COUNT * 3u)
#define LOCAL_SIZE_X       32u
#define RECIPROCAL_DENOMINATOR_COUNT 1024u
#define PAYLOAD_BYTES   ((VALUE_COUNT + INPUT_EXTRA_VALUES) * sizeof(uint32_t))
#define MIN_GUARD_BYTES 256u

struct buffer_layout {
   size_t first_payload_offset;
   size_t payload_stride;
   size_t buffer_bytes;
};

static void
fail(const char *message)
{
   fprintf(stderr, "T8132_GLES_COMPUTE_ADD3_FAIL: %s (EGL=%#x GL=%#x)\n",
           message, eglGetError(), glGetError());
   exit(1);
}

static void
check_gl(const char *operation)
{
   GLenum error = glGetError();
   if (error != GL_NO_ERROR) {
      fprintf(stderr, "T8132_GLES_COMPUTE_ADD3_FAIL: %s returned GL=%#x\n",
              operation, error);
      exit(1);
   }
}

static void
append_source(char *source, size_t capacity, size_t *used,
              const char *format, ...)
{
   va_list args;
   va_start(args, format);
   int written = vsnprintf(source + *used, capacity - *used, format, args);
   va_end(args);
   if (written < 0 || (size_t)written >= capacity - *used)
      fail("compute shader source overflow");
   *used += written;
}

static size_t
align_up(size_t value, size_t alignment)
{
   if (!alignment || value > SIZE_MAX - (alignment - 1))
      fail("invalid alignment");

   return ((value + alignment - 1) / alignment) * alignment;
}

enum workload {
   WORKLOAD_ADD3,
   WORKLOAD_SPARSE_BINDINGS,
   WORKLOAD_AOS_LOAD,
   WORKLOAD_AOS_STORE,
   WORKLOAD_TWO_LOAD_FALU,
   WORKLOAD_SIX_PENDING,
   WORKLOAD_SEVEN_PENDING,
   WORKLOAD_FANOUT,
   WORKLOAD_TWO_LOAD_AND,
   WORKLOAD_TWO_LOAD_OR,
   WORKLOAD_TWO_LOAD_XOR,
   WORKLOAD_VECTOR_SWIZZLE,
   WORKLOAD_VECTOR_FANOUT,
   WORKLOAD_VECTOR2_COPY,
   WORKLOAD_VECTOR3_COPY,
   WORKLOAD_VECTOR4_COPY,
   WORKLOAD_VECTOR4_ALU_STORE,
   WORKLOAD_MIXED,
   WORKLOAD_DEPENDENT_INDEX,
   WORKLOAD_NESTED_DEPENDENT,
   WORKLOAD_SYSTEM_LOAD_INDEX,
   WORKLOAD_SINGLE_LOAD_IADD,
   WORKLOAD_TWO_LOAD_IADD,
   WORKLOAD_AFFINE_INDEX,
   WORKLOAD_VARIABLE_SHL,
   WORKLOAD_VARIABLE_ASHR,
   WORKLOAD_VARIABLE_USHR,
   WORKLOAD_LOADED_COMPARE,
   WORKLOAD_LOADED_FLOAT_DAG,
   WORKLOAD_RECIPROCAL_DIRECT,
   WORKLOAD_RECIPROCAL_RETAIN,
   WORKLOAD_RECIPROCAL_MATERIALIZED,
   WORKLOAD_RECIPROCAL_DENOMINATORS,
   WORKLOAD_PRESSURE40,
   WORKLOAD_FOUR_RESOURCE_MIX,
};

static const char *
workload_name(enum workload workload)
{
   switch (workload) {
   case WORKLOAD_ADD3:
      return "add3";
   case WORKLOAD_SPARSE_BINDINGS:
      return "sparse-bindings";
   case WORKLOAD_AOS_LOAD:
      return "aos-load";
   case WORKLOAD_AOS_STORE:
      return "aos-store";
   case WORKLOAD_TWO_LOAD_FALU:
      return "two-load-falu";
   case WORKLOAD_SIX_PENDING:
      return "six-pending";
   case WORKLOAD_SEVEN_PENDING:
      return "seven-pending";
   case WORKLOAD_FANOUT:
      return "fanout";
   case WORKLOAD_TWO_LOAD_AND:
      return "two-load-and";
   case WORKLOAD_TWO_LOAD_OR:
      return "two-load-or";
   case WORKLOAD_TWO_LOAD_XOR:
      return "two-load-xor";
   case WORKLOAD_VECTOR_SWIZZLE:
      return "vector-swizzle";
   case WORKLOAD_VECTOR_FANOUT:
      return "vector-fanout";
   case WORKLOAD_VECTOR2_COPY:
      return "vector2-copy";
   case WORKLOAD_VECTOR3_COPY:
      return "vector3-copy";
   case WORKLOAD_VECTOR4_COPY:
      return "vector4-copy";
   case WORKLOAD_VECTOR4_ALU_STORE:
      return "vector4-alu-store";
   case WORKLOAD_MIXED:
      return "mixed";
   case WORKLOAD_DEPENDENT_INDEX:
      return "dependent-index";
   case WORKLOAD_NESTED_DEPENDENT:
      return "nested-dependent";
   case WORKLOAD_SYSTEM_LOAD_INDEX:
      return "system-load-index";
   case WORKLOAD_SINGLE_LOAD_IADD:
      return "single-load-iadd";
   case WORKLOAD_TWO_LOAD_IADD:
      return "two-load-iadd";
   case WORKLOAD_AFFINE_INDEX:
      return "affine-index";
   case WORKLOAD_VARIABLE_SHL:
      return "variable-shl";
   case WORKLOAD_VARIABLE_ASHR:
      return "variable-ashr";
   case WORKLOAD_VARIABLE_USHR:
      return "variable-ushr";
   case WORKLOAD_LOADED_COMPARE:
      return "loaded-compare";
   case WORKLOAD_LOADED_FLOAT_DAG:
      return "loaded-float-dag";
   case WORKLOAD_RECIPROCAL_DIRECT:
      return "reciprocal-direct";
   case WORKLOAD_RECIPROCAL_RETAIN:
      return "reciprocal-retain";
   case WORKLOAD_RECIPROCAL_MATERIALIZED:
      return "reciprocal-materialized";
   case WORKLOAD_RECIPROCAL_DENOMINATORS:
      return "reciprocal-denominators-1-1024";
   case WORKLOAD_PRESSURE40:
      return "pressure40";
   case WORKLOAD_FOUR_RESOURCE_MIX:
      return "four-resource-mix";
   }
   fail("unknown workload");
   return NULL;
}

static enum workload
parse_workload(const char *name)
{
   for (unsigned workload = WORKLOAD_ADD3;
        workload <= WORKLOAD_FOUR_RESOURCE_MIX; ++workload) {
      if (strcmp(name, workload_name((enum workload)workload)) == 0)
         return (enum workload)workload;
   }
   fprintf(stderr, "unknown workload: %s\n", name);
   exit(2);
}

static GLuint
build_program(enum workload workload)
{
   static const char add3[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { float v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { float v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { float v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  outbuf.v[i] = a.v[i] + b.v[i];\n"
      "}\n";

   static const char sparse_bindings[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=3) readonly buffer InputA { float v[]; } a;\n"
      "layout(std430, binding=7) readonly buffer InputB { float v[]; } b;\n"
      "layout(std430, binding=5) buffer Output { float v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  outbuf.v[i] = a.v[i] + b.v[i];\n"
      "}\n";

   static const char aos_load[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "struct Item { uint p0; uint value; uint p2; uint p3; };\n"
      "layout(std430, binding=0) readonly buffer InputA { Item v[]; } a;\n"
      "layout(std430, binding=1) buffer Output { uint v[]; } outbuf;\n"
      "void main() { uint i=gl_GlobalInvocationID.x; "
      "outbuf.v[i]=a.v[i].value^0x6d5a4b39u; }\n";

   static const char aos_store[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "struct Item { uint p0; uint p1; uint value; uint p3; };\n"
      "layout(std430, binding=0) readonly buffer InputA { uint v[]; } a;\n"
      "layout(std430, binding=1) buffer Output { Item v[]; } outbuf;\n"
      "void main() { uint i=gl_GlobalInvocationID.x; "
      "outbuf.v[i].value=a.v[i]^0x31415927u; }\n";

   static const char two_load_falu[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { float v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { float v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { float v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  float x = a.v[i];\n"
      "  float y = a.v[i + 1u];\n"
      "  outbuf.v[i] = x + y;\n"
      "}\n";

   static const char six_pending[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { float v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { float v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { float v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  float p0 = a.v[i + 0u] + 1.0;\n"
      "  float p1 = a.v[i + 1u] + 2.0;\n"
      "  float p2 = a.v[i + 2u] + 3.0;\n"
      "  float p3 = a.v[i + 3u] + 4.0;\n"
      "  float p4 = a.v[i + 4u] + 5.0;\n"
      "  float p5 = a.v[i + 5u] + 6.0;\n"
      "  float q0 = p0 + p1;\n"
      "  float q1 = p2 + p3;\n"
      "  float q2 = p4 + p5;\n"
      "  outbuf.v[i] = (q0 + q1) + q2;\n"
      "}\n";

   static const char seven_pending[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { float v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { float v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { float v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  float p0 = a.v[i + 0u] + 1.0;\n"
      "  float p1 = a.v[i + 1u] + 2.0;\n"
      "  float p2 = a.v[i + 2u] + 3.0;\n"
      "  float p3 = a.v[i + 3u] + 4.0;\n"
      "  float p4 = a.v[i + 4u] + 5.0;\n"
      "  float p5 = a.v[i + 5u] + 6.0;\n"
      "  float p6 = a.v[i + 6u] + 7.0;\n"
      "  float q0 = p0 + p1;\n"
      "  float q1 = p2 + p3;\n"
      "  float q2 = p4 + p5;\n"
      "  outbuf.v[i] = ((q0 + q1) + q2) + p6;\n"
      "}\n";

   static const char fanout[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { float v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { float v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { float v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  float x = a.v[i];\n"
      "  float y = b.v[i];\n"
      "  float sum = x + y;\n"
      "  float difference = x - y;\n"
      "  outbuf.v[i] = sum * difference;\n"
      "}\n";

   static const char mixed[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uint v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { uint v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  float f = uintBitsToFloat(a.v[i]) + uintBitsToFloat(b.v[i]);\n"
      "  uint logic = a.v[i + 1u] ^ b.v[i + 1u];\n"
      "  outbuf.v[i] = floatBitsToUint(f) ^ logic;\n"
      "}\n";

   static const char two_load_xor[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uint v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { uint v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  outbuf.v[i] = a.v[i] ^ b.v[i];\n"
      "}\n";

   static const char two_load_and[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uint v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { uint v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  outbuf.v[i] = a.v[i] & b.v[i];\n"
      "}\n";

   static const char two_load_or[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uint v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { uint v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  outbuf.v[i] = a.v[i] | b.v[i];\n"
      "}\n";

   static const char vector_swizzle[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uvec4 v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { uint v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  uvec4 x = a.v[i];\n"
      "  outbuf.v[i] = ((x.z ^ x.x) + (x.w & x.y)) ^ b.v[i];\n"
      "}\n";

   static const char vector_fanout[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uvec4 v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { uint v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  uvec4 x = a.v[i];\n"
      "  uint first = x.x ^ x.z;\n"
      "  uint retained = x.x | x.y;\n"
      "  outbuf.v[i] = (first + retained) ^ b.v[i];\n"
      "}\n";

   static const char vector2_copy[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uvec2 v[]; } a;\n"
      "layout(std430, binding=1) buffer Output { uvec2 v[]; } outbuf;\n"
      "void main() { uint i = gl_GlobalInvocationID.x; outbuf.v[i] = a.v[i]; }\n";

   static const char vector3_copy[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uvec3 v[]; } a;\n"
      "layout(std430, binding=1) buffer Output { uvec3 v[]; } outbuf;\n"
      "void main() { uint i = gl_GlobalInvocationID.x; outbuf.v[i] = a.v[i]; }\n";

   static const char vector4_copy[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uvec4 v[]; } a;\n"
      "layout(std430, binding=1) buffer Output { uvec4 v[]; } outbuf;\n"
      "void main() { uint i = gl_GlobalInvocationID.x; outbuf.v[i] = a.v[i]; }\n";

   static const char vector4_alu_store[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uvec4 v[]; } a;\n"
      "layout(std430, binding=1) buffer Output { uvec4 v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  uvec4 x = a.v[i];\n"
      "  outbuf.v[i] = x + uvec4(1u, 2u, 3u, 4u);\n"
      "}\n";

   static const char dependent_index[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uint v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { uint v[]; } b;\n"
      "layout(std430, binding=2) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  uint j = a.v[i] & 31u;\n"
      "  outbuf.v[i] = b.v[j] ^ i;\n"
      "}\n";

   static const char nested_dependent[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uint v[]; } a;\n"
      "layout(std430, binding=1) readonly buffer InputB { uint v[]; } b;\n"
      "layout(std430, binding=2) readonly buffer InputC { uint v[]; } c;\n"
      "layout(std430, binding=3) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  uint root = a.v[i];\n"
      "  uint j = (root ^ i) & 63u;\n"
      "  uint middle = b.v[j];\n"
      "  uint k = (middle ^ (i * 3u + 1u)) & 63u;\n"
      "  uint leaf = c.v[k];\n"
      "  outbuf.v[i] = (leaf ^ root) ^ middle;\n"
      "}\n";

   static const char system_load_index[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430, binding=0) readonly buffer InputA { uint v[]; } a;\n"
      "layout(std430, binding=1) buffer Output { uint v[]; } outbuf;\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  uint local = gl_LocalInvocationIndex;\n"
      "  uint direct_value = a.v[local];\n"
      "  uint derived_value = a.v[(local + 17u) & 63u];\n"
      "  outbuf.v[i] = direct_value ^ derived_value;\n"
      "}\n";

   static const char single_load_iadd[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;o.v[i]=a.v[i]+7u;}\n";

   static const char two_load_iadd[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { uint v[]; } b;\n"
      "layout(std430,binding=2) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;o.v[i]=a.v[i]+b.v[i];}\n";

   static const char affine_index[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { uint v[]; } b;\n"
      "layout(std430,binding=2) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;"
      "uint j=(a.v[i]*3u+1u)&63u;o.v[i]=b.v[j]^i;}\n";

   static const char variable_shl[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { uint v[]; } b;\n"
      "layout(std430,binding=2) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;"
      "o.v[i]=a.v[i]<<(b.v[i]&31u);}\n";

   static const char variable_ashr[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { uint v[]; } b;\n"
      "layout(std430,binding=2) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;"
      "o.v[i]=uint(int(a.v[i])>>int(b.v[i]&31u));}\n";

   static const char variable_ushr[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { uint v[]; } b;\n"
      "layout(std430,binding=2) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;"
      "o.v[i]=a.v[i]>>(b.v[i]&31u);}\n";

   static const char loaded_compare[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { uint v[]; } b;\n"
      "layout(std430,binding=2) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;uint x=a.v[i],y=b.v[i];"
      "o.v[i]=uint(x<y)|(uint(x>=y)<<1u)|(uint(int(x)<int(y))<<2u)|"
      "(uint(x==y)<<3u);}\n";

   static const char loaded_float_dag[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { float v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { float v[]; } b;\n"
      "layout(std430,binding=2) buffer O { float v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;float x=a.v[i],y=b.v[i];"
      "float s=x+y,d=x-y;o.v[i]=s*d+min(x,y);}\n";

   static const char reciprocal_direct[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { float v[]; } a;\n"
      "layout(std430,binding=1) buffer O { float v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;o.v[i]=1.0/a.v[i];}\n";

   static const char reciprocal_retain[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { float v[]; } a;\n"
      "layout(std430,binding=1) buffer O { float v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;float x=a.v[i];"
      "o.v[i]=(1.0/x)+x;}\n";

   static const char reciprocal_materialized[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { float v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { float v[]; } b;\n"
      "layout(std430,binding=2) buffer O { float v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;"
      "o.v[i]=1.0/(a.v[i]+b.v[i]);}\n";

   static const char reciprocal_denominators[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;"
      "o.v[i]=floatBitsToUint(1.0/float(i+1u));}\n";

   static const char pressure40[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;"
      "o.v[i]=a.v[(i+0u)&255u]+a.v[(i+1u)&255u]+a.v[(i+2u)&255u]+"
      "a.v[(i+3u)&255u]+a.v[(i+4u)&255u]+a.v[(i+5u)&255u]+"
      "a.v[(i+6u)&255u]+a.v[(i+7u)&255u]+a.v[(i+8u)&255u]+"
      "a.v[(i+9u)&255u]+a.v[(i+10u)&255u]+a.v[(i+11u)&255u]+"
      "a.v[(i+12u)&255u]+a.v[(i+13u)&255u]+a.v[(i+14u)&255u]+"
      "a.v[(i+15u)&255u]+a.v[(i+16u)&255u]+a.v[(i+17u)&255u]+"
      "a.v[(i+18u)&255u]+a.v[(i+19u)&255u]+a.v[(i+20u)&255u]+"
      "a.v[(i+21u)&255u]+a.v[(i+22u)&255u]+a.v[(i+23u)&255u]+"
      "a.v[(i+24u)&255u]+a.v[(i+25u)&255u]+a.v[(i+26u)&255u]+"
      "a.v[(i+27u)&255u]+a.v[(i+28u)&255u]+a.v[(i+29u)&255u]+"
      "a.v[(i+30u)&255u]+a.v[(i+31u)&255u]+a.v[(i+32u)&255u]+"
      "a.v[(i+33u)&255u]+a.v[(i+34u)&255u]+a.v[(i+35u)&255u]+"
      "a.v[(i+36u)&255u]+a.v[(i+37u)&255u]+a.v[(i+38u)&255u]+"
      "a.v[(i+39u)&255u];}\n";

   static const char four_resource_mix[] =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=1) readonly buffer B { uint v[]; } b;\n"
      "layout(std430,binding=2) readonly buffer C { uint v[]; } c;\n"
      "layout(std430,binding=3) buffer O { uint v[]; } o;\n"
      "void main(){uint i=gl_GlobalInvocationID.x;"
      "o.v[i]=((a.v[i]*3u)+b.v[i])^(c.v[i]+0x13579bdfu);}\n";

   const char *source = add3;
   switch (workload) {
   case WORKLOAD_ADD3:
      source = add3;
      break;
   case WORKLOAD_SPARSE_BINDINGS:
      source = sparse_bindings;
      break;
   case WORKLOAD_AOS_LOAD:
      source = aos_load;
      break;
   case WORKLOAD_AOS_STORE:
      source = aos_store;
      break;
   case WORKLOAD_TWO_LOAD_FALU:
      source = two_load_falu;
      break;
   case WORKLOAD_SIX_PENDING:
      source = six_pending;
      break;
   case WORKLOAD_SEVEN_PENDING:
      source = seven_pending;
      break;
   case WORKLOAD_FANOUT:
      source = fanout;
      break;
   case WORKLOAD_TWO_LOAD_AND:
      source = two_load_and;
      break;
   case WORKLOAD_TWO_LOAD_OR:
      source = two_load_or;
      break;
   case WORKLOAD_TWO_LOAD_XOR:
      source = two_load_xor;
      break;
   case WORKLOAD_VECTOR_SWIZZLE:
      source = vector_swizzle;
      break;
   case WORKLOAD_VECTOR_FANOUT:
      source = vector_fanout;
      break;
   case WORKLOAD_VECTOR2_COPY:
      source = vector2_copy;
      break;
   case WORKLOAD_VECTOR3_COPY:
      source = vector3_copy;
      break;
   case WORKLOAD_VECTOR4_COPY:
      source = vector4_copy;
      break;
   case WORKLOAD_VECTOR4_ALU_STORE:
      source = vector4_alu_store;
      break;
   case WORKLOAD_MIXED:
      source = mixed;
      break;
   case WORKLOAD_DEPENDENT_INDEX:
      source = dependent_index;
      break;
   case WORKLOAD_NESTED_DEPENDENT:
      source = nested_dependent;
      break;
   case WORKLOAD_SYSTEM_LOAD_INDEX:
      source = system_load_index;
      break;
   case WORKLOAD_SINGLE_LOAD_IADD:
      source = single_load_iadd;
      break;
   case WORKLOAD_TWO_LOAD_IADD:
      source = two_load_iadd;
      break;
   case WORKLOAD_AFFINE_INDEX:
      source = affine_index;
      break;
   case WORKLOAD_VARIABLE_SHL:
      source = variable_shl;
      break;
   case WORKLOAD_VARIABLE_ASHR:
      source = variable_ashr;
      break;
   case WORKLOAD_VARIABLE_USHR:
      source = variable_ushr;
      break;
   case WORKLOAD_LOADED_COMPARE:
      source = loaded_compare;
      break;
   case WORKLOAD_LOADED_FLOAT_DAG:
      source = loaded_float_dag;
      break;
   case WORKLOAD_RECIPROCAL_DIRECT:
      source = reciprocal_direct;
      break;
   case WORKLOAD_RECIPROCAL_RETAIN:
      source = reciprocal_retain;
      break;
   case WORKLOAD_RECIPROCAL_MATERIALIZED:
      source = reciprocal_materialized;
      break;
   case WORKLOAD_RECIPROCAL_DENOMINATORS:
      source = reciprocal_denominators;
      break;
   case WORKLOAD_PRESSURE40:
      source = pressure40;
      break;
   case WORKLOAD_FOUR_RESOURCE_MIX:
      source = four_resource_mix;
      break;
   }

   GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
   const char *sources[] = {source};
   glShaderSource(shader, 1, sources, NULL);
   glCompileShader(shader);

   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      GLsizei size = 0;
      glGetShaderInfoLog(shader, sizeof(log), &size, log);
      fprintf(stderr, "%s compute compile failed: %.*s\n",
              workload_name(workload), size, log);
      fail("compute shader compile");
   }

   GLuint program = glCreateProgram();
   glAttachShader(program, shader);
   glLinkProgram(program);
   glDeleteShader(shader);
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096];
      GLsizei size = 0;
      glGetProgramInfoLog(program, sizeof(log), &size, log);
      fprintf(stderr, "%s compute link failed: %.*s\n", workload_name(workload),
              size, log);
      fail("compute program link");
   }

   return program;
}

static uint32_t
float_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

static float
bits_float(uint32_t bits)
{
   float value;
   memcpy(&value, &bits, sizeof(value));
   return value;
}

static uint32_t
input_a_bits(size_t slot, uint32_t element)
{
   uint32_t numerator = element * 4u + 2u + (uint32_t)slot * 1024u;
   return float_bits((float)numerator * (1.0f / 16.0f));
}

static uint32_t
input_b_bits(size_t slot, uint32_t element)
{
   uint32_t numerator = element * 3u + 2u + (uint32_t)slot * 2048u;
   return float_bits((float)numerator * (1.0f / 16.0f));
}

static uint32_t
input_c_bits(size_t slot, uint32_t element)
{
   uint32_t value = 0x9e3779b9u * (element + 1u);
   value ^= (uint32_t)slot * 0x85ebca6bu;
   value ^= value >> 13;
   return value;
}

static uint32_t
reciprocal_input_bits(size_t slot, uint32_t element)
{
   /* Cycle through exact normal powers of two from 2^-10 through 2^10. The
    * reciprocal and the retain-case sum therefore both have exact FP32
    * oracles, independent of the API's allowed reciprocal error. */
   unsigned exponent = 117u + ((element + 7u * slot) % 21u);
   return exponent << 23;
}

static size_t
payload_offset(const struct buffer_layout *layout, size_t slot)
{
   if (slot >
       (SIZE_MAX - layout->first_payload_offset) / layout->payload_stride)
      fail("payload offset overflow");

   return layout->first_payload_offset + slot * layout->payload_stride;
}

static struct buffer_layout
make_layout(size_t slot_count, size_t alignment)
{
   const size_t first_payload_offset = align_up(MIN_GUARD_BYTES, alignment);
   const size_t payload_stride =
      align_up(PAYLOAD_BYTES + 2 * MIN_GUARD_BYTES, alignment);

   if (slot_count == 0 ||
       slot_count - 1 > (SIZE_MAX - first_payload_offset) / payload_stride)
      fail("buffer layout overflow");

   const size_t last_payload =
      first_payload_offset + (slot_count - 1) * payload_stride;
   if (last_payload > SIZE_MAX - PAYLOAD_BYTES - MIN_GUARD_BYTES)
      fail("buffer size overflow");

   return (struct buffer_layout){
      .first_payload_offset = first_payload_offset,
      .payload_stride = payload_stride,
      .buffer_bytes = last_payload + PAYLOAD_BYTES + MIN_GUARD_BYTES,
   };
}

static uint8_t
poison_byte(unsigned resource, size_t offset)
{
   uint32_t value = (uint32_t)offset * 0x45d9f3bu;
   value ^= 0xa5c39e17u + resource * 0x31415927u;
   value ^= value >> 16;
   return (uint8_t)(value | 1u);
}

static void
seed_buffer(uint8_t *buffer, size_t size, unsigned resource)
{
   for (size_t i = 0; i < size; ++i)
      buffer[i] = poison_byte(resource, i);
}

static void
write_word(uint8_t *buffer, size_t offset, uint32_t value)
{
   memcpy(buffer + offset, &value, sizeof(value));
}

static void
seed_inputs(uint8_t *input_a, uint8_t *input_b, uint8_t *input_c,
            const struct buffer_layout *layout, size_t slot_count)
{
   for (size_t slot = 0; slot < slot_count; ++slot) {
      const size_t base = payload_offset(layout, slot);
      for (uint32_t i = 0; i < VALUE_COUNT + INPUT_EXTRA_VALUES; ++i) {
         /*
          * All three values are exact binary fractions whose numerators stay
          * below 2^24 for this corpus. Their sums therefore have an
          * unambiguous IEEE-754 single-precision bit pattern.
          */
         write_word(input_a, base + i * sizeof(uint32_t),
                    input_a_bits(slot, i));
         write_word(input_b, base + i * sizeof(uint32_t),
                    input_b_bits(slot, i));
         write_word(input_c, base + i * sizeof(uint32_t),
                    input_c_bits(slot, i));
      }
   }
}

static void
seed_reciprocal_inputs(enum workload workload, uint8_t *input_a,
                       uint8_t *input_b, const struct buffer_layout *layout,
                       size_t slot_count)
{
   if (workload != WORKLOAD_RECIPROCAL_DIRECT &&
       workload != WORKLOAD_RECIPROCAL_RETAIN &&
       workload != WORKLOAD_RECIPROCAL_MATERIALIZED)
      return;

   for (size_t slot = 0; slot < slot_count; ++slot) {
      const size_t base = payload_offset(layout, slot);
      for (uint32_t i = 0; i < VALUE_COUNT + INPUT_EXTRA_VALUES; ++i) {
         float x = bits_float(reciprocal_input_bits(slot, i));
         float a = workload == WORKLOAD_RECIPROCAL_MATERIALIZED ? x * 0.25f
                                                                 : x;
         float b = x * 0.75f;
         write_word(input_a, base + i * sizeof(uint32_t), float_bits(a));
         write_word(input_b, base + i * sizeof(uint32_t), float_bits(b));
      }
   }
}

static void
write_expected_output(enum workload workload, uint8_t *output,
                      const struct buffer_layout *layout, size_t slot)
{
   const size_t base = payload_offset(layout, slot);
   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      if (workload == WORKLOAD_VECTOR2_COPY ||
          workload == WORKLOAD_VECTOR3_COPY ||
          workload == WORKLOAD_VECTOR4_COPY ||
          workload == WORKLOAD_VECTOR4_ALU_STORE) {
         const unsigned components = workload == WORKLOAD_VECTOR2_COPY   ? 2
                                     : workload == WORKLOAD_VECTOR3_COPY ? 3
                                                                         : 4;
         const unsigned stride = components == 2 ? 2 : 4;
         for (unsigned c = 0; c < components; ++c) {
            uint32_t value = input_a_bits(slot, i * stride + c);
            if (workload == WORKLOAD_VECTOR4_ALU_STORE)
               value += c + 1;
            write_word(output, base + (i * stride + c) * sizeof(uint32_t),
                       value);
         }
         continue;
      }

      float a[7], b;
      for (unsigned j = 0; j < 7; ++j)
         a[j] = bits_float(input_a_bits(slot, i + j));
      b = bits_float(input_b_bits(slot, i));

      uint32_t result = 0;
      switch (workload) {
      case WORKLOAD_ADD3:
      case WORKLOAD_SPARSE_BINDINGS:
         result = float_bits(a[0] + b);
         break;
      case WORKLOAD_AOS_LOAD:
         result = input_a_bits(slot, i * 4u + 1u) ^ 0x6d5a4b39u;
         break;
      case WORKLOAD_AOS_STORE:
         result = input_a_bits(slot, i) ^ 0x31415927u;
         write_word(output, base + (i * 4u + 2u) * sizeof(uint32_t), result);
         continue;
      case WORKLOAD_TWO_LOAD_FALU:
         result = float_bits(a[0] + a[1]);
         break;
      case WORKLOAD_SIX_PENDING:
      case WORKLOAD_SEVEN_PENDING: {
         float p[7];
         unsigned count = workload == WORKLOAD_SIX_PENDING ? 6 : 7;
         for (unsigned j = 0; j < count; ++j)
            p[j] = a[j] + (float)(j + 1);
         float q0 = p[0] + p[1];
         float q1 = p[2] + p[3];
         float q2 = p[4] + p[5];
         float combined = (q0 + q1) + q2;
         if (count == 7)
            combined += p[6];
         result = float_bits(combined);
         break;
      }
      case WORKLOAD_FANOUT: {
         float sum = a[0] + b;
         float difference = a[0] - b;
         result = float_bits(sum * difference);
         break;
      }
      case WORKLOAD_TWO_LOAD_AND:
         result = input_a_bits(slot, i) & input_b_bits(slot, i);
         break;
      case WORKLOAD_TWO_LOAD_OR:
         result = input_a_bits(slot, i) | input_b_bits(slot, i);
         break;
      case WORKLOAD_TWO_LOAD_XOR:
         result = input_a_bits(slot, i) ^ input_b_bits(slot, i);
         break;
      case WORKLOAD_VECTOR_SWIZZLE: {
         uint32_t x = input_a_bits(slot, i * 4u + 0u);
         uint32_t y = input_a_bits(slot, i * 4u + 1u);
         uint32_t z = input_a_bits(slot, i * 4u + 2u);
         uint32_t w = input_a_bits(slot, i * 4u + 3u);
         result = ((z ^ x) + (w & y)) ^ input_b_bits(slot, i);
         break;
      }
      case WORKLOAD_VECTOR_FANOUT: {
         uint32_t x = input_a_bits(slot, i * 4u + 0u);
         uint32_t y = input_a_bits(slot, i * 4u + 1u);
         uint32_t z = input_a_bits(slot, i * 4u + 2u);
         uint32_t first = x ^ z;
         uint32_t retained = x | y;
         result = (first + retained) ^ input_b_bits(slot, i);
         break;
      }
      case WORKLOAD_VECTOR2_COPY:
      case WORKLOAD_VECTOR3_COPY:
      case WORKLOAD_VECTOR4_COPY:
      case WORKLOAD_VECTOR4_ALU_STORE:
         fail("unreachable vector-copy oracle");
         break;
      case WORKLOAD_MIXED: {
         uint32_t sum = float_bits(a[0] + b);
         uint32_t logic = input_a_bits(slot, i + 1) ^ input_b_bits(slot, i + 1);
         result = sum ^ logic;
         break;
      }
      case WORKLOAD_DEPENDENT_INDEX: {
         uint32_t j = input_a_bits(slot, i) & 31u;
         result = input_b_bits(slot, j) ^ i;
         break;
      }
      case WORKLOAD_NESTED_DEPENDENT: {
         uint32_t root = input_a_bits(slot, i);
         uint32_t j = (root ^ i) & 63u;
         uint32_t middle = input_b_bits(slot, j);
         uint32_t k = (middle ^ (i * 3u + 1u)) & 63u;
         uint32_t leaf = input_c_bits(slot, k);
         result = (leaf ^ root) ^ middle;
         break;
      }
      case WORKLOAD_SYSTEM_LOAD_INDEX: {
         uint32_t local = i & 31u;
         result =
            input_a_bits(slot, local) ^ input_a_bits(slot, (local + 17u) & 63u);
         break;
      }
      case WORKLOAD_SINGLE_LOAD_IADD:
         result = input_a_bits(slot, i) + 7u;
         break;
      case WORKLOAD_TWO_LOAD_IADD:
         result = input_a_bits(slot, i) + input_b_bits(slot, i);
         break;
      case WORKLOAD_AFFINE_INDEX: {
         uint32_t j = (input_a_bits(slot, i) * 3u + 1u) & 63u;
         result = input_b_bits(slot, j) ^ i;
         break;
      }
      case WORKLOAD_VARIABLE_SHL:
         result = input_a_bits(slot, i) << (input_b_bits(slot, i) & 31u);
         break;
      case WORKLOAD_VARIABLE_ASHR:
         result = (uint32_t)((int32_t)input_a_bits(slot, i) >>
                             (input_b_bits(slot, i) & 31u));
         break;
      case WORKLOAD_VARIABLE_USHR:
         result = input_a_bits(slot, i) >> (input_b_bits(slot, i) & 31u);
         break;
      case WORKLOAD_LOADED_COMPARE: {
         uint32_t x = input_a_bits(slot, i), y = input_b_bits(slot, i);
         result = (x < y) | ((x >= y) << 1u) |
                  (((int32_t)x < (int32_t)y) << 2u) | ((x == y) << 3u);
         break;
      }
      case WORKLOAD_LOADED_FLOAT_DAG: {
         float sum = a[0] + b;
         float difference = a[0] - b;
         result = float_bits(sum * difference + (a[0] < b ? a[0] : b));
         break;
      }
      case WORKLOAD_RECIPROCAL_DIRECT: {
         float x = bits_float(reciprocal_input_bits(slot, i));
         result = float_bits(1.0f / x);
         break;
      }
      case WORKLOAD_RECIPROCAL_RETAIN: {
         float x = bits_float(reciprocal_input_bits(slot, i));
         result = float_bits((1.0f / x) + x);
         break;
      }
      case WORKLOAD_RECIPROCAL_MATERIALIZED: {
         float x = bits_float(reciprocal_input_bits(slot, i));
         result = float_bits(1.0f / x);
         break;
      }
      case WORKLOAD_RECIPROCAL_DENOMINATORS:
         fail("workload uses a dedicated oracle");
         break;
      case WORKLOAD_PRESSURE40:
         for (unsigned j = 0; j < 40; ++j)
            result += input_a_bits(slot, (i + j) & 255u);
         break;
      case WORKLOAD_FOUR_RESOURCE_MIX:
         result = (input_a_bits(slot, i) * 3u + input_b_bits(slot, i)) ^
                  (input_c_bits(slot, i) + 0x13579bdfu);
         break;
      }
      write_word(output, base + i * sizeof(uint32_t), result);
   }
}

static void
report_byte_mismatch(const char *name, const uint8_t *actual,
                     const uint8_t *expected, size_t size)
{
   for (size_t i = 0; i < size; ++i) {
      if (actual[i] != expected[i]) {
         size_t word_offset = i & ~(sizeof(uint32_t) - 1);
         uint32_t actual_word = 0, expected_word = 0;
         if (word_offset + sizeof(uint32_t) <= size) {
            memcpy(&actual_word, actual + word_offset, sizeof(actual_word));
            memcpy(&expected_word, expected + word_offset,
                   sizeof(expected_word));
         }
         fprintf(stderr,
                 "%s byte %#zx=%#x expected=%#x word[%#zx]=%#x expected=%#x\n",
                 name, i, actual[i], expected[i], word_offset, actual_word,
                 expected_word);
         fail("guarded buffer mismatch");
      }
   }
}

static void
verify_buffer(GLuint buffer, const char *name, const uint8_t *expected,
              size_t size)
{
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   const uint8_t *mapped =
      glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, GL_MAP_READ_BIT);
   if (!mapped)
      fail("map guarded buffer");

   report_byte_mismatch(name, mapped, expected, size);
   if (!glUnmapBuffer(GL_SHADER_STORAGE_BUFFER))
      fail("unmap guarded buffer");
}

static void
run_reciprocal_denominator_case(void)
{
   GLint alignment_value = 0;
   glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment_value);
   if (alignment_value <= 0)
      fail("invalid reciprocal SSBO alignment");

   const size_t alignment = (size_t)alignment_value > sizeof(uint32_t)
                               ? (size_t)alignment_value
                               : sizeof(uint32_t);
   const size_t payload_offset = align_up(MIN_GUARD_BYTES, alignment);
   const size_t payload_size =
      RECIPROCAL_DENOMINATOR_COUNT * sizeof(uint32_t);
   const size_t buffer_size = payload_offset + payload_size + MIN_GUARD_BYTES;
   uint8_t *initial = malloc(buffer_size);
   if (!initial)
      fail("allocate reciprocal denominator buffer");
   seed_buffer(initial, buffer_size, 0);

   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, buffer_size, initial,
                GL_DYNAMIC_COPY);
   glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffer, payload_offset,
                     payload_size);
   check_gl("create reciprocal denominator buffer");

   GLuint program = build_program(WORKLOAD_RECIPROCAL_DENOMINATORS);
   glUseProgram(program);
   glDispatchCompute(RECIPROCAL_DENOMINATOR_COUNT / LOCAL_SIZE_X, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();
   check_gl("execute reciprocal denominator sweep");

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   const uint8_t *mapped =
      glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffer_size,
                       GL_MAP_READ_BIT);
   if (!mapped)
      fail("map reciprocal denominator buffer");

   report_byte_mismatch("reciprocal leading guard", mapped, initial,
                        payload_offset);
   report_byte_mismatch("reciprocal trailing guard",
                        mapped + payload_offset + payload_size,
                        initial + payload_offset + payload_size,
                        MIN_GUARD_BYTES);

   const double bound = 0x1p-18;
   double worst_error = 0.0;
   uint32_t worst_denominator = 0;
   uint32_t worst_bits = 0;
   for (uint32_t denominator = 1;
        denominator <= RECIPROCAL_DENOMINATOR_COUNT; ++denominator) {
      uint32_t bits = 0;
      memcpy(&bits,
             mapped + payload_offset +
                (denominator - 1u) * sizeof(uint32_t),
             sizeof(bits));
      float reciprocal = bits_float(bits);
      if ((bits & 0x80000000u) || (bits & 0x7f800000u) == 0x7f800000u ||
          reciprocal == 0.0f) {
         fprintf(stderr, "reciprocal D=%u returned bits=%#010x\n",
                 denominator, bits);
         fail("invalid reciprocal result");
      }

      const double product = (double)denominator * (double)reciprocal;
      const double error = product >= 1.0 ? product - 1.0 : 1.0 - product;
      if (error > worst_error) {
         worst_error = error;
         worst_denominator = denominator;
         worst_bits = bits;
      }
      if (error > bound) {
         fprintf(stderr,
                 "reciprocal D=%u bits=%#010x value=%a residual=%.17g "
                 "bound=%.17g\n",
                 denominator, bits, (double)reciprocal, error, bound);
         fail("reciprocal precision contract violated");
      }
   }

   printf("T8132_APPLE9_RECIPROCAL_PRECISION_PASS denominators=%u "
          "worst_d=%u worst_bits=%#010x worst_value=%a residual=%.17g "
          "bound=%.17g\n",
          RECIPROCAL_DENOMINATOR_COUNT, worst_denominator, worst_bits,
          (double)bits_float(worst_bits), worst_error, bound);

   if (!glUnmapBuffer(GL_SHADER_STORAGE_BUFFER))
      fail("unmap reciprocal denominator buffer");
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(initial);
}

static uint32_t
superset_input(unsigned binding, uint32_t index)
{
   return ((index * (0x01010101u + binding * 0x00110011u)) ^
           (0xa5a50000u | (binding * 0x1111u + 0x31u)));
}

static void
run_superset_resource_case(unsigned resource_count)
{
   if (resource_count < 1 || resource_count > 8)
      fail("invalid superset resource count");

   const unsigned input_count = resource_count - 1;
   char source[8192];
   size_t used = 0;
   append_source(source, sizeof(source), &used,
                 "#version 310 es\nlayout(local_size_x=32) in;\n");
   for (unsigned binding = 0; binding < input_count; ++binding)
      append_source(source, sizeof(source), &used,
                    "layout(std430,binding=%u) readonly buffer I%u "
                    "{ uint v[]; } i%u;\n",
                    binding, binding, binding);
   append_source(source, sizeof(source), &used,
                 "layout(std430,binding=%u) buffer O { uint v[]; } o;\n"
                 "void main(){uint x=gl_GlobalInvocationID.x;"
                 "uint value=0x6d2b79f5u;",
                 input_count);
   for (unsigned binding = 0; binding < input_count; ++binding)
      append_source(source, sizeof(source), &used,
                    "value=value*33u+i%u.v[x];", binding);
   append_source(source, sizeof(source), &used, "o.v[x]=value;}\n");

   GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
   const char *sources[] = {source};
   glShaderSource(shader, 1, sources, NULL);
   glCompileShader(shader);
   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      GLsizei size = 0;
      glGetShaderInfoLog(shader, sizeof(log), &size, log);
      fprintf(stderr, "superset-%u compile failed: %.*s\n", resource_count,
              size, log);
      fail("superset compute shader compile");
   }
   GLuint program = glCreateProgram();
   glAttachShader(program, shader);
   glLinkProgram(program);
   glDeleteShader(shader);
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok)
      fail("superset compute program link");

   GLint alignment_value = 0;
   glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment_value);
   if (alignment_value <= 0)
      fail("invalid superset SSBO alignment");
   const size_t alignment =
      (size_t)alignment_value > sizeof(uint32_t)
         ? (size_t)alignment_value
         : sizeof(uint32_t);
   struct buffer_layout layout = make_layout(1, alignment);
   GLuint buffers[8] = {0};
   uint8_t *initial[8] = {0};
   uint8_t *expected_output = malloc(layout.buffer_bytes);
   if (!expected_output)
      fail("allocate superset output oracle");
   glGenBuffers(resource_count, buffers);
   const size_t base = payload_offset(&layout, 0);
   for (unsigned binding = 0; binding < resource_count; ++binding) {
      initial[binding] = malloc(layout.buffer_bytes);
      if (!initial[binding])
         fail("allocate superset buffer image");
      seed_buffer(initial[binding], layout.buffer_bytes, binding);
      if (binding < input_count) {
         for (uint32_t i = 0; i < VALUE_COUNT; ++i)
            write_word(initial[binding], base + i * sizeof(uint32_t),
                       superset_input(binding, i));
      }
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[binding]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes,
                   initial[binding], GL_STATIC_DRAW);
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, binding, buffers[binding],
                        base, PAYLOAD_BYTES);
   }
   memcpy(expected_output, initial[input_count], layout.buffer_bytes);
   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      uint32_t value = 0x6d2b79f5u;
      for (unsigned binding = 0; binding < input_count; ++binding)
         value = value * 33u + superset_input(binding, i);
      write_word(expected_output, base + i * sizeof(uint32_t), value);
   }

   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE_X, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();
   check_gl("superset dispatch");
   for (unsigned binding = 0; binding < input_count; ++binding) {
      char name[32];
      snprintf(name, sizeof(name), "superset-%u input-%u", resource_count,
               binding);
      verify_buffer(buffers[binding], name, initial[binding],
                    layout.buffer_bytes);
   }
   char output_name[32];
   snprintf(output_name, sizeof(output_name), "superset-%u output",
            resource_count);
   verify_buffer(buffers[input_count], output_name, expected_output,
                 layout.buffer_bytes);

   glDeleteProgram(program);
   glDeleteBuffers(resource_count, buffers);
   for (unsigned binding = 0; binding < resource_count; ++binding)
      free(initial[binding]);
   free(expected_output);
}

static void
verify_completed_payloads(GLuint output, const struct buffer_layout *layout,
                          size_t completed_slots, enum workload workload)
{
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, output);
   const uint8_t *mapped = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, layout->buffer_bytes, GL_MAP_READ_BIT);
   if (!mapped)
      fail("map output payloads");

   bool saw_nonzero = false;
   for (size_t slot = 0; slot < completed_slots; ++slot) {
      size_t base = payload_offset(layout, slot);
      for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
         uint32_t value;
         size_t element = workload == WORKLOAD_AOS_STORE ? i * 4u + 2u : i;
         memcpy(&value, mapped + base + element * sizeof(value), sizeof(value));
         saw_nonzero |= value != 0;
      }
   }

   if (!glUnmapBuffer(GL_SHADER_STORAGE_BUFFER))
      fail("unmap output payloads");
   if (!saw_nonzero)
      fail("all completed results are zero");
}

static const char *const memory_cases[] = {
   "superset-1",
   "superset-2",
   "superset-3",
   "superset-4",
   "superset-5",
   "superset-6",
   "superset-7",
   "superset-8",
   "add3",
   "sparse-bindings",
   "aos-load",
   "aos-store",
   "two-load-falu",
   "six-pending",
   "seven-pending",
   "fanout",
   "two-load-and",
   "two-load-or",
   "two-load-xor",
   "vector-swizzle",
   "vector-fanout",
   "vector2-copy",
   "vector3-copy",
   "vector4-copy",
   "vector4-alu-store",
   "mixed",
   "dependent-index",
   "nested-dependent",
   "system-load-index",
   "single-load-iadd",
   "two-load-iadd",
   "affine-index",
   "variable-shl",
   "variable-ashr",
   "variable-ushr",
   "loaded-compare",
   "loaded-float-dag",
   "reciprocal-direct",
   "reciprocal-retain",
   "reciprocal-materialized",
   "reciprocal-denominators-1-1024",
   "pressure40",
   "four-resource-mix",
};

const char *const *
t8132_apple9_memory_case_names(size_t *count)
{
   *count = sizeof(memory_cases) / sizeof(memory_cases[0]);
   return memory_cases;
}

void
t8132_apple9_run_memory_case(const char *name)
{
   if (!strcmp(name, "reciprocal-denominators-1-1024")) {
      run_reciprocal_denominator_case();
      return;
   }

   if (strncmp(name, "superset-", 9) == 0 && name[9] >= '1' &&
       name[9] <= '8' && name[10] == '\0') {
      run_superset_resource_case(name[9] - '0');
      return;
   }

   enum workload workload = parse_workload(name);
   if (workload > WORKLOAD_FOUR_RESOURCE_MIX)
      fail("workload is not an individual memory case");

   const bool has_input_c = workload == WORKLOAD_NESTED_DEPENDENT ||
                            workload == WORKLOAD_FOUR_RESOURCE_MIX;
   const bool has_one_input =
      workload == WORKLOAD_VECTOR2_COPY || workload == WORKLOAD_VECTOR3_COPY ||
      workload == WORKLOAD_VECTOR4_COPY ||
      workload == WORKLOAD_VECTOR4_ALU_STORE ||
      workload == WORKLOAD_SYSTEM_LOAD_INDEX || workload == WORKLOAD_AOS_LOAD ||
      workload == WORKLOAD_AOS_STORE || workload == WORKLOAD_SINGLE_LOAD_IADD ||
      workload == WORKLOAD_RECIPROCAL_DIRECT ||
      workload == WORKLOAD_RECIPROCAL_RETAIN ||
      workload == WORKLOAD_PRESSURE40;
   const unsigned input_count = has_input_c ? 3 : has_one_input ? 1 : 2;
   const unsigned output_binding = input_count;

   GLint alignment_value = 0;
   glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment_value);
   if (alignment_value <= 0)
      fail("invalid SSBO offset alignment");
   size_t alignment = (size_t)alignment_value;
   if (alignment < sizeof(uint32_t))
      alignment = sizeof(uint32_t);

   struct buffer_layout layout = make_layout(1, alignment);
   if (layout.buffer_bytes > (size_t)INTPTR_MAX)
      fail("guarded buffer is too large");

   uint8_t *input_a_seed = malloc(layout.buffer_bytes);
   uint8_t *input_b_seed = malloc(layout.buffer_bytes);
   uint8_t *input_c_seed = malloc(layout.buffer_bytes);
   uint8_t *output_expected = malloc(layout.buffer_bytes);
   if (!input_a_seed || !input_b_seed || !input_c_seed || !output_expected)
      fail("allocate guarded buffers");

   seed_buffer(input_a_seed, layout.buffer_bytes, 0);
   seed_buffer(input_b_seed, layout.buffer_bytes, 1);
   seed_buffer(input_c_seed, layout.buffer_bytes, 2);
   seed_buffer(output_expected, layout.buffer_bytes, 3);
   seed_inputs(input_a_seed, input_b_seed, input_c_seed, &layout, 1);
   seed_reciprocal_inputs(workload, input_a_seed, input_b_seed, &layout, 1);

   GLuint buffers[4] = {0};
   glGenBuffers(4, buffers);
   const void *buffer_seeds[] = {
      input_a_seed,
      input_b_seed,
      input_c_seed,
      output_expected,
   };
   for (unsigned i = 0; i < 4; ++i) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[i]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes,
                   buffer_seeds[i], GL_DYNAMIC_COPY);
   }
   check_gl("create guarded SSBOs");

   GLuint program = build_program(workload);
   glUseProgram(program);
   size_t offset = payload_offset(&layout, 0);
   if (workload == WORKLOAD_SPARSE_BINDINGS) {
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, buffers[0], offset,
                        PAYLOAD_BYTES);
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 7, buffers[1], offset,
                        PAYLOAD_BYTES);
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 5, buffers[3], offset,
                        PAYLOAD_BYTES);
   } else {
      for (unsigned binding = 0; binding < input_count; ++binding) {
         glBindBufferRange(GL_SHADER_STORAGE_BUFFER, binding, buffers[binding],
                           offset, PAYLOAD_BYTES);
      }
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, output_binding, buffers[3],
                        offset, PAYLOAD_BYTES);
   }

   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE_X, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();
   check_gl("execute memory case");

   write_expected_output(workload, output_expected, &layout, 0);
   verify_buffer(buffers[0], "input-a", input_a_seed, layout.buffer_bytes);
   verify_buffer(buffers[1], "input-b", input_b_seed, layout.buffer_bytes);
   verify_buffer(buffers[2], "input-c", input_c_seed, layout.buffer_bytes);
   verify_buffer(buffers[3], "output", output_expected, layout.buffer_bytes);
   verify_completed_payloads(buffers[3], &layout, 1, workload);

   glDeleteProgram(program);
   glDeleteBuffers(4, buffers);
   free(output_expected);
   free(input_c_seed);
   free(input_b_seed);
   free(input_a_seed);
}
