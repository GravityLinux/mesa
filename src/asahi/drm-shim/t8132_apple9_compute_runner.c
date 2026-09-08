/* SPDX-License-Identifier: MIT */

/* Native Piglit runner for GLES 3.1 -> NIR -> Apple9 compute tests. */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "t8132_apple9_compute_cases.h"

#define VALUE_COUNT     16384u
#define LOCAL_SIZE      256u
#define MIN_GUARD_BYTES 256u
#define ARRAY_SIZE(x)   (sizeof(x) / sizeof((x)[0]))

enum workload {
   WORKLOAD_CONSTANT,
   WORKLOAD_CONSTANT32,
   WORKLOAD_CONSTANT32_SPARSE,
   WORKLOAD_GID,
   WORKLOAD_MAD,
   WORKLOAD_DAG,
   WORKLOAD_REUSE_DAG,
   WORKLOAD_SELECT_DAG,
   WORKLOAD_COMPARE_DAG,
   WORKLOAD_COMPARE_COMPLETE,
   WORKLOAD_DEEP_INT_DAG,
   WORKLOAD_DIAMOND_INT_DAG,
   WORKLOAD_FANOUT_INT_DAG,
   WORKLOAD_LOGIC_LIFETIME_DAG,
   WORKLOAD_PRESSURE_INT_DAG,
   WORKLOAD_MINMAX_INT_DAG,
   WORKLOAD_NESTED_SELECT_DAG,
   WORKLOAD_DEEP_FLOAT_DAG,
   WORKLOAD_FANOUT_FLOAT_DAG,
   WORKLOAD_MIXED_DOMAIN_DAG,
   WORKLOAD_RADIX_ALTERNATING_DAG,
   WORKLOAD_SELECT_ALL_LIVE_DAG,
   WORKLOAD_MINMAX_NESTED_LIVE_DAG,
   WORKLOAD_FMA_ALL_LIVE_DAG,
   WORKLOAD_FLOAT_CACHE_RING_DAG,
   WORKLOAD_CROSS_DOMAIN_CACHE_DAG,
   WORKLOAD_LOGIC_MINMAX_SELECT_DAG,
   WORKLOAD_CACHE_PRESSURE_DAG,
   WORKLOAD_ADD,
   WORKLOAD_SUB,
   WORKLOAD_RSUB,
   WORKLOAD_MUL,
   WORKLOAD_AND,
   WORKLOAD_OR,
   WORKLOAD_XOR,
   WORKLOAD_NOT,
   WORKLOAD_INEG,
   WORKLOAD_U2F,
   WORKLOAD_U2F_LAST_USE,
   WORKLOAD_I2F,
   WORKLOAD_I2F_RETAINED,
   WORKLOAD_F2I,
   WORKLOAD_F2U,
   WORKLOAD_SHL,
   WORKLOAD_ASHR,
   WORKLOAD_USHR,
   WORKLOAD_IMIN,
   WORKLOAD_IMAX,
   WORKLOAD_UMIN,
   WORKLOAD_UMAX,
   WORKLOAD_FADD,
   WORKLOAD_FSUB,
   WORKLOAD_RFSUB,
   WORKLOAD_FMUL,
   WORKLOAD_FMIN,
   WORKLOAD_FMAX,
   WORKLOAD_FABS,
   WORKLOAD_FNEG,
   WORKLOAD_FMA,
   WORKLOAD_FMA_NAN_MUL,
   WORKLOAD_ARCHIVE_CROSS_0,
   WORKLOAD_ARCHIVE_CROSS_1,
   WORKLOAD_ARCHIVE_CROSS_2,
   WORKLOAD_ARCHIVE_CROSS_3,
   WORKLOAD_ARCHIVE_CROSS_4,
   WORKLOAD_ARCHIVE_CROSS_5,
   WORKLOAD_ARCHIVE_CROSS_6,
   WORKLOAD_ARCHIVE_CROSS_7,
   WORKLOAD_COUNT,
};

static const char *workload_names[WORKLOAD_COUNT] = {
   [WORKLOAD_CONSTANT] = "constant",
   [WORKLOAD_CONSTANT32] = "constant32",
   [WORKLOAD_CONSTANT32_SPARSE] = "constant32-sparse",
   [WORKLOAD_GID] = "gid",
   [WORKLOAD_MAD] = "mad",
   [WORKLOAD_DAG] = "dag",
   [WORKLOAD_REUSE_DAG] = "reuse-dag",
   [WORKLOAD_SELECT_DAG] = "select-dag",
   [WORKLOAD_COMPARE_DAG] = "compare-dag",
   [WORKLOAD_COMPARE_COMPLETE] = "compare-complete",
   [WORKLOAD_DEEP_INT_DAG] = "deep-int-dag",
   [WORKLOAD_DIAMOND_INT_DAG] = "diamond-int-dag",
   [WORKLOAD_FANOUT_INT_DAG] = "fanout-int-dag",
   [WORKLOAD_LOGIC_LIFETIME_DAG] = "logic-lifetime-dag",
   [WORKLOAD_PRESSURE_INT_DAG] = "pressure-int-dag",
   [WORKLOAD_MINMAX_INT_DAG] = "minmax-int-dag",
   [WORKLOAD_NESTED_SELECT_DAG] = "nested-select-dag",
   [WORKLOAD_DEEP_FLOAT_DAG] = "deep-float-dag",
   [WORKLOAD_FANOUT_FLOAT_DAG] = "fanout-float-dag",
   [WORKLOAD_MIXED_DOMAIN_DAG] = "mixed-domain-dag",
   [WORKLOAD_RADIX_ALTERNATING_DAG] = "radix-alternating-dag",
   [WORKLOAD_SELECT_ALL_LIVE_DAG] = "select-all-live-dag",
   [WORKLOAD_MINMAX_NESTED_LIVE_DAG] = "minmax-nested-live-dag",
   [WORKLOAD_FMA_ALL_LIVE_DAG] = "fma-all-live-dag",
   [WORKLOAD_FLOAT_CACHE_RING_DAG] = "float-cache-ring-dag",
   [WORKLOAD_CROSS_DOMAIN_CACHE_DAG] = "cross-domain-cache-dag",
   [WORKLOAD_LOGIC_MINMAX_SELECT_DAG] = "logic-minmax-select-dag",
   [WORKLOAD_CACHE_PRESSURE_DAG] = "cache-pressure-dag",
   [WORKLOAD_ADD] = "add",
   [WORKLOAD_SUB] = "sub",
   [WORKLOAD_RSUB] = "rsub",
   [WORKLOAD_MUL] = "mul",
   [WORKLOAD_AND] = "and",
   [WORKLOAD_OR] = "or",
   [WORKLOAD_XOR] = "xor",
   [WORKLOAD_NOT] = "not",
   [WORKLOAD_INEG] = "ineg",
   [WORKLOAD_U2F] = "u2f",
   [WORKLOAD_U2F_LAST_USE] = "u2f-last-use",
   [WORKLOAD_I2F] = "i2f",
   [WORKLOAD_I2F_RETAINED] = "i2f-retained",
   [WORKLOAD_F2I] = "f2i",
   [WORKLOAD_F2U] = "f2u",
   [WORKLOAD_SHL] = "shl",
   [WORKLOAD_ASHR] = "ashr",
   [WORKLOAD_USHR] = "ushr",
   [WORKLOAD_IMIN] = "imin",
   [WORKLOAD_IMAX] = "imax",
   [WORKLOAD_UMIN] = "umin",
   [WORKLOAD_UMAX] = "umax",
   [WORKLOAD_FADD] = "fadd",
   [WORKLOAD_FSUB] = "fsub",
   [WORKLOAD_RFSUB] = "rfsub",
   [WORKLOAD_FMUL] = "fmul",
   [WORKLOAD_FMIN] = "fmin",
   [WORKLOAD_FMAX] = "fmax",
   [WORKLOAD_FABS] = "fabs",
   [WORKLOAD_FNEG] = "fneg",
   [WORKLOAD_FMA] = "fma",
   [WORKLOAD_FMA_NAN_MUL] = "fma-nan-mul",
   [WORKLOAD_ARCHIVE_CROSS_0] = "archive-cross-0",
   [WORKLOAD_ARCHIVE_CROSS_1] = "archive-cross-1",
   [WORKLOAD_ARCHIVE_CROSS_2] = "archive-cross-2",
   [WORKLOAD_ARCHIVE_CROSS_3] = "archive-cross-3",
   [WORKLOAD_ARCHIVE_CROSS_4] = "archive-cross-4",
   [WORKLOAD_ARCHIVE_CROSS_5] = "archive-cross-5",
   [WORKLOAD_ARCHIVE_CROSS_6] = "archive-cross-6",
   [WORKLOAD_ARCHIVE_CROSS_7] = "archive-cross-7",
};

static const char *workload_expressions[WORKLOAD_COUNT] = {
   [WORKLOAD_CONSTANT] = "42u",
   [WORKLOAD_CONSTANT32] = "0xdeadbeefu",
   [WORKLOAD_CONSTANT32_SPARSE] = "0x10000001u",
   [WORKLOAD_GID] = "gid",
   [WORKLOAD_MAD] = "gid * 0x01020305u + 0xdeadbeefu",
   [WORKLOAD_DAG] = "((gid * 3u) ^ (gid + 7u)) + 11u",
   [WORKLOAD_REUSE_DAG] = "(gid + 3u) ^ (gid * 2u)",
   [WORKLOAD_SELECT_DAG] =
      "((gid + 3u) < (gid * 2u)) ? (gid ^ 0x55u) : (gid + 100u)",
   [WORKLOAD_COMPARE_DAG] = "uint((gid + 3u) < (gid * 2u))",
   [WORKLOAD_ADD] = "gid + 0x12345678u",
   [WORKLOAD_SUB] = "gid - 0x12345678u",
   [WORKLOAD_RSUB] = "0x12345678u - gid",
   [WORKLOAD_MUL] = "gid * 0x01020305u",
   [WORKLOAD_AND] = "gid & 0x5a5aa5a5u",
   [WORKLOAD_OR] = "gid | 0x5a5aa5a5u",
   [WORKLOAD_XOR] = "gid ^ 0x5a5aa5a5u",
   [WORKLOAD_NOT] = "~gid",
   [WORKLOAD_INEG] = "uint(-int(gid))",
   [WORKLOAD_U2F] = "floatBitsToUint(float(gid))",
   [WORKLOAD_U2F_LAST_USE] = "floatBitsToUint(float(gid ^ 0x80000000u))",
   [WORKLOAD_I2F] = "floatBitsToUint(float(int(gid)-8192))",
   [WORKLOAD_I2F_RETAINED] =
      "floatBitsToUint(float(int(gid)-8192)) ^ uint(int(gid)-8192)",
   [WORKLOAD_F2I] =
      "uint(int(uintBitsToFloat(0x3f000000u|((gid&0x3ffu)<<12u))*37.0-20.0))",
   [WORKLOAD_F2U] =
      "uint(uintBitsToFloat(0x3f000000u|((gid&0x3ffu)<<12u))*37.0)",
   [WORKLOAD_SHL] = "gid << 9u",
   [WORKLOAD_ASHR] = "uint(int(gid^0x80000000u)>>7)",
   [WORKLOAD_USHR] = "(gid^0x80000000u)>>7u",
   [WORKLOAD_IMIN] = "uint(min(int(gid), -7))",
   [WORKLOAD_IMAX] = "uint(max(int(gid), 123))",
   [WORKLOAD_UMIN] = "min(gid, 1234u)",
   [WORKLOAD_UMAX] = "max(gid, 1234u)",
   [WORKLOAD_FADD] = "floatBitsToUint(uintBitsToFloat(gid) + 1.25)",
   [WORKLOAD_FSUB] = "floatBitsToUint(uintBitsToFloat(gid) - 1.0)",
   [WORKLOAD_RFSUB] = "floatBitsToUint(1.0 - uintBitsToFloat(gid))",
   [WORKLOAD_FMUL] = "floatBitsToUint(uintBitsToFloat(gid) * 2.0)",
   [WORKLOAD_FMIN] = "floatBitsToUint(min(uintBitsToFloat(gid), 0.0))",
   [WORKLOAD_FMAX] = "floatBitsToUint(max(uintBitsToFloat(gid), 0.0))",
   [WORKLOAD_FABS] = "floatBitsToUint(abs(uintBitsToFloat(gid)))",
   [WORKLOAD_FNEG] = "floatBitsToUint(-uintBitsToFloat(gid))",
   [WORKLOAD_FMA] = "floatBitsToUint(uintBitsToFloat(gid) * 2.0 + 1.0)",
   [WORKLOAD_FMA_NAN_MUL] = "floatBitsToUint(uintBitsToFloat(gid) * "
                            "uintBitsToFloat(0x7fc00000u) + 1.0)",
   [WORKLOAD_ARCHIVE_CROSS_0] = "gid + 0x0f1e2d3cu",
   [WORKLOAD_ARCHIVE_CROSS_1] = "gid + 0x10293847u",
   [WORKLOAD_ARCHIVE_CROSS_2] = "gid + 0x56473829u",
   [WORKLOAD_ARCHIVE_CROSS_3] = "gid + 0x89abcdefu",
   [WORKLOAD_ARCHIVE_CROSS_4] = "gid + 0xc001d00du",
   [WORKLOAD_ARCHIVE_CROSS_5] = "gid + 0x31415926u",
   [WORKLOAD_ARCHIVE_CROSS_6] = "gid + 0x27182818u",
   [WORKLOAD_ARCHIVE_CROSS_7] = "gid + 0xfeedfaceu",
};

/*
 * Larger graphs use named SSA values so fan-out and long live ranges survive
 * GLSL lowering.  The bodies still contain one ordinary SSBO store and pass
 * through the same Mesa NIR and Apple9 compiler path as the expression cases.
 */
static const char *workload_bodies[WORKLOAD_COUNT] = {
   [WORKLOAD_COMPARE_COMPLETE] = "uint ua=gid*65793u+0x80001000u;"
                                 "uint ub=(gid^0xdeadbeefu)+0x1234u;"
                                 "int ia=int(ua);int ib=int(ub);"
                                 "float fa=float(int(gid&255u)-128);"
                                 "float fb=float(int((gid*37u)&255u)-128);"
                                 "uint r=uint(ua<ub)|(uint(ua>=ub)<<1u)|"
                                 "(uint(ua==ub)<<2u)|(uint(ua!=ub)<<3u)|"
                                 "(uint(ia<ib)<<4u)|(uint(ia>=ib)<<5u)|"
                                 "(uint(ia==ib)<<6u)|(uint(ia!=ib)<<7u)|"
                                 "(uint(fa<fb)<<8u)|(uint(fa>=fb)<<9u)|"
                                 "(uint(fa==fb)<<10u)|(uint(fa!=fb)<<11u);"
                                 "output0.v[gid]=r;",
   [WORKLOAD_DEEP_INT_DAG] = "uint a=gid*3u+0x00010203u;"
                             "uint b=(a^0xa5a5a5a5u)+gid*5u;"
                             "uint c=(b|(gid+17u))^(a&0x00ff00ffu);"
                             "uint d=c*9u+(b^0xdeadbeefu);"
                             "uint e=(d-(a|0x1234u))^(c+0x76543210u);"
                             "output0.v[gid]=e*7u+(d^b);",
   [WORKLOAD_DIAMOND_INT_DAG] =
      "uint root=gid*13u+0x10203040u;"
      "uint left0=(root+0x11111111u)^0x55aa55aau;"
      "uint right0=(root^0xa5a5a5a5u)+0x01020305u;"
      "uint left1=left0*3u+(right0^root);"
      "uint right1=right0*5u^(left0+root);"
      "uint join0=(left1^right1)+(left0|right0);"
      "uint join1=(left1+root)^(right1+left0);"
      "output0.v[gid]=(join0*7u)^(join1*11u)^(left1+right0);",
   [WORKLOAD_FANOUT_INT_DAG] = "uint base=gid*257u+17u;"
                               "uint a=base+0x11111111u;"
                               "uint b=base^0xa5a5a5a5u;"
                               "uint c=base*7u;"
                               "uint d=base|0x01010101u;"
                               "uint p=(a^b)+(c^d);"
                               "uint q=(a+c)^(b+d);"
                               "uint r=(a|d)^(b&c);"
                               "output0.v[gid]=(p*3u+q*5u)^r;",
   [WORKLOAD_LOGIC_LIFETIME_DAG] = "uint a=gid+3u;"
                                   "uint b=gid*5u;"
                                   "uint x=a^b;"
                                   "uint y=a|0x55aa55aau;"
                                   "uint z=b&0xf0f00f0fu;"
                                   "uint w=(x^y)+(z^a);"
                                   "output0.v[gid]=(w|b)^(x&(y+z));",
   [WORKLOAD_PRESSURE_INT_DAG] = "uint a0=gid+1u;"
                                 "uint a1=gid*3u;"
                                 "uint a2=gid^0x13579bdfu;"
                                 "uint a3=gid|0x01010101u;"
                                 "uint a4=gid&0xfefefefeu;"
                                 "uint a5=gid+0x2468ace0u;"
                                 "uint a6=gid*11u;"
                                 "uint a7=~gid;"
                                 "uint p=(a0^a1)+(a2^a3)+(a4^a5)+(a6^a7);"
                                 "uint q=(a0+a2)^(a1+a3)^(a4+a6)^(a5+a7);"
                                 "output0.v[gid]=p+q;",
   [WORKLOAD_MINMAX_INT_DAG] = "uint a=gid*65793u+0x80001000u;"
                               "uint b=(gid^0xdeadbeefu)+0x1234u;"
                               "uint u0=min(a,b);"
                               "uint u1=max(a,b);"
                               "uint i0=uint(min(int(a),int(b)));"
                               "uint i1=uint(max(int(a),int(b)));"
                               "output0.v[gid]=(u0^i1)+(u1^i0)+(a^b);",
   [WORKLOAD_NESTED_SELECT_DAG] = "uint a=gid*3u+5u;"
                                  "uint b=(gid^0x55aa55aau)+7u;"
                                  "uint c=gid+100u;"
                                  "uint d=gid*2u+1u;"
                                  "uint s0=(a<b)?(a^c):(b+d);"
                                  "uint s1=(c<d)?(s0+a):(s0^b);"
                                  "uint s2=(s0<s1)?(s1+d):(s0+c);"
                                  "output0.v[gid]=s2^(a+b);",
   [WORKLOAD_DEEP_FLOAT_DAG] = "float x=uintBitsToFloat(gid|0x3f800000u);"
                               "float a=x*2.0+0.5;"
                               "float b=x*0.5+0.25;"
                               "float c=max(a,b);"
                               "float d=min(a+b,c*2.0);"
                               "float e=abs((d-4.0)+(a-b));"
                               "output0.v[gid]=floatBitsToUint(e*0.5+0.125);",
   [WORKLOAD_FANOUT_FLOAT_DAG] = "float x=uintBitsToFloat(gid|0x3f800000u);"
                                 "float a=x*2.0+0.25;"
                                 "float b=x*0.5+0.125;"
                                 "float c=a+b;"
                                 "float d=a-b;"
                                 "float e=a*b;"
                                 "float f=max(c,e);"
                                 "float g=min(d+2.0,e*0.5);"
                                 "output0.v[gid]=floatBitsToUint((f+g)*d);",
   [WORKLOAD_MIXED_DOMAIN_DAG] =
      "uint u=(gid*0x00010203u)^0x5a5aa5a5u;"
      "float x=uintBitsToFloat((u&0x007fffffu)|0x3f800000u);"
      "float a=x*2.0+0.5;"
      "float b=max(a,x+0.25);"
      "uint bits=floatBitsToUint(b*0.5);"
      "output0.v[gid]=(bits^u)+gid*7u;",
   [WORKLOAD_RADIX_ALTERNATING_DAG] = "uint a=gid*0x10101010u+0xf000000fu;"
                                      "uint b=(gid^0x10010001u)+0x0f0000f0u;"
                                      "uint c=a*0x01000101u+(b^0x90000009u);"
                                      "output0.v[gid]=(c+a)^(b*0x00100001u);",
   [WORKLOAD_SELECT_ALL_LIVE_DAG] = "uint a=gid*3u+5u;"
                                    "uint b=(gid^0x55aa55aau)+7u;"
                                    "uint t=a^(gid+0x101u);"
                                    "uint f=b+(gid*2u+9u);"
                                    "uint s=(a<b)?t:f;"
                                    "output0.v[gid]=s^(a*5u+b*7u+t*11u+f*13u);",
   [WORKLOAD_MINMAX_NESTED_LIVE_DAG] =
      "uint a=gid*65793u+0x80001000u;"
      "uint b=(gid^0xdeadbeefu)+0x1234u;"
      "uint c=gid*17u+0x10203040u;"
      "uint d=(gid+0x76543210u)^0xa5a5a5a5u;"
      "uint lo0=min(a,b);"
      "uint hi0=max(a,b);"
      "uint lo1=min(lo0,c);"
      "uint hi1=max(hi0,d);"
      "uint slo=uint(min(int(a),int(c)));"
      "uint shi=uint(max(int(b),int(d)));"
      "output0.v[gid]=(lo1^hi1)+(slo^shi)+(a+b+c+d+lo0+hi0);",
   [WORKLOAD_FMA_ALL_LIVE_DAG] =
      "float x=uintBitsToFloat((gid&0x3fffu)|0x3f800000u);"
      "float a=x+0.25;"
      "float b=x*0.5+0.125;"
      "float c=x*2.0-0.5;"
      "precise float r=fma(a,b,c);"
      "precise float q=fma(c,a,b);"
      "output0.v[gid]=floatBitsToUint((r+q)+(a-b)+(c*0.25));",
   [WORKLOAD_FLOAT_CACHE_RING_DAG] =
      "float x=uintBitsToFloat((gid&0x3fffu)|0x3f800000u);"
      "float a=x*2.0+0.5;"
      "float b=x*0.5+0.25;"
      "float c=x+0.125;"
      "precise float p0=fma(a,b,c);"
      "float p1=max(p0,a);"
      "precise float p2=fma(p1,b,c);"
      "float p3=min(p2,p0);"
      "precise float p4=fma(p3,a,p1);"
      "output0.v[gid]=floatBitsToUint((p4+p2)-(p3-b));",
   [WORKLOAD_CROSS_DOMAIN_CACHE_DAG] =
      "float x=uintBitsToFloat((gid&0x3fffu)|0x3f800000u);"
      "float fa=x*2.0+0.25;"
      "float fb=x*0.5+0.125;"
      "uint a=floatBitsToUint(fa);"
      "uint b=floatBitsToUint(fb);"
      "uint lo=min(a,b);"
      "uint hi=max(a,b);"
      "uint t=(lo<hi)?(a^gid):(b+gid);"
      "output0.v[gid]=(t+lo)^(hi+a+b);",
   [WORKLOAD_LOGIC_MINMAX_SELECT_DAG] =
      "uint a=gid*9u+0x10203u;"
      "uint b=(gid^0xa5a55a5au)+0x12345u;"
      "uint x=a^b;"
      "uint y=(a|0x55005500u)+(b&0x0ff00ff0u);"
      "uint lo=min(x,y);"
      "uint hi=max(x,y);"
      "uint t=x+lo;"
      "uint f=y^hi;"
      "uint s=(lo<hi)?t:f;"
      "output0.v[gid]=s^(x+y+lo+hi+a+b+t+f);",
   [WORKLOAD_CACHE_PRESSURE_DAG] =
      "uint a0=gid+0x101u;"
      "uint a1=gid*3u+0x202u;"
      "uint a2=(gid^0x13579bdfu)+0x303u;"
      "uint a3=(gid|0x01010101u)+0x404u;"
      "uint a4=(gid&0xfefefefeu)+0x505u;"
      "uint a5=gid*11u+0x606u;"
      "uint lo0=min(a0,a1);"
      "uint hi0=max(a2,a3);"
      "uint lo1=min(a4,a5);"
      "uint s=(lo0<hi0)?lo1:a5;"
      "output0.v[gid]=s+a0+(a1^lo0)+(a2^hi0)+a3+a4+a5+lo1;",
};

static void
fail(const char *message)
{
   fprintf(stderr, "T8132_GLES_COMPUTE_FAIL: %s (EGL=%#x GL=%#x)\n", message,
           eglGetError(), glGetError());
   exit(1);
}

static EGLDisplay
open_asahi_display(void)
{
   PFNEGLQUERYDEVICESEXTPROC query_devices =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
   PFNEGLQUERYDEVICESTRINGEXTPROC query_string =
      (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress(
         "eglQueryDeviceStringEXT");
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
         "eglGetPlatformDisplayEXT");

   if (!query_devices || !query_string || !get_display)
      fail("EGL device enumeration unavailable");

   EGLDeviceEXT devices[16];
   EGLint count = 0;
   if (!query_devices(16, devices, &count))
      fail("eglQueryDevicesEXT");

   for (EGLint i = 0; i < count; ++i) {
      const char *render =
         query_string(devices[i], EGL_DRM_RENDER_NODE_FILE_EXT);
      if (!render || !strstr(render, "renderD"))
         continue;

      EGLDisplay display =
         get_display(EGL_PLATFORM_DEVICE_EXT, devices[i], NULL);
      if (display != EGL_NO_DISPLAY)
         return display;
   }

   fail("Asahi DRM-shim EGL device not found");
   return EGL_NO_DISPLAY;
}

static GLuint
build_compute_source(const char *source)
{
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
      fprintf(stderr, "compute compile failed: %.*s\n", size, log);
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
      fprintf(stderr, "compute link failed: %.*s\n", size, log);
      fail("compute program link");
   }
   return program;
}

static GLuint
build_program(enum workload workload)
{
   const char *expression = workload_expressions[workload];
   const char *body = workload_bodies[workload];
   char statement[3072];
   if (body) {
      snprintf(statement, sizeof(statement), "%s", body);
   } else {
      snprintf(statement, sizeof(statement), "output0.v[gid]=%s;", expression);
   }

   unsigned glsl_version = (workload == WORKLOAD_FMA_ALL_LIVE_DAG ||
                            workload == WORKLOAD_FLOAT_CACHE_RING_DAG)
                              ? 320
                              : 310;
   char source[4096];
   snprintf(source, sizeof(source),
            "#version %u es\n"
            "layout(local_size_x=256) in;\n"
            "layout(std430, binding=0) buffer Output { uint v[]; } output0;\n"
            "void main() { uint gid=gl_GlobalInvocationID.x; %s }\n",
            glsl_version, statement);

   return build_compute_source(source);
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
expected(enum workload workload, uint32_t gid)
{
   float value = bits_float(gid);
   switch (workload) {
   case WORKLOAD_CONSTANT:
      return 42;
   case WORKLOAD_CONSTANT32:
      return 0xdeadbeef;
   case WORKLOAD_CONSTANT32_SPARSE:
      return 0x10000001;
   case WORKLOAD_GID:
      return gid;
   case WORKLOAD_MAD:
      return gid * 0x01020305u + 0xdeadbeefu;
   case WORKLOAD_DAG:
      return ((gid * 3u) ^ (gid + 7u)) + 11u;
   case WORKLOAD_REUSE_DAG:
      return (gid + 3u) ^ (gid * 2u);
   case WORKLOAD_SELECT_DAG:
      return (gid + 3u) < (gid * 2u) ? (gid ^ 0x55u) : (gid + 100u);
   case WORKLOAD_COMPARE_DAG:
      return (gid + 3u) < (gid * 2u);
   case WORKLOAD_COMPARE_COMPLETE: {
      uint32_t ua = gid * 65793u + 0x80001000u;
      uint32_t ub = (gid ^ 0xdeadbeefu) + 0x1234u;
      int32_t ia = (int32_t)ua;
      int32_t ib = (int32_t)ub;
      float fa = (float)((int32_t)(gid & 255u) - 128);
      float fb = (float)((int32_t)((gid * 37u) & 255u) - 128);
      return ((uint32_t)(ua < ub) << 0) | ((uint32_t)(ua >= ub) << 1) |
             ((uint32_t)(ua == ub) << 2) | ((uint32_t)(ua != ub) << 3) |
             ((uint32_t)(ia < ib) << 4) | ((uint32_t)(ia >= ib) << 5) |
             ((uint32_t)(ia == ib) << 6) | ((uint32_t)(ia != ib) << 7) |
             ((uint32_t)(fa < fb) << 8) | ((uint32_t)(fa >= fb) << 9) |
             ((uint32_t)(fa == fb) << 10) | ((uint32_t)(fa != fb) << 11);
   }
   case WORKLOAD_DEEP_INT_DAG: {
      uint32_t a = gid * 3u + 0x00010203u;
      uint32_t b = (a ^ 0xa5a5a5a5u) + gid * 5u;
      uint32_t c = (b | (gid + 17u)) ^ (a & 0x00ff00ffu);
      uint32_t d = c * 9u + (b ^ 0xdeadbeefu);
      uint32_t e = (d - (a | 0x1234u)) ^ (c + 0x76543210u);
      return e * 7u + (d ^ b);
   }
   case WORKLOAD_DIAMOND_INT_DAG: {
      uint32_t root = gid * 13u + 0x10203040u;
      uint32_t left0 = (root + 0x11111111u) ^ 0x55aa55aau;
      uint32_t right0 = (root ^ 0xa5a5a5a5u) + 0x01020305u;
      uint32_t left1 = left0 * 3u + (right0 ^ root);
      uint32_t right1 = right0 * 5u ^ (left0 + root);
      uint32_t join0 = (left1 ^ right1) + (left0 | right0);
      uint32_t join1 = (left1 + root) ^ (right1 + left0);
      return (join0 * 7u) ^ (join1 * 11u) ^ (left1 + right0);
   }
   case WORKLOAD_FANOUT_INT_DAG: {
      uint32_t base = gid * 257u + 17u;
      uint32_t a = base + 0x11111111u;
      uint32_t b = base ^ 0xa5a5a5a5u;
      uint32_t c = base * 7u;
      uint32_t d = base | 0x01010101u;
      uint32_t p = (a ^ b) + (c ^ d);
      uint32_t q = (a + c) ^ (b + d);
      uint32_t r = (a | d) ^ (b & c);
      return (p * 3u + q * 5u) ^ r;
   }
   case WORKLOAD_LOGIC_LIFETIME_DAG: {
      uint32_t a = gid + 3u;
      uint32_t b = gid * 5u;
      uint32_t x = a ^ b;
      uint32_t y = a | 0x55aa55aau;
      uint32_t z = b & 0xf0f00f0fu;
      uint32_t w = (x ^ y) + (z ^ a);
      return (w | b) ^ (x & (y + z));
   }
   case WORKLOAD_PRESSURE_INT_DAG: {
      uint32_t a0 = gid + 1u;
      uint32_t a1 = gid * 3u;
      uint32_t a2 = gid ^ 0x13579bdfu;
      uint32_t a3 = gid | 0x01010101u;
      uint32_t a4 = gid & 0xfefefefeu;
      uint32_t a5 = gid + 0x2468ace0u;
      uint32_t a6 = gid * 11u;
      uint32_t a7 = ~gid;
      uint32_t p = (a0 ^ a1) + (a2 ^ a3) + (a4 ^ a5) + (a6 ^ a7);
      uint32_t q = (a0 + a2) ^ (a1 + a3) ^ (a4 + a6) ^ (a5 + a7);
      return p + q;
   }
   case WORKLOAD_MINMAX_INT_DAG: {
      uint32_t a = gid * 65793u + 0x80001000u;
      uint32_t b = (gid ^ 0xdeadbeefu) + 0x1234u;
      uint32_t u0 = a < b ? a : b;
      uint32_t u1 = a > b ? a : b;
      uint32_t i0 = (int32_t)a < (int32_t)b ? a : b;
      uint32_t i1 = (int32_t)a > (int32_t)b ? a : b;
      return (u0 ^ i1) + (u1 ^ i0) + (a ^ b);
   }
   case WORKLOAD_NESTED_SELECT_DAG: {
      uint32_t a = gid * 3u + 5u;
      uint32_t b = (gid ^ 0x55aa55aau) + 7u;
      uint32_t c = gid + 100u;
      uint32_t d = gid * 2u + 1u;
      uint32_t s0 = a < b ? a ^ c : b + d;
      uint32_t s1 = c < d ? s0 + a : s0 ^ b;
      uint32_t s2 = s0 < s1 ? s1 + d : s0 + c;
      return s2 ^ (a + b);
   }
   case WORKLOAD_DEEP_FLOAT_DAG: {
      float x = bits_float(gid | 0x3f800000u);
      float a = x * 2.0f + 0.5f;
      float b = x * 0.5f + 0.25f;
      float c = a > b ? a : b;
      float sum = a + b;
      float twice = c * 2.0f;
      float d = sum < twice ? sum : twice;
      float e = (d - 4.0f) + (a - b);
      if (e < 0.0f)
         e = -e;
      return float_bits(e * 0.5f + 0.125f);
   }
   case WORKLOAD_FANOUT_FLOAT_DAG: {
      float x = bits_float(gid | 0x3f800000u);
      float a = x * 2.0f + 0.25f;
      float b = x * 0.5f + 0.125f;
      float c = a + b;
      float d = a - b;
      float e = a * b;
      float f = c > e ? c : e;
      float g0 = d + 2.0f;
      float g1 = e * 0.5f;
      float g = g0 < g1 ? g0 : g1;
      return float_bits((f + g) * d);
   }
   case WORKLOAD_MIXED_DOMAIN_DAG: {
      uint32_t u = (gid * 0x00010203u) ^ 0x5a5aa5a5u;
      float x = bits_float((u & 0x007fffffu) | 0x3f800000u);
      float a = x * 2.0f + 0.5f;
      float other = x + 0.25f;
      float b = a > other ? a : other;
      uint32_t bits = float_bits(b * 0.5f);
      return (bits ^ u) + gid * 7u;
   }
   case WORKLOAD_RADIX_ALTERNATING_DAG: {
      uint32_t a = gid * 0x10101010u + 0xf000000fu;
      uint32_t b = (gid ^ 0x10010001u) + 0x0f0000f0u;
      uint32_t c = a * 0x01000101u + (b ^ 0x90000009u);
      return (c + a) ^ (b * 0x00100001u);
   }
   case WORKLOAD_SELECT_ALL_LIVE_DAG: {
      uint32_t a = gid * 3u + 5u;
      uint32_t b = (gid ^ 0x55aa55aau) + 7u;
      uint32_t t = a ^ (gid + 0x101u);
      uint32_t f = b + (gid * 2u + 9u);
      uint32_t s = a < b ? t : f;
      return s ^ (a * 5u + b * 7u + t * 11u + f * 13u);
   }
   case WORKLOAD_MINMAX_NESTED_LIVE_DAG: {
      uint32_t a = gid * 65793u + 0x80001000u;
      uint32_t b = (gid ^ 0xdeadbeefu) + 0x1234u;
      uint32_t c = gid * 17u + 0x10203040u;
      uint32_t d = (gid + 0x76543210u) ^ 0xa5a5a5a5u;
      uint32_t lo0 = a < b ? a : b;
      uint32_t hi0 = a > b ? a : b;
      uint32_t lo1 = lo0 < c ? lo0 : c;
      uint32_t hi1 = hi0 > d ? hi0 : d;
      uint32_t slo = (int32_t)a < (int32_t)c ? a : c;
      uint32_t shi = (int32_t)b > (int32_t)d ? b : d;
      return (lo1 ^ hi1) + (slo ^ shi) + (a + b + c + d + lo0 + hi0);
   }
   case WORKLOAD_FMA_ALL_LIVE_DAG: {
      float x = bits_float((gid & 0x3fffu) | 0x3f800000u);
      float a = x + 0.25f;
      float b = x * 0.5f + 0.125f;
      float c = x * 2.0f - 0.5f;
      float r = fmaf(a, b, c);
      float q = fmaf(c, a, b);
      return float_bits((r + q) + (a - b) + (c * 0.25f));
   }
   case WORKLOAD_FLOAT_CACHE_RING_DAG: {
      float x = bits_float((gid & 0x3fffu) | 0x3f800000u);
      float a = x * 2.0f + 0.5f;
      float b = x * 0.5f + 0.25f;
      float c = x + 0.125f;
      float p0 = fmaf(a, b, c);
      float p1 = p0 > a ? p0 : a;
      float p2 = fmaf(p1, b, c);
      float p3 = p2 < p0 ? p2 : p0;
      float p4 = fmaf(p3, a, p1);
      return float_bits((p4 + p2) - (p3 - b));
   }
   case WORKLOAD_CROSS_DOMAIN_CACHE_DAG: {
      float x = bits_float((gid & 0x3fffu) | 0x3f800000u);
      float fa = x * 2.0f + 0.25f;
      float fb = x * 0.5f + 0.125f;
      uint32_t a = float_bits(fa);
      uint32_t b = float_bits(fb);
      uint32_t lo = a < b ? a : b;
      uint32_t hi = a > b ? a : b;
      uint32_t t = lo < hi ? a ^ gid : b + gid;
      return (t + lo) ^ (hi + a + b);
   }
   case WORKLOAD_LOGIC_MINMAX_SELECT_DAG: {
      uint32_t a = gid * 9u + 0x10203u;
      uint32_t b = (gid ^ 0xa5a55a5au) + 0x12345u;
      uint32_t x = a ^ b;
      uint32_t y = (a | 0x55005500u) + (b & 0x0ff00ff0u);
      uint32_t lo = x < y ? x : y;
      uint32_t hi = x > y ? x : y;
      uint32_t t = x + lo;
      uint32_t f = y ^ hi;
      uint32_t s = lo < hi ? t : f;
      return s ^ (x + y + lo + hi + a + b + t + f);
   }
   case WORKLOAD_CACHE_PRESSURE_DAG: {
      uint32_t a0 = gid + 0x101u;
      uint32_t a1 = gid * 3u + 0x202u;
      uint32_t a2 = (gid ^ 0x13579bdfu) + 0x303u;
      uint32_t a3 = (gid | 0x01010101u) + 0x404u;
      uint32_t a4 = (gid & 0xfefefefeu) + 0x505u;
      uint32_t a5 = gid * 11u + 0x606u;
      uint32_t lo0 = a0 < a1 ? a0 : a1;
      uint32_t hi0 = a2 > a3 ? a2 : a3;
      uint32_t lo1 = a4 < a5 ? a4 : a5;
      uint32_t s = lo0 < hi0 ? lo1 : a5;
      return s + a0 + (a1 ^ lo0) + (a2 ^ hi0) + a3 + a4 + a5 + lo1;
   }
   case WORKLOAD_ADD:
      return gid + 0x12345678u;
   case WORKLOAD_SUB:
      return gid - 0x12345678u;
   case WORKLOAD_RSUB:
      return 0x12345678u - gid;
   case WORKLOAD_MUL:
      return gid * 0x01020305u;
   case WORKLOAD_AND:
      return gid & 0x5a5aa5a5u;
   case WORKLOAD_OR:
      return gid | 0x5a5aa5a5u;
   case WORKLOAD_XOR:
      return gid ^ 0x5a5aa5a5u;
   case WORKLOAD_NOT:
      return ~gid;
   case WORKLOAD_INEG:
      return 0u - gid;
   case WORKLOAD_U2F:
      return float_bits((float)gid);
   case WORKLOAD_U2F_LAST_USE:
      return float_bits((float)(gid ^ 0x80000000u));
   case WORKLOAD_I2F:
      return float_bits((float)((int32_t)gid - 8192));
   case WORKLOAD_I2F_RETAINED: {
      int32_t value = (int32_t)gid - 8192;
      return float_bits((float)value) ^ (uint32_t)value;
   }
   case WORKLOAD_F2I: {
      float x = bits_float(0x3f000000u | ((gid & 0x3ffu) << 12u));
      return (uint32_t)(int32_t)(x * 37.0f - 20.0f);
   }
   case WORKLOAD_F2U: {
      float x = bits_float(0x3f000000u | ((gid & 0x3ffu) << 12u));
      return (uint32_t)(x * 37.0f);
   }
   case WORKLOAD_SHL:
      return gid << 9;
   case WORKLOAD_ASHR:
      return (uint32_t)((int32_t)(gid ^ 0x80000000u) >> 7);
   case WORKLOAD_USHR:
      return (gid ^ 0x80000000u) >> 7;
   case WORKLOAD_IMIN:
      return (uint32_t)(((int32_t)gid < -7) ? (int32_t)gid : -7);
   case WORKLOAD_IMAX:
      return (uint32_t)(((int32_t)gid > 123) ? (int32_t)gid : 123);
   case WORKLOAD_UMIN:
      return gid < 1234 ? gid : 1234;
   case WORKLOAD_UMAX:
      return gid > 1234 ? gid : 1234;
   case WORKLOAD_FADD:
      return float_bits(value + 1.25f);
   case WORKLOAD_FSUB:
      return float_bits(value - 1.0f);
   case WORKLOAD_RFSUB:
      return float_bits(1.0f - value);
   /* Apple9 flushes the gid-derived subnormal input to zero for fmul. */
   case WORKLOAD_FMUL:
      return 0;
   case WORKLOAD_FMIN:
      return float_bits(value < 0.0f ? value : 0.0f);
   case WORKLOAD_FMAX:
      return 0; /* gid bit patterns are positive subnormals. */
   case WORKLOAD_FABS:
      return gid & 0x7fffffffu;
   case WORKLOAD_FNEG:
      return gid ^ 0x80000000u;
   case WORKLOAD_FMA:
      return float_bits(value * 2.0f + 1.0f);
   case WORKLOAD_FMA_NAN_MUL:
      return 0x7fc00000u;
   case WORKLOAD_ARCHIVE_CROSS_0:
      return gid + 0x0f1e2d3cu;
   case WORKLOAD_ARCHIVE_CROSS_1:
      return gid + 0x10293847u;
   case WORKLOAD_ARCHIVE_CROSS_2:
      return gid + 0x56473829u;
   case WORKLOAD_ARCHIVE_CROSS_3:
      return gid + 0x89abcdefu;
   case WORKLOAD_ARCHIVE_CROSS_4:
      return gid + 0xc001d00du;
   case WORKLOAD_ARCHIVE_CROSS_5:
      return gid + 0x31415926u;
   case WORKLOAD_ARCHIVE_CROSS_6:
      return gid + 0x27182818u;
   case WORKLOAD_ARCHIVE_CROSS_7:
      return gid + 0xfeedfaceu;
   case WORKLOAD_COUNT:
      break;
   }
   abort();
}

static const char *
workload_name(enum workload workload)
{
   return workload_names[workload];
}

struct output_layout {
   size_t alignment;
   size_t guard_bytes;
   size_t slot_stride;
   size_t slot_count;
   size_t buffer_bytes;
};

static size_t
align_up_size(size_t value, size_t alignment)
{
   if (!alignment || value > SIZE_MAX - (alignment - 1))
      fail("output layout overflow");

   return ((value + alignment - 1) / alignment) * alignment;
}

static struct output_layout
make_output_layout(size_t slot_count, size_t segment_bytes)
{
   GLint queried_alignment = 0;
   glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &queried_alignment);
   if (glGetError() != GL_NO_ERROR || queried_alignment <= 0)
      fail("query SSBO alignment");

   size_t alignment = queried_alignment;
   size_t guard_bytes = align_up_size(MIN_GUARD_BYTES, alignment);
   if (segment_bytes > SIZE_MAX - 2 * guard_bytes)
      fail("output slot overflow");
   size_t slot_stride =
      align_up_size(guard_bytes + segment_bytes + guard_bytes, alignment);
   if (!slot_count || slot_count > SIZE_MAX / slot_stride)
      fail("output buffer overflow");

   return (struct output_layout){
      .alignment = alignment,
      .guard_bytes = guard_bytes,
      .slot_stride = slot_stride,
      .slot_count = slot_count,
      .buffer_bytes = slot_count * slot_stride,
   };
}

static size_t
slot_output_offset(const struct output_layout *layout, size_t slot)
{
   if (slot >= layout->slot_count)
      fail("output slot out of range");

   return slot * layout->slot_stride + layout->guard_bytes;
}

static uint32_t
guard_word(size_t slot, unsigned side, size_t word, unsigned generation)
{
   uint32_t mixed = UINT32_C(0x6d2b79f5) * (uint32_t)(slot + 1) ^
                    UINT32_C(0x9e3779b9) * (generation + 1) ^
                    UINT32_C(0x85ebca6b) * (uint32_t)(word + 1) ^
                    (side ? UINT32_C(0xc3a5c85c) : UINT32_C(0x4cf5ad43));
   return mixed ?: UINT32_C(0x13579bdf);
}

static uint32_t
poison_word(enum workload workload, uint32_t gid, size_t slot,
            unsigned generation)
{
   /* The nonzero xor mask guarantees that the seed differs from the exact
    * oracle, including workloads whose correct result is entirely zero. */
   uint32_t mask =
      (UINT32_C(0xa5a5a5a4) ^ UINT32_C(0x9e3779b9) * (uint32_t)(slot + 1) ^
       UINT32_C(0x7f4a7c15) * (generation + 1) ^ gid) |
      1u;
   return expected(workload, gid) ^ mask;
}

static void
seed_output_slot(uint8_t *seed, const struct output_layout *layout, size_t slot,
                 enum workload workload, unsigned generation)
{
   const size_t output_offset = slot_output_offset(layout, slot);
   const size_t guard_words = layout->guard_bytes / sizeof(uint32_t);
   uint32_t *before = (uint32_t *)(seed + output_offset - layout->guard_bytes);
   uint32_t *output = (uint32_t *)(seed + output_offset);
   uint32_t *after =
      (uint32_t *)(seed + output_offset + VALUE_COUNT * sizeof(uint32_t));

   for (size_t i = 0; i < guard_words; ++i) {
      before[i] = guard_word(slot, 0, i, generation);
      after[i] = guard_word(slot, 1, i, generation);
   }
   for (uint32_t i = 0; i < VALUE_COUNT; ++i)
      output[i] = poison_word(workload, i, slot, generation);
}

static bool
verify_output_slot(const uint8_t *mapped, const struct output_layout *layout,
                   size_t slot, enum workload workload, unsigned generation,
                   const char *mode, unsigned ordinal)
{
   const size_t output_offset = slot_output_offset(layout, slot);
   const size_t guard_words = layout->guard_bytes / sizeof(uint32_t);
   const uint32_t *before =
      (const uint32_t *)(mapped + output_offset - layout->guard_bytes);
   const uint32_t *output = (const uint32_t *)(mapped + output_offset);
   const uint32_t *after = (const uint32_t *)(mapped + output_offset +
                                              VALUE_COUNT * sizeof(uint32_t));

   for (size_t i = 0; i < guard_words; ++i) {
      uint32_t before_want = guard_word(slot, 0, i, generation);
      uint32_t after_want = guard_word(slot, 1, i, generation);
      if (before[i] != before_want || after[i] != after_want) {
         fprintf(stderr,
                 "%s %u workload %s slot %zu guard %zu changed: "
                 "before=%#x/%#x after=%#x/%#x\n",
                 mode, ordinal, workload_name(workload), slot, i, before[i],
                 before_want, after[i], after_want);
         return false;
      }
   }

   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      uint32_t want = expected(workload, i);
      if (output[i] != want) {
         fprintf(stderr,
                 "%s %u workload %s slot %zu word %u=%#x expected=%#x "
                 "seed=%#x\n",
                 mode, ordinal, workload_name(workload), slot, i, output[i],
                 want, poison_word(workload, i, slot, generation));
         return false;
      }
   }
   return true;
}

static bool
run_formula_case(enum workload workload)
{
   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(1, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   if (!seed)
      fail("allocate formula seed");
   seed_output_slot(seed, &layout, 0, workload, 0);

   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                GL_DYNAMIC_COPY);
   GLuint program = build_program(workload);
   glUseProgram(program);
   glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffer,
                     slot_output_offset(&layout, 0), segment_bytes);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   const uint8_t *mapped = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
   if (!mapped)
      fail("map formula result");
   bool passed =
      verify_output_slot(mapped, &layout, 0, workload, 0, "formula", 1);
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(seed);
   return passed;
}

static void
run_archive_cross_sequence(void)
{
   static const enum workload sequence[] = {
      WORKLOAD_ARCHIVE_CROSS_0, WORKLOAD_ARCHIVE_CROSS_1,
      WORKLOAD_ARCHIVE_CROSS_2, WORKLOAD_ARCHIVE_CROSS_3,
      WORKLOAD_ARCHIVE_CROSS_4, WORKLOAD_ARCHIVE_CROSS_5,
      WORKLOAD_ARCHIVE_CROSS_6, WORKLOAD_ARCHIVE_CROSS_7,
   };
   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(1, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   if (!seed)
      fail("allocate archive sequence seed");
   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, NULL,
                GL_DYNAMIC_COPY);

   GLuint programs[sizeof(sequence) / sizeof(sequence[0])];
   for (unsigned i = 0; i < sizeof(programs) / sizeof(programs[0]); ++i)
      programs[i] = build_program(sequence[i]);

   for (unsigned round = 0; round < 2; ++round) {
      for (unsigned i = 0; i < sizeof(programs) / sizeof(programs[0]); ++i) {
         enum workload workload = sequence[i];
         unsigned generation = round * 8 + i;
         seed_output_slot(seed, &layout, 0, workload, generation);
         glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
         glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes,
                         seed);
         glUseProgram(programs[i]);
         glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffer,
                           slot_output_offset(&layout, 0), segment_bytes);
         glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
         glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                         GL_SHADER_STORAGE_BARRIER_BIT);
         glFinish();
         const uint8_t *mapped = glMapBufferRange(
            GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
         if (!mapped)
            fail("map archive sequence result");
         if (!verify_output_slot(mapped, &layout, 0, workload, generation,
                                 "archive sequence", generation + 1))
            fail("archive sequence output mismatch");
         glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      }
   }

   for (unsigned i = 0; i < sizeof(programs) / sizeof(programs[0]); ++i)
      glDeleteProgram(programs[i]);
   glDeleteBuffers(1, &buffer);
   free(seed);
}

static void
run_repeated_range_dispatch(void)
{
   enum { SLOT_COUNT = 8 };
   const enum workload workload = WORKLOAD_CACHE_PRESSURE_DAG;
   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(SLOT_COUNT, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   if (!seed)
      fail("allocate repeated-range seed");
   for (unsigned slot = 0; slot < SLOT_COUNT; ++slot)
      seed_output_slot(seed, &layout, slot, workload, slot);

   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                GL_DYNAMIC_COPY);
   GLuint program = build_program(workload);
   glUseProgram(program);
   for (unsigned slot = 0; slot < SLOT_COUNT; ++slot) {
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffer,
                        slot_output_offset(&layout, slot), segment_bytes);
      glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   }
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();
   const uint8_t *mapped = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
   if (!mapped)
      fail("map repeated-range result");
   for (unsigned slot = 0; slot < SLOT_COUNT; ++slot)
      if (!verify_output_slot(mapped, &layout, slot, workload, slot,
                              "repeated range", slot + 1))
         fail("repeated range output mismatch");
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(seed);
}

static GLuint
build_publication_program(unsigned slot, uint32_t value)
{
   char source[512];
   int length = snprintf(source, sizeof(source),
                         "#version 310 es\n"
                         "layout(local_size_x=32) in;\n"
                         "layout(std430,binding=0) buffer O { uint v[]; } o;\n"
                         "void main(){uint i=gl_GlobalInvocationID.x;"
                         "o.v[%uu+i]=0x%08xu^(i*0x9e3779b9u);}\n",
                         slot * 32u, value);
   if (length < 0 || (size_t)length >= sizeof(source))
      fail("publication shader source overflow");
   GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
   const char *sources[] = {source};
   glShaderSource(shader, 1, sources, NULL);
   glCompileShader(shader);
   GLint ok = GL_FALSE;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok)
      fail("compile publication shader");
   GLuint program = glCreateProgram();
   glAttachShader(program, shader);
   glLinkProgram(program);
   glDeleteShader(shader);
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok)
      fail("link publication shader");
   return program;
}

static void
run_program_lifecycle_stress(void)
{
   enum { PROGRAMS = 64 };
   const size_t words_per_program = 32;
   const size_t payload_bytes = PROGRAMS * words_per_program * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(1, payload_bytes);
   const size_t output_offset = slot_output_offset(&layout, 0);
   const size_t total_bytes = layout.buffer_bytes;
   uint8_t *expected = malloc(total_bytes);
   if (!expected)
      fail("allocate publication oracle");
   for (size_t i = 0; i < total_bytes; ++i)
      expected[i] = (uint8_t)((i * 0x5du + 0xa7u) | 1u);

   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, total_bytes, expected,
                GL_DYNAMIC_COPY);
   glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffer, output_offset,
                     payload_bytes);

   GLuint first = 0;
   for (unsigned i = 0; i < PROGRAMS; ++i) {
      uint32_t value = 0x10203040u ^ (i * 0x9e3779b9u);
      GLuint program = build_publication_program(i, value);
      if (i == 0)
         first = program;
      glUseProgram(program);
      glDispatchCompute(1, 1, 1);
      for (unsigned lane = 0; lane < words_per_program; ++lane) {
         uint32_t result = value ^ (lane * 0x9e3779b9u);
         memcpy(expected + output_offset +
                   (i * words_per_program + lane) * sizeof(result),
                &result, sizeof(result));
      }
      if (i != 0)
         glDeleteProgram(program);
      if (i == 0) {
         glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                         GL_SHADER_STORAGE_BARRIER_BIT);
         glFinish();
      }
   }

   uint32_t poison[32];
   for (unsigned lane = 0; lane < words_per_program; ++lane)
      poison[lane] = 0xfeedfaceu ^ lane;
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferSubData(GL_SHADER_STORAGE_BUFFER, output_offset, sizeof(poison),
                   poison);
   glUseProgram(first);
   glDispatchCompute(1, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   const uint8_t *actual = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                            total_bytes, GL_MAP_READ_BIT);
   if (!actual)
      fail("map publication stress output");
   if (memcmp(actual, expected, total_bytes))
      fail("publication lifecycle output mismatch");
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(first);
   glDeleteBuffers(1, &buffer);
   free(expected);
}

static void
fill_divergent_input(unsigned variant, unsigned pattern, uint32_t *input)
{
   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      if (variant < 2 || (variant >= 10 && variant < 12)) {
         if (pattern == 0) {
            static const uint32_t mixed[] = {
               0x7fffffffu,
               0x80000000u,
               0x80000001u,
               0x00004567u,
            };
            input[i] = mixed[i & 3] ^ ((i >> 2) & 0x3ffu);
         } else if (pattern == 1) {
            input[i] = 0x7fff0000u | (i & 0xffffu);
         } else {
            input[i] = 0x80000000u | i;
         }
      } else if (variant < 4) {
         if (pattern == 0) {
            static const int32_t mixed[] = {-6, -5, -4, INT32_MIN, INT32_MAX};
            input[i] = (uint32_t)mixed[i % 5];
         } else if (pattern == 1) {
            input[i] = (uint32_t)(-100000 - (int32_t)i);
         } else {
            input[i] = (uint32_t)(-5 + (int32_t)i);
         }
      } else if (variant < 6 || (variant >= 8 && variant < 10)) {
         if (pattern == 0) {
            static const uint32_t mixed[] = {
               0xc0000000u, /* -2.0 */
               0x3fa00000u, /*  1.25 */
               0x40200000u, /*  2.5 */
               0x7fc00000u, /* quiet NaN */
               0x00000000u, /* +0.0 */
               0x80000000u, /* -0.0 */
            };
            input[i] = mixed[i % 6];
         } else {
            const bool equality = variant >= 8;
            input[i] = float_bits(pattern == 1 && equality ? 1.25f
                                  : pattern == 1           ? 0.5f
                                                           : 2.0f);
         }
      } else {
         if (pattern == 0) {
            static const uint32_t mixed[] = {
               0x7fffffffu,
               0x80000000u,
               0x80000001u,
               0x00004567u,
            };
            input[i] = mixed[i & 3];
         } else {
            input[i] = pattern == 1 ? 0x80000000u : 0x80010000u | i;
         }
      }
   }
}

static bool
divergent_condition(unsigned variant, uint32_t value)
{
   switch (variant) {
   case 0:
      return value < UINT32_C(0x80000000);
   case 1:
      return value >= UINT32_C(0x80000000);
   case 2:
      return (int32_t)value < -5;
   case 3:
      return (int32_t)value >= -5;
   case 4:
      return bits_float(value) < 1.25f;
   case 5:
      return bits_float(value) >= 1.25f;
   case 6:
      return value == UINT32_C(0x80000000);
   case 7:
      return value != UINT32_C(0x80000000);
   case 8:
      return bits_float(value) == 1.25f;
   case 9:
      return bits_float(value) != 1.25f;
   case 10:
      return (value < UINT32_C(0x80000000)) != ((value & 1u) != 0);
   case 11:
      return (value & 2u) != 0 ? value < UINT32_C(0x80000000)
                               : value >= UINT32_C(0x80000000);
   default:
      fail("invalid divergent condition variant");
      return false;
   }
}

static void
run_simple_divergent_if_else(void)
{
   static const struct {
      const char *name;
      const char *expression;
   } variants[] = {
      {"ult", "raw < 0x80000000u"},
      {"uge", "raw >= 0x80000000u"},
      {"ilt", "int(raw) < -5"},
      {"ige", "int(raw) >= -5"},
      {"flt", "uintBitsToFloat(raw) < 1.25"},
      {"fge", "uintBitsToFloat(raw) >= 1.25"},
      {"ieq", "raw == 0x80000000u"},
      {"ine", "raw != 0x80000000u"},
      {"feq", "uintBitsToFloat(raw) == 1.25"},
      {"fne", "uintBitsToFloat(raw) != 1.25"},
      {"bool-xor", "(raw < 0x80000000u) ^^ ((raw & 1u) != 0u)"},
      {"bool-select", "((raw & 2u) != 0u) ? (raw < 0x80000000u) : "
                      "(raw >= 0x80000000u)"},
   };
   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(3, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   uint32_t *input = malloc(segment_bytes);
   if (!seed || !input)
      fail("allocate divergent-if buffers");

   GLuint output_buffer = 0, input_buffer = 0;
   glGenBuffers(1, &output_buffer);
   glGenBuffers(1, &input_buffer);

   for (unsigned variant = 0; variant < sizeof(variants) / sizeof(variants[0]);
        ++variant) {
      char source[2048];
      int length = snprintf(
         source, sizeof(source),
         "#version 310 es\n"
         "layout(local_size_x=256) in;\n"
         "layout(std430,binding=0) buffer OutA { uint v[]; } out_a;\n"
         "layout(std430,binding=1) buffer OutB { uint v[]; } out_b;\n"
         "layout(std430,binding=2) readonly buffer Input { uint v[]; } input2;\n"
         "layout(std430,binding=3) buffer Merge { uint v[]; } merge;\n"
         "void main(){\n"
         " uint gid=gl_GlobalInvocationID.x; uint raw=input2.v[gid];\n"
         " uint merged;\n"
         " if(%s) {\n"
         "  out_a.v[gid]=(raw^0x13579bdfu)+gid*3u;\n"
         "  merged=(raw*9u)+(gid^0x10203u);\n"
         " } else {\n"
         "  out_b.v[gid]=(raw+0x2468ace0u)^(gid*5u);\n"
         "  merged=(raw^0xa5a55a5au)-(gid*7u);\n"
         " }\n"
         " merge.v[gid]=merged^0x31415926u;\n"
         "}\n",
         variants[variant].expression);
      if (length < 0 || (size_t)length >= sizeof(source))
         fail("divergent-if shader source overflow");
      GLuint program = build_compute_source(source);
      glUseProgram(program);

      for (unsigned pattern = 0; pattern < 3; ++pattern) {
         const unsigned seed_id = variant * 3 + pattern;
         for (unsigned slot = 0; slot < 3; ++slot)
            seed_output_slot(seed, &layout, slot, WORKLOAD_GID, seed_id);
         fill_divergent_input(variant, pattern, input);

         glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
         glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                      GL_DYNAMIC_COPY);
         glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, output_buffer,
                           slot_output_offset(&layout, 0), segment_bytes);
         glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, output_buffer,
                           slot_output_offset(&layout, 1), segment_bytes);
         glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, output_buffer,
                           slot_output_offset(&layout, 2), segment_bytes);
         glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffer);
         glBufferData(GL_SHADER_STORAGE_BUFFER, segment_bytes, input,
                      GL_DYNAMIC_COPY);
         glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input_buffer);

         glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
         glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                         GL_SHADER_STORAGE_BARRIER_BIT);
         glFinish();

         glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
         const uint8_t *mapped = glMapBufferRange(
            GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
         if (!mapped)
            fail("map divergent-if result");

         unsigned mismatches[3] = {0};
         unsigned changed[3] = {0};
         for (unsigned slot = 0; slot < 3; ++slot) {
            const size_t offset = slot_output_offset(&layout, slot);
            const uint32_t *before =
               (const uint32_t *)(mapped + offset - layout.guard_bytes);
            const uint32_t *output = (const uint32_t *)(mapped + offset);
            const uint32_t *after =
               (const uint32_t *)(mapped + offset + segment_bytes);
            for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
               if (before[i] != guard_word(slot, 0, i, seed_id) ||
                   after[i] != guard_word(slot, 1, i, seed_id))
                  fail("divergent-if guard changed");
            }

            for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
               changed[slot] +=
                  output[i] != poison_word(WORKLOAD_GID, i, slot, seed_id);
               const bool true_arm = divergent_condition(variant, input[i]);
               const bool selected = slot == 2 || slot == (true_arm ? 0u : 1u);
               const uint32_t want =
                  selected
                     ? (slot == 0 ? (input[i] ^ UINT32_C(0x13579bdf)) + i * 3u
                        : slot == 1
                           ? (input[i] + UINT32_C(0x2468ace0)) ^ (i * 5u)
                           : ((true_arm
                                  ? input[i] * 9u + (i ^ UINT32_C(0x10203))
                                  : (input[i] ^ UINT32_C(0xa5a55a5a)) - i * 7u) ^
                              UINT32_C(0x31415926)))
                     : poison_word(WORKLOAD_GID, i, slot, seed_id);
               if (output[i] != want) {
                  if (mismatches[slot] < 4)
                     fprintf(stderr,
                             "divergent-if %s pattern %u slot %u word %u=%#x "
                             "expected=%#x input=%#x\n",
                             variants[variant].name, pattern, slot, i,
                             output[i], want, input[i]);
                  ++mismatches[slot];
               }
            }
         }
         if (mismatches[0] || mismatches[1] || mismatches[2]) {
            fprintf(stderr,
                    "divergent-if %s pattern %u summary: slot0 changed=%u "
                    "mismatches=%u, slot1 changed=%u mismatches=%u, "
                    "slot2 changed=%u mismatches=%u\n",
                    variants[variant].name, pattern, changed[0], mismatches[0],
                    changed[1], mismatches[1], changed[2], mismatches[2]);
            fail("divergent-if output mismatch");
         }
         glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      }
      glDeleteProgram(program);
   }

   glDeleteBuffers(1, &input_buffer);
   glDeleteBuffers(1, &output_buffer);
   free(input);
   free(seed);
}

enum two_source_compare_kind {
   TWO_SOURCE_ULT,
   TWO_SOURCE_UGE,
   TWO_SOURCE_ILT,
   TWO_SOURCE_IGE,
   TWO_SOURCE_FLT,
   TWO_SOURCE_FGE,
   TWO_SOURCE_IEQ,
   TWO_SOURCE_INE,
   TWO_SOURCE_FEQ,
   TWO_SOURCE_FNE,
   TWO_SOURCE_UGT,
   TWO_SOURCE_ULE,
   TWO_SOURCE_IGT,
   TWO_SOURCE_ILE,
   TWO_SOURCE_FGT,
   TWO_SOURCE_FLE,
};

struct two_source_compare_case {
   const char *name;
   const char *expression;
   enum two_source_compare_kind kind;
   unsigned live_sources;
};

static bool
evaluate_two_source_compare(enum two_source_compare_kind kind, uint32_t a,
                            uint32_t b)
{
   switch (kind) {
   case TWO_SOURCE_ULT:
      return a < b;
   case TWO_SOURCE_UGE:
      return a >= b;
   case TWO_SOURCE_ILT:
      return (int32_t)a < (int32_t)b;
   case TWO_SOURCE_IGE:
      return (int32_t)a >= (int32_t)b;
   case TWO_SOURCE_FLT:
      return bits_float(a) < bits_float(b);
   case TWO_SOURCE_FGE:
      return bits_float(a) >= bits_float(b);
   case TWO_SOURCE_IEQ:
      return a == b;
   case TWO_SOURCE_INE:
      return a != b;
   case TWO_SOURCE_FEQ:
      return bits_float(a) == bits_float(b);
   case TWO_SOURCE_FNE:
      return bits_float(a) != bits_float(b);
   case TWO_SOURCE_UGT:
      return a > b;
   case TWO_SOURCE_ULE:
      return a <= b;
   case TWO_SOURCE_IGT:
      return (int32_t)a > (int32_t)b;
   case TWO_SOURCE_ILE:
      return (int32_t)a <= (int32_t)b;
   case TWO_SOURCE_FGT:
      return bits_float(a) > bits_float(b);
   case TWO_SOURCE_FLE:
      return bits_float(a) <= bits_float(b);
   }

   fail("invalid two-source comparison");
   return false;
}

static bool
two_source_compare_is_float(enum two_source_compare_kind kind)
{
   return kind == TWO_SOURCE_FLT || kind == TWO_SOURCE_FGE ||
          kind == TWO_SOURCE_FEQ || kind == TWO_SOURCE_FNE ||
          kind == TWO_SOURCE_FGT || kind == TWO_SOURCE_FLE;
}

static bool
two_source_compare_is_signed(enum two_source_compare_kind kind)
{
   return kind == TWO_SOURCE_ILT || kind == TWO_SOURCE_IGE ||
          kind == TWO_SOURCE_IGT || kind == TWO_SOURCE_ILE;
}

static void
fill_two_source_inputs(enum two_source_compare_kind kind, unsigned pattern,
                       uint32_t *left, uint32_t *right)
{
   static const uint32_t unsigned_pairs[][2] = {
      {0, 0},
      {0, 1},
      {1, 0},
      {0x7fffffff, 0x80000000},
      {0x80000000, 0x7fffffff},
      {UINT32_MAX, UINT32_MAX},
      {UINT32_MAX, 0},
      {0x12345678, 0x12345679},
   };
   static const uint32_t signed_pairs[][2] = {
      {(uint32_t)INT32_MIN, (uint32_t)INT32_MAX},
      {(uint32_t)INT32_MAX, (uint32_t)INT32_MIN},
      {(uint32_t)-1, 0},
      {0, (uint32_t)-1},
      {(uint32_t)-5, (uint32_t)-5},
      {123456, (uint32_t)-654321},
   };
   static const uint32_t float_pairs[][2] = {
      {0x00000000, 0x80000000}, /* +0, -0 */
      {0x80000000, 0x00000000}, /* -0, +0 */
      {0x7fc00000, 0x3f800000}, /* qNaN, 1 */
      {0x3f800000, 0x7fc00000}, /* 1, qNaN */
      {0x7f800000, 0x7f800000}, /* +inf, +inf */
      {0xff800000, 0x7f800000}, /* -inf, +inf */
      {0x7f800000, 0xff800000}, /* +inf, -inf */
      {0x3fa00000, 0x3fa00000}, /* 1.25, 1.25 */
      {0x3f9fffff, 0x3fa00000}, /* adjacent finite values */
      {0x3fa00000, 0x3f9fffff},
   };

   if (pattern == 0) {
      for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
         if (two_source_compare_is_float(kind)) {
            const unsigned p =
               i % (sizeof(float_pairs) / sizeof(float_pairs[0]));
            left[i] = float_pairs[p][0];
            right[i] = float_pairs[p][1];
         } else if (two_source_compare_is_signed(kind)) {
            const unsigned p =
               i % (sizeof(signed_pairs) / sizeof(signed_pairs[0]));
            left[i] = signed_pairs[p][0];
            right[i] = signed_pairs[p][1];
         } else {
            const unsigned p =
               i % (sizeof(unsigned_pairs) / sizeof(unsigned_pairs[0]));
            left[i] = unsigned_pairs[p][0];
            right[i] = unsigned_pairs[p][1];
         }
      }
      return;
   }

   const bool desired = pattern == 1;
   static const uint32_t unsigned_candidates[][2] = {
      {1, 2},
      {2, 1},
      {1, 1},
   };
   static const uint32_t signed_candidates[][2] = {
      {(uint32_t)-2, 1},
      {1, (uint32_t)-2},
      {(uint32_t)-1, (uint32_t)-1},
   };
   static const uint32_t float_candidates[][2] = {
      {0x3f800000, 0x40000000},
      {0x40000000, 0x3f800000},
      {0x3f800000, 0x3f800000},
      {0x7fc00000, 0x3f800000},
   };
   const uint32_t (*candidates)[2] = unsigned_candidates;
   size_t candidate_count =
      sizeof(unsigned_candidates) / sizeof(unsigned_candidates[0]);
   if (two_source_compare_is_float(kind)) {
      candidates = float_candidates;
      candidate_count = sizeof(float_candidates) / sizeof(float_candidates[0]);
   } else if (two_source_compare_is_signed(kind)) {
      candidates = signed_candidates;
      candidate_count =
         sizeof(signed_candidates) / sizeof(signed_candidates[0]);
   }
   unsigned selected = UINT32_MAX;
   for (unsigned p = 0; p < candidate_count; ++p) {
      if (evaluate_two_source_compare(kind, candidates[p][0],
                                      candidates[p][1]) == desired) {
         selected = p;
         break;
      }
   }
   if (selected == UINT32_MAX)
      fail("could not construct uniform comparison population");

   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      left[i] = candidates[selected][0];
      right[i] = candidates[selected][1];
   }
}

static uint32_t
two_source_live_mix(unsigned live_sources, uint32_t a, uint32_t b)
{
   return ((live_sources & 1) ? a : 0) ^ ((live_sources & 2) ? b : 0);
}

static void
run_two_source_comparisons(void)
{
   static const struct two_source_compare_case cases[] = {
      {"ult", "a < b", TWO_SOURCE_ULT, 3},
      {"uge", "a >= b", TWO_SOURCE_UGE, 3},
      {"ilt", "int(a) < int(b)", TWO_SOURCE_ILT, 3},
      {"ige", "int(a) >= int(b)", TWO_SOURCE_IGE, 3},
      {"flt", "uintBitsToFloat(a) < uintBitsToFloat(b)", TWO_SOURCE_FLT, 3},
      {"fge", "uintBitsToFloat(a) >= uintBitsToFloat(b)", TWO_SOURCE_FGE, 3},
      {"ieq", "a == b", TWO_SOURCE_IEQ, 3},
      {"ine", "a != b", TWO_SOURCE_INE, 3},
      {"feq", "uintBitsToFloat(a) == uintBitsToFloat(b)", TWO_SOURCE_FEQ, 3},
      {"fne", "uintBitsToFloat(a) != uintBitsToFloat(b)", TWO_SOURCE_FNE, 3},
      {"ugt", "a > b", TWO_SOURCE_UGT, 3},
      {"ule", "a <= b", TWO_SOURCE_ULE, 3},
      {"igt", "int(a) > int(b)", TWO_SOURCE_IGT, 3},
      {"ile", "int(a) <= int(b)", TWO_SOURCE_ILE, 3},
      {"fgt", "uintBitsToFloat(a) > uintBitsToFloat(b)", TWO_SOURCE_FGT, 3},
      {"fle", "uintBitsToFloat(a) <= uintBitsToFloat(b)", TWO_SOURCE_FLE, 3},
      {"ult-release-both", "a < b", TWO_SOURCE_ULT, 0},
      {"ult-retain-a", "a < b", TWO_SOURCE_ULT, 1},
      {"ult-retain-b", "a < b", TWO_SOURCE_ULT, 2},
      {"ult-retain-both", "a < b", TWO_SOURCE_ULT, 3},
      {"ieq-release-both", "a == b", TWO_SOURCE_IEQ, 0},
      {"ieq-retain-a", "a == b", TWO_SOURCE_IEQ, 1},
      {"ieq-retain-b", "a == b", TWO_SOURCE_IEQ, 2},
      {"ieq-retain-both", "a == b", TWO_SOURCE_IEQ, 3},
   };
   static const char *live_expression[] = {
      "0u",
      "a",
      "b",
      "a ^ b",
   };
   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(5, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   uint32_t *left = malloc(segment_bytes);
   uint32_t *right = malloc(segment_bytes);
   if (!seed || !left || !right)
      fail("allocate two-source comparison buffers");

   GLuint output_buffer = 0;
   GLuint input_buffers[2] = {0};
   glGenBuffers(1, &output_buffer);
   glGenBuffers(2, input_buffers);

   for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
      char source[4096];
      int length = snprintf(
         source, sizeof(source),
         "#version 310 es\n"
         "layout(local_size_x=256) in;\n"
         "layout(std430,binding=0) buffer TrueOut { uint v[]; } true_out;\n"
         "layout(std430,binding=1) buffer FalseOut { uint v[]; } false_out;\n"
         "layout(std430,binding=2) readonly buffer Left { uint v[]; } lhs;\n"
         "layout(std430,binding=3) readonly buffer Right { uint v[]; } rhs;\n"
         "layout(std430,binding=4) buffer MergeA { uint v[]; } merge_a;\n"
         "layout(std430,binding=5) buffer MergeB { uint v[]; } merge_b;\n"
         "layout(std430,binding=6) buffer Pre { uint v[]; } pre;\n"
         "void main(){\n"
         " uint gid=gl_GlobalInvocationID.x; uint a=lhs.v[gid];"
         " uint b=rhs.v[gid];\n"
         " pre.v[gid]=gid^0x55aa33ccu; uint m0; uint m1;\n"
         " if(%s) {\n"
         "  true_out.v[gid]=(gid*3u+0x13579bdfu)^(%s);\n"
         "  m0=(gid+0x10203040u)^(%s);"
         "  m1=(gid*7u+0x31415926u)^(%s);\n"
         " } else {\n"
         "  false_out.v[gid]=(gid*5u+0x2468ace0u)^(%s);\n"
         "  m0=(gid+0x50607080u)^(%s);"
         "  m1=(gid*11u+0x27182818u)^(%s);\n"
         " }\n"
         " merge_a.v[gid]=m0; merge_b.v[gid]=m1;\n"
         "}\n",
         cases[c].expression, live_expression[cases[c].live_sources],
         live_expression[cases[c].live_sources],
         live_expression[cases[c].live_sources],
         live_expression[cases[c].live_sources],
         live_expression[cases[c].live_sources],
         live_expression[cases[c].live_sources]);
      if (length < 0 || (size_t)length >= sizeof(source))
         fail("two-source comparison shader source overflow");
      GLuint program = build_compute_source(source);
      glUseProgram(program);

      for (unsigned pattern = 0; pattern < 3; ++pattern) {
         const unsigned seed_id = c * 3 + pattern;
         for (unsigned slot = 0; slot < 5; ++slot)
            seed_output_slot(seed, &layout, slot, WORKLOAD_COMPARE_COMPLETE,
                             seed_id);
         fill_two_source_inputs(cases[c].kind, pattern, left, right);

         glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
         glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                      GL_DYNAMIC_COPY);
         static const unsigned output_bindings[] = {0, 1, 4, 5, 6};
         for (unsigned slot = 0;
              slot < sizeof(output_bindings) / sizeof(output_bindings[0]);
              ++slot) {
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, output_bindings[slot],
                              output_buffer, slot_output_offset(&layout, slot),
                              segment_bytes);
         }
         for (unsigned input = 0; input < 2; ++input) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffers[input]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, segment_bytes,
                         input ? right : left, GL_DYNAMIC_COPY);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + input,
                             input_buffers[input]);
         }

         glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
         glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                         GL_SHADER_STORAGE_BARRIER_BIT);
         glFinish();

         glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
         const uint8_t *mapped = glMapBufferRange(
            GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
         if (!mapped)
            fail("map two-source comparison result");

         unsigned mismatches[5] = {0};
         for (unsigned slot = 0; slot < 5; ++slot) {
            const size_t offset = slot_output_offset(&layout, slot);
            const uint32_t *before =
               (const uint32_t *)(mapped + offset - layout.guard_bytes);
            const uint32_t *output = (const uint32_t *)(mapped + offset);
            const uint32_t *after =
               (const uint32_t *)(mapped + offset + segment_bytes);
            for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
               if (before[i] != guard_word(slot, 0, i, seed_id) ||
                   after[i] != guard_word(slot, 1, i, seed_id))
                  fail("two-source comparison guard changed");
            }
            for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
               const bool condition =
                  evaluate_two_source_compare(cases[c].kind, left[i], right[i]);
               const uint32_t mix =
                  two_source_live_mix(cases[c].live_sources, left[i], right[i]);
               uint32_t want =
                  poison_word(WORKLOAD_COMPARE_COMPLETE, i, slot, seed_id);
               if (slot == 0 && condition)
                  want = (i * 3u + 0x13579bdfu) ^ mix;
               else if (slot == 1 && !condition)
                  want = (i * 5u + 0x2468ace0u) ^ mix;
               else if (slot == 2)
                  want = (i + (condition ? 0x10203040u : 0x50607080u)) ^ mix;
               else if (slot == 3)
                  want = (i * (condition ? 7u : 11u) +
                          (condition ? 0x31415926u : 0x27182818u)) ^
                         mix;
               else if (slot == 4)
                  want = i ^ 0x55aa33ccu;
               if (output[i] != want) {
                  if (mismatches[slot] < 4)
                     fprintf(stderr,
                             "two-source %s pattern %u slot %u word %u=%#x "
                             "expected=%#x a=%#x b=%#x\n",
                             cases[c].name, pattern, slot, i, output[i], want,
                             left[i], right[i]);
                  ++mismatches[slot];
               }
            }
         }
         for (unsigned slot = 0; slot < 5; ++slot) {
            if (mismatches[slot])
               fail("two-source comparison output mismatch");
         }
         glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      }
      glDeleteProgram(program);
   }

   glDeleteBuffers(2, input_buffers);
   glDeleteBuffers(1, &output_buffer);
   free(right);
   free(left);
   free(seed);
}

static void
run_compare_register_pressure(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Pre { uint v[]; } pre;\n"
      "layout(std430,binding=1) buffer Out { uint v[]; } out_buffer;\n"
      "layout(std430,binding=2) readonly buffer Input { uint v[]; } input2;\n"
      "void main(){\n"
      " uint gid=gl_GlobalInvocationID.x;\n"
      " uint p0=input2.v[gid+0u]; uint p1=input2.v[gid+16384u];\n"
      " uint p2=input2.v[gid+32768u]; uint p3=input2.v[gid+49152u];\n"
      " uint p4=input2.v[gid+65536u]; uint p5=input2.v[gid+81920u];\n"
      " uint p6=input2.v[gid+98304u]; uint p7=input2.v[gid+114688u];\n"
      " uint p8=input2.v[gid+131072u]; uint p9=input2.v[gid+147456u];\n"
      " uint p10=input2.v[gid+163840u]; uint p11=input2.v[gid+180224u];\n"
      " uint p12=input2.v[gid+196608u]; uint p13=input2.v[gid+212992u];\n"
      " uint p14=input2.v[gid+229376u]; uint p15=input2.v[gid+245760u];\n"
      " uint before=p0^p1^p2^p3^p4^p5^p6^p7^p8^p9^p10^p11^p12^p13^p14^p15;\n"
      " pre.v[gid]=before; uint merged;\n"
      " if(p0<p1) merged=(p2+p4)^(p6+p8);"
      " else merged=(p3+p5)^(p7+p9);\n"
      " out_buffer.v[gid]=merged+p0+p1+p2+p3+p4+p5+p6+p7+"
      "p8+p9+p10+p11+p12+p13+p14+p15;\n"
      "}\n";
   const unsigned value_sets = 16;
   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   const size_t input_bytes = value_sets * segment_bytes;
   struct output_layout layout = make_output_layout(2, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   uint32_t *input = malloc(input_bytes);
   if (!seed || !input)
      fail("allocate compare-pressure buffers");

   for (unsigned slot = 0; slot < 2; ++slot)
      seed_output_slot(seed, &layout, slot, WORKLOAD_PRESSURE_INT_DAG, 0);
   for (unsigned set = 0; set < value_sets; ++set) {
      for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
         input[set * VALUE_COUNT + i] = (0x9e3779b9u * (set + 1)) ^
                                        (i * (set * 2u + 3u)) ^
                                        (i >> (set & 7));
      }
   }

   GLuint program = build_compute_source(source);
   GLuint output_buffer = 0, input_buffer = 0;
   glGenBuffers(1, &output_buffer);
   glGenBuffers(1, &input_buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                GL_DYNAMIC_COPY);
   for (unsigned slot = 0; slot < 2; ++slot) {
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, slot, output_buffer,
                        slot_output_offset(&layout, slot), segment_bytes);
   }
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, input_bytes, input, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, input_buffer);
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
   const uint8_t *mapped = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
   if (!mapped)
      fail("map compare-pressure result");
   for (unsigned slot = 0; slot < 2; ++slot) {
      const size_t offset = slot_output_offset(&layout, slot);
      const uint32_t *before =
         (const uint32_t *)(mapped + offset - layout.guard_bytes);
      const uint32_t *output = (const uint32_t *)(mapped + offset);
      const uint32_t *after =
         (const uint32_t *)(mapped + offset + segment_bytes);
      for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
         if (before[i] != guard_word(slot, 0, i, 0) ||
             after[i] != guard_word(slot, 1, i, 0))
            fail("compare-pressure guard changed");
      }
      for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
         uint32_t x = 0;
         for (unsigned set = 0; set < value_sets; ++set)
            x ^= input[set * VALUE_COUNT + i];
         uint32_t sum = 0;
         for (unsigned set = 0; set < value_sets; ++set)
            sum += input[set * VALUE_COUNT + i];
         const bool condition = input[i] < input[VALUE_COUNT + i];
         const uint32_t merged =
            condition
               ? (input[2 * VALUE_COUNT + i] + input[4 * VALUE_COUNT + i]) ^
                    (input[6 * VALUE_COUNT + i] + input[8 * VALUE_COUNT + i])
               : (input[3 * VALUE_COUNT + i] + input[5 * VALUE_COUNT + i]) ^
                    (input[7 * VALUE_COUNT + i] + input[9 * VALUE_COUNT + i]);
         const uint32_t want = slot == 0 ? x : merged + sum;
         if (output[i] != want)
            fail("compare-pressure output mismatch");
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &input_buffer);
   glDeleteBuffers(1, &output_buffer);
   free(input);
   free(seed);
}

static void
run_single_region_shapes(void)
{
   static const struct {
      const char *name;
      const char *then_statement;
      const char *else_statement;
      bool has_then;
      bool has_else;
   } shapes[] = {
      {"empty", "", "", false, false},
      {"then-only", "then_out.v[gid]=gid+0x22220000u;", "", true, false},
      {"else-only", "", "else_out.v[gid]=gid+0x33330000u;", false, true},
      {"both", "then_out.v[gid]=gid+0x22220000u;",
       "else_out.v[gid]=gid+0x33330000u;", true, true},
   };
   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(4, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   if (!seed)
      fail("allocate single-region buffers");
   GLuint output_buffer = 0;
   glGenBuffers(1, &output_buffer);

   for (unsigned shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]);
        ++shape) {
      char source[2048];
      int length = snprintf(
         source, sizeof(source),
         "#version 310 es\n"
         "layout(local_size_x=256) in;\n"
         "layout(std430,binding=0) buffer Pre { uint v[]; } pre;\n"
         "layout(std430,binding=1) buffer Then { uint v[]; } then_out;\n"
         "layout(std430,binding=2) buffer Else { uint v[]; } else_out;\n"
         "layout(std430,binding=3) buffer Post { uint v[]; } post;\n"
         "void main(){ uint gid=gl_GlobalInvocationID.x;"
         " pre.v[gid]=gid^0x11112222u;"
         " if(gid<8192u){%s}else{%s}"
         " post.v[gid]=gid^0x44448888u; }\n",
         shapes[shape].then_statement, shapes[shape].else_statement);
      if (length < 0 || (size_t)length >= sizeof(source))
         fail("single-region shader source overflow");
      GLuint program = build_compute_source(source);
      for (unsigned slot = 0; slot < 4; ++slot)
         seed_output_slot(seed, &layout, slot, WORKLOAD_COMPARE_DAG, shape);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
      glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                   GL_DYNAMIC_COPY);
      for (unsigned slot = 0; slot < 4; ++slot) {
         glBindBufferRange(GL_SHADER_STORAGE_BUFFER, slot, output_buffer,
                           slot_output_offset(&layout, slot), segment_bytes);
      }
      glUseProgram(program);
      glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
      glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                      GL_SHADER_STORAGE_BARRIER_BIT);
      glFinish();

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
      const uint8_t *mapped = glMapBufferRange(
         GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
      if (!mapped)
         fail("map single-region result");
      for (unsigned slot = 0; slot < 4; ++slot) {
         const size_t offset = slot_output_offset(&layout, slot);
         const uint32_t *before =
            (const uint32_t *)(mapped + offset - layout.guard_bytes);
         const uint32_t *output = (const uint32_t *)(mapped + offset);
         const uint32_t *after =
            (const uint32_t *)(mapped + offset + segment_bytes);
         for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
            if (before[i] != guard_word(slot, 0, i, shape) ||
                after[i] != guard_word(slot, 1, i, shape))
               fail("single-region guard changed");
         }
         for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
            const bool selected =
               slot == 0 || slot == 3 ||
               (slot == 1 && shapes[shape].has_then && i < VALUE_COUNT / 2) ||
               (slot == 2 && shapes[shape].has_else && i >= VALUE_COUNT / 2);
            uint32_t want = poison_word(WORKLOAD_COMPARE_DAG, i, slot, shape);
            if (selected) {
               want = slot == 0   ? i ^ 0x11112222u
                      : slot == 1 ? i + 0x22220000u
                      : slot == 2 ? i + 0x33330000u
                                  : i ^ 0x44448888u;
            }
            if (output[i] != want) {
               fprintf(stderr,
                       "single-region %s slot %u word %u=%#x expected=%#x\n",
                       shapes[shape].name, slot, i, output[i], want);
               fail("single-region output mismatch");
            }
         }
      }
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      glDeleteProgram(program);
   }

   glDeleteBuffers(1, &output_buffer);
   free(seed);
}

static void
run_multiple_phi_vectors(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Scalars { uint v[]; } scalars;\n"
      "layout(std430,binding=1) buffer Vectors { uvec4 v[]; } vectors;\n"
      "void main(){\n"
      " uint gid=gl_GlobalInvocationID.x; uint scalar; uvec4 vector;\n"
      " if(gid<8192u){\n"
      "  scalar=gid+0x10203040u;\n"
      "  vector=uvec4(gid+1u,gid+2u,gid+3u,gid+4u);\n"
      " }else{\n"
      "  scalar=gid^0x50607080u;\n"
      "  vector=uvec4(gid^0x10u,gid^0x20u,gid^0x30u,gid^0x40u);\n"
      " }\n"
      " scalars.v[gid]=scalar; vectors.v[gid]=vector;\n"
      "}\n";
   const size_t scalar_bytes = VALUE_COUNT * sizeof(uint32_t);
   const size_t vector_bytes = 4 * scalar_bytes;
   struct output_layout layouts[] = {
      make_output_layout(1, scalar_bytes),
      make_output_layout(1, vector_bytes),
   };
   uint8_t *seed[2] = {
      malloc(layouts[0].buffer_bytes),
      malloc(layouts[1].buffer_bytes),
   };
   if (!seed[0] || !seed[1])
      fail("allocate multiple-phi buffers");

   const size_t output_words[] = {VALUE_COUNT, 4 * VALUE_COUNT};
   for (unsigned binding = 0; binding < 2; ++binding) {
      const size_t offset = slot_output_offset(&layouts[binding], 0);
      uint32_t *before =
         (uint32_t *)(seed[binding] + offset - layouts[binding].guard_bytes);
      uint32_t *output = (uint32_t *)(seed[binding] + offset);
      uint32_t *after =
         (uint32_t *)(seed[binding] + offset + output_words[binding] * 4);
      for (size_t i = 0; i < layouts[binding].guard_bytes / 4; ++i) {
         before[i] = guard_word(binding, 0, i, 0x4d50);
         after[i] = guard_word(binding, 1, i, 0x4d50);
      }
      for (size_t i = 0; i < output_words[binding]; ++i)
         output[i] = 0xc001d00du ^ (binding * 0x11111111u) ^ (uint32_t)i;
   }

   GLuint program = build_compute_source(source);
   GLuint buffers[2] = {0};
   glGenBuffers(2, buffers);
   for (unsigned binding = 0; binding < 2; ++binding) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[binding]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, layouts[binding].buffer_bytes,
                   seed[binding], GL_DYNAMIC_COPY);
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, binding, buffers[binding],
                        slot_output_offset(&layouts[binding], 0),
                        output_words[binding] * 4);
   }
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   for (unsigned binding = 0; binding < 2; ++binding) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[binding]);
      const uint8_t *mapped =
         glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                          layouts[binding].buffer_bytes, GL_MAP_READ_BIT);
      if (!mapped)
         fail("map multiple-phi result");
      const size_t offset = slot_output_offset(&layouts[binding], 0);
      const uint32_t *before =
         (const uint32_t *)(mapped + offset - layouts[binding].guard_bytes);
      const uint32_t *output = (const uint32_t *)(mapped + offset);
      const uint32_t *after =
         (const uint32_t *)(mapped + offset + output_words[binding] * 4);
      for (size_t i = 0; i < layouts[binding].guard_bytes / 4; ++i) {
         if (before[i] != guard_word(binding, 0, i, 0x4d50) ||
             after[i] != guard_word(binding, 1, i, 0x4d50))
            fail("multiple-phi guard changed");
      }
      for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
         const bool then_arm = gid < VALUE_COUNT / 2;
         if (binding == 0) {
            const uint32_t want =
               then_arm ? gid + 0x10203040u : gid ^ 0x50607080u;
            if (output[gid] != want)
               fail("multiple-phi scalar mismatch");
            continue;
         }

         for (unsigned component = 0; component < 4; ++component) {
            const uint32_t want =
               then_arm ? gid + component + 1 : gid ^ ((component + 1) * 0x10u);
            if (output[gid * 4 + component] != want)
               fail("multiple-phi vector mismatch");
         }
      }
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   }

   glDeleteProgram(program);
   glDeleteBuffers(2, buffers);
   free(seed[1]);
   free(seed[0]);
}

static void
run_nested_short_circuit_if_else(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) readonly buffer Input { uint v[]; } input0;\n"
      "layout(std430,binding=1) buffer Nested { uint v[]; } nested_out;\n"
      "layout(std430,binding=2) buffer Phi { uint v[]; } phi_out;\n"
      "layout(std430,binding=3) buffer AndRhs { uint v[]; } and_rhs;\n"
      "layout(std430,binding=4) buffer AndResult { uint v[]; } and_out;\n"
      "layout(std430,binding=5) buffer OrRhs { uint v[]; } or_rhs;\n"
      "layout(std430,binding=6) buffer OrResult { uint v[]; } or_out;\n"
      "layout(std430,binding=7) buffer Sibling { uint v[]; } sibling_out;\n"
      "bool eval_and_rhs(uint gid,uint raw){"
      " and_rhs.v[gid]=raw^0x31415926u; return (raw&64u)!=0u; }\n"
      "bool eval_or_rhs(uint gid,uint raw){"
      " or_rhs.v[gid]=raw+0x27182818u; return (raw&256u)!=0u; }\n"
      "void main(){\n"
      " uint gid=gl_GlobalInvocationID.x; uint raw=input0.v[gid];\n"
      " if((raw&1u)!=0u){"
      "  if((raw&2u)!=0u) nested_out.v[gid]=raw+0x10010010u;"
      "  else nested_out.v[gid]=raw+0x20020020u;"
      " }else{"
      "  if((raw&4u)!=0u) nested_out.v[gid]=raw^0x30030030u;"
      "  else nested_out.v[gid]=raw^0x40040040u;"
      " }\n"
      " uint merged;"
      " if((raw&8u)!=0u){"
      "  if((raw&16u)!=0u){"
      "   sibling_out.v[gid]=raw^0x11115555u;"
      "   merged=raw*3u+0x51515151u;"
      "  }else{"
      "   sibling_out.v[gid]=raw^0x22226666u;"
      "   merged=(raw^0xa5a55a5au)+0x16161616u;"
      "  }"
      " }else merged=raw*5u+0x25252525u;"
      " phi_out.v[gid]=merged^(gid*17u);\n"
      " bool and_value=((raw&32u)!=0u)&&eval_and_rhs(gid,raw);"
      " and_out.v[gid]=and_value?(raw+0x61616161u):(raw^0x62626262u);\n"
      " bool or_value=((raw&128u)!=0u)||eval_or_rhs(gid,raw);"
      " or_out.v[gid]=or_value?(raw+0x71717171u):(raw^0x72727272u);\n"
      " uint sibling=raw+0x81818181u;"
      " if((raw&512u)!=0u) sibling^=0x91919191u;"
      " if((raw&1024u)!=0u){"
      "  if((raw&2048u)!=0u) sibling+=0xa1a1a1a1u;"
      "  else sibling-=0xb2b2b2b2u;"
      " }"
      " sibling_out.v[gid]=sibling;\n"
      "}\n";

   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(7, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   uint32_t *input = malloc(segment_bytes);
   if (!seed || !input)
      fail("allocate nested control-flow buffers");

   const unsigned seed_id = 0x4e4346;
   for (unsigned slot = 0; slot < 7; ++slot)
      seed_output_slot(seed, &layout, slot, WORKLOAD_COMPARE_DAG, seed_id);
   for (uint32_t i = 0; i < VALUE_COUNT; ++i)
      input[i] = (i & 0xfffu) | ((i * 0x9e3779b9u) & 0xfffff000u);

   GLuint program = build_compute_source(source);
   GLuint output_buffer = 0, input_buffer = 0;
   glGenBuffers(1, &output_buffer);
   glGenBuffers(1, &input_buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, segment_bytes, input,
                GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                GL_DYNAMIC_COPY);
   for (unsigned slot = 0; slot < 7; ++slot) {
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, slot + 1, output_buffer,
                        slot_output_offset(&layout, slot), segment_bytes);
   }

   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
   const uint8_t *mapped = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
   if (!mapped)
      fail("map nested control-flow output");

   for (unsigned slot = 0; slot < 7; ++slot) {
      const size_t offset = slot_output_offset(&layout, slot);
      const uint32_t *before =
         (const uint32_t *)(mapped + offset - layout.guard_bytes);
      const uint32_t *output = (const uint32_t *)(mapped + offset);
      const uint32_t *after =
         (const uint32_t *)(mapped + offset + segment_bytes);
      for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
         if (before[i] != guard_word(slot, 0, i, seed_id) ||
             after[i] != guard_word(slot, 1, i, seed_id))
            fail("nested control-flow guard changed");
      }

      for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
         const uint32_t raw = input[i];
         bool selected = true;
         uint32_t want = 0;
         switch (slot) {
         case 0:
            want =
               (raw & 1u) != 0u
                  ? ((raw & 2u) != 0u ? raw + 0x10010010u : raw + 0x20020020u)
                  : ((raw & 4u) != 0u ? raw ^ 0x30030030u : raw ^ 0x40040040u);
            break;
         case 1: {
            uint32_t merged =
               (raw & 8u) != 0u
                  ? ((raw & 16u) != 0u ? raw * 3u + 0x51515151u
                                       : (raw ^ 0xa5a55a5au) + 0x16161616u)
                  : raw * 5u + 0x25252525u;
            want = merged ^ (i * 17u);
            break;
         }
         case 2:
            selected = (raw & 32u) != 0u;
            want = raw ^ 0x31415926u;
            break;
         case 3: {
            const bool result = (raw & 32u) != 0u && (raw & 64u) != 0u;
            want = result ? raw + 0x61616161u : raw ^ 0x62626262u;
            break;
         }
         case 4:
            selected = (raw & 128u) == 0u;
            want = raw + 0x27182818u;
            break;
         case 5: {
            const bool result = (raw & 128u) != 0u || (raw & 256u) != 0u;
            want = result ? raw + 0x71717171u : raw ^ 0x72727272u;
            break;
         }
         case 6:
            want = raw + 0x81818181u;
            if ((raw & 512u) != 0u)
               want ^= 0x91919191u;
            if ((raw & 1024u) != 0u)
               want =
                  (raw & 2048u) != 0u ? want + 0xa1a1a1a1u : want - 0xb2b2b2b2u;
            break;
         }

         if (!selected)
            want = poison_word(WORKLOAD_COMPARE_DAG, i, slot, seed_id);
         if (output[i] != want) {
            fprintf(stderr,
                    "nested control-flow slot %u word %u=%#x expected=%#x "
                    "input=%#x\n",
                    slot, i, output[i], want, raw);
            fail("nested control-flow output mismatch");
         }
      }
   }

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &input_buffer);
   glDeleteBuffers(1, &output_buffer);
   free(input);
   free(seed);
}

static void
run_nested_branch_local_loads(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) readonly buffer Conditions { uint v[]; } cond;\n"
      "layout(std430,binding=1) readonly buffer A { uint v[]; } a;\n"
      "layout(std430,binding=2) readonly buffer B { uint v[]; } b;\n"
      "layout(std430,binding=3) readonly buffer C { uint v[]; } c;\n"
      "layout(std430,binding=4) readonly buffer D { uint v[]; } d;\n"
      "layout(std430,binding=5) buffer Output { uint v[]; } output0;\n"
      "layout(std430,binding=6) buffer Side { uint v[]; } side;\n"
      "void main(){\n"
      " uint gid=gl_GlobalInvocationID.x; uint raw=cond.v[gid]; uint value;\n"
      " if((raw&1u)!=0u){"
      "  if((raw&2u)!=0u){"
      "   uint i=(gid*5u+3u)&16383u; uint x=a.v[i],y=a.v[(i+17u)&16383u];"
      "   side.v[gid]=(x+y)^0x11112222u; value=x*3u+(y^raw);"
      "  }else{"
      "   uint i=(gid*7u+11u)&16383u; uint x=b.v[i],y=b.v[(i+19u)&16383u];"
      "   side.v[gid]=(x^y)+0x33334444u; value=(x+raw)^(y*5u);"
      "  }"
      " }else{"
      "  if((raw&4u)!=0u){"
      "   uint i=(gid*9u+13u)&16383u; uint x=c.v[i],y=c.v[(i+23u)&16383u];"
      "   side.v[gid]=(x*7u)+y; value=(x^raw)+(y^0x55556666u);"
      "  }else{"
      "   uint i=(gid*13u+29u)&16383u; uint x=d.v[i],y=d.v[(i+31u)&16383u];"
      "   side.v[gid]=(x-y)^0x77778888u; value=(x*11u)^(y+raw);"
      "  }"
      " }"
      " output0.v[gid]=value^(gid*0x10203u);\n"
      "}\n";

   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(2, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   uint32_t *inputs[5] = {0};
   for (unsigned i = 0; i < 5; ++i)
      inputs[i] = malloc(segment_bytes);
   if (!seed || !inputs[0] || !inputs[1] || !inputs[2] || !inputs[3] ||
       !inputs[4])
      fail("allocate nested branch-load buffers");

   const unsigned seed_id = 0x4e424c;
   for (unsigned slot = 0; slot < 2; ++slot)
      seed_output_slot(seed, &layout, slot, WORKLOAD_COMPARE_DAG, seed_id);
   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      inputs[0][i] = (i & 7u) | ((i * 0x9e3779b9u) & 0xfffffff8u);
      for (unsigned set = 1; set < 5; ++set)
         inputs[set][i] =
            (0x13579bdfu * set) ^ (i * (0x10203u + set * 0x202u)) ^ (i >> set);
   }

   GLuint program = build_compute_source(source);
   GLuint input_buffers[5] = {0}, output_buffer = 0;
   glGenBuffers(5, input_buffers);
   glGenBuffers(1, &output_buffer);
   for (unsigned binding = 0; binding < 5; ++binding) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffers[binding]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, segment_bytes, inputs[binding],
                   GL_DYNAMIC_COPY);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding,
                       input_buffers[binding]);
   }
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                GL_DYNAMIC_COPY);
   for (unsigned slot = 0; slot < 2; ++slot) {
      glBindBufferRange(GL_SHADER_STORAGE_BUFFER, slot + 5, output_buffer,
                        slot_output_offset(&layout, slot), segment_bytes);
   }

   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
   const uint8_t *mapped = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
   if (!mapped)
      fail("map nested branch-load output");
   for (unsigned slot = 0; slot < 2; ++slot) {
      const size_t offset = slot_output_offset(&layout, slot);
      const uint32_t *before =
         (const uint32_t *)(mapped + offset - layout.guard_bytes);
      const uint32_t *output = (const uint32_t *)(mapped + offset);
      const uint32_t *after =
         (const uint32_t *)(mapped + offset + segment_bytes);
      for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
         if (before[i] != guard_word(slot, 0, i, seed_id) ||
             after[i] != guard_word(slot, 1, i, seed_id))
            fail("nested branch-load guard changed");
      }

      for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
         const uint32_t raw = inputs[0][gid];
         unsigned set, index, delta;
         if ((raw & 1u) != 0u && (raw & 2u) != 0u) {
            set = 1;
            index = (gid * 5u + 3u) & 16383u;
            delta = 17;
         } else if ((raw & 1u) != 0u) {
            set = 2;
            index = (gid * 7u + 11u) & 16383u;
            delta = 19;
         } else if ((raw & 4u) != 0u) {
            set = 3;
            index = (gid * 9u + 13u) & 16383u;
            delta = 23;
         } else {
            set = 4;
            index = (gid * 13u + 29u) & 16383u;
            delta = 31;
         }
         const uint32_t x = inputs[set][index];
         const uint32_t y = inputs[set][(index + delta) & 16383u];
         uint32_t side_value, value;
         if (set == 1) {
            side_value = (x + y) ^ 0x11112222u;
            value = x * 3u + (y ^ raw);
         } else if (set == 2) {
            side_value = (x ^ y) + 0x33334444u;
            value = (x + raw) ^ (y * 5u);
         } else if (set == 3) {
            side_value = x * 7u + y;
            value = (x ^ raw) + (y ^ 0x55556666u);
         } else {
            side_value = (x - y) ^ 0x77778888u;
            value = (x * 11u) ^ (y + raw);
         }
         const uint32_t want =
            slot == 0 ? value ^ (gid * 0x10203u) : side_value;
         if (output[gid] != want) {
            fprintf(stderr,
                    "nested branch-load slot %u word %u=%#x expected=%#x "
                    "set=%u\n",
                    slot, gid, output[gid], want, set);
            fail("nested branch-load output mismatch");
         }
      }
   }

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(5, input_buffers);
   glDeleteBuffers(1, &output_buffer);
   for (unsigned i = 0; i < 5; ++i)
      free(inputs[i]);
   free(seed);
}

static void
fill_branch_load_inputs(unsigned pattern, uint32_t *condition,
                        uint32_t *then_data, uint32_t *else_data,
                        uint32_t *post_data)
{
   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      if (pattern == 0) {
         static const uint32_t mixed[] = {
            0x7fffffffu,
            0x80000000u,
            0x80000001u,
            0x00004567u,
         };
         condition[i] = mixed[i & 3] ^ ((i >> 2) & 0x3ffu);
      } else {
         condition[i] = pattern == 1 ? 0x40000000u | i : 0xc0000000u | i;
      }

      then_data[i] = 0x13579bdfu ^ (i * 0x01020305u) ^ (i >> 3);
      else_data[i] = (0x2468ace0u + (i * 0x9e3779b9u)) ^ (i << 7);
      post_data[i] = 0xa5a55a5au ^ (i * 0x85ebca6bu) ^ (i >> 5);
   }
}

static void
run_branch_local_device_loads(void)
{
   const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer OutThen { uint v[]; } out_then;\n"
      "layout(std430,binding=1) buffer OutElse { uint v[]; } out_else;\n"
      "layout(std430,binding=2) buffer OutMerge { uint v[]; } out_merge;\n"
      "layout(std430,binding=3) readonly buffer Condition { uint v[]; } cond;\n"
      "layout(std430,binding=4) readonly buffer ThenData { uint v[]; } td;\n"
      "layout(std430,binding=5) readonly buffer ElseData { uint v[]; } ed;\n"
      "layout(std430,binding=6) readonly buffer PostData { uint v[]; } pd;\n"
      "void main(){\n"
      " uint gid=gl_GlobalInvocationID.x;\n"
      " uint raw=cond.v[gid]; uint merged;\n"
      " if(raw < 0x80000000u) {\n"
      "  uint ti=(gid*5u+3u)&16383u; uint tv=td.v[ti];\n"
      "  out_then.v[gid]=(tv^0x31415926u)+gid*3u;\n"
      "  merged=(tv*9u)+(gid^0x10203u);\n"
      " } else {\n"
      "  uint ei=(gid*7u+11u)&16383u; uint ev=ed.v[ei];\n"
      "  out_else.v[gid]=(ev+0x27182818u)^(gid*5u);\n"
      "  merged=(ev^0xa5a55a5au)-(gid*7u);\n"
      " }\n"
      " uint pi=(gid*13u+17u)&16383u; uint pv=pd.v[pi];\n"
      " out_merge.v[gid]=(merged^pv)+0xdeadbeefu;\n"
      "}\n";
   const size_t segment_bytes = VALUE_COUNT * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(3, segment_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   uint32_t *condition = malloc(segment_bytes);
   uint32_t *then_data = malloc(segment_bytes);
   uint32_t *else_data = malloc(segment_bytes);
   uint32_t *post_data = malloc(segment_bytes);
   if (!seed || !condition || !then_data || !else_data || !post_data)
      fail("allocate branch-local-load buffers");

   GLuint output_buffer = 0;
   GLuint input_buffers[4] = {0};
   glGenBuffers(1, &output_buffer);
   glGenBuffers(4, input_buffers);
   GLuint program = build_compute_source(source);
   glUseProgram(program);

   for (unsigned pattern = 0; pattern < 3; ++pattern) {
      for (unsigned slot = 0; slot < 3; ++slot)
         seed_output_slot(seed, &layout, slot, WORKLOAD_GID, pattern);
      fill_branch_load_inputs(pattern, condition, then_data, else_data,
                              post_data);

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
      glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                   GL_DYNAMIC_COPY);
      for (unsigned binding = 0; binding < 3; ++binding) {
         glBindBufferRange(GL_SHADER_STORAGE_BUFFER, binding, output_buffer,
                           slot_output_offset(&layout, binding), segment_bytes);
      }

      const uint32_t *inputs[] = {condition, then_data, else_data, post_data};
      for (unsigned i = 0; i < 4; ++i) {
         glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffers[i]);
         glBufferData(GL_SHADER_STORAGE_BUFFER, segment_bytes, inputs[i],
                      GL_DYNAMIC_COPY);
         glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3 + i, input_buffers[i]);
      }

      glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
      glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                      GL_SHADER_STORAGE_BARRIER_BIT);
      glFinish();

      glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
      const uint8_t *mapped = glMapBufferRange(
         GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
      if (!mapped)
         fail("map branch-local-load result");

      unsigned mismatches[3] = {0};
      unsigned changed[3] = {0};
      for (unsigned slot = 0; slot < 3; ++slot) {
         const size_t offset = slot_output_offset(&layout, slot);
         const uint32_t *before =
            (const uint32_t *)(mapped + offset - layout.guard_bytes);
         const uint32_t *output = (const uint32_t *)(mapped + offset);
         const uint32_t *after =
            (const uint32_t *)(mapped + offset + segment_bytes);
         for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
            if (before[i] != guard_word(slot, 0, i, pattern) ||
                after[i] != guard_word(slot, 1, i, pattern))
               fail("branch-local-load guard changed");
         }

         for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
            const bool true_arm = condition[i] < 0x80000000u;
            const uint32_t tv = then_data[(i * 5u + 3u) & 16383u];
            const uint32_t ev = else_data[(i * 7u + 11u) & 16383u];
            const uint32_t pv = post_data[(i * 13u + 17u) & 16383u];
            const bool selected = slot == 2 || slot == (true_arm ? 0u : 1u);
            uint32_t want = poison_word(WORKLOAD_GID, i, slot, pattern);
            if (selected) {
               if (slot == 0)
                  want = (tv ^ 0x31415926u) + i * 3u;
               else if (slot == 1)
                  want = (ev + 0x27182818u) ^ (i * 5u);
               else {
                  const uint32_t merged = true_arm
                                             ? tv * 9u + (i ^ 0x10203u)
                                             : (ev ^ 0xa5a55a5au) - i * 7u;
                  want = (merged ^ pv) + 0xdeadbeefu;
               }
            }

            changed[slot] +=
               output[i] != poison_word(WORKLOAD_GID, i, slot, pattern);
            if (output[i] != want) {
               if (mismatches[slot] < 4)
                  fprintf(stderr,
                          "branch-local-load pattern %u slot %u word %u=%#x "
                          "expected=%#x condition=%#x\n",
                          pattern, slot, i, output[i], want, condition[i]);
               ++mismatches[slot];
            }
         }
      }

      if (mismatches[0] || mismatches[1] || mismatches[2]) {
         fprintf(stderr,
                 "branch-local-load pattern %u summary: slot0 changed=%u "
                 "mismatches=%u, slot1 changed=%u mismatches=%u, "
                 "slot2 changed=%u mismatches=%u\n",
                 pattern, changed[0], mismatches[0], changed[1], mismatches[1],
                 changed[2], mismatches[2]);
         fail("branch-local-load output mismatch");
      }
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   }

   glDeleteProgram(program);
   glDeleteBuffers(4, input_buffers);
   glDeleteBuffers(1, &output_buffer);
   free(post_data);
   free(else_data);
   free(then_data);
   free(condition);
   free(seed);
}

static void
run_basic_loop(bool bottom_tested)
{
   static const char *top_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){ uint gid=gl_GlobalInvocationID.x; uint n=gid&7u;"
      " uint x=gid^0x2468ace0u; uint i=0u;"
      " for(;i<n;++i) x=x*3u+i+1u;"
      " output0.v[gid]=uvec4(x,i,n,0xa1000000u|gid); }\n";
   static const char *bottom_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){ uint gid=gl_GlobalInvocationID.x; uint n=(gid&7u)+1u;"
      " uint x=gid^0x2468ace0u; uint i=0u;"
      " do { x=x*3u+i+1u; ++i; } while(i<n);"
      " output0.v[gid]=uvec4(x,i,n,0xa2000000u|gid); }\n";

   const size_t words = 4 * VALUE_COUNT;
   const size_t bytes = words * sizeof(uint32_t);
   uint32_t *seed = malloc(bytes);
   if (!seed)
      fail("allocate basic-loop output");
   for (size_t i = 0; i < words; ++i)
      seed[i] = 0xdead1000u ^ (uint32_t)i;

   GLuint program =
      build_compute_source(bottom_tested ? bottom_source : top_source);
   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, seed, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   const uint32_t *output =
      glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes, GL_MAP_READ_BIT);
   if (!output)
      fail("map basic-loop output");
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t n = bottom_tested ? (gid & 7u) + 1u : gid & 7u;
      uint32_t x = gid ^ 0x2468ace0u;
      uint32_t i = 0;
      do {
         if (!bottom_tested && i == n)
            break;
         x = x * 3u + i + 1u;
         ++i;
      } while (i < n);
      const uint32_t want[] = {
         x,
         i,
         n,
         (bottom_tested ? 0xa2000000u : 0xa1000000u) | gid,
      };
      for (unsigned c = 0; c < 4; ++c) {
         if (output[gid * 4 + c] != want[c]) {
            fprintf(stderr,
                    "%s-loop word %u component %u=%#x expected=%#x n=%u\n",
                    bottom_tested ? "bottom" : "top", gid, c,
                    output[gid * 4 + c], want[c], n);
            fail("basic-loop output mismatch");
         }
      }
   }

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(seed);
}

/* Exercise the generic structured-loop lowering with a terminating edge that
 * is deliberately neither the first nor last control-flow node in the body.
 * The old Apple9 backend recognized only source-shaped header/latch tests and
 * could not compile this form. */
static void
run_mid_body_break_loop(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){ uint gid=gl_GlobalInvocationID.x; uint limit=gid&15u;"
      " uint value=gid^0x6a09e667u; uint i=0u;"
      " for(;;){"
      "  value+=i*0x10203u+0x11111111u;"
      "  if(((gid+i)&1u)!=0u) value^=0x9e3779b9u+i;"
      "  else value=value*3u+0x7f4a7c15u;"
      "  if(i>=limit) break;"
      "  if(((gid^i)&2u)!=0u) value+=gid*5u+i+0x13579bdfu;"
      "  else value^=gid*7u+i+0x2468ace0u;"
      "  ++i;"
      " }"
      " output0.v[gid]=uvec4(value,i,limit,0xa3000000u|gid); }\n";

   const size_t words = 4 * VALUE_COUNT;
   const size_t bytes = words * sizeof(uint32_t);
   uint32_t *seed = malloc(bytes);
   if (!seed)
      fail("allocate mid-body-loop output");
   for (size_t i = 0; i < words; ++i)
      seed[i] = 0xdead1800u ^ (uint32_t)i;

   GLuint program = build_compute_source(source);
   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, seed, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   const uint32_t *output =
      glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes, GL_MAP_READ_BIT);
   if (!output)
      fail("map mid-body-loop output");
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t limit = gid & 15u;
      uint32_t value = gid ^ 0x6a09e667u;
      uint32_t i = 0;
      for (;;) {
         value += i * 0x10203u + 0x11111111u;
         if (((gid + i) & 1u) != 0)
            value ^= 0x9e3779b9u + i;
         else
            value = value * 3u + 0x7f4a7c15u;
         if (i >= limit)
            break;
         if (((gid ^ i) & 2u) != 0)
            value += gid * 5u + i + 0x13579bdfu;
         else
            value ^= gid * 7u + i + 0x2468ace0u;
         ++i;
      }

      const uint32_t want[] = {value, i, limit, 0xa3000000u | gid};
      for (unsigned c = 0; c < 4; ++c) {
         if (output[gid * 4 + c] != want[c]) {
            fprintf(stderr,
                    "mid-body-loop word %u component %u=%#x expected=%#x "
                    "limit=%u\n",
                    gid, c, output[gid * 4 + c], want[c], limit);
            fail("mid-body-loop output mismatch");
         }
      }
   }

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(seed);
}

static void
run_structured_loops(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){\n"
      " uint gid=gl_GlobalInvocationID.x;"
      " uint limit=gid&7u; uint value=gid^0x13579bdfu; uint executed=0u;"
      " uint break_at=((gid>>8u)&3u)+2u;"
      " for(uint i=0u;i<limit;++i){"
      "  if(((i^gid)&3u)==0u) continue;"
      "  value=value*3u+(i^0x10203040u); ++executed;"
      "  if(executed==break_at) break;"
      " }"
      " uint nested=0u; uint outer_limit=(gid>>3u)&3u;"
      " uint inner_limit=(gid>>5u)&3u; uint inner_break=(gid>>7u)&3u;"
      " for(uint o=0u;o<outer_limit;++o){"
      "  for(uint j=0u;j<inner_limit;++j){"
      "   if(((o+j+gid)&1u)!=0u) continue;"
      "   nested=nested*5u+o*7u+j+1u;"
      "   if(j==inner_break) break;"
      "  }"
      "  nested^=o+0x55u;"
      " }"
      " output0.v[gid]=uvec4(value,executed,nested,0xc0000000u|"
      "  (inner_limit<<20u)|(outer_limit<<18u)|gid);"
      "}\n";

   const size_t output_words = 4 * VALUE_COUNT;
   const size_t output_bytes = output_words * sizeof(uint32_t);
   struct output_layout layout = make_output_layout(1, output_bytes);
   uint8_t *seed = malloc(layout.buffer_bytes);
   if (!seed)
      fail("allocate structured-loop buffer");

   const unsigned seed_id = 0x4c4f4f50;
   const size_t offset = slot_output_offset(&layout, 0);
   uint32_t *before = (uint32_t *)(seed + offset - layout.guard_bytes);
   uint32_t *output = (uint32_t *)(seed + offset);
   uint32_t *after = (uint32_t *)(seed + offset + output_bytes);
   for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
      before[i] = guard_word(0, 0, i, seed_id);
      after[i] = guard_word(0, 1, i, seed_id);
   }
   for (size_t i = 0; i < output_words; ++i)
      output[i] = 0xdead0000u ^ (uint32_t)i;

   GLuint program = build_compute_source(source);
   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, layout.buffer_bytes, seed,
                GL_DYNAMIC_COPY);
   glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffer, offset, output_bytes);
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   const uint8_t *mapped = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, layout.buffer_bytes, GL_MAP_READ_BIT);
   if (!mapped)
      fail("map structured-loop output");
   const uint32_t *mapped_before =
      (const uint32_t *)(mapped + offset - layout.guard_bytes);
   const uint32_t *mapped_output = (const uint32_t *)(mapped + offset);
   const uint32_t *mapped_after =
      (const uint32_t *)(mapped + offset + output_bytes);
   for (size_t i = 0; i < layout.guard_bytes / sizeof(uint32_t); ++i) {
      if (mapped_before[i] != guard_word(0, 0, i, seed_id) ||
          mapped_after[i] != guard_word(0, 1, i, seed_id))
         fail("structured-loop guard changed");
   }

   unsigned mismatches[4] = {0};
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t limit = gid & 7u;
      uint32_t value = gid ^ 0x13579bdfu;
      uint32_t executed = 0;
      const uint32_t break_at = ((gid >> 8) & 3u) + 2u;
      for (uint32_t i = 0; i < limit; ++i) {
         if (((i ^ gid) & 3u) == 0)
            continue;
         value = value * 3u + (i ^ 0x10203040u);
         ++executed;
         if (executed == break_at)
            break;
      }

      uint32_t nested = 0;
      const uint32_t outer_limit = (gid >> 3) & 3u;
      const uint32_t inner_limit = (gid >> 5) & 3u;
      const uint32_t inner_break = (gid >> 7) & 3u;
      for (uint32_t o = 0; o < outer_limit; ++o) {
         for (uint32_t j = 0; j < inner_limit; ++j) {
            if (((o + j + gid) & 1u) != 0)
               continue;
            nested = nested * 5u + o * 7u + j + 1u;
            if (j == inner_break)
               break;
         }
         nested ^= o + 0x55u;
      }

      const uint32_t want[] = {
         value,
         executed,
         nested,
         0xc0000000u | (inner_limit << 20) | (outer_limit << 18) | gid,
      };
      for (unsigned component = 0; component < 4; ++component) {
         const uint32_t got = mapped_output[gid * 4 + component];
         if (got != want[component]) {
            if (mismatches[component] < 8)
               fprintf(stderr,
                       "structured-loop word %u component %u=%#x "
                       "expected=%#x limit=%u executed=%u nested=%#x\n",
                       gid, component, got, want[component], limit, executed,
                       nested);
            ++mismatches[component];
         }
      }
   }

   if (mismatches[0] || mismatches[1] || mismatches[2] || mismatches[3]) {
      fprintf(stderr,
              "structured-loop mismatches: value=%u executed=%u nested=%u "
              "tag=%u\n",
              mismatches[0], mismatches[1], mismatches[2], mismatches[3]);
      fail("structured-loop output mismatch");
   }

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(seed);
}

static void
run_loop_unwind_continue(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){ uint gid=gl_GlobalInvocationID.x;"
      " uint limit=(gid&15u)+1u; uint stop=(gid>>4u)&7u;"
      " uint value=gid^0x31415926u; uint executed=0u; uint i=0u;"
      " for(;i<limit;++i){"
      "  value=value*3u+i+1u;"
      "  if(((i+gid)&1u)!=0u){"
      "   value^=0x10203040u+i;"
      "   if(((i^gid)&2u)!=0u){"
      "    value+=0x22334455u+gid;"
      "    if(i==stop) break;"
      "    value^=0x55667788u+i;"
      "   }else value+=0x66778899u+gid;"
      "  }else value^=0x89abcdefu+gid;"
      "  if(((i+gid)&4u)!=0u){"
      "   value+=0x13579bdfu;"
      "   if(((i^gid)&8u)!=0u) continue;"
      "   value^=0x2468ace0u;"
      "  }else value+=0x0badc0deu;"
      "  value=value*5u+0x55u; ++executed;"
      " }"
      " output0.v[gid]=uvec4(value,i,executed,0xd0000000u|gid); }\n";

   const size_t words = 4 * VALUE_COUNT;
   const size_t bytes = words * sizeof(uint32_t);
   uint32_t *seed = malloc(bytes);
   if (!seed)
      fail("allocate loop-unwind output");
   for (size_t i = 0; i < words; ++i)
      seed[i] = 0xdead2000u ^ (uint32_t)i;

   GLuint program = build_compute_source(source);
   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, seed, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   const uint32_t *output =
      glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes, GL_MAP_READ_BIT);
   if (!output)
      fail("map loop-unwind output");
   unsigned mismatches = 0;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t limit = (gid & 15u) + 1u;
      const uint32_t stop = (gid >> 4) & 7u;
      uint32_t value = gid ^ 0x31415926u;
      uint32_t executed = 0, i = 0;
      for (; i < limit; ++i) {
         value = value * 3u + i + 1u;
         if (((i + gid) & 1u) != 0) {
            value ^= 0x10203040u + i;
            if (((i ^ gid) & 2u) != 0) {
               value += 0x22334455u + gid;
               if (i == stop)
                  break;
               value ^= 0x55667788u + i;
            } else {
               value += 0x66778899u + gid;
            }
         } else {
            value ^= 0x89abcdefu + gid;
         }
         if (((i + gid) & 4u) != 0) {
            value += 0x13579bdfu;
            if (((i ^ gid) & 8u) != 0)
               continue;
            value ^= 0x2468ace0u;
         } else {
            value += 0x0badc0deu;
         }
         value = value * 5u + 0x55u;
         ++executed;
      }

      const uint32_t want[] = {value, i, executed, 0xd0000000u | gid};
      for (unsigned c = 0; c < 4; ++c) {
         const uint32_t got = output[gid * 4 + c];
         if (got != want[c]) {
            if (mismatches < 12)
               fprintf(stderr,
                       "loop-unwind word %u component %u=%#x expected=%#x "
                       "limit=%u stop=%u\n",
                       gid, c, got, want[c], limit, stop);
            ++mismatches;
         }
      }
   }
   if (mismatches)
      fail("loop-unwind output mismatch");

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(seed);
}

static void
run_triple_nested_loops(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){ uint gid=gl_GlobalInvocationID.x;"
      " uint la=gid&3u,lb=(gid>>2u)&3u,lc=(gid>>4u)&3u;"
      " uvec4 s=uvec4(gid^0x01020304u,gid+0x11223344u,"
      "                gid^0x55667788u,gid+0x99aabbccu);"
      " for(uint a=0u;a<la;++a){"
      "  s.x=s.x*3u+a+1u;"
      "  for(uint b=0u;b<lb;++b){"
      "   s.y=(s.y^s.x)+b+0x31u;"
      "   for(uint c=0u;c<lc;++c){"
      "    s.z=s.z*5u+a*17u+b*7u+c+1u;"
      "    if(((a+b+c+gid)&1u)!=0u) continue;"
      "    s.w=(s.w^s.z)+a*13u+b*3u+c;"
      "   }"
      "   s.x^=s.z+b+0x53u;"
      "  }"
      "  s.y+=s.w+a+0x79u;"
      " }"
      " output0.v[gid]=s; }\n";

   const size_t words = 4 * VALUE_COUNT;
   const size_t bytes = words * sizeof(uint32_t);
   uint32_t *seed = malloc(bytes);
   if (!seed)
      fail("allocate triple-loop output");
   for (size_t i = 0; i < words; ++i)
      seed[i] = 0xdead3000u ^ (uint32_t)i;

   GLuint program = build_compute_source(source);
   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, seed, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   const uint32_t *output =
      glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes, GL_MAP_READ_BIT);
   if (!output)
      fail("map triple-loop output");
   unsigned mismatches = 0;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t la = gid & 3u;
      const uint32_t lb = (gid >> 2) & 3u;
      const uint32_t lc = (gid >> 4) & 3u;
      uint32_t s[] = {
         gid ^ 0x01020304u,
         gid + 0x11223344u,
         gid ^ 0x55667788u,
         gid + 0x99aabbccu,
      };
      for (uint32_t a = 0; a < la; ++a) {
         s[0] = s[0] * 3u + a + 1u;
         for (uint32_t b = 0; b < lb; ++b) {
            s[1] = (s[1] ^ s[0]) + b + 0x31u;
            for (uint32_t c = 0; c < lc; ++c) {
               s[2] = s[2] * 5u + a * 17u + b * 7u + c + 1u;
               if (((a + b + c + gid) & 1u) != 0)
                  continue;
               s[3] = (s[3] ^ s[2]) + a * 13u + b * 3u + c;
            }
            s[0] ^= s[2] + b + 0x53u;
         }
         s[1] += s[3] + a + 0x79u;
      }
      for (unsigned c = 0; c < 4; ++c) {
         const uint32_t got = output[gid * 4 + c];
         if (got != s[c]) {
            if (mismatches < 12)
               fprintf(stderr,
                       "triple-loop word %u component %u=%#x expected=%#x "
                       "limits=%u/%u/%u\n",
                       gid, c, got, s[c], la, lb, lc);
            ++mismatches;
         }
      }
   }
   if (mismatches)
      fail("triple-loop output mismatch");

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(seed);
}

static void
run_loop_device_loads(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) readonly buffer Counts { uint v[]; } counts;\n"
      "layout(std430,binding=1) readonly buffer Data { uint v[]; } data0;\n"
      "layout(std430,binding=2) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){ uint gid=gl_GlobalInvocationID.x;"
      " uint limit=counts.v[gid]&31u; uint acc=gid^0xa5a55a5au;"
      " uint consumed=0u;"
      " for(uint i=0u;i<limit;++i){"
      "  uint word=data0.v[(gid*17u+i*13u)&16383u];"
      "  acc=(acc^word)*3u+i+1u;"
      "  if(((word^gid)&3u)==0u) continue;"
      "  acc=acc*5u+(word>>3u); ++consumed;"
      " }"
      " output0.v[gid]=uvec4(acc,limit,consumed,0xe0000000u|gid); }\n";

   const size_t scalar_bytes = VALUE_COUNT * sizeof(uint32_t);
   const size_t output_bytes = 4 * scalar_bytes;
   uint32_t *counts = malloc(scalar_bytes);
   uint32_t *data = malloc(scalar_bytes);
   uint32_t *seed = malloc(output_bytes);
   if (!counts || !data || !seed)
      fail("allocate loop-load buffers");
   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      counts[i] = (i * 11u + (i >> 3) + 7u) & 31u;
      data[i] = (i * 0x9e3779b9u) ^ (i >> 5) ^ 0x6a09e667u;
   }
   for (size_t i = 0; i < 4 * VALUE_COUNT; ++i)
      seed[i] = 0xdead4000u ^ (uint32_t)i;

   GLuint program = build_compute_source(source);
   GLuint buffers[3] = {0};
   glGenBuffers(3, buffers);
   const void *initial[] = {counts, data, seed};
   const size_t sizes[] = {scalar_bytes, scalar_bytes, output_bytes};
   for (unsigned binding = 0; binding < 3; ++binding) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[binding]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizes[binding], initial[binding],
                   GL_DYNAMIC_COPY);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffers[binding]);
   }
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
   const uint32_t *output = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                             output_bytes, GL_MAP_READ_BIT);
   if (!output)
      fail("map loop-load output");
   unsigned mismatches = 0;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t limit = counts[gid] & 31u;
      uint32_t acc = gid ^ 0xa5a55a5au;
      uint32_t consumed = 0;
      for (uint32_t i = 0; i < limit; ++i) {
         const uint32_t word = data[(gid * 17u + i * 13u) & 16383u];
         acc = (acc ^ word) * 3u + i + 1u;
         if (((word ^ gid) & 3u) == 0)
            continue;
         acc = acc * 5u + (word >> 3);
         ++consumed;
      }
      const uint32_t want[] = {acc, limit, consumed, 0xe0000000u | gid};
      for (unsigned c = 0; c < 4; ++c) {
         const uint32_t got = output[gid * 4 + c];
         if (got != want[c]) {
            if (mismatches < 12)
               fprintf(stderr,
                       "loop-load word %u component %u=%#x expected=%#x "
                       "limit=%u\n",
                       gid, c, got, want[c], limit);
            ++mismatches;
         }
      }
   }
   if (mismatches)
      fail("loop-load output mismatch");

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(3, buffers);
   free(seed);
   free(data);
   free(counts);
}

static void
run_computed_addresses(void)
{
   const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) readonly buffer Control { ivec4 v[]; } ctl;\n"
      "layout(std430,binding=1) readonly buffer Data { uvec4 v[]; } data0;\n"
      "layout(std140,binding=2) uniform Palette { uvec4 v[16]; } palette;\n"
      "layout(std430,binding=3) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;"
      "uint idx=uint(-ctl.v[gid].x);"
      "uint alt=(idx*idx+gid)&15u;"
      "uint pick=(gid&1u)==0u?idx:alt;"
      "uvec4 a=data0.v[pick],b=palette.v[idx];"
      "output0.v[gid]=a^b;"
      "output0.v[gid].xz=uvec2(a.y+b.z,a.w+b.x);}";
   int32_t *control = calloc(VALUE_COUNT * 4, sizeof(*control));
   uint32_t *initial = calloc(VALUE_COUNT * 4, sizeof(*initial));
   if (!control || !initial)
      fail("allocate computed-address inputs");
   uint32_t data[16][4], palette[16][4];
   for (unsigned i = 0; i < VALUE_COUNT; ++i)
      control[4 * i] = -(int32_t)((i * 7 + 3) & 15);
   for (unsigned i = 0; i < 16; ++i) {
      for (unsigned c = 0; c < 4; ++c) {
         data[i][c] = 0xdeadbeefu ^ (i * 0x10203u + c * 0x1234567u);
         palette[i][c] = 0x80000000u + i * 0x789abu + c * 0x314159u;
      }
   }
   GLuint buffers[4], program = build_compute_source(source);
   glGenBuffers(4, buffers);
   const void *contents[] = {control, data, palette, initial};
   const size_t sizes[] = {VALUE_COUNT * 16, sizeof(data), sizeof(palette), VALUE_COUNT * 16};
   for (unsigned i = 0; i < 4; ++i) {
      GLenum target = i == 2 ? GL_UNIFORM_BUFFER : GL_SHADER_STORAGE_BUFFER;
      glBindBuffer(target, buffers[i]);
      glBufferData(target, sizes[i], contents[i], GL_DYNAMIC_COPY);
      glBindBufferBase(target, i, buffers[i]);
   }
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[3]);
   const uint32_t *out = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizes[3], GL_MAP_READ_BIT);
   if (!out)
      fail("map computed-address result");
   unsigned mismatches = 0;
   for (unsigned gid = 0; gid < VALUE_COUNT; ++gid) {
      unsigned idx = (gid * 7 + 3) & 15;
      unsigned pick = gid & 1 ? (idx * idx + gid) & 15 : idx;
      uint32_t expected[] = {data[pick][1] + palette[idx][2],
                            data[pick][1] ^ palette[idx][1],
                            data[pick][3] + palette[idx][0],
                            data[pick][3] ^ palette[idx][3]};
      for (unsigned c = 0; c < 4; ++c) {
         if (out[gid * 4 + c] != expected[c]) {
            if (mismatches++ < 8)
               fprintf(stderr, "computed-address gid=%u channel=%u got=%#x expected=%#x\n",
                       gid, c, out[gid * 4 + c], expected[c]);
         }
      }
   }
   if (mismatches)
      fail("computed-address output mismatch");
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteBuffers(4, buffers);
   glDeleteProgram(program);
   free(control);
   free(initial);
}

static void
run_loop_conditions_and_general_break(void)
{
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uvec4 v[]; } output0;\n"
      "void main(){ uint gid=gl_GlobalInvocationID.x;"
      " uint limit=(gid&15u)+2u,stop=(gid>>4u)&15u;"
      " uint x=gid^0x6a09e667u,a=0u;"
      " while(a*3u+(gid&3u)<limit*3u&&a!=stop){"
      "  x=x*3u+(a*3u+(gid&3u))+0x10203u;++a;}"
      " uint b=0u,marker=0u;"
      " for(;b<limit;++b){x=x*5u+b+0x40506u;"
      "  if(((b^gid)&7u)==3u){x^=0x89abcdefu+b;"
      "   marker=b+1u;break;}"
      "  else{x+=0x13579bdfu+gid;}"
      " }"
      " output0.v[gid]=uvec4(x,a,b,0xf1000000u|marker);}\n";

   const size_t words = 4 * VALUE_COUNT;
   const size_t bytes = words * sizeof(uint32_t);
   uint32_t *seed = malloc(bytes);
   if (!seed)
      fail("allocate loop-condition output");
   for (size_t i = 0; i < words; ++i)
      seed[i] = 0xdead5000u ^ (uint32_t)i;

   GLuint program = build_compute_source(source);
   GLuint buffer = 0;
   glGenBuffers(1, &buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, seed, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
   glUseProgram(program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   const uint32_t *output =
      glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes, GL_MAP_READ_BIT);
   if (!output)
      fail("map loop-condition output");
   unsigned mismatches = 0;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t limit = (gid & 15u) + 2u;
      const uint32_t stop = (gid >> 4) & 15u;
      uint32_t x = gid ^ 0x6a09e667u;
      uint32_t a = 0;
      while (a * 3u + (gid & 3u) < limit * 3u && a != stop) {
         x = x * 3u + (a * 3u + (gid & 3u)) + 0x10203u;
         ++a;
      }

      uint32_t b = 0, marker = 0;
      for (; b < limit; ++b) {
         x = x * 5u + b + 0x40506u;
         if (((b ^ gid) & 7u) == 3u) {
            x ^= 0x89abcdefu + b;
            marker = b + 1u;
            break;
         } else {
            x += 0x13579bdfu + gid;
         }
      }

      const uint32_t want[] = {x, a, b, 0xf1000000u | marker};
      for (unsigned c = 0; c < 4; ++c) {
         const uint32_t got = output[gid * 4 + c];
         if (got != want[c]) {
            if (mismatches < 12)
               fprintf(stderr,
                       "loop-condition word %u component %u=%#x expected=%#x "
                       "limit=%u stop=%u\n",
                       gid, c, got, want[c], limit, stop);
            ++mismatches;
         }
      }
   }
   if (mismatches)
      fail("loop-condition output mismatch");

   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(program);
   glDeleteBuffers(1, &buffer);
   free(seed);
}

static int
compare_u32(const void *a_, const void *b_)
{
   const uint32_t a = *(const uint32_t *)a_;
   const uint32_t b = *(const uint32_t *)b_;
   return (a > b) - (a < b);
}

static uint32_t
atomic_shape_input(unsigned slot, unsigned gid)
{
   return (0x1020304u * (slot + 1)) ^
          (gid * (0x10101u + slot * 0x111u));
}

static void
run_device_atomic_native_shape(void)
{
   enum { count = 64, inputs = 6, target_index = 6, output_index = 7 };
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) readonly buffer I0{uint v[];}i0;\n"
      "layout(std430,binding=1) readonly buffer I1{uint v[];}i1;\n"
      "layout(std430,binding=2) readonly buffer I2{uint v[];}i2;\n"
      "layout(std430,binding=3) readonly buffer I3{uint v[];}i3;\n"
      "layout(std430,binding=4) readonly buffer I4{uint v[];}i4;\n"
      "layout(std430,binding=5) readonly buffer I5{uint v[];}i5;\n"
      "layout(std430,binding=6) buffer T{uint v[];}target;\n"
      "layout(std430,binding=7) buffer O{uvec4 v[];}out0;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;"
      "uint operand=((i0.v[gid]+i1.v[gid])^i2.v[gid])+"
      "((i3.v[gid]&i4.v[gid])|i5.v[gid]);"
      "uint old=atomicAdd(target.v[gid],operand);"
      "out0.v[gid]=uvec4(old,old+operand,operand,gid^0xa917c0deu);}\n";

   uint32_t *initial[8] = {0};
   size_t sizes[8] = {0};
   for (unsigned slot = 0; slot < inputs; ++slot) {
      sizes[slot] = count * sizeof(uint32_t);
      initial[slot] = malloc(sizes[slot]);
      if (!initial[slot])
         fail("allocate atomic-shape input");
      for (unsigned gid = 0; gid < count; ++gid)
         initial[slot][gid] = atomic_shape_input(slot, gid);
   }
   sizes[target_index] = count * sizeof(uint32_t);
   sizes[output_index] = count * 4 * sizeof(uint32_t);
   initial[target_index] = malloc(sizes[target_index]);
   initial[output_index] = calloc(1, sizes[output_index]);
   if (!initial[target_index] || !initial[output_index])
      fail("allocate atomic-shape outputs");
   for (unsigned gid = 0; gid < count; ++gid)
      initial[target_index][gid] = 0x60000000u + 17u * gid;

   GLuint program = build_compute_source(source);
   GLuint buffers[8] = {0};
   glGenBuffers(8, buffers);
   for (unsigned slot = 0; slot < 8; ++slot) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[slot]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizes[slot], initial[slot],
                   GL_DYNAMIC_COPY);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, buffers[slot]);
   }
   glUseProgram(program);
   glDispatchCompute(2, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   unsigned mismatches = 0;
   for (unsigned slot = target_index; slot <= output_index; ++slot) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[slot]);
      const uint32_t *got = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                             sizes[slot], GL_MAP_READ_BIT);
      if (!got)
         fail("map atomic-shape result");
      for (unsigned gid = 0; gid < count; ++gid) {
         uint32_t in[inputs];
         for (unsigned i = 0; i < inputs; ++i)
            in[i] = initial[i][gid];
         uint32_t operand = ((in[0] + in[1]) ^ in[2]) +
                            ((in[3] & in[4]) | in[5]);
         uint32_t before = initial[target_index][gid];
         uint32_t expected[4] = {
            before, before + operand, operand, gid ^ 0xa917c0deu,
         };
         unsigned components = slot == target_index ? 1 : 4;
         for (unsigned c = 0; c < components; ++c) {
            uint32_t want = slot == target_index ? expected[1] : expected[c];
            uint32_t value = got[gid * components + c];
            if (value != want) {
               if (mismatches < 8)
                  fprintf(stderr,
                          "atomic-shape slot %u gid %u component %u=%#x "
                          "expected=%#x\n", slot, gid, c, value, want);
               ++mismatches;
            }
         }
      }
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   }

   glDeleteProgram(program);
   glDeleteBuffers(8, buffers);
   for (unsigned slot = 0; slot < 8; ++slot)
      free(initial[slot]);
   if (mismatches)
      fail("atomic native-shape mismatch");
}

static void
run_device_atomic_pending_load_forwarding(void)
{
   enum {
      count = 64,
      output_index = 0,
      target_index = 1,
      direct_index = 2,
      retained_index = 3,
      unrelated_index = 4,
      buffers_count = 5,
   };
   static const char *source =
      "#version 310 es\n"
      "layout(local_size_x=32) in;\n"
      "layout(std430,binding=0) buffer O{uint v[];}out0;\n"
      "layout(std430,binding=1) buffer T{uint v[];}target;\n"
      "layout(std430,binding=2) readonly buffer D{uint v[];}direct_in;\n"
      "layout(std430,binding=3) readonly buffer R{uint v[];}retained_in;\n"
      "layout(std430,binding=4) readonly buffer U{uint v[];}unrelated_in;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;"
      " uint direct=direct_in.v[gid];"
      " uint retained=retained_in.v[gid];"
      " uint unrelated=unrelated_in.v[gid];"
      " uint old0=atomicAdd(target.v[gid*2u],direct);"
      " uint old1=atomicXor(target.v[gid*2u+1u],retained);"
      " out0.v[gid*4u+0u]=old0;"
      " out0.v[gid*4u+1u]=old1;"
      " out0.v[gid*4u+2u]=retained;"
      " out0.v[gid*4u+3u]=unrelated;}\n";

   size_t sizes[buffers_count] = {
      [output_index] = count * 4 * sizeof(uint32_t),
      [target_index] = count * 2 * sizeof(uint32_t),
      [direct_index] = count * sizeof(uint32_t),
      [retained_index] = count * sizeof(uint32_t),
      [unrelated_index] = count * sizeof(uint32_t),
   };
   uint32_t *initial[buffers_count] = {0};
   for (unsigned binding = 0; binding < buffers_count; ++binding) {
      initial[binding] = calloc(1, sizes[binding]);
      if (!initial[binding])
         fail("allocate pending atomic input");
   }
   for (unsigned gid = 0; gid < count; ++gid) {
      initial[target_index][gid * 2 + 0] = 0x50000000u + gid * 0x101u;
      initial[target_index][gid * 2 + 1] = 0xa0000000u ^ (gid * 0x10001u);
      initial[direct_index][gid] = 0x01020304u ^ (gid * 0x1111u);
      initial[retained_index][gid] = 0x10204081u + gid * 17u;
      initial[unrelated_index][gid] = 0xc0010000u ^ (gid * 0x10101u);
   }

   GLuint program = build_compute_source(source);
   GLuint buffers[buffers_count] = {0};
   glGenBuffers(buffers_count, buffers);
   for (unsigned binding = 0; binding < buffers_count; ++binding) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[binding]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, sizes[binding], initial[binding],
                   GL_DYNAMIC_COPY);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffers[binding]);
   }
   glUseProgram(program);
   glDispatchCompute(2, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   unsigned mismatches = 0;
   for (unsigned binding = output_index; binding <= target_index; ++binding) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[binding]);
      const uint32_t *got = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                             sizes[binding], GL_MAP_READ_BIT);
      if (!got)
         fail("map pending atomic result");
      for (unsigned gid = 0; gid < count; ++gid) {
         uint32_t expected_output[4] = {
            initial[target_index][gid * 2 + 0],
            initial[target_index][gid * 2 + 1],
            initial[retained_index][gid],
            initial[unrelated_index][gid],
         };
         uint32_t expected_target[2] = {
            initial[target_index][gid * 2 + 0] + initial[direct_index][gid],
            initial[target_index][gid * 2 + 1] ^ initial[retained_index][gid],
         };
         const uint32_t *expected =
            binding == output_index ? expected_output : expected_target;
         const unsigned components = binding == output_index ? 4 : 2;
         for (unsigned c = 0; c < components; ++c) {
            if (got[gid * components + c] == expected[c])
               continue;
            if (mismatches < 8)
               fprintf(stderr,
                       "pending atomic binding %u gid %u component %u=%#x "
                       "expected=%#x\n",
                       binding, gid, c, got[gid * components + c], expected[c]);
            ++mismatches;
         }
      }
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   }

   glDeleteProgram(program);
   glDeleteBuffers(buffers_count, buffers);
   for (unsigned binding = 0; binding < buffers_count; ++binding)
      free(initial[binding]);
   if (mismatches)
      fail("pending atomic forwarding mismatch");
}

static void
run_device_atomics(void)
{
   /* Start by isolating the RMW side effect from asynchronous return
    * publication.  The ordinary read proves that the same resource/index is
    * addressable; the CPU readback proves that an unused-result atomic was
    * not optimized away or silently discarded by the machine form. */
   static const char *baseline_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uint v[]; } output0;\n"
      "layout(std430,binding=1) buffer Target { uint v[]; } target;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;"
      " uint before=target.v[gid];"
      " atomicAdd(target.v[gid],gid*3u+1u);"
      " output0.v[gid]=before;}\n";
   uint32_t *baseline_output = malloc(VALUE_COUNT * sizeof(uint32_t));
   uint32_t *baseline_target = malloc(VALUE_COUNT * sizeof(uint32_t));
   if (!baseline_output || !baseline_target)
      fail("allocate atomic baseline buffers");
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      baseline_output[gid] = 0xdead5000u ^ gid;
      baseline_target[gid] = 0x60000000u + gid * 17u;
   }

   GLuint baseline_program = build_compute_source(baseline_source);
   GLuint baseline_buffers[2] = {0};
   glGenBuffers(2, baseline_buffers);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, baseline_buffers[0]);
   glBufferData(GL_SHADER_STORAGE_BUFFER,
                VALUE_COUNT * sizeof(uint32_t), baseline_output,
                GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, baseline_buffers[0]);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, baseline_buffers[1]);
   glBufferData(GL_SHADER_STORAGE_BUFFER, VALUE_COUNT * sizeof(uint32_t),
                baseline_target, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, baseline_buffers[1]);
   glUseProgram(baseline_program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, baseline_buffers[0]);
   const uint32_t *baseline_got = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, VALUE_COUNT * sizeof(uint32_t),
      GL_MAP_READ_BIT);
   if (!baseline_got)
      fail("map atomic baseline output");
   unsigned baseline_mismatches = 0;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      if (baseline_got[gid] != baseline_target[gid]) {
         if (baseline_mismatches < 8)
            fprintf(stderr,
                    "atomic baseline word %u before=%#x expected=%#x\n",
                    gid, baseline_got[gid], baseline_target[gid]);
         ++baseline_mismatches;
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, baseline_buffers[1]);
   const uint32_t *baseline_final = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, VALUE_COUNT * sizeof(uint32_t),
      GL_MAP_READ_BIT);
   if (!baseline_final)
      fail("map atomic baseline target");
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t want = baseline_target[gid] + gid * 3u + 1u;
      if (baseline_final[gid] != want) {
         if (baseline_mismatches < 8)
            fprintf(stderr,
                    "atomic baseline final word %u=%#x expected=%#x\n", gid,
                    baseline_final[gid], want);
         ++baseline_mismatches;
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(baseline_program);
   glDeleteBuffers(2, baseline_buffers);
   free(baseline_target);
   free(baseline_output);
   if (baseline_mismatches)
      fail("atomic baseline mismatch");

   /* Keep the first returning probe deliberately small.  Besides making the
    * old-value oracle unambiguous, this isolates the native atomic-return
    * landing from later result reuse and from the full operation matrix. */
   static const char *single_return_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uint v[]; } output0;\n"
      "layout(std430,binding=1) buffer Target { uint v[]; } target;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;"
      " uint old=atomicAdd(target.v[gid],gid*5u+3u);"
      " output0.v[gid]=old;}\n";
   uint32_t *single_output = malloc(VALUE_COUNT * sizeof(uint32_t));
   uint32_t *single_target = malloc(VALUE_COUNT * sizeof(uint32_t));
   if (!single_output || !single_target)
      fail("allocate single-return atomic buffers");
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      single_output[gid] = 0xdead5100u ^ gid;
      single_target[gid] = 0x61000000u + gid * 19u;
   }
   GLuint single_program = build_compute_source(single_return_source);
   GLuint single_buffers[2] = {0};
   glGenBuffers(2, single_buffers);
   const void *single_data[] = {single_output, single_target};
   for (unsigned b = 0; b < 2; ++b) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, single_buffers[b]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, VALUE_COUNT * sizeof(uint32_t),
                   single_data[b], GL_DYNAMIC_COPY);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, b, single_buffers[b]);
   }
   glUseProgram(single_program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   unsigned single_mismatches = 0;
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, single_buffers[0]);
   const uint32_t *single_old = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, VALUE_COUNT * sizeof(uint32_t),
      GL_MAP_READ_BIT);
   if (!single_old)
      fail("map single-return atomic output");
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      if (single_old[gid] != single_target[gid]) {
         if (single_mismatches < 8)
            fprintf(stderr,
                    "single-return atomic old word %u=%#x expected=%#x\n",
                    gid, single_old[gid], single_target[gid]);
         ++single_mismatches;
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, single_buffers[1]);
   const uint32_t *single_final = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, VALUE_COUNT * sizeof(uint32_t),
      GL_MAP_READ_BIT);
   if (!single_final)
      fail("map single-return atomic target");
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t want = single_target[gid] + gid * 5u + 3u;
      if (single_final[gid] != want) {
         if (single_mismatches < 8)
            fprintf(stderr,
                    "single-return atomic final word %u=%#x expected=%#x\n",
                    gid, single_final[gid], want);
         ++single_mismatches;
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(single_program);
   glDeleteBuffers(2, single_buffers);
   free(single_target);
   free(single_output);
   if (single_mismatches)
      fail("single-return atomic mismatch");

   /* One invocation executes every GLES-visible 32-bit operation on a
    * private location. This checks exact returned values, exact final memory,
    * signedness, compare-exchange success/failure, result fanout, discarded
    * results, affine addresses, and two independently bound atomic buffers. */
   static const char *operation_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uint v[]; } output0;\n"
      "layout(std430,binding=1) buffer UTarget { uint v[]; } target_u;\n"
      "layout(std430,binding=2) buffer STarget { int v[]; } target_s;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;uint b=gid*9u;"
      "uint d0=gid*3u+1u,d1=0x0ff00ff0u^gid,d2=0x01020304u^gid;"
      "uint a=atomicAdd(target_u.v[b+0u],d0);"
      "uint n=atomicAnd(target_u.v[b+1u],d1);"
      "uint o=atomicOr(target_u.v[b+2u],d2);"
      "uint x=atomicXor(target_u.v[b+3u],0xa5a55a5au+gid);"
      "uint lo=atomicMin(target_u.v[b+4u],gid*7u+3u);"
      "uint hi=atomicMax(target_u.v[b+5u],0x70000000u-gid);"
      "uint e=atomicExchange(target_u.v[b+6u],0x33000000u+gid);"
      "uint c=atomicCompSwap(target_u.v[b+7u],"
      " (gid&1u)==0u?0x70000000u+gid:0x12340000u+gid,"
      " 0x44000000u+gid);"
      "atomicXor(target_u.v[b+8u],0x00ff00ffu^gid);"
      "int sl=atomicMin(target_s.v[gid*2u],int(gid)-12000);"
      "int sh=atomicMax(target_s.v[gid*2u+1u],12000-int(gid));"
      /* Keep each oracle component in its own plane.  Besides making failures
       * easy to classify, this prevents NIR from coalescing them into a
       * strided vector store, which is outside the current Apple9 store
       * encoding model and unrelated to the atomic behavior under test. */
      "output0.v[gid+0u*16384u]=a;output0.v[gid+1u*16384u]=n;"
      "output0.v[gid+2u*16384u]=o;output0.v[gid+3u*16384u]=x;"
      "output0.v[gid+4u*16384u]=lo;output0.v[gid+5u*16384u]=hi;"
      "output0.v[gid+6u*16384u]=e;output0.v[gid+7u*16384u]=c;"
      "output0.v[gid+8u*16384u]=uint(sl);"
      "output0.v[gid+9u*16384u]=uint(sh);"
      "output0.v[gid+10u*16384u]=(a+0x10203u)^(a+0x40506u);"
      "output0.v[gid+11u*16384u]=0xa7000000u|gid;}\n";

   const size_t output_words = 3 * 4 * VALUE_COUNT;
   const size_t unsigned_words = 9 * VALUE_COUNT;
   const size_t signed_words = 2 * VALUE_COUNT;
   uint32_t *output_seed = malloc(output_words * sizeof(uint32_t));
   uint32_t *unsigned_seed = malloc(unsigned_words * sizeof(uint32_t));
   int32_t *signed_seed = malloc(signed_words * sizeof(int32_t));
   if (!output_seed || !unsigned_seed || !signed_seed)
      fail("allocate atomic operation buffers");
   for (size_t i = 0; i < output_words; ++i)
      output_seed[i] = 0xdead6000u ^ (uint32_t)i;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      uint32_t *u = unsigned_seed + gid * 9;
      u[0] = 0x80000000u + gid;
      u[1] = 0xf0f00000u | gid;
      u[2] = 0x10000000u | gid;
      u[3] = 0x5a5aa5a5u ^ gid;
      u[4] = 0x80000000u + gid;
      u[5] = gid;
      u[6] = 0x66000000u + gid;
      u[7] = 0x70000000u + gid;
      u[8] = 0x0f0ff0f0u + gid;
      signed_seed[gid * 2] = (int32_t)gid - 4000;
      signed_seed[gid * 2 + 1] = 4000 - (int32_t)gid;
   }

   GLuint operation_program = build_compute_source(operation_source);
   GLuint operation_buffers[3] = {0};
   glGenBuffers(3, operation_buffers);
   const size_t operation_sizes[] = {
      output_words * sizeof(uint32_t), unsigned_words * sizeof(uint32_t),
      signed_words * sizeof(int32_t),
   };
   const void *operation_initial[] = {output_seed, unsigned_seed, signed_seed};
   for (unsigned binding = 0; binding < 3; ++binding) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, operation_buffers[binding]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, operation_sizes[binding],
                   operation_initial[binding], GL_DYNAMIC_COPY);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding,
                       operation_buffers[binding]);
   }
   glUseProgram(operation_program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, operation_buffers[0]);
   const uint32_t *output = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, operation_sizes[0], GL_MAP_READ_BIT);
   if (!output)
      fail("map atomic operation output");
   unsigned mismatches = 0;
   unsigned output_mismatches = 0;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t *u = unsigned_seed + gid * 9;
      const uint32_t want[] = {
         u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
         (uint32_t)signed_seed[gid * 2],
         (uint32_t)signed_seed[gid * 2 + 1],
         (u[0] + 0x10203u) ^ (u[0] + 0x40506u),
         0xa7000000u | gid,
      };
      for (unsigned c = 0; c < ARRAY_SIZE(want); ++c) {
         if (output[c * VALUE_COUNT + gid] != want[c]) {
            if (output_mismatches < 12)
               fprintf(stderr,
                       "atomic return word %u component %u=%#x expected=%#x\n",
                       gid, c, output[c * VALUE_COUNT + gid], want[c]);
            ++mismatches;
            ++output_mismatches;
         }
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, operation_buffers[1]);
   const uint32_t *final_u = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, operation_sizes[1], GL_MAP_READ_BIT);
   if (!final_u)
      fail("map atomic unsigned target");
   unsigned final_u_mismatches = 0;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t *initial = unsigned_seed + gid * 9;
      const uint32_t d0 = gid * 3u + 1u;
      const uint32_t d1 = 0x0ff00ff0u ^ gid;
      const uint32_t d2 = 0x01020304u ^ gid;
      const uint32_t want[] = {
         initial[0] + d0,
         initial[1] & d1,
         initial[2] | d2,
         initial[3] ^ (0xa5a55a5au + gid),
         initial[4] < gid * 7u + 3u ? initial[4] : gid * 7u + 3u,
         initial[5] > 0x70000000u - gid ? initial[5]
                                               : 0x70000000u - gid,
         0x33000000u + gid,
         (gid & 1u) == 0 ? 0x44000000u + gid : initial[7],
         initial[8] ^ (0x00ff00ffu ^ gid),
      };
      for (unsigned c = 0; c < ARRAY_SIZE(want); ++c) {
         if (final_u[gid * 9 + c] != want[c]) {
            if (final_u_mismatches < 12)
               fprintf(stderr,
                       "atomic final-u word %u component %u=%#x expected=%#x\n",
                       gid, c, final_u[gid * 9 + c], want[c]);
            ++mismatches;
            ++final_u_mismatches;
         }
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

   glBindBuffer(GL_SHADER_STORAGE_BUFFER, operation_buffers[2]);
   const int32_t *final_s = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, operation_sizes[2], GL_MAP_READ_BIT);
   if (!final_s)
      fail("map atomic signed target");
   unsigned final_s_mismatches = 0;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const int32_t min_data = (int32_t)gid - 12000;
      const int32_t max_data = 12000 - (int32_t)gid;
      const int32_t min_want = signed_seed[gid * 2] < min_data
                                  ? signed_seed[gid * 2]
                                  : min_data;
      const int32_t max_want = signed_seed[gid * 2 + 1] > max_data
                                  ? signed_seed[gid * 2 + 1]
                                  : max_data;
      if (final_s[gid * 2] != min_want ||
          final_s[gid * 2 + 1] != max_want) {
         if (final_s_mismatches < 12)
            fprintf(stderr,
                    "atomic final-s word %u=%d/%d expected=%d/%d\n", gid,
                    final_s[gid * 2], final_s[gid * 2 + 1], min_want,
                    max_want);
         ++mismatches;
         ++final_s_mismatches;
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   if (mismatches)
      fail("per-lane atomic operation mismatch");

   glDeleteProgram(operation_program);
   glDeleteBuffers(3, operation_buffers);
   free(signed_seed);
   free(unsigned_seed);
   free(output_seed);

   /* All 256 lanes contend on one address. Returned values may appear in any
    * lane order, but their sorted set and the final counter are exact. */
   static const char *contended_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uint v[]; } output0;\n"
      "layout(std430,binding=1) buffer Counter { uint v[]; } counter;\n"
      "void main(){uint lid=gl_LocalInvocationID.x;"
      " output0.v[lid]=atomicAdd(counter.v[0],1u);}\n";
   uint32_t contended_output[LOCAL_SIZE];
   for (unsigned i = 0; i < LOCAL_SIZE; ++i)
      contended_output[i] = 0xdead7000u ^ i;
   uint32_t counter = 1000;
   GLuint contended_program = build_compute_source(contended_source);
   GLuint contended_buffers[2] = {0};
   glGenBuffers(2, contended_buffers);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, contended_buffers[0]);
   glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(contended_output),
                contended_output, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, contended_buffers[0]);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, contended_buffers[1]);
   glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(counter), &counter,
                GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, contended_buffers[1]);
   glUseProgram(contended_program);
   glDispatchCompute(1, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, contended_buffers[0]);
   const uint32_t *returns = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, sizeof(contended_output), GL_MAP_READ_BIT);
   if (!returns)
      fail("map contended atomic returns");
   memcpy(contended_output, returns, sizeof(contended_output));
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   qsort(contended_output, LOCAL_SIZE, sizeof(contended_output[0]), compare_u32);
   for (unsigned i = 0; i < LOCAL_SIZE; ++i) {
      if (contended_output[i] != 1000u + i)
         fail("contended atomic return permutation mismatch");
   }
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, contended_buffers[1]);
   const uint32_t *final_counter = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, sizeof(counter), GL_MAP_READ_BIT);
   if (!final_counter || *final_counter != 1000u + LOCAL_SIZE)
      fail("contended atomic final counter mismatch");
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(contended_program);
   glDeleteBuffers(2, contended_buffers);

   /* Dynamic, heavily colliding addresses under nested loop/if masks. The
    * return is intentionally unused, exercising the native discard form; all
    * adds commute, so every one of 1024 final words has an exact CPU oracle. */
   static const char *masked_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uint v[]; } output0;\n"
      "layout(std430,binding=1) buffer Target { uint v[]; } target;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;"
      " uint idx=(gid*40503u+17u)&1023u;uint count=(gid&3u)+1u;"
      " for(uint i=0u;i<count;++i){"
      "  if(((gid+i)&1u)!=0u) atomicAdd(target.v[idx],(gid&7u)+i+1u);"
      "  else atomicAdd(target.v[idx],(gid&3u)+i+3u);"
      " }output0.v[gid]=0xb8000000u|gid;}\n";
   uint32_t *masked_output = malloc(VALUE_COUNT * sizeof(uint32_t));
   uint32_t masked_target[1024];
   uint32_t masked_want[1024];
   if (!masked_output)
      fail("allocate masked atomic output");
   for (uint32_t i = 0; i < VALUE_COUNT; ++i)
      masked_output[i] = 0xdead8000u ^ i;
   for (uint32_t i = 0; i < ARRAY_SIZE(masked_target); ++i)
      masked_target[i] = masked_want[i] = 0x10000000u + i;
   for (uint32_t gid = 0; gid < VALUE_COUNT; ++gid) {
      const uint32_t idx = (gid * 40503u + 17u) & 1023u;
      const uint32_t count = (gid & 3u) + 1u;
      for (uint32_t i = 0; i < count; ++i)
         masked_want[idx] += ((gid + i) & 1u) != 0u ? (gid & 7u) + i + 1u
                                                    : (gid & 3u) + i + 3u;
   }
   GLuint masked_program = build_compute_source(masked_source);
   GLuint masked_buffers[2] = {0};
   glGenBuffers(2, masked_buffers);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, masked_buffers[0]);
   glBufferData(GL_SHADER_STORAGE_BUFFER, VALUE_COUNT * sizeof(uint32_t),
                masked_output, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, masked_buffers[0]);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, masked_buffers[1]);
   glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(masked_target), masked_target,
                GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, masked_buffers[1]);
   glUseProgram(masked_program);
   glDispatchCompute(VALUE_COUNT / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, masked_buffers[0]);
   const uint32_t *masked_result = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, VALUE_COUNT * sizeof(uint32_t),
      GL_MAP_READ_BIT);
   if (!masked_result)
      fail("map masked atomic output");
   for (uint32_t i = 0; i < VALUE_COUNT; ++i) {
      if (masked_result[i] != (0xb8000000u | i))
         fail("masked atomic completion output mismatch");
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, masked_buffers[1]);
   const uint32_t *masked_final = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, sizeof(masked_target), GL_MAP_READ_BIT);
   if (!masked_final)
      fail("map masked atomic target");
   for (unsigned i = 0; i < ARRAY_SIZE(masked_target); ++i) {
      if (masked_final[i] != masked_want[i])
         fail("masked atomic target mismatch");
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(masked_program);
   glDeleteBuffers(2, masked_buffers);
   free(masked_output);

   /* Returned atomics are harder than discarded side effects: every dynamic
    * iteration must consume slot 6 before the next atomic reuses it, and each
    * masked arm must publish its old value before reconvergence. Keep each
    * lane on a private address so the complete return/final-memory oracle is
    * deterministic while still stressing divergent loop-carried phis. */
   enum { returned_count = 4096 };
   static const char *returned_mask_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Output { uint v[]; } output0;\n"
      "layout(std430,binding=1) buffer Target { uint v[]; } target;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;"
      " uint acc=0x811c9dc5u;uint count=(gid&3u)+2u;"
      " for(uint i=0u;i<count;++i){uint old;"
      "  if(((gid+i)&1u)!=0u)"
      "   old=atomicAdd(target.v[gid],((i+1u)*3u)+(gid&7u));"
      "  else old=atomicXor(target.v[gid],"
      "                         0x01010101u+i*0x00110011u);"
      "  acc=(acc*16777619u)^old;"
      " }output0.v[gid*2u]=acc;"
      " output0.v[gid*2u+1u]=0xc9000000u|count;}\n";
   uint32_t *returned_output = malloc(returned_count * 2 * sizeof(uint32_t));
   uint32_t *returned_target = malloc(returned_count * sizeof(uint32_t));
   uint32_t *returned_want = malloc(returned_count * 2 * sizeof(uint32_t));
   uint32_t *returned_target_want =
      malloc(returned_count * sizeof(uint32_t));
   if (!returned_output || !returned_target || !returned_want ||
       !returned_target_want)
      fail("allocate returned masked atomic buffers");
   for (uint32_t gid = 0; gid < returned_count; ++gid) {
      returned_output[gid * 2] = 0xdead9000u ^ gid;
      returned_output[gid * 2 + 1] = 0xdead9100u ^ gid;
      uint32_t value = 0x71000000u + gid * 13u;
      uint32_t acc = 0x811c9dc5u;
      const uint32_t count = (gid & 3u) + 2u;
      for (uint32_t i = 0; i < count; ++i) {
         const uint32_t old = value;
         if (((gid + i) & 1u) != 0u)
            value += ((i + 1u) * 3u) + (gid & 7u);
         else
            value ^= 0x01010101u + i * 0x00110011u;
         acc = acc * 16777619u ^ old;
      }
      returned_want[gid * 2] = acc;
      returned_want[gid * 2 + 1] = 0xc9000000u | count;
      returned_target[gid] = 0x71000000u + gid * 13u;
      returned_target_want[gid] = value;
   }

   GLuint returned_program = build_compute_source(returned_mask_source);
   GLuint returned_buffers[2] = {0};
   glGenBuffers(2, returned_buffers);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, returned_buffers[0]);
   glBufferData(GL_SHADER_STORAGE_BUFFER,
                returned_count * 2 * sizeof(uint32_t), returned_output,
                GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, returned_buffers[0]);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, returned_buffers[1]);
   glBufferData(GL_SHADER_STORAGE_BUFFER, returned_count * sizeof(uint32_t),
                returned_target, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, returned_buffers[1]);
   glUseProgram(returned_program);
   glDispatchCompute(returned_count / LOCAL_SIZE, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();

   unsigned returned_mismatches = 0;
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, returned_buffers[0]);
   const uint32_t *returned_got = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, returned_count * 2 * sizeof(uint32_t),
      GL_MAP_READ_BIT);
   if (!returned_got)
      fail("map returned masked atomic output");
   for (uint32_t i = 0; i < returned_count * 2; ++i) {
      if (returned_got[i] != returned_want[i]) {
         if (returned_mismatches < 12)
            fprintf(stderr,
                    "returned masked atomic output word %u=%#x expected=%#x\n",
                    i, returned_got[i], returned_want[i]);
         ++returned_mismatches;
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, returned_buffers[1]);
   const uint32_t *returned_final = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, returned_count * sizeof(uint32_t),
      GL_MAP_READ_BIT);
   if (!returned_final)
      fail("map returned masked atomic target");
   for (uint32_t i = 0; i < returned_count; ++i) {
      if (returned_final[i] != returned_target_want[i]) {
         if (returned_mismatches < 12)
            fprintf(stderr,
                    "returned masked atomic target word %u=%#x expected=%#x\n",
                    i, returned_final[i], returned_target_want[i]);
         ++returned_mismatches;
      }
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(returned_program);
   glDeleteBuffers(2, returned_buffers);
   free(returned_target_want);
   free(returned_want);
   free(returned_target);
   free(returned_output);
   if (returned_mismatches)
      fail("returned masked atomic mismatch");

   /* Keep one writable binding live across many dispatch records. This catches
    * launch/resource-record reuse and missing inter-dispatch visibility while
    * making the oracle independent of workgroup or lane execution order. */
   enum { repeat_buckets = 32, repeat_dispatches = 32 };
   static const char *repeat_source =
      "#version 310 es\n"
      "layout(local_size_x=256) in;\n"
      "layout(std430,binding=0) buffer Counters { uint v[]; } counters;\n"
      "void main(){uint gid=gl_GlobalInvocationID.x;"
      " atomicAdd(counters.v[gid&31u],1u);}\n";
   uint32_t repeat_initial[repeat_buckets];
   for (unsigned i = 0; i < ARRAY_SIZE(repeat_initial); ++i)
      repeat_initial[i] = 0x1000u + i * 17u;
   GLuint repeat_program = build_compute_source(repeat_source);
   GLuint repeat_buffer = 0;
   glGenBuffers(1, &repeat_buffer);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, repeat_buffer);
   glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(repeat_initial),
                repeat_initial, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, repeat_buffer);
   glUseProgram(repeat_program);
   for (unsigned i = 0; i < repeat_dispatches; ++i)
      glDispatchCompute(4, 1, 1);
   glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                   GL_SHADER_STORAGE_BARRIER_BIT);
   glFinish();
   const uint32_t *repeat_final = glMapBufferRange(
      GL_SHADER_STORAGE_BUFFER, 0, sizeof(repeat_initial), GL_MAP_READ_BIT);
   if (!repeat_final)
      fail("map repeated atomic counters");
   const uint32_t per_bucket = repeat_dispatches * (4u * LOCAL_SIZE) /
                               repeat_buckets;
   for (unsigned i = 0; i < ARRAY_SIZE(repeat_initial); ++i) {
      if (repeat_final[i] != repeat_initial[i] + per_bucket)
         fail("repeated atomic counter mismatch");
   }
   glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   glDeleteProgram(repeat_program);
   glDeleteBuffers(1, &repeat_buffer);
}

static const char *const lifecycle_case_names[] = {
   "archive-cross-program-sequence",
   "repeated-range-dispatch",
   "program-lifecycle-stress",
   "simple-divergent-if-else",
   "two-source-comparisons",
   "compare-register-pressure",
   "single-region-shapes",
   "multiple-phi-vectors",
   "branch-local-device-loads",
   "nested-short-circuit-if-else",
   "nested-branch-local-loads",
   "top-tested-loop",
   "bottom-tested-loop",
   "structured-loops",
   "loop-unwind-continue",
   "triple-nested-loops",
   "loop-device-loads",
   "loop-conditions-general-break",
   "device-atomics",
   "device-atomic-native-shape",
   "device-atomic-pending-load-forwarding",
   "mid-body-break-loop",
   "computed-addresses",
};

static int
run_named_case(const char *name)
{
   if (!strcmp(name, "computed-addresses")) {
      run_computed_addresses();
      return 1;
   }
   for (unsigned i = 0; i < WORKLOAD_ARCHIVE_CROSS_0; ++i) {
      if (!strcmp(name, workload_names[i])) {
         return run_formula_case((enum workload)i) ? 1 : -1;
      }
   }
   if (!strcmp(name, lifecycle_case_names[0])) {
      run_archive_cross_sequence();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[1])) {
      run_repeated_range_dispatch();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[2])) {
      run_program_lifecycle_stress();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[3])) {
      run_simple_divergent_if_else();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[4])) {
      run_two_source_comparisons();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[5])) {
      run_compare_register_pressure();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[6])) {
      run_single_region_shapes();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[7])) {
      run_multiple_phi_vectors();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[8])) {
      run_branch_local_device_loads();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[9])) {
      run_nested_short_circuit_if_else();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[10])) {
      run_nested_branch_local_loads();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[11])) {
      run_basic_loop(false);
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[12])) {
      run_basic_loop(true);
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[13])) {
      run_structured_loops();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[14])) {
      run_loop_unwind_continue();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[15])) {
      run_triple_nested_loops();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[16])) {
      run_loop_device_loads();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[17])) {
      run_loop_conditions_and_general_break();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[18])) {
      run_device_atomics();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[19])) {
      run_device_atomic_native_shape();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[20])) {
      run_device_atomic_pending_load_forwarding();
      return 1;
   }
   if (!strcmp(name, lifecycle_case_names[21])) {
      run_mid_body_break_loop();
      return 1;
   }
   size_t count = 0;
   const char *const *names = t8132_apple9_memory_case_names(&count);
   for (size_t i = 0; i < count; ++i) {
      if (!strcmp(name, names[i])) {
         t8132_apple9_run_memory_case(name);
         return 1;
      }
   }
   names = t8132_apple9_geometry_case_names(&count);
   for (size_t i = 0; i < count; ++i) {
      if (!strcmp(name, names[i])) {
         t8132_apple9_run_geometry_case(name);
         return 1;
      }
   }
   return t8132_apple9_run_math_case(name);
}

static void
list_cases(bool include_archive_stress)
{
   for (unsigned i = 0; i < WORKLOAD_ARCHIVE_CROSS_0; ++i)
      puts(workload_names[i]);
   size_t count = 0;
   const char *const *names = t8132_apple9_memory_case_names(&count);
   for (size_t i = 0; i < count; ++i)
      puts(names[i]);
   names = t8132_apple9_geometry_case_names(&count);
   for (size_t i = 0; i < count; ++i)
      puts(names[i]);
   /* Run the math matrix separately: its generated range-reduction shaders
    * consume a substantial part of the temporary 64 KiB program archive. */
   if (include_archive_stress) {
      names = t8132_apple9_math_case_names(&count);
      for (size_t i = 0; i < count; ++i)
         puts(names[i]);
   }
   for (unsigned i = 0;
        i < sizeof(lifecycle_case_names) / sizeof(lifecycle_case_names[0]);
        ++i) {
      if (!include_archive_stress && (i == 0 || i == 2 || i == 4))
         continue;
      puts(lifecycle_case_names[i]);
   }
}

int
main(int argc, char **argv)
{
   if (argc == 2 &&
       (!strcmp(argv[1], "--list") || !strcmp(argv[1], "--list-default"))) {
      /* The append-only bring-up archive cannot hold every independent shader
       * program in one process. Keep the archive-lifecycle stresses and the
       * 24-program comparison matrix available explicitly and in the complete
       * listing, but omit them from the one-boot semantic-output sequence. */
      list_cases(!strcmp(argv[1], "--list"));
      return 0;
   }
   if (argc < 2) {
      fprintf(stderr, "usage: %s --list | CASE...\n", argv[0]);
      return 2;
   }

   EGLDisplay display = open_asahi_display();
   EGLint major = 0, minor = 0;
   if (!eglInitialize(display, &major, &minor) ||
       !eglBindAPI(EGL_OPENGL_ES_API))
      fail("initialize EGL");

   const EGLint config_attrs[] = {
      EGL_SURFACE_TYPE,       EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
      EGL_OPENGL_ES3_BIT_KHR, EGL_NONE,
   };
   EGLConfig config;
   EGLint config_count = 0;
   if (!eglChooseConfig(display, config_attrs, &config, 1, &config_count) ||
       config_count != 1)
      fail("eglChooseConfig");

   const EGLint surface_attrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
   EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attrs);
   const EGLint context_attrs[] = {
      EGL_CONTEXT_MAJOR_VERSION_KHR,
      3,
      EGL_CONTEXT_MINOR_VERSION_KHR,
      1,
      EGL_NONE,
   };
   EGLContext context =
      eglCreateContext(display, config, EGL_NO_CONTEXT, context_attrs);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      fail("create GLES 3.1 context");

   const char *renderer = (const char *)glGetString(GL_RENDERER);
   const char *version = (const char *)glGetString(GL_VERSION);
   if (!renderer || !strstr(renderer, "Apple M4"))
      fail("unexpected renderer");

   bool failed = false;
   for (int i = 1; i < argc; ++i) {
      printf("PIGLIT TEST: %d - %s\n", i, argv[i]);
      fflush(stdout);
      int result = run_named_case(argv[i]);
      if (result == 0) {
         fprintf(stderr, "unknown Apple9 compute case: %s\n", argv[i]);
         return 2;
      }
      const char *status = result > 0 ? "pass" : "fail";
      failed |= result < 0;
      if (argc == 2)
         printf("PIGLIT: {\"result\": \"%s\"}\n", status);
      else
         printf("PIGLIT: {\"subtest\": {\"%s\": \"%s\"}}\n", argv[i], status);
      fflush(stdout);
   }

   printf("T8132_APPLE9_COMPUTE_RUNNER_%s cases=%d renderer=\"%s\" "
          "version=\"%s\"\n",
          failed ? "FAILED" : "OK", argc - 1, renderer, version);
   eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(display, context);
   eglDestroySurface(display, surface);
   eglTerminate(display);
   return failed ? 1 : 0;
}
