/* SPDX-License-Identifier: MIT
 * Analytic fragment Z/W, varying and depth-buffer checks using Piglit's harness.
 */
#include "piglit-util-gl.h"

PIGLIT_GL_TEST_CONFIG_BEGIN
   config.supports_gl_compat_version = 21;
   config.window_width = 160;
   config.window_height = 140;
   config.window_visual = PIGLIT_GL_VISUAL_RGBA | PIGLIT_GL_VISUAL_DEPTH;
PIGLIT_GL_TEST_CONFIG_END

static const char *vs =
   "varying float v; void main() { gl_Position=gl_Vertex; v=gl_Color.r; }";
static const char *fs =
   "varying float v; void main() {"
   "gl_FragColor=vec4(gl_FragCoord.z,v,gl_FragCoord.w*0.25,1.0); }";

void piglit_init(int argc, char **argv)
{
   GLuint p = piglit_build_simple_program(vs, fs);
   glUseProgram(p);
}

enum piglit_result piglit_display(void)
{
   const float w[3] = {1, 2, 4}, color[3] = {.2, .6, .9};
   const float xy[3][2] = {{-1,-1},{3,-1},{-1,3}};
   const struct { const char *name; float near, far, z0; int x,y,w,h; } cases[] = {
      {"default", 0,1,-.8, 0,0,160,140},
      {"depth-range", .2,.8,-.8, 0,0,160,140},
      {"reversed-depth-range", .9,.1,-.8, 0,0,160,140},
      {"offset-viewport", .15,.75,-.8, 13,17,101,83},
      {"near-clipped", .2,.9,-2, 13,17,101,83},
   };
   bool all = true;
   unsigned char rgba[160*140*4];
   float depth[160*140];
   for (unsigned t=0; t<ARRAY_SIZE(cases); ++t) {
      const float z[3] = {cases[t].z0, .6, .2};
      glDisable(GL_SCISSOR_TEST);
      glClearColor(0,0,0,0); glClearDepth(1);
      glDepthMask(GL_TRUE); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
      glViewport(cases[t].x,cases[t].y,cases[t].w,cases[t].h);
      glDepthRange(cases[t].near,cases[t].far);
      glEnable(GL_DEPTH_TEST); glDepthFunc(GL_ALWAYS);
      glBegin(GL_TRIANGLES);
      for (unsigned i=0;i<3;++i) {
         glColor4f(color[i],0,0,1);
         glVertex4f(xy[i][0]*w[i],xy[i][1]*w[i],z[i]*w[i],w[i]);
      }
      glEnd();
      glReadPixels(0,0,160,140,GL_RGBA,GL_UNSIGNED_BYTE,rgba);
      glReadPixels(0,0,160,140,GL_DEPTH_COMPONENT,GL_FLOAT,depth);
      unsigned errors=0, checked=0;
      for (int y=0;y<140;++y) for (int x=0;x<160;++x) {
         float b=(x+.5-cases[t].x)/(2*cases[t].w);
         float c=(y+.5-cases[t].y)/(2*cases[t].h),a=1-b-c;
         float ndc=a*z[0]+b*z[1]+c*z[2];
         bool inside=x>=cases[t].x && x<cases[t].x+cases[t].w &&
                     y>=cases[t].y && y<cases[t].y+cases[t].h && ndc>=-1 && ndc<=1;
         /* Ignore only a subpixel-width band at the clipped boundary. */
         if (fabsf(ndc+1)<.015f) continue;
         float reciprocal=a/w[0]+b/w[1]+c/w[2];
         float expected_z=cases[t].near+(ndc*.5+.5)*(cases[t].far-cases[t].near);
         float expected[4]={0,0,0,0};
         if (inside) {
            expected[0]=expected_z;
            expected[1]=(a*color[0]/w[0]+b*color[1]/w[1]+c*color[2]/w[2])/reciprocal;
            expected[2]=reciprocal*.25; expected[3]=1;
         }
         unsigned p=y*160+x;
         bool ok=fabsf(depth[p]-(inside?expected_z:1))<2e-5;
         for (unsigned k=0;k<4;++k) ok &= fabsf(rgba[p*4+k]/255.0f-expected[k])<.009;
         ++checked;
         if (!ok && errors++<3)
            printf("%s (%d,%d) rgba=%u,%u,%u,%u depth=%g expected=%g,%g,%g,%g depth=%g\n",
                   cases[t].name,x,y,rgba[p*4],rgba[p*4+1],rgba[p*4+2],rgba[p*4+3],depth[p],
                   expected[0],expected[1],expected[2],expected[3],inside?expected_z:1);
      }
      printf("%s: %u checked, %u errors\n",cases[t].name,checked,errors);
      piglit_report_subtest_result(errors?PIGLIT_FAIL:PIGLIT_PASS,"%s",cases[t].name);
      all &= !errors;
   }
   return all && piglit_check_gl_error(GL_NO_ERROR) ? PIGLIT_PASS : PIGLIT_FAIL;
}
