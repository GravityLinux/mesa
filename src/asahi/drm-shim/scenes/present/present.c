/* SPDX-License-Identifier: MIT */
/* EGL surface identity test: the last GPU render deliberately targets an FBO. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int ok, const char *what)
{
   if (!ok) {
      fprintf(stderr, "PRESENT_FAIL %s EGL=%#x GL=%#x\n", what, eglGetError(), glGetError());
      exit(1);
   }
}
static void step(const char *name)
{
   check(glGetError() == GL_NO_ERROR, name);
   printf("PRESENT_READY %s\n", name);
   fflush(stdout);
   check(getchar() == '\n', "controller acknowledgement");
}
static void clear(float r, float g, float b)
{
   glClearColor(r, g, b, 1);
   glClear(GL_COLOR_BUFFER_BIT);
   glFinish();
}
static EGLSurface surface(EGLDisplay d, EGLConfig c, int w, int h)
{
   EGLint attr[] = {EGL_WIDTH,w,EGL_HEIGHT,h,EGL_NONE};
   EGLSurface s = eglCreatePbufferSurface(d,c,attr);
   check(s != EGL_NO_SURFACE, "pbuffer creation");
   return s;
}
static EGLDisplay display(void)
{
   if (!getenv("PRESENT_TEST_DEVICE"))
      return eglGetDisplay(EGL_DEFAULT_DISPLAY);
   PFNEGLQUERYDEVICESEXTPROC query =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
   PFNEGLQUERYDEVICESTRINGEXTPROC string =
      (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
   PFNEGLGETPLATFORMDISPLAYEXTPROC platform =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   check(query && string && platform, "device platform entrypoints");
   EGLDeviceEXT devices[16]; EGLint count=0;
   check(query(16,devices,&count), "enumerate devices");
   for (int i=0;i<count;++i) {
      const char *node=string(devices[i],EGL_DRM_RENDER_NODE_FILE_EXT);
      if (node && strstr(node,"renderD"))
         return platform(EGL_PLATFORM_DEVICE_EXT,devices[i],NULL);
   }
   check(0,"render device not found");
   return EGL_NO_DISPLAY;
}
int main(void)
{
   EGLDisplay d = display();
   check(eglInitialize(d,NULL,NULL), "initialize");
   check(eglBindAPI(EGL_OPENGL_ES_API), "bind API");
   EGLint attrs[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   EGLConfig config; EGLint count;
   check(eglChooseConfig(d,attrs,&config,1,&count) && count, "choose config");
   EGLint ctxattrs[] = {EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE};
   EGLContext ctx = eglCreateContext(d,config,EGL_NO_CONTEXT,ctxattrs);
   check(ctx != EGL_NO_CONTEXT, "context");
   EGLSurface a = surface(d,config,320,256);
   check(eglMakeCurrent(d,a,a,ctx), "current A");
   step("initial");
   glViewport(0,0,320,256);
   clear(1,0,0);
   GLuint texture, fbo;
   glGenTextures(1,&texture); glBindTexture(GL_TEXTURE_2D,texture);
   glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,320,256,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
   glGenFramebuffers(1,&fbo); glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,texture,0);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"FBO");
   clear(0,1,0);
   step("hidden-red-green");
   check(eglSwapBuffers(d,a), "swap A with FBO bound");
   step("red");
   glBindFramebuffer(GL_FRAMEBUFFER,0);
   clear(0,0,1);
   glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   clear(0,1,0);
   step("still-red");
   check(eglSwapBuffers(d,a), "swap blue");
   step("blue");
   check(eglSwapBuffers(d,a), "repeat swap");
   step("blue-repeat");
   EGLSurface b = surface(d,config,128,192);
   check(eglMakeCurrent(d,b,b,ctx), "current B");
   glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,128,192);
   clear(1,1,0);
   glBindFramebuffer(GL_FRAMEBUFFER,fbo); clear(1,0,1);
   check(eglSwapBuffers(d,b), "swap B");
   step("yellow-small");
   check(eglMakeCurrent(d,a,a,ctx), "restore A");
   check(eglDestroySurface(d,b), "destroy B");
   check(eglSwapBuffers(d,a), "restore A swap without drawing");
   step("blue-restored");
   glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,320,256);
   glClearColor(0,1,1,1); glClear(GL_COLOR_BUFFER_BIT);
   /* No glFinish: the swap implementation must flush this rendering itself. */
   check(eglSwapBuffers(d,a), "flush pending cyan");
   step("cyan-pending");
   check(!eglSwapBuffers(d,b) && eglGetError()==EGL_BAD_SURFACE, "reject destroyed surface");
   step("cyan-invalid-swap");
   glDeleteFramebuffers(1,&fbo); glDeleteTextures(1,&texture);
   check(eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT), "unbind");
   eglDestroySurface(d,a); eglDestroyContext(d,ctx); eglTerminate(d);
   puts("PRESENT_PASS");
   return 0;
}
