#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
int main(int argc,char **argv) {
 setvbuf(stdout,NULL,_IONBF,0);
 unsigned count=argc>1?atoi(argv[1]):400, size=argc>2?strtoul(argv[2],NULL,0):0x564000;
 EGLDisplay d=eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,NULL);
 assert(d!=EGL_NO_DISPLAY);assert(eglInitialize(d,NULL,NULL)); assert(eglBindAPI(EGL_OPENGL_ES_API));
 EGLint attrs[]={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
 EGLContext ctx=eglCreateContext(d,NULL,EGL_NO_CONTEXT,attrs);assert(ctx!=EGL_NO_CONTEXT);
 assert(eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,ctx));
 printf("pid=%d pointer_bits=%zu pagesize=%ld renderer=%s\n",getpid(),sizeof(void*)*8,sysconf(_SC_PAGESIZE),glGetString(GL_RENDERER));
 unsigned reserve_mib=argc>3?atoi(argv[3]):0;
 void *reserved=MAP_FAILED;
 if(reserve_mib){reserved=mmap(NULL,(size_t)reserve_mib*1024*1024,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);assert(reserved!=MAP_FAILED);printf("VA reservation=%u MiB at %p\n",reserve_mib,reserved);}
 GLuint *bo=calloc(count,sizeof(*bo)); assert(bo); glGenBuffers(count,bo);
 unsigned made=0;
 for(unsigned i=0;i<count;i++) {
  glBindBuffer(GL_ARRAY_BUFFER,bo[i]); glBufferData(GL_ARRAY_BUFFER,size,NULL,GL_STATIC_DRAW);
  GLenum err=glGetError(); if(err){printf("ERROR glBufferData i=%u err=0x%x\n",i,err);break;}
  void *p=glMapBufferRange(GL_ARRAY_BUFFER,0,size,GL_MAP_WRITE_BIT|GL_MAP_INVALIDATE_BUFFER_BIT);
  if(!p){printf("FAIL map i=%u total=%llu err=0x%x\n",i,(unsigned long long)i*size,glGetError());break;}
  *(volatile unsigned*)p=i; assert(glUnmapBuffer(GL_ARRAY_BUFFER)); made++;
  if(i%20==0)printf("mapped_then_unmapped=%u bytes=%llu last=%p\n",i+1,(unsigned long long)(i+1)*size,p);
 }
 for(unsigned i=0;i<made;i+=17) {
  glBindBuffer(GL_ARRAY_BUFFER,bo[i]);
  unsigned *p=glMapBufferRange(GL_ARRAY_BUFFER,0,4,GL_MAP_READ_BIT);
  assert(p && *p==i); assert(glUnmapBuffer(GL_ARRAY_BUFFER));
 }
 printf("completed=%u requested=%u; sampled readback verified\n",made,count);
 glDeleteBuffers(count,bo);free(bo);if(reserved!=MAP_FAILED)munmap(reserved,(size_t)reserve_mib*1024*1024);eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);eglDestroyContext(d,ctx);eglTerminate(d);
 return made==count?0:2;
}
