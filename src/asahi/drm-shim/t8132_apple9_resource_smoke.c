/* SPDX-License-Identifier: GPL-2.0-only OR MIT */
/* Verify direct SSBO/UBO resource tables and complete buffer guards.
 * Arguments: total buffers (1..18), UBO count, sparse SSBO bindings (0/1). */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void fail(const char *what)
{
    fprintf(stderr, "%s: EGL %#x GL %#x\n", what, eglGetError(), glGetError());
    exit(1);
}
static GLuint shader(GLenum type, const char *text)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &text, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "shader: %s\n", log);
        fail("shader compilation");
    }
    return s;
}
int main(int argc, char **argv)
{
    setbuf(stdout, NULL);

    PFNEGLQUERYDEVICESEXTPROC devices = (void *)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLQUERYDEVICESTRINGEXTPROC device_string = (void *)eglGetProcAddress("eglQueryDeviceStringEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC platform = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!devices || !device_string || !platform) fail("device extensions");
    EGLDeviceEXT found[16];
    EGLint count = 0;
    if (!devices(16, found, &count)) fail("enumerate devices");
    EGLDisplay d = EGL_NO_DISPLAY;
    for (int i = 0; i < count; i++) {
        const char *path = device_string(found[i], EGL_DRM_RENDER_NODE_FILE_EXT);
        printf("EGL device %d: %s\n", i, path ? path : "no render node");
        if (path && !strcmp(path, "/dev/dri/renderD128"))
            d = platform(EGL_PLATFORM_DEVICE_EXT, found[i], NULL);
    }
    EGLint major, minor;
    if (d == EGL_NO_DISPLAY || !eglInitialize(d, &major, &minor)) fail("initialize G16 EGL device");
    const EGLint attrs[] = { EGL_SURFACE_TYPE, 0, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE };
    EGLConfig cfg;
    if (!eglChooseConfig(d, attrs, &cfg, 1, &count) || !count) fail("choose config");
    if (!eglBindAPI(EGL_OPENGL_ES_API)) fail("bind API");
    const EGLint context_attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, context_attrs);
    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) fail("create context");
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    printf("EGL %d.%d, GL %s, renderer %s\n", major, minor, glGetString(GL_VERSION), renderer);
    if (!renderer || strstr(renderer, "llvmpipe") || strstr(renderer, "softpipe")) fail("software renderer selected");


    enum { WORDS=512, ACTIVE=64, FRAMES=3, MAX_BUFFERS=18 };
    unsigned n=argc>1?atoi(argv[1]):18;
    unsigned mixed=argc>2?atoi(argv[2]):0;
    unsigned sparse=argc>3?atoi(argv[3]):0;
    if(n<1 || n>MAX_BUFFERS || mixed>=n)fail("arguments");
    unsigned values[MAX_BUFFERS][WORDS];
    GLuint kept[FRAMES*MAX_BUFFERS], programs[FRAMES];
    GLint ssbo_max=0,ubo_max=0;
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,&ssbo_max);
    glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS,&ubo_max);
    printf("Resource limits SSBO=%d UBO=%d requested=%u mixed=%u\n",ssbo_max,ubo_max,n,mixed);
    for(unsigned frame=0;frame<FRAMES;frame++) {
        unsigned offset=64+frame*64;
        char source[16384];size_t at=snprintf(source,sizeof(source),"#version 310 es\nlayout(local_size_x=32) in;\n");
        for(unsigned b=0;b<n;b++) {
            unsigned binding=b<mixed?b:(sparse?(b==n-1?31:30-b):b);
            if(b<mixed)at+=snprintf(source+at,sizeof(source)-at,"layout(std140,binding=%u) uniform B%u { uvec4 a%u[64]; };\n",b,b,b);
            else at+=snprintf(source+at,sizeof(source)-at,"layout(std430,binding=%u) %s buffer B%u { uint a%u[]; };\n",binding,b==n-1?"":"readonly",b+100,b);
        }
        uint32_t seed=0x13579bdu+frame*0x10009u;
        at+=snprintf(source+at,sizeof(source)-at,"void main(){uint i=gl_GlobalInvocationID.x; uint v=%uu;\n",seed);
        for(unsigned b=0;b<n;b++)at+=snprintf(source+at,sizeof(source)-at,"v=(v^a%u[i]%s)*17u+%uu;\n",b,b<mixed?".x":"",b);
        at+=snprintf(source+at,sizeof(source)-at,"a%u[i]=v;}\n",n-1);
        GLuint cs=shader(GL_COMPUTE_SHADER,source),program=glCreateProgram();programs[frame]=program;
        glAttachShader(program,cs);glLinkProgram(program);glDeleteShader(cs);
        GLint ok;glGetProgramiv(program,GL_LINK_STATUS,&ok);
        if(!ok){char log[4096];glGetProgramInfoLog(program,sizeof(log),NULL,log);fprintf(stderr,"%s\n",log);fail("link");}
        glUseProgram(program);
        GLuint *buffers=kept+frame*n;glGenBuffers(n,buffers);
        for(unsigned b=0;b<n;b++) {
            for(unsigned j=0;j<WORDS;j++)values[b][j]=0x87654321u^(frame*0x1020304u+b*0x113355u+j*0x1001u);
            GLenum target=b<mixed?GL_UNIFORM_BUFFER:GL_SHADER_STORAGE_BUFFER;
            glBindBuffer(target,buffers[b]);glBufferData(target,sizeof(values[b]),values[b],GL_DYNAMIC_COPY);
            glBindBufferRange(target,b<mixed?b:(sparse?(b==n-1?31:30-b):b),buffers[b],offset*4,ACTIVE*4*(b<mixed?4:1));
        }
        glDispatchCompute(2,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);glFinish();
        if(glGetError()!=GL_NO_ERROR)fail("dispatch");
        for(unsigned b=0;b<n;b++) {
            GLenum target=b<mixed?GL_UNIFORM_BUFFER:GL_SHADER_STORAGE_BUFFER;
            glBindBuffer(target,buffers[b]);const uint32_t *data=glMapBufferRange(target,0,sizeof(values[b]),GL_MAP_READ_BIT);
            if(!data)fail("map");
            for(unsigned j=0;j<WORDS;j++) {
                uint32_t want=values[b][j];
                if(b==n-1 && j>=offset && j<offset+ACTIVE) {
                    want=seed;unsigned index=j-offset;
                    for(unsigned k=0;k<n;k++)want=(want^values[k][offset+index*(k<mixed?4:1)])*17u+k;
                }
                if(data[j]!=want){fprintf(stderr,"FAIL n%u mixed%u frame%u buffer%u word%u got%#x expected%#x\n",n,mixed,frame,b,j,data[j],want);return 1;}
            }
            if(!glUnmapBuffer(target))fail("unmap");
        }
        printf("PASS: %u buffers (%u UBO), frame %u, all %u words verified\n",n,mixed,frame,n*WORDS);
    }
    glDeleteBuffers(FRAMES*n,kept);for(unsigned f=0;f<FRAMES;f++)glDeleteProgram(programs[f]);
    return 0;
}
