/* SPDX-License-Identifier: MIT */
static int
run_vertex_inputs(unsigned width, unsigned height)
{
   const char *vs =
      "#version 300 es\nprecision highp float;\n"
      "layout(location=3) in vec4 position; layout(location=7) in vec2 uv;\n"
      "layout(location=9) in vec4 color; layout(location=12) in vec4 constant_color;\n"
      "uniform vec4 tint; out vec2 texcoord; out vec4 vertex_color;\n"
      "void main(){gl_Position=position;texcoord=uv;vertex_color=color*constant_color*tint;}\n";
   const char *fs =
      "#version 300 es\nprecision highp float;\n"
      "uniform sampler2D image; in vec2 texcoord; in vec4 vertex_color;out vec4 color;\n"
      "void main(){color=texture(image,texcoord)*vertex_color;}\n";
   GLuint program = glCreateProgram();
   GLuint v = compile_shader(GL_VERTEX_SHADER, vs), f = compile_shader(GL_FRAGMENT_SHADER, fs);
   glAttachShader(program,v); glAttachShader(program,f); glLinkProgram(program);
   GLint linked; glGetProgramiv(program,GL_LINK_STATUS,&linked);
   if (!linked) fail("vertex input link");
   glUseProgram(program);
   GLuint vao, vb[2], ib, target, image, fb;
   glGenVertexArrays(1,&vao); glBindVertexArray(vao);
   glGenBuffers(2,vb); glGenBuffers(1,&ib);
   glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb);
   glGenTextures(1,&target); glBindTexture(GL_TEXTURE_2D,target);
   glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,width,height);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,target,0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE) fail("vertex input FBO");
   uint8_t pixels[64*64*4];
   for (unsigned y=0;y<64;++y) for(unsigned x=0;x<64;++x) {
      unsigned at=(y*64+x)*4;
      pixels[at]=17+3*x;pixels[at+1]=23+3*y;pixels[at+2]=31+x+y;pixels[at+3]=255;
   }
   glGenTextures(1,&image);glActiveTexture(GL_TEXTURE3);glBindTexture(GL_TEXTURE_2D,image);
   glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,64,64,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
   glUniform1i(glGetUniformLocation(program,"image"),3);
   /* An unindexed six-vertex quad starting at vertex five, padded both before
    * the first element and within each element. Its ushort UVs and byte colors
    * share a single VBO, while the disabled color attribute is draw state. */
   uint8_t data[12+11*24]; memset(data,0xcd,sizeof(data));
   const unsigned corners[6][2]={{0,0},{1,0},{0,1},{0,1},{1,0},{1,1}};
   for(unsigned i=0;i<6;++i) {
      uint8_t *p=data+12+(i+5)*24;
      float xy[2]={(float)corners[i][0]*2-1,(float)corners[i][1]*2-1};
      uint16_t uv[2]={corners[i][0]*65535,corners[i][1]*65535};
      memcpy(p,xy,8);memcpy(p+8,uv,4);
      /* The earliest attribute base is deliberately not dword aligned. */
      p[-7]=32+192*corners[i][0];p[-6]=32+192*corners[i][1];p[-5]=64;p[-4]=255;
   }
   bool streams = getenv("T8132_VERTEX_INDEPENDENT_STREAMS") != NULL;
   unsigned stream_draws = getenv("T8132_VERTEX_STREAM_DRAWS") ?
      strtoul(getenv("T8132_VERTEX_STREAM_DRAWS"), NULL, 10) : 56;
   if (stream_draws < 1 || stream_draws > 57) fail("vertex stream draw count");
   /* Independently suballocated position and UV/color streams in one BO.
    * Their spacing changes per draw; their shader fetch layout does not. */
   uint8_t stream_data[32768]; memset(stream_data, 0, sizeof(stream_data));
   for (unsigned draw=0; draw<stream_draws; ++draw) {
      for (unsigned i=0; i<6; ++i) {
         const uint8_t *p=data+12+(i+5)*24;
         memcpy(stream_data+draw*128+(i+5)*8,p,8);
         memcpy(stream_data+8192+draw*256+(i+5)*12,p+8,4);
         memcpy(stream_data+8192+draw*256+(i+5)*12+4,p-7,4);
      }
   }
   for(unsigned b=0;b<2;++b) {
      glBindBuffer(GL_ARRAY_BUFFER,vb[b]);
      glBufferData(GL_ARRAY_BUFFER,streams?sizeof(stream_data):sizeof(data),
                   streams?stream_data:data,GL_STREAM_DRAW);
   }
   uint16_t indices[8]={0xffff,0xffff,5,6,7,8,9,10};
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ib);glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(indices),indices,GL_STATIC_DRAW);
   glEnableVertexAttribArray(3);glEnableVertexAttribArray(7);glEnableVertexAttribArray(9);
   glDisableVertexAttribArray(12);
   glViewport(0,0,width,height);glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);glDisable(GL_CULL_FACE);
   for(unsigned frame=0;frame<2;++frame) {
      glBindBuffer(GL_ARRAY_BUFFER,vb[frame]);
      glVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE,24,(void *)12);
      glVertexAttribPointer(7,2,GL_UNSIGNED_SHORT,GL_TRUE,24,(void *)20);
      glVertexAttribPointer(9,4,GL_UNSIGNED_BYTE,GL_TRUE,24,(void *)5);
      glVertexAttrib4f(12,frame?.5f:1,frame?.75f:1,1,1);
      glUniform4f(glGetUniformLocation(program,"tint"),1,.875f,.75f,1);
      for (unsigned draw=0; draw<(streams?stream_draws:1u); ++draw) {
         if (streams) {
            glVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE,8,(void *)(uintptr_t)(draw*128));
            glVertexAttribPointer(7,2,GL_UNSIGNED_SHORT,GL_TRUE,12,
                                  (void *)(uintptr_t)(8192+draw*256));
            glVertexAttribPointer(9,4,GL_UNSIGNED_BYTE,GL_TRUE,12,
                                  (void *)(uintptr_t)(8192+draw*256+4));
         }
         if (frame) glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,(void *)4);
         else glDrawArrays(GL_TRIANGLES,5,6);
      }
      glFinish();if(glGetError()!=GL_NO_ERROR) fail("vertex input draw");
      printf("T8132_VERTEX_INPUT_FRAME frame=%u\n",frame);
   }
   glDeleteProgram(program);glDeleteShader(v);glDeleteShader(f);
   glDeleteBuffers(2,vb);glDeleteBuffers(1,&ib);glDeleteTextures(1,&target);glDeleteTextures(1,&image);
   glDeleteFramebuffers(1,&fb);glDeleteVertexArrays(1,&vao);
   return 0;
}
