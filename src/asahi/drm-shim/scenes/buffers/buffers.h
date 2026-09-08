/* SPDX-License-Identifier: MIT */
static int
run_buffers(unsigned width, unsigned height)
{
   unsigned count =
      getenv("T8132_BUFFER_COUNT") ? atoi(getenv("T8132_BUFFER_COUNT")) : 15;
   bool vbos = getenv("T8132_BUFFER_VBOS") != NULL;
   bool vertex = getenv("T8132_BUFFER_VERTEX") != NULL;
   if (!count || count > 15)
      fail("buffer count");
   char shaders[2][32768];
   for (unsigned stage = 0; stage < 2; ++stage) {
      char *s = shaders[stage];
      size_t n = snprintf(
         s, 32768,
         "#version 300 es\nprecision highp float; precision highp int;\nuniform vec4 shift;\nuniform int selected;\n%s vec4 tint;\n%s\n",
         stage ? "in" : "out", stage ? "out vec4 color;" : "");
      if (!stage && vbos)
         for (unsigned i = 0; i < 16; ++i)
            n += snprintf(s + n, 32768 - n,
                          "layout(location=%u) in vec4 a%u;\n", i, i);
      if (stage || vertex)
         for (unsigned i = 0; i < count; ++i)
            n +=
               snprintf(s + n, 32768 - n,
                        "layout(std140) uniform B%u_%u { vec4 d%u_%u[4]; };\n",
                        stage, i, stage, i);
      n += snprintf(s + n, 32768 - n, "void main(){ vec4 sum=vec4(0);\n");
      if (stage || vertex)
         for (unsigned i = 0; i < count; ++i)
            n += snprintf(
               s + n, 32768 - n,
               "sum += d%u_%u[%u] * (selected==%u ? 1.0 : (selected==15 ? %.9f : 0.0));\n",
               stage, i, i % 4, i, 1.0 / count);
      if (!stage && vbos)
         for (unsigned i = 0; i < 16; ++i)
            n += snprintf(s + n, 32768 - n, "sum += a%u * 0.015625;\n", i);
      if (stage)
         snprintf(s + n, 32768 - n,
                  "color=vec4((sum.rgb+shift.rgb+tint.rgb)*0.5,1); }");
      else
         snprintf(
            s + n, 32768 - n,
            "tint=%s; tint.x+=float(gl_VertexID)/64.0; vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2); float w=1.0+float(gl_VertexID)*0.25; gl_Position=vec4((p*2.0-1.0)*w,0,w); }",
            vertex ? "sum+shift" : "vec4(0.25,0.5,0.75,1)");
   }
   GLuint p = glCreateProgram();
   glAttachShader(p, compile_shader(GL_VERTEX_SHADER, shaders[0]));
   glAttachShader(p, compile_shader(GL_FRAGMENT_SHADER, shaders[1]));
   glLinkProgram(p);
   GLint linked;
   glGetProgramiv(p, GL_LINK_STATUS, &linked);
   if (!linked) {
      char log[4096];
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
      fail("buffers link");
   }
   glUseProgram(p);
   GLuint vao, fb, target, buf[48];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenTextures(1, &target);
   glBindTexture(GL_TEXTURE_2D, target);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          target, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      fail("buffers FBO");
   glGenBuffers(48, buf);
   for (unsigned b = 0; b < 32; ++b) {
      float data[16];
      for (unsigned i = 0; i < 16; ++i)
         data[i] = ((b * 7 + i * 3 + 11) % 64) / 64.0f;
      glBindBuffer(GL_UNIFORM_BUFFER, buf[b]);
      glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
   }
   if (vbos)
      for (unsigned i = 0; i < 16; ++i) {
         float data[24] = {0};
         for (unsigned v = 0; v < 3; ++v)
            for (unsigned c = 0; c < 4; ++c)
               data[v * 8 + 1 + c] = ((i * 5 + c * 9 + 3) % 64) / 64.0f;
         glBindBuffer(GL_ARRAY_BUFFER, buf[32 + i]);
         glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
         glEnableVertexAttribArray(i);
      }
   for (unsigned stage = 0; stage < 2; ++stage)
      if (stage || vertex)
         for (unsigned i = 0; i < count; ++i) {
            char name[32];
            snprintf(name, sizeof(name), "B%u_%u", stage, i);
            glUniformBlockBinding(p, glGetUniformBlockIndex(p, name),
                                  stage * 16 + i);
         }
   glViewport(0, 0, width, height);
   glEnable(GL_SCISSOR_TEST);
   for (unsigned frame = 0; frame < 2; ++frame) {
      for (unsigned draw = 0; draw < 16; ++draw) {
         glUniform4f(glGetUniformLocation(p, "shift"), draw / 128.0f,
                     frame / 16.0f, (frame + draw) / 256.0f, 0);
         glUniform1i(glGetUniformLocation(p, "selected"), draw);
         if (vbos)
            for (unsigned i = 0; i < 16; ++i) {
               glBindBuffer(GL_ARRAY_BUFFER, buf[32 + (i + frame + draw) % 16]);
               glVertexAttribPointer(i, 4, GL_FLOAT, GL_FALSE, 32,
                                     (void *)(uintptr_t)4);
            }
         for (unsigned stage = 0; stage < 2; ++stage)
            if (stage || vertex)
               for (unsigned i = 0; i < count; ++i)
                  glBindBufferBase(
                     GL_UNIFORM_BUFFER, stage * 16 + i,
                     buf[(stage * 16 + i + frame * 5 + draw * 3) % 32]);
         unsigned y = draw * height / 16, end = (draw + 1) * height / 16;
         glScissor(0, y, width, end - y);
         glDrawArrays(GL_TRIANGLES, 0, 3);
      }
      glFinish();
      if (glGetError() != GL_NO_ERROR)
         fail("buffers draw");
      fprintf(stderr, "BUFFERS_FRAME %u count=%u vertex=%u\n", frame, count,
              vertex);
   }
   return 0;
}
