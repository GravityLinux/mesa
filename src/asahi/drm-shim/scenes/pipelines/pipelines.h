/* SPDX-License-Identifier: MIT */
/* Ordinary GLES pipeline changes; no driver-visible fixture selection. */
static int
run_pipelines(unsigned width, unsigned height)
{
   const char *vs[] = {
      "#version 300 es\nprecision highp float;\n"
      "layout(location=0) in vec2 p; uniform vec4 transform; out vec3 a;\n"
      "void main(){gl_Position=vec4(p*transform.w+transform.xy,transform.z,1);"
      "a=vec3(p*.25+.5,(p.x+p.y)*.125+.375);}",
      "#version 300 es\nprecision highp float;\n"
      "layout(location=0) in vec2 p; uniform vec4 transform;\n"
      "out vec3 a; out vec3 b; out vec3 c;\n"
      "void main(){gl_Position=vec4(vec2(-p.x,p.y)*transform.w+transform.xy,transform.z,1);"
      "a=vec3(p*.25+.5,(p.x+p.y)*.125+.375);b=vec3(p.y*.125+.25,p.x*.125+.5,(p.x-p.y)*.0625+.625);"
      "c=vec3(p.x*.0625+.125,p.y*.0625+.375,p.x*.03125+p.y*.015625+.75);}",
      "#version 300 es\nprecision highp float;\n"
      "uniform vec4 transform;\n"
      "void main(){int i=gl_VertexID;vec2 p=vec2(i==0?-.75:(i==1?.75:0.),i==2?.75:-.75);"
      "gl_Position=vec4(p*transform.w+transform.xy,transform.z,1);}",
   };
   const char *fs[] = {
      "#version 300 es\nprecision highp float;\n"
      "in vec3 a; uniform vec3 tint;out vec4 result;"
      "void main(){result=vec4(a*tint,1);}",
      "#version 300 es\nprecision highp float;\n"
      "in vec3 a;in vec3 b;in vec3 c;uniform vec3 tint;out vec4 result;"
      "void main(){result=vec4((a*.5+b*.25+c*.125)*tint,1);}",
      "#version 300 es\nprecision highp float;\n"
      "out vec4 result;void main(){result=vec4(.125,.75,1,1);}",
   };
   bool split = getenv("T8132_GLES_PIPELINE_SPLIT") != NULL;
   bool wide = getenv("T8132_GLES_PIPELINE_WIDE") != NULL;
   bool pin = getenv("T8132_GLES_PIPELINE_PIN") != NULL;
   unsigned program_count = pin ? 20 : 4;
   GLuint programs[20];
   for (unsigned i = 0; i < program_count; ++i) {
      programs[i] = glCreateProgram();
      GLuint v = compile_shader(GL_VERTEX_SHADER, vs[pin || i == 3 ? 2 : i]);
      char constant_fs[256];
      snprintf(constant_fs, sizeof(constant_fs),
         "#version 300 es\nprecision highp float;out vec4 result;\n"
         "void main(){result=vec4(%.9f,%.9f,%.9f,1);}",
         .1+.03*i, .2+.02*i, .9-.03*i);
      char wide_fs[768];
      snprintf(wide_fs, sizeof(wide_fs),
         "#version 300 es\nprecision highp float;out vec4 result;\n"
         "void main(){vec2 p=gl_FragCoord.xy*.01;"
         "vec3 wave=vec3(sin(p.x+%u.),cos(p.y+%u.),sin(p.x+p.y+%u.));"
         "result=vec4(vec3(%.9f,%.9f,%.9f)+.02*wave,1);}",
         i, i, i, .1+.03*i, .2+.02*i, .9-.03*i);
      GLuint f = compile_shader(GL_FRAGMENT_SHADER, wide ? wide_fs : pin ? constant_fs : i == 3 ?
         "#version 300 es\nprecision highp float;out vec4 result;\n"
         "void main(){result=vec4(.25,.5,.75,1);}" : fs[i]);
      glAttachShader(programs[i], v);
      glAttachShader(programs[i], f);
      glLinkProgram(programs[i]);
      GLint ok;
      glGetProgramiv(programs[i], GL_LINK_STATUS, &ok);
      if (!ok) fail("pipeline fixture link");
      glDeleteShader(v);
      glDeleteShader(f);
   }
   GLuint vao, vb;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vb);
   glBindBuffer(GL_ARRAY_BUFFER, vb);
   const float vertices[] = {-.75,-.75, .75,-.75, 0,.75};
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
   glEnableVertexAttribArray(0);

   GLuint index_buffer = 0;
   if (split) {
      const uint16_t indices[] = {0, 1, 2};
      glGenBuffers(1, &index_buffer);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
   }

   GLuint fb, textures[2];
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenTextures(2, textures);
   for (unsigned i = 0; i < 2; ++i) {
      glBindTexture(GL_TEXTURE_2D, textures[i]);
      glTexStorage2D(GL_TEXTURE_2D, 1, i ? GL_DEPTH_COMPONENT32F : GL_RGBA8, width, height);
      glFramebufferTexture2D(GL_FRAMEBUFFER, i ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, textures[i], 0);
   }
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      fail("pipeline fixture framebuffer");
   glViewport(0, 0, width, height);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);
   glDepthMask(GL_TRUE);
   glClearDepthf(1);
   glDisable(GL_BLEND);
   glDisable(GL_CULL_FACE);
   static const unsigned sequence[] = {0,1,0,2,1,0};
   static const float transforms[][4] = {
      {-.3,-.1,.1,.8}, {0,.15,.4,.9}, {.4,-.2,-.2,.6},
      {-.45,.4,-.6,.3}, {0,-.5,-.4,.4}, {.45,.5,0,.35},
   };
   static const float tints[][3] = {
      {1,.5,.25}, {.25,1,.5}, {.5,.25,1}, {1,1,1}, {1,.5,1}, {.5,1,1},
   };
   unsigned frames = read_dimension("T8132_GLES_FRAMES", 4);
   unsigned draws = read_dimension("T8132_GLES_DRAWS", pin ? 21 : 6);
   bool rollover = getenv("T8132_GLES_PIPELINE_ROLLOVER") != NULL;
   for (unsigned frame = 0; frame < frames; ++frame) {
      glClearColor(.75,.73,1,1);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      for (unsigned n = 0; n < draws; ++n) {
         unsigned order = frame % 2 ? draws-1-n : n;
         unsigned d = order % 6;
         unsigned selected = sequence[d];
         if (rollover && frame / 2 == 1 && selected == 2) selected = 3;
         if (pin) selected = order == 20 ? 0 : order;
         GLuint program = programs[selected];
         glUseProgram(program);
         float grid[4] = {-.8f+(order%5)*.4f, -.65f+(order/5)*.4f, 0, .16f};
         if (order == 20) { grid[0] = 0; grid[1] = .85f; }
         glUniform4fv(glGetUniformLocation(program, "transform"), 1,
                      pin ? grid : transforms[d]);
         glUniform3fv(glGetUniformLocation(program, "tint"), 1, tints[d]);
         if (split)
            glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL);
         else
            glDrawArrays(GL_TRIANGLES, 0, 3);
      }
      glFinish();
      if (glGetError() != GL_NO_ERROR) fail("pipeline fixture draw");
      printf("T8132_PIPELINES_FRAME frame=%u draws=%u reverse=%u\n",frame,draws,frame%2);
      fflush(stdout);
   }
   for (unsigned i = 0; i < program_count; ++i)
      glDeleteProgram(programs[i]);
   glDeleteBuffers(1, &vb);
   if (index_buffer) glDeleteBuffers(1, &index_buffer);
   glDeleteVertexArrays(1, &vao);
   glDeleteFramebuffers(1, &fb);
   glDeleteTextures(2, textures);
   return 0;
}
