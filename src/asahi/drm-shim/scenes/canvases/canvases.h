/* SPDX-License-Identifier: MIT */
static int
run_canvases(unsigned width, unsigned height)
{
   const char *vs =
      "#version 300 es\nprecision highp float;uniform vec4 transform;"
      "void main(){int i=gl_VertexID;vec2 p=vec2(i==0?-.75:(i==1?.75:0.),i==2?.75:-.75);"
      "gl_Position=vec4(p*transform.w+transform.xy,transform.z,1);}";
   const char *fs = "#version 300 es\nprecision highp float;uniform vec4 color;"
                    "out vec4 result;void main(){result=color;}";
   GLuint program = glCreateProgram(), v = compile_shader(GL_VERTEX_SHADER, vs),
          f = compile_shader(GL_FRAGMENT_SHADER, fs);
   glAttachShader(program, v);
   glAttachShader(program, f);
   glLinkProgram(program);
   GLint linked;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked)
      fail("blend program link");
   glDeleteShader(v);
   glDeleteShader(f);
   glUseProgram(program);
   bool no_depth = getenv("T8132_GLES_NO_DEPTH") != NULL;
   GLuint vao, fb[2], tex[4];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenFramebuffers(2, fb);
   glGenTextures(4, tex);
   for (unsigned canvas = 0; canvas < 2; ++canvas) {
      glBindFramebuffer(GL_FRAMEBUFFER, fb[canvas]);
      for (unsigned attachment = 0; attachment < (no_depth ? 1u : 2u);
           ++attachment) {
         glBindTexture(GL_TEXTURE_2D, tex[2 * canvas + attachment]);
         glTexStorage2D(GL_TEXTURE_2D, 1,
                        attachment ? GL_DEPTH_COMPONENT32F : GL_RGBA8, width,
                        height);
         glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            attachment ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, tex[2 * canvas + attachment], 0);
      }
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         fail("canvas framebuffer");
   }
   glViewport(0, 0, width, height);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);
   if (no_depth)
      glDisable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   glClearDepthf(1);
   GLint transform = glGetUniformLocation(program, "transform"),
         color = glGetUniformLocation(program, "color");
   const float transforms[][4] = {
      {0, 0, .2, 1},     {-.2, 0, 0, .7},   {.2, 0, -.2, .7},   {0, 0, .8, .5},
      {0, .45, -.5, .2}, {0, -.2, -.1, .5}, {.35, -.4, -.1, .3}};
   const float colors[][4] = {
      {.25, .5, .75, .8}, {.9, .1, .2, .5}, {.1, .8, .3, .25}, {1, 0, 1, 1},
      {.1, .2, .9, 1},    {1, 1, 1, 0},     {.8, .7, .9, 1}};
   if (getenv("T8132_GLES_BATCH_BOUNDARY")) {
      glBindFramebuffer(GL_FRAMEBUFFER, fb[0]);
      glClearColor(.2, .4, .6, .25);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glUniform4fv(transform, 1, transforms[0]);
      glUniform4fv(color, 1, colors[0]);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE); glDepthMask(GL_FALSE);
      const float increment[4] = {1.0/255, 2.0/255, 3.0/255, 4.0/255};
      glUniform4fv(transform, 1, transforms[1]);
      glUniform4fv(color, 1, increment);
      for (unsigned draw = 0; draw < 64; ++draw)
         glDrawArrays(GL_TRIANGLES, 0, 3);
      glFinish();
      if (glGetError() != GL_NO_ERROR) fail("canvas batch boundary");
      goto finish;
   }
   unsigned frames = read_dimension("T8132_GLES_FRAMES", 8);
   for (unsigned frame = 0; frame < frames; ++frame) {
      unsigned canvas = frame % 2, phase = frame / 2;
      glBindFramebuffer(GL_FRAMEBUFFER, fb[canvas]);
      glDisable(GL_BLEND);
      glDepthMask(GL_TRUE);
      if (phase % 2 == 0) {
         if (phase == 0) {
            if (canvas == 0)
               glClearColor(0.2, 0.4, 0.6, 0.25);
            else
               glClearColor(0.6, 0.2, 0.4, 0.75);
         } else
            glClearColor(0, 0, 0, 0);
         glUseProgram(0);
         glClear(GL_COLOR_BUFFER_BIT | (no_depth ? 0 : GL_DEPTH_BUFFER_BIT));
      }
      glUseProgram(program);
      unsigned d = phase % 2;
      if (d) {
         glEnable(GL_BLEND);
         glDepthMask(GL_FALSE);
         glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                             GL_ONE_MINUS_SRC_ALPHA);
      }
      glUniform4fv(transform, 1, transforms[d]);
      glUniform4fv(color, 1, colors[d]);
      if (!getenv("T8132_GLES_CLEAR_ONLY") || d)
         glDrawArrays(GL_TRIANGLES, 0, 3);
      glFinish();
      if (glGetError() != GL_NO_ERROR)
         fail("canvas draw");
      printf("T8132_CANVAS_FRAME frame=%u canvas=%u phase=%u\n", frame, canvas,
             phase);
   }
finish:
   glDeleteProgram(program);
   glDeleteTextures(4, tex);
   glDeleteFramebuffers(2, fb);
   glDeleteVertexArrays(1, &vao);
   return 0;
}
