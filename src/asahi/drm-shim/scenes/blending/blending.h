/* SPDX-License-Identifier: MIT */
static int
run_blending(unsigned width, unsigned height)
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
   GLuint vao, fb, tex[2];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenTextures(2, tex);
   for (unsigned i = 0; i < 2; ++i) {
      glBindTexture(GL_TEXTURE_2D, tex[i]);
      glTexStorage2D(GL_TEXTURE_2D, 1, i ? GL_DEPTH_COMPONENT32F : GL_RGBA8,
                     width, height);
      glFramebufferTexture2D(GL_FRAMEBUFFER,
                             i ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, tex[i], 0);
   }
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      fail("blend framebuffer");
   glViewport(0, 0, width, height);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);
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
   bool probe = getenv("T8132_GLES_BLEND_PROBE") != NULL;
   unsigned frames = read_dimension("T8132_GLES_FRAMES", 8);
   for (unsigned frame = 0; frame < frames; ++frame) {
      glClearColor(4.0/255, 5.0/255, 15.0/255, 1);
      glDepthMask(GL_TRUE);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      const unsigned orders[2][7] = {{0, 1, 2, 5, 6, 3, 4},
                                     {0, 6, 5, 2, 1, 3, 4}};
      for (unsigned n = 0; n < 7; ++n) {
         unsigned d = orders[frame % 2][n];
         if (n == 0 || n == 6) {
            glDisable(GL_BLEND);
            glDepthMask(GL_TRUE);
         } else {
            glEnable(GL_BLEND);
            glDepthMask(GL_FALSE);
            if (probe)
               glBlendFunc(GL_ZERO, GL_ONE);
            else if ((frame / 2) % 2 == 0)
               glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                                   GL_ONE_MINUS_SRC_ALPHA);
            else
               glBlendFunc(GL_SRC_ALPHA, GL_ONE);
         }
         glColorMask(GL_TRUE, frame < 4 || n == 0 || n == 6, GL_TRUE,
                     frame < 4 || n == 0 || n == 6);
         glUniform4fv(transform, 1, transforms[d]);
         glUniform4fv(color, 1, colors[d]);
         glDrawArrays(GL_TRIANGLES, 0, 3);
      }
      glFinish();
      if (glGetError() != GL_NO_ERROR)
         fail("blend draw");
      printf("T8132_BLEND_FRAME frame=%u probe=%u\n", frame, probe);
   }
   glDeleteProgram(program);
   glDeleteTextures(2, tex);
   glDeleteFramebuffers(1, &fb);
   glDeleteVertexArrays(1, &vao);
   return 0;
}
