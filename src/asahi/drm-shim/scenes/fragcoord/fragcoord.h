/* SPDX-License-Identifier: MIT */
static int
run_fragcoord(unsigned width, unsigned height)
{
   char *vs = read_shader_source("T8132_GLES_VERTEX_SOURCE");
   char *fs = read_shader_source("T8132_GLES_FRAGMENT_SOURCE");
   if (!vs || !fs)
      fail("fragcoord shader sources");
   GLuint program = glCreateProgram();
   GLuint v = compile_shader(GL_VERTEX_SHADER, vs),
          f = compile_shader(GL_FRAGMENT_SHADER, fs);
   free(vs);
   free(fs);
   glAttachShader(program, v);
   glAttachShader(program, f);
   glLinkProgram(program);
   GLint linked;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked)
      fail("fragcoord program link");
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
      fail("fragcoord framebuffer");
   glViewport(0, 0, width, height);
   glDisable(GL_CULL_FACE);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_ALWAYS);
   glClearDepthf(1);
   GLint mode = glGetUniformLocation(program, "mode");
   glUniform2f(glGetUniformLocation(program, "extent"), width, height);
   unsigned frames = read_dimension("T8132_GLES_FRAMES", 3);
   for (unsigned frame = 0; frame < frames; ++frame) {
      glClearColor(4.0 / 255, 5.0 / 255, 15.0 / 255, 1);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glUniform1i(mode, frame % 3);
      if (frame == 2)
         glViewport(37, 53, width - 101, height - 127);
      else
         glViewport(0, 0, width, height);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glFinish();
      if (glGetError() != GL_NO_ERROR)
         fail("fragcoord draw");
      printf("T8132_FRAGCOORD_FRAME frame=%u\n", frame);
   }
   glDeleteProgram(program);
   glDeleteTextures(2, tex);
   glDeleteFramebuffers(1, &fb);
   glDeleteVertexArrays(1, &vao);
   return 0;
}
