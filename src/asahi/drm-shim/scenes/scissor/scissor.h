/* SPDX-License-Identifier: MIT */
static int
run_scissor(unsigned width, unsigned height)
{
   char *vs = read_shader_source("T8132_GLES_VERTEX_SOURCE");
   char *fs = read_shader_source("T8132_GLES_FRAGMENT_SOURCE");
   if (!vs || !fs)
      fail("scissor shader sources");
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
      fail("scissor program link");
   glDeleteShader(v);
   glDeleteShader(f);
   glUseProgram(program);
   GLuint vao, fb, tex[2];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenTextures(2, tex);
   bool color_only = getenv("T8132_SCISSOR_COLOR_ONLY") != NULL;
   for (unsigned i = 0; i < (color_only ? 1u : 2u); ++i) {
      glBindTexture(GL_TEXTURE_2D, tex[i]);
      glTexStorage2D(GL_TEXTURE_2D, 1, i ? GL_DEPTH_COMPONENT32F : GL_RGBA8,
                     width, height);
      glFramebufferTexture2D(GL_FRAMEBUFFER,
                             i ? GL_DEPTH_ATTACHMENT : GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, tex[i], 0);
   }
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      fail("scissor framebuffer");
   glViewport(0, 0, width, height);
   glDisable(GL_CULL_FACE);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_ALWAYS);
   glClearDepthf(1);
   GLint tint = glGetUniformLocation(program, "tint");
   unsigned frames = getenv("T8132_SCISSOR_BOUNDARY") ? 8 : 7;
   for (unsigned frame = 0; frame < frames; ++frame) {
      glDisable(GL_SCISSOR_TEST);
      glViewport(0, 0, width, height);
      if (frame < 5) {
         glClearColor(4.0 / 255, 5.0 / 255, 15.0 / 255, 1);
         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      }
      glUniform4f(tint, .75, .25, .5, 1);
      if (frame == 7) {
         glClearColor(4.0 / 255, 5.0 / 255, 15.0 / 255, 1);
         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
         glEnable(GL_SCISSOR_TEST);
         for (unsigned i = 0; i < 40; ++i) {
            glScissor(7 + (i % 8) * 53, 11 + (i / 8) * 67, 41, 55);
            glUniform4f(tint, .25 + .25 * (i % 3), .125 + .125 * (i % 5), .5, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3);
         }
      } else if (frame == 0) {
         glViewport(37, 53, width - 101, height - 127);
         glDrawArrays(GL_TRIANGLES, 0, 3);
      } else if (frame == 1 || frame == 2) {
         glEnable(GL_SCISSOR_TEST);
         if (frame == 1) glScissor(13, 19, 91, 73);
         else glScissor(-11, -7, 43, 39);
         glDrawArrays(GL_TRIANGLES, 0, 3);
      } else if (frame == 3) {
         glEnable(GL_SCISSOR_TEST);
         glScissor(17, 23, 103, 89);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         glUniform4f(tint, .125, .75, .375, .5);
         glScissor(61, 47, 109, 67);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         glUniform4f(tint, 1, 1, 1, 1);
         glScissor(10, 10, 0, 20);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         glScissor(width + 10, 3, 20, 20);
         glDrawArrays(GL_TRIANGLES, 0, 3);
         glViewport(37, 53, 111, 95);
         glScissor(101, 89, 100, 100);
         glUniform4f(tint, .5, .125, .875, .25);
         glDrawArrays(GL_TRIANGLES, 0, 3);
      } else {
         glViewport(2, 3, 7, 9); /* glClear must ignore the API viewport. */
         glEnable(GL_SCISSOR_TEST);
         if (frame == 4) {
            glScissor(29, 41, 107, 83);
            glClearColor(.2, .4, .6, .5);
         } else if (frame == 5) {
            glScissor(83, 97, 113, 71);
            glClearColor(.8, .6, .2, .25);
         } else {
            glScissor(0, 0, 0, 17);
            glClearColor(1, 0, 1, 1);
         }
         glClear(GL_COLOR_BUFFER_BIT);
         if (frame == 6) {
            glScissor(width + 3, height + 7, 13, 19);
            glClear(GL_COLOR_BUFFER_BIT);
            /* A visible draw with a zero color mask forces a submission so
             * the unchanged attachment has a hardware readback. */
            glDisable(GL_SCISSOR_TEST);
            glViewport(0, 0, width, height);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            glDepthMask(GL_FALSE);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);
         }
      }
      glFinish();
      if (glGetError() != GL_NO_ERROR) fail("scissor draw");
      printf("T8132_SCISSOR_FRAME frame=%u\n", frame);
   }
   glDeleteProgram(program);
   glDeleteTextures(2, tex);
   glDeleteFramebuffers(1, &fb);
   glDeleteVertexArrays(1, &vao);
   return 0;
}
