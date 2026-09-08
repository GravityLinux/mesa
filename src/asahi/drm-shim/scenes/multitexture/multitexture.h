/* SPDX-License-Identifier: MIT */
static int
run_multitexture(unsigned width, unsigned height)
{
   bool lod = getenv("T8132_MULTITEXTURE_LOD") != NULL;
   const char *vs =
      "#version 300 es\n"
      "void main(){ vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
      "gl_Position=vec4(p*2.0-1.0,0,1); }";
   char fs[16384];
   size_t length =
      snprintf(fs, sizeof(fs),
               "#version 300 es\nprecision highp float;\nout vec4 color;\n");
   for (unsigned i = 0; i < 16; ++i)
      length += snprintf(fs + length, sizeof(fs) - length,
                         "uniform sampler2D t%u;\n", i);
   length +=
      snprintf(fs + length, sizeof(fs) - length,
               "void main(){ vec2 uv=gl_FragCoord.xy/vec2(%u,%u)*0.875+0.0625;"
               "vec4 sum=vec4(0); float band=floor(gl_FragCoord.x/16.0);\n",
               width, height);
   for (unsigned i = 0; i < 16; ++i)
      length += snprintf(
         fs + length, sizeof(fs) - length,
         "sum+=texture(t%u,uv)*((gl_FragCoord.y<128.0)?0.0625:((band==%u.0)?1.0:0.0));\n",
         i, i);
   snprintf(fs + length, sizeof(fs) - length, "color=sum;}\n");
   GLuint v = compile_shader(GL_VERTEX_SHADER, vs),
          f = compile_shader(GL_FRAGMENT_SHADER, fs);
   GLuint program = glCreateProgram();
   glAttachShader(program, v);
   glAttachShader(program, f);
   glLinkProgram(program);
   GLint linked;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked)
      fail("multitexture link");
   glUseProgram(program);
   for (unsigned i = 0; i < 16; ++i) {
      char name[16];
      snprintf(name, sizeof(name), "t%u", i);
      glUniform1i(glGetUniformLocation(program, name), i);
   }
   GLuint vao, fb, target, images[16], samplers[16];
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
      fail("multitexture framebuffer");
   glGenTextures(16, images);
   glGenSamplers(lod ? 16 : 2, samplers);
   for (unsigned i = 0; i < (lod ? 16 : 2); ++i) {
      glSamplerParameteri(
         samplers[i], GL_TEXTURE_MIN_FILTER,
         lod ? GL_LINEAR_MIPMAP_LINEAR : (i ? GL_LINEAR : GL_NEAREST));
      glSamplerParameteri(samplers[i], GL_TEXTURE_MAG_FILTER,
                          i ? GL_LINEAR : GL_NEAREST);
      glSamplerParameteri(samplers[i], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glSamplerParameteri(samplers[i], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      if (lod) {
         glSamplerParameterf(samplers[i], GL_TEXTURE_MIN_LOD, i * .25f);
         glSamplerParameterf(samplers[i], GL_TEXTURE_MAX_LOD, i * .25f);
      }
   }
   for (unsigned i = 0; i < 16; ++i) {
      glBindTexture(GL_TEXTURE_2D, images[i]);
      glTexStorage2D(GL_TEXTURE_2D, lod ? 5 : 1, GL_RGBA8, lod ? 16 : 8, lod ? 16 : 8);
      for (unsigned level = 0; level < (lod ? 5 : 1); ++level) {
         unsigned side = (lod ? 16 : 8) >> level;
         uint8_t pixels[16 * 16 * 4];
         for (unsigned y = 0; y < side; ++y)
            for (unsigned x = 0; x < side; ++x) {
               unsigned at = (y * side + x) * 4;
               pixels[at] =
                  (17 + i * 11 + (lod ? level * 53 : x * 23 + y * 7)) & 255;
               pixels[at + 1] =
                  (31 + i * 13 + (lod ? level * 37 : x * 5 + y * 29)) & 255;
               pixels[at + 2] =
                  (47 + i * 7 + (lod ? level * 61 : x * 17 + y * 19)) & 255;
               pixels[at + 3] = 255;
            }
         glTexSubImage2D(GL_TEXTURE_2D, level, 0, 0, side, side, GL_RGBA,
                         GL_UNSIGNED_BYTE, pixels);
      }
   }
   glViewport(0, 0, width, height);
   glEnable(GL_SCISSOR_TEST);
   for (unsigned frame = 0; frame < 2; ++frame) {
      for (unsigned draw = 0; draw < 3; ++draw) {
         for (unsigned unit = 0; unit < 16; ++unit) {
            /* Last draw aliases one texture with alternating sampler objects. */
            unsigned image = draw == 2 ? 5 : (unit + frame * 7 + draw * 3) % 16;
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, images[image]);
            glBindSampler(unit,
                          samplers[(unit + draw + frame) % (lod ? 16 : 2)]);
         }
         unsigned y = draw * height / 3, end = (draw + 1) * height / 3;
         glScissor(0, y, width, end - y);
         glDrawArrays(GL_TRIANGLES, 0, 3);
      }
      glFinish();
      if (glGetError() != GL_NO_ERROR)
         fail("multitexture draw");
      fprintf(stderr, "MULTITEXTURE_FRAME %u\n", frame);
   }
   return 0;
}
