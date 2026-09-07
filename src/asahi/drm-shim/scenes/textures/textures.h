/* SPDX-License-Identifier: MIT */
static int
run_textures(unsigned width, unsigned height)
{
   char *vs = read_shader_source("T8132_GLES_VERTEX_SOURCE");
   char *fs = read_shader_source("T8132_GLES_FRAGMENT_SOURCE");
   if (!vs || !fs) fail("texture sources");
   GLuint program = glCreateProgram();
   GLuint v = compile_shader(GL_VERTEX_SHADER, vs), f = compile_shader(GL_FRAGMENT_SHADER, fs);
   free(vs); free(fs);
   unsigned atlas_w = getenv("T8132_TEXTURE_ATLAS") ? atoi(getenv("T8132_TEXTURE_ATLAS")) : 0;
   unsigned atlas_h = getenv("T8132_TEXTURE_ATLAS_HEIGHT") ? (unsigned)atoi(getenv("T8132_TEXTURE_ATLAS_HEIGHT")) : atlas_w;
   unsigned channels = getenv("T8132_TEXTURE_CHANNELS") ? atoi(getenv("T8132_TEXTURE_CHANNELS")) : 4;
   GLenum upload_format = channels == 1 ? GL_RED : channels == 2 ? GL_RG : GL_RGBA;
   GLenum internal_format = channels == 1 ? GL_R8 : channels == 2 ? GL_RG8 : GL_RGBA8;
   bool legacy = getenv("T8132_TEXTURE_LEGACY") != NULL;
   if (legacy) {
      if (channels != 2) fail("legacy atlas requires two channels");
      upload_format = internal_format = GL_LUMINANCE_ALPHA;
   }
   if (channels != 1 && channels != 2 && channels != 4) fail("atlas channel count");
   if (atlas_w && (atlas_w < 160 || atlas_h < 160 || atlas_w > 4096 || atlas_h > 4096))
      fail("atlas dimensions");
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   unsigned mip_size = getenv("T8132_TEXTURE_MIPS") ? atoi(getenv("T8132_TEXTURE_MIPS")) : 0;
   unsigned mip_height = getenv("T8132_TEXTURE_MIP_HEIGHT") ? (unsigned)atoi(getenv("T8132_TEXTURE_MIP_HEIGHT")) : mip_size;
   unsigned mip_levels = mip_size ? (unsigned)floorf(log2f(mip_size > mip_height ? mip_size : mip_height)) + 1 : 1;
   bool generate = getenv("T8132_TEXTURE_GENERATE_MIPS") != NULL;
   bool mip_linear = getenv("T8132_TEXTURE_MIP_LINEAR") != NULL;
   bool blend = getenv("T8132_TEXTURE_BLEND") != NULL;
   bool rtt = getenv("T8132_TEXTURE_RTT") != NULL;
   bool linear = getenv("T8132_TEXTURE_LINEAR") != NULL;
   bool mixed = getenv("T8132_GLES_TEXTURES_MIXED") != NULL;
   glAttachShader(program, v); glAttachShader(program, f); glLinkProgram(program);
   GLint linked;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) fail("texture program link");
   GLuint solid = glCreateProgram();
   GLuint solid_fs = compile_shader(GL_FRAGMENT_SHADER,
      "#version 300 es\nprecision highp float;\nout vec4 color;\n"
      "void main() { color=vec4(0.25,0.5,0.75,1.0); }\n");
   glAttachShader(solid, v); glAttachShader(solid, solid_fs); glLinkProgram(solid);
   glGetProgramiv(solid, GL_LINK_STATUS, &linked);
   if (!linked) fail("solid program link");
   glDeleteShader(solid_fs); glDeleteShader(v); glDeleteShader(f);
   glUseProgram(program);
   GLuint vao, fb, target, images[2];
   glGenVertexArrays(1, &vao); glBindVertexArray(vao);
   glGenFramebuffers(1, &fb); glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenTextures(1, &target); glBindTexture(GL_TEXTURE_2D, target);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target, 0);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      fail("texture framebuffer");
   glGenTextures(2, images);
   for (unsigned image = 0; image < 2; ++image) {
      uint8_t pixels[64 * 64 * 4];
      for (unsigned y = 0; y < 64; ++y) for (unsigned x = 0; x < 64; ++x) {
         unsigned at = (y * 64 + x) * 4;
         pixels[at] = 17 + 3*x + 7*image;
         pixels[at+1] = 23 + 3*y + 11*image;
         pixels[at+2] = 31 + x + y + 13*image;
         pixels[at+3] = 64 + x + y;
      }
      glActiveTexture(GL_TEXTURE0 + (image ? 1 : 3));
      glBindTexture(GL_TEXTURE_2D, images[image]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      if (atlas_w) {
         uint8_t *data = malloc(atlas_w * atlas_h * channels);
         if (!data) fail("atlas allocation");
         for (unsigned y = 0; y < atlas_h; ++y) for (unsigned x = 0; x < atlas_w; ++x)
            for (unsigned c = 0; c < channels; ++c)
               data[(y * atlas_w + x) * channels + c] = (17 + x*3 + y*5 + c*47) & 255;
         glTexImage2D(GL_TEXTURE_2D, 0, internal_format, atlas_w, atlas_h, 0,
                      upload_format, GL_UNSIGNED_BYTE, data);
         free(data);
         if (channels < 4 && !legacy) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, channels == 1 ? GL_ONE : GL_RED);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, channels == 1 ? GL_ONE : GL_RED);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, channels == 1 ? GL_ONE : GL_RED);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, channels == 1 ? GL_RED : GL_GREEN);
         }
      } else if (mip_size) {
         glTexStorage2D(GL_TEXTURE_2D, mip_levels, GL_RGBA8, mip_size, mip_height);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                         mip_linear ? (linear ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_LINEAR)
                                    : (linear ? GL_LINEAR_MIPMAP_NEAREST : GL_NEAREST_MIPMAP_NEAREST));
         for (unsigned l = 0; l < (generate ? 1 : mip_levels); ++l) {
            unsigned size = mip_size >> l; if (!size) size = 1;
            unsigned level_height = mip_height >> l; if (!level_height) level_height = 1;
            uint8_t *data = malloc(size * level_height * 4);
            if (!data) fail("mipmap allocation");
            for (unsigned y = 0; y < level_height; ++y) for (unsigned x = 0; x < size; ++x) {
               unsigned at = (y * size + x) * 4, checker = (x ^ y) & 1;
               data[at] = generate ? 20 + 180*checker + 7*image : 17+19*l+7*image;
               data[at+1] = generate ? 40+120*checker+11*image : 23+11*l+11*image;
               data[at+2] = generate ? 60+80*checker+13*image : 31+7*l+13*image;
               data[at+3] = 255;
            }
            glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, size, level_height, GL_RGBA, GL_UNSIGNED_BYTE, data);
            free(data);
         }
         if (generate) glGenerateMipmap(GL_TEXTURE_2D);
         if (glGetError() != GL_NO_ERROR) fail("mipmap upload/generation");
      } else {
         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
      }
   }
   GLuint canvas = 0, canvas_fb = 0;
   if (rtt) {
      glActiveTexture(GL_TEXTURE2);
      glGenTextures(1, &canvas); glBindTexture(GL_TEXTURE_2D, canvas);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glGenFramebuffers(1, &canvas_fb); glBindFramebuffer(GL_FRAMEBUFFER, canvas_fb);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, canvas, 0);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         fail("canvas framebuffer");
   }
   glViewport(0, 0, width, height);
   glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND); glDisable(GL_CULL_FACE);
   GLint image_location = glGetUniformLocation(program, "image");
   for (unsigned frame = 0; frame < 2; ++frame) {
      if (atlas_w && frame) {
         unsigned patch_w = 97, patch_h = 83, row = patch_w + 7;
         uint8_t *data = malloc(row * patch_h * channels);
         if (!data) fail("atlas update allocation");
         memset(data, 0xa5, row * patch_h * channels);
         for (unsigned y = 0; y < patch_h; ++y) for (unsigned x = 0; x < patch_w; ++x)
            for (unsigned c = 0; c < channels; ++c)
               data[(y * row + x) * channels + c] = 211 - 37*c;
         glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, images[0]);
         glPixelStorei(GL_UNPACK_ROW_LENGTH, row);
         glTexSubImage2D(GL_TEXTURE_2D, 0, atlas_w/3+3, atlas_h/3+5,
                         patch_w, patch_h, upload_format, GL_UNSIGNED_BYTE, data);
         glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
         free(data);
      }
      glBindFramebuffer(GL_FRAMEBUFFER, rtt ? canvas_fb : fb);
      if (blend) {
         glDisable(GL_BLEND); glDisable(GL_SCISSOR_TEST);
         glUseProgram(solid); glDrawArrays(GL_TRIANGLES, 0, 6);
         glEnable(GL_BLEND);
         glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
      }
      glUseProgram(program);
      glUniform2f(glGetUniformLocation(program, "uv_scale"), mixed ? 2.0f : 1.0f,
                  mixed ? 2.0f : 1.0f);
      glUniform2f(glGetUniformLocation(program, "uv_offset"), mixed ? -0.5f : 0.0f,
                  mixed ? -0.5f : 0.0f);
      glUniform4f(glGetUniformLocation(program, "tint"), 1, 1, 1, 1);
      glUniform1i(image_location, atlas_w ? 3 : frame ? 1 : 3);
      glUniform1i(glGetUniformLocation(program, "forward_order"), !frame);
      glUniform1i(glGetUniformLocation(program, "loop_count"),
                  getenv("T8132_TEXTURE_LOOPS") ? atoi(getenv("T8132_TEXTURE_LOOPS")) : 1);
      if (mixed) { glEnable(GL_SCISSOR_TEST); glScissor(0, 0, 120, height); }
      if (mip_size) {
         glEnable(GL_SCISSOR_TEST);
         for (unsigned l = 0; l < mip_levels; ++l) {
            float density = exp2f(l + (mip_linear ? .5f : 0));
            glUniform2f(glGetUniformLocation(program, "uv_scale"),
                        density * width / mip_size, density * height / mip_height);
            glScissor(l * width / mip_levels, 0,
                      (l+1) * width / mip_levels - l * width / mip_levels, height);
            glDrawArrays(GL_TRIANGLES, 0, 6);
         }
      } else {
         glDrawArrays(GL_TRIANGLES, 0, 6);
      }
      if (mixed) {
         glUseProgram(solid); glScissor(120, 0, 16, height);
         glDrawArrays(GL_TRIANGLES, 0, 6);
         glUseProgram(program); glScissor(136, 0, 120, height);
         glUniform1i(image_location, frame ? 3 : 1);
         glUniform4f(glGetUniformLocation(program, "tint"), 0.5f, 1, 0.25f, 1);
         glDrawArrays(GL_TRIANGLES, 0, 6);
      }
      if (rtt) {
         glBindFramebuffer(GL_FRAMEBUFFER, fb);
         glDisable(GL_BLEND);
         glDisable(GL_SCISSOR_TEST);
         glUseProgram(program);
         glUniform1i(image_location, 2);
         glUniform2f(glGetUniformLocation(program, "uv_scale"), 1, 1);
         glUniform2f(glGetUniformLocation(program, "uv_offset"), 0, 0);
         glUniform4f(glGetUniformLocation(program, "tint"), 1, 1, 1, 1);
         glDrawArrays(GL_TRIANGLES, 0, 6);
      }
      glFinish();
      if (glGetError() != GL_NO_ERROR) fail("texture draw");
      printf("T8132_TEXTURE_FRAME frame=%u\n", frame);
   }
   if (rtt) { glDeleteTextures(1, &canvas); glDeleteFramebuffers(1, &canvas_fb); }
   glDeleteTextures(2, images); glDeleteTextures(1, &target);
   glDeleteProgram(solid);
   glDeleteFramebuffers(1, &fb); glDeleteVertexArrays(1, &vao); glDeleteProgram(program);
   return 0;
}
