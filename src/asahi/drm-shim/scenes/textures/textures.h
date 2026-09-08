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
   const char *texture_format = getenv("T8132_TEXTURE_FORMAT");
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
   const char *cull_mode = getenv("T8132_TEXTURE_CULL");
   const char *depth_mode = getenv("T8132_TEXTURE_DEPTH");
   const char *stencil_mode = getenv("T8132_TEXTURE_STENCIL");
   bool stencil_test = stencil_mode != NULL;
   bool stencil_zfail = stencil_test && !strcmp(stencil_mode, "zfail");
   bool stencil_sfail = stencil_test && !strcmp(stencil_mode, "sfail");
   bool stencil_persist = stencil_test && !strcmp(stencil_mode, "persist");
   bool stencil_compare = stencil_test && !strcmp(stencil_mode, "compare");
   bool stencil_discard = stencil_test && !strncmp(stencil_mode, "discard", 7);
   bool stencil_discard_color = stencil_test && !strcmp(stencil_mode, "discard-color");
   const char *blend_mode = getenv("T8132_TEXTURE_BLEND_MODE");
   bool blend = getenv("T8132_TEXTURE_BLEND") != NULL || blend_mode;
   bool rtt = getenv("T8132_TEXTURE_RTT") != NULL;
   const char *wrap_name = getenv("T8132_TEXTURE_WRAP");
   bool wrap_test = wrap_name != NULL;
   GLenum wrap = !wrap_test ? GL_CLAMP_TO_EDGE :
      !strcmp(wrap_name, "repeat") ? GL_REPEAT : GL_MIRRORED_REPEAT;
   bool linear = getenv("T8132_TEXTURE_LINEAR") != NULL;
   bool mixed = getenv("T8132_GLES_TEXTURES_MIXED") != NULL;
   glAttachShader(program, v); glAttachShader(program, f); glLinkProgram(program);
   GLint linked;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   if (!linked) fail("texture program link");
   GLuint solid = glCreateProgram();
   GLuint solid_fs = compile_shader(GL_FRAGMENT_SHADER,
      blend_mode && (!strcmp(blend_mode, "dstalpha") || !strcmp(blend_mode, "saturate")) ?
      "#version 300 es\nprecision highp float;\nout vec4 color;\n"
      "void main() { color=vec4(0.25,0.5,0.75,0.6); }\n" :
      "#version 300 es\nprecision highp float;\nout vec4 color;\n"
      "void main() { color=vec4(0.25,0.5,0.75,1.0); }\n");
   glAttachShader(solid, v); glAttachShader(solid, solid_fs); glLinkProgram(solid);
   glGetProgramiv(solid, GL_LINK_STATUS, &linked);
   if (!linked) fail("solid program link");
   GLuint stencil_writer = solid;
   if (stencil_discard) {
      stencil_writer = glCreateProgram();
      GLuint writer_fs = compile_shader(GL_FRAGMENT_SHADER,
         stencil_discard_color ?
         "#version 300 es\nprecision highp float;\nout vec4 color;\n"
         "void main() { if (gl_FragCoord.y < 128.0) discard;"
         "if (gl_FragCoord.y < 64.0) discard; color=vec4(0.25,0.5,0.75,1); }\n" :
         "#version 300 es\nprecision highp float;\n"
         "void main() { if (gl_FragCoord.y < 128.0) discard;"
         "if (gl_FragCoord.y < 64.0) discard; }\n");
      glAttachShader(stencil_writer, v); glAttachShader(stencil_writer, writer_fs);
      glLinkProgram(stencil_writer);
      glGetProgramiv(stencil_writer, GL_LINK_STATUS, &linked);
      if (!linked) fail("stencil-only discard link");
      glDeleteShader(writer_fs);
   }
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
   if (depth_mode) {
      GLuint depth;
      glGenRenderbuffers(1, &depth); glBindRenderbuffer(GL_RENDERBUFFER, depth);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         fail("depth framebuffer");
   }
   GLuint stencil = 0;
   if (stencil_test) {
      glGenRenderbuffers(1, &stencil);
      glBindRenderbuffer(GL_RENDERBUFFER, stencil);
      glRenderbufferStorage(GL_RENDERBUFFER, stencil_zfail ? GL_DEPTH32F_STENCIL8 : GL_STENCIL_INDEX8, width, height);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, stencil_zfail ? GL_DEPTH_STENCIL_ATTACHMENT : GL_STENCIL_ATTACHMENT,
                                 GL_RENDERBUFFER, stencil);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         fail("stencil framebuffer");
   }
   glGenTextures(2, images);
   for (unsigned image = 0; image < 2; ++image) {
      uint8_t pixels[64 * 64 * 4];
      for (unsigned y = 0; y < 64; ++y) for (unsigned x = 0; x < 64; ++x) {
         unsigned at = (y * 64 + x) * 4;
         pixels[at] = 17 + 3*x + 7*image;
         pixels[at+1] = 23 + 3*y + 11*image;
         pixels[at+2] = 31 + x + y + 13*image;
         pixels[at+3] = 64 + x + y;
         if (texture_format && !strcmp(texture_format, "srgb")) {
            pixels[at] = ((x + image) & 1) ? 255 : 0;
            pixels[at+1] = ((y + image) & 1) ? 192 : 16;
            pixels[at+2] = ((x ^ y ^ image) & 1) ? 128 : 32;
         }
      }
      glActiveTexture(GL_TEXTURE0 + (image ? 1 : 3));
      glBindTexture(GL_TEXTURE_2D, images[image]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
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
         GLenum mip_format = !texture_format ? GL_RGBA8 :
            !strcmp(texture_format, "srgb") ? GL_SRGB8_ALPHA8 :
            !strcmp(texture_format, "rgb565") ? GL_RGB565 : GL_RGBA16F;
         glTexStorage2D(GL_TEXTURE_2D, mip_levels, mip_format, mip_size, mip_height);
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
               if (generate && mip_format == GL_RGB565) {
                  /* Avoid halfway packed-format rounding cases. */
                  data[at] = 16 + 176*checker + 16*image;
                  data[at+1] = 40 + 120*checker + 8*image;
                  data[at+2] = 56 + 80*checker + 16*image;
               }
            }
            if (mip_format == GL_RGB565) {
               uint16_t *packed = malloc(size * level_height * 2);
               if (!packed) fail("packed mip allocation");
               for (unsigned i = 0; i < size * level_height; ++i)
                  packed[i] = ((data[4*i] >> 3) << 11) |
                              ((data[4*i+1] >> 2) << 5) | (data[4*i+2] >> 3);
               glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, size, level_height,
                              GL_RGB, GL_UNSIGNED_SHORT_5_6_5, packed);
               free(packed);
            } else if (mip_format == GL_RGBA16F) {
               float *values = malloc(size * level_height * 4 * sizeof(float));
               if (!values) fail("float mip allocation");
               for (unsigned i = 0; i < size * level_height * 4; ++i)
                  values[i] = data[i] / 255.f;
               glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, size, level_height,
                              GL_RGBA, GL_FLOAT, values);
               free(values);
            } else {
               glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, size, level_height,
                              GL_RGBA, GL_UNSIGNED_BYTE, data);
            }
            free(data);
         }
         if (generate) glGenerateMipmap(GL_TEXTURE_2D);
         if (getenv("T8132_TEXTURE_BASE"))
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, atoi(getenv("T8132_TEXTURE_BASE")));
         if (getenv("T8132_TEXTURE_LAST"))
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, atoi(getenv("T8132_TEXTURE_LAST")));
         if (getenv("T8132_TEXTURE_SAMPLER_BIAS"))
            glTexParameterf(GL_TEXTURE_2D, 0x8501 /* GL_TEXTURE_LOD_BIAS */,
                            atof(getenv("T8132_TEXTURE_SAMPLER_BIAS")) * (image ? -1 : 1));
         if (getenv("T8132_TEXTURE_ANISO"))
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, atof(getenv("T8132_TEXTURE_ANISO")));
         if (glGetError() != GL_NO_ERROR) fail("mipmap upload/generation");
      } else {
         if (texture_format && !strcmp(texture_format, "rgba16f")) {
            float data[64 * 64 * 4];
            for (unsigned y = 0; y < 64; ++y) for (unsigned x = 0; x < 64; ++x) {
               unsigned at = (y * 64 + x) * 4;
               data[at] = -1 + 2.0f*x/63 + image*.0625f;
               data[at+1] = .25f + 4.0f*y/63;
               data[at+2] = .00390625f + (x+y)/128.0f;
               data[at+3] = .5f + image*.125f;
            }
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 64, 64, 0, GL_RGBA, GL_FLOAT, data);
         } else if (texture_format && !strcmp(texture_format, "rgb565")) {
            uint16_t data[64 * 64];
            for (unsigned i = 0; i < 64*64; ++i)
               data[i] = ((pixels[4*i] >> 3) << 11) |
                         ((pixels[4*i+1] >> 2) << 5) | (pixels[4*i+2] >> 3);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB565, 64, 64, 0,
                         GL_RGB, GL_UNSIGNED_SHORT_5_6_5, data);
         } else {
            glTexImage2D(GL_TEXTURE_2D, 0,
                         texture_format && !strcmp(texture_format, "srgb") ? GL_SRGB8_ALPHA8 : GL_RGBA8,
                         64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
         }
      }
   }
   GLuint fetch_sampler = 0;
   if (getenv("T8132_TEXTURE_FETCH_HOSTILE_SAMPLER")) {
      glGenSamplers(1, &fetch_sampler);
      glSamplerParameteri(fetch_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
      glSamplerParameteri(fetch_sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glSamplerParameterf(fetch_sampler, GL_TEXTURE_MIN_LOD, 4);
      glSamplerParameterf(fetch_sampler, GL_TEXTURE_MAX_LOD, 4);
      glBindSampler(1, fetch_sampler);
      glBindSampler(3, fetch_sampler);
   }
   bool copy_test = getenv("T8132_TEXTURE_COPY") != NULL;
   if (copy_test && (!rtt || linear || blend || atlas_w || mip_size || mixed))
      fail("copy test requires plain nearest RTT");
   GLuint canvas = 0, canvas_fb = 0, copied = 0;
   if (rtt) {
      glActiveTexture(GL_TEXTURE2);
      glGenTextures(1, &canvas); glBindTexture(GL_TEXTURE_2D, canvas);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
      glGenFramebuffers(1, &canvas_fb); glBindFramebuffer(GL_FRAMEBUFFER, canvas_fb);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, canvas, 0);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
         fail("canvas framebuffer");
      if (copy_test) {
         glGenTextures(1, &copied); glBindTexture(GL_TEXTURE_2D, copied);
         glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      }
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
      if (cull_mode || depth_mode) {
         glDisable(GL_CULL_FACE);
         glDisable(GL_SCISSOR_TEST);
         if (depth_mode) {
            glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE); glDepthFunc(GL_ALWAYS);
            glDepthRangef(0, 1);
         }
         glUseProgram(solid); glDrawArrays(GL_TRIANGLES, 0, 6);
         if (cull_mode) {
            glEnable(GL_CULL_FACE);
            glCullFace(!strcmp(cull_mode,"front") ? GL_FRONT :
                       !strcmp(cull_mode,"both") ? GL_FRONT_AND_BACK : GL_BACK);
            glFrontFace(frame ? GL_CW : GL_CCW);
         }
         if (depth_mode) {
            GLenum func = !strcmp(depth_mode,"less") ? GL_LESS :
               !strcmp(depth_mode,"lequal") ? GL_LEQUAL :
               !strcmp(depth_mode,"greater") ? GL_GREATER :
               !strcmp(depth_mode,"gequal") ? GL_GEQUAL :
               !strcmp(depth_mode,"equal") ? GL_EQUAL :
               !strcmp(depth_mode,"notequal") ? GL_NOTEQUAL :
               !strcmp(depth_mode,"never") ? GL_NEVER : GL_ALWAYS;
            glDepthFunc(func);
            float depth = frame ? .75f : .5f;
            glDepthRangef(depth, depth);
         }
      }
      if ((getenv("T8132_TEXTURE_SAMPLE_MASK") || getenv("T8132_TEXTURE_DISCARD")) &&
          !depth_mode && !cull_mode && !blend) {
         glUseProgram(solid); glDrawArrays(GL_TRIANGLES, 0, 6);
      }
      if (blend) {
         glDisable(GL_BLEND); glDisable(GL_SCISSOR_TEST);
         glUseProgram(solid); glDrawArrays(GL_TRIANGLES, 0, 6);
         glEnable(GL_BLEND);
         glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
         if (blend_mode) {
            glBlendEquation(GL_FUNC_ADD);
            if (!strcmp(blend_mode, "color")) glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
            else if (!strcmp(blend_mode, "multiply")) glBlendFunc(GL_DST_COLOR, GL_ZERO);
            else if (!strcmp(blend_mode, "dstalpha")) glBlendFunc(GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
            else if (!strcmp(blend_mode, "constant")) {
               glBlendColor(frame ? .75f : .25f, .5f, .25f, .75f);
               glBlendFunc(GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR);
            } else if (!strcmp(blend_mode, "saturate")) glBlendFunc(GL_SRC_ALPHA_SATURATE, GL_ONE);
            else {
               glBlendFunc(GL_ONE, GL_ONE);
               glBlendEquation(!strcmp(blend_mode, "subtract") ? GL_FUNC_SUBTRACT :
                  !strcmp(blend_mode, "reverse") ? GL_FUNC_REVERSE_SUBTRACT :
                  !strcmp(blend_mode, "min") ? GL_MIN : GL_MAX);
            }
         }
      }
      glUseProgram(program);
      glUniform2f(glGetUniformLocation(program, "uv_scale"), wrap_test ? 3.25f : mixed ? 2.0f : 1.0f,
                  wrap_test ? 3.25f : mixed ? 2.0f : 1.0f);
      glUniform2f(glGetUniformLocation(program, "uv_offset"), wrap_test ? -1.3f : mixed ? -0.5f : 0.0f,
                  wrap_test ? -1.3f : mixed ? -0.5f : 0.0f);
      glUniform4f(glGetUniformLocation(program, "tint"), 1, 1, 1, 1);
      glUniform1f(glGetUniformLocation(program, "texture_bias"),
                  getenv("T8132_TEXTURE_BIAS") ? atof(getenv("T8132_TEXTURE_BIAS")) : 0);
      glUniform1i(image_location, atlas_w ? 3 : frame ? 1 : 3);
      glUniform1i(glGetUniformLocation(program, "other_image"), frame ? 3 : 1);
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
      } else if (stencil_test) {
         GLenum ops[] = {GL_KEEP, GL_ZERO, GL_REPLACE, GL_INCR, GL_DECR,
                         GL_INCR_WRAP, GL_DECR_WRAP, GL_INVERT};
         unsigned initial = stencil_compare ? (frame ? 0xa5 : 0x5a) : frame ? 255 : 0;
         unsigned values[] = {initial, 0, 166, frame ? 255 : 1,
                               frame ? 254 : 0, frame ? 0 : 1,
                               frame ? 254 : 255, 255-initial};
         unsigned mask = frame ? 0x5a : 0xff;
         glDisable(GL_SCISSOR_TEST);
         glStencilMask(0xff); glClearStencil(initial);
         glDepthMask(GL_TRUE); glClearDepthf(.5f);
         glClear(GL_STENCIL_BUFFER_BIT | (stencil_zfail ? GL_DEPTH_BUFFER_BIT : 0));
         glEnable(GL_STENCIL_TEST); glEnable(GL_SCISSOR_TEST);
         glFrontFace(frame ? GL_CW : GL_CCW);
         if (stencil_compare) {
            GLenum funcs[] = {GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL,
                              GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS};
            glDisable(GL_STENCIL_TEST); glDisable(GL_SCISSOR_TEST);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glUseProgram(solid); glDrawArrays(GL_TRIANGLES, 0, 6);
            glEnable(GL_STENCIL_TEST); glEnable(GL_SCISSOR_TEST);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
            for (unsigned func = 0; func < 8; ++func) {
               glScissor(func * width / 8, 0, width / 8, height);
               glStencilFunc(funcs[func], frame ? 0xb6 : 0xd9, frame ? 0xf0 : 0x0f);
               glUseProgram(program); glDrawArrays(GL_TRIANGLES, 0, 6);
            }
         } else for (unsigned phase = 0; phase < 2; ++phase) {
            if (phase == 1 && stencil_persist) glFinish();
            if (stencil_zfail && phase == 0) {
               glEnable(GL_DEPTH_TEST); glDepthFunc(GL_NEVER); glDepthMask(GL_FALSE);
            } else {
               glDisable(GL_DEPTH_TEST);
            }
            for (unsigned op = 0; op < 8; ++op) {
               glScissor(op * width / 8, 0, width / 8, height);
               if (phase == 0) {
                  glStencilMask(mask);
                  glStencilFunc(stencil_sfail ? GL_NEVER : GL_ALWAYS, 166, 0xff);
                  for (unsigned face = 0; face < 2; ++face) {
                     GLenum selected = face == frame ? ops[op] : GL_KEEP;
                     glStencilOpSeparate(face ? GL_BACK : GL_FRONT,
                        stencil_sfail ? selected : GL_KEEP,
                        stencil_zfail ? selected : GL_KEEP,
                        !stencil_sfail && !stencil_zfail ? selected : GL_KEEP);
                  }
                  glColorMask(stencil_discard_color, stencil_discard_color,
                              stencil_discard_color, stencil_discard_color);
                  glUseProgram(stencil_writer); glDrawArrays(GL_TRIANGLES, 0, 6);
               } else {
                  for (unsigned half = 0; half < (stencil_discard ? 2 : 1); ++half) {
                     unsigned expected = (values[op] & mask) | (initial & ~mask);
                     if (stencil_discard) {
                        glScissor(op * width / 8, half * height / 2, width / 8, height / 2);
                        if (!half) expected = initial;
                     }
                     glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                     glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
                     glStencilFunc(GL_EQUAL, expected, 0xff);
                     glUseProgram(program); glDrawArrays(GL_TRIANGLES, 0, 6);
                     /* A broken always-pass implementation must fail the test. */
                     glStencilFunc(GL_NOTEQUAL, expected, 0xff);
                     glUseProgram(solid); glDrawArrays(GL_TRIANGLES, 0, 6);
                  }
               }
            }
         }
         glDepthMask(GL_TRUE);
         glDisable(GL_STENCIL_TEST); glDisable(GL_SCISSOR_TEST);
         glFrontFace(GL_CCW);
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
         if (copy_test) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, copied);
            glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
         }
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
      if (copy_test) {
         uint8_t *read = malloc(width * height * 4);
         if (!read) fail("copy read allocation");
         glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, read);
         for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x) {
               unsigned tx = x * 64 / width, ty = y * 64 / height;
               uint8_t expected[] = {17+3*tx+7*frame, 23+3*ty+11*frame,
                                     31+tx+ty+13*frame, 64+tx+ty};
               if (memcmp(read + (y*width+x)*4, expected, 4))
                  fail("GPU copy/readback pixel mismatch");
            }
         free(read);
         printf("T8132_TEXTURE_COPY frame=%u pixels=%u exact=yes\n", frame, width*height);
      }
      if (glGetError() != GL_NO_ERROR) fail("texture draw");
      printf("T8132_TEXTURE_FRAME frame=%u\n", frame);
   }
   if (rtt) { glDeleteTextures(1, &canvas); glDeleteFramebuffers(1, &canvas_fb); }
   if (copied) glDeleteTextures(1, &copied);
   if (fetch_sampler) glDeleteSamplers(1, &fetch_sampler);
   glDeleteTextures(2, images); glDeleteTextures(1, &target);
   if (stencil_writer != solid) glDeleteProgram(stencil_writer);
   glDeleteProgram(solid);
   glDeleteFramebuffers(1, &fb); glDeleteVertexArrays(1, &vao); glDeleteProgram(program);
   if (stencil) glDeleteRenderbuffers(1, &stencil);
   return 0;
}
