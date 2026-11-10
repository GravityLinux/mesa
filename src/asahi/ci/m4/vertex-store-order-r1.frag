#version 450
layout(set = 0, binding = 0) readonly buffer B { uint flag; } b;
layout(push_constant) uniform PC { uint iters; float seed; } pc;
layout(location = 0) out uint o;
void main() {
   float x = gl_FragCoord.x * pc.seed;
   for (uint i = 0; i < pc.iters; ++i)
      x = fract(x * 1.6180339 + 0.1234567);
   /* x is in [0,1): never >= 2, but the compiler cannot know. */
   o = b.flag + (x >= 2.0 ? 100u : 0u);
}
