#version 450
layout(push_constant) uniform PC { uint seed; } pc;
layout(location = 0) in vec2 uv;
layout(location = 1) flat in uint instance;
layout(location = 0) out vec4 out_color0;
layout(location = 1) out vec4 out_color1;
layout(location = 2) out vec4 out_color2;
vec4 pat(uint seed) {
   uvec2 p = uvec2(gl_FragCoord.xy);
   uint h = (p.x * 73856093u) ^ (p.y * 19349663u) ^ (seed * 83492791u);
   h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
   return vec4(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u), 255.0) / 255.0;
}
void main() {
   uint seed = pc.seed * 65536u + instance;
   out_color0 = pat(seed);
   out_color1 = pat(seed + 1000000u);
   out_color2 = pat(seed + 2000000u);
}
