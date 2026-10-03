#version 450
layout(push_constant) uniform PC { uint seed; uint partial_count; } pc;
layout(location = 0) out vec2 uv;
layout(location = 1) flat out uint instance;
void main() {
   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
   /* The first partial_count instances leave the right side of the screen empty. */
   if (uint(gl_InstanceIndex) < pc.partial_count)
      p.x = (p.x + 1.0) * 0.5 - 1.0;
   gl_Position = vec4(p, 0.5, 1.0);
   uv = p * 0.5 + 0.5;
   instance = uint(gl_InstanceIndex);
}
