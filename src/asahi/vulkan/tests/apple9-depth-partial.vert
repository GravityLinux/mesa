#version 450
layout(push_constant) uniform PC { uint near_count; } pc;
layout(location = 0) flat out vec4 color;
void main() {
   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
   bool near = uint(gl_InstanceIndex) < pc.near_count;
   gl_Position = vec4(p, near ? 0.25 : 0.75, 1.0);
   color = near ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);
}
