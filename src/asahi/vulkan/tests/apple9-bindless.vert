#version 450
layout(location = 0) out vec2 uv;
void main() {
   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
   gl_Position = vec4(p, 0.5, 1.0);
   uv = p * 0.5 + 0.5;
}
