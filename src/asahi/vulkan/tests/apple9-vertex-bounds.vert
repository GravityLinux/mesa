#version 450
layout(location=0) in vec4 vertex_value;
layout(location=0) flat out vec4 value;
void main() {
   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   gl_Position = vec4(p * 2.0 - 1.0, 0.5, 1.0);
   value = vec4(vertex_value.rgb, 1.0);
}
