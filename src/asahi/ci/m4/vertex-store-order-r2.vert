#version 450
/* Stores only: reads no buffer, no vertex input. */
layout(set = 0, binding = 0) writeonly buffer B { uint flag; } b;
void main() {
   if (gl_VertexIndex == 0)
      b.flag = 2u;
   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   gl_Position = vec4(p * 0.1 - 1.0, 0.0, 1.0);
}
