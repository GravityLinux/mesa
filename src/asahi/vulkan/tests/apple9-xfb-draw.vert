#version 450
layout(location=0) flat out vec4 color;
void main() {
   int v = gl_VertexIndex % 3;
   vec2 p = vec2((v << 1) & 2, v & 2);
   gl_Position = vec4(p * 2.0 - 1.0, 0.5, 1.0);
   color = vec4(float(gl_InstanceIndex+1)*16.0/255.0,64.0/255.0,128.0/255.0,1.0);
}
