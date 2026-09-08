#version 300 es
#extension GL_NV_shader_noperspective_interpolation : enable
precision highp float;
precision highp int;
layout(location=0) in vec3 position;
layout(location=1) in vec4 vertex_color;
uniform mat4 u_transform;
flat out uint v0;
flat out uint v1;
flat out uint v2;
void main() { gl_Position=(u_transform*vec4(position,1))*(1.+position.x*.5);
v0=uint(position.x>0.)*0x12345678u+0x9abcdef0u;v1=uint(position.y>0.)*0x11111111u+0x87654321u;v2=uint(position.x*position.y>0.)*0x22222222u+0xf00000a5u;}
