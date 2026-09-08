#version 300 es
#extension GL_NV_shader_noperspective_interpolation : enable
precision highp float;
layout(location=0) in vec3 position;
layout(location=1) in vec4 vertex_color;
uniform mat4 u_transform;
flat out float v0;
flat out float v1;
flat out float v2;
void main() { gl_Position=(u_transform*vec4(position,1))*(1.+position.x*.5);
v0=vertex_color.x;v1=vertex_color.y;v2=position.x*position.y*.25+.4;}
