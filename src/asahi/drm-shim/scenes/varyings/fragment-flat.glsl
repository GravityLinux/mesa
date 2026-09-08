#version 300 es
#extension GL_NV_shader_noperspective_interpolation : enable
precision highp float;
uniform vec4 u_tint;
out vec4 frag_color;
flat in float v0;
flat in float v1;
flat in float v2;
void main(){frag_color=vec4(vec3(v0,v1,v2)*u_tint.rgb,1); }
