#version 300 es
#extension GL_NV_shader_noperspective_interpolation : enable
precision highp float;
uniform vec4 u_tint;
out vec4 frag_color;
noperspective in float v0;
noperspective in float v1;
noperspective in float v2;
void main(){frag_color=vec4(vec3(v0,v1,v2)*u_tint.rgb,1); }
