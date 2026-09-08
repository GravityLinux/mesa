#version 300 es
precision highp float;
uniform vec4 u_tint;
out vec4 frag_color;
in vec4 v0;
in vec4 v1;
in vec4 v2;
in vec4 v3;
void main() { vec3 c=vec3(0);
c += (v0.xyz*.75+v0.www*.25)/4.;
c += (v1.xyz*.75+v1.www*.25)/4.;
c += (v2.xyz*.75+v2.www*.25)/4.;
c += (v3.xyz*.75+v3.www*.25)/4.;
frag_color=vec4(c*u_tint.rgb,1); }
