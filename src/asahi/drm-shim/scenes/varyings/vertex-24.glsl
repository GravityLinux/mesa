#version 300 es
precision highp float;
layout(location=0) in vec3 position;
uniform mat4 u_transform;
out vec4 v0;
out vec4 v1;
out vec4 v2;
out vec4 v3;
out vec4 v4;
out vec4 v5;
void main() { gl_Position=(u_transform*vec4(position,1))*(1.+position.x*.5);
v0=vec4(position.x,position.y,position.x*position.y,position.x*position.x+position.y*position.y+.3*position.x+.2*position.y)*0.031250000+vec4(.3,.4,.5,.6);
v1=vec4(position.x,position.y,position.x*position.y,position.x*position.x+position.y*position.y+.3*position.x+.2*position.y)*0.062500000+vec4(.3,.4,.5,.6);
v2=vec4(position.x,position.y,position.x*position.y,position.x*position.x+position.y*position.y+.3*position.x+.2*position.y)*0.093750000+vec4(.3,.4,.5,.6);
v3=vec4(position.x,position.y,position.x*position.y,position.x*position.x+position.y*position.y+.3*position.x+.2*position.y)*0.125000000+vec4(.3,.4,.5,.6);
v4=vec4(position.x,position.y,position.x*position.y,position.x*position.x+position.y*position.y+.3*position.x+.2*position.y)*0.156250000+vec4(.3,.4,.5,.6);
v5=vec4(position.x,position.y,position.x*position.y,position.x*position.x+position.y*position.y+.3*position.x+.2*position.y)*0.187500000+vec4(.3,.4,.5,.6);
}
