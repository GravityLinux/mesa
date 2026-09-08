#version 300 es
precision highp float;
precision highp int;
uniform sampler2D image;
uniform vec4 tint;
uniform int loop_count;
in vec2 uv;
out vec4 color;
void main() {
    color = texelFetch(image, ivec2(uv * 64.0), loop_count - 1) * tint;
}
