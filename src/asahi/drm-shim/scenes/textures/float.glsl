#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
in vec2 uv;
out vec4 color;
void main() {
    vec4 value = texture(image, uv);
    color = vec4((value.r + 1.0) * 0.5, value.g * 0.125, value.ba) * tint;
}
