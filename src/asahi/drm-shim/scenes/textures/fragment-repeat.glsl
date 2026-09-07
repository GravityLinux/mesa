#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
in vec2 uv;
out vec4 color;
void main() {
    color = (texture(image, uv) + texture(image, uv.yx)) * 0.5 * tint;
}
