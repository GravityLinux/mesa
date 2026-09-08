#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
in vec2 uv;
out vec4 color;
void main() {
    vec4 sample_color = texture(image, uv);
    if (uv.x < 0.5) discard;
    color = sample_color * tint;
}
