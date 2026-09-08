#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
uniform vec2 uv_scale;
in vec2 uv;
out vec4 color;
void main() {
    color = textureGrad(image, uv, vec2(uv_scale.x / 256.0, 0.0),
                                  vec2(0.0, uv_scale.y / 256.0)) * tint;
}
