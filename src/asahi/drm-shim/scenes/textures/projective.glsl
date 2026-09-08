#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
in vec2 uv;
out vec4 color;
void main() {
    float q = 1.0 + uv.x * 2.0;
    color = textureProj(image, vec3(uv * q, q)) * tint;
}
