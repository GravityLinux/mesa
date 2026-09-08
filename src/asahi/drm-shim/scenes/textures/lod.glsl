#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
out vec4 color;
void main() {
    float lod = floor(gl_FragCoord.x * 0.03125) + 0.5;
    color = textureLod(image, vec2(0.375, 0.625), lod) * tint;
}
