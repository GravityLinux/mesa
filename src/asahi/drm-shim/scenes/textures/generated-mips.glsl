#version 300 es
precision highp float;
precision highp int;
uniform sampler2D image;
uniform vec4 tint;
out vec4 color;
void main() {
    int lod = int(gl_FragCoord.x * 0.03125);
    ivec2 pos = min(ivec2(gl_FragCoord.xy), ivec2(127)) >> lod;
    color = texelFetch(image, pos, lod) * tint;
}
