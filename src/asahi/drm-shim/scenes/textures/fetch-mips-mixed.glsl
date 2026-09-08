#version 300 es
precision highp float;
precision highp int;
uniform sampler2D image, other_image;
out vec4 color;
void main() {
    int lod = int(gl_FragCoord.x * 0.03125);
    vec4 a = texelFetch(image, ivec2(0), lod);
    vec4 b = texelFetch(other_image, ivec2(0), lod);
    color = ((lod & 1) == 0 ? a : b)
          + (texture(image, vec2(0.5)) + texture(other_image, vec2(0.5))) * 0.00001;
}
