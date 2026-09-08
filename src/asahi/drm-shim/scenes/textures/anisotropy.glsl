#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
out vec4 color;
void main() {
    float major = exp2(floor(gl_FragCoord.x * 0.03125) + 0.5);
    color = textureGrad(image, vec2(0.375, 0.625),
                        vec2(major / 128.0, 0.0),
                        vec2(0.0, 1.0 / 128.0)) * tint;
}
