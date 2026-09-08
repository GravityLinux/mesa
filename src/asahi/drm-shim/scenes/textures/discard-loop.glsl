#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
uniform int loop_count;
in vec2 uv;
out vec4 color;
void main() {
    for (int i = 0; i < loop_count; ++i) {
        if (uv.y < 0.5) {
            if (uv.x < 0.25) discard;
        } else {
            if (uv.x < 0.5) discard;
        }
    }
    color = texture(image, uv) * tint;
}
