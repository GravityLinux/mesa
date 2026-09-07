#version 300 es
precision highp float;
uniform sampler2D image;
uniform vec4 tint;
const int loop_count = 7;
uniform bool forward_order;
in vec2 uv;
out vec4 color;
void main() {
    vec4 sum = vec4(0);
    for (int j = 0; j < loop_count; ++j) {
        for (int i = 0; i < loop_count; ++i) {
            vec2 offset = (vec2(float(i), float(j)) - float(loop_count-1)*0.5) / 64.0;
            if ((uv.x < 0.5) == forward_order)
                sum += texture(image, uv + offset);
            else
                sum += texture(image, uv.yx + offset);
        }
    }
    color = sum / float(loop_count*loop_count) * tint;
}
