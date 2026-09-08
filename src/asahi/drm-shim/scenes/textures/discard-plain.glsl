#version 300 es
precision highp float;
uniform bool forward_order;
in vec2 uv;
out vec4 color;
void main() {
    if (uv.x < 0.5) discard;
    vec2 p = floor(uv * 64.0);
    float frame = forward_order ? 0.0 : 1.0;
    color = vec4(17.0 + 3.0*p.x + 7.0*frame,
                 23.0 + 3.0*p.y + 11.0*frame,
                 31.0 + p.x + p.y + 13.0*frame,
                 64.0 + p.x + p.y) / 255.0;
}
