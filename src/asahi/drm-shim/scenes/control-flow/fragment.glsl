#version 300 es
precision highp float;
precision highp int;
uniform int rounds;
in vec2 uv;
in float vertex_value;
out vec4 result;
void main() {
    int count = uv.x < 0.5 ? rounds : rounds + 2;
    float total = 0.0;
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < rounds; ++j) {
            if (j == 1) continue;
            if (uv.y < 0.5 && i > 1) break;
            total += 0.015625;
        }
    }
    result = vec4(total, vertex_value, uv.y * 0.25, 0.5);
}
