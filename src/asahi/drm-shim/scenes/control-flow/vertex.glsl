#version 300 es
precision highp float;
uniform int rounds;
out vec2 uv;
out float vertex_value;
void main() {
    int id = gl_VertexID;
    vec2 p = vec2(id == 1 ? 3.0 : -1.0, id == 2 ? 3.0 : -1.0);
    float value = 0.0;
    for (int i = 0; i < rounds + id; ++i) {
        if (i == 1) continue;
        value += 0.03125;
    }
    uv = (p + 1.0) * 0.5;
    vertex_value = value;
    gl_Position = vec4(p, 0.0, 1.0);
}
