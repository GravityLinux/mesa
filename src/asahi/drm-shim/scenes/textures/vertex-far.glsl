#version 300 es
precision highp float;
out vec2 uv;
uniform vec2 uv_scale, uv_offset;
void main() {
    int id = gl_VertexID;
    vec2 p = vec2((id == 1 || id == 4 || id == 5) ? 1.0 : -1.0,
                  (id == 2 || id == 3 || id == 5) ? 1.0 : -1.0);
    uv = (p * 0.5 + 0.5) * uv_scale + uv_offset;
    gl_Position = vec4(p * 16384.0, 0.0, 16384.0);
}
