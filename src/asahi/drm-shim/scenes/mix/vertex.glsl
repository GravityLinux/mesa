#version 300 es
precision highp float;
out vec2 uv;
void main() {
    int id = gl_VertexID;
    uv = vec2((id == 1 || id == 3 || id == 4) ? 1.0 : 0.0,
              (id == 2 || id == 4 || id == 5) ? 1.0 : 0.0);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
