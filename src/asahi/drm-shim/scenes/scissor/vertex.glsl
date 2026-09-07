#version 300 es
precision highp float;
void main() {
    int id = gl_VertexID;
    vec2 p = vec2(id == 1 ? 3.0 : -1.0, id == 2 ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
