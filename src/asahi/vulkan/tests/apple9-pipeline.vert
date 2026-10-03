#version 450
layout(push_constant) uniform Params { vec4 tint; vec4 transform; } pc;
#ifdef ATTRIBUTES
layout(location=0) in vec2 position;
#endif
layout(location=0) out vec3 color;
layout(location=1) flat out uint tag;
layout(location=2) noperspective out float linearValue;
void main() {
#ifdef ATTRIBUTES
    gl_Position = vec4(position, 0.25, 1.0);
#else
    vec2 positions[3] = vec2[](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0,0.5));
    gl_Position = vec4(positions[gl_VertexIndex % 3], 0.25, 1.0);
#endif
    gl_Position.xy = gl_Position.xy * pc.transform.xy + pc.transform.zw;
    color = vec3(0.2,0.7,0.4);
    tag = uint(gl_VertexIndex);
    linearValue = gl_Position.x;
}
