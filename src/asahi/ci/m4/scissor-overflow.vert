#version 450
#extension GL_ARB_shader_viewport_layer_array : require
layout(push_constant) uniform Params { float depth; uint red; } pc;
void main()
{
    const vec2 positions[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(positions[gl_VertexIndex], pc.depth, 1);
    gl_ViewportIndex = gl_InstanceIndex;
}
