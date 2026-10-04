#version 450
layout(push_constant) uniform Params { float depth; uint red; } pc;
layout(location = 0) out vec4 color;
void main()
{
    color = pc.red != 0 ? vec4(1, 0, 0, 1) : vec4(0, 1, 0, 1);
}
