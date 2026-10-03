#version 460
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 0, binding = 0) uniform sampler sampler_heap[];
layout(set = 1, binding = 0) uniform texture2D tex;
layout(push_constant) uniform push_data_t { uint sampler_index, pad0, pad1, pad2; } pd;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;
void main() { out_color = textureLod(sampler2D(tex, sampler_heap[pd.sampler_index]), uv, 0.0); }
