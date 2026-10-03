#version 450
layout(push_constant) uniform Params { vec4 tint; vec4 transform; } pc;
layout(location=0) in vec3 color;
layout(location=1) flat in uint tag;
layout(location=2) noperspective in float linearValue;
layout(location=0) out vec4 outColor;
void main() {
    outColor = vec4(color * pc.tint.rgb + vec3(float(tag)*0.01, linearValue*0.01,0), 0.75);
}
