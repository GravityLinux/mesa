#version 300 es
precision highp float;
uniform vec2 extent;
uniform int mode;
uniform vec4 first, second;
out vec4 color;
void main() {
    float t = (gl_FragCoord.x - 0.5) / (extent.x - 1.0);
    if (mode == 0) color = mix(first, second, t);
    else if (mode == 1) color = mix(first, second, t * 3.0 - 1.0);
    else color = mix(first, second, vec4(t, 1.0-t, 0.5, t));
}
