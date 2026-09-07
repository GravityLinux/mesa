#version 300 es
precision highp float;
uniform vec2 extent;
uniform int mode;
in vec2 uv;
out vec4 color;
void main() {
    vec2 p = gl_FragCoord.xy;
    if (mode == 1)
        color = vec4(fract(p.x), fract(p.y), mod(floor(p.x) + 3.0 * floor(p.y), 17.0) / 16.0, 1.0);
    else
        color = vec4(p / extent, uv.y, 1.0);
}
