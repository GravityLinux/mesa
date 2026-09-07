#version 300 es
precision highp float;
precision highp int;
uniform vec2 extent, interval;
uniform int mode;
uniform uint shift16;
out vec4 color;
void main() {
    float index = floor(gl_FragCoord.x * 0.5) + floor(gl_FragCoord.y) * (extent.x * 0.5);
    float x = interval.x + index * interval.y;
    if (mode == 6)
        x = floor(index / 1024.0) * 1.5707963267948966 + (mod(index,1024.0)-512.0) * 0.00000095367431640625;
    float y;
    if ((uint(gl_FragCoord.x) & 1u) == 0u) y = sin(x);
    else y = cos(x);
    uint bits = floatBitsToUint(y);
    color = vec4(bits & 255u, (bits >> 8u) & 255u, (bits >> shift16) & 255u, bits >> 24u) / 255.0;
}
