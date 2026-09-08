#version 300 es
#extension GL_NV_shader_noperspective_interpolation : enable
precision highp float;
precision highp int;
uniform vec4 u_tint;
out vec4 frag_color;
flat in uint v0;
flat in uint v1;
flat in uint v2;
void main()
{
    vec3 c = vec3(v0 == 0x9abcdef0u ? .05 : v0 == 0xacf13568u ? .95 : 0.,
                  v1 == 0x87654321u ? .2 : v1 == 0x98765432u ? .8 : 0.,
                  v2 == 0xf00000a5u ? .1 : v2 == 0x122222c7u ? .7 : 0.);
    frag_color = vec4(c * u_tint.rgb, 1);
}
