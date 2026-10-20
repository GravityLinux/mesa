#version 460
#extension GL_EXT_mesh_shader : require
#ifndef PRIMITIVE
#define PRIMITIVE 0
#endif
#if PRIMITIVE
layout(location=0) perprimitiveEXT in Data { vec4 values[3]; } data;
#else
layout(location=0) in Data { vec4 values[3]; } data;
#endif
layout(location=0) out vec4 color;
void main()
{
   color = (data.values[0] + data.values[1] + data.values[2]) / 3.;
}
