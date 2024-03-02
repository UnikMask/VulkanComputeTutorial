// vim:ft=glsl
#version 460

layout (location = 0) in vec3 fragColor;
layout (location = 0) out vec4 outColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (0.5 - length(coord) <= 0) {
        outColor  = vec4(0);
    } else {
        outColor = vec4(fragColor, 1);
    }
}
