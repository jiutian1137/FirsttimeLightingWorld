#version 330 core

vec2 QUAD[4] = vec2[4](
	vec2(-1.0, -1.0),
	vec2(+1.0, -1.0),
	vec2(-1.0, +1.0),
	vec2(+1.0, +1.0)
);

void main() {
	gl_Position = vec4(QUAD[gl_VertexID], -1.0, 1.0);
}