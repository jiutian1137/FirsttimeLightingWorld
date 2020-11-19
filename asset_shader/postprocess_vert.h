#include "glsl.h"
out vec2 texcoord;

vec2 _VERTICES[4] = vec2[4](
	vec2(-1.0, -1.0),
	vec2(+1.0, -1.0),
	vec2(+1.0, +1.0),
	vec2(-1.0, +1.0)
);

void main() {
	gl_Position = vec4( _VERTICES[gl_VertexID], -1.0, 1.0 );
	texcoord = (_VERTICES[gl_VertexID] + 1) * 0.5;
}