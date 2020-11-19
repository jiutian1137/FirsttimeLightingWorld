#version 330 core
uniform mat4 local_to_world;
uniform mat4 world_to_proj;

layout(location = 0) in vec4 vin_position;
layout(location = 1) in vec4 vin_normal;
layout(location = 2) in vec4 vin_texcoord;
layout(location = 3) in vec4 vin_tangent;
layout(location = 4) in vec4 vin_bitangent;
out vec3 vout_position;
out vec3 vout_normal;
out vec2 vout_texcoord;
out vec3 vout_tangent;
out vec3 vout_bitangent;
out float vout_depth;

void main() {
	vec4 P      = local_to_world * vin_position;
	gl_Position = world_to_proj * P;
	vout_position  = P.xyz;
	vout_normal    = vin_normal.xyz;
	vout_texcoord  = vin_texcoord.xy;
	vout_tangent   = vin_tangent.xyz;
	vout_bitangent = vin_bitangent.xyz;
	vout_depth     = gl_Position.z / gl_Position.w;
}