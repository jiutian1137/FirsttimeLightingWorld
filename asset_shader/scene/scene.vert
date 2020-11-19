#version 330 core
layout(location = 0) in vec4 position;
layout(location = 1) in vec4 texcoord;
layout(location = 2) in vec4 normal;
layout(location = 3) in vec4 tangent;
layout(location = 4) in vec4 bitangent;
uniform mat4 object_to_world;
uniform mat4 world_to_camera;
uniform mat4 camera_to_proj;

out vec3 fposition;
out vec2 ftexcoord;
out vec3 fnormal;
out vec3 ftangent;
out vec3 fbitangent;
out float fdepth;

void main(){
	gl_Position = camera_to_proj * world_to_camera * object_to_world * vec4(position.xyz, 1);
	fposition = (object_to_world * vec4(position.xyz, 1)).xyz;
	ftexcoord = texcoord.xy;

	mat3 normal_transform = transpose(inverse(mat3(object_to_world)));
	fnormal    = normalize(normal_transform * normal.xyz);
	ftangent   = normalize(normal_transform * tangent.xyz);
	fbitangent = normalize(normal_transform * bitangent.xyz);
	fdepth = gl_Position.z / gl_Position.w;
}
