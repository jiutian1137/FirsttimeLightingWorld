#version 330 core
uniform mat4 object_to_proj;

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 texcoord;
layout(location = 2) in vec4 normal;
layout(location = 3) in vec4 tangent;
layout(location = 4) in vec4 bitangent;
out float fdepth;

void main(){
	gl_Position = object_to_proj * vec4(position.xyz, 1.0);
	fdepth = gl_Position.z / gl_Position.w;
}