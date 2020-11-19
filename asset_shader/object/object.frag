#version 330 core
uniform vec3 albedo = vec3(0.7, 0.7, 0.7);
uniform vec3 fresnelF0 = vec3(0.1, 0.1, 0.1);

in vec3 vout_position;
in vec3 vout_normal;
in vec2 vout_texcoord;
in vec3 vout_tangent;
in vec3 vout_bitangent;
layout(location = 0) out vec4 fout_albedo;
layout(location = 1) out vec4 fout_normal;

void main(){
	fout_albedo = vec4(albedo,1);
	fout_normal = vec4(vout_normal, 1);
}