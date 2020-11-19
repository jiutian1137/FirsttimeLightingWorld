#pragma once
#include "../calculation/lapack/vector.h"
#include "../calculation/lapack/matrix.h"
#include "../openglutil/glut.h"
#include <memory>
#include <vector>
#include <cstdint>
#include <filesystem>
#ifndef _CLMAGIC_BEGIN
#define _CLMAGIC_BEGIN namespace clmagic{ 
#define _CLMAGIC_END }
#endif

template<typename _Ty>
class tree {
public:
	using value_type = _Ty;

	_Ty value;
	std::vector<tree> children;
	tree* parent_ptr = nullptr;

	void push_back(const tree<_Ty>& _Node) {
		children.push_back(_Node);
		children.back().parent_ptr = this;
	}
	void push_back(const _Ty& _Val) {
		children.push_back(tree{ _Val });
		children.back().parent_ptr = this;
	}
	void pop_back() {
		children.pop_back();
	}
	void clear() {
		children.clear();
	}

	bool has_parent() const {
		return parent_ptr != nullptr;
	}
	bool has_child() const {
		return !children.empty();
	}
	tree& parent() {
		assert( has_parent() );
		return *parent_ptr;
	}

	template<typename _Fn>
	void traversal(_Fn _Func) {
		if ( _Func(*this) && this->has_child() ) {
			std::for_each(children.begin(), children.end(), 
				[_Func](tree& _Node) { _Node.traversal(_Func); });
		}
	}

	template<typename _Fn>
	void rtraversal(_Fn _Func) {
		if ( _Func(*this) && this->has_parent() ) {
			this->parent().rtraversal(_Func);
		}
	}
};

namespace geometry {
	using clmagic::vector3;
	using clmagic::matrix4x4;

	template<typename _Ty1, typename _Ty2 = uint32_t>
	struct vertex_set {
		using vertex_type = _Ty1;
		using index_type  = _Ty2;

		void push_back(const vertex_type& _Vertex) {
			// 1. Get the index to i
			index_type i = static_cast<index_type>(vertex_data.size());
			// 2. Add a vertex
			vertex_data.push_back(_Vertex);
			// 3. Add a index using i
			index_data.push_back(i);
		}
		void pop_back() {
			// 1. Copy the index to i
			index_type i = index_data.back();
			// 2. Remove the index
			index_data.pop_back();
			// 3. Remove vertex[i], if vertex[i] is unique
			if (std::find(index_data.begin(), index_data.end(), i) == index_data.end()) {
				vertex_data.erase(vertex_data.begin + i);
			}
		}
		size_t size() const {
			return index_data.size();
		}
		vertex_type& at(size_t _Pos) {
			return vertex_data[index_data[_Pos]];
		}
		const vertex_type& at(size_t _Pos) const {
			return vertex_data[index_data[_Pos]];
		}
		void resize(size_t _Newsize, vertex_type _Val) {
			size_t _Diff = _Newsize - index_data.size();
			// 1. record _Oldsize and allocate new memory
			size_t _Old_vertex_size = vertex_data.size();
			vertex_data.resize(_Old_vertex_size + _Diff, _Val);
			// 2. record _Oldsize and allocate new memory
			size_t _Old_index_size = index_data.size();
			index_data.resize(_Old_index_size + _Diff);
			// 3.
			size_t     i = _Old_index_size;
			index_type v = static_cast<index_type>(_Old_vertex_size);
			for (; i != index_data.size(); ++i, ++v) {
				index_data[i] = v;
			}
		}
		void resize(size_t _Newsize) {
			this->resize(_Newsize, vertex_type{});
		}

		std::vector<vertex_type> vertex_data;
		std::vector<index_type>  index_data;
	};


	class unknown_graphic {
	public:
		//implicit_interface setup_render_target(...) const;
		virtual void setup_processor(void*) const { abort(); }
		virtual void setup_uniforms(void*) const { abort(); }
		virtual void setup_varyings(void*) const { abort(); }
		virtual void process(void*) const { abort(); }

		virtual void render(void* userdata) const {
			// setup default render target
			setup_processor(userdata);
			setup_uniforms(userdata);
			setup_varyings(userdata);
			process(userdata);
		}
	};

	template<typename _Ty>
	class opengl_vertex_geometry : public unknown_graphic {
	public:
		// processor
		std::shared_ptr< GLprogram > processor = nullptr;
		GLint local_to_world_location = -1;
		GLint emission_location  = -1;
		GLint albedo_location    = -1;
		GLint fresnelF0_location = -1;
		GLint roughness_location = -1;
		// uniforms
		matrix4x4<float> local_to_world;
		vector3<float> emission  = { 0.0f, 0.0f, 0.0f };
		vector3<float> albedo    = { 0.7f, 0.7f, 0.7f };
		vector3<float> fresnelF0 = { 0.1f, 0.1f, 0.1f };
		float          roughness = 0.4f;
		std::vector<std::shared_ptr<GLtexture>> textures;
		// varyings
		std::shared_ptr< GLvaryings > vertex_desc = nullptr;
		std::shared_ptr< GLbufferx<_Ty> > vertex_data = nullptr;
		// process mode
		GLenum mode = GL_POINTS;

		virtual void setup_processor(void*) const override {
			glUseProgram(*processor);
		}

		virtual void setup_uniforms(void*) const override {
			glUniformMatrix4fv(local_to_world_location, 1, true, local_to_world.data());
			glUniform3fv(emission_location, 1, emission.data());
			glUniform3fv(albedo_location,   1, albedo.data());
			glUniform3fv(fresnelF0_location, 1, fresnelF0.data());
			glUniform1f(roughness_location, roughness);
			
			auto _First = textures.begin();
			auto _Last  = textures.end();
			GLenum _Dest = GL_TEXTURE0;
			for ( ; _First != _Last; ++_First, ++_Dest) {
				glActiveTexture(_Dest);
				glBindTexture((*_First)->target(), (*(*_First)));
			}
		}

		virtual void setup_varyings(void*) const override {
			glBindVertexArray(*vertex_desc);
		}

		virtual void process(void*) const override {
			glDrawArrays(mode, 0, vertex_data->size());
		}
	};

	template<typename _Ty1, typename _Ty2 = uint32_t>
	class opengl_index_geometry : public opengl_vertex_geometry<_Ty1> {
		using _Mybase = opengl_vertex_geometry<_Ty1>;
	public:
		std::shared_ptr< GLbufferx<_Ty2> > index_data;

		virtual void setup_varyings(void*) const override {
			glBindVertexArray(*_Mybase::vertex_desc);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *index_data);
		}

		virtual void process(void*) const override {
			glDrawElements(_Mybase::mode, index_data->size(), glGetDataType(static_cast<_Ty2*>(nullptr)), nullptr);
		}
	};


	struct disney_brdf_material {
		using spectrum = vector3<float>;
		using rate = float;

		spectrum base_color = { 0.7f, 0.7f, 0.7f };
		rate metallic       = 0.0f;
		spectrum emission   = { 0.0f, 0.0f, 0.0f };
		rate subsurface     = 0.0f;
		rate specular       = 0.5f;
		rate roughness      = 0.5f;
		rate specular_tint  = 0.0f;
		rate anisotropic    = 0.0f;
		rate sheen          = 0.0f;
		rate sheen_tint     = 0.5f;
		rate clearcoat      = 0.0f;
		rate clearcoat_gloss= 1.0f;
		bool use_base_color_texture     = false;
		bool use_metallic_texture       = false;
		bool use_emission_texture       = false;
		bool use_subsurface_texture     = false;
		bool use_specular_texture       = false;
		bool use_roughness_texture      = false;
		bool use_specular_tint_texture  = false;
		bool use_anisotropic_texture    = false;
		bool use_sheen_texture          = false;
		bool use_sheen_tint_texture     = false;
		bool use_clearcoat_texture      = false;
		bool use_clearcoat_gloss_texture= false;
		std::shared_ptr<GLtexture2d> base_color_texture     = nullptr;
		std::shared_ptr<GLtexture2d> metallic_texture       = nullptr;
		std::shared_ptr<GLtexture2d> emission_texture       = nullptr;
		std::shared_ptr<GLtexture2d> subsurface_texture     = nullptr;
		std::shared_ptr<GLtexture2d> specular_texture       = nullptr;
		std::shared_ptr<GLtexture2d> roughness_texture      = nullptr;
		std::shared_ptr<GLtexture2d> specular_tint_texture  = nullptr;
		std::shared_ptr<GLtexture2d> anisotropic_texture    = nullptr;
		std::shared_ptr<GLtexture2d> sheen_texture          = nullptr;
		std::shared_ptr<GLtexture2d> sheen_tint_texture     = nullptr;
		std::shared_ptr<GLtexture2d> clearcoat_texture      = nullptr;
		std::shared_ptr<GLtexture2d> clearcoat_gloss_texture= nullptr;

		bool use_normal_texture = false;
		std::shared_ptr<GLtexture2d> normal_texture         = nullptr;
		std::shared_ptr<GLtexture2d> occlusion_texture      = nullptr;
	};

	template<typename _Ty>
	class opengl_vertex_geometry_disneybrdf : public unknown_graphic {
	public:
		// processor
		std::shared_ptr< GLprogram > processor = nullptr;
		GLint local_to_world_location = -1;
		GLint base_color_location     = -1;
		GLint metallic_location       = -1;
		GLint emission_location       = -1;
		GLint subsurface_location     = -1;
		GLint specular_location       = -1;
		GLint roughness_location      = -1;
		GLint specular_tint_location  = -1;
		GLint anisotropic_location    = -1;
		GLint sheen_location          = -1;
		GLint sheen_tint_location     = -1;
		GLint clearcoat_location      = -1;
		GLint clearcoat_gloss_location= -1;
		GLint use_base_color_texture_location = -1;
		GLint use_metallic_texture_location   = -1;
		GLint use_emission_texture_location   = -1;
		GLint use_subsurface_texture_location = -1;
		GLint use_specular_texture_location   = -1;
		GLint use_roughness_texture_location  = -1;
		GLint use_specular_tint_texture_location = -1;
		GLint use_anisotropic_texture_location = -1;
		GLint use_sheen_texture_location      = -1;
		GLint use_sheen_tint_texture_location = -1;
		GLint use_clearcoat_texture_location  = -1;
		GLint use_clearcoat_gloss_texture_location = -1;
		GLint use_normal_texture_location     = -1;
		GLenum base_color_texture_unit     = GL_TEXTURE0;
		GLenum metallic_texture_unit       = GL_TEXTURE1;
		GLenum emission_texture_unit       = GL_TEXTURE2;
		GLenum subsurface_texture_unit     = GL_TEXTURE3;
		GLenum specular_texture_unit       = GL_TEXTURE4;
		GLenum roughness_texture_unit      = GL_TEXTURE5;
		GLenum specular_tint_texture_unit  = GL_TEXTURE6;
		GLenum anisotropic_texture_unit    = GL_TEXTURE7;
		GLenum sheen_texture_unit          = GL_TEXTURE8;
		GLenum sheen_tint_texture_unit     = GL_TEXTURE9;
		GLenum clearcoat_texture_unit      = GL_TEXTURE10;
		GLenum clearcoat_gloss_texture_unit= GL_TEXTURE11;
		GLenum normal_texture_unit         = GL_TEXTURE12;
		GLenum occlusion_texture_unit      = GL_TEXTURE13;
		// uniforms
		std::shared_ptr< matrix4x4<float> > local_to_world;
		std::shared_ptr< disney_brdf_material > material;
		// varyings
		std::shared_ptr< GLvaryings > vertex_desc = nullptr;
		std::shared_ptr< GLbufferx<_Ty> > vertex_data = nullptr;
		// process mode
		GLenum mode = GL_POINTS;

		virtual void setup_processor(void*) const override {
			glUseProgram(*processor);
		}

		virtual void setup_uniforms(void*) const override {
			// set vertex uniform
			glUniformMatrix4fv(local_to_world_location, 1, true, local_to_world->data());
			
			// set pixel uniform
			if ( material->use_base_color_texture ) {
				glUniform1i(use_base_color_texture_location, true);
				glActiveTexture(base_color_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->base_color_texture->identifier);
			} else { 
				glUniform1i(use_base_color_texture_location, false);
				glUniform3fv(base_color_location, 1, material->base_color.data()); 
			}

			if (material->use_metallic_texture) {
				glUniform1i(use_metallic_texture_location, true);
				glActiveTexture(metallic_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->metallic_texture->identifier);
			} else {
				glUniform1i(use_metallic_texture_location, false);
				glUniform1f(metallic_location, material->metallic);
			}

			if (material->use_emission_texture) {
				glUniform1i(use_emission_texture_location, true);
				glActiveTexture(emission_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->emission_texture->identifier);
			} else {
				glUniform1i(use_emission_texture_location, false);
				glUniform1f(metallic_location, material->metallic);
			}

			if (material->use_subsurface_texture) {
				glUniform1i(use_subsurface_texture_location, true);
				glActiveTexture(subsurface_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->subsurface_texture->identifier);
			} else {
				glUniform1i(use_subsurface_texture_location, false);
				glUniform1f(subsurface_location, material->subsurface);
			}

			if (material->use_specular_texture) {
				glUniform1i(use_specular_texture_location, true);
				glActiveTexture(specular_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->specular_texture->identifier);
			} else {
				glUniform1i(use_specular_texture_location, false);
				glUniform1f(specular_location, material->specular);
			}

			if (material->use_roughness_texture) {
				glUniform1i(use_roughness_texture_location, true);
				glActiveTexture(roughness_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->roughness_texture->identifier);
			} else {
				glUniform1i(use_roughness_texture_location, false);
				glUniform1f(roughness_location, material->roughness);
			}

			if (material->use_specular_tint_texture) {
				glUniform1i(use_specular_tint_texture_location, true);
				glActiveTexture(specular_tint_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->specular_tint_texture->identifier);
			} else {
				glUniform1i(use_specular_tint_texture_location, false);
				glUniform1f(specular_tint_location, material->specular_tint);
			}

			if (material->use_anisotropic_texture) {
				glUniform1i(use_anisotropic_texture_location, true);
				glActiveTexture(anisotropic_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->anisotropic_texture->identifier);
			} else {
				glUniform1i(use_anisotropic_texture_location, false);
				glUniform1f(anisotropic_location, material->anisotropic);
			}

			if (material->use_sheen_texture) {
				glUniform1i(use_sheen_texture_location, true);
				glActiveTexture(sheen_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->sheen_texture->identifier);
			} else {
				glUniform1i(use_sheen_texture_location, false);
				glUniform1f(sheen_location, material->sheen);
			}

			if (material->use_sheen_tint_texture) {
				glUniform1i(use_sheen_tint_texture_location, true);
				glActiveTexture(sheen_tint_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->sheen_tint_texture->identifier);
			} else {
				glUniform1i(use_sheen_tint_texture_location, false);
				glUniform1f(sheen_tint_location, material->sheen_tint);
			}

			if (material->use_clearcoat_texture) {
				glUniform1i(use_clearcoat_texture_location, true);
				glActiveTexture(clearcoat_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->clearcoat_texture->identifier);
			} else {
				glUniform1i(use_clearcoat_texture_location, false);
				glUniform1f(clearcoat_location, material->clearcoat);
			}

			if (material->use_clearcoat_gloss_texture) {
				glUniform1i(use_clearcoat_gloss_texture_location, true);
				glActiveTexture(clearcoat_gloss_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->clearcoat_gloss_texture->identifier);
			} else {
				glUniform1i(use_clearcoat_gloss_texture_location, false);
				glUniform1f(clearcoat_gloss_location, material->clearcoat_gloss);
			}

			if (material->use_normal_texture) {
				glUniform1i(use_normal_texture_location, true);
				glActiveTexture(normal_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->normal_texture->identifier);
				glActiveTexture(occlusion_texture_unit);
				glBindTexture(GL_TEXTURE_2D, material->occlusion_texture->identifier);
			} else {
				glUniform1i(use_normal_texture_location, false);
			}
		}

		virtual void setup_varyings(void*) const override {
			glBindVertexArray(*vertex_desc);
		}

		virtual void process(void*) const override {
			glDrawArrays(mode, 0, vertex_data->size());
		}
	};
	
	template<typename _Ty1, typename _Ty2 = uint32_t>
	class opengl_index_geometry_disneybrdf : public opengl_vertex_geometry_disneybrdf<_Ty1> {
		using _Mybase = opengl_vertex_geometry_disneybrdf<_Ty1>;
	public:
		std::shared_ptr< GLbufferx<_Ty2> > index_data;

		virtual void setup_varyings(void*) const override {
			glBindVertexArray(*_Mybase::vertex_desc);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *index_data);
		}

		virtual void process(void*) const override {
			glDrawElements(_Mybase::mode, index_data->size(), glGetDataType(static_cast<_Ty2*>(nullptr)), nullptr);
		}
	};


	struct sky_cloud_userdata {
		float time;
		matrix4x4<float> proj_to_world;
	};

	class opengl_physical_sky_cloud : public unknown_graphic {
		// heightgradient_texture: { image:{ axisX: cloud_type, axisY: height_ratio }, data:{ R: height_gradient } }
		// weather_texture: { image:{ axisX: longitude_ratio, axisY: latitude_ratio }, data:{ R:coverage, G:type, B:Rain } }
		// base_cloud_texture: { image:{ axisXYZ: sample_position }, data:{ R: density } }
		// cloud_detail_texture { image:{ axisXYZ: sample_position }, data:{ R: density } }
		// cirrus_cloud_texture { image:{ axisX: longitude_ratio, axisY: latitude_ratio }, data:{ R: density } }
	public:
		// processor
		GLuint processor;
		
		GLuniformSampler2D atmosphere_transmittance_texture = { GL_TEXTURE0 };
		
		GLuniformSampler3D atmosphere_scattering_texture = { GL_TEXTURE1 };
		
		GLuniformSampler2D sky_irradiance_texture = { GL_TEXTURE2 };

		Euler::Angles<float> sun_direction = { 0.0f, 0.0f, 0.0f };

		mutable GLuniformVec3 sun_vector = { { 0.0f, 1.0f, 0.0f } };

		GLuniformVec2 sun_size = { 0.0f, 0.0f };


		GLuniformSampler2D cloud_height_texture = { GL_TEXTURE3 };

		GLuniformSampler2D cloud_weather_texture = { GL_TEXTURE4 };
		GLuniformFloat     cloud_weather_texture_scale = { 100.0f };

		GLuniformSampler3D base_cloud_texture   = { GL_TEXTURE5 };
		GLuniformFloat     base_cloud_texture_scale = { 0.00016f };

		GLuniformSampler3D cloud_detail_texture = { GL_TEXTURE6 };
		GLuniformFloat     cloud_detail_texture_scale = { 0.0032f };

		GLuniformSampler2D cloud_curl_texture = { GL_TEXTURE7 };

		GLuniformSampler2D cirrus_cloud_texture = { GL_TEXTURE8 };

		GLuniformFloat cloud_density_scale       = { 1.0f };
		GLuniformFloat cloud_forward_scattering  = { 0.80f };
		GLuniformFloat cloud_backward_scattering = { 0.45f };
		GLuniformFloat cloud_transmittance_lowbound = { 0.04f };

		GLuniformVec3  wind_direction = { { 0.77f, 0.0f, 0.77f } };
		GLuniformFloat wind_intensity = { 100.0f };


		GLuniformSampler2D scene_brdf_texture   = { GL_TEXTURE9 };
		GLuniformSampler2D scene_normal_texture = { GL_TEXTURE10 };
		GLuniformSampler2D scene_depth_texture  = { GL_TEXTURE11 };
		GLuniformSampler2D scene_shadow_texture = { GL_TEXTURE12 };

		GLuniformMat4  proj_to_world;
		GLuniformMat4  world_to_Lproj;
		GLuniformFloat time = { 0.0f };

#ifdef _DEBUG
		bool debug_processor = false;
		bool debug_uniform_locations = false;

		bool debug_atmosphere_transmittance_texture = false;
		bool debug_atmosphere_scattering_texture = false;
		bool debug_sky_irradiance_texture = false;
		bool debug_sun_property = false;

		bool debug_cloud_height_texture = false;
		bool debug_cloud_weather_texture = false;
		bool debug_base_cloud_texture = false;
		bool debug_cloud_detail_texture = false;
		bool debug_cloud_curl_texture = false;
		bool debug_cirrus_cloud_texture = false;
		bool debug_cloud_property = false;

		bool debug_wind_property = false;

		bool debug_scene_brdf_texture = false;
		bool debug_scene_normal_texture = false;
		bool debug_scene_depth_texture = false;
		bool debug_scene_shadow_texture = false;
#endif

		virtual void setup_processor(void*) const override { 
			glUseProgram(processor);
		}

		virtual void setup_uniforms(void* userdata) const override { 
			// atmosphere and sun
			glSetUniformSampler2D(atmosphere_transmittance_texture);
			glSetUniformSampler3D(atmosphere_scattering_texture);
			glSetUniformSampler2D(sky_irradiance_texture);

			sun_vector.value[0] = cos(sun_direction.pitch) * cos(sun_direction.yaw);
			sun_vector.value[1] = sin(sun_direction.pitch);
			sun_vector.value[2] = cos(sun_direction.pitch) * sin(sun_direction.yaw);
			glSetUniformVec3(sun_vector);
			glSetUniformVec2(sun_size);

			// cloud
			glSetUniformSampler2D(cloud_height_texture);

			glSetUniformSampler2D(cloud_weather_texture);
			glSetUniformFloat(cloud_weather_texture_scale);
			
			glSetUniformSampler3D(base_cloud_texture);
			glSetUniformFloat(base_cloud_texture_scale);
			
			glSetUniformSampler3D(cloud_detail_texture);
			glSetUniformFloat(cloud_detail_texture_scale);
			
			glSetUniformSampler2D(cloud_curl_texture);
			glSetUniformSampler2D(cirrus_cloud_texture);

			glSetUniformFloat(cloud_density_scale);
			glSetUniformFloat(cloud_forward_scattering);
			glSetUniformFloat(cloud_backward_scattering);
			glSetUniformFloat(cloud_transmittance_lowbound);

			glSetUniformVec3(wind_direction);
			glSetUniformFloat(wind_intensity);

			// scene
			glSetUniformSampler2D(scene_brdf_texture);
			glSetUniformSampler2D(scene_normal_texture);
			glSetUniformSampler2D(scene_depth_texture);
			glSetUniformSampler2D(scene_shadow_texture);

			// userdata
			glSetUniformMat4(proj_to_world);
			glSetUniformMat4(world_to_Lproj);
			glSetUniformFloat(time);
		}

		virtual void setup_varyings(void*) const override {
			// empty
		}

		virtual void process(void*) const override { 
#ifdef _DEBUG
			assert(debug_processor && 
				debug_atmosphere_transmittance_texture &&
				debug_atmosphere_scattering_texture &&
				debug_sky_irradiance_texture &&
				debug_sun_property &&
				debug_cloud_height_texture &&
				debug_cloud_weather_texture &&
				debug_base_cloud_texture &&
				debug_cloud_detail_texture &&
				debug_cloud_curl_texture &&
				debug_cirrus_cloud_texture &&
				debug_cloud_property &&
				debug_wind_property &&
				debug_scene_brdf_texture &&
				debug_scene_normal_texture &&
				debug_scene_depth_texture &&
				debug_scene_shadow_texture);
#endif // _DEBUG
			glDrawNodata(GL_QUADS, 4);
		}

		// <safe function>
		void set_processor(GLuint prog) {
			assert(glIsProgram(prog));
			processor = std::move(prog);
#ifdef _DEBUG
			debug_processor = true;
#endif // _DEBUG
		}

		void set_uniform_locations(
			GLint atmosphere_transmittance_texture_location, 
			GLint atmosphere_scattering_texture_location, 
			GLint sky_irradiance_texture_location,
			GLint sun_vector_location, GLint sun_size_location,

			GLint cloud_height_texture_location, 
			GLint cloud_weather_texture_location, GLint cloud_weather_texture_scale_location,
			GLint base_cloud_texture_location, GLint base_cloud_texture_scale_location,
			GLint cloud_detail_texture_location, GLint cloud_detail_texture_scale_location,
			GLint cloud_curl_texture_location, 
			GLint cirrus_cloud_texture_location,
			GLint cloud_density_scale_location, GLint cloud_forward_scattering_location, GLint cloud_backward_scattering_location, GLint cloud_transmittance_lowbound_location,
			GLint wind_direction_location, GLint wind_intensity_location,

			GLint scene_brdf_texture_location, 
			GLint scene_normal_texture_location, 
			GLint scene_depth_texture_location,
			GLint scene_shadow_texture_location,
			GLint proj_to_world_location, GLint world_to_Lproj_location, GLint time_location)
		{
#define _ASSIGN_UNIFORM_LOCATION(NAME) NAME.location = NAME##_location
			_ASSIGN_UNIFORM_LOCATION(atmosphere_transmittance_texture);
			_ASSIGN_UNIFORM_LOCATION(atmosphere_scattering_texture);
			_ASSIGN_UNIFORM_LOCATION(sky_irradiance_texture);
			_ASSIGN_UNIFORM_LOCATION(sun_vector);
			_ASSIGN_UNIFORM_LOCATION(sun_size);

			_ASSIGN_UNIFORM_LOCATION(cloud_height_texture);
			_ASSIGN_UNIFORM_LOCATION(cloud_weather_texture); _ASSIGN_UNIFORM_LOCATION(cloud_weather_texture_scale);
			_ASSIGN_UNIFORM_LOCATION(base_cloud_texture);    _ASSIGN_UNIFORM_LOCATION(base_cloud_texture_scale);
			_ASSIGN_UNIFORM_LOCATION(cloud_detail_texture);  _ASSIGN_UNIFORM_LOCATION(cloud_detail_texture_scale);
			_ASSIGN_UNIFORM_LOCATION(cloud_curl_texture);
			_ASSIGN_UNIFORM_LOCATION(cirrus_cloud_texture);
			_ASSIGN_UNIFORM_LOCATION(cloud_density_scale); _ASSIGN_UNIFORM_LOCATION(cloud_forward_scattering); _ASSIGN_UNIFORM_LOCATION(cloud_backward_scattering); _ASSIGN_UNIFORM_LOCATION(cloud_transmittance_lowbound);
			_ASSIGN_UNIFORM_LOCATION(wind_direction); _ASSIGN_UNIFORM_LOCATION(wind_intensity);

			_ASSIGN_UNIFORM_LOCATION(scene_brdf_texture);
			_ASSIGN_UNIFORM_LOCATION(scene_normal_texture);
			_ASSIGN_UNIFORM_LOCATION(scene_depth_texture);
			_ASSIGN_UNIFORM_LOCATION(scene_shadow_texture);
			_ASSIGN_UNIFORM_LOCATION(proj_to_world); _ASSIGN_UNIFORM_LOCATION(world_to_Lproj); _ASSIGN_UNIFORM_LOCATION(time);
#undef _ASSIGN_UNIFORM_LOCATION

#ifdef _DEBUG
			debug_uniform_locations = true;
#endif // _DEBUG
		}

		void set_atmosphere_transmittance_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			atmosphere_transmittance_texture.unit  = unit;
			atmosphere_transmittance_texture.value = texture2d;
			//atmosphere_transmittance_texture.location = location; [in set_uniform_locations(...)]
#ifdef _DEBUG
			debug_atmosphere_transmittance_texture = true;
#endif // _DEBUG
		}

		void set_atmosphere_scattering_texture(GLint unit, GLuint texture3d) {
			assert(glIsTexture(texture3d));
			atmosphere_scattering_texture.unit  = unit;
			atmosphere_scattering_texture.value = texture3d;
#ifdef _DEBUG
			debug_atmosphere_scattering_texture = true;
#endif // _DEBUG
		}

		void set_sky_irradiance_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			sky_irradiance_texture.unit = unit;
			sky_irradiance_texture.value = texture2d;
#ifdef _DEBUG
			debug_sky_irradiance_texture = true;
#endif // _DEBUG
		}

		void set_sun_property(const Euler::Angles<float>& sun_direction, const float* sun_size) {
			assert(sun_size != nullptr);
			this->sun_direction     = sun_direction;
			this->sun_size.value[0] = sun_size[0];
			this->sun_size.value[1] = sun_size[1];
#ifdef _DEBUG
			debug_sun_property = true;
#endif // _DEBUG
		}

		void set_cloud_height_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			cloud_height_texture.unit = unit;
			cloud_height_texture.value = texture2d;
#ifdef _DEBUG
			debug_cloud_height_texture = true;
#endif // _DEBUG
		}

		void set_cloud_weather_texture(GLint unit, GLuint texture2d, float scale) {
			assert(glIsTexture(texture2d));
			assert(!isnan(scale));
			cloud_weather_texture.unit = unit;
			cloud_weather_texture.value = texture2d;
			cloud_weather_texture_scale.value = scale;
#ifdef _DEBUG
			debug_cloud_weather_texture = true;
#endif // _DEBUG
		}

		void set_base_cloud_texture(GLint unit, GLuint texture3d, float scale) {
			assert(glIsTexture(texture3d));
			assert(!isnan(scale));
			base_cloud_texture.unit = unit;
			base_cloud_texture.value = texture3d;
			base_cloud_texture_scale.value = scale;
#ifdef _DEBUG
			debug_base_cloud_texture = true;
#endif // _DEBUG
		}

		void set_cloud_detail_texture(GLint unit, GLuint texture3d, float scale) {
			assert(glIsTexture(texture3d));
			assert(!isnan(scale));
			cloud_detail_texture.unit = unit;
			cloud_detail_texture.value = texture3d;
			cloud_detail_texture_scale.value = scale;
#ifdef _DEBUG
			debug_cloud_detail_texture = true;
#endif // _DEBUG

		}

		void set_cloud_curl_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			cloud_curl_texture.unit = unit;
			cloud_curl_texture.value = texture2d;
#ifdef _DEBUG
			debug_cloud_curl_texture = true;
#endif // _DEBUG
		}

		void set_cirrus_cloud_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			cirrus_cloud_texture.unit = unit;
			cirrus_cloud_texture.value = texture2d;
#ifdef _DEBUG
			debug_cirrus_cloud_texture = true;
#endif // _DEBUG
		}

		void set_cloud_property(float density_scale, float forward_scattering,float backward_scattering, float transmittance_lowbound) {
			assert(!isnan(density_scale));
			assert(!isnan(forward_scattering));
			assert(!isnan(backward_scattering));
			assert(!isnan(transmittance_lowbound));
			cloud_density_scale.value = density_scale;
			cloud_forward_scattering.value = forward_scattering;
			cloud_backward_scattering.value = backward_scattering;
			cloud_transmittance_lowbound.value = transmittance_lowbound;
#ifdef _DEBUG
			debug_cloud_property = true;
#endif // _DEBUG
		}

		void set_wind_property(const float* wind_direction, float wind_intensity) {
			assert(wind_direction != nullptr);
			assert(!isnan(wind_intensity));
			this->wind_direction.value[0] = wind_direction[0];
			this->wind_direction.value[1] = wind_direction[1];
			this->wind_direction.value[2] = wind_direction[2];
			this->wind_intensity.value    = wind_intensity;
#ifdef _DEBUG
			debug_wind_property = true;
#endif // _DEBUG

		}

		void set_scene_brdf_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			scene_brdf_texture.unit = unit;
			scene_brdf_texture.value = texture2d;
#ifdef _DEBUG
			debug_scene_brdf_texture = true;
#endif // _DEBUG
		}

		void set_scene_normal_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			scene_normal_texture.unit = unit;
			scene_normal_texture.value = texture2d;
#ifdef _DEBUG
			debug_scene_normal_texture = true;
#endif // _DEBUG
		}

		void set_scene_depth_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			scene_depth_texture.unit = unit;
			scene_depth_texture.value = texture2d;
#ifdef _DEBUG
			debug_scene_depth_texture = true;
#endif // _DEBUG
		}

		void set_scene_shadow_texture(GLint unit, GLuint texture2d) {
			assert(glIsTexture(texture2d));
			scene_shadow_texture.unit  = unit;
			scene_shadow_texture.value = texture2d;
#ifdef _DEBUG
			debug_scene_shadow_texture = true;
#endif // _DEBUG
		}
	};

}// geometry
