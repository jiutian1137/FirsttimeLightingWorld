//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache Licene 2.0
//
// Content of tables
// 1. shader -------------------------------------------------------------
// 2. program ------------------------------------------------------------
// 3. shaderX ------------------------------------------------------------
// 4. varyings -----------------------------------------------------------
// 5. uniform block ------------------------------------------------------
// 6. texture resource ---------------------------------------------------
// 7. programX -----------------------------------------------------------
//--------------------------------------------------------------------------------------
#pragma once

#include "basic.h"
#include "gltexture.h"
#include "glbuffer.h"
#include <map>
#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <filesystem>
#include <regex>

/*<enum name="GLblend_operation">
	GL_FUNC_ADD,
	GL_FUNC_SUBTRACT,
	GL_FUNC_REVERSE_SUBTRACT,
	GL_MIN,
	GL_MAX
</enum>*/

/*<enum>
	GL_ONE,
	GL_ZERO,
	GL_SRC_ALPHA,
	GL_DST_ALPHA,
	GL_SRC_COLOR,
	GL_DST_COLOR,
	GL_ONE_MINUS_SRC_ALPHA,
	GL_ONE_MINUS_DST_ALPHA,
	GL_ONE_MINUS_SRC_COLOR,
	GL_ONE_MINUS_DST_COLOR
</enum>*/

/*<guide>
	<flame effect> fire_blend{ GL_SRC_ALPHA, GL_ONE }, 
		            smoke_blend{ GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA } 
	</flame effect>
</guide>*/

/*<enum name="GLshader_target">
	GL_VERTEX_SHADER,
	GL_TESS_CONTROL_SHADER,
	GL_TESS_EVALUATION_SHADER,
	GL_GEOMETRY_SHADER,
	GL_FRAGMENT_SHADER,
	GL_COMPUTE_SHADER
</enum>*/

/*<enum name="GLshader_status">
	GL_COMPILE_STATUS,
	GL_DELETE_STATUS,
	GL_VALIDATE_STATUS
</enum>*/

struct GLshaderTidyGuard {
	GLuint shader;
	~GLshaderTidyGuard() {
		if (glIsShader(shader)) {
			glDeleteShader(shader);
		}
	}
};
struct GLprogramTidyGuard {
	GLuint program;
	~GLprogramTidyGuard() {
		if (glIsProgram(program)) {
			glDeleteProgram(program);
		}
	}
};
struct GLvaryingsTidyGuard {
	GLuint varyigns;
	~GLvaryingsTidyGuard() {
		if (glIsVertexArray(varyigns)) {
			glDeleteVertexArrays(1, &varyigns);
		}
	}
};


inline void 
glEnableBlend(GLenum sfactor, GLenum dfactor, GLenum mode) {
	glEnable(GL_BLEND);
	glBlendEquation(mode);
	glBlendFunc(sfactor, dfactor);
}

inline void 
glResetBlend(GLenum sfactor, GLenum dfactor, GLenum mode) {
	glBlendEquation(mode);
	glBlendFunc(sfactor, dfactor);
}

/*<enum name="GLprimive">
	GL_POINTS,
	GL_LINES,
	GL_LINE_LOOP,
	GL_LINE_STRIP,
	GL_TRIANGLES,
	GL_TRIANGLE_STRIP,
	GL_TRIANGLE_FAN,
	GL_QUADS,
	GL_QUAD_STRIP,
	GL_POLYGON,
	GL_PATCHES
</enum>*/

inline void 
glDrawNovarying(GLenum mode, GLsizei count) {
	// { No data rendering }
	glDrawArrays(mode, 0, count);
}

inline void 
glDrawArrays(GLenum mode, GLuint _GLvao, GLsizei first, GLsizei count) {
// { ID3D12GraphicsCommandList::DrawInstanced }
	assert(glIsVertexArray(_GLvao));

	glBindVertexArray(_GLvao);
	glDrawArrays(mode, first, count);
	glBindVertexArray(0);
}

inline void 
glDrawElements(GLenum mode, GLuint _GLvao, GLuint _GLebo, GLsizei start, GLsizei count, GLenum type = GL_UNSIGNED_INT) {
// { ID3D12GraphicsCommandList::DrawIndexedInstanced }
	assert(glIsVertexArray(_GLvao));
	assert(glIsBuffer(_GLebo));

	glBindVertexArray(_GLvao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _GLebo);
	glDrawElements(mode, count, type, (const void*)start);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////// Uniforms ////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

inline void glUniformTexture(GLint location, GLenum unit, GLenum target, GLuint texture) {
	glUniform1i(location, unit - GL_TEXTURE0);
	glActiveTexture(unit);
	glBindTexture(target, texture);
}

//struct GLuniformFloat {
//	float value;
//	GLint location = -1;
//};
//struct GLuniformVec2 {
//	float value[2];
//	GLint location = -1;
//};
//struct GLuniformVec3 {
//	float value[3];
//	GLint location = -1;
//};
//struct GLuniformVec4 {
//	float value[4];
//	GLint location = -1;
//};
//struct GLuniformMat2 {
//	float value[2*2];
//	bool  transpose = true;
//	GLint location  = -1;
//};
//struct GLuniformMat3 {
//	float value[3*3];
//	bool  transpose = true;
//	GLint location  = -1;
//};
//struct GLuniformMat4 {
//	float value[4*4];
//	bool  transpose = true;
//	GLint location  = -1;
//};
//struct GLuniformSampler2D {
//	GLenum unit     = GL_TEXTURE0;
//	GLuint value    = static_cast<GLuint>(-1);
//	GLint  location = -1;
//};
//struct GLuniformSampler3D {
//	GLenum unit     = GL_TEXTURE0;
//	GLuint value    = static_cast<GLuint>(-1);
//	GLint  location = -1;
//};
//
//inline void glSetUniformFloat(const GLuniformFloat& v) {
//	glUniform1f(v.location, v.value);
//}
//inline void glSetUniformVec2(const GLuniformVec2& v) {
//	glUniform2fv(v.location, 1, v.value);
//}
//inline void glSetUniformVec3(const GLuniformVec3& v) {
//	glUniform3fv(v.location, 1, v.value);
//}
//inline void glSetUniformVec4(const GLuniformVec4& v) {
//	glUniform4fv(v.location, 1, v.value);
//}
//inline void glSetUniformMat2(const GLuniformMat2& v) {
//	glUniformMatrix2fv(v.location, 1, v.transpose, v.value);
//}
//inline void glSetUniformMat3(const GLuniformMat3& v) {
//	glUniformMatrix3fv(v.location, 1, v.transpose, v.value);
//}
//inline void glSetUniformMat4(const GLuniformMat4& v) {
//	glUniformMatrix4fv(v.location, 1, v.transpose, v.value);
//}
//inline void glSetUniformSampler2D(const GLuniformSampler2D& v) {
//	glActiveTexture(v.unit);
//	glBindTexture(GL_TEXTURE_2D, v.value);
//	glUniform1i(v.location, v.unit - GL_TEXTURE0);
//}
//inline void glSetUniformSampler3D(const GLuniformSampler3D& v) {
//	glActiveTexture(v.unit);
//	glBindTexture(GL_TEXTURE_3D, v.value);
//	glUniform1i(v.location, v.unit - GL_TEXTURE0);
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////// Varyings ////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

struct GLvaryingDesc {
// { example: { 4, GL_FLOAT, GL_FALSE, sizeof(float)*4, 0 } }
	GLint	  size       = 4;// unit:[scalar]
	GLenum    type       = GL_FLOAT;
	GLboolean normalized = GL_FALSE;
	GLsizei   stride     = 16;// unit:[bytes]
	GLint     offset     = 0; // unit:[bytes]
};

using GLvaryingsDesc = std::vector<GLvaryingDesc>;

inline void glGetVaryingsDesc(GLvaryingsDesc* desc_ptr) {
	GLint max_vertex_attribs;
	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &max_vertex_attribs);

	GLint enabled; GLvaryingDesc varying;
	for (GLuint i = 0; i != max_vertex_attribs; ++i) {
		glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
		if (enabled == GL_FALSE) {
			break;
		}
		glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_SIZE, &varying.size);
		glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_TYPE, reinterpret_cast<GLint*>(&varying.type));
		glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, reinterpret_cast<GLint*>(&varying.normalized));
		glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_STRIDE, reinterpret_cast<GLint*>(&varying.stride));
		glGetVertexAttribPointerv(i, GL_VERTEX_ATTRIB_ARRAY_POINTER, reinterpret_cast<void**>(&varying.offset));
		desc_ptr->push_back(varying);
	}

	assert(glGetError() == GL_NO_ERROR);
}

inline void glVaryingsDesc(const GLvaryingsDesc& desc) {
	assert(!desc.empty());

	/*GLint max_vertex_attribs;
	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &max_vertex_attribs);
	for (GLuint i = 0; i != max_vertex_attribs; ++i) {
		glDisableVertexAttribArray(i);
	}*/

	for ( GLsizei i = 0; i != desc.size(); ++i ) {
		switch ( desc[i].type ) {
		case GL_UNSIGNALED:
		case GL_INT:
		case GL_UNSIGNED_INT:
			glVertexAttribIPointer(i, desc[i].size, desc[i].type, desc[i].stride, (void*)(desc[i].offset));
			glEnableVertexAttribArray(i); 
			break;
		case GL_FLOAT:
			glVertexAttribPointer(i, desc[i].size, desc[i].type, desc[i].normalized, desc[i].stride, (void*)(desc[i].offset));
			glEnableVertexAttribArray(i);
			break;
		case GL_DOUBLE:// in glsl "#version 400 core\n"
			glVertexAttribLPointer(i, desc[i].size, desc[i].type, desc[i].stride, (void*)(desc[i].offset));
			glEnableVertexAttribArray(i);
			break;
		default:
			break;
		}
	}

	assert( glGetError() == GL_NO_ERROR );
}
#define glBindVaryings glBindVertexArray
#define glDeleteVaryings glDeleteVertexArrays
#define glIsVaryings glIsVertexArray

struct GLvaryings {
	GLuint identifier = static_cast<GLuint>(-1);
	GLbuffer vertex_buffer;
	GLvaryingsDesc descriptor;
	
	GLvaryings() = default;
	GLvaryings(const GLvaryings&) = delete;
	GLvaryings(GLvaryings&& _Right) noexcept {
		_Right.swap(*this);
		_Right.release();
	}
	explicit GLvaryings(GLuint _GLvao) {
		// 1. Set vertex_array_identitifer
		this->identifier = _GLvao;
		// 2. Set vertex_array_descriptor
		glBindVertexArray(this->identifier);
		glGetVaryingsDesc(&this->descriptor);
		// 3. Set vertex_buffer_identitifer
		glGetVertexAttribiv(0, GL_VERTEX_ARRAY_BUFFER_BINDING, reinterpret_cast<GLint*>(&this->vertex_buffer.identifier));
		// 4. Set vertex_buffer_descriptor
		glBindBuffer(GL_ARRAY_BUFFER, this->vertex_buffer);
		glGetBufferDesc(GL_ARRAY_BUFFER, &this->vertex_buffer.descriptor);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	explicit GLvaryings(GLuint _GLvbo, const GLvaryingsDesc& _Desc) {
		// 
		this->vertex_buffer = GLbuffer(GL_ARRAY_BUFFER, _GLvbo);
		//
		glGenVertexArrays(1, &this->identifier);
		glBindVertexArray(this->identifier);
		glBindBuffer(GL_ARRAY_BUFFER, this->vertex_buffer);
		glVaryingsDesc(_Desc);
		//
		descriptor = _Desc;

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	template<typename _Tyvert>
	explicit GLvaryings(GLbufferx<_Tyvert>&& _Vbo) {
		// 
		vertex_buffer = std::move(_Vbo);
		//
		glGenVertexArrays(1, &this->identifier);
		glBindVertexArray(this->identifier);
		glBindBuffer(GL_ARRAY_BUFFER, this->vertex_buffer);
		glVaryingsDesc(_Tyvert::descriptor());
		//
		descriptor = _Tyvert::descriptor();

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	~GLvaryings() { release(); }
	
	GLvaryings& operator=(const GLvaryings&) = delete;
	GLvaryings& operator=(GLvaryings&& _Right) noexcept {
		_Right.swap(*this);
		_Right.release();
		return *this;
	}
	operator GLuint() const {
		return identifier;
	}

	void release() {
		glDeleteVaryings(1, &identifier);
		glDeleteBuffers(1, &vertex_buffer.identifier);
	}
	void swap(GLvaryings& _Right) {
		std::swap(_Right.identifier,    this->identifier);
		std::swap(_Right.vertex_buffer, this->vertex_buffer);
		std::swap(_Right.descriptor,    this->descriptor);
	}
	bool valid() const {
		return glIsVaryings(identifier);
	}

	void ready() const {
		glBindVertexArray(identifier);
	}
	void done() const {
		glBindVertexArray(0);
	}
};



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////// GLSL(Processor) ///////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

struct text_preprocess {
	static std::string read_file(const std::filesystem::path& _Path) {
		if ( std::filesystem::exists(_Path) ) {
			std::string   _Source;
			std::ifstream _Fin;

			_Fin.open(_Path.string());
			_Fin.seekg(0, std::ios::end);
			_Source.resize(size_t(_Fin.tellg()), ' ');
			_Fin.seekg(std::ios::beg);
			_Fin.read(&_Source[0], _Source.size());
			_Fin.close();
			return std::move(_Source);
		} else {
			return std::string();
		}
	}

	static size_t seek_to(char _Target, const std::string& _Str, size_t _Seek) {
		// seek to _Target in [_Str+_Seek, ...)
		while (_Seek != _Str.size()) {
			if (_Str[_Seek] == _Target) {
				break;
			}
			++_Seek;
		}

		return _Seek != _Str.size() ? _Seek : std::string::npos;
	}

	static size_t skip(char _Target, const std::string& _Str, size_t _Seek) {
		// skip _Target in [_Str+_Seek, ...)
		while (_Seek != _Str.size()) {
			if (_Str[_Seek] != _Target) {
				break;
			}
			++_Seek;
		}

		return _Seek != _Str.size() ? _Seek : std::string::npos;
	}

	static void includefile_recursion(std::string& _Source, const std::filesystem::path& _Parent, size_t _Seek = 0) {
		while ( true ) {
			_Seek = _Source.find("#include", _Seek);
			if (_Seek == std::string::npos) {
				break;
			} else {
				size_t _First = _Seek;
					   _Seek  = seek_to('\"', _Source, _Seek);
				assert(_Seek != std::string::npos);
					   _Seek += 1;// skip '\"'
				size_t _Last  = _Source.find('\"', _Seek);
				assert(_Last != std::string::npos);

				std::filesystem::path _Included_path = std::string(&_Source[_Seek], &_Source[_Last]);
					                  _Included_path = _Parent/_Included_path;
				std::string _Included_source = read_file(_Included_path);
				includefile_recursion(_Included_source, _Included_path.parent_path());
					
				_Last += 1;// contain '\"'
				_Source.erase(_First, _Last - _First);
				_Source.insert(_First, _Included_source);
				_Seek = _First + _Included_source.size();
			}
		}
	}

	static void macros_assign(std::string& _Source, const std::map<std::string, std::string>& _Macros) {
		for (auto _First = _Macros.begin(); _First != _Macros.end(); ++_First) {
			size_t _Seek = _Source.find("#define ");
			while (_Seek != std::string::npos) {
				_Seek += strlen("#define");
				size_t _Macro_name_first = skip(' ', _Source, _Seek);
				assert(_Macro_name_first != std::string::npos);
				size_t _Macro_name_last  = std::min(seek_to('\n', _Source, _Macro_name_first), seek_to(' ', _Source, _Macro_name_first));
				assert(_Macro_name_last != std::string::npos);
				const std::string _Macro_name = std::string(&_Source[_Macro_name_first], &_Source[_Macro_name_last]);
				
				if (_First->first == _Macro_name) {
					size_t _Macro_value_first = _Macro_name_last;
					size_t _Macro_value_last  = seek_to('\n', _Source, _Macro_value_first);
					assert(_Macro_value_last != std::string::npos);
					_Source.erase(_Macro_value_first, _Macro_value_last - _Macro_value_first);
					_Source.insert(_Macro_value_first, ' ' + _First->second);
					break;
				}

				_Seek = _Source.find("#define", _Macro_name_last);
			}
		}
	}
};

inline void glslGetVersion(const std::string& _Source, std::string& _Version) {
	std::regex _MATCH_VERSION = std::regex("^(#version)([ ]+)([[:d:]][[:d:]][[:d:]])([ ]+)(core)([ ]*)(\n)");
	if (std::regex_search(_Source, _MATCH_VERSION)) {
		_Version = std::string(_Source.begin(), _Source.begin() + _Source.find('\n') + 1);
	} else {
		_Version = std::string();
	}
}

inline void glslPreprocess(std::string& _Source, const std::filesystem::path& _Parent) {
	// 1. Find "#version xxx core\n"
	std::string _Version;
	glslGetVersion(_Source, _Version);
	if (_Version.empty()) {
		_Source.insert(size_t(0), "#version 330 core\n");
	}

	// 2. includefile
	text_preprocess::includefile_recursion(_Source, _Parent, 0);
}

inline void glslPreprocess(std::string& _Source, const std::filesystem::path& _Parent, const std::map<std::string, std::string>& _Macros) {
	// 1. Find "#version xxx core\n"
	std::string _Version;
	glslGetVersion(_Source, _Version);
	if (_Version.empty()) {
		_Source.insert(size_t(0), "#version 330 core\n");
	}

	// 2. includefile
	text_preprocess::includefile_recursion(_Source, _Parent, 0);

	// 3. macros assign
	text_preprocess::macros_assign(_Source, _Macros);
}

struct GLshadersource {
	GLshadersource() = default;
	explicit GLshadersource(const std::string& _Source) : _Mypath(), _Mysource(_Source) {
		glslPreprocess(_Mysource, _Mypath.parent_path());
	}
	explicit GLshadersource(const std::string& _Source, const std::map<std::string, std::string>& _Macros) : _Mypath(), _Mysource(_Source) {
		glslPreprocess(_Mysource, _Mypath.parent_path(), _Macros);
	}
	explicit GLshadersource(const std::filesystem::path& _Path) : _Mypath(_Path), _Mysource(text_preprocess::read_file(_Path)) {
		glslPreprocess(_Mysource, _Mypath.parent_path());
	}
	explicit GLshadersource(const std::filesystem::path& _Path, const std::map<std::string, std::string>& _Macros)
		: _Mypath(_Path), _Mysource(text_preprocess::read_file(_Path)) {
		glslPreprocess(_Mysource, _Mypath.parent_path(), _Macros);
	}

	std::string str() const {
		return _Mysource;
	}
	operator std::string() const {
		return str();
	}

	std::string get_version() const {
		std::string _Version;
		glslGetVersion(_Mysource, _Version);
		return std::move(_Version);
	}

	std::filesystem::path _Mypath;
	std::string _Mysource;
};



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////// GLshader //////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void glCompileShader2(GLenum target, const GLchar* const* source, GLuint* shader_ptr, std::ostream& couterror) {
	GLuint shader;

	{
		shader = glCreateShader(target);
		GLshaderTidyGuard _Guard{ shader };

		glShaderSource(shader, 1, source, nullptr);
		glCompileShader(shader);
		GLint status;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
		if (status == GL_FALSE) {
			std::array<char, 1024> message;
			GLsizei                message_size = 0;
			glGetShaderInfoLog(shader, message.size() - 1, &message_size, &message[0]);
			message[message_size] = '\0';
			couterror << message.data();
		}

		assert(glGetError() == GL_NO_ERROR);
		_Guard.shader = -1;
	}

	if (glIsShader(*shader_ptr)) { glDeleteShader(*shader_ptr); }
	*shader_ptr = shader;
}

inline void glLinkProgram2(GLuint program, const GLuint* const shader_array, GLsizei array_size, std::ostream& couterror) {
	for (GLsizei i = 0; i != array_size; ++i) {
		glAttachShader(program, shader_array[i]);
	}
	glLinkProgram(program);

	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		std::array<char, 1024> message;
		GLsizei                message_size = 0;
		glGetProgramInfoLog(program, message.size() - 1, &message_size, &message[0]);
		message[message_size] = '\0';
		couterror << message.data();
	}
}

struct GLshader {
	GLshader() = default;
	GLshader(const GLshader&) = delete;
	GLshader& operator=(const GLshader&) = delete;

	GLshader& operator=(GLshader&& _Right) noexcept {
		_Right.swap(*this);
		_Right.release();
		return *this;
	}
	GLshader(GLshader&& _Right) noexcept { 
		*this = std::move(_Right);
	}

	GLshader(GLenum _Target, const GLchar* _Source) : source_str(_Source) { 
		glCompileShader2(_Target, &_Source, &this->identifier, std::cout);
	}
	GLshader(GLenum _Target, const std::string& _Source) : source_str(_Source) {
		const GLchar* _Ptr = _Source.c_str();
		glCompileShader2(_Target, &_Ptr, &this->identifier, std::cout);
	}

	void release() {
		source_str.clear();
		if (glIsShader(identifier)) { glDeleteShader(identifier); }
		identifier = -1;
	}
	void swap(GLshader& _Right) {
		std::swap(identifier, _Right.identifier);
		std::swap(source_str, _Right.source_str);
	}
	~GLshader() { 
		release();
	}
	operator GLuint() const {
		return this->identifier;
	}

	GLuint      identifier = GLuint(-1); // this is _GLshader
	std::string source_str;
};

template<GLenum _Target>
class GLshaderx : public GLshader {
	using _Mybase = GLshader;
public:
	static constexpr GLuint target = _Target;

	GLshaderx() = default;
		
	GLshaderx(const char* _Source) : _Mybase(_Target, _Source) {}

	GLshaderx(const std::string& _Source) : GLshaderx(_Source.c_str()) {}

	GLshaderx(GLshaderx&& _Right) noexcept : _Mybase( std::move(_Right) ) {}
			
	GLshaderx& operator=(GLshaderx&& _Right) noexcept {
		_Mybase::operator=( std::move(_Right) );
		return *this;
	}

	GLshaderx& operator=(const char* _Source) {
		(*this) = GLshaderx(_Source);
		return *this;
	}
};

using GLvertshader = GLshaderx<GL_VERTEX_SHADER>;
using GLtescshader = GLshaderx<GL_TESS_CONTROL_SHADER>;
using GLteseshader = GLshaderx<GL_TESS_EVALUATION_SHADER>;
using GLgeomshader = GLshaderx<GL_GEOMETRY_SHADER>;
using GLfragshader = GLshaderx<GL_FRAGMENT_SHADER>;
using GLcompshader = GLshaderx<GL_COMPUTE_SHADER>;

	/*<note>
		                          readonly  read_write     readonly        readonly
	@_Interface: Shader-Resource: sampler*,   image*,   uniform varient, uniform_block
	@_Describ: This series of classes don't manage real memory, for example: @Free, @Alloc
	</note>*/

// 4. varyings -----------------------------------------------------------



// 5. uniform block ------------------------------------------------------

	enum class GLuniformblock_status {
		Binding = GL_UNIFORM_BLOCK_BINDING,
		DataSize = GL_UNIFORM_BLOCK_DATA_SIZE,
		NameLength = GL_UNIFORM_BLOCK_NAME_LENGTH,
		ActiveUniforms = GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS,
		ActiveUniformIndices = GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES,
		MaxUniformBlockSize = GL_MAX_UNIFORM_BLOCK_SIZE
	};

	// { glsl::resource }
	template<typename _Ty>
	class GLuniformblock : public _Ty {
		using _Mydata_t = _Ty;
	public:

		GLuint binding_index  = -1;// OpenGL buffer common index
		GLuint binding_point  = -1;// private index with @program 
		GLbufferx<_Ty> binding_buffer;
		GLuint _Ref_GLprogram  = -1;

		GLuniformblock() = default;
			
		GLuniformblock(GLuniformblock&& _Right) : _Mydata_t(_Right),
			binding_index(_Right.binding_index),
			binding_point(_Right.binding_point),
			binding_buffer(std::move(_Right.binding_buffer)),
			_Ref_GLprogram(_Right._Ref_GLprogram) {
			_Right.binding_index = -1;
			_Right.binding_point = -1;
			_Right._Ref_GLprogram    = -1;
		}

		GLuniformblock& operator=(GLuniformblock&& _Right) {
			_Mydata_t::operator=(std::move(_Right));
			this->binding_index      = _Right.binding_index;
			this->binding_point      = _Right.binding_point;
			this->binding_buffer     = std::move(_Right.binding_buffer);
			this->_Ref_GLprogram      = _Right._Ref_GLprogram;

			_Right.binding_index = -1;
			_Right.binding_point = -1;
			_Right._Ref_GLprogram    = -1;
			return *this;
		}

		void upload_uniformbuffer() {
			binding_buffer.upload(&static_cast<_Ty&>(*this), 1);
		}

		GLuint infolog(GLenum _Enum) const {
			GLint _Status;
			glGetActiveUniformBlockiv(_Ref_GLprogram, binding_index, _Enum, &_Status);
			return _Status;
		}

		void ready() const {
			glBindBufferBase(GL_UNIFORM_BUFFER, binding_index, binding_buffer.id); 
		}

		void done() const { 
			glBindBufferBase(GL_UNIFORM_BUFFER, binding_index, 0);
		}
	};	

// 6. texture resource ---------------------------------------------------

//#define _DEFINE_RESOURCE_SAMPLER(DX)                           \
//	template<GLenum _Fmt>                    \
//	struct GLsampler##DX {                                     \
//		using texture_type = GLtexture##DX<_Fmt>;              \
//		static constexpr GLenum target = texture_type::target; \
//															   \
//		GLenum texture_unit = GL_TEXTURE0;                     \
//		std::weak_ptr<texture_type> weak_texture;              \
//															   \
//		void ready() const {                                   \
//			if ( ! weak_texture.expired() ) {                   \
//				glActiveTexture(texture_unit);                  \
//				glBindTexture(target, weak_texture.lock()->id); \
//			}		                                            \
//		}                                                       \
//	\
//		void done() const {                     \
//			glActiveTexture(texture_unit);      \
//			glBindTexture(target, 0/*unbind*/); \
//		}                                       \
//	}
//
//	// { glsl::resource }
//	_DEFINE_RESOURCE_SAMPLER(2D);
//	// { glsl::resource }
//	_DEFINE_RESOURCE_SAMPLER(2DArray);
//	// { glsl::resource }
//	/*_DEFINE_RESOURCE_SAMPLER(2DArrayShadow);*/
//	// { glsl::resource }
//	_DEFINE_RESOURCE_SAMPLER(3D);
//	// { glsl::resource }
//	template<GLenum _Myfmt>
//	struct GLimage2D {
//		static_assert(_Myfmt != GLtexture_internalformat::RGB16f && _Myfmt != GLtexture_internalformat::RGB16i && _Myfmt != GLtexture_internalformat::RGB16ui,
//			"OpenGL shader not have rgb* image*");
//		using texture_type = GLtexture2D<_Myfmt>;
//
//		void ready() const {
//			if (not _Texture.expired()) {
//				glBindImageTexture(_Image_unit, _Texture.lock()->id(), 0, false, _Image_slice, _Image_access, (GLenum)_Myfmt);
//			}
//		}
//
//		void done() const {
//			glBindImageTexture(_Image_unit, 0/*unbind*/, 0, false, _Image_slice, _Image_access, (GLenum)_Myfmt);
//		}
//
//		GLuint _Image_unit   = 0;
//		GLint  _Image_slice  = 0;
//		GLenum _Image_access = GL_READ_WRITE;
//		std::weak_ptr<texture_type> _Texture;
//	};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////// GLprogram /////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// { OpenGL shader program }
class GLprogram {
public:
	GLuint identifier = GLuint(-1);
	
	std::vector<GLuint> _My_uniformblock_indices;
	GLvertshader _Myvert;
	GLgeomshader _Mygeom;
	GLteseshader _Mytese;
	GLtescshader _Mytesc;
	GLfragshader _Myfrag;
	GLcompshader _Mycomp;

	GLprogram() = default;
	GLprogram(const GLprogram&) = delete;
	GLprogram& operator=(const GLprogram&) = delete;

	GLprogram& operator=(GLprogram&& _Right) noexcept {
		_Right.swap(*this);
		_Right.release();
		return *this;
	}
	GLprogram(GLprogram&& _Right) noexcept {
		(*this) = std::move(_Right);
	}
		
	GLprogram(GLvertshader&& _Vert, GLfragshader&& _Frag) {
		_Myvert    = std::move(_Vert);
		_Myfrag    = std::move(_Frag);
		identifier = glCreateProgram();
		glLinkProgram2(identifier, std::array<GLuint, 2>{ _Myvert, _Myfrag }.data(), 2, std::cout);
	}
	GLprogram(GLvertshader&& _Vert, GLgeomshader&& _Geom, GLfragshader&& _Frag) {
		_Myvert = std::move(_Vert);
		_Mygeom = std::move(_Geom);
		_Myfrag = std::move(_Frag);
		identifier = glCreateProgram();
		glLinkProgram2(identifier, std::array<GLuint, 3>{ _Myvert, _Mygeom, _Myfrag }.data(), 3, std::cout);
	}
	explicit GLprogram(GLcompshader&& _Comp) : GLprogram() {
		_Mycomp    = std::move(_Comp);
		identifier = glCreateProgram();
		glLinkProgram2(identifier, std::array<GLuint, 1>{ _Mycomp }.data(), 1, std::cout);
	}

	void release() {
		_My_uniformblock_indices.clear();
		if (glIsProgram(identifier)) { glDeleteProgram(identifier); }
		identifier = -1;
	}
	void swap(GLprogram& _Right) {
		std::swap(identifier, _Right.identifier);
		std::swap(_My_uniformblock_indices, _Right._My_uniformblock_indices);
		std::swap(_Myvert, _Right._Myvert);
		std::swap(_Mygeom, _Right._Mygeom);
		std::swap(_Mytese, _Right._Mytese);
		std::swap(_Mytesc, _Right._Mytesc);
		std::swap(_Myfrag, _Right._Myfrag);
		std::swap(_Mycomp, _Right._Mycomp);
	}
	~GLprogram() {
		this->release();
	}
	operator GLuint() const {
		return this->identifier;
	}

	//template<typename _Ty>
	//GLuniformblock<_Ty> make_uniform_block(const std::string& _Ublock_name) {
	//	GLuniformblock<_Ty> _Ublock;

	//	/* 1. */
	//	GLbufferDesc _Desc;
	//	_Ublock.binding_buffer = GLbufferx<_Ty>( GL_UNIFORM_BUFFER, &_Ublock, 1, GL_DYNAMIC_COPY );
	//	//glGenBuffers(1, &_Block._Binding_buffer);
	//	//glBindBuffer(GL_UNIFORM_BUFFER, _Block._Binding_buffer);
	//	//glBufferData(GL_UNIFORM_BUFFER, sizeof(_Ty), (const void*)&_Block/*default data*/, GL_DYNAMIC_DRAW);// Note: default value
	//	//glBindBufferBase(GL_UNIFORM_BUFFER, 0, _Block._Binding_buffer);

	//	/* 2. */
	//	_Ublock.binding_index = glGetUniformBlockIndex(this->id, _Ublock_name.c_str());
	//	_Ublock.binding_point = static_cast<GLuint>(_My_uniformblock_indices.size());// initial value is 0
	//	glUniformBlockBinding(this->id, _Ublock.binding_index, _Ublock.binding_point);
	//	
	//	/* 3. */
	//	glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);
	//	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	//	_Ublock._Ref_GLprogram = this->id;

	//	assert(glGetError() != GL_NO_ERROR);
	//	_My_uniformblock_indices.push_back(_Ublock.binding_point);
	//	return std::move(_Ublock);
	//	/*
	//	1. create buffer
	//	2. get uniform block index, and binding point
	//	3. binding block and program
	//	*/
	//}

//	template<GLenum _Fmt>
//	GLsampler2D<_Fmt> make_sampler2D(GLenum _Unit, std::shared_ptr<GLtexture2D<_Fmt>> _Texture) {
//		return GLsampler2D<_Fmt>{ _Unit, _Texture };
//	}

//	template<GLenum _Fmt>
//	GLsampler3D<_Fmt> make_sampler3D(GLenum _Unit, std::shared_ptr<GLtexture3D<_Fmt>> _Texture) {
//		return GLsampler3D<_Fmt>{ _Unit, _Texture };
//	}

//	template<GLenum _Fmt>
//	GLsampler2DArray<_Fmt> make_sampler2D_array(GLenum _Unit, std::shared_ptr<GLtexture2DArray<_Fmt>> _Texture) {
//		return GLsampler2DArray<_Fmt>{ _Unit, _Texture };
//	}

///*	template<GLtexture_internalformat _Fmt>
//	GLsampler2DArrayShadow<_Fmt> make_sampler2D_array_shadow(GLenum _Unit, std::shared_ptr<GLtexture2DArray<_Fmt>> _Texture) {
//		return sampler2D_array_shadow<_Myfmt>{ _Unit, _Texture };
//	}*/

//	template<GLenum _Fmt>
//	GLimage2D<_Fmt> make_image2D(size_t _Unit, std::shared_ptr<GLtexture2D<_Fmt>> _Texture = nullptr) {
//		GLimage2D<_Fmt> _Image;
//		_Image._Image_unit = static_cast<GLuint>(_Unit);
//		_Image._Texture    = _Texture;
//		return _Image;
//	}

//	template<typename _Tex>
//	GLimage2D<GLslice2D<_Tex>::internalformat> make_image2D(size_t _Unit, std::shared_ptr<GLslice2D<_Tex>> _Texture) {
//		GLimage2D<GLslice2D<_Tex>::internalformat> _Image;
//		_Image._Image_unit  = static_cast<GLuint>(_Unit);
//		_Image._Texture     = _Texture;
//		_Image._Image_slice = _Texture->index;
//		return _Image;
//	}

	void ready() const {
		glUseProgram(identifier);
	}

private:
	void _Recursion_ready() const {
		/* final */
	}
		
	template<typename _Ty, typename ..._Tys>
	void _Recursion_ready(const _Ty& _Resource, const _Tys&... _Resources) {
		_Resource.ready();
		_Recursion_ready(_Resources...);
	}

	template<typename _Fty, typename ..._Tys>
	void _Recursion_ready(const std::function<_Fty>& _Func, typename std::function<_Fty>::argument_type _Args, const _Tys&... _Resources) const {
		_Func(_Args);
		_Recursion_ready(_Resources...);
	}

public:
	template<typename ..._Tys>
	void ready(const _Tys&... _Resources) const {
		glUseProgram(identifier);
		_Recursion_ready(_Resources...);
	}

	void done() const {
		glUseProgram(0);
	}

	template<typename _Ty, typename ..._Tys>
	void done(const _Ty& _Enditem, const _Tys&... _Enditems) const {
		_Enditem.done();
		done(_Enditems...);
	}
};
