//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache Licene 2.0
//--------------------------------------------------------------------------------------
#pragma once

#include "basic.h"

/*<enum name="GLdatatype">
	GL_BYTE,
	GL_UNSIGNED_BYTE,
	GL_SHORT,
	GL_UNSIGNED_SHORT,
	GL_INT,
	GL_UNSIGNED_INT,
	GL_FLOAT,
	GL_DOUBLE
</enum>*/
inline GLenum glGetDataType(const int8_t*) {
	return GL_BYTE;
}
inline GLenum glGetDataType(const uint8_t*) {
	return GL_UNSIGNED_BYTE;
}
inline GLenum glGetDataType(const int16_t*) {
	return GL_SHORT;
}
inline GLenum glGetDataType(const uint16_t*) {
	return GL_UNSIGNED_SHORT;
}
inline GLenum glGetDataType(const int32_t*) {
	return GL_INT;
}
inline GLenum glGetDataType(const uint32_t*) {
	return GL_UNSIGNED_INT;
}
inline GLenum glGetDataType(const float*) {
	return GL_FLOAT;
}
inline GLenum glGetDataType(const double*) {
	return GL_DOUBLE;
}

/*<enum name="GLbuffer_target">
	GL_ARRAY_BUFFER,
	GL_TEXTURE_BUFFER,
	GL_UNIFORM_BUFFER,
	GL_PIXEL_UNPACK_BUFFER,
	GL_ELEMENT_ARRAY_BUFFER,
	GL_ATOMIC_COUNTER_BUFFER
</enum>*/

/*<enum name="buffer_usage">
	GL_STATIC_DRAW, // DEFAUL_HEAP
	GL_STATIC_READ, // READBACK_HEAP
	GL_STATIC_COPY,
	GL_DYNAMIC_DRAW, // UPLOAD_HEAP
	GL_DYNAMIC_READ, // UPLOAD_HEAP
	GL_DYNAMIC_COPY  // UPLOAD_HEAP
</enum>*/

inline void 
_glDeleteBuffer(GLuint& _GLXbo) {
	if (glIsBuffer(_GLXbo)) {
		glDeleteBuffers(1, &_GLXbo);
	}
	_GLXbo = -1;
}

inline void 
glUploadBuffer(GLenum target, GLintptr offset, GLsizeiptr bytesize, const void* data) {
	// upload [data, data+bytesize) to [target+offset, ...)
	if (data != nullptr) {
		glBufferSubData(target, offset, bytesize, data);
	}
}

inline void 
glReadbackBuffer(GLenum target, GLintptr offset, GLsizeiptr bytesize, void* out_dst) {
	glGetBufferSubData(target, offset, bytesize, out_dst);
}

inline GLuint
glCreateBuffer(GLenum target, GLenum usage, GLsizei bytesize) {
	glGetError();
	GLuint _GLXbo = -1; glGenBuffers(1, &_GLXbo);

	glBindBuffer(target, _GLXbo);
	glBufferData(target, bytesize, nullptr, usage);
	glBindBuffer(target, 0);

	assert( glIsBuffer(_GLXbo) );
	glTestErrorFailedThrow();
	return _GLXbo;
}

// { this function from <OGRE> }
/*inline void   copy_buffer(GLuint dst, GLuint src, size_t _Size) {
	glBindBuffer(GL_COPY_READ_BUFFER, src);
	glBindBuffer(GL_COPY_WRITE_BUFFER, dst);

	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, _Size);

	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
}*/

struct GLbufferDesc {
	GLenum target = GL_ARRAY_BUFFER;
	GLenum usage  = GL_STATIC_DRAW;
	GLsizei bytesize = 0;
};

inline void glGetBufferDesc(GLenum target, GLbufferDesc* desc_ptr) {
	desc_ptr->target = target;
	glGetBufferParameteriv(target, GL_BUFFER_USAGE, reinterpret_cast<GLint*>(&desc_ptr->usage));
	glGetBufferParameteriv(target, GL_BUFFER_SIZE, &desc_ptr->bytesize);
	assert(glGetError() == GL_NO_ERROR);
}

// {}
class GLbuffer {
public:
	GLuint identifier = static_cast<GLuint>(-1);
	GLbufferDesc descriptor;

	GLbuffer() = default;
	GLbuffer(const GLbuffer&) = delete;
	GLbuffer(GLbuffer&& _Right) noexcept {
		_Right.swap(*this);
		_Right.release();
	}
	GLbuffer(GLenum target, GLuint buffer) {
		identifier = buffer;
		glBindBuffer(target, buffer);
		glGetBufferDesc(target, &descriptor);
		glBindBuffer(target, 0);
	}
	explicit GLbuffer(const GLbufferDesc& desc)
		: identifier(glCreateBuffer(desc.target, desc.usage, desc.bytesize)), descriptor(desc) {}
	~GLbuffer() { release(); }

	GLbuffer& operator=(const GLbuffer&) = delete;
	GLbuffer& operator=(GLbuffer&& _Right) noexcept {
		_Right.swap(*this);
		_Right.release();
		return *this;
	}
	operator GLuint() const {
		return identifier;
	}

	bool operator==(const GLbuffer& _Right)const {
		return identifier == _Right.identifier;
	}
	bool operator!=(const GLbuffer& _Right)const {
		return identifier != _Right.identifier;
	}

	void release() {
		_glDeleteBuffer(identifier);
	}
	void swap(GLbuffer& _Right) {
		std::swap(identifier, _Right.identifier);
		std::swap(descriptor, _Right.descriptor);
	}
	bool valid() const {
		return glIsBuffer(identifier);
	}
	GLenum target() const {
		return descriptor.target;
	}
	size_t bytesize() const {
		return descriptor.bytesize;
	}
};

// {}
template<typename _Ty>
class GLbufferx : public GLbuffer {
	using _Mybase = GLbuffer;
public:
	GLbufferx() = default;
	GLbufferx(const GLbufferx&) = delete;
	GLbufferx(GLbufferx&& _Right) noexcept
		: _Mybase(std::move(_Right)) {}
	GLbufferx(GLenum target, GLuint buffer)
		: _Mybase(target, buffer) {}
	explicit GLbufferx(GLenum target, const _Ty* data, GLsizei data_size, GLenum usage = GL_STATIC_DRAW)
		: _Mybase( GLbufferDesc{ target, usage, GLsizei(sizeof(_Ty)*data_size) } ) {
		glBindBuffer(target, this->identifier);
		glUploadBuffer(target, 0, sizeof(_Ty)*data_size, data);
		glBindBuffer(target, 0);
	}

	GLbufferx& operator=(const GLbufferx&) = delete;
	GLbufferx& operator=(GLbufferx&& _Right) noexcept {
		_Mybase::operator=(std::move(_Right));
		return *this;
	}

	void upload(const _Ty* data, size_t data_size) {
		glBindBuffer(this->target(), this->identifier);
		glUploadBuffer(this->target(), 0, sizeof(_Ty)*data_size, data);
		glBindBuffer(this->target(), 0);
	}
	GLsizei size() const {
		return _Mybase::bytesize() / sizeof(_Ty);
	}
};
