//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache Licene 2.0
//
// Content of tables
// 1. GLenum -----------------------------------------------------------------
// 2. GLtexture_desc ---------------------------------------------------------
// 3. GLtexture --------------------------------------------------------------
//--------------------------------------------------------------------------------------
#pragma once

#include "glbuffer.h"

/*<enum name="GLtexture_target">
	GL_TEXTURE_1D,
	GL_TEXTURE_2D,
	GL_TEXTURE_3D,
	GL_TEXTURE_2D_ARRAY
<enum>*/

// replace GL_xxxxUI by GL_xxx
/*<enum name="GLtexture_internalformat">
	GL_RGBA,
	GL_RGBA8I,
	GL_RGBA8UI,
	GL_RGBA16F,
	GL_RGBA16I,
	GL_RGBA16UI,
	GL_RGBA32F,
	GL_RGBA32I,
	GL_RGBA32UI,

	GL_RGB,
	GL_RGB8I,
	GL_RGB8UI,
	GL_RGB16F,
	GL_RGB16I,
	GL_RGB16UI,
	GL_RGB32F,
	GL_RGB32I,
	GL_RGB32UI,

#ifdef GL_ARB_texture_rg
	GL_R,
	GL_R8I,
	GL_R8UI,
	GL_R16F,
	GL_R16I,
	GL_R16UI,
	GL_R32F,
	GL_R32I,
	GL_R32UI,

	GL_RG,
	GL_RG8I,
	GL_RG8UI,
	GL_RG16F,
	GL_RG16I,
	GL_RG16UI,
	GL_RG32F,
	GL_RG32I,
	GL_RG32UI,
#endif

#ifdef GL_ARB_depth_buffer_float
	GL_DEPTH_COMPONENT32F,
	GL_DEPTH32F_STENCIL8,
	GL_FLOAT_32_UNSIGNED_INT_24_8_REV
#endif

<notes>
	RGBA, RGB, R, RG, ...

	16[bit], 32[bit] included all types
		8[bit]          no float type
<notes>
<enum>*/

/*<enum name="GLtexture_dataformat">
	GL_RED,   // float
	GL_RG,
	GL_RGB,
	GL_RGBA,
	GL_DEPTH_COMPONENT, // used to shadow, depth compare

	GL_RED_INTEGER, // integer
	GL_RGB_INTEGER,
	GL_RGBA_INTEGER,
	GL_BGR_INTEGER, // used to OpenCV
	GL_BGRA_INTEGER
</enum>*/

/*<enum name="GLtexture_wrap">
	GL_CLAMP_TO_EDGE,
	GL_CLAMP_TO_BORDER,
	GL_REPEAT,
	GL_MIRRORED_REPEAT
</enum>*/

/*<enum name="GLtexture_filter">
	GL_LINEAR,
	GL_NEAREST
</enum>*/

/*<enum name="GLframebuffer_target">
	GL_FRAMEBUFFER,      // GL_READ_FRAMEBUFFER + GL_DRAW_FRAMEBUFFER
	GL_READ_FRAMEBUFFER, // glReadPixels(...)
	GL_DRAW_FRAMEBUFFER
</enum>*/

inline void
glGetInternalformatChannels(GLenum internalformat, GLsizei* channels_ptr) {
	switch (internalformat) {
		case GL_R8:
		case GL_R8UI:
		case GL_R8I:
		case GL_R16UI:
		case GL_R16I:
		case GL_R32F:
		case GL_DEPTH_COMPONENT24:
		case GL_DEPTH_COMPONENT32:
		case GL_DEPTH_COMPONENT32F:
			*channels_ptr = 1; break;
		case GL_RG8:
		case GL_RG8UI:
		case GL_RG8I:
		case GL_RG16UI:
		case GL_RG16I:
		case GL_RG32F:
			*channels_ptr = 2; break;
		case GL_RGB8:
		case GL_RGB8UI:
		case GL_RGB8I:
		case GL_RGB16UI:
		case GL_RGB16I:
		case GL_RGB32F:
			*channels_ptr = 3; break;
		case GL_RGBA8:
		case GL_RGBA8UI:
		case GL_RGBA8I:
		case GL_RGBA16UI:
		case GL_RGBA16I:
		case GL_RGBA32F:
			*channels_ptr = 4; break;
		default: 
			throw std::exception("clmagic::error::glGetInternalformatChannels(##GL_INVALID_ENUM##, channels_ptr)");
	}
}
inline void 
glGetInternalformatFormatAndType(GLenum internalformat, GLenum* format_ptr, GLenum* type_ptr) {
	switch (internalformat) {
		case GL_R8:
		case GL_R8UI:
		case GL_R8I:
		case GL_R16UI:
		case GL_R16I:
		case GL_R32F:
		case GL_DEPTH_COMPONENT24:
		case GL_DEPTH_COMPONENT32:
		case GL_DEPTH_COMPONENT32F:
			*format_ptr = GL_RED; break;
		case GL_RG8:
		case GL_RG8UI:
		case GL_RG8I:
		case GL_RG16UI:
		case GL_RG16I:
		case GL_RG32F:
			*format_ptr = GL_RG; break;
		case GL_RGB8:
		case GL_RGB8UI:
		case GL_RGB8I:
		case GL_RGB16UI:
		case GL_RGB16I:
		case GL_RGB32F:
			*format_ptr = GL_RGB; break;
		case GL_RGBA8:
		case GL_RGBA8UI:
		case GL_RGBA8I:
		case GL_RGBA16UI:
		case GL_RGBA16I:
		case GL_RGBA32F:
			*format_ptr = GL_RGBA; break;
		default: 
			throw std::exception("clmagic::error::glGetInternalformatFormatAndType(##GL_INVALID_ENUM##, format_ptr, type_ptr)");
	}

	switch (internalformat) {
		case GL_R8UI:
		case GL_RG8UI:
		case GL_RGB8UI:
		case GL_RGBA8UI:
		case GL_R8:
		case GL_RG8:
		case GL_RGB8:
		case GL_RGBA8:
			*type_ptr = GL_UNSIGNED_BYTE; break;
		case GL_R16UI:
		case GL_RG16UI:
		case GL_RGB16UI:
		case GL_RGBA16UI:
			*type_ptr = GL_UNSIGNED_SHORT; break;
		case GL_R32UI:
		case GL_RG32UI:
		case GL_RGB32UI:
		case GL_RGBA32UI:
			*type_ptr = GL_UNSIGNED_INT; break;
		case GL_R8I:
		case GL_RG8I:
		case GL_RGB8I:
		case GL_RGBA8I:
			*type_ptr = GL_BYTE; break;
		case GL_R16I:
		case GL_RG16I:
		case GL_RGB16I:
		case GL_RGBA16I:
			*type_ptr = GL_SHORT; break;
		case GL_R32I:
		case GL_RG32I:
		case GL_RGB32I:
		case GL_RGBA32I:
			*type_ptr = GL_INT; break;
		case GL_R32F:
		case GL_RG32F:
		case GL_RGB32F:
		case GL_RGBA32F:
		case GL_DEPTH_COMPONENT24:
		case GL_DEPTH_COMPONENT32:
		case GL_DEPTH_COMPONENT32F:
			*type_ptr = GL_FLOAT; break;
		default: 
			throw std::exception("clmagic::error::glGetInternalformatFormatAndType(##GL_INVALID_ENUM##, format_ptr, type_ptr)");
	}
}
inline void
glGetTexLevelChannels(GLenum target, GLint level, GLsizei* channels_ptr) {
	GLint internalformat;
	glGetTexLevelParameteriv(target, level, GL_TEXTURE_INTERNAL_FORMAT, &internalformat);
	glGetInternalformatChannels(internalformat, channels_ptr);
}
inline void
glGetTexLevelFormatAndType(GLenum target, GLint level, GLenum* format_ptr, GLenum* type_ptr) {
	GLint internalformat;
	glGetTexLevelParameteriv(target, level, GL_TEXTURE_INTERNAL_FORMAT, &internalformat);
	glGetInternalformatFormatAndType(internalformat, format_ptr, type_ptr);
}

// glTexStorage1D
// glTexStorage2D
// glTexStorage3D
inline void 
glTexStorageXD(GLenum target, GLint levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth) {
	switch (target) {
		case GL_TEXTURE_BUFFER:
		case GL_TEXTURE_1D: glTexStorage1D(target, levels, internalformat, width); break;
		case GL_TEXTURE_2D: glTexStorage2D(target, levels, internalformat, width, height); break;
			/*<example> glTexImage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1024, 1024); </example>*/
		case GL_TEXTURE_2D_ARRAY:
		case GL_TEXTURE_3D: glTexStorage3D(target, levels, internalformat, width, height, depth); break;
			/*<example> glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA32F, 128, 128, 128); </example>*/
		default: throw std::exception("clmagic::error::glTexStorageXD(##GL_INVALID_ENUM##, levels, internalformat, width, height, depth)");
	}

	glTestErrorFailedThrow();
}

inline void
glTexUpload1D(GLenum target, GLint level, GLint offset, GLsizei length, const void* pixel_array) {
	GLenum format;
	GLenum type;
	glGetTexLevelFormatAndType(target, level, &format, &type);
	glTexSubImage1D(target, level, offset, length, format, type, pixel_array);

	glTestErrorFailedThrow();
}
inline void
glTexUpload2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, const void* pixel_array) {
	GLenum format;
	GLenum type;
	glGetTexLevelFormatAndType(target, level, &format, &type);
	glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixel_array);

	glTestErrorFailedThrow();
}
inline void
glTexUpload3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, const void* pixel_array) {
	GLenum format;
	GLenum type;
	glGetTexLevelFormatAndType(target, level, &format, &type);
	glTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixel_array);

	glTestErrorFailedThrow();
}
inline void
glTexUploadXD(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, const void* pixel_array) {
	switch (target) {
		case GL_TEXTURE_BUFFER:
		case GL_TEXTURE_1D: glTexUpload1D(target, level, xoffset, width, pixel_array); break;
		case GL_TEXTURE_2D: glTexUpload2D(target, level, xoffset, yoffset, width, height, pixel_array); break;
		case GL_TEXTURE_2D_ARRAY:
		case GL_TEXTURE_3D: glTexUpload3D(target, level, xoffset, yoffset, zoffset, width, height, depth, pixel_array); break;
		default: throw std::exception("clmagic::error::glTexStorageXD(##GL_INVALID_ENUM##, level, xoffset, yoffset, zoffset, width, ...)");
	}

	glTestErrorFailedThrow();
}

inline void
glTexReadback(GLenum target, GLint level, void* pixel_array) {
	GLint type;
	GLint format;
	glGetTexLevelParameteriv(target, level, GL_TEXTURE_IMAGE_FORMAT, &format);
	glGetTexLevelParameteriv(target, level, GL_TEXTURE_IMAGE_TYPE, &type);
	glGetTexImage(target, level, format, type, pixel_array);

	glTestErrorFailedThrow();
}

inline GLuint 
glCreateTexture(GLenum target, GLint levels, GLenum internalformat, GLsizei cols, GLsizei rows, GLsizei slices,
	GLenum wrapX, GLenum wrapY, GLenum wrapZ, GLenum magfilter, GLenum minfilter) {
	GLuint _GLtexture = -1; 

	glGenTextures(1, &_GLtexture);
	glBindTexture(target, _GLtexture);
	glTexParameteri(target, GL_TEXTURE_WRAP_R, wrapX); 
	glTexParameteri(target, GL_TEXTURE_WRAP_S, wrapY); 
	glTexParameteri(target, GL_TEXTURE_WRAP_T, wrapZ);
	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magfilter); 
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minfilter);
	glTexStorageXD(target, levels, internalformat, cols, rows, slices);
	glBindTexture(target, 0);

	assert( glIsTexture(_GLtexture) );
	assert( glGetError() == GL_NO_ERROR );
	return _GLtexture;
}

inline GLuint 
glCreateTexturebuffer(GLuint buffer, GLenum internalformat) {
	GLuint _GLtexture = -1; glGenTextures(1, &_GLtexture);

	glBindTexture(GL_TEXTURE_BUFFER, _GLtexture);
	glTextureBuffer(GL_TEXTURE_BUFFER, internalformat, buffer);
	glBindTexture(GL_TEXTURE_BUFFER, 0);

	assert( glIsTexture(_GLtexture) );
	assert( glGetError() == GL_NO_ERROR );
	return _GLtexture;
}

inline void glBindTexture(GLenum unit, GLenum target, GLuint texture) {
	glActiveTexture(unit);
	glBindTexture(target, texture);
}


inline GLuint
glCreateFramebuffer(const GLenum* attachments, const GLenum* types, const GLuint* names, GLsizei size) {
	GLuint framebuffer = -1; glGenFramebuffers(1, &framebuffer);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	for (GLsizei i = 0; i != size; ++i) {
		switch (types[i]) {
		case GL_RENDERBUFFER:
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachments[i], types[i], names[i]); break;
		case GL_TEXTURE_2D:
			glFramebufferTexture2D(GL_FRAMEBUFFER, attachments[i], types[i], names[i], 0); break;
		case GL_TEXTURE_3D:
			//glFramebufferTexture3D(GL_FRAMEBUFFER, attachments[i], types[i], names[i], 0, ##ERROR##); break;
		default: break;
		}
	}
	assert( glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE );
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	assert( glGetError() == GL_NO_ERROR );
	return framebuffer;
}

inline GLuint
glCreateRenderbuffer(GLenum internalformat, GLsizei cols, GLsizei rows) {
	GLuint renderbuffer; glGenRenderbuffers(1, &renderbuffer);

	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, internalformat, cols, rows);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	assert( glGetError() == GL_NO_ERROR );
	assert( glIsRenderbuffer(renderbuffer) );
	return renderbuffer;
}



struct GLtextureDesc {
	/*<example>
		texture_descriptor _Desc;
		_Desc.cols = 100;
		_Desc.rows = 100;
		_Desc.internalformat = GL_RGBA32F;
	</example>*/
	GLenum  target = GL_TEXTURE_2D;
	GLsizei levels = 1;
	GLenum  internalformat = GL_RGBA32F;
	GLsizei cols   = 0;/* <requires> */
	GLsizei rows   = 1;/* <requires> */
	GLsizei slices = 1;
	
	GLint  samples   = 1;
	GLenum wrapX     = GL_REPEAT;
	GLenum wrapY     = GL_REPEAT;
	GLenum wrapZ     = GL_REPEAT;
	GLenum magfilter = GL_LINEAR;
	GLenum minfilter = GL_LINEAR;
};

inline GLtextureDesc glGetTextureDesc(GLenum target, GLint level) {
	GLtextureDesc desc;
	desc.target = target;
	glGetTexLevelParameteriv(target, level, GL_TEXTURE_INTERNAL_FORMAT, reinterpret_cast<GLint*>(&desc.internalformat));
	glGetTexLevelParameteriv(target, level, GL_TEXTURE_WIDTH,  &desc.cols);
	glGetTexLevelParameteriv(target, level, GL_TEXTURE_HEIGHT, &desc.rows);
	glGetTexLevelParameteriv(target, level, GL_TEXTURE_DEPTH,  &desc.slices);
	glGetTexParameteriv(target, GL_TEXTURE_WRAP_R, reinterpret_cast<GLint*>(&desc.wrapX));
	glGetTexParameteriv(target, GL_TEXTURE_WRAP_S, reinterpret_cast<GLint*>(&desc.wrapY));
	glGetTexParameteriv(target, GL_TEXTURE_WRAP_T, reinterpret_cast<GLint*>(&desc.wrapZ));
	glGetTexParameteriv(target, GL_TEXTURE_MAG_FILTER, reinterpret_cast<GLint*>(&desc.magfilter));
	glGetTexParameteriv(target, GL_TEXTURE_MIN_FILTER, reinterpret_cast<GLint*>(&desc.minfilter));
	return desc;
}

// { support reinterpret_cast, reinterpret_cast<GLtextureND(...)>(GLtexture(...)) }
class GLtexture {
public:
	GLuint identifier = static_cast<GLuint>(-1);
	GLtextureDesc descriptor;

	GLtexture() = default;
	GLtexture(const GLtexture&) = delete;
	GLtexture(GLtexture&& _Right) noexcept {
		_Right.swap(*this);
		_Right.release();
	}
	GLtexture(GLenum target, GLuint texture) {
		identifier = texture;
		glBindTexture(target, texture);
		descriptor = glGetTextureDesc(target, 0);
		glBindTexture(target, 0);
	}
	explicit GLtexture(const GLtextureDesc& desc)
		: identifier(glCreateTexture(desc.target, desc.levels, desc.internalformat, desc.cols, desc.rows, desc.slices, desc.wrapX, desc.wrapY, desc.wrapZ, desc.magfilter, desc.minfilter)), 
		descriptor(desc) {}
	~GLtexture() { release(); }

	GLtexture& operator=(const GLtexture&) = delete;
	GLtexture& operator=(GLtexture&& _Right) noexcept {
		_Right.swap(*this);
		_Right.release();
		return *this;
	}

	operator GLuint() const {
		return identifier;
	}
	bool operator==(const GLtexture& _Right) const {
		return this->identifier == _Right.identifier;
	}
	bool operator!=(const GLtexture& _Right) const {
		return this->identifier != _Right.identifier;
	}

	void swap(GLtexture& _Right) {
		std::swap(identifier, _Right.identifier);
		std::swap(descriptor, _Right.descriptor);
	}
	virtual void release() {
		if (glIsTexture(identifier)) { glDeleteTextures(1, &identifier); }
		identifier = -1;
	}
	bool valid() const {
		return glIsTexture(identifier);
	}
	GLenum target() const {
		return descriptor.target;
	}
	GLsizei cols() const {
		return descriptor.cols;
	}
	GLsizei rows() const {
		return descriptor.rows;
	}
	GLsizei slices() const {
		return descriptor.slices;
	}
};

// { cols*rows }
class GLtexture2D : public GLtexture {
	using _Mybase = GLtexture;
public:
	GLtexture2D() = default;
	GLtexture2D(const GLtexture2D&) = delete;
	GLtexture2D(GLtexture2D&& _Right) noexcept
		: _Mybase(std::move(_Right)) {}
	GLtexture2D(GLenum internalformat, GLsizei cols, GLsizei rows, GLsizei levels = 1)
		: _Mybase(GLtextureDesc{ GL_TEXTURE_2D, levels, internalformat, cols, rows }) {}
	explicit GLtexture2D(const GLtextureDesc& desc)
		: _Mybase(desc) {}
	explicit GLtexture2D(GLuint texture)
		: _Mybase(GL_TEXTURE_2D, texture) {}

	GLtexture2D& operator=(const GLtexture2D&) = delete;
	GLtexture2D& operator=(GLtexture2D&& _Right) noexcept {
		_Mybase::operator=(std::move(_Right));
		return *this;
	}

	void upload(GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, const void* pixels, GLint level = 0) {// unsupport SNORM
		glBindTexture(this->target(), this->identifier);
		glTexUpload2D(this->target(), level, xoffset, yoffset, width, height, pixels);
		glBindTexture(this->target(), 0);
	}
	void upload(GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels, GLint level = 0) {
		glBindTexture(this->target(), this->identifier);
		glTexSubImage2D(this->target(), level, xoffset, yoffset, width, height, format, type, pixels);
		glBindTexture(this->target(), 0);
	}
};

// { GLtexture3D[ n ] }
class _GLslice2d : public GLtexture2D {
	using _Mybase = GLtexture2D;
public:
	//using iterator_category = std::random_access_iterator_tag;
	using difference_type = GLint;

	GLint indexZ = 0;

	_GLslice2d(const _GLslice2d&) = delete;
	_GLslice2d(GLuint texture, GLtextureDesc desc, GLint idx) {
		this->identifier = texture;
		this->descriptor = desc;
		this->indexZ     = idx;
	}
	_GLslice2d(GLenum target, GLuint texture, GLint idx) {
		this->identifier = texture;
		assert(glIsTexture(texture));

		glBindTexture(target, texture);
		this->descriptor = glGetTextureDesc(target, texture);
		glBindTexture(target, 0);
		
		this->indexZ     = idx;
	}
	_GLslice2d& operator=(const _GLslice2d&) = default;
	
	virtual void release() override {
		// do nothing
	}

	template<typename _Tex2Dty>
	void _Copy_from(const _Tex2Dty& _Right) {
		assert( this->valid()  );
		assert( _Right.valid() );
		assert( this->cols() == _Right.cols() );
		assert( this->rows() == _Right.rows() );
		assert( this->descriptor.internalformat == _Right.descriptor.internalformat );
			
		std::unique_ptr<GLbyte> temp = std::unique_ptr<GLbyte>(new GLbyte[_Right.cols() * _Right.rows() * sizeof(float)*4]);
			
		glBindTexture(_Right.target(), _Right.identifier);
		glTexReadback(_Right.target(), 0, temp.get());
		glBindTexture(_Right.target(), 0);

		glBindTexture(this->target(), this->identifier);
		glTexUpload3D(this->target(), 0,  0,0,indexZ, this->cols(),this->rows(),1, temp.get());
		glBindTexture(this->target(), 0);

		/* // requires hardware optimize
		glCopyImageSubData(_Right.identifier, _Right.descriptor.target, 0, 0, 0, 0,
					        this->identifier,  this->descriptor.target, 0, 0, 0, this->index,
					            this->cols(),  this->rows(), 1 );*/
	}
	void copy_from(const GLtexture2D& _Right) {
		_Copy_from(_Right);
	}
	void copy_from(const _GLslice2d& _Right) {
		_Copy_from(_Right);
	}
	void upload(GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, const void* pixels, GLint level = 0) {
		glBindTexture(this->target(), this->identifier);
		glTexUpload3D(this->target(), level, xoffset, yoffset, indexZ, width, height, 1, pixels);
		glBindTexture(this->target(), 0);
	}

	_GLslice2d operator+(difference_type _Diff) const {
		assert( indexZ + _Diff <= this->slices() );// contain end()
		assert( indexZ + _Diff >= 0 );
		return _GLslice2d(this->identifier, this->descriptor, indexZ + _Diff);
	}
	_GLslice2d operator-(difference_type _Diff) const {
		return (*this) + (-_Diff);
	}

	_GLslice2d& operator+=(difference_type _Diff) {
		assert( indexZ + _Diff < this->slices() );
		assert( indexZ + _Diff >= 0 );
		indexZ += _Diff;
		return *this;
	}
	_GLslice2d& operator-=(difference_type _Diff) {
		return (*this) += (-_Diff);
	}

	bool operator==(const _GLslice2d& _Right) const {
		return _Mybase::operator==(_Right) && indexZ == _Right.indexZ;
	}
	bool operator!=(const _GLslice2d& _Right) const {
		return !(*this == _Right);
	}
};

// { cols*rows*slices }
class GLtexture3D : public GLtexture {
	using _Mybase = GLtexture;
public:
	GLtexture3D() = default;
	GLtexture3D(const GLtexture3D&) = delete;
	GLtexture3D(GLtexture3D&& _Right) noexcept
		: _Mybase(std::move(_Right)) {}
	GLtexture3D(GLenum internalformat, GLsizei cols, GLsizei rows, GLsizei slices, GLsizei levels = 1)
		: _Mybase(GLtextureDesc{ GL_TEXTURE_3D, levels, internalformat, cols, rows, slices }) {}
	explicit GLtexture3D(const GLtextureDesc& desc)
		: _Mybase(desc) {}
	explicit GLtexture3D(GLuint texture)
		: _Mybase(GL_TEXTURE_3D, texture) {}

	GLtexture3D& operator=(const GLtexture3D&) = delete;
	GLtexture3D& operator=(GLtexture3D&& _Right) noexcept {
		_Mybase::operator=(std::move(_Right));
		return *this;
	}

	void upload(GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, const void* pixels, GLint level = 0) {
		glBindTexture(this->target(), this->identifier);
		glTexUpload3D(this->target(), level, xoffset, yoffset, zoffset, width, height, depth, pixels);
		glBindTexture(this->target(), 0);
	}
	void upload(GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels, GLint level = 0) {
		glBindTexture(this->target(), this->identifier);
		glTexSubImage3D(this->target(), level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
		glBindTexture(this->target(), 0);
	}

	_GLslice2d begin() const {
		return _GLslice2d(this->identifier, this->descriptor, 0);
	}
	_GLslice2d end() const {
		return _GLslice2d(this->identifier, this->descriptor, this->slices());
	}
};


struct GLframebufferDesc {
	GLsizei texture_size    = 0;
	GLenum  attachments[20] = { GL_NONE };
	GLenum  types[20]       = { GL_NONE };
	GLuint  names[20]       = { GL_NONE };
};

inline GLframebufferDesc glGetFramebufferDesc(GLuint target) {
	GLframebufferDesc desc;

	// 2. Get all GL_COLOR_ATTACHMENT information
	GLint _MAX_COLOR_ATTACHMENTS = 0;
	glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &_MAX_COLOR_ATTACHMENTS);
	for (GLint i = 0; i != _MAX_COLOR_ATTACHMENTS; ++i) {
		desc.texture_size += 1;
		desc.attachments[i] = GL_COLOR_ATTACHMENT0 + i;
		glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, reinterpret_cast<GLint*>(desc.types + i));
		glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, reinterpret_cast<GLint*>(desc.names + i));
	}
	
	// 3. Get GL_DEPTH_STENCIL_ATTACHMENT or GL_DEPTH_ATTACHMENT and GL_STENCIL_ATTACHMENT information 
	{
		GLint type = GL_NONE, name = GL_NONE;
		glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
		glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
		if (glGetError() == GL_INVALID_OPERATION || (type != GL_NONE && name != GL_NONE)) {
			desc.attachments[desc.texture_size] = GL_DEPTH_STENCIL_ATTACHMENT;
			desc.types[desc.texture_size]       = type;
			desc.names[desc.texture_size]       = name;
			desc.texture_size += 1;
		} else {
			type = name = GL_NONE;
			glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
			glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
			desc.attachments[desc.texture_size] = GL_DEPTH_ATTACHMENT;
			desc.types[desc.texture_size]       = type;
			desc.names[desc.texture_size]       = name;
			desc.texture_size += 1;

			type = name = GL_NONE;
			glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
			glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
			desc.attachments[desc.texture_size] = GL_STENCIL_ATTACHMENT;
			desc.types[desc.texture_size]       = type;
			desc.names[desc.texture_size]       = name;
			desc.texture_size += 1;
		}
	}

	return desc;
}


/*<version 1.0>
	enum GLxxxx_xxx{ ... }
	...
	RESULT glXxxxYyyy(...)
	...
	class GLxxxxxx{ }
</version 1.0>*/

/*<version 2.0>
	<first> new library idea{ nonmath should full ObjectOrientedProgram } </first>
	<second> guide non enum  </second>
	<third> minimize dependency, even won't allow [size_t] </third>
</version 2.0>*/