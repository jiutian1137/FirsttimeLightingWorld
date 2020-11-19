#pragma once
#ifndef glut_NONCOPYABLE_h_
#define glut_NONCOPYABLE_h_

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif
#include "GL/glew.h"
#include <cassert>
#include <exception>

inline 
void glTestErrorFailedThrow() {
#ifdef NDEBUG
	// empty
#else
	GLenum _Error = glGetError();
	switch (_Error) {
		case GL_NO_ERROR: break;
		case GL_INVALID_ENUM: throw std::exception("GL_INVALID_ENUM");
		case GL_INVALID_VALUE: throw std::exception("GL_INVALID_VALUE");
		case GL_INVALID_OPERATION: throw std::exception("GL_INVALID_OPERATION");
		case GL_INVALID_FRAMEBUFFER_OPERATION: throw std::exception("GL_INVALID_FRAMEBUFFER_OPERATION");
		case GL_OUT_OF_MEMORY: throw std::exception("GL_OUT_OF_MEMORY");
		default: break;
	}
#endif
}

#endif
