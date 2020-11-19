//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
//--------------------------------------------------------------------------------------
#pragma once
#ifndef clmagic_calculation_CLMAGIC_h_
#define clmagic_calculation_CLMAGIC_h_
#include <string>

namespace calculation
{
	/* @_clmagic: library */
}

/* reference */
//#define CLMAGIC_USING_MKL 1
#define CLMAGIC_USING_DXmatH 1

/* Intel math kernel library */
#if defined(CLMAGIC_USING_MKL)
#include <mkl.h>
#endif

/* MicrosoftDirectX math library */
#if defined(CLMAGIC_USING_DXmatH)
#include <DirectXmath.h>
#include <DirectXCollision.h>
#endif

/* GLM math library */
#if defined(CLMAGIC_USING_GLM)
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#endif

/* please enter type */
#ifndef _in
#define _in(...) const __VA_ARGS__&
#endif

#ifndef _inout
#define _inout(...) __VA_ARGS__&
#endif

#ifndef _out
#define _out(...) __VA_ARGS__&
#endif

#endif