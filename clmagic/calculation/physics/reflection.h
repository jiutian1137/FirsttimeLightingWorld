#pragma once
#ifndef clmagic_calculation_physic_REFLECTION_h_
#define clmagic_calculation_physic_REFLECTION_h_
#include "../lapack/vector.h"
#include "../general/general.h"

namespace clmagic {
	template<typename _Ts, typename _Tb>
	VECTOR3<_Ts, _Tb> reflect(const VECTOR3<_Ts, _Tb>& L, const unit_vector3<_Ts, _Tb>& N) {// get reflect-vector
		/*<figure>
			/|\
		   / 2.0f * proj(L, N)
		  /  |  \
		 /   |
		/____|    \
		\    |    /
		 L   |   / Result
		  \  N  /
		   \ | /
		____\|/_________
		</figure>*/
		return static_cast<_Ts>(2) * dot(L,N)*N - L;
	}
	template<typename _Ts, typename _Tb>
	unit_vector3<_Ts, _Tb> reflect(const unit_vector3<_Ts, _Tb>& L, const unit_vector3<_Ts, _Tb>& N) {// get reflect-unit_vector
		return unit_vector3<_Ts, _Tb>(static_cast<_Ts>(2) * dot(L, N) * N - L, true);
	}


	/*<Reference> RealTimeRendering-4th-Edition.page324 </Reference>*/
	template<typename _Ts>
	struct substance {
		VECTOR3<_Ts> specular_color;
		VECTOR3<_Ts> diffuse_color;// or diffuse_albedo
	};

	/*<Reference> RealTimeRendering-4th-Edition.page324 </Reference>*/
	template<typename _Ts>
	struct substance_metalness {
		VECTOR3<_Ts> surface_color;
		_Ts metalness;
	};


	template<typename _Ts, typename _Tb>
	struct Fresnel_reflectance {
		/* n1: index of refraction on the outside
		   n2: index of refraction on the inside
		*/
		_Ts external_reflectance_value(_Ts n1, _Ts n2) {
			return pow((n2 - n1) / (n2 + n1), (_Ts)2);
		}
	};

}// namespace clmagic

#endif