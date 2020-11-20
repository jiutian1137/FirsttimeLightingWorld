//--------------------------------------------------------------------------------------
// Copyright (c) 2020 LongJiangnan
// All Rights Reserved
// Apache License 2.0
//--------------------------------------------------------------------------------------
// https://mathshistory.st-andrews.ac.uk/Biographies/
#pragma once
#ifndef clmagic_calculation_lapack_GEOMETRY_h_
#define clmagic_calculation_lapack_GEOMETRY_h_
#include "vector.h"
#include "matrix.h"
#include "algorithm.h"
#include "../complex/quaternion.h"
#include "../physics/fundamental.h"

/*<coordinate>
			   y
			   /|\    _
				|     /| z
				|   /
				| /
	------------+----------->>
			  / |            x
			/	|
		  /	    |
		/		|
	x: Right vector
	y: Up vector, X-Y is humen's viewport
	z: Every time we move forward, it is like a playing animation
	note: I kown the right-hand-coordinate is very popular and cross-product is right-hand, but it very easy to
		bring meaningless and difficult problems.
</coordinate>*/

namespace calculation {

	/*<barycentric>
	<idea>  A*u + B*v + C*(1-u-v) = P
		A*u + B*v + C - C*u - C*v = P      : C*(1-u-v) = C-C*u-C*v
				(A-C)*u + (B-C)*v = P - C  : group by (u,v)
				 v0         v1       v2

		equaltion:
			dot(v0*u + v1*v, v0) = dot(v2, v0)
			dot(v0*u + v1*v, v1) = dot(v2, v1)
			=> 
			dot(v0,v0)*u + dot(v1,v0)*v = dot(v2, v0) : vector-form to scalar-form
			dot(v0,v1)*u + dot(v1,v1)*v = dot(v2, v1)
	</idea>
	</barycentric>*/

	template<typename point_t, typename real_t> inline
	point_t barycentric_interpolate(const point_t& A, const point_t& B, const point_t& C, real_t u, real_t v) {
		// barycentric interpolate a Triangle(A,B,C)
		return A*u + B*v + C*(1 - u - v);
	}

	template<typename point_t, typename real_t> inline
	point_t barycentric_interpolate(const point_t& A, const point_t& B, const point_t& C, const point_t& D, real_t u, real_t v, real_t w) {
		// barycentric interpolate a Volume(A,B,C,D)
		return A*u + B*v + C*w + D*(1 - u - v - w);
	}

	template<typename point_t, typename real_t> inline
	void barycentric_solve(const point_t& A, const point_t& B, const point_t& C, const point_t& P, real_t& u, real_t& v, real_t& w) {
		using vec3_t = decltype(A - C);

		// Declare vector
		const vec3_t v0 = A - C;
		const vec3_t v1 = B - C;
		const vec3_t v2 = P - C;
		// Compute matrix element
		const real_t d00 = dot(v0, v0);
		const real_t d01 = dot(v0, v1);
		const real_t d02 = dot(v0, v2);
		const real_t d11 = dot(v1, v1);
		const real_t d12 = dot(v1, v2);
		// Solve matrix using Cramer-rule
		u = d02 * d11 - d01 * d12;
		v = d00 * d12 - d02 * d01;
		w = 1 - u - v;
	}
	
	template<typename point_t, typename _RealTy> inline
	void barycentric_solve(const point_t& A, const point_t& B, const point_t& C, const point_t& D, const point_t& P, _RealTy& u, _RealTy& v, _RealTy& w, _RealTy& x) {
		/*<idea>     A*u + B*v + C*w + D*(1-u-v-w) = P
		     A*u + B*v + C*w + D - D*u - D*v - D*w = P      : D*(1-u-v-w) = D-D*u-D*v-D*w
				       (A-D)*u + (B-D)*v + (C-D)*w = P - D  : group by (u,v,w)
					    v0         v1       v2        v3

			equaltion:
				dot(v0*u + v1*v + v2*w, v0) = dot(v3, v0)
				dot(v0*u + v1*v + v2*w, v1) = dot(v3, v1)
				dot(v0*u + v1*v + v2*w, v2) = dot(v3, v2)
				=> 
				dot(v0,v0)*u + dot(v1,v0)*v + dot(v2,v0) = dot(v3, v0) : vector-form to scalar-form
				dot(v0,v1)*u + dot(v1,v1)*v + dot(v2,v1) = dot(v3, v1)
				dot(v0,v2)*u + dot(v1,v2)*v + dot(v2,v2) = dot(v3, v2)
		</idea>*/
		assert(false);

		const auto v0 = A - D;
		const auto v1 = B - D;
		const auto v2 = C - D;
		const auto v3 = P - D;
		
		const auto d00 = dot(v0, v0);
		const auto d01 = dot(v0, v1);
		const auto d02 = dot(v0, v2);
		const auto d03 = dot(v0, v3);
		const auto d11 = dot(v1, v1);
		const auto d12 = dot(v1, v2);
		const auto d13 = dot(v1, v3);
		const auto d22 = dot(v2, v2);
		const auto d23 = dot(v2, v3);
		x = 1 - u - v - w;
	}

	template<typename _Ty>
	concept ray = requires(_Ty __r) { __r.origin; __r.direction; };

	template<typename _Ty>
	concept segment = requires(_Ty __sg) { __sg.origin; __sg.direction; __sg.start; __sg.end; };

	template<typename _Ty>
	concept sphere = requires(_Ty __sph) { __sph.center; __sph.radius; };

	template<typename _Ty>
	concept plane = requires(_Ty __pl) { __pl.normal; __pl.distance; };
	
	template<typename _Ty>
	concept AABB = requires(_Ty __bx) { __bx.min_point; __bx.max_point; };

	template<typename _Ty>
	concept OBB = requires(_Ty __bx) { __bx.center; __bx.oriented;  __bx.halfwidth; };


	/*<Reference type="book">
		{ Real-Time Collision Detection }
	</Reference>*/

	/*<Desirable Bounding Volume Characteristics>
		* Inexpensive intersection tests
		* Tight fitting
		* Inexpensive to compute
		* Easy to rotate and transform
		* Use little memory
	</Desirable Bounding Volume Characteristics>*/


	template<typename point_t, typename vector_t, typename length_t>
	length_t ray_intersect_sphere(point_t ray_origin, vector_t ray_direction, point_t sph_center, length_t sph_radius) {
		vector_t center_to_origin = (ray_origin - sph_center);
		length_t lowest_distance, 
				 max_distance;
		bool intersect = quadratic_accurate(
			dot(ray_direction, ray_direction),
			dot(center_to_origin, ray_direction) * static_cast<length_t>(2),
			dot(center_to_origin, center_to_origin) - sph_radius*sph_radius,
			lowest_distance, max_distance
		);

		return intersect ? min(max(lowest_distance, static_cast<length_t>(0)), max_distance) : std::numeric_limits<length_t>::quiet_NaN();
		/*<idea> length_sqrare((ray_origin + ray_direction * t) - sph_center) = sph_radius * sph_radius : x*x + y*y + z*z = R*R
			dot((O+D*t) - C, (O+D*t) - C) = R * R
			   dot(O-C + D*t, O-C + D*t)  = R * R
			dot(O-C,O-C) + dot(D*t,D*t) + 2*(O-C)*(D*t) = R * R
			dot(D,D)*pow(t,2) + dot(O-C,D)*2*t + (dot(O-C,O-C) - R*R) = 0 
			    A                    B                    C
		</idea>*/
	}

	template<typename point_t, typename vector_t, typename length_t>
	length_t ray_intersect_sphere_surface(point_t ray_origin, vector_t ray_direction, point_t sph_center, length_t sph_radius) {
		vector_t center_to_origin = (ray_origin - sph_center);
		length_t lowest_distance, 
				 max_distance;
		bool intersect = quadratic_accurate(
			dot(ray_direction, ray_direction), 
			dot(center_to_origin, ray_direction) * static_cast<length_t>(2),
			dot(center_to_origin, center_to_origin) - sph_radius*sph_radius, 
			lowest_distance, max_distance
		);

		return intersect ? (lowest_distance >= static_cast<length_t>(0) ? lowest_distance : max_distance) : std::numeric_limits<length_t>::quiet_NaN();
	}

	template<typename point_t, typename vector_t, typename length_t = decltype(point_t()[0] - point_t()[0])>
	length_t ray_intersect_aabox(point_t ray_origin, vector_t ray_direction, point_t box_min, point_t box_max) {
		vector_t test_lowside  = (box_min - ray_origin) / ray_direction;
		vector_t test_highside = (box_max - ray_origin) / ray_direction;
				 test_lowside  = cast_nan(test_lowside, std::numeric_limits<length_t>::lowest());
				 test_highside = cast_nan(test_highside, std::numeric_limits<length_t>::max());

		vector_t lowest_distances = min(test_lowside, test_highside);
		vector_t max_distances    = max(test_lowside, test_highside);
		length_t lowest_distance  = max(lowest_distances[0], max(lowest_distances[1], lowest_distances[2]));
		length_t max_distance     = min(max_distances[0], min(max_distances[1], max_distances[2]));
		bool intersect = lowest_distance <= max_distance;

		return intersect ? min( max(lowest_distance, static_cast<length_t>(0)), max_distance )
			: std::numeric_limits<length_t>::quiet_NaN();	
		/*<idea> t = (d - dot(N,O))/dot(N,D) 
				 t = (d - O[i])/D[i]  : axis aligned normal
				 and IEEE754-floating-div-ZERO is "inf", unsupport integer vector
				 and IEEE754-floating_Zero-div-Zero is "nan"
		</idea>*/
	}
	
	template<typename point_t, typename vector_t, typename length_t>
	length_t ray_intersect_plane(point_t ray_origin, vector_t ray_direction, vector_t pl_normal, length_t pl_distance) {
		return (pl_distance - dot(pl_normal, ray_origin)) / dot(pl_normal, ray_direction);
		/*<idea> dot(pl_normal, ray_origin+ray_direction*t) = pl_distance
			      dot(N, O + D*t) = d
			dot(N,O) + dot(N,D*t) = d      : distributive law
			dot(N,O) + dot(N,D)*t = d      : scalar-product scalar law
			t = (d - dot(N,O))/dot(N,D)
		</idea>*/
	}


	template<typename point_t>
	bool aabox_test_aabox(point_t left_min, point_t left_max, point_t right_min, point_t right_max) {
		if (left_max[0] < right_min[0] || left_min[0] > right_min[0]) { return false; }
		if (left_max[1] < right_min[1] || left_min[1] > right_min[1]) { return false; }
		if (left_max[2] < right_min[2] || left_min[2] > right_min[2]) { return false; }
		return true;
	}

	template<typename point_t, typename length_t>
	bool sphere_test_sphere(point_t left_center, length_t left_radius, point_t right_center, length_t right_radius) {
		auto     distance_vector = right_center - left_center;
		length_t sqr_distance    = dot(distance_vector, distance_vector);
		length_t total_radius    = left_radius + right_radius;
		return sqr_distance <= (total_radius*total_radius);
	}
	
	template<typename point_t, typename vector_t, typename length_t>
	bool ray_test_sphere(point_t ray_origin, vector_t ray_direction, point_t sph_center, length_t sph_radius) {
		vector_t center_to_origin = ray_origin - sph_center;

		length_t c = dot(center_to_origin, center_to_origin) - sph_radius * sph_radius;
		if (c <= static_cast<length_t>(0)) { return true; }
		
		length_t b = dot(center_to_origin, ray_direction);
		if (b >= static_cast<length_t>(0)) { return false; }
		
		length_t disc = b * b - c;
		if (disc < static_cast<length_t>(0)) { return false; }
		
		return true;
	}

	template<typename point_t, typename vector_t, typename length_t> inline
	bool ray_test_plane(point_t r_origin, vector_t r_direction, vector_t pl_normal, length_t pl_distance) {
		return (pl_distance - dot(pl_normal, r_origin)) / dot(pl_normal, r_direction) >= static_cast<length_t>(0);
	}

	template<typename point_t, typename length_t>
	bool aabox_test_sphere(point_t box_min, point_t box_max, point_t sph_center, length_t sph_radius) {
		using area_t = decltype( dot(sph_center - box_max, sph_center - box_max) );

		point_t closest_point = clamp(sph_center, box_min, box_max);
		area_t square_distance = dot(closest_point - sph_center, closest_point - sph_center);
		return square_distance <= (sph_radius * sph_radius);
	}


	template<typename point_t>
	point_t closest_point_aabox(point_t the_point, point_t box_min, point_t box_max) {
		point_t closest_point = clamp(the_point, box_min, box_max);
		return closest_point;
	}


	template<typename result_t, ray ray_t, plane plane_t> inline
	result_t ray_intersect_plane(ray_t r, plane_t pl) {
		return ray_intersect_plane(r.origin, r.direction, pl.normal, pl.distance);
	}

	template<ray ray_t, plane plane_t> inline 
	bool ray_test_plane(ray_t r, plane_t pl) {
		return ray_test_plane(r.origin, r.direction, pl.normal, pl.distance);
	}



	

	/*template<typename point_t, typename vector_t, typename real_t, typename real_t2>
	test_state intersect_segment_sphsurf(point_t sg_origin, vector_t sg_direction, real_t sg_start, real_t sg_end, point_t sph_center, real_t sph_radius, real_t2& t) {
		vector_t center_to_origin = sg_origin - sph_center;
		real_t t0, t1;
		if ( quadratic( dot(sg_direction, sg_direction), 
						dot(center_to_origin, sg_direction) * static_cast<real_t>(2),
						dot(center_to_origin, center_to_origin) - sph_radius*sph_radius,
					    t0, t1) ) {
			if (sg_start <= t0 && t0 <= sg_end) {
				t = static_cast<real_t2>(t0); return hit;
			} else if (sg_start <= t1 && t1 <= sg_end) {
				t = static_cast<real_t2>(t1); return hit;
			}
		}

		return no_hit;
	}

	template<segment segment_t, sphere sphere_t, typename real_t> inline
	test_state intersect_segment_sphsurf(segment_t sg, sphere_t sph, real_t& t) {
		return intersect_segment_sphsurf(sg.origin, sg.direction, sg.start, sg.end, sph.center, sph.radius, t);
	}

	template<typename point_t, typename vector_t, typename real_t> inline
	test_state test_segment_sphsurf(point_t sg_origin, vector_t sg_direction, real_t sg_start, real_t sg_end, point_t sph_center, real_t sph_radius) {
		real_t unuse;
		return intersect_segment_sphsurf(sg_origin, sg_direction, sg_start, sg_end, sph_center, sph_radius, unuse);
	}

	template<segment segment_t, sphere sphere_t> inline
	test_state test_segment_sphsurf(segment_t sg, sphere_t sph) {
		return test_segment_sphsurf(sg.origin, sg.direction, sg.start, sg.end, sph.center, sph.radius);
	}


	template<typename point_t, typename vector_t, typename real_t, typename real_t2> inline
	test_state intersect_segment_plane(point_t sg_origin, vector_t sg_direction, real_t sg_start, real_t sg_end, vector_t pl_normal, real_t pl_distance, real_t2& t) {
		intersect_ray_plane(sg_origin, sg_direction, pl_normal, pl_distance, t);
		return static_cast<real_t>(sg_start) <= t && t <= static_cast<real_t>(sg_end);
	}

	template<segment segment_t, plane plane_t, typename real_t> inline
	test_state intersect_segment_plane(segment_t sg, plane_t pl, real_t& t) {
		return intersect_segment_plane(sg.origin, sg.direction, sg.start, sg.end, pl.normal, pl.distance, t);
	}

	template<typename point_t, typename vector_t, typename real_t> inline
	test_state test_segment_plane(point_t sg_origin, vector_t sg_direction, real_t sg_start, real_t sg_end, vector_t pl_normal, real_t pl_distance) {
		real_t t;
		intersect_ray_plane(sg_origin, sg_direction, pl_normal, pl_distance, t);
		return sg_start <= t && t <= sg_end;
	}

	template<segment segment_t, plane plane_t> inline
	test_state test_segment_plane(segment_t sg, plane_t pl) {
		return test_segment_plane(sg.origin, sg.direction, sg.start, sg.end, pl.normal, pl.distance);
	}*/



	//template<ray ray_t, AABB AABB_t, typename real_t2> inline
	//test_state intersect_ray_AABB(ray_t r, AABB_t bx, real_t2& t) {
	//	return intersect_ray_AABB(r.origin, r.direction, bx.min_point, bx.max_point, t);
	//}

	//template<typename point_t, typename vector_t> inline
	//test_state test_ray_AABB(point_t ray_origin, vector_t ray_direction, point_t bx_min, point_t bx_max) {
	//	using real_t = decltype(bx_max[0] - bx_min[0]);
	//	real_t unuse;
	//	return intersect_ray_AABB(ray_origin, ray_direction, bx_min, bx_max, unuse);
	//}

	//template<ray ray_t, AABB AABB_t> inline
	//test_state test_ray_AABB(ray_t r, AABB_t bx) {
	//	return test_ray_AABB(r.origin, r.direction, bx.min_point, bx.max_point);
	//}


	//template<typename point_t, typename vector_t, typename real_t, typename real_t2>
	//test_state intersect_segment_AABB(point_t sg_origin, vector_t sg_direction, real_t sg_start, real_t sg_end, point_t bx_min, point_t bx_max, real_t2& t) {
	//	if ( intersect_ray_AABB(sg_origin + sg_direction * sg_start, sg_direction, bx_min, bx_max, t) ) {
	//		if (t <= static_cast<real_t2>(sg_end - sg_start)) {
	//			return hit;
	//		}
	//	}

	//	return no_hit;
	//}

	//template<segment segment_t, AABB AABB_t, typename real_t>
	//test_state intersect_segment_AABB(segment_t sg, AABB_t bx, real_t& t) {
	//	return intersect_segment_AABB(sg.origin, sg.direction, sg.start, sg.end, bx.min_point, bx.max_point, t);
	//}






	template<typename _SclTy, typename _AngTy> inline
	_SclTy _Get_radians(_AngTy angle) {
#ifdef clmagic_calculation_physics_FUNDAMENTAL
		using radians_type = angle_t<_SclTy, radian>;
		return static_cast<_SclTy>( static_cast<radians_type>( angle ) );
#else
		return angle;
#endif
	}

	template<typename _AngTy, typename _SclTy> inline
	_AngTy _To_angle(_SclTy _Radval) {
#ifdef clmagic_calculation_physics_FUNDAMENTAL
		using radians_type = angle_t<_SclTy, radian>;
		return static_cast<_AngTy>( static_cast<radians_type>( _Radval ) );
#else
		return _Radval;
#endif
	}

}// namespace clmagic


/*<transform>
	<idea>
		+------------+-----------+------------------+---------------+
		 |            |translate  |      rotate     |     idea     |
		+------------+-----------+------------------+---------------+
		| left-hand  |  forward  |        clockwise | toward-Future |
		| right-hand | backward  | counterclockwise | Time-reversal |
	</idea>
</transform>*/

namespace Euler {

	/*<rotation>
		<equation>  [r * cos(a + theta)]   [? ?]   [vy]
					[r * sin(a + theta)] = [? ?] * [vx]
		</equation>
		<explan>
					  [r * cos(a + theta)]
			(vx,vy) = [r * sin(a + theta)]

			=> [ r * ( cos(a)cos(theta) - sin(a)sin(theta) ) ]
			   [ r * ( sin(a)cos(theta) + cos(a)sin(theta) ) ]

			=> [ (r * cos(a)) * cos(theta) - (r * sin(a)) * sin(theta) ] : add_swap, unpack
			   [ (r * cos(a)) * sin(theta) + (r * sin(a)) * cos(theta) ]

			=> [ vx * cos(theta) - vy * sin(theta) ]
			   [ vx * sin(theta) + vy * cos(theta) ]

			=> [vx, vy] * [  cos(theta),   sin(theta)]
						  [- sin(theta), + cos(theta)]

			=> [cos(theta), - sin(theta)] * [vx] : transpose(T), reinterpret_cast_colvector(v)
			   [sin(theta), + cos(theta)]   [vy]
		</explan>
	</rotation>*/

	// { rotation for axis{ 0,1,0 } and angle }
	template<typename _MatTy, typename _RadTy>
	_MatTy rotation_yaw(_RadTy angle) {
		using T = calculation::matrix_scalar_t<_MatTy>;

		T yaw    = calculation::_Get_radians<T>( angle );
		T cosYaw = cos(yaw);
		T sinYaw = sin(yaw);
		return _MatTy{ cosYaw, (T)(0), sinYaw,
					   (T)(0), (T)(1), (T)(0),
					  -sinYaw, (T)(0), cosYaw  };
	}

	// { rotation for axis{ 0,1,0 } and angle }
	template<typename _MatTy, typename _RadTy>
	_MatTy rotation4x4_yaw(_RadTy angle) {
		using T = calculation::matrix_scalar_t<_MatTy>;

		T yaw    = calculation::_Get_radians<T>( angle );
		T cosYaw = cos(yaw);
		T sinYaw = sin(yaw);
		return _MatTy{ cosYaw, (T)(0), sinYaw, (T)(0),
					   (T)(0), (T)(1), (T)(0), (T)(0),
					  -sinYaw, (T)(0), cosYaw, (T)(0),
					   (T)(0), (T)(0), (T)(0), (T)(1) };
	}

	// { rotation for axis{ 1,0,0 } and angle }
	template<typename _MatTy, typename _RadTy>
	_MatTy rotation_pitch(_RadTy angle) {
		using T = calculation::matrix_scalar_t<_MatTy>;

		T pitch    = calculation::_Get_radians<T>( angle );
		T cosPitch = cos(pitch);
		T sinPitch = sin(pitch);
		return _MatTy{ (T)(1),   (T)(0),    (T)(0),
					   (T)(0), cosPitch, -sinPitch,
					   (T)(0), sinPitch,  cosPitch };
	}

	// { rotation for axis{ 1,0,0 } and angle }
	template<typename _MatTy, typename _RadTy>
	_MatTy rotation4x4_pitch(_RadTy angle) {
		using T = calculation::matrix_scalar_t<_MatTy>;

		T pitch    = calculation::_Get_radians<T>( angle );
		T cosPitch = cos(pitch);
		T sinPitch = sin(pitch);
		return _MatTy{ (T)(1),   (T)(0),    (T)(0), (T)(0),
					   (T)(0), cosPitch, -sinPitch, (T)(0),
					   (T)(0), sinPitch,  cosPitch, (T)(0),
					   (T)(0),   (T)(0),    (T)(0), (T)(1) };
	}

	// { rotation for axis{ 0,0,1 } and angle }
	template<typename _MatTy, typename _RadTy>
	_MatTy rotation_roll(_RadTy angle) {
		using T = calculation::matrix_scalar_t<_MatTy>;

		T roll    = calculation::_Get_radians<T>( angle );
		T cosRoll = cos(roll);
		T sinRoll = sin(roll);
		return _MatTy{ cosRoll, -sinRoll, (T)(0),
					   sinRoll,  cosRoll, (T)(0),
						(T)(0),   (T)(0), (T)(1) };
	}

	// { rotation for axis{ 0,0,1 } and angle }
	template<typename _MatTy, typename _RadTy>
	_MatTy rotation4x4_roll(_RadTy angle) {
		using T = calculation::matrix_scalar_t<_MatTy>;

		T roll    = calculation::_Get_radians<T>( angle );
		T cosRoll = cos(roll);
		T sinRoll = sin(roll);
		return _MatTy{
			cosRoll, -sinRoll, (T)(0), (T)(0),
			sinRoll,  cosRoll, (T)(0), (T)(0),
			 (T)(0),   (T)(0), (T)(1), (T)(0),
			 (T)(0),   (T)(0), (T)(0), (T)(1)
		};
	}

	template<typename _MatTy, typename _RadTy>
	_MatTy rotation(_RadTy _Yangle, _RadTy _Pangle, _RadTy _Rangle) {
		// rotation for combination of {yaw, pitch, roll}
		using scalar_type = calculation::matrix_scalar_t<_MatTy>;

		scalar_type yaw      = calculation::_Get_radians<scalar_type>( _Yangle );
		scalar_type cosYaw   = cos(yaw);
		scalar_type sinYaw   = sin(yaw);

		scalar_type pitch    = calculation::_Get_radians<scalar_type>( _Pangle );
		scalar_type cosPitch = cos(pitch);
		scalar_type sinPitch = sin(pitch);
			
		scalar_type roll     = calculation::_Get_radians<scalar_type>( _Rangle );
		scalar_type cosRoll  = cos(roll);
		scalar_type sinRoll  = sin(roll);

		return _MatTy{ cosRoll*cosYaw-sinRoll*sinPitch*sinYaw, -sinRoll*cosPitch, cosRoll*sinYaw+sinRoll*sinPitch*cosYaw,
					   sinRoll*cosYaw+cosRoll*sinPitch*sinYaw,  cosRoll*cosPitch, sinRoll*sinYaw-cosRoll*sinPitch*cosYaw,
					   -cosPitch*sinYaw,                        sinPitch,         cosPitch*cosYaw                         };
		/*<general-method>
			const _MatTy Myaw   = rotation_yaw<_MatTy>(y);
			const _MatTy Mpitch = rotation_pitch<_MatTy>(p);
			const _MatTy Mroll  = rotation_roll<_MatTy>(r);
			return Mroll( Mpitch( Myaw ) );
		</general-method>*/
	}

	template<typename _MatTy, typename _RadTy>
	_MatTy rotation4x4(_RadTy _Yangle, _RadTy _Pangle, _RadTy _Rangle) {
		// rotation for combination of {yaw, pitch, roll}
		using scalar_type = calculation::matrix_scalar_t<_MatTy>;

		scalar_type yaw      = calculation::_Get_radians<scalar_type>( _Yangle );
		scalar_type cosYaw   = cos(yaw);
		scalar_type sinYaw   = sin(yaw);

		scalar_type pitch    = calculation::_Get_radians<scalar_type>( _Pangle );
		scalar_type cosPitch = cos(pitch);
		scalar_type sinPitch = sin(pitch);
			
		scalar_type roll     = calculation::_Get_radians<scalar_type>( _Rangle );
		scalar_type cosRoll  = cos(roll);
		scalar_type sinRoll  = sin(roll);

		return _MatTy{ cosRoll*cosYaw-sinRoll*sinPitch*sinYaw, -sinRoll*cosPitch, cosRoll*sinYaw+sinRoll*sinPitch*cosYaw,  0,
					   sinRoll*cosYaw+cosRoll*sinPitch*sinYaw,  cosRoll*cosPitch, sinRoll*sinYaw-cosRoll*sinPitch*cosYaw,  0,
					   -cosPitch*sinYaw,                        sinPitch,         cosPitch*cosYaw,                         0,
			           0,                                       0,                0,                                       1 };
	}

	template<typename _MatTy, typename _RadTy>
	void get_Angles(const _MatTy& Mrot, _RadTy& yaw, _RadTy& pitch, _RadTy& roll) {
		pitch = asin( Mrot.at(2,1) );
		yaw   = atan2(-Mrot.at(2, 0), Mrot.at(2, 2)); if (yaw < 0) yaw += 6.28318F;
		roll  = atan2(-Mrot.at(0,1), Mrot.at(1,1)); if (yaw < 0) yaw += 6.28318F;
	}

	template<typename _AngTy>
	struct Angles {
		Angles() = default;

		Angles(_AngTy y, _AngTy p, _AngTy r) : yaw(y), pitch(p), roll(r) {}

		template<typename _MatTy>
		_MatTy get_matrix() const {
			return rotation<_MatTy>(yaw, pitch, roll);
		}
		
		template<typename _MatTy>
		_MatTy get_matrix4x4() const {
			return rotation4x4<_MatTy>(yaw, pitch, roll);
		}

		Angles operator+() const {
			return Angles{ abs(yaw), abs(pitch), abs(roll) };
		}

		Angles operator-() const {
			return Angles{ -yaw, -pitch, -roll };
		}

		Angles operator*(float s) const {
			return Angles{ yaw * s, pitch * s, roll * s };
		}

		Angles& operator*=(float s) {
			return (*this) = (*this) * s;
		}

		_AngTy yaw;   // { x-z-plane, x-axis to z-axis, [ 0, 2Pi[rad] ] }
		_AngTy pitch; // { y-z-plane, z-axis to y-axis, [ 0, 2Pi[rad] ] }
		_AngTy roll;  // { x-y-plane, x-axis to y-axis, [ 0, 2Pi[rad] ] }
	};

}// namespace Euler

namespace Rodrigues {

	template<typename _MatTy, typename _UvecTy, typename _RadTy>
	_MatTy rotation(_UvecTy axis, _RadTy angle) {
		using scalar_type = calculation::matrix_scalar_t<_MatTy>;

		scalar_type theta = calculation::_Get_radians<scalar_type>( angle );
		scalar_type c = cos(theta);
		scalar_type s = sin(theta);
		scalar_type x = axis[0];
		scalar_type y = axis[1];
		scalar_type z = axis[2];
		const auto tmp = axis * (1 - c);// from GLM_library

		return _MatTy{ c + tmp[0]*x,           tmp[0]*y - s*z,     tmp[0]*z + s*y,
						   tmp[1]*x + s*z, c + tmp[1]*y,           tmp[1]*z - s*x,
						   tmp[2]*x - s*y,     tmp[2]*y + s*x, c + tmp[2]*z       };
		/*
		@_Note: colume_major
		@_Title: Rodrigues's rotation formula
		①:
			i            j          k
		[c+(1-c)x²  (1-c)xy-sz (1-c)xz+sy]
		[(1-c)xy+sz c+(1-c)y²  (1-c)yz-sx]
		[(1-c)xz-sy (1-c)yz+sx c+(1-c)z² ]

		②:
		Vrot = cos(θ)v + (1-cos(θ))(k(k·v)) + sin(θ)(k X v)
		Rv = Vrot
		R  = cos(θ)I + (1-cos(θ))k*kT + sin(θ)K

		③: c: cos(theta), s: sin(theta)
					[cos(θ)   0      0   ]
		cos(θ)*I = [0      cos(θ)   0   ]
					[0         0   cos(θ)]
								[x]           [(1-c)x², (1-c)yx, (1-c)zx]
		(1-cos(θ))k*kT = (1-c)*[y]*[x y z] = [(1-c)xy, (1-c)y², (1-c)zy]
								[z]           [(1-c)xz, (1-c)yz, (1-c)z²]
					      [ 0 -z  y]
		sin(θ)K = sin(θ)[ z  0 -x]
					      [-y  x  0]

		*/
	}

	template<typename _MatTy, typename _UvecTy, typename _RadTy>
	_MatTy rotation4x4(_UvecTy axis, _RadTy angle) {
		using scalar_type = calculation::matrix_scalar_t<_MatTy>;

		scalar_type theta = calculation::_Get_radians<scalar_type>( angle );
		scalar_type c = cos(theta);
		scalar_type s = sin(theta);
		scalar_type x = axis[0];
		scalar_type y = axis[1];
		scalar_type z = axis[2];
		const auto tmp = axis * (1 - c);// from GLM_library

		return _MatTy{ c + tmp[0]*x,           tmp[0]*y - s*z,     tmp[0]*z + s*y, (scalar_type)(0),
						   tmp[1]*x + s*z, c + tmp[1]*y,           tmp[1]*z - s*x, (scalar_type)(0),
						   tmp[2]*x - s*y,     tmp[2]*y + s*x, c + tmp[2]*z,       (scalar_type)(0),
					     (scalar_type)(0),   (scalar_type)(0),   (scalar_type)(0), (scalar_type)(1) };
	}

}// namespace Rodrigues

namespace calculation{

	template<typename _MatTy, typename _SclTy> inline
	_MatTy make_translation(_SclTy x, _SclTy y, _SclTy z) {
		using T = matrix_scalar_t<_MatTy>;
		return _MatTy{ (T)(1), (T)(0), (T)(0),     x,
					   (T)(0), (T)(1), (T)(0),     y,
					   (T)(0), (T)(0), (T)(1),     z,
					   (T)(0), (T)(0), (T)(0), (T)(1) };
	}

	template<typename _MatTy, typename _Vec3Ty> inline
	_MatTy make_translation(_Vec3Ty v) {
		return make_translation<_MatTy>(v[0], v[1], v[2]);
	}

	template<typename _MatTy, typename _SclTy> inline
	void translation_property(const _MatTy& m, _SclTy& x, _SclTy& y, _SclTy& z) {
		x = m.at(0, 3);
		y = m.at(1, 3);
		z = m.at(2, 3);
	}


	template<typename _MatTy, typename _AngTy> inline
	_MatTy make_rotation(Euler::Angles<_AngTy> theta) {
		// make matrix3x3 from Euler Angles
		return theta.get_matrix<_MatTy>();
	}

	template<typename _MatTy, typename _AngTy> inline
	_MatTy make_rotation(_AngTy yaw, _AngTy pitch, _AngTy roll) {
		// make matrix3x3 from Euler Angles
		return Euler::rotation<_MatTy>( yaw, pitch, roll );
	}

	template<typename _MatTy, typename _UvecTy, typename _AngTy> inline
	_MatTy make_rotation(_UvecTy axis, _AngTy angle) {
		// make matrix3x3 from axis and angle
		return Rodrigues::rotation<_MatTy>(axis, angle);
	}

	template<typename _MatTy, typename _QatTy>
		requires requires(_QatTy __q) { __q.real(); __q.imag(); }
	_MatTy make_rotation(_QatTy q) {
		// make matrix3x3 from a quaternion
		using scalar_type = vector_scalar_t<_QatTy>;

		scalar_type qx = q.imag()[0];
		scalar_type qy = q.imag()[1];
		scalar_type qz = q.imag()[2];
		scalar_type qw = q.real();
		scalar_type s  = 2/(qx*qx + qy*qy + qz*qz + qw*qw);
			
		return _MatTy{ 1-s*(qy*qy+qz*qz),   s*(qx*qy-qw*qz),   s*(qx*qz+qw*qy),
					     s*(qx*qy+qw*qz), 1-s*(qx*qx+qz*qz),   s*(qy*qz-qw*qx),
					     s*(qx*qz-qw*qy),	s*(qy*qz+qw*qx), 1-s*(qx*qx+qy*qy)  };
		/*<Reference type="book"> { Real-Time Rendering, 4th, pXXX } </Reference>*/
	}

	template<typename _MatTy, typename _AngTy> inline
	_MatTy make_rotation4x4(Euler::Angles<_AngTy> theta) {
		// make matrix4x4 from Euler Angles
		return theta.get_matrix4x4<_MatTy>();
	}

	template<typename _MatTy, typename _AngTy> inline
	_MatTy make_rotation4x4(_AngTy yaw, _AngTy pitch, _AngTy roll) {
		// make matrix4x4 from Euler Angles
		return Euler::rotation<_MatTy>(yaw, pitch, roll);
	}

	template<typename _MatTy, typename _UvecTy, typename _AngTy> inline
	_MatTy make_rotation4x4(_UvecTy axis, _AngTy angle) {
		// make matrix4x4 from axis and angle
		return Rodrigues::rotation4x4<_MatTy>(axis, angle);
	}

	template<typename _MatTy, typename _QatTy>
		requires requires(_QatTy __q) { __q.imag(); __q.real(); }
	_MatTy make_rotation4x4(_QatTy q) {
		// make matrix4x4 from a quaternion
		using scalar_type = vector_scalar_t<_QatTy>;

		scalar_type qx = q.imag()[0];
		scalar_type qy = q.imag()[1];
		scalar_type qz = q.imag()[2];
		scalar_type qw = q.real();
		scalar_type s  = 2/(qx*qx + qy*qy + qz*qz + qw*qw);
			
		return _MatTy{ 1-s*(qy*qy+qz*qz),   s*(qx*qy-qw*qz),   s*(qx*qz+qw*qy), (scalar_type)(0),
					     s*(qx*qy+qw*qz), 1-s*(qx*qx+qz*qz),   s*(qy*qz-qw*qx), (scalar_type)(0),
					     s*(qx*qz-qw*qy),	s*(qy*qz+qw*qx), 1-s*(qx*qx+qy*qy), (scalar_type)(0),
						(scalar_type)(0),  (scalar_type)(0),  (scalar_type)(0), (scalar_type)(1) };
	}

	template<typename _MatTy, typename _UvecTy, typename _AngTy>
	void rotation_property(const _MatTy& m, _UvecTy& axis, _AngTy& angle) {
		// get axis and angle from m
		const auto theta = acos( (m.at(0,0) + m.at(1,1) + m.at(2,2) - 1) / 2 );
		
		angle = _To_angle<_AngTy>(theta);

		const auto x = m.at(2, 1) - m.at(1, 2);
		const auto y = m.at(0, 2) - m.at(2, 0);
		const auto z = m.at(1, 0) - m.at(0, 1);
		const auto a = sin(theta) * 2;
		axis = _UvecTy{ x/a, y/a, z/a };

		/*<angle>
		θ = acos( (trace(R)-1) / 2 )
			= acos( [c+(1-c)x² + c+(1-c)y² + c+(1-c)z² - 1] / 2 )
			= acos( [3*c + (1-c)(x²+y²+z²) - 1] / 2 )
			= acos( [3*c + (1-c)*1 - 1] / 2 )
			= acos( [2*c + 1 - 1] / 2 )
			= acos( c )
		</angle>*/
		/*<axis>
					1     [j[2] - k[1]]
		axis = -------- * [k[0] - i[2]]
				2sin(θ)  [i[1] - j[0]]
				  1       [ [(1-c)y*z+s*x] - [(1-c)y*z-s*x] ]
			 = -------- * [ [(1-c)x*z+s*y] - [(1-c)x*z-s*y] ]
				2sin(θ)  [ [(1-c)x*y+s*z] - [(1-c)x*y-s*z] ]
				  1       [ 2*s*x ]
			 = -------- * [ 2*s*y ]
				2sin(θ)  [ 2*s*z ]

				2*sin(θ)[x y z]
			 = -----------------
				  2sin(θ)
			 = [x y z]
		</axis>*/
	}


	template<typename _MatTy, typename _SclTy> inline
	_MatTy make_scaling(_SclTy sx, _SclTy sy, _SclTy sz) {
		using T = matrix_scalar_t<_MatTy>;
		return _MatTy{   sx,   (T)(0), (T)(0),
					   (T)(0),   sy,   (T)(0),
					   (T)(0), (T)(0),   sz   };
	}

	template<typename _MatTy, typename _SclTy> inline
	_MatTy make_scaling4x4(_SclTy sx, _SclTy sy, _SclTy sz) {
		using T = matrix_scalar_t<_MatTy>;
		return _MatTy{   sx,   (T)(0), (T)(0), (T)(0),
					   (T)(0),   sy,   (T)(0), (T)(0),
					   (T)(0), (T)(0),   sz,   (T)(0),
					   (T)(0), (T)(0), (T)(0), (T)(1) };
	}


	template<typename _MatTy, typename _VecTy, typename _UvecTy>
	_MatTy make_lookTo(_VecTy Peye, _UvecTy f_vector, _UvecTy u_vector) {
		// stand in Peye look to f_vector
		auto r_vector = cross3(u_vector, f_vector);
			 normalize(r_vector);
			 u_vector = cross3(f_vector, r_vector);
			 normalize(u_vector);
		return rigid_body_transform<_MatTy>::inverse(r_vector, u_vector, f_vector, Peye);
	}

	template<typename _MatTy, typename _VecTy, typename _UvecTy>
	_MatTy make_lookToRH(_VecTy Peye, _UvecTy f_vector, _UvecTy u_vector) {
		// stand in Peye look to f_vector
		auto r_vector = cross3(f_vector, u_vector);
			 normalize(r_vector);
			 u_vector = cross3(r_vector, f_vector);
			 normalize(u_vector);
		return rigid_body_transform<_MatTy>::inverse(r_vector, u_vector, f_vector, Peye);
	}

	template<typename _MatTy, typename _VecTy, typename _UvecTy>
	_MatTy make_lookAt(_VecTy Peye, _VecTy Ptarget, _UvecTy u_vector) {
		// stand in Peye look at Ptarget
		return make_lookTo<_MatTy>(Peye, _UvecTy(Ptarget - Peye), u_vector);
	}


	// { perspective_projection for fov, aspect, near, far, zRange[0,1) }
	template<typename _MatTy, typename _AngTy, typename _SclTy>
	_MatTy make_perspective(_AngTy fov, _SclTy aspect, _SclTy near, _SclTy far) {
		using T = matrix_scalar_t<_MatTy>;

		T half_fov     = _Get_radians<T>(fov) / 2;
		T cot_half_fov = 1.0F / tan(half_fov);
		T dfactor      = far / (far - near);
		return _MatTy{ cot_half_fov/aspect,       (T)(0),  (T)(0),        (T)(0),
							        (T)(0), cot_half_fov,  (T)(0),        (T)(0),
							        (T)(0),       (T)(0), dfactor, -near*dfactor,
							        (T)(0),       (T)(0),  (T)(1),        (T)(0) };
		/*<dfactor> 
			A*z+B = (z-n)*f/(f-n)
				  = z*(f/(f-n)]) + -n*(f/(f-n))
		</dfactor>*/
	}

	template<typename _MatTy, typename _SclTy>
	_MatTy make_perspective(_SclTy near, _SclTy far, _SclTy width, _SclTy height, _SclTy dist) {
		// sight range is [near, far], must see vector3{width, height, dist}
		assert( near <= dist && dist <= far );
		using T = matrix_scalar_t<_MatTy>;

		T dfactor = far / (far - near);
		return _MatTy{ dist*2/width,        (T)(0),  (T)(0),        (T)(0),
							 (T)(0), dist*2/height,  (T)(0),        (T)(0),
							 (T)(0),        (T)(0), dfactor, -near*dfactor,
							 (T)(0),        (T)(0),  (T)(1),        (T)(0)  };
		/*<idea>
		    (height/2) / dist = tan(fov/2)
			dist / (height/2) = cot(fov/2)
			(dist*2) / height = cot(fov/2)
		</idea>*/
	}

	template<typename _MatTy, typename _AngTy, typename _SclTy>
	void perspective_property(const _MatTy& m, _AngTy& fov, _SclTy& aspect, _SclTy& near, _SclTy& far) {
		fov    = _To_angle<_AngTy>( atan( 1.0F / m.at(1,1) ) * 2.0F );
		aspect = m.at(0,0) / m.at(1,1);
		near   = - ( m.at(2,3) / m.at(2,2) );
		far    = (-m.at(2,2) * near) / ( 1.0F - m.at(2,2) );
		/*<far>
			far / (far - near) = m[2][2]
						far    = m[2][2] * (far - near)
						far    = m[2][2] * far - m[2][2] * near
			far *(1 - m[2][2]) = - m[2][2] * near
						far    = - m[2][2] * near / (1 - m[2][2])
		</far>*/
	}

	// { Not recommend, RH-coordinate-Z-axis makes it very easy to have meaningless and difficult problems }
	template<typename _MatTy, typename _RadTy, typename _SclTy>
	_MatTy make_perspectiveRH(_RadTy fov, _SclTy aspect, _SclTy near, _SclTy far) {
		using scalar_type  = matrix_scalar_t<_MatTy>;
		using radians_type = angle_t<scalar_type, radian>;

		const auto alpha        = static_cast<scalar_type>(static_cast<radians_type>(fov));
		const auto cot_half_fov = 1/tan( alpha / 2 );
		const auto dfactor      = far / (near - far);
		return _MatTy{ cot_half_fov/aspect, 0,             0,        0,
					   0,                   cot_half_fov,  0,        0,
					   0,                   0,            dfactor, near*dfactor,
					   0,                   0,            -1,        0          };
		/*<dfactor> 
			A*z+B = (z-n)*f/(f-n)
				  = z*(f/(f-n)]) + -n*(f/(f-n))
		</dfactor>*/
	}

	template<typename _MatTy, typename _SclTy>
	_MatTy make_ortho(_SclTy x_lower, _SclTy x_upper, _SclTy y_lower, _SclTy y_upper, _SclTy z_lower, _SclTy z_upper) {
		// (x:[xl,xh], y:[yl,yh], z[zl,zh]) to (x:[-1,+1], y:[-1,+1], z[0,1])
		using T = matrix_scalar_t<_MatTy>;

		T x_range = static_cast<T>(x_upper - x_lower);
		T y_range = static_cast<T>(y_upper - y_lower);
		T z_range = static_cast<T>(z_upper - z_lower);
		return _MatTy{ 2.0F/x_range,  (T)(0),       (T)(0),      -(2.0F*x_lower/x_range + 1.0F),
						(T)(0),      2.0F/y_range,  (T)(0),      -(2.0F*y_lower/y_range + 1.0F),
						(T)(0),       (T)(0),      1.0F/z_range, -(z_lower/z_range),
						(T)(0),       (T)(0),       (T)(0),                 (T)(1)               };
		/*<proof>
			(x - left) / (right - left) ==> [left,right]to[0,1]
			(x-left)/(right-left)*2-1 ==> [left,right]to[-1,+1]
			= ( x/(right-left) - left/(right-left) )*2 - 1
			= 2*x/(right-left) - 2*left/(right-left) - 1
			= x*(2/(right-left)) - ( 2*left/(right-left) + 1 )
			= x*(2/(right-left)) + -( 2*left/(right-left) + 1 )
			= x*(2/x_range) + -( 2*x_lower/x_range + 1 )
		</proof>*/
	}

	/*- - - - - - - - - - - - - - - - - - planar_projection - - - - - - - - - - - - - - -*/
	template<typename _SclTy, typename _BlkTy = _SclTy>
	struct planar_projection {
		// convert V to P in the Plane
		using matrix_type       = matrix4x4<_SclTy, _BlkTy>;
		using scalar_type       = _SclTy;
		using vector3_type      = vector3<_SclTy, _BlkTy>;
		using unit_vector3_type = unit_vector3<_SclTy, _BlkTy>;
		
		/*<figure>
				. <-- Peye
			   /\ \ 
			 //  \\ \
			/vvvv\  \  \
			ssssss\   \   \
			sssssss\    \    \
			ssssssss\     \     \
			=========================y=0
		  </figure>*/
		static matrix_type get_matrix(vector3_type Peye) {// project plane(y=0)
			/*<describ>
				<idea> similar-triangles </idea>
				<important> 
					solve-equation: V.x-Peye.x : P.x-Peye.x = V.y-Peye.y : P.y-Peye.y 
					solve-equation: Peye.x-V.x : Peye.x-P.x = Peye.y-V.y : Peye.y 
				</important>
				<process>
					 X:   (Peye.x-V.x)     (Peye.y-V.y)
						 --------------- = ------------
						  (Peye.x-P.x)      (Peye.y)

					 =>   (Peye.x-V.x) * Peye.y
						 ----------------------- = (Peye.x-P.x)
							  (Peye.y-V.y)

					 =>     (Peye.x-V.x) * Peye.y
						 - ----------------------- + Peye.x = P.x
							    (Peye.y-V.y)

					 =>     (Peye.x-V.x) * Peye.y    Peye.x*(Peye.y-V.y)
						 - ----------------------- + -------------------- = P.x    
							    (Peye.y-V.y)           (Peye.y-V.y)

					 =>   Peye.x*(Peye.y-V.y)-(Peye.x-V.x)*Peye.y
						 ------------------------------------------ = P.x    
							           (Peye.y-V.y)

					 =>    -Peye.x*V.y + V.x*Peye.y
						 ----------------------------- = P.x
								(Peye.y-V.y)
			
					 Z:   -Peye.z*V.y + V.z*Peye.y 
						 ------------------------- = P.z    
							     Peye.y-V.y
				</process>
				<result>
					[-Peye.x*V.y + V.x*Peye.y]
					[            0           ]
					[-Peye.z*V.y + V.z*Peye.y]
					[     Peye.y - V.y       ]
				</result>
				<reference>
					is equivalent to <<RealTimeReandering-4th>> page:225
				</reference>
			  </describ>*/
			if (matrix_type::col_major()) {
				return matrix_type{
					Peye[1], -Peye[0], (_SclTy)0,  (_SclTy)0,
					(_SclTy)0,   (_SclTy)0,  (_SclTy)0,  (_SclTy)0,
					(_SclTy)0,  -Peye[2], Peye[1], (_SclTy)0,
					(_SclTy)0,   (_SclTy)-1, (_SclTy)0,  Peye[1] };
			} else {
				return matrix_type{
					 Peye[1], (_SclTy)0,  (_SclTy)0,  (_SclTy)0,
					-Peye[0], (_SclTy)0, -Peye[2], (_SclTy)0,
					 (_SclTy)0,  (_SclTy)0,  Peye[1], (_SclTy)0,
					 (_SclTy)0,  (_SclTy)-1, (_SclTy)0,  Peye[1] };
			}
		}

		static matrix_type get_matrix(vector3_type Peye, unit_vector3_type N, scalar_type d) {// project any-plane
			/*<describ>
				<process>
				                d + dot(N,Peye)
					P = Peye - ----------------(V-Peye)
					            dot(N,(V-Peye))
							(d + dot(N,Peye))*(V-Peye)    Peye*dot(N,(V-Peye))
					P =  - --------------------------- + ---------------------
					             dot(N,(V-Peye))           dot(N,(V-Peye))
				</process>
				<result>
					[(dot(N,Peye) + d)*Vx - Peye.x*N.x*V.x - Peye.x*N.y*V.y - Peye.x*N.z*V.z - Peye.x*d]
					[(dot(N,Peye) + d)*Vy - Peye.y*N.x*V.x - Peye.y*N.y*V.y - Peye.y*N.z*V.z - Peye.y*d]
					[(dot(N,Peye) + d)*Vz - Peye.z*N.x*V.x - Peye.z*N.y*V.y - Peye.z*N.z*V.z - Peye.z*d]
					[-dot(N,V)+dot(N,Peye)]
				</result>
			  </describ>*/
			const auto N_dot_Peye  = dot(N, Peye);
			const auto Nx_mul_Peye = Peye * N[0];
			const auto Ny_mul_Peye = Peye * N[1];
			const auto Nz_mul_Peye = Peye * N[2];
			const auto d_mul_Pey   = Peye * d;
			if ( matrix_type::col_major() ) {
				return matrix_type{
					N_dot_Peye+d - Nx_mul_Peye[0],              - Ny_mul_Peye[0],              - Nz_mul_Peye[0], -d_mul_Pey[0],
					             - Nx_mul_Peye[1], N_dot_Peye+d - Ny_mul_Peye[1],              - Nz_mul_Peye[1], -d_mul_Pey[1],
					             - Nx_mul_Peye[2],              - Ny_mul_Peye[2], N_dot_Peye+d - Nz_mul_Peye[2], -d_mul_Pey[2],
					                        -N[0],                         -N[1],                         -N[3],  N_dot_Peye };
			} else {
				return matrix_type{
					N_dot_Peye+d - Nx_mul_Peye[0],              - Nx_mul_Peye[1],              - Nx_mul_Peye[2],       -N[0],
							     - Ny_mul_Peye[0], N_dot_Peye+d - Ny_mul_Peye[1],              - Ny_mul_Peye[2],       -N[1],
								 - Nz_mul_Peye[0],              - Nz_mul_Peye[1], N_dot_Peye+d - Nz_mul_Peye[2],       -N[3],
								   - d_mul_Pey[0],                - d_mul_Pey[1],                - d_mul_Pey[2],  N_dot_Peye };
			}
		}
	};

	/*- - - - - - - - - - - - - - - - - - translation - - - - - - - - - - - - - - -*/
	//template<typename _SclTy, typename _BlkTy, typename _Major>
	//MATRIX4x4 translation(MATRIX4x4 axis, SCALAR x, SCALAR y, SCALAR z) {
	//	/*
	//	<idea>
	//		axis * translation(x,y,z,1) = translation(x1,y1,z1,1)
	//	</idea>
	//	<col-major-order>
	//		[rx ux fx 0]   [1 0 0 x]   [rx ux fx rx*x+ux*y+fx*z]
	//		[ry uy fy 0]   [0 1 0 y]   [ry uy fy ry*x+uy*y+fy*z]
	//		[rz uz fz 0] * [0 0 1 z] = [rz uz fz rz*x+uz*y+fz*z]
	//		[ 0  0  0 1]   [0 0 0 1]   [ 0  0  0       1       ]
	//	</col-major-order>
	//	<row-major-order>
	//		[1 0 0 0]   [rx ry rz 0]   [rx ry rz 0]
	//		[0 1 0 0]   [ux uy uz 0]   [ux uy uz 0]
	//		[0 0 1 0] * [fx fy fz 0] = [fx fy fz 0]
	//		[x y z 1]   [ 0  0  0 1]   [rx*x+ux*y+fx*z ry*x+uy*y+fy*z rz*x+uz*y+fz*z 1]
	//	</row-major-order>
	//	*/
	//	if _CONSTEXPR_IF( MATRIX4x4::col_major() ) {
	//		const auto rx = axis.ref(0), ry = axis.ref(4), rz = axis.ref(8);// col(0)
	//		const auto ux = axis.ref(1), uy = axis.ref(5), uz = axis.ref(9);// col(1)
	//		const auto fx = axis.ref(2), fy = axis.ref(6), fz = axis.ref(10);// col(2)
	//		return MATRIX4x4{
	//				rx,     ux,     fx,   rx*x+ux*y+fx*z,
	//				ry,     uy,     fy,   ry*x+uy*y+fy*z,
	//				rz,     uz,     fz,   rz*x+uz*y+fz*z,
	//			(_SclTy)0, (_SclTy)0, (_SclTy)0,           (_SclTy)1 };
	//	} else {
	//		const auto rx = axis.ref(0), ry = axis.ref(1), rz = axis.ref(2);// row(0)
	//		const auto ux = axis.ref(4), uy = axis.ref(5), uz = axis.ref(6);// row(1)
	//		const auto fx = axis.ref(8), fy = axis.ref(9), fz = axis.ref(10);// row(2)
	//		return MATRIX4x4{
	//				  rx,             ry,             rz,       (_SclTy)0,
	//				  ux,             uy,             uz,       (_SclTy)0,
	//				  fx,             fy,             fz,       (_SclTy)0,
	//			rx*x+ux*y+fx*z, ry*x+uy*y+fy*z, rz*x+uz*y+fz*z, (_SclTy)1 };
	//	}
	//}
	//
	//template<typename _SclTy, typename _BlkTy, typename _Major>
	//MATRIX4x4 translate(MATRIX4x4 M, SCALAR x, SCALAR y, SCALAR z) {
	//	if _CONSTEXPR_IF( MATRIX4x4::col_major() ) {
	//		M.ref(3) += x; 
	//		M.ref(3+4) += y; 
	//		M.ref(3+4+4) += z;
	//	} else {
	//		M.ref(8) += x;
	//		M.ref(8 + 1) += y;
	//		M.ref(3 + 2) += z;
	//	}
	//	return M;
	//}
	//template<typename _SclTy, typename _BlkTy, typename _Major>
	//MATRIX4x4 translate_local(MATRIX4x4 M, SCALAR x, SCALAR y, SCALAR z) {
	//	MATRIX4x4 Tm = translation(x, y, z);
	//	return Tm(M);
	//}

	/*- - - - - - - - - - - - - - - - - - rotation - - - - - - - - - - - - - - -*/
	/*using WilliamRowanHamilton::quaternion;
	using WilliamRowanHamilton::polar;*/

	//template<typename _Tq, typename _Tm, typename _SclTy>
	//_Tq _Matrix_to_quaternion(SCALAR qw, _Tm M) {
	//	/*
	//	s = 2/pow(norm(q),2)

	//	tr(M) = 4-2s(qx*qx+qy*qy+qz*qz), 
	//		    = 4-4*(qx*qx+qy*qy+qz*qz)/pow(norm(q),2)
	//			= 4 * (1 - (qx*qx+qy*qy+qz*qz)/pow(norm(q),2))
	//			= 4 * (1 - (qx*qx+qy*qy+qz*qz)/(qx*qx+qy*qy+qz*qz+qw*qw))
	//			= 4 * ((qw*qw)/(qx*qx+qy*qy+qz*qz+qw*qw))
	//			= 4*qw*qw / pow(norm(q),2)
	//	4*qw*qw = tr(M)
	//	qw = sqrt( tr(M)/4 ) <------------

	//	M[2,0]-M[0,2] = s*(qx*qz+qw*qy)-s*(qx*qz-qw*qy)
	//		            = s*qx*qz + s*qw*qy - s*qx*qz + s*qw*qy
	//				    = 2*s*qw*qy
	//	qy = (M[2,0]-M[0,2])/(2*s*qw)
	//	   = (M[2,0]-M[0,2])/(4/pow(norm(q),2)*qw)
	//	qy = (M[2,0]-M[0,2])/(4*qw) <------------

	//	M[0,1]-M[1,0] = s*(qx*qy+qw*qz) - s*(qx*qy-qw*qz)
	//		            = s*qx*qy + s*qw*qz - s*qx*qy + s*qw*qz
	//					= 2*s*qw*qz
	//	qz = (M[0,1]-M[1,0])/(2*s*qw)
	//	   = (M[0,1]-M[1,0])/(4/pow(norm(q),2)*qw)
	//	qz = (M[0,1]-M[1,0])/(4*qw) <------------

	//	M[1,2]-M[2,1] = s*(qy*qz+qw*qx)-s*(qy*qz-qw*qx)
	//		            = s*qy*qz + s*qw*qx - s*qy*qz + s*qw*qx
	//					= 2*s*qw*qx
	//	qx = (M[1,2]-M[2,1]) / (2*s*qw)
	//	   = (M[1,2]-M[2,1]) / (4*qw) <------------
	//	*/
	//	const auto qw_mul4 = qw * 4;

	//	if _CONSTEXPR_IF(_Tm::col_major() ) {
	//		const auto qx = (M.at(2,1) - M.at(1,2)) / qw_mul4;
	//		const auto qy = (M.at(0,2) - M.at(2,0)) / qw_mul4;
	//		const auto qz = (M.at(1,0) - M.at(0,1)) / qw_mul4;
	//		return _Tq(qw, qx, qy, qz);
	//	} else {
	//		const auto qx = (M.at(1,2) - M.at(2,1)) / qw_mul4;
	//		const auto qy = (M.at(2,0) - M.at(0,2)) / qw_mul4;
	//		const auto qz = (M.at(0,1) - M.at(1,0)) / qw_mul4;
	//		return _Tq(qw, qx, qy, qz);
	//	}
	//}

	//template<typename _Tq, typename _Tm, typename _SclTy>
	//_Tq _Matrix_to_quaternion_0qw(SCALAR qw, _Tm M) {
	//	const auto M00 = M.at(0, 0);
	//	const auto M11 = M.at(1, 1);
	//	const auto M22 = M.at(2, 2);
	//	const auto M33 = static_cast<SCALAR>(1);
	//	const auto qx  = clmagic::sqrt((+M00 - M11 - M22 + M33) / 4);
	//	const auto qy  = clmagic::sqrt((-M00 + M11 - M22 + M33) / 4);
	//	const auto qz  = clmagic::sqrt((-M00 - M11 + M22 + M33) / 4);
	//	return _Tq(qw, qx, qy, qz);
	//}

	//template<typename _SclTy, typename _BlkTy, typename _Major>
	//QUATERNION rotation_quaternion(const MATRIX4x4& M) {
	//	const auto trM = M.at(0,0) + M.at(1,1) + M.at(2,2) + M.at(3,3);
	//	const auto qw  = clmagic::sqrt(trM / 4);
	//	if (qw > std::numeric_limits<_SclTy>::epsilon()) {
	//		return _Matrix_to_quaternion<QUATERNION>(qw, M);
	//	} else {
	//		return _Matrix_to_quaternion_0qw<QUATERNION>(qw, M);
	//	}
	//}
	//template<typename _SclTy, typename _BlkTy, typename _Major>
	//QUATERNION rotation3x3_quaternion(const MATRIX4x4& M) {
	//	const auto trM = M.at(0,0) + M.at(1,1) + M.at(2,2) + 1;
	//	const auto qw  = clmagic::sqrt(trM / 4);
	//	if (qw > std::numeric_limits<_SclTy>::epsilon()) {
	//		return _Matrix_to_quaternion<QUATERNION>(qw, M);
	//	} else {
	//		return _Matrix_to_quaternion_0qw<QUATERNION>(qw, M);
	//	}
	//}


	/*- - - - - - - - - - - - - - - - - - rigid_body_transform - - - - - - - - - - - - - - -*/
	template<typename _Tm>
	struct rigid_body_transform {
		using matrix_type = _Tm;
		using scalar_type = matrix_scalar_t<_Tm>;
		using block_type  = matrix_block_t<_Tm>;
		
		using vector3_type      = vector3<scalar_type, block_traits<block_type>>;
		using unit_vector3_type = unit_vector3<scalar_type, block_traits<block_type>>;

		/* inverse( rigid_body )
		 = inverse( translation(t)*rotation )
		 = inverse(rotation) * inverse(translation(t))
		 = transpose(rotation) * translation(-t)
		               
		transpose(rotation) * translation(-t)
		 [rx ry rz 0]           [1 0 0 -x]   [rx ry rz dot(r,-t)]
		 [ux uy uz 0]     *     [0 1 0 -y] = [ux uy uz dot(u,-t)]
		 [fx fy fz 0]           [0 0 1 -z]   [fx fy fz dot(f,-t)]
		 [0  0  0  1]           [0 0 0  1]   [0  0  0       1   ]
		*/
		static matrix_type inverse(unit_vector3_type r, unit_vector3_type u, unit_vector3_type f, vector3_type t) {
			const vector3_type neg_t = -t;

			return matrix_type{
					r[0],   r[1],   r[2], dot(neg_t, r),
					u[0],   u[1],   u[2], dot(neg_t, u),
					f[0],   f[1],   f[2], dot(neg_t, f),
				(scalar_type)0, (scalar_type)0, (scalar_type)0, (scalar_type)1 };
		}
		static matrix_type inverse(const matrix_type& M) {
			const auto r = unit_vector3_type({ M.at(0,0), M.at(1,0), M.at(2,0) }, true);
			const auto u = unit_vector3_type({ M.at(0,1), M.at(1,1), M.at(2,1) }, true);
			const auto f = unit_vector3_type({ M.at(0,2), M.at(1,2), M.at(2,2) }, true);
			const auto t =       vector3_type{ M.at(0,3), M.at(1,3), M.at(2,3) };
			return inverse(r, u, f, t);
		}
	};


	// { first person camera, proj(viewRot(viewTrans) }
	template<typename _SclTy, typename _RadTy = _SclTy>
	class perspective_camera {
	public:
		using vector3_type      = vector3<_SclTy>;
		using unit_vector3_type = unit_vector3<_SclTy>;
		using scalar_type       = _SclTy;
		using radians_type      = _RadTy;

		// view
		vector3_type _Myori;
		Euler::Angles<radians_type> _Myrot;

		// field
		radians_type fov;
		scalar_type aspect;
		scalar_type nearZ;
		scalar_type farZ;

		perspective_camera() = default;

		perspective_camera(vector3_type _Position, unit_vector3_type _Direction, radians_type _Fov) : _Myori(_Position) {
			const unit_vector3_type _FORWARD = unit_vector3_type{ 0.F, 0.F, 1.0F };
			
			scalar_type angle = acos(dot(_Direction, _FORWARD));
			if ( abs(angle) > std::numeric_limits<scalar_type>::epsilon() ) {
				unit_vector3_type axis = cross3(_FORWARD, _Direction);// left-hand
				auto Mrot = make_rotation<matrix3x3<scalar_type>>(axis, angle);
				Euler::get_Angles(Mrot, _Myrot.yaw, _Myrot.pitch, _Myrot.roll);
			} else {
				_Myrot = { 0.0F, 0.0F, 0.0F };
			}

			/*_Myrot.pitch = asin(_Direction[1]);
			scalar_type r = cos(_Myrot.pitch);
			_Myrot.yaw = asin(_Direction[2] / r);*/
			fov    = _Fov;
			aspect = static_cast<scalar_type>(16.0F / 9.0F);
			nearZ  = static_cast<scalar_type>(1.F);
			farZ   = static_cast<scalar_type>(40000.0F);
		}

		template<typename _MatTy>
		_MatTy view_matrix() const {// inverse(trans*rot) = inverse(rot) * inverse(trans)
			_MatTy invTrans = make_translation<_MatTy>( - _Myori );
			_MatTy invRot   = make_rotation4x4<_MatTy>( - _Myrot );
			return invRot * invTrans;
		}
		template<typename _MatTy>
		_MatTy projection_matrix() const {
			return make_perspective<_MatTy>(fov, aspect, nearZ, farZ);
		}

		vector3_type position() const {
			return _Myori;
		}
		unit_vector3_type f_vector() const {
			matrix3x3<scalar_type> invRot = (-_Myrot).get_matrix<matrix3x3<scalar_type>>();
			return unit_vector3_type{ invRot.at(2,0), invRot.at(2,1), invRot.at(2,2) };
		}
		unit_vector3_type u_vector() const {
			matrix3x3<scalar_type> invRot = (-_Myrot).get_matrix<matrix3x3<scalar_type>>();
			return unit_vector3_type{ invRot.at(1,0), invRot.at(1,1), invRot.at(1,2) };
		}
		unit_vector3_type r_vector() const {
			matrix3x3<scalar_type> invRot = (-_Myrot).get_matrix<matrix3x3<scalar_type>>();
			return unit_vector3_type{ invRot.at(0,0), invRot.at(0,1), invRot.at(0,2) };
		}

		perspective_camera& moveX(scalar_type _X) {
			_Myori += r_vector() * _X;
			return *this;
		}
		perspective_camera& moveY(scalar_type _Y) {
			_Myori += u_vector() * _Y;
			return *this;
		}
		perspective_camera& moveZ(scalar_type _Z) {
			_Myori += f_vector() * _Z;
			return *this;
		}

		perspective_camera& yaw(radians_type angle) {
			_Myrot.yaw += angle;
			return *this;
		}
		perspective_camera& pitch(radians_type angle) {
			_Myrot.pitch += angle;
			return *this;
		}
		perspective_camera& roll(radians_type angle) {
			_Myrot.roll += angle;
			return *this;
		}
	};

	template<typename T, typename A = T>
	class ortho_camera {
	public:
		using scalar_type  = T;
		using radians_type = A;

		// view
		Euler::Angles<A> _Myrot;
		// proj
		vector2<T> _Myhorizon;
		vector2<T> _Myvertical;
		vector2<T> _Mydepthrange;

		ortho_camera() {
			_Myrot = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };
			_Myhorizon = { static_cast<T>(-1), static_cast<T>(+1) };
			_Myvertical = { static_cast<T>(-1), static_cast<T>(+1) };
			_Mydepthrange = { static_cast<T>(-1), static_cast<T>(+1) };
		}

		template<typename _MatTy>
		_MatTy view_matrix() const {
			return make_rotation4x4<_MatTy>(-_Myrot);
		}
		template<typename _MatTy>
		_MatTy projection_matrix() const {
			return make_ortho<_MatTy>(_Myhorizon[0], _Myhorizon[1], _Myvertical[0], _Myvertical[1], _Mydepthrange[0], _Mydepthrange[1]);
		}

		vector3<T> position() const {
			return vector3<T>{ (T)0, (T)0, (T)0 };
		}
		vector3<T> f_vector() const {
			matrix3x3<scalar_type> invRot = (-_Myrot).get_matrix<matrix3x3<scalar_type>>();
			return vector3<T>{ invRot.at(2,0), invRot.at(2,1), invRot.at(2,2) };
		}
		vector3<T> u_vector() const {
			matrix3x3<scalar_type> invRot = (-_Myrot).get_matrix<matrix3x3<scalar_type>>();
			return vector3<T>{ invRot.at(1,0), invRot.at(1,1), invRot.at(1,2) };
		}
		vector3<T> r_vector() const {
			matrix3x3<scalar_type> invRot = (-_Myrot).get_matrix<matrix3x3<scalar_type>>();
			return vector3<T>{ invRot.at(0,0), invRot.at(0,1), invRot.at(0,2) };
		}

		template<typename _RayTy>
		_RayTy get_ray(T u, T v) const {
			_RayTy r;
			r.origin    = { lerp(_Myhorizon[0], _Myhorizon[1], u), lerp(_Myvertical[0], _Myvertical[1], v), 0.0F };
			r.direction = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(1) };
			r.start = _Mydepthrange[0];
			r.end   = _Mydepthrange[1];
			return std::move(r);
		}

		ortho_camera& moveX(scalar_type _X) {
			_Myhorizon += _X;
			return *this;
		}
		ortho_camera& moveY(scalar_type _Y) {
			_Myvertical += _Y;
			return *this;
		}
		ortho_camera& moveZ(scalar_type _Z) {
			_Mydepthrange += _Z;
			return *this;
		}

		ortho_camera& yaw(radians_type angle) {
			_Myrot.yaw += angle;
			return *this;
		}
		ortho_camera& pitch(radians_type angle) {
			_Myrot.pitch += angle;
			return *this;
		}
		ortho_camera& roll(radians_type angle) {
			_Myrot.roll += angle;
			return *this;
		}
	};




	template<typename _SclTy>
	class Geometry {
	public:
		enum { _VP = 0, _VT = 1, _TAN = 2, _BTAN = 3 };
		using vector4_type   = vector4<_SclTy>;
		using attribute_type = std::vector<vector4_type>;
		struct face_index_type {
			struct _Findex_t {
				size_t start_index = 0;
				size_t index_count = 0;
			};

			std::vector<uint32_t> _Mydata;
			std::vector<_Findex_t> _Myfacei;

			size_t size() const {
				return _Myfacei.size();
			}
			size_t vertex_size() const {
				return _Mydata.size();
			}
			size_t vertex_size(size_t f) const {
				return _Myfacei[f].index_count;
			}

			uint32_t& at(size_t f, size_t v) {
				return _Mydata[ _Myfacei[f].start_index + v ];
			}
			const uint32_t& at(size_t f, size_t v) const {
				return _Mydata[ _Myfacei[f].start_index + v ];
			}

			void insert_a_vertex(size_t f, uint32_t _Val) {
				// 1. Insert _Val into _Mydata
				_Mydata.insert( std::next(_Mydata.begin(), _Myfacei[f].start_index), _Val );
				// 2. Update [_Myfacei[f+1], end)
				for (size_t i = f+1; i != _Myfacei.size(); ++i) {
					_Myfacei[i].start_index += 1;
				}
			}
			void insert_a_face(const uint32_t* _Pval, size_t _Size) {
				_Findex_t _Newfacei;
				_Newfacei.start_index = _Mydata.size();
				_Newfacei.index_count = _Size;
				// 1. Insert [_Pval, ...) to _Mydata
				_Mydata.insert( _Mydata.end(), _Pval, _Pval + _Size );// allow memory error, call check()
				// 2. Move _Newfacei to _Myfacei
				_Myfacei.push_back( std::move(_Newfacei) );
			}
		};

		std::vector<attribute_type> attributes;
		std::vector<face_index_type> face_indices;

		Geometry() : attributes(2), face_indices(2) {}

		bool _Is_bad() const {
			// requires { has_position, has_texcoord }
			if (face_indices.size() < 2) {
				return true;
			}

			// requires { face_indices.not_have_memory_exception }
			auto       _First = face_indices.begin();
			const auto _Last  = face_indices.end();
			size_t     _Fsize = _First->size(); ++_First;
			for ( ; _First != _Last; ++_First) {
				if (_Fsize != _First->size()) {
					return true;
				}
			}

			return false;
		}

		template<typename _Fn>
		void expand_attribute(_Fn _Fx) {
			assert( !_Is_bad() );
			// 1. Get UV, computation based UV
			const auto&     _UV       = attributes[_VT];
			const auto&     _UVi      = face_indices[_VT];
			// 2. Create new_memory
			attribute_type  _Newattri = attribute_type(_UV.size());
			face_index_type _Newindex = _UVi;
			// 3. Compute new_value
			for (size_t f = 0; f != _UVi.size(); ++f) {
				for (size_t v = 0; v != _UVi[f].size(); ++v) {
					const size_t         i = _UVi.at(f,v);
					const vector4_type& _X = _UV[i];
					_Newattri[i] = _Fx(_X);
				}
			}
			// 4. Move new_memory to result_memory
			attributes.push_back(std::move(_Newattri));
			face_indices.push_back(std::move(_Newindex));
		}
		
		void expand_attribute() {
			assert( !_Is_bad() );
			this->expand_attribute( [](vector4_type){ return vector4_type(static_cast<_SclTy>(0)); } );
		}

		size_t size() const {
			return face_indices[_VT].size();
		}

		size_t vertex_size() const {
			return face_indices[_VT].vertex_size();
		}


		struct face {
			std::vector<vector4_type> position;
			std::vector<vector4_type> texcoord;
			vector4_type normal;
			vector4_type tangent;
			vector4_type bitangent;
		};

		const face operator[](size_t f) const {
			assert( !_Is_bad() );
			// 1. Get vertex_size at face[f]
			size_t _Vertex_size = face_indices[_VT].vertex_size(f);
			// 2. Create _Face and malloc memory
			face _Face;
			_Face.position.resize(_Vertex_size);
			_Face.texcoord.resize(_Vertex_size);
			// 3. Copy position and texcoord
			for (size_t v = 0; v != _Vertex_size; ++v) {
				const size_t vi = face_indices[_VP].at(f, v);
				const size_t ti = face_indices[_VT].at(f, v);
				_Face.position[v] = attributes[_VP][vi];
				_Face.texcoord[v] = attributes[_VT][ti];
			}
			// 4. Compute normal and tangent
			if (_Vertex_size >= 3) {
				_Face.normal = normalize(cross3(_Face.position[2]-_Face.position[0], _Face.position[1]-_Face.position[0]));
			}
			// tangent ...

			return std::move(_Face);
		}
	};

	template<typename _Ty, typename _Fn>
	Geometry<_Ty> make_differential_geometry(_Ty xi, _Ty xf, size_t nx, _Ty yi, _Ty yf, size_t ny, _Fn _Func) {
		Geometry<_Ty> _Dest;

		// 1. Assert( integral requires xi<xf && yi<yf )
		if (xi > xf) { std::swap(xi, xf); }
		if (yi > yf) { std::swap(yi, yf); }

		// 2. Compute position and UV
		auto deltaX = (xf - xi);
		auto deltaY = (yf - yi);
		auto dX     = deltaX / static_cast<_Ty>(nx);
		auto dY     = deltaY / static_cast<_Ty>(ny);
		/*// floating-point round-error
		for (_Ty y = yi; (y < yf) || (abs(yf-y) < 0.02F); y += dY) {
			for (_Ty x = xi; (x < xf) || (abs(xf-x) < 0.02F); x += dX) {
				_Dest.attributes[Geometry<_Ty>::_VP].push_back( _Func(x, y) );
				_Dest.attributes[Geometry<_Ty>::_VT].push_back( { x/deltaX, y/deltaY, 0.0F, 0.0F } );
			}
		}*/
		for (size_t iy = 0; iy <= ny; ++iy) {
			for (size_t ix = 0; ix <= nx; ++ix) {
				_Ty x = xi + static_cast<_Ty>(ix) * dX;
				_Ty y = yi + static_cast<_Ty>(iy) * dY;
				_Dest.attributes[Geometry<_Ty>::_VP].push_back(_Func(x, y));
				_Dest.attributes[Geometry<_Ty>::_VT].push_back({ x / deltaX, y / deltaY, 0.0F, 0.0F });
			}
		}

		// 3. Compute <Quad> face_indices
		/*<graph>
		   /|\ Y
			|
			18---19---20---21---22---23
			|    |    |    |    |    |
			12---13---14---15---f----end      f:final, end:end_point
			|    |    |    |    |    |
			6----7----8----9----10---11
			|    |    |    |    |    |
			0----1----2----3----4----5---->X
			nx = 5
			ny = 3
			iy < 3=ny, ix < 5=nx
		</graph>*/
		for (size_t iy = 0; iy != ny; ++iy) {
			for (size_t ix = 0; ix != nx; ++ix) {
				std::array<uint32_t, 4> vN;
				vN[0] = iy * (nx + 1) + ix;
				vN[1] = vN[0] + 1;
				vN[2] = vN[1] + (nx + 1); 
				vN[3] = vN[2] - 1;

				_Dest.face_indices[Geometry<_Ty>::_VP].insert_a_face(vN.data(), vN.size());
				_Dest.face_indices[Geometry<_Ty>::_VT].insert_a_face(vN.data(), vN.size());
			}
		}

		return std::move(_Dest);
	}

}// namespace clmagic

#endif