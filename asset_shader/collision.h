#ifndef real_t
#define real_t float
#endif
#ifndef vec2_t
#define vec2_t vec2
#endif
#ifndef vec3_t
#define vec3_t vec3
#endif
#ifndef vec4_t
#define vec4_t vec4
#endif

#ifndef _out
#ifdef __cplusplus
#define _out(TY) TY& 
#else
#define _out(TY) out TY
#endif
#endif

bool quadratic(real_t A, real_t B, real_t C, _out(real_t) t0, _out(real_t) t1) {// return false if(B^2-4AC < 0)
	real_t _Discrim = B * B - 4 * A * C;
	if (_Discrim < 0.0) {// no solution
		return false;
	} else {// compute t0 and t1
		_Discrim = sqrt(_Discrim);
		t0 = (-B - _Discrim) * 0.5F / A;
		t1 = (-B + _Discrim) * 0.5F / A;
		return true;
	}
}

struct segment_t {
	vec3_t origin;
	vec3_t direction;
	real_t min_t;
	real_t max_t;
#ifdef __cplusplus
	segment_t(vec3_t o, vec3_t v, real_t t0, real_t t1) : origin(o), direction(v), min_t(t0), max_t(t1) {}
#endif
};

struct sphere_t {
	vec3_t center;
	real_t radius;
#ifdef __cplusplus
	sphere_t(vec3_t C, real_t r) : center(C), radius(r) {}
#endif
};

struct hit_t {
	real_t t;
};

bool sphere_equation(vec3_t origin, vec3_t direction, vec3_t center, real_t radius, out float t0, out float t1) {
	vec3_t Co = (origin - center);
	real_t A = dot(direction, direction);
	real_t B = dot(direction, Co) * 2;
	real_t C = dot(Co, Co) - radius * radius;
	return quadratic(A, B, C, t0, t1);
}

bool test_segment_sphsurf(segment_t r, sphere_t sph) {
	vec3_t Co = (r.origin - sph.center);
	real_t A  = dot(r.direction, r.direction);
	real_t B  = dot(r.direction, Co) * 2;
	real_t C  = dot(Co, Co) - sph.radius * sph.radius;
	real_t t0, t1;
	if (quadratic(A, B, C, t0, t1)) {
		if (r.min_t <= t0 && t0 <= r.max_t || r.min_t <= t1 && t1 <= r.max_t) {
			return true;
		}
	}
	return false;
}

bool intersect_ray_sphsurf(vec3_t r_origin, vec3_t r_direction, vec3_t sph_center, real_t sph_radius, out float t) {
	vec3_t center_to_origin = r_origin - sph_center;
	real_t t0, t1;
	if ( quadratic( dot(r_direction, r_direction), 
					dot(center_to_origin, r_direction) * 2.0f,
					dot(center_to_origin, center_to_origin) - sph_radius*sph_radius,
					t0, t1) ) {
		if (t0 >= 0) {
			t = t0; return true;
		} else if (t1 >= 0) {
			t = t1; return true;
		}
	}

	return false;
}

bool intersect_segment_sphsurf(segment_t r, sphere_t sph, out float res) {
	// 1. Solve sphere equation
	vec3_t Co = (r.origin - sph.center);
	real_t A  = dot(r.direction, r.direction);
	real_t B  = dot(r.direction, Co) * 2;
	real_t C  = dot(Co, Co) - sph.radius * sph.radius;
	real_t t0, t1;
	if (quadratic(A, B, C, t0, t1)) {
		// 2. Get result.t from t0 and t1
		if (r.min_t <= t0 && t0 <= r.max_t) {
			res = t0;
			return true;
		} else if (r.min_t <= t1 && t1 <= r.max_t) {
			res = t1;
			return true;
		}
	}
	return false;
}

bool test_ray_sphere(vec3_t r_origin, vec3_t r_direction, vec3_t sph_center, real_t sph_radius) {
	vec3_t center_to_origin = r_origin - sph_center;

	real_t c = dot(center_to_origin, center_to_origin) - sph_radius * sph_radius;
	if (c <= 0.0f) { return true; }

	real_t b = dot(center_to_origin, r_direction);
	if (b >= 0.0f) { return false; }

	real_t disc = b * b - c;
	if (disc < 0.0f) { return false; }

	return true;
}
