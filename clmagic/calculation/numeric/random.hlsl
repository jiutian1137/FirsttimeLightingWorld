/*<requires>
real_t : half | float | double
vec2_t : vec2 | float2 | dvec2 | double2
vec3_t : vec3 | float3 | dvec3 | double3
vec4_t : vec4 | float4 | dvec4 | double4
ivec4_t : ivec4 | int4
real_t fract(real_t)
vec2_t fract(vec2_t)
vec3_t fract(vec3_t)
vec4_t fract(vec4_t)
vec4_t invsqrt(vec4_t)
real_t fade(real_t)
vec3_t fade(vec3_t)
real_t bilerp(real_t,real_t,real_t,real_t,real_t,real_t)
real_t trilerp(real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t, real_t)
<requires>*/

//#define real_t float
//#define vec2_t float2
//#define vec3_t float3
//#define vec4_t float4
//#define ivec3_t int3
//#define ivec4_t int4
//#define fract frac 

ivec4_t permute(ivec4_t x)  {
	return (x * x * 34 + x) % 289;
}

real_t cnoise2(vec2_t P){
	ivec4_t Pi = ivec4_t(floor(vec4_t(P[0], P[1], P[0], P[1]))) + ivec4_t(0, 0, 1, 1);
			Pi = Pi % 289; // To avoid truncation effects in permutation
	vec4_t  Pf = fract(vec4_t(P[0], P[1], P[0], P[1])) - vec4_t(0, 0, 1.0F, 1.0F);
	ivec4_t ix = ivec4_t(Pi[0], Pi[2], Pi[0], Pi[2]);
	ivec4_t iy = ivec4_t(Pi[1], Pi[1], Pi[3], Pi[3]);
	vec4_t  fx = vec4_t(Pf[0], Pf[2], Pf[0], Pf[2]);
	vec4_t  fy = vec4_t(Pf[1], Pf[1], Pf[3], Pf[3]);

	// 2. Compute gradient_vectors for <four> corners
	vec4_t i   = vec4_t( permute(permute(ix) + iy) );
	vec4_t gx  = fract(i / 41.0F) * 2.0F - 1.0F;
	vec4_t gy  = abs(gx) - 0.5F;
		   gx  = gx - floor(gx + 0.5F);
	vec2_t g00 = vec2_t(gx[0], gy[0]);
	vec2_t g10 = vec2_t(gx[1], gy[1]);
	vec2_t g01 = vec2_t(gx[2], gy[2]);
	vec2_t g11 = vec2_t(gx[3], gy[3]);
	vec4_t norm = invsqrt(vec4_t(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
	g00 *= norm[0]; g01 *= norm[1]; g10 *= norm[2]; g11 *= norm[3];
		
	// 3. Compute gradient weights
	real_t w00 = dot(g00, vec2_t(fx[0], fy[0]));
	real_t w10 = dot(g10, vec2_t(fx[1], fy[1]));
	real_t w01 = dot(g01, vec2_t(fx[2], fy[2]));
	real_t w11 = dot(g11, vec2_t(fx[3], fy[3]));

	// 4. Compute bilinear interpolation of weights
	real_t result = bilerp( w00, w10, w01, w11, fade(Pf[0]), fade(Pf[1]) ) * 2.2F;

	// 5. Avoid floating-point-error
	return clamp(result, -1.0F, 1.0F);
}

real_t cnoise3(vec3_t P) {
	// 1. Compute noise cell coordinates and offsets
	ivec3_t Pi0 = ivec3_t( floor(P) );// Integer part for indexing
	ivec3_t Pi1 = Pi0 + 1; // Integer part + 1
			Pi0 = Pi0 % 289;
			Pi1 = Pi1 % 289;
	vec3_t  Pf0 = fract(P);// Fractional part for interpolation
	vec3_t  Pf1 = Pf0 - 1.0F; // Fractional part - 1.0
	ivec4_t ix  = ivec4_t( Pi0[0], Pi1[0], Pi0[0], Pi1[0] );
	ivec4_t iy  = ivec4_t( Pi0[1], Pi0[1], Pi1[1], Pi1[1] );
	ivec4_t iz0 = ivec4_t( Pi0[2] );
	ivec4_t iz1 = ivec4_t( Pi1[2] );

	// 2. Compute gradient_vectors
	ivec4_t ixy  = permute( permute( ix ) + iy );
	vec4_t  ixy0 = vec4_t( permute( ixy + iz0 ) );
	vec4_t  ixy1 = vec4_t( permute( ixy + iz1 ) );

	vec4_t gx0 = ixy0 * (1.0F/7.0F);
	vec4_t gy0 = fract( floor(gx0) * (1.0F/7.0F) ) - 0.5F;
			gx0 = fract( gx0 );
	vec4_t gz0 = vec4_t(0.5F) - abs(gx0) - abs(gy0);
	vec4_t sz0 = vec4_t(0.0F) >= gz0;
			gx0 -= sz0 * ( (gx0 >= vec4_t(0.0F)) - 0.5F );
			gy0 -= sz0 * ( (gy0 >= vec4_t(0.0F)) - 0.5F );

	vec4_t gx1 = ixy1 * (1.0F/7.0F);
	vec4_t gy1 = fract( floor(gx1) * (1.0F/7.0F) ) - 0.5F;
			gx1 = fract( gx1 );
	vec4_t gz1 = vec4_t(0.5F) - abs(gx1) - abs(gy1);
	vec4_t sz1 = vec4_t(0.0F) >= gz1;
			gx1 -= sz1 * ( (gx1 >= vec4_t(0.0F)) - 0.5F );
			gy1 -= sz1 * ( (gy1 >= vec4_t(0.0F)) - 0.5F );

	vec3_t g000 = vec3_t(gx0[0], gy0[0], gz0[0]);
	vec3_t g100 = vec3_t(gx0[1], gy0[1], gz0[1]);
	vec3_t g010 = vec3_t(gx0[2], gy0[2], gz0[2]);
	vec3_t g110 = vec3_t(gx0[3], gy0[3], gz0[3]);
	vec3_t g001 = vec3_t(gx1[0], gy1[0], gz1[0]);
	vec3_t g101 = vec3_t(gx1[1], gy1[1], gz1[1]);
	vec3_t g011 = vec3_t(gx1[2], gy1[2], gz1[2]);
	vec3_t g111 = vec3_t(gx1[3], gy1[3], gz1[3]);

	vec4_t norm0 = invsqrt(vec4_t(dot(g000, g000), dot(g100, g100), dot(g010, g010), dot(g110, g110)));
	g000 *= norm0[0]; g100 *= norm0[1]; g010 *= norm0[2]; g110 *= norm0[3];
	vec4_t norm1 = invsqrt(vec4_t(dot(g001, g001), dot(g101, g101), dot(g011, g011), dot(g111, g111)));
	g001 *= norm1[0]; g101 *= norm1[1]; g011 *= norm1[2]; g111 *= norm1[3];

	// 3. Compute gradient weights
	real_t w000 = dot(g000, Pf0);
	real_t w100 = dot(g100, vec3_t(Pf1[0], Pf0[1], Pf0[2]));
	real_t w010 = dot(g010, vec3_t(Pf0[0], Pf1[1], Pf0[2]));
	real_t w110 = dot(g110, vec3_t(Pf1[0], Pf1[1], Pf0[2]));
	real_t w001 = dot(g001, vec3_t(Pf0[0], Pf0[1], Pf1[2]));
	real_t w101 = dot(g101, vec3_t(Pf1[0], Pf0[1], Pf1[2]));
	real_t w011 = dot(g011, vec3_t(Pf0[0], Pf1[1], Pf1[2]));
	real_t w111 = dot(g111, Pf1);

	// 4. Compute trilinear interpolation of weights
	vec3_t uvw    = fade(Pf0);
	real_t result = trilerp(w000, w100, w010, w110, w001, w101, w011, w111, uvw[0], uvw[1], uvw[2]) * 2.2F;

	// 5. Avoid floating-point-error
	return clamp(result, -1.0F, 1.0F);
}


real_t hash11(real_t x) {
	return fract(sin(x) * 43758.5453F);
}

vec2_t hash22(vec2_t x){
	x = vec2_t(dot(x, vec2_t(127.1F, 311.7F)),
			   dot(x, vec2_t(269.5F, 183.3F)));
	return fract( sin(x) * 43758.5453F );
}

vec3_t hash33(vec3_t x) {
	x = vec3_t(dot(x, vec3_t(127.1F, 311.7F, 74.7F)),
			   dot(x, vec3_t(269.5F, 183.3F, 246.1F)),
			   dot(x, vec3_t(113.5F, 271.9F, 124.6F)));
	return fract( sin(x) * 43758.5453123F );
}

real_t cells2(vec2_t P) {
	vec2_t Pi = floor(P);
	vec2_t Pf = P - Pi;

	real_t d = 8;
	vec2_t g = vec2_t(real_t(0));
	for (int j = -1; j <= 1; ++j) {
		for (int i = -1; i <= 1; ++i) {
			g[0] = real_t(j);
			g[1] = real_t(i);
			vec2_t o = hash22(Pi + g);
			vec2_t r = g + o - Pf;
			d = min(d, dot(r, r));
		}
	}
	return sqrt(d);
}

real_t cells3(vec3_t P) {
	vec3_t Pi = floor(P);
	vec3_t Pf = P - Pi;

	real_t d = 8;
	vec3_t g = vec3_t(real_t(0));
	for (int k = -1; k <= 1; ++k) {
		for (int j = -1; j <= 1; ++j) {
			for (int i = -1; i <= 1; ++i) {
				g[0] = real_t(i);
				g[1] = real_t(j);
				g[2] = real_t(k);
				vec3_t o = hash33(Pi + g);
				vec3_t r = g + o - Pf;
				d = min(d, dot(r, r));
			}
		}
	}
	return sqrt(d);
}
