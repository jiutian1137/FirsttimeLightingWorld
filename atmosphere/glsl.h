#ifndef __cplusplus
/* only shader */
#define block_begin(NAME) NAME {
#define block_end };

//#define lerp mix
//#define atan2 atan
//#define invsqrt rsqrt
//#define NOT not
//#define ddx dFdx
//#define ddy dFdy
//#define ddx_fine dFdxFine
//#define ddy_fine dFdyFine
//#define ddx_coarse dFdxCoarse
//#define ddy_coarse dFdyCoarse

#else
/* only cplusplus */
#define block_begin(NAME) 
#define block_end 

#define ADD_SUB_MUL_DIV_OP(...) \
__VA_ARGS__& operator+=(__VA_ARGS__); \
__VA_ARGS__& operator-=(__VA_ARGS__); \
__VA_ARGS__& operator*=(__VA_ARGS__); \
__VA_ARGS__& operator/=(__VA_ARGS__); \
friend __VA_ARGS__ operator+(__VA_ARGS__, __VA_ARGS__); \
friend __VA_ARGS__ operator-(__VA_ARGS__, __VA_ARGS__); \
friend __VA_ARGS__ operator*(__VA_ARGS__, __VA_ARGS__); \
friend __VA_ARGS__ operator/(__VA_ARGS__, __VA_ARGS__);

#define MUL_DIV_SCALAR_OP(SCL, ...) \
__VA_ARGS__& operator*=(SCL); \
__VA_ARGS__& operator/=(SCL); \
friend __VA_ARGS__ operator*(SCL, __VA_ARGS__); \
friend __VA_ARGS__ operator*(__VA_ARGS__, SCL); \
friend __VA_ARGS__ operator/(SCL, __VA_ARGS__); \
friend __VA_ARGS__ operator/(__VA_ARGS__, SCL);

/* define keyword */
#define layout(...) 
#define in const
#define out
#define uniform const
#define inout

template<typename _Ty> struct vec2_;
template<typename _Ty> struct vec3_;
template<typename _Ty> struct vec4_;

template<typename _Ty>
struct vec2_ {
	vec2_();
	vec2_(_Ty);
	vec2_(_Ty, _Ty);
	template<typename _Ty2> vec2_(vec2_<_Ty2>);
	template<typename _Ty2> vec2_(vec3_<_Ty2>);
	template<typename _Ty2> vec2_(vec4_<_Ty2>);

	_Ty& operator[](size_t);
	_Ty operator[](size_t) const;
	vec2_ operator-() const;
	ADD_SUB_MUL_DIV_OP(vec2_<_Ty>);
	MUL_DIV_SCALAR_OP(_Ty, vec2_<_Ty>);
	union  {
		struct { _Ty x, y; };
		struct { _Ty r, g; };
		vec2_<_Ty> xy;
		vec2_<_Ty> yx;
		vec4_<_Ty> xyxy;
		vec4_<_Ty> xxyy;
	};
};

template<typename _Ty>
struct vec3_ {
	vec3_();
	/* x x x */vec3_(_Ty);
	/* x y z */vec3_(_Ty, _Ty, _Ty);
	/* xy  z */vec3_(vec2_<_Ty>, _Ty);
	/* x  yz */vec3_(_Ty, vec2_<_Ty>);
	template<typename _Ty2> explicit vec3_(vec2_<_Ty2>);
	template<typename _Ty2> vec3_(vec3_<_Ty2>);
	template<typename _Ty2> vec3_(vec4_<_Ty2>);

	_Ty& operator[](size_t);
	_Ty operator[](size_t) const;
	vec3_ operator-() const;
	ADD_SUB_MUL_DIV_OP(vec3_<_Ty>);
	MUL_DIV_SCALAR_OP(_Ty, vec3_<_Ty>);
	union {
		struct { _Ty x, y, z; };
		struct { _Ty r, g, b; };
		vec2_<_Ty> xy;
		vec2_<_Ty> yx;
		vec2_<_Ty> xz;
		vec3_<_Ty> xyz;
		vec3_<_Ty> xzy;
		vec4_<_Ty> xyxy;
		vec4_<_Ty> xxyy;
		vec4_<_Ty> xyxz;
		vec3_<_Ty> rgb;
	};
};

template<typename _Ty>
struct vec4_ {
	vec4_();
	/* x x x x */vec4_(_Ty);
	/* x y z w */vec4_(_Ty, _Ty, _Ty, _Ty);
	/* xy  z w */vec4_(vec2_<_Ty>, _Ty, _Ty);
	/* x yz  w */vec4_(_Ty, vec2_<_Ty>, _Ty);
	/* x y zw  */vec4_(_Ty, _Ty, vec2_<_Ty>);
	/* xy  zw  */vec4_(vec2_<_Ty>, vec2_<_Ty>);
	/* xyz   w */vec4_(vec3_<_Ty>, _Ty);
	/* x yzw   */vec4_(_Ty, vec3_<_Ty>);
	template<typename _Ty2> explicit vec4_(vec2_<_Ty2>);
	template<typename _Ty2> explicit vec4_(vec3_<_Ty2>);
	template<typename _Ty2> vec4_(vec4_<_Ty2>);

	_Ty& operator[](size_t);
	_Ty operator[](size_t) const;
	vec4_ operator-() const;
	ADD_SUB_MUL_DIV_OP(vec4_<_Ty>);
	MUL_DIV_SCALAR_OP(_Ty, vec4_<_Ty>);
	union {
		struct { _Ty x, y, z, w; };
		struct { _Ty r, g, b, a; };
		vec2_<_Ty> xy;
		vec2_<_Ty> yx;
		vec2_<_Ty> xz;
		vec3_<_Ty> xyz;
		vec3_<_Ty> xzy;
		vec4_<_Ty> xyxy;
		vec4_<_Ty> xxyy;
		vec4_<_Ty> xyxz;
		vec3_<_Ty> rgb;
	};
};

/* genType */
using vec2 = vec2_<float>;
using vec3 = vec3_<float>;
using vec4 = vec4_<float>;

/* genDType */
using dvec2 = vec2_<double>;
using dvec3 = vec3_<double>;
using dvec4 = vec4_<double>;

/* genIType */
using ivec2 = vec2_<int>;
using ivec3 = vec3_<int>;
using ivec4 = vec4_<int>;

/* genOType */
using uint = unsigned int;
using uvec2 = vec2_<unsigned int>;
using uvec3 = vec3_<unsigned int>;
using uvec4 = vec4_<unsigned int>;

/* genBTyoe */
using bvec2 = vec2_<bool>;
using bvec3 = vec3_<bool>;
using bvec4 = vec4_<bool>;

struct sampler1D { };
struct sampler2D { };
struct sampler2DShadow { };
struct sampler2DArray { };
struct sampler2DArrayShadow { };
struct sampler3D { };
struct samplerCube { };
struct samplerBuffer { };

struct  image1D { };
struct iimage1D { };
struct uimage1D { };
struct  image2D { };
struct iimage2D { };
struct uimage2D { };
struct  image3D { };
struct iimage3D { };
struct uimage3D { };

struct atomic_uint { };

template<size_t M, size_t N, typename _Ty>
struct mat_ {
	struct mat_col {
		mat_col(vec2_<_Ty>);
		mat_col(vec3_<_Ty>);
		mat_col(vec4_<_Ty>);

		float& operator[](size_t) const;
		float operator[](size_t) const;
	};

	mat_(_Ty _All);
	mat_(mat_col);
	mat_(mat_col, mat_col);
	mat_(mat_col, mat_col, mat_col);
	mat_(mat_col, mat_col, mat_col, mat_col);
	template<size_t M2, size_t N2> mat_(mat_<M2, N2, _Ty>);
	
	mat_col& operator[](size_t);
	mat_col operator[](size_t) const;
};

using mat2 = mat_<2, 2, float>;
using mat3 = mat_<3, 3, float>;
using mat4 = mat_<4, 4, float>;
using mat3x4 = mat_<3, 4, float>;

using dmat2 = mat_<2, 2, double>;
using dmat3 = mat_<3, 3, double>;
using dmat4 = mat_<4, 4, double>;
using dmat3x4 = mat_<3, 4, double>;

//using genType  = float/* | vec2 | vec3 | vec4 | mat2 | mat3 | mat4 */;
//using genIType = int/* | ivec2 | ivec3 | ivec4 | imat2 | imat3 | imat4 | uint | uvec2 | uvec3 | uvec4 | umat2 | umat3 | umat4 */;
//// requires{ #version 410 core }
//using genDType = double/* | dvec2 | dvec3 | dvec4 | dmat2 | dmat3 | dmat4 */;

/* vertex and tess  */
static vec4  gl_Position;
static float gl_PointSize;
static float gl_ClipDistance[];
static float gl_CullDistance[];

/* only vertex shader */
static const int gl_VertexID;
static const int gl_InstanceID;

/* only tess shader */
static const int gl_PatchVerticesIn;
/*     const int gl_PrimitiveID */
static const int gl_InvocationID;
static float gl_TessLevelOuter[4];
static float gl_TessLevelInner[4];

/* only geometry shader */
static const int gl_PrimitiveIDIn;
static int gl_PrimitiveID;
static int gl_Layer;
static int gl_ViewportIndex;

/* only fragment shader */
static const vec4  gl_FragCoord;
static const bool  gl_FrontFacing;
/*     const float gl_ClipDistance[] */
/*     const float gl_CullDistance[] */
static const vec2  gl_PointCoord;
/*     const int   gl_PrimitiveID */
static const int   gl_SampleID;
static const vec2  gl_SamplePosition;
static const int   gl_SampleMaskIn[];
/*     const int   gl_Layer */
/*     const int   gl_ViewportIndex */
static const bool  gl_HelperInvocation;
static float gl_FragDepth;
static int   gl_SampleMask[];
static vec4  gl_FragColor;

/* only compute shader */
static const uvec3 gl_NumWorkGroups;
static const uvec3 gl_WorkGroupSize;
static const uvec3 gl_WorkGroupID;
static const uvec3 gl_LocalInvocationID;
static const uvec3 gl_GlobalInvocationID;
static const uint  gl_LocalInvocationIndex;


template<typename genType> bool isnan(genType A);
bvec2 isnan(vec2 A);
bvec3 isnan(vec3 A);
bvec4 isnan(vec4 A);

template<typename genType> bool isinf(genType A);
bvec2 isinf(vec2 A);
bvec3 isinf(vec3 A);
bvec4 isinf(vec4 A);

template<typename genType>
genType dFdx(genType p);
template<typename genType>
genType dFdy(genType p);
template<typename genType>
genType dFdxCoarse(genType p);
template<typename genType>
genType dFdyCoarse(genType p);
template<typename genType>
genType dFdxFine(genType p);
template<typename genType>
genType dFdyFine(genType p);

template<typename genType> 
genType abs(genType x);
/* A>0 ? 1, A=0 ? 0, A<0 ? -1 */
template<typename genType> 
genType sign(genType x);
template<typename genType> 
genType floor(genType x);
template<typename genType> 
genType ceil(genType x);
template<typename genType> 
genType trunc(genType x);
template<typename genType> 
genType fract(genType x);
template<typename genType> 
genType round(genType x);

template<typename genType> 
genType mod(genType x, genType y);
template<typename genType, typename genT> 
genType mod(genType x, genT y);

template<typename genType>
genType min(genType A, genType B);
template<typename genType>
genType max(genType A, genType B);

template<typename genType >
genType clamp(genType  x, genType  minVal, genType  maxVal);
template<typename genType, typename genT >
genType clamp(genType  x, genT  minVal, genT  maxVal);

template<typename genType>
genType mix(genType x, genType y, genType t);
template<typename genType, typename genT>
genType mix(genType x, genType y, genT t);

/* t = clamp( (x-edge0) / (edge1-edge0), 0, 1)
   s_curve( t );
*/
template<typename genType>
genType smoothstep(genType edge0, genType edge1, genType x);
template<typename genT, typename genType> 
genType smoothstep(genT edge0, genT edge1, genType x);

/* round( clamp(A, 0, 1) * 65535 ) */
uint packUnorm2x16(vec2 A);
/* round( clamp(A, -1, 1) * 32767 )*/
uint packSnorm2x16(vec2 A);
/* round( clamp(A, 0, 1) * 256 ) */
uint packUnorm4x8(vec4 A);
/* round( clamp(A, -1, 1) * 127 )*/
uint packSnorm4x8(vec4 A);
/* clamp( A/65535, 0, 1) */
vec2 unpackUnorm2x16(uint A);
/* clamp( A/32767, -1, 1) */
vec2 unpackSnorm2x16(uint A);
/* clamp( A/256, 0, 1) */
vec4 unpackUnorm4x8(uint A);
/* clamp( A/127, -1, 1) */
vec4 unpackSnorm4x8(uint A);

template<typename genType> 
genType radians(genType degrees);
template<typename genType>
genType degrees(genType radians);

template<typename genType> genType sin(genType radians);
template<typename genType> genType cos(genType radians);
template<typename genType> genType tan(genType radians);
template<typename genType> genType asin(genType x);
template<typename genType> genType acos(genType x);
template<typename genType> genType atan(genType y_over_x);
template<typename genType> genType atan(genType y, genType x);
/* ( exp(x) - exp(-x) ) / 2 */
template<typename Ty> Ty sinh(Ty);
/* ( exp(x) + exp(-x) ) / 2 */
template<typename Ty> Ty cosh(Ty);
/* sinh(x)/cosh(x) */
template<typename Ty> Ty tanh(Ty);
template<typename Ty> Ty asinh(Ty);
template<typename Ty> Ty acosh(Ty);
template<typename Ty> Ty atanh(Ty);

/* n power of x */
template<typename _Ty> _Ty pow(_Ty x, _Ty n);
/* n power of e */
template<typename _Ty> _Ty exp(_Ty n);
/* log(e, n) */
template<typename _Ty> _Ty log(_Ty n);
/* n power of 2 */
template<typename _Ty> _Ty exp2(_Ty n);
/* log(2, n) */
template<typename _Ty> _Ty log2(_Ty n);
/* sqrt(x) */
template<typename _Ty> _Ty sqrt(_Ty x);
/* 1/sqrt(x) */
template<typename _Ty> _Ty invsqrt(_Ty x);

/* sum(A.i * B.i) */
template<typename _Ty> _Ty dot(vec2_<_Ty> v0, vec2_<_Ty> v1);
/* sum(A.i * B.i) */
template<typename _Ty> _Ty dot(vec3_<_Ty> v0, vec3_<_Ty> v1);
/* sum(A.i * B.i) */
template<typename _Ty> _Ty dot(vec4_<_Ty> v0, vec4_<_Ty> v1);

/* A.yzx*B.zxy - A.zxy*B.yzx */
template<typename _Ty> vec3_<_Ty> cross(vec3_<_Ty>, vec3_<_Ty>);

/* sqrt( sum(A.i*A.i) ) */
template<typename _Ty> _Ty length(vec2_<_Ty>);
/* sqrt( sum(A.i*A.i) ) */
template<typename _Ty> _Ty length(vec3_<_Ty>);
/* sqrt( sum(A.i*A.i) ) */
template<typename _Ty> _Ty length(vec4_<_Ty>);

/* sqrt( sum(A.i*B.i) ) */
template<typename _Ty> _Ty distance(vec2_<_Ty>, vec2_<_Ty>);
/* sqrt( sum(A.i*B.i) ) */
template<typename _Ty> _Ty distance(vec3_<_Ty>, vec3_<_Ty>);
/* sqrt( sum(A.i*B.i) ) */
template<typename _Ty> _Ty distance(vec4_<_Ty>, vec4_<_Ty>);

/* normalize(v) */
template<typename _Ty> vec2_<_Ty> normalize(vec2_<_Ty> v);
/* normalize(v) */
template<typename _Ty> vec3_<_Ty> normalize(vec3_<_Ty> v);
/* normalize(v) */
template<typename _Ty> vec4_<_Ty> normalize(vec4_<_Ty> v);

/* dot(I, Nref) < 0 ? N : -N */
template<typename _Ty> vec2_<_Ty> faceforward(vec2_<_Ty> N, vec2_<_Ty> I, vec2_<_Ty> _Nref);
/* dot(I, Nref) < 0 ? N : -N */
template<typename _Ty> vec3_<_Ty> faceforward(vec3_<_Ty> N, vec3_<_Ty> I, vec3_<_Ty> Nref);
/* dot(I, Nref) < 0 ? N : -N */
template<typename _Ty> vec4_<_Ty> faceforward(vec4_<_Ty> N, vec4_<_Ty> I, vec4_<_Ty> Nref);

/* V - 2 * proj(V,N) */
template<typename _Ty> vec2_<_Ty> reflect(vec2_<_Ty> V, vec2_<_Ty> N);
/* V - 2 * proj(V,N) */
template<typename _Ty> vec3_<_Ty> reflect(vec3_<_Ty> V, vec3_<_Ty> N);
/* V - 2 * proj(V,N) */
template<typename _Ty> vec4_<_Ty> reflect(vec4_<_Ty> V, vec4_<_Ty> N);

/* k = 1 - eta*eta*( 1 - dot(V,N)*dot(V,N) ) 
   k < 0 ? 0 : eta*V - (eta*dot(V,N) + sqrt(k))*N
*/
template<typename _Ty> vec2_<_Ty> refract(vec2_<_Ty> V, vec2_<_Ty> N, _Ty eta);
/* k = 1 - eta*eta*( 1 - dot(V,N)*dot(V,N) )
   k < 0 ? 0 : eta*V - (eta*dot(V,N) + sqrt(k))*N
*/
template<typename _Ty> vec3_<_Ty> refract(vec3_<_Ty> V, vec3_<_Ty> N, _Ty eta);
/* k = 1 - eta*eta*( 1 - dot(V,N)*dot(V,N) )
   k < 0 ? 0 : eta*V - (eta*dot(V,N) + sqrt(k))*N
*/
template<typename _Ty> vec4_<_Ty> refract(vec4_<_Ty> V, vec4_<_Ty> N, _Ty eta);

/* A.i < B.i */
template<typename _Ty> bvec2 lessThan(vec2_<_Ty> A, vec2_<_Ty> B);
template<typename _Ty> bvec3 lessThan(vec3_<_Ty> A, vec3_<_Ty> B);
template<typename _Ty> bvec4 lessThan(vec4_<_Ty> A, vec4_<_Ty> B);

/* A.i <= B.i */
template<typename _Ty> bvec2 lessThanEqual(vec2_<_Ty> A, vec2_<_Ty> B);
template<typename _Ty> bvec3 lessThanEqual(vec3_<_Ty> A, vec3_<_Ty> B);
template<typename _Ty> bvec4 lessThanEqual(vec4_<_Ty> A, vec4_<_Ty> B);

/* A.i > B.i */
template<typename _Ty> bvec2 greaterThan(vec2_<_Ty> A, vec2_<_Ty> B);
template<typename _Ty> bvec3 greaterThan(vec3_<_Ty> A, vec3_<_Ty> B);
template<typename _Ty> bvec4 greaterThan(vec4_<_Ty> A, vec4_<_Ty> B);

/* A.i >= B.i */
template<typename _Ty> bvec2 greaterThanEqual(vec2_<_Ty> A, vec2_<_Ty> B);
template<typename _Ty> bvec3 greaterThanEqual(vec3_<_Ty> A, vec3_<_Ty> B);
template<typename _Ty> bvec4 greaterThanEqual(vec4_<_Ty> A, vec4_<_Ty> B);

/* A.i == B.i */
template<typename _Ty> bvec2 equal(vec2_<_Ty> A, vec2_<_Ty> B);
template<typename _Ty> bvec3 equal(vec3_<_Ty> A, vec3_<_Ty> B);
template<typename _Ty> bvec4 equal(vec4_<_Ty> A, vec4_<_Ty> B);

/* A.i != B.i */
template<typename _Ty> bvec2 notEqual(vec2_<_Ty> A, vec2_<_Ty> B);
template<typename _Ty> bvec3 notEqual(vec3_<_Ty> A, vec3_<_Ty> B);
template<typename _Ty> bvec4 notEqual(vec4_<_Ty> A, vec4_<_Ty> B);

/* A[0] | A[1] | ... | A[n] */
bool any(bvec2 A);
bool any(bvec3 A);
bool any(bvec4 A);

/* A[0] & A[1] & ... & A[n] */
bool all(bvec2 A);
bool all(bvec3 A);
bool all(bvec4 A);

bvec2 NOT(bvec2);
bvec3 NOT(bvec3);
bvec4 NOT(bvec4);

#pragma region mat
template<size_t M, size_t N, typename _Ty> 
mat_<N, M, _Ty> transpose(mat_<M, N, _Ty>);

float determinant(mat2);
float determinant(mat3);
float determinant(mat4);

mat2 inverse(mat2 m);
mat3 inverse(mat3 m);
mat4 inverse(mat4 m);
dmat2 inverse(dmat2 m);
dmat3 inverse(dmat3 m);
dmat4 inverse(dmat4 m);

mat2 operator*(mat2, mat2);
mat3 operator*(mat3, mat3);
mat4 operator*(mat4, mat4);
dmat2 operator*(dmat2, dmat2);
dmat3 operator*(dmat3, dmat3);
dmat4 operator*(dmat4, dmat4);

vec2 operator*(mat2, vec2);
vec3 operator*(mat3, vec3);
vec4 operator*(mat4, vec4);
vec2 operator*(vec2, mat2);
vec3 operator*(vec3, mat3);
vec4 operator*(vec4, mat4);
#pragma endregion



#pragma region texture
int   textureSize(sampler1D, int Lod);
ivec2 textureSize(sampler2D, int Lod);
ivec3 textureSize(sampler2DArray, int Lod);
ivec3 textureSize(sampler3D,   int Lod);
ivec2 textureSize(samplerCube, int Lod);
ivec3 textureSize(sampler2DArrayShadow, int Lod);
ivec2 textureSize(sampler2DShadow, int Lod);
int   textureSize(samplerBuffer);

int textureQueryLevels(sampler1D);
int textureQueryLevels(sampler2D);
int textureQueryLevels(sampler2DArray);
int textureQueryLevels(sampler3D);
int textureQueryLevels(samplerCube);
int textureQueryLevels(sampler2DShadow);

vec4  texture(sampler1D,      float P, float bias = 0.f);
vec4  texture(sampler2D,       vec2 P, float bias = 0.f);
/* index: round(P[2]) */
vec4  texture(sampler2DArray,  vec3 P, float bias = 0.f);
vec4  texture(sampler3D,       vec3 P, float bias = 0.f);
vec4  texture(samplerCube,     vec3 P, float bias = 0.f);
float texture(sampler2DShadow, vec3 P, float bias = 0.f);
/* P:{ posX, posY, posZ, refZ } */
float texture(sampler2DArrayShadow, vec4 P);

/* unsupport cubemap */
vec4  textureOffset(sampler1D,      float P, int   offset, float bias = 0.f);
vec4  textureOffset(sampler2D,       vec2 P, ivec2 offset, float bias = 0.f);
vec4  textureOffset(sampler2DArray,  vec3 P, ivec2 offset, float bias = 0.f);
vec4  textureOffset(sampler3D,       vec3 P, ivec3 offset, float bias = 0.f);
float textureOffset(sampler2DShadow, vec3 P, ivec2 offset, float bias = 0.f);

vec4  texelFetch(sampler1D, int P, int Lod);
vec4  texelFetch(sampler2D, ivec2 P, int Lod);
vec4  texelFetch(sampler2DArray, ivec3 P, int Lod);
vec4  texelFetch(sampler3D, ivec3 P, int Lod);

vec4 textureGather(sampler2D,      vec2 P, int comp = 0);
vec4 textureGather(sampler2DArray, vec3 P, int comp = 0);
vec4 textureGather(sampler2D,      vec2 P, ivec2 offset,int comp = 0);
vec4 textureGather(sampler2DArray, vec3 P, ivec2 offset, int comp = 0);
vec4 textureGather(samplerCube,          vec3 P, int comp = 0);
vec4 textureGather(sampler2DShadow,      vec2 P, float refZ);
vec4 textureGather(sampler2DShadow,      vec2 P, float refZ, ivec2 offset);
vec4 textureGather(sampler2DArrayShadow, vec3 P, float refZ);

int   imageSize( image1D);
int   imageSize(iimage1D);
int   imageSize(uimage1D);
ivec2 imageSize( image2D);
ivec2 imageSize(iimage2D);
ivec2 imageSize(uimage2D);
ivec3 imageSize( image3D);
ivec3 imageSize(iimage3D);
ivec3 imageSize(uimage3D);
 vec4 imageLoad( image1D, int);
ivec4 imageLoad(iimage1D, int);
uvec4 imageLoad(uimage1D, int);
 vec4 imageLoad( image2D, ivec2);
ivec4 imageLoad(iimage2D, ivec2);
uvec4 imageLoad(uimage2D, ivec2);
 vec4 imageLoad( image3D, ivec3);
ivec4 imageLoad(iimage3D, ivec3);
uvec4 imageLoad(uimage3D, ivec3);
void imageStore( image1D, int,  vec4);
void imageStore(iimage1D, int, ivec4);
void imageStore(uimage1D, int, uvec4);
void imageStore( image2D, ivec2,  vec4);
void imageStore(iimage2D, ivec2, ivec4);
void imageStore(uimage2D, ivec2, uvec4);
void imageStore( image3D, ivec3,  vec4);
void imageStore(iimage3D, ivec3, ivec4);
void imageStore(uimage3D, ivec3, uvec4);

/* img[P] += data, return old_val, Note:r32ui */
uint imageAtomicAdd(uimage1D, int P, uint data);
uint imageAtomicAdd(uimage2D, int P, uint data);
uint imageAtomicAdd(uimage3D, int P, uint data);
/* img[P] += data, return old_val, Note:r32i */
 int imageAtomicAdd(iimage1D, int P,  int data);
 int imageAtomicAdd(iimage2D, int P,  int data);
 int imageAtomicAdd(iimage3D, int P,  int data);

/* img[P] &= data, return old_val, Note:r32ui */
uint imageAtomicAnd(uimage1D, int P, uint data);
uint imageAtomicAnd(uimage2D, int P, uint data);
uint imageAtomicAnd(uimage3D, int P, uint data);
/* img[P] &= data, return old_val, Note:r32i */
 int imageAtomicAnd(iimage1D, int P,  int data);
 int imageAtomicAnd(iimage2D, int P,  int data);
 int imageAtomicAnd(iimage3D, int P,  int data);

/* img[P] |= data, return old_val, Note:r32ui */
uint imageAtomicOr(uimage1D, int P, uint data);
uint imageAtomicOr(uimage2D, int P, uint data);
uint imageAtomicOr(uimage3D, int P, uint data);
/* img[P] |= data, return old_val, Note:r32i */
 int imageAtomicOr(iimage1D, int P,  int data);
 int imageAtomicOr(iimage2D, int P,  int data);
 int imageAtomicOr(iimage3D, int P,  int data);

/* img[P] ^= data, return old_val, Note:r32ui */
uint imageAtomicXor(uimage1D, int P, uint data);
uint imageAtomicXor(uimage2D, int P, uint data);
uint imageAtomicXor(uimage3D, int P, uint data);
/* img[P] ^= data, return old_val, Note:r32i */
 int imageAtomicXor(iimage1D, int P,  int data);
 int imageAtomicXor(iimage2D, int P,  int data);
 int imageAtomicXor(iimage3D, int P,  int data);

/* img[P] = data, return old_val, Note:r32ui */
uint imageAtomicExchange(uimage1D, int P, uint data);
uint imageAtomicExchange(uimage2D, int P, uint data);
uint imageAtomicExchange(uimage3D, int P, uint data);
/* img[P] = data, return old_val, Note:r32i */
 int imageAtomicExchange(iimage1D, int P,  int data);
 int imageAtomicExchange(iimage2D, int P,  int data);
 int imageAtomicExchange(iimage3D, int P,  int data);
#pragma endregion


#pragma region atomic
uint atomicCounter(atomic_uint);
/* return old_val */
uint atomicCounterIncrement(atomic_uint);
/* return old_val */
uint atomicCounterDecrement(atomic_uint);

/* mem += s return old_val */
uint atomicAdd(_inout(uint) mem, uint);
int  atomicAdd(_inout(int) mem, int);

/* mem &= s */
uint atomicAnd(_inout(uint) mem, uint);
int  atomicAnd(_inout(int) mem, int);

/* mem |= s */
uint atomicOr(_inout(uint) mem, uint);
int  atomicOr(_inout(int) mem, int);

/* mem ^= s */
uint atomicXor(_inout(uint) mem, uint);
int  atomicXor(_inout(int) mem, int);

/* mem = s */
uint atomicExchange(_inout(uint) mem, uint s);
int  atomicExchange(_inout(int) mem, int);
#pragma endregion


/* only geometry shader */
void EmitStreamVertex(int);
void EndStreamPrimitive(int);
void EmitVertex();
void EndPrimitive();

/* only fragment shader */
template<typename _Ty> _Ty ddx(_Ty x);
template<typename _Ty> _Ty ddy(_Ty x);
template<typename _Ty> _Ty ddx_fine(_Ty x);
template<typename _Ty> _Ty ddy_fine(_Ty x);
template<typename _Ty> _Ty ddx_coarse(_Ty x);
template<typename _Ty> _Ty ddy_coarse(_Ty x);
#endif