#define _in(TYPE) TYPE
#define _out(TYPE) out TYPE
#define _inout(TYPE) out TYPE;
#define Real float
#define Vec2 vec2
#define Vec3 vec3
#define Vec4 vec4
#define Mat2 mat2
#define Mat3 mat3
#define Mat4 mat4


bool quadratic(_in(Real) _A, _in(Real) _B, _in(Real) _C, _out(Real) _t0, _out(Real) _t1)
	{	// [-B＼〔(B^2-4AC)] / 2A
	Real _Discrim = _B * _B - 4.0 * _A * _C;
	
	if( _Discrim < 0.0 )
		{	// no solve
		return (false);
		}
	else
		{
		_Discrim = sqrt(_Discrim);
		Real _Tmp = (_B < 0.0) ? -0.5*(_B - _Discrim) : -0.5*(_B + _Discrim);
		_t0 = _Tmp / _A;
		_t1 = _C / _Tmp;
		if( _t0 > _t1 )
			{	// swap 
			_Tmp = _t0;
			_t0 = _t1;
			_t1 = _Tmp;
			}
		return (true);
		}
	}

/* < struct _Tag_normal > */
struct _Tag_normal{ int a; };
/* </ struct _Tag_normal > */

/* < struct MatrixTransform > */
struct MatrixTransform
{
	Mat4 _Mymat;
	Mat4 _Myinvmat;
};

MatrixTransform create_MatrixTransform(_in(Mat4) _Matrix)
{ return ( MatrixTransform(_Matrix, inverse(_Matrix)) ); }

Vec3 transform( _in(MatrixTransform) _Transform, _in(Vec3) _Vec )
{ return ( Vec3(_Transform._Mymat * Vec4(_Vec, 1.0)) ); }

Vec3 transform( _in(MatrixTransform) _Transform, _in(Vec3) _Normal, _Tag_normal )
{ return ( transpose(Mat3(_Transform._Myinvmat)) * _Normal ); }

Vec3 inv_transform( _in(MatrixTransform) _Transform, _in(Vec3) _Vec )
{ return ( Vec3( _Transform._Myinvmat * Vec4(_Vec, 1.0) ) ); }

Vec3 inv_transform( _in(MatrixTransform) _Transform, _in(Vec3) _Normal, _Tag_normal )
{ return ( transpose(Mat3(_Transform._Mymat)) * _Normal ); }
/* </ struct MatrixTransform > */


/* < struct Hit > */
struct Hit
{
	int _Myresult;
	Real _Myt;
	Vec3 _Myorigin;
	Vec3 _Mynormal;
	Vec2 _Myuv;
};

Hit no_hit()
{ return ( Hit( 0, 0.0, Vec3(0.0), Vec3(0.0), Vec2(0.0) ) ); }
/* </ struct Hit > */


/* < struct Sphere> */
const int SPHERE = 1;
struct Sphere
{
	MatrixTransform _Myworld;
	Real _Myradius;
	Real _Mytheta_min;
	Real _Mytheta_max;
	Real _Myphi_max;
};

Sphere create_Sphere(_in(Mat4) _World, _in(Real) _Radius, _in(Real) _Thetamin, _in(Real) _Thetamax, _in(Real) _Phimax)
{ return ( Sphere(create_MatrixTransform(_World), _Radius, _Thetamin, _Thetamax, _Phimax) ); }
/* </ struct Sphere> */


/* < struct Cylinder > */
const int CYLINDER = 2;
struct Cylinder
{
	MatrixTransform _Myworld;
	Real _Myradius;
	Real _Myht_min;
	Real _Myht_max;
	Real _Myphi_max;
};

Cylinder create_Cylinder( _in(Mat4) _World, _in(Real) _Radius, _in(Real) _Heightmin, _in(Real) _Heightmax, _in(Real) _Phimax )
{ return ( Cylinder( create_MatrixTransform(_World), _Radius, _Heightmin, _Heightmax, _Phimax ) ); }
/* </ struct Cylinder > */


/* < struct Triangle > */
const int TRIANGLE = 6;
struct Triangle
{
	Vec3 _Myp0;
	Vec3 _Myp1;
	Vec3 _Myp2;
};

Triangle create_Triangle( _in(Vec3) _P0, _in(Vec3) _P1, _in(Vec3) _P2)
{ return ( Triangle(_P0, _P1, _P2) ); }
/* </ struct Triangle > */


/* < struct Ray > */
struct Ray
{
	Vec3 _Myorigin;
	Real _Mymint;
	Vec3 _Mydirection;
	Real _Mymaxt;
};

Ray create_Ray( _in(Vec3) _Origin, _in(Vec3) _Direction )
{ return ( Ray(_Origin, 0.0, _Direction, 10000000.0) ); }


bool intersect( _in(Ray) _Ray, _in(Sphere) _Sphere, _out(Hit) _Result)
	{	// intersect: Ray and Sphere
	/*
	x^2 + y^2 + z^2 - r^2 = 0
	((o+td).x)^2 + ((o+td).y)^2 + ((o+td).z)^2 - r^2 = 0
	(o.x+td.x)*(o.x+td.x) + (o.y+td.y)*(o.y+td.y) + (o.z+td.z)*(o.z+td.z) - r^2 = 0
	t^2*(d.x^2 + d.y^2 + d.z^2) + t*2*(o.x*d.x + o.y*d.y + o.z*d.z) + o.x^2 + o.y^2 + o.z^2 - r^2 = 0
	*/
	Real _r = _Sphere._Myradius;
	Vec3 _o = inv_transform(_Sphere._Myworld, _Ray._Myorigin);
	Vec3 _d = inv_transform(_Sphere._Myworld, _Ray._Mydirection, _Tag_normal(1));
	Real _A = _d.x * _d.x + _d.y * _d.y + _d.z * _d.z;
	Real _B = 2.0 * (_o.x * _d.x + _o.y * _d.y + _o.z * _d.z);
	Real _C = _o.x * _o.x + _o.y * _o.y + _o.z * _o.z - _r * _r;
	Real _t0, _t1;
	if( !quadratic(_A, _B, _C, _t0, _t1) )
		return ( false );// no solution

	if(ceil(_t0) < _Ray._Mymint || floor(_t1) > _Ray._Mymaxt)
		return ( false );// out of sight

	Real _t = _t0;
	Vec3 _Impact = _o + _t * _d;
	Real _Phi = atan(_Impact.y, _Impact.x); if(_Phi < 0.0) _Phi += 6.283185482;
	Real _Theta = acos( clamp(_Impact.z / _r, -1.0, 1.0) );
	if(_Theta < _Sphere._Mytheta_min || _Theta > _Sphere._Mytheta_max || _Phi > _Sphere._Myphi_max)
		{
		_t = _t1;
		_Impact = _o + _t * _d;
		_Phi = atan(_Impact.y, _Impact.x); if(_Phi < 0.0) _Phi += 6.283185482;
		_Theta = acos( clamp(_Impact.z / _r, -1.0, 1.0) );
		if(_Theta < _Sphere._Mytheta_min || _Theta > _Sphere._Mytheta_max || _Phi > _Sphere._Myphi_max)
			{
			return ( false );// clipping test
			}
		}

	_Result._Myresult = SPHERE;
	_Result._Myt = _t;
	_Result._Myorigin = transform(_Sphere._Myworld, _Impact);
	_Result._Mynormal = transform(_Sphere._Myworld, _Impact / _r, _Tag_normal(1));
	_Result._Myuv.x = _Phi / _Sphere._Myphi_max;
	_Result._Myuv.y = (_Theta - _Sphere._Mytheta_min) / (_Sphere._Mytheta_max - _Sphere._Mytheta_min);
	return (true);
	}


bool intersect( _in(Ray) _Ray, _in(Cylinder) _Cylinder, _out(Hit) _Result )
	{	// intersect: Ray and Cylinder
	Vec3 _o = inv_transform( _Cylinder._Myworld, _Ray._Myorigin );
	Vec3 _d = inv_transform( _Cylinder._Myworld, _Ray._Mydirection, _Tag_normal(1) );
	Real _r = _Cylinder._Myradius;
	/*
	x^2 + z^2 - r^2 = 0
	(o+td).x^2 + (o+td).z^2 - r^2 = 0
	o.x^2 + (td.x)^2 + 2(o+td).x + o.z^2 + (td.z)^2 + 2(o+td).z - r^2 = 0
	t^2*(d.x^2 + d.z^2) + t*(2*(o.x*d.x + o.z*d.z)) + (o.x^2 + o.z^2 - r^2) = 0
	*/
	Real _A = _d.x * _d.x + _d.z * _d.z;
	Real _B = 2.0 * (_o.x * _d.x + _o.z * _d.z);
	Real _C = _o.x * _o.x + _o.z * _o.z - _r * _r; 
	Real _t0, _t1;
	if( !quadratic(_A, _B, _C, _t0, _t1) )
		return (false);// on solution

	if( ceil(_t0) < _Ray._Mymint || floor(_t1) > _Ray._Mymaxt )
		return (false);// out of sight

	Real _t = _t0;
	Vec3 _Impact = _o + _t * _d;
	Real _Phi = atan(_Impact.z, _Impact.x); if(_Phi < 0.0) _Phi + 6.283185482;
	if( _Impact.y < _Cylinder._Myht_min || _Impact.y > _Cylinder._Myht_max || _Phi > _Cylinder._Myphi_max )
		{
		_t = _t1;
		_Impact = _o + _t * _d;
		_Phi = atan(_Impact.z, _Impact.x); if(_Phi < 0.0) _Phi + 6.283185482;
		if( _Impact.y < _Cylinder._Myht_min || _Impact.y > _Cylinder._Myht_max || _Phi > _Cylinder._Myphi_max )
			{
			return (false); // 
			}
		}

	_Result._Myresult = CYLINDER;
	_Result._Myt = _t;
	_Result._Myorigin = transform( _Cylinder._Myworld, _Impact );
	_Result._Mynormal = transform( _Cylinder._Myworld, Vec3(cos(_Phi), 0.0, sin(_Phi)), _Tag_normal(1) );
	_Result._Myuv.x = _Phi / _Cylinder._Myphi_max;
	_Result._Myuv.y = (_Impact.y - _Cylinder._Myht_min) / (_Cylinder._Myht_max - _Cylinder._Myht_min);
	return ( true );
	}

bool intersect(_in(Ray) _Ray, _in(Triangle) _Triangle, _out(Hit) _Result)
	{
	/*
	e1 = p2-p0, e2 = p3-p0, s = o-p0

	[t ]         1		 [(s X e1)，e2]
	[b1] = --------------[(d X e2)，s ]
	[b2]    (d X e2)，e1 [(s X e1)，d ]

	s1 = (d X e2), s2= (s X e1)

	[t ]         1		 [s2，e2]
	[b1] = --------------[s1，s ]
	[b2]       s1，e1    [s2，d ]
	*/
	Vec3 _e1 = _Triangle._Myp1 - _Triangle._Myp0;
	Vec3 _e2 = _Triangle._Myp2 - _Triangle._Myp0;
	Vec3 _s = _Ray._Myorigin - _Triangle._Myp0;

	Vec3 _s1 = cross(_Ray._Mydirection, _e2);
	Real _Invdet = dot(_s1, _e1); if ( abs(_Invdet) < 0.00002 ) return ( false );
	_Invdet = 1.0 / _Invdet;

	Real _b1 = _Invdet * dot(_s1, _s);
	if ( _b1 < 0.0 || _b1 > 1.0 ) 
		return ( false );

	Vec3 _s2 = cross(_s, _e1);
	Real _b2 = _Invdet * dot(_s2, _Ray._Mydirection);
	if ( _b2 < 0.0 || _b2 + _b1 > 1.0 )
		return ( false );

	Real _t = _Invdet * dot(_s2, _e2);
	if ( _t < _Ray._Mymint || _t > _Ray._Mymaxt )
		return ( false );// direction different, or out of sight

	_Result._Myresult = TRIANGLE;
	_Result._Myt = _t;
	_Result._Myorigin = _Ray._Myorigin + _t * _Ray._Mydirection;
	_Result._Mynormal = normalize( cross(_Triangle._Myp1 - _Triangle._Myp0, _Triangle._Myp2 - _Triangle._Myp0) );
	_Result._Myuv = Vec2(_b1, _b2);
	return ( true );
	}

/* </ struct Ray > */





