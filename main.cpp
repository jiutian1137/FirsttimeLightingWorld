#include <map>
#include <array>
#include <vector>
#include <string>
#include <numbers>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include "clmagic/calculation/lapack/vector.h"
#include "clmagic/calculation/lapack/geometry.h"
#include "clmagic/calculation/to_string.h"
#include "clmagic/openglutil/glut.h"
using namespace::calculation;

#include "glfw/glfw3.h"
#pragma comment(lib, "glfw3.lib")

#include "assimp/scene.h"
#include "assimp/cimport.h"
#include "assimp/postprocess.h"
#ifdef _DEBUG
#pragma comment(lib, "assimp-vc142-mtd.lib")
#else 
#pragma comment(lib, "assimp-vc142-mt.lib")
#endif

#include "opencv/opencv.hpp"
#ifdef _DEBUG
#pragma comment(lib, "opencv_world440d.lib")
#else
#pragma comment(lib, "opencv_world440.lib")
#endif

inline
bool glLoadTexture2D(const std::filesystem::path& filename, GLsizei width, GLsizei height, GLuint* texture_ptr) {
	cv::Mat image = cv::imread(filename.generic_string(), cv::IMREAD_UNCHANGED);
	cv::resize(image, image, cv::Size((width == GLsizei(-1)) ? image.cols : width, 
									  (height == GLsizei(-1)) ? image.rows : height));
	cv::flip(image, image, 0);
	
	glDeleteTextures(1, texture_ptr);
	glGenTextures(1, texture_ptr);
	glBindTexture(GL_TEXTURE_2D, *texture_ptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	if (image.channels() == 1) {
		glTexStorage2D(GL_TEXTURE_2D, 1, (image.type() == CV_8UC1 ? GL_R8 : GL_R32F), image.cols, image.rows);
		glTexUpload2D(GL_TEXTURE_2D, 0, 0, 0, image.cols, image.rows, image.data);
	} else {
		if (image.channels() == 4) {// BGRA to RGBA
			cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
		} else if(image.channels() == 3){// BGR to RGBA
			cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
			cv::Mat colors[4];
			cv::split(image, colors);
			colors[3] = cv::Mat(colors[0].rows, colors[0].cols, colors[0].type());
			cv::merge(colors, 4, image);
		} else {// GR to RGBA
			cv::Mat colors[4];
			cv::split(image, colors);
			std::swap(colors[0], colors[1]);
			colors[2] = cv::Mat(colors[0].rows, colors[0].cols, colors[0].type());
			colors[3] = cv::Mat(colors[0].rows, colors[0].cols, colors[0].type());
			cv::merge(colors, 4, image);
		}
		glTexStorage2D(GL_TEXTURE_2D, 1, (image.type() == CV_8UC4 ? GL_RGBA8 : GL_RGBA32F), image.cols, image.rows);
		glTexUpload2D(GL_TEXTURE_2D, 0, 0, 0, image.cols, image.rows, image.data);
	}
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glTestErrorFailedThrow();
	return true;
}

inline 
bool glLoadTexture3D(const std::filesystem::path& filename, GLsizei width, GLsizei height, GLsizei depth, GLuint* texture_ptr) {
	cv::Mat image = cv::imread(filename.generic_string(), cv::IMREAD_UNCHANGED);
	assert(width * height * depth == image.rows * image.cols);
	//cv::resize(image, image, cv::Size(width, height));
	cv::flip(image, image, 0);

	glDeleteTextures(1, texture_ptr);
	glGenTextures(1, texture_ptr);
	glBindTexture(GL_TEXTURE_3D, *texture_ptr);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	if (image.channels() == 1) {
		glTexStorage3D(GL_TEXTURE_3D, 1, (image.type() == CV_8UC1 ? GL_R8 : GL_R32F), width, height, depth);
		glTexUpload3D(GL_TEXTURE_3D, 0, 0, 0, 0, width, height, depth, image.data);
	} else {
		if (image.channels() == 4) {// BGRA to RGBA
			cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
		} else if(image.channels() == 3){// BGR to RGBA
			cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
			cv::Mat colors[4];
			cv::split(image, colors);
			colors[3] = cv::Mat(colors[0].rows, colors[0].cols, colors[0].type());
			cv::merge(colors, 4, image);
		} else {// GR to RGBA
			cv::Mat colors[4];
			cv::split(image, colors);
			std::swap(colors[0], colors[1]);
			colors[2] = cv::Mat(colors[0].rows, colors[0].cols, colors[0].type());
			colors[3] = cv::Mat(colors[0].rows, colors[0].cols, colors[0].type());
			cv::merge(colors, 4, image);
		}
		glTexStorage3D(GL_TEXTURE_3D, 1, (image.type() == CV_8UC4 ? GL_RGBA8 : GL_RGBA32F), width, height, depth);
		glTexUpload3D(GL_TEXTURE_3D, 0, 0, 0, 0, width, height, depth, image.data);
	}
	glBindTexture(GL_TEXTURE_3D, 0);

	glTestErrorFailedThrow();
	return true;
}

template<typename _Ty>
_Ty min(const _Ty& value0, const _Ty& value1, const _Ty& value2) {
	return min(value0, min(value1, value2));
}

template<typename _Ty, typename ..._Tys>
_Ty min(const _Ty& value0, const _Ty& value1, const _Ty& value2, _Tys&... _Values) {
	return min( value0, min( value1, value2, std::forward<_Tys>(_Values)... ) );
}

template<typename _Ty>
_Ty max(const _Ty& value0, const _Ty& value1, const _Ty& value2) {
	return max(value0, max(value1, value2));
}

template<typename _Ty, typename ..._Tys>
_Ty max(const _Ty& value0, const _Ty& value1, const _Ty& value2, _Tys&... _Values) {
	return max( value0, max( value1, value2, std::forward<_Tys>(_Values)... ) );
}

template<typename _Iter>
typename std::iterator_traits<_Iter>::value_type 
	min_value(_Iter _First, const _Iter _Last) {
	// find smallest value
	assert( _First != _Last );
	auto _Min_value = *_First++;

	for ( ; _First != _Last; ++_First) {
		_Min_value = min(_Min_value, *_First);
	}

	return std::move(_Min_value);
}

template<typename _Iter>
typename std::iterator_traits<_Iter>::value_type 
	max_value(_Iter _First, const _Iter _Last) {
	// find largest value
	assert( _First != _Last );
	auto _Max_value = *_First++;

	for ( ; _First != _Last; ++_First) {
		_Max_value = max(_Max_value, *_First);
	}

	return std::move(_Max_value);
}

template<typename _Iter, typename _Diff>
typename std::iterator_traits<_Iter>::value_type
	min_value_n(_Iter _First, _Diff _Count_raw) {
	assert(_Count_raw != 0);
	auto _Min_value = *_First++;
	
	std::_Algorithm_int_t<_Diff> _Count = _Count_raw;
	for ( ; 1 < _Count; --_Count, ++_First) {
		_Min_value = min(_Min_value, *_First);
	}
	
	return std::move(_Min_value);
}

template<typename _Iter, typename _Diff>
typename std::iterator_traits<_Iter>::value_type
	max_value_n(_Iter _First, _Diff _Count_raw) {
	assert(_Count_raw != 0);
	auto _Max_value = *_First++;
	
	std::_Algorithm_int_t<_Diff> _Count = _Count_raw;
	for ( ; 1 < _Count; --_Count, ++_First) {
		_Max_value = max(_Max_value, *_First);
	}
	
	return std::move(_Max_value);
}

struct Vertex {
	std::array<float, 4> position;
	std::array<float, 4> texcoord;
	std::array<float, 4> normal;
	std::array<float, 4> tangent;
	std::array<float, 4> bitangent;

	static std::vector<GLvaryingDesc> descriptor() {
		return std::vector<GLvaryingDesc>{
			GLvaryingDesc{ sizeof(Vertex::position) / sizeof(float), GL_FLOAT, GL_FALSE, sizeof(Vertex), offsetof(Vertex, position) },
			GLvaryingDesc{ sizeof(Vertex::texcoord) / sizeof(float), GL_FLOAT, GL_FALSE, sizeof(Vertex), offsetof(Vertex, texcoord) },
			GLvaryingDesc{ sizeof(Vertex::normal) / sizeof(float), GL_FLOAT, GL_FALSE, sizeof(Vertex), offsetof(Vertex, normal) },
			GLvaryingDesc{ sizeof(Vertex::tangent) / sizeof(float), GL_FLOAT, GL_FALSE, sizeof(Vertex), offsetof(Vertex, tangent) },
			GLvaryingDesc{ sizeof(Vertex::bitangent) / sizeof(float), GL_FLOAT, GL_FALSE, sizeof(Vertex), offsetof(Vertex, bitangent) }
		};
	}
};


cv::VideoWriter video_writer;

std::map<std::string, GLprogram> processor;
std::map<std::string, GLtexture> texture;
std::map<std::string, GLvaryings> vbuffer;
std::map<std::string, GLbuffer>   buffer;

perspective_camera<float> camera;
ortho_camera<float> sun;
std::chrono::milliseconds this_time;


int ScreenResolution[2] = { 1600, 900 };

#define SceneProcessor      "SceneProcessor"
#define SceneShadowProcessor "SceneShadowProcessor"
#define VolumeProcessor       "VolumeProcessor"
#define VolumeShadowProcessor "VolumeShadowProcessor"
#define FinalProcessor         "FinalProcessor"

// scene render target
int SceneShadowResolution[2]  = { 2048, 2048 };
#define SceneShadowTexture   "SceneShadowTexture"
int SceneTextureResolution[2] = { 2048, 2048 };
#define SceneRadianceTexture "SceneRadianceTexture"
#define SceneDepthTexture    "SceneDepthTexture"

// volume render target
int VolumeShadowResolution[2] = { 1024, 1024 };
#define VolumeShadowTexture "VolumeShadowTexture"
#define VolumeTransmittanceTexture "VolumeTransmittanceTexture"
int VolumeTextureResolution[2] = { 1024, 1024 };
#define VolumeRadianceAndTransmittanceTexture "VolumeRadianceAndTransmittanceTexture"
#define VolumeDepthTexture "VolumeDepthTexture"

// scene resource
#define EzeFranceChunk0Vertex  "EzeFranceChunk0Vertex"
#define EzeFranceChunk0Index   "EzeFranceChunk0Index"
#define EzeFranceChunk0Texture "EzeFranceChunk0Texture"

#define EzeFranceChunk1Vertex  "EzeFranceChunk1Vertex"
#define EzeFranceChunk1Index   "EzeFranceChunk1Index"
#define EzeFranceChunk1Texture "EzeFranceChunk1Texture"

#define EzeFranceChunk2Vertex  "EzeFranceChunk2Vertex"
#define EzeFranceChunk2Index   "EzeFranceChunk2Index"
#define EzeFranceChunk2Texture "EzeFranceChunk2Texture"

// volume resource
#define BaseCloudTexture "BaseCloudTexture"
#define CloudDetailTexture "CloudDetailTexture"
#define CloudWeatherTextureFirst "CloudWeatherTextureFirst"
#define CloudWeatherTextureSecond "CloudWeatherTextureSecond"
#define CloudHeightTexture     "CloudHeightTexture"
#define CirrusWeatherTexture "CirrusWeatherTexture"
#define CloudCurlTexture "CloudCurlTexture"

float cloud_density_scale = 0.4f;
float base_cloud_texture_scale = 0.0016f * 0.5f * 0.5f * 0.5f;
float cloud_detail_texture_scale = 0.0016f * 2.0f * 2.0f;

#include "atmosphere/model.h"

enum Luminance {
	// Render the spectral radiance at kLambdaR, kLambdaG, kLambdaB.
	NONE,
	// Render the sRGB luminance, using an approximate (on the fly) conversion
	// from 3 spectral radiance values only (see section 14.3 in <a href=
	// "https://arxiv.org/pdf/1612.04336.pdf">A Qualitative and Quantitative
	//  Evaluation of 8 Clear Sky Models</a>).
	APPROXIMATE,
	// Render the sRGB luminance, precomputed from 15 spectral radiance values
	// (see section 4.4 in <a href=
	// "http://www.oskee.wz.cz/stranka/uploads/SCCG10ElekKmoch.pdf">Real-time
	//  Spectral Scattering in Large-scale Natural Participating Media</a>).
	PRECOMPUTED
};

std::unique_ptr<atmosphere::Model> make_default_EricBrunetonSkyModel(
	std::array<float, 3>* p_white_point = nullptr,
	bool use_constant_solar_spectrum_ = false,
	bool use_ozone_ = true,
	Luminance use_luminance_ = NONE)
{
	constexpr double kPi = std::numbers::pi_v<double>;
	constexpr double kSunAngularRadius = 0.00935 / 2.0;
	constexpr double kSunSolidAngle     = kPi * kSunAngularRadius * kSunAngularRadius;
	constexpr double kLengthUnitInMeters = 1.0;

	// Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
	 // (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
	 // summed and averaged in each bin (e.g. the value for 360nm is the average
	 // of the ASTM G-173 values for all wavelengths between 360 and 370nm).
	 // Values in W.m^-2.
	constexpr int kLambdaMin = 360;
	constexpr int kLambdaMax = 830;
	constexpr double kSolarIrradiance[48] = {
	  1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
	  1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
	  1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
	  1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
	  1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
	  1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
	};
	// Values from http://www.iup.uni-bremen.de/gruppen/molspec/databases/
	// referencespectra/o3spectra2011/index.html for 233K, summed and averaged in
	// each bin (e.g. the value for 360nm is the average of the original values
	// for all wavelengths between 360 and 370nm). Values in m^2.
	constexpr double kOzoneCrossSection[48] = {
	  1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
	  8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
	  1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
	  4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
	  2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
	  6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
	  2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
	};
	// From https://en.wikipedia.org/wiki/Dobson_unit, in molecules.m^-2.
	constexpr double kDobsonUnit = 2.687e20;
	// Maximum number density of ozone molecules, in m^-3 (computed so at to get
	// 300 Dobson units of ozone - for this we divide 300 DU by the integral of
	// the ozone density profile defined below, which is equal to 15km).
	constexpr double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;
	// Wavelength independent solar irradiance "spectrum" (not physically
	// realistic, but was used in the original implementation).
	constexpr double kConstantSolarIrradiance = 1.5;
	constexpr double kBottomRadius = 6360000.0;
	constexpr double kTopRadius = 6420000.0;
	constexpr double kRayleigh = 1.24062e-6;
	constexpr double kRayleighScaleHeight = 8000.0;
	constexpr double kMieScaleHeight = 1200.0;
	constexpr double kMieAngstromAlpha = 0.0;
	constexpr double kMieAngstromBeta = 5.328e-3;
	constexpr double kMieSingleScatteringAlbedo = 0.8;
	constexpr double kMiePhaseFunctionG = 0.76;
	constexpr double kGroundAlbedo = 0.1;
	const double max_sun_zenith_angle =
		(false ? 102.0 : 120.0) / 180.0 * kPi;

	atmosphere::DensityProfileLayer
		rayleigh_layer{ 0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0 };
	atmosphere::DensityProfileLayer
		mie_layer{ 0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0 };

	// Density profile increasing linearly from 0 to 1 between 10 and 25km, and
	// decreasing linearly from 1 to 0 between 25 and 40km. This is an approximate
	// profile from http://www.kln.ac.lk/science/Chemistry/Teaching_Resources/
	// Documents/Introduction%20to%20atmospheric%20chemistry.pdf (page 10).
	std::vector<atmosphere::DensityProfileLayer> ozone_density;
	ozone_density.push_back(
		atmosphere::DensityProfileLayer{ 25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0 });
	ozone_density.push_back(
		atmosphere::DensityProfileLayer{ 0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0 });

	std::vector<double> wavelengths;
	std::vector<double> solar_irradiance;
	std::vector<double> rayleigh_scattering;
	std::vector<double> mie_scattering;
	std::vector<double> mie_extinction;
	std::vector<double> absorption_extinction;
	std::vector<double> ground_albedo;
	for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
		double lambda = static_cast<double>(l) * 1e-3;  // micro-meters
		double mie =
			kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
		wavelengths.push_back(l);
		if (use_constant_solar_spectrum_) {
			solar_irradiance.push_back(kConstantSolarIrradiance);
		}
		else {
			solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
		}
		rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
		mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
		mie_extinction.push_back(mie);
		absorption_extinction.push_back(use_ozone_ ?
			kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] :
			0.0);
		ground_albedo.push_back(kGroundAlbedo);
	}


	atmosphere::AtmosphereParameters parameters;
	parameters.length_unit_in_meters = kLengthUnitInMeters;
	parameters.wavelengths = wavelengths;

	parameters.solar_irradiance      = solar_irradiance;
	parameters.sun_angular_radius    = kSunAngularRadius;
	parameters.bottom_radius         = kBottomRadius;
	parameters.top_radius            = kTopRadius;

	parameters.rayleigh_density.push_back(atmosphere::DensityProfileLayer{ 0.0, 0.0, 0.0, 0.0, 0.0 });
	parameters.rayleigh_density.push_back(rayleigh_layer);
	parameters.rayleigh_scattering = rayleigh_scattering;

	parameters.mie_density.push_back(atmosphere::DensityProfileLayer{ 0.0, 0.0, 0.0, 0.0, 0.0 });
	parameters.mie_density.push_back(mie_layer);
	parameters.mie_scattering = mie_scattering;
	parameters.mie_extinction = mie_extinction;
	parameters.mie_phase_function_g = kMiePhaseFunctionG;

	parameters.absorption_density = ozone_density;
	parameters.absorption_extinction = absorption_extinction;

	parameters.ground_albedo = ground_albedo;

	parameters.max_sun_zenith_angle = max_sun_zenith_angle;

	parameters.num_precomputed_wavelengths = use_luminance_ == PRECOMPUTED ? 15 : 3;

	std::unique_ptr<atmosphere::Model> model_;
	model_.reset(new atmosphere::Model(parameters));
	model_->Init();

	/*std::array<double,3> white___pp_ = physics::ConvertSpectrumToLinearSrgb(wavelengths, solar_irradiance);
	double white_point = (white___pp_[0] + white___pp_[1] + white___pp_[2]) / 3.0;
	white___pp_[0] /= white_point;
	white___pp_[1] /= white_point;
	white___pp_[2] /= white_point;
	*p_white_point = { float(white___pp_[0]), float(white___pp_[1]), float(white___pp_[2]) };*/

	return model_;
}

std::unique_ptr<atmosphere::Model> skymodel = nullptr;

void _Update_sun_projection() {
	vector4<double> box_points[8] = {// order: left to right, bottom to top, near to far
		vector4<double>{ -1.0, -1.0, 0.0, 1.0 },
		vector4<double>{ +1.0, -1.0, 0.0, 1.0 },
		vector4<double>{ -1.0, +1.0, 0.0, 1.0 },
		vector4<double>{ +1.0, +1.0, 0.0, 1.0 },
		vector4<double>{ -1.0, -1.0, 1.0, 1.0 },
		vector4<double>{ +1.0, -1.0, 1.0, 1.0 },
		vector4<double>{ -1.0, +1.0, 1.0, 1.0 },
		vector4<double>{ +1.0, +1.0, 1.0, 1.0 }
	};

	// 1. Construct camera's frustom
	matrix4x4<double> frustom_to_world = 
		inverse( camera.projection_matrix<matrix4x4<double>>() *
				 camera.view_matrix <matrix4x4<double>>() );
	std::for_each(box_points, box_points + 8,
		[frustom_to_world](vector4<double>& point) {
			point = frustom_to_world * point;
			point /= point[3];
		}
	);

	// 2. Convert camera's frustom to sun_view's frustom
	matrix4x4<double> world_to_camera = sun.view_matrix<matrix4x4<double>>();
	std::for_each(box_points, box_points + 8, 
		[world_to_camera](vector4<double>& point) {
			point = world_to_camera * point;
			point /= point[3];
		}
	);

	// 3. AABB and sphere in sun_view
	vector4<double> sph_center = world_to_camera * vector4<double>{ 0.0, -6360000.0, 0.0, 1.0 };
	double          sph_radius = 6360000.0 + 7000.0;
	vector4<double> b_min = min_value_n(box_points, 8);
	vector4<double> b_max = max_value_n(box_points, 8);
	
	if ( aabox_test_sphere(b_min, b_max, sph_center, sph_radius) ) {
		double left   = b_min[0];
		double right  = b_max[0];
		double bottom = b_min[1];
		double top    = b_max[1];
		double far    = b_max[2];
		double t0 = ray_intersect_sphere_surface(vector3<double>{ left, bottom,  far }, vector3<double>{ 0, 0, -1 }, vector3_cast<vector3<double>,0,1,2>(sph_center), sph_radius);
		double t1 = ray_intersect_sphere_surface(vector3<double>{ right, bottom, far }, vector3<double>{ 0, 0, -1 }, vector3_cast<vector3<double>,0,1,2>(sph_center), sph_radius);
		double t2 = ray_intersect_sphere_surface(vector3<double>{ left,  top,    far }, vector3<double>{ 0, 0, -1 }, vector3_cast<vector3<double>,0,1,2>(sph_center), sph_radius);
		double t3 = ray_intersect_sphere_surface(vector3<double>{ right, top,    far }, vector3<double>{ 0, 0, -1 }, vector3_cast<vector3<double>,0,1,2>(sph_center), sph_radius);
		t0 = isnan(t0) ? 0.0 : t0;
		t1 = isnan(t1) ? 0.0 : t1;
		t2 = isnan(t2) ? 0.0 : t2;
		t3 = isnan(t3) ? 0.0 : t3;
		double near = far - max(t0, max(t1, max(t2, t3)));

		std::cout << "t0:" << t0 << ",t1:" << t1 << ",t2:" << t2 << "t3:" << t3 << std::endl;

		sun._Myhorizon = { float(left), float(right) };
		sun._Myvertical = { float(bottom), float(top) };
		sun._Mydepthrange = { float(near), float(far) };
		/*auto a = sun.f_vector();
		auto h = sun._Myhorizon;
		auto v = sun._Myvertical;
		auto d = sun._Mydepthrange;
		std::cout << a[0] << "," << a[1] << "," << a[2] << "," << std::endl;
		std::cout << h[0] << "," << h[1] << std::endl;
		std::cout << v[0] << "," << v[1] << std::endl;
		std::cout << d[0] << "," << d[1] << std::endl;
		std::cout << std::endl;*/
	}
}

void sceneInit() {
	/*<test macro assign>
		GLshadersource source = GLshadersource(
			std::string("#version 330 core\n"
			"#include \"atmosphere/functions.glsl.h\" "),
			std::map<std::string, std::string>{ {"Length", "10000[m]"}, {"Wavelength", "100[ns]"} }
		);
	</test macro assign>*/

	camera = calculation::perspective_camera<float>(
		{ 123.6f, 72.5241f, 212.012f }, 
		{ 0.0f, 0.0f, 1.0f }, 
		3.14f * 0.333f);

	sun._Myrot = { 0.0f, 1.57f, 0.0f };
	sun._Myhorizon = { -350.0f, 350.0f };
	sun._Myvertical = { -350.0f, 350.0f };
	sun._Mydepthrange = { -350.0f, 350.0f };
	
	/////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////

	processor[SceneProcessor] = GLprogram(
		GLvertshader(GLshadersource(std::filesystem::path("asset_shader/scene/scene.vert"))), 
		GLfragshader(GLshadersource(std::filesystem::path("asset_shader/scene/texture.frag.h"))));

	texture[SceneRadianceTexture]
		= GLtexture2D(GL_RGBA32F, SceneTextureResolution[0], SceneTextureResolution[1]);
	texture[SceneDepthTexture]
		= GLtexture2D(GL_DEPTH_COMPONENT32F, SceneTextureResolution[0], SceneTextureResolution[1]);
	glBindTexture(GL_TEXTURE_2D, texture[SceneDepthTexture]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	const aiScene* EzeFrance
		= aiImportFile("asset_model/EzeFrance/EzeFrance.obj", 
			aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_CalcTangentSpace);
	for (size_t i = 0; i != EzeFrance->mNumMeshes; ++i) {
		auto vertices = std::vector<Vertex>(EzeFrance->mMeshes[i]->mNumVertices);
		for (size_t vi = 0; vi != EzeFrance->mMeshes[i]->mNumVertices; ++vi) {
			vertices[vi].position  = calculation::vector3_cast<std::array<float, 4>, 0,1,2>( EzeFrance->mMeshes[i]->mVertices[vi] );
			vertices[vi].texcoord  = calculation::vector3_cast<std::array<float, 4>, 0,1,2>( EzeFrance->mMeshes[i]->mTextureCoords[0][vi] );
			vertices[vi].normal    = calculation::vector3_cast<std::array<float, 4>, 0,1,2>( EzeFrance->mMeshes[i]->mNormals[vi] );
			vertices[vi].tangent   = calculation::vector3_cast<std::array<float, 4>, 0,1,2>( EzeFrance->mMeshes[i]->mTangents[vi] );
			vertices[vi].bitangent = calculation::vector3_cast<std::array<float, 4>, 0,1,2>( EzeFrance->mMeshes[i]->mBitangents[vi] );
		}

		auto indices = std::vector<uint32_t>(EzeFrance->mMeshes[i]->mNumFaces * 3);
		for (size_t fi = 0; fi != EzeFrance->mMeshes[i]->mNumFaces; ++fi) {
			assert(EzeFrance->mMeshes[i]->mFaces[fi].mNumIndices == 3);
			indices[fi * 3 + 0] = EzeFrance->mMeshes[i]->mFaces[fi].mIndices[0];
			indices[fi * 3 + 1] = EzeFrance->mMeshes[i]->mFaces[fi].mIndices[1];
			indices[fi * 3 + 2] = EzeFrance->mMeshes[i]->mFaces[fi].mIndices[2];
		}

		vbuffer[std::string("EzeFrance") + EzeFrance->mMeshes[i]->mName.C_Str() + "Vertex"] = GLvaryings( GLbufferx<Vertex>(GL_ARRAY_BUFFER, vertices.data(), vertices.size()) );
		buffer[std::string("EzeFrance") + EzeFrance->mMeshes[i]->mName.C_Str() + "Index"]  = GLbufferx<uint32_t>(GL_ELEMENT_ARRAY_BUFFER, indices.data(), indices.size());
	}

	GLuint tex = -1;
	glLoadTexture2D("asset_model/EzeFrance/Chunk0_Color.png", -1, -1, &tex);
	texture[EzeFranceChunk0Texture] = GLtexture(GL_TEXTURE_2D, tex); tex = -1;
	glLoadTexture2D("asset_model/EzeFrance/Chunk1_Color.png", -1, -1, &tex);
	texture[EzeFranceChunk1Texture] = GLtexture(GL_TEXTURE_2D, tex); tex = -1;
	glLoadTexture2D("asset_model/EzeFrance/Chunk2_Color.png", -1, -1, &tex);
	texture[EzeFranceChunk2Texture] = GLtexture(GL_TEXTURE_2D, tex); tex = -1;

	assert( glIsVaryings(vbuffer[EzeFranceChunk0Vertex]) );
	assert( glIsVaryings(vbuffer[EzeFranceChunk1Vertex]) );
	assert( glIsVaryings(vbuffer[EzeFranceChunk2Vertex]) );
	assert( glIsBuffer(buffer[EzeFranceChunk0Index]) );
	assert( glIsBuffer(buffer[EzeFranceChunk1Index]) );
	assert( glIsBuffer(buffer[EzeFranceChunk2Index]) );
	assert( glIsTexture(texture[EzeFranceChunk0Texture]) );
	assert( glIsTexture(texture[EzeFranceChunk1Texture]) );
	assert( glIsTexture(texture[EzeFranceChunk2Texture]) );

	/////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////

	processor[SceneShadowProcessor] = GLprogram(
		GLvertshader(GLshadersource(
			std::filesystem::path("asset_shader/scene/scene_depth.vert"))),
		GLfragshader(GLshadersource(
			std::filesystem::path("asset_shader/scene/scene_depth.frag"))));
	texture[SceneShadowTexture]
		= GLtexture2D(GL_DEPTH_COMPONENT32F, SceneShadowResolution[0], SceneShadowResolution[1]);
	glBindTexture(GL_TEXTURE_2D, texture[SceneShadowTexture]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void volumeInit() {
	processor[VolumeProcessor] = GLprogram(
		GLvertshader(GLshadersource(std::filesystem::path("asset_shader/volume/volume.vert"))), 
		GLfragshader(GLshadersource(std::filesystem::path("asset_shader/volume/volume.frag.h"))));
	
	texture[VolumeRadianceAndTransmittanceTexture]
		= GLtexture2D(GL_RGBA32F, VolumeTextureResolution[0], VolumeTextureResolution[1]);
	texture[VolumeDepthTexture]
		= GLtexture2D(GL_DEPTH_COMPONENT32F, VolumeTextureResolution[0], VolumeTextureResolution[1]);
	glBindTexture(GL_TEXTURE_2D, texture[VolumeDepthTexture]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);


	GLuint tex = -1;
	
	glLoadTexture3D("asset_model/cloud/base_cloud.png", 128, 128, 128, &tex);
	texture[BaseCloudTexture] = GLtexture3D(tex); tex = -1;
	
	glLoadTexture3D("asset_model/cloud/cloud_detail.png", 32, 32, 32, &tex);
	texture[CloudDetailTexture] = GLtexture3D(tex); tex = -1;
	
	glLoadTexture2D("asset_model/cloud/cloud_weather5.png", -1, -1, &tex);
	texture[CloudWeatherTextureFirst] = GLtexture2D(tex); tex = -1;

	glLoadTexture2D("asset_model/cloud/cloud_weather3.png", -1, -1, &tex);
	texture[CloudWeatherTextureSecond] = GLtexture2D(tex); tex = -1;
	
	glLoadTexture2D("asset_model/cloud/cloud_height.png", -1, -1, &tex);
	texture[CloudHeightTexture] = GLtexture2D(tex); tex = -1;
	
	glLoadTexture2D("asset_model/cloud/cirrus_weather.png", -1, -1, &tex);
	texture[CirrusWeatherTexture] = GLtexture2D(tex); tex = -1;
	
	glLoadTexture2D("asset_model/cloud/cloud_curl.png", -1, -1, &tex);
	texture[CloudCurlTexture] = GLtexture2D(tex); tex = -1;


	processor[VolumeShadowProcessor] = GLprogram(
		GLvertshader(GLshadersource(
			std::filesystem::path("asset_shader/volume/volume.vert"))),
		GLfragshader(GLshadersource(
			std::filesystem::path("asset_shader/volume/shadow.frag.h"))));
	texture[VolumeShadowTexture]
		= GLtexture2D(GL_DEPTH_COMPONENT32F, VolumeShadowResolution[0], VolumeShadowResolution[1]);
	glBindTexture(GL_TEXTURE_2D, texture[VolumeShadowTexture]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	texture[VolumeTransmittanceTexture]
		= GLtexture2D(GL_R32F, VolumeShadowResolution[0], VolumeShadowResolution[1]);
}

void atmosphereInit() {
	processor[FinalProcessor] = GLprogram(
		GLvertshader(GLshadersource(
			std::filesystem::path("asset_shader/final.vert"))),
		GLfragshader(GLshadersource(
			std::filesystem::path("asset_shader/final.frag.h"))));

	skymodel = make_default_EricBrunetonSkyModel();
}

void renderSceneShadow() {
	auto Identity = matrix4x4<float>(); Identity.set_identity();
	auto sun_vector       = - sun.f_vector();
	auto world_to_sunproj = sun.projection_matrix<matrix4x4<float>>() * sun.view_matrix<matrix4x4<float>>();

	GLuint framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture[SceneShadowTexture], 0);
	glViewport(0, 0, SceneShadowResolution[0], SceneShadowResolution[1]);
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	{
		GLint object_to_proj_location = glGetUniformLocation(processor[SceneShadowProcessor], "object_to_proj");
		glUseProgram(processor[SceneShadowProcessor]);

		auto object_to_proj = world_to_sunproj * Identity;
		glUniformMatrix4fv(object_to_proj_location, 1, true, object_to_proj.data());
		glBindVertexArray(vbuffer[EzeFranceChunk0Vertex]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[EzeFranceChunk0Index]);
		glDrawElements(GL_TRIANGLES, buffer[EzeFranceChunk0Index].bytesize() / sizeof(uint32_t), GL_UNSIGNED_INT, nullptr);

		glUniformMatrix4fv(object_to_proj_location, 1, true, object_to_proj.data());
		glBindVertexArray(vbuffer[EzeFranceChunk1Vertex]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[EzeFranceChunk1Index]);
		glDrawElements(GL_TRIANGLES, buffer[EzeFranceChunk1Index].bytesize() / sizeof(uint32_t), GL_UNSIGNED_INT, nullptr);

		glUniformMatrix4fv(object_to_proj_location, 1, true, object_to_proj.data());
		glBindVertexArray(vbuffer[EzeFranceChunk2Vertex]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[EzeFranceChunk2Index]);
		glDrawElements(GL_TRIANGLES, buffer[EzeFranceChunk2Index].bytesize() / sizeof(uint32_t), GL_UNSIGNED_INT, nullptr);
	}

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
	glDeleteFramebuffers(1, &framebuffer);
	assert(glGetError() == GL_NO_ERROR);
}

void renderVolumeShadow() {
	auto world_to_sunproj = sun.projection_matrix<matrix4x4<double>>() * sun.view_matrix<matrix4x4<double>>();
	//std::cout << to_string(world_to_sunproj) << std::endl;
	matrix4x4<double> sunproj_to_world = inverse(world_to_sunproj);
	//std::cout << to_string(sunproj_to_world) << std::endl;
	matrix4x4<float> sunproj_to_world_32f;
	matrix4x4<float> world_to_proj_32f;
	for (size_t i = 0; i != sunproj_to_world_32f.rows(); ++i) {
		for (size_t j = 0; j < sunproj_to_world_32f.cols(); ++j) {
			sunproj_to_world_32f.at(i, j) = static_cast<float>(sunproj_to_world.at(i, j));
			world_to_proj_32f.at(i, j) = static_cast<float>(world_to_sunproj.at(i, j));
		}
	}
	
	vector3<float>   sun_vector       = - sun.f_vector();



	GLuint framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[VolumeTransmittanceTexture],    0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture[VolumeShadowTexture],    0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	glViewport(0, 0, VolumeShadowResolution[0], VolumeShadowResolution[1]);
	glClearColor(1.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glEnable( GL_DEPTH_TEST);

	{
		glUseProgram(processor[VolumeShadowProcessor]);
		
		glUniform2iv(glGetUniformLocation(processor[VolumeShadowProcessor], "resolution"),
					 1, VolumeTextureResolution);
		glUniform1f(glGetUniformLocation(processor[VolumeShadowProcessor], "time"),
					this_time.count() / float(1000.0f));
		glUniform1f(glGetUniformLocation(processor[VolumeShadowProcessor], "time_per_day"),
					100.0f);
		//std::cout << this_time.count() / float(1000.0f) << std::endl;

		glUniformTexture(glGetUniformLocation(processor[VolumeShadowProcessor], "base_cloud_texture"),
					     GL_TEXTURE0, GL_TEXTURE_3D, texture[BaseCloudTexture]);
		glUniformTexture(glGetUniformLocation(processor[VolumeShadowProcessor], "cloud_detail_texture"),
						 GL_TEXTURE1, GL_TEXTURE_3D, texture[CloudDetailTexture]);
		glUniformTexture(glGetUniformLocation(processor[VolumeShadowProcessor], "cloud_weather_texture_first"),
						 GL_TEXTURE2, GL_TEXTURE_2D, texture[CloudWeatherTextureFirst]);
		glUniformTexture(glGetUniformLocation(processor[VolumeShadowProcessor], "cloud_weather_texture_second"),
						 GL_TEXTURE3, GL_TEXTURE_2D, texture[CloudWeatherTextureSecond]);
		glUniformTexture(glGetUniformLocation(processor[VolumeShadowProcessor], "cloud_height_texture"),
						 GL_TEXTURE4, GL_TEXTURE_2D, texture[CloudHeightTexture]);
		glUniformTexture(glGetUniformLocation(processor[VolumeShadowProcessor], "cirrus_weather_texture"),
						 GL_TEXTURE5, GL_TEXTURE_2D, texture[CirrusWeatherTexture]);
		glUniformTexture(glGetUniformLocation(processor[VolumeShadowProcessor], "cloud_curl_texture"),
						 GL_TEXTURE6, GL_TEXTURE_2D, texture[CloudCurlTexture]);

		glUniform3fv(glGetUniformLocation(processor[VolumeShadowProcessor], "sun_vector"),
					 1, sun_vector.data());

		glUniformTexture(glGetUniformLocation(processor[VolumeShadowProcessor], "scene_shadow_texture"),
						 GL_TEXTURE7, GL_TEXTURE_2D, texture[SceneShadowTexture]);
		glUniformMatrix4fv(glGetUniformLocation(processor[VolumeShadowProcessor], "sunproj_to_world"),
						   1, true, sunproj_to_world_32f.data());
		glUniformMatrix4fv(glGetUniformLocation(processor[VolumeShadowProcessor], "world_to_proj"),
						   1, true, world_to_proj_32f.data());

		glUniform1f(glGetUniformLocation(processor[VolumeShadowProcessor], "CLOUDSPHERE0.density_scale"), cloud_density_scale);
		glUniform1f(glGetUniformLocation(processor[VolumeShadowProcessor], "base_cloud_texture_scale"), base_cloud_texture_scale);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
	glDeleteFramebuffers(1, &framebuffer);
	assert(glGetError() == GL_NO_ERROR);
}

void renderScene() {
	auto Identity = matrix4x4<float>(); Identity.set_identity();
	auto world_to_camera = camera.view_matrix<matrix4x4<float>>();
	auto camera_to_proj  = camera.projection_matrix<matrix4x4<float>>();

	auto sun_vector       = - sun.f_vector();
	auto world_to_sunview = sun.view_matrix<matrix4x4<float>>();
	auto sunview_to_sunproj = sun.projection_matrix<matrix4x4<float>>();
	auto world_to_sunproj     = sunview_to_sunproj * world_to_sunview;

	/// /////////////////////////////////////////////////////////////////////////
	/// /////////////////////////////////////////////////////////////////////////
	/// /////////////////////////////////////////////////////////////////////////

	GLuint framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[SceneRadianceTexture], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, texture[SceneDepthTexture],    0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	glViewport(0, 0, SceneTextureResolution[0], SceneTextureResolution[1]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	{
		glUseProgram(processor[SceneProcessor]);
	
		glUniformMatrix4fv(glGetUniformLocation(processor[SceneProcessor], "object_to_world"),
							1, true, Identity.data());
		glUniformMatrix4fv(glGetUniformLocation(processor[SceneProcessor], "world_to_camera"),
							1, true, world_to_camera.data());
		glUniformMatrix4fv(glGetUniformLocation(processor[SceneProcessor], "camera_to_proj"),
							1, true, camera_to_proj.data());
			
		glUniform3fv(glGetUniformLocation(processor[SceneProcessor], "sun_vector"),
						1, sun_vector.data());
		glUniform3fv(glGetUniformLocation(processor[SceneProcessor], "camera"),
						1, camera.position().data());

		glUniformTexture(glGetUniformLocation(processor[SceneProcessor], "transmittance_texture"),
						    GL_TEXTURE0, GL_TEXTURE_2D, skymodel->GetTransmittanceTexture());
		glUniformTexture(glGetUniformLocation(processor[SceneProcessor], "irradiance_texture"),
							GL_TEXTURE1, GL_TEXTURE_2D, skymodel->GetIrradianceTexture());

		glUniformTexture(glGetUniformLocation(processor[SceneProcessor], "scene_shadow_texture"),  
							GL_TEXTURE2, GL_TEXTURE_2D, texture[SceneShadowTexture]);
		glUniformTexture(glGetUniformLocation(processor[SceneProcessor], "volume_transmittance_texture"),  
							GL_TEXTURE3, GL_TEXTURE_2D, texture[VolumeTransmittanceTexture]);
		glUniformMatrix4fv(glGetUniformLocation(processor[SceneProcessor], "world_to_sunproj"),
							1, true, world_to_sunproj.data());
		glUniformMatrix4fv(glGetUniformLocation(processor[SceneProcessor], "world_to_sunview"),
							1, true, world_to_sunview.data());
		glUniformMatrix4fv(glGetUniformLocation(processor[SceneProcessor], "sunview_to_sunproj"),
							1, true, sunview_to_sunproj.data());
		{
			glUniformTexture(glGetUniformLocation(processor[SceneProcessor], "albedo_texture"),
								GL_TEXTURE4, GL_TEXTURE_2D, texture[EzeFranceChunk0Texture]);
			glBindVertexArray(vbuffer[EzeFranceChunk0Vertex]);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[EzeFranceChunk0Index]);
			glDrawElements(GL_TRIANGLES, buffer[EzeFranceChunk0Index].bytesize() / sizeof(uint32_t), GL_UNSIGNED_INT, nullptr);

			glUniformTexture(glGetUniformLocation(processor[SceneProcessor], "albedo_texture"),
								GL_TEXTURE4, GL_TEXTURE_2D, texture[EzeFranceChunk1Texture]);
			glBindVertexArray(vbuffer[EzeFranceChunk1Vertex]);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[EzeFranceChunk1Index]);
			glDrawElements(GL_TRIANGLES, buffer[EzeFranceChunk1Index].bytesize() / sizeof(uint32_t), GL_UNSIGNED_INT, nullptr);

			glUniformTexture(glGetUniformLocation(processor[SceneProcessor], "albedo_texture"),
								GL_TEXTURE4, GL_TEXTURE_2D, texture[EzeFranceChunk2Texture]);
			glBindVertexArray(vbuffer[EzeFranceChunk2Vertex]);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[EzeFranceChunk2Index]);
			glDrawElements(GL_TRIANGLES, buffer[EzeFranceChunk2Index].bytesize() / sizeof(uint32_t), GL_UNSIGNED_INT, nullptr);
		}
	}
	
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
	glDeleteFramebuffers(1, &framebuffer);
	assert( glGetError() == GL_NO_ERROR );
}

void renderVolume() {
	auto world_to_camera = camera.view_matrix<matrix4x4<float>>();
	auto camera_to_proj  = camera.projection_matrix<matrix4x4<float>>();
	auto world_to_proj   = camera_to_proj * world_to_camera;
	auto proj_to_world   = inverse(world_to_proj);
	auto sun_vector      = -sun.f_vector();

	GLuint framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[VolumeRadianceAndTransmittanceTexture], 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, texture[VolumeDepthTexture], 0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	glViewport(0, 0, VolumeTextureResolution[0], VolumeTextureResolution[1]);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glEnable(GL_DEPTH_TEST);

	{
		glUseProgram(processor[VolumeProcessor]);

		glUniform2iv(glGetUniformLocation(processor[VolumeProcessor], "resolution"),
					 1, VolumeTextureResolution);
		glUniform1f(glGetUniformLocation(processor[VolumeProcessor], "time"),
					this_time.count() / float(1000.0f));
		glUniform1f(glGetUniformLocation(processor[VolumeProcessor], "time_per_day"),
					100.0f);
		//std::cout << this_time.count() / float(1000.0f) << std::endl;

		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "base_cloud_texture"),
					     GL_TEXTURE0, GL_TEXTURE_3D, texture[BaseCloudTexture]);
		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "cloud_detail_texture"),
						 GL_TEXTURE1, GL_TEXTURE_3D, texture[CloudDetailTexture]);
		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "cloud_weather_texture_first"),
						 GL_TEXTURE2, GL_TEXTURE_2D, texture[CloudWeatherTextureFirst]);
		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "cloud_weather_texture_second"),
						 GL_TEXTURE3, GL_TEXTURE_2D, texture[CloudWeatherTextureSecond]);
		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "cloud_height_texture"),
						 GL_TEXTURE4, GL_TEXTURE_2D, texture[CloudHeightTexture]);
		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "cirrus_weather_texture"),
						 GL_TEXTURE5, GL_TEXTURE_2D, texture[CirrusWeatherTexture]);
		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "cloud_curl_texture"),
						 GL_TEXTURE6, GL_TEXTURE_2D, texture[CloudCurlTexture]);

		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "transmittance_texture"),
						 GL_TEXTURE7, GL_TEXTURE_2D, skymodel->GetTransmittanceTexture());
		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "irradiance_texture"),
						 GL_TEXTURE8, GL_TEXTURE_2D, skymodel->GetIrradianceTexture());
		glUniform3fv(glGetUniformLocation(processor[VolumeProcessor], "sun_vector"),
						1, sun_vector.data());

		glUniformTexture(glGetUniformLocation(processor[VolumeProcessor], "scene_depth_texture"),
						 GL_TEXTURE9, GL_TEXTURE_2D, texture[SceneDepthTexture]);
		glUniformMatrix4fv(glGetUniformLocation(processor[VolumeProcessor], "proj_to_world"),
						   1, true, proj_to_world.data());
		glUniformMatrix4fv(glGetUniformLocation(processor[VolumeProcessor], "world_to_proj"),
						   1, true, world_to_proj.data());
		
		glUniform1f(glGetUniformLocation(processor[VolumeProcessor], "CLOUDSPHERE0.density_scale"), cloud_density_scale);
		glUniform1f(glGetUniformLocation(processor[VolumeProcessor], "base_cloud_texture_scale"), base_cloud_texture_scale);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
	glDeleteFramebuffers(1, &framebuffer);
	assert( glGetError() == GL_NO_ERROR );
}

void renderAtmosphere() {
	auto world_to_camera = camera.view_matrix<matrix4x4<float>>();
	auto camera_to_proj  = camera.projection_matrix<matrix4x4<float>>();
	auto world_to_proj   = camera_to_proj * world_to_camera;
	auto proj_to_world   = inverse(world_to_proj);

	auto sun_vector       = - sun.f_vector();
	auto world_to_sunproj = sun.projection_matrix<matrix4x4<float>>() * sun.view_matrix<matrix4x4<float>>();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, ScreenResolution[0], ScreenResolution[1]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	{
		glUseProgram(processor[FinalProcessor]);
		
		glUniform2iv(glGetUniformLocation(processor[FinalProcessor], "resolution"),
					 1, ScreenResolution);

		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "transmittance_texture"),
						 GL_TEXTURE0, GL_TEXTURE_2D, skymodel->GetTransmittanceTexture());
		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "scattering_texture"),
						 GL_TEXTURE1, GL_TEXTURE_3D, skymodel->GetScatteringTexture());
		glUniform3fv(glGetUniformLocation(processor[FinalProcessor], "sun_vector"),
					 1, sun_vector.data());

		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "scene_radiance_texture"),
						 GL_TEXTURE2, GL_TEXTURE_2D, texture[SceneRadianceTexture]);
		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "scene_depth_texture"),
						 GL_TEXTURE3, GL_TEXTURE_2D, texture[SceneDepthTexture]);
		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "volume_radiance_and_transmittance_texture"),
						 GL_TEXTURE4, GL_TEXTURE_2D, texture[VolumeRadianceAndTransmittanceTexture]);
		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "volume_depth_texture"),
						 GL_TEXTURE5, GL_TEXTURE_2D, texture[VolumeDepthTexture]);
		glUniformMatrix4fv(glGetUniformLocation(processor[FinalProcessor], "proj_to_world"),
						   1, true, proj_to_world.data());

		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "scene_shadow_texture"),
						 GL_TEXTURE6, GL_TEXTURE_2D, texture[SceneShadowTexture]);
		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "volume_shadow_texture"),
						 GL_TEXTURE7, GL_TEXTURE_2D, texture[VolumeShadowTexture]);
		glUniformTexture(glGetUniformLocation(processor[FinalProcessor], "volume_shadow_transmittance_texture"),
						 GL_TEXTURE8, GL_TEXTURE_2D, texture[VolumeShadowTexture]);
		glUniformMatrix4fv(glGetUniformLocation(processor[FinalProcessor], "world_to_sunproj"),
						   1, true, world_to_sunproj.data());
		
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}
	
	assert(glGetError() == GL_NO_ERROR);
}

void cursorposfunc(GLFWwindow* Win, double x, double y) {
	static double prev_x = x;
	static double prev_y = y;

	if (glfwGetMouseButton(Win, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_RELEASE) {
		float dx = static_cast<float>(x - prev_x);
		float dy = static_cast<float>(y - prev_y);
		camera.pitch(3.14F / 180.F * dy * 0.5F);
		camera.yaw(3.14F / 180.F * dx);
		_Update_sun_projection();
	}else if (glfwGetMouseButton(Win, GLFW_MOUSE_BUTTON_LEFT) != GLFW_RELEASE) {
		float dx = static_cast<float>(x - prev_x);
		float dy = static_cast<float>(y - prev_y);
		sun.yaw(3.14F/180.F * dx);
		sun.pitch(-3.14F/180.F * dy * 0.5F);
		_Update_sun_projection();
	}

	prev_x = x;
	prev_y = y;
}

void keyfunc(GLFWwindow* Win, int key, int scancode, int action, int mods) {
	if (action != GLFW_RELEASE) {
		float speed = 100.0f;
		switch (key) {
		case GLFW_KEY_W: camera.moveZ(+speed); break;
		case GLFW_KEY_S: camera.moveZ(-speed); break;
		case GLFW_KEY_A: camera.moveX(-speed); break;
		case GLFW_KEY_D: camera.moveX(+speed); break;
		case GLFW_KEY_Q: camera._Myori += calculation::vector3<float>{0.0f, 1.0f, 0.0f} *speed; break;
		case GLFW_KEY_E: camera._Myori += calculation::vector3<float>{0.0f, -1.0f, 0.0f} *speed; break;
		
		case GLFW_KEY_O: 
			cloud_density_scale *= 2.0f; break;
		case GLFW_KEY_L: 
			cloud_density_scale *= 0.5f; break;
		case GLFW_KEY_U: 
			base_cloud_texture_scale *= 2.0f; break;
		case GLFW_KEY_J:
			base_cloud_texture_scale *= 0.5f; break;
		case GLFW_KEY_I: 
			cloud_detail_texture_scale *= 2.0f; break;
		case GLFW_KEY_K:
			cloud_detail_texture_scale *= 0.5f; break;
		
		case GLFW_KEY_V: 
			if (video_writer.isOpened()) {
				video_writer.release();
			} else {
				video_writer.open("shadowed_atmospherescattering.mp4", cv::CAP_OPENCV_MJPEG, 30, cv::Size(ScreenResolution[0], ScreenResolution[1]), true);
			}
			break;
		default: break;
		}
	}
}


int main(int argc, char** argv) {
	vector3<double> box_min = { -1, -1, -1 };
	vector3<double> box_max = { +1, +1, +1 };
	vector3<double> ray_origin = { 0, 0, 0 };
	vector3<double> ray_direction = { 1, 0, 0 };
	//
	//if (ray_intersect_axisaligned_box(ray_origin, ray_direction, box_min, box_max) >= 0) {
	//	std::cout << ray_intersect_axisaligned_box(ray_origin, ray_direction, box_min, box_max) << std::endl;
	//}

	//ray_origin = { 1, 1, 1 };
	//if (ray_intersect_axisaligned_box(ray_origin, ray_direction, box_min, box_max) >= 0) {
	//	std::cout << ray_intersect_axisaligned_box(ray_origin, ray_direction, box_min, box_max) << std::endl;
	//}

	//ray_origin = { -1.2, 1, 1 };
	//if (ray_intersect_axisaligned_box(ray_origin, ray_direction, box_min, box_max) >= 0) {
	//	std::cout << ray_intersect_axisaligned_box(ray_origin, ray_direction, box_min, box_max) << std::endl;
	//}


	//exit(0);

	//cv::Mat red = cv::imread("asset_model/cloud/red.png", cv::IMREAD_GRAYSCALE);
	//cv::Mat green = cv::imread("asset_model/cloud/green.png", cv::IMREAD_GRAYSCALE);
	//cv::Mat blue = cv::imread("asset_model/cloud/blue.png", cv::IMREAD_GRAYSCALE);
	//cv::Mat src = cv::Mat(red.rows, red.cols, CV_8UC3);
	//for (size_t i = 0; i != src.rows; ++i) {
	//	for (size_t j = 0; j != src.cols; ++j) {
	//		auto& r = red.at<unsigned char>(i, j);
	//		auto& g = green.at<unsigned char>(i, j);
	//		auto& b = blue.at<unsigned char>(i, j);
	//		
	//		src.at<cv::Vec3b>(i, j) = { b, g, r };
	//	}
	//}
	//cv::imwrite("asset_model/cloud/temp.png", src);
	//exit(0);

	glfwInit();
	GLFWwindow* window = 
	glfwCreateWindow(ScreenResolution[0], ScreenResolution[1], "dreamworld1", nullptr, nullptr);
	glfwMakeContextCurrent(window);
	glewInit();
	sceneInit();
	volumeInit();
	atmosphereInit();

	glfwSetCursorPosCallback(window, cursorposfunc);
	glfwSetKeyCallback(window, keyfunc);

	auto start_time = std::chrono::steady_clock::now();
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		renderSceneShadow();
		renderVolumeShadow();
		renderScene();
		renderVolume();
		renderAtmosphere();
		glfwSwapBuffers(window);
		
		if (video_writer.isOpened()) {
			cv::Mat snap = cv::Mat(ScreenResolution[1], ScreenResolution[0], CV_8UC3);
			glReadPixels(0, 0, ScreenResolution[0], ScreenResolution[1], GL_RGB, GL_UNSIGNED_BYTE, snap.data);
			cv::cvtColor(snap, snap, cv::COLOR_BGR2RGB);
			cv::flip(snap, snap, 0);

			video_writer.write(snap);
		}

		auto current_time = std::chrono::steady_clock::now();
		this_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
	}

	glfwTerminate();
	glfwDestroyWindow(window);
	return 0;
}