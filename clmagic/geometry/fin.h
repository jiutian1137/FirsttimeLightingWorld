#pragma once
#include "geometry.h"
#include <memory>
#include <string>
#include <assimp/mesh.h>
#include <assimp/material.h>

//_CLMAGIC_BEGIN
namespace geometry {
	template<typename _InIt1, typename _InIt2, typename _OutIt, typename _Fn>
	_OutIt composite(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _Fn _Func) {
		for ( ; _First1 != _Last1; ++_First1, ++_First2, ++_Dest) {
			_Func(*_First1, *_First2, *_Dest);
		}

		return _Dest;
	}

	template<typename _InIt1, typename _InIt2, typename _InIt3, typename _OutIt, typename _Fn>
	_OutIt composite(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _InIt3 _First3, _OutIt _Dest, _Fn _Func) {
		for (; _First1 != _Last1; ++_First1, ++_First2, ++_First3, ++_Dest) {
			_Func(*_First1, *_First2, *_First3, *_Dest);
		}

		return _Dest;
	}

	template<typename _InIt, typename _OutIt1, typename _OutIt2, typename _Fn>
	_InIt decomposite(_InIt _First, _InIt _Last, _OutIt1 _Dest1, _OutIt2 _Dest2, _Fn _Func) {
		for (; _First != _Last; ++_First, ++_Dest1, ++_Dest2) {
			_Func(*_First, *_Dest1, *_Dest2);
		}

		return _First;
	}

	template<typename _InIt, typename _OutIt1, typename _OutIt2, typename _OutIt3, typename _Fn>
	_InIt decomposite(_InIt _First, _InIt _Last, _OutIt1 _Dest1, _OutIt2 _Dest2, _OutIt3 _Dest3, _Fn _Func) {
		for (; _First != _Last; ++_First, ++_Dest1, ++_Dest2, ++_Dest3) {
			_Func(*_First, *_Dest1, *_Dest2, *_Dest3);
		}

		return _First;
	}


	template<typename _Ty1, typename _Ty2 = uint32_t>
		requires requires(_Ty1 __v) { __v.position; __v.normal; __v.texcoord; __v.descriptor(); }
	opengl_index_geometry<_Ty1, _Ty2> make_opengl_index_geometry(const aiMesh& _Aimesh, const aiMaterial& _Aimaterial, std::shared_ptr<GLprogram> _Processor) {
		// 1. Load data
		std::vector<_Ty1> _Vdata = std::vector<_Ty1>(_Aimesh.mNumVertices);
		assert(_Aimesh.mVertices != nullptr);
		for (size_t i = 0; i != _Aimesh.mNumVertices; ++i) {
			const aiVector3D& v  = _Aimesh.mVertices[i];
			_Vdata[i].position = {  v[0],  v[1],  v[2] };
		}
		if (_Aimesh.mNormals != nullptr) {
			for (size_t i = 0; i != _Aimesh.mNumVertices; ++i) {
				const aiVector3D& vn = _Aimesh.mNormals[i];
				_Vdata[i].normal = { vn[0], vn[1], vn[2] };
			}
		}
		if (_Aimesh.mTextureCoords != nullptr && _Aimesh.mTextureCoords[0] != nullptr) {
			for (size_t i = 0; i != _Aimesh.mNumVertices; ++i) {
				const aiVector3D& vt = _Aimesh.mTextureCoords[0][i];
				_Vdata[i].texcoord = { vt[0], vt[1], vt[2] };
			}
		}

		assert(_Aimesh.mPrimitiveTypes != aiPrimitiveType::aiPrimitiveType_POLYGON);
		std::vector<_Ty2> _Idata = std::vector<_Ty2>(_Aimesh.mNumFaces * 3);
		for (size_t i = 0; i != _Aimesh.mNumFaces; ++i) {
			const aiFace& f = _Aimesh.mFaces[i];
			std::copy(f.mIndices, f.mIndices + 3, &_Idata[i * 3]);
		}

		aiColor3D diffuse;
		aiColor3D specular;
		_Aimaterial.Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
		_Aimaterial.Get(AI_MATKEY_COLOR_SPECULAR, specular);

		// 2. Create
		opengl_index_geometry<_Ty1, _Ty2> _Result;
		// processor
		_Result.processor = _Processor;
		// uniforms
		_Result.albedo_location    = glGetUniformLocation(*_Processor, "albedo");
		_Result.fresnelF0_location = glGetUniformLocation(*_Processor, "fresnelF0");
		_Result.local_to_world_location = glGetUniformLocation(*_Processor, "local_to_world");
		_Result.albedo    = { diffuse[0], diffuse[1], diffuse[2] };
		_Result.fresnelF0 = { specular[0], specular[1], specular[2] };
		_Result.local_to_world.set_identity();
		// varyings
		_Result.vertex_data = std::make_shared<GLbufferx<_Ty1>>(GL_ARRAY_BUFFER, _Vdata.data(), _Vdata.size());
		auto temp = _Ty1().descriptor();
		_Result.vertex_desc = std::make_shared<GLvaryings>(*(_Result.vertex_data), temp.data(), temp.size());
		_Result.index_data  = std::make_shared<GLbufferx<_Ty2>>(GL_ELEMENT_ARRAY_BUFFER, _Idata.data(), _Idata.size());
		// process
		_Result.mode = GL_TRIANGLES;

		return std::move(_Result);
	}

}// namespace geometry
//_CLMAGIC_END