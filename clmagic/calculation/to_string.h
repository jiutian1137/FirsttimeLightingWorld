#pragma once
#include "lapack/vector.h"
#include "lapack/matrix.h"

namespace calculation {
	template<typename T, size_t N, typename Tr> inline
	std::string to_string(const vector<T,N,Tr>& vect) {
		using std::to_string;

		std::string result = "{";
		for (size_t i = 0; i != N; ++i) {
			result += to_string(vect[i]); result += ',';
		}
		result.back() = '}';

		return std::move(result);
	}

	template<typename T, size_t M, size_t N>
	std::string to_string(const matrix<T,M,N>& matx) {
		const T* _Ptr = matx.data();

		std::string _Result = "{";
		for (size_t i = 0; i != matx.size(); ++i, ++_Ptr) {
			_Result += std::to_string(*_Ptr);
			if ( (i + 1) % matx.cols() == 0 ) {
				_Result += '\n';
			} else {
				_Result += ',';
			}
		}
		_Result.back() = '}';

		return std::move(_Result);
	}
}