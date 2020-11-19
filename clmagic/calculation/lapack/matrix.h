//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache License 2.0
//--------------------------------------------------------------------------------------
#pragma once
#ifndef clmagic_calculation_lapack_MATRIX_h_
#define clmagic_calculation_lapack_MATRIX_h_
#include "vector.h"
#include <iostream>
#include <string>

namespace calculation {
	/*<undetermined>
	template<typename _SclTy> inline
	void minor(matrix_slice<const _SclTy> _Source, matrix_slice<_SclTy> _Dest, size_t i, size_t j) {
		size_t _Rseek = 0, _Cseek;
		for (size_t _Rfirst = 0; _Rfirst < _Dest.rows(); ++_Rfirst) {
			if (_Rfirst != i) {
				_Cseek = 0;
				for (size_t _Cfirst = 0; _Cfirst < _Dest.cols(); ++_Cfirst) {
					if (_Cfirst != j) {
						_Dest.at(_Rseek, _Cseek) = _Source.at(_Rfirst, _Cfirst);
						++_Cseek;
					}
				}
				++_Rseek;
			}
		}
	}
	</undetermined>*/

	struct general_matrix_tag {};
	struct diagonal_matrix_tag {};
	struct orthogonal_matrix_tag {};


	template<typename _MatTy, typename = void>
	struct matrix_traits {
		using scalar_type     = typename _MatTy::value_type;
		using block_type      = scalar_type;
		using matrix_category = general_matrix_tag;
	};

	template<typename _MatTy>
	struct matrix_traits<_MatTy, std::void_t<typename _MatTy::block_type>> {
		using scalar_type     = typename _MatTy::scalar_type;
		using block_type      = typename _MatTy::block_type;
		using matrix_category = typename _MatTy::matrix_category;
	};

	template<typename _Ty>
	struct matrix_traits<_Ty*> {
		using scalar_type     = _Ty;
		using block_type      = _Ty;
		using matrix_category = general_matrix_tag;
	};
	
	template<typename _MatTy>
	using matrix_scalar_t = typename matrix_traits<_MatTy>::scalar_type;

	template<typename _MatTy>
	using matrix_block_t = typename matrix_traits<_MatTy>::block_type;

	template<typename _MatTy>
	using matrix_major_category_t = typename matrix_traits<_MatTy>::major_category;

	template<typename _MatTy>
	using matrix_matrix_category_t = typename matrix_traits<_MatTy>::matrix_category;

	template<typename _MatTy>
	concept row_optimized_matrix = requires(_MatTy __m) { 
		__m.row(0); // full row
		__m.row(0) * __m.at(0, 0);
		__m.row(0, 0); // i_row[j, ...)
		__m.row(0, 0) * __m.at(0, 0);
		__m.row(0, 0, 1); // i_row[j, end)
		__m.row(0, 0, 1) * __m.at(0, 0); 
	};

	template<typename _MatTy>
		requires requires(_MatTy __m) { __m.data(); __m.cols(); }
	void matrix_row_swap(_MatTy& matrix, size_t Arow, size_t Brow, size_t offset = 0) {
		auto*       row1     = matrix.data() + Arow * matrix.cols() + offset;
		const auto* row1_end = matrix.data() + (Arow+1) * matrix.cols();
		auto*       row2     = matrix.data() + Brow * matrix.cols() + offset;
		for ( ; row1 != row1_end; ++row1, ++row2) {
			std::swap(*row1, *row2);
		}
	}

	template<typename _MatTy>
		requires requires(_MatTy __m) { __m.data(); __m.cols(); }
	void matrix_row_swap(_MatTy& matrix, size_t Arow, size_t Brow, size_t row_first, size_t row_last) {
		auto*       row1     = matrix.data() + Arow * matrix.cols() + row_first;
		const auto* row1_end = matrix.data() + Arow * matrix.cols() + row_last;
		auto*       row2     = matrix.data() + Brow * matrix.cols() + row_first;
		for ( ; row1 != row1_end; ++row1, ++row2) {
			std::swap(*row1, *row2);
		}
	}

	template<typename _MatTy, typename _RealTy>
		requires requires(_MatTy __m) { __m.data(); __m.cols(); }
	void matrix_row_scale(_MatTy& matrix, size_t matrix_row, _RealTy rate, size_t offset = 0) {
		if constexpr (row_optimized_matrix<_MatTy>) {
			matrix.row(matrix_row, offset) *= rate;
		} else {
			auto*       row     = matrix.data() + matrix_row * matrix.cols() + offset;
			const auto* row_end = matrix.data() + (matrix_row+1) * matrix.cols();
			for ( ; row != row_end; ++row) {
				(*row) = (*row) * rate;
			}
		}
	}

	template<typename _MatTy, typename _RealTy>
		requires requires(_MatTy __m) { __m.data(); __m.cols(); __m.rows(); }
	void matrix_row_scale(_MatTy& matrix, size_t matrix_row, _RealTy rate, size_t row_first, size_t row_last) {
		if constexpr (row_optimized_matrix<_MatTy>) {
			matrix.row(matrix_row, row_first, row_last) *= rate;
		} else {
			auto*       row     = matrix.data() + matrix_row * matrix.cols() + row_first;
			const auto* row_end = matrix.data() + matrix_row * matrix.cols() + row_last;
			for ( ; row != row_end; ++row) {
				(*row) = (*row) * rate;
			}
		}

		if (matrix.rows() != matrix.cols()) {
			if (row_last != matrix.cols()) {// augmented-matrix, scale the y in Ax=y,  
				size_t diags      = std::min(matrix.rows(), matrix.cols());
				size_t row_first2 = std::max(diags, row_last);
				matrix_row_scale(matrix, matrix_row, rate, row_first2);
			}
		}
	}

	template<typename _MatTy, typename _RealTy>
		requires requires(_MatTy __m) { __m.data(); __m.cols(); }
	void matrix_row_eliminate(_MatTy& matrix, size_t Arow, size_t Brow, _RealTy rate, size_t offset = 0) {
		if constexpr (row_optimized_matrix<_MatTy>) {
			matrix.row(Brow, offset) *= rate;
			matrix.row(Arow, offset) += matrix.row(Brow, offset);
		} else {
			auto*       row1     = matrix.data() + Arow * matrix.cols() + offset;
			const auto* row1_end = matrix.data() + (Arow+1) * matrix.cols();
			auto*       row2     = matrix.data() + Brow * matrix.cols() + offset;
			for ( ; row1 != row1_end; ++row1, ++row2) {
				(*row1) += (*row2) * rate;
			}
		}
	}

	template<typename _MatTy, typename _RealTy>
		requires requires(_MatTy __m) { __m.data(); __m.cols(); __m.rows(); }
	void matrix_row_eliminate(_MatTy& matrix, size_t Arow, size_t Brow, _RealTy rate, size_t row_first, size_t row_last) {
		if constexpr (row_optimized_matrix<_MatTy>) {
			matrix.row(Brow, row_first, row_last) *= rate;
			matrix.row(Arow, row_first, row_last) += matrix.row(Brow, row_first, row_last);
		} else {
			auto*       row1     = matrix.data() + Arow * matrix.cols() + row_first;
			const auto* row1_end = matrix.data() + Arow * matrix.cols() + row_last;
			auto*       row2     = matrix.data() + Brow * matrix.cols() + row_first;
			for ( ; row1 != row1_end; ++row1, ++row2) {
				(*row1) += (*row2) * rate;
			}
		}

		if (matrix.rows() != matrix.cols()) {
			if (row_last != matrix.cols()) {// augmented-matrix, eliminate the y in Ax=y,  
				size_t diags      = std::min(matrix.rows(), matrix.cols());
				size_t row_first2 = std::max(diags, row_last);
				matrix_row_eliminate(matrix, Arow, Brow, rate, row_first2);
			}
		}
	}

	template<typename _MatTy1, typename _MatTy2>
		requires requires(_MatTy1 __m1, _MatTy2 __m2) { __m1.data(); __m1.cols(); __m2.data(); __m2.cols(); }
	void matrix_row_swap(_MatTy1& matrix1, size_t matrix1_row, _MatTy2& matrix2, size_t matrix2_row, size_t offset = 0) {
		assert( matrix1.cols() == matrix2.cols() );

		auto* row1 = matrix1.data() + matrix1_row * matrix1.cols() + offset;
		auto* row2 = matrix2.data() + matrix2_row * matrix2.cols() + offset;
		for (size_t j = offset; j != matrix1.cols(); ++j) {
			std::swap(*row1++, *row2++);
		}
	}

	// { find matrix major position }
	template<typename _MatTy>
	struct const_major_iterator {
		using matrix_type = _MatTy;
		using scalar_type = typename matrix_traits<_MatTy>::scalar_type;

		static std::array<size_t,2> _First_major(const matrix_type& _Matrix) {
			// search a not equal than 0.0+-_Threshould in every colume from {0,0}
			std::array<size_t,2> _Where = { 0, 0 };
			for ( ; _Where[1] != _Matrix.cols(); ++_Where[1], _Where[0] = 0) {
				for ( ; _Where[0] != _Matrix.rows(); ++_Where[0]) {// Has major in the colume ?
					if ( abs(_Matrix.at(_Where[0],_Where[1])) > std::numeric_limits<scalar_type>::epsilon() ) {
						return _Where;
					}
				}
			}

			return std::array<size_t,2>{ _Matrix.rows(), _Matrix.cols() };
		}

		static std::array<size_t,2> _Last_major(const matrix_type& _Matrix) {
			// inverse_search a not equal than 0.0+-_Threshould in every row from {_Matrix.rows(), _Matrix.cols()}
			std::array<size_t,2> _Where = { _Matrix.rows(), _Matrix.cols() };
			do {
				_Where[0] -= 1;
				_Where[1] = 0;
				for ( ; _Where[1] != _Matrix.cols(); ++_Where[1]) {// Has major in the row ?
					if (abs(_Matrix.at(_Where[0],_Where[1])) > std::numeric_limits<scalar_type>::epsilon() ) {
						return _Where;
					}
				}
			} while (_Where[0] != 0);

			return std::array<size_t,2>{ static_cast<size_t>(-1) , static_cast<size_t>(-1) };
		}

		static std::array<size_t,2> _Next_major(const matrix_type& _Matrix, std::array<size_t,2> _Pos) {// _Next major-pos, _Pos must be valid major-pos
			assert( _Pos[0] < _Matrix.rows() && _Pos[1] < _Matrix.cols() );

			std::array<size_t,2> _Where = { _Pos[0] + 1, _Pos[1] + 1 };
			for ( ; _Where[1] != _Matrix.cols(); ++_Where[1], _Where[0] = _Pos[0] + 1) {
				for ( ; _Where[0] != _Matrix.rows(); ++_Where[0]) {// [ _Pos[1]+1, rows )
					if ( abs(_Matrix.at(_Where[0],_Where[1])) > std::numeric_limits<scalar_type>::epsilon() ) {
						return _Where;
					}
				}
			}

			return std::array<size_t,2>{ _Matrix.rows(), _Matrix.cols() };
		}

		static std::array<size_t,2> _Prev_major(const matrix_type& _Matrix, std::array<size_t,2> _Pos) {// _Pos must be valid major-pos
			assert( _Pos[0] < _Matrix.rows() && _Pos[1] < _Matrix.cols() );

			if (_Pos[0] == _Pos[1]) {// Rank(_Matrix) == _Matrix.diags()
				return std::array<size_t, 2>{ _Pos[0] - 1, _Pos[1] - 1 };
			} else {
				std::array<size_t,2> _Where = { _Pos[0] - 1, _Pos[1] - 1 };
				do {
					_Where[0] -= 1;
					_Where[1] = 0;
					for ( ; _Where[1] != _Pos[1]; ++_Where[1]) {// Has major in the row ?
						if ( abs(_Matrix.at(_Where[0],_Where[1])) > std::numeric_limits<scalar_type>::epsilon() ) {
							return _Where;
						}
					}
				} while (_Where[0] != 0);
				
				return std::array<size_t,2>{ static_cast<size_t>(-1), static_cast<size_t>(-1) };
			}
		}
	
		explicit const_major_iterator(const matrix_type& _Matrix) : matrix_ptr(&_Matrix), position(_First_major(_Matrix)) { 
			// Check major position
			for (size_t i = 0; i != position[0]; ++i) {
				for (size_t j = position[1] + 1; j != matrix_ptr->cols(); ++j) {
					if ( abs(matrix_ptr->at(i, j)) > std::numeric_limits<scalar_type>::epsilon() ) {
						throw std::exception("calculation::major_iterator::no-swapable");
					}
				}
			}
		}

		const_major_iterator(const matrix_type& _Matrix, std::array<size_t,2> _Mjpos) : matrix_ptr(&_Matrix), position(_Mjpos) {
			/*if (_Mjpos[0] == _Matrix.rows() && _Mjpos[1] == _Matrix.cols()) {
		
			} else {
				if ( _Mjpos[0] > _Matrix.rows() || _Mjpos[1] > _Matrix.cols() ) {
					throw std::exception("clmagic::major_iterator::invalid-major-pos");
				}
				if ( abs(_Matrix.at(_Mjpos[0], _Mjpos[1])) < std::numeric_limits<scalar_type>::epsilon() ) {
					throw std::exception("clmagic::major_iterator::invalid-major-pos");
				}
			}*/
		}

		const scalar_type& operator*() const {
			return matrix_ptr->at(position[0], position[1]);
		}

		const_major_iterator& operator++() {
			// 1. Get _Next major position
			std::array<size_t, 2> _Next = _Next_major(*matrix_ptr, position);
			// 2. Check _Next major position
			for (size_t i = position[0] + 1; i != _Next[0]; ++i) {
				for (size_t j = position[1] + 1; j != matrix_ptr->cols(); ++j) {
					if (abs(matrix_ptr->at(i, j)) > std::numeric_limits<scalar_type>::epsilon()) {
						throw std::exception("calculation::major_iterator::no-swapable");
					}
				}
			}

			position = _Next;
			return (*this);
		}
		const_major_iterator& operator--() {
			position = _Prev_major(*matrix_ptr, position);
			return (*this);
		}
		const_major_iterator operator--(int) {
			const_major_iterator _Tmp = *this;
			--(*this);
			return _Tmp;
		}
		const_major_iterator operator++(int) {
			const_major_iterator _Tmp = *this;
			++(*this);
			return _Tmp;
		}

		bool operator==(const const_major_iterator& _Right) const {
			return (matrix_ptr == _Right.matrix_ptr) && (position[0] == _Right.position[0]) && (position[1] == _Right.position[1]);
		}
		bool operator!=(const const_major_iterator& _Right) const {
			return !(*this == _Right);
		}
		bool operator<(const const_major_iterator& _Right) const {
			assert(matrix_ptr == _Right.matrix_ptr);
			return (position[0] + position[1]) < (_Right.position[0] + _Right.position[1]);
		}
		bool operator>(const const_major_iterator& _Right) const {
			return (_Right < *this);
		}
		bool operator<=(const const_major_iterator& _Right) const {
			return !(*this > _Right);
		}
		bool operator>=(const const_major_iterator& _Right) const {
			return !(*this < _Right);
		}

		const scalar_type* ptr() const {
			return matrix_ptr->data() + position[0] * matrix_ptr->cols() + position[1];
		}
		size_t size() const {
			return matrix_ptr->cols() - position[1];
		}

		void seek_to_first() {
			this->position = _First_major(*matrix_ptr);
		}
		void seek_to_last() {
			this->position = _Last_major(*matrix_ptr);
		}

		const matrix_type* matrix_ptr;
		std::array<size_t,2> position;
	};

	template<typename _MatTy>
	struct major_iterator : public const_major_iterator<_MatTy> {
		using _Mybase     = const_major_iterator<_MatTy>;
		using matrix_type = _MatTy;
		using scalar_type = typename matrix_traits<_MatTy>::scalar_type;

		explicit major_iterator(matrix_type& _Matrix) : _Mybase(_Matrix) {}

		major_iterator(matrix_type& _Matrix, std::array<size_t,2> _Mjpos) : _Mybase(_Matrix, _Mjpos) {}

		scalar_type& operator*() {
			return const_cast<scalar_type&>(_Mybase::operator*());
		}

		major_iterator& operator++() {
			std::array<size_t,2> _Next = _Mybase::_Next_major(*_Mybase::matrix_ptr, _Mybase::position);
			if (_Next[0] != _Mybase::position[0] + 1 && /*safe_check*/_Next[0] < _Mybase::matrix_ptr->rows()) {
				matrix_type& _Matrix = const_cast<matrix_type&>(*_Mybase::matrix_ptr);
				matrix_row_swap(_Matrix, _Next[0], _Matrix, _Mybase::position[0] + 1);
				_Next[0] = _Mybase::position[0] + 1;
			}

			_Mybase::position = _Next;
			return *this;
		}

		major_iterator& operator--() {
			_Mybase::operator--();
			return *this;
		}

		major_iterator operator--(int) {
			major_iterator _Tmp = *this;
			--(*this);
			return _Tmp;
		}

		major_iterator operator++(int) {
			major_iterator _Tmp = *this;
			++(*this);
			return _Tmp;
		}
	};

	template<bool _Scaleable, typename _MatTy>
	void matrix_solve_to_upper_triangular(_MatTy& A) {
		using scalar_type = typename matrix_traits<_MatTy>::scalar_type;

		major_iterator<_MatTy> _First(A);
		major_iterator<_MatTy> _Last(A, { A.rows(), A.cols() });

		for ( ; _First != _Last; ++_First) {
			size_t      major_row = _First.position[0];
			size_t      major_col = _First.position[1];
			scalar_type major     = A.at(major_row, major_col);
		
			if constexpr (_Scaleable) {
				// 1. Scale major row
				if (major != 1) {
					matrix_row_scale(A, major_row, (1 / major), /*offset*/major_col);
				}

				// 2. Eliminate below row
				for (size_t i = major_row + 1; i != A.rows(); ++i) {
					if (A.at(i, major_col) != 0) {
						matrix_row_eliminate(A, i, major_row, -A.at(i,major_col), /*offset*/major_col);
					}
				}
			} else {
				// 1. Direct eliminate below row
				for (size_t i = major_row + 1; i != A.rows(); ++i) {
					if (A.at(i, major_col) != 0) {
						matrix_row_eliminate(A, i, major_row, (-A.at(i,major_col)) / major, /*offset*/major_col);
					}
				}
			}
		}
		/*<idea> major * C = 1        : We get C
							C = 1/major 
		</idea>*/
		/*<idea> A[i][j] + major * C = 0 : We get C
					C = (-A[i][j]) / major
		</idea>*/
	}

	template<bool _Scaleable, typename _MatTy>
	void matrix_solve_to_lower_triangular(_MatTy& A) {
		using scalar_type = typename matrix_traits<_MatTy>::scalar_type;

		major_iterator<_MatTy> _First(A, major_iterator<_MatTy>::_Last_major(A));
		major_iterator<_MatTy> _Last(A, { static_cast<size_t>(-1) , static_cast<size_t>(-1) });

		for ( ; _First != _Last; --_First) {
			size_t      major_row = _First.position[0];
			size_t      major_col = _First.position[1];
			scalar_type major     = A.at(major_row, major_col);

			if constexpr (_Scaleable) {
				if (major != 1) {
					matrix_row_scale(A, major_row, (1 / major), /*offset*/ 0, major_col+1);
				}

				for (size_t i = major_row-1; i != size_t(-1); --i) {
					if (A.at(i, major_col) != 0) {
						matrix_row_eliminate(A, i, major_row, (-A.at(i,major_col)), /*offset*/ 0, major_col+1);
					}
				}
			} else {
				for (size_t i = major_row-1; i != size_t(-1); --i) {
					if (A.at(i, major_col) != 0) {
						matrix_row_eliminate(A, i, major_row, (-A.at(i,major_col)) / major, /*offset*/ 0, major_col+1);
					}
				}
			}
		}
	}


	template<typename _MatTy1, typename _MatTy2>
	struct const_augmented_matrix {
		using first_matrix_type  = _MatTy1;
		using second_matrix_type = _MatTy2;
		using scalar_type        = typename matrix_traits<_MatTy1>::scalar_type;

		const_augmented_matrix(const first_matrix_type& _Arg1, const first_matrix_type& _Arg2) : first(&_Arg1), second(&_Arg2) {}

		size_t rows() const {
			return first.rows() + second.rows();
		}
		size_t cols() const {
			return std::min<size_t>(first.cols(), second.cols());
		}
		size_t rank() const {
			return std::max(first.rank(), second.rank());
		}

		const scalar_type& at(size_t i, size_t j) const {
			return ( j < first->cols()
				? first->at(i, j)
				: second->at(i, j - first->cols()) );
		}
		template<typename _Idxty>
		const scalar_type& at(const _Idxty& pos) const {
			if ( pos[1] < first->cols() ) {
				return first->at(pos);
			} else {
				_Idxty pos2 = pos;
				pos2[1] -= first->cols();
				return second->at(pos2);
			}
		}

		std::string to_string() const {
			using std::to_string;
			std::string _Str = "{";
			for (size_t i = 0; i != this->rows(); ++i) {
				_Str += '{';
				
				for (size_t j = 0; j != this->first->cols(); ++j) {
					_Str += to_string(this->first->at(i, j));
					_Str += ' ';
				}

				_Str += " | ";

				for (size_t j = 0; j != this->second->cols(); ++j) {
					_Str += to_string(this->second->at(i, j));
					_Str += ' ';
				}

				_Str.back() = '}';
				_Str += '\n';
			}
			_Str.back() = '}';

			return std::move(_Str);
		}

		const first_matrix_type*  first;
		const second_matrix_type* second;
	};

	template<typename _MatTy1, typename _MatTy2>
	struct augmented_matrix : public const_augmented_matrix<_MatTy1, _MatTy2> {
		using _Mybase            = const_augmented_matrix<_MatTy1, _MatTy2>;
		using first_matrix_type  = _MatTy1;
		using second_matrix_type = _MatTy2;
		using scalar_type        = typename matrix_traits<_MatTy1>::scalar_type;

		augmented_matrix(first_matrix_type& _Arg1, second_matrix_type& _Arg2) : _Mybase(_Arg1, _Arg2) {}

		scalar_type& at(size_t i, size_t j) {
			return const_cast<scalar_type&>(_Mybase::at(i, j));
		}
		template<typename _Idxty>
		scalar_type& at(const _Idxty& pos) {
			return const_cast<scalar_type&>(_Mybase::at(pos));
		}
	};

	template<typename _MatTy11, typename _MatTy12, typename _MatTy21,typename _MatTy22>
	void matrix_row_swap(augmented_matrix<_MatTy11,_MatTy12>& matrix1, size_t matrix1_row, augmented_matrix<_MatTy11,_MatTy12>& matrix2, size_t matrix2_row, size_t offset = 0) {
		assert( matrix1.cols() == matrix2.cols() );

		for (size_t j = offset; j != matrix1.cols(); ++j) {
			std::swap( matrix1.at(matrix1_row,j), matrix2.at(matrix2_row,j) );
		}
	}
}

namespace Gaussian {
	// { eliminate to simplest-matrix }
	template<typename _MatTy> inline
	void elimination(_MatTy& matrix) {
		calculation::matrix_solve_to_upper_triangular<false>(matrix);
		//remove_error(A.data(), A.size(), 0.00002F);
		calculation::matrix_solve_to_lower_triangular<true>(matrix);
	}
}

namespace Bareiss {
	// { eliminate to simple-matrix(Not-simplest) }
	template<typename _MatTy>
	void elimination(_MatTy& matrix) {
		using scalar_type = typename calculation::matrix_traits<_MatTy>::scalar_type;

		for (size_t k = 0; k < matrix.diags(); ++k) {
			scalar_type pivot   = matrix.at(k, k);
			scalar_type divider = k != 0 ? matrix.at(k-1, k-1) : static_cast<scalar_type>(1);

			for (size_t i = 0; i != matrix.rows(); ++i) {
				if (i != k) {
					scalar_type lb = matrix.at(i, k);// the value overlapped, when in-place
					for (size_t j = 0; j != matrix.cols(); ++j) {
						matrix.at(i,j) = (pivot * matrix.at(i,j) - lb * matrix.at(k,j)) / divider;
					}
				}
			}
		}
	}
}

namespace calculation {
	/*<theorem>
		inv(A*B)  = inv(B) * inv(A)
		inv(T(A)) = T(inv(A))
	</theorem>*/

	// { L(identity), U(source), in place }
	template<typename _MatTy1, typename _MatTy2>
		requires requires(_MatTy1 __m1) { __m1.rows(); __m1.cols(); __m1.at(0, 0); }
	bool LU_decompose_inplace(_MatTy1& L, _MatTy2& U) {
		for (size_t k = 0; k < U.rows() - 1; ++k) {// k:[ 0, U.rows()-1 )
			const auto& _Major = U.at(k, k);
			
			// 1. Check major is ZERO
			if ( iszero(_Major) ) {
				return false;
			}

			// 2. Loop over rows [ k+1, rows ) 
			for (size_t i = k + 1; i != U.rows(); ++i) {// i:[ k+1, U.rows() )
				
				// 3. Get row-scale-factor into L.at(i,k)
				L.at(i, k) = U.at(i, k) / _Major;
				
				// 4. Eliminate row, after L.row(i)*U.col(j) == A.at(i,j) 
				if constexpr ( row_optimized_matrix<_MatTy2> ) {
					U.row(i, k, U.cols()) += U.row(k, k, U.cols()) * (-L.at(i, k));
				} else {
					for (size_t j = k; j != U.cols(); ++j) {
						U.at(i,j) += U.at(i,j) * (-L.at(i, k));
					}
				}

				/*<debug>
					for (size_t j = k; j != U.cols(); ++j) {
						std::cout << std::to_string(dot(L.row(i), U.col(j))) << ' ';
					}
					std::cout << std::endl;
				</debug>*/
			}
		}
		return true;
	}

	// { A to L*U }
	template<typename _MatTy1, typename _MatTy2>
		requires requires(_MatTy1 __m) { __m.rows(); __m.cols(); __m.at(0, 0); }
	bool LU_decompose(const _MatTy1& A, _MatTy2& L, _MatTy1& U) {
		if (&U != &A) {
			U = A;
		}
		return LU_decompose_inplace(L, U);
	}

	template<typename _MatTy1, typename _MatTy2, typename _MatTy3>
	void LU_apply(const _MatTy1& L, const _MatTy2& U, const _MatTy3& b, _MatTy3& x) {
		if (&x != &b) {
			x = b;
		}
		
		// solve [L,b] to [E,y]
		for (size_t k = 0; k != x.rows() - 1; ++k) {
			for (size_t i = k + 1; i != x.rows(); ++i) {
				x.row(i) += -L.at(i, k) * x.row(k);
			}
		}

		// solve [U,y] to [E,x]
		for (size_t k = x.rows() - 1; k != -1; --k) {
			x.row(k) /= U.at(k, k);// scalar to 1
			for (size_t i = k - 1; i != -1; --i) {
				x.row(i) += -U.at(i, k) * x.row(k);
			}
		}

		return x;
	}




	template<typename _Ty>
	void _Matrix_copy(const _Ty* _Src_data, size_t _Src_width, size_t _Src_height, size_t _Src_width_offset,
						    _Ty* _Dst_data, size_t _Dst_width, size_t _Dst_height, size_t _Dst_width_offset,
							size_t _Width, size_t _Height) {
		assert(_Height <= _Src_height);
		assert(_Height <= _Dst_height);
		assert(_Src_width_offset + _Width <= _Src_width);
		assert(_Dst_width_offset + _Width <= _Dst_width);
		for (size_t j = 0; j != _Height; ++j) {
			auto       _First = _Src_data + j * _Src_width + _Src_width_offset;
			const auto _Last  = _First + _Width;
			auto       _Dest  = _Dst_data + j * _Dst_width + _Dst_width_offset;
			std::copy(_First, _Last, _Dest);
		}
	}

	template<typename _Ty>
	void _Matrix_copy(const _Ty* _Src_data, size_t _Src_width, size_t _Src_height, size_t _Src_width_offset, size_t _Src_height_offset, 
		                    _Ty* _Dst_data, size_t _Dst_width, size_t _Dst_height, size_t _Dst_width_offset, size_t _Dst_height_offset,
							size_t _Width, size_t _Height) {
		assert(_Src_height_offset + _Height <= _Src_height);
		assert(_Dst_height_offset + _Height <= _Dst_height);
		_Matrix_copy(_Src_data + _Src_height_offset*_Src_width, _Src_width, _Src_height, _Src_height_offset,
			         _Dst_data + _Dst_height_offset*_Dst_width, _Dst_width, _Dst_height, _Dst_width_offset, _Width, _Height);
	}

	template<typename _Ty>
	void _Matrix_copy(const _Ty* _Src_data, size_t _Src_width, size_t _Src_height, size_t _Src_width_offset, size_t _Src_height_offset, 
		                    _Ty* _Dst_data, size_t _Dst_width, size_t _Dst_height, size_t _Dst_width_offset, size_t _Dst_height_offset) {
		auto _Common_width  = std::min(_Src_width-_Src_width_offset, _Dst_width-_Dst_width_offset);
		auto _Common_height = std::min(_Src_height-_Src_height_offset, _Dst_height-_Dst_height_offset);
		_Matrix_copy(_Src_data, _Src_width, _Src_height, _Src_width_offset, _Src_height_offset,
			_Dst_data, _Dst_width, _Dst_height, _Dst_width_offset, _Dst_height_offset,
			_Common_width, _Common_height);
	}


	template<typename _SclTy, typename _BlkTy>
	class _Submatrix_data {
		_SclTy*         _My_first   = nullptr;
		_SclTy*         _My_last    = nullptr;
		std::unique_ptr<_SclTy> _My_data = nullptr;

		size_t       _My_dims       = 0;
		size_t       _My_sizes[4]   = {0, 0, 0, 0};// [0]:width, [1]:height, [2]:depth, [3]:time
		size_t       _My_strides[4] = {1, 0, 0, 0};// [0]:elements_stride, [1]:rows_stride, [2]:slices_stride, [3]:spaces_stride

	public:
		using block_type  = _BlkTy;
		using scalar_type = _SclTy;

		size_t dims() const {
			return _My_dims;
		}
		size_t size() const {
			if (_My_sizes == nullptr) {
				return 0;
			} else {
				return std::accumulate(_My_sizes, _My_sizes+_My_dims, size_t(1), std::multiplies<size_t>());
			}
		}
		size_t width() const {// vector of perline, the vector's size is cols 
			assert(_My_dims >= 1);
			return _My_sizes[0];
		}
		size_t height() const {
			assert(_My_dims >= 2);
			return _My_sizes[1];
		}
		size_t depth() const {
			assert(_My_dims >= 3);
			return _My_sizes[3];
		}
		size_t time() const {
			assert(_My_dims >= 4);
			return _My_sizes[3];
		}

		size_t elements_stride() const {
			assert(_My_dims >= 1);
			return _My_strides[0];
		}
		size_t rows_stride() const {
			assert(_My_dims >= 2);
			return _My_strides[1];
		}
		size_t slices_stride() const {
			assert(_My_dims >= 3);
			return _My_strides[2];
		}
		size_t spaces_stride() const {
			assert(_My_dims >= 4);
			return _My_strides[3];
		}

		bool is_dynamic_memory() const {
			return (_My_data != nullptr);
		}
		bool is_continue() const {
			bool _Is_continue = _My_strides[0] == 1;
			for (size_t i = 1; i != _My_dims; ++i) {
				_Is_continue |= (_My_strides[i] == 0);
			}
			return _Is_continue;
		}

		size_t index(size_t _Row, size_t _Col) const {
			return (_Row*(width()*elements_stride() + rows_stride()) + _Col * elements_stride());
		}

		scalar_type& at(size_t _Row, size_t _Col) {
			return _My_first[index(_Row, _Col)];
		}
		
		const scalar_type& at(size_t _Row, size_t _Col) const {
			return _My_first[index(_Row, _Col)];
		}
		
		scalar_type* ptr() {
			assert( this->is_continue() );
			return _My_first;
		}

		const scalar_type* ptr() const {
			assert( this->is_continue() );
			return _My_first;
		}
	
		void resize(size_t _Width, size_t _Height) {
			std::unique_ptr<scalar_type> _New_data = std::unique_ptr<scalar_type>(new scalar_type[_Width * _Height]);// exception safety
			if (_My_dims >= 2) {
				size_t _Common_height = std::min(_Height, this->height());
				size_t _Common_width  = std::min(_Width, this->width());
				for (size_t j = 0; j != _Common_height; ++j) {
					auto       _First = _My_first + index(j, 0);
					const auto _Last  = _First + _Common_width * elements_stride();
					auto       _Dest  = _New_data.get() + j * _Width;
					for ( ; _First != _Last; _First += elements_stride(), ++_Dest) {
						*_Dest = *_First;
					}
				}
			} else if(_My_dims == 1) {
				size_t _Common_width = std::min(_Width, this->width());
				auto       _First = _My_first;
				const auto _Last  = _First + _Common_width * elements_stride();
				auto       _Dest  = _New_data.get();
				for ( ; _First != _Last; _First += elements_stride(), ++_Dest) {
					*_Dest = *_First;
				}
			} else {// _My_dims == 0
				// do nothing
			}
			_My_data       = std::move(_New_data);
			_My_first      = _My_data.get();
			_My_last       = _My_first + _Width*_Height;
			_My_sizes[0]   = _Width;
			_My_sizes[1]   = _Height;
			_My_strides[0] = 1;
			_My_strides[1] = 0;
			_My_dims       = 2;
		}

		void swap(_Submatrix_data& _Right) {
			std::swap(_My_first, _Right._My_first);
			std::swap(_My_last, _Right._My_last);
			std::swap(_My_data, _Right._My_data);

			std::swap(_My_dims, _Right._My_dims);
			for (size_t i = 0; i != 4; ++i) {
				std::swap(_My_sizes[i], _Right._My_sizes[i]);
				std::swap(_My_strides[i], _Right._My_strides[i]);
			}
		}

		void release() {
			_My_first = nullptr;
			_My_last  = nullptr;
			_My_data  = nullptr;
			_My_dims  = 0;
		}

		/*subvector<scalar_type, block_type> row(size_t _Row) {
			auto _First = _My_first + index(_Row, 0);
			auto _Last  = _First + width()*elements_stride();
			return subvector<scalar_type, block_type>(_First, _Last, elements_stride());
		}

		const subvector<scalar_type, block_type> row(size_t _Row) const {
			return const_cast<_Submatrix_data&>(*this).row(_Row);
		}
	
		subvector<scalar_type, block_type> col(size_t _Col) {
			auto _First = _My_first + index(0, _Col);
			auto _Last  = _My_first + index(height(), _Col);
			return subvector<scalar_type, block_type>(_First, _Last, width()*elements_stride()+rows_stride());
		}

		const subvector<scalar_type, block_type> col(size_t _Col) const {
			return const_cast<_Submatrix_data&>(*this).col(_Col);
		}*/

		template<typename _UnaryOp>
		void for_each_scalar(_UnaryOp _Transform_op) {
			auto _First = _My_first;
			for (size_t t = 0; t != time(); ++t, _First += spaces_stride()) {
				for (size_t d = 0; d != depth(); ++d, _First += slices_stride()) {
					for (size_t h = 0; h != height(); ++h, _First += rows_stride()) {
						for (size_t w = 0; w != width(); ++w, _First += elements_stride()) {
							*_First = _Transform_op(*_First);
						}
					}
				}
			}
		}
	
		std::string to_string() const {
			std::string _Str;
			if (_My_dims == 2) {
				_Str += this->row(0).to_string();
				for (size_t i = 1; i != this->height(); ++i) {
					_Str += '\n';
					_Str += this->row(i).to_string();
				}
			}
			return _Str;
		}
	};

	template<typename _SclTy, typename _BlkTy = _SclTy>
	class submatrix : public _Submatrix_data<_SclTy, _BlkTy> {

	public:
		using block_type  = _BlkTy;
		using scalar_type = _SclTy;

		submatrix() = default;

		submatrix(submatrix&& _Right) noexcept {
			_Right.swap(*this);
			_Right.release();
		}

		submatrix operator*(const submatrix& _Right) const {
			assert( this->dims() == 2 );
			assert( this->width() == _Right.height() );
			submatrix _Result;
			_Result.resize(_Right.width(), this->height());
			for (size_t i = 0; i != _Result.height(); ++i) {
				for (size_t j = 0; j != _Result.width(); ++j) {
					scalar_type _Dot = scalar_type(0);
					for (size_t k = 0; k != this->width(); ++k) {
						_Dot += this->at(i, k) * _Right.at(k, j);
					}
					_Result.at(i, j) = _Dot;
				}
			}
			return std::move(_Result);
		}
	};


	// { static_size }
	template<typename _SclTy, size_t _Rows, size_t _Cols, typename _BlkTy = _SclTy>
	struct matrix {
		using scalar_type     = _SclTy;
		using block_type      = _BlkTy;
		using matrix_category = general_matrix_tag;

		using scalar_pointer       = scalar_type*;
		using const_scalar_pointer = const scalar_type*;
		using scalar_reference       = scalar_type&;
		using const_scalar_reference = const scalar_type&;

		_SclTy _Mydata[_Rows * _Cols];

		scalar_pointer data() {
			return _Mydata;
		}
		const_scalar_pointer data() const {
			return _Mydata;
		}
		constexpr size_t rows() const {
			return _Rows;
		}
		constexpr size_t cols() const {
			return _Cols;
		}
		constexpr size_t diags() const {
			return std::min(rows(), cols());
		}
		constexpr size_t size() const {
			return rows() * cols();
		}
		constexpr bool empty() const {
			return false;
		}
		size_t rank() const {
			size_t _Rank = 0;
			for (size_t i = 0; i != rows(); ++i) {
				const_scalar_pointer _The_row  = data() + i * cols();
				const_scalar_pointer _Next_row = _The_row + cols();
				if (std::find_if(_The_row, _Next_row, [](scalar_type value) { return value != static_cast<scalar_type>(0); }) != _Next_row) {
					++_Rank;
				}
			}

			return _Rank;
		}

		template<typename _Ty = scalar_type>
		_Ty* ptr(size_t i, size_t j) {
			return reinterpret_cast<_Ty*>(_Mydata + i * cols() + j);
		}
		template<typename _Ty = scalar_type>
		_Ty& at(size_t i, size_t j) {
			return *reinterpret_cast<_Ty*>(_Mydata + i * cols() + j);
		}
		template<typename _Ty = scalar_type>
		const _Ty* ptr(size_t i, size_t j) const {
			return reinterpret_cast<const _Ty*>(_Mydata + i * cols() + j);
		}
		template<typename _Ty = scalar_type>
		const _Ty& at(size_t i, size_t j) const {
			return *reinterpret_cast<const _Ty*>(_Mydata + i * cols() + j);
		}

		//row(size_t i);
		//row(size_t i, size_t start, size_t end);

		void fill(const_scalar_reference _Val) {
			std::fill(data(), data() + size(), _Val);
		}
		template<typename _Iter>
		void assign(_Iter _First, _Iter _Last) {
			std::copy(_First, _Last, data());
		}
		void assign(const_scalar_pointer _Array) {
			std::copy(_Array, _Array + size(), data());
		}

		//to_vector();

		void set_diag(const_scalar_reference _Val) {
			std::fill(data(), data() + size(), static_cast<scalar_type>(0));
			for (size_t k = 0; k != diags(); ++k) { this->at(k, k) = _Val; }
		}
		void set_identity() {
			set_diag(static_cast<scalar_type>(1));
		}
	};
	

	template<size_t _Rows, size_t _Cols, typename _BlkTy = float>
	using fmatrix = matrix<float, _Rows, _Cols, _BlkTy>;

	template<size_t _Rows, size_t _Cols, typename _BlkTy = double>
	using dmatrix = matrix<double, _Rows, _Cols, _BlkTy>;

	template<size_t _Rows, size_t _Cols, typename _BlkTy = int32_t>
	using imatrix = matrix<int32_t, _Rows, _Cols, _BlkTy>;

	template<typename _SclTy, size_t _Rows, typename _BlkTy = _SclTy>
	using square_matrix = matrix<_SclTy, _Rows, _Rows, _BlkTy>;

	template<typename _SclTy, typename _BlkTy = _SclTy>
	using matrix2x2 = matrix<_SclTy, 2, 2, _BlkTy>;

	template<typename _SclTy, typename _BlkTy = _SclTy>
	using matrix3x3 = matrix<_SclTy, 3, 3, _BlkTy>;

	template<typename _SclTy, typename _BlkTy = _SclTy>
	using matrix4x4 = matrix<_SclTy, 4, 4, _BlkTy>;


	/*- - - - - - - - - - - - - - - - - _Matrix_cast - - - - - - - - - - - - - - - - - - - -*/
	//template<typename _OutMatrix, 
	//	typename _InTy, size_t _InRows, size_t _InCols, typename _InBlock, typename _InMajor, typename _InMtag>
	//struct _Matrix_cast {
	//	using dest_matrix_type   = _OutMatrix;
	//	using source_matrix_type = matrix<_InTy, _InRows, _InCols, _InBlock, _InMajor, _InMtag>;

	//	dest_matrix_type operator()(const source_matrix_type& _Src) const {
	//		abort();
	//	}
	//};

	//template<typename _OutTy, size_t _OutRows, size_t _OutCols, typename _OutBlock, typename _OutMajor, typename _OutMtag,
	//		 typename _InTy,  size_t _InRows,  size_t _InCols,  typename _InBlock,  typename _InMajor,  typename _InMtag>
	//struct _Matrix_cast< matrix<_OutTy, _OutRows, _OutCols, _OutBlock, _OutMajor, _OutMtag>,
	//						    _InTy,  _InRows,  _InCols,  _InBlock,  _InMajor,  _InMtag > {
	//	using dest_matrix_type   = matrix<_OutTy, _OutRows, _OutCols, _OutBlock, _OutMajor, _OutMtag>;
	//	using source_matrix_type = matrix<_InTy,  _InRows,  _InCols,  _InBlock,  _InMajor,  _InMtag>;

	//	dest_matrix_type operator()(const source_matrix_type& _Source) const {
	//		dest_matrix_type _Destination(static_cast<_OutTy>(0));
	//	
	//		if _CONSTEXPR_IF( std::is_same_v<_InMajor,_OutMajor> ) {
	//			constexpr size_t _Common_rows = std::min(_OutRows, _InRows);
	//			constexpr size_t _Common_cols = std::min(_OutCols, _InCols);
	//			auto       _First = _Source.begin();
	//			const auto _Last  = _First + _Common_rows * _Source.cols();
	//			auto       _Dest  = _Destination.begin();
	//			for ( ; _First != _Last; _First += _Source.cols(), _Dest += _Destination.cols()) {
	//				::std::copy(_First, _First + _Common_cols, _Dest);
	//			}
	//		} else {// transpose(_Source)
	//			constexpr size_t _Common_rows = std::min(_OutRows, _InCols);
	//			constexpr size_t _Common_cols = std::min(_OutCols, _InRows);
	//			for (size_t i = 0; i != _Common_rows; ++i) {
	//				for (size_t j = 0; j != _Common_cols; ++j) {
	//					_Destination.at(i, j) = static_cast<_OutTy>(_Source.at(j, i));
	//				}
	//			}
	//		}

	//		return _Destination;
	//	}
	//};

	//template<typename _OutTy, size_t _OutRows, size_t _OutCols, typename _OutBlock, typename _OutMajor/*,diagonal-matrix*/,
	//		 typename _InTy,  size_t _InRows,  size_t _InCols,  typename _InBlock,  typename _InMajor, typename _InMtag>
	//struct _Matrix_cast< matrix<_OutTy, _OutRows, _OutCols, _OutBlock, _OutMajor, diagonal_matrix_tag>,
	//						    _InTy,  _InRows,  _InCols,  _InBlock,  _InMajor,  _InMtag > {
	//	using dest_matrix_type   = matrix<_OutTy, _OutRows, _OutCols, _OutBlock, _OutMajor, diagonal_matrix_tag>;
	//	using source_matrix_type = matrix<_InTy,  _InRows,  _InCols,  _InBlock,  _InMajor,  _InMtag>;

	//	dest_matrix_type operator()(const source_matrix_type& _Source) const {
	//		dest_matrix_type _Dest(static_cast<_OutTy>(0));
	//		const size_t _Common_diags = std::min(_Dest.diags(), _Source.diags());
	//		for (size_t k = 0; k != _Common_diags; ++k) {
	//			_Dest.at(k, k) = _Source.at(k, k);
	//		}
	//		return _Dest;
	//	}
	//};

	//template<typename _OutMatrix, 
	//	typename _InTy, size_t _InRows, size_t _InCols, typename _InBlock, typename _InMajor, typename _InMtag>
	//_OutMatrix matrix_cast(const matrix<_InTy, _InRows, _InCols, _InBlock, _InMajor, _InMtag>& _Source) {
	//		return _Matrix_cast<_OutMatrix, _InTy, _InRows, _InCols, _InBlock, _InMajor, _InMtag>()(_Source);
	//}

	/*- - - - - - - - - - - - - - - - - _Matrix_multiples - - - - - - - - - - - - - - - - - - -*/
	//template<typename _TyMat1, typename _TyMat2>
	//struct _Matrix_multiplies { };

	//template<typename _SclTy, size_t _Mx, size_t _Nx, size_t _Px, typename _BlkTy, typename _Major,
	//	typename _Mtag1, typename _Mtag2>
	//struct _Matrix_multiplies< matrix<_SclTy, _Mx, _Nx, _BlkTy, _Major, _Mtag1>, 
	//							matrix<_SclTy, _Nx, _Px, _BlkTy, _Major, _Mtag2> > {
	//	using dest_type  = matrix<_SclTy, _Mx, _Px, _BlkTy, _Major, _Mtag1>;
	//	using left_type  = matrix<_SclTy, _Mx, _Nx, _BlkTy, _Major, _Mtag1>;
	//	using right_type = matrix<_SclTy, _Nx, _Px, _BlkTy, _Major, _Mtag2>;

	//	dest_type operator()(const left_type& _Left, const right_type& _Right) const {
	//		dest_type _Result;
	//		_Result._Correct_tail_elements();
	//		for (size_t i = 0; i != _Result.rows(); ++i) {
	//			for (size_t j = 0; j != _Result.cols(); ++j) {
	//				_SclTy value = static_cast<_SclTy>(0);
	//				for (size_t k = 0; k != _Left.cols(); ++k) {
	//					value += _Left.at(i,k) * _Right.at(k,j);
	//				}
	//				_Result.at(i, j) = value;
	//			}
	//		}
	//		return _Result;
	//	}
	//};

	//// diagonal-matrix * diagonal-matrix
	//template<typename _SclTy, size_t _Mx, size_t _Nx, size_t _Px, typename _BlkTy, typename _Major>
	//struct _Matrix_multiplies< diagonal_matrix<_SclTy, _Mx, _Nx, _BlkTy, _Major>,
	//							diagonal_matrix<_SclTy, _Nx, _Px, _BlkTy, _Major> > {
	//	using dest_type  = diagonal_matrix<_SclTy, _Mx, _Px, _BlkTy, _Major>;
	//	using left_type  = diagonal_matrix<_SclTy, _Mx, _Nx, _BlkTy, _Major>;
	//	using right_type = diagonal_matrix<_SclTy, _Nx, _Px, _BlkTy, _Major>;

	//	dest_type operator()(const left_type& _Left, const right_type& _Right) const {
	//		const auto _Left_diag  = _Left.diag();
	//		const auto _Right_diag = _Right.diag();
	//		return dest_type(_Left_diag * _Right_diag);
	//	}
	//};

	//// matrix * diagonal-matrix
	//template<typename _SclTy, size_t _Mx, size_t _Nx, size_t _Px, typename _BlkTy, typename _Major, typename _Mtag1>
	//struct _Matrix_multiplies< matrix<_SclTy, _Mx, _Nx, _BlkTy, _Major, _Mtag1>,
	//						   diagonal_matrix<_SclTy, _Nx, _Px, _BlkTy, _Major> > {
	//	using dest_type  = matrix<_SclTy, _Mx, _Px, _BlkTy, _Major, _Mtag1>;
	//	using left_type  = matrix<_SclTy, _Mx, _Nx, _BlkTy, _Major, _Mtag1>;
	//	using right_type = diagonal_matrix<_SclTy, _Nx, _Px, _BlkTy, _Major>;

	//	dest_type operator()(const left_type& _Left, const right_type& _Right) const {
	//		dest_type    _Dest        = matrix_cast<dest_type>(_Left);
	//		const auto   _Right_diag  = _Right.diag();// constexpr
	//		const size_t _Common_cols = std::min(_Dest.cols(), _Right.cols());
	//		for (size_t i = 0; i != _Dest.rows(); ++i) {
	//			_Dest.row(i, 0, _Common_cols) *= _Right_diag(0, _Common_cols);
	//			/*for (size_t j = 0; j != _Common_cols; ++j) {
	//				_Dest.at(i, j) *= _Right_diag[j];
	//			}*/
	//		}

	//		return _Dest;
	//		/*
	//		[a b c]   [x 0 0]   [a*x b*y c*z]
	//		[d e f] * [  y  ] = [d*x e*y f*z]
	//		[g h i]   [    z]   [g*x h*y i*z]

	//		[a b c]   [x 0 0 0]   [a*x b*y c*z 0]
	//		[d e f] * [  y   0] = [d*x e*y f*z 0]
	//		[g h i]   [    z 0]   [g*x h*y i*z 0]

	//		[a b c]   [x 0]   [a*x b*y]
	//		[d e f] * [  y] = [d*x e*y]
	//		[g h i]   [   ]   [g*x h*y]
	//		*/
	//	}
	//};

	//// diagonal-matrix * matrix
	//template<typename _SclTy, size_t _Mx, size_t _Nx, size_t _Px, typename _BlkTy, typename _Major, typename _Mtag2>
	//struct _Matrix_multiplies< diagonal_matrix<_SclTy, _Mx, _Nx, _BlkTy, _Major>, 
	//						   matrix<_SclTy, _Nx, _Px, _BlkTy, _Major, _Mtag2> > {
	//	using dest_type  = matrix<_SclTy, _Mx, _Px, _BlkTy, _Major, _Mtag2>;
	//	using left_type  = diagonal_matrix<_SclTy, _Mx, _Nx, _BlkTy, _Major>;
	//	using right_type = matrix<_SclTy, _Nx, _Px, _BlkTy, _Major, _Mtag2>;

	//	dest_type operator()(const left_type& _Left, const right_type& _Right) const {
	//		dest_type _Dest = matrix_cast<dest_type>(_Right);
	//		for (size_t k = 0; k != _Dest.diags(); ++k) {
	//			_Dest.row(k) *= _Left.at(k, k);
	//		}

	//		return _Dest;
	//		/*
	//		① Aii * Bi., i∈{0,...M}
	//			[ x 0 0 ]     [ a b c ]   [x*a x*b x*c]   [x * [a b c]]
	//		A:[ 0 y 0 ] * B:[ d e f ] = [y*d y*e y*f] = [y * [d e f]]
	//			[ 0 0 z ]     [ g h i ]   [z*g z*h z*i]   [z * [g h i]]

	//		②Same of above
	//			[x 0 0 0]     [a b c d]
	//		A:[0 y 0 0] * B:[e f g h]
	//			[0 0 z 0]     [i j k l]
	//						[m n r s]

	//		[x*a x*b x*c x*d] = x * B0.
	//		[y*e y*f y*g y*h] = y * B1.
	//		[z*i z*j z*k z*l] = z * B2.

	//		③ 1.construct Zero matrix 2.compute K=min(M,N) matrix-mul
	//			[x 0 0]     [a b c]   [x * [a b c]]
	//		A:[0 y 0] * B:[d e f] = [y * [d e f]]
	//			[0 0 z]     [g h i]   [z * [g h i]]
	//			[0 0 0]               [     0     ]
	//		*/
	//	}
	//};

	//// matrixMxN * matrixNx1 = matrixMx1
	//template<typename _SclTy, size_t _Mx, size_t _Nx, typename _BlkTy, typename _Mtag>
	//struct _Matrix_multiplies < matrix<_SclTy, _Mx, _Nx, _BlkTy, colume_matrix_tag, _Mtag>,
	//							vector<_SclTy, _Nx, _BlkTy> >{
	//	using dest_type  = vector<_SclTy, _Mx, _BlkTy>;
	//	using left_type  = matrix<_SclTy, _Mx, _Nx, _BlkTy, colume_matrix_tag, _Mtag>;
	//	using right_type = vector<_SclTy, _Nx, _BlkTy>;

	//	dest_type operator()(const left_type& _Left, const right_type& _Right) const {
	//		dest_type _Result; 
	//		_Result._Correct_tail_elements();
	//		for (size_t i = 0; i != _Left.rows(); ++i) {
	//			_SclTy value = static_cast<_SclTy>(0);
	//			for (size_t j = 0; j != _Right.size(); ++j) {
	//				value += _Left.at(i, j) * _Right.at(j);
	//			}
	//			_Result[i] = value;
	//		}

	//		return _Result;
	//	}
	//};

	//// matrix1xM * matrixMxN *  = matrix1xN
	//template<typename _SclTy, size_t _Mx, size_t _Nx, typename _BlkTy, typename _Mtag>
	//struct _Matrix_multiplies < vector<_SclTy, _Mx, _BlkTy>,
	//							matrix<_SclTy, _Mx, _Nx, _BlkTy, row_matrix_tag, _Mtag> > {
	//	using dest_type  = vector<_SclTy, _Nx, _BlkTy>;
	//	using left_type  = vector<_SclTy, _Mx, _BlkTy>;
	//	using right_type = matrix<_SclTy, _Mx, _Nx, _BlkTy, row_matrix_tag, _Mtag>;

	//	dest_type operator()(const left_type& _Left, const right_type& _Right) const {
	//		/*        [a b]
	//		[x y z] * [d e]
	//				  [g h]
	//	   
	//		=> [x*a+y*d+z*g x*b+y*e+z*h]

	//		=> [x*a x*b] + [y*d y*e] + [z*g z*h]
	//		=> sum_of(V[i] * M.row(i))
	//		*/
	//		dest_type  _Result;
	//		_Result._Correct_tail_elements();

	//		for (size_t j = 0; j != _Right.cols(); ++j) {
	//			_Result[j] = _Left[0] * _Right.at(0, j);
	//		}
	//		for (size_t i = 1; i != _Right.rows(); ++i) {
	//			for (size_t j = 0; j != _Right.cols(); ++j) {
	//				_Result[j] += _Left[i] * _Right.at(i, j);
	//			}
	//		}
	//		return _Result;
	//	}
	//};

	template<typename _Iter>
	size_t count_invseq(_Iter _First, _Iter _Last) {// get sum of inverse sequence 
		auto _Count = size_t(0);
		auto _Where = std::is_sorted_until(_First, _Last);

		while (_Where != _Last) {
			_Count += std::count_if(_First, _Where, [_Where](size_t _Val) { return (_Val > (*_Where)); });
			++_Where;
		}
		return (_Count);
	}

	inline void _Seque(const std::vector<size_t>& _Numbers, std::vector<size_t>& _Tmp, std::vector<std::vector<size_t>>& _Result, size_t n) {
		if (n == 1) {// push last value
			_Tmp.push_back(_Numbers.front());
			_Result.push_back(_Tmp);
			_Tmp.pop_back();
			return;
		} else {// push _Numbers[any], and not have it at _Next cascade 
			auto _Next = std::vector<size_t>(n - 1);
			for (size_t i = 0; i != n; ++i) {
				_Tmp.push_back(_Numbers[i]);
				std::copy_if(_Numbers.begin(), _Numbers.end(), _Next.begin(), [&_Numbers, i](size_t a) { return (a != _Numbers[i]); });
				_Seque(_Next, _Tmp, _Result, n - 1);
				_Tmp.pop_back();
			}
		}
	}

	inline std::vector<std::vector<size_t>> seque(const std::vector<size_t>& _Numbers) {
		auto _Sequences = std::vector<std::vector<size_t>>();
		auto _Tmp = std::vector<size_t>();
		_Seque(_Numbers, _Tmp, _Sequences, _Numbers.size());
		return (_Sequences);
	}

	struct determinant_seque : public std::vector<std::pair<std::vector<size_t>, size_t>> {
		using number_type = size_t;
		using seque_type  = std::vector<number_type>;
		using pair_type   = std::pair<seque_type, number_type>;
		using _Mybase     = std::vector<pair_type>;

		determinant_seque() { }

		determinant_seque(const std::vector<size_t>& _Numbers) : _Mybase() {
			auto _Seq = seque(_Numbers);
			for (auto _First = _Seq.begin(); _First != _Seq.end(); ++_First) {
				auto& s = *_First;
				_Mybase::push_back(pair_type(s, count_invseq(s.cbegin(), s.cend())));
			}
			_Mysource = _Numbers;
		}

		std::vector<size_t> _Mysource;
	};

	template<typename T, size_t N, typename B> inline
	T determinant(const square_matrix<T,N,B>& matrix) {
		/* ∑(0,!M): (-1)^t * D[0][s₀] * D[1][s₁] * D[2][s₂]... * D[M-1][s(m-₁)] */
		if _CONSTEXPR_IF(N == 2) {
			return (matrix.at(0, 0) * matrix.at(1, 1) - matrix.at(0, 1) * matrix.at(1, 0));
		}

		T sigma = T(0);
		T S     = determinant_seque(set_number<determinant_seque::number_type>(0, matrix.rows()));
		/*for (size_t i = 0; i != S.size(); ++i) {
			auto _Test = S[i].first;
			for (size_t j = 0; j != _Test.size(); ++j) {
				std::cout << _Test[j] << " ";
			}
			std::cout << std::endl;
		}*/
		assert( S.size() == fact(matrix.rows()) );
		for (size_t i = 0; i != S.size(); ++i) {
			const auto& s = S[i].first;
			const auto& t = S[i].second;
			auto      rou = matrix.at(0, s[0]);
			for (size_t j = 1; j != s.size(); ++j) {
				rou *= matrix.at(j, s[j]);// high light
			}
			sigma += static_cast<T>(std::pow(-1, t)) * rou;
		}
		return sigma;
	}

}// namespace clmagic

// operator+, operator-, operator*, transpose, inverse, [solve]
template<typename T, size_t M, size_t N> inline
calculation::matrix<T,M,N> operator+(const calculation::matrix<T,M,N>& matrix1, const calculation::matrix<T,M,N>& matrix2){
	calculation::matrix<T,M,N> result;
	std::transform(matrix1.data(), matrix1.data() + matrix1.size(),
				   matrix2.data(),
				   result.data(), [](T a, T b) { return a + b; });
	return std::move(result);
}

template<typename T, size_t M, size_t N> inline
calculation::matrix<T,M,N> operator-(const calculation::matrix<T,M,N>& matrix1, const calculation::matrix<T,M,N>& matrix2){
	calculation::matrix<T,M,N> result;
	std::transform(matrix1.data(), matrix1.data() + matrix1.size(),
				   matrix2.data(),
				   result.data(), [](T a, T b) { return a - b; });
	return std::move(result);
}

template<typename T, size_t M, size_t N, size_t P>
calculation::matrix<T,M,P> operator*(const calculation::matrix<T,M,N>& matrix1, const calculation::matrix<T,N,P>& matrix2) {
	/*<theorem>
	result = { dot(A.row(i), B.col(0)), dot(A.row(i), B.col(1)), dot(A.row(i), B.col(2)), ..., dot(A.row(i), B.col(P)) }
			 { ... }
		   = { sum(A[i][k] * B[k][0]), sum(A[i][k] * B[k][1]), sum(A[i][k] * B[k][2]), ..., sum(A[i][k] * B[k][P]) }
	         { ... }
		   = sum(A[i][k], { B[k][0], B[k][1], B[k][2], ..., B[k][P] })
		     { ... }

	[0] [1] [2] [3] X [----0----]
	                  [----1----]
	                  [----2----]
	                  [----3----]
	</theorem>*/
	calculation::matrix<T,M,P> result{ static_cast<T>(0) };

	for (size_t i = 0; i != M; ++i) {
		if constexpr ( calculation::row_optimized_matrix<calculation::matrix<T, N, P>> ) {
			for (size_t k = 0; k != N; ++k) {
				result.row(i) += matrix1.at(i,k) * matrix2.row(k);
			}
		} else {
			for (size_t k = 0; k != N; ++k) {
				T a = matrix1.at(i, k);
				std::transform(matrix2.ptr(k,0), matrix2.ptr(k+1,0), 
							   result.ptr(i,0),
							   result.ptr(i,0), [a](T b, T c){ return a * b + c; });
			}
		}
	}

	return std::move(result);
}

template<typename T, size_t M, size_t N> inline
calculation::vector<T,M> operator*(const calculation::matrix<T,M,N>& matrix, const calculation::vector<T,N>& vector) {
	// MxN * Nx1 = Mx1
	calculation::matrix<T,M,1> result = matrix * reinterpret_cast<const calculation::matrix<T,N,1>&>(vector);
	return reinterpret_cast<const calculation::vector<T,M>&>( result );
}

template<typename T, size_t M, size_t N>
calculation::matrix<T,N,M> transpose(const calculation::matrix<T,M,N>& matrix) {
	calculation::matrix<T,N,M> result;
	for (size_t i = 0; i != M; ++i) {
		for (size_t j = 0; j != N; ++j) {
			result.at(j,i) = matrix.at(i,j);
		}
	}

	return std::move(result);
}

template<typename T, size_t M, size_t N>
bool solve(calculation::matrix<T,M,N>& matrix) {
	Bareiss::elimination(matrix);
	calculation::matrix_solve_to_upper_triangular<true>(matrix);
	return true;
}

template<typename T, size_t N>
calculation::matrix<T,N,N> inverse(const calculation::matrix<T,N,N>& matrix) {
	// 1. Construct a [matrix, Identity]
	calculation::matrix<T,N,N*2> temp{ static_cast<T>(0) };
	for (size_t i = 0; i != N; ++i) {
		std::copy(matrix.ptr(i, 0), matrix.ptr(i, N), temp.ptr(i, 0));
	}
	for (size_t k = 0; k != N; ++k) {
		temp.at(k,k+N) = static_cast<T>(1);
	}

	// 2. Bareiss eliminate [matrix, Identity] to _Result
	Bareiss::elimination(temp);

	// 3. Check the _Result is can_inverse
	bool can_inverse = true;
	for (size_t k = 0; k != N; ++k) {
		if ( abs(temp.at(k, k)) < std::numeric_limits<T>::min() ) {// equal 0
			can_inverse = false;
			break;
		}
	}

	if (can_inverse) {
		// 4. Eliminate _Result to [Identity, inverse_matrix]
		calculation::matrix_solve_to_upper_triangular<true>(temp);

		// 5. copy invers_matrix
		calculation::matrix<T,N,N> result;
		for (size_t i = 0; i != N; ++i) {
			std::copy(temp.ptr(i, N), temp.ptr(i, N * 2), result.ptr(i, 0));
		}
		return result;
	} else {
		return calculation::matrix<T,N,N>{ static_cast<T>(0) };
	}
}


// { matrix[M*N] to matrix[N*M], in place }
template<typename _Ty>
void transpose_inplace(_Ty* matrix, size_t M, size_t N) {
	assert( M != 0 && N != 0 );

	size_t MN    = M * N;
	size_t MNm1  = MN - 1;
	auto   flags = std::vector<bool>(MNm1, false);
	for (size_t i = 1; i != MNm1; ++i) {
		if (flags[i] == true) {
			continue;
		}

		size_t next = i;
		do {
			next = (next * M) % MNm1;
			std::next(matrix[i], matrix[next]);
			flags[next] = true;
		} while (next != i);
	}
}


#ifdef _INCLUDED_MM2
namespace calculation {
	template<size_t _Rows, size_t _Cols>
	using m128matrix = matrix<float, _Rows, _Cols, __m128>;

	using m128matrix4x4 = matrix<float, 4, 4, __m128>;
}

inline calculation::m128matrix<4,4> operator+(const calculation::m128matrix<4,4>& matrix1, const calculation::m128matrix<4,4>& matrix2) {
    calculation::m128matrix<4, 4> result;
    const __m128* lhs = reinterpret_cast<const __m128*>(matrix1.data());
    const __m128* rhs = reinterpret_cast<const __m128*>(matrix2.data());
    __m128*       dst = reinterpret_cast<__m128*>(result.data());

    *dst++ = _mm_add_ps(*lhs++, *rhs++);
    *dst++ = _mm_add_ps(*lhs++, *rhs++);
    *dst++ = _mm_add_ps(*lhs++, *rhs++);
    *dst = _mm_add_ps(*lhs, *rhs);

    return std::move(result);
}

inline calculation::m128matrix<4,4> operator-(const calculation::m128matrix<4,4>& matrix1, const calculation::m128matrix<4,4>& matrix2) {
    calculation::m128matrix<4, 4> result;
    const __m128* lhs = reinterpret_cast<const __m128*>(matrix1.data());
    const __m128* rhs = reinterpret_cast<const __m128*>(matrix2.data());
    __m128*       dst = reinterpret_cast<__m128*>(result.data());

    *dst++ = _mm_sub_ps(*lhs++, *rhs++);
    *dst++ = _mm_sub_ps(*lhs++, *rhs++);
    *dst++ = _mm_sub_ps(*lhs++, *rhs++);
    *dst = _mm_sub_ps(*lhs, *rhs);

    return std::move(result);
}

inline calculation::m128matrix<4,4> operator*(const calculation::m128matrix<4,4>& matrix, const float scalar) {
    calculation::m128matrix<4, 4> result;
    const __m128* lhs = reinterpret_cast<const __m128*>(matrix.data());
    const __m128  rhs = _mm_set1_ps(scalar);
    __m128*       dst = reinterpret_cast<__m128*>(result.data());

    *dst++ = _mm_sub_ps(*lhs++, rhs);
    *dst++ = _mm_sub_ps(*lhs++, rhs);
    *dst++ = _mm_sub_ps(*lhs++, rhs);
    *dst = _mm_sub_ps(*lhs, rhs);

    return std::move(result);
}

inline calculation::m128matrix<4,4> operator*(const calculation::m128matrix<4,4>& matrix1, const calculation::m128matrix<4,4>& matrix2) {
    calculation::m128matrix<4,4> result;
    const float*  lhs = matrix1.data();
    const __m128* rhs = reinterpret_cast<const __m128*>(matrix2.data());
    __m128*       dst = reinterpret_cast<__m128*>(result.data());

    for (size_t i = 0; i != 4; ++i) {
        *dst = _mm_mul_ps(_mm_set1_ps(*lhs), *rhs); ++lhs; ++rhs;
        *dst = _mm_add_ps(*dst, _mm_mul_ps(_mm_set1_ps(*lhs), *rhs)); ++lhs; ++rhs;
        *dst = _mm_add_ps(*dst, _mm_mul_ps(_mm_set1_ps(*lhs), *rhs)); ++lhs; ++rhs;
        *dst = _mm_add_ps(*dst, _mm_mul_ps(_mm_set1_ps(*lhs), *rhs)); ++lhs; rhs = reinterpret_cast<const __m128*>(matrix2.data()); ++dst;
    }

    return std::move(result);
}

inline calculation::m128matrix<4,4> transpose(const calculation::m128matrix<4,4>& matrix) {
    calculation::m128matrix<4,4> result;

    __m128 _Tmp3, _Tmp2, _Tmp1, _Tmp0;
    _Tmp0 = _mm_shuffle_ps(matrix.at<__m128>(0,0), matrix.at<__m128>(1,0), 0x44);
    _Tmp2 = _mm_shuffle_ps(matrix.at<__m128>(0,0), matrix.at<__m128>(1,0), 0xEE);
    _Tmp1 = _mm_shuffle_ps(matrix.at<__m128>(2,0), matrix.at<__m128>(3,0), 0x44);
    _Tmp3 = _mm_shuffle_ps(matrix.at<__m128>(2,0), matrix.at<__m128>(3,0), 0xEE);

    result.at<__m128>(0,0) = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88);
    result.at<__m128>(1,0) = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
    result.at<__m128>(2,0) = _mm_shuffle_ps(_Tmp2, _Tmp3, 0x88);
    result.at<__m128>(3,0) = _mm_shuffle_ps(_Tmp2, _Tmp3, 0xDD);

    return std::move(result);
}

inline calculation::m128matrix<4,4> inverse(const calculation::m128matrix<4,4>& matrix) {
    auto result = inverse(reinterpret_cast<const calculation::matrix<float,4,4>&>(matrix));
    return reinterpret_cast<const calculation::m128matrix<4,4>&>(result);
}

#endif // _INCLUDED_MM2



//	//template<typename T> inline
//	//vector3<T> transform_coord(_in(vector3<T>) _Lhs, _in(matrix_<4, 4, T>) _matrix)
//	//	{	// _matrix * Vector[_Lhs, 1.0]
//	//	vector4<T> _Result  = _matrix.row(3);
//	//			   _Result += _matrix.row(2) * _Lhs[2];
//	//			   _Result += _matrix.row(1) * _Lhs[1];
//	//			   _Result += _matrix.row(0) * _Lhs[0];
//
//	//	if ( _Result[3] != T(1) ) 
//	//		{	/*  _Result[3]   = X
//	//				_Result[3]   = X*1
//	//				_Result[3]/X = 1
//	//			*/
//	//		_Result /= _Result[3];
//	//		}
//
//	//	return ( reference_cast<vector3<T>>(_Result) );
//	//	}
//
//	//template<typename T> inline 
//	//unit_vector<T> transform_normal(_in(unit_vector<T>) _Lhs, _in(matrix_<4, 4, T>) _matrix)
//	//	{	// mat3x3(_matrix) * Vec3, igonore translate
//	//	vector4<T> _Result  = _Lhs[0] * _matrix.row(0);
//	//			   _Result += _Lhs[1] * _matrix.row(1);
//	//			   _Result += _Lhs[2] * _matrix.row(2);
//	//	return ( reference_cast<unit_vector<T>>(_Result) );
//	//	}
//
//	//inline vector3<float> screen_to_world(_in(vector3<float>) _Vec, _in(mat4) _Viewmatrix, _in(mat4) _Projmatrix)
//	//	{
//	//	auto _Vec128 = DirectX::XMLoadFloat3(reinterpret_cast<const DirectX::XMFLOAT3*>(&_Vec));
//	//	auto _mat128view = DirectX::XMLoadFloat4x4(reinterpret_cast<const DirectX::XMFLOAT4X4*>(&_Viewmatrix));
//	//	auto _mat128proj = DirectX::XMLoadFloat4x4(reinterpret_cast<const DirectX::XMFLOAT4X4*>(&_Projmatrix));
//	//	_Vec128.m128_f32[3] = 1.0f;
//	//	_mat128view = DirectX::XMMatrixInverse(nullptr, _mat128view);
//	//	_mat128proj = DirectX::XMMatrixInverse(nullptr, _mat128proj);
//
//	//	_Vec128 = DirectX::XMVector3TransformCoord(_Vec128, _mat128proj);
//	//	_Vec128 = DirectX::XMVector3TransformCoord(_Vec128, _mat128view);
//	//	_Vec128 = DirectX::XMVector4Normalize(_Vec128);
//
//	//	return (*reinterpret_cast<vector3<float>*>(&_Vec128));
//	//	}
//
//	///* < gen_mat > */
//	//template<typename T> inline 
//	//	matrix_<4, 4, T> scaling_matrix(_in(T) _Width, _in(T) _Height, _in(T) _Depth)
//	//	{
//	//	return ( matrix_<4, 4, T>(
//	//		_Width,    T(0),   T(0), T(0),
//	//		  T(0), _Height,   T(0), T(0),
//	//		  T(0),    T(0), _Depth, T(0),
//	//		  T(0),    T(0),   T(0), T(1)) );
//	//	}
//
//	//template<typename T> inline 
//	//	matrix_<4, 4, T> scaling_matrix(_in(vector3<T>) _sXyz) {
//	//		return ( scaling_matrix(_sXyz[0], _sXyz[1], _sXyz[2]) );
//	//	}
//
//	//template<typename T> inline
//	//matrix_<4, 4, T> rotation_matrix(_in(unit_vector<T>) _Axis, _in(T) _Radians)
//	//	{
//	//	using namespace::DirectX;
//	//	auto _A = _mm_loadu_ps(_Axis.ptr());
//	//	return (*(mat4*)(&DirectX::XMMatrixRotationAxis(_A, _Radians)));
//	//	}
//
//	//template<typename T> inline
//	//matrix_<4, 4, T> rotation_matrix(const T Ax, const T Ay, const T Az, const T rad) {
//	//	return rotation_matrix(unit_vector<T>(Ax, Ay, Az), rad);
//	//}
//
//	//template<typename T> inline
//	//	matrix_<4, 4, T> translation_matrix( _in(T) _X, _in(T) _Y, _in(T) _Z) {
//	//		return ( matrix_<4, 4, T>(
//	//			T(1), T(0), T(0), T(0),
//	//			T(0), T(1), T(0), T(0),
//	//			T(0), T(0), T(1), T(0),
//	//			  _X,   _Y,   _Z, T(1)) );
//	//	}
//
//	//template<typename T> inline
//	//	matrix_<4, 4, T> translation_matrix( _in(vector3<T>) _dXyz) {
//	//		return (translation_matrix(_dXyz[0], _dXyz[1], _dXyz[2]));
//	//	}
//
//}// namespace clmagic
//



#endif