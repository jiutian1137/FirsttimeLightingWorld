//+------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache License 2.0
// Look forward to your valuable comments(Any contact method)
// ------------------------------------------------------------------------------------+
#pragma once
// base idea
#include <cassert>
#include <array>
#include <numeric>
#include <algorithm>
#include <functional>
#include <type_traits>
// extend
#include <intrin.h>

namespace calculation {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////// BLOCK TRAITS /// ///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename _Ty>
	struct block_traits {// default smallest_block
		using block_type  = _Ty;
		using scalar_type = _Ty;

		static inline size_t size() {
			return 1;
		}
		static inline block_type set(const scalar_type& _Val) {
			return _Val;
		}
		static inline block_type load(const scalar_type* _Ptr) {
			return *_Ptr;
		}
		static inline void store(scalar_type* _Dest, const block_type& _Source) {
			*_Dest = _Source;
		}

		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, block_type& _Destination) {
			_Destination = *_First++;
			return _First;
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const block_type& _Source) {
			*_Dest++ = _Source;
			return _Dest;
		}
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, block_type& _Destination, size_t _Count) {
			_Destination = *_First++;
			return _First;
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const block_type& _Source, size_t _Count) {
			return store(_Dest, _Source);
		}

		using const_iterator = const scalar_type*;
		using iterator       = scalar_type*;

		static inline const_iterator begin(const block_type& _Source) {
			return &_Source;
		}
		static inline const_iterator end(const block_type& _Source) {
			return (&_Source) + 1;
		}
		static inline iterator begin(block_type& _Source) {
			return &_Source;
		}
		static inline iterator end(block_type& _Source) {
			return (&_Source) + 1;
		}
	};

	template<class _InIt, class _OutIt>
	_InIt copy_return_first(_InIt _First, _InIt _Last, _OutIt _Dest) {
		// copy [_First, _Last) to [_Dest, ...), arbitrary iterators
		for ( ; _First != _Last; ++_First, ++_Dest) {
			*_Dest = *_First;
		}

		return _First;
	}

#ifdef _INCLUDED_MM2
	template<>
	struct block_traits<__m128> {
		using scalar_type = float;
		using block_type  = __m128;
		
		static inline size_t size() {
			return 4;
		}
		static inline __m128 set(float _Val) {
			return _mm_set1_ps(_Val);
		}        
		static inline __m128 load(const float* _Ptr) {
			return _mm_load_ps(_Ptr);
		}
		static inline void store(float* _Dest, __m128 _Source) {
			_mm_store_ps(_Dest, _Source);
		}
		
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, __m128& _Destination) {
			return copy_return_first(_First, std::next(_First, size()), _Destination.m128_f32);
		}
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, __m128& _Destination, size_t _Count) {
			assert( _Count <= size() );
			return copy_return_first(_First, std::next(_First, _Count), _Destination.m128_f32);
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const __m128& _Source) {
			return std::copy(_Source.m128_f32, _Source.m128_f32 + size(), _Dest);
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const __m128& _Source, size_t _Count) {
			assert( _Count <= size() );
			return std::copy(_Source.m128_f32, _Source.m128_f32 + _Count, _Dest);
		}
		
		using const_iterator = const float*;
		using iterator       = float*;
		
		static inline const_iterator begin(const __m128& _Source) {
			return _Source.m128_f32;
		}
		static inline const_iterator end(const __m128& _Source) {
			return _Source.m128_f32 + size();
		}
		static inline iterator begin(__m128& _Source) {
			return _Source.m128_f32;
		}
		static inline iterator end(__m128& _Source) {
			return _Source.m128_f32 + size();
		}
	};
#endif

#ifdef _INCLUDED_EMM
	template<>
	struct block_traits<__m128d> {
		using scalar_type = double;
		using block_type  = __m128d;
		
		static inline size_t size() {
			return 2;
		}
		static inline __m128d set(double _Val) {
			return _mm_set1_pd(_Val);
		}        
		static inline __m128d load(const double* _Ptr) {
			return _mm_load_pd(_Ptr);
		}
		static inline void store(double* _Dest, __m128d _Source) {
			_mm_store_pd(_Dest, _Source);
		}
		
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, __m128d& _Destination) {
			return copy_return_first(_First, std::next(_First, size()), _Destination.m128d_f64);
		}
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, __m128d& _Destination, size_t _Count) {
			assert( _Count <= size() );
			return copy_return_first(_First, std::next(_First, _Count), _Destination.m128d_f64);
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const __m128d& _Source) {
			return std::copy(_Source.m128d_f64, _Source.m128d_f64 + size(), _Dest);
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const __m128d& _Source, size_t _Count) {
			assert( _Count <= size() );
			return std::copy(_Source.m128d_f64, _Source.m128d_f64 + _Count, _Dest);
		}
		
		using const_iterator = const double*;
		using iterator       = double*;
		
		static inline const_iterator begin(const __m128d& _Source) {
			return _Source.m128d_f64;
		}
		static inline const_iterator end(const __m128d& _Source) {
			return _Source.m128d_f64 + size();
		}
		static inline iterator begin(__m128d& _Source) {
			return _Source.m128d_f64;
		}
		static inline iterator end(__m128d& _Source) {
			return _Source.m128d_f64 + size();
		}
	};

	template<>
	struct block_traits<__m128i> {
		using scalar_type = int;
		using block_type  = __m128i;
		
		static inline size_t size() {
			return 4;
		}
		static inline __m128i set(int _Val) {
			return _mm_set1_epi32(_Val);
		}        
		static inline __m128i load(const int* _Ptr) {
			return _mm_loadu_epi32(_Ptr);
		}
		static inline void store(int* _Dest, __m128i _Source) {
			_mm_storeu_epi32(_Dest, _Source);
		}
		
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, __m128i& _Destination) {
			return copy_return_first(_First, std::next(_First, size()), _Destination.m128i_i32);
		}
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, __m128i& _Destination, size_t _Count) {
			assert( _Count <= size() );
			return copy_return_first(_First, std::next(_First, _Count), _Destination.m128i_i32);
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const __m128i& _Source) {
			return std::copy(_Source.m128i_i32, _Source.m128i_i32 + size(), _Dest);
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const __m128i& _Source, size_t _Count) {
			assert( _Count <= size() );
			return std::copy(_Source.m128i_i32, _Source.m128i_i32 + _Count, _Dest);
		}
		
		using const_iterator = const int*;
		using iterator       = int*;
		
		static inline const_iterator begin(const __m128i& _Source) {
			return _Source.m128i_i32;
		}
		static inline const_iterator end(const __m128i& _Source) {
			return _Source.m128i_i32 + size();
		}
		static inline iterator begin(__m128i& _Source) {
			return _Source.m128i_i32;
		}
		static inline iterator end(__m128i& _Source) {
			return _Source.m128i_i32 + size();
		}
	};
#endif

#ifdef _INCLUDED_IMM
	template<>
	struct block_traits<__m256> {
		using scalar_type = float;
		using block_type  = __m256;
		
		static inline size_t size() {
			return 8;
		}
		static inline __m256 set(float _Val) {
			return _mm256_set1_ps(_Val);
		}        
		static inline __m256 load(const float* _Ptr) {
			return _mm256_load_ps(_Ptr);
		}
		static inline void store(float* _Dest, __m256 _Source) {
			_mm256_store_ps(_Dest, _Source);
		}
		
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, __m256& _Destination) {
			return copy_return_first(_First, std::next(_First, size()), _Destination.m256_f32);
		}
		template<typename _InIt>
		static inline _InIt load_any(_InIt _First, __m256& _Destination, size_t _Count) {
			assert( _Count <= size() );
			return copy_return_first(_First, std::next(_First, _Count), _Destination.m256_f32);
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const __m256& _Source) {
			return std::copy(_Source.m256_f32, _Source.m256_f32 + size(), _Dest);
		}
		template<typename _OutIt>
		static inline _OutIt store_any(_OutIt _Dest, const __m256& _Source, size_t _Count) {
			assert( _Count <= size() );
			return std::copy(_Source.m256_f32, _Source.m256_f32 + _Count, _Dest);
		}
		
		using const_iterator = const float*;
		using iterator       = float*;
		
		static inline const_iterator begin(const __m256& _Source) {
			return _Source.m256_f32;
		}
		static inline const_iterator end(const __m256& _Source) {
			return _Source.m256_f32 + size();
		}
		static inline iterator begin(__m256& _Source) {
			return _Source.m256_f32;
		}
		static inline iterator end(__m256& _Source) {
			return _Source.m256_f32 + size();
		}
	};
#endif

	template <bool _First_value, typename  _Left, typename _Right, class... _Rest>
	struct _Is_same_block { // handle false trait or last pair trait
		static constexpr bool value = std::is_same_v<typename _Left::block_type, typename _Right::block_type>;
	};

	template <typename  _Left, typename _Right, typename _Next, class... _Rest>
	struct _Is_same_block<true, _Left, _Right, _Next, _Rest...> { // the first trait is true, try the next pair
		static constexpr bool value = _Is_same_block<std::is_same_v<typename _Right::block_type, typename _Next::block_type>, _Right, _Next, _Rest...>::value;
	};

	template<typename _Ty> constexpr
	bool is_smallest_block_v = std::is_same_v<typename block_traits<_Ty>::block_type, typename block_traits<_Ty>::scalar_type>;

	template<typename _VecTy1, typename _VecTy2, typename... _VecTys> constexpr
	bool is_same_block_v = _Is_same_block<std::is_same_v<typename _VecTy1::block_type, typename _VecTy2::block_type>, _VecTy1, _VecTy2, _VecTys...>::value;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////// BLOCK TRAITS /// ///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	using std::alignment_of_v;

	template<typename _Ty> constexpr
	size_t alignment_mask_of_v = std::alignment_of_v<_Ty> - 1;

	template<size_t _Val, size_t _Bound>
	struct alignment_ceil {
		static constexpr size_t value = (_Val + (_Bound - 1)) & (~(_Bound - 1));
	};

	template<size_t _Val, size_t _Bound> constexpr
	size_t alignment_ceil_v = alignment_ceil<_Val, _Bound>::value;

	template<typename _Sclty, size_t _SclSize, typename _Alignty>
	struct static_alignment_traits {
		constexpr static size_t aligned_bound   = std::alignment_of_v<_Alignty>;
		
		constexpr static size_t unaligned_bytes = sizeof(_Sclty) * _SclSize;
		constexpr static size_t unaligned_size  = _SclSize;

		constexpr static size_t aligned_bytes   = alignment_ceil_v<unaligned_bytes, aligned_bound>;
		constexpr static size_t aligned_size    = aligned_bytes / sizeof(_Sclty);
		// { aligned_size = total_size, aligned_size >= unaligned_size }

		constexpr static size_t valid_size   = unaligned_size;
		constexpr static size_t invalid_size = aligned_size - unaligned_size;
	};



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////// VECTOR SIZE///// ///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*<example>
		using scalar_type = float; 
		using block_type  = __m128;

		scalar_pointer FIRST          = ?;
		scalar_pointer LAST           = ?;
		size_t         SCLBYTES       = sizeof(scalar_type);
		size_t         ALIGNMENT_SIZE = sizeof(block_type) / sizeof(scalar_type);// How many scalars the block_type contained?
		size_t         ALIGNMENT_MASK = clmagic::alignment_mask_of_v<block_type>;
	</example>*/

#define _Check_vector_alignment_begin(FIRST, LAST, SCLBYTES, ALIGNMENT_SIZE, ALIGNMENT_MASK) \
	/* 1. check empty  */                                                                    \
	assert( ##FIRST## != ##LAST## );                                                         \
	size_t _Scl_size   = ##LAST## - ##FIRST##;/*unit:[scalars]*/                             \
	size_t _Blk_offset = reinterpret_cast<size_t>(FIRST) & ##ALIGNMENT_MASK##;/*unit:[bytes]*/ \
	/* 2. check memory_error */                                                              \
	assert( _Blk_offset % ##SCLBYTES## == 0 );                                               \
	_Blk_offset /= ##SCLBYTES##;/*unit:[scalars]*/                                           \
	/* 3. check through_alignement_point */                                                  \
	if ( (_Blk_offset + _Scl_size) >= ##ALIGNMENT_SIZE## ) {
#define _Check_vector_alignment_end }

#define _Check_vector_alignment2_begin(FIRST1, LAST1, FIRST2, SCLBYTES, ALIGNMENT_SIZE, ALIGNMENT_MASK) \
	/* 1. check empty */                                                                                \
	assert( ##FIRST1## != ##LAST1## );                                                                  \
	size_t _Scl_size    = ##LAST1## - ##FIRST1##;/*unit:[scalars]*/                                     \
	size_t _Blk_offset  = reinterpret_cast<size_t>(##FIRST1##) & ##ALIGNMENT_MASK##;/*unit:[bytes]*/    \
	size_t _Blk_offset2 = reinterpret_cast<size_t>(##FIRST2##) & ##ALIGNMENT_MASK##;/*unit:[bytes]*/    \
	/* 2. check memory similar */                                                                       \
	if ( _Blk_offset == _Blk_offset2 ) {                                                                \
		/* 3. check memory_error */                                                                     \
		assert( _Blk_offset % ##SCLBYTES## == 0 );                                                      \
		_Blk_offset /= ##SCLBYTES##;/*unit:[scalars]*/                                                  \
		/* 4. check through_alignement_point */                                                         \
		if ( (_Blk_offset + _Scl_size) >= ##ALIGNMENT_SIZE## ) {
#define _Check_vector_alignment2_end }}

#define _Check_vector_lead_size_begin(ALIGNMENT_SIZE)                             \
	size_t lead_size = (_Blk_offset == 0) ? 0 : ##ALIGNMENT_SIZE## - _Blk_offset; \
	if (lead_size != 0) {
#define _Check_vector_lead_size_end _Scl_size -= lead_size; } 

#define _Check_vector_block_size_begin(ALIGNMENT_SIZE)  \
	size_t block_size = _Scl_size / ##ALIGNMENT_SIZE##; \
	if (block_size != 0) {
#define _Check_vector_block_size_end }

#define _Check_vector_tail_size_begin(ALIGNMENT_SIZE)  \
	size_t tail_size = _Scl_size % ##ALIGNMENT_SIZE##; \
	if (tail_size != 0) {
#define _Check_vector_tail_size_end }

	// { vector_size = lead_size + block_size*block_traits<_BlkTy>::size() + tail_size }
	template<typename _Sty, typename _Bty>
	struct vector_size {
		using scalar_type = _Sty;
		using block_type  = _Bty;

		vector_size() = default;
		vector_size(const _Sty* _First, const _Sty* _Last) {
			_Check_vector_alignment_begin( _First, _Last, sizeof(_Sty), block_traits<_Bty>::size(), alignment_mask_v<_Bty> )
				this->lead_size  = (_Blk_offset == 0) ? 0 : block_traits<_Bty>::size() - _Blk_offset;
				_Scl_size       -= lead_size;
				this->block_size = _Scl_size / block_traits<_Bty>::size();
				this->tail_size  = _Scl_size % block_traits<_Bty>::size();
			_Check_vector_alignment_end
		}
		vector_size(const _Sty* _First1, const _Sty* _Last1, const _Sty* _First2) {
			_Check_vector_alignment2_begin( _First1, _Last1, _First2, sizeof(_Sty), block_traits<_Bty>::size(), alignment_mask_v<_Bty> )
				this->lead_size  = (_Blk_offset == 0) ? 0 : block_traits<_Bty>::size() - _Blk_offset;
				_Scl_size       -= lead_size;
				this->block_size = _Scl_size / block_traits<_Bty>::size();
				this->tail_size  = _Scl_size % block_traits<_Bty>::size();
			_Check_vector_alignment2_end
		}

		bool empty() const {
			return (block_size == 0 && lead_size == 0 && tail_size == 0);
		}
		bool can_fastest() const {
			return (lead_size == 0 && tail_size == 0);
		}

		bool operator==(const vector_size& _Right) const {
			return (this->lead_size  == _Right.lead_size &&
				    this->block_size == _Right.block_size &&
					this->tail_size  == _Right.tail_size);
		}
		bool operator!=(const vector_size& _Right) const {
			return !(*this == _Right);
		}

		size_t lead_size  = 0;
		size_t block_size = 0;
		size_t tail_size  = 0;
	};



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////// VECTOR OPERATION ///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename _Ty>
	class _Any_array_const_iterator {
	public:
		using iterator_category = std::random_access_iterator_tag;
		using value_type        = _Ty;
		using difference_type   = ptrdiff_t;
		using pointer           = const _Ty*;
		using reference         = const _Ty&;

		_Any_array_const_iterator()
			: array_ptr(nullptr), stride(1) {}
		_Any_array_const_iterator(pointer _Ptr, ptrdiff_t _Stride = 1)
			: array_ptr(_Ptr), stride(_Stride) {}
		_Any_array_const_iterator(pointer _Parray, ptrdiff_t _Off, ptrdiff_t _Index, ptrdiff_t _Stride = 1)
			: array_ptr(_Parray + _Off + _Index * _Stride), stride(_Stride) {}

		reference operator*() const {
			return *array_ptr;
		}
		pointer operator&() const {
			return array_ptr;
		}

		_Any_array_const_iterator operator+(ptrdiff_t _Diff) const {
			return _Any_array_const_iterator(array_ptr + _Diff * stride, stride);
		}
		_Any_array_const_iterator operator-(ptrdiff_t _Diff) const {
			return (*this) + (-_Diff);
		}

		_Any_array_const_iterator& operator+=(ptrdiff_t _Diff) {
			(*this) = (*this) + _Diff;
			return *this;
		}
		_Any_array_const_iterator& operator-=(ptrdiff_t _Diff) {
			(*this) = (*this) - _Diff;
			return *this;
		}

		_Any_array_const_iterator& operator++() {
			array_ptr += stride;
			return *this;
		}
		_Any_array_const_iterator& operator--() {
			array_ptr -= stride;
			return *this;
		}

		_Any_array_const_iterator operator++(int) {
			_Any_array_const_iterator _Tmp = (*this);
			++(*this);
			return _Tmp;
		}
		_Any_array_const_iterator operator--(int) {
			_Any_array_const_iterator _Tmp = (*this);
			--(*this);
			return _Tmp;
		}

		bool operator==(const _Any_array_const_iterator& _Right) const {
			assert(stride == _Right.stride);
			return array_ptr == _Right.array_ptr;
		}
		bool operator!=(const _Any_array_const_iterator& _Right) const {
			assert(stride == _Right.stride);
			return array_ptr != _Right.array_ptr;
		}
		bool continuous() const { return stride == 1; }

		difference_type operator-(const _Any_array_const_iterator& _Right) const {
			assert(stride == _Right.stride);
			return (array_ptr - _Right.array_ptr) / stride;
		}

		// N-element = array_ptr + _Myoff + _Myidx*stride
		pointer array_ptr;
		//ptrdiff_t _Myoff;
		//ptrdiff_t _Myidx;
		ptrdiff_t stride;
	};

	template<typename _Ty>
	class _Any_array_iterator : public _Any_array_const_iterator<_Ty> {
		using _Mybase = _Any_array_const_iterator<_Ty>;
	public:
		using iterator_category = std::random_access_iterator_tag;
		using value_type        = _Ty;
		using difference_type   = ptrdiff_t;
		using pointer           = _Ty*;
		using reference         = _Ty&;

		_Any_array_iterator() : _Mybase() {}
		_Any_array_iterator(pointer _Ptr, ptrdiff_t _Stride = 1)
			: _Mybase(_Ptr, _Stride) {}
		_Any_array_iterator(pointer _Parray, ptrdiff_t _Off, ptrdiff_t _Index, ptrdiff_t _Stride = 1)
			: _Mybase(_Parray, _Off, _Index, _Stride) {}

		reference operator*() const {
			return const_cast<reference>(_Mybase::operator*());
		}
		pointer operator&() const {
			return const_cast<pointer>(_Mybase::operator&());
		}

		_Any_array_iterator operator+(ptrdiff_t _Diff) const {
			return reinterpret_cast<const _Any_array_iterator&>(_Mybase::operator+(_Diff));
		}
		_Any_array_iterator operator-(ptrdiff_t _Diff) const {
			return (*this) + (-_Diff);
		}

		_Any_array_iterator& operator+=(ptrdiff_t _Diff) {
			(*this) = (*this) + _Diff;
			return *this;
		}
		_Any_array_iterator& operator-=(ptrdiff_t _Diff) {
			(*this) = (*this) - _Diff;
			return *this;
		}

		_Any_array_iterator& operator++() {
			_Mybase::operator++();
			return *this;
		}
		_Any_array_iterator& operator--() {
			_Mybase::operator--();
			return *this;
		}

		_Any_array_iterator operator++(int) {
			_Any_array_iterator _Tmp = (*this);
			++(*this);
			return _Tmp;
		}
		_Any_array_iterator operator--(int) {
			_Any_array_iterator _Tmp = (*this);
			--(*this);
			return _Tmp;
		}
	};

	// { fastest_vector_operation }
	template<typename _BlkTy, typename _Traits = block_traits<_BlkTy>>
	struct aligned_vector_operation {
		using traits_type = _Traits;
		using scalar_type = typename _Traits::scalar_type;
		using block_type  = typename _Traits::block_type;
		
		template<typename _InIt, typename _OutIt, typename _UnaryOp>
		static _OutIt transform(_InIt _First, _InIt _Last, _OutIt _Dest, _UnaryOp _Transform_op) {
			return std::transform(_First, _Last, _Dest, _Transform_op);
		}

		template<typename _InIt1, typename _InIt2, typename _OutIt, typename _BinOp>
		static _OutIt transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _BinOp _Transform_op) {
			return std::transform(_First1, _Last1, _First2, _Dest, _Transform_op);
		}

		template<typename _InIt, typename _BinOp1, typename _BinOp2>
		static scalar_type reduce(_InIt _First, _InIt _Last, scalar_type _Val, _BinOp1 _Reduceblock_op, _BinOp2 _Reducescalar_op) {
			assert( _First != _Last );
			block_type _Result =  std::reduce(std::next(_First), _Last, (*_First), _Reduceblock_op);
			return std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Reducescalar_op);
		}

		template<typename _InIt, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
		static scalar_type transform_reduce(_InIt _First, _InIt _Last, scalar_type _Val, _BinOp1 _Blkreduce_op, _BinOp2 _Sclreduce_op, _UnaryOp _Transform_op) {
			assert( _First != _Last );
			block_type _Result = std::transform_reduce(std::next(_First), _Last, _Transform_op(*_First), _Blkreduce_op, _Transform_op);
			return std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Sclreduce_op);
		}

		template<typename _InIt1, typename _InIt2, typename _BinOp1, typename _BinOp2, typename _BinOp3>
		static scalar_type transform_reduce(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, scalar_type _Val, _BinOp1 _Blkreduce_op, _BinOp2 _Sclreduce_op, _BinOp3 _Transform_op) {
			assert( _First1 != _Last1 );
			block_type _Result = std::transform_reduce(std::next(_First1), _Last1, std::next(_First2), _Transform_op(*_First1, *_First2), _Blkreduce_op, _Transform_op);
			return std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Sclreduce_op);
		}
	};
	
	// { norm_vector_operation }
	template<typename _BlkTy, typename _Traits = block_traits<_BlkTy>>
	struct discontinuous_vector_operation {
		using traits_type = _Traits;
		using scalar_type = typename _Traits::scalar_type;
		using block_type  = typename _Traits::block_type;

		template<typename _InIt, typename _OutIt, typename _UnaryOp>
		static _OutIt transform_aligned_size(_InIt _First, _InIt _Last, _OutIt _Dest, _UnaryOp _Transform_op) {
			block_type _Left;
			for ( ; _First != _Last; ) {
				_First = traits_type::load_any(_First, _Left);
				_Dest  = traits_type::store_any(_Dest, _Transform_op(_Left));
			}

			return _Dest;
		}

		template<typename _InIt, typename _OutIt, typename _UnaryOp>
		static _OutIt transform_unaligned_size(_InIt _First, _InIt _Mid, _InIt _Last, _OutIt _Dest, _UnaryOp _Transform_op, size_t _Tailsize) {
			block_type _Left;
			for ( ; _First != _Mid; ) {
				_First = traits_type::load_any(_First, _Left);
				_Left  = _Transform_op( _Left );
				_Dest  = traits_type::store_any(_Dest, _Left);
			}

			traits_type::load_any(_First, _Left, _Tailsize);
			_Left = _Transform_op( _Left );
			return traits_type::store_any(_Dest, _Left, _Tailsize);
		}

		template<typename _InIt, typename _OutIt, typename _UnaryOp> 
		static _OutIt transform(_InIt _First, _InIt _Last, _OutIt _Dest, _UnaryOp _Transform_op) {
			const size_t _Size = std::distance(_First, _Last);
			const size_t _Tailsize = _Size % traits_type::size();
			if (_Tailsize == 0) {
				return transform_aligned_size(_First, _Last, _Dest, _Transform_op);
			} else {
				const auto _Mid = std::next(_First, _Size - _Tailsize);
				return transform_unaligned_size(_First, _Mid, _Last, _Dest, _Transform_op, _Tailsize);
			}
		}


		template<typename _InIt1, typename _InIt2, typename _OutIt, typename _BinOp>
		static _OutIt transform_aligned_size(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _BinOp _Transform_op) {
			block_type _Left, _Right;
			for ( ; _First1 != _Last1; ) {
				_First1 = traits_type::load_any(_First1, _Left);
				_First2 = traits_type::load_any(_First2, _Right);
				_Left   = _Transform_op(_Left, _Right);
				_Dest   = traits_type::store_any(_Dest, _Left);
			}

			return _Dest;
		}

		template<typename _InIt1, typename _InIt2, typename _OutIt, typename _BinOp>
		static _OutIt transform_unaligned_size(_InIt1 _First1, _InIt1 _Mid1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _BinOp _Transform_op, size_t _Tailsize) {
			_BlkTy _Left, _Right;
			for ( ; _First1 != _Mid1; ) {
				_First1 = traits_type::load_any(_First1, _Left);
				_First2 = traits_type::load_any(_First2, _Right);
				_Left   = _Transform_op(_Left, _Right);
				_Dest   = traits_type::store_any(_Dest, _Left);
			}

			traits_type::load_any(_First1, _Left, _Tailsize);
			traits_type::load_any(_First2, _Right, _Tailsize);
			_Left = _Transform_op(_Left, _Right);
			return traits_type::store_any(_Dest, _Left, _Tailsize);
		}

		template<typename _InIt1, typename _InIt2, typename _OutIt, typename _BinOp>
		static _OutIt transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _BinOp _Transform_op) {
			const size_t _Diff = std::distance(_First1, _Last1);
			const size_t _Tailsize = _Diff % block_traits<_BlkTy>::size();
			if (_Tailsize == 0) {
				return transform_aligned_size(_First1, _Last1, _First2, _Dest, _Transform_op);
			} else {
				const auto _Mid1 = std::next(_First1, _Diff-_Tailsize);
				return transform_unaligned_size(_First1, _Mid1, _Last1, _First2, _Dest, _Transform_op, _Tailsize);
			}
		}


		template<typename _InIt, typename _BinOp1, typename _BinOp2>
		static scalar_type reduce_aligned_size(_InIt _First, _InIt _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op) {
			// 1. Init _Result
			block_type _Left, _Result;
			_First = traits_type::load_any(_First, _Result);
			
			// 2. block-reduce [_First, _Last) to _Result
			for ( ; _First != _Last; ) {
				_First  = traits_type::load_any(_First, _Left);
				_Result = _Reduce_block_op( std::move(_Result), std::move(_Left) );
			}
			
			// 3. scalar-reduce _Result
			return std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _InIt, typename _BinOp1, typename _BinOp2>
		static scalar_type reduce_unaligned_size(_InIt _First, _InIt _Mid, _InIt _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op) {
			if ( _First != _Mid ) {
				_Val = reduce_aligned_size(_First, _Mid, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op);
			}

			// 4. scalar-reduce _Val and [_Mid, _Last)
			return std::reduce(_Mid, _Last, std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _InIt, typename _BinOp1, typename _BinOp2>
		static scalar_type reduce(_InIt _First, _InIt _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op) {
			const size_t _Size     = std::distance(_First, _Last);
			const size_t _Tailsize = _Size % block_traits<_BlkTy>::size();
			if (_Tailsize == 0) {
				return reduce_aligned_size(_First, _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op);
			} else {
				const _InIt _Mid = std::next(_First, _Size - _Tailsize);
				return reduce_unaligned_size(_First, _Mid, _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op);
			}
		}


		template<typename _InIt, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
		static scalar_type transform_reduce_aligned_size(_InIt _First, _InIt _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _UnaryOp _Transform_op) {
			// 1. Init _Result
			block_type _Left, _Result;
			_First  = traits_type::load_any(_First, _Left);
			_Result = _Transform_op( std::move(_Left) );

			// 2. block-transform-reduce [_First, _Last) to _Result
			for ( ; _First != _Last; ) {
				_First  = traits_type::load_any(_First, _Left);
				_Result = _Reduce_block_op(std::move(_Result), _Transform_op( std::move(_Left) ));
			}

			// 3. scalar-reduce _Result
			return std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _InIt, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
		static scalar_type transform_reduce_unaligned_size(_InIt _First, _InIt _Mid, _InIt _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _UnaryOp _Transform_op, size_t _Tailsize) {
			if (_First != _Mid) {
				_Val = transform_reduce_aligned_size(_First, _Last, std::move(_Val), 
					_Reduce_block_op, _Reduce_scalar_op, _Transform_op);
			}

			block_type _Tailblock;
			traits_type::load_any(_Mid, _Tailblock, _Tailsize);
			_Tailblock = _Transform_op( std::move(_Tailblock) );
			return std::reduce(traits_type::begin(_Tailblock), traits_type::begin(_Tailblock) + _Tailsize, std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _InIt, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
		static scalar_type transform_reduce(_InIt _First, _InIt _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _UnaryOp _Transform_op) {
			const size_t _Size     = std::distance(_First, _Last);
			const size_t _Tailsize = _Size % block_traits<_BlkTy>::size();
			if (_Tailsize == 0) {
				return transform_reduce_aligned_size(_First, _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op);
			} else {
				const _InIt _Mid = std::next(_First, _Size-_Tailsize);
				return transform_reduce_unaligned_size(_First, _Mid, _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op, _Tailsize);
			}
		}


		template<typename _InIt1, typename _InIt2, typename _BinOp1, typename _BinOp2, typename _BinOp3 >
		static scalar_type transform_reduce_aligned_size(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _BinOp3 _Transform_op) {
			// 1. Init _Result 
			block_type _Left, _Right, _Result;
			_First1 = traits_type::load_any(_First1, _Left);
			_First2 = traits_type::load_any(_First2, _Right);
			_Result = _Transform_op( std::move(_Left), std::move(_Right) );
			
			// 2. block-transform-reduce [_First1, _Last1) and [_First2, ...) to _Result
			for ( ; _First1 != _Last1; ) {
				_First1 = traits_type::load_any(_First1, _Left);
				_First2 = traits_type::load_any(_First2, _Right);
				_Result = _Reduce_block_op( std::move(_Result), _Transform_op( std::move(_Left), std::move(_Right) ) );
			}

			// 3. scalar-reduce _Result
			return std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _InIt1, typename _InIt2, typename _BinOp1, typename _BinOp2, typename _BinOp3 >
		static scalar_type transform_reduce_unaligned_size(_InIt1 _First1, _InIt1 _Mid1, _InIt1 _Last1, _InIt2 _First2, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _BinOp3 _Transform_op, size_t _Tailsize) {
			if ( _First1 != _Mid1 ) {
				// 1. Init _Result
				block_type _Left, _Right, _Result;
				_First1 = traits_type::load_any(_First1, _Left);
				_First2 = traits_type::load_any(_First2, _Right);
				_Result = _Transform_op( std::move(_Left), std::move(_Right) );

				// 2. block-transform-reduce [_First1, _Last1) and [_First2, ...) to _Result
				for ( ; _First1 != _Mid1; ) {
					_First1 = traits_type::load_any(_First1, _Left);
					_First2 = traits_type::load_any(_First2, _Right);
					_Result = _Reduce_block_op( std::move(_Result), _Transform_op( std::move(_Left), std::move(_Right) ) );
				}

				// 3. scalar-reduce _Result to _Val
				_Val = std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Reduce_scalar_op);
			}

			// 4. _Val + scalar-transform-reduce [_Mid1, _Last1) and [.., ..)
			block_type _Tailblock1, _Tailblock2;
			traits_type::load_any(_First1, _Tailblock1, _Tailsize);
			traits_type::load_any(_First2, _Tailblock2, _Tailsize);
			_Tailblock1 = _Transform_op( std::move(_Tailblock1), std::move(_Tailblock2) );
			return std::reduce(traits_type::begin(_Tailblock1), traits_type::begin(_Tailblock1) + _Tailsize, std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _InIt1, typename _InIt2, typename _BinOp1, typename _BinOp2, typename _BinOp3>
		static scalar_type transform_reduce(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _BinOp3 _Transform_op) {
			const size_t _Size     = std::distance(_First1, _Last1);
			const size_t _Tailsize = _Size % block_traits<_BlkTy>::size();
			if (_Tailsize == 0) {
				return transform_reduce_aligned_size(_First1, _Last1, _First2, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op);
			} else {
				const _InIt1 _Mid1 = std::next(_First1, _Size-_Tailsize);
				return transform_reduce_unaligned_size(_First1, _Mid1, _Last1, _First2, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op, _Tailsize);
			}
		}
	};

	// { fast_vector_operation }
	template<typename _BlkTy, typename _Traits = block_traits<_BlkTy>>
	struct continuous_vector_operation {
		using traits_type = _Traits;
		using scalar_type = typename _Traits::scalar_type;
		using block_type  = typename _Traits::block_type;
		
		using block_pointer        = block_type*;
		using block_const_pointer  = const block_type*;
		using scalar_pointer       = scalar_type*;
		using scalar_const_pointer = const scalar_type*;

		template<typename _UnaryOp> inline
		static scalar_pointer transform_lead(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_pointer _Dest, _UnaryOp _Transform_op, size_t _Leadsize) {
			assert( _Leadsize != 0 );
			const auto _Result = _Transform_op( * (reinterpret_cast<block_const_pointer>(_Last) - 1) );
			auto       _Rlast  = traits_type::end(_Result);
			return std::copy(_Rlast - _Leadsize, _Rlast, _Dest);
		}
		
		template<typename _UnaryOp> inline
		static scalar_pointer transform_block(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_pointer _Dest, _UnaryOp _Transform_op) {
			block_pointer _Udest = std::transform(reinterpret_cast<block_const_pointer>(_First), 
												  reinterpret_cast<block_const_pointer>(_Last),
												  reinterpret_cast<block_pointer>(_Dest), 
												 _Transform_op);
			return reinterpret_cast<scalar_pointer>(_Udest);
		}

		template<typename _UnaryOp>
		static scalar_pointer transform_block_and_tail(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_pointer _Dest, _UnaryOp _Transform_op, size_t _Tailsize) {
			//assert( _First    != _Last - _Tailsize );
			assert( _Tailsize != 0 );
			// transform block
			block_const_pointer       _Ufirst = reinterpret_cast<block_const_pointer>(_First);
			const block_const_pointer _Ulast  = reinterpret_cast<block_const_pointer>(_Last - _Tailsize);
			block_const_pointer       _Udest  = reinterpret_cast<block_pointer>(_Dest);
			for ( ; _Ufirst != _Ulast; ++_Ufirst, ++_Udest) {
				*_Udest = _Transform_op( *_Ufirst );
			}
			// transform tail
			const auto _Result = _Transform_op( *_Ufirst );
			auto       _Rfirst = block_type::begin(_Result);
			return std::copy(_Rfirst, _Rfirst + _Tailsize, reinterpret_cast<scalar_pointer>(_Udest));
		}

		template<typename _UnaryOp> inline
		static scalar_pointer transform_tail(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_pointer _Dest, _UnaryOp _Transform_op, size_t _Tailsize) {
			assert( _Tailsize != 0 );
			const auto _Result = _Transform_op( *reinterpret_cast<block_const_pointer>(_First) );
			auto       _Rfirst = block_type::begin(_Result);
			return std::copy(_Rfirst, std::next(_Rfirst, _Tailsize), _Dest);
		}

		template<typename _UnaryOp> inline
		static scalar_pointer transform(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_pointer _Dest, _UnaryOp _Transform_op) {
			const size_t _ALIGNMENT_SIZE = traits_type::size();
			_Check_vector_alignment_begin(_First, _Last, sizeof(scalar_type), _ALIGNMENT_SIZE, alignment_mask_of_v<block_type>)
				_Check_vector_lead_size_begin(_ALIGNMENT_SIZE)
					_Dest  = transform_lead(_First, _First+lead_size, _Dest, _Transform_op, lead_size);
					_First = _First + lead_size;
				_Check_vector_lead_size_end

				_Check_vector_tail_size_begin(_ALIGNMENT_SIZE)
					return transform_block_and_tail(_First, _Last, _Dest, _Transform_op, tail_size);
				_Check_vector_tail_size_end else{
					return transform_block(_First, _Last, _Dest, _Transform_op);
				}
			_Check_vector_alignment_end
			// otherwise
			return discontinuous_vector_operation<_BlkTy,_Traits>::transform(_First, _Last, _Dest, _Transform_op);
		}


		template<typename _BinOp> inline
		static scalar_pointer transform_lead(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_pointer _Dest, _BinOp _Transform_op, size_t _Leadsize) {
			assert( _Leadsize != 0 );
			const size_t _Offset = traits_type::size() - _Leadsize;
			const auto   _Result = _Transform_op( * reinterpret_cast<block_const_pointer>(_First1 - _Offset),
												  * reinterpret_cast<block_const_pointer>(_First2 - _Offset) );
			auto         _Rlast  = traits_type::end(_Result);
			return std::copy(std::prev(_Rlast, _Leadsize), _Rlast, _Dest);
		}
		
		template<typename _BinOp> inline
		static scalar_pointer transform_block(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_pointer _Dest, _BinOp _Transform_op) {
			//assert( _First1 != _Last1 );
			block_pointer _Udest = std::transform(reinterpret_cast<block_const_pointer>(_First1), 
										reinterpret_cast<block_const_pointer>(_Last1),
										reinterpret_cast<block_const_pointer>(_First2), 
										reinterpret_cast<block_pointer>(_Dest), 
										_Transform_op);
			return reinterpret_cast<scalar_pointer>(_Udest);
		}

		template<typename _BinOp> inline
		static scalar_pointer transform_tail(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_pointer _Dest, _BinOp _Transform_op, size_t _Tailsize) {
			assert(_Tailsize != 0);
			const auto _Result = _Transform_op( * reinterpret_cast<block_const_pointer>(_First1), * reinterpret_cast<block_const_pointer>(_First2) );
			auto       _Rfirst = traits_type::begin(_Result);
			return std::copy(_Rfirst, std::next(_Rfirst, _Tailsize), _Dest);
		}
		
		template<typename _BinOp> inline
		static scalar_pointer transform_block_and_tail(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_pointer _Dest, _BinOp _Transform_op, size_t _Tailsize) {
			//assert( _First1 != _Last1 - _Tailsize );
			assert( _Tailsize != 0 );
			// 1.
			auto       _Ufirst1 = reinterpret_cast<block_const_pointer>(_First1);
			const auto _Ulast1  = reinterpret_cast<block_const_pointer>(std::prev(_Last1, _Tailsize));
			auto       _Ufirst2 = reinterpret_cast<block_const_pointer>(_First2);
			auto       _Udest   = reinterpret_cast<block_pointer>(_Dest);
			for (; _Ufirst1 != _Ulast1; ++_Ufirst1, ++_Ufirst2, ++_Udest) {
				*_Udest = _Transform_op(*_Ufirst1, *_Ufirst2);
			}
			// 2.
			const auto  _Result = _Transform_op(*_Ufirst1, *_Ufirst2);
			auto        _Rfirst = traits_type::begin(_Result);
			return std::copy(_Rfirst, std::next(_Rfirst, _Tailsize), reinterpret_cast<scalar_pointer>(_Udest));
		}

		template<typename _BinOp> inline
		static scalar_pointer transform(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_pointer _Dest, _BinOp _Transform_op) {
			const size_t _ALIGNMENT_SIZE = traits_type::size();
			_Check_vector_alignment2_begin( _First1, _Last1, _First2, sizeof(scalar_type), _ALIGNMENT_SIZE, alignment_mask_of_v<block_type> )
				_Check_vector_lead_size_begin(_ALIGNMENT_SIZE)
					_Dest     = transform_lead(_First1, _Last1, _First2, _Dest, _Transform_op, lead_size);
					_First1   = _First1 + lead_size;
					_First2   = _First2 + lead_size;
				_Check_vector_lead_size_end

				_Check_vector_tail_size_begin(_ALIGNMENT_SIZE)
					return transform_block_and_tail(_First1, _Last1, _First2, _Dest, _Transform_op, tail_size);
				_Check_vector_tail_size_end else {
					return transform_block(_First1, _Last1, _First2, _Dest, _Transform_op);
				}
			_Check_vector_alignment2_end
			// otherwise
			return discontinuous_vector_operation<_BlkTy, _Traits>::transform(_First1, _Last1, _First2, _Dest, _Transform_op);
		}
		

		template<typename _BinOp> inline
		static scalar_type reduce_lead(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val, _BinOp _Reduce_scalar_op) {
			return std::reduce(_First, _Last, std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _BinOp1, typename _BinOp2> inline
		static scalar_type reduce_block(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val, _BinOp2 _Reduce_block_op, _BinOp1 _Reduce_scalar_op) {
			assert( _First != _Last );
			return aligned_vector_operation<_BlkTy, _Traits>::reduce(reinterpret_cast<block_const_pointer>(_First), 
																	 reinterpret_cast<block_const_pointer>(_Last), 
																	 std::move(_Val), 
																	_Reduce_block_op, _Reduce_scalar_op);
		}

		template<typename _BinOp> inline
		static scalar_type reduce_tail(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val, _BinOp _Reduce_scalar_op) {
			return std::reduce(_First, _Last, std::move(_Val), _Reduce_scalar_op);
		}
		
		template<typename _BinOp1, typename _BinOp2>
		static scalar_type reduce_block_and_tail(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val, _BinOp2 _Reduce_block_op, _BinOp1 _Reduce_scalar_op, size_t _Tailsize) {
			assert( _Tailsize != 0 );
			assert( _First != _Last-_Tailsize );
			// 1.
			block_const_pointer       _Ufirst = reinterpret_cast<block_const_pointer>(_First);
			const block_const_pointer _Ulast  = reinterpret_cast<block_const_pointer>(_Last - _Tailsize);
			block_const_pointer       _Result = *_Ufirst++;
			for ( ; _Ufirst != _Ulast; ++_Ufirst) {
				_Result = _Reduce_block_op(std::move(_Result), *_Ufirst);
			}
			_Val = std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Reduce_scalar_op);
			// 2.
			return std::reduce(reinterpret_cast<scalar_const_pointer>(_Ufirst), _Last, std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _BinOp1, typename _BinOp2>
		static scalar_type reduce(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op) {
			const size_t _ALIGNMENT_SIZE = traits_type::size();
			_Check_vector_alignment_begin(_First, _Last, sizeof(scalar_type), _ALIGNMENT_SIZE, alignment_mask_of_v<block_type>)
				_Check_vector_lead_size_begin(_ALIGNMENT_SIZE)
					_Val   = reduce_lead(_First, _First+lead_size, std::move(_Val), _Reduce_scalar_op);
					_First = _First + lead_size;
				_Check_vector_lead_size_end

				_Check_vector_block_size_begin(_ALIGNMENT_SIZE)
					_Check_vector_tail_size_begin(_ALIGNMENT_SIZE)
						return reduce_block_and_tail(_First, _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, tail_size);
					_Check_vector_tail_size_end else {
						return reduce_block(_First, _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op);
					}
				_Check_vector_block_size_end else{
					_Check_vector_tail_size_begin(_ALIGNMENT_SIZE)
						return reduce_tail(_First, _Last, std::move(_Val), _Reduce_scalar_op);
					_Check_vector_tail_size_end else {
						return std::move(_Val);
					}
				}
			_Check_vector_alignment_end
			// otherwise
			return discontinuous_vector_operation<_BlkTy, _Traits>::reduce(_First, _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op);
		}
		
		
		template<typename _BinOp, typename _UnaryOp>
		static scalar_type transform_reduce_lead(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val,
			_BinOp _Sclreduce_op, _UnaryOp _Transform_op, size_t _Leadsize) {
			assert( _Leadsize != 0 );
			const auto _Result = _Transform_op( * (reinterpret_cast<block_const_pointer>(_Last) - 1) );
			auto       _Rlast  = traits_type::end(_Result);
			return std::reduce(std::prev(_Rlast, _Leadsize), _Rlast, std::move(_Val), _Sclreduce_op);
		}

		template<typename _BinOp1, typename _BinOp2, typename _UnaryOp>
		static scalar_type transform_reduce_block(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val,
			_BinOp2 _Blkreduce_op, _BinOp1 _Sclreduce_op, _UnaryOp _Transform_op) {
			assert( _First != _Last );
			return aligned_vector_operation<_BlkTy, _Traits>::transform_reduce(reinterpret_cast<block_const_pointer>(_First), 
																			   reinterpret_cast<block_const_pointer>(_Last), 
																			   std::move(_Val), 
																			  _Blkreduce_op, _Sclreduce_op, _Transform_op);
		}

		template<typename _BinOp, typename _UnaryOp>
		static scalar_type transform_reduce_tail(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val,
			_BinOp _Sclreduce_op, _UnaryOp _Transform_op, size_t _Tailsize) {
			assert( _Tailsize != 0 );
			const auto _Result = _Transform_op( * reinterpret_cast<block_const_pointer>(_First) );
			auto       _Rfirst = traits_type::begin(_Result);
			return std::reduce(_Rfirst, std::next(_Rfirst, _Tailsize), std::move(_Val), _Sclreduce_op);
		}

		template<typename _BinOp1, typename _BinOp2, typename _UnaryOp>
		static scalar_type transform_reduce_block_and_tail(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val,
			_BinOp1 _Blkreduce_op, _BinOp2 _Sclreduce_op, _UnaryOp _Transform_op, size_t _Tailsize) {
			assert( _Tailsize != 0 );
			assert( _First != _Last - _Tailsize );
			// 1.
			block_const_pointer       _Ufirst = reinterpret_cast<block_const_pointer>(_First);
			const block_const_pointer _Ulast  = reinterpret_cast<block_const_pointer>(std::prev(_Last, _Tailsize));
			block_const_pointer       _Result = _Transform_op( *_Ufirst++ );
			for ( ; _Ufirst != _Ulast; ++_Ufirst) {
				_Result = _Blkreduce_op( std::move(_Result), _Transform_op( *_Ufirst ) );
			}
			_Val = std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Sclreduce_op);
			// 2.
			     _Result = _Transform_op(*_Ufirst);
			auto _Rfirst = traits_type::begin(_Result);
			return std::reduce(_Rfirst, std::next(_Rfirst, _Tailsize), std::move(_Val), _Sclreduce_op);
		}

		template<typename _BinOp1, typename _BinOp2, typename _UnaryOp>
		static scalar_type transform_reduce(scalar_const_pointer _First, scalar_const_pointer _Last, scalar_type _Val,
			_BinOp1 _Blkreduce_op, _BinOp2 _Sclreduce_op, _UnaryOp _Transform_op) {
			// reduce _Transform_op( [_First. _Last) )
			const size_t _ALIGNMENT_SIZE = traits_type::size();

			_Check_vector_alignment_begin(_First, _Last, sizeof(scalar_type), _ALIGNMENT_SIZE, alignment_mask_of_v<block_type>)
				_Check_vector_lead_size_begin(_ALIGNMENT_SIZE)
					_Val    = transform_reduce_lead(_First, _First+lead_size, std::move(_Val), _Sclreduce_op, _Transform_op, lead_size);
					_First += lead_size;
				_Check_vector_lead_size_end

				_Check_vector_block_size_begin(_ALIGNMENT_SIZE)
					_Check_vector_tail_size_begin(_ALIGNMENT_SIZE)
						return transform_reduce_block_and_tail(_First, _Last, std::move(_Val), _Blkreduce_op, _Sclreduce_op, _Transform_op, tail_size);
					_Check_vector_tail_size_end else{
						return transform_reduce_block(_First, _Last, std::move(_Val), _Blkreduce_op, _Sclreduce_op, _Transform_op);
					}
				_Check_vector_block_size_end else {
					_Check_vector_tail_size_begin(_ALIGNMENT_SIZE)
						return transform_reduce_tail(_First, _Last, std::move(_Val), _Sclreduce_op, _Transform_op, tail_size);
					_Check_vector_tail_size_end else {
						return std::move(_Val);
					}
				}
			_Check_vector_alignment_end
			// otherwise
			return discontinuous_vector_operation<_BlkTy, _Traits>::transform_reduce(_First, _Last, std::move(_Val), _Blkreduce_op, _Sclreduce_op, _Transform_op);
		}

		
		template<typename _BinOp1, typename _BinOp2>
		static scalar_type transform_reduce_lead(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_type _Val,
			_BinOp1 _Sclreduce_op, _BinOp2 _Transform_op, size_t _Leadsize) {
			// intrin, reduce _Transform_op( [_First1, _Last1), [_First2, ...) ) and _Val
			assert( _Leadsize != 0 );
			const size_t _Offset = traits_type::size() - _Leadsize;
			const auto   _Result = _Transform_op( * reinterpret_cast<block_const_pointer>(_First1 - _Offset), 
												  * reinterpret_cast<block_const_pointer>(_First2 - _Offset) );
			auto         _Rlast  = traits_type::end(_Result);
			return std::reduce(std::prev(_Rlast,_Leadsize), _Rlast, std::move(_Val), _Sclreduce_op);
		}

		template<typename _BinOp1, typename _BinOp2, typename _BinOp3>
		static scalar_type transform_reduce_block(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_type _Val,
			_BinOp1 _Blkreduce_op, _BinOp2 _Sclreduce_op, _BinOp3 _Transform_op) {
			// intrin, reduce _Transform_op( [_First1, _Last1), [_First2, ...) ) and _Val
			assert( _First1 != _Last1 );
			return aligned_vector_operation<_BlkTy, _Traits>::transform_reduce( reinterpret_cast<block_const_pointer>(_First1), 
																			reinterpret_cast<block_const_pointer>(_Last1), 
																			reinterpret_cast<block_const_pointer>(_First2), 
																			std::move(_Val),
																			_Blkreduce_op, _Sclreduce_op, _Transform_op );
		}

		template<typename _BinOp1, typename _BinOp2>
		static scalar_type transform_reduce_tail(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_type _Val,
			_BinOp1 _Sclreduce_op, _BinOp2 _Transform_op, size_t _Tailsize) {
			assert( _Tailsize != 0 );
			const auto _Result = _Transform_op( *reinterpret_cast<block_const_pointer>(_First1), *reinterpret_cast<block_const_pointer>(_First2) );
			auto       _Rfirst = traits_type::begin(_Result);
			return std::reduce(_Rfirst, std::next(_Rfirst, _Tailsize), std::move(_Val), _Sclreduce_op);
		}

		template<typename _BinOp1, typename _BinOp2, typename _BinOp3>
		static scalar_type transform_reduce_block_and_tail(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_type _Val,
			_BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _BinOp3 _Transform_op, size_t _Tailsize) {
			assert( _Tailsize != 0 );
			assert( _First1 != _Last1 - _Tailsize );
			// 1.
			block_const_pointer       _Ufirst1 = reinterpret_cast<block_const_pointer>(_First1);
			const block_const_pointer _Ulast1  = reinterpret_cast<block_const_pointer>(_Last1 - _Tailsize);
			block_const_pointer       _Ufirst2 = reinterpret_cast<block_const_pointer>(_First2);
			block_const_pointer       _Result  = _Transform_op( *_Ufirst1++, *_Ufirst2++ );
			for ( ; _Ufirst1 != _Ulast1; ++_Ufirst1, ++_Ufirst2) {
				_Result = _Reduce_block_op(std::move(_Result), _Transform_op( *_Ufirst1, *_Ufirst2 ));
			}
			_Val = std::reduce(traits_type::begin(_Result), traits_type::end(_Result), std::move(_Val), _Reduce_scalar_op);
			// 2.
			           _Result = _Transform_op( *_Ufirst1, *_Ulast1 );
			auto       _Rfirst = traits_type::begin(_Result);
			return std::reduce(_Rfirst, std::next(_Rfirst, _Tailsize), std::move(_Val), _Reduce_scalar_op);
		}

		template<typename _BinOp1, typename _BinOp2, typename _BinOp3>
		static scalar_type transform_reduce(scalar_const_pointer _First1, scalar_const_pointer _Last1, scalar_const_pointer _First2, scalar_type _Val,
			_BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _BinOp3 _Transform_op) {
			// reduce _Tranform_op( [_First1, _Last1), [_First2, ...) )
			const size_t _ALIGNMENT_SIZE = traits_type::size();
			
			_Check_vector_alignment2_begin(_First1, _Last1, _First2, sizeof(scalar_type), _ALIGNMENT_SIZE, alignment_mask_of_v<block_type>)
				_Check_vector_lead_size_begin(_ALIGNMENT_SIZE)
					_Val     = transform_reduce_lead(_First1, _Last1, _First2, std::move(_Val), _Reduce_scalar_op, _Transform_op, lead_size);
					_First1 += lead_size;
					_First2 += lead_size;
				_Check_vector_lead_size_end

				_Check_vector_block_size_begin(_ALIGNMENT_SIZE)
					_Check_vector_tail_size_begin(_ALIGNMENT_SIZE)
						return transform_reduce_block_and_tail(_First1, _Last1, _First2, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op, tail_size);
					_Check_vector_tail_size_end else{
						return transform_reduce_block(_First1, _Last1, _First2, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op);
					}
				_Check_vector_block_size_end else {
					_Check_vector_tail_size_begin(_ALIGNMENT_SIZE)
						return transform_reduce_tail(_First1, _Last1, _First2, std::move(_Val), _Reduce_scalar_op, _Transform_op, tail_size);
					_Check_vector_tail_size_end else {
						return std::move(_Val);
					}
				}
			_Check_vector_alignment2_end
			// otherwise
			return discontinuous_vector_operation<_BlkTy, _Traits>::transform_reduce(_First1, _Last1, _First2, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op);
		}
	};
	
	// { auto_vector_operation }
	template<typename _BlkTy, typename _Traits = block_traits<_BlkTy>>
	struct auto_vector_operation {
		using traits_type = _Traits;
		using scalar_type = typename _Traits::scalar_type;
		using block_type  = typename _Traits::block_type;

		using iterator       = _Any_array_iterator<scalar_type>;
		using const_iterator = _Any_array_const_iterator<scalar_type>;

		template<typename _UnaryOp>
		static iterator transform(const_iterator _First, const_iterator _Last, iterator _Dest, _UnaryOp _Transform_op) {
			// transform [_First, _Last) with _Transform_op
			if ( _First.continuous() && _Dest.continuous() ) {
				return continuous_vector_operation<_BlkTy, _Traits>::transform(&_First, &_Last, &_Dest, _Transform_op);
			} else {
				return discontinuous_vector_operation<_BlkTy, _Traits>::transform( _First, _Last, _Dest, _Transform_op );
			}
		}
	
		template<typename _BinOp>
		static iterator transform(const_iterator _First1, const_iterator _Last1, const_iterator _First2, iterator _Dest, _BinOp _Transform_op) {
			// transform [_First1, _Last1) and [_Frist2, ...) with _Transform_op
			if ( _First1.continuous() && _First2.continuous() && _Dest.continuous() ) {
				return continuous_vector_operation<_BlkTy, _Traits>::transform(&_First1, &_Last1, &_First2, &_Dest, _Transform_op);
			} else {
				return discontinuous_vector_operation<_BlkTy, _Traits>::transform( _First1, _Last1, _First2, _Dest, _Transform_op );
			}
		}

		template<typename _BinOp1, typename _BinOp2>
		static scalar_type reduce(const_iterator _First, const_iterator _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op) {
			// accumulate [_First, _Last) with _Reduce_op
			if ( _First.continuous() ) {
				continuous_vector_operation<_BlkTy, _Traits>::reduce(&_First, &_Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op );
			} else {
				return discontinuous_vector_operation<_BlkTy, _Traits>::reduce(_First,  _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op);
			}
		}

		template<typename _BinOp1, typename _BinOp2, typename _UnaryOp>
		static scalar_type transform_reduce(const_iterator _First, const_iterator _Last, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _UnaryOp _Transform_op) {
			// transform_accumulate [_First, _Last) with _Transform_op and _Reduce_op
			if ( _First.continuous() ) {
				return continuous_vector_operation<_BlkTy, _Traits>::transform_reduce(&_First, &_Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op);
			} else {
				return discontinuous_vector_operation<_BlkTy, _Traits>::transform_reduce( _First, _Last, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op );
			}
		}

		template<typename _BinOp1, typename _BinOp2, typename _BinOp3>
		static scalar_type transform_reduce(const_iterator _First1, const_iterator _Last1, const_iterator _First2, scalar_type _Val, _BinOp1 _Reduce_block_op, _BinOp2 _Reduce_scalar_op, _BinOp3 _Transform_op) {
			// transform accumulate [_First1, _Last1) and [_First2, ...) with _Transform_op and _Reduce_op
			if ( _First1.continuous() && _First2.continuous() ) {
				return continuous_vector_operation<_BlkTy, _Traits>::transform_reduce( &_First1, &_Last1, &_First2, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op );
			} else {
				return discontinuous_vector_operation<_BlkTy, _Traits>::transform_reduce( _First1, _Last1, _First2, std::move(_Val), _Reduce_block_op, _Reduce_scalar_op, _Transform_op );
			}
		}
	};

	// vector memory category
	struct discontinuous_vector_tag {};
	struct continuous_vector_tag {};
	struct aligned_vector_tag {};
	struct static_aligned_vector_tag {};

	template<typename _VectorTy, typename = void>
	struct vector_traits {
		using scalar_type     = typename _VectorTy::value_type;
		using block_type      = scalar_type;
		using vector_category = continuous_vector_tag;
	};

	template<typename _VectorTy>
	struct vector_traits<_VectorTy, std::void_t<typename _VectorTy::scalar_type, typename _VectorTy::block_type, typename _VectorTy::vector_category>> {
		using scalar_type     = typename _VectorTy::scalar_type;
		using block_type      = typename _VectorTy::block_type;
		using vector_category = typename _VectorTy::vector_category;
	};

	template<typename _Ty>
	struct vector_traits<_Ty*> {
		using scalar_type     = _Ty;
		using block_type      = _Ty;
		using vector_category = continuous_vector_tag;
	};

	template<typename _Vty>
	using vector_scalar_t = typename vector_traits<_Vty>::scalar_type;

	template<typename _Vty>
	using vector_block_t = typename vector_traits<_Vty>::block_type;

	template<typename _VectorTy> constexpr
	bool is_discontinuous_vector_v = std::is_same_v<typename vector_traits<_VectorTy>::vector_category, discontinuous_vector_tag>;

	template<typename _VectorTy> constexpr
	bool is_continuous_vector_v = std::is_same_v<typename vector_traits<_VectorTy>::vector_category, continuous_vector_tag>;

	template<typename _VectorTy> constexpr
	bool is_aligned_vector_v = std::is_same_v<typename vector_traits<_VectorTy>::vector_category, aligned_vector_tag>;

	template<typename _VectorTy> constexpr
	bool is_static_aligned_vector_v = std::is_same_v<typename vector_traits<_VectorTy>::vector_category, static_aligned_vector_tag>;

#ifdef _HAS_CXX20
	template<typename _VecTy> 
	concept discontinuous_vector =
		requires(_VecTy __v) { __v.begin(); __v.end(); };
		
	template<typename _VecTy>
	concept continuous_vector = 
		requires(_VecTy __v) { __v.data(); __v.size(); };

	template<typename _VecTy>
	concept aligned_vector = 
		requires(_VecTy __v) { __v.data(); __v.size(); };

	template<typename _VectorTy>
	concept static_aligned_vector = 
		requires(_VectorTy __v) { __v.data(); __v.size(); }
		&& std::is_same_v<static_aligned_vector_tag, typename vector_traits<_VectorTy>::vector_category>;
#endif



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////// VECTOR OPERATION ///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename _BlkTy/*noAuto*/, size_t _Sclsize/*noAuto*/, typename _SclTy, typename _UnaryOp>
	void transform_static_aligned_vector(const _SclTy* _First_data, _SclTy* _Dest_data, _UnaryOp _Transform_op) {
		if constexpr ( _Sclsize % block_traits<_BlkTy>::size() == 0 ){// aligned
			aligned_vector_operation<_BlkTy>::transform(reinterpret_cast<const _BlkTy*>(_First_data), reinterpret_cast<const _BlkTy*>(_First_data + _Sclsize),
														reinterpret_cast<_BlkTy*>(_Dest_data),
														_Transform_op);
		} else {
			if constexpr ( _Sclsize < block_traits<_BlkTy>::size() ) {
				continuous_vector_operation<_BlkTy>::transform_tail(_First_data, _First_data + _Sclsize, _Dest_data, _Transform_op,
					_Sclsize % block_traits<_BlkTy>::size());
			} else {
				continuous_vector_operation<_BlkTy>::transform_block_and_tail(_First_data, _First_data + _Sclsize, _Dest_data, _Transform_op,
					_Sclsize % block_traits<_BlkTy>::size());
			}
		}
	}
	
	template<typename _BlkTy/*noAuto*/, typename _SclTy, typename _UnaryOp>
	void transform_aligned_vector(const _SclTy* _First_data, size_t _Raw_count, _SclTy* _Dest_data, _UnaryOp _Transform_op) {
		if ( _Raw_count % block_traits<_BlkTy>::size() == 0 ) {// aligned
			aligned_vector_operation<_BlkTy>::transform(reinterpret_cast<const _BlkTy*>(_First_data), reinterpret_cast<const _BlkTy*>(_First_data + _Raw_count),
														reinterpret_cast<_BlkTy*>(_Dest_data),
														_Transform_op);
		} else {
			if ( _Raw_count < block_traits<_BlkTy>::size() ) {
				continuous_vector_operation<_BlkTy>::transform_tail(_First_data, _First_data + _Raw_count, _Dest_data, _Transform_op, 
					_Raw_count % block_traits<_BlkTy>::size());
			} else {
				continuous_vector_operation<_BlkTy>::transform_block_and_tail(_First_data, _First_data + _Raw_count, _Dest_data, _Transform_op, 
					_Raw_count % block_traits<_BlkTy>::size());
			}
		}
	}

	template<typename _BlkTy/*noAuto*/, typename _SclTy, typename _UnaryOp>
	_SclTy* transform_continuous_vector(const _SclTy* _First_data, size_t _Raw_count, _SclTy* _Dest_data, _UnaryOp _Transform_op) {
		return continuous_vector_operation<_BlkTy>::transform(_First_data, _First_data + _Raw_count, _Dest_data, _Transform_op);
	}
	
	template<typename _BlkTy/*noAuto*/, typename _InIt, typename _OutIt, typename _UnaryOp>
	_OutIt transform_discontinuous_vector(_InIt _First, _InIt _Last, _OutIt _Dest, _UnaryOp _Transform_op) {
		return discontinuous_vector_operation<_BlkTy>::transform(_First, _Last, _Dest, _Transform_op);
	}

	template<typename _BlkTy/*noAuto*/, typename _InVec, typename _OutVec, typename _UnaryOp>
	void transform_vector(const _InVec& _Vector, _OutVec& _Resultvector, _UnaryOp _Transform_op) {
		// transform [ _Vector.begin(), _Vector.end() ) into [ _Resultvector.begin(), ... ) with _Transform_op
		assert( ! _Vector.empty() );
		assert( _Vector.size() == _Resultvector.size() );

		if constexpr ( is_discontinuous_vector_v<_InVec> || is_discontinuous_vector_v<_OutVec> ) {// requires{ begin(), end() }
			transform_discontinuous_vector<_BlkTy>(_Vector.begin(), _Vector.end(), _Resultvector.begin(), _Transform_op);
		
		} else if constexpr ( is_aligned_vector_v<_InVec> ) {// requires{ data(), size() } 
			transform_aligned_vector<_BlkTy>(_Vector.data(), _Vector.size(), _Resultvector.data(), _Transform_op);

		} else /*if constexpr ( is_continuous_vector_v<_InVec> )*/ {// requires{ data(), size() } 
			transform_continuous_vector<_BlkTy>(_Vector.data(), _Vector.size(), _Resultvector.data(), _Transform_op);

		}
	}


	template<typename _BlkTy/*noAuto*/, size_t _SclSize/*noAuto*/, typename _SclTy, typename _BinOp>
	void transform_static_aligned_vector(const _SclTy* _First1_data, const _SclTy* _First2_data, _SclTy* _Dest_data, _BinOp _Transform_op) {
		if constexpr ( _SclSize % block_traits<_BlkTy>::size() == 0 ){// aligned
			aligned_vector_operation<_BlkTy>::transform(reinterpret_cast<const _BlkTy*>(_First1_data), reinterpret_cast<const _BlkTy*>(_First1_data + _SclSize),
														reinterpret_cast<const _BlkTy*>(_First2_data),
														reinterpret_cast<_BlkTy*>(_Dest_data),
														_Transform_op);
		} else {
			if constexpr ( _SclSize < block_traits<_BlkTy>::size() ) {
				continuous_vector_operation<_BlkTy>::transform_tail(_First1_data, _First1_data + _SclSize, _First2_data, _Dest_data, _Transform_op, 
					_SclSize % block_traits<_BlkTy>::size());
			} else {
				continuous_vector_operation<_BlkTy>::transform_block_and_tail(_First1_data, _First1_data + _SclSize, _First2_data, _Dest_data, _Transform_op, 
					_SclSize % block_traits<_BlkTy>::size());
			}
		}
	}
	
	template<typename _BlkTy/*noAuto*/, typename _SclTy, typename _BinOp>
	void transform_aligned_vector(const _SclTy* _First1_data, size_t _Raw_count, const _SclTy* _First2_data, _SclTy* _Dest_data, _BinOp _Transform_op) {
		if ( _Raw_count % block_traits<_BlkTy>::size() == 0 ){// aligned
			aligned_vector_operation<_BlkTy>::transform(reinterpret_cast<const _BlkTy*>(_First1_data), reinterpret_cast<const _BlkTy*>(_First1_data + _Raw_count),
														reinterpret_cast<const _BlkTy*>(_First2_data),
														reinterpret_cast<_BlkTy*>(_Dest_data),
														_Transform_op);
		} else {
			if ( _Raw_count < block_traits<_BlkTy>::size() ) {
				continuous_vector_operation<_BlkTy>::transform_tail(_First1_data, _First1_data + _Raw_count, _First2_data, _Dest_data, _Transform_op, 
					_Raw_count % block_traits<_BlkTy>::size());
			} else {
				continuous_vector_operation<_BlkTy>::transform_block_and_tail(_First1_data, _First1_data + _Raw_count, _First2_data, _Dest_data, _Transform_op,
					_Raw_count % block_traits<_BlkTy>::size());
			}
		}
	}
	
	template<typename _BlkTy/*noAuto*/, typename _SclTy, typename _BinOp>
	_SclTy* transform_continuous_vector(const _SclTy* _First1_data, size_t _Raw_count, const _SclTy* _First2_data, _SclTy* _Dest_data, _BinOp _Transform_op) {
		return continuous_vector_operation<_BlkTy>::transform(_First1_data, _First1_data + _Raw_count, _First2_data, _Dest_data, _Transform_op);
	}

	template<typename _BlkTy/*noAuto*/, typename _InIt1, typename _InIt2, typename _OutIt, typename _BinOp>
	_OutIt transform_discontinuous_vector(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _BinOp _Transform_op) {
		return discontinuous_vector_operation<_BlkTy>::transform(_First1, _Last1, _First2, _Dest, _Transform_op);
	}

	template<typename _BlkTy/*noAuto*/, typename _InVec1, typename _InVec2, typename _OutVec, typename _BinOp>
	void transform_vector(const _InVec1& _Lvector, const _InVec2& _Rvector, _OutVec& _Resultvector, _BinOp _Transform_op) {
		// transform [ _Vector.begin(), _Vector.end() ) and [_Rvector.begin(), ... ) into [ _Resultvector.begin(), ... ) with _Transform_op
		assert( ! _Lvector.empty() );
		assert( _Lvector.size() == _Rvector.size() && _Lvector.size() == _Resultvector.size() );

		if constexpr ( is_discontinuous_vector_v<_InVec1> || is_discontinuous_vector_v<_InVec2> || is_discontinuous_vector_v<_OutVec> ) {// requires{ begin(), end() }
			transform_discontinuous_vector<_BlkTy>(_Lvector.begin(), _Lvector.end(), _Rvector.begin(), _Resultvector.begin(), _Transform_op);

		} else if constexpr ( (is_aligned_vector_v<_InVec1> || is_static_aligned_vector_v<_InVec1>) && (is_aligned_vector_v<_InVec2> || is_static_aligned_vector_v<_InVec2>) ) {// requires{ data(), size() } 
			transform_aligned_vector<_BlkTy>(_Lvector.data(), _Lvector.size(), _Rvector.data(), _Resultvector.data(), _Transform_op);
		
		} else {// requires{ data(), size() } 
			transform_continuous_vector<_BlkTy>(_Lvector.data(), _Lvector.size(), _Rvector.data(), _Resultvector.data(), _Transform_op);

		}
	}


	template<typename _BlkTy/*noAuto*/, size_t _SclSize/*noAuto*/, typename _SclTy, typename _BinOp1, typename _BinOp2>
	_SclTy reduce_static_aligned_vector(const _SclTy* _First1_data, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op) {
		if constexpr ( _SclSize % block_traits<_BlkTy>::size() == 0 ){// aligned
			return aligned_vector_operation<_BlkTy>::reduce(reinterpret_cast<const _BlkTy*>(_First1_data), reinterpret_cast<const _BlkTy*>(_First1_data + _SclSize), std::move(_Val),
					_BReduce_op, _SReduce_op );
		} else {
			if constexpr ( _SclSize < block_traits<_BlkTy>::size() ) {
				return continuous_vector_operation<_BlkTy>::reduce_tail(_First1_data, _First1_data + _SclSize, std::move(_Val), 
					_BReduce_op, _SReduce_op, _SclSize % block_traits<_BlkTy>::size() );
			} else {
				return continuous_vector_operation<_BlkTy>::reduce_block_and_tail(_First1_data, _First1_data + _SclSize, std::move(_Val),
					_BReduce_op, _SReduce_op, _SclSize % block_traits<_BlkTy>::size() );
			}
		}
	}
	
	template<typename _BlkTy/*noAuto*/, typename _SclTy, typename _BinOp1, typename _BinOp2>
	_SclTy reduce_aligned_vector(const _SclTy* _First1_data, size_t _Raw_count, _SclTy _Val, _BinOp1 _Breduce_op, _BinOp2 _Sreduce_op) {
		if ( _Raw_count % block_traits<_BlkTy>::size() == 0 ){// aligned
			return aligned_vector_operation<_BlkTy>::reduce(reinterpret_cast<const _BlkTy*>(_First1_data), reinterpret_cast<const _BlkTy*>(_First1_data + _Raw_count), std::move(_Val), 
				_Breduce_op, _Sreduce_op );
		} else {
			if ( _Raw_count < block_traits<_BlkTy>::size() ) {
				return continuous_vector_operation<_BlkTy>::reduce_tail(_First1_data, _First1_data + _Raw_count, std::move(_Val),
					_Breduce_op, _Sreduce_op, _Raw_count % block_traits<_BlkTy>::size() );
			} else {
				return continuous_vector_operation<_BlkTy>::reduce_block_and_tail(_First1_data, _First1_data + _Raw_count, std::move(_Val),
					_Breduce_op, _Sreduce_op, _Raw_count % block_traits<_BlkTy>::size() );
			}
		}
	}

	template<typename _BlkTy/*noAuto*/, typename _InVec, typename _SclTy, typename _BinOp1, typename _BinOp2>
	_SclTy reduce_continuous_vector(const _SclTy* _First1_data, size_t _Raw_count, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op) {
		return continuous_vector_operation<_BlkTy>::reduce(_First1_data, _First1_data + _Raw_count, std::move(_Val), _BReduce_op, _SReduce_op );
	}

	template<typename _BlkTy/*noAuto*/, typename _InIt, typename _SclTy, typename _BinOp1, typename _BinOp2>
	_SclTy reduce_discontinuous_vector(_InIt _First, _InIt _Last, _SclTy _Val, _BinOp1 _Breduce_op, _BinOp2 _Sreduce_op) {
		return discontinuous_vector_operation<_BlkTy>::reduce(_First, _Last, std::move(_Val), _Breduce_op, _Sreduce_op);
	}

	template<typename _BlkTy/*noAuto*/, typename _InVec, typename _SclTy, typename _BinOp1, typename _BinOp2>
	_SclTy reduce_vector(const _InVec& _Vector, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op) {
		if ( _Vector.empty() ) {
			return std::move(_Val);
		}

		if constexpr ( is_discontinuous_vector_v<_InVec> ) {// requires{ begin(), end() }
			return reduce_discontinuous_vector<_BlkTy>(_Vector.begin(), _Vector.end(), std::move(_Val), _BReduce_op, _SReduce_op);

		} else if constexpr ( is_aligned_vector_v<_InVec> ) {// requires{ data(), size() } 
			return reduce_aligned_vector<_BlkTy>(_Vector.data(), _Vector.size(), std::move(_Val), _BReduce_op, _SReduce_op);
		
		} else /*if constexpr ( is_continuous_vector_v<_InVec> )*/ {// requires{ data(), size() } 
			return reduce_continuous_vector<_BlkTy>(_Vector.data(), _Vector.size(), std::move(_Val), _BReduce_op, _SReduce_op);

		}
	}


	template<typename _BlkTy/*noAuto*/, size_t _SclSize/*noAuto*/, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
	_SclTy transform_reduce_static_aligned_vector(const _SclTy* _First_data, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _UnaryOp _Transform_op) {
		if constexpr ( _SclSize % block_traits<_BlkTy>::size() == 0 ) {// aligned
			return aligned_vector_operation<_BlkTy>::transform_reduce(reinterpret_cast<const _BlkTy*>(_First_data), reinterpret_cast<const _BlkTy*>(_First_data + _SclSize), std::move(_Val),
				_BReduce_op, _SReduce_op, _Transform_op );
		} else {
			if constexpr ( _SclSize < block_traits<_BlkTy>::size() ) {
				return continuous_vector_operation<_BlkTy>::transform_reduce_tail(_First_data, _First_data + _SclSize, std::move(_Val),
					_SReduce_op, _Transform_op, _SclSize % block_traits<_BlkTy>::size() );
			} else {
				return continuous_vector_operation<_BlkTy>::transform_reduce_block_and_tail(_First_data, _First_data + _SclSize, std::move(_Val),
					_BReduce_op, _SReduce_op, _Transform_op, _SclSize % block_traits<_BlkTy>::size() );
			}
		}
	}

	template<typename _BlkTy/*noAuto*/, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
	_SclTy transform_reduce_aligned_vector(const _SclTy* _First1_data, size_t _Raw_count, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _UnaryOp _Transform_op) {
		if ( _Raw_count % block_traits<_BlkTy>::size() == 0 ){// aligned
			return aligned_vector_operation<_BlkTy>::transform_reduce(reinterpret_cast<const _BlkTy*>(_First1_data), reinterpret_cast<const _BlkTy*>(_First1_data + _Raw_count), std::move(_Val),
				_BReduce_op, _SReduce_op, _Transform_op );
		} else {
			if ( _Raw_count < block_traits<_BlkTy>::size() ) {
				return continuous_vector_operation<_BlkTy>::transform_reduce_tail(_First1_data, _First1_data + _Raw_count, std::move(_Val),
					_SReduce_op, _Transform_op, _Raw_count % block_traits<_BlkTy>::size() );
			} else {
				return continuous_vector_operation<_BlkTy>::transform_reduce_block_and_tail(_First1_data, _First1_data + _Raw_count, std::move(_Val), 
					_BReduce_op, _SReduce_op, _Transform_op, _Raw_count % block_traits<_BlkTy>::size() );
			}
		}
	}

	template<typename _BlkTy/*noAuto*/, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
	_SclTy transform_reduce_continuous_vector(const _SclTy* _First1_data, size_t _Raw_count, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _UnaryOp _Transform_op) {
		return continuous_vector_operation<_BlkTy>::transform_reduce(_First1_data, _First1_data + _Raw_count, std::move(_Val), 
			_BReduce_op, _SReduce_op, _Transform_op );
	}

	template<typename _BlkTy/*noAuto*/, typename _InIt, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
	_SclTy transform_reduce_discontinuous_vector(_InIt _First, _InIt _Last, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _UnaryOp _Transform_op) {
		return discontinuous_vector_operation<_BlkTy>::transform_reduce(_First, _Last, std::move(_Val), 
			_BReduce_op, _SReduce_op, _Transform_op);
	}
	
	template<typename _BlkTy/*noAuto*/, typename _InVec, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _UnaryOp>
	_SclTy transform_reduce_vector(const _InVec& _Vector, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _UnaryOp _Transform_op) {
		if ( _Vector.empty() ) {
			return std::move(_Val);
		}

		if constexpr ( is_discontinuous_vector_v<_InVec> ) {// requires{ begin(), end() }
			return transform_reduce_discontinuous_vector<_BlkTy>(_Vector.begin(), _Vector.end(), std::move(_Val), _BReduce_op, _SReduce_op, _Transform_op);

		} else if constexpr ( is_aligned_vector_v<_InVec> ) {// requires{ data(), size() } 
			return transform_reduce_aligned_vector<_BlkTy>(_Vector.data(), _Vector.size(), std::move(_Val), _BReduce_op, _SReduce_op, _Transform_op);
		
		} else /*if constexpr ( is_continuous_vector_v<_InVec> )*/ {// requires{ data(), size() } 
			return transform_reduce_continuous_vector<_BlkTy>(_Vector.data(), _Vector.size(), std::move(_Val), _BReduce_op, _SReduce_op, _Transform_op);

		}
	}


	template<typename _BlkTy, size_t _SclSize, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _BinOp3>
	_SclTy transform_reduce_static_aligned_vector(const _SclTy* _First1_data, const _SclTy* _First2_data, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _BinOp3 _Transform_op) {
		if constexpr ( _SclSize % block_traits<_BlkTy>::size() == 0 ){// aligned
			return aligned_vector_operation<_BlkTy>::transform_reduce(reinterpret_cast<const _BlkTy*>(_First1_data), reinterpret_cast<const _BlkTy*>(_First1_data + _SclSize),
				reinterpret_cast<const _BlkTy*>(_First2_data), std::move(_Val),
				_BReduce_op, _SReduce_op, _Transform_op );
		} else {
			if constexpr ( _SclSize < block_traits<_BlkTy>::size() ) {
				return continuous_vector_operation<_BlkTy>::transform_reduce_tail(_First1_data, _First1_data + _SclSize, _First2_data, std::move(_Val),
					_SReduce_op, _Transform_op, _SclSize % block_traits<_BlkTy>::size() );
			} else {
				return continuous_vector_operation<_BlkTy>::transform_reduce_block_and_tail(_First1_data, _First1_data + _SclSize, _First2_data, std::move(_Val),
					_BReduce_op, _SReduce_op, _Transform_op, _SclSize % block_traits<_BlkTy>::size() );
			}
		}
	}
	
	template<typename _BlkTy, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _BinOp3>
	_SclTy transform_reduce_aligned_vector(const _SclTy* _First1_data, size_t _Raw_count, const _SclTy* _First2_data, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _BinOp3 _Transform_op) {
		if ( _Raw_count % block_traits<_BlkTy>::size() == 0 ){// aligned
			return aligned_vector_operation<_BlkTy>::transform_reduce(reinterpret_cast<const _BlkTy*>(_First1_data), reinterpret_cast<const _BlkTy*>(_First1_data+_Raw_count),
				reinterpret_cast<const _BlkTy*>(_First2_data), std::move(_Val),
				_BReduce_op, _SReduce_op, _Transform_op );
		} else {
			if ( _Raw_count < block_traits<_BlkTy>::size() ) {
				return continuous_vector_operation<_BlkTy>::transform_reduce_tail(_First1_data, _First1_data + _Raw_count, _First2_data, std::move(_Val),
					_SReduce_op, _Transform_op, _Raw_count % block_traits<_BlkTy>::size() );
			} else {
				return continuous_vector_operation<_BlkTy>::transform_reduce_block_and_tail(_First1_data, _First1_data + _Raw_count, _First2_data, std::move(_Val),
					_BReduce_op, _SReduce_op, _Transform_op, _Raw_count % block_traits<_BlkTy>::size() );
			}
		}
	}

	template<typename _BlkTy, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _BinOp3>
	_SclTy transform_reduce_continuous_vector(const _SclTy* _First1_data, size_t _Raw_count, const _SclTy* _First2_data, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _BinOp3 _Transform_op) {
		return continuous_vector_operation<_BlkTy>::transform_reduce(_First1_data, _First1_data + _Raw_count, _First2_data, std::move(_Val),
			_BReduce_op, _SReduce_op, _Transform_op );
	}
	
	template<typename _BlkTy, typename _InIt1, typename _InIt2, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _BinOp3>
	_SclTy transform_reduce_discontinuous_vector(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _BinOp3 _Transform_op) {
		return discontinuous_vector_operation<_BlkTy>::transform_reduce(_First1, _Last1, _First2, std::move(_Val),
			_BReduce_op, _SReduce_op, _Transform_op);
	}

	template<typename _BlkTy, typename _InVec1, typename _InVec2, typename _SclTy, typename _BinOp1, typename _BinOp2, typename _BinOp3>
	_SclTy transform_reduce_vector(const _InVec1& _Lvector, const _InVec2& _Rvector, _SclTy _Val, _BinOp1 _BReduce_op, _BinOp2 _SReduce_op, _BinOp3 _Transform_op) {
		assert( _Lvector.size() == _Rvector.size() );
		if ( _Lvector.empty() ) {
			return std::move(_Val);
		}

		if constexpr ( is_discontinuous_vector_v<_InVec1> || is_discontinuous_vector_v<_InVec2> ) {// requires{ begin(), end() }
			return transform_reduce_discontinuous_vector<_BlkTy>(_Lvector.begin(), _Lvector.end(), _Rvector.begin(), std::move(_Val), _BReduce_op, _SReduce_op, _Transform_op);

		} else if constexpr ( (is_aligned_vector_v<_InVec1> || is_static_aligned_vector_v<_InVec1>) && (is_aligned_vector_v<_InVec2> || is_static_aligned_vector_v<_InVec2>) ) {// requires{ data(), size() } 
			return transform_reduce_aligned_vector<_BlkTy>(_Lvector.data(), _Lvector.size(), _Rvector.data(), std::move(_Val), _BReduce_op, _SReduce_op, _Transform_op);
		
		} else {// requires{ data(), size() } 
			return transform_reduce_continuous_vector<_BlkTy>(_Lvector.data(), _Lvector.size(), _Rvector.data(), std::move(_Val), _BReduce_op, _SReduce_op, _Transform_op);

		}
	}

}// namespace clmagic

/*<graph>
	array:  |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----| unit:[scalars]
	vector: |-----|-----------------|-----------------|-----------------|-----------------|-----| unit:[blocks]
			A0   A1														                  B0    B1
			|-led-|---------------------------- block ------------------------------------|-tai-|

	A0 = first
	B1 = last

	block_offset = A0 & alignment_mask_of_v<_BlkTy> unit:[bytes]
	lead_size    = A1 - A0 unit:[scalars]
	block_size   = B0 - A1 unit:[blocks]
	tail_size    = B1 - B0 unit:[scalars]
</graph>*/