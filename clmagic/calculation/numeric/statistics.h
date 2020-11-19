#pragma once
#include "real.h"
#include <vector>
#include <map>
#include <random>

using probability_t = realmax_t;

namespace Bernoulli {
	class probability_distribution : private std::bernoulli_distribution {
		/*<Reference> { Statistics for Engineering, Chapter4.5 } </Reference>*/
		using _Mybase = std::bernoulli_distribution;
	public:
		probability_distribution() = default;
		explicit probability_distribution(probability_t p) : _Mybase(p) {}

		probability_t p() const {
			return _Mybase::p();
		}
		probability_t q() const {
			return 1.0 - p();
		}

		template<typename _Engine>
		_Mybase::result_type evaluate(_Engine _Rng) const {
			return _Mybase::operator()(_Rng);
		}
		
		probability_t probability(size_t y) const {// Success probability for y times event
			return pow(p(), y) * pow(q(), 1.0 - y);
		}

		realmax_t expected_value() const {
			return p();
		}

		realmax_t variance() const {
			return p() * q();
		}
	};
}// namespace Bernoulli

class binomial_probability_distribution : private std::binomial_distribution<size_t> {
	/*<Reference> { Statistics for Engineering, Chapter4.6 } </Reference>*/
	using _Mybase = std::binomial_distribution<size_t>;
public:
	binomial_probability_distribution() = default;
	explicit binomial_probability_distribution(size_t t, probability_t p) : _Mybase(t, p) {}

	size_t t() const {
		return _Mybase::t();
	}
	probability_t p() const {
		return _Mybase::p();
	}
	probability_t q() const {
		return 1.0 - p();
	}

	template<typename _Engine>
	_Mybase::result_type evaluate(_Engine _Rng) const {
		return _Mybase::operator()(_Rng);
	}

	probability_t probability(size_t y) const {// probability for Success y times event in total t times
		return Newton::binomial_coefficient(t(), y) * pow(p(), y) * pow(q(), t() - y);
	}

	realmax_t expected_value() const {
		return t() * p();
	}

	realmax_t variance() const {
		return t() * p() * q();
	}
};




namespace calculation {

	template<typename _Ty>
	struct Range {
		_Ty lower, upper;

		Range() = default;
		Range(_Ty _Lower, _Ty _Upper) : lower(_Lower), upper(_Upper) {}

		Range operator&(Range _Right) const {
			return Range(std::min(this->lower,_Right.lower), std::max(this->upper, _Right.upper));
		}
		Range intersect(Range _Right) const {
			if (_Right.lower < this->upper) {
				return Range(_Right.lower, this->upper);
			} else if (this->lower < _Right.upper) {
				return Range(this->lower, _Right.upper);
			} else {
				return Range();
			}
		}
		bool exclude(_Ty _Val) const { return (_Val < lower || upper < _Val); }
		bool include(_Ty _Val) const { return !this->exclude(_Val); }
		bool exclude(Range _Right) const { return exclude(_Right.lower) || exclude(_Right.upper); }
		bool include(Range _Right) const { return include(_Right.lower) && include(_Right.upper); }
		_Ty range() const { return (upper - lower); }
		_Ty remap(_Ty _Val,_Ty _Lower, _Ty _Upper) const {
			return (_Val - _Lower)/(_Upper - _Lower) * range() + lower;
		}
	};

	template<typename _DataSet>
	using dataset_value_t = typename _DataSet::value_type;

	template<typename _DestTy,typename _SourceTy>
	_DestTy real_cast(_SourceTy _Val) {
		if constexpr (std::is_integral_v<_DestTy> && std::is_floating_point_v<_SourceTy>) {
			return static_cast<_DestTy>(round(_Val));
		} else {
			return static_cast<_DestTy>(_Val);
		}
	}

	template<typename _Pdf, typename _DataSet>
	dataset_value_t<_DataSet> expected_value(_Pdf P, const _DataSet& Y) { // expected value of P(Y)
		assert( !std::empty(Y) );
		return real_cast<dataset_value_t<_DataSet>>( 
			sum_of(Y, [P](dataset_value_t<_DataSet> y){ return y * P(y); }) 
		);
	}

	template<typename _Pdf, typename _DataSet,typename _Fn>
	dataset_value_t<_DataSet> expected_value(_Pdf P, const _DataSet& Y, _Fn g) { // expected value of P(g(Y))
		assert( !std::empty(Y) );
		return real_cast<dataset_value_t<_DataSet>>( 
			sum_of(Y, [P, g](dataset_value_t<_DataSet> y){ return g(y) * P(y); }) 
		);
	}


	/*<Theorem>
		expected_value(P, Y, [](auto){ return CONSTANT; })
			= CONSTANT

		expected_value(P, Y*CONSTANT)
			= exoected_value(P, Y) * CONSTANT

		expected_value(P, g1(Y) + g2(Y) + g3(Y) + ... + gN(Y))
			= expected_value(P,g1(Y)) + expected_value(P,g2(Y)) + expected_value(P,g3(Y)) + ... + expected_value(P,gN(Y))
	</Theorem>*/

	template<typename _Pdf, typename _DataSet>
	dataset_value_t<_DataSet> variance(_Pdf P, const _DataSet& Y) {// sum( pow(y-u,2) * P(y) ), u = E(P,Y), y in {Y}
		const dataset_value_t<_DataSet> u = expected_value(P, Y);
		return expected_value(P, Y, [u](dataset_value_t<_DataSet> y){ return pow(y - u, 2); });
		/*<another> return discrete_expected_value(P,Y,_Sqr) - _Sqr(discrete_expected_value(P, Y)); </another>*/
	}

	template<typename _Pdf,typename _DataSet>
	typename _DataSet::value_type standard_deviation(_Pdf P, const _DataSet& Y) {
		return real_cast<typename _DataSet::value_type>( sqrt(variance(P, Y)) );
	}

	/*<Theorem>
		variance(c * X)
			= _Sqr(c) * variance(X)

		variance(c + X)
			= variance(X)
		
		variance(X + Y) 
			= variance(X) + variance(Y)
	</Theorem>*/



	template<typename _Kty>
	struct _Table_index {
		_Kty   _Keyval;
		size_t _Index_start;
		size_t _Index_count;

		bool operator==(const _Table_index& _Right) const {
			return _Keyval == _Right._Keyval;
		}
		bool operator!=(const _Table_index& _Right) const {
			return _Keyval != _Right._Keyval;
		}
		bool operator<(const _Table_index& _Right) const {
			return _Keyval < _Right._Keyval;
		}
	};

	template<typename _Kty, typename _Ty>
	struct _Table_const_item {
		using index_type = _Table_index<_Kty>;
		using data_type  = std::vector<_Ty>;

		using value_type = typename std::vector<_Ty>::value_type;
		using iterator   = typename std::vector<_Ty>::const_iterator;
		using reference  = typename std::vector<_Ty>::const_reference;

		_Table_const_item(const index_type& _Idx, const data_type& _Data)
			: _Pidx(&_Idx), _Pdata(&_Data) {}

		size_t _Cont_offset() const {
			return _Pidx->_Index_start;
		}
		
		size_t size() const {
			return _Pidx->_Index_count;
		}
		bool empty() const {
			return _Pidx->_Index_count == 0;
		}

		iterator begin() const {
			return std::next(_Pdata->begin(), _Cont_offset());
		}
		iterator end() const {
			return std::next(_Pdata->begin(), _Cont_offset() + size());
		}
		reference at(size_t _Pos) const {
			assert(_Pos < size());
			return _Pdata->at(_Cont_offset() + _Pos);
		}
		reference operator[](size_t _Pos) const {
			assert(_Pos < size());
			return (*_Pdata)[_Cont_offset() + _Pos];
		}

		const index_type* _Pidx;
		const data_type* _Pdata;
	};

	template<typename _Kty, typename _Ty, typename _ContidxTy>
	struct _Table_item : public _Table_const_item<_Kty, _Ty> {
		_ContidxTy& _Mycont_indices;

		using _Mybase = _Table_const_item<_Kty, _Ty>;
		using index_type = typename _Mybase::index_type;
		using data_type  = typename _Mybase::data_type;

		using value_type = typename std::vector<_Ty>::value_type;
		using iterator   = typename std::vector<_Ty>::iterator;
		using reference  = typename std::vector<_Ty>::reference;

		_Table_item(index_type& _Idx, data_type& _Data, _ContidxTy& _Contidx)
			: _Mybase(_Idx, _Data), _Mycont_indices(_Contidx) {}

		index_type& _Get_idx() const {
			return * const_cast<_Table_index<_Kty>*>(_Mybase::_Pidx);
		}
		data_type& _Get_data() const {
			return * const_cast<std::vector<_Ty>*>(_Mybase::_Pdata);
		}
		void push_back(const _Ty& _Val) const {
			// insert _Val at _Mycont.end()
			_Get_data().insert(end(), _Val);
			// update _Mycont_indices._Index_start
			size_t     _Start_idx = _Mybase::_Cont_offset();
			size_t     _End_index = _Mybase::_Cont_offset() + _Mybase::size();
			const auto _Pred      = [_Start_idx, &_End_index](const _Table_index<_Kty>& _Idx) { return _Idx._Index_start != _Start_idx && _Idx._Index_start == _End_index; };
			auto       _Where     = std::find_if(_Mycont_indices.begin(), _Mycont_indices.end(), _Pred);
			for ( ; _Where != _Mycont_indices.end(); _Where = std::find_if(_Mycont_indices.begin(), _Mycont_indices.end(), _Pred)) {
				_Where->_Index_start += 1;
				_End_index            = _Where->_Index_start + _Where->_Index_count;
			}
			// update _Myidx._Index_count
			_Get_idx()._Index_count += 1;
		}

		iterator begin() const {
			return std::next(_Get_data().begin(), _Mybase::_Cont_offset());
		}
		iterator end() const {
			return std::next(_Get_data().begin(), _Mybase::_Cont_offset() + _Mybase::size());
		}
		reference at(size_t _Pos) const {
			assert(_Pos < _Mybase::size());
			return _Get_data().at(_Mybase::_Cont_offset() + _Pos);
		}
		reference operator[](size_t _Pos) const {
			assert(_Pos < _Mybase::size());
			return _Get_data()[_Mybase::_Cont_offset() + _Pos];
		}
	};

	template<typename _Kty, typename _Ty>
	struct _Table_const_iterator {
		using index_type = _Table_index<_Kty>;
		using data_type  = std::vector<_Ty>;

		using iterator_category = std::random_access_iterator_tag;
		using value_type        = _Table_const_item<_Kty, _Ty>;
		using difference_type   = ptrdiff_t;
		using pointer           = _Ty*;
		using reference         = value_type&;

		_Table_const_iterator(const index_type& _Idx, const data_type& _Data)
			: first( & reinterpret_cast<const _Kty&>(_Idx) ), second( _Idx, _Data ) {}

		const _Kty& key_value() const {
			return *first;
		}

		reference value() {
			return second;
		}

		reference operator*() {
			return value();
		}

		_Table_const_iterator operator+(difference_type _Diff) const {
			auto& _Seek = *(second._Pidx + _Diff);
			return _Table_const_iterator( _Seek, *(second._Pdata) );
		}

		_Table_const_iterator operator-(difference_type _Diff) const {
			return (*this) + (-_Diff);
		}

		_Table_const_iterator& operator+=(difference_type _Diff) {
			(*this) = (*this) + _Diff;
			return *this;
		}

		_Table_const_iterator& operator-=(difference_type _Diff) {
			(*this) = (*this) - _Diff;
			return *this;
		}

		_Table_const_iterator& operator++() {
			*this += 1;
			return *this;
		}

		_Table_const_iterator& operator--() {
			*this -= 1;
			return *this;
		}

		_Table_const_iterator operator++(int) {
			_Table_const_iterator _Tmp = *this;
			++(*this);
			return _Tmp;
		}

		_Table_const_iterator& operator--(int) {
			_Table_const_iterator _Tmp = *this;
			--(*this);
			return _Tmp;
		}

		bool operator==(const _Table_const_iterator& _Right) const {
			return first == _Right.first;
		}

		bool operator!=(const _Table_const_iterator& _Right) const {
			return first != _Right.first;
		}

		const _Kty* first;
		_Table_const_item<_Kty, _Ty> second;
	};

	template<typename _FwdIt, typename _Ty, typename _Pr, typename _Pr2>
	_FwdIt binary_find(_FwdIt _First, _FwdIt _Last, const _Ty& _Val, _Pr2 _Dividepred, _Pr _Equalpred) {
		_First = std::lower_bound(_First, _Last, _Val, _Dividepred);
		return _First != _Last && _Equalpred(*_First, _Val) ? _First : _Last;
	}

	// { std::map<_Kty, std::vector<_Ty>>, continuous_memory }
	template<typename _Kty, typename _Ty>
	class table {
	public:
		using reference       = _Table_item<_Kty, _Ty, std::vector<_Table_index<_Kty>>>;
		using const_reference = _Table_const_item<_Kty, _Ty>;
		using const_iterator  = _Table_const_iterator<_Kty, _Ty>;

		table() = default;
		table(const table&) = default;
		table(table&& _Right) : _Myidx(std::move(_Right._Myidx)), _Mydata(std::move(_Right._Mydata)) {}

		table& operator=(const table&) = default;
		table& operator=(table&& _Right) {
			_Myidx = std::move(_Right._Myidx);
			_Mydata = std::move(_Right._Mydata);
			return *this;
		}

		const_reference operator[](const _Kty& _Keyval) const {
			// binary_search _Where from _Keyval
			auto _Where = binary_find(_Myidx.begin(), _Myidx.end(), _Keyval,
							[](const _Table_index<_Kty>& _Left, const _Kty& _Right) { return _Left._Keyval < _Right; },
							[](const _Table_index<_Kty>& _Left, const _Kty& _Right) { return _Left._Keyval == _Right; });
			assert(_Where != _Myidx.end());
			return const_reference( *_Where, _Mydata );
		}
		reference operator[](const _Kty& _Keyval) {
			// binary_search _Where from _Keyval
			auto _Where = binary_find(_Myidx.begin(), _Myidx.end(), _Keyval,
							[](const _Table_index<_Kty>& _Left, const _Kty& _Right) { return _Left._Keyval < _Right; },
							[](const _Table_index<_Kty>& _Left, const _Kty& _Right) { return _Left._Keyval == _Right; });
			// Return or Create new _Table_item
			if (_Where != _Myidx.end()) {
				return reference( *_Where, _Mydata, _Myidx );
			} else {
				_Myidx.push_back( _Table_index<_Kty>{ _Keyval, _Mydata.size(), size_t(0)} );
				std::sort(_Myidx.begin(), _Myidx.end());
				return (*this)[_Keyval];
			}
		}

		const_iterator begin() const {
			return const_iterator(*_Myidx.data(), _Mydata);
		}
		const_iterator end() const {
			return const_iterator(*(_Myidx.data() + _Myidx.size()), _Mydata);
		}

		void insert_or_assign(const _Kty& _Keyval, const std::vector<_Ty>& _Mapval) {
			// insert [_Mapval.begin(), _Mapval.end()) into [_Mydata.end(), ...)
			_Mydata.insert(_Mydata.end(), _Mapval.begin(), _Mapval.end());
			// _Myidx push_back a new _Table_index
			_Myidx.push_back( _Table_index<_Kty>{ _Keyval, _Mydata.size(), _Mapval.size() } );
			// resort _Myidx
			std::sort(_Myidx.begin(), _Myidx.end());
		}

		size_t size() const {
			return _Myidx.size();
		}

		std::vector<_Table_index<_Kty>> _Myidx;
		std::vector<_Ty> _Mydata;
	};
	
	template<typename _Kty, typename _Ty>
	std::string to_string(const _Table_const_item<_Kty, _Ty>& _Source) {
		using std::to_string;
		if (!_Source.empty()) {
			std::string _Str;
			
			auto       _First = _Source.begin();
			const auto _Last  = _Source.end();
			
			_Str = to_string(*_First++);
			for ( ; _First != _Last; ++_First) {
				_Str += ',';
				_Str += to_string(*_First);
			}
			
			return _Str;
		} else {
			return "{}";
		}
	}

	template<typename _Kty, typename _Ty>
	std::string to_string(const table<_Kty, _Ty>& _Source) {
		using std::to_string;
		std::string _Str;
		std::string _Temp;
		auto       _First = _Source.begin();
		const auto _Last  = _Source.end();
		for (; _First != _Last; ++_First) {
			_Str += to_string(_First.key_value()) + ": " + to_string(_First.value());
			_Str += '\n';
		}
		return _Str;
	}


	struct Candidate_selection_problem {
		template<typename _Iter>
		table<size_t, typename std::iterator_traits<_Iter>::value_type> apply(_Iter _First, _Iter _Last) const {
			table<size_t, typename std::iterator_traits<_Iter>::value_type> result;
			return result;
		}

		template<typename _Iter>
		size_t size(_Iter _First, _Iter _Last) const {
			size_t N =  std::distance(_First, _Last);
			return factorial(N);
		}
	};

	struct Assignment_problem {
		template<typename _Iter>
		uintmax_t size(_Iter _First, _Iter _Last, size_t n) const {
			uintmax_t N = std::distance(_First, _Last);
			return factorial(N, N - n + 1);
		}
	};

}// namespace clmagic