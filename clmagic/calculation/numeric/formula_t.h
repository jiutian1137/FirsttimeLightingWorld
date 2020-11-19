//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Free License
//
// Table of contents
// 1. ratio_cast -----------------------------------------------------------
// 2. unit_symbol string ---------------------------------------------------
// 3. operation_symbol string ----------------------------------------------
// 4. formula_t<_Fml> ------------------------------------------------------
// 5. formula_t unit cast --------------------------------------------------
// 6. extend operation object ----------------------------------------------
//--------------------------------------------------------------------------------------
#pragma once
#ifndef clmagic_calculation_numeric_FORMULAT_h_
#define clmagic_calculation_numeric_FORMULAT_h_
#include <ratio>// std::ratio<>
#include <string>
#include <iostream>

#include "formula.h"
#include "real.h"// clmagic::scalar_t<>

namespace calculation {
	
#pragma region unit_symbol

	template<typename _Unit> constexpr
	const char* unit_symbol_v = _Unit::symbol;

	template<> constexpr
	const char* unit_symbol_v<void> = "";

	inline std::string _Unit_symbol_style(const std::string& _Symbol) {
		return _Symbol.empty() ? std::string() : ('['+_Symbol+']');
	}

#pragma endregion

#pragma region operation_symbol

	template<typename _Op> constexpr
	const char* operation_symbol_v = "?";

	template<typename _Ty> constexpr
	const char* operation_symbol_v<std::plus<_Ty>> = "+";

	template<typename _Ty> constexpr
	const char* operation_symbol_v<std::minus<_Ty>> = "-";

	template<typename _Ty> constexpr
	const char* operation_symbol_v<std::multiplies<_Ty>> = "*";

	template<typename _Ty> constexpr
	const char* operation_symbol_v<std::divides<_Ty>> = "/";
	
#pragma endregion

#pragma region formula_t<_Fml>

	template<typename _Fml>// recursion with _BinOp
	class formula_t {
	public:
		using _Myformula  = _Fml;

		using first_type  = formula_t<typename _Myformula::first_formula>;
		using second_type = formula_t<typename _Myformula::second_formula>;
		using result_type = typename _Myformula::result_type;

		first_type  first;
		second_type second;

		formula_t() = default;
		
		formula_t(const formula_t&) = default;
		
		formula_t(const first_type& _Val1, const second_type& _Val2) : first(_Val1), second(_Val2) {}
		
		template<typename _OtherFml, std::enable_if_t<is_similar_formula_v<_Myformula, _OtherFml>, int> = 0>
		formula_t(const formula_t<_OtherFml>& _Right) {// construct with unit cast
			*this = _Right; 
		}
	
		formula_t& operator=(const formula_t&) = default;
		
		formula_t& operator=(const std::pair<first_type, second_type>& _Pair) { 
			this->first  = _Pair.first; 
			this->second = _Pair.second; 
			return *this; 
		}
		
		template<typename _OtherFml, std::enable_if_t<is_similar_formula_v<_Myformula, _OtherFml>, int> = 0>
		formula_t& operator=(const formula_t<_OtherFml>& _Right);

		std::string to_string() const {
			using std::to_string;
			return ('(' + first.to_string() + operation_symbol_v<typename _Myformula::operation>
						+ second.to_string() + ')');
		}
		
		result_type get_result() const {
			return _Myformula()(first.get_result(), second.get_result());
		}

		explicit operator result_type() const {
			return get_result();
		}

		template<typename _OtherFml>
		formula_t<formula_plus<_Myformula, _OtherFml>> operator+(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_plus<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_minus<_Myformula, _OtherFml>> operator-(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_minus<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_multiplies<_Myformula, _OtherFml>> operator*(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_multiplies<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_divides<_Myformula, _OtherFml>> operator/(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_divides<_Myformula, _OtherFml>>(*this, _Right);
		}
	};
	
	// template recursion with _UnaryOp
	template<typename _Uty, typename _UnaryOp>
	class formula_t< formula<_Uty, void, _UnaryOp> > {
	public:
		using _Myformula  = formula<_Uty, void, _UnaryOp>;

		using first_type  = formula_t<typename _Myformula::first_formula>;
		using result_type = typename _Myformula::result_type;

		first_type first;

		formula_t() = default;

		formula_t(const formula_t&) = default;

		formula_t(const first_type& _Val) : first(_Val) {}
		
		template<typename _OtherFml, std::enable_if_t<is_similar_formula_v<_Myformula, _OtherFml>, int> = 0>
		formula_t(const formula_t<_OtherFml>& _Right) {// construct with unit cast
			(*this) = _Right; 
		}

		formula_t& operator=(const formula_t&) = default;
		
		formula_t& operator=(const first_type& _Val) {
			first = _Val; 
			return *this; 
		}
		
		template<typename _OtherFml, std::enable_if_t<is_similar_formula_v<_Myformula, _OtherFml>, int> = 0>
		formula_t& operator=(const formula_t<_OtherFml>& _Right);

		std::string to_string() const {
			return operation_symbol_v<typename _Myformula::operation> + first.to_string() + ')';
		}

		result_type get_result() const {
			return _Myformula()(first.get_result());
		}
		
		explicit operator result_type() const {
			return get_result();
		}

		template<typename _OtherFml>
		formula_t<formula_plus<_Myformula, _OtherFml>> operator+(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_plus<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_minus<_Myformula, _OtherFml>> operator-(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_minus<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_multiplies<_Myformula, _OtherFml>> operator*(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_multiplies<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_divides<_Myformula, _OtherFml>> operator/(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_divides<_Myformula, _OtherFml>>(*this, _Right);
		}
	};
	
	// template until
	template<typename _Uty>
	class formula_t< formula<_Uty, void, void> > {
	public:
		using _Myformula = formula<_Uty, void, void>;

		using first_type = typename _Myformula::result_type;
		using first_unit = typename _Myformula::result_unit;

		first_type first;

		formula_t() = default;
		
		formula_t(const formula_t&) = default;
		
		formula_t(const first_type& _Val) : first(_Val) {}
		
		template<typename _OtherFml, std::enable_if_t<is_similar_formula_v<_Myformula, _OtherFml>, int> = 0>
		formula_t(const formula_t<_OtherFml>& _Right) {
			*this = _Right;
		}

		formula_t& operator=(const formula_t&) = default;
		
		formula_t& operator=(const first_type& _Val) { first = _Val; return *this; }
		
		template<typename _OtherFml, std::enable_if_t<is_similar_formula_v<_Myformula, _OtherFml>, int> = 0>
		formula_t& operator=(const formula_t<_OtherFml>& _Right);

		std::string to_string() const {
			using std::to_string;
			return to_string(this->first) + _Unit_symbol_style(unit_symbol_v<first_unit>);
		}

		first_type get_result() const {
			return first;
		}

		explicit operator first_type() const {
			return get_result();
		}

		template<typename _OtherFml>
		formula_t<formula_plus<_Myformula, _OtherFml>> operator+(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_plus<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_minus<_Myformula, _OtherFml>> operator-(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_minus<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_multiplies<_Myformula, _OtherFml>> operator*(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_multiplies<_Myformula, _OtherFml>>(*this, _Right);
		}

		template<typename _OtherFml>
		formula_t<formula_divides<_Myformula, _OtherFml>> operator/(const formula_t<_OtherFml>& _Right) const {
			return formula_t<formula_divides<_Myformula, _OtherFml>>(*this, _Right);
		}
	};


	template<typename _Ty, typename _Fml = formula<_Ty, void, void>>
	using quantity = formula_t< _Fml >;

	template<typename _Fml> inline
	std::string to_string(const formula_t<_Fml>& _Left) {
		return _Left.to_string();
	}

	template<typename _Fml> inline
	typename _Fml::result_type get_result(const formula_t<_Fml>& _Left) {
		return _Left.get_result();
	}

	template<typename _Fml> inline
	std::ostream& operator<<(std::ostream& _Ostr, const formula_t<_Fml>& _Left) {
		return (_Ostr << _Left.to_string());
	}

#pragma endregion

#pragma region ratio_cast

	template<typename _ToRatio, typename _FromRatio, typename _Ty> inline
	_Ty ratio_cast(const _Ty& x) {
		using _CvtRatio = std::ratio_divide<_ToRatio, _FromRatio>;
		return x * ( static_cast<float_t>(_CvtRatio::num) / static_cast<float_t>(_CvtRatio::den) );// otherUnit / stdUnit
		/*<another>
		return x / ( static_cast<Ts>(_CvtRatio::num) / static_cast<Ts>(_CvtRatio::den) );// stdUnit / otherUnit
		</another>*/
		/*<unit-cast>
			<idea>unit-convert</idea>
			<example>
						1000[m]          1[km]         1[km]     X
				X[m] * ------- = X[m] * ------- = X * ------ = ------ [km]
						1[km]           1000[m]        1000     1000

						1000[m]       1000[m]
				X[km] * ------- = X * ------- = X*1000[m]
							1[km]           1
			</example>
			<formula>
				1[Lm] = 50[m] => std::ratio<1,50> = _FromRatio
				1[Rm] = 6[m]  => std::ratio<1,6>  = _ToRatio
							1[Lm]     1[Rm]         50     1[Rm]
				X[Lm] * ------- * ------ = X * ----- * -------
							50[m]     6[m]          1        6

				=> (X * (1/_FromRatio) * _ToRatio)[Rm]
				=> X * (_ToRatio/_FromRatio)[Rm]
			</formula>
		</unit-cast>*/
	}

#ifdef clmagic_calculation_general_RATIONAL_h_
	template<typename _Ty, typename _Ti>
	_Ty ratio_cast(const _Ty& x, rational<_Ti> _Ratio, rational<_Ti> _Dest_ratio) {
		using _Ts = scalar_t<_Ty>;
		return x * static_cast<_Ts>(_Dest_ratio / _Ratio);
		/*<another>
		return x / static_cast<_Ts>(_Dest_ratio / _Ratio);
		</another>*/
	}
#endif

#pragma endregion

#pragma region unit_cast

	template<typename _InQt, typename _OutQt>
	struct _Unit_cast {};
	
	// template recursion with _BinOp
	template<typename _InFml1, typename _InFml2, typename _InBinOp, 
		     typename _OutFml1, typename _OutFml2, typename _OutBinOp>
	struct _Unit_cast< formula_t<formula<_InFml1, _InFml2, _InBinOp>>, 
		               formula_t<formula<_OutFml1, _OutFml2, _OutBinOp>> > {
		using source_type      = formula_t<formula<_InFml1, _InFml2, _InBinOp>>;
		using destination_type = formula_t<formula<_OutFml1, _OutFml2, _OutBinOp>>;

		using _Divide_and_conquer_first  = _Unit_cast<typename source_type::first_type, typename destination_type::first_type>;
		using _Divide_and_conquer_second = _Unit_cast<typename source_type::second_type, typename destination_type::second_type>;

		void operator()(const source_type& _Source, destination_type& _Dest) const {
			_Divide_and_conquer_first()(_Source.first, _Dest.first);
			_Divide_and_conquer_second()(_Source.second, _Dest.second);
		}
	};
	
	// template recursion with _UnaryOp
	template<typename _InFml1, typename _InUnaryOp, typename _OutFml1, typename _OutUnaryOp>
	struct _Unit_cast< formula_t<formula<_InFml1, void, _InUnaryOp>>, 
		               formula_t<formula<_OutFml1, void, _OutUnaryOp>> > {
		using source_type      = formula_t<formula<_InFml1, void, _InUnaryOp>>;
		using destination_type = formula_t<formula<_OutFml1, void, _OutUnaryOp>>;

		using _Divide_and_conquer = _Unit_cast<typename source_type::first_type, typename destination_type::first_type>;

		void operator()(const source_type& _Source, destination_type& _Dest) const {
			_Divide_and_conquer()(_Source.first, _Source.first);
		}
	};
	
	// template until
	template<typename _InTy, typename _OutTy>
	struct _Unit_cast< formula_t<formula<_InTy, void, void>>, 
		               formula_t<formula<_OutTy, void, void>> > {
		using source_type      = formula_t<formula<_InTy, void, void>>;
		using destination_type = formula_t<formula<_OutTy, void, void>>;

		void operator()(const source_type& _Source, destination_type& _Dest) const {
			_Dest.first = static_cast<_OutTy>(_Source.first);
		}
	};
	
	// template until
	template<typename _InTy, typename _InUnit, typename _OutTy, typename _OutUnit>
	struct _Unit_cast< formula_t<formula<std::pair<_InTy,_InUnit>, void, void>>, 
		               formula_t<formula<std::pair<_OutTy,_OutUnit>, void, void>> > {
		using source_type      = formula_t<formula<std::pair<_InTy, _InUnit>,   void, void>>;
		using destination_type = formula_t<formula<std::pair<_OutTy, _OutUnit>, void, void>>;

		void operator()(const source_type& _Source, destination_type& _Dest) const {
			_Dest.first = static_cast<_OutTy>(
				ratio_cast<typename _OutUnit::ratio, typename _InUnit::ratio>(_Source.first)
			);
		}
	};

	template<typename _InQt, typename _OutQt> inline
	void unit_cast(const _InQt& _Source, _OutQt& _Dest) {
		_Unit_cast<_InQt, _OutQt>()(_Source, _Dest);
	}

	template<typename _OutQt, typename _InQt>
	_OutQt unit_cast(const _InQt& _Source) {
		_OutQt _Dest;
		unit_cast<_OutQt, _InQt>(_Source, _Dest);
		return _Dest;
	}

	template<typename _Fml> 
		template<typename _Fml2, std::enable_if_t<is_similar_formula_v<_Fml, _Fml2>, 
			int>> inline
	formula_t<_Fml>& formula_t<_Fml>::operator=(const formula_t<_Fml2>& _Right) {
		// assign with unit_cast
		unit_cast(_Right, *this);
		return *this;
	}

	template<typename _Uty, typename _UnaryOp>
		template<typename _Fml2, std::enable_if_t<is_similar_formula_v<formula<_Uty, void, _UnaryOp>, _Fml2>,
			int>> inline
	formula_t<formula<_Uty, void, _UnaryOp>>& formula_t<formula<_Uty, void, _UnaryOp>>::operator=(const formula_t<_Fml2>& _Right) {
		// assign with unit cast
		unit_cast(_Right, *this);
		return *this;
	}

	template<typename _Uty>
		template<typename _Fml2, std::enable_if_t<is_similar_formula_v<formula<_Uty, void, void>, _Fml2>,
			int>> inline
	formula_t<formula<_Uty, void, void>>& formula_t<formula<_Uty, void, void>>::operator=(const formula_t<_Fml2>& _Right) {
		// assign with unit cast
		unit_cast(_Right, *this);
		return *this;
	}

#pragma endregion


// 6. extend operation object ----------------------------------------------

	template<typename _Ty = void>
	struct sine {
		_Ty operator()(const _Ty& _Left) const {
			return sin(_Left);
		}
	};

	template<>
	struct sine<void> {
		template<typename _Ty>
		_Ty operator()(_Ty&& _Left) const {
			return sin(_Left);
		}
	};

	template<typename _Ty = void>
	struct cosine {
		_Ty operator()(const _Ty& _Left) const {
			return cos(_Left);
		}
	};

	template<>
	struct cosine<void> {
		template<typename _Ty>
		_Ty operator()(_Ty&& _Left) const {
			return cos(_Left);
		}
	};

	template<typename _Ty = void>
	struct tangent {
		_Ty operator()(const _Ty& _Left) const {
			return tan(_Left);
		}
	};

	template<>
	struct tangent<void> {
		template<typename _Ty>
		_Ty operator()(_Ty&& _Left) const {
			return tan(_Left);
		}
	};

	template<typename _Ty = void, bool _Is_scalar = std::is_scalar_v<_Ty>>
	struct scalar_product {
		template<typename _Ty>
		auto operator()(const _Ty& _Left, const _Ty& _Right) const -> decltype(_Left * _Right) {
			return _Left * _Right;
		}
	};

	template<typename _Ty>
	struct scalar_product<_Ty, false> {
		template<typename _Ty>
		auto operator()(const _Ty& _Left, const _Ty& _Right) const -> decltype(_Left.dot(_Right)) {
			return _Left.dot(_Right);
		}
	};

	template<>
	struct scalar_product<void, false> {
		template<typename _Ty1, typename _Ty2>
			requires requires(_Ty1 __v1, _Ty2 __v2) { __v1.dot(__v2); }
		auto operator()(_Ty1&& _Left, _Ty2&& _Right) const -> decltype(_Left.dot(_Right)) {
			return _Left.dot(_Right);
		}

		template<typename _Ty1, typename _Ty2>
		auto operator()(_Ty1&& _Left, _Ty2&& _Right) const -> decltype(_Left * _Right) {
			return _Left * _Right;
		}
	};

	template<typename _Ty> constexpr
	const char* operation_symbol_v<sine<_Ty>> = "sin(";

	template<typename _Ty> constexpr
	const char* operation_symbol_v<cosine<_Ty>> = "cos(";

	template<typename _Ty> constexpr
	const char* operation_symbol_v<tangent<_Ty>> = "tan(";

	template<typename _Ty> constexpr
	const char* operation_symbol_v<scalar_product<_Ty>> = " dot ";

}// namespace clmagic

#endif