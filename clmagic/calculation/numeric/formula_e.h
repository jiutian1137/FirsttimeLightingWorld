//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Free License
//--------------------------------------------------------------------------------------
#pragma once
#include "formula_t.h"

namespace calculation {

	template<typename _Fml>
	struct formula_result : public _Fml { };

	template<typename _Fml>
	struct formula<formula_result<_Fml>, void, void> {
		using result_formula = formula_result<_Fml>;
		using result_type    = typename result_formula::result_type;
		using result_unit    = typename result_formula::result_unit;
	};


	template<typename _Unit>
	struct _Result_unit_symbol {
		std::string operator()() const {
			return std::string(unit_symbol_v<_Unit>);
		}
	};

	// _Xu {+-*/} _Yu = _Xu ? _Yu
	template<typename _Xu, typename _Yu, typename _BinOp>
	struct _Result_unit_symbol<_Ternary_template<_Xu, _Yu, _BinOp>> {
		std::string operator()() const {
			return _Result_unit_symbol<_Xu>()() + operation_symbol_v<_BinOp>  + _Result_unit_symbol<_Yu>()();
		}
	};

	// _Xu {+-*/} _Number = _Xu
	template<typename _Xu, typename _BinOp>
	struct _Result_unit_symbol<_Ternary_template<_Xu, void, _BinOp>> {
		std::string operator()() const {
			return _Result_unit_symbol<_Xu>()();
		}
	};

	// _Number {+-*/} _Yu = _Yu
	template<typename _Yu, typename _BinOp>
	struct _Result_unit_symbol<_Ternary_template<void, _Yu, _BinOp>> {
		std::string operator()() const {
			return _Result_unit_symbol<_Yu>()();
		}
	};


	template<typename _Fml>
	struct _Formula_result_ratio {
		using ratio = std::ratio<1, 1>;
	};

	template<typename _Ty1, typename _Un1, typename _Ty2, typename _Un2, typename _BinOp/* = std::multiplies<>*/>
	struct _Formula_result_ratio<formula<std::pair<_Ty1, _Un1>, std::pair<_Ty2, _Un2>, _BinOp>> {
		using ratio = std::ratio_multiply<typename _Un1::ratio, typename _Un2::ratio>;
	};

	template<typename _Ty1, typename _Un1, typename _Ty2, typename _Un2>
	struct _Formula_result_ratio<formula<std::pair<_Ty1, _Un1>, std::pair<_Ty2, _Un2>, std::divides<>>> {
		using ratio = std::ratio_divide<typename _Un1::ratio, typename _Un2::ratio>;
	};

	template<typename _Fml>
	using formula_result_ratio = typename _Formula_result_ratio<_Fml>::ratio;


	// template until formula_t to formula_result_t
	template<typename _InTy, typename _InUnit, typename _OutTy, typename _OutUnit>
	struct _Unit_cast< formula_t< formula<std::pair<_InTy,_InUnit>, void, void> >, 
					   formula_t< formula_result< formula<std::pair<_OutTy,_OutUnit>, void, void> > > > {
		using source_type      = formula_t<                 formula<std::pair<_InTy, _InUnit>,  void, void> >;
		using destination_type = formula_t< formula_result< formula<std::pair<_OutTy,_OutUnit>, void, void> > >;

		void operator()(const source_type& _Source, destination_type& _Dest) const {
			// unit_cast to _Temp result
			formula_t<formula<std::pair<_OutTy, _OutUnit>, void, void>> _Temp;
			_Unit_cast< formula_t<formula<std::pair<_InTy, _InUnit>, void, void>>, decltype(_Temp) >()(_Source, _Temp);
			// _Temp result copy to _Dest.result_value
			_Dest.result_value = _Temp.get_result();
		}
	};

	// template recursion formula_result_t to formula_result_t
	template<typename _InTy1, typename _InTy2, typename _InOp, typename _OutTy1, typename _OutTy2, typename _OutOp>
	struct _Unit_cast< formula_t< formula_result<formula<_InTy1, _InTy2, _InOp>> >,
					   formula_t< formula_result<formula<_OutTy1,_OutTy2,_OutOp>> > > {
		using source_type      = formula_t< formula_result<formula<_InTy1, _InTy2, _InOp>> >;
		using destination_type = formula_t< formula_result<formula<_OutTy1,_OutTy2,_OutOp>> >;

		void operator()(const source_type& _Source, destination_type& _Dest) const {
			_Dest.result_value = ratio_cast<formula_result_ratio<formula<_OutTy1, _OutTy2, _OutOp>>,
											formula_result_ratio<formula<_InTy1, _InTy2, _InOp>>>(_Source.result_value);
		}
	};


	// template recursion with funknown
	template<typename _Fml>
	class formula_t< formula_result<_Fml> > {
	public:
		using result_type = typename formula_result<_Fml>::result_type;
		using result_unit = typename formula_result<_Fml>::result_unit;
		
		result_type result_value;

		formula_t() = default;
		formula_t(const formula_t&) = default;

		template<typename ..._Tys, std::enable_if_t<std::is_constructible_v<result_type, _Tys&&...> , int> = 0>
		formula_t(_Tys&&... _Args) : result_value(std::forward<_Tys>(_Args)...) {}

		formula_t& operator=(const formula_t&) = default;

		template<typename _Ty, std::enable_if_t<std::is_convertible_v<_Ty, result_type>, int> = 0>
		formula_t& operator=(const _Ty& _Arg) { 
			result_value = static_cast<result_type>(_Arg); 
			return *this; 
		}
	
		result_type get_result() const {
			return result_value;
		}

		std::string to_string() const {
			using std::to_string;
			return to_string(result_value) + _Unit_symbol_style(_Result_unit_symbol<result_unit>()());
		}
	};

}// namespace clmagic