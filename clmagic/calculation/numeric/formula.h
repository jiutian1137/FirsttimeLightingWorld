//--------------------------------------------------------------------------------------
// Copyright (c) 2020 LongJiangnan
// All Rights Reserved
// Free License
//
// Table of contents
// 1. formula<_Fml1, _Fml2, _BinOp> ------------------------------------------------
// 2. clean_formula<_Fml> ----------------------------------------------------------
// 3. formula operation ------------------------------------------------------------
// 4. formula util -----------------------------------------------------------------
// 5. formula exception ------------------------------------------------------------
//--------------------------------------------------------------------------------------
#pragma once
#include <functional>

/*<Version1.0>
			formula_1 = (_Ty | formula)
				|
				|
formula --- operation
				|
				|
			formula_2 = (_Ty | formula)
</Version1.0>*/

/*<Version2.0>
			formula_1 = (_Ty | formula | formula_result)
				|
				|
formula --- operation
				|
				|
			formula_2 = (_Ty | formula | formula_result)
</Version2.0>*/

/*<Version3.0>
new linear_formula<_Begin, _Ty1, _Op1, _Begin, _Ty2, _Op2, _Ty3, _End, _End, ...>
</Version3.0>*/

namespace calculation {

	template<typename _Ty1, typename _Ty2>
	struct _Binary_template {};

	template<typename _Ty1, typename _Ty2, typename _Ty3>
	struct _Ternary_template {};

	template<typename _Ty1, typename _UnaryOp>
	using _Unary_result_t = decltype( _UnaryOp()(_Ty1()) );

	template<typename _Ty1, typename _Ty2, typename _BinOp>
	using _Binary_result_t = decltype( _BinOp()(_Ty1(), _Ty2()) );

#pragma region formula<_Fml1, _Fml2, _BinOp>

	// template recursion with _BinOp(_Fml1, _Fml2)
	template<typename _Ty1, typename _Ty2 = void, typename _BinOp = void>
	struct formula {
		//static_assert( !( (!std::is_void_v<_Fml2>) & std::is_void_v<_BinOp> ), "clmagic::error" );

		using first_formula  = typename formula<_Ty1, void, void>::result_formula;
		using first_type     = typename first_formula::result_type;
		using first_unit     = typename first_formula::result_unit;

		using second_formula = typename formula<_Ty2, void, void>::result_formula;;
		using second_type    = typename second_formula::result_type;
		using second_unit    = typename second_formula::result_unit;

		using operation      = _BinOp;

		using result_formula = formula<_Ty1, _Ty2, _BinOp>;// <recursion requires>
		using result_type    = _Binary_result_t<first_type, second_type, operation>;
		using result_unit    = _Ternary_template<first_unit, second_unit, operation>;

		result_type operator()(const first_type& _Left, const second_type& _Right) const {
			return operation()(_Left, _Right);
		}
	};
	
	// template recursion with _UnaryOp(_Fml1)
	template<typename _Fml1, typename _UnaryOp>
	struct formula<_Fml1, void, _UnaryOp> {
		using first_formula  = typename formula<_Fml1, void, void>::result_formula;
		using first_type     = typename first_formula::result_type;
		using first_unit     = typename first_formula::result_unit;

		using operation      = _UnaryOp;

		using result_formula = formula<_Fml1, void, _UnaryOp>;// <recursion requires>
		using result_type    = _Unary_result_t<first_type, operation>;
		using result_unit    = first_unit;// wait fix

		result_type operator()(const first_type& _Left) const {
			return operation()(_Left);
		}
	};
	
	// template recursion with optimize
	template<typename _Fml1, typename _Fml2, typename _BinOp>
	struct formula< formula<_Fml1, _Fml2, _BinOp>, void, void >
		: public formula<_Fml1, _Fml2, _BinOp> {};

	// template until
	template<typename _Ty>
	struct formula<_Ty, void, void> {
		using first_formula  = formula<_Ty, void, void>;
		using first_type     = _Ty;
		using first_unit     = void;

		using operation      =  void;

		using result_formula = formula<_Ty, void, void>;
		using result_type    = _Ty;
		using result_unit    = void;

		_Ty operator()(const _Ty& _Left) const {
			return _Left;
		}
	};
	
	// template until
	template<typename _Ty, typename _Unit>
	struct formula<std::pair<_Ty, _Unit>, void, void> {
		using first_formula  = formula<std::pair<_Ty, _Unit>, void, void>;
		using first_type     = _Ty;
		using first_unit     = _Unit;

		using operation      =  void;

		using result_formula = formula<std::pair<_Ty, _Unit>, void, void>;
		using result_type    = _Ty;
		using result_unit    = _Unit;

		_Ty operator()(const _Ty& _Left) const {
			return _Left;
		}
	};

#pragma endregion

#pragma region formula_clean<_Fml>

	template<typename _Ty/* = void*/>
	struct _Formula_clean {
		using type = formula<_Ty, void, void>;
	};

	template<typename _Fml1, typename _Fml2, typename _BinOp> 
	struct _Formula_clean< formula<_Fml1, _Fml2, _BinOp> > {
		using type = formula<_Fml1, _Fml2, _BinOp>;
	};

	template<typename _Fml1, typename _Fml2, typename _BinOp>
	struct _Formula_clean< formula<formula<_Fml1, void, void>, formula<_Fml2, void, void>, _BinOp> > {
		using type = formula<_Fml1, _Fml2, _BinOp>;
	};

	template<typename _Fml11, typename _Fml12, typename _BinOp1, typename _Fml2, typename _BinOp>
	struct _Formula_clean< formula<formula<_Fml11, _Fml12, _BinOp1>, formula<_Fml2, void, void>, _BinOp> > {
		using type = formula< formula<_Fml11, _Fml12, _BinOp1>, _Fml2, _BinOp >;
	};

	template<typename _Fml1, typename _Fml21, typename _Fml22, typename _BinOp2, typename _BinOp>
	struct _Formula_clean< formula<formula<_Fml1, void, void>, formula<_Fml21, _Fml22, _BinOp2>, _BinOp> > {
		using type = formula< _Fml1, formula<_Fml21, _Fml22, _BinOp2>, _BinOp >;
	};

	template<typename _Fml>
	using formula_clean = typename _Formula_clean<_Fml>::type;

#pragma endregion

#pragma region formula operation

	template<typename _Xt>
	using formula_negate = formula_clean< formula<_Xt, void, std::negate<>> >;

	template<typename _Xt, typename _Yt>
	using formula_plus   = formula_clean< formula<_Xt, _Yt, std::plus<>> >;

	template<typename _Xt, typename _Yt>
	using formula_minus  = formula_clean< formula<_Xt, _Yt, std::minus<>> >;

	template<typename _Xt, typename _Yt>
	using formula_multiplies = formula_clean< formula<_Xt, _Yt, std::multiplies<>> >;

	template<typename _Xt, typename _Yt>
	using formula_divides = formula_clean< formula<_Xt, _Yt, std::divides<>> >;

	template<typename _Xt, typename _Yt>
	using formula_modulus = formula_clean< formula<_Xt, _Yt, std::modulus<>> >;

#pragma endregion

#pragma region formula util

	template<typename _Fml1, typename _Fml2>
	struct _Is_similar_formula {
		static constexpr bool value = std::is_same_v<_Fml1, _Fml2>;
	};
	
	// template recursion
	template<typename _Ty11, typename _Ty12, typename _BinOp1, typename _Ty21, typename _Ty22, typename _BinOp2>
	struct _Is_similar_formula<formula<_Ty11, _Ty12, _BinOp1>, formula<_Ty21, _Ty22, _BinOp2>> {
		static constexpr bool value = std::is_same_v<_BinOp1, _BinOp2> 
									&& _Is_similar_formula<_Ty11, _Ty21>::value 
									&& _Is_similar_formula<_Ty12, _Ty22>::value;
	};
	
	// template until
	template<typename _Ty1, typename _Unit1, typename _Ty2, typename _Unit2>
	struct _Is_similar_formula<std::pair<_Ty1, _Unit1>, std::pair<_Ty2, _Unit2>> {
		static constexpr bool value = std::is_same_v<_Ty1, _Ty2>
									&& std::is_same_v<typename _Unit1::unit_category, typename _Unit2::unit_category>;
	};
	
	// template until, false
	template<typename _Ty1, typename _Unit1, typename _Ty2>
	struct _Is_similar_formula<std::pair<_Ty1, _Unit1>, _Ty2> {
		static constexpr bool value = false;
	};
	
	// template until, false
	template<typename _Ty1, typename _Ty2, typename _Unit2>
	struct _Is_similar_formula<_Ty1, std::pair<_Ty2, _Unit2>> {
		static constexpr bool value = false;
	};

	template<typename _Fml1, typename _Fml2> constexpr
	bool is_similar_formula_v = _Is_similar_formula<_Fml1, _Fml2>::value;


	template<typename _Ty>
	struct _Is_formula {
		static constexpr bool value = false;
	};
	
	template<typename _Fml1, typename _Fml2, typename _BinOp>
	struct _Is_formula<formula<_Fml1, _Fml2, _BinOp>> {
		static constexpr bool value = true;
	};

	template<typename _Ty> constexpr
	bool is_formula_v = _Is_formula<_Ty>::value;

#pragma endregion

#pragma region formula exception

	template<typename _Fml>
	struct _Is_correct_formula {
		static constexpr bool value = true;
	};
	// template recursion
	template<typename _Fml1, typename _Fml2, typename _BinOp>
	struct _Is_correct_formula<formula<_Fml1, _Fml2, _BinOp>> {
		static constexpr bool value = _Is_correct_formula<_Fml1>::value && _Is_correct_formula<_Fml2>::value;
	};
	// template until
	template<typename _Fml2, typename _UnaryOp>
	struct _Is_correct_formula<formula<void, _Fml2, _UnaryOp>> {
		static constexpr bool value = false;

		using error_formula = formula<void, _Fml2, _UnaryOp>;
		static constexpr const char* error_message = "_Fml1 cannot be <void>";
	};
	// template until
	template<typename _Op>
	struct _Is_correct_formula<formula<void, void, _Op>> {
		static constexpr bool value = false;

		using error_formula = formula<void, void, _Op>;
		static constexpr const char* error_message = "_Fml1 cannot be <void>";
	};
	// template until
	template<typename _Fml1, typename _Fml2>
	struct _Is_correct_formula<formula<_Fml1, _Fml2, void>> {
		static constexpr bool value = false;

		using error_formula = formula<_Fml1, _Fml2, void>;
		static constexpr const char* error_message = "template<OK,OK,ERROR>";
	};
	// template value
	template<typename _Fml> constexpr
	bool is_correct_formula_v = is_formula_v<_Fml> && _Is_correct_formula<_Fml>::value;
	template<typename _Fml> constexpr
	bool is_error_formula_v = !is_correct_formula_v<_Fml>;

#pragma endregion

}// namespace clmagic