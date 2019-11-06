#pragma once
#ifndef SIMPLE_AUTO_DIFF_H__
#define SIMPLE_AUTO_DIFF_H__

#include <iostream>
#include <cmath>
#include <type_traits>
#include <Eigen/Dense>


namespace SimpleAD
{
	// Forward declarations
	template <typename T, int nder>	struct Scalar;
	template <typename T, int ndim, int nder> struct Array1D;
	template <typename T, int nrows, int ncols, int nder> struct Matrix;

	namespace meta
	{
		template <typename ValType, typename DerType, int nder>
		struct ScalarBase
		{
			typedef typename std::decay<ValType>::type ValTypeBase;
			static_assert(std::is_same<ValTypeBase, float>::value || std::is_same<ValTypeBase, double>::value, "Invalid value type");

			typedef Eigen::Array<ValTypeBase, nder, 1> DerBaseType;
			typedef ScalarBase<ValTypeBase, DerBaseType, nder> ReturnType;

			inline ScalarBase(const ValType &_val, const DerType &_der) : val(_val), der(_der) {}

			template <typename ValType2, typename DerType2>
			inline ScalarBase(const ScalarBase<ValType2, DerType2, nder> &var) : val(var.val), der(var.der) {
				static_assert(std::is_same<typename std::decay<ValType2>::type, ValTypeBase>::value, "Invalid scalar type conversion");
			}

			template <typename T>
			inline Scalar<T, nder> cast() const { return Scalar<T, nder>(static_cast<T>(val), der.template cast<T>()); }

			inline const ValTypeBase &grad(int idx = 0) const { return der(idx); }
			inline ValTypeBase &grad(int idx = 0) { return der(idx); }

			template <typename ValType2, typename DerType2>
			inline ScalarBase &operator=(const ScalarBase<ValType2, DerType2, nder> &var) {
				static_assert(std::is_same<typename std::decay<ValType2>::type, ValTypeBase>::value, "Invalid scalar type conversion");
				der = var.der;
				val = var.val;
				return *this;
			}

			inline ValTypeBase advance(ValTypeBase stepSize, int grad_idx = 0) const {
				return val + stepSize*der(grad_idx);
			}

			inline bool operator<(ValTypeBase a) const { return val < a; }
			inline bool operator>(ValTypeBase a) const { return val > a; }
			inline friend bool operator<(ValTypeBase a, const ScalarBase &var) { return a < var.val; }
			inline friend bool operator>(ValTypeBase a, const ScalarBase &var) { return a > var.val; }

			template <typename ValType2, typename DerType2>
			inline bool operator<(const ScalarBase<ValType2, DerType2, nder> &var) const {
				static_assert(std::is_same<typename std::decay<ValType2>::type, ValTypeBase>::value, "Invalid scalar type conversion");
				return val < var.val;
			}

			template <typename ValType2, typename DerType2>
			inline bool operator>(const ScalarBase<ValType2, DerType2, nder> &var) const {
				static_assert(std::is_same<typename std::decay<ValType2>::type, ValTypeBase>::value, "Invalid scalar type conversion");
				return val > var.val;
			}

			inline ReturnType operator-() const {
				return ReturnType(-val, -der);
			}

			template <typename ValType2, typename DerType2>
			inline ReturnType operator+(const ScalarBase<ValType2, DerType2, nder> &var) const {
				return ReturnType(val + var.val, der + var.der);
			}

			inline ReturnType operator+(ValTypeBase a) const {
				return ReturnType(val + a, der);
			}

			template <typename ValType2, typename DerType2>
			inline ScalarBase &operator+=(const ScalarBase<ValType2, DerType2, nder> &var) {
				der += var.der;
				val += var.val;
				return *this;
			}

			inline ScalarBase &operator+=(ValTypeBase a) {
				val += a;
				return *this;
			}

			friend inline ReturnType operator+(ValTypeBase a, const ScalarBase &var) {
				return ReturnType(a + var.val, var.der);
			}

			template <typename ValType2, typename DerType2>
			inline ReturnType operator-(const ScalarBase<ValType2, DerType2, nder> &var) const {
				return ReturnType(val - var.val, der - var.der);
			}

			inline ReturnType operator-(ValTypeBase a) const {
				return ReturnType(val - a, der);
			}

			template <typename ValType2, typename DerType2>
			inline ScalarBase &operator-=(const ScalarBase<ValType2, DerType2, nder> &var) {
				der -= var.der;
				val -= var.val;
				return *this;
			}

			inline ScalarBase &operator-=(ValTypeBase a) {
				val -= a;
				return *this;
			}

			friend inline ReturnType operator-(ValTypeBase a, const ScalarBase &var) {
				return ReturnType(a - var.val, -var.der);
			}

			template <typename ValType2, typename DerType2>
			inline ReturnType operator*(const ScalarBase<ValType2, DerType2, nder> &var) const {
				return ReturnType(val*var.val, der*var.val + val*var.der);
			}

			inline ReturnType operator*(ValTypeBase a) const {
				return ReturnType(val*a, der*a);
			}

			template <typename ValType2, typename DerType2>
			inline ScalarBase &operator*=(const ScalarBase<ValType2, DerType2, nder> &var) {
				der = der*var.val + val*var.der;
				val *= var.val;
				return *this;
			}

			inline ScalarBase &operator*=(ValTypeBase a) {
				der *= a;
				val *= a;
				return *this;
			}

			friend inline ReturnType operator*(ValTypeBase a, const ScalarBase &var) {
				return ReturnType(a*var.val, a*var.der);
			}

			template <typename ValType2, typename DerType2>
			inline ReturnType operator/(const ScalarBase<ValType2, DerType2, nder> &var) const {
				return ReturnType(val/var.val, (der*var.val - val*var.der)/(var.val*var.val));
			}

			inline ReturnType operator/(ValTypeBase a) const {
				return ReturnType(val/a, der/a);
			}

			template <typename ValType2, typename DerType2>
			inline ScalarBase &operator/=(const ScalarBase<ValType2, DerType2, nder> &var) {
				der = (der*var.val - val*var.der)/(var.val*var.val);
				val /= var.val;
				return *this;
			}

			inline ScalarBase &operator/=(ValTypeBase a) {
				der /= a;
				val /= a;
				return *this;
			}

			friend inline ReturnType operator/(ValTypeBase a, const ScalarBase &var) {
				return ReturnType(a/var.val, -a*var.der/(var.val*var.val));
			}

			inline ReturnType sin() const {
				return ReturnType(std::sin(val), der*std::cos(val));
			}

			inline ReturnType cos() const {
				return ReturnType(std::cos(val), -der*std::sin(val));
			}

			inline ReturnType tan() const {
				ValTypeBase tmp = std::cos(val);
				return ReturnType(std::tan(val), der/(tmp*tmp));
			}

			inline ReturnType asin() const {
				return ReturnType(std::asin(val), der/static_cast<ValTypeBase>(std::sqrt(1.0 - val*val)));
			}

			inline ReturnType acos() const {
				return ReturnType(std::acos(val), -der/static_cast<ValTypeBase>(std::sqrt(1.0 - val*val)));
			}

			inline ReturnType atan() const {
				return ReturnType(std::atan(val), der/static_cast<ValTypeBase>(1.0 + val*val));
			}

			inline ReturnType square() const {
				return ReturnType(val*val, static_cast<ValTypeBase>(2.0)*val*der);
			}

			inline ReturnType sqrt() const {
				ValTypeBase tmp = std::sqrt(val);
				return ReturnType(tmp, static_cast<ValTypeBase>(0.5)*der/tmp);
			}

			inline ReturnType pow(const ScalarBase &var) const {
				return (var*log()).exp();
			}

			inline ReturnType pow(ValTypeBase a) const {
				ValTypeBase tmp = std::pow(val, static_cast<ValTypeBase>(a - 1.0));
				return ReturnType(val*tmp, der*a*tmp);
			}

			inline ReturnType exp() const {
				return ReturnType(std::exp(val), der*std::exp(val));
			}

			inline ReturnType log() const {
				return ReturnType(std::log(val), der/val);
			}

			inline ReturnType abs() const {
				return ReturnType(std::abs(val), val > 0.0 ? der : -der);
			}

			friend inline std::ostream &operator<<(std::ostream &os, const ScalarBase &var) {
				os << "Scalar[Val: [" << var.val << "], Grad: [";
				for ( int i = 0; i < nder; ++i ) {
					os << var.der(i);
					if ( i + 1 < nder ) os << ", ";
				}
				os << "]]";
				return os;
			}

			ValType val;
			DerType der;
		}; //struct ScalarBase
	} // namespace meta


	template <typename T, int nder = 1>
	struct Scalar : public meta::ScalarBase<T, Eigen::Array<T, nder, 1>, nder>
	{
		static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "Invalid value type");

		typedef Eigen::Array<T, nder, 1> DerType;
		typedef meta::ScalarBase<T, DerType, nder> Parent;

		using Parent::val;
		using Parent::der;

		inline Scalar() : Parent(0.0f, DerType::Zero()) {}
		inline Scalar(T _val) : Parent(_val, DerType::Zero()) {}
		inline Scalar(T _val, T _der) : Parent(_val, DerType::Ones()*_der) {}
		inline Scalar(T _val, const DerType &_der) : Parent(_val, _der) {}

		template <typename ValType, typename DerType>
		inline Scalar(const meta::ScalarBase<ValType, DerType, nder> &var) : Parent(var) {}

		inline Scalar(T _val, const std::array<T, nder> &_der) : Parent(_val, DerType::Zero()) {
			for ( int i = 0; i < nder; ++i ) der[i] = _der[i];
		}

		inline void zero() { val = 0.0f; zeroGrad(); }
		inline void zeroGrad() { der.setZero(); }
		inline bool isZero(T Epsilon) const { return std::abs(val) > Epsilon; }
	};


	template <typename T, int nder = 1>
	struct ScalarRef : public meta::ScalarBase<T&, Eigen::Array<T, nder, 1>&, nder>
	{
		static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "Invalid value type");

		typedef Eigen::Array<T, nder, 1> &DerType;
		typedef meta::ScalarBase<T&, DerType, nder> Parent;

		using Parent::val;
		using Parent::der;

		inline ScalarRef(Scalar<T, nder> &var) : Parent(var.val, var.der) {}

		template <typename DerType>
		inline ScalarRef(T &_val, DerType _der) : Parent(_val, _der) {}

		template <typename ValType2, typename DerType2>
		inline ScalarRef &operator=(const meta::ScalarBase<ValType2, DerType2, nder> &var) {
			static_assert(std::is_same<typename std::decay<ValType2>::type, T>::value, "Invalid scalar type conversion");
			der = var.der;
			val = var.val;
			return *this;
		}

		inline void zero() { val = 0.0f; zeroGrad(); }
		inline void zeroGrad() { der.setZero(); }
		inline bool isZero(T Epsilon) const { return std::abs(val) > Epsilon; }
	};


	template <typename T, int nder = 1>
	struct ScalarConstRef : public meta::ScalarBase<const T&, const Eigen::Array<T, nder, 1>&, nder>
	{
		static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "Invalid value type");

		typedef const Eigen::Array<T, nder, 1>& DerType;
		typedef meta::ScalarBase<const T&, DerType, nder> Parent;

		using Parent::val;
		using Parent::der;

		inline ScalarConstRef(Scalar<T, nder> &var) : Parent(var) {}
		inline ScalarConstRef(ScalarRef<T, nder> &var) : Parent(var) {}

		template <typename DerType>
		inline ScalarConstRef(T& _val, DerType _der) : Parent(_val, _der) {}

		inline bool isZero(T Epsilon) const { return std::abs(val) > Epsilon; }
	};


	template <typename T, int ndim, int nder = 1>
	struct Array1D
	{
		typedef Eigen::Array<T, 1, ndim> ValType;
		typedef Eigen::Array<T, nder, ndim> DerType;

		typedef Scalar<T, nder> ElementType;

		typedef Eigen::Block<DerType, nder, 1, nder != 1> ElementDerRefType;
		typedef Eigen::Block<const DerType, nder, 1, nder != 1> ElementDerConstRefType;

		typedef Eigen::Block<DerType, 1, ndim, ndim != 1 && nder == 1> AllDerRefType;
		typedef Eigen::Block<const DerType, 1, ndim, ndim != 1 && nder == 1> AllDerConstRefType;

		typedef meta::ScalarBase<T&, ElementDerRefType, nder> ElementRefType;
		typedef meta::ScalarBase<const T&, ElementDerConstRefType, nder> ElementConstRefType;

		inline Array1D() : val(ValType::Zero()), der(DerType::Zero()) {}
		inline Array1D(const ValType &_val) : val(_val), der(DerType::Zero()) {}
		inline Array1D(const ValType &_val, const DerType &_der) : val(_val), der(_der) {}
		inline Array1D(const Array1D &arr) : val(arr.val), der(arr.der) {}

		inline Array1D(const ElementType &_x, const ElementType &_y) {
			static_assert(ndim == 2);
			(*this)[0] = _x; (*this)[1] = _y;
		}

		inline Array1D(const ElementType &_x, const ElementType &_y, const ElementType &_z) {
			static_assert(ndim == 3);
			(*this)[0] = _x; (*this)[1] = _y; (*this)[2] = _z;
		}

		inline Array1D(const std::array<ElementType, ndim> &arr) {
			for ( int i = 0; i < ndim; ++i ) (*this)[i] = arr[i];
		}

		template <typename T2>
		inline Array1D<T2, ndim, nder> cast() const {
			return Array1D<T2, ndim, nder>(val.template cast<T2>(), der.template cast<T2>());
		}

		inline ValType advance(T stepSize, int grad_idx = 0) const {
			return val + stepSize*grad(grad_idx);
		}

		inline void fill(const ElementType &var) {
			der.colwise() = var.der;
			val = ValType::Ones()*var.val;
		}

		//inline ElementDerRefType eleGrad(int ele_idx)				{ return der.col(ele_idx); }
		//inline ElementDerConstRefType eleGrad(int ele_idx) const	{ return der.col(ele_idx); }

		inline AllDerRefType grad(int grad_idx = 0)					{ return der.row(grad_idx); }
		inline AllDerConstRefType grad(int grad_idx = 0)	const	{ return der.row(grad_idx); }

		inline void zero() { val.setZero(); zeroGrad(); }
		inline void zeroGrad() { der.setZero(); }
		inline bool isZero(T Epsilon) const { return val.isZero(Epsilon); }

		inline ElementRefType operator[](int idx) { return ElementRefType(val(idx), der.col(idx)); }
		inline ElementConstRefType operator[](int idx) const { return ElementConstRefType(val(idx), der.col(idx)); }

		inline ElementRefType operator()(int idx) { return ElementRefType(val(idx), der.col(idx)); }
		inline ElementConstRefType operator()(int idx) const { return ElementConstRefType(val(idx), der.col(idx)); }

		inline Array1D operator-() const {
			return Array1D(-val, -der);
		}

		inline Array1D operator+(const Array1D &arr) const {
			return Array1D(val + arr.val, der + arr.der);
		}

		inline Array1D operator+(const ValType &arr) const {
			return Array1D(val + arr, der);
		}

		inline Array1D operator+(const ElementType &a) const {
			return Array1D(val + a.val, der.colwise() + a.der);
		}

		inline Array1D operator+(T a) const {
			return Array1D(val + a, der);
		}

		inline Array1D &operator+=(const Array1D &arr) {
			der += arr.der;
			val += arr.val;
			return *this;
		}

		inline Array1D &operator+=(const ValType &arr) {
			val += arr;
			return *this;
		}

		inline Array1D &operator+=(const ElementType &a) {
			der.colwise() += a.der;
			val += a.val;
			return *this;
		}

		inline Array1D &operator+=(T a) {
			val += a;
			return *this;
		}

		friend inline Array1D operator+(const ValType &arr0, const Array1D &arr1) {
			return Array1D(arr0 + arr1.val, arr1.der);
		}

		friend inline Array1D operator+(const ElementType &a, const Array1D &arr) {
			return Array1D(a.val + arr.val, arr.der.colwise() + a.der);
		}

		friend inline Array1D operator+(T a, const Array1D &arr) {
			return Array1D(a + arr.val, arr.der);
		}

		inline Array1D operator-(const Array1D &arr) const {
			return Array1D(val - arr.val, der - arr.der);
		}

		inline Array1D operator-(const ValType &arr) const {
			return Array1D(val - arr, der);
		}

		inline Array1D operator-(const ElementType &a) const {
			return Array1D(val - a.val, der.colwise() - a.der);
		}

		inline Array1D operator-(T a) const {
			return Array1D(val - a, der);
		}

		inline Array1D &operator-=(const Array1D &arr) {
			der -= arr.der;
			val -= arr.val;
			return *this;
		}

		inline Array1D &operator-=(const ValType &arr) {
			val -= arr;
			return *this;
		}

		inline Array1D &operator-=(const ElementType &a) {
			der.colwise() -= a.der;
			val -= a.val;
			return *this;
		}

		inline Array1D &operator-=(T a) {
			val -= a;
			return *this;
		}

		friend inline Array1D operator-(const ValType &arr0, const Array1D &arr1) {
			return Array1D(arr0 - arr1.val, -arr1.der);
		}

		friend inline Array1D operator-(const ElementType &a, const Array1D &arr) {
			return Array1D(a.val - arr.val, -(arr.der.colwise() - a.der));
		}

		friend inline Array1D operator-(T a, const Array1D &arr) {
			return Array1D(a - arr.val, -arr.der);
		}

		inline Array1D operator*(const Array1D &arr) const {
			return Array1D(val*arr.val, der.rowwise()*arr.val + arr.der.rowwise()*val);
		}

		inline Array1D operator*(const ValType &arr) const {
			return Array1D(val*arr, der.rowwise()*arr);
		}

		inline Array1D operator*(const ElementType &a) const {
			return Array1D(val*a.val, der*a.val + (a.der.matrix()*val.matrix()).array());
		}

		inline Array1D operator*(T a) const {
			return Array1D(val*a, der*a);
		}

		inline Array1D &operator*=(const Array1D &arr) {
			der = der.rowwise()*arr.val + arr.der.rowwise()*val;
			val *= arr.val;
			return *this;
		}

		inline Array1D &operator*=(const ValType &arr) {
			der.rowwise() *= arr;
			val *= arr;
			return *this;
		}

		inline Array1D &operator*=(const ElementType &a) {
			der = der*a.val + (a.der.matrix()*val.matrix()).array();
			val *= a.val;
			return *this;
		}

		inline Array1D &operator*=(T a) {
			der *= a;
			val *= a;
			return *this;
		}

		friend inline Array1D operator*(const ValType &arr0, const Array1D &arr1) {
			return Array1D(arr0*arr1.val, arr1.der.rowwise()*arr0);
		}

		friend inline Array1D operator*(const ElementType &a, const Array1D &arr) {
			return Array1D(a.val*arr.val, (a.der.matrix()*arr.val.matrix()).array() + a.val*arr.der);
		}

		friend inline Array1D operator*(T a, const Array1D &arr) {
			return Array1D(a*arr.val, a*arr.der);
		}

		inline Array1D operator/(const Array1D &arr) const {
			return Array1D(val/arr.val, (der.rowwise()*arr.val - arr.der.rowwise()*val).rowwise()/arr.val.square());
		}

		inline Array1D operator/(const ValType &arr) const {
			return Array1D(val/arr, der.rowwise()/arr);
		}

		inline Array1D operator/(const ElementType &a) const {
			return Array1D(val/a.val, (der*a.val - (a.der.matrix()*val.matrix()).array())/(a.val*a.val));
		}

		inline Array1D operator/(T a) const {
			return Array1D(val/a, der/a);
		}

		inline Array1D &operator/=(const Array1D &arr) {
			der = (der.rowwise()*arr.val - arr.der.rowwise()*val).rowwise()/arr.val.square();
			val /= arr.val;
			return *this;
		}

		inline Array1D &operator/=(const ValType &arr) {
			der.rowwise() /= arr;
			val /= arr;
			return *this;
		}

		inline Array1D &operator/=(const ElementType &a) {
			der = (der*a.val - (a.der.matrix()*val.matrix()).array())/(a.val*a.val);
			val /= a.val;
			return *this;
		}

		inline Array1D &operator/=(T a) {
			der /= a;
			val /= a;
			return *this;
		}

		friend inline Array1D operator/(const ElementType &a, const Array1D &arr) {
			return Array1D(a.val/arr.val, ((a.der.matrix()*arr.val.matrix()).array() - arr.der*a.val).rowwise()/arr.val.square());
		}

		friend inline Array1D operator/(const ValType &arr0, const Array1D &arr1) {
			return Array1D(arr0/arr1.val, (-arr1.der).rowwise()*(arr0/arr1.val.square()));
		}

		friend inline Array1D operator/(T a, const Array1D &arr) {
			return Array1D(a/arr.val, (-a*arr.der).rowwise()/arr.val.square());
		}

		inline Array1D sin() const {
			return Array1D(val.sin(), der.rowwise()*val.cos());
		}

		inline Array1D cos() const {
			return Array1D(val.cos(), -(der.rowwise()*val.sin()));
		}

		inline Array1D tan() const {
			return Array1D(val.tan(), der.rowwise()/val.cos().square());
		}

		inline Array1D asin() const {
			return Array1D(val.asin(), der.rowwise()/(static_cast<T>(1.0) - val.square()).sqrt());
		}

		inline Array1D acos() const {
			return Array1D(val.acos(), -(der.rowwise()/(static_cast<T>(1.0) - val.square()).sqrt()));
		}

		inline Array1D atan() const {
			return Array1D(val.atan(), der.rowwise()/(static_cast<T>(1.0) + val.square()));
		}

		inline Array1D square() const {
			return Array1D(val*val, static_cast<T>(2.0)*(der.rowwise()*val));
		}

		inline Array1D sqrt() const {
			ValType tmp = val.sqrt();
			return Array1D(tmp, static_cast<T>(0.5)*(der.rowwise()/tmp));
		}

		inline Array1D pow(const Array1D &arr) const {
			return (arr*log()).exp();
		}

		inline Array1D pow(const ElementType &a) const {
			return (a*log()).exp();
		}

		inline Array1D pow(T x) const {
			ValType tmp = val.pow(static_cast<T>(x - 1.0));
			return Array1D(val*tmp, x*(der.rowwise()*tmp));
		}

		inline Array1D exp() const {
			ValType tmp = val.exp();
			return Array1D(tmp, der.rowwise()*tmp);
		}

		inline Array1D log() const {
			return Array1D(val.log(), der.rowwise()/val);
		}

		inline Array1D abs() const {
			return Array1D(val.abs(), der.rowwise()*val.sign());
		}

		inline ElementType sum() const {
			return ElementType(val.sum(), der.rowwise().sum());
		}

		friend inline std::ostream &operator<<(std::ostream &os, const Array1D &arr) {
			os << "Array[\n";
			for ( int i = 0; i < ndim; ++i )
				os << "  " << arr[i] << '\n';
			os << "]\n";
			return os;
		}

		ValType val;
		DerType der;
	}; //struct Array


	template <typename T, int nrows, int ncols, int nder = 1>
	struct Matrix
	{
		typedef Eigen::Matrix<T, nrows, ncols> ValType;
		typedef Eigen::Matrix<T, nrows, ncols*nder, Eigen::ColMajor> DerType;

		typedef Eigen::Block<DerType, nrows, ncols> DerRefType;
		typedef Eigen::Block<const DerType, nrows, ncols> ConstDerRefType;

		typedef Eigen::Map<Eigen::Array<T, nder, 1>, Eigen::Unaligned, Eigen::InnerStride<nrows*ncols>
						  > ElementDerRefType;

		typedef Eigen::Map<const Eigen::Array<T, nder, 1>, Eigen::Unaligned, Eigen::InnerStride<nrows*ncols>
						  > ElementDerConstRefType;

		typedef Scalar<T, nder> ElementType;
		typedef meta::ScalarBase<T&, ElementDerRefType, nder> ElementRefType;
		typedef meta::ScalarBase<const T&, ElementDerConstRefType, nder> ElementConstRefType;

		inline Matrix() { zero(); }
		inline Matrix(const ValType &_val) : val(_val) { zeroGrad();  }
		inline Matrix(const ValType &_val, const DerType &_der) : val(_val), der(_der) {}
		inline Matrix(const Matrix &mat) : val(mat.val), der(mat.der) {}

		inline Matrix(const ValType &_val, const std::array<ValType, nder> &_der) : val(_val) {
			for ( int i = 0; i < nder; ++i ) grad(i) = _der[i];
		}

		inline Matrix(const ElementType &_x, const ElementType &_y) {
			static_assert((nrows == 1 || ncols == 1) && nrows*ncols == 2);
			(*this)(0) = _x; (*this)(1) = _y;
		}

		inline Matrix(const ElementType &_x, const ElementType &_y, const ElementType &_z) {
			static_assert((nrows == 1 || ncols == 1) && nrows*ncols == 3);
			(*this)(0) = _x; (*this)(1) = _y; (*this)(2) = _z;
		}

		inline Matrix(const std::array<ElementType, nrows*ncols> &data) {
			for ( int i = 0; i < nrows*ncols; ++i ) (*this)(i) = data[i];
		}

		inline ValType advance(T stepSize, int grad_idx = 0) const {
			return val + stepSize*grad(grad_idx);
		}

		template <typename T2>
		inline Matrix<T2, nrows, ncols, nder> cast() const {
			return Matrix<T2, nrows, ncols, nder>(val.template cast<T2>(), der.template cast<T2>());
		}

		inline void zero() { val.setZero(); zeroGrad(); }
		inline void zeroGrad() { der.setZero(); }
		inline bool isZero(T Epsilon) const { return val.isZero(Epsilon); }

		inline ConstDerRefType grad(int idx = 0) const { return der.template block<nrows, ncols>(0, idx*ncols); }
		inline DerRefType grad(int idx = 0) { return der.template block<nrows, ncols>(0, idx*ncols); }

		inline ElementConstRefType x() const {
			static_assert(nrows == 1 || ncols == 1);
			return (*this)(0);
		}

		inline ElementRefType x() {
			static_assert(nrows == 1 || ncols == 1);
			return (*this)(0);
		}

		inline ElementConstRefType y() const {
			static_assert((nrows == 1 || ncols == 1) && nrows*ncols >= 2);
			return (*this)(1);
		}

		inline ElementRefType y() {
			static_assert((nrows == 1 || ncols == 1) && nrows*ncols >= 2);
			return (*this)(1);
		}

		inline ElementConstRefType z() const {
			static_assert((nrows == 1 || ncols == 1) && nrows*ncols >= 3);
			return (*this)(2);
		}

		inline ElementRefType z() {
			static_assert((nrows == 1 || ncols == 1) && nrows*ncols >= 3);
			return (*this)(2);
		}

		inline ElementConstRefType operator()(int idx) const {
			return ElementConstRefType(val(idx), ElementDerConstRefType(der.data() + idx));
		}

		inline ElementRefType operator()(int idx) {
			return ElementRefType(val(idx), ElementDerRefType(der.data() + idx));
		}

		inline ElementConstRefType operator()(int i, int j) const {
			return ElementConstRefType(val(i, j), ElementDerConstRefType(der.data() + j*nrows + i));
		}

		inline ElementRefType operator()(int i, int j) {
			return ElementRefType(val(i, j), ElementDerRefType(der.data() + j*nrows + i));
		}

		inline Matrix operator-() const {
			return Matrix(-val, -der);
		}

		inline Matrix operator+(const Matrix &var) const {
			return Matrix(val + var.val, der + var.der);
		}

		inline Matrix operator+(const ValType &m) const {
			return Matrix(val + m, der);
		}

		inline Matrix operator+(const ElementType &a) const {
			Matrix ret(val + ValType::Ones()*a.val, der);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) += ValType::Ones()*a.grad(i);
			return ret;
		}

		inline Matrix operator+(T a) const {
			return Matrix(val + ValType::Ones()*a, der);
		}

		inline Matrix &operator+=(const Matrix &var) {
			der += var.der;
			val += var.val;
			return *this;
		}

		inline Matrix &operator+=(const ValType &m) {
			val += m;
			return *this;
		}

		inline Matrix &operator+=(const ElementType &a) {
			for ( int i = 0; i < nder; ++i ) grad(i) += ValType::Ones()*a.grad(i);
			val += ValType::Ones()*a.val;
			return *this;
		}

		inline Matrix &operator+=(T a) {
			val += ValType::Ones()*a;
			return *this;
		}

		friend inline Matrix operator+(const ValType &m, const Matrix &var) {
			return Matrix(m + var.val, var.der);
		}

		friend inline Matrix operator+(const ElementType &a, const Matrix &var) {
			Matrix ret(ValType::Ones()*a.val + var.val);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = ValType::Ones()*a.grad(i) + var.grad(i);
			return ret;
		}

		friend inline Matrix operator+(T a, const Matrix &var) {
			return Matrix(ValType::Ones()*a + var.val, var.der);
		}

		inline Matrix operator-(const Matrix &var) const {
			Matrix ret(val - var.val, der);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) -= var.grad(i);
			return ret;
		}

		inline Matrix operator-(const ValType &m) const {
			return Matrix(val - m, der);
		}

		inline Matrix operator-(const ElementType &a) const {
			Matrix ret(val - ValType::Ones()*a.val, der);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) -= ValType::Ones()*a.grad(i);
			return ret;
		}

		inline Matrix operator-(T a) const {
			return Matrix(val - ValType::Ones()*a, der);
		}

		inline Matrix &operator-=(const Matrix &var) {
			for ( int i = 0; i < nder; ++i ) grad(i) -= var.grad(i);
			val -= var.val;
			return *this;
		}

		inline Matrix &operator-=(const ValType &m) {
			val -= m;
			return *this;
		}

		inline Matrix &operator-=(const ElementType &a) {
			for ( int i = 0; i < nder; ++i ) grad(i) -= ValType::Ones()*a.grad(i);
			val -= ValType::Ones()*a.val;
			return *this;
		}

		inline Matrix &operator-=(T a) {
			val -= ValType::Ones()*a;
			return *this;
		}

		friend inline Matrix operator-(const ValType &m, const Matrix &var) {
			return Matrix(m - var.val, -var.der);
		}

		friend inline Matrix operator-(const ElementType &a, const Matrix &var) {
			Matrix ret(ValType::Ones()*a.val - var.val);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = ValType::Ones()*a.grad(i) - var.grad(i);
			return ret;
		}

		friend inline Matrix operator-(T a, const Matrix &var) {
			return Matrix(ValType::Ones()*a - var.val, -var.der);
		}

		inline Matrix operator*(const ElementType &a) const {
			Matrix ret(val*a.val);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = grad(i)*a.val + val*a.grad(i);
			return ret;
		}

		inline Matrix operator*(T a) const {
			return Matrix(val*a, der*a);
		}

		inline Matrix &operator*=(const ElementType &a) {
			for ( int i = 0; i < nder; ++i ) grad(i) = grad(i)*a.val + val*a.grad(i);
			val *= a.val;
			return *this;
		}

		inline Matrix &operator*=(T a) {
			der *= a;
			val *= a;
			return *this;
		}

		friend inline Matrix operator*(const ElementType &a, const Matrix &var) {
			Matrix ret(a.val*var.val);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = a.grad(i)*var.val + a.val*var.grad(i);
			return ret;
		}

		template <int ncols2>
		inline Matrix<T, nrows, ncols2, nder> operator*(const Matrix<T, ncols, ncols2, nder> &var) const {
			Matrix<T, nrows, ncols2, nder> ret(val*var.val);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = grad(i)*var.val + val*var.grad(i);
			return ret;
		}

		inline Matrix &operator*=(const Matrix &var) {
			static_assert(nrows == ncols);
			for ( int i = 0; i < nder; ++i ) grad(i) = grad(i)*var.val + val*var.grad(i);
			val *= var.val;
			return *this;
		}

		inline Matrix &operator*=(const ValType &m) {
			static_assert(nrows == ncols);
			for ( int i = 0; i < nder; ++i ) grad(i) = grad(i)*m;
			val *= m;
			return *this;
		}

		inline Matrix operator/(const ElementType &a) const {
			Matrix ret(val/a.val);
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = (grad(i)*a.val - val*a.grad(i))/(a.val*a.val);
			return ret;
		}

		inline Matrix operator/(T a) const {
			return Matrix(val/a, der/a);
		}

		inline Matrix &operator/=(const ElementType &a) {
			for ( int i = 0; i < nder; ++i ) grad(i) = (grad(i)*a.val - val*a.grad(i))/(a.val*a.val);
			val /= a.val;
			return *this;
		}

		inline Matrix &operator/=(T a) {
			der /= a;
			val /= a;
			return *this;
		}

		inline ElementType dot(const Matrix &var) const {
			ElementType ret(val.dot(var.val));
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = grad(i).dot(var.val) + val.dot(var.grad(i));
			return ret;
		}

		inline Matrix cross(const Matrix &var) const {
			Matrix ret(val.cross(var.val));
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = grad(i).cross(var.val) + val.cross(var.grad(i));
			return ret;
		}

		inline ElementType squaredNorm() const {
			ElementType ret(val.squaredNorm());
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = static_cast<T>(2.0)*val.cwiseProduct(grad(i)).sum();
			return ret;
		}

		inline ElementType norm() const {
			return squaredNorm().sqrt();
		}

		inline Matrix normalized() const {
			return (*this)/norm();
		}

		inline void normalize() {
			(*this) /= norm();
		}

		inline ElementType sum() const {
			ElementType ret(val.sum());
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = grad(i).sum();
			return ret;
		}

		inline Matrix<T, ncols, nrows, nder> transpose() const {
			Matrix<T, ncols, nrows, nder> ret(val.transpose());
			for ( int i = 0; i < nder; ++i ) ret.grad(i) = grad(i).transpose();
			return ret;
		}

		friend inline std::ostream &operator<<(std::ostream &os, const Matrix &var) {
			os << "Matrix[\n  Val: [\n";
			for ( int i = 0; i < nrows; ++i ) {
				os << "    ";
				for ( int j = 0; j < ncols; ++j ) os << var.val(i, j) << ", ";
				os << '\n';
			}
			os << "  ],\n";
			for ( int k = 0; k < nder; ++k ) {
				os << "  Grad: [\n";
				for ( int i = 0; i < nrows; ++i ) {
					os << "    ";
					for ( int j = 0; j < ncols; ++j ) os << var.grad(k)(i, j) << ", ";
					os << '\n';
				}
				os << "  ]\n";
			}
			os << "]\n";
			return os;
		}

		inline Array1D<T, nrows*ncols, nder> toArray1D() const {
			Array1D<T, nrows*ncols, nder> ret;
			for ( int j = 0; j < ncols; ++j )
				for ( int i = 0; i < nrows; ++i )
					ret[i + j*nrows] = (*this)(i, j);
			return ret;
		}

		ValType val;
		DerType der;
	}; //struct Matrix


	typedef Scalar<float>			Scalarf;
	typedef Scalar<double>			Scalard;
	typedef ScalarRef<float>		ScalarReff;
	typedef ScalarRef<double>		ScalarRefd;
	typedef ScalarConstRef<float>	ScalarConstReff;
	typedef ScalarConstRef<double>	ScalarConstRefd;

	typedef Array1D<float, 2>		Array2f;
	typedef Array1D<double, 2>		Array2d;
	typedef Array1D<float, 3>		Array3f;
	typedef Array1D<double, 3>		Array3d;

	typedef Matrix<float, 2, 1>		Vector2f;
	typedef Matrix<double, 2, 1>	Vector2d;
	typedef Matrix<float, 3, 1>		Vector3f;
	typedef Matrix<double, 3, 1>	Vector3d;
	typedef Matrix<float, 4, 1>		Vector4f;
	typedef Matrix<double, 4, 1>	Vector4d;

	typedef Matrix<float, 2, 2>		Matrix2f;
	typedef Matrix<double, 2, 2>	Matrix2d;
	typedef Matrix<float, 3, 3>		Matrix3f;
	typedef Matrix<double, 3, 3>	Matrix3d;
	typedef Matrix<float, 4, 4>		Matrix4f;
	typedef Matrix<double, 4, 4>	Matrix4d;
} //namespace SimpleAD


#endif //SIMPLE_AUTO_DIFF_H__
