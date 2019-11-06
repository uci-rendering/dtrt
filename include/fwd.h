#pragma once
#ifndef FWD_H__
#define FWD_H__

#include <Eigen/Dense>
#include "simpleAD.h"
#include "config.h"

// Specify calculation precision
#define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
typedef double      Float;
#define Epsilon     1e-7
#define M_PI        3.14159265358979323846
#define INV_PI      0.31830988618379067154
#define INV_TWOPI   0.15915494309189534561
#define INV_FOURPI  0.07957747154594766788
#else
typedef float       Float;
#define Epsilon     1e-4f
#define M_PI        3.14159265358979323846f
#define INV_PI      0.31830988618379067154f
#define INV_TWOPI   0.15915494309189534561f
#define INV_FOURPI  0.07957747154594766788f
#endif

typedef Eigen::Matrix<Float, 2, 1>              Vector2;
typedef Eigen::Matrix<float, 2, 1>              Vector2f;
typedef Eigen::Matrix<double, 2, 1>             Vector2d;
typedef Eigen::Matrix<int, 2, 1>                Vector2i;

typedef Eigen::Array<Float, 2, 1>               Array2;
typedef Eigen::Array<float, 2, 1>               Array2f;
typedef Eigen::Array<double, 2, 1>              Array2d;
typedef Eigen::Array<int, 2, 1>                 Array2i;

typedef Eigen::Matrix<Float, 3, 1>              Vector;
typedef Eigen::Matrix<Float, 3, 1>              Vector3;
typedef Eigen::Matrix<float, 3, 1>              Vector3f;
typedef Eigen::Matrix<double, 3, 1>             Vector3d;
typedef Eigen::Matrix<int, 3, 1>                Vector3i;

typedef Eigen::Array<Float, 3, 1>               Array;
typedef Eigen::Array<Float, 3, 1>               Array3;
typedef Eigen::Array<float, 3, 1>               Array3f;
typedef Eigen::Array<double, 3, 1>              Array3d;
typedef Eigen::Array<int, 3, 1>                 Array3i;

typedef Eigen::Matrix<Float, 4, 1>              Vector4;
typedef Eigen::Matrix<float, 4, 1>              Vector4f;
typedef Eigen::Matrix<double, 4, 1>             Vector4d;
typedef Eigen::Matrix<int, 4, 1>                Vector4i;

typedef Eigen::Array<Float, 4, 1>               Array4;
typedef Eigen::Array<float, 4, 1>               Array4f;
typedef Eigen::Array<double, 4, 1>              Array4d;
typedef Eigen::Array<int, 4, 1>                 Array4i;

typedef Eigen::Matrix<Float, 3, 3>              Matrix3x3;
typedef Eigen::Matrix<float, 3, 3>              Matrix3x3f;
typedef Eigen::Matrix<double, 3, 3>             Matrix3x3d;
typedef Eigen::Matrix<Float, 4, 4>              Matrix4x4;
typedef Eigen::Matrix<float, 4, 4>              Matrix4x4f;
typedef Eigen::Matrix<double, 4, 4>             Matrix4x4d;

typedef Array3                                  Spectrum;
typedef Array3f                                 Spectrum3f;
typedef Array3d                                 Spectrum3d;

typedef SimpleAD::Scalar<Float, nder>           FloatAD;

typedef SimpleAD::Array1D<Float, 2, nder>       Array2AD;
typedef SimpleAD::Array1D<float, 2, nder>       Array2fAD;
typedef SimpleAD::Array1D<double, 2, nder>      Array2dAD;

typedef SimpleAD::Array1D<Float, 3, nder>       ArrayAD;
typedef SimpleAD::Array1D<Float, 3, nder>       Array3AD;
typedef SimpleAD::Array1D<float, 3, nder>       Array3fAD;
typedef SimpleAD::Array1D<double, 3, nder>      Array3dAD;

typedef SimpleAD::Matrix<Float, 2, 1, nder>     Vector2AD;
typedef SimpleAD::Matrix<float, 2, 1, nder>     Vector2fAD;
typedef SimpleAD::Matrix<double, 2, 1, nder>    Vector2dAD;

typedef SimpleAD::Matrix<Float, 3, 1, nder>     VectorAD;
typedef SimpleAD::Matrix<Float, 3, 1, nder>     Vector3AD;
typedef SimpleAD::Matrix<float, 3, 1, nder>     Vector3fAD;
typedef SimpleAD::Matrix<double, 3, 1, nder>    Vector3dAD;

typedef SimpleAD::Matrix<Float, 3, 3, nder>     Matrix3x3AD;
typedef SimpleAD::Matrix<float, 3, 3, nder>     Matrix3x3fAD;
typedef SimpleAD::Matrix<double, 3, 3, nder>    Matrix3x3dAD;

typedef Array3AD                                SpectrumAD;
typedef Array3fAD                               Spectrum3fAD;
typedef Array3dAD                               Spectrum3dAD;


#endif //FWD_H__
