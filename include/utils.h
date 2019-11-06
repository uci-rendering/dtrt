#pragma once
#ifndef UTILITY_H__
#define UTILITY_H__

#include "fwd.h"
#include "ray.h"
#include "rayAD.h"

template <typename T>
inline T clamp(const T &v, const T &lo, const T &hi) {
    if (v < lo) return lo;
    else if (v > hi) return hi;
    else return v;
}

template <typename T> inline T square(const T &x) { return x * x; }
template <typename T> inline T cubic(const T &x) { return x * x * x; }

Vector3 xfm_point(const Matrix4x4 &T, const Vector3& pos);
Vector3 xfm_vector(const Matrix4x4 &T, const Vector3& vec);

Vector2 squareToUniformDiskConcentric(const Vector2 &sample);
Vector squareToCosineHemisphere(const Vector2 &sample);
Vector squareToUniformSphere(const Vector2 &sample);
Float squareToCosineHemispherePdf(const Vector &d);

void coordinateSystem(const Vector &n, Vector &s, Vector &t);
void coordinateSystemAD(const VectorAD &n, VectorAD &s, VectorAD &t);

Float luminance(const Vector &v);
bool isPixelValueValid(const Spectrum3f &val);
void progressIndicator(Float progress);

Float fresnelDielectricExt(Float cosThetaI, Float &cosThetaT, Float eta);
FloatAD fresnelDielectricExtAD(const FloatAD &cosThetaI, FloatAD &cosThetaT, const FloatAD &eta);

inline Float fresnelDielectricExt(Float cosThetaI, Float eta) {
    Float cosThetaT;
    return fresnelDielectricExt(cosThetaI, cosThetaT, eta);
}

inline FloatAD fresnelDielectricExtAD(const FloatAD &cosThetaI, const FloatAD &eta) {
    FloatAD cosThetaT;
    return fresnelDielectricExtAD(cosThetaI, cosThetaT, eta);
}

inline Vector reflect(const Vector &wi, const Vector &n) { return 2.0f*wi.dot(n)*n - wi; }

Vector refract(const Vector &wi, const Vector &n, Float eta, Float cosThetaT);
Vector refract(const Vector &wi, const Vector &n, Float eta);
Vector refract(const Vector &wi, const Vector &n, Float eta, Float &cosThetaT, Float &F);

Array rayIntersectTriangle(const Vector &v0, const Vector &v1, const Vector &v2, const Ray &ray);
ArrayAD rayIntersectTriangleAD(const VectorAD &v0, const VectorAD &v1, const VectorAD &v2, const RayAD &ray);

Float computeIntersectionInTri(const Vector3& a, const Vector3& b0, const Vector3& c0, const Vector3& b1, const Vector3& c1, Float t0);

#endif //UTILITY_H__
