#pragma once
#ifndef MATH_H__
#define MATH_H__

#include "fwd.h"
#include <cmath>

namespace math
{

    inline void sincos(float theta, float &sinTheta, float &cosTheta) {
#ifdef _MSC_VER
        sinTheta = std::sin(theta);
        cosTheta = std::cos(theta);
#else
        ::sincosf(theta, &sinTheta, &cosTheta);
#endif
    }

    inline void sincos(double theta, double &sinTheta, double &cosTheta) {
#ifdef _MSC_VER
        sinTheta = std::sin(theta);
        cosTheta = std::cos(theta);
#else
        ::sincos(theta, &sinTheta, &cosTheta);
#endif
    }

    inline Float signum(Float value) {
    #if defined(SINGLE_PRECISION)
        return copysignf((float) 1.0, value);
    #elif defined(DOUBLE_PRECISION)
        return copysign((double) 1.0, value);
    #endif
    }

    Float erf(Float x);
    Float erfinv(Float x);

} //namespace math

#endif //MATH_H__
