#include "math_func.h"

namespace math
{

    Float erf(Float x) {
        Float a1 = (Float)  0.254829592;
        Float a2 = (Float) -0.284496736;
        Float a3 = (Float)  1.421413741;
        Float a4 = (Float) -1.453152027;
        Float a5 = (Float)  1.061405429;
        Float p  = (Float)  0.3275911;

        // Save the sign of x
        Float sign = math::signum(x);
        x = std::abs(x);

        // A&S formula 7.1.26
        Float t = (Float) 1.0 / ((Float) 1.0 + p*x);
        Float y = (Float) 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*std::exp(-x*x);

        return sign*y;
    }


    Float erfinv(Float x) {
        // Based on "Approximating the erfinv function" by Mark Giles
        Float w = -std::log(((Float) 1 - x)*((Float) 1 + x));
        Float p;
        if (w < (Float) 5) {
            w = w - (Float) 2.5;
            p = (Float) 2.81022636e-08;
            p = (Float) 3.43273939e-07 + p*w;
            p = (Float) -3.5233877e-06 + p*w;
            p = (Float) -4.39150654e-06 + p*w;
            p = (Float) 0.00021858087 + p*w;
            p = (Float) -0.00125372503 + p*w;
            p = (Float) -0.00417768164 + p*w;
            p = (Float) 0.246640727 + p*w;
            p = (Float) 1.50140941 + p*w;
        } else {
            w = std::sqrt(w) - (Float) 3;
            p = (Float) -0.000200214257;
            p = (Float) 0.000100950558 + p*w;
            p = (Float) 0.00134934322 + p*w;
            p = (Float) -0.00367342844 + p*w;
            p = (Float) 0.00573950773 + p*w;
            p = (Float) -0.0076224613 + p*w;
            p = (Float) 0.00943887047 + p*w;
            p = (Float) 1.00167406 + p*w;
            p = (Float) 2.83297682 + p*w;
        }
        return p*x;
    }

} //namespace math
