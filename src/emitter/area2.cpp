#include "area2.h"
#include "intersection.h"
#include "intersectionAD.h"

Spectrum AreaLightEx::eval(const Intersection &its, const Vector &d) const {
    Float tmp;
    if ( (tmp = its.shFrame.n.dot(d)) > Epsilon )
        return intensity.val*kappa.val*INV_TWOPI*std::exp(kappa.val*(tmp - 1.0f));
    else
        return Spectrum::Zero();
}

SpectrumAD AreaLightEx::evalAD(const IntersectionAD &its, const VectorAD &d) const {
    FloatAD tmp;
    if ( (tmp = its.shFrame.n.dot(d)) > Epsilon )
        return intensity*((kappa*(tmp - 1.0f)).exp()*kappa*INV_TWOPI);
    else
        return SpectrumAD();
}

Spectrum AreaLightEx::eval(const Vector &norm, const Vector &d) const {
    Float tmp;
    if ( (tmp = norm.dot(d)) > Epsilon )
        return intensity.val*kappa.val*INV_TWOPI*std::exp(kappa.val*(tmp - 1.0f));
    else
        return Spectrum::Zero();
};

SpectrumAD AreaLightEx::evalAD(const VectorAD &norm, const VectorAD &d) const {
    FloatAD tmp;
    if ( (tmp = norm.dot(d)) > Epsilon )
        return intensity*((kappa*(tmp - 1.0f)).exp()*kappa*INV_TWOPI);
    else
        return SpectrumAD();
};

Float AreaLightEx::evalDirection(const Vector& norm, const Vector& d) const {
    Float dp = norm.dot(d);
    return std::max((Float)0.0, dp);
}

Float AreaLightEx::sampleDirection(const Array2 &rnd, Vector& dir) const {
     dir = squareToCosineHemisphere(Vector2(rnd[0], rnd[1]));
     return M_PI;
}