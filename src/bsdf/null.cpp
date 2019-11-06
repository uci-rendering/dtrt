#include "null.h"
#include "intersection.h"
#include "intersectionAD.h"

Spectrum NullBSDF::eval(const Intersection &its, const Vector &wo, bool importance) const{
    return Spectrum::Ones();
}

SpectrumAD NullBSDF::evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance) const{
    return SpectrumAD(Spectrum::Ones());
}

Spectrum NullBSDF::sample(const Intersection &its, const Array3 &rnd, Vector &wo, Float &pdf, Float &eta, bool importance) const {
    wo = -its.wi;
    pdf = 1.0;
    eta = 1.0;
    return Spectrum::Ones();
}

SpectrumAD NullBSDF::sampleAD(const IntersectionAD &its, const Array3 &rnd, VectorAD &wo, Float &pdf, Float &eta, bool importance) const {
    wo = -its.wi;
    pdf = 1.0;
    eta = 1.0;
    return SpectrumAD(Spectrum::Ones());
}

Float NullBSDF::pdf(const Intersection &its, const Vector &wo) const{
    return 1.0;
}
