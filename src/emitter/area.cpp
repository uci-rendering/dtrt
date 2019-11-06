#include "area.h"
#include "intersection.h"
#include "intersectionAD.h"

Spectrum AreaLight::eval(const Intersection &its, const Vector &d) const {
    return (two_sided || its.geoFrame.n.dot(d) > 0) ? intensity : Spectrum::Zero();
}

SpectrumAD AreaLight::evalAD(const IntersectionAD &its, const VectorAD &d) const {
    return (two_sided || its.geoFrame.n.val.dot(d.val) > 0) ? SpectrumAD(intensity) : SpectrumAD();
}

Spectrum AreaLight::eval(const Vector &norm, const Vector &d) const {
        return (two_sided || norm.dot(d) > 0) ? intensity : Spectrum::Zero();
}

SpectrumAD AreaLight::evalAD(const VectorAD &norm, const VectorAD &d) const {
        return (two_sided || norm.val.dot(d.val) > 0) ? SpectrumAD(intensity) : SpectrumAD();
}

Float AreaLight::evalDirection(const Vector& norm, const Vector& d) const {
	Float dp = norm.dot(d);
	return std::max((Float)0.0, dp);
}

Float AreaLight::sampleDirection(const Array2 &rnd, Vector& dir) const {
	 dir = squareToCosineHemisphere(Vector2(rnd[0], rnd[1]));
	 return M_PI;
}