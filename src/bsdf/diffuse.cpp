#include "diffuse.h"
#include "intersection.h"
#include "intersectionAD.h"
#include "utils.h"
#include <assert.h>

Spectrum DiffuseBSDF::eval(const Intersection &its, const Vector &wo, bool importance) const {
    if (its.wi.z() < Epsilon || wo.z() < Epsilon)
        return Spectrum::Zero();
    else
        return reflectance.val*INV_PI*wo.z();
}


SpectrumAD DiffuseBSDF::evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance) const {
    if (its.wi.z().val < Epsilon || wo.z().val < Epsilon)
        return SpectrumAD();
    else
        return reflectance*INV_PI*wo.z();
}


Spectrum DiffuseBSDF::sample(const Intersection &its, const Array3 &rnd, Vector &wo, Float &pdf, Float &eta, bool importance) const {
    if (its.wi.z() < Epsilon)
        return Spectrum::Zero();
    wo = squareToCosineHemisphere(Vector2(rnd[0], rnd[1]));
    eta = 1.0f;
    pdf = squareToCosineHemispherePdf(wo);
    return reflectance.val;
}


Float DiffuseBSDF::pdf(const Intersection &its, const Vector &wo) const{
    if (its.wi.z() < Epsilon || wo.z() < Epsilon)
        return 0.0;
    else
        return squareToCosineHemispherePdf(wo);
}
