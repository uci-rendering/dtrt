#include "bsdf.h"
#include "intersection.h"
#include "intersectionAD.h"


SpectrumAD BSDF::evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance) const {
    assert(false);
    return SpectrumAD();
}

SpectrumAD BSDF::sampleAD(const IntersectionAD &its, const Array3 &rnd, VectorAD &wo, Float &pdf, Float &eta, bool importance) const {
    SpectrumAD ret;
    Intersection its0 = its.toIntersection();
    Vector wo0;
    if ( !sample(its0, rnd, wo0, pdf, eta).isZero(Epsilon) )
        if ( pdf > Epsilon ) {
            wo = its.toLocal(its0.toWorld(wo0));
            ret = evalAD(its, wo)/pdf;
        }
    return ret;
}

Float BSDF::pdf(const IntersectionAD &its, const Vector3 &wo) const { return pdf(its.toIntersection(), wo);}

std::string BSDF::toString() const {
    std::ostringstream oss;
    oss << "Base BSDF []" << std::endl;
    return oss.str();
}
