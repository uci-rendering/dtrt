#pragma once
#ifndef BSDF_NULL_H__
#define BSDF_NULL_H__

#include "bsdf.h"

struct NullBSDF: BSDF {
    NullBSDF() {}
    Spectrum eval(const Intersection &its, const Vector &wo, bool importance = false) const;
    SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance = false) const;
    Spectrum sample(const Intersection &its, const Array3 &sample, Vector &wo, Float &pdf, Float &eta, bool importance = false) const;
    SpectrumAD sampleAD(const IntersectionAD &its, const Array3 &rnd, VectorAD &wo, Float &pdf, Float &eta, bool importance = false) const;
    Float pdf(const Intersection &its, const Vector &wo) const;
    bool isTransmissive() const { return true; }
    bool isNull() const { return true; }
    std::string toString() const {
        std::ostringstream oss;
        oss << "BSDF_null []" << std::endl;
        return oss.str();
    }
};

#endif
