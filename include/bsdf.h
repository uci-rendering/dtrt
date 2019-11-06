#pragma once
#ifndef BSDF_H__
#define BSDF_H__

#include <sstream>
#include <string>
#include "fwd.h"

struct Intersection;
struct IntersectionAD;


struct BSDF {
    // Evaluate the cosine weighted BSDF value
    virtual Spectrum eval(const Intersection &its, const Vector &wo, bool importance = false) const = 0;
    virtual SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance = false) const;

    // Sample the BSDF and return the BSDF value divided by pdf
    virtual Spectrum sample(const Intersection &its, const Array3 &sample, Vector &wo, Float &pdf, Float &eta, bool importance = false) const = 0;
    virtual SpectrumAD sampleAD(const IntersectionAD &its, const Array3 &sample, VectorAD &wo, Float &pdf, Float &eta, bool importance = false) const;

    // Compute the probability of sampling wo (given wi)
    virtual Float pdf(const Intersection &its, const Vector3 &wo) const = 0;
    virtual Float pdf(const IntersectionAD &its, const Vector3 &wo) const;

    // Check if the BSDF is transmissive
    virtual bool isTransmissive() const = 0;

    // Check if the BSDF is null
    virtual bool isNull() const = 0;

    /// Return a readable string representation of this BSDF
    virtual std::string toString() const;
};

#endif
