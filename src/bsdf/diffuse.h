#pragma once
#ifndef BSDF_DIFFUSE_H__
#define BSDF_DIFFUSE_H__

#include "bsdf.h"
#include "ptr.h"

struct DiffuseBSDF: BSDF {
    inline DiffuseBSDF(const Spectrum3f& reflectance): reflectance(reflectance.cast<Float>()) {}
    inline DiffuseBSDF(const Spectrum3f& reflectance, ptr<float> dReflectance): reflectance(reflectance.cast<Float>()) {
        initVelocities(Eigen::Map<Eigen::Array<float, nder, 3, Eigen::RowMajor> >(dReflectance.get(), nder, 3).cast<Float>());
    }
    inline void initVelocities(const Eigen::Array<Float, nder, 3> &der) { reflectance.der = der; }

    Spectrum eval(const Intersection &its, const Vector &wo, bool importance = false) const;
    SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance = false) const;
    Spectrum sample(const Intersection &its, const Array3 &rnd, Vector &wo, Float &pdf, Float &eta, bool importance=false) const;
    Float pdf(const Intersection &its, const Vector &wo) const;

    inline bool isTransmissive() const { return false; }
    inline bool isNull() const { return false; }
    std::string toString() const {
        std::ostringstream oss;
        oss << "BSDF_diffuse [" << std::endl
            << "  reflectance = " << "(" << reflectance(0) << "," << reflectance(1) << "," << reflectance(2) << ")" << std::endl
            << "]" << std::endl;
        return oss.str();
    }

    SpectrumAD reflectance;
};

#endif
