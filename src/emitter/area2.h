#pragma once
#ifndef AREA_EMITTER_EXTENDED_H__
#define AREA_EMITTER_EXTENDED_H__

#include "emitter.h"

struct AreaLightEx : Emitter {
    AreaLightEx() {}

    AreaLightEx(int shape_id, const Spectrum3f &intensity, float kappa)
        : Emitter(shape_id), intensity(intensity.transpose().cast<Float>()), kappa(kappa) {}

    AreaLightEx(int shape_id, const Spectrum3f &intensity, float kappa, ptr<float> dIntensity, ptr<float> dKappa)
        : Emitter(shape_id)
        , intensity(intensity.transpose().cast<Float>(), Eigen::Map<Eigen::Array<float, nder, 3, Eigen::RowMajor> >(dIntensity.get(), nder, 3).cast<Float>())
        , kappa(kappa, Eigen::Map<Eigen::Array<float, nder, 1> >(dKappa.get(), nder, 1).cast<Float>()) {}

    inline int getShapeID() const { return shape_id; }
    Spectrum getIntensity() const { return intensity.val.transpose(); }

    // Return the radiant emittance for the given surface intersection
    Spectrum eval(const Intersection &its, const Vector &d) const;
    SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &d) const;
    Spectrum eval(const Vector &norm, const Vector &d) const;
    SpectrumAD evalAD(const VectorAD &norm, const VectorAD &d) const;

    Float evalDirection(const Vector& norm, const Vector& d) const;
    Float sampleDirection(const Array2 &rnd, Vector& dir) const;

    std::string toString() const {
        std::ostringstream oss;
        oss << "AreaLightEx[" << std::endl
            << "  intensity = " << "(" << intensity(0) << "," << intensity(1) << "," << intensity(2) << ")\n"
            << "  kappa = " << kappa.val << "\n"
            << "  shape_id = " << shape_id << "\n"
            << "]" << std::endl;
        return oss.str();
    }

    SpectrumAD intensity;
    FloatAD kappa;
};

#endif //AREA_EMITTER_EXTENDED_H__
