#pragma once
#ifndef AREA_EMITTER_H__
#define AREA_EMITTER_H__

#include "emitter.h"

struct AreaLight : Emitter {
    AreaLight() {}

    AreaLight(int shape_id, const ptr<float> intensity_data, bool two_sided = false)
    : Emitter(shape_id), two_sided(two_sided) {
        intensity(0) = intensity_data[0];
        intensity(1) = intensity_data[1];
        intensity(2) = intensity_data[2];
    }

    AreaLight(int shape_id, const Spectrum3f &intensity, bool two_sided = false)
    : Emitter(shape_id), intensity(intensity.cast<Float>()), two_sided(two_sided) {}

    inline int getShapeID() const { return shape_id; }
    Spectrum getIntensity() const { return intensity; }

    // Return the radiant emittance for the given surface intersection
    Spectrum eval(const Intersection &its, const Vector &d) const;
    SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &d) const;
    Spectrum eval(const Vector &norm, const Vector &d) const;
    SpectrumAD evalAD(const VectorAD &norm, const VectorAD &d) const;

    Float evalDirection(const Vector& norm, const Vector& d) const;
    Float sampleDirection(const Array2 &rnd, Vector& dir) const;

    std::string toString() const {
        std::ostringstream oss;
        oss << "AreaLight[" << std::endl
            << "  intensity = " << "(" << intensity(0) << "," << intensity(1) << "," << intensity(2) << ")"
            << "  shape_id = " << shape_id << std::endl
            << "]" << std::endl;
        return oss.str();
    }


    Spectrum intensity;
    bool two_sided;
};

#endif //AREA_EMITTER_H__
