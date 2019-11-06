#pragma once
#ifndef EMITTER_H__
#define EMITTER_H__

#include "fwd.h"
#include "ptr.h"

struct Intersection;
struct IntersectionAD;

struct Emitter {
    Emitter() : shape_id(-1) {}
    Emitter(int shape_id) : shape_id(shape_id) {}

    virtual Spectrum eval(const Intersection &its, const Vector &d) const = 0;
    virtual Spectrum eval(const Vector& norm, const Vector &d) const = 0;
    virtual Float evalDirection(const Vector& norm, const Vector& d) const = 0;
    virtual Float sampleDirection(const Array2 &rnd, Vector& dir) const = 0;

    virtual SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &d) const {
        assert(false);
        return SpectrumAD();
    }

    virtual SpectrumAD evalAD(const VectorAD &norm, const VectorAD &d) const {
        assert(false);
        return SpectrumAD();
    }

    virtual inline int getShapeID() const { return -1; }
    virtual Spectrum getIntensity() const = 0;
    virtual std::string toString() const {
        std::ostringstream oss;
        oss << "Emitter [ ]" << std::endl;
        return oss.str();
    }

    int shape_id;
};

#endif //EMITTER_H__
