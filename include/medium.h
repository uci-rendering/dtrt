#pragma once
#ifndef MEDIUM_H__
#define MEDIUM_H__

#include "fwd.h"

struct Ray;
struct RayAD;
struct RndSampler;

struct Medium {
    Medium(int phase_id):phase_id(phase_id) {}

    virtual bool sampleDistance(const Ray& ray, const Float &tmax, const Array2 &rnd2, RndSampler* sampler,
                                Vector& p_scatter, Spectrum& throughput) const = 0;

    virtual bool sampleDistanceAD(const RayAD& ray, const FloatAD &tmax, const Array2 &rnd2, RndSampler* sampler,
                                  VectorAD& p_scatter, SpectrumAD& throughput) const
    {
        assert(false);
        return false;
    }

    virtual Float evalTransmittance(const Ray& ray, const Float &tmin, const Float &tmax, RndSampler* sampler) const = 0;

    virtual FloatAD evalTransmittanceAD(const RayAD& ray, const FloatAD &tmin, const FloatAD &tmax, RndSampler* sampler) const
    {
        assert(false);
        return FloatAD();
    }

    virtual bool isHomogeneous() const = 0;

    virtual Spectrum sigS(const Vector& x) const = 0;

    int phase_id;
};

#endif //MEDIUM_H__
