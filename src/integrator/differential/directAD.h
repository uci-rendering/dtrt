#pragma once
#ifndef DIRECT_INTEGRATOR_AD_H__
#define DIRECT_INTEGRATOR_AD_H__

#include "integratorAD.h"

struct Ray;
struct RayAD;

struct DirectIntegratorAD : IntegratorAD {
    SpectrumAD radianceAD(const Scene& scene, RndSampler* sampler, const RayAD& ray, int nEdgeSamples) const;
    Spectrum radiance(const Scene& scene, RndSampler* sampler, Ray ray) const;

    Spectrum pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const;
    SpectrumAD pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const;

    std::string getName() const { return "directAD"; }
};

#endif //DIRECT_INTEGRATOR_AD_H__
