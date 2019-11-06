#pragma once
#ifndef PATH_TRACER_AD_H__
#define PATH_TRACER_AD_H__

#include "integratorAD.h"

struct Ray;
struct RayAD;
struct IntersectionAD;

struct PathTracerAD : IntegratorAD {
    Spectrum Li(const Scene& scene, RndSampler* sampler, const Ray& ray, int max_depth) const;
    SpectrumAD LiAD(const Scene& scene, const IntersectionAD& its, RndSampler* sampler, const RayAD& ray,
                    int max_depth, EMode mode, bool incEmission, int nEdgeSamples) const;

    Spectrum pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const;
    SpectrumAD pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const;

    std::string getName() const { return "pathAD"; }
};

#endif //PATH_TRACER_AD_H__
