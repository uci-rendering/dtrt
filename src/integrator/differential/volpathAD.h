#pragma once
#ifndef VOLUMETRIC_PATH_TRACER_AD_H__
#define VOLUMETRIC_PATH_TRACER_AD_H__

#include "integratorAD.h"

struct Ray;
struct RayAD;
struct IntersectionAD;
struct Medium;

struct VolPathTracerAD : IntegratorAD {
    VolPathTracerAD();
    ~VolPathTracerAD();

    Spectrum Li(const Scene& scene, RndSampler* sampler, const Ray& ray, int max_depth, const Medium* init_med,
                bool incEmission = true) const;
    SpectrumAD LiAD(const Scene& scene, const IntersectionAD& its, RndSampler* sampler, const RayAD& ray, const Medium *med,
                    int max_depth, EMode mode, int nEdgeSamples, Float ddistCoeff) const;

    Spectrum pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const;
    SpectrumAD pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const;

    std::string getName() const { return "volpathAD"; }
};

#endif //VOLUMETRIC_PATH_TRACER_AD_H__
