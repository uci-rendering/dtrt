#pragma once
#ifndef DIRECT_INTEGRATOR_H__
#define DIRECT_INTEGRATOR_H__

#include "integrator.h"

struct RndSampler;
struct Ray;

struct DirectIntegrator : Integrator {
    Spectrum3f radiance(const Scene& scene, RndSampler* sampler, Ray& ray) const;
    void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const;
};

#endif //DIRECT_INTEGRATOR_H__
