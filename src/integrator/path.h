#pragma once
#ifndef PATH_TRACER_H__
#define PATH_TRACER_H__

#include "integrator.h"

struct RndSampler;
struct Ray;

struct PathTracer : Integrator {
	Spectrum3f radiance(const Scene& scene, RndSampler* sampler, Ray& ray, int max_depth) const;
    void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const;
};

#endif