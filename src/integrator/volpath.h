#pragma once
#ifndef VOL_PATH_TRACER_H__
#define VOL_PATH_TRACER_H__

#include "integrator.h"

struct Medium;
struct RndSampler;
struct Ray;

struct VolPathTracer : Integrator {
	Spectrum3f radiance(const Scene& scene, RndSampler* sampler, Ray& ray, int max_depth, const Medium* med_default = nullptr) const;
    void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const;
};

#endif
