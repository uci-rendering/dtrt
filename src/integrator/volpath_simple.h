#pragma once
#ifndef VOL_PATH_TRACER_SIMPLE_H__
#define VOL_PATH_TRACER_SIMPLE_H__

#include "integrator.h"

struct Medium;
struct Ray;
struct RndSampler;

struct VolPathTracerSimple : Integrator {
	Spectrum3f radiance(const Scene& scene, RndSampler* sampler, Ray& ray, int max_depth, const Medium* med_default = nullptr) const;
    void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const;
};

#endif
