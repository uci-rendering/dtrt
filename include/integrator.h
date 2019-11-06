#pragma once
#ifndef INTEGRATOR_H__
#define INTEGRATOR_H__

#include "ptr.h"
#include "fwd.h"

struct Scene;

struct RenderOptions {
    RenderOptions(uint64_t seed, int num_samples, int max_bounces, int num_samples_primary_edge, int num_samples_secondary_edge, bool quiet, int mode = -1, float ddistCoeff = 0.0f)
        : seed(seed), num_samples(num_samples), max_bounces(max_bounces)
        , num_samples_primary_edge(num_samples_primary_edge), num_samples_secondary_edge(num_samples_secondary_edge)
        , quiet(quiet), mode(mode), ddistCoeff(ddistCoeff)
    {}

    uint64_t seed;
    int num_samples;
    int max_bounces;
    int num_samples_primary_edge;       // Camera ray
    int num_samples_secondary_edge;     // Secondary (i.e., reflected/scattered) rays
    bool quiet;
    int mode;
    Float ddistCoeff;
};

struct Integrator {
    virtual void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const = 0;
};

#endif //INTEGRATOR_H__
