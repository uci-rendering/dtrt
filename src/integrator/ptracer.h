#pragma once
#ifndef PARTICLE_TRACER_H__
#define PARTICLE_TRACER_H__

#include "integrator.h"

struct Medium;
struct RndSampler;
struct Intersection;
struct Ray;

struct ParticleTracer : Integrator {
    void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const;
    void traceParticle(const Scene& scene, RndSampler *sampler, int max_bounces, int thread_id) const;
    void handleEmission(const Intersection& its, const Scene& scene, RndSampler *sampler, 
    					const Spectrum& weight, int max_bounces, int thread_id) const;
    void handleSurfaceInteraction(const Intersection& its, const Scene& scene, const Medium* ptr_med, RndSampler *sampler,
    							  const Spectrum& weight, int max_bounces, int thread_id) const;
	void handleMediumInteraction(const Scene& scene, const Medium* ptr_med, const Ray& ray, RndSampler *sampler,
								 const Spectrum& weight, int max_bounces, int thread_id) const;
    mutable std::vector<std::vector<Spectrum>> image_per_thread;
    mutable omp_lock_t messageLock;
};

#endif