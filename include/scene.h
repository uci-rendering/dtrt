#pragma once
#ifndef SCENE_H__
#define SCENE_H__

#include "ptr.h"
#include "utils.h"
#include "ray.h"
#include "rayAD.h"
#include "intersection.h"
#include "intersectionAD.h"
#include "camera.h"
#include "emitter.h"
#include "shape.h"
#include "bsdf.h"
#include "medium.h"
#include "sampler.h"
#include "phase.h"
#include "pmf.h"
#include "edge_manager.h"
#include <vector>
#include <memory>
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>

struct Scene {
    Scene(const Camera &camera,
          const std::vector<const Shape*> &shapes,
          const std::vector<const BSDF*> &bsdfs,
          const std::vector<const Emitter*> &area_lights,
          const std::vector<const PhaseFunction*> &phases,
          const std::vector<const Medium*> &media);
    ~Scene();

    void initEdges(const Eigen::Array<Float, -1, 1> &samplingWeights);
    
    // For pyBind
    void initEdgesPy(ptr<float> samplingWeights) {
        initEdges(Eigen::Map<Eigen::Array<float, -1, 1> >(samplingWeights.get(), shape_list.size(), 1).cast<Float>());
    }

    Camera camera;
    std::vector<const Shape*> shape_list;
    std::vector<const BSDF*> bsdf_list;
    std::vector<const PhaseFunction*> phase_list;
    std::vector<const Medium*> medium_list;
    std::vector<const Emitter*> emitter_list;

    // Embree handles
    RTCDevice embree_device;
    RTCScene embree_scene;

    // Light sampling
    int num_lights;
    std::vector<Float> light_pmf;
    std::vector<Float> light_cdf;
    std::vector<Float> light_areas;
    // Triangle on light sampling
    std::vector<int>   area_cdf_begin;
    std::vector<Float> area_cdf_pool;

    // Edge sampling related
    EdgeManager* ptr_edgeManager;

    // Simple visibility test (IGNORING null interfaces!)
    bool isVisible(const Vector &p, bool pOnSurface, const Vector &q, bool qOnSurface) const;

    // Path Tracer
    bool rayIntersect(const Ray &ray, bool onSurface, Intersection& its) const;
    bool rayIntersectAD(const RayAD &ray, bool onSurface, IntersectionAD& its) const;

    Spectrum sampleEmitterDirect(const Intersection &its, const Array4 &rnd_light, RndSampler* sampler, Vector& wo, Float &pdf) const;
    SpectrumAD sampleEmitterDirectAD(const IntersectionAD &its, const Array4 &rnd_light, RndSampler* sampler, VectorAD& wo, Float &pdf) const;

    // Volume Path Tracer
    Spectrum rayIntersectAndLookForEmitter(const Ray &ray, bool onSurface, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                           Intersection &its, Float& pdf_nee) const;

    SpectrumAD rayIntersectAndLookForEmitterAD(const RayAD &ray, bool onSurface, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                               IntersectionAD &its, Float& pdf_nee, IntersectionAD *itsFar = nullptr) const;

    Spectrum sampleAttenuatedEmitterDirect(const Intersection& its, const Array4 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                           Vector& wo, Float& pdf, bool flag = false) const; // wo in local space

    SpectrumAD sampleAttenuatedEmitterDirectAD(const IntersectionAD& its, const Array4 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                               VectorAD& wo, Float& pdf) const; // wo in local space

    Spectrum sampleAttenuatedEmitterDirect(const Vector &pscatter, const Array4 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                           Vector& wo, Float& pdf) const; // wo in *world* space

    SpectrumAD sampleAttenuatedEmitterDirectAD(const VectorAD &pscatter, const Array4 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                               VectorAD& wo, Float& pdf) const; // wo in *world* space

    Float evalTransmittance(const Ray& ray, bool onSurface, const Medium* ptr_medium, Float remaining, RndSampler* sampler, int max_interactions) const;
    FloatAD evalTransmittanceAD(const RayAD& ray, bool onSurface, const Medium* ptr_medium, FloatAD remaining, RndSampler* sampler, int max_interactions) const;

    Float pdfEmitterSample(const Intersection& its) const;
    Float pdfEmitterSample(const IntersectionAD& its) const;

    inline const Edge* sampleEdge(const Vector& p, const Frame* ptr_frame, Float& rnd, int& shape_id, Float& pdf) const {
        return ptr_edgeManager->sampleSecondaryEdge(*this, p, ptr_frame, rnd, shape_id, pdf);
    }

    inline const Edge* sampleEdge(const Vector& p, const Frame* ptr_frame, Float rnd, int& shape_id, Float &t, Float& pdf) const {
        t = rnd;
        const Edge* ret = sampleEdge(p, ptr_frame, t, shape_id, pdf);
        if (ret != nullptr)
            pdf /= ret->length;
        return ret;
    }

    Spectrum sampleEmitterPosition(const Array4 &rnd_light, Intersection& its) const;
    Float sampleAttenuatedSensorDirect(const Intersection& its, RndSampler* sampler, int max_interactions, Vector2& pixel_uv, Vector& dir) const;
    Float sampleAttenuatedSensorDirect(const Vector& p, const Medium* ptr_med, RndSampler* sampler, int max_interactions, Vector2& pixel_uv, Vector& dir) const;
};
#endif
