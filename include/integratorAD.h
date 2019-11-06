#pragma once
#ifndef INTEGRATOR_AD_H__
#define INTEGRATOR_AD_H__

#include "integrator.h"
#include <omp.h>

struct RndSampler;

struct IntegratorAD : Integrator {

    enum EMode {
        EEmission               = 0x0001,
        ESurfaceMain            = 0x0002,
        ESurfaceBoundary        = 0x0004,
        ESurface                = ESurfaceMain | ESurfaceBoundary,
        // Only valid for volume path tracer
        EVolumeMain             = 0x0008,
        EVolumeBoundary1        = 0x0010,
        EVolumeBoundary2        = 0x0020,
        EVolume                 = EVolumeMain | EVolumeBoundary1 | EVolumeBoundary2,
        EAllButEmission         = ESurface | EVolume,
        EAll                    = EEmission | EAllButEmission
    };

    IntegratorAD();
    virtual ~IntegratorAD();
    void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const;

    virtual Spectrum pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const = 0;
    virtual SpectrumAD pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const = 0;
    virtual std::string getName() const = 0;

    mutable omp_lock_t messageLock;
};

#endif //INTEGRATOR_AD_H__
