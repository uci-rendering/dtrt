#include "homogeneous.h"
#include "sampler.h"
#include "ray.h"
#include "rayAD.h"

bool Homogeneous::sampleDistance(const Ray& ray, const Float &tmax, const Array2 &rnd2, RndSampler* sampler,
                                 Vector& p_scatter, Spectrum& throughput) const
{
    if ( sampling_weight > Epsilon ) {
        Float t = -std::log(rnd2.x())/sigma_t.val;
        if ( t < tmax ) {
            p_scatter = ray(t);
            if ( rnd2.y() < sampling_weight )
                throughput *= albedo.val/sampling_weight;
            else
                throughput = Spectrum::Zero();
            return true;
        } else
            return false;
    } else {
        throughput *= std::exp(-tmax*sigma_t.val);
        return false;
    }
}


bool Homogeneous::sampleDistanceAD(const RayAD& ray, const FloatAD &tmax, const Array2 &rnd2, RndSampler* sampler,
                                   VectorAD& p_scatter, SpectrumAD& throughput) const
{
    if ( sampling_weight > Epsilon ) {
        Float t = -std::log(rnd2.x())/sigma_t.val;
        if ( t < tmax.val ) {
            p_scatter = ray(t);
            if (rnd2.y() < sampling_weight) {
                FloatAD tmp = sigma_t*(-t*sigma_t).exp();
                throughput *= albedo*tmp/(tmp.val*sampling_weight);
            } else
                throughput = SpectrumAD();
            return true;
        } else {
            FloatAD tmp = (-tmax*sigma_t).exp();
            throughput *= (tmp/tmp.val);
            return false;
        }
    } else {
        throughput *= (-tmax*sigma_t).exp();
        return false;
    }
}


Float Homogeneous::evalTransmittance(const Ray& ray, const Float &tmin, const Float &tmax, RndSampler* sampler) const {
    return std::exp((tmin - tmax)*sigma_t.val);
}


FloatAD Homogeneous::evalTransmittanceAD(const RayAD& ray, const FloatAD &tmin, const FloatAD &tmax, RndSampler* sampler) const {
    return ((tmin - tmax)*sigma_t).exp();
}
