#include "heterogeneous.h"
#include "sampler.h"
#include "ray.h"
#include "rayAD.h"

#define DELTA_TRACKING_MAX_DEPTH 10000


bool Heterogeneous::sampleDistance(const Ray& ray, const Float &tmax, const Array2 &rnd2, RndSampler* sampler,
                                   Vector& p_scatter, Spectrum& throughput) const
{
    // assert(std::isfinite(tmax));
    Float tNear, tFar;
    if ( !m_densityVol.getAABB().rayIntersect(ray, tNear, tFar) ) return false;
    tNear = std::max(tNear, static_cast<Float>(0.0)); tFar = std::min(tFar, tmax);

    // Delta tracking
    Float t = tNear;
    for ( int dep = 0; ; ++dep ) {
        t -= std::log(sampler->next1D())*m_maxInvDensity;
        if ( t > tFar ) break;
        if ( dep > DELTA_TRACKING_MAX_DEPTH ) {
            std::cerr << "Warning: delta tracking does not terminate after " << DELTA_TRACKING_MAX_DEPTH << " iterations." << std::endl;
            break;
        }

        Vector p = ray(t);
        Float densityAtT = m_densityVol.lookupFloat(p);
        if ( sampler->next1D()*m_densityVol.getMaximumFloatValue() < densityAtT ) {
            p_scatter = p;
            throughput *= m_albedoVol.isValid() ?
                m_albedoVol.lookupSpectrum(p) : m_albedo.val.transpose();
            return true;
        }
    }
    return false;
}


bool Heterogeneous::sampleDistanceAD(const RayAD& ray, const FloatAD &tmax, const Array2 &rnd2, RndSampler* sampler,
                                     VectorAD& p_scatter, SpectrumAD& throughput) const
{
    // assert(std::isfinite(tmax.val));
    Float tNear, tFar;
    if ( !m_densityVol.getAABB().rayIntersect(ray.toRay(), tNear, tFar) ) return false;
    tNear = std::max(tNear, static_cast<Float>(0.0)); tFar = std::min(tFar, tmax.val);

    // Delta tracking
    Float t = tNear;
    for ( int dep = 0; ; ++dep ) {
        t -= std::log(sampler->next1D())*m_maxInvDensity;
        if ( t > tFar ) break;
        if ( dep > DELTA_TRACKING_MAX_DEPTH ) {
            std::cerr << "Warning: delta tracking does not terminate after " << DELTA_TRACKING_MAX_DEPTH << " iterations." << std::endl;
            break;
        }

        VectorAD p = ray(t);
        FloatAD densityAtT = m_densityVol.lookupFloatAD(p);
        if ( sampler->next1D()*m_densityVol.getMaximumFloatValue() < densityAtT.val ) {
            p_scatter = p;
            SpectrumAD albedoVal = m_albedoVol.isValid() ?
                m_albedoVol.lookupSpectrumAD(p) : m_albedo;

            FloatAD sigT = m_densityScale*densityAtT, tmp(1.0f, -intSigT(ray, 0.0f, t, sampler));
            throughput *= albedoVal*(sigT*tmp/sigT.val);
            return true;
        }
    }

    FloatAD tmp(1.0f);
    tmp.der = -intSigT(ray, 0.0f, tmax, sampler);
    throughput *= tmp;
    return false;
}


Float Heterogeneous::evalTransmittance(const Ray& ray, const Float &tmin, const Float &tmax, RndSampler* sampler) const {
    if ( m_residualRatioTracking ) {
        // Residual ratio tracking
        Float ret = 1.0f, t = tmin;
        while ( ret > Epsilon ) {
            t -= std::log(sampler->next1D())*m_maxInvDensity;
            if ( t > tmax ) break;

            Vector p = ray(t);
            Float densityAtT = m_densityVol.lookupFloat(p);
            ret *= 1.0f - densityAtT/m_densityVol.getMaximumFloatValue();
        }
        return ret;
    } else {
        // Delta tracking
        constexpr int nsamples = 2;
        Float ret = 0.0f;
        for ( int sampleId = 0; sampleId < nsamples; ++sampleId ) {
            Float t = tmin;
            for ( ; ; ) {
                t -= std::log(sampler->next1D())*m_maxInvDensity;
                if ( t > tmax ) {
                    ret += 1.0f; break;
                }

                Vector p = ray(t);
                Float densityAtT = m_densityVol.lookupFloat(p);
                if ( sampler->next1D()*m_densityVol.getMaximumFloatValue() < densityAtT ) break;
            }
        }
        return ret/static_cast<Float>(nsamples);
    }
}


FloatAD Heterogeneous::evalTransmittanceAD(const RayAD& ray, const FloatAD &tmin, const FloatAD &tmax, RndSampler* sampler) const {
    Ray ray0 = ray.toRay();
    FloatAD ret(evalTransmittance(ray0, tmin.val, tmax.val, sampler));
    ret.der = -ret.val*intSigT(ray, tmin, tmax, sampler);
    return ret;
}


Eigen::Array<Float, nder, 1> Heterogeneous::intSigT(const RayAD &ray, const FloatAD &tmin, const FloatAD &tmax, RndSampler* sampler) const {
    constexpr int lineIntegralSamples = 10;
    Eigen::Array<Float, nder, 1> ret = Eigen::Array<Float, nder, 1>::Zero();

    const Float len = tmax.val - tmin.val;
    for ( int i = 0; i < lineIntegralSamples; ++i ) {
        Float rnd = static_cast<Float>(i + sampler->next1D())/lineIntegralSamples;
        FloatAD sigT = m_densityScale*m_densityVol.lookupFloatAD(ray(tmin + len*rnd));
        ret += sigT.der;
    }
    ret *= len/static_cast<Float>(lineIntegralSamples);

    // Boundary terms
    Ray ray0 = ray.toRay();
    if ( !tmax.der.isZero(Epsilon) )
        ret += tmax.der*m_densityScale.val*m_densityVol.lookupFloat(ray0(tmax.val - Epsilon));
    if ( !tmin.der.isZero(Epsilon) )
        ret -= tmin.der*m_densityScale.val*m_densityVol.lookupFloat(ray0(tmin.val + Epsilon));

    return ret;
}
