#pragma once
#ifndef HOMOGENEOUS_MEDIUM_H__
#define HOMOGENEOUS_MEDIUM_H__

#include "medium.h"
#include "ptr.h"
#include <sstream>
#include <string>


struct Homogeneous : Medium {
    inline Homogeneous(Float sigma_t, const Spectrum3f& albedo, int phase_id)
        : Medium(phase_id), sigma_t(sigma_t), albedo(albedo.cast<Float>())
    {
        sigma_s = sigma_t*this->albedo.val.transpose();
        sampling_weight = albedo.maxCoeff();
    }

    // For pyBind
    inline Homogeneous(float sigma_t, const Spectrum3f& albedo, int phase_id, ptr<float> dSigT, ptr<float> dAlbedo)
        : Medium(phase_id), sigma_t(sigma_t), albedo(albedo.cast<Float>())
    {
        sigma_s = sigma_t*this->albedo.val.transpose();
        sampling_weight = albedo.maxCoeff();
        initVelocities(
            Eigen::Map<Eigen::Array<float, nder, 1> >(dSigT.get(), nder, 1).cast<Float>(),
            Eigen::Map<Eigen::Array<float, nder, 3,  Eigen::RowMajor> >(dAlbedo.get(), nder, 3).cast<Float>()
        );
    }

    inline void initVelocities(const Eigen::Array<Float, nder, 1> &dSigT,
                               const Eigen::Array<Float, nder, 3> &dAlbedo) {
        sigma_t.der = dSigT;
        albedo.der = dAlbedo;
    }

    bool sampleDistance(const Ray& ray, const Float &tmax, const Array2 &rnd2, RndSampler* sampler, Vector& p_scatter, Spectrum& throughput) const;
    bool sampleDistanceAD(const RayAD& ray, const FloatAD &tmax, const Array2 &rnd2, RndSampler* sampler, VectorAD& p_scatter, SpectrumAD& throughput) const;

    Float evalTransmittance(const Ray& ray, const Float &tmin, const Float &tmax, RndSampler* sampler) const;
    FloatAD evalTransmittanceAD(const RayAD& ray, const FloatAD &tmin, const FloatAD &tmax, RndSampler* sampler) const;

    inline bool isHomogeneous() const { return true; }

    inline Spectrum sigS(const Vector& x) const { return sigma_s; }

    inline std::string toString() const {
        std::ostringstream oss;
        oss << "Homogeneous Medium...";
        return oss.str();
    }

    FloatAD sigma_t;
    SpectrumAD albedo;

    Spectrum sigma_s;
    Float sampling_weight;
};

#endif //HOMOGENEOUS_MEDIUM_H__
