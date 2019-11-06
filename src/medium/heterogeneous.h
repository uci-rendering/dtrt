#pragma once
#ifndef HETEROGENEOUS_MEDIUM_H__
#define HETEROGENEOUS_MEDIUM_H__

#include <sstream>
#include <string>

#include "medium.h"
#include "ptr.h"
#include "gridvolume.h"

struct Heterogeneous : Medium {
    inline Heterogeneous(const std::string &densityFile, const Matrix4x4 &toWorld, Float densityScale, const Spectrum& albedo, int phase_id)
        : Medium(phase_id), m_densityScale(densityScale), m_densityVol(densityFile, toWorld), m_albedo(albedo)
    {
        m_maxInvDensity = 1.0f/(m_densityScale.val*m_densityVol.getMaximumFloatValue());
        m_residualRatioTracking = true;
    }

    inline Heterogeneous(const std::string &densityFile, const Matrix4x4 &toWorld, Float densityScale, const std::string &albedoFile, int phase_id)
        : Medium(phase_id), m_densityScale(densityScale), m_densityVol(densityFile, toWorld), m_albedoVol(albedoFile, toWorld)
    {
        m_maxInvDensity = 1.0f/(m_densityScale.val*m_densityVol.getMaximumFloatValue());
        m_residualRatioTracking = true;
    }

    // For pyBind (without velocities)
    inline Heterogeneous(const std::string &densityFile, ptr<float> toWorld, float densityScale, const Spectrum3f& albedo, int phase_id)
        : Medium(phase_id), m_densityScale(densityScale)
        , m_densityVol(densityFile, Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> >(toWorld.get()).cast<Float>())
        , m_albedo(albedo.cast<Float>())
    {
        m_maxInvDensity = 1.0f/(m_densityScale.val*m_densityVol.getMaximumFloatValue());
        m_residualRatioTracking = true;
    }

    inline Heterogeneous(const std::string &densityFile, ptr<float> toWorld, float densityScale, const std::string &albedoFile, int phase_id)
        : Medium(phase_id), m_densityScale(densityScale)
        , m_densityVol(densityFile, Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> >(toWorld.get()).cast<Float>())
        , m_albedoVol(albedoFile, Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> >(toWorld.get()).cast<Float>())
    {
        m_maxInvDensity = 1.0f/(m_densityScale.val*m_densityVol.getMaximumFloatValue());
        m_residualRatioTracking = true;
    }

    // For pyBind (with velocities)
    inline Heterogeneous(const std::string &densityFile, ptr<float> toWorld, float densityScale, const Spectrum3f& albedo, int phase_id,
                         ptr<float> transVelocities, ptr<float> rotVelocities, ptr<float> dDensity, ptr<float> dAlbedo)
        : Medium(phase_id)
        , m_densityScale(densityScale, Eigen::Map<Eigen::Array<float, nder, 1> >(dDensity.get(), nder, 1).cast<Float>())
        , m_densityVol(densityFile, Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> >(toWorld.get()).cast<Float>())
        , m_albedo(albedo.cast<Float>(), Eigen::Map<Eigen::Array<float, nder, 3, Eigen::RowMajor> >(dAlbedo.get(), nder, 3).cast<Float>())
    {
        m_maxInvDensity = 1.0f/(m_densityScale.val*m_densityVol.getMaximumFloatValue());
        m_densityVol.initVelocities(
            Eigen::Map<Eigen::Matrix<float, 3, nder> >(transVelocities.get(), 3, nder).cast<Float>(),
            Eigen::Map<Eigen::Matrix<float, 3, nder> >(rotVelocities.get(), 3, nder).cast<Float>()
        );
        m_residualRatioTracking = true;
    }

    inline Heterogeneous(const std::string &densityFile, ptr<float> toWorld, float densityScale, const std::string &albedoFile, int phase_id,
                         ptr<float> transVelocities, ptr<float> rotVelocities, ptr<float> dDensity)
        : Medium(phase_id)
        , m_densityScale(densityScale, Eigen::Map<Eigen::Array<float, nder, 1> >(dDensity.get(), nder, 1).cast<Float>())
        , m_densityVol(densityFile, Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> >(toWorld.get()).cast<Float>())
        , m_albedoVol(albedoFile, Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> >(toWorld.get()).cast<Float>())
    {
        m_maxInvDensity = 1.0f/(m_densityScale.val*m_densityVol.getMaximumFloatValue());
        m_densityVol.initVelocities(
            Eigen::Map<Eigen::Matrix<float, 3, nder> >(transVelocities.get(), 3, nder).cast<Float>(),
            Eigen::Map<Eigen::Matrix<float, 3, nder> >(rotVelocities.get(), 3, nder).cast<Float>()
        );
        m_residualRatioTracking = true;
    }


    bool sampleDistance(const Ray& ray, const Float &tmax, const Array2 &rnd2, RndSampler* sampler, Vector& p_scatter, Spectrum& throughput) const;
    bool sampleDistanceAD(const RayAD& ray, const FloatAD &tmax, const Array2 &rnd2, RndSampler* sampler, VectorAD& p_scatter, SpectrumAD& throughput) const;

    Float evalTransmittance(const Ray& ray, const Float &tmin, const Float &tmax, RndSampler* sampler) const;
    FloatAD evalTransmittanceAD(const RayAD& ray, const FloatAD &tmin, const FloatAD &tmax, RndSampler* sampler) const;

    Eigen::Array<Float, nder, 1> intSigT(const RayAD &ray, const FloatAD &tmin, const FloatAD &tmax, RndSampler* sampler) const;

    inline bool isHomogeneous() const { return false; }

    inline Spectrum sigS(const Vector& x) const {
        return m_densityScale.val*(m_albedo.val.transpose()*m_densityVol.lookupFloat(x));
    }

    inline std::string toString() const {
        std::ostringstream oss;
        oss << "Heterogeneous Medium";
        return oss.str();
    }

    FloatAD m_densityScale;
    GridVolume m_densityVol;
    Float m_maxInvDensity;

    // Only one of the following should be valid
    GridVolume m_albedoVol;
    SpectrumAD m_albedo;

    bool m_residualRatioTracking;
};

#endif //HETEROGENEOUS_MEDIUM_H__
