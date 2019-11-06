#pragma once
#ifndef BSDF_ROUGH_DIELECTRIC_H__
#define BSDF_ROUGH_DIELECTRIC_H__

#include "bsdf.h"
#include "ptr.h"
#include "microfacet.h"

struct RoughDielectricBSDF: BSDF
{
    inline RoughDielectricBSDF(Float alpha, Float intIOR, Float extIOR) : m_distr(alpha) {
        m_eta = intIOR/extIOR;
        m_invEta = 1.0f/m_eta;
    }

    // For pyBind
    inline RoughDielectricBSDF(float alpha, float intIOR, float extIOR, ptr<float> dAlpha)
        : m_distr(alpha), m_eta(intIOR/extIOR)
    {
        m_invEta = 1.0f/m_eta;
        initVelocities(Eigen::Map<Eigen::Array<float, nder, 1> >(dAlpha.get(), nder, 1).cast<Float>());
    }

    inline RoughDielectricBSDF(float alpha, float intIOR, float extIOR, ptr<float> dAlpha, ptr<float> dEta)
        : m_distr(alpha), m_eta(intIOR/extIOR)
    {
        initVelocities(Eigen::Map<Eigen::Array<float, nder, 1> >(dAlpha.get(), nder, 1).cast<Float>(),
                       Eigen::Map<Eigen::Array<float, nder, 1> >(dEta.get(), nder, 1).cast<Float>());
    }

    inline void initVelocities(const Eigen::Array<Float, nder, 1> &dAlpha) {
        m_distr.initVelocities(dAlpha);
    }

    inline void initVelocities(const Eigen::Array<Float, nder, 1> &dAlpha, const Eigen::Array<Float, nder, 1> &dEta) {
        m_distr.initVelocities(dAlpha);
        m_eta.der = dEta;
        m_invEta = 1.0f/m_eta;
    }


    Spectrum eval(const Intersection &its, const Vector &wo, bool importance = false) const;
    SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance = false) const;

    Spectrum sample(const Intersection &its, const Array3 &rnd, Vector &wo, Float &pdf, Float &eta, bool importance = false) const;
    Float pdf(const Intersection &its, const Vector &wo) const;

    bool isTransmissive() const { return true; }

    bool isNull() const { return false; }

    std::string toString() const {
        std::ostringstream oss;
        oss << "BSDF_rough_dielectric [" << '\n'
            //<< "  alpha = " << m_alpha << '\n'
            << "  eta = " << m_eta << '\n'
            << "]" << std::endl;
        return oss.str();
    }

    MicrofacetDistribution m_distr;
    FloatAD m_eta, m_invEta;
};

#endif
