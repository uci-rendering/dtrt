#pragma once
#ifndef BSDF_PHONG_H__
#define BSDF_PHONG_H__

#include "bsdf.h"
#include "ptr.h"

struct PhongBSDF: BSDF {
    inline PhongBSDF(const Spectrum3f& diffuse, const Spectrum3f &specular, Float exponent)
        : m_diffuseReflectance(diffuse.cast<Float>()), m_specularReflectance(specular.cast<Float>())
        , m_exponent(exponent)
    {
        Float v0 = m_diffuseReflectance.val.mean(), v1 = m_specularReflectance.val.mean();
        m_specularSamplingWeight = v1/(v0 + v1);
    }

    // For pyBind
    inline PhongBSDF(const Spectrum3f& diffuse, const Spectrum3f &specular, float exponent,
              ptr<float> dDiffuse, ptr<float> dSpecular, ptr<float> dExponent)
        : m_diffuseReflectance(diffuse.cast<Float>()), m_specularReflectance(specular.cast<Float>())
        , m_exponent(exponent)
    {
        Float v0 = m_diffuseReflectance.val.mean(), v1 = m_specularReflectance.val.mean();
        m_specularSamplingWeight = v1/(v0 + v1);
        initVelocities(
            Eigen::Map<Eigen::Array<float, nder, 3, Eigen::RowMajor> >(dDiffuse.get(), nder, 3).cast<Float>(),
            Eigen::Map<Eigen::Array<float, nder, 3, Eigen::RowMajor> >(dSpecular.get(), nder, 3).cast<Float>(),
            Eigen::Map<Eigen::Array<float, nder, 1> >(dExponent.get(), nder, 1).cast<Float>()
        );
    }

    inline void initVelocities(const Eigen::Array<Float, nder, 3> &dDiffuse,
                               const Eigen::Array<Float, nder, 3> &dSpecular,
                               const Eigen::Array<Float, nder, 1> &dExponent) {
        m_diffuseReflectance.der = dDiffuse;
        m_specularReflectance.der = dSpecular;
        m_exponent.der = dExponent;
    }

    Spectrum eval(const Intersection &its, const Vector &wo, bool importance = false) const;
    SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance = false) const;

    Spectrum sample(const Intersection &its, const Array3 &rnd, Vector &wo, Float &pdf, Float &eta, bool importance = false) const;
    Float pdf(const Intersection &its, const Vector &wo) const;

    inline bool isTransmissive() const { return false; }
    inline bool isNull() const { return false; }

    std::string toString() const {
        std::ostringstream oss;
        oss << "BSDF_Phong [" << std::endl
            << "  diffuseReflectance = " << "(" << m_diffuseReflectance << ")" << '\n'
            << "  specularReflectance = " << "(" << m_specularReflectance << ")" << '\n'
            << "  exponent = " << m_exponent
            << "]" << std::endl;
        return oss.str();
    }

    SpectrumAD m_diffuseReflectance, m_specularReflectance;
    FloatAD m_exponent;
    Float m_specularSamplingWeight;
};

#endif
