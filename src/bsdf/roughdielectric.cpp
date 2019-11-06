#include "roughdielectric.h"
#include "intersection.h"
#include "intersectionAD.h"
#include "utils.h"
#include "microfacet.h"
#include <assert.h>


Spectrum RoughDielectricBSDF::eval(const Intersection &its, const Vector &wo, bool importance) const{
    if (std::abs(Frame::cosTheta(its.wi)) < Epsilon || std::abs(Frame::cosTheta(wo)) < Epsilon)
        return Spectrum::Zero();

    /* Determine the type of interaction */
    bool reflect = Frame::cosTheta(its.wi)*Frame::cosTheta(wo) > 0;

    Vector H;
    if (reflect) {
        /* Calculate the reflection half-vector */
        H = (wo + its.wi).normalized();
    } else {
        /* Calculate the transmission half-vector */
        Float eta = Frame::cosTheta(its.wi) > 0 ? m_eta.val : m_invEta.val;
        H = (its.wi + wo*eta).normalized();
    }

    /* Ensure that the half-vector points into the
       same hemisphere as the macrosurface normal */
    H *= math::signum(Frame::cosTheta(H));

    /* Evaluate the microfacet normal distribution */
    const Float D = m_distr.eval(H);
    if (std::abs(D) < Epsilon) return Spectrum::Zero();

    /* Fresnel factor */
    const Float F = fresnelDielectricExt(its.wi.dot(H), m_eta.val);

    /* Smith's shadow-masking function */
    const Float G = m_distr.G(its.wi, wo, H);

    if (reflect) {
        /* Calculate the total amount of reflection */
        return Spectrum::Ones()*(F*D*G/(4.0f*std::abs(Frame::cosTheta(its.wi))));
    } else {
        Float eta = Frame::cosTheta(its.wi) > 0.0f ? m_eta.val : m_invEta.val;

        /* Calculate the total amount of transmission */
        Float sqrtDenom = its.wi.dot(H) + eta*wo.dot(H);
        Float value = ((1.0f - F)*D*G*eta*eta*its.wi.dot(H)*wo.dot(H))/(Frame::cosTheta(its.wi)*sqrtDenom*sqrtDenom);

        Float factor = importance ? 1.0f : (Frame::cosTheta(its.wi) > 0 ? m_invEta.val : m_eta.val);
        return Spectrum::Ones()*std::abs(value*factor*factor);
    }
}


SpectrumAD RoughDielectricBSDF::evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance) const {
    if (std::abs(Frame::cosTheta(its.wi.val)) < Epsilon || std::abs(Frame::cosTheta(wo.val)) < Epsilon)
        return SpectrumAD();

    /* Determine the type of interaction */
    bool reflect = Frame::cosTheta(its.wi.val)*Frame::cosTheta(wo.val) > 0;

    VectorAD H;
    if (reflect) {
        /* Calculate the reflection half-vector */
        H = (wo + its.wi).normalized();
    } else {
        /* Calculate the transmission half-vector */
        const auto &eta = Frame::cosTheta(its.wi.val) > 0 ? m_eta : m_invEta;
        H = (its.wi + wo*eta).normalized();
    }

    /* Ensure that the half-vector points into the
       same hemisphere as the macrosurface normal */
    H *= math::signum(Frame::cosTheta(H.val));

    /* Evaluate the microfacet normal distribution */
    const FloatAD D = m_distr.evalAD(H);
    if (std::abs(D.val) < Epsilon) return SpectrumAD();

    /* Fresnel factor */
    const FloatAD F = fresnelDielectricExtAD(its.wi.dot(H), m_eta);

    /* Smith's shadow-masking function */
    const FloatAD G = m_distr.GAD(its.wi, wo, H);

    SpectrumAD ret;
    if (reflect) {
        /* Calculate the total amount of reflection */
        ret.fill(F*D*G/(4.0f*FrameAD::cosTheta(its.wi).abs()));
    } else {
        const auto &eta = Frame::cosTheta(its.wi.val) > 0.0f ? m_eta : m_invEta;

        /* Calculate the total amount of transmission */
        FloatAD sqrtDenom = its.wi.dot(H) + eta*wo.dot(H);
        FloatAD value = ((1.0f - F)*D*G*eta*eta*its.wi.dot(H)*wo.dot(H))/(FrameAD::cosTheta(its.wi)*sqrtDenom*sqrtDenom);
        if (importance)
            ret.fill(value.abs());
        else {
            FloatAD factor = Frame::cosTheta(its.wi.val) > 0 ? m_invEta : m_eta;
            ret.fill((value*factor*factor).abs());
        }
    }
    return ret;
}


Spectrum RoughDielectricBSDF::sample(const Intersection &its, const Array3 &rnd, Vector &wo, Float &pdf, Float &eta, bool importance) const {
    Array2 sample(rnd[0], rnd[1]);

    /* Sample M, the microfacet normal */
    Float microfacetPDF;
    const Vector m = m_distr.sample(math::signum(Frame::cosTheta(its.wi))*its.wi, sample, microfacetPDF);
    if (microfacetPDF < Epsilon) return Spectrum::Zero();
    pdf = microfacetPDF;

    Float cosThetaT;
    Float F = fresnelDielectricExt(its.wi.dot(m), cosThetaT, m_eta.val);
    Float weight = 1.0f;

    bool sampleReflection;
    if (rnd[2] > F) {
        sampleReflection = false;
        pdf *= 1.0f - F;
    } else {
        sampleReflection = true;
        pdf *= F;
    }

    Float dwh_dwo;
    if (sampleReflection) {
        /* Perfect specular reflection based on the microfacet normal */
        wo = reflect(its.wi, m);
        eta = 1.0f;

        /* Side check */
        if (Frame::cosTheta(its.wi)*Frame::cosTheta(wo) < Epsilon)
            return Spectrum::Zero();

        /* Jacobian of the half-direction mapping */
        dwh_dwo = 1.0f/(4.0f*wo.dot(m));
    } else {
        if (std::abs(cosThetaT) < Epsilon)
            return Spectrum::Zero();

        /* Perfect specular transmission based on the microfacet normal */
        wo = refract(its.wi, m, m_eta.val, cosThetaT);
        eta = cosThetaT < 0 ? m_eta.val : m_invEta.val;

        /* Side check */
        if (Frame::cosTheta(its.wi)*Frame::cosTheta(wo) >= 0)
            return Spectrum::Zero();

        /* Radiance must be scaled to account for the solid angle compression
           that occurs when crossing the interface. */
        Float factor = importance ? 1.0f : (cosThetaT < 0 ? m_invEta.val : m_eta.val);

        weight *= (factor*factor);

        /* Jacobian of the half-direction mapping */
        Float sqrtDenom = its.wi.dot(m) + eta*wo.dot(m);
        dwh_dwo = (eta*eta*wo.dot(m))/(sqrtDenom*sqrtDenom);
    }

    weight *= m_distr.smithG1(wo, m);
    pdf *= std::abs(dwh_dwo);

    return Spectrum::Ones()*weight;
}


Float RoughDielectricBSDF::pdf(const Intersection &its, const Vector &wo) const{
    bool reflect = Frame::cosTheta(its.wi)*Frame::cosTheta(wo) > 0;

    Vector H;
    Float dwh_dwo;

    if (reflect) {
        /* Calculate the reflection half-vector */
        H = (its.wi + wo).normalized();

        /* Jacobian of the half-direction mapping */
        dwh_dwo = 1.0f/(4.0f*wo.dot(H));
    } else {
        /* Calculate the transmission half-vector */
        Float eta = Frame::cosTheta(its.wi) > 0 ? m_eta.val : m_invEta.val;

        H = (its.wi + eta*wo).normalized();

        /* Jacobian of the half-direction mapping */
        Float sqrtDenom = its.wi.dot(H) + eta*wo.dot(H);
        dwh_dwo = (eta*eta*wo.dot(H))/(sqrtDenom*sqrtDenom);
    }

    /* Ensure that the half-vector points into the
       same hemisphere as the macrosurface normal */
    H *= math::signum(Frame::cosTheta(H));

    /* Evaluate the microfacet model sampling density function */
    Float prob = m_distr.pdf(math::signum(Frame::cosTheta(its.wi))*its.wi, H);

    Float F = fresnelDielectricExt(its.wi.dot(H), m_eta.val);
    prob *= reflect ? F : (1.0f - F);

    return std::abs(prob*dwh_dwo);
}
