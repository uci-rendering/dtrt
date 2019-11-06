#include "phong.h"
#include "intersection.h"
#include "intersectionAD.h"
#include "utils.h"
#include "frame.h"
#include "frameAD.h"
#include <assert.h>

inline static Vector reflect(const Vector &wi) {
    return Vector(-wi.x(), -wi.y(), wi.z());
}


inline static VectorAD reflectAD(const VectorAD &wi) {
    return VectorAD(-wi.x(), -wi.y(), wi.z());
}


Spectrum PhongBSDF::eval(const Intersection &its, const Vector &wo, bool importance) const {
    if (Frame::cosTheta(its.wi) < Epsilon || Frame::cosTheta(wo) < Epsilon)
        return Spectrum::Zero();

    Spectrum result = Spectrum::Zero();
    // Specular
    Float alpha = wo.dot(reflect(its.wi));
    if (alpha > 0.0f) {
        result += m_specularReflectance.val*((m_exponent.val + 2.0f)*std::pow(alpha, m_exponent.val))*INV_TWOPI;
    }

    // Diffuse
    result += m_diffuseReflectance.val*INV_PI;

    return result*Frame::cosTheta(wo);
}


SpectrumAD PhongBSDF::evalAD(const IntersectionAD &its, const VectorAD &wo, bool importance) const
{
    if (Frame::cosTheta(its.wi.val) < Epsilon || Frame::cosTheta(wo.val) < Epsilon)
        return SpectrumAD();

    SpectrumAD result;
    // Specular
    FloatAD alpha = wo.dot(reflectAD(its.wi));
    if (alpha > 0.0f) {
        result += m_specularReflectance*((m_exponent + 2.0f)*alpha.pow(m_exponent))*INV_TWOPI;
    }

    // Diffuse
    result += m_diffuseReflectance*INV_PI;

    return result*FrameAD::cosTheta(wo);
}


Spectrum PhongBSDF::sample(const Intersection &its, const Array3 &rnd, Vector &wo, Float &_pdf, Float &eta, bool importance) const {
    if ( rnd[2] < m_specularSamplingWeight ) {
        Vector R = reflect(its.wi);

        /* Sample from a Phong lobe centered around (0, 0, 1) */
        Float sinAlpha = std::sqrt(1.0f - std::pow(rnd[1], 2.0f/(m_exponent.val + 1.0f)));
        Float cosAlpha = std::pow(rnd[1], 1.0f/(m_exponent.val + 1.0f));
        Float phi = (2.0f*M_PI)*rnd[0];
        Vector localDir = Vector(sinAlpha*std::cos(phi), sinAlpha*std::sin(phi), cosAlpha);

        /* Rotate into the correct coordinate system */
        wo = Frame(R).toWorld(localDir);

        if (Frame::cosTheta(wo) < Epsilon)
            return Spectrum::Zero();
    } else {
        wo = squareToCosineHemisphere(Vector2(rnd[0], rnd[1]));
    }
    eta = 1.0f;

    _pdf = pdf(its, wo);
    if (_pdf < Epsilon)
        return Spectrum::Zero();
    else
        return eval(its, wo)/_pdf;
}


Float PhongBSDF::pdf(const Intersection &its, const Vector &wo) const {
    if (Frame::cosTheta(its.wi) < Epsilon || Frame::cosTheta(wo) < Epsilon)
        return 0.0f;

    Float diffuseProb = 0.0, specProb = 0.0;
    diffuseProb = squareToCosineHemispherePdf(wo);
    Float alpha = wo.dot(reflect(its.wi));
    if (alpha > 0)
        specProb = std::pow(alpha, m_exponent.val)*(m_exponent.val + 1.0f)/(2.0f*M_PI);

    return m_specularSamplingWeight*specProb + (1.0f - m_specularSamplingWeight)*diffuseProb;
}
