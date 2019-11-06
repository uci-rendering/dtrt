#pragma once
#ifndef MICROFACET_DISTRB_H__
#define MICROFACET_DISTRB_H__

#include "math_func.h"
#include "frame.h"
#include "frameAD.h"


struct MicrofacetDistribution {
    inline MicrofacetDistribution(const FloatAD &alpha) {
        m_alpha = alpha;
        if ( m_alpha < 1e-4f ) m_alpha.val = 1e-4f;
    }

    inline void initVelocities(const typename FloatAD::DerType &dAlpha) { m_alpha.der = dAlpha; }

    /**
     * \brief Evaluate the microfacet distribution function
     *
     * \param m
     *     The microfacet normal
     */
    inline Float eval(const Vector &m) const {
        if (Frame::cosTheta(m) < Epsilon) return static_cast<Float>(0.0);

        Float cosTheta2 = Frame::cosTheta2(m);
        Float beckmannExponent = (m.x()*m.x() + m.y()*m.y())/(m_alpha.val*m_alpha.val*cosTheta2);

        /* Beckmann distribution function for Gaussian random surfaces - [Walter 2005] evaluation */
        Float result = std::exp(-beckmannExponent)/static_cast<Float>(M_PI*m_alpha.val*m_alpha.val*cosTheta2*cosTheta2);
        /* Prevent potential numerical issues in other stages of the model */
        if (result*Frame::cosTheta(m) < 1e-20) result = static_cast<Float>(0.0);

        return result;
    }


    inline FloatAD evalAD(const VectorAD &m) const {
        if (Frame::cosTheta(m.val) < Epsilon) return FloatAD();

        FloatAD cosTheta2 = FrameAD::cosTheta2(m);
        FloatAD beckmannExponent = (m.x()*m.x() + m.y()*m.y())/(m_alpha*m_alpha*cosTheta2);

        /* Beckmann distribution function for Gaussian random surfaces - [Walter 2005] evaluation */
        FloatAD result = (-beckmannExponent).exp()/(M_PI*m_alpha*m_alpha*cosTheta2*cosTheta2);
        /* Prevent potential numerical issues in other stages of the model */
        if (result.val*Frame::cosTheta(m.val) < 1e-20) result = FloatAD();

        return result;
    }


    /**
     * \brief Draw a sample from the distribution of *visible* normals
     * and return the associated probability density
     *
     * \param _wi
     *    A reference direction that defines the set of visible normals
     * \param sample
     *    A uniformly distributed 2D sample
     * \param pdf
     *    The probability density wrt. solid angles
     */
    inline Vector sample(const Vector &_wi, const Array2 &sample) const {
        /* Step 1: stretch wi */
        Vector wi = Vector(m_alpha.val*_wi.x(), m_alpha.val*_wi.y(), _wi.z()).normalized();

        /* Get polar coordinates */
        Float theta = 0.0f, phi = 0.0f;
        if (Frame::cosTheta(wi) < static_cast<Float>(0.99999)) {
            theta = std::acos(wi.z());
            phi = std::atan2(wi.y(), wi.x());
        }
        Float sinPhi, cosPhi;
        math::sincos(phi, sinPhi, cosPhi);

        /* Step 2: simulate P22_{wi}(slope.x, slope.y, 1, 1) */
        Vector2 slope = sampleVisible11(theta, sample);

        /* Step 3: rotate */
        slope = Vector2(cosPhi*slope.x() - sinPhi*slope.y(),
                        sinPhi*slope.x() + cosPhi*slope.y());

        /* Step 4: unstretch */
        slope *= m_alpha.val;

        /* Step 5: compute normal */
        Float normalization = static_cast<Float>(1.0)/std::sqrt(slope.squaredNorm() + static_cast<Float>(1.0));

        return Vector(-slope.x()*normalization, -slope.y()*normalization, normalization);
    }


    /// Implements the probability density of the function \ref sample()
    Float pdf(const Vector &wi, const Vector &m) const {
        if(std::abs(Frame::cosTheta(wi)) < Epsilon) return static_cast<Float>(0.0);
        return smithG1(wi, m)*std::abs(wi.dot(m))*eval(m)/std::abs(Frame::cosTheta(wi));
    }


    inline Vector sample(const Vector &wi, const Array2 &_sample, Float &_pdf) const {
        Vector m = sample(wi, _sample);
        _pdf = pdf(wi, m);
        return m;
    }


    /**
     * \brief Smith's shadowing-masking function G1 for each
     * of the supported microfacet distributions
     *
     * \param v
     *     An arbitrary direction
     * \param m
     *     The microfacet normal
     */
    Float smithG1(const Vector &v, const Vector &m) const {
        /* Ensure consistent orientation (can't see the back
           of the microfacet from the front and vice versa) */
        if (v.dot(m)*Frame::cosTheta(v) < Epsilon) return 0.0f;

        /* Perpendicular incidence -- no shadowing/masking */
        Float tanTheta = std::abs(Frame::tanTheta(v));
        if (tanTheta == 0.0f)
            return 1.0f;

        Float a = 1.0f/(m_alpha.val*tanTheta);
        if (a >= 1.6f) return 1.0f;

        /* Use a fast and accurate (<0.35% rel. error) rational
           approximation to the shadowing-masking function */
        Float aSqr = a*a;
        return (3.535f*a + 2.181f*aSqr)/(1.0f + 2.276f*a + 2.577f*aSqr);
    }


    FloatAD smithG1AD(const VectorAD &v, const VectorAD &m) const {
        /* Ensure consistent orientation (can't see the back
           of the microfacet from the front and vice versa) */
        if (v.val.dot(m.val)*Frame::cosTheta(v.val) < Epsilon) return FloatAD();

        /* Perpendicular incidence -- no shadowing/masking */
        FloatAD tanTheta = FrameAD::tanTheta(v).abs();
        if (tanTheta.val == 0.0f)
            return FloatAD(1.0f);

        FloatAD a = 1.0f/(m_alpha*tanTheta);
        if (a.val >= 1.6f) return 1.0f;

        /* Use a fast and accurate (<0.35% rel. error) rational
           approximation to the shadowing-masking function */
        FloatAD aSqr = a*a;
        return (3.535f*a + 2.181f*aSqr)/(1.0f + 2.276f*a + 2.577f*aSqr);
    }


    /**
     * \brief Separable shadow-masking function based on Smith's
     * one-dimensional masking model
     */
    inline Float G(const Vector &wi, const Vector &wo, const Vector &m) const {
        return smithG1(wi, m)*smithG1(wo, m);
    }


    inline FloatAD GAD(const VectorAD &wi, const VectorAD &wo, const VectorAD &m) const {
        return smithG1AD(wi, m)*smithG1AD(wo, m);
    }


    /**
     * \brief Visible normal sampling code for the alpha=1 case
     *
     * Source: supplemental material of "Importance Sampling
     * Microfacet-Based BSDFs using the Distribution of Visible Normals"
     */
    Vector2 sampleVisible11(Float thetaI, const Array2 &sample) const {
        const Float SQRT_PI_INV = 1.0f/std::sqrt(M_PI);
        Vector2 slope;

        /* Special case (normal incidence) */
        if (thetaI < 1e-4f) {
            Float sinPhi, cosPhi;
            Float r = std::sqrt(-std::log(1.0f - sample[0]));
            math::sincos(static_cast<Float>(2.0*M_PI*sample[1]), sinPhi, cosPhi);
            return Vector2(r*cosPhi, r*sinPhi);
        }

        /* The original inversion routine from the paper contained
           discontinuities, which causes issues for QMC integration
           and techniques like Kelemen-style MLT. The following code
           performs a numerical inversion with better behavior */
        Float tanThetaI = std::tan(thetaI);
        Float cotThetaI = 1.0f/tanThetaI;

        /* Search interval -- everything is parameterized
           in the erf() domain */
        Float a = -1.0f, c = math::erf(cotThetaI);
        Float sample_x = std::max(sample[0], static_cast<Float>(1e-6));

        /* Start with a good initial guess */
        //Float b = (1-sample_x) * a + sample_x * c;

        /* We can do better (inverse of an approximation computed in Mathematica) */
        Float fit = 1.0f + thetaI*(-0.876f + thetaI*(0.4265f - 0.0594f*thetaI));
        Float b = c - (1.0f + c)*std::pow(1.0f - sample_x, fit);

        /* Normalization factor for the CDF */
        Float normalization = 1.0f/(1.0f + c + SQRT_PI_INV*tanThetaI*std::exp(-cotThetaI*cotThetaI));

        int it = 0;
        while (++it < 10) {
            /* Bisection criterion -- the oddly-looking
               boolean expression are intentional to check
               for NaNs at little additional cost */
            if (!(b >= a && b <= c))
                b = 0.5f * (a + c);

            /* Evaluate the CDF and its derivative
               (i.e. the density function) */
            Float invErf = math::erfinv(b);
            Float value = normalization*(1.0f + b + SQRT_PI_INV*tanThetaI*std::exp(-invErf*invErf)) - sample_x;
            Float derivative = normalization * (1.0f - invErf*tanThetaI);

            if (std::abs(value) < 1e-5f) break;

            /* Update bisection intervals */
            if (value > 0)
                c = b;
            else
                a = b;

            b -= value/derivative;
        }

        /* Now convert back into a slope value */
        slope[0] = math::erfinv(b);

        /* Simulate Y component */
        slope[1] = math::erfinv(2.0f*std::max(sample[1], static_cast<Float>(1e-6f)) - 1.0f);

        return slope;
    }

    FloatAD m_alpha;
    bool m_sampleVisible;
};


#endif //MICROFACET_DISTRB_H__
