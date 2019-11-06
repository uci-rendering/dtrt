#pragma once
#ifndef PHASE_H__
#define PHASE_H__

#include "fwd.h"
#include "ptr.h"
#include "frame.h"
#include "utils.h"

struct PhaseFunction {
    virtual Float sample(const Vector& wi, const Vector2 &rnd2, Vector& wo) const = 0;
    virtual FloatAD sampleAD(const VectorAD& wi, const Vector2 &rnd2, VectorAD &wo) const {
        sample(wi.val, rnd2, wo.val);
        wo.zeroGrad();
        Float _pdf = pdf(wi.val, wo.val);
        return _pdf > Epsilon ? evalAD(wi, wo)/_pdf : FloatAD();
    }

    virtual Float pdf(const Vector& wi, const Vector& wo) const = 0;
    virtual Float eval(const Vector& wi, const Vector& wo) const = 0;
    virtual FloatAD evalAD(const VectorAD& wi, const VectorAD& wo) const { assert(false); return FloatAD(); }
};

struct HGPhaseFunction : PhaseFunction {
    inline HGPhaseFunction(Float g) : g(g) { }
    inline HGPhaseFunction(float g, ptr<float> dG) : g(g, Eigen::Map<Eigen::Array<float, nder, 1> >(dG.get(), nder, 1).cast<Float>()) {}

    Float eval(const Vector& wi, const Vector& wo) const {
        Float temp = 1.0f + g.val*g.val + 2.0f*g.val*wi.dot(wo);
        return INV_FOURPI*(1.0f - g.val*g.val)/(temp*std::sqrt(temp));
    }

    FloatAD evalAD(const VectorAD& wi, const VectorAD& wo) const {
        FloatAD temp = 1.0f + g*g + 2.0f*g*wi.dot(wo);
        return INV_FOURPI*(1.0f - g*g)/temp.pow(1.5f);
    }

    Float sample(const Vector& wi, const Vector2 &rnd2, Vector& wo) const {
        Float cosTheta;
        if ( std::abs(g.val) < Epsilon ) {
            cosTheta = 1.0f - 2.0f*rnd2.x();
        } else {
            Float sqrTerm = (1.0f - g.val*g.val)/(1.0f - g.val + 2.0f*g.val*rnd2.x());
            cosTheta = (1.0f + g.val*g.val - sqrTerm*sqrTerm)/(2.0f*g.val);
        }
        Float sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);
        Float sinPhi = std::sin(2.0f*M_PI*rnd2.y()),
              cosPhi = std::cos(2.0f*M_PI*rnd2.y());
        wo = Frame(wi).toWorld(Vector(sinTheta*cosPhi, sinTheta*sinPhi, cosTheta));
        return 1.0f;
    }

    FloatAD sampleAD(const VectorAD& wi, const Vector2 &rnd2, VectorAD &wo) const {
        sample(wi.val, rnd2, wo.val);
        wo.zeroGrad();
        FloatAD tmp = evalAD(wi, wo);
        return tmp.val > Epsilon ? tmp/tmp.val : FloatAD();
    }

    Float pdf(const Vector& wi, const Vector& wo) const { return eval(wi, wo);}

    FloatAD g;
};

struct IsotropicPhaseFunction : PhaseFunction {
    Float eval(const Vector& wi, const Vector& wo) const { return INV_FOURPI;}
    FloatAD evalAD(const VectorAD& wi, const VectorAD& wo) const { return INV_FOURPI; }
    Float pdf(const Vector& wi, const Vector& wo) const { return INV_FOURPI; }
    
    Float sample(const Vector& wi, const Vector2 &rnd2, Vector& wo) const {
        wo = squareToUniformSphere(rnd2);
        return 1.0f;
    }

    FloatAD sampleAD(const VectorAD& wi, const Vector2 &rnd2, Vector &wo) const {
        wo = squareToUniformSphere(rnd2);
        return 1.0f;
    }
};

#endif