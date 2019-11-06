#include "directAD.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"


SpectrumAD DirectIntegratorAD::radianceAD(const Scene& scene, RndSampler* sampler, const RayAD& _ray, int nEdgeSamples) const
{
    RayAD ray(_ray);

    IntersectionAD its;
    SpectrumAD ret;
    // Perform the first intersection
    scene.rayIntersectAD(ray, true, its);
    if (its.isValid()) {
        if (its.isEmitter()) ret += its.Le(-ray.dir);

        // Area sampling
        {
            SpectrumAD value;

            // Light sampling
            Float pdf_nee;
            VectorAD wo;
            value = scene.sampleEmitterDirectAD(its, sampler->next4D(), sampler, wo, pdf_nee);
            if ( !value.isZero(Epsilon) ) {
                SpectrumAD bsdf_val = its.evalBSDF(wo);
                Float bsdf_pdf = its.pdfBSDF(wo.val);
                Float mis_weight = pdf_nee/(pdf_nee + bsdf_pdf);
                ret += value*bsdf_val*mis_weight;
            }

            // BSDF sampling
            Float bsdf_pdf, bsdf_eta;
            value = its.sampleBSDF(sampler->next3D(), wo, bsdf_pdf, bsdf_eta);
            if ( !value.isZero(Epsilon) ) {
                wo = its.toWorld(wo);
                assert(wo.grad().isZero(Epsilon));
                ray = RayAD(its.p, wo);
                IntersectionAD its1;
                if (scene.rayIntersectAD(ray, true, its1)) {
                    if (its1.isEmitter()) {
                        SpectrumAD light_contrib = its1.Le(-ray.dir);
                        if (!light_contrib.val.isZero()) {
                            Float dist_sq = (its1.p.val - ray.org.val).squaredNorm();
                            Float geometry_term = its1.wi.val.z()/dist_sq;
                            pdf_nee = scene.pdfEmitterSample(its1)/geometry_term;
                            Float mis_weight = bsdf_pdf/(pdf_nee + bsdf_pdf);
                            ret += value*light_contrib*mis_weight;
                        }
                    }
                }
            }
        }

        // Edge sampling
        {
            Intersection its0 = its.toIntersection();
            Eigen::Array<Float, nder, 3> der = Eigen::Array<Float, nder, 3>::Zero();

            const VectorAD &x1 = its.p;
            for ( int i = 0; i < nEdgeSamples; ++i ) {
                int shape_id;
                Float t, edgePdf;
                const Edge* ptr_edge = scene.sampleEdge(x1.val, &(its0.geoFrame), sampler->next1D(), shape_id, t, edgePdf);
                if (ptr_edge == nullptr)
                    continue;
                const Shape *shape = scene.shape_list[shape_id];
                if ( shape->isSihoulette(*ptr_edge, x1.val) ) {
                    const VectorAD &v0 = shape->getVertexAD(ptr_edge->v0), &v1 = shape->getVertexAD(ptr_edge->v1);

                    VectorAD y1 = v0*(1.0f - t) + v1*t, w1AD = y1 - x1;
                    Vector tang = (v1.val - v0.val).normalized(),
                           norm = (v0.val - x1.val).cross(v1.val - x1.val).normalized();
                    FloatAD distAD = w1AD.norm();
                    w1AD /= distAD;
                    assert(std::abs(w1AD.grad().dot(w1AD.val)) < Epsilon);

                    Spectrum bsdfVal = its0.evalBSDF(its0.toLocal(w1AD.val));
                    if ( !bsdfVal.isZero(Epsilon) )
                        if ( scene.isVisible(x1.val, true, y1.val, true) ) {
                            Intersection its1;
                            Spectrum deltaFunc = Spectrum::Zero();

                            Vector w2;
                            w2 = (w1AD.val - AngleEpsilon*norm).normalized();
                            if ( scene.rayIntersect(Ray(x1.val, w2), true, its1) )
                                if ( its1.isEmitter() ) deltaFunc += its1.Le(-w1AD.val);
                            w2 = (w1AD.val + AngleEpsilon*norm).normalized();
                            if ( scene.rayIntersect(Ray(x1.val, w2), true, its1) )
                                if ( its1.isEmitter() ) deltaFunc -= its1.Le(-w1AD.val);

                            if ( !deltaFunc.isZero(Epsilon) ) {
                                Float cosTheta = -w1AD.val.dot(tang),
                                      sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);

                                for ( int j = 0; j < nder; ++j )
                                    der.row(j) += norm.dot(w1AD.grad(j))*bsdfVal*deltaFunc*sinTheta/(edgePdf*distAD.val);
                            }
                        }
                }
            }
            ret.der += der/static_cast<Float>(nEdgeSamples);
        }
    }
    return ret;
}


Spectrum DirectIntegratorAD::radiance(const Scene& scene, RndSampler* sampler, Ray ray) const
{
    Intersection its;
    Spectrum ret = Spectrum::Zero();
    // Perform the first intersection
    scene.rayIntersect(ray, true, its);
    if (its.isValid()) {
        if (its.isEmitter())
            ret += its.Le(-ray.dir);
        // Direct illumination
        Float pdf_nee;
        Vector wo;
        auto value = scene.sampleEmitterDirect(its, sampler->next4D(), sampler, wo, pdf_nee);
        if (!value.isZero()) {
            auto bsdf_val = its.evalBSDF(wo);
            Float bsdf_pdf = its.pdfBSDF(wo);
            auto mis_weight = square(pdf_nee) / (square(pdf_nee) + square(bsdf_pdf));
            ret += value * bsdf_val * mis_weight;
        }

        // Indirect illumination
        Float bsdf_pdf, bsdf_eta;
        auto bsdf_weight = its.sampleBSDF(sampler->next3D(), wo, bsdf_pdf, bsdf_eta);
        if (!bsdf_weight.isZero()) {
            wo = its.toWorld(wo);
            ray = Ray(its.p, wo);
            if (scene.rayIntersect(ray, true, its)) {
                if (its.isEmitter()) {
                    Spectrum light_contrib = its.Le(-ray.dir);
                    if (!light_contrib.isZero()) {
                        auto dist_sq = (its.p - ray.org).squaredNorm();
                        auto geometry_term = its.wi.z() / dist_sq;
                        pdf_nee = scene.pdfEmitterSample(its) / geometry_term;
                        auto mis_weight = square(bsdf_pdf) / (square(pdf_nee) + square(bsdf_pdf));
                        ret += bsdf_weight * light_contrib * mis_weight;
                    }
                }
            }
        }
    }
    return ret;
}


Spectrum DirectIntegratorAD::pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const {
    Ray ray(scene.camera.samplePrimaryRay(x, y));
    return radiance(scene, sampler, ray);
}


SpectrumAD DirectIntegratorAD::pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const {
    RayAD ray(scene.camera.samplePrimaryRayAD(x, y));
    return radianceAD(scene, sampler, ray, options.num_samples_secondary_edge);
}
