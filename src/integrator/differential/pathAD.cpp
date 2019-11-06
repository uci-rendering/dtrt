#include "pathAD.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"
#include "intersectionAD.h"
#include <iomanip>

#define ALTERNATIVE_EDGE_SAMPLING

Spectrum PathTracerAD::Li(const Scene& scene, RndSampler* sampler, const Ray& _ray, int max_depth) const {
    Ray ray(_ray);
    Intersection its;
    Spectrum ret = Spectrum::Zero();
    scene.rayIntersect(ray, true, its);
    if (its.isValid()) {
        Spectrum throughput = Spectrum::Ones();
        Float eta = 1.0f;
        int depth = 0;
        while (depth <= max_depth && its.isValid()) {
            if (its.isEmitter() && depth == 0)
                ret += throughput*its.Le(-ray.dir);
            if (depth >= max_depth) break;
            // Direct illumination
            Float pdf_nee;
            Vector wo;
            auto value = scene.sampleEmitterDirect(its, sampler->next4D(), sampler, wo, pdf_nee);
            if (!value.isZero()) {
                auto bsdf_val = its.evalBSDF(wo);
                Float bsdf_pdf = its.pdfBSDF(wo);
                auto mis_weight = square(pdf_nee)/(square(pdf_nee) + square(bsdf_pdf));
                ret += throughput * value * bsdf_val * mis_weight;
            }
            // Indirect illumination
            Float bsdf_pdf, bsdf_eta;
            auto bsdf_weight = its.sampleBSDF(sampler->next3D(), wo, bsdf_pdf, bsdf_eta);
            if (bsdf_weight.isZero())
                break;

            wo = its.toWorld(wo);
            ray = Ray(its.p, wo);

            if (!scene.rayIntersect(ray, true, its))
                break;

            throughput *= bsdf_weight;
            eta *= bsdf_eta;
            if (its.isEmitter()) {
                Spectrum light_contrib = its.Le(-ray.dir);
                if (!light_contrib.isZero()) {
                    auto dist_sq = (its.p - ray.org).squaredNorm();
                    auto geometry_term = its.wi.z()/dist_sq;
                    pdf_nee = scene.pdfEmitterSample(its)/geometry_term;
                    auto mis_weight = square(bsdf_pdf)/(square(pdf_nee) + square(bsdf_pdf));
                    ret += throughput * light_contrib * mis_weight;
                }
            }

            depth++;
        }
    }
    return ret;
}


SpectrumAD PathTracerAD::LiAD(const Scene &scene, const IntersectionAD &its, RndSampler* sampler, const RayAD &_ray,
                              int max_depth, EMode mode, bool incEmission, int nEdgeSamples) const
{
    assert(max_depth >= 0);
    if ( mode == 0 ) return SpectrumAD();

    RayAD ray(_ray);

    SpectrumAD ret;
    if ( its.isValid() ) {
        if ( its.isEmitter() && (mode & EEmission) && incEmission ) ret += its.Le(-ray.dir);
        if ( max_depth == 0 ) return ret;

        // Area sampling
        {
            SpectrumAD value;

            // Light sampling
            Float pdf_nee;
            VectorAD wo;
            if (mode & ESurfaceMain) {
                value = scene.sampleEmitterDirectAD(its, sampler->next4D(), sampler, wo, pdf_nee);
                if ( !value.isZero(Epsilon)) {
                    SpectrumAD bsdf_val = its.evalBSDF(wo);
                    Float bsdf_pdf = its.pdfBSDF(wo.val);
                    Float mis_weight = square(pdf_nee)/(square(pdf_nee) + square(bsdf_pdf));
                    ret += value*bsdf_val*mis_weight;
                }
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
                    if ( its1.isEmitter() && (mode & ESurfaceMain)) {
                        SpectrumAD light_contrib = its1.Le(-ray.dir);
                        if (!light_contrib.val.isZero()) {
                            Float dist_sq = (its1.p.val - ray.org.val).squaredNorm();
                            Float geometry_term = its1.wi.val.z()/dist_sq;
                            pdf_nee = scene.pdfEmitterSample(its1)/geometry_term;
                            Float mis_weight = square(bsdf_pdf)/(square(pdf_nee) + square(bsdf_pdf));
                            ret += value*light_contrib*mis_weight;
                        }
                    }

                    // Indirect illumination
                    ret += value*LiAD(scene, its1, sampler, ray, max_depth - 1, mode, false, nEdgeSamples);
                }
            }
        }

        // Edge sampling
        if (nEdgeSamples > 0 && (mode & ESurfaceBoundary))
        {
            Intersection its0 = its.toIntersection();
            Eigen::Array<Float, nder, 3> der = Eigen::Array<Float, nder, 3>::Zero();

            const VectorAD &x1 = its.p;
            for ( int i = 0; i < nEdgeSamples; ++i ) {
                int shape_id;
                Float t, edgePdf;
                Float rnd = sampler->next1D();
                const Edge* ptr_edge = scene.sampleEdge(x1.val, &(its0.geoFrame), rnd, shape_id, edgePdf);                
                if (ptr_edge == nullptr)
                    continue;
                t = rnd;
                const Shape *shape = scene.shape_list[shape_id];
                if ( shape->isSihoulette(*ptr_edge, x1.val) ) {
                    const VectorAD e0 = shape->getVertexAD(ptr_edge->v0) - x1, e1 = shape->getVertexAD(ptr_edge->v1) - x1;
                    const Vector norm = e0.val.cross(e1.val).normalized();
                    VectorAD y1, w1AD;
#ifdef ALTERNATIVE_EDGE_SAMPLING
                    {
                        Vector ne0 = e0.val.normalized(), ne1 = e1.val.normalized();
                        Float theta = std::acos(ne0.dot(ne1)), sinTheta = std::sin(theta);
                        if ( theta < Epsilon || theta > M_PI - Epsilon ) continue;

                        Float w0 = std::sin(theta*(1.0f - t))/sinTheta, w1 = std::sin(theta*t)/sinTheta;
                        w0 /= e0.val.norm(); w1 /= e1.val.norm();
                        Float tmp = w0 + w1;
                        w0 /= tmp; w1 /= tmp;
                        if ( w0 < -Epsilon || w1 < -Epsilon || std::abs(w0 + w1 - 1.0f) > Epsilon) {
                            omp_set_lock(&messageLock);
                            std::cerr << std::scientific << std::setprecision(2)
                                      << "\nWarning: invalid edge sampling weight: (" << w0 << ", " << w1 << ")" << std::endl;
                            omp_unset_lock(&messageLock);
                            continue;
                        }

                        w1AD = e0*w0 + e1*w1;
                        y1 = x1 + w1AD;
                        FloatAD distAD = w1AD.norm();
                        w1AD /= distAD;
                        edgePdf /= theta;
                    }
#else
                    {
                        w1AD = e0*(1.0f - t) + e1*t;
                        y1 = x1 + w1AD;
                        FloatAD distAD = w1AD.norm();
                        w1AD /= distAD;
                        Float cosTheta = -w1AD.val.dot((e1.val - e0.val).normalized()),
                              sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);
                        edgePdf *= distAD.val/(ptr_edge->length*sinTheta);
                    }
#endif

                    Float tmp = (w1AD.der.transpose()*w1AD.val).array().abs().maxCoeff();
                    if ( tmp > Epsilon ) {
                        omp_set_lock(&messageLock);
                        std::cerr << std::scientific << std::setprecision(2)
                                  << "\nWarning: |dot(w', \\partial w')| nonzero: (" << tmp << ")" << std::endl;
                        omp_unset_lock(&messageLock);
                        continue;
                    }

                    if ( !w1AD.der.isZero(Epsilon) ) {
                        Spectrum bsdfVal = its0.evalBSDF(its0.toLocal(w1AD.val));
                        if ( !bsdfVal.isZero(Epsilon) ) {
                            if ( scene.isVisible(x1.val, true, y1.val, true) ) {
                                Intersection its1;
                                Spectrum deltaFunc = Spectrum::Zero();

                                Vector w2;
                                w2 = (w1AD.val - AngleEpsilon*norm).normalized();
                                deltaFunc += Li(scene, sampler, Ray(x1.val, w2), max_depth - 1);
                                w2 = (w1AD.val + AngleEpsilon*norm).normalized();
                                deltaFunc -= Li(scene, sampler, Ray(x1.val, w2), max_depth - 1);

                                if ( !deltaFunc.isZero(Epsilon) ) {
                                    for ( int j = 0; j < nder; ++j )
                                        der.row(j) += norm.dot(w1AD.grad(j))*bsdfVal*deltaFunc/edgePdf;
                                }
                            }
                        }
                    }
                }
            }
            ret.der += der/static_cast<Float>(nEdgeSamples);
        }
    }

    return ret;
}


Spectrum PathTracerAD::pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const {
    Ray ray(scene.camera.samplePrimaryRay(x, y));
    return Li(scene, sampler, ray, options.max_bounces);
}


SpectrumAD PathTracerAD::pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const {
    RayAD ray(scene.camera.samplePrimaryRayAD(x, y));
    IntersectionAD its;
    EMode mode = options.mode >= 0 ? static_cast<EMode>(options.mode) : EAll;
    return scene.rayIntersectAD(ray, false, its) ? LiAD(scene, its, sampler, ray, options.max_bounces, mode, true, options.num_samples_secondary_edge)
                                                 : SpectrumAD();
}
