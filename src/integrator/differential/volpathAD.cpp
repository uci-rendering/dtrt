#include "volpathAD.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"
#include "intersectionAD.h"
#include "stats.h"
#include <iomanip>
#include "omp.h"

#define STRATIFIED_SAMPLING
#define COLLECT_STATS

#define ALTERNATIVE_EDGE_SAMPLING
#define ALTERNATIVE_LINS_BOUNDARY_SAMPLING

#ifdef COLLECT_STATS
    static std::vector<Statistics> stats;
    static Eigen::ArrayXi medInconsistencies;
#endif


VolPathTracerAD::VolPathTracerAD() {
#ifdef COLLECT_STATS
    int nworker = omp_get_num_procs();
    stats.resize(nworker);
    medInconsistencies = Eigen::ArrayXi::Zero(nworker);
#endif
}


VolPathTracerAD::~VolPathTracerAD() {
#ifdef COLLECT_STATS
    for ( size_t i = 1; i < stats.size(); ++i ) stats[0].push(stats[i]);
    std::cout << "\n===== Extra Statistics =====\n";
    std::cout << std::setprecision(2) << std::fixed
              << "- dot(D) stats: mean = " << stats[0].getMean() << ", std = " << std::sqrt(stats[0].getVar()) << '\n'
              << "- Medium inconsistencies: " << medInconsistencies.sum()
              << std::endl;
#endif
}


static inline Float computeMisWeight(Float pdf1, Float pdf2) {
    // Power heuristic
    pdf1 *= pdf1; pdf2 *= pdf2;
    return pdf1/(pdf1 + pdf2);
}


Spectrum VolPathTracerAD::Li(const Scene& scene, RndSampler* sampler, const Ray& _ray, int max_depth, const Medium* init_med, bool incEmission) const {
    assert(max_depth >= 0);

    Ray ray(_ray);
    const Medium* ptr_med = init_med;

    Intersection its;
    Spectrum ret = Spectrum::Zero(), throughput = Spectrum::Ones();

    scene.rayIntersect(ray, true, its);
    for ( int depth = 0; depth <= max_depth; ++depth ) {
        int max_interactions = max_depth - depth - 1;
        Array2 rnd_1 = sampler->next2D();
        Array4 rnd_2 = sampler->next4D();
        Array3 rnd_3 = sampler->next3D();

        bool inside_med = ptr_med != nullptr &&
                          ptr_med->sampleDistance(ray, its.t, rnd_1, sampler, ray.org, throughput);
        if (inside_med) {
            if ( depth >= max_depth ) break;
            if ( throughput.isZero(Epsilon) ) break;

            // Light sampling
            Vector wo;
            Float pdf_nee;
            const PhaseFunction *ptr_phase = scene.phase_list[ptr_med->phase_id];
            auto value = scene.sampleAttenuatedEmitterDirect(ray.org, rnd_2, sampler, ptr_med, max_interactions, wo, pdf_nee);
            if ( !value.isZero(Epsilon) ) {
                auto phase_val = ptr_phase->eval(-ray.dir, wo);
                if ( phase_val > Epsilon ) {
                    Float phase_pdf = ptr_phase->pdf(-ray.dir, wo);
                    auto mis_weight = computeMisWeight(pdf_nee, phase_pdf);
                    ret += throughput*value*mis_weight*phase_val;
                }
            }

            // Phase function sampling & Indirect illumination
            Float phase_val = ptr_phase->sample(-ray.dir, Array2(rnd_3(0), rnd_3(1)), wo);
            if ( phase_val < Epsilon ) break;
            throughput *= phase_val;

            ray.dir = wo;
            Spectrum attenuated_radiance = scene.rayIntersectAndLookForEmitter(ray, false, sampler, ptr_med, max_interactions, its, pdf_nee);
            if (!attenuated_radiance.isZero(Epsilon)) {
                Float phase_pdf = ptr_phase->pdf(-ray.dir, wo);
                auto mis_weight = computeMisWeight(phase_pdf, pdf_nee);
                ret += throughput*attenuated_radiance*mis_weight;
            }

            incEmission = false;
        } else {
            if (!its.isValid()) break;
            if (its.isEmitter() && incEmission) ret += throughput*its.Le(-ray.dir);
            if (depth >= max_depth) break;

            // Light sampling
            Vector wo;
            Float pdf_nee;
            if ( !its.ptr_bsdf->isNull() ) {
                Spectrum value = scene.sampleAttenuatedEmitterDirect(its, rnd_2, sampler, ptr_med, max_interactions, wo, pdf_nee);
                if ( !value.isZero(Epsilon) ) {
                    Spectrum bsdf_val = its.evalBSDF(wo);
                    Float bsdf_pdf = its.pdfBSDF(wo);
                    Float mis_weight = computeMisWeight(pdf_nee, bsdf_pdf);
                    ret += throughput*value*bsdf_val*mis_weight;
                }
            }

            // BSDF sampling & Indirect illumination
            Float bsdf_pdf, bsdf_eta;
            Spectrum bsdf_weight = its.sampleBSDF(rnd_3, wo, bsdf_pdf, bsdf_eta);
            if ( bsdf_weight.isZero(Epsilon) ) break;
            wo = its.toWorld(wo);
            ray = Ray(its.p, wo);

            if ( its.isMediumTransition() ) ptr_med = its.getTargetMedium(wo);
            if ( its.ptr_bsdf->isNull() ) {
                scene.rayIntersect(ray, true, its);
                continue;
            }

            Spectrum attenuated_radiance = scene.rayIntersectAndLookForEmitter(ray, true, sampler, ptr_med, max_interactions, its, pdf_nee);
            throughput *= bsdf_weight;
            if (!attenuated_radiance.isZero()) {
                Float mis_weight = computeMisWeight(bsdf_pdf, pdf_nee);
                ret += throughput*attenuated_radiance*mis_weight;
            }

            incEmission = false;
        }
    }
    return ret;
}


SpectrumAD VolPathTracerAD::LiAD(const Scene &scene, const IntersectionAD &its, RndSampler* sampler, const RayAD &_ray,
                                 const Medium *med, int max_depth, EMode mode, int nEdgeSamples, Float ddistCoeff) const
{
#ifdef COLLECT_STATS
    const int tid = omp_get_thread_num();
    assert(tid < static_cast<int>(stats.size()));
#endif

    assert(max_depth >= 0);
    if ( mode == 0 ) return SpectrumAD();

    RayAD ray(_ray);

    Array2 rnd_1 = sampler->next2D();
    Array4 rnd_2 = sampler->next4D();
    Array3 rnd_3 = sampler->next3D();

    SpectrumAD ret, throughput(Spectrum::Ones());
    IntersectionAD its1;
    if ( med != nullptr && med->sampleDistanceAD(ray, its.t, rnd_1, sampler, ray.org, throughput) ) {
        /*
         * Volumetric
         */
        if ( max_depth == 0 || throughput.isZero(Epsilon) ) return ret;

        const PhaseFunction *ptr_phase = scene.phase_list[med->phase_id];
        // Area sampling
        {
            // Light sampling
            Float pdf_nee;
            VectorAD wo;
            if ( mode & EVolumeMain ) {
                SpectrumAD value = scene.sampleAttenuatedEmitterDirectAD(ray.org, rnd_2, sampler, med, max_depth - 1, wo, pdf_nee);
                if ( !value.isZero(Epsilon) ) {
                    FloatAD phase_val = ptr_phase->evalAD(-ray.dir, wo);
                    if ( phase_val > Epsilon ) {
                        Float phase_pdf = ptr_phase->pdf(-ray.dir.val, wo.val);
                        Float mis_weight = computeMisWeight(pdf_nee, phase_pdf);
                        ret += throughput*value*mis_weight*phase_val;
                    }
                }
            }

            // Phase function sampling & Indirect illumination
            FloatAD phase_val = ptr_phase->sampleAD(-ray.dir, Array2(rnd_3(0), rnd_3(1)), wo);
            if ( phase_val > Epsilon ) {
                assert(wo.der.isZero(Epsilon));
                SpectrumAD throughput1 = throughput*phase_val;

                ray.dir = wo;
                SpectrumAD attenuated_radiance = scene.rayIntersectAndLookForEmitterAD(ray, false, sampler, med, max_depth - 1, its1, pdf_nee);
                if ( !attenuated_radiance.isZero(Epsilon) && (mode & EVolumeMain) ) {
                    Float phase_pdf = ptr_phase->pdf(-ray.dir.val, wo.val);
                    Float mis_weight = computeMisWeight(phase_pdf, pdf_nee);
                    ret += throughput1*attenuated_radiance*mis_weight;
                }

                ret += throughput1*LiAD(scene, its1, sampler, ray, med, max_depth - 1,
                                        static_cast<EMode>(mode & ~EEmission), nEdgeSamples, ddistCoeff);
            }
        }

        // Edge sampling, volume
        if ( nEdgeSamples > 0 && (mode & EVolumeBoundary2) ) {
            Eigen::Array<Float, nder, 3> der = Eigen::Array<Float, nder, 3>::Zero();
            const VectorAD &x1 = ray.org;

            for ( int i = 0; i < nEdgeSamples; ++i ) {
#ifdef STRATIFIED_SAMPLING
                Float rnd = static_cast<Float>((i + sampler->next1D())/nEdgeSamples);
#else
                Float rnd = sampler->next1D();
#endif

                int shape_id; Float t, edgePdf;
                const Edge* ptr_edge = scene.sampleEdge(x1.val, nullptr, rnd, shape_id, edgePdf);
                t = rnd;

                const Shape *shape = scene.shape_list[shape_id];
                int isSihoulette = shape->isSihoulette(*ptr_edge, x1.val);
#ifdef INCLUDE_NULL_BOUNDARIES
                bool isValid = ( isSihoulette == 2 || (isSihoulette == 1 && !scene.bsdf_list[shape->bsdf_id]->isNull()) );
#else
                bool isValid = ( isSihoulette > 0 && !scene.bsdf_list[shape->bsdf_id]->isNull() );
#endif
                if ( isValid ) {
                    const VectorAD e0 = shape->getVertexAD(ptr_edge->v0) - x1, e1 = shape->getVertexAD(ptr_edge->v1) - x1;
                    const Vector norm = e0.val.cross(e1.val).normalized();
                    VectorAD w1AD;
                    Float dist;

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
                        FloatAD distAD = w1AD.norm();
                        w1AD /= distAD;

                        dist = distAD.val;
                        edgePdf /= theta;
                    }
#else
                    {
                        w1AD = e0*(1.0f - t) + e1*t;
                        FloatAD distAD = w1AD.norm();
                        w1AD /= distAD;

                        dist = distAD.val;
                        Float cosTheta = -w1AD.val.dot((e1.val - e0.val).normalized()),
                              sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);
                        edgePdf *= dist/(ptr_edge->length*sinTheta);
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
                        Spectrum phaseVal = throughput.val.transpose()*ptr_phase->eval(-_ray.dir.val, w1AD.val);
                        if ( !phaseVal.isZero(Epsilon) ) {
                            Float trans = scene.evalTransmittance(Ray(x1.val, w1AD.val), false, med, dist, sampler, max_depth - 1);
                            if ( trans > Epsilon ) {
                                Intersection its1;
                                Spectrum deltaFunc;

                                Vector w2;
                                w2 = (w1AD.val - AngleEpsilon*norm).normalized();
                                deltaFunc  = Li(scene, sampler, Ray(x1.val, w2), max_depth - 1, med);
                                w2 = (w1AD.val + AngleEpsilon*norm).normalized();
                                deltaFunc -= Li(scene, sampler, Ray(x1.val, w2), max_depth - 1, med);

                                if ( !deltaFunc.isZero(Epsilon) )
                                    for ( int j = 0; j < nder; ++j )
                                        der.row(j) += norm.dot(w1AD.grad(j))*phaseVal*deltaFunc/edgePdf;
                            }
                        }
                    }
                }
            }
            if ( nEdgeSamples > 0 ) ret.der += der/static_cast<Float>(nEdgeSamples);
        }
    } else {
        /*
         * Interfacial
         */
        if ( !its.isValid() || throughput.isZero(Epsilon) ) return ret;
        if ( its.isMediumTransition() && its.getTargetMedium(-ray.dir.val) != med ) {
#ifdef COLLECT_STATS
            ++medInconsistencies[tid];
#endif
            return ret;
        }

        if ( its.isEmitter() && (mode & EEmission) ) ret += throughput*its.Le(-ray.dir);
        if ( max_depth == 0 ) return ret;

        // Boundary term (Lins(x0))
        if ( med != nullptr && !its.t.der.isZero(Epsilon) && (mode & EVolumeBoundary1) ) {
            const Float derVal = its.t.der.abs().maxCoeff();
#ifdef COLLECT_STATS
            stats[tid].push(derVal);
#endif

            const Intersection its0 = its.toIntersection();
            Vector n = its0.geoFrame.n;
            if ( n.dot(ray.dir.val) < 0.0f ) n = -n;
            Eigen::Array<Float, nder, 3> der = (its.t.der.matrix()*med->sigS(its0.p - ShadowEpsilon*n).transpose().matrix()).array();
            const PhaseFunction *phase = scene.phase_list[med->phase_id];

            const int nsamples = clamp(static_cast<int>(ddistCoeff*derVal), 1, 100);
            Spectrum Lins = Spectrum::Zero();
            for ( int i = 0; i < nsamples; ++i ) {
                Vector wo;
#ifdef ALTERNATIVE_LINS_BOUNDARY_SAMPLING
                Float pdf_nee, pdf_phase, pdf_bsdf, mis_weight;
                Float woDotN, phase_val;
                Spectrum bsdf_val, attenuated_radiance;
                Intersection its1;
                Ray ray1;
                const Medium *med1;

                if ( its0.ptr_bsdf->isNull() ) {
                    // Light sampling
                    attenuated_radiance = scene.sampleAttenuatedEmitterDirect(its0, sampler->next4D(), sampler, med, max_depth - 1, wo, pdf_nee, true);
                    if ( !attenuated_radiance.isZero(Epsilon) ) {
                        wo = its0.toWorld(wo);
                        phase_val = phase->eval(-ray.dir.val, wo);
                        if ( phase_val > Epsilon ) {
                            pdf_phase = phase->pdf(-ray.dir.val, wo);
                            mis_weight = computeMisWeight(pdf_nee, pdf_phase);
                            Lins += phase_val*attenuated_radiance*mis_weight;
                        }
                    }

                    // Phase sampling
                    if ( (phase_val = phase->sample(-ray.dir.val, sampler->next2D(), wo)) > Epsilon ) {
                        int max_depth1 = max_depth - 1;
                        if ( (woDotN = wo.dot(n)) > Epsilon ) {
                            if ( its0.isEmitter() ) Lins += phase_val*its0.Le(-wo);
                            --max_depth1;
                        }

                        if ( max_depth1 > 0 ) {
                            med1 = its0.isMediumTransition() ? its0.getTargetMedium(wo) : med;
                            ray1 = Ray(its0.p, wo);
                            attenuated_radiance = scene.rayIntersectAndLookForEmitter(ray1, true, sampler, med1, max_depth1, its1, pdf_nee);
                            if ( !attenuated_radiance.isZero(Epsilon) ) {
                                pdf_phase = phase->pdf(-ray.dir.val, wo);
                                mis_weight = computeMisWeight(pdf_phase, pdf_nee);
                                Lins += phase_val*attenuated_radiance*mis_weight;
                            }
                            Lins += phase_val*Li(scene, sampler, ray1, max_depth1, med1, false);
                        }
                    }
                } else {
                    // Light sampling
                    attenuated_radiance = scene.sampleAttenuatedEmitterDirect(its0, sampler->next4D(), sampler, med, max_depth - 1, wo, pdf_nee);
                    if ( !attenuated_radiance.isZero(Epsilon) ) {
                        wo = its0.toWorld(wo);
                        if ( wo.dot(n) < -Epsilon ) {
                            phase_val = phase->eval(-ray.dir.val, wo);
                            if ( phase_val > Epsilon ) {
                                pdf_phase = phase->pdf(-ray.dir.val, wo);
                                mis_weight = computeMisWeight(pdf_nee, pdf_phase);
                                Lins += phase_val*attenuated_radiance*mis_weight;
                            }
                        }
                    }

                    // Phase sampling
                    if ( (phase_val = phase->sample(-ray.dir.val, sampler->next2D(), wo)) > Epsilon ) {
                        woDotN = wo.dot(n);
                        if ( woDotN < -Epsilon ) {
                            ray1 = Ray(its0.p, wo);
                            attenuated_radiance = scene.rayIntersectAndLookForEmitter(ray1, true, sampler, med, max_depth - 1, its1, pdf_nee);
                            if ( !attenuated_radiance.isZero(Epsilon) ) {
                                pdf_phase = phase->pdf(-ray.dir.val, wo);
                                mis_weight = computeMisWeight(pdf_phase, pdf_nee);
                                Lins += phase_val*attenuated_radiance*mis_weight;
                            }
                            Lins += phase_val*Li(scene, sampler, ray1, max_depth - 1, med, false);
                        }
                        else if ( woDotN > Epsilon ) {
                            if ( its0.isEmitter() ) Lins += phase_val*its0.Le(-wo);
                            if ( max_depth > 1 ) {
                                its1 = its0;
                                its1.wi = its1.toLocal(-wo);

                                // Light sampling
                                attenuated_radiance = scene.sampleAttenuatedEmitterDirect(its1, sampler->next4D(), sampler, med, max_depth - 2, wo, pdf_nee);
                                if ( !attenuated_radiance.isZero(Epsilon) ) {
                                    bsdf_val = its1.evalBSDF(wo);
                                    pdf_bsdf = its1.pdfBSDF(wo);
                                    mis_weight = computeMisWeight(pdf_nee, pdf_bsdf);
                                    Lins += phase_val*bsdf_val*attenuated_radiance*mis_weight;
                                }

                                // BSDF sampling & Indirect illumination
                                Float eta;
                                bsdf_val = its1.sampleBSDF(sampler->next3D(), wo, pdf_bsdf, eta);
                                if ( !bsdf_val.isZero(Epsilon) ) {
                                    Spectrum throughput1 = phase_val*bsdf_val;
                                    wo = its1.toWorld(wo);
                                    med1 = its1.isMediumTransition() ? its1.getTargetMedium(wo) : med;
                                    ray1 = Ray(its1.p, wo);

                                    attenuated_radiance = scene.rayIntersectAndLookForEmitter(ray1, true, sampler, med1, max_depth - 2, its1, pdf_nee);
                                    if (!attenuated_radiance.isZero()) {
                                        mis_weight = computeMisWeight(pdf_bsdf, pdf_nee);
                                        Lins += throughput1*attenuated_radiance*mis_weight;
                                    }
                                    Lins += throughput1*Li(scene, sampler, ray1, max_depth - 2, med1, false);
                                }
                            }
                        }
                    }
                }
#else
                Float phase_val = phase->sample(-ray.dir.val, sampler->next2D(), wo);
                if ( phase_val > Epsilon ) {
                    woDotN = wo.dot(n);
                    if ( woDotN < -Epsilon )
                        Lins += phase_val*Li(scene, sampler, Ray(its0.p, wo), max_depth - 1, med);
                    else if ( woDotN > Epsilon ) {
                        if ( its0.isEmitter() ) Lins += phase_val*its0.Le(-wo);
                        if ( max_depth > 1 ) {
                            Intersection its1(its0);
                            its1.wi = its1.toLocal(-wo);
                            Float bsdf_pdf, bsdf_eta;
                            Spectrum bsdf_val = its1.sampleBSDF(sampler->next3D(), wo, bsdf_pdf, bsdf_eta);
                            if ( !bsdf_val.isZero(Epsilon) ) {
                                wo = its1.toWorld(wo);
                                const Medium *med1 = its1.isMediumTransition() ? its1.getTargetMedium(wo) : med;
                                Lins += phase_val*bsdf_val*Li(scene, sampler, Ray(its1.p, wo), max_depth - 2, med1);
                            }
                        }
                    }
                }
#endif
            }
            Lins /= static_cast<Float>(nsamples);
            ret.der += der.rowwise()*(Lins.transpose()*throughput.val);
        }

        // Area sampling
        {
            // Light sampling
            VectorAD wo;
            Float pdf_nee;
            if ( !its.ptr_bsdf->isNull() && (mode & ESurfaceMain) ) {
                SpectrumAD value = scene.sampleAttenuatedEmitterDirectAD(its, rnd_2, sampler, med, max_depth - 1, wo, pdf_nee);
                if ( !value.isZero(Epsilon) ) {
                    SpectrumAD bsdf_val = its.evalBSDF(wo);
                    Float bsdf_pdf = its.pdfBSDF(wo.val);
                    Float mis_weight = computeMisWeight(pdf_nee, bsdf_pdf);
                    ret += throughput*value*bsdf_val*mis_weight;
                }
            }

            // BSDF sampling & Indirect illumination
            Float bsdf_pdf, bsdf_eta;
            SpectrumAD bsdf_weight = its.sampleBSDF(rnd_3, wo, bsdf_pdf, bsdf_eta);
            if ( !bsdf_weight.isZero(Epsilon) ) {
                wo = its.toWorld(wo);
                assert(its.ptr_bsdf->isNull() || wo.der.isZero(Epsilon));
                ray = RayAD(its.p, wo);

                const Medium *med1 = its.isMediumTransition() ? its.getTargetMedium(wo.val) : med;
                SpectrumAD throughput1 = throughput;
                EMode mode1;
                if ( its.ptr_bsdf->isNull() ) {
                    assert(ret.isZero(Epsilon));
                    scene.rayIntersectAD(ray, true, its1);
                    mode1 = mode;
                }
                else {
                    SpectrumAD attenuated_radiance = scene.rayIntersectAndLookForEmitterAD(ray, true, sampler, med1, max_depth - 1, its1, pdf_nee);
                    throughput1 *= bsdf_weight;
                    if ( !attenuated_radiance.isZero(Epsilon) && (mode & ESurfaceMain) ) {
                        Float mis_weight = computeMisWeight(bsdf_pdf, pdf_nee);
                        ret += throughput1*attenuated_radiance*mis_weight;
                    }
                    mode1 = static_cast<EMode>(mode & ~EEmission);
                }
                ret += throughput1*LiAD(scene, its1, sampler, ray, med1, max_depth - 1, mode1, nEdgeSamples, ddistCoeff);
            }
        }

        // Edge sampling, interface
        if ( !its.ptr_bsdf->isNull() && nEdgeSamples > 0 && (mode & ESurfaceBoundary) ) {
            Eigen::Array<Float, nder, 3> der = Eigen::Array<Float, nder, 3>::Zero();
            const VectorAD &x0 = its.p;
            const Intersection its0 = its.toIntersection();

            for ( int i = 0; i < nEdgeSamples; ++i ) {
#ifdef STRATIFIED_SAMPLING
                Float rnd = static_cast<Float>((i + sampler->next1D())/nEdgeSamples);
#else
                Float rnd = sampler->next1D();
#endif

                int shape_id; Float t, edgePdf;
                const Edge* ptr_edge = scene.sampleEdge(x0.val, &(its0.geoFrame), rnd, shape_id, edgePdf);
                if (ptr_edge == nullptr)
                    continue;
                t = rnd;

                const Shape *shape = scene.shape_list[shape_id];
                int isSihoulette = shape->isSihoulette(*ptr_edge, x0.val);
#ifdef INCLUDE_NULL_BOUNDARIES
                bool isValid = ( isSihoulette == 2 || (isSihoulette == 1 && !scene.bsdf_list[shape->bsdf_id]->isNull()) );
#else
                bool isValid = ( isSihoulette > 0 && !scene.bsdf_list[shape->bsdf_id]->isNull() );
#endif
                if ( isValid ) {
                    const VectorAD e0 = shape->getVertexAD(ptr_edge->v0) - x0, e1 = shape->getVertexAD(ptr_edge->v1) - x0;
                    const Vector norm = e0.val.cross(e1.val).normalized();
                    VectorAD w1AD;
                    Float dist;

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
                        FloatAD distAD = w1AD.norm();
                        w1AD /= distAD;

                        dist = distAD.val;
                        edgePdf /= theta;
                    }
#else
                    {
                        w1AD = e0*(1.0f - t) + e1*t;
                        FloatAD distAD = w1AD.norm();
                        w1AD /= distAD;

                        dist = distAD.val;
                        Float cosTheta = -w1AD.val.dot((e1.val - e0.val).normalized()),
                              sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);
                        edgePdf *= dist/(ptr_edge->length*sinTheta);
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
                        Spectrum bsdfVal = throughput.val.transpose()*its0.evalBSDF(its0.toLocal(w1AD.val));
                        if ( !bsdfVal.isZero(Epsilon) ) {
                            const Medium *med1 = its0.isMediumTransition() ? its0.getTargetMedium(w1AD.val) : med;

                            Float trans = scene.evalTransmittance(Ray(x0.val, w1AD.val), true, med1, dist, sampler, max_depth - 1);
                            if ( trans > Epsilon ) {
                                Intersection its1;
                                Spectrum deltaFunc;

                                Vector w2;
                                w2 = (w1AD.val - AngleEpsilon*norm).normalized();
                                deltaFunc  = Li(scene, sampler, Ray(x0.val, w2), max_depth - 1, med1);
                                w2 = (w1AD.val + AngleEpsilon*norm).normalized();
                                deltaFunc -= Li(scene, sampler, Ray(x0.val, w2), max_depth - 1, med1);

                                if ( !deltaFunc.isZero(Epsilon) )
                                    for ( int j = 0; j < nder; ++j )
                                        der.row(j) += norm.dot(w1AD.grad(j))*bsdfVal*deltaFunc/edgePdf;
                            }
                        }
                    }
                }
            }
            if ( nEdgeSamples > 0 ) ret.der += der/static_cast<Float>(nEdgeSamples);
        }
    }

    return ret;
}


Spectrum VolPathTracerAD::pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const {
    const auto &camera = scene.camera;
    const Medium* init_med = camera.getMedID() == -1 ? nullptr : scene.medium_list[camera.getMedID()];

    Ray ray(scene.camera.samplePrimaryRay(x, y));
    return Li(scene, sampler, ray, options.max_bounces, init_med);
}


SpectrumAD VolPathTracerAD::pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y) const {
    const auto &camera = scene.camera;
    const Medium* init_med = camera.getMedID() == -1 ? nullptr : scene.medium_list[camera.getMedID()];

    RayAD ray(scene.camera.samplePrimaryRayAD(x, y));
    IntersectionAD its;
    EMode mode = options.mode >= 0 ? static_cast<EMode>(options.mode) : EAll;
    if ( scene.rayIntersectAD(ray, false, its) )
        return LiAD(scene, its, sampler, ray, init_med, options.max_bounces, mode, options.num_samples_secondary_edge, options.ddistCoeff);
    else
        return SpectrumAD();
}
