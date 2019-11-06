#include "direct.h"
#include "scene.h"
#include "ray.h"
#include "sampler.h"


Spectrum3f DirectIntegrator::radiance(const Scene& scene, RndSampler* sampler, Ray& ray) const
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
    return ret.cast<float>();
}


void DirectIntegrator::render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const {
    const auto &camera = scene.camera;
    bool cropped = camera.rect.isValid();
    int num_pixels = cropped ? camera.rect.crop_width * camera.rect.crop_height 
                             : camera.width * camera.height;
    int size_block = 16;
    int num_block = std::ceil((Float)num_pixels/size_block);
    for (int index_block = 0; index_block < num_block; index_block++) {
        int block_start = index_block*size_block;
        int block_end = std::min((index_block+1)*size_block, num_pixels);
#pragma omp parallel for
        for (int idx_pixel = block_start; idx_pixel < block_end; idx_pixel++) {
            int ix = cropped ? camera.rect.offset_x + idx_pixel % camera.rect.crop_width
                             : idx_pixel % camera.width;
            int iy = cropped ? camera.rect.offset_y + idx_pixel / camera.rect.crop_width
                             : idx_pixel / camera.width;
            RndSampler sampler(options.seed, idx_pixel);
            std::vector<Spectrum3f> sample_vals(options.num_samples, Spectrum3f::Zero());
            Spectrum3f pixel_val = Spectrum3f::Zero();
            for (int idx_sample = 0; idx_sample < options.num_samples; idx_sample++) {
                Ray ray = camera.samplePrimaryRay(ix, iy, sampler.next2D());
                sample_vals[idx_sample] = radiance(scene, &sampler, ray);
                pixel_val += sample_vals[idx_sample];
            }
            // write to the image buffer
            pixel_val /= options.num_samples;
            rendered_image[idx_pixel*3    ] = pixel_val(0);
            rendered_image[idx_pixel*3 + 1] = pixel_val(1);
            rendered_image[idx_pixel*3 + 2] = pixel_val(2);
        }

        progressIndicator(Float(index_block)/num_block);
    }
    std::cout << std::endl;
}
