#include "volpath_simple.h"
#include "scene.h"
#include "ray.h"
#include "sampler.h"
#include <assert.h>
#include <omp.h>
#include <numeric>
#include <iostream>

Spectrum3f VolPathTracerSimple::radiance(const Scene& scene, RndSampler* sampler, Ray& ray, int max_depth, const Medium* med_default) const {
	Intersection its;
	Spectrum ret = Spectrum::Zero();
	const Medium* ptr_med = med_default;
	Spectrum throughput = Spectrum::Ones();
	Float eta = 1.0;
	int depth = 0;
	bool incEmission = true;
	scene.rayIntersect(ray, false, its);
	while (depth <= max_depth) {
		int max_interactions = max_depth-depth-1;
		Array2 rnd_1 = sampler->next2D();
		Array4 rnd_2 = sampler->next4D();
		Array3 rnd_3 = sampler->next3D();
		// inside medium
		bool inside_med = ptr_med != nullptr &&
						  ptr_med->sampleDistance(Ray(ray), its.t, rnd_1, sampler, ray.org, throughput);
		if (inside_med) {
			if (depth >= max_depth) break;
			if (throughput.isZero())
				break;
			// Direct illumination
			Vector wo;
			Float pdf_nee;
			const PhaseFunction* ptr_phase = scene.phase_list[ptr_med->phase_id];
			auto value = scene.sampleAttenuatedEmitterDirect(ray.org, rnd_2, sampler, ptr_med, max_interactions, wo, pdf_nee);

			if (!value.isZero()) {
				auto phase_val = ptr_phase->eval(-ray.dir, wo);
				if (phase_val != 0)
					ret += throughput * value * phase_val;
			}
			// Indirect illumination
			Float phase_val = ptr_phase->sample(-ray.dir, Array2(rnd_3(0), rnd_3(1)), wo);
			if (phase_val == 0)
				break;
			throughput *= phase_val;
			// trace a new ray in this direction
			ray.dir = wo;
			scene.rayIntersect(ray, 0.0f, its);
			incEmission = false;
		} else {
			if (!its.isValid())
				break;
			if (its.isEmitter() && incEmission)
				ret += throughput * its.Le(-ray.dir);
			if (depth >= max_depth) break;

			Vector wo;
			if (!its.ptr_bsdf->isNull()) {
				// Direct illumination
				Float pdf_nee;
				auto value = scene.sampleAttenuatedEmitterDirect(its, rnd_2, sampler, ptr_med, max_interactions, wo, pdf_nee);
				if (!value.isZero())
					ret += throughput * value * its.evalBSDF(wo);
				incEmission = false;
			}

			// Indirect illumination
			Float bsdf_pdf, bsdf_eta;
			auto bsdf_weight = its.sampleBSDF(rnd_3, wo, bsdf_pdf, bsdf_eta);
			if (bsdf_weight.isZero())
				break;
			throughput = throughput * bsdf_weight;
			eta *= bsdf_eta;

			wo = its.toWorld(wo);
			ray = Ray(its.p, wo);

			if (its.isMediumTransition())
				ptr_med = its.getTargetMedium(wo);

			scene.rayIntersect(ray, true, its);
		}
		depth++;
	}
	return ret.cast<float>();
}


void VolPathTracerSimple::render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const {
	const auto &camera = scene.camera;
	const Medium* med_default = camera.getMedID() == -1 ? nullptr : scene.medium_list[camera.getMedID()];
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
				sample_vals[idx_sample] = radiance(scene, &sampler, ray, options.max_bounces, med_default);
				pixel_val += sample_vals[idx_sample];
			}
			// write to the image buffer
			pixel_val /= options.num_samples;
			rendered_image[idx_pixel*3 	  ] = pixel_val(0);
			rendered_image[idx_pixel*3 + 1] = pixel_val(1);
			rendered_image[idx_pixel*3 + 2] = pixel_val(2);
		}

		progressIndicator(Float(index_block)/num_block);
 	}
 	std::cout << std::endl;
}