#include "ptracer.h"
#include "scene.h"
#include "ray.h"
#include "sampler.h"
#include <assert.h>
#include <omp.h>
#include <numeric>
#include <iostream>

void ParticleTracer::render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const {
	int size_block = 100000;
	int num_block = options.num_samples;
	const auto &camera = scene.camera;
    bool cropped = camera.rect.isValid();
 	int num_pixels = cropped ? camera.rect.crop_width * camera.rect.crop_height 
 							 : camera.width * camera.height;
 	const int nworker = omp_get_num_procs();
	image_per_thread.resize(nworker);
	for (int i = 0; i < nworker; i++) {
		image_per_thread[i].resize(num_pixels);
	}

#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
	for (int index_block = 0; index_block < num_block; index_block++) {
        int block_start = index_block*size_block;
        int block_end = (index_block+1)*size_block;
		for (int index_sample = block_start; index_sample < block_end; index_sample++) {
			RndSampler sampler(options.seed, index_sample);
			int thread_id = omp_get_thread_num();
			traceParticle(scene, &sampler, options.max_bounces, thread_id);
		}

        if ( !options.quiet ) {
            omp_set_lock(&messageLock);
            progressIndicator(Float(index_block)/num_block);
            omp_unset_lock(&messageLock);
        }		
	}

	for (int ithread = 0; ithread < nworker; ithread++) {
		for (int idx_pixel = 0; idx_pixel < num_pixels; idx_pixel++) {
			for (int ichannel = 0; ichannel < 3; ichannel++) {
				rendered_image[idx_pixel*3 + ichannel] += image_per_thread[ithread][idx_pixel][ichannel];
			}
		}
	}

	size_t num_samples = size_t(size_block) * num_block;
	for (int idx_pixel = 0; idx_pixel < num_pixels; idx_pixel++) {
		for (int ichannel = 0; ichannel < 3; ichannel++) {
			rendered_image[idx_pixel*3 + ichannel] /= num_samples;
		}
	}
}

void ParticleTracer::traceParticle(const Scene& scene, RndSampler *sampler, int max_bounces, int thread_id) const {
	// Sample position on emitter
	Intersection its;
	Spectrum power = scene.sampleEmitterPosition(sampler->next4D(), its);
	// connect emitter to sensor directly
	handleEmission(its, scene, sampler, power, max_bounces, thread_id);
	Ray ray;
	ray.org = its.p;
	power *= its.ptr_emitter->sampleDirection(sampler->next2D(), ray.dir);
	ray.dir = its.shFrame.toWorld(ray.dir);
	int depth = 0;
	Spectrum throughput = Spectrum::Ones();
	const Medium* ptr_med = its.getTargetMedium(ray.dir);
	bool on_surface = true;
	while(!throughput.isZero() && depth < max_bounces) {
		scene.rayIntersect(ray, on_surface, its);
		int max_interactions = max_bounces-depth-1;
		bool inside_med = ptr_med != nullptr &&
						  ptr_med->sampleDistance(ray, its.t, sampler->next2D(), sampler, ray.org, throughput);
		if (inside_med) {
			handleMediumInteraction(scene, ptr_med, ray, sampler, throughput*power, max_interactions, thread_id);
			const PhaseFunction* ptr_phase = scene.phase_list[ptr_med->phase_id];
			Vector wo;
			Float phase_val = ptr_phase->sample(-ray.dir, sampler->next2D(), wo);
			if (phase_val == 0)
				break;
			throughput *= phase_val;
			ray.dir = wo;
			on_surface = false;
		} else {
			if (!its.isValid()) break;
			if (!its.ptr_bsdf->isNull())
				handleSurfaceInteraction(its, scene, ptr_med, sampler, power*throughput, max_interactions, thread_id);
			Float bsdf_pdf, bsdf_eta;
			Vector wo_local, wo;
			auto bsdf_weight = its.sampleBSDF(sampler->next3D(), wo_local, bsdf_pdf, bsdf_eta, true);
			if (bsdf_weight.isZero())
				break;
			wo = its.toWorld(wo_local);
	        /* Prevent light leaks due to the use of shading normals -- [Veach, p. 158] */
            Vector wi = -ray.dir;
            Float wiDotGeoN = wi.dot(its.geoFrame.n),
                  woDotGeoN = wo.dot(its.geoFrame.n);
            if (wiDotGeoN * its.wi.z() <= 0 ||
                woDotGeoN * wo_local.z() <= 0) {
				break;
            }
            throughput *= bsdf_weight;
            if (its.isMediumTransition())
                ptr_med = its.getTargetMedium(woDotGeoN);
            ray = Ray(its.p, wo);
            on_surface = true;
		}
        depth++;
	}

}

void ParticleTracer::handleEmission(const Intersection& its, const Scene& scene, RndSampler *sampler,
									const Spectrum& weight, int max_bounces, int thread_id) const {
	Vector2 pix_uv;
	Vector dir;
	const CropRectangle& rect = scene.camera.rect;
	Float transmittance = scene.sampleAttenuatedSensorDirect(its, sampler, max_bounces, pix_uv, dir);
	if (transmittance != 0.0f) {
		Spectrum value = weight * transmittance * its.ptr_emitter->evalDirection(its.shFrame.n, dir);
		int idx_pixel = rect.isValid() ? (int)pix_uv.x() + rect.crop_width * (int)pix_uv.y()
									 : (int)pix_uv.x() + scene.camera.width * (int)pix_uv.y();
		image_per_thread[thread_id][idx_pixel] += value;
	}
}

void ParticleTracer::handleSurfaceInteraction(const Intersection& its, const Scene& scene, const Medium* ptr_med, RndSampler *sampler,
    							  			  const Spectrum& weight, int max_bounces, int thread_id) const {
	Vector2 pix_uv;
	Vector dir;
	const CropRectangle& rect = scene.camera.rect;
	Float transmittance = scene.sampleAttenuatedSensorDirect(its, sampler, max_bounces, pix_uv, dir);
	if (transmittance != 0.0f) {
		Vector wi = its.toWorld(its.wi);
		Vector wo = dir, wo_local = its.toLocal(wo);
		/* Prevent light leaks due to the use of shading normals -- [Veach, p. 158] */
    	Float wiDotGeoN = wi.dot(its.geoFrame.n),
          	  woDotGeoN = wo.dot(its.geoFrame.n);
    	if (wiDotGeoN * its.wi.z() <= 0 ||
        	woDotGeoN * wo_local.z() <= 0)
        	return;

    	/* Adjoint BSDF for shading normals -- [Veach, p. 155] */
    	Float correction = std::abs((its.wi.z() * woDotGeoN)/(wo_local.z() * wiDotGeoN));
    	Spectrum value = transmittance * its.ptr_bsdf->eval(its, wo_local, true) * correction * weight;
		int idx_pixel = rect.isValid() ? (int)pix_uv.x() + rect.crop_width * (int)pix_uv.y()
									   : (int)pix_uv.x() + scene.camera.width * (int)pix_uv.y();
		image_per_thread[thread_id][idx_pixel] += value;     
	}
}

void ParticleTracer::handleMediumInteraction(const Scene& scene, const Medium* ptr_med, const Ray& ray, RndSampler *sampler,
											 const Spectrum& weight, int max_bounces, int thread_id) const {
	Vector2 pix_uv;
	Vector wi = -ray.dir, wo;
	const CropRectangle& rect = scene.camera.rect;
	Float transmittance = scene.sampleAttenuatedSensorDirect(ray.org, ptr_med, sampler, max_bounces, pix_uv, wo);
	if (transmittance != 0.0f) {
		const PhaseFunction* ptr_phase = scene.phase_list[ptr_med->phase_id];
		Spectrum value = transmittance * ptr_phase->eval(wi, wo) * weight;
		int idx_pixel = rect.isValid() ? (int)pix_uv.x() + rect.crop_width * (int)pix_uv.y()
									   : (int)pix_uv.x() + scene.camera.width * (int)pix_uv.y();
		image_per_thread[thread_id][idx_pixel] += value;
	}
}