#include "integratorAD.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"
#include <chrono>
#include <iomanip>


IntegratorAD::IntegratorAD() {
    omp_init_lock(&messageLock);
}


IntegratorAD::~IntegratorAD() {
    omp_destroy_lock(&messageLock);
}


void IntegratorAD::render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image) const {
    using namespace std::chrono;

    const auto &camera = scene.camera;
    const bool cropped = camera.rect.isValid();
    const int num_pixels = cropped ? camera.rect.crop_width * camera.rect.crop_height 
                                   : camera.width * camera.height;
    const int nworker = omp_get_num_procs();
    const int size_block = 4;

    std::cout << "Rendering using [ " << getName() << " ] ..." << std::endl;

    // Pixel sampling
    if ( options.num_samples > 0 )
    {
        int num_block = std::ceil((Float)num_pixels/size_block);
        auto _start = high_resolution_clock::now();

        int finished_block = 0;
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
        for (int index_block = 0; index_block < num_block; index_block++) {
            int block_start = index_block*size_block;
            int block_end = std::min((index_block+1)*size_block, num_pixels);

            for (int idx_pixel = block_start; idx_pixel < block_end; idx_pixel++) {
                int ix = cropped ? camera.rect.offset_x + idx_pixel % camera.rect.crop_width
                                 : idx_pixel % camera.width;
                int iy = cropped ? camera.rect.offset_y + idx_pixel / camera.rect.crop_width
                                 : idx_pixel / camera.width;
                RndSampler sampler(options.seed, idx_pixel);

                SpectrumAD pixel_val;
                for (int idx_sample = 0; idx_sample < options.num_samples; idx_sample++) {
                    const Array2 rnd = options.num_samples_primary_edge >= 0 ? Array2(sampler.next1D(), sampler.next1D()) : Array2(0.5f, 0.5f);
                    pixel_val += pixelColorAD(scene, options, &sampler,
                                              static_cast<Float>(ix + rnd.x()),
                                              static_cast<Float>(iy + rnd.y()));
                }
                pixel_val /= options.num_samples;

                if ( pixel_val.val.minCoeff() < 0.0f ) {
                    omp_set_lock(&messageLock);
                    std::cerr << std::fixed << std::setprecision(2)
                              << "\n[Warning] Negative pixel value: (" << pixel_val.val.transpose() << ")" << std::endl;
                    omp_unset_lock(&messageLock);
                }
                else {
                    rendered_image[idx_pixel*3    ] = static_cast<float>(pixel_val.val(0));
                    rendered_image[idx_pixel*3 + 1] = static_cast<float>(pixel_val.val(1));
                    rendered_image[idx_pixel*3 + 2] = static_cast<float>(pixel_val.val(2));
                }

                for ( int ch = 1; ch <= nder; ++ch ) {
                    int offset = (ch*num_pixels + idx_pixel)*3;
                    if ( !std::isfinite((pixel_val.grad(ch - 1))(0)) ||
                         !std::isfinite((pixel_val.grad(ch - 1))(1)) ||
                         !std::isfinite((pixel_val.grad(ch - 1))(2)) )
                    {
                        omp_set_lock(&messageLock);
                        std::cerr << std::fixed << std::setprecision(2)
                                  << "\n[Warning] Invalid derivative: (" << pixel_val.grad(ch - 1).transpose() << ")" << std::endl;
                        omp_unset_lock(&messageLock);
                    }
                    else {
                        rendered_image[offset    ] = static_cast<float>((pixel_val.grad(ch - 1))(0));
                        rendered_image[offset + 1] = static_cast<float>((pixel_val.grad(ch - 1))(1));
                        rendered_image[offset + 2] = static_cast<float>((pixel_val.grad(ch - 1))(2));
                    }
                }
            }

            if ( !options.quiet ) {
                omp_set_lock(&messageLock);
                progressIndicator(Float(++finished_block)/num_block);
                omp_unset_lock(&messageLock);
            }
        }
        if ( !options.quiet )
            std::cout << "\nDone in " << duration_cast<seconds>(high_resolution_clock::now() - _start).count() << " seconds." << std::endl;
    }

    // Primary edge sampling
    const EdgeManager* ptr_eManager = scene.ptr_edgeManager;
    if ( options.num_samples_primary_edge > 0 && ptr_eManager->getNumPrimaryEdges() > 0) {
        std::vector<RndSampler> samplers;
        for (int iworker = 0; iworker < nworker; iworker++)
            samplers.push_back(RndSampler(options.seed, iworker));

        constexpr Float imageDelta = 1e-4f;
        Eigen::Array<Float, -1, -1> edge_contrib = Eigen::Array<Float, -1, -1>::Zero(num_pixels*nder*3, nworker);

        const int num_samples_per_block = 128;
        const int num_block = static_cast<int>(std::ceil(ptr_eManager->getPrimaryEdgePDFsum() * 
                                                         options.num_samples_primary_edge/num_samples_per_block));
        const int num_samples = num_block*num_samples_per_block;

        auto _start = high_resolution_clock::now();
        const Medium* med_cam = camera.getMedID() == -1 ? nullptr : scene.medium_list[camera.getMedID()];

        int finished_block = 0;
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
        for (int index_block = 0; index_block < num_block; ++index_block) {
            for (int omp_i = 0; omp_i < num_samples_per_block; omp_i++) {
                const int tid = omp_get_thread_num();
                Vector2i xyPixel;
                Vector2AD p;
                Vector2 norm, p1;
                Float maxD = scene.ptr_edgeManager->samplePrimaryEdge(camera, samplers[tid].next1D(), xyPixel, p, norm);
                if ( !p.der.isZero(Epsilon) ) {
                    Ray ray = camera.samplePrimaryRay(xyPixel.x(), xyPixel.y(), p.val);
                    Float trans = scene.evalTransmittance(ray, false, med_cam, maxD, &samplers[tid], options.max_bounces - 1);
                    if (trans > Epsilon) {
                        Spectrum deltaVal;
                        p1 = xyPixel.cast<Float>() + p.val - norm*imageDelta;
                        deltaVal  = pixelColor(scene, options, &samplers[tid], p1.x(), p1.y());
                        p1 = xyPixel.cast<Float>() + p.val + norm*imageDelta;
                        deltaVal -= pixelColor(scene, options, &samplers[tid], p1.x(), p1.y());

                        int idx_pixel = cropped ? (xyPixel.x() - camera.rect.offset_x) + 
                                                  camera.rect.crop_width * (xyPixel.y() - camera.rect.offset_y)
                                                : xyPixel.x() + camera.width*xyPixel.y();
                        for ( int j = 0; j < nder; ++j ) {
                            int offset = (j*num_pixels + idx_pixel)*3;
                            edge_contrib.block<3, 1>(offset, tid) += norm.dot(p.grad(j))*deltaVal*ptr_eManager->getPrimaryEdgePDFsum();
                        }
                    }
                }
            }

            if ( !options.quiet ) {
                omp_set_lock(&messageLock);
                progressIndicator(Float(++finished_block)/num_block);
                omp_unset_lock(&messageLock);
            }
        }
        edge_contrib /= static_cast<Float>(num_samples);
        Eigen::ArrayXf output = edge_contrib.rowwise().sum().cast<float>();
        for ( int i = 0; i < num_pixels; ++i )
            for ( int j = 0; j < nder; ++j ) {
                int offset0 = ((j + 1)*num_pixels + i)*3, offset1 = (j*num_pixels + i)*3;
                rendered_image[offset0    ] += output[offset1    ];
                rendered_image[offset0 + 1] += output[offset1 + 1];
                rendered_image[offset0 + 2] += output[offset1 + 2];
            }
        if ( !options.quiet )
            std::cout << "\nDone in " << duration_cast<seconds>(high_resolution_clock::now() - _start).count() << " seconds." << std::endl;
    }
}
