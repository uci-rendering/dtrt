#include "../medium/gridvolume.h"
#include "../medium/heterogeneous.h"
#include "ray.h"
#include "rayAD.h"
#include "sampler.h"
#include "stats.h"
#include <random>
#include <ctime>
#include <iomanip>

static bool sameVector(const Vector &u, const Vector &v, Float absTor = 1e-4f, Float relTor = 1e-3f) {
    Float dist = (u - v).norm(), minLen = std::min(u.norm(), v.norm());
    return ( dist < absTor || dist/minLen < relTor );
}


static bool sameFloat(const Float &u, const Float &v, Float absTor = 1e-4f, Float relTor = 1e-3f) {
    Float dist = std::abs(u - v), minLen = std::min(std::abs(u), std::abs(v));
    return ( dist < absTor || dist/minLen < relTor );
}


void volume_test() {
    const int nworkers = omp_get_num_procs();

    std::srand(static_cast<unsigned>(time(NULL)));
    std::uniform_real_distribution<Float> distrb;
    std::mt19937_64 engine;

    // Basic operations
    {
        std::cout << "Performing basic tests ..." << std::flush;

        constexpr int nTests = 1000;
        GridVolume gridvol("simple.vol");

        for ( int testId = 0; testId < nTests; ++testId ) {
            Vector x0 = Vector::Random();
            float v = gridvol.lookupFloat(x0);
            assert( sameFloat(v, x0.sum() + 3.0f) );
        }

        std::cout << " done!" << std::endl;
    }

    // AD operations
    {
        std::cout << "Performing AD tests ..." << std::flush;

        Matrix4x4 volumeToWorld = Matrix4x4::Identity();
        volumeToWorld.block<3, 3>(0, 0) = Eigen::AngleAxis(distrb(engine)*M_PI*2.0f, Vector::Random().normalized()).matrix();
        volumeToWorld.block<3, 1>(0, 3) = Vector::Random()*10.0f;

        constexpr Float delta = 1e-7f;
        constexpr int nTests = 1000;
        GridVolume gridvol("frame_000250.vol", volumeToWorld);

        for ( int testId = 0; testId < nTests; ++testId ) {
            Eigen::Matrix<Float, 3, nder> transVelocities, rotVelocities;
            for ( int i = 0; i < nder; ++i ) {
                transVelocities.col(i).setRandom();
                rotVelocities.col(i) = Vector::Random().normalized();
            }
            gridvol.initVelocities(transVelocities, rotVelocities);

            for ( int i = 0; i < nder; ++i ) {
                GridVolume gridvol_2(gridvol);
                gridvol_2.advance(delta, i);

                // Transformation tests
                {
                    Vector x0 = Vector::Random();
                    Vector x1 = gridvol.volumeToWorld(x0);
                    Vector x2 = gridvol.worldToVolume(x1); assert( sameVector(x0, x2) );
                    Vector x3 = gridvol.worldToVolume(x0);

                    VectorAD x0AD(x0); x0AD.der.setRandom();
                    VectorAD x1AD = gridvol.volumeToWorldAD(x0AD); assert( sameVector(x1, x1AD.val) );
                    VectorAD x2AD = gridvol.worldToVolumeAD(x1AD); assert( sameVector(x2, x2AD.val) );
                    VectorAD x3AD = gridvol.worldToVolumeAD(x0AD); assert( sameVector(x3, x3AD.val) );

                    Vector x0_2 = x0AD.advance(delta, i);
                    Vector x1_2 = gridvol_2.volumeToWorld(x0_2), x1FD = (x1_2 - x1)/delta;
                    assert( sameVector(x1AD.grad(i), x1FD, 1e-3f, 1e-2f) );

                    Vector x3_2 = gridvol_2.worldToVolume(x0_2), x3FD = (x3_2 - x3)/delta;
                    assert( sameVector(x3AD.grad(i), x3FD, 1e-3f, 1e-2f) );
                }

                // Float lookup tests
                {
                    Vector x0;
                    Float fx, fy, fz;
                    do {
                        x0 = gridvol.m_min + ((Vector::Random() + Vector::Ones())*0.5f).cwiseProduct(gridvol.m_extent);
                        Vector tmp = gridvol.volumeToGrid(x0);
                        fx = tmp.x() - std::floor(tmp.x());
                        fy = tmp.y() - std::floor(tmp.y());
                        fz = tmp.z() - std::floor(tmp.z());
                    }
                    while ( fx < ShadowEpsilon || fx > 1.0f - ShadowEpsilon ||
                            fy < ShadowEpsilon || fy > 1.0f - ShadowEpsilon ||
                            fz < ShadowEpsilon || fz > 1.0f - ShadowEpsilon );
                    x0 = gridvol.volumeToWorld(x0);

                    VectorAD x0AD(x0); x0AD.der.setRandom();
                    FloatAD vAD = gridvol.lookupFloatAD(x0AD);
                    Float v = gridvol.lookupFloat(x0), v_2 = gridvol_2.lookupFloat(x0AD.advance(delta, i));
                    Float vFD = (v_2 - v)/delta;

                    assert( sameFloat(vAD.val, v) );
                    if ( !sameFloat(vAD.grad(i), vFD, 1e-3f, 1e-2f) ) {
                        std::cout << "\nMismatch: " << vAD.grad(i) << " <=> " << vFD << std::endl;
                        assert(false);
                    }
                }
            }
        }

        std::cout << " done!" << std::endl;
    }

    // Medium operations
    {
        std::cout << "Performing medium tests with " << nworkers << " threads ..." << std::endl;

        constexpr long long N = 50000000LL;
        Heterogeneous medium("simple.vol", Matrix4x4::Identity(), 2.0f, Spectrum::Ones(), -1);
        Ray ray(Vector(0.0f, 0.0f, -1.0f), Vector(0.0f, 0.0f, 1.0f));

        std::vector<RndSampler> samplers;
        for ( int i = 0; i < nworkers; ++i ) samplers.push_back(RndSampler(123, i));
        std::vector<Statistics> stats(nworkers);

        // Transmittance, residual ratio tracking

        medium.m_residualRatioTracking = true;
        for ( int i = 0; i < nworkers; ++i ) stats[i].reset();
#pragma omp parallel for num_threads(nworkers)
        for ( long long i = 0; i < N; ++i ) {
            const int tid = omp_get_thread_num();
            stats[tid].push(medium.evalTransmittance(ray, 0.0f, 2.0f, &samplers[tid]));
        }
        for ( int i = 1; i < nworkers; ++i )
            stats[0].push(stats[i]);

        std::cout << std::setiosflags(std::ios::scientific) << std::setprecision(2)
                  << "  RT: " << stats[0].getMean() << " +- " << stats[0].getCI() << std::endl;

        // Transmittance, delta tracking

        medium.m_residualRatioTracking = false;
        for ( int i = 0; i < nworkers; ++i ) stats[i].reset();
#pragma omp parallel for num_threads(nworkers)
        for ( long long i = 0; i < N; ++i ) {
            const int tid = omp_get_thread_num();
            stats[tid].push(medium.evalTransmittance(ray, 0.0f, 2.0f, &samplers[tid]));
        }
        for ( int i = 1; i < nworkers; ++i )
            stats[0].push(stats[i]);

        std::cout << std::setiosflags(std::ios::scientific) << std::setprecision(2)
                  << "  DT: " << stats[0].getMean() << " +- " << stats[0].getCI() << std::endl;

        // TransmittanceAD

        RayAD rayAD(ray);
        rayAD.org.grad() = Vector(0.0f, 0.0f, 1.0f);
        FloatAD tmax(2.0f, -1.0f);

        medium.m_residualRatioTracking = true;
        for ( int i = 0; i < nworkers; ++i ) stats[i].reset();
#pragma omp parallel for num_threads(nworkers)
        for ( long long i = 0; i < N; ++i ) {
            const int tid = omp_get_thread_num();
            stats[tid].push(medium.evalTransmittanceAD(rayAD, 0.0f, tmax, &samplers[tid]).grad());
        }
        for ( int i = 1; i < nworkers; ++i )
            stats[0].push(stats[i]);

        std::cout << std::setiosflags(std::ios::scientific) << std::setprecision(2)
                  << "  Grad: " << stats[0].getMean() << " +- " << stats[0].getCI() << std::endl;

        std::cout << "done!" << std::endl;
    }
}
