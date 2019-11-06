#include "edge.h"
#include "shape.h"
#include "intersection.h"
#include "sampler.h"
#include "utils.h"
#include "stats.h"
#include "camera.h"
#include "edge_test0.h"
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <fstream>

Float rayIntersectTri(const Vector &v0, const Vector &v1, const Vector &v2, const Ray &ray) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = ray.dir.cross(e2);
    auto divisor = pvec.dot(e1);
    if (std::abs(divisor) < Epsilon)
        return std::numeric_limits<Float>::infinity();
    auto s = ray.org - v0;
    auto dot_s_pvec = s.dot(pvec);
    auto u = dot_s_pvec / divisor;
    if (u < -Epsilon || u > 1.0+Epsilon)
        return std::numeric_limits<Float>::infinity();
    auto qvec = s.cross(e1);
    auto dot_dir_qvec = ray.dir.dot(qvec);
    auto v = dot_dir_qvec / divisor;
    if (v < -Epsilon || u + v > 1.0+Epsilon)
        return std::numeric_limits<Float>::infinity();
    auto dot_e2_qvec = e2.dot(qvec);
    auto t = dot_e2_qvec / divisor;
    if ( t > Epsilon)
        return t;
    else
        return std::numeric_limits<Float>::infinity();
}

void projectPointToCamera(VectorAD v, const Camera& camera, Vector2AD& p) {
    v = (v - camera.cpos)/(v - camera.cpos).dot(camera.cframe.n);
    v = camera.cframe.toLocal(v);
    auto fov_factor = camera.cam_to_ndc(0,0);
    auto aspect_ratio = Float(camera.width) / Float(camera.height);
    p.x() = v.x() * fov_factor * 0.5f + 0.5;
    p.y() = v.y() * fov_factor * (-0.5f) * aspect_ratio + 0.5f;
}

Prob0_7::Prob0_7() {
    // Initialize Shape...
    constexpr int nvertices = 3;
    constexpr int nfaces = 1;
    float vtxPositions[nvertices*3] = {
        -1.0f, -1.0f, 5.0f,
         1.0f, -1.0f, 5.0f,
         0.0f,  1.0f, 5.0f,
    };

    int vtxIndices[nfaces*3] = {
        0, 1, 2,
    };
    shape = Shape(vtxPositions, vtxIndices, nullptr, nullptr, nvertices, nfaces, -1, -1, -1, -1);
    Eigen::Matrix<Float, 3, 3> vtxVelocities;
    vtxVelocities << 1.0f, 0.0f, 0.0f,
                     0.0f, 1.0f, 0.0f,
                     0.0f, 0.0f, 1.0f;
    shape.initVelocities(vtxVelocities, 0);
    a = shape.getVertexAD(0);
    b = shape.getVertexAD(1);
    c = shape.getVertexAD(2);

    // Initialize Camera
    Vector ang_vel = Vector(0,0,0);
    Vector pos_vel = Vector(0,0,0);
    VectorAD x, w;
    x.val = Vector::Zero();
    x.grad() = pos_vel;
    w.val = (a.val + 0.5*(b.val-a.val) + 0.5*(c.val-a.val)).normalized();
    w.grad() = ang_vel.cross(w.val);

    VectorAD right, up;
    coordinateSystemAD(w, right, up);

    fov = 120.0f;
    camera = Camera(x.val, up.val, w.val, fov, 1, 1, 0.0, -1);
    camera.initVelocities(pos_vel, ang_vel,0);

    camera.edge_distrb.clear();
    camera.initCameraEdgesFromShape(shape, false);
    if (camera.edge_distrb.size() > 0)
        camera.edge_distrb.normalize();
}

Float Prob0_7::eval(const Ray &ray) {
    return std::isfinite(rayIntersectTri(a.val, b.val, c.val, ray)) ? 1.0: 0.0;
}

void Prob0_7::computeReference() {
    const VectorAD& x = camera.cpos;
    const VectorAD& w = camera.cframe.n;
    const VectorAD& s = camera.cframe.s;
    const VectorAD& t = camera.cframe.t;
    VectorAD a1 = x + (a - x)/(a - x).dot(w),
             b1 = x + (b - x)/(b - x).dot(w),
             c1 = x + (c - x)/(c - x).dot(w);

    if ( std::abs((a1.val - (x.val + w.val)).dot(w.val)) > Epsilon ||
         std::abs((b1.val - (x.val + w.val)).dot(w.val)) > Epsilon ||
         std::abs((c1.val - (x.val + w.val)).dot(w.val)) > Epsilon )
    {
        std::cerr << "Badness 1: Behind camera" << std::endl;
        return;
    }

    Float aspect_ratio = (Float)camera.height/camera.width;
    Vector2 viewFrustrum;
    viewFrustrum.x() = tan(0.5*fov);
    viewFrustrum.y() = viewFrustrum.x() * aspect_ratio;

    if (std::abs(a1.val.x()) > viewFrustrum.x() || std::abs(a1.val.y()) > viewFrustrum.y() ||
        std::abs(b1.val.x()) > viewFrustrum.x() || std::abs(b1.val.y()) > viewFrustrum.y() ||
        std::abs(c1.val.x()) > viewFrustrum.x() || std::abs(c1.val.y()) > viewFrustrum.y() )
    {
        std::cerr << "Badness 2: Clipped by screen" << std::endl;
        return;
    }

    VectorAD a2(a1.dot(s), a1.dot(t), 0.0),
             b2(b1.dot(s), b1.dot(t), 0.0),
             c2(c1.dot(s), c1.dot(t), 0.0);
    Float pixel_area = square(2*tan(0.5*fov * M_PI/180.0)/camera.width);
    FloatAD val_ad = 0.5*((b2 - a2).cross(c2 - a2)).norm()/pixel_area;
    printf("[Ref.]  : %.2le\n", val_ad.grad());
}


void Prob0_7::computeEdgeIntegral() {
    long long N = 10000000LL;
    Float delta = 1e-5;
    int nworker = omp_get_num_procs();
    std::vector<Statistics> stats(nworker + 1);
    std::vector<RndSampler> samplers;

    int start = 0;
    int interval = 10000;
    std::ofstream file_out;
    file_out.open("edge_stats.txt");
    while (start < N) {
        stats[nworker].reset();
        for (int i = 0; i < nworker; i++)
            samplers.push_back(RndSampler(0, i));
#pragma omp parallel for num_threads(nworker)
        for (long long omp_i = 0; omp_i < interval; omp_i++) {
            // const int tid = 0;
            const int tid = omp_get_thread_num();
            Vector2i ipixel;
            Vector2AD y;
            Vector2 norm;
            camera.sampleEdge(samplers[tid].next1D(), ipixel, y, norm);
            Ray ray_p = camera.samplePrimaryRay(ipixel.x(), ipixel.y(), y.val + norm*delta);
            Ray ray_m = camera.samplePrimaryRay(ipixel.x(), ipixel.y(), y.val - norm*delta);
            Float deltaFunc = eval(ray_m) - eval(ray_p);
            Float val = norm.dot(y.grad())*deltaFunc*camera.edge_distrb.getSum();
            stats[tid].push(val);
        }
        for ( int i = 0; i < nworker; ++i ) stats[nworker].push(stats[i]);
        file_out << stats[nworker].getMean()-stats[nworker].getCI() << " "
                 << stats[nworker].getMean()+stats[nworker].getCI() << std::endl;
        start += interval;
    }
    printf("[Edge.] %.2le +- %.2le\n", stats[nworker].getMean(), stats[nworker].getCI());
    file_out.close();
}

void Prob0_7::computeFiniteDiff(Float delta, const std::string& fn) {
    Vector _a = a.val, _b = b.val, _c = c.val;
    Vector _a2 = _a + delta * a.grad(0),
           _b2 = _b + delta * b.grad(0),
           _c2 = _c + delta * c.grad(0);
    long long N = 10000000LL;
    int nworker = omp_get_num_procs();
    std::vector<Statistics> stats(nworker + 1);
    std::vector<RndSampler> samplers;

    int start = 0;
    int interval = 10000;
    std::ofstream file_out;
    file_out.open(fn);
    while (start < N) {
        stats[nworker].reset();
        for (int i = 0; i < nworker; i++)
            samplers.push_back(RndSampler(0, i));
#pragma omp parallel for num_threads(nworker)
        for (long long omp_i = 0; omp_i < interval; omp_i++) {
            // const int tid = 0;
            const int tid = omp_get_thread_num();
            Vector2i ipixel;
            Vector2AD y;
            Vector2 norm;
            Ray ray = camera.samplePrimaryRay(0, 0, samplers[tid].next2D());
            Float val_0 = std::isfinite(rayIntersectTri(_a, _b, _c, ray)) ? 1.0: 0.0;
            Float val_1 = std::isfinite(rayIntersectTri(_a2, _b2, _c2, ray)) ? 1.0: 0.0;
            stats[tid].push((val_1-val_0)/delta);
        }
        for ( int i = 0; i < nworker; ++i ) stats[nworker].push(stats[i]);
        file_out << stats[nworker].getMean()-stats[nworker].getCI() << " "
                 << stats[nworker].getMean()+stats[nworker].getCI() << std::endl;
        start += interval;
    }

    for ( int i = 1; i < nworker; ++i ) stats[0].push(stats[i]);
    printf("[FD delta = %.2le] %.2le +- %.2le\n", delta, stats[0].getMean(), stats[0].getCI());


}

void Prob0_7::run() {
    computeReference();
    computeEdgeIntegral();
    computeFiniteDiff(1e-1, "fd_stats0.txt");
    computeFiniteDiff(1e-2, "fd_stats1.txt");
}

void edge_test0() {
    Prob0_7 prob;
    prob.run();
}