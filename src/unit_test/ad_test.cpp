#include "ad_test.h"
#include "scene.h"
#include "shape.h"
#include "frame.h"
#include "frameAD.h"
#include "intersection.h"
#include "intersectionAD.h"
#include "ray.h"
#include "rayAD.h"
#include "../emitter/area.h"
#include "../bsdf/null.h"
#include "../bsdf/diffuse.h"
#include "../bsdf/phong.h"
#include "../bsdf/roughdielectric.h"
#include "phase.h"
#include "../medium/homogeneous.h"
#include "camera.h"
#include "sampler.h"
#include "stats.h"
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <array>

static bool sameVector(const Vector &u, const Vector &v, Float absTor = 1e-4f, Float relTor = 1e-3f) {
    Float dist = (u - v).norm(), minLen = std::min(u.norm(), v.norm());
    return ( dist < absTor || dist/minLen < relTor );
}

static bool sameVector2(const Vector2 &u, const Vector2 &v, Float absTor = 1e-4f, Float relTor = 1e-3f) {
    Float dist = (u - v).norm(), minLen = std::min(u.norm(), v.norm());
    return ( dist < absTor || dist/minLen < relTor );
}

static bool sameFloat(const Float &u, const Float &v, Float absTor = 1e-4f, Float relTor = 1e-3f) {
    Float dist = std::abs(u - v), minLen = std::min(std::abs(u), std::abs(v));
    return ( dist < absTor || dist/minLen < relTor );
}

static void testFrameAD();
static void testRayIntersectionAD();
static void testShape();
static void testBSDF();
static void testMediumAD();
static void testPhaseAD();
static void testSceneAD1();
static void testSceneAD2(); // Prob0-3
static void testSceneAD3(); // Prob0-6


void ad_test()
{
    testFrameAD();
    testRayIntersectionAD();
    testShape();
    testBSDF();
    testMediumAD();
    testPhaseAD();
    testSceneAD1();
    testSceneAD2(); // Prob0-3
    testSceneAD3(); // Prob0-6
}


static void testFrameAD()
{
    std::cout << "Testing frame AD ... " << std::flush;

    constexpr int ntests = 1000;
    constexpr Float delta = 1e-7;

    std::array<Vector, ntests> n;

    for ( int i = 0; i < ntests; ++i )
        n[i] = Vector::Random().normalized();

    for ( int i = 0; i < ntests; ++i ) {
        Vector dotn = Vector::Random()*10.0f;
        dotn -= dotn.dot(n[i])*n[i];
        assert(std::abs(dotn.dot(n[i])) < Epsilon);

        Frame frame0(n[i]), frame1(n[i] + delta*dotn);
        VectorAD nAD(n[i]);
        nAD.grad() = dotn;
        FrameAD frameAD(nAD);

        // Test coordinate system
        {
            Vector dots = (frame1.s - frame0.s).transpose()/delta, dott = (frame1.t - frame0.t).transpose()/delta;
            assert( sameVector(dots, frameAD.s.grad()) &&
                    sameVector(dott, frameAD.t.grad()) &&
                    sameVector(dotn, frameAD.n.grad()) );
        }

        Vector v = Vector::Random()*10.0f, dotv = Vector::Random()*10.0f;
        VectorAD vAD(v);
        vAD.grad() = dotv;

        // Test toLocal
        {
            Vector dotLocalV = (frame1.toLocal(v + delta*dotv) - frame0.toLocal(v))/delta;
            assert( sameVector(dotLocalV, frameAD.toLocal(vAD).grad()) );
        }

        // Test toWorld
        {
            Vector dotWorldV = (frame1.toWorld(v + delta*dotv) - frame0.toWorld(v))/delta;
            assert( sameVector(dotWorldV, frameAD.toWorld(vAD).grad()) );
        }

        // Test cosTheta2
        {
            Float dotCosTheta2 = (frame1.cosTheta2(v + delta*dotv) - frame0.cosTheta2(v))/delta;
            assert( sameFloat(dotCosTheta2, frameAD.cosTheta2(vAD).grad()) );
        }

        // Test cosTheta
        {
            Float dotCosTheta = (frame1.cosTheta(v + delta*dotv) - frame0.cosTheta(v))/delta;
            assert( sameFloat(dotCosTheta, frameAD.cosTheta(vAD).grad()) );
        }

        // Test sinTheta2
        {
            Float dotSinTheta2 = (frame1.sinTheta2(v + delta*dotv) - frame0.sinTheta2(v))/delta;
            assert( sameFloat(dotSinTheta2, frameAD.sinTheta2(vAD).grad()) );
        }

        // Test sinTheta
        {
            Float dotSinTheta = (frame1.sinTheta(v + delta*dotv) - frame0.sinTheta(v))/delta;
            assert( sameFloat(dotSinTheta, frameAD.sinTheta(vAD).grad()) );
        }

        // Test tanTheta
        {
            Float dotTanTheta = (frame1.tanTheta(v + delta*dotv) - frame0.tanTheta(v))/delta;
            assert( sameFloat(dotTanTheta, frameAD.tanTheta(vAD).grad()) );
        }
    }

    std::cout << "done." << std::endl;
}


static void testRayIntersectionAD()
{
    std::cout << "Testing ray-intersect AD ... " << std::flush;

    constexpr int ntests = 1000;
    constexpr Float delta = 1e-7;

    for ( int i = 0; i < ntests; ++i ) {
        Vector p0 = Vector::Random()*10.0f, p1 = Vector::Random()*10.0f, p2 = Vector::Random()*10.0f;
        Vector dp0 = Vector::Random()*10.0f, dp1 = Vector::Random()*10.0f, dp2 = Vector::Random()*10.0f;

        Vector w = Vector::Random().normalized(), dw = Vector::Random()*10.0f;
        dw -= dw.dot(w)*w;

        Vector st = (Vector::Random() + Vector::Ones())*0.5f;
        st(1) *= 1.0f - st(0);
        st(2) *= 10.0f;

        Vector x = p0 + (p1 - p0)*st(0) + (p2 - p0)*st(1) - w*st(2), dx = Vector::Random()*10.0f;

        // Finite difference
        Array uvt0 = rayIntersectTriangle(p0, p1, p2, Ray(x, w)),
              uvt1 = rayIntersectTriangle(p0 + delta*dp0, p1 + delta*dp1, p2 + delta*dp2, Ray(x + delta*dx, w + delta*dw));
        assert( sameVector(uvt0.matrix(), st, Epsilon, Epsilon) );
        Array duvt = (uvt1 - uvt0)/delta;

        VectorAD p0AD(p0), p1AD(p1), p2AD(p2);
        p0AD.grad() = dp0; p1AD.grad() = dp1; p2AD.grad() = dp2;

        VectorAD xAD(x), wAD(w);
        xAD.grad() = dx; wAD.grad() = dw;

        ArrayAD uvtAD = rayIntersectTriangleAD(p0AD, p1AD, p2AD, RayAD(xAD, wAD));
        assert( sameVector(duvt.matrix(), uvtAD.grad().transpose().matrix(), 1e-3f, 1e-2f) );
    }

    std::cout << "done." << std::endl;
}


static void testShape()
{
    std::cout << "Testing shape AD ... " << std::flush;

    constexpr int nvertices = 8, nfaces = 12;
    constexpr int ntests = 1000;
    constexpr Float delta = 1e-7;

    std::random_device rd;
    std::uniform_real_distribution<Float> distr_real(0.01f, 10.0f);
    std::uniform_int_distribution<int> distr_int(0, nfaces - 1);
    std::mt19937_64 engine(rd());

    float vtxPositions[nvertices*3] = {
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f
    };

    int vtxIndices[nfaces*3] = {
        0, 2, 1,
        0, 3, 2,
        0, 1, 5,
        0, 5, 4,
        1, 2, 6,
        1, 6, 5,
        4, 5, 6,
        4, 6, 7,
        3, 6, 2,
        3, 7, 6,
        0, 7, 3,
        0, 4, 7
    };

    Shape shape(vtxPositions, vtxIndices, nullptr, vtxPositions, nvertices, nfaces, -1, -1, -1, -1);

    for ( int testId = 0; testId < ntests; ++testId ) {
        Eigen::Matrix<Float, -1, -1> vtxVelocities = Eigen::Matrix<Float, -1, -1>::Random(3*nder, nvertices)*10.0f,
                                     normalVelocities = Eigen::Matrix<Float, -1, -1>::Random(3*nder, nvertices);
        shape.initVelocities(vtxVelocities, normalVelocities);

        for ( int i = 0; i < nvertices; ++i )
            for ( int j = 0; j < nder; ++j ) {
                assert( sameVector(vtxVelocities.block<3, 1>(3*j, i), shape.vertices[i].grad(j), Epsilon, Epsilon) );
                assert( sameVector(normalVelocities.block<3, 1>(3*j, i), shape.normals[i].grad(j), Epsilon, Epsilon) );
            }

        Shape shape1(shape);
        shape1.advance(delta);

        const int idx = distr_int(engine);
        const auto indices = shape.getIndices(idx);

        Vector2 st = (Vector2::Random() + Vector2::Ones())*0.5f;
        st(1) *= 1.0f - st(0);

        const auto p0 = shape.getVertex(indices(0)), p1 = shape.getVertex(indices(1)), p2 = shape.getVertex(indices(2));
        Vector p = (p0 + (p1 - p0)*st(0) + (p2 - p0)*st(1));
        Vector w = -p.normalized(), dw = Vector::Random()*10.0f;
        dw -= dw.dot(w)*w;

        Vector x = p*(1.1f + distr_real(engine)), dx = Vector::Random()*10.0f;
        VectorAD xAD(x), wAD(w);
        xAD.grad() = dx; wAD.grad() = dw;

        Intersection its0, its1;
        IntersectionAD itsAD;

        shape.rayIntersect(idx, Ray(x, w), its0);
        shape1.rayIntersect(idx, Ray(x + delta*dx, w + delta*dw), its1);
        shape.rayIntersectAD(idx, RayAD(xAD, wAD), itsAD);

        // Check t
        {
            Float tFD = (its1.t - its0.t)/delta;
            assert( sameFloat(its0.t, itsAD.t.val) );
            assert( sameFloat(tFD, itsAD.t.grad(), 1e-3f, 1e-2f) );
        }

        // Check p
        {
            Vector pFD = (its1.p - its0.p)/delta;
            assert( sameVector(its0.p, itsAD.p.val) );
            assert( sameVector(pFD, itsAD.p.grad(), 1e-3f, 1e-2f) );
        }

        // Check geoFrame
        {
            assert( sameVector(its0.geoFrame.s, itsAD.geoFrame.s.val) &&
                    sameVector(its0.geoFrame.t, itsAD.geoFrame.t.val) &&
                    sameVector(its0.geoFrame.n, itsAD.geoFrame.n.val) );

            Vector tFD = (its1.geoFrame.t - its0.geoFrame.t)/delta;
            assert( sameVector(tFD, itsAD.geoFrame.t.grad(), 1e-3f, 1e-2f) );

            Vector sFD = (its1.geoFrame.s - its0.geoFrame.s)/delta;
            assert( sameVector(sFD, itsAD.geoFrame.s.grad(), 1e-3f, 1e-2f) );

            Vector nFD = (its1.geoFrame.n - its0.geoFrame.n)/delta;
            assert( sameVector(nFD, itsAD.geoFrame.n.grad(), 1e-3f, 1e-2f) );
        }

        // Check shFrame
        {
            assert( sameVector(its0.shFrame.s, itsAD.shFrame.s.val) &&
                    sameVector(its0.shFrame.t, itsAD.shFrame.t.val) &&
                    sameVector(its0.shFrame.n, itsAD.shFrame.n.val) );

            Vector tFD = (its1.shFrame.t - its0.shFrame.t)/delta;
            assert( sameVector(tFD, itsAD.shFrame.t.grad(), 1e-3f, 1e-2f) );

            Vector sFD = (its1.shFrame.s - its0.shFrame.s)/delta;
            assert( sameVector(sFD, itsAD.shFrame.s.grad(), 1e-3f, 1e-2f) );

            Vector nFD = (its1.shFrame.n - its0.shFrame.n)/delta;
            assert( sameVector(nFD, itsAD.shFrame.n.grad(), 1e-3f, 1e-2f) );
        }

        // Check uv
        {
            assert( sameVector2(its0.uv, itsAD.uv.val, Epsilon, Epsilon) );

            Vector2 uvFD = (its1.uv - its0.uv)/delta;
            assert( sameVector2(uvFD, itsAD.uv.grad(), 1e-3f, 1e-2f) );
        }

        // Check wi
        {
            assert( sameVector(its0.wi, itsAD.wi.val, Epsilon, Epsilon) );

            Vector wiFD = (its1.wi - its0.wi)/delta;
            assert( sameVector(wiFD, itsAD.wi.grad(), 1e-3f, 1e-2f) );
        }
    }

    std::cout << "done." << std::endl;
}


static void testBSDF()
{
    std::cout << "Testing BSDF AD ... " << std::flush;

    constexpr int ntests = 1000;
    constexpr Float delta = 1e-7;

    DiffuseBSDF diffuse(Spectrum3f(0.4f, 0.5f, 0.6f));
    PhongBSDF phong(Spectrum3f(0.3f, 0.4f, 0.5f), Spectrum3f(0.3f, 0.2f, 0.1f), 50.0f);
    RoughDielectricBSDF roughDielectric(0.05f, 1.5f, 1.0f);

    std::vector<const BSDF *> bsdfLst;
    bsdfLst.push_back(&diffuse);
    bsdfLst.push_back(&phong);
    bsdfLst.push_back(&roughDielectric);

    std::random_device rd;
    std::uniform_real_distribution<Float> distr;
    std::mt19937_64 engine(rd());

    for ( auto bsdf : bsdfLst ) {
        for ( int testId = 0; testId < ntests; ++testId ) {
            Vector wi, wo;

            if ( bsdf->isTransmissive() ) {
                wi = squareToUniformSphere(Vector2(distr(engine), distr(engine)));
                wo = squareToUniformSphere(Vector2(distr(engine), distr(engine)));
            }
            else {
                wi = squareToCosineHemisphere(Vector2(distr(engine), distr(engine)));
                wo = squareToCosineHemisphere(Vector2(distr(engine), distr(engine)));
            }

            Vector dwi = Vector::Random()*10.0f, dwo = Vector::Random()*10.0f;
            dwi -= dwi.dot(wi)*wi; dwo -= dwo.dot(wo)*wo;

            Intersection its, its1;
            its.wi = wi; its1.wi = wi + delta*dwi;
            Vector wo1 = wo + delta*dwo;

            IntersectionAD itsAD;
            itsAD.wi.val = wi;
            itsAD.wi.grad() = dwi;
            VectorAD woAD(wo);
            woAD.grad() = dwo;

            Spectrum bsdfVal = bsdf->eval(its, wo), bsdfVal1 = bsdf->eval(its1, wo1);
            Spectrum bsdfValFD = (bsdfVal1 - bsdfVal)/delta;
            SpectrumAD bsdfValAD = bsdf->evalAD(itsAD, woAD);

            assert( sameVector(bsdfVal.matrix(), bsdfValAD.val.transpose().matrix(), Epsilon, Epsilon) );
            assert( sameVector(bsdfValFD.matrix(), bsdfValAD.grad().transpose().matrix(), 1e-3f, 1e-2f) );
        }
    }

    std::cout << "done." << std::endl;
}


static void testMediumAD()
{
    std::cout << "Testing medium AD ... " << std::flush;

    constexpr int ntests = 1000;
    constexpr Float delta = 1e-7;

    Homogeneous medium(2.0, Spectrum3f(0.9, 0.9, 0.9), -1);
    for ( int testId = 0; testId < ntests; ++testId ) {
        Vector p, dp;
        p = Eigen::Vector3d::Random()*10.0;
        dp = Eigen::Vector3d::Random()*10.0;

        Vector w, dw;
        w = Eigen::Vector3d::Random().normalized();
        dw = Eigen::Vector3d::Random()*10.0;
        dw -= dw.dot(w)*w;
        assert( std::abs(w.dot(dw)) < Epsilon );

        Array2 t = 0.5*(Array2::Random() + Array2::Ones());
        t[1] += t[0];
        Array2 dt = Array2::Random()*10.0;

        VectorAD org(p), dir(w);
        org.grad() = dp; dir.grad() = dw;
        FloatAD tmin(t[0], dt[0]), tmax(t[1], dt[1]);

        Ray ray0(org.val, dir.val),
            ray1(org.val + delta*org.grad(), (dir.val + delta*dir.grad()).normalized());
        Float trans0 = medium.evalTransmittance(ray0, tmin.val, tmax.val, nullptr),
              trans1 = medium.evalTransmittance(ray1, tmin.advance(delta), tmax.advance(delta), nullptr);
        Float transFD = (trans1 - trans0)/delta;

        RayAD rayAD(org, dir);
        FloatAD transAD = medium.evalTransmittanceAD(rayAD, tmin, tmax, nullptr);

        assert( sameFloat(trans0, transAD.val, Epsilon, Epsilon) );
        assert( sameFloat(transFD, transAD.grad(), 1e-3f, 1e-2f) );
    }

    std::cout << "done." << std::endl;
}


static void testPhaseAD()
{
    std::cout << "Testing phase function AD ... " << std::flush;

    constexpr int ntests = 1000;
    constexpr Float delta = 1e-7;

    HGPhaseFunction phase(0.8f);
    for ( int testId = 0; testId < ntests; ++testId ) {
        Vector wi, dwi;
        wi = Eigen::Vector3d::Random().normalized();
        dwi = Eigen::Vector3d::Random()*10.0;
        dwi -= dwi.dot(wi)*wi;
        assert( std::abs(wi.dot(dwi)) < Epsilon );

        Vector wo, dwo;
        wo = Eigen::Vector3d::Random().normalized();
        dwo = Eigen::Vector3d::Random()*10.0;
        dwo -= dwo.dot(wo)*wo;
        assert( std::abs(wo.dot(dwo)) < Epsilon );

        VectorAD wiAD(wi), woAD(wo);
        wiAD.grad() = dwi; woAD.grad() = dwo;

        Float hg0 = phase.eval(wi, wo),
              hg1 = phase.eval((wi + delta*dwi).normalized(), (wo + delta*dwo).normalized());
        Float hgFD = (hg1 - hg0)/delta;

        FloatAD hgAD = phase.evalAD(wiAD, woAD);

        assert( sameFloat(hg0, hgAD.val, Epsilon, Epsilon) );
        assert( sameFloat(hgFD, hgAD.grad(), 1e-3f, 1e-2f) );
    }

    std::cout << "done." << std::endl;
}


static void testSceneAD1()
{
    std::cout << "Testing scene AD #1 ..." << std::endl;

    constexpr int ntests = 1000;
    constexpr Float delta = 1e-7;

    AreaLight area(0, Spectrum3f(10.0f, 10.0f, 10.0f));
    NullBSDF nullBSDF;
    DiffuseBSDF diffuseBSDF(Spectrum3f(0.5f, 0.5f, 0.5f));
    Homogeneous medium(2.0, Spectrum3f(0.9, 0.9, 0.9), 0);
    IsotropicPhaseFunction phase;
    Camera camera;
    RndSampler sampler(123, 0);

    float cubeVtxPositions[] = {
        -0.5f, -0.5f, 0.5f,
         0.5f, -0.5f, 0.5f,
         0.5f,  0.5f, 0.5f,
        -0.5f,  0.5f, 0.5f,
        -0.5f, -0.5f, 1.5f,
         0.5f, -0.5f, 1.5f,
         0.5f,  0.5f, 1.5f,
        -0.5f,  0.5f, 1.5f
    };

    int cubeVtxIndices[] = {
        0, 2, 1,
        0, 3, 2,
        0, 1, 5,
        0, 5, 4,
        1, 2, 6,
        1, 6, 5,
        4, 5, 6,
        4, 6, 7,
        3, 6, 2,
        3, 7, 6,
        0, 7, 3,
        0, 4, 7
    };

    float triVtxPositions[] = {
        -1.0f, -1.0f, 2.0f,
         1.0f, -1.0f, 2.0f,
         1.0f,  1.0f, 2.0f,
    };

    int triVtxIndices[] = {
        0, 2, 1
    };

    Shape cube(cubeVtxPositions, cubeVtxIndices, nullptr, nullptr, 8, 12, -1, 0, 0, -1),
          triangle(triVtxPositions, triVtxIndices, nullptr, nullptr, 3, 1, 0, 1, -1, -1);

    Scene scene0(camera,
                 std::vector<const Shape*>{&cube, &triangle},
                 std::vector<const BSDF*>{&nullBSDF, &diffuseBSDF},
                 std::vector<const Emitter*>{&area},
                 std::vector<const PhaseFunction*>{&phase},
                 std::vector<const Medium*>{&medium});

    Vector x(0.0f, 0.0f, -1.0f);

    Ray tmpRay;
    tmpRay = Ray(x, Vector(0.0, 0.0, 1.0));
    std::cout << "  evalTransmittance = " << scene0.evalTransmittance(tmpRay, 0.0, nullptr, 3.0f, &sampler, 100) << std::endl;
    tmpRay = Ray(x, Vector(0.0, 0.0, 1.0));
    std::cout << "  evalTransmittance = " << scene0.evalTransmittance(tmpRay, 0.0, nullptr, 3.01f, &sampler, 100) << std::endl;

    for ( int testId = 0; testId < ntests; ++testId ) {
        Eigen::Matrix<Float, -1, -1> cubeVtxVelocities = Eigen::Matrix<Float, -1, -1>::Random(3*nder, 8)*10.0f;
        cube.initVelocities(cubeVtxVelocities);

        Eigen::Matrix<Float, -1, -1> triVtxVelocities = Eigen::Matrix<Float, -1, -1>::Random(3*nder, 3)*10.0f;
        triangle.initVelocities(triVtxVelocities);

        Shape cube1(cube), triangle1(triangle);
        cube1.advance(delta); triangle1.advance(delta);

        Camera camera1;
        Scene scene1(camera1,
                     std::vector<const Shape*>{&cube1, &triangle1},
                     std::vector<const BSDF*>{&nullBSDF, &diffuseBSDF},
                     std::vector<const Emitter*>{&area},
                     std::vector<const PhaseFunction*>{&phase},
                     std::vector<const Medium*>{&medium});

        Vector2 st = (Eigen::Vector2d::Random() + Eigen::Vector2d::Ones())*0.5f;
        st(1) *= 1.0f - st(0);

        Vector p = triangle.getVertex(0) +
                   (triangle.getVertex(1) - triangle.getVertex(0))*st(0) +
                   (triangle.getVertex(2) - triangle.getVertex(0))*st(1);

        Vector p1 = triangle1.getVertex(0) +
                    (triangle1.getVertex(1) - triangle1.getVertex(0))*st(0) +
                    (triangle1.getVertex(2) - triangle1.getVertex(0))*st(1);

        VectorAD pAD = triangle.getVertexAD(0) +
                       (triangle.getVertexAD(1) - triangle.getVertexAD(0))*st(0) +
                       (triangle.getVertexAD(2) - triangle.getVertexAD(0))*st(1);

        VectorAD xAD(x);
        xAD.der.setRandom(); xAD.der *= 10.0f;

        assert( sameVector(p, pAD.val, Epsilon, Epsilon) );
        assert( sameVector((p1 - p)/delta, pAD.grad()) );

        // Test rayIntersectAndLookForEmitterAD
        {
            VectorAD wAD = (pAD - xAD).normalized();
            Ray ray0(xAD.val, wAD.val), ray1(xAD.advance(delta), wAD.advance(delta));

            Intersection its0, its1;
            Spectrum spec0, spec1;
            Float pdf0, pdf1;

            spec0 = scene0.rayIntersectAndLookForEmitter(ray0, false, &sampler, NULL, 100, its0, pdf0);
            spec1 = scene1.rayIntersectAndLookForEmitter(ray1, false, &sampler, NULL, 100, its1, pdf1);
            Spectrum specFD = (spec1 - spec0)/delta;

            RayAD rayAD(xAD, wAD);
            IntersectionAD itsAD;
            Float pdfAD;
            SpectrumAD specAD = scene0.rayIntersectAndLookForEmitterAD(rayAD, false, &sampler, NULL, 100, itsAD, pdfAD);

            assert( sameVector(spec0.matrix(), specAD.val.matrix(), Epsilon, Epsilon) );
            assert( sameFloat(its0.t, itsAD.t.val, Epsilon, Epsilon) );
            assert( sameFloat(pdf0, pdfAD, Epsilon, Epsilon) );
            assert( sameVector(specFD.matrix(), specAD.grad().matrix(), 1e-3f, 1e-2f) );
        }

        // Test evalTransmittanceAD
        {
            Vector y = Vector::Random()*0.45f + Vector(0.0f, 0.0f, 1.0f);
            VectorAD yAD(y);
            yAD.der.setRandom(); yAD.der *= 10.0f;

            Float dist0 = (yAD.val - xAD.val).norm(),
                  dist1 = (yAD.advance(delta) - xAD.advance(delta)).norm();

            FloatAD distAD = (yAD - xAD).norm();
            VectorAD wAD = (yAD - xAD)/distAD;
            Ray ray0(xAD.val, wAD.val), ray1(xAD.advance(delta), wAD.advance(delta));

            Float trans0 = scene0.evalTransmittance(ray0, false, nullptr, dist0, &sampler, 100),
                  trans1 = scene1.evalTransmittance(ray1, false, nullptr, dist1, &sampler, 100);
            Float transFD = (trans1 - trans0)/delta;

            FloatAD transAD = scene0.evalTransmittanceAD(RayAD(xAD, wAD), false, nullptr, distAD, &sampler, 100);

            assert( sameFloat(trans0, transAD.val, Epsilon, Epsilon) );
            assert( sameFloat(transFD, transAD.grad(), 1e-3f, 1e-2f) );
        }
    }

    std::cout << "done." << std::endl;
}


/*
 * Same as Prob0-3
 */
static void testSceneAD2()
{
    int nworker = omp_get_num_procs();
    std::vector<RndSampler*> samplers(nworker, nullptr);
    for ( int i = 0; i < nworker; ++i ) samplers[i] = new RndSampler(123, i);
    std::vector<Statistics> stat0(nworker), stat1(nworker);

    std::cout << "Testing scene AD #2 with " << nworker << " threads ..." << std::endl;
    constexpr Float delta = 1e-4f;
    constexpr Float angleDelta = 1e-4f;

    AreaLight area(0, Spectrum3f(1.0f, 1.0f, 1.0f));
    NullBSDF nullBSDF;
    Homogeneous medium(1.0f, Spectrum3f(0.0f, 0.0f, 0.0f), 0);
    IsotropicPhaseFunction phase;

    float triVtxPositions[] = {
        -0.5f, -0.5f, 1.0f,
         0.5f, -0.5f, 1.0f,
         0.0f,  0.5f, 1.0f
    };

    int triVtxIndices[] = {
        0, 2, 1
    };

    Shape triangle0(triVtxPositions, triVtxIndices, nullptr, nullptr, 3, 1, 0, 0, 0, 0);
    Eigen::Matrix<Float, -1, -1> triVtxVelocities = Eigen::Matrix<Float, -1, -1>::Zero(3*nder, 3);
    triVtxVelocities.block<3, 1>(0, 0) << 0.0f, 0.0f, 1.0f;
    triVtxVelocities.block<3, 1>(0, 1) << 0.0f, 0.0f, 1.0f;
    triVtxVelocities.block<3, 1>(0, 2) << 0.0f, 0.0f, 1.0f;
    triangle0.initVelocities(triVtxVelocities);

    Shape triangle1(triangle0);
    triangle1.advance(delta);

    Camera camera0;
    Scene scene0(camera0,
                 std::vector<const Shape*>{&triangle0},
                 std::vector<const BSDF*>{&nullBSDF},
                 std::vector<const Emitter*>{&area},
                 std::vector<const PhaseFunction*>{&phase},
                 std::vector<const Medium*>{&medium});

    Camera camera1;
    Scene scene1(camera1,
                 std::vector<const Shape*>{&triangle1},
                 std::vector<const BSDF*>{&nullBSDF},
                 std::vector<const Emitter*>{&area},
                 std::vector<const PhaseFunction*>{&phase},
                 std::vector<const Medium*>{&medium});

    Vector x(0.2f, 0.3f, 0.1f);

    for ( int i = 0; i < nworker; ++i ) {
        stat0[i].reset(); stat1[i].reset();
    }
#pragma omp parallel for
    for ( long long omp_i = 0; omp_i < 10000000LL; ++omp_i ) {
        const int tid = omp_get_thread_num();
        RndSampler *sampler = samplers[tid];

        Spectrum val0, val1;
        Vector wo0, wo1;
        Float pdf0, pdf1;
        auto rnd = sampler->next4D();

        val0 = scene0.sampleAttenuatedEmitterDirect(x, rnd, sampler, &medium, 100, wo0, pdf0); val0 *= INV_FOURPI;
        val1 = scene1.sampleAttenuatedEmitterDirect(x, rnd, sampler, &medium, 100, wo1, pdf1); val1 *= INV_FOURPI;
        stat0[tid].push(val0[0]);
        stat1[tid].push((val1[0] - val0[0])/delta);
    }
    for ( int i = 1; i < nworker; ++i ) {
        stat0[0].push(stat0[i]); stat1[0].push(stat1[i]);
    }
    std::cout << std::setprecision(2) << std::setiosflags(std::ios::scientific)
              << "  FD  : " << stat0[0].getMean() << " +- " << stat0[0].getCI() << '\n'
              << "  FD  : " << stat1[0].getMean() << " +- " << stat1[0].getCI() << std::endl;

    for ( int i = 0; i < nworker; ++i ) {
        stat0[i].reset(); stat1[i].reset();
    }
#pragma omp parallel for
    for ( long long omp_i = 0; omp_i < 100000000LL; ++omp_i ) {
        const int tid = omp_get_thread_num();
        RndSampler *sampler = samplers[tid];
        FloatAD ret;

        // Area sampling
        {
            VectorAD wo;
            Float pdf;
            auto rnd = sampler->next4D();
            ret = scene0.sampleAttenuatedEmitterDirectAD(VectorAD(x), rnd, sampler, &medium, 100, wo, pdf)[0];
            ret *= INV_FOURPI;
        }

        // Edge sampling
        {
            int shape_id;
            Float t, pdf;
            const Edge* ptr_edge = scene0.sampleEdge(x, nullptr, sampler->next1D(), shape_id, t, pdf);
            if (ptr_edge != nullptr) {
                const Shape *shape = scene0.shape_list[shape_id];
                const VectorAD &v0 = shape->getVertexAD(ptr_edge->v0), &v1 = shape->getVertexAD(ptr_edge->v1);

                VectorAD w1AD = v0*(1.0f - t) + v1*t - x;
                Vector tang = (v1.val - v0.val).normalized(),
                       norm = (v0.val - x).cross(v1.val - x).normalized();
                FloatAD distAD = w1AD.norm();
                w1AD /= distAD;

                Intersection its1;
                Float deltaFunc = 0.0f;
                if ( scene0.rayIntersect(Ray(x, (w1AD.val - angleDelta*norm).normalized()), true, its1) )
                    if ( its1.isEmitter() ) {
                        assert(its1.ptr_shape == &triangle0);
                        deltaFunc += its1.Le(-w1AD.val)[0];
                    }
                if ( scene0.rayIntersect(Ray(x, (w1AD.val + angleDelta*norm).normalized()), true, its1) )
                    if ( its1.isEmitter() ) {
                        assert(its1.ptr_shape == &triangle0);
                        deltaFunc -= its1.Le(-w1AD.val)[0];
                    }

                if ( std::abs(deltaFunc) > Epsilon ) {
                    deltaFunc *= scene0.evalTransmittance(Ray(x, w1AD.val), false, &medium, distAD.val, sampler, 100);
                    Float cosTheta = -w1AD.val.dot(tang),
                          sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);

                    ret.grad() += norm.dot(w1AD.grad())*deltaFunc*sinTheta/(pdf*distAD.val*4.0f*M_PI);
                }                
            }
        }

        stat0[tid].push(ret.val);
        stat1[tid].push(ret.grad());
    }
    for ( int i = 1; i < nworker; ++i ) {
        stat0[0].push(stat0[i]); stat1[0].push(stat1[i]);
    }
    std::cout << std::setprecision(2) << std::setiosflags(std::ios::scientific)
              << "  Edge: " << stat0[0].getMean() << " +- " << stat0[0].getCI() << '\n'
              << "  Edge: " << stat1[0].getMean() << " +- " << stat1[0].getCI() << std::endl;

    std::cout << "done." << std::endl;
}


/*
 * Same as Prob0-6
 */
static void testSceneAD3()
{
    int nworker = omp_get_num_procs();
    std::vector<RndSampler*> samplers(nworker, nullptr);
    for ( int i = 0; i < nworker; ++i ) samplers[i] = new RndSampler(123, i);
    std::vector<Statistics> stat0(nworker), stat1(nworker);

    std::cout << "Testing scene AD #3 with " << nworker << " threads ..." << std::endl;
    constexpr Float delta = 1e-5f;
    constexpr Float angleDelta = 1e-4f;

    AreaLight area(1, Spectrum3f(1.0f, 1.0f, 1.0f));
    NullBSDF nullBSDF;
    DiffuseBSDF diffuseBSDF(Spectrum3f(1.0f, 1.0f, 1.0f));
    Homogeneous medium(0.5f, Spectrum3f(0.0f, 0.0f, 0.0f), 0);
    IsotropicPhaseFunction phase;

    float tri0VtxPositions[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         0.0f,  1.00, 0.0f
    };

    int tri0VtxIndices[] = {
        0, 1, 2
    };

    float tri1VtxPositions[] = {
         -0.5f, -1.0f, 1.4f,
          0.0f, -1.0f, 1.6f,
        -0.25f, -0.5f, 1.5f
    };

    int tri1VtxIndices[] = {
        0, 2, 1
    };

    Shape triangle0(tri0VtxPositions, tri0VtxIndices, nullptr, nullptr, 3, 1, -1, 1, -1, -1),
          triangle1(tri1VtxPositions, tri1VtxIndices, nullptr, nullptr, 3, 1, 0, 0, 0, 0);

    Eigen::Matrix<Float, -1, -1> tri0VtxVelocities = Eigen::Matrix<Float, -1, -1>::Zero(3*nder, 3),
                                 tri1VtxVelocities = Eigen::Matrix<Float, -1, -1>::Zero(3*nder, 3);
    tri0VtxVelocities.block<3, 1>(0, 0) << 1.0f, 0.0f, 0.0f;
    tri0VtxVelocities.block<3, 1>(0, 1) << 0.0f, 1.0f, 0.0f;
    tri0VtxVelocities.block<3, 1>(0, 2) << 0.0f, 0.0f, 1.0f;
    tri1VtxVelocities.block<3, 1>(0, 0) << 0.0f, -1.0f, 0.0f;
    tri1VtxVelocities.block<3, 1>(0, 1) << 0.0f, 0.0f, -1.0f;
    tri1VtxVelocities.block<3, 1>(0, 2) << -1.0f, 0.0f, 0.0f;

    triangle0.initVelocities(tri0VtxVelocities);
    triangle1.initVelocities(tri1VtxVelocities);

    Shape triangle0_2(triangle0), triangle1_2(triangle1);
    triangle0_2.advance(delta);
    triangle1_2.advance(delta);

    Camera camera0;
    Scene scene0(camera0,
                 std::vector<const Shape*>{&triangle0, &triangle1},
                 std::vector<const BSDF*>{&nullBSDF, &diffuseBSDF},
                 std::vector<const Emitter*>{&area},
                 std::vector<const PhaseFunction*>{&phase},
                 std::vector<const Medium*>{&medium});

    Camera camera1;
    Scene scene1(camera1,
                 std::vector<const Shape*>{&triangle0_2, &triangle1_2},
                 std::vector<const BSDF*>{&nullBSDF, &diffuseBSDF},
                 std::vector<const Emitter*>{&area},
                 std::vector<const PhaseFunction*>{&phase},
                 std::vector<const Medium*>{&medium});

    Vector x(1.0f, 1.0f, 0.5f), dx(0.3f, 0.2f, 0.1f);
    Vector dw(0.2f, 0.3f, -0.1f);
    VectorAD xAD(x), wAD(x);
    xAD.grad() = dx; wAD.grad() = dw;
    wAD.normalize();

    Ray ray0(xAD.val, -wAD.val), ray1(xAD.advance(delta), -wAD.advance(delta).normalized());
    RayAD rayAD(xAD, -wAD);

    Intersection its0, its1;
    scene0.rayIntersect(ray0, false, its0);
    scene1.rayIntersect(ray1, false, its1);

    Float trans0 = scene0.evalTransmittance(ray0, false, &medium, its0.t, samplers[0], 100),
          trans1 = scene1.evalTransmittance(ray1, false, &medium, its1.t, samplers[0], 100);

    for ( int i = 0; i < nworker; ++i ) {
        stat0[i].reset(); stat1[i].reset();
    }
#pragma omp parallel for
    for ( long long omp_i = 0; omp_i < 10000000LL; ++omp_i ) {
        const int tid = omp_get_thread_num();
        RndSampler *sampler = samplers[tid];

        Spectrum val0, val1;
        Vector wo0, wo1;
        Float pdf0, pdf1;
        auto rnd = sampler->next4D();

        val0 = scene0.sampleEmitterDirect(its0, rnd, sampler, wo0, pdf0); val0 *= its0.evalBSDF(wo0);
        val1 = scene1.sampleEmitterDirect(its1, rnd, sampler, wo1, pdf1); val1 *= its1.evalBSDF(wo1);
        stat0[tid].push((val1[0] - val0[0])/delta);

        val0 = scene0.sampleAttenuatedEmitterDirect(its0, rnd, sampler, &medium, 100, wo0, pdf0); val0 *= its0.evalBSDF(wo0)*trans0;
        val1 = scene1.sampleAttenuatedEmitterDirect(its1, rnd, sampler, &medium, 100, wo1, pdf1); val1 *= its1.evalBSDF(wo1)*trans1;
        stat1[tid].push((val1[0] - val0[0])/delta);
    }
    for ( int i = 1; i < nworker; ++i ) {
        stat0[0].push(stat0[i]); stat1[0].push(stat1[i]);
    }
    std::cout << std::setprecision(2) << std::setiosflags(std::ios::scientific)
              << "  FD  : " << stat0[0].getMean() << " +- " << stat0[0].getCI() << '\n'
              << "  FD  : " << stat1[0].getMean() << " +- " << stat1[0].getCI() << std::endl;

    IntersectionAD itsAD;
    Intersection its = itsAD.toIntersection();
    scene0.rayIntersectAD(rayAD, false, itsAD);
    assert(sameFloat(itsAD.t.val, its0.t, Epsilon, Epsilon));

    FloatAD transAD = scene0.evalTransmittanceAD(rayAD, false, &medium, itsAD.t, samplers[0], 100);
    assert(sameFloat(transAD.val, trans0, Epsilon, Epsilon));

    for ( int i = 0; i < nworker; ++i ) {
        stat0[i].reset(); stat1[i].reset();
    }
#pragma omp parallel for
    for ( long long omp_i = 0; omp_i < 100000000LL; ++omp_i ) {
        const int tid = omp_get_thread_num();
        RndSampler *sampler = samplers[tid];

        Float res0 = 0.0f, res1 = 0.0f;

        // Area sampling
        {
            SpectrumAD val;
            VectorAD wo;
            Float pdf;
            auto rnd = sampler->next4D();

            val = scene0.sampleEmitterDirectAD(itsAD, rnd, sampler, wo, pdf); val *= itsAD.evalBSDF(wo);
            res0 += val[0].grad();

            val = scene0.sampleAttenuatedEmitterDirectAD(itsAD, rnd, sampler, &medium, 100, wo, pdf); val *= itsAD.evalBSDF(wo)*transAD;
            //res1 += val[0].val;
            res1 += val[0].grad();
        }

        // Edge sampling
        {
            int shape_id;
            Float t, pdf;
            const Edge* ptr_edge = scene0.sampleEdge(itsAD.p.val, &its.geoFrame, sampler->next1D(), shape_id, t, pdf);
            if (ptr_edge != nullptr) {
                const Shape *shape = scene0.shape_list[shape_id];
                const VectorAD &v0 = shape->getVertexAD(ptr_edge->v0), &v1 = shape->getVertexAD(ptr_edge->v1);
                const VectorAD &x1 = itsAD.p;

                VectorAD w1AD = v0*(1.0f - t) + v1*t - x1;
                Vector tang = (v1.val - v0.val).normalized(),
                       norm = (v0.val - x1.val).cross(v1.val - x1.val).normalized();
                FloatAD distAD = w1AD.norm();
                w1AD /= distAD;

                Spectrum bsdfVal = its0.evalBSDF(its0.toLocal(w1AD.val));
                if ( !bsdfVal.isZero(Epsilon) ) {
                    Intersection its1;
                    Float deltaFunc0 = 0.0f;
                    if ( scene0.rayIntersect(Ray(x1.val, (w1AD.val - angleDelta*norm).normalized()), true, its1) )
                        if ( its1.isEmitter() ) {
                            assert(its1.ptr_shape == &triangle1);
                            deltaFunc0 += its1.Le(-w1AD.val)[0];
                        }
                    if ( scene0.rayIntersect(Ray(x1.val, (w1AD.val + angleDelta*norm).normalized()), true, its1) )
                        if ( its1.isEmitter() ) {
                            assert(its1.ptr_shape == &triangle1);
                            deltaFunc0 -= its1.Le(-w1AD.val)[0];
                        }

                    if ( std::abs(deltaFunc0) > Epsilon ) {
                        Float deltaFunc1 = deltaFunc0*scene0.evalTransmittance(Ray(x1.val, w1AD.val), true, &medium, distAD.val, sampler, 100);
                        Float cosTheta = -w1AD.val.dot(tang),
                              sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);

                        res0 += norm.dot(w1AD.grad())*bsdfVal[0]*deltaFunc0*sinTheta/(pdf*distAD.val);
                        res1 += transAD.val*norm.dot(w1AD.grad())*bsdfVal[0]*deltaFunc1*sinTheta/(pdf*distAD.val);
                    }
                }
            }

        }

        stat0[tid].push(res0); stat1[tid].push(res1);
    }
    for ( int i = 1; i < nworker; ++i ) {
        stat0[0].push(stat0[i]); stat1[0].push(stat1[i]);
    }
    std::cout << std::setprecision(2) << std::setiosflags(std::ios::scientific)
              << "  Edge: " << stat0[0].getMean() << " +- " << stat0[0].getCI() << '\n'
              << "  Edge: " << stat1[0].getMean() << " +- " << stat1[0].getCI() << std::endl;

    std::cout << "done." << std::endl;

    for ( int i = 0; i < nworker; ++i ) delete samplers[i];
}
