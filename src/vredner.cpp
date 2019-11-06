#include "scene.h"
#include "emitter/area.h"
#include "emitter/area2.h"
#include "bsdf/null.h"
#include "bsdf/diffuse.h"
#include "bsdf/phong.h"
#include "bsdf/roughdielectric.h"
#include "medium/homogeneous.h"
#include "medium/heterogeneous.h"
#include "integrator/path.h"
#include "integrator/ptracer.h"
#include "integrator/volpath_simple.h"
#include "integrator/volpath.h"
#include "integrator/direct.h"
#include "integrator/differential/directAD.h"
#include "integrator/differential/pathAD.h"
#include "integrator/differential/volpathAD.h"
#include "unit_test/ad_test.h"
#include "config.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(vredner, m) {
    m.doc() = "vRedner"; // optional module docstring

    py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<std::size_t>());
    py::class_<ptr<int>>(m, "int_ptr")
        .def(py::init<std::size_t>());

    py::class_<Spectrum3f>(m, "Spectrum3f")
        .def(py::init<float, float, float>());

    py::class_<Camera>(m, "Camera")
        .def(py::init<int, int, ptr<float>, ptr<float>, float, int, ptr<float>>())
        .def("set_rect", &Camera::setCropRect);

    py::class_<Shape>(m, "Shape")
        .def(py::init<ptr<float>,
                      ptr<int>,
                      ptr<float>,
                      ptr<float>,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      ptr<float>>())
        .def_readonly("num_vertices", &Shape::num_vertices)
        .def("has_uvs", &Shape::hasUVs)
        .def("has_normals", &Shape::hasNormals);

    py::class_<BSDF>(m, "BSDF");
    py::class_<DiffuseBSDF, BSDF>(m, "BSDF_diffuse")
        .def(py::init<Spectrum3f>())
        .def(py::init<Spectrum3f, ptr<float>>());
    py::class_<NullBSDF, BSDF>(m, "BSDF_null")
        .def(py::init<>());
    py::class_<PhongBSDF, BSDF>(m, "BSDF_Phong")
        .def(py::init<Spectrum3f, Spectrum3f, float>())
        .def(py::init<Spectrum3f, Spectrum3f, float, ptr<float>, ptr<float>, ptr<float>>());
    py::class_<RoughDielectricBSDF, BSDF>(m, "BSDF_roughdielectric")
        .def(py::init<float, float, float>())
        .def(py::init<float, float, float, ptr<float>>())
        .def(py::init<float, float, float, ptr<float>, ptr<float>>());

    py::class_<PhaseFunction>(m, "Phase");
    py::class_<HGPhaseFunction, PhaseFunction>(m, "HG")
        .def(py::init<float>())
        .def(py::init<float, ptr<float>>());
    py::class_<IsotropicPhaseFunction, PhaseFunction>(m, "Isotropic")
        .def(py::init<>());

    py::class_<Emitter>(m, "Emitter");
    py::class_<AreaLight, Emitter>(m, "AreaLight")
        .def(py::init<int, ptr<float>, bool>());
    py::class_<AreaLightEx, Emitter>(m, "AreaLightEx")
        .def(py::init<int, Spectrum3f, float>())
        .def(py::init<int, Spectrum3f, float, ptr<float>, ptr<float>>());

    py::class_<Medium>(m, "Medium");
    py::class_<Homogeneous, Medium>(m, "Homogeneous")
        .def(py::init<float, Spectrum3f, int>())
        .def(py::init<float, Spectrum3f, int, ptr<float>, ptr<float>>());
    py::class_<Heterogeneous, Medium>(m, "Heterogeneous")
        .def(py::init<const std::string &, ptr<float>, float, Spectrum3f, int>())
        .def(py::init<const std::string &, ptr<float>, float, Spectrum3f, int,
                      ptr<float>,  ptr<float>,  ptr<float>,  ptr<float>>())
        .def(py::init<const std::string &, ptr<float>, float, const std::string &, int>())
        .def(py::init<const std::string &, ptr<float>, float, const std::string &, int,
                      ptr<float>,  ptr<float>,  ptr<float>>());

    py::class_<Scene>(m, "Scene")
        .def(py::init<const Camera &,
                      const std::vector<const Shape*> &,
                      const std::vector<const BSDF*> &,
                      const std::vector<const Emitter*> &,
                      const std::vector<const PhaseFunction*> &,
                      const std::vector<const Medium*>>())
        .def("initEdges", &Scene::initEdgesPy);

    py::class_<RenderOptions>(m, "RenderOptions")
        .def(py::init<uint64_t, int, int, int, int, bool>())
        .def(py::init<uint64_t, int, int, int, int, bool, int>())
        .def(py::init<uint64_t, int, int, int, int, bool, int, float>())
        .def_readwrite("seed", &RenderOptions::seed)
        .def_readwrite("spp", &RenderOptions::num_samples)
        .def_readwrite("sppe", &RenderOptions::num_samples_primary_edge)
        .def_readwrite("sppse", &RenderOptions::num_samples_secondary_edge)
        .def_readwrite("max_bounces", &RenderOptions::max_bounces)
        .def_readwrite("quiet", &RenderOptions::quiet)
        .def_readwrite("mode", &RenderOptions::mode)
        .def_readwrite("ddistCoeff", &RenderOptions::ddistCoeff);

    py::class_<Integrator>(m, "Integrator");

    py::class_<DirectIntegrator, Integrator>(m, "DirectIntegrator")
        .def(py::init<>())
        .def("render",&DirectIntegrator::render);

    py::class_<PathTracer, Integrator>(m, "PathTracer")
        .def(py::init<>())
        .def("render",&PathTracer::render);

    py::class_<ParticleTracer, Integrator>(m, "ParticleTracer")
        .def(py::init<>())
        .def("render",&ParticleTracer::render);


    py::class_<VolPathTracer, Integrator>(m, "VolPathTracer")
        .def(py::init<>())
        .def("render",&VolPathTracer::render);

    py::class_<VolPathTracerSimple, Integrator>(m, "VolPathTracerSimple")
        .def(py::init<>())
        .def("render",&VolPathTracerSimple::render);

    py::class_<DirectIntegratorAD, Integrator>(m, "DirectAD")
        .def(py::init<>())
        .def("render",&DirectIntegratorAD::render);

    py::class_<PathTracerAD, Integrator>(m, "PathTracerAD")
        .def(py::init<>())
        .def("render",&PathTracerAD::render);

    py::class_<VolPathTracerAD, Integrator>(m, "VolPathTracerAD")
        .def(py::init<>())
        .def("render",&VolPathTracerAD::render);

    m.attr("nder") = nder;
    m.attr("angleEps") = AngleEpsilon;

    // Unit Tests
    m.def("ad_test", &ad_test, "");
}
