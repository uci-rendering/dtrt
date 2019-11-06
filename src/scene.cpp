#include "scene.h"
#include "math_func.h"
#include "edge_manager/bruteforce.h"
#include "edge_manager/tree.h"
#include <algorithm>
#include <assert.h>
#include <iostream>


Scene::Scene(const Camera &camera,
             const std::vector<const Shape*> &shapes,
             const std::vector<const BSDF*> &bsdfs,
             const std::vector<const Emitter*> &area_lights,
             const std::vector<const PhaseFunction*> &phases,
             const std::vector<const Medium*> &mediums)
        : camera(camera)
{
    // Initialize Embree scene
    embree_device = rtcNewDevice(nullptr);
    embree_scene = rtcNewScene(embree_device);
    rtcSetSceneBuildQuality(embree_scene, RTC_BUILD_QUALITY_HIGH);
    rtcSetSceneFlags(embree_scene, RTC_SCENE_FLAG_ROBUST);
    // Copy the scene into Embree (since Embree requires 16 bytes alignment)
    for (const Shape *shape : shapes) {
        auto mesh = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        auto vertices = (Vector4f*)rtcSetNewGeometryBuffer(
            mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
            sizeof(Vector4f), shape->num_vertices);
        for (auto i = 0; i < shape->num_vertices; i++) {
            auto vertex = shape->getVertex(i);
            vertices[i] = Vector4f(vertex(0), vertex(1), vertex(2), 0.f);
        }
        auto triangles = (Vector3i*) rtcSetNewGeometryBuffer(
            mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
            sizeof(Vector3i), shape->num_triangles);
        for (auto i = 0; i < shape->num_triangles; i++) {
            triangles[i] = shape->getIndices(i);
        }
        rtcSetGeometryVertexAttributeCount(mesh, 1);
        rtcCommitGeometry(mesh);
        rtcAttachGeometry(embree_scene, mesh);
        rtcReleaseGeometry(mesh);
    }
    rtcCommitScene(embree_scene);

    num_lights = area_lights.size();
    if (area_lights.size() > 0) {
        auto num_lights = (int)area_lights.size();

        // Build Light CDFs
        light_pmf.resize(num_lights);
        light_cdf.resize(num_lights);
        light_areas.resize(num_lights);
        auto total_light_triangles = 0;
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            const Emitter &light = *area_lights[light_id];
            const Shape &shape = *shapes[light.getShapeID()];
            total_light_triangles += shape.num_triangles;
        }
        area_cdf_pool.resize(total_light_triangles);
        area_cdf_begin.resize(num_lights);

        auto cur_tri_id = 0;
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            const Emitter &light = *area_lights[light_id];
            const Shape &shape = *shapes[light.getShapeID()];
            area_cdf_begin[light_id] = cur_tri_id;
            cur_tri_id += shape.num_triangles;
        }

        auto total_importance = Float(0);
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            const Emitter &light = *area_lights[light_id];
            const Shape &shape = *shapes[light.getShapeID()];

            Float area_sum = 0.0;
            for (int itri = 0; itri < shape.num_triangles; itri++) {
                area_sum += shape.getArea(itri);
                area_cdf_pool[itri + area_cdf_begin[light_id]] = area_sum;
            }
            light_areas[light_id] = area_sum;
            for (int itri = 0; itri < shape.num_triangles; itri++)
                area_cdf_pool[itri + area_cdf_begin[light_id]] /= area_sum;

            // Power of an area light
            light_pmf[light_id] = area_sum * luminance(light.getIntensity()) * Float(M_PI);
            total_importance += light_pmf[light_id];
        }

        assert(total_importance > Float(0));
        // Normalize PMF
        std::transform(light_pmf.begin(), light_pmf.begin() + num_lights,
                       light_pmf.begin(), [=](Float x) {return x / total_importance;});
        // Prefix sum for CDF
        light_cdf[0] = light_pmf[0];
        for (int i = 1; i < num_lights; i++) {
            light_cdf[i] = light_cdf[i - 1] + light_pmf[i];
        }
    }

    // Flatten the scene into array
    if (shapes.size() > 0) {
        shape_list.resize(shapes.size(), nullptr);
        for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
            shape_list[shape_id] = shapes[shape_id];
        }
    }
    if (bsdfs.size() > 0) {
        bsdf_list.resize(bsdfs.size(), nullptr);
        for (int bsdf_id = 0; bsdf_id < (int)bsdfs.size(); bsdf_id++) {
            bsdf_list[bsdf_id] = bsdfs[bsdf_id];
        }
    }

    if (area_lights.size() > 0) {
        emitter_list.resize(area_lights.size(), nullptr);
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            emitter_list[light_id] = area_lights[light_id];
        }
    }

    if (phases.size() > 0) {
        phase_list.resize(phases.size(), nullptr);
        for (int phase_id = 0; phase_id < (int)phases.size(); phase_id++) {
            phase_list[phase_id] = phases[phase_id];
        }
    }

    if (mediums.size() > 0) {
        medium_list.resize(mediums.size(), nullptr);
        for (int medium_id = 0; medium_id < (int)mediums.size(); medium_id++) {
            medium_list[medium_id] = mediums[medium_id];
        }
    }

    initEdges(Eigen::Array<Float, -1, 1>::Ones(shapes.size(), 1));
}

void Scene::initEdges(const Eigen::Array<Float, -1, 1> &samplingWeights) {
    assert(static_cast<size_t>(samplingWeights.rows()) == shape_list.size());    
    // ptr_edgeManager = new BruteForceEdgeManager(*this, samplingWeights);
    ptr_edgeManager = new TreeEdgeManager(*this, samplingWeights);
}

Scene::~Scene() {
    rtcReleaseScene(embree_scene);
    rtcReleaseDevice(embree_device);
    delete ptr_edgeManager;
}

bool Scene::isVisible(const Vector &p, bool pOnSurface, const Vector &q, bool qOnSurface) const {
    Vector dir = q - p;
    Float dist = dir.norm();
    dir /= dist;

    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = p.x();
    rtc_ray_hit.ray.org_y = p.y();
    rtc_ray_hit.ray.org_z = p.z();
    rtc_ray_hit.ray.dir_x = dir.x();
    rtc_ray_hit.ray.dir_y = dir.y();
    rtc_ray_hit.ray.dir_z = dir.z();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.ray.tnear = pOnSurface ? ShadowEpsilon : 0.0f;
    rtc_ray_hit.ray.tfar = qOnSurface ? (1.0f - ShadowEpsilon)*dist : dist;
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);

    return rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID;
}

Float Scene::evalTransmittance(const Ray& _ray, bool onSurface, const Medium* ptr_medium, Float remaining, RndSampler* sampler, int max_interactions) const {
    Ray ray(_ray);
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    Float transmittance = 1.0f;
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.x();
    rtc_ray_hit.ray.org_y = ray.org.y();
    rtc_ray_hit.ray.org_z = ray.org.z();
    rtc_ray_hit.ray.dir_x = ray.dir.x();
    rtc_ray_hit.ray.dir_y = ray.dir.y();
    rtc_ray_hit.ray.dir_z = ray.dir.z();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = (1.0f - ShadowEpsilon) * remaining;
    const Shape* ptr_shape = nullptr;
    int interactions = 0;
    Intersection new_its;
    while (remaining > 0) {
        Float tmax = remaining;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
        if (rtc_ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
            if (interactions == max_interactions || !bsdf_list[ptr_shape->bsdf_id]->isNull()) {
                return 0.0f;
            }
            ptr_shape->rayIntersect((int)rtc_ray_hit.hit.primID, ray, new_its);
            if ( new_its.t < tmax ) tmax = new_its.t;
        }
        if (ptr_medium != nullptr) {
            transmittance *= ptr_medium->evalTransmittance(ray, tmin, tmax, sampler);
        }
        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            break;
        // If null surface, consider transmittance
        if (ptr_shape->isMediumTransition()) {
            int med_id  = new_its.geoFrame.n.dot(ray.dir)>0.0
                          ? ptr_shape->med_ext_id : ptr_shape->med_int_id;
            ptr_medium = med_id != -1 ? medium_list[med_id] : nullptr;
        }
        ray.org = ray(tmax);
        rtc_ray_hit.ray.org_x = ray.org.x();
        rtc_ray_hit.ray.org_y = ray.org.y();
        rtc_ray_hit.ray.org_z = ray.org.z();
        rtc_ray_hit.ray.tnear = tmin = ShadowEpsilon;
        remaining -= tmax;
        rtc_ray_hit.ray.tfar = tmax = (1 - ShadowEpsilon)*remaining;
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        interactions++;
    }
    return transmittance;
}

FloatAD Scene::evalTransmittanceAD(const RayAD& _ray, bool onSurface, const Medium* ptr_medium, FloatAD remaining, RndSampler* sampler, int max_interactions) const {
    RayAD ray(_ray);
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    FloatAD transmittance(1.0f);
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.val.x();
    rtc_ray_hit.ray.org_y = ray.org.val.y();
    rtc_ray_hit.ray.org_z = ray.org.val.z();
    rtc_ray_hit.ray.dir_x = ray.dir.val.x();
    rtc_ray_hit.ray.dir_y = ray.dir.val.y();
    rtc_ray_hit.ray.dir_z = ray.dir.val.z();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = (1.0f - ShadowEpsilon) * remaining.val;
    const Shape* ptr_shape = nullptr;
    int interactions = 0;
    IntersectionAD new_its;
    while (remaining > 0) {
        FloatAD tmax = remaining;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
        if (rtc_ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
            if (interactions == max_interactions || !bsdf_list[ptr_shape->bsdf_id]->isNull()) {
                return FloatAD();
            }
            ptr_shape->rayIntersectAD((int)rtc_ray_hit.hit.primID, ray, new_its);
            if ( new_its.t < tmax ) tmax = new_its.t;
        }
        if (ptr_medium != nullptr) {
            transmittance *= ptr_medium->evalTransmittanceAD(ray, tmin, tmax, sampler);
        }
        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            break;
        // If null surface, consider transmittance
        if (ptr_shape->isMediumTransition()) {
            int med_id  = new_its.geoFrame.n.dot(ray.dir)>0.0
                          ? ptr_shape->med_ext_id : ptr_shape->med_int_id;
            ptr_medium = med_id != -1 ? medium_list[med_id] : nullptr;
        }
        ray.org = ray(tmax);
        rtc_ray_hit.ray.org_x = ray.org.val.x();
        rtc_ray_hit.ray.org_y = ray.org.val.y();
        rtc_ray_hit.ray.org_z = ray.org.val.z();
        rtc_ray_hit.ray.tnear = tmin = ShadowEpsilon;
        remaining -= tmax;
        tmax = (1.0f - ShadowEpsilon)*remaining;
        rtc_ray_hit.ray.tfar = tmax.val;
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        interactions++;
    }
    return transmittance;
}

Spectrum Scene::sampleEmitterDirect(const Intersection &its, const Array4 &rnd_light, RndSampler* sampler, Vector& wo, Float &pdf) const {
    // Sample point on emitter
    int light_id = std::upper_bound(light_cdf.begin(), light_cdf.end(), rnd_light(0)) - light_cdf.begin();
    light_id = clamp(light_id, 0, num_lights-1);
    const auto &light = *emitter_list[light_id];
    const auto &shape = *shape_list[light.getShapeID()];
    auto cdf_begin = area_cdf_pool.begin() + area_cdf_begin[light_id];
    int tri_id = std::upper_bound(cdf_begin, cdf_begin + shape.num_triangles, rnd_light(1)) - cdf_begin;
    tri_id = clamp(tri_id, 0, shape.num_triangles-1);
    Vector light_pos, light_norm;
    shape.samplePosition(tri_id, Vector2(rnd_light(2), rnd_light(3)), light_pos, light_norm);

    Vector dir = light_pos - its.p;
    pdf = light_pmf[light_id]/light_areas[light_id];
    if (dir.dot(its.shFrame.n)>0 && dir.dot(light_norm)<0 && pdf != 0) {
        Float dist = dir.norm();
        dir = dir/dist;
        RTCIntersectContext rtc_context;
        rtcInitIntersectContext(&rtc_context);
        RTCRayHit rtc_ray_hit;
        rtc_ray_hit.ray.org_x = its.p.x();
        rtc_ray_hit.ray.org_y = its.p.y();
        rtc_ray_hit.ray.org_z = its.p.z();
        rtc_ray_hit.ray.dir_x = dir.x();
        rtc_ray_hit.ray.dir_y = dir.y();
        rtc_ray_hit.ray.dir_z = dir.z();
        rtc_ray_hit.ray.mask = (unsigned int)(-1);
        rtc_ray_hit.ray.time = 0.f;
        rtc_ray_hit.ray.flags = 0;
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.ray.tnear = ShadowEpsilon;
        rtc_ray_hit.ray.tfar = (1 - ShadowEpsilon) * dist;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);

        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            pdf *= dist*dist / light_norm.dot(-dir);
            wo = its.toLocal(dir);
            return light.eval(light_norm, -dir)/pdf;
        }
    }
    return Spectrum::Zero();
}

SpectrumAD Scene::sampleEmitterDirectAD(const IntersectionAD &its, const Array4 &rnd_light, RndSampler* sampler, VectorAD& wo, Float &pdf) const {
    // Sample point on emitter
    int light_id = std::upper_bound(light_cdf.begin(), light_cdf.end(), rnd_light(0)) - light_cdf.begin();
    light_id = clamp(light_id, 0, num_lights-1);
    const auto &light = *emitter_list[light_id];
    const auto &shape = *shape_list[light.getShapeID()];
    auto cdf_begin = area_cdf_pool.begin() + area_cdf_begin[light_id];
    int tri_id = std::upper_bound(cdf_begin, cdf_begin + shape.num_triangles, rnd_light(1)) - cdf_begin;
    tri_id = clamp(tri_id, 0, shape.num_triangles-1);
    Vector light_pos, light_norm;
    shape.samplePosition(tri_id, Vector2(rnd_light(2), rnd_light(3)), light_pos, light_norm);

    Vector dir = light_pos - its.p.val;
    pdf = light_pmf[light_id]/light_areas[light_id];
    if (dir.dot(its.shFrame.n.val)>0 && dir.dot(light_norm)<0 && pdf != 0) {
        Float dist = dir.norm();
        dir = dir/dist;
        RTCIntersectContext rtc_context;
        rtcInitIntersectContext(&rtc_context);
        RTCRayHit rtc_ray_hit;
        rtc_ray_hit.ray.org_x = its.p.val.x();
        rtc_ray_hit.ray.org_y = its.p.val.y();
        rtc_ray_hit.ray.org_z = its.p.val.z();
        rtc_ray_hit.ray.dir_x = dir.x();
        rtc_ray_hit.ray.dir_y = dir.y();
        rtc_ray_hit.ray.dir_z = dir.z();
        rtc_ray_hit.ray.mask = (unsigned int)(-1);
        rtc_ray_hit.ray.time = 0.f;
        rtc_ray_hit.ray.flags = 0;
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.ray.tnear = ShadowEpsilon;
        rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);

        // Only works for area light
        assert(light.shape_id >= 0);
        if (static_cast<int>(rtc_ray_hit.hit.geomID) == light.shape_id) {
            const auto &shape = shape_list[light.shape_id];
            IntersectionAD new_its;
            shape->rayIntersectAD(rtc_ray_hit.hit.primID, RayAD(its.p, dir), new_its);
            if ( std::abs(new_its.t.val - dist) < ShadowEpsilon ) {
                pdf *= dist*dist / light_norm.dot(-dir);
                wo = its.toLocal(dir);
                return light.evalAD(new_its.shFrame.n, VectorAD(-dir))/pdf;
            }
        }
    }
    return SpectrumAD();
}

Spectrum Scene::sampleAttenuatedEmitterDirect(const Vector &pscatter, const Array4 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                              Vector& wo, Float& pdf) const {
    // sample a ray on the emitter
    int light_id = std::upper_bound(light_cdf.begin(), light_cdf.end(), rnd_light(0)) - light_cdf.begin();
    light_id = clamp(light_id, 0, num_lights-1);
    const auto &light = *emitter_list[light_id];
    const auto &shape = *shape_list[light.getShapeID()];
    auto cdf_begin = area_cdf_pool.begin() + area_cdf_begin[light_id];
    int tri_id = std::upper_bound(cdf_begin, cdf_begin + shape.num_triangles, rnd_light(1)) - cdf_begin;
    tri_id = clamp(tri_id, 0, shape.num_triangles-1);
    Vector light_pos, light_norm;
    shape.samplePosition(tri_id, Vector2(rnd_light(2), rnd_light(3)), light_pos, light_norm);

    Vector dir = light_pos - pscatter;
    pdf = light_pmf[light_id]/light_areas[light_id];
    if (dir.dot(light_norm)<0 && pdf != 0) {
        Float dist = dir.norm();
        dir = dir/dist;
        Ray shadow_ray(pscatter, dir);
        Float transmittance = evalTransmittance(shadow_ray, 0.0, ptr_medium, dist, sampler, max_interactions);
        if (transmittance != 0) {
            pdf *= dist*dist / light_norm.dot(-dir);
            wo = dir;
            return transmittance * light.eval(light_norm, -dir)/pdf;
        }
    }
    return Spectrum::Zero();
}

SpectrumAD Scene::sampleAttenuatedEmitterDirectAD(const VectorAD &pscatter, const Array4 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                                  VectorAD& wo, Float& pdf) const {
    // sample a ray on the emitter
    int light_id = std::upper_bound(light_cdf.begin(), light_cdf.end(), rnd_light(0)) - light_cdf.begin();
    light_id = clamp(light_id, 0, num_lights-1);
    const auto &light = *emitter_list[light_id];
    const auto &shape = *shape_list[light.getShapeID()];
    auto cdf_begin = area_cdf_pool.begin() + area_cdf_begin[light_id];
    int tri_id = std::upper_bound(cdf_begin, cdf_begin + shape.num_triangles, rnd_light(1)) - cdf_begin;
    tri_id = clamp(tri_id, 0, shape.num_triangles-1);
    Vector light_pos, light_norm;
    shape.samplePosition(tri_id, Vector2(rnd_light(2), rnd_light(3)), light_pos, light_norm);

    Vector dir = light_pos - pscatter.val;
    if (dir.dot(light_norm)<0) {
        Float dist = dir.norm();
        dir /= dist;

        IntersectionAD itsNear, itsFar;
        SpectrumAD ret = rayIntersectAndLookForEmitterAD(RayAD(pscatter, dir), false, sampler, ptr_medium, max_interactions, itsNear, pdf, &itsFar);
        if ( !ret.isZero(Epsilon) && pdf > Epsilon && itsFar.ptr_shape == &shape && std::abs(itsFar.t.val - dist) < ShadowEpsilon ) {
            wo = dir;
            return ret/pdf;
        }
    }
    return SpectrumAD();
}

Spectrum Scene::sampleAttenuatedEmitterDirect(const Intersection& its, const Array4 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                              Vector& wo, Float& pdf, bool flag) const {
    // sample a ray on the emitter
    int light_id = std::upper_bound(light_cdf.begin(), light_cdf.end(), rnd_light(0)) - light_cdf.begin();
    light_id = clamp(light_id, 0, num_lights-1);
    const auto &light = *emitter_list[light_id];
    const auto &shape = *shape_list[light.getShapeID()];
    auto cdf_begin = area_cdf_pool.begin() + area_cdf_begin[light_id];
    int tri_id = std::upper_bound(cdf_begin, cdf_begin + shape.num_triangles, rnd_light(1)) - cdf_begin;
    tri_id = clamp(tri_id, 0, shape.num_triangles-1);
    Vector light_pos, light_norm;
    shape.samplePosition(tri_id, Vector2(rnd_light(2), rnd_light(3)), light_pos, light_norm);

    Vector dir = light_pos - its.p;
    pdf = light_pmf[light_id]/light_areas[light_id];
    bool is_transmissive = bsdf_list[shape.bsdf_id]->isTransmissive();
    if ((is_transmissive || dir.dot(its.shFrame.n)>0) && dir.dot(light_norm)<0 && pdf != 0)
    {
        if (its.isMediumTransition())
            ptr_medium = its.getTargetMedium(dir);

        Float dist = dir.norm();
        dir = dir/dist;
        Ray shadow_ray(its.p, dir);
        wo = its.toLocal(dir);
        if ( flag && math::signum(its.wi.z()) != math::signum(wo.z()) ) --max_interactions;
        if ( max_interactions >= 0 ) {
            Float transmittance = evalTransmittance(shadow_ray, ShadowEpsilon, ptr_medium, dist, sampler, max_interactions);
            if (transmittance != 0) {
                pdf *= dist*dist / light_norm.dot(-dir);
                return transmittance * light.eval(light_norm, -dir)/pdf;
            }
        }
    }
    return Spectrum::Zero();
}

SpectrumAD Scene::sampleAttenuatedEmitterDirectAD(const IntersectionAD& its, const Array4 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                                  VectorAD& wo, Float& pdf) const {
    // sample a ray on the emitter
    int light_id = std::upper_bound(light_cdf.begin(), light_cdf.end(), rnd_light(0)) - light_cdf.begin();
    light_id = clamp(light_id, 0, num_lights-1);
    const auto &light = *emitter_list[light_id];
    const auto &shape = *shape_list[light.getShapeID()];
    auto cdf_begin = area_cdf_pool.begin() + area_cdf_begin[light_id];
    int tri_id = std::upper_bound(cdf_begin, cdf_begin + shape.num_triangles, rnd_light(1)) - cdf_begin;
    tri_id = clamp(tri_id, 0, shape.num_triangles-1);
    Vector light_pos, light_norm;
    shape.samplePosition(tri_id, Vector2(rnd_light(2), rnd_light(3)), light_pos, light_norm);

    Vector dir = light_pos - its.p.val;
    bool is_transmissive = bsdf_list[shape.bsdf_id]->isTransmissive();
    if ((is_transmissive || dir.dot(its.shFrame.n.val)>0) && dir.dot(light_norm)<0)
    {
        if (its.isMediumTransition())
            ptr_medium = its.getTargetMedium(dir);

        Float dist = dir.norm();
        dir /= dist;

        IntersectionAD itsNear, itsFar;
        SpectrumAD ret = rayIntersectAndLookForEmitterAD(RayAD(its.p, dir), true, sampler, ptr_medium, max_interactions, itsNear, pdf, &itsFar);
        if ( !ret.isZero(Epsilon) && pdf > Epsilon && itsFar.ptr_shape == &shape && std::abs(itsFar.t.val - dist) < ShadowEpsilon ) {
            wo = its.toLocal(dir);
            return ret/pdf;
        }
    }
    return SpectrumAD();
}

Float Scene::pdfEmitterSample(const Intersection& its) const {
    int light_id = its.ptr_shape->light_id;
    assert(light_id >= 0);
    return light_pmf[light_id]/light_areas[light_id];
}

Float Scene::pdfEmitterSample(const IntersectionAD& its) const {
    int light_id = its.ptr_shape->light_id;
    assert(light_id >= 0);
    return light_pmf[light_id]/light_areas[light_id];
}

bool Scene::rayIntersect(const Ray &ray, bool onSurface, Intersection& its) const {
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.x();
    rtc_ray_hit.ray.org_y = ray.org.y();
    rtc_ray_hit.ray.org_z = ray.org.z();
    rtc_ray_hit.ray.dir_x = ray.dir.x();
    rtc_ray_hit.ray.dir_y = ray.dir.y();
    rtc_ray_hit.ray.dir_z = ray.dir.z();
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        its.t = std::numeric_limits<Float>::infinity();
        its.ptr_shape = nullptr;
        return false;
    } else {
        // Fill in the corresponding pointers
        its.ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        its.ptr_med_int = (its.ptr_shape->med_int_id >= 0) ? medium_list[its.ptr_shape->med_int_id] : nullptr;
        its.ptr_med_ext = (its.ptr_shape->med_ext_id >= 0) ? medium_list[its.ptr_shape->med_ext_id] : nullptr;
        its.ptr_emitter = (its.ptr_shape->light_id>= 0) ? emitter_list[its.ptr_shape->light_id] : nullptr;
        assert(its.ptr_shape->bsdf_id >= 0);
        its.ptr_bsdf = bsdf_list[its.ptr_shape->bsdf_id];
        // Ray-Shape intersection
        int tri_id = (int)rtc_ray_hit.hit.primID;
        its.ptr_shape->rayIntersect(tri_id, ray, its);
        return true;
    }
}

bool Scene::rayIntersectAD(const RayAD &ray, bool onSurface, IntersectionAD& its) const {
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.val.x();
    rtc_ray_hit.ray.org_y = ray.org.val.y();
    rtc_ray_hit.ray.org_z = ray.org.val.z();
    rtc_ray_hit.ray.dir_x = ray.dir.val.x();
    rtc_ray_hit.ray.dir_y = ray.dir.val.y();
    rtc_ray_hit.ray.dir_z = ray.dir.val.z();
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        its.t = std::numeric_limits<Float>::infinity();
        its.ptr_shape = nullptr;
        return false;
    } else {
        // Fill in the corresponding pointers
        its.ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        its.ptr_med_int = (its.ptr_shape->med_int_id >= 0) ? medium_list[its.ptr_shape->med_int_id] : nullptr;
        its.ptr_med_ext = (its.ptr_shape->med_ext_id >= 0) ? medium_list[its.ptr_shape->med_ext_id] : nullptr;
        its.ptr_emitter = (its.ptr_shape->light_id>= 0) ? emitter_list[its.ptr_shape->light_id] : nullptr;
        assert(its.ptr_shape->bsdf_id >= 0);
        its.ptr_bsdf = bsdf_list[its.ptr_shape->bsdf_id];
        // Ray-Shape intersection
        int tri_id = (int)rtc_ray_hit.hit.primID;
        its.ptr_shape->rayIntersectAD(tri_id, ray, its);
        return true;
    }
}

Spectrum Scene::rayIntersectAndLookForEmitter(const Ray &_ray, bool onSurface, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                              Intersection &its, Float& pdf_nee) const {
    Ray ray(_ray);
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    int interactions = 0;
    RTCIntersectContext rtc_context;
    Vector scattering_pos = ray.org;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.x();
    rtc_ray_hit.ray.org_y = ray.org.y();
    rtc_ray_hit.ray.org_z = ray.org.z();
    rtc_ray_hit.ray.dir_x = ray.dir.x();
    rtc_ray_hit.ray.dir_y = ray.dir.y();
    rtc_ray_hit.ray.dir_z = ray.dir.z();
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    // Apply first intersection and store the intersection record to its
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    const Shape* ptr_shape = nullptr;
    its.ptr_shape = nullptr;
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        its.t = std::numeric_limits<Float>::infinity();
        return Spectrum::Zero();
    } else {
        its.ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        its.ptr_med_int = (its.ptr_shape->med_int_id >= 0) ? medium_list[its.ptr_shape->med_int_id] : nullptr;
        its.ptr_med_ext = (its.ptr_shape->med_ext_id >= 0) ? medium_list[its.ptr_shape->med_ext_id] : nullptr;
        its.ptr_emitter = (its.ptr_shape->light_id>= 0) ? emitter_list[its.ptr_shape->light_id] : nullptr;
        assert(its.ptr_shape->bsdf_id >= 0);
        its.ptr_bsdf = bsdf_list[its.ptr_shape->bsdf_id];
        int tri_id = (int)rtc_ray_hit.hit.primID;
        its.ptr_shape->rayIntersect(tri_id, ray, its);
    }

    Float transmittance = 1.0;
    Intersection new_its;
    while (true) {
        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            return Spectrum::Zero();

        ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        ptr_shape->rayIntersect((int)rtc_ray_hit.hit.primID, ray, new_its);
        Float tmax = new_its.t;

        if (ptr_medium)
            transmittance *= ptr_medium->evalTransmittance(ray, tmin, tmax, sampler);
        // check if hit emitter
        if (ptr_shape->isEmitter()) {
            auto dist_sq = (new_its.p - scattering_pos).squaredNorm();
            auto geometry_term = new_its.wi.z()/ dist_sq;
            int light_id = ptr_shape->light_id;
            pdf_nee = light_pmf[light_id]/(light_areas[light_id] * geometry_term);
            return transmittance * emitter_list[light_id]->eval(new_its, -ray.dir);
        }
        // check if hit a surface (not null surface) or emitter
        if (interactions == max_interactions || !bsdf_list[ptr_shape->bsdf_id]->isNull())
            return Spectrum::Zero();

        // If null surface, keep tracing
        if (ptr_shape->isMediumTransition()) {
            int med_id  = new_its.geoFrame.n.dot(ray.dir)>0.0 ? ptr_shape->med_ext_id
                                                              : ptr_shape->med_int_id;
            ptr_medium = med_id != -1 ? medium_list[med_id] : nullptr;
        }
        ray.org += ray.dir*tmax;
        rtc_ray_hit.ray.org_x = ray.org.x();
        rtc_ray_hit.ray.org_y = ray.org.y();
        rtc_ray_hit.ray.org_z = ray.org.z();
        rtc_ray_hit.ray.tnear = tmin = ShadowEpsilon;
        rtc_ray_hit.ray.tfar = tmax = std::numeric_limits<Float>::infinity();
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        interactions++;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    }
}

SpectrumAD Scene::rayIntersectAndLookForEmitterAD(const RayAD &_ray, bool onSurface, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                                  IntersectionAD &its, Float& pdf_nee, IntersectionAD *itsFar) const
{
    RayAD ray(_ray);
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    int interactions = 0;
    RTCIntersectContext rtc_context;
    Vector scattering_pos = ray.org.val;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.val.x();
    rtc_ray_hit.ray.org_y = ray.org.val.y();
    rtc_ray_hit.ray.org_z = ray.org.val.z();
    rtc_ray_hit.ray.dir_x = ray.dir.val.x();
    rtc_ray_hit.ray.dir_y = ray.dir.val.y();
    rtc_ray_hit.ray.dir_z = ray.dir.val.z();
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    // Apply first intersection and store the intersection record to its
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    const Shape* ptr_shape = nullptr;
    its.ptr_shape = nullptr;
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        its.t = std::numeric_limits<Float>::infinity();
        return SpectrumAD();
    } else {
        its.ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        its.ptr_med_int = (its.ptr_shape->med_int_id >= 0) ? medium_list[its.ptr_shape->med_int_id] : nullptr;
        its.ptr_med_ext = (its.ptr_shape->med_ext_id >= 0) ? medium_list[its.ptr_shape->med_ext_id] : nullptr;
        its.ptr_emitter = (its.ptr_shape->light_id>= 0) ? emitter_list[its.ptr_shape->light_id] : nullptr;
        assert(its.ptr_shape->bsdf_id >= 0);
        its.ptr_bsdf = bsdf_list[its.ptr_shape->bsdf_id];
        int tri_id = (int)rtc_ray_hit.hit.primID;
        its.ptr_shape->rayIntersectAD(tri_id, ray, its);
    }

    FloatAD transmittance(1.0), dist;
    IntersectionAD new_its;
    while (true) {
        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            return SpectrumAD();

        ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        ptr_shape->rayIntersectAD((int)rtc_ray_hit.hit.primID, ray, new_its);
        FloatAD tmax = new_its.t;
        dist += new_its.t;

        if (ptr_medium)
            transmittance *= ptr_medium->evalTransmittanceAD(ray, tmin, tmax, sampler);
        // check if hit emitter
        if (ptr_shape->isEmitter()) {
            auto dist_sq = (new_its.p.val - scattering_pos).squaredNorm();
            auto geometry_term = new_its.wi.val.z()/dist_sq;
            int light_id = ptr_shape->light_id;
            pdf_nee = light_pmf[light_id]/(light_areas[light_id] * geometry_term);
            if ( itsFar != nullptr ) {
                *itsFar = new_its; itsFar->ptr_shape = ptr_shape; itsFar->t = dist;
            }
            return transmittance * emitter_list[light_id]->evalAD(new_its, -ray.dir);
        }
        // check if hit a surface (not null surface) or emitter
        if (interactions == max_interactions || !bsdf_list[ptr_shape->bsdf_id]->isNull())
            return SpectrumAD();

        // If null surface, keep tracing
        if (ptr_shape->isMediumTransition()) {
            int med_id  = new_its.geoFrame.n.val.dot(ray.dir.val) > 0.0 ? ptr_shape->med_ext_id
                                                                        : ptr_shape->med_int_id;
            ptr_medium = med_id != -1 ? medium_list[med_id] : nullptr;
        }
        ray.org += ray.dir*tmax;
        tmax = std::numeric_limits<Float>::infinity();
        rtc_ray_hit.ray.org_x = ray.org.val.x();
        rtc_ray_hit.ray.org_y = ray.org.val.y();
        rtc_ray_hit.ray.org_z = ray.org.val.z();
        rtc_ray_hit.ray.tnear = tmin = ShadowEpsilon;
        rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        interactions++;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    }
}

Spectrum Scene::sampleEmitterPosition(const Array4 &rnd_light, Intersection& its) const {
    int light_id = std::upper_bound(light_cdf.begin(), light_cdf.end(), rnd_light(0)) - light_cdf.begin();
    light_id = clamp(light_id, 0, num_lights-1);
    its.ptr_emitter = emitter_list[light_id];
    its.ptr_shape = shape_list[its.ptr_emitter->getShapeID()];

    int med_id = its.ptr_shape->med_ext_id;
    its.ptr_med_ext =  med_id >= 0 ? medium_list[med_id] : nullptr;
    med_id = its.ptr_shape->med_int_id;
    its.ptr_med_int = med_id >= 0 ? medium_list[med_id] : nullptr;

    auto cdf_begin = area_cdf_pool.begin() + area_cdf_begin[light_id];
    int num_triangles = its.ptr_shape->num_triangles;
    int tri_id = std::upper_bound(cdf_begin, cdf_begin + num_triangles, rnd_light(1)) - cdf_begin;
    tri_id = clamp(tri_id, 0, num_triangles-1);
    Vector shading_norm;
    its.ptr_shape->samplePosition(tri_id, Vector2(rnd_light(2), rnd_light(3)), its.p, shading_norm);
    its.shFrame = Frame(shading_norm);
    return its.ptr_emitter->getIntensity() / light_pmf[light_id] * light_areas[light_id];
}

Float Scene::sampleAttenuatedSensorDirect(const Intersection& its, RndSampler* sampler, int max_interactions, Vector2& pixel_uv, Vector& dir) const {
    Float value = camera.sampleDirect(its.p, pixel_uv, dir);
    if (value != 0.0f) {
        const Medium* ptr_medium = its.getTargetMedium(dir);
        Float dist = (its.p-camera.cpos.val).norm();
        value *= evalTransmittance(Ray(its.p, dir), true, ptr_medium, dist, sampler, max_interactions);
        return value;
    } else {
        return 0.0f;
    }
}

Float Scene::sampleAttenuatedSensorDirect(const Vector& p, const Medium* ptr_med, RndSampler* sampler, int max_interactions, Vector2& pixel_uv, Vector& dir) const {
    Float value = camera.sampleDirect(p, pixel_uv, dir);
    if (value != 0.0f) {
        Float dist = (p-camera.cpos.val).norm();
        value *= evalTransmittance(Ray(p, dir), false, ptr_med, dist, sampler, max_interactions);
        return value;
    } else {
        return 0.0f;
    }    
}