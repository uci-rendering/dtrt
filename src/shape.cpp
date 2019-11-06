#include "shape.h"
#include "ray.h"
#include "rayAD.h"
#include "intersection.h"
#include "intersectionAD.h"
#include <map>
#include <iomanip>


Shape::Shape(ptr<float> vertices, ptr<int> indices, ptr<float> uvs, ptr<float> normals, int num_vertices, int num_triangles,
            int light_id, int bsdf_id, int med_int_id, int med_ext_id, ptr<float> velocities):
            num_vertices(num_vertices), num_triangles(num_triangles), light_id(light_id), bsdf_id(bsdf_id), med_int_id(med_int_id), med_ext_id(med_ext_id)
{
    this->vertices.resize(num_vertices);
    for (int i = 0; i < num_vertices; i++)
        this->vertices[i] = Vector3AD(Vector3(vertices[3*i], vertices[3*i + 1], vertices[3*i + 2]));

    if (normals.get() != nullptr) {
        this->normals.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++)
            this->normals[i] = Vector3AD(Vector3(normals[3*i], normals[3*i + 1], normals[3*i + 2]).normalized());
    }

    this->indices.resize(num_triangles);
    for (int i = 0; i < num_triangles; i++)
        this->indices[i] = Vector3i(indices[3*i], indices[3*i + 1], indices[3*i + 2]);

    if (uvs.get() != nullptr ) {
        this->uvs.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++)
            this->uvs[i] = Vector2(uvs[2*i], uvs[2*i + 1]);
    }

    computeFaceNormals();
    constructEdges();

    if (!velocities.is_null()) {
        int nrows = 3;
        int ncols = num_vertices;
        for (int ider = 0; ider < nder; ider++) {
            Eigen::MatrixXf dx(nrows, ncols);
            dx = Eigen::Map<Eigen::MatrixXf>(velocities.get() + ider*nrows*ncols, dx.rows(), dx.cols());
            if (normals.get() != nullptr) {
                Eigen::MatrixXf dn(nrows, ncols);
                dn = Eigen::Map<Eigen::MatrixXf>(velocities.get() + num_vertices*3*nder, dn.rows(), dn.cols());
                initVelocities(dx.cast<Float>(), dn.cast<Float>(), ider);
            } else {
                initVelocities(dx.cast<Float>(), ider);
            }
        }
    }
}

void Shape::zeroVelocities() {
    for ( int i = 0; i < num_vertices; i++ )
        vertices[i].zeroGrad();
    if ( !normals.empty() ) {
        assert(static_cast<int>(normals.size()) == num_vertices);
        for ( int i = 0; i < num_vertices; i++ )
            normals[i].zeroGrad();
    }
    computeFaceNormals();
}

void Shape::initVelocities(const Eigen::Matrix<Float, -1, -1> &dx) {
    assert(dx.rows() == 3*nder && dx.cols() == num_vertices);
    for ( int i = 0; i < num_vertices; i++ )
        for ( int j = 0; j < nder; ++j )
            vertices[i].grad(j) = dx.block(j*3, i, 3, 1);
    computeFaceNormals();  
}

void Shape::initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, int der_index) {
    assert(dx.rows() == 3 && dx.cols() == num_vertices &&
           der_index >= 0 && der_index < nder);
    for ( int i = 0; i < num_vertices; i++ )
        vertices[i].grad(der_index) = dx.col(i);
    computeFaceNormals();   
}

void Shape::initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, const Eigen::Matrix<Float, -1, -1> &dn) {
    assert(dx.rows() == 3*nder && dx.cols() == num_vertices &&
           dn.rows() == 3*nder && dn.cols() == num_vertices && normals.size() == vertices.size());
    initVelocities(dx);
    for ( int i = 0; i < num_vertices; i++ )
        for ( int j = 0; j < nder; ++j )
            normals[i].grad(j) = dn.block(j*3, i, 3, 1);
}

void Shape::initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, const Eigen::Matrix<Float, -1, -1> &dn, int der_index) {
    assert(dn.rows() == 3 && dn.cols() == num_vertices && normals.size() == vertices.size() &&
           der_index >= 0 && der_index < nder);
    initVelocities(dx, der_index);
    for ( int i = 0; i < num_vertices; i++ )
        normals[i].grad(der_index) = dn.col(i);
}

void Shape::advance(Float stepSize, int derId) {
    assert(derId >= 0 && derId < nder);
    for ( int i = 0; i < num_vertices; ++i ) {
        vertices[i].val = vertices[i].advance(stepSize, derId);
        if ( !normals.empty() )
            normals[i].val = normals[i].advance(stepSize, derId);
    }
    computeFaceNormals();
    constructEdges();
}

void Shape::computeFaceNormals() {
    faceNormals.resize(num_triangles);
    for ( int i = 0; i < num_triangles; ++i ) {
        const auto& ind = getIndices(i);
        const auto& v0 = getVertexAD(ind(0));
        const auto& v1 = getVertexAD(ind(1));
        const auto& v2 = getVertexAD(ind(2));
        auto& cur = faceNormals[i];
        cur = (v1 - v0).cross(v2 - v0);
        Float tmp = cur.val.norm();
        if ( tmp < Epsilon ) {
            std::cerr << std::scientific << std::setprecision(2)
                      << "[Warning] Vanishing normal for face #" << i << " (norm = " << tmp << ")" << std::endl;
        }
        else
            cur.normalize();
    }
}

FloatAD unitAngle(const Vector3AD &u, const Vector3AD &v) {
    if (u.val.dot(v.val) < 0.0)
        return M_PI - (0.5f * (v+u).norm()).asin();
    else
        return 2.0 * (0.5f * (v-u).norm()).asin();
}

Float Shape::getArea(int index) const {
    auto& ind = getIndices(index);
    auto& v0 = getVertex(ind(0));
    auto& v1 = getVertex(ind(1));
    auto& v2 = getVertex(ind(2));
    return 0.5f * (v1 - v0).cross(v2 - v0).norm();
}

FloatAD Shape::getAreaAD(int index) const {
    auto& ind = getIndices(index);
    auto& v0 = getVertexAD(ind(0));
    auto& v1 = getVertexAD(ind(1));
    auto& v2 = getVertexAD(ind(2));
    return 0.5f * (v1 - v0).cross(v2 - v0).norm();
}

void Shape::samplePosition(int index, const Vector2 &rnd2, Vector &pos, Vector &norm) const {
    auto& ind = getIndices(index);
    auto& v0 = getVertex(ind(0));
    auto& v1 = getVertex(ind(1));
    auto& v2 = getVertex(ind(2));
    auto a = std::sqrt(rnd2.x());
    auto b1 = 1.f - a;
    auto b2 = a * rnd2.y();
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto n = e1.cross(e2);
    norm = n.normalized();
    pos = v0 + e1 * b1 + e2 * b2;
}

void Shape::rayIntersect(int tri_index, const Ray &ray, Intersection &its) const {
    auto& ind = getIndices(tri_index);
    auto& v0 = getVertex(ind(0));
    auto& v1 = getVertex(ind(1));
    auto& v2 = getVertex(ind(2));
    Vector2 uvs0, uvs1, uvs2;
    if (uvs.size() != 0) {
        uvs0 = getUV(ind(0));
        uvs1 = getUV(ind(1));
        uvs2 = getUV(ind(2));
    } else {
        uvs0 = Vector2{0, 0};
        uvs1 = Vector2{1, 0};
        uvs2 = Vector2{1, 1};
    }
    auto uvt = rayIntersectTriangle(v0, v1, v2, ray);
    auto u = uvt(0);
    auto v = uvt(1);
    auto w = 1.f - (u + v);
    auto t = uvt(2);
    auto uv = w * uvs0 + u * uvs1 + v * uvs2;
    auto hit_pos = ray.org + ray.dir * t;
    auto geom_normal = faceNormals[tri_index].val;
    auto shading_normal = geom_normal;
    if (normals.size() != 0) {
        auto n0 = getShadingNormal(ind(0));
        auto n1 = getShadingNormal(ind(1));
        auto n2 = getShadingNormal(ind(2));
        auto nn = w * n0 + u * n1 + v * n2;
        // Shading normal computation
        shading_normal = nn.normalized();
        // Flip geometric normal to the same side of shading normal
        if (geom_normal.dot(shading_normal) < 0.f) {
            geom_normal = -geom_normal;
        }
    }
    its.geoFrame = Frame(geom_normal);
    its.shFrame = Frame(shading_normal);
    its.p = hit_pos;
    its.t = t;
    its.uv = uv;
    its.wi = its.toLocal(-ray.dir);
}

void Shape::rayIntersectAD(int tri_index, const RayAD &ray, IntersectionAD &its) const {
    auto& ind = getIndices(tri_index);
    auto& v0 = getVertexAD(ind(0));
    auto& v1 = getVertexAD(ind(1));
    auto& v2 = getVertexAD(ind(2));
    Vector2 uvs0, uvs1, uvs2;
    if (uvs.size() != 0) {
        uvs0 = getUV(ind(0));
        uvs1 = getUV(ind(1));
        uvs2 = getUV(ind(2));
    } else {
        uvs0 = Vector2{0, 0};
        uvs1 = Vector2{1, 0};
        uvs2 = Vector2{1, 1};
    }
    auto uvt = rayIntersectTriangleAD(v0, v1, v2, ray);
    auto u = uvt(0);
    auto v = uvt(1);
    auto w = 1.f - (u + v);
    auto t = uvt(2);
    auto uv = w*Vector2AD(uvs0) + u*Vector2AD(uvs1) + v*Vector2AD(uvs2);
    auto hit_pos = ray.org + ray.dir * t;
    auto geom_normal = faceNormals[tri_index];
    auto shading_normal = geom_normal;
    if (normals.size() != 0) {
        auto n0 = getShadingNormalAD(ind(0));
        auto n1 = getShadingNormalAD(ind(1));
        auto n2 = getShadingNormalAD(ind(2));
        auto nn = w * n0 + u * n1 + v * n2;
        // Shading normal computation
        shading_normal = nn.normalized();
        // Flip geometric normal to the same side of shading normal
        if (geom_normal.dot(shading_normal) < 0.f) {
            geom_normal = -geom_normal;
        }
    }
    its.geoFrame = FrameAD(geom_normal);
    its.shFrame = FrameAD(shading_normal);
    its.p = hit_pos;
    its.t = t;
    its.uv = uv;
    its.wi = its.toLocal(-ray.dir);
}

void Shape::constructEdges() {
    std::map<std::pair<int,int>, std::vector<int>> edge_map;
    for (int itri = 0; itri < num_triangles; itri++) {
        auto ind = getIndices(itri);
        for (int iedge = 0; iedge < 3; iedge++) {
            int k1 = iedge, k2 = (iedge+1)%3;
            auto key = (ind[k1] < ind[k2]) ? std::make_pair(ind[k1], ind[k2]) : std::make_pair(ind[k2], ind[k1]);
            if (edge_map.find(key) == edge_map.end())
                edge_map[key] = std::vector<int>();
            edge_map[key].push_back(itri);
        }
    }

    edges.clear();
    for (auto const& it: edge_map) {
        Float length = (getVertex(it.first.first) - getVertex(it.first.second)).norm();

        // check if good mesh
        if (it.second.size() > 2) {
            std::cerr << "Every edge can be shared by at most 2 faces!" << std::endl;
            assert(false);
        }
        else if (it.second.size() == 2) {
            if (it.second[0] == it.second[1]) {
                std::cerr << "Duplicated faces!" << std::endl;
                assert(false);
            }
            else {
                Float val = faceNormals[it.second[0]].val.dot(faceNormals[it.second[1]].val);
                if ( val < -1.0f + Epsilon ) {
                    std::cerr << "Inconsistent normal orientation!" << std::endl;
                    assert(false);
                }
                else if ( val < 1.0f - Epsilon )
                    edges.push_back(Edge(it.first.first, it.first.second, it.second[0], it.second[1], length));
            }
        }
        else {
            assert(it.second.size() == 1);
            edges.push_back(Edge(it.first.first, it.first.second, it.second[0], -1, length));
        }
    }

    edge_distrb.clear();
    for ( const Edge &edge : edges ) edge_distrb.append(edge.length);
    edge_distrb.normalize();
}

int Shape::isSihoulette(const Edge& edge, const Vector& p) const {
    if (edge.f0 == -1 || edge.f1 == -1) {
        // Only adjacent to one face
        return 2;
    }
    const Vector &v0 = getVertex(edge.v0), &v1 = getVertex(edge.v1);

    bool frontfacing0 = false;
    const Vector3i &ind0 = getIndices(edge.f0);
    for (int i = 0; i < 3; i++) {
        if (ind0[i] != edge.v0 && ind0[i] != edge.v1) {
            const Vector& v = getVertex(ind0[i]);
            Vector n0 = (v0 - v).cross(v1 - v).normalized();
            frontfacing0 = n0.dot(p - v) > 0.0f;
            break;
        }
    }

    bool frontfacing1 = false;
    const Vector3i &ind1 = getIndices(edge.f1);
    for (int i = 0; i < 3; i++) {
        if (ind1[i] != edge.v0 && ind1[i] != edge.v1) {
            const Vector& v = getVertex(ind1[i]);
            Vector n1 = (v1 - v).cross(v0 - v).normalized();
            frontfacing1 = n1.dot(p - v) > 0.0f;
            break;
        }
    }
    if ((frontfacing0 && !frontfacing1) || (!frontfacing0 && frontfacing1))
        return 2;

    // If we are not using Phong normal, every edge is silhouette
    return hasNormals() ? 0 : 1;
}

const Edge& Shape::sampleEdge(Float& rnd, Float& pdf) const {
    Float pdf1;
    int idx_edge = edge_distrb.sampleReuse(rnd, pdf1);
    pdf *= pdf1;
    return edges[idx_edge];
}

// void Shape::computeVertexNormals() {
//     normals.resize(num_vertices);
//     for ( int i = 0; i < num_triangles; ++i ) {
//         const auto& ind = getIndices(i);
//         Vector3AD fn;
//         for (int j = 0; j < 3; j++) {
//             const auto& v0 = getVertexAD(ind(j));
//             const auto& v1 = getVertexAD(ind((j+1)%3));
//             const auto& v2 = getVertexAD(ind((j+2)%3));
//             Vector3AD sideA = v1 - v0;
//             Vector3AD sideB = v2 - v0;
//             if (j == 0) {
//                 fn = sideA.cross(sideB);
//                 if (fn.val.norm() == 0.0)
//                     break;
//                 fn.normalize();
//             }
//             auto angleAD = unitAngle(sideA.normalized(), sideB.normalized());
//             normals[ind(j)] += fn * angleAD;
//         }
//     }

//     for (int i = 0; i < num_vertices; i++) {
//         auto &n = normals[i];
//         Float length = n.val.norm();
//         if (length == 0.0) {
//             n = Vector3AD(1.0, 0.0, 0.0);
//         } else {
//             n.normalize();
//         }
//     }
// }