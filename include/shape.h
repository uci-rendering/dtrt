#pragma once
#ifndef SHAPE_H__
#define SHAPE_H__

#include "fwd.h"
#include "ptr.h"
#include "edge_manager.h"
#include "pmf.h"
#include <cmath>
#include <string>

struct Ray;
struct RayAD;
struct Intersection;
struct IntersectionAD;
struct Edge;

struct Shape {
    Shape() {}

    inline Shape(const Shape &shape)
        : vertices(shape.vertices), normals(shape.normals), indices(shape.indices), faceNormals(shape.faceNormals), uvs(shape.uvs)
        , num_vertices(shape.num_vertices), num_triangles(shape.num_triangles)
        , light_id(shape.light_id), bsdf_id(shape.bsdf_id), med_int_id(shape.med_int_id), med_ext_id(shape.med_ext_id)
        , edges(shape.edges), edge_distrb(shape.edge_distrb) {}
    Shape(ptr<float>, ptr<int>, ptr<float>, ptr<float>, int, int, int, int, int, int, ptr<float> velocities = ptr<float>(nullptr));

    void zeroVelocities();
    void initVelocities(const Eigen::Matrix<Float, -1, -1> &dx);
    void initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, int der_index);
    void initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, const Eigen::Matrix<Float, -1, -1> &dn);
    void initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, const Eigen::Matrix<Float, -1, -1> &dn, int der_index);

    void advance(Float stepSize, int derId = 0);
    void computeFaceNormals();
    // void computeVertexNormals();

    inline bool isMediumTransition() const { return med_ext_id >= 0 || med_int_id >= 0;}
    inline bool hasUVs() const { return uvs.size() != 0; }
    inline bool hasNormals() const { return normals.size() != 0; }
    inline bool isEmitter() const { return light_id >= 0; }
    inline const Vector3& getVertex(int index) const { return vertices[index].val; }
    inline const Vector3AD& getVertexAD(int index) const { return vertices[index]; }
    inline const Vector3& getShadingNormal(int index) const { return normals[index].val; }
    inline const Vector3AD& getShadingNormalAD(int index) const { return normals[index]; }
    inline const Vector3& getGeoNormal(int index) const { return faceNormals[index].val; }
    inline const Vector3AD& getGeoNormalAD(int index) const { return faceNormals[index]; }
    inline const Vector3i& getIndices(int index) const { return indices[index]; }
    inline const Vector2& getUV(int index) const { return uvs[index]; }
    inline Float getEdgeTotLength() const { return edge_distrb.getSum(); }
    inline const Edge& getEdge(int index) const { return edges[index]; }

    Float getArea(int index) const;
    FloatAD getAreaAD(int index) const;
    void samplePosition(int index, const Vector2 &rnd2, Vector &p, Vector &n) const;
    void rayIntersect(int tri_index, const Ray &ray, Intersection& its) const;
    void rayIntersectAD(int tri_index, const RayAD &ray, IntersectionAD& its) const;

    void constructEdges();
    const Edge& sampleEdge(Float& rnd, Float& pdf) const;
    int isSihoulette(const Edge& edge, const Vector& p) const;


    std::vector<Vector3AD> vertices, normals;
    std::vector<Vector3i> indices;
    std::vector<Vector3AD> faceNormals;
    std::vector<Vector2> uvs;
    int num_vertices;
    int num_triangles;

    int light_id;
    int bsdf_id;
    int med_int_id;
    int med_ext_id;

    std::vector<Edge> edges;
    DiscreteDistribution edge_distrb;
};

#endif
