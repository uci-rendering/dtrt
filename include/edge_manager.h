#pragma once
#ifndef EDGE_MANAGER_H__
#define EDGE_MANAGER_H__

#include "fwd.h"
#include "pmf.h"

struct Camera;
struct Scene;
struct CropRectangle;
struct Shape;
struct Frame;

struct Edge {
    Edge(): v0(-1), v1(-1), f0(-1), f1(-1), length(0.0f) {}
    Edge(int v0, int v1, int f0, int f1, Float length): v0(v0), v1(v1), f0(f0), f1(f1), length(length) {}
    inline bool operator== (const Edge &other) const {
        return v0 == other.v0 && v1 == other.v1 &&
               f0 == other.f0 && f1 == other.f1;
    }
    inline bool isValid() const { return f0 >= 0; }

    int v0, v1;		//vertex ID
    int f0, f1;		//face ID
    Float length;	//edge length
};

struct PrimaryEdge
{
    Vector2AD v0s, v1s;         // screen space
    Vector    v0p, v1p;         // image plane
    Vector    v0c, v1c;         // camera local space
};

struct EdgeManager {
    EdgeManager(const Scene& scene);
	virtual ~EdgeManager() {};
	virtual Float getPrimaryEdgePDFsum() const { return primary_edges_distrb.getSum(); }
	virtual int getNumPrimaryEdges() const { return primary_edges.size(); }
    void initPrimaryEdgesFromShape(const Shape& shape, const Camera& cam);
    void projectEdgeToScreen(VectorAD v0, VectorAD v1, const Camera& cam);

    virtual Float samplePrimaryEdge(const Camera& cam, Float rnd, Vector2i& xyPixel, Vector2AD& p, Vector2& norm) const;
    virtual const Edge* sampleSecondaryEdge(const Scene& scene, const Vector& p, const Frame* ptr_frame, Float& rnd, int& shape_id, Float& pdf) const = 0;    
    //Primary edge sampling related
    std::vector<PrimaryEdge> primary_edges;
    DiscreteDistribution primary_edges_distrb;
};


#endif
