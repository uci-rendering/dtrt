#pragma once
#ifndef TREE_EDGE_MANAGER_H__
#define TREE_EDGE_MANAGER_H__

#include "edge_manager.h"
#include "sampler.h"
#include "utils.h"
#include "aabb.h"
#include <algorithm>
#include <iostream>

struct Scene;

struct AABB6 {
	AABB6(const Vector& p_min, const Vector& p_max, const Vector& d_min, const Vector& d_max)
	: p_min(p_min), p_max(p_max), d_min(d_min), d_max(d_max) {}

	AABB6() {
		p_min.fill( std::numeric_limits<Float>::infinity());
    	p_max.fill(-std::numeric_limits<Float>::infinity());
    	d_min.fill( std::numeric_limits<Float>::infinity());
    	d_max.fill(-std::numeric_limits<Float>::infinity());
	}

	inline AABB getAABB() const { return AABB(p_min, p_max); }
	
	inline Float surfaceArea() const {
		auto dp = p_max - p_min;
        auto dd = d_max - d_min;
        return 2 * ((dp.x() * dp.y() + dp.x() * dp.z() + dp.y() * dp.z()) +
                    (dd.x() * dd.y() + dd.x() * dd.z() + dd.y() * dd.z()));
	}

	inline void expandBy(AABB6 aabb) { 
		for ( int i = 0; i < 3; ++i) {
        	p_min[i] = std::min(p_min[i], aabb.p_min[i]);
        	p_max[i] = std::max(p_max[i], aabb.p_max[i]);
        	d_min[i] = std::min(d_min[i], aabb.d_min[i]);
        	d_max[i] = std::max(d_max[i], aabb.d_max[i]);            	
    	}
	}

	inline bool contains(const Vector3 &p) const {
    	return p.x() >= p_min.x() && p.x() <= p_max.x() &&
           	   p.y() >= p_min.y() && p.y() <= p_max.y() &&
               p.z() >= p_min.z() && p.z() <= p_max.z();
	}


	Vector p_min, p_max;	// Cartesian AABB
	Vector d_min, d_max;	// Hough AABB
};

struct BVHNode3 {
	BVHNode3(): bounds(AABB()), weighted_total_length(0.0), parent(nullptr), edge_id(-1), cost(0.0)
	{
		children[0] = nullptr;
		children[1] = nullptr;
	}
	AABB bounds;
	Float weighted_total_length;
	BVHNode3 *parent;
	BVHNode3 *children[2];
	int edge_id;
	Float cost;

    void print(int num_tabs) const {
        for (int i = 0; i < num_tabs; i++)
            printf("\t");
        printf("[BVHNode3] bounds.min = (%.4f, %.4f, %.4f)\n", bounds.min.x(), bounds.min.y(), bounds.min.z());
        for (int i = 0; i < num_tabs; i++)
            printf("\t");        
        printf("[BVHNode3] bounds.max = (%.4f, %.4f, %.4f)\n", bounds.max.x(), bounds.max.y(), bounds.max.z());
        for (int i = 0; i < num_tabs; i++)
            printf("\t");
        printf("[BVHNode3] cost = %.4f, tot_length = %.4f\n", cost, weighted_total_length);
        if (children[0] != nullptr)
            children[0]->print(num_tabs+1);
        if (children[1] != nullptr)
            children[1]->print(num_tabs+1);
    }

};

struct BVHNode6 {
	BVHNode6(): bounds(AABB6()), weighted_total_length(0.0), parent(nullptr), edge_id(-1), cost(0.0)
	{
		children[0] = nullptr;
		children[1] = nullptr;
	}
	AABB6 bounds;
	Float weighted_total_length;
	BVHNode6 *parent;
	BVHNode6 *children[2];
	int edge_id;
	Float cost;

    void print(int num_tabs) const {
        for (int i = 0; i < num_tabs; i++)
            printf("\t");
        printf("[BVHNode6] bounds.p_min = (%.4f, %.4f, %.4f)\n", bounds.p_min.x(), bounds.p_min.y(), bounds.p_min.z());
        for (int i = 0; i < num_tabs; i++)
            printf("\t");        
        printf("[BVHNode6] bounds.p_max = (%.4f, %.4f, %.4f)\n", bounds.p_max.x(), bounds.p_max.y(), bounds.p_max.z());
        for (int i = 0; i < num_tabs; i++)
            printf("\t");
        printf("[BVHNode6] bounds.d_min = (%.4f, %.4f, %.4f)\n", bounds.d_min.x(), bounds.d_min.y(), bounds.d_min.z());
        for (int i = 0; i < num_tabs; i++)
            printf("\t");        
        printf("[BVHNode6] bounds.d_max = (%.4f, %.4f, %.4f)\n", bounds.d_max.x(), bounds.d_max.y(), bounds.d_max.z());
        for (int i = 0; i < num_tabs; i++)
            printf("\t");
        printf("[BVHNode6] cost = %.4f, tot_length = %.4f\n", cost, weighted_total_length);
        if (children[0] != nullptr)
            children[0]->print(num_tabs+1);
        if (children[1] != nullptr)
            children[1]->print(num_tabs+1);
    }
};

Float importance(const BVHNode6 &node, const Vector &p, const Frame* ptr_frame, const Vector& cam_org);
Float importance(const BVHNode3 &node, const Vector &p, const Frame* ptr_frame);

struct BVHNodePtr {
    BVHNodePtr() {}
    BVHNodePtr(const BVHNode3 *ptr3) : is_bvh_node3(true), ptr3(ptr3) {}
    BVHNodePtr(const BVHNode6 *ptr6) : is_bvh_node3(false), ptr6(ptr6) {}
    inline bool isLeaf() const { return is_bvh_node3 ? (ptr3->children[0] == nullptr) 
    												 : (ptr6->children[0] == nullptr); }
    inline int getEdgeId() const { return is_bvh_node3 ? ptr3->edge_id : ptr6->edge_id; }
	inline void getChildren(BVHNodePtr children[2]) const {
		children[0] = is_bvh_node3 ? BVHNodePtr(ptr3->children[0]) : BVHNodePtr(ptr6->children[0]);
		children[1] = is_bvh_node3 ? BVHNodePtr(ptr3->children[1]) : BVHNodePtr(ptr6->children[1]);
	}
	inline bool contains(const Vector& p) const { return is_bvh_node3 ? ptr3->bounds.contains(p)
																: ptr6->bounds.contains(p); }
	Float evalImportance(const Vector &p, const Frame* ptr_frame, const Vector& cam_org) const {
       	return is_bvh_node3 ? importance(*ptr3, p, ptr_frame)
       						: importance(*ptr6, p, ptr_frame, cam_org);
    }
    bool is_bvh_node3;
    union {
        const BVHNode3 *ptr3;
        const BVHNode6 *ptr6;
    };
};

struct RadixTreeBuilder {
	RadixTreeBuilder(const std::vector<int>& ids, const std::vector<uint64_t>& codes, const std::vector<int>& permutation);
	int longestCommonPrefix(int idx0, int idx1);
	template<typename BVHNodeType>
	void constructTree(std::vector<BVHNodeType>& nodes, std::vector<BVHNodeType>& leaves);
	std::vector<uint64_t> morton_codes;
	std::vector<int> edge_ids;
	int num_primitives;
};

struct TreeEdgeManager: EdgeManager
{
	struct EdgeWithID {
		int shape_id;
		Edge edge;
	};

	template <typename BVHNodeType>
	void computeBVHInfo(const std::vector<AABB6>& bounds, const std::vector<const Shape*>& shape_list, const std::vector<int>& edges_id,
                        std::vector<BVHNodeType>& nodes,  std::vector<BVHNodeType>& leaves);
	TreeEdgeManager(const Scene& scene, const Eigen::Array<Float, -1, 1> &samplingWeights);
	~TreeEdgeManager() {};
	const Edge* sampleSecondaryEdge(const Scene& scene, const Vector& p, const Frame* ptr_frame, Float& rnd, int& shape_id, Float& pdf) const;

    std::vector<EdgeWithID> edge_list;
	std::vector<BVHNode3> cs_bvh_nodes;
	std::vector<BVHNode3> cs_bvh_leaves;
	std::vector<BVHNode6> ncs_bvh_nodes;
	std::vector<BVHNode6> ncs_bvh_leaves;
	// Float edge_bounds_expand;

};


#endif