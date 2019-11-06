#pragma once
#ifndef BRUTEFORCE_EDGE_MANAGER_H__
#define BRUTEFORCE_EDGE_MANAGER_H__

#include "edge_manager.h"
#include "sampler.h"
#include "utils.h"
#include <algorithm>
#include <iostream>

struct Scene;

struct BruteForceEdgeManager: EdgeManager
{
	BruteForceEdgeManager(const Scene& scene, const Eigen::Array<Float, -1, 1> &samplingWeights);
	~BruteForceEdgeManager() {};
	const Edge* sampleSecondaryEdge(const Scene& scene, const Vector& p, const Frame* ptr_frame, Float& rnd, int& shape_id, Float& pdf) const;
	DiscreteDistribution edges_distrb;
};


#endif