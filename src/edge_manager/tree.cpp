#include "tree.h"
#include "scene.h"
#include "camera.h"
#include <numeric>

// #define VERBOSE

// Insert two zero after every bit given a 21-bit integer
// https://github.com/leonardo-domingues/atrbvh/blob/master/BVHRT-Core/src/Commons.cuh#L599	
uint64_t expandBits3D(uint64_t x) {
    uint64_t expanded = x;
    expanded &= 0x1fffff;
    expanded = (expanded | expanded << 32) & 0x1f00000000ffff;
    expanded = (expanded | expanded << 16) & 0x1f0000ff0000ff;
    expanded = (expanded | expanded << 8) & 0x100f00f00f00f00f;
    expanded = (expanded | expanded << 4) & 0x10c30c30c30c30c3;
    expanded = (expanded | expanded << 2) & 0x1249249249249249;
    return expanded;
}

// For 6D Morton code, insert 5 zeros before each bit of a 10-bit integer
// I'm doing this in a very slow way by manipulating each bit.
// This is not the bottleneck anyway and I want readability.
uint64_t expandBits6D(uint64_t x) {
    constexpr uint64_t mask = 0x1u;
    // We start from LSB (bit 63)
    auto result = (x & (mask << 0u));
    result |= ((x & (mask << 1u)) << 5u);
    result |= ((x & (mask << 2u)) << 10u);
    result |= ((x & (mask << 3u)) << 15u);
    result |= ((x & (mask << 4u)) << 20u);
    result |= ((x & (mask << 5u)) << 25u);
    result |= ((x & (mask << 6u)) << 30u);
    result |= ((x & (mask << 7u)) << 35u);
    result |= ((x & (mask << 8u)) << 40u);
    result |= ((x & (mask << 9u)) << 45u);
    return result;
}

// Convert a 3D vector p to morton code, 0 <= p[k] <= 1 where k=0,1,2
uint64_t computeMorton3D(const Vector& p) {
        auto scale = (1 << 21) - 1;
        uint64_t x = p.x() * scale;
        uint64_t y = p.y() * scale;
        uint64_t z = p.z() * scale;
        return (expandBits3D(x) << 2u) | (expandBits3D(y) << 1u) | (expandBits3D(z) << 0u);		
}

// Convert a 6D vector (p,d) to morton code
uint64_t computeMorton6D(const Vector& p, const Vector& d) {
	uint64_t p_x = p.x() * 1023, p_y = p.y() * 1023, p_z = p.z() * 1023;
	uint64_t d_x = d.x() * 1023, d_y = d.y() * 1023, d_z = d.z() * 1023;
    return (expandBits6D(p_x) << 5u) |
           (expandBits6D(p_y) << 4u) |
           (expandBits6D(p_z) << 3u) |
           (expandBits6D(d_x) << 2u) |
           (expandBits6D(d_y) << 1u) |
           (expandBits6D(d_z) << 0u);	
}

RadixTreeBuilder::RadixTreeBuilder(const std::vector<int>& ids, const std::vector<uint64_t>& codes, const std::vector<int>& permutation) {
	num_primitives = ids.size();
	morton_codes.resize(num_primitives);
	edge_ids.resize(num_primitives);
    for (int i = 0; i < num_primitives; i++) {
    	morton_codes[i] = codes[permutation[i]];
    	edge_ids[i] = ids[permutation[i]];
        // printf("[code %d] %ld\n", i, morton_codes[i]);
    }

}

// https://github.com/henrikdahlberg/GPUPathTracer/blob/master/Source/Core/BVHConstruction.cu#L62
int RadixTreeBuilder::longestCommonPrefix(int idx0, int idx1) {
    if (idx0 < 0 || idx0 >= num_primitives ||
    	idx1 < 0 || idx1 >= num_primitives) {
        return -1;
    }
    uint64_t mc0 = morton_codes[idx0];
    uint64_t mc1 = morton_codes[idx1];
    if (mc0 == mc1) {
        // Break even when the Morton codes are the same
        uint64_t id0 = (uint64_t)edge_ids[idx0];
        uint64_t id1 = (uint64_t)edge_ids[idx1];
        return __builtin_clzll(mc0 ^ mc1) + __builtin_clzll(id0 ^ id1);
    }
    else {
        return __builtin_clzll(mc0 ^ mc1);
    }
}

// Mostly adapted from https://github.com/henrikdahlberg/GPUPathTracer/blob/master/Source/Core/BVHConstruction.cu#L161
// Also see Figure 4 in https://devblogs.nvidia.com/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
template<typename BVHNodeType>
void RadixTreeBuilder::constructTree(std::vector<BVHNodeType>& nodes, std::vector<BVHNodeType>& leaves) {
	for (int idx = 0; idx < num_primitives-1; idx++) {
        auto d = longestCommonPrefix(idx, idx + 1) -
                 longestCommonPrefix(idx, idx - 1) >= 0 ? 1 : -1;
        auto delta_min = longestCommonPrefix(idx, idx - d);
        auto lmax = 2;
        while (longestCommonPrefix(idx, idx + lmax * d) > delta_min) {
            lmax *= 2;
        }
        // Find the other end using binary search
        auto l = 0;
        auto divider = 2;
        for (int t = lmax / divider; t >= 1;) {
            if (longestCommonPrefix(idx, idx + (l + t) * d) > delta_min) {
                l += t;
            }
            if (t == 1) {
                break;
            }
            divider *= 2;
            t = lmax / divider;
        }
        auto j = idx + l * d;
        // Find the split position using binary search
        auto delta_node = longestCommonPrefix(idx, j);
        auto s = 0;
        divider = 2;
        for (int t = (l + (divider - 1)) / divider; t >= 1;) {
            if (longestCommonPrefix(idx, idx + (s + t) * d) > delta_node) {
                s += t;
            }
            if (t == 1) {
                break;
            }
            divider *= 2;
            t = (l + (divider - 1)) / divider;
        }
        auto gamma = idx + s * d + std::min(d, 0);
        assert(gamma >= 0 && gamma + 1 < num_primitives);
        auto &node = nodes[idx];
        if (std::min(idx, j) == gamma) {
            node.children[0] = &leaves[gamma];
            leaves[gamma].parent = &node;
        } else {
            node.children[0] = &nodes[gamma];
            nodes[gamma].parent = &node;
        }
        if (std::max(idx, j) == gamma + 1) {
            node.children[1] = &leaves[gamma + 1];
            leaves[gamma + 1].parent = &node;
        } else {
            node.children[1] = &nodes[gamma + 1];
            nodes[gamma + 1].parent = &node;
        }
	}
}

Float computeExteriorDihedralAngle(const Shape* ptr_shape, const Edge& edge) {
   Float exterior_dihedral = M_PI;
   if (edge.f1 != -1) {
        const Vector& n0 = ptr_shape->getGeoNormal(edge.f0);
        const Vector& n1 = ptr_shape->getGeoNormal(edge.f1);
        exterior_dihedral = std::acos(clamp(n0.dot(n1), -1.0, 1.0));
   }
   return exterior_dihedral;
}

template<typename T> inline T convertAABB(const AABB6 &b) { assert(false); }
template<> inline AABB convertAABB(const AABB6 &b) { return AABB(b.p_min, b.p_max); }
template<> inline AABB6 convertAABB(const AABB6 &b) { return b; }

template<typename BVHNodeType>
void TreeEdgeManager::computeBVHInfo(const std::vector<AABB6>& bounds, const std::vector<const Shape*>& shape_list, const std::vector<int>& edges_id,
                                     std::vector<BVHNodeType>& nodes,  std::vector<BVHNodeType>& leaves)
{
    int num_edges = edges_id.size();
    size_t num_nodes = nodes.size();
    std::vector<int> node_counters(num_nodes, 0);
    for (int idx = 0; idx < num_edges; idx++) {
        int edge_id = edges_id[idx];
        const Edge& edge = edge_list[edge_id].edge;
        const Shape* ptr_shape = shape_list[edge_list[edge_id].shape_id];
        const Vector& v0 = ptr_shape->getVertex(edge.v0);
        const Vector& v1 = ptr_shape->getVertex(edge.v1);

        BVHNodeType* leaf = &leaves[idx];
        leaf->bounds = convertAABB<decltype(BVHNodeType::bounds)>(bounds[edge_id]);
        leaf->weighted_total_length = (v0-v1).norm() * computeExteriorDihedralAngle(ptr_shape, edge);
        leaf->edge_id = edges_id[idx];

        if (num_edges == 1) {
            nodes[0] = leaves[0];
            return;
        }

        // Trace from leaf to root and merge bounding boxes & length
        BVHNodeType* current = leaf->parent;
        size_t node_idx = current - nodes.data();
        if (current != nullptr) {
            while(true) {
                assert(node_idx >= 0 && node_idx < num_nodes);
                node_counters[node_idx]++;
                if (node_counters[node_idx] == 1) {
                    // Terminate the first thread entering this node to avoid duplicate computation
                    // It is important to terminate the first not the second so we ensure all children
                    // are processed
                    break;
                }
                auto bbox = current->children[0]->bounds;
                auto weighted_length = current->children[0]->weighted_total_length;
                for (int i = 1; i < 2; i++) {
                    bbox.expandBy(current->children[i]->bounds);
                    weighted_length += current->children[i]->weighted_total_length;
                }
                current->bounds = bbox;
                current->weighted_total_length = weighted_length;
                if (current->parent == nullptr) {
                    break;
                }
                current = current->parent;
                node_idx = current - nodes.data();
            }
        }
    }
}

template<typename BVHNodeType>
void calculateOptimalTreelet(int n, BVHNodeType **leaves, uint8_t *p_opt) {
    static constexpr Float Ci = 1.0;
    // Algorithm 2 in Karras et al.
    auto num_subsets = (0x1 << n) - 1;
    assert(num_subsets < 128);
    // TODO: move the following two arrays into shared memory
    Float a[128];
    Float c_opt[128];
    // Total cost of each subset
    for (uint32_t s = 1; s <= (uint32_t)num_subsets; s++) {
        // Compute total area
        decltype(BVHNodeType::bounds) bounds = leaves[0]->bounds;
        for (int i = 1; i < n; i++) {
            if (((s >> i) & 1) == 1) {
                bounds.expandBy(leaves[i]->bounds);
            }
        }
        a[s] = bounds.surfaceArea();
    }

    // Costs of leaves
    for (uint32_t i = 0; i < (uint32_t)n; i++) {
        c_opt[(0x1 << i)] = leaves[i]->cost;       
    }
    // Optimize every subsets of leaves
    for (uint32_t k = 2; k <= (uint32_t)n; k++) {
        for (uint32_t s = 1; s <= (uint32_t)num_subsets; s++) {
            // bool print_more = (k == 3) && (s == 50);
            if (__builtin_popcount(s) == (int)k) {
                // Try each way of partitioning the leaves
                auto c_s = std::numeric_limits<Float>::infinity();
                auto p_s = uint32_t(0);
                auto d = (s - 1u) & s;
                auto p = (-d) & s;
                do {
                    auto c = c_opt[p] + c_opt[s ^ p];                   
                    if (c < c_s + Epsilon) {
                        c_s = c;
                        p_s = p;   
                    }
                    p = (p - d) & s;
                } while (p != 0);
                // SAH
                c_opt[s] = Ci * a[s] + c_s;
                p_opt[s] = p_s;
            }
        }
    } 
}

template<typename BVHNodeType>
void propagateCost(BVHNodeType *root, BVHNodeType **leaves, int num_leaves) {
    static constexpr Float Ci = 1.0;
    for (int i = 0; i < num_leaves; i++) {
        auto current = leaves[i];
        while (current != root) {
            if (current->cost < 0) {
                if (current->children[0]->cost >= 0 && current->children[1]->cost >= 0) {
                    current->bounds = current->children[0]->bounds;
                    current->bounds.expandBy(current->children[1]->bounds);
                    current->weighted_total_length =
                        current->children[0]->weighted_total_length +
                        current->children[1]->weighted_total_length;
                    current->cost = Ci * current->bounds.surfaceArea() +
                        current->children[0]->cost + current->children[1]->cost;
                } else {
                    break;
                }
            }
            current = current->parent;
        }
    }
    root->bounds = root->children[0]->bounds;
    root->bounds.expandBy(root->children[1]->bounds);
    root->weighted_total_length = root->children[0]->weighted_total_length +
                                  root->children[1]->weighted_total_length;
    root->cost = Ci * root->bounds.surfaceArea() + root->children[0]->cost + root->children[1]->cost;
}

template<typename BVHNodeType>
void restructTree(BVHNodeType *parent, BVHNodeType **leaves, BVHNodeType **nodes,
                  uint8_t partition,
                  uint8_t *optimal,
                  int &index,
                  int num_leaves,
                  uint8_t child_index)
{
    struct PartitionEntry {
        uint8_t partition;
        uint8_t child_index;
        BVHNodeType *parent;
    };

    PartitionEntry stack[8];
    auto stack_ptr = &stack[0];
    *stack_ptr++ = PartitionEntry{partition, child_index, parent};

    while (stack_ptr != &stack[0]) {
        assert(stack_ptr >= stack && stack_ptr < stack + 8);
        auto &entry = *--stack_ptr;
        auto partition = entry.partition;
        auto child_id = entry.child_index;
        auto parent = entry.parent;
        if (__builtin_popcount(partition) == 1) {
            // Leaf
            auto leaf_index = __builtin_ffs(partition) - 1;
            auto leaf = leaves[leaf_index];
            parent->children[child_id] = leaf;
            leaf->parent = parent;
        } else {
            // Internal
            assert(index < 5);
            auto node = nodes[index++];
            node->cost = -1;
            parent->children[child_id] = node;
            node->parent = parent;
            auto left_partition = optimal[partition];
            auto right_partition = uint8_t((~left_partition) & partition);
            *stack_ptr++ = PartitionEntry{left_partition, 0, node};
            *stack_ptr++ = PartitionEntry{right_partition, 1, node};
        }
    }
    propagateCost(parent, leaves, num_leaves);
}

template<typename BVHNodeType>
void optimizeTreelet(BVHNodeType *root) {
        static constexpr Float Ci = 1.0;
        if (root->edge_id != -1) {
            return;
        }
        // Form a treelet with max number of leaves being 7
        BVHNodeType *leaves[7];
        int counter = 0;
        leaves[counter++] = root->children[0];
        leaves[counter++] = root->children[1];
        // Also remember the internal nodes
        // Max 7 (leaves) - 1 (root doesn't count) - 1
        BVHNodeType *nodes[5];
        int nodes_counter = 0;
        Float max_area = 0.0;
        int max_idx = 0;
        while (counter < 7 && max_idx != -1) {
            max_idx = -1;
            max_area = -1.0;
            // Find the node with largest area and expand it
            for (int i = 0; i < counter; i++) {
                if (leaves[i]->edge_id == -1) {
                    Float area = leaves[i]->bounds.surfaceArea();
                    if (area > max_area) {
                        max_area = area;
                        max_idx = i;
                    }
                }
            }

            if (max_idx != -1) {
                BVHNodeType *tmp = leaves[max_idx];
                assert(nodes_counter < 5);
                nodes[nodes_counter++] = tmp;
                leaves[max_idx] = leaves[counter - 1];
                leaves[counter - 1] = tmp->children[0];
                leaves[counter] = tmp->children[1];
                counter++;
            }    
        }

        unsigned char optimal[128];
        calculateOptimalTreelet(counter, leaves, optimal);
        // Use complement on right tree, and use original on left tree
        auto mask = (unsigned char)((1u << counter) - 1);
        auto index = 0;
        auto left_index = mask;
        auto left = optimal[left_index];
        restructTree(root, leaves, nodes, left, optimal, index, counter, 0);
        auto right = (~left) & mask;
        restructTree(root, leaves, nodes, right, optimal, index, counter, 1);

        // Compute bounds & cost
        root->bounds = root->children[0]->bounds;
        root->bounds.expandBy(root->children[1]->bounds);
        root->weighted_total_length =
            root->children[0]->weighted_total_length +
            root->children[1]->weighted_total_length;
        root->cost = Ci * root->bounds.surfaceArea() +
            root->children[0]->cost + root->children[1]->cost;           
}

// Adapted from https://github.com/andrewwuan/smallpt-parallel-bvh-gpu/blob/master/gpu.cu
template<typename BVHNodeType>
void optimizeTree(std::vector<BVHNodeType>& nodes,  std::vector<BVHNodeType>& leaves)
{
    static constexpr Float Ci = 1.0;
    //static constexpr Float Ct = 1.0;
    int num_leaves = leaves.size();
    int num_nodes = nodes.size();
    std::vector<int> node_counters(num_nodes, 0);
    for (int idx = 0; idx < num_leaves; idx++) {
        BVHNodeType* leaf = &leaves[idx];
        leaf->cost = Ci * leaf->bounds.surfaceArea();
        BVHNodeType* current = leaf->parent;
        size_t node_idx = current - nodes.data();
        if ( current != nullptr) {
            while(true) {
                node_counters[node_idx]++;
                if (node_counters[node_idx] == 1) {
                    // Terminate the first thread entering this node to avoid duplicate computation
                    // It is important to terminate the first not the second so we ensure all children
                    // are processed
                    break;
                }
                optimizeTreelet(current);
                if (current == &nodes[0]) {
                    break;
                }
                current = current->parent;
                node_idx = current - nodes.data();
            }
        }
    }
}

TreeEdgeManager::TreeEdgeManager(const Scene& scene, const Eigen::Array<Float, -1, 1> &samplingWeights): EdgeManager(scene)
{
	const std::vector<const Shape*>& shape_list = scene.shape_list;
	for (size_t shape_id = 0; shape_id < shape_list.size(); shape_id++) {
		const std::vector<Edge>& edges = shape_list[shape_id]->edges;
		int num_edge_shape = edges.size();
		edge_list.reserve(edge_list.size() + num_edge_shape);
		for (int j = 0; j < num_edge_shape; j++)
			edge_list.push_back(EdgeWithID{(int)shape_id, edges[j]});
	}
	if (edge_list.size() == 0) return;

	int num_edges_tot = edge_list.size();
	std::vector<int> edges_id(num_edges_tot);
	for (int i = 0; i < num_edges_tot; i++)
		edges_id[i] = i;

	/*  partition all edges into 2 sets
	  - silouette viewing from cameras (including (1) boundary edges and (2) edges from shape not using shading normal)
	  - non-silouette viewing from cameras 
	*/
	const Vector& cam_org = scene.camera.cpos.val;
	std::vector<int> cs_edges_id;
	std::vector<int> ncs_edges_id;
	for (int i = 0; i < num_edges_tot; i++) {
		int shape_id = edge_list[i].shape_id;
		if(shape_list[shape_id]->isSihoulette(edge_list[i].edge, cam_org))
			cs_edges_id.push_back(i);
		else
			ncs_edges_id.push_back(i);
	}

#ifdef VERBOSE
    printf("Edges partitioned to camera silhouettes(len=%d) and non camera silhouettes(len=%d)\n", (int)cs_edges_id.size(), (int)ncs_edges_id.size());
#endif

	// compute the bounding box in 3D cartesian space and 3D hough space
	std::vector<int> node_counters(num_edges_tot, 0);
	std::vector<AABB6> edge_bounds(num_edges_tot);
	for (int i = 0; i < num_edges_tot; i++) {
		const Edge& edge = edge_list[i].edge;
		const Shape* ptr_shape = shape_list[edge_list[i].shape_id];
		const Vector& v0 = ptr_shape->getVertex(edge.v0);
		const Vector& v1 = ptr_shape->getVertex(edge.v1);
        for (int j = 0; j < 3; j++) {
            edge_bounds[i].p_min(j) = std::min(v0(j), v1(j));
            edge_bounds[i].p_max(j) = std::max(v0(j), v1(j));
        }

        // 3D Hough transform, see "Silhouette extraction in hough space", Olson and Zhang
        const Vector& n0 = ptr_shape->getGeoNormal(edge.f0);
        Vector n1 = (edge.f1 == -1) ? -n0 : ptr_shape->getGeoNormal(edge.f1);
        Vector p = 0.5f * (v0 + v1) - cam_org;
        Vector h0 = n0 * p.dot(n0);
        Vector h1 = n1 * p.dot(n1);
        for (int j = 0; j < 3; j++) {
            edge_bounds[i].d_min(j) = std::min(h0(j), h1(j));
            edge_bounds[i].d_max(j) = std::max(h0(j), h1(j));
        }
	}

	// Build 3D BVH over camera silhouettes
	int num_cs_edges = cs_edges_id.size();
	if (num_cs_edges > 0) {
		AABB cs_scene_bounds;
		for (int i = 0; i < num_cs_edges; i++) {
			cs_scene_bounds.expandBy(edge_bounds[cs_edges_id[i]].getAABB());
		}
		Vector extents = cs_scene_bounds.getExtents();  
        // Compute Morton code for LBVH
        std::vector<uint64_t> cs_morton_codes(num_cs_edges);
        for (int i = 0; i < num_cs_edges; i++) {
        	Vector p = 0.5 * (edge_bounds[cs_edges_id[i]].p_min + edge_bounds[cs_edges_id[i]].p_max);
            Vector pp;
        	for (int j = 0; j < 3; j++) {
        		pp[j] = (extents[j] > 0.0) ? (p[j]-cs_scene_bounds.min[j])/extents[j] : 0.5;
        	}
        	cs_morton_codes[i] = computeMorton3D(pp);
        }

        std::vector<int> permutation(num_cs_edges);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::sort(permutation.begin(), permutation.end(), [&](size_t i, size_t j) { return cs_morton_codes[i] < cs_morton_codes[j];});

		RadixTreeBuilder tree_builder(cs_edges_id, cs_morton_codes, permutation);      
        cs_bvh_nodes.resize(std::max(int(cs_morton_codes.size())-1, 1));
        cs_bvh_leaves.resize(cs_morton_codes.size());
        // build tree ("Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees")
        tree_builder.constructTree(cs_bvh_nodes, cs_bvh_leaves);       
        // compute BVH node information (bbox, length of edges, etc)
        computeBVHInfo(edge_bounds, shape_list, tree_builder.edge_ids, cs_bvh_nodes, cs_bvh_leaves);
        optimizeTree(cs_bvh_nodes, cs_bvh_leaves);    
	}

	// Build 6D BVH over non camera silhouettes
	int num_ncs_edges = ncs_edges_id.size();
	if (num_ncs_edges > 0) {
		AABB6 ncs_scene_bounds;
		for (int i = 0; i < num_ncs_edges; i++) {
			ncs_scene_bounds.expandBy(edge_bounds[ncs_edges_id[i]]);
		}
		Vector extents_p = ncs_scene_bounds.p_max - ncs_scene_bounds.p_min;
		Vector extents_d = ncs_scene_bounds.d_max - ncs_scene_bounds.d_min;

        // Compute Morton code for LBVH
        std::vector<uint64_t> ncs_morton_codes(num_ncs_edges);
        for (int i = 0; i < num_ncs_edges; i++) {
        	Vector p = 0.5 * (edge_bounds[ncs_edges_id[i]].p_min + edge_bounds[ncs_edges_id[i]].p_max);
        	Vector d = 0.5 * (edge_bounds[ncs_edges_id[i]].d_min + edge_bounds[ncs_edges_id[i]].d_max);
        	for (int j = 0; j < 3; j++) {
        		p[j] = (extents_p[j] > 0.0) ? (p[j]-ncs_scene_bounds.p_min[j])/extents_p[j] : 0.5;
        		d[j] = (extents_d[j] > 0.0) ? (d[j]-ncs_scene_bounds.d_min[j])/extents_d[j] : 0.5;
        	}
        	ncs_morton_codes[i] = computeMorton6D(p, d);
        }
        std::vector<int> permutation(num_ncs_edges);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::sort(permutation.begin(), permutation.end(), [&](size_t i, size_t j) { return ncs_morton_codes[i] < ncs_morton_codes[j];});

		RadixTreeBuilder tree_builder(ncs_edges_id, ncs_morton_codes, permutation);
        ncs_bvh_nodes.resize(std::max(int(ncs_morton_codes.size())-1, 1));
        ncs_bvh_leaves.resize(ncs_morton_codes.size());
        // build tree ("Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees")
        tree_builder.constructTree(ncs_bvh_nodes, ncs_bvh_leaves);        
        computeBVHInfo(edge_bounds, shape_list, tree_builder.edge_ids, ncs_bvh_nodes, ncs_bvh_leaves);
        optimizeTree(ncs_bvh_nodes, ncs_bvh_leaves);
    }

}

const Edge* TreeEdgeManager::sampleSecondaryEdge(const Scene& scene, const Vector& p,  const Frame* ptr_frame, Float& rnd, int& shape_id, Float& pdf) const {
    struct BVHStackItemH {
        BVHNodePtr node_ptr;
        Float pmf;
    };
    constexpr auto buffer_size = 128;
    BVHStackItemH buffer[buffer_size];
    auto stack_ptr = &buffer[0];
    // either sample ncs or nc edges
    Float imp_cs = 0.0;
    Float imp_ncs = 0.0;
    if (cs_bvh_leaves.size() > 0)
        imp_cs = 1.0;
    if (ncs_bvh_leaves.size() > 0)
        imp_ncs = 1.0;
    assert(imp_cs > 0 || imp_ncs > 0);
    Float prob_cs = imp_cs / (imp_cs + imp_ncs), prob_ncs = 1.0 - prob_cs;
    if (rnd < prob_cs) {
        *stack_ptr++ = BVHStackItemH{BVHNodePtr{cs_bvh_nodes.data()}, prob_cs};
        rnd = rnd / prob_cs;
    }
    else {
        *stack_ptr++ = BVHStackItemH{BVHNodePtr{ncs_bvh_nodes.data()}, prob_ncs};
        rnd = (rnd-prob_cs)/prob_ncs;
    }


    int lvl = 0;
    while (stack_ptr != &buffer[0]) {
        assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);
        // pop from stack
        const auto &stack_item = *--stack_ptr;
        if (stack_item.node_ptr.isLeaf()) {
            pdf = stack_item.pmf;
            int edge_id = stack_item.node_ptr.getEdgeId();
            shape_id = edge_list[edge_id].shape_id;
            return &(edge_list[edge_id].edge);
        } else {
            BVHNodePtr children[2];
            stack_item.node_ptr.getChildren(children);
            Float imp0 = 0.0, imp1 = 0.0;
            if (stack_item.node_ptr.contains(p)) {
                imp0 = imp1 = 1.0;
            } else {
                imp0 = children[0].evalImportance(p, ptr_frame, scene.camera.cpos.val);
                imp1 = children[1].evalImportance(p, ptr_frame, scene.camera.cpos.val);
            }
            if (imp0 > 0 || imp1 > 0) {
                auto current_pmf = stack_item.pmf;
                Float prob0 = imp0 / (imp0 + imp1);
                Float prob1 = 1.0 - prob0;
                if (rnd < prob0) {
                    rnd = rnd / prob0;
                    *stack_ptr++ = BVHStackItemH{BVHNodePtr(children[0]), current_pmf * prob0};                 
                } else {                 
                    rnd = (rnd-prob0) / prob1;
                    *stack_ptr++ = BVHStackItemH{BVHNodePtr(children[1]), current_pmf * prob1};                    
                }
            }
            lvl++;            
        }
    }
    return nullptr;
}

Float min_abs_bound(Float min, Float max) 
{
    if (min <= 0 && max >= 0) {
        return 0.0;
    }
    if (min <= 0 && max <= 0) {
        return max;
    }
    assert(min >= 0.f && max >= 0.f);
    return min;
}

Float ltc_bound(const AABB &bounds, const Vector &p, const Frame& frame) 
{
    // Due to the linear invariance, the maximum remains the same after applying M^{-1}
    // Therefore we transform the bounds using M^{-1}, find the largest possible z and smallest possible x, y in terms of magnitude.
    Vector dir = Vector(0, 0, 1);
    if (!bounds.contains(p)) {
        AABB b;
        for (int i = 0; i < 8; i++)
            b.expandBy(frame.toLocal(bounds.getCorner(i)-p));
        if (b.max.z() < 0) {
            return 0;
        }
        dir.x() = min_abs_bound(b.min.x(), b.max.x());
        dir.y() = min_abs_bound(b.min.y(), b.max.y());
        dir.z() = b.max.z();
        auto dir_len = dir.norm();
        if (dir_len <= 0) {
            dir = Vector(0, 0, 1);
        } else {
            dir = dir / dir_len;
        }
    }
    return dir.z();
}

Float ltc_bound(const AABB6 &bounds, const Vector &p, const Frame& frame) 
{
    auto p_bounds = bounds.getAABB();
    return ltc_bound(p_bounds, p, frame);
}

Float importance(const BVHNode3 &node, const Vector &p, const Frame* ptr_frame)
{
    // importance = BRDF * weighted length / dist
    // For BRDF we estimate the bound using linearly transformed cosine distribution
    auto brdf_term =  ptr_frame == nullptr ? 1.0 : ltc_bound(node.bounds, p, *ptr_frame);
    auto center = 0.5f * (node.bounds.min + node.bounds.max);
    return brdf_term * node.weighted_total_length / std::max((center - p).norm(), Float(1e-3));
}

Float importance(const BVHNode6 &node, const Vector &p, const Frame* ptr_frame, const Vector& cam_org)
{
    // importance = BRDF * weighted length / dist
    // Except if the sphere centered at 0.5 * (p - cam_org),
    // which has radius of 0.5 * distance(p, cam_org)
    // does not intersect the directional bounding box of node, 
    // the importance is zero (see Olson and Zhang 2006)
    AABB d_bounds = AABB(node.bounds.d_min, node.bounds.d_max);
    if (!d_bounds.sphereIntersect(0.5*(p - cam_org), 0.5*(p- cam_org).norm())) {
        // Not silhouette
        return 0;
    }
    auto p_bounds = AABB(node.bounds.p_min, node.bounds.p_max);
    auto brdf_term = ptr_frame == nullptr ? 1.0 : ltc_bound(p_bounds, p, *ptr_frame);
    auto center = 0.5f * (p_bounds.min + p_bounds.max);
    return brdf_term * node.weighted_total_length / std::max((center-p).norm(), Float(1e-3));
}