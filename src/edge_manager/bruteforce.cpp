#include "bruteforce.h"
#include "scene.h"

BruteForceEdgeManager::BruteForceEdgeManager(const Scene& scene, const Eigen::Array<Float, -1, 1> &samplingWeights) : EdgeManager(scene)
{
    const std::vector<const Shape*>& shape_list = scene.shape_list;
    Eigen::Array<Float, -1, 1> shapeWeights(shape_list.size(), 1);
    for (size_t i = 0; i < shape_list.size(); i++) {
        bool ignoreShape = scene.bsdf_list[shape_list[i]->bsdf_id]->isNull() && (shape_list[i]->light_id < 0);
        shapeWeights[i] = ignoreShape ? 0.0f : shape_list[i]->getEdgeTotLength();
    }
    shapeWeights *= samplingWeights;
    assert(shapeWeights.minCoeff() > -Epsilon);
    edges_distrb.clear();
    if ( shapeWeights.sum() > Epsilon ) {
        for ( size_t i = 0; i < shape_list.size(); ++i )
            edges_distrb.append(shapeWeights[i]);
        edges_distrb.normalize();
    }
}

const Edge* BruteForceEdgeManager::sampleSecondaryEdge(const Scene& scene, const Vector& p, const Frame* ptr_frame, Float& rnd, int& shape_id, Float& pdf) const {
    assert(edges_distrb.getSum() > Epsilon);
    shape_id = edges_distrb.sampleReuse(rnd, pdf);
    return &(scene.shape_list[shape_id]->sampleEdge(rnd, pdf));
}