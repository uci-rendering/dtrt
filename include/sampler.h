#pragma once
#ifndef RND_SAMPLER_H__
#define RND_SAMPLER_H__

#include <cstdint>
#include "fwd.h"

struct RndSampler {
    RndSampler(uint64_t seed, int idx);
    uint32_t next_pcg32();
    void sampleRndSeries(Float* rnd_series, int N);
    Array2 next2D();
    Array3 next3D();
    Array4 next4D();
    Float next1D();
    uint64_t state;
    uint64_t inc;
};

#endif