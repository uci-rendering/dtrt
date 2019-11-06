#include "sampler.h"

RndSampler::RndSampler(uint64_t seed, int idx) {
    state = 0U;
    inc = (((uint64_t)idx + 1) << 1u) | 1u;
    next_pcg32();
    state += (0x853c49e6748fea9bULL + seed);
    next_pcg32();
}

uint32_t RndSampler::next_pcg32() {
    uint64_t oldstate = state;
    // Advance internal state
    state = oldstate * 6364136223846793005ULL + (inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

Float RndSampler::next1D() {
#ifdef DOUBLE_PRECISION
    union {
        uint64_t u;
        double d;
    } x;
    x.u = ((uint64_t) next_pcg32() << 20) | 0x3ff0000000000000ULL;
    return x.d - 1.0;
#else
    union {
        uint32_t u;
        float f;
    } x;
    x.u = (next_pcg32() >> 9) | 0x3f800000u;
    return x.f - 1.0f;
#endif
}

Array2 RndSampler::next2D() { return Array2(next1D(), next1D());}
Array3 RndSampler::next3D() { return Array3(next1D(), next1D(), next1D());}
Array4 RndSampler::next4D() { return Array4(next1D(), next1D(), next1D(), next1D());}

void RndSampler::sampleRndSeries(Float* series, int N) {
#ifdef DOUBLE_PRECISION
    union {
        uint64_t u;
        double d;
    } x;
    for (int i = 0; i < N; i++) {
        x.u = ((uint64_t) next_pcg32() << 20) | 0x3ff0000000000000ULL;
        series[i] = x.d - 1.0;
    }
#else
    union {
        uint32_t u;
        float f;
    } x;

    for (int i = 0; i < N; i++) {
        x.u = (next_pcg32() >> 9) | 0x3f800000u;
        series[i] = x.f - 1.0f;
    }
#endif
}