#pragma once
#ifndef RAY_AD_H__
#define RAY_AD_H__


#include <limits>
#include "fwd.h"
#include "ray.h"

struct RayAD {
    inline RayAD() {}
    inline RayAD(const VectorAD &org, const VectorAD& dir) : org(org), dir(dir) {}
    inline RayAD(const RayAD &ray) : org(ray.org), dir(ray.dir) {}
    inline RayAD(const Ray &ray) : org(ray.org), dir(ray.dir) {}

    inline VectorAD operator() (const FloatAD &t) const { return org + t*dir; }
    inline RayAD flipped() const { return RayAD(org, -dir); }
    inline Ray toRay() const { return Ray(org.val, dir.val); }

    VectorAD org, dir;
};

#endif //RAY_AD_H__
